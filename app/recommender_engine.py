from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


def parse_dt(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce", format="mixed")


@dataclass
class HybridRecommender:
    interactions: pd.DataFrame
    product_stats: pd.DataFrame
    user_history: pd.DataFrame
    item_factors: np.ndarray
    item_similarity: np.ndarray
    product_index: dict[int, int]
    index_product: dict[int, int]

    def recommend_for_user(
        self,
        user_id: int,
        top_k: int = 10,
        exclude_seen: bool = True,
        high_risk: bool = False,
        loyal_customer: bool = False,
    ) -> pd.DataFrame:
        history = self.user_history[self.user_history["user_id"] == user_id].copy()
        global_return_median = self.product_stats["return_rate"].median()
        global_popularity_q75 = self.product_stats["safe_popularity_score"].quantile(0.75)
        if history.empty:
            # fallback to safe popular products
            out = self.product_stats.sort_values(
                ["safe_popularity_score", "return_rate"],
                ascending=[False, True],
            ).head(top_k)
            return out.assign(
                reason="У клиента мало истории, поэтому система выбрала популярный и надежный товар с низким риском возврата."
            )

        seen_products = set(history["product_id"].tolist())
        category_affinity = history.groupby("category")["weight"].sum().to_dict()
        brand_affinity = history.groupby("brand")["weight"].sum().to_dict()
        user_avg_weight = history["weight"].mean()

        sim_scores: dict[int, float] = {}
        for _, row in history.iterrows():
            pid = int(row["product_id"])
            if pid not in self.product_index:
                continue
            idx = self.product_index[pid]
            sims = self.item_similarity[idx]
            for other_idx, sim in enumerate(sims):
                if other_idx == idx or sim <= 0:
                    continue
                other_pid = self.index_product[other_idx]
                sim_scores[other_pid] = sim_scores.get(other_pid, 0.0) + sim * float(row["weight"])

        candidates = self.product_stats.copy()
        candidates["collab_score"] = candidates["product_id"].map(sim_scores).fillna(0.0)
        candidates["category_affinity"] = candidates["category"].map(category_affinity).fillna(0.0)
        candidates["brand_affinity"] = candidates["brand"].map(brand_affinity).fillna(0.0)
        candidates["quality_penalty"] = 1.0 - candidates["return_rate"].clip(0, 0.8)
        user_mean_price = history.merge(
            self.product_stats[["product_id", "avg_price"]], on="product_id", how="left"
        )["avg_price"].mean()
        if pd.isna(user_mean_price):
            user_mean_price = candidates["avg_price"].median()
        candidates["price_fit"] = 1.0 / (1.0 + ((candidates["avg_price"] - user_mean_price).abs() / max(user_mean_price, 1.0)))
        candidates["hybrid_score"] = (
            0.45 * candidates["collab_score"]
            + 0.25 * candidates["safe_popularity_score"]
            + 0.20 * candidates["category_affinity"]
            + 0.10 * candidates["brand_affinity"]
        ) * candidates["quality_penalty"] * (0.7 + 0.3 * candidates["price_fit"])

        if high_risk:
            # For churn-risk users, emphasize safer and more familiar categories.
            candidates["hybrid_score"] = candidates["hybrid_score"] * (
                1.15 + 0.25 * (candidates["category_affinity"] > 0).astype(int)
            ) * (1.05 + 0.35 * (candidates["return_rate"] < candidates["return_rate"].median()).astype(int))
        if loyal_customer:
            candidates["hybrid_score"] = candidates["hybrid_score"] * (
                1.10 + 0.30 * (candidates["brand_affinity"] > 0).astype(int)
            ) * (1.0 + 0.15 * (candidates["avg_price"] >= user_mean_price).astype(int))

        if exclude_seen:
            candidates = candidates[~candidates["product_id"].isin(seen_products)]

        out = (
            candidates.sort_values(["hybrid_score", "return_rate"], ascending=[False, True])
            .head(top_k)
            .copy()
        )
        def explain_reason(row):
            reasons = []

            if row["category"] in category_affinity:
                reasons.append("Товар попадает в знакомую для клиента категорию, поэтому вероятность отклика выше.")
            elif row["category_affinity"] > 0:
                reasons.append("Категория близка к интересам клиента по его прошлым покупкам.")

            if row["brand"] in brand_affinity:
                reasons.append("Бренд уже встречался у клиента, поэтому рекомендация выглядит более естественно.")

            if row["return_rate"] <= global_return_median:
                reasons.append("У товара низкая доля возвратов, значит он безопаснее для рекомендации.")

            if row["price_fit"] >= 0.85:
                reasons.append("Цена очень близка к привычному для клиента уровню.")
            elif row["price_fit"] >= 0.70:
                reasons.append("Цена не выбивается из обычного чека клиента.")

            if row["safe_popularity_score"] >= global_popularity_q75:
                reasons.append("Товар хорошо показывает себя по спросу и качеству среди других покупателей.")

            if high_risk:
                reasons.append("В режиме удержания система специально поднимает более надежные и понятные клиенту товары.")

            if loyal_customer:
                reasons.append("Для лояльного клиента система делает акцент на более сильные и релевантные позиции.")

            if not reasons:
                reasons.append("Товар получил высокий итоговый балл по совокупности похожести, качества и популярности.")

            return " ".join(reasons[:3])

        out["reason"] = out.apply(explain_reason, axis=1)
        return out


def prepare_interaction_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "created_at" in df.columns:
        df["created_at"] = parse_dt(df["created_at"])
    if "returned_at" in df.columns:
        df["returned_at"] = parse_dt(df["returned_at"])
    df["brand"] = df["brand"].fillna("unknown")
    df["product_name_clean"] = df["product_name_clean"].fillna("unknown")
    df["is_returned"] = df["returned_at"].notna().astype(int)

    max_date = df["created_at"].max()
    recency_days = (max_date - df["created_at"]).dt.total_seconds() / 86400
    # fresh, repeated and non-returned interactions get more weight
    df["base_weight"] = np.exp(-recency_days.fillna(recency_days.median()) / 365.0)
    df["weight"] = df["base_weight"] * (1 - 0.7 * df["is_returned"])
    return df


def load_interaction_data(data_path: str | Path) -> pd.DataFrame:
    cols = [
        "user_id",
        "order_item_id",
        "product_id",
        "product_name_clean",
        "category",
        "brand",
        "sale_price",
        "returned_at",
        "created_at",
    ]
    df = pd.read_csv(data_path, usecols=cols)
    return prepare_interaction_df(df)


def _build_recommender_from_interactions(
    df: pd.DataFrame,
    min_item_interactions: int = 20,
    n_components: int = 40,
) -> HybridRecommender:
    if df.empty:
        raise ValueError("Interaction dataframe is empty.")

    user_history = (
        df.groupby(["user_id", "product_id", "category", "brand"], as_index=False)
        .agg(
            interactions=("order_item_id", "count"),
            weight=("weight", "sum"),
        )
    )

    product_stats = (
        df.groupby(["product_id", "product_name_clean", "category", "brand"], as_index=False)
        .agg(
            order_items=("order_item_id", "count"),
            users=("user_id", "nunique"),
            revenue=("sale_price", "sum"),
            avg_price=("sale_price", "mean"),
            return_rate=("is_returned", "mean"),
        )
        .query("order_items >= @min_item_interactions")
        .copy()
    )

    if product_stats.empty:
        fallback_threshold = max(2, min_item_interactions // 4)
        product_stats = (
            df.groupby(["product_id", "product_name_clean", "category", "brand"], as_index=False)
            .agg(
                order_items=("order_item_id", "count"),
                users=("user_id", "nunique"),
                revenue=("sale_price", "sum"),
                avg_price=("sale_price", "mean"),
                return_rate=("is_returned", "mean"),
            )
            .query("order_items >= @fallback_threshold")
            .copy()
        )

    if product_stats.empty:
        product_stats = (
            df.groupby(["product_id", "product_name_clean", "category", "brand"], as_index=False)
            .agg(
                order_items=("order_item_id", "count"),
                users=("user_id", "nunique"),
                revenue=("sale_price", "sum"),
                avg_price=("sale_price", "mean"),
                return_rate=("is_returned", "mean"),
            )
            .sort_values("order_items", ascending=False)
            .head(min(500, df["product_id"].nunique()))
            .copy()
        )

    product_stats["safe_popularity_score"] = (
        np.log1p(product_stats["order_items"])
        * np.log1p(product_stats["users"])
        * (1 - product_stats["return_rate"].clip(0, 0.8))
    )

    filtered_history = user_history[user_history["product_id"].isin(product_stats["product_id"])].copy()
    user_ids = filtered_history["user_id"].astype(int).unique()
    product_ids = product_stats["product_id"].astype(int).unique()

    user_index = {uid: idx for idx, uid in enumerate(user_ids)}
    product_index = {pid: idx for idx, pid in enumerate(product_ids)}
    index_product = {idx: pid for pid, idx in product_index.items()}

    rows = filtered_history["user_id"].map(user_index).to_numpy()
    cols = filtered_history["product_id"].map(product_index).to_numpy()
    vals = filtered_history["weight"].to_numpy(dtype=float)
    matrix = sparse.csr_matrix((vals, (rows, cols)), shape=(len(user_ids), len(product_ids)))

    if matrix.shape[0] < 2 or matrix.shape[1] < 2:
        item_factors = np.eye(matrix.shape[1], dtype=float)
        item_similarity = np.zeros((matrix.shape[1], matrix.shape[1]), dtype=float)
    else:
        svd = TruncatedSVD(n_components=min(n_components, max(2, matrix.shape[1] - 1)), random_state=42)
        item_factors = svd.fit_transform(matrix.T)
        item_similarity = cosine_similarity(item_factors)
        np.fill_diagonal(item_similarity, 0.0)

    return HybridRecommender(
        interactions=df,
        product_stats=product_stats,
        user_history=filtered_history,
        item_factors=item_factors,
        item_similarity=item_similarity,
        product_index=product_index,
        index_product=index_product,
    )


def build_hybrid_recommender(
    data_path: str | Path,
    min_item_interactions: int = 20,
    n_components: int = 40,
) -> HybridRecommender:
    df = load_interaction_data(data_path)
    return _build_recommender_from_interactions(df, min_item_interactions=min_item_interactions, n_components=n_components)


def build_hybrid_recommender_from_df(
    df: pd.DataFrame,
    min_item_interactions: int = 20,
    n_components: int = 40,
) -> HybridRecommender:
    prepared = prepare_interaction_df(df)
    return _build_recommender_from_interactions(prepared, min_item_interactions=min_item_interactions, n_components=n_components)
