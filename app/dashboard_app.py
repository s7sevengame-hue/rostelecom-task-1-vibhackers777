from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from recommender_engine import build_hybrid_recommender_from_df


st.set_page_config(
    page_title="Retail Retention Dashboard",
    page_icon="📊",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DATA_PATH = PROJECT_DIR / "data" / "data.csv"
EVENTS_PATH = PROJECT_DIR / "data" / "events.csv"
CUTOFF_DATE = pd.Timestamp("2025-06-01")
LOOKBACK_START = CUTOFF_DATE - pd.Timedelta(days=365)
RECOMMENDATIONS_COUNT = 6
DATA_COLS = [
    "order_id",
    "order_item_id",
    "user_id",
    "status",
    "gender",
    "created_at",
    "returned_at",
    "shipped_at",
    "delivered_at",
    "sale_price",
    "age",
    "state",
    "city",
    "traffic_source",
    "category",
    "department",
    "brand",
    "product_id",
    "product_name_clean",
    "warehouse_name",
    "is_loyal",
]
EVENT_COLS = [
    "id",
    "user_id",
    "session_id",
    "sequence_number",
    "created_at",
    "traffic_source",
    "browser",
    "uri",
    "event_type",
]


def parse_dt(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce", format="mixed")


def safe_mode(series: pd.Series):
    mode = series.mode(dropna=True)
    if not mode.empty:
        return mode.iat[0]
    non_null = series.dropna()
    return non_null.iat[0] if len(non_null) else np.nan

@st.cache_data(show_spinner=True)
def prepare_dataframes(data: pd.DataFrame, events: pd.DataFrame):
    data = data.copy()
    events = events.copy()

    for col in ["created_at", "returned_at", "shipped_at", "delivered_at"]:
        data[col] = parse_dt(data[col])
    events["created_at"] = parse_dt(events["created_at"])

    data = data.drop_duplicates(subset=["order_item_id"]).copy()
    events = events.drop_duplicates(subset=["id"]).copy()

    data["city"] = data["city"].fillna("unknown")
    data["brand"] = data["brand"].fillna("unknown")
    data["is_returned"] = data["returned_at"].notna().astype(int)
    data["is_cancelled"] = data["status"].eq("Cancelled").astype(int)
    data["is_delivered"] = data["delivered_at"].notna().astype(int)
    data["created_at_naive"] = data["created_at"].dt.tz_localize(None)
    data["ship_delay_days"] = (data["shipped_at"] - data["created_at"]).dt.total_seconds() / 86400
    data["delivery_days"] = (data["delivered_at"] - data["shipped_at"]).dt.total_seconds() / 86400

    events = events.dropna(subset=["user_id", "created_at"]).copy()
    events["user_id"] = events["user_id"].astype(int)
    events["created_at_naive"] = events["created_at"].dt.tz_localize(None)
    events["is_product"] = events["event_type"].eq("product").astype(int)
    events["is_cart"] = events["event_type"].eq("cart").astype(int)
    events["is_purchase"] = events["event_type"].eq("purchase").astype(int)

    return data, events


@st.cache_data(show_spinner=True)
def load_source_data():
    data = pd.read_csv(DATA_PATH, usecols=DATA_COLS)
    events = pd.read_csv(EVENTS_PATH, usecols=EVENT_COLS)
    return prepare_dataframes(data, events)


def ensure_columns(df: pd.DataFrame, required_columns: list[str], label: str):
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        st.error(f"В файле {label} не хватает колонок: {', '.join(missing)}")
        st.stop()


def load_uploaded_data(data_file, events_file):
    data = pd.read_csv(data_file)
    events = pd.read_csv(events_file)
    ensure_columns(data, DATA_COLS, "data.csv")
    ensure_columns(events, EVENT_COLS, "events.csv")
    return prepare_dataframes(data[DATA_COLS], events[EVENT_COLS])


@st.cache_data(show_spinner=True)
def build_orders(data: pd.DataFrame) -> pd.DataFrame:
    orders = (
        data.groupby("order_id", as_index=False)
        .agg(
            user_id=("user_id", "first"),
            created_at=("created_at_naive", "min"),
            status=("status", "first"),
            order_value=("sale_price", "sum"),
            items_count=("order_item_id", "nunique"),
            is_loyal=("is_loyal", "max"),
            traffic_source=("traffic_source", "first"),
            state=("state", "first"),
        )
        .dropna(subset=["user_id", "created_at"])
    )
    orders["month"] = orders["created_at"].dt.to_period("M").astype(str)
    return orders


@st.cache_data(show_spinner=True)
def build_sales_marts(data: pd.DataFrame, orders: pd.DataFrame):
    monthly_sales = (
        orders.groupby("month", as_index=False)
        .agg(
            orders=("order_id", "nunique"),
            revenue=("order_value", "sum"),
            avg_order_value=("order_value", "mean"),
        )
    )
    category_sales = (
        data.groupby(["department", "category"], as_index=False)
        .agg(
            revenue=("sale_price", "sum"),
            order_items=("order_item_id", "count"),
            return_rate=("is_returned", "mean"),
        )
        .sort_values("revenue", ascending=False)
    )
    state_sales = (
        orders.groupby("state", as_index=False)
        .agg(
            revenue=("order_value", "sum"),
            orders=("order_id", "nunique"),
        )
        .sort_values("revenue", ascending=False)
        .head(15)
    )
    return monthly_sales, category_sales, state_sales


@st.cache_data(show_spinner=True)
def build_customer_marts(data: pd.DataFrame, orders: pd.DataFrame, events: pd.DataFrame):
    customer_orders = (
        orders.groupby("user_id", as_index=False)
        .agg(
            orders_count=("order_id", "nunique"),
            total_revenue=("order_value", "sum"),
            avg_order_value=("order_value", "mean"),
            first_order_at=("created_at", "min"),
            last_order_at=("created_at", "max"),
            loyal_flag=("is_loyal", "max"),
        )
    )
    customer_orders["recency_days"] = (orders["created_at"].max() - customer_orders["last_order_at"]).dt.days
    customer_orders["lifetime_days"] = (customer_orders["last_order_at"] - customer_orders["first_order_at"]).dt.days

    r = pd.qcut(customer_orders["recency_days"].rank(method="first"), 4, labels=[4, 3, 2, 1]).astype(int)
    f = pd.qcut(customer_orders["orders_count"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
    m = pd.qcut(customer_orders["total_revenue"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
    customer_orders["rfm_score"] = r + f + m
    customer_orders["segment"] = pd.cut(
        customer_orders["rfm_score"],
        bins=[0, 5, 8, 12],
        labels=["Риск", "Средний сегмент", "Ценные"],
        include_lowest=True,
    )

    event_mart = (
        events.groupby("user_id", as_index=False)
        .agg(
            sessions=("session_id", "nunique"),
            events=("id", "count"),
            product_views=("is_product", "sum"),
            cart_events=("is_cart", "sum"),
            purchase_events=("is_purchase", "sum"),
            last_event_at=("created_at_naive", "max"),
        )
    )
    event_mart["days_since_last_event"] = (events["created_at_naive"].max() - event_mart["last_event_at"]).dt.days
    event_mart["events_per_session"] = event_mart["events"] / event_mart["sessions"]

    customer_360 = customer_orders.merge(event_mart, on="user_id", how="left")
    return customer_orders, customer_360


@st.cache_data(show_spinner=True)
def build_risk_features(data: pd.DataFrame, events: pd.DataFrame):
    orders = (
        data.groupby("order_id", as_index=False)
        .agg(
            user_id=("user_id", "first"),
            created_at=("created_at_naive", "min"),
            order_value=("sale_price", "sum"),
            items_count=("order_item_id", "nunique"),
            is_loyal=("is_loyal", "max"),
        )
        .dropna(subset=["user_id", "created_at"])
    )

    history_orders = orders[orders["created_at"] < CUTOFF_DATE].copy()
    window_orders = history_orders[(history_orders["created_at"] >= LOOKBACK_START) & (history_orders["created_at"] < CUTOFF_DATE)].copy()
    window_items = data[(data["created_at_naive"] >= LOOKBACK_START) & (data["created_at_naive"] < CUTOFF_DATE)].copy()
    window_events = events[(events["created_at_naive"] >= LOOKBACK_START) & (events["created_at_naive"] < CUTOFF_DATE)].copy()

    base_users = (
        history_orders.groupby("user_id", as_index=False)
        .agg(
            total_orders_before_cutoff=("order_value", "count"),
            loyal_flag=("is_loyal", "max"),
            first_order_at=("created_at", "min"),
            last_order_at=("created_at", "max"),
        )
    )
    base_users = base_users[(base_users["total_orders_before_cutoff"] >= 2) | (base_users["loyal_flag"] == True)].copy()
    base_users["days_since_last_order"] = (CUTOFF_DATE - base_users["last_order_at"]).dt.days
    base_users["customer_age_days"] = (CUTOFF_DATE - base_users["first_order_at"]).dt.days

    order_features = (
        window_orders.groupby("user_id", as_index=False)
        .agg(
            orders_last_365d=("order_value", "count"),
            revenue_last_365d=("order_value", "sum"),
            avg_order_value_last_365d=("order_value", "mean"),
        )
    )
    item_features = (
        window_items.groupby("user_id", as_index=False)
        .agg(
            return_rate_items=("is_returned", "mean"),
            returned_items_count=("is_returned", "sum"),
            main_department=("department", lambda s: safe_mode(s)),
            main_category=("category", lambda s: safe_mode(s)),
            second_category=("category", lambda s: s.value_counts().index[1] if s.nunique() > 1 else np.nan),
            main_brand=("brand", lambda s: safe_mode(s)),
            avg_price_last_365d=("sale_price", "mean"),
            gender=("gender", "first"),
            age=("age", "median"),
            state=("state", lambda s: safe_mode(s)),
            city=("city", lambda s: safe_mode(s)),
        )
    )
    event_features = (
        window_events.groupby("user_id", as_index=False)
        .agg(
            sessions_last_365d=("session_id", "nunique"),
            events_last_365d=("id", "count"),
            product_views_last_365d=("is_product", "sum"),
            cart_events_last_365d=("is_cart", "sum"),
            purchase_events_last_365d=("is_purchase", "sum"),
            last_event_at=("created_at_naive", "max"),
        )
    )
    event_features["days_since_last_event"] = (CUTOFF_DATE - event_features["last_event_at"]).dt.days
    event_features["events_per_session_last_365d"] = event_features["events_last_365d"] / event_features["sessions_last_365d"]

    risk_features = (
        base_users.merge(order_features, on="user_id", how="left")
        .merge(item_features, on="user_id", how="left")
        .merge(event_features, on="user_id", how="left")
    )

    for col in [
        "days_since_last_order",
        "days_since_last_event",
        "return_rate_items",
        "orders_last_365d",
        "sessions_last_365d",
        "purchase_events_last_365d",
    ]:
        risk_features[col] = risk_features[col].fillna(risk_features[col].median())

    risk_features["risk_score"] = (
        0.30 * (risk_features["days_since_last_order"] / risk_features["days_since_last_order"].max())
        + 0.20 * (risk_features["days_since_last_event"] / risk_features["days_since_last_event"].max())
        + 0.15 * risk_features["return_rate_items"].clip(0, 1)
        + 0.15 * (1 - risk_features["orders_last_365d"] / risk_features["orders_last_365d"].max())
        + 0.10 * (1 - risk_features["sessions_last_365d"] / risk_features["sessions_last_365d"].max())
        + 0.10 * (1 - risk_features["purchase_events_last_365d"] / risk_features["purchase_events_last_365d"].replace(0, np.nan).max())
    )
    risk_features["risk_score"] = risk_features["risk_score"].fillna(risk_features["risk_score"].median())
    risk_features["risk_percentile"] = risk_features["risk_score"].rank(pct=True)
    risk_features["risk_segment"] = pd.cut(
        risk_features["risk_percentile"],
        bins=[0, 0.5, 0.8, 1.0],
        labels=["Низкий риск", "Средний риск", "Высокий риск"],
        include_lowest=True,
    )
    return risk_features


@st.cache_data(show_spinner=True)
def build_category_stats(data: pd.DataFrame):
    category_stats = (
        data.groupby(["department", "category"], as_index=False)
        .agg(
            order_items=("order_item_id", "count"),
            revenue=("sale_price", "sum"),
            return_rate=("is_returned", "mean"),
        )
        .query("order_items >= 500")
        .copy()
    )
    category_stats["revenue_per_item"] = category_stats["revenue"] / category_stats["order_items"]
    category_stats["safety_score"] = (1 - category_stats["return_rate"]) * category_stats["revenue_per_item"]
    return category_stats


@st.cache_resource(show_spinner=True)
def build_recommender_resource(data: pd.DataFrame):
    interaction_df = data[
        [
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
    ].copy()
    return build_hybrid_recommender_from_df(interaction_df, min_item_interactions=20, n_components=40)


def get_recommendation_modes(recommender, client_row: pd.Series):
    user_id = int(client_row["user_id"])
    modes = [
        ("Обычные", {"high_risk": False, "loyal_customer": False}),
        ("Удерживающие", {"high_risk": True, "loyal_customer": False}),
        ("Для лояльных", {"high_risk": False, "loyal_customer": True}),
    ]
    outputs = {}
    if recommender is None:
        return outputs
    for mode_name, kwargs in modes:
        recs = recommender.recommend_for_user(user_id=user_id, top_k=RECOMMENDATIONS_COUNT, **kwargs)
        outputs[mode_name] = recs
    return outputs


def explain_mode(mode_name: str, client_row: pd.Series) -> str:
    if mode_name == "Обычные":
        return "Базовая персональная подборка: учитываются история покупок, похожие товары и популярные качественные позиции."
    if mode_name == "Удерживающие":
        return "Режим для клиентов с риском ухода: выше приоритет у знакомых и более безопасных товаров с низкой долей возвратов."
    if mode_name == "Для лояльных":
        if bool(client_row.get("loyal_flag", False)):
            return "Режим для ценных клиентов: система сильнее учитывает любимые бренды и может поднимать более сильные по чеку товары."
        return "Показываем, как выглядела бы выдача для ценного клиента: больше веса у любимых брендов и товаров ближе к сильному чеку."
    return ""


def render_recommendation_cards(recs: pd.DataFrame, mode_name: str):
    if recs is None or recs.empty:
        st.info("Для этого клиента система пока не нашла уверенные товары в этом режиме.")
        return

    for idx, (_, rec) in enumerate(recs.head(RECOMMENDATIONS_COUNT).iterrows(), start=1):
        card = st.container(border=True)
        product_name = rec["product_name_clean"] if pd.notna(rec["product_name_clean"]) else "Нет данных"
        category_name = rec["category"] if pd.notna(rec["category"]) else "Нет данных"
        brand_name = rec["brand"] if pd.notna(rec["brand"]) else "Нет данных"
        price_text = f"{rec['avg_price']:.1f}" if pd.notna(rec.get("avg_price")) else "Нет данных"
        return_rate_text = f"{rec['return_rate']:.1%}" if pd.notna(rec.get("return_rate")) else "Нет данных"
        score_text = f"{rec['hybrid_score']:.3f}" if pd.notna(rec.get("hybrid_score")) else "Нет данных"
        with card:
            top_left, top_right = st.columns([1.8, 1])
            with top_left:
                st.markdown(f"**Рекомендация {idx}.** {product_name}")
                st.caption(f"{category_name} | {brand_name}")
            with top_right:
                st.markdown(f"**Цена:** {price_text} у.е.")
                st.caption(f"Возвраты: {return_rate_text} | Балл модели: {score_text}")

            reason_text = str(rec.get("reason", "")).strip()
            if not reason_text:
                reason_text = "Товар хорошо подходит по суммарному баллу модели."
            st.caption(f"Почему в выдаче: {reason_text}")


def choose_recommendation_mode(client_row: pd.Series) -> str:
    if bool(client_row.get("loyal_flag", False)):
        return "Для лояльных"
    if str(client_row.get("risk_segment", "")) == "Высокий риск" or float(client_row.get("risk_percentile", 0)) >= 0.8:
        return "Удерживающие"
    return "Обычные"


st.sidebar.markdown("### Источник данных")
use_uploaded = st.sidebar.toggle("Загрузить свои CSV", value=False)
with st.sidebar.expander("Описание формата CSV"):
    st.write("Нужно загрузить два файла: `data.csv` и `events.csv`.")
    st.write("`data.csv` должен содержать колонки:")
    st.code(
        "order_id, order_item_id, user_id, status, gender, created_at, returned_at, "
        "shipped_at, delivered_at, sale_price, age, state, city, traffic_source, "
        "category, department, brand, product_id, product_name_clean, warehouse_name, is_loyal",
        language="text",
    )
    st.write("`events.csv` должен содержать колонки:")
    st.code(
        "id, user_id, session_id, sequence_number, created_at, traffic_source, "
        "browser, uri, event_type",
        language="text",
    )
    st.write(
        "Названия колонок должны совпадать. Лишние колонки допустимы, но обязательные поля должны присутствовать."
    )
    st.write(
        "Если каких-то обязательных колонок не хватает, приложение покажет сообщение об ошибке."
    )
uploaded_data_file = None
uploaded_events_file = None
if use_uploaded:
    uploaded_data_file = st.sidebar.file_uploader("Загрузите файл заказов `data.csv`", type=["csv"])
    uploaded_events_file = st.sidebar.file_uploader("Загрузите файл событий `events.csv`", type=["csv"])

if use_uploaded:
    if uploaded_data_file is None or uploaded_events_file is None:
        st.title("Раздел")
        st.info("Загрузите оба файла: `data.csv` и `events.csv`. После этого дашборд автоматически пересчитает метрики, риск и рекомендации.")
        st.stop()
    data, events = load_uploaded_data(uploaded_data_file, uploaded_events_file)
else:
    data, events = load_source_data()

page = st.sidebar.radio(
    "Раздел",
    ["Продажи и клиенты", "Риски и рекомендации", "Карточка клиента"],
)

st.title(page)

if page == "Продажи и клиенты":
    orders = build_orders(data)
    monthly_sales, category_sales, state_sales = build_sales_marts(data, orders)
    customer_orders, customer_360 = build_customer_marts(data, orders, events)

    col1, col2, col3 = st.columns(3)
    col1.metric("Выручка, у.е.", f"{orders['order_value'].sum():,.0f}")
    col2.metric("Число заказов", f"{orders['order_id'].nunique():,}")
    col3.metric("Средний чек, у.е.", f"{orders['order_value'].mean():.1f}")

    monthly_sales_ru = monthly_sales.rename(
        columns={
            "month": "Месяц",
            "revenue": "Выручка",
            "orders": "Число заказов",
            "avg_order_value": "Средний чек",
        }
    )
    fig_month = px.line(
        monthly_sales_ru,
        x="Месяц",
        y=["Выручка", "Число заказов"],
        markers=True,
        title="Динамика выручки и числа заказов по месяцам",
    )
    fig_month.update_layout(legend_title_text="")
    st.plotly_chart(fig_month, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        top_categories_ru = category_sales.head(15).rename(
            columns={"revenue": "Выручка", "category": "Категория", "department": "Отдел"}
        )
        fig_cat = px.bar(
            top_categories_ru,
            x="Выручка",
            y="Категория",
            color="Отдел",
            orientation="h",
            title="Топ категорий по выручке",
        )
        fig_cat.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_cat, use_container_width=True)
    with col_b:
        state_sales_ru = state_sales.rename(columns={"revenue": "Выручка", "state": "Штат"})
        fig_state = px.bar(
            state_sales_ru,
            x="Выручка",
            y="Штат",
            orientation="h",
            title="Топ штатов по выручке",
        )
        fig_state.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_state, use_container_width=True)

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    col1.metric("Клиентов", f"{customer_orders['user_id'].nunique():,}")
    col2.metric("Лояльных клиентов", f"{customer_orders['loyal_flag'].sum():,}")
    col3.metric("Средний RFM-скор", f"{customer_orders['rfm_score'].mean():.1f}")

    segment_stats = customer_orders["segment"].value_counts().rename_axis("Сегмент").reset_index(name="Число клиентов")
    fig_segment = px.pie(segment_stats, names="Сегмент", values="Число клиентов", title="Сегменты клиентов")
    st.plotly_chart(fig_segment, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        fig_orders = px.histogram(
            customer_orders.rename(columns={"orders_count": "Число заказов"}),
            x="Число заказов",
            nbins=20,
            title="Распределение числа заказов",
        )
        st.plotly_chart(fig_orders, use_container_width=True)
    with col_b:
        fig_revenue = px.histogram(
            customer_orders.rename(columns={"total_revenue": "Выручка на клиента"}),
            x="Выручка на клиента",
            nbins=30,
            title="Распределение выручки на клиента",
        )
        st.plotly_chart(fig_revenue, use_container_width=True)

elif page == "Риски и рекомендации":
    risk_features = build_risk_features(data, events)
    category_stats = build_category_stats(data)

    st.markdown("## Общая картина риска")
    col1, col2, col3 = st.columns(3)
    high_risk_share = (risk_features["risk_segment"] == "Высокий риск").mean()
    col1.metric("Клиентов в модели", f"{len(risk_features):,}")
    col2.metric("Высокий риск", f"{high_risk_share:.1%}")
    col3.metric("Средний риск-скор", f"{risk_features['risk_score'].mean():.3f}")

    risk_dist = risk_features["risk_segment"].value_counts().rename_axis("Уровень риска").reset_index(name="Число клиентов")
    fig_risk = px.bar(risk_dist, x="Уровень риска", y="Число клиентов", color="Уровень риска", title="Распределение клиентов по риску ухода")
    st.plotly_chart(fig_risk, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        top_reasons = (
            risk_features[["days_since_last_order", "days_since_last_event", "return_rate_items", "orders_last_365d", "sessions_last_365d"]]
            .median()
            .rename_axis("Фактор")
            .reset_index(name="Медианное значение")
        )
        top_reasons["Фактор"] = top_reasons["Фактор"].replace(
            {
                "days_since_last_order": "Дней с последнего заказа",
                "days_since_last_event": "Дней с последней активности",
                "return_rate_items": "Доля возвратов",
                "orders_last_365d": "Заказов за 365 дней",
                "sessions_last_365d": "Сессий за 365 дней",
            }
        )
        fig_factors = px.bar(top_reasons, x="Медианное значение", y="Фактор", orientation="h", title="Типичные факторы риска")
        fig_factors.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_factors, use_container_width=True)
    with col_b:
        risky_categories = category_stats.sort_values("return_rate", ascending=False).head(12).rename(
            columns={"return_rate": "Доля возвратов", "category": "Категория", "department": "Отдел"}
        )
        fig_risky = px.bar(
            risky_categories,
            x="Доля возвратов",
            y="Категория",
            color="Отдел",
            orientation="h",
            title="Категории с повышенным уровнем возврата",
        )
        fig_risky.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_risky, use_container_width=True)

    st.markdown("## Топ-20 клиентов с самым высоким риском")
    top20_risk = (
        risk_features[
            ["user_id", "risk_score", "risk_segment", "days_since_last_order", "days_since_last_event", "main_category"]
        ]
        .sort_values("risk_score", ascending=False)
        .head(20)
        .reset_index(drop=True)
    )
    st.dataframe(top20_risk, use_container_width=True)

elif page == "Карточка клиента":
    orders = build_orders(data)
    risk_features = build_risk_features(data, events)
    recommender = None
    with st.spinner("Собираем рекомендательный движок..."):
        try:
            recommender = build_recommender_resource(data)
        except Exception:
            recommender = None

    st.markdown("## Карточка клиента")
    available_users = risk_features["user_id"].sort_values().tolist()
    default_user = int(
        risk_features.sort_values("risk_score", ascending=False).iloc[0]["user_id"]
    ) if len(risk_features) else None
    selected_user = st.selectbox(
        "Выберите клиента",
        options=available_users,
        index=available_users.index(default_user) if default_user in available_users else 0,
    )

    client = risk_features[risk_features["user_id"] == selected_user].iloc[0]
    client_orders = orders[orders["user_id"] == selected_user].sort_values("created_at", ascending=False)

    summary = st.container(border=True)
    with summary:
        top1, top2 = st.columns(2)
        top1.metric("Риск-скор", f"{client['risk_score']:.3f}")
        top2.metric("Сегмент риска", str(client["risk_segment"]))

        main_category = client.get("main_category")
        second_category = client.get("second_category")
        info1, info2 = st.columns(2)
        with info1:
            st.metric("Лояльный клиент", "Да" if bool(client.get("loyal_flag", False)) else "Нет")
            st.write(f"**Основная категория:** {main_category if pd.notna(main_category) else 'Нет данных'}")
            st.write(f"**Доп. категория:** {second_category if pd.notna(second_category) else 'Нет данных'}")
            avg_check = (
                f"{client['avg_order_value_last_365d']:.1f}"
                if pd.notna(client.get("avg_order_value_last_365d"))
                else "Нет данных"
            )
            st.metric("Средний чек, у.е.", avg_check)
        with info2:
            st.write("**Активность клиента**")
            row1, row2 = st.columns(2)
            row1.metric(
                "Дней с последнего заказа",
                int(client["days_since_last_order"]),
            )
            row2.metric(
                "Дней с последней активности",
                int(client["days_since_last_event"]) if pd.notna(client["days_since_last_event"]) else 0,
            )
            row3, row4 = st.columns(2)
            row3.metric(
                "Заказов за 365 дней",
                int(client["orders_last_365d"]) if pd.notna(client["orders_last_365d"]) else 0,
            )
            row4.metric(
                "Сессий за 365 дней",
                int(client["sessions_last_365d"]) if pd.notna(client["sessions_last_365d"]) else 0,
            )

    st.markdown("## Рекомендации")
    if recommender is None:
        st.warning("Рекомендательный движок сейчас недоступен. Проверьте данные по товарам и покупкам.")
    else:
        mode_outputs = get_recommendation_modes(recommender, client)
        auto_mode = choose_recommendation_mode(client)
        st.write(f"Система автоматически выбрала режим: `{auto_mode}`")
        mode_name = st.radio(
            "Режим рекомендаций",
            ["Обычные", "Удерживающие", "Для лояльных"],
            index=["Обычные", "Удерживающие", "Для лояльных"].index(auto_mode),
            horizontal=True,
        )
        st.write(explain_mode(mode_name, client))
        render_recommendation_cards(mode_outputs.get(mode_name), mode_name)

    st.markdown("## Последние заказы клиента")
    st.dataframe(
        client_orders[["order_id", "created_at", "status", "order_value", "items_count"]].head(10),
        use_container_width=True,
    )
