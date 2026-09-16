import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
import json

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="NeoBank Health & Churn Dashboard",
    page_icon="🏦",
    layout="wide",
)

# 2. LOAD DATASET AND MODEL (Cached for speed)
@st.cache_data
def load_dashboard_data():
    df_model = pd.read_csv("data/df_model.csv")
    df_users = pd.read_csv("data/df_users_cleaned.csv")
    df_notifications = pd.read_csv("data/df_notifications.csv")
    df_devices = pd.read_csv("data/df_devices.csv")

    # Convert notification date to datetime format
    df_notifications["created_date"] = pd.to_datetime(
        df_notifications["created_date"]
    )

    return (
        df_model,
        df_users,
        df_notifications,
        df_devices,
    )

(
    df_model,
    df_users,
    df_notifications,
    df_devices,
) = load_dashboard_data()

def load_tx_summary():
    with open("data/tx_summary.json", "r") as f:
        return json.load(f)

@st.cache_resource
def load_model():
    return joblib.load("churn_model.pkl")

clf = load_model()

# 3. NAVIGATION
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to", ["Project Overview", "Overview & KPIs", "User Activity Analytics", "Churn Predictor"]
)

# 4. PAGE LOGIC
if page == "Project Overview":
    st.title("Introduction")
    st.subheader("Welcome to the NeoBank Health & Churn Dashboard!")

    # Create two columns (adjust the ratio as needed, e.g., [3, 2] or [1, 1])
    col1, col2 = st.columns([5, 2])

    with col1:
        st.markdown("""
        This is a world-famous neo-bank. It is one of the first to have eliminated hidden bank charges when paying with other currencies.

        The dataset contains a large volume of transactions made with credit cards, along with user and device information.

        This project aims to provide a comprehensive overview of the bank's health, user engagement, and churn prediction capabilities through an interactive dashboard.
        """)

    with col2:
        st.image(
            "assets/neobank_banner.png",
            use_container_width=True,
        )

elif page == "Overview & KPIs":
    st.title("🏦 NeoBank Health Overview")
    st.caption(
        "Executive dashboard tracking user acquisition, engagement, and operational metrics."
    )
    st.divider()

    # --- Section 1: Core User & Reach Metrics ---
    st.subheader("👥 User Base & Global Reach")

    # Metrics calculation
    total_users = df_model["user_id"].nunique() if "user_id" in df_model.columns else len(df_model)
    total_countries = df_users["country"].nunique() if "country" in df_users.columns else 0
    churn_rate = (df_model["churn"].mean()) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Users", f"{total_users:,}")
    col2.metric("Countries Served", f"{total_countries}")
    col3.metric(
        "Overall Churn Rate",
        f"{churn_rate:.1f}%",
        delta=f"+{churn_rate:.1f}%",
        delta_color="inverse",
    )

    st.divider()

    # --- Section 2: Transaction Performance ---
    st.subheader("💳 Transaction Dynamics")

    tx_summary = load_tx_summary()
    total_tx = tx_summary["total_tx_count"]

    successful_tx = (
        df_model["num_completed_transactions"].sum()
        if "num_completed_transactions" in df_model.columns
        else 0
    )
    total_volume_usd = (
        df_model["total_completed_amount_usd"].sum()
        if "total_completed_amount_usd" in df_model.columns
        else 0.0
    )
    success_rate = (successful_tx / total_tx * 100) if total_tx > 0 else 0.0

    col_tx1, col_tx2, col_tx3 = st.columns(3)
    col_tx1.metric("Total Volume (USD)", f"${total_volume_usd:,.2f}")
    col_tx2.metric(
        "Completed Transactions",
        f"{successful_tx:,}",
        help="Transactions with status COMPLETED",
    )
    col_tx3.metric(
        "Transaction Success Rate",
        f"{success_rate:.1f}%",
        f"{total_tx:,} total attempts",
    )

    st.divider()

    # ==========================================
    # SECTION 3: USER ENGAGEMENT & PREFERENCES
    # ==========================================
    st.subheader("📲 Marketing Reach & Network Density")

    # Push Notification Opt-ins
    push_users = (
        df_model["attributes_notifications_marketing_push"].sum()
        if "attributes_notifications_marketing_push" in df_model.columns
        else 0
    )

    # Email Notification Opt-ins
    email_users = (
        df_model["attributes_notifications_marketing_email"].sum()
        if "attributes_notifications_marketing_email" in df_model.columns
        else 0
    )

    # >5 Contacts Calculation
    users_gt_5_contacts = (
        (df_model["num_contacts"] > 5).sum()
        if "num_contacts" in df_model.columns
        else 0
    )

    col_eng1, col_eng2, col_eng3 = st.columns(3)

    # Helper function to create small, consistent donut charts
    def create_opt_in_pie(opt_in_count, total, title, label_opt_in="Opted In", label_opt_out="Opted Out"):
        opt_out_count = max(total - opt_in_count, 0)
        df_pie = pd.DataFrame({
            "Status": [label_opt_in, label_opt_out],
            "Users": [opt_in_count, opt_out_count]
        })
        fig = px.pie(
            df_pie,
            names="Status",
            values="Users",
            hole=0.5,
            title=title,
            color="Status",
            color_discrete_map={label_opt_in: "#1f77b4", label_opt_out: "#e0e0e0"}
        )
        fig.update_traces(
            textposition="inside",
            textinfo="percent",
            hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Percentage: %{percent}"
        )
        fig.update_layout(
            showlegend=True,
            height=220,
            margin=dict(t=40, b=10, l=10, r=10),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        return fig

    with col_eng1:
        if total_users > 0:
            fig_push = create_opt_in_pie(
                push_users, total_users, "Marketing Push Opt-Ins"
            )
            st.plotly_chart(fig_push, use_container_width=True)

    with col_eng2:
        if total_users > 0:
            fig_email = create_opt_in_pie(
                email_users, total_users, "Email Opt-Ins"
            )
            st.plotly_chart(fig_email, use_container_width=True)

    with col_eng3:
        if total_users > 0:
            fig_contacts = create_opt_in_pie(
                users_gt_5_contacts,
                total_users,
                "Connected Users (>5 Contacts)",
                label_opt_in=">5 Contacts",
                label_opt_out="≤5 Contacts"
            )
            st.plotly_chart(fig_contacts, use_container_width=True)

elif page == "User Activity Analytics":
    st.title("📊 Activity Analytics")

    # 1. Initialize session state keys for both filters
    if "selected_country" not in st.session_state:
        st.session_state.selected_country = None
    if "selected_decade" not in st.session_state:
        st.session_state.selected_decade = None

    # Compute decade categories globally across df_users
    if "birth_year" in df_users.columns:
        min_year = int(df_users["birth_year"].min())
        max_year = int(df_users["birth_year"].max())
        start_decade = (min_year // 10) * 10
        end_decade = ((max_year // 10) + 1) * 10
        bins = list(range(start_decade, end_decade + 10, 10))
        labels = [f"{b}s" for b in bins[:-1]]
        df_users["decade"] = pd.cut(
            df_users["birth_year"],
            bins=bins,
            labels=labels,
            right=False,
        )

    # 2. Add Explicit Filter Controls (Country & Age Group)
    col_filter1, col_filter2 = st.columns(2)
    with col_filter1:
        country_options = ["All"] + sorted(list(df_users["country"].dropna().unique()))
        current_country_idx = country_options.index(st.session_state.selected_country) if st.session_state.selected_country in country_options else 0
        selected_country_input = st.selectbox("Filter by Country", country_options, index=current_country_idx)
        st.session_state.selected_country = None if selected_country_input == "All" else selected_country_input

    with col_filter2:
        decade_options = ["All"] + sorted(list(df_users["decade"].dropna().unique()))
        current_decade_idx = decade_options.index(st.session_state.selected_decade) if st.session_state.selected_decade in decade_options else 0
        selected_decade_input = st.selectbox("Filter by Age Group (Decade)", decade_options, index=current_decade_idx)
        st.session_state.selected_decade = None if selected_decade_input == "All" else selected_decade_input

    # 3. Apply BOTH filters to create df_users_filtered
    df_users_filtered = df_users.copy()
    if st.session_state.selected_country:
        df_users_filtered = df_users_filtered[
            df_users_filtered["country"] == st.session_state.selected_country
        ]
    if st.session_state.selected_decade:
        df_users_filtered = df_users_filtered[
            df_users_filtered["decade"] == st.session_state.selected_decade
        ]

    # 4. Display Active Filters Bar & Reset Button
    if st.session_state.selected_country or st.session_state.selected_decade:
        col_title, col_reset = st.columns([4, 1])
        with col_title:
            active_filters = []
            if st.session_state.selected_country:
                active_filters.append(f"Country = **{st.session_state.selected_country}**")
            if st.session_state.selected_decade:
                active_filters.append(f"Age Group = **{st.session_state.selected_decade}**")
            st.info(f"Active Filters: {' | '.join(active_filters)}")
        with col_reset:
            if st.button("Reset Filters"):
                st.session_state.selected_country = None
                st.session_state.selected_decade = None
                st.rerun()

    # ==========================================
    # SECTION 1: DEMOGRAPHICS (AGE & GEOGRAPHY)
    # ==========================================
    st.subheader("Demographic Breakdown")
    col_demo1, col_demo2 = st.columns(2)

    with col_demo1:
        # Pie chart updates based on active filters
        if "decade" in df_users_filtered.columns:
            decade_counts = (
                df_users_filtered["decade"]
                .value_counts()
                .reset_index()
                .sort_values("decade")
            )
            decade_counts.columns = ["Decade", "User Count"]

            pie_title = "User Age Distribution by Decades"
            if st.session_state.selected_country:
                pie_title += f" ({st.session_state.selected_country})"

            fig_decade = px.pie(
                decade_counts,
                names="Decade",
                values="User Count",
                hole=0.4,
                title=pie_title,
                color_discrete_sequence=px.colors.sequential.RdBu,
            )
            fig_decade.update_traces(
                textposition="inside",
                textinfo="percent+label",
                hovertemplate="<b>%{label}</b><br>Users: %{value:,}<br>Percentage: %{percent}",
            )
            fig_decade.update_layout(
                showlegend=True, margin=dict(t=40, b=20, l=20, r=20)
            )

            st.plotly_chart(fig_decade, use_container_width=True)

    with col_demo2:
        # Geographic Map reflects df_users_filtered (filtered by selected age decade if active)
        if "country" in df_users_filtered.columns:
            country_counts = (
                df_users_filtered.groupby("country")["user_id"].nunique().reset_index()
            )
            country_counts.columns = ["country", "User Count"]

            iso2_to_iso3 = {
                "AF": "AFG", "AL": "ALB", "DZ": "DZA", "AD": "AND", "AO": "AGO",
                "AR": "ARG", "AM": "ARM", "AU": "AUS", "AT": "AUT", "AZ": "AZE",
                "BE": "BEL", "BG": "BGR", "BR": "BRA", "CA": "CAN", "CH": "CHE",
                "CL": "CHL", "CN": "CHN", "CO": "COL", "CY": "CYP", "CZ": "CZE",
                "DE": "DEU", "DK": "DNK", "ES": "ESP", "EE": "EST", "FI": "FIN",
                "FR": "FRA", "GB": "GBR", "GR": "GRC", "HK": "HKG", "HR": "HRV",
                "HU": "HUN", "ID": "IDN", "IE": "IRL", "IL": "ISR", "IN": "IND",
                "IS": "ISL", "IT": "ITA", "JP": "JPN", "KR": "KOR", "LT": "LTU",
                "LU": "LUX", "LV": "LVA", "MT": "MLT", "MX": "MEX", "MY": "MYS",
                "NL": "NLD", "NO": "NOR", "NZ": "NZL", "PL": "POL", "PT": "PRT",
                "RO": "ROU", "RU": "RUS", "SG": "SGP", "SK": "SVK", "SI": "SVN",
                "SE": "SWE", "TR": "TUR", "UA": "UKR", "US": "USA", "ZA": "ZAF",
            }

            country_counts["iso_alpha"] = country_counts["country"].map(iso2_to_iso3)

            map_title = "Global User Distribution by Country"
            if st.session_state.selected_decade:
                map_title += f" ({st.session_state.selected_decade})"

            fig_map = px.choropleth(
                country_counts,
                locations="iso_alpha",
                color="User Count",
                hover_name="country",
                custom_data=["country"],
                title=map_title,
                color_continuous_scale="Viridis",
            )

            fig_map.update_layout(
                geo=dict(
                    showframe=False,
                    showcoastlines=True,
                    projection_type="equirectangular",
                    visible=True,
                ),
                margin=dict(t=40, b=20, l=20, r=20),
            )

            map_event = st.plotly_chart(
                fig_map,
                use_container_width=True,
                config={"scrollZoom": False},
                on_select="rerun",
                selection_mode="points",
            )

            if (
                map_event
                and "selection" in map_event
                and map_event["selection"]["points"]
            ):
                clicked_point = map_event["selection"]["points"][0]
                if "customdata" in clicked_point:
                    selected_code = clicked_point["customdata"][0]
                    if st.session_state.selected_country != selected_code:
                        st.session_state.selected_country = selected_code
                        st.rerun()

    st.divider()
    # ==========================================
    # SECTION 2: TIME-SERIES MONITORING CHART
    # ==========================================
    st.subheader("Time-Series Monitoring Chart")

    @st.cache_data
    def load_summary_data():
        df_summary = pd.read_csv("data/df_daily_transaction_summary.csv")
        df_summary["created_date"] = pd.to_datetime(
            df_summary["created_date"], format="mixed"
        )
        return df_summary

    daily_metrics_raw = load_summary_data()
    filtered_df = daily_metrics_raw.copy()

    # Filter summary data by selected country
    if st.session_state.selected_country and "country" in filtered_df.columns:
        filtered_df = filtered_df[
            filtered_df["country"] == st.session_state.selected_country
        ]

    # Filter summary data by selected age group (decade)
    if st.session_state.selected_decade and "decade" in filtered_df.columns:
        filtered_df = filtered_df[
            filtered_df["decade"] == st.session_state.selected_decade
        ]

    if not filtered_df.empty:
        daily_metrics = (
            filtered_df.groupby("created_date")["daily_volume"]
            .sum()
            .reset_index()
        )
        title_parts = []
        if st.session_state.selected_country:
            title_parts.append(st.session_state.selected_country)
        if st.session_state.selected_decade:
            title_parts.append(st.session_state.selected_decade)

        suffix = f" — {' | '.join(title_parts)}" if title_parts else " — Global"
        chart_title = f"Daily Transaction Volume (USD){suffix}"
    else:
        daily_metrics = pd.DataFrame(columns=["created_date", "daily_volume"])
        chart_title = "Daily Transaction Volume (USD) (No Data)"

    if daily_metrics.empty:
        st.warning("No transaction records found for the selected filter combination.")
    else:
        fig = px.line(
            daily_metrics,
            x="created_date",
            y="daily_volume",
            title=chart_title,
            labels={
                "created_date": "Date",
                "daily_volume": "Total Volume ($USD)",
            },
        )
        fig.update_traces(line_color="#1f77b4", line_width=2)
        fig.update_layout(
            xaxis_title="Date", yaxis_title="Volume ($USD)", hovermode="x"
        )
        st.plotly_chart(fig, use_container_width=True)

elif page == "Churn Predictor":
    st.title("🎯 Single User Churn Prediction")
    st.write(
        "Enter user attributes below to evaluate their predicted churn probability."
    )

    # 1. Create Input Form
    with st.form("churn_prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            crypto_unlocked = st.selectbox(
                "Crypto Unlocked",
                options=[0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
            )
            push_marketing = st.selectbox(
                "Marketing Push Notifications",
                options=[0, 1],
                format_func=lambda x: "Opted In" if x == 1 else "Opted Out",
            )
            email_marketing = st.selectbox(
                "Marketing Email Notifications",
                options=[0, 1],
                format_func=lambda x: "Opted In" if x == 1 else "Opted Out",
            )

        with col2:
            num_contacts = st.number_input(
                "Number of Contacts", min_value=0, value=10, step=1
            )
            num_transactions = st.number_input(
                "Completed Transactions", min_value=0, value=15, step=1
            )
            total_amount_usd = st.number_input(
                "Total Amount Transacted (USD)",
                min_value=0.0,
                value=250.0,
                step=0.1,
            )
        submit_button = st.form_submit_button("Predict Churn Risk")

    # 2. Process Prediction on Form Submission
    if submit_button:
        # Construct DataFrame matching model input schema
        input_data = pd.DataFrame(
            [
                {
                    "user_settings_crypto_unlocked": crypto_unlocked,
                    "attributes_notifications_marketing_push": push_marketing,
                    "attributes_notifications_marketing_email": email_marketing,
                    "num_contacts": num_contacts,
                    "num_completed_transactions": num_transactions,
                    "total_completed_amount_usd": total_amount_usd,
                }
            ]
        )

        # Predict using NumPy array values to suppress feature name warnings
        prediction = clf.predict(input_data.values)[0]
        churn_prob = clf.predict_proba(input_data.values)[0][1]

        # 3. Display Visual Results
        THRESHOLD = 0.5

        st.divider()
        st.subheader("Prediction Outcome")

        col_metric1, col_metric2 = st.columns(2)
        # Extract probability for Class 1 (Churn)
        churn_prob = clf.predict_proba(input_data)[0][1]
        if churn_prob >= THRESHOLD:
            col1.error("Risk Assessment: HIGH CHURN RISK")
            col2.metric("Predicted Churn Risk", f"{churn_prob * 100:.1f}%")
            st.warning(
                "Action Recommended: Trigger an automated re-engagement campaign or offer incentive perks."
            )
        else:
            col1.success("Risk Assessment: LOW CHURN RISK")
            col2.metric("Predicted Churn Risk", f"{churn_prob * 100:.1f}%")
            st.info("User activity looks healthy. No immediate intervention needed.")
