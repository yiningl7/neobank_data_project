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

# 3. NAVIGATION (Sidebar)
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to", ["Overview & KPIs", "User Activity Analytics", "Churn Predictor"]
)

# 4. PAGE LOGIC (Using loaded data/model)
if page == "Overview & KPIs":
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

    # --- Section 3: User Engagement & Preferences ---
    st.subheader("📲 Marketing Reach & Network Density")

    # Push Notification Opt-ins
    push_users = (
        df_model["attributes_notifications_marketing_push"].sum()
        if "attributes_notifications_marketing_push" in df_model.columns
        else 0
    )
    push_pct = (push_users / total_users * 100) if total_users > 0 else 0

    # Email Notification Opt-ins
    email_users = (
        df_model["attributes_notifications_marketing_email"].sum()
        if "attributes_notifications_marketing_email" in df_model.columns
        else 0
    )
    email_pct = (email_users / total_users * 100) if total_users > 0 else 0

    # >5 Contacts Calculation
    users_gt_5_contacts = (
        (df_model["num_contacts"] > 5).sum()
        if "num_contacts" in df_model.columns
        else 0
    )
    pct_gt_5_contacts = (
        (users_gt_5_contacts / total_users * 100) if total_users > 0 else 0
    )

    col_eng1, col_eng2, col_eng3 = st.columns(3)

    with col_eng1:
        st.metric("Marketing Push Opt-Ins", f"{int(push_users):,}")
        st.caption(f"**{push_pct:.1f}%** of users have opted in")
        st.progress(min(int(push_pct), 100))

    with col_eng2:
        st.metric("Email Opt-Ins", f"{int(email_users):,}")
        st.caption(f"**{email_pct:.1f}%** of users have opted in")
        st.progress(min(int(email_pct), 100))

    with col_eng3:
        st.metric("Connected Users (>5 Contacts)", f"{users_gt_5_contacts:,}")
        st.caption(f"**{pct_gt_5_contacts:.1f}%** of users have >5 contacts")
        st.progress(min(int(pct_gt_5_contacts), 100))

elif page == "User Activity Analytics":
    st.title("📊 Activity Analytics")

    # ==========================================
    # SECTION 1: DEMOGRAPHICS (AGE & GEOGRAPHY)
    # ==========================================
    st.subheader("Demographic Breakdown")

    col_demo1, col_demo2 = st.columns(2)

    with col_demo1:
        # Create decade bins for birth year
        if "birth_year" in df_users.columns:
            # Determine decade boundaries dynamically
            min_year = int(df_users["birth_year"].min())
            max_year = int(df_users["birth_year"].max())

            # Define decade bins (e.g., 1950s to 2010s)
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

            decade_counts = (
                df_users["decade"]
                .value_counts()
                .reset_index()
                .sort_values("decade")
            )
            decade_counts.columns = ["Decade", "User Count"]

            # Plotly Pie Chart for Decades
            fig_decade = px.pie(
                decade_counts,
                names="Decade",
                values="User Count",
                hole=0.4,
                title="User Age Distribution by Decades",
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
        # Geographic Distribution Map
        if "country" in df_users.columns:
            country_counts = (
                df_users.groupby("country")["user_id"]
                .nunique()
                .reset_index()
            )
            country_counts.columns = ["country", "User Count"]

            # Map 2-letter ISO codes (ISO-2) to 3-letter ISO codes (ISO-3) for Plotly
            # (Plotly choropleths require ISO-3 codes to map two-letter country identifiers properly)
            iso2_to_iso3 = {
                "AF": "AFG",
                "AL": "ALB",
                "DZ": "DZA",
                "AD": "AND",
                "AO": "AGO",
                "AR": "ARG",
                "AM": "ARM",
                "AU": "AUS",
                "AT": "AUT",
                "AZ": "AZE",
                "BE": "BEL",
                "BG": "BGR",
                "BR": "BRA",
                "CA": "CAN",
                "CH": "CHE",
                "CL": "CHL",
                "CN": "CHN",
                "CO": "COL",
                "CY": "CYP",
                "CZ": "CZE",
                "DE": "DEU",
                "DK": "DNK",
                "ES": "ESP",
                "EE": "EST",
                "FI": "FIN",
                "FR": "FRA",
                "GB": "GBR",
                "GR": "GRC",
                "HK": "HKG",
                "HR": "HRV",
                "HU": "HUN",
                "ID": "IDN",
                "IE": "IRL",
                "IL": "ISR",
                "IN": "IND",
                "IS": "ISL",
                "IT": "ITA",
                "JP": "JPN",
                "KR": "KOR",
                "LT": "LTU",
                "LU": "LUX",
                "LV": "LVA",
                "MT": "MLT",
                "MX": "MEX",
                "MY": "MYS",
                "NL": "NLD",
                "NO": "NOR",
                "NZ": "NZL",
                "PL": "POL",
                "PT": "PRT",
                "RO": "ROU",
                "RU": "RUS",
                "SG": "SGP",
                "SK": "SVK",
                "SI": "SVN",
                "SE": "SWE",
                "TR": "TUR",
                "UA": "UKR",
                "US": "USA",
                "ZA": "ZAF",
            }

            country_counts["iso_alpha"] = country_counts["country"].map(
                iso2_to_iso3
            )

            # Flat 2D Map using ISO-3 codes
            fig_map = px.choropleth(
                country_counts,
                locations="iso_alpha",
                color="User Count",
                hover_name="country",
                title="Global User Distribution by Country",
                color_continuous_scale="Viridis",
            )

            # Configure static flat projection with scroll/zoom controls disabled
            fig_map.update_layout(
                geo=dict(
                    showframe=False,
                    showcoastlines=True,
                    projection_type="equirectangular",  # Flat 2D map projection
                    visible=True,
                ),
                margin=dict(t=40, b=20, l=20, r=20),
            )

            st.plotly_chart(
                fig_map, use_container_width=True, config={"scrollZoom": False}
            )

    # ==========================================
    # SECTION 2: NOTIFICATIONS & DEVICES
    # ==========================================
    # Group by reason to find top campaigns
    top_reasons = (
        df_notifications["reason"].value_counts().nlargest(5).index.tolist()
    )

    # Replace long-tail reasons with 'Other'
    df_notif_clean = df_notifications.copy()
    df_notif_clean["reason_clean"] = df_notif_clean["reason"].apply(
        lambda x: x if x in top_reasons else "Other"
    )
    category_order = list(reversed(top_reasons)) + ["Other"]
    # Plot top campaigns
    fig_notif = px.histogram(
        df_notif_clean,
        y="reason_clean",  # Horizontal orientation is easier to read
        color="status",
        barmode="group",
        title="Top 5 Notification Campaigns by Delivery Status",
        labels={"reason_clean": "Campaign Reason", "count": "Notifications Sent"},
        category_orders={"reason_clean": category_order},
    )

    # Sort by count descending
    fig_notif.update_layout(
        margin=dict(t=40, b=20, l=20, r=20),
    )

    st.plotly_chart(fig_notif, use_container_width=True)

    # Classify phone brands into Apple vs Android
    def categorize_brand(brand):
        brand_str = str(brand).lower()
        if "apple" in brand_str:
            return "Apple"
        else:
            return "Android"

    # Apply brand mapping
    df_devices["device_type"] = df_devices["brand"].apply(categorize_brand)

    # Aggregate unique user count by device type
    device_counts = (
        df_devices.groupby("device_type")["user_id"].nunique().reset_index()
    )
    device_counts.columns = ["Device Category", "User Count"]

    # Create interactive Donut Chart using Plotly
    fig_donut = px.pie(
        device_counts,
        names="Device Category",
        values="User Count",
        hole=0.5,
        title="User Distribution by Device",
        color="Device Category",
        color_discrete_map={
            "Apple": "#000000",
            "Android": "#3DDC84",
        },
    )

    # Customise hover and layout styling
    fig_donut.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>Users: %{value:,}<br>Percentage: %{percent}",
    )

    fig_donut.update_layout(showlegend=True, margin=dict(t=40, b=20, l=20, r=20))

    # Render chart in Streamlit
    st.plotly_chart(fig_donut, use_container_width=True)

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
