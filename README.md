# 🏦 NeoBank User Engagement & Churn Dashboard

An interactive Streamlit dashboard and churn prediction model that helps a neo-bank understand how its users engage with the app, which users are at risk of leaving, and what the bank can do to keep them.

**Live demo:** https://neobankdataproject.streamlit.app/

## Why this project

The bank in this case is a well-known neo-bank, one of the first to remove hidden fees on payments in other currencies. Acquiring users is expensive, so keeping them active matters. The bank wanted to understand:

- what an "active" user looks like, and how engagement differs across countries, age groups and product usage;
- which users are likely to churn;
- which product and marketing levers could reduce churn and increase engagement.

## Features

The app has four pages, accessible from the sidebar:

- **Project overview:** context on the bank and the goals of the analysis.
- **Overview & KPIs:** an executive view of the business, with a world map of users, retained vs. churned users, transaction success rate, and marketing reach (push and email opt-ins, users with more than 5 contacts).
- **User activity analytics:** filter by country or age group (or click a country on the map) to explore user demographics, daily transaction volume over time, and a funnel showing how users progress from sign-up to marketing opt-in to staying active.
- **Churn predictor:** enter a user's profile and get their predicted churn probability, with a high or low risk flag and a suggested action.

## Defining activity and churn

A user is considered **churned** if they have no activity for 60 days, based on their completed transactions.

Only **completed** transactions count as activity. Declined, failed and reverted transactions are excluded, since they don't reflect a user successfully using the product.

## Key insights

- **Churn rate:** 35% of users churned under the 60-day definition.
- **Engagement drivers:** users who have more contacts churn noticeably less.
- **Transactions:** 87.9% of transaction attempts were completed successfully.

## Business recommendations

- **Grow the user's network:** users with more contacts are more engaged, so make it easier to invite and pay friends.
- **Promote features linked to retention:** surface features like crypto during onboarding.
- **Act on churn risk early:** use the model to flag at-risk users and trigger personalised offers or reminders.

## Churn prediction model

The model is trained in `neo_bank.ipynb` and saved as `churn_model.pkl`. It uses six user features:

| Feature | Description |
|---|---|
| `user_settings_crypto_unlocked` | Whether the user has unlocked crypto in the app |
| `attributes_notifications_marketing_push` | Opted in to marketing push notifications |
| `attributes_notifications_marketing_email` | Opted in to marketing emails |
| `num_contacts` | Number of contacts the user has on the app |
| `num_completed_transactions` | Number of completed transactions |
| `total_completed_amount_usd` | Total amount of completed transactions (USD) |

Users with a predicted probability of 50% or more are flagged as high churn risk.

Model Type: Logistic Regression Model
Precision: 67.9%
Recall:    78.2%
Accuracy:  79.4%

## Data

The dataset was provided through the Le Wagon bootcamp. It contains four tables:

| Table | Description |
|---|---|
| `users` | One row per user: birth year, country, city, sign-up date, plan, crypto unlocked, marketing opt-ins, number of contacts and referrals |
| `transactions` | One row per transaction: type, currency, amount in USD, state (completed, declined, failed, reverted), merchant details, direction |
| `notifications` | One row per notification sent: reason, channel, status, date |
| `devices` | The phone brand associated with each user |

### Data preparation

The raw tables are too large to load directly in a web app, so the notebook prepares smaller, pre-aggregated files for the dashboard: per-user model features, country counts, a daily transaction summary by country and age group, the conversion funnel, and an overall transaction summary.

## Tech stack

Python · pandas · scikit-learn · Plotly · Streamlit · joblib

## Project structure

```
├── app.py                 # Streamlit dashboard
├── neo_bank.ipynb         # EDA, feature engineering and model training
├── churn_model.pkl        # Trained churn model
├── requirements.txt
├── assets/                # Images used in the app and README
└── data/
    ├── df_model.csv                       # Per-user features and churn label
    ├── df_users_cleaned.csv
    ├── df_notifications.csv
    ├── df_devices.csv
    ├── df_country_counts.csv
    ├── df_daily_transaction_summary.csv
    ├── df_conversion_funnel.csv
    └── tx_summary.json
```

## Run it locally

```bash
git clone https://github.com/yiningl7/neobank_data_project.git
cd neobank_data_project
pip install -r requirements.txt
streamlit run app.py
```

## Next steps

- Cohort retention matrix by sign-up month
- Compare retention across plans, device brands and referral behaviour
- Measure the time from sign-up to first transaction and its link to churn
- Explain individual predictions in the churn predictor (which features drive the risk)

## Author

**Andrea Yining Liu** · [GitHub](https://github.com/yiningl7) · [LinkedIn](https://www.linkedin.com/in/yining-liu-10b030221)
