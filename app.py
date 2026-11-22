
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

DEPLOYMENT_PATH = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------
# LOAD MODELS AND SCHEMA
# --------------------------------------------------

@st.cache_resource
def load_models():
    original = joblib.load(
        os.path.join(DEPLOYMENT_PATH, "original_random_forest.joblib")
    )

    screened = joblib.load(
        os.path.join(DEPLOYMENT_PATH, "screened_random_forest.joblib")
    )

    return {
        "Original Random Forest": original,
        "Screened Random Forest": screened
    }

@st.cache_data
def load_schema():
    with open(
        os.path.join(DEPLOYMENT_PATH, "input_schema.json"),
        "r"
    ) as file:
        return json.load(file)

models = load_models()
schema = load_schema()

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("📊 AI-Based Customer Churn Prediction")
st.subheader("Customer Risk Analysis and Retention Recommendation System")

st.write(
    "This application uses machine learning to estimate customer "
    "churn probability and suggest suitable retention actions."
)

st.warning(
    "Research prototype: predictions are provisional. The timing and "
    "availability of certain input features must be validated before "
    "using this model for real customer-retention decisions."
)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header("Dashboard Settings")

selected_model_name = st.sidebar.selectbox(
    "Select Prediction Model",
    list(models.keys())
)

selected_bundle = models[selected_model_name]
selected_schema = schema[selected_model_name]
selected_model = selected_bundle["model"]

st.sidebar.info(
    f"Selected model: {selected_model_name}\n\n"
    f"Input features: {len(selected_schema['feature_columns'])}"
)

# --------------------------------------------------

# --------------------------------------------------
# DASHBOARD OVERVIEW
# --------------------------------------------------

@st.cache_data
def load_customer_data():
    path = os.path.join(DEPLOYMENT_PATH, "cleaned_customer_churn.csv")
    return pd.read_csv(path)

df_dashboard = load_customer_data()


# --------------------------------------------------
# CUSTOMER RISK SEGMENTATION
# --------------------------------------------------

st.divider()
st.header("🎯 Customer Risk Segmentation")

st.write(
    "Analyze historical customer records and classify their predicted "
    "churn risk using the selected machine learning model."
)

# Prepare customer features
batch_features = df_dashboard[selected_schema["feature_columns"]].copy()

# Predict churn probability for all customers
batch_probabilities = selected_model.predict_proba(
    batch_features
)[:, 1]

batch_predictions = selected_model.predict(batch_features)

# Create customer prediction report
customer_risk_df = pd.DataFrame({
    "CustomerID": df_dashboard["CustomerID"].values,
    "Actual_Churn": df_dashboard["ChurnLabel"].values,
    "Churn_Probability": batch_probabilities,
    "Churn_Probability_Percentage": (
        batch_probabilities * 100
    ).round(2),
    "Predicted_Churn": batch_predictions
})

# Classify risk
customer_risk_df["Risk_Category"] = np.select(
    [
        customer_risk_df["Churn_Probability"] >= 0.70,
        customer_risk_df["Churn_Probability"] >= 0.30
    ],
    [
        "High Risk",
        "Medium Risk"
    ],
    default="Low Risk"
)

# Retention recommendations
retention_recommendations = {
    "High Risk": (
        "Priority personal contact, investigate service issues, "
        "and provide a personalized retention offer."
    ),
    "Medium Risk": (
        "Conduct satisfaction follow-ups, identify customer concerns, "
        "and provide targeted retention offers."
    ),
    "Low Risk": (
        "Maintain service quality, provide loyalty benefits, "
        "and encourage continued engagement."
    )
}

customer_risk_df["Retention_Recommendation"] = (
    customer_risk_df["Risk_Category"]
    .map(retention_recommendations)
)

# Risk summary
risk_summary = (
    customer_risk_df["Risk_Category"]
    .value_counts()
    .reindex(["High Risk", "Medium Risk", "Low Risk"], fill_value=0)
    .reset_index()
)

risk_summary.columns = ["Risk Category", "Customers"]

risk_summary["Percentage"] = (
    risk_summary["Customers"] /
    len(customer_risk_df) * 100
).round(2)

# Risk metrics
high_count = (customer_risk_df["Risk_Category"] == "High Risk").sum()
medium_count = (customer_risk_df["Risk_Category"] == "Medium Risk").sum()
low_count = (customer_risk_df["Risk_Category"] == "Low Risk").sum()

c1, c2, c3 = st.columns(3)

c1.metric("High Risk Customers", f"{high_count:,}")
c2.metric("Medium Risk Customers", f"{medium_count:,}")
c3.metric("Low Risk Customers", f"{low_count:,}")

# Risk distribution chart
st.subheader("Customer Risk Distribution")

fig_risk = px.pie(
    risk_summary,
    names="Risk Category",
    values="Customers",
    hole=0.45,
    title="Predicted Customer Churn Risk"
)

st.plotly_chart(fig_risk, use_container_width=True)

st.dataframe(
    risk_summary,
    use_container_width=True,
    hide_index=True
)

st.caption(
    "These are retrospective predictions on historical dataset records. "
    "They are not validated forecasts of future customer churn."
)

# --------------------------------------------------
# RETENTION STRATEGY MANAGEMENT
# --------------------------------------------------

st.divider()
st.header("💡 Retention Strategy Management")

st.write(
    "Identify customers requiring attention and view their "
    "recommended retention actions."
)

selected_risk = st.selectbox(
    "Filter customers by risk category",
    ["All Customers", "High Risk", "Medium Risk", "Low Risk"],
    key="retention_risk_filter"
)

if selected_risk == "All Customers":
    filtered_customers = customer_risk_df.copy()
else:
    filtered_customers = customer_risk_df[
        customer_risk_df["Risk_Category"] == selected_risk
    ].copy()

st.metric(
    "Customers in Selected Category",
    f"{len(filtered_customers):,}"
)

st.dataframe(
    filtered_customers[
        [
            "CustomerID",
            "Churn_Probability_Percentage",
            "Risk_Category",
            "Retention_Recommendation"
        ]
    ],
    use_container_width=True,
    hide_index=True
)

# --------------------------------------------------
# CUSTOMER REPORT DOWNLOAD
# --------------------------------------------------

st.divider()
st.header("📥 Customer Reports")

st.write(
    "Download the complete customer churn prediction and "
    "retention recommendation report."
)

csv_report = customer_risk_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Complete Customer Retention Report",
    data=csv_report,
    file_name="customer_churn_retention_report.csv",
    mime="text/csv"
)

high_risk_report = customer_risk_df[
    customer_risk_df["Risk_Category"] == "High Risk"
]

high_risk_csv = high_risk_report.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download High-Risk Customer Report",
    data=high_risk_csv,
    file_name="high_risk_customers.csv",
    mime="text/csv"
)



st.header("📊 Dashboard Overview")

total_customers = len(df_dashboard)
churned_customers = (df_dashboard["ChurnLabel"] == "Yes").sum()
retained_customers = (df_dashboard["ChurnLabel"] == "No").sum()
historical_churn_rate = churned_customers / total_customers * 100

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Customers", f"{total_customers:,}")
col2.metric("Churned Customers", f"{churned_customers:,}")
col3.metric("Retained Customers", f"{retained_customers:,}")
col4.metric("Historical Churn Rate", f"{historical_churn_rate:.2f}%")

st.caption(
    "These are historical dataset statistics, not live customer counts."
)

st.subheader("Historical Customer Churn Distribution")

import plotly.express as px

churn_counts = (
    df_dashboard["ChurnLabel"]
    .value_counts()
    .rename_axis("Customer Status")
    .reset_index(name="Number of Customers")
)

fig_churn = px.pie(
    churn_counts,
    names="Customer Status",
    values="Number of Customers",
    hole=0.45,
    title="Historical Churn Distribution"
)

st.plotly_chart(fig_churn, use_container_width=True)

st.divider()

# --------------------------------------------------
# MODEL PERFORMANCE COMPARISON
# --------------------------------------------------

st.header("🏆 Machine Learning Model Performance")

performance_data = pd.DataFrame({
    "Model": [
        "Random Forest",
        "Gradient Boosting",
        "Support Vector Machine",
        "Logistic Regression",
        "Decision Tree"
    ],
    "Accuracy": [0.9595, 0.9532, 0.9517, 0.9510, 0.9482],
    "Precision": [0.9621, 0.8949, 0.8903, 0.8784, 0.9169],
    "Recall": [0.8824, 0.9332, 0.9332, 0.9465, 0.8850],
    "F1-Score": [0.9205, 0.9136, 0.9112, 0.9112, 0.9007],
    "ROC-AUC": [0.9854, 0.9920, 0.9896, 0.9920, 0.9280]
})

st.dataframe(
    performance_data.style.format({
        "Accuracy": "{:.2%}",
        "Precision": "{:.2%}",
        "Recall": "{:.2%}",
        "F1-Score": "{:.2%}",
        "ROC-AUC": "{:.2%}"
    }),
    use_container_width=True,
    hide_index=True
)

selected_metric = st.selectbox(
    "Select evaluation metric",
    ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
)

fig_performance = px.bar(
    performance_data,
    x="Model",
    y=selected_metric,
    title=f"Model Comparison - {selected_metric}",
    text=performance_data[selected_metric].map(lambda x: f"{x:.2%}")
)

fig_performance.update_layout(
    yaxis_tickformat=".0%",
    xaxis_title="Machine Learning Model",
    yaxis_title=selected_metric
)

st.plotly_chart(fig_performance, use_container_width=True)

st.caption(
    "Performance metrics are from the original held-out test-set evaluation. "
    "The screened Random Forest is not included in this five-model comparison."
)

st.divider()

# CUSTOMER INPUT FORM
# --------------------------------------------------

st.header("🔍 Individual Customer Churn Prediction")

st.write(
    "Enter the customer's details below to estimate their churn risk."
)

customer_id = st.text_input(
    "Customer ID (optional)",
    placeholder="Enter customer ID"
)

with st.form("customer_prediction_form"):

    st.subheader("Customer Information")

    input_data = {}

    numerical_features = selected_schema["numerical_features"]
    categorical_features = selected_schema["categorical_features"]
    defaults = selected_schema["defaults"]
    options = selected_schema["options"]

    st.markdown("**Numerical Features**")

    numeric_columns = st.columns(3)

    for i, feature in enumerate(numerical_features):
        default_value = float(defaults[feature])

        with numeric_columns[i % 3]:
            input_data[feature] = st.number_input(
                feature,
                value=default_value,
                format="%.2f",
                key=f"num_{feature}"
            )

    st.markdown("**Categorical Features**")

    categorical_columns = st.columns(2)

    for i, feature in enumerate(categorical_features):

        feature_options = options[feature]
        default_value = defaults[feature]

        if default_value not in feature_options and feature_options:
            default_value = feature_options[0]

        with categorical_columns[i % 2]:
            input_data[feature] = st.selectbox(
                feature,
                options=feature_options,
                index=feature_options.index(default_value),
                key=f"cat_{feature}"
            )

    submitted = st.form_submit_button(
        "Predict Customer Churn",
        type="primary",
        use_container_width=True
    )

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if submitted:

    input_df = pd.DataFrame(
        [input_data],
        columns=selected_schema["feature_columns"]
    )

    try:
        churn_probability = selected_model.predict_proba(
            input_df
        )[0][1]

        prediction = int(
            selected_model.predict(input_df)[0]
        )

        probability_percentage = churn_probability * 100

        if churn_probability >= 0.70:
            risk_category = "High Risk"
            recommendation = (
                "Prioritize personal contact, investigate service issues, "
                "and consider a personalized retention offer."
            )

        elif churn_probability >= 0.30:
            risk_category = "Medium Risk"
            recommendation = (
                "Conduct a satisfaction follow-up, identify customer "
                "concerns, and consider a targeted retention offer."
            )

        else:
            risk_category = "Low Risk"
            recommendation = (
                "Maintain service quality, provide suitable loyalty "
                "benefits, and encourage continued engagement."
            )

        st.divider()
        st.header("Prediction Results")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Churn Probability",
            f"{probability_percentage:.2f}%"
        )

        col2.metric(
            "Predicted Outcome",
            "Likely to Churn" if prediction == 1 else "Likely to Stay"
        )

        col3.metric(
            "Customer Risk",
            risk_category
        )

        st.progress(float(churn_probability))

        st.subheader("Recommended Retention Strategy")
        st.info(recommendation)

        result_df = pd.DataFrame([{
            "CustomerID": customer_id if customer_id else "Not Provided",
            "Selected_Model": selected_model_name,
            "Churn_Probability": round(churn_probability, 4),
            "Churn_Probability_Percentage": round(
                probability_percentage, 2
            ),
            "Predicted_Churn": prediction,
            "Risk_Category": risk_category,
            "Retention_Recommendation": recommendation
        }])

        csv_data = result_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Prediction Result",
            data=csv_data,
            file_name="customer_churn_prediction.csv",
            mime="text/csv"
        )

    except Exception as error:
        st.error(f"Prediction error: {error}")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()


