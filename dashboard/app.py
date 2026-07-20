import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from pathlib import Path
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
st.set_page_config(page_title="GridCast | Electricity Demand Forecasting", page_icon="⚡", layout="wide")
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "clean_energy.csv"
MODEL_PATH = BASE_DIR / "models" / "xgboost.pkl"
@st.cache_data
def load_data():
    try:
        data = pd.read_csv(DATA_PATH)
        if "Datetime" in data.columns:
            data["Datetime"] = pd.to_datetime(data["Datetime"])
        return data
    except FileNotFoundError:
        st.error(f"Dataset not found at: {DATA_PATH}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.stop()
df = load_data()
@st.cache_resource
def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        return model

    except FileNotFoundError:
        st.error(f"XGBoost model not found at: {MODEL_PATH}")
        st.stop()

    except Exception as e:
        st.error(f"Error loading XGBoost model: {e}")
        st.stop()
xgb_model = load_model()
FEATURES = ["Year","Month","Day","Hour","DayOfWeek","Lag_1","Lag_24","Lag_168","Rolling24","Rolling168"]
FEATURE_LABELS = {
    "Year": "Year", "Month": "Month", "Day": "Day of Month",
    "Hour": "Hour of Day", "DayOfWeek": "Day of Week",
    "Lag_1": "Demand 1 Hour Earlier",
    "Lag_24": "Demand 24 Hours Earlier",
    "Lag_168": "Demand 1 Week Earlier",
    "Rolling24": "Average Demand Over Previous 24 Hours",
    "Rolling168": "Average Demand Over Previous 7 Days",
}
@st.cache_data
def create_features(data):
    feature_df = data.copy()
    if "Datetime" not in feature_df.columns or "PJME_MW" not in feature_df.columns:
        return feature_df
    feature_df = feature_df.sort_values("Datetime").reset_index(drop=True)
    feature_df["Year"] = feature_df["Datetime"].dt.year
    feature_df["Month"] = feature_df["Datetime"].dt.month
    feature_df["Day"] = feature_df["Datetime"].dt.day
    feature_df["Hour"] = feature_df["Datetime"].dt.hour
    feature_df["DayOfWeek"] = feature_df["Datetime"].dt.dayofweek
    feature_df["Lag_1"] = feature_df["PJME_MW"].shift(1)
    feature_df["Lag_24"] = feature_df["PJME_MW"].shift(24)
    feature_df["Lag_168"] = feature_df["PJME_MW"].shift(168)
    feature_df["Rolling24"] = feature_df["PJME_MW"].shift(1).rolling(24).mean()
    feature_df["Rolling168"] = feature_df["PJME_MW"].shift(1).rolling(168).mean()
    return feature_df

model_df = create_features(df)
def get_prediction_features(data, selected_datetime):
    history = data[["Datetime", "PJME_MW"]].dropna().copy()
    history = history.sort_values("Datetime").drop_duplicates("Datetime", keep="last")
    history = history.set_index("Datetime")["PJME_MW"]
    target = pd.Timestamp(selected_datetime)
    t1 = target - pd.Timedelta(hours=1)
    t24 = target - pd.Timedelta(hours=24)
    t168 = target - pd.Timedelta(hours=168)
    previous_24 = pd.date_range(end=t1, periods=24, freq="h")
    previous_168 = pd.date_range(end=t1, periods=168, freq="h")
    required = [t1, t24, t168]
    if (not all(t in history.index for t in required)
            or not previous_24.isin(history.index).all()
            or not previous_168.isin(history.index).all()):
        raise ValueError("Required continuous historical demand data is unavailable for this time.")
    values = {
        "Year": target.year, "Month": target.month, "Day": target.day,
        "Hour": target.hour, "DayOfWeek": target.dayofweek,
        "Lag_1": float(history.loc[t1]),
        "Lag_24": float(history.loc[t24]),
        "Lag_168": float(history.loc[t168]),
        "Rolling24": float(history.loc[previous_24].mean()),
        "Rolling168": float(history.loc[previous_168].mean()),
    }
    return pd.DataFrame([values], columns=FEATURES)
st.markdown("""
<style>
/* Theme-aware GridCast UI: inherits Streamlit light/dark theme automatically */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

h1 {
    font-weight: 700;
    letter-spacing: -0.5px;
}

h2, h3 {
    font-weight: 600;
}

.app-subtitle {
    font-size: 1.05rem;
    color: var(--text-color);
    opacity: 0.70;
    margin-top: -0.75rem;
    margin-bottom: 1.75rem;
}

/* Metric cards adapt to Streamlit's active theme */
[data-testid="stMetric"] {
    background: color-mix(in srgb, var(--secondary-background-color) 88%, transparent);
    border: 1px solid color-mix(in srgb, var(--text-color) 16%, transparent);
    padding: 18px;
    border-radius: 12px;
}

[data-testid="stMetricLabel"] {
    font-size: 15px;
}

[data-testid="stMetricValue"] {
    font-size: 26px;
    font-weight: 700;
}

/* Sidebar uses Streamlit's native theme colors */
[data-testid="stSidebar"] {
    background-color: var(--secondary-background-color);
    border-right: 1px solid color-mix(in srgb, var(--text-color) 14%, transparent);
}

[data-testid="stSidebar"] [role="radiogroup"] {
    gap: 0.25rem;
}

/* Theme-aware controls */
.stButton > button,
.stDownloadButton > button {
    border-radius: 8px;
    font-weight: 600;
    transition: transform 0.2s ease;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    transform: translateY(-1px);
}

[data-testid="stDataFrame"],
[data-testid="stAlert"],
[data-testid="stExpander"],
[data-testid="stPlotlyChart"] {
    border-radius: 10px;
}

div[data-testid="stVerticalBlock"] {
    gap: 1rem;
}

/* Keep Streamlit theme handling intact while removing default branding clutter */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)
st.title("GridCast")
st.markdown(
    '<p class="app-subtitle">Short-Term Electricity Demand Forecasting and Explainable Analytics</p>',
    unsafe_allow_html=True
)
st.sidebar.title("GridCast")

st.sidebar.caption(
    "Electricity Demand Forecasting & Analytics"
)
st.sidebar.divider()
page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Next-Hour Forecast",
        "Historical Forecast Simulator",
        "Model Evaluation",
        "Feature Importance",
        "SHAP Explainability",
        "Analytics",
        "Dataset"
    ]
)
if page == "Dashboard":
    st.header("Demand Overview")
    st.caption("Historical electricity demand patterns and key dataset statistics.")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Rows", len(df))
    c2.metric("Columns", len(df.columns))
    c3.metric("Average Demand", f"{df['PJME_MW'].mean():,.2f} MW")
    c4.metric("Maximum Demand", f"{df['PJME_MW'].max():,.2f} MW")

    fig = px.line(df, x="Datetime", y="PJME_MW", title="Electricity Demand Over Time")
    st.plotly_chart(fig, use_container_width=True)

    monthly = (
        df.groupby(df["Datetime"].dt.month)["PJME_MW"]
        .mean()
        .reset_index(name="Average Demand")
        .rename(columns={"Datetime":"Month"})
    )
    st.plotly_chart(
        px.bar(monthly,x="Month",y="Average Demand",title="Average Monthly Demand"),
        use_container_width=True
    )

    corr = df.select_dtypes(include=np.number).corr()
    st.plotly_chart(
        px.imshow(corr,text_auto=True,title="Correlation Matrix"),
        use_container_width=True
    )

elif page == "Next-Hour Forecast":
    st.header("Next-Hour Electricity Demand Forecast")
    st.write(
        "Forecast the electricity demand for the hour immediately after "
        "the latest observation available in the dataset."
    )

    history = df[["Datetime", "PJME_MW"]].dropna().sort_values("Datetime")
    latest_time = history["Datetime"].max()
    forecast_time = latest_time + pd.Timedelta(hours=1)

    c1, c2, c3 = st.columns(3)
    c1.metric("Latest Available Reading", latest_time.strftime("%d %b %Y, %H:%M"))
    c2.metric("Latest Recorded Demand", f"{history.iloc[-1]['PJME_MW']:,.2f} MW")
    c3.metric("Forecast Time", forecast_time.strftime("%d %b %Y, %H:%M"))

    st.caption(
        "Recent demand history is prepared automatically from the dataset. "
        "No manual lag or rolling-average values are required."
    )

    if st.button("Generate Next-Hour Forecast", type="primary"):
        try:
            X = get_prediction_features(df, forecast_time)
            prediction = float(xgb_model.predict(X)[0])

            st.subheader("Forecast Result")
            st.metric("Forecasted Electricity Demand", f"{prediction:,.2f} MW")

            reference = float(X.iloc[0]["Lag_1"])
            difference = prediction - reference
            percent_change = (difference / reference * 100) if reference else 0.0

            a, b, c = st.columns(3)
            a.metric("Previous Hour Demand", f"{reference:,.2f} MW")
            b.metric("Expected Change", f"{difference:+,.2f} MW")
            c.metric("Expected Change (%)", f"{percent_change:+.2f}%")

            with st.expander("View forecast context"):
                context = pd.DataFrame({
                    "Historical Measure": [
                        "Demand 1 Hour Earlier",
                        "Demand 24 Hours Earlier",
                        "Demand 1 Week Earlier",
                        "Average Demand Over Previous 24 Hours",
                        "Average Demand Over Previous 7 Days"
                    ],
                    "Value (MW)": [
                        X.iloc[0]["Lag_1"],
                        X.iloc[0]["Lag_24"],
                        X.iloc[0]["Lag_168"],
                        X.iloc[0]["Rolling24"],
                        X.iloc[0]["Rolling168"]
                    ]
                })
                st.dataframe(context, use_container_width=True, hide_index=True)

            result = pd.DataFrame({
                "Forecast Time": [forecast_time],
                "Forecasted Demand (MW)": [prediction],
                "Previous Hour Demand (MW)": [reference],
                "Expected Change (MW)": [difference]
            })

            st.download_button(
                "Download Forecast",
                result.to_csv(index=False),
                file_name="next_hour_demand_forecast.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Next-hour forecast could not be generated: {e}")


elif page == "Historical Forecast Simulator":
    st.header("Historical Forecast Simulator")
    st.write(
        "Test the forecasting model on a past timestamp and compare its "
        "prediction with the electricity demand that was actually recorded."
    )

    simulation_df = model_df.dropna(subset=FEATURES + ["PJME_MW"]).copy()
    simulation_df = simulation_df.sort_values("Datetime").reset_index(drop=True)

    min_sim_date = simulation_df["Datetime"].min().date()
    max_sim_date = simulation_df["Datetime"].max().date()

    col1, col2 = st.columns(2)

    with col1:
        simulation_date = st.date_input(
            "Simulation Date",
            value=max_sim_date,
            min_value=min_sim_date,
            max_value=max_sim_date,
            key="simulation_date"
        )

    valid_hours = sorted(
        simulation_df.loc[
            simulation_df["Datetime"].dt.date == simulation_date,
            "Datetime"
        ].dt.hour.unique().tolist()
    )

    with col2:
        if valid_hours:
            simulation_hour = st.selectbox(
                "Simulation Hour",
                valid_hours,
                format_func=lambda h: f"{h:02d}:00",
                key="simulation_hour"
            )
        else:
            simulation_hour = None
            st.warning("No hourly observations are available for this date.")

    if simulation_hour is not None:
        simulation_time = pd.Timestamp(simulation_date) + pd.Timedelta(hours=simulation_hour)
        st.caption(f"Selected historical time: {simulation_time.strftime('%d %B %Y, %I:%M %p')}")

        if st.button("Run Historical Simulation", type="primary"):
            try:
                row = simulation_df.loc[
                    simulation_df["Datetime"] == simulation_time
                ]

                if row.empty:
                    raise ValueError("No recorded demand was found for the selected timestamp.")

                X_sim = row[FEATURES]
                actual = float(row.iloc[0]["PJME_MW"])
                predicted = float(xgb_model.predict(X_sim)[0])
                error = predicted - actual
                absolute_error = abs(error)
                percentage_error = (absolute_error / actual * 100) if actual else 0.0

                st.subheader("Simulation Result")
                m1, m2, m3 = st.columns(3)
                m1.metric("Model Forecast", f"{predicted:,.2f} MW")
                m2.metric("Actual Recorded Demand", f"{actual:,.2f} MW")
                m3.metric("Absolute Error", f"{absolute_error:,.2f} MW")

                st.metric("Absolute Percentage Error", f"{percentage_error:.2f}%")

                comparison = pd.DataFrame({
                    "Series": ["Model Forecast", "Actual Demand"],
                    "Demand (MW)": [predicted, actual]
                })

                fig = px.bar(
                    comparison,
                    x="Series",
                    y="Demand (MW)",
                    title="Forecast vs Actual Demand"
                )
                st.plotly_chart(fig, use_container_width=True)

                with st.expander("View historical context used by the model"):
                    context = pd.DataFrame({
                        "Historical Measure": [
                            "Demand 1 Hour Earlier",
                            "Demand 24 Hours Earlier",
                            "Demand 1 Week Earlier",
                            "Average Demand Over Previous 24 Hours",
                            "Average Demand Over Previous 7 Days"
                        ],
                        "Value (MW)": [
                            X_sim.iloc[0]["Lag_1"],
                            X_sim.iloc[0]["Lag_24"],
                            X_sim.iloc[0]["Lag_168"],
                            X_sim.iloc[0]["Rolling24"],
                            X_sim.iloc[0]["Rolling168"]
                        ]
                    })
                    st.dataframe(context, use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"Historical simulation could not be completed: {e}")

elif page == "Model Evaluation":

    st.header("XGBoost Model Evaluation")

    st.write("""
    This page evaluates the trained XGBoost model using the
    feature-engineered electricity demand dataset.
    """)

    try:
        features = [
            "Year",
            "Month",
            "Day",
            "Hour",
            "DayOfWeek",
            "Lag_1",
            "Lag_24",
            "Lag_168",
            "Rolling24",
            "Rolling168"
        ]

        evaluation_df = model_df.dropna(
            subset=features + ["PJME_MW"]
        ).copy()

        # Keep chronological order
        evaluation_df = evaluation_df.sort_values(
            "Datetime"
        ).reset_index(drop=True)

        split_index = int(
            len(evaluation_df) * 0.8
        )

        test_df = evaluation_df.iloc[
            split_index:
        ].copy()

        X_test = test_df[features]

        y_test = test_df["PJME_MW"]

        y_pred = xgb_model.predict(
            X_test
        )

        test_df["Prediction"] = y_pred

        test_df["Residual"] = (
            test_df["PJME_MW"]
            - test_df["Prediction"]
        )

        mae = mean_absolute_error(
            y_test,
            y_pred
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                y_pred
            )
        )

        r2 = r2_score(
            y_test,
            y_pred
        )

        st.subheader(
            "Model Performance Metrics"
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "MAE",
            f"{mae:,.2f} MW"
        )

        col2.metric(
            "RMSE",
            f"{rmse:,.2f} MW"
        )

        col3.metric(
            "R² Score",
            f"{r2:.4f}"
        )

        st.divider()

        st.subheader(
            "Actual vs Predicted Demand"
        )

        plot_df = test_df.head(
            500
        )

        fig = px.line(
            plot_df,
            x="Datetime",
            y=[
                "PJME_MW",
                "Prediction"
            ],
            title=(
                "Actual vs Predicted "
                "Electricity Demand"
            )
        )

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Demand (MW)",
            legend_title="Series",
            hovermode="x unified"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader(
            "Residual Analysis"
        )

        fig2 = px.scatter(
            test_df.sample(
                n=min(
                    2000,
                    len(test_df)
                ),
                random_state=42
            ),
            x="Prediction",
            y="Residual",
            title=(
                "Prediction Residual Plot"
            )
        )

        fig2.add_hline(
            y=0
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

        st.subheader(
            "Prediction Error Distribution"
        )

        fig3 = px.histogram(
            test_df,
            x="Residual",
            nbins=50,
            title=(
                "Distribution of Prediction Errors"
            )
        )

        st.plotly_chart(
            fig3,
            use_container_width=True
        )

        st.subheader(
            "Evaluation Summary"
        )

        st.write(
            f"""
            **Test Samples:** {len(test_df):,}

            **Mean Absolute Error (MAE):**
            {mae:,.2f} MW

            **Root Mean Squared Error (RMSE):**
            {rmse:,.2f} MW

            **R² Score:**
            {r2:.4f}
            """
        )

        result_csv = test_df[
            [
                "Datetime",
                "PJME_MW",
                "Prediction",
                "Residual"
            ]
        ].to_csv(
            index=False
        )

        st.download_button(
            label=(
                "📥 Download Model "
                "Evaluation Results"
            ),
            data=result_csv,
            file_name=(
                "xgboost_model_evaluation.csv"
            ),
            mime="text/csv"
        )

    except Exception as e:

        st.error(
            f"Model evaluation failed: {e}"
        )

        st.info(
            "Check that the XGBoost model was trained "
            "using the same 10 features used by this dashboard."
        )

elif page == "Feature Importance":

    st.header("XGBoost Feature Importance")

    st.write("""
    This page displays the actual feature importance values learned
    by the trained XGBoost model.
    """)

    features = [
        "Year",
        "Month",
        "Day",
        "Hour",
        "DayOfWeek",
        "Lag_1",
        "Lag_24",
        "Lag_168",
        "Rolling24",
        "Rolling168"
    ]

    try:

        importance_values = xgb_model.feature_importances_

        if len(features) != len(importance_values):

            st.error(
                "The number of feature names does not match "
                "the number of features expected by the model."
            )

        else:

            importance_df = pd.DataFrame({
                "Feature": [FEATURE_LABELS.get(f, f) for f in features],
                "Importance": importance_values
            })

            importance_df = importance_df.sort_values(
                by="Importance",
                ascending=False
            )

            top_feature = importance_df.iloc[0]

            st.success(
                f"Most Important Feature: "
                f"{top_feature['Feature']} "
                f"({top_feature['Importance']:.4f})"
            )

            fig = px.bar(
                importance_df.sort_values(
                    by="Importance",
                    ascending=True
                ),
                x="Importance",
                y="Feature",
                orientation="h",
                title="Actual XGBoost Feature Importance",
                text="Importance"
            )

            fig.update_traces(
                texttemplate="%{text:.4f}",
                textposition="outside"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.subheader("Feature Importance Table")

            st.dataframe(
                importance_df,
                use_container_width=True,
                hide_index=True
            )

    except AttributeError:

        st.error(
            "The loaded model does not provide "
            "feature_importances_. Please check that xgboost.pkl "
            "contains the trained XGBoost model."
        )

    except Exception as e:

        st.error(
            f"Could not calculate feature importance: {e}"
        )
elif page == "SHAP Explainability":

    st.header("SHAP Model Explainability")

    st.write("""
    SHAP (SHapley Additive exPlanations) explains how individual
    features influence the predictions made by the XGBoost model.
    """)

    features = [
        "Year",
        "Month",
        "Day",
        "Hour",
        "DayOfWeek",
        "Lag_1",
        "Lag_24",
        "Lag_168",
        "Rolling24",
        "Rolling168"
    ]

    try:

        missing_features = [
            feature
            for feature in features
            if feature not in model_df.columns
        ]

        if missing_features:

            st.warning(
                "The dashboard dataset does not contain all "
                "feature-engineered columns required for SHAP."
            )

            st.write("Missing features:")

            st.write(missing_features)

            st.info(
                "Load your feature-engineered dataset containing "
                "Year, Month, Day, Hour, DayOfWeek, lag features, "
                "and rolling-average features to enable real SHAP analysis."
            )

        else:
            shap_sample = model_df[features].dropna().sample(
                n=min(300, len(model_df[features].dropna())),
                random_state=42
            )

            with st.spinner(
                "Calculating SHAP values..."
            ):

                explainer = shap.TreeExplainer(
                    xgb_model
                )

                shap_values = explainer.shap_values(
                    shap_sample
                )

            st.subheader(
                "Global SHAP Feature Importance"
            )

            fig, ax = plt.subplots()

            shap.summary_plot(
                shap_values,
                shap_sample,
                plot_type="bar",
                show=False
            )

            st.pyplot(fig)

            plt.close(fig)

            st.subheader(
                "SHAP Summary Plot"
            )

            fig2, ax2 = plt.subplots()

            shap.summary_plot(
                shap_values,
                shap_sample,
                show=False
            )

            st.pyplot(fig2)

            plt.close(fig2)

            st.success(
                "SHAP analysis successfully generated "
                "from the trained XGBoost model."
            )

    except Exception as e:

        st.error(
            f"Could not generate SHAP explanations: {e}"
        )
elif page == "Analytics":

    st.header("Analytics")

    numeric_columns = df.select_dtypes(include="number").columns.tolist()

    feature = st.selectbox(
        "Select Numeric Feature",
        numeric_columns,
        format_func=lambda x: FEATURE_LABELS.get(
            x, "Electricity Demand" if x == "PJME_MW" else x
        )
    )

    fig = px.histogram(
        df,
        x=feature,
        nbins=40,
        title=f"Distribution of {feature}"
    )

    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure()

    fig2.add_trace(
        go.Box(
            y=df[feature],
            name=feature
        )
    )

    st.plotly_chart(fig2, use_container_width=True)

elif page == "Dataset":
    st.header("Dataset Preview")
    st.dataframe(df,use_container_width=True)
    st.download_button(
        "Download Dataset",
        df.to_csv(index=False),
        "energy_dataset.csv",
        "text/csv"
    )
st.divider()
st.markdown(
    """
    <div style="text-align:center; opacity:0.60; padding:16px 10px 8px 10px; font-size:0.9rem;">
        GridCast &nbsp;·&nbsp; Electricity Demand Forecasting &nbsp;·&nbsp; Developed by Aswanth Thiruchuthan
    </div>
    """,
    unsafe_allow_html=True
)