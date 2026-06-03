from sqlalchemy import label
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from datetime import date, datetime

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Airport MHE Failure Risk Prediction System",
    layout="wide"
)

# ---------------------------------------------------
# LOAD FILES
# ---------------------------------------------------

with open("model/catboost_mhe_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/feature_encoders.pkl", "rb") as f:
    feature_encoders = pickle.load(f)

dataset = pd.read_csv(
    "dataset/Airport_MHE_Final_Dataset.csv"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------

st.markdown("""
<style>
.block-container {
    padding-top: 1rem;
    padding-bottom: 0rem;
}

.main {
    background-color: #f5f7fb;
}

h1 {
    color: #0b1b5e;
    font-weight: 800;
}

h2, h3 {
    color: #0b1b5e;
}

.stButton>button {
    background-color: #1565ff;
    color: white;
    border-radius: 10px;
    height: 50px;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
    border: none;
    margin-top:0px
}

.result-box-low {
    background-color: #dcfce7;
    padding: 25px;
    border-radius: 10px;
    font-size: 22px;
    font-weight: bold;
    color: #166534;
}

.result-box-medium {
    background-color: #fef9c3;
    padding: 25px;
    border-radius: 10px;
    font-size: 22px;
    font-weight: bold;
    color: #854d0e;
}

.result-box-high {
    background-color: #fee2e2;
    padding: 25px;
    border-radius: 10px;
    font-size: 22px;
    font-weight: bold;
    color: #991b1b;
}

.stSelectbox div[data-baseweb="select"] > div {
    min-height: 40px;
}

.stNumberInput input {
    height: 40px;
}

.stDateInput input {
    height: 40px;
}

.metric-card {
    background-color: white;
    padding: 15px;
    border-radius: 12px;
}

/* KPI CARD */
[data-testid="stMetric"]{
    background:white;
    padding:20px;
    border-radius:18px;
    border-top:5px solid #111827;
    box-shadow:0px 2px 8px rgba(0,0,0,0.05);
    text-align:left;
}

/* KPI VALUE */
[data-testid="stMetricValue"]{
    font-size:38px;
    font-weight:700;
    color:#111827;
}

/* KPI LABEL */
[data-testid="stMetricLabel"]{
    font-size:16px;
    color:gray;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.markdown(
    """
    <h1 style='
        font-size:40px;
        color:#1d2340;
        font-weight:600;
        margin-bottom:0px;
    '>
    Airport MHE Failure Risk Prediction System
    </h1>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------
# TABS
# ---------------------------------------------------

tab1, tab2 = st.tabs([
    "Prediction",
    "Analytics Dashboard"
])

# ===================================================
# PREDICTION PAGE
# ===================================================

with tab1:
    col1, col2 = st.columns(2)

    # --------------------------------------------
    # LEFT COLUMN
    # --------------------------------------------

    with col1:
        record_date = st.date_input(
            "Inspection Date",
            value=date.today()
        )
        
        equipment_type = st.selectbox(
            "Equipment Type",
            sorted(dataset["Equipment_Type"].unique()),help="Type of airport ground support equipment."
        )

        filtered_ids = (
            dataset[dataset["Equipment_Type"] == equipment_type]["Equipment_ID"].unique()
        )
        
        equipment_id = st.selectbox(
            "Equipment ID",
            sorted(filtered_ids),help="Unique identifier of the equipment."
        )


        equipment_age = st.number_input(
            "Equipment Age (Years)",
            min_value=0.0,
            max_value=30.0,
            value=15.0,help=" Age of equipment in years.Higher age generally increases failure risk."
        )

        hydraulic_leak = st.selectbox(
            "Hydraulic Leak Observed",
            ["No", "Yes"]
        )

        consecutive_days = st.number_input(
            "Consecutive Working Days",
            min_value=0,
            max_value=60,
            value=30
        )
        downtime_hours = st.number_input(
            "Downtime Hours Last 30 Days",
            min_value=0,
            max_value=100,
            value=25
        )
        
        daily_operating_hours = st.slider(
            "Daily Operating Hours",
            0,
            24,
            22
        )

    # --------------------------------------------
    # RIGHT COLUMN
    # --------------------------------------------

    with col2:
        cargo_weight = st.number_input(
            "Cargo Weight Handled (KG)",
            min_value=0,
            max_value=50000,
            value=45000
        )
        
        overload_count = st.number_input(
            "Overload Event Count",
            min_value=0,
            max_value=20,
            value=10
        )

        warning_count = st.number_input(
            "Warning Indicator Count",
            min_value=0,
            max_value=20,
            value=10
        )

        outdoor_usage = st.selectbox(
            "Outdoor Usage",
            ["No", "Yes"]
        )

        peak_hour = st.selectbox(
            "Peak Hour Operations",
            ["No", "Yes"]
        )

        uld_count = st.number_input(
            "ULD Handling Count",
            min_value=0,
            max_value=100,
            value=90
        )

    # ------------------------------------------------
    # PREDICT BUTTON
    # ------------------------------------------------

    if st.button("Predict Failure Risk Level"):
        # Date Features
        year = record_date.year
        month = record_date.month
        quarter = ((month - 1) // 3) + 1
        # ----------------------------------------
        # INPUT DATAFRAME
        # ----------------------------------------

        input_data = pd.DataFrame({
            'Equipment_ID':[equipment_id],
            'Record_Date':[record_date.strftime("%Y-%m-%d")],
            'Equipment_Age_Year':[equipment_age],
            'Daily_Operating_Hours':[daily_operating_hours],
            'Cargo_Weight_Handled_KG':[cargo_weight],
            'Overload_Event_Count':[overload_count],
            'Hydraulic_Leak_Observed':[hydraulic_leak],
            'Consecutive_Working_Days':[consecutive_days],
            'Downtime_Hours_Last_30D':[downtime_hours],
            'Warning_Indicator_Count':[warning_count],
            'Outdoor_Usage':[outdoor_usage],
            'Peak_Hour_Operations':[peak_hour],
            'ULD_Handling_Count':[uld_count],
            'Month':[month],
            'Quarter':[quarter],
            'Year':[year]

        })
        input_data['Operator_Experience_Years'] = dataset['Operator_Experience_Years'].mode()[0]
        input_data['Breakdown_Count_6M'] = dataset['Breakdown_Count_6M'].mode()[0]
        input_data['Battery_Status'] = dataset['Battery_Status'].mode()[0]
        
        input_data = input_data[[
        'Equipment_ID',
        'Equipment_Age_Year',
        'Daily_Operating_Hours',
        'ULD_Handling_Count',
        'Consecutive_Working_Days',
        'Cargo_Weight_Handled_KG',
        'Overload_Event_Count',
        'Breakdown_Count_6M',
        'Operator_Experience_Years',
        'Battery_Status',
        'Hydraulic_Leak_Observed',
        'Outdoor_Usage',
        'Warning_Indicator_Count',
        'Downtime_Hours_Last_30D',
        'Peak_Hour_Operations',
        'Month',
        'Quarter',
        'Year'
    ]]

        # ----------------------------------------
        # ENCODING
        # ----------------------------------------

        categorical_cols = input_data.select_dtypes(
            include='object'
        ).columns

        for col in categorical_cols:
            if col in feature_encoders:
                try:
                    input_data[col] = feature_encoders[col].transform(
                        input_data[col]
                    )
                except Exception:
                    input_data[col] = 0

        # ----------------------------------------
        # PREDICTION
        # ----------------------------------------

        prediction = model.predict(input_data)[0]

        # Handle array output from CatBoost
        if hasattr(prediction, '__len__'):
            prediction = prediction[0]

        prediction = str(prediction)

        # Try getting probabilities
        try:
            proba = model.predict_proba(input_data)[0]
            classes = model.classes_
            proba_dict = {
                str(c): round(float(p) * 100, 1)
                for c, p in zip(classes, proba)
            }
        except Exception:
            proba_dict = {}

        result_col, chart_col = st.columns([1, 1])

        # ----------------------------------------
        # OUTPUT
        # ----------------------------------------

        with result_col:
            st.subheader("Prediction Result")

            if prediction == "High":
                st.markdown(f"""
                <div class="result-box-high">
                🚨 HIGH Failure Risk Detected
                <br><br>
                Immediate maintenance recommended.
                </div>
                """, unsafe_allow_html=True)

            elif prediction == "Medium":
                st.markdown(f"""
                <div class="result-box-medium">
                ⚠️ MEDIUM Failure Risk Detected
                <br><br>
                Schedule maintenance soon.
                </div>
                """, unsafe_allow_html=True)

            else:
                st.markdown(f"""
                <div class="result-box-low">
                ✅ LOW Failure Risk Detected
                <br><br>
                Equipment is in good condition.
                </div>
                """, unsafe_allow_html=True)

            # Show probability breakdown if available
            if proba_dict:
                st.markdown("<br>", unsafe_allow_html=True)
                st.subheader("Risk Probability Breakdown")

                prob_df = pd.DataFrame(
                    list(proba_dict.items()),
                    columns=["Risk Level", "Probability (%)"]
                )

                color_map = {
                    "Low": "#16a34a",
                    "Medium": "#ca8a04",
                    "High": "#dc2626"
                }

                fig_prob = px.pie(
                    prob_df,
                    values="Probability (%)",
                    names="Risk Level",
                    title="Risk Probability Breakdown",
                    color_discrete_map=color_map
                )   
                
                

                fig_prob.update_traces(
                    textinfo='label+percent+value',
                    texttemplate='%{label}<br>%{value:.1f}%',
                    textposition='outside'
                )

                fig_prob.update_layout(
                    height=300,
                    showlegend=False,
                    template="plotly_white",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=10, r=10, t=20, b=20),
                    xaxis=dict(title=""),
                    yaxis=dict(
                        title="Probability (%)",
                        range=[0, 110],
                        gridcolor="#E5E7EB"
                    )
                )

                st.plotly_chart(
                    fig_prob,
                    use_container_width=True,
                    config={"displayModeBar": False}
                )

        # ----------------------------------------
        # FEATURE IMPACT CHART
        # ----------------------------------------

        with chart_col:
            st.subheader("Feature Impact Analysis")

            try:
                importance = model.get_feature_importance()
                feat_names = input_data.columns.tolist()

                feat_df = pd.DataFrame({
                    'Feature': feat_names,
                    'Importance': importance[:len(feat_names)]
                }).sort_values(
                    by='Importance',
                    ascending=False
                ).head(10)

                fig_imp, ax = plt.subplots(figsize=(7, 5))
                ax.barh(
                    feat_df['Feature'],
                    feat_df['Importance'],
                    color='#1565ff'
                )
                ax.invert_yaxis()
                ax.set_xlabel("Importance Score")
                ax.set_title("Top Features Affecting Prediction")
                st.pyplot(fig_imp)

            except Exception:
                st.info("Feature importance chart not available for this prediction.")

# ===================================================
# ANALYTICS DASHBOARD
# ===================================================

with tab2:

    # =========================================================
    # DATASET DATE RANGE
    # =========================================================

    dataset['Record_Date'] = pd.to_datetime(
        dataset['Record_Date'],
        format='mixed',
        dayfirst=True,
        errors='coerce'
    )

    dataset_start_date = dataset['Record_Date'].min()
    dataset_end_date = dataset['Record_Date'].max()

    # --------------------------------------------
    # FILTERS
    # --------------------------------------------

    filter_col1, filter_col2, filter_col3 = st.columns(3)

    with filter_col1:
        from_date = st.date_input(
            "From Date",
            value=dataset_start_date,
            min_value=dataset_start_date,
            max_value=dataset_end_date
        )

    with filter_col2:
        to_date = st.date_input(
            "To Date",
            value=dataset_end_date,
            min_value=dataset_start_date,
            max_value=dataset_end_date
        )

    with filter_col3:
        quick_filter = st.selectbox(
            "Quick Filter", [
                "All Time",
                "Last 30 Days",
                "Last 90 Days",
                "Last 6 Months",
                "Last 1 Year"
            ]
        )

    # =========================================================
    # DATE FILTERING
    # =========================================================

    filtered_data = dataset.copy()
    filtered_data['Record_Date'] = pd.to_datetime(
        filtered_data['Record_Date'],
        format='mixed',
        dayfirst=True,
        errors='coerce'
    )

    today = filtered_data['Record_Date'].max()

    if quick_filter == "Last 30 Days":
        from_date = today - pd.Timedelta(days=30)
    elif quick_filter == "Last 90 Days":
        from_date = today - pd.Timedelta(days=90)
    elif quick_filter == "Last 6 Months":
        from_date = today - pd.DateOffset(months=6)
    elif quick_filter == "Last 1 Year":
        from_date = today - pd.DateOffset(years=1)

    filtered_data = filtered_data[
        (filtered_data['Record_Date'] >= pd.to_datetime(from_date))
        &
        (filtered_data['Record_Date'] <= pd.to_datetime(to_date))
    ]

    # =========================================================
    # KPI VALUES
    # =========================================================

    total_records = len(filtered_data)

    high_risk_count = filtered_data[
        filtered_data['Failure_Risk_Level'] == 'High'
    ].shape[0]

    medium_risk_count = filtered_data[
        filtered_data['Failure_Risk_Level'] == 'Medium'
    ].shape[0]

    high_risk_rate = (
        high_risk_count / total_records * 100
        if total_records > 0 else 0
    )

    avg_age = filtered_data['Equipment_Age_Year'].mean()

    avg_operating_hours = filtered_data['Daily_Operating_Hours'].mean()

    # =========================================================
    # KPI CARDS
    # =========================================================

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

    with kpi1:
        st.metric(
            label="Total Records",
            value=f"{total_records:,}"
        )

    with kpi2:
        st.metric(
            label="High Risk Count",
            value=f"{high_risk_count:,}"
        )

    with kpi3:
        st.metric(
            label="High Risk Rate",
            value=f"{round(high_risk_rate, 1)}%"
        )

    with kpi4:
        st.metric(
            label="Avg Equipment Age",
            value=f"{round(avg_age, 1)} yrs"
        )

    with kpi5:
        st.metric(
            label="Avg Daily Hours",
            value=f"{round(avg_operating_hours, 1)} hrs"
        )

    # =========================================================
    # ROW 1
    # =========================================================

    chart1, chart2 = st.columns(2)

    # =========================================================
    # 1. MONTHLY FAILURE RISK TREND
    # =========================================================

    with chart1:
        monthly_risk = (
            filtered_data[
                filtered_data['Failure_Risk_Level'] == 'High'
            ]
            .groupby(
                filtered_data['Record_Date'].dt.month_name().str[:3]
            )
            .size()
            .reset_index(name='High_Risk_Count')
        )

        month_order = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
        ]

        monthly_risk['Record_Date'] = pd.Categorical(
            monthly_risk['Record_Date'],
            categories=month_order,
            ordered=True
        )

        monthly_risk = monthly_risk.sort_values('Record_Date')

        fig1 = px.line(
            monthly_risk,
            x='Record_Date',
            y='High_Risk_Count',
            markers=True
        )

        fig1.update_traces(
            mode="lines+markers+text",
            text=monthly_risk['High_Risk_Count'],
            textposition="top center",
            line=dict(color="#0b66c3", width=3),
            marker=dict(size=8, color="#0b66c3"),
            textfont=dict(size=12, color="#111827")
        )

        fig1.update_layout(
            title="1. Monthly High Risk Trend",
            title_font_size=24,
            title_font_color="#111827",
            height=420,
            template="plotly_white",
            paper_bgcolor="white",
            plot_bgcolor="white",
            margin=dict(l=20, r=20, t=70, b=20),
            xaxis=dict(
                title="Month",
                showgrid=False,
                tickfont=dict(size=14)
            ),
            yaxis=dict(
                title="High Risk Count",
                gridcolor="#E5E7EB",
                tickfont=dict(size=14)
            )
        )

        st.plotly_chart(fig1, use_container_width=True)

    # =========================================================
    # 2. SHIFT-WISE FAILURE RISK
    # =========================================================

    with chart2:
        equipment_risk = (
            dataset[dataset['Failure_Risk_Level'] == 'High']
            .groupby('Equipment_Type')
            .size()
            .reset_index(name='Count')
        )

        fig_equipment = px.pie(
            equipment_risk,
            names='Equipment_Type',
            values='Count',
            hole=0,
            color_discrete_sequence=[ '#0B3D91', '#1565C0', '#1976D2', '#1E88E5', '#42A5F5', '#64B5F6' ],
            title='Equipment Type Risk Distribution'
        )

        fig_equipment.update_traces(
            textposition='outside',
            textinfo='percent+label',
            pull=[0.03] * len(equipment_risk)
        )

        fig_equipment.update_layout(
            title={
                'text': '<b>2. Equipment Type vs High Risk</b>',
                'x': 0.43,   # Adjust because legend occupies right side
                'xanchor': 'center',
                'y':0.97
            },
            showlegend=False,
            title_font_size=24,
            title_font_color="#111827",
            height=450,
        )

        st.plotly_chart(
            fig_equipment,
            use_container_width=True
        )

    # =========================================================
    # ROW 2
    # =========================================================

    chart3, chart4, chart5 = st.columns(3)

    # =========================================================
    # 3. EQUIPMENT TYPE VS FAILURE RISK
    # =========================================================

    with chart3:
        equip_data = (
            filtered_data[
                filtered_data['Failure_Risk_Level'] == 'High'
            ]
            .groupby('Equipment_Type')
            .size()
            .reset_index(name='High_Risk_Count')
            .sort_values('High_Risk_Count', ascending=False)
            .head(6)
        )

        fig3 = px.bar(
            equip_data,
            x='Equipment_Type',
            y='High_Risk_Count',
            text='High_Risk_Count',
            title="3. Equipment Type vs High Risk",
            color='High_Risk_Count',
            color_continuous_scale=["#BFDBFE", "#1D4ED8"]
        )

        fig3.update_traces(
            textposition='outside'
        )

        fig3.update_layout(
            title_font_size=24,
            title_font_color="#111827",
            height=420,
            showlegend=False,
            coloraxis_showscale=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=70, b=20),
            xaxis=dict(
                title="High Risk Count",
                gridcolor="#E5E7EB",
                tickfont=dict(size=12)
            ),
            yaxis=dict(title="", tickfont=dict(size=12))
        )

        st.plotly_chart(
            fig3,
            use_container_width=True,
            config={"displayModeBar": False}
        )

    # =========================================================
    # 4. RISK LEVEL DISTRIBUTION (PIE)
    # =========================================================

    with chart4:
        risk_dist = (
            filtered_data['Failure_Risk_Level']
            .value_counts()
            .reset_index()
        )
        risk_dist.columns = ['Risk_Level', 'Count']

        fig4 = px.pie(
            risk_dist,
            names='Risk_Level',
            values='Count',
            hole=0.55,
            color='Risk_Level',
            color_discrete_map={
                'Low': '#0B3D91',
                'Medium': '#1565C0',
                'High': '#1976D2'
            }
        )
        fig4.update_traces(
            textposition='outside',
            textinfo='percent+label',
            hoverinfo="skip",
            hovertemplate=None,
            pull=[0.03, 0, 0],
            marker=dict(
                line=dict(color='white', width=2)
            )
        )

        fig4.update_layout(
            title=dict(
                text="4. Failure Risk Distribution",
                x=0,
                font=dict(size=22, color="#111827")
            ),
            height=420,
            template="plotly_white",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=70, b=20),
            showlegend=False
        )

        st.plotly_chart(
            fig4,
            use_container_width=True,
            config={"displayModeBar": False}
        )

    # =========================================================
    # 5. EQUIPMENT AGE VS FAILURE RISK
    # =========================================================

    with chart5:
        age_risk = (
            filtered_data
            .groupby('Failure_Risk_Level')['Equipment_Age_Year']
            .mean()
            .reset_index()
        )

        age_risk['Equipment_Age_Year'] = (
            age_risk['Equipment_Age_Year']
        ).round(1)

        # Keep proper order
        order_map = {'Low': 0, 'Medium': 1, 'High': 2}
        age_risk['order'] = age_risk['Failure_Risk_Level'].map(order_map)
        age_risk = age_risk.sort_values('order')

        fig5 = px.bar(
            age_risk,
            x='Failure_Risk_Level',
            y='Equipment_Age_Year',
            text='Equipment_Age_Year',
            title="5. Equipment Age vs Failure Risk",
            color='Failure_Risk_Level',
            color_discrete_map={
                'Low': '#bfdbfe',
                'Medium': '#60a5fa',
                'High': '#1d4ed8'
            }
        )

        fig5.update_traces(
            texttemplate='%{text:.1f} yrs',
            textposition='outside'
        )

        fig5.update_layout(
            height=420,
            template="plotly_white",
            showlegend=False,
            paper_bgcolor="white",
            plot_bgcolor="white",
            title_font_size=24,
            title_font_color="#111827",
            margin=dict(l=20, r=20, t=70, b=20),
            xaxis=dict(
                title="Failure Risk Level",
                tickfont=dict(size=15)
            ),
            yaxis=dict(
                title="Avg Equipment Age (Years)",
                gridcolor="#E5E7EB",
                tickfont=dict(size=13)
            )
        )

        st.plotly_chart(
            fig5,
            use_container_width=True,
            config={"displayModeBar": False}
        )
