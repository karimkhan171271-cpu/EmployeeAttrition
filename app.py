import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Set matplotlib style
plt.style.use("dark_background")

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="HR Strategic Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# ================= STYLING =================
st.markdown("""
<style>
.main { background: linear-gradient(135deg, #0f2027, #203a43, #2c5364); color: white; }
h1, h2, h3 { color: #00e5ff !important; font-weight: 700; }
.glass-card {
    background: rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 25px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.1);
    margin-bottom: 25px;
}
.reason-box {
    background: rgba(255, 75, 75, 0.05);
    border-left: 4px solid #ff4b4b;
    padding: 12px;
    margin: 8px 0;
}
.stMetric { background: rgba(0,0,0,0.2); padding: 15px; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

st.title("Attrition Sense: A Real Time HR Attrition Explainable System")

# ================= FILE UPLOAD =================
uploaded_file = st.file_uploader("Upload HR Data (CSV)", type=["csv"])

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file).dropna(how='all', axis=1)
    
    # --- DATA PROCESSING ---
    for col in df_raw.columns:
        if any(k in col.lower() for k in ['date', 'dob', 'birth']):
            try:
                df_raw[col] = pd.to_datetime(df_raw[col], errors='coerce')
                if 'hire' in col.lower():
                    df_raw['TenureDays'] = (pd.Timestamp.now() - df_raw[col]).dt.days
                if 'dob' in col.lower() or 'birth' in col.lower():
                    df_raw['CalculatedAge'] = (pd.Timestamp.now() - df_raw[col]).dt.days // 365
            except: pass

    # --- COLUMN DETECTION ---
    attr_keywords = ['attrition', 'left', 'turnover', 'exit', 'termd', 'employmentstatus']
    attrition_col = next((c for c in df_raw.columns if c.lower() in attr_keywords), None)
    salary_col = next((c for c in df_raw.columns if any(k in c.lower() for k in ['income', 'salary', 'pay'])), None)
    dept_col = next((c for c in df_raw.columns if 'dept' in c.lower() or 'department' in c.lower()), None)
    gender_col = next((c for c in df_raw.columns if 'gender' in c.lower() or 'sex' in c.lower()), None)
    emp_id_col = next((c for c in df_raw.columns if any(k in c.lower() for k in ['id', 'number', 'code'])), None)
    age_col = next((c for c in df_raw.columns if ('age' in c.lower() or 'calculatedage' in c.lower()) and pd.api.types.is_numeric_dtype(df_raw[c])), None)

    # ================= SIDEBAR FILTERS =================
    st.sidebar.header("Global Filters")
    df = df_raw.copy()
    
    if dept_col:
        depts = st.sidebar.multiselect("Department", df[dept_col].unique(), default=df[dept_col].unique())
        df = df[df[dept_col].isin(depts)]
    
    if gender_col:
        genders = st.sidebar.multiselect("Gender", df[gender_col].unique(), default=df[gender_col].unique())
        df = df[df[gender_col].isin(genders)]

    # ================= KPI SECTION =================
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Headcount", len(df))
    if attrition_col:
        temp_attr = df[attrition_col].apply(lambda x: 1 if str(x).lower() in ['yes', '1', 'true', 'left', 'terminated', 'voluntarily terminated'] else 0)
        c2.metric("Attrition Rate", f"{temp_attr.mean()*100:.1f}%")
    if salary_col:
        df[salary_col] = pd.to_numeric(df[salary_col], errors='coerce')
        c3.metric("Avg Monthly Pay", f"${int(df[salary_col].mean()):,}")
    if age_col:
        c4.metric("Avg Employee Age", f"{int(df[age_col].mean())} yrs")
    st.markdown("</div>", unsafe_allow_html=True)

    # ================= VISUAL INSIGHTS HUB =================
    st.header("Strategic Insights")
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("3D Talent Attrition Mapping")
        z_axis = age_col if age_col else df.columns[0]
        y_axis = 'TenureDays' if 'TenureDays' in df.columns else df.columns[1]
        
        fig3d = px.scatter_3d(
            df, x=salary_col, y=y_axis, z=z_axis,
            color=attrition_col,
            template="plotly_dark",
            opacity=0.8,
            color_discrete_sequence=['#00e5ff', '#ff4b4b']
        )
        st.plotly_chart(fig3d, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ================= ML PREP =================
    X_raw = df_raw.dropna(subset=[attrition_col]).copy()
    y = X_raw[attrition_col].apply(lambda x: 1 if str(x).lower() in ['yes', '1', 'true', 'left', 'terminated', 'voluntarily terminated'] else 0)
    
    cols_to_drop = [attrition_col]
    if emp_id_col: cols_to_drop.append(emp_id_col)
    for c in X_raw.columns:
        if any(k in c.lower() for k in ['date', 'name', 'surname', 'zip', 'email', 'id']):
            if c not in cols_to_drop and c not in ['CalculatedAge', 'TenureDays']:
                cols_to_drop.append(c)
    
    X = X_raw.drop(columns=cols_to_drop, errors='ignore')
    encoders = {}
    for col in X.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
    X = X.select_dtypes(include=[np.number]).fillna(0)

    # Train Model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)

    with col_r:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Key Attrition Drivers")
        importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=True).tail(10)
        fig_imp = px.bar(importances, orientation='h', template='plotly_dark')
        st.plotly_chart(fig_imp, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ================= SIMULATOR SECTION =================
    st.header("Individual Retention Simulator")
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    
    if emp_id_col:
        selected_emp = st.selectbox("Select Employee for Analysis", df[emp_id_col].unique())
        
        # Get actual data
        idx = df_raw[df_raw[emp_id_col] == selected_emp].index[0]
        actual_data = X.loc[[idx]].copy()
        
        s1, s2 = st.columns([1, 2])
        
        with s1:
            st.subheader("Intervention Controls")
            st.info("Adjust parameters to simulate retention strategies.")
            
            sim_input = actual_data.copy()
            for col in X.columns:
                val = float(actual_data[col].iloc[0])
                if col == salary_col:
                    sim_input[col] = st.slider(f"Adjust {col}", val*0.5, val*2.0, val)
                elif X[col].nunique() < 10:
                    sim_input[col] = st.selectbox(f"Modify {col}", sorted(X[col].unique()), index=list(sorted(X[col].unique())).index(val))
            
        with s2:
            orig_prob = model.predict_proba(actual_data)[0][1] * 100
            sim_prob = model.predict_proba(sim_input)[0][1] * 100
            
            st.subheader("Predictive Risk Assessment")
            diff = sim_prob - orig_prob
            st.metric("Retention Risk Score", f"{sim_prob:.1f}%", delta=f"{diff:.1f}%", delta_color="inverse")
            
            # --- SHAP WATERFALL & REASONING ---
            explainer = shap.Explainer(model, X_train)
            shap_values = explainer(sim_input)
            exp_to_plot = shap_values[0, :, 1] if len(shap_values.shape) == 3 else shap_values[0]

            fig_sim, ax = plt.subplots(figsize=(10, 5))
            shap.plots.waterfall(exp_to_plot, show=False)
            plt.tight_layout()
            st.pyplot(fig_sim)

            # --- DYNAMIC REASONS & SUGGESTIONS ---
            st.markdown("### Decision Factor Breakdown")
            
            # Extract top 3 drivers from SHAP
            sv = exp_to_plot.values
            top_reasons_idx = np.argsort(sv)[-3:][::-1]
            
            for r_idx in top_reasons_idx:
                raw_col_name = X.columns[r_idx]
                feature_name_lower = str(raw_col_name).lower()
                
                st.markdown(f"<div class='reason-box'>Primary Driver: <b>{raw_col_name}</b></div>", unsafe_allow_html=True)
                
                # Prescriptive Suggestion Logic
                if 'salary' in feature_name_lower or 'income' in feature_name_lower:
                    st.info(f"Strategic Recommendation: Compensation review required. Adjust {raw_col_name} to align with market benchmarks.")
                elif 'overtime' in feature_name_lower:
                    st.info("Strategic Recommendation: Work-life balance risk. Monitor workload and evaluate resource allocation.")
                elif 'travel' in feature_name_lower or 'distance' in feature_name_lower:
                    st.info("Strategic Recommendation: Logistical friction detected. Evaluate remote work eligibility or travel frequency.")
                else:
                    st.info(f"Strategic Recommendation: Targeted management intervention required regarding {raw_col_name}.")

            if sim_prob < orig_prob:
                st.success("Intervention Status: Positive. The simulated changes indicate a reduction in attrition risk.")
            elif sim_prob > orig_prob:
                st.error("Intervention Status: Negative. The simulated changes indicate an increased likelihood of attrition.")

    st.markdown("</div>", unsafe_allow_html=True)