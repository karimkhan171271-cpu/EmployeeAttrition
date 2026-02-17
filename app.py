import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt

plt.style.use("dark_background")

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="HR Interactive Dashboard",
    page_icon="📊",
    layout="wide"
)

# ================= POWER BI + 3D CSS =================
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}
h1, h2, h3 {
    color: #00e5ff;
    font-weight: 700;
}
.glass-card {
    background: rgba(255, 255, 255, 0.12);
    border-radius: 20px;
    padding: 20px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.37);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.18);
    margin-bottom: 20px;
}
.stButton > button {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    border-radius: 12px;
    padding: 10px 24px;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

st.title("📊 HR Interactive Analytics Dashboard")
st.caption("Power BI–style 3D HR dashboard with Explainable AI (SHAP)")

# ================= HELPER FUNCTIONS =================
def decode_feature_value(feature, value, encoders):
    if feature in encoders:
        try:
            return encoders[feature].inverse_transform([int(value)])[0]
        except:
            return value

    ordinal_maps = {
        "JobSatisfaction": {1: "Low", 2: "Medium", 3: "High", 4: "Very High"},
        "WorkLifeBalance": {1: "Bad", 2: "Good", 3: "Better", 4: "Best"}
    }
    return ordinal_maps.get(feature, {}).get(value, value)

def generate_risk_summary(prob, reasons):
    level = "low" if prob <= 40 else "medium" if prob <= 70 else "high"
    reasons_text = ", ".join(reasons[:3])
    return f"This employee shows a **{level} risk of attrition**, mainly due to {reasons_text}."

# ================= FILE UPLOAD =================
uploaded_file = st.file_uploader("Upload HR CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # ================= KPI CARDS =================
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("👥 Employees", len(df))
    if "Attrition" in df.columns:
        rate = df["Attrition"].value_counts(normalize=True).get("Yes", 0) * 100
        c2.metric("📉 Attrition Rate", f"{rate:.2f}%")
    if "MonthlyIncome" in df.columns:
        c3.metric("💰 Avg Salary", f"₹{int(df['MonthlyIncome'].mean())}")
    if "YearsAtCompany" in df.columns:
        c4.metric("⏳ Avg Tenure", f"{df['YearsAtCompany'].mean():.1f} yrs")
    st.markdown("</div>", unsafe_allow_html=True)

    # ================= VISUALS =================
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("📈 Attrition by Job Role")
        fig = px.histogram(df, x="JobRole", color="Attrition", barmode="group", 
                           template="plotly_dark", color_discrete_map={"Yes": "#ff5252", "No": "#00e5ff"})
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col_r:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("🧊 3D Salary Analysis")
        fig3d = px.scatter_3d(df, x="MonthlyIncome", y="YearsAtCompany", z="Age", color="Attrition",
                              template="plotly_dark", opacity=0.7, color_discrete_map={"Yes": "#ff5252", "No": "#00e5ff"})
        st.plotly_chart(fig3d, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ================= PREDICTION + SHAP =================
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("🧠 Individual Attrition Analysis")

    try:
        model = joblib.load("attrition_model.pkl")
        encoders = joblib.load("encoders.pkl")
        model_features = joblib.load("model_features.pkl")
        explainer = shap.TreeExplainer(model)
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        st.stop()

    emp_col = "EmployeeNumber" if "EmployeeNumber" in df.columns else None

    if emp_col:
        selected_emp = st.selectbox("Select Employee to Analyze", df[emp_col].unique())

        if st.button("Generate Detailed Prediction"):
            input_df = df[df[emp_col] == selected_emp].iloc[[0]].copy()
            input_df.drop(columns=["Attrition"], errors="ignore", inplace=True)

            for col, le in encoders.items():
                if col in input_df.columns:
                    try:
                        input_df[col] = le.transform(input_df[col])
                    except: pass

            input_df = input_df.reindex(columns=model_features, fill_value=0)
            prob = model.predict_proba(input_df)[0][1] * 100
            
            # Display Probability
            st.metric("Risk Score", f"{prob:.2f}%")

            # SHAP Calculation Logic
            shap_values = explainer.shap_values(input_df)
            if isinstance(shap_values, list):
                shap_vals = shap_values[1][0]
                base_val = explainer.expected_value[1]
            else:
                if len(shap_values.shape) == 3:
                    shap_vals = shap_values[0, :, 1]
                    base_val = explainer.expected_value[1]
                else:
                    shap_vals = shap_values[0]
                    base_val = explainer.expected_value

            final_base_val = float(base_val[0]) if isinstance(base_val, (np.ndarray, list)) else float(base_val)

            # --- PLOT SECTION ---
            st.subheader("🔍 Breakdown of Decision Factors")
            exp = shap.Explanation(
                values=np.array(shap_vals, dtype=float),
                base_values=final_base_val,
                data=input_df.iloc[0].values,
                feature_names=list(input_df.columns)
            )

            fig, ax = plt.subplots(figsize=(10, 5))
            shap.plots.waterfall(exp, show=False)
            plt.gcf().set_facecolor("#0f2027")
            st.pyplot(plt.gcf())
            plt.close()

            # --- DYNAMIC EXPLANATION SECTION ---
            st.markdown("---")
            st.subheader("📝 Automated Insight & Recommendations")
            
            # Sort factors for text explanation
            feature_impacts = pd.DataFrame({'Feature': input_df.columns, 'Impact': shap_vals}).sort_values(by='Impact', ascending=False)
            top_risk = feature_impacts.iloc[0]
            top_retention = feature_impacts.iloc[-1]

            c1, c2 = st.columns(2)
            with c1:
                st.info(f"**Why the risk is high:**\n\nThe most significant driver is **{top_risk['Feature']}**, which increased the risk probability by **{top_risk['Impact']*100:.1f}%**. This suggests an issue in this specific area that needs HR intervention.")
            
            with c2:
                st.success(f"**What is working well:**\n\n**{top_retention['Feature']}** is the strongest reason this employee stays. It reduces their attrition risk by **{abs(top_retention['Impact'])*100:.1f}%**. Leveraging this factor is key to retention.")

            # Strategic Recommendation
            st.markdown("### 💡 Recommended Action Plan")
            if top_risk['Feature'] == "OverTime":
                st.write("👉 **Action:** Review workload distribution. High overtime is the primary exit driver for this employee.")
            elif top_risk['Feature'] == "MonthlyIncome":
                st.write("👉 **Action:** Conduct a salary benchmark review. Current compensation is not meeting market expectations for this role.")
            else:
                st.write(f"👉 **Action:** Schedule a 1-on-1 feedback session focusing on **{top_risk['Feature']}** to address concerns before they lead to resignation.")

    st.markdown("</div>", unsafe_allow_html=True)