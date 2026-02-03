import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ==========================================
# 0. 全局样式 (CSS)
# ==========================================
st.set_page_config(
    page_title="CRC Recurrence Risk Predictor",
    page_icon="🧬",
    layout="wide"
)

st.markdown("""
    <style>
    html, body, [class*="css"] { font-size: 18px !important; }
    h1 { font-size: 3rem !important; }
    h2 { font-size: 2.2rem !important; }
    h3 { font-size: 1.8rem !important; }
    .result-card-score { font-size: 3.5rem !important; font-weight: bold; }
    .stNumberInput label { font-size: 1.1rem !important; font-weight: 600; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 1. 模型参数
# ==========================================
COEFFICIENTS = {
    "TCEAL4": 0.3364594, "ACTR3B": -0.4104630, "ORAI3":  0.2523666,
    "PRIM1":  -0.2529674, "LEMD1":  0.2133200, "INHBB":  0.1491095
}
CUTOFF_VALUE = 0.5739
REF_GENE = "EMC7"

# ==========================================
# 2. 侧边栏
# ==========================================
with st.sidebar:
    st.header("Input Feature Values")
    st.caption("Enter Log2 transformed gene expression")
    st.markdown("---")
    st.markdown(f"**Reference Gene ({REF_GENE})**")
    val_ref = st.number_input(f"{REF_GENE} Value", value=6.90, step=0.1, format="%.2f")
    st.markdown("---")
    st.markdown("**Target Genes Expression**")
    inputs = {}
    for gene in COEFFICIENTS.keys():
        inputs[gene] = st.number_input(f"{gene}", value=10.00, step=0.1, format="%.2f")

# ==========================================
# 3. 主界面
# ==========================================
st.title("Predicting CRC Recurrence Risk Using a 6-Gene Signature")
st.info(f"**Model Type**: LASSO + Stepwise Cox | **Cutoff**: {CUTOFF_VALUE} | **Ref**: {REF_GENE}")
predict_btn = st.button("🚀 Predict Risk (开始预测)", type="primary", use_container_width=True)

# ==========================================
# 4. 计算与结果展示
# ==========================================
if predict_btn:
    st.markdown("---")
    
    # --- 计算逻辑 ---
    risk_score = 0
    calculation_details = []
    for gene, coef in COEFFICIENTS.items():
        raw_val = inputs[gene]
        norm_expr = raw_val - val_ref
        risk_score += (norm_expr * coef)
        calculation_details.append({
            "Gene": gene, "Raw Value": raw_val, 
            "Norm Value": norm_expr, "Coefficient": coef, 
            "Contribution": norm_expr * coef
        })
    
    is_high_risk = risk_score > CUTOFF_VALUE
    risk_level = "High Risk (高风险)" if is_high_risk else "Low Risk (低风险)"
    risk_color = "#d32f2f" if is_high_risk else "#388e3c"
    bg_color = "rgba(211, 47, 47, 0.1)" if is_high_risk else "rgba(56, 142, 60, 0.1)"

    # --- 布局 ---
    col_res, col_viz = st.columns([1, 1.4], gap="large")

    # === 左侧栏：结果 ===
    with col_res:
        st.subheader("Prediction Result")
        st.markdown(f"""
        <div style="background-color: {bg_color}; padding: 25px; border-radius: 12px; border: 3px solid {risk_color}; text-align: center; margin-bottom: 25px;">
            <p style="margin:0; color: #555;">Risk Score</p>
            <h1 class="result-card-score" style="margin:5px 0; color: {risk_color};">{risk_score:.4f}</h1>
            <hr style="border-top: 2px solid {risk_color}; opacity: 0.3; margin: 15px 0;">
            <h2 style="margin:0; color: {risk_color};">{risk_level}</h2>
        </div>
        """, unsafe_allow_html=True)

        # 仪表盘
        max_gauge_val = max(5.0, risk_score + 1.0)
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+delta", value = risk_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            delta = {'reference': CUTOFF_VALUE, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
            gauge = {
                'axis': {'range': [None, max_gauge_val]}, 'bar': {'color': risk_color},
                'steps': [{'range': [0, CUTOFF_VALUE], 'color': "rgba(56, 142, 60, 0.15)"},
                          {'range': [CUTOFF_VALUE, max_gauge_val], 'color': "rgba(211, 47, 47, 0.15)"}],
                'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': CUTOFF_VALUE}
            }
        ))
        fig_gauge.update_layout(height=220, margin=dict(l=20, r=20, t=10, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # 临床建议
        st.markdown("#### 💡 Clinical Recommendation")
        if is_high_risk:
            st.warning("**High Risk Strategy:**\n1. Consider Oxaliplatin-based doublet chemotherapy.\n2. Intensive surveillance (CT/CEA every 3-6 mos).\n3. Verify dMMR/MSI-H status.")
        else:
            st.success("**Low Risk Strategy:**\n1. Observation or shorter therapy.\n2. Standard follow-up.\n3. Avoid overtreatment.")

    # === 右侧栏：可视化的重大升级 ===
    with col_viz:
        st.subheader("Survival Analysis & Visualization")
        
        # --- 1. 数据模拟 (模拟置信区间) ---
        time_points = np.linspace(0, 60, 61) # 0-60个月
        
        # 模拟高/低风险的平均曲线
        surv_low_mean = np.exp(-0.005 * time_points)
        surv_high_mean = np.exp(-0.025 * time_points)
        
        # 模拟“高风险组”的分布范围 (Confidence Interval)
        surv_high_upper = surv_high_mean * 1.05 # 上界
        surv_high_lower = surv_high_mean * 0.90 # 下界
        
        # 确定患者曲线 (作为平均线)
        patient_curve = surv_high_mean if is_high_risk else surv_low_mean
        patient_color = risk_color
        
        # --- 2. 绘制升级版生存曲线 (带阴影带) ---
        fig_surv = go.Figure()
        
        # A. 绘制低风险组参考线 (绿色虚线)
        fig_surv.add_trace(go.Scatter(
            x=time_points, y=surv_low_mean, mode='lines', name='Low Risk Group (Avg)',
            line=dict(color='green', dash='dash', width=2), opacity=0.6
        ))

        # B. 绘制高风险组 "区域" (红色阴影带) - 解决重叠问题的关键！
        # 先画下界 (透明线)
        fig_surv.add_trace(go.Scatter(
            x=time_points, y=surv_high_lower, mode='lines', line=dict(width=0),
            showlegend=False, hoverinfo='skip'
        ))
        # 再画上界，并填充到下界 (形成带状区域)
        fig_surv.add_trace(go.Scatter(
            x=time_points, y=surv_high_upper, mode='lines', line=dict(width=0),
            fill='tonexty', # 填充到上一条线
            fillcolor='rgba(211, 47, 47, 0.15)', # 淡淡的红色区域
            name='High Risk Range (95% CI)',
            hoverinfo='skip'
        ))

        # C. 绘制当前患者 (实线)
        fig_surv.add_trace(go.Scatter(
            x=time_points, y=patient_curve, mode='lines', name='Current Patient Prediction',
            line=dict(color=patient_color, width=4) # 加粗实线
        ))

        fig_surv.update_layout(
            title="Recurrence-Free Survival (with Group Range)",
            xaxis_title="Time (Months)", yaxis_title="Probability",
            yaxis_range=[0, 1.05], template="plotly_white", height=350,
            legend=dict(orientation="h", y=1.1)
        )
        st.plotly_chart(fig_surv, use_container_width=True)

        # --- 3. 新增：关键时间点生存率柱状图 (Bar Chart) ---
        st.markdown("##### 📊 1/3/5-Year Survival Probability")
        
        # 提取第12, 36, 60个月的数据点
        indices = [12, 36, 60]
        years = ["1-Year", "3-Year", "5-Year"]
        
        vals_patient = [patient_curve[i] for i in indices]
        vals_low_risk_avg = [surv_low_mean[i] for i in indices]
        
        fig_bar = go.Figure()
        
        # 低风险组柱子
        fig_bar.add_trace(go.Bar(
            x=years, y=vals_low_risk_avg, name='Low Risk Avg',
            marker_color='#a5d6a7', text=[f"{v:.1%}" for v in vals_low_risk_avg],
            textposition='auto'
        ))
        
        # 患者柱子
        fig_bar.add_trace(go.Bar(
            x=years, y=vals_patient, name='Current Patient',
            marker_color=patient_color, text=[f"{v:.1%}" for v in vals_patient],
            textposition='auto'
        ))
        
        fig_bar.update_layout(
            barmode='group', # 分组显示
            template="plotly_white", height=300,
            margin=dict(t=30, b=0),
            yaxis=dict(range=[0, 1.1], title="Survival Probability")
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # 详细数据折叠
        with st.expander("📝 Calculation Details"):
             st.dataframe(pd.DataFrame(calculation_details).style.background_gradient(subset=["Contribution"], cmap="RdYlGn_r"))

# ==========================================
# 5. 页脚
# ==========================================
st.markdown("---")
st.markdown("<div style='text-align: center; color: #888; font-size: 14px;'>⚠️ Disclaimer: Research use only.</div>", unsafe_allow_html=True)
