import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ==========================================
# 1. 配置与参数
# ==========================================
st.set_page_config(
    page_title="CRC Recurrence Risk Predictor",
    page_icon="🧬",
    layout="wide"
)

# 模型系数 (Coefficients)
COEFFICIENTS = {
    "TCEAL4": 0.3364594,
    "ACTR3B": -0.4104630,
    "ORAI3":  0.2523666,
    "PRIM1":  -0.2529674,
    "LEMD1":  0.2133200,
    "INHBB":  0.1491095
}

CUTOFF_VALUE = 0.5739
REF_GENE = "EMC7"

# ==========================================
# 2. 侧边栏：数值输入 (Input Feature Values)
# ==========================================
with st.sidebar:
    st.header("Input Feature Values")
    st.caption("请输入 Log2 转化后的基因表达量")
    
    st.markdown("---")
    
    # 2.1 内参基因 (增加了 min_value 和 max_value 防止误操作)
    st.markdown(f"**Reference Gene ({REF_GENE})**")
    val_ref = st.number_input(
        f"{REF_GENE} Value", 
        value=6.90, 
        min_value=0.0,
        max_value=25.0,
        step=0.1,
        format="%.2f",
        help="内参基因用于标准化数据 (Normalizer)"
    )

    st.markdown("---")
    
    # 2.2 风险基因循环生成输入框
    st.markdown("**Target Genes Expression**")
    inputs = {}
    
    # 这里设置默认值为 10.0，模拟常见表达水平
    for gene in COEFFICIENTS.keys():
        inputs[gene] = st.number_input(
            f"{gene}", 
            value=10.00, 
            min_value=0.0,
            max_value=25.0,
            step=0.1,
            format="%.2f"
        )

# ==========================================
# 3. 主界面：标题与介绍
# ==========================================
st.title("Predicting CRC Recurrence Risk Using a 6-Gene Signature")
st.markdown("""
This application predicts the likelihood of postoperative recurrence in Stage II/III Colorectal Cancer based on gene expression profiles.
""")

# 顶部简要信息条
st.info(f"""
* **Model Type**: LASSO + Stepwise Cox Regression
* **Cutoff Value**: {CUTOFF_VALUE}
* **Standardization**: $\Delta Log2$ (Target - {REF_GENE})
""")

st.write("Input the relevant feature values in the sidebar to obtain predictions and probability estimates.")

st.write("") 
predict_btn = st.button("🚀 Predict Risk (开始预测)", type="primary")

# ==========================================
# 4. 计算逻辑与结果展示
# ==========================================
if predict_btn:
    st.markdown("---")
    
    # --- A. 计算风险评分 ---
    risk_score = 0
    for gene, coef in COEFFICIENTS.items():
        norm_expr = inputs[gene] - val_ref
        risk_score += norm_expr * coef
    
    # --- B. 判定风险等级 ---
    is_high_risk = risk_score > CUTOFF_VALUE
    risk_level = "High Risk (高风险)" if is_high_risk else "Low Risk (低风险)"
    
    # 定义颜色方案
    risk_color = "#d32f2f" if is_high_risk else "#388e3c" # 深红 vs 深绿
    bg_color = "rgba(211, 47, 47, 0.1)" if is_high_risk else "rgba(56, 142, 60, 0.1)"

    # --- C. 结果布局 (左右分栏) ---
    col_res, col_viz = st.columns([1, 1.5], gap="medium")

    # === 左侧栏：数值结果 & 仪表盘 & 临床建议 ===
    with col_res:
        st.subheader("Prediction Result")
        
        # 1. 结果卡片 (数值展示)
        st.markdown(f"""
        <div style="
            background-color: {bg_color};
            padding: 20px;
            border-radius: 10px;
            border: 2px solid {risk_color};
            text-align: center;
            margin-bottom: 20px;
        ">
            <p style="margin:0; color: #555; font-size: 14px;">Risk Score</p>
            <h1 style="margin:0; font-size: 3em; color: {risk_color};">{risk_score:.4f}</h1>
            <hr style="border-top: 1px solid {risk_color}; opacity: 0.3; margin: 10px 0;">
            <h3 style="margin:0; color: {risk_color};">{risk_level}</h3>
        </div>
        """, unsafe_allow_html=True)

        # 2. 仪表盘 (Gauge Chart) - 新增功能！
        st.markdown("**Risk Gauge (仪表盘)**")
        
        # 确定仪表盘的最大值 (为了美观，取 Cutoff 的 3 倍或者 5)
        max_gauge_val = max(5.0, risk_score + 1)
        
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+delta",
            value = risk_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            delta = {'reference': CUTOFF_VALUE, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
            gauge = {
                'axis': {'range': [None, max_gauge_val], 'tickwidth': 1},
                'bar': {'color': risk_color},
                'bgcolor': "white",
                'borderwidth': 1,
                'bordercolor': "#eee",
                'steps': [
                    {'range': [0, CUTOFF_VALUE], 'color': "rgba(56, 142, 60, 0.15)"}, # 绿色区域
                    {'range': [CUTOFF_VALUE, max_gauge_val], 'color': "rgba(211, 47, 47, 0.15)"} # 红色区域
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 3},
                    'thickness': 0.8,
                    'value': CUTOFF_VALUE
                }
            }
        ))
        fig_gauge.update_layout(height=200, margin=dict(l=20, r=20, t=10, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # 3. 临床建议
        st.markdown("#### 💡 Clinical Recommendation")
        if is_high_risk:
            st.warning(
                "**High Risk Strategy:**\n\n"
                "1. **Adjuvant Therapy**: Consider aggressive chemotherapy (e.g., oxaliplatin-based).\n"
                "2. **Follow-up**: Shorten intervals (e.g., CT/CEA every 3 months).\n"
                "3. **Genetics**: Check MSI/MMR status."
            )
        else:
            st.success(
                "**Low Risk Strategy:**\n\n"
                "1. **Standard Care**: Maintain standard follow-up intervals.\n"
                "2. **QoL**: Focus on quality of life and avoid overtreatment.\n"
                "3. **Checkup**: Regular checkups every 6 months."
            )

    # === 右侧栏：生存曲线 ===
    with col_viz:
        st.subheader("Predicted Survival Analysis")
        st.caption("Simulation based on risk group stratification")
        
        # --- 模拟生存数据 ---
        time_points = np.linspace(0, 60, 100) # 60个月
        
        # 模拟曲线数学公式
        surv_low = np.exp(-0.005 * time_points)  
        surv_high = np.exp(-0.025 * time_points) 
        
        # 确定当前患者属于哪条线
        patient_curve = surv_high if is_high_risk else surv_low
        curve_color = risk_color
        
        # 绘图
        fig_surv = go.Figure()
        
        # 1. 低风险背景线 (虚线)
        fig_surv.add_trace(go.Scatter(
            x=time_points, y=surv_low,
            mode='lines', name='Low Risk Group',
            line=dict(color='green', width=1, dash='dash'),
            opacity=0.4
        ))
        
        # 2. 高风险背景线 (虚线)
        fig_surv.add_trace(go.Scatter(
            x=time_points, y=surv_high,
            mode='lines', name='High Risk Group',
            line=dict(color='red', width=1, dash='dash'),
            opacity=0.4
        ))

        # 3. 当前患者预测线 (实线 + 半透明填充)
        # 优化点：fillcolor 使用 rgba 设置透明度
        fill_color_rgba = "rgba(211, 47, 47, 0.1)" if is_high_risk else "rgba(56, 142, 60, 0.1)"
        
        fig_surv.add_trace(go.Scatter(
            x=time_points, y=patient_curve,
            mode='lines',
            name='Current Patient',
            line=dict(color=curve_color, width=3),
            fill='tozeroy', 
            fillcolor=fill_color_rgba # <--- 关键修改：半透明填充
        ))

        fig_surv.update_layout(
            title="Recurrence-Free Survival (RFS) Probability",
            xaxis_title="Time (Months)",
            yaxis_title="Survival Probability",
            yaxis_range=[0, 1.05],
            template="plotly_white",
            height=500, # 稍微调高一点，看起来更大气
            hovermode="x unified",
            legend=dict(orientation="h", y=1.02, yanchor="bottom", x=1, xanchor="right")
        )
        
        st.plotly_chart(fig_surv, use_container_width=True)

# ==========================================
# 5. 页脚免责声明 (Footer)
# ==========================================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #888; font-size: 12px;'>
    ⚠️ <b>Disclaimer:</b> This tool is intended for <b>research purposes only</b> and should not be used as the sole basis for clinical decision-making. 
    The predictions should be interpreted by qualified healthcare professionals in conjunction with other clinical findings.
    <br>
    © 2026 CRC Research Group. All Rights Reserved.
    </div>
    """, 
    unsafe_allow_html=True
)
