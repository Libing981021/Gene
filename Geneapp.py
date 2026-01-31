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
# 2. 侧边栏：数值输入
# ==========================================
with st.sidebar:
    st.header("Input Feature Values")
    st.caption("请输入 Log2 转化后的基因表达量")
    
    st.markdown("---")
    
    # 2.1 内参基因
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
    
    # 2.2 风险基因
    st.markdown("**Target Genes Expression**")
    inputs = {}
    
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
st.markdown("This application predicts the likelihood of postoperative recurrence in Stage II/III Colorectal Cancer.")

st.info(f"""
* **Model Type**: LASSO + Stepwise Cox Regression
* **Cutoff Value**: {CUTOFF_VALUE}
* **Standardization**: $\Delta Log2$ (Target - {REF_GENE})
""")

st.write("Input the relevant feature values in the sidebar to obtain predictions.")
st.write("") 
predict_btn = st.button("🚀 Predict Risk (开始预测)", type="primary")

# ==========================================
# 4. 计算逻辑与结果展示
# ==========================================
if predict_btn:
    st.markdown("---")
    
    # --- A. 计算过程 ---
    risk_score = 0
    calculation_details = [] # 用于存储详细计算步骤
    
    for gene, coef in COEFFICIENTS.items():
        raw_val = inputs[gene]
        # 核心公式：归一化值 = 目标基因 - 内参基因
        norm_expr = raw_val - val_ref
        contribution = norm_expr * coef
        risk_score += contribution
        
        # 收集数据用于展示
        calculation_details.append({
            "Gene": gene,
            "Raw Value (Log2)": raw_val,
            "Ref Value": val_ref,
            "Norm Value (ΔLog2)": norm_expr,
            "Coefficient": coef,
            "Contribution": contribution
        })
    
    # --- B. 判定风险 ---
    is_high_risk = risk_score > CUTOFF_VALUE
    risk_level = "High Risk (高风险)" if is_high_risk else "Low Risk (低风险)"
    risk_color = "#d32f2f" if is_high_risk else "#388e3c"
    bg_color = "rgba(211, 47, 47, 0.1)" if is_high_risk else "rgba(56, 142, 60, 0.1)"

    # --- C. 结果布局 (两列) ---
    col_res, col_viz = st.columns([1, 1.5], gap="medium")

    # === 左列：结果与建议 ===
    with col_res:
        st.subheader("Prediction Result")
        
        # 结果卡片
        st.markdown(f"""
        <div style="background-color: {bg_color}; padding: 20px; border-radius: 10px; border: 2px solid {risk_color}; text-align: center; margin-bottom: 20px;">
            <p style="margin:0; color: #555;">Risk Score</p>
            <h1 style="margin:0; font-size: 3em; color: {risk_color};">{risk_score:.4f}</h1>
            <hr style="border-top: 1px solid {risk_color}; opacity: 0.3; margin: 10px 0;">
            <h3 style="margin:0; color: {risk_color};">{risk_level}</h3>
        </div>
        """, unsafe_allow_html=True)

        # 仪表盘
        max_gauge_val = max(5.0, risk_score + 1)
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+delta", value = risk_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            delta = {'reference': CUTOFF_VALUE, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
            gauge = {
                'axis': {'range': [None, max_gauge_val]},
                'bar': {'color': risk_color},
                'steps': [
                    {'range': [0, CUTOFF_VALUE], 'color': "rgba(56, 142, 60, 0.15)"},
                    {'range': [CUTOFF_VALUE, max_gauge_val], 'color': "rgba(211, 47, 47, 0.15)"}
                ],
                'threshold': {'line': {'color': "red", 'width': 3}, 'thickness': 0.8, 'value': CUTOFF_VALUE}
            }
        ))
        fig_gauge.update_layout(height=200, margin=dict(l=20, r=20, t=10, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # 临床建议
        st.markdown("#### 💡 Clinical Recommendation")
        if is_high_risk:
            st.warning("**High Risk Strategy:**\n\n1. Consider aggressive chemotherapy.\n2. Shorten follow-up intervals.\n3. Check MSI/MMR status.")
        else:
            st.success("**Low Risk Strategy:**\n\n1. Maintain standard follow-up.\n2. Avoid overtreatment.\n3. Regular checkups every 6 months.")

    # === 右列：生存曲线 ===
    with col_viz:
        st.subheader("Predicted Survival Analysis")
        st.caption("Simulation based on risk group stratification")
        
        # 模拟数据
        time_points = np.linspace(0, 60, 100)
        surv_low = np.exp(-0.005 * time_points)
        surv_high = np.exp(-0.025 * time_points)
        patient_curve = surv_high if is_high_risk else surv_low
        fill_color_rgba = "rgba(211, 47, 47, 0.1)" if is_high_risk else "rgba(56, 142, 60, 0.1)"
        
        fig_surv = go.Figure()
        fig_surv.add_trace(go.Scatter(x=time_points, y=surv_low, mode='lines', name='Low Risk Group', line=dict(color='green', dash='dash'), opacity=0.4))
        fig_surv.add_trace(go.Scatter(x=time_points, y=surv_high, mode='lines', name='High Risk Group', line=dict(color='red', dash='dash'), opacity=0.4))
        fig_surv.add_trace(go.Scatter(
            x=time_points, y=patient_curve, mode='lines', name='Current Patient',
            line=dict(color=risk_color, width=3), fill='tozeroy', fillcolor=fill_color_rgba
        ))

        fig_surv.update_layout(
            title="Recurrence-Free Survival (RFS)", xaxis_title="Time (Months)", yaxis_title="Probability",
            yaxis_range=[0, 1.05], template="plotly_white", height=450, hovermode="x unified",
            legend=dict(orientation="h", y=1.02, x=1, xanchor="right")
        )
        st.plotly_chart(fig_surv, use_container_width=True)

    # ==========================================
    # 5. 新增：详细计算过程 (Expandable Section)
    # ==========================================
    st.markdown("---")
    with st.expander("📝 查看详细计算过程 (Calculation Details & Formula)", expanded=False):
        st.markdown("#### 1. 计算公式 (Formula)")
        st.latex(r"""
        RiskScore = \sum_{i=1}^{n} \left[ (Expression_{Gene_i} - Expression_{Ref}) \times Coefficient_i \right]
        """)
        
        st.markdown("#### 2. 数据明细 (Data Table)")
        st.write(f"**内参基因 ({REF_GENE}) 值:** `{val_ref:.2f}`")
        
        # 创建 DataFrame
        df_details = pd.DataFrame(calculation_details)
        
        # 格式化显示（保留4位小数，增加颜色）
        # 我们使用 Pandas Style 来给 Contribution 列加颜色条，直观显示正负贡献
        st.dataframe(
            df_details.style
            .format("{:.4f}", subset=["Raw Value (Log2)", "Ref Value", "Norm Value (ΔLog2)", "Coefficient", "Contribution"])
            .background_gradient(subset=["Contribution"], cmap="RdYlGn_r", vmin=-0.5, vmax=0.5),
            use_container_width=True
        )
        
        st.caption("""
        * **Norm Value**: The normalized expression ($\Delta Log2$).
        * **Contribution**: The impact of this gene on the final risk score. (Red = Increases Risk, Green = Decreases Risk).
        """)

# ==========================================
# 6. 页脚
# ==========================================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #888; font-size: 12px;'>
    ⚠️ <b>Disclaimer:</b> Research use only. Not for clinical diagnosis.
    <br>© 2026 CRC Research Group.
    </div>
    """, 
    unsafe_allow_html=True
)
