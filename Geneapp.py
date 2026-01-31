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

# 模型系数
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
    
    # 2.1 内参基因
    st.markdown(f"**Reference Gene ({REF_GENE})**")
    val_ref = st.number_input(
        f"{REF_GENE} Value", 
        value=6.90, 
        step=0.1,
        format="%.2f",
        help="内参基因用于标准化数据"
    )

    st.markdown("---")
    
    # 2.2 风险基因循环生成输入框
    st.markdown("**Target Genes Expression**")
    inputs = {}
    
    # 为了布局好看，如果你想让输入框紧凑一点，可以不做分列，直接垂直排列
    # 这里完全模仿左侧栏的样式
    for gene in COEFFICIENTS.keys():
        inputs[gene] = st.number_input(
            f"{gene}", 
            value=10.00, 
            step=0.1,
            format="%.2f"
        )

# ==========================================
# 3. 主界面区域
# ==========================================

# 3.1 标题与介绍
st.title("Predicting CRC Recurrence Risk Using a 6-Gene Signature")
st.markdown("""
This application predicts the likelihood of postoperative recurrence in Stage II/III Colorectal Cancer based on gene expression profiles.
""")

st.info(f"""
* **Model Type**: LASSO + Stepwise Cox Regression
* **Cutoff Value**: {CUTOFF_VALUE}
* **Standardization**: $\Delta Log2$ (Target - {REF_GENE})
""")

st.write("Input the relevant feature values in the sidebar to obtain predictions and probability estimates.")

# 3.2 预测按钮
st.write("") # 增加一点间距
predict_btn = st.button("Predict Risk (开始预测)", type="primary")

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
    risk_color = "#d32f2f" if is_high_risk else "#388e3c" # 深红 vs 深绿
    bg_color = "rgba(211, 47, 47, 0.1)" if is_high_risk else "rgba(56, 142, 60, 0.1)"

    # --- C. 结果布局 ---
    col_res, col_viz = st.columns([1, 1.5])

    # 左列：数值结果 & 临床建议
    with col_res:
        st.subheader("Prediction Result")
        
        # 结果卡片
        st.markdown(f"""
        <div style="
            background-color: {bg_color};
            padding: 20px;
            border-radius: 8px;
            border-left: 6px solid {risk_color};
            margin-bottom: 20px;
        ">
            <p style="margin:0; color: #555; font-size: 0.9em;">Risk Score</p>
            <h2 style="margin:0; color: {risk_color};">{risk_score:.4f}</h2>
            <hr style="border-top: 1px solid {risk_color}; opacity: 0.3; margin: 10px 0;">
            <strong style="color: {risk_color}; font-size: 1.2em;">{risk_level}</strong>
        </div>
        """, unsafe_allow_html=True)

        # 临床建议 (根据你提供的图片内容)
        st.markdown("#### 💡 临床建议 (Clinical Recommendation)")
        if is_high_risk:
            st.warning(
                "**建议方案 (High Risk Strategy):**\n\n"
                "1. **辅助治疗**: 建议考虑更积极的辅助化疗方案（如 oxaliplatin-based）。\n"
                "2. **随访监测**: 建议缩短术后随访间隔（如每 3 个月一次 CT/CEA 检测）。\n"
                "3. **基因检测**: 建议进行 MSI/MMR 状态及其他驱动基因检测。"
            )
        else:
            st.success(
                "**建议方案 (Low Risk Strategy):**\n\n"
                "1. **常规护理**: 可维持标准临床随访计划。\n"
                "2. **生活质量**: 避免过度医疗，关注患者术后生活质量。\n"
                "3. **定期复查**: 建议每 6 个月进行一次常规复查。"
            )

    # 右列：生存曲线 (模拟数据)
    with col_viz:
        st.subheader("Predicted Survival Curve (Simulation)")
        
        # --- 模拟生存数据 (仅用于展示效果) ---
        # 这里的数学公式仅为了生成形状正确的曲线，实际应用应替换为 Cox 模型的 baseline hazard
        time_points = np.linspace(0, 60, 100) # 0到60个月
        
        # 模拟：低风险组衰减慢，高风险组衰减快
        surv_low = np.exp(-0.005 * time_points)  
        surv_high = np.exp(-0.025 * time_points) 
        
        # 绘图
        fig = go.Figure()
        
        # 1. 绘制低风险背景线
        fig.add_trace(go.Scatter(
            x=time_points, y=surv_low,
            mode='lines',
            name='Low Risk Group (Avg)',
            line=dict(color='green', width=2, dash='dash' if is_high_risk else 'solid'),
            opacity=0.3 if is_high_risk else 1.0
        ))
        
        # 2. 绘制高风险背景线
        fig.add_trace(go.Scatter(
            x=time_points, y=surv_high,
            mode='lines',
            name='High Risk Group (Avg)',
            line=dict(color='red', width=2, dash='dash' if not is_high_risk else 'solid'),
            opacity=0.3 if not is_high_risk else 1.0
        ))

        # 3. 标记患者当前预测位置 (用散点表示该患者所属的曲线)
        patient_curve = surv_high if is_high_risk else surv_low
        patient_color = 'red' if is_high_risk else 'green'
        
        fig.add_trace(go.Scatter(
            x=time_points, y=patient_curve,
            mode='lines',
            name='Current Patient Prediction',
            line=dict(color=patient_color, width=4),
            fill='tozeroy', # 填充下方颜色，视觉效果更强
            fillcolor=f"rgba({'255,0,0' if is_high_risk else '0,255,0'}, 0.1)"
        ))

        fig.update_layout(
            title="Recurrence-Free Survival (RFS) Probability",
            xaxis_title="Time (Months)",
            yaxis_title="Survival Probability",
            yaxis_range=[0, 1.05],
            template="plotly_white",
            height=400,
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.caption("注：此生存曲线基于风险评分生成的示意图，仅供参考，不代表真实临床统计数据。")
