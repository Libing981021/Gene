import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# ==========================================
# 1. 核心模型参数
# ==========================================
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
# 2. 页面配置
# ==========================================
st.set_page_config(
    page_title="CRC Metastasis Predictor",
    page_icon="🧬",
    layout="wide" # 宽屏模式，模仿参考图的大气布局
)

# ==========================================
# 3. 侧边栏：输入区域 (模仿参考图左侧)
# ==========================================
with st.sidebar:
    st.header("Input Feature Values")
    st.caption("Enter Log2 transformed expression values")
    
    st.markdown("---")
    
    # 3.1 内参基因输入
    st.markdown("#### **Reference Gene**")
    val_ref = st.number_input(
        f"{REF_GENE} (Internal Control)", 
        value=6.90, 
        step=0.1,
        help="内参基因用于标准化"
    )

    # 3.2 风险基因输入
    st.markdown("#### **Target Genes**")
    
    inputs = {}
    # 遍历生成输入框
    for gene in COEFFICIENTS.keys():
        inputs[gene] = st.number_input(
            f"{gene}", 
            value=10.0, 
            step=0.1
        )

# ==========================================
# 4. 主界面：标题与说明 (模仿参考图右侧)
# ==========================================

# 4.1 标题和简介
st.title("Predicting CRC Recurrence Risk Using a 6-Gene Signature")

st.markdown("""
This app predicts the likelihood of postoperative recurrence in Stage II/III Colorectal Cancer (CRC) based on a specific gene expression profile.

The model calculates a risk score using the following features:
""")

# 4.2 特征说明列表
st.markdown(f"""
* **{REF_GENE}**: The internal reference gene used for data normalization ($\Delta Log2$).
* **Target Genes**: A panel of 6 genes (TCEAL4, ACTR3B, ORAI3, PRIM1, LEMD1, INHBB) identified as prognostic markers.
* **Risk Score**: Calculated using LASSO-derived coefficients.
* **Probability**: The likelihood of high-risk recurrence based on the cutoff value (**{CUTOFF_VALUE}**).
""")

st.markdown("Input the relevant gene expression values in the **sidebar** to obtain predictions and risk stratification.")

# 4.3 预测按钮
# 使用一点空行让按钮和文字分开
st.write("") 
predict_btn = st.button("Predict Risk", type="primary")

# ==========================================
# 5. 计算与结果展示
# ==========================================
if predict_btn:
    st.markdown("---")
    
    # --- 计算逻辑 ---
    risk_score = 0
    details = []
    
    for gene, coef in COEFFICIENTS.items():
        norm_expr = inputs[gene] - val_ref
        contribution = norm_expr * coef
        risk_score += contribution
        
        details.append({
            "Gene": gene,
            "Norm Value": norm_expr,
            "Contribution": contribution
        })
    
    # --- 判定结果 ---
    is_high_risk = risk_score > CUTOFF_VALUE
    risk_level = "High Risk (高风险)" if is_high_risk else "Low Risk (低风险)"
    risk_color = "#ff4b4b" if is_high_risk else "#09ab3b"
    
    # --- 布局：左侧显示结论卡片，右侧显示可视化图 ---
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.subheader("Prediction Result")
        st.info(f"Risk Score: **{risk_score:.4f}**")
        
        # 结果卡片
        st.markdown(f"""
        <div style="
            background-color: {'rgba(255, 75, 75, 0.1)' if is_high_risk else 'rgba(9, 171, 59, 0.1)'};
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid {risk_color};
            margin-bottom: 20px;
        ">
            <h3 style="margin:0; color: {risk_color};">{risk_level}</h3>
            <p style="margin:5px 0 0 0; color: #666;">Cutoff: {CUTOFF_VALUE}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 临床建议
        if is_high_risk:
            st.warning("**Recommendation:** Consider more aggressive adjuvant chemotherapy and shorter follow-up intervals.")
        else:
            st.success("**Recommendation:** Standard follow-up plan is recommended.")

    with col2:
        st.subheader("Gene Contribution Analysis")
        # 准备绘图数据
        df_details = pd.DataFrame(details)
        df_details['Color'] = df_details['Contribution'].apply(lambda x: '#ff4b4b' if x > 0 else '#09ab3b')
        df_details = df_details.sort_values(by="Contribution", ascending=True)

        # 绘制条形图
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=df_details['Gene'],
            x=df_details['Contribution'],
            orientation='h',
            marker=dict(color=df_details['Color']),
            text=[f"{v:.3f}" for v in df_details['Contribution']],
            textposition='auto'
        ))
        
        fig.update_layout(
            height=350,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis_title="Contribution to Risk Score",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
