import streamlit as st
import pandas as pd
import numpy as np

# ==========================================
# 1. 核心模型参数 (已根据您的 R 运行结果自动填入)
# ==========================================
COEFFICIENTS = {
    "TCEAL4": 0.3364594,
    "ACTR3B": -0.4104630,
    "ORAI3":  0.2523666,
    "PRIM1":  -0.2529674,
    "LEMD1":  0.2133200,
    "INHBB":  0.1491095
}

# 训练集确定的固定截断值
CUTOFF_VALUE = 0.5739
REF_GENE = "EMC7"

# ==========================================
# 2. 页面配置
# ==========================================
st.set_page_config(
    page_title="CRC Prognostic Tool",
    page_icon="🧬",
    layout="centered"
)

# 侧边栏信息
with st.sidebar:
    st.header("关于模型 (About Model)")
    st.info(
        f"""
        **基于 EMC7 内参的 6 基因结直肠癌预后模型**
        
        该工具用于预测 II-III 期结直肠癌患者的术后复发风险。
        
        - **核心算法**: LASSO + Stepwise Cox
        - **标准化**: $\Delta Log2$ (Target - {REF_GENE})
        - **截断值**: {CUTOFF_VALUE}
        """
    )
    st.markdown("---")
    st.write("**模型系数 (Coefficients):**")
    st.json(COEFFICIENTS)

# 主界面
st.title("🧬 CRC Recurrence Risk Predictor")
st.markdown("##### A Clinical Tool for Stage II/III Colorectal Cancer")
st.markdown("---")

# ==========================================
# 3. 输入面板
# ==========================================
st.subheader("1. 输入基因表达量 (Input Expression)")
st.caption("请输入 Log2 转化后的表达值 (如 Microarray Log2 Intensity 或 qPCR Ct值)")

# 使用表单组织输入，看起来更整洁
with st.form("prediction_form"):
    col_ref, col_space = st.columns([1, 0.1])
    with col_ref:
        st.markdown(f"**内参基因 ({REF_GENE})**")
        val_ref = st.number_input(f"{REF_GENE} Value", value=10.0, step=0.1, help="内参基因的表达量用于标准化")

    st.markdown("---")
    st.markdown("**风险基因 (Target Genes)**")
    
    # 分两列排列输入框
    c1, c2 = st.columns(2)
    inputs = {}
    
    # 前3个基因放左边，后3个放右边
    keys = list(COEFFICIENTS.keys())
    with c1:
        for gene in keys[:3]:
            inputs[gene] = st.number_input(f"{gene}", value=10.0, step=0.1)
    with c2:
        for gene in keys[3:]:
            inputs[gene] = st.number_input(f"{gene}", value=10.0, step=0.1)

    submitted = st.form_submit_button("🚀 开始计算 (Calculate Risk)", type="primary")

# ==========================================
# 4. 计算与结果展示
# ==========================================
if submitted:
    # 1. 计算逻辑
    risk_score = 0
    details = []
    
    for gene, coef in COEFFICIENTS.items():
        # 核心公式: (Target - Ref) * Coef
        norm_expr = inputs[gene] - val_ref
        contribution = norm_expr * coef
        risk_score += contribution
        
        details.append({
            "Gene": gene,
            "Raw Value": inputs[gene],
            "Norm Value (Gene - Ref)": round(norm_expr, 4),
            "Coefficient": coef,
            "Contribution": round(contribution, 4)
        })
    
    # 2. 判定风险
    risk_level = "High Risk (高风险)" if risk_score > CUTOFF_VALUE else "Low Risk (低风险)"
    risk_color = "red" if risk_score > CUTOFF_VALUE else "green"
    
    # 3. 展示结果
    st.markdown("### 📊 预测结果 (Results)")
    
    res_box = st.container()
    with res_box:
        c_res1, c_res2 = st.columns(2)
        
        with c_res1:
            st.metric(label="风险评分 (Risk Score)", value=f"{risk_score:.4f}", delta=f"Cutoff: {CUTOFF_VALUE}")
        
        with c_res2:
            st.markdown(f"""
            <div style="
                background-color: {'#ffebee' if risk_score > CUTOFF_VALUE else '#e8f5e9'};
                padding: 15px;
                border-radius: 10px;
                border: 1px solid {risk_color};
                text-align: center;
            ">
                <h3 style="color: {risk_color}; margin:0;">{risk_level}</h3>
            </div>
            """, unsafe_allow_html=True)

    # 4. 临床建议
    st.markdown("#### 💡 临床建议 (Recommendation)")
    if risk_score > CUTOFF_VALUE:
        st.warning(
            "该患者属于**高风险组**。根据回顾性队列研究结果，其术后复发风险显著高于平均水平。\n\n"
            "建议：\n"
            "- 考虑更积极的辅助化疗方案。\n"
            "- 缩短术后随访间隔（如每3个月一次）。"
        )
    else:
        st.success(
            "该患者属于**低风险组**。根据回顾性队列研究结果，其预后相对较好。\n\n"
            "建议：\n"
            "- 可维持标准随访计划。\n"
            "- 避免过度治疗，关注生活质量。"
        )

    # 5. 详细数据展开
    with st.expander("查看详细计算过程 (Details)"):
        st.dataframe(pd.DataFrame(details))
        st.caption(f"计算公式: RiskScore = Σ ( (Expression_Gene - Expression_{REF_GENE}) × Coefficient )")