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

# 训练集确定的固定截断值
CUTOFF_VALUE = 0.5739
REF_GENE = "EMC7"

# ==========================================
# 2. 页面配置
# ==========================================
st.set_page_config(
    page_title="CRC Risk Predictor",
    page_icon="🧬",
    layout="wide", # 使用宽屏模式，让可视化图表展示更舒服
    initial_sidebar_state="expanded"
)

# 侧边栏样式优化
with st.sidebar:
    st.header("关于模型 (About Model)")
    st.info(
        f"""
        **基于 {REF_GENE} 内参的 6 基因结直肠癌预后模型**
        
        用于预测 II-III 期结直肠癌患者的术后复发风险。
        
        - **核心算法**: LASSO + Stepwise Cox
        - **标准化**: $\Delta Log2$ (Target - {REF_GENE})
        - **High/Low 截断值**: `{CUTOFF_VALUE}`
        """
    )
    st.markdown("---")
    with st.expander("查看模型系数 (Coefficients)"):
        st.json(COEFFICIENTS)
    
    st.caption("Designed for Clinical Research Use Only.")

# 主标题区域
st.title("🧬 CRC Recurrence Risk Predictor")
st.markdown("##### A Clinical Tool for Stage II/III Colorectal Cancer")
st.markdown("---")

# ==========================================
# 3. 输入面板 (分两栏布局)
# ==========================================
col_input, col_result_placeholder = st.columns([1, 1.5], gap="large")

with col_input:
    st.subheader("1. 输入基因表达量")
    st.caption("请输入 Log2 转化后的表达值 (Input Log2 Expression)")

    with st.form("prediction_form"):
        st.markdown(f"**🟢 内参基因 ({REF_GENE})**")
        val_ref = st.number_input(
            f"{REF_GENE} Value", 
            value=6.90, 
            step=0.1, 
            help="内参基因用于标准化其他基因的表达量"
        )

        st.markdown("---")
        st.markdown("**🔴 风险基因 (Target Genes)**")
        
        # 使用 Grid 布局让输入框更紧凑
        cols = st.columns(2)
        inputs = {}
        keys = list(COEFFICIENTS.keys())
        
        # 遍历生成输入框
        for i, gene in enumerate(keys):
            col_idx = i % 2
            with cols[col_idx]:
                inputs[gene] = st.number_input(f"{gene}", value=10.0, step=0.1)

        st.markdown("---")
        submitted = st.form_submit_button("🚀 开始计算 (Calculate Risk)", type="primary", use_container_width=True)

# ==========================================
# 4. 计算与可视化逻辑
# ==========================================
if submitted:
    # --- 1. 计算逻辑 ---
    risk_score = 0
    details = []
    
    for gene, coef in COEFFICIENTS.items():
        norm_expr = inputs[gene] - val_ref
        contribution = norm_expr * coef
        risk_score += contribution
        
        details.append({
            "Gene": gene,
            "Raw Value": inputs[gene],
            "Norm Value": norm_expr,
            "Coefficient": coef,
            "Contribution": contribution
        })
    
    df_details = pd.DataFrame(details)
    
    # 判定风险
    is_high_risk = risk_score > CUTOFF_VALUE
    risk_level = "High Risk (高风险)" if is_high_risk else "Low Risk (低风险)"
    risk_color = "#ff4b4b" if is_high_risk else "#09ab3b" # Streamlit 标准红/绿
    bg_color = "rgba(255, 75, 75, 0.1)" if is_high_risk else "rgba(9, 171, 59, 0.1)"

    # --- 2. 结果展示 (在右侧栏) ---
    with col_result_placeholder:
        st.subheader("2. 预测结果与分析")
        
        # 结果卡片
        st.markdown(f"""
        <div style="
            background-color: {bg_color};
            padding: 20px;
            border-radius: 10px;
            border: 2px solid {risk_color};
            text-align: center;
            margin-bottom: 20px;
        ">
            <h4 style="margin:0; color: #555;">Risk Score</h4>
            <h1 style="margin:0; font-size: 3em; color: {risk_color};">{risk_score:.4f}</h1>
            <p style="margin:0; color: #666;">Cutoff: {CUTOFF_VALUE}</p>
            <hr style="border-color: {risk_color}; opacity: 0.3;">
            <h2 style="margin:0; color: {risk_color};">{risk_level}</h2>
        </div>
        """, unsafe_allow_html=True)

        # --- 3. 可视化图表 (新增功能) ---
        st.markdown("**📊 基因风险贡献图 (Gene Contribution Analysis)**")
        st.caption("展示每个基因对最终评分的贡献度 (Coef × Normalized Expression)")

        # 准备绘图数据
        df_details['Color'] = df_details['Contribution'].apply(lambda x: '#ff4b4b' if x > 0 else '#09ab3b')
        df_details = df_details.sort_values(by="Contribution", ascending=True)

        # 使用 Plotly 绘制条形图
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=df_details['Gene'],
            x=df_details['Contribution'],
            orientation='h',
            marker=dict(color=df_details['Color'], opacity=0.8),
            text=[f"{val:.3f}" for val in df_details['Contribution']],
            textposition='auto',
            hovertemplate='<b>%{y}</b><br>贡献值: %{x:.4f}<br>原始表达: %{customdata[0]}<extra></extra>',
            customdata=df_details[['Raw Value']]
        ))

        fig.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=20, b=20),
            xaxis_title="Risk Contribution Score",
            yaxis_title=None,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=True, gridcolor='#eee'),
        )
        # 添加一条 0 轴参考线
        fig.add_vline(x=0, line_width=2, line_color="black")
        
        st.plotly_chart(fig, use_container_width=True)

        # --- 4. 临床建议 ---
        with st.expander("💡 查看临床建议 (Clinical Recommendation)", expanded=True):
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

    # 底部显示详细数据表格
    st.markdown("---")
    st.markdown("#### 📋 详细计算数据 (Detail Table)")
    st.dataframe(
        df_details.sort_values("Gene").style.format({
            "Raw Value": "{:.2f}",
            "Norm Value": "{:.4f}",
            "Coefficient": "{:.4f}",
            "Contribution": "{:.4f}"
        }), 
        use_container_width=True
    )
