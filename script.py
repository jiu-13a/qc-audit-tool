import streamlit as st
import pandas as pd
import os
import re
from openai import OpenAI

# --- 1. 配置 ---
# 请填入您真实的 DeepSeek API Key
client = OpenAI(
    api_key=st.secrets["deepseek_key"],
    base_url="https://api.deepseek.com"
)

CODE_FILE = "code.csv"
CALC_FILE = "calculate.csv"
EXP_FILE = "experience.csv"
CAT_FILE = "category.csv"  # 新增的人员对应表

# 初始化精简版经验库 (仅保留范围和代码)
if not os.path.exists(EXP_FILE):
    pd.DataFrame(columns=["范围", "代码"]).to_csv(EXP_FILE, index=False, encoding='utf-8-sig')

# 初始化 Session State
if 'calc_done' not in st.session_state:
    st.session_state.calc_done = False
if 'val1' not in st.session_state:
    st.session_state.val1 = 0.0
if 'val2' not in st.session_state:
    st.session_state.val2 = 0.0
if 'ai_ans' not in st.session_state:
    st.session_state.ai_ans = ""
if 'detected_code' not in st.session_state:
    st.session_state.detected_code = ""


# --- 2. 核心逻辑函数 ---

def smart_read_csv(file_path, **kwargs):
    """尝试多种编码读取CSV"""
    try:
        return pd.read_csv(file_path, encoding='utf-8-sig', **kwargs)
    except:
        return pd.read_csv(file_path, encoding='gbk', **kwargs)


def calculate_logic(people_val, q_risk, e_risk, s_risk, df_calc):
    """计算各体系基础人日"""
    try:
        df_calc.iloc[:, 0] = df_calc.iloc[:, 0].astype(str).str.strip()
        search_val = str(people_val).strip()
        row = df_calc[df_calc.iloc[:, 0] == search_val]
        if row.empty: return 0, 0, 0
        risk_map = {"高": 0, "中": 1, "低": 2}
        q = float(row.iloc[0, 1 + risk_map[q_risk]])
        e = float(row.iloc[0, 4 + risk_map[e_risk]])
        s = float(row.iloc[0, 7 + risk_map[s_risk]])
        return q, e, s
    except Exception as e:
        st.error(f"人日提取出错: {e}")
        return 0, 0, 0


def get_auditors(code_str, df_cat):
    """根据代码前两位查询人员"""
    try:
        # 提取前两位数字
        match = re.search(r'^(\d{1,2})', str(code_str).strip())
        if not match: return None
        major_category = int(match.group(1))

        # 在category表中查找 (假设第一列是'审核代码大类')
        row = df_cat[df_cat.iloc[:, 0].astype(str).str.strip() == str(major_category)]
        if row.empty: return None

        return {
            "QMS": row.iloc[0, 1],
            "EMS": row.iloc[0, 2],
            "OHSMS": row.iloc[0, 3]
        }
    except:
        return None


# --- 3. 界面逻辑 ---
st.set_page_config(page_title="启辰认证-合同评审工具", layout="wide")
st.title("🛡️ 启辰认证-合同评审自动化处理系统")

if all(os.path.exists(f) for f in [CODE_FILE, CALC_FILE, CAT_FILE]):
    df_code = smart_read_csv(CODE_FILE)
    df_calc = smart_read_csv(CALC_FILE, header=1)
    df_exp = smart_read_csv(EXP_FILE)
    df_cat = smart_read_csv(CAT_FILE)

    # ==========================================
    # 模块一：确认人日数目
    # ==========================================
    st.header("📊 模块一：确认人日数目")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        people_list = df_calc.iloc[:, 0].dropna().unique().tolist()
        sel_people = st.selectbox("受审核人数", options=people_list)
    with c2:
        q_r = st.selectbox("QMS风险", ["高", "中", "低"], index=2)
    with c3:
        e_r = st.selectbox("EMS风险", ["高", "中", "低"], index=2)
    with c4:
        s_r = st.selectbox("OHSMS风险", ["高", "中", "低"], index=2)

    if st.button("🧮 计算人日数"):
        q, e, s = calculate_logic(sel_people, q_r, e_r, s_r, df_calc)
        base_sum = q + e + s
        st.session_state.val1 = base_sum * 0.8
        st.session_state.val2 = base_sum * 0.8 * 0.8
        st.session_state.calc_done = True

    if st.session_state.calc_done:
        m_col1, m_col2 = st.columns(2)
        m_col1.metric("数值一：(Q+E+S) × 0.8", f"{st.session_state.val1:.3f}")
        m_col2.metric("数值二：(Q+E+S) × 0.8 × 0.8", f"{st.session_state.val2:.3f}")

    st.divider()

    # ==========================================
    # 模块二：范围判定与人员查询 (集成)
    # ==========================================
    st.header("🤖 模块二：范围判定与人员查询")
    scope = st.text_area("输入受审核方的范围描述：", height=100)

    if st.button("🚀 开始全量检索判定"):
        if not scope.strip():
            st.warning("请输入范围。")
        else:
            # 【核心改变】：将整个代码文件转为文本喂给 AI
            # 仅选取核心列以节省 Token 并提高精准度
            code_library_text = df_code.to_string(index=False)
            exp_context = df_exp.tail(10).to_string(index=False) if not df_exp.empty else "暂无历史经验"

            prompt = f"""
你现在是【北京北方启辰认证服务有限公司】的技术评审专家。
你的任务是：从下方提供的【全量代码库】中检索出与《待评审范围》最匹配的6位小类代码。

### 待评审范围：
{scope}

### 历史判定参考：
{exp_context}

### 全量代码库（请逐行检索）：
{code_library_text}

### 判定要求：
1. **唯一性**：必须且只能从上方的代码库中选择一个真实存在的代码。
2. **禁止幻觉**：严禁发明任何不在库中的代码（如 33.02.00 这种）。
3. **逻辑**：先分析业务属性，再对比包含项与排除项。
4. **输出格式**：最后一行必须严格遵守“最终代码：XX.XX.XX”的格式。

请开始深度分析并给出判定：
"""
            with st.spinner("DeepSeek 正在全文扫描代码库..."):
                try:
                    res = client.chat.completions.create(
                        model="deepseek-v4-pro",
                        messages=[{"role": "system", "content": "你是一个严谨的认证代码判定机器人，只根据提供的知识库回答。"},
                                  {"role": "user", "content": prompt}],
                        temperature=0.1,
                        reasoning_effort="high"
                    )
                    ai_ans = res.choices[0].message.content
                    st.session_state.ai_ans = ai_ans
                    
                    # 提取代码
                    code_match = re.search(r"最终代码：\s*([0-9.]+)", ai_ans)
                    if code_match:
                        st.session_state.detected_code = code_match.group(1).strip('.')
                    else:
                        st.session_state.detected_code = ""
                except Exception as e:
                    st.error(f"AI 调用失败: {e}")

    if st.session_state.ai_ans:
        st.info(st.session_state.ai_ans)

        # 自动显示人员信息
        if st.session_state.detected_code:
            st.subheader(f"👤 代码 {st.session_state.detected_code} 对应评审人员")
            auditors = get_auditors(st.session_state.detected_code, df_cat)
            if auditors:
                a1, a2, a3 = st.columns(3)
                a1.success(f"**QMS 审核员**\n\n{auditors['QMS']}")
                a2.success(f"**EMS 审核员**\n\n{auditors['EMS']}")
                a3.success(f"**OHSMS 审核员**\n\n{auditors['OHSMS']}")
            else:
                st.warning("未在 category 表中找到对应大类的人员信息。")

    st.divider()

    # ==========================================
    # 模块三：结果确认与经验库同步
    # ==========================================
    st.header("💾 模块三：同步至经验库")
    f_code = st.text_input("确认最终代码：", value=st.session_state.detected_code)

    if st.button("✅ 仅同步范围与代码"):
        if f_code and scope.strip():
            new_log = pd.DataFrame([[scope.replace('\n', ' '), f_code]], columns=["范围", "代码"])
            new_log.to_csv(EXP_FILE, mode='a', header=False, index=False, encoding='utf-8-sig')
            st.success("经验库已更新！")
            st.balloons()
        else:
            st.error("请确保范围和代码均已填写。")
else:
    st.error("缺失必要文件：请确保 code.csv, calculate.csv, category.csv 均在目录下。")
