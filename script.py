import streamlit as st
import pandas as pd
import os
import re
from openai import OpenAI
from github import Github
from docx import Document
import easyocr
import numpy as np
from PIL import Image
from datetime import datetime

# ==========================================
# 1. 基础配置与初始化
# ==========================================
st.set_page_config(page_title="启辰认证-合同评审工具", layout="wide", page_icon="🛡️")

# DeepSeek 客户端初始化 (建议在线上部署时将 key 放入 st.secrets)
API_KEY = st.secrets["deepseek_key"] 
client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

# 初始化 Session State (防止按钮点击后刷新页面导致数据丢失)
if 'calc_done' not in st.session_state:
    st.session_state.calc_done = False
if 'q' not in st.session_state: st.session_state.q = 0
if 'e' not in st.session_state: st.session_state.e = 0
if 's' not in st.session_state: st.session_state.s = 0
if 'val1' not in st.session_state: st.session_state.val1 = 0.0
if 'val2' not in st.session_state: st.session_state.val2 = 0.0
if 'ai_ans' not in st.session_state: st.session_state.ai_ans = ""
if 'candidate_codes' not in st.session_state: st.session_state.candidate_codes = []

# ==========================================
# 2. 核心工具函数
# ==========================================
def get_actual_path(filename):
    """解决 Linux/Streamlit Cloud 环境下文件名大小写敏感的问题"""
    if os.path.exists(filename): return filename
    upper_file = filename.replace(".csv", ".CSV")
    if os.path.exists(upper_file): return upper_file
    return filename

def smart_read_csv(file_path, **kwargs):
    """智能读取 CSV，防止不同编码格式导致的乱码"""
    try:
        return pd.read_csv(file_path, encoding='utf-8-sig', **kwargs)
    except:
        return pd.read_csv(file_path, encoding='gbk', **kwargs)

def calculate_logic(people_val, q_risk, e_risk, s_risk, df_calc):
    """计算各体系的基础人日"""
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
        st.error(f"人日提取出错，请检查 calculate.csv 格式: {e}")
        return 0, 0, 0

def get_auditors(code_str, df_cat):
    """根据 AI 输出的代码匹配评审专家"""
    try:
        # 提取前 1-2 位数字作为大类 (如 17.02.01 -> 17)
        match = re.search(r'^(\d{1,2})', str(code_str).strip())
        if not match: return None
        major_cat_num = str(int(match.group(1))) # 转换为字符串且去前导0
        
        # 确保表的第一列按字符串对比
        df_cat.iloc[:, 0] = df_cat.iloc[:, 0].astype(str).str.strip()
        row = df_cat[df_cat.iloc[:, 0] == major_cat_num]
        
        if row.empty: return None
        
        # 处理空值为 "/"
        def fill_na(val):
            return val if pd.notna(val) and str(val).strip() != "" else "/"

        return {
            "Q": fill_na(row.iloc[0, 1]),
            "E": fill_na(row.iloc[0, 2]),
            "S": fill_na(row.iloc[0, 3])
        }
    except Exception:
        return None
def sync_to_github(new_scope, new_code):
    """通过 GitHub API 将新经验实时推送到仓库"""
    try:
        # 从 Secrets 读取 GitHub 配置
        token = st.secrets["github_token"]
        repo_name = st.secrets["github_repo"]
        file_path = "experience.csv"
        
        g = Github(token)
        repo = g.get_repo(repo_name)
        
        # 1. 获取 GitHub 仓库中该文件的当前内容
        file_content = repo.get_contents(file_path)
        old_data_raw = file_content.decoded_content.decode('utf-8-sig')
        
        # 2. 准备新行数据 (清理掉 scope 里的换行和逗号防止 CSV 格式错乱)
        clean_scope = str(new_scope).replace(',', '，').replace('\n', ' ')
        new_line = f"\n{clean_scope},{new_code}"
        
        # 3. 合并内容并提交回 GitHub
        updated_content = old_data_raw.strip() + new_line
        repo.update_file(
            path=file_path,
            message=f"System: 自动同步新经验 - {new_code}",
            content=updated_content,
            sha=file_content.sha
        )
        return True, "同步成功"
    except Exception as e:
        return False, str(e)

# ==========================================
# 3. 文件检查与加载
# ==========================================
CODE_FILE = get_actual_path("code.csv")
CALC_FILE = get_actual_path("calculate.csv")
CAT_FILE = get_actual_path("category.csv")
EXP_FILE = get_actual_path("experience.csv")

# 初始化精简版经验库 (若不存在则创建)
if not os.path.exists(EXP_FILE):
    pd.DataFrame(columns=["范围", "代码"]).to_csv(EXP_FILE, index=False, encoding='utf-8-sig')

# 检查核心文件是否齐备
if all(os.path.exists(f) for f in [CODE_FILE, CALC_FILE, CAT_FILE]):
    df_code = smart_read_csv(CODE_FILE)
    df_calc = smart_read_csv(CALC_FILE, header=1) # 第2行作为表头
    df_cat = smart_read_csv(CAT_FILE)
    df_exp = smart_read_csv(EXP_FILE)

    st.title("🛡️ 启辰认证-合同评审自动化处理系统")

    # ==========================================
    # 模块一：确认人日数目
    # ==========================================
    st.header("📊 模块一：确认人日数目")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        people_list = df_calc.iloc[:, 0].dropna().unique().tolist()
        sel_people = st.selectbox("受审核人数", options=people_list)
    with c2: q_r = st.selectbox("QMS风险", ["高", "中", "低"], index=2)
    with c3: e_r = st.selectbox("EMS风险", ["高", "中", "低"], index=2)
    with c4: s_r = st.selectbox("OHSMS风险", ["高", "中", "低"], index=2)

    if st.button("🧮 计算人日数"):
        q, e, s = calculate_logic(sel_people, q_r, e_r, s_r, df_calc)
        st.session_state.q, st.session_state.e, st.session_state.s = q, e, s
        base_sum = q + e + s
        
        # 按照新需求计算两个打折数值
        st.session_state.val1 = base_sum * 0.8
        st.session_state.val2 = base_sum * 0.8 * 0.8
        st.session_state.calc_done = True

    # 展示计算结果
    if st.session_state.calc_done:
        st.write(f"**提取到的各体系基础人日：** QMS: `{st.session_state.q}` | EMS: `{st.session_state.e}` | OHSMS: `{st.session_state.s}`")
        m_col1, m_col2 = st.columns(2)
        m_col1.metric("数值一：(Q+E+S) × 0.8", f"{st.session_state.val1:.3f}")
        m_col2.metric("数值二：(Q+E+S) × 0.8 × 0.8", f"{st.session_state.val2:.3f}")

    st.divider()

    # ==========================================
    # 模块二：全量代码检索与人员判定
    # ==========================================
    st.header("🤖 模块二：范围判定与专家匹配")
    scope = st.text_area("输入受审核方的范围描述：", height=120, placeholder="例如：电子元器件的生产、加工及销售...")

    if st.button("🚀 开始判定"):
        if not scope.strip():
            st.warning("请输入审核范围。")
        else:
            # --- 步骤 1：精确匹配拦截逻辑 ---
            # 预处理输入范围：去除首尾空格、去除换行符，统一对比标准
            clean_input = scope.strip().replace('\n', ' ')
            
            # 在经验库 df_exp 中查找（假设列名为 '范围' 和 '代码'）
            # 我们取最后一条匹配记录（代表最近的判定）
            exact_match = df_exp[df_exp['范围'].str.strip().replace('\n', ' ') == clean_input]

            if not exact_match.empty:
                # 情况 A：命中完全对应的经验，直接输出结果，不调用 AI
                matched_code = str(exact_match.iloc[-1]['代码']).strip()
                st.session_state.ai_ans = f"✅ **系统提示：** 在经验库中找到与当前描述【完全一致】的历史记录。已自动采用历史判定结果，未消耗 AI 额度。"
                st.session_state.candidate_codes = [matched_code]
                st.success("检测到完全匹配的既往经验，已直接提取。")
            else:
                # 情况 B：无完全对应经验，AI 介入进行模糊检索和逻辑判断
                with st.spinner("未发现完全一致的经验，正在启动 AI 结合代码库进行综合判定..."):
                    # 提取最近的 100 条相关经验作为 AI 的参考
                    exp_context = df_exp.tail(100).to_string(index=False) if not df_exp.empty else "尚无历史经验"
                    code_library_text = df_code.to_string(index=False)

                    prompt = f"""
你是一位精通管理体系认证机构认证业务范围分类指南的专业认证技术专家。你的唯一任务是根据用户提供的“认证业务范围描述”，检索代码库并匹配最准确的“专业代码”。
任务：请从下方的【全量代码库】中检索出与《待评审范围》最匹配的 **3 到 5 个** 可能的代码候选。

### 1. 待评审范围：
{scope}

### 2. 历史判定经验：
{exp_context}

### 3. 全量代码库：
{code_library_text}

### 4. 判定要求：
- **禁止幻觉**：必须且只能从上方的全量代码库中选择真实存在的代码，严禁自行编造。
- **深度比对**：仔细核对“包含项”以及“不包括”项，排除容易混淆的代码。
- **多候选输出**：按匹配程度从高到低排序，给出 3 到 5 个候选方案。
- **固定提取格式**：为了系统自动抓取，请务必在每个建议后另起一行，严格写入：建议代码：[6位代码号]
- **优先参考经验**：仔细比对《历史判定经验》。如果当前的范围描述与历史记录中的范围高度相似，必须优先采用历史记录中的代码，以确保评审尺度的一致性。

### 5. 输出格式示例：
匹配代码： [例如：19.02.00]
分类名称： [例如：计算机及其外部设备的制造]
依据说明： [简述为什么选择该代码，例如：根据产品最终用途属于计算机硬件配套。]

### 6. 特殊指令：
如果一个范围可能对应多个代码，请全部列出并简要说明区别。
"""
            with st.spinner("DeepSeek 正在扫描代码库并深度推理..."):
                try:
                    res = client.chat.completions.create(
                        model="deepseek-v4-pro",
                        messages=[
                            {"role": "system", "content": "你是一个极其严谨的认证代码判定机器人。"},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.1,
                        reasoning_effort="high",
                        extra_body={"thinking": {"type": "enabled"}}
                    )
                    ai_ans = res.choices[0].message.content
                    st.session_state.ai_ans = ai_ans
                    
                    # 正则提取所有建议的代码
                    all_codes = re.findall(r"(?:建议代码|匹配代码)[:：]\s*\[?([0-9.]+)\]?", ai_ans)
                    # 去重清理
                    st.session_state.candidate_codes = list(dict.fromkeys([c.strip('.') for c in all_codes]))
                except Exception as e:
                    st.error(f"AI 调用失败: {e}")

    # 展示判定结果与人员卡片
    if st.session_state.ai_ans:
        st.info("### 💡 AI 判定建议详情")
        st.markdown(st.session_state.ai_ans)
        
    if st.session_state.candidate_codes:
        st.divider()
        st.subheader("👥 候选代码对应专家名单")
            
        for code in st.session_state.candidate_codes:
            auditors = get_auditors(code, df_cat)
            # 使用折叠面板美观展示多个人员组合
            with st.expander(f"📍 候选代码 {code} 的评审人员配置", expanded=True):
                if auditors:
                    a1, a2, a3 = st.columns(3)
                    a1.success(f"**QMS 专家**\n\n{auditors['Q']}")
                    a2.success(f"**EMS 专家**\n\n{auditors['E']}")
                    a3.success(f"**OHSMS 专家**\n\n{auditors['S']}")
                else:
                    st.warning(f"未能在大类表中找到代码 {code[:2]} 开头的对应专家，请检查 category.csv。")

    st.divider()

    # ==========================================
    # 模块三：批量录入与同步经验库
    # ==========================================
    st.divider()
    st.header("💾 模块三：批量手动录入经验库")
    st.info("💡 此模块为独立功能，支持一次性录入三组经验。仅填写完整（范围+代码）的行会被提交。")

    # 创建表头
    h_col1, h_col2 = st.columns([3, 1])
    h_col1.markdown("**受审核方范围**")
    h_col2.markdown("**审核代码**")

    # 定义数据列表，用于存放输入
    batch_data = []

    # 循环生成 3 组输入框
    for i in range(3):
        r_col1, r_col2 = st.columns([3, 1])
        with r_col1:
            input_scope = st.text_input(f"范围 {i+1}", label_visibility="collapsed", key=f"manual_scope_{i}", placeholder=f"请输入第 {i+1} 组范围描述...")
        with r_col2:
            input_code = st.text_input(f"代码 {i+1}", label_visibility="collapsed", key=f"manual_code_{i}", placeholder="例如: 29.01.01")
        
        # 只要范围和代码都不为空，就加入待提交列表
        if input_scope.strip() and input_code.strip():
            batch_data.append({"scope": input_scope.strip(), "code": input_code.strip()})

    # 提交按钮
    if st.button("🚀 批量确认并同步至 GitHub", type="primary"):
        if not batch_data:
            st.warning("⚠️ 请至少完整填写一组“范围”和“代码”后再提交。")
        else:
            success_count = 0
            fail_logs = []
            
            with st.spinner(f"正在同步 {len(batch_data)} 条数据至 GitHub..."):
                for item in batch_data:
                    success, msg = sync_to_github(item["scope"], item["code"])
                    if success:
                        success_count += 1
                    else:
                        fail_logs.append(f"代码 {item['code']} 同步失败: {msg}")
            
            # 结果反馈
            if success_count > 0:
                st.success(f"✅ 成功同步 {success_count} 条新经验！")
                if success_count == len(batch_data):
                    st.balloons()
            
            if fail_logs:
                for err in fail_logs:
                    st.error(err)
        
            st.info("提示：GitHub 仓库已更新，数据生效可能有延迟（约 1 分钟）。")
            
else:
    st.error("⚠️ 核心文件缺失：请确保项目根目录下包含 `code.csv`, `calculate.csv`, `category.csv` 三个文件。")
