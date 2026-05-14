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
import time

# ==========================================
# 1. 基础配置与 API 初始化
# ==========================================
st.set_page_config(page_title="启辰认证-智能评审系统", layout="wide", page_icon="🛡️")

# DeepSeek 客户端初始化
API_KEY = st.secrets["deepseek_key"] 
client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

# 初始化 Session State
if 'calc_done' not in st.session_state: st.session_state.calc_done = False
if 'q' not in st.session_state: st.session_state.q = 0
if 'e' not in st.session_state: st.session_state.e = 0
if 's' not in st.session_state: st.session_state.s = 0
if 'val1' not in st.session_state: st.session_state.val1 = 0.0
if 'val2' not in st.session_state: st.session_state.val2 = 0.0
if 'ai_ans' not in st.session_state: st.session_state.ai_ans = ""
if 'candidate_codes' not in st.session_state: st.session_state.candidate_codes = []

# ==========================================
# 2. 核心辅助函数
# ==========================================
def get_actual_path(filename):
    if os.path.exists(filename): return filename
    upper_file = filename.replace(".csv", ".CSV")
    if os.path.exists(upper_file): return upper_file
    return filename

def smart_read_csv(file_path, **kwargs):
    try: return pd.read_csv(file_path, encoding='utf-8-sig', **kwargs)
    except: return pd.read_csv(file_path, encoding='gbk', **kwargs)

def calculate_logic(people_val, q_risk, e_risk, s_risk, df_calc):
    try:
        df_calc.iloc[:, 0] = df_calc.iloc[:, 0].astype(str).str.strip()
        search_val = str(people_val).strip()
        row = df_calc[df_calc.iloc[:, 0] == search_val]
        if row.empty: return 0.0, 0.0, 0.0
        
        risk_map = {"高": 0, "中": 1, "低": 2, "未评级": 1}
        q = float(row.iloc[0, 1 + risk_map.get(q_risk, 1)])
        e = float(row.iloc[0, 4 + risk_map.get(e_risk, 1)])
        s = float(row.iloc[0, 7 + risk_map.get(s_risk, 1)])
        return q, e, s
    except Exception as e:
        st.error(f"人日提取出错: {e}")
        return 0.0, 0.0, 0.0

def get_auditors(code_str, df_cat):
    try:
        match = re.search(r'^(\d{1,2})', str(code_str).strip())
        if not match: return None
        major_cat_num = str(int(match.group(1))) 
        df_cat.iloc[:, 0] = df_cat.iloc[:, 0].astype(str).str.strip()
        row = df_cat[df_cat.iloc[:, 0] == major_cat_num]
        if row.empty: return None
        def fill_na(val): return val if pd.notna(val) and str(val).strip() != "" else "/"
        return {"Q": fill_na(row.iloc[0, 1]), "E": fill_na(row.iloc[0, 2]), "S": fill_na(row.iloc[0, 3])}
    except Exception: return None

def sync_to_github(new_scope, new_code):
    try:
        token = st.secrets["github_token"]
        repo_name = st.secrets["github_repo"]
        file_path = "experience.csv"
        g = Github(token)
        repo = g.get_repo(repo_name)
        file_content = repo.get_contents(file_path)
        old_data_raw = file_content.decoded_content.decode('utf-8-sig')
        clean_scope = str(new_scope).replace(',', '，').replace('\n', ' ')
        new_line = f"\n{clean_scope},{new_code}"
        updated_content = old_data_raw.strip() + new_line
        repo.update_file(path=file_path, message=f"Sync: {new_code}", content=updated_content, sha=file_content.sha)
        return True, "成功"
    except Exception as e: return False, str(e)

# --- 针对模块四的新增特征提取函数 ---
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['ch_sim', 'en'])

def extract_app_fields(file):
    doc = Document(file)
    full_text_original = ""
    for p in doc.paragraphs: full_text_original += p.text + "\n"
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells: full_text_original += cell.text + " "
    
    full_text = full_text_original.replace(" ", "").replace("\u3000", "")
    
    # 1. 提取公司名称 (锚定表头，提取其后的非空字块)
    # 逻辑：匹配“申请组织名称”及其后的冒号、空格或方括号，直到遇到下一个空格、换行或右方括号停止
    comp_match = re.search(r"(?:申请组织名称|企业名称|组织名称)[：:\s]*[\[［]?([^\s\]］\n]+)", full_text_original)
    
    if comp_match:
        comp_name = comp_match.group(1).strip()
    else:
        # 如果没找到标签，再退而求其次找第一个出现的“有限公司”作为保底
        backup_match = re.search(r"([^\s\n：:\[\]]+?(?:有限公司|股份有限公司))", full_text_original)
        comp_name = backup_match.group(1).strip() if backup_match else "未知公司名称"

    num_match = re.search(r"员工总人数[：:](\d+)", full_text)
    emp_count = num_match.group(1) if num_match else "未识别到人数"
    
    na_match = re.search(r"QMS不适用条款[：:](.*?)是否存在外包过程", full_text, re.DOTALL)
    na_clause = na_match.group(1).strip() if na_match else "未填写"

    date_match = re.search(r"管理体系开始运行时间[：:](.*?)(内审时间|管理体系内部审核时间)", full_text, re.DOTALL)
    run_date = date_match.group(1).strip() if date_match else "未识别到时间"

    is_outsourced = "未知"
    if "是否存在外包过程" in full_text:
        out_part = full_text.split("是否存在外包过程")[1][:20]
        if any(x in out_part for x in ["■是", "☑是", "√是"]): is_outsourced = "是"
        elif any(x in out_part for x in ["■否", "☑否", "√否"]): is_outsourced = "否"
        else: is_outsourced = "未清晰勾选"

    scope_content = "未识别到范围"
    if "QMS:" in full_text_original and "EMS:" in full_text_original:
        try: scope_content = full_text_original.split("QMS:")[1].split("EMS:")[0].strip()
        except: pass
            
    return {"comp_name": comp_name, "emp_count": emp_count, "is_outsourced": is_outsourced, 
            "scope": scope_content, "na_clause": na_clause, "run_date": run_date}

def extract_manual_sections(file):
    doc = Document(file)
    sections = {"scope_43": "", "outsource_8153": ""}
    current_section = None
    full_text_original = ""
    
    for p in doc.paragraphs:
        text = p.text.strip()
        full_text_original += text + "\n"
        if "4.3" in text and "范围" in text: current_section = "scope_43"
        elif "8.1.5.3" in text and "外包" in text: current_section = "outsource_8153"
        elif current_section and re.match(r"^\d+\.\d+", text) and not text.startswith("4.") and not text.startswith("8."):
            current_section = None
        if current_section: sections[current_section] += text + "\n"
        
    table_83_info = []
    for table in doc.tables:
        for row in table.rows:
            row_text = " ".join([cell.text.strip() for cell in row.cells])
            full_text_original += row_text + "\n"
            row_no_space = row_text.replace(" ", "")
            if "8.3" in row_no_space and ("设计" in row_no_space or "开发" in row_no_space or "▲" in row_no_space or "●" in row_no_space or "○" in row_no_space):
                table_83_info.append(row_text)

    outsource_status = sections["outsource_8153"]
    if "无外包过程" in outsource_status or "无外包过程" in full_text_original:
        outsource_status = "【明确无外包】经识别公司无外包过程"

    # 提取日期 (定位“总经理”签字后的第一个日期，通常位于 0.4 颁布令)
    # [\s\S]*? 表示跨行寻找最近的日期格式
    date_match = re.search(r"总经理[\s\S]*?([0-9]{4}\s*年\s*[0-9]{1,2}\s*月\s*[0-9]{1,2}\s*日|[0-9]{4}[-/\.][0-9]{1,2}[-/\.][0-9]{1,2})", full_text_original)
    # 顺便去除了日期中可能因为排版产生的多余空格
    pub_date = date_match.group(1).replace(" ", "") if date_match else "未找到颁布令日期"
    
    doc_no_match = re.search(r"文件编号[：:]?\s*([A-Za-z0-9\-\.]+)", full_text_original)
    doc_no = doc_no_match.group(1) if doc_no_match else "未找到编号"

    return {"scope_43": sections["scope_43"], "outsource_8153": outsource_status, "pub_date": pub_date, 
            "doc_no": doc_no, "table_83": "\n".join(set(table_83_info)) if table_83_info else "未找到8.3条款记录"}

# ==========================================
# 3. 文件检查与加载
# ==========================================
CODE_FILE = get_actual_path("code.csv")
CALC_FILE = get_actual_path("calculate.csv")
CAT_FILE = get_actual_path("category.csv")
EXP_FILE = get_actual_path("experience.csv")
RISK_FILE = get_actual_path("risk.csv")

if not os.path.exists(EXP_FILE):
    pd.DataFrame(columns=["范围", "代码"]).to_csv(EXP_FILE, index=False, encoding='utf-8-sig')

if all(os.path.exists(f) for f in [CODE_FILE, CALC_FILE, CAT_FILE, RISK_FILE]):
    df_code = smart_read_csv(CODE_FILE)
    df_calc = smart_read_csv(CALC_FILE, header=1) 
    df_cat = smart_read_csv(CAT_FILE)
    df_exp = smart_read_csv(EXP_FILE)
    df_risk = smart_read_csv("risk.csv")

    st.title("🛡️ 启辰认证-智能评审自动化系统")

    # ==========================================
    # 模块一：智能综合评审 (整合版)
    # ==========================================
    st.header("📊 模块一：智能综合评审")
    st.info("💡 流程：营业执照拦截(可选) -> 检索经验库 -> AI定代码 -> 定风险 -> 算人日 -> 配老师")

    col1, col2 = st.columns([2, 1])
    
    with col1:
        scope_input = st.text_area("✍️ 请输入受审核方的范围描述：", height=120, placeholder="例如：电子元器件的生产...")
        people_list = df_calc.iloc[:, 0].dropna().unique().tolist()
        c_p, c_s = st.columns(2)
        with c_p: emp_count = st.selectbox("受审核组织人数", options=people_list)
        with c_s: system_type = st.multiselect("认证体系 (多选)", ["QMS", "EMS", "OHSMS"], default=["QMS"])
        
    with col2:
        st.markdown("**📷 前置一致性拦截 (可选)**")
        license_file = st.file_uploader("上传营业执照比对范围", type=["jpg", "png", "jpeg"], key="pre_lic")

    if st.button("🚀 开始全链路解析", type="primary"):
        if not scope_input.strip():
            st.warning("⚠️ 请输入审核范围描述！")
        elif not system_type:
            st.warning("⚠️ 请至少选择一个认证体系！")
        else:
            # === 阶段 1：可选的营业执照一致性校验 ===
            license_pass = True
            if license_file is not None:
                with st.spinner("📷 正在 OCR 识别执照并比对经营范围..."):
                    reader = load_ocr()
                    lic_text = " ".join(reader.readtext(np.array(Image.open(license_file)), detail=0))
                    
                    check_prompt = f"""
                    用户输入的认证范围是：{scope_input}
                    营业执照的经营范围是：{lic_text}
                    
                    请严格核查：认证范围是否完全被包含在营业执照的经营范围内？
                    如果完全一致或被包含，请在回复开头写上“【比对通过】”。
                    如果认证范围超出了营业执照，请在回复开头写上“【比对失败】”，并简述超出的内容。
                    """
                    try:
                        res_lic = client.chat.completions.create(model="deepseek-v4-pro", messages=[{"role": "user", "content": check_prompt}], temperature=0.1)
                        check_result = res_lic.choices[0].message.content
                        if "【比对失败】" in check_result:
                            st.error("🚨 营业执照前置比对未通过！流程已终止。")
                            st.warning(check_result)
                            license_pass = False
                            st.stop()  # 阻断后续运行
                        else:
                            st.success("✅ 营业执照核对通过，申请范围在经营范围内。")
							time.sleep(3)
                    except Exception as e:
                        st.error(f"执照比对调用失败: {e}")
                        st.stop()

            # === 阶段 2：经验检索 (优先) ===
            candidate_codes = []
            ai_ans_text = ""
            
            clean_input = scope_input.strip().replace('\n', ' ')
            exact_match = df_exp[df_exp['范围'].str.strip().replace('\n', ' ') == clean_input]

            if not exact_match.empty:
                matched_code = str(exact_match.iloc[-1]['代码']).strip()
                ai_ans_text = f"✅ **系统提示：** 在经验库中找到完全一致的历史记录。已自动采用历史判定，未消耗 AI 额度。"
                candidate_codes = [matched_code]
                st.success("🎯 检测到经验库完全匹配，跳过 AI 推理！")
            
            # === 阶段 3：如果经验库没找到，走 AI 判定 ===
            else:
                with st.spinner("🤖 未命中经验库，AI 正在全库检索审核代码..."):
                    exp_context = df_exp.tail(100).to_string(index=False) if not df_exp.empty else "尚无历史经验"
                    code_library_text = df_code.to_string(index=False)
                    
                    prompt = f"""
你是一位精通管理体系认证机构认证业务范围分类指南的专业认证技术专家。你的唯一任务是根据用户提供的“认证业务范围描述”，检索代码库并匹配最准确的“专业代码”。
任务：请从下方的【全量代码库】中检索出与《待评审范围》最匹配的 **3 到 5 个** 可能的代码候选。

### 1. 待评审范围：
{scope_input}

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
                    try:
                        res = client.chat.completions.create(model="deepseek-v4-pro", messages=[{"role": "user", "content": prompt}], temperature=0.1)
                        ai_ans_text = res.choices[0].message.content
                        all_codes = re.findall(r"(?:建议代码|匹配代码)[:：]\s*\[?([0-9.]+)\]?", ai_ans_text)
                        candidate_codes = list(dict.fromkeys([c.strip('.') for c in all_codes]))
                    except Exception as e:
                        st.error(f"AI 调用失败: {e}")

            # === 阶段 4：核算展示 ===
            if candidate_codes:
                st.markdown("### 💡 范围解析结果")
                st.info(ai_ans_text)
                
                st.divider()
                st.markdown("### 📊 资源配置与人日核算清单")
                
                df_risk.iloc[:, 0] = df_risk.iloc[:, 0].astype(str).str.strip()
                num_to_text = {1: "高", 2: "中", 3: "低"} # 修正的风险映射

                for code in candidate_codes:
                    with st.expander(f"📍 候选代码 【{code}】 的完整配置方案", expanded=True):
                        # 查风险
                        risk_info = df_risk[df_risk.iloc[:, 0] == code]
                        if not risk_info.empty:
                            q_level = num_to_text.get(int(risk_info.iloc[0, 1]), "中") if pd.notna(risk_info.iloc[0, 1]) else "中"
                            e_level = num_to_text.get(int(risk_info.iloc[0, 2]), "中") if pd.notna(risk_info.iloc[0, 2]) else "中"
                            s_level = num_to_text.get(int(risk_info.iloc[0, 3]), "中") if pd.notna(risk_info.iloc[0, 3]) else "中"
                        else:
                            q_level = e_level = s_level = "中"
                            st.caption(f"⚠️ risk.csv 中无此代码，已默认按中风险计算。")

                        
						# 算天数
                        q_base, e_base, s_base = calculate_logic(emp_count, q_level, e_level, s_level, df_calc)
                        q_days = q_base if "QMS" in system_type else 0.0
                        e_days = e_base if "EMS" in system_type else 0.0
                        s_days = s_base if "OHSMS" in system_type else 0.0
                        
                        total_base = q_days + e_days + s_days
                        final_days = total_base * 0.7 * 0.8 # 自动多体系优惠计算
						
                        # 配老师
                        auditors = get_auditors(code, df_cat)

                        c_q, c_e, c_s, c_sum = st.columns(4)
                        with c_q:
                            st.markdown("**QMS 体系**")
                            st.write(f"风险: **{q_level}** | 初始: `{q_days}`")
                            st.write(f"专家: {auditors['Q'] if auditors else '/'}")
                        with c_e:
                            st.markdown("**EMS 体系**")
                            st.write(f"风险: **{e_level}** | 初始: `{e_days}`")
                            st.write(f"专家: {auditors['E'] if auditors else '/'}")
                        with c_s:
                            st.markdown("**OHSMS 体系**")
                            st.write(f"风险: **{s_level}** | 初始: `{s_days}`")
                            st.write(f"专家: {auditors['S'] if auditors else '/'}")
                        with c_sum:
                            st.markdown("🎯 **建议核发人日**")
                            st.metric(label="(Q+E+S)×70%×80%", value=f"{final_days:.2f} 天")
            else:
                st.error("未能从范围内提取出有效的代码，请检查范围描述。")

    st.divider()

    # ==========================================
    # 模块二：批量录入与同步经验库 (原样保留)
    # ==========================================
    st.header("💾 模块二：批量同步经验库")
    batch_data = []
    for i in range(3):
        r_col1, r_col2 = st.columns([3, 1])
        with r_col1: input_scope = st.text_input(f"范围 {i}", label_visibility="collapsed", key=f"s_{i}", placeholder="范围")
        with r_col2: input_code = st.text_input(f"代码 {i}", label_visibility="collapsed", key=f"c_{i}", placeholder="代码")
        if input_scope.strip() and input_code.strip(): batch_data.append({"scope": input_scope.strip(), "code": input_code.strip()})

    if st.button("🚀 批量同步至 GitHub", type="primary"):
        if not batch_data: st.warning("请填写完整。")
        else:
            success_count = 0
            with st.spinner(f"同步 {len(batch_data)} 条..."):
                for item in batch_data:
                    success, msg = sync_to_github(item["scope"], item["code"])
                    if success: success_count += 1
                    else: st.error(f"失败: {msg}")
            if success_count > 0: st.success(f"成功同步 {success_count} 条！")

    st.divider()

    # ==========================================
    # 模块三：合同与体系文件精准核查 (原样保留)
    # ==========================================
    st.header("📂 模块三：合同与体系文件精准核查")
    st.caption("系统将自动拆解表格提取人数/外包等关键信息，并与手册及执照比对。")

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1: f_app = st.file_uploader("1. 申请书 (.docx/.doc)", type=["docx", "doc"], key="file_app")
    with col_f2: f_manual = st.file_uploader("2. 管理手册 (.docx/.doc)", type=["docx", "doc"], key="file_man")
    with col_f3: f_license = st.file_uploader("3. 营业执照 (.jpg/.png)", type=["jpg", "png", "jpeg"], key="file_lic")

    if st.button("🔍 运行精准核查 (OCR+解析)"):
        if not (f_app and f_manual and f_license):
            st.error("⚠️ 请先将申请书、手册、营业执照三个文件全部上传。")
        else:
            try:
                with st.spinner("正在穿透表格提取核心数据..."):
                    processed_app = handle_doc_file(f_app)
                    processed_manual = handle_doc_file(f_manual)
                    
                    app_data = extract_app_fields(processed_app)
                    man_data = extract_manual_sections(processed_manual)
                    
                    reader = load_ocr()
                    license_text = " ".join(reader.readtext(np.array(Image.open(f_license)), detail=0))

                with st.spinner("数据已提取，AI 正在进行逻辑冲突检测..."):
                    check_prompt = f"""
                    你现在是【北京北方启辰认证服务有限公司】的高级评审专家。
                    请根据我提取出的精准数据，出具一份高标准的《预审纠错报告》。

                    【资料 1：申请书提取结果】
                    - 申请组织全称：{app_data['comp_name']}
                    - 申报总人数：{app_data['emp_count']}
                    - 认证范围描述：{app_data['scope']}
                    - 外包勾选状态：{app_data['is_outsourced']}
                    - QMS不适用条款：{app_data['na_clause']}
                    - 体系开始运行时间：{app_data['run_date']}

                    【资料 2：管理手册提取结果】
                    - 手册文件编号：{man_data['doc_no']}
                    - 手册发布/实施日期：{man_data['pub_date']}
                    - 4.3 体系范围描述：{man_data['scope_43'] if man_data['scope_43'] else '未找到4.3章节'}
                    - 8.1.5.3 外包控制描述：{man_data['outsource_8153']}
                    - 职能分配表8.3记录：{man_data['table_83']}

                    【资料 3：营业执照 OCR 内容】
                    - 内容：{license_text}

                    核心逻辑
                    1. **外包控制一致性**：如果手册{man_data['outsource_8153']}描述为“【明确无外包】”，且申请书{app_data['is_outsourced']}勾选“否”，直接通过。如果有外包描述但申请书选“否”，必须严重警告。（不比较营业执照内容）
                    2. **不适用条款核查**：对比申请书的“QMS不适用条款（{app_data['na_clause']}）”与“职能分配表8.3记录{man_data['table_83']}”。若申请书写了8.3不适用，手册的分配表中也应标注为无责任或不适用。（不参考营业执照内容，不做多余分析，仅比较申请书内容与手册是否一致）。
                    3. **运行日期倒推**：比较申请书的“体系开始运行时间”与手册的“发布日期”是否一致。且是否满 3 个月（以今日 {datetime.now().date()} 为准）
                    4. **文件编号校验**：提取手册文件编号（{man_data['doc_no']}）中的字母缩写，判断它是否是“{app_data['comp_name']}”公司名称的拼音首字母缩写？如果完全无关，请提醒可能套用模板未改编号。
                    5. **资质时限检查**：营业执照成立是否满 3 个月（以今日 {datetime.now().date()} 为准）？
                    6. **业务范围覆盖**：申请书的范围是否在营业执照的经营范围之内？（不做多余分析，仅比较内容）

                    输出格式要求：
                    一、**识别内容**：
                    1.**申请书内容**
	                    [申报人数]：
	                    [认证范围]：
	                    [外包勾选状态]：
	                    [QMS不适用条款]：
	                    [管理体系运行日期]：
	                    [公司名称]：
                    2.**管理手册内容**
                    	[体系范围描述]：
	                    [外包控制描述]：
	                    [8.3适用情况]：
	                    [文件编号]：
                    3.**营业执照内容**
	                    [经营范围]：
                二、**预审报告**：
                    1.**结论**：予以受理 / 需修改/ 不予受理
                    2.**成立时长**：满足三个月/不满三个月
                    3.**范围匹配**：匹配/不匹配
                    4.**外包情况**：存在/不存在/申请书和手册内容不一致
                    5.**不适用条款核查**：一致/不一致
                    6.**体系运行时间**：满足三个月/不满三个月
                    7.**文件编号核查**：符合/不符合
                    """
                    
                    res = client.chat.completions.create(model="deepseek-v4-pro", messages=[{"role":"user","content":check_prompt}], temperature=0.1)
                    st.success("解析比对完成！")
                    st.markdown("### 📋 自动合同评审纠错报告")
                    st.markdown(res.choices[0].message.content)
                    
            except Exception as e:
                st.error(f"核查过程中出现错误: {e}")

else:
    st.error("⚠️ 核心文件缺失：请确保项目根目录下包含 code, calculate, category, experience, risk 等 5 个 CSV 文件。")
