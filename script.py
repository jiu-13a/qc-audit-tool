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
# 2. 核心工具函数
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
        if row.empty: return 0, 0, 0
        risk_map = {"高": 0, "中": 1, "低": 2}
        q = float(row.iloc[0, 1 + risk_map[q_risk]])
        e = float(row.iloc[0, 4 + risk_map[e_risk]])
        s = float(row.iloc[0, 7 + risk_map[s_risk]])
        return q, e, s
    except Exception as e:
        st.error(f"计算出错: {e}")
        return 0, 0, 0

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
    """精准抓取申请书内容"""
    doc = Document(file)
    full_text_original = ""
    for p in doc.paragraphs: full_text_original += p.text + "\n"
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells: full_text_original += cell.text + " "
    
    # 彻底去除空格，方便正则定位
    full_text = full_text_original.replace(" ", "").replace("\u3000", "")
    
    # 1. 提取公司名称 (用于后续文件编号比对)
    comp_match = re.search(r"(申请组织名称|企业名称|组织名称)[：:]+(.*?)\n", full_text_original)
    comp_name = comp_match.group(2).strip() if comp_match else "未知公司名称"

    # 2. 提取人数
    num_match = re.search(r"员工总人数[：:](\d+)", full_text)
    emp_count = num_match.group(1) if num_match else "未识别到人数"
    
    # 3. 提取不适用条款
    na_match = re.search(r"QMS不适用条款[：:](.*?)是否存在外包过程", full_text, re.DOTALL)
    na_clause = na_match.group(1).strip() if na_match else "未填写"

    # 4. 提取运行时间
    date_match = re.search(r"管理体系开始运行时间[：:](.*?)(内审时间|管理体系内部审核时间)", full_text, re.DOTALL)
    run_date = date_match.group(1).strip() if date_match else "未识别到时间"

    # 5. 提取外包勾选
    is_outsourced = "未知"
    if "是否存在外包过程" in full_text:
        out_part = full_text.split("是否存在外包过程")[1][:20]
        if any(x in out_part for x in ["■是", "☑是", "√是"]): is_outsourced = "是"
        elif any(x in out_part for x in ["■否", "☑否", "√否"]): is_outsourced = "否"
        else: is_outsourced = "未清晰勾选"

    # 6. 提取范围
    scope_content = "未识别到范围"
    if "QMS:" in full_text_original and "EMS:" in full_text_original:
        try: scope_content = full_text_original.split("QMS:")[1].split("EMS:")[0].strip()
        except: pass
            
    return {
        "comp_name": comp_name,
        "emp_count": emp_count,
        "is_outsourced": is_outsourced,
        "scope": scope_content,
        "na_clause": na_clause,
        "run_date": run_date
    }

def extract_manual_sections(file):
    """精准抓取管理手册章节与属性"""
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
            # 专门定位包含8.3的表格行
            if "8.3" in row_no_space and ("设计" in row_no_space or "开发" in row_no_space or "▲" in row_no_space or "●" in row_no_space or "○" in row_no_space):
                table_83_info.append(row_text)

    # 优化外包状态判定
    outsource_status = sections["outsource_8153"]
    if "无外包过程" in outsource_status or "无外包过程" in full_text_original:
        outsource_status = "【明确无外包】经识别公司无外包过程"

    # 提取日期与编号
    date_match = re.search(r"(发布日期|实施日期)[：:]?\s*([0-9]{4}年[0-9]{1,2}月[0-9]{1,2}日|[0-9\-\./]+)", full_text_original)
    pub_date = date_match.group(2) if date_match else "未找到日期"
    
    doc_no_match = re.search(r"文件编号[：:]?\s*([A-Za-z0-9\-\.]+)", full_text_original)
    doc_no = doc_no_match.group(1) if doc_no_match else "未找到编号"

    return {
        "scope_43": sections["scope_43"],
        "outsource_8153": outsource_status,
        "pub_date": pub_date,
        "doc_no": doc_no,
        "table_83": "\n".join(set(table_83_info)) if table_83_info else "未在表格中找到8.3条款记录"
    }

# ==========================================
# 3. 文件检查与加载
# ==========================================
CODE_FILE = get_actual_path("code.csv")
CALC_FILE = get_actual_path("calculate.csv")
CAT_FILE = get_actual_path("category.csv")
EXP_FILE = get_actual_path("experience.csv")

if not os.path.exists(EXP_FILE):
    pd.DataFrame(columns=["范围", "代码"]).to_csv(EXP_FILE, index=False, encoding='utf-8-sig')

if all(os.path.exists(f) for f in [CODE_FILE, CALC_FILE, CAT_FILE]):
    df_code = smart_read_csv(CODE_FILE)
    df_calc = smart_read_csv(CALC_FILE, header=1) 
    df_cat = smart_read_csv(CAT_FILE)
    df_exp = smart_read_csv(EXP_FILE)

    st.title("🛡️ 启辰认证-合同评审自动化处理系统")

    # --- 模块一：确认人日数目 ---
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
        st.session_state.val1 = base_sum * 0.8
        st.session_state.val2 = base_sum * 0.8 * 0.8
        st.session_state.calc_done = True

    if st.session_state.calc_done:
        st.write(f"**提取到的各体系基础人日：** QMS: `{st.session_state.q}` | EMS: `{st.session_state.e}` | OHSMS: `{st.session_state.s}`")
        m_col1, m_col2 = st.columns(2)
        m_col1.metric("数值一：(Q+E+S) × 0.8", f"{st.session_state.val1:.3f}")
        m_col2.metric("数值二：(Q+E+S) × 0.8 × 0.8", f"{st.session_state.val2:.3f}")

    st.divider()

    # --- 模块二：全量代码检索 ---
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


    # --- 模块三：批量录入 ---
    st.header("💾 模块三：批量同步经验库")
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
    # 模块四：合同文件多维度评审 (新增)
    # ==========================================
    st.header("📂 模块四：合同与体系文件精准核查")
    st.caption("系统将自动拆解表格提取人数/外包等关键信息，并与手册及执照比对。")

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1: f_app = st.file_uploader("1. 申请书 (.docx)", type=["docx"], key="file_app")
    with col_f2: f_manual = st.file_uploader("2. 管理手册 (.docx)", type=["docx"], key="file_man")
    with col_f3: f_license = st.file_uploader("3. 营业执照 (.jpg/.png)", type=["jpg", "png", "jpeg"], key="file_lic")

    if st.button("🔍 运行精准核查 (OCR+解析)"):
        if not (f_app and f_manual and f_license):
            st.error("⚠️ 请先将申请书、手册、营业执照三个文件全部上传。")
        else:
            try:
                with st.spinner("正在穿透表格提取核心数据..."):
                    # 提取申请书
                    app_data = extract_app_fields(f_app)
                    # 提取手册
                    man_data = extract_manual_sections(f_manual)
                    # OCR 执照
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
                    2. **不适用条款核查**：对比申请书的“QMS不适用条款（{app_data['na_clause']}）”与“职能分配表8.3记录{man_data['table_83']}”。若申请书写了8.3不适用，手册的分配表中也应标注为无责任或不适用。（不比较营业执照内容）。
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
                    
                    res = client.chat.completions.create(
                        model="deepseek-v4-pro",
                        messages=[{"role":"user","content":check_prompt}],
                        temperature=0.1
                    )
                    st.success("解析比对完成！")
                    st.markdown("### 📋 自动合同评审纠错报告")
                    st.markdown(res.choices[0].message.content)
                    
            except Exception as e:
                st.error(f"核查过程中出现错误: {e}")
                st.info("请确保上传的是标准格式的 docx 文件。")

else:
    st.error("⚠️ 核心文件缺失：请确保项目根目录下包含 `code.csv`, `calculate.csv`, `category.csv` 三个文件。")
