import streamlit as st
import pandas as pd
import altair as alt
import os
import json
import re
from typing import Optional
from dotenv import load_dotenv
from openai import OpenAI

from dbclient import DatabaseClient
from safeish import SafeishPythonExecutor
from app_db import AppDBManager

load_dotenv(".env")

APP_DB_URL = os.getenv("APP_DB_URL", "sqlite:///app.db")
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")

st.set_page_config(
    page_title="Data Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CSS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }

/* Hide default Streamlit chrome */
#MainMenu, footer { display: none !important; }
.stDeployButton { display: none !important; }
header[data-testid="stHeader"] { display: none !important; }

/* App background */
.stApp { background-color: #f1f5f9 !important; color: #1e293b !important; }

/* Main content area */
.main .block-container {
    padding: 1.5rem 2.5rem 4rem !important;
    max-width: 1000px !important;
}

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e2e8f0 !important;
}
section[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.2rem !important;
}

/* Sidebar nav buttons */
section[data-testid="stSidebar"] .stButton > button {
    width: 100% !important;
    text-align: left !important;
    justify-content: flex-start !important;
    border: none !important;
    border-bottom: 1px solid #f1f5f9 !important;
    border-radius: 0 !important;
    padding: 0.65rem 1rem !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    background: transparent !important;
    color: #475569 !important;
    box-shadow: none !important;
    margin-bottom: 0 !important;
    transition: background 0.12s !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: #f8fafc !important;
    color: #1e40af !important;
    transform: none !important;
    box-shadow: none !important;
}
section[data-testid="stSidebar"] .stButton > button p,
section[data-testid="stSidebar"] .stButton > button div,
section[data-testid="stSidebar"] .stButton > button span {
    text-align: left !important;
    width: 100% !important;
}

/* Buttons inside expanders: restore normal style */
section[data-testid="stSidebar"] [data-testid="stExpanderContent"] .stButton > button {
    border-bottom: none !important;
    border-radius: 8px !important;
    justify-content: center !important;
    padding: 0.4rem 0.75rem !important;
}
section[data-testid="stSidebar"] [data-testid="stExpanderContent"] .stButton > button p,
section[data-testid="stSidebar"] [data-testid="stExpanderContent"] .stButton > button div,
section[data-testid="stSidebar"] [data-testid="stExpanderContent"] .stButton > button span {
    text-align: center !important;
    width: auto !important;
}

/* ---- Status chips ---- */
.chip {
    display: inline-flex; align-items: center; gap: 4px;
    padding: 3px 10px; border-radius: 20px;
    font-size: 0.73rem; font-weight: 500;
}
.chip-ok  { background: #ecfdf5; color: #059669; border: 1px solid #a7f3d0; }
.chip-err { background: #fef2f2; color: #dc2626; border: 1px solid #fecaca; }
.chip-warn{ background: #fffbeb; color: #d97706; border: 1px solid #fde68a; }

/* ---- Sidebar logo ---- */
.sidebar-logo {
    text-align: center;
    padding: 0.4rem 0 1.2rem;
    border-bottom: 1px solid #f1f5f9;
    margin-bottom: 0.8rem;
}
.sidebar-logo .logo-icon { font-size: 1.6rem; }
.sidebar-logo .logo-name { font-size: 1rem; font-weight: 700; color: #1e293b; }
.sidebar-logo .logo-sub  { font-size: 0.68rem; color: #94a3b8; }

/* ---- Sidebar user card ---- */
.sidebar-user {
    padding: 0.6rem 0.9rem;
    border-radius: 10px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
}
.sidebar-user .uname { font-weight: 600; font-size: 0.88rem; color: #1e293b; }
.sidebar-user .urole  { font-size: 0.7rem; color: #94a3b8; }

/* ---- Page headings ---- */
.page-title {
    font-size: 1.35rem; font-weight: 700; color: #1e293b;
    margin-bottom: 0.1rem;
}
.page-subtitle {
    font-size: 0.82rem; color: #94a3b8; margin-bottom: 1.2rem;
}

/* ---- Login card ---- */
.login-card {
    background: #fff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 2.2rem 2rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.07);
}
.login-icon  { text-align: center; font-size: 2.2rem; margin-bottom: 0.5rem; }
.login-title { text-align: center; font-size: 1.55rem; font-weight: 700; color: #1e40af; }
.login-sub   { text-align: center; font-size: 0.82rem; color: #94a3b8; margin-bottom: 1.6rem; }

/* ---- General buttons ---- */
.stButton > button {
    border-radius: 8px !important;
    font-weight: 500 !important;
    transition: all 0.15s !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 2px 8px rgba(30,64,175,0.15) !important;
}

/* ---- Inputs — force light ---- */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stNumberInput > div > div > input {
    border-radius: 8px !important;
    background-color: #ffffff !important;
    color: #1e293b !important;
    border: 1px solid #cbd5e1 !important;
}
.stTextInput > div > div > input::placeholder,
.stTextArea > div > div > textarea::placeholder {
    color: #94a3b8 !important;
}
.stSelectbox > div > div,
.stSelectbox > div > div > div {
    border-radius: 8px !important;
    background-color: #ffffff !important;
    color: #1e293b !important;
}
/* Selectbox dropdown list */
[data-baseweb="select"] > div,
[data-baseweb="popover"] ul {
    background-color: #ffffff !important;
    color: #1e293b !important;
}
/* Multiselect tags */
[data-baseweb="tag"] {
    background-color: #eff6ff !important;
    color: #1e40af !important;
}
/* Password input eye icon */
.stTextInput button { color: #64748b !important; }

/* ---- Dataframe ---- */
.stDataFrame { border-radius: 8px !important; overflow: hidden !important; }

/* ---- Expander ---- */
.streamlit-expanderHeader {
    font-weight: 500 !important;
    font-size: 0.84rem !important;
    color: #475569 !important;
}

/* ---- Tabs ---- */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    border-bottom: 2px solid #e2e8f0;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0 !important;
    font-weight: 500 !important;
    padding: 0.5rem 1.2rem !important;
    background: #f1f5f9 !important;
    border: 1px solid #e2e8f0 !important;
    border-bottom: none !important;
    color: #64748b !important;
}
.stTabs [aria-selected="true"] {
    background: #fff !important;
    color: #1e40af !important;
    border-bottom: 2px solid #fff !important;
    margin-bottom: -2px;
}

/* ---- Chat messages ---- */
.stChatMessage { border-radius: 12px !important; }

/* ---- Suggestion chips ---- */
.sug-chip > button {
    background: #eff6ff !important;
    border: 1px solid #bfdbfe !important;
    color: #1e40af !important;
    font-size: 0.8rem !important;
    padding: 0.3rem 0.7rem !important;
    border-radius: 20px !important;
    font-weight: 400 !important;
}
.sug-chip > button:hover {
    background: #dbeafe !important;
    transform: none !important;
    box-shadow: none !important;
}

h1 a, h2 a, h3 a { display: none !important; }

/* Hide "Press Enter to submit form" */
.stTextInput div[data-testid="InputInstructions"] { display: none !important; }
.stTextAreaInput div[data-testid="InputInstructions"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE — init
# ============================================================
if "app_db" not in st.session_state:
    st.session_state.app_db = AppDBManager(db_url=APP_DB_URL, encryption_key=ENCRYPTION_KEY)
if "user_id" not in st.session_state:
    st.session_state.user_id = None

# ============================================================
# AUTH
# ============================================================
if st.session_state.user_id is None:
    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        st.markdown("""
        <div class="login-card">
            <div class="login-icon">📊</div>
            <div class="login-title">AI Data Intelligence</div>
            <div class="login-sub">Nền tảng phân tích dữ liệu · Đa mô hình AI · Bảo mật</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        t_login, t_register = st.tabs(["Đăng nhập", "Tạo tài khoản"])

        with t_login:
            with st.form("login_form"):
                l_user = st.text_input("Tài khoản", placeholder="Tên đăng nhập")
                l_pass = st.text_input("Mật khẩu", type="password", placeholder="Mật khẩu")
                submitted = st.form_submit_button("Đăng nhập", use_container_width=True, type="primary")
            if submitted:
                success, uid = st.session_state.app_db.verify_user(l_user, l_pass)
                if success:
                    st.session_state.user_id = uid
                    st.session_state.username = l_user
                    st.rerun()
                else:
                    st.error("Sai tài khoản hoặc mật khẩu")

        with t_register:
            with st.form("register_form"):
                r_user  = st.text_input("Tài khoản", placeholder="Tên đăng nhập")
                r_email = st.text_input("Email", placeholder="email@example.com")
                r_pass  = st.text_input("Mật khẩu", type="password", placeholder="Tối thiểu 6 ký tự")
                r_cpass = st.text_input("Xác nhận mật khẩu", type="password", placeholder="Nhập lại")
                submitted_r = st.form_submit_button("Tạo tài khoản", use_container_width=True, type="primary")
            if submitted_r:
                if not r_user or not r_pass:
                    st.error("Vui lòng nhập đầy đủ thông tin")
                elif r_pass != r_cpass:
                    st.error("Mật khẩu không khớp!")
                else:
                    try:
                        success, msg = st.session_state.app_db.create_user(r_user, r_email, r_pass)
                        if success:
                            st.success("Đăng ký thành công! Hãy chuyển sang Đăng nhập.")
                        else:
                            st.error(msg)
                    except Exception as e:
                        st.error(f"Lỗi hệ thống: {str(e)}")
    st.stop()

# ============================================================
# SESSION STATE — other vars
# ============================================================
defaults = {
    "messages":                  [],
    "db_client":                 None,
    "db_schema":                 None,
    "available_models":          [],
    "openai_client":             None,
    "query_history":             [],
    "current_context": {
        "dia_phuong": "Chưa xác định",
        "doi_tuong":  "Chưa xác định",
        "ma_lo_dat":  "Chưa xác định",
        "loai_rung":  "Chưa xác định",
    },
    "active_provider":           "OpenAI",
    "active_api_key":            "",
    "active_model":              None,
    "current_page":              "chat",
    "suggestions":               [],
    "auto_process_last_message": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================================
# HELPER FUNCTIONS  (defined before sidebar so they're callable)
# ============================================================
def update_memory(response_text, context_key="<context_update>"):
    pattern = f'{context_key}(.*?){context_key.replace("<", "</")}'
    match = re.search(pattern, response_text, re.DOTALL)
    if match:
        try:
            new_data = json.loads(match.group(1).strip())
            for key, value in new_data.items():
                if value and value != "Chưa xác định":
                    st.session_state.current_context[key] = value
            return re.sub(pattern, '', response_text, flags=re.DOTALL).strip()
        except Exception:
            return response_text
    return response_text


def get_system_prompt():
    context_json = json.dumps(st.session_state.current_context, ensure_ascii=False)
    return f"""Bạn là một chuyên gia phân tích dữ liệu địa chính và lâm nghiệp Việt Nam.
Nhiệm vụ của bạn là hỗ trợ truy vấn cơ sở dữ liệu về địa giới hành chính, các loại rừng và lô đất.

KHO DỮ LIỆU NGỮ CẢNH (CONTEXT):
{context_json}

HƯỚNG DẪN XỬ LÝ:
1. Luôn ưu tiên sử dụng thông tin trong KHO DỮ LIỆU NGỮ CẢNH để hiểu các đại từ thay thế.
2. Nếu người dùng hỏi về địa danh hoặc mã lô đất mới, hãy cập nhật vào câu trả lời.
3. Phản hồi 2 phần:
   - PHẦN 1: Câu trả lời / SQL / Biểu đồ.
   - PHẦN 2 (trong thẻ <context_update>): JSON cập nhật bộ nhớ.

VÍ DỤ:
Người dùng: "Diện tích rừng sản xuất của xã Ea H'leo là bao nhiêu?"
AI: ... (kết quả) ...
<context_update>{{"dia_phuong": "Xã Ea H'leo", "doi_tuong": "Rừng sản xuất"}}</context_update>"""


def fetch_available_models(provider, api_key):
    try:
        if not api_key or not api_key.strip():
            st.error("API Key rỗng hoặc không thể giải mã. Vui lòng xoá và thêm lại API Key.")
            return []
            
        api_key = api_key.strip()
        if provider in ("OpenAI", "Grok (xAI)"):
            base_url = "https://api.x.ai/v1" if provider == "Grok (xAI)" else None
            temp_client = OpenAI(api_key=api_key, base_url=base_url)
            models = temp_client.models.list()
            model_list = []
            for m in models:
                mid = m.id.lower()
                if any(p in mid for p in ["gpt-4o", "gpt-4-turbo", "gpt-4-0", "gpt-3.5-turbo-0"]):
                    model_list.append(m.id)
                elif mid in ["gpt-4", "gpt-3.5-turbo"]:
                    model_list.append(m.id)
                elif "grok" in mid:
                    model_list.append(m.id)
            return sorted(list(set(model_list)))
        elif provider == "Gemini":
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            models = genai.list_models()
            return sorted([
                m.name.replace("models/", "")
                for m in models
                if "generateContent" in m.supported_generation_methods
            ])
        return []
    except Exception as e:
        st.error(f"Lỗi kết nối / xác thực API: {e}")
        return []


def load_dashboard():
    return st.session_state.app_db.get_dashboard(st.session_state.user_id)


def save_to_dashboard(item):
    dashboard = load_dashboard()
    dashboard.append(item)
    st.session_state.app_db.save_dashboard(st.session_state.user_id, dashboard)


def show_data_widget(df, key_prefix):
    """Show dataframe + export buttons + collapsible quick chart."""
    st.dataframe(df.astype(str), use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.download_button(
            "⬇ CSV", df.to_csv(index=False).encode("utf-8"),
            f"{key_prefix}.csv", "text/csv", key=f"csv_{key_prefix}")
    with c2:
        from io import BytesIO
        b = BytesIO()
        df.to_excel(b, index=False, engine="openpyxl")
        st.download_button(
            "⬇ Excel", b.getvalue(),
            f"{key_prefix}.xlsx", key=f"xls_{key_prefix}")
    with c3:
        try:
            json_str = df.astype(str).to_json(orient="records", indent=2)
        except Exception:
            json_str = "[]"
        st.download_button(
            "⬇ JSON", json_str.encode("utf-8"),
            f"{key_prefix}.json", key=f"jsn_{key_prefix}")
    with c4:
        if st.button("📌 Ghim", key=f"pin_{key_prefix}"):
            safe_data = json.loads(df.astype(str).to_json(orient="records"))
            save_to_dashboard({
                "type": "table",
                "data": safe_data,
                "timestamp": str(pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")),
            })
            st.toast("Đã ghim vào Dashboard!")

    num_df = df.select_dtypes(include=["number"])
    if not num_df.empty:
        with st.expander("📈 Biểu đồ nhanh"):
            try:
                index_col = next(
                    (c for c in df.columns if "name" in c.lower() or "ten" in c.lower()), None
                )
                if index_col:
                    chart_data = num_df.copy()
                    chart_data.index = df[index_col]
                    st.bar_chart(chart_data)
                else:
                    st.bar_chart(num_df)
            except Exception:
                st.info("Không thể vẽ biểu đồ nhanh.")


# ============================================================
# CORE FUNCTIONS
# ============================================================
def query_database(sql: str) -> str:
    try:
        result = st.session_state.db_client.query(sql)
        df = DatabaseClient.to_dataframe(result)
        st.session_state.last_df = df
        from datetime import datetime
        st.session_state.query_history.append({
            "sql": sql,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "rows": len(df),
        })
        return DatabaseClient.describe_dataframe_for_llm(df)
    except Exception as e:
        return (
            f"STOP!DATABASE Error: {str(e)}.\n"
            "Please analyze this error, FIX your SQL query, and RUN the query_database tool again."
        )


def create_chart(python_code: str) -> str:
    try:
        executor = SafeishPythonExecutor(safe_globals={"alt": alt, "pd": pd})
        df = st.session_state.get("last_df", pd.DataFrame())
        res = executor.run(python_code, context={"df": df}, return_locals=True)
        if res.ok:
            chart = res.locals.get("chart")
            if chart:
                st.session_state.last_chart = chart
                return "Chart created successfully"
            return "No chart variable found in code"
        return f"Error creating chart: {res.error}"
    except Exception as e:
        return f"Error: {str(e)}"


# OpenAI function-calling tools spec
tools = [
    {
        "type": "function",
        "function": {
            "name": "query_database",
            "description": "Query the PostgreSQL database with a SELECT statement and return results",
            "parameters": {
                "type": "object",
                "properties": {"sql": {"type": "string", "description": "SQL SELECT query"}},
                "required": ["sql"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_chart",
            "description": "Create an Altair chart from the last query result. DataFrame is available as 'df'.",
            "parameters": {
                "type": "object",
                "properties": {"python_code": {"type": "string", "description": "Python code assigning Altair chart to variable 'chart'"}},
                "required": ["python_code"],
            },
        },
    },
]

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown("""
<div class="sidebar-logo">
    <div class="logo-icon">📊</div>
    <div class="logo-name">Data Intelligence</div>
    <div class="logo-sub">AI-powered Analytics</div>
</div>
""", unsafe_allow_html=True)

# Navigation
_pages = [
    ("💬", "Chat",         "chat"),
    ("📊", "Dashboard",    "dashboard"),
    ("🔍", "Explorer",     "explorer"),
    ("⚙️",  "Cài đặt",     "settings"),
]
for _icon, _label, _pid in _pages:
    _btn_type = "primary" if st.session_state.current_page == _pid else "secondary"
    if st.sidebar.button(f"{_icon}  {_label}", key=f"nav_{_pid}",
                         use_container_width=True, type=_btn_type):
        st.session_state.current_page = _pid
        st.rerun()

st.sidebar.markdown("---")

# Status chips
_db_ok = st.session_state.db_client is not None
_ai_ok = bool(st.session_state.active_api_key)
_db_chip = '<span class="chip chip-ok">● DB</span>' if _db_ok \
    else '<span class="chip chip-err">● DB</span>'
_ai_chip = (
    f'<span class="chip chip-ok">● {st.session_state.active_provider[:6]}</span>'
    if _ai_ok else '<span class="chip chip-warn">● AI</span>'
)
st.sidebar.markdown(f"{_db_chip} &nbsp; {_ai_chip}", unsafe_allow_html=True)
st.sidebar.markdown("")

# Quick connect expander
with st.sidebar.expander("⚡ Kết nối nhanh", expanded=(not _db_ok or not _ai_ok)):
    # --- DB ---
    _saved_conns = st.session_state.app_db.get_db_connections(st.session_state.user_id)
    _conn_names  = [c["profile_name"] for c in _saved_conns]
    if _conn_names:
        _sel_name = st.selectbox(
            "Database", _conn_names, key="sidebar_db_select", label_visibility="collapsed"
        )
        _sel_conn = next((c for c in _saved_conns if c["profile_name"] == _sel_name), None)
        if st.button("Kết nối DB", key="sidebar_connect_db", use_container_width=True):
            if not _sel_conn.get("db_host"):
                st.error("Lỗi: Không thể kết nối. Cấu hình DB bị rớt hoặc lỗi mã hoá.")
            else:
                try:
                    os.environ["DB_HOST"]     = _sel_conn["db_host"]
                    os.environ["DB_PORT"]     = _sel_conn["db_port"]
                    os.environ["DB_DATABASE"] = _sel_conn["db_name"]
                    os.environ["DB_USER"]     = _sel_conn["db_user"]
                    os.environ["DB_PASSWORD"] = _sel_conn["db_password"]
                    st.session_state.db_client = DatabaseClient()
                    st.session_state.db_schema = st.session_state.db_client.get_schema_summary()
                    st.success("Đã kết nối!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Lỗi: {e}")
    else:
        st.caption("Chưa có DB. Vào Cài đặt để thêm.")

    st.markdown("")

    # --- API Key + Model ---
    _saved_keys = st.session_state.app_db.get_api_keys(st.session_state.user_id)
    if _saved_keys:
        _key_labels = [f"{k['profile_name']} ({k['provider']})" for k in _saved_keys]
        _sel_key_idx = st.selectbox(
            "API Key", range(len(_key_labels)),
            format_func=lambda i: _key_labels[i],
            key="sidebar_api_select", label_visibility="collapsed",
        )
        _sel_key = _saved_keys[_sel_key_idx]

        if st.session_state.active_api_key != _sel_key["api_key"]:
            st.session_state.active_api_key  = _sel_key["api_key"]
            st.session_state.active_provider = _sel_key["provider"]
            st.session_state.available_models = []
            if _sel_key["provider"] in ("OpenAI", "Grok (xAI)"):
                _base_url = "https://api.x.ai/v1" if _sel_key["provider"] == "Grok (xAI)" else None
                st.session_state.openai_client = OpenAI(
                    api_key=_sel_key["api_key"], base_url=_base_url
                )

        if st.button("Lấy Model", key="sidebar_fetch_models", use_container_width=True):
            _models = fetch_available_models(
                st.session_state.active_provider, st.session_state.active_api_key
            )
            if _models:
                st.session_state.available_models = _models

        if st.session_state.available_models:
            _sel_model = st.selectbox(
                "Model", st.session_state.available_models,
                key="sidebar_model_select", label_visibility="collapsed",
            )
            st.session_state.active_model = _sel_model
    else:
        st.caption("Chưa có API Key. Vào Cài đặt để thêm.")

st.sidebar.markdown("---")

# Query history
if st.session_state.query_history:
    st.sidebar.caption("LỊCH SỬ QUERY")
    for _i, _q in enumerate(reversed(st.session_state.query_history[-5:])):
        with st.sidebar.expander(f"{_q['timestamp']} · {_q['rows']} dòng"):
            st.code(_q["sql"], language="sql")
            if st.button("Chạy lại", key=f"hist_rerun_{_i}", use_container_width=True):
                st.session_state.messages.append({
                    "role": "user",
                    "content": f"Hãy chạy lại query này:\n```sql\n{_q['sql']}\n```",
                })
                st.session_state.auto_process_last_message = True
                st.session_state.current_page = "chat"
                st.rerun()
    st.sidebar.markdown("---")

# User info + actions
st.sidebar.markdown(f"""
<div class="sidebar-user">
    <div class="uname">👤 {st.session_state.username}</div>
    <div class="urole">Đã đăng nhập</div>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("")

with st.sidebar.expander("🔑 Đổi mật khẩu"):
    with st.form("change_pw_form"):
        _old_pw  = st.text_input("Mật khẩu cũ",        type="password", placeholder="••••••••")
        _new_pw  = st.text_input("Mật khẩu mới",        type="password", placeholder="Tối thiểu 6 ký tự")
        _new_pw2 = st.text_input("Xác nhận mật khẩu mới", type="password", placeholder="Nhập lại")
        _pw_submit = st.form_submit_button("Cập nhật", use_container_width=True, type="primary")
    if _pw_submit:
        if not _old_pw or not _new_pw:
            st.error("Vui lòng nhập đầy đủ")
        elif _new_pw != _new_pw2:
            st.error("Mật khẩu mới không khớp")
        else:
            _ok, _msg = st.session_state.app_db.change_password(
                st.session_state.user_id, _old_pw, _new_pw
            )
            if _ok:
                st.success(_msg)
            else:
                st.error(_msg)

_sc1, _sc2 = st.sidebar.columns(2)
with _sc1:
    if st.button("🗑 Xoá chat", key="clear_chat_btn", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
with _sc2:
    if st.button("Đăng xuất", key="logout_btn", use_container_width=True):
        for _k in list(st.session_state.keys()):
            del st.session_state[_k]
        st.rerun()

# ============================================================
# Resolve active AI vars
# ============================================================
api_key_input = st.session_state.active_api_key
provider      = st.session_state.active_provider
model         = st.session_state.active_model

if api_key_input and not st.session_state.openai_client:
    if provider in ("OpenAI", "Grok (xAI)"):
        _base_url = "https://api.x.ai/v1" if provider == "Grok (xAI)" else None
        st.session_state.openai_client = OpenAI(api_key=api_key_input, base_url=_base_url)

# ============================================================
# PAGE: CÀI ĐẶT
# ============================================================
if st.session_state.current_page == "settings":
    st.markdown('<div class="page-title">⚙️ Cài đặt</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Quản lý kết nối database và API key</div>',
                unsafe_allow_html=True)

    s_tab1, s_tab2 = st.tabs(["🗄 Database", "🔑 API Key & Model"])

    # ---- Tab 1: Database ----
    with s_tab1:
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("##### Kết nối đã lưu")
            saved_conns = st.session_state.app_db.get_db_connections(st.session_state.user_id)
            if saved_conns:
                for conn in saved_conns:
                    with st.container(border=True):
                        ca, cb, cc = st.columns([3, 1, 1])
                        with ca:
                            st.markdown(f"**{conn['profile_name']}**")
                            st.caption(f"{conn['db_user']}@{conn['db_host']}:{conn['db_port']}/{conn['db_name']}")
                        with cb:
                            if st.button("Kết nối", key=f"conn_{conn['id']}"):
                                if not conn.get("db_host"):
                                    st.error("Lỗi: Không thể kết nối. Cấu hình DB bị rớt hoặc lỗi mã hoá.")
                                else:
                                    try:
                                        os.environ["DB_HOST"]     = conn["db_host"]
                                        os.environ["DB_PORT"]     = conn["db_port"]
                                        os.environ["DB_DATABASE"] = conn["db_name"]
                                        os.environ["DB_USER"]     = conn["db_user"]
                                        os.environ["DB_PASSWORD"] = conn["db_password"]
                                        st.session_state.db_client = DatabaseClient()
                                        st.session_state.db_schema = st.session_state.db_client.get_schema_summary()
                                        st.success("Đã kết nối!")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Lỗi: {e}")
                        with cc:
                            if st.button("Xoá", key=f"del_conn_{conn['id']}"):
                                st.session_state.app_db.delete_db_connection(conn["id"], st.session_state.user_id)
                                st.rerun()
            else:
                st.info("Chưa có kết nối. Thêm mới ở bên phải.")

            if st.session_state.db_schema:
                with st.expander("Schema Explorer"):
                    st.text(st.session_state.db_schema)
                if st.button("Kiểm tra kết nối"):
                    try:
                        test_res = st.session_state.db_client.query(
                            "SELECT current_database(), current_user, now() as current_time"
                        )
                        st.success("Kết nối thành công!")
                        st.dataframe(DatabaseClient.to_dataframe(test_res))
                    except Exception as e:
                        st.error(f"Lỗi: {e}")

        with col_r:
            st.markdown("##### Thêm kết nối mới")
            with st.form("db_config_new"):
                new_profile  = st.text_input("Tên kết nối", placeholder="VD: Server Sản xuất")
                db_host      = st.text_input("Host", value="localhost")
                db_port      = st.text_input("Port", value="5432")
                db_name      = st.text_input("Database")
                db_user      = st.text_input("Username")
                db_password  = st.text_input("Password", type="password")
                is_default_db = st.checkbox("Đặt làm mặc định")
                save_connect = st.form_submit_button("Lưu & Kết nối", use_container_width=True, type="primary")
            if save_connect:
                if not new_profile:
                    new_profile = f"{db_name}@{db_host}"
                try:
                    st.session_state.app_db.save_db_connection(
                        user_id=st.session_state.user_id,
                        profile_name=new_profile,
                        db_host=db_host, db_port=db_port,
                        db_name=db_name, db_user=db_user,
                        db_password=db_password, is_default=is_default_db,
                    )
                    os.environ["DB_HOST"]     = db_host
                    os.environ["DB_PORT"]     = db_port
                    os.environ["DB_DATABASE"] = db_name
                    os.environ["DB_USER"]     = db_user
                    os.environ["DB_PASSWORD"] = db_password
                    st.session_state.db_client = DatabaseClient()
                    st.session_state.db_schema = st.session_state.db_client.get_schema_summary()
                    st.success(f"Đã lưu và kết nối: {new_profile}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Lỗi: {e}")

    # ---- Tab 2: API Key & Model ----
    with s_tab2:
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("##### API Keys đã lưu")
            saved_keys = st.session_state.app_db.get_api_keys(st.session_state.user_id)
            if saved_keys:
                for pk in saved_keys:
                    with st.container(border=True):
                        ca, cb = st.columns([3, 1])
                        with ca:
                            st.markdown(f"**{pk['profile_name']}** `{pk['provider']}`")
                            masked = (pk["api_key"][:8] + "..." + pk["api_key"][-4:]
                                      if len(pk["api_key"]) > 12 else "***")
                            st.caption(masked)
                        with cb:
                            if st.button("Xoá", key=f"del_key_{pk['id']}"):
                                st.session_state.app_db.delete_api_key(pk["id"], st.session_state.user_id)
                                st.rerun()
            else:
                st.info("Chưa có API Key. Thêm mới ở bên phải.")

        with col_r:
            st.markdown("##### Thêm API Key mới")
            with st.form("api_key_new"):
                new_key_name  = st.text_input("Tên hiển thị", placeholder="VD: OpenAI Key 1")
                new_api_key   = st.text_input("API Key", type="password", placeholder="sk-...")
                key_provider  = st.selectbox("Provider",
                    ["OpenAI", "Grok (xAI)", "Gemini", "Claude (Anthropic)"])
                is_default_key = st.checkbox("Đặt làm mặc định")
                save_key = st.form_submit_button("Lưu API Key", use_container_width=True, type="primary")
            if save_key and new_api_key:
                if not new_key_name:
                    new_key_name = f"{key_provider} Key"
                success, msg = st.session_state.app_db.save_api_key(
                    user_id=st.session_state.user_id,
                    profile_name=new_key_name,
                    provider=key_provider,
                    api_key=new_api_key,
                    is_default=is_default_key,
                )
                if success:
                    st.success(f"Đã lưu: {new_key_name}")
                    st.rerun()
                else:
                    st.error(msg)

    st.stop()

# ============================================================
# PAGE: DASHBOARD
# ============================================================
elif st.session_state.current_page == "dashboard":
    st.markdown('<div class="page-title">📊 Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Bảng và biểu đồ đã ghim</div>', unsafe_allow_html=True)

    items = load_dashboard()
    if not items:
        st.info("Chưa có mục nào. Ghim bảng hoặc biểu đồ từ trang Chat.")
    else:
        for idx, item in enumerate(reversed(items)):
            with st.expander(
                f"{item.get('timestamp', 'Item')} · {item.get('type', 'table').upper()}",
                expanded=True,
            ):
                if item["type"] == "table":
                    df = pd.DataFrame(item["data"])
                    show_data_widget(df, f"dash_{idx}")
                if st.button("Gỡ bỏ", key=f"del_dash_{idx}"):
                    new_items = [i for i in items if i != item]
                    st.session_state.app_db.save_dashboard(st.session_state.user_id, new_items)
                    st.rerun()
    st.stop()

# ============================================================
# PAGE: EXPLORER
# ============================================================
elif st.session_state.current_page == "explorer":
    st.markdown('<div class="page-title">🔍 Database Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Khám phá cấu trúc và dữ liệu của database</div>',
                unsafe_allow_html=True)

    if not st.session_state.db_client:
        st.info("Vui lòng kết nối Database trước (⚡ Kết nối nhanh ở sidebar).")
    else:
        try:
            tables_res = st.session_state.db_client.query(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema='public' ORDER BY table_name"
            )
            table_names = [r[0] for r in tables_res.rows]
            if not table_names:
                st.warning("Không tìm thấy bảng nào trong schema 'public'.")
            else:
                selected_table = st.selectbox("Chọn bảng:", table_names)
                if selected_table:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"##### Cấu trúc: `{selected_table}`")
                        schema_res = st.session_state.db_client.query(f"""
                            SELECT column_name, data_type, is_nullable
                            FROM information_schema.columns
                            WHERE table_name = '{selected_table}'
                            ORDER BY ordinal_position
                        """)
                        st.dataframe(
                            pd.DataFrame(schema_res.rows, columns=schema_res.columns),
                            use_container_width=True,
                        )
                    with col2:
                        st.markdown(f"##### Dữ liệu mẫu: `{selected_table}` (10 dòng)")
                        sample_res = st.session_state.db_client.query(
                            f"SELECT * FROM {selected_table}", limit=10
                        )
                        sample_df = DatabaseClient.to_dataframe(sample_res)
                        show_data_widget(sample_df, f"explorer_{selected_table}")

                    if st.button(f"AI giải thích bảng `{selected_table}`"):
                        with st.spinner("AI đang phân tích..."):
                            prompt_text = (
                                f"Mô tả ngắn bảng '{selected_table}' với các cột: "
                                f"{', '.join([f'{r[0]} ({r[1]})' for r in schema_res.rows])}. "
                                "Tiếng Việt, ngắn gọn."
                            )
                            if st.session_state.openai_client and model:
                                resp = st.session_state.openai_client.chat.completions.create(
                                    model=model,
                                    messages=[{"role": "user", "content": prompt_text[:1000]}],
                                    max_tokens=300,
                                )
                                st.success(resp.choices[0].message.content)
        except Exception as e:
            st.error(f"Lỗi: {e}")
    st.stop()

# ============================================================
# PAGE: CHAT
# ============================================================

# --- Prerequisites check ---
if not api_key_input:
    st.markdown('<div class="page-title">💬 Chat</div>', unsafe_allow_html=True)
    st.info("⚡ Vui lòng chọn API Key ở sidebar hoặc vào **Cài đặt** để thêm.")
    st.stop()
if not model:
    st.markdown('<div class="page-title">💬 Chat</div>', unsafe_allow_html=True)
    st.warning("Vui lòng chọn Model AI ở sidebar (mục Kết nối nhanh → Lấy Model).")
    st.stop()
if not st.session_state.db_client:
    st.markdown('<div class="page-title">💬 Chat</div>', unsafe_allow_html=True)
    st.info("⚡ Vui lòng kết nối Database ở sidebar (mục Kết nối nhanh).")
    st.stop()

# --- Init system message ---
if not st.session_state.messages:
    semantic_layer_text = ""
    try:
        if os.path.exists("glossary.json"):
            with open("glossary.json", "r", encoding="utf-8") as f:
                glossary = json.load(f)
                terms = "\n".join([
                    f"- {item['term']}: {item['definition']}"
                    for item in glossary.get("business_terms", [])
                ])
                rules = "\n".join([f"- {r}" for r in glossary.get("sql_rules", [])])
                semantic_layer_text = f"\n\n[SEMANTIC LAYER]\n{terms}\n{rules}\n"
    except Exception:
        pass

    st.session_state.messages.append({
        "role": "system",
        "content": (
            "Bạn là một trợ lý SQL chuyên nghiệp, kết nối trực tiếp với cơ sở dữ liệu PostgreSQL. "
            "Bạn phải phản hồi bằng TIẾNG VIỆT tự nhiên và lịch sự. "
            "Luôn chạy truy vấn SQL để trả lời — không giả định kết quả. "
            "Chỉ dùng SELECT (không INSERT, UPDATE, DELETE). "
            "Khi tạo biểu đồ, dùng Altair, gán vào biến 'chart', width=600. "
            "Dữ liệu là DataFrame 'df'. KHÔNG tạo dữ liệu mẫu. "
            "KHÔNG viết Python vào tin nhắn — chỉ dùng tool 'create_chart'. "
            f"Cấu trúc database:\n{st.session_state.db_schema}"
            f"{semantic_layer_text}"
        ),
    })

# --- Page header ---
_db_name = os.getenv("DB_DATABASE", "?")
st.markdown('<div class="page-title">💬 Chat</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="page-subtitle">Model: <b>{model}</b> &nbsp;·&nbsp; DB: <b>{_db_name}</b></div>',
    unsafe_allow_html=True,
)

# --- Suggested questions row ---
if st.session_state.suggestions:
    sug_cols = st.columns(min(len(st.session_state.suggestions), 3))
    for _si, _sq in enumerate(st.session_state.suggestions[:3]):
        with sug_cols[_si]:
            st.markdown('<div class="sug-chip">', unsafe_allow_html=True)
            if st.button(_sq, key=f"sug_{_si}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": _sq})
                st.session_state.auto_process_last_message = True
                st.session_state.skip_user_display = True
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("")

# --- Generate suggestions button ---
if st.button("✨ Gợi ý câu hỏi", key="gen_suggestions"):
    if st.session_state.db_schema:
        with st.spinner("Đang tạo gợi ý..."):
            try:
                _schema_short  = st.session_state.db_schema[:600]
                _context_json  = json.dumps(st.session_state.current_context, ensure_ascii=False)
                _prompt_sug = (
                    f"Schema: {_schema_short}\nContext: {_context_json}\n"
                    "Đề xuất 3 câu hỏi ngắn về địa chính/lâm nghiệp bằng tiếng Việt. Chỉ list."
                )
                if provider == "Gemini":
                    import google.generativeai as genai
                    genai.configure(api_key=api_key_input)
                    _resp_text = genai.GenerativeModel(model).generate_content(_prompt_sug).text
                elif provider == "Claude (Anthropic)":
                    _resp_text = "Diện tích rừng theo xã?\nSố lô đất theo loại rừng?\nThống kê theo năm?"
                elif st.session_state.openai_client:
                    _resp = st.session_state.openai_client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": _prompt_sug[:1000]}],
                        max_tokens=200,
                    )
                    _resp_text = _resp.choices[0].message.content
                else:
                    _resp_text = ""
                st.session_state.suggestions = [
                    line.strip().strip("-*•1234567890. ")
                    for line in _resp_text.split("\n") if line.strip()
                ][:3]
                st.rerun()
            except Exception as e:
                st.error(f"Lỗi: {e}")

st.markdown("---")

# --- Display chat history ---
for _mi, message in enumerate(st.session_state.messages):
    if message["role"] == "system":
        continue
    if message["role"] == "tool":
        continue
    if message["role"] == "user" and "viết TRỰC TIẾP câu lệnh SQL" in message.get("content", ""):
        continue
    if message["role"] == "assistant" and "tool_calls" in message:
        continue

    with st.chat_message(message["role"]):
        if message.get("content"):
            st.markdown(message["content"])

        if "sql_query" in message:
            with st.expander("🔍 SQL đã chạy"):
                st.code(message["sql_query"], language="sql")

        if "data" in message:
            df_data = pd.DataFrame(message["data"])
            show_data_widget(df_data, f"hist_{_mi}")

        if "chart_code" in message:
            try:
                _executor = SafeishPythonExecutor(safe_globals={"alt": alt, "pd": pd})
                _df_hist   = pd.DataFrame(message.get("data", []))
                _res_hist  = _executor.run(message["chart_code"], context={"df": _df_hist}, return_locals=True)
                if _res_hist.ok and _res_hist.locals.get("chart"):
                    st.altair_chart(_res_hist.locals["chart"], use_container_width=True)
            except Exception:
                pass

# --- Chat input ---
if _prompt := st.chat_input("Hỏi về dữ liệu của bạn..."):
    st.session_state.messages.append({"role": "user", "content": _prompt})
    st.session_state.auto_process_last_message = True

# --- Process message ---
if st.session_state.get("auto_process_last_message", False) and st.session_state.messages:
    _last_msg = st.session_state.messages[-1]
    if _last_msg["role"] == "user":
        prompt = _last_msg["content"]
        st.session_state.auto_process_last_message = False

        if not st.session_state.get("skip_user_display", False):
            with st.chat_message("user"):
                st.markdown(prompt)
        if "skip_user_display" in st.session_state:
            del st.session_state["skip_user_display"]

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("⏳ Đang xử lý...")

        try:
            # ---- Gemini ----
            if provider == "Gemini":
                import google.generativeai as genai
                genai.configure(api_key=api_key_input)

                def query_db_wrapper(sql: str):
                    return query_database(sql)

                def create_chart_wrapper(python_code: str):
                    return create_chart(python_code)

                gemini_model_obj = genai.GenerativeModel(
                    model_name=model, tools=[query_db_wrapper, create_chart_wrapper]
                )
                gemini_history = []
                for msg in st.session_state.messages[:-1]:
                    role = "user" if msg["role"] == "user" else "model"
                    gemini_history.append({"role": role, "parts": [{"text": msg.get("content") or ""}]})

                chat_session = gemini_model_obj.start_chat(history=gemini_history)
                response = chat_session.send_message(prompt)

                for part in response.candidates[0].content.parts:
                    if fn := part.function_call:
                        fn_name = fn.name
                        fn_args = dict(fn.args)
                        if fn_name == "query_db_wrapper":
                            sql_q = fn_args["sql"]
                            with st.expander("🔍 SQL"):
                                st.code(sql_q, language="sql")
                            result = query_database(sql_q)
                            if "last_df" in st.session_state:
                                df = st.session_state.last_df
                                show_data_widget(df, f"gem_{id(df)}")
                                st.session_state.current_sql  = sql_q
                                st.session_state.current_data = df.to_dict("records")
                            response = chat_session.send_message(genai.types.Content(
                                parts=[genai.types.FunctionResponse(name=fn_name, response={"result": result})]
                            ))
                        elif fn_name == "create_chart_wrapper":
                            result = create_chart(fn_args["python_code"])
                            if "last_chart" in st.session_state:
                                st.altair_chart(st.session_state.last_chart, use_container_width=True)
                                st.session_state.current_chart = fn_args["python_code"]
                            response = chat_session.send_message(genai.types.Content(
                                parts=[genai.types.FunctionResponse(name=fn_name, response={"result": result})]
                            ))

                final_text = response.text
                message_placeholder.markdown(final_text)
                final_msg = {"role": "assistant", "content": final_text}
                if "current_sql"   in st.session_state: final_msg["sql_query"]  = st.session_state.current_sql
                if "current_data"  in st.session_state: final_msg["data"]        = st.session_state.current_data
                if "current_chart" in st.session_state: final_msg["chart_code"]  = st.session_state.current_chart
                st.session_state.messages.append(final_msg)
                for _k in ["current_sql", "current_data", "current_chart"]:
                    st.session_state.pop(_k, None)

            # ---- Claude (Anthropic) ----
            elif provider == "Claude (Anthropic)":
                import anthropic
                anthropic_client = anthropic.Anthropic(api_key=api_key_input)

                claude_messages = []
                for msg in st.session_state.messages[:-1]:
                    if msg["role"] == "user":
                        claude_messages.append({"role": "user", "content": msg["content"]})
                    elif msg["role"] == "assistant":
                        claude_messages.append({"role": "assistant", "content": msg["content"]})

                claude_tools = [
                    {"name": "query_database",
                     "description": "Query the PostgreSQL database",
                     "input_schema": {"type": "object",
                                      "properties": {"sql": {"type": "string"}},
                                      "required": ["sql"]}},
                    {"name": "create_chart",
                     "description": "Create an Altair chart",
                     "input_schema": {"type": "object",
                                      "properties": {"python_code": {"type": "string"}},
                                      "required": ["python_code"]}},
                ]
                response = anthropic_client.messages.create(
                    model=model, max_tokens=4096,
                    system="You are a data assistant. Query PostgreSQL and create charts.",
                    messages=claude_messages, tools=claude_tools,
                )

                while response.stop_reason == "tool_use":
                    tool_use     = next(b for b in response.content if b.type == "tool_use")
                    tool_name    = tool_use.name
                    tool_input   = tool_use.input

                    if tool_name == "query_database":
                        sql_q = tool_input["sql"]
                        with st.expander("🔍 SQL"):
                            st.code(sql_q, language="sql")
                        result = query_database(sql_q)
                        if "last_df" in st.session_state:
                            df = st.session_state.last_df
                            show_data_widget(df, f"cl_{id(df)}")
                            st.session_state.current_sql  = sql_q
                            st.session_state.current_data = df.to_dict("records")
                        tool_result_content = result
                    elif tool_name == "create_chart":
                        result = create_chart(tool_input["python_code"])
                        if "last_chart" in st.session_state:
                            st.altair_chart(st.session_state.last_chart, use_container_width=True)
                            st.session_state.current_chart = tool_input["python_code"]
                        tool_result_content = result

                    claude_messages.append({"role": "assistant", "content": response.content})
                    claude_messages.append({
                        "role": "user",
                        "content": [{"type": "tool_result",
                                     "tool_use_id": tool_use.id,
                                     "content": tool_result_content}],
                    })
                    response = anthropic_client.messages.create(
                        model=model, max_tokens=4096,
                        system="You are a data assistant.",
                        messages=claude_messages, tools=claude_tools,
                    )

                final_text = response.content[0].text
                message_placeholder.markdown(final_text)
                final_msg = {"role": "assistant", "content": final_text}
                if "current_sql"   in st.session_state: final_msg["sql_query"]  = st.session_state.current_sql
                if "current_data"  in st.session_state: final_msg["data"]        = st.session_state.current_data
                if "current_chart" in st.session_state: final_msg["chart_code"]  = st.session_state.current_chart
                st.session_state.messages.append(final_msg)
                for _k in ["current_sql", "current_data", "current_chart"]:
                    st.session_state.pop(_k, None)

            # ---- OpenAI / Grok ----
            else:
                use_tools = True
                while True:
                    try:
                        specialized_system = get_system_prompt()
                        max_messages       = 8
                        msgs_to_send = (
                            st.session_state.messages[-max_messages:]
                            if len(st.session_state.messages) > max_messages
                            else st.session_state.messages
                        )
                        truncated_messages = [{"role": "system", "content": specialized_system}]
                        for msg in msgs_to_send:
                            content     = str(msg.get("content", ""))[:2000] if msg.get("content") else ""
                            msg_payload = {"role": msg["role"], "content": content}
                            if "tool_calls" in msg:
                                tc_list = []
                                for tc in msg["tool_calls"]:
                                    if hasattr(tc, "model_dump"):
                                        dumped       = tc.model_dump()
                                        dumped["id"] = tc.id
                                        tc_list.append(dumped)
                                    elif hasattr(tc, "id"):
                                        tc_list.append({
                                            "id": tc.id, "type": "function",
                                            "function": {
                                                "name": tc.function.name,
                                                "arguments": getattr(tc.function, "arguments", "{}"),
                                            },
                                        })
                                    else:
                                        tc_list.append(tc)
                                msg_payload["tool_calls"] = tc_list
                            if "tool_call_id" in msg:
                                msg_payload["tool_call_id"] = msg["tool_call_id"]
                            if msg["role"] == "tool" and "name" in msg:
                                msg_payload["name"] = msg["name"]
                            truncated_messages.append(msg_payload)

                        api_params = {
                            "model": model,
                            "messages": truncated_messages,
                            "max_tokens": 1500,
                        }
                        if use_tools:
                            api_params["tools"]       = tools
                            api_params["tool_choice"] = "auto"

                        response         = st.session_state.openai_client.chat.completions.create(**api_params)
                        response_message = response.choices[0].message

                        if response_message.tool_calls:
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": response_message.content,
                                "tool_calls": response_message.tool_calls,
                            })
                            for tool_call in response_message.tool_calls:
                                fn_name = tool_call.function.name
                                fn_args = json.loads(tool_call.function.arguments)

                                if fn_name == "query_database":
                                    sql_q = fn_args["sql"]
                                    with st.expander("🔍 SQL đã chạy"):
                                        st.code(sql_q, language="sql")
                                    result_text = query_database(sql_q)
                                    if "last_df" in st.session_state:
                                        df = st.session_state.last_df
                                        show_data_widget(df, f"new_{id(df)}")
                                        st.session_state.current_sql  = sql_q
                                        st.session_state.current_data = df.to_dict("records")
                                    st.session_state.messages.append({
                                        "role": "tool",
                                        "tool_call_id": tool_call.id,
                                        "name": fn_name,
                                        "content": str(result_text),
                                    })

                                elif fn_name == "create_chart":
                                    result_text = create_chart(fn_args["python_code"])
                                    if "last_chart" in st.session_state:
                                        st.altair_chart(st.session_state.last_chart, use_container_width=True)
                                        st.session_state.current_chart = fn_args["python_code"]
                                    st.session_state.messages.append({
                                        "role": "tool",
                                        "tool_call_id": tool_call.id,
                                        "name": fn_name,
                                        "content": str(result_text),
                                    })
                            continue  # loop for next response

                        # Final text response
                        assistant_content = response_message.content or ""
                        final_content     = update_memory(assistant_content)

                        # Fallback: extract SQL from text if tools not supported
                        if not use_tools:
                            extracted_sql = DatabaseClient.extract_sql(final_content)
                            if extracted_sql:
                                with st.expander("🔍 SQL phát hiện"):
                                    st.code(extracted_sql, language="sql")
                                try:
                                    query_database(extracted_sql)
                                    if "last_df" in st.session_state:
                                        df = st.session_state.last_df
                                        if not df.empty:
                                            show_data_widget(df, f"fallback_{id(df)}")
                                            st.session_state.current_sql  = extracted_sql
                                            st.session_state.current_data = df.to_dict("records")
                                except Exception as sql_err:
                                    st.error(f"Lỗi SQL: {sql_err}")

                        message_placeholder.markdown(final_content)
                        final_msg = {"role": "assistant", "content": final_content}
                        if "current_sql"   in st.session_state: final_msg["sql_query"]  = st.session_state.current_sql
                        if "current_data"  in st.session_state: final_msg["data"]        = st.session_state.current_data
                        if "current_chart" in st.session_state: final_msg["chart_code"]  = st.session_state.current_chart
                        st.session_state.messages.append(final_msg)
                        for _k in ["current_sql", "current_data", "current_chart"]:
                            st.session_state.pop(_k, None)
                        break

                    except Exception as e:
                        err_lower = str(e).lower()
                        if ("tools" in err_lower or "404" in err_lower or "not supported" in err_lower) and use_tools:
                            st.warning("Model không hỗ trợ Function Calling. Chuyển chế độ text...")
                            message_placeholder.markdown("Đang thử lại (Fallback)...")
                            use_tools = False
                            st.session_state.messages.append({
                                "role": "user",
                                "content": "Tool không khả dụng. Viết TRỰC TIẾP câu lệnh SQL vào block code (```sql ... ```).",
                            })
                            continue
                        raise e

        except Exception as e:
            st.error(f"Lỗi: {str(e)}")
            message_placeholder.markdown(f"Lỗi: {str(e)}")
