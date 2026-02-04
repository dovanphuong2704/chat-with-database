import streamlit as st
import pandas as pd
import altair as alt
import os
from typing import Optional
from dotenv import load_dotenv
from openai import OpenAI

from dbclient import DatabaseClient
from safeish import SafeishPythonExecutor

# Load environment variables
load_dotenv("env")

# Page config
st.set_page_config(
    page_title="Data Intelligence Platform",
    page_icon="terminal",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "db_client" not in st.session_state:
    st.session_state.db_client = None
if "db_schema" not in st.session_state:
    st.session_state.db_schema = None
if "available_models" not in st.session_state:
    st.session_state.available_models = [] # Start empty
if "openai_client" not in st.session_state:
    st.session_state.openai_client = None

# Helper function to fetch models based on provider
def fetch_available_models(provider, api_key):
    try:
        if provider == "OpenAI" or provider == "Grok (xAI)":
            # Grok is OpenAI compatible
            base_url = "https://api.x.ai/v1" if provider == "Grok (xAI)" else None
            temp_client = OpenAI(api_key=api_key, base_url=base_url)
            models = temp_client.models.list()
            # Filter for models that support tools (function calling)
            model_list = []
            for m in models:
                mid = m.id.lower()
                # Known tool-supporting models
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
            return sorted([m.name.replace('models/', '') for m in models if 'generateContent' in m.supported_generation_methods])
        return []
    except Exception as e:
        st.sidebar.error(f"Error fetching models: {str(e)}")
        return []

# Sidebar - Database Configuration
st.sidebar.title("Configuration")

with st.sidebar.form("db_config"):
    db_host = st.text_input("Host", value=os.getenv("DB_HOST", "localhost"))
    db_port = st.text_input("Port", value=os.getenv("DB_PORT", "5432"))
    db_name = st.text_input("Database", value=os.getenv("DB_DATABASE", ""))
    db_user = st.text_input("Username", value=os.getenv("DB_USER", ""))
    db_password = st.text_input("Password", value=os.getenv("DB_PASSWORD", ""), type="password")
    
    connect_button = st.form_submit_button("Connect Database")

if connect_button:
    try:
        # Set environment variables for DatabaseClient
        os.environ["DB_HOST"] = db_host
        os.environ["DB_PORT"] = db_port
        os.environ["DB_DATABASE"] = db_name
        os.environ["DB_USER"] = db_user
        os.environ["DB_PASSWORD"] = db_password
        
        # Create database client
        st.session_state.db_client = DatabaseClient()
        st.session_state.db_schema = st.session_state.db_client.get_schema_summary()
        
        st.sidebar.success("Thành công: Đã kết nối Database")
    except Exception as e:
        st.sidebar.error(f"Lỗi: Kết nối thất bại: {str(e)}")
        st.session_state.db_client = None
        st.session_state.db_schema = None

# Show database schema if connected
if st.session_state.db_schema:
    with st.sidebar.expander("Schema Explorer", expanded=False):
        st.text(st.session_state.db_schema)

# AI Provider Configuration
st.sidebar.markdown("---")
st.sidebar.title("AI Engine")
provider = st.sidebar.selectbox(
    "Provider",
    ["OpenAI", "Grok (xAI)", "Gemini", "Claude (Anthropic)"],
    index=0
)

api_key_input = st.sidebar.text_input(
    f"API Key {provider}",
    value=os.getenv(f"{provider.upper().replace(' ', '_')}_API_KEY", ""),
    type="password",
    help=f"Nhập mã API Key của {provider}"
)

# Fetch models button
if st.sidebar.button("Lấy danh sách Model"):
    if api_key_input:
        with st.sidebar.status("Đang kết nối API...", expanded=False):
            models = fetch_available_models(provider, api_key_input)
            if models:
                st.session_state.available_models = models
                st.sidebar.success(f"Đã tải {len(models)} model")
            else:
                st.sidebar.warning("Không tìm thấy model hoặc có lỗi xảy ra")
    else:
        st.sidebar.error("Vui lòng nhập API Key trước")

# Model selection from dynamic list
model = None
if st.session_state.available_models:
    model = st.sidebar.selectbox(
        "Active Model",
        st.session_state.available_models,
        index=0
    )
else:
    st.sidebar.info("Vui lòng tải danh sách model để tiếp tục")

# Initialize Client based on provider
if api_key_input:
    if provider == "OpenAI" or provider == "Grok (xAI)":
        base_url = "https://api.x.ai/v1" if provider == "Grok (xAI)" else None
        st.session_state.openai_client = OpenAI(api_key=api_key_input, base_url=base_url)
    # Note: Gemini/Claude initialization will be handled in the chat logic
else:
    st.session_state.openai_client = None

# Main area
st.title("AI Data Intelligence Platform")
st.markdown("Khai thác sức mạnh dữ liệu của bạn thông qua ngôn ngữ tự nhiên. Hệ thống sẽ tự động phân tích và trực quan hóa kết quả cho bạn.")

# Check prerequisites
if not st.session_state.openai_client:
    st.warning("Warning: Please configure API key in sidebar.")
    st.stop()

if not st.session_state.db_client:
    st.info("Info: Please connect to database in sidebar.")
    st.stop()

# Function definitions for OpenAI function calling
def query_database(sql: str) -> str:
    """Query database and return results as JSON string
    
    Args:
        sql: SQL query to execute
        
    Returns:
        JSON string with query results
    """
    try:
        result = st.session_state.db_client.query(sql)
        df = DatabaseClient.to_dataframe(result)
        
        # Store in session state for chart generation
        st.session_state.last_df = df
        
        return DatabaseClient.describe_dataframe_for_llm(df)
    except Exception as e:
        return f"Error executing query: {str(e)}"

def create_chart(python_code: str) -> str:
    """Execute Python code to create an Altair chart
    
    Args:
        python_code: Python code that creates an Altair chart
        
    Returns:
        Status message
    """
    try:
        executor = SafeishPythonExecutor(safe_globals={"alt": alt, "pd": pd})
        
        df = st.session_state.get("last_df", pd.DataFrame())
        
        res = executor.run(
            python_code,
            context={"df": df},
            return_locals=True,
        )
        
        if res.ok:
            chart = res.locals.get("chart")
            if chart:
                st.session_state.last_chart = chart
                return "Chart created successfully"
            else:
                return "No chart variable found in code"
        else:
            return f"Error creating chart: {res.error}"
    except Exception as e:
        return f"Error: {str(e)}"

# OpenAI function calling tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "query_database",
            "description": "Query the PostgreSQL database with a SELECT statement and return results",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "SQL SELECT query to execute"
                    }
                },
                "required": ["sql"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_chart",
            "description": "Create an Altair chart from the last query result. The dataframe is available as 'df' variable.",
            "parameters": {
                "type": "object",
                "properties": {
                    "python_code": {
                        "type": "string",
                        "description": "Python code that creates an Altair chart and assigns it to variable 'chart'"
                    }
                },
                "required": ["python_code"]
            }
        }
    }
]

# Initialize system message
if not st.session_state.messages:
    system_message = {
        "role": "system",
        "content": (
            "Bạn là một trợ lý SQL chuyên nghiệp, kết nối trực tiếp với cơ sở dữ liệu PostgreSQL. "
            "Bạn phải phản hồi người dùng bằng TIẾNG VIỆT một cách tự nhiên và lịch sự. "
            "Nhiệm vụ của bạn là thực thi các câu lệnh SELECT trên cơ sở dữ liệu này để trả lời các câu hỏi. "
            "Hãy luôn cố gắng trả lời bằng cách tạo và chạy truy vấn SQL trước, ngay cả khi bạn nghĩ rằng mình đã biết câu trả lời. "
            "Không bao giờ giả định kết quả — luôn xác minh trong database. "
            "Nếu câu hỏi không thể trả lời bằng SQL, hãy yêu cầu làm rõ bằng tiếng Việt. "
            "Chỉ sử dụng câu lệnh SELECT (không dùng INSERT, UPDATE, DELETE). "
            "Khi tạo biểu đồ, hãy sử dụng thư viện Altair. "
            "Tạo đối tượng biểu đồ và gán cho biến 'chart'. Đặt chiều rộng là 600px. "
            "Dữ liệu nằm trong dataframe pandas tên là 'df'. KHÔNG tạo dữ liệu mẫu. "
            "QUAN TRỌNG: KHÔNG ĐƯỢC viết mã Python trực tiếp vào tin nhắn phản hồi. "
            "Bạn CHỈ ĐƯỢC phép tạo biểu đồ thông qua công cụ 'create_chart'. "
            "Nếu bạn viết mã Python vào tin nhắn thay vì dùng công cụ, người dùng sẽ không thấy biểu đồ. "
            f"Cấu trúc database hiện tại:\n{st.session_state.db_schema}"
        )
    }
    st.session_state.messages.append(system_message)

# Display chat messages
for message in st.session_state.messages:
    if message["role"] == "system":
        continue
    
    # Skip tool messages (internal only)
    if message["role"] == "tool":
        continue
    
    # Skip assistant messages with tool_calls (internal only)
    if message["role"] == "assistant" and "tool_calls" in message:
        continue
    
    with st.chat_message(message["role"]):
        if message.get("content"):
            st.markdown(message["content"])
        
        # Display SQL query if present in history
        if "sql_query" in message:
            with st.expander("SQL Query"):
                st.code(message["sql_query"], language="sql")
        
        # Display Dataframe from history
        if "data" in message:
            st.dataframe(pd.DataFrame(message["data"]), use_container_width=True)
            
        # Display Chart from history
        if "chart_code" in message:
            try:
                executor = SafeishPythonExecutor(safe_globals={"alt": alt, "pd": pd})
                df = pd.DataFrame(message.get("data", []))
                res = executor.run(message["chart_code"], context={"df": df}, return_locals=True)
                if res.ok and res.locals.get("chart"):
                    st.altair_chart(res.locals.get("chart"), use_container_width=True)
            except Exception:
                pass

# Chat input
if prompt := st.chat_input("Hỏi tôi bất cứ điều gì về dữ liệu..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get AI response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Processing query...")
        
        try:
            # Loop to handle sequential tool calls
            while True:
                response = st.session_state.openai_client.chat.completions.create(
                    model=model,
                    messages=st.session_state.messages,
                    tools=tools,
                    tool_choice="auto"
                )
                
                response_message = response.choices[0].message
                
                # If there are no tool calls, this is the final response
                if not response_message.tool_calls:
                    assistant_content = response_message.content or ""
                    message_placeholder.markdown(assistant_content)
                    
                    # Create the final assistant message for history
                    final_assistant_msg = {"role": "assistant", "content": assistant_content}
                    
                    # Transfer metadata from session state (captured during tool execution)
                    if "current_sql" in st.session_state:
                        final_assistant_msg["sql_query"] = st.session_state.current_sql
                    if "current_data" in st.session_state:
                        final_assistant_msg["data"] = st.session_state.current_data
                    if "current_chart" in st.session_state:
                        final_assistant_msg["chart_code"] = st.session_state.current_chart
                        
                    st.session_state.messages.append(final_assistant_msg)
                    
                    # Clean up temporary storage
                    for key in ["current_sql", "current_data", "current_chart"]:
                        if key in st.session_state: del st.session_state[key]
                    break
                
                # Process tool calls
                # First, add the assistant's request to tool calls to the history (OpenAI requirement)
                st.session_state.messages.append(response_message)
                
                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = eval(tool_call.function.arguments)
                    
                    if function_name == "query_database":
                        sql_query = function_args["sql"]
                        with st.expander("Executed SQL"):
                            st.code(sql_query, language="sql")
                        
                        result_text = query_database(sql_query)
                        
                        if "last_df" in st.session_state:
                            df = st.session_state.last_df
                            st.dataframe(df, use_container_width=True)
                            st.session_state.current_sql = sql_query
                            st.session_state.current_data = df.to_dict('records')
                        
                        st.session_state.messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": function_name,
                            "content": result_text
                        })
                    
                    elif function_name == "create_chart":
                        python_code = function_args["python_code"]
                        result_text = create_chart(python_code)
                        
                        if "last_chart" in st.session_state:
                            st.altair_chart(st.session_state.last_chart, use_container_width=True)
                            st.session_state.current_chart = python_code
                        
                        st.session_state.messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": function_name,
                            "content": result_text
                        })
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            message_placeholder.markdown(f"❌ Error: {str(e)}")

# Sidebar footer
st.sidebar.markdown("---")
st.sidebar.title("Hướng dẫn vận hành")
st.sidebar.markdown("""
- Đặt câu hỏi bằng ngôn ngữ tự nhiên
- Tự động hóa hình ảnh dữ liệu
- Tự động tạo truy vấn SQL chuẩn
- Lớp truy cập dữ liệu chỉ đọc (an toàn)
""")

# Clear chat button
if st.sidebar.button("Bắt đầu lại: Xóa lịch sử"):
    st.session_state.messages = []
    st.rerun()
