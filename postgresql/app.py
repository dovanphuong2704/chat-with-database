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
    page_title="Trò chuyện với Cơ sở dữ liệu PostgreSQL",
    page_icon="🗄️",
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
if "openai_client" not in st.session_state:
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        st.session_state.openai_client = OpenAI(api_key=api_key)
    else:
        st.session_state.openai_client = None

# Sidebar - Database Configuration
st.sidebar.title("🗄️ Cấu hình Cơ sở dữ liệu")

with st.sidebar.form("db_config"):
    db_host = st.text_input("Host", value=os.getenv("DB_HOST", "localhost"))
    db_port = st.text_input("Cổng (Port)", value=os.getenv("DB_PORT", "5432"))
    db_name = st.text_input("Tên Database", value=os.getenv("DB_DATABASE", ""))
    db_user = st.text_input("Tên đăng nhập", value=os.getenv("DB_USER", ""))
    db_password = st.text_input("Mật khẩu", value=os.getenv("DB_PASSWORD", ""), type="password")
    
    connect_button = st.form_submit_button("🔌 Kết nối Database")

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
        
        st.sidebar.success("✅ Kết nối thành công!")
    except Exception as e:
        st.sidebar.error(f"❌ Kết nối thất bại: {str(e)}")
        st.session_state.db_client = None
        st.session_state.db_schema = None

# Show database schema if connected
if st.session_state.db_schema:
    with st.sidebar.expander("📋 Cấu trúc Database", expanded=False):
        st.text(st.session_state.db_schema)

# OpenAI API Key configuration
st.sidebar.markdown("---")
st.sidebar.title("🤖 OpenAI Configuration")

api_key_input = st.sidebar.text_input(
    "API Key",
    value=os.getenv("OPENAI_API_KEY", ""),
    type="password",
    help="Enter your OpenAI API key"
)

if api_key_input and api_key_input != os.getenv("OPENAI_API_KEY", ""):
    st.session_state.openai_client = OpenAI(api_key=api_key_input)
    st.sidebar.success("✅ API Key set!")

# Model selection
model = st.sidebar.selectbox(
    "Model",
    ["gpt-5","gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
    index=0,
    help="Select OpenAI model to use"
)

# Main area
st.title("💬 Chat với Database PostgreSQL")
st.markdown("Đặt câu hỏi về dữ liệu của bạn bằng ngôn ngữ tự nhiên!")

# Check prerequisites
if not st.session_state.openai_client:
    st.warning("⚠️ Vui lòng cấu hình OpenAI API key ở thanh bên.")
    st.stop()

if not st.session_state.db_client:
    st.info("ℹ️ Vui lòng kết nối với database ở thanh bên.")
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
            with st.expander("⚒️ Truy vấn SQL"):
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
        message_placeholder.markdown("🤔 Đang suy nghĩ...")
        
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
                        with st.expander("⚒️ Truy vấn SQL"):
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
st.sidebar.markdown("### 💡 Mẹo nhỏ")
st.sidebar.markdown("""
- Đặt câu hỏi bằng tiếng Việt tự nhiên
- Yêu cầu vẽ biểu đồ hoặc biểu diễn dữ liệu
- AI sẽ tự động tạo mã SQL và truy vấn
- Tất cả truy vấn đều là Read-only (chỉ đọc) để đảm bảo an toàn
""")

# Clear chat button
if st.sidebar.button("🗑️ Xóa lịch sử trò chuyện"):
    st.session_state.messages = []
    st.rerun()
