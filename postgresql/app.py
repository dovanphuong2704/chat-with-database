import streamlit as st
import pandas as pd
import altair as alt
import os
import json
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
if "query_history" not in st.session_state:
    st.session_state.query_history = []

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
# Query History in Sidebar
if st.session_state.query_history:
    st.sidebar.markdown("---")
    st.sidebar.title("🕒 Lịch sử Query")
    
    # Show last 10 queries
    for i, q in enumerate(reversed(st.session_state.query_history[-10:])):
        with st.sidebar.expander(f"{q['timestamp']} - {q['rows']} dòng"):
            st.code(q['sql'], language="sql")
            if st.button("🔄 Chạy lại", key=f"rerun_{len(st.session_state.query_history)-i}"):
                st.session_state.messages.append({"role": "user", "content": f"Hãy chạy lại query này giúp tôi:\n```sql\n{q['sql']}\n```"})
                st.rerun()

st.title("AI Data Intelligence Platform")
st.markdown("Khai thác sức mạnh dữ liệu của bạn thông qua ngôn ngữ tự nhiên. Hệ thống sẽ tự động phân tích và trực quan hóa kết quả cho bạn.")

# Check prerequisites
if not api_key_input:
    st.warning("Warning: Please configure API key in sidebar.")
    st.stop()

if not model:
    st.warning("Warning: Please fetch and select a model in sidebar.")
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
        
        # Add to query history
        from datetime import datetime
        st.session_state.query_history.append({
            "sql": sql,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "rows": len(df)
        })
        
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
    
    # Hide fallback instruction messages from UI
    if message["role"] == "user" and "viết TRỰC TIẾP câu lệnh SQL" in message.get("content", ""):
        continue
    
    # Skip assistant messages with tool_calls (internal only)
    if message["role"] == "assistant" and "tool_calls" in message:
        continue
    
    with st.chat_message(message["role"]):
        if message.get("content"):
            st.markdown(message["content"])
        
        # Display SQL query if present in history
        if "sql_query" in message:
            with st.expander("🛠️ SQL Query"):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.code(message["sql_query"], language="sql")
                with col2:
                    if st.button("📋 Copy", key=f"copy_sql_{id(message)}", help="Copy SQL to clipboard"):
                        st.write("")  # Streamlit auto-copies from code block when clicked
        
        # Display Dataframe from history
        if "data" in message:
            df_data = pd.DataFrame(message["data"])
            st.dataframe(df_data, use_container_width=True)
            
            # Add download buttons (multiple formats)
            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv_data = df_data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 CSV",
                    data=csv_data,
                    file_name=f"data_export.csv",
                    mime='text/csv',
                    key=f"csv_history_{id(message)}"
                )
            
            with col2:
                # Excel export
                from io import BytesIO
                buffer = BytesIO()
                df_data.to_excel(buffer, index=False, engine='openpyxl')
                excel_data = buffer.getvalue()
                st.download_button(
                    label="📊 Excel",
                    data=excel_data,
                    file_name=f"data_export.xlsx",
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    key=f"excel_history_{id(message)}"
                )
            
            with col3:
                json_data = df_data.to_json(orient='records', indent=2).encode('utf-8')
                st.download_button(
                    label="📄 JSON",
                    data=json_data,
                    file_name=f"data_export.json",
                    mime='application/json',
                    key=f"json_history_{id(message)}"
                )
            
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
            if provider == "Gemini":
                import google.generativeai as genai
                genai.configure(api_key=api_key_input)
                
                # Setup Gemini tools
                # We need to wrap our local functions for Gemini
                def query_db_wrapper(sql: str):
                    return query_database(sql)
                
                def create_chart_wrapper(python_code: str):
                    return create_chart(python_code)

                gemini_model = genai.GenerativeModel(
                    model_name=model,
                    tools=[query_db_wrapper, create_chart_wrapper]
                )
                
                # Convert history to Gemini format
                gemini_history = []
                for msg in st.session_state.messages[:-1]:
                    role = "user" if msg["role"] == "user" else "model"
                    gemini_history.append({"role": role, "parts": [{"text": msg["content"] or ""}]})
                
                chat = gemini_model.start_chat(history=gemini_history)
                response = chat.send_message(prompt)
                
                # Final content to display
                final_text = ""
                
                # Process parts for tool calls
                for part in response.candidates[0].content.parts:
                    if fn := part.function_call:
                        function_name = fn.name
                        function_args = dict(fn.args)
                        
                        if function_name == "query_db_wrapper":
                            sql_query = function_args["sql"]
                            with st.expander("Executed SQL"):
                                st.code(sql_query, language="sql")
                            
                            result = query_database(sql_query)
                            
                            if "last_df" in st.session_state:
                                df = st.session_state.last_df
                                st.dataframe(df, use_container_width=True)
                                st.session_state.current_sql = sql_query
                                st.session_state.current_data = df.to_dict('records')
                            
                            response = chat.send_message(
                                genai.types.Content(
                                    parts=[genai.types.FunctionResponse(name=function_name, response={'result': result})]
                                )
                            )
                        elif function_name == "create_chart_wrapper":
                            python_code = function_args["python_code"]
                            result = create_chart(python_code)
                            
                            if "last_chart" in st.session_state:
                                st.altair_chart(st.session_state.last_chart, use_container_width=True)
                                st.session_state.current_chart = python_code
                                
                            response = chat.send_message(
                                genai.types.Content(
                                    parts=[genai.types.FunctionResponse(name=function_name, response={'result': result})]
                                )
                            )
                
                final_text = response.text
                message_placeholder.markdown(final_text)
                
                # Save to history
                final_assistant_msg = {"role": "assistant", "content": final_text}
                if "current_sql" in st.session_state: final_assistant_msg["sql_query"] = st.session_state.current_sql
                if "current_data" in st.session_state: final_assistant_msg["data"] = st.session_state.current_data
                if "current_chart" in st.session_state: final_assistant_msg["chart_code"] = st.session_state.current_chart
                st.session_state.messages.append(final_assistant_msg)
                
                # Clean up
            elif provider == "Claude (Anthropic)":
                import anthropic
                anthropic_client = anthropic.Anthropic(api_key=api_key_input)
                
                # Convert messages to Claude format
                claude_messages = []
                system_prompt = "You are a data assistant. You can query a PostgreSQL database and create charts."
                for msg in st.session_state.messages[:-1]:
                    if msg["role"] == "user":
                        claude_messages.append({"role": "user", "content": msg["content"]})
                    elif msg["role"] == "assistant":
                        claude_messages.append({"role": "assistant", "content": msg["content"]})

                # Define tools for Claude
                claude_tools = [
                    {
                        "name": "query_database",
                        "description": "Query the PostgreSQL database",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "sql": {"type": "string", "description": "The SQL query to run"}
                            },
                            "required": ["sql"]
                        }
                    },
                    {
                        "name": "create_chart",
                        "description": "Create an Altair chart from the data",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "python_code": {"type": "string", "description": "The Python code for the chart"}
                            },
                            "required": ["python_code"]
                        }
                    }
                ]

                # Initial request to Claude
                response = anthropic_client.messages.create(
                    model=model,
                    max_tokens=4096,
                    system=system_prompt,
                    messages=claude_messages,
                    tools=claude_tools
                )

                while response.stop_reason == "tool_use":
                    # Handle tool calls
                    tool_use = next(block for block in response.content if block.type == "tool_use")
                    tool_name = tool_use.name
                    tool_input = tool_use.input
                    
                    if tool_name == "query_database":
                        sql_query = tool_input["sql"]
                        with st.expander("Executed SQL"):
                            st.code(sql_query, language="sql")
                        result = query_database(sql_query)
                        
                        if "last_df" in st.session_state:
                            df = st.session_state.last_df
                            st.dataframe(df, use_container_width=True)
                            st.session_state.current_sql = sql_query
                            st.session_state.current_data = df.to_dict('records')
                            
                        tool_result_content = result
                    elif tool_name == "create_chart":
                        python_code = tool_input["python_code"]
                        result = create_chart(python_code)
                        if "last_chart" in st.session_state:
                            st.altair_chart(st.session_state.last_chart, use_container_width=True)
                            st.session_state.current_chart = python_code
                        tool_result_content = result

                    # Send tool result back
                    claude_messages.append({"role": "assistant", "content": response.content})
                    claude_messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_use.id,
                                "content": tool_result_content,
                            }
                        ],
                    })
                    
                    response = anthropic_client.messages.create(
                        model=model,
                        max_tokens=4096,
                        system=system_prompt,
                        messages=claude_messages,
                        tools=claude_tools
                    )

                final_text = response.content[0].text
                message_placeholder.markdown(final_text)
                
                final_assistant_msg = {"role": "assistant", "content": final_text}
                if "current_sql" in st.session_state: final_assistant_msg["sql_query"] = st.session_state.current_sql
                if "current_data" in st.session_state: final_assistant_msg["data"] = st.session_state.current_data
                if "current_chart" in st.session_state: final_assistant_msg["chart_code"] = st.session_state.current_chart
                st.session_state.messages.append(final_assistant_msg)
                
                for key in ["current_sql", "current_data", "current_chart"]:
                    if key in st.session_state: del st.session_state[key]

            else:
                # OpenAI / Grok Logic with FALLBACK
                use_tools = True
                
                while True:
                    try:
                        # Prepare API call parameters
                        api_params = {
                            "model": model,
                            "messages": st.session_state.messages,
                        }
                        
                        # Only add tools if supported
                        if use_tools:
                            api_params["tools"] = tools
                            api_params["tool_choice"] = "auto"
                        
                        response = st.session_state.openai_client.chat.completions.create(**api_params)
                        
                        response_message = response.choices[0].message
                        
                        # Handle tool calls (if model supports it)
                        if response_message.tool_calls:
                            # Convert to dict to avoid "not subscriptable" error
                            msg_dict = {
                                "role": "assistant",
                                "content": response_message.content,
                                "tool_calls": response_message.tool_calls
                            }
                            st.session_state.messages.append(msg_dict)
                            
                            for tool_call in response_message.tool_calls:
                                function_name = tool_call.function.name
                                function_args = json.loads(tool_call.function.arguments)
                                
                                if function_name == "query_database":
                                    sql_query = function_args["sql"]
                                    with st.expander("🛠️ Executed SQL"):
                                        st.code(sql_query, language="sql")
                                    
                                    result_text = query_database(sql_query)
                                    
                                    if "last_df" in st.session_state:
                                        df = st.session_state.last_df
                                        st.dataframe(df, use_container_width=True)
                                        
                                        # Add download buttons (multiple formats)
                                        col1, col2, col3 = st.columns(3)
                                        
                                        with col1:
                                            csv_data = df.to_csv(index=False).encode('utf-8')
                                            st.download_button(
                                                label="📥 CSV",
                                                data=csv_data,
                                                file_name=f"query_result.csv",
                                                mime='text/csv',
                                                key=f"csv_new_{id(df)}"
                                            )
                                        
                                        with col2:
                                            from io import BytesIO
                                            buffer = BytesIO()
                                            df.to_excel(buffer, index=False, engine='openpyxl')
                                            excel_data = buffer.getvalue()
                                            st.download_button(
                                                label="📊 Excel",
                                                data=excel_data,
                                                file_name=f"query_result.xlsx",
                                                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                                                key=f"excel_new_{id(df)}"
                                            )
                                        
                                        with col3:
                                            json_data = df.to_json(orient='records', indent=2).encode('utf-8')
                                            st.download_button(
                                                label="📄 JSON",
                                                data=json_data,
                                                file_name=f"query_result.json",
                                                mime='application/json',
                                                key=f"json_new_{id(df)}"
                                            )
                                        
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
                            continue  # Loop to get next response
                        
                        # No tool calls - display final response
                        assistant_content = response_message.content or ""
                        
                        # FALLBACK: Try to extract and execute SQL from text
                        if not use_tools:
                            extracted_sql = DatabaseClient.extract_sql(assistant_content)
                            if extracted_sql:
                                with st.expander("🛠️ Detected & Executing SQL"):
                                    st.code(extracted_sql, language="sql")
                                
                                try:
                                    result_text = query_database(extracted_sql)
                                    if "last_df" in st.session_state:
                                        df = st.session_state.last_df
                                        st.dataframe(df, use_container_width=True)
                                        
                                        # Add download buttons (multiple formats)
                                        col1, col2, col3 = st.columns(3)
                                        
                                        with col1:
                                            csv_data = df.to_csv(index=False).encode('utf-8')
                                            st.download_button(
                                                label="📥 CSV",
                                                data=csv_data,
                                                file_name=f"query_result.csv",
                                                mime='text/csv',
                                                key=f"csv_fallback_{id(df)}"
                                            )
                                        
                                        with col2:
                                            from io import BytesIO
                                            buffer = BytesIO()
                                            df.to_excel(buffer, index=False, engine='openpyxl')
                                            excel_data = buffer.getvalue()
                                            st.download_button(
                                                label="📊 Excel",
                                                data=excel_data,
                                                file_name=f"query_result.xlsx",
                                                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                                                key=f"excel_fallback_{id(df)}"
                                            )
                                        
                                        with col3:
                                            json_data = df.to_json(orient='records', indent=2).encode('utf-8')
                                            st.download_button(
                                                label="📄 JSON",
                                                data=json_data,
                                                file_name=f"query_result.json",
                                                mime='application/json',
                                                key=f"json_fallback_{id(df)}"
                                            )

                                        
                                        st.session_state.current_sql = extracted_sql
                                        st.session_state.current_data = df.to_dict('records')
                                except Exception as sql_err:
                                    st.error(f"Lỗi khi chạy SQL: {sql_err}")
                        
                        message_placeholder.markdown(assistant_content)
                        
                        final_assistant_msg = {"role": "assistant", "content": assistant_content}
                        if "current_sql" in st.session_state: 
                            final_assistant_msg["sql_query"] = st.session_state.current_sql
                        if "current_data" in st.session_state: 
                            final_assistant_msg["data"] = st.session_state.current_data
                        if "current_chart" in st.session_state: 
                            final_assistant_msg["chart_code"] = st.session_state.current_chart
                        st.session_state.messages.append(final_assistant_msg)
                        
                        for key in ["current_sql", "current_data", "current_chart"]:
                            if key in st.session_state: del st.session_state[key]
                        break
                        
                    except Exception as e:
                        error_msg = str(e).lower()
                        # Detect "tools not supported" error
                        if ("tools" in error_msg or "404" in error_msg or "not supported" in error_msg) and use_tools:
                            st.warning("⚠️ Model này không hỗ trợ Function Calling. Đang chuyển sang chế độ phân tích văn bản...")
                            message_placeholder.markdown("🔄 Đang thử lại với chế độ văn bản (Fallback)...")
                            use_tools = False
                            
                            # Force model to output SQL code directly (Use 'user' role for better compatibility)
                            st.session_state.messages.append({
                                "role": "user", 
                                "content": "Hệ thống tool function calling không khả dụng. Vui lòng viết TRỰC TIẾP câu lệnh SQL vào trong block code (```sql ... ```) để tôi có thể trích xuất và thực thi. Đừng chỉ mô tả."
                            })
                            continue  # Retry without tools
                        else:
                            # Other errors - re-raise
                            raise e
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
