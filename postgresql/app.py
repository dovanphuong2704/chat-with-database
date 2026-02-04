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
    page_title="Chat with PostgreSQL Database",
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
st.sidebar.title("🗄️ Database Configuration")

with st.sidebar.form("db_config"):
    db_host = st.text_input("Host", value=os.getenv("DB_HOST", "localhost"))
    db_port = st.text_input("Port", value=os.getenv("DB_PORT", "5432"))
    db_name = st.text_input("Database", value=os.getenv("DB_DATABASE", ""))
    db_user = st.text_input("Username", value=os.getenv("DB_USER", ""))
    db_password = st.text_input("Password", value=os.getenv("DB_PASSWORD", ""), type="password")
    
    connect_button = st.form_submit_button("🔌 Connect to Database")

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
        
        st.sidebar.success("✅ Connected successfully!")
    except Exception as e:
        st.sidebar.error(f"❌ Connection failed: {str(e)}")
        st.session_state.db_client = None
        st.session_state.db_schema = None

# Show database schema if connected
if st.session_state.db_schema:
    with st.sidebar.expander("📋 Database Schema", expanded=False):
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
st.title("💬 Chat with Your PostgreSQL Database")
st.markdown("Ask questions about your database in natural language!")

# Check prerequisites
if not st.session_state.openai_client:
    st.warning("⚠️ Please configure your OpenAI API key in the sidebar.")
    st.stop()

if not st.session_state.db_client:
    st.info("ℹ️ Please connect to a database using the sidebar form.")
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
            "You are an SQL assistant connected directly to a PostgreSQL database. "
            "You can execute SELECT queries on this database, "
            "and your system will automatically run any SQL query you provide. "
            "Always try to answer user questions by generating and executing an SQL query first, "
            "even if you think you already know the answer logically. "
            "Never assume the result — always verify it in the database. "
            "Only if the question cannot possibly be answered with SQL, then ask for clarification. "
            "Use SELECT statements only (no INSERT, UPDATE, DELETE). "
            "When creating visualizations (such as charts, graphs, or plots), "
            "use the Altair library for all visual outputs. "
            "Create chart object with Altair and assign to variable 'chart'. Set width to 600px. "
            "Data is in pandas dataframe called df - use df variable. DON'T create sample df variable. "
            "If you think you need another library, do not attempt to import it — "
            "simply explain that it is not available. "
            f"Database schema:\n{st.session_state.db_schema}"
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
        
        # Display SQL query if present
        if "sql_query" in message:
            with st.expander("⚒️ SQL Query"):
                st.code(message["sql_query"], language="sql")

# Chat input
if prompt := st.chat_input("Ask a question about your database..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get AI response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🤔 Thinking...")
        
        try:
            # Call OpenAI API
            response = st.session_state.openai_client.chat.completions.create(
                model=model,
                messages=st.session_state.messages,
                tools=tools,
                tool_choice="auto"
            )
            
            response_message = response.choices[0].message
            assistant_message = {"role": "assistant", "content": response_message.content or ""}
            
            # Handle tool calls
            if response_message.tool_calls:
                # IMPORTANT: Add assistant message with tool_calls FIRST
                # This is required by OpenAI API before adding tool responses
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        } for tc in response_message.tool_calls
                    ]
                })
                
                # Now process each tool call
                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = eval(tool_call.function.arguments)
                    
                    if function_name == "query_database":
                        sql_query = function_args["sql"]
                        
                        # Show SQL query
                        with st.expander("⚒️ SQL Query"):
                            st.code(sql_query, language="sql")
                        
                        # Execute query
                        result_text = query_database(sql_query)
                        
                        # Show dataframe (but don't store in message - causes serialization error)
                        if "last_df" in st.session_state:
                            st.dataframe(st.session_state.last_df, use_container_width=True)
                            # Store SQL query for display later
                            assistant_message["sql_query"] = sql_query
                        
                        # Add tool response to messages
                        st.session_state.messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": function_name,
                            "content": result_text
                        })
                    
                    elif function_name == "create_chart":
                        python_code = function_args["python_code"]
                        
                        # Execute chart code
                        result_text = create_chart(python_code)
                        
                        # Show chart (but don't store in message - causes serialization error)
                        if "last_chart" in st.session_state:
                            st.altair_chart(st.session_state.last_chart, use_container_width=True)
                        
                        # Add tool response to messages
                        st.session_state.messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": function_name,
                            "content": result_text
                        })
                
                # Get final response after tool calls
                final_response = st.session_state.openai_client.chat.completions.create(
                    model=model,
                    messages=st.session_state.messages
                )
                
                final_message = final_response.choices[0].message.content
                assistant_message["content"] = final_message
                message_placeholder.markdown(final_message)
            else:
                # No tool calls, just display response
                message_placeholder.markdown(response_message.content)
            
            # Add assistant message to history
            st.session_state.messages.append(assistant_message)
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            message_placeholder.markdown(f"❌ Error: {str(e)}")

# Sidebar footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Tips")
st.sidebar.markdown("""
- Ask questions in natural language
- Request charts and visualizations
- The AI will generate SQL queries automatically
- All queries are read-only for safety
""")

# Clear chat button
if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.messages = []
    st.rerun()
