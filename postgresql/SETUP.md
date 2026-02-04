# 🚀 Hướng dẫn chạy Streamlit Web App

## Yêu cầu hệ thống

- Python 3.8 trở lên
- Kết nối internet (để gọi OpenAI API)
- PostgreSQL database (hoặc dùng database có sẵn của bạn)

## Bước 1: Cài đặt Python

Nếu chưa có Python, tải và cài đặt từ: https://www.python.org/downloads/

**Lưu ý:** Khi cài đặt, nhớ check ✅ "Add Python to PATH"

## Bước 2: Cài đặt dependencies

Mở PowerShell hoặc Command Prompt, chạy:

```bash
cd d:\Phuong\workspace\chat-with-your-database\postgresql
pip install -r requirements.txt
```

## Bước 3: Cấu hình

File `env` đã được cấu hình sẵn với:
- Database connection của bạn
- OpenAI API key

Nếu cần thay đổi, chỉnh sửa file `env`:

```env
# Database connection
DB_HOST=103.118.28.2
DB_PORT=5432
DB_DATABASE=gionglamnghiep
DB_USER=postgres
DB_PASSWORD=AppraisalQuail1Agent

# OpenAI API
OPENAI_API_KEY=your-api-key-here
```

## Bước 4: Chạy ứng dụng

```bash
streamlit run app.py
```

App sẽ tự động mở tại: **http://localhost:8501**

## Cách sử dụng

### 1. Kết nối Database (Sidebar)

- Thông tin database đã được điền sẵn từ file `env`
- Click **"🔌 Connect to Database"**
- Nếu kết nối thành công, bạn sẽ thấy ✅ và database schema

### 2. Chat với Database

Ví dụ các câu hỏi bạn có thể hỏi:

```
- How many tables are in the database?
- Show me the first 10 rows from [table_name]
- What are the column names in [table_name]?
- Count the number of records in [table_name]
- Create a bar chart showing the distribution of [column_name]
```

### 3. Xem SQL Query

- Mỗi khi AI generate SQL, nó sẽ hiển thị trong expander **"⚒️ SQL Query"**
- Click để xem câu SQL được thực thi

### 4. Xem kết quả

- Kết quả query hiển thị dưới dạng bảng (DataFrame)
- Nếu yêu cầu chart, sẽ hiển thị biểu đồ Altair

## Tính năng

✅ Chat interface với AI  
✅ Tự động generate SQL queries  
✅ Hiển thị kết quả dạng bảng  
✅ Tạo biểu đồ tự động  
✅ Read-only queries (an toàn)  
✅ Session state lưu lịch sử chat  
✅ Form config database linh hoạt  

## Troubleshooting

### Lỗi: "Python not found"
- Cài đặt Python từ python.org
- Đảm bảo đã check "Add to PATH" khi cài

### Lỗi: "pip not found"
```bash
python -m pip install -r requirements.txt
```

### Lỗi: "OpenAI API key not found"
- Kiểm tra file `env` có OPENAI_API_KEY chưa
- Hoặc nhập trực tiếp vào sidebar

### Lỗi: "Database connection failed"
- Kiểm tra database có đang chạy không
- Kiểm tra thông tin kết nối (host, port, user, password)
- Kiểm tra firewall/network

## Tech Stack

- **Streamlit** - Web framework
- **OpenAI API** - LLM (GPT-4/GPT-3.5)
- **PostgreSQL** - Database
- **pandas** - Data processing
- **Altair** - Visualization
- **psycopg** - PostgreSQL driver

## Lưu ý bảo mật

⚠️ **QUAN TRỌNG:**
- File `env` chứa thông tin nhạy cảm (passwords, API keys)
- **KHÔNG** commit file này lên Git
- File `.gitignore` đã được cấu hình để ignore `env`

## Tùy chỉnh

### Thay đổi OpenAI model

Trong sidebar, chọn model:
- `gpt-4o-mini` (rẻ, nhanh)
- `gpt-4o` (mạnh nhất)
- `gpt-3.5-turbo` (cân bằng)

### Clear chat history

Click button **"🗑️ Clear Chat History"** ở sidebar

## Deploy lên Cloud (Optional)

Bạn có thể deploy app lên:
- **Streamlit Cloud** (miễn phí): https://streamlit.io/cloud
- **Heroku**
- **AWS/GCP/Azure**

Hướng dẫn deploy: https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app
