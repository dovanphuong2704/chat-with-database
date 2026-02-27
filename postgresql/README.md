# AI Data Intelligence Platform

Ứng dụng chat với cơ sở dữ liệu PostgreSQL bằng ngôn ngữ tự nhiên. Người dùng đặt câu hỏi, AI tự động sinh SQL, truy vấn database và trả kết quả dưới dạng bảng hoặc biểu đồ.

## Tính năng

- **Chat với database** — đặt câu hỏi bằng tiếng Việt, AI sinh và thực thi SQL tự động
- **Đa mô hình AI** — hỗ trợ OpenAI (GPT-4o), Grok (xAI), Google Gemini, Anthropic Claude
- **Multi-user** — đăng ký / đăng nhập, mỗi user quản lý riêng DB connections và API keys
- **Mã hoá dữ liệu** — API keys và mật khẩu DB được mã hoá AES-256 (Fernet) trước khi lưu
- **Trực quan hoá** — tự động vẽ biểu đồ Altair từ kết quả truy vấn
- **Dashboard** — ghim bảng/biểu đồ để xem lại
- **Database Explorer** — khám phá schema và dữ liệu mẫu từng bảng
- **An toàn** — chỉ cho phép SELECT, timeout 5 giây, tự động giới hạn 1000 dòng

## Kiến trúc

```
Streamlit (app.py)
    ├── Xác thực người dùng (app_db.py)
    ├── Gọi AI API (OpenAI / Gemini / Claude / Grok)
    ├── Thực thi SQL an toàn (dbclient.py)
    └── Chạy Python chart an toàn (safeish.py)
```

## Cấu trúc thư mục

```
postgresql/
├── app.py                 # Ứng dụng Streamlit chính
├── app_db.py              # Quản lý user, DB connections, API keys
├── dbclient.py            # Kết nối PostgreSQL (psycopg v3, connection pool)
├── safeish.py             # Executor Python bị giới hạn (vẽ chart)
├── glossary.json          # Semantic layer — định nghĩa thuật ngữ nghiệp vụ
├── requirements.txt       # Thư viện Python
├── setup.sh               # Script cài đặt tự động
├── env.example            # Mẫu file cấu hình
├── generate_sales_csv.py  # Tạo dữ liệu demo
├── setup_sales_db.sh      # Khởi tạo DB demo qua Docker
└── csv/                   # Dữ liệu demo đã tạo
```

## Cài đặt

### 1. Clone và chạy script setup

```bash
git clone <repo-url>
cd postgresql

chmod +x setup.sh
./setup.sh
```

Script sẽ tự động:
- Tạo virtual environment
- Cài đặt tất cả thư viện từ `requirements.txt`
- Tạo file `.env` từ `env.example`

### 2. Cấu hình file `.env`

```env
DB_HOST=localhost
DB_PORT=5432
DB_DATABASE=your_database
DB_USER=your_user
DB_PASSWORD=your_password

APP_DB_URL=sqlite:///app.db

# Tạo encryption key:
# python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
ENCRYPTION_KEY=your_generated_key
```

> `APP_DB_URL` là database lưu thông tin user/config của app (mặc định SQLite).
> `DB_*` là database dữ liệu bạn muốn chat (cũng có thể cấu hình trong giao diện).

### 3. Chạy ứng dụng

```bash
source venv/bin/activate   # Linux/Mac
# hoặc
source venv/Scripts/activate  # Windows Git Bash

streamlit run app.py
```

Mở trình duyệt tại `http://localhost:8501`

## Hướng dẫn sử dụng

1. **Đăng ký / Đăng nhập** tài khoản
2. Vào **Cài đặt → Database** để thêm kết nối PostgreSQL
3. Vào **Cài đặt → API Key** để thêm API key của provider AI
4. Quay lại **Chat**, chọn DB + Model ở sidebar → bắt đầu đặt câu hỏi

## Database demo (tuỳ chọn)

Nếu chưa có database, có thể dùng dataset demo bán hàng:

```bash
chmod +x setup_sales_db.sh
./setup_sales_db.sh
```

Tạo 3 bảng: `customers`, `products`, `orders` trong PostgreSQL Docker.

```bash
# Tuỳ chỉnh quy mô
N_CUSTOMERS=5000 N_ORDERS=200000 ./setup_sales_db.sh
```

## Semantic Layer

File `glossary.json` định nghĩa thuật ngữ nghiệp vụ và quy tắc SQL:

```json
{
  "business_terms": [
    { "term": "doanh thu", "definition": "SUM(quantity * unit_price) WHERE status = 'paid'" }
  ],
  "sql_rules": [
    "Luôn dùng alias rõ ràng cho các cột tính toán"
  ]
}
```

AI sẽ tự động đọc file này để sinh SQL chính xác hơn với nghiệp vụ của bạn.

## Bảo mật

| Lớp | Biện pháp |
|-----|-----------|
| SQL | Chỉ SELECT · Chặn DDL/DML · Timeout 5s · Auto LIMIT 1000 |
| Python | Chặn `eval/exec/open/__import__` · Validate AST · Biến bị giới hạn |
| Dữ liệu | API keys và DB passwords mã hoá Fernet (AES-128-CBC) |
| Mật khẩu | bcrypt hash |

## Yêu cầu hệ thống

- Python 3.10+
- PostgreSQL 13+
- Kết nối internet để gọi AI API
