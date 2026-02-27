# Hướng dẫn cài đặt chi tiết

## Yêu cầu hệ thống

- Python 3.10 trở lên
- PostgreSQL 13 trở lên
- Kết nối internet để gọi AI API

---

## Bước 1: Cài đặt Python

Tải và cài đặt từ https://www.python.org/downloads/

> Khi cài trên Windows, nhớ check **"Add Python to PATH"**

Kiểm tra:
```bash
python --version
```

---

## Bước 2: Cài đặt thư viện

### Cách nhanh (dùng script tự động)

```bash
cd postgresql
chmod +x setup.sh
./setup.sh
```

Script sẽ tự tạo virtual environment và cài tất cả thư viện.

### Cách thủ công

```bash
cd postgresql
python -m venv venv

# Kích hoạt venv
source venv/bin/activate        # Linux / Mac
source venv/Scripts/activate    # Windows Git Bash
venv\Scripts\activate.bat       # Windows CMD

pip install -r requirements.txt
```

---

## Bước 3: Cấu hình file `.env`

Sao chép file mẫu:

```bash
cp env.example .env
```

Mở file `.env` và điền thông tin:

```env
# PostgreSQL — database bạn muốn chat
DB_HOST=localhost
DB_PORT=5432
DB_DATABASE=your_database
DB_USER=your_user
DB_PASSWORD=your_password

# Database lưu thông tin user/config của app
APP_DB_URL=sqlite:///app.db

# Khoá mã hoá API keys và mật khẩu DB
# Tạo bằng lệnh:
# python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
ENCRYPTION_KEY=your_generated_key
```

> **Lưu ý:** Không commit file `.env` lên Git. File `.gitignore` đã được cấu hình để bỏ qua nó.

---

## Bước 4: Chạy ứng dụng

```bash
streamlit run app.py
```

Mở trình duyệt tại **http://localhost:8501**

---

## Bước 5: Thiết lập lần đầu

### 1. Tạo tài khoản
- Chọn tab **Tạo tài khoản** ở màn hình đăng nhập
- Nhập tên đăng nhập, email, mật khẩu (tối thiểu 6 ký tự)

### 2. Thêm kết nối Database
- Vào **Cài đặt → Database**
- Nhập thông tin kết nối PostgreSQL → **Lưu & Kết nối**

### 3. Thêm API Key
- Vào **Cài đặt → API Key & Model**
- Chọn provider (OpenAI / Grok / Gemini / Claude)
- Dán API key vào → **Lưu**

### 4. Bắt đầu chat
- Quay lại trang **Chat**
- Sidebar → chọn Database và API Key → **Kết nối DB** → **Lấy Model**
- Gõ câu hỏi bằng tiếng Việt

---

## Troubleshooting

### Lỗi kết nối Database
- Kiểm tra PostgreSQL đang chạy
- Kiểm tra host, port, username, password
- Kiểm tra firewall / network cho phép kết nối

### Lỗi "Missing environment variables"
- Đảm bảo file `.env` tồn tại và đã điền đủ `DB_HOST`, `DB_PORT`, `DB_DATABASE`, `DB_USER`, `DB_PASSWORD`

### Lỗi module not found
```bash
pip install -r requirements.txt
```

### Lỗi ENCRYPTION_KEY
Tạo khoá mới:
```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```
Dán kết quả vào `ENCRYPTION_KEY=` trong file `.env`

### Model không hỗ trợ Function Calling
App sẽ tự động chuyển sang chế độ fallback — trích xuất SQL trực tiếp từ văn bản.

---

## Tech Stack

| Thành phần | Công nghệ |
|------------|-----------|
| Web UI | Streamlit |
| AI Providers | OpenAI · Grok (xAI) · Google Gemini · Anthropic Claude |
| Database | PostgreSQL (psycopg v3) |
| App DB | SQLite / PostgreSQL (SQLAlchemy) |
| Visualisation | Altair |
| Bảo mật | bcrypt · Fernet (AES-128-CBC) |
