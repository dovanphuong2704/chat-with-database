#!/usr/bin/env bash
set -e

echo "================================================"
echo "  AI Data Intelligence Platform - Setup"
echo "================================================"

# ---- Python version check ----
PYTHON=$(command -v python3 || command -v python)
if [ -z "$PYTHON" ]; then
    echo "[ERROR] Python không tìm thấy. Cài đặt Python 3.10+ trước."
    exit 1
fi

PY_VER=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "[OK] Python $PY_VER"

# ---- Virtual environment ----
if [ ! -d "venv" ]; then
    echo "[...] Tạo virtual environment..."
    $PYTHON -m venv venv
fi

# Activate venv
if [ -f "venv/Scripts/activate" ]; then
    # Windows Git Bash
    source venv/Scripts/activate
else
    source venv/bin/activate
fi
echo "[OK] Virtual environment đã kích hoạt"

# ---- Upgrade pip ----
pip install --upgrade pip --quiet

# ---- Install dependencies ----
echo "[...] Cài đặt các thư viện..."
pip install -r requirements.txt

echo ""
echo "================================================"
echo "  Cài đặt hoàn tất!"
echo "================================================"

# ---- Env file check ----
if [ ! -f ".env" ]; then
    if [ -f "env.example" ]; then
        cp env.example .env
        echo "[!] Đã tạo file .env từ env.example"
        echo "    -> Hãy cập nhật thông tin kết nối DB và API Key trong file .env"
    else
        echo "[!] Chưa có file .env — hãy tạo file .env với nội dung:"
        echo ""
        echo "    DB_HOST=localhost"
        echo "    DB_PORT=5432"
        echo "    DB_DATABASE=your_db"
        echo "    DB_USER=your_user"
        echo "    DB_PASSWORD=your_password"
        echo "    OPENAI_API_KEY=sk-..."
        echo "    APP_DB_URL=sqlite:///app.db"
        echo "    ENCRYPTION_KEY="
    fi
fi

echo ""
echo "  Chạy ứng dụng:"
echo "    streamlit run app.py"
echo ""
