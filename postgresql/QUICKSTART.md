# 🚀 Quick Start - Chạy Streamlit App

## ✅ App đang chạy!

Streamlit app đã được khởi động thành công!

## 📍 Truy cập App

Mở browser và truy cập:
```
http://localhost:8501
```

Hoặc Streamlit sẽ tự động mở browser cho bạn.

## 🎯 Sử dụng App

### 1. Kết nối Database (Sidebar)
- Thông tin database đã được điền sẵn từ file `env`
- Click button **"🔌 Connect to Database"**
- Đợi message ✅ "Connected successfully!"

### 2. Chat với Database
Thử các câu hỏi sau:

```
How many tables are in the database?
```

```
Show me the first 5 rows from any table
```

```
What are all the table names?
```

```
Create a bar chart showing data distribution
```

### 3. Xem Kết quả
- SQL query sẽ hiển thị trong expander "⚒️ SQL Query"
- Kết quả hiển thị dưới dạng bảng
- Charts (nếu có) hiển thị dưới bảng

## ⚙️ Cấu hình

### OpenAI API Key
- Đã được cấu hình sẵn trong file `env`
- Nếu cần thay đổi, nhập vào sidebar

### Model Selection
Chọn model trong sidebar:
- **gpt-4o-mini** (khuyến nghị - rẻ, nhanh)
- **gpt-4o** (mạnh nhất, đắt hơn)
- **gpt-3.5-turbo** (cân bằng)

## 🛑 Dừng App

Nhấn `Ctrl+C` trong terminal để dừng app.

## 🔧 Troubleshooting

### App không mở browser tự động?
Mở thủ công: http://localhost:8501

### Lỗi "Connection failed"?
- Kiểm tra database có đang chạy không
- Kiểm tra thông tin trong file `env`

### Lỗi "OpenAI API error"?
- Kiểm tra API key trong file `env`
- Kiểm tra API key còn credit không

## 📝 Lưu ý

- Streamlit yêu cầu nhập email lần đầu (có thể bỏ qua bằng Enter)
- Chat history sẽ mất khi refresh page
- Mỗi query tốn một chút OpenAI credit (~$0.001-0.01)

## 🎉 Enjoy!

Bây giờ bạn có thể chat với database bằng ngôn ngữ tự nhiên!
