# 🚀 AI Data Intelligence Platform - Danh sách Chức năng

Nền tảng phân tích dữ liệu thông minh, cho phép trò chuyện trực tiếp với cơ sở dữ liệu PostgreSQL bằng ngôn ngữ tự nhiên.

## 1. 🤖 Bộ não Trí tuệ Nhân tạo (AI Engine)
*   **Hỗ trợ đa nền tảng:** Kết nối linh hoạt với OpenAI (GPT-4o, GPT-3.5), Google Gemini, Anthropic Claude, xAI Grok.
*   **Hỗ trợ Local AI:** Tương thích với **Ollama** và các Custom API chuẩn OpenAI cho các mô hình chạy cục bộ.
*   **Kỹ thuật Function Calling:** AI tự động quyết định khi nào cần truy vấn database và khi nào cần vẽ biểu đồ.
*   **Cơ chế Fallback thông minh:** Tự động chuyển sang chế độ văn bản nếu Model không hỗ trợ công cụ (Tools), đảm bảo hoạt động ổn định trên mọi loại Model.

## 2. 💬 Giao diện Chat Thông minh (Tab 1: Chat)
*   **Truy vấn Tiếng Việt:** Chuyển đổi ngôn ngữ tự nhiên thành câu lệnh SQL chính xác.
*   **Trực quan hóa tự động:** Tự động tạo biểu đồ sinh động (Altair) ngay trong khung chat khi có yêu cầu phân tích xu hướng.
*   **Công cụ SQL:**
    *   **SQL Preview:** Hiển thị code SQL đi kèm kết quả.
    *   **📋 Copy SQL:** Sao chép nhanh câu lệnh.
    *   **🧐 Giải thích (Explain):** Phân tích logic câu lệnh SQL sang ngôn ngữ dễ hiểu cho người không chuyên.
*   **📌 Ghim (Pin):** Lưu lại các bảng số liệu hoặc biểu đồ "tâm đắc" vào Dashboard chỉ với một nút bấm.

## 5. 🛠️ Tiện ích và Phụ trợ
*   **Sidebar Tiện lợi:**
    *   Cấu hình kết nối Database và API Key nhanh chóng.
    *   **🕒 Lịch sử Query:** Xem lại 10 câu truy vấn gần nhất.
    *   **🔄 Nút Chạy lại:** Thực thi lại nhanh các câu hỏi cũ.
    *   **💡 Gợi ý thông minh (Smart Suggestions):** Tự động tạo câu hỏi mẫu dựa theo ngành nghề dữ liệu trong database.
*   **Xuất dữ liệu (Export):** Tải kết quả về máy với 3 định dạng: **CSV, Excel (.xlsx), và JSON**.

## 6. 🛡️ Bảo mật và An toàn
*   **Chế độ Read-only:** Hệ thống chỉ thực thi lệnh `SELECT`, ngăn chặn hoàn toàn việc xóa hoặc sửa dữ liệu gốc.
*   **Thực thi Python cô lập:** Sử dụng `SafeishPythonExecutor` để chạy code vẽ biểu đồ trong môi trường an toàn.

---
*Tài liệu này được cập nhật vào ngày 06/02/2026.*
