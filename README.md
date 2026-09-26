# GUF - Quản lý nhiều profile Google/Chrome

Quản lý 100+ tài khoản Google mà không cần 100+ thư mục profile đầy đủ.

- Mỗi tài khoản được lưu dưới dạng **Cold State**: một file `.zip` chứa user-data-dir của Chrome
  đã bỏ cache. Mỗi file thường chỉ vài MB.
- Chỉ có **N slot** (mặc định 5) là thư mục profile đầy đủ để chạy Chrome.
- Khi mở một profile chưa nằm trong slot nào, app tự **bung** Cold State vào một slot rảnh rồi mở
  Chrome. Nếu slot đó đang chứa một profile khác chưa lưu, app lưu profile đó về Cold State trước.

## Cài đặt

```bash
pip install -r requirements.txt
python main.py
```

App tự tìm Chrome. Nếu không tìm thấy, anh vào **Cài đặt** để chọn `chrome.exe`.

## Sử dụng

1. **Thêm profile**: nhập tên (ví dụ email). Chrome mở trang đăng nhập Google, anh đăng nhập
   bình thường.
2. **Cold State**: lưu profile đang chọn.
   - Nếu Chrome còn mở, app hỏi: **Yes** để app đóng Chrome rồi lưu, **No** để anh tự đóng
     và app lưu ngay khi Chrome tắt.
   - Chrome phải đóng thì mới lưu được, vì cookie chỉ được ghi xuống đĩa an toàn khi Chrome tắt.
3. **Mở profile**: chọn một hoặc nhiều profile trong danh sách rồi nhấn nút (hoặc double-click).
   - Nếu profile đã nằm sẵn trong một slot, app mở luôn.
   - Nếu chưa, app bung Cold State vào slot trống hoặc slot lâu không dùng nhất, rồi mở.
   - Khi đóng Chrome, app tự lưu lại Cold State. Có thể tắt tính năng này trong Cài đặt.

## Dữ liệu

```
guf_data/
  settings.json     cài đặt
  profiles.json     danh sách profile
  cold/<id>.zip     Cold State của từng profile
  slots/slot1..N/   thư mục profile đầy đủ đang dùng
```

Đổi vị trí thư mục dữ liệu bằng biến môi trường `GUF_DATA`.

## Lưu ý quan trọng

- **Chỉ dùng Cold State trên cùng máy và cùng user Windows đã tạo ra nó.** Chrome mã hoá cookie
  bằng khoá gắn với máy (DPAPI hoặc App-Bound Encryption), nên mang sang máy khác sẽ bị đăng xuất.
- File trong `guf_data/` có giá trị ngang mật khẩu. Đừng chia sẻ và đừng commit lên git.
- Nên mở từng tài khoản định kỳ (1-2 tuần một lần) để Google không đòi xác minh lại, và giữ
  mỗi tài khoản một IP/proxy cố định (có thể thêm `--proxy-server=...` trong Cài đặt nếu cần).
- Nếu app bị tắt khi Chrome còn mở, lần mở app kế tiếp sẽ tự lưu Cold State cho các slot còn
  dữ liệu chưa lưu.

## Test

```bash
pip install pytest
python -m pytest tests
```
