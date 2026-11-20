# Ảnh container đã dùng (ghim để tái lập)

Workflow `chay-noi*.yml` dùng thẻ `dolfinx/dolfinx:stable` (không ghim). Digest đọc từ log Actions (bước "Initialize containers"):

| Lần chạy | Ngày | Digest |
|---|---|---|
| run 44481078149 (vòng 11-A khe hình học nx400; job 135413399006) | 2026-11-19 | `sha256:2ae4bfbc0d9077268880faf04c72750528bee986c94ab223a2c159969bd56fa8` |

Tái lập: `docker pull dolfinx/dolfinx@sha256:2ae4bfbc0d9077268880faf04c72750528bee986c94ab223a2c159969bd56fa8`
(DOLFINx 0.11.0.post0, PETSc 3.25.1 theo log mã). Các lần chạy trước 19/11 dùng thẻ `stable` tại thời điểm chạy; digest chưa được ghi.
