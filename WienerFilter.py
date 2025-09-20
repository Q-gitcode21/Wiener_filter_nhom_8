import streamlit as st
import cv2
import numpy as np
from PIL import Image

# --- Hàm lọc Wiener ---
def wiener_filter(img, kernel_size=5):
    img = np.float32(img)
    local_mean = cv2.blur(img, (kernel_size, kernel_size))
    local_var = cv2.blur(img**2, (kernel_size, kernel_size)) - local_mean**2
    noise_var = np.mean(local_var)
    result = local_mean + (np.maximum(local_var - noise_var, 0) /
                           (local_var + 1e-8)) * (img - local_mean)
    return np.uint8(np.clip(result, 0, 255))

# --- Hàm sharpen (Unsharp Mask) ---
def unsharp_mask(img, ksize=5, amount=1.5):
    blurred = cv2.GaussianBlur(img, (ksize, ksize), 0)
    sharp = cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)
    return np.uint8(np.clip(sharp, 0, 255))

# --- Hàm tính toán MSE và PSNR ---
def mse_psnr(original, filtered):
    mse = np.mean((original.astype(np.float32) - filtered.astype(np.float32)) ** 2)
    if mse == 0:
        return mse, float("inf")
    psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    return mse, psnr


# ------------------- STREAMLIT APP -------------------
st.title("Khôi phục ảnh bằng Wiener Filter")

# Upload file ảnh
uploaded_file = st.file_uploader("Chọn một ảnh", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    # Đọc ảnh gốc (xám)
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_gray = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    # 1) Median filter khử nhiễu muối tiêu
    median = cv2.medianBlur(img_gray, 3)

    # 2) Wiener filter
    wiener = wiener_filter(median, kernel_size=5)

    # 3) Sharpen (Unsharp Mask)
    filtered = unsharp_mask(wiener, ksize=5, amount=1.2)

    # Tính MSE, PSNR
    mse, psnr = mse_psnr(img_gray, filtered)
    st.write(f"**MSE:** {mse:.2f} | **PSNR:** {psnr:.2f} dB")

    # Hiển thị ảnh gốc và ảnh sau lọc
    col1, col2 = st.columns(2)
    with col1:
        st.image(img_gray, caption="Ảnh gốc", use_container_width=True, channels="GRAY")
    with col2:
        st.image(filtered, caption="Ảnh sau Wiener filter", use_container_width=True, channels="GRAY")

    # Cho phép tải ảnh kết quả
    result_pil = Image.fromarray(filtered)
    st.download_button(
        label="📥 Tải ảnh đã lọc",
        data=cv2.imencode(".jpg", filtered)[1].tobytes(),
        file_name="filtered.jpg",
        mime="image/jpeg"
    )
