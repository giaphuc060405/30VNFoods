"""Ứng dụng Streamlit nhận diện 30 món ăn Việt Nam bằng SimpleCNN (tự huấn luyện từ đầu)."""
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import transforms as T

# ----------------------------------------------------------------------------
# 1. Cấu hình
# ----------------------------------------------------------------------------
WEIGHTS = Path(__file__).parent / "best.pt"   # file trọng số lấy từ Google Drive
IMG = 128                                      # phải trùng với lúc huấn luyện
LOW_CONF = 0.40                                # dưới ngưỡng này sẽ hiện cảnh báo

# Thứ tự PHẢI đúng như ds["train"].features["label"].names trong notebook
CLASS_NAMES = [
    "Banh beo", "Banh bot loc", "Banh can", "Banh canh", "Banh chung", "Banh cuon",
    "Banh duc", "Banh gio", "Banh khot", "Banh mi", "Banh pia", "Banh tet",
    "Banh trang nuong", "Banh xeo", "Bun bo Hue", "Bun dau mam tom", "Bun mam",
    "Bun rieu", "Bun thit nuong", "Ca kho to", "Canh chua", "Cao lau", "Chao long",
    "Com tam", "Goi cuon", "Hu tieu", "Mi quang", "Nem chua", "Pho", "Xoi xeo",
]
# Tên hiển thị có dấu cho dễ đọc (chỉ để hiển thị, không ảnh hưởng mô hình)
DISPLAY = {
    "Banh beo": "Bánh bèo", "Banh bot loc": "Bánh bột lọc", "Banh can": "Bánh căn",
    "Banh canh": "Bánh canh", "Banh chung": "Bánh chưng", "Banh cuon": "Bánh cuốn",
    "Banh duc": "Bánh đúc", "Banh gio": "Bánh giò", "Banh khot": "Bánh khọt",
    "Banh mi": "Bánh mì", "Banh pia": "Bánh pía", "Banh tet": "Bánh tét",
    "Banh trang nuong": "Bánh tráng nướng", "Banh xeo": "Bánh xèo",
    "Bun bo Hue": "Bún bò Huế", "Bun dau mam tom": "Bún đậu mắm tôm",
    "Bun mam": "Bún mắm", "Bun rieu": "Bún riêu", "Bun thit nuong": "Bún thịt nướng",
    "Ca kho to": "Cá kho tộ", "Canh chua": "Canh chua", "Cao lau": "Cao lầu",
    "Chao long": "Cháo lòng", "Com tam": "Cơm tấm", "Goi cuon": "Gỏi cuốn",
    "Hu tieu": "Hủ tiếu", "Mi quang": "Mì Quảng", "Nem chua": "Nem chua",
    "Pho": "Phở", "Xoi xeo": "Xôi xéo",
}


# ----------------------------------------------------------------------------
# 2. Kiến trúc: sao chép NGUYÊN VĂN từ notebook (tên lớp phải khớp với file .pt)
# ----------------------------------------------------------------------------
def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
                         nn.ReLU(), nn.MaxPool2d(2))


class SimpleCNN(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.features = nn.Sequential(block(3, 32), block(32, 64),
                                      block(64, 128), block(128, 256))
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                        nn.Dropout(0.4), nn.Linear(256, n))

    def forward(self, x):
        return self.classifier(self.features(x))


# Giống hệt eval_tf trong notebook (Val/Test không augmentation)
eval_tf = T.Compose([T.Resize((IMG, IMG)), T.ToTensor(),
                     T.Normalize([0.5] * 3, [0.5] * 3)])


@st.cache_resource(show_spinner="Đang nạp mô hình...")
def load_model():
    model = SimpleCNN(len(CLASS_NAMES))
    state = torch.load(WEIGHTS, map_location="cpu")   # chạy được cả khi không có GPU
    model.load_state_dict(state)
    model.eval()                                       # tắt Dropout, BatchNorm dùng thống kê đã học
    return model


def predict(model, img: Image.Image):
    img = ImageOps.exif_transpose(img).convert("RGB")  # xoay đúng chiều ảnh chụp từ điện thoại
    x = eval_tf(img).unsqueeze(0)                      # (1, 3, 128, 128)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    top = probs.topk(5)
    return [(CLASS_NAMES[i], p.item()) for p, i in zip(top.values, top.indices)]


# ----------------------------------------------------------------------------
# 3. Giao diện
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Nhận diện món ăn Việt", page_icon="🍜", layout="centered")
st.title("🍜 Nhận diện món ăn Việt Nam")
st.caption("Mạng CNN tự xây dựng từ đầu (SimpleCNN, 397.086 tham số) · 30 món ăn")

with st.sidebar:
    st.header("Về mô hình")
    st.markdown(
        "- Dữ liệu: 30VNFoods (25.136 ảnh)\n"
        "- Ảnh đầu vào: 128×128\n"
        "- Test Top-1: **64,31%**\n"
        "- Test Top-5: **91,19%**"
    )
    st.info("Mô hình chỉ biết 30 món trong danh sách. Với ảnh không phải món ăn, "
            "kết quả vẫn sẽ là một trong 30 món này.")
    with st.expander("Danh sách 30 món"):
        st.write(", ".join(DISPLAY[c] for c in CLASS_NAMES))

if not WEIGHTS.exists():
    st.error(f"Không tìm thấy file trọng số `{WEIGHTS.name}`. "
             "Hãy đặt best.pt cùng thư mục với app.py.")
    st.stop()

model = load_model()

uploaded = st.file_uploader("Chọn ảnh món ăn (JPG, PNG, WEBP)", type=["jpg", "jpeg", "png", "webp"])

if uploaded is None:
    st.info("Hãy tải lên một ảnh món ăn để bắt đầu.")
    st.stop()

try:
    image = Image.open(uploaded)
    image.load()
except Exception:
    st.error("Không đọc được file ảnh này. Hãy thử ảnh khác.")
    st.stop()

results = predict(model, image)
best_name, best_p = results[0]

col_img, col_res = st.columns([1, 1])
with col_img:
    st.image(ImageOps.exif_transpose(image), caption="Ảnh đầu vào")
with col_res:
    st.subheader("Kết quả dự đoán")
    st.metric(DISPLAY[best_name], f"{best_p:.1%}")
    if best_p < LOW_CONF:
        st.warning("Độ tin cậy thấp: mô hình không chắc chắn. Hãy xem các gợi ý bên dưới "
                   "hoặc thử ảnh rõ hơn, chụp gần món ăn.")

st.subheader("Top 5 gợi ý")
for name, p in results:                       # đã sắp theo xác suất giảm dần
    c_name, c_bar = st.columns([1, 3])
    c_name.write(DISPLAY[name])
    c_bar.progress(min(max(p, 0.0), 1.0), text=f"{p:.1%}")
st.caption("Xác suất do Softmax tính ra, chỉ mang tính tham khảo (mô hình chưa được hiệu chỉnh độ tin cậy).")