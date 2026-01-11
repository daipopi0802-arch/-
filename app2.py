import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import math
import japanize_matplotlib # グラフの日本語化

# --- 1. ページ構成とデザイン設定 ---
st.set_page_config(page_title="材力解析システム Pro Max", layout="wide")

# 全体をゴシック体に統一するCSS
st.markdown("""
<style>
    html, body, [class*="css"], .stMarkdown, .stMetric, .stTable {
        font-family: "Hiragino Kaku Gothic ProN", "Hiragino Sans", "Meiryo", "sans-serif" !important;
    }
    .main { background-color: #fcfcfc; }
    .stAlert { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

st.title("🚀 材料力学・画像解析システム Pro Max")
st.write("き裂解析、組織観察、破壊力学評価のためのオールインワンツールです。")

# --- 2. サイドバー：解析条件の設定 ---
st.sidebar.header("🛠️ 解析パラメータ設定")

mode = st.sidebar.selectbox(
    "1. 解析モードを選択", 
    ["き裂進展 (Crack/Line)", "結晶粒・穴 (Circle)", "楕円・パーツ (Ellipse)"],
    key="select_mode"
)

# 前処理設定
st.sidebar.subheader("🌓 前処理 (ノイズ除去)")
use_blur = st.sidebar.checkbox("Gaussian Blurを適用", value=True, key="blur_check")
blur_size = st.sidebar.slider("フィルタ強度", 1, 15, 5, step=2, key="blur_val")

# キャリブレーション
st.sidebar.subheader("📏 長さ校正 (Calibration)")
mm_per_px = st.sidebar.number_input("1ピクセルあたりの長さ (mm/px)", value=0.0100, format="%.4f", key="cal_val")

# 材力パラメータ（き裂モード時のみ）
if mode == "き裂進展 (Crack/Line)":
    st.sidebar.subheader("🏗️ 破壊力学パラメータ")
    sigma = st.sidebar.number_input("負荷応力 σ (MPa)", value=100.0, key="stress_val")
    geo_f = st.sidebar.number_input("形状補正係数 F", value=1.12, key="geo_val")
    danger_th = st.sidebar.slider("🚨 警告しきい値 (mm)", 0.1, 10.0, 2.0, key="danger_val")
    show_heatmap = st.sidebar.checkbox("🔥 密度ヒートマップを表示", value=False, key="heat_check")
else:
    show_heatmap = False

# 画像処理パラメータ
st.sidebar.subheader("⚙️ 検出感度設定")
c_low = st.sidebar.slider("エッジ検出下限", 0, 255, 50, key="canny_l")
c_high = st.sidebar.slider("エッジ検出上限", 0, 255, 150, key="canny_h")

# --- 3. メイン処理：画像アップロード ---
uploaded_file = st.file_uploader("解析用画像をアップロード (jpg, png, tif, bmp)", type=["jpg", "jpeg", "png", "tif", "bmp"])

if uploaded_file:
    # 画像読み込み
    raw_img = Image.open(uploaded_file)
    img_array = np.array(raw_img)
    if img_array.shape[2] == 4: img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    # グレースケール化とノイズ除去
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    if use_blur:
        gray = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    
    output_img = img_array.copy()
    heatmap_layer = np.zeros_like(gray)
    results_list = []

    # --- 4. 解析アルゴリズム実行 ---
    
    # 【A】き裂進展解析
    if mode == "き裂進展 (Crack/Line)":
        h_thresh = st.sidebar.slider("直線検出感度", 10, 200, 50, key="ht_val")
        min_l = st.sidebar.slider("最小長さ (px)", 1, 500, 30, key="hl_val")
        
        edges = cv2.Canny(gray, c_low, c_high)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, h_thresh, minLineLength=min_l, maxLineGap=15)
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                l_mm = np.sqrt((x2-x1)**2 + (y2-y1)**2) * mm_per_px
                ang = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
                
                # 応力拡大係数算出: KI = F * σ * √(π * a) ※aはき裂長(m)
                k_val = geo_f * sigma * math.sqrt(math.pi * (l_mm / 1000))
                
                # 描画
                color = (255, 0, 0) if l_mm > danger_th else (0, 255, 0)
                if not show_heatmap:
                    cv2.line(output_img, (x1, y1), (x2, y2), color, 3)
                cv2.line(heatmap_layer, (x1, y1), (x2, y2), 255, 5)
                
                results_list.append({"長さ(mm)": l_mm, "角度(deg)": ang, "K値(MPa√m)": k_val})
        
        if show_heatmap and lines is not None:
            h_blur = cv2.GaussianBlur(heatmap_layer, (101, 101), 0)
            h_color = cv2.applyColorMap(h_blur, cv2.COLORMAP_JET)
            output_img = cv2.addWeighted(img_array, 0.6, h_color, 0.4, 0)

    # 【B】円形・組織解析
    elif mode == "結晶粒・穴 (Circle)":
        p2 = st.sidebar.slider("検出精度", 10, 100, 30, key="cp2")
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30, 
                                   param1=c_high, param2=p2, minRadius=5, maxRadius=500)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                r_mm = i[2] * mm_per_px
                cv2.circle(output_img, (i[0], i[1]), i[2], (0, 255, 0), 3)
                results_list.append({"長さ(mm)": r_mm*2, "面積(mm2)": np.pi*(r_mm**2), "角度(deg)": 0})

    # 【C】楕円解析
    elif mode == "楕円・パーツ (Ellipse)":
        edges = cv2.Canny(gray, c_low, c_high)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if len(c) >= 5:
                el = cv2.fitEllipse(c)
                (x,y), (MA, ma), ang = el
                area = np.pi * (MA/2*mm_per_px) * (ma/2*mm_per_px)
                cv2.ellipse(output_img, el, (255, 255, 0), 2)
                results_list.append({"長さ(mm)": (MA+ma)/2*mm_per_px, "面積(mm2)": area, "角度(deg)": ang})

    # --- 5. 画面表示とレポート ---
    col1, col2 = st.columns(2)
    with col1: st.image(raw_img, caption="元画像", use_container_width=True)
    with col2: st.image(output_img, caption="解析結果", use_container_width=True)

    if results_list:
        df = pd.DataFrame(results_list)
        st.divider()
        st.subheader("📊 解析サマリーレポート")
        
        # 指標表示
        m1, m2, m3 = st.columns(3)
        m1.metric("検出個数", f"{len(df)} 箇所")
        m2.metric("平均サイズ", f"{df['長さ(mm)'].mean():.3f} mm")
        if "K値(MPa√m)" in df.columns:
            m3.metric("最大K値", f"{df['K値(MPa√m)'].max():.2f} MPa√m")
            # 自動テキストレポート
            st.info(f"【考察用メモ】検出された最大き裂長は {df['長さ(mm)'].max():.3f} mm です。負荷応力 {sigma} MPa における推定最大応力拡大係数は {df['K値(MPa√m)'].max():.2f} MPa√m となりました。")
            if df["長さ(mm)"].max() > danger_th:
                st.error(f"🚨 判定: しきい値 {danger_th}mm を超える重大なき裂を検出しました。破壊の危険性があります。")
        
        # 統計グラフ
        if st.button("📈 詳細な分布グラフを表示"):
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            sns.histplot(df["長さ(mm)"], kde=True, ax=ax[0], color="#3498db")
            ax[0].set_title("サイズの分布 (mm)")
            
            target = "K値(MPa√m)" if "K値(MPa√m)" in df.columns else "面積(mm2)"
            sns.histplot(df[target], kde=True, ax=ax[1], color="#e67e22")
            ax[1].set_title(f"{target} の分布")
            st.pyplot(fig)

        # データダウンロード
        st.table(df.describe().loc[['max', 'min', 'mean']])
        st.download_button("📝 解析データをCSVで保存", df.to_csv(index=False).encode('utf-8'), "analysis_result.csv")
    else:
        st.warning("対象が検出されませんでした。パラメータを調整してください。")
else:

    st.info("💡 画像をアップロードすると解析を開始します。")
    pass # import japanize_matplotlib を無効化
