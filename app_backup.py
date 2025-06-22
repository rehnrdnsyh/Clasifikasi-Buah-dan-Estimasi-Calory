import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image
import pickle
import io

from helper.functions import (
    preprocess_image
)
from helper.scrap import scrape_nutrition_data

# Set page config for better appearance
st.set_page_config(
    page_title="🍎 Klasifikasi Buah & Nutrisi",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for colorful design
st.markdown("""
<style>
    /* Main background gradient */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1, #96CEB4, #FFEAA7);
        background-size: 300% 300%;
        animation: gradient 3s ease infinite;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .main-header h1 {
        color: white;
        font-size: 3rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        margin: 0;
    }
    
    .main-header p {
        color: white;
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    
    /* Custom cards */
    .info-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .success-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .error-card {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .nutrition-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    /* Colorful buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: bold;
        font-size: 1.1rem;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
    }
    
    /* Colorful tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        font-weight: bold;
        padding: 0.5rem 1rem;
    }
    
    /* Upload area styling */
    .uploadedFile {
        border: 3px dashed #667eea;
        border-radius: 15px;
        padding: 2rem;
        background: rgba(255,255,255,0.9);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# Load the model
with open('model/model.pkl', 'rb') as f:
    model = pickle.load(f)

fruits_list = ['Apel', 'Pisang', 'Alpukat', 'Ceri', 'Kiwi', 'Mangga', 'Jeruk', 'Nanas', 'Stroberi', 'Semangka']

# Database nutrisi buah per 100 gram untuk rekomendasi
fruits_nutrition_db = {
    'Apel': {'kalori': 52, 'lemak': 0.2, 'karbohidrat': 14, 'protein': 0.3, 'serat': 2.4},
    'Pisang': {'kalori': 89, 'lemak': 0.3, 'karbohidrat': 23, 'protein': 1.1, 'serat': 2.6},
    'Alpukat': {'kalori': 160, 'lemak': 15, 'karbohidrat': 9, 'protein': 2, 'serat': 7},
    'Ceri': {'kalori': 63, 'lemak': 0.2, 'karbohidrat': 16, 'protein': 1.1, 'serat': 2.1},
    'Kiwi': {'kalori': 61, 'lemak': 0.5, 'karbohidrat': 15, 'protein': 1.1, 'serat': 3},
    'Mangga': {'kalori': 60, 'lemak': 0.4, 'karbohidrat': 15, 'protein': 0.8, 'serat': 1.6},
    'Jeruk': {'kalori': 47, 'lemak': 0.1, 'karbohidrat': 12, 'protein': 0.9, 'serat': 2.4},
    'Nanas': {'kalori': 50, 'lemak': 0.1, 'karbohidrat': 13, 'protein': 0.5, 'serat': 1.4},
    'Stroberi': {'kalori': 32, 'lemak': 0.3, 'karbohidrat': 8, 'protein': 0.7, 'serat': 2},
    'Semangka': {'kalori': 30, 'lemak': 0.2, 'karbohidrat': 8, 'protein': 0.6, 'serat': 0.4}
}

def prepare_image_from_bytes(image_bytes):
    """
    Process image directly from bytes and predict fruit/vegetable class
    """
    try:
        # Preprocess the image for prediction
        image_array = preprocess_image(image_bytes)
        
        # Make prediction using the model
        prediction = model.predict(image_array)
        
        # Get the predicted class index
        pred_idx = np.argmax(prediction[0])
        
        # Get the confidence score
        confidence = float(prediction[0][pred_idx])
        
        # Set confidence threshold - if prediction confidence is too low, return None
        confidence_threshold = 0.6  # Adjust this value as needed
        
        if confidence < confidence_threshold:
            return None, confidence
        
        # Get the food name
        fruit_name = fruits_list[pred_idx] if pred_idx < len(fruits_list) else None
        
        return fruit_name, confidence
    except Exception as e:
        st.error(f"Error predicting image: {str(e)}")
        return None, 0.0

def get_fruit_recommendations(detected_fruit, goal):
    """
    Generate fruit combination recommendations based on detected fruit and goal
    goal: 'lose_weight' or 'gain_weight'
    """
    recommendations = {}
    
    if goal == 'lose_weight':
        # Buah rendah kalori untuk menurunkan berat badan
        low_cal_fruits = []
        for fruit, nutrition in fruits_nutrition_db.items():
            if nutrition['kalori'] <= 60:  # Kalori rendah
                low_cal_fruits.append((fruit, nutrition['kalori']))
        
        # Sort berdasarkan kalori terendah
        low_cal_fruits.sort(key=lambda x: x[1])
        
        recommendations = {
            'title': '🍃 Rekomendasi untuk Menurunkan Berat Badan',
            'description': 'Kombinasi buah rendah kalori dan tinggi serat untuk membantu diet',
            'combinations': [
                {
                    'name': 'Kombinasi Ultra Low-Cal',
                    'fruits': ['Semangka', 'Stroberi', 'Jeruk'],
                    'benefits': 'Sangat rendah kalori (30-47 kal/100g), tinggi air, membantu hidrasi',
                    'total_cal': sum([fruits_nutrition_db[f]['kalori'] for f in ['Semangka', 'Stroberi', 'Jeruk']]) // 3
                },
                {
                    'name': 'Kombinasi Serat Tinggi',
                    'fruits': ['Apel', 'Kiwi', 'Stroberi'], 
                    'benefits': 'Tinggi serat, memberikan rasa kenyang lebih lama',
                    'total_cal': sum([fruits_nutrition_db[f]['kalori'] for f in ['Apel', 'Kiwi', 'Stroberi']]) // 3
                },
                {
                    'name': 'Kombinasi Vitamin C',
                    'fruits': ['Jeruk', 'Kiwi', 'Stroberi'],
                    'benefits': 'Kaya vitamin C, meningkatkan metabolisme, rendah kalori',
                    'total_cal': sum([fruits_nutrition_db[f]['kalori'] for f in ['Jeruk', 'Kiwi', 'Stroberi']]) // 3
                }
            ]
        }
        
    elif goal == 'gain_weight':
        # Buah tinggi kalori untuk menambah berat badan
        high_cal_fruits = []
        for fruit, nutrition in fruits_nutrition_db.items():
            if nutrition['kalori'] >= 60:  # Kalori tinggi
                high_cal_fruits.append((fruit, nutrition['kalori']))
        
        # Sort berdasarkan kalori tertinggi
        high_cal_fruits.sort(key=lambda x: x[1], reverse=True)
        
        recommendations = {
            'title': '💪 Rekomendasi untuk Menambah Berat Badan',
            'description': 'Kombinasi buah tinggi kalori dan nutrisi untuk menambah massa tubuh sehat',
            'combinations': [
                {
                    'name': 'Kombinasi High-Energy',
                    'fruits': ['Alpukat', 'Pisang', 'Mangga'],
                    'benefits': 'Tinggi kalori dan lemak sehat, karbohidrat kompleks',
                    'total_cal': sum([fruits_nutrition_db[f]['kalori'] for f in ['Alpukat', 'Pisang', 'Mangga']]) // 3
                },
                {
                    'name': 'Kombinasi Protein & Kalori',
                    'fruits': ['Alpukat', 'Pisang', 'Ceri'],
                    'benefits': 'Kombinasi protein, lemak sehat, dan karbohidrat',
                    'total_cal': sum([fruits_nutrition_db[f]['kalori'] for f in ['Alpukat', 'Pisang', 'Ceri']]) // 3
                },
                {
                    'name': 'Kombinasi Natural Sugar',
                    'fruits': ['Pisang', 'Mangga', 'Ceri'],
                    'benefits': 'Gula alami untuk energi cepat, mendukung penambahan berat badan',
                    'total_cal': sum([fruits_nutrition_db[f]['kalori'] for f in ['Pisang', 'Mangga', 'Ceri']]) // 3
                }
            ]
        }
    
    # Tambahkan rekomendasi khusus berdasarkan buah yang terdeteksi
    if detected_fruit in fruits_nutrition_db:
        detected_nutrition = fruits_nutrition_db[detected_fruit]
        if goal == 'lose_weight' and detected_nutrition['kalori'] <= 60:
            recommendations['detected_fruit_note'] = f"✅ {detected_fruit} sangat cocok untuk diet Anda (hanya {detected_nutrition['kalori']} kalori/100g)"
        elif goal == 'gain_weight' and detected_nutrition['kalori'] >= 60:
            recommendations['detected_fruit_note'] = f"✅ {detected_fruit} bagus untuk menambah berat badan ({detected_nutrition['kalori']} kalori/100g)"
        elif goal == 'lose_weight' and detected_nutrition['kalori'] > 60:
            recommendations['detected_fruit_note'] = f"⚠️ {detected_fruit} cukup tinggi kalori ({detected_nutrition['kalori']} kal/100g), konsumsi dalam porsi kecil"
        else:
            recommendations['detected_fruit_note'] = f"ℹ️ {detected_fruit} rendah kalori ({detected_nutrition['kalori']} kal/100g), tambahkan buah tinggi kalori lainnya"
    
    return recommendations

def run():
    # Colorful sidebar
    with st.sidebar:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                   padding: 1.5rem; border-radius: 15px; color: white; margin-bottom: 1rem;">
            <h2>🌟 Tentang Aplikasi</h2>
            <p>Sistem klasifikasi buah bertenaga AI ini dapat mengidentifikasi 10 jenis buah berbeda dan memberikan rekomendasi nutrisi yang dipersonalisasi.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #ff9a56 0%, #ffad56 100%); 
                   padding: 1.5rem; border-radius: 15px; color: white; margin-bottom: 1rem;">
            <h3>🎯 Fitur:</h3>
            <ul>
                <li>🔍 Pengenalan Buah AI</li>
                <li>📊 Analisis Nutrisi</li>
                <li>🍽️ Rekomendasi Diet</li>
                <li>📱 Antarmuka Mudah Digunakan</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                   padding: 1.5rem; border-radius: 15px; color: white;">
            <h3>📈 Akurasi:</h3>
            <p>Model kami telah dilatih dengan ribuan gambar buah untuk memberikan hasil klasifikasi yang akurat.</p>
        </div>
        """, unsafe_allow_html=True)

    # Colorful header with animation
    st.markdown("""
    <div class="main-header">
        <h1>🍎 Klasifikasi Buah & Nutrisi 🥝</h1>
        <p>✨ Identifikasi buah dan dapatkan rekomendasi nutrisi yang dipersonalisasi ✨</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create columns for better layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>🌟 Cara kerja:</h3>
            <p>📸 Unggah gambar buah → 🤖 AI mengidentifikasinya → 📊 Dapatkan info nutrisi → 🍽️ Terima rekomendasi diet</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Upload image through streamlit interface
        img_file = st.file_uploader("🖼️ Pilih Gambar Buah", type=["jpg", "png", "jpeg"])
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <h4>🎯 Buah yang Didukung:</h4>
            <p>🍎 🍌 🥑 🍒 🥝</p>
            <p>🥭 🍊 🍍 🍓 🍉</p>
        </div>
        """, unsafe_allow_html=True)
    
    if img_file is not None:
        # Display the uploaded image with better styling
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            img = Image.open(img_file).resize((300, 300))
            st.image(img, caption="🖼️ Gambar yang Diunggah", use_container_width=True)
        
        # Center the predict button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            predict_button = st.button("🔍 Analisis Buah", use_container_width=True)
        
        # Add a prediction button
        if predict_button:
            # Show a spinner while processing
            with st.spinner("🔍 Menganalisis gambar buah Anda..."):
                # Get image bytes directly from uploaded file
                img_file.seek(0)  # Reset file pointer to beginning
                image_bytes = img_file.read()
                
                # Process image directly from bytes
                result, confidence = prepare_image_from_bytes(image_bytes)
                
                if result:
                    # Get nutrition data and portion info
                    nutrition_data, volume = scrape_nutrition_data(result)
                    portion_text = volume if volume else "100 gram"
                    
                    # Display prediction result with colorful card and portion info
                    st.markdown(f"""
                    <div class="success-card">
                        <h2>🎉 Hasil Prediksi</h2>
                        <h1 style="text-align: center; font-size: 3rem;">🍎 {result} ({portion_text}) 🍎</h1>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display nutrition information if available
                    if nutrition_data:
                        st.markdown("### 🌈 Informasi Nutrisi")
                        
                        # Create colorful nutrition cards
                        col1, col2, col3, col4 = st.columns(4)
                        
                        # Display each nutrition category with colorful cards
                        if "Kalori" in nutrition_data:
                            with col1:
                                st.markdown(f"""
                                <div style="background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%); 
                                           padding: 1.5rem; border-radius: 15px; text-align: center; color: white; margin: 0.5rem 0;">
                                    <h3>🔥 Kalori</h3>
                                    <h2>{nutrition_data["Kalori"]}</h2>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        if "Lemak" in nutrition_data:
                            with col2:
                                st.markdown(f"""
                                <div style="background: linear-gradient(135deg, #feca57 0%, #ff9ff3 100%); 
                                           padding: 1.5rem; border-radius: 15px; text-align: center; color: white; margin: 0.5rem 0;">
                                    <h3>🥑 Lemak</h3>
                                    <h2>{nutrition_data["Lemak"]}</h2>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        if "Karbohidrat" in nutrition_data:
                            with col3:
                                st.markdown(f"""
                                <div style="background: linear-gradient(135deg, #48dbfb 0%, #0abde3 100%); 
                                           padding: 1.5rem; border-radius: 15px; text-align: center; color: white; margin: 0.5rem 0;">
                                    <h3>🌾 Karbohidrat</h3>
                                    <h2>{nutrition_data["Karbohidrat"]}</h2>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        if "Protein" in nutrition_data:
                            with col4:
                                st.markdown(f"""
                                <div style="background: linear-gradient(135deg, #5f27cd 0%, #00d2d3 100%); 
                                           padding: 1.5rem; border-radius: 15px; text-align: center; color: white; margin: 0.5rem 0;">
                                    <h3>💪 Protein</h3>
                                    <h2>{nutrition_data["Protein"]}</h2>
                                </div>
                                """, unsafe_allow_html=True)
                        
                    else:
                        st.markdown("""
                        <div class="error-card">
                            <h3>⚠️ Informasi nutrisi tidak tersedia</h3>
                            <p>Mohon maaf, data nutrisi untuk buah ini sedang tidak tersedia</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Add recommendation section with colorful header
                    st.markdown("### 🍽️ Rekomendasi Diet")
                    
                    # Add reference sources
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%); 
                               padding: 1rem; border-radius: 10px; color: white; margin: 1rem 0;">
                        <h4>📚 Sumber Referensi Terpercaya:</h4>
                        <ul>
                            <li><a href="https://www.who.int/news-room/fact-sheets/detail/healthy-diet" target="_blank" style="color: #ddd;">WHO - Diet Sehat</a></li>
                            <li><a href="https://www.p2ptm.kemkes.go.id/" target="_blank" style="color: #ddd;">Kementerian Kesehatan RI - P2PTM</a></li>
                            <li><a href="https://www.nutrition.gov/" target="_blank" style="color: #ddd;">Nutrition.gov - Panduan Nutrisi</a></li>
                            <li><a href="https://www.fatsecret.co.id/" target="_blank" style="color: #ddd;">FatSecret Indonesia - Data Nutrisi</a></li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Create colorful tabs for different goals
                    tab1, tab2 = st.tabs(["🍃 Menurunkan Berat Badan", "💪 Menambah Berat Badan"])
                    
                    with tab1:
                        recommendations_lose = get_fruit_recommendations(result, 'lose_weight')
                        
                        st.markdown(f"### {recommendations_lose['title']}")
                        st.write(recommendations_lose['description'])
                        
                        # Display note about detected fruit
                        if 'detected_fruit_note' in recommendations_lose:
                            st.info(recommendations_lose['detected_fruit_note'])
                        
                        # Display combinations
                        for i, combo in enumerate(recommendations_lose['combinations']):
                            with st.expander(f"{combo['name']} (Rata-rata: {combo['total_cal']} kal/100g)"):
                                st.write(f"**Buah yang disarankan:** {', '.join(combo['fruits'])}")
                                st.write(f"**Manfaat:** {combo['benefits']}")
                                
                                # Show individual nutrition for each fruit in combination
                                cols = st.columns(len(combo['fruits']))
                                for j, fruit in enumerate(combo['fruits']):
                                    if fruit in fruits_nutrition_db:
                                        with cols[j]:
                                            nutrition = fruits_nutrition_db[fruit]
                                            st.metric(
                                                label=fruit,
                                                value=f"{nutrition['kalori']} kal",
                                                delta=f"Serat: {nutrition['serat']}g"
                                            )
                    
                    with tab2:
                        recommendations_gain = get_fruit_recommendations(result, 'gain_weight')
                        
                        st.markdown(f"### {recommendations_gain['title']}")
                        st.write(recommendations_gain['description'])
                        
                        # Display note about detected fruit
                        if 'detected_fruit_note' in recommendations_gain:
                            st.info(recommendations_gain['detected_fruit_note'])
                        
                        # Display combinations
                        for i, combo in enumerate(recommendations_gain['combinations']):
                            with st.expander(f"{combo['name']} (Rata-rata: {combo['total_cal']} kal/100g)"):
                                st.write(f"**Buah yang disarankan:** {', '.join(combo['fruits'])}")
                                st.write(f"**Manfaat:** {combo['benefits']}")
                                
                                # Show individual nutrition for each fruit in combination
                                cols = st.columns(len(combo['fruits']))
                                for j, fruit in enumerate(combo['fruits']):
                                    if fruit in fruits_nutrition_db:
                                        with cols[j]:
                                            nutrition = fruits_nutrition_db[fruit]
                                            st.metric(
                                                label=fruit,
                                                value=f"{nutrition['kalori']} kal",
                                                delta=f"Protein: {nutrition['protein']}g"
                                            )
                else:
                    # Colorful error message
                    st.markdown("""
                    <div class="error-card">
                        <h2>❌ Tidak Dapat Mengidentifikasi Buah</h2>
                        <h3>🔄 Gunakan foto buah yang sesuai dengan sistem</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Tampilkan daftar buah yang dapat diprediksi dengan styling colorful
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                               padding: 2rem; border-radius: 15px; color: white; margin: 1rem 0;">
                        <h3>📝 Buah yang Didukung oleh Sistem Kami:</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("""
                        <div style="background: linear-gradient(135deg, #ff9a56 0%, #ffad56 100%); 
                                   padding: 1.5rem; border-radius: 15px; color: white; margin: 0.5rem;">
                            <h4>🍎 Apel</h4>
                            <h4>🍌 Pisang</h4>
                            <h4>🥑 Alpukat</h4>
                            <h4>🍒 Ceri</h4>
                            <h4>🥝 Kiwi</h4>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("""
                        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                   padding: 1.5rem; border-radius: 15px; color: white; margin: 0.5rem;">
                            <h4>🥭 Mangga</h4>
                            <h4>🍊 Jeruk</h4>
                            <h4>🍍 Nanas</h4>
                            <h4>🍓 Stroberi</h4>
                            <h4>🍉 Semangka</h4>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Tips dengan styling colorful
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                               padding: 2rem; border-radius: 15px; color: white; margin: 1rem 0;">
                        <h3>💡 Tips untuk Hasil Terbaik:</h3>
                        <ul style="font-size: 1.1rem; line-height: 1.8;">
                            <li>📸 Gunakan foto buah dari daftar di atas</li>
                            <li>🔍 Pastikan gambar fokus dan jelas</li>
                            <li>✨ Gunakan foto buah segar yang utuh</li>
                            <li>🎯 Hindari gambar dengan background yang ramai</li>
                            <li>💡 Pastikan pencahayaan cukup baik</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)


# Run the application
if __name__ == "__main__":
    run()
