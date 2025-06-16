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
            return None
        
        # Get the food name
        fruit_name = fruits_list[pred_idx] if pred_idx < len(fruits_list) else None
        
        return fruit_name
    except Exception as e:
        st.error(f"Error predicting image: {str(e)}")
        return None

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
    # Set up the UI
    st.title("Fruits🍍 Classification")
    st.write("Upload an image of a fruit or vegetable to classify it")
    
    # Upload image through streamlit interface
    img_file = st.file_uploader("Choose an Image", type=["jpg", "png", "jpeg"])
    
    if img_file is not None:
        # Display the uploaded image
        img = Image.open(img_file).resize((250, 250))
        st.image(img, use_container_width=False, caption="Uploaded Image")
        
        # Add a prediction button
        if st.button("Predict"):
            # Show a spinner while processing
            with st.spinner("Analyzing image..."):
                # Get image bytes directly from uploaded file
                img_file.seek(0)  # Reset file pointer to beginning
                image_bytes = img_file.read()
                
                # Process image directly from bytes
                result = prepare_image_from_bytes(image_bytes)
                
                if result:
                    # Display prediction result
                    st.success(f"**Predicted : {result}**")
                    
                    # Display nutrition information if available
                    nutrition_data, volume = scrape_nutrition_data(result)
                    if nutrition_data:
                        st.subheader("📊 Informasi Nutrisi")
                        
                        # Create columns for nutrition display
                        col1, col2, col3, col4 = st.columns(4)
                        
                        # Display each nutrition category in its own column
                        if "Kalori" in nutrition_data:
                            with col1:
                                st.metric(
                                    label="🔥 Kalori",
                                    value=nutrition_data["Kalori"]
                                )
                        
                        if "Lemak" in nutrition_data:
                            with col2:
                                st.metric(
                                    label="🥑 Lemak", 
                                    value=nutrition_data["Lemak"]
                                )
                        
                        if "Karbohidrat" in nutrition_data:
                            with col3:
                                st.metric(
                                    label="🌾 Karbohidrat",
                                    value=nutrition_data["Karbohidrat"]
                                )
                        
                        if "Protein" in nutrition_data:
                            with col4:
                                st.metric(
                                    label="💪 Protein",
                                    value=nutrition_data["Protein"]
                                )
                        
                        # Display portion information
                        if volume:
                            st.info(f"📏 **Porsi: {volume}**")
                        else:
                            st.info("📏 **Porsi: 100 gram**")
                    else:
                        st.warning("**Informasi nutrisi tidak tersedia**")
                    
                    # Add recommendation section
                    st.subheader("🍽️ Rekomendasi Kombinasi Buah")
                    
                    # Create tabs for different goals
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
                    st.error("❌ **Tidak dapat mengidentifikasi buah dari gambar**")
                    st.warning("🔄 **Coba gunakan foto buah yang lain**")
                    st.info("💡 **Tips:**")
                    st.write("- Pastikan gambar fokus dan jelas")
                    st.write("- Gunakan foto buah segar yang utuh")
                    st.write("- Hindari gambar dengan background yang ramai")
                    st.write("- Pastikan pencahayaan cukup baik")


# Run the application
if __name__ == "__main__":
    run()