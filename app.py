import streamlit as st
from src.pipeline.predict_pipeline import PredictionPipeline

st.set_page_config(
    page_title="Support Ticket Classification System",
    page_icon="🎫",
    layout="centered"
)

st.title("🎫 Support Ticket Classification")
st.write("Enter your support ticket description below to predict Ticket Type and Priority.")

# =========================
# Text Input
# =========================

user_input = st.text_area(
    "Enter Ticket Description",
    height=150,
    placeholder="Example: My product keeps crashing after the latest update..."
)

# =========================
# Predict Button
# =========================

if st.button("Predict"):

    if user_input.strip() == "":
        st.warning("Please enter a ticket description.")
    else:
        try:
            with st.spinner("Predicting..."):
                pipeline = PredictionPipeline()
                result = pipeline.predict(user_input)

            st.success("Prediction Successful")

            st.subheader("📌 Prediction Results")
            st.write(f"**Ticket Type:** {result['Ticket Type']}")
            st.write(f"**Ticket Priority:** {result['Ticket Priority']}")

        except Exception as e:
            st.error(f"Error occurred: {e}")