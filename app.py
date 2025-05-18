import streamlit as st
import json
import os
from PIL import Image

st.set_page_config(page_title="Vogelmomenten", layout="wide")
st.title("🦉 Gevonden Vogelmomenten")
st.markdown("Blader door geanalyseerde fragmenten met bewegings- en vogelherkenning.")

DATA_FILE = "combined_with_feedback.json"

if not os.path.exists(DATA_FILE):
    st.error("Het bestand 'combined_with_feedback.json' is niet gevonden.")
    st.stop()

with open(DATA_FILE) as f:
    data = json.load(f)

feedback_filter = st.selectbox("Filter op feedback:", ["alles", "yes", "no", "unknown"])

col1, col2 = st.columns(2)
shown = 0

for item in data:
    if feedback_filter != "alles" and item.get("user_feedback") != feedback_filter:
        continue

    image_path = item["image_file"]
    if not os.path.exists(image_path):
        continue

    box_count = len(item.get("bounding_boxes", []))
    birds = item.get("bird_detections", [])
    bird_info = ", ".join([f"{b['class']} ({b['confidence']})" for b in birds]) if birds else "Geen"

    with (col1 if shown % 2 == 0 else col2):
        st.image(image_path, caption=f"{item['timestamp']}s – {bird_info}", use_column_width=True)
        with st.expander("Details"):
            st.write(f"**Afbeelding:** {image_path}")
            st.write(f"**Tijd:** {item['timestamp']} seconden")
            st.write(f"**Beweging:** {box_count} gebieden")
            st.write(f"**Vogels:** {bird_info}")
            st.write(f"**Feedback:** {item.get('user_feedback', 'unknown')}")
    shown += 1

if shown == 0:
    st.info("Geen momenten gevonden voor deze filter.")