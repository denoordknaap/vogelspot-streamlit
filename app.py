import streamlit as st
import os
import cv2
import json
import tempfile
from PIL import Image
from ultralytics import YOLO
from moviepy.editor import VideoFileClip
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events

st.set_page_config(page_title="Vogelspotter Upload", layout="wide")
st.title("🦉 Upload je vogelvideo")

uploaded_file = st.file_uploader("📤 Upload een .mp4 bestand", type=["mp4"])

if uploaded_file:
    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, uploaded_file.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success("✅ Video geüpload. Analyse wordt gestart...")

    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    interval = int(frame_rate * 2)

    metadata = []
    timestamps = []
    confidences = []
    saved = 0

    while True:
        frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % interval == 0:
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            img_path = os.path.join(temp_dir, f"frame_{saved}.jpg")
            cv2.imwrite(img_path, frame)

            results = model(frame, verbose=False)[0]
            birds = []
            conf_level = 0
            for box in results.boxes:
                cls = results.names[int(box.cls)]
                if "bird" in cls.lower():
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = round(box.conf[0].item(), 2)
                    conf_level = max(conf_level, conf)
                    birds.append({
                        "class": cls,
                        "confidence": conf,
                        "box": {
                            "x": int(x1),
                            "y": int(y1),
                            "w": int(x2 - x1),
                            "h": int(y2 - y1)
                        }
                    })

            clip_path = os.path.join(temp_dir, f"clip_{saved}.mp4")
            with VideoFileClip(video_path).subclip(max(timestamp - 2, 0), timestamp + 2) as clip:
                clip.write_videofile(clip_path, codec="libx264", audio=False, verbose=False, logger=None)

            metadata.append({
                "id": saved,
                "timestamp": round(timestamp, 2),
                "image_file": img_path,
                "clip_file": clip_path,
                "bird_detections": birds,
                "confidence": conf_level,
                "interesting": None
            })
            timestamps.append(round(timestamp, 2))
            confidences.append(conf_level)
            saved += 1

    cap.release()
    st.success(f"✅ Analyse voltooid: {saved} fragmenten verwerkt.")

    st.subheader("🕐 Tijdlijn met vogelactiviteit:")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=[1] * len(timestamps),
        mode="markers",
        marker=dict(
            size=12,
            color=confidences,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Confidence")
        ),
        text=[f"Fragment {i} op {t}s (Confidence: {c})" for i, (t, c) in enumerate(zip(timestamps, confidences))],
        customdata=list(range(len(timestamps))),
        hoverinfo="text"
    ))
    fig.update_layout(
        height=200,
        yaxis=dict(showticklabels=False),
        xaxis_title="Tijd (seconden)",
        title="Vogelactiviteit over tijd",
        margin=dict(t=30, b=10),
    )

    selected_points = plotly_events(fig, click_event=True, select_event=False)

    if selected_points:
        selected_index = selected_points[0]['pointIndex']
        selected = metadata[selected_index]
        st.subheader("🔎 Geselecteerd fragment uit tijdlijn:")
        st.image(selected["image_file"], caption=f"{selected['timestamp']}s", width=300)
        st.video(selected["clip_file"])
        if selected["bird_detections"]:
            for det in selected["bird_detections"]:
                st.write(f"- {det['class']} ({det['confidence']}) @ {det['box']}")

    st.subheader("📸 Gevonden momenten:")

    cols = st.columns(4)

    for i, entry in enumerate(metadata):
        with cols[i % 4]:
            st.image(entry["image_file"], caption=f"{entry['timestamp']}s", width=180)

            with st.expander("Details"):
                if entry["bird_detections"]:
                    for det in entry["bird_detections"]:
                        st.write(f"- {det['class']} ({det['confidence']}) @ {det['box']}")
                else:
                    st.write("Geen vogel gedetecteerd.")

                st.write(f"🎞️ Fragment: van {max(entry['timestamp'] - 2, 0)}s tot {entry['timestamp'] + 2}s")
                st.video(entry["clip_file"])

                feedback = st.radio(
                    f"Was dit fragment interessant? (Fragment {i})",
                    ["Onbekend", "Ja", "Nee"],
                    index=0,
                    key=f"feedback_{i}"
                )
                metadata[i]["interesting"] = None if feedback == "Onbekend" else feedback

    json_path = os.path.join(temp_dir, "resultaten.json")
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)

    st.download_button("📥 Download metadata JSON", data=open(json_path).read(), file_name="resultaten.json")
