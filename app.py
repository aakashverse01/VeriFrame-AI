
import os
import tempfile
from io import BytesIO
from datetime import datetime
from xml.sax.saxutils import escape

import cv2
import numpy as np
import torch
import streamlit as st

from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    KeepTogether
)

from model import create_model
from preprocess import preprocess_frame


# ================= APP SETTINGS =================

st.set_page_config(
    page_title="VeriFrame AI | Video Authenticity Analysis",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ================= DESIGN =================

st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(
            ellipse at 85% 5%,
            rgba(79, 70, 229, 0.15),
            transparent 35%
        ),
        linear-gradient(135deg, #080D19, #0D1525);
    color: #EAF0FC;
}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1160px;
    padding-top: 4.5rem !important;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    color: #F8FAFC !important;
}

p {
    color: #C3CEE2;
}

.brand {
    display: flex;
    align-items: center;
    min-height: 45px;
    line-height: 1.4;
    white-space: nowrap;
    font-size: 25px;
    font-weight: 800;
    letter-spacing: -0.7px;
    color: #F8FAFC;
}

.brand span {
    color: #818CF8;
}

.hero {
    padding: 65px 30px;
    margin: 25px 0 30px;
    text-align: center;
    border: 1px solid #273550;
    border-radius: 24px;
    background:
        radial-gradient(
            circle at 50% 0%,
            rgba(99, 102, 241, 0.20),
            transparent 65%
        ),
        #111A2C;
}

.hero .eyebrow {
    color: #A5B4FC;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.hero h1 {
    font-size: clamp(38px, 5vw, 64px);
    font-weight: 850;
    letter-spacing: -1.8px;
    margin: 15px 0;
}

.hero .tagline {
    color: #C5D0E5;
    font-size: 19px;
    max-width: 650px;
    margin: auto;
    line-height: 1.8;
}

.feature {
    background: #131E31;
    border: 1px solid #293852;
    padding: 25px;
    border-radius: 16px;
    min-height: 170px;
}

.feature h3 {
    color: #A5B4FC !important;
    font-size: 21px;
}

.feature p {
    color: #B9C6DB;
    line-height: 1.7;
}

.section-heading {
    color: #F8FAFC;
    font-size: 28px;
    font-weight: 750;
    margin: 15px 0;
}

.muted {
    color: #A9B8D0;
    font-size: 15px;
}

div.stButton > button {
    border: 1px solid #35425D;
    border-radius: 11px;
    background: #17233A;
    color: #F1F5FF;
    font-weight: 650;
    min-height: 42px;
}

div.stButton > button:hover {
    background: #233454;
    border-color: #818CF8;
    color: white;
}

div.stButton > button[kind="primary"] {
    background: #4F46E5;
    border: 1px solid #6366F1;
    color: white;
}

div.stButton > button[kind="primary"]:hover {
    background: #6366F1;
    color: white;
}

[data-testid="stMetric"] {
    background: #131E31;
    border: 1px solid #293852;
    border-radius: 14px;
    padding: 17px;
}

[data-testid="stFileUploader"] {
    background: #111A2C;
    border: 1px solid #293852;
    border-radius: 15px;
    padding: 12px;
}

hr {
    border-color: #293852;
}
</style>
""", unsafe_allow_html=True)


# ================= AI MODEL =================

@st.cache_resource
def load_ai_model():
    model = create_model()
    model.eval()
    return model


# ================= PDF GENERATOR =================

def create_pdf_report(
    filename,
    size_mb,
    duration,
    fps,
    width,
    height,
    total_frames,
    extracted_count,
    ai_processed_count,
    timestamps
):
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=(595.28, 841.89),
        rightMargin=45,
        leftMargin=45,
        topMargin=45,
        bottomMargin=45
    )

    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="BrandTitle",
        parent=styles["Title"],
        fontSize=23,
        leading=28,
        textColor=colors.HexColor("#4338CA"),
        alignment=TA_CENTER,
        spaceAfter=10
    ))

    styles.add(ParagraphStyle(
        name="ReportSubtitle",
        parent=styles["Normal"],
        fontSize=12,
        textColor=colors.HexColor("#475569"),
        alignment=TA_CENTER,
        spaceAfter=22
    ))

    styles.add(ParagraphStyle(
        name="ReportSection",
        parent=styles["Heading2"],
        fontSize=12,
        textColor=colors.HexColor("#312E81"),
        spaceBefore=17,
        spaceAfter=10
    ))

    styles.add(ParagraphStyle(
        name="ReportBody",
        parent=styles["BodyText"],
        fontSize=9,
        leading=15,
        textColor=colors.HexColor("#334155")
    ))

    story = []

    story.append(
        Paragraph("VERIFRAME AI", styles["BrandTitle"])
    )

    story.append(
        Paragraph(
            "VIDEO ANALYSIS REPORT",
            styles["ReportSubtitle"]
        )
    )

    story.append(
        Paragraph(
            "Generated: " +
            datetime.now().strftime("%d %B %Y, %I:%M %p"),
            styles["ReportBody"]
        )
    )

    def section(title):
        story.append(
            Paragraph(title, styles["ReportSection"])
        )

    def info_table(rows):
        table_data = []

        for key, value in rows:
            table_data.append([
                Paragraph(
                    escape(str(key)),
                    styles["ReportBody"]
                ),
                Paragraph(
                    escape(str(value)),
                    styles["ReportBody"]
                )
            ])

        table = Table(
            table_data,
            colWidths=[190, 305],
            hAlign="LEFT"
        )

        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1),
             colors.HexColor("#EEF2FF")),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [
                colors.white,
                colors.HexColor("#F8FAFC")
            ]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 9),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("LINEBELOW", (0, 0), (-1, -1),
             0.4, colors.HexColor("#E2E8F0"))
        ]))

        story.append(table)

    section("1. Video Information")

    info_table([
        ("File Name", filename),
        ("File Size", f"{size_mb:.2f} MB"),
        ("Duration", f"{duration:.2f} seconds"),
        ("Frame Rate", f"{fps:.2f} FPS"),
        ("Resolution", f"{width} x {height}"),
        ("Total Video Frames", total_frames)
    ])

    section("2. Frame Analysis")

    info_table([
        ("Frames Extracted", extracted_count),
        ("AI Frames Processed", ai_processed_count)
    ])

    section("3. Extracted Frame Timestamps")

    timestamp_rows = [
        (f"Frame {i}", f"{time:.2f} seconds")
        for i, time in enumerate(timestamps, start=1)
    ]

    info_table(timestamp_rows)

    section("4. AI Processing Pipeline")

    info_table([
        ("Architecture", "ResNet18"),
        ("Framework", "PyTorch"),
        ("Input Tensor", "[1, 3, 224, 224]"),
        ("Output Tensor", "[1, 2]"),
        ("Forward Pass", "Successful")
    ])

    section("5. Model Development Status")

    info_table([
        ("Model Weights", "Untrained"),
        ("Authenticity Classification", "Not Available"),
        ("Real/Fake Prediction", "Not Generated"),
        ("Training and Validation", "Pending")
    ])

    story.append(Spacer(1, 18))

    note = (
        "<b>Important:</b> This report documents video "
        "inspection and neural network pipeline execution. "
        "The ResNet18 model has not been trained to classify "
        "authentic or manipulated videos. No authenticity "
        "decision or confidence score is provided."
    )

    story.append(
        Paragraph(note, styles["ReportBody"])
    )

    story.append(Spacer(1, 25))

    story.append(
        Paragraph(
            "VeriFrame AI | See Beyond the Frame.",
            styles["ReportSubtitle"]
        )
    )

    doc.build(story)

    pdf_bytes = buffer.getvalue()
    buffer.close()

    return pdf_bytes


# ================= NAVIGATION =================

if "page" not in st.session_state:
    st.session_state.page = "Home"


def navigate(page):
    st.session_state.page = page
    st.rerun()


logo, home, analyze, about = st.columns(
    [3, 1, 1.4, 1],
    vertical_alignment="center"
)

with logo:
    st.markdown(
        '<div class="brand">◈ VeriFrame&nbsp;<span>AI</span></div>',
        unsafe_allow_html=True
    )

with home:
    if st.button("Home", use_container_width=True):
        navigate("Home")

with analyze:
    if st.button("Analyze Video", use_container_width=True):
        navigate("Analyze")

with about:
    if st.button("About", use_container_width=True):
        navigate("About")

st.divider()


# ================= HOME =================

if st.session_state.page == "Home":

    st.markdown("""
    <div class="hero">
        <div class="eyebrow">Intelligent Video Analysis</div>
        <h1>See Beyond the Frame.</h1>
        <p class="tagline">
            Explore video authenticity with intelligent
            frame analysis and computer vision.
            Upload, inspect and understand your video
            through a streamlined experience.
        </p>
    </div>
    """, unsafe_allow_html=True)

    left, middle, right = st.columns(3)

    with left:
        st.markdown("""
        <div class="feature">
            <h3>01 — Video Upload</h3>
            <p>
                Upload supported video files
                for detailed inspection.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with middle:
        st.markdown("""
        <div class="feature">
            <h3>02 — Frame Analysis</h3>
            <p>
                Extract representative frames
                and inspect video properties.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown("""
        <div class="feature">
            <h3>03 — AI Verification</h3>
            <p>
                Deep learning classification
                integration is in development.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    a, b, c = st.columns([1, 2, 1])

    with b:
        if st.button(
            "Start Video Analysis",
            type="primary",
            use_container_width=True
        ):
            navigate("Analyze")

    st.caption(
        "Available capabilities: Video inspection, "
        "frame extraction and neural network pipeline testing."
    )


# ================= ANALYZE =================

elif st.session_state.page == "Analyze":

    st.markdown(
        '<div class="section-heading">Video Analysis Studio</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="muted">Upload a video to inspect '
        'its properties, frames and AI processing pipeline.</p>',
        unsafe_allow_html=True
    )

    uploaded_video = st.file_uploader(
        "Select a video file",
        type=["mp4", "avi", "mov"],
        key="video_file"
    )

    if uploaded_video is not None:

        size_mb = uploaded_video.size / (1024 * 1024)

        if size_mb > 200:
            st.error("Please upload a video under 200 MB.")
            st.stop()

        st.success("Video ready for analysis")

        info1, info2 = st.columns(2)

        with info1:
            st.write("**File name:**", uploaded_video.name)

        with info2:
            st.write("**File size:**", f"{size_mb:.2f} MB")

        st.subheader("Video Preview")
        st.video(uploaded_video)

        if st.button(
            "Run Frame Analysis",
            type="primary",
            use_container_width=True
        ):

            temp_path = None
            cap = None

            try:
                with st.spinner("Analyzing video and AI pipeline..."):

                    suffix = os.path.splitext(
                        uploaded_video.name
                    )[1].lower()

                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=suffix
                    ) as temp_file:
                        temp_file.write(
                            uploaded_video.getbuffer()
                        )
                        temp_path = temp_file.name

                    cap = cv2.VideoCapture(temp_path)

                    if not cap.isOpened():
                        raise ValueError(
                            "Unable to read the uploaded video."
                        )

                    fps = cap.get(cv2.CAP_PROP_FPS)
                    total_frames = int(
                        cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    )
                    width = int(
                        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    )
                    height = int(
                        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    )

                    if fps <= 0 or total_frames <= 0:
                        raise ValueError(
                            "Invalid video metadata."
                        )

                    duration = total_frames / fps

                    indices = np.linspace(
                        0,
                        total_frames - 1,
                        min(8, total_frames),
                        dtype=int
                    )

                    frames = []
                    timestamps = []
                    ai_processed_count = 0

                    ai_model = load_ai_model()

                    for index in indices:

                        cap.set(
                            cv2.CAP_PROP_POS_FRAMES,
                            int(index)
                        )

                        success, frame = cap.read()

                        if not success:
                            continue

                        rgb = cv2.cvtColor(
                            frame,
                            cv2.COLOR_BGR2RGB
                        )

                        preview = cv2.resize(
                            rgb, (320, 180)
                        )

                        frames.append(preview)
                        timestamps.append(index / fps)

                        input_tensor = preprocess_frame(frame)

                        with torch.no_grad():
                            model_output = ai_model(input_tensor)

                        if model_output.shape != (1, 2):
                            raise ValueError(
                                "Unexpected ResNet18 output shape."
                            )

                        ai_processed_count += 1

                if not frames:
                    st.error("No usable frames were extracted.")

                else:
                    st.success("Video analysis completed")

                    st.subheader("Video Properties")

                    m1, m2, m3, m4 = st.columns(4)

                    m1.metric(
                        "Duration",
                        f"{duration:.1f} sec"
                    )

                    m2.metric(
                        "Frame Rate",
                        f"{fps:.1f} FPS"
                    )

                    m3.metric(
                        "Resolution",
                        f"{width} × {height}"
                    )

                    m4.metric(
                        "Total Frames",
                        total_frames
                    )

                    st.subheader("Extracted Video Frames")

                    columns = st.columns(4)

                    for i, frame in enumerate(frames):
                        with columns[i % 4]:
                            st.image(
                                frame,
                                caption=(
                                    f"Frame {i + 1} • "
                                    f"{timestamps[i]:.1f}s"
                                ),
                                use_container_width=True
                            )

                    st.subheader("AI Pipeline Status")

                    p1, p2, p3 = st.columns(3)

                    p1.metric(
                        "Frames Extracted",
                        len(frames)
                    )

                    p2.metric(
                        "AI Frames Processed",
                        ai_processed_count
                    )

                    p3.metric(
                        "Model Output",
                        "2 Classes"
                    )

                    st.success(
                        "Video frames successfully passed "
                        "through the ResNet18 architecture."
                    )

                    st.write(
                        "**Input format:** "
                        "RGB tensor [1, 3, 224, 224]"
                    )

                    st.write(
                        "**Model architecture:** ResNet18"
                    )

                    st.write(
                        "**Forward pass:** Successful"
                    )

                    st.warning(
                        "Model weights are currently untrained. "
                        "Real/Fake classification is not available."
                    )

                    # Create professional PDF report
                    pdf_report = create_pdf_report(
                        filename=uploaded_video.name,
                        size_mb=size_mb,
                        duration=duration,
                        fps=fps,
                        width=width,
                        height=height,
                        total_frames=total_frames,
                        extracted_count=len(frames),
                        ai_processed_count=ai_processed_count,
                        timestamps=timestamps
                    )

                    st.subheader("Analysis Report")

                    st.write(
                        "Download your video analysis results "
                        "as a professional PDF document."
                    )

                    st.download_button(
                        label="📄 Download PDF Analysis Report",
                        data=pdf_report,
                        file_name="VeriFrame_Analysis_Report.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )

            except Exception as error:
                st.error(f"Video analysis failed: {error}")

            finally:
                if cap is not None:
                    cap.release()

                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)


# ================= ABOUT =================

elif st.session_state.page == "About":

    st.title("About VeriFrame AI")

    st.write("""
    VeriFrame AI is a video authenticity analysis
    application built to inspect digital video content
    through computer vision and neural network processing.

    The application supports video uploading,
    metadata inspection, frame extraction and
    ResNet18-compatible preprocessing.

    Its current AI pipeline can pass extracted video
    frames through a ResNet18 architecture.

    Authenticity classification will be available
    after model training and validation.
    """)

    st.subheader("Technology Stack")

    st.write("**Interface:** Streamlit")
    st.write("**Video Processing:** Python + OpenCV")
    st.write("**Numerical Computing:** NumPy")
    st.write("**Deep Learning:** PyTorch + ResNet18")
    st.write("**PDF Reports:** ReportLab")

    st.subheader("System Status")

    st.success(
        "Video processing and ResNet18 pipeline available"
    )

    st.warning(
        "Model training and authenticity classification pending"
    )


# ================= FOOTER =================

st.divider()

st.caption(
    "© 2026 VeriFrame AI · See Beyond the Frame."
)
