import os
import sys
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

def set_cell_background(cell, fill_hex):
    tcPr = cell._element.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def create_docx(filename):
    doc = Document()
    
    # Page setup - 0.8 inch margins
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.8)
        section.right_margin = Inches(0.8)

    # Palette
    NAVY = RGBColor(15, 32, 67)
    TEAL = RGBColor(14, 116, 144)
    DARK_GRAY = RGBColor(51, 65, 85)
    LIGHT_BG = "F1F5F9"
    HEADER_BG = "0F2043"

    # Title
    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_after = Pt(4)
    run_title = p_title.add_run("CAMPUS SENTINEL")
    run_title.font.name = 'Arial'
    run_title.font.size = Pt(26)
    run_title.font.bold = True
    run_title.font.color.rgb = NAVY

    p_sub = doc.add_paragraph()
    p_sub.paragraph_format.space_after = Pt(16)
    run_sub = p_sub.add_run("Comprehensive Technical Report, Codebase Audit & Strategic AI Roadmap")
    run_sub.font.name = 'Arial'
    run_sub.font.size = Pt(13)
    run_sub.font.italic = True
    run_sub.font.color.rgb = TEAL

    # Meta banner
    p_meta = doc.add_paragraph()
    p_meta.paragraph_format.space_after = Pt(20)
    run_meta = p_meta.add_run("Repository: d:\\CampusSentinal  |  Generated for Architecture & AI Discovery  |  Date: October 2026")
    run_meta.font.name = 'Arial'
    run_meta.font.size = Pt(9.5)
    run_meta.font.color.rgb = DARK_GRAY

    def add_h1(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(16)
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.font.name = 'Arial'
        run.font.size = Pt(16)
        run.font.bold = True
        run.font.color.rgb = NAVY

    def add_h2(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.font.name = 'Arial'
        run.font.size = Pt(12)
        run.font.bold = True
        run.font.color.rgb = TEAL

    def add_p(text, bold_prefix="", italic=False):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(5)
        p.paragraph_format.line_spacing = 1.15
        if bold_prefix:
            r_pre = p.add_run(bold_prefix)
            r_pre.font.name = 'Arial'
            r_pre.font.size = Pt(10)
            r_pre.font.bold = True
            r_pre.font.color.rgb = NAVY
        r = p.add_run(text)
        r.font.name = 'Arial'
        r.font.size = Pt(10)
        r.font.italic = italic
        r.font.color.rgb = DARK_GRAY

    def add_bullet(text, bold_prefix=""):
        p = doc.add_paragraph(style='List Bullet')
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.line_spacing = 1.15
        if bold_prefix:
            r_pre = p.add_run(bold_prefix)
            r_pre.font.name = 'Arial'
            r_pre.font.size = Pt(9.5)
            r_pre.font.bold = True
            r_pre.font.color.rgb = NAVY
        r = p.add_run(text)
        r.font.name = 'Arial'
        r.font.size = Pt(9.5)
        r.font.color.rgb = DARK_GRAY

    # Section 1
    add_h1("1. Executive Summary & System Overview")
    add_p(
        "Campus Sentinel is an enterprise-oriented computer vision Network Video Recorder (NVR) and automated surveillance system specifically engineered for educational campuses. "
        "The system replaces conventional passive security monitors with active AI agents capable of continuous person tracking, real-time crowd occupancy counting, polygon-bounded restricted zone (ROI) intrusion detection, and automated exam proctoring with biomechanical malpractice detection."
    )
    add_p(
        "By leveraging deep learning models (YOLOv8 & YOLOv8-Pose) synchronized with multi-process video ingest and asynchronous WebSocket telemetry, Campus Sentinel detects illicit behaviors in sub-second latency, logs evidence directly into an SQLite audit trail, and automatically packages synchronized 15-second MP4 evidence clips and high-resolution snapshots for review."
    )

    # Section 2
    add_h1("2. Full Codebase & File Inventory")
    add_p("A structured audit of all assets, models, scripts, configurations, and storage directories:")

    table_data = [
        ("File / Path", "Type / Size", "Primary Functional Role"),
        ("run.py", "Python (8.0 KB)", "Master entry point; coordinates FastAPI, WebSocket broadcasting, MJPEG streaming, and multi-process camera workers."),
        ("core/inference.py", "Python (16.7 KB)", "Core CV engine; CameraPipeline class with decoupled background AI inference, ByteTrack, 17-keypoint pose heuristics, and ring buffers."),
        ("core/config_manager.py", "Python (2.3 KB)", "Configuration manager; reads/writes camera entries, operating modes (CLASSROOM vs EXAM), and ROI coordinates in JSON."),
        ("core/event_manager.py", "Python (2.5 KB)", "Audit logging service; initializes SQLite schema, records incidents, writes JPEG snapshots, and links video evidence clips."),
        ("ui/templates/index.html", "HTML/JS/CSS (22.6 KB)", "Cyber-dark glassmorphism enterprise dashboard with live multi-camera grid, scrubber, 1x/2x/4x speed controls, and incident modal."),
        ("storage/config.json", "JSON Config", "Persistent camera registry holding live feeds, webcam indices, IP RTSP URLs, modes, and polygon boundary arrays."),
        ("storage/db/campus.sqlite3", "SQLite DB (20.5 KB)", "Relational database maintaining timestamped incident records, event types, camera IDs, snapshot paths, and clip paths."),
        ("storage/clips/", "Directory", "Repository for saved MP4 incident video recordings (both automated 15-second incident clips and manual 2-minute captures)."),
        ("storage/screenshots/", "Directory", "Repository for high-resolution JPEG snapshots captured at the exact moment an anomaly is detected."),
        ("storage/model school-Channel 50.MP4", "Test Video (85.2 MB)", "Pre-recorded CCTV classroom/exam monitoring video used for test benching and verification."),
        ("yolov8n.pt", "Model (6.5 MB)", "Ultralytics YOLOv8 Nano object detection weights; utilized in CLASSROOM mode for person detection & ROI intrusion."),
        ("yolov8s.pt", "Model (22.6 MB)", "Ultralytics YOLOv8 Small object detection weights; higher accuracy fallback for object detection."),
        ("yolov8n-pose.pt", "Model (6.8 MB)", "Ultralytics YOLOv8 Nano Pose weights; detects 17 skeletal landmarks per individual in EXAM mode."),
        ("yolov8s-pose.pt", "Model (23.5 MB)", "Ultralytics YOLOv8 Small Pose weights; enhanced keypoint localization for crowded or distant classrooms."),
        ("main.py", "Python (3.7 KB)", "Standalone OpenCV scale-testing prototype evaluating multi-student head-turn heuristics via BoT-SORT."),
        ("detailed skeletal.txt", "Text (4.9 KB)", "Early prototype design notes analyzing MediaPipe Pose, IoU collision for fight detection, and wrist extension reach."),
        ("readme.txt", "Text (175 B)", "Deployment prerequisite notes (pip install fastapi uvicorn websockets ultralytics).")
    ]

    t = doc.add_table(rows=len(table_data), cols=3)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False

    col_widths = [Inches(2.2), Inches(1.3), Inches(3.4)]

    for row_idx, row in enumerate(t.rows):
        is_header = (row_idx == 0)
        for col_idx, cell in enumerate(row.cells):
            cell.width = col_widths[col_idx]
            text = table_data[row_idx][col_idx]
            cell.text = ""
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(3)
            p.paragraph_format.space_after = Pt(3)
            r = p.add_run(text)
            r.font.name = 'Arial'
            r.font.size = Pt(8.5 if not is_header else 9)
            if is_header:
                r.font.bold = True
                r.font.color.rgb = RGBColor(255, 255, 255)
                set_cell_background(cell, HEADER_BG)
            else:
                r.font.color.rgb = DARK_GRAY
                if row_idx % 2 == 1:
                    set_cell_background(cell, LIGHT_BG)

    # Section 3
    add_h1("3. Core Architecture & Workflow Analysis")
    add_h2("3.1 Multiprocessing & Low-Latency Streaming Pipeline")
    add_p(
        "Video surveillance architectures frequently suffer from Python's Global Interpreter Lock (GIL) and frame serialization lag. "
        "Campus Sentinel solves this using a two-tier concurrency model:"
    )
    add_bullet("Each camera stream is spawned as an independent multiprocessing.Process (isolated memory space, separate core execution).", "Dedicated Worker Process: ")
    add_bullet("Inside each process, an Inference Thread continuously runs YOLOv8 tracking and pose analysis, while a Main Output Thread reads frames from cv2.VideoCapture at native camera frame-rate, paints bounding boxes, manages playback speed, and pushes frames to the queue.", "Decoupled IO & AI Loop: ")
    add_bullet("The inter-process queue between the worker and FastAPI has maxsize=1. If FastAPI is slow to serve a web client, the queue automatically discards stale frames, guaranteeing zero buffer accumulation and sub-second live feeds.", "Zero-Latency Frame Dropping: ")
    add_bullet("Frames are stored in a circular collections.deque(maxlen=900), retaining the preceding 30 to 60 seconds of video in memory without writing to disk continuously.", "Rolling Memory Ring Buffer: ")

    add_h2("3.2 Surveillance & Proctoring Modes")
    add_bullet(
        "Employs yolov8n.pt with ByteTrack (bytetrack.yaml). Continuously tallies room occupancy. "
        "Evaluates the bottom-center coordinates (feet) of each bounding box against custom polygon vertices via cv2.pointPolygonTest. "
        "If a person steps into a restricted area, a security alert is dispatched immediately.",
        "A. CLASSROOM Mode (Security & Space Management): "
    )
    add_bullet(
        "Employs yolov8n-pose.pt to track 17 anatomical keypoints per person simultaneously in a single forward pass. "
        "Implements a 3-tier biomechanical heuristic: (1) Severe Profile View (looking 90° sideways to a neighbor), "
        "(2) Facing Backwards (180° turn with hidden facial features but visible shoulders), and "
        "(3) Moderate Head Turn (yaw deviation calculated via nose-to-ear ratios exceeding ±45°).",
        "B. EXAM Mode (Automated Invigilation): "
    )
    add_bullet(
        "Transient head movements (e.g., glancing at a clock or rubbing eyes) do not trigger false alarms. "
        "Anomalies must persist for at least 30 consecutive frames (approx. 1.5–2.0 seconds). "
        "Normal postures decay the suspicion counter at double speed (-2/frame). "
        "Upon confirmation, the system automatically pulls the preceding 450 frames (15s) from memory, encodes an MP4 clip, snaps a high-res image, and notifies proctors.",
        "C. Temporal Persistence Buffer & Auto-Evidence Packager: "
    )

    # Section 4
    add_h1("4. Current Architectural & Algorithmic Limitations")
    add_p("To strategically determine where to take the system with AI, the current bottlenecks must be understood:")
    add_bullet("Current pose rules detect head rotation angles, but cannot detect micro-actions like looking down at concealed cheat sheets, smartphone reflections, whispering, or passing notes under desks.", "Rule-Based Geometrics vs Action Understanding: ")
    add_bullet("Server-Sent MJPEG multipart streams consume significant network bandwidth (~1.5–3 MB/s per camera stream), which does not scale gracefully to 50+ simultaneous campus feeds.", "MJPEG Streaming Bandwidth: ")
    add_bullet("SQLite is ideal for single-node deployments but lacks distributed horizontal scaling, connection pooling, and multi-tenant access control needed for campus-wide operations.", "Single-Node SQLite Concurrency: ")
    add_bullet("Track IDs are local to each camera feed. If a student leaves Camera 1 and enters Camera 2, the system assigns a new ID rather than maintaining a persistent identity.", "Absence of Cross-Camera Re-Identification (ReID): ")
    add_bullet("Surveillance is currently visual-only; audio signals (whispering, paper rustling, verbal collusion, distress calls) are ignored.", "No Audio Intelligence: ")

    # Section 5
    add_h1("5. The AI Strategic Roadmap: Where to Go Next")
    add_p(
        "This roadmap outlines high-impact AI, Computer Vision, and Generative AI pathways to evolve Campus Sentinel into a world-class autonomous campus intelligence platform."
    )

    add_h2("Phase 1: Advanced Action Recognition & Multimodal Malpractice Detection")
    add_bullet("Integrate Spatio-Temporal Graph Convolutional Networks (ST-GCN) or VideoMAE to classify sequential human motion patterns rather than relying on static frame-by-frame yaw angles. Accurately identifies passing objects, desk tampering, and concealed phone usage.", "Spatio-Temporal Action Recognition (ST-GCN / VideoMAE): ")
    add_bullet("Deploy lightweight gaze estimation models (e.g., L2CS-Net or ETH-XGaze) to estimate exact 3D gaze vectors from the eye region. Differentiates between looking down at the exam paper versus glancing at a neighbor's answer sheet.", "3D Gaze Vector Tracking: ")
    add_bullet("Leverage open-vocabulary object detectors (YOLO-World / Grounding DINO) to identify unauthorized objects (smartphones, smartwatches, concealed cheat slips, wired/wireless earphones) without retraining custom detectors.", "Zero-Shot Prohibited Object Detection: ")

    add_h2("Phase 2: Intelligent Audio Surveillance & Whispering Analysis")
    add_bullet("Deploy lightweight audio classification models (YAMNet / Whisper / Audio Spectrogram Transformer) on edge microphone streams to detect covert whispering, paper rustling, or commotion in exam halls.", "Acoustic Malpractice & Whispering Detection: ")
    add_bullet("Correlate audio amplitude spikes and classified speech events with visual head-turn clusters to produce dual-evidence incident confidence scores.", "Audio-Visual Sensor Fusion: ")

    add_h2("Phase 3: Cross-Camera Re-Identification (ReID) & Campus Safety")
    add_bullet("Implement deep feature embedding extractors (OSNet / FastReID) to track persons of interest across multiple cameras without relying on facial recognition, preserving student privacy while mapping movement trajectories.", "Cross-Camera Person Re-Identification: ")
    add_bullet("Provide optional opt-in facial recognition (InsightFace / ArcFace) at exam hall entry portals to verify student identities against university registration rosters and prevent exam impersonation.", "Biometric Identity Verification: ")

    add_h2("Phase 4: Scalable Edge AI & Streaming Infrastructure")
    add_bullet("Convert PyTorch/YOLO pipelines to NVIDIA TensorRT and ONNX Runtime with INT8/FP16 quantization. Enables 60+ FPS processing per GPU on Jetson or RTX servers.", "TensorRT & ONNX Quantization Acceleration: ")
    add_bullet("Migrate from MJPEG to WebRTC (via aiortc) or low-latency HLS, cutting video streaming bandwidth by over 80% while reducing browser CPU consumption.", "WebRTC / HLS Low-Bandwidth Streaming: ")
    add_bullet("Decouple inference into microservices using Redis / Celery / Kafka with PostgreSQL and MinIO S3 object storage for multi-camera campus deployments.", "Distributed Microservice Architecture: ")

    add_h2("Phase 5: Generative AI Incident Intelligence & Conversational NVR")
    add_bullet("Feed the 15-second incident clip and metadata to a Vision-Language Model (e.g., Qwen2-VL, Gemini Flash, or Claude 3.5 Sonnet) to generate formal, human-readable disciplinary reports with timestamped evidence breakdowns.", "Autonomous Disciplinary Report Writer: ")
    add_bullet("Enable security chiefs and proctors to query campus surveillance in natural language: 'Show me all instances where a student looked behind them for more than 5 seconds in Exam Hall 2 between 10 AM and 12 PM'.", "Conversational Natural Language Video Search (Video-RAG): ")

    # Section 6
    add_h1("6. Recommended Immediate Action Plan")
    add_p("To execute this roadmap efficiently, the recommended sequential steps are:")
    add_bullet("Swap MJPEG streaming in run.py with WebRTC for scalable multi-camera viewing.", "Step 1 (Infrastructure): ")
    add_bullet("Upgrade pose heuristics in core/inference.py with L2CS-Net 3D gaze estimation to reduce false positives.", "Step 2 (Computer Vision): ")
    add_bullet("Incorporate YOLO-World for real-time mobile phone and unauthorized device detection in exam halls.", "Step 3 (Object Detection): ")
    add_bullet("Connect an LLM pipeline to automatically generate disciplinary incident briefs when incidents are logged.", "Step 4 (Generative AI): ")

    doc.save(filename)
    print(f"Successfully generated DOCX: {filename}")


# ----------------------------------------------------
# ReportLab PDF Generator
# ----------------------------------------------------
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 750, "Campus Sentinel — Technical Architecture & AI Roadmap Report")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(54, 742, 558, 742)

        # Footer (all pages)
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(54, 45, 558, 45)
        self.drawString(54, 32, "Confidential — For Research, Architecture & AI Discovery")
        self.drawRightString(558, 32, f"Page {self._pageNumber} of {page_count}")
        self.restoreState()

def create_pdf(filename):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=colors.HexColor("#0F2043"),
        spaceAfter=4
    )

    sub_style = ParagraphStyle(
        'DocSub',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#0E7490"),
        spaceAfter=12
    )

    meta_style = ParagraphStyle(
        'DocMeta',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#64748B"),
        spaceAfter=16
    )

    h1_style = ParagraphStyle(
        'DocH1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#0F2043"),
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'DocH2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor("#0E7490"),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#334155"),
        spaceAfter=6
    )

    bullet_style = ParagraphStyle(
        'DocBullet',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12.5,
        textColor=colors.HexColor("#334155"),
        leftIndent=14,
        spaceAfter=4
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.white
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#334155")
    )

    story = []

    # Title & Header
    story.append(Paragraph("CAMPUS SENTINEL", title_style))
    story.append(Paragraph("Comprehensive Technical Report, Codebase Audit & Strategic AI Roadmap", sub_style))
    story.append(Paragraph("Repository: <b>d:\\CampusSentinal</b> &nbsp;|&nbsp; Enterprise NVR & AI Proctoring System &nbsp;|&nbsp; October 2026", meta_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0E7490"), spaceBefore=0, spaceAfter=14))

    # Section 1
    story.append(Paragraph("1. Executive Summary & System Overview", h1_style))
    story.append(Paragraph(
        "<b>Campus Sentinel</b> is an enterprise-oriented computer vision Network Video Recorder (NVR) and automated surveillance system specifically engineered for academic institutions. "
        "The system replaces conventional passive security monitors with active AI agents capable of continuous person tracking, real-time crowd occupancy counting, polygon-bounded restricted zone (ROI) intrusion detection, and automated exam proctoring with biomechanical malpractice detection.",
        body_style
    ))
    story.append(Paragraph(
        "By leveraging deep learning models (YOLOv8 & YOLOv8-Pose) synchronized with multi-process video ingest and asynchronous WebSocket telemetry, Campus Sentinel detects illicit behaviors in sub-second latency, logs evidence directly into an SQLite audit trail, and automatically packages synchronized 15-second MP4 evidence clips and high-resolution snapshots for review.",
        body_style
    ))

    # Section 2
    story.append(Paragraph("2. Full Codebase & File Inventory", h1_style))
    story.append(Paragraph("Structured breakdown of all files, models, configuration files, and persistence storage within the repository:", body_style))

    raw_table_data = [
        ("File / Component", "Type / Size", "Primary Functional Role"),
        ("run.py", "Python\n8.0 KB", "Master entry point; coordinates FastAPI, WebSocket broadcasting, MJPEG streaming, and multi-process camera workers."),
        ("core/inference.py", "Python\n16.7 KB", "Core CV engine; CameraPipeline with decoupled AI inference, ByteTrack, 17-keypoint pose heuristics, and ring buffers."),
        ("core/config_manager.py", "Python\n2.3 KB", "Configuration manager; reads/writes camera entries, operating modes (CLASSROOM vs EXAM), and ROI coordinates in JSON."),
        ("core/event_manager.py", "Python\n2.5 KB", "Audit logging service; initializes SQLite schema, records incidents, writes JPEG snapshots, and links video evidence clips."),
        ("ui/templates/index.html", "HTML/JS/CSS\n22.6 KB", "Cyber-dark glassmorphism enterprise dashboard with live multi-camera grid, scrubber, 1x/2x/4x speed controls, and incident modal."),
        ("storage/config.json", "JSON Config\n781 B", "Persistent camera registry holding live feeds, webcam indices, IP RTSP URLs, modes, and polygon boundary arrays."),
        ("storage/db/campus.sqlite3", "SQLite DB\n20.5 KB", "Relational database maintaining timestamped incident records, event types, camera IDs, snapshot paths, and clip paths."),
        ("storage/clips/", "Directory", "Repository for saved MP4 incident video recordings (both automated 15-second incident clips and manual 2-minute captures)."),
        ("storage/screenshots/", "Directory", "Repository for high-resolution JPEG snapshots captured at the exact moment an anomaly is detected."),
        ("storage/model school-Channel 50.MP4", "Test Video\n85.2 MB", "Pre-recorded CCTV classroom/exam monitoring video used for test benching and verification."),
        ("yolov8n.pt", "Model\n6.5 MB", "Ultralytics YOLOv8 Nano object detection weights; utilized in CLASSROOM mode for person detection & ROI intrusion."),
        ("yolov8s.pt", "Model\n22.6 MB", "Ultralytics YOLOv8 Small object detection weights; higher accuracy fallback for object detection."),
        ("yolov8n-pose.pt", "Model\n6.8 MB", "Ultralytics YOLOv8 Nano Pose weights; detects 17 skeletal landmarks per individual in EXAM mode."),
        ("yolov8s-pose.pt", "Model\n23.5 MB", "Ultralytics YOLOv8 Small Pose weights; enhanced keypoint localization for crowded or distant classrooms."),
        ("main.py", "Python\n3.7 KB", "Standalone OpenCV scale-testing prototype evaluating multi-student head-turn heuristics via BoT-SORT."),
        ("detailed skeletal.txt", "Text\n4.9 KB", "Early prototype design notes analyzing MediaPipe Pose, IoU collision for fight detection, and wrist extension reach."),
        ("readme.txt", "Text\n175 B", "Deployment prerequisite notes (pip install fastapi uvicorn websockets ultralytics).")
    ]

    table_cells = []
    for row_idx, r in enumerate(raw_table_data):
        if row_idx == 0:
            table_cells.append([
                Paragraph(r[0], table_header_style),
                Paragraph(r[1], table_header_style),
                Paragraph(r[2], table_header_style)
            ])
        else:
            table_cells.append([
                Paragraph(r[0], table_cell_style),
                Paragraph(r[1], table_cell_style),
                Paragraph(r[2], table_cell_style)
            ])

    pdf_table = Table(table_cells, colWidths=[120, 75, 309])
    pdf_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0F2043")),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
    ]))

    story.append(pdf_table)
    story.append(Spacer(1, 12))

    # Section 3
    story.append(Paragraph("3. Core Architecture & Workflow Analysis", h1_style))
    story.append(Paragraph("3.1 Multiprocessing & Low-Latency Streaming Pipeline", h2_style))
    story.append(Paragraph("Video surveillance architectures frequently suffer from Python's Global Interpreter Lock (GIL) and frame serialization lag. Campus Sentinel solves this using a two-tier concurrency model:", body_style))
    story.append(Paragraph("• <b>Dedicated Worker Process:</b> Each camera stream is spawned as an independent <code>multiprocessing.Process</code> (isolated memory space, separate core execution).", bullet_style))
    story.append(Paragraph("• <b>Decoupled IO & AI Loop:</b> Inside each process, an Inference Thread continuously runs YOLOv8 tracking and pose analysis, while a Main Output Thread reads frames from <code>cv2.VideoCapture</code> at native camera frame-rate, paints bounding boxes, manages playback speed, and pushes frames to the queue.", bullet_style))
    story.append(Paragraph("• <b>Zero-Latency Frame Dropping:</b> The inter-process queue between the worker and FastAPI has <code>maxsize=1</code>. If FastAPI is slow to serve a web client, the queue automatically discards stale frames, guaranteeing zero buffer accumulation.", bullet_style))
    story.append(Paragraph("• <b>Rolling Memory Ring Buffer:</b> Frames are stored in a circular <code>collections.deque(maxlen=900)</code>, retaining the preceding 30 to 60 seconds of video in memory without writing to disk continuously.", bullet_style))

    story.append(Paragraph("3.2 Surveillance & Proctoring Modes", h2_style))
    story.append(Paragraph("• <b>CLASSROOM Mode:</b> Employs <code>yolov8n.pt</code> with ByteTrack (<code>bytetrack.yaml</code>). Evaluates the bottom-center coordinates (feet) of each bounding box against custom polygon vertices via <code>cv2.pointPolygonTest</code>. If a person enters a restricted zone, an alert is triggered immediately.", bullet_style))
    story.append(Paragraph("• <b>EXAM Mode:</b> Employs <code>yolov8n-pose.pt</code> to track 17 anatomical keypoints per person simultaneously in a single forward pass. Implements a 3-tier biomechanical heuristic: (1) Severe Profile View (looking 90° sideways to a neighbor), (2) Facing Backwards (180° turn with hidden facial features but visible shoulders), and (3) Moderate Head Turn (yaw deviation calculated via nose-to-ear ratios exceeding ±45°).", bullet_style))
    story.append(Paragraph("• <b>Temporal Persistence Buffer & Auto-Evidence Packager:</b> Anomalies must persist for at least 30 consecutive frames (~1.5–2.0s). Normal postures decay the suspicion counter at double speed (-2/frame). Upon confirmation, the system automatically pulls the preceding 450 frames (15s) from memory, encodes an MP4 clip, snaps a high-res image, and notifies proctors over WebSockets.", bullet_style))

    # Section 4
    story.append(Paragraph("4. Current Architectural & Algorithmic Limitations", h1_style))
    story.append(Paragraph("• <b>Rule-Based Geometrics vs Action Understanding:</b> Current pose rules detect head rotation angles, but cannot detect micro-actions like looking down at concealed cheat sheets, smartphone reflections, whispering, or passing notes under desks.", bullet_style))
    story.append(Paragraph("• <b>MJPEG Streaming Bandwidth:</b> Server-Sent MJPEG multipart streams consume significant network bandwidth (~1.5–3 MB/s per camera stream), which does not scale gracefully to 50+ simultaneous campus feeds.", bullet_style))
    story.append(Paragraph("• <b>Single-Node SQLite Concurrency:</b> SQLite is ideal for single-node deployments but lacks distributed horizontal scaling, connection pooling, and multi-tenant access control needed for campus-wide operations.", bullet_style))
    story.append(Paragraph("• <b>Absence of Cross-Camera Re-Identification (ReID):</b> Track IDs are local to each camera feed. When a student transitions between rooms, identity persistence is lost.", bullet_style))
    story.append(Paragraph("• <b>No Audio Intelligence:</b> Surveillance is currently visual-only; acoustic indicators like murmuring, paper rustling, and verbal collusion are unmonitored.", bullet_style))

    # Section 5
    story.append(Paragraph("5. Strategic AI Roadmap: The Part to Go Next Using AI", h1_style))
    story.append(Paragraph("This roadmap details high-impact AI, Computer Vision, and Generative AI pathways to evolve Campus Sentinel into a state-of-the-art autonomous campus intelligence platform.", body_style))

    story.append(Paragraph("Phase 1: Advanced Action Recognition & Multimodal Malpractice Detection", h2_style))
    story.append(Paragraph("• <b>Spatio-Temporal Action Recognition (ST-GCN / VideoMAE):</b> Integrate Graph Convolutional Networks or VideoMAE to classify sequential human motion patterns rather than relying on static frame yaw angles. Accurately identifies passing objects, desk tampering, and concealed phone usage.", bullet_style))
    story.append(Paragraph("• <b>3D Gaze Vector Tracking (L2CS-Net / ETH-XGaze):</b> Deploy lightweight gaze estimation models to estimate exact 3D gaze vectors from the eye region. Differentiates between looking down at the exam paper versus glancing at a neighbor's answer sheet.", bullet_style))
    story.append(Paragraph("• <b>Zero-Shot Prohibited Object Detection (YOLO-World):</b> Leverage open-vocabulary object detectors to identify unauthorized items (smartphones, smartwatches, cheat notes, ear-pieces) without retraining custom detectors.", bullet_style))

    story.append(Paragraph("Phase 2: Intelligent Audio Surveillance & Whispering Analysis", h2_style))
    story.append(Paragraph("• <b>Acoustic Malpractice & Whispering Detection:</b> Deploy lightweight audio classification models (YAMNet / Whisper / Audio Spectrogram Transformer) on edge microphone streams to detect covert whispering, paper rustling, or commotion in exam halls.", bullet_style))
    story.append(Paragraph("• <b>Audio-Visual Sensor Fusion:</b> Correlate audio amplitude spikes and classified speech events with visual head-turn clusters to produce dual-evidence incident confidence scores.", bullet_style))

    story.append(Paragraph("Phase 3: Cross-Camera Re-Identification (ReID) & Campus Safety", h2_style))
    story.append(Paragraph("• <b>Cross-Camera Person Re-Identification (FastReID / OSNet):</b> Implement deep feature embedding extractors to track persons of interest across multiple cameras without relying on facial recognition, preserving student privacy while mapping movement trajectories.", bullet_style))
    story.append(Paragraph("• <b>Biometric Identity Verification:</b> Provide optional opt-in facial recognition (InsightFace / ArcFace) at exam hall entry portals to verify student identities against university registration rosters and prevent exam impersonation.", bullet_style))

    story.append(Paragraph("Phase 4: Scalable Edge AI & Streaming Infrastructure", h2_style))
    story.append(Paragraph("• <b>TensorRT & ONNX Quantization Acceleration:</b> Convert PyTorch/YOLO pipelines to NVIDIA TensorRT and ONNX Runtime with INT8/FP16 quantization. Enables 60+ FPS processing per GPU on Jetson or RTX servers.", bullet_style))
    story.append(Paragraph("• <b>WebRTC / HLS Low-Bandwidth Streaming:</b> Migrate from MJPEG to WebRTC (via aiortc) or low-latency HLS, cutting video streaming bandwidth by over 80% while reducing browser CPU consumption.", bullet_style))
    story.append(Paragraph("• <b>Distributed Microservice Architecture:</b> Decouple inference into microservices using Redis / Celery / Kafka with PostgreSQL and MinIO S3 object storage for multi-camera campus deployments.", bullet_style))

    story.append(Paragraph("Phase 5: Generative AI Incident Intelligence & Conversational NVR", h2_style))
    story.append(Paragraph("• <b>Autonomous Disciplinary Report Writer:</b> Feed the 15-second incident clip and metadata to a Vision-Language Model (e.g., Qwen2-VL, Gemini Flash, or Claude 3.5 Sonnet) to generate formal, human-readable disciplinary reports with timestamped evidence breakdowns.", bullet_style))
    story.append(Paragraph("• <b>Conversational Natural Language Video Search (Video-RAG):</b> Enable security chiefs and proctors to query campus surveillance in natural language: <i>'Show me all instances where a student looked behind them for more than 5 seconds in Exam Hall 2 between 10 AM and 12 PM'</i>.", bullet_style))

    # Section 6
    story.append(Paragraph("6. Recommended Immediate Action Plan", h1_style))
    story.append(Paragraph("• <b>Step 1 (Infrastructure):</b> Swap MJPEG streaming in <code>run.py</code> with WebRTC for scalable multi-camera viewing.", bullet_style))
    story.append(Paragraph("• <b>Step 2 (Computer Vision):</b> Upgrade pose heuristics in <code>core/inference.py</code> with L2CS-Net 3D gaze estimation to reduce false positives.", bullet_style))
    story.append(Paragraph("• <b>Step 3 (Object Detection):</b> Incorporate YOLO-World for real-time mobile phone and unauthorized device detection in exam halls.", bullet_style))
    story.append(Paragraph("• <b>Step 4 (Generative AI):</b> Connect an LLM pipeline to automatically generate disciplinary incident briefs when incidents are logged.", bullet_style))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated PDF: {filename}")

if __name__ == "__main__":
    out_dir = r"d:\CampusSentinal"
    docx_path = os.path.join(out_dir, "Campus_Sentinel_Detailed_Report_and_AI_Roadmap.docx")
    pdf_path = os.path.join(out_dir, "Campus_Sentinel_Detailed_Report_and_AI_Roadmap.pdf")
    create_docx(docx_path)
    create_pdf(pdf_path)
