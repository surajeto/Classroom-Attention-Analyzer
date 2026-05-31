from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn, cv2, os, smtplib
import pandas as pd
from pydantic import BaseModel
from email.message import EmailMessage
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors as rl_colors
from analyzer import ClassroomAnalyzer, scan_cameras_safe

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

analyzer = ClassroomAnalyzer()

# ── Request models ─────────────────────────────────────────────────────────────

class CameraRequest(BaseModel):
    port: int

class EmailRequest(BaseModel):
    to_email:   str
    smtp_email: str
    smtp_pass:  str

class FrameData(BaseModel):
    image: str

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

# ── Video stream ──────────────────────────────────────────────────────────────

def gen_frames():
    while True:
        frame, _ = analyzer.process_frame()
        ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            continue
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/process_frame")
async def process_frame_api(data: FrameData):
    import base64
    import numpy as np
    import asyncio
    try:
        header, encoded = data.image.split(",", 1) if "," in data.image else ("", data.image)
        img_data = base64.b64decode(encoded)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"ok": False, "msg": "Cannot decode image"}
        loop = asyncio.get_event_loop()
        processed_frame, stats = await loop.run_in_executor(None, analyzer.process_custom_frame, frame)
        ret, buf = cv2.imencode(".jpg", processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            return {"ok": False, "msg": "Cannot encode processed frame"}
        processed_base64 = base64.b64encode(buf).decode('utf-8')
        return {
            "ok": True,
            "image": f"data:image/jpeg;base64,{processed_base64}",
            "stats": stats
        }
    except Exception as e:
        return {"ok": False, "msg": f"Error: {e}"}

# ── Stats & controls ──────────────────────────────────────────────────────────

@app.get("/stats")
async def stats():
    return analyzer.stats

@app.get("/cameras")
async def list_cameras():
    """Run in a thread so it doesn't block the event loop."""
    import asyncio
    loop = asyncio.get_event_loop()
    ports = await loop.run_in_executor(None, scan_cameras_safe)
    return {"ports": ports}

@app.post("/set_camera")
async def set_camera(req: CameraRequest):
    import asyncio
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, analyzer.set_camera, req.port)
    return result

@app.post("/reset")
async def reset():
    analyzer.reset_session()
    return {"ok": True}

@app.post("/toggle_recording")
async def toggle_rec():
    state = analyzer.toggle_recording()
    return {"recording": state}

# ── Export ────────────────────────────────────────────────────────────────────

@app.get("/export/excel")
async def export_excel():
    data = analyzer.report_data
    if not data:
        return {"error": "ยังไม่มีข้อมูล (รอให้ระบบบันทึกข้อมูลก่อน 5 วินาที)"}

    df = pd.DataFrame(data)
    # Rename columns to Thai
    rename = {
        "time":"เวลา","total":"จำนวนคน","attentive":"ตั้งใจ",
        "phone":"เล่นมือถือ","computer":"ดูคอม","talking":"คุยกัน",
        "looking_away":"หันไปทางอื่น","leaving":"ลุกออก","attention_score":"คะแนน %",
        "duration":"เวลาที่บันทึก"
    }
    df = df.rename(columns=rename)
    df = df[[c for c in rename.values() if c in df.columns]]

    path = "classroom_report.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="รายงานห้องเรียน")
        ws = writer.sheets["รายงานห้องเรียน"]
        for col in ws.columns:
            ws.column_dimensions[col[0].column_letter].width = 18

    return FileResponse(path, filename="classroom_report.xlsx",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

@app.get("/export/pdf")
async def export_pdf():
    data = analyzer.report_data
    if not data:
        return {"error": "ยังไม่มีข้อมูล"}

    path = "classroom_report.pdf"
    c = rl_canvas.Canvas(path, pagesize=A4)
    W, H = A4

    # Header
    c.setFillColor(rl_colors.HexColor("#1e3a5f"))
    c.rect(0, H-70, W, 70, fill=1, stroke=0)
    c.setFillColor(rl_colors.white)
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(W/2, H-42, "Classroom Attention Report")
    c.setFont("Helvetica", 10)
    from datetime import datetime
    c.drawCentredString(W/2, H-58, datetime.now().strftime("Generated: %d/%m/%Y %H:%M"))

    # Summary block
    last = data[-1]
    summary_items = [
        ("Total Session Time", last.get("duration","—")),
        ("Final Attention Score", f"{last.get('attention_score',0)}%"),
        ("Total Snapshots Recorded", str(len(data))),
        ("Times Someone Left Class", str(last.get("leaving",0))),
    ]
    c.setFillColor(rl_colors.HexColor("#f0f4f8"))
    c.rect(40, H-170, W-80, 85, fill=1, stroke=0)
    c.setFillColor(rl_colors.HexColor("#1e3a5f"))
    c.setFont("Helvetica-Bold", 11)
    c.drawString(50, H-100, "Session Summary")
    c.setFont("Helvetica", 10)
    c.setFillColor(rl_colors.black)
    for i, (label, val) in enumerate(summary_items):
        col = 50 if i < 2 else 300
        row = H-118 if i % 2 == 0 else H-136
        c.drawString(col, row, f"{label}: {val}")

    # Table header
    y = H-195
    headers = ["Time","Attentive","Phone","Computer","Talking","Looking Away","Score%"]
    col_x   = [50, 120, 195, 270, 345, 420, 510]
    c.setFillColor(rl_colors.HexColor("#1e3a5f"))
    c.rect(40, y-5, W-80, 20, fill=1, stroke=0)
    c.setFillColor(rl_colors.white)
    c.setFont("Helvetica-Bold", 9)
    for h, x in zip(headers, col_x):
        c.drawString(x, y+2, h)

    # Table rows
    y -= 20
    c.setFont("Helvetica", 9)
    for i, row in enumerate(data[-40:]):   # max 40 rows
        bg = rl_colors.HexColor("#f7fafc") if i % 2 == 0 else rl_colors.white
        c.setFillColor(bg)
        c.rect(40, y-4, W-80, 16, fill=1, stroke=0)
        c.setFillColor(rl_colors.black)
        vals = [row.get("time",""), row.get("attentive",0), row.get("phone",0),
                row.get("computer",0), row.get("talking",0),
                row.get("looking_away",0), f"{row.get('attention_score',0)}%"]
        for v, x in zip(vals, col_x):
            c.drawString(x, y, str(v))
        y -= 16
        if y < 60:
            c.showPage()
            y = H - 60

    c.save()
    return FileResponse(path, filename="classroom_report.pdf", media_type="application/pdf")

# ── Email ─────────────────────────────────────────────────────────────────────

@app.post("/send_email")
async def send_email(req: EmailRequest):
    if not req.smtp_email or not req.smtp_pass:
        return {"ok": False, "msg": "กรุณากรอก Email และ App Password ให้ครบก่อน"}
    try:
        msg = EmailMessage()
        stats = analyzer.stats
        msg.set_content(
            f"รายงานผลการเรียนในห้องเรียน\n\n"
            f"ระยะเวลา: {stats['duration']}\n"
            f"คะแนนความตั้งใจ: {stats['attention_score']}%\n"
            f"จำนวนนักศึกษา: {stats['total']} คน\n"
            f"ตั้งใจฟัง: {stats['attentive']} คน\n"
            f"เล่นมือถือ: {stats['phone']} คน\n"
            f"คุยกัน: {stats['talking']} คน\n"
            f"หันไปทางอื่น: {stats['looking_away']} คน\n"
            f"ลุกออกจากห้อง: {stats['leaving']} ครั้ง\n"
        )
        msg["Subject"] = f"รายงานห้องเรียน — Score {stats['attention_score']}%"
        msg["From"]    = req.smtp_email
        msg["To"]      = req.to_email
        with smtplib.SMTP("smtp.gmail.com", 587) as s:
            s.starttls()
            s.login(req.smtp_email, req.smtp_pass)
            s.send_message(msg)
        return {"ok": True, "msg": f"ส่งอีเมลไปที่ {req.to_email} เรียบร้อยแล้ว"}
    except Exception as e:
        return {"ok": False, "msg": str(e)}

# ── Start ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("╔══════════════════════════════════════╗")
    print("║  Classroom Attention Analyzer  v2.0  ║")
    print("║  http://localhost:8000               ║")
    print("╚══════════════════════════════════════╝")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
