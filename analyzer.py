import cv2
import numpy as np
from ultralytics import YOLO
import time
import math
import threading
from datetime import datetime

# ── module-level lock: ensures scan & switch never run at the same time ────────
_hw_lock = threading.Lock()

# ── cached scan result ─────────────────────────────────────────────────────────
_scan_cache       = None   # list[int] or None
_scan_cache_time  = 0
_SCAN_CACHE_TTL   = 30     # seconds before re-scanning

def scan_cameras_safe(max_ports: int = 4) -> list[int]:
    """
    Scan for available camera ports.
    Uses the module-level _hw_lock so it never runs while a camera is being
    switched, and vice-versa.  Results are cached for _SCAN_CACHE_TTL seconds.
    """
    global _scan_cache, _scan_cache_time
    now = time.time()
    if _scan_cache is not None and (now - _scan_cache_time) < _SCAN_CACHE_TTL:
        return _scan_cache

    found = []
    with _hw_lock:
        for i in range(max_ports):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ok, _ = cap.read()
                    if ok:
                        found.append(i)
                cap.release()
            except Exception:
                pass
    _scan_cache      = found
    _scan_cache_time = time.time()
    print(f"[Camera scan] found ports: {found}")
    return found


class ClassroomAnalyzer:
    def __init__(self):
        print("Loading AI models...")
        self.yolo_obj  = YOLO("yolov8n.pt")
        self.yolo_pose = YOLO("yolov8n-pose.pt")
        print("Models loaded.")

        # Camera state — protected by _hw_lock (shared with scan_cameras_safe)
        self.camera_port   = None
        self.cap           = None

        # Frame generation control
        self._frame_errors = 0

        # Object detection cache
        self.obj_detect_every = 5
        self.frame_count      = 0
        self.cached_phones    = []
        self.cached_laptops   = []

        # Session
        self.session_start = time.time()
        self.is_recording  = True

        # Tracking
        self.student_history = {}
        self.absence_counts  = {}
        self.leave_threshold = 10.0

        # Report
        self.report_data   = []
        self.last_snapshot = 0

        self.stats = self._empty_stats()

    # ── helpers ────────────────────────────────────────────────────────────────

    def _empty_stats(self):
        return {
            "total": 0, "attentive": 0, "phone": 0,
            "computer": 0, "talking": 0, "looking_away": 0,
            "leaving": 0, "attention_score": 100, "duration": "00:00:00",
            "is_recording": True, "camera_port": self.camera_port
        }

    def _dist(self, p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    def _overlaps(self, box, items):
        x1, y1, x2, y2 = box
        return any(ix1 < x2 and ix2 > x1 and iy1 < y2 and iy2 > y1
                   for (ix1, iy1, ix2, iy2) in items)

    def get_duration(self):
        d = int(time.time() - self.session_start)
        return f"{d//3600:02d}:{(d%3600)//60:02d}:{d%60:02d}"

    # ── public camera API ──────────────────────────────────────────────────────

    def set_camera(self, port: int) -> dict:
        """
        Synchronously switch to the given camera port.
        Acquires _hw_lock, so it cannot run at the same time as a scan.
        Returns {"ok": bool, "port": int, "msg": str}
        """
        global _scan_cache  # invalidate cache on switch
        with _hw_lock:
            try:
                new_cap = cv2.VideoCapture(port)
                if not new_cap.isOpened():
                    new_cap.release()
                    msg = f"Port {port} ไม่พบกล้อง (ไม่สามารถเปิดได้)"
                    print(f"[Camera] {msg}")
                    return {"ok": False, "port": port, "msg": msg}

                ok, _ = new_cap.read()
                if not ok:
                    new_cap.release()
                    msg = f"Port {port} เปิดได้แต่อ่านภาพไม่ได้ (กล้องอาจถูกใช้งานอยู่)"
                    print(f"[Camera] {msg}")
                    return {"ok": False, "port": port, "msg": msg}

                new_cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
                new_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

                # Release old camera before assigning new one
                if self.cap is not None:
                    self.cap.release()

                self.cap           = new_cap
                self.camera_port   = port
                self._frame_errors = 0
                self.student_history.clear()
                _scan_cache = None   # force re-scan next time
                msg = f"เปิดกล้อง Port {port} สำเร็จ"
                print(f"[Camera] {msg}")
                return {"ok": True, "port": port, "msg": msg}

            except Exception as e:
                msg = f"เกิดข้อผิดพลาด: {e}"
                print(f"[Camera] {msg}")
                return {"ok": False, "port": port, "msg": msg}

    # ── session API ────────────────────────────────────────────────────────────

    def reset_session(self):
        self.session_start = time.time()
        self.report_data   = []
        self.student_history.clear()
        self.absence_counts.clear()
        self.last_snapshot = 0
        self.stats         = self._empty_stats()

    def toggle_recording(self):
        self.is_recording = not self.is_recording
        return self.is_recording

    # ── placeholder frame ──────────────────────────────────────────────────────

    def _placeholder(self, lines: list[str]) -> np.ndarray:
        img = np.zeros((480, 640, 3), np.uint8)
        y = 190
        for line in lines:
            cv2.putText(img, line, (60, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (70, 140, 255), 1, cv2.LINE_AA)
            y += 38
        return img

    # ── main processing ────────────────────────────────────────────────────────

    def _analyze_frame(self, frame):
        self._frame_errors = 0
        self.frame_count  += 1
        now = time.time()

        # ── Object detection (phone / laptop) every N frames ──────────────────
        if self.frame_count % self.obj_detect_every == 0:
            self.cached_phones  = []
            self.cached_laptops = []
            obj_res = self.yolo_obj(frame, classes=[63, 67], verbose=False, imgsz=320)
            for r in obj_res:
                for box in r.boxes:
                    cls, conf = int(box.cls[0]), float(box.conf[0])
                    if conf > 0.4:
                        x1,y1,x2,y2 = [int(v) for v in box.xyxy[0]]
                        if cls == 67: self.cached_phones.append((x1,y1,x2,y2))
                        elif cls == 63: self.cached_laptops.append((x1,y1,x2,y2))

        for (x1,y1,x2,y2) in self.cached_phones:
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,165,255),1)
            cv2.putText(frame,"Phone",(x1,y1-4),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,165,255),1)
        for (x1,y1,x2,y2) in self.cached_laptops:
            cv2.rectangle(frame,(x1,y1),(x2,y2),(255,255,0),1)
            cv2.putText(frame,"Laptop",(x1,y1-4),cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,255,0),1)

        # ── Pose + Tracking ───────────────────────────────────────────────────
        pose_res = self.yolo_pose.track(frame, persist=True, classes=[0],
                                        verbose=False, imgsz=320)
        counts     = {"attentive":0,"phone":0,"computer":0,"talking":0,"looking_away":0}
        active_ids = []
        people     = []

        if (pose_res and pose_res[0].boxes is not None
                and pose_res[0].boxes.id is not None):
            boxes     = pose_res[0].boxes.xyxy.cpu().numpy()
            track_ids = pose_res[0].boxes.id.cpu().numpy().astype(int)
            keypoints = pose_res[0].keypoints.xy.cpu().numpy()

            for box, tid, kpts in zip(boxes, track_ids, keypoints):
                x1,y1,x2,y2 = map(int, box)
                active_ids.append(tid)
                self.student_history[tid] = now
                if tid not in self.absence_counts:
                    self.absence_counts[tid] = 0

                has_phone  = self._overlaps((x1,y1,x2,y2), self.cached_phones)
                has_laptop = self._overlaps((x1,y1,x2,y2), self.cached_laptops)

                nose, l_eye, r_eye = kpts[0], kpts[1], kpts[2]
                looking_away = False
                if all(p[0] != 0 for p in [nose, l_eye, r_eye]):
                    dl = self._dist(nose, l_eye)
                    dr = self._dist(nose, r_eye)
                    ratio = dl / dr if dr > 0 else 99
                    if ratio > 2.2 or ratio < 0.45:
                        looking_away = True

                if has_phone:        status, color = "Phone",        (0,165,255)
                elif has_laptop:     status, color = "Computer",     (255,255,0)
                elif looking_away:   status, color = "Looking Away", (0,0,255)
                else:                status, color = "Attentive",    (0,220,80)

                people.append({"id":tid,"box":(x1,y1,x2,y2),
                                "nose":nose,"status":status,"color":color})

            # Talking check
            for i in range(len(people)):
                for j in range(i+1, len(people)):
                    n1,n2 = people[i]["nose"], people[j]["nose"]
                    b1,b2 = people[i]["box"],  people[j]["box"]
                    if n1[0] and n2[0]:
                        d = self._dist(n1, n2)
                        avg_w = ((b1[2]-b1[0])+(b2[2]-b2[0]))/2
                        if d < avg_w * 0.8:
                            people[i]["status"] = people[j]["status"] = "Talking"
                            people[i]["color"]  = people[j]["color"]  = (255,50,255)

            STATUS_KEY = {"Attentive":"attentive","Phone":"phone","Computer":"computer",
                          "Looking Away":"looking_away","Talking":"talking"}
            for p in people:
                x1,y1,x2,y2 = p["box"]
                s,c,tid = p["status"], p["color"], p["id"]
                counts[STATUS_KEY.get(s,"attentive")] += 1
                overlay = frame.copy()
                cv2.rectangle(overlay,(x1,y1-28),(x1+len(f"ID:{tid} {s}")*9,y1),(0,0,0),-1)
                cv2.addWeighted(overlay,0.5,frame,0.5,0,frame)
                cv2.rectangle(frame,(x1,y1),(x2,y2),c,2)
                cv2.putText(frame,f"ID:{tid} {s}",(x1+2,y1-8),
                            cv2.FONT_HERSHEY_SIMPLEX,0.55,c,1,cv2.LINE_AA)

        # Leave detection
        for sid, last in list(self.student_history.items()):
            if sid not in active_ids and (now - last) > self.leave_threshold:
                self.absence_counts[sid] += 1
                del self.student_history[sid]

        total = len(active_ids)
        score = int(((counts["attentive"]+counts["computer"])/total)*100) if total else 100
        dur   = self.get_duration()

        self.stats = {
            "total": total, "attentive": counts["attentive"],
            "phone": counts["phone"], "computer": counts["computer"],
            "talking": counts["talking"], "looking_away": counts["looking_away"],
            "leaving": sum(self.absence_counts.values()),
            "attention_score": score, "duration": dur,
            "is_recording": self.is_recording,
            "camera_port": self.camera_port
        }

        if self.is_recording and (now - self.last_snapshot) >= 5:
            row = self.stats.copy()
            row["time"] = datetime.now().strftime("%H:%M:%S")
            self.report_data.append(row)
            self.last_snapshot = now

        score_color = (0,220,80) if score>=70 else (0,165,255) if score>=40 else (0,50,255)
        cv2.putText(frame,f"Score: {score}%",(10,32),
                    cv2.FONT_HERSHEY_SIMPLEX,1.0,score_color,2,cv2.LINE_AA)
        cv2.putText(frame,dur,(10,62),
                    cv2.FONT_HERSHEY_SIMPLEX,0.65,(200,200,200),1,cv2.LINE_AA)
        cv2.circle(frame,(620,20),8,(0,220,80) if self.is_recording else (80,80,80),-1)

        return frame, self.stats

    def process_frame(self):
        # No camera yet → show instruction screen (sleep so we don't spin)
        if self.cap is None or not self.cap.isOpened():
            time.sleep(0.1)
            return self._placeholder([
                "ยังไม่ได้เลือกกล้อง",
                "",
                "กด  'สแกนกล้อง'  แล้วกด  'สลับกล้อง'",
                "ในแผงควบคุมด้านบน",
            ]), self.stats

        # Read a frame (not under _hw_lock — read is lightweight & frequent)
        ok, frame = self.cap.read()

        if not ok or frame is None:
            self._frame_errors += 1
            time.sleep(0.1)
            return self._placeholder([
                f"กล้อง Port {self.camera_port} ไม่ตอบสนอง",
                "",
                "กรุณาตรวจสอบการเชื่อมต่อ",
                "หรือเลือกกล้องใหม่",
            ]), self.stats

        return self._analyze_frame(frame)

    def process_custom_frame(self, frame):
        if frame is None:
            return self._placeholder([
                "ไม่พบภาพเฟรมจากเว็บแคม",
                "กรุณาแชร์สิทธิ์กล้องในเบราว์เซอร์",
            ]), self.stats
        return self._analyze_frame(frame)

    def release(self):
        if self.cap:
            self.cap.release()
