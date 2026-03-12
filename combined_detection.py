import cv2
import math
import time
from ultralytics import YOLO

# ---------- CONFIG ----------
VIDEO_PATH = "sample_video.mp4"
VEHICLE_MODEL_PATH = "yolov8n.pt"
AMBULANCE_MODEL_PATH = "runs/detect/train/weights/best.pt"
OUTPUT_PATH = "output_counting_fixed.mp4"

LINE_POSITION_RATIO = 0.55   # counting line as fraction of frame height (0..1)
DIST_THRESHOLD = 60         # for associating ambulance detections to tracks (pixels)
STALE_FRAMES = 10           # remove ambulance tracks not seen for this many streamed frames
# ----------------------------

vehicle_model = YOLO(VEHICLE_MODEL_PATH)
ambulance_model = YOLO(AMBULANCE_MODEL_PATH)

# Prepare video writer using sample frame to get size/fps
cap_tmp = cv2.VideoCapture(VIDEO_PATH)
if not cap_tmp.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")
fps = cap_tmp.get(cv2.CAP_PROP_FPS) or 25.0
w = int(cap_tmp.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap_tmp.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap_tmp.release()

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (w, h))

line_y = int(h * LINE_POSITION_RATIO)

# Vehicle counting bookkeeping (uses tracker IDs from YOLO.track)
vehicle_prev_y = {}    # track_id -> previous center_y
vehicle_counted_ids = set()
vehicle_count = 0

# Ambulance centroid-tracker bookkeeping
# each track: {id:int, cx:int, cy:int, prev_y:int, last_seen:int, counted:bool}
ambulance_tracks = []
next_amb_id = 0
ambulance_count = 0
frame_idx = 0

start_time = time.time()

# Use vehicle_model.track streaming (keeps consistent IDs)
for result in vehicle_model.track(source=VIDEO_PATH, persist=True, stream=True, verbose=False):
    frame_idx += 1
    frame = result.orig_img.copy()

    # ---------- VEHICLE (tracked by YOLO) ----------
    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            # each box has xyxy, cls, id (when track used)
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy)
            cls = int(box.cls[0].cpu().numpy())
            # In ultralytics result, .id may be present when using track
            try:
                track_id = int(box.id[0].cpu().numpy())
            except Exception:
                track_id = None

            label = result.names[cls] if hasattr(result, "names") else str(cls)
            if label in ["car", "truck", "bus", "motorbike"]:
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                # draw
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label}-{track_id}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                # maintain prev y for this track_id
                if track_id is not None:
                    if track_id in vehicle_prev_y:
                        prev_y = vehicle_prev_y[track_id]
                        # count when object crosses line downward (prev_y < line_y and cy >= line_y)
                        if prev_y < line_y <= cy and track_id not in vehicle_counted_ids:
                            vehicle_count += 1
                            vehicle_counted_ids.add(track_id)
                    # update prev y
                    vehicle_prev_y[track_id] = cy

    # ---------- AMBULANCE (custom model) ----------
    # Run detection on this streamed frame (we didn't stream ambulance_model)
    amb_res = ambulance_model.predict(frame, verbose=False)
    new_detections = []
    if len(amb_res) > 0 and len(amb_res[0].boxes) > 0:
        for box in amb_res[0].boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            # optional: check class name if your custom model outputs multiple classes
            new_detections.append((cx, cy, x1, y1, x2, y2))

    # Associate detections with existing ambulance_tracks by distance
    used_dets = set()
    for track in ambulance_tracks:
        track['seen_this_frame'] = False

    for di, det in enumerate(new_detections):
        cx, cy, x1, y1, x2, y2 = det
        # find closest existing track
        best = None
        best_dist = None
        for track in ambulance_tracks:
            dist = math.hypot(cx - track['cx'], cy - track['cy'])
            if best is None or dist < best_dist:
                best = track
                best_dist = dist
        if best is not None and best_dist is not None and best_dist < DIST_THRESHOLD:
            # update track
            best['prev_y'] = best['cy']
            best['cx'] = cx
            best['cy'] = cy
            best['last_seen'] = frame_idx
            best['seen_this_frame'] = True
            used_dets.add(di)
        else:
            # create new track
            ambulance_tracks.append({
                'id': next_amb_id,
                'cx': cx, 'cy': cy,
                'prev_y': cy,
                'last_seen': frame_idx,
                'counted': False,
                'seen_this_frame': True
            })
            next_amb_id += 1
            used_dets.add(di)

    # Remove stale tracks (not seen for STALE_FRAMES)
    ambulance_tracks = [t for t in ambulance_tracks if frame_idx - t['last_seen'] <= STALE_FRAMES]

    # Draw ambulance boxes and check counts
    for det_idx, det in enumerate(new_detections):
        cx, cy, x1, y1, x2, y2 = det
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # After association, check each ambulance track for crossing the line
    for track in ambulance_tracks:
        tid = track['id']
        cx, cy = track['cx'], track['cy']
        prev_y = track.get('prev_y', cy)
        # When object moves downward across the line:
        if prev_y < line_y <= cy and not track.get('counted', False):
            ambulance_count += 1
            track['counted'] = True
        # draw id and marker
        cv2.putText(frame, f"amb-{tid}", (int(cx)-20, int(cy)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        cv2.circle(frame, (int(cx), int(cy)), 4, (0,0,255), -1)

    # ---------- DRAW line & stats ----------
    cv2.line(frame, (0, line_y), (w, line_y), (255,255,0), 2)
    cv2.putText(frame, f"Vehicles: {vehicle_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    cv2.putText(frame, f"Ambulances: {ambulance_count}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

    # write & show
    out.write(frame)
    cv2.imshow("Vehicle & Ambulance Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# cleanup
out.release()
cv2.destroyAllWindows()
print(f"Done. Vehicles={vehicle_count} Ambulances={ambulance_count}")
print("Saved:", OUTPUT_PATH)
