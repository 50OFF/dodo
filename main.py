import cv2
from ultralytics import YOLO
import pandas as pd
import argparse

# CLI аргументы
parser = argparse.ArgumentParser(description="Table occupancy detection")
parser.add_argument("--video", type=str, required=True, help="Path to input video")
args = parser.parse_args()

video_path = args.video
output_path = "output.mp4"

# параметры
model_path = "yolov8s.pt"
occupy_delay = 10.0
empty_delay = 2.0

# модель
model = YOLO(model_path)
# model.to("cuda") # Я запускал на видеокарте для ускорения обработки

# видео
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Ошибка открытия видео")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# ROI
ret, frame = cap.read()
if not ret:
    print("Ошибка чтения первого кадра")
    exit()

roi = cv2.selectROI("Выберите стол", frame, False, False)
cv2.destroyWindow("Выберите стол")
x, y, w, h = map(int, roi)

# состояния
state = "empty"
state_start_time = 0
last_seen_time = 0
last_leave_time = None
frame_id = 0

events = []

# цвета для состояний
colors = {
    "empty": (0, 255, 0),
    "approached": (0, 255, 255),
    "occupied": (0, 0, 255)
}

# функция для записи смены состояний
def log_event(ts, state):
    events.append({"timestamp": ts, "event": state})

log_event(0, state)

# основной цикл
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_timestamp = frame_id / fps

    # YOLO
    results = model(frame)[0]
    people_boxes = []

    for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
        if int(cls) == 0:
            x1, y1, x2, y2 = map(int, box)
            people_boxes.append((x1, y1, x2, y2))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # проверка центра bbox
    person_in_roi = False
    for x1, y1, x2, y2 in people_boxes:
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        if x <= cx <= x + w and y <= cy <= y + h:
            person_in_roi = True
            last_seen_time = frame_timestamp
            break

    prev_state = state

    # логика состояний
    if state == "empty":
        if person_in_roi:
            state = "approached"
            state_start_time = frame_timestamp

    elif state == "approached":
        if not person_in_roi:
            if frame_timestamp - last_seen_time > empty_delay:
                state = "empty"
                state_start_time = frame_timestamp
                last_leave_time = last_seen_time  # фиксируем уход
        else:
            if frame_timestamp - state_start_time >= occupy_delay:
                state = "occupied"
                state_start_time = frame_timestamp

    elif state == "occupied":
        if not person_in_roi:
            if frame_timestamp - last_seen_time > empty_delay:
                state = "empty"
                state_start_time = frame_timestamp
                last_leave_time = last_seen_time  # фиксируем уход

    # лог событий
    if state != prev_state:
        log_event(frame_timestamp, state)

    # визуализация
    color = colors.get(state, (0,0,0))

    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)

    # текст с состоянием над ROI
    label = f"{state}"
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

    text_x = x
    text_y = max(text_h + 5, y - 10)

    cv2.putText(
        frame,
        label,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2,
    )

    # время
    cv2.putText(
        frame,
        f"time: {frame_timestamp:.1f}s",
        (20, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    # запись
    out.write(frame)
    cv2.imshow("Table Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

    frame_id += 1

# завершение
cap.release()
out.release()
cv2.destroyAllWindows()

# аналитика
df = pd.DataFrame(events)

delays = []

last_empty_time = None

for _, row in df.iterrows():
    if row["event"] == "empty":
        last_empty_time = row["timestamp"]

    elif row["event"] == "approached" and last_empty_time is not None:
        delay = row["timestamp"] - last_empty_time

        if delay >= 0:
            delays.append(delay)

        last_empty_time = None

# отчет
print("\n--- REPORT ---")
print(f"Всего событий: {len(df)}")
print(f"Количество задержек: {len(delays)}")

if delays:
    avg_delay = sum(delays) / len(delays)
    print(f"Средняя задержка: {round(avg_delay, 2)} сек")
    print(f"Минимальная: {round(min(delays), 2)} сек")
    print(f"Максимальная: {round(max(delays), 2)} сек")
    print(f"Все задержки: {[round(d, 2) for d in delays]}")
else:
    print("Недостаточно данных")

# сохранение
df.to_csv("events.csv", index=False)
