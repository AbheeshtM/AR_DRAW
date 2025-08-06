import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(1)
cv2.namedWindow("🎨 Finger Draw App", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("🎨 Finger Draw App", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

draw_color = (0, 0, 0)
prev_x, prev_y = None, None
drawing = False
background_mode = 0  # 0 = camera, 1 = white canvas, 2 = black canvas

undo_stack = deque(maxlen=20)
redo_stack = deque(maxlen=20)
canvas = None

# Color Palette
palette_items = [
    ((148, 0, 211), "Violet"),
    ((75, 0, 130), "Indigo"),
    ((255, 0, 0), "Red"),
    ((255, 165, 0), "Orange"),
    ((255, 255, 0), "Yellow"),
    ((0, 255, 0), "Green"),
    ((0, 0, 255), "Blue"),
]
clear_button_pos = (10, 80)
exit_button_pos = (10, 140)
button_size = 50

def draw_palette_ui(frame):
    for i, (color, _) in enumerate(palette_items):
        x = i * (button_size + 10) + 10
        y = 10
        cv2.rectangle(frame, (x, y), (x + button_size, y + button_size), color, -1)
        cv2.rectangle(frame, (x, y), (x + button_size, y + button_size), (255, 255, 255), 2)

    cx, cy = clear_button_pos
    cv2.rectangle(frame, (cx, cy), (cx + 120, cy + 40), (0, 0, 0), -1)
    cv2.putText(frame, "CLEAR", (cx + 15, cy + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    ex, ey = exit_button_pos
    cv2.rectangle(frame, (ex, ey), (ex + 120, ey + 40), (0, 0, 0), -1)
    cv2.putText(frame, "EXIT", (ex + 25, ey + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

def get_hovered_color(x, y):
    for i, (color, _) in enumerate(palette_items):
        bx = i * (button_size + 10) + 10
        by = 10
        if bx < x < bx + button_size and by < y < by + button_size:
            return color
    return None

def is_clear_button_pressed(x, y):
    cx, cy = clear_button_pos
    return cx < x < cx + 120 and cy < y < cy + 40

def is_exit_button_pressed(x, y):
    ex, ey = exit_button_pos
    return ex < x < ex + 120 and ey < y < ey + 40

def count_fingers(hand_landmarks):
    tips = [8, 12, 16, 20]  # Index to pinky
    pip_joints = [6, 10, 14, 18]
    fingers_up = 0
    for tip, pip in zip(tips, pip_joints):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
            fingers_up += 1
    if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:  # Thumb
        fingers_up += 1
    return fingers_up

# === Main Loop ===
while True:
    if background_mode == 0:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
    elif background_mode == 1:
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
    else:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

    h, w, _ = frame.shape
    if canvas is None:
        canvas = np.zeros_like(frame)

    display = frame.copy()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    draw_palette_ui(display)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(display, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            index_tip = hand_landmarks.landmark[8]
            middle_tip = hand_landmarks.landmark[12]
            index_pip_y = hand_landmarks.landmark[6].y
            middle_pip_y = hand_landmarks.landmark[10].y
            x = int(index_tip.x * w)
            y = int(index_tip.y * h)

            fingers = count_fingers(hand_landmarks)

            # Draw hand trace (circle)
            cv2.circle(display, (x, y), 8, draw_color, -1)

            # 5 Fingers = Erase
            if fingers == 5:
                erase_color = (255, 255, 255) if background_mode == 1 else (0, 0, 0)
                if not drawing:
                    drawing = True
                    prev_x, prev_y = x, y
                    undo_stack.append(canvas.copy())
                    redo_stack.clear()
                else:
                    cv2.line(canvas, (prev_x, prev_y), (x, y), erase_color, 40)
                    prev_x, prev_y = x, y
                continue

            # 2 fingers = SELECT
            if index_tip.y < index_pip_y and middle_tip.y < middle_pip_y:
                hovered_color = get_hovered_color(x, y)
                if hovered_color:
                    draw_color = hovered_color
                    drawing = False
                    prev_x, prev_y = None, None
                    continue
                elif is_clear_button_pressed(x, y):
                    undo_stack.append(canvas.copy())
                    canvas = np.zeros_like(frame)
                    drawing = False
                    prev_x, prev_y = None, None
                    continue
                elif is_exit_button_pressed(x, y):
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()

            # DRAW mode (1 finger only)
            elif index_tip.y < index_pip_y and middle_tip.y > middle_pip_y:
                if not drawing:
                    drawing = True
                    prev_x, prev_y = x, y
                    undo_stack.append(canvas.copy())
                    redo_stack.clear()
                else:
                    cv2.line(canvas, (prev_x, prev_y), (x, y), draw_color, 8)
                    prev_x, prev_y = x, y
            else:
                drawing = False
                prev_x, prev_y = None, None

    # Merge canvas with frame
    display = cv2.add(display, canvas)
    cv2.imshow("🎨 Finger Draw App", display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('u') and undo_stack:
        redo_stack.append(canvas.copy())
        canvas = undo_stack.pop()
    elif key == ord('r') and redo_stack:
        undo_stack.append(canvas.copy())
        canvas = redo_stack.pop()
    elif key == ord('s'):
        cv2.imwrite("drawing.png", canvas)
    elif key == ord('t'):
        background_mode = (background_mode + 1) % 3

cap.release()
cv2.destroyAllWindows()
