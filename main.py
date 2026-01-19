import cv2
import ollama
import threading
import time
from ultralytics import YOLO

# --- 配置 ---
# 1. 加载 YOLO 模型 (第一次运行会自动下载 yolov8n.pt, 非常快)
yolo_model = YOLO('yolov8n.pt') 

# 2. Ollama 模型 (建议使用 moondream 或 llama3.2-vision)
OLLAMA_MODEL = "moondream" 

# 全局变量，用于线程间通信
current_frame_analysis = "Waiting for analysis..."
is_analyzing = False

def analyze_frame_with_ollama(frame):
    """
    这个函数将在单独的线程中运行，负责发送图像给 Ollama。
    """
    global current_frame_analysis, is_analyzing
    is_analyzing = True
    
    try:
        # 编码图像
        _, buffer = cv2.imencode('.jpg', frame)
        image_bytes = buffer.tobytes()

        # 发送给 Ollama
        # 这里你可以根据 YOLO 的结果定制 Prompt
        # 比如： "I see a person here. What are they holding?"
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{
                'role': 'user',
                'content': 'Describe the atmosphere and main acticity in this image very briefly.',
                'images': [image_bytes]
            }]
        )
        current_frame_analysis = response['message']['content']
        
    except Exception as e:
        print(f"Ollama Error: {e}")
    finally:
        is_analyzing = False

# --- 主循环 ---
# 将 0 改为你刚才测出来的数字 (例如 2)
cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)

# 设置每隔多少帧触发一次 LLM 分析 (例如每 5 秒触发一次)
# 或者你可以改为基于事件触发 (例如: 检测到 'person' 时触发)
last_analysis_time = 0
ANALYSIS_INTERVAL = 5 # seconds

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. YOLO 推理 (Real-time Object Detection)
    # verbose=False 防止控制台刷屏
    results = yolo_model(frame, verbose=False) 
    
    # 在帧上绘制 YOLO 的检测框
    annotated_frame = results[0].plot()

    # 获取检测到的类别列表 (例如: [0, 0, 15] -> ['person', 'person', 'cat'])
    detected_classes = results[0].boxes.cls.tolist()
    class_names = [results[0].names[int(cls)] for cls in detected_classes]

    # 2. 触发逻辑 (Trigger Logic)
    current_time = time.time()
    
    # 逻辑示例：如果距离上次分析超过 5 秒，并且画面中检测到了 'person' (人)
    # 这种 "Conditional Trigger" 是结合两者的精髓
    if (current_time - last_analysis_time > ANALYSIS_INTERVAL) and not is_analyzing:
        if 'person' in class_names: # 只有看到人才问 LLM
            # 启动后台线程，不阻塞视频流
            threading.Thread(target=analyze_frame_with_ollama, args=(frame.copy(),)).start()
            last_analysis_time = current_time

    # 3. UI 显示
    # 将 YOLO 的画面和 LLM 的分析文字叠加显示
    cv2.putText(annotated_frame, f"YOLO: {', '.join(set(class_names))}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # 显示 LLM 的分析结果 (可能延迟较高，但不会卡住画面)
    cv2.putText(annotated_frame, f"LLM: {current_frame_analysis}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow('YOLO + Ollama Hybrid', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()