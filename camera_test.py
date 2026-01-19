import cv2
import time

def list_cameras():
    # 尝试遍历前 5 个索引
    for index in range(5):
        print(f"Checking camera index {index}...")
        
        # Mac 上强制使用 AVFOUNDATION 后端更稳
        cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        
        if cap.isOpened():
            print(f"✅ Camera {index} is OPEN. Reading frame...")
            ret, frame = cap.read()
            if ret:
                # 显示这个摄像头画面，让你确认是不是 OBS
                window_name = f"Camera Index {index}"
                cv2.imshow(window_name, frame)
                print(f"   -> Displaying Camera {index}. Press any key in the window to check next...")
                cv2.waitKey(0) # 等待按键
                cv2.destroyWindow(window_name)
            else:
                print(f"   -> Camera {index} opened but returned no frame (Black screen?).")
            cap.release()
        else:
            print(f"❌ Camera {index} failed to open.")
        
        print("---")

if __name__ == "__main__":
    print("请确保你已经在 OBS 点击了 '启动虚拟摄像机'！")
    list_cameras()
    cv2.destroyAllWindows()