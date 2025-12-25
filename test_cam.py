import cv2

URL = "http://10.247.62.9:8080/video?dummy=param.mjpg"

print("Opening stream:", URL)
cap = cv2.VideoCapture(URL, cv2.CAP_FFMPEG) 

print("cap.isOpened() ->", cap.isOpened())

for i in range(10):
    ret, frame = cap.read()
    print(f"Frame {i}: ret={ret}, frame_is_None={frame is None}")
    if ret and frame is not None:
        cv2.imshow("Test Camera", frame)
        if cv2.waitKey(1) == 27:  # ESC to exit
            break

cap.release()
cv2.destroyAllWindows()