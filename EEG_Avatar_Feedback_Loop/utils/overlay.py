import cv2

def show_emotion(emotion="neutral"):
    icon_path = f"media/emotion_icons/{emotion}.png"
    emotion_img = cv2.imread(icon_path)

    cv2.imshow("Emotion Overlay", emotion_img)
    cv2.waitKey(1)