# Shoe-recognition-program
This code counts the amount of shoes in a picture
import cv2
import numpy as np

# خواندن تصویر
image = cv2.imread('shoes.jpg')
original = image.copy()  # کپی از تصویر اصلی

# پیش‌پردازش
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 0)
edges = cv2.Canny(blur, 50, 150)

# پیدا کردن کانتورها
contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# شمارش و رسم کانتورهای معتبر
shoe_count = 0
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 500:
        approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
        if len(approx) >=4:
            shoe_count += 1
            cv2.drawContours(image, [cnt], -1, (0,255,0), 2)
            
            # محاسبه مرکز برای نوشتن شماره
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(image, f"Shoe {shoe_count}", (cX-40, cY), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

# ایجاد خروجی زیبا
result = np.hstack((original, image))  # کنار هم قرار دادن تصاویر

# نمایش نتایج با ویندوز زیبا
cv2.namedWindow('نتایج تشخیص کفش', cv2.WINDOW_NORMAL)
cv2.resizeWindow('نتایج تشخیص کفش', 1200, 600)
cv2.imshow('نتایج تشخیص کفش', result)

# متن نتایج در کنسول
print("\n" + "="*50)
print(f"🔍 تعداد کفش‌های تشخیص داده شده: {shoe_count} عدد")
print("="*50 + "\n")

cv2.waitKey(0)
cv2.destroyAllWindows()

 #ذخیره نتیجه
cv2.imwrite('result.jpg', result)
print("✅ تصویر نتیجه در فایل 'result.jpg' ذخیره شد.")

#------------------------------- برای دقت بالا تر از مدل yolo مدل اماده 
#from ultralytics import YOLO
#model = YOLO('yolov8n.pt')  # مدل پیش‌آموزش‌دیده
#results = model.predict('test.jpg')
#results[0].show()  # نمایش نتایج
