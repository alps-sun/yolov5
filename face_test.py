# coding=gbk
"""
����ͷ����ʶ��
���ߣ�����
@ʱ��  : 2021/9/5 17:15
Haar�����������ͷ
"""
import cv2

#�����µ�cam����
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
#��ʼ������ʶ������Ĭ�ϵ�����haar������
face_cascade = cv2.CascadeClassifier(r'./default.xml')

while True:
    # ������ͷ��ȡͼ��
    _, image = cap.read()
    # ת��Ϊ�Ҷ�
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # ���ͼ���е���������
    faces = face_cascade.detectMultiScale(image_gray, 1.3, 5)
    # Ϊÿ����������һ����ɫ����
    for x, y, width, height in faces:
        cv2.rectangle(image, (x, y), (x + width, y + height), color=(255, 0, 0), thickness=2)
    cv2.imshow("image", image)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()