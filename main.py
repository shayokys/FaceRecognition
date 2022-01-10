import os
import dlib
import cv2
from scipy.spatial import distance
#Вывод названия файлов, нужно будет для перебора файлов и поиска нужного
flag=False
for filename in os.listdir('faces'):

    # Получение изображения с вебкамеры и сохранение его
    cap = cv2.VideoCapture(0)
    # "Прогреваем" камеру, чтобы снимок не был тёмным
    for i in range(30):
        cap.read()

    # Делаем снимок
    ret, frame = cap.read()

    # Записываем в файл
    cv2.imwrite('cam.png', frame)

    # Отключаем камеру
    cap.release()

    # Выявление степени схожести фотографий, если найдено фото с которым есть совпадение, то человеку разрешен доступ, иначе запрещен
    # creating models' objects
    sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
    detector = dlib.get_frontal_face_detector()
    # proccessing the first picture
    image_1 = cv2.imread('cam.png')
    faces_1 = detector(image_1, 1)
    shape = sp(image_1, faces_1[0])
    face_descriptor_1 = facerec.compute_face_descriptor(image_1, shape)

    # proccessing the second picture
    image_2 = cv2.imread('faces/'+filename)
    faces_2 = detector(image_2, 1)
    shape = sp(image_2, faces_2[0])
    face_descriptor_2 = facerec.compute_face_descriptor(image_2, shape)

    # calculating euclidean distance
    result = distance.euclidean(face_descriptor_1, face_descriptor_2)

    if result < 0.6:
        flag=True
        break
if(flag==True):
    print('Доступ разрешен!')
else:
    print('Доступ запрещен!')
