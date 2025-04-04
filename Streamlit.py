import streamlit as st
from PIL import Image
import requests
import io
from ultralytics import YOLO
import torch
import cv2
import numpy as np
from io import BytesIO

DEVICE = "cpu"

model1 = YOLO("Models/yolo666.pt")
model2 = YOLO("Models/yolo_1.pt")
Model3 = torch.jit.load('Models/best_model_new.pt', map_location=DEVICE)

def mask_faces(image_np):
    """
    Функция для маскировки лиц на изображении.
    :param image_np: Изображение в формате NumPy (BGR).
    :return: Изображение с замаскированными лицами.
    """
    # Преобразование изображения в grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

    # Загрузка предварительно обученной модели Haar Cascade для обнаружения лиц
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Обнаружение лиц
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Проход по каждому лицу
    for (x, y, w, h) in faces:
        # Размытие области лица
        face_roi = image_np[y:y+h, x:x+w]
        blurred_face = cv2.GaussianBlur(face_roi, (99, 99), 30)
        image_np[y:y+h, x:x+w] = blurred_face

    return image_np


# Обработка загруженных файлов
def process_uploaded_files(files):
    for file in files:
        st.write(f"Имя файла: {file.name}")
        img = Image.open(file)
        st.image(img, caption="Загруженное изображение", use_column_width=True)

def process_url_image(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content))
        st.image(img, caption="Изображение из URL", use_column_width=True)
    except Exception as e:
        st.error(f"Ошибка при загрузке изображения по URL: {e}")

def page0():
    st.title("Проект делали")
    st.subheader("Артём - Детекция лиц 🤨")
    st.subheader("Дмитрий - Детекции опухулей 🧠")
    st.subheader("Ксения - Сегментация аэрокосмических снимков 🌎")


def page1():
    st.title("Детекция лиц 🤨")

    # Загрузка изображений
    uploaded_files = st.file_uploader(
        "Загрузи одно или несколько изображений:", 
        type=["png", "jpg", "jpeg"], 
        accept_multiple_files=True
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Конвертация загруженного файла в PIL.Image
            image = Image.open(uploaded_file)
            
            # Отображение исходного изображения
            st.subheader("Исходное изображение:")
            st.image(image, caption="Исходное изображение", use_container_width=True)

            # Преобразование изображения в формат NumPy и конвертация RGB -> BGR
            image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # Обработка изображения моделью YOLO
            results = model1.predict(source=image_np, conf=0.6)  # conf - порог уверенности

            # Вывод результатов
            st.subheader("Результаты детекции:")
            for result in results:
                # Получение bounding box'ов
                boxes = result.boxes.xyxy.cpu().numpy()  # Координаты bounding box'ов
                confidences = result.boxes.conf.cpu().numpy()  # Уверенность модели

                # Копия изображения для маскировки
                masked_image = image_np.copy()

                # Проход по каждому обнаруженному лицу
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)  # Координаты bounding box'а
                    face_roi = masked_image[y1:y2, x1:x2]  # Область лица
                    blurred_face = cv2.GaussianBlur(face_roi, (99, 99), 30)  # Размытие области лица
                    masked_image[y1:y2, x1:x2] = blurred_face  # Замена области лица на размытое изображение

                # Конвертация обратно из BGR в RGB для отображения в Streamlit
                masked_image_rgb = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
                st.image(masked_image_rgb, caption="Результат маскировки", use_container_width=True)


        # Функция для обработки изображений по URL
    def process_url_image(url):
        try:
            # Скачивание изображения по URL
            response = requests.get(url)
            response.raise_for_status()  # Проверка на ошибки при запросе

            # Открытие изображения из байтов
            image = Image.open(BytesIO(response.content))

            # Отображение исходного изображения
            st.subheader("Исходное изображение (по URL):")
            st.image(image, caption="Исходное изображение", use_container_width=True)

            # Преобразование изображения в формат NumPy и конвертация RGB -> BGR
            image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # Обработка изображения моделью YOLO
            results = model1.predict(source=image_np, conf=0.6)  # conf - порог уверенности

            # Вывод результатов
            st.subheader("Результаты детекции:")
            for result in results:
                # Получение bounding box'ов
                boxes = result.boxes.xyxy.cpu().numpy()  # Координаты bounding box'ов
                confidences = result.boxes.conf.cpu().numpy()  # Уверенность модели

                # Копия изображения для маскировки
                masked_image = image_np.copy()

                # Проход по каждому обнаруженному лицу
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)  # Координаты bounding box'а
                    face_roi = masked_image[y1:y2, x1:x2]  # Область лица
                    blurred_face = cv2.GaussianBlur(face_roi, (99, 99), 30)  # Размытие области лица
                    masked_image[y1:y2, x1:x2] = blurred_face  # Замена области лица на размытое изображение

                # Конвертация обратно из BGR в RGB для отображения в Streamlit
                masked_image_rgb = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
                st.image(masked_image_rgb, caption="Результат маскировки", use_container_width=True)

        except Exception as e:
            st.error(f"Ошибка при загрузке изображения по URL: {e}")

    # Поле для ввода URL
    url = st.text_input("Введи ссылку на изображение:")
    if url:
        st.write("Обработка изображения по URL...")
        process_url_image(url)

    st.subheader("Число эпох обучения: 5")
    st.subheader("Объем выборки: 13.386 изображений")
    st.subheader("Метрики")

    st.image("/home/artem/Загрузки/confusion_matrix.png")
    st.image("/home/artem/Загрузки/results(1).png")
    st.image("/home/artem/Загрузки/F1_curve(2).png")

def page2():
    st.title("Детекции опухулей 🧠")

    # Загрузка изображений
    uploaded_files = st.file_uploader(
        "Загрузи одно или несколько изображений:", 
        type=["png", "jpg", "jpeg"], 
        accept_multiple_files=True
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Конвертация загруженного файла в PIL.Image
            image = Image.open(uploaded_file)
            
            # Отображение исходного изображения
            st.subheader("Исходное изображение:")
            st.image(image, caption="Исходное изображение", use_container_width=True)

            # Преобразование изображения в формат NumPy и конвертация RGB -> BGR
            image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # Обработка изображения моделью YOLO
            results = model2.predict(source=image_np, conf=0.3)  # conf - порог уверенности

            # Вывод результатов
            st.subheader("Результаты детекции:")
            for result in results:
                # Получение изображения с нарисованными bounding box'ами
                result_image = result.plot()  # plot() возвращает изображение с разметкой
                
                # Конвертация обратно из BGR в RGB для отображения в Streamlit
                result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                
                # Отображение результата
                st.image(result_image_rgb, caption="Результат детекции", use_container_width=True)


    # Функция для обработки изображений по URL
    def process_url_image(url):
        try:
            # Скачивание изображения по URL
            response = requests.get(url)
            response.raise_for_status()  # Проверка на ошибки при запросе

            # Открытие изображения из байтов
            image = Image.open(BytesIO(response.content))

            # Отображение исходного изображения
            st.subheader("Исходное изображение (по URL):")
            st.image(image, caption="Исходное изображение", use_container_width=True)

            # Преобразование изображения в формат NumPy и конвертация RGB -> BGR
            image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # Обработка изображения моделью YOLO
            results = model2.predict(source=image_np, conf=0.3)  # conf - порог уверенности

            # Вывод результатов
            st.subheader("Результаты детекции:")
            for result in results:
                # Получение изображения с нарисованными bounding box'ами
                result_image = result.plot()  # plot() возвращает изображение с разметкой
                
                # Конвертация обратно из BGR в RGB для отображения в Streamlit
                result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                
                # Отображение результата
                st.image(result_image_rgb, caption="Результат детекции", use_container_width=True)

        except Exception as e:
            st.error(f"Ошибка при загрузке изображения по URL: {e}")

    # Поле для ввода URL
    url = st.text_input("Введи ссылку на изображение:")
    if url:
        st.write("Обработка изображения по URL...")
        process_url_image(url)

    st.subheader("Число эпох обучения: 30")
    st.subheader("Объем выборки: 893 изображений")
    st.subheader("Метрики")

    st.image("/home/artem/Загрузки/Дима/confusion_matrix.png")
    st.image("/home/artem/Загрузки/Дима/results.png")
    st.image("/home/artem/Загрузки/Дима/F1_curve.png")


def page3():
    st.title("Сегментация аэрокосмических снимков 🌎")

    # Загрузка изображений
    uploaded_files = st.file_uploader(
        "Загрузи одно или несколько изображений:", 
        type=["png", "jpg", "jpeg"], 
        accept_multiple_files=True
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Конвертация загруженного файла в PIL.Image
            image = Image.open(uploaded_file)
            
            # Отображение исходного изображения
            st.subheader("Исходное изображение:")
            st.image(image, caption="Исходное изображение", use_container_width=True)

            # Преобразование изображения в формат NumPy и конвертация RGB -> BGR
            image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # Обработка изображения моделью YOLO
            results = Model3.predict(source=image_np, conf=0.3)  # conf - порог уверенности

            # Вывод результатов
            st.subheader("Результаты детекции:")
            for result in results:
                # Получение изображения с нарисованными bounding box'ами
                result_image = result.plot()  # plot() возвращает изображение с разметкой
                
                # Конвертация обратно из BGR в RGB для отображения в Streamlit
                result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                
                # Отображение результата
                st.image(result_image_rgb, caption="Результат детекции", use_container_width=True)


    # Функция для обработки изображений по URL
    def process_url_image(url):
        try:
            # Скачивание изображения по URL
            response = requests.get(url)
            response.raise_for_status()  # Проверка на ошибки при запросе

            # Открытие изображения из байтов
            image = Image.open(BytesIO(response.content))

            # Отображение исходного изображения
            st.subheader("Исходное изображение (по URL):")
            st.image(image, caption="Исходное изображение", use_container_width=True)

            # Преобразование изображения в формат NumPy и конвертация RGB -> BGR
            image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # Обработка изображения моделью YOLO
            results = model2.predict(source=image_np, conf=0.3)  # conf - порог уверенности

            # Вывод результатов
            st.subheader("Результаты детекции:")
            for result in results:
                # Получение изображения с нарисованными bounding box'ами
                result_image = result.plot()  # plot() возвращает изображение с разметкой
                
                # Конвертация обратно из BGR в RGB для отображения в Streamlit
                result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                
                # Отображение результата
                st.image(result_image_rgb, caption="Результат детекции", use_container_width=True)

        except Exception as e:
            st.error(f"Ошибка при загрузке изображения по URL: {e}")

    # Поле для ввода URL
    url = st.text_input("Введи ссылку на изображение:")
    if url:
        st.write("Обработка изображения по URL...")
        process_url_image(url)

    st.subheader("Число эпох обучения: 5")
    st.subheader("Объем выборки: 5.000")
    st.subheader("Метрики")


def main():
    st.sidebar.title("Навигация")
    
    page = st.sidebar.selectbox("Выберите страницу", ("Команда Yolo", "Детекция лиц", "Детекции опухулей", "Сегментация аэрокосмических снимков"))

    if page == "Команда Yolo":
        page0()
    elif page == "Детекция лиц":
        page1()
    elif page == "Детекции опухулей":
        page2()
    elif page == "Сегментация аэрокосмических снимков":
        page3()

if __name__ == "__main__":
    main()