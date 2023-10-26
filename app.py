# -*- coding: utf-8 -*-

from flask import Flask, request, render_template, url_for
import cv2
import numpy as np
import os

app = Flask(__name__)

# YOLO 모델 초기화 및 가중치 불러오기 (필요에 따라 경로 변경)
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'static'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

@app.route('/')
def upload_file():
    return render_template('upload.html')

@app.route('/uploader', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        f = request.files['file']
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
        f.save(file_path)
        
        img = cv2.imread(file_path)

        # YOLOv4를 사용하여 객체 탐지
        blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
        net.setInput(blob)
        output_layers_names = net.getUnconnectedOutLayersNames()
        layerOutputs = net.forward(output_layers_names)

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:  # 예측 신뢰도가 0.5보다 큰 경우에만 객체로 간주
                    center_x = int(detection[0] * img.shape[1])
                    center_y = int(detection[1] * img.shape[0])
                    w = int(detection[2] * img.shape[1])
                    h = int(detection[3] * img.shape[0])
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(img, classes[class_id], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 분석 결과 이미지 저장
        result_path = os.path.join(app.config['RESULT_FOLDER'], 'result.jpg')
        cv2.imwrite(result_path, img)

        return render_template('upload.html', result_image='result.jpg')

if __name__ == '__main__':
    app.run(debug=True)