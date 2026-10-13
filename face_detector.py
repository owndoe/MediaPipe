#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import mediapipe as mp

# =========================================
# 🧩 Mediapipe 초기화 (얼굴 감지 모듈)
# =========================================
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# model_selection: 0은 카메라와 가까운 거리(2m 이내), 1은 먼 거리(5m 이내)에 적합합니다.
face_detection = mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5  # 탐지 신뢰도
)

# =========================================
# 📸 카메라/동영상 파일 연결
# =========================================
# cap = cv2.VideoCapture(0)            # 기본 카메라 사용 시
cap = cv2.VideoCapture("face.mp4")     # 동영상 파일 사용 시 

print("📷 카메라 스트림 시작 — ESC를 눌러 종료합니다.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("⚠️ 영상이 끝났거나 프레임을 읽지 못했습니다. (카메라 사용 시 연결을 확인하세요)")
        break

    # 좌우 반전 (셀카 뷰)
    image = cv2.flip(image, 1)

    # BGR → RGB 변환 (MediaPipe는 RGB 입력을 사용합니다)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 얼굴 검출 수행
    result = face_detection.process(image_rgb)

    # 👤 얼굴 감지 상자 및 주요 랜드마크(눈, 코, 입, 귀 등) 표시
    if result.detections:
        for detection in result.detections:
            mp_drawing.draw_detection(image, detection)

    # 화면에 맞게 크기 조정 (세로 영상이 화면보다 커서 창이 잘리는 것 방지)
    h, w = image.shape[:2]
    if h > 720:
        image = cv2.resize(image, (int(w * 720 / h), 720))

    # 화면 표시
    cv2.imshow('👤 MediaPipe Face Detector', image)

    # ESC 키로 종료
    if cv2.waitKey(5) & 0xFF == 27:
        print("👋 종료합니다.")
        break

# =========================================
# 🔚 종료 처리
# =========================================
cap.release()
cv2.destroyAllWindows()
