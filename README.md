# USB 카메라를 가지고 Calibration ~ Rectify ~ Disparity Map ~ Depth 과정 코드 구현

### 환경
---
- **OS** : Ubuntu 22.04 LTS
- **개발 툴** : VSCode
- **프로그램 언어** : Python
- **사용한 카메라** : Zed 2i를 USB 카메라로 인식하여 사용(제공된 라이브러리 사용하지 않음.)

### 라이브러리 설치
---
``` bash
pip install -r requirements.txt
```
</br>
### 코드 실행 순서
---
| 순서 | 파일명 | 간단 설명 |
| :-: | :-: | :-: |
|1|00_cam_capture.py|checkboard 캡쳐 후 left, right 각 폴더 저장|
|2|01_calibration.py|좌,우 calibration -> stereoRectify 과정 시각화|
|3|02_disparity_map.py|stereoBM(블록매칭), stereoSGBM(이웃 블록매칭) Disparity Map 생성|
|4|03_depth.py|depth = (baseline * focal_length_pixcel)/disparity 공식 적용|

### 상세 코드 설명
---
1. 

### 참고 문서 및 자료
- checkboard : <a>https://calib.io/pages/camera-calibration-pattern-generator</a>
- opencv 튜토리얼 : <a>https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html</a>
