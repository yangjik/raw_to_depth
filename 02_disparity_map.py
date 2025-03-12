import numpy as np
import cv2, os, sys, argparse, json

def image_size(cam_mode):
    
    if cam_mode == 'VGA':
        width = 1344
        height = 376
    elif cam_mode == 'HD':
        width = 2560
        height = 720
    elif cam_mode == 'FHD':
        width = 3840
        height = 1080
    else:
        txt = f'Error : -m, --device {cam_mode}'
        sys.exit()

    return width, height

def read_json(json_path):
    
    with open(json_path, "r") as json_data:
        data = json.load(json_data)
    
    return data    

def stereo_bm(gray_left, gray_right):
    
    num = 2
    numDisparities = 16 * num # disparity 검색범위 0 ~ 지정한범위(16배수)
    blockSize = 11          # 블록 선형 크기(홀수) <- 크면 매끄럽지만 정확도 낮음, 작으면 정확하지만 잘못된 대응을 찾을 가능성 높음

    stereo = cv2.StereoBM.create(numDisparities = numDisparities, blockSize = blockSize)
    disparity = stereo.compute(gray_left, gray_right)   # int16 -> 16bit
            
    # disparity 값을 0~255 범위로 정규화
    disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    disparity = disparity_normalized.astype(np.uint8)   # 8비트 변환

    return disparity

def stereo_sgbm(gray_left, gray_right):
    
    number_of_image_channels = 1
    minDisparity = 0    # 최소 disparity값
    numDisparities = 16   # 최대 disparity - 최소 disparity 값 (16배수 이면서 0보다 커야함)
    blockSize = 11      # 매칭 블럭 size 작을수록 연산 적음 3 ~ 11 범위
    P1 = 8 * number_of_image_channels * blockSize * blockSize       # disparity 부드러움을 제어
    P2 = 32 *number_of_image_channels * blockSize * blockSize       # disparity 부드러움을 제어 값이 클수록 불일치가 부드러워짐
    disp12MaxDiff=1     # 좌우 disparity 검사에 허용되는 최대 차이(픽셀단위)
    preFilterCap=63     # 필터링된 이미지 픽셀 자름
    uniquenessRatio = 15    # 5 ~ 15
    speckleWindowSize = 0       # 얼룩제거
    speckleRange = 1    # 
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY # 
    
    
    stereo = cv2.StereoSGBM.create(minDisparity = minDisparity,
                                   numDisparities = numDisparities,
                                   blockSize=blockSize,
                                   P1=P1,
                                   P2=P2,
                                   disp12MaxDiff = disp12MaxDiff,
                                   preFilterCap = preFilterCap,
                                   uniquenessRatio = uniquenessRatio,
                                   speckleWindowSize = speckleWindowSize,
                                   speckleRange = speckleRange,
                                   mode = mode
                                   )
  
    disparity = stereo.compute(gray_left, gray_right)

    # disparity 값을 0~255 범위로 정규화
    disparity = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    disparity = disparity.astype(np.uint8)   # 8비트 변환

    return disparity

# LUT 사용해야하는 이유
# https://mrsnake.tistory.com/142
# cv2 LUT 사용 안했을때
def streaming_video(device_num, js_data, img_width, img_height):
    
    # json 
    P1 = np.array(js_data["P1"])
    P2 = np.array(js_data['P2'])
    R1 = np.array(js_data['R1'])
    R2 = np.array(js_data['R2'])
    K1 = np.array(js_data['K1'])
    K2 = np.array(js_data['K2'])
    D1 = np.array(js_data['D1'])
    D2 = np.array(js_data['D2'])
    left_path = js_data['left_path']
    right_path = js_data['right_path']    
    
    left_gray = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
    right_gray = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
    
    left_map1, left_map2 = cv2.initUndistortRectifyMap(K1, D1, R1, P1, left_gray.shape[::-1], cv2.CV_32FC2)     # 첫번째 인자 k1값이 CV_16SC2 이면 반환되는 값도 같은 형태로 줘야하므로
    right_map1, right_map2 = cv2.initUndistortRectifyMap(K2, D2, R2, P2, right_gray.shape[::-1], cv2.CV_32FC2)  # 첫번째 인자 k2값이 CV_16SC2 이면 반환되는 값도 같은 형태로 줘야하므로
    
    video = cv2.VideoCapture(device_num)

    video.set(cv2.CAP_PROP_FRAME_WIDTH, img_width)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, img_height)
    
    if not video.isOpened():
        print("cam not connect. check divice num")
        sys.exit()

    while True:
        
        ret, frame = video.read()
        
        if ret == False:
            break
                
        # color left right frame
        color_left = frame[:, :int(img_width/2), :]
        color_right = frame[:, int(img_width/2):, :]
        
        c_rectify_left = cv2.remap(color_left, left_map1, left_map2, cv2.INTER_LINEAR)
        c_rectify_right = cv2.remap(color_right, right_map1, right_map2, cv2.INTER_LINEAR)
        
        # gray left right frame
        gray_left = cv2.cvtColor(c_rectify_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(c_rectify_right, cv2.COLOR_BGR2GRAY)
                
        # by StereoBM                
        bm_disparity = stereo_bm(gray_left, gray_right)

        # by StereoSGBM
        sgbm_disparity = stereo_sgbm(gray_left, gray_right)

        # --------------------------------------------
        # 원본
        # cv2.imshow("org", frame)

        # 보정 후 칼라 영상
        # cv2.imshow("left", c_rectify_left)
        # cv2.imshow("right", c_rectify_right)
        
        # 보정 후 회색영상
        # cv2.imshow("gray_left", gray_left)
        # cv2.imshow("gray_right", gray_right)

        # disparity map
        cv2.imshow("bm", bm_disparity)
        cv2.imshow("sgbm", sgbm_disparity)
        # --------------------------------------------

        key = cv2.waitKey(10)
        
        if key == 27:
            cv2.destroyAllWindows()
            video.release()


def main(json_path, device_num, cam_mode):

    img_width, img_height = image_size(cam_mode)

    js_data = read_json(json_path)

    streaming_video(device_num, js_data, img_width, img_height)

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # json path
    parser.add_argument('-p', '--path', type=str, default="./opencv_calibration/stereo_rectify.json",
                        help="stereo_rectify.json file path")
    
    parser.add_argument('-d', '--device', type=int, default=0,
                        help='usb camera number(cv2.videoCapture)')
    
    parser.add_argument('-m', '--mode', type=str, default='HD',
                        help='Camera mode. VGA, HD, FHD')
        
    arg = parser.parse_args()
        
    json_path = arg.path
    device_num = arg.device
    cam_mode = arg.mode
    
    main(json_path, device_num, cam_mode)