import cv2
import sys, os
import argparse, json
import numpy as np

left_last_path = ''
right_last_path = ''

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

def find_corner(image_path, check_col, check_raw, check_size):
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    
    objPt        = np.zeros((check_col * check_raw, 3), np.float32)
    objPt[:, :2] = np.mgrid[0:check_col, 0:check_raw].T.reshape(-1,2)
    
    objPt        = objPt * check_size
    
    point_3d = []
    point_left_2d = []
    point_rigdt_2d = []
        
    left_path = image_path + '/imgLeft'
    right_path = image_path + '/imgRight'
    dir_left = os.listdir(left_path)
    dir_right = os.listdir(right_path)
    
    for left, right in zip(dir_left, dir_right):
        
        left_imread = left_path + '/' + left
        right_imread = right_path + '/' + right
        
        img_left = cv2.imread(left_imread)
        img_right = cv2.imread(right_imread)
        
        gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(img_right,cv2.COLOR_BGR2GRAY)
                
        # ------------- captuer image viwe ---------------- 
        # cv2.imshow('left', img_left)cd
        # cv2.imshow('right', img_right)
        # cv2.imshow('left', gray_left)
        # cv2.imshow('right', gray_right)
        
        # key = cv2.waitKey(0)
        
        # if key == 27:
        #     cv2.destroyAllWindows()
        # --------------------------------------------------
        left_retval, left_corner= cv2.findChessboardCorners(gray_left, (check_col, check_raw), None)
        right_retval, right_corner= cv2.findChessboardCorners(gray_right, (check_col, check_raw), None)

        if left_retval == True and right_retval == True:
            left_corner = cv2.cornerSubPix(gray_left, left_corner, (11,11), (-1,-1), criteria)
            right_corner = cv2.cornerSubPix(gray_right, right_corner, (11,11), (-1,-1), criteria)
            
            point_3d.append(objPt)
            point_left_2d.append(left_corner)
            point_rigdt_2d.append(right_corner)
        
            cv2.drawChessboardCorners(img_left, (check_col, check_raw), left_corner, left_retval)
            cv2.drawChessboardCorners(img_right, (check_col, check_raw), right_corner, right_retval)         
            # cv2.imshow('left', img_left)
            # cv2.imshow('right', img_right)
            
            key = cv2.waitKey(0)
            
            if key == 27:
                cv2.destroyAllWindows()

    global left_last_path
    left_last_path = left_imread
    
    global right_last_path
    right_last_path = right_imread

    return point_3d, point_left_2d, point_rigdt_2d, gray_left, gray_right


def single_pair_calibration(point_3d, point_left_2d, point_right_2d, gray_left, gray_right):
    
    # single (왼쪽, 오른쪽)
    left_ret, left_mtx, left_dist, left_rvecs, left_tvecs = cv2.calibrateCamera(point_3d, point_left_2d, 
                                                                                gray_left.shape[: : -1], None, None)   # gray_left.shape[: : -1] : (w, h)
    right_ret, right_mtx, right_dist, right_rvecs, right_tvecs = cv2.calibrateCamera(point_3d, point_right_2d, 
                                                                                     gray_right.shape[: : -1], None, None)

    # print("left : ", left_mtx)
    # print()
    # print("right : ", right_mtx)
    # print()

    w,h = gray_left.shape
    
    # left calibration -> 왜곡 해제 alpha=0이면 불필요한 픽셀이 최소인 왜곡되지 않은 이미지를 반환
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(left_mtx, left_dist, (w,h), 0, (w,h))
    left_dst = cv2.undistort(gray_left, left_mtx, left_dist, None, newcameramtx)    

    # right calibration -> 왜곡 해제 alpha=0이면 불필요한 픽셀이 최소인 왜곡되지 않은 이미지를 반환
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(right_mtx, right_dist, (w,h), 0, (w,h))
    right_dst = cv2.undistort(gray_right, right_mtx, right_dist, None, newcameramtx)    
    
    cv2.imshow('left_Undistortion', left_dst)
    cv2.imshow('right_Undistortion', right_dst)

    # pair (좌우 이미지 동시에)
    flags = 0
    flags |= cv2.CALIB_USE_INTRINSIC_GUESS
    
    criteria_stereo= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-6)

    # K1, K2(내부 행렬)/ D1, D2(왜곡 계수) / R(회전행렬) / T(이동 백터)
    stereo_ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(point_3d, point_left_2d, point_right_2d, left_mtx, left_dist,
                                                               right_mtx, right_dist, gray_right.shape[: : -1], criteria=criteria_stereo, flags=flags)
    
    # print('K1 : ', K1)
    # print()
    # print('D1 : ', D1)
    # print()
    # print('R : ', R)
    # print()    
    # print('T : ', T)
    # print()

    # 오차율(낮을수록 일치 1.0 넘으면 x)
    if left_ret > 1 or right_ret > 1 or stereo_ret > 1:
        print(f'left score : {left_ret}\nright score : {right_ret}\nstereo score : {stereo_ret}')
        print('checkboard add capture')
    else :
        print('-' * 30)
        print('calibration score')
        print(f'left score : {left_ret:.3f}')
        print(f'right score : {right_ret:.3f}')
        print(f'stereo score : {stereo_ret:.3f}')
        print('-' * 30)
    
    # R1, R2(보정된 카메라 회전행렬) / P1, P2(보정된 투영행렬)
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(K1, D1, K2, D2, imageSize=gray_right.shape[: : -1], R=R, T=T,
                                                      flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.0 )
    
    # print('R1 : ', R1)
    # print()
    # print('P1 : ', P1)

    # save data
    # save_json(R1, R2, P1, P2, K1, K2, D1, D2)
    
    # 왜곡 보정 -> 카메라 정렬 CV_16SC2, CV_32FC1, or CV_32FC2
    left_map1, left_map2 = cv2.initUndistortRectifyMap(K1, D1, R1, P1, gray_left.shape[::-1], cv2.CV_32FC2)     # 첫번째 인자 k1값이 CV_16SC2 이면 반환되는 값도 같은 형태로 줘야하므로
    right_map1, right_map2 = cv2.initUndistortRectifyMap(K2, D2, R2, P2, gray_right.shape[::-1], cv2.CV_32FC2)  # 첫번째 인자 k2값이 CV_16SC2 이면 반환되는 값도 같은 형태로 줘야하므로

    # 보정된 영상 생성
    rectified_left = cv2.remap(gray_left, left_map1, left_map2, cv2.INTER_LINEAR)
    rectified_right = cv2.remap(gray_right, right_map1, right_map2, cv2.INTER_LINEAR)

    w, h = rectified_left.shape[::-1]

    # 렉티파이 이후 이미지에다가 선그리기
    for i in range(0, h, int(h/20)):
        
        cv2.line(rectified_left, (0, i), (w, i), (0,255), 2)
        cv2.line(rectified_right, (0, i), (w, i), (0,255), 2)

    # 결과 이미지 보기
    # cv2.imshow('Rectified Left', rectified_left)
    # cv2.imshow('Rectified Right', rectified_right)
    
    key = cv2.waitKey(0)
    if key == 27:
        cv2.destroyAllWindows()
        
def save_json(R1, R2, P1, P2, K1, K2, D1, D2):

    # R1, R2, P1, P2 = stereoRectify matrix
    # K1, K2, D1, D2 = stereoCalibration matrix 
    R1 = R1.tolist()
    R2 = R2.tolist()
    P1 = P1.tolist()
    P2 = P2.tolist()
    K1 = K1.tolist()
    K2 = K2.tolist()
    D1 = D1.tolist()
    D2 = D2.tolist()
    
    rectify_matrix = {
            "R1" : R1,
            "R2" : R2,
            "P1" : P1,
            "P2" : P2,
            "K1" : K1,
            "K2" : K2,
            "D1" : D1,
            "D2" : D2,
            "left_path" : left_last_path,
            "right_path" : right_last_path,
            }
    
    save_data = [rectify_matrix,
                 ]
    
    save_path =  ['./opencv_calibration/stereo_rectify.json', 
                ]
    
    for path, data in zip(save_path, save_data):    

        with open(path, 'w') as json_file:
            json.dump(data, json_file, indent=2)
        
    
def main(image_path, yaml_path, cam_mode, check_col, check_raw, check_size):

    # input parameter get width, height
    img_width, img_height = image_size(cam_mode)

    # capture left, right image get corner data
    point_3d, point_left_2d, point_rigdt_2d, gray_left, gray_right = find_corner(image_path, check_col, check_raw, check_size)
    
    # left, right, pair calibration
    single_pair_calibration(point_3d, point_left_2d, point_rigdt_2d, gray_left, gray_right)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='camera calibration')
    
    parser.add_argument('-p', '--path', type=str, default='./opencv_calibration/cam_capture',
                        help='capture image path')

    parser.add_argument('-o', '--output', type=str, default='./opencv_calibration/config',
                        help='camera calibration & rectify result yaml file')
    
    parser.add_argument('-m', '--mode', type=str, default='HD',
                        help='Camera mode. VGA, HD, FHD')
    
    parser.add_argument('-c', '--column', type=int, default=8,
                        help='check board column(width)')
    
    parser.add_argument('-r', '--raw', type=int, default=6,
                        help='check board raw(height)')
    
    parser.add_argument('-s', '--size', type=int, default=15,
                        help='chech board size(mm)')
    
    args = parser.parse_args()
    image_path = args.path
    yaml_path = args.output
    cam_mode = args.mode
    check_col = args.column
    check_raw = args.raw
    check_size = args.size
    
    main(image_path, yaml_path, cam_mode, check_col, check_raw, check_size)