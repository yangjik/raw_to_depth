import cv2
import shutil, os, sys
import numpy as np
import argparse

zed_cam = 1 

def create_folder(image_path):
    
    if os.path.exists(image_path) == 0:
        os.mkdir(image_path)
        os.mkdir(image_path + '/imgLeft')
        os.mkdir(image_path + '/imgRight')
        
        print('create ', image_path, '\n')
    else:
        shutil.rmtree(image_path)
        os.mkdir(image_path)
        os.mkdir(image_path + '/imgLeft')
        os.mkdir(image_path + '/imgRight')
        
        print('delete before create ', image_path, '\n')

def image_size(camera_mode):
    
    if zed_cam == 1:
        if camera_mode == 'VGA':
            width = 1344
            hight = 376
        elif camera_mode == 'HD':
            width = 2560
            hight = 720
        elif camera_mode == 'FHD':
            width = 3840
            hight = 1080
        else:
            txt = f'Error : parameter -d / --device input : {camera_mode}'
            test = '-' * len(txt)

            sys.exit(f'\n{test}\n{txt}\n{test}\n')
            
    return width, hight

def caputer_img(image_path, camera_device, width, hight):
    
    video = cv2.VideoCapture(camera_device)
    video.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, hight)
    
    if not video.isOpened():
        sys.exit(f'Error : {camera_device} camera is none. Change device number\n')

    num = 1
    while True:
        ret, frame = video.read()
    
        if not ret:
            break
        
        # split left, right image
        frame_left = frame[:, :int(width/2), :]
        frame_right = frame[:, int(width/2):, :]
        
        # cv2.imshow('org', frame)
        cv2.imshow('left', frame_left)
        cv2.imshow('right', frame_right)
        
        key = cv2.waitKey(1)
        
        if key == 27:   # esc
            cv2.destroyAllWindows()
            video.release()
            break 
        
        elif key == 83 or key == 115: # s or S
            left_save_path = f'{image_path}/imgLeft/left_{num:02d}.png'
            right_save_path = f'{image_path}/imgRight/right_{num:02d}.png'

            l_success = cv2.imwrite(left_save_path, frame_left)
            r_success = cv2.imwrite(right_save_path, frame_right)
            
            if l_success == 1 and r_success == 1:
                print("left, right save\n")
                num += 1
            else:
                print('\nnot save image\n')
            
        
def main(image_path, camera_mode, camera_device):
    
    create_folder(image_path)
    
    org_img_w,  org_img_h = image_size(camera_mode)
    
    caputer_img(image_path, camera_device, org_img_w, org_img_h)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Camera capture')

    # save directory
    parser.add_argument('-p', '--path', type=str, default='./opencv_calibration/cam_capture',
                        help='useing opencv and save left, right image direactory'
                        )
    
    # camera resolution
    parser.add_argument('-m', '--mode', type=str, default='HD',
                        help='camera resolution : VGA, HD, FHD'
                        )

    # usb device num
    parser.add_argument('-d', '--device', type=int, default=0 )

    # arg save
    args = parser.parse_args()
    
    image_path = args.path
    camera_mode = args.mode
    camera_device = args.device
    
    main(image_path, camera_mode, camera_device)