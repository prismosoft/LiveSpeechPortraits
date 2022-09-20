import os
import cv2
import numpy as np
import json
from tqdm import tqdm
import subprocess
import argparse
from pathlib import Path


def parser_json(jp):
    kp = []
    with open(jp, 'r') as f:
        data = json.load(f)
    shapes = data['shapes']
    for p in shapes:
        ps = p['points'][0]
        kp.append(ps)

    return kp


def img2video(image_folder, video_name, img_format='png', fps=60.0):
    f = lambda x: float(x.split('_')[-1][:-4])  # case by case
    images = [img for img in os.listdir(image_folder) if img.endswith(".{}".format(img_format))]
    images = sorted(images, key=f)
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'MP4v')

    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height), True)

    for image in images:
        img = cv2.imread(os.path.join(image_folder, image))
        video.write(img)

    video.release()

    print("Succeeds!")


def lucas_kanade_method_imgs(resources_path, kp=None, save_kp_dir='./'):
    if kp is None:
        raise ValueError(f'{kp} should not be None, using labelme to get the keypoints what u need on first frame')

    save_kps = []
    if not os.path.exists(save_kp_dir):
        os.makedirs(save_kp_dir, exist_ok=True)

    imgs = [img for img in os.listdir(resources_path) if img.endswith(".{}".format('png'))]
    f = lambda x: float(x.split('.')[0].split('_')[-1])
    imgs = sorted(imgs, key=f)
    print(imgs[:30])
    frame_first = cv2.imread(os.path.join(resources_path, imgs[0]))

    # Take first frame and find corners in it
    old_gray = cv2.cvtColor(frame_first, cv2.COLOR_BGR2GRAY)

    # Parameters for Lucas Kanade optical flow
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )
    # append the first kp
    kp_array = []
    for p in kp:
        kp_array.append(np.array(p).reshape(1, 2))
    p0 = np.array(kp_array, dtype=np.float32)
    save_kps.append(p0.reshape(-1, 2))

    mask = np.zeros_like(frame_first)
    # Create a mask image for drawing purposes
    color = [0, 0, 255]
    img_nums = len(imgs)

    save_randome = np.random.choice(range(img_nums), 10, replace=False)
    print(save_randome)
    # save_randome = random.sample(range(img_nums), 10)
    # change_frame = [750, 800, 1520, 1600, 1800, 2300, 2472, 2700]   # Video frame cut in the middle
    change_frame = None
    for i, img in tqdm(enumerate(imgs[1:])):
        frame_after = cv2.imread(os.path.join(resources_path, img))
        # frame = frame_after.copy()
        if change_frame is not None and i + 1 in change_frame:
            p0 = parser_json(f'./crop_out_512_{i + 1}.json')
            p0 = np.array(p0, dtype=np.float32).reshape(-1, 1, 2)

        frame_gray = cv2.cvtColor(frame_after, cv2.COLOR_BGR2GRAY)

        # Calculate Optical Flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            old_gray, frame_gray, p0, None, **lk_params
        )
        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        if i in save_randome:
            # Draw the tracks
            for j, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                a, b, c, d = list(map(int, [a, b, c, d]))
                mask = cv2.line(mask, (a, b), (c, d), color, 2)
                frame = cv2.circle(frame_after, (a, b), 5, color, -1)

                # Display the demo
                img_draw = cv2.add(frame, mask)
                cv2.imwrite(os.path.join(save_kp_dir, f'save_random_kp_{i}.png'), img_draw)
                # cv2.imshow("frame", img_draw)
                # k = cv2.waitKey(25) & 0xFF
                # if k == 27:
                #     break

        # Update the previous frame and previous points
        old_gray = frame_gray.copy()
        save_kps.append(good_new)
        p0 = good_new.reshape(-1, 1, 2)
    print(len(save_kps))

    npy_save_path = os.path.join(save_kp_dir, 'shoulder_2D.npy')
    # if not os.path.exists(npy_save_path):
    np.save(npy_save_path, save_kps)


def check_should2d(shoulder_npy, imgs_dir):
    imgs = [img for img in os.listdir(imgs_dir) if img.endswith(".{}".format('png'))]
    f = lambda x: float(x.split('.')[0].split('_')[-1])
    imgs = sorted(imgs, key=f)
    shoulder = np.load(shoulder_npy, allow_pickle=True)
    if len(imgs) != shoulder.shape[0]:
        raise ValueError(f'{len(imgs)}, {shoulder.shape[0]}')
    for i, im in enumerate(imgs):
        img = cv2.imread(os.path.join(imgs_dir, im))
        points = shoulder[i].astype(np.int16)
        pint = points
        # pint = list(map(int, points))
        # mask = cv2.line(mask, (a, b), (c, d), (0, 0, 255), 2)
        for p in pint:
            frame = cv2.circle(img, (p[0], p[1]), 5, (0, 0, 255), -1)
        frame = cv2.putText(frame, f'{i}', org=(20, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                            color=(255, 0, 0), thickness=1)
        cv2.imshow("frame", frame)
        k = cv2.waitKey(25) & 0xFF
        if k == 27:
            break


class FAN(object):
    '''
    Get the basic information of the video, format interconversion, fps interconversion, audio extraction, character-centered, video cropping, video cutting, crop, size, etc.
    crop can reference
    https://superuser.com/questions/547296/resizing-videos-with-ffmpeg-avconv-to-fit-into-static-sized-player/1136305#1136305
    ffmpeg -i input -vf "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:-1:-1:color=black" output
    '''

    def __init__(self, video):
        import face_alignment
        self.model = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

        dir = os.path.dirname(video)

        self.video = video
        self.name = os.path.basename(video).split('.')[0]
        self.process_dir = os.path.join(dir, self.name)
        self.frames_dir = os.path.join(self.process_dir, 'frames')

        Path(self.process_dir).mkdir(parents=True, exist_ok=True)
        Path(self.frames_dir).mkdir(parents=True, exist_ok=True)

 
    def detect_img(self, image):
        '''
        image: 0-255, uint8, rgb, [h, w, 3]
        return: detected box list
        '''
        out = self.model.get_landmarks(image)
        if out is None:
            return [0], 'kpt68'
        else:
            kpt = out[0].squeeze()
            left = np.min(kpt[:, 0]);
            right = np.max(kpt[:, 0]);
            top = np.min(kpt[:, 1]);
            bottom = np.max(kpt[:, 1])
            bbox = [left, top, right, bottom]
            return bbox, 'kpt68'
    
    
    def draw_retangle(self, img, bbox):
        l, t, r, b = list(map(int, bbox))
    
        img = cv2.rectangle(img, (l, t), (r, b), (0, 255, 0), cv2.LINE_4, 2)
        cv2.imwrite("random_face.png", img)
    
    
    def detect_video(self):
        '''
        Sampling every 2 seconds, detect all sampled faces in the video, 
        and calculate the minimum outer frame of all faces, so as to determine 
        the boundary of the video that needs to be cropped.
        '''
        cap = cv2.VideoCapture(self.video)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = 0
        bboxes = []
        draw_one = True
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                frame_count += 1
                if frame_count % fps == 2:
                    bbox, _ = self.detect_img(frame)
                    if draw_one:
                        self.draw_retangle(frame.copy(), bbox)
                        draw_one = False
                    bboxes.append(bbox)
            else:
                break
    
        bboxes = np.array(bboxes)
        l, t = np.min(bboxes[:, 0]), np.min(bboxes[:, 1])
        r, b = np.max(bboxes[:, 2]), np.max(bboxes[:, 3])
    
        cap.release()
        # cv2.destroyAllWindows()  # cvDestroyAllWindows
    
        return [l, t, r, b], (width, height)
    
    
    def crop_video_letterbox(self, keep_size=(512, 512), face_head_scale=1.5):
        '''
        In the case that the video itself does not cut the screen, 
        it contains the human body and is cropped to a fixed size. 
        If the width is too large, the entire video will be resized. 
        If the longest width is too small, the cropping result will be affected.
        '''
        video = self.video
        crop_video = self.video
        [l, t, r, b], (w, h) = self.detect_video()
        # here can write more complicate func
        wx_sacled = (r - l) * face_head_scale
        hy_scaled = (b - t) * face_head_scale
    
        scaled_video = os.path.join(self.process_dir, 'half_' + os.path.basename(self.video))
        if max(wx_sacled, hy_scaled) > keep_size[0]:
            cmd = f'ffmpeg -y -i {video} -vf scale=iw/2:ih/2 {scaled_video}'
            subprocess.call([cmd], shell=True)
        if os.path.exists(scaled_video):
            wx_sacled //= 2
            hy_scaled //= 2
            l //= 2
            t //= 2
            crop_video = scaled_video
    
        if max(wx_sacled, hy_scaled) > keep_size[0]:
            raise ValueError('common box in video of face is so large, try to get the data by hand!')
    
        # crop original video by keep size to get the needed video
        pad_x = (keep_size[0] - wx_sacled) // 2
        pad_y = (keep_size[1] - hy_scaled) // 2
        position = (max(l - pad_x, 0), max(0, t - pad_y))  # can add w, h check
        cropped_video = os.path.join(self.process_dir, 'cropped_' + os.path.basename(crop_video))
    
        print(pad_x, pad_y, position)
    
        cmd = f'ffmpeg -y -i {crop_video} -strict -2 -vf crop=w={keep_size[0]}:h={keep_size[1]}:x={position[0]}:y={position[1]} {cropped_video}'
        subprocess.call(cmd, shell=True)
        print('Crop Succeeded!')
    
    
    def video_read_message(self, video=None):
	    
        if video is None:
          video = self.video
    
        if not os.path.exists(video):
            return 0
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            return 0
    
        fileSize = os.path.getsize(video) / (2 ** 20)  # Unit: Mib
        # fileSize = os.path.getsize(path_video) / (10**6)  # Unit: Mb
        fps = int(cap.get(cv2.CAP_PROP_FPS))  # frame rate
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vcodec = int(cap.get(cv2.CAP_PROP_FOURCC))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # total image frames
    
        video_message = {"vcode": vcodec, 'filesize': f"{fileSize} Mib", 'fps': fps, 'w': width, 'h': height,
                         'frame_count': total_frames}
        print(f"Video basic information = {video_message}")
    
        return video_message
    
    
    # ffmpeg.exe -i video/kanghui_5.mp4 -r 60 -acodec aac -vcodec h264 out.mp4
    def change_video_fps(self):
        name = self.name
        video = self.video
        video_path = os.path.join(self.process_dir, name) + '_60fps.mp4'
    
        print(name, video_path)
        # ffmpeg -i ./video/kanghui_5.mp4  -r 60 -acodec aac -vcodec h264 ./video/kanghui5__60fps.mp4
        cmd = f'ffmpeg -y -i {video} -strict -2 -r 60 -acodec aac -vcodec h264 {video_path}'
        subprocess.call([cmd], shell=True)
        
        print("Step: New FPS Video Info")
        self.video_read_message(video_path)
        
        print("Change FPS Succeeded")
    
    def extract_audio(self, sr=16000):
        name = self.name
        video = self.video
        audio = os.path.join(self.process_dir, name) + '.wav'
        no_audio_path = os.path.join(self.process_dir, 'noaudio_' + name) + '.mp4'
    
        cmd = f'ffmpeg -y -i {video} -f wav -ar {sr} {audio}'
        cmd_v = f'ffmpeg -y -i {video} -vcodec copy -an {no_audio_path}'
        subprocess.call([cmd, cmd_v], shell=True)
        # subprocess.call([cmd_v], shell=True)
    
        self.audio = audio
        print("Extract Audio Succeeded")
     
        return audio
    
    
    def extract_imgs(self):
        name = self.name
        video = self.video
        frames_path = self.frames_dir
        frame_path = os.path.join(frames_path, name) + '_%0d.png'
    
        cmd = f'ffmpeg -y -i {video} -strict -2 -filter:v fps=60 {frame_path}'
        subprocess.call([cmd], shell=True)
        print("Extract Images Succeeded")
        bash = f'ls {frames_path} | wc -l'
        subprocess.call([bash], shell=True)
    
    
    def merge_audio_video(self):
        '''
        Combine video and audio
        '''
        name = self.name
        audio = self.audio
        video = self.video
        final_video_path = os.path.join(self.process_dir, name + "_audio") + '.mp4'
        
        cmd = f'ffmpeg -y -i {video} -i {audio} -strict -2 {final_video_path}'
        print("Final video: {final_video_path}")

        subprocess.call([cmd], shell=True)
        print("Merge Audio / Video Succeeded")
    
    
    def clip_video(self):
        '''
        Mainly mouth and video alignment issues
        '''
        # ffmpeg.exe -ss 180 -t 300 -accurate_seek -i kh_fake.mp4  -c copy  -avoid_negative_ts 1 hk_fake_38.mp4
        # ffmpeg -ss 10 -t 15 -i test.mp4 -c:v libx264 -c:a aac -strict experimental -b:a 98k cut.mp4 -y
        pass
    
    
    def video_preprocess(self):
        '''
        handle all
        '''
        print("Step: Crop Video")
        self.crop_video_letterbox(face_head_scale=1.5)
    
        print("Step: Video Info")
        self.video_read_message()
    
        print("Step: Change FPS")
        self.change_video_fps()
    
        print("Step: Extract Audio")
        self.extract_audio()
    
        print("Step: Extract Images")
        self.extract_imgs()
    
        print("Step: Merge Audio / Video")
        self.merge_audio_video()
    
        print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', default='./data/Input/Georges.mp4', help="path to video input")

    ############################### I/O Settings ##############################
    # load config files
    opt = parser.parse_args()
    # video = './hk_fake_38_half.mp4'
    video = opt.video

    # img2video('./imgs', './video/kanghui_5_10.avi')
    # kp = parser_json('./kanghui_5_1.json')
    # kp = parser_json('./kanghui_5_60fps_crop_512_1.json')
    # print(kp)
    # lucas_kanade_method_imgs('./kanghui_imgs_512', kp, save_kp_dir='./kp_save_kh')
    # check_should2d('./kp_save_kh/shoulder_2D.npy', './kanghui_imgs_512')
    # x = np.load('./kp_save_girl/shoulder_2D.npy')
    # print(x.shape)

    fan = FAN(video)
    fan.video_preprocess()
