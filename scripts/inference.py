import os
import traceback
from pathlib import Path
from argparse import Namespace
import copy
import sys
from tqdm import tqdm

import numpy as np
import cv2
import torch
import glob
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from musetalk.utils.utils import get_file_type, get_video_fps, datagen
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder, get_bbox_range
from musetalk.utils.blending import get_image
from musetalk.utils.utils import load_all_model
import shutil
from scripts import states
from configs import config


class Process:

    def __init__(self, input_path: str, audio_path: str, *, options: dict):
        self.inputPath, self.audioPath = Path(input_path), Path(audio_path)
        self.options = Namespace(**options)

        # load model weights
        self.audio_processor, self.vae, self.unet, self.pe = load_all_model()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.timesteps = torch.tensor([0], device=device)
        if config.FLOAT16 is True:
            self.pe = self.pe.half()
            self.vae.vae = self.vae.vae.half()
            self.unet.model = self.unet.model.half()

        self.state = states.State.PENDING

        self.basename = f'{self.inputPath.stem}-{self.audioPath.stem}'
        self.output_vid_path = os.path.join(config.OUTPUT_DIR, f'{self.basename}.mp4')
        self.frame_path_dir = os.path.join(config.TEMP_DIR, f'frame-{self.basename}')  # 存储原始视频帧
        self.crop_frame_path_dir = os.path.join(config.TEMP_DIR, f'crop-{self.basename}')  # 存储结果视频帧
        self.crop_coord_path = os.path.join(config.INPUT_DIR, self.inputPath.stem + ".pkl")  # 原始视频坐标文件
        os.makedirs(self.frame_path_dir, exist_ok=True)
        os.makedirs(self.crop_frame_path_dir, exist_ok=True)

    def __extract(self, input_images, fps):
        # ------------------------- extract audio feature -------------------------
        whisper_feature = self.audio_processor.audio2feat(str(self.audioPath))
        whisper_chunks = self.audio_processor.feature2chunks(feature_array=whisper_feature, fps=fps)

        # ------------------------- preprocess input image -------------------------
        print(f'start preprocess input image: {str(self.inputPath)}')
        self.state = states.State.PREPROCESS_INPUT_IMG
        if os.path.exists(self.crop_coord_path) and self.options.use_saved_coord:
            print("using extracted coordinates")
            with open(self.crop_coord_path, 'rb') as f:
                coord_list = pickle.load(f)
            frame_list = read_imgs(input_images)
        else:
            print("extracting landmarks...time consuming")
            coord_list, frame_list = get_landmark_and_bbox(input_images, self.options.bbox_shift)
            with open(self.crop_coord_path, 'wb') as f:
                pickle.dump(coord_list, f)

        bbox_shift_text = get_bbox_range(input_images, self.options.bbox_shift)
        print(bbox_shift_text)

        print(f'start get latents for unet: {str(self.inputPath)}')
        self.state = states.State.CROP_IMG
        input_latent_list = []
        for bbox, frame in zip(coord_list, frame_list):
            if bbox == coord_placeholder:
                continue
            x1, y1, x2, y2 = bbox
            crop_frame = frame[y1:y2, x1:x2]
            crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            latents = self.vae.get_latents_for_unet(crop_frame)
            input_latent_list.append(latents)

        # to smooth the first and the last frame
        frame_list_cycle = frame_list + frame_list[::-1]
        coord_list_cycle = coord_list + coord_list[::-1]
        input_latent_list_cycle = input_latent_list + input_latent_list[::-1]

        # ------------------------- inference batch by batch -------------------------
        print(f'start inference: {str(self.inputPath)}')
        self.state = states.State.INFERENCE
        video_num = len(whisper_chunks)
        batch_size = self.options.batch_size
        gen = datagen(whisper_chunks, input_latent_list_cycle, batch_size)
        res_frame_list = []
        for i, (whisper_batch, latent_batch) in enumerate(tqdm(gen, total=int(np.ceil(float(video_num) / batch_size)))):
            audio_feature_batch = torch.from_numpy(whisper_batch)
            audio_feature_batch = audio_feature_batch.to(device=self.unet.device,
                                                         dtype=self.unet.model.dtype)  # torch, B, 5*N,384
            audio_feature_batch = self.pe(audio_feature_batch)
            latent_batch = latent_batch.to(dtype=self.unet.model.dtype)

            pred_latents = self.unet.model(latent_batch, self.timesteps,
                                           encoder_hidden_states=audio_feature_batch).sample
            recon = self.vae.decode_latents(pred_latents)
            for res_frame in recon:
                res_frame_list.append(res_frame)

        # ------------------------- pad to full image -------------------------
        print("pad talking image to original video")
        self.state = states.State.PAD_IMG
        for i, res_frame in enumerate(tqdm(res_frame_list)):
            bbox = coord_list_cycle[i % (len(coord_list_cycle))]
            ori_frame = copy.deepcopy(frame_list_cycle[i % (len(frame_list_cycle))])
            x1, y1, x2, y2 = bbox
            try:
                res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
            except:
                continue

            combine_frame = get_image(ori_frame, res_frame, bbox)
            cv2.imwrite(f"{self.crop_frame_path_dir}/{str(i).zfill(8)}.png", combine_frame)

        # ------------------------- gen video -------------------------
        print("gen video")
        self.state = states.State.GEN_VIDEO
        temp_video_path = os.path.join(config.TEMP_DIR, f'temp-{self.basename}.mp4')
        cmd_img2video = f"ffmpeg -y -v warning -r {fps} -f image2 -i {self.crop_frame_path_dir}/%08d.png -vcodec libx264 -vf format=rgb24,scale=out_color_matrix=bt709,format=yuv420p -crf 18 {temp_video_path}"
        print(cmd_img2video)
        os.system(cmd_img2video)

        cmd_combine_audio = f"ffmpeg -y -v warning -i {str(self.audioPath)} -i {temp_video_path} {self.output_vid_path}"
        print(cmd_combine_audio)
        os.system(cmd_combine_audio)

        os.remove(temp_video_path)
        shutil.rmtree(self.frame_path_dir)  # 移除视频原始帧
        shutil.rmtree(self.crop_frame_path_dir)  # 移除视频结果帧
        print(f"result is save to {self.output_vid_path}")

    @torch.no_grad()
    def run(self):
        try:
            if get_file_type(self.inputPath) == "video":
                cmd = f"ffmpeg -v fatal -i {str(self.inputPath)} -start_number 0 {self.frame_path_dir}/%08d.png"
                os.system(cmd)
                input_images = sorted(glob.glob(os.path.join(self.frame_path_dir, '*.[jpJP][pnPN]*[gG]')))
                fps = get_video_fps(str(self.inputPath))
            elif get_file_type(self.inputPath) == "image":
                input_images = [str(self.inputPath), ]
                fps = self.options.fps
            elif os.path.isdir(self.inputPath):  # input img folder
                input_images = glob.glob(os.path.join(str(self.inputPath), '*.[jpJP][pnPN]*[gG]'))
                input_images = sorted(input_images, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                fps = self.options.fps
            else:
                raise ValueError(f"{self.inputPath} should be a video file, an image file or a directory of images")

            self.__extract(input_images, fps)
            self.state = states.State.END
        except Exception as e:
            print(traceback.format_exception(e))
            self.state = states.State.ERROR


if __name__ == "__main__":
    video = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/video/sun.mp4'))
    audio = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/audio/sun.wav'))
    kwargs = {
        'use_saved_coord': False,
        'batch_size': 8,
        'fps': 25,
    }
    process = Process(video, audio, options=kwargs)
    process.run()
    print(f'结果视频：{process.output_vid_path}')
