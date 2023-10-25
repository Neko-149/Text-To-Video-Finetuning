import sys
sys.path.append('/home/yuxueqing/Projects/videodiffusion')
import os
from datetime import datetime
import random
from abc import abstractmethod
import math
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from decord import VideoReader, cpu
import torchvision.transforms._transforms_video as transforms_video
from torch.utils.data.dataset import ConcatDataset
import time
import torchvision
import csv
from einops import rearrange
import json
from tqdm import tqdm
import numpy as np
import libs.utils.transforms as data
import PIL
import imageio
import cv2 as cv

class HDVGDataset(Dataset):
    """
    HDVG Dataset.
    Assumes HDVG data is structured as follows.
    
    first:
    work_path/
            HDVG_local_loader.py
            vila-vg-metas/
                vilavg-part0-vg.jsonl
            video_clips/
                video_id/ 
                    clip/ # HD-VILA-100M 中的 clip id
                        clip_id.mp4 # HDVG-130M 再切分的 clip id

    second:
        vg-clips/
            clip_id.mp4
                        ...
                    ...
                ...
    """
    def __init__(self,
                meta_count=10000,
                data_dir='/tos-bj-dataset/HDVG-130M/vg-clips',
                meta_file='/nas/lijing/HDVG-130M/new_valid_meta/new_hdvg_long_23M.json',
                use_new=True,
                width=512,
                height=320,
                n_sample_frames=8,
                sample_frame_rate=2,
                sample_start_idx=0,
                accelerator=None,
                use_frame_rate=True,
                sample_fps=8,
                ):        
        try:
            host_gpu_num = accelerator.num_processes
            host_num = 1
            all_rank = host_gpu_num * host_num
            global_rank = accelerator.local_process_index
        except:
            all_rank = 1
            global_rank = 1
            pass

        self.meta_file= meta_file
        self.data_dir= data_dir
        self.use_new= use_new
        self.text_name = meta_file
        self.meta_count = meta_count
        self.global_rank = global_rank
        self.all_rank = all_rank
        self.video_length = n_sample_frames
        self.frame_stride = [sample_frame_rate] if isinstance(sample_frame_rate, int) else sample_frame_rate
        self.fps_max = None
        self.use_frame_rate = use_frame_rate
        self.sample_fps = [sample_fps] if isinstance(sample_fps, int) else sample_fps
        if self.use_new:
            self._load_metadata_new()
        else:
            self._load_metadata()
        #x
        
        # for video resize & crop
        self.load_raw_resolution = False
        self.load_resize_keep_ratio = True
        spatial_transform = 'center_crop'
        self.resolution = [height, width]
        if spatial_transform is not None:
            if spatial_transform == "random_crop":
                self.spatial_transform = transforms_video.RandomCropVideo(self.resolution) # original line
                print("random crop not supported")
            elif spatial_transform == "resize_center_crop":
                self.spatial_transform = transforms.Compose([
                    transforms.Resize(height),
                    transforms_video.CenterCropVideo(self.resolution),
                    ])
            elif spatial_transform == "center_crop":
                self.spatial_transform = transforms_video.CenterCropVideo(self.resolution)
            else:
                raise NotImplementedError
        else:
            self.spatial_transform = None
        
        self.vid_trans = data.Compose([
            data.CenterCropWide(size=(448, 448)),
            data.Resize([224, 224]),
            data.ToTensor(),
            data.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])
        
    def _load_metadata_new(self): 
        self.metadata = []
        count=0
        with open(self.meta_file,'rb') as f:
            meta_json=json.load(f)
            for item in meta_json:
                count+=1
                self.metadata.append(item)
                if(count>=self.meta_count):
                    break
            
        print("metadata length:",len(self.metadata))
    
    def _load_metadata(self):
        # videoid - caption 
        last_videoid = ''
        self.metadata = []
        count=-1
        with open(self.meta_file, 'rb') as f:
            meta_json = json.load(f)
        for video_id, meta in tqdm(meta_json.items()):
            count += 1
            if count >= self.meta_count:
                break
            clips = meta['clip']
            for ori_clip, info in clips.items():
                scene_splits = info['scene_split']
                for split in scene_splits:
                    caption = split['caption']
                    self.metadata.append([video_id, split['clip_id']]) 
                    self.metadata[-1].append(caption)
    
    def _get_video_path(self, sample):
        video_id = sample[0]
        clip_id = sample[1]
        if self.use_new:
            video_path = os.path.join(self.data_dir,'%s.mp4' % clip_id)
        else:
            video_path = os.path.join(self.data_dir, video_id, '%s.mp4' % clip_id)

        return video_path
    
    def __getitem__(self, index):
        while True:
            index = index % len(self.metadata)
            sample = self.metadata[index]
            video_path = self._get_video_path(sample)
            if not os.path.isfile(video_path):
                print("video_path not exist!")
                video_path = video_path.replace("/aishi-dataset/HDVG-130M", "/nas/lijing/HDVG-130M/video_clips")
            if not os.path.isfile(video_path): # video 不存在
                index += 1
                continue
            try:
                if self.load_raw_resolution:
                    video_reader = VideoReader(video_path, ctx=cpu(0))
                elif self.load_resize_keep_ratio:
                    h, w, c = VideoReader(video_path, ctx=cpu(0))[0].shape
                    # choose resize scale according to the video h&w ratio and resolution
                    if h/w < self.resolution[0]/self.resolution[1]:
                        scale = h / self.resolution[0]
                    else:
                        scale = w / self.resolution[1]

                    h = math.ceil(h / scale)
                    w = math.ceil(w / scale)
                    video_reader = VideoReader(video_path, ctx=cpu(0), width=w, height=h)
                else:
                    video_reader = VideoReader(video_path, ctx=cpu(0), width=self.resolution[1], height=self.resolution[0])
                if len(video_reader) < self.video_length:
                    print(f"video length ({len(video_reader)}) is smaller than target length({self.video_length})")
                    index += 1
                    continue
                else:
                    pass
            except:
                index += 1
                print(f"Load video failed! path = {video_path}")
                continue
            fps_ori = video_reader.get_avg_fps()
            if self.use_frame_rate:
                fs = random.choice(self.frame_stride)
                cur_sample_fps = float(fps_ori) / fs
            else:
                cur_sample_fps = random.choice(self.sample_fps)
                fs = max(1, round(fps_ori / cur_sample_fps))
            allf = len(video_reader)
            if fs != 1:
                all_frames = list(range(0, len(video_reader), fs))
                if len(all_frames) < self.video_length:
                    # fs = len(video_reader) // self.video_length
                    # assert(fs != 0)
                    # all_frames = list(range(0, len(video_reader), fs))
                    print("frames length less thant video length!")
                    index += 1
                    continue
            else:
                all_frames = list(range(len(video_reader)))
            
            # select a random clip
            rand_idx = random.randint(0, len(all_frames) - self.video_length)
            frame_indices = all_frames[rand_idx:rand_idx+self.video_length]
            try:
                frames = video_reader.get_batch(frame_indices)
                assert(frames.shape[0] == self.video_length),f'{len(frames)}, self.video_length={self.video_length}'
                
                ##########TODO
                #frames = torch.tensor(frames.asnumpy()).permute(3, 0, 1, 2).float() # [t,h,w,c] -> [c,t,h,w]
                frames = torch.tensor(frames).permute(3, 0, 1, 2).float() # [t,h,w,c] -> [c,t,h,w]
                h_ = frames.shape[2]
                #frames = frames[:,:,:int(float(h_*6.0)/7.0),:]
                if self.spatial_transform is not None:
                    """
                    #resize video to approriate shape
                    resize_dim=0 if self.resolution[0]/self.resolution[1]<frames.shape[2]/frames.shape[3] else 1
                    frames=transforms.Resize(self.resolution[resize_dim])(frames)
                    """
                    try:
                        ref_values = frames[:, 0, :, :] #frames[:, random.randint(0, frames.shape[1]-1), :, :]
                        ref_values = PIL.Image.fromarray(ref_values.permute(1,2,0).cpu().numpy().astype(np.uint8))
                        ref_values = self.vid_trans(ref_values)
                        frames = self.spatial_transform(frames)
                    except:
                        print("transform error!")
                        print("frames shape:{}, video name{}".format(frames.shape, video_path))
                        frames = None
                
                if frames is None:
                    index += 1
                    continue
                else:
                    break
            except:
                print(f"Get frames failed! path = {video_path}")
                index += 1
                continue

        assert(frames.shape[2] == self.resolution[0] and frames.shape[3] == self.resolution[1]), f'frames={frames.shape}, self.resolution={self.resolution}'
        frames = frames.byte()
        caption = sample[-1]
        frames = frames.permute(1,0,2,3)
        #ref_values = frames[random.randint(0, frames.shape[0]-1), :, :, :]
        frames = frames / 127.5 - 1
        # example = {'vidioid':video_path, 'pixel_values': frames, 'sentence': caption, 'fs': fs, 'fps_ori': fps_ori, 'fps_sample': cur_sample_fps}
        example = {'vidioid':video_path, 'pixel_values': frames, 'sentence': caption, 'fs': fs, 'fps_ori': fps_ori, 'fps_sample': cur_sample_fps, 'ref_values': ref_values}
        return example
    
    def __len__(self):
        return len(self.metadata)

class HumanSceneDataset(Dataset):
    #*** this dataset only contains video, batch['sentence'] is set to empty string  ***
    def __init__(self,
                meta_count=10000,
                data_dir='/tos-bj-dataset/self-built-dataset/scene_clips/human_scenes',
                meta_file='/nas/lijing/self-built-dataset/metas/human_1.5M.json',
                width=512,
                height=320,
                n_sample_frames=32, 
                sample_frame_rate=2,
                sample_start_idx=0,
                accelerator=None,
                use_frame_rate=True,
                sample_fps=8,
                ):        
        try:
            host_gpu_num = accelerator.num_processes
            host_num = 1
            all_rank = host_gpu_num * host_num
            global_rank = accelerator.local_process_index
        except:
            all_rank = 1
            global_rank = 1
            pass
        self.meta_file= meta_file
        self.data_dir= data_dir
        self.text_name = meta_file
        self.meta_count = meta_count
        self.global_rank = global_rank
        self.all_rank = all_rank
        self.video_length = n_sample_frames
        self.frame_stride = [sample_frame_rate] if isinstance(sample_frame_rate, int) else sample_frame_rate
        self.fps_max = None
        self.use_frame_rate = use_frame_rate
        self.sample_fps = [sample_fps] if isinstance(sample_fps, int) else sample_fps
        self._load_metadata()
    
        
        # for video resize & crop
        self.load_raw_resolution = False
        self.load_resize_keep_ratio = True
        spatial_transform = 'center_crop'
        self.resolution = [height, width]
        if spatial_transform is not None:
            if spatial_transform == "random_crop":
                self.spatial_transform = transforms_video.RandomCropVideo(self.resolution) # original line
                print("random crop not supported")
            elif spatial_transform == "resize_center_crop":
                self.spatial_transform = transforms.Compose([
                    transforms.Resize(height),
                    transforms_video.CenterCropVideo(self.resolution),
                    ])
            elif spatial_transform == "center_crop":
                self.spatial_transform = transforms_video.CenterCropVideo(self.resolution)
            else:
                raise NotImplementedError
        else:
            self.spatial_transform = None
        
        self.vid_trans = data.Compose([
            data.CenterCropWide(size=(448, 448)),
            data.Resize([224, 224]),
            data.ToTensor(),
            data.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])
        
        print("metadata length:",len(self.metadata))
    
    def _load_metadata(self):
        self.metadata = []
        count=-1
        with open(self.meta_file, 'rb') as f:
            meta_json = json.load(f)
        for meta in meta_json:
            count += 1
            if count >= self.meta_count:
                break
            self.metadata.append([meta['clip_id'],meta['path'],meta['duration']])
    
    def _get_video_path(self, sample):
        path=sample[1]
        video_path=os.path.join('/tos-bj-dataset',path)
        return video_path
    
    def __getitem__(self, index):
        while True:
            index = index % len(self.metadata)
            sample = self.metadata[index]
            video_path = self._get_video_path(sample)
            if not os.path.isfile(video_path):
                print(video_path)
                print("video_path not exist!")
            if not os.path.isfile(video_path): # video 不存在
                index += 1
                continue
            try:
                if self.load_raw_resolution:
                    video_reader = VideoReader(video_path, ctx=cpu(0))
                elif self.load_resize_keep_ratio:
                    # resize scale is according to the short side
                    h, w, c = VideoReader(video_path, ctx=cpu(0))[0].shape
                    if h/w < self.resolution[0]/self.resolution[1]:
                        scale = h / self.resolution[0]
                    else:
                        scale = w / self.resolution[1]

                    h = math.ceil(h / scale)
                    w = math.ceil(w / scale)
                    video_reader = VideoReader(video_path, ctx=cpu(0), width=w, height=h)
                else:
                    video_reader = VideoReader(video_path, ctx=cpu(0), width=self.resolution[1], height=self.resolution[0])
                if len(video_reader) < self.video_length:
                    print(f"video length ({len(video_reader)}) is smaller than target length({self.video_length})")
                    index += 1
                    continue
                else:
                    pass
            except:
                index += 1
                # print(f"Load video failed! path = {video_path}")
                continue
            fps_ori = video_reader.get_avg_fps()
            if self.use_frame_rate:
                fs = random.choice(self.frame_stride)
                cur_sample_fps = float(fps_ori) / fs
            else:
                cur_sample_fps = random.choice(self.sample_fps)
                fs = max(1, round(fps_ori / cur_sample_fps))
            allf = len(video_reader)
            if fs != 1:
                all_frames = list(range(0, len(video_reader), fs))
                if len(all_frames) < self.video_length:
                    index += 1
                    continue
            else:
                all_frames = list(range(len(video_reader)))
            
            # select a random clip
            rand_idx = random.randint(0, len(all_frames) - self.video_length)
            frame_indices = all_frames[rand_idx:rand_idx+self.video_length]
            try:
                frames = video_reader.get_batch(frame_indices)
                assert(frames.shape[0] == self.video_length),f'{len(frames)}, self.video_length={self.video_length}'
                #frames = torch.tensor(frames.asnumpy()).permute(3, 0, 1, 2).float() # [t,h,w,c] -> [c,t,h,w]
                frames = torch.tensor(frames).permute(3, 0, 1, 2).float() # [t,h,w,c] -> [c,t,h,w]
                h_ = frames.shape[2]
                #frames = frames[:,:,:int(float(h_*6.0)/7.0),:]
                if self.spatial_transform is not None:
                    try:
                        ref_values = frames[:, 0, :, :] #frames[:, random.randint(0, frames.shape[1]-1), :, :]
                        ref_values = PIL.Image.fromarray(ref_values.permute(1,2,0).cpu().numpy().astype(np.uint8))
                        ref_values = self.vid_trans(ref_values)
                        frames = self.spatial_transform(frames)
                    except:
                        print("frames shape:{}, video name{}".format(frames.shape, video_path))
                        frames = None
                
                if frames is None:
                    index += 1
                    continue
                else:
                    break
            except:
                print(f"Get frames failed! path = {video_path}")
                index += 1
                continue

        assert(frames.shape[2] == self.resolution[0] and frames.shape[3] == self.resolution[1]), f'frames={frames.shape}, self.resolution={self.resolution}'
        frames = frames.byte()
        #caption = sample[-1]
        frames = frames.permute(1,0,2,3)
        #ref_values = frames[random.randint(0, frames.shape[0]-1), :, :, :]
        frames = frames / 127.5 - 1
        # example = {'vidioid':video_path, 'pixel_values': frames, 'sentence': caption, 'fs': fs, 'fps_ori': fps_ori, 'fps_sample': cur_sample_fps}
        example = {'vidioid':video_path, 'pixel_values': frames, 'sentence':"", 'fs': fs, 'fps_ori': fps_ori, 'fps_sample': cur_sample_fps, 'ref_values': ref_values}
        return example
    
    def __len__(self):
        return len(self.metadata)

class WebvidDataset(Dataset):
    """
    Webvid Dataset.
    Assumes webvid data is structured as follows.
    work_path/
            webvid_loader.py
            webvid/
                data/
                    $csv_file/
                        N.csv # 0-49
                train/
                    train_part_N/ # 0-49
                            1.mp4           (videoid.mp4)
                            ...
                            5000.mp4
                    ...
    """
    def __init__(self,
                meta_count=10000,
                part_size=50,
                data_dir='/nas/lijing/TOS_webvid/webvid',
                csv_file='data/results_10M_train_50',
                width=512,
                height=320,
                n_sample_frames=8,
                sample_frame_rate=2,
                sample_fps=8,
                use_frame_rate=True,
                sample_start_idx=0,
                accelerator=None,
                ak="AKLTNjY0M2I1NmZiMDk2NDBhM2EwNjZiMDlkMTJjNmI0MWQ",
                sk="WXpjMU9XWmhaRGs0TlRNMU5HRTBObUl6TldNNU1tWXhPVFU1TkRkbVlUVQ==",
                endpoint="tos-cn-beijing.ivolces.com",
                region="cn-beijing",
                bucket_name="aishi-dataset",
                debug=False,
                ):        
        try:
            host_gpu_num = accelerator.num_processes
            host_num = 1
            all_rank = host_gpu_num * host_num
            global_rank = accelerator.local_process_index
        except:
            all_rank=1
            global_rank=1
            pass
        #print('dataset rank:', global_rank, ' / ',all_rank, ' ')

        self.part_size = part_size
        self.data_dir = data_dir
        self.text_name = csv_file
        self.meta_path = os.path.join(self.data_dir, self.text_name)
        self.client = None#tos.TosClientV2(ak, sk, endpoint, region, max_retry_count=3, max_connections=1024)
        self.bucket_name = bucket_name
        self.meta_count = meta_count
        self.debug = debug
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.water_mark_path = current_dir
        self.resize = transforms.Resize([336, 596])
        
        
        self.resolution = [height, width]
        load_raw_resolution=True

        video_length= n_sample_frames
        fps_max=None
        load_resize_keep_ratio=True
        spatial_transform = 'center_crop'
    
        self.global_rank = global_rank
        self.all_rank = all_rank
        self.video_length = video_length
        
        self.frame_stride = [sample_frame_rate] if isinstance(sample_frame_rate, int) else sample_frame_rate
        self.sample_fps = [sample_fps] if isinstance(sample_fps, int) else sample_fps

        self.load_raw_resolution = load_raw_resolution
        self.fps_max = fps_max
        self.load_resize_keep_ratio = load_resize_keep_ratio
        if self.debug:
            print('start load meta data')
        self._load_metadata()
        if self.debug:
            print('load meta data done!!!')
        if spatial_transform is not None:
            if spatial_transform == "random_crop":
                self.spatial_transform = transforms_video.RandomCropVideo(crop_resolution) # original line
                print("random crop not supported")
            elif spatial_transform == "resize_center_crop":
                self.spatial_transform = transforms.Compose([
                    transforms.Resize(height),
                    transforms_video.CenterCropVideo(self.resolution),
                    ])
            elif spatial_transform == "center_crop":
                self.spatial_transform = transforms_video.CenterCropVideo(self.resolution)
            else:
                raise NotImplementedError
        else:
            self.spatial_transform = None
        
        self.vid_trans = data.Compose([
            data.CenterCropWide(size=(448, 448)),
            data.Resize([224, 224]),
            data.ToTensor(),
            data.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])
        
        self.alpha = np.array(Image.open(os.path.join(self.water_mark_path, 'shutterstock_alpha.png')))
        self.alpha = self.alpha.astype(float) / 255

        self.W = np.array(Image.open(os.path.join(self.water_mark_path, 'shutterstock_W.png')))
        self.bg = self.blur_mask(self.alpha, self.W)
        self.use_frame_rate = use_frame_rate
        

    def move_box(self, alpha, W, offset):
        pos = alpha.nonzero()
        box = [pos[0].min(), pos[1].min(), pos[0].max(), pos[1].max()]
        new_alpha = np.zeros(alpha.shape, dtype=alpha.dtype)
        new_W =  np.zeros(alpha.shape, dtype=W.dtype)
        box_2 = [box[i]-offset[i%2] for i in range(4)]
        idx1 = np.ix_(range(box[0],box[2]), range(box[1], box[3]))
        idx2 = np.ix_(range(box_2[0],box_2[2]), range(box_2[1], box_2[3]))
        new_alpha[idx2] = alpha[idx1]
        new_W[idx2] = W[idx1]
        return new_alpha, new_W

    def blur_mask(self, alpha, W):
        alpha,W = self.move_box(alpha, W, [-4,0])
        pos = alpha.max(axis=-1).nonzero()
        box = [pos[0].min(), pos[1].min(), pos[0].max(), pos[1].max()]

        med = np.array(Image.open(os.path.join(self.water_mark_path, 'shutterstock_difference.png')))

        bg = np.zeros(W.shape)
        bg[box[0]:box[2],box[1]:box[3],:] = med

        return bg

    def find_offset(self, img1, img2):
        img1 = cv.cvtColor(img1.astype(np.uint8), cv.COLOR_BGR2GRAY)
        pos = img1.nonzero()
        box = [pos[0].min(), pos[1].min(), pos[0].max(), pos[1].max()]
        idx1 = np.ix_(range(box[0],box[2]), range(box[1], box[3]))
        img1 = img1[idx1]

        box2 = [box[0]-10, box[1]-10, box[2]+10, box[3]+10]
        idx2 = np.ix_(range(box2[0],box2[2]), range(box2[1], box2[3]))
        img2 = img2[idx2]
        img2 = cv.cvtColor(img2.astype(np.uint8), cv.COLOR_BGR2GRAY)

        img3 = cv.matchTemplate(img2,img1,cv.TM_CCORR_NORMED)
        cent = [img3.shape[0]//2, img3.shape[1]//2]
        img3 = img3[cent[0]-10:cent[0]+10, cent[1]-10:cent[1]+10]
        maxindex = img3.argmax()
        row, col = maxindex//img3.shape[1], maxindex%img3.shape[1]
        assert img3[row,col]==img3.max()

        return [10-row, 10-col]

    '''
    webvid-meta
    videoid,name,page_idx,page_dir,duration,contentUrl
    1053841541,Travel blogger shoot a story on top of mountains.,27732,027701_027750,PT00H00M17S,
    https://ak.picdn.net/shutterstock/videos/1053841541/preview/stock-footage-travel-blogger-shoot
        -a-story-on-top-of-mountains-young-man-holds-camera-in-forest.mp4
    '''
    def _load_metadata(self):
        # videoid - caption 
        last_videoid = ''
        self.metadata = []
        count=-1
        total_count = 8854264 #8856312 - 2048
        for part_id in range(self.part_size):
            start_time = time.time()
            caption_path = os.path.join(self.meta_path,'%d.csv' % part_id)
            with open(caption_path, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    if row['videoid'] != last_videoid:
                        count+=1
                        if count >= (total_count // self.all_rank)*self.all_rank: # drop last
                            break
                        last_videoid = row['videoid']
                        #if count % self.all_rank == self.global_rank:
                        self.metadata.append([int(part_id), row['videoid']]) 
                        self.metadata[-1].append([row['name']])
                    else:
                        #if count % self.all_rank == self.global_rank:
                        self.metadata[-1][-1].append(row['name'])
            end_time = time.time()
            if self.debug:
                print('load part %d, %d - %d items use time: %.1f;' % (part_id, len(self.metadata), count, end_time-start_time))
    
    def _get_video_path(self, sample):
        part_id = int(sample[0])
        videoid = sample[1]
        video_path = os.path.join(self.data_dir,'train/train_part%d' % part_id, '%d.mp4' % int(videoid))
        return video_path
    
    def remove_watermark(self, frames):
        frames = torch.tensor(frames).permute(0, 3, 1, 2)
        frames = self.resize(frames)
        frames = frames.permute(0, 2, 3, 1)
        frames = frames.cpu().numpy()

        for i in range(frames.shape[0]):
            frame = frames[i, :, :, :]
            if i == 0:
                offset = self.find_offset(self.W, frame)
                alpha, W = self.move_box(self.alpha, self.W, offset)
                bg, _ = self.move_box(self.bg, self.bg, offset)

                pos = alpha.nonzero()
                box = [pos[0].min(), pos[1].min(), pos[0].max(), pos[1].max()]
                ROI = np.ix_(range(box[0]-3,box[2]+3), range(box[1]-3,box[3]+3))
                aW = (alpha*W)[ROI]
                a1 = (1/(1 - alpha))[ROI]
                med = bg[ROI]
            frame = frame.astype(float)
            J = frame[ROI]
            I = (J - aW) * a1
            I[I<0] = 0
            I[I>255] = 255
            frame[ROI] = I

            fI = I.copy()
            fI = cv.medianBlur(fI.astype(np.uint8), 5)
            I[med!=0] = fI[med!=0]
            I[I<0] = 0
            I[I>255] = 255
            frame[ROI] = I
            frame = frame.astype(np.uint8)
            frames[i, :, :, :] = frame
        return frames

    def __getitem__(self, index):
        # print("in getitem idx: %d begin" % index)
        begin_time = datetime.now()
        while True:
            # print("in getitem idx: %d begin get video" % index)
            index = index % len(self.metadata)
            sample = self.metadata[index]
            video_path = self._get_video_path(sample)
            # print("in getitem idx: %d begin req TOS" % index)
            # self.get_video_from_tos(video_path)
            # print("in getitem idx: %d got video from TOS" % index)
            if not os.path.isfile(video_path): # video 不存在
                print("video_path not exist!!")
                index += 1
                continue
                
            try:
                if self.load_raw_resolution:
                    video_reader = VideoReader(video_path, ctx=cpu(0))
                elif self.load_resize_keep_ratio:
                    # resize scale is according to the short side
                    print("get videoreader")
                    h, w, c = VideoReader(video_path, ctx=cpu(0))[0].shape
                    print("videoreader done")
                    if h/w < self.resolution[0]/self.resolution[1]:
                        scale = h / self.resolution[0]
                    else:
                        scale = w / self.resolution[1]

                    h = math.ceil(h / scale)
                    w = math.ceil(w / scale)
                    video_reader = VideoReader(video_path, ctx=cpu(0), width=w, height=h)
                else:
                    video_reader = VideoReader(video_path, ctx=cpu(0), width=self.resolution[1], height=self.resolution[0])
                if len(video_reader) < self.video_length:
                    if self.debug:
                        print(f"video length ({len(video_reader)}) is smaller than target length({self.video_length})")
                    index += 1
                    continue
                else:
                    pass
            except:
                index += 1
                if self.debug:
                    print(f"Load video failed! path = {video_path}")
                continue
            fps_ori = video_reader.get_avg_fps()
            # print("in getitem idx: %d video reader init success" % index)
            if self.use_frame_rate:
                fs = random.choice(self.frame_stride)
                cur_sample_fps = float(fps_ori) / fs
            else:
                cur_sample_fps = random.choice(self.sample_fps)
                fs = max(1, round(fps_ori / cur_sample_fps))

            allf = len(video_reader)
            if fs != 1:
                all_frames = list(range(0, len(video_reader), fs))
                if len(all_frames) < self.video_length:
                    continue
            else:
                all_frames = list(range(len(video_reader)))
            
            # select a random clip
            rand_idx = random.randint(0, len(all_frames) - self.video_length)
            frame_indices = all_frames[rand_idx:rand_idx+self.video_length]
            try:
                frames = video_reader.get_batch(frame_indices)
                break
            except:
                if self.debug:
                    print(f"Get frames failed! path = {video_path}")
                index += 1
                continue
        # print("in getitem idx: %d begin cut frames " % index)
        start_time = datetime.now()
        assert(frames.shape[0] == self.video_length),f'{len(frames)}, self.video_length={self.video_length}'
        #frames = self.remove_watermark(frames.asnumpy())
        frames = self.remove_watermark(frames)
        frames = torch.tensor(frames).permute(3, 0, 1, 2).float() # [t,h,w,c] -> [c,t,h,w]
       
        if self.spatial_transform is not None:
            ref_values = frames[:, 0, :, :] #frames[:, random.randint(0, frames.shape[1]-1), :, :]
            ref_values = PIL.Image.fromarray(ref_values.permute(1,2,0).cpu().numpy().astype(np.uint8))
            ref_values = self.vid_trans(ref_values)
            frames = self.spatial_transform(frames)
        assert(frames.shape[2] == self.resolution[0] and frames.shape[3] == self.resolution[1]), f'frames={frames.shape}, self.resolution={self.resolution}'
        frames = frames.byte()
        #end_time = datetime.now()
        #time_cost = (end_time - start_time)
        caption = sample[-1][0]

        frames = frames.permute(1,0,2,3) #[c,t,h,w]->t,c,h,w
        #ref_values = frames[random.randint(0, frames.shape[0]-1), :, :, :]
        frames = frames / 127.5 - 1.0
        example = {'vidioid':video_path, 'pixel_values': frames, 'sentence': caption, 'fs': fs, 'fps_ori': fps_ori, 'fps_sample': cur_sample_fps, 'ref_values': ref_values}
        # print("in getitem idx: %d done" % index)
        #end_time = datetime.now()
        #time_cost = (end_time - begin_time)
        # print("in getitem idx: %d cost %s ms " % (index, (time_cost.microseconds / 1000)))
        return example
    
    def __len__(self):
        return len(self.metadata)

class Webvid_HDVG_Human(Dataset):
    def __init__(self,
                HDVG_meta_count=10000,
                HDVG_data_dir=None,
                HDVG_meta_file=None,
                HDVG_use_new=False,
                webvid_meta_count=10000,
                webvid_data_dir=None,
                webvid_csv_file=None,
                width=512,
                height=320,
                n_sample_frames=16,
                sample_frame_rate=2,
                sample_fps=8,
                sample_start_idx=0,
                accelerator=None,
                #need_count=100,
                human_meta_count=10000,
                human_data_dir=None, 
                human_meta_file=None, 
                transform=None,
                webvid_part_size=50,
                use_frame_rate=False,
                ):
        dataset1 = None
        dataset2 = None
        dataset3 = None
        try:
            if webvid_csv_file is not None and webvid_data_dir is not None:
                dataset1 = WebvidDataset(
                    meta_count=webvid_meta_count,
                    part_size=webvid_part_size,
                    data_dir=webvid_data_dir,
                    csv_file=webvid_csv_file,
                    width=width,
                    height=height,
                    n_sample_frames=n_sample_frames,
                    sample_frame_rate=sample_frame_rate,
                    sample_fps=sample_fps,
                    use_frame_rate=use_frame_rate,
                    sample_start_idx=0,
                    accelerator=accelerator,
                    ak="AKLTNjY0M2I1NmZiMDk2NDBhM2EwNjZiMDlkMTJjNmI0MWQ",
                    sk="WXpjMU9XWmhaRGs0TlRNMU5HRTBObUl6TldNNU1tWXhPVFU1TkRkbVlUVQ==",
                    endpoint="tos-cn-beijing.ivolces.com",
                    region="cn-beijing",
                    bucket_name="aishi-dataset",
                    #need_count=need_count,
                    debug=False,
                )
                
                print("webvid initialization success!")
        except ValueError:
            print("webvid initialization error!")
        
        try:
            if HDVG_data_dir is not None and HDVG_meta_file is not None:
                dataset2 = HDVGDataset(
                    meta_count=HDVG_meta_count,
                    data_dir=HDVG_data_dir,
                    meta_file=HDVG_meta_file,
                    use_new=HDVG_use_new,
                    width=width,
                    height=height,
                    n_sample_frames=n_sample_frames,
                    sample_frame_rate=sample_frame_rate,
                    sample_start_idx=0,
                    accelerator=accelerator,
                    #need_count=need_count,
                    use_frame_rate=use_frame_rate,
                    sample_fps=sample_fps
                )
        except ValueError:
            print("HDVG initialization error!")
        
        try:
            if human_data_dir is not None and human_meta_file is not None:
                dataset3 = HumanSceneDataset(
                                meta_count=human_meta_count,
                                data_dir=human_data_dir,
                                meta_file=human_meta_file,
                                width=width,
                                height=height,
                                n_sample_frames=n_sample_frames, 
                                sample_frame_rate=sample_frame_rate,
                                sample_start_idx=0,
                                accelerator=accelerator,
                                #need_count=need_count,
                                use_frame_rate=use_frame_rate,
                                sample_fps=sample_fps,
                            )
        except ValueError:
            print("HumanScene initialization error!")
        
        dataset = []
        if dataset1 is not None:
            dataset.append(dataset1)
        if dataset2 is not None:
            dataset.append(dataset2)    
        if dataset3 is not None:
            dataset.append(dataset3)
        self.concat_data = ConcatDataset(dataset)
        print('concat_data: ', len(self.concat_data))
        
    def __len__(self):
        return len(self.concat_data)
    

    def __getitem__(self, index):
        return self.concat_data[index]
       #return self.concat_data.__getitem__(index)
        

if __name__ == "__main__":
    from accelerate import Accelerator
    from accelerate.logging import get_logger
    from accelerate.utils import set_seed

    gradient_accumulation_steps = 1

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision='no',
    )
    #batch_size = 3
    #root_dir1 = None
    #json_dir1 = '/nas/lijing/laion2B-en-human/url_metas/'
    def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=4, fps=8):
        videos = rearrange(videos, "b c t h w -> t b c h w")
        #print("videos shape",videos.shape)
        outputs = []
        for x in videos:
            x = torchvision.utils.make_grid(x, nrow=n_rows)
            x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
            if rescale:
                x = (x + 1.0) / 2.0  # -1,1 -> 0,1
            x = (x * 255).numpy().astype(np.uint8)
            outputs.append(x)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        #kwargs_write = {'fps':5.0, 'quantizer':'nq'}
        #imageio.mimsave(path, outputs,'GIF-FI', **kwargs_write)
        imageio.mimsave(path, outputs,quality=10)

    concat_data = Webvid_HDVG_Human(HDVG_meta_count=10000000000, 
                                webvid_part_size=50,
                                human_meta_count=10000000, 
                                accelerator=accelerator,
                                width=512,
                                height=320,
                                HDVG_data_dir='/tos-bj-dataset/HDVG-130M/vg-clips',
                                HDVG_meta_file='/nas/lijing/HDVG-130M/new_valid_meta/new_hdvg_long_23M.json',
                                HDVG_use_new=True,
                                webvid_data_dir='/aishi-dataset/webvid',
                                webvid_csv_file='data/results_10M_train_50',
                                human_data_dir='/tos-bj-dataset/self-built-dataset/scene_clips/human_scenes', 
                                human_meta_file='/nas/lijing/self-built-dataset/metas/human_1.5M.json',)
    train_dataloader = torch.utils.data.DataLoader(concat_data, batch_size=1, num_workers=5, shuffle = True)
    
    for step, batch in enumerate(train_dataloader):
        #pixel_values = batch["pixel_values"]
        pixel_values = batch["pixel_values"].permute(0,2,1,3,4).cpu()
        #print(pixel_values.shape)
        prompt = batch["sentence"]
        #save_videos_grid(pixel_values, "/home/yuxueqing/Projects/test_output/mix_data_test2/{}_{}.mp4".format(step,prompt[0:15]), rescale=True)
        print(prompt)
