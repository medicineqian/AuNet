# Dataset utils and dataloaders

import glob
import logging
import math
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from threading import Thread
from torchvision.utils import save_image

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ExifTags,ImageDraw
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.general import check_requirements, xyxy2xywh, xywh2xyxy, xywhn2xyxy, xyn2xy, segment2box, segments2boxes, \
    resample_segments, clean_str
from utils.torch_utils import torch_distributed_zero_first


import global_var
EPSILON = 1e-5

# Parameters
help_url = 'https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'
img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes
logger = logging.getLogger(__name__)


class RandomSampler(torch.utils.data.sampler.RandomSampler):

    def __init__(self, data_source, replacement=False, num_samples=None):
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples

        if not isinstance(self.replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(self.replacement))

        if self._num_samples is not None and not replacement:
            raise ValueError("With replacement=False, num_samples should not be specified, "
                             "since a random permute will be performed.")

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))

    @property
    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        n = len(self.data_source)
        if self.replacement:
            return iter(torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64).tolist())
        # print("-------------------------")
        s = global_var.get_value('s')
        return iter(s)

    def __len__(self):
        return self.num_samples


# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


def get_hash(files):
    # Returns a single hash value of a list of files
    return sum(os.path.getsize(f) for f in files if os.path.isfile(f))


def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except:
        pass

    return s

def create_dataloader(path, imgsz, batch_size, stride, opt, hyp=None, augment=False, cache=False, pad=0.0, rect=False,
                      rank=-1, world_size=1, workers=8, image_weights=False, quad=False, prefix='', sampler=None):
    # Make sure only the first process in DDP process the dataset first, and the following others can use the cache
    with torch_distributed_zero_first(rank):
        dataset = LoadImagesAndLabels(path, imgsz, batch_size,
                                      augment=augment,  # augment images
                                      hyp=hyp,  # augmentation hyperparameters
                                      rect=rect,  # rectangular training
                                      cache_images=cache,
                                      single_cls=opt.single_cls,
                                      stride=int(stride),
                                      pad=pad,
                                      image_weights=image_weights,
                                      prefix=prefix)

    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count() // world_size, batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if rank != -1 else None
    loader = torch.utils.data.DataLoader if image_weights else InfiniteDataLoader


    # global_var.set_value('s', torch.randperm(len(dataset)).tolist())
    # sampler = RandomSampler(dataset)

    # sampler = torch.utils.data.sampler.RandomSampler(dataset)

    # Use torch.utils.data.DataLoader() if dataset.properties will update during training else InfiniteDataLoader()
    dataloader = loader(dataset,
                        batch_size=batch_size,
                        num_workers=nw,
                        sampler=sampler,
                        pin_memory=True,
                        collate_fn=LoadImagesAndLabels.collate_fn4 if quad else LoadImagesAndLabels.collate_fn)

    return dataloader, dataset


def create_dataloader_rgb_ir(path1, path2,path3, imgsz, batch_size, stride, opt, hyp=None, augment=False, cache=False, pad=0.0, rect=False,
                             rank=-1, world_size=1, workers=8, image_weights=False, quad=False, prefix='', sampler=None):
    # Make sure only the first process in DDP process the dataset first, and the following others can use the cache
    patch = opt.patch
    with torch_distributed_zero_first(rank):
        dataset = LoadMultiModalImagesAndLabels(path1, path2,path3, imgsz, batch_size,
                                                augment=augment,  # augment images
                                                hyp=hyp,  # augmentation hyperparameters
                                                rect=rect,  # rectangular training
                                                patch = patch,
                                                cache_images=cache,
                                                single_cls=opt.single_cls,
                                                stride=int(stride),
                                                pad=pad,
                                                image_weights=image_weights,
                                                prefix=prefix)

    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count() // world_size, batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if rank != -1 else None
    loader = torch.utils.data.DataLoader if image_weights else InfiniteDataLoader

    # global_var.set_value('s', torch.randperm(len(dataset)).tolist())
    # sampler = RandomSampler(dataset)

    # sampler = torch.utils.data.sampler.RandomSampler(dataset)

    # Use torch.utils.data.DataLoader() if dataset.properties will update during training else InfiniteDataLoader()
    dataloader = loader(dataset,
                        batch_size=batch_size,
                        num_workers=nw,
                        sampler=sampler,
                        pin_memory=True,
                        collate_fn=LoadImagesAndLabels.collate_fn4 if quad else LoadImagesAndLabels.collate_fn)

    return dataloader, dataset


class InfiniteDataLoader(torch.utils.data.dataloader.DataLoader):
    """ Dataloader that reuses workers

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class LoadImages:  # for inference
    def __init__(self, path, img_size=640, stride=32):
        p = str(Path(path).absolute())  # os-agnostic absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in img_formats]
        videos = [x for x in files if x.split('.')[-1].lower() in vid_formats]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {img_formats}\nvideos: {vid_formats}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            print(f'video {self.count + 1}/{self.nf} ({self.frame}/{self.nframes}) {path}: ', end='')

        else:
            # Read image
            self.count += 1
            img0 = cv2.imread(path)  # BGR
            assert img0 is not None, 'Image Not Found ' + path
            print(f'image {self.count}/{self.nf} {path}: ', end='')

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return path, img, img0, self.cap

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files


class LoadWebcam:  # for inference
    def __init__(self, pipe='0', img_size=640, stride=32):
        self.img_size = img_size
        self.stride = stride

        if pipe.isnumeric():
            pipe = eval(pipe)  # local camera
        # pipe = 'rtsp://192.168.1.64/1'  # IP camera
        # pipe = 'rtsp://username:password@192.168.1.64/1'  # IP camera with login
        # pipe = 'http://wmccpinetop.axiscam.net/mjpg/video.mjpg'  # IP golf camera

        self.pipe = pipe
        self.cap = cv2.VideoCapture(pipe)  # video capture object
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # set buffer size

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if cv2.waitKey(1) == ord('q'):  # q to quit
            self.cap.release()
            cv2.destroyAllWindows()
            raise StopIteration

        # Read frame
        if self.pipe == 0:  # local camera
            ret_val, img0 = self.cap.read()
            img0 = cv2.flip(img0, 1)  # flip left-right
        else:  # IP camera
            n = 0
            while True:
                n += 1
                self.cap.grab()
                if n % 30 == 0:  # skip frames
                    ret_val, img0 = self.cap.retrieve()
                    if ret_val:
                        break

        # Print
        assert ret_val, f'Camera Error {self.pipe}'
        img_path = 'webcam.jpg'
        print(f'webcam {self.count}: ', end='')

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return img_path, img, img0, None

    def __len__(self):
        return 0


class LoadStreams:  # multiple IP or RTSP cameras
    def __init__(self, sources='streams.txt', img_size=640, stride=32):
        self.mode = 'stream'
        self.img_size = img_size
        self.stride = stride

        if os.path.isfile(sources):
            with open(sources, 'r') as f:
                sources = [x.strip() for x in f.read().strip().splitlines() if len(x.strip())]
        else:
            sources = [sources]

        n = len(sources)
        self.imgs = [None] * n
        self.sources = [clean_str(x) for x in sources]  # clean source names for later
        for i, s in enumerate(sources):  # index, source
            # Start thread to read frames from video stream
            print(f'{i + 1}/{n}: {s}... ', end='')
            if 'youtube.com/' in s or 'youtu.be/' in s:  # if source is YouTube video
                check_requirements(('pafy', 'youtube_dl'))
                import pafy
                s = pafy.new(s).getbest(preftype="mp4").url  # YouTube URL
            s = eval(s) if s.isnumeric() else s  # i.e. s = '0' local webcam
            cap = cv2.VideoCapture(s)
            assert cap.isOpened(), f'Failed to open {s}'
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = cap.get(cv2.CAP_PROP_FPS) % 100

            _, self.imgs[i] = cap.read()  # guarantee first frame
            thread = Thread(target=self.update, args=([i, cap]), daemon=True)
            print(f' success ({w}x{h} at {self.fps:.2f} FPS).')
            thread.start()
        print('')  # newline

        # check for common shapes
        s = np.stack([letterbox(x, self.img_size, stride=self.stride)[0].shape for x in self.imgs], 0)  # shapes
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        if not self.rect:
            print('WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams.')

    def update(self, index, cap):
        # Read next stream frame in a daemon thread
        n = 0
        while cap.isOpened():
            n += 1
            # _, self.imgs[index] = cap.read()
            cap.grab()
            if n == 4:  # read every 4th frame
                success, im = cap.retrieve()
                self.imgs[index] = im if success else self.imgs[index] * 0
                n = 0
            time.sleep(1 / self.fps)  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        img0 = self.imgs.copy()
        if cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # Letterbox
        img = [letterbox(x, self.img_size, auto=self.rect, stride=self.stride)[0] for x in img0]

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
        img = np.ascontiguousarray(img)

        return self.sources, img, img0, None

    def __len__(self):
        return 0  # 1E12 frames = 32 streams at 30 FPS for 30 years

def create_mask(img_paths, img_height, img_width):
    bs = len(img_paths)
    
    mask = torch.zeros((bs, img_height, img_width), device='cuda')
    for j,img_path in enumerate(img_paths):
        bboxes = []
        label_path = img2label_path(img_path)  
        with open(label_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                class_id, x_c, y_c, w, h = map(float, parts)
                # 转换坐标
                x_c = x_c * 640
                y_c = y_c * 480
                w = w * 640
                h = h * 480
                x_min = int(x_c - w / 2)
                y_min = int(y_c - h / 2)
                x_max = int(x_c + w / 2)
                y_max = int(y_c + h / 2)

                bbox = [x_min, y_min, x_max, y_max]
                bboxes.append(bbox)
        # 得到bboxes之后，就遍历bboxes
        n = len(bboxes)
        mask_ = torch.zeros((n, img_height, img_width), device='cuda')
        # print("bboxes的形状：",np.array(bboxes).shape,bboxes)
        for i, batch_bboxes in enumerate(bboxes):
            m_= torch.zeros((img_height, img_width), device='cuda')
            # print(batch_bboxes)
            xmin, ymin, xmax, ymax = batch_bboxes[0], batch_bboxes[1], batch_bboxes[2], batch_bboxes[3]
            a = (xmax - xmin)/2
            b = (ymax - ymin)/2
            cx = (xmax + xmin)/2
            cy = (ymax + ymin)/2
            x = torch.tensor(np.arange(0, img_width, 1), device='cuda:0')
            y = torch.tensor(np.arange(0, img_height, 1), device='cuda:0')
            xx, yy = torch.meshgrid(x, y,indexing='xy')
            # binary mask
            m_ = (xx-cx)**2/a**2 + (yy-cy)**2/b**2 < 1+EPSILON
            # print("!!!!!!!!!!!",a,b)
            m_.to(dtype=torch.float32)
            mask_[i, :, :] = m_
            # print("mask_的形状",mask_.shape,'m_的形状',m_.shape)
        mask[j, :, :] = mask_.sum(0).clamp_(min=0, max=1)
        # mask[j, :, :] = (mask_[i, :, :] > 0).float().clamp_(min=0, max=1)
        # mask_path = img_path.replace('VIS','Mask').replace('.jpg', '_mask.png').replace('.png', '_mask.png')  # 根据需要修改扩展名
        # save_image(mask[j], mask_path)
        # print(f"Mask saved to {mask_path}")
    # print("mask的形状",mask.shape)
    return mask            


def mask_0_1(img_paths,is_train=False):
    original_img_size = (640, 480)
    if is_train:
        resized_img_size = (160, 160)
    else:
        resized_img_size = (168, 128)
    tensors_list = []  # 创建一个空列表来收集所有的张量

    for img_path in img_paths:
        label_path = img2label_path(img_path)  # 假设你已经有这个函数定义了

        label_img = Image.new('L', original_img_size, 0)
        draw = ImageDraw.Draw(label_img)
        with open(label_path, 'r') as file:
            for line in file:
                # 解析标签
                parts = line.strip().split()
                class_id, x_c, y_c, w, h = map(float, parts)

                # 转换坐标
                x_c = x_c * original_img_size[0]
                y_c = y_c * original_img_size[1]
                w = w * original_img_size[0]
                h = h * original_img_size[1]
                x_min = int(x_c - w / 2)
                y_min = int(y_c - h / 2)
                x_max = int(x_c + w / 2)
                y_max = int(y_c + h / 2)
                draw.rectangle([x_min, y_min, x_max, y_max], outline=1, fill=1)

        # 调整大小并转换为张量
        label_img_resized = label_img.resize(resized_img_size, Image.NEAREST)
        label_img_np = np.array(label_img_resized)
        label_img_tensor = torch.from_numpy(label_img_np).unsqueeze(0)  # 添加一个维度
        tensors_list.append(label_img_tensor)

    # 假设 tensors_list 现在包含 8 个张量
    # 使用 stack 将列表中的所有张量堆叠起来
    tensors_stack = torch.stack(tensors_list, dim=0)

    # 如果需要的话，添加一个通道维度
    # tensors_stack = tensors_stack.unsqueeze(1)

    return tensors_stack


def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sb, t = 'label', []  # /images/, /labels/
    for x in img_paths:
        if 'VIS' in x.split('/'):
            sa = 'VIS'
        elif 'TIR' in x.split('/'):
            sa = 'TIR'
        elif 'NIR' in x.split('/'):
            sa = 'NIR'
        t.append('txt'.join(x.replace(sa, sb, 1).rsplit(x.split('.')[-1], 1)))
    # for x in img_paths:
    #     if 'rgb' in x.split('/'):
    #         sa = 'rgb'
    #     elif 'finf' in x.split('/'):
    #         sa = 'finf'
    #     elif 'ninf' in x.split('/'):
    #         sa = 'ninf'
    #     t.append('txt'.join(x.replace(sa, sb, 1).rsplit(x.split('.')[-1], 1)))
    return t

def img2label_path(x):
    # Define label paths as a function of image paths
    sb, t = 'label', []  # /images/, /labels/
    if 'VIS' in x.split('/'):
        sa = 'VIS'
    elif 'TIR' in x.split('/'):
        sa = 'TIR'
    elif 'NIR' in x.split('/'):
        sa = 'NIR'
    t = 'txt'.join(x.replace(sa, sb, 1).rsplit(x.split('.')[-1], 1))
    return t

class LoadImagesAndLabels(Dataset):  # for training/testing
    def __init__(self, path, img_size=640, batch_size=16, augment=False, hyp=None, rect=False, image_weights=False,
                 cache_images=False, single_cls=False, stride=32, pad=0.0, prefix=''):
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        self.path = path

        try:
            f = []  # image files
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                    # f = list(p.rglob('**/*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p, 'r') as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                        # f += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise Exception(f'{prefix}{p} does not exist')
            self.img_files = sorted([x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in img_formats])
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in img_formats])  # pathlib
            assert self.img_files, f'{prefix}No images found'
        except Exception as e:
            raise Exception(f'{prefix}Error loading data from {path}: {e}\nSee {help_url}')

        # Check cache
        self.label_files = img2label_paths(self.img_files)  # labels
        # print(self.label_files)
        cache_path = (p if p.is_file() else Path(self.label_files[0]).parent).with_suffix('.cache')  # cached labels
        if cache_path.is_file():
            cache, exists = torch.load(cache_path), True  # load
            if cache['hash'] != get_hash(self.label_files + self.img_files) or 'version' not in cache:  # changed
                cache, exists = self.cache_labels(cache_path, prefix), False  # re-cache
        else:
            cache, exists = self.cache_labels(cache_path, prefix), False  # cache

        # Display cache
        nf, nm, ne, nc, n = cache.pop('results')  # found, missing, empty, corrupted, total
        if exists:
            d = f"Scanning '{cache_path}' images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupted"
            tqdm(None, desc=prefix + d, total=n, initial=n)  # display cache results
        assert nf > 0 or not augment, f'{prefix}No labels in {cache_path}. Can not train without labels. See {help_url}'

        # Read cache
        cache.pop('hash')  # remove hash
        cache.pop('version')  # remove version
        labels, shapes, self.segments = zip(*cache.values())
        self.labels = list(labels)
        self.shapes = np.array(shapes, dtype=np.float64)
        self.img_files = list(cache.keys())  # update
        self.label_files = img2label_paths(cache.keys())  # update
        if single_cls:
            for x in self.labels:
                x[:, 0] = 0

        n = len(shapes)  # number of images
        bi = np.floor(np.arange(n) / batch_size).astype(np.int32)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.n = n
        self.indices = range(n)

        # Rectangular Training
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.img_files = [self.img_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(np.int32) * stride

        # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
        self.imgs = [None] * n
        if cache_images:
            gb = 0  # Gigabytes of cached images
            self.img_hw0, self.img_hw = [None] * n, [None] * n
            results = ThreadPool(8).imap(lambda x: load_image(*x), zip(repeat(self), range(n)))  # 8 threads
            pbar = tqdm(enumerate(results), total=n)
            for i, x in pbar:
                self.imgs[i], self.img_hw0[i], self.img_hw[i] = x  # img, hw_original, hw_resized = load_image(self, i)
                gb += self.imgs[i].nbytes
                pbar.desc = f'{prefix}Caching images ({gb / 1E9:.1f}GB)'
            pbar.close()

    def cache_labels(self, path=Path('./labels.cache'), prefix=''):
        # Cache dataset labels, check images and read shapes
        x = {}  # dict
        nm, nf, ne, nc = 0, 0, 0, 0  # number missing, found, empty, duplicate
        pbar = tqdm(zip(self.img_files, self.label_files), desc='Scanning images', total=len(self.img_files))
        for i, (im_file, lb_file) in enumerate(pbar):
            try:
                # verify images
                im = Image.open(im_file)
                im.verify()  # PIL verify
                shape = exif_size(im)  # image size
                segments = []  # instance segments
                assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
                assert im.format.lower() in img_formats, f'invalid image format {im.format}'

                # verify labels
                if os.path.isfile(lb_file):
                    nf += 1  # label found
                    with open(lb_file, 'r') as f:
                        l = [x.split() for x in f.read().strip().splitlines()]
                        if any([len(x) > 8 for x in l]):  # is segment
                            classes = np.array([x[0] for x in l], dtype=np.float32)
                            segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in l]  # (cls, xy1...)
                            l = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
                        l = np.array(l, dtype=np.float32)
                    if len(l):
                        assert l.shape[1] == 5, 'labels require 5 columns each'
                        assert (l >= 0).all(), 'negative labels'
                        assert (l[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels'
                        assert np.unique(l, axis=0).shape[0] == l.shape[0], 'duplicate labels'
                    else:
                        ne += 1  # label empty
                        l = np.zeros((0, 5), dtype=np.float32)
                else:
                    nm += 1  # label missing
                    l = np.zeros((0, 5), dtype=np.float32)
                x[im_file] = [l, shape, segments]
            except Exception as e:
                nc += 1
                print(f'{prefix}WARNING: Ignoring corrupted image and/or label {im_file}: {e}')

            pbar.desc = f"{prefix}Scanning '{path.parent / path.stem}' images and labels... " \
                        f"{nf} found, {nm} missing, {ne} empty, {nc} corrupted"
        pbar.close()

        if nf == 0:
            print(f'{prefix}WARNING: No labels found in {path}. See {help_url}')

        x['hash'] = get_hash(self.label_files + self.img_files)
        x['results'] = nf, nm, ne, nc, i + 1
        x['version'] = 0.1  # cache version
        torch.save(x, path)  # save for next time
        logging.info(f'{prefix}New cache created: {path}')
        return x

    def __len__(self):
        return len(self.img_files)

    # def __iter__(self):
    #     self.count = -1
    #     print('ran dataset iter')
    #     #self.shuffled_vector = np.random.permutation(self.nF) if self.augment else np.arange(self.nF)
    #     return self

    def __getitem__(self, index):

        index = self.indices[index]  # linear, shuffled, or image_weights
        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp['mosaic']

        if mosaic:
            # Load mosaic
            img, labels = load_mosaic(self, index)
            shapes = None

            if random.random() < hyp['mixup']:
                img2, labels2 = load_mosaic(self, random.randint(0, self.n - 1))
                r = np.random.beta(8.0, 8.0)  # mixup ratio, alpha=beta=8.0
                img = (img * r + img2 * (1 - r)).astype(np.uint8)
                labels = np.concatenate((labels, labels2), 0)

        else:
            # Load image
            img, (h0, w0), (h, w) = load_image(self, index)

            # Letterbox
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            labels = self.labels[index].copy()
            if labels.size:  # normalized xywh to pixel xyxy format
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

        if self.augment:
            # Augment imagespace
            if not mosaic:
                img, labels = random_perspective(img, labels,
                                                 degrees=hyp['degrees'],
                                                 translate=hyp['translate'],
                                                 scale=hyp['scale'],
                                                 shear=hyp['shear'],
                                                 perspective=hyp['perspective'])

            # Augment colorspace
            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

            # Apply cutouts
            # if random.random() < 0.9:
            #     labels = cutout(img, labels)

        nL = len(labels)  # number of labels
        if nL:
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])  # convert xyxy to xywh
            labels[:, [2, 4]] /= img.shape[0]  # normalized height 0-1
            labels[:, [1, 3]] /= img.shape[1]  # normalized width 0-1

        random.seed(index)
        if self.augment:
            # flip up-down
            if random.random() < hyp['flipud']:
                img = np.flipud(img)
                if nL:
                    labels[:, 2] = 1 - labels[:, 2]

            # flip left-right
            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)
                if nL:
                    labels[:, 1] = 1 - labels[:, 1]

        labels_out = torch.zeros((nL, 6))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img), labels_out, self.img_files[index], shapes

    @staticmethod
    def collate_fn(batch):
        img, label, absence, path, shapes = zip(*batch)
        # 取第一个样本的缺失模式作为 batch 全部缺失模式
        batch_absence = absence[0].unsqueeze(0).repeat(len(batch), 1)
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), batch_absence, path, shapes

    @staticmethod
    def collate_fn4(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        n = len(shapes) // 4
        img4, label4, path4, shapes4 = [], [], path[:n], shapes[:n]

        print("n = len(shapes) // 4", n)
        print("Image Shape", img.shape)

        ho = torch.tensor([[0., 0, 0, 1, 0, 0]])
        wo = torch.tensor([[0., 0, 1, 0, 0, 0]])
        s = torch.tensor([[1, 1, .5, .5, .5, .5]])  # scale
        for i in range(n):  # zidane torch.zeros(16,3,720,1280)  # BCHW
            i *= 4
            if random.random() < 0.5:
                im = F.interpolate(img[i].unsqueeze(0).float(), scale_factor=2., mode='bilinear', align_corners=False)[
                    0].type(img[i].type())
                l = label[i]
            else:
                im = torch.cat((torch.cat((img[i], img[i + 1]), 1), torch.cat((img[i + 2], img[i + 3]), 1)), 2)
                l = torch.cat((label[i], label[i + 1] + ho, label[i + 2] + wo, label[i + 3] + ho + wo), 0) * s
            img4.append(im)
            label4.append(l)

        for i, l in enumerate(label4):
            l[:, 0] = i  # add target image index for build_targets()

        return torch.stack(img4, 0), torch.cat(label4, 0), path4, shapes4


class LoadMultiModalImagesAndLabels(Dataset):  # for training/testing
    """
             载入多模态数据  TIR NIR VIS
    """
    def __init__(self, path_rgb, path_tir, path_nir,img_size=640, batch_size=16, augment=False, hyp=None, rect=False, patch=False, image_weights=False,
                 cache_images=False, single_cls=False, stride=32, pad=0.0, prefix=''):
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        print("====rect,mosic value:",self.rect,self.mosaic)     #train rect:False mosaic:True   test rect:True mosaic:False
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        self.path_rgb = path_rgb
        self.path_tir = path_tir
        self.path_nir =  path_nir
        self.patch = patch
        # self.path_file = path_file
        self.matching_filenames = []
        try:
            f_rgb = []  # image files
            f_tir = []
            f_nir = []
            # -----------------------------  rgb   -----------------------------
            for p_rgb in path_rgb if isinstance(path_rgb, list) else [path_rgb]:
                p_rgb = Path(p_rgb)  # os-agnostic
                if p_rgb.is_dir():  # dir
                    f_rgb += glob.glob(str(p_rgb / '**' / '*.*'), recursive=True)
                    # f = list(p.rglob('**/*.*'))  # pathlib
                elif p_rgb.is_file():  # file
                    with open(p_rgb, 'r') as t:
                        t = t.read().strip().splitlines()
                        parent = str(p_rgb.parent) + os.sep
                        f_rgb += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                        # f += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise Exception(f'{prefix}{path_rgb} does not exist')

            # -----------------------------  TIR  -----------------------------
            for p_tir in path_tir if isinstance(path_tir, list) else [path_tir]:
                p_tir = Path(p_tir)  # os-agnostic
                if p_tir.is_dir():  # dir
                    f_tir += glob.glob(str(p_tir / '**' / '*.*'), recursive=True)
                    # f = list(p.rglob('**/*.*'))  # pathlib
                elif p_tir.is_file():  # file
                    with open(p_tir, 'r') as t:
                        t = t.read().strip().splitlines()
                        parent = str(p_tir.parent) + os.sep
                        f_tir += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                        # f += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise Exception(f'{prefix}{p_tir} does not exist')
            
            # -----------------------------  NIR  -----------------------------
            for p_nir in path_nir if isinstance(path_nir, list) else [path_nir]:
                p_nir = Path(p_nir)  # os-agnostic
                if p_nir.is_dir():  # dir
                    f_nir += glob.glob(str(p_nir / '**' / '*.*'), recursive=True)
                    # f = list(p.rglob('**/*.*'))  # pathlib
                elif p_nir.is_file():  # file
                    with open(p_nir, 'r') as t:
                        t = t.read().strip().splitlines()
                        parent = str(p_nir.parent) + os.sep
                        f_nir += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                        # f += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise Exception(f'{prefix}{p_nir} does not exist')
            
            # 得到各个光谱的所有图片
            self.img_files_rgb = sorted([x.replace('/', os.sep) for x in f_rgb if x.split('.')[-1].lower() in img_formats])
            self.img_files_tir = sorted([x.replace('/', os.sep) for x in f_tir if x.split('.')[-1].lower() in img_formats])
            self.img_files_nir = sorted([x.replace('/', os.sep) for x in f_nir if x.split('.')[-1].lower() in img_formats])

            assert (self.img_files_rgb, self.img_files_tir,self.img_files_nir), (f'{prefix}No images found', f'{prefix}No images found',f'{prefix}No images found')
        except Exception as e:
            raise Exception(f'{prefix}Error loading data from {path_rgb,path_tir,path_nir}: {e}\nSee {help_url}')

        # 获取图片对应的标签
        ####################################################################################### Check rgb cache
        # 得到rgb图片的labels标签路径
        self.label_files_rgb = img2label_paths(self.img_files_rgb)  # labels
        
        # print(self.label_files)
        cache_rgb_path = (p_rgb if p_rgb.is_file() else Path(self.label_files_rgb[0]).parent).with_suffix('.cache')  # cached labels
        if cache_rgb_path.is_file():
            cache_rgb, exists_rgb = torch.load(cache_rgb_path), True  # load
            if cache_rgb['hash'] != get_hash(self.label_files_rgb + self.img_files_rgb) or 'version' not in cache_rgb:  # changed
                cache_rgb, exists_rgb = self.cache_labels(self.img_files_rgb,self.label_files_rgb,
                                                          cache_rgb_path, prefix), False  # re-cache
        else:
            cache_rgb, exists_rgb = self.cache_labels(self.img_files_rgb,self.label_files_rgb,
                                                      cache_rgb_path, prefix), False  # cache
    
        ####################################################################################### Check tir cache
        self.label_files_tir = img2label_paths(self.img_files_tir)  # labels
        # print(self.label_files)
        cache_tir_path = (p_tir if p_tir.is_file() else Path(self.label_files_tir[0]).parent).with_suffix('.cache')  # cached labels
        if cache_tir_path.is_file():
            cache_tir, exists_tir = torch.load(cache_tir_path), True  # load
            if cache_tir['hash'] != get_hash(self.label_files_tir + self.img_files_tir) or 'version' not in cache_tir:  # changed
                cache_tir, exists_tir = self.cache_labels(self.img_files_tir, self.label_files_tir,
                                                        cache_tir_path, prefix), False  # re-cache
        else:
            cache_tir, exists_tir = self.cache_labels(self.img_files_tir, self.label_files_tir,
                                                    cache_tir_path, prefix), False  # cache

        ####################################################################################### Check nir cache
        self.label_files_nir = img2label_paths(self.img_files_nir)  # labels
        # print(self.label_files)
        cache_nir_path = (p_nir if p_nir.is_file() else Path(self.label_files_nir[0]).parent).with_suffix('.cache')  # cached labels
        if cache_nir_path.is_file():
            cache_nir, exists_nir = torch.load(cache_nir_path), True  # load
            if cache_nir['hash'] != get_hash(self.label_files_nir + self.img_files_nir) or 'version' not in cache_nir:  # changed
                cache_nir, exists_nir = self.cache_labels(self.img_files_nir, self.label_files_nir,
                                                        cache_nir_path, prefix), False  # re-cache
        else:
            cache_nir, exists_nir = self.cache_labels(self.img_files_nir, self.label_files_nir,
                                                    cache_nir_path, prefix), False  # cache

        ########################################################################################### 展示TIR，NIR，VIS的加载情况
        nf_rgb, nm_rgb, ne_rgb, nc_rgb, n_rgb = cache_rgb.pop('results')  # found, missing, empty, corrupted, total
        nf_ir, nm_ir, ne_ir, nc_ir, n_ir = cache_tir.pop('results')  # found, missing, empty, corrupted, total
        nf_nir, nm_nir, ne_nir, nc_nir, n_nir = cache_nir.pop('results')  # found, missing, empty, corrupted, total

        if exists_rgb:
            d = f"Scanning RGB '{cache_rgb_path}' images and labels... {nf_rgb} found, {nm_rgb} missing, {ne_rgb} empty, {nc_rgb} corrupted"
            tqdm(None, desc=prefix + d, total=n_rgb, initial=n_rgb)  # display cache results
        if exists_tir:
            d = f"Scanning IR '{cache_tir_path}' images and labels... {nf_ir} found, {nm_ir} missing, {ne_ir} empty, {nc_ir} corrupted"
            tqdm(None, desc=prefix + d, total=n_ir, initial=n_ir)  # display cache results
        if exists_nir:
            d = f"Scanning IR '{cache_nir_path}' images and labels... {nf_nir} found, {nm_nir} missing, {ne_nir} empty, {nc_nir} corrupted"
            tqdm(None, desc=prefix + d, total=n_nir, initial=n_nir)  # display cache results

        assert nf_rgb > 0 or not augment, f'{prefix}No labels in {cache_rgb_path}. Can not train without labels. See {help_url}'

        # Read cache
        # Read RGB cache
        cache_rgb.pop('hash')  # remove hash
        cache_rgb.pop('version')  # remove version
        labels_rgb, shapes_rgb, self.segments_rgb = zip(*cache_rgb.values())
        self.labels_rgb = list(labels_rgb)
        self.shapes_rgb = np.array(shapes_rgb, dtype=np.float64)
        self.img_files_rgb = list(cache_rgb.keys())  # update
        self.label_files_rgb = img2label_paths(cache_rgb.keys())  # update
        if single_cls:
            for x in self.labels_rgb:
                x[:, 0] = 0

        n_rgb = len(shapes_rgb)  # number of images
        bi_rgb = np.floor(np.arange(n_rgb) / batch_size).astype(np.int32)  # batch index
        nb_rgb = bi_rgb[-1] + 1  # number of batches
        self.batch_rgb = bi_rgb  # batch index of image
        self.n_rgb = n_rgb
        self.indices_rgb = range(n_rgb)

        # Read IR cache
        cache_tir.pop('hash')  # remove hash
        cache_tir.pop('version')  # remove version
        labels_ir, shapes_ir, self.segments_ir = zip(*cache_tir.values())
        self.labels_ir = list(labels_ir)
        self.shapes_ir = np.array(shapes_ir, dtype=np.float64)
        self.img_files_ir = list(cache_tir.keys())  # update
        self.label_files_ir = img2label_paths(cache_tir.keys())  # update
        if single_cls:
            for x in self.labels_ir:
                x[:, 0] = 0

        n_ir = len(shapes_ir)  # number of images
        bi_ir = np.floor(np.arange(n_ir) / batch_size).astype(np.int32)  # batch index
        nb_ir = bi_ir[-1] + 1  # number of batches
        self.batch_ir = bi_ir  # batch index of image
        self.n_ir = n_ir
        self.indices_ir = range(n_ir)

        # Read NIR cache
        cache_nir.pop('hash')  # remove hash
        cache_nir.pop('version')  # remove version
        labels_nir, shapes_nir, self.segments_nir = zip(*cache_nir.values())
        self.labels_nir = list(labels_nir)
        self.shapes_nir = np.array(shapes_nir, dtype=np.float64)
        self.img_files_nir = list(cache_nir.keys())  # update
        self.label_files_nir = img2label_paths(cache_nir.keys())  # update
        if single_cls:
            for x in self.labels_nir:
                x[:, 0] = 0

        n_nir = len(shapes_nir)  # number of images
        bi_nir = np.floor(np.arange(n_nir) / batch_size).astype(np.int32)  # batch index
        nb_nir = bi_nir[-1] + 1  # number of batches
        self.batch_nir = bi_nir  # batch index of image
        self.n_nir = n_nir
        self.indices_nir = range(n_nir) 


        # Rectangular Training
        if self.rect:
            # RGB
            # Sort by aspect ratio
            s_rgb = self.shapes_rgb  # wh
            ar_rgb = s_rgb[:, 1] / s_rgb[:, 0]  # aspect ratio
            irect_rgb = ar_rgb.argsort()
            self.img_files_rgb = [self.img_files_rgb[i] for i in irect_rgb]
            self.label_files_rgb = [self.label_files_rgb[i] for i in irect_rgb]
            self.labels_rgb = [self.labels_rgb[i] for i in irect_rgb]
            self.shapes_rgb = s_rgb[irect_rgb]  # wh
            ar_rgb = ar_rgb[irect_rgb]

            # Set training image shapes
            shapes_rgb = [[1, 1]] * nb_rgb
            for i in range(nb_rgb):
                ari_rgb = ar_rgb[bi_rgb == i]
                mini, maxi = ari_rgb.min(), ari_rgb.max()
                if maxi < 1:
                    shapes_rgb[i] = [maxi, 1]
                elif mini > 1:
                    shapes_rgb[i] = [1, 1 / mini]

            self.batch_shapes_rgb = np.ceil(np.array(shapes_rgb) * img_size / stride + pad).astype(np.int32) * stride

            # IR
            # Sort by aspect ratio
            s_ir = self.shapes_ir  # wh
            ar_ir = s_ir[:, 1] / s_ir[:, 0]  # aspect ratio
            irect_ir = ar_ir.argsort()
            self.img_files_ir = [self.img_files_ir[i] for i in irect_ir]
            self.label_files_ir = [self.label_files_ir[i] for i in irect_ir]
            self.labels_ir = [self.labels_ir[i] for i in irect_ir]
            self.shapes_ir = s_ir[irect_ir]  # wh
            ar_ir = ar_ir[irect_ir]

            # Set training image shapes
            shapes_ir = [[1, 1]] * nb_ir
            for i in range(nb_ir):
                ari_ir = ar_ir[bi_ir == i]
                mini, maxi = ari_ir.min(), ari_ir.max()
                if maxi < 1:
                    shapes_ir[i] = [maxi, 1]
                elif mini > 1:
                    shapes_ir[i] = [1, 1 / mini]

            self.batch_shapes_ir = np.ceil(np.array(shapes_ir) * img_size / stride + pad).astype(np.int32) * stride


            # NIR
            s_nir = self.shapes_nir  # wh
            ar_nir = s_nir[:, 1] / s_nir[:, 0]  # aspect ratio
            irect_nir = ar_nir.argsort()
            self.img_files_nir = [self.img_files_nir[i] for i in irect_nir]
            self.label_files_nir = [self.label_files_nir[i] for i in irect_nir]
            self.labels_nir = [self.labels_nir[i] for i in irect_nir]
            self.shapes_nir = s_nir[irect_nir]  # wh
            ar_nir = ar_nir[irect_nir]

            # Set training image shapes
            shapes_nir = [[1, 1]] * nb_nir
            for i in range(nb_nir):
                ari_nir = ar_nir[bi_nir == i]
                mini, maxi = ari_nir.min(), ari_nir.max()
                if maxi < 1:
                    shapes_nir[i] = [maxi, 1]
                elif mini > 1:
                    shapes_nir[i] = [1, 1 / mini]

            self.batch_shapes_nir = np.ceil(np.array(shapes_nir) * img_size / stride + pad).astype(np.int32) * stride


        # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
        self.imgs_rgb = [None] * n_rgb
        self.imgs_ir = [None] * n_ir
        self.imgs_nir = [None] * n_nir

        self.labels = self.labels_rgb
        self.shapes = self.shapes_rgb
        self.indices = self.indices_rgb

        # 7 种缺失模式（0 = present, 1 = absent）
        self.absence_modes = [
            [0, 0, 0],  # none missing
            [1, 0, 0],  # missing rgb
            [0, 1, 0],  # missing tir
            [0, 0, 1],  # missing nir
            [1, 1, 0],  # missing rgb + tir
            [1, 0, 1],  # missing rgb + nir
            [0, 1, 1],  # missing tir + nir
        ]
        # 是否启用缺失模拟（建议只在 train）
        self.simulate_absence = True

    def cache_labels(self, imgfiles, labelfiles, path=Path('./labels.cache'), prefix=''):
        # Cache dataset labels, check images and read shapes
        img_files = imgfiles
        label_files = labelfiles
        x = {}  # dict
        nm, nf, ne, nc = 0, 0, 0, 0  # number missing, found, empty, duplicate
        pbar = tqdm(zip(img_files, label_files), desc='Scanning images', total=len(img_files))
        for i, (im_file, lb_file) in enumerate(pbar):
            try:
                # verify images
                im = Image.open(im_file)
                im.verify()  # PIL verify
                shape = exif_size(im)  # image size
                segments = []  # instance segments
                assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
                assert im.format.lower() in img_formats, f'invalid image format {im.format}'

                # verify labels
                if os.path.isfile(lb_file):
                    nf += 1  # label found
                    with open(lb_file, 'r') as f:
                        l = [x.split() for x in f.read().strip().splitlines()]
                        if any([len(x) > 8 for x in l]):  # is segment
                            classes = np.array([x[0] for x in l], dtype=np.float32)
                            segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in l]  # (cls, xy1...)
                            l = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
                        l = np.array(l, dtype=np.float32)
                    if len(l):
                        # 分别检查 l的列数是否为5
                        # 所有的标签值是否都是非负的
                        # 所有的坐标值在0-1之间
                        # 确保标签中没有重复的项
                        assert l.shape[1] == 5, 'labels require 5 columns each' 
                        assert (l >= 0).all(), 'negative labels'
                        assert (l[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels'
                        assert np.unique(l, axis=0).shape[0] == l.shape[0], 'duplicate labels'
                    else:
                        ne += 1  # label empty
                        l = np.zeros((0, 5), dtype=np.float32)
                else:
                    nm += 1  # label missing
                    l = np.zeros((0, 5), dtype=np.float32)
                x[im_file] = [l, shape, segments]
            except Exception as e:
                nc += 1
                print(f'{prefix}WARNING: Ignoring corrupted image and/or label {lb_file}: {e}')

            pbar.desc = f"{prefix}Scanning '{path.parent / path.stem}' images and labels... " \
                        f"{nf} found, {nm} missing, {ne} empty, {nc} corrupted"
        pbar.close()

        if nf == 0:
            print(f'{prefix}WARNING: No labels found in {path}. See {help_url}')

        x['hash'] = get_hash(label_files + img_files)
        x['results'] = nf, nm, ne, nc, i + 1
        x['version'] = 0.1  # cache version
        torch.save(x, path)  # save for next time
        logging.info(f'{prefix}New cache created: {path}')
        return x

    def get_matching_filenames(self):
        print(self.matching_filenames)
        return self.matching_filenames
    
    def __len__(self):
        return len(self.img_files_rgb)

    # def __iter__(self):
    #     self.count = -1
    #     print('ran dataset iter')
    #     #self.shuffled_vector = np.random.permutation(self.nF) if self.augment else np.arange(self.nF)
    #     return self

    def __getitem__(self, index):
        # index = self.indices[index]  # linear, shuffled, or image_weights
        index_rgb = self.indices_rgb[index]  # linear, shuffled, or image_weights
        index_ir = self.indices_ir[index]  # linear, shuffled, or image_weights
        index_nir = self.indices_nir[index]  # linear, shuffled, or image_weights

        hyp = self.hyp
        # 在mosaic之前进行处理   which modal is absence,0-rgb,1-ir,2-none
        # 仅在训练的过程中缺失，训练时测试不缺失，测试不缺失
        mosaic = self.mosaic and random.random() < hyp['mosaic']
        
        if mosaic:
            # Load mosaic
            # img, labels = load_mosaic(self, index)
            img_rgb, labels_rgb, img_ir, labels_ir,img_nir,labels_nir = load_mosaic_RGB_IR(self, index_rgb, index_ir,index_nir)

            shapes = None

        else:
            # Load image
            img_rgb, img_ir, img_nir,(h0, w0), (h, w) = load_image_rgb_ir(self, index)

            # Letterbox
            shape = self.batch_shapes_rgb[self.batch_rgb[index]] if self.rect else self.img_size  # final letterboxed shape
            img_rgb, ratio, pad = letterbox(img_rgb, shape, auto=False, scaleup=self.augment)
            img_ir, ratio, pad = letterbox(img_ir, shape, auto=False, scaleup=self.augment)
            img_nir, ratio, pad = letterbox(img_nir, shape, auto=False, scaleup=self.augment)

            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            labels = self.labels_rgb[index].copy()
            if labels.size:  # normalized xywh to pixel xyxy format
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

            labels_rgb = labels
            labels_ir = labels

        if self.augment:
            # Augment colorspace
            img_rgb = augment_hsv(img_rgb, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])
            img_ir = augment_hsv(img_ir, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])
            img_nir = augment_hsv(img_nir, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'],vgain=hyp['hsv_v'])

            # 给热红外图片加上高斯噪声
            # img_ir = add_gaussian_noise(img_ir, 0, 0.8)
            # img_rgb = add_gaussian_noise(img_rgb, 0, 0.5)
            # img_rgb = augment_hsv(img_rgb, hgain=hyp['hsv_h'], sgain=1.0, vgain=hyp['hsv_v'])
            # img_ir = augment_hsv(img_ir, hgain=hyp['hsv_h'], sgain=0.5, vgain=hyp['hsv_v'])
            
        
            # 处理NIR
            # augment_hsv(img_nir, hgain=0.0, sgain= 2.0, vgain= 2.0)

        nL = len(labels_rgb)  # number of labels
        if nL:
            labels_rgb[:, 1:5] = xyxy2xywh(labels_rgb[:, 1:5])  # convert xyxy to xywh
            labels_rgb[:, [2, 4]] /= img_rgb.shape[0]  # normalized height 0-1
            labels_rgb[:, [1, 3]] /= img_rgb.shape[1]  # normalized width 0-1

        if self.augment:
            # flip up-down
            if random.random() < hyp['flipud']:
                img_rgb = np.flipud(img_rgb)
                img_ir = np.flipud(img_ir)
                # 翻转NIR图像
                img_nir = np.flipud(img_nir)
                if nL:
                    labels_rgb[:, 2] = 1 - labels_rgb[:, 2]

            # flip left-right
            if random.random() < hyp['fliplr']:
                img_rgb = np.fliplr(img_rgb)
                img_ir = np.fliplr(img_ir)
                img_nir = np.fliplr(img_nir)
                if nL:
                    labels_rgb[:, 1] = 1 - labels_rgb[:, 1]

        labels_out = torch.zeros((nL, 6))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels_rgb)

        # Convert
        # img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        # img = np.ascontiguousarray(img)
        # BGR to RGB：img_rgb[:, :, ::-1] 和 img_ir[:, :, ::-1] 将图像的颜色通道从BGR转换为RGB。[:, :, ::-1] 表示在第三个维度（颜色通道）上进行反转。
        # 维度转换：transpose(2, 0, 1) 将图像的维度从 (height, width, channels) 转换为 (channels, height, width)。
        # 确保内存连续：np.ascontiguousarray 确保图像数据在内存中是连续的，以提高处理效率。

        img_rgb = img_rgb[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img_rgb = np.ascontiguousarray(img_rgb)
        img_ir = img_ir[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img_ir = np.ascontiguousarray(img_ir)
        # 处理NIR
        img_nir = img_nir[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img_nir = np.ascontiguousarray(img_nir)
        
        # 随机采样缺失模式
        if self.mosaic and self.simulate_absence:
            if random.random() < 0.3:
                mode = random.choice(self.absence_modes[1:])
            else:
                mode = [0, 0, 0]
        else:
            mode = [0, 0, 0]
        
        absent_rgb, absent_tir, absent_nir = mode
        if absent_rgb:
            img_rgb = np.zeros_like(img_rgb)

        if absent_tir:
            img_ir = np.zeros_like(img_ir)

        if absent_nir:
            img_nir = np.zeros_like(img_nir)

        absence_mask = torch.tensor([absent_rgb, absent_tir, absent_nir],dtype=torch.float32)

        img_all = np.concatenate((img_rgb, img_ir,img_nir), axis=0)
        
        return torch.from_numpy(img_all), labels_out, absence_mask, self.img_files_rgb[index], shapes

    @staticmethod
    def collate_fn(batch):
        img, label, absence, path, shapes = zip(*batch)
        # 取第一个样本的缺失模式作为 batch 全部缺失模式
        batch_absence = absence[0].unsqueeze(0).repeat(len(batch), 1)
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), batch_absence, path, shapes

    @staticmethod
    def collate_fn4(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        n = len(shapes) // 4
        img4, label4, path4, shapes4 = [], [], path[:n], shapes[:n]

        ho = torch.tensor([[0., 0, 0, 1, 0, 0]])
        wo = torch.tensor([[0., 0, 1, 0, 0, 0]])
        s = torch.tensor([[1, 1, .5, .5, .5, .5]])  # scale
        for i in range(n):  # zidane torch.zeros(16,3,720,1280)  # BCHW
            i *= 4
            if random.random() < 0.5:
                im = F.interpolate(img[i].unsqueeze(0).float(), scale_factor=2., mode='bilinear', align_corners=False)[
                    0].type(img[i].type())
                l = label[i]
            else:
                im = torch.cat((torch.cat((img[i], img[i + 1]), 1), torch.cat((img[i + 2], img[i + 3]), 1)), 2)
                l = torch.cat((label[i], label[i + 1] + ho, label[i + 2] + wo, label[i + 3] + ho + wo), 0) * s
            img4.append(im)
            label4.append(l)

        for i, l in enumerate(label4):
            l[:, 0] = i  # add target image index for build_targets()

        return torch.stack(img4, 0), torch.cat(label4, 0), path4, shapes4

# Ancillary functions --------------------------------------------------------------------------------------------------

def shift_augment(self, img):
    direction = [[-1, 1], [1, 1], [1, -1], [-1, -1]]  # left-up, right-up, right-down, left-down

    shift_X, shift_Y = random.randint(0, 10), random.randint(0, 10)
    shift_direction = direction[random.randint(0, 3)]
    c, h, w = img.shape

    shift_img = np.full((c, h, w), 114, dtype=np.uint8)
    if shift_direction == [-1, 1]:  # left-up shift
        shift_img[:, :h - shift_Y, :w - shift_X] = img[:, shift_Y:, shift_X:]
    elif shift_direction == [1, 1]:  # right-up shift
        shift_img[:, :h - shift_Y, shift_X:] = img[:, shift_Y:, :w - shift_X]
    elif shift_direction == [1, -1]:  # right-down shift
        shift_img[:, shift_Y:, shift_X:] = img[:, :h - shift_Y, :w - shift_X]
    else:
        shift_img[:, :h - shift_Y, shift_X:] = img[:, shift_Y:, :w - shift_X]

    return shift_img

def load_image(self, index):
    # loads 1 image from dataset, returns img, original hw, resized hw
    img = self.imgs[index]
    if img is None:  # not cached
        path = self.img_files[index]
        img = cv2.imread(path)  # BGR
        assert img is not None, 'Image Not Found ' + path
        h0, w0 = img.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)),
                             interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR)
        return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized
    else:
        return self.imgs[index], self.img_hw0[index], self.img_hw[index]  # img, hw_original, hw_resized


def load_image_rgb_ir(self, index):
    # loads 1 image from dataset, returns img, original hw, resized hw

    img_rgb = self.imgs_rgb[index]
    img_ir = self.imgs_ir[index]
    img_nir = self.imgs_nir[index]
    # img_rgb = None
    # img_ir = None

    if (img_rgb is None) and (img_ir is None) and (img_nir is None):  # not cached

        path_rgb = self.img_files_rgb[index]
        path_ir = self.img_files_ir[index]
        path_nir = self.img_files_nir[index]

        img_rgb = cv2.imread(path_rgb)  # BGR
        img_ir = cv2.imread(path_ir)  # BGR
        img_nir = cv2.imread(path_nir) #  BGR

        assert img_rgb is not None, 'Image RGB Not Found ' + path_rgb
        assert img_ir is not None, 'Image IR Not Found ' + path_ir
        assert img_nir is not None, 'Image NIR Not Found ' + path_nir

        h0, w0 = img_rgb.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            img_rgb = cv2.resize(img_rgb, (int(w0 * r), int(h0 * r)),
                             interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR)
            img_ir = cv2.resize(img_ir, (int(w0 * r), int(h0 * r)),
                             interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR)
            img_nir = cv2.resize(img_nir, (int(w0 * r), int(h0 * r)),
                             interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR)
        return img_rgb, img_ir, img_nir,(h0, w0), img_rgb.shape[:2]  # img, hw_original, hw_resized
    else:
        return self.imgs_rgb[index], self.imgs_ir[index], self.img_nir[index], self.img_hw0_rgb[index], self.img_hw_rgb[index]  # img, hw_original, hw_resized

def add_patch_over_images (img,min_patch_width=1,min_patch_height=1,max_patch_width=50,max_patch_height=50,max_num_patches=5):

    # patch的数量和位置是随机的
    # patch大小要在约束范围内
    img_height, img_width, _ = img.shape
    num_patches = random.randint(0,max_num_patches)
    for _ in range(num_patches):
        # 随机生成patch块的大小
        patch_height = random.randint(min_patch_height,max_patch_height)
        patch_width = random.randint(min_patch_width, max_patch_width)

        # 随机生成patch块的位置
        x = random.randint(0, img_width - patch_width)
        y = random.randint(0, img_height - patch_height)

        # 随机生成patch块的颜色
        patch_color = 0

        # 添加patch块到图像 channel,height,width
        img[:,y:y + patch_height, x:x + patch_width] = patch_color

    return img
    

def augment_hsv_oral(img, hgain=0.5, sgain=0.5, vgain=0.5):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed

def augment_hsv(img, hgain=0.0, sgain=0.0, vgain=0.0):
    # 将图像从BGR转换到HSV
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 计算增益并应用到HSV通道
    hsv_img[:, :, 0] = cv2.add(hsv_img[:, :, 0], hgain)  # 色相
    hsv_img[:, :, 1] = cv2.add(hsv_img[:, :, 1], sgain)  # 饱和度
    hsv_img[:, :, 2] = cv2.add(hsv_img[:, :, 2], vgain)  # 亮度
    
    # 把超出范围的值裁剪到[0, 255]
    hsv_img = np.clip(hsv_img, 0, 255).astype(np.uint8)
    
    # 将图像从HSV转换回BGR
    img_augmented = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    return img_augmented

def add_gaussian_noise(image, mean=0, sigma=25):
    row, col, ch = image.shape
    gauss = np.random.normal(mean, sigma, (row, col, ch)).astype('uint8')
    noisy_image = cv2.add(image, gauss)
    return noisy_image

def hist_equalize(img, clahe=True, bgr=False):
    # Equalize histogram on BGR image 'img' with img.shape(n,m,3) and range 0-255
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV if bgr else cv2.COLOR_RGB2YUV)
    if clahe:
        c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        yuv[:, :, 0] = c.apply(yuv[:, :, 0])
    else:
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])  # equalize Y channel histogram
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR if bgr else cv2.COLOR_YUV2RGB)  # convert YUV image to RGB


def load_mosaic(self, index):
    # loads images in a 4-mosaic
    labels4, segments4 = [], []
    s = self.img_size
    yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # mosaic center x, y
    indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
    for i, index in enumerate(indices):
        # Load image
        img, _, (h, w) = load_image(self, index)

        # place img in img4
        if i == 0:  # top left
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

        # Labels
        labels, segments = self.labels[index].copy(), self.segments[index].copy()
        if labels.size:
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
            segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
        labels4.append(labels)
        segments4.extend(segments)

    # Concat/clip labels
    labels4 = np.concatenate(labels4, 0)
    for x in (labels4[:, 1:], *segments4):
        np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
    # img4, labels4 = replicate(img4, labels4)  # replicate

    # Augment
    img4, labels4 = random_perspective(img4, labels4, segments4,
                                       degrees=self.hyp['degrees'],
                                       translate=self.hyp['translate'],
                                       scale=self.hyp['scale'],
                                       shear=self.hyp['shear'],
                                       perspective=self.hyp['perspective'],
                                       border=self.mosaic_border)  # border to remove
    # print(labels4)
    return img4, labels4

def load_mosaic_RGB_IR(self, index1, index2,index3):
    # loads images in a 4-mosaic
    index_rgb = index1
    index_ir = index2
    index_nir = index3

    labels4_rgb, segments4_rgb = [], []
    labels4_ir, segments4_ir = [], []
    labels4_nir, segments4_nir = [], []

    s = self.img_size

    # print("image size ", s)

    yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # mosaic center x, y

    assert index_rgb == index_ir == index_nir, 'INDEX RGB 不等于 INDEX IR,INDEX NIR'

    indices = [index_rgb] + random.choices(self.indices_rgb, k=3)  # 3 additional image indices

    for i, index in enumerate(indices):
        # Load image

        # img, _, (h, w) = load_image(self, index)
        img_rgb, img_ir,img_nir, _, (h, w) = load_image_rgb_ir(self, index)
        # 返回RGB和IR和NIR三种光谱的图像
        # 在这里加上patch
            # img_rgb, img_ir, (h0, w0), (h, w) = load_image_rgb_ir(self, index)
        # place img in img4
        if i == 0:  # top left
            img4_rgb = np.full((s * 2, s * 2, img_rgb.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            img4_ir = np.full((s * 2, s * 2, img_ir.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            img4_nir = np.full((s * 2, s * 2, img_nir.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        # img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        # padw = x1a - x1b
        # padh = y1a - y1b

        img4_rgb[y1a:y2a, x1a:x2a] = img_rgb[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        img4_ir[y1a:y2a, x1a:x2a] = img_ir[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        img4_nir[y1a:y2a, x1a:x2a] = img_nir[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

        labels_rgb, segments_rgb = self.labels_rgb[index].copy(), self.segments_rgb[index].copy()
        labels_ir, segments_ir = self.labels_ir[index].copy(), self.segments_ir[index].copy()
        labels_nir, segments_nir = self.labels_nir[index].copy(), self.segments_nir[index].copy()
        if labels_rgb.size:
            labels_rgb[:, 1:] = xywhn2xyxy(labels_rgb[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
            labels_ir[:, 1:] = xywhn2xyxy(labels_ir[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
            labels_nir[:, 1:] = xywhn2xyxy(labels_nir[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
            segments_rgb = [xyn2xy(x, w, h, padw, padh) for x in segments_rgb]
            segments_ir = [xyn2xy(x, w, h, padw, padh) for x in segments_ir]
            segments_nir = [xyn2xy(x, w, h, padw, padh) for x in segments_nir]
        labels4_rgb.append(labels_rgb)
        segments4_rgb.extend(segments_rgb)
        labels4_ir.append(labels_ir)
        segments4_ir.extend(segments_ir)
        labels4_nir.append(labels_nir)
        segments4_nir.extend(segments_nir)

    # # Concat/clip labels
    # labels4 = np.concatenate(labels4, 0)
    # for x in (labels4[:, 1:], *segments4):
    #     np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
    # # img4, labels4 = replicate(img4, labels4)  # replicate

    labels4_rgb = np.concatenate(labels4_rgb, 0)
    labels4_ir = np.concatenate(labels4_ir, 0)
    labels4_nir = np.concatenate(labels4_nir, 0)
    for x in (labels4_rgb[:, 1:], *segments4_rgb):
        np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
    for x in (labels4_ir[:, 1:], *segments4_ir):
        np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
    # img4, labels4 = replicate(img4, labels4)  # replicate
    for x in (labels4_nir[:, 1:], *segments4_nir):
        np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()

    img4_rgb, img4_ir,img4_nir, labels4_rgb, labels4_ir,labels4_nir = random_perspective_rgb_ir(img4_rgb,
                                                                           img4_ir,
                                                                           img4_nir,
                                                                           labels4_rgb,
                                                                           labels4_ir,
                                                                           labels4_nir,
                                                                           segments4_rgb,
                                                                           segments4_ir,
                                                                           segments_nir,
                                                                           degrees=self.hyp['degrees'],
                                                                           translate=self.hyp['translate'],
                                                                           scale=self.hyp['scale'],
                                                                           shear=self.hyp['shear'],
                                                                           perspective=self.hyp['perspective'],
                                                                           border=self.mosaic_border
                                                                           )  # border to remove


    # print(labels_rgb)
    # print(labels4_ir)

    # assert labels4_rgb == labels4_ir, 'LABEL4 RGB 不等于 LABEL4 IR'
    # print(" labels4_rgb == labels4_ir ", labels4_rgb == labels4_ir)
    labels4_ir = labels4_rgb

    # cv2.imwrite("rgb_%s.jpg" % str(index1), img4_rgb)
    # cv2.imwrite("ir_%s.jpg" % str(index2), img4_ir)


    return img4_rgb, labels4_rgb, img4_ir, labels4_ir,img4_nir,labels4_nir

def load_mosaic9(self, index):
    # loads images in a 9-mosaic

    labels9, segments9 = [], []
    s = self.img_size
    indices = [index] + random.choices(self.indices, k=8)  # 8 additional image indices
    for i, index in enumerate(indices):
        # Load image
        img, _, (h, w) = load_image(self, index)

        # place img in img9
        if i == 0:  # center
            img9 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            h0, w0 = h, w
            c = s, s, s + w, s + h  # xmin, ymin, xmax, ymax (base) coordinates
        elif i == 1:  # top
            c = s, s - h, s + w, s
        elif i == 2:  # top right
            c = s + wp, s - h, s + wp + w, s
        elif i == 3:  # right
            c = s + w0, s, s + w0 + w, s + h
        elif i == 4:  # bottom right
            c = s + w0, s + hp, s + w0 + w, s + hp + h
        elif i == 5:  # bottom
            c = s + w0 - w, s + h0, s + w0, s + h0 + h
        elif i == 6:  # bottom left
            c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h
        elif i == 7:  # left
            c = s - w, s + h0 - h, s, s + h0
        elif i == 8:  # top left
            c = s - w, s + h0 - hp - h, s, s + h0 - hp

        padx, pady = c[:2]
        x1, y1, x2, y2 = [max(x, 0) for x in c]  # allocate coords

        # Labels
        labels, segments = self.labels[index].copy(), self.segments[index].copy()
        if labels.size:
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padx, pady)  # normalized xywh to pixel xyxy format
            segments = [xyn2xy(x, w, h, padx, pady) for x in segments]
        labels9.append(labels)
        segments9.extend(segments)

        # Image
        img9[y1:y2, x1:x2] = img[y1 - pady:, x1 - padx:]  # img9[ymin:ymax, xmin:xmax]
        hp, wp = h, w  # height, width previous

    # Offset
    yc, xc = [int(random.uniform(0, s)) for _ in self.mosaic_border]  # mosaic center x, y
    img9 = img9[yc:yc + 2 * s, xc:xc + 2 * s]

    # Concat/clip labels
    labels9 = np.concatenate(labels9, 0)
    labels9[:, [1, 3]] -= xc
    labels9[:, [2, 4]] -= yc
    c = np.array([xc, yc])  # centers
    segments9 = [x - c for x in segments9]

    for x in (labels9[:, 1:], *segments9):
        np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
    # img9, labels9 = replicate(img9, labels9)  # replicate

    # Augment
    img9, labels9 = random_perspective(img9, labels9, segments9,
                                       degrees=self.hyp['degrees'],
                                       translate=self.hyp['translate'],
                                       scale=self.hyp['scale'],
                                       shear=self.hyp['shear'],
                                       perspective=self.hyp['perspective'],
                                       border=self.mosaic_border)  # border to remove

    return img9, labels9


def replicate(img, labels):
    # Replicate labels
    h, w = img.shape[:2]
    boxes = labels[:, 1:].astype(int)
    x1, y1, x2, y2 = boxes.T
    s = ((x2 - x1) + (y2 - y1)) / 2  # side length (pixels)
    for i in s.argsort()[:round(s.size * 0.5)]:  # smallest indices
        x1b, y1b, x2b, y2b = boxes[i]
        bh, bw = y2b - y1b, x2b - x1b
        yc, xc = int(random.uniform(0, h - bh)), int(random.uniform(0, w - bw))  # offset x, y
        x1a, y1a, x2a, y2a = [xc, yc, xc + bw, yc + bh]
        img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        labels = np.append(labels, [[labels[i, 0], x1a, y1a, x2a, y2a]], axis=0)

    return img, labels


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

    # if auto:  # minimum rectangle
    #     dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    # elif scaleFill:  # stretch
    #     dw, dh = 0.0, 0.0
    #     new_unpad = (new_shape[1], new_shape[0])
    #     ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios
    #
    # dw /= 2  # divide padding into 2 sides
    # dh /= 2
    #
    # if shape[::-1] != new_unpad:  # resize
    #     img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    # top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    # left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    # img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def random_perspective(img, targets=(), segments=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0,
                       border=(0, 0)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]

    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Visualize
    # import matplotlib.pyplot as plt
    # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    # ax[0].imshow(img[:, :, ::-1])  # base
    # ax[1].imshow(img2[:, :, ::-1])  # warped

    # Transform label coordinates
    n = len(targets)
    if n:
        use_segments = any(x.any() for x in segments)
        new = np.zeros((n, 4))
        if use_segments:  # warp segments
            segments = resample_segments(segments)  # upsample
            for i, segment in enumerate(segments):
                xy = np.ones((len(segment), 3))
                xy[:, :2] = segment
                xy = xy @ M.T  # transform
                xy = xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]  # perspective rescale or affine

                # clip
                new[i] = segment2box(xy, width, height)

        else:  # warp boxes
            xy = np.ones((n * 4, 3))
            xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # transform
            xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # clip
            new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
            new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(box1=targets[:, 1:5].T * s, box2=new.T, area_thr=0.01 if use_segments else 0.10)
        targets = targets[i]
        targets[:, 1:5] = new[i]

    return img, targets

def random_perspective_rgb_ir(img_rgb, img_ir, img_nir,targets_rgb=(),targets_ir=(),targets_nir=(), segments_rgb=(), segments_ir=(),segments_nir=(),
                              degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0, border=(0, 0)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]

    img = img_rgb
    targets = targets_rgb
    segments = segments_rgb

    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            # img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
            img_rgb = cv2.warpPerspective(img_rgb, M, dsize=(width, height), borderValue=(114, 114, 114))
            img_ir = cv2.warpPerspective(img_ir, M, dsize=(width, height), borderValue=(114, 114, 114))
            img_nir = cv2.warpPerspective(img_nir, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            # img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))
            img_rgb = cv2.warpAffine(img_rgb, M[:2], dsize=(width, height), borderValue=(114, 114, 114))
            img_ir = cv2.warpAffine(img_ir, M[:2], dsize=(width, height), borderValue=(114, 114, 114))
            img_nir = cv2.warpAffine(img_nir, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Visualize
    # import matplotlib.pyplot as plt
    # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    # ax[0].imshow(img_rgb[:, :, ::-1])  # base
    # ax[1].imshow(img_ir[:, :, ::-1])  # warped


    # Transform label coordinates
    n = len(targets)
    if n:
        use_segments = any(x.any() for x in segments)
        new = np.zeros((n, 4))
        if use_segments:  # warp segments
            segments = resample_segments(segments)  # upsample
            for i, segment in enumerate(segments):
                xy = np.ones((len(segment), 3))
                xy[:, :2] = segment
                xy = xy @ M.T  # transform
                xy = xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]  # perspective rescale or affine

                # clip
                new[i] = segment2box(xy, width, height)

        else:  # warp boxes
            xy = np.ones((n * 4, 3))
            xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # transform
            xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # clip
            new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
            new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(box1=targets[:, 1:5].T * s, box2=new.T, area_thr=0.01 if use_segments else 0.10)
        targets = targets[i]
        targets[:, 1:5] = new[i]

    return img_rgb, img_ir, img_nir,targets, targets,targets


def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1, eps=1e-16):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates


def cutout(image, labels):
    # Applies image cutout augmentation https://arxiv.org/abs/1708.04552
    h, w = image.shape[:2]

    def bbox_ioa(box1, box2):
        # Returns the intersection over box2 area given box1, box2. box1 is 4, box2 is nx4. boxes are x1y1x2y2
        box2 = box2.transpose()

        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

        # Intersection area
        inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                     (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

        # box2 area
        box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + 1e-16

        # Intersection over box2 area
        return inter_area / box2_area

    # create random masks
    scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16  # image size fraction
    for s in scales:
        mask_h = random.randint(1, int(h * s))
        mask_w = random.randint(1, int(w * s))

        # box
        xmin = max(0, random.randint(0, w) - mask_w // 2)
        ymin = max(0, random.randint(0, h) - mask_h // 2)
        xmax = min(w, xmin + mask_w)
        ymax = min(h, ymin + mask_h)

        # apply random color mask
        image[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(3)]

        # return unobscured labels
        if len(labels) and s > 0.03:
            box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
            ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
            labels = labels[ioa < 0.60]  # remove >60% obscured labels

    return labels


def create_folder(path='./new'):
    # Create folder
    if os.path.exists(path):
        shutil.rmtree(path)  # delete output folder
    os.makedirs(path)  # make new output folder


def flatten_recursive(path='../coco128'):
    # Flatten a recursive directory by bringing all files to top level
    new_path = Path(path + '_flat')
    create_folder(new_path)
    for file in tqdm(glob.glob(str(Path(path)) + '/**/*.*', recursive=True)):
        shutil.copyfile(file, new_path / Path(file).name)


def extract_boxes(path='../coco128/'):  # from utils.datasets import *; extract_boxes('../coco128')
    # Convert detection dataset into classification dataset, with one directory per class

    path = Path(path)  # images dir
    shutil.rmtree(path / 'classifier') if (path / 'classifier').is_dir() else None  # remove existing
    files = list(path.rglob('*.*'))
    n = len(files)  # number of files
    for im_file in tqdm(files, total=n):
        if im_file.suffix[1:] in img_formats:
            # image
            im = cv2.imread(str(im_file))[..., ::-1]  # BGR to RGB
            h, w = im.shape[:2]

            # labels
            lb_file = Path(img2label_paths([str(im_file)])[0])
            if Path(lb_file).exists():
                with open(lb_file, 'r') as f:
                    lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels

                for j, x in enumerate(lb):
                    c = int(x[0])  # class
                    f = (path / 'classifier') / f'{c}' / f'{path.stem}_{im_file.stem}_{j}.jpg'  # new filename
                    if not f.parent.is_dir():
                        f.parent.mkdir(parents=True)

                    b = x[1:] * [w, h, w, h]  # box
                    # b[2:] = b[2:].max()  # rectangle to square
                    b[2:] = b[2:] * 1.2 + 3  # pad
                    b = xywh2xyxy(b.reshape(-1, 4)).ravel().astype(np.int32)

                    b[[0, 2]] = np.clip(b[[0, 2]], 0, w)  # clip boxes outside of image
                    b[[1, 3]] = np.clip(b[[1, 3]], 0, h)
                    assert cv2.imwrite(str(f), im[b[1]:b[3], b[0]:b[2]]), f'box failure in {f}'


def autosplit(path='../coco128', weights=(0.9, 0.1, 0.0), annotated_only=False):
    """ Autosplit a dataset into train/val/test splits and save path/autosplit_*.txt files
    Usage: from utils.datasets import *; autosplit('../coco128')
    Arguments
        path:           Path to images directory
        weights:        Train, val, test weights (list)
        annotated_only: Only use images with an annotated txt file
    """
    path = Path(path)  # images dir
    files = sum([list(path.rglob(f"*.{img_ext}")) for img_ext in img_formats], [])  # image files only
    n = len(files)  # number of files
    indices = random.choices([0, 1, 2], weights=weights, k=n)  # assign each image to a split

    txt = ['autosplit_train.txt', 'autosplit_val.txt', 'autosplit_test.txt']  # 3 txt files
    [(path / x).unlink() for x in txt if (path / x).exists()]  # remove existing

    print(f'Autosplitting images from {path}' + ', using *.txt labeled images only' * annotated_only)
    for i, img in tqdm(zip(indices, files), total=n):
        if not annotated_only or Path(img2label_paths([str(img)])[0]).exists():  # check label
            with open(path / txt[i], 'a') as f:
                f.write(str(img) + '\n')  # add image to txt file
