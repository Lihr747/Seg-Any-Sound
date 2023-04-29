import os
from wave import _wave_params
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
import pickle

import cv2
from PIL import Image
from torchvision import transforms
import librosa
from config import cfg
import pdb

def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)

def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)

def load_image_in_PIL_to_Tensor(path, mode='RGB', transform=None):
    img_PIL = Image.open(path).convert(mode)
    if transform:
        img_tensor = transform(img_PIL)
        return img_tensor
    return img_PIL

def load_RGB_raw_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def load_audio_lm(audio_lm_path):
    with open(audio_lm_path, 'rb') as fr:
        audio_log_mel = pickle.load(fr)
    audio_log_mel = audio_log_mel.detach() # [5, 1, 96, 64]
    return audio_log_mel


def load_audio_wav(audio_wav_path, transform=None):
    sr = 48000
    track, _ = librosa.load(audio_wav_path, sr=sr)
    #track = track.detach() # [220500] TODO: => [5, XXXX]
    # 30 of 4925 less than 5s, TODO: padding last 1s to 5s
    MAX_LENGTH = 5 * sr
    if track.shape[0] > MAX_LENGTH:
        track = track[:MAX_LENGTH]
    elif track.shape[0] < MAX_LENGTH:
        second = track.shape[0] // sr
        rest_second = 5 - second
        last_second_feature = track[-sr:]
        track = track[0:second*sr]
        track = np.concatenate((track, np.tile(last_second_feature, rest_second)))
    track = torch.from_numpy(int16_to_float32(float32_to_int16(track))).float()
    track = track.reshape(1, -1)
    track = track.reshape(5, 1, -1) # 5 x 1 x 44100
    return track


class S4Dataset(Dataset):
    """Dataset for single sound source segmentation"""
    def __init__(self, split='train'):
        super(S4Dataset, self).__init__()
        self.split = split
        self.mask_num = 1 if self.split == 'train' else 5
        df_all = pd.read_csv(cfg.DATA.ANNO_CSV, sep=',')
        self.df_split = df_all[df_all['split'] == split]
        print("{}/{} videos are used for {}".format(len(self.df_split), len(df_all), self.split))
        self.categories = set([])
        for df_one_video in self.df_split.iloc:
            category = df_one_video[2]
            category = category.replace('_', ' ')
            category = category.replace('-', ' ')
            self.categories.add(category)
        self.categories = list(self.categories)
        print("category number:", len(self.categories))



        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(cfg.DATA.IMAGE_SIZE, interpolation=Image.BICUBIC),
            transforms.CenterCrop(cfg.DATA.IMAGE_SIZE),
            transforms.Normalize(cfg.DATA.IMAGE_MEAN, cfg.DATA.IMAGE_STD)
        ])
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.audio_transform = transforms.Compose([
            transforms.ToTensor(),
        ])


    def __getitem__(self, index):
        df_one_video = self.df_split.iloc[index]
        video_name, category = df_one_video[0], df_one_video[2]
        img_base_path =  os.path.join(cfg.DATA.DIR_IMG, self.split, category, video_name)
        audio_wav_path = os.path.join(cfg.DATA.DIR_AUDIO, self.split, category, video_name + '.wav')
        mask_base_path = os.path.join(cfg.DATA.DIR_MASK, self.split, category, video_name)
        # audio_log_mel = load_audio_wav(audio_wav_path)
        # audio_lm_tensor = torch.from_numpy(audio_log_mel)
        img_paths, masks = [], []
        for img_id in range(1, 6):
            img_path = os.path.join(img_base_path, "%s_%d.png"%(video_name, img_id))
            img_paths.append(img_path)
        for mask_id in range(1, self.mask_num + 1):
            mask = load_image_in_PIL_to_Tensor(os.path.join(mask_base_path, "%s_%d.png"%(video_name, mask_id)), transform=self.mask_transform, mode='1')
            masks.append(mask)
        audio_tensor = load_audio_wav(audio_wav_path, self.audio_transform)
        masks_tensor = torch.stack(masks, dim=0)

        if self.split == 'train':
            return img_paths, audio_tensor, masks_tensor
        else:
            return img_paths, audio_tensor, masks_tensor, category, video_name


    def __len__(self):
        return len(self.df_split)

    def get_category(self):
        return self.categories




if __name__ == "__main__":
    train_dataset = S4Dataset('train')
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                     batch_size=2,
                                                     shuffle=False,
                                                     num_workers=8,
                                                     pin_memory=True)

    for n_iter, batch_data in enumerate(train_dataloader):
        imgs, audio, mask = batch_data # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 1, 1, 224, 224]
        # imgs, audio, mask, category, video_name = batch_data # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 1, 1, 224, 224]
        pdb.set_trace()
    print('n_iter', n_iter)
    pdb.set_trace()
