import os
import time
import random
import shutil
import torch
import numpy as np
import argparse
import logging
from PIL import Image, ImageDraw
from torchvision import transforms
from config import cfg
from dataloader import S4Dataset
from torch.nn.functional import interpolate

from utils import pyutils
from utils.utility import logger, mask_iou, Eval_Fmeasure, save_mask
from utils.system import setup_logging
import pdb
from model import AudioCLIP

import cv2
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
###########################################
# Load AudioCLIP
# optional: /home/yujr/workstation/Audio-Visual-Seg/pretrained_backbones/AudioCLIP-Partial-Training.pt
aclp = AudioCLIP(pretrained='../../pretrained_backbones/AudioCLIP-Full-Training.pt')
sam = sam_model_registry["default"](checkpoint="../../pretrained_backbones/sam_vit_h_4b8939.pth")
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)
aclp.eval()
aclp.to(device=device)
visual_encoder = aclp.visual
audio_encoder = aclp.audio

def myfn(batch_list):
    image_paths = []
    for i in range(len(batch_list)):
        image_paths += batch_list[i][0]
    audio = torch.stack([item[1] for item in batch_list])
    mask = torch.stack([item[2] for item in batch_list])
    category = [item[3] for item in batch_list]
    video_name = [item[4] for item in batch_list]

    return image_paths, audio, mask, category, video_name



def load_img(image_path):
    transform = transforms.Compose([ 
                    transforms.ToTensor(), 
                    transforms.Resize(cfg.DATA.IMAGE_SIZE, interpolation=Image.BICUBIC),
                    transforms.CenterCrop(cfg.DATA.IMAGE_SIZE),
                    transforms.Normalize(cfg.DATA.IMAGE_MEAN, cfg.DATA.IMAGE_STD)
                    ])
    img_PIL = Image.open(image_path).convert(mode='RGB')
    image = img_PIL.convert(mode='RGB')
    img_tensor = transform(image)
    return img_tensor


def get_mask_embd(mask, image_feature):
    mask = mask.reshape(1, 1, H, W) # [1, 1, 224, 224]
    mask = interpolate(mask, [7, 7]) # [1, 1, 7, 7]
    mask = mask.repeat((1, 1024, 1, 1)) # [1, 1024, 7, 7]
    mask_feature = mask * image_feature
    mask_embed = mask_feature.mean([-1, -2])
    return mask_embed.squeeze(0)

def retriev(masks_embed, audio_embed):
    masks_embed /= masks_embed.norm(dim=-1, keepdim=True)
    audio_embed /= audio_embed.norm(dim=-1, keepdim=True)

    probs = 100. * masks_embed @ audio_embed.T
    scores = probs[:, 0].softmax(dim=0)
 
    return scores

def get_indices_of_values_above_threshold(values, threshold):
    return [i for i, v in enumerate(values) if v > threshold]

###########################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_batch_size", default=1, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument('--save_path', default='../../avsbench_data/save_masks/s4_aclp_image_retrival_pooling', type=str)

    # Test data
    args = parser.parse_args()
    split = 'test'
    test_dataset = S4Dataset(split)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                        batch_size=args.test_batch_size,
                                                        shuffle=False,
                                                        num_workers=args.num_workers,
                                                        collate_fn=myfn,
                                                        pin_memory=True)

    avg_meter_miou = pyutils.AverageMeter('miou')
    avg_meter_F = pyutils.AverageMeter('F_score')

    with torch.no_grad():
        for n_iter, batch_data in enumerate(test_dataloader):
            image_paths, audio, masks, category_list, video_name_list = batch_data # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 1, 1, 224, 224]
            # imgs is transformed !!! NEED add more preprocess to process for DINO
            audio = audio.to(device=device)
            masks = masks.to(device=device)
            B, frame, C, H, W = masks.shape
            masks = masks.view(B*frame, H, W)
            audio = audio.view(-1, audio.shape[2], audio.shape[3])
            audio_embeds = audio_encoder(audio)



            pred_masks = []
            for i in range(len(image_paths)):
                image_path = image_paths[i]
                audio_embed = audio_embeds[i].reshape(1, -1) 

                image_tensor = load_img(image_path).reshape(1, 3, H, W).to(device) 
                features_list, last_embed = visual_encoder(image_tensor)
                image_feature = features_list[4]  # [1, 1024, 7, 7]

                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                generate_results = mask_generator.generate(image)
                image = Image.open(image_path)

                masks_list = []
                masks_embed_list = []
                for result in generate_results:
                    mask = result['segmentation']
                    mask = torch.from_numpy(mask).to(device).to(torch.float32)
                    mask_embed = get_mask_embd(mask, image_feature)
                    mask_embed_sum = mask_embed.sum()
                    if mask_embed_sum.item() == 0:
                        continue
                    masks_list.append(mask)
                    masks_embed_list.append(mask_embed)

                masks_embed = torch.stack(masks_embed_list).to(device)
                scores = retriev(masks_embed, audio_embed)
                indices = get_indices_of_values_above_threshold(scores, 0.05)

                pred_mask = torch.zeros([H, W]).to(device)
                for indice in indices:
                    this_mask = masks_list[indice]
                    pred_mask = 1 - (1 - pred_mask) * (1 - this_mask)
                pred_masks.append(pred_mask)
            pred_masks = torch.stack(pred_masks).to(device)
            
            ###############################
            # TODO: get predict mask [bs*5, 1, 224, 224]
            # predict_mask = None
            ###############################
            
            save_mask(pred_masks, masks, args.save_path, category_list, video_name_list, vis_raw_img=True, raw_img_path=os.path.join(cfg.DATA.DIR_IMG, split))

            miou = mask_iou(pred_masks, masks)
            avg_meter_miou.add({'miou': miou})
            F_score = Eval_Fmeasure(pred_masks, masks, args.save_path)
            avg_meter_F.add({'F_score': F_score})
            print('n_iter: {}, iou: {}, F_score: {}'.format(n_iter, miou, F_score))
            # /home/yujr/workstation/Audio-Visual-Seg/avsbench_data/train_logs/aclp_s4_logs/S4_train_fully_audiocliprealfpn_visual_training_Adam0.0001_lr_mult.sh_20230331-064343/checkpoints/S4_train_fully_audiocliprealfpn_visual_training_Adam0.0001_lr_mult.sh_best.pth

        miou = (avg_meter_miou.pop('miou'))
        F_score = (avg_meter_F.pop('F_score'))
        print('test miou:', miou.item())
        print('test F_score:', F_score)
        logger.info('test miou: {}, F_score: {}'.format(miou.item(), F_score))












