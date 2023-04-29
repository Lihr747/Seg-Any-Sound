import os
import time
import random
import shutil
import torch
import numpy as np
import argparse
import logging

from config import cfg
from dataloader import S4Dataset

from utils import pyutils
from utils.utility import logger, mask_iou, Eval_Fmeasure, save_mask
from utils.system import setup_logging
import pdb
from model import AudioCLIP
###########################################
# Load AudioCLIP
# optional: /home/yujr/workstation/Audio-Visual-Seg/pretrained_backbones/AudioCLIP-Partial-Training.pt
aclp = AudioCLIP(pretrained='pretrained_backbones/AudioCLIP-Full-Training.pt')
aclp.eval()
visual_encoder = aclp.visual
audio_encoder = aclp.audio
###########################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_batch_size", default=1, type=int)

    parser.add_argument("--save_pred_mask", action='store_true', default=False, help="save predited masks or not")
    parser.add_argument('--log_dir', default='../../avsbench_data/save_masks/aclp_s4_logs', type=str)

    # Test data
    args = parser.parse_args()
    split = 'test'
    test_dataset = S4Dataset(split)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                        batch_size=args.test_batch_size,
                                                        shuffle=False,
                                                        num_workers=args.num_workers,
                                                        pin_memory=True)

    avg_meter_miou = pyutils.AverageMeter('miou')
    avg_meter_F = pyutils.AverageMeter('F_score')

    with torch.no_grad():
        for n_iter, batch_data in enumerate(test_dataloader):
            imgs, audio, mask, category_list, video_name_list = batch_data # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 1, 1, 224, 224]
            # imgs is transformed !!! NEED add more preprocess to process for DINO

            imgs = imgs.cuda()
            audio = audio.cuda()
            mask = mask.cuda()
            B, frame, C, H, W = imgs.shape
            imgs = imgs.view(B*frame, C, H, W)
            mask = mask.view(B*frame, H, W)
            audio = audio.view(-1, audio.shape[2], audio.shape[3])

            [x1, x2, x3, x4, last_feat], x_final = visual_encoder(imgs)
            audio_feature = audio_encoder(audio)

            ###############################
            # TODO: get predict mask [bs*5, 1, 224, 224]
            predict_mask = None
            ###############################
            
            mask_save_path = os.path.join(args.log_dir, 'pred_masks')
            save_mask(predict_mask, mask_save_path, category_list, video_name_list, vis_raw_img=True, raw_img_path=os.path.join(cfg.DATA.DIR_IMG, split))

            miou = mask_iou(predict_mask, mask)
            avg_meter_miou.add({'miou': miou})
            F_score = Eval_Fmeasure(predict_mask, mask, args.log_dir)
            avg_meter_F.add({'F_score': F_score})
            print('n_iter: {}, iou: {}, F_score: {}'.format(n_iter, miou, F_score))
            # /home/yujr/workstation/Audio-Visual-Seg/avsbench_data/train_logs/aclp_s4_logs/S4_train_fully_audiocliprealfpn_visual_training_Adam0.0001_lr_mult.sh_20230331-064343/checkpoints/S4_train_fully_audiocliprealfpn_visual_training_Adam0.0001_lr_mult.sh_best.pth

        miou = (avg_meter_miou.pop('miou'))
        F_score = (avg_meter_F.pop('F_score'))
        print('test miou:', miou.item())
        print('test F_score:', F_score)
        logger.info('test miou: {}, F_score: {}'.format(miou.item(), F_score))












