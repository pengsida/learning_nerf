import torch.utils.data as data
from lib.utils import base_utils
from PIL import Image
import numpy as np
import json
import os
import imageio
import cv2
from lib.config import cfg
from plyfile import PlyData


class Dataset(data.Dataset):
    def __init__(self, **kwargs):
        super(Dataset, self).__init__()
        self.data_root = kwargs['data_root']
        self.split = kwargs['split']
        self.input_ratio = kwargs['input_ratio']

        annots = np.load(os.path.join(self.data_root, 'annots.npy'), allow_pickle=True).item()
        self.cams = annots['cams']
        num_cams = len(self.cams['K'])
        start_cam, end_cam, skip_cam = kwargs['cameras'][0], kwargs['cameras'][1], kwargs['cameras'][2]
        end_cam = num_cams if end_cam == -1 else end_cam
        self.render_cameras = np.arange(num_cams)[start_cam:end_cam:skip_cam]

        num_frames = len(annots['ims'])
        start_frames, end_frames, skip_frames = kwargs['frames'][0], kwargs['frames'][1], kwargs['frames'][2]
        end_frames = num_frames if end_frames == -1 else end_frames

        # recording items and compute the world bbox
        self.items = []
        self.bboxs = {}

        t_start, t_end, t_skip = cfg.train_dataset.frames
        t_end = num_frames if t_end == -1 else t_end
        training_frames = np.arange(num_frames)[t_start:t_end:t_skip].tolist()
        for frame_index in np.arange(num_frames)[start_frames:end_frames:skip_frames]:
            latent_index = training_frames.index(frame_index)
            for camera_index in self.render_cameras:
                item = {}
                item.update({'img_path': os.path.join(self.data_root, annots['ims'][frame_index]['ims'][camera_index])})
                item.update({'frame_index': frame_index, 'camera_index': camera_index, 'latent_index': latent_index})
                self.items.append(item)
            vertices_dir = 'new_vertices'
            vertices_start = 1 if '313' in self.data_root or '315' in self.data_root else 0
            vertices = np.load(os.path.join(self.data_root, vertices_dir, '{}.npy'.format(frame_index + vertices_start)))
            self.bboxs.update({frame_index: np.concatenate([np.min(vertices, axis=0) - 0.05, np.max(vertices, axis=0) + 0.05])})
        bboxs = np.stack([self.bboxs[k] for k in self.bboxs])
        self.wbbox = np.concatenate([np.min(bboxs[:, :3], axis=0), np.max(bboxs[:, 3:6], axis=0)])
        cfg.aabb = self.wbbox.tolist()
        self.caches = {}


    def get_mask(self, item):
        img_path = '/'.join(item['img_path'].split('/')[3:])
        msk_path = os.path.join(self.data_root, 'mask_cihp',
                                img_path)[:-4] + '.png'
        if not os.path.exists(msk_path):
            msk_path = os.path.join(self.data_root, 'mask',
                                    img_path)[:-4] + '.png'
        if not os.path.exists(msk_path):
            msk_path = os.path.join(self.data_root, img_path.replace(
                'images', 'mask'))[:-4] + '.png'
        msk_cihp = imageio.imread(msk_path)
        if len(msk_cihp.shape) == 3:
            msk_cihp = msk_cihp[..., 0]
        msk_cihp = (msk_cihp != 0).astype(np.uint8)
        msk = msk_cihp
        orig_msk = msk.copy()

        border = 5
        kernel = np.ones((border, border), np.uint8)
        msk_erode = cv2.erode(msk.copy(), kernel)
        msk_dilate = cv2.dilate(msk.copy(), kernel)
        msk[(msk_dilate - msk_erode) == 1] = 100

        return msk, orig_msk

    def read_data(self, item, index):
        frame_index, camera_index = item['frame_index'], item['camera_index']
        if index in self.caches:
            img, bbox, msk = self.caches[index]
        else:
            img = imageio.imread(item['img_path']).astype(np.float32) / 255.
            H, W = img.shape[:2]

            msk, _ = self.get_mask(item)
            msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)

            D, K = np.array(self.cams['D'][item['camera_index']]), np.array(self.cams['K'])[item['camera_index']]
            img = cv2.undistort(img, K, D)
            msk = cv2.undistort(msk, K, D)

            if self.input_ratio != 1.:
                msk = cv2.resize(msk, None, fx=self.input_ratio, fy=self.input_ratio, interpolation=cv2.INTER_NEAREST)
                img = cv2.resize(img, None, fx=self.input_ratio, fy=self.input_ratio, interpolation=cv2.INTER_AREA)

            img[msk == 0] = 0.

            non_zero = msk.nonzero()
            bbox = [non_zero[1].min() - 1, non_zero[0].min() - 1, non_zero[1].max() + 1, non_zero[0].max() + 1]
            self.caches[index] = (img, bbox, msk)
        K, R, T = np.array(self.cams['K'][camera_index]).copy(), np.array(self.cams['R'][camera_index]), np.array(self.cams['T'][camera_index]) / 1000.
        if self.input_ratio != 1.:
            K[:2] *= self.input_ratio
        return img, np.concatenate([R, T], axis=1), K, bbox, msk

    def __getitem__(self, index):
        item = self.items[index]
        img, ext, ixt, bbox, mask = self.read_data(item, index)
        H, W = img.shape[:2]
        if self.split == 'train':
            fg_num = int(cfg.N_rays * 0.5)
            bg_num = cfg.N_rays - fg_num
            pixel_x = np.random.randint(low=max(int(bbox[0]), 0), high=min(int(bbox[2])+1, W), size=4*fg_num)
            pixel_y = np.random.randint(low=max(int(bbox[1]), 0), high=min(int(bbox[3])+1, H), size=4*fg_num)
            mask_val = mask[pixel_y, pixel_x]
            pixel_x = pixel_x[mask_val == 1][:fg_num]
            pixel_y = pixel_y[mask_val == 1][:fg_num]

            mask = base_utils.get_bound_2d_mask(self.wbbox.reshape((2, 3)), ixt, ext, H, W)
            bbox = base_utils.get_bbox_2d(self.wbbox.reshape((2, 3)), ixt, ext) # min_x, min_y, max_x, max_y
            pixel_x_ = np.random.randint(low=max(int(bbox[0]), 0), high=min(int(bbox[2])+1, W), size=3*bg_num)
            pixel_y_ = np.random.randint(low=max(int(bbox[1]), 0), high=min(int(bbox[3])+1, H), size=3*bg_num)
            mask_val = mask[pixel_y_, pixel_x_]
            pixel_x_ = pixel_x_[mask_val == 1][:bg_num]
            pixel_y_ = pixel_y_[mask_val == 1][:bg_num]
            pixel_x, pixel_y = np.concatenate([pixel_x, pixel_x_]), np.concatenate([pixel_y, pixel_y_])

            rgb = img[pixel_y, pixel_x]
        else:
            rgb = img.reshape(-1, 3)
            pixel_x, pixel_y = np.meshgrid(np.linspace(0, W-1, W), np.linspace(0, H-1, H))
            pixel_x, pixel_y = pixel_x.reshape(-1), pixel_y.reshape(-1)

        c2w_inv = np.eye(4)
        c2w_inv[:3] = ext
        c2w = np.linalg.inv(c2w_inv)

        rays_o = c2w[:3, 3][None].repeat(len(pixel_x), 0)
        rays_d = np.stack([pixel_x, pixel_y, np.ones_like(pixel_x)], axis=-1)
        rays_d = rays_d @ np.linalg.inv(ixt).T @ c2w[:3, :3].T
        rays_d = rays_d / np.linalg.norm(rays_d, axis=-1)[..., None]
        rays_t = np.ones_like(pixel_x)[:, None] * item['latent_index']
        rays = np.concatenate([rays_o, rays_d, rays_t], axis=-1)

        ret = {'rays': rays.astype(np.float32), 'rgb': rgb, 'wbounds': self.wbbox.astype(np.float32)}
        meta = {'H': H, 'W': W, 'item': item}
        ret['meta'] = meta

        return ret

    def __len__(self):
        return len(self.items)
