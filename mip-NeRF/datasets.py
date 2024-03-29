import os
from os import path
import json
import numpy as np
import cv2
from PIL import Image
import jittor
from ray_utils import Rays, convert_to_ndc, namedtuple_map
from pose_utils import normalize, look_at, poses_avg, recenter_poses, to_float, generate_spiral_cam_to_world, generate_spherical_cam_to_world, flatten
from jittor.dataset import Dataset
jittor.flags.use_cuda = 1


def get_dataset(dataset_name, base_dir, split, factor=4):
    d = dataset_dict[dataset_name](base_dir, split, factor=factor)
    return d


def get_dataloader(dataset_name, base_dir, split, factor=4, batch_size=None, shuffle=True):
    d = get_dataset(dataset_name, base_dir, split, factor)
    # make the batchsize height*width, so that one "batch" from the dataloader corresponds to one
    # image used to render a video, and don't shuffle dataset
    if split == "render":
        batch_size = d.w * d.h
        shuffle = False
    loader = d.set_attrs(batch_size=batch_size, shuffle=shuffle)
    loader.h = d.h
    loader.w = d.w
    print("height: "+str(loader.h))
    print("width: "+str(loader.w))
    loader.near = d.near
    loader.far = d.far
    return loader


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


class NeRFDataset(Dataset):
    def __init__(self, base_dir, split, spherify=False, near=2, far=6, white_bkgd=False, factor=1, n_poses=120, radius=None, radii=None, h=None, w=None):
        super().__init__()
        self.base_dir = base_dir
        self.split = split
        self.spherify = spherify
        self.near = near
        self.far = far
        self.white_bkgd = white_bkgd
        self.factor = factor
        self.n_poses = n_poses
        self.n_poses_copy = n_poses
        self.radius = radius
        self.radii = radii
        self.h = h
        self.w = w
        self.rays = None
        self.images = None

        self.load()

    def load(self):
        if self.split == "render":
            self.generate_render_rays()
        else:
            self.generate_training_rays()

        self.flatten_to_jittor()
        print('Done')
        print()

    def generate_training_poses(self):
        """
        Generate training poses, datasets should implement this function to load the proper data from disk.
        Should initialize self.h, self.w, self.focal, self.cam_to_world, and self.images
        """
        raise ValueError('no generate_training_poses(self).')

    def generate_render_poses(self):
        """
        Generate arbitrary poses (views)
        """
        self.focal = 1200
        self.n_poses = self.n_poses_copy
        if self.spherify:
            self.generate_spherical_poses(self.n_poses)
        else:
            self.generate_spiral_poses(self.n_poses)

    def generate_spherical_poses(self, n_poses=120):
        self.poses = generate_spherical_cam_to_world(self.radius, n_poses)
        self.cam_to_world = self.poses[:, :3, :4]

    def generate_spiral_poses(self, n_poses=120):
        self.cam_to_world = generate_spiral_cam_to_world(self.radii, self.focal, n_poses)

    def generate_training_rays(self):
        """
        Generates rays to train mip-NeRF
        """
        print("Loading Training Poses")
        self.generate_training_poses()
        print("Generating rays")
        self.generate_rays()

    def generate_render_rays(self):
        """
        Generates rays used to render a video using a trained mip-NeRF
        """
        print("Generating Render Poses")
        self.generate_render_poses()
        print("Generating rays")
        self.generate_rays()

    def generate_rays(self):
        """Computes rays using a General Pinhole Camera Model
        Assumes self.h, self.w, self.focal, and self.cam_to_world exist
        """
        x, y = np.meshgrid(
            np.arange(self.w, dtype=np.float32),  # X-Axis (columns)
            np.arange(self.h, dtype=np.float32),  # Y-Axis (rows)
            indexing='xy')
        camera_directions = np.stack(
            [-(x - self.w * 0.5 + 0.5) / self.focal,
             -(y - self.h * 0.5 + 0.5) / self.focal,
             np.ones_like(x)],
            axis=-1)
        # Rotate ray directions from camera frame to the world frame
        directions = ((camera_directions[None, ..., None, :] * self.cam_to_world[:, None, None, :3, :3]).sum(axis=-1))  # Translate camera frame's origin to the world frame
        origins = np.broadcast_to(self.cam_to_world[:, None, None, :3, -1], directions.shape)
        viewdirs = directions / np.linalg.norm(directions, axis=-1, keepdims=True)

        # Distance from each unit-norm direction vector to its x-axis neighbor
        dx = np.sqrt(np.sum((directions[:, :-1, :, :] - directions[:, 1:, :, :]) ** 2, -1))
        dx = np.concatenate([dx, dx[:, -2:-1, :]], 1)

        # Cut the distance in half, and then round it out so that it's
        # halfway between inscribed by / circumscribed about the pixel.
        radii = dx[..., None] * 2 / np.sqrt(12)

        ones = np.ones_like(origins[..., :1])

        self.rays = Rays(
            origins=origins,
            directions=directions,
            viewdirs=viewdirs,
            radii=radii,
            lossmult=ones,
            near=ones * self.near,
            far=ones * self.far)

    def flatten_to_jittor(self):
        if self.rays is not None:
            self.rays = namedtuple_map(lambda r: jittor.float32(r).reshape([-1, r.shape[-1]]), self.rays)
        if self.images is not None:
            self.images = jittor.float32(self.images.reshape([-1, 3]))

    def ray_to_device(self, rays):
        return namedtuple_map(lambda r: r, rays)

    def __getitem__(self, i):
        ray = namedtuple_map(lambda r: r[i], self.rays)
        if self.split == "render":
            # render rays
            return ray  # Don't put on device, will batch it using config.chunks in mipNeRF.render_image() function
        else:
            # training rays
            pixel = self.images[i]  # Don't put pixel on device yet, waste of space
            return self.ray_to_device(ray), pixel

    def __len__(self):
        if self.split == "render":
            return self.rays[0].shape[0]
        else:
            return len(self.images)


class Multicam(NeRFDataset):
    def __init__(self, base_dir, split, factor=1, spherify=True, white_bkgd=True, near=2, far=6, radius=4, radii=1, h=800, w=800):
        pass

    def generate_training_poses(self):
        pass

    def generate_rays(self):
        pass


class Blender(NeRFDataset):
    """Blender Dataset."""
    def __init__(self, base_dir, split, factor=1, spherify=False, white_bkgd=True, near=2, far=6, radius=4, radii=1, h=800, w=800):
        super().__init__(base_dir, split, factor=factor, spherify=spherify, near=near, far=far, white_bkgd=white_bkgd, radius=radius, radii=radii, h=h, w=w)

    def generate_training_poses(self):
        """Load data from disk"""
        split_dir = self.split
        with open(path.join(self.base_dir, 'transforms_{}.json'.format(split_dir)), 'r') as fp:
            meta = json.load(fp)
        images = []
        cams = []
        for i in range(len(meta['frames'])):
            frame = meta['frames'][i]
            fname = os.path.join(self.base_dir, frame['file_path'] + '.png')
            with open(fname, 'rb') as imgin:
                image = np.array(Image.open(imgin), dtype=np.float32) / 255.
                if self.factor >= 2:
                    [halfres_h, halfres_w] = [hw for hw in image.shape[:2]]
                    image = cv2.resize(
                        image, (halfres_w, halfres_h), interpolation=cv2.INTER_AREA)
            cams.append(np.array(frame['transform_matrix'], dtype=np.float32))
            images.append(image)
        self.images = np.stack(np.array(images), axis=0)
        if self.white_bkgd:
            self.images = (
                    self.images[..., :3] * self.images[..., -1:] +
                    (1. - self.images[..., -1:]))
        else:
            self.images = self.images[..., :3]
        self.h, self.w = self.images.shape[1:3]
        self.cam_to_world = np.stack(cams, axis=0)
        camera_angle_x = float(meta['camera_angle_x'])
        self.focal = .5 * self.w / np.tan(.5 * camera_angle_x)
        self.n_poses = self.images.shape[0]


class LLFF(NeRFDataset):
    def __init__(self, base_dir, split, factor=4, spherify=False, near=0, far=1, white_bkgd=False):
        pass

    def generate_training_poses(self):
        pass

    def generate_render_poses(self):
        pass

    def generate_training_rays(self):
        pass

    def generate_spherical_poses(self, n_poses=120):
        pass

    def generate_spiral_poses(self, n_poses=120):
        pass

    def generate_rays(self):
        pass


dataset_dict = {
    'blender': Blender,
    'llff': LLFF,
    'multicam': Multicam,
}
