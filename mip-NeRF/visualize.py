import imp
import json
import jittor
from os import path
from config import get_config
from model import MipNeRF
import imageio
from datasets import get_dataloader, cycle
from tqdm import tqdm
from pose_utils import visualize_depth, visualize_normals, to8b
from ray_utils import Rays
from PIL import Image
import numpy as np
jittor.flags.use_cuda = 1

category = "Easyship"

def visualize(config):
    #data = get_dataloader(config.dataset_name, config.base_dir, split="render", factor=config.factor, shuffle=False)
    data = iter(cycle(get_dataloader(dataset_name=config.dataset_name, base_dir=config.base_dir, split="test", factor=config.factor, batch_size=800, shuffle=False)))
    model = MipNeRF(
        use_viewdirs=config.use_viewdirs,
        randomized=config.randomized,
        ray_shape=config.ray_shape,
        white_bkgd=config.white_bkgd,
        num_levels=config.num_levels,
        num_samples=config.num_samples,
        hidden=config.hidden,
        density_noise=config.density_noise,
        density_bias=config.density_bias,
        rgb_padding=config.rgb_padding,
        resample_padding=config.resample_padding,
        min_deg=config.min_deg,
        max_deg=config.max_deg,
        viewdirs_min_deg=config.viewdirs_min_deg,
        viewdirs_max_deg=config.viewdirs_max_deg,
    )
    model.load_state_dict(jittor.load(config.model_weight_path))
    model.eval()

    #print("Generating Video using", len(data), "different view points")
    rgb_frames = []
    if config.visualize_depth:
        depth_frames = []
    if config.visualize_normals:
        normal_frames = []

    # with open("first_test.json", "r") as f:
    #     dic = json.load(f)

    #for ray in tqdm(data):
    for i in tqdm(range(0,30)):
        imgs = []
        #imgs = dic["img"]
        with jittor.no_grad():
            for j in tqdm(range(0, 800)):
                ray, _ = next(data)
                #print(ray)
                ray = Rays(origins=ray[0],
                            directions=ray[1],
                            viewdirs=ray[2],
                            radii=ray[3],
                            lossmult=ray[4],
                            near=ray[5],
                            far=ray[6])
                img, dist, acc = model(ray)
                img = img.numpy()[0][:][:].tolist()
                imgs.append(img)

        with open("first_test.json", "w") as f:
            dic = {"img" : imgs}
            json.dump(dic, f)
        
        print("save "+category+"_r_"+str(i)+".png")
        save_path = "./log/"+category+"/"+category+"_r_"+str(i)+".png"
        save_img(save_path, np.array(imgs))


    #     if config.visualize_depth:
    #         depth_frames.append(to8b(visualize_depth(dist, acc, data.near, data.far)))
    #     if config.visualize_normals:
    #         normal_frames.append(to8b(visualize_normals(dist, acc)))

    # imageio.mimwrite(path.join(config.log_dir, "video.mp4"), rgb_frames, fps=30, quality=10, codecs="hvec")
    # if config.visualize_depth:
    #     imageio.mimwrite(path.join(config.log_dir, "depth.mp4"), depth_frames, fps=30, quality=10, codecs="hvec")
    # if config.visualize_normals:
    #     imageio.mimwrite(path.join(config.log_dir, "normals.mp4"), normal_frames, fps=30, quality=10, codecs="hvec")

def save_img(path, img):
    if isinstance(img, np.ndarray):
        print("np.ndarray")
        ndarr = (img*255+0.5).clip(0, 255).astype('uint8')
    elif isinstance(img, jittor.Var):
        print("jittor.Var")
        ndarr = (img*255+0.5).clamp(0, 255).uint8().numpy()
    im = Image.fromarray(ndarr)
    im.save(path)

if __name__ == "__main__":
    config = get_config()
    visualize(config)
