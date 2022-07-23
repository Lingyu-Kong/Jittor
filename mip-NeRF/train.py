import os.path
import shutil
from config import get_config
from scheduler import MipLRDecay
from loss import NeRFLoss, mse_to_psnr
from model import MipNeRF
import jittor
import jittor.optim as optim
from os import path
from datasets import get_dataloader, cycle
import numpy as np
from tqdm import tqdm
from ray_utils import Rays
jittor.flags.use_cuda = 1


def train_model(config):
    model_save_path = path.join(config.log_dir, "model.pt")
    optimizer_save_path = path.join(config.log_dir, "optim.pt")

    data = iter(cycle(get_dataloader(dataset_name=config.dataset_name, base_dir=config.base_dir, split="train", factor=config.factor, batch_size=config.batch_size, shuffle=True)))
    eval_data = None
    if config.do_eval:
        eval_data = iter(cycle(get_dataloader(dataset_name=config.dataset_name, base_dir=config.base_dir, split="test", factor=config.factor, batch_size=config.batch_size, shuffle=True)))

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
        viewdirs_max_deg=config.viewdirs_max_deg
    )
    optimizer = optim.AdamW(model.parameters(), lr=config.lr_init, weight_decay=config.weight_decay)
    if config.continue_training:
        model.load_state_dict(jittor.load(model_save_path))
        optimizer.load_state_dict(jittor.load(optimizer_save_path))

    scheduler = MipLRDecay(optimizer, lr_init=config.lr_init, lr_final=config.lr_final, max_steps=config.max_steps, lr_delay_steps=config.lr_delay_steps, lr_delay_mult=config.lr_delay_mult)
    loss_func = NeRFLoss(config.coarse_weight_decay)
    model.train()

    os.makedirs(config.log_dir, exist_ok=True)
    shutil.rmtree(path.join(config.log_dir, 'train'), ignore_errors=True)

    current_max_psnr = 0

    for step in tqdm(range(0, config.max_steps)):
        rays, pixels = next(data)

        rays = Rays(origins=rays[0],
                    directions=rays[1],
                    viewdirs=rays[2],
                    radii=rays[3],
                    lossmult=rays[4],
                    near=rays[5],
                    far=rays[6])

        comp_rgb, _, _ = model(rays)

        # Compute loss and update model weights.
        loss_val, psnr = loss_func(comp_rgb, pixels, rays.lossmult)
        optimizer.zero_grad()
        optimizer.backward(loss_val)
        optimizer.step()
        scheduler.step()

        psnr = psnr.detach().numpy()

        if step % config.save_every == 0:
            tmp_psnr = None
            if eval_data:
                del rays
                del pixels
                psnr = eval_model(config, model, eval_data)
                psnr = psnr.detach().numpy()
                psnr = np.max(psnr, axis=0).item()       
                tmp_psnr = psnr
                print("when validating, psnr is "+str(psnr))

            if tmp_psnr > current_max_psnr:
                current_max_psnr = tmp_psnr
                jittor.save(model.state_dict(), model_save_path)
                jittor.save(optimizer.state_dict(), optimizer_save_path)

    # jittor.save(model.state_dict(), model_save_path)
    # jittor.save(optimizer.state_dict(), optimizer_save_path)


def eval_model(config, model, data):
    model.eval()
    rays, pixels = next(data)
    rays = Rays(origins=rays[0],
                    directions=rays[1],
                    viewdirs=rays[2],
                    radii=rays[3],
                    lossmult=rays[4],
                    near=rays[5],
                    far=rays[6])
    with jittor.no_grad():
        comp_rgb, _, _ = model(rays)
    model.train()
    return jittor.float32([mse_to_psnr(jittor.mean((rgb - pixels[..., :3])**2)) for rgb in comp_rgb])


if __name__ == "__main__":
    config = get_config()
    train_model(config)
