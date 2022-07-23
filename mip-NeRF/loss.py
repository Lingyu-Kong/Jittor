import jittor
import numpy
jittor.flags.use_cuda = 1

class NeRFLoss(jittor.nn.Module):
    def __init__(self, coarse_weight_decay=0.1):
        super(NeRFLoss, self).__init__()
        self.coarse_weight_decay = coarse_weight_decay

    def execute(self, input, target, mask):
        losses = []
        psnrs = []
        for rgb in input:
            mse = (mask * ((rgb - target[..., :3]) ** 2)).sum() / mask.sum()
            losses.append(mse)
            with jittor.no_grad():
                psnrs.append(mse_to_psnr(mse))
        losses = jittor.stack(losses)
        loss = self.coarse_weight_decay * jittor.sum(losses[:-1]) + losses[-1]
        return loss, jittor.float32(psnrs)


def mse_to_psnr(mse):
    return -10.0 * numpy.log10(mse)
