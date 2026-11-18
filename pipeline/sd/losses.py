"""Task losses for TFG, all taking the clean-prediction latent x0 (scaled SD latent, 1x4x64x64)."""
import torch
import torch.nn.functional as F
import torchvision

from .models import DEVICE, DTYPE


def gram(f):
    b, c, h, w = f.shape
    f = f.reshape(b, c, h * w).float()
    return f @ f.transpose(1, 2) / (c * h * w)


class ClipLoss:
    """CLIP zero-shot / embedding loss on the TAESD-decoded clean prediction."""

    def __init__(self, scorer):
        self.scorer = scorer

    def __call__(self, x0):
        return self.scorer.loss(self.scorer.decode_tiny(x0))


class VGGStyleLoss:
    """Gram-matrix style loss (Gatys et al.) on the TAESD-decoded clean prediction vs a reference image."""

    LAYERS = (3, 8, 15, 22)  # relu1_2, relu2_2, relu3_3, relu4_3
    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __init__(self, decode_tiny, ref_img, size=256):
        vgg = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1).features[:23]
        self.vgg = vgg.to(DEVICE, DTYPE).eval().requires_grad_(False)
        self.decode_tiny, self.size = decode_tiny, size
        with torch.no_grad():
            self.ref = [gram(f) for f in self._feats(ref_img.to(DEVICE))]

    def _feats(self, img):
        x = (img.float() + 1) / 2
        x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", antialias=True)
        x = ((x - self.MEAN.to(x.device)) / self.STD.to(x.device)).to(DTYPE)
        out = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.LAYERS:
                out.append(x)
        return out

    def __call__(self, x0):
        fs = self._feats(self.decode_tiny(x0))
        return sum(F.mse_loss(gram(f), g, reduction="sum") for f, g in zip(fs, self.ref)) * 1e3


class LatentStatsLoss:
    """Match per-channel mean/std and the channel Gram matrix of the latent to a reference latent."""

    def __init__(self, ref_latent):
        r = ref_latent.float()
        self.mean = r.mean(dim=(2, 3))
        self.std = r.std(dim=(2, 3))
        self.g = gram(r)

    def __call__(self, x0):
        x = x0.float()
        return (F.mse_loss(x.mean(dim=(2, 3)), self.mean, reduction="sum")
                + F.mse_loss(x.std(dim=(2, 3)), self.std, reduction="sum")
                + F.mse_loss(gram(x), self.g, reduction="sum"))
