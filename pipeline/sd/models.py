"""Model loading for the Stable Diffusion experiments (SD 1.5, fp16 weights only)."""
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, AutoencoderTiny, UNet2DConditionModel
from transformers import CLIPModel, CLIPTextModel, CLIPTokenizer

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
SD_DIR = MODELS / "stable-diffusion-v1-5"
CLIP_DIR = MODELS / "clip-vit-large-patch14"
TAESD_DIR = MODELS / "taesd"

DEVICE = "cuda"
DTYPE = torch.float16

# Linear latent -> RGB projection (standard SD 1.5 preview factors), for z_t previews.
LATENT_RGB = torch.tensor([
    [0.298, 0.207, 0.208],
    [0.187, 0.286, 0.173],
    [-0.158, 0.189, 0.264],
    [-0.184, -0.271, -0.473],
])


def _freeze(m):
    m.eval()
    m.requires_grad_(False)
    return m


class StableDiffusion:
    """Bare SD 1.5 components plus the VP noise schedule (alpha_t, sigma_t)."""

    def __init__(self):
        self.tokenizer = CLIPTokenizer.from_pretrained(SD_DIR, subfolder="tokenizer")
        self.text_encoder = _freeze(CLIPTextModel.from_pretrained(
            SD_DIR, subfolder="text_encoder", variant="fp16", torch_dtype=DTYPE).to(DEVICE))
        self.unet = _freeze(UNet2DConditionModel.from_pretrained(
            SD_DIR, subfolder="unet", variant="fp16", torch_dtype=DTYPE).to(DEVICE))
        self.vae = _freeze(AutoencoderKL.from_pretrained(
            SD_DIR, subfolder="vae", variant="fp16", torch_dtype=DTYPE).to(DEVICE))
        self.vae_scale = self.vae.config.scaling_factor

        # scaled_linear schedule of SD 1.5
        betas = torch.linspace(0.00085 ** 0.5, 0.012 ** 0.5, 1000, dtype=torch.float64) ** 2
        self.alphas_cumprod = torch.cumprod(1.0 - betas, 0).float().to(DEVICE)

    def timesteps(self, n_steps):
        """DDIM 'leading' spacing with steps_offset=1, as in diffusers for SD 1.5."""
        stride = 1000 // n_steps
        ts = [i * stride + 1 for i in range(n_steps)][::-1]
        prev = ts[1:] + [None]  # None -> final step goes to the clean sample
        return list(zip(ts, prev))

    def alpha_sigma(self, t):
        if t is None:
            return 1.0, 0.0
        ac = self.alphas_cumprod[t].item()
        return ac ** 0.5, (1 - ac) ** 0.5

    @torch.no_grad()
    def encode_text(self, prompts):
        ids = self.tokenizer(prompts, padding="max_length", max_length=self.tokenizer.model_max_length,
                             truncation=True, return_tensors="pt").input_ids.to(DEVICE)
        return self.text_encoder(ids)[0]

    def eps(self, z, t, emb):
        return self.unet(z, t, encoder_hidden_states=emb).sample

    @torch.no_grad()
    def decode(self, z0, size=None):
        img = self.vae.decode(z0.to(DTYPE) / self.vae_scale).sample.float().clamp(-1, 1)
        if size is not None:
            img = F.interpolate(img, size=(size, size), mode="bilinear", antialias=True)
        return img

    @staticmethod
    def latent_preview(z, size=64):
        rgb = torch.einsum("bchw,cr->brhw", z.float().cpu(), LATENT_RGB)
        rgb = rgb.clamp(-1, 1)
        if size != rgb.shape[-1]:
            rgb = F.interpolate(rgb, size=(size, size), mode="bilinear", antialias=True)
        return rgb


class ClipStyleScorer:
    """Zero-shot CLIP classifier over a pair of texts; class 0 is the target style.

    Used both as the TFG task loss (on the decoded clean prediction) and as a metric.
    """

    MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
    STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)

    def __init__(self, texts, n_aug=4):
        self.model = _freeze(CLIPModel.from_pretrained(CLIP_DIR, torch_dtype=DTYPE).to(DEVICE))
        tok = CLIPTokenizer.from_pretrained(CLIP_DIR)
        with torch.no_grad():
            ids = tok(texts, padding=True, return_tensors="pt").input_ids.to(DEVICE)
            tf = self.model.get_text_features(input_ids=ids).float()
        self.text_feats = tf / tf.norm(dim=-1, keepdim=True)
        self.scale = self.model.logit_scale.exp().float().item()
        self.n_aug = n_aug
        self.taesd = _freeze(AutoencoderTiny.from_pretrained(TAESD_DIR, torch_dtype=DTYPE).to(DEVICE))

    def _prep(self, img, augment):
        x = (img.float() + 1) / 2
        if augment and self.n_aug > 1:
            views = []
            H = x.shape[-1]
            for _ in range(self.n_aug):
                s = int(H * float(torch.empty(1).uniform_(0.75, 1.0)))
                i, j = torch.randint(0, H - s + 1, (2,)).tolist()
                views.append(x[..., i:i + s, j:j + s])
            x = torch.cat([F.interpolate(v, size=(224, 224), mode="bilinear", antialias=True) for v in views])
        else:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", antialias=True)
        return ((x - self.MEAN.to(x.device)) / self.STD.to(x.device)).to(DTYPE)

    def logits(self, img, augment=False):
        f = self.model.get_image_features(pixel_values=self._prep(img, augment)).float()
        f = f / f.norm(dim=-1, keepdim=True)
        return self.scale * f @ self.text_feats.T

    loss_mode = "softmax"   # softmax | text_cos | ref_cos
    ref_feats = None        # CLIP image embedding of a reference style image (ref_cos)

    def image_feats(self, img, augment=False):
        f = self.model.get_image_features(pixel_values=self._prep(img, augment)).float()
        return f / f.norm(dim=-1, keepdim=True)

    def set_reference(self, img):
        with torch.no_grad():
            self.ref_feats = self.image_feats(img.to(DEVICE)).mean(0, keepdim=True)
            self.ref_feats = self.ref_feats / self.ref_feats.norm(dim=-1, keepdim=True)

    def loss(self, img, augment=True):
        if self.loss_mode == "softmax":
            return -torch.log_softmax(self.logits(img, augment), dim=-1)[:, 0].mean()
        f = self.image_feats(img, augment)
        target = self.text_feats[0:1] if self.loss_mode == "text_cos" else self.ref_feats
        return (1 - f @ target.T).mean()

    @torch.no_grad()
    def prob(self, img):
        return torch.softmax(self.logits(img), dim=-1)[:, 0]

    def decode_tiny(self, z0):
        """Differentiable TAESD decode of a (scaled) SD latent -> image in [-1, 1]."""
        return self.taesd.decode(z0.to(DTYPE)).sample
