import comfy
from comfy import samplers
from comfy_extras import nodes_custom_sampler
import nodes

import torch
import math
import numpy as np

SCHEDULERS = samplers.KSampler.SCHEDULERS + ['AYS SDXL', 'AYS SD1', 'AYS SVD', "GITS[coeff=1.2]"]

def prepare_noise(latent_image, seed, noise_device="cpu", incremental_seed_mode="comfy"):
    """
    Creates random noise given a latent image and a seed.
    """
    latent_size = latent_image.size()
    latent_size_1batch = [1, *latent_size[1:]]

    if incremental_seed_mode == "incremental":
        batch_cnt = latent_size[0]
        latents = []

        for i in range(batch_cnt):
            generator = torch.manual_seed(seed + i) if noise_device == "cpu" else None
            if noise_device != "cpu":
                torch.cuda.manual_seed(seed + i)
            latent = torch.randn(latent_size_1batch, dtype=latent_image.dtype, layout=latent_image.layout,
                                 generator=generator, device=noise_device)
            latents.append(latent)

        return torch.cat(latents, dim=0)
    else:
        generator = torch.manual_seed(seed) if noise_device == "cpu" else None
        if noise_device != "cpu":
            torch.cuda.manual_seed(seed)
        return torch.randn(latent_size, dtype=latent_image.dtype, layout=latent_image.layout,
                           generator=generator, device=noise_device)

class Texturaizer_RandomNoise:
    def __init__(self, seed, mode, incremental_seed_mode):
        device = comfy.model_management.get_torch_device()
        self.seed = seed
        self.noise_device = "cpu" if mode == "CPU" else device
        self.incremental_seed_mode = incremental_seed_mode

    def generate_noise(self, input_latent):
        latent_image = input_latent["samples"]
        noise = prepare_noise(latent_image, self.seed, self.noise_device, self.incremental_seed_mode)
        return noise.cpu()

class GenerateNoise_texturaizer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "noise_mode": (["GPU(=A1111)", "CPU"],),
                "batch_seed_mode": (["incremental", "comfy"],),
            }
        }

    RETURN_TYPES = ("NOISE",)
    FUNCTION = "get_noise"
    CATEGORY = "Texturaizer"

    def get_noise(self, noise_seed, noise_mode, batch_seed_mode):
        return (Texturaizer_RandomNoise(noise_seed, noise_mode, batch_seed_mode),)

class SigmasSelector_texturaizer: 
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                    {"model": ("MODEL",),
                     "scheduler": (SCHEDULERS,),
                     "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "denoise": ("FLOAT", {"default": 1, "min": 0, "max": 1}),
                     },
                }

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "calculate_sigmas"
    CATEGORY = "Texturaizer"

    def calculate_sigmas(self, model, scheduler, steps, denoise):
        if scheduler.startswith('AYS'):
            print("AYS")
            sigmas = nodes.NODE_CLASS_MAPPINGS['AlignYourStepsScheduler']().get_sigmas(scheduler[4:], steps, denoise)
        elif scheduler.startswith('GITS[coeff='):
            print("GITS")
            sigmas = nodes.NODE_CLASS_MAPPINGS['GITSScheduler']().get_sigmas(float(scheduler[11:-1]), steps, denoise)
        else:
            sigmas = nodes_custom_sampler.NODE_CLASS_MAPPINGS['BasicScheduler']().get_sigmas(model, scheduler, steps, denoise)

        return sigmas


class KSamplerAdvanced_texturaizer:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                     "add_noise": ("BOOLEAN", {"default": True, "label_on": "enable", "label_off": "disable"}),
                     "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                     "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.5, "round": 0.01}),
                     "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                     "scheduler": (SCHEDULERS,),
                     "positive": ("CONDITIONING",),
                     "negative": ("CONDITIONING",),
                     "latent_image": ("LATENT",),
                     "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                     "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                     "noise_mode": (["GPU(=A1111)", "CPU"],),
                     "return_with_leftover_noise": ("BOOLEAN", {"default": False, "label_on": "enable", "label_off": "disable"}),
                     "batch_seed_mode": (["incremental", "comfy"],),
                     },
                "optional": {}
                }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "doit"
    CATEGORY = "Texturaizer"

    @staticmethod
    def sample(model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
               start_at_step, end_at_step, noise_mode, return_with_leftover_noise, denoise=1.0, batch_seed_mode="comfy", callback=None):
        force_full_denoise = True

        if return_with_leftover_noise:
            force_full_denoise = False

        disable_noise = not add_noise

        return texturaizer_ksampler(model, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                    denoise=denoise, disable_noise=disable_noise, start_step=start_at_step, last_step=end_at_step,
                                    force_full_denoise=force_full_denoise, noise_mode=noise_mode, incremental_seed_mode=batch_seed_mode,
                                    callback=callback)

    def doit(self, *args, **kwargs):
        return (self.sample(*args, **kwargs)[0],)


def impact_sampling(*args, **kwargs):
    if 'RegionalSampler' not in nodes.NODE_CLASS_MAPPINGS:
        raise Exception(f"[ERROR] You need to install 'ComfyUI-Impact-Pack'")

    return nodes.NODE_CLASS_MAPPINGS['RegionalSampler'].separated_sample(*args, **kwargs)


def texturaizer_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0,
                         noise_mode="CPU", disable_noise=False, start_step=None, last_step=None, force_full_denoise=False,
                         incremental_seed_mode="comfy", callback=None):
    device = comfy.model_management.get_torch_device()
    noise_device = "cpu" if noise_mode == "CPU" else device
    latent_image = latent["samples"]
    if hasattr(comfy.sample, 'fix_empty_latent_channels'):
        latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)

    latent = latent.copy()

    if disable_noise:
        torch.manual_seed(seed)
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device=noise_device)
    else:
        batch_inds = latent.get("batch_index", None)
        # noise = prepare_noise(latent_image, seed, batch_inds, noise_device, incremental_seed_mode)

    if start_step is None:
        if denoise == 1.0:
            start_step = 0
        else:
            advanced_steps = math.floor(steps / denoise)
            start_step = advanced_steps - steps
            steps = advanced_steps

    samples = impact_sampling(
        model=model, add_noise=not disable_noise, seed=seed, steps=steps, cfg=cfg, sampler_name=sampler_name,
        scheduler=scheduler, positive=positive, negative=negative, latent_image=latent, start_at_step=start_step,
        end_at_step=last_step, return_with_leftover_noise=not force_full_denoise, noise=noise, callback=callback)

    return samples, noise



NODE_CLASS_MAPPINGS = {
    "Texturaizer_KSamplerAdvanced": KSamplerAdvanced_texturaizer,
    "Texturaizer_GenerateNoise": GenerateNoise_texturaizer,
    "Texturaizer_SigmasSelector": SigmasSelector_texturaizer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Texturaizer_KSamplerAdvanced": "KSamplerAdvanced (Texturaizer)",
    "Texturaizer_GenerateNoise": "Generate Noise (Texturaizer)",
    "Texturaizer_SigmasSelector": "Sigmas Selector (Texturaizer)"
}
