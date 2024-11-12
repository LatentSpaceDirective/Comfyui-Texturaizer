import comfy
import nodes
from nodes import CLIPTextEncode
from comfy_extras.nodes_clip_sdxl import CLIPTextEncodeSDXL

import torch
import torchvision.transforms.v2 as T
import numpy as np
import scipy

def mask_from_color(mask_color, color_mask, threshold=50):
    try:
        if mask_color.startswith("#"):
            selected = int(mask_color[1:], 16)
        else:
            selected = int(mask_color, 10)
    except Exception:
        raise Exception(f"[ERROR] Invalid mask_color value. mask_color should be a color value for RGB")

    selected_r = (selected >> 16) & 0xFF
    selected_g = (selected >> 8) & 0xFF
    selected_b = selected & 0xFF

    color_mask = (torch.clamp(color_mask, 0, 1.0) * 255.0).round()

    r_diff = (color_mask[:, :, :, 0] - selected_r).abs()
    g_diff = (color_mask[:, :, :, 1] - selected_g).abs()
    b_diff = (color_mask[:, :, :, 2] - selected_b).abs()

    distance = torch.sqrt(r_diff ** 2 + g_diff ** 2 + b_diff ** 2)
    mask = torch.where(distance <= threshold, 1.0, 0.0)

    # Ensure mask has a single channel
    if mask.dim() == 3:
        mask = mask.unsqueeze(0)
    if mask.size(0) > 1:
        mask = mask[0:1, :, :]  # Take only one channel if more than one exists

    return mask

def invert_mask(mask):
    # Invert the mask by subtracting from 1 (assuming mask is binary with values 0 and 1)
    inverted_mask = 1 - mask
    return inverted_mask

def combine_masks(mask_list):
    # Combine all the masks using max operation
    combined_mask = torch.max(torch.stack(mask_list), dim=0).values

    # Ensure combined_mask has a single channel
    if combined_mask.dim() == 3:
        combined_mask = combined_mask.unsqueeze(0)
    if combined_mask.size(0) > 1:
        combined_mask = combined_mask[0:1, :, :]  # Take only one channel if more than one exists

    return combined_mask

def encode(clip, text_g, text_l, width, height, clip_scale, version_select):
    width = width * clip_scale
    height = height * clip_scale
    crop_w = 0
    crop_h = 0
    text = text_g + ", " + text_l
    if version_select == 'SDXL':
        conditioning = CLIPTextEncodeSDXL().encode(clip, width, height, crop_w, crop_h, width, height, text_g, text_l)[0]
    else: #if version_select == 'SD1.5
        conditioning = CLIPTextEncode().encode(clip, text)[0]
        
    return conditioning

def set_mask(conditioning, mask, strength, set_cond_area = 'default'):
    conditioning = nodes.ConditioningSetMask().append(conditioning, mask, set_cond_area, strength)[0]
    return conditioning

def expand_mask(mask, expand, tapered_corners):
    c = 0 if tapered_corners else 1
    kernel = np.array([[c, 1, c],
                       [1, 1, 1],
                       [c, 1, c]])
    mask = mask.reshape((-1, mask.shape[-2], mask.shape[-1]))
    out = []
    for m in mask:
        output = m.numpy()
        for _ in range(abs(expand)):
            if expand < 0:
                output = scipy.ndimage.grey_erosion(output, footprint=kernel)
            else:
                output = scipy.ndimage.grey_dilation(output, footprint=kernel)
        output = torch.from_numpy(output)
        out.append(output)

    return torch.stack(out, dim=0)

def blur_mask(mask, amount):
    if amount > 0:
        mask = mask.to(comfy.model_management.get_torch_device())
        size = int(6 * amount + 1)
        if size % 2 == 0:
            size+= 1
        
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        blurred = mask.unsqueeze(1)
        blurred = T.GaussianBlur(size, amount)(blurred)
        blurred = blurred.squeeze(1)
        blurred = blurred.to(comfy.model_management.intermediate_device())

        return blurred
    else:
        return mask

def combine_style_prompts(base_prompt="", base_prompt_l="", style_prompt=""):
    # 1. If the style_prompt is empty, return the base_prompt and base_prompt_l
    if not style_prompt:
        return base_prompt, base_prompt_l

    # 2. Check if the style_prompt contains a "."
    if "." in style_prompt:
        # Split the style_prompt into two parts
        style_part_1, style_part_2 = style_prompt.split(".", 1)
        
        # Combine the first part with the base_prompt
        final_base_prompt = base_prompt
        if "{prompt}" in style_part_1:
            final_base_prompt = style_part_1.replace("{prompt}", base_prompt)
        else:
            final_base_prompt = base_prompt + ", " + style_part_1
        
        # Combine the second part with base_prompt_l if it exists
        final_base_prompt_l = base_prompt_l
        if base_prompt_l:
            if "{prompt}" in style_part_2:
                final_base_prompt_l = style_part_2.replace("{prompt}", base_prompt_l)
            else:
                final_base_prompt_l = base_prompt_l + ", " + style_part_2
        else:
            final_base_prompt_l = style_part_2 if "{prompt}" not in style_part_2 else ""

        return final_base_prompt, final_base_prompt_l

    # 3. Check if the style_prompt contains "{prompt}"
    if "{prompt}" in style_prompt:
        final_base_prompt = style_prompt.replace("{prompt}", base_prompt)
        final_base_prompt_l = style_prompt.replace("{prompt}", base_prompt_l) if base_prompt_l else ""
        return final_base_prompt, final_base_prompt_l

    # 4. If no {prompt} exists, append the style_prompt to the end with ", " separator
    final_base_prompt = base_prompt + ", " + style_prompt
    final_base_prompt_l = base_prompt_l + ", " + style_prompt if base_prompt_l else ""

    return final_base_prompt, final_base_prompt_l

class CombinedConditioningFromColors:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", ),
                "image": ("IMAGE", ),
                "scene_data": ("DICTIONARY", {"default": "{}"}),
                "threshold": ("INT", { "default": 4, "min": 0, "max": 127, "step": 1, }),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "DATA", "MASK", )
    FUNCTION = "execute"
    CATEGORY = "Texturaizer"

    def create_conditioning_masks(self, scene_data, image, clip, threshold):
        conditionings = []
        data = []

        scene_info = scene_data['scene_info']
        positive_l = scene_info['positive_prompt_l']
        positive_g = scene_info['positive_prompt_g']
        use_other_prompt = scene_info['use_other_prompt']
        other_prompt = scene_info['other_prompt']
        
        prepend_pos_prompt_g = scene_info['prepend_pos_prompt_g']
        delimiter = scene_info['delimiter']
        append_pos_prompt_l = scene_info['append_pos_prompt_l']
        
        width = scene_info['width']
        height = scene_info['height']
        use_segment_data = scene_info['use_segment_data']
        segment_type = scene_info['segment_type']
        strength = scene_info['condition_strength']
        expand = scene_info['mask_expand']
        blur = scene_info['mask_blur']
        clip_scale = 4
        version_select = scene_info['version_select']
        use_style = scene_info['use_style']
        style_pos = scene_info['style_pos']

        if not use_segment_data:    
            mask = torch.zeros((1, 1, height, width), dtype=torch.float32)
            prompt = positive_g
            if use_other_prompt:
                prompt = positive_g + other_prompt
            if use_style:
                prompt, positive_l = combine_style_prompts(prompt, positive_l, style_pos)
            conditioning = encode(clip, prompt, positive_l, width, height, clip_scale, version_select)
            conditionings.append(conditioning)

            data.append(prompt)
            data.append(positive_l)

            return conditionings, data, mask
        
        print("SEGMENT TYPE: ", segment_type, ", SEGMENT COUNT: ", len(scene_data[segment_type]))
        unused_segments = 0

        mask = None
        used_segment_masks = []
        empty_semgents = []
        unused_colors = ['#000000']
        data.append("SEGMENT PROMPTS")
        for i in scene_data[segment_type]:
            # Get data for color:
            colors = []
            if segment_type in ['Collections', 'Assets']:
                colors = i['colors']
            else:
                color = i['color']
                colors.append(color)

            if i['enable'] == False:
                unused_segments += 1
                # print("DISSABLED: ", i['name'])
                unused_colors.extend(colors)
                continue
            
            seg_prompt = i['prompt']

            # Combine prompts
            if prepend_pos_prompt_g:
                concat = positive_g + delimiter + seg_prompt
            else: concat = seg_prompt

            if not append_pos_prompt_l:
                positive_l = ""

            if use_style:
                concat, positive_l = combine_style_prompts(concat, positive_l, style_pos)

            if len(colors) > 1:
                masks = [mask_from_color(color, image, threshold) for color in colors]
                mask = combine_masks(masks)
            else:
                mask = mask_from_color(colors[0], image, threshold)
                
            if mask.sum()>0:
                mask = expand_mask(mask, expand, True)
                mask = blur_mask(mask, blur)

                # condition
                conditioning = encode(clip, concat, positive_l, width, height, clip_scale, version_select)
                conditioning = set_mask(conditioning, mask, strength)
                conditionings.append(conditioning)

                data.append("")
                data.append(f"{i['name']} _ {i['id']}")
                data.append(colors)
                data.append(seg_prompt)
                data.append(concat)
                data.append(positive_l)
                data.append(str(mask.sum()/torch.numel(mask)))

                used_segment_masks.append(mask)
            else:
                empty_semgents.append(seg_prompt)

        if empty_semgents:
            print("empty segments: ", empty_semgents)

        if used_segment_masks:
            print("USED SEGMENTS MASKS", used_segment_masks)
            used_mask = combine_masks(used_segment_masks)
            used_mask[used_mask >= 0.1] = 1.0
            used_mask[used_mask < 0.1] = 0.0
            unused_mask = 1.0 - used_mask
            mask = unused_mask
        elif len(unused_colors) > 1:
            print("USED SEGMENTS: ", len(scene_data[segment_type])-unused_segments, ", UNUSED SEGMENTS: ", unused_segments)
            masks = [mask_from_color(color, image, threshold) for color in unused_colors]
            mask = combine_masks(masks)
        else:
            if use_other_prompt:
                prompt = positive_g + other_prompt
            else: prompt = positive_g
            if use_style:
                prompt, positive_l = combine_style_prompts(prompt, positive_l, style_pos)
            print("WARNING: All Masks are Empty. Using Only Scene Prompt")
            conditioning = encode(clip, prompt, positive_l, width, height, clip_scale, version_select)
            conditionings.append(conditioning)

            data.append("")
            data.append("OTHER PROMPT")
            data.append(prompt)
            data.append(positive_l)
        #     mask = mask_from_color("#000000", image, threshold)

        if mask.sum()>0:
            mask = expand_mask(mask, expand, True)
            mask = blur_mask(mask, blur)

            if use_other_prompt and other_prompt != "":
                concat = positive_g + delimiter + other_prompt
            else:
                concat = positive_g
            if use_style:
                concat, positive_l = combine_style_prompts(concat, positive_l, style_pos)

            # condition
            conditioning = encode(clip, concat, positive_l, width, height, clip_scale, version_select)
            conditioning = set_mask(conditioning, mask, strength)
            conditionings.append(conditioning)

            data.append("")
            data.append("OTHER PROMPT")
            data.append(concat)
            data.append(positive_l)
            
        if conditionings == []:
            if use_other_prompt:
                prompt = positive_g + other_prompt
            else: prompt = positive_g
            if use_style:
                prompt, positive_l = combine_style_prompts(prompt, positive_l, style_pos)
            print("WARNING: All Masks are Empty. Using Only Scene Prompt")
            conditioning = encode(clip, prompt, positive_l, width, height, clip_scale, version_select)
            conditionings.append(conditioning)

            data.append("")
            data.append("OTHER PROMPT")
            data.append(prompt)
            data.append(positive_l)

        return conditionings, data, mask

    def combine_conditionings(self, conditionings):
        if not conditionings:
            return None

        while len(conditionings) > 1:
            conditioning_1 = conditionings.pop(0)
            conditioning_2 = conditionings.pop(0)
            combined = nodes.ConditioningCombine().combine(conditioning_1, conditioning_2)
            conditionings.append(combined[0])

        return conditionings[0],

    def execute(self, scene_data, image, clip, threshold):
        conditionings, data, other_mask = self.create_conditioning_masks(scene_data, image, clip, threshold)
        conditioning = self.combine_conditionings(conditionings)
        return (conditioning[0], data, other_mask, )
    
 
class ClipEncodeSwitchVersion:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", ),
                "text_g": ("STRING", { "default": ""}),
                "text_l": ("STRING", { "default": ""}),
                "use_sdxl": ("BOOLEAN", ),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING", )
    RETURN_NAMES = ("CONDITIONING", )
    FUNCTION = "execute"
    CATEGORY = "Texturaizer"

    def execute(self, clip, text_g, text_l, use_sdxl):
        width = 1024
        height = 1024
        clip_scale = 4

        if use_sdxl:
            version_select = "SDXL"
        else:
            version_select = "SD1.5"

        conditioning = encode(clip, text_g, text_l, width, height, clip_scale, version_select)
        return (conditioning, )
    
class ApplyStyleToPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "apply_style": ("BOOLEAN", {"default": True}),
                "base_prompt_g": ("STRING", { "default": ""}),
                "base_prompt_l": ("STRING", { "default": ""}),
                "style_prompt": ("STRING", { "default": ""}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("PROMPT G", "PROMPT L")
    FUNCTION = "execute"
    CATEGORY = "Texturaizer"

    def execute(self, apply_style, base_prompt_g, base_prompt_l, style_prompt):
        if apply_style:
            prompt_g, prompt_l = combine_style_prompts(base_prompt_g, base_prompt_l, style_prompt)
        else:
            prompt_g = base_prompt_g
            prompt_l = base_prompt_l
            
        return (prompt_g, prompt_l)

NODE_CLASS_MAPPINGS = {
    "Texturaizer_CombinedConditioningFromColors": CombinedConditioningFromColors,
    "Texturaizer_ClipEncodeSwitchVersion": ClipEncodeSwitchVersion,
    "Texturaizer_ApplyStyleToPrompt": ApplyStyleToPrompt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Texturaizer_CombinedConditioningFromColors": "Combined Conditioning From Colors (Texturaizer)",
    "Texturaizer_ClipEncodeSwitchVersion": "Clip Encode Switch (Texturaizer)",
    "Texturaizer_ApplyStyleToPrompt": "Apply Style Prompt (Texturaizer)",
}