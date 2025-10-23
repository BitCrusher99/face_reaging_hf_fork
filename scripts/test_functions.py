import face_recognition
import numpy as np
import os
import torch
from torch.autograd import Variable
from torchvision import transforms
from torchvision.io import write_video
import tempfile
import subprocess
from ffmpy import FFmpeg, FFprobe
from PIL import Image

print("🧩 test_functions.py loaded", flush=True)

mask_file = torch.from_numpy(np.array(Image.open('assets/mask1024.jpg').convert('L'))) / 255
small_mask_file = torch.from_numpy(np.array(Image.open('assets/mask512.jpg').convert('L'))) / 255


def sliding_window_tensor(input_tensor, window_size, stride, your_model, mask=mask_file, small_mask=small_mask_file):
    print(f"⚙️ sliding_window_tensor: window={window_size}, stride={stride}", flush=True)
    input_tensor = input_tensor.to(next(your_model.parameters()).device)
    mask = mask.to(next(your_model.parameters()).device)
    small_mask = small_mask.to(next(your_model.parameters()).device)

    n, c, h, w = input_tensor.size()
    print(f"📐 Input tensor shape: {input_tensor.shape}", flush=True)
    output_tensor = torch.zeros((n, 3, h, w), dtype=input_tensor.dtype, device=input_tensor.device)
    count_tensor = torch.zeros((n, 3, h, w), dtype=torch.float32, device=input_tensor.device)

    add = 2 if window_size % stride != 0 else 1
    print("🚀 Starting sliding window loop...", flush=True)
    for y in range(0, h - window_size + add, stride):
        for x in range(0, w - window_size + add, stride):
            window = input_tensor[:, :, y:y + window_size, x:x + window_size]
            with torch.no_grad():
                output = your_model(window)
            output_tensor[:, :, y:y + window_size, x:x + window_size] += output * small_mask
            count_tensor[:, :, y:y + window_size, x:x + window_size] += small_mask
    print("✅ Sliding window done.", flush=True)

    output_tensor /= torch.clamp(count_tensor, min=1.0)
    output_tensor *= mask
    return output_tensor.cpu()


def process_image(your_model, image, video, source_age, target_age=0,
                  window_size=512, stride=256, steps=18):
    print(f"🖼️ process_image(video={video}, source_age={source_age}, target_age={target_age})", flush=True)

    if video:
        target_age = 0
    input_size = (1024, 1024)

    image = np.array(image)
    print(f"📏 Image shape: {image.shape}", flush=True)

    if video:
        width, height, depth = image.shape
        new_width = width if width % 2 == 0 else width - 1
        new_height = height if height % 2 == 0 else height - 1
        image.resize((new_width, new_height, depth))

    print("🔍 Detecting face locations...", flush=True)
    faces = face_recognition.face_locations(image)
    print(f"   ➡️ Detected {len(faces)} faces", flush=True)
    if not faces:
        print("⚠️ No face detected, returning original image", flush=True)
        return image

    fl = faces[0]
    print(f"   🧠 Using face box: {fl}", flush=True)

    # Crop region math
    margin_y_t = int((fl[2] - fl[0]) * .63 * .85)
    margin_y_b = int((fl[2] - fl[0]) * .37 * .85)
    margin_x = int((fl[1] - fl[3]) // (2 / .85))
    margin_y_t += 2 * margin_x - margin_y_t - margin_y_b

    l_y = max([fl[0] - margin_y_t, 0])
    r_y = min([fl[2] + margin_y_b, image.shape[0]])
    l_x = max([fl[3] - margin_x, 0])
    r_x = min([fl[1] + margin_x, image.shape[1]])

    cropped_image = image[l_y:r_y, l_x:r_x, :]
    orig_size = cropped_image.shape[:2]
    print(f"   ✂️ Cropped to {orig_size}", flush=True)

    cropped_image = transforms.ToTensor()(cropped_image)
    cropped_image_resized = transforms.Resize(input_size, interpolation=Image.BILINEAR, antialias=True)(cropped_image)

    source_age_channel = torch.full_like(cropped_image_resized[:1, :, :], source_age / 100)
    target_age_channel = torch.full_like(cropped_image_resized[:1, :, :], target_age / 100)
    input_tensor = torch.cat([cropped_image_resized, source_age_channel, target_age_channel], dim=0).unsqueeze(0)

    image = transforms.ToTensor()(image)

    if video:
        print("🎞️ Starting animation step loop...", flush=True)
        interval = .8 / steps
        aged_cropped_images = torch.zeros((steps, 3, input_size[1], input_size[0]))
        for i in range(steps):
            print(f"   ⏱️ Step {i+1}/{steps}", flush=True)
            input_tensor[:, -1, :, :] += interval
            aged_cropped_images[i, ...] = sliding_window_tensor(input_tensor, window_size, stride, your_model)
        print("✅ All steps done.", flush=True)

        aged_cropped_images_resized = transforms.Resize(orig_size, interpolation=Image.BILINEAR, antialias=True)(aged_cropped_images)
        image = image.repeat(steps, 1, 1, 1)
        image[:, :, l_y:r_y, l_x:r_x] += aged_cropped_images_resized
        image = torch.clamp(image, 0, 1)
        image = (image * 255).to(torch.uint8)
        output_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        write_video(output_file.name, image.permute(0, 2, 3, 1), 2)
        print(f"🎬 Video written → {output_file.name}", flush=True)
        return output_file.name
    else:
        print("🧠 Running single-frame inference...", flush=True)
        aged_cropped_image = sliding_window_tensor(input_tensor, window_size, stride, your_model)
        aged_cropped_image_resized = transforms.Resize(orig_size, interpolation=Image.BILINEAR, antialias=True)(aged_cropped_image)
        image[:, l_y:r_y, l_x:r_x] += aged_cropped_image_resized.squeeze(0)
        image = torch.clamp(image, 0, 1)
        print("✅ Image inference complete", flush=True)
        return transforms.functional.to_pil_image(image)
import face_recognition
import numpy as np
import os
import torch
from torch.autograd import Variable
from torchvision import transforms
from torchvision.io import write_video
import tempfile
import subprocess
from ffmpy import FFmpeg, FFprobe
from PIL import Image

print("🧩 test_functions.py loaded", flush=True)

mask_file = torch.from_numpy(np.array(Image.open('assets/mask1024.jpg').convert('L'))) / 255
small_mask_file = torch.from_numpy(np.array(Image.open('assets/mask512.jpg').convert('L'))) / 255


def sliding_window_tensor(input_tensor, window_size, stride, your_model, mask=mask_file, small_mask=small_mask_file):
    print(f"⚙️ sliding_window_tensor: window={window_size}, stride={stride}", flush=True)
    input_tensor = input_tensor.to(next(your_model.parameters()).device)
    mask = mask.to(next(your_model.parameters()).device)
    small_mask = small_mask.to(next(your_model.parameters()).device)

    n, c, h, w = input_tensor.size()
    print(f"📐 Input tensor shape: {input_tensor.shape}", flush=True)
    output_tensor = torch.zeros((n, 3, h, w), dtype=input_tensor.dtype, device=input_tensor.device)
    count_tensor = torch.zeros((n, 3, h, w), dtype=torch.float32, device=input_tensor.device)

    add = 2 if window_size % stride != 0 else 1
    print("🚀 Starting sliding window loop...", flush=True)
    for y in range(0, h - window_size + add, stride):
        for x in range(0, w - window_size + add, stride):
            window = input_tensor[:, :, y:y + window_size, x:x + window_size]
            with torch.no_grad():
                output = your_model(window)
            output_tensor[:, :, y:y + window_size, x:x + window_size] += output * small_mask
            count_tensor[:, :, y:y + window_size, x:x + window_size] += small_mask
    print("✅ Sliding window done.", flush=True)

    output_tensor /= torch.clamp(count_tensor, min=1.0)
    output_tensor *= mask
    return output_tensor.cpu()


def process_image(your_model, image, video, source_age, target_age=0,
                  window_size=512, stride=256, steps=18):
    print(f"🖼️ process_image(video={video}, source_age={source_age}, target_age={target_age})", flush=True)

    if video:
        target_age = 0
    input_size = (1024, 1024)

    image = np.array(image)
    print(f"📏 Image shape: {image.shape}", flush=True)

    if video:
        width, height, depth = image.shape
        new_width = width if width % 2 == 0 else width - 1
        new_height = height if height % 2 == 0 else height - 1
        image.resize((new_width, new_height, depth))

    print("🔍 Detecting face locations...", flush=True)
    faces = face_recognition.face_locations(image)
    print(f"   ➡️ Detected {len(faces)} faces", flush=True)
    if not faces:
        print("⚠️ No face detected, returning original image", flush=True)
        return image

    fl = faces[0]
    print(f"   🧠 Using face box: {fl}", flush=True)

    # Crop region math
    margin_y_t = int((fl[2] - fl[0]) * .63 * .85)
    margin_y_b = int((fl[2] - fl[0]) * .37 * .85)
    margin_x = int((fl[1] - fl[3]) // (2 / .85))
    margin_y_t += 2 * margin_x - margin_y_t - margin_y_b

    l_y = max([fl[0] - margin_y_t, 0])
    r_y = min([fl[2] + margin_y_b, image.shape[0]])
    l_x = max([fl[3] - margin_x, 0])
    r_x = min([fl[1] + margin_x, image.shape[1]])

    cropped_image = image[l_y:r_y, l_x:r_x, :]
    orig_size = cropped_image.shape[:2]
    print(f"   ✂️ Cropped to {orig_size}", flush=True)

    cropped_image = transforms.ToTensor()(cropped_image)
    cropped_image_resized = transforms.Resize(input_size, interpolation=Image.BILINEAR, antialias=True)(cropped_image)

    source_age_channel = torch.full_like(cropped_image_resized[:1, :, :], source_age / 100)
    target_age_channel = torch.full_like(cropped_image_resized[:1, :, :], target_age / 100)
    input_tensor = torch.cat([cropped_image_resized, source_age_channel, target_age_channel], dim=0).unsqueeze(0)

    image = transforms.ToTensor()(image)

    if video:
        print("🎞️ Starting animation step loop...", flush=True)
        interval = .8 / steps
        aged_cropped_images = torch.zeros((steps, 3, input_size[1], input_size[0]))
        for i in range(steps):
            print(f"   ⏱️ Step {i+1}/{steps}", flush=True)
            input_tensor[:, -1, :, :] += interval
            aged_cropped_images[i, ...] = sliding_window_tensor(input_tensor, window_size, stride, your_model)
        print("✅ All steps done.", flush=True)

        aged_cropped_images_resized = transforms.Resize(orig_size, interpolation=Image.BILINEAR, antialias=True)(aged_cropped_images)
        image = image.repeat(steps, 1, 1, 1)
        image[:, :, l_y:r_y, l_x:r_x] += aged_cropped_images_resized
        image = torch.clamp(image, 0, 1)
        image = (image * 255).to(torch.uint8)
        output_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        write_video(output_file.name, image.permute(0, 2, 3, 1), 2)
        print(f"🎬 Video written → {output_file.name}", flush=True)
        return output_file.name
    else:
        print("🧠 Running single-frame inference...", flush=True)
        aged_cropped_image = sliding_window_tensor(input_tensor, window_size, stride, your_model)
        aged_cropped_image_resized = transforms.Resize(orig_size, interpolation=Image.BILINEAR, antialias=True)(aged_cropped_image)
        image[:, l_y:r_y, l_x:r_x] += aged_cropped_image_resized.squeeze(0)
        image = torch.clamp(image, 0, 1)
        print("✅ Image inference complete", flush=True)
        return transforms.functional.to_pil_image(image)
