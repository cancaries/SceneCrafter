import os
import re
import imageio
import numpy as np
from PIL import Image

def process_image(file_name):
    if file_name.endswith(".png"):
        image = Image.open(file_name)
    # resize image to 960p
    image = image.resize((1280, 960), Image.Resampling.LANCZOS)
    return np.array(image.convert("RGB"))

def create_video_from_images(image_folder, video_name, fps=30):
    images = []
    # 寻找所有 png 文件
    image_files = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    if len(image_files) < fps * 2:
        return False
    image_files = sorted(image_files, key=lambda x: int(re.findall(r'\d+', x)[0]))
    # print(image_files)
    # 利用线程池并行处理图像
    images = [process_image(os.path.join(image_folder,image)) for image in image_files]

    with imageio.get_writer(video_name, fps=fps) as video:
        for image in images:
            video.append_data(image)
    
    return True