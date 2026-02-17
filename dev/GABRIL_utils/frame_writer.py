import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np

class FrameWriter:
    def __init__(self, input_dim, num_actions):
        self.input_dim = input_dim

        self.num_actions = num_actions

        try:
            font = ImageFont.truetype("DejaVuSans.ttf", size=16)
        except OSError:
            font = ImageFont.load_default()


        pre_rendered_frames = []
        for action in range(num_actions):
            text_img = Image.new('RGBA', (84,84), (0, 0, 0, 0))
            draw = ImageDraw.Draw(text_img)

            text = str(action)
            draw.text((11, 55), text, font=font, fill=(255, 255, 255, 255))
            text_tensor = torch.from_numpy(np.array(text_img))
            text_tensor = text_tensor[:, :, :3] * (text_tensor[:, :, 3:] / 255)
            text_tensor = text_tensor[:, :, 0]
            # print(text_tensor.shape, text_tensor.max(), text_tensor.min())
            if self.input_dim != (84, 84):
                text_tensor = torchvision.transforms.functional.resize(text_tensor[None], self.input_dim)[0]
            pre_rendered_frames.append(text_tensor)

        pre_rendered_frames = torch.stack(pre_rendered_frames).numpy().astype(np.uint16)




        self.pre_rendered_frames = pre_rendered_frames


    def add_text_tensor_to_frame(self, frame, action = None, channel_first=True):
        if action is None:
            return frame

        action_frame = self.pre_rendered_frames[action ]
        frame = frame.astype(np.uint16)

        if channel_first:
            addable_frame = action_frame[None]
        else:
            addable_frame = action_frame[:, :, None]

        return np.clip(frame + addable_frame, 0, 255).astype(np.uint8)



