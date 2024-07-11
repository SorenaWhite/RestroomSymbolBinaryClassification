import gradio as gr
import torch
import cv2
from PIL import Image
import numpy as np
import clip
from transformer_decoder import TransformerDecoder

device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model, preprocess = clip.load('ViT-B/32', device)
model = TransformerDecoder(num_classes=2).to(device)
model = model.load_state_dict(torch.load("model_last.pth"))


def inference(im_pil):
    image_feature = clip_model.encode_image(preprocess(im_pil).unsqueeze(0)).to(device)
    image_tensor = torch.cat([image_feature, image_feature])

    male_text_feature = clip_model.encode_text(clip.tokenize("restroom sign of male").to(device))
    female_text_feature = clip_model.encode_text(clip.tokenize("restroom sign of female").to(device))
    text_tensor = torch.cat([male_text_feature, female_text_feature])

    with torch.no_grad():
        print(image_tensor.shape, text_tensor.shape)
        preds = model(image_tensor, text_tensor)
        result = preds.topk(1, 1, True, True).t()
        print(result)
    return result


demo = gr.Interface(fn=inference,
                    inputs=gr.inputs.Image(type="pil"),
                    outputs=gr.outputs.Label(num_top_classes=1),
                    examples=[["data/493_1.png"]],
                    title='Inference demo'
                    )

demo.launch(server_port=6006)
