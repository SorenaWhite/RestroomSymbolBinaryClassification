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
model.load_state_dict(torch.load("model_last.pth"))
model.eval()


def inference(im_pil):
    image_feature = clip_model.encode_image(preprocess(im_pil).unsqueeze(0).to(device))
    image_tensor = torch.cat([image_feature, image_feature]).float()

    text_feature = clip_model.encode_text(clip.tokenize("toilet sign").to(device))
    text_tensor = torch.cat([text_feature, text_feature]).float()

    with torch.no_grad():
        print(image_tensor.dtype, text_tensor.dtype)
        preds = model(image_tensor, text_tensor)
        # print(preds)
        result = preds.topk(1, 1, True, True).indices.cpu().item()[0]

    return result


demo = gr.Interface(fn=inference,
                    inputs=gr.inputs.Image(type="pil"),
                    outputs=gr.outputs.Label(num_top_classes=1),
                    examples=[["data/493_1.png"]],
                    title='Inference demo'
                    )

demo.launch(server_port=6006)
