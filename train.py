from torch.utils.data import Dataset, DataLoader
import torch
import clip
from torch import nn, optim
import pandas as pd
from PIL import Image
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class image_caption_dataset(Dataset):
    def __init__(self, df, preprocess):
        self.images = df["image"]
        self.caption = df["caption"]
        self.preprocess = preprocess

    def __len__(self):
        return len(self.caption)

    def __getitem__(self, idx):
        images = self.preprocess(Image.open(self.images[idx]))
        caption = self.caption[idx]
        return images, caption



def load_data(cup_path, cupnot_path, batch_size, preprocess):
    df = {'image': [], 'caption':[]}
    cup_list = os.listdir(cup_path)
    cupnot_list = os.listdir(cupnot_path)

    caption = cup_path.split('/')[-1]
    for img in cup_list:
        img_path = os.path.join(cup_path, img)
        df['image'].append(img_path)
        df['caption'].append(caption)

    caption = cupnot_path.split('/')[-1]
    for img in cupnot_list:
        img_path = os.path.join(cupnot_path, img)
        df['image'].append(img_path)
        df['caption'].append(caption)

    dataset = image_caption_dataset(df, preprocess)
    train_dataloader = DataLoader(dataset, batch_size=batch_size)
    return train_dataloader


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()


def load_pretrian_model(model_path):
    model, preprocess = clip.load(model_path, device=device, jit=False)  # 训练时 jit必须设置为false
    if device == "cpu":
        model.float()
    else:
        clip.model.convert_weights(model)
    return model, preprocess


def train(epoch, batch_size, learning_rate, cup_path, cupnot_path):
    # 加载模型
    model, preprocess = load_pretrian_model('ViT-B/32')

    #加载数据集
    train_dataloader = load_data(cup_path, cupnot_path, batch_size, preprocess)

    #设置参数
    loss_img = nn.CrossEntropyLoss().to(device)
    loss_txt = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)

    for i in range(epoch):
        for batch in train_dataloader:
            list_image, list_txt = batch  # list_images is list of image in numpy array(np.uint8), or list of PIL images

            #list_image = list_image.to(device)

            texts = clip.tokenize(list_txt).to(device)
            images = list_image.to(device)

            logits_per_image, logits_per_text = model(images, texts)
            print(logits_per_image.shape)
            print(logits_per_text.shape)
            if device == "cpu":
                ground_truth = torch.arange(batch_size).long().to(device)
            else:
                #ground_truth = torch.arange(batch_size).half().to(device)
                ground_truth = torch.arange(batch_size, dtype=torch.long, device=device)


            #反向传播
            total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
            optimizer.zero_grad()
            total_loss.backward()
            if device == "cpu":
                optimizer.step()
            else:
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)

        print('[%d] loss: %.3f' %(i + 1, total_loss))
    torch.save(model, './model/model1.pkl')


def main():
    epoch = 100
    batch_size = 2
    learning_rate = 5e-5
    cup_path = 'data/cup/'
    cupnot_path = 'data/not_cup/'
    train(epoch, batch_size, learning_rate, cup_path, cupnot_path)


if __name__ == '__main__':
    main()
