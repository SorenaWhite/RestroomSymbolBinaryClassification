import os
import glob
import clip
import random
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from timm.data import create_transform
from timm.data.constants import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    IMAGENET_INCEPTION_MEAN,
    IMAGENET_INCEPTION_STD
)


# def build_transform(args, is_train):
#     resize_im = args.input_size > 32
#     imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
#     mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
#     std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD
#
#     if is_train:
#         # this should always dispatch to transforms_imagenet_train
#         transform = create_transform(
#             input_size=args.input_size,
#             is_training=True,
#             color_jitter=args.color_jitter,
#             auto_augment=args.aa,
#             interpolation=args.train_interpolation,
#             re_prob=args.reprob,
#             re_mode=args.remode,
#             re_count=args.recount,
#             mean=mean,
#             std=std,
#         )
#         if not resize_im:
#             transform.transforms[0] = transforms.RandomCrop(
#                 args.input_size, padding=4)
#         return transform
#
#     t = []
#     if resize_im:
#         # warping (no cropping) when evaluated at 384 or larger
#         if args.input_size >= 384:
#             t.append(
#             transforms.Resize((args.input_size, args.input_size),
#                             interpolation=transforms.InterpolationMode.BICUBIC),
#         )
#             print(f"Warping {args.input_size} size input images...")
#         else:
#             if args.crop_pct is None:
#                 args.crop_pct = 224 / 256
#             size = int(args.input_size / args.crop_pct)
#             t.append(
#                 # to maintain same ratio w.r.t. 224 images
#                 transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
#             )
#             t.append(transforms.CenterCrop(args.input_size))
#
#     t.append(transforms.ToTensor())
#     t.append(transforms.Normalize(mean, std))
#     return transforms.Compose(t)

def build_transform(args, is_train):
    pass


class MMLRestroomSign(Dataset):
    def __init__(self, data_root, transform, device, clip_model, preprocess, is_train=True):
        if is_train:
            train_root = os.path.join(data_root, "train")
            self.symbol_pairs = self.read_from_disk(train_root)
        else:
            val_root = os.path.join(data_root, "val")
            self.symbol_pairs = self.read_from_disk(val_root)
        self.device = "cpu"
        self.clip_model = clip_model
        self.preprocess = preprocess

        self.male_text_feature = self.get_clip_text_feature("male toilet sign")
        self.female_text_feature = self.get_clip_text_feature("female toilet sign")

        self.transform = transform

    def __len__(self):
        return len(self.symbol_pairs)

    def choose_text(self):
        if random.random() < 0.5:
            return self.male_text_feature, 0
        else:
            return self.female_text_feature, 1

    def read_from_disk(self, image_folder):
        all_paths = glob.glob(os.path.join(image_folder, "*.png"))
        all_names = [os.path.basename(os.path.splitext(name)[0]).split("_")[0] for name in all_paths]
        all_names = list(set(all_names))

        pairs = []
        for name in all_names:
            pairs.append([os.path.join(image_folder, f"{name}_0.png"), os.path.join(image_folder, f"{name}_1.png")])
        return pairs

    def get_clip_image_feature(self, image_path):
        with torch.no_grad():
            image_input = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
            image_feature = self.clip_model.encode_image(image_input)
            return image_feature.detach()

    def get_clip_text_feature(self, text):
        with torch.no_grad():
            text_input = clip.tokenize(text).to(self.device)
            text_feature = self.clip_model.encode_text(text_input)
            return text_feature.detach()

    def get_data_from_clip(self, item):
        male_sign_path, female_sign_path = self.symbol_pairs[item]

        male_image_feature = self.get_clip_image_feature(male_sign_path)
        female_image_feature = self.get_clip_image_feature(female_sign_path)

        if random.random() < 0.5:
            image_tensor = torch.cat([male_image_feature, female_image_feature])

            text_feature_0, label_0 = self.choose_text()
            correct_0 = 1 - label_0 ^ 0

            text_feature_1, label_1 = self.choose_text()
            correct_1 = 1 - label_1 ^ 1
            print(label_0, correct_0)
            print(label_1, correct_1)
            text_tensor = torch.cat([text_feature_0, text_feature_1])
            target_tensor = torch.tensor([correct_0, correct_1], dtype=torch.long)
        else:
            image_tensor = torch.cat([female_image_feature, male_image_feature])

            text_feature_0, label = self.choose_text()
            correct_0 = 1 - label ^ 1

            text_feature_1, label = self.choose_text()
            correct_1 = 1 - label ^ 0

            text_tensor = torch.cat([text_feature_0, text_feature_1])
            target_tensor = torch.tensor([correct_0, correct_1], dtype=torch.long)

        return image_tensor, text_tensor, target_tensor


    def __getitem__(self, item):
        return self.get_data_from_clip(item)
