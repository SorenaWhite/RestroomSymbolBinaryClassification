import os
import glob
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


def build_transform(args, is_train):
    resize_im = args.input_size > 32
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        # warping (no cropping) when evaluated at 384 or larger
        if args.input_size >= 384:
            t.append(
            transforms.Resize((args.input_size, args.input_size),
                            interpolation=transforms.InterpolationMode.BICUBIC),
        )
            print(f"Warping {args.input_size} size input images...")
        else:
            if args.crop_pct is None:
                args.crop_pct = 224 / 256
            size = int(args.input_size / args.crop_pct)
            t.append(
                # to maintain same ratio w.r.t. 224 images
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            )
            t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


class MMLRestroomSign(Dataset):
    def __init__(self, args, transform, is_train=True):
        if args.if_llm:
            self.get_data = self.get_data_with_llm
        else:
            self.get_data = self.get_data_from_raw

        if is_train:
            train_root = os.path.join(args.data_root, "train")
            self.symbol_pairs = self.read_from_disk(train_root)
        else:
            val_root = os.path.join(args.data_root, "val")
            self.symbol_pairs = self.read_from_disk(val_root)

        self.transform = transform

    def __len__(self):
        return len(self.symbol_pairs)

    def read_from_disk(self, image_folder):
        all_paths = glob.glob(os.path.join(image_folder, "*.png"))
        all_names = [os.path.basename(os.path.splitext(name)[0]).split("_")[0] for name in all_paths]
        all_names = list(set(all_names))

        pairs = []
        for name in all_names:
            pairs.append([os.path.join(image_folder, f"{name}_0.png"), os.path.join(image_folder, f"{name}_1.png")])
        return pairs

    def get_data_with_llm(self, item):
        pass

    def get_data_from_raw(self, item):
        male_sign_path, female_sign_path = self.symbol_pairs[item]

        return self.transform(Image.open(male_sign_path).convert("RGB")), \
               self.transform(Image.open(female_sign_path))

    def __getitem__(self, item):
        return self.get_data(item)
