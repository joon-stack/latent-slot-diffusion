import os
import glob
import random
import torchvision
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, TensorDataset, DataLoader

import argparse

import tensorflow_datasets as tfds
from tensorflow_datasets.core.utils import gcs_utils

from transformers import BertTokenizerFast, BertModel
import torch


def decode_inst(inst):
  """Utility to decode encoded language instruction"""
  return bytes(inst[np.where(inst != 0)].tolist()).decode("utf-8") 



class GlobDataset(Dataset):
    def __init__(self, root, img_size, img_glob='*.png', 
                 data_portion=(),  random_data_on_portion=True,
                vit_norm=False, random_flip=False, vit_input_resolution=448):
        super().__init__()
        if isinstance(root, str) or not hasattr(root, '__iter__'):
            root = [root]
            img_glob = [img_glob]
        if not all(hasattr(sublist, '__iter__') for sublist in data_portion) or data_portion == (): # if not iterable or empty
            data_portion = [data_portion]
        self.root = root
        self.img_size = img_size
        self.episodes = []
        self.vit_norm = vit_norm
        self.random_flip = random_flip

        for n, (r, g) in enumerate(zip(root, img_glob)):
            episodes = glob.glob(os.path.join(r, g), recursive=True)

            episodes = sorted(episodes)

            data_p = data_portion[n]

            assert (len(data_p) == 0 or len(data_p) == 2)
            if len(data_p) == 2:
                assert max(data_p) <= 1.0 and min(data_p) >= 0.0

            if data_p and data_p != (0., 1.):
                if random_data_on_portion:
                    random.Random(42).shuffle(episodes) # fix results
                episodes = \
                    episodes[int(len(episodes)*data_p[0]):int(len(episodes)*data_p[1])]

            self.episodes += episodes
        
        # print(self.episodes)
        
        # resize the shortest side to img_size and center crop
        self.transform = transforms.Compose([
            # transforms.Resize(img_size, interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
            # transforms.CenterCrop(img_size),
            transforms.Resize((img_size, img_size), interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        if vit_norm:
            self.transform_vit = transforms.Compose([
                transforms.Resize(vit_input_resolution, interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(vit_input_resolution),

                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
    
    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, i):
        example = {}
        image = Image.open(self.episodes[i])
        if not image.mode == "RGB":
            image = image.convert("RGB")
        if self.random_flip:
            if random.random() > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
        pixel_values = self.transform(image)
        example["pixel_values"] = pixel_values
        if self.vit_norm:
            image_vit = self.transform_vit(image)
            example["pixel_values_vit"] = image_vit
        return example


class GSDataset(Dataset):
    def __init__(self, data_split, section, img_size, local=False, predict_steps=1, img_glob='*.png', 
                 data_portion=0.9,  random_data_on_portion=True,
                vit_norm=False, random_flip=False, vit_input_resolution=448):
        super().__init__()


        if local:
            ds, ds_info = tfds.load("0.0.1", 
                                    data_dir='/home/s2/youngjoonjeong/datasets', with_info=True)
        else:
            ds, ds_info = tfds.load(f"{data_split}:0.0.1", 
                                    data_dir='gs://gresearch/robotics', with_info=True)

        # dataset_path = os.path.join('gs://gresearch/robotics/', data_split, "0.0.1") 
        # builder = tfds.builder_from_directory(dataset_path)
        # episode_ds = builder.as_dataset(split=section)
        # steps_ds = episode_ds.flat_map(lambda x: x['steps'])
        # section in ["train", "validation", "test"]
        data_size = int(ds_info.splits['train'].num_examples * data_portion)
        dataset = ds['train'].take(data_size) if section == 'train' else ds['train'].skip(data_size)
        data_iter = iter(tfds.as_numpy(dataset))
        
        print(f"{data_split} loaded successfully")

        tokenizer = BertTokenizerFast.from_pretrained("google-bert/bert-base-uncased")
        print("Tokenizer loaded successfully")

        model = BertModel.from_pretrained("google-bert/bert-base-uncased")
        print("BERT loaded successfully")

        x_list = []
        y_list = []
        ins_list = []
        for i, record in enumerate(data_iter):
            steps = list(record['steps'])
            # extract frames based on the predict_steps (how many steps will y contains)
            frame_num = np.linspace(0, len(steps)- 1, predict_steps+1).astype(int)
            y_step_list = []
            for i in frame_num:
                step = steps[i]
                img = step['observation']['rgb']
                ins = step['observation']['instruction']
                if i == 0:
                    x_list += [torch.from_numpy(img)]
                else:
                    y_step_list += [torch.from_numpy(img)]

            y_list += [torch.stack(y_step_list)]
            ins_decode = decode_inst(ins)
            ins_list += [ins_decode.strip()]
            
        x_list_cat = torch.stack(x_list)
        y_list_cat = torch.stack(y_list)
        tok = tokenizer(ins_list, padding=True, truncation=True, return_tensors='pt')
        ins_list_cat = model(tok['input_ids'], tok['attention_mask']).last_hidden_state

        self.length = ins_list_cat.shape[0]
                
    
        self.data_list = {'x': x_list_cat, 'y': y_list_cat, 'ins': ins_list_cat}
        # self.dataset = TensorDataset(*data_list)

                
        
        # resize the shortest side to img_size and center crop
        self.transform = transforms.Compose([
            # transforms.Resize(img_size, interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
            # transforms.CenterCrop(img_size),
            transforms.Resize((img_size, img_size), interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        if vit_norm:
            self.transform_vit = transforms.Compose([
                transforms.Resize(vit_input_resolution, interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(vit_input_resolution),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
    

    def __len__(self):
        return self.data_list['ins'].shape[0]

    def __getitem__(self, i):
        example = {}
        x = torch.FloatTensor(self.data_list['x'][i])
        y = torch.FloatTensor(self.data_list['y'][i])
        ins = torch.FloatTensor(self.data_list['ins'][i])
    
        # if self.random_flip:
        #     if random.random() > 0.5:
        #         x = image.transpose(Image.FLIP_LEFT_RIGHT)
        x = self.transform(x)
        y = self.transform(y)
        example["x"] = x
        example["y"] = y
        example["ins"] = ins
        # if self.vit_norm:
        #     image_vit = self.transform_vit(image)
        #     example["pixel_values_vit"] = image_vit
        return example

if __name__ == "__main__":
    # dataset = GlobDataset(
    #     root="/shared/s2/lab01/dataset/lsd/movi_/movi-e/movi-e-train-with-label/images/",
    #     # root='gs://gresearch/robotics/language_table/0.0.1/',
    #     # root='/home/s2/youngjoonjeong/github/latent-slot-diffusion/',
    #     img_size=256,
    #     img_glob="**/*.png",
    #     # img_glob='*.out',
    #     data_portion=(0.0, 0.9)
    # )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predict_steps",
        type=int,
        default=4,
        help="Number of images that should be predicted.",
    )
    parser.add_argument(
        "--data_split",
        type=str,
        default='language_table_blocktoblock_4block_sim',
        help="Data Split",
        choices=['language_table_blocktoblock_4block_sim', 'language_table', 'language_table_sim', 'language_table_blocktoblock_sim',
        'language_table_blocktoblock_oracle_sim', 'language_table_blocktoblockrelative_oracle_sim', 'language_table_blocktoabsolute_oracle_sim',
        'language_table_blocktorelative_oracle_sim', 'language_table_separate_oracle_sim']
    )

    parser.add_argument(
        "--section",
        type=str,
        default='train',
        help="Train or Validation",
        choices=['train', 'val']
    )

    args = parser.parse_args()

    dataset = GSDataset(
        data_split=args.data_split,
        section=args.section,
        img_size=256,
        predict_steps=args.predict_steps,
    )
    print(dataset)
    torch.save(dataset, f'/shared/s2/lab01/dataset/lsd/{args.data_split}_predict{args.predict_steps}_{args.train}.pth')
    print("saved successfully")
    dataset_loaded = torch.load(f'/shared/s2/lab01/dataset/lsd/{args.data_split}_predict{args.predict_steps}_{args.train}.pth')
    train_dataloader = torch.utils.data.DataLoader(
                                                    dataset_loaded,
                                                    batch_size=16,
                                                    shuffle=True,
                                                    num_workers=4,
    )
    for i, x in enumerate(train_dataloader):
        if i >= 2:
            break
        print(x)
    
    pass