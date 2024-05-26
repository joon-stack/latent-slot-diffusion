import os
import glob
import random
import torchvision
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, TensorDataset, DataLoader, ConcatDataset

import argparse

import tensorflow_datasets as tfds
from tensorflow_datasets.core.utils import gcs_utils

from transformers import BertTokenizerFast, BertModel
import torch


def decode_inst(inst):
  """Utility to decode encoded language instruction"""
  return bytes(inst[np.where(inst != 0)].tolist()).decode("utf-8") 


class GSLocalDataset(Dataset):
    def __init__(self, root, img_size, section, predict_steps=1, img_glob='*.png', 
                random_flip=False):

        super().__init__()
        # if isinstance(root, str) or not hasattr(root, '__iter__'):
        #     root = [root]
        #     img_glob = [img_glob]
        img_glob = [img_glob]
        root_img = [os.path.join(root, 'images', section)]
        root_ins = os.path.join(root, 'labels', section)
        self.root_img = root_img
        self.root_ins = root_ins
        self.img_size = img_size
        self.episodes = {}
        inst_raw = []
        self.random_flip = random_flip
        self.predict_steps = predict_steps

        tokenizer = BertTokenizerFast.from_pretrained("google-bert/bert-base-uncased")
        # print("Tokenizer loaded successfully")

        model = BertModel.from_pretrained("google-bert/bert-base-uncased")
        # print("BERT loaded successfully")

        def extract_folder_name(file_path):
            return file_path.split('/')[-2]

        def encode(ins_list: list) -> torch.tensor:
            tok = tokenizer(ins_list, padding='max_length', truncation=True, max_length=24, return_tensors='pt')
            ins_list_cat = model(tok['input_ids'], tok['attention_mask']).last_hidden_state.detach().numpy()
            return ins_list_cat
            

        for n, (r, g) in enumerate(zip(root_img, img_glob)):
            episodes = glob.glob(os.path.join(r, g), recursive=True)
            episodes = sorted(episodes)
            numbers = sorted(list(map(extract_folder_name, episodes)))
            numbers_no_overlap = sorted(list(set(list(map(extract_folder_name, episodes)))))
            instructions = list(map(lambda x: os.path.join(root_ins, str(x) + '.npy'), numbers_no_overlap))

            # self.episodes += episodes
            # inst_raw += instructions
        for k, v in zip(numbers, episodes):
            if k in self.episodes:
                self.episodes[k].append(v)
            else:
                self.episodes[k] = [v]
        # self.episodes = dict(zip(numbers, episodes))
        inst_raw += instructions
        # print(self.episodes)

        self.ids = list(self.episodes.keys())
        self.episodes = list(self.episodes.values())

        
        self.instructions_nl = list(map(lambda x: decode_inst(np.load(x)), inst_raw))
        self.instructions = encode(self.instructions_nl)
        # print(self.episodes)
        # print(list(self.episodes.values()))
        # idx = -5
        # key = list(self.episodes.keys())[idx]
        # print(self.episodes[idx], inst_raw[idx], self.instructions[idx].shape, self.instructions_nl[idx])
        # resize the shortest side to img_size and center crop
        self.transform = transforms.Compose([
            # transforms.Resize(img_size, interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
            # transforms.CenterCrop(img_size),
            transforms.Resize((img_size, img_size), interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    
    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, i):
        example = {}
        steps = self.episodes[i]
        predict_idx = np.linspace(0, len(steps)- 1, self.predict_steps+1).astype(int)
        x = Image.open(steps[predict_idx[0]])
        if self.predict_steps <= 1:
            y = steps[predict_idx[1]]
        else:
            y = []
            for i in range(1, len(predict_idx)):
                y.append(Image.open(steps[predict_idx[i]]))
        if not x.mode == "RGB":
            x = x.convert("RGB")
        x = self.transform(x)
        example['x'] = x
        if self.predict_steps <= 1:
            y = Image.open(y)
            if not y.mode == "RGB":
                y = y.convert("RGB")
            y = self.transform(y)
            example['y'] = y
        else:
            y_ = []
            for img in y:
                y_.append(self.transform(Image.open(img)))
                example['y'] = y_
        
        example['ins'] = self.instructions[i]

        return example



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
        # print(image)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        # print(image)
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
            ds, ds_info = tfds.load('datasets:0.0.1', data_dir='/home/s2/youngjoonjeong/', with_info=True,
                                    download=False,
            # download_and_prepare_kwargs={'download_config': tfds.download.DownloadConfig(
            #                             max_workers=4  # 동시에 처리할 최대 작업자 수
            #                         )}
                                    )
        else:
            ds, ds_info = tfds.load(f"{data_split}:0.0.1", 
                                    data_dir='gs://gresearch/robotics', with_info=True, download=False
                                    # download_and_prepare_kwargs={'download_config': tfds.download.DownloadConfig(
                                    #     max_workers=4  # 동시에 처리할 최대 작업자 수
                                    # )}
                                    )

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
        # print("Tokenizer loaded successfully")

        model = BertModel.from_pretrained("google-bert/bert-base-uncased")
        # print("BERT loaded successfully")

        x_list = []
        y_list = []
        ins_list = []
        tmp = 2
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
                    # x_list += [torch.from_numpy(img)]
                    x_list += [img]
                else:
                    # y_step_list += [torch.from_numpy(img)]
                    y_step_list += [img]
            if len(y_step_list) != tmp:
                tmp = len(y_step_list)
                print(len(y_step_list))

            y_list += [np.stack(y_step_list)]
            ins_decode = decode_inst(ins)
            ins_list += [ins_decode.strip()]
            
        x_list_cat = np.stack(x_list)
        y_list_cat = np.stack(y_list)
        tok = tokenizer(ins_list, padding='max_length', truncation=True, return_tensors='pt')
        ins_list_cat = model(tok['input_ids'], tok['attention_mask']).last_hidden_state.detach().numpy()

        self.length = ins_list_cat.shape[0]

        print(type(x_list_cat), type(y_list_cat), type(ins_list_cat))
                
    
        self.data_list = {'x': x_list_cat, 'y': y_list_cat, 'ins': ins_list_cat}
        # self.dataset = TensorDataset(*data_list)

                
        
        # resize the shortest side to img_size and center crop
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_size, img_size), interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
            
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
        # x = torch.FloatTensor(self.data_list['x'][i])
        # y = torch.FloatTensor(self.data_list['y'][i])
        # ins = torch.FloatTensor(self.data_list['ins'][i])
        x = self.data_list['x'][i]
        y = self.data_list['y'][i]
        ins = self.data_list['ins'][i]
        # print(x.shape, y.shape, ins.shape)
        if y.ndim >= 4:
            y = y.squeeze()
        
        # if self.random_flip:
        #     if random.random() > 0.5:
        #         x = image.transpose(Image.FLIP_LEFT_RIGHT)
        x = self.transform(x)
        if y.ndim >= 4:
            img_list = []
            y = y.permute(1, 0, 2, 3)
            for img in y:
                img_list += self.transform(y)
            y = torch.stack(y)
            y = y.permute(1, 0, 2, 3)
        else:
            y = self.transform(y)
        example["x"] = x
        example["y"] = y
        example["ins"] = ins
        # if self.vit_norm:
        #     image_vit = self.transform_vit(image)
        #     example["pixel_values_vit"] = image_vit
        return example

if __name__ == "__main__":

    dataset_1 = GSLocalDataset(
            root='/shared/youngjoon/langtable/language_table',
            img_size=256,
            img_glob="**/*.png",
            section='train',
            predict_steps=1,
        )
    # dataset_2 = GSLocalDataset(
    #     root='/shared/youngjoon/langtable/language_table_sim',
    #     img_size=256,
    #     img_glob="**/*.png",
    #     section='train',
    #     predict_steps=1,
    # )
    # dataset_3 = GSLocalDataset(
    #     root='/shared/youngjoon/langtable/language_table_blocktoblock_sim',
    #     img_size=256,
    #     img_glob="**/*.png",
    #     section='train',
    #     predict_steps=1,
    # )
    # dataset_4 = GSLocalDataset(
    #     root='/shared/youngjoon/langtable/language_table_blocktoblock_4block_sim',
    #     img_size=256,
    #     img_glob="**/*.png",
    #     section='train',
    #     predict_steps=1,
    # )

    # train_dataset = ConcatDataset([dataset_1, dataset_2, dataset_3, dataset_4])

    val_dataset_1 = GSLocalDataset(
        root='/shared/youngjoon/langtable/language_table',
        img_size=256,
        img_glob="**/*.png",
        section='val',
        predict_steps=1,
    )
    # val_dataset_2 = GSLocalDataset(
    #     root='/shared/youngjoon/langtable/language_table_sim',
    #     img_size=256,
    #     img_glob="**/*.png",
    #     section='val',
    #     predict_steps=1,
    # )
    # val_dataset_3 = GSLocalDataset(
    #     root='/shared/youngjoon/langtable/language_table_blocktoblock_sim',
    #     img_size=256,
    #     img_glob="**/*.png",
    #     section='val',
    #     predict_steps=1,
    # )
    # val_dataset_4 = GSLocalDataset(
    #     root='/shared/youngjoon/langtable/language_table_blocktoblock_4block_sim',
    #     img_size=256,
    #     img_glob="**/*.png",
    #     section='val',
    #     predict_steps=1,
    # )

    # val_dataset = ConcatDataset([val_dataset_1, val_dataset_2, val_dataset_3, val_dataset_4])

    torch.save(dataset_1, '/shared/youngjoon/trn_single.pth', pickle_protocol=4)
    torch.save(val_dataset_1, '/shared/youngjoon/val_single.pth', pickle_protocol=4)
    # print("saved successfully")
    # dataset = GSLocalDataset(
    #     root="/shared/s2/lab01/dataset/lsd/language_table",
    #     # root='gs://gresearch/robotics/language_table/0.0.1/',
    #     # root='/home/s2/youngjoonjeong/github/latent-slot-diffusion/',
    #     img_size=256,
    #     img_glob="**/*.png",
    #     section='test',
    #     predict_steps=1
    #     # img_glob='*.out',
    # )



    # dataset = GlobDataset(
    #     root="/shared/s2/lab01/dataset/lsd/movi_/movi-e/movi-e-train-with-label/images/",
    #     # root='gs://gresearch/robotics/language_table/0.0.1/',
    #     # root='/home/s2/youngjoonjeong/github/latent-slot-diffusion/',
    #     img_size=256,
    #     img_glob="**/*.png",
    #     # img_glob='*.out',
    #     data_portion=(0.0, 0.9)
    # )

    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--predict_steps",
    #     type=int,
    #     default=4,
    #     help="Number of images that should be predicted.",
    # )
    # parser.add_argument(
    #     "--data_split",
    #     type=str,
    #     default='language_table_blocktoblock_4block_sim',
    #     help="Data Split",
    #     choices=['language_table_blocktoblock_4block_sim', 'language_table', 'language_table_sim', 'language_table_blocktoblock_sim',
    #     'language_table_blocktoblock_oracle_sim', 'language_table_blocktoblockrelative_oracle_sim', 'language_table_blocktoabsolute_oracle_sim',
    #     'language_table_blocktorelative_oracle_sim', 'language_table_separate_oracle_sim']
    # )

    # parser.add_argument(
    #     "--section",
    #     type=str,
    #     default='train',
    #     help="Train or Validation",
    #     choices=['train', 'val']
    # )

    # args = parser.parse_args()

    # # dataset = GlobDataset(
    # #     root="/shared/s2/lab01/dataset/lsd/language_table_blocktoblock_4block_sim/language_table_blocktoblock_4block_sim/language_table_blocktoblock_4block_sim-train-with-label/images/",
    # #     img_size=256,
    # #     img_glob="**/*.png",
    # #     data_portion=(0.0, 0.9)
    # # )
    # # pass
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # dataset = GSDataset(
    #     data_split=args.data_split,
    #     section=args.section,
    #     img_size=256,
    #     predict_steps=args.predict_steps,
    # )
    # print(dataset)

    # train_dataloader = torch.utils.data.DataLoader(
    #                                                 dataset,
    #                                                 batch_size=16,
    #                                                 shuffle=True,
    #                                                 num_workers=4,
    # )
    # for i, x in enumerate(train_dataloader):
    #     if i >= 1:
    #         break
    #     print(x['x'][0], x['y'][0], x['ins'][0])

    # torch.save(dataset, f'/shared/s2/lab01/dataset/lsd/{args.data_split}_predict{args.predict_steps}_{args.section}.pth', pickle_protocol=4)
    # print("saved successfully")

    # dataset_loaded = torch.load(f'/shared/s2/lab01/dataset/lsd/{args.data_split}_predict{args.predict_steps}_{args.section}.pth')

    # train_dataloader = torch.utils.data.DataLoader(
    #                                                 dataset_loaded,
    #                                                 batch_size=16,
    #                                                 shuffle=True,
    #                                                 num_workers=4,
    # )

    # for i, x in enumerate(train_dataloader):
    #     if i >= 1:
    #         break
    #     print(x['x'][0], x['y'][0], x['ins'][0])

    # train_dataloader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=16,
    #     shuffle=True,
    #     num_workers=4
    # )

    # for i, x in enumerate(train_dataloader):
    #     if i >= 1:
    #         break
    #     print(x)
    
    pass