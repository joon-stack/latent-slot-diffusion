# conda create -n tfds && conda activate tfds && pip install tensorflow-datasets gcfs tqdm pillow
import os
from tqdm.auto import tqdm
import json
import argparse
from PIL import Image
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds 

parser = argparse.ArgumentParser()
parser.add_argument(
    '--dataset_split', 
    type=str, 
    default='language_table_blocktoblock_4block_sim',
    help='Dataset split to use (movi-a, movi-b, movi-c, movi-d, movi-e, movi-f)'
)
parser.add_argument(
    '--data_dir',
    type=str,
    default="/shared/youngjoon/langtable",
    help='Directory to save the dataset'
)

parser.add_argument(
    '--data_size',
    type=int,
    default=8000,
    help='Data size to download'
)

args = parser.parse_args()

# ds = tfds.load("movi_e", data_dir="gs://kubric-public/tfds") 

from tensorflow_datasets.core.utils import gcs_utils
gcs_utils._is_gcs_disabled = True

ds, ds_info = tfds.load(f"{args.dataset_split}:0.0.1", 
                        data_dir='gs://gresearch/robotics', with_info=True)
                        
# ds, ds_info = tfds.load("movi_e", 
#                         data_dir="gs://kubric-public/tfds", with_info=True)

# ds, ds_info = tfds.load(f"{args.dataset_split.replace('-', '_')}/2017", 
#                         data_dir="gs://kubric-public/tfds", with_info=True)                        

for section in ["train", "validation", "test"]:
    out_path_images = os.path.join(args.data_dir, f'{args.dataset_split}/{args.dataset_split}-{section}-with-label/images')
    out_path_labels = os.path.join(args.data_dir, f'{args.dataset_split}/{args.dataset_split}-{section}-with-label/labels')

    try:
        # builder = tfds.builder_from_directory(dataset_path)
        # dataset = builder.as_dataset(split=section)
        # print(dataset)
        dataset = tfds.as_numpy(ds[section])
        data_iter = iter(dataset)
    except:
        continue

    # to_tensor = transforms.ToTensor()

    class JsonEncoder(json.JSONEncoder):
        """ Special json encoder for numpy types """
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, tf.RaggedTensor):
                return obj.to_tensor().numpy().tolist()
            elif isinstance(obj, bytes):
                return obj.decode()

            return json.JSONEncoder.default(self, obj)

    b = 0
    progress_bar = tqdm(
        range(0, len(dataset)),
        initial=0,
        desc=f"{section} Steps",
        position=0, leave=True
    )
    for i, record in enumerate(data_iter):
        if i >= args.data_size:
            break
        path_vid_images = os.path.join(out_path_images, f"{i:08}")
        os.makedirs(path_vid_images, exist_ok=True)
        path_vid_labels = os.path.join(out_path_labels, f"{i:08}")
        os.makedirs(path_vid_labels, exist_ok=True)
        for t, step in enumerate(record['steps']):
            img = step['observation']['rgb']
            ins = step['observation']['instruction']
            Image.fromarray(img).save(os.path.join(path_vid_images, f"{t:08}_image.png"), optimize=False)
            np.save(path_vid_labels, ins)


        b += 1
        progress_bar.update(1)