import logging

import torch
import numpy as np
import os

from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler, TensorDataset, Dataset
import json

def np_read_with_tensor_output(file):
    with open(file, "rb") as outfile:
        data = np.load(outfile)
    return torch.from_numpy(data.astype(np.float32))

def read_one_json_results(path):
    with open(path, "r") as outfile:
        data = json.load(outfile)
    return data

class FeatureDataset(Dataset):
    def __init__(self, input_dir, annotation_dir, args, length = 0, shift = 0):
        self.annotations = np_read_with_tensor_output(annotation_dir)
        self.lens = self.annotations.shape[0] if length == 0 else length
        self.shift = shift
        self.annotation_dir = input_dir + "/annotation/"
        self.feature_dir = input_dir + "/LLALFeature"

    def __getitem__(self, index):
        index = index + self.shift
        one_annotation = read_one_json_results(self.annotation_dir+ str(index)+".json")
        feature_index = one_annotation['LLALFeature']
        features = []
        for i in range(6):
            path = os.path.join(self.feature_dir, str(i), str(feature_index)+".npy")
            feature = np_read_with_tensor_output(path)
            features.append(feature)
        annotation = self.annotations[index]
        return tuple((features, annotation))

    def __len__(self):
        return self.lens

def get_loader_feature(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
        
    model_data_path = args.data_dir

    split = "train"
    inputs_path = model_data_path + split
    store_preprocess_annotations_path = model_data_path + split + "/image_true_losses.npy"
    train_datasets = FeatureDataset(inputs_path, store_preprocess_annotations_path, args)
    
    split = "val"
    inputs_path = model_data_path + split
    store_preprocess_annotations_path = model_data_path + split + "/image_true_losses.npy"
    test_datasets = FeatureDataset(inputs_path, store_preprocess_annotations_path, args)

    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler(train_datasets)
    test_sampler = SequentialSampler(test_datasets)
    train_loader = DataLoader(train_datasets,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=4,
                              pin_memory=True)
    test_loader = DataLoader(test_datasets,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=4,
                             pin_memory=True) if test_datasets is not None else None

    return train_loader, test_loader, train_datasets, test_datasets