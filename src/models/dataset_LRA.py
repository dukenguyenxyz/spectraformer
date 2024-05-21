# https://github.com/pkuzengqi/Skyformer
import torch
import torch.nn as nn
import math
from torch.utils.data.dataset import Dataset
import sys
import os
import random
import json
import pickle
import numpy as np

class LRADataset(Dataset):
    def __init__(self, file_path, endless, bucket_name=None):
        self.endless = endless

        if not bucket_name:
            with open(file_path, "rb") as f:
                self.examples = pickle.load(f)
                random.shuffle(self.examples)
                self.curr_idx = 0
                
            print(f"Loaded {file_path}... size={len(self.examples)}", flush = True)
        
        else:
            # GCP
            from google.cloud import storage
            blob_name = file_path
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)

            with blob.open("rb") as f:
                self.examples = pickle.load(f)
                random.shuffle(self.examples)
                self.curr_idx = 0
                
            print(f"Loaded bucket '{bucket_name}' at {file_path}... size={len(self.examples)}", flush = True)

    def __len__(self):
        if self.endless:
            return 1000000000
        else:
            return len(self.examples)

    def create_inst(self, inst):
        output = {}
        output["input_ids_0"] = torch.tensor(inst["input_ids_0"], dtype = torch.long)
        output["mask_0"] = (output["input_ids_0"] != 0).float()
        if "input_ids_1" in inst:
            output["input_ids_1"] = torch.tensor(inst["input_ids_1"], dtype = torch.long)
            output["mask_1"] = (output["input_ids_1"] != 0).float()
        output["label"] = torch.tensor(inst["label"], dtype = torch.long)
        return output
    
    def __getitem__(self, i):
        if not self.endless:
            return self.create_inst(self.examples[i])
        
        if self.curr_idx >= len(self.examples):
            random.shuffle(self.examples)
            self.curr_idx = 0
        inst = self.examples[self.curr_idx]
        self.curr_idx += 1
        
        return self.create_inst(inst)
