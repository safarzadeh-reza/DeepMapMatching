
from src.util import normalization, make_input_mask, make_target_mask
import pandas as pd
from models.Transformer import *
import numpy as np
import torch.nn as nn
import torch
import os


def datacombination(path, PAD):
    filelist = os.listdir(path)
    df_len = []
    total_df = []
    for file in filelist:
        df = pd.read_csv(path+file)
        df = df.sort_values(by=['time'])
        df_len.append(len(df))
        test_df = df.values[:, 1:3]
        total_df.append(test_df)

    max_len = max(df_len)

    for i, length in enumerate(df_len):
        temp = total_df[i]
        added_row = np.tile([PAD, PAD], ((max_len-length), 1))
        temp = np.append(temp, added_row, axis=0)
        total_df[i] = np.expand_dims(temp, axis=0)
    total_df = np.concatenate(total_df)
    df_len = np.array(df_len)

    return (total_df, df_len, filelist)


file_path = 'validation_data/validation_data_1/'
val_file_list = os.listdir(file_path)
test_input_raw, test_len, filelist_raw = datacombination(file_path, -1)

for num in range(len(val_file_list)):
    test_input = test_input_raw[num]
    # test_input = test_input[num][90:180]
    test_input = np.expand_dims(test_input, axis=0)
    # print(test_input)
    # test_len = np.array([75])
    filelist = filelist_raw[num]

    PATH = "pytorch_model/transformer/best_transformer.pt"
    model = torch.load(PATH)

    max_len = 7
    device = "cuda"
    boundaries = [127.015, 127.095, 37.47, 37.55]

    test_input = normalization(
        test_input, boundaries[0], boundaries[1], boundaries[2], boundaries[3])
    test_input = torch.Tensor(test_input).to(device)
    # test_input = test_input.unsqueeze(0)

    encoder = model.encoder
    decoder = model.decoder

    # test_target_1 = torch.empty(2, 10)
    # test_target_1.fill_(0)
    # test_target_1 = test_target_1.type(torch.LongTensor).to('cuda')
    trg_indexes = []
    #trg_indexes = [[229]]
    for i in range(1):
        trg_indexes.append([229])

    model = model.to(device)
    for i in range(20):
        trg_tensor = torch.LongTensor(trg_indexes).to(device)
        test_target_mask = make_target_mask(trg_tensor, device)
        test_input_mask = test_input[:, :, 0]

        # test_input_mask = make_input_mask(
        #     test_input_mask, -1, device)
        # test_input_mask = None
        with torch.no_grad():
            output = model(test_input, trg_tensor,
                           test_input_mask, test_target_mask)
            pred_token = output.argmax(2)[:, -1]
            for j in range(1):
                trg_indexes[j].append(pred_token[j].item())

        if pred_token[j].item() == 230:
            break

    out = np.array(trg_indexes)

    print(out[0][1:-1])
    print(filelist)

    input("next")
