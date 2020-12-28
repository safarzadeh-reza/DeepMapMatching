
import torch
import torch.nn as nn
import torch.optim as optim

import time

from tensorboardX import SummaryWriter
from src.DataManager import DataManager
from models.models_attention import EncoderRNN, DecoderRNN, Seq2Seq


class AttentionEDTrain(object):
    def __init__(self, params, device, data_manager: DataManager):
        self.params = params

        self.encoder = EncoderRNN(params["Encoder_in_feature"],
                                  params["Encoder_embbeding_size"],
                                  params["Encoder_hidden_size"],
                                  params["Encoder_num_layers"],
                                  params["Encoder_dropout"]).to(device)

        self.decoder = DecoderRNN(params["Decoder_embbeding_size"],
                                  params["Decoder_hidden_size"],
                                  params["Decoder_output_size"],
                                  params["Decoder_num_layers"],
                                  params["Decoder_dropout"]).to(device)

        self.model = Seq2Seq(self.encoder, self.decoder,
                             params["Model_input_pad_index"],
                             device).to(device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=params["learning_rate"], eps=1e-10)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

        self.model_summary_writer = SummaryWriter('log/encoder_decoder_{}'.format(time.time()/60))
        self.data_manager = data_manager

    def train(self):
        self.encoder.train()
        self.decoder.train()
        self.model.train()

        acc_result = []
        loss_result = []

        for input, label, length in self.data_manager.train_loader:
            length, idx = torch.sort(length, descending=True)
            input = input[idx]
            label = label[idx]

            length = length.to(self.device)
            input = input.to(self.device)
            label = label.to(self.device)

            input = input.permute(1, 0, 2)
            label = label.permute(1, 0)

            output = self.model(input, length, label)[0]

            output_dim = output.shape[-1]
            output_1 = output[1:].reshape(-1, output_dim)
            trg = label[1:].reshape(-1)
            loss = self.criterion(output_1, trg)
            acc = self.get_accuracy(output, label)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params["clip"])
            self.optimizer.step()

            acc_result.append(acc)
            loss_result.append(loss.item())

        ACC = sum(acc_result)/len(acc_result)
        LOSS = sum(loss_result)/len(loss_result)
        return ACC, LOSS

    def test(self):
        self.encoder.eval()
        self.decoder.eval()
        self.model.eval()

        return 0

    def get_accuracy(self, output, target):
        return 0

    def get_attention_plot(self):
        # summary.add_figure
        return 0
