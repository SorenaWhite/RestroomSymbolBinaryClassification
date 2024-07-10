import torch
import torch.nn as nn
import torch.nn.functional as F

import math


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_classes=2):
        super(MultiHeadAttention, self).__init__()
        self.depth = 4

        # 定义线性层和输出线性层
        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        self.final_linear = nn.Linear(d_model, d_model)

        self.mlp_head = nn.Sequential(
            nn.Linear(d_model, num_classes)
        )  # self.depth * 2 ** (len(self.depths) - 1)

    def forward(self, query, key, value, mask=None):
        # 1. 线性层和分割到多头
        query = self.query_linear(query)
        key = self.query_linear(key)
        value = self.query_linear(value)

        # 2. 缩放点积注意力
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.depth)
        if mask is not None:
            scores = scores.masked_fill(mask == True, float('-inf'))
        attention_weights = F.softmax(scores, dim=-1)

        # 3. 将注意力权重应用到值上
        output = torch.matmul(attention_weights, value)
        output = self.mlp_head(output)
        return output


if __name__ == '__main__':
    x1 = torch.randn(2, 512)
    x2 = torch.randn(2, 512)
    model = MultiHeadAttention(512)
    print(get_parameter_number(model))
    y = model(x1, x2, x2)
    print(y)
    torch.save(model, './model1.pth')
