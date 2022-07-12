import torch
from torch import nn
from torch.nn import functional as F


class FocalLoss(nn.Module):
    def __init__(self, num_classes, alpha=0.25, gamma=2, size_average=True):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:   阿尔法α,类别权重. 当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ...]
        :param gamma:   伽马γ,难易样本调节参数
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """
        super().__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha) == num_classes  # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1  # 如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1 - alpha)  # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]

        self.gamma = gamma

    def forward(self, preds, labels):
        """
        preds: (B,C,H,W)
        labels: (B,H,W)
        """

        # 固定类别维度，其余合并
        preds = preds.permute(0, 2, 3, 1)
        preds = preds.reshape(-1, preds.size(-1))
        self.alpha = self.alpha.to(preds.device)

        # 使用log_softmax解决溢出问题，方便交叉熵计算而不用考虑值域
        preds_log_softmax = F.log_softmax(preds, dim=1)

        # log_softmax是softmax+log运算，那再exp就算回去了变成softmax
        preds_softmax = torch.exp(preds_log_softmax)

        # 这部分实现nll_loss ( cross_entropy = log_softmax + nll)
        preds_softmax = preds_softmax.gather(dim=1, index=labels.reshape(-1, 1))
        preds_log_softmax = preds_log_softmax.gather(dim=1, index=labels.reshape(-1, 1))

        self.alpha = self.alpha.gather(dim=0, index=labels.reshape(-1))

        # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

        # torch.mul() 矩阵对应位置相乘（点乘）
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma), preds_log_softmax)

        # torch.t()求转置
        loss = torch.mul(self.alpha, loss.t().squeeze())

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()

        return loss


if __name__ == '__main__':
    preds = torch.rand(size=(1, 2, 10, 10))
    labels = torch.randint(high=2, size=(1, 10, 10))

    focal_loss = FocalLoss(num_classes=2)
    print(focal_loss(preds, labels))
