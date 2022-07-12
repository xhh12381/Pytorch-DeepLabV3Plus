import torch
from torch import Tensor


def dice_coeff(input: Tensor, target: Tensor, epsilon=1e-6):
    assert input.size() == target.size()

    if input.dim() == 2:  # 无 batch 维度
        # dot:计算具有相同元素数量的两个一维张量的内积(点乘),结果是一个标量,reshape(-1):展平为1维张量
        inter = torch.dot(input.reshape(-1), target.reshape(-1))  # 交集
        sets_sum = torch.sum(input) + torch.sum(target)  # 并集
        if sets_sum.item() == 0:
            sets_sum = 2 * inter
        return (2 * inter + epsilon) / (sets_sum + epsilon)
    elif input.dim() == 3:  # 有 batch 维度
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]
    else:
        raise ValueError("The dimension of the input tensor must be 2 or 3!")


def multiclass_dice_coeff(input: Tensor, target: Tensor):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):
        dice += dice_coeff(input[:, channel, ...], target[:, channel, ...])  # 降维：(B,C,H,W)->(B,H,W)

    return dice / input.shape[1]


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target)
