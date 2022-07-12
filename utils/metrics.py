import numpy as np

__all__ = ['SegmentationMetric']

"""
confusionMetric  # 注意：此处横着代表真实值，竖着代表预测值
      P     N
P     TP    FP
N     FN    TN
"""


class SegmentationMetric(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((self.num_classes,) * 2)

    def pixel_accuracy(self):
        # return all class overall pixel accuracy
        #  PA = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return acc

    def class_pixel_accuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        class_acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=0)
        return class_acc  # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率

    def mean_pixel_accuracy(self):
        class_acc = self.class_pixel_accuracy()
        mean_acc = np.nanmean(class_acc)  # np.nanmean 求平均值，nan表示遇到Nan类型，其值取为0
        return mean_acc  # 返回单个值，如：np.nanmean([0.90, 0.80, 0.96, nan, nan]) = (0.90 + 0.80 + 0.96） / 3 =  0.89

    def mean_intersection_over_union(self):
        # Intersection = TP, Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusion_matrix)  # 取对角元素的值，返回列表
        union = np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) - np.diag(
            self.confusion_matrix)  # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表
        IoU = intersection / union  # 返回列表，其值为各个类别的IoU
        mIoU = np.nanmean(IoU)  # 求各类别IoU的平均
        return mIoU

    def gen_confusion_matrix(self, img_predict, img_label):  # 同FCN中score.py的fast_hist()函数
        # remove classes from unlabeled pixels in gt image and predict
        mask = (img_label >= 0) & (img_label < self.num_classes)
        label = self.num_classes * img_label[mask] + img_predict[mask]
        count = np.bincount(label, minlength=self.num_classes ** 2)
        confusion_matrix = count.reshape(self.num_classes, self.num_classes)
        return confusion_matrix

    def frequency_weighted_intersection_over_union(self):
        # FWIOU = [(TP + FN) / (TP + FP + TN + FN)] * [TP / (TP + FP + FN)]
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (np.sum(self.confusion_matrix, axis=1)
                                               + np.sum(self.confusion_matrix, axis=0) - np.diag(self.confusion_matrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def add_batch(self, img_predict, img_label):
        assert img_predict.shape == img_label.shape
        self.confusion_matrix += self.gen_confusion_matrix(img_predict, img_label)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))


if __name__ == '__main__':
    imgPredict = np.array([2, 0, 1, 1, 2, 2])  # 可直接换成预测图片
    imgLabel = np.array([0, 0, 1, 1, 2, 2])  # 可直接换成标注图片
    metric = SegmentationMetric(3)  # 3表示有3个分类，有几个分类就填几
    metric.add_batch(imgPredict, imgLabel)
    print(metric.confusion_matrix)
    pa = metric.pixel_accuracy()
    cpa = metric.class_pixel_accuracy()
    mpa = metric.mean_pixel_accuracy()
    mIoU = metric.mean_intersection_over_union()
    print('pa is : %f' % pa)
    print('cpa is :')  # 列表
    print(cpa)
    print('mpa is : %f' % mpa)
    print('mIoU is : %f' % mIoU)
