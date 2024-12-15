import torch as t
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import sys


class SaliencyLoss(nn.Module):
    def __init__(self):
        super(SaliencyLoss, self).__init__()

    def forward(self, preds, labels, loss_type='cc'):
        losses = []
        if loss_type == 'cc':
            for i in range(labels.shape[0]):  # labels.shape[0] is batch size
                loss = loss_CC(preds[i], labels[i])
                losses.append(loss)

        elif loss_type == 'kldiv':
            for i in range(labels.shape[0]):
                loss = loss_KLdiv(preds[i], labels[i])
                losses.append(loss)

        elif loss_type == 'sim':
            for i in range(labels.shape[0]):
                loss = loss_similarity(preds[i], labels[i])
                losses.append(loss)

        elif loss_type == 'nss':
            for i in range(labels.shape[0]):
                loss = loss_NSS(preds[i], labels[i])
                losses.append(loss)

        elif loss_type == 'nss_new':
            for i in range(labels.shape[0]):
                loss = loss_NSS_new(preds[i], labels[i])
                losses.append(loss)

        elif loss_type == 'kldiv_new':
            for i in range(labels.shape[0]):
                loss = loss_kl_new(preds[i], labels[i])
                losses.append(loss)

        elif loss_type == 'ce':
            loss = loss_cross_entropy(preds, labels)
            losses.append(loss)

        elif loss_type == 'tvdist':
            loss = TVdist(preds, labels)
            losses.append(loss)

        return t.stack(losses).mean(dim=0, keepdim=True)


def loss_KLdiv(pred_map, gt_map):
    eps = 2.2204e-16
    eps = sys.float_info.epsilon
    pred_map = pred_map / t.sum(pred_map)
    gt_map = gt_map / t.sum(gt_map)
    div = t.sum(t.mul(gt_map, t.log(eps + t.div(gt_map, pred_map + eps))))
    return div


def loss_CC(pred_map, gt_map):
    gt_map_ = (gt_map - t.mean(gt_map))
    pred_map_ = (pred_map - t.mean(pred_map))
    cc = t.sum(t.mul(gt_map_, pred_map_)) / t.sqrt(t.sum(t.mul(gt_map_, gt_map_)) * t.sum(t.mul(pred_map_, pred_map_)))
    return cc


def loss_similarity(pred_map, gt_map):
    gt_map = (gt_map - t.min(gt_map)) / (t.max(gt_map) - t.min(gt_map))
    gt_map = gt_map / t.sum(gt_map)

    pred_map = (pred_map - t.min(pred_map)) / (t.max(pred_map) - t.min(pred_map))
    pred_map = pred_map / t.sum(pred_map)

    diff = t.min(gt_map, pred_map)
    score = t.sum(diff)

    return score


def loss_NSS(pred_map, fix_map):
    '''ground truth here is fixation map'''

    pred_map_ = (pred_map - t.mean(pred_map)) / t.std(pred_map)
    mask = fix_map.gt(0)
    score = t.mean(t.masked_select(pred_map_, mask))
    return score


def loss_NSS_new(pred_map, fix_map):
    pred_map = pred_map[0]
    fix_map = fix_map[0]
    pred_map_ = (pred_map - t.mean(pred_map)) / t.std(pred_map)
    sum = 0
    count = 0
    for i in range(pred_map_.shape[0]):
        for j in range(pred_map_.shape[1]):
            if fix_map[i][j] != 0:
                sum += pred_map_[i][j]
                count += 1
    score = (float)(sum) / (count)
    return score


def auc_judd(s_map, gt):
    # ground truth is discrete, s_map is continous and normalized
    # thresholds are calculated from the salience map, only at places where fixations are present
    thresholds = []
    for i in range(0, gt.shape[0]):
        for k in range(0, gt.shape[1]):
            if gt[i][k] > 0:
                thresholds.append(s_map[i][k])

    num_fixations = np.sum(gt)
    # num fixations is no. of salience map values at gt >0

    thresholds = sorted(set(thresholds))

    # fp_list = []
    # tp_list = []
    area = []
    area.append((0.0, 0.0))
    for thresh in thresholds:
        # in the salience map, keep only those pixels with values above threshold
        temp = np.zeros(s_map.shape)
        temp[s_map >= thresh] = 1.0
        # assert np.max(gt) == 1.0, 'something is wrong with ground truth..not discretized properly max value > 1.0'
        # assert np.max(s_map) == 1.0, 'something is wrong with salience map..not normalized properly max value > 1.0'
        num_overlap = np.where(np.add(temp, gt) == 2)[0].shape[0]
        tp = num_overlap / (num_fixations * 1.0)

        # total number of pixels > threshold - number of
        # pixels that overlap with gt / total number of non fixated pixels
        # this becomes nan when gt is full of fixations..this won't happen
        fp = (np.sum(temp) - num_overlap) / ((np.shape(gt)[0] * np.shape(gt)[1]) - num_fixations)

        area.append((round(tp, 4), round(fp, 4)))
    # tp_list.append(tp)
    # fp_list.append(fp)

    # tp_list.reverse()
    # fp_list.reverse()
    area.append((1.0, 1.0))
    # tp_list.append(1.0)
    # fp_list.append(1.0)
    # print tp_list
    area.sort(key=lambda x: x[0])
    tp_list = [x[0] for x in area]
    fp_list = [x[1] for x in area]
    return np.trapz(np.array(tp_list), np.array(fp_list))


def loss_kl_new(x, y):
    logp_x = F.log_softmax(x, dim=-1)
    p_y = F.softmax(y, dim=-1)

    kl_sum = F.kl_div(logp_x, p_y, reduction='sum')
    kl_mean = F.kl_div(logp_x, p_y, reduction='mean')
    return kl_sum


def loss_cross_entropy(output, label):
    batch_size = output.size(0)
    output = output.view(batch_size, -1)
    label = label.view(batch_size, -1)

    final_loss = F.binary_cross_entropy_with_logits(output, label, reduce=False).sum(1)
    final_loss = final_loss.mean()
    return final_loss


def TVdist(y_pred, y, eps=1e-7):
    P = y / (eps + y.sum(dim=[1, 2, 3]).view(-1, 1, 1, 1))
    Q = y_pred / (eps + y_pred.sum(dim=[1, 2, 3]).view(-1, 1, 1, 1))
    tv = torch.sum(torch.abs(P - Q)) * 0.5
    return tv


class EdgeSaliencyLoss(nn.Module):
    def __init__(self, device, alpha_sal=0.7):
        super(EdgeSaliencyLoss, self).__init__()

        self.alpha_sal = alpha_sal

        self.laplacian_kernel = torch.tensor([[-1., -1., -1.], [-1., 8., -1.], [-1., -1., -1.]], dtype=torch.float,
                                             requires_grad=False)
        self.laplacian_kernel = self.laplacian_kernel.view((1, 1, 3, 3))  # Shape format of weight for convolution
        self.laplacian_kernel = self.laplacian_kernel.to(device)

    @staticmethod
    def weighted_bce(input_, target, weight_0=1.0, weight_1=1.0, eps=1e-15):
        wbce_loss = -weight_1 * target * torch.log(input_ + eps) - weight_0 * (1 - target) * torch.log(
            1 - input_ + eps)
        return torch.mean(wbce_loss)

    def forward(self, y_pred, y_gt):
        # Generate edge maps
        y_gt_edges = F.relu(torch.tanh(F.conv2d(y_gt, self.laplacian_kernel, padding=(1, 1))))
        y_pred_edges = F.relu(torch.tanh(F.conv2d(y_pred, self.laplacian_kernel, padding=(1, 1))))

        # sal_loss = F.binary_cross_entropy(input=y_pred, target=y_gt)
        sal_loss = self.weighted_bce(input_=y_pred, target=y_gt, weight_0=1.0, weight_1=1.12)
        edge_loss = F.binary_cross_entropy(input=y_pred_edges, target=y_gt_edges)

        total_loss = self.alpha_sal * sal_loss + (1 - self.alpha_sal) * edge_loss
        return total_loss


def _pointwise_loss(lambd, input, target, size_average=True, reduce=True):
    d = lambd(input, target)
    if not reduce:
        return d
    return torch.mean(d) if size_average else torch.sum(d)


class KLDLoss(nn.Module):
    def __init__(self):
        super(KLDLoss, self).__init__()

    def KLD(self, inp, trg):
        inp = inp / torch.sum(inp)
        trg = trg / torch.sum(trg)
        eps = sys.float_info.epsilon

        return torch.sum(trg * torch.log(eps + torch.div(trg, (inp + eps))))

    def forward(self, inp, trg):
        return _pointwise_loss(lambda a, b: self.KLD(a, b), inp, trg)


def AUC_Borji(s_map, gt, Nsplits=100, stepSize=0.1):
    if np.sum(gt) <= 1:
        print('no fixationMap')
        return
    saliencyMap = s_map.astype(float)
    fixationMap = gt.astype(float)
    saliencyMap = (saliencyMap - np.min(saliencyMap)) / (np.max(saliencyMap) - np.min(saliencyMap))
    S = saliencyMap
    S = np.reshape(S, S.shape[0] * S.shape[1], order='F')
    F = fixationMap
    F = np.reshape(F, F.shape[0] * F.shape[1], order='F')
    Sth = S[np.where(F > 0)]
    Nfixations = len(Sth)
    Npixels = len(S)
    r = np.random.randint(Npixels, size=(Nfixations, Nsplits))
    randfix = S[r]

    auc = [0] * Nsplits
    for s in range(Nsplits):
        curfix = randfix[:, s]
        temp = list(Sth)
        temp.extend(list(curfix))
        allthreshes = np.arange(0, np.max(temp) + stepSize, stepSize)
        allthreshes = allthreshes[::-1]
        tp = np.zeros(len(allthreshes) + 2)
        fp = np.zeros(len(allthreshes) + 2)
        tp[0] = 0
        tp[-1] = 1
        fp[0] = 0
        fp[-1] = 1
        for i in range(len(allthreshes)):
            thresh = allthreshes[i]
            tp[i + 1] = np.sum(len(np.where(Sth >= thresh)[0])) / (float)(Nfixations)
            fp[i + 1] = np.sum(len(np.where(curfix >= thresh)[0])) / (float)(Nfixations)
        auc[s] = np.trapz(x=fp, y=tp)
        # print(auc)
    score = np.mean(auc)
    return score


def AUC_Judd(s_map, gt, jitter=1):
    saliencyMap = s_map.astype(float)
    fixationMap = gt.astype(float)
    if jitter:
        saliencyMap = saliencyMap + np.random.rand(fixationMap.shape[0], fixationMap.shape[1]) / 10000000.0
    saliencyMap = (saliencyMap - np.min(saliencyMap)) / (np.max(saliencyMap) - np.min(saliencyMap))
    S = saliencyMap
    S = np.reshape(S, S.shape[0] * S.shape[1], order='F')
    F = fixationMap
    F = np.reshape(F, F.shape[0] * F.shape[1], order='F')
    Sth = S[np.where(F > 0)]
    Nfixations = len(Sth)
    Npixels = len(S)
    allthreshes = np.sort(Sth, axis=None)
    allthreshes = allthreshes[::-1]
    tp = np.zeros(Nfixations + 2)
    fp = np.zeros(Nfixations + 2)
    tp[0] = 0
    tp[-1] = 1
    fp[0] = 0
    fp[-1] = 1
    for i in range(0, Nfixations):
        thresh = allthreshes[i]
        aboveth = np.sum(len(np.where(S >= thresh)[0]))
        tp[i + 1] = (float)(i) / Nfixations
        fp[i + 1] = (float)(aboveth - i) / (Npixels - Nfixations)
    score = np.trapz(x=fp, y=tp)
    return score
