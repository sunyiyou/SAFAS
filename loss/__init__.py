import torch.nn.functional as F
import torch.nn as nn
import torch


class ContrastLoss(nn.Module):
    
    def __init__(self):
        super(ContrastLoss, self).__init__()
        pass

    def forward(self, anchor_fea, reassembly_fea, contrast_label):
        contrast_label = contrast_label.float()
        anchor_fea = anchor_fea.detach()
        loss = -(F.cosine_similarity(anchor_fea, reassembly_fea, dim=-1))
        loss = loss*contrast_label
        return loss.mean()


def AAMS(logits, spoof_label, type_label, num_classes):
    s = 30
    log_terms = []
    type_label = type_label * 2
    spoof_label = list(spoof_label.data.cpu().numpy()) * 2
    for i, logit in enumerate(logits):
        cls = type_label[i]
        if spoof_label[i] == 1:# live
            m = 0.4
        else:# spoof
            m = 0.1
        pos_mask = F.one_hot(torch.Tensor([cls]).long(), num_classes=num_classes)[0].to(logits.device)
        neg_mask = 1 - pos_mask

        logit_am = s * (logit - pos_mask * m)
        logit_max = torch.max(logit_am)
        logit_am = logit_am - logit_max.detach()

        pos_term = (logit_am * pos_mask).sum() / (pos_mask.sum() + 1e-10)
        neg_term = (torch.exp(logit_am)).sum()

        log_term = pos_term - torch.log(neg_term + 1e-15)
        log_terms.append(log_term)

    loss = -sum(log_terms) / len(log_terms)
    return loss

def feat_sim_loss(feat1, feat2):
    return torch.norm(feat1 - feat2, dim=1).mean()


def supcon_loss(features, labels=None, mask=None, temperature = 0.1):
    base_temperature = 0.07
    """Compute loss for model. If both `labels` and `mask` are None,
    it degenerates to SimCLR unsupervised loss:
    https://arxiv.org/pdf/2002.05709.pdf
    Args:
        features: hidden vector of shape [bsz, n_views, ...].
        labels: ground truth of shape [bsz].
        mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
            has the same class as sample i. Can be asymmetric.
    Returns:
        A loss scalar.
    """
    device = (torch.device('cuda')
              if features.is_cuda
              else torch.device('cpu'))

    if len(features.shape) < 3:
        raise ValueError('`features` needs to be [bsz, n_views, ...],'
                         'at least 3 dimensions are required')
    if len(features.shape) > 3:
        features = features.view(features.shape[0], features.shape[1], -1)

    batch_size = features.shape[0]
    if labels is not None and mask is not None:
        raise ValueError('Cannot define both `labels` and `mask`')
    elif labels is None and mask is None:
        mask = torch.eye(batch_size, dtype=torch.float32).to(device)
    elif labels is not None:
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device)
    else:
        mask = mask.float().to(device)

    contrast_count = features.shape[1]
    contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
    anchor_feature = contrast_feature
    anchor_count = contrast_count

    # compute logits
    anchor_dot_contrast = torch.div(
        torch.matmul(anchor_feature, contrast_feature.T),
    temperature)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # tile mask
    mask = mask.repeat(anchor_count, contrast_count)
    # mask-out self-contrast cases
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
        0
    )
    mask = mask * logits_mask

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

    # loss
    loss = - (temperature / base_temperature) * mean_log_prob_pos
    loss = loss.view(anchor_count, batch_size).mean()

    return loss



def simclr_loss(features):

    labels = torch.cat([torch.arange(len(features) // 2) for i in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    device = (torch.device('cuda')
              if features.is_cuda
              else torch.device('cpu'))
    labels = labels.to(device)

    # features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)
    # assert similarity_matrix.shape == (
    #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / 0.7
    criterion = torch.nn.CrossEntropyLoss().to(device)
    return criterion(logits, labels)