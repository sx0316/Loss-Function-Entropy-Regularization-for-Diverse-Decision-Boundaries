"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
EPS=1e-8


class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(MaskedCrossEntropyLoss, self).__init__()

    def forward(self, input, target, mask, weight, reduction='mean'):
        if not (mask != 0).any():
            raise ValueError('Mask in MaskedCrossEntropyLoss is all zeros.')
        target = torch.masked_select(target, mask)
        b, c = input.size()
        n = target.size(0)
        input = torch.masked_select(input, mask.view(b, 1)).view(n, c)
        return F.cross_entropy(input, target, weight = weight, reduction = reduction)


class ConfidenceBasedCE(nn.Module):
    def __init__(self, threshold, apply_class_balancing):
        super(ConfidenceBasedCE, self).__init__()
        self.loss = MaskedCrossEntropyLoss()
        self.softmax = nn.Softmax(dim = 1)
        self.threshold = threshold
        self.apply_class_balancing = apply_class_balancing

    def forward(self, anchors_weak, anchors_strong):
        """
        Loss function during self-labeling
        input: logits for original samples and for its strong augmentations
        output: cross entropy
        """
        # Retrieve target and mask based on weakly augmentated anchors
        weak_anchors_prob = self.softmax(anchors_weak)
        max_prob, target = torch.max(weak_anchors_prob, dim = 1)
        mask = max_prob > self.threshold
        b, c = weak_anchors_prob.size()
        target_masked = torch.masked_select(target, mask.squeeze())
        n = target_masked.size(0)

        # Inputs are strongly augmented anchors
        input_ = anchors_strong

        # Class balancing weights
        if self.apply_class_balancing:
            idx, counts = torch.unique(target_masked, return_counts = True)
            freq = 1/(counts.float()/n)
            weight = torch.ones(c).cuda()
            weight[idx] = freq

        else:
            weight = None

        # Loss
        loss = self.loss(input_, target, mask, weight = weight, reduction='mean')

        return loss


def entropy(x, input_as_probabilities):
    """
    Helper function to compute the entropy over the batch
    input: batch w/ shape [b, num_classes]
    output: entropy value [is ideally -log(num_classes)]
    """

    if input_as_probabilities:
        x_ =  torch.clamp(x, min = EPS)
        b =  x_ * torch.log(x_)
    else:
        b = F.softmax(x, dim = 1) * F.log_softmax(x, dim = 1)

    if len(b.size()) == 2: # Sample-wise entropy
    #    print("Sample Entropy")
        return -b.sum(dim = 1).mean()

    elif len(b.size()) == 1: # Distribution-wise entropy
    #    print("Distribution Entropy")
        return - b.sum()
    else:
        raise ValueError('Input tensor is %d-Dimensional' %(len(b.size())))


class SCANLoss(nn.Module):
    def __init__(self, entropy_weight = 2.0):
        super(SCANLoss, self).__init__()
        self.softmax = nn.Softmax(dim = 1)
        self.bce = nn.BCELoss()
        self.entropy_weight = entropy_weight # Default = 2.0
        self.diff_weight = 4
        self.third_weight = 16

    def forward(self, anchors, neighbors):
        """
        input:
            - anchors: logits for anchor images w/ shape [b, num_classes]
            - neighbors: logits for neighbor images w/ shape [b, num_classes]
        output:
            - Loss
        """
        # Softmax
        b, n = anchors.size()
        anchors_prob = self.softmax(anchors)
        positives_prob = self.softmax(neighbors)

        # Similarity in output space
        similarity = torch.bmm(anchors_prob.view(b, 1, n), positives_prob.view(b, n, 1)).squeeze()
        ones = torch.ones_like(similarity)
        consistency_loss = self.bce(similarity, ones)

        lg_sim = torch.log(similarity)
      #  print("cross_term", lg_sim.size())

        # Entropy loss
        entropy_loss = entropy(torch.mean(anchors_prob, 0), input_as_probabilities = True)
        dissim_loss = entropy(torch.mean(anchors_prob, 0), input_as_probabilities = True) - entropy(torch.mean(positives_prob, 0), input_as_probabilities = True)
        cross_loss = torch.dot(torch.clamp(similarity, min = EPS), torch.clamp(lg_sim, min = EPS)).squeeze()
        second_order_loss = torch.mean(cross_loss, 0)

        tiled_sim = np.tile(similarity.cpu().detach().numpy(), n)
        tiled_sim = torch.from_numpy(tiled_sim)
        tiled_sim = torch.reshape(tiled_sim, (n, b)).cuda()
 #       print(similarity.size(), anchors_prob.size(), test_sim2.size())

        third_order_entropy = torch.bmm(anchors_prob.view(b, 1, n), tiled_sim.view(b, n, 1)).squeeze()
        lg_third = torch.log(third_order_entropy)
        third_order_loss = torch.dot(torch.clamp(third_order_entropy, min = EPS), torch.clamp(lg_third, min = EPS)).squeeze()
        third_order_loss = torch.mean(third_order_loss, 0) / n
        self.third_weight = 0.5/np.sqrt(n)
        self.diff_weight = 0.25/n

        total_loss = consistency_loss - self.entropy_weight * entropy_loss + self.diff_weight * second_order_loss - self.third_weight * third_order_loss

        return total_loss, consistency_loss, entropy_loss, second_order_loss, third_order_loss


class SimCLRLoss(nn.Module):
    # Based on the implementation of SupContrast
    def __init__(self, temperature):
        super(SimCLRLoss, self).__init__()
        self.temperature = temperature

    
    def forward(self, features):
        """
        input:
            - features: hidden feature representation of shape [b, 2, dim]
        output:
            - loss: loss computed according to SimCLR 
        """

        b, n, dim = features.size()
        assert(n == 2)
        mask = torch.eye(b, dtype=torch.float32).cuda()

        contrast_features = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor = features[:, 0]

#        print(contrast_features.T.size())
      #  print(anchor)
       # print("-------------")

 #       anchor_tile = anchor.repeat(1024,1)
  #      print(anchor.size())
   #     print(anchor_tile.size())
    #    print(contrast_features.T.size())

        diff = anchor - contrast_features.T[:,0]
        diff = diff + anchor - contrast_features.T[:,1]
        diff_entropy = entropy(diff, input_as_probabilities = False)

        # Dot product
        dot_product = torch.matmul(anchor, contrast_features.T) / self.temperature
      #  print(dot_product)


        # Log-sum trick for numerical stability
        logits_max, _ = torch.max(dot_product, dim=1, keepdim=True)
        logits = dot_product - logits_max.detach()
      #  print(logits)
        logits_entropy = entropy(logits, input_as_probabilities=True)
        reg_lambda = 2

        mask = mask.repeat(1, 2)
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(b).view(-1, 1).cuda(), 0)
        mask = mask * logits_mask

        # Log-softmax
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # Mean log-likelihood for positive
        loss = - ((mask * log_prob).sum(1) / mask.sum(1)).mean() + reg_lambda * logits_entropy

        return loss
