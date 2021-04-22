import torch
import torch.nn as nn


class DSHLoss(nn.Module):
    def __init__(self, args, bit):
        super(DSHLoss, self).__init__()
        self.m = 2 * bit
        self.U = torch.zeros(args.num_train, bit).float().to(args.device)
        self.Y = torch.zeros(args.num_train, args.train_classes).float().to(args.device)
        self.alpha = args.alpha

    def forward(self, u, y, ind, args):
        self.U[ind, :] = u.data
        self.Y[ind, :] = y.float()

        dist = (u.unsqueeze(1) - self.U.unsqueeze(0)).pow(2).sum(dim=2)
        y = (y @ self.Y.t() == 0).float()

        loss = (1 - y) / 2 * dist + y / 2 * (self.m - dist).clamp(min=0)
        loss1 = loss.mean()
        loss2 = args.alpha * (1 - u.sign()).abs().mean()

        return loss1 + loss2


class DCHLoss(nn.Module):
    def __init__(self, args, bit):
        super(DCHLoss, self).__init__()
        self.gamma = args.gamma
        self.lambd = args.lambd
        self.K = bit
        self.one = torch.ones((args.train_batch_size, bit)).to(args.device)

    def d(self, hi, hj):
        inner_product = hi @ hj.t()
        norm = hi.pow(2).sum(dim=1, keepdim=True).pow(0.5) @ hj.pow(2).sum(dim=1, keepdim=True).pow(0.5).t()
        cos = inner_product / norm.clamp(min=0.0001)
        # formula 6
        return (1 - cos.clamp(max=0.99)) * self.K / 2

    def forward(self, u, y):
        s = (y @ y.t() > 0).float()

        if (1 - s).sum() != 0 and s.sum() != 0:
            # formula 2
            positive_w = s * s.numel() / s.sum()
            negative_w = (1 - s) * s.numel() / (1 - s).sum()
            w = positive_w + negative_w
        else:
            # maybe |S1|==0 or |S2|==0
            w = 1

        d_hi_hj = self.d(u, u)
        # formula 8
        cauchy_loss = w * (s * torch.log(d_hi_hj / self.gamma) + torch.log(1 + self.gamma / d_hi_hj))
        # formula 9
        quantization_loss = torch.log(1 + self.d(u.abs(), self.one) / self.gamma)
        # formula 7
        #loss = cauchy_loss.mean() + self.lambd * quantization_loss.mean()

        return cauchy_loss.mean(), quantization_loss.mean()
