import torch
import numpy as np
import torch.nn.functional as F

def feat_extractor(model, data_loader, logger=None):
    model.eval()
    feats = list()
    if logger is not None:
        logger.info("Begin extract")
    for i, batch in enumerate(data_loader):
        imgs = batch[0].cuda()

        with torch.no_grad():
            out, _ = model(imgs)
            out = F.normalize(out, p=2, dim=1)
            out = out.data.cpu().numpy()
            feats.append(out)

        if logger is not None and (i + 1) % 100 == 0:
            logger.debug(f"Extract Features: [{i + 1}/{len(data_loader)}]")
        del out
    feats = np.vstack(feats)
    return feats


def code_generator(model, data_loader, logger=None):
    model.eval()
    codes = list()
    labels = list()
    if logger is not None:
        logger.info("Begin generate")
    for i, batch in enumerate(data_loader):
        imgs = batch[0].cuda()
        labels.append(batch[1])
        with torch.no_grad():
            out = model(imgs)
            out = out.data.cpu()
            codes.append(out)

        if logger is not None and (i + 1) % 100 == 0:
            logger.debug(f"Extract Features: [{i + 1}/{len(data_loader)}]")
        del out
    codes = torch.cat(codes).sign().numpy()
    labels = torch.cat(labels)
    return codes, labels
