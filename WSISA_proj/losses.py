# losses.py

import torch

def cox_loss(pred_risk, surv_time, surv_status, eps=1e-6):
    pred_risk = pred_risk.view(-1)
    surv_time = surv_time.view(-1)
    surv_status = surv_status.view(-1)

    if torch.sum(surv_status) == 0:
        return torch.tensor(0.0, device=pred_risk.device, requires_grad=True)

    sort_idx = torch.argsort(-surv_time)
    pred_risk = pred_risk[sort_idx]
    surv_status = surv_status[sort_idx]

    hazard_ratio = torch.exp(pred_risk - pred_risk.max())
    cumsum_hr = torch.cumsum(hazard_ratio, dim=0)
    log_risk = torch.log(cumsum_hr + eps)
    loss = -torch.sum((pred_risk - log_risk) * surv_status)

    return loss / (torch.sum(surv_status) + eps)
