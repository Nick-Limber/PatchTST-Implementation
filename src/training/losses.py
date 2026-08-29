import torch

def mse_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    
    Parameters
    ----------
    prediction : torch.Tensor, shape [batch, horizon]
        Model output (normalized).
    target : torch.Tensor, shape [batch, horizon]
        Ground truth future values (normalized).

    Returns
    -------
    scalar tensor -- mean loss across all batch elements and horizon steps.
    """
    error = prediction - target
    squared = error ** 2   
    return squared.mean()                


def mae_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Parameters
    ----------
    prediction : torch.Tensor, shape [batch, horizon]
    target : torch.Tensor, shape [batch, horizon]

    Returns
    -------
    scalar tensor
    """
    error = prediction - target 
    absolute = error.abs()
    return absolute.mean()               


def quantile_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    quantile_levels: torch.Tensor,
) -> torch.Tensor:
    """
    Parameters
    ----------
    prediction : torch.Tensor, shape [batch, horizon, n_quantiles]
        One prediction per quantile level per horizon step.
    target : torch.Tensor, shape [batch, horizon]
        Ground truth (expanded internally to match prediction shape).
    quantile_levels : torch.Tensor, shape [n_quantiles]
        Quantile levels to target, e.g. torch.tensor([0.1, 0.5, 0.9]).
        Each value must be in (0, 1).

    Returns
    -------
    scalar tensor -- mean loss across batch, horizon, and quantile levels.
    
    Notes
    -----
    Switch from using torch.max to torch.clamp to enable gpu optimizations    
    """
    

    target_expanded = target.unsqueeze(-1).expand_as(prediction)

    error = target_expanded - prediction

    q = quantile_levels.view(1, 1, -1)

    under_prediction_penalty = q * torch.clamp(error, min=0)
    over_prediction_penalty  = (1 - q) * torch.clamp(-error, min=0)

    loss = under_prediction_penalty + over_prediction_penalty 
    return loss.mean()

POINT_LOSSES = {
    "mse": mse_loss,
    "mae": mae_loss,
}
