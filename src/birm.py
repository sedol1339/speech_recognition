from typing import Any, Literal
from collections.abc import Callable

import torch
from torch.nn.functional import cross_entropy


class _EBD(torch.nn.Module):
    """

    Example:
        ebd = _EBD(num_envs=3, num_classes=50_000, device='cuda')
        ebd.re_init_with_noise(noise_sd=0.01)
        environments = torch.tensor([1., 2., 0.], device='cuda')
        ebd.forward().shape
        >>> torch.Size([3, 1, 50000])
    """

    def __init__(self, num_envs: int, num_classes: int, device: Any):
        super().__init__()
        self.embeddings = torch.nn.Embedding(num_envs, num_classes).to(device)
        self.embeddings.weight.data.fill_(1.0)

    def re_init_with_noise(self, std: float):
        """
        Re-initialize weights using normal distribution centered around 1.
        """
        self.embeddings.weight.data = torch.normal(
            mean=1.0,
            std=std,
            size=self.embeddings.weight.shape,
            device=self.embeddings.weight.get_device(),
        )

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        """
        Input is an array of indices of environments. For each environment index,
        will return the corresponding weights vector of len num_classes.

        Input shape (E,).
        Output shape (E, 1, num_classes).
        """
        return self.embeddings(e.long())[:, None, :]


class BayesianInvariantRiskMinimization_BayesFullbatch:
    """Calculates BIRM loss, implements "bayes_fullbatch" method.

    The batch size actually may have any size, although it is unclear how correct this is.

    Taken from authors' repo with minimal changes:
    https://github.com/linyongver/Bayesian-Invariant-Risk-Minmization
    """

    def __init__(
        self,
        num_envs: int,
        num_classes: int,
        device: Any,
    ):
        self.num_envs = num_envs
        self.num_classes = num_classes
        self.ebd = _EBD(num_envs=num_envs, num_classes=num_classes, device=device)

    def calc_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        env_indices: torch.Tensor,
        n_birm_samples: int = -1,
        loss_fn: Callable | Literal['cross_entropy'] = 'cross_entropy',
    ) -> torch.Tensor:
        losses = []
        for _ in range(n_birm_samples):
            self.ebd.re_init_with_noise(std=0.1)
            env_embeddings = self.ebd(env_indices)
            logits_perturbed = env_embeddings * logits
            if loss_fn == 'cross_entropy':
                loss = cross_entropy(
                    logits_perturbed.view(-1, self.num_classes),
                    labels.view(-1),
                )
            else:
                loss = torch.nanmean(loss_fn(logits_perturbed, labels))
            grad = torch.autograd.grad(
                loss * self.num_envs, self.ebd.parameters(), create_graph=True
            )[0]
            losses.append(torch.mean(grad**2))
        return torch.mean(torch.stack(losses))


class BayesianInvariantRiskMinimization_BayesByVariance:
    """Calculates BIRM loss, implements "bayes_variance" method.

    Here we try to fix the authors' possible error: in their code each iteration of the
    loop `for i in range(sampleN)` returns equivalent results, so `train_penalty` is
    an array of equal loss values.

    Taken from authors' repo with minimal changes:
    https://github.com/linyongver/Bayesian-Invariant-Risk-Minmization
    """

    def __init__(
        self,
        num_envs: int,
        num_classes: int,
        device: Any,
    ):
        self.num_envs = num_envs
        self.num_classes = num_classes

    def calc_loss(
        self,
        losses: torch.Tensor,
        env_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Losses are an array of losses, when the first index is sample index in batch.
        """
        losses_per_env_list = []
        for env_idx in range(self.num_envs):
            losses_per_env_list.append(torch.nanmean(losses[env_indices == env_idx]))
        losses_per_env = torch.stack(losses_per_env_list)
        return ((losses_per_env - losses_per_env.mean()) ** 2).mean()
