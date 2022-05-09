
from typing import Any, Dict, Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch
from torch import nn

from tianshou.utils.net.common import MLP

SIGMA_MIN = -20
SIGMA_MAX = 2

class Critic(nn.Module):
    """Simple critic network. Will create an actor operated in continuous \
    action space with structure of preprocess_net ---> 1(q value).

    :param preprocess_net: a self-defined preprocess_net which output a
        flattened hidden state.
    :param hidden_sizes: a sequence of int for constructing the MLP after
        preprocess_net. Default to empty sequence (where the MLP now contains
        only a single linear layer).
    :param int preprocess_net_output_dim: the output dimension of
        preprocess_net.
    :param linear_layer: use this module as linear layer. Default to nn.Linear.
    :param bool flatten_input: whether to flatten input data for the last layer.
        Default to True.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.

    .. seealso::

        Please refer to :class:`~tianshou.utils.net.common.Net` as an instance
        of how preprocess_net is suggested to be defined.
    """

    def __init__(
        self,
        preprocess_net: nn.Module,
        hidden_sizes: Sequence[int] = (),
        device: Union[str, int, torch.device] = "cpu",
        preprocess_net_output_dim: Optional[int] = None,
        linear_layer: Type[nn.Linear] = nn.Linear,
        flatten_input: bool = True,
        flatten_obs: bool = True,
        convert_to_torch: bool = True,
    ) -> None:
        super().__init__()
        self.device = device
        self.preprocess = preprocess_net
        self.output_dim = 1
        input_dim = getattr(preprocess_net, "output_dim", preprocess_net_output_dim)
        self.last = MLP(
            input_dim,  # type: ignore
            1,
            hidden_sizes,
            device=self.device,
            linear_layer=linear_layer,
            flatten_input=flatten_input,
        )
        self.flatten_obs = flatten_obs
        self.convert_to_torch = convert_to_torch

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        act: Optional[Union[np.ndarray, torch.Tensor]] = None,
        info: Dict[str, Any] = {},
    ) -> torch.Tensor:
        """Mapping: (s, a) -> logits -> Q(s, a)."""
        if self.convert_to_torch:
            obs = torch.as_tensor(
                obs,
                device=self.device,  # type: ignore
                dtype=torch.float32,
            )
            if self.flatten_obs: obs = obs.flatten(1)
        if act is not None:
            if self.convert_to_torch:
                act = torch.as_tensor(
                    act,
                    device=self.device,  # type: ignore
                    dtype=torch.float32,
                )
                if self.flatten_obs: act = act.flatten(1)
                obs = torch.cat([obs, act], dim=1)
            else:
                obs = (obs, act)
        logits, hidden = self.preprocess(obs)
        # print(logits.abs().mean())
        logits = self.last(logits)
        # print(logits.abs().mean())
        return logits
