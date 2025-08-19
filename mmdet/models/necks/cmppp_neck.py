from mmengine.model import BaseModule

from mmdet.registry import MODELS
from mmdet.utils import OptMultiConfig


@MODELS.register_module()
class CMPPPNeck(BaseModule):
    """The neck used in CMPPPNet - essentially empty and without trainable parameters."""
    def __init__(self, init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.fp16_enabled = False

    def forward(self, x):
        """Forward function."""
        return x