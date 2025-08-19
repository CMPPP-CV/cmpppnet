from mmdet.registry import MODELS
from mmseg.models import EncoderDecoder

@MODELS.register_module()
class SegEncoderDecoder(EncoderDecoder):

    def __init__(self, **kwargs):
        with MODELS.switch_scope_and_registry('mmseg') as registry:
            super(SegEncoderDecoder, self).__init__(**kwargs)

    def _forward(self, img, data_samples):
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=img.shape[2:],
                    img_shape=img.shape[2:],
                    pad_shape=img.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * img.shape[0]
        y = self.inference(img, batch_img_metas)

        return y