"""YOLOX-s Exp of the detector fine-tuned on UAVDT-M.

PGR uses it to rebuild the network around the fine-tuned weights; only the
YOLOX-s topology and the class count matter here. Fine-tuning recipe (paper
Sec. 4.2): COCO-pretrained YOLOX-s, full fine-tune with no frozen layer,
input of about 1280 px, mosaic and mixup disabled for the last 15 epochs.
Classes: car, truck, bus.
"""

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super().__init__()
        self.depth = 0.33
        self.width = 0.50
        self.num_classes = 3
        self.exp_name = "yoloxs_uavdt_m"
