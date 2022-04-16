import torch
import hydra
import logging
import numpy as np

from omegaconf.omegaconf import OmegaConf
from vision.datasets.voc_dataset import VOCDataset
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.ssd.data_preprocessing import PredictionTransform

logger = logging.getLogger(__name__)


@hydra.main(config_path="./configs", config_name="config")
def convert_model(cfg):
    root_dir = hydra.utils.get_original_cwd()
    model_path = f"{root_dir}/models/{cfg.model.name}"
    label_path = f"{root_dir}/models/{cfg.data.label_path}"
    dataset_path  = cfg.data.data_path
    class_names = [name.strip() for name in open(label_path).readlines()]
    logger.info(f"Loading pre-trained model from: {model_path}")

    net = create_mobilenetv2_ssd_lite(len(class_names), width_mult=cfg.model.mb2_width_mult, is_test=True)
    net.load(model_path)
    #predictor = create_mobilenetv2_ssd_lite_predictor(net, nms_method='hard')
    print("load success")
    dataset = VOCDataset(dataset_path, is_test=True)
    input_img = dataset.get_image(0)
    transform = PredictionTransform(int(cfg.data.size),
                                    [int(cfg.data.mean)] * 3,
                                    float(cfg.data.std))
    input_img = transform(input_img)

    input_sample = {
        "image": input_img.unsqueeze(0)
    }

    # Export the model
    logger.info(f"Converting the model into ONNX format")
    torch.onnx.export(
        net,  # model being run
        (input_sample['image'],), # model input (or a tuple for multiple inputs)
        f"{root_dir}/models/model.onnx",  # where to save the model (can be a file or file-like object)
        export_params=True,
        opset_version=10,
        input_names=["image"],  # the model's input names
        output_names=["output"],  # the model's output names
        dynamic_axes={
            "image": {0: "batch_size"},  # variable length axes
            "output": {0: "batch_size"},
        },
    )

    logger.info(
        f"Model converted successfully. ONNX format model is at: {root_dir}/models/model.onnx"
    )


if __name__ == "__main__":
    convert_model()
