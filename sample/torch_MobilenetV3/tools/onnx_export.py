import argparse
import torch
import sys
sys.path.append("../code")
import geffnet
import onnx

def main(model_name,checkpoint,onnx_path):
    geffnet.config.set_exportable(True)
    model = geffnet.create_model(
        model_name,
        num_classes=1000,
        in_chans=3,
        pretrained=False,
        checkpoint_path=checkpoint)
    model.eval()
    x = torch.randn((1, 3, 224, 224), requires_grad=True)
    model(x)
    input_names = ["input0"]
    output_names = ["output0"]
    optional_args = dict(keep_initializers_as_inputs=True)
    try:
        torch_out = torch.onnx._export(
            model, x, onnx_path, export_params=True, verbose=False,
            input_names=input_names, output_names=output_names, **optional_args)
    except TypeError:
        torch_out = torch.onnx._export(
            model, x, onnx_path, export_params=True, verbose=False,
            input_names=input_names, output_names=output_names)
    onnx.save(onnx.shape_inference.infer_shapes(onnx.load(onnx_path)), onnx_path)


if __name__ == '__main__':
    # model_name="tf_mobilenetv3_small_075"
    # checkpoint="../models/pth/tf_mobilenetv3_small_075.pth"
    # onnx_path="../models/onnx/mobilenetv3_small_075.onnx"

    model_name="tf_mobilenetv3_small_100"
    checkpoint="tf_mobilenetv3_small_100-37f49e2b.pth"
    onnx_path="tf_mobilenetv3_small_100-37f49e2b.onnx"

    
    main(model_name,checkpoint,onnx_path)
