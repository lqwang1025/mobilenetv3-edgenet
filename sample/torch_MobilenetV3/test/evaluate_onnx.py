from __future__ import division
import onnx
import onnxruntime as ort
from caffe2.python.onnx import backend
from torchvision import transforms
from PIL import Image
import os
import numpy 
import torch
import argparse

def predict():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx_path', metavar='MODEL', default="../models/onnx/mobilenetv3_small_075.onnx", help='Input pth model')
    parser.add_argument('--image_dir', metavar='DATA', default="/data/cli/data_zoo/quan-imagenet/imagenet", help='ImageNet dataset')
    parser.add_argument('--class_path', metavar='LABEL', default='/mldb/dataset/ImageNet/raw-data/imagenet_lsvrc_2015_synsets.txt', help='ImageNet class')
    parser.add_argument('--h', action="store_true",  help='python evaluate_pth.py model_path image_dir class_path')
    args = parser.parse_args()
    print("Evaluation...")
    
    onnx_path = args.onnx_path
    image_dir = args.image_dir
    class_path = args.class_path
    predictor = onnx.load(onnx_path)
    onnx.checker.check_model(predictor)
    onnx.helper.printable_graph(predictor.graph)
    predictor = backend.prepare(predictor, device="CPU")
    ort_session = ort.InferenceSession(onnx_path)
    input_name = ort_session.get_inputs()[0].name

    with open(class_path) as f:
         idx_to_class = [line.strip() for line in f.readlines()]
         
    tfms = transforms.Compose([transforms.Resize(size=256,interpolation=Image.BICUBIC),
                               transforms.CenterCrop(size=224),
                               transforms.ToTensor(),
                               transforms.Normalize([0.5, 0.5, 0.5],
                                  [0.5, 0.5, 0.5])])
    total_num = 0
    running_corrects = 0
    for class_id in os.listdir(image_dir):
        class_dir = os.path.join(image_dir, class_id)
        for image_id in os.listdir(class_dir):
            total_num += 1
            image_path = os.path.join(image_dir, class_id, image_id)
            try:
                image = tfms(Image.open(image_path)).unsqueeze(0).numpy()
                image = image.astype(numpy.float32)
                out = ort_session.run(None, {input_name: image})
                idx = numpy.argmax(out)
                running_corrects += numpy.sum(class_id == idx_to_class[idx])
                print("onnx accuracy: ", running_corrects / total_num)
            except:
                total_num -= 1
    print("total_accuracy: ", running_corrects / total_num)

if __name__ == '__main__':
    predict()
