import argparse
import os
import platform
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from utils.google_utils import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer)
from utils.plots import butter_lowpass_filtfilt, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

from models.models import *
from utils.datasets import *
from utils.general import *

def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)

def init_model(save_img = False):
    global opt
    print("In Init")
    print(opt)
    out, source, weights, view_img, save_txt, imgsz, cfg, names = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.cfg, opt.names
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    # model = Darknet(cfg, imgsz).cuda()
    model = Darknet(cfg, imgsz).to(device)
    model.load_state_dict(torch.load(weights, map_location=device)['model']) # (weights[0], map_location=device)['model'])
    #model = attempt_load(weights, map_location=device)  # load FP32 model
    #imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    model.to(device).eval()
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    modelc = None
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    

    # Get names and colors
    names = load_classes(names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    return save_img, model, device, half, modelc, colors, names


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, auto_size=32):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, auto_size), np.mod(dh, auto_size)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


# def detect(path, save_img, model, device, half, modelc, colors, names):
def detect(in_img, save_img, model, device, half, modelc, colors, names):

    out, source, weights, view_img, save_txt, imgsz, cfg = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.cfg
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    im0s = in_img # cv2.imread(path)
    img = letterbox(im0s, new_shape=opt.img_size, auto_size=64)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)

    # dataset = (path, img, img0, None)
    
    # Run inference
    t0 = time.time()
    # img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    # _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    
    # for path, img, im0s, vid_cap in dataset:
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = time_synchronized()
    pred = model(img, augment=opt.augment)[0]

    # Apply NMS
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
    t2 = time_synchronized()

    # Apply Classifier
    classify = False
    if classify:
        pred = apply_classifier(pred, modelc, img, im0s)

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        if webcam:  # batch_size >= 1
            p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
        else:
            p, s, im0 = path, '', im0s

        # save_path = str(Path(out) / Path(p).name)
        # txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
        s += '%gx%g ' % img.shape[2:]  # print string
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += '%g %ss, ' % (n, names[int(c)])  # add to string

            # Write results
            for *xyxy, conf, cls in det:
                if save_txt:  # Write to file
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    # with open(txt_path + '.txt', 'a') as f:
                    #     f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                if True: # save_img or view_img:  # Add bbox to image
                    label = '%s %.2f' % (names[int(cls)], conf)
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                    print()
                    print(int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))
                    print(names[int(cls)])

        # Print time (inference + NMS)
        print('%sDone. (%.3fs)' % (s, t2 - t1))

    return im0


# def detect_boxes(path, save_img, model, device, half, modelc, colors, names):
def detect_boxes(in_img, save_img, model, device, half, modelc, colors, names):

    out, source, weights, view_img, save_txt, imgsz, cfg = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.cfg
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    im0s = in_img # cv2.imread(path)
    img = letterbox(im0s, new_shape=opt.img_size, auto_size=64)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)

    # dataset = (path, img, img0, None)
    
    # Run inference
    t0 = time.time()
    # img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    # _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    
    # for path, img, im0s, vid_cap in dataset:
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = time_synchronized()
    pred = model(img, augment=opt.augment)[0]

    # Apply NMS
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
    t2 = time_synchronized()

    # Apply Classifier
    classify = False
    if classify:
        pred = apply_classifier(pred, modelc, img, im0s)

    pred_dict = {}

    ri = 0

    # print(pred)

    path = None

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        if webcam:  # batch_size >= 1
            p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
        else:
            p, s, im0 = path, '', im0s

        # save_path = str(Path(out) / Path(p).name)
        # txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
        s += '%gx%g ' % img.shape[2:]  # print string
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += '%g %ss, ' % (n, names[int(c)])  # add to string

            # Write results
            for *xyxy, conf, cls in det:
                label = '%s %.2f' % (names[int(cls)], conf)
                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
              
                pred_dict[ri] = {}
                pred_dict[ri]["class"] = names[int(cls)]
                pred_dict[ri]["coords"] = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                ri += 1

        # Print time (inference + NMS)
        print('%sDone. (%.3fs)' % (s, t2 - t1))


    return pred_dict

# def detect_box(path, save_img, model, device, half, modelc, colors, names):
def detect_box(in_img, save_img, model, device, half, modelc, colors, names):

    out, source, weights, view_img, save_txt, imgsz, cfg = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.cfg
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    im0s = in_img # cv2.imread(path)
    img = letterbox(im0s, new_shape=opt.img_size, auto_size=64)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)

    # dataset = (path, img, img0, None)
    
    # Run inference
    t0 = time.time()
    # img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    # _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    
    # for path, img, im0s, vid_cap in dataset:
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = time_synchronized()
    pred = model(img, augment=opt.augment)[0]

    # Apply NMS
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
    t2 = time_synchronized()

    # Apply Classifier
    classify = False
    if classify:
        pred = apply_classifier(pred, modelc, img, im0s)

    pred_dict = {}
    best_yet = 0

    ri = 0

    # print(pred)

    path = None

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        if webcam:  # batch_size >= 1
            p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
        else:
            p, s, im0 = path, '', im0s

        # save_path = str(Path(out) / Path(p).name)
        # txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
        s += '%gx%g ' % img.shape[2:]  # print string
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += '%g %ss, ' % (n, names[int(c)])  # add to string

            # Write results
            for *xyxy, conf, cls in det:
                label = '%s %.2f' % (names[int(cls)], conf)
                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                # print()
                # print(int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))
                # print(names[int(cls)])
                # print(ri)

                if (((int(xyxy[2])) - int(xyxy[0])) * (int(xyxy[3]) - int(xyxy[1]))) > best_yet:
                    pred_dict[ri] = {}
                    pred_dict[ri]["class"] = names[int(cls)]
                    pred_dict[ri]["coords"] = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                    best_yet = ((int(xyxy[2])) - int(xyxy[0])) * (int(xyxy[3]) - int(xyxy[1]))
                    # ri += 1

        # Print time (inference + NMS)
        print('%sDone. (%.3fs)' % (s, t2 - t1))


    return pred_dict


from flask import Flask, request, Response
import jsonpickle
import numpy as np

app = Flask(__name__)

@app.before_first_request
def _declareStuff():
    global parser, opt, save_img, model, device, half, modelc, colors, names
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolor_p6.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images/horses.jpg', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold') # 0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--cfg', type=str, default='cfg/yolor_p6.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/coco.names', help='*.cfg path')
    opt = parser.parse_args()
    print(opt)

    opt.view_img = False
    opt.save_img = False

    # opt.update = False

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['']:
                detect()
                strip_optimizer(opt.weights)
        else:
            save_img, model, device, half, modelc, colors, names = init_model()
            # print("Model initialised!")

@app.route('/')
def test():
    return "Hello world!"

# route http posts to this method
@app.route('/upload', methods=['POST'])
def upload_im():
    r = request
    # convert string of image data to uint8
    nparr = np.fromstring(r.data, np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # do some fancy processing here....

    # build a response dict to send back to client
    response = {'message': 'image received. size={}x{}'.format(img.shape[1], img.shape[0])
                }
    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")



@app.route('/det_obj/all_detection', methods=['POST'])
def det_obj_all():

    # def detect(in_img, save_img, model, device, half, modelc, colors, names):
    global save_img, model, device, half, modelc, colors, names

    r = request
    # convert string of image data to uint8
    nparr = np.fromstring(r.data, np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    response = detect_boxes(img, save_img, model, device, half, modelc, colors, names)

    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")


@app.route('/det_obj/largest_detection', methods=['POST'])
def det_obj_largest():

    # def detect(in_img, save_img, model, device, half, modelc, colors, names):
    global save_img, model, device, half, modelc, colors, names

    r = request
    # convert string of image data to uint8
    nparr = np.fromstring(r.data, np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    response = detect_box(img, save_img, model, device, half, modelc, colors, names)

    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")

if __name__ == '__main__':
    app.run()
