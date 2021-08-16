import os
import cv2
import numpy as np
import torch
import warnings
import argparse
import onnxruntime as ort
from utils.datasets import LoadStreams, LoadImages,LoadWebcam
from utils.draw import draw_boxes, draw_person
from utils.general import check_img_size
from utils.torch_utils import time_synchronized
from P_Net_backbone_detect import Person_detect
from deep_sort import build_tracker
from utils.parser import get_config
from utils.log import get_logger
from utils.torch_utils import select_device, load_classifier, time_synchronized
from sklearn.metrics.pairwise import cosine_similarity
import time







class yolo_reid():
    def __init__(self, cfg, args,whether_improve,light_improve, video_path,cam_path,model,tar_id=None):
        self.args = args
        self.video_path = video_path
        self.cam = cam_path

        pic_ways = []
        pic_ways.append(whether_improve)
        pic_ways.append(light_improve)
        self.pic_ways = pic_ways
        # self.camera_path = camer_path
        use_cuda = args.use_cuda and torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)
        # Person_detect行人检测类  _单纯的yolov5
        self.person_detect = Person_detect(self.args)
        # deepsort 类
        self.deepsort = build_tracker(cfg, args.sort, use_cuda=use_cuda)
        imgsz = check_img_size(args.img_size, s=32)  # self.model.stride.max())  # check img_size
        #
        if model == 'cam':
            self.dataset = LoadWebcam(self.pic_ways,pipe =self.cam, img_size=imgsz)
        else:
            self.dataset = LoadImages(self.video_path, img_size=imgsz)
        self.query_feat = np.load(args.query)
        self.names = np.load(args.names)
        # self.way = way
    def deep_sort(self):
        idx_frame = 0
        # results = []

        for video_path, img, ori_img, vid_cap in self.dataset:
            # print(ori_img)
            idx_frame += 1
            # print('aaaaaaaa', video_path, img.shape, im0s.shape, vid_cap)

            if idx_frame % 2 ==0:
                continue
            # yolo detection
            bbox_xywh, cls_conf, cls_ids, xy = self.person_detect.detect(img, ori_img)
            # print(xy)

            # do tracking
            out = self.deepsort.update(bbox_xywh, cls_conf, ori_img)
            if len(out) != 0:
                outputs, features = out
                print(len(outputs), len(bbox_xywh), features.shape)

                person_cossim = cosine_similarity(features, self.query_feat)

                max_idx = np.argmax(person_cossim, axis=1)
                maximum = np.max(person_cossim, axis=1)
                max_idx[maximum < 0.7] = -1
                reid_results = max_idx
                way = '1'
                draw_person(ori_img, xy, reid_results, self.names,way)  # draw_person name

            # reid

                if len(outputs) > 0:
                    bbox_tlwh = []
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    ori_im = draw_boxes(ori_img, bbox_xyxy, identities)

                    for bb_xyxy in bbox_xyxy:
                        bbox_tlwh.append(self.deepsort._xyxy_to_tlwh(bb_xyxy))

                    # results.append((idx_frame - 1, bbox_tlwh, identities))
                # print("yolo+deepsort:", time_synchronized() - t1)

            if self.args.display:
                cv2.imshow("test", ori_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        #
        # # 输出视频保存
        # save_video_path = r'C:\Users\Leon\Desktop\ln'
        # fps = 16
        # w = ori_img.shape[0]
        # h = ori_img.shape[1]
        # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        # writer = cv2.VideoWriter(save_video_path, fourcc, fps, (int(w), int(h)))
        #     # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     #     break



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", default='./test.mp4', type=str)
    parser.add_argument("--camera", action="store", dest="cam", type=int, default="0")
    parser.add_argument('--device', default='c'
                                            'uda:0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # yolov5
    parser.add_argument('--weights', nargs='+', type=str, default='./weights/yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=1080, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--classes', default=[0], type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')

    # deep_sort
    parser.add_argument("--sort", default=False, help='True: sort model or False: reid model')
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    parser.add_argument("--display", default=True, help='show resule')
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)

    # reid
    parser.add_argument("--query", type=str, default="./fast_reid/query/query_features.npy")
    parser.add_argument("--names", type=str, default="./fast_reid/query/names.npy")

    # pic_
    parser.add_argument("--whether_improve", type=str, default="True")
    # parser.add_argument("--light_average", type=str, default="False")
    parser.add_argument("--light_improve", type=str, default="False")
    parser.add_argument("--time_", type=str, default="False")
    parser.add_argument("--pic_size_improve", type=str, default="False")

    return parser.parse_args()


if __name__ == '__main__':
    # 一阶段
    args = parse_args()
    cfg = get_config()
    cfg.merge_from_file(args.config_deepsort)



    yolo_reid = yolo_reid(cfg, args, whether_improve=args.whether_improve,light_improve=args.light_improve,video_path=args.video_path,cam_path=args.cam,model='cam' )

    start = time.clock()
    with torch.no_grad():
        yolo_reid.deep_sort()
    end = time.clock()

    print('Running time: %s Seconds' % (end - start))
    # # 人物 创建与 修改  查询
    # os.system(r'python .\fast_reid\demo\person_bank.py path')
