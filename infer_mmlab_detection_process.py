# Copyright (C) 2021 Ikomia SAS
# Contact: https://www.ikomia.com
#
# This file is part of the IkomiaStudio software.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from ikomia import utils, core, dataprocess
import copy
# Your imports below
from mmdet.apis import init_detector, inference_detector
import os
import numpy as np
from mmdet.core import INSTANCE_OFFSET


# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferMmlabDetectionParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        self.cuda = True
        self.model_config = "yolox_tiny_8x8_300e_coco"
        self.model_name = "yolox"
        self.model_url = "https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_x_8x8_300e_coco" \
                         "/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth "
        self.conf_thr = 0.5
        self.use_custom_model = False
        self.custom_cfg = ""
        self.custom_weights = ""
        self.update = False

    def setParamMap(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.cuda = utils.strtobool(param_map["cuda"])
        self.model_config = param_map["model_config"]
        self.model_name = param_map["model_name"]
        self.model_url = param_map["model_url"]
        self.conf_thr = float(param_map["conf_thr"])
        self.use_custom_model = utils.strtobool(param_map["use_custom_model"])
        self.custom_cfg = param_map["custom_cfg"]
        self.custom_weights = param_map["custom_weights"]
        self.update = True

    def getParamMap(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = core.ParamMap()
        param_map["cuda"] = str(self.cuda)
        param_map["model_config"] = self.model_config
        param_map["model_name"] = self.model_name
        param_map["model_url"] = self.model_url
        param_map["conf_thr"] = str(self.conf_thr)
        param_map["use_custom_model"] = str(self.use_custom_model)
        param_map["custom_cfg"] = self.custom_cfg
        param_map["custom_weights"] = self.custom_weights
        return param_map


# --------------------
# - Class which implements the process
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferMmlabDetection(dataprocess.C2dImageTask):

    def __init__(self, name, param):
        dataprocess.C2dImageTask.__init__(self, name)
        self.model = None
        # Add object detection output
        self.addOutput(dataprocess.CObjectDetectionIO())
        # Create parameters class
        if param is None:
            self.setParam(InferMmlabDetectionParam())
        else:
            self.setParam(copy.deepcopy(param))

    def getProgressSteps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 1

    def run(self):
        # Core function of your process
        # Call beginTaskRun for initialization
        self.beginTaskRun()

        # Get parameters :
        param = self.getParam()

        if self.model is None or param.update:
            if param.use_custom_model:
                cfg_file = param.custom_cfg
                ckpt_file = param.custom_weights
            else:
                cfg_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", param.model_name,
                                        param.model_config + '.py')
                ckpt_file = param.model_url
            self.model = init_detector(cfg_file, ckpt_file, device='cuda:0' if param.cuda else 'cpu')
            self.classes = self.model.CLASSES
            self.colors = np.array(np.random.randint(0, 255, (len(self.classes), 3)))
            self.colors = [[int(c[0]), int(c[1]), int(c[2])] for c in self.colors]
            print("Inference will run on " + ('cuda' if param.cuda else 'cpu'))
            param.update = False
        # Examples :
        # Get input :
        img_input = self.getInput(0)

        # Forward input image
        self.forwardInputImage(0, 0)

        # Get image from input/output (numpy array):
        srcImage = img_input.getImage()

        if self.model:
            self.infer(srcImage, param.conf_thr)

        # Step progress bar:
        self.emitStepProgress()

        # Call endTaskRun to finalize process
        self.endTaskRun()

    def infer(self, img, conf_thr):
        h, w = np.shape(img)[:2]
        out = inference_detector(self.model, img)

        # Transform model output in an Ikomia format to be displayed
        index = 0
        if isinstance(out, list):
            self.setOutput(dataprocess.CObjectDetectionIO(), 1)
            # Get output :
            obj_detect_out = self.getOutput(1)
            obj_detect_out.init("Mmlab_detection", 0)
            for cls, bboxes in enumerate(out):
                for bbox in bboxes:
                    conf = float(bbox[-1])
                    if conf < conf_thr:
                        continue
                    x_rect = float(self.clamp(bbox[0], 0, w))
                    y_rect = float(self.clamp(bbox[1], 0, h))
                    w_rect = float(self.clamp(bbox[2] - x_rect, 0, w))
                    h_rect = float(self.clamp(bbox[3] - y_rect, 0, h))
                    obj_detect_out.addObject(index, self.classes[cls], conf,
                                             x_rect, y_rect, w_rect, h_rect, self.colors[cls])
                    index += 1
        elif isinstance(out, tuple):
            self.setOutput(dataprocess.CInstanceSegIO(), 1)
            # Get output :
            instance_seg_out = self.getOutput(1)
            instance_seg_out.init("Mmlab_detection", 0, w, h)
            for cls, (bboxes, masks) in enumerate(zip(*out)):
                for bbox, mask in zip(bboxes, masks):
                    conf = float(bbox[-1])
                    if conf < conf_thr:
                        continue
                    x_rect = float(self.clamp(bbox[0], 0, w))
                    y_rect = float(self.clamp(bbox[1], 0, h))
                    w_rect = float(self.clamp(bbox[2] - x_rect, 0, w))
                    h_rect = float(self.clamp(bbox[3] - y_rect, 0, h))
                    instance_seg_out.addInstance(index, 1, cls, self.classes[cls], conf, x_rect, y_rect, w_rect, h_rect,
                                                 mask.astype(dtype='uint8'), self.colors[cls])
                    index += 1
        elif isinstance(out, dict):
            self.setOutput(dataprocess.CInstanceSegIO(), 1)
            # Get output :
            pan_seg_out = self.getOutput(1)
            pan_seg_out.init("Mmlab_detection", 0, w, h)
            pan_results = out['pan_results']
            ids = np.unique(pan_results)[::-1]
            legal_indices = ids != self.model.num_classes  # for VOID label
            ids = ids[legal_indices]
            labels = np.array([id % INSTANCE_OFFSET for id in ids], dtype=np.int64)
            segms = (pan_results[None] == ids[:, None, None])
            for label, seg in zip(labels, segms):
                y, x = np.median(seg.nonzero(), axis=1)
                pan_seg_out.addInstance(index, 0, int(label), self.classes[label], float(1), float(x), float(y), 0, 0,
                                        seg.astype(dtype='uint8'), self.colors[label])
                index += 1

    def clamp(self, x, mini, maxi):
        return mini if x < mini else maxi - 1 if x > maxi - 1 else x


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferMmlabDetectionFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "infer_mmlab_detection"
        self.info.shortDescription = "Inference for MMDET from MMLAB detection models"
        self.info.description = "If custom training is disabled, models will come from MMLAB's model zoo." \
                                "If not, you can also choose to load a model you trained yourself with our plugin " \
                                "train_mmlab_detection. In this case make sure you give to the plugin" \
                                "a config file (.py) and a model file (.pth). Both of these files are produced " \
                                "by the train plugin."
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Detection"
        self.info.version = "1.1.0"
        self.info.iconPath = "icons/mmlab.png"
        self.info.authors = """Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and
             Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and
             Liu, Ziwei and Xu, Jiarui and Zhang, Zheng and Cheng, Dazhi and
             Zhu, Chenchen and Cheng, Tianheng and Zhao, Qijie and Li, Buyu and
             Lu, Xin and Zhu, Rui and Wu, Yue and Dai, Jifeng and Wang, Jingdong
             and Shi, Jianping and Ouyang, Wanli and Loy, Chen Change and Lin, Dahua"""
        self.info.article = "MMDetection: Open MMLab Detection Toolbox and Benchmark"
        self.info.journal = "arXiv preprint arXiv:1906.07155"
        self.info.year = 2019
        self.info.license = "Apache-2.0 license"
        # URL of documentation
        self.info.documentationLink = "https://mmdetection.readthedocs.io/en/latest/"
        # Code source repository
        self.info.repository = "https://github.com/open-mmlab/mmdetection"
        # Keywords used for search
        self.info.keywords = "mmdet, mmlab, detection, yolo, yolor, yolox, mask, rcnn"

    def create(self, param=None):
        # Create process object
        return InferMmlabDetection(self.info.name, param)
