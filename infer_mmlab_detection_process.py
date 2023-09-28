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
from mmdet.utils import register_all_modules
from mmdet.apis import DetInferencer
import os
import numpy as np
from panopticapi.utils import rgb2id
from pycocotools.mask import decode
import yaml
import torch


# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferMmlabDetectionParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        self.cuda = True
        self.model_config = "yolox_s_8x8_300e_coco"
        self.model_name = "yolox"
        self.conf_thres = 0.5
        self.use_custom_model = False
        self.config_file = ""
        self.model_weight_file = ""
        self.update = False

    def set_values(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.cuda = utils.strtobool(param_map["cuda"])
        self.model_config = param_map["model_config"]
        self.model_name = param_map["model_name"]
        self.conf_thres = float(param_map["conf_thres"])
        self.use_custom_model = utils.strtobool(param_map["use_custom_model"])
        self.config_file = param_map["config_file"]
        self.model_weight_file = param_map["model_weight_file"]
        self.update = True

    def get_values(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = {}
        param_map["cuda"] = str(self.cuda)
        param_map["model_config"] = self.model_config
        param_map["model_name"] = self.model_name
        param_map["conf_thres"] = str(self.conf_thres)
        param_map["use_custom_model"] = str(self.use_custom_model)
        param_map["config_file"] = self.config_file
        param_map["model_weight_file"] = self.model_weight_file
        return param_map


# --------------------
# - Class which implements the process
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferMmlabDetection(dataprocess.C2dImageTask):

    def __init__(self, name, param):
        dataprocess.C2dImageTask.__init__(self, name)
        self.model = None
        register_all_modules()
        # Add object detection output
        self.add_output(dataprocess.CObjectDetectionIO())
        # Create parameters class
        if param is None:
            self.set_param_object(InferMmlabDetectionParam())
        else:
            self.set_param_object(copy.deepcopy(param))

    def get_progress_steps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 1

    def run(self):
        # Core function of your process
        # Call begin_task_run for initialization
        self.begin_task_run()

        # Get parameters :
        param = self.get_param_object()

        # Set cache dir in the algorithm folder to simplify deployment
        old_torch_hub = torch.hub.get_dir()
        torch.hub.set_dir(os.path.join(os.path.dirname(__file__), "models"))

        if self.model is None or param.update:
            cuda_available = torch.cuda.is_available()
            cfg_file, ckpt_file = self.get_absolute_paths(param)
            self.model = DetInferencer(cfg_file, ckpt_file, device='cuda:0' if param.cuda and cuda_available else 'cpu')
            self.classes = self.model.model.dataset_meta['classes']
            self.colors = np.array(np.random.randint(0, 255, (len(self.classes), 3)))
            self.colors = [[int(c[0]), int(c[1]), int(c[2])] for c in self.colors]
            print("Inference will run on " + ('cuda' if param.cuda and cuda_available else 'cpu'))
            param.update = False
        # Examples :
        # Get input :
        img_input = self.get_input(0)

        # Forward input image
        self.forward_input_image(0, 0)

        # Get image from input/output (numpy array):
        srcImage = img_input.get_image()

        if self.model:
            self.infer(srcImage, param.conf_thres)

        # Reset torch cache dir for next algorithms in the workflow
        torch.hub.set_dir(old_torch_hub)

        # Step progress bar:
        self.emit_step_progress()

        # Call end_task_run to finalize process
        self.end_task_run()

    @staticmethod
    def get_absolute_paths(param):
        if param.model_weight_file == "":
            yaml_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", param.model_name,
                                     "metafile.yml")

            if param.model_config.endswith('.py'):
                param.model_config = param.model_config[:-3]
            if os.path.isfile(yaml_file):
                with open(yaml_file, "r") as f:
                    models_list = yaml.load(f, Loader=yaml.FullLoader)['Models']

                available_cfg_ckpt = {model_dict["Name"]: {'cfg': model_dict["Config"],
                                                           'ckpt': model_dict["Weights"]}
                                      for model_dict in models_list}
                if param.model_config in available_cfg_ckpt:
                    cfg_file = available_cfg_ckpt[param.model_config]['cfg']
                    ckpt_file = available_cfg_ckpt[param.model_config]['ckpt']
                    cfg_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), cfg_file)
                    return cfg_file, ckpt_file
                else:
                    raise Exception(
                        f"{param.model_config} does not exist for {param.model_name}. Available configs for are {', '.join(list(available_cfg_ckpt.keys()))}")
            else:
                raise Exception(f"Model name {param.model_name} does not exist.")
        else:
            if os.path.isfile(param.model_config):
                cfg_file = param.model_config
            else:
                cfg_file = param.config_file
            ckpt_file = param.model_weight_file
            return cfg_file, ckpt_file

    @staticmethod
    def get_model_zoo():
        configs_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
        available_pairs = []
        for model_name in os.listdir(configs_folder):
            if model_name.startswith('_'):
                continue
            yaml_file = os.path.join(configs_folder, model_name, "metafile.yml")
            if os.path.isfile(yaml_file):
                with open(yaml_file, "r") as f:
                    models_list = yaml.load(f, Loader=yaml.FullLoader)
                    if 'Models' in models_list:
                        models_list = models_list['Models']
                    if not isinstance(models_list, list):
                        continue
                for model_dict in models_list:
                    available_pairs.append({"model_name": model_name, "model_config": os.path.basename(model_dict["Name"])})
        return available_pairs


    def infer(self, img, conf_thr):
        h, w = np.shape(img)[:2]
        out = self.model(img, draw_pred=False)['predictions'][0]
        # Transform model output in an Ikomia format to be displayed
        index = 0
        self.get_output(1).clear_data()
        if "panoptic_seg" in out:
            self.set_output(dataprocess.CSemanticSegmentationIO(), 1)
            # Get output :
            pan_seg_out = self.get_output(1)
            pan_results = rgb2id(out['panoptic_seg'])
            pan_seg_out.set_class_names(self.classes)
            pan_seg_out.set_mask(pan_results.astype(dtype='uint8'))
            self.set_output_color_map(0, 1, self.colors, True)

        elif "bboxes" in out and "masks" in out:
            self.set_output(dataprocess.CInstanceSegmentationIO(), 1)
            # Get output :
            obj_detect_out = self.get_output(1)
            obj_detect_out.init("Mmlab_detection", 0, w, h)
            for bbox, label, score, mask in zip(out["bboxes"], out["labels"], out["scores"], out["masks"]):
                conf = float(score)
                if conf < conf_thr:
                    continue
                x_rect = float(self.clamp(bbox[0], 0, w))
                y_rect = float(self.clamp(bbox[1], 0, h))
                w_rect = float(self.clamp(bbox[2] - x_rect, 0, w))
                h_rect = float(self.clamp(bbox[3] - y_rect, 0, h))
                cls = int(label)
                mask = decode(mask)
                obj_detect_out.add_object(index, 0,
                                          cls, self.classes[cls], conf,
                                         x_rect, y_rect, w_rect, h_rect, mask, self.colors[cls])
                index += 1
            self.set_output_color_map(0, 1, self.colors, True)
        elif "bboxes" in out:
            self.set_output(dataprocess.CObjectDetectionIO(), 1)
            # Get output :
            obj_detect_out = self.get_output(1)
            obj_detect_out.init("Mmlab_detection", 0)
            for bbox, label, score in zip(out["bboxes"], out["labels"], out["scores"]):
                conf = float(score)
                if conf < conf_thr:
                    continue
                x_rect = float(self.clamp(bbox[0], 0, w))
                y_rect = float(self.clamp(bbox[1], 0, h))
                w_rect = float(self.clamp(bbox[2] - x_rect, 0, w))
                h_rect = float(self.clamp(bbox[3] - y_rect, 0, h))
                cls = int(label)
                obj_detect_out.add_object(index, self.classes[cls], conf,
                                         x_rect, y_rect, w_rect, h_rect, self.colors[cls])
                index += 1
        else:
            print("This model task is not one of Object Detection, Instance Segmentation or Panoptic Segmentation. Try another one")

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
        self.info.short_description = "Inference for MMDET from MMLAB detection models"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Detection"
        self.info.version = "2.0.0"
        self.info.icon_path = "icons/mmlab.png"
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
        self.info.documentation_link = "https://mmdetection.readthedocs.io/en/latest/"
        # Code source repository
        self.info.repository = "https://github.com/Ikomia-hub/infer_mmlab_detection"
        self.info.original_repository = "https://github.com/open-mmlab/mmdetection"
        # Keywords used for search
        self.info.keywords = "mmdet, mmlab, detection, yolo, yolor, yolox, mask, rcnn"

    def create(self, param=None):
        # Create process object
        return InferMmlabDetection(self.info.name, param)
