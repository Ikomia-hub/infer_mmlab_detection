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

from ikomia import core, dataprocess
from ikomia.utils import pyqtutils, qtconversion
from infer_mmlab_detection.infer_mmlab_detection_process import InferMmlabDetectionParam

# PyQt GUI framework
from PyQt5.QtWidgets import *
import os
from torch.cuda import is_available as is_cuda_available
import yaml
from infer_mmlab_detection.utils import Autocomplete

# --------------------
# - Class which implements widget associated with the process
# - Inherits PyCore.CWorkflowTaskWidget from Ikomia API
# --------------------
class InferMmlabDetectionWidget(core.CWorkflowTaskWidget):

    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = InferMmlabDetectionParam()
        else:
            self.parameters = param

        # Create layout : QGridLayout by default
        self.gridLayout = QGridLayout()

        self.available_models = []
        configs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
        for model_dir in os.listdir(configs_dir):
            if os.path.isfile(os.path.join(configs_dir, model_dir, 'metafile.yml')):
                self.available_models.append(model_dir)
        self.combo_model = Autocomplete(self.available_models, parent=None, i=True, allow_duplicates=False)
        self.label_model = QLabel("Model name")
        self.gridLayout.addWidget(self.combo_model, 0, 1)
        self.gridLayout.addWidget(self.label_model, 0, 0)
        self.combo_config = pyqtutils.append_combo(self.gridLayout, "Config")

        self.combo_model.currentTextChanged.connect(self.on_model_changed)

        self.combo_model.setCurrentText(self.parameters.model_name)

        self.combo_config.setCurrentText(self.parameters.model_config)

        self.check_cuda = pyqtutils.append_check(self.gridLayout, "Use cuda",
                                                 self.parameters.cuda and is_cuda_available())

        self.check_cuda.setEnabled(is_cuda_available())

        self.double_spin_conf_thres = pyqtutils.append_double_spin(self.gridLayout, "Confidence threshold",
                                                                 self.parameters.conf_thres, min=0, max=1, step=0.01)

        self.check_custom_model = pyqtutils.append_check(self.gridLayout, "Use custom model",
                                                         self.parameters.use_custom_model)

        self.browse_custom_cfg = pyqtutils.append_browse_file(self.gridLayout, "Custom config (.py)",
                                                              self.parameters.config_file)
        self.browse_custom_weights = pyqtutils.append_browse_file(self.gridLayout, "Custom weights (.pth)",
                                                                  self.parameters.model_weight_file)
        enabled = self.check_custom_model.isChecked()
        self.combo_model.setEnabled(not enabled)
        self.combo_config.setEnabled(not enabled)
        self.browse_custom_cfg.setEnabled(enabled)
        self.browse_custom_weights.setEnabled(enabled)

        self.check_custom_model.stateChanged.connect(self.on_check_custom_changed)

        # PyQt -> Qt wrapping
        layout_ptr = qtconversion.PyQtToQt(self.gridLayout)

        # Set widget layout
        self.set_layout(layout_ptr)

    def on_check_custom_changed(self, b):
        enabled = self.check_custom_model.isChecked()
        self.combo_model.setEnabled(not enabled)
        self.combo_config.setEnabled(not enabled)
        self.browse_custom_cfg.setEnabled(enabled)
        self.browse_custom_weights.setEnabled(enabled)

    def on_model_changed(self, int):
        self.combo_config.clear()
        model = self.combo_model.currentText()
        yaml_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", model, "metafile.yml")
        if os.path.isfile(yaml_file):
            with open(yaml_file, "r") as f:
                models_list = yaml.load(f, Loader=yaml.FullLoader)['Models']
            available_cfg = [model_dict["Name"] for
                                       model_dict in models_list
                                       if "Weights" in model_dict]
            self.combo_config.addItems(available_cfg)
            self.combo_config.setCurrentText(available_cfg[0])

    def on_apply(self):
        # Apply button clicked slot

        # Get parameters from widget
        # Example : self.parameters.windowSize = self.spinWindowSize.value()
        self.parameters.cuda = self.check_cuda.isChecked()
        self.parameters.model_config = self.combo_config.currentText()
        self.parameters.model_name = self.combo_model.currentText()
        self.parameters.conf_thres = self.double_spin_conf_thres.value()
        self.parameters.use_custom_model = self.check_custom_model.isChecked()
        self.parameters.config_file = self.browse_custom_cfg.path
        self.parameters.model_weight_file = self.browse_custom_weights.path
        self.parameters.update = True
        # Send signal to launch the process
        self.emit_apply(self.parameters)


# --------------------
# - Factory class to build process widget object
# - Inherits PyDataProcess.CWidgetFactory from Ikomia API
# --------------------
class InferMmlabDetectionWidgetFactory(dataprocess.CWidgetFactory):

    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        # Set the name of the process -> it must be the same as the one declared in the process factory class
        self.name = "infer_mmlab_detection"

    def create(self, param):
        # Create widget object
        return InferMmlabDetectionWidget(param, None)
