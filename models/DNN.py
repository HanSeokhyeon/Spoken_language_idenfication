"""
Copyright 2019-present Han Seokhyeon.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch.nn as nn
import torch.nn.functional as F


class DNN(nn.Module):

    def __init__(self):
        super(DNN, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(40, 300),
            nn.BatchNorm1d(300),
            nn.Hardtanh(0, 20),
            nn.Linear(300, 2),
        )

    def forward(self, input_var):
        x = self.hidden(input_var)
        x = F.log_softmax(x)

        return x
