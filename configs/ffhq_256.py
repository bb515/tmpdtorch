# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Training NCSN++ on Church with VE SDE."""

from configs.default_lsun_configs import get_default_configs


def get_config():
  config = get_default_configs()

  # data
  data = config.data
  data.dataset = 'FFHQ'
  # NOTE: The ffhq-rn record stands for 2**n image_size
  # data.tfrecords_path = './assets/ffhq/ffhq-r08.tfrecords'
  data.tfrecords_path = '../tmpd/assets/ffhq/ffhq-r08.tfrecords'

  evaluate = config.eval
  evaluate.batch_size = 1

  return config
