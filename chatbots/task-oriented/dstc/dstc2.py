# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
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
"""Data generators for DSTCS2 challenge.
"""
import json
import os
import re

from tensor2tensor.data_generators import (generator_utils, problem,
                                           text_encoder, text_problems,
                                           tokenizer)
from tensor2tensor.layers import common_hparams
from tensor2tensor.utils import metrics, registry
import tensorflow as tf


CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
UTILS_DIR = os.path.join(CURRENT_DIR,"../../../utils")
import sys
sys.path.append(UTILS_DIR)
from google_data_reader import DSTCDataReader


def _build_vocab(vocab_dir, vocab_name):
  """Build a vocabulary from examples.

  Args:
    generator: text generator for creating vocab.
    vocab_dir: directory where to save the vocabulary.
    vocab_name: vocab file name.

  Returns:
    text encoder.
  """
  vocab_path = os.path.join(vocab_dir, vocab_name)
  
  if not tf.gfile.Exists(vocab_path):

    dr = DSTCDataReader()
    
    dev_tokens = dr.get_data_tokens("dev")
    test_tokens = dr.get_data_tokens("test")
    train_tokens = dr.get_data_tokens("train")

    tokens = list(set(dev_tokens+test_tokens+train_tokens))
    encoder = text_encoder.TokenTextEncoder(None, vocab_list=tokens)
    encoder.store_to_file(vocab_path)
  else:
    encoder = text_encoder.TokenTextEncoder(vocab_path)
  return encoder


def _get_examples(dataset_split):
  data_reader = DSTCDataReader()
  dialogues = []

  if dataset_split==problem.DatasetSplit.TRAIN:
    dialgoues = data_reader.get_dataset("train")
  elif dataset_split==problem.DatasetSplit.EVAL:
    dialgoues = data_reader.get_dataset("dev")
    

  return dialgoues

@registry.register_problem
class DstcProblem(text_problems.Text2TextProblem):
  """Base class for bAbi question answering problems."""

  def __init__(self, *args, **kwargs):

    super(DstcProblem, self).__init__(*args, **kwargs)


  def dataset_filename(self):
    return "DSTC"

  @property
  def vocab_file(self):
    return "DSTC" + '.vocab'

  @property
  def dataset_splits(self):
    return [{
        'split': problem.DatasetSplit.TRAIN,
        'shards': 1,
    }, {
        'split': problem.DatasetSplit.EVAL,
        'shards': 1,
    }]

  @property
  def is_generate_per_split(self):
    return True


  @property
  def vocab_type(self):
    return text_problems.VocabType.TOKEN

  def get_labels_encoder(self, data_dir):
    """Builds encoder for the given class labels.

    Args:
      data_dir: data directory

    Returns:
      An encoder for class labels.
    """
    label_filepath = os.path.join(data_dir, self.vocab_filename)
    return text_encoder.TokenTextEncoder(label_filepath)


  def generate_samples(self, data_dir, tmp_dir, dataset_split):

    _build_vocab(data_dir, self.vocab_filename)
    diaglogues = _get_examples(dataset_split)

    def _generate_samples():
      """sample generator.

      Yields:
        A dict.

      """
      for diag in diaglogues:
        turns_text = []
        for turn in diag["turns"]:
            s,u = turn["tokens"]
            if not s:
                s = ["sil"]
            if not u:
                u = ["sil"]
            turns_text.append(' '.join(s))
            turns_text.append(' '.join(u))
        # drop the welcome message
        turns_text = list(turns_text[1:-1])
        for i in range(0,len(turns_text),2):
            yield {
                'inputs': " ".join(turns_text[0:i+1]),
                'targets': turns_text[i+1]
            }

    return _generate_samples()

  def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
    """A generator that generates samples that are encoded.

    Args:
      data_dir: data directory
      tmp_dir: temp directory
      dataset_split: dataset split

    Yields:
      A dict.

    """
    generator = self.generate_samples(data_dir, tmp_dir, dataset_split)
    encoder = self.get_or_create_vocab(data_dir, tmp_dir)
    label_encoder = self.get_labels_encoder(data_dir)
    for sample in generator:
      inputs = encoder.encode(sample['inputs'])
      inputs.append(text_encoder.EOS_ID)
      # context = encoder.encode(sample['context'])
      # context.append(text_encoder.EOS_ID)
      targets = label_encoder.encode(sample['targets'])
      targets.append(text_encoder.EOS_ID)
      # sample['targets'] = targets
      yield {'inputs': inputs, 'targets': targets}


  def eval_metrics(self):
    """Specify the set of evaluation metrics for this problem.

    Returns:
      List of evaluation metrics of interest.
    """
    return [metrics.Metrics.APPROX_BLEU,metrics.Metrics.NEG_LOG_PERPLEXITY]



#BEST TRANSFOMER HPARAMS
from tensor2tensor.models.transformer import transformer_tiny

@registry.register_hparams
def m2m_m_transformer_hparams():
  hparams = transformer_tiny()
  hparams.num_hidden_layers = 4
  hparams.hidden_size = 128
  hparams.filter_size = 512
  hparams.num_heads = 4
  return hparams
