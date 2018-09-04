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
"""Data generators for SquaAD (https://rajpurkar.github.io/SQuAD-explorer/).
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
UTILS_DIR = os.path.join(CURRENT_DIR,"../../utils")
import sys
sys.path.append(UTILS_DIR)
from google_data_reader import GoogleDataReader

# Use register_model for a new T2TModel
# Use register_problem for a new Problem
# Use register_hparams for a new hyperparameter set


def _build_vocab(vocab_dir, vocab_name):
  """Build a vocabulary from examples.

  Args:
    generator: text generator for creating vocab.
    vocab_dir: directory where to save the vocabulary.
    vocab_name: vocab file name.

  Returns:
    text encoder.
  """
  data_reader = GoogleDataReader()
  vocab_path = os.path.join(vocab_dir, vocab_name)
  
  if not tf.gfile.Exists(vocab_path):

    file_names = ["sim-R/train.json","sim-R/dev.json","sim-R/test.json"]
    data_reader = GoogleDataReader(file_names)
    
    tokens = []
    tokens_dict = data_reader.get_token_dict()

    tokens += list(tokens_dict["regular_tokens"])
    for _,slot_tokens in tokens_dict["entity_tokens"].items():
      for st in list(slot_tokens):
        tokens += st.split(" ")
    
    encoder = text_encoder.TokenTextEncoder(None, vocab_list=tokens)
    encoder.store_to_file(vocab_path)
  else:
    encoder = text_encoder.TokenTextEncoder(vocab_path)
  return encoder


def _get_examples(dataset_split):
  data_reader = GoogleDataReader()
  
  if dataset_split==problem.DatasetSplit.TRAIN:
    data_reader.read_data_from("sim-R/train.json")
  elif dataset_split==problem.DatasetSplit.EVAL:
    data_reader.read_data_from("sim-R/dev.json")

  dialgoues = data_reader.get_dialogues_turn_tuples()
    

  return dialgoues

@registry.register_problem
class M2MProblem(text_problems.Text2TextProblem):
  """Base class for bAbi question answering problems."""

  def __init__(self, *args, **kwargs):

    super(M2MProblem, self).__init__(*args, **kwargs)


  def dataset_filename(self):
    return "M2M-R"

  @property
  def vocab_file(self):
    return "M2M-R" + '.vocab'

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

    # tmp_dir = _prepare_babi_data(tmp_dir, data_dir)
    _build_vocab(data_dir, self.vocab_filename)
    diaglogues = _get_examples(dataset_split)

    def _generate_samples():
      """sample generator.

      Yields:
        A dict.

      """
      for diag in diaglogues:
        turns_text = []
        for (u,s) in diag["turn_tuples"]:
            turns_text.append(u)
            turns_text.append(s)
        # turns_text = [t["text"] for turns in diag["turns"] for t in turns]
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


# @registry.register_hparams
# def my_very_own_hparams():
#   # Start with the base set
#   hp = common_hparams.basic_params1()
#   # Modify existing hparams
#   hp.num_hidden_layers = 2
#   # Add new hparams
#   hp.add_hparam("filter_size", 2048)
#   return hp