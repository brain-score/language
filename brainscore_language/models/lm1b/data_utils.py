# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
"""A library for loading 1B word benchmark dataset."""

import numpy as np
import tensorflow.compat.v1 as tf


class Vocabulary(object):
  """Class that holds a vocabulary for the dataset."""

  def __init__(self, filename):
    """Initialize vocabulary.

    Args:
      filename: Vocabulary file name.
    """

    self._id_to_word = []
    self._word_to_id = {}
    self._unk = -1
    self._bos = -1
    self._eos = -1

    with tf.gfile.Open(filename) as f:
      idx = 0
      for line in f:
        word_name = line.strip()
        if word_name == '<S>':
          self._bos = idx
        elif word_name == '</S>':
          self._eos = idx
        elif word_name == '<UNK>':
          self._unk = idx
        if word_name == '!!!MAXTERMID':
          continue

        self._id_to_word.append(word_name)
        self._word_to_id[word_name] = idx
        idx += 1

  @property
  def bos(self):
    return self._bos

  @property
  def eos(self):
    return self._eos

  @property
  def unk(self):
    return self._unk

  @property
  def size(self):
    return len(self._id_to_word)

  def word_to_id(self, word):
    if word in self._word_to_id:
      return self._word_to_id[word]
    return self.unk

  def id_to_word(self, cur_id):
    if cur_id < self.size:
      return self._id_to_word[cur_id]
    return 'ERROR'

  def decode(self, cur_ids):
    """Convert a list of ids to a sentence, with space inserted."""
    return ' '.join([self.id_to_word(cur_id) for cur_id in cur_ids])

  def encode(self, sentence):
    """Convert a sentence to a list of ids, with special tokens added."""
    word_ids = [self.word_to_id(cur_word) for cur_word in sentence.split()]
    return np.array([self.bos] + word_ids + [self.eos], dtype=np.int32)


class CharsVocabulary(Vocabulary):
  """Vocabulary containing character-level information."""

  def __init__(self, filename, max_word_length):
    super(CharsVocabulary, self).__init__(filename)
    self._max_word_length = max_word_length
    chars_set = set()

    for word in self._id_to_word:
      chars_set |= set(word)

    free_ids = []
    for i in range(256):
      if chr(i) in chars_set:
        continue
      free_ids.append(chr(i))

    if len(free_ids) < 5:
      raise ValueError('Not enough free char ids: %d' % len(free_ids))

    self.bos_char = free_ids[0]  # <begin sentence>
    self.eos_char = free_ids[1]  # <end sentence>
    self.bow_char = free_ids[2]  # <begin word>
    self.eow_char = free_ids[3]  # <end word>
    self.pad_char = free_ids[4]  # <padding>

    chars_set |= {self.bos_char, self.eos_char, self.bow_char, self.eow_char,
                  self.pad_char}

    self._char_set = chars_set
    num_words = len(self._id_to_word)

    self._word_char_ids = np.zeros([num_words, max_word_length], dtype=np.int32)

    self.bos_chars = self._convert_word_to_char_ids(self.bos_char)
    self.eos_chars = self._convert_word_to_char_ids(self.eos_char)

    for i, word in enumerate(self._id_to_word):
      self._word_char_ids[i] = self._convert_word_to_char_ids(word)

  @property
  def word_char_ids(self):
    return self._word_char_ids

  @property
  def max_word_length(self):
    return self._max_word_length

  def _convert_word_to_char_ids(self, word):
    code = np.zeros([self.max_word_length], dtype=np.int32)
    code[:] = ord(self.pad_char)

    if len(word) > self.max_word_length - 2:
      word = word[:self.max_word_length-2]
    cur_word = self.bow_char + word + self.eow_char
    for j in range(len(cur_word)):
      code[j] = ord(cur_word[j])
    return code

  def word_to_char_ids(self, word):
    if word in self._word_to_id:
      return self._word_char_ids[self._word_to_id[word]]
    else:
      return self._convert_word_to_char_ids(word)

  def encode_chars(self, sentence):
    chars_ids = [self.word_to_char_ids(cur_word)
                 for cur_word in sentence.split()]
    return np.vstack([self.bos_chars] + chars_ids + [self.eos_chars])