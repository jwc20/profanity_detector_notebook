#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import List, Optional, Set, Tuple
import regex

from dataset import DatasetLine
from base import Filter, FilteringCounts
from text_normalizer import replace_unicode_punct

# Class to handle toxic word lists
class ToxicityList:
    """A toxicity list is a list of toxic token sequences."""

    def __init__(self, word_list_paths: List[str]):
        # Regular expression to split on punctuation, symbols, and Han characters
        self._split = regex.compile(r"(\p{P}|\p{S}|\p{Han})")
        
        # Set to store tokenized toxic items
        self.toxicity: Set[str] = set()
        
        # Load and tokenize toxic items from provided word list paths
        for path in word_list_paths:
            with open(path, "rt") as fin:
                for line in fin:
                    line = line.strip()
                    tokenized = self._tokenize(line)
                    self.toxicity.add(tokenized)
                    self.toxicity.add(tokenized.lower())

    def _tokenize(self, s: str):
        """Tokenize a string for toxicity detection."""
        # Replace special unicode punctuations
        s = replace_unicode_punct(s.strip())
        # Split based on regular expression defined earlier
        tok = self._split.sub(r" \1 ", s)
        # Collapse multiple spaces into one
        tok = " ".join(tok.split())
        # Add spaces before and after the tokenized string
        # Helps in substring matching without false positives due to partial matches
        return " " + tok + " "

    def toxicity_count(self, s: str):
        """Return count of toxic items in a given string."""
        tokenized = self._tokenize(replace_unicode_punct(s))
        regular = sum(1 for t in self.toxicity if t in tokenized)
        lowercased = sum(1 for t in self.toxicity if t in tokenized.lower())
        return max(regular, lowercased)

# New class to hold the dataset line and its toxicity label
class LabeledDatasetLine(DatasetLine):
    def __init__(self, src: str, tgt: Optional[str], is_toxic: bool):
        super().__init__(src, tgt)
        self.is_toxic = is_toxic

# Class to filter dataset lines based on their toxicity
class ToxicityFilter(Filter):
    # Initialization of filter with paths, thresholds, and languages
    def __init__(
        self,
        twl_path_template: str,
        eng_porn_twl_path: Optional[str],
        max_toxicity: Optional[int],
        max_toxicity_difference: Optional[int],
        src_lang: str,
        tgt_lang: Optional[str],
    ):
        self.max_toxicity = max_toxicity
        self.max_toxicity_difference = max_toxicity_difference
        self.tgt_toxicity_list: Optional[ToxicityList] = None
        self.src_toxicity_list: Optional[ToxicityList] = None

        # Load source language's toxic list
        src_paths = []
        src_twl_path = twl_path_template.format(lang=src_lang)
        if os.path.isfile(src_twl_path):
            src_paths.append(src_twl_path)
        # Concatenate with the English list (if provided)
        if eng_porn_twl_path is not None:
            src_paths.append(eng_porn_twl_path)
        if src_paths:
            self.src_toxicity_list = ToxicityList(src_paths)

        # Load target language's toxic list
        tgt_paths = []
        if tgt_lang is not None:
            tgt_twl_path = twl_path_template.format(lang=tgt_lang)
            if os.path.isfile(tgt_twl_path):
                tgt_paths.append(tgt_twl_path)
        # Concatenate with the English list (if provided)
        if eng_porn_twl_path is not None:
            tgt_paths.append(eng_porn_twl_path)
        if tgt_paths:
            self.tgt_toxicity_list = ToxicityList(tgt_paths)

    def filter_line(self, line: DatasetLine, counts: FilteringCounts) -> LabeledDatasetLine:
        """Check if line's source and target are toxic and return labeled line."""
        is_src_toxic, is_tgt_toxic = False, False
        
        # Check toxicity in source text
        if self.src_toxicity_list is not None:
            src_toxicity = self.src_toxicity_list.toxicity_count(line.src)
            if self.max_toxicity is not None and src_toxicity > self.max_toxicity:
                counts.max_toxicity += 1
                is_src_toxic = True

        # Check toxicity in target text
        if line.tgt is not None and self.tgt_toxicity_list is not None:
            tgt_toxicity = self.tgt_toxicity_list.toxicity_count(line.tgt)
            if self.max_toxicity is not None and tgt_toxicity > self.max_toxicity:
                counts.max_toxicity += 1
                is_tgt

