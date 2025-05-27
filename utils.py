#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 13:28:20 2022

@author: jannik
"""

import torch
from torch.autograd import Variable
import pathlib



def to_var(tensor):
    return Variable(tensor.cuda() if torch.cuda.is_available() else tensor)


def get_filenames_of_path(path: pathlib.Path, ext: str = '*'):
    """Returns a list of files in a directory/path. Uses pathlib."""
    filenames = [file for file in sorted(path.glob(ext)) if file.is_file()]
    return filenames

def get_sorted_filenames_of_path(path: pathlib.Path, ext: str = '*'):
    """Returns a list of files in a directory/path. Uses pathlib."""
    filenames = [file for file in sorted(path.glob(ext),
                                         key=lambda path: int(path.stem.rsplit("_")[0])) 
                 if file.is_file()]
    return filenames
