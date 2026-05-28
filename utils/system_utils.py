#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from errno import EEXIST
from os import makedirs, path
import os

def mkdir_p(folder_path):
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        makedirs(folder_path)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise

def searchForMaxIteration(folder):
    """
    Search for the largest iteration number in a folder.。
    Supports two naming formats：iteration_XXX  coarse_iteration_XXX
    """
    saved_iters = []
    for fname in os.listdir(folder):
        if fname.startswith("iteration_") or fname.startswith("coarse_iteration_"):
            try:
                iter_num = int(fname.split("_")[-1])
                saved_iters.append(iter_num)
            except ValueError:
                continue
    if not saved_iters:
        raise ValueError(f"No valid iteration folders found in {folder}")
    return max(saved_iters)
