# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
import numpy as np
import os
from cntk import load_model
from TransferLearning import *
import pandas as pd

 # once:
    # load the trained transfer learning model
new_model_file=('C:\FISHmodel\Output\TransferLearning.model')

trained_model = load_model(new_model_file)

    # for every new image:
    # get predictions for a single image
img_file="alopias10.jpeg"
image_height = 224
image_width = 224
fish_names = pd.DataFrame(['aplopias','manta_mobula','sharks','sphyrna','tunnel'])
probs = eval_single_image(trained_model, img_file, image_width, image_height)
probs_max = np.argmax(probs)
result_string = fish_names.loc[probs_max]

print (result_string)

# define new_model_file)
