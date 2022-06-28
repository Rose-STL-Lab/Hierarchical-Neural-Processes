from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import yaml
from lib import utils
from model.pytorch.supervisor import Supervisor
import random
import numpy as np
import os


def main(args):
    random_seed = [42,43,44] #42
    for i in range(3):
        with open(args.config_filename) as f:
            supervisor_config = yaml.load(f)
            supervisor = Supervisor(random_seed=random_seed[i], **supervisor_config)
            supervisor.train()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default='data/model/model_cov.yaml', type=str,
                        help='Configuration filename for restoring the model.')
    parser.add_argument('--use_cpu_only', default=False, type=bool, help='Set to true to only use cpu.')
    args = parser.parse_args()
    main(args)


