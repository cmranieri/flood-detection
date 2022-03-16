import re
import os


def get_ckpt_epoch( checkpoint_dir ):
    epochs_list = [0]
    for fname in os.listdir(checkpoint_dir):
        mtc = re.match( r'.*model\.(\d+)', fname )
        if not mtc:
            continue
        epochs_list.append(int( mtc.groups()[0]) )
    return max(epochs_list)