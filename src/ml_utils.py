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

def clear_old_ckpt( checkpoint_dir, keep=5 ):
    fnames = list()
    for fname in os.listdir(checkpoint_dir):
        mtc = re.match( r'.*model\.(\d+)', fname )
        if not mtc:
            continue
        fnames.append(fname)
    if len(fnames>keep):
        [ os.remove(os.path.join(checkpoint_dir,fname))
          for fname in sorted(fnames)[:keep] ]
    return
