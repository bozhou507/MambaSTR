import os
this_dir = os.path.dirname(os.path.abspath((__file__)))
parent_dir = os.path.abspath(os.path.join(this_dir, '..'))
os.chdir(parent_dir)  # set current wordking directory as parent_dir
import sys
if parent_dir not in sys.path:
    sys.path.append(parent_dir)  # append parent_dir to package searching paths
