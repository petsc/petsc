#!/usr/bin/env python3
""" Reduce the top and bottom margins for <h2> to not waste so much vertical space in the manual pages """

import os
import re
import fileinput

def fix_pydata_margins(root):
    filename = os.path.join(root, '_static','styles','pydata-sphinx-theme.css')
    with open(filename, 'r') as source:
      str = source.read()
    str = str.replace('var(--pst-font-size-h2)}','var(--pst-font-size-h2);margin-bottom:4px;margin-top:-5px}')
    with open(filename,'w') as target:
      target.write(str)
