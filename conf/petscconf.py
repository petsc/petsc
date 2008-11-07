# --------------------------------------------------------------------

__all__ = ['setup',
           'Extension',
           'config',
           'build',
           'build_py',
           'build_ext',
           ]

# --------------------------------------------------------------------

import sys, os, platform

if not hasattr(sys, 'version_info') or \
       sys.version_info < (2, 4, 0,'final'):
    raise SystemExit("Python 2.4 or later is required "
                     "to build PETSc for Python package.")

# --------------------------------------------------------------------

from core import PetscConfig
from core import setup, Extension
from core import config
from core import build
from core import build_py
from core import build_ext

# --------------------------------------------------------------------
