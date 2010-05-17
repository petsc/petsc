# --------------------------------------------------------------------

__all__ = ['setup',
           'Extension',
           'config',
           'build',
           'build_ext',
           ]

# --------------------------------------------------------------------

import sys, os, platform

if (not hasattr(sys, 'version_info') or 
    sys.version_info < (2, 4, 0, 'final')):
    raise SystemExit("Python 2.4 or later is required")

# --------------------------------------------------------------------

from conf.core import PetscConfig as BasePetscConfig
from conf.core import setup, Extension
from conf.core import config, build, build_ext

# --------------------------------------------------------------------

class PetscConfig(BasePetscConfig):
    pass

config.Configure = PetscConfig

# --------------------------------------------------------------------
