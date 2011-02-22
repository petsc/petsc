# --------------------------------------------------------------------

__all__ = ['setup',
           'Extension',
           'config',
           'build',
           'build_src',
           'build_ext',
           'sdist',
           ]

# --------------------------------------------------------------------

import sys, os, platform

if (not hasattr(sys, 'version_info') or 
    sys.version_info < (2, 4, 0, 'final')):
    raise SystemExit("Python 2.4 or later is required")

# --------------------------------------------------------------------

from conf.baseconf import PetscConfig as BasePetscConfig
from conf.baseconf import setup, Extension
from conf.baseconf import config, build, build_src, build_ext, sdist

# --------------------------------------------------------------------

class PetscConfig(BasePetscConfig):
    pass

config.Configure = PetscConfig

# --------------------------------------------------------------------
