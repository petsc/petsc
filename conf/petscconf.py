# --------------------------------------------------------------------

__all__ = ['setup',
           'Extension',
           'config',
           'build',
           'build_src',
           'build_ext',
           'test',
           'sdist',
           ]

# --------------------------------------------------------------------

from conf.baseconf import PetscConfig as BasePetscConfig
from conf.baseconf import setup, Extension
from conf.baseconf import config, build, build_src, build_ext
from conf.baseconf import test, sdist

# --------------------------------------------------------------------

class PetscConfig(BasePetscConfig):
    pass

config.Configure = PetscConfig

# --------------------------------------------------------------------
