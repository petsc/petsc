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

    def configure_extension(self, extension):
        BasePetscConfig.configure_extension(self, extension)
        # --- some hackery for petsc 2.3.2 ---
        bmake_dir = os.path.join(self['PETSC_DIR'], 'bmake')
        incprv_dir = os.path.join(self['PETSC_DIR'], 'include', 'private')
        if os.path.isdir(bmake_dir) and os.path.isdir(incprv_dir):
            matimpl_h = os.path.join(incprv_dir, 'matimpl.h')
            if not os.path.exists(matimpl_h):
                conf = {'include_dirs': ['src/include/compat/232']}
                self._configure_ext(extension, conf)

        # -------------------------------------

config.Configure = PetscConfig

# --------------------------------------------------------------------
