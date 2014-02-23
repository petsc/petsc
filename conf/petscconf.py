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

    def configure_extension(self, extension):
        BasePetscConfig.configure_extension(self, extension)
        version, release = petsc_version(self.PETSC_DIR)
        if release and version < (3, 5, 0):
            self.configure_extension_tao(extension)

    def configure_extension_tao(self, extension):
        import os
        TAO_DIR = os.environ.get('TAO_DIR')
        PETSC_ARCH = self.PETSC_ARCH
        if not TAO_DIR: return
        if not os.path.exists(TAO_DIR): return
        # define macros
        macros = [('TAO_DIR', TAO_DIR)]
        extension.define_macros.extend(macros)
        # includes and libraries
        TAO_INCLUDE = os.path.join(TAO_DIR, 'include')
        if os.path.exists(os.path.join(TAO_DIR, PETSC_ARCH)):
            TAO_LIB_DIR = os.path.join(TAO_DIR, PETSC_ARCH, 'lib')
        else:
            TAO_LIB_DIR = os.path.join(TAO_DIR, 'lib')
        tao_cfg = { }
        tao_cfg['include_dirs'] = [TAO_INCLUDE]
        tao_cfg['libraries'] = ["tao"]
        tao_cfg['library_dirs'] = [TAO_LIB_DIR]
        tao_cfg['runtime_library_dirs'] = tao_cfg['library_dirs']
        self._configure_ext(extension, tao_cfg)

def petsc_version(PETSC_DIR):
    import os, re
    version_re = {
        'major'  : re.compile(r"#define\s+PETSC_VERSION_MAJOR\s+(\d+)"),
        'minor'  : re.compile(r"#define\s+PETSC_VERSION_MINOR\s+(\d+)"),
        'micro'  : re.compile(r"#define\s+PETSC_VERSION_SUBMINOR\s+(\d+)"),
        'patch'  : re.compile(r"#define\s+PETSC_VERSION_PATCH\s+(\d+)"),
        'release': re.compile(r"#define\s+PETSC_VERSION_RELEASE\s+(\d+)"),
        }
    petscversion_h = os.path.join(PETSC_DIR,'include','petscversion.h')
    data = open(petscversion_h, 'rt').read()
    major = int(version_re['major'].search(data).groups()[0])
    minor = int(version_re['minor'].search(data).groups()[0])
    micro = int(version_re['micro'].search(data).groups()[0])
    release = int(version_re['release'].search(data).groups()[0])
    return  (major, minor, micro), bool(release)

config.Configure = PetscConfig

# --------------------------------------------------------------------
