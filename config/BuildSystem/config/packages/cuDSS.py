import config.package
import os

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.version                = '1.0.0'
    self.minversion             = '0.8.0'
    self.versionname            = 'CUDSS_VERSION_MAJOR.CUDSS_VERSION_MINOR.CUDSS_VERSION_PATCH'
    self.versioninclude         = 'cudss.h'
    self.download               = 'NoDefault'
    self.download_linux_x86_64  = ['https://developer.download.nvidia.com/compute/cudss/redist/libcudss/linux-x86_64/libcudss-linux-x86_64-1.0.0.35_cuda12-archive.tar.xz']
    self.download_linux_aarch64 = ['https://developer.download.nvidia.com/compute/cudss/redist/libcudss/linux-sbsa/libcudss-linux-sbsa-1.0.0.35_cuda12-archive.tar.xz']
    self.functions              = ['cudssCreate', 'nvrtcVersion']
    self.includes               = ['cudss.h', 'nvrtc.h']
    self.liblist                = [['libcudss.so', 'libnvrtc.so'], ['libcudss_static.a', 'libnvrtc.so'], ['libcudss.a', 'libnvrtc.so']]
    self.downloaddirnames       = ['libcudss']
    self.precisions             = ['single', 'double']
    self.requires32bitint       = 0  # 64-bit PetscInt: the interface narrows its index arrays to cuDSS's 32-bit ints
    self.buildLanguages         = ['CUDA']
    self.hastests               = 1
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.cuda = framework.require('config.packages.CUDA', self)
    self.deps = [self.cuda]
    return

  def consistencyChecks(self):
    config.package.Package.consistencyChecks(self)
    if self.argDB['with-'+self.package]:
      if self.cuda.version_tuple[0] < 12:
        raise RuntimeError('Cannot use cuDSS: requires CUDA version >=12, currently using: '+str(self.cuda.foundversion))
      if self.cuda.version_tuple[0] >= 13 and isinstance(self.download, list):
        self.download = [u.replace('_cuda12-archive', '_cuda13-archive') for u in self.download]
    return

  def getSearchDirectories(self):
    '''Also search the CUDA toolkit directory for cuDSS'''
    dirs = config.package.Package.getSearchDirectories(self)
    if hasattr(self.cuda, 'cudaDir') and self.cuda.cudaDir not in dirs:
      dirs.append(self.cuda.cudaDir)
    return dirs

  def Install(self):
    import shutil

    conffile = 'petsc.cudss'
    with open(os.path.join(self.packageDir, conffile), 'w') as f:
      f.write(self.version + '\n')

    if not self.installNeeded(conffile):
      return self.installDir

    srcIncDir = os.path.join(self.packageDir, 'include')
    srcLibDir = os.path.join(self.packageDir, 'lib')
    dstIncDir = os.path.join(self.installDir, 'include')
    dstLibDir = os.path.join(self.installDir, 'lib')

    self.logPrintBox('Installing cuDSS; this may take several seconds')
    # Copy include files
    if os.path.isdir(srcIncDir):
      if not os.path.isdir(dstIncDir):
        os.makedirs(dstIncDir)
      for f in os.listdir(srcIncDir):
        src = os.path.join(srcIncDir, f)
        if os.path.isfile(src):
          shutil.copy2(src, dstIncDir)

    # Copy library files, preserving symlinks
    if os.path.isdir(srcLibDir):
      if not os.path.isdir(dstLibDir):
        os.makedirs(dstLibDir)
      for f in os.listdir(srcLibDir):
        src = os.path.join(srcLibDir, f)
        dst = os.path.join(dstLibDir, f)
        if os.path.islink(src):
          linkto = os.readlink(src)
          if os.path.islink(dst) or os.path.isfile(dst):
            os.remove(dst)
          os.symlink(linkto, dst)
        elif os.path.isfile(src):
          shutil.copy2(src, dstLibDir)

    self.postInstall(conffile)
    return self.installDir
