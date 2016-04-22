import config.package
import os

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.gitcommit        = 'v5.2.0'
    self.download         = ['http://crd-legacy.lbl.gov/~xiaoye/SuperLU/superlu_5.2.0.tar.gz','git://https://github.com/xiaoyeli/superlu']
    self.functions        = ['set_default_options']
    self.includes         = ['slu_ddefs.h']
    self.liblist          = [['libsuperlu.a']]
    # SuperLU has NO support for 64 bit integers, use SuperLU_Dist if you need that
    self.requires32bitint = 1;  # 1 means that the package will not work with 64 bit integers
    self.excludedDirs     = ['SuperLU_DIST','SuperLU_MT']
    # SuperLU does not work with --download-fblaslapack with Compaqf90 compiler on windows.
    # However it should work with intel ifort.
    self.downloadonWindows= 1
    self.hastests         = 1
    self.hastestsdatafiles= 1
    return

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.blasLapack = self.framework.require('config.packages.BlasLapack',self)
    self.deps       = [self.blasLapack]
    return

  def formCMakeConfigureArgs(self):
    args = config.package.CMakePackage.formCMakeConfigureArgs(self)
    args.append('-DUSE_XSDK_DEFAULTS=YES')

    args.append('-DTPL_BLAS_LIBRARIES="'+self.libraries.toString(self.blasLapack.dlib)+'"')

    #  Tests are broken on Apple since they depend on a shared library that is not resolved against BLAS
    args.append('-Denable_tests=0')
    #  CMake in SuperLU should set this; but like many other packages it does not
    args.append('-DCMAKE_INSTALL_NAME_DIR:STRING="'+os.path.join(self.installDir,self.libdir)+'"')
    return args

