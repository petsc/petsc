import config.package
import os

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.gitcommit        = '65328ea'
    self.download         = ['git://https://github.com/MeshToolkit/MSTK','https://github.com/MeshToolkit/MSTK/archive'+self.gitcommit+'.tar.gz']
    self.downloaddirnames = ['mstk']
    self.includes         = ['MSTK.h']
    self.liblist          = [['libmstk.a']]
    self.functions        = []
    self.cxx              = 1
    self.requirescxx11    = 0
    self.downloadonWindows= 0
    self.hastests         = 1
    return

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.compilerFlags   = framework.require('config.compilerFlags', self)
    self.mpi             = framework.require('config.packages.MPI',self)
    self.metis           = framework.require('config.packages.metis',self)
    self.parmetis        = framework.require('config.packages.parmetis',self)
    self.ptscotch        = framework.require('config.packages.PTScotch',self)
    self.zoltan          = framework.require('config.packages.Zoltan',self)
    self.exodusii        = framework.require('config.packages.exodusii',self)
    self.trilinos        = framework.require('config.packages.Trilinos',self)
    self.deps            = [self.mpi]
    return

  def formCMakeConfigureArgs(self):
    if self.checkSharedLibrariesEnabled():
      raise RuntimeError('mstk cannot be built with shared libraries, run with --download-mstk-shared=0')

    args = config.package.CMakePackage.formCMakeConfigureArgs(self)
    args.append('-DUSE_XSDK_DEFAULTS=YES')
    if self.compilerFlags.debugging:
      args.append('-DXSDK_ENABLE_DEBUG=YES')
    else:
      args.append('-DXSDK_ENABLE_DEBUG=NO')

    args.append('-DENABLE_PARALLEL=yes')
    if not self.metis.found and not self.zoltan.found and not self.trilinos:
      raise  RuntimeError('MSTK requires either Metis (--download-metis) or Zoltan (--download-zoltan (or --download-trilinos) --download-parmetis --download-metis)!')
    if self.metis.found:
      args.append('-DENABLE_METIS:BOOL=ON')
      args.append('-DMETIS_DIR:FILEPATH='+self.metis.directory)
    if self.zoltan.found or self.trilinos.found:
      args.append('-DENABLE_ZOLTAN:BOOL=ON')
      if self.zoltan.found:
        args.append('-DZOLTAN_DIR:FILEPATH='+self.zoltan.directory)
      else:
       args.append('-DZOLTAN_DIR:FILEPATH='+self.trilinos.directory)
      if self.parmetis.found:
        args.append('-DZOLTAN_NEEDS_ParMETIS=yes')
        args.append('-DParMETIS_DIR:FILEPATH='+self.parmetis.directory)
      if self.ptscotch.found:
        args.append('-DZOLTAN_NEEDS_PTSCOTCH=yes')
        args.append('-DPTSCOTCH_DIR:FILEPATH='+self.ptscotch.directory)

    if self.exodusii.found or self.trilinos.found:
      args.append('-DENABLE_EXODUSII:BOOL=ON')
      if self.exodusii.found:
        args.append('-DEXODUSII_DIR:FILEPATH='+self.exodusii.directory)
      else:
        args.append('-DEXODUSII_DIR:FILEPATH='+self.trilinos.directory)

    #  Need to pass -DMETIS_5 to C and C++ compiler flags otherwise assumes older Metis
    args = self.rmArgsStartsWith(args,['-DCMAKE_CXX_FLAGS:STRING','-DCMAKE_C_FLAGS:STRING'])
    args.append('-DCMAKE_C_FLAGS:STRING="'+self.updatePackageCFlags(self.getCompilerFlags())+' -DMETIS_5"')
    if hasattr(self.compilers, 'CXX'):
      self.pushLanguage('Cxx')
      args.append('-DCMAKE_CXX_FLAGS:STRING="'+self.updatePackageCxxFlags(self.getCompilerFlags())+' -DMETIS_5"')
    self.popLanguage()

    # mstk does not use the standard -DCMAKE_INSTALL_PREFIX
    args.append('-DINSTALL_DIR='+self.installDir)
    return args

