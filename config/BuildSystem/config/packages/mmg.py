import config.package
import os

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.gitcommit        = '9bd0023fe5f4a7de55d2959dd8dd4bf5d6929a36' # develop dec-22-2022
    self.download         = ['git://https://github.com/MmgTools/mmg.git','https://github.com/MmgTools/mmg/archive/'+self.gitcommit+'.tar.gz']
    self.versionname      = 'MMG_VERSION_RELEASE'
    self.includes         = ['mmg/libmmg.h']
    self.liblist          = [['libmmg.a','libmmg3d.a']]
    self.functions        = ['MMG5_paramUsage1']
    self.precisions       = ['double']
    return

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.externalpackagesdir = framework.require('PETSc.options.externalpackagesdir',self)
    self.compilerFlags = framework.require('config.compilerFlags',self)
    self.mathlib       = framework.require('config.packages.mathlib',self)
    self.ptscotch      = framework.require('config.packages.PTScotch',self)
    self.deps          = [self.mathlib,self.ptscotch]
    return

  def formCMakeConfigureArgs(self):
    args = config.package.CMakePackage.formCMakeConfigureArgs(self)
    args.append('-DUSE_ELAS=OFF')
    args.append('-DUSE_VTK=OFF')
    args.append('-DMMG_INSTALL_PRIVATE_HEADERS=ON')
    args.append('-DSCOTCH_DIR:STRING="'+self.ptscotch.directory+'"')
    if self.getDefaultIndexSize() == 64:
      int64_t = '''
#if !(defined(PETSC_HAVE_STDINT_H) && defined(PETSC_HAVE_INTTYPES_H) && defined(PETSC_HAVE_MPI_INT64_T))
#error PetscInt64 != int64_t
#endif
'''
      same_int64 = self.checkCompile(int64_t)
      if same_int64:
        args.append('-DMMG5_INT=int64_t')
      else:
        raise RuntimeError('Cannot use --download-mmg with a PetscInt64 type different than int64_t')
    return args
