import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.gitcommit              = 'bf469647ffbcccabed2fe05bb54f407e00430b7d' # nov-17-2026, https://gitlab.com/knepley/bamg/-/merge_requests/5
    self.download               = ['git://https://gitlab.com/knepley/bamg.git','https://gitlab.com/knepley/bamg/archive/'+self.gitcommit+'.tar.gz']
    self.functions              = []
    self.includes               = []
    self.linkedbypetsc          = 0
    self.builtafterpetsc        = 1
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.python          = framework.require('config.packages.Python',self)
    self.setCompilers    = framework.require('config.setCompilers',self)
    self.sharedLibraries = framework.require('PETSc.options.sharedLibraries',self)
    self.mathlib         = framework.require('config.packages.mathlib',self)
    self.mpi             = framework.require('config.packages.MPI',self)
    self.blasLapack      = framework.require('config.packages.BlasLapack',self)
    self.slepc           = framework.require('config.packages.SLEPc',self)
    self.parch           = framework.require('PETSc.options.arch',self)
    self.scalartypes     = framework.require('PETSc.options.scalarTypes',self)
    self.deps            = [self.blasLapack,self.mathlib,self.mpi,self.slepc]
    return

  def Install(self):
    import os
    if self.checkSharedLibrariesEnabled():
      # if installing prefix location then need to set new value for PETSC_DIR/PETSC_ARCH
      if self.argDB['prefix'] and not 'package-prefix-hash' in self.argDB:
         iarch = 'installed-'+self.parch.nativeArch
         if self.scalartypes.scalartype != 'real':
           iarch += '-' + self.scalartypes.scalartype
         carg = 'BAMG_DIR='+self.packageDir+' PETSC_DIR='+os.path.abspath(os.path.expanduser(self.argDB['prefix']))+' PETSC_ARCH="" '
         barg = 'BAMG_DIR='+self.packageDir+' PETSC_DIR='+os.path.abspath(os.path.expanduser(self.argDB['prefix']))+' PETSC_ARCH='+iarch+' '+' SLEPC_DIR='+self.slepc.installDir+' '
         prefix = os.path.abspath(os.path.expanduser(self.argDB['prefix']))
      else:
         carg = ' BAMG_DIR='+self.packageDir+' '
         barg = ' BAMG_DIR='+self.packageDir+' SLEPC_DIR='+self.slepc.installDir+' '
         prefix = os.path.join(self.petscdir.dir,self.arch)
         iarch  = self.arch
      self.addPost(self.packageDir,[carg + self.python.pyexe + ' ./configure --prefix=' + prefix + ' --with-clean',
                                    'mkdir -p ' + os.path.join(iarch,'tests'),
                                    'touch ' + os.path.join(iarch,'tests','testfiles'),
                                    barg + '${OMAKE} ' + barg,
                                    barg + '${OMAKE} ' + barg + ' install'])
    return self.installDir
