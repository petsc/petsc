import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.gitcommit              = '9f90a1f759c3ab14cf6d9e558f4c8276950d565d' #master Aug 6 2025
    self.download               = ['git://https://gitlab.com/knepley/bamg.git','https://gitlab.com/knepley/bamg/archive/'+self.gitcommit+'.tar.gz']
    self.functions              = []
    self.includes               = []
    self.useddirectly           = 0
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

    # Must force --have-petsc4py into SLEPc configure arguments so it does not test PETSc before BAMG is built
    if self.argDB['download-'+self.downloadname.lower()]:
      if 'download-slepc-configure-arguments' in self.argDB:
        if not '--have-petsc4py' in self.argDB['download-slepc-configure-arguments']:
          self.argDB['download-slepc-configure-arguments'] = self.argDB['download-slepc-configure-arguments']+' --have-petsc4py'
      else:
        self.argDB['download-slepc-configure-arguments'] = '--have-petsc4py'
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
      if not hasattr(self.framework, 'packages'):
        self.framework.packages = []
      self.framework.packages.append(self)
      oldFlags = self.compilers.CPPFLAGS
      self.addMakeMacro('BAMG','yes')
      self.addPost(self.packageDir,[carg + self.python.pyexe + ' ./configure --prefix=' + prefix + ' --with-clean',
                                    'mkdir -p ' + os.path.join(iarch,'tests'),
                                    'touch ' + os.path.join(iarch,'tests','testfiles'),
                                    barg + '${OMAKE} ' + barg,
                                    barg + '${OMAKE} ' + barg + ' install'])
    return self.installDir
