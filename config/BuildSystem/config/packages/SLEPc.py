import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.gitcommit              = '487db32e39e0e9704092fe825dabbf4461e30421' # main jan-5-2026
    #self.gitcommit             = 'v'+self.version
    self.download               = ['git://https://gitlab.com/slepc/slepc.git','https://gitlab.com/slepc/slepc/-/archive/'+self.gitcommit+'/slepc-'+self.gitcommit+'.tar.gz']
    self.functions              = []
    self.includes               = []
    self.skippackagewithoptions = 1
    self.useddirectly           = 0
    self.linkedbypetsc          = 0
    self.builtafterpetsc        = 1
    return

  def setupHelp(self, help):
    import nargs
    config.package.Package.setupHelp(self,help)
    help.addArgument('SLEPC', '-download-slepc-configure-arguments=string', nargs.Arg(None, None, 'Additional configure arguments for the build of SLEPc'))
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.python          = framework.require('config.packages.Python',self)
    self.setCompilers    = framework.require('config.setCompilers',self)
    self.sharedLibraries = framework.require('PETSc.options.sharedLibraries', self)
    self.installdir      = framework.require('PETSc.options.installDir',self)
    self.parch           = framework.require('PETSc.options.arch',self)
    self.scalartypes     = framework.require('PETSc.options.scalarTypes',self)
    self.cuda            = framework.require('config.packages.CUDA',self)
    self.thrust          = framework.require('config.packages.Thrust',self)
    self.hypre           = framework.require('config.packages.hypre',self)
    self.SuiteSparse     = framework.require('config.packages.SuiteSparse',self)
    self.odeps           = [self.cuda,self.thrust,self.hypre,self.SuiteSparse]
    return

  def consistencyChecks(self):
    config.package.Package.consistencyChecks(self)
    if 'download-slepc' in self.argDB and self.argDB['download-slepc']:
      if 'download-slepc-configure-arguments' in self.argDB and '--with-slepc4py' in self.argDB['download-slepc-configure-arguments']:
        raise RuntimeError('You should set --with-slepc4py as a separate option of PETSc configure, not within --download-slepc-configure-arguments')
      if self.argDB['with-petsc4py'] and not self.argDB['with-slepc4py']:
        raise RuntimeError('You should also set --with-slepc4py when using both --with-petsc4py and --download-slepc')
    return

  def Install(self):
    import os
    # if installing prefix location then need to set new value for PETSC_DIR/PETSC_ARCH
    if self.argDB['prefix'] and not 'package-prefix-hash' in self.argDB:
       iarch = 'installed-'+self.parch.nativeArch.replace('linux-','linux2-')
       if self.scalartypes.scalartype != 'real':
         iarch += '-' + self.scalartypes.scalartype
       carg = 'SLEPC_DIR='+self.packageDir+' PETSC_DIR='+os.path.abspath(os.path.expanduser(self.argDB['prefix']))+' PETSC_ARCH=""'
       barg = 'SLEPC_DIR='+self.packageDir+' PETSC_DIR='+os.path.abspath(os.path.expanduser(self.argDB['prefix']))+' PETSC_ARCH='+iarch
       checkarg = 'SLEPC_DIR='+os.path.abspath(os.path.expanduser(self.argDB['prefix']))+' PETSC_DIR='+os.path.abspath(os.path.expanduser(self.argDB['prefix']))+' PETSC_ARCH=""'
       prefix = os.path.abspath(os.path.expanduser(self.argDB['prefix']))
    else:
       carg = 'SLEPC_DIR=' + self.packageDir
       barg = 'SLEPC_DIR=' + self.packageDir + ' PETSC_ARCH=${PETSC_ARCH}'
       checkarg = barg
       prefix = os.path.join(self.petscdir.dir,self.arch)
    self.directory = prefix

    if 'download-slepc-configure-arguments' in self.argDB and self.argDB['download-slepc-configure-arguments']:
      configargs = self.argDB['download-slepc-configure-arguments']
    else:
      configargs = ''

    self.include = [os.path.join(prefix,'include')]
    if self.argDB['with-single-library']:
      self.lib = [os.path.join(prefix,'lib','libslepc')]
    else:
      self.lib = [os.path.join(prefix,'lib','libslepclme'),'-lslepcmfn -lslepcnep -lslepcpep -lslepcsvd -lslepceps -lslepcsys']
    self.addPost(self.packageDir,[carg + ' ' + self.python.pyexe + ' ./configure --prefix=' + prefix + ' ' + configargs,
                                  barg + ' ${OMAKE} ' + barg,
                                  barg + ' ${OMAKE} ' + barg + ' install'])
    self.addMakeCheck(self.packageDir, '${OMAKE} ' + checkarg + ' check')
    self.addTest(self.packageDir, barg + ' ${OMAKE} ' + barg + ' test')
    self.logPrintBox('SLEPc examples are available at '+os.path.join(self.packageDir,'src','*','tutorials'))
    return self.installDir
