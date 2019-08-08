import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.gitcommit = 'v3.2p5'
    self.download  = ['git://https://bitbucket.org/petsc/pkg-parms.git','https://bitbucket.org/petsc/pkg-metis/get/'+self.gitcommit+'.tar.gz']
    self.downloaddirnames  = ['petsc-pkg-parms','pARMS']
    self.functions = ['parms_PCCreate']
    self.includes  = ['parms.h']
    self.liblist   = [['libparms.a']]
    #self.license   = 'http://www-users.cs.umn.edu/~saad/software/pARMS'
    self.precisions = ['double']
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.scalartypes =  framework.require('PETSc.options.scalarTypes',self)
    self.blasLapack  = framework.require('config.packages.BlasLapack',self)
    self.mpi         = framework.require('config.packages.MPI',self)
    self.deps        = [self.mpi,self.blasLapack]
    return

  def Install(self):
    import os

    # Configure and Build pARMS
    g = open(os.path.join(self.packageDir,'makefile.in'),'w')
    g.write('SHELL =	/bin/sh\n')
    g.write('.SUFFIXES:\n')
    g.write('.SUFFIXES: .c .o .f .F\n')

    # C compiler
    self.setCompilers.pushLanguage('C')
    g.write('CC         = '+self.setCompilers.getCompiler()+'\n')
    g.write('CFLAGS     = '+self.removeWarningFlags(self.setCompilers.getCompilerFlags())+' -DUSE_MPI -DREAL=double -DHAS_BLAS ')
    if self.scalartypes.scalartype == 'complex':
      g.write('-DDBL_CMPLX\n')
    else:
      g.write('-DDBL\n')
    self.setCompilers.popLanguage()

    # BLAS mangling
    if self.blasLapack.mangling == 'underscore':
      g.write('CFDEFS     = -DFORTRAN_UNDERSCORE\n')
    elif self.blasLapack.mangling == 'caps':
      g.write('CFDEFS     = -DFORTRAN_CAPS\n')
    elif self.blasLapack.mangling == 'unchanged':
      g.write('CFDEFS     = \n')
    else:
      raise RuntimeError('Unknown blas mangling: cannot proceed with pARMS: '+str(self.blasLapack.mangling))
    g.write('CFFLAGS    = ${CFDEFS} -DVOID_POINTER_SIZE_'+str(self.types.sizes['void-p'])+'\n')

    g.write('RM         = rm\n')
    g.write('RMFLAGS    = -rf\n')
    g.write('EXTFLAGS   = -x\n')

    # archive and options
    g.write('AR         = '+self.setCompilers.AR+'\n')
    g.write('ARFLAGS    = '+self.setCompilers.AR_FLAGS+'\n')

    # pARMS lib and its directory
    g.write('LIBDIR     = '+self.installDir+'/lib\n')
    g.write('LIB        = ${LIBDIR}/libparms.a\n')
    g.write('LIBFLAGS   = -L${LIBDIR}\n')
    g.write('PARMS_LIBS = -lparms\n')

    #-----------------------------------------
    g.close()

    if self.installNeeded('makefile.in'):
      try:
        self.logPrintBox('Compiling pARMS; this may take several minutes')
        libDir = os.path.join(self.installDir, self.libdir,'')
        incDir = os.path.join(self.installDir, self.includedir,'')
        if not os.path.isdir(libDir):
          os.mkdir(libDir)
        self.installDirProvider.printSudoPasswordMessage()
        output,err,ret = config.package.Package.executeShellCommand(self.installSudo+'mkdir -p '+libDir, timeout=2500, log=self.log)
        output,err,ret = config.package.Package.executeShellCommand(self.installSudo+'mkdir -p '+incDir, timeout=2500, log=self.log)
        output,err,ret  = config.package.Package.executeShellCommand('cd '+self.packageDir+' && make cleanall && make OBJ3="" && '+self.installSudo+'cp -f include/*.h '+incDir +' && '+self.installSudo+'cp lib/* '+libDir, timeout=2500, log = self.log)
      except RuntimeError as e:
        raise RuntimeError('Error running make on pARMS: '+str(e))
      self.postInstall(output+err,'makefile.in')
    return self.installDir
