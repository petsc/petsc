import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.download  = ['http://ftp.mcs.anl.gov/pub/petsc/externalpackages/sprng-1.0.tar.gz']
    self.functions = ['make_new_seed_mpi']
    self.includes  = ['sprng.h']
    self.liblist   = [['liblcg.a']]
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.mpi             = framework.require('config.packages.MPI',self)
    self.deps = [self.mpi]
    return

  def Install(self):
    import os

    g = open(os.path.join(self.packageDir,'SRC','make.PETSC'),'w')

    g.write('AR             = '+self.setCompilers.AR+'\n')
    g.write('ARFLAGS        = '+self.setCompilers.AR_FLAGS+'\n')
    g.write('AR_LIB_SUFFIX  = '+self.setCompilers.AR_LIB_SUFFIX+'\n')
    g.write('RANLIB         = '+self.setCompilers.RANLIB+'\n')

    self.setCompilers.pushLanguage('C')
    cflags = self.updatePackageCFlags(self.setCompilers.getCompilerFlags())
    cflags += ' ' + self.headers.toString(self.mpi.include)+' '+self.headers.toString('.')
    cflags += ' ' + '-DSPRNG_MPI' # either using MPI or MPIUNI

    g.write('CC             = '+self.setCompilers.getCompiler()+'\n')
    g.write('CFLAGS         = '+cflags+'\n')
    g.write('CLD            = $(CC)\n')
    g.write('MPICC          = $(CC)\n')
    g.write('CPP            ='+self.getPreprocessor()+'\n')
    self.setCompilers.popLanguage()

    # extra unused options
    g.write('CLDFLAGS       = \n')
    g.write('F77            = echo\n')
    g.write('F77LD          = $(F77)\n')
    g.write('FFXN 	    = -DAdd_\n')
    g.write('FSUFFIX 	    = F\n')
    g.write('MPIF77 	    = echo\n')
    g.write('FFLAGS 	    = \n')
    g.write('F77LDFLAGS     = \n')
    g.close()

    if self.installNeeded(os.path.join('SRC','make.PETSC')):
      try:
        self.logPrintBox('Compiling and installing SPRNG; this may take several minutes')
        self.installDirProvider.printSudoPasswordMessage()
        output,err,ret = config.package.Package.executeShellCommand(self.installSudo+'mkdir -p '+os.path.join(self.installDir,'lib'), timeout=2500, log=self.log)
        output,err,ret = config.package.Package.executeShellCommand(self.installSudo+'mkdir -p '+os.path.join(self.installDir,'include'), timeout=2500, log=self.log)
        output,err,ret  = config.package.Package.executeShellCommand('cd '+self.packageDir+' && make realclean && cd SRC && make && cd .. && '+self.installSudo+' cp -f lib/*.a '+os.path.join(self.installDir,self.libdir,'')+' && '+self.installSudo+'cp -f include/*.h '+os.path.join(self.installDir,self.includedir,''), timeout=2500, log = self.log)
      except RuntimeError as e:
        raise RuntimeError('Error running make on SPRNG: '+str(e))
      self.postInstall(output+err,os.path.join('SRC','make.PETSC'))
    return self.installDir
