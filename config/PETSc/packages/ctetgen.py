import PETSc.package

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.download          = ['http://petsc.cs.iit.edu/petsc/externalpackages/ctetgen/archive/ctetgen-0.1.tar.gz',
                              'http://ftp.mcs.anl.gov/pub/petsc/externalpackages/ctetgen-0.1.tar.gz']
    self.functions         = []
    self.includes          = []
    self.liblist           = [['libctetgen.a']]
    self.compileCtetgen    = 0
    return

  def setupDependencies(self, framework):
    PETSc.package.NewPackage.setupDependencies(self, framework)
    self.deps       = []
    return

  def Install(self):
    import os
    import sys
    # Configure ctetgen
    g = open(os.path.join(self.packageDir,'make.inc'),'w')
    g.write('SHELL          = '+self.programs.SHELL+'\n')
    g.write('CP             = '+self.programs.cp+'\n')
    g.write('RM             = '+self.programs.RM+'\n')
    g.write('MKDIR          = '+self.programs.mkdir+'\n')

    g.write('AR             = '+self.setCompilers.AR+'\n')
    g.write('ARFLAGS        = '+self.setCompilers.AR_FLAGS+'\n')
    g.write('AR_LIB_SUFFIX  = '+self.setCompilers.AR_LIB_SUFFIX+'\n')
    g.write('RANLIB         = '+self.setCompilers.RANLIB+'\n')
    g.write('PREFIX         = '+self.installDir+'\n')
    g.write('CTETGENLIB     = libctetgen.$(AR_LIB_SUFFIX)\n')

    self.setCompilers.pushLanguage(self.languages.clanguage)
    g.write('CC             = '+self.setCompilers.getCompiler()+'\n')
    g.write('CFLAGS         = '+self.setCompilers.getCompilerFlags()+' -I'+os.path.join(self.petscdir.dir,self.arch,'include')+' -I'+os.path.join(self.petscdir.dir,'include')+' '+self.headers.toString(self.mpi.include)+'\n')
    # ctetgen uses defaut 'make' targets, and this uses TARGET_ARCH var. If this var
    # is set incorrectly in user env - build breaks.
    g.write('TARGET_ARCH    = \n')

    self.setCompilers.popLanguage()
    g.close()

    if self.installNeeded('make.inc'):
      self.compileCtetgen = 1
    return self.installDir

  def postProcess(self):
    if self.compileCtetgen:
      try:
        self.logPrintBox('Compiling Ctetgen; this may take several minutes')
        output,err,ret  = PETSc.package.NewPackage.executeShellCommand('cd '+self.packageDir+' && make clean && make lib && make install',timeout=1000, log = self.framework.log)
        self.framework.log.write(output)
      except RuntimeError, e:
        raise RuntimeError('Error running make on Ctetgen: '+str(e))
      self.postInstall(output+err,'make.inc')
