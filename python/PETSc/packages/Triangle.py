import PETSc.package

class Configure(PETSc.package.Package):
  def __init__(self, framework):
    PETSc.package.Package.__init__(self, framework)
    self.download  = ['bk://triangle.bkbits.net/triangle-dev','ftp://ftp.mcs.anl.gov/pub/petsc/externalpackages/triangle.tar.gz']
    self.functions = ['triangulate']
    self.includes  = ['triangle.h']
    self.liblist   = [['libtriangle.a']]
    self.needsMath = 1
    self.complex   = 1
    return

  def setupDependencies(self, framework):
    PETSc.package.Package.setupDependencies(self, framework)
    self.deps = []
    return

  def Install(self):
    import sys
    triangleDir = self.getDir()
    installDir = os.path.join(triangleDir, self.arch.arch)
    if not os.path.isdir(installDir):
      os.mkdir(installDir)
    # We could make a check of the md5 of the current configure framework
    self.logPrintBox('Configuring and compiling Triangle; this may take several minutes')
    try:
      import cPickle
      import logging
      # Split Graphs into its own repository
      oldDir = os.getcwd()
      os.chdir(triangleDir)
      oldLog = logging.Logger.defaultLog
      logging.Logger.defaultLog = file(os.path.join(triangleDir, 'build.log'), 'w')
      make = self.getModule(triangleDir, 'make').Make(configureParent = cPickle.loads(cPickle.dumps(self.framework)))
      make.prefix = installDir
      make.framework.argDB['with-petsc'] = 1
      make.builder.argDB['ignoreCompileOutput'] = 1
      make.run()
      logging.Logger.defaultLog = oldLog
      os.chdir(oldDir)
    except RuntimeError, e:
      raise RuntimeError('Error running configure on Triangle: '+str(e))
    self.framework.actions.addArgument('Triangle', 'Install', 'Installed Triangle into '+installDir)
    return triangleDir
