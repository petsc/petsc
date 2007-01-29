#!/usr/bin/env python
import PETSc.package

class Configure(PETSc.package.Package):
  def __init__(self, framework):
    PETSc.package.Package.__init__(self, framework)
    self.download     = ['hg://petsc.cs.iit.edu/petsc/ParMetis-dev','ftp://ftp.mcs.anl.gov/pub/petsc/externalpackages/ParMetis-dev.tar.gz']
    self.functions    = ['ParMETIS_V3_PartKway']
    self.includes     = ['parmetis.h']
    self.liblist      = [['libparmetis.a','libmetis.a']]
    self.needsMath    = 1
    self.complex      = 1
    return

  def setupDependencies(self, framework):
    PETSc.package.Package.setupDependencies(self, framework)
    self.mpi = framework.require('config.packages.MPI', self)
    self.deps = [self.mpi]
    return

  def Install(self):
    import os, sys
    parmetisDir = self.getDir()
    installDir = os.path.join(parmetisDir, self.arch.arch)
    if not os.path.isdir(installDir):
      os.mkdir(installDir)
    # We could make a check of the md5 of the current configure framework
    self.logPrintBox('Configuring and compiling ParMetis; this may take several minutes')
    try:
      import cPickle
      import logging
      # Split Graphs into its own repository
      oldDir = os.getcwd()
      os.chdir(parmetisDir)
      oldLog = logging.Logger.defaultLog
      logging.Logger.defaultLog = file(os.path.join(parmetisDir, 'build.log'), 'w')
      mod  = self.getModule(parmetisDir, 'make')
      make = mod.Make(configureParent = cPickle.loads(cPickle.dumps(self.framework)), module = mod)
      make.prefix = installDir
      make.framework.argDB['with-petsc'] = 1
      make.builder.argDB['ignoreCompileOutput'] = 1
      make.run()
      del sys.modules['make']
      logging.Logger.defaultLog = oldLog
      os.chdir(oldDir)
    except RuntimeError, e:
      raise RuntimeError('Error running configure on ParMetis: '+str(e))
    self.framework.actions.addArgument('ParMetis', 'Install', 'Installed ParMetis into '+installDir)
    return parmetisDir
