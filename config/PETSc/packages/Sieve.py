import PETSc.package

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    import os
    PETSc.package.NewPackage.__init__(self, framework)
    self.include         = [os.path.abspath(os.path.join('include', 'sieve'))]
    self.libdir          = ''
    self.archIndependent = 1
    self.cxx             = 1
    self.complex         = 1
    self.worksonWindows  = 1
    self.double          = 0
    return

  def setupDependencies(self, framework):
    PETSc.package.NewPackage.setupDependencies(self, framework)
    self.boost    = self.framework.require('config.packages.boost',self)
    self.mpi      = self.framework.require('config.packages.MPI',self)
    #self.triangle = self.framework.require('PETSc.packages.Triangle',self)
    #self.tetgen   = self.framework.require('PETSc.packages.TetGen',self)
    self.deps = [self.boost, self.mpi]
    return

  def getSearchDirectories(self):
    return [self.petscdir.dir]
    
  def setupHelp(self, help):
    PETSc.package.NewPackage.setupHelp(self, help)
    import nargs
    help.addArgument('Sieve', '-with-opt-sieve=<bool>', nargs.ArgBool(None, 1, 'Use IMesh which are optimized for interval point sets'))
    help.addArgument('Sieve', '-with-sieve-memory-logging=<bool>', nargs.ArgBool(None, 0, 'Turn on memory logging for Sieve objects'))
    return

  def Install(self):
    import sys
    sieveDir = self.getDir()
    self.framework.actions.addArgument('Sieve', 'Install', 'Installed Sieve into '+sieveDir)
    return sieveDir

  def consistencyChecks(self):
    PETSc.package.NewPackage.consistencyChecks(self)
    if self.framework.argDB['with-'+self.package]:
      if not self.languages.clanguage == 'Cxx':
        raise RuntimeError('Sieve requires C++. Suggest using --with-clanguage=cxx')
      if not self.boost.found:
        raise RuntimeError('Sieve requires boost, and configure could not locate it. Suggest using --download-boost=1')
      if 'with-opt-sieve' in self.argDB and self.argDB['with-opt-sieve']:
        self.addDefine('OPT_SIEVE', 1)
        self.addDefine('MESH_TYPE', 'ALE::IMesh<PetscInt, PetscScalar>')
      else:
        self.addDefine('MESH_TYPE', 'ALE::Mesh<PetscInt, PetscScalar>')
      if 'with-sieve-memory-logging' in self.argDB and self.argDB['with-sieve-memory-logging']:
        self.framework.addDefine('ALE_MEM_LOGGING', 1)
    return
