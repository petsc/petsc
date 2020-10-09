from __future__ import generators
import config.base
import os

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = 'PETSC'
    self.substPrefix  = 'PETSC'
    self.optionName = 'with-macos-firewall-rules'
    self.isEnabled = False
    self.isDarwin = config.setCompilers.Configure.isDarwin(self.log)
    return

  def __str1__(self):
    if self.isEnabled:
      return '  NOTE: %s is on, so make test will ask for your sudo password to set macOS firewall rules\n' % self.optionName
    else:
      return ''

  def setupDependencies(self, framework):
    if not self.isDarwin:
      return
    config.base.Configure.setupDependencies(self, framework)
    self.mpi = framework.require('config.packages.MPI', self)

  def setupHelp(self, help):
    if not self.isDarwin:
      return
    import nargs
    help.addArgument('PETSc', '-%s=<bool>' % self.optionName, nargs.ArgBool(None, 0, 'Activates automatic addition of firewall rules (blocking incoming connections) during testing on macOS to prevent firewall popup windows. Uses sudo so gmakefile.test will ask for your password.'))

  def configure(self):
    self.isEnabled = self.isDarwin and self.argDB[self.optionName]
    if not self.isEnabled:
      return
    # self.mpi.mpiexecExecutable is already absolute path but realpath also removes symlinks
    mpiexec = os.path.realpath(self.mpi.mpiexecExecutable)
    self.addMakeMacro('MACOS_FIREWALL_MPIEXEC', mpiexec)
