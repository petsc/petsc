import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.pyver = None
    self.cyver = None
    self.cython = 0
    self.numpy = 0
    self.skippackagewithoptions = 1
    return

  def setupHelp(self,help):
    import nargs
    help.addArgument('PETSc', '-with-python-exec=<executable>', nargs.Arg(None, None, 'Alternate Python executable to use for mpi4py/petsc4py'))
    help.addArgument('PETSc', '-have-numpy=<bool>', nargs.ArgBool(None, None, 'Whether numpy python module is installed (default: autodetect)'))
    return

  def configure(self):
    '''determine python binary to use'''
    if 'with-python-exec' in self.argDB:
      self.getExecutable(self.argDB['with-python-exec'], getFullPath=1, resultName='pyexe', setMakeMacro = 0)
    else:
      import sys
      self.pyexe = sys.executable
    self.addDefine('PYTHON_EXE','"'+self.pyexe+'"')
    self.addMakeMacro('PYTHON_EXE','"'+self.pyexe+'"')
    self.executablename = 'pyexe'
    self.found = 1

    try:
      self.pyver,err1,ret1  = config.package.Package.executeShellCommand([self.pyexe,'-c','import sysconfig;print(sysconfig.get_python_version())'],timeout=60, log = self.log)
    except:
      self.logPrint('Unable to determine version of',self.pyexe)

    try:
      self.cyver,err1,ret1  = config.package.Package.executeShellCommand([self.pyexe,'-c','import cython;print(cython.__version__)'],timeout=60, log = self.log)
      self.cython = 1
    except:
      self.logPrint('Python being used '+self.pyexe+' does not have the Cython package')

    have_numpy = self.argDB.get('have-numpy', None)
    if have_numpy is not None:
      self.numpy = int(have_numpy)
    else:
      try:
        output1,err1,ret1  = config.package.Package.executeShellCommand(self.pyexe + ' -c "import numpy"',timeout=60, log = self.log)
        self.numpy = 1
      except:
        self.logPrint('Python being used '+self.pyexe+' does not have the numpy package')
    return
