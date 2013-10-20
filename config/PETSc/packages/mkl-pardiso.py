import PETSc.package
import os

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.libdir           = os.path.join(self.libdir, 'intel64')     # location of libraries in the package directory tree
    self.altlibdir        = os.path.join(self.libdir, 'mic')   # alternate location of libraries in the package directory tree
    self.includes     = ['mkl_pardiso.h']
    self.liblist      = [['-lmkl_intel_lp64', '-lmkl_intel_thread', '-lmkl_core', '-liomp5', 'pthread', '-lm']]
    self.double       = 0
    self.requires32bitint = 0
    self.complex      = 1
    return

  def setupDependencies(self, framework):
    PETSc.package.NewPackage.setupDependencies(self, framework)
    self.blasLapack = framework.require('config.packages.BlasLapack',self)
    self.deps       = [self.blasLapack]
    return
  
  def generateGuesses(self):
    if 'with-'+self.package+'-dir' in self.framework.argDB:
      d = self.framework.argDB['with-'+self.package+'-dir']
      yield('User specified root directory '+self.PACKAGE, d, ['-Wl,-rpath,' + os.path.join(d, self.libdir) + ' -Wl,--start-group -L' + os.path.join(d, self.libdir) + ' -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -Wl,--end-group -lpthread -lm -liomp5'], [os.path.join(d, 'include')])
      yield('User specified root directory '+self.PACKAGE, d, ['-Wl,-rpath,' + os.path.join(d, self.libdir) + ' -Wl,--start-group -L' + os.path.join(d, self.altlibdir) + ' -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -Wl,--end-group -lpthread -lm -liomp5'], [os.path.join(d, 'include')])
      if 'with-'+self.package+'-include' in self.framework.argDB:
        raise RuntimeError('Do not set --with-'+self.package+'-include if you set --with-'+self.package+'-dir')
      if 'with-'+self.package+'-lib' in self.framework.argDB:
        raise RuntimeError('Do not set --with-'+self.package+'-lib if you set --with-'+self.package+'-dir')
      raise RuntimeError('--with-'+self.package+'-dir='+self.framework.argDB['with-'+self.package+'-dir']+' did not work')

    if 'with-'+self.package+'-include' in self.framework.argDB and not 'with-'+self.package+'-lib' in self.framework.argDB:
      raise RuntimeError('If you provide --with-'+self.package+'-include you must also supply with-'+self.package+'-lib\n')
    if 'with-'+self.package+'-lib' in self.framework.argDB and not 'with-'+self.package+'-include' in self.framework.argDB:
      if self.includes:
        raise RuntimeError('If you provide --with-'+self.package+'-lib you must also supply with-'+self.package+'-include\n')
    if 'with-'+self.package+'-include-dir' in self.framework.argDB:
        raise RuntimeError('Use --with-'+self.package+'-include; not --with-'+self.package+'-include-dir')

    if 'with-'+self.package+'-include' in self.framework.argDB or 'with-'+self.package+'-lib' in self.framework.argDB:
      libs = self.framework.argDB['with-'+self.package+'-lib']
      inc  = []
      if self.includes:
        inc = self.framework.argDB['with-'+self.package+'-include']
      # hope that package root is one level above first include directory specified
        d   = os.path.dirname(inc[0])
      else:
        d   = None
      if not isinstance(inc, list): inc = inc.split(' ')
      if not isinstance(libs, list): libs = libs.split(' ')
      inc = [os.path.abspath(i) for i in inc]
      yield('User specified '+self.PACKAGE+' libraries', d, libs, inc)
      msg = '--with-'+self.package+'-lib='+str(self.framework.argDB['with-'+self.package+'-lib'])
      if self.includes:
        msg += ' and \n'+'--with-'+self.package+'-include='+str(self.framework.argDB['with-'+self.package+'-include'])
      msg += ' did not work'
      raise RuntimeError(msg)

    if not self.lookforbydefault:
      raise RuntimeError('You must specify a path for '+self.name+' with --with-'+self.package+'-dir=<directory>\nIf you do not want '+self.name+', then give --with-'+self.package+'=0\nYou might also consider using --download-'+self.package+' instead')

 
