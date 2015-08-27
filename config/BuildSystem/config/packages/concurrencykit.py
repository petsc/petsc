import config.package
import os

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.download  = ['http://concurrencykit.org/releases/ck-0.4.5.tar.gz']
    self.functions = []
    self.includes  = ['ck_spinlock.h']
    self.liblist   = [['libck.a']]
    self.downloadonWindows = 0
    self.downloadfilename  = 'ck'

  def formGNUConfigureArgs(self):
    # onfigure errors out on certain standard configure arguments
    args = config.package.GNUPackage.formGNUConfigureArgs(self)
    rejects = ['--disable-cxx','--disable-fortran', '--disable-fc','--disable-f77','--disable-f90']
    self.logPrint('MPICH is rejecting configure arguments '+str(rejects))
    return [arg for arg in args if not arg in rejects]

  def checkForCorrectness(self):
    include = '#include <ck_spinlock.h>'
    body    = 'ck_spinlock_t ck_spinlock; ck_spinlock_init(&ck_spinlock);ck_spinlock_lock(&ck_spinlock);ck_spinlock_unlock(&ck_spinlock);'
    oldFlags = self.compilers.CPPFLAGS
    oldLibs  = self.compilers.LIBS
    self.compilers.CPPFLAGS += ' '+self.headers.toString(self.include)
    self.compilers.LIBS = self.libraries.toString(self.lib)+' '+self.compilers.LIBS
    if not self.checkLink(include, body):
      raise RuntimeError('Concurrencykit cannot be used')
    self.compilers.CPPFLAGS = oldFlags
    self.compilers.LIBS = oldLibs

  def configureLibrary(self):
    '''Calls the regular package configureLibrary and then does an additional test needed'''
    if 'with-'+self.package+'-shared' in self.argDB:
      self.argDB['with-'+self.package] = 1
    config.package.Package.configureLibrary(self)
    self.executeTest(self.checkForCorrectness)
