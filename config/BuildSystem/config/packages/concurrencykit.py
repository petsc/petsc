import config.package
import os

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.download          = ['http://concurrencykit.org/releases/ck-0.4.5.tar.gz',
                              'http://ftp.mcs.anl.gov/pub/petsc/externalpackages/ck-0.4.5.tar.gz']
    self.functions         = []
    self.includes          = ['ck_spinlock.h']
    self.liblist           = [['libck.a']]
    self.downloadonWindows = 0
    self.downloaddirnames  = ['ck-']

  def setupDependencies(self, framework):
    config.package.GNUPackage.setupDependencies(self, framework)
    self.languages      = framework.require('PETSc.options.languages',   self)

  def formGNUConfigureArgs(self):
    if not self.languages.clanguage == 'C':
      raise RuntimeError('Concurrencykit cannot be used with --with-clanguage=cxx since it cannot be included in C++ code\nUse --with-clanguage=c but you can still compile your application with C++')

    args = config.package.GNUPackage.formGNUConfigureArgs(self)

    # CK configure is buggy and ignores --disable-shared; you must turn off PIC to turn off shared libraries
    if not ((self.argDB['with-shared-libraries'] and 'download-'+self.package+'-shared' not in self.framework.clArgDB) or  self.argDB['download-'+self.package+'-shared']):
      args.append('--without-pic')

    # Concurrency configure errors out on certain standard configure arguments
    return self.rmArgs(args,['--disable-cxx','--disable-fortran', '--disable-fc','--disable-f77','--disable-f90'])

  def checkForCorrectness(self):
    include = '#include <ck_spinlock.h>'
    body    = 'ck_spinlock_t ck_spinlock; ck_spinlock_init(&ck_spinlock);ck_spinlock_lock(&ck_spinlock);ck_spinlock_unlock(&ck_spinlock);'
    oldFlags = self.compilers.CPPFLAGS
    oldLibs  = self.compilers.LIBS
    self.compilers.CPPFLAGS += ' '+self.headers.toString(self.include)
    self.compilers.LIBS = self.libraries.toString(self.lib)+' '+self.compilers.LIBS
    self.pushLanguage('C')
    if not self.checkLink(include, body):
      raise RuntimeError('Concurrencykit cannot be used')
    self.popLanguage()
    self.compilers.CPPFLAGS = oldFlags
    self.compilers.LIBS = oldLibs

  def configureLibrary(self):
    '''Calls the regular package configureLibrary and then does an additional test needed'''
    if 'with-'+self.package+'-shared' in self.argDB:
      self.argDB['with-'+self.package] = 1
    config.package.Package.configureLibrary(self)
    self.executeTest(self.checkForCorrectness)
