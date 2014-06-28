import PETSc.package
import os

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.functions         = ['SSLv23_method']
    self.includes          = ['openssl/ssl.h']
    self.liblist           = [['libssl.a','libcrypto.a']]
    self.complex           = 1   # 0 means cannot use complex
    self.lookforbydefault  = 1
    self.double            = 0   # 1 means requires double precision
    self.requires32bitint  = 0;  # 1 means that the package will not work with 64 bit integers
    return

  def setupHelp(self, help):
    import nargs
    PETSc.package.NewPackage.setupHelp(self, help)
    help.addArgument('SSL', '-with-ssl-certificate=<bool>',nargs.ArgBool(None, 0, 'Require certificate with SSL'))

  def setupDependencies(self, framework):
    PETSc.package.NewPackage.setupDependencies(self, framework)
    self.deps = []
    return

  def getSearchDirectories(self):
    yield ''
    return

  def configureLibrary(self):
    if self.framework.argDB['with-ios']: 
      self.found = 0
      return
    PETSc.package.NewPackage.configureLibrary(self)

  def consistencyChecks(self):
   PETSc.package.NewPackage.consistencyChecks(self)
   if self.framework.argDB['with-'+self.package]:
     if self.framework.argDB['with-ssl-certificate']:
       self.addDefine('USE_SSL_CERTIFICATE','1')
