import PETSc.package
import os

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.functions         = ['SSLv23_method']
    self.includes          = ['openssl/ssl.h']
    self.liblist           = [['libssl.a','libcrypto.a']]
    self.complex           = 1   # 0 means cannot use complex
    self.lookforbydefault  = 0
    self.double            = 0   # 1 means requires double precision
    self.requires32bitint  = 0;  # 1 means that the package will not work with 64 bit integers
    self.worksonWindows    = 0  # 1 means that package can be used on Microsof Windows
    return

  def setupDependencies(self, framework):
    PETSc.package.NewPackage.setupDependencies(self, framework)
    return

  def getSearchDirectories(self):
    yield ''
    return
