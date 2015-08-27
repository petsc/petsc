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
