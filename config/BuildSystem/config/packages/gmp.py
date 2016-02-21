import config.package

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.download  = ['https://gmplib.org/download/gmp/gmp-6.0.0a.tar.bz2']
    self.functions = ['__gmpz_init']
    self.includes  = ['gmp.h']
    self.liblist   = [['libgmp.a']]
    self.pkgname   = 'gmp-6.0.0'
    return
