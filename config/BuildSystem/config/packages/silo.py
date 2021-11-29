import config.package

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.download  = ['https://wci.llnl.gov/content/assets/docs/simulation/computer-codes/silo/silo-4.10/silo-4.10-bsd-smalltest.tar.gz']
    self.functions = ['DBReadVar']
    self.includes  = ['silo.h']
    self.liblist   = [['libsilo.a']]
    self.pkgname   = 'silo-4.10-bsd'
    self.buildLanguages    = ['Cxx']
    return

  def formGNUConfigureArgs(self):
    args = config.package.GNUPackage.formGNUConfigureArgs(self)
    args.append('--disable-silex')
    args.append('--without-readline')
    return args
