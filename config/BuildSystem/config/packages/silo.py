import config.package

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.version   = '4.11'
    self.download  = ['https://github.com/LLNL/Silo/releases/download/v'+self.version+'/silo-'+self.version+'-bsd.tar.gz']
    self.functions = ['DBReadVar']
    self.includes  = ['silo.h']
    self.liblist   = [['libsilo.a']]
    self.buildLanguages    = ['Cxx']
    return

  def formGNUConfigureArgs(self):
    args = config.package.GNUPackage.formGNUConfigureArgs(self)
    args.append('--disable-silex')
    args.append('--without-readline')
    return args
