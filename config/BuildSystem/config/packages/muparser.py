# MuParser is found at https://github.com/beltoforion/muparser/
import config.package

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.download  = ['http://ftp.mcs.anl.gov/pub/petsc/externalpackages/muparser_v2_2_4.tar.gz']
    self.functions = []
    self.includes  = ['muParser.h']
    self.liblist   = [['libmuparser.a']]
    self.pkgname   = 'muparser-2.2.4'
    self.cxx       = 1
    return

  def formGNUConfigureArgs(self):
    args = config.package.GNUPackage.formGNUConfigureArgs(self)
    args.append('--enable-shared=no')
    args.append('--enable-samples=no')
    args.append('--enable-debug=no')
    return args
