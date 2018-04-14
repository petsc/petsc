import config.package

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.download     = ['http://pyyaml.org/download/libyaml/yaml-0.1.4.tar.gz',
                         'http://ftp.mcs.anl.gov/pub/petsc/externalpackages/yaml-0.1.4.tar.gz']
    self.functions = ['yaml_parser_initialize']
    self.includes  = ['yaml.h']
    self.liblist   = [['libyaml.a']]
    self.pkgname   = 'yaml-0.1'
    return


