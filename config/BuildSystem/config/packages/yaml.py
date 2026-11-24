import config.package

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.version   = '0.2.7'
    self.download  = ['https://github.com/yaml/libyaml/releases/download/'+self.version+'/yaml-'+self.version+'.tar.gz',
                      'https://web.cels.anl.gov/projects/petsc/download/externalpackages/yaml-'+self.version+'.tar.gz']
    self.functions = ['yaml_parser_load']
    self.includes  = ['yaml.h']
    self.liblist   = [['libyaml.a']]
    return
