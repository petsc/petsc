import config.package

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.version   = '6.3.0'
    self.download  = ['https://gmplib.org/download/gmp/gmp-'+self.version+'.tar.bz2',
                      'https://web.cels.anl.gov/projects/petsc/download/externalpackages/gmp-'+self.version+'.tar.bz2']
    self.functions = ['__gmpz_init']
    self.includes  = ['gmp.h']
    self.liblist   = [['libgmp.a']]
    return
