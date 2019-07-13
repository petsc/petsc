import config.package

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.download          = ['https://support.hdfgroup.org/ftp/lib-external/szip/2.1.1/src/szip-2.1.1.tar.gz',
                              'http://ftp.mcs.anl.gov/pub/petsc/externalpackages/szip-2.1.1.tar.gz']
    self.functions         = ['SZ_BufftoBuffCompress', 'SZ_BufftoBuffDecompress']
    self.includes          = ['szlib.h']
    self.liblist           = [['libsz.a'],['szlib.lib']]
    self.downloaddirnames  = ['szip']
    return

