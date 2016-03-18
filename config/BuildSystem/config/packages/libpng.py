import config.package

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.download         = ['http://downloads.sourceforge.net/project/libpng/libpng16/1.6.21/libpng-1.6.21.tar.gz']
    self.includes         = ['png.h']
    self.liblist          = [['libpng.a']]
    self.functions        = ['png_create_write_struct']
    self.lookforbydefault = 0
    self.needsMath        = 1
    self.needsCompression = 1

  def generateLibList(self, framework):
    '''First try library list without compression libraries (zlib) then try with'''
    if self.libraries.compression:
      zlib = self.libraries.compression
      for libs in self.liblist[:]:
        self.liblist.append(libs + zlib)
    return config.package.GNUPackage.generateLibList(self, framework)
