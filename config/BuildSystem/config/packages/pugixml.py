import config.package

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.download         = ['https://github.com/zeux/pugixml/releases/download/v1.15/pugixml-1.15.tar.gz']
    self.downloaddirnames = ['pugixml']
    self.functions        = []
    self.includes         = ['pugixml.hpp']
    self.liblist          = [['libpugixml.a']]
    self.precisions       = ['double']
    self.buildLanguages   = ['Cxx']
    return
