import config.package

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.download         = ['http://www.ijg.org/files/jpegsrc.v9b.tar.gz']
    self.includes         = ['jpeglib.h']
    self.liblist          = [['libjpeg.a']]
    self.functions        = ['jpeg_destroy_compress']
    self.lookforbydefault = 0
