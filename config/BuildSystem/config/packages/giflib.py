import config.package

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.download         = ['http://downloads.sourceforge.net/project/giflib/giflib-5.1.2.tar.bz2']
    self.includes         = ['gif_lib.h']
    self.liblist          = [['libgif.a'], ['libungif.a']]
    self.functions        = ['EGifOpenFileName']
    self.lookforbydefault = 0
