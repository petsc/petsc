import config.package

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.download         = ['http://www.ijg.org/files/jpegsrc.v9c.tar.gz',
                             'http://ftp.mcs.anl.gov/pub/petsc/externalpackages/jpegsrc.v9c.tar.gz']
    self.downloaddirnames  = ['jpeg']
    self.includes         = ['jpeglib.h']
    self.liblist          = [['libjpeg.a']]
    self.functions        = ['jpeg_destroy_compress']
    self.lookforbydefault = 0

  def configureLibrary(self):
    config.package.Package.configureLibrary(self)
    # removed from the list of defines because there is already and entry from checking the library exists
    # note the duplication that would otherwise occur comes from the package having a lib at the beginning of the name
    self.libraries.delDefine('HAVE_LIBJPEG')