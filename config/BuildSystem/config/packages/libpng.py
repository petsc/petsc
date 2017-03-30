import config.package

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.download         = ['https://sourceforge.net/projects/libpng/files/libpng16/1.6.28/libpng-1.6.28.tar.gz']
    self.includes         = ['png.h']
    self.liblist          = [['libpng.a']]
    self.functions        = ['png_create_write_struct']
    self.lookforbydefault = 0

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.mathlib        = framework.require('config.packages.mathlib',self)
    self.zlib           = framework.require('config.packages.zlib',self)
    self.deps           = [self.mathlib,self.zlib]
    return

  def formGNUConfigureArgs(self):
    args = config.package.GNUPackage.formGNUConfigureArgs(self)
    args.append('CPPFLAGS="'+self.headers.toStringNoDupes(self.dinclude)+'"')
    args.append('LIBS="'+self.libraries.toStringNoDupes(self.dlib)+'"')
    return args
