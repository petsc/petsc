import config.package

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.versionname      = 'PNG_HEADER_VERSION_STRING'
    self.version          = '1.6.37'
    self.download         = ['https://sourceforge.net/projects/libpng/files/libpng16/'+self.version+'/libpng-1.6.37.tar.gz',
                             'http://ftp.mcs.anl.gov/pub/petsc/externalpackages/libpng-'+self.version+'.tar.gz']
    self.includes         = ['png.h']
    self.liblist          = [['libpng.a']]
    self.functions        = ['png_create_write_struct']

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.mathlib        = framework.require('config.packages.mathlib',self)
    self.zlib           = framework.require('config.packages.zlib',self)
    self.deps           = [self.mathlib,self.zlib]
    return

  def versionToStandardForm(self,ver):
    import re
    return re.compile(r'[=A-Za-z]([\.0-9]*)-[A-Za-z]*').search(ver).group(1)

  def formGNUConfigureArgs(self):
    args = config.package.GNUPackage.formGNUConfigureArgs(self)
    args.append('CPPFLAGS="'+self.headers.toStringNoDupes(self.dinclude)+'"')
    args.append('LIBS="'+self.libraries.toStringNoDupes(self.dlib)+'"')
    return args

  def configureLibrary(self):
    config.package.Package.configureLibrary(self)
    # removed from the list of defines because there is already and entry from checking the library exists
    # note the duplication that would otherwise occur comes from the package having a lib at the beginning of the name
    self.libraries.delDefine('HAVE_LIBPNG')
