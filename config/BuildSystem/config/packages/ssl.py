import config.package
import os

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.gitcommit         = 'openssl-3.0.8'
    self.download          = ['git://https://github.com/openssl/openssl.git']
    self.functions         = ['SSL_version']
    self.includes          = ['openssl/ssl.h']
    self.liblist           = [['libssl.a','libcrypto.a']]
    self.configureName     = 'Configure'

  def setupHelp(self, help):
    import nargs
    config.package.GNUPackage.setupHelp(self, help)
    help.addArgument('SSL', '-with-ssl-certificate=<bool>',nargs.ArgBool(None, 0, 'Require certificate with SSL'))

  def setupDependencies(self, framework):
    config.package.GNUPackage.setupDependencies(self, framework)
    self.deps = []

  def getSearchDirectories(self):
    '''macOS no longer provides openssl include files. On macOS brew puts them in the second location listed here'''
    return ['',os.path.join('/usr','local','opt','openssl@1.1'),os.path.join('/usr','local','opt','openssl')]

  def formGNUConfigureArgs(self):
    '''This sets up the prefix, compiler flags, shared flags, and other generic arguments
       that are fed into the configure script supplied with the package.
       Override this to set options needed by a particular package'''
    args=[]
    ## prefix
    args.append('--prefix='+self.installDir)
    args.append('MAKE='+self.make.make)
    args.append('--libdir='+os.path.join(self.installDir,self.libdir))
    ## compiler args
    self.pushLanguage('C')
    if not self.installwithbatch and hasattr(self.setCompilers,'cross_cc'):
      args.append('CC="'+self.setCompilers.cross_cc+'"')
    else:
      args.append('CC="'+self.getCompiler()+'"')
    args.append('CFLAGS="'+self.updatePackageCFlags(self.getCompilerFlags())+'"')
    args.append('AR="'+self.setCompilers.AR+'"')
    args.append('ARFLAGS="'+self.setCompilers.AR_FLAGS+'"')
    if not self.installwithbatch and hasattr(self.setCompilers,'cross_LIBS'):
      args.append('LIBS="'+self.setCompilers.cross_LIBS+'"')
    if self.setCompilers.LDFLAGS:
      args.append('LDFLAGS="'+self.setCompilers.LDFLAGS+'"')
    self.popLanguage()
    return args

  def configureLibrary(self):
    if 'with-ios' in self.argDB and self.argDB['with-ios']:
      self.found = 0
      return
    config.package.GNUPackage.configureLibrary(self)

  def consistencyChecks(self):
   config.package.GNUPackage.consistencyChecks(self)
   if self.argDB['with-'+self.package]:
     if self.argDB['with-ssl-certificate']:
       self.addDefine('USE_SSL_CERTIFICATE','1')
