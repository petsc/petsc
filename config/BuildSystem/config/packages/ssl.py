import config.package
import os

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.functions         = ['SSL_version']
    self.includes          = ['openssl/ssl.h']
    self.liblist           = [['libssl.a','libcrypto.a']]

  def setupHelp(self, help):
    import nargs
    config.package.Package.setupHelp(self, help)
    help.addArgument('SSL', '-with-ssl-certificate=<bool>',nargs.ArgBool(None, 0, 'Require certificate with SSL'))

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.deps = []

  def getSearchDirectories(self):
    '''macOS no longer provides openssl include files. On macOS brew puts them in the second location listed here'''
    return ['',os.path.join('/usr','local','opt','openssl@1.1'),os.path.join('/usr','local','opt','openssl')]

  def configureLibrary(self):
    if 'with-ios' in self.argDB and self.argDB['with-ios']: 
      self.found = 0
      return
    config.package.Package.configureLibrary(self)

  def consistencyChecks(self):
   config.package.Package.consistencyChecks(self)
   if self.argDB['with-'+self.package]:
     if self.argDB['with-ssl-certificate']:
       self.addDefine('USE_SSL_CERTIFICATE','1')
