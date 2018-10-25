import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    import os
    config.package.Package.__init__(self, framework)
    self.download          = ['http://www.deshawresearch.com/downloads/download_random123.cgi/Random123-1.09.tar.gz']
    self.functions         = []
    self.includes          = ['Random123/ars.h','Random123/philox.h','Random123/threefry.h']
    self.liblist           = []
    self.cxx               = 0
    self.linkedbypetsc     = 0
    self.defaultLanguage   = 'C++'
    return

  def Install(self):
    import os
    conffile = os.path.join(self.packageDir,self.package+'.petscconf')
    fd = open(conffile, 'w')
    fd.write(self.installDir)
    fd.close()
    if not self.installNeeded(conffile):
      return self.installDir
    try:
      self.logPrintBox('Installing '+self.PACKAGE+' headers')
      packageIncludeDir = os.path.join(self.packageDir,'include','Random123')
      destDir = os.path.join(self.includeDir,'Random123')
      import shutil
      shutil.copytree(packageIncludeDir,destDir)
    except RuntimeError as e:
      raise RuntimeError('Error installing '+self.PACKAGE+' headers: '+str(e))
    self.postInstall('Headers successfully copied\n',conffile)
    return self.installDir


