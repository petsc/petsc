import config.package

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.includes         = ['']
    self.includedir       = ''
    self.libdir           = ''
    self.useddirectly     = 0
    return

  def configureLibrary(self):
    try:
      import numpy
    except:
      raise RuntimeError('Could not find numpy, either fix PYTHONPATH and rerun or install it')
    return
