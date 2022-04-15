import config.package
import os

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.versionname       = "GSL_MAJOR_VERSION.GSL_MINOR_VERSION"
    self.download          = ['ftp://ftp.gnu.org/gnu/gsl/gsl-2.7.1.tar.gz']
    self.functions         = ['gsl_sf_hermite_zero']
    self.includes          = ['gsl/gsl_version.h']
    self.liblist           = [['libgsl.a','libgslcblas.a']]
    self.downloadonWindows = 1
    return

  def Install(self):
    macos_deployment = ''
    if 'MACOSX_DEPLOYMENT_TARGET' in os.environ:
      macos_deployment = os.environ['MACOSX_DEPLOYMENT_TARGET']
      msg = 'WARNING! Found environment variable: %s=%s\n' % ('MACOSX_DEPLOYMENT_TARGET', os.environ['MACOSX_DEPLOYMENT_TARGET'])
      self.logPrintBox(msg+'Removing it for GSL build, since it breaks the GSL build')
      del os.environ['MACOSX_DEPLOYMENT_TARGET']
    installDir = config.package.GNUPackage.Install(self)
    if macos_deployment:
      os.environ['MACOSX_DEPLOYMENT_TARGET'] = macos_deployment
    return installDir



