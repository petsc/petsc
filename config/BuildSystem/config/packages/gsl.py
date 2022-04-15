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




