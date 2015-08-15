import config.package
import os

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.download  = ['http://concurrencykit.org/releases/ck-0.4.5.tar.gz']
    self.functions = []
    self.includes  = ['ck_spinlock.h']
    self.liblist   = [['libck.a']]
    self.downloadonWindows = 0
    self.downloadfilename  = 'ck'

