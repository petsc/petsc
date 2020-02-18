import config.package
import os

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.functions        = ['regexec', 'regcomp', 'regfree']
    self.includes         = ['regex.h']
    self.liblist          = [['libregex.a']]
    self.lookforbydefault = 1
    return
