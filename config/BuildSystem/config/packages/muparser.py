import config.package
import os

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.gitcommit  = 'v2.3.3-1'
    self.download   = ['git://https://github.com/beltoforion/muparser/','https://github.com/beltoforion/muparser/archive/'+self.gitcommit+'.tar.gz']
    self.includes   = ['muParser.h']
    self.liblist    = [['libmuparser.a']]
    self.functions  = ['mupCreate']
    self.precisions = ['double']
    self.buildLanguages = ['Cxx']
    return

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.compilerFlags = framework.require('config.compilerFlags',self)
    self.mathlib       = framework.require('config.packages.mathlib',self)
    self.deps          = [self.mathlib]
    return

  def formCMakeConfigureArgs(self):
    args = config.package.CMakePackage.formCMakeConfigureArgs(self)
    args.append('-DENABLE_SAMPLES=OFF')
    args.append('-DENABLE_OPENMP=OFF')
    args.append('-DENABLE_WIDE_CHAR=OFF')
    return args
