import config.package

class Configure(config.package.MesonPackage):
  def __init__(self, framework):
    config.package.MesonPackage.__init__(self, framework)
    self.version          = '2.7.1'
    self.gitcommit        = 'v'+self.version
    self.download         = ['git://https://github.com/ralna/CUTEst.git', 'https://github.com/ralna/CUTEst/archive/refs/tags/'+self.gitcommit+'.tar.gz']
    self.downloaddirnames = ['CUTEst']
    self.includes         = ['cutest.h']
    self.buildLanguages   = ['C', 'FC']
    self.precisions       = ['single', 'double', '__float128']
    self.complex          = 0
    self.linkedbypetsc    = 0

  def formMesonConfigureArgs(self):
    args = config.package.MesonPackage.formMesonConfigureArgs(self)
    args.extend(['-Ddefault_library=shared', '-Dtests=false', '-Dmodules=false', '-Dquadruple=true', '-Dint64=false'])
    return args

  def configureLibrary(self):
    precision, suffix = {'single': ('single', '_s'), 'double': ('double', ''), '__float128': ('quadruple', '_q')}[self.defaultPrecision]
    self.functions = [name+suffix+'_' for name in ['cutest_usetup', 'cutest_cint_uofg', 'cutest_cint_uhprod', 'cutest_load_routines', 'cutest_unload_routines']]
    self.liblist = [['libcutest_'+precision+'.a']]
    config.package.MesonPackage.configureLibrary(self)
