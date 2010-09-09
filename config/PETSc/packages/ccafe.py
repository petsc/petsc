import PETSc.package

import os

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.version = ''
    return

  def miscSetup(self):
    ccafe_bin_dir = os.path.join(self.framework.argDB['with-ccafe-dir'], 'bin')
    self.getExecutable('ccafe-config', path = ccafe_bin_dir, getFullPath=1, resultName = 'ccafe_config')
    if not hasattr(self,'ccafe_config'):
      raise RuntimeError('Cannot locate ccafe-config executable in the location specified with --with-ccafe-dir')
    
    try:
      self.version  = PETSc.package.NewPackage.executeShellCommand(self.ccafe_config + ' --var CCAFE_VERSION')[0].rstrip()
    except RuntimeError, e:
      raise RuntimeError('Error when attempting to determine ccafe version')
    
    #self.liblist  = [['libccaffeine_'+self.version.replace('.','_')+'.a']]
    try:
      self.ccaspec_config = PETSc.package.NewPackage.executeShellCommand(self.ccafe_config + ' --var CCAFE_CCA_SPEC_BABEL_CONFIG')[0].rstrip()
      if not os.path.exists(self.ccaspec_config):
        raise RuntimeError('Cannot locate cca-spec-babel-config executable')
      self.specversion = PETSc.package.NewPackage.executeShellCommand(self.ccaspec_config + ' --var CCASPEC_VERSION')[0].rstrip()
    except RuntimeError, e:
      raise RuntimeError('Error when attempting to determine ccafe cca-spec-babel version')

    self.specpkg = PETSc.package.NewPackage.executeShellCommand(self.ccaspec_config + ' --var CCASPEC_PKG_NAME')[0].rstrip()
    
    self.babel = PETSc.package.NewPackage.executeShellCommand(self.ccaspec_config + ' --var CCASPEC_BABEL_BABEL')[0].rstrip()
    if not self.getExecutable(self.babel, resultName = 'babel'):
      raise RuntimeError('Located Babel library and include file but could not find babel executable')
    if not self.getExecutable(self.babel + '-config', resultName = 'babel_config'):
      raise RuntimeError('Located Babel library, includes, and babel executable but could not find babel-config executable')

    ccafePkgName = PETSc.package.NewPackage.executeShellCommand(self.ccafe_config + ' --var CCAFE_PKG_NAME')[0].rstrip()
    self.includes = [ccafePkgName + '/cmd/Cmd.h']

    dir = PETSc.package.NewPackage.executeShellCommand(self.babel_config + ' --query-var=prefix')[0].rstrip()
    if os.path.isdir(dir):
      self.framework.argDB['with-babel-dir'] = dir
    return

  def configureLibrary(self):
    '''Calls the regular package configureLibrary and then does an additional test needed by babel'''
    '''Normally you do not need to provide this method'''

    self.miscSetup()
    self.babelpackage      = self.framework.require('PETSc.packages.babel', self)
    
    PETSc.package.NewPackage.configureLibrary(self)

    self.addMakeMacro('CCAFE_HOME',self.framework.argDB['with-ccafe-dir'])
    self.addMakeMacro('CCAFE_CONFIG', self.ccafe_config)
    self.addMakeMacro('CCAFE_VERSION', self.version)
    self.addMakeMacro('CCASPEC_CONFIG', self.ccaspec_config)
    self.addMakeMacro('CCASPEC_VERSION', PETSc.package.NewPackage.executeShellCommand(self.ccaspec_config + ' --var CCASPEC_VERSION')[0].rstrip())
    self.addMakeMacro('CCASPEC_VARS', PETSc.package.NewPackage.executeShellCommand(self.ccaspec_config + ' --var CCASPEC_MAKEINCL')[0].rstrip())
    self.addMakeMacro('CCASPEC_BABEL_BABEL',  self.babel)
    self.addMakeMacro('CCASPEC_BABEL_VERSION',  PETSc.package.NewPackage.executeShellCommand(self.ccaspec_config + ' --var CCASPEC_BABEL_VERSION')[0].rstrip())
    self.addMakeMacro('BABEL_CONFIG', self.babel_config)
    self.addMakeMacro('CCA_REPO','${CCAFE_HOME}/share/' + self.specpkg + '/xml')
    self.addMakeMacro('HAVE_CCA','-DHAVE_CCA')
    return
