from __future__ import generators
import config.base
import os
import re

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = 'PETSC'
    self.substPrefix  = 'PETSC'
    self.updated  = 0
    self.strmsg   = ''
    return

  def __str__(self):
    return self.strmsg
     
  def configureHelp(self, help):
    import nargs
    help.addArgument('Update', '-freeform-arch',             nargs.ArgBool(None, 0, 'Allow arbitrary configure PETSC_ARCH specification'))
    help.addArgument('Update', '-with-default-arch',         nargs.ArgBool(None, 1, 'Allow using the most recently configured arch without setting PETSC_ARCH'))
    help.addArgument('Update', '-with-default-language=<c,c++,c++-complex,0(zero for no default)>', nargs.Arg(None, 'c', 'Specifiy default language of libraries'))
    help.addArgument('Update', '-with-default-optimization=<g,O,0(zero for no default)>',           nargs.Arg(None, 'g', 'Specifiy default optimization of libraries'))
    return

  def configureDirectories(self):
    '''Checks PETSC_DIR and sets if not set'''
    if not self.framework.argDB.has_key('PETSC_DIR'):
      self.framework.argDB['PETSC_DIR'] = os.getcwd()
    self.dir = self.framework.argDB['PETSC_DIR']
    # Check for version
    if not os.path.exists(os.path.join(self.dir, 'include', 'petscversion.h')):
      raise RuntimeError('Invalid PETSc directory '+str(self.dir)+' it may not exist?')
    self.addSubstitution('DIR', self.dir)
    self.addDefine('DIR', self.dir)
    return

  def configureArchitecture(self):
    '''Sets PETSC_ARCH'''
    import sys
    # Find auxilliary directory by checking for config.sub
    auxDir = None
    for dir in [os.path.abspath(os.path.join('bin', 'config')), os.path.abspath('config')] + sys.path:
      if os.path.isfile(os.path.join(dir, 'config.sub')):
        auxDir      = dir
        configSub   = os.path.join(auxDir, 'config.sub')
        configGuess = os.path.join(auxDir, 'config.guess')
        break
    if not auxDir: raise RuntimeError('Unable to locate config.sub in order to determine architecture.Your PETSc directory is incomplete.\n Get PETSc again')
    try:
      # Guess host type (should allow user to specify this
      host = config.base.Configure.executeShellCommand(self.shell+' '+configGuess, log = self.framework.log)[0]
      # Get full host description
      output = config.base.Configure.executeShellCommand(self.shell+' '+configSub+' '+host, log = self.framework.log)[0]
    except RuntimeError, e:
      raise RuntimeError('Unable to determine host type using '+configSub+': '+str(e))
    # Parse output
    m = re.match(r'^(?P<cpu>[^-]*)-(?P<vendor>[^-]*)-(?P<os>.*)$', output)
    if not m: raise RuntimeError('Unable to parse output of config.sub: '+output)
    self.framework.host_cpu    = m.group('cpu')
    self.host_vendor = m.group('vendor')
    self.host_os     = m.group('os')

##    results = self.executeShellCode(self.macroToShell(self.hostMacro))
##    self.host_cpu    = results['host_cpu']
##    self.host_vendor = results['host_vendor']
##    self.host_os     = results['host_os']

    if not self.framework.argDB.has_key('PETSC_ARCH'):
      arch = self.host_os
    else:
      arch = self.framework.argDB['PETSC_ARCH']
    if not arch.startswith(self.host_os):
      if not self.framework.argDB['freeform-arch']:
        raise RuntimeError('PETSC_ARCH ('+arch+') does not have our guess ('+self.host_os+') as a prefix!\nRun bin/petscarch --suggest and set the environment variable PETSC_ARCH to the suggested value.')
    archBase = re.sub(r'^(\w+)[-_]?.*$', r'\1', arch)
    self.framework.argDB['PETSC_ARCH']      = arch
    self.framework.argDB['PETSC_ARCH_BASE'] = archBase
    self.addArgumentSubstitution('ARCH', 'PETSC_ARCH')
    self.addDefine('ARCH', archBase)
    self.addDefine('ARCH_NAME', '"'+arch+'"')

    # if PETSC_ARCH is not set use one last created with configure
    if self.framework.argDB['with-default-arch']:
      fd = file(os.path.join('bmake', 'variables'), 'w')
      fd.write('PETSC_ARCH='+arch+'\n')
      fd.write('include ${PETSC_DIR}/bmake/'+arch+'/variables\n')
      fd.close()
    else:
      os.unlink(os.path.join('bmake', 'variables'))
    return

  def configureOptimization(self):
    '''Allow a default optimization level and language'''
    # We use the framework in order to remove the PETSC_ namespace
    self.framework.addSubstitution('CC_SHARED_OPT', '')

    # if BOPT is not set determines what libraries to use
    bopt = self.framework.argDB['with-default-optimization']
    if self.framework.argDB['with-default-language'] == '0' or self.framework.argDB['with-default-optimization'] == '0':
      fd = file(os.path.join('bmake', 'common', 'bopt_'), 'w')
      fd.write('PETSC_LANGUAGE  = CONLY\nPETSC_SCALAR    = real\nPETSC_PRECISION = double\n')
      fd.close()
    elif not ((bopt == 'O') or (bopt == 'g')):
      raise RuntimeError('Unknown option given with --with-default-optimization='+self.framework.argDB['with-default-optimization'])
    else:
      if self.framework.argDB['with-default-language'] == 'c': pass
      elif self.framework.argDB['with-default-language'] == 'c++': bopt += '_c++'
      elif self.framework.argDB['with-default-language'].find('complex') >= 0: bopt += '_complex'
      else:
        raise RuntimeError('Unknown option given with --with-default-language='+self.framework.argDB['with-default-language'])
      fd = open(os.path.join('bmake', 'common', 'bopt_'), 'w')
      fd.write('BOPT='+bopt+'\n')
      fd.write('include ${PETSC_DIR}/bmake/common/bopt_'+bopt+'\n')
      fd.close()
    return

  def configure(self):
    self.executeTest(self.configureDirectories)
    self.executeTest(self.configureArchitecture)
    self.executeTest(self.configureOptimization)
    return
