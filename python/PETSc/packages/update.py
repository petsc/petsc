from __future__ import generators
import config.base
import os
import re

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = 'PETSC'
    self.substPrefix  = 'PETSC'
    self.updated      = 0
    self.strmsg       = ''
    self.hasdatafiles = 0
    self.arch         = self.framework.require('PETSc.packages.arch', self)
    return

  def __str__(self):
    return self.strmsg
     
  def setupHelp(self, help):
    import nargs
    help.addArgument('PETSc', '-with-default-arch=<bool>',                nargs.ArgBool(None, 1, 'Allow using the last configured arch without setting PETSC_ARCH'))
    help.addArgument('PETSc', '-with-default-language=<c,c++,complex,0>', nargs.Arg(None, 'c', 'Specifiy default language of libraries. 0 indicates no default'))
    help.addArgument('PETSc', '-with-default-optimization=<g,O,0>',       nargs.Arg(None, 'g', 'Specifiy default optimization of libraries. 0 indicates no default'))
    help.addArgument('PETSc', '-DATAFILESPATH=directory',                 nargs.Arg(None, None, 'Specifiy location of PETSc datafiles, e.g. test matrices'))    
    return

  def configureDirectories(self):
    '''Verifies that PETSC_DIR is acceptable'''
    if not os.path.samefile(self.arch.dir, os.getcwd()):
      raise RuntimeError('  Wrong PETSC_DIR option specified: '+ self.framework.argDB['PETSC_DIR'] + '\n  Configure invoked in: '+ os.path.realpath(os.getcwd()))
    if not os.path.exists(os.path.join(self.arch.dir, 'include', 'petscversion.h')):
      raise RuntimeError('Invalid PETSc directory '+str(self.arch.dir)+' it may not exist?')
    return

  def configureArchitecture(self):
    '''Verify that PETSC_ARCH is acceptable and setup a default architecture'''
    # Check if PETSC_ARCH is a built-in arch
    if os.path.isdir(os.path.join('bmake', self.arch.arch)) and not os.path.isfile(os.path.join('bmake', self.arch.arch, 'configure.py')):
      dirs   = os.listdir('bmake')
      arches = ''
      for d in dirs:
        if os.path.isdir(os.path.join('bmake', d)) and not os.path.isfile(os.path.join('bmake', d, 'configure.py')):
          arches = arches + ' '+d
      raise RuntimeError('The selected PETSC_ARCH is not allowed with config/configure.py\nbecause it clashes with a built-in PETSC_ARCH, rerun config/configure.py with -PETSC_ARCH=somethingelse;\n   DO NOT USE the following names:'+arches)
    # if PETSC_ARCH is not set use one last created with configure
    if self.framework.argDB['with-default-arch']:
      fd = file(os.path.join('bmake', 'variables'), 'w')
      fd.write('PETSC_ARCH='+self.arch.arch+'\n')
      fd.write('include ${PETSC_DIR}/bmake/'+self.arch.arch+'/variables\n')
      fd.close()
      self.framework.actions.addArgument('PETSc', 'Build', 'Set default architecture to '+self.arch.arch+' in bmake/variables')
    else:
      os.unlink(os.path.join('bmake', 'variables'))
    return

  def configureOptimization(self):
    '''Allow a default optimization level and language'''
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
      fd = file(os.path.join('bmake', 'common', 'bopt_'), 'w')
      fd.write('BOPT='+bopt+'\n')
      fd.write('include ${PETSC_DIR}/bmake/common/bopt_'+bopt+'\n')
      fd.close()
      self.addSubstitution('BOPT', bopt)
      self.framework.actions.addArgument('PETSc', 'Build', 'Set default optimization to '+bopt+' in bmake/common/bopt_')
    return

  def datafilespath(self):
    '''Checks what DATAFILESPATH should be'''
    datafilespath = None
    if self.framework.argDB.has_key('DATAFILESPATH'):
      if os.path.isdir(self.framework.argDB['DATAFILESPATH']) & os.path.isdir(os.path.join(self.framework.argDB['DATAFILESPATH'], 'matrices')):
        datafilespath = self.framework.argDB['DATAFILESPATH']
      else:
        raise RuntimeError('Path given with option -DATAFILES='+self.framework.argDB['DATAFILESPATH']+' is not a valid datafiles directory')
    elif os.path.isdir(os.path.join('/home','petsc','datafiles')) & os.path.isdir(os.path.join('/home','petsc','datafiles','matrices')):
      datafilespath = os.path.join('/home','petsc','datafiles')
    elif os.path.isdir(os.path.join(self.arch.dir, '..', 'datafiles')) &  os.path.isdir(os.path.join(self.arch.dir, '..', 'datafiles', 'matrices')):
      datafilespath = os.path.join(self.arch.dir, '..', 'datafiles')
    if datafilespath:
      self.framework.addSubstitution('SET_DATAFILESPATH', 'DATAFILESPATH ='+datafilespath)
      self.hasdatafiles = 1
    else:
      self.framework.addSubstitution('SET_DATAFILESPATH', '')
    return

  def configure(self):
    self.executeTest(self.configureDirectories)
    self.executeTest(self.configureArchitecture)
    self.executeTest(self.configureOptimization)
    self.executeTest(self.datafilespath)
    return
