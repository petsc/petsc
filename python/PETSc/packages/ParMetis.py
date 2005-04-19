#!/usr/bin/env python
from __future__ import generators
import user
import config.base
import re
import os
import PETSc.package

class Configure(PETSc.package.Package):
  def __init__(self, framework):
    PETSc.package.Package.__init__(self, framework)
    self.mpi          = self.framework.require('PETSc.packages.MPI',self)
    self.blasLapack   = self.framework.require('PETSc.packages.BlasLapack',self)
    self.download     = ['bk://parmetis.bkbits.net/ParMetis-dev','ftp://ftp.mcs.anl.gov/pub/petsc/externalpackages/parmetis.tar.gz']
    self.deps         = [self.mpi,self.blasLapack]
    self.functions    = ['ParMETIS_V3_PartKway']
    self.includes     = ['parmetis.h']
    self.liblist      = [['libparmetis.a','libmetis.a']]
    self.downloadname = 'parmetis'
    return

  def Install(self):
    import sys
    parmetisDir = self.getDir()
    installDir = os.path.join(parmetisDir, self.arch.arch)
    if not os.path.isdir(installDir):
      os.mkdir(installDir)
    # Configure and Build ParMetis
    self.framework.pushLanguage('C')
    args = ['--prefix='+installDir, '--with-cc="'+self.framework.getCompiler()+' '+self.framework.getCompilerFlags()+'"', '-PETSC_DIR='+self.arch.dir]
    self.framework.popLanguage()
    if not 'FC' in self.framework.argDB:
      args.append('--with-fc=0')
    if not self.framework.argDB['with-shared']:
      args.append('--with-shared=0')
    if not 'with-mpi-dir' in self.framework.argDB:
      args.extend(['--with-mpi-include='+self.mpi.include[0], '--with-mpi-lib='+str(self.mpi.lib).replace(' ','').replace("'","")])
    if self.framework.argDB['with-mpi-shared']:
      args.append('--with-mpi-shared')
    args.append('--ignoreCompileOutput')
    args.extend(filter(lambda a: a.find('configModules') < 0, sys.argv[1:]))
    argsStr = ' '.join(args)
    try:
      fd         = file(os.path.join(installDir,'config.args'))
      oldArgsStr = fd.readline()
      fd.close()
    except:
      oldArgsStr = ''
    if not oldArgsStr == argsStr:
      self.framework.log.write('Have to rebuild ParMetis oldargs = '+oldArgsStr+'\n new args = '+argsStr+'\n')
      self.logPrintBox('Configuring and compiling ParMetis; this may take several minutes')
      try:
        import logging
        # Split Graphs into its own repository
        oldDir = os.getcwd()
        os.chdir(parmetisDir)
        oldLog = logging.Logger.defaultLog
        logging.Logger.defaultLog = file(os.path.join(parmetisDir, 'build.log'), 'w')
        if os.path.exists('RDict.db'):
          os.remove('RDict.db')
        if os.path.exists('bsSource.db'):
          os.remove('bsSource.db')
        make = self.getModule(parmetisDir, 'make').Make(clArgs = [arg.replace('"', '') for arg in args])
        make.prefix = installDir
        make.run()
        logging.Logger.defaultLog = oldLog
        os.chdir(oldDir)
      except RuntimeError, e:
        raise RuntimeError('Error running configure on ParMetis: '+str(e))
      fd = file(os.path.join(installDir,'config.args'), 'w')
      fd.write(argsStr)
      fd.close()
      self.framework.actions.addArgument('ParMetis', 'Install', 'Installed ParMetis into '+installDir)
    return parmetisDir

if __name__ == '__main__':
  import config.framework
  import sys
  framework = config.framework.Framework(sys.argv[1:])
  framework.setup()
  framework.addChild(Configure(framework))
  framework.configure()
  framework.dumpSubstitutions()
