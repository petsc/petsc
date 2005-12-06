#!/usr/bin/env python
from __future__ import generators
import user
import config.base
import os
import PETSc.package

class Configure(PETSc.package.Package):
  def __init__(self, framework):
    PETSc.package.Package.__init__(self, framework)
    self.download  = ['ftp://ftp.mcs.anl.gov/pub/mpi/mpe/mpe2.tar.gz']
    self.functions = ['MPE_Log_event']
    self.includes  = ['mpe.h']
    self.liblist   = [['libmpe.a']]
    return

  def setupDependencies(self, framework):
    PETSc.package.Package.setupDependencies(self, framework)
    self.mpi        = framework.require('PETSc.packages.MPI',self)
    self.deps       = [self.mpi]
    return
          
  def Install(self):
    # Get the MPE directories
    self.downloadname = 'mpe2'
    mpeDir = self.getDir()
    installDir  = os.path.join(mpeDir, self.arch.arch)
        
    # Configure MPE 
    args = ['--prefix='+installDir]
    
    self.framework.pushLanguage('C')
    args.append('CFLAGS="'+self.framework.getCompilerFlags()+'"')
    args.append('CC="'+self.framework.getCompiler()+'"')
    self.framework.popLanguage()

    args.append('--disable-f77')

    if self.mpi.include and not self.mpi.include == ['']:
      args.append('MPI_INC="-I'+self.mpi.include[0]+'"')

    if self.mpi.lib:
      libdir = os.path.dirname(self.mpi.lib[0])
      libs = []
      for l in self.mpi.lib:
        ll = os.path.basename(l)
        libs.append('-l'+ll[3:-2])
      libs = ' '.join(libs) # '-lmpich -lpmpich'
      args.append('MPI_LIBS="'+'-L'+libdir+' '+libs+'"')

    args = ' '.join(args)
    
    try:
      fd      = file(os.path.join(installDir,'config.args'))
      oldargs = fd.readline()
      fd.close()
    except:
      oldargs = ''
    if not oldargs == args:
      self.framework.log.write('Have to rebuild MPE oldargs = '+oldargs+'\n new args ='+args+'\n')
      try:
        self.logPrintBox('Configuring mpe; this may take several minutes')
        output  = config.base.Configure.executeShellCommand('cd '+mpeDir+';./configure '+args, timeout=2000, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running configure on MPE: '+str(e))
      # Build MPE
      try:
        self.logPrintBox('Compiling mpe; this may take several minutes')
        output  = config.base.Configure.executeShellCommand('cd '+mpeDir+';make clean; make; make install', timeout=2500, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running make on MPE: '+str(e))
      if not os.path.isdir(os.path.join(installDir,'lib')):
        self.framework.log.write('Error running make on MPE   ******(libraries not installed)*******\n')
        self.framework.log.write('********Output of running make on MPE follows *******\n')        
        self.framework.log.write(output)
        self.framework.log.write('********End of Output of running make on MPE *******\n')
        raise RuntimeError('Error running make on MPE, libraries not installed')
      
      fd = file(os.path.join(installDir,'config.args'), 'w')
      fd.write(args)
      fd.close()

      self.framework.actions.addArgument(self.PACKAGE, 'Install', 'Installed MPE into '+installDir)
    return self.getDir()
  
if __name__ == '__main__':
  import config.framework
  import sys
  framework = config.framework.Framework(sys.argv[1:])
  framework.setup()
  framework.addChild(Configure(framework))
  framework.configure()
  framework.dumpSubstitutions()
