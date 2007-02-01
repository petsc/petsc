
#!/usr/bin/env python
from __future__ import generators
import user
import config.base
import os
import PETSc.package

class Configure(PETSc.package.Package):
  def __init__(self, framework):
    PETSc.package.Package.__init__(self, framework)
    self.download  = ['ftp://ftp.mcs.anl.gov/pub/petsc/externalpackages/zoltan_distrib.tar.gz']
    self.functions = ['Zoltan_LB_Partition']
    self.includes  = ['dr_loadbal_const.h'] 
    self.liblist   = [['libzoltan.a']] 
    self.license   = 'http://www.cs.sandia.gov/Zoltan/Zoltan.html'
    return

  def setupDependencies(self, framework):
    PETSc.package.Package.setupDependencies(self, framework)
    self.mpi  = framework.require('config.packages.MPI',self)
    self.deps = [self.mpi]
    return
          
  def Install(self):
    # Get the ZOLTAN directories
    zoltanDir  = self.getDir()
    installDir = os.path.join(zoltanDir, self.arch.arch)
    
    # Configure ZOLTAN 
    self.framework.pushLanguage('C')
    ccompiler=self.framework.getCompiler()
    args = ['--prefix='+installDir, 'CC="'+self.framework.getCompiler()+'"']
    args.append('--with-cflags="'+self.framework.getCompilerFlags()+'"')
    self.framework.popLanguage()
    if hasattr(self.compilers, 'CXX'):
      self.framework.pushLanguage('Cxx')
      args.append('CXX="'+self.framework.getCompiler()+'"')
      args.append('--with-cppflags="'+self.framework.getCompilerFlags()+'"')
      self.framework.popLanguage()
    
    # use --with-mpi-root if we know it works
    if self.mpi.directory and (os.path.realpath(ccompiler)).find(os.path.realpath(self.mpi.directory)) >=0:
      self.framework.log.write('Zoltan configure: using --with-mpi-root='+self.mpi.directory+'\n')
      args.append('--with-mpi-root="'+self.mpi.directory+'"')
    # else provide everything!
    else:
      #print a message if the previous check failed
      if self.mpi.directory:
        self.framework.log.write('Zoltan configure: --with-mpi-dir specified - but could not use it\n')
        self.framework.log.write(str(os.path.realpath(ccompiler))+' '+str(os.path.realpath(self.mpi.directory))+'\n')
        
      args.append('--without-mpicc')  
      if self.mpi.include:
        args.append('--with-mpi-incdir="'+self.mpi.include[0]+'"')
      else: 
        args.append('--with-mpi-incdir="/usr/include"')  # dummy case

      if self.mpi.lib:
        args.append('--with-mpi-libdir="'+os.path.dirname(self.mpi.lib[0])+'"')
        libs = []
        for l in self.mpi.lib:
          ll = os.path.basename(l)
          libs.append(ll[3:-2])
        libs = '-l' + ' -l'.join(libs)
        args.append('--with-mpi-libs="'+libs+'"')
      else:
        args.append('--with-mpi-libdir="/usr/lib"')  # dummy case
        args.append('--with-mpi-libs="-lc"')
     
    args = ' '.join(args)
    try:
      fd      = file(os.path.join(installDir,'config.args'))
      oldargs = fd.readline()
      fd.close()
    except:
      oldargs = ''
    if not oldargs == args:
      self.framework.log.write('Have to rebuild ZOLTAN oldargs = '+oldargs+'\n new args ='+args+'\n')
      try:
        self.logPrintBox('Configuring zoltan; this may take several minutes')
        output  = config.base.Configure.executeShellCommand('cd '+zoltanDir+'; ./configure '+args, timeout=900, log = self.framework.log)[0]

      except RuntimeError, e:
        raise RuntimeError('Error running configure on ZOLTAN: '+str(e))
      # Build ZOLTAN
      try:
        self.logPrintBox('Compiling zoltan; this may take several minutes')
        output  = config.base.Configure.executeShellCommand('cd '+zoltanDir+'; make; make install; make clean', timeout=2500, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running make on ZOLTAN: '+str(e))
      if not os.path.isdir(os.path.join(installDir,'lib')):
        self.framework.log.write('Error running make on ZOLTAN   ******(libraries not installed)*******\n')
        self.framework.log.write('********Output of running make on ZOLTAN follows *******\n')        
        self.framework.log.write(output)
        self.framework.log.write('********End of Output of running make on ZOLTAN *******\n')
        raise RuntimeError('Error running make on ZOLTAN, libraries not installed')
      
      fd = file(os.path.join(installDir,'config.args'), 'w')
      fd.write(args)
      fd.close()

      self.framework.actions.addArgument(self.PACKAGE, 'Install', 'Installed ZOLTAN into '+installDir)
    return self.getDir()
  
if __name__ == '__main__':
  import config.framework
  import sys
  framework = config.framework.Framework(sys.argv[1:])
  framework.setup()
  framework.addChild(Configure(framework))
  framework.configure()
  framework.dumpSubstitutions()
