
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
    self.includes  = ['zoltan.h'] 
    self.liblist   = [['libzoltan.a']] 
    self.license   = 'http://www.cs.sandia.gov/Zoltan/Zoltan.html'
    return

  def setupDependencies(self, framework):
    PETSc.package.Package.setupDependencies(self, framework)
    self.x11      = framework.require('PETSc.packages.X11',self)
    self.mpi      = framework.require('config.packages.MPI',self)
    self.parmetis = framework.require('PETSc.packages.ParMetis',self)
    self.deps = [self.x11, self.mpi, self.parmetis]
    return
          
  def Install(self):
    # Get the ZOLTAN directories
    zoltanDir  = self.getDir()
    installDir = os.path.join(zoltanDir, self.arch.arch)
    # Build ZOLTAN 
    self.framework.pushLanguage('C')
    ccompiler=self.framework.getCompiler()
    args = ['ZOLTAN_ARCH="'+self.arch.arch+'"']
    args.append('CC="'+self.framework.getCompiler()+'"')
    args.append('CFLAGS="'+self.framework.getCompilerFlags()+'"')
    self.framework.popLanguage()
    if hasattr(self.compilers, 'CXX'):
      self.framework.pushLanguage('Cxx')
      args.append('CPPC="'+self.framework.getCompiler()+'"')
      self.framework.popLanguage()
    args.append('AR="'+self.compilers.AR+' '+self.compilers.AR_FLAGS+'"')
    args.append('RANLIB="'+self.compilers.RANLIB+'"')
    if self.x11.found:
      args.append('X_LIBS="'+str(self.x11.lib)+'"')
    if self.mpi.found:
      if self.mpi.include:
        args.append('MPI_INCPATH="'+' '.join([self.headers.getIncludeArgument(inc) for inc in self.mpi.include])+'"')
      if self.mpi.lib:
        args.append('MPI_LIB="'+' '.join([self.libraries.getLibArgument(lib) for lib in self.mpi.lib])+'"')
    if self.parmetis.found:
      if self.parmetis.include:
        args.append('PARMETIS_INCPATH="'+' '.join([self.headers.getIncludeArgument(inc) for inc in self.parmetis.include])+'"')
      if self.parmetis.lib:
        args.append('PARMETIS_LIBPATH="'+' '.join([self.libraries.getLibArgument(lib) for lib in self.parmetis.lib])+'"')
    args = ' '.join(args)
    try:
      fd      = file(os.path.join(installDir, 'config.args'))
      oldargs = fd.readline()
      fd.close()
    except:
      oldargs = ''
    if not oldargs == args:
      self.framework.log.write('Have to rebuild ZOLTAN oldargs = '+oldargs+'\n new args ='+args+'\n')
      fd = file(os.path.join(zoltanDir, 'Utilities', 'Config', 'Config.'+self.arch.arch), 'w')
      fd.write('''
##############################################################################
#  Environment variables for compiling the Zoltan and test drivers using PETSc
##############################################################################
# The location of the VTK libraries, built with OpenGL
#   We do not do these correctly
VTK_LIBPATH = 
VTK_INCPATH = 
# The location of the GL or Mesa libraries, and the libraries
#   We do not do these correctly
GL_LIBPATH = -L/usr/lib
GL_INCPATH = -I/usr/include
GL_LIBS    = -lGL -lGLU
# Have no idea about VTK_OFFSCREEN_* and MESA_* stuff
''')
      fd.close()
      try:
        self.logPrintBox('Compiling zoltan; this may take several minutes')
        output  = config.base.Configure.executeShellCommand('cd '+zoltanDir+'; make '+args+' zoltan', timeout=2500, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running make on ZOLTAN: '+str(e))
      if not os.path.isdir(os.path.join(zoltanDir, 'Obj_'+self.arch.arch)):
        self.framework.log.write('Error running make on ZOLTAN   ******(libraries not installed)*******\n')
        self.framework.log.write('********Output of running make on ZOLTAN follows *******\n')        
        self.framework.log.write(output)
        self.framework.log.write('********End of Output of running make on ZOLTAN *******\n')
        raise RuntimeError('Error running make on ZOLTAN, libraries not installed')
      import shutil
      if os.path.isdir(os.path.join(installDir, 'lib')):
        shutil.rmtree(os.path.join(installDir, 'lib'))
      shutil.copytree(os.path.join(zoltanDir, 'Obj_'+self.arch.arch), os.path.join(installDir, 'lib'))
      if os.path.isdir(os.path.join(installDir, 'include')):
        shutil.rmtree(os.path.join(installDir, 'include'))
      shutil.copytree(os.path.join(zoltanDir, 'include'), os.path.join(installDir, 'include'))
      fd = file(os.path.join(installDir, 'config.args'), 'w')
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
