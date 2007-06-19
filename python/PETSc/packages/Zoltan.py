
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

    fd = file(os.path.join(zoltanDir, 'Zoltanconfig'),'w')
    fd.write(args)
    fd.close()

    if not os.path.isfile(self.confDir+'Zoltan') or not (self.getChecksum(self.confDir+'Zoltan') == self.getChecksum(zoltanDir+'Zoltanconfig')):  
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
        output  = config.base.Configure.executeShellCommand('rm -f '+self.installDir+'lib/libzoltan*', timeout=2500, log = self.framework.log)[0]
        output  = config.base.Configure.executeShellCommand('cd '+zoltanDir+'; make clean; make '+args+' zoltan', timeout=2500, log = self.framework.log)[0]        
      except RuntimeError, e:
        raise RuntimeError('Error running make on ZOLTAN: '+str(e))

      output  = config.base.Configure.executeShellCommand('mv -f '+os.path.join(zoltanDir, 'Obj_'+self.arch.arch)+'/* '+os.path.join(self.installDir, 'lib'))
      output  = config.base.Configure.executeShellCommand('cp -f '+os.path.join(zoltanDir, 'include')+'/* '+os.path.join(self.installDir, 'include'))
      self.checkInstall(output,'Zoltanconfig')
    return self.installDir
  
if __name__ == '__main__':
  import config.framework
  import sys
  framework = config.framework.Framework(sys.argv[1:])
  framework.setup()
  framework.addChild(Configure(framework))
  framework.configure()
  framework.dumpSubstitutions()
