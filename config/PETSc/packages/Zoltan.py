import PETSc.package

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.download  = ['http://ftp.mcs.anl.gov/pub/petsc/externalpackages/zoltan_distrib.tar.gz']
    self.functions = ['Zoltan_LB_Partition']
    self.includes  = ['zoltan.h'] 
    self.liblist   = [['libzoltan.a']] 
    self.license   = 'http://www.cs.sandia.gov/Zoltan/Zoltan.html'
    return

  def setupDependencies(self, framework):
    PETSc.package.NewPackage.setupDependencies(self, framework)
    self.x        = framework.require('PETSc.packages.X',self)
    self.parmetis = framework.require('PETSc.packages.parmetis',self)
    self.deps = [self.x, self.mpi, self.parmetis]
    return
          
  def Install(self):
    import os
    self.framework.pushLanguage('C')
    ccompiler=self.framework.getCompiler()
    args = ['ZOLTAN_ARCH="'+self.arch+'"']
    args.append('CC="'+self.framework.getCompiler()+'"')
    args.append('CFLAGS="'+self.framework.getCompilerFlags()+'"')
    self.framework.popLanguage()
    if hasattr(self.compilers, 'CXX'):
      self.framework.pushLanguage('Cxx')
      args.append('CPPC="'+self.framework.getCompiler()+'"')
      self.framework.popLanguage()
    args.append('AR="'+self.compilers.AR+' '+self.compilers.AR_FLAGS+'"')
    args.append('RANLIB="'+self.compilers.RANLIB+'"')
    if self.x.found:
      args.append('X_LIBS="'+str(self.x.lib)+'"')
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

    fd = file(os.path.join(self.packageDir, 'Zoltanconfig'),'w')
    fd.write(args)
    fd.close()

    if self.installNeeded('Zoltanconfig'):
      fd = file(os.path.join(self.packageDir, 'Utilities', 'Config', 'Config.'+self.arch), 'w')
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
        output1,err1,ret1  = PETSc.package.NewPackage.executeShellCommand('rm -f '+self.installDir+'lib/libzoltan*', timeout=2500, log = self.framework.log)
        output2,err2,ret2  = PETSc.package.NewPackage.executeShellCommand('cd '+self.packageDir+' && make clean && make '+args+' zoltan', timeout=2500, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running make on ZOLTAN: '+str(e))

      output3,err3,ret3  = PETSc.package.NewPackage.executeShellCommand('mv -f '+os.path.join(self.packageDir, 'Obj_'+self.arch)+'/lib* '+os.path.join(self.installDir, 'lib'))
      output4,err4,ret4  = PETSc.package.NewPackage.executeShellCommand('cp -f '+os.path.join(self.packageDir, 'include')+'/*.h '+os.path.join(self.installDir, 'include'))
      self.postInstall(output1+err1+output2+err2+output3+err3+output4+err4,'Zoltanconfig')
    return self.installDir
