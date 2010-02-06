import PETSc.package
import os

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.download     = ['http://ftp.mcs.anl.gov/pub/petsc/externalpackages/hdf5-1.8.4.tar.gz']
    self.functions = ['H5T_init']
    self.includes  = ['hdf5.h']
    self.liblist   = [['libhdf5.a']]
    self.needsMath = 1
    self.extraLib  = ['libz.a']
    self.complex   = 1
    self.requires32bitint = 0;    
    return

  def setupDependencies(self, framework):
    PETSc.package.NewPackage.setupDependencies(self, framework)
    self.deps = [self.mpi]  
    return

  def Install(self):
    import os

    args = []
    self.framework.pushLanguage('C')
    args.append('--prefix='+self.installDir)
    args.append('CC="'+self.framework.getCompiler()+'"')
    args.append('CFLAGS="'+self.framework.getCompilerFlags()+'"')
    self.framework.popLanguage()
    args.append('--enable-parallel')
    if hasattr(self.compilers, 'FC'):
      self.setCompilers.pushLanguage('FC')
      args.append('--enable-fortran')
      args.append('FC="'+self.setCompilers.getCompiler()+'"')
      args.append('F9X="'+self.setCompilers.getCompiler()+'"')
      args.append('F90="'+self.setCompilers.getCompiler()+'"')      
      self.setCompilers.popLanguage()

    args = ' '.join(args)
    fd = file(os.path.join(self.packageDir,'hdf5'), 'w')
    fd.write(args)
    fd.close()

    if self.installNeeded('hdf5'):
      try:
        self.logPrintBox('Configuring HDF5; this may take several minutes')
        output1,err1,ret1  = PETSc.package.NewPackage.executeShellCommand('cd '+self.packageDir+';./configure '+args, timeout=900, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running configure on HDF5: '+str(e))
      try:
        self.logPrintBox('Compiling HDF5; this may take several minutes')
        output2,err2,ret2  = PETSc.package.NewPackage.executeShellCommand('cd '+self.packageDir+'; make ; make install', timeout=2500, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running make on HDF5: '+str(e))
      self.postInstall(output1+err1+output2+err2,'hdf5')
    return self.installDir

  def configureLibrary(self):
    if hasattr(self.compilers, 'FC'):
      self.liblist   = [['libhdf5_fortran.a', 'libhdf5.a']]
    PETSc.package.NewPackage.configureLibrary(self)
    if hasattr(self.compilers, 'FC'):
      # hdf5 puts its modules into the lib directory so add that directory to the include directories to search
      # there is no correct way to determine the location of the library directory, this is a hack that will usually work

      # should check that modules exist and work properly
      if 'with-hdf5-dir' in self.framework.argDB:
        libDir = self.framework.argDB['with-'+self.package+'-dir']
        libDir = os.path.join(libDir,'lib')        
        self.include.append(libDir)
      elif 'with-hdf5-lib' in self.framework.argDB:
        self.include.append(os.path.dirname(self.framework.argDB['with-hdf5-lib'][0]))
      elif self.framework.argDB['download-hdf5']:
        libDir = self.installDir
        libDir = os.path.join(libDir,'lib')        
        self.include.append(libDir)
      else:
        self.log('Cannot determine HDF5 library directory therefor skipping module include for HDF5')
    if self.libraries.check(self.dlib, 'H5Pset_fapl_mpio'):
      self.addDefine('HAVE_H5PSET_FAPL_MPIO', 1)
