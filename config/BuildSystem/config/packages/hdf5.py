import config.package
import os

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.download     = ['http://www.hdfgroup.org/ftp/HDF5/current/src/hdf5-1.8.10-patch1.tar.gz',
                         'http://ftp.mcs.anl.gov/pub/petsc/externalpackages/hdf5-1.8.10-patch1.tar.gz']
    self.functions = ['H5T_init']
    self.includes  = ['hdf5.h']
    self.liblist   = [['libhdf5_hl.a', 'libhdf5.a']]
    self.needsMath = 1
    self.needsCompression = 0
    self.complex   = 1
    self.worksonWindows = 1
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.deps = [self.mpi]
    return

  def generateLibList(self, framework):
    '''First try library list without compression libraries (zlib) then try with'''
    list = []
    for l in self.liblist:
      list.append(l)
    if self.libraries.compression:
      for l in self.liblist:
        list.append(l + self.libraries.compression)
    self.liblist = list
    return config.package.Package.generateLibList(self,framework)

  def Install(self):
    import os

    args = []
    self.framework.pushLanguage('C')
    args.append('--prefix='+self.installDir)
    args.append('--libdir='+os.path.join(self.installDir,self.libdir))
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
    if self.framework.argDB['with-shared-libraries']:
      args.append('--enable-shared')
    else:
      args.append('--disable-shared')

    args = ' '.join(args)
    fd = file(os.path.join(self.packageDir,'hdf5'), 'w')
    fd.write(args)
    fd.close()

    if self.installNeeded('hdf5'):
      try:
        self.logPrintBox('Configuring HDF5; this may take several minutes')
        output1,err1,ret1  = config.package.Package.executeShellCommand('cd '+self.packageDir+' && ./configure '+args, timeout=900, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running configure on HDF5: '+str(e))
      try:
        self.logPrintBox('Compiling HDF5; this may take several minutes')
        output2,err2,ret2  = config.package.Package.executeShellCommand('cd '+self.packageDir+' && '+self.make.make+' clean && '+self.make.make_jnp+' && '+self.make.make+' install', timeout=2500, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running make on HDF5: '+str(e))
      self.postInstall(output1+err1+output2+err2,'hdf5')
    return self.installDir

  def configureLibrary(self):
    if hasattr(self.compilers, 'FC'):
      # PETSc does not need the Fortran interface, but some users will call the Fortran interface
      # and expect our standard linking to be sufficient.  Thus we try to link the Fortran
      # libraries, but fall back to linking only C.
      self.liblist = [['libhdf5_fortran.a'] + libs for libs in self.liblist] + self.liblist
    config.package.Package.configureLibrary(self)
    if self.libraries.check(self.dlib, 'H5Pset_fapl_mpio'):
      self.addDefine('HAVE_H5PSET_FAPL_MPIO', 1)
    return
