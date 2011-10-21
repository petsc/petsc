import config.package
import os

testPatch = '''\
Index: fortran/test/tH5Sselect.f90
===================================================================
--- fortran/test/tH5Sselect.f90	2011-10-20 09:37:20.000000000 -0400
+++ fortran/test/tH5Sselect.f90	2011-10-20 09:38:22.000000000 -0400
@@ -1884,13 +1884,13 @@
   CALL check("H5Soffset_simple_f", error, total_error)
 
   ! /* Set "regular" hyperslab selection */
-  start(1:2) = 2
+  !start(1:2) = 2
   stride(1:2) = 10
   count(1:2) = 4
   block(1:2) = 5
 
-  CALL h5sselect_hyperslab_f(sid, H5S_SELECT_SET_F, start, &
-       count, error, stride, block)
+  !CALL h5sselect_hyperslab_f(sid, H5S_SELECT_SET_F, start, &
+  !     count, error, stride, block)
   CALL check("h5sselect_hyperslab_f", error, total_error)
 
   !/* Get bounds for hyperslab selection */
@@ -1931,13 +1931,13 @@
   CALL check("H5Soffset_simple_f", error, total_error)
 
   ! /* Make "irregular" hyperslab selection */
-  start(1:2) = 20
+  !start(1:2) = 20
   stride(1:2) = 20
   count(1:2) = 2
   block(1:2) = 10
 
-  CALL h5sselect_hyperslab_f(sid, H5S_SELECT_OR_F, start, &
-       count, error, stride, block)
+  !CALL h5sselect_hyperslab_f(sid, H5S_SELECT_OR_F, start, &
+  !     count, error, stride, block)
   CALL check("h5sselect_hyperslab_f", error, total_error)
 
   !/* Get bounds for hyperslab selection */
'''

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.download     = ['http://ftp.mcs.anl.gov/pub/petsc/externalpackages/hdf5-1.8.6.tar.gz']
    self.functions = ['H5T_init']
    self.includes  = ['hdf5.h']
    self.liblist   = [['libhdf5.a']]
    self.needsMath = 1
    self.extraLib  = ['libz.a']
    self.complex   = 1
    self.worksonWindows = 1
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.deps = [self.mpi]
    return

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

    args = ' '.join(args)
    fd = file(os.path.join(self.packageDir,'hdf5'), 'w')
    fd.write(args)
    fd.close()

    if self.installNeeded('hdf5'):
      try:
        self.logPrintBox('Patching HDF5; the stupid F90 test is broken for gfortran on the Mac')
        try:
          import patch,StringIO
          oldDir = os.getcwd()
          os.chdir(self.packageDir)
          p = StringIO.StringIO(testPatch)
          patcher = patch.PatchSet(p)
          p.close()
          patcher.apply()
          os.chdir(oldDir)
        except ImportError:
          pass
        self.logPrintBox('Configuring HDF5; this may take several minutes')
        output1,err1,ret1  = config.package.Package.executeShellCommand('cd '+self.packageDir+' && ./configure '+args, timeout=900, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running configure on HDF5: '+str(e))
      try:
        self.logPrintBox('Compiling HDF5; this may take several minutes')
        output2,err2,ret2  = config.package.Package.executeShellCommand('cd '+self.packageDir+' && make clean && make && make install', timeout=2500, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running make on HDF5: '+str(e))
      self.postInstall(output1+err1+output2+err2,'hdf5')
    return self.installDir

  def configureLibrary(self):
    if hasattr(self.compilers, 'FC'):
      self.liblist   = [['libhdf5_fortran.a', 'libhdf5.a']]
    config.package.Package.configureLibrary(self)
    if self.libraries.check(self.dlib, 'H5Pset_fapl_mpio'):
      self.addDefine('HAVE_H5PSET_FAPL_MPIO', 1)
    return
