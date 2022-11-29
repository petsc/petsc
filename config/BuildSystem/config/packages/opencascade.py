import config.package

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.gitcommit       = 'f136d2a3ac89b3203affef1f04da6fde9d60bf7e'
    self.download        = ['git://https://github.com/bldenton/oce.git']
    self.version         = '7.5.0'
    self.minversion      = '7.5.0'
    self.versionname     = 'OCC_VERSION_COMPLETE'
    self.requiresversion = 1
    self.functions       = []
    self.includes        = ['opencascade/Standard_Version.hxx']
    self.liblist         = [['libTKXSBase.a', 'libTKSTEPBase.a', 'libTKSTEPAttr.a', 'libTKSTEP209.a', 'libTKSTEP.a', 'libTKIGES.a', 'libTKGeomAlgo.a', 'libTKTopAlgo.a', 'libTKPrim.a', 'libTKBO.a', 'libTKBool.a', 'libTKHLR.a', 'libTKFillet.a', 'libTKOffset.a', 'libTKFeat.a', 'libTKMesh.a', 'libTKXMesh.a', 'libTKShHealing.a', 'libTKG2d.a', 'libTKG3d.a', 'libTKGeomBase.a', 'libTKBRep.a', 'libTKernel.a', 'libTKMath.a']]
    self.pkgname         = 'opencascade'
    self.buildLanguages  = ['Cxx']
    self.hastests        = 1
    return

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.compilerFlags = framework.require('config.compilerFlags', self)
    self.deps          = []
    return

  def formCMakeConfigureArgs(self):
    args = config.package.CMakePackage.formCMakeConfigureArgs(self)
    if not hasattr(self.compilers, 'CXX'):
      raise RuntimeError("%s requires a C++ compiler\n" % self.pkgname)
    return args

  def updateControlFiles(self):
    import os.path
    # The OCE build relies on files in the adm directory
    controlDir = os.path.join(self.packageDir, 'adm')
    modfile = os.path.join(controlDir, 'MODULES')
    fd = open(modfile, 'w')
    fd.write('''\
FoundationClasses TKernel TKMath
ModelingData TKG2d TKG3d TKGeomBase TKBRep
ModelingAlgorithms TKGeomAlgo TKTopAlgo TKPrim TKBO TKBool TKHLR TKFillet TKOffset TKFeat TKMesh TKXMesh TKShHealing
Visualization
ApplicationFramework
DataExchange TKXSBase TKSTEPBase TKSTEPAttr TKSTEP209 TKSTEP TKIGES
Draw
''')
    fd.close()
    toolfile = os.path.join(controlDir, 'TOOLS')
    fd = open(toolfile, 'w')
    fd.write('')
    fd.close()
    sampfile = os.path.join(controlDir, 'SAMPLES')
    fd = open(sampfile, 'w')
    fd.write('')
    fd.close()
    return
