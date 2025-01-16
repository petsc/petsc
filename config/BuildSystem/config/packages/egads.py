import config.package
import os

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.gitcommit         = 'ee0890bb2ce96bdd878fc2b628602640027eaa85'
    self.download          = ['git://https://github.com/bldenton/EGADSlite.git']
    self.functions         = ['EG_open']
    self.includes          = ['egads.h']
    self.hastests          = 1
    self.buildLanguages    = ['Cxx']
    self.hasegadslite      = 1
    self.hasegads          = 1
    self.skippackagelibincludedirs = 1
    return

  def setupDependencies(self, framework):
    config.package.GNUPackage.setupDependencies(self, framework)
    self.pthread = self.framework.require('config.packages.pthread',self)
    self.oce     = self.framework.require('config.packages.opencascade',self)
    self.deps    = [self.pthread]
    self.odeps   = [self.oce]
    return

  def setupHelp(self, help):
    config.package.GNUPackage.setupHelp(self,help)
    import nargs
    help.addArgument('EGADS', '-egads-full', nargs.ArgBool(None, 1, 'Install EGADS in addition to EGADSLite'))
    return

  def createEGADSLiteMakefile(self):
    if not self.hasegadslite: return
    makeinc = os.path.join(self.packageDir, 'make_lite.inc')
    g = open(makeinc,'w')
    g.write('''
include $(PETSC_DIR)/lib/petsc/conf/variables

EGADSFLAGS = -DLITE -IEGADSlite/include

INCDIR     = EGADSlite/include
SRCDIR     = EGADSlite/src
LIBBASE    = libegadslite
LIBNAME    = ${LIBBASE}.${AR_LIB_SUFFIX}
LIBSRC.h   = $(INCDIR)/egads_lite.h $(INCDIR)/egadsErrors.h \
             $(INCDIR)/egadsInternals_lite.h $(INCDIR)/egadsTris_lite.h $(INCDIR)/egadsTypes.h \
             $(INCDIR)/emp_lite.h $(INCDIR)/liteClasses.h $(INCDIR)/liteString.h \
             $(INCDIR)/regQuads_lite.h
LIBSRC.c   = $(SRCDIR)/liteAttrs.c $(SRCDIR)/liteBase.c $(SRCDIR)/liteGeom.c $(SRCDIR)/liteImport.c \
             $(SRCDIR)/liteMemory.c $(SRCDIR)/liteString.c $(SRCDIR)/egadsTess_lite.c $(SRCDIR)/egadsTessInp_lite.c \
             $(SRCDIR)/egadsTris_lite.c $(SRCDIR)/liteTopo.c $(SRCDIR)/egadsQuads_lite.c $(SRCDIR)/regQuads_lite.c \
             $(SRCDIR)/egadsRobust_lite.c $(SRCDIR)/emp_lite.c $(SRCDIR)/evaluate_lite.c $(SRCDIR)/rational_lite.c \
             $(SRCDIR)/limitTessBody_lite.c $(SRCDIR)/liteTest.c \
             $(SRCDIR)/retessFaces_lite.c
LIBSRC.o   = $(LIBSRC.c:%.c=%.o)

lib : $(LIBNAME) ;

$(LIBSRC.o) : $(LIBSRC.h)

define ARCHIVE_RECIPE_WIN32FE_LIB
  @$(RM) $@ $@.args
  @cygpath -w $^ > $@.args
  $(AR) $(AR_FLAGS) $@ @$@.args
  @$(RM) $@.args
endef

define ARCHIVE_RECIPE_DEFAULT
  @$(RM) $@
  $(AR) $(AR_FLAGS) $@ $^
  $(RANLIB) $@
endef

$(LIBNAME) : $(LIBSRC.o)
	$(if $(findstring win32fe lib,$(AR)),$(ARCHIVE_RECIPE_WIN32FE_LIB),$(ARCHIVE_RECIPE_DEFAULT))

COMPILE.c   = $(CC)  $(CC_FLAGS)  $(FLAGS)    $(CPPFLAGS)    $(EGADSFLAGS) $(TARGET_ARCH) -c
COMPILE.cpp = $(CXX) $(CXX_FLAGS) $(CXXFLAGS) $(CXXCPPFLAGS) $(EGADSFLAGS) $(TARGET_ARCH) -c

# This is unusual; usually prefix would default to /usr/local
prefix ?= $(PETSC_DIR)/$(PETSC_ARCH)
libdir = $(prefix)/lib
includedir = $(prefix)/include
INSTALL = install
INSTALL_DATA = $(INSTALL) -m644
MKDIR_P = mkdir -p

install-egads: $(LIBNAME)
	$(MKDIR_P) "$(DESTDIR)$(includedir)" "$(DESTDIR)$(libdir)"
	$(INSTALL_DATA) $(LIBSRC.h) "$(DESTDIR)$(includedir)/"
	$(INSTALL_DATA) $(LIBNAME) "$(DESTDIR)$(libdir)/"

clean:
	$(RM) $(LIBNAME) $(LIBSRC.o)

.PHONY: lib clean install-egads
    ''')
    g.close()
    return

  def createEGADSMakefile(self):
    if not self.hasegads: return
    makeinc = os.path.join(self.packageDir, 'make.inc')
    g = open(makeinc,'w')
    g.write('''
include $(PETSC_DIR)/lib/petsc/conf/variables

EGADSFLAGS = -IEGADS/include -IEGADS/src -IEGADS/util -IEGADS/util/uvmap -IEGADSlite/include '''+self.headers.toStringNoDupes(self.oce.include)+'/opencascade'+
'''

INCDIR     = EGADS/include
ELIDIR     = EGADSlite/include
SRCDIR     = EGADS/src
UTLDIR     = EGADS/util
UVLDIR     = EGADS/util/uvmap
LIBBASE    = libegads
LIBNAME    = ${LIBBASE}.${AR_LIB_SUFFIX}
LIBSRC.h   = $(INCDIR)/egads.h $(INCDIR)/egads_dot.h $(INCDIR)/egadsErrors.h \
             $(INCDIR)/egadsTypes.h $(INCDIR)/emp.h $(INCDIR)/prm.h $(INCDIR)/wsserver.h $(INCDIR)/wsss.h \
             $(SRCDIR)/Surreal/SurrealD.h $(SRCDIR)/Surreal/SurrealD_Lazy.h $(SRCDIR)/Surreal/SurrealD_Trad.h \
             $(SRCDIR)/Surreal/SurrealS.h $(SRCDIR)/Surreal/SurrealS_Lazy.h $(SRCDIR)/Surreal/SurrealS_Trad.h \
             $(SRCDIR)/Surreal/always_inline.h $(UTLDIR)/regQuads.h $(SRCDIR)/BRepLib_FuseEdges.h \
             $(SRCDIR)/egadsInternals.h $(SRCDIR)/egadsStack.h $(SRCDIR)/egadsTris.h $(SRCDIR)/egadsOCC.h \
             $(SRCDIR)/egadsClasses.h $(SRCDIR)/egadsSplineFit.h $(INCDIR)/egads.inc $(INCDIR)/wsserver.inc \
             $(SRCDIR)/egadsSplineVels.h $(INCDIR)/egadsf90.inc \
             $(UVLDIR)/EG_uvmapFindUV.h $(UVLDIR)/EG_uvmapGen.h $(UVLDIR)/EG_uvmapStructFree.h \
             $(UVLDIR)/EG_uvmap_Read.h $(UVLDIR)/EG_uvmap_Write.h $(UVLDIR)/EG_uvmapTest.h \
             $(UVLDIR)/uvmap_add.h $(UVLDIR)/uvmap_bnd_adj.h $(UVLDIR)/uvmap_chk_area_uv.h \
             $(UVLDIR)/uvmap_chk_edge_ratio.h $(UVLDIR)/uvmap_cpu_message.h $(UVLDIR)/uvmap_find_uv.h \
             $(UVLDIR)/uvmap_from_egads.h $(UVLDIR)/uvmap_gen.h $(UVLDIR)/uvmap_gen_uv.h \
             $(UVLDIR)/uvmap_ibeibe.h $(UVLDIR)/uvmap_ibfibf.h $(UVLDIR)/uvmap_ibfin.h \
             $(UVLDIR)/uvmap_iccibe.h $(UVLDIR)/uvmap_iccin.h $(UVLDIR)/uvmap_idibe.h \
             $(UVLDIR)/uvmap_inibe.h $(UVLDIR)/uvmap_inl_uv_bnd.h $(UVLDIR)/uvmap_malloc.h \
             $(UVLDIR)/uvmap_mben_disc.h $(UVLDIR)/uvmap_message.h $(UVLDIR)/uvmap_norm_uv.h \
             $(UVLDIR)/uvmap_read.h $(UVLDIR)/uvmap_solve.h $(UVLDIR)/uvmap_struct.h \
             $(UVLDIR)/uvmap_struct_tasks.h $(UVLDIR)/uvmap_test.h $(UVLDIR)/uvmap_to_egads.h \
             $(UVLDIR)/uvmap_version.h $(UVLDIR)/uvmap_write.h \
             $(UVLDIR)/UVMAP_LIB.h $(UVLDIR)/UVMAP_LIB_INC.h \
             $(ELIDIR)/liteString.h $(ELIDIR)/liteClasses.h

LIBSRC.c   = $(SRCDIR)/egadsAttrs.c $(SRCDIR)/egadsBase.c $(SRCDIR)/egadsCopy.c $(SRCDIR)/egadsExport.c \
             $(SRCDIR)/egadsFit.c $(SRCDIR)/egadsGeom.c $(SRCDIR)/egadsHLevel.c $(SRCDIR)/egadsIO.c \
             $(SRCDIR)/egadsMemory.c $(SRCDIR)/egadsQuads.c $(SRCDIR)/egadsRobust.c $(SRCDIR)/egadsSBO.c \
             $(SRCDIR)/egadsSkinning.c $(SRCDIR)/egadsSolids.c $(SRCDIR)/egadsSpline.c $(SRCDIR)/egadsSplineFit.c \
             $(SRCDIR)/egadsTess.c $(SRCDIR)/egadsTessInp.c $(SRCDIR)/egadsTessSens.c $(SRCDIR)/egadsTopo.c \
             $(SRCDIR)/egadsTris.c $(UTLDIR)/emp.c $(SRCDIR)/prmCfit.c $(SRCDIR)/prmGrid.c \
             $(SRCDIR)/prmUV.c $(UTLDIR)/regQuads.c $(SRCDIR)/egadsEffect.c  $(SRCDIR)/fgadsBase.c \
             $(SRCDIR)/fgadsMemory.c $(SRCDIR)/fgadsAttrs.c $(SRCDIR)/fgadsTess.c $(SRCDIR)/fgadsHLevel.c \
             $(SRCDIR)/fgadsGeom.c $(SRCDIR)/fgadsTopo.c \
             $(UVLDIR)/main/uvmap.c $(UVLDIR)/EG_uvmapFindUV.c $(UVLDIR)/EG_uvmapGen.c \
             $(UVLDIR)/EG_uvmap_Read.c $(UVLDIR)/EG_uvmapStructFree.c $(UVLDIR)/EG_uvmapTest.c \
             $(UVLDIR)/EG_uvmap_Write.c $(UVLDIR)/uvmap_add.c $(UVLDIR)/uvmap_bnd_adj.c \
             $(UVLDIR)/uvmap_chk_area_uv.c $(UVLDIR)/uvmap_chk_area_uv.c $(UVLDIR)/uvmap_chk_edge_ratio.c \
             $(UVLDIR)/uvmap_cpu_message.c $(UVLDIR)/uvmap_find_uv.c $(UVLDIR)/uvmap_from_egads.c \
             $(UVLDIR)/uvmap_gen.c $(UVLDIR)/uvmap_gen_uv.c $(UVLDIR)/uvmap_ibeibe.c \
             $(UVLDIR)/uvmap_ibfibf.c $(UVLDIR)/uvmap_ibfin.c $(UVLDIR)/uvmap_iccibe.c \
             $(UVLDIR)/uvmap_iccin.c $(UVLDIR)/uvmap_idibe.c $(UVLDIR)/uvmap_inibe.c \
             $(UVLDIR)/uvmap_inl_uv_bnd.c $(UVLDIR)/uvmap_malloc.c $(UVLDIR)/uvmap_mben_disc.c \
             $(UVLDIR)/uvmap_message.c $(UVLDIR)/uvmap_norm_uv.c $(UVLDIR)/uvmap_read.c \
             $(UVLDIR)/uvmap_solve.c $(UVLDIR)/uvmap_struct_tasks.c $(UVLDIR)/uvmap_to_egads.c \
             $(UVLDIR)/uvmap_test.c $(UVLDIR)/uvmap_version.c $(UVLDIR)/uvmap_write.c \
             $(UTLDIR)/egadsHOtess.c $(UTLDIR)/egadsUVmap.c \
             $(UTLDIR)/extractTess.c $(UTLDIR)/limitTessBody.c $(UTLDIR)/retessFaces.c \
             $(UTLDIR)/ThreadTest.c $(UTLDIR)/triServer.c $(UTLDIR)/vHOtess.c 
LIBSRC.cpp = $(UTLDIR)/evaluate.cpp $(UTLDIR)/rational.cpp $(SRCDIR)/BRepLib_FuseEdges.cpp \
             $(UTLDIR)/SurrealD1_btest.cpp $(UTLDIR)/SurrealD4_btest.cpp $(UTLDIR)/SurrealS1_btest.cpp \
             $(UTLDIR)/SurrealS4_btest.cpp $(SRCDIR)/egadsCopy.cpp $(SRCDIR)/egadsGeom.cpp \
             $(SRCDIR)/egadsHLevel.cpp $(SRCDIR)/egadsIO.cpp $(SRCDIR)/egadsSkinning.cpp \
             $(SRCDIR)/egadsSpline.cpp $(SRCDIR)/egadsSplineFit.cpp $(SRCDIR)/egadsTessSens.cpp \
             $(SRCDIR)/egadsTopo.cpp
LIBSRC.o   = $(LIBSRC.c:%.c=%.o) $(LIBSRC.cpp:%.cpp=%.o)

lib : $(LIBNAME) ;

$(LIBSRC.o) : $(LIBSRC.h)

define ARCHIVE_RECIPE_WIN32FE_LIB
  @$(RM) $@ $@.args
  @cygpath -w $^ > $@.args
  $(AR) $(AR_FLAGS) $@ @$@.args
  @$(RM) $@.args
endef

define ARCHIVE_RECIPE_DEFAULT
  @$(RM) $@
  $(AR) $(AR_FLAGS) $@ $^
  $(RANLIB) $@
endef

$(LIBNAME) : $(LIBSRC.o)
	$(if $(findstring win32fe lib,$(AR)),$(ARCHIVE_RECIPE_WIN32FE_LIB),$(ARCHIVE_RECIPE_DEFAULT))

COMPILE.c   = $(CC)  $(CC_FLAGS)  $(FLAGS)    $(CPPFLAGS)    $(EGADSFLAGS) $(TARGET_ARCH) -c
COMPILE.cpp = $(CXX) $(CXX_FLAGS) $(CXXFLAGS) $(CXXCPPFLAGS) $(EGADSFLAGS) $(TARGET_ARCH) -c

# This is unusual; usually prefix would default to /usr/local
prefix ?= $(PETSC_DIR)/$(PETSC_ARCH)
libdir = $(prefix)/lib
includedir = $(prefix)/include
INSTALL = install
INSTALL_DATA = $(INSTALL) -m644
MKDIR_P = mkdir -p

install-egads: $(LIBNAME)
	$(MKDIR_P) "$(DESTDIR)$(includedir)" "$(DESTDIR)$(libdir)"
	$(INSTALL_DATA) $(LIBSRC.h) "$(DESTDIR)$(includedir)/"
	$(INSTALL_DATA) $(LIBNAME) "$(DESTDIR)$(libdir)/"

clean:
	$(RM) $(LIBNAME) $(LIBSRC.o)

.PHONY: lib clean install-egads
    ''')
    g.close()
    return

  # the install is delayed until postProcess() since egads install requires PETSc to have created its build/makefiles before installing
  # note that egads can (and is) built before PETSc is built.
  def Install(self):
    self.createEGADSMakefile()
    self.createEGADSLiteMakefile()
    return self.installDir

  def configureLibrary(self):
    ''' Since egads cannot be built until after PETSc configure is complete we need to just assume the downloaded library will work'''
    if 'with-egads' in self.framework.clArgDB:
      raise RuntimeError('egads does not support --with-egads; only --download-egads')
    if 'with-egads-dir' in self.framework.clArgDB:
      self.egadsDir = self.framework.argDB['with-egads-dir']
    if 'with-egads-shared' in self.framework.clArgDB:
      raise RuntimeError('egads does not support --with-egads-shared')

    self.hasegads = self.framework.argDB['egads-full']
    if self.hasegads and not self.oce.found:
      raise RuntimeError('egads requires open cascade if doing a full install\nReconfigure using --download-opencascade')

    if not hasattr(self,'egadsDir'):
      self.checkDownload()
      self.egadsDir = self.installDir
    self.include = [os.path.join(self.egadsDir,'include')]
    self.lib     = []
    if self.hasegadslite: self.lib.append(os.path.join(self.egadsDir,'lib','libegadslite.a'))
    if self.hasegads:     self.lib.append(os.path.join(self.egadsDir,'lib','libegads.a'))
    self.found   = 1
    self.dlib    = self.lib
    if not hasattr(self.framework, 'packages'):
      self.framework.packages = []
    self.framework.packages.append(self)

  def buildEGADS(self):
    self.logPrintBox('Compiling egads; this may take several minutes')
    # uses the regular PETSc library builder and then moves result
    # turn off any compiler optimizations as they may break egads
    self.pushLanguage('C')
    cflags = self.checkNoOptFlag()+' '+self.getSharedFlag(self.getCompilerFlags())+' '+self.getPointerSizeFlag(self.getCompilerFlags())+'  '+self.getWindowsNonOptFlags(self.getCompilerFlags())+' '+self.getDebugFlags(self.getCompilerFlags())
    self.popLanguage()
    output,err,ret  = config.package.GNUPackage.executeShellCommand(self.make.make+' -f make.inc PETSC_DIR=' + self.petscdir.dir + ' clean lib PCC_FLAGS="' + cflags + '"', timeout=1000, log = self.log, cwd=self.packageDir)
    self.log.write(output+err)
    self.logPrintBox('Installing egads; this may take several minutes')
    output,err,ret  = config.package.GNUPackage.executeShellCommand(self.make.make+' -f make.inc PETSC_DIR='+self.petscdir.dir+' prefix='+self.installDir+' install-egads',timeout=1000, log = self.log, cwd=self.packageDir)
    self.log.write(output+err)
    return

  def buildEGADSLite(self):
    self.logPrintBox('Compiling egads lite; this may take several minutes')
    # uses the regular PETSc library builder and then moves result
    # turn off any compiler optimizations as they may break egads
    self.pushLanguage('C')
    cflags = self.checkNoOptFlag()+' '+self.getSharedFlag(self.getCompilerFlags())+' '+self.getPointerSizeFlag(self.getCompilerFlags())+'  '+self.getWindowsNonOptFlags(self.getCompilerFlags())+' '+self.getDebugFlags(self.getCompilerFlags())
    self.popLanguage()
    output,err,ret  = config.package.GNUPackage.executeShellCommand(self.make.make+' -f make_lite.inc PETSC_DIR=' + self.petscdir.dir + ' clean lib PCC_FLAGS="' + cflags + '"', timeout=1000, log = self.log, cwd=self.packageDir)
    self.log.write(output+err)
    self.logPrintBox('Installing egads lite; this may take several minutes')
    output,err,ret  = config.package.GNUPackage.executeShellCommand(self.make.make+' -f make_lite.inc PETSC_DIR='+self.petscdir.dir+' prefix='+self.installDir+' install-egads',timeout=1000, log = self.log, cwd=self.packageDir)
    self.log.write(output+err)
    return

  def postProcess(self):
    if not hasattr(self,'installDir'):
      return
    try:
      self.buildEGADSLite()
    except RuntimeError as e:
      raise RuntimeError('Error running make on egads lite: '+str(e))
    try:
      self.buildEGADS()
    except RuntimeError as e:
      raise RuntimeError('Error running make on egads: '+str(e))
