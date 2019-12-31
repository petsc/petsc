import config.package
import os

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.gitcommit         = '09fe8f0fe689bea60354eb3e9977fd8452c05573'
    self.download          = ['git://https://github.com/bldenton/EGADSlite.git']
    self.functions         = []
    self.includes          = []
    self.hastests          = 1
    return

  def setupDependencies(self, framework):
    config.package.GNUPackage.setupDependencies(self, framework)
    return

  def createMakefile(self):
    makeinc = os.path.join(self.packageDir, 'make.inc')
    g = open(makeinc,'w')
    g.write('''
include $(PETSC_DIR)/lib/petsc/conf/variables

CFLAGS     = -DLITE -Iinclude

INCDIR     = include
SRCDIR     = src
LIBBASE    = libegadslite
LIBNAME    = ${LIBBASE}.${AR_LIB_SUFFIX}
LIBSRC.h   = $(INCDIR)/egads.h $(INCDIR)/egadsErrors.h $(INCDIR)/egadsInternals.h $(INCDIR)/egadsTris.h \
             $(INCDIR)/egadsTypes.h $(INCDIR)/emp.h $(INCDIR)/liteClasses.h
LIBSRC.c   = $(SRCDIR)/liteAttrs.c $(SRCDIR)/liteBase.c $(SRCDIR)/liteGeom.c $(SRCDIR)/liteImport.c \
             $(SRCDIR)/liteMemory.c $(SRCDIR)/liteTopo.c $(SRCDIR)/egadsTess.c $(SRCDIR)/egadsTris.c \
             $(SRCDIR)/egadsQuads.c $(SRCDIR)/egadsTessInp.c $(SRCDIR)/egadsRobust.c \
			 		 	 $(SRCDIR)/emp.c $(SRCDIR)/evaluate.c $(SRCDIR)/rational.c
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

COMPILE.c = $(CC) $(PCC_FLAGS) $(CFLAGS) $(CCPPFLAGS) $(TARGET_ARCH) -c

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
    self.createMakefile()
    return self.installDir

  def configureLibrary(self):
    ''' Since egads cannot be built until after PETSc configure is complete we need to just assume the downloaded library will work'''
    if 'with-egads' in self.framework.clArgDB:
      raise RuntimeError('egads does not support --with-egads; only --download-egads')
    if 'with-egads-dir' in self.framework.clArgDB:
      self.egadsDir = self.framework.argDB['with-egads-dir']
    if 'with-egads-include' in self.framework.clArgDB:
      raise RuntimeError('egads does not support --with-egads-include; only --download-egads')
    if 'with-egads-lib' in self.framework.clArgDB:
      raise RuntimeError('egads does not support --with-egads-lib; only --download-egads')
    if 'with-egads-shared' in self.framework.clArgDB:
      raise RuntimeError('egads does not support --with-egads-shared')

    if not hasattr(self,'egadsDir'):
      self.checkDownload()
      self.egadsDir = self.installDir
    self.include = [os.path.join(self.egadsDir,'include')]
    self.lib     = [os.path.join(self.egadsDir,'lib','libegadslite.a')]
    self.found   = 1
    self.dlib    = self.lib
    if not hasattr(self.framework, 'packages'):
      self.framework.packages = []
    self.framework.packages.append(self)

  def postProcess(self):
    if not hasattr(self,'installDir'):
      return
    try:
      self.logPrintBox('Compiling egads; this may take several minutes')
      # uses the regular PETSc library builder and then moves result
      # turn off any compiler optimizations as they may break egads
      self.setCompilers.pushLanguage('C')
      cflags = self.checkNoOptFlag()+' '+self.getSharedFlag(self.setCompilers.getCompilerFlags())+' '+self.getPointerSizeFlag(self.setCompilers.getCompilerFlags())+' '+self.getWindowsNonOptFlags(self.setCompilers.getCompilerFlags())+' '+self.getDebugFlags(self.setCompilers.getCompilerFlags())
      self.setCompilers.popLanguage()
      output,err,ret  = config.package.GNUPackage.executeShellCommand(self.make.make+' -f make.inc PETSC_DIR=' + self.petscdir.dir + ' clean lib PCC_FLAGS="' + cflags + '"', timeout=1000, log = self.log, cwd=self.packageDir)
      self.log.write(output+err)
      self.logPrintBox('Installing egads; this may take several minutes')
      # TODO: This message should not be printed if egads is install in PETSc arch directory; need self.printSudoPasswordMessage() defined in package.py
      self.installDirProvider.printSudoPasswordMessage(1)
      output,err,ret  = config.package.GNUPackage.executeShellCommand(self.installSudo+self.make.make+' -f make.inc PETSC_DIR='+self.petscdir.dir+' prefix='+self.installDir+' install-egads',timeout=1000, log = self.log, cwd=self.packageDir)
      self.log.write(output+err)
    except RuntimeError as e:
      raise RuntimeError('Error running make on egads: '+str(e))
