PETSC_DIR = ../

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I.. -I$(IPETSCDIR) $(CONF) $(PCONF) 
SOURCEC  =
SOURCEF  =
SOURCEH  = draw.h is.h ksp.h mat.h options.h pc.h petsc.h sys.h \
           sysio.h vec.h sles.h mg.h viewer.h snes.h \
           da.h mpiu.h plog.h phead.h
OBJSC    =
OBJSF    =
LIBBASE  = libpetscvec
DIRS     = pinclude FINCLUDE 
MANSEC   = 7

runexamples:

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)


