PETSC_DIR = ../

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I.. -I$(IPETSCDIR) $(CONF) $(PCONF) 
SOURCEC  =
SOURCEF  =
SOURCEH  = draw.h is.h ksp.h mat.h options.h pc.h petsc.h sys.h \
           vec.h sles.h mg.h viewer.h snes.h \
           da.h plog.h phead.h ts.h
OBJSC    =
OBJSF    =
LIBBASE  = libpetscvec
DIRS     = pinclude FINCLUDE 
LOCDIR   = include/
MANSEC   = p

runexamples:

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base


