IPETSCDIR = ../

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I.. -I$(IPETSCDIR) $(CONF)
SOURCEC  =
SOURCEF  =
SOURCEH  = draw.h is.h ksp.h mat.h options.h pc.h petsc.h sys.h \
           sysio.h vec.h sles.h grid.h mg.h stencil.h viewer.h snes.h \
           da.h mpiu.h ptime.h plog.h phead.h
OBJSC    =
OBJSF    =
LIBBASE  = libpetscvec
DIRS     = pinclude finclude 
MANSEC   = 7

runexamples:

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)


