#PETSC_DIR = ../

CFLAGS   =  $(CPPFLAGS) $(CONF) -D__DIR__='"$(LOCDIR)"'
SOURCEC  =
SOURCEF  =
SOURCEH  = draw.h is.h ksp.h mat.h options.h pc.h petsc.h sys.h \
           vec.h sles.h mg.h viewer.h snes.h dfvec.h  ec.h ao.h\
           da.h petsclog.h petschead.h petscerror.h ts.h gvec.h
OBJSC    =
OBJSF    =
LIBBASE  = libpetscvec
DIRS     = pinclude FINCLUDE 
LOCDIR   = include/
MANSEC   = p

runexamples:

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base


