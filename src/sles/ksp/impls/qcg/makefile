ALL: lib 
PETSC_DIR = ../../../../

CFLAGS   =  $(PETSC_INCLUDE) $(CONF) $(PCONF) \
           -I../.. -D__DIR__='"ksp/impls/qcg/"'
FFLAGS   =
SOURCEC  = qcg.c
SOURCEF  =
SOURCEH  = qcg.h 
OBJSC    = qcg.o
OBJSF    =
LIBBASE  = libpetscksp
LINCLUDE = 

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
