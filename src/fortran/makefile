PETSC_DIR = ../..

ALL: lib

CFLAGS   =  $(PETSC_DIR) -I..  \
          $(CONF) -D__DIR__='"fortran/"'
SOURCEC  = 
SOURCEF  =
SOURCEH  = 
OBJSC    = 
OBJSF    =
LIBBASE  = libpetscfortran
MANSEC   = 1
DIRS     = custom

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)

