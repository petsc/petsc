#PETSC_DIR = ../../../..

ALL: lib

CFLAGS   =  $(PETSC_INCLUDE) $(PCONF) $(CONF) -D__SDIR__='"$(LOCDIR)"'
SOURCEC  = 
SOURCEF  =
SOURCEH  = 
OBJSC    = 
OBJSF    =
LIBBASE  = libpetscts
MANSEC   = t
LOCDIR   = src/ts/impls/explicit/
DIRS     = euler

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base

