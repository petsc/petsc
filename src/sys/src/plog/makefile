PETSC_DIR = ../../..

ALL: lib 
CFLAGS	  = $(CPPFLAGS) $(CONF) -D__DIR__='"$(LOCDIR)"'
SOURCEC	  = plog.c plogmpe.c
SOURCEF	  =
SOURCEH	  = 
OBJSC	  = plog.o plogmpe.o
OBJSF	  =
MANSEC	  = p
LIBBASE	  = libpetscsys
DIRS	  = 
LOCDIR	  = src/plog/src/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
