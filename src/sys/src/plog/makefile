PETSC_DIR = ../../..

ALL: lib 
CFLAGS	  = $(PETSC_INCLUDE) $(CONF) $(PCONF) -D__DIR__='"plog/src/"'
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
