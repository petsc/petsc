# $Id: makefile,v 1.7 1997/08/22 15:19:22 bsmith Exp bsmith $ 

ALL: lib 
CFLAGS	  = $(CPPFLAGS) $(CONF) -D__SDIR__='"$(LOCDIR)"'
SOURCEC	  = plog.c plogmpe.c ploginfo.c
SOURCEF	  =
SOURCEH	  = 
OBJSC	  = plog.o plogmpe.o ploginfo.o
OBJSF	  =
MANSEC	  = p
LIBBASE	  = libpetscsys
DIRS	  = 
LOCDIR	  = src/plog/src/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
