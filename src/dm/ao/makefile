# $Id: makefile,v 1.5 1997/08/22 15:19:31 bsmith Exp bsmith $ 

ALL: lib

CFLAGS	 = $(CPPFLAGS) $(CONF) -D__SDIR__='"$(LOCDIR)"'
SOURCEC	 =
SOURCEF	 =
SOURCEH	 = aoimpl.h
OBJSC	 =
OBJSF	 =
LIBBASE	 = libpetscvec
DIRS	 = interface impls  examples utils
LOCDIR   = src/is/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base

