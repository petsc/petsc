# $Id: makefile,v 1.179 1997/08/15 16:24:04 bsmith Exp bsmith $ 

ALL: lib

CFLAGS	 = $(CPPFLAGS) $(CONF) -D__SDIR__='"$(LOCDIR)"'
SOURCEC	 =
SOURCEF	 =
SOURCEH	 = isimpl.h
OBJSC	 =
OBJSF	 =
LIBBASE	 = libpetscvec
DIRS	 = interface impls  examples utils
LOCDIR   = src/is/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base

