# -*- mode: makefile -*-

#  This SAMPLE Makefile can be used to compile PETSc applications
#  Copy this file to your source directory as "Makefile" and MODIFY AS NEEDED.
#  See also $PETSC_DIR/share/petsc/Makefile.user for the preferred approach
#
#  You must set PETSC_DIR either via environment variable, or define here, or via 'make' command line
#  (and similarly PETSC_ARCH - especially when PETSc is built without --prefix, and with multiple PETSC_ARCH values)
#
# PETSC_DIR  =
# PETSC_ARCH =
#
#  This will use all the compiler flags that PETSc was compiled with, if you do not want the flags use $PETSC_DIR/share/petsc/Makefile.user
#  and modify it as noted in the comments
#
#  You can set specific values below, or specify to 'make' via command line, but that should be rarely needed
#
CPPFLAGS   = # Used by compile target for C sources
FPPFLAGS   = # Used by compile target for Fortran sources
CXXPPFLAGS = # Used by compile target for Cxx sources
CFLAGS     = # Used by both compile and link targets for C sources
FFLAGS     = # Used by both compile and link targets for Fortran sources
CXXFLAGS   = # Used by both compile and link targets for Cxx sources
LDFLAGS    = # Used by link targets - option occurs after the linker, before objs, libraries
PRELIBS    = # Used by link targets - option occurs before PETSC_LIB
LIBS       = # Used by link targets - option occurs after PETSC_LIB

#  For example - a single source file (ex1.c or ex1.F90) can be compiled with:
#
#      make ex1
#
#  You do not need to edit this makefile at all.
#
#  For a multi-file case, suppose you have the source files app.c [with main()], b.c, and c.cxx,
#  the binary can be built by uncommenting the following line:
#
# app : b.o c.o
#
#  If the default link target above doesn't work (for any reason), and an explicit link target
#  is needed (say with FLINKER, or a different name for the binary), you may need to use
#  (with additional modifications - if needed):
#
# myapp : app.o b.o c.o
#	${LINK.F} $^ ${LOADLIBES} ${LDLIBS} -o $@

# Say you have source files in the "src" dir, include files in the "include" dir, and desire the compiler-generated
# object files be in the "obj" dir, and the application binary be in the "bin" dir, use:
#
# APP        = bin/app
# OBJS       = obj/app.o obj/b.o obj/c.o
# CPPFLAGS   = -Iinclude
# CXXPPFLAGS = -Iinclude
# CLEANFILES = ${APP} ${OBJS}
#
# all: ${APP}
#
# obj/%.o : src/%.c
#	${PETSC_CCOMPILE_SINGLE} $<
# obj/%.o : src/%.cxx
#	${PETSC_CXXCOMPILE_SINGLE} $<
#
# ${APP}: ${OBJS}
#	${LINK.C} $^ ${LOADLIBES} ${LDLIBS} -o $@

#  Access variables for the build, including compilers, compiler flags, libraries etc
#
include ${PETSC_DIR}/lib/petsc/conf/variables

#  Access compile targets here. However if you manage them yourself (rarely needed) comment out the next line
#
include ${PETSC_DIR}/lib/petsc/conf/rules

