# $Id: makefile,v 1.184 1997/09/10 16:23:37 bsmith Exp $ 

# -----------------------------------------------------------------------
# Specify PETSc installation and user-defined compiler flags
# -----------------------------------------------------------------------
CFLAGS    = ${CPPFLAGS}
# -----------------------------------------------------------------------
# Define application source and examples
# -----------------------------------------------------------------------
#
# Use following variables for use with PETSc commands "make ci" and
# "make co" for checking in/out an entire directory of files via RCS.
#    SOURCEC - C source code
#    OBJSC   - C source with .o extension
#    SOURCEF - Fortran source code
#    SOURCEH - include files
# 
# Note: "make clean" removes the files
#    *.o 
#    CLEANFILES - variable to specify other files to delete via "make clean"
#    

ALL: pfull

SOURCEC    = pfull.c pfunc.c
SOURCEF    =
SOURCEH    = pfull.h
OBJSC      = pfull.o pfunc.o
CLEANFILES = pfull

pfull: chkopts ${OBJSC} ${SOURCEH}
	-${CLINKER} -o pfull ${OBJSC} ${PETSC_LIB}

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

# -----------------------------------------------------------------------
# Define example tests
# -----------------------------------------------------------------------
#
# Do make BOPT=g runpfull1 or runpfull4 for testing
#
runpfull1:
	-@${MPIRUN} -np 1 pfull -options_file in.1 > pfull1_1.tmp 2>&1; \
	   if (diff output/pfull1_1.out pfull1_1.tmp) then true; \
	   else echo "Possible problem with pfull1_1, diffs above"; fi; \
	   ${RM} -f pfull1_1.tmp

runpfull4:
	-@${MPIRUN} -np 4 pfull -options_file in.4 > pfull4_1.tmp 2>&1; \
	   if (diff output/pfull4_1.out pfull4_1.tmp) then true; \
	   else echo "Possible problem with pfull4_1, diffs above"; fi; \
	   ${RM} -f pfull4_1.tmp


# ---------------------------------------------------------------
#                    Utilities
# ---------------------------------------------------------------

etags_pfull:
	${RM} TAGS_PFULL
	etags -f TAGS_PFULL ${SOURCEC} ${SOURCEF} ${SOURCEH}
 
