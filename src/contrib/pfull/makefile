
# -----------------------------------------------------------------------
# Specify PETSc installation and user-defined compiler flags
# -----------------------------------------------------------------------

PETSC_DIR = /home/petsc/petsc-2.0.14
CFLAGS    = $(CPPFLAGS)

# -----------------------------------------------------------------------
# Define application source and examples
# -----------------------------------------------------------------------
#
# Use following variables for use with PETSc commands "make ci" and
# "make co" for checking in/out an entire directory of files via RCS.
#    SOURCEC - C source code
#    SOURCEF - Fortran source code
#    SOURCEH - include files
# Also can use:
#    EXAMPLESC - C examples
#    EXAMPLESF - Fortran examples
# 
# Note: "make clean" removes the files
#    *.o, ex[0-9], ex[0-9][0-9], plus others (see petsc/bmake/common)
#    CLEANFILES - variable to specify other files to delete via "make clean"
#    

ALL: pfull

PFULL    = pfull.o pdsfi.o formjac.o
PFULL_C  = pfull.c pdsfi.c formjac.c

SOURCEC    = $(PFULL_C)
SOURCEF    =
SOURCEH    = puser.h
CLEANFILES = pfull

# Both conventional and  matrix-free versions
pfull: chkopts $(PFULL) $(SOURCEH)
	-$(CLINKER) -o pfull $(PFULL) $(PETSC_LIB)

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base

# -----------------------------------------------------------------------
# Define example tests
# -----------------------------------------------------------------------

# runex1 - test for comparison with Jie's output, as given in Readme
# See Readme for summary of differences between this version of the
# full potential code and the old version.
runex1:
	-@$(MPIRUN) pfull -nox -print_param -matrix_free -jfreq 10 \
	    -mat_mpiaij -pc_type bjacobi -sub_pc_type lu -ksp_smonitor \
	    -mx 20 -my 20 -snes_view -optionsleft > ex1_1.tmp 2>&1; \
	   if (diff output/ex1_1.out ex1_1.tmp) then true; \
	   else echo "Possible problem with ex1_1, diffs above"; fi; \
	   $(RM) -f ex1_1.tmp

runex1_2:
	-@$(MPIRUN) -np 4 pfull -nox -pc_type bjacobi -sub_pc_type lu \
	    -mx 20 -my 20 -snes_view -optionsleft > ex1_2.tmp 2>&1; \
	   if (diff output/ex1_2.out ex1_2.tmp) then true; \
	   else echo "Possible problem with ex1_2, diffs above"; fi; \
	   $(RM) -f ex1_2.tmp


# ---------------------------------------------------------------
#                    Utilities
# ---------------------------------------------------------------

etags_pfull:
	$(RM) TAGS_PFULL
	etags -f TAGS_PFULL $(SOURCEC) $(SOURCEF) $(SOURCEH)
 
# Redefine some utilities from petsc/bmake/common for the ODU system

# RCS file check-in
ci_odu: 
	@-ci -u -q -mAutoCheckin $(SOURCEH) $(SOURCE) makefile\
	  $(EXAMPLESC) $(EXAMPLESF) $(TESTSC) $(TESTSF) 

# RCS file check-out
co_odu:
	@-co -l -q $(SOURCEH) $(SOURCE) makefile $(EXAMPLESC)\
	  $(EXAMPLESF) $(TESTSC) $(TESTSF)
