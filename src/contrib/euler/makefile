#
# $Id: makefile,v 1.19 1997/10/12 15:42:00 curfman Exp curfman $ 
#
# PETSc makefile for the Euler application code
#
# Organization for application makefiles using PETSc:
#  - Specify PETSc installation and user-defined compiler flags
#  - Define application source code, program tests, and utilities
#  - Specify "include" statement for using PETSc portable makefiles
#
# -----------------------------------------------------------------------
# Specify PETSc installation and user-defined compiler flags
# -----------------------------------------------------------------------

# Argonne Quad machine
  PETSC_DIR = /home/petsc/petsc-2.0.20
#   PETSC_DIR = /ccsthome/petsc/petsc
# Argonne tmp directories
#   PETSC_DIR = /tmp/petsc
# Argonne network
#   PETSC_DIR = /home/curfman/petsc
# Argonne SPx
#   PETSC_DIR = /sphome/curfman/petsc
# Origin2000 and Power Challenge - NCSA
#   PETSC_DIR = /u/ac/barrys/petsc
# Origin2000 in Norway
#   PETSC_DIR = /Home/origin/keyes/petsc
# Cray T3E - NERSC
#   PETSC_DIR = /u3/bsmith/petsc
# Cray T3D - NERSC
#   PETSC_DIR = /u/ddscf/bsmith/petsc
# SP2 - NASA
#   PETSC_DIR = /u/mcinnes/work/petsc
#
# Compiler flags:  Set variable CFLAGS
#
# ACTIVATE_JULIANNE - Flag to indicate that we activate the original 
#    Julianne solver and thus must compile _all_ Fortran source, even
#    that used only by the Julianne solver.  Not defining this flag
#    saves some memory for the PETSc version.  This flag is used only
#    in the routine julianne() in xnd.F.
#
# ACTIVATE_PVAR - Flag to indicate that we compile the pvar() routine
#    for computing physical data.  We can deactivate this to save a
#    bit of memory.
#
# ACTIVATE_OLD_ASSEMBLY - Flag to indicate that we activate old-style
#    matrix and vector assembly options
#
# CPPFLAGS - Basic compiler flags, as specified by petsc/bmake/common
# FCONF    - Additional compiler flags for Fortran, needed here to pick
#            up the C/Fortran interface flag 
# CONF     - needed for handling 64-bit pointer conversion to Fortran
#             (shouldn't be needed by most applications)

#  CFLAGS    = $(CPPFLAGS) $(CONF) $(FCONF) -DACTIVATE_JULIANNE
 CFLAGS    = $(CPPFLAGS) $(CONF) $(FCONF) -D__SDIR__='"src/contrib/euler/"' \
             -DACTIVATE_PVAR 

# -----------------------------------------------------------------------
# Define application source code
# -----------------------------------------------------------------------
#
# Use following variables for use with PETSc commands "make ci" and
# "make co" for checking in/out an entire directory of files via RCS.
#    SOURCEC - C source code
#    SOURCEF - Fortran source code
#    SOURCEH - include files
# 
# Note: "make clean" removes the files
#    *.o, ex[0-9], ex[0-9][0-9], plus others (see petsc/bmake/common)
#    CLEANFILES - variable to specify other files to delete via "make clean"
#    
# Define default make rule
#
ALL: ex1

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#
# Define Fortran source code:
#
# [SOURCEF,OBJSF]_COMMON   - routines used by both the PETSc and Julianne solvers
# [SOURCEF,OBJSF]_JULIANNE - routines used only by the Julianne solver
# [SOURCEF,OBJSF]_PETSC    - routines used only by the PETSc solvers
# [SOURCEF,OBJSF]          - all application Fortran source
#
# We can (temporarily) remove OBJSF_JULIANNE, which are object files that use
# common blocks with lots of memory and are needed only by the original Julianne
# solver.  This saves some memory for the PETSc version of code.
#
  SOURCEF_COMMON   = xnd.F resid.F jac.F pvar.F
  SOURCEF_JULIANNE = xjul.F
  SOURCEF_PETSC    = xmod.F mat.F obselete.F 
  SOURCEF_FP       = resid_fp.F xnd_fp.F rbuild_fp.F
  SOURCEF          = $(SOURCEF_COMMON) $(SOURCEF_PETSC) $(SOURCEF_FP) $(SOURCEF_JULIANNE)

  OBJSF_COMMON     = xnd.o resid.o jac.o pvar.o
  OBJSF_JULIANNE   = xjul.o
  OBJSF_PETSC      = xmod.o mat.o obselete.o

  OBJSF_FP         = resid_fp.o xnd_fp.o rbuild_fp.o
  OBJSF            = $(OBJSF_COMMON) $(OBJSF_PETSC) $(OBJSF_FP) 
#  OBJSF           = $(OBJSF_COMMON) $(OBJSF_PETSC) $(OBJSF_JULIANNE)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#
# Define C source code:
  OBJSC_MM   = mmregis.o mmset.o mmbasic.o mmmethods.o matv.o fdda.o
  SOURCEC_MM = mmregis.c mmset.c mmbasic.c mmmethods.c matv.c fdda.c
  OBJSC      = user1.o grid.o jacob.o bc.o misc.o mfj.o monitor.o vrml.o $(OBJSC_MM)
  SOURCEC    = user1.c grid.c jacob.c bc.c misc.c mfj.c monitor.c vrml.c $(SOURCEC_MM)

# Source no longer used.
# SOURCE_SURPLUS = regmpimat.c matbd.F

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#
# Define application include files:
#
# SOURCEH_C - C include files 
# SOURCEH_F - Fortran include files 
# SOURCEH   - all application include files
#
  SOURCEH_C = user.h 
  SOURCEH_F = matrixb.h diag.h dt.h varbls.h dvarbls.h param.h ang.h \
              avctrs.h condi.h consts.h coord.h bc.h es.h freq.h fv.h \
              gs.h init.h sada.h sigma.h wing.h bcimpl.h ibcwork.h \
              aux_fp.h
  SOURCEH   = $(SOURCEH_C) $(SOURCEH_F)

# -----------------------------------------------------------------------
# Define programs
# -----------------------------------------------------------------------

ex1: chkopts $(OBJSC) $(OBJSF) $(SOURCEH)
	-$(FLINKER) -o ex1 $(OBJSC) $(OBJSF) $(PETSC_FORTRAN_LIB) $(PETSC_LIB)

# We have trouble with multiply defined main when using the fortran linker on ptera (alpha)
# The following works though.
ex1_alpha: chkopts $(OBJSC) $(OBJSF) $(SOURCEH)
	-$(CLINKER) -o ex1 $(OBJSC) $(OBJSF) $(PETSC_FORTRAN_LIB) $(PETSC_LIB) \
              /usr/lib/libm_4sqrt.a 

# -----------------------------------------------------------------------
# Define program tests
# -----------------------------------------------------------------------

runex1:
	-@$(MPIRUN) ex1 -mat_seqbaij -mat_view_info_detailed -pc_type ilu -snes_max_it 14 > ex1_1.tmp 2>&1;	  \
	   if (diff output/ex1_1.out ex1_1.tmp) then true; \
	   else echo "Possible problem with ex1_1, diffs above"; fi; \
	   $(RM) -f ex1_1.tmp

runex1_2:
	-@$(MPIRUN) ex1 -pc_type none -snes_max_it 5 -ksp_monitor > ex1_2.tmp 2>&1;	  \
	   if (diff output/ex1_2.out ex1_2.tmp) then true; \
	   else echo "Possible problem with ex1_2, diffs above"; fi; \
	   $(RM) -f ex1_2.tmp

runex1_3:
	-@$(MPIRUN) ex1 -mat_mpirowbs -mat_view_info_detailed -pc_type ilu -snes_max_it 14 > ex1_1.tmp 2>&1;	  \
	   if (diff output/ex1_3.out ex1_3.tmp) then true; \
	   else echo "Possible problem with ex1_3, diffs above"; fi; \
	   $(RM) -f ex1_3.tmp

runex1_4:
	   if (diff output/ex1_4.out ex1_4.tmp) then true; \
	   else echo "Possible problem with ex1_4, diffs above"; fi; \
	   $(RM) -f ex1_4.tmp

# original solver (uniprocessor only!)
# output in file nd.out.5
runex1_5:
	-@$(MPIRUN) ex1 -julianne

# -----------------------------------------------------------------------
# Define utilities
# -----------------------------------------------------------------------

# To access the tags in emacs, type M-x visit-tags-table and specify
# the file TAGS_EULER.	Then, to move to where a routine is
# defined, enter M-. and the function name.  To search for a string
# and move to the first occurrence, use M-x tags-search and the string.
# To locate later occurrences, use M-,

etags_euler:
	$(RM) TAGS_EULER
	etags -f TAGS_EULER $(SOURCEC) $(SOURCEF) $(SOURCEH)

etags_fortran:
	$(RM) TAGS_FORTRAN
	etags -f TAGS_FORTRAN $(SOURCEF) $(SOURCEH_F)

clean_f:
	$(RM) $(OBJSF)

clean_mm:
	$(RM) $(OBJSC_MM)

clean_c:
	$(RM) $(OBJSC)

#
# Make PETSc tar file (containing all but input files that define meshes)
#
tar_euler:
	cp $(SOURCEC) $(SOURCEF) $(SOURCEH) makefile nd.in euler 
	chmod 664 euler/*
	tar -cf euler.tar euler
	gzip euler.tar

# -----------------------------------------------------------------------
# Include statement for using PETSc portable makefiles
# -----------------------------------------------------------------------

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base

# needed for freebsd
# FOPTFLAGS = -g -dalign -Nn802


# -----------------------------------------------------------------------
# Junk
# -----------------------------------------------------------------------
#
# We temporarily redefine some of the default compilation rules keep the
# Fortran *.f files (generated from the *.F files) for use with debuggers
# on certain machines.  This is dangerous and not portable! 

# Compilation rule for rs6000 only (using debugger)
# Use the compiler option -C for array bounds checking
#.F.o:
#	-$(RM) $*.f __$*.c
#	-cp $*.F __$*.c
#	-$(CC) $(CFLAGS) -E  __$*.c | grep -v '^ *#' > $*.f
#	-$(FC) -c $(FFLAGS) $(FOPTFLAGS) $*.f
##	-$(RM) $*.f __$*.c

# Compilation rule for freebsd only (using debugger)
#.F.o:
#	$(RM) $*.f
#	-cpp $(CFLAGS)  $*.F > $*.f 
#	-$(FC) -c $(FFLAGS) $(FOPTFLAGS) $*.f
##	$(RM) $*.f

# Compilation rule for sun4, IRIX64 (using array bounds checking)
#.F.o: 
#	$(FC)  -c $(FFLAGS)  $(FOPTFLAGS) $(CFLAGS) -C $*.F
##	$(FC)  -c $(FFLAGS)  $(FOPTFLAGS) $(CFLAGS) $*.F

# Compilation rule for Cray t3e
#.F.o:
#	-cpp  -DPARCH_t3d $(CFLAGS) -P $*.F $*.f
#	-$(FC) -c $(FFLAGS) $(FOPTFLAGS) $*.f
##	-$(RM) $*.f 
