#
# $Id: makefile,v 1.28 1998/06/09 00:04:31 curfman Exp curfman $ 
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
#  PETSC_DIR = /home/petsc/petsc-2.0.22
#  PETSC_DIR = /tmp/curfman/petsc/
   PETSC_DIR = /ccsthome/petsc/petsc
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

#  CFLAGS    = ${CPPFLAGS} -DACTIVATE_JULIANNE
 CFLAGS    = ${CPPFLAGS} -D__SDIR__='"src/contrib/euler/"' \
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
  SOURCEF_COMMON   = xnd.F resid.F jac.F pvar.F resid_duct.F
  SOURCEF_JULIANNE = xjul.F
  SOURCEF_PETSC    = xmod.F mat.F obselete.F 
  SOURCEF_FP       = resid_fp.F xnd_fp.F rbuild_fp.F
  SOURCEF          = ${SOURCEF_COMMON} ${SOURCEF_PETSC} ${SOURCEF_FP} ${SOURCEF_JULIANNE}

  OBJSF_COMMON     = xnd.o resid.o jac.o pvar.o resid_duct.o
  OBJSF_JULIANNE   = xjul.o
  OBJSF_PETSC      = xmod.o mat.o obselete.o

  OBJSF_FP         = resid_fp.o xnd_fp.o rbuild_fp.o 
  OBJSF            = ${OBJSF_COMMON} ${OBJSF_PETSC} ${OBJSF_FP} 
#  OBJSF           = ${OBJSF_COMMON} ${OBJSF_PETSC} ${OBJSF_JULIANNE}

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#
# Define C source code:
  OBJSC_MM   = mmregis.o mmset.o mmbasic.o mmmethods.o mmdlregis.o matv.o fdda.o
  SOURCEC_MM = mmregis.c mmset.c mmbasic.c mmmethods.c mmdlregis.c matv.c fdda.c
  SOURCEH_MM = mm.h mmimpl.h mmfort.h 
  OBJSC      = user1.o grid.o jacob.o bc.o misc.o mfj.o monitor.o vrml.o ${OBJSC_MM}
  SOURCEC    = user1.c grid.c jacob.c bc.c misc.c mfj.c monitor.c vrml.c ${SOURCEC_MM}

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
  SOURCEH_C = user.h ${SOURCEH_MM}
  SOURCEH_F = matrixb.h diag.h dt.h varbls.h dvarbls.h param.h \
              ang.h avctrs.h consts.h coord.h bc.h es.h fv.h freq.h \
              gs.h init.h sada.h sigma.h wing.h bcimpl.h ibcwork.h \
              aux_fp.h
  SOURCEH   = ${SOURCEH_C} ${SOURCEH_F}

# -----------------------------------------------------------------------
# Define programs
# -----------------------------------------------------------------------

ex1: chkopts ${OBJSC} ${OBJSF} ${SOURCEH}
	-${FLINKER} -o ex1 ${OBJSC} ${OBJSF} ${PETSC_FORTRAN_LIB} ${PETSC_LIB}

# We have trouble with multiply defined main when using the fortran linker on ptera (alpha)
# The following works though.
ex1_alpha: chkopts ${OBJSC} ${OBJSF} ${SOURCEH}
	-${CLINKER} -o ex1 ${OBJSC} ${OBJSF} ${PETSC_FORTRAN_LIB} ${PETSC_LIB} \
              /usr/lib/libm_4sqrt.a 

# -----------------------------------------------------------------------
# Define program tests
# -----------------------------------------------------------------------

#
#  All of these experiments use implicit boundary conditions.
#
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#
#  The following use the smallest grid, m6c (problem 1, the default)

# Explicitly form Jacobian via default Julianne FD
runex1_1:
	-@${MPIRUN} ex1 -optionsleft > log.m6c.3d.me.lva.p1 2>&1; \
	   if (diff output/order2/log.m6c.3d.me.lva.p1 log.m6c.3d.me.lva.p1) then true; \
	   else echo "Output differences as above"; fi; \
	   mv f_m6c_cc1_asm0_p1.m f_m6c_3d_me_lva_p1.m

# Same as runex1_1 except with 1st order discretization, no limiters
runex1_1b:
	-@${MPIRUN} ex1 -optionsleft -order 1 -limiter none > log.m6c.3d.me.lnone.p1 2>&1; \
	   if (diff output/order1/log.m6c.3d.me.lnone.p1 log.m6c.3d.me.lnone.p1) then true; \
	   else echo "Output differences as above"; fi; \
	   mv f_m6c_cc1_asm0_p1.m f_m6c_3d_me_lnone_p1.m

# Matrix-free multiplication, advancing CFL
# Explicitly form preconditioning Jacobian via default Julianne FD
# (no wake boundary conditions in preconditioning matrix)
runex1_2:
	-@${MPIRUN}  ex1 -optionsleft \
	   -matrix_free -cfl_advance > log.m6c.3d.mf.lva.p1 2>&1; \
	   if (diff output/order2/log.m6c.3d.mf.lva.p1 log.m6c.3d.mf.lva.p1) then true; \
	   else echo "Output differences as above"; fi; \
	   mv f_m6c_cc1_asm0_p1.m f_m6c_3d_mf_lva_p1.m

# Same as runex1_2 except with 1st order discretization, no limiters
runex1_2b:
	-@${MPIRUN}  ex1 -optionsleft -order 1 -limiter none \
	   -matrix_free -cfl_advance > log.m6c.3d.mf.lnone.p1 2>&1; \
	   if (diff output/order1/log.m6c.3d.mf.lnone.p1 log.m6c.3d.mf.lnone.p1) then true; \
	   else echo "Output differences as above"; fi; \
	   mv f_m6c_cc1_asm0_p1.m f_m6c_3d_mf_lnone_p1.m

# Same as runex1_2 except with 1st order discretization, no limiters, no wake Jacobian contributions
runex1_2d_1c:
	-@${MPIRUN} ex1 -optionsleft -order 1 -limiter none -jac_no_wake > log.m6c.3d.me.lnone.nowake.p1 2>&1; \
	   if (diff output/order1/log.m6c.3d.me.lnone.p1 log.m6c.3d.me.lnone.nowake.p1) then true; \
	   else echo "Output differences as above"; fi; \
	   mv f_m6c_cc1_asm0_p1.m f_m6c_3d_me_lnone_nowake_p1.m

# Matrix-free multiplication, advancing CFL
# Explicitly form preconditioning Jacobian via default Julianne FD
# (include wake boundary conditions in preconditioning matrix)
runex1_3:
	-@${MPIRUN}  ex1 -optionsleft \
	   -jac_wake -matrix_free -cfl_advance > log.m6c.3d.mf.lva.jwake.p1 2>&1; \
	   if (diff output/order2/log.m6c.3d.mf.lva.jwake.p1 log.m6c.3d.mf.lva.jwake.p1) then true; \
	   else echo "Output differences as above"; fi; \
	   mv f_m6c_cc1_asm0_p1.m f_m6c_3d_mf_lva_jwake_p1.m

# Matrix-free multiplication, advancing CFL
# Explicitly form preconditioning Jacobian via default Julianne FD
# (no wake boundary conditions in preconditioning matrix)
# 8 processors, PC type: Restricted Additive Schwarz (RASM) with overlap 2
runex1_4:
	-@${MPIRUN} -np 8 ex1 -optionsleft \
	   -pc_type asm -pc_asm_type restrict -pc_asm_overlap 2 \
	   -matrix_free -cfl_advance > log.m6c.3d.mf.lva.asm2.p8 2>&1; \
	   if (diff output/order2/log.m6c.3d.mf.lva.asm2.p8 log.m6c.3d.mf.lva.asm2.p8) then true; \
	   else echo "Output differences as above"; fi; \
	   mv f_m6c_cc1_asm2_p8.m f_m6c_3d_mf_lva_asm2_p8.m

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# Experiments with switching from first to second order discretization after
# convergence to a relative tolerance of specified by -order_rtol.  

# We achieve good convergence by using 2nd order discretization immediately
# for the m6c grid; this fails for the more finely discretized versions though.
runtest_1a:
	-@${MPIRUN} ex1 -problem 1 -optionsleft \
	   -order 1.5 -order_rtol 1.e-3 -limiter none -bc_imperm 0 \
	   -matrix_free -cfl_advance > log.m6c.3d.mf.lnone.p1  2>&1; \
	   if (diff output/order1-2/log.m6c.3d.mf.lnone.p1 log.m6c.3d.mf.lnone.p1) then true; \
	   else echo "Output differences as above"; fi; \
	   mv f_m6c_cc1_asm0_p1.m f_m6c_3d_mf_lnone_p1.m

runtest_1b:
	-@${MPIRUN} -np 2 ex1 -problem 1 -optionsleft -vecscatter_rr \
	   -order 1.5 -order_rtol 1.e-3 -limiter none -bc_imperm 0 \
	   -pc_type asm -pc_asm_type restrict -pc_asm_overlap 2 \
	   -matrix_free -cfl_advance > log.m6c.3d.mf.lnone.asm2.p2  2>&1; \
	   if (diff output/order1-2/log.m6c.3d.mf.lnone.asm2.p2 log.m6c.3d.mf.lnone.asm2.p2) then true; \
	   else echo "Output differences as above"; fi; \
	   mv f_m6c_cc1_asm2_p2.m f_m6c_3d_mf_lnone_asm2_p2.m

# -order_rtol = 1.e-3 works well for the m6f grid
runtest_2a:
	-@${MPIRUN} ex1 -problem 2 -optionsleft \
	   -order 1.5 -order_rtol 1.e-3 -limiter none -bc_imperm 0 \
	   -matrix_free -cfl_advance > log.m6f.3d.mf.lnone.p1 2>&1; \
	   if (diff output/order1-2/log.m6f.3d.mf.lnone.p1 log.m6f.3d.mf.lnone.p1) then true; \
	   else echo "Output differences as above"; fi; \
	   mv f_m6f_cc1_asm0_p1.m f_m6f_3d_mf_lnone_p1.m

runtest_2b:
	-@${MPIRUN} -np 16 ex1 -problem 2 -optionsleft -vecscatter_rr \
	   -order 1.5 -order_rtol 1.e-3 -limiter none -bc_imperm 0 \
	   -pc_type asm -pc_asm_type restrict -pc_asm_overlap 2 \
	   -matrix_free -cfl_advance > log.m6f.3d.mf.lnone.asm2.p16 2>&1; \
	   if (diff output/order1-2/log.m6f.3d.mf.lnone.asm2.p16 log.m6f.3d.mf.lnone.asm2.p16) then true; \
	   else echo "Output differences as above"; fi; \
	   mv f_m6f_cc1_asm2_p16.m f_m6f_3d_mf_lnone_asm2_p16.m

# -order_rtol = 1.e-7 works well for the m6n grid
runtest_3:
	-@${MPIRUN} -np 16 -maxtime 30 ex1 -problem 3 -optionsleft -vecscatter_rr \
	   -order 1.5 -order_rtol 1.e-7 -limiter none -bc_imperm 0 \
	   -pc_type asm -pc_asm_type restrict -pc_asm_overlap 2 \
	   -matrix_free -cfl_advance > log.m6n.3d.mf.asm2.p16 2>&1; \
	   if (diff output/order1-2/log.m6n.3d.mf.lnone.asm2.p16 log.m6n.3d.mf.lnone.asm2.p16) then true; \
	   else echo "Output differences as above"; fi; \
	   mv f_m6n_cc1_asm2_p16.m f_m6n_3d_mf_lnone_asm2_p16.m

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#
# 2-dimensional tests, simply so that we have small problems for use in debugging

# Explicitly form Jacobian via default Julianne FD
runex1_2d_1:
	-@${MPIRUN} ex1 -dim2 -optionsleft > log.m6c.2d.me.lva.p1 2>&1; \
	   if (diff output/order2/log.m6c.2d.me.lva.p1 log.m6c.2d.me.lva.p1) then true; \
	   else echo "Output differences as above"; fi; \
	   mv f_m6c_cc1_asm0_p1.m f_m6c_2d_me_lva_p1.m

# Same as runex1_2d_1 except with 1st order discretization, no limiters
runex1_2d_1b:
	-@${MPIRUN} ex1 -dim2 -optionsleft -order 1 -limiter none > log.m6c.2d.me.lnone.p1 2>&1; \
	   if (diff output/order1/log.m6c.2d.me.lnone.p1 log.m6c.2d.me.lnone.p1) then true; \
	   else echo "Output differences as above"; fi; \
	   mv f_m6c_cc1_asm0_p1.m f_m6c_2d_me_lnone_p1.m

# Same as runex1_2d_1 except with 1st order discretization, no limiters, no wake Jacobian contributions
runex1_2d_1c:
	-@${MPIRUN} ex1 -dim2 -optionsleft -order 1 -limiter none -jac_no_wake > log.m6c.2d.me.lnone.nowake.p1 2>&1; \
	   if (diff output/order1/log.m6c.2d.me.lnone.p1 log.m6c.2d.me.lnone.nowake.p1) then true; \
	   else echo "Output differences as above"; fi; \
	   mv f_m6c_cc1_asm0_p1.m f_m6c_2d_me_lnone_nowake_p1.m

# Matrix-free multiplication, advancing CFL
# Explicitly form preconditioning Jacobian via default Julianne FD
# (no wake boundary conditions in preconditioning matrix)
runex1_2d_2:
	-@${MPIRUN} ex1 -dim2 -optionsleft \
	   -matrix_free -cfl_advance > log.m6c.2d.mf.lva.p1 2>&1; \
	   if (diff output/order2/log.m6c.2d.mf.lva.p1 log.m6c.2d.mf.lva.p1) then true; \
	   else echo "Output differences as above"; fi; \
	   mv f_m6c_cc1_asm0_p1.m f_m6c_2d_mf_lva_p1.m

# Same as runex1_2d_2 except with 1st order discretization, no limiters
runex1_2d_2b:
	-@${MPIRUN} ex1 -dim2 -optionsleft -order 1 -limiter none \
	   -matrix_free -cfl_advance > log.m6c.2d.mf.lnone.p1 2>&1; \
	   if (diff output/order1/log.m6c.2d.mf.lnone.p1 log.m6c.2d.mf.lnone.p1) then true; \
	   else echo "Output differences as above"; fi; \
	   mv f_m6c_cc1_asm0_p1.m f_m6c_2d_mf_lnone_p1.m


# -----------------------------------------------------------------------
# Define utilities
# -----------------------------------------------------------------------

# To access the tags in emacs, type M-x visit-tags-table and specify
# the file TAGS_EULER.	Then, to move to where a routine is
# defined, enter M-. and the function name.  To search for a string
# and move to the first occurrence, use M-x tags-search and the string.
# To locate later occurrences, use M-,

etags_euler:
	${RM} TAGS_EULER
	etags -f TAGS_EULER ${SOURCEC} ${SOURCEF} ${SOURCEH}

etags_fortran:
	${RM} TAGS_FORTRAN
	etags -f TAGS_FORTRAN ${SOURCEF} ${SOURCEH_F}

clean_f:
	${RM} ${OBJSF}

clean_mm:
	${RM} ${OBJSC_MM}

clean_c:
	${RM} ${OBJSC}

#
# Make PETSc tar file (containing all but input files that define grids)
#
tar_euler:
	cp ${SOURCEC} ${SOURCEF} ${SOURCEH} makefile euler 
	chmod 664 euler/*
	tar -cf euler.tar euler
	gzip euler.tar

# -----------------------------------------------------------------------
# Include statement for using PETSc portable makefiles
# -----------------------------------------------------------------------

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

# redefine PETSc lib location to use tmp space
# LDIR = /tmp/curfman/petsc-2.0.20/libg/

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
#	-${RM} $*.f __$*.c
#	-cp $*.F __$*.c
#	-${CC} ${CFLAGS} -E  __$*.c | grep -v '^ *#' > $*.f
#	-${FC} -c ${FFLAGS} ${FOPTFLAGS} $*.f
##	-${RM} $*.f __$*.c

