
# -----------------------------------------------------------------------
# Specify PETSc installation and user-defined compiler flags
# -----------------------------------------------------------------------
CFLAGS    = $(CPPFLAGS)
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

pfull: chkopts $(OBJSC) $(SOURCEH)
	-$(CLINKER) -o pfull $(OBJSC) $(PETSC_LIB)

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base

# -----------------------------------------------------------------------
# Define example tests
# -----------------------------------------------------------------------
#
# runex1 - test for comparison with validated output.
#
runpfull1:
	-@$(MPIRUN) -np 1 pfull -nox -print_param -snes_mf_operator -jfreq 10 \
	    -mat_mpiaij -pc_type bjacobi -sub_pc_type lu -snes_smonitor -ksp_smonitor \
	    -mx 20 -my 20 -snes_view -snes_fd_coloring > pfull1_1.tmp 2>&1; \
	   if (diff output/pfull1_1.out pfull1_1.tmp) then true; \
	   else echo "Possible problem with pfull1_1, diffs above"; fi; \
	   $(RM) -f pfull1_1.tmp

runpfull4:
	-@$(MPIRUN) -np 4 pfull -nox -pc_type bjacobi -sub_pc_type lu \
	    -mx 20 -my 20 -snes_view -snes_fd_coloring -snes_smonitor -ksp_smonitor \
           > pfull4_1.tmp 2>&1; \
	   if (diff output/pfull4_1.out pfull4_1.tmp) then true; \
	   else echo "Possible problem with pfull4_1, diffs above"; fi; \
	   $(RM) -f pfull4_1.tmp

runcool:
	-@$(MPIRUN) $(NP) pfull -print_param -snes_mf_operator  \
	    -mat_mpiaij -pc_type bjacobi -sub_pc_type lu -ksp_smonitor \
	    -mx 20 -my 20 -snes_view -optionsleft -draw_x_private_colormap \
            -draw_pause -1

runcoolnox:
	-@$(MPIRUN) $(NP) pfull -nox -print_param -snes_mf_operator \
	    -mat_mpiaij -pc_type bjacobi -sub_pc_type lu -ksp_smonitor \
	    -mx 20 -my 20 -snes_view -optionsleft -draw_x_private_colormap \
            -draw_pause -1

# ---------------------------------------------------------------
#                    Utilities
# ---------------------------------------------------------------

etags_pfull:
	$(RM) TAGS_PFULL
	etags -f TAGS_PFULL $(SOURCEC) $(SOURCEF) $(SOURCEH)
 
