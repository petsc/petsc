# $Id: makefile,v 1.3 2000/01/12 14:15:22 bsmith Exp bsmith $ 

ALL: flow

USERCPPFLAGS     =  -DCFL3D_AXIS -DINTERLACING -DBLOCKING 
CFLAGS	         =
FFLAGS	         =	 
#/real_size:64 /integer_size:32
CPPFLAGS         = $(USERCPPFLAGS) $(FCONF)
FPPFLAGS         = $(USERCPPFLAGS)
SOURCEC	         = 
SOURCEF	         =
SOURCEH	         =
OBJSC	         = 
OBJSF	         = user.o userJac.o
CLEANFILES       = flow.exe flow.o flow
LOCDIR           = src/contrib/fun3d
TESTEXAMPLES_3	 = flow.PETSc flow.rm
EXAMPLESC        = flow.c flow.h
EXAMPLESF        = user.c userJac.c
#
include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base

flow: flow.o $(OBJSF) chkopts
	-$(FLINKER) flow.o $(OBJSF) $(PETSC_FORTRAN_LIB) $(PETSC_LIB)

user.o:
#f77 -c -r8 -i4 -O4 -xs user.f
#	$(FC) -c $(FFLAGS) $(CFLAGS) $(FOPTFLAGS) user.F

userJac.o:
#	$(FC) -c $(FFLAGS) $(CFLAGS) $(FOPTFLAGS) userJac.F

puns3d: puns3d.a.o
	$(FLINKER) -o puns3d puns3d.a.o $(PETSC_FORTRAN_LIB) $(PETSC_LIB)
puns3d.a.o:

runflow:
	@$(MPIRUN) -maxtime 80 -np 16 flow \
-snes_monitor -ksp_monitor -ksp_max_it 10 -ksp_gmres_unmodifiedgramschmidt \
-snes_max_it 1 -max_st 100 -matrix_free -pc_type bjacobi \
-sub_pc_type ilu -sub_pc_ilu_levels 0 -sub_pc_ilu_in_place 
#-pc_bjacobi_blocks 2 -sub_pc_type ilu -sub_pc_ilu_levels 0 -sub_pc_ilu_in_place \
#-log_summary -log_info -trdump -trmalloc -trmalloc_log
#-start_in_debugger xxgdb \
#-start_in_debugger gdb \

flow4:
	poe flow -procs 4 \
-ksp_rtol 0.01 \
-cfl_ini 20.0 -cfl_max 100000.0 -first_order_it 10 \
-ksp_max_it 50 -ksp_gmres_unmodifiedgramschmidt -optionsleft \
-snes_max_it 1 -snes_no_convergence_test -max_st 100 -matrix_free -pc_type bjacobi \
-sub_pc_type ilu -sub_pc_ilu_levels 0 -sub_pc_ilu_in_place \
-print -ksp_monitor -snes_monitor \
-log_summary 
#> out.4

 
runflow1:
	@flow \
-ksp_rtol 0.01 -cfl_ini 150.0 -cfl_max 1.0e+10 \
-ksp_max_it 20 -ksp_gmres_unmodifiedgramschmidt -optionsleft \
-snes_max_it 1 -snes_no_convergence_test -max_st 200 -matrix_free -pc_type bjacobi \
-sub_pc_type ilu -sub_pc_ilu_levels 0 -sub_pc_ilu_in_place \
-print -ksp_monitor -snes_monitor \
-print -ksp_monitor -snes_monitor \
-log_summary -get_resident_set_size 
#> out.1
#-print -ksp_monitor -snes_monitor \

runflow1.block:
	@flow.block \
-ksp_max_it 10 -ksp_gmres_unmodifiedgramschmidt -optionsleft \
-snes_max_it 1 -snes_no_convergence_test -max_st 200 -matrix_free -pc_type bjacobi \
-sub_pc_type ilu -sub_pc_ilu_levels 0 -sub_pc_ilu_in_place \
-log_summary -get_resident_set_size 
#> out.1.block

runflow1.block_noer:
	@flow.block \
-ksp_max_it 10 -ksp_gmres_unmodifiedgramschmidt -optionsleft \
-snes_max_it 1 -snes_no_convergence_test -max_st 200 -matrix_free -pc_type bjacobi \
-sub_pc_type ilu -sub_pc_ilu_levels 0 -sub_pc_ilu_in_place \
-no_edge_reordering \
-log_summary -get_resident_set_size \
> out.1.block_noer

runflow1.intl:
	@flow.intl \
-ksp_max_it 10 -ksp_gmres_unmodifiedgramschmidt -optionsleft \
-snes_max_it 1 -snes_no_convergence_test -max_st 200 -matrix_free -pc_type bjacobi \
-sub_pc_type ilu -sub_pc_ilu_levels 0 -sub_pc_ilu_in_place \
-log_summary -get_resident_set_size \
> out.1.intl

runflow1.intl_noer:
	@flow.intl \
-ksp_max_it 10 -ksp_gmres_unmodifiedgramschmidt -optionsleft \
-snes_max_it 1 -snes_no_convergence_test -max_st 200 -matrix_free -pc_type bjacobi \
-sub_pc_type ilu -sub_pc_ilu_levels 0 -sub_pc_ilu_in_place \
-no_edge_reordering \
-log_summary -get_resident_set_size \
> out.1.intl_noer

runflow1.base:
	@flow.base \
-ksp_max_it 10 -ksp_gmres_unmodifiedgramschmidt -optionsleft \
-snes_max_it 1 -snes_no_convergence_test -max_st 200 -matrix_free -pc_type bjacobi \
-sub_pc_type ilu -sub_pc_ilu_levels 0 -sub_pc_ilu_in_place \
-log_summary -get_resident_set_size \
> out.1.base

runflow1.base_noer:
	@flow.base \
-ksp_max_it 10 -ksp_gmres_unmodifiedgramschmidt -optionsleft \
-snes_max_it 1 -snes_no_convergence_test -max_st 200 -matrix_free -pc_type bjacobi \
-sub_pc_type ilu -sub_pc_ilu_levels 0 -sub_pc_ilu_in_place \
-no_edge_reordering \
-log_summary -get_resident_set_size \
> out.1.base_noer

runflow2:
	@$(MPIRUN) -maxtime 60 -np 2 flow \
-ksp_rtol 0.01 \
-ksp_max_it 20 -ksp_gmres_unmodifiedgramschmidt -optionsleft \
-snes_max_it 1 -snes_no_convergence_test -max_st 100 -matrix_free -pc_type bjacobi \
-sub_pc_type ilu -sub_pc_ilu_levels 0 -sub_pc_ilu_in_place \
-log_summary \
> out.2 

runflow4:
	@$(MPIRUN) -np 4 flow \
-ksp_rtol 0.01 \
-cfl_ini 20.0 -cfl_max 100000.0 -first_order_it 10 \
-ksp_max_it 50 -ksp_gmres_unmodifiedgramschmidt -optionsleft \
-snes_max_it 1 -snes_no_convergence_test -max_st 100 -matrix_free -pc_type bjacobi \
-sub_pc_type ilu -sub_pc_ilu_levels 0 -sub_pc_ilu_in_place \
-print -ksp_monitor -snes_monitor \
-log_summary 
#> out.4

runflow8:
	@$(MPIRUN) -maxtime 15 -np 8 flow \
-ksp_rtol 0.01 \
-cfl_ini 20.0 -cfl_max 100000.0 -first_order_it 10 \
-ksp_max_it 50 -ksp_gmres_unmodifiedgramschmidt -optionsleft \
-snes_max_it 1 -snes_no_convergence_test -max_st 100 -matrix_free -pc_type bjacobi \
-sub_pc_type ilu -sub_pc_ilu_levels 0 -sub_pc_ilu_in_place \
-print -ksp_monitor -snes_monitor \
-log_summary \
> out.8

runflow12:
	@$(MPIRUN) -maxtime 40 -np 12 flow \
-ksp_rtol 0.01 \
-ksp_max_it 20 -ksp_gmres_unmodifiedgramschmidt -optionsleft \
-snes_max_it 1 -snes_no_convergence_test -max_st 100 -matrix_free -pc_type bjacobi \
-sub_pc_type ilu -sub_pc_ilu_levels 0 -sub_pc_ilu_in_place \
-log_summary \
> out.12

runflow16:
	@$(MPIRUN) -maxtime 10 -np 16 flow \
-ksp_rtol 0.0001 -ksp_max_it 45 -ksp_gmres_restart 16 \
-ksp_gmres_unmodifiedgramschmidt -optionsleft \
-snes_max_it 1 -snes_no_convergence_test -max_st 300 -matrix_free -pc_type bjacobi \
-sub_pc_type ilu -sub_pc_ilu_levels 0 -sub_pc_ilu_in_place \
-print -ksp_monitor -snes_monitor \
-log_summary \
> out.16

runflow24:
	@$(MPIRUN) -maxtime 40 -np 24 flow \
-ksp_rtol 0.01 \
-ksp_max_it 20 -ksp_gmres_unmodifiedgramschmidt -optionsleft \
-snes_max_it 1 -snes_no_convergence_test -max_st 100 -matrix_free -pc_type bjacobi \
-sub_pc_type ilu -sub_pc_ilu_levels 0 -sub_pc_ilu_in_place \
-log_summary \
> out.24

runflow32:
	@$(MPIRUN) -np 32 flow \
-ksp_rtol 0.0001 -ksp_max_it 45 -ksp_gmres_restart 16 \
-ksp_gmres_unmodifiedgramschmidt -optionsleft \
-snes_max_it 1 -snes_no_convergence_test -max_st 200 -matrix_free -pc_type bjacobi \
-sub_pc_type ilu -sub_pc_ilu_levels 0 -sub_pc_ilu_in_place \
-log_summary \
> out.32

runflow40:
	@$(MPIRUN) -np 40 flow \
-ksp_rtol 0.0001 -ksp_max_it 45 -ksp_gmres_restart 16 \
-ksp_gmres_unmodifiedgramschmidt -optionsleft \
-snes_max_it 1 -snes_no_convergence_test -max_st 200 -matrix_free -pc_type bjacobi \
-sub_pc_type ilu -sub_pc_ilu_levels 0 -sub_pc_ilu_in_place \
-log_summary \
> out.40

runflow48:
	@$(MPIRUN) -np 48 flow \
-ksp_rtol 0.0001 -ksp_max_it 45 -ksp_gmres_restart 16 \
-ksp_gmres_unmodifiedgramschmidt -optionsleft \
-snes_max_it 1 -snes_no_convergence_test -max_st 200 -matrix_free -pc_type bjacobi \
-sub_pc_type ilu -sub_pc_ilu_levels 0 -sub_pc_ilu_in_place \
-print -ksp_monitor -snes_monitor \
-log_summary \
> out.48

runflow64:
	@$(MPIRUN) -np 64 flow \
-ksp_rtol 0.0001 -ksp_max_it 45 -ksp_gmres_restart 16 \
-ksp_gmres_unmodifiedgramschmidt -optionsleft \
-snes_max_it 1 -snes_no_convergence_test -max_st 200 -matrix_free -pc_type bjacobi \
-sub_pc_type ilu -sub_pc_ilu_levels 0 -sub_pc_ilu_in_place \
-print -ksp_monitor -snes_monitor \
-log_summary \
> out.64

runflow80:
	@$(MPIRUN) -np 80 flow \
-ksp_rtol 0.0001 -ksp_max_it 45 -ksp_gmres_restart 16 \
-ksp_gmres_unmodifiedgramschmidt -optionsleft \
-snes_max_it 1 -snes_no_convergence_test -max_st 300 -matrix_free -pc_type bjacobi \
-sub_pc_type ilu -sub_pc_ilu_levels 0 -sub_pc_ilu_in_place \
-print -ksp_monitor -snes_monitor \
-log_summary \
> out.80

runflow96:
	@$(MPIRUN) -np 96 flow \
-ksp_rtol 0.0001 -ksp_max_it 45 -ksp_gmres_restart 16 \
-ksp_gmres_unmodifiedgramschmidt -optionsleft \
-snes_max_it 1 -snes_no_convergence_test -max_st 200 -matrix_free -pc_type bjacobi \
-sub_pc_type ilu -sub_pc_ilu_levels 0 -sub_pc_ilu_in_place \
-print -ksp_monitor -snes_monitor \
-log_summary \
> out.96

runflow112:
	@$(MPIRUN) -np 112 flow \
-ksp_rtol 0.0001 -ksp_max_it 45 -ksp_gmres_restart 16 \
-ksp_gmres_unmodifiedgramschmidt -optionsleft \
-snes_max_it 1 -snes_no_convergence_test -max_st 200 -matrix_free -pc_type bjacobi \
-sub_pc_type ilu -sub_pc_ilu_levels 0 -sub_pc_ilu_in_place \
-log_summary \
> out.112

runflow120:
	@$(MPIRUN) -np 120 flow \
-ksp_rtol 0.0001 -ksp_max_it 45 -ksp_gmres_restart 16 \
-ksp_gmres_unmodifiedgramschmidt -optionsleft \
-snes_max_it 1 -snes_no_convergence_test -max_st 200 -matrix_free -pc_type bjacobi \
-sub_pc_type ilu -sub_pc_ilu_levels 0 -sub_pc_ilu_in_place \
-log_summary \
> out.120

runflow128:
	@$(MPIRUN) -np 128 flow \
-ksp_rtol 0.0001 -ksp_max_it 45 -ksp_gmres_restart 16 \
-ksp_gmres_unmodifiedgramschmidt -optionsleft \
-snes_max_it 1 -snes_no_convergence_test -max_st 200 -matrix_free -pc_type bjacobi \
-sub_pc_type ilu -sub_pc_ilu_levels 0 -sub_pc_ilu_in_place \
-log_summary \
> out.128

tags:
	etags -f TAGS *.F *.c *.h

#.F.o:
#	-${FC} -c ${FOPTFLAGS} ${FFLAGS} ${CPPFLAGS} ${USERCPPFLAGS} $*.F











