PETSC_DIR = ../..
ALL: 

CFLAGS	      =	 $(PETSC_INCLUDE) $(COPT) $(PCONF)
FFLAGS	      =	 
SOURCEC	      = 
SOURCEF	      =
SOURCEH	      =
OBJSC	      =
OBJSF	      =
LIBBASE	      = libpetscsys
EXAMPLES_1    = 
RUNEXAMPLES_1 = 
EXAMPLES_2    = PetscTime PetscGetTime MPI_Wtime PLogEvent PetscMalloc
RUNEXAMPLES_2 = 
EXAMPLES_3    = 
RUNEXAMPLES_3 = 
EXAMPLES_4    = 
RUNEXAMPLES_4 = 
EXAMPLESC     = 
EXAMPLESF     = 

PetscTime: PetscTime.o  chkopts 
	-$(CLINKER) -o PetscTime PetscTime.o $(PETSC_LIB)
	$(RM) -f PetscTime.o

PetscGetTime: PetscGetTime.o  chkopts 
	-$(CLINKER) -o PetscGetTime PetscGetTime.o $(PETSC_LIB)
	$(RM) -f PetscGetTime.o

MPI_Wtime: MPI_Wtime.o  chkopts 
	-$(CLINKER) -o MPI_Wtime MPI_Wtime.o $(PETSC_LIB)
	$(RM) -f MPI_Wtime.o

PLogEvent: PLogEvent.o  chkopts 
	-$(CLINKER) -o PLogEvent PLogEvent.o $(PETSC_LIB)
	$(RM) -f PLogEvent.o

PetscMalloc: PetscMalloc.o  chkopts 
	-$(CLINKER) -o PetscMalloc PetscMalloc.o $(PETSC_LIB)
	$(RM) -f PetscMalloc.o

test: $(EXAMPLES_2)

runtest: 
	-@echo "Time Taken by some PETSc routines are as follows:"
	-@echo "------------------------------------------------"
	-@$(MPIRUN) PetscTime    > /dev/null
	-@$(MPIRUN) PetscGetTime > /dev/null
	-@$(MPIRUN) MPI_Wtime    > /dev/null
	-@echo " "
	-@echo "PLogEventBegin and PLogEventEnd together with options"
	-@echo "------------------------------------------------"
	-@$(MPIRUN) PLogEvent                    > /dev/null
	-@$(MPIRUN) PLogEvent -log               > /dev/null
	-@$(MPIRUN) PLogEvent -log_summary       > /dev/null
	-@$(MPIRUN) PLogEvent -log_all           > /dev/null
	-@$(MPIRUN) PLogEvent -log_mpe           > /dev/null
	-@$(MPIRUN) PLogEvent -log_mpe -log      > /dev/null
	-@echo " "
	-@echo "PetscMalloc and PetscFree together  with options"
	-@echo "------------------------------------------------"
	-@$(MPIRUN) PetscMalloc                    > /dev/null
	-@$(MPIRUN) PetscMalloc -trmalloc          > /dev/null


include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)

