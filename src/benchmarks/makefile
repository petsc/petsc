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
EXAMPLES_2    = 
RUNEXAMPLES_2 = 
EXAMPLES_3    = 
RUNEXAMPLES_3 = 
EXAMPLES_4    = 
RUNEXAMPLES_4 = 
EXAMPLESC     = 
EXAMPLESF     = 
TESTS         = PetscTime PetscGetTime MPI_Wtime PLogEvent PetscMalloc \
		PetscMemcpy PetscMemzero PetscMemcmp Index

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

PetscMemcpy: PetscMemcpy.o  chkopts 
	-$(CLINKER) -o PetscMemcpy PetscMemcpy.o $(PETSC_LIB)
	$(RM) -f PetscMemcpy.o

PetscMemzero: PetscMemzero.o  chkopts 
	-$(CLINKER) -o PetscMemzero PetscMemzero.o $(PETSC_LIB)
	$(RM) -f PetscMemzero.o

PetscMemcmp: PetscMemcmp.o  chkopts 
	-$(CLINKER) -o PetscMemcmp PetscMemcmp.o $(PETSC_LIB)
	$(RM) -f PetscMemcmp.o

Index: Index.o  chkopts 
	-$(CLINKER) -o Index Index.o $(PETSC_LIB)
	$(RM) -f Index.o


test: $(TESTS)

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
	-@$(MPIRUN) PetscMalloc            > /dev/null
	-@$(MPIRUN) PetscMalloc -trmalloc  > /dev/null
	-@echo " "
	-@echo "Memory Operations "
	-@echo "------------------------------------------------"
	-@$(MPIRUN) PetscMemcpy  > /dev/null
	-@$(MPIRUN) PetscMemzero > /dev/null
	-@$(MPIRUN) PetscMemcmp  > /dev/null
	-@echo " "
	-@echo "Memory Operations "
	-@echo "------------------------------------------------"
	-@$(MPIRUN) Index  > /dev/null

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)

