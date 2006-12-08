

#include "petsc.h"        /*I  "petsc.h"   I*/
#include "petscsys.h"

static MPI_Comm saved_PETSC_COMM_WORLD = 0;
MPI_Comm PETSC_COMM_LOCAL_WORLD        = 0;        /* comm for a single node (local set of processes) */
static PetscTruth used_PetscOpenMP     = PETSC_FALSE;

#undef __FUNCT__  
#define __FUNCT__ "PetscOpenMPInitialize"
/*@C
   PetscOpenMPInitialize - Initializes the PETSc and MPI to work with OpenMP 
      PetscMPInitialize() calls MPI_Init() if that has yet to be called,
      so this routine should always be called near the beginning of 
      your program -- usually the very first line! 

   Collective on MPI_COMM_WORLD or PETSC_COMM_WORLD if it has been set

   Input Parameter:
.  nodesize - size of each compute node that will share processors

   Level: developer

   Concepts: OpenMP
   
.seealso: PetscFinalize(), PetscInitializeFortran(), PetscGetArgs(), PetscOpenMPFinalize(), PetscInitialize()

@*/
PetscErrorCode PETSC_DLLEXPORT PetscOpenMPInitialize(PetscMPIInt nodesize)
{
  PetscErrorCode ierr;
  PetscMPIInt    size,rank,*ranks,i;
  MPI_Group      group,newgroup;

  PetscFunctionBegin;
  saved_PETSC_COMM_WORLD = PETSC_COMM_WORLD;

  ierr = MPI_Comm_size(saved_PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size % nodesize) SETERRQ2(PETSC_ERR_ARG_SIZ,"Total number of process nodes %d is not divisible by number of processes per node %d",size,nodesize);
  ierr = MPI_Comm_rank(saved_PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);


  /* create two communicators 
      *) one that contains the first process from each node: 0,nodesize,2*nodesize,...
      *) one that contains all processes in a node:  (0,1,2...,nodesize-1), (nodesize,nodesize+1,...2*nodesize-), ...
  */
  ierr = MPI_Comm_group(saved_PETSC_COMM_WORLD,&group);CHKERRQ(ierr);
  ierr = PetscMalloc((size/nodesize)*sizeof(PetscMPIInt),&ranks);CHKERRQ(ierr);
  for (i=0; i<(size/nodesize); i++) ranks[i] = i*nodesize;
  ierr = MPI_Group_incl(group,size/nodesize,ranks,&newgroup);CHKERRQ(ierr);
  ierr = PetscFree(ranks);CHKERRQ(ierr);
  ierr = MPI_Comm_create(saved_PETSC_COMM_WORLD,newgroup,&PETSC_COMM_WORLD);CHKERRQ(ierr);
  if (rank % nodesize) PETSC_COMM_WORLD = 0; /* mark invalid processes for easy debugging */
  ierr = MPI_Group_free(&group);CHKERRQ(ierr);
  ierr = MPI_Group_free(&newgroup);CHKERRQ(ierr);

  ierr = MPI_Comm_split(saved_PETSC_COMM_WORLD,rank/nodesize,rank % nodesize,&PETSC_COMM_LOCAL_WORLD);CHKERRQ(ierr);

  ierr = PetscInfo2(0,"PETSc OpenMP successfully started: number of nodes = %d node size = %d\n",size/nodesize,nodesize);CHKERRQ(ierr);
  ierr = PetscInfo1(0,"PETSc OpenMP process %sactive\n",(rank % nodesize) ? "in" : "");CHKERRQ(ierr);

  used_PetscOpenMP = PETSC_TRUE;
  /* 
     All process not involved in user application code wait here
  */
  if (!PETSC_COMM_WORLD) {
    ierr = PetscEnd();  /* cannot continue into user code */
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscOpenMPFinalize"
/*@C
   PetscOpenMPFinalizes - Finalizes the PETSc and MPI to work with OpenMP. Called by PetscFinalize() cannot
       be called by user.

   Collective on the entire system

   Level: developer
           
.seealso: PetscFinalize(), PetscInitializeFortran(), PetscGetArgs(), PetscOpenMPInitialize()

@*/
PetscErrorCode PETSC_DLLEXPORT PetscOpenMPFinalize(void)
{
  PetscErrorCode ierr = 0;

  PetscFunctionBegin;
  if (!used_PetscOpenMP) PetscFunctionReturn(0);

  ierr = MPI_Barrier(saved_PETSC_COMM_WORLD);CHKERRQ(ierr);
  PETSC_COMM_WORLD = saved_PETSC_COMM_WORLD;
  PetscFunctionReturn(ierr);
}
