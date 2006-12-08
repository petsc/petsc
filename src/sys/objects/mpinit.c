

#include "petsc.h"        /*I  "petsc.h"   I*/
#include "petscsys.h"

static MPI_Comm saved_PETSC_COMM_WORLD = 0;
MPI_Comm PETSC_COMM_LOCAL_WORLD        = 0;        /* comm for a single node (local set of processes) */

#undef __FUNCT__  
#define __FUNCT__ "PetscOpenMPInitialize"
/*@C
   PetscOpenMPInitialize - Initializes the PETSc and MPI to work with OpenMP 
      PetscMPInitialize() calls MPI_Init() if that has yet to be called,
      so this routine should always be called near the beginning of 
      your program -- usually the very first line! 

   Collective on MPI_COMM_WORLD or PETSC_COMM_WORLD if it has been set

   Input Parameters:
+  nodesize - size of each compute node that will share processors
.  argc - count of number of command line arguments
.  args - the command line arguments
.  file - [optional] PETSc database file, defaults to ~username/.petscrc
          (use PETSC_NULL for default)
-  help - [optional] Help message to print, use PETSC_NULL for no message

   If you wish PETSc to run on a subcommunicator of MPI_COMM_WORLD, create that
   communicator first and assign it to PETSC_COMM_WORLD BEFORE calling PetscInitialize()

   Options Database Keys:
.    see PetscInitialize() for all options


   Level: beginner

   Notes:
   If for some reason you must call MPI_Init() separately, call
   it before PetscInitialize().

   Fortran Version:
   In Fortran this routine has the format
$       call PetscOpenMPInitialize(nodesize,file,ierr)

+   ierr - error return code
-   file - [optional] PETSc database file name, defaults to 
           ~username/.petscrc (use PETSC_NULL_CHARACTER for default)
           
   Important Fortran Note:
   In Fortran, you MUST use PETSC_NULL_CHARACTER to indicate a
   null character string; you CANNOT just use PETSC_NULL as 
   in the C version.  See the users manual for details.


   Concepts: initializing PETSc
   
.seealso: PetscFinalize(), PetscInitializeFortran(), PetscGetArgs(), PetscOpenMPFinalize()

@*/
PetscErrorCode PETSC_DLLEXPORT PetscOpenMPInitialize(int nodesize,int *argc,char ***args,const char file[],const char help[])
{
  PetscErrorCode ierr;
  PetscMPIInt    size,rank,*ranks,i;
  MPI_Group      group,newgroup;

  PetscFunctionBegin;
  if (PetscInitializeCalled) PetscFunctionReturn(0);
  ierr = PetscInitialize(argc,args,file,help);if (ierr) PetscFunctionReturn(ierr);

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

  /* 
     All process not involved in user application code wait here
  */
  if (!PETSC_COMM_WORLD) {
    ierr = MPI_Barrier(saved_PETSC_COMM_WORLD);CHKERRQ(ierr);
    PETSC_COMM_WORLD = saved_PETSC_COMM_WORLD;
    ierr = PetscEnd();  /* cannot continue into user code */
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscOpenMPFinalize"
/*@C
   PetscOpenMPFinalizes - Finalizes the PETSc and MPI to work with OpenMP. Do not
     use any MPI or PETSc calls after this call.

   Collective on the entire system

   Options Database:
.      See PetscFinalize() for all options

   Level: beginner


   Fortran Version:
   In Fortran this routine has the format
$       call PetscOpenMPFinalize(ierr)

.   ierr - error return code
           
.seealso: PetscFinalize(), PetscInitializeFortran(), PetscGetArgs(), PetscOpenMPInitialize

@*/
PetscErrorCode PETSC_DLLEXPORT PetscOpenMPFinalize(void)
{
  PetscErrorCode ierr = 0;

  PetscFunctionBegin;

  if (!PETSC_COMM_WORLD) SETERRQ(PETSC_ERR_PLIB,"Non-user process should never reach here");

  ierr = MPI_Barrier(saved_PETSC_COMM_WORLD);CHKERRQ(ierr);
  PETSC_COMM_WORLD = saved_PETSC_COMM_WORLD;
  ierr = PetscFinalize();
  PetscFunctionReturn(ierr);
}
