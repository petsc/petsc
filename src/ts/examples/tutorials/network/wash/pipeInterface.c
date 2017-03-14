#include "pipe.h"

/* Subroutines for Pipe                                  */
/* -------------------------------------------------------*/

/*
   PipeCreate - Create Pipe object.

   Input Parameters:
   comm - MPI communicator
   
   Output Parameter:
.  pipe - location to put the PIPE context
*/
PetscErrorCode PipeCreate(MPI_Comm comm,Pipe *pipe)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNew(pipe);CHKERRQ(ierr);
  (*pipe)->comm = comm;
  PetscFunctionReturn(0);
}

/*
   PipeDestroy - Destroy Pipe object.

   Input Parameters:
   pipe - Reference to pipe intended to be destroyed.
*/
PetscErrorCode PipeDestroy(Pipe *pipe)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*pipe) PetscFunctionReturn(0);
  
  ierr = VecDestroy(&(*pipe)->x);CHKERRQ(ierr);
  ierr = DMDestroy(&(*pipe)->da);CHKERRQ(ierr);
  ierr = PetscFree(*pipe);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   PipeSetParameters - Set parameters for Pipe context

   Input Parameter:
+  pipe - PIPE object
.  length - 
.  nnodes -
.  D - 
.  a -
-  fric -
*/
PetscErrorCode PipeSetParameters(Pipe pipe,PetscReal length,PetscInt nnodes,PetscReal D,PetscReal a,PetscReal fric)
{
  PetscFunctionBegin;
  pipe->length = length;
  pipe->nnodes = nnodes;
  pipe->D      = D;   
  pipe->a      = a;
  pipe->fric   = fric;
  PetscFunctionReturn(0);
}

/*
    PipeSetUp - Set up pipe based on set parameters.
*/
PetscErrorCode PipeSetUp(Pipe pipe)
{
  DMDALocalInfo  info;
  PetscErrorCode ierr;
  MPI_Comm       comm = pipe->comm;
    
  PetscFunctionBegin;
  ierr = DMDACreate1d(comm, DM_BOUNDARY_GHOSTED, pipe->nnodes, 2, 1, NULL, &pipe->da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(pipe->da);CHKERRQ(ierr);
  ierr = DMSetUp(pipe->da);CHKERRQ(ierr);
  ierr = DMDASetFieldName(pipe->da, 0, "Q");CHKERRQ(ierr);
  ierr = DMDASetFieldName(pipe->da, 1, "H");CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(pipe->da, 0, pipe->length, 0, 0, 0, 0);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(pipe->da, &(pipe->x));CHKERRQ(ierr);

  ierr = DMDAGetLocalInfo(pipe->da, &info);CHKERRQ(ierr);

  pipe->rad = pipe->D / 2;
  pipe->A   = PETSC_PI*pipe->rad*pipe->rad;
  pipe->R   = pipe->fric / (2*pipe->D*pipe->A);
  PetscFunctionReturn(0);
}
