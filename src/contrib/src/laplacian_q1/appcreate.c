
#include "appctx.h"

/*
         -  Generates the "global" parallel vector to contain the 
	    right hand side and solution.
         -  Generates "ghosted" local vectors for local computations etc.
         -  Generates scatter context for updating ghost points etc.
*/
#undef __FUNC__
#define __FUNC__ "AppCxtCreateRhs"
int AppCtxCreateRhs(AppCtx *appctx)
{
  AppGrid                *grid = &appctx->grid;
  AppAlgebra             *algebra = &appctx->algebra;
  MPI_Comm               comm = appctx->comm;
  int ierr;

  PetscFunctionBegin;
  /*  Create vector to contain load,  local size should be number of  vertices  on this proc.  */
  ierr = VecCreateMPI(comm,grid->vertex_local_count,PETSC_DECIDE,&algebra->b);CHKERRQ(ierr);
  ierr = VecSetLocalToGlobalMapping(algebra->b,grid->ltog);CHKERRQ(ierr);

  ierr = VecDuplicate(algebra->b, &algebra->x);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


#undef __FUNC__
#define __FUNC__ "AppCxtCreateMatrix"
int AppCtxCreateMatrix(AppCtx* appctx)
{
/********* Collect context informatrion ***********/
  AppAlgebra             *algebra = &appctx->algebra;
  AppGrid                *grid    = &appctx->grid;
  MPI_Comm               comm = appctx->comm;
int ierr; 
  PetscFunctionBegin;
  /* now create the matrix */
  /* using very rough estimate for nonzeros on and off the diagonal */
  ierr = MatCreateMPIAIJ(comm, grid->vertex_local_count, grid->vertex_local_count, PETSC_DETERMINE, PETSC_DETERMINE, 9 , 0, 3 , 0, &algebra->A); CHKERRQ(ierr);
  /* Set the local to global mapping */
  ierr = MatSetLocalToGlobalMapping(algebra->A, grid->ltog);  CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

