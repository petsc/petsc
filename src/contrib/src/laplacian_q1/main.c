
static char help[] ="Solves 2d-laplacian on quadrilateral grid.\n  Generate grid using input.\n  Options:\n  -matlab_graphics\t pipe solution to matlab (visualize with bscript).\n -show_is\t print the local index sets and local to global mappings (for use with >1 processor).\n  -show_ao\t print the contents of the ao database.\n  -show_matrix\t visualize the sparsity structure of the stiffness matrix.\n See README file for more information.\n";

#include "appctx.h"

int main( int argc, char **argv )
{
  int            ierr;
  AppCtx         *appctx;

  /* ---------------------------------------------------------------------
     Initialize PETSc
     --------------------- ---------------------------------------------------*/
  PetscInitialize(&argc,&argv,(char *)0,help);

  /*  Load the grid database*/
  ierr = AppCtxCreate(PETSC_COMM_WORLD,&appctx); CHKERRA(ierr);

 /* setup the graphics routines to view the grid  */
  ierr = AppCtxGraphics(appctx); CHKERRA(ierr);
 
  /*   Setup the linear system and solve it*/
  ierr = AppCtxSolve(appctx);CHKERRQ(ierr);

  /* Send to  matlab viewer */
  if (appctx->view.matlabgraphics) {    AppCtxViewMatlab(appctx);  }

  /*      Destroy all datastructures  */
  ierr = AppCtxDestroy(appctx); CHKERRA(ierr);

  PetscFinalize();
  PetscFunctionReturn(0);
}



/*
         Sets up the linear system associated with the PDE and solves it
*/
#undef __FUNC__
#define __FUNC__ "AppCxtSolve"
int AppCtxSolve(AppCtx* appctx)
{
  AppGrid                *grid = &appctx->grid;
  AppAlgebra             *algebra = &appctx->algebra;
  MPI_Comm               comm = appctx->comm;
  SLES                   sles;
  int ierr, its;

  PetscFunctionBegin;

  /*        Create vector to contain load and various work vectors  */
  ierr = AppCtxCreateRhs(appctx); CHKERRQ(ierr);

  /*      Create the sparse matrix, with correct nonzero pattern  */
  ierr = AppCtxCreateMatrix(appctx); CHKERRQ(ierr);

  /*     Set the quadrature values for the reference square element  */
  ierr = SetReferenceElement(appctx);CHKERRQ(ierr);

  /*      Set the right hand side values into the vectors   */
  ierr = AppCtxSetRhs(appctx); CHKERRQ(ierr);

  /*      Set the rhs boundary conditions */
  ierr = SetBoundaryConditions(appctx); CHKERRQ(ierr);

  /*      Set the matrix entries   */
  ierr = AppCtxSetMatrix(appctx); CHKERRQ(ierr);

  /* view sparsity structure of the matrix */
  if( appctx->view.show_matrix ) {  
    printf("The stiffness matrix, before bc applied\n");
    ierr = MatView(appctx->algebra.A, VIEWER_DRAWX_WORLD );CHKERRQ(ierr);
  }

  /*      Set the matrix boundary conditions */
  ierr = SetMatrixBoundaryConditions(appctx); CHKERRQ(ierr);

  /* view sparsity structure of the matrix */
  if( appctx->view.show_matrix ) {  
    printf("The stiffness matrix, after bc applied\n");
    ierr = MatView(appctx->algebra.A, VIEWER_DRAWX_WORLD );CHKERRQ(ierr);
  }
  
  /*       Solve the linear system  */
  ierr = SLESCreate(comm,&sles);CHKERRQ(ierr);
  ierr = SLESSetOperators(sles,algebra->A,algebra->A,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = SLESSetFromOptions(sles);CHKERRQ(ierr);
  ierr = SLESSolve(sles,algebra->b,algebra->x,&its);CHKERRQ(ierr);
  ierr = SLESDestroy(sles); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}




