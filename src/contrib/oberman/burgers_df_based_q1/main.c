
static char help[] ="Solves the 2d Burgers equation. \n  u*du/dx + v*du/dy - c(lap(u)) = f. \n  u*dv/dx + v*dv/dy - c(lap(v)) = g.  This has exact solution, see Fletcher.\n  This version has new indexing of Degrees of Freedom";

#include "appctx.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  int            ierr;
  AppCtx         *appctx;
  AppAlgebra     *algebra;
 
  /* ---------------------------------------------------------------------
     Initialize PETSc
     --------------------- ---------------------------------------------------*/
  PetscInitialize(&argc,&argv,(char *)0,help);

  /*  Load the grid database*/
  ierr = AppCtxCreate(PETSC_COMM_WORLD,&appctx);CHKERRQ(ierr);

  /*      Initialize graphics  */
  ierr = AppCtxGraphics(appctx);CHKERRQ(ierr); 

  /*   Setup the nonlinear system and solve it*/
  ierr = AppCtxSolve(appctx);CHKERRQ(ierr);

  /*      Visualize solution   */
  if (appctx->view.show_solution) {
    algebra = &appctx->algebra;
    ierr = VecScatterBegin(algebra->g,algebra->f_local,INSERT_VALUES,SCATTER_FORWARD,algebra->dfgtol);CHKERRQ(ierr);
    ierr = VecScatterEnd(algebra->g,algebra->f_local,INSERT_VALUES,SCATTER_FORWARD,algebra->dfgtol);CHKERRQ(ierr);
    ierr = PetscDrawZoom(appctx->view.drawglobal,AppCtxViewSolution,appctx);CHKERRQ(ierr);
  }

  /* Send solution to  matlab viewer */
  if (appctx->view.matlabgraphics) {AppCtxViewMatlab(appctx);  }

  /*      Destroy all datastructures  */
  ierr = AppCtxDestroy(appctx);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

