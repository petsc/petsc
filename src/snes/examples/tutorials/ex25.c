static const char help[] ="Minimum surface problem in 2D.\n\
Uses 2-dimensional distributed arrays.\n\
\n\
  Solves the linear systems via multilevel methods \n\
\n\n";

/*T
   Concepts: SNES^solving a system of nonlinear equations
   Concepts: DMDA^using distributed arrays
   Concepts: multigrid;
   Processors: n
T*/

/*  
  
    This example models the partial differential equation 
   
         - Div((1 + ||GRAD T||^2)^(1/2) (GRAD T)) = 0.
       
    
    in the unit square, which is uniformly discretized in each of x and 
    y in this simple encoding.  The degrees of freedom are vertex centered
 
    A finite difference approximation with the usual 5-point stencil 
    is used to discretize the boundary value problem to obtain a 
    nonlinear system of equations. 
 
*/

#include <petscsnes.h>
#include <petscdmda.h>
#include <petscpcmg.h>

extern PetscErrorCode FormFunctionLocal(DMDALocalInfo*,PetscScalar**,PetscScalar**,void*);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  SNES           snes;                      
  PetscErrorCode ierr;
  PetscInt       its,lits;
  PetscReal      litspit;
  DM             da;

  PetscInitialize(&argc,&argv,PETSC_NULL,help);

  /*
      Set the DMDA (grid structure) for the grids.
  */
  ierr = DMDACreate2d(PETSC_COMM_WORLD, DMDA_BOUNDARY_NONE, DMDA_BOUNDARY_NONE,DMDA_STENCIL_STAR,-5,-5,PETSC_DECIDE,PETSC_DECIDE,1,1,0,0,&da);CHKERRQ(ierr);
  ierr = DMDASetLocalFunction(da,(DMDALocalFunction1)FormFunctionLocal);CHKERRQ(ierr);
  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
  ierr = SNESSetDM(snes,da);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);

  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  ierr = SNESSolve(snes,0,0);CHKERRQ(ierr);
  ierr = SNESGetIterationNumber(snes,&its);CHKERRQ(ierr);
  ierr = SNESGetLinearSolveIterations(snes,&lits);CHKERRQ(ierr);
  litspit = ((PetscReal)lits)/((PetscReal)its);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of SNES iterations = %D\n",its);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of Linear iterations = %D\n",lits);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Average Linear its / SNES = %e\n",litspit);CHKERRQ(ierr);

  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = PetscFinalize();

  return 0;
}

#undef __FUNCT__  
#define __FUNCT__ "FormFunctionLocal"
PetscErrorCode FormFunctionLocal(DMDALocalInfo *info,PetscScalar **t,PetscScalar **f,void *ptr)
{
  PetscInt     i,j;
  PetscScalar  hx,hy;
  PetscScalar  gradup,graddown,gradleft,gradright,gradx,grady;
  PetscScalar  coeffup,coeffdown,coeffleft,coeffright;

  PetscFunctionBegin;
  hx    = 1.0/(PetscReal)(info->mx-1);  hy    = 1.0/(PetscReal)(info->my-1);
 
  /* Evaluate function */
  for (j=info->ys; j<info->ys+info->ym; j++) {
    for (i=info->xs; i<info->xs+info->xm; i++) {

      if (i == 0 || i == info->mx-1 || j == 0 || j == info->my-1) {

        f[j][i] = t[j][i] - (1.0 - (2.0*hx*(PetscReal)i - 1.0)*(2.0*hx*(PetscReal)i - 1.0));
      
      } else {

        gradup     = (t[j+1][i] - t[j][i])/hy;
        graddown   = (t[j][i] - t[j-1][i])/hy;
        gradright  = (t[j][i+1] - t[j][i])/hx;
        gradleft   = (t[j][i] - t[j][i-1])/hx;

        gradx      = .5*(t[j][i+1] - t[j][i-1])/hx;
        grady      = .5*(t[j+1][i] - t[j-1][i])/hy;

        coeffup    = 1.0/PetscSqrtScalar(1.0 + gradup*gradup + gradx*gradx); 
        coeffdown  = 1.0/PetscSqrtScalar(1.0 + graddown*graddown + gradx*gradx); 

        coeffleft  = 1.0/PetscSqrtScalar(1.0 + gradleft*gradleft + grady*grady); 
        coeffright = 1.0/PetscSqrtScalar(1.0 + gradright*gradright + grady*grady); 

        f[j][i] = (coeffup*gradup - coeffdown*graddown)*hx + (coeffright*gradright - coeffleft*gradleft)*hy; 
    
      }

    }
  }
  PetscFunctionReturn(0);
} 
