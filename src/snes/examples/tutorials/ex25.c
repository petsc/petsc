
static char help[] ="Minimum surface problem\n\
Uses 2-dimensional distributed arrays.\n\
\n\
  Solves the linear systems via multilevel methods \n\
\n\n";

/*T
   Concepts: SNES^solving a system of nonlinear equations
   Concepts: DA^using distributed arrays
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

#include "petscsnes.h"
#include "petscda.h"
#include "petscmg.h"
#include "petscdmmg.h"

extern PetscErrorCode FormFunction(SNES,Vec,Vec,void*);
extern PetscErrorCode FormFunctionLocal(DALocalInfo*,PetscScalar**,PetscScalar**,void*);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  DMMG           *dmmg;
  SNES           snes;                      
  PetscErrorCode ierr;
  PetscInt       its,lits;
  PetscReal      litspit;
  DA             da;

  PetscInitialize(&argc,&argv,PETSC_NULL,help);


  /*
      Create the multilevel DA data structure 
  */
  ierr = DMMGCreate(PETSC_COMM_WORLD,3,0,&dmmg);CHKERRQ(ierr);

  /*
      Set the DA (grid structure) for the grids.
  */
  ierr = DACreate2d(PETSC_COMM_WORLD,DA_NONPERIODIC,DA_STENCIL_STAR,-5,-5,PETSC_DECIDE,PETSC_DECIDE,1,1,0,0,&da);CHKERRQ(ierr);
  ierr = DMMGSetDM(dmmg,(DM)da);CHKERRQ(ierr);
  ierr = DADestroy(da);CHKERRQ(ierr);

  /*
       Process adiC(36): FormFunctionLocal FormFunctionLocali

     Create the nonlinear solver, and tell the DMMG structure to use it
  */
  /*  ierr = DMMGSetSNES(dmmg,FormFunction,0);CHKERRQ(ierr); */
  ierr = DMMGSetSNESLocal(dmmg,FormFunctionLocal,0,ad_FormFunctionLocal,0);CHKERRQ(ierr);
  ierr = DMMGSetFromOptions(dmmg);CHKERRQ(ierr);

  /*
      PreLoadBegin() means that the following section of code is run twice. The first time
     through the flag PreLoading is on this the nonlinear solver is only run for a single step.
     The second time through (the actually timed code) the maximum iterations is set to 10
     Preload of the executable is done to eliminate from the timing the time spent bring the 
     executable into memory from disk (paging in).
  */
  PreLoadBegin(PETSC_TRUE,"Solve");
    ierr = DMMGSolve(dmmg);CHKERRQ(ierr);
  PreLoadEnd();
  snes = DMMGGetSNES(dmmg);
  ierr = SNESGetIterationNumber(snes,&its);CHKERRQ(ierr);
  ierr = SNESGetLinearSolveIterations(snes,&lits);CHKERRQ(ierr);
  litspit = ((PetscReal)lits)/((PetscReal)its);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of Newton iterations = %D\n",its);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of Linear iterations = %D\n",lits);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Average Linear its / Newton = %e\n",litspit);CHKERRQ(ierr);

  ierr = DMMGDestroy(dmmg);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);

  return 0;
}
/* --------------------  Evaluate Function F(x) --------------------- */
#undef __FUNCT__
#define __FUNCT__ "FormFunction"
PetscErrorCode FormFunction(SNES snes,Vec T,Vec F,void* ptr)
{
  DMMG           dmmg = (DMMG)ptr;
  PetscErrorCode ierr;
  PetscInt       i,j,mx,my,xs,ys,xm,ym;
  PetscScalar    hx,hy;
  PetscScalar    **t,**f,gradup,graddown,gradleft,gradright,gradx,grady;
  PetscScalar    coeffup,coeffdown,coeffleft,coeffright;
  Vec            localT;

  PetscFunctionBegin;
  ierr = DAGetLocalVector((DA)dmmg->dm,&localT);CHKERRQ(ierr);
  ierr = DAGetInfo((DA)dmmg->dm,PETSC_NULL,&mx,&my,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  hx    = 1.0/(PetscReal)(mx-1);  hy    = 1.0/(PetscReal)(my-1);
 
  /* Get ghost points */
  ierr = DAGlobalToLocalBegin((DA)dmmg->dm,T,INSERT_VALUES,localT);CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd((DA)dmmg->dm,T,INSERT_VALUES,localT);CHKERRQ(ierr);
  ierr = DAGetCorners((DA)dmmg->dm,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);
  ierr = DAVecGetArray((DA)dmmg->dm,localT,&t);CHKERRQ(ierr);
  ierr = DAVecGetArray((DA)dmmg->dm,F,&f);CHKERRQ(ierr);

  /* Evaluate function */
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {

      if (i == 0 || i == mx-1 || j == 0 || j == my-1) {

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
  ierr = DAVecRestoreArray((DA)dmmg->dm,localT,&t);CHKERRQ(ierr);
  ierr = DAVecRestoreArray((DA)dmmg->dm,F,&f);CHKERRQ(ierr);
  ierr = DARestoreLocalVector((DA)dmmg->dm,&localT);CHKERRQ(ierr);
  PetscFunctionReturn(0);
} 

PetscErrorCode FormFunctionLocal(DALocalInfo *info,PetscScalar **t,PetscScalar **f,void *ptr)
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
