
static char help[] = "Model single-physics solver. Modified from ex19.c \n\\n";

/* ------------------------------------------------------------------------

    See ex19.c for discussion of the problem 

  ------------------------------------------------------------------------- */
#include "mp.h"

extern PetscErrorCode FormInitialGuess(DMMG,Vec);
extern PetscErrorCode FormFunction(SNES,Vec,Vec,void*);
extern PetscLogEvent  EVENT_FORMFUNCTIONLOCAL1;

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  DMMG           *dmmg;               /* multilevel grid structure */
  AppCtx         user;                /* user-defined work context */
  PetscInt       mx,my,its;
  PetscErrorCode ierr;
  MPI_Comm       comm;
  SNES           snes;
  DM             da1;

  PetscInitialize(&argc,&argv,(char *)0,help);
  ierr = PetscLogEventRegister("FormFunc1", 0,&EVENT_FORMFUNCTIONLOCAL1);CHKERRQ(ierr);
  comm = PETSC_COMM_WORLD;

  /* Problem parameters (velocity of lid, prandtl, and grashof numbers) */
  ierr = PetscOptionsGetReal(PETSC_NULL,"-lidvelocity",&user.lidvelocity,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(PETSC_NULL,"-prandtl",&user.prandtl,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(PETSC_NULL,"-grashof",&user.grashof,PETSC_NULL);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create user context, set problem data, create vector data structures.
     Also, compute the initial guess.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Setup Physics 1: 
        - Lap(U) - Grad_y(Omega) = 0
	- Lap(V) + Grad_x(Omega) = 0
	- Lap(Omega) + Div([U*Omega,V*Omega]) - GR*Grad_x(T) = 0
        where T is given by the given x.temp
        - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMDACreate2d(comm,DMDA_BOUNDARY_NONE,DMDA_BOUNDARY_NONE,DMDA_STENCIL_STAR,-4,-4,PETSC_DECIDE,PETSC_DECIDE,3,1,0,0,&da1);CHKERRQ(ierr);
  ierr = DMDASetFieldName(da1,0,"x-velocity");CHKERRQ(ierr);
  ierr = DMDASetFieldName(da1,1,"y-velocity");CHKERRQ(ierr);
  ierr = DMDASetFieldName(da1,2,"Omega");CHKERRQ(ierr);

  /* Create the solver object and attach the grid/physics info */
  ierr = DMMGCreate(comm,1,&user,&dmmg);CHKERRQ(ierr);
  ierr = DMMGSetDM(dmmg,da1);CHKERRQ(ierr);
  ierr = DMMGSetISColoringType(dmmg,IS_COLORING_GLOBAL);CHKERRQ(ierr);

  ierr = DMMGSetInitialGuess(dmmg,FormInitialGuess);CHKERRQ(ierr);
  ierr = DMMGSetSNES(dmmg,FormFunction,0);CHKERRQ(ierr);
  ierr = DMMGSetFromOptions(dmmg);CHKERRQ(ierr);

  ierr = DMDAGetInfo(da1,PETSC_NULL,&mx,&my,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  user.lidvelocity = 1.0/(mx*my);
  user.prandtl     = 1.0;
  user.grashof     = 1.0;

  /* Solve the nonlinear system */
  ierr = DMMGSolve(dmmg);CHKERRQ(ierr); 
  snes = DMMGGetSNES(dmmg);
  ierr = SNESGetIterationNumber(snes,&its);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"Physics 1: Number of SNES iterations = %D\n\n", its);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free spaces 
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMDestroy(&da1);CHKERRQ(ierr);
  ierr = DMMGDestroy(dmmg);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "FormInitialGuess"
PetscErrorCode FormInitialGuess(DMMG dmmg,Vec X)
{
  PetscErrorCode ierr;
  DM             da1 = dmmg->dm;
  Field1         **x1;
  DMDALocalInfo    info1;

  PetscFunctionBegin;

  /* Access the array inside of X */
  ierr = DMDAVecGetArray(da1,X,(void**)&x1);CHKERRQ(ierr);

  ierr = DMDAGetLocalInfo(da1,&info1);CHKERRQ(ierr);

  /* Evaluate local user provided function */
  ierr = FormInitialGuessLocal1(&info1,x1);CHKERRQ(ierr);

  ierr = DMDAVecRestoreArray(da1,X,(void**)&x1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormFunction"
PetscErrorCode FormFunction(SNES snes,Vec X,Vec F,void *ctx)
{
  PetscErrorCode ierr;
  DMMG           dmmg = (DMMG)ctx;
  AppCtx         *user = (AppCtx*)dmmg->user;
  DM             da1 = dmmg->dm;
  DMDALocalInfo    info1;
  Field1         **x1,**f1;
  Vec            X1;

  PetscFunctionBegin;

  ierr = DMDAGetLocalInfo(da1,&info1);CHKERRQ(ierr);

  /* Get local vectors to hold ghosted parts of X */
  ierr = DMGetLocalVector(da1,&X1);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da1,X,INSERT_VALUES,X1);CHKERRQ(ierr); 
  ierr = DMGlobalToLocalEnd(da1,X,INSERT_VALUES,X1);CHKERRQ(ierr); 

  /* Access the arrays inside X1 */
  ierr = DMDAVecGetArray(da1,X1,(void**)&x1);CHKERRQ(ierr);

  /* Access the subvectors in F. 
     These are not ghosted so directly access the memory locations in F */
  ierr = DMDAVecGetArray(da1,F,(void**)&f1);CHKERRQ(ierr);

  /* Evaluate local user provided function */    
  ierr = FormFunctionLocal1(&info1,x1,0,f1,(void**)user);CHKERRQ(ierr);


  ierr = DMDAVecRestoreArray(da1,F,(void**)&f1);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da1,X1,(void**)&x1);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da1,&X1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

