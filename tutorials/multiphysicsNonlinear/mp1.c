
static char help[] = "Model nonlinear multi-physics solver. Modified from mp.c \n\\n";

/* ------------------------------------------------------------------------
    See ex19.c for discussion of the problem 

    Examples of command line options:
      ./mp -dmmg_jacobian_mf_fd_operator
      ./mp -dmcomposite_dense_jacobian #inefficient, but compute entire Jacobian for testing

      ./mp1 -snes_monitor -mp_max_it 14 -grashof 1000.0
  ----------------------------------------------------------------------------------------- */
#include <petsctime.h>
#include "mp1.h"

extern PetscErrorCode FormInitialGuessComp(DMMG,Vec);
extern PetscErrorCode FormFunctionComp(SNES,Vec,Vec,void*);

extern PetscErrorCode FormInitialGuess1(DMMG,Vec);
extern PetscErrorCode FormFunction1(SNES,Vec,Vec,void *);

extern PetscErrorCode FormInitialGuess2(DMMG,Vec);
extern PetscErrorCode FormFunction2(SNES,Vec,Vec,void *);


#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  DMMG           *dmmg_comp;          /* multilevel grid structure */
  AppCtx         user;                /* user-defined work context */
  PetscInt       mx,my,its,max_its,i;
  PetscErrorCode ierr;
  MPI_Comm       comm;
  SNES           snes;
  DM             da1,da2;
  DM             pack;

  DMMG           *dmmg1,*dmmg2;
  PetscBool      SolveSubPhysics=PETSC_FALSE,GaussSeidel=PETSC_TRUE,Jacobi=PETSC_FALSE;
  Vec            X1,X1_local,X2,X2_local;

  PetscInitialize(&argc,&argv,(char *)0,help);
  comm = PETSC_COMM_WORLD;

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
  ierr = DMMGCreate(comm,1,&user,&dmmg1);CHKERRQ(ierr);
  ierr = DMMGSetDM(dmmg1,da1);CHKERRQ(ierr);
  ierr = DMMGSetISColoringType(dmmg1,IS_COLORING_GLOBAL);CHKERRQ(ierr);

  ierr = DMMGSetInitialGuess(dmmg1,FormInitialGuess1);CHKERRQ(ierr);
  ierr = DMMGSetSNES(dmmg1,FormFunction1,0);CHKERRQ(ierr);
  ierr = DMMGSetFromOptions(dmmg1);CHKERRQ(ierr);

  /* Set problem parameters (velocity of lid, prandtl, and grashof numbers) */  
  ierr = DMDAGetInfo(da1,PETSC_NULL,&mx,&my,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  user.lidvelocity = 1.0/(mx*my);
  user.prandtl     = 1.0;
  user.grashof     = 1000.0; 
  ierr = PetscOptionsGetReal(PETSC_NULL,"-lidvelocity",&user.lidvelocity,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(PETSC_NULL,"-prandtl",&user.prandtl,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(PETSC_NULL,"-grashof",&user.grashof,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL,"-solvesubphysics",&SolveSubPhysics);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL,"-Jacobi",&Jacobi);CHKERRQ(ierr);
  if (Jacobi) GaussSeidel=PETSC_FALSE;
  
  ierr = PetscPrintf(comm,"grashof: %g, ",user.grashof);CHKERRQ(ierr);
  if (GaussSeidel){
    ierr = PetscPrintf(comm,"use Block Gauss-Seidel\n");CHKERRQ(ierr);
  } else {
    ierr = PetscPrintf(comm,"use Block Jacobi\n");CHKERRQ(ierr);
  }
  ierr = PetscPrintf(comm,"===========================================\n");CHKERRQ(ierr);

  /* Solve the nonlinear system 1 */
  if (SolveSubPhysics){
    ierr = DMMGSolve(dmmg1);CHKERRQ(ierr); 
    snes = DMMGGetSNES(dmmg1);
    ierr = SNESGetIterationNumber(snes,&its);CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"Physics 1: Number of Newton iterations = %D\n\n", its);CHKERRQ(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Setup Physics 2: 
        - Lap(T) + PR*Div([U*T,V*T]) = 0        
        where U and V are given by the given x.u and x.v
        - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMDACreate2d(comm,DMDA_BOUNDARY_NONE,DMDA_BOUNDARY_NONE,DMDA_STENCIL_STAR,-4,-4,PETSC_DECIDE,PETSC_DECIDE,1,1,0,0,&da2);CHKERRQ(ierr);
  ierr = DMDASetFieldName(da2,0,"temperature");CHKERRQ(ierr);

  /* Create the solver object and attach the grid/physics info */
  ierr = DMMGCreate(comm,1,&user,&dmmg2);CHKERRQ(ierr);
  ierr = DMMGSetDM(dmmg2,da2);CHKERRQ(ierr);
  ierr = DMMGSetISColoringType(dmmg2,IS_COLORING_GLOBAL);CHKERRQ(ierr);

  ierr = DMMGSetInitialGuess(dmmg2,FormInitialGuess2);CHKERRQ(ierr);
  ierr = DMMGSetSNES(dmmg2,FormFunction2,0);CHKERRQ(ierr);
  ierr = DMMGSetFromOptions(dmmg2);CHKERRQ(ierr);

  /* Solve the nonlinear system 2 */
  if (SolveSubPhysics){
    ierr = DMMGSolve(dmmg2);CHKERRQ(ierr); 
    snes = DMMGGetSNES(dmmg2);
    ierr = SNESGetIterationNumber(snes,&its);CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"Physics 2: Number of Newton iterations = %D\n\n", its);CHKERRQ(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve system 1 and 2 iteratively 
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMCreateLocalVector(da1,&X1_local);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(da2,&X2_local);CHKERRQ(ierr);

  /* Only 1 snes iteration is allowed for each subphysics */
  /*
  snes = DMMGGetSNES(dmmg1);
  ierr = SNESSetTolerances(snes,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,1,PETSC_DEFAULT);CHKERRQ(ierr);
  snes = DMMGGetSNES(dmmg2);
  ierr = SNESSetTolerances(snes,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,1,PETSC_DEFAULT);CHKERRQ(ierr);
  */
  max_its = 5;
  ierr = PetscOptionsGetInt(PETSC_NULL,"-mp_max_it",&max_its,PETSC_NULL);CHKERRQ(ierr);

  user.nsolve = 0;
  for (i=0; i<max_its; i++){
    ierr = PetscPrintf(comm,"\nIterative nsolve %D ...\n", user.nsolve);CHKERRQ(ierr);
    if (!GaussSeidel){
      /* get the ghosted X1_local for Physics 2 */
      X1   = DMMGGetx(dmmg1); //Jacobian
      if (i){ierr = DMDAVecRestoreArray(da1,X1_local,(Field1 **)&user.x1);CHKERRQ(ierr);}

      ierr = DMGlobalToLocalBegin(da1,X1,INSERT_VALUES,X1_local);CHKERRQ(ierr);
      ierr = DMGlobalToLocalEnd(da1,X1,INSERT_VALUES,X1_local);CHKERRQ(ierr);
      ierr = DMDAVecGetArray(da1,X1_local,(Field1 **)&user.x1);CHKERRQ(ierr);
    }

    ierr = DMMGSolve(dmmg1);CHKERRQ(ierr); 
    snes = DMMGGetSNES(dmmg1);
    ierr = SNESGetIterationNumber(snes,&its);CHKERRQ(ierr);

    if (GaussSeidel){
      /* get the ghosted X1_local for Physics 2 */
      X1   = DMMGGetx(dmmg1); 
      if (i){ierr = DMDAVecRestoreArray(da1,X1_local,(Field1 **)&user.x1);CHKERRQ(ierr);}

      ierr = DMGlobalToLocalBegin(da1,X1,INSERT_VALUES,X1_local);CHKERRQ(ierr);
      ierr = DMGlobalToLocalEnd(da1,X1,INSERT_VALUES,X1_local);CHKERRQ(ierr);
      ierr = DMDAVecGetArray(da1,X1_local,(Field1 **)&user.x1);CHKERRQ(ierr);
    }

    ierr = PetscPrintf(comm,"  Iterative physics 1: Number of Newton iterations = %D\n", its);CHKERRQ(ierr);
    user.nsolve++;

    ierr = DMMGSolve(dmmg2);CHKERRQ(ierr); 
    snes = DMMGGetSNES(dmmg2);
    ierr = SNESGetIterationNumber(snes,&its);CHKERRQ(ierr);

    /* get the ghosted X2_local for Physics 1 */
    X2   = DMMGGetx(dmmg2);
    if (i){ierr = DMDAVecRestoreArray(da2,X2_local,(Field2 **)&user.x2);CHKERRQ(ierr);}
    ierr = DMGlobalToLocalBegin(da2,X2,INSERT_VALUES,X2_local);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(da2,X2,INSERT_VALUES,X2_local);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da2,X2_local,(Field2 **)&user.x2);CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"  Iterative physics 2: Number of Newton iterations = %D\n", its);CHKERRQ(ierr);  
    //user.nsolve++;
  }
  ierr = DMDAVecRestoreArray(da1,X1_local,(Field1 **)&user.x1);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da2,X2_local,(Field2 **)&user.x2);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Create the DMComposite object to manage the two grids/physics. 
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscPrintf(comm,"  \n\n DMComposite iteration......\n");CHKERRQ(ierr);  
  ierr = DMCompositeCreate(comm,&pack);CHKERRQ(ierr);
  ierr = DMCompositeAddDM(pack,da1);CHKERRQ(ierr);
  ierr = DMCompositeAddDM(pack,da2);CHKERRQ(ierr);

  /* Create the solver object and attach the grid/physics info */
  ierr = DMMGCreate(comm,1,&user,&dmmg_comp);CHKERRQ(ierr);
  ierr = DMMGSetDM(dmmg_comp,pack);CHKERRQ(ierr);
  ierr = DMMGSetISColoringType(dmmg_comp,IS_COLORING_GLOBAL);CHKERRQ(ierr);

  ierr = DMMGSetInitialGuess(dmmg_comp,FormInitialGuessComp);CHKERRQ(ierr);
  ierr = DMMGSetSNES(dmmg_comp,FormFunctionComp,0);CHKERRQ(ierr);
  ierr = DMMGSetFromOptions(dmmg_comp);CHKERRQ(ierr);

  /* Solve the nonlinear system */
  /*  ierr = DMMGSolve(dmmg_comp);CHKERRQ(ierr); 
  snes = DMMGGetSNES(dmmg_comp);
  ierr = SNESGetIterationNumber(snes,&its);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"Composite Physics: Number of Newton iterations = %D\n\n", its);CHKERRQ(ierr);*/

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free spaces 
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMDestroy(&pack);CHKERRQ(ierr);
  ierr = DMDestroy(&da1);CHKERRQ(ierr);
  ierr = DMDestroy(&da2);CHKERRQ(ierr);
  ierr = DMMGDestroy(dmmg_comp);CHKERRQ(ierr);

 
  /* -snes_view */  
  //snes = DMMGGetSNES(dmmg1);CHKERRQ(ierr);

  ierr = DMMGDestroy(dmmg1);CHKERRQ(ierr);
  ierr = DMMGDestroy(dmmg2);CHKERRQ(ierr);

  ierr = VecDestroy(&X1_local);CHKERRQ(ierr);
  ierr = VecDestroy(&X2_local);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "FormInitialGuessComp"
/* 
   FormInitialGuessComp - 
              Forms the initial guess for the composite model
              Unwraps the global solution vector and passes its local pieces into the user functions
 */
PetscErrorCode FormInitialGuessComp(DMMG dmmg,Vec X)
{
  PetscErrorCode ierr;
  AppCtx         *user = (AppCtx*)dmmg->user;
  DM             dm = dmmg->dm;
  Vec            X1,X2;
  Field1         **x1;
  Field2         **x2;
  DMDALocalInfo  info1,info2;
  DM             da1,da2;

  PetscFunctionBegin;
  ierr = DMCompositeGetEntries(dm,&da1,&da2);CHKERRQ(ierr);
  /* Access the subvectors in X */
  ierr = DMCompositeGetAccess(dm,X,&X1,&X2);CHKERRQ(ierr);
  /* Access the arrays inside the subvectors of X */
  ierr = DMDAVecGetArray(da1,X1,(void**)&x1);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da2,X2,(void**)&x2);CHKERRQ(ierr);

  ierr = DMDAGetLocalInfo(da1,&info1);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da2,&info2);CHKERRQ(ierr);

  /* Evaluate local user provided function */
  ierr = FormInitialGuessLocal1(&info1,x1);CHKERRQ(ierr);
  ierr = FormInitialGuessLocal2(&info2,x2,user);CHKERRQ(ierr);

  ierr = DMDAVecRestoreArray(da1,X1,(void**)&x1);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da2,X2,(void**)&x2);CHKERRQ(ierr);
  ierr = DMCompositeRestoreAccess(dm,X,&X1,&X2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormFunctionComp"
/* 
   FormFunctionComp  - Unwraps the input vector and passes its local ghosted pieces into the user function
*/
PetscErrorCode FormFunctionComp(SNES snes,Vec X,Vec F,void *ctx)
{
  PetscErrorCode ierr;
  DMMG           dmmg = (DMMG)ctx;
  AppCtx         *user = (AppCtx*)dmmg->user;
  DM             dm = dmmg->dm;
  DMDALocalInfo  info1,info2;
  DM             da1,da2;
  Field1         **x1,**f1;
  Field2         **x2,**f2;
  Vec            X1,X2,F1,F2;

  PetscFunctionBegin;

  ierr = DMCompositeGetEntries(dm,&da1,&da2);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da1,&info1);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da2,&info2);CHKERRQ(ierr);

  /* Get local vectors to hold ghosted parts of X */
  ierr = DMCompositeGetLocalVectors(dm,&X1,&X2);CHKERRQ(ierr);
  ierr = DMCompositeScatter(dm,X,X1,X2);CHKERRQ(ierr); 

  /* Access the arrays inside the subvectors of X */
  ierr = DMDAVecGetArray(da1,X1,(void**)&x1);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da2,X2,(void**)&x2);CHKERRQ(ierr);

  /* Access the subvectors in F. 
     These are not ghosted so directly access the memory locations in F */
  ierr = DMCompositeGetAccess(dm,F,&F1,&F2);CHKERRQ(ierr);

  /* Access the arrays inside the subvectors of F */  
  ierr = DMDAVecGetArray(da1,F1,(void**)&f1);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da2,F2,(void**)&f2);CHKERRQ(ierr);

  /* Evaluate local user provided function */    
  ierr = FormFunctionLocal1(&info1,x1,x2,f1,(void**)user);CHKERRQ(ierr);
  ierr = FormFunctionLocal2(&info2,x1,x2,f2,(void**)user);CHKERRQ(ierr);

  ierr = DMDAVecRestoreArray(da1,F1,(void**)&f1);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da2,F2,(void**)&f2);CHKERRQ(ierr);
  ierr = DMCompositeRestoreAccess(dm,F,&F1,&F2);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da1,X1,(void**)&x1);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da2,X2,(void**)&x2);CHKERRQ(ierr);
  ierr = DMCompositeRestoreLocalVectors(dm,&X1,&X2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Copied from p1.c */
#undef __FUNCT__
#define __FUNCT__ "FormInitialGuess1"
PetscErrorCode FormInitialGuess1(DMMG dmmg,Vec X)
{
  PetscErrorCode ierr;
  AppCtx         *user = (AppCtx*)dmmg->user;
  DM             da1 = dmmg->dm;
  Field1         **x1;
  DMDALocalInfo    info1;

  PetscFunctionBegin;
  if (user->nsolve) PetscFunctionReturn(0);
  printf(" FormInitialGuess1 ... user.nsolve %d\n",user->nsolve);

  /* Access the array inside of X */
  ierr = DMDAVecGetArray(da1,X,(void**)&x1);CHKERRQ(ierr);

  ierr = DMDAGetLocalInfo(da1,&info1);CHKERRQ(ierr);

  /* Evaluate local user provided function */
  ierr = FormInitialGuessLocal1(&info1,x1);CHKERRQ(ierr);

  ierr = DMDAVecRestoreArray(da1,X,(void**)&x1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormFunction1"
PetscErrorCode FormFunction1(SNES snes,Vec X,Vec F,void *ctx) 
{
  PetscErrorCode ierr;
  DMMG           dmmg = (DMMG)ctx;
  AppCtx         *user = (AppCtx*)dmmg->user;
  DM             da1 = dmmg->dm;
  DMDALocalInfo  info1;
  Field1         **x1,**f1;
  Vec            X1;
  Field2         **x2;

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
  if (user->nsolve){
    x2 = user->x2;
    ierr = FormFunctionLocal1(&info1,x1,x2,f1,(void**)user);CHKERRQ(ierr);
  } else {
    ierr = FormFunctionLocal1(&info1,x1,0,f1,(void**)user);CHKERRQ(ierr);
  }


  ierr = DMDAVecRestoreArray(da1,F,(void**)&f1);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da1,X1,(void**)&x1);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da1,&X1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Copied from p2.c */
#undef __FUNCT__
#define __FUNCT__ "FormInitialGuess2"
PetscErrorCode FormInitialGuess2(DMMG dmmg,Vec X)
{
  PetscErrorCode ierr;
  AppCtx         *user = (AppCtx*)dmmg->user;
  DM             da2 = dmmg->dm;
  Field2         **x2;
  DMDALocalInfo    info2;

  PetscFunctionBegin;
  if (user->nsolve) PetscFunctionReturn(0);
  printf(" FormInitialGuess2 ... user.nsolve %d\n",user->nsolve);

  /* Access the arrays inside  of X */
  ierr = DMDAVecGetArray(da2,X,(void**)&x2);CHKERRQ(ierr);

  ierr = DMDAGetLocalInfo(da2,&info2);CHKERRQ(ierr);

  /* Evaluate local user provided function */
  ierr = FormInitialGuessLocal2(&info2,x2,user);CHKERRQ(ierr);

  ierr = DMDAVecRestoreArray(da2,X,(void**)&x2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormFunction2"
PetscErrorCode FormFunction2(SNES snes,Vec X,Vec F,void *ctx)
{
  PetscErrorCode ierr;
  DMMG           dmmg = (DMMG)ctx;
  AppCtx         *user = (AppCtx*)dmmg->user;
  DM             da2 = dmmg->dm;
  DMDALocalInfo    info2;
  Field2         **x2,**f2;
  Vec            X2;
  Field1         **x1;

  PetscFunctionBegin;
  ierr = DMDAGetLocalInfo(da2,&info2);CHKERRQ(ierr);

  /* Get local vectors to hold ghosted parts of X */
  ierr = DMGetLocalVector(da2,&X2);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da2,X,INSERT_VALUES,X2);CHKERRQ(ierr); 
  ierr = DMGlobalToLocalEnd(da2,X,INSERT_VALUES,X2);CHKERRQ(ierr); 

  /* Access the array inside of X1 */
  ierr = DMDAVecGetArray(da2,X2,(void**)&x2);CHKERRQ(ierr);

  /* Access the subvectors in F. 
     These are not ghosted so directly access the memory locations in F */
  ierr = DMDAVecGetArray(da2,F,(void**)&f2);CHKERRQ(ierr);

  /* Evaluate local user provided function */    
  if (user->nsolve){
    x1 = user->x1;
    ierr = FormFunctionLocal2(&info2,x1,x2,f2,(void**)user);CHKERRQ(ierr);
  } else {
    ierr = FormFunctionLocal2(&info2,0,x2,f2,(void**)user);CHKERRQ(ierr);
  }

  ierr = DMDAVecRestoreArray(da2,F,(void**)&f2);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da2,X2,(void**)&x2);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da2,&X2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

