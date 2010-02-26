
static char help[] = "Model nonlinear multi-physics solver. Modified from mp.c \n\\n";

/* ------------------------------------------------------------------------
    See ex19.c for discussion of the problem 

    Examples of command line options:
      ./mp -dmmg_jacobian_mf_fd_operator
      ./mp -dmcomposite_dense_jacobian #inefficient, but compute entire Jacobian for testing

      ./mp1 -snes_monitor -mp_max_it 14 -grashof 1000.0
  ----------------------------------------------------------------------------------------- */
#include "petsctime.h"
#include "mp1.h"

extern PetscErrorCode FormInitialGuessComp(DMMG,Vec);
extern PetscErrorCode FormFunctionComp(SNES,Vec,Vec,void*);

extern PetscErrorCode FormInitialGuess1(DMMG,Vec);
extern PetscErrorCode FormFunction1(SNES,Vec,Vec,void *);

extern PetscErrorCode FormInitialGuess2(DMMG,Vec);
extern PetscErrorCode FormFunction2(SNES,Vec,Vec,void *);

extern PetscErrorCode PetscLogPrintSummaryToPy(MPI_Comm,PetscViewer);

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
  DA             da1,da2;
  DMComposite    pack;

  DMMG           *dmmg1,*dmmg2;
  PetscTruth     SolveSubPhysics=PETSC_FALSE,GaussSeidel=PETSC_TRUE,Jacobi=PETSC_FALSE;
  Vec            X1,X1_local,X2,X2_local;
  PetscViewer    viewer;

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
  ierr = DACreate2d(comm,DA_NONPERIODIC,DA_STENCIL_STAR,-4,-4,PETSC_DECIDE,PETSC_DECIDE,3,1,0,0,&da1);CHKERRQ(ierr);
  ierr = DASetFieldName(da1,0,"x-velocity");CHKERRQ(ierr);
  ierr = DASetFieldName(da1,1,"y-velocity");CHKERRQ(ierr);
  ierr = DASetFieldName(da1,2,"Omega");CHKERRQ(ierr);

  /* Create the solver object and attach the grid/physics info */
  ierr = DMMGCreate(comm,1,&user,&dmmg1);CHKERRQ(ierr);
  ierr = DMMGSetDM(dmmg1,(DM)da1);CHKERRQ(ierr);
  ierr = DMMGSetISColoringType(dmmg1,IS_COLORING_GLOBAL);CHKERRQ(ierr);

  ierr = DMMGSetInitialGuess(dmmg1,FormInitialGuess1);CHKERRQ(ierr);
  ierr = DMMGSetSNES(dmmg1,FormFunction1,0);CHKERRQ(ierr);
  ierr = DMMGSetFromOptions(dmmg1);CHKERRQ(ierr);

  /* Set problem parameters (velocity of lid, prandtl, and grashof numbers) */  
  ierr = DAGetInfo(da1,PETSC_NULL,&mx,&my,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
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
  ierr = DACreate2d(comm,DA_NONPERIODIC,DA_STENCIL_STAR,-4,-4,PETSC_DECIDE,PETSC_DECIDE,1,1,0,0,&da2);CHKERRQ(ierr);
  ierr = DASetFieldName(da2,0,"temperature");CHKERRQ(ierr);

  /* Create the solver object and attach the grid/physics info */
  ierr = DMMGCreate(comm,1,&user,&dmmg2);CHKERRQ(ierr);
  ierr = DMMGSetDM(dmmg2,(DM)da2);CHKERRQ(ierr);
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
  ierr = DACreateLocalVector(da1,&X1_local);CHKERRQ(ierr);
  ierr = DACreateLocalVector(da2,&X2_local);CHKERRQ(ierr);

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
      if (i){ierr = DAVecRestoreArray(da1,X1_local,(Field1 **)&user.x1);CHKERRQ(ierr);}

      ierr = DAGlobalToLocalBegin(da1,X1,INSERT_VALUES,X1_local);CHKERRQ(ierr);
      ierr = DAGlobalToLocalEnd(da1,X1,INSERT_VALUES,X1_local);CHKERRQ(ierr);
      ierr = DAVecGetArray(da1,X1_local,(Field1 **)&user.x1);CHKERRQ(ierr);
    }

    ierr = DMMGSolve(dmmg1);CHKERRQ(ierr); 
    snes = DMMGGetSNES(dmmg1);
    ierr = SNESGetIterationNumber(snes,&its);CHKERRQ(ierr);

    if (GaussSeidel){
      /* get the ghosted X1_local for Physics 2 */
      X1   = DMMGGetx(dmmg1); 
      if (i){ierr = DAVecRestoreArray(da1,X1_local,(Field1 **)&user.x1);CHKERRQ(ierr);}

      ierr = DAGlobalToLocalBegin(da1,X1,INSERT_VALUES,X1_local);CHKERRQ(ierr);
      ierr = DAGlobalToLocalEnd(da1,X1,INSERT_VALUES,X1_local);CHKERRQ(ierr);
      ierr = DAVecGetArray(da1,X1_local,(Field1 **)&user.x1);CHKERRQ(ierr);
    }

    ierr = PetscPrintf(comm,"  Iterative physics 1: Number of Newton iterations = %D\n", its);CHKERRQ(ierr);
    user.nsolve++;

    ierr = DMMGSolve(dmmg2);CHKERRQ(ierr); 
    snes = DMMGGetSNES(dmmg2);
    ierr = SNESGetIterationNumber(snes,&its);CHKERRQ(ierr);

    /* get the ghosted X2_local for Physics 1 */
    X2   = DMMGGetx(dmmg2);
    if (i){ierr = DAVecRestoreArray(da2,X2_local,(Field2 **)&user.x2);CHKERRQ(ierr);}
    ierr = DAGlobalToLocalBegin(da2,X2,INSERT_VALUES,X2_local);CHKERRQ(ierr);
    ierr = DAGlobalToLocalEnd(da2,X2,INSERT_VALUES,X2_local);CHKERRQ(ierr);
    ierr = DAVecGetArray(da2,X2_local,(Field2 **)&user.x2);CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"  Iterative physics 2: Number of Newton iterations = %D\n", its);CHKERRQ(ierr);  
    //user.nsolve++;
  }
  ierr = DAVecRestoreArray(da1,X1_local,(Field1 **)&user.x1);CHKERRQ(ierr);
  ierr = DAVecRestoreArray(da2,X2_local,(Field2 **)&user.x2);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Create the DMComposite object to manage the two grids/physics. 
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscPrintf(comm,"  \n\n DMComposite iteration......\n");CHKERRQ(ierr);  
  ierr = DMCompositeCreate(comm,&pack);CHKERRQ(ierr);
  ierr = DMCompositeAddDM(pack,(DM)da1);CHKERRQ(ierr);
  ierr = DMCompositeAddDM(pack,(DM)da2);CHKERRQ(ierr);

  /* Create the solver object and attach the grid/physics info */
  ierr = DMMGCreate(comm,1,&user,&dmmg_comp);CHKERRQ(ierr);
  ierr = DMMGSetDM(dmmg_comp,(DM)pack);CHKERRQ(ierr);
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
  ierr = DMCompositeDestroy(pack);CHKERRQ(ierr);
  ierr = DADestroy(da1);CHKERRQ(ierr);
  ierr = DADestroy(da2);CHKERRQ(ierr);
  ierr = DMMGDestroy(dmmg_comp);CHKERRQ(ierr);

  ierr = PetscViewerASCIIOpen(comm,"log.py",&viewer);CHKERRQ(ierr);
  /* -log_summary */
  ierr = PetscLogPrintSummaryToPy(comm,viewer);CHKERRQ(ierr);
 
  /* -snes_view */  
  //snes = DMMGGetSNES(dmmg1);CHKERRQ(ierr);

  ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
    
  ierr = DMMGDestroy(dmmg1);CHKERRQ(ierr);
  ierr = DMMGDestroy(dmmg2);CHKERRQ(ierr);

  ierr = VecDestroy(X1_local);CHKERRQ(ierr);
  ierr = VecDestroy(X2_local);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
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
  DMComposite    dm = (DMComposite)dmmg->dm;
  Vec            X1,X2;
  Field1         **x1;
  Field2         **x2;
  DALocalInfo    info1,info2;
  DA             da1,da2;

  PetscFunctionBegin;
  ierr = DMCompositeGetEntries(dm,&da1,&da2);CHKERRQ(ierr);
  /* Access the subvectors in X */
  ierr = DMCompositeGetAccess(dm,X,&X1,&X2);CHKERRQ(ierr);
  /* Access the arrays inside the subvectors of X */
  ierr = DAVecGetArray(da1,X1,(void**)&x1);CHKERRQ(ierr);
  ierr = DAVecGetArray(da2,X2,(void**)&x2);CHKERRQ(ierr);

  ierr = DAGetLocalInfo(da1,&info1);CHKERRQ(ierr);
  ierr = DAGetLocalInfo(da2,&info2);CHKERRQ(ierr);

  /* Evaluate local user provided function */
  ierr = FormInitialGuessLocal1(&info1,x1);CHKERRQ(ierr);
  ierr = FormInitialGuessLocal2(&info2,x2,user);CHKERRQ(ierr);

  ierr = DAVecRestoreArray(da1,X1,(void**)&x1);CHKERRQ(ierr);
  ierr = DAVecRestoreArray(da2,X2,(void**)&x2);CHKERRQ(ierr);
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
  DMComposite    dm = (DMComposite)dmmg->dm;
  DALocalInfo    info1,info2;
  DA             da1,da2;
  Field1         **x1,**f1;
  Field2         **x2,**f2;
  Vec            X1,X2,F1,F2;

  PetscFunctionBegin;

  ierr = DMCompositeGetEntries(dm,&da1,&da2);CHKERRQ(ierr);
  ierr = DAGetLocalInfo(da1,&info1);CHKERRQ(ierr);
  ierr = DAGetLocalInfo(da2,&info2);CHKERRQ(ierr);

  /* Get local vectors to hold ghosted parts of X */
  ierr = DMCompositeGetLocalVectors(dm,&X1,&X2);CHKERRQ(ierr);
  ierr = DMCompositeScatter(dm,X,X1,X2);CHKERRQ(ierr); 

  /* Access the arrays inside the subvectors of X */
  ierr = DAVecGetArray(da1,X1,(void**)&x1);CHKERRQ(ierr);
  ierr = DAVecGetArray(da2,X2,(void**)&x2);CHKERRQ(ierr);

  /* Access the subvectors in F. 
     These are not ghosted so directly access the memory locations in F */
  ierr = DMCompositeGetAccess(dm,F,&F1,&F2);CHKERRQ(ierr);

  /* Access the arrays inside the subvectors of F */  
  ierr = DAVecGetArray(da1,F1,(void**)&f1);CHKERRQ(ierr);
  ierr = DAVecGetArray(da2,F2,(void**)&f2);CHKERRQ(ierr);

  /* Evaluate local user provided function */    
  ierr = FormFunctionLocal1(&info1,x1,x2,f1,(void**)user);CHKERRQ(ierr);
  ierr = FormFunctionLocal2(&info2,x1,x2,f2,(void**)user);CHKERRQ(ierr);

  ierr = DAVecRestoreArray(da1,F1,(void**)&f1);CHKERRQ(ierr);
  ierr = DAVecRestoreArray(da2,F2,(void**)&f2);CHKERRQ(ierr);
  ierr = DMCompositeRestoreAccess(dm,F,&F1,&F2);CHKERRQ(ierr);
  ierr = DAVecRestoreArray(da1,X1,(void**)&x1);CHKERRQ(ierr);
  ierr = DAVecRestoreArray(da2,X2,(void**)&x2);CHKERRQ(ierr);
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
  DA             da1 = (DA)dmmg->dm;
  Field1         **x1;
  DALocalInfo    info1;

  PetscFunctionBegin;
  if (user->nsolve) PetscFunctionReturn(0);
  printf(" FormInitialGuess1 ... user.nsolve %d\n",user->nsolve);

  /* Access the array inside of X */
  ierr = DAVecGetArray(da1,X,(void**)&x1);CHKERRQ(ierr);

  ierr = DAGetLocalInfo(da1,&info1);CHKERRQ(ierr);

  /* Evaluate local user provided function */
  ierr = FormInitialGuessLocal1(&info1,x1);CHKERRQ(ierr);

  ierr = DAVecRestoreArray(da1,X,(void**)&x1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormFunction1"
PetscErrorCode FormFunction1(SNES snes,Vec X,Vec F,void *ctx) 
{
  PetscErrorCode ierr;
  DMMG           dmmg = (DMMG)ctx;
  AppCtx         *user = (AppCtx*)dmmg->user;
  DA             da1 = (DA)dmmg->dm;
  DALocalInfo    info1;
  Field1         **x1,**f1;
  Vec            X1;
  Field2         **x2;

  PetscFunctionBegin;
  ierr = DAGetLocalInfo(da1,&info1);CHKERRQ(ierr);

  /* Get local vectors to hold ghosted parts of X */
  ierr = DAGetLocalVector(da1,&X1);CHKERRQ(ierr);
  ierr = DAGlobalToLocalBegin(da1,X,INSERT_VALUES,X1);CHKERRQ(ierr); 
  ierr = DAGlobalToLocalEnd(da1,X,INSERT_VALUES,X1);CHKERRQ(ierr); 

  /* Access the arrays inside X1 */
  ierr = DAVecGetArray(da1,X1,(void**)&x1);CHKERRQ(ierr);

  /* Access the subvectors in F. 
     These are not ghosted so directly access the memory locations in F */
  ierr = DAVecGetArray(da1,F,(void**)&f1);CHKERRQ(ierr);

  /* Evaluate local user provided function */   
  if (user->nsolve){
    x2 = user->x2;
    ierr = FormFunctionLocal1(&info1,x1,x2,f1,(void**)user);CHKERRQ(ierr);
  } else {
    ierr = FormFunctionLocal1(&info1,x1,0,f1,(void**)user);CHKERRQ(ierr);
  }


  ierr = DAVecRestoreArray(da1,F,(void**)&f1);CHKERRQ(ierr);
  ierr = DAVecRestoreArray(da1,X1,(void**)&x1);CHKERRQ(ierr);
  ierr = DARestoreLocalVector(da1,&X1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Copied from p2.c */
#undef __FUNCT__
#define __FUNCT__ "FormInitialGuess2"
PetscErrorCode FormInitialGuess2(DMMG dmmg,Vec X)
{
  PetscErrorCode ierr;
  AppCtx         *user = (AppCtx*)dmmg->user;
  DA             da2 = (DA)dmmg->dm;
  Field2         **x2;
  DALocalInfo    info2;

  PetscFunctionBegin;
  if (user->nsolve) PetscFunctionReturn(0);
  printf(" FormInitialGuess2 ... user.nsolve %d\n",user->nsolve);

  /* Access the arrays inside  of X */
  ierr = DAVecGetArray(da2,X,(void**)&x2);CHKERRQ(ierr);

  ierr = DAGetLocalInfo(da2,&info2);CHKERRQ(ierr);

  /* Evaluate local user provided function */
  ierr = FormInitialGuessLocal2(&info2,x2,user);CHKERRQ(ierr);

  ierr = DAVecRestoreArray(da2,X,(void**)&x2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormFunction2"
PetscErrorCode FormFunction2(SNES snes,Vec X,Vec F,void *ctx)
{
  PetscErrorCode ierr;
  DMMG           dmmg = (DMMG)ctx;
  AppCtx         *user = (AppCtx*)dmmg->user;
  DA             da2 = (DA)dmmg->dm;
  DALocalInfo    info2;
  Field2         **x2,**f2;
  Vec            X2;
  Field1         **x1;

  PetscFunctionBegin;
  ierr = DAGetLocalInfo(da2,&info2);CHKERRQ(ierr);

  /* Get local vectors to hold ghosted parts of X */
  ierr = DAGetLocalVector(da2,&X2);CHKERRQ(ierr);
  ierr = DAGlobalToLocalBegin(da2,X,INSERT_VALUES,X2);CHKERRQ(ierr); 
  ierr = DAGlobalToLocalEnd(da2,X,INSERT_VALUES,X2);CHKERRQ(ierr); 

  /* Access the array inside of X1 */
  ierr = DAVecGetArray(da2,X2,(void**)&x2);CHKERRQ(ierr);

  /* Access the subvectors in F. 
     These are not ghosted so directly access the memory locations in F */
  ierr = DAVecGetArray(da2,F,(void**)&f2);CHKERRQ(ierr);

  /* Evaluate local user provided function */    
  if (user->nsolve){
    x1 = user->x1;
    ierr = FormFunctionLocal2(&info2,x1,x2,f2,(void**)user);CHKERRQ(ierr);
  } else {
    ierr = FormFunctionLocal2(&info2,0,x2,f2,(void**)user);CHKERRQ(ierr);
  }

  ierr = DAVecRestoreArray(da2,F,(void**)&f2);CHKERRQ(ierr);
  ierr = DAVecRestoreArray(da2,X2,(void**)&x2);CHKERRQ(ierr);
  ierr = DARestoreLocalVector(da2,&X2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#include "../src/sys/viewer/impls/ascii/asciiimpl.h"  /*I     "petscsys.h"   I*/
/* 
   PetscLogPrintSummaryToPy() is customized from PetscLogPrintSummary().
   It replaces input parameter "const char filename[]" with "PetscViewer viewer".
*/
#undef __FUNCT__  
#define __FUNCT__ "PetscLogPrintSummaryToPy"
PetscErrorCode PetscLogPrintSummaryToPy(MPI_Comm comm, PetscViewer viewer) 
{
  PetscViewer_ASCII *ascii = (PetscViewer_ASCII*)viewer->data;
  FILE              *fd = ascii->fd; 
  PetscLogDouble zero = 0.0;
  StageLog       stageLog;
  StageInfo     *stageInfo = PETSC_NULL;
  EventPerfInfo *eventInfo = PETSC_NULL;
  ClassPerfInfo *classInfo;
  const char    *name;
  PetscLogDouble locTotalTime, TotalTime, TotalFlops;
  PetscLogDouble numMessages, messageLength, avgMessLen, numReductions;
  PetscLogDouble stageTime, flops, mem, mess, messLen, red;
  PetscLogDouble fracTime, fracFlops, fracMessages, fracLength;
  PetscLogDouble fracReductions;
  PetscLogDouble tot,avg,x,y,*mydata;
  PetscMPIInt    minCt, maxCt;
  PetscMPIInt    size, rank, *mycount;
  PetscTruth    *localStageUsed,    *stageUsed;
  PetscTruth    *localStageVisible, *stageVisible;
  int            numStages, localNumEvents, numEvents;
  int            stage, lastStage;
  PetscLogEvent  event;
  PetscErrorCode ierr;
  PetscInt       i;

  /* remove these two lines! */
  PetscLogDouble PETSC_DLLEXPORT BaseTime = 0.0;
  int            numObjects = 0;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = PetscMalloc(size*sizeof(PetscLogDouble), &mydata);CHKERRQ(ierr);
  ierr = PetscMalloc(size*sizeof(PetscMPIInt), &mycount);CHKERRQ(ierr);

  /* Pop off any stages the user forgot to remove */
  lastStage = 0;
  ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
  ierr = StageLogGetCurrent(stageLog, &stage);CHKERRQ(ierr);
  while (stage >= 0) {
    lastStage = stage;
    ierr = StageLogPop(stageLog);CHKERRQ(ierr);
    ierr = StageLogGetCurrent(stageLog, &stage);CHKERRQ(ierr);
  }
  /* Get the total elapsed time */
  PetscTime(locTotalTime);  locTotalTime -= BaseTime;

  ierr = PetscFPrintf(comm, fd, "\n#------ PETSc Performance Summary ----------\n\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm, fd, "Nproc = %d\n",size);CHKERRQ(ierr);

  /* Must preserve reduction count before we go on */
  red  = (allreduce_ct + gather_ct + scatter_ct)/((PetscLogDouble) size);

  /* Calculate summary information */
 
  /*   Time */  
  ierr = MPI_Gather(&locTotalTime,1,MPIU_PETSCLOGDOUBLE,mydata,1,MPIU_PETSCLOGDOUBLE,0,comm);CHKERRQ(ierr);
  if (!rank){
    ierr = PetscFPrintf(comm, fd, "Time = [ " );CHKERRQ(ierr); 
    tot  = 0.0;
    for (i=0; i<size; i++){
      tot += mydata[i];
      ierr = PetscFPrintf(comm, fd, "  %5.3e,",mydata[i] );CHKERRQ(ierr); 
    }
    ierr = PetscFPrintf(comm, fd, "]\n" );CHKERRQ(ierr); 
    avg  = (tot)/((PetscLogDouble) size);
    TotalTime = tot;
  }

  /*   Objects */
  avg  = (PetscLogDouble) numObjects;
  ierr = MPI_Gather(&avg,1,MPIU_PETSCLOGDOUBLE,mydata,1,MPIU_PETSCLOGDOUBLE,0,comm);CHKERRQ(ierr);
  if (!rank){
    ierr = PetscFPrintf(comm, fd, "Objects = [ " );CHKERRQ(ierr); 
    for (i=0; i<size; i++){
      ierr = PetscFPrintf(comm, fd, "  %5.3e,",mydata[i] );CHKERRQ(ierr); 
    }
    ierr = PetscFPrintf(comm, fd, "]\n" );CHKERRQ(ierr); 
  }

  /*   Flops */
  ierr = MPI_Gather(&_TotalFlops,1,MPIU_PETSCLOGDOUBLE,mydata,1,MPIU_PETSCLOGDOUBLE,0,comm);CHKERRQ(ierr);
  if (!rank){
    ierr = PetscFPrintf(comm, fd, "Flops = [ " );CHKERRQ(ierr); 
    tot  = 0.0;
    for (i=0; i<size; i++){
      tot += mydata[i];
      ierr = PetscFPrintf(comm, fd, "  %5.3e,",mydata[i] );CHKERRQ(ierr); 
    }
    ierr = PetscFPrintf(comm, fd, "]\n");CHKERRQ(ierr); 
    TotalFlops = tot;
  }

  /*   Memory */
  ierr = PetscMallocGetMaximumUsage(&mem);CHKERRQ(ierr); 
  ierr = MPI_Gather(&mem,1,MPIU_PETSCLOGDOUBLE,mydata,1,MPIU_PETSCLOGDOUBLE,0,comm);CHKERRQ(ierr);
  if (!rank){
    ierr = PetscFPrintf(comm, fd, "Memory = [ " );CHKERRQ(ierr); 
    for (i=0; i<size; i++){
      ierr = PetscFPrintf(comm, fd, "  %5.3e,",mydata[i] );CHKERRQ(ierr); 
    }
    ierr = PetscFPrintf(comm, fd, "]\n" );CHKERRQ(ierr); 
  }
 
  /*   Messages */
  mess = 0.5*(irecv_ct + isend_ct + recv_ct + send_ct);
  ierr = MPI_Gather(&mess,1,MPIU_PETSCLOGDOUBLE,mydata,1,MPIU_PETSCLOGDOUBLE,0,comm);CHKERRQ(ierr);
  if (!rank){
    ierr = PetscFPrintf(comm, fd, "MPIMessages = [ " );CHKERRQ(ierr); 
    tot  = 0.0;
    for (i=0; i<size; i++){
      tot += mydata[i];
      ierr = PetscFPrintf(comm, fd, "  %5.3e,",mydata[i] );CHKERRQ(ierr); 
    }
    ierr = PetscFPrintf(comm, fd, "]\n" );CHKERRQ(ierr); 
    numMessages = tot;
  }

  /*   Message Lengths */
  mess = 0.5*(irecv_len + isend_len + recv_len + send_len);
  ierr = MPI_Gather(&mess,1,MPIU_PETSCLOGDOUBLE,mydata,1,MPIU_PETSCLOGDOUBLE,0,comm);CHKERRQ(ierr);
  if (!rank){
    ierr = PetscFPrintf(comm, fd, "MPIMessageLengths = [ " );CHKERRQ(ierr); 
    tot  = 0.0;
    for (i=0; i<size; i++){
      tot += mydata[i];
      ierr = PetscFPrintf(comm, fd, "  %5.3e,",mydata[i] );CHKERRQ(ierr); 
    }
    ierr = PetscFPrintf(comm, fd, "]\n" );CHKERRQ(ierr); 
    messageLength = tot;
  }

  /*   Reductions */
  ierr = MPI_Gather(&red,1,MPIU_PETSCLOGDOUBLE,mydata,1,MPIU_PETSCLOGDOUBLE,0,comm);CHKERRQ(ierr);
  if (!rank){
    ierr = PetscFPrintf(comm, fd, "MPIReductions = [ " );CHKERRQ(ierr); 
    tot  = 0.0;
    for (i=0; i<size; i++){
      tot += mydata[i];
      ierr = PetscFPrintf(comm, fd, "  %5.3e,",mydata[i] );CHKERRQ(ierr); 
    }
    ierr = PetscFPrintf(comm, fd, "]\n" );CHKERRQ(ierr); 
    numReductions = tot;
  }

  /* Get total number of stages --
       Currently, a single processor can register more stages than another, but stages must all be registered in order.
       We can removed this requirement if necessary by having a global stage numbering and indirection on the stage ID.
       This seems best accomplished by assoicating a communicator with each stage.
  */
  ierr = MPI_Allreduce(&stageLog->numStages, &numStages, 1, MPI_INT, MPI_MAX, comm);CHKERRQ(ierr);
  ierr = PetscMalloc(numStages * sizeof(PetscTruth), &localStageUsed);CHKERRQ(ierr);
  ierr = PetscMalloc(numStages * sizeof(PetscTruth), &stageUsed);CHKERRQ(ierr);
  ierr = PetscMalloc(numStages * sizeof(PetscTruth), &localStageVisible);CHKERRQ(ierr);
  ierr = PetscMalloc(numStages * sizeof(PetscTruth), &stageVisible);CHKERRQ(ierr);
  if (numStages > 0) {
    stageInfo = stageLog->stageInfo;
    for(stage = 0; stage < numStages; stage++) {
      if (stage < stageLog->numStages) {
        localStageUsed[stage]    = stageInfo[stage].used;
        localStageVisible[stage] = stageInfo[stage].perfInfo.visible;
      } else {
        localStageUsed[stage]    = PETSC_FALSE;
        localStageVisible[stage] = PETSC_TRUE;
      }
    }
    ierr = MPI_Allreduce(localStageUsed,    stageUsed,    numStages, MPI_INT, MPI_LOR,  comm);CHKERRQ(ierr);
    ierr = MPI_Allreduce(localStageVisible, stageVisible, numStages, MPI_INT, MPI_LAND, comm);CHKERRQ(ierr);
for(stage = 0; stage < numStages; stage++) {
      if (stageUsed[stage]) {
        ierr = PetscFPrintf(comm, fd, "\n#Summary of Stages:   ----- Time ------  ----- Flops -----  --- Messages ---  -- Message Lengths --  -- Reductions --\n");CHKERRQ(ierr);
        ierr = PetscFPrintf(comm, fd, "#                       Avg     %%Total     Avg     %%Total   counts   %%Total     Avg         %%Total   counts   %%Total \n");CHKERRQ(ierr);
        break;
      }
    }
    for(stage = 0; stage < numStages; stage++) {
      if (!stageUsed[stage]) continue;
      if (localStageUsed[stage]) {
        ierr = MPI_Allreduce(&stageInfo[stage].perfInfo.time,          &stageTime, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm);CHKERRQ(ierr);
        ierr = MPI_Allreduce(&stageInfo[stage].perfInfo.flops,         &flops,     1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm);CHKERRQ(ierr);
        ierr = MPI_Allreduce(&stageInfo[stage].perfInfo.numMessages,   &mess,      1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm);CHKERRQ(ierr);
        ierr = MPI_Allreduce(&stageInfo[stage].perfInfo.messageLength, &messLen,   1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm);CHKERRQ(ierr);
        ierr = MPI_Allreduce(&stageInfo[stage].perfInfo.numReductions, &red,       1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm);CHKERRQ(ierr);
        name = stageInfo[stage].name;
      } else {
        ierr = MPI_Allreduce(&zero,                           &stageTime, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm);CHKERRQ(ierr);
        ierr = MPI_Allreduce(&zero,                           &flops,     1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm);CHKERRQ(ierr);
        ierr = MPI_Allreduce(&zero,                           &mess,      1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm);CHKERRQ(ierr);
        ierr = MPI_Allreduce(&zero,                           &messLen,   1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm);CHKERRQ(ierr);
        ierr = MPI_Allreduce(&zero,                           &red,       1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm);CHKERRQ(ierr);
        name = "";
      }
      mess *= 0.5; messLen *= 0.5; red /= size;
      if (TotalTime     != 0.0) fracTime       = stageTime/TotalTime;    else fracTime       = 0.0;
      if (TotalFlops    != 0.0) fracFlops      = flops/TotalFlops;       else fracFlops      = 0.0;
      /* Talk to Barry if (stageTime     != 0.0) flops          = (size*flops)/stageTime; else flops          = 0.0; */
      if (numMessages   != 0.0) fracMessages   = mess/numMessages;       else fracMessages   = 0.0;
      if (numMessages   != 0.0) avgMessLen     = messLen/numMessages;    else avgMessLen     = 0.0;
      if (messageLength != 0.0) fracLength     = messLen/messageLength;  else fracLength     = 0.0;
      if (numReductions != 0.0) fracReductions = red/numReductions;      else fracReductions = 0.0;
      ierr = PetscFPrintf(comm, fd, "# ");
      ierr = PetscFPrintf(comm, fd, "%2d: %15s: %6.4e %5.1f%%  %6.4e %5.1f%%  %5.3e %5.1f%%  %5.3e      %5.1f%%  %5.3e %5.1f%% \n",
                          stage, name, stageTime/size, 100.0*fracTime, flops, 100.0*fracFlops,
                          mess, 100.0*fracMessages, avgMessLen, 100.0*fracLength, red, 100.0*fracReductions);CHKERRQ(ierr);
    }
  }

  /* Report events */
  ierr = PetscFPrintf(comm, fd,"\n# Event\n");CHKERRQ(ierr);                        
  ierr = PetscFPrintf(comm,fd,"# ------------------------------------------------------\n");
                                                                                                          CHKERRQ(ierr); 
  /* Problem: The stage name will not show up unless the stage executed on proc 1 */
  for(stage = 0; stage < numStages; stage++) {
    if (!stageVisible[stage]) continue;
    if (localStageUsed[stage]) {
      ierr = MPI_Allreduce(&stageInfo[stage].perfInfo.time,          &stageTime, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm);CHKERRQ(ierr);
      ierr = MPI_Allreduce(&stageInfo[stage].perfInfo.flops,         &flops,     1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm);CHKERRQ(ierr);
      ierr = MPI_Allreduce(&stageInfo[stage].perfInfo.numMessages,   &mess,      1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm);CHKERRQ(ierr);
      ierr = MPI_Allreduce(&stageInfo[stage].perfInfo.messageLength, &messLen,   1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm);CHKERRQ(ierr);
      ierr = MPI_Allreduce(&stageInfo[stage].perfInfo.numReductions, &red,       1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm);CHKERRQ(ierr);
    } else {
      ierr = PetscFPrintf(comm, fd, "\n--- Event Stage %d: Unknown\n\n", stage);CHKERRQ(ierr);
      ierr = MPI_Allreduce(&zero,                           &stageTime, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm);CHKERRQ(ierr);
      ierr = MPI_Allreduce(&zero,                           &flops,     1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm);CHKERRQ(ierr);
      ierr = MPI_Allreduce(&zero,                           &mess,      1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm);CHKERRQ(ierr);
      ierr = MPI_Allreduce(&zero,                           &messLen,   1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm);CHKERRQ(ierr);
      ierr = MPI_Allreduce(&zero,                           &red,       1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm);CHKERRQ(ierr);
    }
    mess *= 0.5; messLen *= 0.5; red /= size;

    /* Get total number of events in this stage --
       Currently, a single processor can register more events than another, but events must all be registered in order,
       just like stages. We can removed this requirement if necessary by having a global event numbering and indirection
       on the event ID. This seems best accomplished by assoicating a communicator with each stage.

       Problem: If the event did not happen on proc 1, its name will not be available.
       Problem: Event visibility is not implemented
    */
    
    if (!rank){
      ierr = PetscFPrintf(comm, fd, "class Dummy(object):\n");CHKERRQ(ierr);
      ierr = PetscFPrintf(comm, fd, "    def foo(x):\n");CHKERRQ(ierr);
      ierr = PetscFPrintf(comm, fd, "        print x\n");CHKERRQ(ierr);
      ierr = PetscFPrintf(comm, fd, "Event = {}\n");CHKERRQ(ierr);
    }

    if (localStageUsed[stage]) {
      eventInfo      = stageLog->stageInfo[stage].eventLog->eventInfo;
      localNumEvents = stageLog->stageInfo[stage].eventLog->numEvents;
    } else {
      localNumEvents = 0;
    }
    ierr = MPI_Allreduce(&localNumEvents, &numEvents, 1, MPI_INT, MPI_MAX, comm);CHKERRQ(ierr);
    for(event = 0; event < numEvents; event++) {
      if (localStageUsed[stage] && (event < stageLog->stageInfo[stage].eventLog->numEvents) && (eventInfo[event].depth == 0)) {
        ierr = MPI_Allreduce(&eventInfo[event].count, &maxCt, 1, MPI_INT, MPI_MAX, comm);CHKERRQ(ierr);
        name = stageLog->eventLog->eventInfo[event].name;
      } else {
        ierr = MPI_Allreduce(&ierr, &maxCt, 1, MPI_INT, MPI_MAX, comm);CHKERRQ(ierr);
        name = "";
      }
     
      if (maxCt != 0) {
        ierr = PetscFPrintf(comm, fd,"#\n");CHKERRQ(ierr);
        if (!rank){
          ierr = PetscFPrintf(comm, fd, "%s = Dummy()\n",name);CHKERRQ(ierr);
          ierr = PetscFPrintf(comm, fd, "Event['%s'] = %s\n",name,name);CHKERRQ(ierr);
        }
        /* Count */
        ierr = MPI_Gather(&eventInfo[event].count,1,MPI_INT,mycount,1,MPI_INT,0,comm);CHKERRQ(ierr);
        ierr = PetscFPrintf(comm, fd, "%s.Count = [ ", name);CHKERRQ(ierr); 
          for (i=0; i<size; i++){
            ierr = PetscFPrintf(comm, fd, "  %7d,",mycount[i] );CHKERRQ(ierr); 
          }
          ierr = PetscFPrintf(comm, fd, "]\n" );CHKERRQ(ierr); 

        /* Time */
        ierr = MPI_Gather(&eventInfo[event].time,1,MPIU_PETSCLOGDOUBLE,mydata,1,MPIU_PETSCLOGDOUBLE,0,comm);CHKERRQ(ierr);
        if (!rank){
          ierr = PetscFPrintf(comm, fd, "%s.Time  = [ ", name);CHKERRQ(ierr);
          for (i=0; i<size; i++){
            ierr = PetscFPrintf(comm, fd, "  %5.3e,",mydata[i] );CHKERRQ(ierr); 
          }
          ierr = PetscFPrintf(comm, fd, "]\n" );CHKERRQ(ierr); 
        }
        /* Flops */
        ierr = MPI_Gather(&eventInfo[event].flops,1,MPIU_PETSCLOGDOUBLE,mydata,1,MPIU_PETSCLOGDOUBLE,0,comm);CHKERRQ(ierr);
        if (!rank){
          ierr = PetscFPrintf(comm, fd, "%s.Flops = [ ", name);CHKERRQ(ierr);
          for (i=0; i<size; i++){
            ierr = PetscFPrintf(comm, fd, "  %5.3e,",mydata[i] );CHKERRQ(ierr); 
          }
          ierr = PetscFPrintf(comm, fd, "]\n" );CHKERRQ(ierr); 
        }       
      }
    }
  }

  /* Right now, only stages on the first processor are reported here, meaning only objects associated with
     the global communicator, or MPI_COMM_SELF for proc 1. We really should report global stats and then
     stats for stages local to processor sets.
  */
  for(stage = 0; stage < numStages; stage++) {
    if (localStageUsed[stage]) {
      classInfo = stageLog->stageInfo[stage].classLog->classInfo;
    } else {
      ierr = PetscFPrintf(comm, fd, "\n--- Event Stage %d: Unknown\n\n", stage);CHKERRQ(ierr);
    }
  }

  ierr = PetscFree(localStageUsed);CHKERRQ(ierr);
  ierr = PetscFree(stageUsed);CHKERRQ(ierr);
  ierr = PetscFree(localStageVisible);CHKERRQ(ierr);
  ierr = PetscFree(stageVisible);CHKERRQ(ierr);
  ierr = PetscFree(mydata);CHKERRQ(ierr);
  ierr = PetscFree(mycount);CHKERRQ(ierr);

  /* Information unrelated to this particular run */
  ierr = PetscFPrintf(comm, fd,
    "# ========================================================================================================================\n");CHKERRQ(ierr);
  PetscTime(y); 
  PetscTime(x);
  PetscTime(y); PetscTime(y); PetscTime(y); PetscTime(y); PetscTime(y);
  PetscTime(y); PetscTime(y); PetscTime(y); PetscTime(y); PetscTime(y);
  ierr = PetscFPrintf(comm,fd,"AveragetimetogetPetscTime = %g\n", (y-x)/10.0);CHKERRQ(ierr);
  /* MPI information */
  if (size > 1) {
    MPI_Status  status;
    PetscMPIInt tag;
    MPI_Comm    newcomm;

    ierr = MPI_Barrier(comm);CHKERRQ(ierr);
    PetscTime(x);
    ierr = MPI_Barrier(comm);CHKERRQ(ierr);
    ierr = MPI_Barrier(comm);CHKERRQ(ierr);
    ierr = MPI_Barrier(comm);CHKERRQ(ierr);
    ierr = MPI_Barrier(comm);CHKERRQ(ierr);
    ierr = MPI_Barrier(comm);CHKERRQ(ierr);
    PetscTime(y);
    ierr = PetscFPrintf(comm, fd, "AveragetimeforMPI_Barrier = %g\n", (y-x)/5.0);CHKERRQ(ierr);
    ierr = PetscCommDuplicate(comm,&newcomm, &tag);CHKERRQ(ierr);
    ierr = MPI_Barrier(comm);CHKERRQ(ierr);
    if (rank) {
      ierr = MPI_Recv(0, 0, MPI_INT, rank-1,            tag, newcomm, &status);CHKERRQ(ierr);
      ierr = MPI_Send(0, 0, MPI_INT, (rank+1)%size, tag, newcomm);CHKERRQ(ierr);
    } else {
      PetscTime(x);
      ierr = MPI_Send(0, 0, MPI_INT, 1,          tag, newcomm);CHKERRQ(ierr);
      ierr = MPI_Recv(0, 0, MPI_INT, size-1, tag, newcomm, &status);CHKERRQ(ierr);
      PetscTime(y);
      ierr = PetscFPrintf(comm,fd,"AveragetimforzerosizeMPI_Send = %g\n", (y-x)/size);CHKERRQ(ierr);
    }
    ierr = PetscCommDestroy(&newcomm);CHKERRQ(ierr);
  }
  if (!rank) { /* print Optiontable */
    ierr = PetscFPrintf(comm,fd,"# ");CHKERRQ(ierr);
    //ierr = PetscOptionsPrint(fd);CHKERRQ(ierr);
  }

  /* Cleanup */
  ierr = PetscFPrintf(comm, fd, "\n");CHKERRQ(ierr);
  ierr = StageLogPush(stageLog, lastStage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
