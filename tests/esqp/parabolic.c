#include "taosolver.h"
#include "petsctime.h"
typedef struct {
  PetscInt n; // Number of variables
  PetscInt m; // Number of constraints
  PetscInt mx; // grid points in each direction
  PetscInt nt; // Number of time steps; as of now, must be divisible by 8
  PetscInt ndata; // Numbers of data points per sample
  IS s_is;
  IS d_is;
  VecScatter state_scatter;
  VecScatter design_scatter;
  VecScatter *yi_scatter;
  VecScatter *di_scatter;

  Mat Js,Jd,JsBlockPrec,JsInv,JsBlock;
  PetscBool jformed,dsg_formed;

  PetscReal alpha; // Regularization parameter
  PetscReal beta; // Weight attributed to ||u||^2 in regularization functional
  PetscReal noise; // Amount of noise to add to data
  PetscReal ht; // Time step
  //Mat Q,QT,Qblock,QblockT;
  Mat Qblock,QblockT;
  Mat L,LT;
  Mat Div,Divwork;
  Mat Grad;
  Mat Av,Avwork,AvT;
  Mat DSG;
  //Mat Hs,Hd,Hsd;
  Vec q;
  Vec ur; // reference

  Vec d;
  Vec dwork;
  Vec *di;

  Vec y; // state variables
  Vec ywork;
  Vec ysave;
  Vec ytrue;
  Vec *yi,*yiwork;

  Vec u; // design variables
  Vec uwork;
  Vec usave;
  Vec utrue;
 
  Vec js_diag;
  
  Vec c; // constraint vector
  Vec cwork;
  
  Vec lwork;
  Vec S;
  Vec Rwork,Swork,Twork;
  Vec Av_u;

  KSP solver;
  PC prec;

} AppCtx;


PetscErrorCode FormFunction(TaoSolver, Vec, PetscReal*, void*);
PetscErrorCode FormGradient(TaoSolver, Vec, Vec, void*);
PetscErrorCode FormFunctionGradient(TaoSolver, Vec, PetscReal*, Vec, void*);
PetscErrorCode FormJacobianState(TaoSolver, Vec, Mat*, Mat*, Mat*, MatStructure*,void*);
PetscErrorCode FormJacobianDesign(TaoSolver, Vec, Mat*, Mat*, MatStructure*,void*);
PetscErrorCode FormConstraints(TaoSolver, Vec, Vec, void*);
PetscErrorCode FormHessian(TaoSolver, Vec, Mat*, Mat*, MatStructure*, void*);
PetscErrorCode Gather(Vec x, Vec state, VecScatter s_scat, Vec design, VecScatter d_scat);
PetscErrorCode Scatter(Vec x, Vec state, VecScatter s_scat, Vec design, VecScatter d_scat);
PetscErrorCode ParabolicInitialize(AppCtx *user);
PetscErrorCode ParabolicDestroy(AppCtx *user);
PetscErrorCode ParabolicMonitor(TaoSolver, void*);

PetscErrorCode StateMatMult(Mat,Vec,Vec);
PetscErrorCode StateMatBlockMult(Mat,Vec,Vec);
PetscErrorCode StateMatMultTranspose(Mat,Vec,Vec);
PetscErrorCode StateMatGetDiagonal(Mat,Vec); 
PetscErrorCode StateMatDuplicate(Mat,MatDuplicateOption,Mat*);
PetscErrorCode StateMatInvMult(Mat,Vec,Vec);
PetscErrorCode StateMatInvTransposeMult(Mat,Vec,Vec);
PetscErrorCode StateMatBlockPrecMult(PC,Vec,Vec);

PetscErrorCode DesignMatMult(Mat,Vec,Vec);
PetscErrorCode DesignMatMultTranspose(Mat,Vec,Vec);

PetscErrorCode Gather_yi(Vec,Vec*,VecScatter*,PetscInt);
PetscErrorCode Scatter_yi(Vec,Vec*,VecScatter*,PetscInt);

static  char help[]="";

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  PetscErrorCode ierr;
  Vec x,x0;
  
  TaoSolver tao;
  TaoSolverTerminationReason reason;
  AppCtx user;
  IS is_allstate,is_alldesign;
  PetscInt lo,hi,hi2,lo2;
  PetscBool flag;
  

  PetscInitialize(&argc, &argv, (char*)0,help);
  TaoInitialize(&argc, &argv, (char*)0,help);

  user.mx = 8;
  ierr = PetscOptionsInt("-mx","Number of grid points in each direction","",user.mx,&user.mx,&flag); CHKERRQ(ierr);
  user.nt = 8;
  ierr = PetscOptionsInt("-nt","Number of time steps","",user.nt,&user.nt,&flag); CHKERRQ(ierr);
  user.ndata = 64;
  ierr = PetscOptionsInt("-ndata","Numbers of data points per sample","",user.ndata,&user.ndata,&flag); CHKERRQ(ierr);
  user.alpha = 1.0;
  ierr = PetscOptionsReal("-alpha","Regularization parameter","",user.alpha,&user.alpha,&flag); CHKERRQ(ierr);
  user.beta = 0.01;
  ierr = PetscOptionsReal("-beta","Weight attributed to ||u||^2 in regularization functional","",user.beta,&user.beta,&flag); CHKERRQ(ierr);
  user.noise = 0.01;
  ierr = PetscOptionsReal("-noise","Amount of noise to add to data","",user.noise,&user.noise,&flag); CHKERRQ(ierr);

  user.m = user.mx*user.mx*user.mx; // number of constraints
  user.n = user.m*(user.nt+1); // number of variables
  user.ht = (PetscReal)1/user.nt;
  /*ierr = PetscMalloc(user.m*sizeof(PetscInt),&idx); CHKERRQ(ierr);
  for (i=0;i<user.m;i++) idx[i]=i;
  ierr = ISCreateGeneral(PETSC_COMM_SELF,user.m,idx,PETSC_COPY_VALUES,&user.s_is); CHKERRQ(ierr);
  ierr = PetscFree(idx); CHKERRQ(ierr);*/

  ierr = VecCreate(PETSC_COMM_WORLD,&user.u); CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&user.y); CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&user.c); CHKERRQ(ierr);
  ierr = VecSetSizes(user.u,PETSC_DECIDE,user.n-user.m*user.nt); CHKERRQ(ierr);
  ierr = VecSetSizes(user.y,PETSC_DECIDE,user.m*user.nt); CHKERRQ(ierr);
  ierr = VecSetSizes(user.c,PETSC_DECIDE,user.m*user.nt); CHKERRQ(ierr);
  ierr = VecSetFromOptions(user.u); CHKERRQ(ierr);
  ierr = VecSetFromOptions(user.y); CHKERRQ(ierr);
  ierr = VecSetFromOptions(user.c); CHKERRQ(ierr);

  /* Create scatters for reduced spaces.
     If the state vector y and design vector u are partitioned as 
     [y_1; y_2; ...; y_np] and [u_1; u_2; ...; u_np] (with np = # of processors),
     then the solution vector x is organized as
     [y_1; u_1; y_2; u_2; ...; y_np; u_np]. 
     The index sets user.s_is and user.d_is correspond to the indices of the
     state and design variables owned by the current processor.
  */
  ierr = VecCreate(PETSC_COMM_WORLD,&x); CHKERRQ(ierr);

  ierr = VecGetOwnershipRange(user.y,&lo,&hi); CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(user.u,&lo2,&hi2); CHKERRQ(ierr); 

  ierr = ISCreateStride(PETSC_COMM_SELF,hi-lo,lo,1,&is_allstate); CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_SELF,hi-lo,lo+lo2,1,&user.s_is); CHKERRQ(ierr);

  ierr = ISCreateStride(PETSC_COMM_SELF,hi2-lo2,lo2,1,&is_alldesign); CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_SELF,hi2-lo2,hi+lo2,1,&user.d_is); CHKERRQ(ierr);

  ierr = VecSetSizes(x,hi-lo+hi2-lo2,user.n); CHKERRQ(ierr);
  ierr = VecSetFromOptions(x); CHKERRQ(ierr);

  ierr = VecScatterCreate(x,user.s_is,user.y,is_allstate,&user.state_scatter); CHKERRQ(ierr);
  ierr = VecScatterCreate(x,user.d_is,user.u,is_alldesign,&user.design_scatter); CHKERRQ(ierr);
  ierr = ISDestroy(&is_alldesign); CHKERRQ(ierr);
  ierr = ISDestroy(&is_allstate); CHKERRQ(ierr);

  /* Set up initial vectors and matrices */
  ierr = ParabolicInitialize(&user); CHKERRQ(ierr);

  ierr = Gather(x,user.y,user.state_scatter,user.u,user.design_scatter); CHKERRQ(ierr);
  ierr = VecDuplicate(x,&x0); CHKERRQ(ierr);
  ierr = VecCopy(x,x0); CHKERRQ(ierr);


  /* Create TAO solver and set desired solution method */
  ierr = TaoSolverCreate(PETSC_COMM_WORLD,&tao); CHKERRQ(ierr);
  //ierr = TaoSolverSetType(tao,"tao_sqpcon"); CHKERRQ(ierr);
  ierr = TaoSolverSetType(tao,"tao_lcl"); CHKERRQ(ierr);

  //ierr = TaoSolverSetMonitor(tao,ParabolicMonitor,&user); CHKERRQ(ierr);

  /* Set solution vector with an initial guess */
  ierr = TaoSolverSetInitialVector(tao,x); CHKERRQ(ierr);
  ierr = TaoSolverSetObjectiveRoutine(tao, FormFunction, (void *)&user); CHKERRQ(ierr);
  ierr = TaoSolverSetGradientRoutine(tao, FormGradient, (void *)&user); CHKERRQ(ierr);
  ierr = TaoSolverSetConstraintsRoutine(tao, user.c, FormConstraints, (void *)&user); CHKERRQ(ierr);

  ierr = TaoSolverSetJacobianStateRoutine(tao, user.Js, user.Js, user.JsInv, FormJacobianState, (void *)&user); CHKERRQ(ierr); // TODO(?): remove JsInv, use MATOP_SOLVE and MATOP_SOLVE_TRANSPOSE
  ierr = TaoSolverSetJacobianDesignRoutine(tao, user.Jd, user.Jd, FormJacobianDesign, (void *)&user); CHKERRQ(ierr);

  ierr = TaoSolverSetFromOptions(tao); CHKERRQ(ierr);
  ierr = TaoSolverLCLSetStateDesignIS(tao,user.s_is,user.d_is); CHKERRQ(ierr);
  ierr = TaoSolverSQPCONSetStateDesignIS(tao,user.s_is,user.d_is); CHKERRQ(ierr);

 /* SOLVE THE APPLICATION */
  PetscInt ntests = 1;
  ierr = PetscOptionsInt("-ntests","Number of times to repeat TaoSolverSolve","",ntests,&ntests,&flag); CHKERRQ(ierr);
  PetscLogDouble v1,v2;
  PetscInt i;
  for (i=0; i<ntests; i++){
    ierr = PetscGetTime(&v1); CHKERRQ(ierr);
    ierr = TaoSolverSolve(tao);  CHKERRQ(ierr);
    ierr = PetscGetTime(&v2); CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD,"Elapsed time = %10.8f\n",v2-v1);
    ierr = VecCopy(x0,x); CHKERRQ(ierr);
    ierr = TaoSolverSetInitialVector(tao,x); CHKERRQ(ierr);
    //ierr = TaoSolverSetFromOptions(tao); CHKERRQ(ierr);
  }

  ierr = TaoSolverGetTerminationReason(tao,&reason); CHKERRQ(ierr);

  if (reason < 0)
  {
    PetscPrintf(MPI_COMM_WORLD, "TAO failed to converge.\n");
  }
  else
  {
    PetscPrintf(MPI_COMM_WORLD, "Optimization terminated with status %2d.\n", reason);
  }


  /* Free TAO data structures */
  ierr = TaoSolverDestroy(&tao); CHKERRQ(ierr);

  /* Free PETSc data structures */
  ierr = VecDestroy(&x); CHKERRQ(ierr);
  ierr = VecDestroy(&x0); CHKERRQ(ierr);
  //ierr = VecDestroy(&c); CHKERRQ(ierr);
  ierr = ParabolicDestroy(&user); CHKERRQ(ierr);

  /* Finalize TAO, PETSc */
  TaoFinalize();
  PetscFinalize();

  return 0;
}
/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "FormFunction"
/* 
   dwork = Qy - d  
   lwork = L*(u-ur)
   f = 1/2 * (dwork.dork + alpha*lwork.lwork)
*/
PetscErrorCode FormFunction(TaoSolver tao,Vec X,PetscReal *f,void *ptr)
{
  PetscErrorCode ierr;
  PetscReal d1=0,d2=0;
  PetscInt i;
  AppCtx *user = (AppCtx*)ptr;
  PetscFunctionBegin;
  ierr = Scatter(X,user->y,user->state_scatter,user->u,user->design_scatter); CHKERRQ(ierr);
  //ierr = MatMult(user->Q,user->y,user->dwork); CHKERRQ(ierr);
  ierr = Scatter_yi(user->y,user->yi,user->yi_scatter,user->nt); CHKERRQ(ierr);
  // TODO: fix this -- should only sample at 8 time points, not all nt of them
  for (i=0; i<user->nt; i++){
    ierr = MatMult(user->Qblock,user->yi[i],user->di[i]); CHKERRQ(ierr);
  }
  ierr = Gather_yi(user->dwork,user->di,user->di_scatter,user->nt); CHKERRQ(ierr);
  ierr = VecAXPY(user->dwork,-1.0,user->d); CHKERRQ(ierr);
  ierr = VecDot(user->dwork,user->dwork,&d1); CHKERRQ(ierr);

  ierr = VecWAXPY(user->uwork,-1.0,user->ur,user->u); CHKERRQ(ierr);
  ierr = MatMult(user->L,user->uwork,user->lwork); CHKERRQ(ierr);
  ierr = VecDot(user->lwork,user->lwork,&d2); CHKERRQ(ierr);

  ierr = Scatter_yi(user->dwork,user->di,user->di_scatter,user->nt); CHKERRQ(ierr);

  *f = 0.5 * (d1 + user->alpha*d2);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "FormGradient"
/*  
    state: g_s = Q' *(Qy - d)
    design: g_d = alpha*L'*L*(u-ur)
*/
PetscErrorCode FormGradient(TaoSolver tao,Vec X,Vec G,void *ptr)
{
  PetscErrorCode ierr;
  PetscInt i;
  AppCtx *user = (AppCtx*)ptr;
  PetscFunctionBegin;
  CHKMEMQ;
  ierr = Scatter(X,user->y,user->state_scatter,user->u,user->design_scatter); CHKERRQ(ierr);
  //ierr = MatMult(user->Q,user->y,user->dwork); CHKERRQ(ierr);
  ierr = Scatter_yi(user->y,user->yi,user->yi_scatter,user->nt); CHKERRQ(ierr);
  for (i=0; i<user->nt; i++){
    ierr = MatMult(user->Qblock,user->yi[i],user->di[i]); CHKERRQ(ierr);
  }
  ierr = Gather_yi(user->dwork,user->di,user->di_scatter,user->nt); CHKERRQ(ierr);
  ierr = VecAXPY(user->dwork,-1.0,user->d); CHKERRQ(ierr);
  //ierr = MatMultTranspose(user->Q,user->dwork,user->ywork); CHKERRQ(ierr);
  //ierr = MatMult(user->QT,user->dwork,user->ywork); CHKERRQ(ierr);
  ierr = Scatter_yi(user->dwork,user->di,user->di_scatter,user->nt); CHKERRQ(ierr);
    for (i=0; i<user->nt; i++){
    ierr = MatMult(user->QblockT,user->di[i],user->yiwork[i]); CHKERRQ(ierr);
  }
  ierr = Gather_yi(user->ywork,user->yiwork,user->yi_scatter,user->nt); CHKERRQ(ierr);
  
  ierr = VecWAXPY(user->uwork,-1.0,user->ur,user->u); CHKERRQ(ierr);
  ierr = MatMult(user->L,user->uwork,user->lwork); CHKERRQ(ierr);
  //ierr = MatMultTranspose(user->L,user->lwork,user->uwork); CHKERRQ(ierr);
  ierr = MatMult(user->LT,user->lwork,user->uwork); CHKERRQ(ierr);
  ierr = VecScale(user->uwork, user->alpha); CHKERRQ(ierr);

		      
  ierr = Gather(G,user->ywork,user->state_scatter,user->uwork,user->design_scatter); CHKERRQ(ierr);
  CHKMEMQ;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormFunctionGradient"
PetscErrorCode FormFunctionGradient(TaoSolver tao, Vec X, PetscReal *f, Vec G, void *ptr)
{
  PetscErrorCode ierr;
  PetscReal d1,d2;
  PetscInt i;
  AppCtx *user = (AppCtx*)ptr;
  PetscFunctionBegin;
  ierr = Scatter(X,user->y,user->state_scatter,user->u,user->design_scatter); CHKERRQ(ierr);

  //ierr = MatMult(user->Q,user->y,user->dwork); CHKERRQ(ierr);
  ierr = Scatter_yi(user->y,user->yi,user->yi_scatter,user->nt); CHKERRQ(ierr);
  for (i=0; i<user->nt; i++){
    ierr = MatMult(user->Qblock,user->yi[i],user->di[i]); CHKERRQ(ierr);
  }
  ierr = Gather_yi(user->dwork,user->di,user->di_scatter,user->nt); CHKERRQ(ierr);
  ierr = VecAXPY(user->dwork,-1.0,user->d); CHKERRQ(ierr);
  ierr = VecDot(user->dwork,user->dwork,&d1); CHKERRQ(ierr);
  //ierr = MatMultTranspose(user->Q,user->dwork,user->ywork); CHKERRQ(ierr);
  //ierr = MatMult(user->QT,user->dwork,user->ywork); CHKERRQ(ierr);
  ierr = Scatter_yi(user->dwork,user->di,user->di_scatter,user->nt); CHKERRQ(ierr);
    for (i=0; i<user->nt; i++){
    ierr = MatMult(user->QblockT,user->di[i],user->yiwork[i]); CHKERRQ(ierr);
  }
  ierr = Gather_yi(user->ywork,user->yiwork,user->yi_scatter,user->nt); CHKERRQ(ierr);

  ierr = VecWAXPY(user->uwork,-1.0,user->ur,user->u); CHKERRQ(ierr);
  ierr = MatMult(user->L,user->uwork,user->lwork); CHKERRQ(ierr);
  ierr = VecDot(user->lwork,user->lwork,&d2); CHKERRQ(ierr);
  //ierr = MatMultTranspose(user->L,user->lwork,user->uwork); CHKERRQ(ierr);
  ierr = MatMult(user->LT,user->lwork,user->uwork); CHKERRQ(ierr);
  ierr = VecScale(user->uwork, user->alpha); CHKERRQ(ierr);
  *f = 0.5 * (d1 + user->alpha*d2); 
  
  ierr = Gather(G,user->ywork,user->state_scatter,user->uwork,user->design_scatter); CHKERRQ(ierr);
  PetscFunctionReturn(0);

}


/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "FormJacobianState"
/* A 
MatShell object
*/
PetscErrorCode FormJacobianState(TaoSolver tao, Vec X, Mat *J, Mat* JPre, Mat* JInv, MatStructure* flag, void *ptr)
{
  PetscErrorCode ierr;
  AppCtx *user = (AppCtx*)ptr;
  PetscFunctionBegin;
  ierr = Scatter(X,user->ysave,user->state_scatter,user->usave,user->design_scatter); CHKERRQ(ierr);
  ierr = VecSet(user->uwork,0); CHKERRQ(ierr);
  ierr = VecAXPY(user->uwork,-1.0,user->usave); CHKERRQ(ierr);
  ierr = VecExp(user->uwork); CHKERRQ(ierr);
  ierr = MatMult(user->Av,user->uwork,user->Av_u); CHKERRQ(ierr);
  ierr = VecCopy(user->Av_u,user->Swork); CHKERRQ(ierr); 
  ierr = VecReciprocal(user->Swork); CHKERRQ(ierr);
  ierr = MatCopy(user->Div,user->Divwork,SAME_NONZERO_PATTERN); CHKERRQ(ierr);
  ierr = MatDiagonalScale(user->Divwork,PETSC_NULL,user->Swork); CHKERRQ(ierr);
  if (user->dsg_formed) {
    ierr = MatMatMult(user->Divwork,user->Grad,MAT_REUSE_MATRIX,1,&user->DSG); CHKERRQ(ierr);
  } else {
    //ierr = MatMatMult(user->Div,user->Grad,MAT_INITIAL_MATRIX,1,&user->DSG); CHKERRQ(ierr);
    ierr = MatMatMult(user->Divwork,user->Grad,MAT_INITIAL_MATRIX,1,&user->DSG); CHKERRQ(ierr);
    user->dsg_formed = PETSC_TRUE;
  }
  
  /* B = speye(nx^3) + ht*DSG; */
  ierr = MatScale(user->DSG,user->ht); CHKERRQ(ierr);
  ierr = MatShift(user->DSG,1.0); CHKERRQ(ierr);
    
  //*JPre = user->DSG;
  //*flag = SAME_NONZERO_PATTERN;
  
  *JPre = user->JsBlockPrec;
  *flag = DIFFERENT_NONZERO_PATTERN;

  *JInv = user->JsInv;

  PetscFunctionReturn(0);
}
/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "FormJacobianDesign"
/* B */
PetscErrorCode FormJacobianDesign(TaoSolver tao, Vec X, Mat *J, Mat* JPre, MatStructure* flag, void *ptr)
{
  PetscErrorCode ierr;
  AppCtx *user = (AppCtx*)ptr;
  PetscFunctionBegin;

  ierr = Scatter(X,user->y,user->state_scatter,user->u,user->design_scatter); CHKERRQ(ierr);
  /*ierr = MatMult(user->Grad,user->y,user->Swork); CHKERRQ(ierr);
  ierr = VecScale(user->u, -1.0); CHKERRQ(ierr);
  ierr = VecExp(user->u); CHKERRQ(ierr);
  ierr = MatMult(user->Av, user->u, user->S); CHKERRQ(ierr);
  ierr = VecPointwiseMult(user->S,user->S,user->S); CHKERRQ(ierr);
  ierr = VecPointwiseDivide(user->Swork,user->Swork,user->S); CHKERRQ(ierr);
  ierr = MatCopy(user->Av,user->Avwork,SAME_NONZERO_PATTERN); CHKERRQ(ierr);
  ierr = MatDiagonalScale(user->Avwork,user->Swork,user->u); CHKERRQ(ierr);
  if (user->jformed) {
    ierr = MatMatMult(user->Div,user->Avwork,MAT_REUSE_MATRIX,1,J); CHKERRQ(ierr);
  }
 
  else {
    user->jformed=PETSC_TRUE;
    ierr = MatMatMult(user->Div,user->Avwork,MAT_INITIAL_MATRIX,1,J); CHKERRQ(ierr);
  }
  *flag = DIFFERENT_NONZERO_PATTERN;*/

  *J = user->Jd;
  *flag = DIFFERENT_NONZERO_PATTERN;

  PetscFunctionReturn(0);
}



#undef __FUNCT__
#define __FUNCT__ "StateMatMult"
PetscErrorCode StateMatMult(Mat J_shell, Vec X, Vec Y) 
{
  PetscErrorCode ierr;
  PetscInt i;
  void *ptr;
  AppCtx *user;
  PetscFunctionBegin;
  ierr = MatShellGetContext(J_shell,&ptr); CHKERRQ(ierr);
  user = (AppCtx*)ptr;

  ierr = Scatter_yi(X,user->yi,user->yi_scatter,user->nt); CHKERRQ(ierr);
  
  ierr = MatMult(user->JsBlock,user->yi[0],user->yiwork[0]); CHKERRQ(ierr);

  for (i=1; i<user->nt; i++){
    ierr = MatMult(user->JsBlock,user->yi[i],user->yiwork[i]); CHKERRQ(ierr);
    ierr = VecAXPY(user->yiwork[i],-1.0,user->yi[i-1]); CHKERRQ(ierr);
  }

  ierr = Gather_yi(Y,user->yiwork,user->yi_scatter,user->nt); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StateMatMultTranspose"
PetscErrorCode StateMatMultTranspose(Mat J_shell, Vec X, Vec Y) 
{
  PetscErrorCode ierr;
  PetscInt i;
  void *ptr;
  AppCtx *user;
  PetscFunctionBegin;
  ierr = MatShellGetContext(J_shell,&ptr); CHKERRQ(ierr);
  user = (AppCtx*)ptr;

  ierr = Scatter_yi(X,user->yi,user->yi_scatter,user->nt); CHKERRQ(ierr);

  for (i=0; i<user->nt-1; i++){
    ierr = MatMult(user->JsBlock,user->yi[i],user->yiwork[i]); CHKERRQ(ierr);
    ierr = VecAXPY(user->yiwork[i],-1.0,user->yi[i+1]); CHKERRQ(ierr);
  }

  i = user->nt-1;
  ierr = MatMult(user->JsBlock,user->yi[i],user->yiwork[i]); CHKERRQ(ierr);

  ierr = Gather_yi(Y,user->yiwork,user->yi_scatter,user->nt); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StateMatBlockMult"
PetscErrorCode StateMatBlockMult(Mat J_shell, Vec X, Vec Y) 
{
  PetscErrorCode ierr;
  void *ptr;
  AppCtx *user;
  PetscFunctionBegin;
  ierr = MatShellGetContext(J_shell,&ptr); CHKERRQ(ierr);
  user = (AppCtx*)ptr;
   
  ierr = MatMult(user->Grad,X,user->Swork); CHKERRQ(ierr); 
  ierr = VecPointwiseDivide(user->Swork,user->Swork,user->Av_u); CHKERRQ(ierr);
  ierr = MatMult(user->Div,user->Swork,Y); CHKERRQ(ierr); 
  ierr = VecAYPX(Y,user->ht,X); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DesignMatMult"
PetscErrorCode DesignMatMult(Mat J_shell, Vec X, Vec Y) 
{
  PetscErrorCode ierr;
  void *ptr;
  PetscInt i;
  AppCtx *user;
  PetscFunctionBegin;
  ierr = MatShellGetContext(J_shell,&ptr); CHKERRQ(ierr);
  user = (AppCtx*)ptr;
 
  /* sdiag(1./v) */ 
  ierr = VecSet(user->uwork,0); CHKERRQ(ierr);
  ierr = VecAXPY(user->uwork,-1.0,user->u); CHKERRQ(ierr);
  ierr = VecExp(user->uwork); CHKERRQ(ierr);  

  /* sdiag(1./((Av*(1./v)).^2)) */
  ierr = MatMult(user->Av,user->uwork,user->Swork); CHKERRQ(ierr);
  ierr = VecPointwiseMult(user->Swork,user->Swork,user->Swork); CHKERRQ(ierr);
  ierr = VecReciprocal(user->Swork); CHKERRQ(ierr); 

  /* (Av * (sdiag(1./v) * b)) */ 
  ierr = VecPointwiseMult(user->uwork,user->uwork,X); CHKERRQ(ierr);
  ierr = MatMult(user->Av,user->uwork,user->Twork); CHKERRQ(ierr);

  /* (sdiag(1./((Av*(1./v)).^2)) * (Av * (sdiag(1./v) * b))) */
  ierr = VecPointwiseMult(user->Swork,user->Twork,user->Swork); CHKERRQ(ierr); 

  ierr = Scatter_yi(user->y,user->yi,user->yi_scatter,user->nt); CHKERRQ(ierr);
  for (i=0; i<user->nt; i++){
    /* (sdiag(Grad*y(:,i)) */
    ierr = MatMult(user->Grad,user->yi[i],user->Twork); CHKERRQ(ierr);
  
    /* ht * Div * (sdiag(Grad*y(:,i)) * (sdiag(1./((Av*(1./v)).^2)) * (Av * (sdiag(1./v) * b)))) */
    ierr = VecPointwiseMult(user->Twork,user->Twork,user->Swork); CHKERRQ(ierr); 
    ierr = MatMult(user->Div,user->Twork,user->yiwork[i]); CHKERRQ(ierr);
    ierr= VecScale(user->yiwork[i],user->ht); CHKERRQ(ierr);
  }
  ierr = Gather_yi(Y,user->yiwork,user->yi_scatter,user->nt); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DesignMatMultTranspose"
PetscErrorCode DesignMatMultTranspose(Mat J_shell, Vec X, Vec Y) 
{
  PetscErrorCode ierr;
  void *ptr;
  PetscInt i;
  AppCtx *user;
  PetscFunctionBegin;
  ierr = MatShellGetContext(J_shell,&ptr); CHKERRQ(ierr);
  user = (AppCtx*)ptr;

  /* sdiag(1./((Av*(1./v)).^2)) */
  ierr = VecSet(user->uwork,0); CHKERRQ(ierr);
  ierr = VecAXPY(user->uwork,-1.0,user->u); CHKERRQ(ierr);
  ierr = VecExp(user->uwork); CHKERRQ(ierr);
  ierr = MatMult(user->Av,user->uwork,user->Rwork); CHKERRQ(ierr);
  ierr = VecPointwiseMult(user->Rwork,user->Rwork,user->Rwork); CHKERRQ(ierr);
  ierr = VecReciprocal(user->Rwork); CHKERRQ(ierr);

  ierr = VecSet(Y,0.0); CHKERRQ(ierr);
  ierr = Scatter_yi(user->y,user->yi,user->yi_scatter,user->nt); CHKERRQ(ierr);
  ierr = Scatter_yi(X,user->yiwork,user->yi_scatter,user->nt); CHKERRQ(ierr);
  for (i=0; i<user->nt; i++){
    /* (Div' * b(:,i)) */
    ierr = MatMult(user->Grad,user->yiwork[i],user->Swork); CHKERRQ(ierr);

    /* sdiag(Grad*y(:,i)) */
    ierr = MatMult(user->Grad,user->yi[i],user->Twork); CHKERRQ(ierr);

    /* (sdiag(Grad*y(:,i)) * (Div' * b(:,i))) */
    ierr = VecPointwiseMult(user->Twork,user->Swork,user->Twork); CHKERRQ(ierr);

    /* (sdiag(1./((Av*(1./v)).^2)) * (sdiag(Grad*y(:,i)) * (Div' * b(:,i)))) */
    ierr = VecPointwiseMult(user->Twork,user->Rwork,user->Twork); CHKERRQ(ierr);

    /* (Av' * (sdiag(1./((Av*(1./v)).^2)) * (sdiag(Grad*y(:,i)) * (Div' * b(:,i))))) */
    ierr = MatMult(user->AvT,user->Twork,user->yiwork[i]); CHKERRQ(ierr);
  
    /* sdiag(1./v) * (Av' * (sdiag(1./((Av*(1./v)).^2)) * (sdiag(Grad*y(:,i)) * (Div' * b(:,i))))) */
    ierr = VecPointwiseMult(user->yiwork[i],user->uwork,user->yiwork[i]); CHKERRQ(ierr);
    ierr = VecAXPY(Y,user->ht,user->yiwork[i]); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StateMatBlockPrecMult"
PetscErrorCode StateMatBlockPrecMult(PC PC_shell, Vec X, Vec Y) 
{
  PetscErrorCode ierr;
  void *ptr;
  AppCtx *user;
  PetscFunctionBegin;
  ierr = PCShellGetContext(PC_shell,&ptr); CHKERRQ(ierr);
  user = (AppCtx*)ptr;

  if (user->dsg_formed) {
    //ierr = MatSOR(user->DSG,X,1.0,(SOR_ZERO_INITIAL_GUESS | SOR_SYMMETRIC_SWEEP),0.0,1,1,Y); CHKERRQ(ierr);
    ierr = MatSOR(user->DSG,X,1.0,(SOR_ZERO_INITIAL_GUESS | SOR_LOCAL_SYMMETRIC_SWEEP),0.0,1,1,Y); CHKERRQ(ierr);
    //VecCopy(X,Y);
  }
  else {
    printf("DSG not formed"); abort();
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StateMatInvMult"
PetscErrorCode StateMatInvMult(Mat J_shell, Vec X, Vec Y)
{
  PetscErrorCode ierr;
  void *ptr;
  AppCtx *user;
  PetscInt its,i;
  KSPConvergedReason reason;
  PetscFunctionBegin;
  ierr = MatShellGetContext(J_shell,&ptr); CHKERRQ(ierr);
  user = (AppCtx*)ptr;

  ierr = Scatter_yi(X,user->yi,user->yi_scatter,user->nt); CHKERRQ(ierr);
  
  ierr = KSPSolve(user->solver,user->yi[0],user->yiwork[0]); CHKERRQ(ierr); 

  /*ierr = KSPGetConvergedReason(user->solver,&reason); CHKERRQ(ierr);
  if (reason==KSP_DIVERGED_INDEFINITE_PC) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Divergence because of indefinite preconditioner;\n"); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Run the executable again but with -pc_factor_shift_positive_definite option.\n"); CHKERRQ(ierr);
  } else if (reason<0) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Other kind of divergence: this should not happen.\n"); CHKERRQ(ierr);
  } else if (reason==KSP_CONVERGED_RTOL){
    ierr = KSPGetIterationNumber(user->solver,&its); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"RTOL, Convergence in %d iterations.\n",(int)its); CHKERRQ(ierr);
  } else if (reason==KSP_CONVERGED_ATOL){
    ierr = KSPGetIterationNumber(user->solver,&its); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"ATOL, Convergence in %d iterations.\n",(int)its); CHKERRQ(ierr);
    }*/

  for (i=1; i<user->nt; i++){
    ierr = VecAXPY(user->yi[i],1.0,user->yiwork[i-1]); CHKERRQ(ierr);
    ierr = KSPSolve(user->solver,user->yi[i],user->yiwork[i]); CHKERRQ(ierr);

    /*ierr = KSPGetConvergedReason(user->solver,&reason); CHKERRQ(ierr);
    if (reason==KSP_DIVERGED_INDEFINITE_PC) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Divergence because of indefinite preconditioner;\n"); CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Run the executable again but with -pc_factor_shift_positive_definite option.\n"); CHKERRQ(ierr);
    } else if (reason<0) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Other kind of divergence: this should not happen.\n"); CHKERRQ(ierr);
    } else if (reason==KSP_CONVERGED_RTOL){
      ierr = KSPGetIterationNumber(user->solver,&its); CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"RTOL, Convergence in %d iterations.\n",(int)its); CHKERRQ(ierr);
    } else if (reason==KSP_CONVERGED_ATOL){
      ierr = KSPGetIterationNumber(user->solver,&its); CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"ATOL, Convergence in %d iterations.\n",(int)its); CHKERRQ(ierr);
      }*/
  }

  ierr = Gather_yi(Y,user->yiwork,user->yi_scatter,user->nt); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StateMatInvTransposeMult"
PetscErrorCode StateMatInvTransposeMult(Mat J_shell, Vec X, Vec Y)
{
  PetscErrorCode ierr;
  void *ptr;
  AppCtx *user;
  PetscInt its,i;
  KSPConvergedReason reason;
  PetscFunctionBegin;
  ierr = MatShellGetContext(J_shell,&ptr); CHKERRQ(ierr);
  user = (AppCtx*)ptr;

  ierr = Scatter_yi(X,user->yi,user->yi_scatter,user->nt); CHKERRQ(ierr);
  
  i = user->nt - 1;
  ierr = KSPSolve(user->solver,user->yi[i],user->yiwork[i]); CHKERRQ(ierr); 

  /*KSPGetConvergedReason(user->solver,&reason);
  if (reason==KSP_DIVERGED_INDEFINITE_PC) {
    PetscPrintf(PETSC_COMM_WORLD,"Divergence because of indefinite preconditioner;\n");
    PetscPrintf(PETSC_COMM_WORLD,"Run the executable again but with -pc_factor_shift_positive_definite option.\n");
  } else if (reason<0) {
      PetscPrintf(PETSC_COMM_WORLD,"Other kind of divergence: this should not happen.\n");
  } else if (reason==KSP_CONVERGED_RTOL){
    KSPGetIterationNumber(user->solver,&its);
    PetscPrintf(PETSC_COMM_WORLD,"RTOL, Convergence in %d iterations.\n",(int)its);
  } else if (reason==KSP_CONVERGED_ATOL){
    KSPGetIterationNumber(user->solver,&its);
    PetscPrintf(PETSC_COMM_WORLD,"ATOL, Convergence in %d iterations.\n",(int)its);
    }*/

  for (i=user->nt-2; i>=0; i--){
    ierr = VecAXPY(user->yi[i],1.0,user->yiwork[i+1]); CHKERRQ(ierr);
    ierr = KSPSolve(user->solver,user->yi[i],user->yiwork[i]); CHKERRQ(ierr);

    /*KSPGetConvergedReason(user->solver,&reason);
    if (reason==KSP_DIVERGED_INDEFINITE_PC) {
      PetscPrintf(PETSC_COMM_WORLD,"Divergence because of indefinite preconditioner;\n");
      PetscPrintf(PETSC_COMM_WORLD,"Run the executable again but with -pc_factor_shift_positive_definite option.\n");
    } else if (reason<0) {
      PetscPrintf(PETSC_COMM_WORLD,"Other kind of divergence: this should not happen.\n");
    } else if (reason==KSP_CONVERGED_RTOL){
      KSPGetIterationNumber(user->solver,&its);
      PetscPrintf(PETSC_COMM_WORLD,"RTOL, Convergence in %d iterations.\n",(int)its);
    } else if (reason==KSP_CONVERGED_ATOL){
      KSPGetIterationNumber(user->solver,&its);
      PetscPrintf(PETSC_COMM_WORLD,"ATOL, Convergence in %d iterations.\n",(int)its);
      }*/
  }

  ierr = Gather_yi(Y,user->yiwork,user->yi_scatter,user->nt); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StateMatDuplicate"
PetscErrorCode StateMatDuplicate(Mat J_shell, MatDuplicateOption opt, Mat *new_shell)
{
  PetscErrorCode ierr;
  void *ptr;
  AppCtx *user;
  PetscFunctionBegin;
  ierr = MatShellGetContext(J_shell,&ptr); CHKERRQ(ierr);
  user = (AppCtx*)ptr;

  ierr = MatCreateShell(PETSC_COMM_WORLD,PETSC_DETERMINE,PETSC_DETERMINE,user->m,user->m,user,new_shell); CHKERRQ(ierr);
  ierr = MatShellSetOperation(*new_shell,MATOP_MULT,(void(*)(void))StateMatMult); CHKERRQ(ierr);
  ierr = MatShellSetOperation(*new_shell,MATOP_DUPLICATE,(void(*)(void))StateMatDuplicate); CHKERRQ(ierr);
  ierr = MatShellSetOperation(*new_shell,MATOP_MULT_TRANSPOSE,(void(*)(void))StateMatMultTranspose); CHKERRQ(ierr);
  ierr = MatShellSetOperation(*new_shell,MATOP_GET_DIAGONAL,(void(*)(void))StateMatGetDiagonal); CHKERRQ(ierr);
  //ierr = MatSetOption(*new_shell,MAT_SYMMETRY_ETERNAL,PETSC_TRUE); CHKERRQ(ierr);
  //ierr = MatSetOption(user->Js,MAT_SYMMETRY_ETERNAL,PETSC_TRUE); CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StateMatGetDiagonal"
PetscErrorCode StateMatGetDiagonal(Mat J_shell, Vec X)
{
  PetscErrorCode ierr;
  void *ptr;
  AppCtx *user;
  PetscFunctionBegin;
  ierr = MatShellGetContext(J_shell,&ptr); CHKERRQ(ierr);
  user = (AppCtx*)ptr;
  ierr =  VecCopy(user->js_diag,X); CHKERRQ(ierr);
  PetscFunctionReturn(0);
  
}
/*#undef __FUNCT__
#define __FUNCT__ "DesignMatMult"
PetscErrorCode DesignMatMult(Mat J_shell, Vec X, Vec Y) 
{
  PetscErrorCode ierr;
  void *ptr;
  AppCtx *user;
  PetscFunctionBegin;
  ierr = MatShellGetContext(J_shell,&ptr); CHKERRQ(ierr);
  user = (AppCtx*)ptr;
  ierr = Scatter(X,user->y,user->state_scatter,user->u,user->design_scatter); CHKERRQ(ierr);
  PetscFunctionReturn(0);
  }*/

#undef __FUNCT__
#define __FUNCT__ "FormConstraints"
PetscErrorCode FormConstraints(TaoSolver tao, Vec X, Vec C, void *ptr)
{
  /* con = Ay - q, A = [B  0  0 ... 0; 
                       -I  B  0 ... 0; 
                        0 -I  B ... 0;
                             ...     ;
                        0    ... -I B] 
     B = ht * Div * Sigma * Grad + eye */
  PetscErrorCode ierr;
  PetscInt i;
  AppCtx *user = (AppCtx*)ptr;
  PetscFunctionBegin;
   
  ierr = Scatter(X,user->y,user->state_scatter,user->u,user->design_scatter); CHKERRQ(ierr);
  ierr = Scatter_yi(user->y,user->yi,user->yi_scatter,user->nt); CHKERRQ(ierr);
   
  ierr = MatMult(user->JsBlock,user->yi[0],user->yiwork[0]); CHKERRQ(ierr);

  for (i=1; i<user->nt; i++){
    ierr = MatMult(user->JsBlock,user->yi[i],user->yiwork[i]); CHKERRQ(ierr);
    ierr = VecAXPY(user->yiwork[i],-1.0,user->yi[i-1]); CHKERRQ(ierr);		    
  }

  ierr = Gather_yi(C,user->yiwork,user->yi_scatter,user->nt); CHKERRQ(ierr);
  ierr = VecAXPY(C,-1.0,user->q); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "Scatter"
PetscErrorCode Scatter(Vec x, Vec state, VecScatter s_scat, Vec design, VecScatter d_scat)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = VecScatterBegin(s_scat,x,state,INSERT_VALUES,SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(s_scat,x,state,INSERT_VALUES,SCATTER_FORWARD); CHKERRQ(ierr);

  ierr = VecScatterBegin(d_scat,x,design,INSERT_VALUES,SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(d_scat,x,design,INSERT_VALUES,SCATTER_FORWARD); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "Scatter_yi"
PetscErrorCode Scatter_yi(Vec y, Vec *yi, VecScatter *scat, PetscInt nt)
{
  PetscErrorCode ierr;
  PetscInt i;
  PetscFunctionBegin;
  for (i=0; i<nt; i++){
    ierr = VecScatterBegin(scat[i],y,yi[i],INSERT_VALUES,SCATTER_FORWARD); CHKERRQ(ierr);
    ierr = VecScatterEnd(scat[i],y,yi[i],INSERT_VALUES,SCATTER_FORWARD); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "Gather"
PetscErrorCode Gather(Vec x, Vec state, VecScatter s_scat, Vec design, VecScatter d_scat)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = VecScatterBegin(s_scat,state,x,INSERT_VALUES,SCATTER_REVERSE); CHKERRQ(ierr);
  ierr = VecScatterEnd(s_scat,state,x,INSERT_VALUES,SCATTER_REVERSE); CHKERRQ(ierr);
  ierr = VecScatterBegin(d_scat,design,x,INSERT_VALUES,SCATTER_REVERSE); CHKERRQ(ierr);
  ierr = VecScatterEnd(d_scat,design,x,INSERT_VALUES,SCATTER_REVERSE); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "Gather_yi"
PetscErrorCode Gather_yi(Vec y, Vec *yi, VecScatter *scat, PetscInt nt)
{
  PetscErrorCode ierr;
  PetscInt i;
  PetscFunctionBegin;
  for (i=0; i<nt; i++){
    ierr = VecScatterBegin(scat[i],yi[i],y,INSERT_VALUES,SCATTER_REVERSE); CHKERRQ(ierr);
    ierr = VecScatterEnd(scat[i],yi[i],y,INSERT_VALUES,SCATTER_REVERSE); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
  
    
#undef __FUNCT__
#define __FUNCT__ "ParabolicInitialize"
PetscErrorCode ParabolicInitialize(AppCtx *user)
{
  PetscErrorCode ierr;
  PetscInt m,n,i,j,k,linear_index,istart,iend,iblock,lo,hi,lo2,hi2;
  PetscInt nnz[user->mx * user->mx * user->mx + 3 * user->mx * user->mx * (user->mx-1)];
  Vec XX,YY,ZZ,XXwork,YYwork,ZZwork,UTwork,yi,di,bc;
  PetscReal x[user->mx],y[user->mx],z[user->mx];
  PetscReal h;
  PetscScalar hinv,neg_hinv,half = 0.5,sqrt_beta;
  PetscInt im,indx1,indx2,indy1,indy2,indz1,indz2,nx,ny,nz;
  PetscReal xri,yri,zri,xim,yim,zim,dx1,dx2,dy1,dy2,dz1,dz2,Dx,Dy,Dz;
  PetscScalar v,vx,vy,vz;
  IS is_from_y,is_to_yi,is_from_d,is_to_di;

  PetscFunctionBegin;
  user->jformed = PETSC_FALSE;
  user->dsg_formed = PETSC_FALSE;

  n = user->mx * user->mx * user->mx;
  m = 3 * user->mx * user->mx * (user->mx-1);
  sqrt_beta = sqrt(user->beta);

  ierr = VecCreate(PETSC_COMM_WORLD,&XX); CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&user->q); CHKERRQ(ierr);
  ierr = VecSetSizes(XX,PETSC_DECIDE,n); CHKERRQ(ierr);
  ierr = VecSetSizes(user->q,PETSC_DECIDE,n*user->nt); CHKERRQ(ierr);
  ierr = VecSetFromOptions(XX); CHKERRQ(ierr);
  ierr = VecSetFromOptions(user->q); CHKERRQ(ierr);

  ierr = VecDuplicate(XX,&YY); CHKERRQ(ierr);
  ierr = VecDuplicate(XX,&ZZ); CHKERRQ(ierr);
  ierr = VecDuplicate(XX,&XXwork); CHKERRQ(ierr);
  ierr = VecDuplicate(XX,&YYwork); CHKERRQ(ierr);
  ierr = VecDuplicate(XX,&ZZwork); CHKERRQ(ierr);
  ierr = VecDuplicate(XX,&UTwork); CHKERRQ(ierr);
  ierr = VecDuplicate(XX,&user->utrue); CHKERRQ(ierr);
  ierr = VecDuplicate(XX,&bc); CHKERRQ(ierr);

  /* Generate 3D grid, and collect ns (1<=ns<=8) right-hand-side vectors into user->q */
  h = 1.0/user->mx; 
  hinv = user->mx;
  neg_hinv = -hinv;
  for (i=0; i<user->mx; i++){ // TODO: loop over VecGetOwnershipRange values
    for (j=0; j<user->mx; j++){
      for (k=0; k<user->mx; k++){
	linear_index = i + j*user->mx + k*user->mx*user->mx;
	vx = h*(i+0.5); 
	vy = h*(j+0.5);
	vz = h*(k+0.5);	
	ierr = VecSetValues(XX,1,&linear_index,&vx,INSERT_VALUES); CHKERRQ(ierr);
	ierr = VecSetValues(YY,1,&linear_index,&vy,INSERT_VALUES); CHKERRQ(ierr);
	ierr = VecSetValues(ZZ,1,&linear_index,&vz,INSERT_VALUES); CHKERRQ(ierr);
	if ((vx<0.6) && (vx>0.4) && (vy<0.6) && (vy>0.4) && (vy<0.6) && (vz<0.6) && (vz>0.4)){
	  v = 1000.0;
	  ierr = VecSetValues(bc,1,&linear_index,&v,INSERT_VALUES); CHKERRQ(ierr);
	}
      }
    }
  }

  ierr = VecAssemblyBegin(XX); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(XX); CHKERRQ(ierr);
  ierr = VecAssemblyBegin(YY); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(YY); CHKERRQ(ierr);
  ierr = VecAssemblyBegin(ZZ); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(ZZ); CHKERRQ(ierr);
  ierr = VecAssemblyBegin(bc); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(bc); CHKERRQ(ierr);  

  /* Compute true parameter function
     ut = 0.5 + exp(-10*((x-0.5)^2+(y-0.5)^2+(z-0.5)^2)) */
  ierr = VecCopy(XX,XXwork); CHKERRQ(ierr);
  ierr = VecCopy(YY,YYwork); CHKERRQ(ierr);
  ierr = VecCopy(ZZ,ZZwork); CHKERRQ(ierr);

  ierr = VecShift(XXwork,-0.5); CHKERRQ(ierr);
  ierr = VecShift(YYwork,-0.5); CHKERRQ(ierr);
  ierr = VecShift(ZZwork,-0.5); CHKERRQ(ierr);

  ierr = VecPointwiseMult(XXwork,XXwork,XXwork); CHKERRQ(ierr);
  ierr = VecPointwiseMult(YYwork,YYwork,YYwork); CHKERRQ(ierr);
  ierr = VecPointwiseMult(ZZwork,ZZwork,ZZwork); CHKERRQ(ierr);

  ierr = VecCopy(XXwork,user->utrue); CHKERRQ(ierr);
  ierr = VecAXPY(user->utrue,1.0,YYwork); CHKERRQ(ierr);
  ierr = VecAXPY(user->utrue,1.0,ZZwork); CHKERRQ(ierr);
  ierr = VecScale(user->utrue,-10.0); CHKERRQ(ierr);
  ierr = VecExp(user->utrue); CHKERRQ(ierr);
  ierr = VecShift(user->utrue,0.5); CHKERRQ(ierr);

  ierr = VecDestroy(&XX); CHKERRQ(ierr);
  ierr = VecDestroy(&YY); CHKERRQ(ierr);
  ierr = VecDestroy(&ZZ); CHKERRQ(ierr);
  ierr = VecDestroy(&XXwork); CHKERRQ(ierr);
  ierr = VecDestroy(&YYwork); CHKERRQ(ierr);
  ierr = VecDestroy(&ZZwork); CHKERRQ(ierr);
  ierr = VecDestroy(&UTwork); CHKERRQ(ierr);
 
  /* Initial guess and reference model */
  ierr = VecDuplicate(user->utrue,&user->ur); CHKERRQ(ierr);
  ierr = VecSet(user->ur,0.0); CHKERRQ(ierr);
  ierr = VecCopy(user->ur,user->u); CHKERRQ(ierr);

  /* Generate Grad matrix */
  ierr = MatCreate(PETSC_COMM_WORLD,&user->Grad); CHKERRQ(ierr);
  ierr = MatSetSizes(user->Grad,PETSC_DECIDE,PETSC_DECIDE,m,n); CHKERRQ(ierr);
  ierr = MatSetFromOptions(user->Grad); CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(user->Grad,2,PETSC_NULL,2,PETSC_NULL); CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(user->Grad,2,PETSC_NULL); CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(user->Grad,&istart,&iend); CHKERRQ(ierr);

  for (i=istart; i<iend; i++){
    if (i<m/3){
      iblock = i / (user->mx-1);
      j = iblock*user->mx + (i % (user->mx-1));
      ierr = MatSetValues(user->Grad,1,&i,1,&j,&neg_hinv,INSERT_VALUES); CHKERRQ(ierr);
      j = j+1;
      ierr = MatSetValues(user->Grad,1,&i,1,&j,&hinv,INSERT_VALUES); CHKERRQ(ierr);
    }
    if (i>=m/3 && i<2*m/3){
      iblock = (i-m/3) / (user->mx*(user->mx-1));
      j = iblock*user->mx*user->mx + ((i-m/3) % (user->mx*(user->mx-1)));
      ierr = MatSetValues(user->Grad,1,&i,1,&j,&neg_hinv,INSERT_VALUES); CHKERRQ(ierr);
      j = j + user->mx;
      ierr = MatSetValues(user->Grad,1,&i,1,&j,&hinv,INSERT_VALUES); CHKERRQ(ierr);
    }
    if (i>=2*m/3){
      j = i-2*m/3;
      ierr = MatSetValues(user->Grad,1,&i,1,&j,&neg_hinv,INSERT_VALUES); CHKERRQ(ierr);
      j = j + user->mx*user->mx;
      ierr = MatSetValues(user->Grad,1,&i,1,&j,&hinv,INSERT_VALUES); CHKERRQ(ierr);
    }
  }

  ierr = MatAssemblyBegin(user->Grad,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(user->Grad,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);


  /* Generate arithmetic averaging matrix Av */
  //ierr = MatDuplicate(user->Grad,MAT_DO_NOT_COPY_VALUES,&user->Av);
  ierr = MatCreate(PETSC_COMM_WORLD,&user->Av); CHKERRQ(ierr);
  ierr = MatSetSizes(user->Av,PETSC_DECIDE,PETSC_DECIDE,m,n); CHKERRQ(ierr);
  ierr = MatSetFromOptions(user->Av); CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(user->Av,2,PETSC_NULL,2,PETSC_NULL); CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(user->Av,2,PETSC_NULL); CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(user->Av,&istart,&iend); CHKERRQ(ierr);

  for (i=istart; i<iend; i++){
    if (i<m/3){
      iblock = i / (user->mx-1);
      j = iblock*user->mx + (i % (user->mx-1));
      ierr = MatSetValues(user->Av,1,&i,1,&j,&half,INSERT_VALUES); CHKERRQ(ierr);
      j = j+1;
      ierr = MatSetValues(user->Av,1,&i,1,&j,&half,INSERT_VALUES); CHKERRQ(ierr);
    }
    if (i>=m/3 && i<2*m/3){
      iblock = (i-m/3) / (user->mx*(user->mx-1));
      j = iblock*user->mx*user->mx + ((i-m/3) % (user->mx*(user->mx-1)));
      ierr = MatSetValues(user->Av,1,&i,1,&j,&half,INSERT_VALUES); CHKERRQ(ierr);
      j = j + user->mx;
      ierr = MatSetValues(user->Av,1,&i,1,&j,&half,INSERT_VALUES); CHKERRQ(ierr);
    }
    if (i>=2*m/3){
      j = i-2*m/3;
      ierr = MatSetValues(user->Av,1,&i,1,&j,&half,INSERT_VALUES); CHKERRQ(ierr);
      j = j + user->mx*user->mx;
      ierr = MatSetValues(user->Av,1,&i,1,&j,&half,INSERT_VALUES); CHKERRQ(ierr);
    }
  }

  ierr = MatAssemblyBegin(user->Av,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(user->Av,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  /* Generate transpose of averaging matrix Av */
  ierr = MatTranspose(user->Av,MAT_INITIAL_MATRIX,&user->AvT); CHKERRQ(ierr);


  /* Generate regularization matrix L */
  for (i=0; i<m+n; i++){
    if (i<m){
      nnz[i] = 2; 
    }
    else {
      nnz[i] = 1;
    }
  }

  ierr = MatCreate(PETSC_COMM_WORLD,&user->L); CHKERRQ(ierr);
  ierr = MatSetSizes(user->L,PETSC_DECIDE,PETSC_DECIDE,m+n,n); CHKERRQ(ierr);
  ierr = MatSetFromOptions(user->L); CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(user->L,2,PETSC_NULL,2,PETSC_NULL); CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(user->L,2,nnz); CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(user->L,&istart,&iend);

  for (i=istart; i<iend; i++){
    if (i<m/3){
      iblock = i / (user->mx-1);
      j = iblock*user->mx + (i % (user->mx-1));
      ierr = MatSetValues(user->L,1,&i,1,&j,&neg_hinv,INSERT_VALUES); CHKERRQ(ierr);
      j = j+1;
      ierr = MatSetValues(user->L,1,&i,1,&j,&hinv,INSERT_VALUES); CHKERRQ(ierr);
    }
    if (i>=m/3 && i<2*m/3){
      iblock = (i-m/3) / (user->mx*(user->mx-1));
      j = iblock*user->mx*user->mx + ((i-m/3) % (user->mx*(user->mx-1)));
      ierr = MatSetValues(user->L,1,&i,1,&j,&neg_hinv,INSERT_VALUES); CHKERRQ(ierr);
      j = j + user->mx;
      ierr = MatSetValues(user->L,1,&i,1,&j,&hinv,INSERT_VALUES); CHKERRQ(ierr);
    }
    if (i>=2*m/3 && i<m){
      j = i-2*m/3;
      ierr = MatSetValues(user->L,1,&i,1,&j,&neg_hinv,INSERT_VALUES); CHKERRQ(ierr);
      j = j + user->mx*user->mx;
      ierr = MatSetValues(user->L,1,&i,1,&j,&hinv,INSERT_VALUES); CHKERRQ(ierr);
    }
    if (i>=m){
      j = i - m;
      ierr = MatSetValues(user->L,1,&i,1,&j,&sqrt_beta,INSERT_VALUES); CHKERRQ(ierr);
    }
  }

  ierr = MatAssemblyBegin(user->L,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(user->L,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  ierr = MatScale(user->L,pow(h,1.5)); CHKERRQ(ierr);

  /* Generate Div matrix */
  ierr = MatTranspose(user->Grad,MAT_INITIAL_MATRIX,&user->Div);


  /* Build work vectors and matrices */
  ierr = VecCreate(PETSC_COMM_WORLD,&user->S); CHKERRQ(ierr);
  ierr = VecSetSizes(user->S, PETSC_DECIDE, user->mx*user->mx*(user->mx-1)*3); CHKERRQ(ierr);
  ierr = VecSetFromOptions(user->S); CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD,&user->lwork); CHKERRQ(ierr);
  ierr = VecSetSizes(user->lwork,PETSC_DECIDE,m+user->mx*user->mx*user->mx); CHKERRQ(ierr); 
  ierr = VecSetFromOptions(user->lwork); CHKERRQ(ierr);

  ierr = MatDuplicate(user->Div,MAT_SHARE_NONZERO_PATTERN,&user->Divwork); CHKERRQ(ierr);

  ierr = MatDuplicate(user->Av,MAT_SHARE_NONZERO_PATTERN,&user->Avwork); CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD,&user->d); CHKERRQ(ierr);
  ierr = VecSetSizes(user->d,PETSC_DECIDE,user->ndata*user->nt); CHKERRQ(ierr);
  ierr = VecSetFromOptions(user->d); CHKERRQ(ierr);

  ierr = VecDuplicate(user->S,&user->Swork); CHKERRQ(ierr);
  ierr = VecDuplicate(user->S,&user->Av_u); CHKERRQ(ierr);
  ierr = VecDuplicate(user->S,&user->Twork); CHKERRQ(ierr);
  ierr = VecDuplicate(user->S,&user->Rwork); CHKERRQ(ierr);
  ierr = VecDuplicate(user->y,&user->ywork); CHKERRQ(ierr);
  ierr = VecDuplicate(user->y,&user->ysave); CHKERRQ(ierr);
  ierr = VecDuplicate(user->u,&user->uwork); CHKERRQ(ierr);
  ierr = VecDuplicate(user->u,&user->usave); CHKERRQ(ierr);
  ierr = VecDuplicate(user->u,&user->js_diag); CHKERRQ(ierr);
  //ierr = VecDuplicate(user->u,&user->c); CHKERRQ(ierr);
  ierr = VecDuplicate(user->y,&user->c); CHKERRQ(ierr);
  ierr = VecDuplicate(user->c,&user->cwork); CHKERRQ(ierr);
  ierr = VecDuplicate(user->d,&user->dwork); CHKERRQ(ierr);

  /* Create matrix-free shell user->Js for computing A*x */
  ierr = MatCreateShell(PETSC_COMM_WORLD,PETSC_DETERMINE,PETSC_DETERMINE,user->m*user->nt,user->m*user->nt,user,&user->Js); CHKERRQ(ierr);
  ierr = MatShellSetOperation(user->Js,MATOP_MULT,(void(*)(void))StateMatMult); CHKERRQ(ierr);
  ierr = MatShellSetOperation(user->Js,MATOP_DUPLICATE,(void(*)(void))StateMatDuplicate); CHKERRQ(ierr);
  ierr = MatShellSetOperation(user->Js,MATOP_MULT_TRANSPOSE,(void(*)(void))StateMatMultTranspose); CHKERRQ(ierr);
  ierr = MatShellSetOperation(user->Js,MATOP_GET_DIAGONAL,(void(*)(void))StateMatGetDiagonal); CHKERRQ(ierr);

  /*PetscInt np,ip,processors_per_block;
  ierr = MatGetOwnershipRange(user->Js,&lo,&hi); CHKERRQ(ierr);
  np = user->m*user->nt / (hi-lo);
  ip = lo * np / (user->m*user->nt); 
  printf("%i ",ip); printf("%i\n",np);*/

  /* Diagonal blocks of user->Js */
  ierr = MatCreateShell(PETSC_COMM_WORLD,PETSC_DETERMINE,PETSC_DETERMINE,user->m,user->m,user,&user->JsBlock); CHKERRQ(ierr);
  ierr = MatShellSetOperation(user->JsBlock,MATOP_MULT,(void(*)(void))StateMatBlockMult); CHKERRQ(ierr);
  /* Blocks are symmetric */
  ierr = MatShellSetOperation(user->JsBlock,MATOP_MULT_TRANSPOSE,(void(*)(void))StateMatBlockMult); CHKERRQ(ierr);

  /* Create a matrix-free shell user->JsBlockPrec for computing (U+D)\D*(L+D)\x, where JsBlock = L+D+U,
     D is diagonal, L is strictly lower triangular, and U is strictly upper triangular.
     This is an SSOR preconditioner for user->JsBlock. */
  ierr = MatCreateShell(PETSC_COMM_WORLD,PETSC_DETERMINE,PETSC_DETERMINE,user->m,user->m,user,&user->JsBlockPrec); CHKERRQ(ierr);
  ierr = MatShellSetOperation(user->JsBlockPrec,MATOP_MULT,(void(*)(void))StateMatBlockPrecMult); CHKERRQ(ierr); 
  /* JsBlockPrec is symmetric */
  ierr = MatShellSetOperation(user->JsBlockPrec,MATOP_MULT_TRANSPOSE,(void(*)(void))StateMatBlockPrecMult); CHKERRQ(ierr);
  ierr = MatSetOption(user->JsBlockPrec,MAT_SYMMETRY_ETERNAL,PETSC_TRUE); CHKERRQ(ierr);
  
  /* Create a matrix-free shell user->Jd for computing B*x */
  ierr = MatCreateShell(PETSC_COMM_WORLD,PETSC_DETERMINE,PETSC_DETERMINE,user->m*user->nt,user->m,user,&user->Jd); CHKERRQ(ierr);
  ierr = MatShellSetOperation(user->Jd,MATOP_MULT,(void(*)(void))DesignMatMult); CHKERRQ(ierr);
  ierr = MatShellSetOperation(user->Jd,MATOP_MULT_TRANSPOSE,(void(*)(void))DesignMatMultTranspose); CHKERRQ(ierr);

  /* User-defined routines for computing user->Js\x and user->Js^T\x*/
  ierr = MatCreateShell(PETSC_COMM_WORLD,PETSC_DETERMINE,PETSC_DETERMINE,user->m*user->nt,user->m*user->nt,user,&user->JsInv); CHKERRQ(ierr);
  ierr = MatShellSetOperation(user->JsInv,MATOP_MULT,(void(*)(void))StateMatInvMult); CHKERRQ(ierr);
  ierr = MatShellSetOperation(user->JsInv,MATOP_MULT_TRANSPOSE,(void(*)(void))StateMatInvTransposeMult); CHKERRQ(ierr);

  /* Solver options and tolerances */
  ierr = KSPCreate(PETSC_COMM_WORLD,&user->solver); CHKERRQ(ierr);
  ierr = KSPSetType(user->solver,KSPCG); CHKERRQ(ierr);
  ierr = KSPSetOperators(user->solver,user->JsBlock,user->JsBlockPrec,DIFFERENT_NONZERO_PATTERN); CHKERRQ(ierr);
  ierr = KSPSetInitialGuessNonzero(user->solver,PETSC_FALSE); CHKERRQ(ierr);
  ierr = KSPSetTolerances(user->solver,1e-4,1e-8,1e3,500); CHKERRQ(ierr);
  //ierr = KSPSetTolerances(user->solver,1e-10,1e-20,1e3,500); CHKERRQ(ierr);
  ierr = KSPGetPC(user->solver,&user->prec); CHKERRQ(ierr);
  ierr = PCSetType(user->prec,PCSHELL); CHKERRQ(ierr);

  ierr = PCShellSetApply(user->prec,StateMatBlockPrecMult); CHKERRQ(ierr);
  ierr = PCShellSetApplyTranspose(user->prec,StateMatBlockPrecMult); CHKERRQ(ierr);
  ierr = PCShellSetContext(user->prec,user); CHKERRQ(ierr);

  ierr = KSPSetNormType(user->solver,KSP_NORM_UNPRECONDITIONED); CHKERRQ(ierr);
  ierr = KSPSetFromOptions(user->solver); CHKERRQ(ierr);

  /* Create scatter from y to y_1,y_2,...,y_nt */
  ierr = PetscMalloc(user->nt*user->m*sizeof(PetscInt),&user->yi_scatter);
  ierr = VecCreate(PETSC_COMM_WORLD,&yi); CHKERRQ(ierr);
  ierr = VecSetSizes(yi,PETSC_DECIDE,user->mx*user->mx*user->mx); CHKERRQ(ierr);
  ierr = VecSetFromOptions(yi); CHKERRQ(ierr);
  ierr = VecDuplicateVecs(yi,user->nt,&user->yi); CHKERRQ(ierr);
  ierr = VecDuplicateVecs(yi,user->nt,&user->yiwork); CHKERRQ(ierr);

  ierr = VecGetOwnershipRange(user->y,&lo2,&hi2); CHKERRQ(ierr);
  istart = 0;
  for (i=0; i<user->nt; i++){
    ierr = VecGetOwnershipRange(user->yi[i],&lo,&hi); CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF,hi-lo,lo,1,&is_to_yi); CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF,hi-lo,lo2+istart,1,&is_from_y); CHKERRQ(ierr);
    ierr = VecScatterCreate(user->y,is_from_y,user->yi[i],is_to_yi,&user->yi_scatter[i]); CHKERRQ(ierr);
    istart = istart + hi-lo;
  }
  ierr = ISDestroy(&is_to_yi); CHKERRQ(ierr);
  ierr = ISDestroy(&is_from_y); CHKERRQ(ierr);
  ierr = VecDestroy(&yi); CHKERRQ(ierr);

  /* Create scatter from d to d_1,d_2,...,d_nt */
  ierr = PetscMalloc(user->nt*user->ndata*sizeof(PetscInt),&user->di_scatter);
  ierr = VecCreate(PETSC_COMM_WORLD,&di); CHKERRQ(ierr);
  ierr = VecSetSizes(di,PETSC_DECIDE,user->ndata); CHKERRQ(ierr);
  ierr = VecSetFromOptions(di); CHKERRQ(ierr);
  ierr = VecDuplicateVecs(di,user->nt,&user->di); CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(user->d,&lo2,&hi2); CHKERRQ(ierr);
  istart = 0;
  for (i=0; i<user->nt; i++){
    ierr = VecGetOwnershipRange(user->di[i],&lo,&hi); CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF,hi-lo,lo,1,&is_to_di); CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF,hi-lo,lo2+istart,1,&is_from_d); CHKERRQ(ierr);
    ierr = VecScatterCreate(user->d,is_from_d,user->di[i],is_to_di,&user->di_scatter[i]); CHKERRQ(ierr);
    istart = istart + hi-lo;
  }
  ierr = ISDestroy(&is_to_di); CHKERRQ(ierr);
  ierr = ISDestroy(&is_from_d); CHKERRQ(ierr);
  ierr = VecDestroy(&di); CHKERRQ(ierr);

  /* Assemble RHS of forward problem */
  ierr = VecCopy(bc,user->yiwork[0]);
  for (i=1; i<user->nt; i++){
    ierr = VecSet(user->yiwork[i],0.0); CHKERRQ(ierr);
  }
  ierr = Gather_yi(user->q,user->yiwork,user->yi_scatter,user->nt); CHKERRQ(ierr);
  ierr = VecDestroy(&bc); CHKERRQ(ierr);

  /* Compute true state function yt given ut */
  ierr = VecCreate(PETSC_COMM_WORLD,&user->ytrue); CHKERRQ(ierr);
  ierr = VecSetSizes(user->ytrue,PETSC_DECIDE,n*user->nt); CHKERRQ(ierr);
  ierr = VecSetFromOptions(user->ytrue); CHKERRQ(ierr);
  //ierr = StateMatInvMult(user->Js,user->q,user->utrue,user->ytrue); CHKERRQ(ierr);

  /* First compute Av_u = Av*exp(-u) */
  ierr = VecSet(user->uwork,0);
  ierr = VecAXPY(user->uwork,-1.0,user->utrue); // Note: user->utrue
  ierr = VecExp(user->uwork); CHKERRQ(ierr);
  ierr = MatMult(user->Av,user->uwork,user->Av_u); CHKERRQ(ierr);

  /* Next form DSG = Div*S*Grad */
  ierr = MatCopy(user->Div,user->Divwork,SAME_NONZERO_PATTERN); CHKERRQ(ierr);
  ierr = MatDiagonalScale(user->Divwork,PETSC_NULL,user->Av_u); CHKERRQ(ierr);
  if (user->dsg_formed) {
    ierr = MatMatMult(user->Divwork,user->Grad,MAT_REUSE_MATRIX,1,&user->DSG); CHKERRQ(ierr);
  } else {
    ierr = MatMatMult(user->Div,user->Grad,MAT_INITIAL_MATRIX,1,&user->DSG); CHKERRQ(ierr);
    user->dsg_formed = PETSC_TRUE;
  } 
  /* B = speye(nx^3) + ht*DSG; */
  ierr = MatScale(user->DSG,user->ht); CHKERRQ(ierr);
  ierr = MatShift(user->DSG,1.0); CHKERRQ(ierr);

  /* Now solve for ytrue */
  ierr = StateMatInvMult(user->Js,user->q,user->ytrue); CHKERRQ(ierr);

  /* Initial guess y0 for state given u0 */
  ierr = VecCreate(PETSC_COMM_WORLD,&user->y); CHKERRQ(ierr);
  ierr = VecSetSizes(user->y,PETSC_DECIDE,n*user->nt); CHKERRQ(ierr);
  ierr = VecSetFromOptions(user->y); CHKERRQ(ierr);
  //ierr = StateMatInvMult(user->Js,user->q,user->u,user->y); CHKERRQ(ierr);

  /* First compute Av_u = Av*exp(-u) */
  ierr = VecSet(user->uwork,0);
  ierr = VecAXPY(user->uwork,-1.0,user->u); // Note: user->u
  ierr = VecExp(user->uwork); CHKERRQ(ierr);
  ierr = MatMult(user->Av,user->uwork,user->Av_u); CHKERRQ(ierr);

  /* Next form DSG = Div*S*Grad */
  ierr = MatCopy(user->Div,user->Divwork,SAME_NONZERO_PATTERN); CHKERRQ(ierr);
  ierr = MatDiagonalScale(user->Divwork,PETSC_NULL,user->Av_u); CHKERRQ(ierr);
  if (user->dsg_formed) {
    ierr = MatMatMult(user->Divwork,user->Grad,MAT_REUSE_MATRIX,1,&user->DSG); CHKERRQ(ierr);
  } else {
    ierr = MatMatMult(user->Div,user->Grad,MAT_INITIAL_MATRIX,1,&user->DSG); CHKERRQ(ierr);
    user->dsg_formed = PETSC_TRUE;
  }
  /* B = speye(nx^3) + ht*DSG; */
  ierr = MatScale(user->DSG,user->ht); CHKERRQ(ierr);
  ierr = MatShift(user->DSG,1.0); CHKERRQ(ierr);

  /* Now solve for y */
  ierr = StateMatInvMult(user->Js,user->q,user->y); CHKERRQ(ierr);
  
  /* Construct projection matrix Q, a block diagonal matrix consisting of nt copies of Qblock along the diagonal */
  ierr = MatCreate(PETSC_COMM_WORLD,&user->Qblock); CHKERRQ(ierr);
  ierr = MatSetSizes(user->Qblock,PETSC_DECIDE,PETSC_DECIDE,user->ndata,n); CHKERRQ(ierr);
  ierr = MatSetFromOptions(user->Qblock); CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(user->Qblock,8,PETSC_NULL,8,PETSC_NULL); CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(user->Qblock,8,PETSC_NULL); CHKERRQ(ierr);

  /* Data locations */
  PetscScalar xr[64] = {0.4970,     0.8498,     0.7814,     0.6268,     0.7782,     0.6402,     0.3617,     0.3160,     
			0.3610,     0.5298,     0.6987,     0.3331,     0.7962,     0.5596,     0.3866,     0.6774,     
			0.5407,     0.4518,     0.6702,     0.6061,     0.7580,     0.8997,     0.5198,     0.8326,     
			0.2138,     0.9198,     0.3000,     0.2833,     0.8288,     0.7076,     0.1820,     0.0728,     
			0.8447,     0.2367,     0.3239,     0.6413,     0.3114,     0.4731,     0.1192,     0.9273,     
			0.5724,     0.4331,     0.5136,     0.3547,     0.4413,     0.2602,     0.5698,     0.7278,     
			0.5261,     0.6230,     0.2454,     0.3948,     0.7479,     0.6582,     0.4660,     0.5594,     
			0.7574,     0.1143,     0.5900,     0.1065,     0.4260,     0.3294,     0.8276,     0.0756};
  
  PetscScalar yr[64] = {0.7345,     0.9120,     0.9288,     0.7528,     0.4463,     0.4985,     0.2497,     0.6256,     
			0.3425,     0.9026,     0.6983,     0.4230,     0.7140,     0.2970,     0.4474,     0.8792,     
			0.6604,     0.2485,     0.7968,     0.6127,     0.1796,     0.2437,     0.5938,     0.6137,     
			0.3867,     0.5658,     0.4575,     0.1009,     0.0863,     0.3361,     0.0738,     0.3985,     
			0.6602,     0.1437,     0.0934,     0.5983,     0.5950,     0.0763,     0.0768,     0.2288,     
			0.5761,     0.1129,     0.3841,     0.6150,     0.6904,     0.6686,     0.1361,     0.4601,     
			0.4491,     0.3716,     0.1969,     0.6537,     0.6743,     0.6991,     0.4811,     0.5480,     
			0.1684,     0.4569,     0.6889,     0.8437,     0.3015,     0.2854,     0.8199,     0.2658};
  
  PetscScalar zr[64] = {0.7668,     0.8573,     0.2654,     0.2719,     0.1060,     0.1311,     0.6232,     0.2295,     
			0.8009,     0.2147,     0.2119,     0.9325,     0.4473,     0.3600,     0.3374,     0.3819,     
			0.4066,     0.5801,     0.1673,     0.0959,     0.4638,     0.8236,     0.8800,     0.2939,     
			0.2028,     0.8262,     0.2706,     0.6276,     0.9085,     0.6443,     0.8241,     0.0712,     
			0.1824,     0.7789,     0.4389,     0.8415,     0.7055,     0.6639,     0.3653,     0.2078,     
			0.1987,     0.2297,     0.4321,     0.8115,     0.4915,     0.7764,     0.4657,     0.4627,     
			0.4569,     0.4232,     0.8514,     0.0674,     0.3227,     0.1055,     0.6690,     0.6313,     
			0.9226,     0.5461,     0.4126,     0.2364,     0.6096,     0.7042,     0.3914,     0.0711};
 
  for (i=0; i<user->mx; i++){
    x[i] = h*(i+0.5);
    y[i] = h*(i+0.5);
    z[i] = h*(i+0.5);
  }
  
  ierr = MatGetOwnershipRange(user->Qblock,&istart,&iend);

  nx = user->mx; ny = user->mx; nz = user->mx;
  for (i=istart; i<iend; i++){
    
    xri = xr[i];
    im = 0;
    xim = x[im];
    while (xri>xim && im<nx){
      im = im+1;
      xim = x[im];
    }
    indx1 = im-1;
    indx2 = im;
    dx1 = xri - x[indx1];
    dx2 = x[indx2] - xri;

    yri = yr[i];
    im = 0;
    yim = y[im];
    while (yri>yim && im<ny){
      im = im+1;
      yim = y[im];
    }
    indy1 = im-1;
    indy2 = im;
    dy1 = yri - y[indy1];
    dy2 = y[indy2] - yri;
    
    zri = zr[i];
    im = 0;
    zim = z[im];
    while (zri>zim && im<nz){
      im = im+1;
      zim = z[im];
    }
    indz1 = im-1;
    indz2 = im;
    dz1 = zri - z[indz1];
    dz2 = z[indz2] - zri;

    Dx = x[indx2] - x[indx1];
    Dy = y[indy2] - y[indy1];
    Dz = z[indz2] - z[indz1];

    j = indx1 + indy1*nx + indz1*nx*ny;
    v = (1-dx1/Dx)*(1-dy1/Dy)*(1-dz1/Dz);
    ierr = MatSetValues(user->Qblock,1,&i,1,&j,&v,INSERT_VALUES); CHKERRQ(ierr);

    j = indx1 + indy1*nx + indz2*nx*ny;
    v = (1-dx1/Dx)*(1-dy1/Dy)*(1-dz2/Dz);
    ierr = MatSetValues(user->Qblock,1,&i,1,&j,&v,INSERT_VALUES); CHKERRQ(ierr);

    j = indx1 + indy2*nx + indz1*nx*ny;
    v = (1-dx1/Dx)*(1-dy2/Dy)*(1-dz1/Dz);
    ierr = MatSetValues(user->Qblock,1,&i,1,&j,&v,INSERT_VALUES); CHKERRQ(ierr);

    j = indx1 + indy2*nx + indz2*nx*ny;
    v = (1-dx1/Dx)*(1-dy2/Dy)*(1-dz2/Dz);
    ierr = MatSetValues(user->Qblock,1,&i,1,&j,&v,INSERT_VALUES); CHKERRQ(ierr);

    j = indx2 + indy1*nx + indz1*nx*ny;
    v = (1-dx2/Dx)*(1-dy1/Dy)*(1-dz1/Dz);
    ierr = MatSetValues(user->Qblock,1,&i,1,&j,&v,INSERT_VALUES); CHKERRQ(ierr);

    j = indx2 + indy1*nx + indz2*nx*ny;
    v = (1-dx2/Dx)*(1-dy1/Dy)*(1-dz2/Dz);
    ierr = MatSetValues(user->Qblock,1,&i,1,&j,&v,INSERT_VALUES); CHKERRQ(ierr);

    j = indx2 + indy2*nx + indz1*nx*ny;
    v = (1-dx2/Dx)*(1-dy2/Dy)*(1-dz1/Dz);
    ierr = MatSetValues(user->Qblock,1,&i,1,&j,&v,INSERT_VALUES); CHKERRQ(ierr);

    j = indx2 + indy2*nx + indz2*nx*ny;
    v = (1-dx2/Dx)*(1-dy2/Dy)*(1-dz2/Dz);
    ierr = MatSetValues(user->Qblock,1,&i,1,&j,&v,INSERT_VALUES); CHKERRQ(ierr);

  }

  ierr = MatAssemblyBegin(user->Qblock,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(user->Qblock,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  /*Mat diagQ[user->nt*user->nt];
  //IS isQ;
  for (i=0; i<user->nt; i++){
    for (j=0; j<user->nt; j++){
      if (i==j){
	ierr = MatDuplicate(user->Qblock,MAT_COPY_VALUES,&diagQ[i+j*user->nt]); CHKERRQ(ierr);
      }
      else {
	diagQ[i+j*user->nt] = PETSC_NULL;
      }
    }
  }
  //ierr = ISCreateStride(PETSC_COMM_SELF,user->nt,0,1,&isQ);
  ierr = MatCreateNest(PETSC_COMM_WORLD,user->nt,PETSC_NULL,user->nt,PETSC_NULL,diagQ,&user->Q); CHKERRQ(ierr); 
  ierr = MatAssemblyBegin(user->Q,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(user->Q,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  ierr = MatDestroy(diagQ); CHKERRQ(ierr);*/

  /* Generate transpose of Q and L */
  //ierr = MatTranspose(user->Q,MAT_INITIAL_MATRIX,&user->QT); CHKERRQ(ierr);

  ierr = MatTranspose(user->Qblock,MAT_INITIAL_MATRIX,&user->QblockT); CHKERRQ(ierr);
 
  /*Mat diagQT[user->nt*user->nt];
  //IS isQ;
  for (i=0; i<user->nt; i++){
    for (j=0; j<user->nt; j++){
      if (i==j){
	ierr = MatDuplicate(user->QblockT,MAT_COPY_VALUES,&diagQT[i+j*user->nt]); CHKERRQ(ierr);
      }
      else {
	diagQT[i+j*user->nt] = PETSC_NULL;
      }
    }
  }
  //ierr = ISCreateStride(PETSC_COMM_SELF,user->nt,0,1,&isQ);
  ierr = MatCreateNest(PETSC_COMM_WORLD,user->nt,PETSC_NULL,user->nt,PETSC_NULL,diagQT,&user->QT); CHKERRQ(ierr); 
  ierr = MatAssemblyBegin(user->QT,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(user->QT,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  ierr = MatDestroy(diagQT); CHKERRQ(ierr);*/

  ierr = MatTranspose(user->L,MAT_INITIAL_MATRIX,&user->LT); CHKERRQ(ierr);

  /* Add noise to the measurement data */
  ierr = VecSet(user->ywork,1.0); CHKERRQ(ierr);
  ierr = VecAYPX(user->ywork,user->noise,user->ytrue); CHKERRQ(ierr);
  //ierr = MatMult(user->Q,user->ywork,user->d); CHKERRQ(ierr);
  ierr = Scatter_yi(user->ywork,user->yiwork,user->yi_scatter,user->nt); CHKERRQ(ierr);
  for (i=0; i<user->nt; i++){
    ierr = MatMult(user->Qblock,user->yiwork[i],user->di[i]);
  }
  ierr = Gather_yi(user->d,user->di,user->di_scatter,user->nt); CHKERRQ(ierr);

  //ierr = MatCreateSeqAIJ(PETSC_COMM_WORLD,user->m,user->n-user->m,7,PETSC_NULL,&user->Jd); CHKERRQ(ierr);
  //ierr = MatCreateSeqAIJ(PETSC_COMM_WORLD,user->m,user->m,0,PETSC_NULL,&user->Hsd); CHKERRQ(ierr);

  /*ierr = MatCreate(PETSC_COMM_WORLD,&user->Jd); CHKERRQ(ierr);
  ierr = MatSetSizes(user->Jd,PETSC_DECIDE,PETSC_DECIDE,user->m,user->m); CHKERRQ(ierr);
  ierr = MatSetFromOptions(user->Jd); CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(user->Jd,7,PETSC_NULL,7,PETSC_NULL); CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(user->Jd,7,PETSC_NULL); CHKERRQ(ierr);*/

  /*ierr = MatCreate(PETSC_COMM_WORLD,&user->Hsd); CHKERRQ(ierr);
  ierr = MatSetSizes(user->Hsd,PETSC_DECIDE,PETSC_DECIDE,user->m,user->m); CHKERRQ(ierr);
  ierr = MatSetFromOptions(user->Hsd); CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(user->Hsd,0,PETSC_NULL,0,PETSC_NULL); CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(user->Hsd,0,PETSC_NULL); CHKERRQ(ierr);

  ierr = MatZeroEntries(user->Hsd); CHKERRQ(ierr);*/

  //ierr = MatMatMultTranspose(user->Q,user->Q,MAT_INITIAL_MATRIX,2.0,&user->Hs); CHKERRQ(ierr);
  //ierr = MatMatMultTranspose(user->L,user->L,MAT_INITIAL_MATRIX,2.0,&user->Hd); CHKERRQ(ierr);
  //ierr = MatScale(user->Hd,user->alpha); CHKERRQ(ierr);


  /* Assemble the matrix */
/*  ierr = MatAssemblyBegin(user->Js,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(user->Js,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyBegin(user->Jd,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(user->Jd,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);*/

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ParabolicDestroy"
PetscErrorCode ParabolicDestroy(AppCtx *user)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  //ierr = MatDestroy(&user->Q); CHKERRQ(ierr);
  //ierr = MatDestroy(&user->QT); CHKERRQ(ierr);
  ierr = MatDestroy(&user->Qblock); CHKERRQ(ierr);
  ierr = MatDestroy(&user->QblockT); CHKERRQ(ierr);
  ierr = MatDestroy(&user->Div); CHKERRQ(ierr);
  ierr = MatDestroy(&user->Divwork); CHKERRQ(ierr);
  ierr = MatDestroy(&user->Grad); CHKERRQ(ierr);
  ierr = MatDestroy(&user->Av); CHKERRQ(ierr);
  ierr = MatDestroy(&user->Avwork); CHKERRQ(ierr);
  ierr = MatDestroy(&user->AvT); CHKERRQ(ierr);
  ierr = MatDestroy(&user->DSG); CHKERRQ(ierr);
  ierr = MatDestroy(&user->L); CHKERRQ(ierr);
  ierr = MatDestroy(&user->LT); CHKERRQ(ierr);
  ierr = KSPDestroy(&user->solver); CHKERRQ(ierr);
  ierr = MatDestroy(&user->Js); CHKERRQ(ierr);
  ierr = MatDestroy(&user->Jd); CHKERRQ(ierr);
  //ierr = MatDestroy(&user->JsBlockPrec); CHKERRQ(ierr);
  ierr = MatDestroy(&user->JsInv); CHKERRQ(ierr);
  ierr = MatDestroy(&user->JsBlock); CHKERRQ(ierr);
  //ierr = MatDestroy(&user->Hs); CHKERRQ(ierr);
  //ierr = MatDestroy(&user->Hd); CHKERRQ(ierr);
  //ierr = MatDestroy(&user->Hsd); CHKERRQ(ierr);
  ierr = VecDestroy(&user->u); CHKERRQ(ierr);
  ierr = VecDestroy(&user->uwork); CHKERRQ(ierr);
  ierr = VecDestroy(&user->usave); CHKERRQ(ierr);
  ierr = VecDestroy(&user->utrue); CHKERRQ(ierr);
  ierr = VecDestroy(&user->y); CHKERRQ(ierr);
  ierr = VecDestroy(&user->ywork); CHKERRQ(ierr);
  ierr = VecDestroy(&user->ysave); CHKERRQ(ierr);
  ierr = VecDestroy(&user->ytrue); CHKERRQ(ierr);
  ierr = VecDestroyVecs(user->nt,&user->yi); CHKERRQ(ierr);
  ierr = VecDestroyVecs(user->nt,&user->yiwork); CHKERRQ(ierr);
  ierr = VecDestroyVecs(user->nt,&user->di); CHKERRQ(ierr);
  ierr = PetscFree(user->yi); CHKERRQ(ierr);
  ierr = PetscFree(user->yiwork); CHKERRQ(ierr);
  ierr = PetscFree(user->di); CHKERRQ(ierr);
  ierr = VecDestroy(&user->c); CHKERRQ(ierr);
  ierr = VecDestroy(&user->cwork); CHKERRQ(ierr);
  ierr = VecDestroy(&user->ur); CHKERRQ(ierr);
  ierr = VecDestroy(&user->q); CHKERRQ(ierr);
  ierr = VecDestroy(&user->d); CHKERRQ(ierr);
  ierr = VecDestroy(&user->dwork); CHKERRQ(ierr);
  ierr = VecDestroy(&user->lwork); CHKERRQ(ierr);
  ierr = VecDestroy(&user->S); CHKERRQ(ierr);
  ierr = VecDestroy(&user->Swork); CHKERRQ(ierr);
  ierr = VecDestroy(&user->Av_u); CHKERRQ(ierr);
  ierr = VecDestroy(&user->Twork); CHKERRQ(ierr);
  ierr = VecDestroy(&user->Rwork); CHKERRQ(ierr);
  ierr = VecDestroy(&user->js_diag); CHKERRQ(ierr);
  ierr = ISDestroy(&user->s_is); CHKERRQ(ierr);
  ierr = ISDestroy(&user->d_is); CHKERRQ(ierr);
  ierr = VecScatterDestroy(&user->state_scatter); CHKERRQ(ierr);
  ierr = VecScatterDestroy(&user->design_scatter); CHKERRQ(ierr);
  ierr = VecScatterDestroy(user->yi_scatter); CHKERRQ(ierr);
  ierr = VecScatterDestroy(user->di_scatter); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ParabolicMonitor"
PetscErrorCode ParabolicMonitor(TaoSolver tao, void *ptr)
{
  PetscErrorCode ierr;
  Vec X;
  PetscReal unorm,ynorm;
  AppCtx *user = (AppCtx*)ptr;
  PetscFunctionBegin;
  ierr = TaoSolverGetSolutionVector(tao,&X); CHKERRQ(ierr);
  ierr = Scatter(X,user->ywork,user->state_scatter,user->uwork,user->design_scatter); CHKERRQ(ierr);
  ierr = VecAXPY(user->ywork,-1.0,user->ytrue); CHKERRQ(ierr);
  ierr = VecAXPY(user->uwork,-1.0,user->utrue); CHKERRQ(ierr);
  ierr = VecNorm(user->uwork,NORM_2,&unorm); CHKERRQ(ierr);
  ierr = VecNorm(user->ywork,NORM_2,&ynorm); CHKERRQ(ierr);
  ierr = PetscPrintf(MPI_COMM_WORLD, "||u-ut||=%7g ||y-yt||=%7g\n",unorm,ynorm); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
