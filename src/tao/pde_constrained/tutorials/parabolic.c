#include <petsc/private/taoimpl.h>

/*T
   Concepts: TAO^Solving a system of nonlinear equations, nonlinear least squares
   Routines: TaoCreate();
   Routines: TaoSetType();
   Routines: TaoSetSolution();
   Routines: TaoSetObjective();
   Routines: TaoSetGradient();
   Routines: TaoSetConstraintsRoutine();
   Routines: TaoSetJacobianStateRoutine();
   Routines: TaoSetJacobianDesignRoutine();
   Routines: TaoSetStateDesignIS();
   Routines: TaoSetFromOptions();
   Routines: TaoSolve();
   Routines: TaoDestroy();
   Processors: n
T*/

typedef struct {
  PetscInt n; /*  Number of variables */
  PetscInt m; /*  Number of constraints per time step */
  PetscInt mx; /*  grid points in each direction */
  PetscInt nt; /*  Number of time steps; as of now, must be divisible by 8 */
  PetscInt ndata; /*  Number of data points per sample */
  PetscInt ns; /*  Number of samples */
  PetscInt *sample_times; /*  Times of samples */
  IS       s_is;
  IS       d_is;

  VecScatter state_scatter;
  VecScatter design_scatter;
  VecScatter *yi_scatter;
  VecScatter *di_scatter;

  Mat       Js,Jd,JsBlockPrec,JsInv,JsBlock;
  PetscBool jformed,dsg_formed;

  PetscReal alpha; /*  Regularization parameter */
  PetscReal beta; /*  Weight attributed to ||u||^2 in regularization functional */
  PetscReal noise; /*  Amount of noise to add to data */
  PetscReal ht; /*  Time step */

  Mat Qblock,QblockT;
  Mat L,LT;
  Mat Div,Divwork;
  Mat Grad;
  Mat Av,Avwork,AvT;
  Mat DSG;
  Vec q;
  Vec ur; /*  reference */

  Vec d;
  Vec dwork;
  Vec *di;

  Vec y; /*  state variables */
  Vec ywork;

  Vec ytrue;
  Vec *yi,*yiwork;

  Vec u; /*  design variables */
  Vec uwork;

  Vec utrue;
  Vec js_diag;
  Vec c; /*  constraint vector */
  Vec cwork;

  Vec lwork;
  Vec S;
  Vec Rwork,Swork,Twork;
  Vec Av_u;

  KSP solver;
  PC prec;

  PetscInt ksp_its;
  PetscInt ksp_its_initial;
} AppCtx;

PetscErrorCode FormFunction(Tao, Vec, PetscReal*, void*);
PetscErrorCode FormGradient(Tao, Vec, Vec, void*);
PetscErrorCode FormFunctionGradient(Tao, Vec, PetscReal*, Vec, void*);
PetscErrorCode FormJacobianState(Tao, Vec, Mat, Mat, Mat, void*);
PetscErrorCode FormJacobianDesign(Tao, Vec, Mat, void*);
PetscErrorCode FormConstraints(Tao, Vec, Vec, void*);
PetscErrorCode FormHessian(Tao, Vec, Mat, Mat, void*);
PetscErrorCode Gather(Vec x, Vec state, VecScatter s_scat, Vec design, VecScatter d_scat);
PetscErrorCode Scatter(Vec x, Vec state, VecScatter s_scat, Vec design, VecScatter d_scat);
PetscErrorCode ParabolicInitialize(AppCtx *user);
PetscErrorCode ParabolicDestroy(AppCtx *user);
PetscErrorCode ParabolicMonitor(Tao, void*);

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

PetscErrorCode Gather_i(Vec,Vec*,VecScatter*,PetscInt);
PetscErrorCode Scatter_i(Vec,Vec*,VecScatter*,PetscInt);

static  char help[]="";

int main(int argc, char **argv)
{
  PetscErrorCode     ierr;
  Vec                x,x0;
  Tao                tao;
  AppCtx             user;
  IS                 is_allstate,is_alldesign;
  PetscInt           lo,hi,hi2,lo2,ksp_old;
  PetscInt           ntests = 1;
  PetscInt           i;
#if defined(PETSC_USE_LOG)
  PetscLogStage      stages[1];
#endif

  ierr = PetscInitialize(&argc, &argv, (char*)0,help);if (ierr) return ierr;
  user.mx = 8;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"parabolic example",NULL);CHKERRQ(ierr);
  CHKERRQ(PetscOptionsInt("-mx","Number of grid points in each direction","",user.mx,&user.mx,NULL));
  user.nt = 8;
  CHKERRQ(PetscOptionsInt("-nt","Number of time steps","",user.nt,&user.nt,NULL));
  user.ndata = 64;
  CHKERRQ(PetscOptionsInt("-ndata","Numbers of data points per sample","",user.ndata,&user.ndata,NULL));
  user.ns = 8;
  CHKERRQ(PetscOptionsInt("-ns","Number of samples","",user.ns,&user.ns,NULL));
  user.alpha = 1.0;
  CHKERRQ(PetscOptionsReal("-alpha","Regularization parameter","",user.alpha,&user.alpha,NULL));
  user.beta = 0.01;
  CHKERRQ(PetscOptionsReal("-beta","Weight attributed to ||u||^2 in regularization functional","",user.beta,&user.beta,NULL));
  user.noise = 0.01;
  CHKERRQ(PetscOptionsReal("-noise","Amount of noise to add to data","",user.noise,&user.noise,NULL));
  CHKERRQ(PetscOptionsInt("-ntests","Number of times to repeat TaoSolve","",ntests,&ntests,NULL));
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  user.m = user.mx*user.mx*user.mx; /*  number of constraints per time step */
  user.n = user.m*(user.nt+1); /*  number of variables */
  user.ht = (PetscReal)1/user.nt;

  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&user.u));
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&user.y));
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&user.c));
  CHKERRQ(VecSetSizes(user.u,PETSC_DECIDE,user.n-user.m*user.nt));
  CHKERRQ(VecSetSizes(user.y,PETSC_DECIDE,user.m*user.nt));
  CHKERRQ(VecSetSizes(user.c,PETSC_DECIDE,user.m*user.nt));
  CHKERRQ(VecSetFromOptions(user.u));
  CHKERRQ(VecSetFromOptions(user.y));
  CHKERRQ(VecSetFromOptions(user.c));

  /* Create scatters for reduced spaces.
     If the state vector y and design vector u are partitioned as
     [y_1; y_2; ...; y_np] and [u_1; u_2; ...; u_np] (with np = # of processors),
     then the solution vector x is organized as
     [y_1; u_1; y_2; u_2; ...; y_np; u_np].
     The index sets user.s_is and user.d_is correspond to the indices of the
     state and design variables owned by the current processor.
  */
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&x));

  CHKERRQ(VecGetOwnershipRange(user.y,&lo,&hi));
  CHKERRQ(VecGetOwnershipRange(user.u,&lo2,&hi2));

  CHKERRQ(ISCreateStride(PETSC_COMM_SELF,hi-lo,lo,1,&is_allstate));
  CHKERRQ(ISCreateStride(PETSC_COMM_SELF,hi-lo,lo+lo2,1,&user.s_is));

  CHKERRQ(ISCreateStride(PETSC_COMM_SELF,hi2-lo2,lo2,1,&is_alldesign));
  CHKERRQ(ISCreateStride(PETSC_COMM_SELF,hi2-lo2,hi+lo2,1,&user.d_is));

  CHKERRQ(VecSetSizes(x,hi-lo+hi2-lo2,user.n));
  CHKERRQ(VecSetFromOptions(x));

  CHKERRQ(VecScatterCreate(x,user.s_is,user.y,is_allstate,&user.state_scatter));
  CHKERRQ(VecScatterCreate(x,user.d_is,user.u,is_alldesign,&user.design_scatter));
  CHKERRQ(ISDestroy(&is_alldesign));
  CHKERRQ(ISDestroy(&is_allstate));

  /* Create TAO solver and set desired solution method */
  CHKERRQ(TaoCreate(PETSC_COMM_WORLD,&tao));
  CHKERRQ(TaoSetType(tao,TAOLCL));

  /* Set up initial vectors and matrices */
  CHKERRQ(ParabolicInitialize(&user));

  CHKERRQ(Gather(x,user.y,user.state_scatter,user.u,user.design_scatter));
  CHKERRQ(VecDuplicate(x,&x0));
  CHKERRQ(VecCopy(x,x0));

  /* Set solution vector with an initial guess */
  CHKERRQ(TaoSetSolution(tao,x));
  CHKERRQ(TaoSetObjective(tao, FormFunction, &user));
  CHKERRQ(TaoSetGradient(tao, NULL, FormGradient, &user));
  CHKERRQ(TaoSetConstraintsRoutine(tao, user.c, FormConstraints, &user));

  CHKERRQ(TaoSetJacobianStateRoutine(tao, user.Js, user.JsBlockPrec, user.JsInv, FormJacobianState, &user));
  CHKERRQ(TaoSetJacobianDesignRoutine(tao, user.Jd, FormJacobianDesign, &user));

  CHKERRQ(TaoSetFromOptions(tao));
  CHKERRQ(TaoSetStateDesignIS(tao,user.s_is,user.d_is));

 /* SOLVE THE APPLICATION */
  CHKERRQ(PetscLogStageRegister("Trials",&stages[0]));
  CHKERRQ(PetscLogStagePush(stages[0]));
  user.ksp_its_initial = user.ksp_its;
  ksp_old = user.ksp_its;
  for (i=0; i<ntests; i++) {
    CHKERRQ(TaoSolve(tao));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"KSP Iterations = %D\n",user.ksp_its-ksp_old));
    CHKERRQ(VecCopy(x0,x));
    CHKERRQ(TaoSetSolution(tao,x));
  }
  CHKERRQ(PetscLogStagePop());
  CHKERRQ(PetscBarrier((PetscObject)x));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"KSP iterations within initialization: "));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"%D\n",user.ksp_its_initial));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Total KSP iterations over %D trial(s): ",ntests));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"%D\n",user.ksp_its));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"KSP iterations per trial: "));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"%D\n",(user.ksp_its-user.ksp_its_initial)/ntests));

  CHKERRQ(TaoDestroy(&tao));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&x0));
  CHKERRQ(ParabolicDestroy(&user));

  ierr = PetscFinalize();
  return ierr;
}
/* ------------------------------------------------------------------- */
/*
   dwork = Qy - d
   lwork = L*(u-ur)
   f = 1/2 * (dwork.dork + alpha*lwork.lwork)
*/
PetscErrorCode FormFunction(Tao tao,Vec X,PetscReal *f,void *ptr)
{
  PetscReal      d1=0,d2=0;
  PetscInt       i,j;
  AppCtx         *user = (AppCtx*)ptr;

  PetscFunctionBegin;
  CHKERRQ(Scatter(X,user->y,user->state_scatter,user->u,user->design_scatter));
  CHKERRQ(Scatter_i(user->y,user->yi,user->yi_scatter,user->nt));
  for (j=0; j<user->ns; j++) {
    i = user->sample_times[j];
    CHKERRQ(MatMult(user->Qblock,user->yi[i],user->di[j]));
  }
  CHKERRQ(Gather_i(user->dwork,user->di,user->di_scatter,user->ns));
  CHKERRQ(VecAXPY(user->dwork,-1.0,user->d));
  CHKERRQ(VecDot(user->dwork,user->dwork,&d1));

  CHKERRQ(VecWAXPY(user->uwork,-1.0,user->ur,user->u));
  CHKERRQ(MatMult(user->L,user->uwork,user->lwork));
  CHKERRQ(VecDot(user->lwork,user->lwork,&d2));

  *f = 0.5 * (d1 + user->alpha*d2);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
/*
    state: g_s = Q' *(Qy - d)
    design: g_d = alpha*L'*L*(u-ur)
*/
PetscErrorCode FormGradient(Tao tao,Vec X,Vec G,void *ptr)
{
  PetscInt       i,j;
  AppCtx         *user = (AppCtx*)ptr;

  PetscFunctionBegin;
  CHKERRQ(Scatter(X,user->y,user->state_scatter,user->u,user->design_scatter));
  CHKERRQ(Scatter_i(user->y,user->yi,user->yi_scatter,user->nt));
  for (j=0; j<user->ns; j++) {
    i = user->sample_times[j];
    CHKERRQ(MatMult(user->Qblock,user->yi[i],user->di[j]));
  }
  CHKERRQ(Gather_i(user->dwork,user->di,user->di_scatter,user->ns));
  CHKERRQ(VecAXPY(user->dwork,-1.0,user->d));
  CHKERRQ(Scatter_i(user->dwork,user->di,user->di_scatter,user->ns));
  CHKERRQ(VecSet(user->ywork,0.0));
  CHKERRQ(Scatter_i(user->ywork,user->yiwork,user->yi_scatter,user->nt));
  for (j=0; j<user->ns; j++) {
    i = user->sample_times[j];
    CHKERRQ(MatMult(user->QblockT,user->di[j],user->yiwork[i]));
  }
  CHKERRQ(Gather_i(user->ywork,user->yiwork,user->yi_scatter,user->nt));

  CHKERRQ(VecWAXPY(user->uwork,-1.0,user->ur,user->u));
  CHKERRQ(MatMult(user->L,user->uwork,user->lwork));
  CHKERRQ(MatMult(user->LT,user->lwork,user->uwork));
  CHKERRQ(VecScale(user->uwork, user->alpha));
  CHKERRQ(Gather(G,user->ywork,user->state_scatter,user->uwork,user->design_scatter));
  PetscFunctionReturn(0);
}

PetscErrorCode FormFunctionGradient(Tao tao, Vec X, PetscReal *f, Vec G, void *ptr)
{
  PetscReal      d1,d2;
  PetscInt       i,j;
  AppCtx         *user = (AppCtx*)ptr;

  PetscFunctionBegin;
  CHKERRQ(Scatter(X,user->y,user->state_scatter,user->u,user->design_scatter));
  CHKERRQ(Scatter_i(user->y,user->yi,user->yi_scatter,user->nt));
  for (j=0; j<user->ns; j++) {
    i = user->sample_times[j];
    CHKERRQ(MatMult(user->Qblock,user->yi[i],user->di[j]));
  }
  CHKERRQ(Gather_i(user->dwork,user->di,user->di_scatter,user->ns));
  CHKERRQ(VecAXPY(user->dwork,-1.0,user->d));
  CHKERRQ(VecDot(user->dwork,user->dwork,&d1));
  CHKERRQ(Scatter_i(user->dwork,user->di,user->di_scatter,user->ns));
  CHKERRQ(VecSet(user->ywork,0.0));
  CHKERRQ(Scatter_i(user->ywork,user->yiwork,user->yi_scatter,user->nt));
  for (j=0; j<user->ns; j++) {
    i = user->sample_times[j];
    CHKERRQ(MatMult(user->QblockT,user->di[j],user->yiwork[i]));
  }
  CHKERRQ(Gather_i(user->ywork,user->yiwork,user->yi_scatter,user->nt));

  CHKERRQ(VecWAXPY(user->uwork,-1.0,user->ur,user->u));
  CHKERRQ(MatMult(user->L,user->uwork,user->lwork));
  CHKERRQ(VecDot(user->lwork,user->lwork,&d2));
  CHKERRQ(MatMult(user->LT,user->lwork,user->uwork));
  CHKERRQ(VecScale(user->uwork, user->alpha));
  *f = 0.5 * (d1 + user->alpha*d2);

  CHKERRQ(Gather(G,user->ywork,user->state_scatter,user->uwork,user->design_scatter));
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
/* A
MatShell object
*/
PetscErrorCode FormJacobianState(Tao tao, Vec X, Mat J, Mat JPre, Mat JInv, void *ptr)
{
  AppCtx         *user = (AppCtx*)ptr;

  PetscFunctionBegin;
  CHKERRQ(Scatter(X,user->y,user->state_scatter,user->u,user->design_scatter));
  CHKERRQ(VecSet(user->uwork,0));
  CHKERRQ(VecAXPY(user->uwork,-1.0,user->u));
  CHKERRQ(VecExp(user->uwork));
  CHKERRQ(MatMult(user->Av,user->uwork,user->Av_u));
  CHKERRQ(VecCopy(user->Av_u,user->Swork));
  CHKERRQ(VecReciprocal(user->Swork));
  CHKERRQ(MatCopy(user->Div,user->Divwork,SAME_NONZERO_PATTERN));
  CHKERRQ(MatDiagonalScale(user->Divwork,NULL,user->Swork));
  if (user->dsg_formed) {
    CHKERRQ(MatProductNumeric(user->DSG));
  } else {
    CHKERRQ(MatMatMult(user->Divwork,user->Grad,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&user->DSG));
    user->dsg_formed = PETSC_TRUE;
  }

  /* B = speye(nx^3) + ht*DSG; */
  CHKERRQ(MatScale(user->DSG,user->ht));
  CHKERRQ(MatShift(user->DSG,1.0));
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
/* B */
PetscErrorCode FormJacobianDesign(Tao tao, Vec X, Mat J, void *ptr)
{
  AppCtx         *user = (AppCtx*)ptr;

  PetscFunctionBegin;
  CHKERRQ(Scatter(X,user->y,user->state_scatter,user->u,user->design_scatter));
  PetscFunctionReturn(0);
}

PetscErrorCode StateMatMult(Mat J_shell, Vec X, Vec Y)
{
  PetscInt       i;
  AppCtx         *user;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(J_shell,&user));
  CHKERRQ(Scatter_i(X,user->yi,user->yi_scatter,user->nt));
  CHKERRQ(MatMult(user->JsBlock,user->yi[0],user->yiwork[0]));
  for (i=1; i<user->nt; i++) {
    CHKERRQ(MatMult(user->JsBlock,user->yi[i],user->yiwork[i]));
    CHKERRQ(VecAXPY(user->yiwork[i],-1.0,user->yi[i-1]));
  }
  CHKERRQ(Gather_i(Y,user->yiwork,user->yi_scatter,user->nt));
  PetscFunctionReturn(0);
}

PetscErrorCode StateMatMultTranspose(Mat J_shell, Vec X, Vec Y)
{
  PetscInt       i;
  AppCtx         *user;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(J_shell,&user));
  CHKERRQ(Scatter_i(X,user->yi,user->yi_scatter,user->nt));
  for (i=0; i<user->nt-1; i++) {
    CHKERRQ(MatMult(user->JsBlock,user->yi[i],user->yiwork[i]));
    CHKERRQ(VecAXPY(user->yiwork[i],-1.0,user->yi[i+1]));
  }
  i = user->nt-1;
  CHKERRQ(MatMult(user->JsBlock,user->yi[i],user->yiwork[i]));
  CHKERRQ(Gather_i(Y,user->yiwork,user->yi_scatter,user->nt));
  PetscFunctionReturn(0);
}

PetscErrorCode StateMatBlockMult(Mat J_shell, Vec X, Vec Y)
{
  AppCtx         *user;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(J_shell,&user));
  CHKERRQ(MatMult(user->Grad,X,user->Swork));
  CHKERRQ(VecPointwiseDivide(user->Swork,user->Swork,user->Av_u));
  CHKERRQ(MatMult(user->Div,user->Swork,Y));
  CHKERRQ(VecAYPX(Y,user->ht,X));
  PetscFunctionReturn(0);
}

PetscErrorCode DesignMatMult(Mat J_shell, Vec X, Vec Y)
{
  PetscInt       i;
  AppCtx         *user;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(J_shell,&user));

  /* sdiag(1./v) */
  CHKERRQ(VecSet(user->uwork,0));
  CHKERRQ(VecAXPY(user->uwork,-1.0,user->u));
  CHKERRQ(VecExp(user->uwork));

  /* sdiag(1./((Av*(1./v)).^2)) */
  CHKERRQ(MatMult(user->Av,user->uwork,user->Swork));
  CHKERRQ(VecPointwiseMult(user->Swork,user->Swork,user->Swork));
  CHKERRQ(VecReciprocal(user->Swork));

  /* (Av * (sdiag(1./v) * b)) */
  CHKERRQ(VecPointwiseMult(user->uwork,user->uwork,X));
  CHKERRQ(MatMult(user->Av,user->uwork,user->Twork));

  /* (sdiag(1./((Av*(1./v)).^2)) * (Av * (sdiag(1./v) * b))) */
  CHKERRQ(VecPointwiseMult(user->Swork,user->Twork,user->Swork));

  CHKERRQ(Scatter_i(user->y,user->yi,user->yi_scatter,user->nt));
  for (i=0; i<user->nt; i++) {
    /* (sdiag(Grad*y(:,i)) */
    CHKERRQ(MatMult(user->Grad,user->yi[i],user->Twork));

    /* ht * Div * (sdiag(Grad*y(:,i)) * (sdiag(1./((Av*(1./v)).^2)) * (Av * (sdiag(1./v) * b)))) */
    CHKERRQ(VecPointwiseMult(user->Twork,user->Twork,user->Swork));
    CHKERRQ(MatMult(user->Div,user->Twork,user->yiwork[i]));
    CHKERRQ(VecScale(user->yiwork[i],user->ht));
  }
  CHKERRQ(Gather_i(Y,user->yiwork,user->yi_scatter,user->nt));

  PetscFunctionReturn(0);
}

PetscErrorCode DesignMatMultTranspose(Mat J_shell, Vec X, Vec Y)
{
  PetscInt       i;
  AppCtx         *user;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(J_shell,&user));

  /* sdiag(1./((Av*(1./v)).^2)) */
  CHKERRQ(VecSet(user->uwork,0));
  CHKERRQ(VecAXPY(user->uwork,-1.0,user->u));
  CHKERRQ(VecExp(user->uwork));
  CHKERRQ(MatMult(user->Av,user->uwork,user->Rwork));
  CHKERRQ(VecPointwiseMult(user->Rwork,user->Rwork,user->Rwork));
  CHKERRQ(VecReciprocal(user->Rwork));

  CHKERRQ(VecSet(Y,0.0));
  CHKERRQ(Scatter_i(user->y,user->yi,user->yi_scatter,user->nt));
  CHKERRQ(Scatter_i(X,user->yiwork,user->yi_scatter,user->nt));
  for (i=0; i<user->nt; i++) {
    /* (Div' * b(:,i)) */
    CHKERRQ(MatMult(user->Grad,user->yiwork[i],user->Swork));

    /* sdiag(Grad*y(:,i)) */
    CHKERRQ(MatMult(user->Grad,user->yi[i],user->Twork));

    /* (sdiag(Grad*y(:,i)) * (Div' * b(:,i))) */
    CHKERRQ(VecPointwiseMult(user->Twork,user->Swork,user->Twork));

    /* (sdiag(1./((Av*(1./v)).^2)) * (sdiag(Grad*y(:,i)) * (Div' * b(:,i)))) */
    CHKERRQ(VecPointwiseMult(user->Twork,user->Rwork,user->Twork));

    /* (Av' * (sdiag(1./((Av*(1./v)).^2)) * (sdiag(Grad*y(:,i)) * (Div' * b(:,i))))) */
    CHKERRQ(MatMult(user->AvT,user->Twork,user->yiwork[i]));

    /* sdiag(1./v) * (Av' * (sdiag(1./((Av*(1./v)).^2)) * (sdiag(Grad*y(:,i)) * (Div' * b(:,i))))) */
    CHKERRQ(VecPointwiseMult(user->yiwork[i],user->uwork,user->yiwork[i]));
    CHKERRQ(VecAXPY(Y,user->ht,user->yiwork[i]));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode StateMatBlockPrecMult(PC PC_shell, Vec X, Vec Y)
{
  AppCtx         *user;

  PetscFunctionBegin;
  CHKERRQ(PCShellGetContext(PC_shell,&user));

  if (user->dsg_formed) {
    CHKERRQ(MatSOR(user->DSG,X,1.0,(MatSORType)(SOR_ZERO_INITIAL_GUESS | SOR_LOCAL_SYMMETRIC_SWEEP),0.0,1,1,Y));
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"DSG not formed");
  PetscFunctionReturn(0);
}

PetscErrorCode StateMatInvMult(Mat J_shell, Vec X, Vec Y)
{
  AppCtx         *user;
  PetscInt       its,i;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(J_shell,&user));

  if (Y == user->ytrue) {
    /* First solve is done with true solution to set up problem */
    CHKERRQ(KSPSetTolerances(user->solver,1e-8,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT));
  } else {
    CHKERRQ(KSPSetTolerances(user->solver,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT));
  }

  CHKERRQ(Scatter_i(X,user->yi,user->yi_scatter,user->nt));
  CHKERRQ(KSPSolve(user->solver,user->yi[0],user->yiwork[0]));
  CHKERRQ(KSPGetIterationNumber(user->solver,&its));
  user->ksp_its = user->ksp_its + its;

  for (i=1; i<user->nt; i++) {
    CHKERRQ(VecAXPY(user->yi[i],1.0,user->yiwork[i-1]));
    CHKERRQ(KSPSolve(user->solver,user->yi[i],user->yiwork[i]));
    CHKERRQ(KSPGetIterationNumber(user->solver,&its));
    user->ksp_its = user->ksp_its + its;
  }
  CHKERRQ(Gather_i(Y,user->yiwork,user->yi_scatter,user->nt));
  PetscFunctionReturn(0);
}

PetscErrorCode StateMatInvTransposeMult(Mat J_shell, Vec X, Vec Y)
{
  AppCtx         *user;
  PetscInt       its,i;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(J_shell,&user));

  CHKERRQ(Scatter_i(X,user->yi,user->yi_scatter,user->nt));

  i = user->nt - 1;
  CHKERRQ(KSPSolve(user->solver,user->yi[i],user->yiwork[i]));

  CHKERRQ(KSPGetIterationNumber(user->solver,&its));
  user->ksp_its = user->ksp_its + its;

  for (i=user->nt-2; i>=0; i--) {
    CHKERRQ(VecAXPY(user->yi[i],1.0,user->yiwork[i+1]));
    CHKERRQ(KSPSolve(user->solver,user->yi[i],user->yiwork[i]));

    CHKERRQ(KSPGetIterationNumber(user->solver,&its));
    user->ksp_its = user->ksp_its + its;

  }

  CHKERRQ(Gather_i(Y,user->yiwork,user->yi_scatter,user->nt));
  PetscFunctionReturn(0);
}

PetscErrorCode StateMatDuplicate(Mat J_shell, MatDuplicateOption opt, Mat *new_shell)
{
  AppCtx         *user;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(J_shell,&user));

  CHKERRQ(MatCreateShell(PETSC_COMM_WORLD,PETSC_DETERMINE,PETSC_DETERMINE,user->m,user->m,user,new_shell));
  CHKERRQ(MatShellSetOperation(*new_shell,MATOP_MULT,(void(*)(void))StateMatMult));
  CHKERRQ(MatShellSetOperation(*new_shell,MATOP_DUPLICATE,(void(*)(void))StateMatDuplicate));
  CHKERRQ(MatShellSetOperation(*new_shell,MATOP_MULT_TRANSPOSE,(void(*)(void))StateMatMultTranspose));
  CHKERRQ(MatShellSetOperation(*new_shell,MATOP_GET_DIAGONAL,(void(*)(void))StateMatGetDiagonal));
  PetscFunctionReturn(0);
}

PetscErrorCode StateMatGetDiagonal(Mat J_shell, Vec X)
{
  AppCtx         *user;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(J_shell,&user));
  CHKERRQ(VecCopy(user->js_diag,X));
  PetscFunctionReturn(0);

}

PetscErrorCode FormConstraints(Tao tao, Vec X, Vec C, void *ptr)
{
  /* con = Ay - q, A = [B  0  0 ... 0;
                       -I  B  0 ... 0;
                        0 -I  B ... 0;
                             ...     ;
                        0    ... -I B]
     B = ht * Div * Sigma * Grad + eye */
  PetscInt       i;
  AppCtx         *user = (AppCtx*)ptr;

  PetscFunctionBegin;
  CHKERRQ(Scatter(X,user->y,user->state_scatter,user->u,user->design_scatter));
  CHKERRQ(Scatter_i(user->y,user->yi,user->yi_scatter,user->nt));
  CHKERRQ(MatMult(user->JsBlock,user->yi[0],user->yiwork[0]));
  for (i=1; i<user->nt; i++) {
    CHKERRQ(MatMult(user->JsBlock,user->yi[i],user->yiwork[i]));
    CHKERRQ(VecAXPY(user->yiwork[i],-1.0,user->yi[i-1]));
  }
  CHKERRQ(Gather_i(C,user->yiwork,user->yi_scatter,user->nt));
  CHKERRQ(VecAXPY(C,-1.0,user->q));
  PetscFunctionReturn(0);
}

PetscErrorCode Scatter(Vec x, Vec state, VecScatter s_scat, Vec design, VecScatter d_scat)
{
  PetscFunctionBegin;
  CHKERRQ(VecScatterBegin(s_scat,x,state,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(s_scat,x,state,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterBegin(d_scat,x,design,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(d_scat,x,design,INSERT_VALUES,SCATTER_FORWARD));
  PetscFunctionReturn(0);
}

PetscErrorCode Scatter_i(Vec y, Vec *yi, VecScatter *scat, PetscInt nt)
{
  PetscInt       i;

  PetscFunctionBegin;
  for (i=0; i<nt; i++) {
    CHKERRQ(VecScatterBegin(scat[i],y,yi[i],INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecScatterEnd(scat[i],y,yi[i],INSERT_VALUES,SCATTER_FORWARD));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode Gather(Vec x, Vec state, VecScatter s_scat, Vec design, VecScatter d_scat)
{
  PetscFunctionBegin;
  CHKERRQ(VecScatterBegin(s_scat,state,x,INSERT_VALUES,SCATTER_REVERSE));
  CHKERRQ(VecScatterEnd(s_scat,state,x,INSERT_VALUES,SCATTER_REVERSE));
  CHKERRQ(VecScatterBegin(d_scat,design,x,INSERT_VALUES,SCATTER_REVERSE));
  CHKERRQ(VecScatterEnd(d_scat,design,x,INSERT_VALUES,SCATTER_REVERSE));
  PetscFunctionReturn(0);
}

PetscErrorCode Gather_i(Vec y, Vec *yi, VecScatter *scat, PetscInt nt)
{
  PetscInt       i;

  PetscFunctionBegin;
  for (i=0; i<nt; i++) {
    CHKERRQ(VecScatterBegin(scat[i],yi[i],y,INSERT_VALUES,SCATTER_REVERSE));
    CHKERRQ(VecScatterEnd(scat[i],yi[i],y,INSERT_VALUES,SCATTER_REVERSE));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode ParabolicInitialize(AppCtx *user)
{
  PetscInt       m,n,i,j,k,linear_index,istart,iend,iblock,lo,hi,lo2,hi2;
  Vec            XX,YY,ZZ,XXwork,YYwork,ZZwork,UTwork,yi,di,bc;
  PetscReal      *x, *y, *z;
  PetscReal      h,stime;
  PetscScalar    hinv,neg_hinv,half = 0.5,sqrt_beta;
  PetscInt       im,indx1,indx2,indy1,indy2,indz1,indz2,nx,ny,nz;
  PetscReal      xri,yri,zri,xim,yim,zim,dx1,dx2,dy1,dy2,dz1,dz2,Dx,Dy,Dz;
  PetscScalar    v,vx,vy,vz;
  IS             is_from_y,is_to_yi,is_from_d,is_to_di;
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

  PetscFunctionBegin;
  CHKERRQ(PetscMalloc1(user->mx,&x));
  CHKERRQ(PetscMalloc1(user->mx,&y));
  CHKERRQ(PetscMalloc1(user->mx,&z));
  user->jformed = PETSC_FALSE;
  user->dsg_formed = PETSC_FALSE;

  n = user->mx * user->mx * user->mx;
  m = 3 * user->mx * user->mx * (user->mx-1);
  sqrt_beta = PetscSqrtScalar(user->beta);

  user->ksp_its = 0;
  user->ksp_its_initial = 0;

  stime = (PetscReal)user->nt/user->ns;
  CHKERRQ(PetscMalloc1(user->ns,&user->sample_times));
  for (i=0; i<user->ns; i++) {
    user->sample_times[i] = (PetscInt)(stime*((PetscReal)i+1.0)-0.5);
  }

  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&XX));
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&user->q));
  CHKERRQ(VecSetSizes(XX,PETSC_DECIDE,n));
  CHKERRQ(VecSetSizes(user->q,PETSC_DECIDE,n*user->nt));
  CHKERRQ(VecSetFromOptions(XX));
  CHKERRQ(VecSetFromOptions(user->q));

  CHKERRQ(VecDuplicate(XX,&YY));
  CHKERRQ(VecDuplicate(XX,&ZZ));
  CHKERRQ(VecDuplicate(XX,&XXwork));
  CHKERRQ(VecDuplicate(XX,&YYwork));
  CHKERRQ(VecDuplicate(XX,&ZZwork));
  CHKERRQ(VecDuplicate(XX,&UTwork));
  CHKERRQ(VecDuplicate(XX,&user->utrue));
  CHKERRQ(VecDuplicate(XX,&bc));

  /* Generate 3D grid, and collect ns (1<=ns<=8) right-hand-side vectors into user->q */
  h = 1.0/user->mx;
  hinv = user->mx;
  neg_hinv = -hinv;

  CHKERRQ(VecGetOwnershipRange(XX,&istart,&iend));
  for (linear_index=istart; linear_index<iend; linear_index++) {
    i = linear_index % user->mx;
    j = ((linear_index-i)/user->mx) % user->mx;
    k = ((linear_index-i)/user->mx-j) / user->mx;
    vx = h*(i+0.5);
    vy = h*(j+0.5);
    vz = h*(k+0.5);
    CHKERRQ(VecSetValues(XX,1,&linear_index,&vx,INSERT_VALUES));
    CHKERRQ(VecSetValues(YY,1,&linear_index,&vy,INSERT_VALUES));
    CHKERRQ(VecSetValues(ZZ,1,&linear_index,&vz,INSERT_VALUES));
    if ((vx<0.6) && (vx>0.4) && (vy<0.6) && (vy>0.4) && (vy<0.6) && (vz<0.6) && (vz>0.4)) {
      v = 1000.0;
      CHKERRQ(VecSetValues(bc,1,&linear_index,&v,INSERT_VALUES));
    }
  }

  CHKERRQ(VecAssemblyBegin(XX));
  CHKERRQ(VecAssemblyEnd(XX));
  CHKERRQ(VecAssemblyBegin(YY));
  CHKERRQ(VecAssemblyEnd(YY));
  CHKERRQ(VecAssemblyBegin(ZZ));
  CHKERRQ(VecAssemblyEnd(ZZ));
  CHKERRQ(VecAssemblyBegin(bc));
  CHKERRQ(VecAssemblyEnd(bc));

  /* Compute true parameter function
     ut = 0.5 + exp(-10*((x-0.5)^2+(y-0.5)^2+(z-0.5)^2)) */
  CHKERRQ(VecCopy(XX,XXwork));
  CHKERRQ(VecCopy(YY,YYwork));
  CHKERRQ(VecCopy(ZZ,ZZwork));

  CHKERRQ(VecShift(XXwork,-0.5));
  CHKERRQ(VecShift(YYwork,-0.5));
  CHKERRQ(VecShift(ZZwork,-0.5));

  CHKERRQ(VecPointwiseMult(XXwork,XXwork,XXwork));
  CHKERRQ(VecPointwiseMult(YYwork,YYwork,YYwork));
  CHKERRQ(VecPointwiseMult(ZZwork,ZZwork,ZZwork));

  CHKERRQ(VecCopy(XXwork,user->utrue));
  CHKERRQ(VecAXPY(user->utrue,1.0,YYwork));
  CHKERRQ(VecAXPY(user->utrue,1.0,ZZwork));
  CHKERRQ(VecScale(user->utrue,-10.0));
  CHKERRQ(VecExp(user->utrue));
  CHKERRQ(VecShift(user->utrue,0.5));

  CHKERRQ(VecDestroy(&XX));
  CHKERRQ(VecDestroy(&YY));
  CHKERRQ(VecDestroy(&ZZ));
  CHKERRQ(VecDestroy(&XXwork));
  CHKERRQ(VecDestroy(&YYwork));
  CHKERRQ(VecDestroy(&ZZwork));
  CHKERRQ(VecDestroy(&UTwork));

  /* Initial guess and reference model */
  CHKERRQ(VecDuplicate(user->utrue,&user->ur));
  CHKERRQ(VecSet(user->ur,0.0));

  /* Generate Grad matrix */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&user->Grad));
  CHKERRQ(MatSetSizes(user->Grad,PETSC_DECIDE,PETSC_DECIDE,m,n));
  CHKERRQ(MatSetFromOptions(user->Grad));
  CHKERRQ(MatMPIAIJSetPreallocation(user->Grad,2,NULL,2,NULL));
  CHKERRQ(MatSeqAIJSetPreallocation(user->Grad,2,NULL));
  CHKERRQ(MatGetOwnershipRange(user->Grad,&istart,&iend));

  for (i=istart; i<iend; i++) {
    if (i<m/3) {
      iblock = i / (user->mx-1);
      j = iblock*user->mx + (i % (user->mx-1));
      CHKERRQ(MatSetValues(user->Grad,1,&i,1,&j,&neg_hinv,INSERT_VALUES));
      j = j+1;
      CHKERRQ(MatSetValues(user->Grad,1,&i,1,&j,&hinv,INSERT_VALUES));
    }
    if (i>=m/3 && i<2*m/3) {
      iblock = (i-m/3) / (user->mx*(user->mx-1));
      j = iblock*user->mx*user->mx + ((i-m/3) % (user->mx*(user->mx-1)));
      CHKERRQ(MatSetValues(user->Grad,1,&i,1,&j,&neg_hinv,INSERT_VALUES));
      j = j + user->mx;
      CHKERRQ(MatSetValues(user->Grad,1,&i,1,&j,&hinv,INSERT_VALUES));
    }
    if (i>=2*m/3) {
      j = i-2*m/3;
      CHKERRQ(MatSetValues(user->Grad,1,&i,1,&j,&neg_hinv,INSERT_VALUES));
      j = j + user->mx*user->mx;
      CHKERRQ(MatSetValues(user->Grad,1,&i,1,&j,&hinv,INSERT_VALUES));
    }
  }

  CHKERRQ(MatAssemblyBegin(user->Grad,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(user->Grad,MAT_FINAL_ASSEMBLY));

  /* Generate arithmetic averaging matrix Av */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&user->Av));
  CHKERRQ(MatSetSizes(user->Av,PETSC_DECIDE,PETSC_DECIDE,m,n));
  CHKERRQ(MatSetFromOptions(user->Av));
  CHKERRQ(MatMPIAIJSetPreallocation(user->Av,2,NULL,2,NULL));
  CHKERRQ(MatSeqAIJSetPreallocation(user->Av,2,NULL));
  CHKERRQ(MatGetOwnershipRange(user->Av,&istart,&iend));

  for (i=istart; i<iend; i++) {
    if (i<m/3) {
      iblock = i / (user->mx-1);
      j = iblock*user->mx + (i % (user->mx-1));
      CHKERRQ(MatSetValues(user->Av,1,&i,1,&j,&half,INSERT_VALUES));
      j = j+1;
      CHKERRQ(MatSetValues(user->Av,1,&i,1,&j,&half,INSERT_VALUES));
    }
    if (i>=m/3 && i<2*m/3) {
      iblock = (i-m/3) / (user->mx*(user->mx-1));
      j = iblock*user->mx*user->mx + ((i-m/3) % (user->mx*(user->mx-1)));
      CHKERRQ(MatSetValues(user->Av,1,&i,1,&j,&half,INSERT_VALUES));
      j = j + user->mx;
      CHKERRQ(MatSetValues(user->Av,1,&i,1,&j,&half,INSERT_VALUES));
    }
    if (i>=2*m/3) {
      j = i-2*m/3;
      CHKERRQ(MatSetValues(user->Av,1,&i,1,&j,&half,INSERT_VALUES));
      j = j + user->mx*user->mx;
      CHKERRQ(MatSetValues(user->Av,1,&i,1,&j,&half,INSERT_VALUES));
    }
  }

  CHKERRQ(MatAssemblyBegin(user->Av,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(user->Av,MAT_FINAL_ASSEMBLY));

  /* Generate transpose of averaging matrix Av */
  CHKERRQ(MatTranspose(user->Av,MAT_INITIAL_MATRIX,&user->AvT));

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&user->L));
  CHKERRQ(MatSetSizes(user->L,PETSC_DECIDE,PETSC_DECIDE,m+n,n));
  CHKERRQ(MatSetFromOptions(user->L));
  CHKERRQ(MatMPIAIJSetPreallocation(user->L,2,NULL,2,NULL));
  CHKERRQ(MatSeqAIJSetPreallocation(user->L,2,NULL));
  CHKERRQ(MatGetOwnershipRange(user->L,&istart,&iend));

  for (i=istart; i<iend; i++) {
    if (i<m/3) {
      iblock = i / (user->mx-1);
      j = iblock*user->mx + (i % (user->mx-1));
      CHKERRQ(MatSetValues(user->L,1,&i,1,&j,&neg_hinv,INSERT_VALUES));
      j = j+1;
      CHKERRQ(MatSetValues(user->L,1,&i,1,&j,&hinv,INSERT_VALUES));
    }
    if (i>=m/3 && i<2*m/3) {
      iblock = (i-m/3) / (user->mx*(user->mx-1));
      j = iblock*user->mx*user->mx + ((i-m/3) % (user->mx*(user->mx-1)));
      CHKERRQ(MatSetValues(user->L,1,&i,1,&j,&neg_hinv,INSERT_VALUES));
      j = j + user->mx;
      CHKERRQ(MatSetValues(user->L,1,&i,1,&j,&hinv,INSERT_VALUES));
    }
    if (i>=2*m/3 && i<m) {
      j = i-2*m/3;
      CHKERRQ(MatSetValues(user->L,1,&i,1,&j,&neg_hinv,INSERT_VALUES));
      j = j + user->mx*user->mx;
      CHKERRQ(MatSetValues(user->L,1,&i,1,&j,&hinv,INSERT_VALUES));
    }
    if (i>=m) {
      j = i - m;
      CHKERRQ(MatSetValues(user->L,1,&i,1,&j,&sqrt_beta,INSERT_VALUES));
    }
  }

  CHKERRQ(MatAssemblyBegin(user->L,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(user->L,MAT_FINAL_ASSEMBLY));

  CHKERRQ(MatScale(user->L,PetscPowScalar(h,1.5)));

  /* Generate Div matrix */
  CHKERRQ(MatTranspose(user->Grad,MAT_INITIAL_MATRIX,&user->Div));

  /* Build work vectors and matrices */
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&user->S));
  CHKERRQ(VecSetSizes(user->S, PETSC_DECIDE, user->mx*user->mx*(user->mx-1)*3));
  CHKERRQ(VecSetFromOptions(user->S));

  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&user->lwork));
  CHKERRQ(VecSetSizes(user->lwork,PETSC_DECIDE,m+user->mx*user->mx*user->mx));
  CHKERRQ(VecSetFromOptions(user->lwork));

  CHKERRQ(MatDuplicate(user->Div,MAT_SHARE_NONZERO_PATTERN,&user->Divwork));
  CHKERRQ(MatDuplicate(user->Av,MAT_SHARE_NONZERO_PATTERN,&user->Avwork));

  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&user->d));
  CHKERRQ(VecSetSizes(user->d,PETSC_DECIDE,user->ndata*user->nt));
  CHKERRQ(VecSetFromOptions(user->d));

  CHKERRQ(VecDuplicate(user->S,&user->Swork));
  CHKERRQ(VecDuplicate(user->S,&user->Av_u));
  CHKERRQ(VecDuplicate(user->S,&user->Twork));
  CHKERRQ(VecDuplicate(user->S,&user->Rwork));
  CHKERRQ(VecDuplicate(user->y,&user->ywork));
  CHKERRQ(VecDuplicate(user->u,&user->uwork));
  CHKERRQ(VecDuplicate(user->u,&user->js_diag));
  CHKERRQ(VecDuplicate(user->c,&user->cwork));
  CHKERRQ(VecDuplicate(user->d,&user->dwork));

  /* Create matrix-free shell user->Js for computing A*x */
  CHKERRQ(MatCreateShell(PETSC_COMM_WORLD,PETSC_DETERMINE,PETSC_DETERMINE,user->m*user->nt,user->m*user->nt,user,&user->Js));
  CHKERRQ(MatShellSetOperation(user->Js,MATOP_MULT,(void(*)(void))StateMatMult));
  CHKERRQ(MatShellSetOperation(user->Js,MATOP_DUPLICATE,(void(*)(void))StateMatDuplicate));
  CHKERRQ(MatShellSetOperation(user->Js,MATOP_MULT_TRANSPOSE,(void(*)(void))StateMatMultTranspose));
  CHKERRQ(MatShellSetOperation(user->Js,MATOP_GET_DIAGONAL,(void(*)(void))StateMatGetDiagonal));

  /* Diagonal blocks of user->Js */
  CHKERRQ(MatCreateShell(PETSC_COMM_WORLD,PETSC_DETERMINE,PETSC_DETERMINE,user->m,user->m,user,&user->JsBlock));
  CHKERRQ(MatShellSetOperation(user->JsBlock,MATOP_MULT,(void(*)(void))StateMatBlockMult));
  /* Blocks are symmetric */
  CHKERRQ(MatShellSetOperation(user->JsBlock,MATOP_MULT_TRANSPOSE,(void(*)(void))StateMatBlockMult));

  /* Create a matrix-free shell user->JsBlockPrec for computing (U+D)\D*(L+D)\x, where JsBlock = L+D+U,
     D is diagonal, L is strictly lower triangular, and U is strictly upper triangular.
     This is an SSOR preconditioner for user->JsBlock. */
  CHKERRQ(MatCreateShell(PETSC_COMM_WORLD,PETSC_DETERMINE,PETSC_DETERMINE,user->m,user->m,user,&user->JsBlockPrec));
  CHKERRQ(MatShellSetOperation(user->JsBlockPrec,MATOP_MULT,(void(*)(void))StateMatBlockPrecMult));
  /* JsBlockPrec is symmetric */
  CHKERRQ(MatShellSetOperation(user->JsBlockPrec,MATOP_MULT_TRANSPOSE,(void(*)(void))StateMatBlockPrecMult));
  CHKERRQ(MatSetOption(user->JsBlockPrec,MAT_SYMMETRY_ETERNAL,PETSC_TRUE));

  /* Create a matrix-free shell user->Jd for computing B*x */
  CHKERRQ(MatCreateShell(PETSC_COMM_WORLD,PETSC_DETERMINE,PETSC_DETERMINE,user->m*user->nt,user->m,user,&user->Jd));
  CHKERRQ(MatShellSetOperation(user->Jd,MATOP_MULT,(void(*)(void))DesignMatMult));
  CHKERRQ(MatShellSetOperation(user->Jd,MATOP_MULT_TRANSPOSE,(void(*)(void))DesignMatMultTranspose));

  /* User-defined routines for computing user->Js\x and user->Js^T\x*/
  CHKERRQ(MatCreateShell(PETSC_COMM_WORLD,PETSC_DETERMINE,PETSC_DETERMINE,user->m*user->nt,user->m*user->nt,user,&user->JsInv));
  CHKERRQ(MatShellSetOperation(user->JsInv,MATOP_MULT,(void(*)(void))StateMatInvMult));
  CHKERRQ(MatShellSetOperation(user->JsInv,MATOP_MULT_TRANSPOSE,(void(*)(void))StateMatInvTransposeMult));

  /* Solver options and tolerances */
  CHKERRQ(KSPCreate(PETSC_COMM_WORLD,&user->solver));
  CHKERRQ(KSPSetType(user->solver,KSPCG));
  CHKERRQ(KSPSetOperators(user->solver,user->JsBlock,user->JsBlockPrec));
  CHKERRQ(KSPSetInitialGuessNonzero(user->solver,PETSC_FALSE));
  CHKERRQ(KSPSetTolerances(user->solver,1e-4,1e-20,1e3,500));
  CHKERRQ(KSPSetFromOptions(user->solver));
  CHKERRQ(KSPGetPC(user->solver,&user->prec));
  CHKERRQ(PCSetType(user->prec,PCSHELL));

  CHKERRQ(PCShellSetApply(user->prec,StateMatBlockPrecMult));
  CHKERRQ(PCShellSetApplyTranspose(user->prec,StateMatBlockPrecMult));
  CHKERRQ(PCShellSetContext(user->prec,user));

  /* Create scatter from y to y_1,y_2,...,y_nt */
  CHKERRQ(PetscMalloc1(user->nt*user->m,&user->yi_scatter));
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&yi));
  CHKERRQ(VecSetSizes(yi,PETSC_DECIDE,user->mx*user->mx*user->mx));
  CHKERRQ(VecSetFromOptions(yi));
  CHKERRQ(VecDuplicateVecs(yi,user->nt,&user->yi));
  CHKERRQ(VecDuplicateVecs(yi,user->nt,&user->yiwork));

  CHKERRQ(VecGetOwnershipRange(user->y,&lo2,&hi2));
  istart = 0;
  for (i=0; i<user->nt; i++) {
    CHKERRQ(VecGetOwnershipRange(user->yi[i],&lo,&hi));
    CHKERRQ(ISCreateStride(PETSC_COMM_SELF,hi-lo,lo,1,&is_to_yi));
    CHKERRQ(ISCreateStride(PETSC_COMM_SELF,hi-lo,lo2+istart,1,&is_from_y));
    CHKERRQ(VecScatterCreate(user->y,is_from_y,user->yi[i],is_to_yi,&user->yi_scatter[i]));
    istart = istart + hi-lo;
    CHKERRQ(ISDestroy(&is_to_yi));
    CHKERRQ(ISDestroy(&is_from_y));
  }
  CHKERRQ(VecDestroy(&yi));

  /* Create scatter from d to d_1,d_2,...,d_ns */
  CHKERRQ(PetscMalloc1(user->ns*user->ndata,&user->di_scatter));
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&di));
  CHKERRQ(VecSetSizes(di,PETSC_DECIDE,user->ndata));
  CHKERRQ(VecSetFromOptions(di));
  CHKERRQ(VecDuplicateVecs(di,user->ns,&user->di));
  CHKERRQ(VecGetOwnershipRange(user->d,&lo2,&hi2));
  istart = 0;
  for (i=0; i<user->ns; i++) {
    CHKERRQ(VecGetOwnershipRange(user->di[i],&lo,&hi));
    CHKERRQ(ISCreateStride(PETSC_COMM_SELF,hi-lo,lo,1,&is_to_di));
    CHKERRQ(ISCreateStride(PETSC_COMM_SELF,hi-lo,lo2+istart,1,&is_from_d));
    CHKERRQ(VecScatterCreate(user->d,is_from_d,user->di[i],is_to_di,&user->di_scatter[i]));
    istart = istart + hi-lo;
    CHKERRQ(ISDestroy(&is_to_di));
    CHKERRQ(ISDestroy(&is_from_d));
  }
  CHKERRQ(VecDestroy(&di));

  /* Assemble RHS of forward problem */
  CHKERRQ(VecCopy(bc,user->yiwork[0]));
  for (i=1; i<user->nt; i++) {
    CHKERRQ(VecSet(user->yiwork[i],0.0));
  }
  CHKERRQ(Gather_i(user->q,user->yiwork,user->yi_scatter,user->nt));
  CHKERRQ(VecDestroy(&bc));

  /* Compute true state function yt given ut */
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&user->ytrue));
  CHKERRQ(VecSetSizes(user->ytrue,PETSC_DECIDE,n*user->nt));
  CHKERRQ(VecSetFromOptions(user->ytrue));

  /* First compute Av_u = Av*exp(-u) */
  CHKERRQ(VecSet(user->uwork,0));
  CHKERRQ(VecAXPY(user->uwork,-1.0,user->utrue)); /*  Note: user->utrue */
  CHKERRQ(VecExp(user->uwork));
  CHKERRQ(MatMult(user->Av,user->uwork,user->Av_u));

  /* Symbolic DSG = Div * Grad */
  CHKERRQ(MatProductCreate(user->Div,user->Grad,NULL,&user->DSG));
  CHKERRQ(MatProductSetType(user->DSG,MATPRODUCT_AB));
  CHKERRQ(MatProductSetAlgorithm(user->DSG,"default"));
  CHKERRQ(MatProductSetFill(user->DSG,PETSC_DEFAULT));
  CHKERRQ(MatProductSetFromOptions(user->DSG));
  CHKERRQ(MatProductSymbolic(user->DSG));

  user->dsg_formed = PETSC_TRUE;

  /* Next form DSG = Div*Grad */
  CHKERRQ(MatCopy(user->Div,user->Divwork,SAME_NONZERO_PATTERN));
  CHKERRQ(MatDiagonalScale(user->Divwork,NULL,user->Av_u));
  if (user->dsg_formed) {
    CHKERRQ(MatProductNumeric(user->DSG));
  } else {
    CHKERRQ(MatMatMult(user->Div,user->Grad,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&user->DSG));
    user->dsg_formed = PETSC_TRUE;
  }
  /* B = speye(nx^3) + ht*DSG; */
  CHKERRQ(MatScale(user->DSG,user->ht));
  CHKERRQ(MatShift(user->DSG,1.0));

  /* Now solve for ytrue */
  CHKERRQ(StateMatInvMult(user->Js,user->q,user->ytrue));

  /* Initial guess y0 for state given u0 */

  /* First compute Av_u = Av*exp(-u) */
  CHKERRQ(VecSet(user->uwork,0));
  CHKERRQ(VecAXPY(user->uwork,-1.0,user->u)); /*  Note: user->u */
  CHKERRQ(VecExp(user->uwork));
  CHKERRQ(MatMult(user->Av,user->uwork,user->Av_u));

  /* Next form DSG = Div*S*Grad */
  CHKERRQ(MatCopy(user->Div,user->Divwork,SAME_NONZERO_PATTERN));
  CHKERRQ(MatDiagonalScale(user->Divwork,NULL,user->Av_u));
  if (user->dsg_formed) {
    CHKERRQ(MatProductNumeric(user->DSG));
  } else {
    CHKERRQ(MatMatMult(user->Div,user->Grad,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&user->DSG));

    user->dsg_formed = PETSC_TRUE;
  }
  /* B = speye(nx^3) + ht*DSG; */
  CHKERRQ(MatScale(user->DSG,user->ht));
  CHKERRQ(MatShift(user->DSG,1.0));

  /* Now solve for y */
  CHKERRQ(StateMatInvMult(user->Js,user->q,user->y));

  /* Construct projection matrix Q, a block diagonal matrix consisting of nt copies of Qblock along the diagonal */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&user->Qblock));
  CHKERRQ(MatSetSizes(user->Qblock,PETSC_DECIDE,PETSC_DECIDE,user->ndata,n));
  CHKERRQ(MatSetFromOptions(user->Qblock));
  CHKERRQ(MatMPIAIJSetPreallocation(user->Qblock,8,NULL,8,NULL));
  CHKERRQ(MatSeqAIJSetPreallocation(user->Qblock,8,NULL));

  for (i=0; i<user->mx; i++) {
    x[i] = h*(i+0.5);
    y[i] = h*(i+0.5);
    z[i] = h*(i+0.5);
  }

  CHKERRQ(MatGetOwnershipRange(user->Qblock,&istart,&iend));
  nx = user->mx; ny = user->mx; nz = user->mx;
  for (i=istart; i<iend; i++) {
    xri = xr[i];
    im = 0;
    xim = x[im];
    while (xri>xim && im<nx) {
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
    while (yri>yim && im<ny) {
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
    while (zri>zim && im<nz) {
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
    CHKERRQ(MatSetValues(user->Qblock,1,&i,1,&j,&v,INSERT_VALUES));

    j = indx1 + indy1*nx + indz2*nx*ny;
    v = (1-dx1/Dx)*(1-dy1/Dy)*(1-dz2/Dz);
    CHKERRQ(MatSetValues(user->Qblock,1,&i,1,&j,&v,INSERT_VALUES));

    j = indx1 + indy2*nx + indz1*nx*ny;
    v = (1-dx1/Dx)*(1-dy2/Dy)*(1-dz1/Dz);
    CHKERRQ(MatSetValues(user->Qblock,1,&i,1,&j,&v,INSERT_VALUES));

    j = indx1 + indy2*nx + indz2*nx*ny;
    v = (1-dx1/Dx)*(1-dy2/Dy)*(1-dz2/Dz);
    CHKERRQ(MatSetValues(user->Qblock,1,&i,1,&j,&v,INSERT_VALUES));

    j = indx2 + indy1*nx + indz1*nx*ny;
    v = (1-dx2/Dx)*(1-dy1/Dy)*(1-dz1/Dz);
    CHKERRQ(MatSetValues(user->Qblock,1,&i,1,&j,&v,INSERT_VALUES));

    j = indx2 + indy1*nx + indz2*nx*ny;
    v = (1-dx2/Dx)*(1-dy1/Dy)*(1-dz2/Dz);
    CHKERRQ(MatSetValues(user->Qblock,1,&i,1,&j,&v,INSERT_VALUES));

    j = indx2 + indy2*nx + indz1*nx*ny;
    v = (1-dx2/Dx)*(1-dy2/Dy)*(1-dz1/Dz);
    CHKERRQ(MatSetValues(user->Qblock,1,&i,1,&j,&v,INSERT_VALUES));

    j = indx2 + indy2*nx + indz2*nx*ny;
    v = (1-dx2/Dx)*(1-dy2/Dy)*(1-dz2/Dz);
    CHKERRQ(MatSetValues(user->Qblock,1,&i,1,&j,&v,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(user->Qblock,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(user->Qblock,MAT_FINAL_ASSEMBLY));

  CHKERRQ(MatTranspose(user->Qblock,MAT_INITIAL_MATRIX,&user->QblockT));
  CHKERRQ(MatTranspose(user->L,MAT_INITIAL_MATRIX,&user->LT));

  /* Add noise to the measurement data */
  CHKERRQ(VecSet(user->ywork,1.0));
  CHKERRQ(VecAYPX(user->ywork,user->noise,user->ytrue));
  CHKERRQ(Scatter_i(user->ywork,user->yiwork,user->yi_scatter,user->nt));
  for (j=0; j<user->ns; j++) {
    i = user->sample_times[j];
    CHKERRQ(MatMult(user->Qblock,user->yiwork[i],user->di[j]));
  }
  CHKERRQ(Gather_i(user->d,user->di,user->di_scatter,user->ns));

  /* Now that initial conditions have been set, let the user pass tolerance options to the KSP solver */
  CHKERRQ(KSPSetFromOptions(user->solver));
  CHKERRQ(PetscFree(x));
  CHKERRQ(PetscFree(y));
  CHKERRQ(PetscFree(z));
  PetscFunctionReturn(0);
}

PetscErrorCode ParabolicDestroy(AppCtx *user)
{
  PetscInt       i;

  PetscFunctionBegin;
  CHKERRQ(MatDestroy(&user->Qblock));
  CHKERRQ(MatDestroy(&user->QblockT));
  CHKERRQ(MatDestroy(&user->Div));
  CHKERRQ(MatDestroy(&user->Divwork));
  CHKERRQ(MatDestroy(&user->Grad));
  CHKERRQ(MatDestroy(&user->Av));
  CHKERRQ(MatDestroy(&user->Avwork));
  CHKERRQ(MatDestroy(&user->AvT));
  CHKERRQ(MatDestroy(&user->DSG));
  CHKERRQ(MatDestroy(&user->L));
  CHKERRQ(MatDestroy(&user->LT));
  CHKERRQ(KSPDestroy(&user->solver));
  CHKERRQ(MatDestroy(&user->Js));
  CHKERRQ(MatDestroy(&user->Jd));
  CHKERRQ(MatDestroy(&user->JsInv));
  CHKERRQ(MatDestroy(&user->JsBlock));
  CHKERRQ(MatDestroy(&user->JsBlockPrec));
  CHKERRQ(VecDestroy(&user->u));
  CHKERRQ(VecDestroy(&user->uwork));
  CHKERRQ(VecDestroy(&user->utrue));
  CHKERRQ(VecDestroy(&user->y));
  CHKERRQ(VecDestroy(&user->ywork));
  CHKERRQ(VecDestroy(&user->ytrue));
  CHKERRQ(VecDestroyVecs(user->nt,&user->yi));
  CHKERRQ(VecDestroyVecs(user->nt,&user->yiwork));
  CHKERRQ(VecDestroyVecs(user->ns,&user->di));
  CHKERRQ(PetscFree(user->yi));
  CHKERRQ(PetscFree(user->yiwork));
  CHKERRQ(PetscFree(user->di));
  CHKERRQ(VecDestroy(&user->c));
  CHKERRQ(VecDestroy(&user->cwork));
  CHKERRQ(VecDestroy(&user->ur));
  CHKERRQ(VecDestroy(&user->q));
  CHKERRQ(VecDestroy(&user->d));
  CHKERRQ(VecDestroy(&user->dwork));
  CHKERRQ(VecDestroy(&user->lwork));
  CHKERRQ(VecDestroy(&user->S));
  CHKERRQ(VecDestroy(&user->Swork));
  CHKERRQ(VecDestroy(&user->Av_u));
  CHKERRQ(VecDestroy(&user->Twork));
  CHKERRQ(VecDestroy(&user->Rwork));
  CHKERRQ(VecDestroy(&user->js_diag));
  CHKERRQ(ISDestroy(&user->s_is));
  CHKERRQ(ISDestroy(&user->d_is));
  CHKERRQ(VecScatterDestroy(&user->state_scatter));
  CHKERRQ(VecScatterDestroy(&user->design_scatter));
  for (i=0; i<user->nt; i++) {
    CHKERRQ(VecScatterDestroy(&user->yi_scatter[i]));
  }
  for (i=0; i<user->ns; i++) {
    CHKERRQ(VecScatterDestroy(&user->di_scatter[i]));
  }
  CHKERRQ(PetscFree(user->yi_scatter));
  CHKERRQ(PetscFree(user->di_scatter));
  CHKERRQ(PetscFree(user->sample_times));
  PetscFunctionReturn(0);
}

PetscErrorCode ParabolicMonitor(Tao tao, void *ptr)
{
  Vec            X;
  PetscReal      unorm,ynorm;
  AppCtx         *user = (AppCtx*)ptr;

  PetscFunctionBegin;
  CHKERRQ(TaoGetSolution(tao,&X));
  CHKERRQ(Scatter(X,user->ywork,user->state_scatter,user->uwork,user->design_scatter));
  CHKERRQ(VecAXPY(user->ywork,-1.0,user->ytrue));
  CHKERRQ(VecAXPY(user->uwork,-1.0,user->utrue));
  CHKERRQ(VecNorm(user->uwork,NORM_2,&unorm));
  CHKERRQ(VecNorm(user->ywork,NORM_2,&ynorm));
  CHKERRQ(PetscPrintf(MPI_COMM_WORLD, "||u-ut||=%g ||y-yt||=%g\n",(double)unorm,(double)ynorm));
  PetscFunctionReturn(0);
}

/*TEST

   build:
      requires: !complex

   test:
      args: -tao_cmonitor -tao_type lcl -ns 1 -tao_gatol 1.e-4 -ksp_max_it 30
      requires: !single

TEST*/
