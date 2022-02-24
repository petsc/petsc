#include <petsctao.h>

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
   Processors: 1
T*/

typedef struct {
  PetscInt n; /*  Number of variables */
  PetscInt m; /*  Number of constraints */
  PetscInt mx; /*  grid points in each direction */
  PetscInt nt; /*  Number of time steps */
  PetscInt ndata; /*  Number of data points per sample */
  IS       s_is;
  IS       d_is;
  VecScatter state_scatter;
  VecScatter design_scatter;
  VecScatter *uxi_scatter,*uyi_scatter,*ux_scatter,*uy_scatter,*ui_scatter;
  VecScatter *yi_scatter;

  Mat       Js,Jd,JsBlockPrec,JsInv,JsBlock;
  PetscBool jformed,c_formed;

  PetscReal alpha; /*  Regularization parameter */
  PetscReal gamma;
  PetscReal ht; /*  Time step */
  PetscReal T; /*  Final time */
  Mat Q,QT;
  Mat L,LT;
  Mat Div,Divwork,Divxy[2];
  Mat Grad,Gradxy[2];
  Mat M;
  Mat *C,*Cwork;
  /* Mat Hs,Hd,Hsd; */
  Vec q;
  Vec ur; /*  reference */

  Vec d;
  Vec dwork;

  Vec y; /*  state variables */
  Vec ywork;
  Vec ytrue;
  Vec *yi,*yiwork,*ziwork;
  Vec *uxi,*uyi,*uxiwork,*uyiwork,*ui,*uiwork;

  Vec u; /*  design variables */
  Vec uwork,vwork;
  Vec utrue;

  Vec js_diag;

  Vec c; /*  constraint vector */
  Vec cwork;

  Vec lwork;

  KSP      solver;
  PC       prec;
  PetscInt block_index;

  PetscInt ksp_its;
  PetscInt ksp_its_initial;
} AppCtx;

PetscErrorCode FormFunction(Tao, Vec, PetscReal*, void*);
PetscErrorCode FormGradient(Tao, Vec, Vec, void*);
PetscErrorCode FormFunctionGradient(Tao, Vec, PetscReal*, Vec, void*);
PetscErrorCode FormJacobianState(Tao, Vec, Mat, Mat, Mat, void*);
PetscErrorCode FormJacobianDesign(Tao, Vec, Mat,void*);
PetscErrorCode FormConstraints(Tao, Vec, Vec, void*);
PetscErrorCode FormHessian(Tao, Vec, Mat, Mat, void*);
PetscErrorCode Gather(Vec x, Vec state, VecScatter s_scat, Vec design, VecScatter d_scat);
PetscErrorCode Scatter(Vec x, Vec state, VecScatter s_scat, Vec design, VecScatter d_scat);
PetscErrorCode HyperbolicInitialize(AppCtx *user);
PetscErrorCode HyperbolicDestroy(AppCtx *user);
PetscErrorCode HyperbolicMonitor(Tao, void*);

PetscErrorCode StateMatMult(Mat,Vec,Vec);
PetscErrorCode StateMatBlockMult(Mat,Vec,Vec);
PetscErrorCode StateMatBlockMultTranspose(Mat,Vec,Vec);
PetscErrorCode StateMatMultTranspose(Mat,Vec,Vec);
PetscErrorCode StateMatGetDiagonal(Mat,Vec);
PetscErrorCode StateMatDuplicate(Mat,MatDuplicateOption,Mat*);
PetscErrorCode StateMatInvMult(Mat,Vec,Vec);
PetscErrorCode StateMatInvTransposeMult(Mat,Vec,Vec);
PetscErrorCode StateMatBlockPrecMult(PC,Vec,Vec);

PetscErrorCode DesignMatMult(Mat,Vec,Vec);
PetscErrorCode DesignMatMultTranspose(Mat,Vec,Vec);

PetscErrorCode Scatter_yi(Vec,Vec*,VecScatter*,PetscInt); /*  y to y1,y2,...,y_nt */
PetscErrorCode Gather_yi(Vec,Vec*,VecScatter*,PetscInt);
PetscErrorCode Scatter_uxi_uyi(Vec,Vec*,VecScatter*,Vec*,VecScatter*,PetscInt); /*  u to ux_1,uy_1,ux_2,uy_2,...,u */
PetscErrorCode Gather_uxi_uyi(Vec,Vec*,VecScatter*,Vec*,VecScatter*,PetscInt);

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
  user.mx = 32;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"hyperbolic example",NULL);CHKERRQ(ierr);
  CHKERRQ(PetscOptionsInt("-mx","Number of grid points in each direction","",user.mx,&user.mx,NULL));
  user.nt = 16;
  CHKERRQ(PetscOptionsInt("-nt","Number of time steps","",user.nt,&user.nt,NULL));
  user.ndata = 64;
  CHKERRQ(PetscOptionsInt("-ndata","Numbers of data points per sample","",user.ndata,&user.ndata,NULL));
  user.alpha = 10.0;
  CHKERRQ(PetscOptionsReal("-alpha","Regularization parameter","",user.alpha,&user.alpha,NULL));
  user.T = 1.0/32.0;
  CHKERRQ(PetscOptionsReal("-Tfinal","Final time","",user.T,&user.T,NULL));
  CHKERRQ(PetscOptionsInt("-ntests","Number of times to repeat TaoSolve","",ntests,&ntests,NULL));
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  user.m = user.mx*user.mx*user.nt; /*  number of constraints */
  user.n = user.mx*user.mx*3*user.nt; /*  number of variables */
  user.ht = user.T/user.nt; /*  Time step */
  user.gamma = user.T*user.ht / (user.mx*user.mx);

  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&user.u));
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&user.y));
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&user.c));
  CHKERRQ(VecSetSizes(user.u,PETSC_DECIDE,user.n-user.m));
  CHKERRQ(VecSetSizes(user.y,PETSC_DECIDE,user.m));
  CHKERRQ(VecSetSizes(user.c,PETSC_DECIDE,user.m));
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
  CHKERRQ(HyperbolicInitialize(&user));

  CHKERRQ(Gather(x,user.y,user.state_scatter,user.u,user.design_scatter));
  CHKERRQ(VecDuplicate(x,&x0));
  CHKERRQ(VecCopy(x,x0));

  /* Set solution vector with an initial guess */
  CHKERRQ(TaoSetSolution(tao,x));
  CHKERRQ(TaoSetObjective(tao, FormFunction, &user));
  CHKERRQ(TaoSetGradient(tao, NULL, FormGradient, &user));
  CHKERRQ(TaoSetConstraintsRoutine(tao, user.c, FormConstraints, &user));
  CHKERRQ(TaoSetJacobianStateRoutine(tao, user.Js, user.Js, user.JsInv, FormJacobianState, &user));
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
  CHKERRQ(HyperbolicDestroy(&user));
  ierr = PetscFinalize();
  return ierr;
}
/* ------------------------------------------------------------------- */
/*
   dwork = Qy - d
   lwork = L*(u-ur).^2
   f = 1/2 * (dwork.dork + alpha*y.lwork)
*/
PetscErrorCode FormFunction(Tao tao,Vec X,PetscReal *f,void *ptr)
{
  PetscReal      d1=0,d2=0;
  AppCtx         *user = (AppCtx*)ptr;

  PetscFunctionBegin;
  CHKERRQ(Scatter(X,user->y,user->state_scatter,user->u,user->design_scatter));
  CHKERRQ(MatMult(user->Q,user->y,user->dwork));
  CHKERRQ(VecAXPY(user->dwork,-1.0,user->d));
  CHKERRQ(VecDot(user->dwork,user->dwork,&d1));

  CHKERRQ(VecWAXPY(user->uwork,-1.0,user->ur,user->u));
  CHKERRQ(VecPointwiseMult(user->uwork,user->uwork,user->uwork));
  CHKERRQ(MatMult(user->L,user->uwork,user->lwork));
  CHKERRQ(VecDot(user->y,user->lwork,&d2));
  *f = 0.5 * (d1 + user->alpha*d2);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
/*
    state: g_s = Q' *(Qy - d) + 0.5*alpha*L*(u-ur).^2
    design: g_d = alpha*(L'y).*(u-ur)
*/
PetscErrorCode FormGradient(Tao tao,Vec X,Vec G,void *ptr)
{
  AppCtx         *user = (AppCtx*)ptr;

  PetscFunctionBegin;
  CHKERRQ(Scatter(X,user->y,user->state_scatter,user->u,user->design_scatter));
  CHKERRQ(MatMult(user->Q,user->y,user->dwork));
  CHKERRQ(VecAXPY(user->dwork,-1.0,user->d));

  CHKERRQ(MatMult(user->QT,user->dwork,user->ywork));

  CHKERRQ(MatMult(user->LT,user->y,user->uwork));
  CHKERRQ(VecWAXPY(user->vwork,-1.0,user->ur,user->u));
  CHKERRQ(VecPointwiseMult(user->uwork,user->vwork,user->uwork));
  CHKERRQ(VecScale(user->uwork,user->alpha));

  CHKERRQ(VecPointwiseMult(user->vwork,user->vwork,user->vwork));
  CHKERRQ(MatMult(user->L,user->vwork,user->lwork));
  CHKERRQ(VecAXPY(user->ywork,0.5*user->alpha,user->lwork));

  CHKERRQ(Gather(G,user->ywork,user->state_scatter,user->uwork,user->design_scatter));
  PetscFunctionReturn(0);
}

PetscErrorCode FormFunctionGradient(Tao tao, Vec X, PetscReal *f, Vec G, void *ptr)
{
  PetscReal      d1,d2;
  AppCtx         *user = (AppCtx*)ptr;

  PetscFunctionBegin;
  CHKERRQ(Scatter(X,user->y,user->state_scatter,user->u,user->design_scatter));
  CHKERRQ(MatMult(user->Q,user->y,user->dwork));
  CHKERRQ(VecAXPY(user->dwork,-1.0,user->d));

  CHKERRQ(MatMult(user->QT,user->dwork,user->ywork));

  CHKERRQ(VecDot(user->dwork,user->dwork,&d1));

  CHKERRQ(MatMult(user->LT,user->y,user->uwork));
  CHKERRQ(VecWAXPY(user->vwork,-1.0,user->ur,user->u));
  CHKERRQ(VecPointwiseMult(user->uwork,user->vwork,user->uwork));
  CHKERRQ(VecScale(user->uwork,user->alpha));

  CHKERRQ(VecPointwiseMult(user->vwork,user->vwork,user->vwork));
  CHKERRQ(MatMult(user->L,user->vwork,user->lwork));
  CHKERRQ(VecAXPY(user->ywork,0.5*user->alpha,user->lwork));

  CHKERRQ(VecDot(user->y,user->lwork,&d2));

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
  PetscInt       i;
  AppCtx         *user = (AppCtx*)ptr;

  PetscFunctionBegin;
  CHKERRQ(Scatter(X,user->y,user->state_scatter,user->u,user->design_scatter));
  CHKERRQ(Scatter_yi(user->u,user->ui,user->ui_scatter,user->nt));
  CHKERRQ(Scatter_uxi_uyi(user->u,user->uxi,user->uxi_scatter,user->uyi,user->uyi_scatter,user->nt));
  for (i=0; i<user->nt; i++) {
    CHKERRQ(MatCopy(user->Divxy[0],user->C[i],SUBSET_NONZERO_PATTERN));
    CHKERRQ(MatCopy(user->Divxy[1],user->Cwork[i],SAME_NONZERO_PATTERN));

    CHKERRQ(MatDiagonalScale(user->C[i],NULL,user->uxi[i]));
    CHKERRQ(MatDiagonalScale(user->Cwork[i],NULL,user->uyi[i]));
    CHKERRQ(MatAXPY(user->C[i],1.0,user->Cwork[i],SUBSET_NONZERO_PATTERN));
    CHKERRQ(MatScale(user->C[i],user->ht));
    CHKERRQ(MatShift(user->C[i],1.0));
  }
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
  CHKERRQ(Scatter_yi(X,user->yi,user->yi_scatter,user->nt));
  user->block_index = 0;
  CHKERRQ(MatMult(user->JsBlock,user->yi[0],user->yiwork[0]));

  for (i=1; i<user->nt; i++) {
    user->block_index = i;
    CHKERRQ(MatMult(user->JsBlock,user->yi[i],user->yiwork[i]));
    CHKERRQ(MatMult(user->M,user->yi[i-1],user->ziwork[i-1]));
    CHKERRQ(VecAXPY(user->yiwork[i],-1.0,user->ziwork[i-1]));
  }
  CHKERRQ(Gather_yi(Y,user->yiwork,user->yi_scatter,user->nt));
  PetscFunctionReturn(0);
}

PetscErrorCode StateMatMultTranspose(Mat J_shell, Vec X, Vec Y)
{
  PetscInt       i;
  AppCtx         *user;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(J_shell,&user));
  CHKERRQ(Scatter_yi(X,user->yi,user->yi_scatter,user->nt));

  for (i=0; i<user->nt-1; i++) {
    user->block_index = i;
    CHKERRQ(MatMultTranspose(user->JsBlock,user->yi[i],user->yiwork[i]));
    CHKERRQ(MatMult(user->M,user->yi[i+1],user->ziwork[i+1]));
    CHKERRQ(VecAXPY(user->yiwork[i],-1.0,user->ziwork[i+1]));
  }

  i = user->nt-1;
  user->block_index = i;
  CHKERRQ(MatMultTranspose(user->JsBlock,user->yi[i],user->yiwork[i]));
  CHKERRQ(Gather_yi(Y,user->yiwork,user->yi_scatter,user->nt));
  PetscFunctionReturn(0);
}

PetscErrorCode StateMatBlockMult(Mat J_shell, Vec X, Vec Y)
{
  PetscInt       i;
  AppCtx         *user;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(J_shell,&user));
  i = user->block_index;
  CHKERRQ(VecPointwiseMult(user->uxiwork[i],X,user->uxi[i]));
  CHKERRQ(VecPointwiseMult(user->uyiwork[i],X,user->uyi[i]));
  CHKERRQ(Gather(user->uiwork[i],user->uxiwork[i],user->ux_scatter[i],user->uyiwork[i],user->uy_scatter[i]));
  CHKERRQ(MatMult(user->Div,user->uiwork[i],Y));
  CHKERRQ(VecAYPX(Y,user->ht,X));
  PetscFunctionReturn(0);
}

PetscErrorCode StateMatBlockMultTranspose(Mat J_shell, Vec X, Vec Y)
{
  PetscInt       i;
  AppCtx         *user;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(J_shell,&user));
  i = user->block_index;
  CHKERRQ(MatMult(user->Grad,X,user->uiwork[i]));
  CHKERRQ(Scatter(user->uiwork[i],user->uxiwork[i],user->ux_scatter[i],user->uyiwork[i],user->uy_scatter[i]));
  CHKERRQ(VecPointwiseMult(user->uxiwork[i],user->uxi[i],user->uxiwork[i]));
  CHKERRQ(VecPointwiseMult(user->uyiwork[i],user->uyi[i],user->uyiwork[i]));
  CHKERRQ(VecWAXPY(Y,1.0,user->uxiwork[i],user->uyiwork[i]));
  CHKERRQ(VecAYPX(Y,user->ht,X));
  PetscFunctionReturn(0);
}

PetscErrorCode DesignMatMult(Mat J_shell, Vec X, Vec Y)
{
  PetscInt       i;
  AppCtx         *user;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(J_shell,&user));
  CHKERRQ(Scatter_yi(user->y,user->yi,user->yi_scatter,user->nt));
  CHKERRQ(Scatter_uxi_uyi(X,user->uxiwork,user->uxi_scatter,user->uyiwork,user->uyi_scatter,user->nt));
  for (i=0; i<user->nt; i++) {
    CHKERRQ(VecPointwiseMult(user->uxiwork[i],user->yi[i],user->uxiwork[i]));
    CHKERRQ(VecPointwiseMult(user->uyiwork[i],user->yi[i],user->uyiwork[i]));
    CHKERRQ(Gather(user->uiwork[i],user->uxiwork[i],user->ux_scatter[i],user->uyiwork[i],user->uy_scatter[i]));
    CHKERRQ(MatMult(user->Div,user->uiwork[i],user->ziwork[i]));
    CHKERRQ(VecScale(user->ziwork[i],user->ht));
  }
  CHKERRQ(Gather_yi(Y,user->ziwork,user->yi_scatter,user->nt));
  PetscFunctionReturn(0);
}

PetscErrorCode DesignMatMultTranspose(Mat J_shell, Vec X, Vec Y)
{
  PetscInt       i;
  AppCtx         *user;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(J_shell,&user));
  CHKERRQ(Scatter_yi(user->y,user->yi,user->yi_scatter,user->nt));
  CHKERRQ(Scatter_yi(X,user->yiwork,user->yi_scatter,user->nt));
  for (i=0; i<user->nt; i++) {
    CHKERRQ(MatMult(user->Grad,user->yiwork[i],user->uiwork[i]));
    CHKERRQ(Scatter(user->uiwork[i],user->uxiwork[i],user->ux_scatter[i],user->uyiwork[i],user->uy_scatter[i]));
    CHKERRQ(VecPointwiseMult(user->uxiwork[i],user->yi[i],user->uxiwork[i]));
    CHKERRQ(VecPointwiseMult(user->uyiwork[i],user->yi[i],user->uyiwork[i]));
    CHKERRQ(Gather(user->uiwork[i],user->uxiwork[i],user->ux_scatter[i],user->uyiwork[i],user->uy_scatter[i]));
    CHKERRQ(VecScale(user->uiwork[i],user->ht));
  }
  CHKERRQ(Gather_yi(Y,user->uiwork,user->ui_scatter,user->nt));
  PetscFunctionReturn(0);
}

PetscErrorCode StateMatBlockPrecMult(PC PC_shell, Vec X, Vec Y)
{
  PetscInt       i;
  AppCtx         *user;

  PetscFunctionBegin;
  CHKERRQ(PCShellGetContext(PC_shell,&user));
  i = user->block_index;
  if (user->c_formed) {
    CHKERRQ(MatSOR(user->C[i],X,1.0,(MatSORType)(SOR_ZERO_INITIAL_GUESS | SOR_LOCAL_SYMMETRIC_SWEEP),0.0,1,1,Y));
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not formed");
  PetscFunctionReturn(0);
}

PetscErrorCode StateMatBlockPrecMultTranspose(PC PC_shell, Vec X, Vec Y)
{
  PetscInt       i;
  AppCtx         *user;

  PetscFunctionBegin;
  CHKERRQ(PCShellGetContext(PC_shell,&user));

  i = user->block_index;
  if (user->c_formed) {
    CHKERRQ(MatSOR(user->C[i],X,1.0,(MatSORType)(SOR_ZERO_INITIAL_GUESS | SOR_LOCAL_SYMMETRIC_SWEEP),0.0,1,1,Y));
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not formed");
  PetscFunctionReturn(0);
}

PetscErrorCode StateMatInvMult(Mat J_shell, Vec X, Vec Y)
{
  AppCtx         *user;
  PetscInt       its,i;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(J_shell,&user));

  if (Y == user->ytrue) {
    /* First solve is done using true solution to set up problem */
    CHKERRQ(KSPSetTolerances(user->solver,1e-4,1e-20,PETSC_DEFAULT,PETSC_DEFAULT));
  } else {
    CHKERRQ(KSPSetTolerances(user->solver,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT));
  }
  CHKERRQ(Scatter_yi(X,user->yi,user->yi_scatter,user->nt));
  CHKERRQ(Scatter_yi(Y,user->yiwork,user->yi_scatter,user->nt));
  CHKERRQ(Scatter_uxi_uyi(user->u,user->uxi,user->uxi_scatter,user->uyi,user->uyi_scatter,user->nt));

  user->block_index = 0;
  CHKERRQ(KSPSolve(user->solver,user->yi[0],user->yiwork[0]));

  CHKERRQ(KSPGetIterationNumber(user->solver,&its));
  user->ksp_its = user->ksp_its + its;
  for (i=1; i<user->nt; i++) {
    CHKERRQ(MatMult(user->M,user->yiwork[i-1],user->ziwork[i-1]));
    CHKERRQ(VecAXPY(user->yi[i],1.0,user->ziwork[i-1]));
    user->block_index = i;
    CHKERRQ(KSPSolve(user->solver,user->yi[i],user->yiwork[i]));

    CHKERRQ(KSPGetIterationNumber(user->solver,&its));
    user->ksp_its = user->ksp_its + its;
  }
  CHKERRQ(Gather_yi(Y,user->yiwork,user->yi_scatter,user->nt));
  PetscFunctionReturn(0);
}

PetscErrorCode StateMatInvTransposeMult(Mat J_shell, Vec X, Vec Y)
{
  AppCtx         *user;
  PetscInt       its,i;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(J_shell,&user));

  CHKERRQ(Scatter_yi(X,user->yi,user->yi_scatter,user->nt));
  CHKERRQ(Scatter_yi(Y,user->yiwork,user->yi_scatter,user->nt));
  CHKERRQ(Scatter_uxi_uyi(user->u,user->uxi,user->uxi_scatter,user->uyi,user->uyi_scatter,user->nt));

  i = user->nt - 1;
  user->block_index = i;
  CHKERRQ(KSPSolveTranspose(user->solver,user->yi[i],user->yiwork[i]));

  CHKERRQ(KSPGetIterationNumber(user->solver,&its));
  user->ksp_its = user->ksp_its + its;

  for (i=user->nt-2; i>=0; i--) {
    CHKERRQ(MatMult(user->M,user->yiwork[i+1],user->ziwork[i+1]));
    CHKERRQ(VecAXPY(user->yi[i],1.0,user->ziwork[i+1]));
    user->block_index = i;
    CHKERRQ(KSPSolveTranspose(user->solver,user->yi[i],user->yiwork[i]));

    CHKERRQ(KSPGetIterationNumber(user->solver,&its));
    user->ksp_its = user->ksp_its + its;
  }
  CHKERRQ(Gather_yi(Y,user->yiwork,user->yi_scatter,user->nt));
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
  /* con = Ay - q, A = [C(u1)  0     0     ...   0;
                         -M  C(u2)   0     ...   0;
                          0   -M   C(u3)   ...   0;
                                      ...         ;
                          0    ...      -M C(u_nt)]
     C(u) = eye + ht*Div*[diag(u1); diag(u2)]       */
  PetscInt       i;
  AppCtx         *user = (AppCtx*)ptr;

  PetscFunctionBegin;
  CHKERRQ(Scatter(X,user->y,user->state_scatter,user->u,user->design_scatter));
  CHKERRQ(Scatter_yi(user->y,user->yi,user->yi_scatter,user->nt));
  CHKERRQ(Scatter_uxi_uyi(user->u,user->uxi,user->uxi_scatter,user->uyi,user->uyi_scatter,user->nt));

  user->block_index = 0;
  CHKERRQ(MatMult(user->JsBlock,user->yi[0],user->yiwork[0]));

  for (i=1; i<user->nt; i++) {
    user->block_index = i;
    CHKERRQ(MatMult(user->JsBlock,user->yi[i],user->yiwork[i]));
    CHKERRQ(MatMult(user->M,user->yi[i-1],user->ziwork[i-1]));
    CHKERRQ(VecAXPY(user->yiwork[i],-1.0,user->ziwork[i-1]));
  }

  CHKERRQ(Gather_yi(C,user->yiwork,user->yi_scatter,user->nt));
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

PetscErrorCode Scatter_uxi_uyi(Vec u, Vec *uxi, VecScatter *scatx, Vec *uyi, VecScatter *scaty, PetscInt nt)
{
  PetscInt       i;

  PetscFunctionBegin;
  for (i=0; i<nt; i++) {
    CHKERRQ(VecScatterBegin(scatx[i],u,uxi[i],INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecScatterEnd(scatx[i],u,uxi[i],INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecScatterBegin(scaty[i],u,uyi[i],INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecScatterEnd(scaty[i],u,uyi[i],INSERT_VALUES,SCATTER_FORWARD));
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

PetscErrorCode Gather_uxi_uyi(Vec u, Vec *uxi, VecScatter *scatx, Vec *uyi, VecScatter *scaty, PetscInt nt)
{
  PetscInt       i;

  PetscFunctionBegin;
  for (i=0; i<nt; i++) {
    CHKERRQ(VecScatterBegin(scatx[i],uxi[i],u,INSERT_VALUES,SCATTER_REVERSE));
    CHKERRQ(VecScatterEnd(scatx[i],uxi[i],u,INSERT_VALUES,SCATTER_REVERSE));
    CHKERRQ(VecScatterBegin(scaty[i],uyi[i],u,INSERT_VALUES,SCATTER_REVERSE));
    CHKERRQ(VecScatterEnd(scaty[i],uyi[i],u,INSERT_VALUES,SCATTER_REVERSE));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode Scatter_yi(Vec y, Vec *yi, VecScatter *scat, PetscInt nt)
{
  PetscInt       i;

  PetscFunctionBegin;
  for (i=0; i<nt; i++) {
    CHKERRQ(VecScatterBegin(scat[i],y,yi[i],INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecScatterEnd(scat[i],y,yi[i],INSERT_VALUES,SCATTER_FORWARD));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode Gather_yi(Vec y, Vec *yi, VecScatter *scat, PetscInt nt)
{
  PetscInt       i;

  PetscFunctionBegin;
  for (i=0; i<nt; i++) {
    CHKERRQ(VecScatterBegin(scat[i],yi[i],y,INSERT_VALUES,SCATTER_REVERSE));
    CHKERRQ(VecScatterEnd(scat[i],yi[i],y,INSERT_VALUES,SCATTER_REVERSE));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode HyperbolicInitialize(AppCtx *user)
{
  PetscInt       n,i,j,linear_index,istart,iend,iblock,lo,hi;
  Vec            XX,YY,XXwork,YYwork,yi,uxi,ui,bc;
  PetscReal      h,sum;
  PetscScalar    hinv,neg_hinv,quarter=0.25,one=1.0,half_hinv,neg_half_hinv;
  PetscScalar    vx,vy,zero=0.0;
  IS             is_from_y,is_to_yi,is_from_u,is_to_uxi,is_to_uyi;

  PetscFunctionBegin;
  user->jformed = PETSC_FALSE;
  user->c_formed = PETSC_FALSE;

  user->ksp_its = 0;
  user->ksp_its_initial = 0;

  n = user->mx * user->mx;

  h = 1.0/user->mx;
  hinv = user->mx;
  neg_hinv = -hinv;
  half_hinv = hinv / 2.0;
  neg_half_hinv = neg_hinv / 2.0;

  /* Generate Grad matrix */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&user->Grad));
  CHKERRQ(MatSetSizes(user->Grad,PETSC_DECIDE,PETSC_DECIDE,2*n,n));
  CHKERRQ(MatSetFromOptions(user->Grad));
  CHKERRQ(MatMPIAIJSetPreallocation(user->Grad,3,NULL,3,NULL));
  CHKERRQ(MatSeqAIJSetPreallocation(user->Grad,3,NULL));
  CHKERRQ(MatGetOwnershipRange(user->Grad,&istart,&iend));

  for (i=istart; i<iend; i++) {
    if (i<n) {
      iblock = i / user->mx;
      j = iblock*user->mx + ((i+user->mx-1) % user->mx);
      CHKERRQ(MatSetValues(user->Grad,1,&i,1,&j,&half_hinv,INSERT_VALUES));
      j = iblock*user->mx + ((i+1) % user->mx);
      CHKERRQ(MatSetValues(user->Grad,1,&i,1,&j,&neg_half_hinv,INSERT_VALUES));
    }
    if (i>=n) {
      j = (i - user->mx) % n;
      CHKERRQ(MatSetValues(user->Grad,1,&i,1,&j,&half_hinv,INSERT_VALUES));
      j = (j + 2*user->mx) % n;
      CHKERRQ(MatSetValues(user->Grad,1,&i,1,&j,&neg_half_hinv,INSERT_VALUES));
    }
  }

  CHKERRQ(MatAssemblyBegin(user->Grad,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(user->Grad,MAT_FINAL_ASSEMBLY));

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&user->Gradxy[0]));
  CHKERRQ(MatSetSizes(user->Gradxy[0],PETSC_DECIDE,PETSC_DECIDE,n,n));
  CHKERRQ(MatSetFromOptions(user->Gradxy[0]));
  CHKERRQ(MatMPIAIJSetPreallocation(user->Gradxy[0],3,NULL,3,NULL));
  CHKERRQ(MatSeqAIJSetPreallocation(user->Gradxy[0],3,NULL));
  CHKERRQ(MatGetOwnershipRange(user->Gradxy[0],&istart,&iend));

  for (i=istart; i<iend; i++) {
    iblock = i / user->mx;
    j = iblock*user->mx + ((i+user->mx-1) % user->mx);
    CHKERRQ(MatSetValues(user->Gradxy[0],1,&i,1,&j,&half_hinv,INSERT_VALUES));
    j = iblock*user->mx + ((i+1) % user->mx);
    CHKERRQ(MatSetValues(user->Gradxy[0],1,&i,1,&j,&neg_half_hinv,INSERT_VALUES));
    CHKERRQ(MatSetValues(user->Gradxy[0],1,&i,1,&i,&zero,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(user->Gradxy[0],MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(user->Gradxy[0],MAT_FINAL_ASSEMBLY));

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&user->Gradxy[1]));
  CHKERRQ(MatSetSizes(user->Gradxy[1],PETSC_DECIDE,PETSC_DECIDE,n,n));
  CHKERRQ(MatSetFromOptions(user->Gradxy[1]));
  CHKERRQ(MatMPIAIJSetPreallocation(user->Gradxy[1],3,NULL,3,NULL));
  CHKERRQ(MatSeqAIJSetPreallocation(user->Gradxy[1],3,NULL));
  CHKERRQ(MatGetOwnershipRange(user->Gradxy[1],&istart,&iend));

  for (i=istart; i<iend; i++) {
    j = (i + n - user->mx) % n;
    CHKERRQ(MatSetValues(user->Gradxy[1],1,&i,1,&j,&half_hinv,INSERT_VALUES));
    j = (j + 2*user->mx) % n;
    CHKERRQ(MatSetValues(user->Gradxy[1],1,&i,1,&j,&neg_half_hinv,INSERT_VALUES));
    CHKERRQ(MatSetValues(user->Gradxy[1],1,&i,1,&i,&zero,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(user->Gradxy[1],MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(user->Gradxy[1],MAT_FINAL_ASSEMBLY));

  /* Generate Div matrix */
  CHKERRQ(MatTranspose(user->Grad,MAT_INITIAL_MATRIX,&user->Div));
  CHKERRQ(MatTranspose(user->Gradxy[0],MAT_INITIAL_MATRIX,&user->Divxy[0]));
  CHKERRQ(MatTranspose(user->Gradxy[1],MAT_INITIAL_MATRIX,&user->Divxy[1]));

  /* Off-diagonal averaging matrix */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&user->M));
  CHKERRQ(MatSetSizes(user->M,PETSC_DECIDE,PETSC_DECIDE,n,n));
  CHKERRQ(MatSetFromOptions(user->M));
  CHKERRQ(MatMPIAIJSetPreallocation(user->M,4,NULL,4,NULL));
  CHKERRQ(MatSeqAIJSetPreallocation(user->M,4,NULL));
  CHKERRQ(MatGetOwnershipRange(user->M,&istart,&iend));

  for (i=istart; i<iend; i++) {
    /* kron(Id,Av) */
    iblock = i / user->mx;
    j = iblock*user->mx + ((i+user->mx-1) % user->mx);
    CHKERRQ(MatSetValues(user->M,1,&i,1,&j,&quarter,INSERT_VALUES));
    j = iblock*user->mx + ((i+1) % user->mx);
    CHKERRQ(MatSetValues(user->M,1,&i,1,&j,&quarter,INSERT_VALUES));

    /* kron(Av,Id) */
    j = (i + user->mx) % n;
    CHKERRQ(MatSetValues(user->M,1,&i,1,&j,&quarter,INSERT_VALUES));
    j = (i + n - user->mx) % n;
    CHKERRQ(MatSetValues(user->M,1,&i,1,&j,&quarter,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(user->M,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(user->M,MAT_FINAL_ASSEMBLY));

  /* Generate 2D grid */
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&XX));
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&user->q));
  CHKERRQ(VecSetSizes(XX,PETSC_DECIDE,n));
  CHKERRQ(VecSetSizes(user->q,PETSC_DECIDE,n*user->nt));
  CHKERRQ(VecSetFromOptions(XX));
  CHKERRQ(VecSetFromOptions(user->q));

  CHKERRQ(VecDuplicate(XX,&YY));
  CHKERRQ(VecDuplicate(XX,&XXwork));
  CHKERRQ(VecDuplicate(XX,&YYwork));
  CHKERRQ(VecDuplicate(XX,&user->d));
  CHKERRQ(VecDuplicate(XX,&user->dwork));

  CHKERRQ(VecGetOwnershipRange(XX,&istart,&iend));
  for (linear_index=istart; linear_index<iend; linear_index++) {
    i = linear_index % user->mx;
    j = (linear_index-i)/user->mx;
    vx = h*(i+0.5);
    vy = h*(j+0.5);
    CHKERRQ(VecSetValues(XX,1,&linear_index,&vx,INSERT_VALUES));
    CHKERRQ(VecSetValues(YY,1,&linear_index,&vy,INSERT_VALUES));
  }

  CHKERRQ(VecAssemblyBegin(XX));
  CHKERRQ(VecAssemblyEnd(XX));
  CHKERRQ(VecAssemblyBegin(YY));
  CHKERRQ(VecAssemblyEnd(YY));

  /* Compute final density function yT
     yT = 1.0 + exp(-30*((x-0.25)^2+(y-0.25)^2)) + exp(-30*((x-0.75)^2+(y-0.75)^2))
     yT = yT / (h^2*sum(yT)) */
  CHKERRQ(VecCopy(XX,XXwork));
  CHKERRQ(VecCopy(YY,YYwork));

  CHKERRQ(VecShift(XXwork,-0.25));
  CHKERRQ(VecShift(YYwork,-0.25));

  CHKERRQ(VecPointwiseMult(XXwork,XXwork,XXwork));
  CHKERRQ(VecPointwiseMult(YYwork,YYwork,YYwork));

  CHKERRQ(VecCopy(XXwork,user->dwork));
  CHKERRQ(VecAXPY(user->dwork,1.0,YYwork));
  CHKERRQ(VecScale(user->dwork,-30.0));
  CHKERRQ(VecExp(user->dwork));
  CHKERRQ(VecCopy(user->dwork,user->d));

  CHKERRQ(VecCopy(XX,XXwork));
  CHKERRQ(VecCopy(YY,YYwork));

  CHKERRQ(VecShift(XXwork,-0.75));
  CHKERRQ(VecShift(YYwork,-0.75));

  CHKERRQ(VecPointwiseMult(XXwork,XXwork,XXwork));
  CHKERRQ(VecPointwiseMult(YYwork,YYwork,YYwork));

  CHKERRQ(VecCopy(XXwork,user->dwork));
  CHKERRQ(VecAXPY(user->dwork,1.0,YYwork));
  CHKERRQ(VecScale(user->dwork,-30.0));
  CHKERRQ(VecExp(user->dwork));

  CHKERRQ(VecAXPY(user->d,1.0,user->dwork));
  CHKERRQ(VecShift(user->d,1.0));
  CHKERRQ(VecSum(user->d,&sum));
  CHKERRQ(VecScale(user->d,1.0/(h*h*sum)));

  /* Initial conditions of forward problem */
  CHKERRQ(VecDuplicate(XX,&bc));
  CHKERRQ(VecCopy(XX,XXwork));
  CHKERRQ(VecCopy(YY,YYwork));

  CHKERRQ(VecShift(XXwork,-0.5));
  CHKERRQ(VecShift(YYwork,-0.5));

  CHKERRQ(VecPointwiseMult(XXwork,XXwork,XXwork));
  CHKERRQ(VecPointwiseMult(YYwork,YYwork,YYwork));

  CHKERRQ(VecWAXPY(bc,1.0,XXwork,YYwork));
  CHKERRQ(VecScale(bc,-50.0));
  CHKERRQ(VecExp(bc));
  CHKERRQ(VecShift(bc,1.0));
  CHKERRQ(VecSum(bc,&sum));
  CHKERRQ(VecScale(bc,1.0/(h*h*sum)));

  /* Create scatter from y to y_1,y_2,...,y_nt */
  /*  TODO: Reorder for better parallelism. (This will require reordering Q and L as well.) */
  CHKERRQ(PetscMalloc1(user->nt*user->mx*user->mx,&user->yi_scatter));
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&yi));
  CHKERRQ(VecSetSizes(yi,PETSC_DECIDE,user->mx*user->mx));
  CHKERRQ(VecSetFromOptions(yi));
  CHKERRQ(VecDuplicateVecs(yi,user->nt,&user->yi));
  CHKERRQ(VecDuplicateVecs(yi,user->nt,&user->yiwork));
  CHKERRQ(VecDuplicateVecs(yi,user->nt,&user->ziwork));
  for (i=0; i<user->nt; i++) {
    CHKERRQ(VecGetOwnershipRange(user->yi[i],&lo,&hi));
    CHKERRQ(ISCreateStride(PETSC_COMM_SELF,hi-lo,lo,1,&is_to_yi));
    CHKERRQ(ISCreateStride(PETSC_COMM_SELF,hi-lo,lo+i*user->mx*user->mx,1,&is_from_y));
    CHKERRQ(VecScatterCreate(user->y,is_from_y,user->yi[i],is_to_yi,&user->yi_scatter[i]));
    CHKERRQ(ISDestroy(&is_to_yi));
    CHKERRQ(ISDestroy(&is_from_y));
  }

  /* Create scatter from u to ux_1,uy_1,ux_2,uy_2,...,ux_nt,uy_nt */
  /*  TODO: reorder for better parallelism */
  CHKERRQ(PetscMalloc1(user->nt*user->mx*user->mx,&user->uxi_scatter));
  CHKERRQ(PetscMalloc1(user->nt*user->mx*user->mx,&user->uyi_scatter));
  CHKERRQ(PetscMalloc1(user->nt*user->mx*user->mx,&user->ux_scatter));
  CHKERRQ(PetscMalloc1(user->nt*user->mx*user->mx,&user->uy_scatter));
  CHKERRQ(PetscMalloc1(2*user->nt*user->mx*user->mx,&user->ui_scatter));
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&uxi));
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&ui));
  CHKERRQ(VecSetSizes(uxi,PETSC_DECIDE,user->mx*user->mx));
  CHKERRQ(VecSetSizes(ui,PETSC_DECIDE,2*user->mx*user->mx));
  CHKERRQ(VecSetFromOptions(uxi));
  CHKERRQ(VecSetFromOptions(ui));
  CHKERRQ(VecDuplicateVecs(uxi,user->nt,&user->uxi));
  CHKERRQ(VecDuplicateVecs(uxi,user->nt,&user->uyi));
  CHKERRQ(VecDuplicateVecs(uxi,user->nt,&user->uxiwork));
  CHKERRQ(VecDuplicateVecs(uxi,user->nt,&user->uyiwork));
  CHKERRQ(VecDuplicateVecs(ui,user->nt,&user->ui));
  CHKERRQ(VecDuplicateVecs(ui,user->nt,&user->uiwork));
  for (i=0; i<user->nt; i++) {
    CHKERRQ(VecGetOwnershipRange(user->uxi[i],&lo,&hi));
    CHKERRQ(ISCreateStride(PETSC_COMM_SELF,hi-lo,lo,1,&is_to_uxi));
    CHKERRQ(ISCreateStride(PETSC_COMM_SELF,hi-lo,lo+2*i*user->mx*user->mx,1,&is_from_u));
    CHKERRQ(VecScatterCreate(user->u,is_from_u,user->uxi[i],is_to_uxi,&user->uxi_scatter[i]));

    CHKERRQ(ISDestroy(&is_to_uxi));
    CHKERRQ(ISDestroy(&is_from_u));

    CHKERRQ(VecGetOwnershipRange(user->uyi[i],&lo,&hi));
    CHKERRQ(ISCreateStride(PETSC_COMM_SELF,hi-lo,lo,1,&is_to_uyi));
    CHKERRQ(ISCreateStride(PETSC_COMM_SELF,hi-lo,lo+(2*i+1)*user->mx*user->mx,1,&is_from_u));
    CHKERRQ(VecScatterCreate(user->u,is_from_u,user->uyi[i],is_to_uyi,&user->uyi_scatter[i]));

    CHKERRQ(ISDestroy(&is_to_uyi));
    CHKERRQ(ISDestroy(&is_from_u));

    CHKERRQ(VecGetOwnershipRange(user->uxi[i],&lo,&hi));
    CHKERRQ(ISCreateStride(PETSC_COMM_SELF,hi-lo,lo,1,&is_to_uxi));
    CHKERRQ(ISCreateStride(PETSC_COMM_SELF,hi-lo,lo,1,&is_from_u));
    CHKERRQ(VecScatterCreate(user->ui[i],is_from_u,user->uxi[i],is_to_uxi,&user->ux_scatter[i]));

    CHKERRQ(ISDestroy(&is_to_uxi));
    CHKERRQ(ISDestroy(&is_from_u));

    CHKERRQ(VecGetOwnershipRange(user->uyi[i],&lo,&hi));
    CHKERRQ(ISCreateStride(PETSC_COMM_SELF,hi-lo,lo,1,&is_to_uyi));
    CHKERRQ(ISCreateStride(PETSC_COMM_SELF,hi-lo,lo+user->mx*user->mx,1,&is_from_u));
    CHKERRQ(VecScatterCreate(user->ui[i],is_from_u,user->uyi[i],is_to_uyi,&user->uy_scatter[i]));

    CHKERRQ(ISDestroy(&is_to_uyi));
    CHKERRQ(ISDestroy(&is_from_u));

    CHKERRQ(VecGetOwnershipRange(user->ui[i],&lo,&hi));
    CHKERRQ(ISCreateStride(PETSC_COMM_SELF,hi-lo,lo,1,&is_to_uxi));
    CHKERRQ(ISCreateStride(PETSC_COMM_SELF,hi-lo,lo+2*i*user->mx*user->mx,1,&is_from_u));
    CHKERRQ(VecScatterCreate(user->u,is_from_u,user->ui[i],is_to_uxi,&user->ui_scatter[i]));

    CHKERRQ(ISDestroy(&is_to_uxi));
    CHKERRQ(ISDestroy(&is_from_u));
  }

  /* RHS of forward problem */
  CHKERRQ(MatMult(user->M,bc,user->yiwork[0]));
  for (i=1; i<user->nt; i++) {
    CHKERRQ(VecSet(user->yiwork[i],0.0));
  }
  CHKERRQ(Gather_yi(user->q,user->yiwork,user->yi_scatter,user->nt));

  /* Compute true velocity field utrue */
  CHKERRQ(VecDuplicate(user->u,&user->utrue));
  for (i=0; i<user->nt; i++) {
    CHKERRQ(VecCopy(YY,user->uxi[i]));
    CHKERRQ(VecScale(user->uxi[i],150.0*i*user->ht));
    CHKERRQ(VecCopy(XX,user->uyi[i]));
    CHKERRQ(VecShift(user->uyi[i],-10.0));
    CHKERRQ(VecScale(user->uyi[i],15.0*i*user->ht));
  }
  CHKERRQ(Gather_uxi_uyi(user->utrue,user->uxi,user->uxi_scatter,user->uyi,user->uyi_scatter,user->nt));

  /* Initial guess and reference model */
  CHKERRQ(VecDuplicate(user->utrue,&user->ur));
  for (i=0; i<user->nt; i++) {
    CHKERRQ(VecCopy(XX,user->uxi[i]));
    CHKERRQ(VecShift(user->uxi[i],i*user->ht));
    CHKERRQ(VecCopy(YY,user->uyi[i]));
    CHKERRQ(VecShift(user->uyi[i],-i*user->ht));
  }
  CHKERRQ(Gather_uxi_uyi(user->ur,user->uxi,user->uxi_scatter,user->uyi,user->uyi_scatter,user->nt));

  /* Generate regularization matrix L */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&user->LT));
  CHKERRQ(MatSetSizes(user->LT,PETSC_DECIDE,PETSC_DECIDE,2*n*user->nt,n*user->nt));
  CHKERRQ(MatSetFromOptions(user->LT));
  CHKERRQ(MatMPIAIJSetPreallocation(user->LT,1,NULL,1,NULL));
  CHKERRQ(MatSeqAIJSetPreallocation(user->LT,1,NULL));
  CHKERRQ(MatGetOwnershipRange(user->LT,&istart,&iend));

  for (i=istart; i<iend; i++) {
    iblock = (i+n) / (2*n);
    j = i - iblock*n;
    CHKERRQ(MatSetValues(user->LT,1,&i,1,&j,&user->gamma,INSERT_VALUES));
  }

  CHKERRQ(MatAssemblyBegin(user->LT,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(user->LT,MAT_FINAL_ASSEMBLY));

  CHKERRQ(MatTranspose(user->LT,MAT_INITIAL_MATRIX,&user->L));

  /* Build work vectors and matrices */
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&user->lwork));
  CHKERRQ(VecSetType(user->lwork,VECMPI));
  CHKERRQ(VecSetSizes(user->lwork,PETSC_DECIDE,user->m));
  CHKERRQ(VecSetFromOptions(user->lwork));

  CHKERRQ(MatDuplicate(user->Div,MAT_SHARE_NONZERO_PATTERN,&user->Divwork));

  CHKERRQ(VecDuplicate(user->y,&user->ywork));
  CHKERRQ(VecDuplicate(user->u,&user->uwork));
  CHKERRQ(VecDuplicate(user->u,&user->vwork));
  CHKERRQ(VecDuplicate(user->u,&user->js_diag));
  CHKERRQ(VecDuplicate(user->c,&user->cwork));

  /* Create matrix-free shell user->Js for computing A*x */
  CHKERRQ(MatCreateShell(PETSC_COMM_WORLD,PETSC_DETERMINE,PETSC_DETERMINE,user->m,user->m,user,&user->Js));
  CHKERRQ(MatShellSetOperation(user->Js,MATOP_MULT,(void(*)(void))StateMatMult));
  CHKERRQ(MatShellSetOperation(user->Js,MATOP_DUPLICATE,(void(*)(void))StateMatDuplicate));
  CHKERRQ(MatShellSetOperation(user->Js,MATOP_MULT_TRANSPOSE,(void(*)(void))StateMatMultTranspose));
  CHKERRQ(MatShellSetOperation(user->Js,MATOP_GET_DIAGONAL,(void(*)(void))StateMatGetDiagonal));

  /* Diagonal blocks of user->Js */
  CHKERRQ(MatCreateShell(PETSC_COMM_WORLD,PETSC_DETERMINE,PETSC_DETERMINE,n,n,user,&user->JsBlock));
  CHKERRQ(MatShellSetOperation(user->JsBlock,MATOP_MULT,(void(*)(void))StateMatBlockMult));
  CHKERRQ(MatShellSetOperation(user->JsBlock,MATOP_MULT_TRANSPOSE,(void(*)(void))StateMatBlockMultTranspose));

  /* Create a matrix-free shell user->JsBlockPrec for computing (U+D)\D*(L+D)\x, where JsBlock = L+D+U,
     D is diagonal, L is strictly lower triangular, and U is strictly upper triangular.
     This is an SOR preconditioner for user->JsBlock. */
  CHKERRQ(MatCreateShell(PETSC_COMM_WORLD,PETSC_DETERMINE,PETSC_DETERMINE,n,n,user,&user->JsBlockPrec));
  CHKERRQ(MatShellSetOperation(user->JsBlockPrec,MATOP_MULT,(void(*)(void))StateMatBlockPrecMult));
  CHKERRQ(MatShellSetOperation(user->JsBlockPrec,MATOP_MULT_TRANSPOSE,(void(*)(void))StateMatBlockPrecMultTranspose));

  /* Create a matrix-free shell user->Jd for computing B*x */
  CHKERRQ(MatCreateShell(PETSC_COMM_WORLD,PETSC_DETERMINE,PETSC_DETERMINE,user->m,user->n-user->m,user,&user->Jd));
  CHKERRQ(MatShellSetOperation(user->Jd,MATOP_MULT,(void(*)(void))DesignMatMult));
  CHKERRQ(MatShellSetOperation(user->Jd,MATOP_MULT_TRANSPOSE,(void(*)(void))DesignMatMultTranspose));

  /* User-defined routines for computing user->Js\x and user->Js^T\x*/
  CHKERRQ(MatCreateShell(PETSC_COMM_WORLD,PETSC_DETERMINE,PETSC_DETERMINE,user->m,user->m,user,&user->JsInv));
  CHKERRQ(MatShellSetOperation(user->JsInv,MATOP_MULT,(void(*)(void))StateMatInvMult));
  CHKERRQ(MatShellSetOperation(user->JsInv,MATOP_MULT_TRANSPOSE,(void(*)(void))StateMatInvTransposeMult));

  /* Build matrices for SOR preconditioner */
  CHKERRQ(Scatter_uxi_uyi(user->u,user->uxi,user->uxi_scatter,user->uyi,user->uyi_scatter,user->nt));
  CHKERRQ(PetscMalloc1(5*n,&user->C));
  CHKERRQ(PetscMalloc1(2*n,&user->Cwork));
  for (i=0; i<user->nt; i++) {
    CHKERRQ(MatDuplicate(user->Divxy[0],MAT_COPY_VALUES,&user->C[i]));
    CHKERRQ(MatDuplicate(user->Divxy[1],MAT_COPY_VALUES,&user->Cwork[i]));

    CHKERRQ(MatDiagonalScale(user->C[i],NULL,user->uxi[i]));
    CHKERRQ(MatDiagonalScale(user->Cwork[i],NULL,user->uyi[i]));
    CHKERRQ(MatAXPY(user->C[i],1.0,user->Cwork[i],DIFFERENT_NONZERO_PATTERN));
    CHKERRQ(MatScale(user->C[i],user->ht));
    CHKERRQ(MatShift(user->C[i],1.0));
  }

  /* Solver options and tolerances */
  CHKERRQ(KSPCreate(PETSC_COMM_WORLD,&user->solver));
  CHKERRQ(KSPSetType(user->solver,KSPGMRES));
  CHKERRQ(KSPSetOperators(user->solver,user->JsBlock,user->JsBlockPrec));
  CHKERRQ(KSPSetTolerances(user->solver,1e-4,1e-20,1e3,500));
  /* CHKERRQ(KSPSetTolerances(user->solver,1e-8,1e-16,1e3,500)); */
  CHKERRQ(KSPGetPC(user->solver,&user->prec));
  CHKERRQ(PCSetType(user->prec,PCSHELL));

  CHKERRQ(PCShellSetApply(user->prec,StateMatBlockPrecMult));
  CHKERRQ(PCShellSetApplyTranspose(user->prec,StateMatBlockPrecMultTranspose));
  CHKERRQ(PCShellSetContext(user->prec,user));

  /* Compute true state function yt given ut */
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&user->ytrue));
  CHKERRQ(VecSetSizes(user->ytrue,PETSC_DECIDE,n*user->nt));
  CHKERRQ(VecSetFromOptions(user->ytrue));
  user->c_formed = PETSC_TRUE;
  CHKERRQ(VecCopy(user->utrue,user->u)); /*  Set u=utrue temporarily for StateMatInv */
  CHKERRQ(VecSet(user->ytrue,0.0)); /*  Initial guess */
  CHKERRQ(StateMatInvMult(user->Js,user->q,user->ytrue));
  CHKERRQ(VecCopy(user->ur,user->u)); /*  Reset u=ur */

  /* Initial guess y0 for state given u0 */
  CHKERRQ(StateMatInvMult(user->Js,user->q,user->y));

  /* Data discretization */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&user->Q));
  CHKERRQ(MatSetSizes(user->Q,PETSC_DECIDE,PETSC_DECIDE,user->mx*user->mx,user->m));
  CHKERRQ(MatSetFromOptions(user->Q));
  CHKERRQ(MatMPIAIJSetPreallocation(user->Q,0,NULL,1,NULL));
  CHKERRQ(MatSeqAIJSetPreallocation(user->Q,1,NULL));

  CHKERRQ(MatGetOwnershipRange(user->Q,&istart,&iend));

  for (i=istart; i<iend; i++) {
    j = i + user->m - user->mx*user->mx;
    CHKERRQ(MatSetValues(user->Q,1,&i,1,&j,&one,INSERT_VALUES));
  }

  CHKERRQ(MatAssemblyBegin(user->Q,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(user->Q,MAT_FINAL_ASSEMBLY));

  CHKERRQ(MatTranspose(user->Q,MAT_INITIAL_MATRIX,&user->QT));

  CHKERRQ(VecDestroy(&XX));
  CHKERRQ(VecDestroy(&YY));
  CHKERRQ(VecDestroy(&XXwork));
  CHKERRQ(VecDestroy(&YYwork));
  CHKERRQ(VecDestroy(&yi));
  CHKERRQ(VecDestroy(&uxi));
  CHKERRQ(VecDestroy(&ui));
  CHKERRQ(VecDestroy(&bc));

  /* Now that initial conditions have been set, let the user pass tolerance options to the KSP solver */
  CHKERRQ(KSPSetFromOptions(user->solver));
  PetscFunctionReturn(0);
}

PetscErrorCode HyperbolicDestroy(AppCtx *user)
{
  PetscInt       i;

  PetscFunctionBegin;
  CHKERRQ(MatDestroy(&user->Q));
  CHKERRQ(MatDestroy(&user->QT));
  CHKERRQ(MatDestroy(&user->Div));
  CHKERRQ(MatDestroy(&user->Divwork));
  CHKERRQ(MatDestroy(&user->Grad));
  CHKERRQ(MatDestroy(&user->L));
  CHKERRQ(MatDestroy(&user->LT));
  CHKERRQ(KSPDestroy(&user->solver));
  CHKERRQ(MatDestroy(&user->Js));
  CHKERRQ(MatDestroy(&user->Jd));
  CHKERRQ(MatDestroy(&user->JsBlockPrec));
  CHKERRQ(MatDestroy(&user->JsInv));
  CHKERRQ(MatDestroy(&user->JsBlock));
  CHKERRQ(MatDestroy(&user->Divxy[0]));
  CHKERRQ(MatDestroy(&user->Divxy[1]));
  CHKERRQ(MatDestroy(&user->Gradxy[0]));
  CHKERRQ(MatDestroy(&user->Gradxy[1]));
  CHKERRQ(MatDestroy(&user->M));
  for (i=0; i<user->nt; i++) {
    CHKERRQ(MatDestroy(&user->C[i]));
    CHKERRQ(MatDestroy(&user->Cwork[i]));
  }
  CHKERRQ(PetscFree(user->C));
  CHKERRQ(PetscFree(user->Cwork));
  CHKERRQ(VecDestroy(&user->u));
  CHKERRQ(VecDestroy(&user->uwork));
  CHKERRQ(VecDestroy(&user->vwork));
  CHKERRQ(VecDestroy(&user->utrue));
  CHKERRQ(VecDestroy(&user->y));
  CHKERRQ(VecDestroy(&user->ywork));
  CHKERRQ(VecDestroy(&user->ytrue));
  CHKERRQ(VecDestroyVecs(user->nt,&user->yi));
  CHKERRQ(VecDestroyVecs(user->nt,&user->yiwork));
  CHKERRQ(VecDestroyVecs(user->nt,&user->ziwork));
  CHKERRQ(VecDestroyVecs(user->nt,&user->uxi));
  CHKERRQ(VecDestroyVecs(user->nt,&user->uyi));
  CHKERRQ(VecDestroyVecs(user->nt,&user->uxiwork));
  CHKERRQ(VecDestroyVecs(user->nt,&user->uyiwork));
  CHKERRQ(VecDestroyVecs(user->nt,&user->ui));
  CHKERRQ(VecDestroyVecs(user->nt,&user->uiwork));
  CHKERRQ(VecDestroy(&user->c));
  CHKERRQ(VecDestroy(&user->cwork));
  CHKERRQ(VecDestroy(&user->ur));
  CHKERRQ(VecDestroy(&user->q));
  CHKERRQ(VecDestroy(&user->d));
  CHKERRQ(VecDestroy(&user->dwork));
  CHKERRQ(VecDestroy(&user->lwork));
  CHKERRQ(VecDestroy(&user->js_diag));
  CHKERRQ(ISDestroy(&user->s_is));
  CHKERRQ(ISDestroy(&user->d_is));
  CHKERRQ(VecScatterDestroy(&user->state_scatter));
  CHKERRQ(VecScatterDestroy(&user->design_scatter));
  for (i=0; i<user->nt; i++) {
    CHKERRQ(VecScatterDestroy(&user->uxi_scatter[i]));
    CHKERRQ(VecScatterDestroy(&user->uyi_scatter[i]));
    CHKERRQ(VecScatterDestroy(&user->ux_scatter[i]));
    CHKERRQ(VecScatterDestroy(&user->uy_scatter[i]));
    CHKERRQ(VecScatterDestroy(&user->ui_scatter[i]));
    CHKERRQ(VecScatterDestroy(&user->yi_scatter[i]));
  }
  CHKERRQ(PetscFree(user->uxi_scatter));
  CHKERRQ(PetscFree(user->uyi_scatter));
  CHKERRQ(PetscFree(user->ux_scatter));
  CHKERRQ(PetscFree(user->uy_scatter));
  CHKERRQ(PetscFree(user->ui_scatter));
  CHKERRQ(PetscFree(user->yi_scatter));
  PetscFunctionReturn(0);
}

PetscErrorCode HyperbolicMonitor(Tao tao, void *ptr)
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
      requires: !single
      args: -tao_cmonitor -tao_max_funcs 10 -tao_type lcl -tao_gatol 1.e-5

   test:
      suffix: guess_pod
      requires: !single
      args: -tao_cmonitor -tao_max_funcs 10 -tao_type lcl -ksp_guess_type pod -tao_gatol 1.e-5

TEST*/
