#include <petsctao.h>

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

  PetscCall(PetscInitialize(&argc, &argv, (char*)0,help));
  user.mx = 32;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"hyperbolic example",NULL);PetscCall(ierr);
  PetscCall(PetscOptionsInt("-mx","Number of grid points in each direction","",user.mx,&user.mx,NULL));
  user.nt = 16;
  PetscCall(PetscOptionsInt("-nt","Number of time steps","",user.nt,&user.nt,NULL));
  user.ndata = 64;
  PetscCall(PetscOptionsInt("-ndata","Numbers of data points per sample","",user.ndata,&user.ndata,NULL));
  user.alpha = 10.0;
  PetscCall(PetscOptionsReal("-alpha","Regularization parameter","",user.alpha,&user.alpha,NULL));
  user.T = 1.0/32.0;
  PetscCall(PetscOptionsReal("-Tfinal","Final time","",user.T,&user.T,NULL));
  PetscCall(PetscOptionsInt("-ntests","Number of times to repeat TaoSolve","",ntests,&ntests,NULL));
  ierr = PetscOptionsEnd();PetscCall(ierr);

  user.m = user.mx*user.mx*user.nt; /*  number of constraints */
  user.n = user.mx*user.mx*3*user.nt; /*  number of variables */
  user.ht = user.T/user.nt; /*  Time step */
  user.gamma = user.T*user.ht / (user.mx*user.mx);

  PetscCall(VecCreate(PETSC_COMM_WORLD,&user.u));
  PetscCall(VecCreate(PETSC_COMM_WORLD,&user.y));
  PetscCall(VecCreate(PETSC_COMM_WORLD,&user.c));
  PetscCall(VecSetSizes(user.u,PETSC_DECIDE,user.n-user.m));
  PetscCall(VecSetSizes(user.y,PETSC_DECIDE,user.m));
  PetscCall(VecSetSizes(user.c,PETSC_DECIDE,user.m));
  PetscCall(VecSetFromOptions(user.u));
  PetscCall(VecSetFromOptions(user.y));
  PetscCall(VecSetFromOptions(user.c));

  /* Create scatters for reduced spaces.
     If the state vector y and design vector u are partitioned as
     [y_1; y_2; ...; y_np] and [u_1; u_2; ...; u_np] (with np = # of processors),
     then the solution vector x is organized as
     [y_1; u_1; y_2; u_2; ...; y_np; u_np].
     The index sets user.s_is and user.d_is correspond to the indices of the
     state and design variables owned by the current processor.
  */
  PetscCall(VecCreate(PETSC_COMM_WORLD,&x));

  PetscCall(VecGetOwnershipRange(user.y,&lo,&hi));
  PetscCall(VecGetOwnershipRange(user.u,&lo2,&hi2));

  PetscCall(ISCreateStride(PETSC_COMM_SELF,hi-lo,lo,1,&is_allstate));
  PetscCall(ISCreateStride(PETSC_COMM_SELF,hi-lo,lo+lo2,1,&user.s_is));

  PetscCall(ISCreateStride(PETSC_COMM_SELF,hi2-lo2,lo2,1,&is_alldesign));
  PetscCall(ISCreateStride(PETSC_COMM_SELF,hi2-lo2,hi+lo2,1,&user.d_is));

  PetscCall(VecSetSizes(x,hi-lo+hi2-lo2,user.n));
  PetscCall(VecSetFromOptions(x));

  PetscCall(VecScatterCreate(x,user.s_is,user.y,is_allstate,&user.state_scatter));
  PetscCall(VecScatterCreate(x,user.d_is,user.u,is_alldesign,&user.design_scatter));
  PetscCall(ISDestroy(&is_alldesign));
  PetscCall(ISDestroy(&is_allstate));

  /* Create TAO solver and set desired solution method */
  PetscCall(TaoCreate(PETSC_COMM_WORLD,&tao));
  PetscCall(TaoSetType(tao,TAOLCL));

  /* Set up initial vectors and matrices */
  PetscCall(HyperbolicInitialize(&user));

  PetscCall(Gather(x,user.y,user.state_scatter,user.u,user.design_scatter));
  PetscCall(VecDuplicate(x,&x0));
  PetscCall(VecCopy(x,x0));

  /* Set solution vector with an initial guess */
  PetscCall(TaoSetSolution(tao,x));
  PetscCall(TaoSetObjective(tao, FormFunction, &user));
  PetscCall(TaoSetGradient(tao, NULL, FormGradient, &user));
  PetscCall(TaoSetConstraintsRoutine(tao, user.c, FormConstraints, &user));
  PetscCall(TaoSetJacobianStateRoutine(tao, user.Js, user.Js, user.JsInv, FormJacobianState, &user));
  PetscCall(TaoSetJacobianDesignRoutine(tao, user.Jd, FormJacobianDesign, &user));
  PetscCall(TaoSetFromOptions(tao));
  PetscCall(TaoSetStateDesignIS(tao,user.s_is,user.d_is));

  /* SOLVE THE APPLICATION */
  PetscCall(PetscLogStageRegister("Trials",&stages[0]));
  PetscCall(PetscLogStagePush(stages[0]));
  user.ksp_its_initial = user.ksp_its;
  ksp_old = user.ksp_its;
  for (i=0; i<ntests; i++) {
    PetscCall(TaoSolve(tao));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"KSP Iterations = %D\n",user.ksp_its-ksp_old));
    PetscCall(VecCopy(x0,x));
    PetscCall(TaoSetSolution(tao,x));
  }
  PetscCall(PetscLogStagePop());
  PetscCall(PetscBarrier((PetscObject)x));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"KSP iterations within initialization: "));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"%D\n",user.ksp_its_initial));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Total KSP iterations over %D trial(s): ",ntests));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"%D\n",user.ksp_its));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"KSP iterations per trial: "));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"%D\n",(user.ksp_its-user.ksp_its_initial)/ntests));

  PetscCall(TaoDestroy(&tao));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&x0));
  PetscCall(HyperbolicDestroy(&user));
  PetscCall(PetscFinalize());
  return 0;
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
  PetscCall(Scatter(X,user->y,user->state_scatter,user->u,user->design_scatter));
  PetscCall(MatMult(user->Q,user->y,user->dwork));
  PetscCall(VecAXPY(user->dwork,-1.0,user->d));
  PetscCall(VecDot(user->dwork,user->dwork,&d1));

  PetscCall(VecWAXPY(user->uwork,-1.0,user->ur,user->u));
  PetscCall(VecPointwiseMult(user->uwork,user->uwork,user->uwork));
  PetscCall(MatMult(user->L,user->uwork,user->lwork));
  PetscCall(VecDot(user->y,user->lwork,&d2));
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
  PetscCall(Scatter(X,user->y,user->state_scatter,user->u,user->design_scatter));
  PetscCall(MatMult(user->Q,user->y,user->dwork));
  PetscCall(VecAXPY(user->dwork,-1.0,user->d));

  PetscCall(MatMult(user->QT,user->dwork,user->ywork));

  PetscCall(MatMult(user->LT,user->y,user->uwork));
  PetscCall(VecWAXPY(user->vwork,-1.0,user->ur,user->u));
  PetscCall(VecPointwiseMult(user->uwork,user->vwork,user->uwork));
  PetscCall(VecScale(user->uwork,user->alpha));

  PetscCall(VecPointwiseMult(user->vwork,user->vwork,user->vwork));
  PetscCall(MatMult(user->L,user->vwork,user->lwork));
  PetscCall(VecAXPY(user->ywork,0.5*user->alpha,user->lwork));

  PetscCall(Gather(G,user->ywork,user->state_scatter,user->uwork,user->design_scatter));
  PetscFunctionReturn(0);
}

PetscErrorCode FormFunctionGradient(Tao tao, Vec X, PetscReal *f, Vec G, void *ptr)
{
  PetscReal      d1,d2;
  AppCtx         *user = (AppCtx*)ptr;

  PetscFunctionBegin;
  PetscCall(Scatter(X,user->y,user->state_scatter,user->u,user->design_scatter));
  PetscCall(MatMult(user->Q,user->y,user->dwork));
  PetscCall(VecAXPY(user->dwork,-1.0,user->d));

  PetscCall(MatMult(user->QT,user->dwork,user->ywork));

  PetscCall(VecDot(user->dwork,user->dwork,&d1));

  PetscCall(MatMult(user->LT,user->y,user->uwork));
  PetscCall(VecWAXPY(user->vwork,-1.0,user->ur,user->u));
  PetscCall(VecPointwiseMult(user->uwork,user->vwork,user->uwork));
  PetscCall(VecScale(user->uwork,user->alpha));

  PetscCall(VecPointwiseMult(user->vwork,user->vwork,user->vwork));
  PetscCall(MatMult(user->L,user->vwork,user->lwork));
  PetscCall(VecAXPY(user->ywork,0.5*user->alpha,user->lwork));

  PetscCall(VecDot(user->y,user->lwork,&d2));

  *f = 0.5 * (d1 + user->alpha*d2);
  PetscCall(Gather(G,user->ywork,user->state_scatter,user->uwork,user->design_scatter));
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
  PetscCall(Scatter(X,user->y,user->state_scatter,user->u,user->design_scatter));
  PetscCall(Scatter_yi(user->u,user->ui,user->ui_scatter,user->nt));
  PetscCall(Scatter_uxi_uyi(user->u,user->uxi,user->uxi_scatter,user->uyi,user->uyi_scatter,user->nt));
  for (i=0; i<user->nt; i++) {
    PetscCall(MatCopy(user->Divxy[0],user->C[i],SUBSET_NONZERO_PATTERN));
    PetscCall(MatCopy(user->Divxy[1],user->Cwork[i],SAME_NONZERO_PATTERN));

    PetscCall(MatDiagonalScale(user->C[i],NULL,user->uxi[i]));
    PetscCall(MatDiagonalScale(user->Cwork[i],NULL,user->uyi[i]));
    PetscCall(MatAXPY(user->C[i],1.0,user->Cwork[i],SUBSET_NONZERO_PATTERN));
    PetscCall(MatScale(user->C[i],user->ht));
    PetscCall(MatShift(user->C[i],1.0));
  }
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
/* B */
PetscErrorCode FormJacobianDesign(Tao tao, Vec X, Mat J, void *ptr)
{
  AppCtx         *user = (AppCtx*)ptr;

  PetscFunctionBegin;
  PetscCall(Scatter(X,user->y,user->state_scatter,user->u,user->design_scatter));
  PetscFunctionReturn(0);
}

PetscErrorCode StateMatMult(Mat J_shell, Vec X, Vec Y)
{
  PetscInt       i;
  AppCtx         *user;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(J_shell,&user));
  PetscCall(Scatter_yi(X,user->yi,user->yi_scatter,user->nt));
  user->block_index = 0;
  PetscCall(MatMult(user->JsBlock,user->yi[0],user->yiwork[0]));

  for (i=1; i<user->nt; i++) {
    user->block_index = i;
    PetscCall(MatMult(user->JsBlock,user->yi[i],user->yiwork[i]));
    PetscCall(MatMult(user->M,user->yi[i-1],user->ziwork[i-1]));
    PetscCall(VecAXPY(user->yiwork[i],-1.0,user->ziwork[i-1]));
  }
  PetscCall(Gather_yi(Y,user->yiwork,user->yi_scatter,user->nt));
  PetscFunctionReturn(0);
}

PetscErrorCode StateMatMultTranspose(Mat J_shell, Vec X, Vec Y)
{
  PetscInt       i;
  AppCtx         *user;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(J_shell,&user));
  PetscCall(Scatter_yi(X,user->yi,user->yi_scatter,user->nt));

  for (i=0; i<user->nt-1; i++) {
    user->block_index = i;
    PetscCall(MatMultTranspose(user->JsBlock,user->yi[i],user->yiwork[i]));
    PetscCall(MatMult(user->M,user->yi[i+1],user->ziwork[i+1]));
    PetscCall(VecAXPY(user->yiwork[i],-1.0,user->ziwork[i+1]));
  }

  i = user->nt-1;
  user->block_index = i;
  PetscCall(MatMultTranspose(user->JsBlock,user->yi[i],user->yiwork[i]));
  PetscCall(Gather_yi(Y,user->yiwork,user->yi_scatter,user->nt));
  PetscFunctionReturn(0);
}

PetscErrorCode StateMatBlockMult(Mat J_shell, Vec X, Vec Y)
{
  PetscInt       i;
  AppCtx         *user;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(J_shell,&user));
  i = user->block_index;
  PetscCall(VecPointwiseMult(user->uxiwork[i],X,user->uxi[i]));
  PetscCall(VecPointwiseMult(user->uyiwork[i],X,user->uyi[i]));
  PetscCall(Gather(user->uiwork[i],user->uxiwork[i],user->ux_scatter[i],user->uyiwork[i],user->uy_scatter[i]));
  PetscCall(MatMult(user->Div,user->uiwork[i],Y));
  PetscCall(VecAYPX(Y,user->ht,X));
  PetscFunctionReturn(0);
}

PetscErrorCode StateMatBlockMultTranspose(Mat J_shell, Vec X, Vec Y)
{
  PetscInt       i;
  AppCtx         *user;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(J_shell,&user));
  i = user->block_index;
  PetscCall(MatMult(user->Grad,X,user->uiwork[i]));
  PetscCall(Scatter(user->uiwork[i],user->uxiwork[i],user->ux_scatter[i],user->uyiwork[i],user->uy_scatter[i]));
  PetscCall(VecPointwiseMult(user->uxiwork[i],user->uxi[i],user->uxiwork[i]));
  PetscCall(VecPointwiseMult(user->uyiwork[i],user->uyi[i],user->uyiwork[i]));
  PetscCall(VecWAXPY(Y,1.0,user->uxiwork[i],user->uyiwork[i]));
  PetscCall(VecAYPX(Y,user->ht,X));
  PetscFunctionReturn(0);
}

PetscErrorCode DesignMatMult(Mat J_shell, Vec X, Vec Y)
{
  PetscInt       i;
  AppCtx         *user;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(J_shell,&user));
  PetscCall(Scatter_yi(user->y,user->yi,user->yi_scatter,user->nt));
  PetscCall(Scatter_uxi_uyi(X,user->uxiwork,user->uxi_scatter,user->uyiwork,user->uyi_scatter,user->nt));
  for (i=0; i<user->nt; i++) {
    PetscCall(VecPointwiseMult(user->uxiwork[i],user->yi[i],user->uxiwork[i]));
    PetscCall(VecPointwiseMult(user->uyiwork[i],user->yi[i],user->uyiwork[i]));
    PetscCall(Gather(user->uiwork[i],user->uxiwork[i],user->ux_scatter[i],user->uyiwork[i],user->uy_scatter[i]));
    PetscCall(MatMult(user->Div,user->uiwork[i],user->ziwork[i]));
    PetscCall(VecScale(user->ziwork[i],user->ht));
  }
  PetscCall(Gather_yi(Y,user->ziwork,user->yi_scatter,user->nt));
  PetscFunctionReturn(0);
}

PetscErrorCode DesignMatMultTranspose(Mat J_shell, Vec X, Vec Y)
{
  PetscInt       i;
  AppCtx         *user;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(J_shell,&user));
  PetscCall(Scatter_yi(user->y,user->yi,user->yi_scatter,user->nt));
  PetscCall(Scatter_yi(X,user->yiwork,user->yi_scatter,user->nt));
  for (i=0; i<user->nt; i++) {
    PetscCall(MatMult(user->Grad,user->yiwork[i],user->uiwork[i]));
    PetscCall(Scatter(user->uiwork[i],user->uxiwork[i],user->ux_scatter[i],user->uyiwork[i],user->uy_scatter[i]));
    PetscCall(VecPointwiseMult(user->uxiwork[i],user->yi[i],user->uxiwork[i]));
    PetscCall(VecPointwiseMult(user->uyiwork[i],user->yi[i],user->uyiwork[i]));
    PetscCall(Gather(user->uiwork[i],user->uxiwork[i],user->ux_scatter[i],user->uyiwork[i],user->uy_scatter[i]));
    PetscCall(VecScale(user->uiwork[i],user->ht));
  }
  PetscCall(Gather_yi(Y,user->uiwork,user->ui_scatter,user->nt));
  PetscFunctionReturn(0);
}

PetscErrorCode StateMatBlockPrecMult(PC PC_shell, Vec X, Vec Y)
{
  PetscInt       i;
  AppCtx         *user;

  PetscFunctionBegin;
  PetscCall(PCShellGetContext(PC_shell,&user));
  i = user->block_index;
  if (user->c_formed) {
    PetscCall(MatSOR(user->C[i],X,1.0,(MatSORType)(SOR_ZERO_INITIAL_GUESS | SOR_LOCAL_SYMMETRIC_SWEEP),0.0,1,1,Y));
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not formed");
  PetscFunctionReturn(0);
}

PetscErrorCode StateMatBlockPrecMultTranspose(PC PC_shell, Vec X, Vec Y)
{
  PetscInt       i;
  AppCtx         *user;

  PetscFunctionBegin;
  PetscCall(PCShellGetContext(PC_shell,&user));

  i = user->block_index;
  if (user->c_formed) {
    PetscCall(MatSOR(user->C[i],X,1.0,(MatSORType)(SOR_ZERO_INITIAL_GUESS | SOR_LOCAL_SYMMETRIC_SWEEP),0.0,1,1,Y));
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not formed");
  PetscFunctionReturn(0);
}

PetscErrorCode StateMatInvMult(Mat J_shell, Vec X, Vec Y)
{
  AppCtx         *user;
  PetscInt       its,i;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(J_shell,&user));

  if (Y == user->ytrue) {
    /* First solve is done using true solution to set up problem */
    PetscCall(KSPSetTolerances(user->solver,1e-4,1e-20,PETSC_DEFAULT,PETSC_DEFAULT));
  } else {
    PetscCall(KSPSetTolerances(user->solver,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT));
  }
  PetscCall(Scatter_yi(X,user->yi,user->yi_scatter,user->nt));
  PetscCall(Scatter_yi(Y,user->yiwork,user->yi_scatter,user->nt));
  PetscCall(Scatter_uxi_uyi(user->u,user->uxi,user->uxi_scatter,user->uyi,user->uyi_scatter,user->nt));

  user->block_index = 0;
  PetscCall(KSPSolve(user->solver,user->yi[0],user->yiwork[0]));

  PetscCall(KSPGetIterationNumber(user->solver,&its));
  user->ksp_its = user->ksp_its + its;
  for (i=1; i<user->nt; i++) {
    PetscCall(MatMult(user->M,user->yiwork[i-1],user->ziwork[i-1]));
    PetscCall(VecAXPY(user->yi[i],1.0,user->ziwork[i-1]));
    user->block_index = i;
    PetscCall(KSPSolve(user->solver,user->yi[i],user->yiwork[i]));

    PetscCall(KSPGetIterationNumber(user->solver,&its));
    user->ksp_its = user->ksp_its + its;
  }
  PetscCall(Gather_yi(Y,user->yiwork,user->yi_scatter,user->nt));
  PetscFunctionReturn(0);
}

PetscErrorCode StateMatInvTransposeMult(Mat J_shell, Vec X, Vec Y)
{
  AppCtx         *user;
  PetscInt       its,i;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(J_shell,&user));

  PetscCall(Scatter_yi(X,user->yi,user->yi_scatter,user->nt));
  PetscCall(Scatter_yi(Y,user->yiwork,user->yi_scatter,user->nt));
  PetscCall(Scatter_uxi_uyi(user->u,user->uxi,user->uxi_scatter,user->uyi,user->uyi_scatter,user->nt));

  i = user->nt - 1;
  user->block_index = i;
  PetscCall(KSPSolveTranspose(user->solver,user->yi[i],user->yiwork[i]));

  PetscCall(KSPGetIterationNumber(user->solver,&its));
  user->ksp_its = user->ksp_its + its;

  for (i=user->nt-2; i>=0; i--) {
    PetscCall(MatMult(user->M,user->yiwork[i+1],user->ziwork[i+1]));
    PetscCall(VecAXPY(user->yi[i],1.0,user->ziwork[i+1]));
    user->block_index = i;
    PetscCall(KSPSolveTranspose(user->solver,user->yi[i],user->yiwork[i]));

    PetscCall(KSPGetIterationNumber(user->solver,&its));
    user->ksp_its = user->ksp_its + its;
  }
  PetscCall(Gather_yi(Y,user->yiwork,user->yi_scatter,user->nt));
  PetscFunctionReturn(0);
}

PetscErrorCode StateMatDuplicate(Mat J_shell, MatDuplicateOption opt, Mat *new_shell)
{
  AppCtx         *user;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(J_shell,&user));

  PetscCall(MatCreateShell(PETSC_COMM_WORLD,PETSC_DETERMINE,PETSC_DETERMINE,user->m,user->m,user,new_shell));
  PetscCall(MatShellSetOperation(*new_shell,MATOP_MULT,(void(*)(void))StateMatMult));
  PetscCall(MatShellSetOperation(*new_shell,MATOP_DUPLICATE,(void(*)(void))StateMatDuplicate));
  PetscCall(MatShellSetOperation(*new_shell,MATOP_MULT_TRANSPOSE,(void(*)(void))StateMatMultTranspose));
  PetscCall(MatShellSetOperation(*new_shell,MATOP_GET_DIAGONAL,(void(*)(void))StateMatGetDiagonal));
  PetscFunctionReturn(0);
}

PetscErrorCode StateMatGetDiagonal(Mat J_shell, Vec X)
{
  AppCtx         *user;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(J_shell,&user));
  PetscCall(VecCopy(user->js_diag,X));
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
  PetscCall(Scatter(X,user->y,user->state_scatter,user->u,user->design_scatter));
  PetscCall(Scatter_yi(user->y,user->yi,user->yi_scatter,user->nt));
  PetscCall(Scatter_uxi_uyi(user->u,user->uxi,user->uxi_scatter,user->uyi,user->uyi_scatter,user->nt));

  user->block_index = 0;
  PetscCall(MatMult(user->JsBlock,user->yi[0],user->yiwork[0]));

  for (i=1; i<user->nt; i++) {
    user->block_index = i;
    PetscCall(MatMult(user->JsBlock,user->yi[i],user->yiwork[i]));
    PetscCall(MatMult(user->M,user->yi[i-1],user->ziwork[i-1]));
    PetscCall(VecAXPY(user->yiwork[i],-1.0,user->ziwork[i-1]));
  }

  PetscCall(Gather_yi(C,user->yiwork,user->yi_scatter,user->nt));
  PetscCall(VecAXPY(C,-1.0,user->q));

  PetscFunctionReturn(0);
}

PetscErrorCode Scatter(Vec x, Vec state, VecScatter s_scat, Vec design, VecScatter d_scat)
{
  PetscFunctionBegin;
  PetscCall(VecScatterBegin(s_scat,x,state,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd(s_scat,x,state,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterBegin(d_scat,x,design,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd(d_scat,x,design,INSERT_VALUES,SCATTER_FORWARD));
  PetscFunctionReturn(0);
}

PetscErrorCode Scatter_uxi_uyi(Vec u, Vec *uxi, VecScatter *scatx, Vec *uyi, VecScatter *scaty, PetscInt nt)
{
  PetscInt       i;

  PetscFunctionBegin;
  for (i=0; i<nt; i++) {
    PetscCall(VecScatterBegin(scatx[i],u,uxi[i],INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecScatterEnd(scatx[i],u,uxi[i],INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecScatterBegin(scaty[i],u,uyi[i],INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecScatterEnd(scaty[i],u,uyi[i],INSERT_VALUES,SCATTER_FORWARD));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode Gather(Vec x, Vec state, VecScatter s_scat, Vec design, VecScatter d_scat)
{
  PetscFunctionBegin;
  PetscCall(VecScatterBegin(s_scat,state,x,INSERT_VALUES,SCATTER_REVERSE));
  PetscCall(VecScatterEnd(s_scat,state,x,INSERT_VALUES,SCATTER_REVERSE));
  PetscCall(VecScatterBegin(d_scat,design,x,INSERT_VALUES,SCATTER_REVERSE));
  PetscCall(VecScatterEnd(d_scat,design,x,INSERT_VALUES,SCATTER_REVERSE));
  PetscFunctionReturn(0);
}

PetscErrorCode Gather_uxi_uyi(Vec u, Vec *uxi, VecScatter *scatx, Vec *uyi, VecScatter *scaty, PetscInt nt)
{
  PetscInt       i;

  PetscFunctionBegin;
  for (i=0; i<nt; i++) {
    PetscCall(VecScatterBegin(scatx[i],uxi[i],u,INSERT_VALUES,SCATTER_REVERSE));
    PetscCall(VecScatterEnd(scatx[i],uxi[i],u,INSERT_VALUES,SCATTER_REVERSE));
    PetscCall(VecScatterBegin(scaty[i],uyi[i],u,INSERT_VALUES,SCATTER_REVERSE));
    PetscCall(VecScatterEnd(scaty[i],uyi[i],u,INSERT_VALUES,SCATTER_REVERSE));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode Scatter_yi(Vec y, Vec *yi, VecScatter *scat, PetscInt nt)
{
  PetscInt       i;

  PetscFunctionBegin;
  for (i=0; i<nt; i++) {
    PetscCall(VecScatterBegin(scat[i],y,yi[i],INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecScatterEnd(scat[i],y,yi[i],INSERT_VALUES,SCATTER_FORWARD));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode Gather_yi(Vec y, Vec *yi, VecScatter *scat, PetscInt nt)
{
  PetscInt       i;

  PetscFunctionBegin;
  for (i=0; i<nt; i++) {
    PetscCall(VecScatterBegin(scat[i],yi[i],y,INSERT_VALUES,SCATTER_REVERSE));
    PetscCall(VecScatterEnd(scat[i],yi[i],y,INSERT_VALUES,SCATTER_REVERSE));
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
  PetscCall(MatCreate(PETSC_COMM_WORLD,&user->Grad));
  PetscCall(MatSetSizes(user->Grad,PETSC_DECIDE,PETSC_DECIDE,2*n,n));
  PetscCall(MatSetFromOptions(user->Grad));
  PetscCall(MatMPIAIJSetPreallocation(user->Grad,3,NULL,3,NULL));
  PetscCall(MatSeqAIJSetPreallocation(user->Grad,3,NULL));
  PetscCall(MatGetOwnershipRange(user->Grad,&istart,&iend));

  for (i=istart; i<iend; i++) {
    if (i<n) {
      iblock = i / user->mx;
      j = iblock*user->mx + ((i+user->mx-1) % user->mx);
      PetscCall(MatSetValues(user->Grad,1,&i,1,&j,&half_hinv,INSERT_VALUES));
      j = iblock*user->mx + ((i+1) % user->mx);
      PetscCall(MatSetValues(user->Grad,1,&i,1,&j,&neg_half_hinv,INSERT_VALUES));
    }
    if (i>=n) {
      j = (i - user->mx) % n;
      PetscCall(MatSetValues(user->Grad,1,&i,1,&j,&half_hinv,INSERT_VALUES));
      j = (j + 2*user->mx) % n;
      PetscCall(MatSetValues(user->Grad,1,&i,1,&j,&neg_half_hinv,INSERT_VALUES));
    }
  }

  PetscCall(MatAssemblyBegin(user->Grad,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(user->Grad,MAT_FINAL_ASSEMBLY));

  PetscCall(MatCreate(PETSC_COMM_WORLD,&user->Gradxy[0]));
  PetscCall(MatSetSizes(user->Gradxy[0],PETSC_DECIDE,PETSC_DECIDE,n,n));
  PetscCall(MatSetFromOptions(user->Gradxy[0]));
  PetscCall(MatMPIAIJSetPreallocation(user->Gradxy[0],3,NULL,3,NULL));
  PetscCall(MatSeqAIJSetPreallocation(user->Gradxy[0],3,NULL));
  PetscCall(MatGetOwnershipRange(user->Gradxy[0],&istart,&iend));

  for (i=istart; i<iend; i++) {
    iblock = i / user->mx;
    j = iblock*user->mx + ((i+user->mx-1) % user->mx);
    PetscCall(MatSetValues(user->Gradxy[0],1,&i,1,&j,&half_hinv,INSERT_VALUES));
    j = iblock*user->mx + ((i+1) % user->mx);
    PetscCall(MatSetValues(user->Gradxy[0],1,&i,1,&j,&neg_half_hinv,INSERT_VALUES));
    PetscCall(MatSetValues(user->Gradxy[0],1,&i,1,&i,&zero,INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(user->Gradxy[0],MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(user->Gradxy[0],MAT_FINAL_ASSEMBLY));

  PetscCall(MatCreate(PETSC_COMM_WORLD,&user->Gradxy[1]));
  PetscCall(MatSetSizes(user->Gradxy[1],PETSC_DECIDE,PETSC_DECIDE,n,n));
  PetscCall(MatSetFromOptions(user->Gradxy[1]));
  PetscCall(MatMPIAIJSetPreallocation(user->Gradxy[1],3,NULL,3,NULL));
  PetscCall(MatSeqAIJSetPreallocation(user->Gradxy[1],3,NULL));
  PetscCall(MatGetOwnershipRange(user->Gradxy[1],&istart,&iend));

  for (i=istart; i<iend; i++) {
    j = (i + n - user->mx) % n;
    PetscCall(MatSetValues(user->Gradxy[1],1,&i,1,&j,&half_hinv,INSERT_VALUES));
    j = (j + 2*user->mx) % n;
    PetscCall(MatSetValues(user->Gradxy[1],1,&i,1,&j,&neg_half_hinv,INSERT_VALUES));
    PetscCall(MatSetValues(user->Gradxy[1],1,&i,1,&i,&zero,INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(user->Gradxy[1],MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(user->Gradxy[1],MAT_FINAL_ASSEMBLY));

  /* Generate Div matrix */
  PetscCall(MatTranspose(user->Grad,MAT_INITIAL_MATRIX,&user->Div));
  PetscCall(MatTranspose(user->Gradxy[0],MAT_INITIAL_MATRIX,&user->Divxy[0]));
  PetscCall(MatTranspose(user->Gradxy[1],MAT_INITIAL_MATRIX,&user->Divxy[1]));

  /* Off-diagonal averaging matrix */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&user->M));
  PetscCall(MatSetSizes(user->M,PETSC_DECIDE,PETSC_DECIDE,n,n));
  PetscCall(MatSetFromOptions(user->M));
  PetscCall(MatMPIAIJSetPreallocation(user->M,4,NULL,4,NULL));
  PetscCall(MatSeqAIJSetPreallocation(user->M,4,NULL));
  PetscCall(MatGetOwnershipRange(user->M,&istart,&iend));

  for (i=istart; i<iend; i++) {
    /* kron(Id,Av) */
    iblock = i / user->mx;
    j = iblock*user->mx + ((i+user->mx-1) % user->mx);
    PetscCall(MatSetValues(user->M,1,&i,1,&j,&quarter,INSERT_VALUES));
    j = iblock*user->mx + ((i+1) % user->mx);
    PetscCall(MatSetValues(user->M,1,&i,1,&j,&quarter,INSERT_VALUES));

    /* kron(Av,Id) */
    j = (i + user->mx) % n;
    PetscCall(MatSetValues(user->M,1,&i,1,&j,&quarter,INSERT_VALUES));
    j = (i + n - user->mx) % n;
    PetscCall(MatSetValues(user->M,1,&i,1,&j,&quarter,INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(user->M,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(user->M,MAT_FINAL_ASSEMBLY));

  /* Generate 2D grid */
  PetscCall(VecCreate(PETSC_COMM_WORLD,&XX));
  PetscCall(VecCreate(PETSC_COMM_WORLD,&user->q));
  PetscCall(VecSetSizes(XX,PETSC_DECIDE,n));
  PetscCall(VecSetSizes(user->q,PETSC_DECIDE,n*user->nt));
  PetscCall(VecSetFromOptions(XX));
  PetscCall(VecSetFromOptions(user->q));

  PetscCall(VecDuplicate(XX,&YY));
  PetscCall(VecDuplicate(XX,&XXwork));
  PetscCall(VecDuplicate(XX,&YYwork));
  PetscCall(VecDuplicate(XX,&user->d));
  PetscCall(VecDuplicate(XX,&user->dwork));

  PetscCall(VecGetOwnershipRange(XX,&istart,&iend));
  for (linear_index=istart; linear_index<iend; linear_index++) {
    i = linear_index % user->mx;
    j = (linear_index-i)/user->mx;
    vx = h*(i+0.5);
    vy = h*(j+0.5);
    PetscCall(VecSetValues(XX,1,&linear_index,&vx,INSERT_VALUES));
    PetscCall(VecSetValues(YY,1,&linear_index,&vy,INSERT_VALUES));
  }

  PetscCall(VecAssemblyBegin(XX));
  PetscCall(VecAssemblyEnd(XX));
  PetscCall(VecAssemblyBegin(YY));
  PetscCall(VecAssemblyEnd(YY));

  /* Compute final density function yT
     yT = 1.0 + exp(-30*((x-0.25)^2+(y-0.25)^2)) + exp(-30*((x-0.75)^2+(y-0.75)^2))
     yT = yT / (h^2*sum(yT)) */
  PetscCall(VecCopy(XX,XXwork));
  PetscCall(VecCopy(YY,YYwork));

  PetscCall(VecShift(XXwork,-0.25));
  PetscCall(VecShift(YYwork,-0.25));

  PetscCall(VecPointwiseMult(XXwork,XXwork,XXwork));
  PetscCall(VecPointwiseMult(YYwork,YYwork,YYwork));

  PetscCall(VecCopy(XXwork,user->dwork));
  PetscCall(VecAXPY(user->dwork,1.0,YYwork));
  PetscCall(VecScale(user->dwork,-30.0));
  PetscCall(VecExp(user->dwork));
  PetscCall(VecCopy(user->dwork,user->d));

  PetscCall(VecCopy(XX,XXwork));
  PetscCall(VecCopy(YY,YYwork));

  PetscCall(VecShift(XXwork,-0.75));
  PetscCall(VecShift(YYwork,-0.75));

  PetscCall(VecPointwiseMult(XXwork,XXwork,XXwork));
  PetscCall(VecPointwiseMult(YYwork,YYwork,YYwork));

  PetscCall(VecCopy(XXwork,user->dwork));
  PetscCall(VecAXPY(user->dwork,1.0,YYwork));
  PetscCall(VecScale(user->dwork,-30.0));
  PetscCall(VecExp(user->dwork));

  PetscCall(VecAXPY(user->d,1.0,user->dwork));
  PetscCall(VecShift(user->d,1.0));
  PetscCall(VecSum(user->d,&sum));
  PetscCall(VecScale(user->d,1.0/(h*h*sum)));

  /* Initial conditions of forward problem */
  PetscCall(VecDuplicate(XX,&bc));
  PetscCall(VecCopy(XX,XXwork));
  PetscCall(VecCopy(YY,YYwork));

  PetscCall(VecShift(XXwork,-0.5));
  PetscCall(VecShift(YYwork,-0.5));

  PetscCall(VecPointwiseMult(XXwork,XXwork,XXwork));
  PetscCall(VecPointwiseMult(YYwork,YYwork,YYwork));

  PetscCall(VecWAXPY(bc,1.0,XXwork,YYwork));
  PetscCall(VecScale(bc,-50.0));
  PetscCall(VecExp(bc));
  PetscCall(VecShift(bc,1.0));
  PetscCall(VecSum(bc,&sum));
  PetscCall(VecScale(bc,1.0/(h*h*sum)));

  /* Create scatter from y to y_1,y_2,...,y_nt */
  /*  TODO: Reorder for better parallelism. (This will require reordering Q and L as well.) */
  PetscCall(PetscMalloc1(user->nt*user->mx*user->mx,&user->yi_scatter));
  PetscCall(VecCreate(PETSC_COMM_WORLD,&yi));
  PetscCall(VecSetSizes(yi,PETSC_DECIDE,user->mx*user->mx));
  PetscCall(VecSetFromOptions(yi));
  PetscCall(VecDuplicateVecs(yi,user->nt,&user->yi));
  PetscCall(VecDuplicateVecs(yi,user->nt,&user->yiwork));
  PetscCall(VecDuplicateVecs(yi,user->nt,&user->ziwork));
  for (i=0; i<user->nt; i++) {
    PetscCall(VecGetOwnershipRange(user->yi[i],&lo,&hi));
    PetscCall(ISCreateStride(PETSC_COMM_SELF,hi-lo,lo,1,&is_to_yi));
    PetscCall(ISCreateStride(PETSC_COMM_SELF,hi-lo,lo+i*user->mx*user->mx,1,&is_from_y));
    PetscCall(VecScatterCreate(user->y,is_from_y,user->yi[i],is_to_yi,&user->yi_scatter[i]));
    PetscCall(ISDestroy(&is_to_yi));
    PetscCall(ISDestroy(&is_from_y));
  }

  /* Create scatter from u to ux_1,uy_1,ux_2,uy_2,...,ux_nt,uy_nt */
  /*  TODO: reorder for better parallelism */
  PetscCall(PetscMalloc1(user->nt*user->mx*user->mx,&user->uxi_scatter));
  PetscCall(PetscMalloc1(user->nt*user->mx*user->mx,&user->uyi_scatter));
  PetscCall(PetscMalloc1(user->nt*user->mx*user->mx,&user->ux_scatter));
  PetscCall(PetscMalloc1(user->nt*user->mx*user->mx,&user->uy_scatter));
  PetscCall(PetscMalloc1(2*user->nt*user->mx*user->mx,&user->ui_scatter));
  PetscCall(VecCreate(PETSC_COMM_WORLD,&uxi));
  PetscCall(VecCreate(PETSC_COMM_WORLD,&ui));
  PetscCall(VecSetSizes(uxi,PETSC_DECIDE,user->mx*user->mx));
  PetscCall(VecSetSizes(ui,PETSC_DECIDE,2*user->mx*user->mx));
  PetscCall(VecSetFromOptions(uxi));
  PetscCall(VecSetFromOptions(ui));
  PetscCall(VecDuplicateVecs(uxi,user->nt,&user->uxi));
  PetscCall(VecDuplicateVecs(uxi,user->nt,&user->uyi));
  PetscCall(VecDuplicateVecs(uxi,user->nt,&user->uxiwork));
  PetscCall(VecDuplicateVecs(uxi,user->nt,&user->uyiwork));
  PetscCall(VecDuplicateVecs(ui,user->nt,&user->ui));
  PetscCall(VecDuplicateVecs(ui,user->nt,&user->uiwork));
  for (i=0; i<user->nt; i++) {
    PetscCall(VecGetOwnershipRange(user->uxi[i],&lo,&hi));
    PetscCall(ISCreateStride(PETSC_COMM_SELF,hi-lo,lo,1,&is_to_uxi));
    PetscCall(ISCreateStride(PETSC_COMM_SELF,hi-lo,lo+2*i*user->mx*user->mx,1,&is_from_u));
    PetscCall(VecScatterCreate(user->u,is_from_u,user->uxi[i],is_to_uxi,&user->uxi_scatter[i]));

    PetscCall(ISDestroy(&is_to_uxi));
    PetscCall(ISDestroy(&is_from_u));

    PetscCall(VecGetOwnershipRange(user->uyi[i],&lo,&hi));
    PetscCall(ISCreateStride(PETSC_COMM_SELF,hi-lo,lo,1,&is_to_uyi));
    PetscCall(ISCreateStride(PETSC_COMM_SELF,hi-lo,lo+(2*i+1)*user->mx*user->mx,1,&is_from_u));
    PetscCall(VecScatterCreate(user->u,is_from_u,user->uyi[i],is_to_uyi,&user->uyi_scatter[i]));

    PetscCall(ISDestroy(&is_to_uyi));
    PetscCall(ISDestroy(&is_from_u));

    PetscCall(VecGetOwnershipRange(user->uxi[i],&lo,&hi));
    PetscCall(ISCreateStride(PETSC_COMM_SELF,hi-lo,lo,1,&is_to_uxi));
    PetscCall(ISCreateStride(PETSC_COMM_SELF,hi-lo,lo,1,&is_from_u));
    PetscCall(VecScatterCreate(user->ui[i],is_from_u,user->uxi[i],is_to_uxi,&user->ux_scatter[i]));

    PetscCall(ISDestroy(&is_to_uxi));
    PetscCall(ISDestroy(&is_from_u));

    PetscCall(VecGetOwnershipRange(user->uyi[i],&lo,&hi));
    PetscCall(ISCreateStride(PETSC_COMM_SELF,hi-lo,lo,1,&is_to_uyi));
    PetscCall(ISCreateStride(PETSC_COMM_SELF,hi-lo,lo+user->mx*user->mx,1,&is_from_u));
    PetscCall(VecScatterCreate(user->ui[i],is_from_u,user->uyi[i],is_to_uyi,&user->uy_scatter[i]));

    PetscCall(ISDestroy(&is_to_uyi));
    PetscCall(ISDestroy(&is_from_u));

    PetscCall(VecGetOwnershipRange(user->ui[i],&lo,&hi));
    PetscCall(ISCreateStride(PETSC_COMM_SELF,hi-lo,lo,1,&is_to_uxi));
    PetscCall(ISCreateStride(PETSC_COMM_SELF,hi-lo,lo+2*i*user->mx*user->mx,1,&is_from_u));
    PetscCall(VecScatterCreate(user->u,is_from_u,user->ui[i],is_to_uxi,&user->ui_scatter[i]));

    PetscCall(ISDestroy(&is_to_uxi));
    PetscCall(ISDestroy(&is_from_u));
  }

  /* RHS of forward problem */
  PetscCall(MatMult(user->M,bc,user->yiwork[0]));
  for (i=1; i<user->nt; i++) {
    PetscCall(VecSet(user->yiwork[i],0.0));
  }
  PetscCall(Gather_yi(user->q,user->yiwork,user->yi_scatter,user->nt));

  /* Compute true velocity field utrue */
  PetscCall(VecDuplicate(user->u,&user->utrue));
  for (i=0; i<user->nt; i++) {
    PetscCall(VecCopy(YY,user->uxi[i]));
    PetscCall(VecScale(user->uxi[i],150.0*i*user->ht));
    PetscCall(VecCopy(XX,user->uyi[i]));
    PetscCall(VecShift(user->uyi[i],-10.0));
    PetscCall(VecScale(user->uyi[i],15.0*i*user->ht));
  }
  PetscCall(Gather_uxi_uyi(user->utrue,user->uxi,user->uxi_scatter,user->uyi,user->uyi_scatter,user->nt));

  /* Initial guess and reference model */
  PetscCall(VecDuplicate(user->utrue,&user->ur));
  for (i=0; i<user->nt; i++) {
    PetscCall(VecCopy(XX,user->uxi[i]));
    PetscCall(VecShift(user->uxi[i],i*user->ht));
    PetscCall(VecCopy(YY,user->uyi[i]));
    PetscCall(VecShift(user->uyi[i],-i*user->ht));
  }
  PetscCall(Gather_uxi_uyi(user->ur,user->uxi,user->uxi_scatter,user->uyi,user->uyi_scatter,user->nt));

  /* Generate regularization matrix L */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&user->LT));
  PetscCall(MatSetSizes(user->LT,PETSC_DECIDE,PETSC_DECIDE,2*n*user->nt,n*user->nt));
  PetscCall(MatSetFromOptions(user->LT));
  PetscCall(MatMPIAIJSetPreallocation(user->LT,1,NULL,1,NULL));
  PetscCall(MatSeqAIJSetPreallocation(user->LT,1,NULL));
  PetscCall(MatGetOwnershipRange(user->LT,&istart,&iend));

  for (i=istart; i<iend; i++) {
    iblock = (i+n) / (2*n);
    j = i - iblock*n;
    PetscCall(MatSetValues(user->LT,1,&i,1,&j,&user->gamma,INSERT_VALUES));
  }

  PetscCall(MatAssemblyBegin(user->LT,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(user->LT,MAT_FINAL_ASSEMBLY));

  PetscCall(MatTranspose(user->LT,MAT_INITIAL_MATRIX,&user->L));

  /* Build work vectors and matrices */
  PetscCall(VecCreate(PETSC_COMM_WORLD,&user->lwork));
  PetscCall(VecSetType(user->lwork,VECMPI));
  PetscCall(VecSetSizes(user->lwork,PETSC_DECIDE,user->m));
  PetscCall(VecSetFromOptions(user->lwork));

  PetscCall(MatDuplicate(user->Div,MAT_SHARE_NONZERO_PATTERN,&user->Divwork));

  PetscCall(VecDuplicate(user->y,&user->ywork));
  PetscCall(VecDuplicate(user->u,&user->uwork));
  PetscCall(VecDuplicate(user->u,&user->vwork));
  PetscCall(VecDuplicate(user->u,&user->js_diag));
  PetscCall(VecDuplicate(user->c,&user->cwork));

  /* Create matrix-free shell user->Js for computing A*x */
  PetscCall(MatCreateShell(PETSC_COMM_WORLD,PETSC_DETERMINE,PETSC_DETERMINE,user->m,user->m,user,&user->Js));
  PetscCall(MatShellSetOperation(user->Js,MATOP_MULT,(void(*)(void))StateMatMult));
  PetscCall(MatShellSetOperation(user->Js,MATOP_DUPLICATE,(void(*)(void))StateMatDuplicate));
  PetscCall(MatShellSetOperation(user->Js,MATOP_MULT_TRANSPOSE,(void(*)(void))StateMatMultTranspose));
  PetscCall(MatShellSetOperation(user->Js,MATOP_GET_DIAGONAL,(void(*)(void))StateMatGetDiagonal));

  /* Diagonal blocks of user->Js */
  PetscCall(MatCreateShell(PETSC_COMM_WORLD,PETSC_DETERMINE,PETSC_DETERMINE,n,n,user,&user->JsBlock));
  PetscCall(MatShellSetOperation(user->JsBlock,MATOP_MULT,(void(*)(void))StateMatBlockMult));
  PetscCall(MatShellSetOperation(user->JsBlock,MATOP_MULT_TRANSPOSE,(void(*)(void))StateMatBlockMultTranspose));

  /* Create a matrix-free shell user->JsBlockPrec for computing (U+D)\D*(L+D)\x, where JsBlock = L+D+U,
     D is diagonal, L is strictly lower triangular, and U is strictly upper triangular.
     This is an SOR preconditioner for user->JsBlock. */
  PetscCall(MatCreateShell(PETSC_COMM_WORLD,PETSC_DETERMINE,PETSC_DETERMINE,n,n,user,&user->JsBlockPrec));
  PetscCall(MatShellSetOperation(user->JsBlockPrec,MATOP_MULT,(void(*)(void))StateMatBlockPrecMult));
  PetscCall(MatShellSetOperation(user->JsBlockPrec,MATOP_MULT_TRANSPOSE,(void(*)(void))StateMatBlockPrecMultTranspose));

  /* Create a matrix-free shell user->Jd for computing B*x */
  PetscCall(MatCreateShell(PETSC_COMM_WORLD,PETSC_DETERMINE,PETSC_DETERMINE,user->m,user->n-user->m,user,&user->Jd));
  PetscCall(MatShellSetOperation(user->Jd,MATOP_MULT,(void(*)(void))DesignMatMult));
  PetscCall(MatShellSetOperation(user->Jd,MATOP_MULT_TRANSPOSE,(void(*)(void))DesignMatMultTranspose));

  /* User-defined routines for computing user->Js\x and user->Js^T\x*/
  PetscCall(MatCreateShell(PETSC_COMM_WORLD,PETSC_DETERMINE,PETSC_DETERMINE,user->m,user->m,user,&user->JsInv));
  PetscCall(MatShellSetOperation(user->JsInv,MATOP_MULT,(void(*)(void))StateMatInvMult));
  PetscCall(MatShellSetOperation(user->JsInv,MATOP_MULT_TRANSPOSE,(void(*)(void))StateMatInvTransposeMult));

  /* Build matrices for SOR preconditioner */
  PetscCall(Scatter_uxi_uyi(user->u,user->uxi,user->uxi_scatter,user->uyi,user->uyi_scatter,user->nt));
  PetscCall(PetscMalloc1(5*n,&user->C));
  PetscCall(PetscMalloc1(2*n,&user->Cwork));
  for (i=0; i<user->nt; i++) {
    PetscCall(MatDuplicate(user->Divxy[0],MAT_COPY_VALUES,&user->C[i]));
    PetscCall(MatDuplicate(user->Divxy[1],MAT_COPY_VALUES,&user->Cwork[i]));

    PetscCall(MatDiagonalScale(user->C[i],NULL,user->uxi[i]));
    PetscCall(MatDiagonalScale(user->Cwork[i],NULL,user->uyi[i]));
    PetscCall(MatAXPY(user->C[i],1.0,user->Cwork[i],DIFFERENT_NONZERO_PATTERN));
    PetscCall(MatScale(user->C[i],user->ht));
    PetscCall(MatShift(user->C[i],1.0));
  }

  /* Solver options and tolerances */
  PetscCall(KSPCreate(PETSC_COMM_WORLD,&user->solver));
  PetscCall(KSPSetType(user->solver,KSPGMRES));
  PetscCall(KSPSetOperators(user->solver,user->JsBlock,user->JsBlockPrec));
  PetscCall(KSPSetTolerances(user->solver,1e-4,1e-20,1e3,500));
  /* PetscCall(KSPSetTolerances(user->solver,1e-8,1e-16,1e3,500)); */
  PetscCall(KSPGetPC(user->solver,&user->prec));
  PetscCall(PCSetType(user->prec,PCSHELL));

  PetscCall(PCShellSetApply(user->prec,StateMatBlockPrecMult));
  PetscCall(PCShellSetApplyTranspose(user->prec,StateMatBlockPrecMultTranspose));
  PetscCall(PCShellSetContext(user->prec,user));

  /* Compute true state function yt given ut */
  PetscCall(VecCreate(PETSC_COMM_WORLD,&user->ytrue));
  PetscCall(VecSetSizes(user->ytrue,PETSC_DECIDE,n*user->nt));
  PetscCall(VecSetFromOptions(user->ytrue));
  user->c_formed = PETSC_TRUE;
  PetscCall(VecCopy(user->utrue,user->u)); /*  Set u=utrue temporarily for StateMatInv */
  PetscCall(VecSet(user->ytrue,0.0)); /*  Initial guess */
  PetscCall(StateMatInvMult(user->Js,user->q,user->ytrue));
  PetscCall(VecCopy(user->ur,user->u)); /*  Reset u=ur */

  /* Initial guess y0 for state given u0 */
  PetscCall(StateMatInvMult(user->Js,user->q,user->y));

  /* Data discretization */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&user->Q));
  PetscCall(MatSetSizes(user->Q,PETSC_DECIDE,PETSC_DECIDE,user->mx*user->mx,user->m));
  PetscCall(MatSetFromOptions(user->Q));
  PetscCall(MatMPIAIJSetPreallocation(user->Q,0,NULL,1,NULL));
  PetscCall(MatSeqAIJSetPreallocation(user->Q,1,NULL));

  PetscCall(MatGetOwnershipRange(user->Q,&istart,&iend));

  for (i=istart; i<iend; i++) {
    j = i + user->m - user->mx*user->mx;
    PetscCall(MatSetValues(user->Q,1,&i,1,&j,&one,INSERT_VALUES));
  }

  PetscCall(MatAssemblyBegin(user->Q,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(user->Q,MAT_FINAL_ASSEMBLY));

  PetscCall(MatTranspose(user->Q,MAT_INITIAL_MATRIX,&user->QT));

  PetscCall(VecDestroy(&XX));
  PetscCall(VecDestroy(&YY));
  PetscCall(VecDestroy(&XXwork));
  PetscCall(VecDestroy(&YYwork));
  PetscCall(VecDestroy(&yi));
  PetscCall(VecDestroy(&uxi));
  PetscCall(VecDestroy(&ui));
  PetscCall(VecDestroy(&bc));

  /* Now that initial conditions have been set, let the user pass tolerance options to the KSP solver */
  PetscCall(KSPSetFromOptions(user->solver));
  PetscFunctionReturn(0);
}

PetscErrorCode HyperbolicDestroy(AppCtx *user)
{
  PetscInt       i;

  PetscFunctionBegin;
  PetscCall(MatDestroy(&user->Q));
  PetscCall(MatDestroy(&user->QT));
  PetscCall(MatDestroy(&user->Div));
  PetscCall(MatDestroy(&user->Divwork));
  PetscCall(MatDestroy(&user->Grad));
  PetscCall(MatDestroy(&user->L));
  PetscCall(MatDestroy(&user->LT));
  PetscCall(KSPDestroy(&user->solver));
  PetscCall(MatDestroy(&user->Js));
  PetscCall(MatDestroy(&user->Jd));
  PetscCall(MatDestroy(&user->JsBlockPrec));
  PetscCall(MatDestroy(&user->JsInv));
  PetscCall(MatDestroy(&user->JsBlock));
  PetscCall(MatDestroy(&user->Divxy[0]));
  PetscCall(MatDestroy(&user->Divxy[1]));
  PetscCall(MatDestroy(&user->Gradxy[0]));
  PetscCall(MatDestroy(&user->Gradxy[1]));
  PetscCall(MatDestroy(&user->M));
  for (i=0; i<user->nt; i++) {
    PetscCall(MatDestroy(&user->C[i]));
    PetscCall(MatDestroy(&user->Cwork[i]));
  }
  PetscCall(PetscFree(user->C));
  PetscCall(PetscFree(user->Cwork));
  PetscCall(VecDestroy(&user->u));
  PetscCall(VecDestroy(&user->uwork));
  PetscCall(VecDestroy(&user->vwork));
  PetscCall(VecDestroy(&user->utrue));
  PetscCall(VecDestroy(&user->y));
  PetscCall(VecDestroy(&user->ywork));
  PetscCall(VecDestroy(&user->ytrue));
  PetscCall(VecDestroyVecs(user->nt,&user->yi));
  PetscCall(VecDestroyVecs(user->nt,&user->yiwork));
  PetscCall(VecDestroyVecs(user->nt,&user->ziwork));
  PetscCall(VecDestroyVecs(user->nt,&user->uxi));
  PetscCall(VecDestroyVecs(user->nt,&user->uyi));
  PetscCall(VecDestroyVecs(user->nt,&user->uxiwork));
  PetscCall(VecDestroyVecs(user->nt,&user->uyiwork));
  PetscCall(VecDestroyVecs(user->nt,&user->ui));
  PetscCall(VecDestroyVecs(user->nt,&user->uiwork));
  PetscCall(VecDestroy(&user->c));
  PetscCall(VecDestroy(&user->cwork));
  PetscCall(VecDestroy(&user->ur));
  PetscCall(VecDestroy(&user->q));
  PetscCall(VecDestroy(&user->d));
  PetscCall(VecDestroy(&user->dwork));
  PetscCall(VecDestroy(&user->lwork));
  PetscCall(VecDestroy(&user->js_diag));
  PetscCall(ISDestroy(&user->s_is));
  PetscCall(ISDestroy(&user->d_is));
  PetscCall(VecScatterDestroy(&user->state_scatter));
  PetscCall(VecScatterDestroy(&user->design_scatter));
  for (i=0; i<user->nt; i++) {
    PetscCall(VecScatterDestroy(&user->uxi_scatter[i]));
    PetscCall(VecScatterDestroy(&user->uyi_scatter[i]));
    PetscCall(VecScatterDestroy(&user->ux_scatter[i]));
    PetscCall(VecScatterDestroy(&user->uy_scatter[i]));
    PetscCall(VecScatterDestroy(&user->ui_scatter[i]));
    PetscCall(VecScatterDestroy(&user->yi_scatter[i]));
  }
  PetscCall(PetscFree(user->uxi_scatter));
  PetscCall(PetscFree(user->uyi_scatter));
  PetscCall(PetscFree(user->ux_scatter));
  PetscCall(PetscFree(user->uy_scatter));
  PetscCall(PetscFree(user->ui_scatter));
  PetscCall(PetscFree(user->yi_scatter));
  PetscFunctionReturn(0);
}

PetscErrorCode HyperbolicMonitor(Tao tao, void *ptr)
{
  Vec            X;
  PetscReal      unorm,ynorm;
  AppCtx         *user = (AppCtx*)ptr;

  PetscFunctionBegin;
  PetscCall(TaoGetSolution(tao,&X));
  PetscCall(Scatter(X,user->ywork,user->state_scatter,user->uwork,user->design_scatter));
  PetscCall(VecAXPY(user->ywork,-1.0,user->ytrue));
  PetscCall(VecAXPY(user->uwork,-1.0,user->utrue));
  PetscCall(VecNorm(user->uwork,NORM_2,&unorm));
  PetscCall(VecNorm(user->ywork,NORM_2,&ynorm));
  PetscCall(PetscPrintf(MPI_COMM_WORLD, "||u-ut||=%g ||y-yt||=%g\n",(double)unorm,(double)ynorm));
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
