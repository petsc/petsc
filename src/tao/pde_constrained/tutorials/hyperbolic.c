#include <petsctao.h>

/*T
   Concepts: TAO^Solving a system of nonlinear equations, nonlinear least squares
   Routines: TaoCreate();
   Routines: TaoSetType();
   Routines: TaoSetInitialVector();
   Routines: TaoSetObjectiveRoutine();
   Routines: TaoSetGradientRoutine();
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
  ierr = PetscOptionsInt("-mx","Number of grid points in each direction","",user.mx,&user.mx,NULL);CHKERRQ(ierr);
  user.nt = 16;
  ierr = PetscOptionsInt("-nt","Number of time steps","",user.nt,&user.nt,NULL);CHKERRQ(ierr);
  user.ndata = 64;
  ierr = PetscOptionsInt("-ndata","Numbers of data points per sample","",user.ndata,&user.ndata,NULL);CHKERRQ(ierr);
  user.alpha = 10.0;
  ierr = PetscOptionsReal("-alpha","Regularization parameter","",user.alpha,&user.alpha,NULL);CHKERRQ(ierr);
  user.T = 1.0/32.0;
  ierr = PetscOptionsReal("-Tfinal","Final time","",user.T,&user.T,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-ntests","Number of times to repeat TaoSolve","",ntests,&ntests,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  user.m = user.mx*user.mx*user.nt; /*  number of constraints */
  user.n = user.mx*user.mx*3*user.nt; /*  number of variables */
  user.ht = user.T/user.nt; /*  Time step */
  user.gamma = user.T*user.ht / (user.mx*user.mx);

  ierr = VecCreate(PETSC_COMM_WORLD,&user.u);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&user.y);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&user.c);CHKERRQ(ierr);
  ierr = VecSetSizes(user.u,PETSC_DECIDE,user.n-user.m);CHKERRQ(ierr);
  ierr = VecSetSizes(user.y,PETSC_DECIDE,user.m);CHKERRQ(ierr);
  ierr = VecSetSizes(user.c,PETSC_DECIDE,user.m);CHKERRQ(ierr);
  ierr = VecSetFromOptions(user.u);CHKERRQ(ierr);
  ierr = VecSetFromOptions(user.y);CHKERRQ(ierr);
  ierr = VecSetFromOptions(user.c);CHKERRQ(ierr);

  /* Create scatters for reduced spaces.
     If the state vector y and design vector u are partitioned as
     [y_1; y_2; ...; y_np] and [u_1; u_2; ...; u_np] (with np = # of processors),
     then the solution vector x is organized as
     [y_1; u_1; y_2; u_2; ...; y_np; u_np].
     The index sets user.s_is and user.d_is correspond to the indices of the
     state and design variables owned by the current processor.
  */
  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);

  ierr = VecGetOwnershipRange(user.y,&lo,&hi);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(user.u,&lo2,&hi2);CHKERRQ(ierr);

  ierr = ISCreateStride(PETSC_COMM_SELF,hi-lo,lo,1,&is_allstate);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_SELF,hi-lo,lo+lo2,1,&user.s_is);CHKERRQ(ierr);

  ierr = ISCreateStride(PETSC_COMM_SELF,hi2-lo2,lo2,1,&is_alldesign);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_SELF,hi2-lo2,hi+lo2,1,&user.d_is);CHKERRQ(ierr);

  ierr = VecSetSizes(x,hi-lo+hi2-lo2,user.n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);

  ierr = VecScatterCreate(x,user.s_is,user.y,is_allstate,&user.state_scatter);CHKERRQ(ierr);
  ierr = VecScatterCreate(x,user.d_is,user.u,is_alldesign,&user.design_scatter);CHKERRQ(ierr);
  ierr = ISDestroy(&is_alldesign);CHKERRQ(ierr);
  ierr = ISDestroy(&is_allstate);CHKERRQ(ierr);

  /* Create TAO solver and set desired solution method */
  ierr = TaoCreate(PETSC_COMM_WORLD,&tao);CHKERRQ(ierr);
  ierr = TaoSetType(tao,TAOLCL);CHKERRQ(ierr);

  /* Set up initial vectors and matrices */
  ierr = HyperbolicInitialize(&user);CHKERRQ(ierr);

  ierr = Gather(x,user.y,user.state_scatter,user.u,user.design_scatter);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&x0);CHKERRQ(ierr);
  ierr = VecCopy(x,x0);CHKERRQ(ierr);

  /* Set solution vector with an initial guess */
  ierr = TaoSetInitialVector(tao,x);CHKERRQ(ierr);
  ierr = TaoSetObjectiveRoutine(tao, FormFunction, &user);CHKERRQ(ierr);
  ierr = TaoSetGradientRoutine(tao, FormGradient, &user);CHKERRQ(ierr);
  ierr = TaoSetConstraintsRoutine(tao, user.c, FormConstraints, &user);CHKERRQ(ierr);
  ierr = TaoSetJacobianStateRoutine(tao, user.Js, user.Js, user.JsInv, FormJacobianState, &user);CHKERRQ(ierr);
  ierr = TaoSetJacobianDesignRoutine(tao, user.Jd, FormJacobianDesign, &user);CHKERRQ(ierr);
  ierr = TaoSetFromOptions(tao);CHKERRQ(ierr);
  ierr = TaoSetStateDesignIS(tao,user.s_is,user.d_is);CHKERRQ(ierr);

  /* SOLVE THE APPLICATION */
  ierr = PetscLogStageRegister("Trials",&stages[0]);CHKERRQ(ierr);
  ierr = PetscLogStagePush(stages[0]);CHKERRQ(ierr);
  user.ksp_its_initial = user.ksp_its;
  ksp_old = user.ksp_its;
  for (i=0; i<ntests; i++) {
    ierr = TaoSolve(tao);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"KSP Iterations = %D\n",user.ksp_its-ksp_old);CHKERRQ(ierr);
    ierr = VecCopy(x0,x);CHKERRQ(ierr);
    ierr = TaoSetInitialVector(tao,x);CHKERRQ(ierr);
  }
  ierr = PetscLogStagePop();CHKERRQ(ierr);
  ierr = PetscBarrier((PetscObject)x);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"KSP iterations within initialization: ");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"%D\n",user.ksp_its_initial);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Total KSP iterations over %D trial(s): ",ntests);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"%D\n",user.ksp_its);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"KSP iterations per trial: ");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"%D\n",(user.ksp_its-user.ksp_its_initial)/ntests);CHKERRQ(ierr);

  ierr = TaoDestroy(&tao);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&x0);CHKERRQ(ierr);
  ierr = HyperbolicDestroy(&user);CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  PetscReal      d1=0,d2=0;
  AppCtx         *user = (AppCtx*)ptr;

  PetscFunctionBegin;
  ierr = Scatter(X,user->y,user->state_scatter,user->u,user->design_scatter);CHKERRQ(ierr);
  ierr = MatMult(user->Q,user->y,user->dwork);CHKERRQ(ierr);
  ierr = VecAXPY(user->dwork,-1.0,user->d);CHKERRQ(ierr);
  ierr = VecDot(user->dwork,user->dwork,&d1);CHKERRQ(ierr);

  ierr = VecWAXPY(user->uwork,-1.0,user->ur,user->u);CHKERRQ(ierr);
  ierr = VecPointwiseMult(user->uwork,user->uwork,user->uwork);CHKERRQ(ierr);
  ierr = MatMult(user->L,user->uwork,user->lwork);CHKERRQ(ierr);
  ierr = VecDot(user->y,user->lwork,&d2);CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  AppCtx         *user = (AppCtx*)ptr;

  PetscFunctionBegin;
  ierr = Scatter(X,user->y,user->state_scatter,user->u,user->design_scatter);CHKERRQ(ierr);
  ierr = MatMult(user->Q,user->y,user->dwork);CHKERRQ(ierr);
  ierr = VecAXPY(user->dwork,-1.0,user->d);CHKERRQ(ierr);

  ierr = MatMult(user->QT,user->dwork,user->ywork);CHKERRQ(ierr);

  ierr = MatMult(user->LT,user->y,user->uwork);CHKERRQ(ierr);
  ierr = VecWAXPY(user->vwork,-1.0,user->ur,user->u);CHKERRQ(ierr);
  ierr = VecPointwiseMult(user->uwork,user->vwork,user->uwork);CHKERRQ(ierr);
  ierr = VecScale(user->uwork,user->alpha);CHKERRQ(ierr);

  ierr = VecPointwiseMult(user->vwork,user->vwork,user->vwork);CHKERRQ(ierr);
  ierr = MatMult(user->L,user->vwork,user->lwork);CHKERRQ(ierr);
  ierr = VecAXPY(user->ywork,0.5*user->alpha,user->lwork);CHKERRQ(ierr);

  ierr = Gather(G,user->ywork,user->state_scatter,user->uwork,user->design_scatter);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode FormFunctionGradient(Tao tao, Vec X, PetscReal *f, Vec G, void *ptr)
{
  PetscErrorCode ierr;
  PetscReal      d1,d2;
  AppCtx         *user = (AppCtx*)ptr;

  PetscFunctionBegin;
  ierr = Scatter(X,user->y,user->state_scatter,user->u,user->design_scatter);CHKERRQ(ierr);
  ierr = MatMult(user->Q,user->y,user->dwork);CHKERRQ(ierr);
  ierr = VecAXPY(user->dwork,-1.0,user->d);CHKERRQ(ierr);

  ierr = MatMult(user->QT,user->dwork,user->ywork);CHKERRQ(ierr);

  ierr = VecDot(user->dwork,user->dwork,&d1);CHKERRQ(ierr);

  ierr = MatMult(user->LT,user->y,user->uwork);CHKERRQ(ierr);
  ierr = VecWAXPY(user->vwork,-1.0,user->ur,user->u);CHKERRQ(ierr);
  ierr = VecPointwiseMult(user->uwork,user->vwork,user->uwork);CHKERRQ(ierr);
  ierr = VecScale(user->uwork,user->alpha);CHKERRQ(ierr);

  ierr = VecPointwiseMult(user->vwork,user->vwork,user->vwork);CHKERRQ(ierr);
  ierr = MatMult(user->L,user->vwork,user->lwork);CHKERRQ(ierr);
  ierr = VecAXPY(user->ywork,0.5*user->alpha,user->lwork);CHKERRQ(ierr);

  ierr = VecDot(user->y,user->lwork,&d2);CHKERRQ(ierr);

  *f = 0.5 * (d1 + user->alpha*d2);
  ierr = Gather(G,user->ywork,user->state_scatter,user->uwork,user->design_scatter);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
/* A
MatShell object
*/
PetscErrorCode FormJacobianState(Tao tao, Vec X, Mat J, Mat JPre, Mat JInv, void *ptr)
{
  PetscErrorCode ierr;
  PetscInt       i;
  AppCtx         *user = (AppCtx*)ptr;

  PetscFunctionBegin;
  ierr = Scatter(X,user->y,user->state_scatter,user->u,user->design_scatter);CHKERRQ(ierr);
  ierr = Scatter_yi(user->u,user->ui,user->ui_scatter,user->nt);CHKERRQ(ierr);
  ierr = Scatter_uxi_uyi(user->u,user->uxi,user->uxi_scatter,user->uyi,user->uyi_scatter,user->nt);CHKERRQ(ierr);
  for (i=0; i<user->nt; i++) {
    ierr = MatCopy(user->Divxy[0],user->C[i],SUBSET_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatCopy(user->Divxy[1],user->Cwork[i],SAME_NONZERO_PATTERN);CHKERRQ(ierr);

    ierr = MatDiagonalScale(user->C[i],NULL,user->uxi[i]);CHKERRQ(ierr);
    ierr = MatDiagonalScale(user->Cwork[i],NULL,user->uyi[i]);CHKERRQ(ierr);
    ierr = MatAXPY(user->C[i],1.0,user->Cwork[i],SUBSET_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatScale(user->C[i],user->ht);CHKERRQ(ierr);
    ierr = MatShift(user->C[i],1.0);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
/* B */
PetscErrorCode FormJacobianDesign(Tao tao, Vec X, Mat J, void *ptr)
{
  PetscErrorCode ierr;
  AppCtx         *user = (AppCtx*)ptr;

  PetscFunctionBegin;
  ierr = Scatter(X,user->y,user->state_scatter,user->u,user->design_scatter);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode StateMatMult(Mat J_shell, Vec X, Vec Y)
{
  PetscErrorCode ierr;
  PetscInt       i;
  AppCtx         *user;

  PetscFunctionBegin;
  ierr = MatShellGetContext(J_shell,&user);CHKERRQ(ierr);
  ierr = Scatter_yi(X,user->yi,user->yi_scatter,user->nt);CHKERRQ(ierr);
  user->block_index = 0;
  ierr = MatMult(user->JsBlock,user->yi[0],user->yiwork[0]);CHKERRQ(ierr);

  for (i=1; i<user->nt; i++) {
    user->block_index = i;
    ierr = MatMult(user->JsBlock,user->yi[i],user->yiwork[i]);CHKERRQ(ierr);
    ierr = MatMult(user->M,user->yi[i-1],user->ziwork[i-1]);CHKERRQ(ierr);
    ierr = VecAXPY(user->yiwork[i],-1.0,user->ziwork[i-1]);CHKERRQ(ierr);
  }
  ierr = Gather_yi(Y,user->yiwork,user->yi_scatter,user->nt);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode StateMatMultTranspose(Mat J_shell, Vec X, Vec Y)
{
  PetscErrorCode ierr;
  PetscInt       i;
  AppCtx         *user;

  PetscFunctionBegin;
  ierr = MatShellGetContext(J_shell,&user);CHKERRQ(ierr);
  ierr = Scatter_yi(X,user->yi,user->yi_scatter,user->nt);CHKERRQ(ierr);

  for (i=0; i<user->nt-1; i++) {
    user->block_index = i;
    ierr = MatMultTranspose(user->JsBlock,user->yi[i],user->yiwork[i]);CHKERRQ(ierr);
    ierr = MatMult(user->M,user->yi[i+1],user->ziwork[i+1]);CHKERRQ(ierr);
    ierr = VecAXPY(user->yiwork[i],-1.0,user->ziwork[i+1]);CHKERRQ(ierr);
  }

  i = user->nt-1;
  user->block_index = i;
  ierr = MatMultTranspose(user->JsBlock,user->yi[i],user->yiwork[i]);CHKERRQ(ierr);
  ierr = Gather_yi(Y,user->yiwork,user->yi_scatter,user->nt);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode StateMatBlockMult(Mat J_shell, Vec X, Vec Y)
{
  PetscErrorCode ierr;
  PetscInt       i;
  AppCtx         *user;

  PetscFunctionBegin;
  ierr = MatShellGetContext(J_shell,&user);CHKERRQ(ierr);
  i = user->block_index;
  ierr = VecPointwiseMult(user->uxiwork[i],X,user->uxi[i]);CHKERRQ(ierr);
  ierr = VecPointwiseMult(user->uyiwork[i],X,user->uyi[i]);CHKERRQ(ierr);
  ierr = Gather(user->uiwork[i],user->uxiwork[i],user->ux_scatter[i],user->uyiwork[i],user->uy_scatter[i]);CHKERRQ(ierr);
  ierr = MatMult(user->Div,user->uiwork[i],Y);CHKERRQ(ierr);
  ierr = VecAYPX(Y,user->ht,X);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode StateMatBlockMultTranspose(Mat J_shell, Vec X, Vec Y)
{
  PetscErrorCode ierr;
  PetscInt       i;
  AppCtx         *user;

  PetscFunctionBegin;
  ierr = MatShellGetContext(J_shell,&user);CHKERRQ(ierr);
  i = user->block_index;
  ierr = MatMult(user->Grad,X,user->uiwork[i]);CHKERRQ(ierr);
  ierr = Scatter(user->uiwork[i],user->uxiwork[i],user->ux_scatter[i],user->uyiwork[i],user->uy_scatter[i]);CHKERRQ(ierr);
  ierr = VecPointwiseMult(user->uxiwork[i],user->uxi[i],user->uxiwork[i]);CHKERRQ(ierr);
  ierr = VecPointwiseMult(user->uyiwork[i],user->uyi[i],user->uyiwork[i]);CHKERRQ(ierr);
  ierr = VecWAXPY(Y,1.0,user->uxiwork[i],user->uyiwork[i]);CHKERRQ(ierr);
  ierr = VecAYPX(Y,user->ht,X);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DesignMatMult(Mat J_shell, Vec X, Vec Y)
{
  PetscErrorCode ierr;
  PetscInt       i;
  AppCtx         *user;

  PetscFunctionBegin;
  ierr = MatShellGetContext(J_shell,&user);CHKERRQ(ierr);
  ierr = Scatter_yi(user->y,user->yi,user->yi_scatter,user->nt);CHKERRQ(ierr);
  ierr = Scatter_uxi_uyi(X,user->uxiwork,user->uxi_scatter,user->uyiwork,user->uyi_scatter,user->nt);CHKERRQ(ierr);
  for (i=0; i<user->nt; i++) {
    ierr = VecPointwiseMult(user->uxiwork[i],user->yi[i],user->uxiwork[i]);CHKERRQ(ierr);
    ierr = VecPointwiseMult(user->uyiwork[i],user->yi[i],user->uyiwork[i]);CHKERRQ(ierr);
    ierr = Gather(user->uiwork[i],user->uxiwork[i],user->ux_scatter[i],user->uyiwork[i],user->uy_scatter[i]);CHKERRQ(ierr);
    ierr = MatMult(user->Div,user->uiwork[i],user->ziwork[i]);CHKERRQ(ierr);
    ierr = VecScale(user->ziwork[i],user->ht);CHKERRQ(ierr);
  }
  ierr = Gather_yi(Y,user->ziwork,user->yi_scatter,user->nt);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DesignMatMultTranspose(Mat J_shell, Vec X, Vec Y)
{
  PetscErrorCode ierr;
  PetscInt       i;
  AppCtx         *user;

  PetscFunctionBegin;
  ierr = MatShellGetContext(J_shell,&user);CHKERRQ(ierr);
  ierr = Scatter_yi(user->y,user->yi,user->yi_scatter,user->nt);CHKERRQ(ierr);
  ierr = Scatter_yi(X,user->yiwork,user->yi_scatter,user->nt);CHKERRQ(ierr);
  for (i=0; i<user->nt; i++) {
    ierr = MatMult(user->Grad,user->yiwork[i],user->uiwork[i]);CHKERRQ(ierr);
    ierr = Scatter(user->uiwork[i],user->uxiwork[i],user->ux_scatter[i],user->uyiwork[i],user->uy_scatter[i]);CHKERRQ(ierr);
    ierr = VecPointwiseMult(user->uxiwork[i],user->yi[i],user->uxiwork[i]);CHKERRQ(ierr);
    ierr = VecPointwiseMult(user->uyiwork[i],user->yi[i],user->uyiwork[i]);CHKERRQ(ierr);
    ierr = Gather(user->uiwork[i],user->uxiwork[i],user->ux_scatter[i],user->uyiwork[i],user->uy_scatter[i]);CHKERRQ(ierr);
    ierr = VecScale(user->uiwork[i],user->ht);CHKERRQ(ierr);
  }
  ierr = Gather_yi(Y,user->uiwork,user->ui_scatter,user->nt);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode StateMatBlockPrecMult(PC PC_shell, Vec X, Vec Y)
{
  PetscErrorCode ierr;
  PetscInt       i;
  AppCtx         *user;

  PetscFunctionBegin;
  ierr = PCShellGetContext(PC_shell,&user);CHKERRQ(ierr);
  i = user->block_index;
  if (user->c_formed) {
    ierr = MatSOR(user->C[i],X,1.0,(MatSORType)(SOR_ZERO_INITIAL_GUESS | SOR_LOCAL_SYMMETRIC_SWEEP),0.0,1,1,Y);CHKERRQ(ierr);
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not formed");
  PetscFunctionReturn(0);
}

PetscErrorCode StateMatBlockPrecMultTranspose(PC PC_shell, Vec X, Vec Y)
{
  PetscErrorCode ierr;
  PetscInt       i;
  AppCtx         *user;

  PetscFunctionBegin;
  ierr = PCShellGetContext(PC_shell,&user);CHKERRQ(ierr);

  i = user->block_index;
  if (user->c_formed) {
    ierr = MatSOR(user->C[i],X,1.0,(MatSORType)(SOR_ZERO_INITIAL_GUESS | SOR_LOCAL_SYMMETRIC_SWEEP),0.0,1,1,Y);CHKERRQ(ierr);
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not formed");
  PetscFunctionReturn(0);
}

PetscErrorCode StateMatInvMult(Mat J_shell, Vec X, Vec Y)
{
  PetscErrorCode ierr;
  AppCtx         *user;
  PetscInt       its,i;

  PetscFunctionBegin;
  ierr = MatShellGetContext(J_shell,&user);CHKERRQ(ierr);

  if (Y == user->ytrue) {
    /* First solve is done using true solution to set up problem */
    ierr = KSPSetTolerances(user->solver,1e-4,1e-20,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
  } else {
    ierr = KSPSetTolerances(user->solver,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
  }
  ierr = Scatter_yi(X,user->yi,user->yi_scatter,user->nt);CHKERRQ(ierr);
  ierr = Scatter_yi(Y,user->yiwork,user->yi_scatter,user->nt);CHKERRQ(ierr);
  ierr = Scatter_uxi_uyi(user->u,user->uxi,user->uxi_scatter,user->uyi,user->uyi_scatter,user->nt);CHKERRQ(ierr);

  user->block_index = 0;
  ierr = KSPSolve(user->solver,user->yi[0],user->yiwork[0]);CHKERRQ(ierr);

  ierr = KSPGetIterationNumber(user->solver,&its);CHKERRQ(ierr);
  user->ksp_its = user->ksp_its + its;
  for (i=1; i<user->nt; i++) {
    ierr = MatMult(user->M,user->yiwork[i-1],user->ziwork[i-1]);CHKERRQ(ierr);
    ierr = VecAXPY(user->yi[i],1.0,user->ziwork[i-1]);CHKERRQ(ierr);
    user->block_index = i;
    ierr = KSPSolve(user->solver,user->yi[i],user->yiwork[i]);CHKERRQ(ierr);

    ierr = KSPGetIterationNumber(user->solver,&its);CHKERRQ(ierr);
    user->ksp_its = user->ksp_its + its;
  }
  ierr = Gather_yi(Y,user->yiwork,user->yi_scatter,user->nt);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode StateMatInvTransposeMult(Mat J_shell, Vec X, Vec Y)
{
  PetscErrorCode ierr;
  AppCtx         *user;
  PetscInt       its,i;

  PetscFunctionBegin;
  ierr = MatShellGetContext(J_shell,&user);CHKERRQ(ierr);

  ierr = Scatter_yi(X,user->yi,user->yi_scatter,user->nt);CHKERRQ(ierr);
  ierr = Scatter_yi(Y,user->yiwork,user->yi_scatter,user->nt);CHKERRQ(ierr);
  ierr = Scatter_uxi_uyi(user->u,user->uxi,user->uxi_scatter,user->uyi,user->uyi_scatter,user->nt);CHKERRQ(ierr);

  i = user->nt - 1;
  user->block_index = i;
  ierr = KSPSolveTranspose(user->solver,user->yi[i],user->yiwork[i]);CHKERRQ(ierr);

  ierr = KSPGetIterationNumber(user->solver,&its);CHKERRQ(ierr);
  user->ksp_its = user->ksp_its + its;

  for (i=user->nt-2; i>=0; i--) {
    ierr = MatMult(user->M,user->yiwork[i+1],user->ziwork[i+1]);CHKERRQ(ierr);
    ierr = VecAXPY(user->yi[i],1.0,user->ziwork[i+1]);CHKERRQ(ierr);
    user->block_index = i;
    ierr = KSPSolveTranspose(user->solver,user->yi[i],user->yiwork[i]);CHKERRQ(ierr);

    ierr = KSPGetIterationNumber(user->solver,&its);CHKERRQ(ierr);
    user->ksp_its = user->ksp_its + its;
  }
  ierr = Gather_yi(Y,user->yiwork,user->yi_scatter,user->nt);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode StateMatDuplicate(Mat J_shell, MatDuplicateOption opt, Mat *new_shell)
{
  PetscErrorCode ierr;
  AppCtx         *user;

  PetscFunctionBegin;
  ierr = MatShellGetContext(J_shell,&user);CHKERRQ(ierr);

  ierr = MatCreateShell(PETSC_COMM_WORLD,PETSC_DETERMINE,PETSC_DETERMINE,user->m,user->m,user,new_shell);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*new_shell,MATOP_MULT,(void(*)(void))StateMatMult);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*new_shell,MATOP_DUPLICATE,(void(*)(void))StateMatDuplicate);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*new_shell,MATOP_MULT_TRANSPOSE,(void(*)(void))StateMatMultTranspose);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*new_shell,MATOP_GET_DIAGONAL,(void(*)(void))StateMatGetDiagonal);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode StateMatGetDiagonal(Mat J_shell, Vec X)
{
  PetscErrorCode ierr;
  AppCtx         *user;

  PetscFunctionBegin;
  ierr = MatShellGetContext(J_shell,&user);CHKERRQ(ierr);
  ierr =  VecCopy(user->js_diag,X);CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  PetscInt       i;
  AppCtx         *user = (AppCtx*)ptr;

  PetscFunctionBegin;
  ierr = Scatter(X,user->y,user->state_scatter,user->u,user->design_scatter);CHKERRQ(ierr);
  ierr = Scatter_yi(user->y,user->yi,user->yi_scatter,user->nt);CHKERRQ(ierr);
  ierr = Scatter_uxi_uyi(user->u,user->uxi,user->uxi_scatter,user->uyi,user->uyi_scatter,user->nt);CHKERRQ(ierr);

  user->block_index = 0;
  ierr = MatMult(user->JsBlock,user->yi[0],user->yiwork[0]);CHKERRQ(ierr);

  for (i=1; i<user->nt; i++) {
    user->block_index = i;
    ierr = MatMult(user->JsBlock,user->yi[i],user->yiwork[i]);CHKERRQ(ierr);
    ierr = MatMult(user->M,user->yi[i-1],user->ziwork[i-1]);CHKERRQ(ierr);
    ierr = VecAXPY(user->yiwork[i],-1.0,user->ziwork[i-1]);CHKERRQ(ierr);
  }

  ierr = Gather_yi(C,user->yiwork,user->yi_scatter,user->nt);CHKERRQ(ierr);
  ierr = VecAXPY(C,-1.0,user->q);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode Scatter(Vec x, Vec state, VecScatter s_scat, Vec design, VecScatter d_scat)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecScatterBegin(s_scat,x,state,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(s_scat,x,state,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterBegin(d_scat,x,design,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(d_scat,x,design,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode Scatter_uxi_uyi(Vec u, Vec *uxi, VecScatter *scatx, Vec *uyi, VecScatter *scaty, PetscInt nt)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  for (i=0; i<nt; i++) {
    ierr = VecScatterBegin(scatx[i],u,uxi[i],INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(scatx[i],u,uxi[i],INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterBegin(scaty[i],u,uyi[i],INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(scaty[i],u,uyi[i],INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode Gather(Vec x, Vec state, VecScatter s_scat, Vec design, VecScatter d_scat)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecScatterBegin(s_scat,state,x,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(s_scat,state,x,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterBegin(d_scat,design,x,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(d_scat,design,x,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode Gather_uxi_uyi(Vec u, Vec *uxi, VecScatter *scatx, Vec *uyi, VecScatter *scaty, PetscInt nt)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  for (i=0; i<nt; i++) {
    ierr = VecScatterBegin(scatx[i],uxi[i],u,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(scatx[i],uxi[i],u,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterBegin(scaty[i],uyi[i],u,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(scaty[i],uyi[i],u,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode Scatter_yi(Vec y, Vec *yi, VecScatter *scat, PetscInt nt)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  for (i=0; i<nt; i++) {
    ierr = VecScatterBegin(scat[i],y,yi[i],INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(scat[i],y,yi[i],INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode Gather_yi(Vec y, Vec *yi, VecScatter *scat, PetscInt nt)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  for (i=0; i<nt; i++) {
    ierr = VecScatterBegin(scat[i],yi[i],y,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(scat[i],yi[i],y,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode HyperbolicInitialize(AppCtx *user)
{
  PetscErrorCode ierr;
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
  ierr = MatCreate(PETSC_COMM_WORLD,&user->Grad);CHKERRQ(ierr);
  ierr = MatSetSizes(user->Grad,PETSC_DECIDE,PETSC_DECIDE,2*n,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(user->Grad);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(user->Grad,3,NULL,3,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(user->Grad,3,NULL);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(user->Grad,&istart,&iend);CHKERRQ(ierr);

  for (i=istart; i<iend; i++) {
    if (i<n) {
      iblock = i / user->mx;
      j = iblock*user->mx + ((i+user->mx-1) % user->mx);
      ierr = MatSetValues(user->Grad,1,&i,1,&j,&half_hinv,INSERT_VALUES);CHKERRQ(ierr);
      j = iblock*user->mx + ((i+1) % user->mx);
      ierr = MatSetValues(user->Grad,1,&i,1,&j,&neg_half_hinv,INSERT_VALUES);CHKERRQ(ierr);
    }
    if (i>=n) {
      j = (i - user->mx) % n;
      ierr = MatSetValues(user->Grad,1,&i,1,&j,&half_hinv,INSERT_VALUES);CHKERRQ(ierr);
      j = (j + 2*user->mx) % n;
      ierr = MatSetValues(user->Grad,1,&i,1,&j,&neg_half_hinv,INSERT_VALUES);CHKERRQ(ierr);
    }
  }

  ierr = MatAssemblyBegin(user->Grad,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(user->Grad,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&user->Gradxy[0]);CHKERRQ(ierr);
  ierr = MatSetSizes(user->Gradxy[0],PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(user->Gradxy[0]);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(user->Gradxy[0],3,NULL,3,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(user->Gradxy[0],3,NULL);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(user->Gradxy[0],&istart,&iend);CHKERRQ(ierr);

  for (i=istart; i<iend; i++) {
    iblock = i / user->mx;
    j = iblock*user->mx + ((i+user->mx-1) % user->mx);
    ierr = MatSetValues(user->Gradxy[0],1,&i,1,&j,&half_hinv,INSERT_VALUES);CHKERRQ(ierr);
    j = iblock*user->mx + ((i+1) % user->mx);
    ierr = MatSetValues(user->Gradxy[0],1,&i,1,&j,&neg_half_hinv,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatSetValues(user->Gradxy[0],1,&i,1,&i,&zero,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(user->Gradxy[0],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(user->Gradxy[0],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&user->Gradxy[1]);CHKERRQ(ierr);
  ierr = MatSetSizes(user->Gradxy[1],PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(user->Gradxy[1]);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(user->Gradxy[1],3,NULL,3,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(user->Gradxy[1],3,NULL);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(user->Gradxy[1],&istart,&iend);CHKERRQ(ierr);

  for (i=istart; i<iend; i++) {
    j = (i + n - user->mx) % n;
    ierr = MatSetValues(user->Gradxy[1],1,&i,1,&j,&half_hinv,INSERT_VALUES);CHKERRQ(ierr);
    j = (j + 2*user->mx) % n;
    ierr = MatSetValues(user->Gradxy[1],1,&i,1,&j,&neg_half_hinv,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatSetValues(user->Gradxy[1],1,&i,1,&i,&zero,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(user->Gradxy[1],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(user->Gradxy[1],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Generate Div matrix */
  ierr = MatTranspose(user->Grad,MAT_INITIAL_MATRIX,&user->Div);CHKERRQ(ierr);
  ierr = MatTranspose(user->Gradxy[0],MAT_INITIAL_MATRIX,&user->Divxy[0]);CHKERRQ(ierr);
  ierr = MatTranspose(user->Gradxy[1],MAT_INITIAL_MATRIX,&user->Divxy[1]);CHKERRQ(ierr);

  /* Off-diagonal averaging matrix */
  ierr = MatCreate(PETSC_COMM_WORLD,&user->M);CHKERRQ(ierr);
  ierr = MatSetSizes(user->M,PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(user->M);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(user->M,4,NULL,4,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(user->M,4,NULL);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(user->M,&istart,&iend);CHKERRQ(ierr);

  for (i=istart; i<iend; i++) {
    /* kron(Id,Av) */
    iblock = i / user->mx;
    j = iblock*user->mx + ((i+user->mx-1) % user->mx);
    ierr = MatSetValues(user->M,1,&i,1,&j,&quarter,INSERT_VALUES);CHKERRQ(ierr);
    j = iblock*user->mx + ((i+1) % user->mx);
    ierr = MatSetValues(user->M,1,&i,1,&j,&quarter,INSERT_VALUES);CHKERRQ(ierr);

    /* kron(Av,Id) */
    j = (i + user->mx) % n;
    ierr = MatSetValues(user->M,1,&i,1,&j,&quarter,INSERT_VALUES);CHKERRQ(ierr);
    j = (i + n - user->mx) % n;
    ierr = MatSetValues(user->M,1,&i,1,&j,&quarter,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(user->M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(user->M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Generate 2D grid */
  ierr = VecCreate(PETSC_COMM_WORLD,&XX);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&user->q);CHKERRQ(ierr);
  ierr = VecSetSizes(XX,PETSC_DECIDE,n);CHKERRQ(ierr);
  ierr = VecSetSizes(user->q,PETSC_DECIDE,n*user->nt);CHKERRQ(ierr);
  ierr = VecSetFromOptions(XX);CHKERRQ(ierr);
  ierr = VecSetFromOptions(user->q);CHKERRQ(ierr);

  ierr = VecDuplicate(XX,&YY);CHKERRQ(ierr);
  ierr = VecDuplicate(XX,&XXwork);CHKERRQ(ierr);
  ierr = VecDuplicate(XX,&YYwork);CHKERRQ(ierr);
  ierr = VecDuplicate(XX,&user->d);CHKERRQ(ierr);
  ierr = VecDuplicate(XX,&user->dwork);CHKERRQ(ierr);

  ierr = VecGetOwnershipRange(XX,&istart,&iend);CHKERRQ(ierr);
  for (linear_index=istart; linear_index<iend; linear_index++) {
    i = linear_index % user->mx;
    j = (linear_index-i)/user->mx;
    vx = h*(i+0.5);
    vy = h*(j+0.5);
    ierr = VecSetValues(XX,1,&linear_index,&vx,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValues(YY,1,&linear_index,&vy,INSERT_VALUES);CHKERRQ(ierr);
  }

  ierr = VecAssemblyBegin(XX);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(XX);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(YY);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(YY);CHKERRQ(ierr);

  /* Compute final density function yT
     yT = 1.0 + exp(-30*((x-0.25)^2+(y-0.25)^2)) + exp(-30*((x-0.75)^2+(y-0.75)^2))
     yT = yT / (h^2*sum(yT)) */
  ierr = VecCopy(XX,XXwork);CHKERRQ(ierr);
  ierr = VecCopy(YY,YYwork);CHKERRQ(ierr);

  ierr = VecShift(XXwork,-0.25);CHKERRQ(ierr);
  ierr = VecShift(YYwork,-0.25);CHKERRQ(ierr);

  ierr = VecPointwiseMult(XXwork,XXwork,XXwork);CHKERRQ(ierr);
  ierr = VecPointwiseMult(YYwork,YYwork,YYwork);CHKERRQ(ierr);

  ierr = VecCopy(XXwork,user->dwork);CHKERRQ(ierr);
  ierr = VecAXPY(user->dwork,1.0,YYwork);CHKERRQ(ierr);
  ierr = VecScale(user->dwork,-30.0);CHKERRQ(ierr);
  ierr = VecExp(user->dwork);CHKERRQ(ierr);
  ierr = VecCopy(user->dwork,user->d);CHKERRQ(ierr);

  ierr = VecCopy(XX,XXwork);CHKERRQ(ierr);
  ierr = VecCopy(YY,YYwork);CHKERRQ(ierr);

  ierr = VecShift(XXwork,-0.75);CHKERRQ(ierr);
  ierr = VecShift(YYwork,-0.75);CHKERRQ(ierr);

  ierr = VecPointwiseMult(XXwork,XXwork,XXwork);CHKERRQ(ierr);
  ierr = VecPointwiseMult(YYwork,YYwork,YYwork);CHKERRQ(ierr);

  ierr = VecCopy(XXwork,user->dwork);CHKERRQ(ierr);
  ierr = VecAXPY(user->dwork,1.0,YYwork);CHKERRQ(ierr);
  ierr = VecScale(user->dwork,-30.0);CHKERRQ(ierr);
  ierr = VecExp(user->dwork);CHKERRQ(ierr);

  ierr = VecAXPY(user->d,1.0,user->dwork);CHKERRQ(ierr);
  ierr = VecShift(user->d,1.0);CHKERRQ(ierr);
  ierr = VecSum(user->d,&sum);CHKERRQ(ierr);
  ierr = VecScale(user->d,1.0/(h*h*sum));CHKERRQ(ierr);

  /* Initial conditions of forward problem */
  ierr = VecDuplicate(XX,&bc);CHKERRQ(ierr);
  ierr = VecCopy(XX,XXwork);CHKERRQ(ierr);
  ierr = VecCopy(YY,YYwork);CHKERRQ(ierr);

  ierr = VecShift(XXwork,-0.5);CHKERRQ(ierr);
  ierr = VecShift(YYwork,-0.5);CHKERRQ(ierr);

  ierr = VecPointwiseMult(XXwork,XXwork,XXwork);CHKERRQ(ierr);
  ierr = VecPointwiseMult(YYwork,YYwork,YYwork);CHKERRQ(ierr);

  ierr = VecWAXPY(bc,1.0,XXwork,YYwork);CHKERRQ(ierr);
  ierr = VecScale(bc,-50.0);CHKERRQ(ierr);
  ierr = VecExp(bc);CHKERRQ(ierr);
  ierr = VecShift(bc,1.0);CHKERRQ(ierr);
  ierr = VecSum(bc,&sum);CHKERRQ(ierr);
  ierr = VecScale(bc,1.0/(h*h*sum));CHKERRQ(ierr);

  /* Create scatter from y to y_1,y_2,...,y_nt */
  /*  TODO: Reorder for better parallelism. (This will require reordering Q and L as well.) */
  ierr = PetscMalloc1(user->nt*user->mx*user->mx,&user->yi_scatter);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&yi);CHKERRQ(ierr);
  ierr = VecSetSizes(yi,PETSC_DECIDE,user->mx*user->mx);CHKERRQ(ierr);
  ierr = VecSetFromOptions(yi);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(yi,user->nt,&user->yi);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(yi,user->nt,&user->yiwork);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(yi,user->nt,&user->ziwork);CHKERRQ(ierr);
  for (i=0; i<user->nt; i++) {
    ierr = VecGetOwnershipRange(user->yi[i],&lo,&hi);CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF,hi-lo,lo,1,&is_to_yi);CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF,hi-lo,lo+i*user->mx*user->mx,1,&is_from_y);CHKERRQ(ierr);
    ierr = VecScatterCreate(user->y,is_from_y,user->yi[i],is_to_yi,&user->yi_scatter[i]);CHKERRQ(ierr);
    ierr = ISDestroy(&is_to_yi);CHKERRQ(ierr);
    ierr = ISDestroy(&is_from_y);CHKERRQ(ierr);
  }

  /* Create scatter from u to ux_1,uy_1,ux_2,uy_2,...,ux_nt,uy_nt */
  /*  TODO: reorder for better parallelism */
  ierr = PetscMalloc1(user->nt*user->mx*user->mx,&user->uxi_scatter);CHKERRQ(ierr);
  ierr = PetscMalloc1(user->nt*user->mx*user->mx,&user->uyi_scatter);CHKERRQ(ierr);
  ierr = PetscMalloc1(user->nt*user->mx*user->mx,&user->ux_scatter);CHKERRQ(ierr);
  ierr = PetscMalloc1(user->nt*user->mx*user->mx,&user->uy_scatter);CHKERRQ(ierr);
  ierr = PetscMalloc1(2*user->nt*user->mx*user->mx,&user->ui_scatter);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&uxi);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&ui);CHKERRQ(ierr);
  ierr = VecSetSizes(uxi,PETSC_DECIDE,user->mx*user->mx);CHKERRQ(ierr);
  ierr = VecSetSizes(ui,PETSC_DECIDE,2*user->mx*user->mx);CHKERRQ(ierr);
  ierr = VecSetFromOptions(uxi);CHKERRQ(ierr);
  ierr = VecSetFromOptions(ui);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(uxi,user->nt,&user->uxi);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(uxi,user->nt,&user->uyi);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(uxi,user->nt,&user->uxiwork);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(uxi,user->nt,&user->uyiwork);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(ui,user->nt,&user->ui);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(ui,user->nt,&user->uiwork);CHKERRQ(ierr);
  for (i=0; i<user->nt; i++) {
    ierr = VecGetOwnershipRange(user->uxi[i],&lo,&hi);CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF,hi-lo,lo,1,&is_to_uxi);CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF,hi-lo,lo+2*i*user->mx*user->mx,1,&is_from_u);CHKERRQ(ierr);
    ierr = VecScatterCreate(user->u,is_from_u,user->uxi[i],is_to_uxi,&user->uxi_scatter[i]);CHKERRQ(ierr);

    ierr = ISDestroy(&is_to_uxi);CHKERRQ(ierr);
    ierr = ISDestroy(&is_from_u);CHKERRQ(ierr);

    ierr = VecGetOwnershipRange(user->uyi[i],&lo,&hi);CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF,hi-lo,lo,1,&is_to_uyi);CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF,hi-lo,lo+(2*i+1)*user->mx*user->mx,1,&is_from_u);CHKERRQ(ierr);
    ierr = VecScatterCreate(user->u,is_from_u,user->uyi[i],is_to_uyi,&user->uyi_scatter[i]);CHKERRQ(ierr);

    ierr = ISDestroy(&is_to_uyi);CHKERRQ(ierr);
    ierr = ISDestroy(&is_from_u);CHKERRQ(ierr);

    ierr = VecGetOwnershipRange(user->uxi[i],&lo,&hi);CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF,hi-lo,lo,1,&is_to_uxi);CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF,hi-lo,lo,1,&is_from_u);CHKERRQ(ierr);
    ierr = VecScatterCreate(user->ui[i],is_from_u,user->uxi[i],is_to_uxi,&user->ux_scatter[i]);CHKERRQ(ierr);

    ierr = ISDestroy(&is_to_uxi);CHKERRQ(ierr);
    ierr = ISDestroy(&is_from_u);CHKERRQ(ierr);

    ierr = VecGetOwnershipRange(user->uyi[i],&lo,&hi);CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF,hi-lo,lo,1,&is_to_uyi);CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF,hi-lo,lo+user->mx*user->mx,1,&is_from_u);CHKERRQ(ierr);
    ierr = VecScatterCreate(user->ui[i],is_from_u,user->uyi[i],is_to_uyi,&user->uy_scatter[i]);CHKERRQ(ierr);

    ierr = ISDestroy(&is_to_uyi);CHKERRQ(ierr);
    ierr = ISDestroy(&is_from_u);CHKERRQ(ierr);

    ierr = VecGetOwnershipRange(user->ui[i],&lo,&hi);CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF,hi-lo,lo,1,&is_to_uxi);CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF,hi-lo,lo+2*i*user->mx*user->mx,1,&is_from_u);CHKERRQ(ierr);
    ierr = VecScatterCreate(user->u,is_from_u,user->ui[i],is_to_uxi,&user->ui_scatter[i]);CHKERRQ(ierr);

    ierr = ISDestroy(&is_to_uxi);CHKERRQ(ierr);
    ierr = ISDestroy(&is_from_u);CHKERRQ(ierr);
  }

  /* RHS of forward problem */
  ierr = MatMult(user->M,bc,user->yiwork[0]);CHKERRQ(ierr);
  for (i=1; i<user->nt; i++) {
    ierr = VecSet(user->yiwork[i],0.0);CHKERRQ(ierr);
  }
  ierr = Gather_yi(user->q,user->yiwork,user->yi_scatter,user->nt);CHKERRQ(ierr);

  /* Compute true velocity field utrue */
  ierr = VecDuplicate(user->u,&user->utrue);CHKERRQ(ierr);
  for (i=0; i<user->nt; i++) {
    ierr = VecCopy(YY,user->uxi[i]);CHKERRQ(ierr);
    ierr = VecScale(user->uxi[i],150.0*i*user->ht);CHKERRQ(ierr);
    ierr = VecCopy(XX,user->uyi[i]);CHKERRQ(ierr);
    ierr = VecShift(user->uyi[i],-10.0);CHKERRQ(ierr);
    ierr = VecScale(user->uyi[i],15.0*i*user->ht);CHKERRQ(ierr);
  }
  ierr = Gather_uxi_uyi(user->utrue,user->uxi,user->uxi_scatter,user->uyi,user->uyi_scatter,user->nt);CHKERRQ(ierr);

  /* Initial guess and reference model */
  ierr = VecDuplicate(user->utrue,&user->ur);CHKERRQ(ierr);
  for (i=0; i<user->nt; i++) {
    ierr = VecCopy(XX,user->uxi[i]);CHKERRQ(ierr);
    ierr = VecShift(user->uxi[i],i*user->ht);CHKERRQ(ierr);
    ierr = VecCopy(YY,user->uyi[i]);CHKERRQ(ierr);
    ierr = VecShift(user->uyi[i],-i*user->ht);CHKERRQ(ierr);
  }
  ierr = Gather_uxi_uyi(user->ur,user->uxi,user->uxi_scatter,user->uyi,user->uyi_scatter,user->nt);CHKERRQ(ierr);

  /* Generate regularization matrix L */
  ierr = MatCreate(PETSC_COMM_WORLD,&user->LT);CHKERRQ(ierr);
  ierr = MatSetSizes(user->LT,PETSC_DECIDE,PETSC_DECIDE,2*n*user->nt,n*user->nt);CHKERRQ(ierr);
  ierr = MatSetFromOptions(user->LT);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(user->LT,1,NULL,1,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(user->LT,1,NULL);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(user->LT,&istart,&iend);CHKERRQ(ierr);

  for (i=istart; i<iend; i++) {
    iblock = (i+n) / (2*n);
    j = i - iblock*n;
    ierr = MatSetValues(user->LT,1,&i,1,&j,&user->gamma,INSERT_VALUES);CHKERRQ(ierr);
  }

  ierr = MatAssemblyBegin(user->LT,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(user->LT,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatTranspose(user->LT,MAT_INITIAL_MATRIX,&user->L);CHKERRQ(ierr);

  /* Build work vectors and matrices */
  ierr = VecCreate(PETSC_COMM_WORLD,&user->lwork);CHKERRQ(ierr);
  ierr = VecSetType(user->lwork,VECMPI);CHKERRQ(ierr);
  ierr = VecSetSizes(user->lwork,PETSC_DECIDE,user->m);CHKERRQ(ierr);
  ierr = VecSetFromOptions(user->lwork);CHKERRQ(ierr);

  ierr = MatDuplicate(user->Div,MAT_SHARE_NONZERO_PATTERN,&user->Divwork);CHKERRQ(ierr);

  ierr = VecDuplicate(user->y,&user->ywork);CHKERRQ(ierr);
  ierr = VecDuplicate(user->u,&user->uwork);CHKERRQ(ierr);
  ierr = VecDuplicate(user->u,&user->vwork);CHKERRQ(ierr);
  ierr = VecDuplicate(user->u,&user->js_diag);CHKERRQ(ierr);
  ierr = VecDuplicate(user->c,&user->cwork);CHKERRQ(ierr);

  /* Create matrix-free shell user->Js for computing A*x */
  ierr = MatCreateShell(PETSC_COMM_WORLD,PETSC_DETERMINE,PETSC_DETERMINE,user->m,user->m,user,&user->Js);CHKERRQ(ierr);
  ierr = MatShellSetOperation(user->Js,MATOP_MULT,(void(*)(void))StateMatMult);CHKERRQ(ierr);
  ierr = MatShellSetOperation(user->Js,MATOP_DUPLICATE,(void(*)(void))StateMatDuplicate);CHKERRQ(ierr);
  ierr = MatShellSetOperation(user->Js,MATOP_MULT_TRANSPOSE,(void(*)(void))StateMatMultTranspose);CHKERRQ(ierr);
  ierr = MatShellSetOperation(user->Js,MATOP_GET_DIAGONAL,(void(*)(void))StateMatGetDiagonal);CHKERRQ(ierr);

  /* Diagonal blocks of user->Js */
  ierr = MatCreateShell(PETSC_COMM_WORLD,PETSC_DETERMINE,PETSC_DETERMINE,n,n,user,&user->JsBlock);CHKERRQ(ierr);
  ierr = MatShellSetOperation(user->JsBlock,MATOP_MULT,(void(*)(void))StateMatBlockMult);CHKERRQ(ierr);
  ierr = MatShellSetOperation(user->JsBlock,MATOP_MULT_TRANSPOSE,(void(*)(void))StateMatBlockMultTranspose);CHKERRQ(ierr);

  /* Create a matrix-free shell user->JsBlockPrec for computing (U+D)\D*(L+D)\x, where JsBlock = L+D+U,
     D is diagonal, L is strictly lower triangular, and U is strictly upper triangular.
     This is an SOR preconditioner for user->JsBlock. */
  ierr = MatCreateShell(PETSC_COMM_WORLD,PETSC_DETERMINE,PETSC_DETERMINE,n,n,user,&user->JsBlockPrec);CHKERRQ(ierr);
  ierr = MatShellSetOperation(user->JsBlockPrec,MATOP_MULT,(void(*)(void))StateMatBlockPrecMult);CHKERRQ(ierr);
  ierr = MatShellSetOperation(user->JsBlockPrec,MATOP_MULT_TRANSPOSE,(void(*)(void))StateMatBlockPrecMultTranspose);CHKERRQ(ierr);

  /* Create a matrix-free shell user->Jd for computing B*x */
  ierr = MatCreateShell(PETSC_COMM_WORLD,PETSC_DETERMINE,PETSC_DETERMINE,user->m,user->n-user->m,user,&user->Jd);CHKERRQ(ierr);
  ierr = MatShellSetOperation(user->Jd,MATOP_MULT,(void(*)(void))DesignMatMult);CHKERRQ(ierr);
  ierr = MatShellSetOperation(user->Jd,MATOP_MULT_TRANSPOSE,(void(*)(void))DesignMatMultTranspose);CHKERRQ(ierr);

  /* User-defined routines for computing user->Js\x and user->Js^T\x*/
  ierr = MatCreateShell(PETSC_COMM_WORLD,PETSC_DETERMINE,PETSC_DETERMINE,user->m,user->m,user,&user->JsInv);CHKERRQ(ierr);
  ierr = MatShellSetOperation(user->JsInv,MATOP_MULT,(void(*)(void))StateMatInvMult);CHKERRQ(ierr);
  ierr = MatShellSetOperation(user->JsInv,MATOP_MULT_TRANSPOSE,(void(*)(void))StateMatInvTransposeMult);CHKERRQ(ierr);

  /* Build matrices for SOR preconditioner */
  ierr = Scatter_uxi_uyi(user->u,user->uxi,user->uxi_scatter,user->uyi,user->uyi_scatter,user->nt);CHKERRQ(ierr);
  ierr = PetscMalloc1(5*n,&user->C);CHKERRQ(ierr);
  ierr = PetscMalloc1(2*n,&user->Cwork);CHKERRQ(ierr);
  for (i=0; i<user->nt; i++) {
    ierr = MatDuplicate(user->Divxy[0],MAT_COPY_VALUES,&user->C[i]);CHKERRQ(ierr);
    ierr = MatDuplicate(user->Divxy[1],MAT_COPY_VALUES,&user->Cwork[i]);CHKERRQ(ierr);

    ierr = MatDiagonalScale(user->C[i],NULL,user->uxi[i]);CHKERRQ(ierr);
    ierr = MatDiagonalScale(user->Cwork[i],NULL,user->uyi[i]);CHKERRQ(ierr);
    ierr = MatAXPY(user->C[i],1.0,user->Cwork[i],DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatScale(user->C[i],user->ht);CHKERRQ(ierr);
    ierr = MatShift(user->C[i],1.0);CHKERRQ(ierr);
  }

  /* Solver options and tolerances */
  ierr = KSPCreate(PETSC_COMM_WORLD,&user->solver);CHKERRQ(ierr);
  ierr = KSPSetType(user->solver,KSPGMRES);CHKERRQ(ierr);
  ierr = KSPSetOperators(user->solver,user->JsBlock,user->JsBlockPrec);CHKERRQ(ierr);
  ierr = KSPSetTolerances(user->solver,1e-4,1e-20,1e3,500);CHKERRQ(ierr);
  /* ierr = KSPSetTolerances(user->solver,1e-8,1e-16,1e3,500);CHKERRQ(ierr); */
  ierr = KSPGetPC(user->solver,&user->prec);CHKERRQ(ierr);
  ierr = PCSetType(user->prec,PCSHELL);CHKERRQ(ierr);

  ierr = PCShellSetApply(user->prec,StateMatBlockPrecMult);CHKERRQ(ierr);
  ierr = PCShellSetApplyTranspose(user->prec,StateMatBlockPrecMultTranspose);CHKERRQ(ierr);
  ierr = PCShellSetContext(user->prec,user);CHKERRQ(ierr);

  /* Compute true state function yt given ut */
  ierr = VecCreate(PETSC_COMM_WORLD,&user->ytrue);CHKERRQ(ierr);
  ierr = VecSetSizes(user->ytrue,PETSC_DECIDE,n*user->nt);CHKERRQ(ierr);
  ierr = VecSetFromOptions(user->ytrue);CHKERRQ(ierr);
  user->c_formed = PETSC_TRUE;
  ierr = VecCopy(user->utrue,user->u);CHKERRQ(ierr); /*  Set u=utrue temporarily for StateMatInv */
  ierr = VecSet(user->ytrue,0.0);CHKERRQ(ierr); /*  Initial guess */
  ierr = StateMatInvMult(user->Js,user->q,user->ytrue);CHKERRQ(ierr);
  ierr = VecCopy(user->ur,user->u);CHKERRQ(ierr); /*  Reset u=ur */

  /* Initial guess y0 for state given u0 */
  ierr = StateMatInvMult(user->Js,user->q,user->y);CHKERRQ(ierr);

  /* Data discretization */
  ierr = MatCreate(PETSC_COMM_WORLD,&user->Q);CHKERRQ(ierr);
  ierr = MatSetSizes(user->Q,PETSC_DECIDE,PETSC_DECIDE,user->mx*user->mx,user->m);CHKERRQ(ierr);
  ierr = MatSetFromOptions(user->Q);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(user->Q,0,NULL,1,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(user->Q,1,NULL);CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(user->Q,&istart,&iend);CHKERRQ(ierr);

  for (i=istart; i<iend; i++) {
    j = i + user->m - user->mx*user->mx;
    ierr = MatSetValues(user->Q,1,&i,1,&j,&one,INSERT_VALUES);CHKERRQ(ierr);
  }

  ierr = MatAssemblyBegin(user->Q,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(user->Q,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatTranspose(user->Q,MAT_INITIAL_MATRIX,&user->QT);CHKERRQ(ierr);

  ierr = VecDestroy(&XX);CHKERRQ(ierr);
  ierr = VecDestroy(&YY);CHKERRQ(ierr);
  ierr = VecDestroy(&XXwork);CHKERRQ(ierr);
  ierr = VecDestroy(&YYwork);CHKERRQ(ierr);
  ierr = VecDestroy(&yi);CHKERRQ(ierr);
  ierr = VecDestroy(&uxi);CHKERRQ(ierr);
  ierr = VecDestroy(&ui);CHKERRQ(ierr);
  ierr = VecDestroy(&bc);CHKERRQ(ierr);

  /* Now that initial conditions have been set, let the user pass tolerance options to the KSP solver */
  ierr = KSPSetFromOptions(user->solver);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode HyperbolicDestroy(AppCtx *user)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  ierr = MatDestroy(&user->Q);CHKERRQ(ierr);
  ierr = MatDestroy(&user->QT);CHKERRQ(ierr);
  ierr = MatDestroy(&user->Div);CHKERRQ(ierr);
  ierr = MatDestroy(&user->Divwork);CHKERRQ(ierr);
  ierr = MatDestroy(&user->Grad);CHKERRQ(ierr);
  ierr = MatDestroy(&user->L);CHKERRQ(ierr);
  ierr = MatDestroy(&user->LT);CHKERRQ(ierr);
  ierr = KSPDestroy(&user->solver);CHKERRQ(ierr);
  ierr = MatDestroy(&user->Js);CHKERRQ(ierr);
  ierr = MatDestroy(&user->Jd);CHKERRQ(ierr);
  ierr = MatDestroy(&user->JsBlockPrec);CHKERRQ(ierr);
  ierr = MatDestroy(&user->JsInv);CHKERRQ(ierr);
  ierr = MatDestroy(&user->JsBlock);CHKERRQ(ierr);
  ierr = MatDestroy(&user->Divxy[0]);CHKERRQ(ierr);
  ierr = MatDestroy(&user->Divxy[1]);CHKERRQ(ierr);
  ierr = MatDestroy(&user->Gradxy[0]);CHKERRQ(ierr);
  ierr = MatDestroy(&user->Gradxy[1]);CHKERRQ(ierr);
  ierr = MatDestroy(&user->M);CHKERRQ(ierr);
  for (i=0; i<user->nt; i++) {
    ierr = MatDestroy(&user->C[i]);CHKERRQ(ierr);
    ierr = MatDestroy(&user->Cwork[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(user->C);CHKERRQ(ierr);
  ierr = PetscFree(user->Cwork);CHKERRQ(ierr);
  ierr = VecDestroy(&user->u);CHKERRQ(ierr);
  ierr = VecDestroy(&user->uwork);CHKERRQ(ierr);
  ierr = VecDestroy(&user->vwork);CHKERRQ(ierr);
  ierr = VecDestroy(&user->utrue);CHKERRQ(ierr);
  ierr = VecDestroy(&user->y);CHKERRQ(ierr);
  ierr = VecDestroy(&user->ywork);CHKERRQ(ierr);
  ierr = VecDestroy(&user->ytrue);CHKERRQ(ierr);
  ierr = VecDestroyVecs(user->nt,&user->yi);CHKERRQ(ierr);
  ierr = VecDestroyVecs(user->nt,&user->yiwork);CHKERRQ(ierr);
  ierr = VecDestroyVecs(user->nt,&user->ziwork);CHKERRQ(ierr);
  ierr = VecDestroyVecs(user->nt,&user->uxi);CHKERRQ(ierr);
  ierr = VecDestroyVecs(user->nt,&user->uyi);CHKERRQ(ierr);
  ierr = VecDestroyVecs(user->nt,&user->uxiwork);CHKERRQ(ierr);
  ierr = VecDestroyVecs(user->nt,&user->uyiwork);CHKERRQ(ierr);
  ierr = VecDestroyVecs(user->nt,&user->ui);CHKERRQ(ierr);
  ierr = VecDestroyVecs(user->nt,&user->uiwork);CHKERRQ(ierr);
  ierr = VecDestroy(&user->c);CHKERRQ(ierr);
  ierr = VecDestroy(&user->cwork);CHKERRQ(ierr);
  ierr = VecDestroy(&user->ur);CHKERRQ(ierr);
  ierr = VecDestroy(&user->q);CHKERRQ(ierr);
  ierr = VecDestroy(&user->d);CHKERRQ(ierr);
  ierr = VecDestroy(&user->dwork);CHKERRQ(ierr);
  ierr = VecDestroy(&user->lwork);CHKERRQ(ierr);
  ierr = VecDestroy(&user->js_diag);CHKERRQ(ierr);
  ierr = ISDestroy(&user->s_is);CHKERRQ(ierr);
  ierr = ISDestroy(&user->d_is);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&user->state_scatter);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&user->design_scatter);CHKERRQ(ierr);
  for (i=0; i<user->nt; i++) {
    ierr = VecScatterDestroy(&user->uxi_scatter[i]);CHKERRQ(ierr);
    ierr = VecScatterDestroy(&user->uyi_scatter[i]);CHKERRQ(ierr);
    ierr = VecScatterDestroy(&user->ux_scatter[i]);CHKERRQ(ierr);
    ierr = VecScatterDestroy(&user->uy_scatter[i]);CHKERRQ(ierr);
    ierr = VecScatterDestroy(&user->ui_scatter[i]);CHKERRQ(ierr);
    ierr = VecScatterDestroy(&user->yi_scatter[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(user->uxi_scatter);CHKERRQ(ierr);
  ierr = PetscFree(user->uyi_scatter);CHKERRQ(ierr);
  ierr = PetscFree(user->ux_scatter);CHKERRQ(ierr);
  ierr = PetscFree(user->uy_scatter);CHKERRQ(ierr);
  ierr = PetscFree(user->ui_scatter);CHKERRQ(ierr);
  ierr = PetscFree(user->yi_scatter);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode HyperbolicMonitor(Tao tao, void *ptr)
{
  PetscErrorCode ierr;
  Vec            X;
  PetscReal      unorm,ynorm;
  AppCtx         *user = (AppCtx*)ptr;

  PetscFunctionBegin;
  ierr = TaoGetSolutionVector(tao,&X);CHKERRQ(ierr);
  ierr = Scatter(X,user->ywork,user->state_scatter,user->uwork,user->design_scatter);CHKERRQ(ierr);
  ierr = VecAXPY(user->ywork,-1.0,user->ytrue);CHKERRQ(ierr);
  ierr = VecAXPY(user->uwork,-1.0,user->utrue);CHKERRQ(ierr);
  ierr = VecNorm(user->uwork,NORM_2,&unorm);CHKERRQ(ierr);
  ierr = VecNorm(user->ywork,NORM_2,&ynorm);CHKERRQ(ierr);
  ierr = PetscPrintf(MPI_COMM_WORLD, "||u-ut||=%g ||y-yt||=%g\n",(double)unorm,(double)ynorm);CHKERRQ(ierr);
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
