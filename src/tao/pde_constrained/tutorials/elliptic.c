#include <petsc/private/taoimpl.h>

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
   Processors: n
T*/



typedef struct {
  PetscInt n; /* Number of total variables */
  PetscInt m; /* Number of constraints */
  PetscInt nstate;
  PetscInt ndesign;
  PetscInt mx; /* grid points in each direction */
  PetscInt ns; /* Number of data samples (1<=ns<=8)
                  Currently only ns=1 is supported */
  PetscInt ndata; /* Number of data points per sample */
  IS       s_is;
  IS       d_is;

  VecScatter state_scatter;
  VecScatter design_scatter;
  VecScatter *yi_scatter, *di_scatter;
  Vec        suby,subq,subd;
  Mat        Js,Jd,JsPrec,JsInv,JsBlock;

  PetscReal alpha; /* Regularization parameter */
  PetscReal beta; /* Weight attributed to ||u||^2 in regularization functional */
  PetscReal noise; /* Amount of noise to add to data */
  PetscReal *ones;
  Mat       Q;
  Mat       MQ;
  Mat       L;

  Mat Grad;
  Mat Av,Avwork;
  Mat Div, Divwork;
  Mat DSG;
  Mat Diag,Ones;


  Vec q;
  Vec ur; /* reference */

  Vec d;
  Vec dwork;

  Vec x; /* super vec of y,u */

  Vec y; /* state variables */
  Vec ywork;

  Vec ytrue;

  Vec u; /* design variables */
  Vec uwork;

  Vec utrue;

  Vec js_diag;

  Vec c; /* constraint vector */
  Vec cwork;

  Vec lwork;
  Vec S;
  Vec Swork,Twork,Sdiag,Ywork;
  Vec Av_u;

  KSP solver;
  PC  prec;

  PetscReal tola,tolb,tolc,told;
  PetscInt  ksp_its;
  PetscInt  ksp_its_initial;
  PetscLogStage stages[10];
  PetscBool use_ptap;
  PetscBool use_lrc;
} AppCtx;

PetscErrorCode FormFunction(Tao, Vec, PetscReal*, void*);
PetscErrorCode FormGradient(Tao, Vec, Vec, void*);
PetscErrorCode FormFunctionGradient(Tao, Vec, PetscReal*, Vec, void*);
PetscErrorCode FormJacobianState(Tao, Vec, Mat, Mat, Mat, void*);
PetscErrorCode FormJacobianDesign(Tao, Vec, Mat,void*);
PetscErrorCode FormConstraints(Tao, Vec, Vec, void*);
PetscErrorCode FormHessian(Tao, Vec, Mat, Mat, void*);
PetscErrorCode Gather(Vec, Vec, VecScatter, Vec, VecScatter);
PetscErrorCode Scatter(Vec, Vec, VecScatter, Vec, VecScatter);
PetscErrorCode EllipticInitialize(AppCtx*);
PetscErrorCode EllipticDestroy(AppCtx*);
PetscErrorCode EllipticMonitor(Tao, void*);

PetscErrorCode StateBlockMatMult(Mat,Vec,Vec);
PetscErrorCode StateMatMult(Mat,Vec,Vec);

PetscErrorCode StateInvMatMult(Mat,Vec,Vec);
PetscErrorCode DesignMatMult(Mat,Vec,Vec);
PetscErrorCode DesignMatMultTranspose(Mat,Vec,Vec);

PetscErrorCode QMatMult(Mat,Vec,Vec);
PetscErrorCode QMatMultTranspose(Mat,Vec,Vec);

static  char help[]="";

int main(int argc, char **argv)
{
  PetscErrorCode     ierr;
  Vec                x0;
  Tao                tao;
  AppCtx             user;
  PetscInt           ntests = 1;
  PetscInt           i;

  ierr = PetscInitialize(&argc, &argv, (char*)0,help);if (ierr) return ierr;
  user.mx = 8;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"elliptic example",NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-mx","Number of grid points in each direction","",user.mx,&user.mx,NULL);CHKERRQ(ierr);
  user.ns = 6;
  ierr = PetscOptionsInt("-ns","Number of data samples (1<=ns<=8)","",user.ns,&user.ns,NULL);CHKERRQ(ierr);
  user.ndata = 64;
  ierr = PetscOptionsInt("-ndata","Numbers of data points per sample","",user.ndata,&user.ndata,NULL);CHKERRQ(ierr);
  user.alpha = 0.1;
  ierr = PetscOptionsReal("-alpha","Regularization parameter","",user.alpha,&user.alpha,NULL);CHKERRQ(ierr);
  user.beta = 0.00001;
  ierr = PetscOptionsReal("-beta","Weight attributed to ||u||^2 in regularization functional","",user.beta,&user.beta,NULL);CHKERRQ(ierr);
  user.noise = 0.01;
  ierr = PetscOptionsReal("-noise","Amount of noise to add to data","",user.noise,&user.noise,NULL);CHKERRQ(ierr);

  user.use_ptap = PETSC_FALSE;
  ierr = PetscOptionsBool("-use_ptap","Use ptap matrix for DSG","",user.use_ptap,&user.use_ptap,NULL);CHKERRQ(ierr);
  user.use_lrc = PETSC_FALSE;
  ierr = PetscOptionsBool("-use_lrc","Use lrc matrix for Js","",user.use_lrc,&user.use_lrc,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-ntests","Number of times to repeat TaoSolve","",ntests,&ntests,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  user.m = user.ns*user.mx*user.mx*user.mx; /* number of constraints */
  user.nstate =  user.m;
  user.ndesign = user.mx*user.mx*user.mx;
  user.n = user.nstate + user.ndesign; /* number of variables */

  /* Create TAO solver and set desired solution method */
  ierr = TaoCreate(PETSC_COMM_WORLD,&tao);CHKERRQ(ierr);
  ierr = TaoSetType(tao,TAOLCL);CHKERRQ(ierr);

  /* Set up initial vectors and matrices */
  ierr = EllipticInitialize(&user);CHKERRQ(ierr);

  ierr = Gather(user.x,user.y,user.state_scatter,user.u,user.design_scatter);CHKERRQ(ierr);
  ierr = VecDuplicate(user.x,&x0);CHKERRQ(ierr);
  ierr = VecCopy(user.x,x0);CHKERRQ(ierr);

  /* Set solution vector with an initial guess */
  ierr = TaoSetInitialVector(tao,user.x);CHKERRQ(ierr);
  ierr = TaoSetObjectiveRoutine(tao, FormFunction, (void *)&user);CHKERRQ(ierr);
  ierr = TaoSetGradientRoutine(tao, FormGradient, (void *)&user);CHKERRQ(ierr);
  ierr = TaoSetConstraintsRoutine(tao, user.c, FormConstraints, (void *)&user);CHKERRQ(ierr);

  ierr = TaoSetJacobianStateRoutine(tao, user.Js, NULL, user.JsInv, FormJacobianState, (void *)&user);CHKERRQ(ierr);
  ierr = TaoSetJacobianDesignRoutine(tao, user.Jd, FormJacobianDesign, (void *)&user);CHKERRQ(ierr);

  ierr = TaoSetStateDesignIS(tao,user.s_is,user.d_is);CHKERRQ(ierr);
  ierr = TaoSetFromOptions(tao);CHKERRQ(ierr);

  /* SOLVE THE APPLICATION */
  ierr = PetscLogStageRegister("Trials",&user.stages[1]);CHKERRQ(ierr);
  ierr = PetscLogStagePush(user.stages[1]);CHKERRQ(ierr);
  for (i=0; i<ntests; i++){
    ierr = TaoSolve(tao);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"KSP Iterations = %D\n",user.ksp_its);CHKERRQ(ierr);
    ierr = VecCopy(x0,user.x);CHKERRQ(ierr);
  }
  ierr = PetscLogStagePop();CHKERRQ(ierr);
  ierr = PetscBarrier((PetscObject)user.x);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"KSP iterations within initialization: ");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"%D\n",user.ksp_its_initial);CHKERRQ(ierr);

  ierr = TaoDestroy(&tao);CHKERRQ(ierr);
  ierr = VecDestroy(&x0);CHKERRQ(ierr);
  ierr = EllipticDestroy(&user);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
/* ------------------------------------------------------------------- */
/*
   dwork = Qy - d
   lwork = L*(u-ur)
   f = 1/2 * (dwork.dwork + alpha*lwork.lwork)
*/
PetscErrorCode FormFunction(Tao tao,Vec X,PetscReal *f,void *ptr)
{
  PetscErrorCode ierr;
  PetscReal      d1=0,d2=0;
  AppCtx         *user = (AppCtx*)ptr;

  PetscFunctionBegin;
  ierr = Scatter(X,user->y,user->state_scatter,user->u,user->design_scatter);CHKERRQ(ierr);
  ierr = MatMult(user->MQ,user->y,user->dwork);CHKERRQ(ierr);
  ierr = VecAXPY(user->dwork,-1.0,user->d);CHKERRQ(ierr);
  ierr = VecDot(user->dwork,user->dwork,&d1);CHKERRQ(ierr);
  ierr = VecWAXPY(user->uwork,-1.0,user->ur,user->u);CHKERRQ(ierr);
  ierr = MatMult(user->L,user->uwork,user->lwork);CHKERRQ(ierr);
  ierr = VecDot(user->lwork,user->lwork,&d2);CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  AppCtx         *user = (AppCtx*)ptr;

  PetscFunctionBegin;
  ierr = Scatter(X,user->y,user->state_scatter,user->u,user->design_scatter);CHKERRQ(ierr);
  ierr = MatMult(user->MQ,user->y,user->dwork);CHKERRQ(ierr);
  ierr = VecAXPY(user->dwork,-1.0,user->d);CHKERRQ(ierr);
  ierr = MatMultTranspose(user->MQ,user->dwork,user->ywork);CHKERRQ(ierr);
  ierr = VecWAXPY(user->uwork,-1.0,user->ur,user->u);CHKERRQ(ierr);
  ierr = MatMult(user->L,user->uwork,user->lwork);CHKERRQ(ierr);
  ierr = MatMultTranspose(user->L,user->lwork,user->uwork);CHKERRQ(ierr);
  ierr = VecScale(user->uwork, user->alpha);CHKERRQ(ierr);
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
  ierr = MatMult(user->MQ,user->y,user->dwork);CHKERRQ(ierr);
  ierr = VecAXPY(user->dwork,-1.0,user->d);CHKERRQ(ierr);
  ierr = VecDot(user->dwork,user->dwork,&d1);CHKERRQ(ierr);
  ierr = MatMultTranspose(user->MQ,user->dwork,user->ywork);CHKERRQ(ierr);

  ierr = VecWAXPY(user->uwork,-1.0,user->ur,user->u);CHKERRQ(ierr);
  ierr = MatMult(user->L,user->uwork,user->lwork);CHKERRQ(ierr);
  ierr = VecDot(user->lwork,user->lwork,&d2);CHKERRQ(ierr);
  ierr = MatMultTranspose(user->L,user->lwork,user->uwork);CHKERRQ(ierr);
  ierr = VecScale(user->uwork, user->alpha);CHKERRQ(ierr);
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
  AppCtx         *user = (AppCtx*)ptr;

  PetscFunctionBegin;
  ierr = Scatter(X,user->y,user->state_scatter,user->u,user->design_scatter);CHKERRQ(ierr);
  /* DSG = Div * (1/Av_u) * Grad */
  ierr = VecSet(user->uwork,0);CHKERRQ(ierr);
  ierr = VecAXPY(user->uwork,-1.0,user->u);CHKERRQ(ierr);
  ierr = VecExp(user->uwork);CHKERRQ(ierr);
  ierr = MatMult(user->Av,user->uwork,user->Av_u);CHKERRQ(ierr);
  ierr = VecCopy(user->Av_u,user->Swork);CHKERRQ(ierr);
  ierr = VecReciprocal(user->Swork);CHKERRQ(ierr);
  if (user->use_ptap) {
    ierr = MatDiagonalSet(user->Diag,user->Swork,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatPtAP(user->Diag,user->Grad,MAT_REUSE_MATRIX,1.0,&user->DSG);CHKERRQ(ierr);
  } else {
    ierr = MatCopy(user->Div,user->Divwork,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatDiagonalScale(user->Divwork,NULL,user->Swork);CHKERRQ(ierr);
    ierr = MatProductNumeric(user->DSG);CHKERRQ(ierr);
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

PetscErrorCode StateBlockMatMult(Mat J_shell, Vec X, Vec Y)
{
  PetscErrorCode ierr;
  PetscReal      sum;
  AppCtx         *user;

  PetscFunctionBegin;
  ierr = MatShellGetContext(J_shell,(void**)&user);CHKERRQ(ierr);
  ierr = MatMult(user->DSG,X,Y);CHKERRQ(ierr);
  ierr = VecSum(X,&sum);CHKERRQ(ierr);
  sum /= user->ndesign;
  ierr = VecShift(Y,sum);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode StateMatMult(Mat J_shell, Vec X, Vec Y)
{
  PetscErrorCode ierr;
  PetscInt       i;
  AppCtx         *user;

  PetscFunctionBegin;
  ierr = MatShellGetContext(J_shell,(void**)&user);CHKERRQ(ierr);
  if (user->ns == 1) {
    ierr = MatMult(user->JsBlock,X,Y);CHKERRQ(ierr);
  } else {
    for (i=0;i<user->ns;i++) {
      ierr = Scatter(X,user->subq,user->yi_scatter[i],0,0);CHKERRQ(ierr);
      ierr = Scatter(Y,user->suby,user->yi_scatter[i],0,0);CHKERRQ(ierr);
      ierr = MatMult(user->JsBlock,user->subq,user->suby);CHKERRQ(ierr);
      ierr = Gather(Y,user->suby,user->yi_scatter[i],0,0);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode StateInvMatMult(Mat J_shell, Vec X, Vec Y)
{
  PetscErrorCode ierr;
  PetscInt       its,i;
  AppCtx         *user;

  PetscFunctionBegin;
  ierr = MatShellGetContext(J_shell,(void**)&user);CHKERRQ(ierr);
  ierr = KSPSetOperators(user->solver,user->JsBlock,user->DSG);CHKERRQ(ierr);
  if (Y == user->ytrue) {
    /* First solve is done using true solution to set up problem */
    ierr = KSPSetTolerances(user->solver,1e-8,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
  } else {
    ierr = KSPSetTolerances(user->solver,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
  }
  if (user->ns == 1) {
    ierr = KSPSolve(user->solver,X,Y);CHKERRQ(ierr);
    ierr = KSPGetIterationNumber(user->solver,&its);CHKERRQ(ierr);
    user->ksp_its+=its;
  } else {
    for (i=0;i<user->ns;i++) {
      ierr = Scatter(X,user->subq,user->yi_scatter[i],0,0);CHKERRQ(ierr);
      ierr = Scatter(Y,user->suby,user->yi_scatter[i],0,0);CHKERRQ(ierr);
      ierr = KSPSolve(user->solver,user->subq,user->suby);CHKERRQ(ierr);
      ierr = KSPGetIterationNumber(user->solver,&its);CHKERRQ(ierr);
      user->ksp_its+=its;
      ierr = Gather(Y,user->suby,user->yi_scatter[i],0,0);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}
PetscErrorCode QMatMult(Mat J_shell, Vec X, Vec Y)
{
  PetscErrorCode ierr;
  AppCtx         *user;
  PetscInt       i;

  PetscFunctionBegin;
  ierr = MatShellGetContext(J_shell,(void**)&user);CHKERRQ(ierr);
  if (user->ns == 1) {
    ierr = MatMult(user->Q,X,Y);CHKERRQ(ierr);
  } else {
    for (i=0;i<user->ns;i++) {
      ierr = Scatter(X,user->subq,user->yi_scatter[i],0,0);CHKERRQ(ierr);
      ierr = Scatter(Y,user->subd,user->di_scatter[i],0,0);CHKERRQ(ierr);
      ierr = MatMult(user->Q,user->subq,user->subd);CHKERRQ(ierr);
      ierr = Gather(Y,user->subd,user->di_scatter[i],0,0);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode QMatMultTranspose(Mat J_shell, Vec X, Vec Y)
{
  PetscErrorCode ierr;
  AppCtx         *user;
  PetscInt       i;

  PetscFunctionBegin;
  ierr = MatShellGetContext(J_shell,(void**)&user);CHKERRQ(ierr);
  if (user->ns == 1) {
    ierr = MatMultTranspose(user->Q,X,Y);CHKERRQ(ierr);
  } else {
    for (i=0;i<user->ns;i++) {
      ierr = Scatter(X,user->subd,user->di_scatter[i],0,0);CHKERRQ(ierr);
      ierr = Scatter(Y,user->suby,user->yi_scatter[i],0,0);CHKERRQ(ierr);
      ierr = MatMultTranspose(user->Q,user->subd,user->suby);CHKERRQ(ierr);
      ierr = Gather(Y,user->suby,user->yi_scatter[i],0,0);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DesignMatMult(Mat J_shell, Vec X, Vec Y)
{
  PetscErrorCode ierr;
  PetscInt       i;
  AppCtx         *user;

  PetscFunctionBegin;
  ierr = MatShellGetContext(J_shell,(void**)&user);CHKERRQ(ierr);

  /* sdiag(1./v) */
  ierr = VecSet(user->uwork,0);CHKERRQ(ierr);
  ierr = VecAXPY(user->uwork,-1.0,user->u);CHKERRQ(ierr);
  ierr = VecExp(user->uwork);CHKERRQ(ierr);

  /* sdiag(1./((Av*(1./v)).^2)) */
  ierr = MatMult(user->Av,user->uwork,user->Swork);CHKERRQ(ierr);
  ierr = VecPointwiseMult(user->Swork,user->Swork,user->Swork);CHKERRQ(ierr);
  ierr = VecReciprocal(user->Swork);CHKERRQ(ierr);

  /* (Av * (sdiag(1./v) * b)) */
  ierr = VecPointwiseMult(user->uwork,user->uwork,X);CHKERRQ(ierr);
  ierr = MatMult(user->Av,user->uwork,user->Twork);CHKERRQ(ierr);

  /* (sdiag(1./((Av*(1./v)).^2)) * (Av * (sdiag(1./v) * b))) */
  ierr = VecPointwiseMult(user->Swork,user->Twork,user->Swork);CHKERRQ(ierr);

  if (user->ns == 1) {
    /* (sdiag(Grad*y(:,i)) */
    ierr = MatMult(user->Grad,user->y,user->Twork);CHKERRQ(ierr);

    /* Div * (sdiag(Grad*y(:,i)) * (sdiag(1./((Av*(1./v)).^2)) * (Av * (sdiag(1./v) * b)))) */
    ierr = VecPointwiseMult(user->Swork,user->Twork,user->Swork);CHKERRQ(ierr);
    ierr = MatMultTranspose(user->Grad,user->Swork,Y);CHKERRQ(ierr);
  } else {
    for (i=0;i<user->ns;i++) {
      ierr = Scatter(user->y,user->suby,user->yi_scatter[i],0,0);CHKERRQ(ierr);
      ierr = Scatter(Y,user->subq,user->yi_scatter[i],0,0);CHKERRQ(ierr);

      ierr = MatMult(user->Grad,user->suby,user->Twork);CHKERRQ(ierr);
      ierr = VecPointwiseMult(user->Twork,user->Twork,user->Swork);CHKERRQ(ierr);
      ierr = MatMultTranspose(user->Grad,user->Twork,user->subq);CHKERRQ(ierr);
      ierr = Gather(user->y,user->suby,user->yi_scatter[i],0,0);CHKERRQ(ierr);
      ierr = Gather(Y,user->subq,user->yi_scatter[i],0,0);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DesignMatMultTranspose(Mat J_shell, Vec X, Vec Y)
{
  PetscErrorCode ierr;
  PetscInt       i;
  AppCtx         *user;

  PetscFunctionBegin;
  ierr = MatShellGetContext(J_shell,(void**)&user);CHKERRQ(ierr);
  ierr = VecZeroEntries(Y);CHKERRQ(ierr);

  /* Sdiag = 1./((Av*(1./v)).^2) */
  ierr = VecSet(user->uwork,0);CHKERRQ(ierr);
  ierr = VecAXPY(user->uwork,-1.0,user->u);CHKERRQ(ierr);
  ierr = VecExp(user->uwork);CHKERRQ(ierr);
  ierr = MatMult(user->Av,user->uwork,user->Swork);CHKERRQ(ierr);
  ierr = VecPointwiseMult(user->Sdiag,user->Swork,user->Swork);CHKERRQ(ierr);
  ierr = VecReciprocal(user->Sdiag);CHKERRQ(ierr);

  for (i=0;i<user->ns;i++) {
    ierr = Scatter(X,user->subq,user->yi_scatter[i],0,0);CHKERRQ(ierr);
    ierr = Scatter(user->y,user->suby,user->yi_scatter[i],0,0);CHKERRQ(ierr);

    /* Swork = (Div' * b(:,i)) */
    ierr = MatMult(user->Grad,user->subq,user->Swork);CHKERRQ(ierr);

    /* Twork = Grad*y(:,i) */
    ierr = MatMult(user->Grad,user->suby,user->Twork);CHKERRQ(ierr);

    /* Twork = sdiag(Twork) * Swork */
    ierr = VecPointwiseMult(user->Twork,user->Swork,user->Twork);CHKERRQ(ierr);


    /* Swork = pointwisemult(Sdiag,Twork) */
    ierr = VecPointwiseMult(user->Swork,user->Twork,user->Sdiag);CHKERRQ(ierr);

    /* Ywork = Av' * Swork */
    ierr = MatMultTranspose(user->Av,user->Swork,user->Ywork);CHKERRQ(ierr);

    /* Ywork = pointwisemult(uwork,Ywork) */
    ierr = VecPointwiseMult(user->Ywork,user->uwork,user->Ywork);CHKERRQ(ierr);
    ierr = VecAXPY(Y,1.0,user->Ywork);CHKERRQ(ierr);
    ierr = Gather(user->y,user->suby,user->yi_scatter[i],0,0);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode FormConstraints(Tao tao, Vec X, Vec C, void *ptr)
{
   /* C=Ay - q      A = Div * Sigma * Grad + hx*hx*hx*ones(n,n) */
   PetscErrorCode ierr;
   PetscReal      sum;
   PetscInt       i;
   AppCtx         *user = (AppCtx*)ptr;

   PetscFunctionBegin;
   ierr = Scatter(X,user->y,user->state_scatter,user->u,user->design_scatter);CHKERRQ(ierr);
   if (user->ns == 1) {
     ierr = MatMult(user->Grad,user->y,user->Swork);CHKERRQ(ierr);
     ierr = VecPointwiseDivide(user->Swork,user->Swork,user->Av_u);CHKERRQ(ierr);
     ierr = MatMultTranspose(user->Grad,user->Swork,C);CHKERRQ(ierr);
     ierr = VecSum(user->y,&sum);CHKERRQ(ierr);
     sum /= user->ndesign;
     ierr = VecShift(C,sum);CHKERRQ(ierr);
   } else {
     for (i=0;i<user->ns;i++) {
      ierr = Scatter(user->y,user->suby,user->yi_scatter[i],0,0);CHKERRQ(ierr);
      ierr = Scatter(C,user->subq,user->yi_scatter[i],0,0);CHKERRQ(ierr);
      ierr = MatMult(user->Grad,user->suby,user->Swork);CHKERRQ(ierr);
      ierr = VecPointwiseDivide(user->Swork,user->Swork,user->Av_u);CHKERRQ(ierr);
      ierr = MatMultTranspose(user->Grad,user->Swork,user->subq);CHKERRQ(ierr);

      ierr = VecSum(user->suby,&sum);CHKERRQ(ierr);
      sum /= user->ndesign;
      ierr = VecShift(user->subq,sum);CHKERRQ(ierr);

      ierr = Gather(user->y,user->suby,user->yi_scatter[i],0,0);CHKERRQ(ierr);
      ierr = Gather(C,user->subq,user->yi_scatter[i],0,0);CHKERRQ(ierr);
     }
   }
   ierr = VecAXPY(C,-1.0,user->q);CHKERRQ(ierr);
   PetscFunctionReturn(0);
}

PetscErrorCode Scatter(Vec x, Vec sub1, VecScatter scat1, Vec sub2, VecScatter scat2)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecScatterBegin(scat1,x,sub1,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(scat1,x,sub1,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  if (sub2) {
    ierr = VecScatterBegin(scat2,x,sub2,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(scat2,x,sub2,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode Gather(Vec x, Vec sub1, VecScatter scat1, Vec sub2, VecScatter scat2)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecScatterBegin(scat1,sub1,x,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(scat1,sub1,x,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  if (sub2) {
    ierr = VecScatterBegin(scat2,sub2,x,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(scat2,sub2,x,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode EllipticInitialize(AppCtx *user)
{
  PetscErrorCode ierr;
  PetscInt       m,n,i,j,k,l,linear_index,is,js,ks,ls,istart,iend,iblock;
  Vec            XX,YY,ZZ,XXwork,YYwork,ZZwork,UTwork;
  PetscReal      *x,*y,*z;
  PetscReal      h,meanut;
  PetscScalar    hinv,neg_hinv,half = 0.5,sqrt_beta;
  PetscInt       im,indx1,indx2,indy1,indy2,indz1,indz2,nx,ny,nz;
  IS             is_alldesign,is_allstate;
  IS             is_from_d;
  IS             is_from_y;
  PetscInt       lo,hi,hi2,lo2,ysubnlocal,dsubnlocal;
  const PetscInt *ranges, *subranges;
  PetscMPIInt    size;
  PetscReal      xri,yri,zri,xim,yim,zim,dx1,dx2,dy1,dy2,dz1,dz2,Dx,Dy,Dz;
  PetscScalar    v,vx,vy,vz;
  PetscInt       offset,subindex,subvec,nrank,kk;

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
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  ierr = PetscLogStageRegister("Elliptic Setup",&user->stages[0]);CHKERRQ(ierr);
  ierr = PetscLogStagePush(user->stages[0]);CHKERRQ(ierr);

  /* Create u,y,c,x */
  ierr = VecCreate(PETSC_COMM_WORLD,&user->u);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&user->y);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&user->c);CHKERRQ(ierr);
  ierr = VecSetSizes(user->u,PETSC_DECIDE,user->ndesign);CHKERRQ(ierr);
  ierr = VecSetFromOptions(user->u);CHKERRQ(ierr);
  ierr = VecGetLocalSize(user->u,&ysubnlocal);CHKERRQ(ierr);
  ierr = VecSetSizes(user->y,ysubnlocal*user->ns,user->nstate);CHKERRQ(ierr);
  ierr = VecSetSizes(user->c,ysubnlocal*user->ns,user->m);CHKERRQ(ierr);
  ierr = VecSetFromOptions(user->y);CHKERRQ(ierr);
  ierr = VecSetFromOptions(user->c);CHKERRQ(ierr);

  /*
     *******************************
     Create scatters for x <-> y,u
     *******************************

     If the state vector y and design vector u are partitioned as
     [y_1; y_2; ...; y_np] and [u_1; u_2; ...; u_np] (with np = # of processors),
     then the solution vector x is organized as
     [y_1; u_1; y_2; u_2; ...; y_np; u_np].
     The index sets user->s_is and user->d_is correspond to the indices of the
     state and design variables owned by the current processor.
  */
  ierr = VecCreate(PETSC_COMM_WORLD,&user->x);CHKERRQ(ierr);

  ierr = VecGetOwnershipRange(user->y,&lo,&hi);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(user->u,&lo2,&hi2);CHKERRQ(ierr);

  ierr = ISCreateStride(PETSC_COMM_SELF,hi-lo,lo,1,&is_allstate);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_SELF,hi-lo,lo+lo2,1,&user->s_is);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_SELF,hi2-lo2,lo2,1,&is_alldesign);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_SELF,hi2-lo2,hi+lo2,1,&user->d_is);CHKERRQ(ierr);

  ierr = VecSetSizes(user->x,hi-lo+hi2-lo2,user->n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(user->x);CHKERRQ(ierr);

  ierr = VecScatterCreate(user->x,user->s_is,user->y,is_allstate,&user->state_scatter);CHKERRQ(ierr);
  ierr = VecScatterCreate(user->x,user->d_is,user->u,is_alldesign,&user->design_scatter);CHKERRQ(ierr);
  ierr = ISDestroy(&is_alldesign);CHKERRQ(ierr);
  ierr = ISDestroy(&is_allstate);CHKERRQ(ierr);
  /*
     *******************************
     Create scatter from y to y_1,y_2,...,y_ns
     *******************************
  */
  ierr = PetscMalloc1(user->ns,&user->yi_scatter);CHKERRQ(ierr);
  ierr = VecDuplicate(user->u,&user->suby);CHKERRQ(ierr);
  ierr = VecDuplicate(user->u,&user->subq);CHKERRQ(ierr);

  ierr = VecGetOwnershipRange(user->y,&lo2,&hi2);CHKERRQ(ierr);
  istart = 0;
  for (i=0; i<user->ns; i++){
    ierr = VecGetOwnershipRange(user->suby,&lo,&hi);CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF,hi-lo,lo2+istart,1,&is_from_y);CHKERRQ(ierr);
    ierr = VecScatterCreate(user->y,is_from_y,user->suby,NULL,&user->yi_scatter[i]);CHKERRQ(ierr);
    istart = istart + hi-lo;
    ierr = ISDestroy(&is_from_y);CHKERRQ(ierr);
  }
  /*
     *******************************
     Create scatter from d to d_1,d_2,...,d_ns
     *******************************
  */
  ierr = VecCreate(PETSC_COMM_WORLD,&user->subd);CHKERRQ(ierr);
  ierr = VecSetSizes(user->subd,PETSC_DECIDE,user->ndata);CHKERRQ(ierr);
  ierr = VecSetFromOptions(user->subd);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&user->d);CHKERRQ(ierr);
  ierr = VecGetLocalSize(user->subd,&dsubnlocal);CHKERRQ(ierr);
  ierr = VecSetSizes(user->d,dsubnlocal*user->ns,user->ndata*user->ns);CHKERRQ(ierr);
  ierr = VecSetFromOptions(user->d);CHKERRQ(ierr);
  ierr = PetscMalloc1(user->ns,&user->di_scatter);CHKERRQ(ierr);

  ierr = VecGetOwnershipRange(user->d,&lo2,&hi2);CHKERRQ(ierr);
  istart = 0;
  for (i=0; i<user->ns; i++){
    ierr = VecGetOwnershipRange(user->subd,&lo,&hi);CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF,hi-lo,lo2+istart,1,&is_from_d);CHKERRQ(ierr);
    ierr = VecScatterCreate(user->d,is_from_d,user->subd,NULL,&user->di_scatter[i]);CHKERRQ(ierr);
    istart = istart + hi-lo;
    ierr = ISDestroy(&is_from_d);CHKERRQ(ierr);
  }

  ierr = PetscMalloc1(user->mx,&x);CHKERRQ(ierr);
  ierr = PetscMalloc1(user->mx,&y);CHKERRQ(ierr);
  ierr = PetscMalloc1(user->mx,&z);CHKERRQ(ierr);

  user->ksp_its = 0;
  user->ksp_its_initial = 0;

  n = user->mx * user->mx * user->mx;
  m = 3 * user->mx * user->mx * (user->mx-1);
  sqrt_beta = PetscSqrtScalar(user->beta);

  ierr = VecCreate(PETSC_COMM_WORLD,&XX);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&user->q);CHKERRQ(ierr);
  ierr = VecSetSizes(XX,ysubnlocal,n);CHKERRQ(ierr);
  ierr = VecSetSizes(user->q,ysubnlocal*user->ns,user->m);CHKERRQ(ierr);
  ierr = VecSetFromOptions(XX);CHKERRQ(ierr);
  ierr = VecSetFromOptions(user->q);CHKERRQ(ierr);

  ierr = VecDuplicate(XX,&YY);CHKERRQ(ierr);
  ierr = VecDuplicate(XX,&ZZ);CHKERRQ(ierr);
  ierr = VecDuplicate(XX,&XXwork);CHKERRQ(ierr);
  ierr = VecDuplicate(XX,&YYwork);CHKERRQ(ierr);
  ierr = VecDuplicate(XX,&ZZwork);CHKERRQ(ierr);
  ierr = VecDuplicate(XX,&UTwork);CHKERRQ(ierr);
  ierr = VecDuplicate(XX,&user->utrue);CHKERRQ(ierr);

  /* map for striding q */
  ierr = VecGetOwnershipRanges(user->q,&ranges);CHKERRQ(ierr);
  ierr = VecGetOwnershipRanges(user->u,&subranges);CHKERRQ(ierr);

  ierr = VecGetOwnershipRange(user->q,&lo2,&hi2);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(user->u,&lo,&hi);CHKERRQ(ierr);
  /* Generate 3D grid, and collect ns (1<=ns<=8) right-hand-side vectors into user->q */
  h = 1.0/user->mx;
  hinv = user->mx;
  neg_hinv = -hinv;

  ierr = VecGetOwnershipRange(XX,&istart,&iend);CHKERRQ(ierr);
  for (linear_index=istart; linear_index<iend; linear_index++){
    i = linear_index % user->mx;
    j = ((linear_index-i)/user->mx) % user->mx;
    k = ((linear_index-i)/user->mx-j) / user->mx;
    vx = h*(i+0.5);
    vy = h*(j+0.5);
    vz = h*(k+0.5);
    ierr = VecSetValues(XX,1,&linear_index,&vx,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValues(YY,1,&linear_index,&vy,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValues(ZZ,1,&linear_index,&vz,INSERT_VALUES);CHKERRQ(ierr);
    for (is=0; is<2; is++){
      for (js=0; js<2; js++){
        for (ks=0; ks<2; ks++){
          ls = is*4 + js*2 + ks;
          if (ls<user->ns){
            l =ls*n + linear_index;
            /* remap */
            subindex = l%n;
            subvec = l/n;
            nrank=0;
            while (subindex >= subranges[nrank+1]) nrank++;
            offset = subindex - subranges[nrank];
            istart=0;
            for (kk=0;kk<nrank;kk++) istart+=user->ns*(subranges[kk+1]-subranges[kk]);
            istart += (subranges[nrank+1]-subranges[nrank])*subvec;
            l = istart+offset;
            v = 100*PetscSinScalar(2*PETSC_PI*(vx+0.25*is))*PetscSinScalar(2*PETSC_PI*(vy+0.25*js))*PetscSinScalar(2*PETSC_PI*(vz+0.25*ks));
            ierr = VecSetValues(user->q,1,&l,&v,INSERT_VALUES);CHKERRQ(ierr);
          }
        }
      }
    }
  }

  ierr = VecAssemblyBegin(XX);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(XX);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(YY);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(YY);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(ZZ);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(ZZ);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(user->q);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(user->q);CHKERRQ(ierr);

  /* Compute true parameter function
     ut = exp(-((x-0.25)^2+(y-0.25)^2+(z-0.25)^2)/0.05) - exp((x-0.75)^2-(y-0.75)^2-(z-0.75))^2/0.05) */
  ierr = VecCopy(XX,XXwork);CHKERRQ(ierr);
  ierr = VecCopy(YY,YYwork);CHKERRQ(ierr);
  ierr = VecCopy(ZZ,ZZwork);CHKERRQ(ierr);

  ierr = VecShift(XXwork,-0.25);CHKERRQ(ierr);
  ierr = VecShift(YYwork,-0.25);CHKERRQ(ierr);
  ierr = VecShift(ZZwork,-0.25);CHKERRQ(ierr);

  ierr = VecPointwiseMult(XXwork,XXwork,XXwork);CHKERRQ(ierr);
  ierr = VecPointwiseMult(YYwork,YYwork,YYwork);CHKERRQ(ierr);
  ierr = VecPointwiseMult(ZZwork,ZZwork,ZZwork);CHKERRQ(ierr);

  ierr = VecCopy(XXwork,UTwork);CHKERRQ(ierr);
  ierr = VecAXPY(UTwork,1.0,YYwork);CHKERRQ(ierr);
  ierr = VecAXPY(UTwork,1.0,ZZwork);CHKERRQ(ierr);
  ierr = VecScale(UTwork,-20.0);CHKERRQ(ierr);
  ierr = VecExp(UTwork);CHKERRQ(ierr);
  ierr = VecCopy(UTwork,user->utrue);CHKERRQ(ierr);

  ierr = VecCopy(XX,XXwork);CHKERRQ(ierr);
  ierr = VecCopy(YY,YYwork);CHKERRQ(ierr);
  ierr = VecCopy(ZZ,ZZwork);CHKERRQ(ierr);

  ierr = VecShift(XXwork,-0.75);CHKERRQ(ierr);
  ierr = VecShift(YYwork,-0.75);CHKERRQ(ierr);
  ierr = VecShift(ZZwork,-0.75);CHKERRQ(ierr);

  ierr = VecPointwiseMult(XXwork,XXwork,XXwork);CHKERRQ(ierr);
  ierr = VecPointwiseMult(YYwork,YYwork,YYwork);CHKERRQ(ierr);
  ierr = VecPointwiseMult(ZZwork,ZZwork,ZZwork);CHKERRQ(ierr);

  ierr = VecCopy(XXwork,UTwork);CHKERRQ(ierr);
  ierr = VecAXPY(UTwork,1.0,YYwork);CHKERRQ(ierr);
  ierr = VecAXPY(UTwork,1.0,ZZwork);CHKERRQ(ierr);
  ierr = VecScale(UTwork,-20.0);CHKERRQ(ierr);
  ierr = VecExp(UTwork);CHKERRQ(ierr);

  ierr = VecAXPY(user->utrue,-1.0,UTwork);CHKERRQ(ierr);

  ierr = VecDestroy(&XX);CHKERRQ(ierr);
  ierr = VecDestroy(&YY);CHKERRQ(ierr);
  ierr = VecDestroy(&ZZ);CHKERRQ(ierr);
  ierr = VecDestroy(&XXwork);CHKERRQ(ierr);
  ierr = VecDestroy(&YYwork);CHKERRQ(ierr);
  ierr = VecDestroy(&ZZwork);CHKERRQ(ierr);
  ierr = VecDestroy(&UTwork);CHKERRQ(ierr);

  /* Initial guess and reference model */
  ierr = VecDuplicate(user->utrue,&user->ur);CHKERRQ(ierr);
  ierr = VecSum(user->utrue,&meanut);CHKERRQ(ierr);
  meanut = meanut / n;
  ierr = VecSet(user->ur,meanut);CHKERRQ(ierr);
  ierr = VecCopy(user->ur,user->u);CHKERRQ(ierr);

  /* Generate Grad matrix */
  ierr = MatCreate(PETSC_COMM_WORLD,&user->Grad);CHKERRQ(ierr);
  ierr = MatSetSizes(user->Grad,PETSC_DECIDE,ysubnlocal,m,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(user->Grad);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(user->Grad,2,NULL,2,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(user->Grad,2,NULL);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(user->Grad,&istart,&iend);CHKERRQ(ierr);

  for (i=istart; i<iend; i++){
    if (i<m/3){
      iblock = i / (user->mx-1);
      j = iblock*user->mx + (i % (user->mx-1));
      ierr = MatSetValues(user->Grad,1,&i,1,&j,&neg_hinv,INSERT_VALUES);CHKERRQ(ierr);
      j = j+1;
      ierr = MatSetValues(user->Grad,1,&i,1,&j,&hinv,INSERT_VALUES);CHKERRQ(ierr);
    }
    if (i>=m/3 && i<2*m/3){
      iblock = (i-m/3) / (user->mx*(user->mx-1));
      j = iblock*user->mx*user->mx + ((i-m/3) % (user->mx*(user->mx-1)));
      ierr = MatSetValues(user->Grad,1,&i,1,&j,&neg_hinv,INSERT_VALUES);CHKERRQ(ierr);
      j = j + user->mx;
      ierr = MatSetValues(user->Grad,1,&i,1,&j,&hinv,INSERT_VALUES);CHKERRQ(ierr);
    }
    if (i>=2*m/3){
      j = i-2*m/3;
      ierr = MatSetValues(user->Grad,1,&i,1,&j,&neg_hinv,INSERT_VALUES);CHKERRQ(ierr);
      j = j + user->mx*user->mx;
      ierr = MatSetValues(user->Grad,1,&i,1,&j,&hinv,INSERT_VALUES);CHKERRQ(ierr);
    }
  }

  ierr = MatAssemblyBegin(user->Grad,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(user->Grad,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Generate arithmetic averaging matrix Av */
  ierr = MatCreate(PETSC_COMM_WORLD,&user->Av);CHKERRQ(ierr);
  ierr = MatSetSizes(user->Av,PETSC_DECIDE,ysubnlocal,m,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(user->Av);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(user->Av,2,NULL,2,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(user->Av,2,NULL);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(user->Av,&istart,&iend);CHKERRQ(ierr);

  for (i=istart; i<iend; i++){
    if (i<m/3){
      iblock = i / (user->mx-1);
      j = iblock*user->mx + (i % (user->mx-1));
      ierr = MatSetValues(user->Av,1,&i,1,&j,&half,INSERT_VALUES);CHKERRQ(ierr);
      j = j+1;
      ierr = MatSetValues(user->Av,1,&i,1,&j,&half,INSERT_VALUES);CHKERRQ(ierr);
    }
    if (i>=m/3 && i<2*m/3){
      iblock = (i-m/3) / (user->mx*(user->mx-1));
      j = iblock*user->mx*user->mx + ((i-m/3) % (user->mx*(user->mx-1)));
      ierr = MatSetValues(user->Av,1,&i,1,&j,&half,INSERT_VALUES);CHKERRQ(ierr);
      j = j + user->mx;
      ierr = MatSetValues(user->Av,1,&i,1,&j,&half,INSERT_VALUES);CHKERRQ(ierr);
    }
    if (i>=2*m/3){
      j = i-2*m/3;
      ierr = MatSetValues(user->Av,1,&i,1,&j,&half,INSERT_VALUES);CHKERRQ(ierr);
      j = j + user->mx*user->mx;
      ierr = MatSetValues(user->Av,1,&i,1,&j,&half,INSERT_VALUES);CHKERRQ(ierr);
    }
  }

  ierr = MatAssemblyBegin(user->Av,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(user->Av,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&user->L);CHKERRQ(ierr);
  ierr = MatSetSizes(user->L,PETSC_DECIDE,ysubnlocal,m+n,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(user->L);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(user->L,2,NULL,2,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(user->L,2,NULL);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(user->L,&istart,&iend);CHKERRQ(ierr);

  for (i=istart; i<iend; i++){
    if (i<m/3){
      iblock = i / (user->mx-1);
      j = iblock*user->mx + (i % (user->mx-1));
      ierr = MatSetValues(user->L,1,&i,1,&j,&neg_hinv,INSERT_VALUES);CHKERRQ(ierr);
      j = j+1;
      ierr = MatSetValues(user->L,1,&i,1,&j,&hinv,INSERT_VALUES);CHKERRQ(ierr);
    }
    if (i>=m/3 && i<2*m/3){
      iblock = (i-m/3) / (user->mx*(user->mx-1));
      j = iblock*user->mx*user->mx + ((i-m/3) % (user->mx*(user->mx-1)));
      ierr = MatSetValues(user->L,1,&i,1,&j,&neg_hinv,INSERT_VALUES);CHKERRQ(ierr);
      j = j + user->mx;
      ierr = MatSetValues(user->L,1,&i,1,&j,&hinv,INSERT_VALUES);CHKERRQ(ierr);
    }
    if (i>=2*m/3 && i<m){
      j = i-2*m/3;
      ierr = MatSetValues(user->L,1,&i,1,&j,&neg_hinv,INSERT_VALUES);CHKERRQ(ierr);
      j = j + user->mx*user->mx;
      ierr = MatSetValues(user->L,1,&i,1,&j,&hinv,INSERT_VALUES);CHKERRQ(ierr);
    }
    if (i>=m){
      j = i - m;
      ierr = MatSetValues(user->L,1,&i,1,&j,&sqrt_beta,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(user->L,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(user->L,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatScale(user->L,PetscPowScalar(h,1.5));CHKERRQ(ierr);

  /* Generate Div matrix */
  if (!user->use_ptap) {
    /* Generate Div matrix */
    ierr = MatCreate(PETSC_COMM_WORLD,&user->Div);CHKERRQ(ierr);
    ierr = MatSetSizes(user->Div,ysubnlocal,PETSC_DECIDE,n,m);CHKERRQ(ierr);
    ierr = MatSetFromOptions(user->Div);CHKERRQ(ierr);
    ierr = MatMPIAIJSetPreallocation(user->Div,4,NULL,4,NULL);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(user->Div,6,NULL);CHKERRQ(ierr);
    ierr = MatGetOwnershipRange(user->Grad,&istart,&iend);CHKERRQ(ierr);

    for (i=istart; i<iend; i++){
      if (i<m/3){
        iblock = i / (user->mx-1);
        j = iblock*user->mx + (i % (user->mx-1));
        ierr = MatSetValues(user->Div,1,&j,1,&i,&neg_hinv,INSERT_VALUES);CHKERRQ(ierr);
        j = j+1;
        ierr = MatSetValues(user->Div,1,&j,1,&i,&hinv,INSERT_VALUES);CHKERRQ(ierr);
      }
      if (i>=m/3 && i<2*m/3){
        iblock = (i-m/3) / (user->mx*(user->mx-1));
        j = iblock*user->mx*user->mx + ((i-m/3) % (user->mx*(user->mx-1)));
        ierr = MatSetValues(user->Div,1,&j,1,&i,&neg_hinv,INSERT_VALUES);CHKERRQ(ierr);
        j = j + user->mx;
        ierr = MatSetValues(user->Div,1,&j,1,&i,&hinv,INSERT_VALUES);CHKERRQ(ierr);
      }
      if (i>=2*m/3){
        j = i-2*m/3;
        ierr = MatSetValues(user->Div,1,&j,1,&i,&neg_hinv,INSERT_VALUES);CHKERRQ(ierr);
        j = j + user->mx*user->mx;
        ierr = MatSetValues(user->Div,1,&j,1,&i,&hinv,INSERT_VALUES);CHKERRQ(ierr);
      }
    }

    ierr = MatAssemblyBegin(user->Div,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(user->Div,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatDuplicate(user->Div,MAT_SHARE_NONZERO_PATTERN,&user->Divwork);CHKERRQ(ierr);
  } else {
    ierr = MatCreate(PETSC_COMM_WORLD,&user->Diag);CHKERRQ(ierr);
    ierr = MatSetSizes(user->Diag,PETSC_DECIDE,PETSC_DECIDE,m,m);CHKERRQ(ierr);
    ierr = MatSetFromOptions(user->Diag);CHKERRQ(ierr);
    ierr = MatMPIAIJSetPreallocation(user->Diag,1,NULL,0,NULL);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(user->Diag,1,NULL);CHKERRQ(ierr);
  }

  /* Build work vectors and matrices */
  ierr = VecCreate(PETSC_COMM_WORLD,&user->S);CHKERRQ(ierr);
  ierr = VecSetSizes(user->S, PETSC_DECIDE, m);CHKERRQ(ierr);
  ierr = VecSetFromOptions(user->S);CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD,&user->lwork);CHKERRQ(ierr);
  ierr = VecSetSizes(user->lwork,PETSC_DECIDE,m+user->mx*user->mx*user->mx);CHKERRQ(ierr);
  ierr = VecSetFromOptions(user->lwork);CHKERRQ(ierr);

  ierr = MatDuplicate(user->Av,MAT_SHARE_NONZERO_PATTERN,&user->Avwork);CHKERRQ(ierr);

  ierr = VecDuplicate(user->S,&user->Swork);CHKERRQ(ierr);
  ierr = VecDuplicate(user->S,&user->Sdiag);CHKERRQ(ierr);
  ierr = VecDuplicate(user->S,&user->Av_u);CHKERRQ(ierr);
  ierr = VecDuplicate(user->S,&user->Twork);CHKERRQ(ierr);
  ierr = VecDuplicate(user->y,&user->ywork);CHKERRQ(ierr);
  ierr = VecDuplicate(user->u,&user->Ywork);CHKERRQ(ierr);
  ierr = VecDuplicate(user->u,&user->uwork);CHKERRQ(ierr);
  ierr = VecDuplicate(user->u,&user->js_diag);CHKERRQ(ierr);
  ierr = VecDuplicate(user->c,&user->cwork);CHKERRQ(ierr);
  ierr = VecDuplicate(user->d,&user->dwork);CHKERRQ(ierr);

  /* Create a matrix-free shell user->Jd for computing B*x */
  ierr = MatCreateShell(PETSC_COMM_WORLD,ysubnlocal*user->ns,ysubnlocal,user->nstate,user->ndesign,user,&user->Jd);CHKERRQ(ierr);
  ierr = MatShellSetOperation(user->Jd,MATOP_MULT,(void(*)(void))DesignMatMult);CHKERRQ(ierr);
  ierr = MatShellSetOperation(user->Jd,MATOP_MULT_TRANSPOSE,(void(*)(void))DesignMatMultTranspose);CHKERRQ(ierr);

  /* Compute true state function ytrue given utrue */
  ierr = VecDuplicate(user->y,&user->ytrue);CHKERRQ(ierr);

  /* First compute Av_u = Av*exp(-u) */
  ierr = VecSet(user->uwork, 0);CHKERRQ(ierr);
  ierr = VecAXPY(user->uwork,-1.0,user->utrue);CHKERRQ(ierr); /* Note: user->utrue */
  ierr = VecExp(user->uwork);CHKERRQ(ierr);
  ierr = MatMult(user->Av,user->uwork,user->Av_u);CHKERRQ(ierr);

  /* Next form DSG = Div*S*Grad */
  ierr = VecCopy(user->Av_u,user->Swork);CHKERRQ(ierr);
  ierr = VecReciprocal(user->Swork);CHKERRQ(ierr);
  if (user->use_ptap) {
    ierr = MatDiagonalSet(user->Diag,user->Swork,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatPtAP(user->Diag,user->Grad,MAT_INITIAL_MATRIX,1.0,&user->DSG);CHKERRQ(ierr);
  } else {
    ierr = MatCopy(user->Div,user->Divwork,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatDiagonalScale(user->Divwork,NULL,user->Swork);CHKERRQ(ierr);

    ierr = MatMatMult(user->Divwork,user->Grad,MAT_INITIAL_MATRIX,1.0,&user->DSG);CHKERRQ(ierr);
  }

  ierr = MatSetOption(user->DSG,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatSetOption(user->DSG,MAT_SYMMETRY_ETERNAL,PETSC_TRUE);CHKERRQ(ierr);

  if (user->use_lrc == PETSC_TRUE) {
    v=PetscSqrtReal(1.0 /user->ndesign);
    ierr = PetscMalloc1(user->ndesign,&user->ones);CHKERRQ(ierr);

    for (i=0;i<user->ndesign;i++) {
      user->ones[i]=v;
    }
    ierr = MatCreateDense(PETSC_COMM_WORLD,ysubnlocal,PETSC_DECIDE,user->ndesign,1,user->ones,&user->Ones);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(user->Ones, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(user->Ones, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatCreateLRC(user->DSG,user->Ones,NULL,user->Ones,&user->JsBlock);CHKERRQ(ierr);
    ierr = MatSetUp(user->JsBlock);CHKERRQ(ierr);
  } else {
    /* Create matrix-free shell user->Js for computing (A + h^3*e*e^T)*x */
    ierr = MatCreateShell(PETSC_COMM_WORLD,ysubnlocal,ysubnlocal,user->ndesign,user->ndesign,user,&user->JsBlock);CHKERRQ(ierr);
    ierr = MatShellSetOperation(user->JsBlock,MATOP_MULT,(void(*)(void))StateBlockMatMult);CHKERRQ(ierr);
    ierr = MatShellSetOperation(user->JsBlock,MATOP_MULT_TRANSPOSE,(void(*)(void))StateBlockMatMult);CHKERRQ(ierr);
  }
  ierr = MatSetOption(user->JsBlock,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatSetOption(user->JsBlock,MAT_SYMMETRY_ETERNAL,PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatCreateShell(PETSC_COMM_WORLD,ysubnlocal*user->ns,ysubnlocal*user->ns,user->nstate,user->nstate,user,&user->Js);CHKERRQ(ierr);
  ierr = MatShellSetOperation(user->Js,MATOP_MULT,(void(*)(void))StateMatMult);CHKERRQ(ierr);
  ierr = MatShellSetOperation(user->Js,MATOP_MULT_TRANSPOSE,(void(*)(void))StateMatMult);CHKERRQ(ierr);
  ierr = MatSetOption(user->Js,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatSetOption(user->Js,MAT_SYMMETRY_ETERNAL,PETSC_TRUE);CHKERRQ(ierr);

  ierr = MatCreateShell(PETSC_COMM_WORLD,ysubnlocal*user->ns,ysubnlocal*user->ns,user->nstate,user->nstate,user,&user->JsInv);CHKERRQ(ierr);
  ierr = MatShellSetOperation(user->JsInv,MATOP_MULT,(void(*)(void))StateInvMatMult);CHKERRQ(ierr);
  ierr = MatShellSetOperation(user->JsInv,MATOP_MULT_TRANSPOSE,(void(*)(void))StateInvMatMult);CHKERRQ(ierr);
  ierr = MatSetOption(user->JsInv,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatSetOption(user->JsInv,MAT_SYMMETRY_ETERNAL,PETSC_TRUE);CHKERRQ(ierr);

  ierr = MatSetOption(user->DSG,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatSetOption(user->DSG,MAT_SYMMETRY_ETERNAL,PETSC_TRUE);CHKERRQ(ierr);
  /* Now solve for ytrue */
  ierr = KSPCreate(PETSC_COMM_WORLD,&user->solver);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(user->solver);CHKERRQ(ierr);

  ierr = KSPSetOperators(user->solver,user->JsBlock,user->DSG);CHKERRQ(ierr);

  ierr = MatMult(user->JsInv,user->q,user->ytrue);CHKERRQ(ierr);
  /* First compute Av_u = Av*exp(-u) */
  ierr = VecSet(user->uwork,0);CHKERRQ(ierr);
  ierr = VecAXPY(user->uwork,-1.0,user->u);CHKERRQ(ierr); /* Note: user->u */
  ierr = VecExp(user->uwork);CHKERRQ(ierr);
  ierr = MatMult(user->Av,user->uwork,user->Av_u);CHKERRQ(ierr);

  /* Next update DSG = Div*S*Grad  with user->u */
  ierr = VecCopy(user->Av_u,user->Swork);CHKERRQ(ierr);
  ierr = VecReciprocal(user->Swork);CHKERRQ(ierr);
  if (user->use_ptap) {
    ierr = MatDiagonalSet(user->Diag,user->Swork,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatPtAP(user->Diag,user->Grad,MAT_REUSE_MATRIX,1.0,&user->DSG);CHKERRQ(ierr);
  } else {
    ierr = MatCopy(user->Div,user->Divwork,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatDiagonalScale(user->Divwork,NULL,user->Av_u);CHKERRQ(ierr);
    ierr = MatProductNumeric(user->DSG);CHKERRQ(ierr);
  }

  /* Now solve for y */

  ierr = MatMult(user->JsInv,user->q,user->y);CHKERRQ(ierr);

  user->ksp_its_initial = user->ksp_its;
  user->ksp_its = 0;
  /* Construct projection matrix Q (blocks) */
  ierr = MatCreate(PETSC_COMM_WORLD,&user->Q);CHKERRQ(ierr);
  ierr = MatSetSizes(user->Q,dsubnlocal,ysubnlocal,user->ndata,user->ndesign);CHKERRQ(ierr);
  ierr = MatSetFromOptions(user->Q);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(user->Q,8,NULL,8,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(user->Q,8,NULL);CHKERRQ(ierr);

  for (i=0; i<user->mx; i++){
    x[i] = h*(i+0.5);
    y[i] = h*(i+0.5);
    z[i] = h*(i+0.5);
  }
  ierr = MatGetOwnershipRange(user->Q,&istart,&iend);CHKERRQ(ierr);

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
    ierr = MatSetValues(user->Q,1,&i,1,&j,&v,INSERT_VALUES);CHKERRQ(ierr);

    j = indx1 + indy1*nx + indz2*nx*ny;
    v = (1-dx1/Dx)*(1-dy1/Dy)*(1-dz2/Dz);
    ierr = MatSetValues(user->Q,1,&i,1,&j,&v,INSERT_VALUES);CHKERRQ(ierr);

    j = indx1 + indy2*nx + indz1*nx*ny;
    v = (1-dx1/Dx)*(1-dy2/Dy)*(1-dz1/Dz);
    ierr = MatSetValues(user->Q,1,&i,1,&j,&v,INSERT_VALUES);CHKERRQ(ierr);

    j = indx1 + indy2*nx + indz2*nx*ny;
    v = (1-dx1/Dx)*(1-dy2/Dy)*(1-dz2/Dz);
    ierr = MatSetValues(user->Q,1,&i,1,&j,&v,INSERT_VALUES);CHKERRQ(ierr);

    j = indx2 + indy1*nx + indz1*nx*ny;
    v = (1-dx2/Dx)*(1-dy1/Dy)*(1-dz1/Dz);
    ierr = MatSetValues(user->Q,1,&i,1,&j,&v,INSERT_VALUES);CHKERRQ(ierr);

    j = indx2 + indy1*nx + indz2*nx*ny;
    v = (1-dx2/Dx)*(1-dy1/Dy)*(1-dz2/Dz);
    ierr = MatSetValues(user->Q,1,&i,1,&j,&v,INSERT_VALUES);CHKERRQ(ierr);

    j = indx2 + indy2*nx + indz1*nx*ny;
    v = (1-dx2/Dx)*(1-dy2/Dy)*(1-dz1/Dz);
    ierr = MatSetValues(user->Q,1,&i,1,&j,&v,INSERT_VALUES);CHKERRQ(ierr);

    j = indx2 + indy2*nx + indz2*nx*ny;
    v = (1-dx2/Dx)*(1-dy2/Dy)*(1-dz2/Dz);
    ierr = MatSetValues(user->Q,1,&i,1,&j,&v,INSERT_VALUES);CHKERRQ(ierr);
  }

  ierr = MatAssemblyBegin(user->Q,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(user->Q,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  /* Create MQ (composed of blocks of Q */
  ierr = MatCreateShell(PETSC_COMM_WORLD,dsubnlocal*user->ns,PETSC_DECIDE,user->ndata*user->ns,user->nstate,user,&user->MQ);CHKERRQ(ierr);
  ierr = MatShellSetOperation(user->MQ,MATOP_MULT,(void(*)(void))QMatMult);CHKERRQ(ierr);
  ierr = MatShellSetOperation(user->MQ,MATOP_MULT_TRANSPOSE,(void(*)(void))QMatMultTranspose);CHKERRQ(ierr);

  /* Add noise to the measurement data */
  ierr = VecSet(user->ywork,1.0);CHKERRQ(ierr);
  ierr = VecAYPX(user->ywork,user->noise,user->ytrue);CHKERRQ(ierr);
  ierr = MatMult(user->MQ,user->ywork,user->d);CHKERRQ(ierr);

  /* Now that initial conditions have been set, let the user pass tolerance options to the KSP solver */
  ierr = PetscFree(x);CHKERRQ(ierr);
  ierr = PetscFree(y);CHKERRQ(ierr);
  ierr = PetscFree(z);CHKERRQ(ierr);
  ierr = PetscLogStagePop();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode EllipticDestroy(AppCtx *user)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  ierr = MatDestroy(&user->DSG);CHKERRQ(ierr);
  ierr = KSPDestroy(&user->solver);CHKERRQ(ierr);
  ierr = MatDestroy(&user->Q);CHKERRQ(ierr);
  ierr = MatDestroy(&user->MQ);CHKERRQ(ierr);
  if (!user->use_ptap) {
    ierr = MatDestroy(&user->Div);CHKERRQ(ierr);
    ierr = MatDestroy(&user->Divwork);CHKERRQ(ierr);
  } else {
    ierr = MatDestroy(&user->Diag);CHKERRQ(ierr);
  }
  if (user->use_lrc) {
    ierr = MatDestroy(&user->Ones);CHKERRQ(ierr);
  }

  ierr = MatDestroy(&user->Grad);CHKERRQ(ierr);
  ierr = MatDestroy(&user->Av);CHKERRQ(ierr);
  ierr = MatDestroy(&user->Avwork);CHKERRQ(ierr);
  ierr = MatDestroy(&user->L);CHKERRQ(ierr);
  ierr = MatDestroy(&user->Js);CHKERRQ(ierr);
  ierr = MatDestroy(&user->Jd);CHKERRQ(ierr);
  ierr = MatDestroy(&user->JsBlock);CHKERRQ(ierr);
  ierr = MatDestroy(&user->JsInv);CHKERRQ(ierr);

  ierr = VecDestroy(&user->x);CHKERRQ(ierr);
  ierr = VecDestroy(&user->u);CHKERRQ(ierr);
  ierr = VecDestroy(&user->uwork);CHKERRQ(ierr);
  ierr = VecDestroy(&user->utrue);CHKERRQ(ierr);
  ierr = VecDestroy(&user->y);CHKERRQ(ierr);
  ierr = VecDestroy(&user->ywork);CHKERRQ(ierr);
  ierr = VecDestroy(&user->ytrue);CHKERRQ(ierr);
  ierr = VecDestroy(&user->c);CHKERRQ(ierr);
  ierr = VecDestroy(&user->cwork);CHKERRQ(ierr);
  ierr = VecDestroy(&user->ur);CHKERRQ(ierr);
  ierr = VecDestroy(&user->q);CHKERRQ(ierr);
  ierr = VecDestroy(&user->d);CHKERRQ(ierr);
  ierr = VecDestroy(&user->dwork);CHKERRQ(ierr);
  ierr = VecDestroy(&user->lwork);CHKERRQ(ierr);
  ierr = VecDestroy(&user->S);CHKERRQ(ierr);
  ierr = VecDestroy(&user->Swork);CHKERRQ(ierr);
  ierr = VecDestroy(&user->Sdiag);CHKERRQ(ierr);
  ierr = VecDestroy(&user->Ywork);CHKERRQ(ierr);
  ierr = VecDestroy(&user->Twork);CHKERRQ(ierr);
  ierr = VecDestroy(&user->Av_u);CHKERRQ(ierr);
  ierr = VecDestroy(&user->js_diag);CHKERRQ(ierr);
  ierr = ISDestroy(&user->s_is);CHKERRQ(ierr);
  ierr = ISDestroy(&user->d_is);CHKERRQ(ierr);
  ierr = VecDestroy(&user->suby);CHKERRQ(ierr);
  ierr = VecDestroy(&user->subd);CHKERRQ(ierr);
  ierr = VecDestroy(&user->subq);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&user->state_scatter);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&user->design_scatter);CHKERRQ(ierr);
  for (i=0;i<user->ns;i++) {
    ierr = VecScatterDestroy(&user->yi_scatter[i]);CHKERRQ(ierr);
    ierr = VecScatterDestroy(&user->di_scatter[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(user->yi_scatter);CHKERRQ(ierr);
  ierr = PetscFree(user->di_scatter);CHKERRQ(ierr);
  if (user->use_lrc) {
    ierr = PetscFree(user->ones);CHKERRQ(ierr);
    ierr = MatDestroy(&user->Ones);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode EllipticMonitor(Tao tao, void *ptr)
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
      args: -tao_cmonitor -ns 1 -tao_type lcl -tao_gatol 1.e-3 -tao_max_it 11
      requires: !single

   test:
      suffix: 2
      args: -tao_cmonitor -tao_type lcl -tao_max_it 11 -use_ptap -use_lrc -ns 1 -tao_gatol 1.e-3
      requires: !single

TEST*/
