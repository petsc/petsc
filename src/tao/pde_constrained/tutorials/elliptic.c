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

  CHKERRQ(PetscInitialize(&argc, &argv, (char*)0,help));
  user.mx = 8;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"elliptic example",NULL);CHKERRQ(ierr);
  CHKERRQ(PetscOptionsInt("-mx","Number of grid points in each direction","",user.mx,&user.mx,NULL));
  user.ns = 6;
  CHKERRQ(PetscOptionsInt("-ns","Number of data samples (1<=ns<=8)","",user.ns,&user.ns,NULL));
  user.ndata = 64;
  CHKERRQ(PetscOptionsInt("-ndata","Numbers of data points per sample","",user.ndata,&user.ndata,NULL));
  user.alpha = 0.1;
  CHKERRQ(PetscOptionsReal("-alpha","Regularization parameter","",user.alpha,&user.alpha,NULL));
  user.beta = 0.00001;
  CHKERRQ(PetscOptionsReal("-beta","Weight attributed to ||u||^2 in regularization functional","",user.beta,&user.beta,NULL));
  user.noise = 0.01;
  CHKERRQ(PetscOptionsReal("-noise","Amount of noise to add to data","",user.noise,&user.noise,NULL));

  user.use_ptap = PETSC_FALSE;
  CHKERRQ(PetscOptionsBool("-use_ptap","Use ptap matrix for DSG","",user.use_ptap,&user.use_ptap,NULL));
  user.use_lrc = PETSC_FALSE;
  CHKERRQ(PetscOptionsBool("-use_lrc","Use lrc matrix for Js","",user.use_lrc,&user.use_lrc,NULL));
  CHKERRQ(PetscOptionsInt("-ntests","Number of times to repeat TaoSolve","",ntests,&ntests,NULL));
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  user.m = user.ns*user.mx*user.mx*user.mx; /* number of constraints */
  user.nstate =  user.m;
  user.ndesign = user.mx*user.mx*user.mx;
  user.n = user.nstate + user.ndesign; /* number of variables */

  /* Create TAO solver and set desired solution method */
  CHKERRQ(TaoCreate(PETSC_COMM_WORLD,&tao));
  CHKERRQ(TaoSetType(tao,TAOLCL));

  /* Set up initial vectors and matrices */
  CHKERRQ(EllipticInitialize(&user));

  CHKERRQ(Gather(user.x,user.y,user.state_scatter,user.u,user.design_scatter));
  CHKERRQ(VecDuplicate(user.x,&x0));
  CHKERRQ(VecCopy(user.x,x0));

  /* Set solution vector with an initial guess */
  CHKERRQ(TaoSetSolution(tao,user.x));
  CHKERRQ(TaoSetObjective(tao, FormFunction, &user));
  CHKERRQ(TaoSetGradient(tao, NULL, FormGradient, &user));
  CHKERRQ(TaoSetConstraintsRoutine(tao, user.c, FormConstraints, &user));

  CHKERRQ(TaoSetJacobianStateRoutine(tao, user.Js, NULL, user.JsInv, FormJacobianState, &user));
  CHKERRQ(TaoSetJacobianDesignRoutine(tao, user.Jd, FormJacobianDesign, &user));

  CHKERRQ(TaoSetStateDesignIS(tao,user.s_is,user.d_is));
  CHKERRQ(TaoSetFromOptions(tao));

  /* SOLVE THE APPLICATION */
  CHKERRQ(PetscLogStageRegister("Trials",&user.stages[1]));
  CHKERRQ(PetscLogStagePush(user.stages[1]));
  for (i=0; i<ntests; i++) {
    CHKERRQ(TaoSolve(tao));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"KSP Iterations = %D\n",user.ksp_its));
    CHKERRQ(VecCopy(x0,user.x));
  }
  CHKERRQ(PetscLogStagePop());
  CHKERRQ(PetscBarrier((PetscObject)user.x));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"KSP iterations within initialization: "));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"%D\n",user.ksp_its_initial));

  CHKERRQ(TaoDestroy(&tao));
  CHKERRQ(VecDestroy(&x0));
  CHKERRQ(EllipticDestroy(&user));
  CHKERRQ(PetscFinalize());
  return 0;
}
/* ------------------------------------------------------------------- */
/*
   dwork = Qy - d
   lwork = L*(u-ur)
   f = 1/2 * (dwork.dwork + alpha*lwork.lwork)
*/
PetscErrorCode FormFunction(Tao tao,Vec X,PetscReal *f,void *ptr)
{
  PetscReal      d1=0,d2=0;
  AppCtx         *user = (AppCtx*)ptr;

  PetscFunctionBegin;
  CHKERRQ(Scatter(X,user->y,user->state_scatter,user->u,user->design_scatter));
  CHKERRQ(MatMult(user->MQ,user->y,user->dwork));
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
  AppCtx         *user = (AppCtx*)ptr;

  PetscFunctionBegin;
  CHKERRQ(Scatter(X,user->y,user->state_scatter,user->u,user->design_scatter));
  CHKERRQ(MatMult(user->MQ,user->y,user->dwork));
  CHKERRQ(VecAXPY(user->dwork,-1.0,user->d));
  CHKERRQ(MatMultTranspose(user->MQ,user->dwork,user->ywork));
  CHKERRQ(VecWAXPY(user->uwork,-1.0,user->ur,user->u));
  CHKERRQ(MatMult(user->L,user->uwork,user->lwork));
  CHKERRQ(MatMultTranspose(user->L,user->lwork,user->uwork));
  CHKERRQ(VecScale(user->uwork, user->alpha));
  CHKERRQ(Gather(G,user->ywork,user->state_scatter,user->uwork,user->design_scatter));
  PetscFunctionReturn(0);
}

PetscErrorCode FormFunctionGradient(Tao tao, Vec X, PetscReal *f, Vec G, void *ptr)
{
  PetscReal      d1,d2;
  AppCtx         *user = (AppCtx*)ptr;

  PetscFunctionBegin;
  CHKERRQ(Scatter(X,user->y,user->state_scatter,user->u,user->design_scatter));
  CHKERRQ(MatMult(user->MQ,user->y,user->dwork));
  CHKERRQ(VecAXPY(user->dwork,-1.0,user->d));
  CHKERRQ(VecDot(user->dwork,user->dwork,&d1));
  CHKERRQ(MatMultTranspose(user->MQ,user->dwork,user->ywork));

  CHKERRQ(VecWAXPY(user->uwork,-1.0,user->ur,user->u));
  CHKERRQ(MatMult(user->L,user->uwork,user->lwork));
  CHKERRQ(VecDot(user->lwork,user->lwork,&d2));
  CHKERRQ(MatMultTranspose(user->L,user->lwork,user->uwork));
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
  /* DSG = Div * (1/Av_u) * Grad */
  CHKERRQ(VecSet(user->uwork,0));
  CHKERRQ(VecAXPY(user->uwork,-1.0,user->u));
  CHKERRQ(VecExp(user->uwork));
  CHKERRQ(MatMult(user->Av,user->uwork,user->Av_u));
  CHKERRQ(VecCopy(user->Av_u,user->Swork));
  CHKERRQ(VecReciprocal(user->Swork));
  if (user->use_ptap) {
    CHKERRQ(MatDiagonalSet(user->Diag,user->Swork,INSERT_VALUES));
    CHKERRQ(MatPtAP(user->Diag,user->Grad,MAT_REUSE_MATRIX,1.0,&user->DSG));
  } else {
    CHKERRQ(MatCopy(user->Div,user->Divwork,SAME_NONZERO_PATTERN));
    CHKERRQ(MatDiagonalScale(user->Divwork,NULL,user->Swork));
    CHKERRQ(MatProductNumeric(user->DSG));
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

PetscErrorCode StateBlockMatMult(Mat J_shell, Vec X, Vec Y)
{
  PetscReal      sum;
  AppCtx         *user;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(J_shell,&user));
  CHKERRQ(MatMult(user->DSG,X,Y));
  CHKERRQ(VecSum(X,&sum));
  sum /= user->ndesign;
  CHKERRQ(VecShift(Y,sum));
  PetscFunctionReturn(0);
}

PetscErrorCode StateMatMult(Mat J_shell, Vec X, Vec Y)
{
  PetscInt       i;
  AppCtx         *user;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(J_shell,&user));
  if (user->ns == 1) {
    CHKERRQ(MatMult(user->JsBlock,X,Y));
  } else {
    for (i=0;i<user->ns;i++) {
      CHKERRQ(Scatter(X,user->subq,user->yi_scatter[i],0,0));
      CHKERRQ(Scatter(Y,user->suby,user->yi_scatter[i],0,0));
      CHKERRQ(MatMult(user->JsBlock,user->subq,user->suby));
      CHKERRQ(Gather(Y,user->suby,user->yi_scatter[i],0,0));
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode StateInvMatMult(Mat J_shell, Vec X, Vec Y)
{
  PetscInt       its,i;
  AppCtx         *user;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(J_shell,&user));
  CHKERRQ(KSPSetOperators(user->solver,user->JsBlock,user->DSG));
  if (Y == user->ytrue) {
    /* First solve is done using true solution to set up problem */
    CHKERRQ(KSPSetTolerances(user->solver,1e-8,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT));
  } else {
    CHKERRQ(KSPSetTolerances(user->solver,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT));
  }
  if (user->ns == 1) {
    CHKERRQ(KSPSolve(user->solver,X,Y));
    CHKERRQ(KSPGetIterationNumber(user->solver,&its));
    user->ksp_its+=its;
  } else {
    for (i=0;i<user->ns;i++) {
      CHKERRQ(Scatter(X,user->subq,user->yi_scatter[i],0,0));
      CHKERRQ(Scatter(Y,user->suby,user->yi_scatter[i],0,0));
      CHKERRQ(KSPSolve(user->solver,user->subq,user->suby));
      CHKERRQ(KSPGetIterationNumber(user->solver,&its));
      user->ksp_its+=its;
      CHKERRQ(Gather(Y,user->suby,user->yi_scatter[i],0,0));
    }
  }
  PetscFunctionReturn(0);
}
PetscErrorCode QMatMult(Mat J_shell, Vec X, Vec Y)
{
  AppCtx         *user;
  PetscInt       i;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(J_shell,&user));
  if (user->ns == 1) {
    CHKERRQ(MatMult(user->Q,X,Y));
  } else {
    for (i=0;i<user->ns;i++) {
      CHKERRQ(Scatter(X,user->subq,user->yi_scatter[i],0,0));
      CHKERRQ(Scatter(Y,user->subd,user->di_scatter[i],0,0));
      CHKERRQ(MatMult(user->Q,user->subq,user->subd));
      CHKERRQ(Gather(Y,user->subd,user->di_scatter[i],0,0));
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode QMatMultTranspose(Mat J_shell, Vec X, Vec Y)
{
  AppCtx         *user;
  PetscInt       i;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(J_shell,&user));
  if (user->ns == 1) {
    CHKERRQ(MatMultTranspose(user->Q,X,Y));
  } else {
    for (i=0;i<user->ns;i++) {
      CHKERRQ(Scatter(X,user->subd,user->di_scatter[i],0,0));
      CHKERRQ(Scatter(Y,user->suby,user->yi_scatter[i],0,0));
      CHKERRQ(MatMultTranspose(user->Q,user->subd,user->suby));
      CHKERRQ(Gather(Y,user->suby,user->yi_scatter[i],0,0));
    }
  }
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

  if (user->ns == 1) {
    /* (sdiag(Grad*y(:,i)) */
    CHKERRQ(MatMult(user->Grad,user->y,user->Twork));

    /* Div * (sdiag(Grad*y(:,i)) * (sdiag(1./((Av*(1./v)).^2)) * (Av * (sdiag(1./v) * b)))) */
    CHKERRQ(VecPointwiseMult(user->Swork,user->Twork,user->Swork));
    CHKERRQ(MatMultTranspose(user->Grad,user->Swork,Y));
  } else {
    for (i=0;i<user->ns;i++) {
      CHKERRQ(Scatter(user->y,user->suby,user->yi_scatter[i],0,0));
      CHKERRQ(Scatter(Y,user->subq,user->yi_scatter[i],0,0));

      CHKERRQ(MatMult(user->Grad,user->suby,user->Twork));
      CHKERRQ(VecPointwiseMult(user->Twork,user->Twork,user->Swork));
      CHKERRQ(MatMultTranspose(user->Grad,user->Twork,user->subq));
      CHKERRQ(Gather(user->y,user->suby,user->yi_scatter[i],0,0));
      CHKERRQ(Gather(Y,user->subq,user->yi_scatter[i],0,0));
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DesignMatMultTranspose(Mat J_shell, Vec X, Vec Y)
{
  PetscInt       i;
  AppCtx         *user;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(J_shell,&user));
  CHKERRQ(VecZeroEntries(Y));

  /* Sdiag = 1./((Av*(1./v)).^2) */
  CHKERRQ(VecSet(user->uwork,0));
  CHKERRQ(VecAXPY(user->uwork,-1.0,user->u));
  CHKERRQ(VecExp(user->uwork));
  CHKERRQ(MatMult(user->Av,user->uwork,user->Swork));
  CHKERRQ(VecPointwiseMult(user->Sdiag,user->Swork,user->Swork));
  CHKERRQ(VecReciprocal(user->Sdiag));

  for (i=0;i<user->ns;i++) {
    CHKERRQ(Scatter(X,user->subq,user->yi_scatter[i],0,0));
    CHKERRQ(Scatter(user->y,user->suby,user->yi_scatter[i],0,0));

    /* Swork = (Div' * b(:,i)) */
    CHKERRQ(MatMult(user->Grad,user->subq,user->Swork));

    /* Twork = Grad*y(:,i) */
    CHKERRQ(MatMult(user->Grad,user->suby,user->Twork));

    /* Twork = sdiag(Twork) * Swork */
    CHKERRQ(VecPointwiseMult(user->Twork,user->Swork,user->Twork));

    /* Swork = pointwisemult(Sdiag,Twork) */
    CHKERRQ(VecPointwiseMult(user->Swork,user->Twork,user->Sdiag));

    /* Ywork = Av' * Swork */
    CHKERRQ(MatMultTranspose(user->Av,user->Swork,user->Ywork));

    /* Ywork = pointwisemult(uwork,Ywork) */
    CHKERRQ(VecPointwiseMult(user->Ywork,user->uwork,user->Ywork));
    CHKERRQ(VecAXPY(Y,1.0,user->Ywork));
    CHKERRQ(Gather(user->y,user->suby,user->yi_scatter[i],0,0));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode FormConstraints(Tao tao, Vec X, Vec C, void *ptr)
{
   /* C=Ay - q      A = Div * Sigma * Grad + hx*hx*hx*ones(n,n) */
   PetscReal      sum;
   PetscInt       i;
   AppCtx         *user = (AppCtx*)ptr;

   PetscFunctionBegin;
   CHKERRQ(Scatter(X,user->y,user->state_scatter,user->u,user->design_scatter));
   if (user->ns == 1) {
     CHKERRQ(MatMult(user->Grad,user->y,user->Swork));
     CHKERRQ(VecPointwiseDivide(user->Swork,user->Swork,user->Av_u));
     CHKERRQ(MatMultTranspose(user->Grad,user->Swork,C));
     CHKERRQ(VecSum(user->y,&sum));
     sum /= user->ndesign;
     CHKERRQ(VecShift(C,sum));
   } else {
     for (i=0;i<user->ns;i++) {
      CHKERRQ(Scatter(user->y,user->suby,user->yi_scatter[i],0,0));
      CHKERRQ(Scatter(C,user->subq,user->yi_scatter[i],0,0));
      CHKERRQ(MatMult(user->Grad,user->suby,user->Swork));
      CHKERRQ(VecPointwiseDivide(user->Swork,user->Swork,user->Av_u));
      CHKERRQ(MatMultTranspose(user->Grad,user->Swork,user->subq));

      CHKERRQ(VecSum(user->suby,&sum));
      sum /= user->ndesign;
      CHKERRQ(VecShift(user->subq,sum));

      CHKERRQ(Gather(user->y,user->suby,user->yi_scatter[i],0,0));
      CHKERRQ(Gather(C,user->subq,user->yi_scatter[i],0,0));
     }
   }
   CHKERRQ(VecAXPY(C,-1.0,user->q));
   PetscFunctionReturn(0);
}

PetscErrorCode Scatter(Vec x, Vec sub1, VecScatter scat1, Vec sub2, VecScatter scat2)
{
  PetscFunctionBegin;
  CHKERRQ(VecScatterBegin(scat1,x,sub1,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(scat1,x,sub1,INSERT_VALUES,SCATTER_FORWARD));
  if (sub2) {
    CHKERRQ(VecScatterBegin(scat2,x,sub2,INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecScatterEnd(scat2,x,sub2,INSERT_VALUES,SCATTER_FORWARD));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode Gather(Vec x, Vec sub1, VecScatter scat1, Vec sub2, VecScatter scat2)
{
  PetscFunctionBegin;
  CHKERRQ(VecScatterBegin(scat1,sub1,x,INSERT_VALUES,SCATTER_REVERSE));
  CHKERRQ(VecScatterEnd(scat1,sub1,x,INSERT_VALUES,SCATTER_REVERSE));
  if (sub2) {
    CHKERRQ(VecScatterBegin(scat2,sub2,x,INSERT_VALUES,SCATTER_REVERSE));
    CHKERRQ(VecScatterEnd(scat2,sub2,x,INSERT_VALUES,SCATTER_REVERSE));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode EllipticInitialize(AppCtx *user)
{
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
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  CHKERRQ(PetscLogStageRegister("Elliptic Setup",&user->stages[0]));
  CHKERRQ(PetscLogStagePush(user->stages[0]));

  /* Create u,y,c,x */
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&user->u));
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&user->y));
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&user->c));
  CHKERRQ(VecSetSizes(user->u,PETSC_DECIDE,user->ndesign));
  CHKERRQ(VecSetFromOptions(user->u));
  CHKERRQ(VecGetLocalSize(user->u,&ysubnlocal));
  CHKERRQ(VecSetSizes(user->y,ysubnlocal*user->ns,user->nstate));
  CHKERRQ(VecSetSizes(user->c,ysubnlocal*user->ns,user->m));
  CHKERRQ(VecSetFromOptions(user->y));
  CHKERRQ(VecSetFromOptions(user->c));

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
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&user->x));

  CHKERRQ(VecGetOwnershipRange(user->y,&lo,&hi));
  CHKERRQ(VecGetOwnershipRange(user->u,&lo2,&hi2));

  CHKERRQ(ISCreateStride(PETSC_COMM_SELF,hi-lo,lo,1,&is_allstate));
  CHKERRQ(ISCreateStride(PETSC_COMM_SELF,hi-lo,lo+lo2,1,&user->s_is));
  CHKERRQ(ISCreateStride(PETSC_COMM_SELF,hi2-lo2,lo2,1,&is_alldesign));
  CHKERRQ(ISCreateStride(PETSC_COMM_SELF,hi2-lo2,hi+lo2,1,&user->d_is));

  CHKERRQ(VecSetSizes(user->x,hi-lo+hi2-lo2,user->n));
  CHKERRQ(VecSetFromOptions(user->x));

  CHKERRQ(VecScatterCreate(user->x,user->s_is,user->y,is_allstate,&user->state_scatter));
  CHKERRQ(VecScatterCreate(user->x,user->d_is,user->u,is_alldesign,&user->design_scatter));
  CHKERRQ(ISDestroy(&is_alldesign));
  CHKERRQ(ISDestroy(&is_allstate));
  /*
     *******************************
     Create scatter from y to y_1,y_2,...,y_ns
     *******************************
  */
  CHKERRQ(PetscMalloc1(user->ns,&user->yi_scatter));
  CHKERRQ(VecDuplicate(user->u,&user->suby));
  CHKERRQ(VecDuplicate(user->u,&user->subq));

  CHKERRQ(VecGetOwnershipRange(user->y,&lo2,&hi2));
  istart = 0;
  for (i=0; i<user->ns; i++) {
    CHKERRQ(VecGetOwnershipRange(user->suby,&lo,&hi));
    CHKERRQ(ISCreateStride(PETSC_COMM_SELF,hi-lo,lo2+istart,1,&is_from_y));
    CHKERRQ(VecScatterCreate(user->y,is_from_y,user->suby,NULL,&user->yi_scatter[i]));
    istart = istart + hi-lo;
    CHKERRQ(ISDestroy(&is_from_y));
  }
  /*
     *******************************
     Create scatter from d to d_1,d_2,...,d_ns
     *******************************
  */
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&user->subd));
  CHKERRQ(VecSetSizes(user->subd,PETSC_DECIDE,user->ndata));
  CHKERRQ(VecSetFromOptions(user->subd));
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&user->d));
  CHKERRQ(VecGetLocalSize(user->subd,&dsubnlocal));
  CHKERRQ(VecSetSizes(user->d,dsubnlocal*user->ns,user->ndata*user->ns));
  CHKERRQ(VecSetFromOptions(user->d));
  CHKERRQ(PetscMalloc1(user->ns,&user->di_scatter));

  CHKERRQ(VecGetOwnershipRange(user->d,&lo2,&hi2));
  istart = 0;
  for (i=0; i<user->ns; i++) {
    CHKERRQ(VecGetOwnershipRange(user->subd,&lo,&hi));
    CHKERRQ(ISCreateStride(PETSC_COMM_SELF,hi-lo,lo2+istart,1,&is_from_d));
    CHKERRQ(VecScatterCreate(user->d,is_from_d,user->subd,NULL,&user->di_scatter[i]));
    istart = istart + hi-lo;
    CHKERRQ(ISDestroy(&is_from_d));
  }

  CHKERRQ(PetscMalloc1(user->mx,&x));
  CHKERRQ(PetscMalloc1(user->mx,&y));
  CHKERRQ(PetscMalloc1(user->mx,&z));

  user->ksp_its = 0;
  user->ksp_its_initial = 0;

  n = user->mx * user->mx * user->mx;
  m = 3 * user->mx * user->mx * (user->mx-1);
  sqrt_beta = PetscSqrtScalar(user->beta);

  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&XX));
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&user->q));
  CHKERRQ(VecSetSizes(XX,ysubnlocal,n));
  CHKERRQ(VecSetSizes(user->q,ysubnlocal*user->ns,user->m));
  CHKERRQ(VecSetFromOptions(XX));
  CHKERRQ(VecSetFromOptions(user->q));

  CHKERRQ(VecDuplicate(XX,&YY));
  CHKERRQ(VecDuplicate(XX,&ZZ));
  CHKERRQ(VecDuplicate(XX,&XXwork));
  CHKERRQ(VecDuplicate(XX,&YYwork));
  CHKERRQ(VecDuplicate(XX,&ZZwork));
  CHKERRQ(VecDuplicate(XX,&UTwork));
  CHKERRQ(VecDuplicate(XX,&user->utrue));

  /* map for striding q */
  CHKERRQ(VecGetOwnershipRanges(user->q,&ranges));
  CHKERRQ(VecGetOwnershipRanges(user->u,&subranges));

  CHKERRQ(VecGetOwnershipRange(user->q,&lo2,&hi2));
  CHKERRQ(VecGetOwnershipRange(user->u,&lo,&hi));
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
    for (is=0; is<2; is++) {
      for (js=0; js<2; js++) {
        for (ks=0; ks<2; ks++) {
          ls = is*4 + js*2 + ks;
          if (ls<user->ns) {
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
            CHKERRQ(VecSetValues(user->q,1,&l,&v,INSERT_VALUES));
          }
        }
      }
    }
  }

  CHKERRQ(VecAssemblyBegin(XX));
  CHKERRQ(VecAssemblyEnd(XX));
  CHKERRQ(VecAssemblyBegin(YY));
  CHKERRQ(VecAssemblyEnd(YY));
  CHKERRQ(VecAssemblyBegin(ZZ));
  CHKERRQ(VecAssemblyEnd(ZZ));
  CHKERRQ(VecAssemblyBegin(user->q));
  CHKERRQ(VecAssemblyEnd(user->q));

  /* Compute true parameter function
     ut = exp(-((x-0.25)^2+(y-0.25)^2+(z-0.25)^2)/0.05) - exp((x-0.75)^2-(y-0.75)^2-(z-0.75))^2/0.05) */
  CHKERRQ(VecCopy(XX,XXwork));
  CHKERRQ(VecCopy(YY,YYwork));
  CHKERRQ(VecCopy(ZZ,ZZwork));

  CHKERRQ(VecShift(XXwork,-0.25));
  CHKERRQ(VecShift(YYwork,-0.25));
  CHKERRQ(VecShift(ZZwork,-0.25));

  CHKERRQ(VecPointwiseMult(XXwork,XXwork,XXwork));
  CHKERRQ(VecPointwiseMult(YYwork,YYwork,YYwork));
  CHKERRQ(VecPointwiseMult(ZZwork,ZZwork,ZZwork));

  CHKERRQ(VecCopy(XXwork,UTwork));
  CHKERRQ(VecAXPY(UTwork,1.0,YYwork));
  CHKERRQ(VecAXPY(UTwork,1.0,ZZwork));
  CHKERRQ(VecScale(UTwork,-20.0));
  CHKERRQ(VecExp(UTwork));
  CHKERRQ(VecCopy(UTwork,user->utrue));

  CHKERRQ(VecCopy(XX,XXwork));
  CHKERRQ(VecCopy(YY,YYwork));
  CHKERRQ(VecCopy(ZZ,ZZwork));

  CHKERRQ(VecShift(XXwork,-0.75));
  CHKERRQ(VecShift(YYwork,-0.75));
  CHKERRQ(VecShift(ZZwork,-0.75));

  CHKERRQ(VecPointwiseMult(XXwork,XXwork,XXwork));
  CHKERRQ(VecPointwiseMult(YYwork,YYwork,YYwork));
  CHKERRQ(VecPointwiseMult(ZZwork,ZZwork,ZZwork));

  CHKERRQ(VecCopy(XXwork,UTwork));
  CHKERRQ(VecAXPY(UTwork,1.0,YYwork));
  CHKERRQ(VecAXPY(UTwork,1.0,ZZwork));
  CHKERRQ(VecScale(UTwork,-20.0));
  CHKERRQ(VecExp(UTwork));

  CHKERRQ(VecAXPY(user->utrue,-1.0,UTwork));

  CHKERRQ(VecDestroy(&XX));
  CHKERRQ(VecDestroy(&YY));
  CHKERRQ(VecDestroy(&ZZ));
  CHKERRQ(VecDestroy(&XXwork));
  CHKERRQ(VecDestroy(&YYwork));
  CHKERRQ(VecDestroy(&ZZwork));
  CHKERRQ(VecDestroy(&UTwork));

  /* Initial guess and reference model */
  CHKERRQ(VecDuplicate(user->utrue,&user->ur));
  CHKERRQ(VecSum(user->utrue,&meanut));
  meanut = meanut / n;
  CHKERRQ(VecSet(user->ur,meanut));
  CHKERRQ(VecCopy(user->ur,user->u));

  /* Generate Grad matrix */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&user->Grad));
  CHKERRQ(MatSetSizes(user->Grad,PETSC_DECIDE,ysubnlocal,m,n));
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
  CHKERRQ(MatSetSizes(user->Av,PETSC_DECIDE,ysubnlocal,m,n));
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

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&user->L));
  CHKERRQ(MatSetSizes(user->L,PETSC_DECIDE,ysubnlocal,m+n,n));
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
  if (!user->use_ptap) {
    /* Generate Div matrix */
    CHKERRQ(MatCreate(PETSC_COMM_WORLD,&user->Div));
    CHKERRQ(MatSetSizes(user->Div,ysubnlocal,PETSC_DECIDE,n,m));
    CHKERRQ(MatSetFromOptions(user->Div));
    CHKERRQ(MatMPIAIJSetPreallocation(user->Div,4,NULL,4,NULL));
    CHKERRQ(MatSeqAIJSetPreallocation(user->Div,6,NULL));
    CHKERRQ(MatGetOwnershipRange(user->Grad,&istart,&iend));

    for (i=istart; i<iend; i++) {
      if (i<m/3) {
        iblock = i / (user->mx-1);
        j = iblock*user->mx + (i % (user->mx-1));
        CHKERRQ(MatSetValues(user->Div,1,&j,1,&i,&neg_hinv,INSERT_VALUES));
        j = j+1;
        CHKERRQ(MatSetValues(user->Div,1,&j,1,&i,&hinv,INSERT_VALUES));
      }
      if (i>=m/3 && i<2*m/3) {
        iblock = (i-m/3) / (user->mx*(user->mx-1));
        j = iblock*user->mx*user->mx + ((i-m/3) % (user->mx*(user->mx-1)));
        CHKERRQ(MatSetValues(user->Div,1,&j,1,&i,&neg_hinv,INSERT_VALUES));
        j = j + user->mx;
        CHKERRQ(MatSetValues(user->Div,1,&j,1,&i,&hinv,INSERT_VALUES));
      }
      if (i>=2*m/3) {
        j = i-2*m/3;
        CHKERRQ(MatSetValues(user->Div,1,&j,1,&i,&neg_hinv,INSERT_VALUES));
        j = j + user->mx*user->mx;
        CHKERRQ(MatSetValues(user->Div,1,&j,1,&i,&hinv,INSERT_VALUES));
      }
    }

    CHKERRQ(MatAssemblyBegin(user->Div,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(user->Div,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatDuplicate(user->Div,MAT_SHARE_NONZERO_PATTERN,&user->Divwork));
  } else {
    CHKERRQ(MatCreate(PETSC_COMM_WORLD,&user->Diag));
    CHKERRQ(MatSetSizes(user->Diag,PETSC_DECIDE,PETSC_DECIDE,m,m));
    CHKERRQ(MatSetFromOptions(user->Diag));
    CHKERRQ(MatMPIAIJSetPreallocation(user->Diag,1,NULL,0,NULL));
    CHKERRQ(MatSeqAIJSetPreallocation(user->Diag,1,NULL));
  }

  /* Build work vectors and matrices */
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&user->S));
  CHKERRQ(VecSetSizes(user->S, PETSC_DECIDE, m));
  CHKERRQ(VecSetFromOptions(user->S));

  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&user->lwork));
  CHKERRQ(VecSetSizes(user->lwork,PETSC_DECIDE,m+user->mx*user->mx*user->mx));
  CHKERRQ(VecSetFromOptions(user->lwork));

  CHKERRQ(MatDuplicate(user->Av,MAT_SHARE_NONZERO_PATTERN,&user->Avwork));

  CHKERRQ(VecDuplicate(user->S,&user->Swork));
  CHKERRQ(VecDuplicate(user->S,&user->Sdiag));
  CHKERRQ(VecDuplicate(user->S,&user->Av_u));
  CHKERRQ(VecDuplicate(user->S,&user->Twork));
  CHKERRQ(VecDuplicate(user->y,&user->ywork));
  CHKERRQ(VecDuplicate(user->u,&user->Ywork));
  CHKERRQ(VecDuplicate(user->u,&user->uwork));
  CHKERRQ(VecDuplicate(user->u,&user->js_diag));
  CHKERRQ(VecDuplicate(user->c,&user->cwork));
  CHKERRQ(VecDuplicate(user->d,&user->dwork));

  /* Create a matrix-free shell user->Jd for computing B*x */
  CHKERRQ(MatCreateShell(PETSC_COMM_WORLD,ysubnlocal*user->ns,ysubnlocal,user->nstate,user->ndesign,user,&user->Jd));
  CHKERRQ(MatShellSetOperation(user->Jd,MATOP_MULT,(void(*)(void))DesignMatMult));
  CHKERRQ(MatShellSetOperation(user->Jd,MATOP_MULT_TRANSPOSE,(void(*)(void))DesignMatMultTranspose));

  /* Compute true state function ytrue given utrue */
  CHKERRQ(VecDuplicate(user->y,&user->ytrue));

  /* First compute Av_u = Av*exp(-u) */
  CHKERRQ(VecSet(user->uwork, 0));
  CHKERRQ(VecAXPY(user->uwork,-1.0,user->utrue)); /* Note: user->utrue */
  CHKERRQ(VecExp(user->uwork));
  CHKERRQ(MatMult(user->Av,user->uwork,user->Av_u));

  /* Next form DSG = Div*S*Grad */
  CHKERRQ(VecCopy(user->Av_u,user->Swork));
  CHKERRQ(VecReciprocal(user->Swork));
  if (user->use_ptap) {
    CHKERRQ(MatDiagonalSet(user->Diag,user->Swork,INSERT_VALUES));
    CHKERRQ(MatPtAP(user->Diag,user->Grad,MAT_INITIAL_MATRIX,1.0,&user->DSG));
  } else {
    CHKERRQ(MatCopy(user->Div,user->Divwork,SAME_NONZERO_PATTERN));
    CHKERRQ(MatDiagonalScale(user->Divwork,NULL,user->Swork));

    CHKERRQ(MatMatMult(user->Divwork,user->Grad,MAT_INITIAL_MATRIX,1.0,&user->DSG));
  }

  CHKERRQ(MatSetOption(user->DSG,MAT_SYMMETRIC,PETSC_TRUE));
  CHKERRQ(MatSetOption(user->DSG,MAT_SYMMETRY_ETERNAL,PETSC_TRUE));

  if (user->use_lrc == PETSC_TRUE) {
    v=PetscSqrtReal(1.0 /user->ndesign);
    CHKERRQ(PetscMalloc1(user->ndesign,&user->ones));

    for (i=0;i<user->ndesign;i++) {
      user->ones[i]=v;
    }
    CHKERRQ(MatCreateDense(PETSC_COMM_WORLD,ysubnlocal,PETSC_DECIDE,user->ndesign,1,user->ones,&user->Ones));
    CHKERRQ(MatAssemblyBegin(user->Ones, MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(user->Ones, MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatCreateLRC(user->DSG,user->Ones,NULL,user->Ones,&user->JsBlock));
    CHKERRQ(MatSetUp(user->JsBlock));
  } else {
    /* Create matrix-free shell user->Js for computing (A + h^3*e*e^T)*x */
    CHKERRQ(MatCreateShell(PETSC_COMM_WORLD,ysubnlocal,ysubnlocal,user->ndesign,user->ndesign,user,&user->JsBlock));
    CHKERRQ(MatShellSetOperation(user->JsBlock,MATOP_MULT,(void(*)(void))StateBlockMatMult));
    CHKERRQ(MatShellSetOperation(user->JsBlock,MATOP_MULT_TRANSPOSE,(void(*)(void))StateBlockMatMult));
  }
  CHKERRQ(MatSetOption(user->JsBlock,MAT_SYMMETRIC,PETSC_TRUE));
  CHKERRQ(MatSetOption(user->JsBlock,MAT_SYMMETRY_ETERNAL,PETSC_TRUE));
  CHKERRQ(MatCreateShell(PETSC_COMM_WORLD,ysubnlocal*user->ns,ysubnlocal*user->ns,user->nstate,user->nstate,user,&user->Js));
  CHKERRQ(MatShellSetOperation(user->Js,MATOP_MULT,(void(*)(void))StateMatMult));
  CHKERRQ(MatShellSetOperation(user->Js,MATOP_MULT_TRANSPOSE,(void(*)(void))StateMatMult));
  CHKERRQ(MatSetOption(user->Js,MAT_SYMMETRIC,PETSC_TRUE));
  CHKERRQ(MatSetOption(user->Js,MAT_SYMMETRY_ETERNAL,PETSC_TRUE));

  CHKERRQ(MatCreateShell(PETSC_COMM_WORLD,ysubnlocal*user->ns,ysubnlocal*user->ns,user->nstate,user->nstate,user,&user->JsInv));
  CHKERRQ(MatShellSetOperation(user->JsInv,MATOP_MULT,(void(*)(void))StateInvMatMult));
  CHKERRQ(MatShellSetOperation(user->JsInv,MATOP_MULT_TRANSPOSE,(void(*)(void))StateInvMatMult));
  CHKERRQ(MatSetOption(user->JsInv,MAT_SYMMETRIC,PETSC_TRUE));
  CHKERRQ(MatSetOption(user->JsInv,MAT_SYMMETRY_ETERNAL,PETSC_TRUE));

  CHKERRQ(MatSetOption(user->DSG,MAT_SYMMETRIC,PETSC_TRUE));
  CHKERRQ(MatSetOption(user->DSG,MAT_SYMMETRY_ETERNAL,PETSC_TRUE));
  /* Now solve for ytrue */
  CHKERRQ(KSPCreate(PETSC_COMM_WORLD,&user->solver));
  CHKERRQ(KSPSetFromOptions(user->solver));

  CHKERRQ(KSPSetOperators(user->solver,user->JsBlock,user->DSG));

  CHKERRQ(MatMult(user->JsInv,user->q,user->ytrue));
  /* First compute Av_u = Av*exp(-u) */
  CHKERRQ(VecSet(user->uwork,0));
  CHKERRQ(VecAXPY(user->uwork,-1.0,user->u)); /* Note: user->u */
  CHKERRQ(VecExp(user->uwork));
  CHKERRQ(MatMult(user->Av,user->uwork,user->Av_u));

  /* Next update DSG = Div*S*Grad  with user->u */
  CHKERRQ(VecCopy(user->Av_u,user->Swork));
  CHKERRQ(VecReciprocal(user->Swork));
  if (user->use_ptap) {
    CHKERRQ(MatDiagonalSet(user->Diag,user->Swork,INSERT_VALUES));
    CHKERRQ(MatPtAP(user->Diag,user->Grad,MAT_REUSE_MATRIX,1.0,&user->DSG));
  } else {
    CHKERRQ(MatCopy(user->Div,user->Divwork,SAME_NONZERO_PATTERN));
    CHKERRQ(MatDiagonalScale(user->Divwork,NULL,user->Av_u));
    CHKERRQ(MatProductNumeric(user->DSG));
  }

  /* Now solve for y */

  CHKERRQ(MatMult(user->JsInv,user->q,user->y));

  user->ksp_its_initial = user->ksp_its;
  user->ksp_its = 0;
  /* Construct projection matrix Q (blocks) */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&user->Q));
  CHKERRQ(MatSetSizes(user->Q,dsubnlocal,ysubnlocal,user->ndata,user->ndesign));
  CHKERRQ(MatSetFromOptions(user->Q));
  CHKERRQ(MatMPIAIJSetPreallocation(user->Q,8,NULL,8,NULL));
  CHKERRQ(MatSeqAIJSetPreallocation(user->Q,8,NULL));

  for (i=0; i<user->mx; i++) {
    x[i] = h*(i+0.5);
    y[i] = h*(i+0.5);
    z[i] = h*(i+0.5);
  }
  CHKERRQ(MatGetOwnershipRange(user->Q,&istart,&iend));

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
    CHKERRQ(MatSetValues(user->Q,1,&i,1,&j,&v,INSERT_VALUES));

    j = indx1 + indy1*nx + indz2*nx*ny;
    v = (1-dx1/Dx)*(1-dy1/Dy)*(1-dz2/Dz);
    CHKERRQ(MatSetValues(user->Q,1,&i,1,&j,&v,INSERT_VALUES));

    j = indx1 + indy2*nx + indz1*nx*ny;
    v = (1-dx1/Dx)*(1-dy2/Dy)*(1-dz1/Dz);
    CHKERRQ(MatSetValues(user->Q,1,&i,1,&j,&v,INSERT_VALUES));

    j = indx1 + indy2*nx + indz2*nx*ny;
    v = (1-dx1/Dx)*(1-dy2/Dy)*(1-dz2/Dz);
    CHKERRQ(MatSetValues(user->Q,1,&i,1,&j,&v,INSERT_VALUES));

    j = indx2 + indy1*nx + indz1*nx*ny;
    v = (1-dx2/Dx)*(1-dy1/Dy)*(1-dz1/Dz);
    CHKERRQ(MatSetValues(user->Q,1,&i,1,&j,&v,INSERT_VALUES));

    j = indx2 + indy1*nx + indz2*nx*ny;
    v = (1-dx2/Dx)*(1-dy1/Dy)*(1-dz2/Dz);
    CHKERRQ(MatSetValues(user->Q,1,&i,1,&j,&v,INSERT_VALUES));

    j = indx2 + indy2*nx + indz1*nx*ny;
    v = (1-dx2/Dx)*(1-dy2/Dy)*(1-dz1/Dz);
    CHKERRQ(MatSetValues(user->Q,1,&i,1,&j,&v,INSERT_VALUES));

    j = indx2 + indy2*nx + indz2*nx*ny;
    v = (1-dx2/Dx)*(1-dy2/Dy)*(1-dz2/Dz);
    CHKERRQ(MatSetValues(user->Q,1,&i,1,&j,&v,INSERT_VALUES));
  }

  CHKERRQ(MatAssemblyBegin(user->Q,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(user->Q,MAT_FINAL_ASSEMBLY));
  /* Create MQ (composed of blocks of Q */
  CHKERRQ(MatCreateShell(PETSC_COMM_WORLD,dsubnlocal*user->ns,PETSC_DECIDE,user->ndata*user->ns,user->nstate,user,&user->MQ));
  CHKERRQ(MatShellSetOperation(user->MQ,MATOP_MULT,(void(*)(void))QMatMult));
  CHKERRQ(MatShellSetOperation(user->MQ,MATOP_MULT_TRANSPOSE,(void(*)(void))QMatMultTranspose));

  /* Add noise to the measurement data */
  CHKERRQ(VecSet(user->ywork,1.0));
  CHKERRQ(VecAYPX(user->ywork,user->noise,user->ytrue));
  CHKERRQ(MatMult(user->MQ,user->ywork,user->d));

  /* Now that initial conditions have been set, let the user pass tolerance options to the KSP solver */
  CHKERRQ(PetscFree(x));
  CHKERRQ(PetscFree(y));
  CHKERRQ(PetscFree(z));
  CHKERRQ(PetscLogStagePop());
  PetscFunctionReturn(0);
}

PetscErrorCode EllipticDestroy(AppCtx *user)
{
  PetscInt       i;

  PetscFunctionBegin;
  CHKERRQ(MatDestroy(&user->DSG));
  CHKERRQ(KSPDestroy(&user->solver));
  CHKERRQ(MatDestroy(&user->Q));
  CHKERRQ(MatDestroy(&user->MQ));
  if (!user->use_ptap) {
    CHKERRQ(MatDestroy(&user->Div));
    CHKERRQ(MatDestroy(&user->Divwork));
  } else {
    CHKERRQ(MatDestroy(&user->Diag));
  }
  if (user->use_lrc) {
    CHKERRQ(MatDestroy(&user->Ones));
  }

  CHKERRQ(MatDestroy(&user->Grad));
  CHKERRQ(MatDestroy(&user->Av));
  CHKERRQ(MatDestroy(&user->Avwork));
  CHKERRQ(MatDestroy(&user->L));
  CHKERRQ(MatDestroy(&user->Js));
  CHKERRQ(MatDestroy(&user->Jd));
  CHKERRQ(MatDestroy(&user->JsBlock));
  CHKERRQ(MatDestroy(&user->JsInv));

  CHKERRQ(VecDestroy(&user->x));
  CHKERRQ(VecDestroy(&user->u));
  CHKERRQ(VecDestroy(&user->uwork));
  CHKERRQ(VecDestroy(&user->utrue));
  CHKERRQ(VecDestroy(&user->y));
  CHKERRQ(VecDestroy(&user->ywork));
  CHKERRQ(VecDestroy(&user->ytrue));
  CHKERRQ(VecDestroy(&user->c));
  CHKERRQ(VecDestroy(&user->cwork));
  CHKERRQ(VecDestroy(&user->ur));
  CHKERRQ(VecDestroy(&user->q));
  CHKERRQ(VecDestroy(&user->d));
  CHKERRQ(VecDestroy(&user->dwork));
  CHKERRQ(VecDestroy(&user->lwork));
  CHKERRQ(VecDestroy(&user->S));
  CHKERRQ(VecDestroy(&user->Swork));
  CHKERRQ(VecDestroy(&user->Sdiag));
  CHKERRQ(VecDestroy(&user->Ywork));
  CHKERRQ(VecDestroy(&user->Twork));
  CHKERRQ(VecDestroy(&user->Av_u));
  CHKERRQ(VecDestroy(&user->js_diag));
  CHKERRQ(ISDestroy(&user->s_is));
  CHKERRQ(ISDestroy(&user->d_is));
  CHKERRQ(VecDestroy(&user->suby));
  CHKERRQ(VecDestroy(&user->subd));
  CHKERRQ(VecDestroy(&user->subq));
  CHKERRQ(VecScatterDestroy(&user->state_scatter));
  CHKERRQ(VecScatterDestroy(&user->design_scatter));
  for (i=0;i<user->ns;i++) {
    CHKERRQ(VecScatterDestroy(&user->yi_scatter[i]));
    CHKERRQ(VecScatterDestroy(&user->di_scatter[i]));
  }
  CHKERRQ(PetscFree(user->yi_scatter));
  CHKERRQ(PetscFree(user->di_scatter));
  if (user->use_lrc) {
    CHKERRQ(PetscFree(user->ones));
    CHKERRQ(MatDestroy(&user->Ones));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode EllipticMonitor(Tao tao, void *ptr)
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
      args: -tao_cmonitor -ns 1 -tao_type lcl -tao_gatol 1.e-3 -tao_max_it 11
      requires: !single

   test:
      suffix: 2
      args: -tao_cmonitor -tao_type lcl -tao_max_it 11 -use_ptap -use_lrc -ns 1 -tao_gatol 1.e-3
      requires: !single

TEST*/
