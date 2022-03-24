/* Program usage: mpiexec -n 2 ./ex1 [-help] [all TAO options] */

/* ----------------------------------------------------------------------
min f(x) = (x0 - 2)^2 + (x1 - 2)^2 - 2*(x0 + x1)
s.t.  x0^2 + x1 - 2 = 0
      0  <= x0^2 - x1 <= 1
      -1 <= x0, x1 <= 2
-->
      g(x)  = 0
      h(x) >= 0
      -1 <= x0, x1 <= 2
where
      g(x) = x0^2 + x1 - 2
      h(x) = [x0^2 - x1
              1 -(x0^2 - x1)]
---------------------------------------------------------------------- */

#include <petsctao.h>

static  char help[]= "Solves constrained optimiztion problem using pdipm.\n\
Input parameters include:\n\
  -tao_type pdipm    : sets Tao solver\n\
  -no_eq             : removes the equaility constraints from the problem\n\
  -init_view         : view initial object setup\n\
  -snes_fd           : snes with finite difference Jacobian (needed for pdipm)\n\
  -snes_compare_explicit : compare user Jacobian with finite difference Jacobian \n\
  -tao_cmonitor      : convergence monitor with constraint norm \n\
  -tao_view_solution : view exact solution at each iteration\n\
  Note: external package MUMPS is required to run pdipm in parallel. This is designed for a maximum of 2 processors, the code will error if size > 2.\n";

/*
   User-defined application context - contains data needed by the application
*/
typedef struct {
  PetscInt   n;  /* Global length of x */
  PetscInt   ne; /* Global number of equality constraints */
  PetscInt   ni; /* Global number of inequality constraints */
  PetscBool  noeqflag, initview;
  Vec        x,xl,xu;
  Vec        ce,ci,bl,bu,Xseq;
  Mat        Ae,Ai,H;
  VecScatter scat;
} AppCtx;

/* -------- User-defined Routines --------- */
PetscErrorCode InitializeProblem(AppCtx *);
PetscErrorCode DestroyProblem(AppCtx *);
PetscErrorCode FormFunctionGradient(Tao,Vec,PetscReal *,Vec,void *);
PetscErrorCode FormHessian(Tao,Vec,Mat,Mat, void*);
PetscErrorCode FormInequalityConstraints(Tao,Vec,Vec,void*);
PetscErrorCode FormEqualityConstraints(Tao,Vec,Vec,void*);
PetscErrorCode FormInequalityJacobian(Tao,Vec,Mat,Mat, void*);
PetscErrorCode FormEqualityJacobian(Tao,Vec,Mat,Mat, void*);

PetscErrorCode main(int argc,char **argv)
{
  Tao            tao;
  KSP            ksp;
  PC             pc;
  AppCtx         user;  /* application context */
  Vec            x,G,CI,CE;
  PetscMPIInt    size;
  TaoType        type;
  PetscReal      f;
  PetscBool      pdipm;

  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheck(size <= 2,PETSC_COMM_SELF,PETSC_ERR_WRONG_MPI_SIZE,"More than 2 processors detected. Example written to use max of 2 processors.");

  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"---- Constrained Problem -----\n"));
  CHKERRQ(InitializeProblem(&user)); /* sets up problem, function below */
  CHKERRQ(TaoCreate(PETSC_COMM_WORLD,&tao));
  CHKERRQ(TaoSetType(tao,TAOPDIPM));
  CHKERRQ(TaoSetSolution(tao,user.x)); /* gets solution vector from problem */
  CHKERRQ(TaoSetVariableBounds(tao,user.xl,user.xu)); /* sets lower upper bounds from given solution */
  CHKERRQ(TaoSetObjectiveAndGradient(tao,NULL,FormFunctionGradient,(void*)&user));

  if (!user.noeqflag) {
    CHKERRQ(TaoSetEqualityConstraintsRoutine(tao,user.ce,FormEqualityConstraints,(void*)&user));
  }
  CHKERRQ(TaoSetInequalityConstraintsRoutine(tao,user.ci,FormInequalityConstraints,(void*)&user));
  if (!user.noeqflag) {
    CHKERRQ(TaoSetJacobianEqualityRoutine(tao,user.Ae,user.Ae,FormEqualityJacobian,(void*)&user)); /* equality jacobian */
  }
  CHKERRQ(TaoSetJacobianInequalityRoutine(tao,user.Ai,user.Ai,FormInequalityJacobian,(void*)&user)); /* inequality jacobian */
  CHKERRQ(TaoSetTolerances(tao,1.e-6,1.e-6,1.e-6));
  CHKERRQ(TaoSetConstraintTolerances(tao,1.e-6,1.e-6));

  CHKERRQ(TaoGetKSP(tao,&ksp));
  CHKERRQ(KSPGetPC(ksp,&pc));
  CHKERRQ(PCSetType(pc,PCCHOLESKY));
  /*
      This algorithm produces matrices with zeros along the diagonal therefore we use
    MUMPS which provides solver for indefinite matrices
  */
#if defined(PETSC_HAVE_MUMPS)
  CHKERRQ(PCFactorSetMatSolverType(pc,MATSOLVERMUMPS));  /* requires mumps to solve pdipm */
#else
  PetscCheck(size == 1,PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"Requires an external package that supports parallel PCCHOLESKY, e.g., MUMPS.");
#endif
  CHKERRQ(KSPSetType(ksp,KSPPREONLY));
  CHKERRQ(KSPSetFromOptions(ksp));

  CHKERRQ(TaoSetFromOptions(tao));
  CHKERRQ(TaoGetType(tao,&type));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)tao,TAOPDIPM,&pdipm));
  if (pdipm) {
    CHKERRQ(TaoSetHessian(tao,user.H,user.H,FormHessian,(void*)&user));
    if (user.initview) {
      CHKERRQ(TaoSetUp(tao));
      CHKERRQ(VecDuplicate(user.x, &G));
      CHKERRQ(FormFunctionGradient(tao, user.x, &f, G, (void*)&user));
      CHKERRQ(FormHessian(tao, user.x, user.H, user.H, (void*)&user));
      CHKERRQ(PetscViewerASCIIPushTab(PETSC_VIEWER_STDOUT_WORLD));
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nInitial point X:\n",f));
      CHKERRQ(VecView(user.x, PETSC_VIEWER_STDOUT_WORLD));
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nInitial objective f(x) = %g\n",f));
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nInitial gradient and Hessian:\n",f));
      CHKERRQ(VecView(G, PETSC_VIEWER_STDOUT_WORLD));
      CHKERRQ(MatView(user.H, PETSC_VIEWER_STDOUT_WORLD));
      CHKERRQ(VecDestroy(&G));
      CHKERRQ(FormInequalityJacobian(tao, user.x, user.Ai, user.Ai, (void*)&user));
      CHKERRQ(MatCreateVecs(user.Ai, NULL, &CI));
      CHKERRQ(FormInequalityConstraints(tao, user.x, CI, (void*)&user));
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nInitial inequality constraints and Jacobian:\n",f));
      CHKERRQ(VecView(CI, PETSC_VIEWER_STDOUT_WORLD));
      CHKERRQ(MatView(user.Ai, PETSC_VIEWER_STDOUT_WORLD));
      CHKERRQ(VecDestroy(&CI));
      if (!user.noeqflag) {
        CHKERRQ(FormEqualityJacobian(tao, user.x, user.Ae, user.Ae, (void*)&user));
        CHKERRQ(MatCreateVecs(user.Ae, NULL, &CE));
        CHKERRQ(FormEqualityConstraints(tao, user.x, CE, (void*)&user));
        CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nInitial equality constraints and Jacobian:\n",f));
        CHKERRQ(VecView(CE, PETSC_VIEWER_STDOUT_WORLD));
        CHKERRQ(MatView(user.Ae, PETSC_VIEWER_STDOUT_WORLD));
        CHKERRQ(VecDestroy(&CE));
      }
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "\n"));
      CHKERRQ(PetscViewerASCIIPopTab(PETSC_VIEWER_STDOUT_WORLD));
    }
  }

  CHKERRQ(TaoSolve(tao));
  CHKERRQ(TaoGetSolution(tao,&x));
  CHKERRQ(VecView(x,PETSC_VIEWER_STDOUT_WORLD));

  /* Free objects */
  CHKERRQ(DestroyProblem(&user));
  CHKERRQ(TaoDestroy(&tao));
  CHKERRQ(PetscFinalize());
  return 0;
}

PetscErrorCode InitializeProblem(AppCtx *user)
{
  PetscMPIInt    size;
  PetscMPIInt    rank;
  PetscInt       nloc,neloc,niloc;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  user->noeqflag = PETSC_FALSE;
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-no_eq",&user->noeqflag,NULL));
  user->initview = PETSC_FALSE;
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-init_view",&user->initview,NULL));

  if (!user->noeqflag) {
    /* Tell user the correct solution, not an error checking */
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Solution should be f(1,1)=-2\n"));
  }

  /* create vector x and set initial values */
  user->n = 2; /* global length */
  nloc = (size==1)?user->n:1;
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&user->x));
  CHKERRQ(VecSetSizes(user->x,nloc,user->n));
  CHKERRQ(VecSetFromOptions(user->x));
  CHKERRQ(VecSet(user->x,0));

  /* create and set lower and upper bound vectors */
  CHKERRQ(VecDuplicate(user->x,&user->xl));
  CHKERRQ(VecDuplicate(user->x,&user->xu));
  CHKERRQ(VecSet(user->xl,-1.0));
  CHKERRQ(VecSet(user->xu,2.0));

  /* create scater to zero */
  CHKERRQ(VecScatterCreateToZero(user->x,&user->scat,&user->Xseq));

  user->ne = 1;
  user->ni = 2;
  neloc = (rank==0)?user->ne:0;
  niloc = (size==1)?user->ni:1;

  if (!user->noeqflag) {
    CHKERRQ(VecCreate(PETSC_COMM_WORLD,&user->ce)); /* a 1x1 vec for equality constraints */
    CHKERRQ(VecSetSizes(user->ce,neloc,user->ne));
    CHKERRQ(VecSetFromOptions(user->ce));
    CHKERRQ(VecSetUp(user->ce));
  }

  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&user->ci)); /* a 2x1 vec for inequality constraints */
  CHKERRQ(VecSetSizes(user->ci,niloc,user->ni));
  CHKERRQ(VecSetFromOptions(user->ci));
  CHKERRQ(VecSetUp(user->ci));

  /* nexn & nixn matricies for equally and inequalty constraints */
  if (!user->noeqflag) {
    CHKERRQ(MatCreate(PETSC_COMM_WORLD,&user->Ae));
    CHKERRQ(MatSetSizes(user->Ae,neloc,nloc,user->ne,user->n));
    CHKERRQ(MatSetFromOptions(user->Ae));
    CHKERRQ(MatSetUp(user->Ae));
  }

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&user->Ai));
  CHKERRQ(MatSetSizes(user->Ai,niloc,nloc,user->ni,user->n));
  CHKERRQ(MatSetFromOptions(user->Ai));
  CHKERRQ(MatSetUp(user->Ai));

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&user->H));
  CHKERRQ(MatSetSizes(user->H,nloc,nloc,user->n,user->n));
  CHKERRQ(MatSetFromOptions(user->H));
  CHKERRQ(MatSetUp(user->H));
  PetscFunctionReturn(0);
}

PetscErrorCode DestroyProblem(AppCtx *user)
{
  PetscFunctionBegin;
  if (!user->noeqflag) {
    CHKERRQ(MatDestroy(&user->Ae));
  }
  CHKERRQ(MatDestroy(&user->Ai));
  CHKERRQ(MatDestroy(&user->H));

  CHKERRQ(VecDestroy(&user->x));
  if (!user->noeqflag) {
    CHKERRQ(VecDestroy(&user->ce));
  }
  CHKERRQ(VecDestroy(&user->ci));
  CHKERRQ(VecDestroy(&user->xl));
  CHKERRQ(VecDestroy(&user->xu));
  CHKERRQ(VecDestroy(&user->Xseq));
  CHKERRQ(VecScatterDestroy(&user->scat));
  PetscFunctionReturn(0);
}

/* Evaluate
   f(x) = (x0 - 2)^2 + (x1 - 2)^2 - 2*(x0 + x1)
   G = grad f = [2*(x0 - 2) - 2;
                 2*(x1 - 2) - 2]
*/
PetscErrorCode FormFunctionGradient(Tao tao, Vec X, PetscReal *f, Vec G, void *ctx)
{
  PetscScalar       g;
  const PetscScalar *x;
  MPI_Comm          comm;
  PetscMPIInt       rank;
  PetscReal         fin;
  AppCtx            *user=(AppCtx*)ctx;
  Vec               Xseq=user->Xseq;
  VecScatter        scat=user->scat;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)tao,&comm));
  CHKERRMPI(MPI_Comm_rank(comm,&rank));

  CHKERRQ(VecScatterBegin(scat,X,Xseq,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(scat,X,Xseq,INSERT_VALUES,SCATTER_FORWARD));

  fin = 0.0;
  if (rank == 0) {
    CHKERRQ(VecGetArrayRead(Xseq,&x));
    fin = (x[0]-2.0)*(x[0]-2.0) + (x[1]-2.0)*(x[1]-2.0) - 2.0*(x[0]+x[1]);
    g = 2.0*(x[0]-2.0) - 2.0;
    CHKERRQ(VecSetValue(G,0,g,INSERT_VALUES));
    g = 2.0*(x[1]-2.0) - 2.0;
    CHKERRQ(VecSetValue(G,1,g,INSERT_VALUES));
    CHKERRQ(VecRestoreArrayRead(Xseq,&x));
  }
  CHKERRMPI(MPI_Allreduce(&fin,f,1,MPIU_REAL,MPIU_SUM,comm));
  CHKERRQ(VecAssemblyBegin(G));
  CHKERRQ(VecAssemblyEnd(G));
  PetscFunctionReturn(0);
}

/* Evaluate
   H = fxx + grad (grad g^T*DI) - grad (grad h^T*DE)]
     = [ 2*(1+de[0]-di[0]+di[1]), 0;
                   0,             2]
*/
PetscErrorCode FormHessian(Tao tao, Vec x,Mat H, Mat Hpre, void *ctx)
{
  AppCtx            *user=(AppCtx*)ctx;
  Vec               DE,DI;
  const PetscScalar *de,*di;
  PetscInt          zero=0,one=1;
  PetscScalar       two=2.0;
  PetscScalar       val=0.0;
  Vec               Deseq,Diseq;
  VecScatter        Descat,Discat;
  PetscMPIInt       rank;
  MPI_Comm          comm;

  PetscFunctionBegin;
  CHKERRQ(TaoGetDualVariables(tao,&DE,&DI));

  CHKERRQ(PetscObjectGetComm((PetscObject)tao,&comm));
  CHKERRMPI(MPI_Comm_rank(comm,&rank));

  if (!user->noeqflag) {
   CHKERRQ(VecScatterCreateToZero(DE,&Descat,&Deseq));
   CHKERRQ(VecScatterBegin(Descat,DE,Deseq,INSERT_VALUES,SCATTER_FORWARD));
   CHKERRQ(VecScatterEnd(Descat,DE,Deseq,INSERT_VALUES,SCATTER_FORWARD));
  }
  CHKERRQ(VecScatterCreateToZero(DI,&Discat,&Diseq));
  CHKERRQ(VecScatterBegin(Discat,DI,Diseq,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(Discat,DI,Diseq,INSERT_VALUES,SCATTER_FORWARD));

  if (rank == 0) {
    if (!user->noeqflag) {
      CHKERRQ(VecGetArrayRead(Deseq,&de));  /* places equality constraint dual into array */
    }
    CHKERRQ(VecGetArrayRead(Diseq,&di));  /* places inequality constraint dual into array */

    if (!user->noeqflag) {
      val = 2.0 * (1 + de[0] - di[0] + di[1]);
      CHKERRQ(VecRestoreArrayRead(Deseq,&de));
      CHKERRQ(VecRestoreArrayRead(Diseq,&di));
    } else {
      val = 2.0 * (1 - di[0] + di[1]);
    }
    CHKERRQ(VecRestoreArrayRead(Diseq,&di));
    CHKERRQ(MatSetValues(H,1,&zero,1,&zero,&val,INSERT_VALUES));
    CHKERRQ(MatSetValues(H,1,&one,1,&one,&two,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(H,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(H,MAT_FINAL_ASSEMBLY));
  if (!user->noeqflag) {
    CHKERRQ(VecScatterDestroy(&Descat));
    CHKERRQ(VecDestroy(&Deseq));
  }
  CHKERRQ(VecScatterDestroy(&Discat));
  CHKERRQ(VecDestroy(&Diseq));
  PetscFunctionReturn(0);
}

/* Evaluate
   h = [ x0^2 - x1;
         1 -(x0^2 - x1)]
*/
PetscErrorCode FormInequalityConstraints(Tao tao,Vec X,Vec CI,void *ctx)
{
  const PetscScalar *x;
  PetscScalar       ci;
  MPI_Comm          comm;
  PetscMPIInt       rank;
  AppCtx            *user=(AppCtx*)ctx;
  Vec               Xseq=user->Xseq;
  VecScatter        scat=user->scat;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)tao,&comm));
  CHKERRMPI(MPI_Comm_rank(comm,&rank));

  CHKERRQ(VecScatterBegin(scat,X,Xseq,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(scat,X,Xseq,INSERT_VALUES,SCATTER_FORWARD));

  if (rank == 0) {
    CHKERRQ(VecGetArrayRead(Xseq,&x));
    ci = x[0]*x[0] - x[1];
    CHKERRQ(VecSetValue(CI,0,ci,INSERT_VALUES));
    ci = -x[0]*x[0] + x[1] + 1.0;
    CHKERRQ(VecSetValue(CI,1,ci,INSERT_VALUES));
    CHKERRQ(VecRestoreArrayRead(Xseq,&x));
  }
  CHKERRQ(VecAssemblyBegin(CI));
  CHKERRQ(VecAssemblyEnd(CI));
  PetscFunctionReturn(0);
}

/* Evaluate
   g = [ x0^2 + x1 - 2]
*/
PetscErrorCode FormEqualityConstraints(Tao tao,Vec X,Vec CE,void *ctx)
{
  const PetscScalar *x;
  PetscScalar       ce;
  MPI_Comm          comm;
  PetscMPIInt       rank;
  AppCtx            *user=(AppCtx*)ctx;
  Vec               Xseq=user->Xseq;
  VecScatter        scat=user->scat;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)tao,&comm));
  CHKERRMPI(MPI_Comm_rank(comm,&rank));

  CHKERRQ(VecScatterBegin(scat,X,Xseq,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(scat,X,Xseq,INSERT_VALUES,SCATTER_FORWARD));

  if (rank == 0) {
    CHKERRQ(VecGetArrayRead(Xseq,&x));
    ce = x[0]*x[0] + x[1] - 2.0;
    CHKERRQ(VecSetValue(CE,0,ce,INSERT_VALUES));
    CHKERRQ(VecRestoreArrayRead(Xseq,&x));
  }
  CHKERRQ(VecAssemblyBegin(CE));
  CHKERRQ(VecAssemblyEnd(CE));
  PetscFunctionReturn(0);
}

/*
  grad h = [  2*x0, -1;
             -2*x0,  1]
*/
PetscErrorCode FormInequalityJacobian(Tao tao, Vec X, Mat JI, Mat JIpre,  void *ctx)
{
  AppCtx            *user=(AppCtx*)ctx;
  PetscInt          zero=0,one=1,cols[2];
  PetscScalar       vals[2];
  const PetscScalar *x;
  Vec               Xseq=user->Xseq;
  VecScatter        scat=user->scat;
  MPI_Comm          comm;
  PetscMPIInt       rank;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)tao,&comm));
  CHKERRMPI(MPI_Comm_rank(comm,&rank));
  CHKERRQ(VecScatterBegin(scat,X,Xseq,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(scat,X,Xseq,INSERT_VALUES,SCATTER_FORWARD));

  CHKERRQ(VecGetArrayRead(Xseq,&x));
  if (!rank) {
    cols[0] = 0; cols[1] = 1;
    vals[0] = 2*x[0]; vals[1] = -1.0;
    CHKERRQ(MatSetValues(JI,1,&zero,2,cols,vals,INSERT_VALUES));
    vals[0] = -2*x[0]; vals[1] = 1.0;
    CHKERRQ(MatSetValues(JI,1,&one,2,cols,vals,INSERT_VALUES));
  }
  CHKERRQ(VecRestoreArrayRead(Xseq,&x));
  CHKERRQ(MatAssemblyBegin(JI,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(JI,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

/*
  grad g = [2*x0
             1.0 ]
*/
PetscErrorCode FormEqualityJacobian(Tao tao,Vec X,Mat JE,Mat JEpre,void *ctx)
{
  PetscInt          zero=0,cols[2];
  PetscScalar       vals[2];
  const PetscScalar *x;
  PetscMPIInt       rank;
  MPI_Comm          comm;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)tao,&comm));
  CHKERRMPI(MPI_Comm_rank(comm,&rank));

  if (rank == 0) {
    CHKERRQ(VecGetArrayRead(X,&x));
    cols[0] = 0;       cols[1] = 1;
    vals[0] = 2*x[0];  vals[1] = 1.0;
    CHKERRQ(MatSetValues(JE,1,&zero,2,cols,vals,INSERT_VALUES));
    CHKERRQ(VecRestoreArrayRead(X,&x));
  }
  CHKERRQ(MatAssemblyBegin(JE,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(JE,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

/*TEST

   build:
      requires: !complex !defined(PETSC_USE_CXX)

   test:
      args: -tao_converged_reason -tao_pdipm_kkt_shift_pd
      requires: mumps

   test:
      suffix: 2
      nsize: 2
      args: -tao_converged_reason -tao_pdipm_kkt_shift_pd
      requires: mumps

   test:
      suffix: 3
      args: -tao_converged_reason -no_eq
      requires: mumps

   test:
      suffix: 4
      nsize: 2
      args: -tao_converged_reason -no_eq
      requires: mumps

   test:
      suffix: 5
      args: -tao_cmonitor -tao_type almm
      requires: mumps

   test:
      suffix: 6
      args: -tao_cmonitor -tao_type almm -tao_almm_type phr
      requires: mumps

   test:
      suffix: 7
      nsize: 2
      requires: mumps
      args: -tao_cmonitor -tao_type almm

   test:
      suffix: 8
      nsize: 2
      requires: cuda mumps
      args: -tao_cmonitor -tao_type almm -vec_type cuda -mat_type aijcusparse

   test:
      suffix: 9
      nsize: 2
      args: -tao_cmonitor -tao_type almm -no_eq
      requires: mumps

   test:
      suffix: 10
      nsize: 2
      args: -tao_cmonitor -tao_type almm -tao_almm_type phr -no_eq
      requires: mumps

TEST*/
