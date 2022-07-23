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

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheck(size <= 2,PETSC_COMM_SELF,PETSC_ERR_WRONG_MPI_SIZE,"More than 2 processors detected. Example written to use max of 2 processors.");

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"---- Constrained Problem -----\n"));
  PetscCall(InitializeProblem(&user)); /* sets up problem, function below */
  PetscCall(TaoCreate(PETSC_COMM_WORLD,&tao));
  PetscCall(TaoSetType(tao,TAOPDIPM));
  PetscCall(TaoSetSolution(tao,user.x)); /* gets solution vector from problem */
  PetscCall(TaoSetVariableBounds(tao,user.xl,user.xu)); /* sets lower upper bounds from given solution */
  PetscCall(TaoSetObjectiveAndGradient(tao,NULL,FormFunctionGradient,(void*)&user));

  if (!user.noeqflag) {
    PetscCall(TaoSetEqualityConstraintsRoutine(tao,user.ce,FormEqualityConstraints,(void*)&user));
  }
  PetscCall(TaoSetInequalityConstraintsRoutine(tao,user.ci,FormInequalityConstraints,(void*)&user));
  if (!user.noeqflag) {
    PetscCall(TaoSetJacobianEqualityRoutine(tao,user.Ae,user.Ae,FormEqualityJacobian,(void*)&user)); /* equality jacobian */
  }
  PetscCall(TaoSetJacobianInequalityRoutine(tao,user.Ai,user.Ai,FormInequalityJacobian,(void*)&user)); /* inequality jacobian */
  PetscCall(TaoSetTolerances(tao,1.e-6,1.e-6,1.e-6));
  PetscCall(TaoSetConstraintTolerances(tao,1.e-6,1.e-6));

  PetscCall(TaoGetKSP(tao,&ksp));
  PetscCall(KSPGetPC(ksp,&pc));
  PetscCall(PCSetType(pc,PCCHOLESKY));
  /*
      This algorithm produces matrices with zeros along the diagonal therefore we use
    MUMPS which provides solver for indefinite matrices
  */
#if defined(PETSC_HAVE_MUMPS)
  PetscCall(PCFactorSetMatSolverType(pc,MATSOLVERMUMPS));  /* requires mumps to solve pdipm */
#else
  PetscCheck(size == 1,PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"Requires an external package that supports parallel PCCHOLESKY, e.g., MUMPS.");
#endif
  PetscCall(KSPSetType(ksp,KSPPREONLY));
  PetscCall(KSPSetFromOptions(ksp));

  PetscCall(TaoSetFromOptions(tao));
  PetscCall(TaoGetType(tao,&type));
  PetscCall(PetscObjectTypeCompare((PetscObject)tao,TAOPDIPM,&pdipm));
  if (pdipm) {
    PetscCall(TaoSetHessian(tao,user.H,user.H,FormHessian,(void*)&user));
    if (user.initview) {
      PetscCall(TaoSetUp(tao));
      PetscCall(VecDuplicate(user.x, &G));
      PetscCall(FormFunctionGradient(tao, user.x, &f, G, (void*)&user));
      PetscCall(FormHessian(tao, user.x, user.H, user.H, (void*)&user));
      PetscCall(PetscViewerASCIIPushTab(PETSC_VIEWER_STDOUT_WORLD));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\nInitial point X:\n"));
      PetscCall(VecView(user.x, PETSC_VIEWER_STDOUT_WORLD));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\nInitial objective f(x) = %g\n",(double)f));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\nInitial gradient and Hessian:\n"));
      PetscCall(VecView(G, PETSC_VIEWER_STDOUT_WORLD));
      PetscCall(MatView(user.H, PETSC_VIEWER_STDOUT_WORLD));
      PetscCall(VecDestroy(&G));
      PetscCall(FormInequalityJacobian(tao, user.x, user.Ai, user.Ai, (void*)&user));
      PetscCall(MatCreateVecs(user.Ai, NULL, &CI));
      PetscCall(FormInequalityConstraints(tao, user.x, CI, (void*)&user));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\nInitial inequality constraints and Jacobian:\n"));
      PetscCall(VecView(CI, PETSC_VIEWER_STDOUT_WORLD));
      PetscCall(MatView(user.Ai, PETSC_VIEWER_STDOUT_WORLD));
      PetscCall(VecDestroy(&CI));
      if (!user.noeqflag) {
        PetscCall(FormEqualityJacobian(tao, user.x, user.Ae, user.Ae, (void*)&user));
        PetscCall(MatCreateVecs(user.Ae, NULL, &CE));
        PetscCall(FormEqualityConstraints(tao, user.x, CE, (void*)&user));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\nInitial equality constraints and Jacobian:\n"));
        PetscCall(VecView(CE, PETSC_VIEWER_STDOUT_WORLD));
        PetscCall(MatView(user.Ae, PETSC_VIEWER_STDOUT_WORLD));
        PetscCall(VecDestroy(&CE));
      }
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n"));
      PetscCall(PetscViewerASCIIPopTab(PETSC_VIEWER_STDOUT_WORLD));
    }
  }

  PetscCall(TaoSolve(tao));
  PetscCall(TaoGetSolution(tao,&x));
  PetscCall(VecView(x,PETSC_VIEWER_STDOUT_WORLD));

  /* Free objects */
  PetscCall(DestroyProblem(&user));
  PetscCall(TaoDestroy(&tao));
  PetscCall(PetscFinalize());
  return 0;
}

PetscErrorCode InitializeProblem(AppCtx *user)
{
  PetscMPIInt    size;
  PetscMPIInt    rank;
  PetscInt       nloc,neloc,niloc;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  user->noeqflag = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-no_eq",&user->noeqflag,NULL));
  user->initview = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-init_view",&user->initview,NULL));

  if (!user->noeqflag) {
    /* Tell user the correct solution, not an error checking */
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Solution should be f(1,1)=-2\n"));
  }

  /* create vector x and set initial values */
  user->n = 2; /* global length */
  nloc = (size==1)?user->n:1;
  PetscCall(VecCreate(PETSC_COMM_WORLD,&user->x));
  PetscCall(VecSetSizes(user->x,nloc,user->n));
  PetscCall(VecSetFromOptions(user->x));
  PetscCall(VecSet(user->x,0));

  /* create and set lower and upper bound vectors */
  PetscCall(VecDuplicate(user->x,&user->xl));
  PetscCall(VecDuplicate(user->x,&user->xu));
  PetscCall(VecSet(user->xl,-1.0));
  PetscCall(VecSet(user->xu,2.0));

  /* create scater to zero */
  PetscCall(VecScatterCreateToZero(user->x,&user->scat,&user->Xseq));

  user->ne = 1;
  user->ni = 2;
  neloc = (rank==0)?user->ne:0;
  niloc = (size==1)?user->ni:1;

  if (!user->noeqflag) {
    PetscCall(VecCreate(PETSC_COMM_WORLD,&user->ce)); /* a 1x1 vec for equality constraints */
    PetscCall(VecSetSizes(user->ce,neloc,user->ne));
    PetscCall(VecSetFromOptions(user->ce));
    PetscCall(VecSetUp(user->ce));
  }

  PetscCall(VecCreate(PETSC_COMM_WORLD,&user->ci)); /* a 2x1 vec for inequality constraints */
  PetscCall(VecSetSizes(user->ci,niloc,user->ni));
  PetscCall(VecSetFromOptions(user->ci));
  PetscCall(VecSetUp(user->ci));

  /* nexn & nixn matricies for equally and inequalty constraints */
  if (!user->noeqflag) {
    PetscCall(MatCreate(PETSC_COMM_WORLD,&user->Ae));
    PetscCall(MatSetSizes(user->Ae,neloc,nloc,user->ne,user->n));
    PetscCall(MatSetFromOptions(user->Ae));
    PetscCall(MatSetUp(user->Ae));
  }

  PetscCall(MatCreate(PETSC_COMM_WORLD,&user->Ai));
  PetscCall(MatSetSizes(user->Ai,niloc,nloc,user->ni,user->n));
  PetscCall(MatSetFromOptions(user->Ai));
  PetscCall(MatSetUp(user->Ai));

  PetscCall(MatCreate(PETSC_COMM_WORLD,&user->H));
  PetscCall(MatSetSizes(user->H,nloc,nloc,user->n,user->n));
  PetscCall(MatSetFromOptions(user->H));
  PetscCall(MatSetUp(user->H));
  PetscFunctionReturn(0);
}

PetscErrorCode DestroyProblem(AppCtx *user)
{
  PetscFunctionBegin;
  if (!user->noeqflag) {
    PetscCall(MatDestroy(&user->Ae));
  }
  PetscCall(MatDestroy(&user->Ai));
  PetscCall(MatDestroy(&user->H));

  PetscCall(VecDestroy(&user->x));
  if (!user->noeqflag) {
    PetscCall(VecDestroy(&user->ce));
  }
  PetscCall(VecDestroy(&user->ci));
  PetscCall(VecDestroy(&user->xl));
  PetscCall(VecDestroy(&user->xu));
  PetscCall(VecDestroy(&user->Xseq));
  PetscCall(VecScatterDestroy(&user->scat));
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
  PetscCall(PetscObjectGetComm((PetscObject)tao,&comm));
  PetscCallMPI(MPI_Comm_rank(comm,&rank));

  PetscCall(VecScatterBegin(scat,X,Xseq,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd(scat,X,Xseq,INSERT_VALUES,SCATTER_FORWARD));

  fin = 0.0;
  if (rank == 0) {
    PetscCall(VecGetArrayRead(Xseq,&x));
    fin = (x[0]-2.0)*(x[0]-2.0) + (x[1]-2.0)*(x[1]-2.0) - 2.0*(x[0]+x[1]);
    g = 2.0*(x[0]-2.0) - 2.0;
    PetscCall(VecSetValue(G,0,g,INSERT_VALUES));
    g = 2.0*(x[1]-2.0) - 2.0;
    PetscCall(VecSetValue(G,1,g,INSERT_VALUES));
    PetscCall(VecRestoreArrayRead(Xseq,&x));
  }
  PetscCallMPI(MPI_Allreduce(&fin,f,1,MPIU_REAL,MPIU_SUM,comm));
  PetscCall(VecAssemblyBegin(G));
  PetscCall(VecAssemblyEnd(G));
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
  PetscCall(TaoGetDualVariables(tao,&DE,&DI));

  PetscCall(PetscObjectGetComm((PetscObject)tao,&comm));
  PetscCallMPI(MPI_Comm_rank(comm,&rank));

  if (!user->noeqflag) {
   PetscCall(VecScatterCreateToZero(DE,&Descat,&Deseq));
   PetscCall(VecScatterBegin(Descat,DE,Deseq,INSERT_VALUES,SCATTER_FORWARD));
   PetscCall(VecScatterEnd(Descat,DE,Deseq,INSERT_VALUES,SCATTER_FORWARD));
  }
  PetscCall(VecScatterCreateToZero(DI,&Discat,&Diseq));
  PetscCall(VecScatterBegin(Discat,DI,Diseq,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd(Discat,DI,Diseq,INSERT_VALUES,SCATTER_FORWARD));

  if (rank == 0) {
    if (!user->noeqflag) {
      PetscCall(VecGetArrayRead(Deseq,&de));  /* places equality constraint dual into array */
    }
    PetscCall(VecGetArrayRead(Diseq,&di));  /* places inequality constraint dual into array */

    if (!user->noeqflag) {
      val = 2.0 * (1 + de[0] - di[0] + di[1]);
      PetscCall(VecRestoreArrayRead(Deseq,&de));
      PetscCall(VecRestoreArrayRead(Diseq,&di));
    } else {
      val = 2.0 * (1 - di[0] + di[1]);
    }
    PetscCall(VecRestoreArrayRead(Diseq,&di));
    PetscCall(MatSetValues(H,1,&zero,1,&zero,&val,INSERT_VALUES));
    PetscCall(MatSetValues(H,1,&one,1,&one,&two,INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(H,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(H,MAT_FINAL_ASSEMBLY));
  if (!user->noeqflag) {
    PetscCall(VecScatterDestroy(&Descat));
    PetscCall(VecDestroy(&Deseq));
  }
  PetscCall(VecScatterDestroy(&Discat));
  PetscCall(VecDestroy(&Diseq));
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
  PetscCall(PetscObjectGetComm((PetscObject)tao,&comm));
  PetscCallMPI(MPI_Comm_rank(comm,&rank));

  PetscCall(VecScatterBegin(scat,X,Xseq,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd(scat,X,Xseq,INSERT_VALUES,SCATTER_FORWARD));

  if (rank == 0) {
    PetscCall(VecGetArrayRead(Xseq,&x));
    ci = x[0]*x[0] - x[1];
    PetscCall(VecSetValue(CI,0,ci,INSERT_VALUES));
    ci = -x[0]*x[0] + x[1] + 1.0;
    PetscCall(VecSetValue(CI,1,ci,INSERT_VALUES));
    PetscCall(VecRestoreArrayRead(Xseq,&x));
  }
  PetscCall(VecAssemblyBegin(CI));
  PetscCall(VecAssemblyEnd(CI));
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
  PetscCall(PetscObjectGetComm((PetscObject)tao,&comm));
  PetscCallMPI(MPI_Comm_rank(comm,&rank));

  PetscCall(VecScatterBegin(scat,X,Xseq,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd(scat,X,Xseq,INSERT_VALUES,SCATTER_FORWARD));

  if (rank == 0) {
    PetscCall(VecGetArrayRead(Xseq,&x));
    ce = x[0]*x[0] + x[1] - 2.0;
    PetscCall(VecSetValue(CE,0,ce,INSERT_VALUES));
    PetscCall(VecRestoreArrayRead(Xseq,&x));
  }
  PetscCall(VecAssemblyBegin(CE));
  PetscCall(VecAssemblyEnd(CE));
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
  PetscCall(PetscObjectGetComm((PetscObject)tao,&comm));
  PetscCallMPI(MPI_Comm_rank(comm,&rank));
  PetscCall(VecScatterBegin(scat,X,Xseq,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd(scat,X,Xseq,INSERT_VALUES,SCATTER_FORWARD));

  PetscCall(VecGetArrayRead(Xseq,&x));
  if (rank == 0) {
    cols[0] = 0; cols[1] = 1;
    vals[0] = 2*x[0]; vals[1] = -1.0;
    PetscCall(MatSetValues(JI,1,&zero,2,cols,vals,INSERT_VALUES));
    vals[0] = -2*x[0]; vals[1] = 1.0;
    PetscCall(MatSetValues(JI,1,&one,2,cols,vals,INSERT_VALUES));
  }
  PetscCall(VecRestoreArrayRead(Xseq,&x));
  PetscCall(MatAssemblyBegin(JI,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(JI,MAT_FINAL_ASSEMBLY));
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
  PetscCall(PetscObjectGetComm((PetscObject)tao,&comm));
  PetscCallMPI(MPI_Comm_rank(comm,&rank));

  if (rank == 0) {
    PetscCall(VecGetArrayRead(X,&x));
    cols[0] = 0;       cols[1] = 1;
    vals[0] = 2*x[0];  vals[1] = 1.0;
    PetscCall(MatSetValues(JE,1,&zero,2,cols,vals,INSERT_VALUES));
    PetscCall(VecRestoreArrayRead(X,&x));
  }
  PetscCall(MatAssemblyBegin(JE,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(JE,MAT_FINAL_ASSEMBLY));
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
