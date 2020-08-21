/* Program usage: mpiexec -n 2 ./ex1 [-help] [all TAO options] */

/* ----------------------------------------------------------------------
min f = (x0 - 2)^2 + (x1 - 2)^2 - 2*(x0 + x1)
s.t.  x0^2 + x1 - 2 = 0
      0  <= x0^2 - x1 <= 1
      -1 <= x0, x1 <= 2
---------------------------------------------------------------------- */

#include <petsctao.h>

static  char help[]= "Solves constrained optimiztion problem using pdipm.\n\
Input parameters include:\n\
  -tao_type pdipm    : sets Tao solver\n\
  -no_eq             : removes the equaility constraints from the problem\n\
  -snes_fd           : snes with finite difference Jacobian (needed for pdipm)\n\
  -tao_cmonitor      : convergence monitor with constraint norm \n\
  -tao_view_solution : view exact solution at each itteration\n\
  Note: external package superlu_dist is requried to run either for ipm or pdipm. Additionally This is designed for a maximum of 2 processors, the code will error if size > 2.\n\n";

/*
   User-defined application context - contains data needed by the
   application-provided call-back routines, FormFunction(),
   FormGradient(), and FormHessian().
*/

/*
   x,d in R^n
   f in R
   bin in R^mi
   beq in R^me
   Aeq in R^(me x n)
   Ain in R^(mi x n)
   H in R^(n x n)
   min f=(1/2)*x'*H*x + d'*x
   s.t.  Aeq*x == beq
         Ain*x >= bin
*/
typedef struct {
  PetscInt n;  /* Global length of x */
  PetscInt ne; /* Global number of equality constraints */
  PetscInt ni; /* Global number of inequality constraints */
  PetscBool noeqflag;
  Vec      x,xl,xu;
  Vec      ce,ci,bl,bu,Xseq;
  Mat      Ae,Ai,H;
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
  PetscErrorCode ierr;  /* used to check for functions returning nonzeros */
  Tao            tao;
  KSP            ksp;
  PC             pc;
  AppCtx         user;  /* application context */
  Vec            x;
  PetscMPIInt    rank;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  if (rank>1){
    SETERRQ(PETSC_COMM_SELF,1,"More than 2 processors detected. Example written to use max of 2 processors.\n");
  }

  ierr = PetscPrintf(PETSC_COMM_WORLD,"---- Constrained Problem -----\n");CHKERRQ(ierr);
  ierr = InitializeProblem(&user);CHKERRQ(ierr); /* sets up problem, function below */
  ierr = TaoCreate(PETSC_COMM_WORLD,&tao);CHKERRQ(ierr);
  ierr = TaoSetType(tao,TAOPDIPM);CHKERRQ(ierr);
  ierr = TaoSetInitialVector(tao,user.x);CHKERRQ(ierr); /* gets solution vector from problem */
  ierr = TaoSetVariableBounds(tao,user.xl,user.xu);CHKERRQ(ierr); /* sets lower upper bounds from given solution */
  ierr = TaoSetObjectiveAndGradientRoutine(tao,FormFunctionGradient,(void*)&user);CHKERRQ(ierr);

  if (!user.noeqflag){
    ierr = TaoSetEqualityConstraintsRoutine(tao,user.ce,FormEqualityConstraints,(void*)&user);CHKERRQ(ierr);
  }
  ierr = TaoSetInequalityConstraintsRoutine(tao,user.ci,FormInequalityConstraints,(void*)&user);CHKERRQ(ierr);

  if (!user.noeqflag){
    ierr = TaoSetJacobianEqualityRoutine(tao,user.Ae,user.Ae,FormEqualityJacobian,(void*)&user);CHKERRQ(ierr); /* equality jacobian */
  }
  ierr = TaoSetJacobianInequalityRoutine(tao,user.Ai,user.Ai,FormInequalityJacobian,(void*)&user);CHKERRQ(ierr); /* inequality jacobian */
  ierr = TaoSetHessianRoutine(tao,user.H,user.H,FormHessian,(void*)&user);CHKERRQ(ierr);
  ierr = TaoSetTolerances(tao,1.e-6,1.e-6,1.e-6);CHKERRQ(ierr);
  ierr = TaoSetConstraintTolerances(tao,1.e-6,1.e-6);CHKERRQ(ierr);

  ierr = TaoGetKSP(tao,&ksp);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCLU);CHKERRQ(ierr);
  /*
      This algorithm produces matrices with zeros along the diagonal therefore we use
    SuperLU_DIST which provides shift to the diagonal
  */
  ierr = PCFactorSetMatSolverType(pc,MATSOLVERSUPERLU_DIST);CHKERRQ(ierr);  /* requires superlu_dist to solve pdipm */
  ierr = KSPSetType(ksp,KSPPREONLY);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

  ierr = TaoSetFromOptions(tao);CHKERRQ(ierr);

  ierr = TaoSolve(tao);CHKERRQ(ierr);
  ierr = TaoGetSolutionVector(tao,&x);CHKERRQ(ierr);
  ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* Free objects */
  ierr = DestroyProblem(&user);CHKERRQ(ierr);
  ierr = TaoDestroy(&tao);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

PetscErrorCode InitializeProblem(AppCtx *user)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;
  PetscMPIInt    rank;
  PetscInt       nloc,neloc,niloc;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  user->noeqflag = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-no_eq",&user->noeqflag,NULL);CHKERRQ(ierr);
  if (!user->noeqflag) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Solution should be f(1,1)=-2\n");CHKERRQ(ierr);
  }

  /* create vector x and set initial values */
  user->n = 2; /* global length */
  nloc = (rank==0)?user->n:0;
  ierr = VecCreateMPI(PETSC_COMM_WORLD,nloc,user->n,&user->x);CHKERRQ(ierr);
  ierr = VecSet(user->x,0);CHKERRQ(ierr);

  /* create and set lower and upper bound vectors */
  ierr = VecDuplicate(user->x,&user->xl);CHKERRQ(ierr);
  ierr = VecDuplicate(user->x,&user->xu);CHKERRQ(ierr);
  ierr = VecSet(user->xl,-1.0);CHKERRQ(ierr);
  ierr = VecSet(user->xu,2.0);CHKERRQ(ierr);

  /* create scater to zero */
  ierr = VecScatterCreateToZero(user->x,&user->scat,&user->Xseq);CHKERRQ(ierr);

    user->ne = 1;
    neloc = (rank==0)?user->ne:0;

  if (!user->noeqflag){
    ierr = VecCreateMPI(PETSC_COMM_WORLD,neloc,user->ne,&user->ce);CHKERRQ(ierr); /* a 1x1 vec for equality constraints */
  }
  user->ni = 2;
  niloc = (rank==0)?user->ni:0;
  ierr = VecCreateMPI(PETSC_COMM_WORLD,niloc,user->ni,&user->ci);CHKERRQ(ierr); /* a 2x1 vec for inequality constraints */

  /* nexn & nixn matricies for equaly and inequalty constriants */
  if (!user->noeqflag){
    ierr = MatCreate(PETSC_COMM_WORLD,&user->Ae);CHKERRQ(ierr);
    ierr = MatSetSizes(user->Ae,neloc,nloc,user->ne,user->n);CHKERRQ(ierr);
    ierr = MatSetUp(user->Ae);CHKERRQ(ierr);
    ierr = MatSetFromOptions(user->Ae);CHKERRQ(ierr);
  }

  ierr = MatCreate(PETSC_COMM_WORLD,&user->Ai);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&user->H);CHKERRQ(ierr);
 
  ierr = MatSetSizes(user->Ai,niloc,nloc,user->ni,user->n);CHKERRQ(ierr);
  ierr = MatSetSizes(user->H,nloc,nloc,user->n,user->n);CHKERRQ(ierr);

  ierr = MatSetUp(user->Ai);CHKERRQ(ierr);
  ierr = MatSetUp(user->H);CHKERRQ(ierr);

  ierr = MatSetFromOptions(user->Ai);CHKERRQ(ierr);
  ierr = MatSetFromOptions(user->H);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DestroyProblem(AppCtx *user)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!user->noeqflag){
   ierr = MatDestroy(&user->Ae);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&user->Ai);CHKERRQ(ierr);
  ierr = MatDestroy(&user->H);CHKERRQ(ierr);

  ierr = VecDestroy(&user->x);CHKERRQ(ierr);
  if (!user->noeqflag){
    ierr = VecDestroy(&user->ce);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&user->ci);CHKERRQ(ierr);
  ierr = VecDestroy(&user->xl);CHKERRQ(ierr);
  ierr = VecDestroy(&user->xu);CHKERRQ(ierr);
  ierr = VecDestroy(&user->Xseq);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&user->scat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  f(X) = (x0 - 2)^2 + (x1 - 2)^2 - 2*(x0 + x1)
  G(X) = fx = [2.0*(x[0]-2.0) - 2.0; 2.0*(x[1]-2.0) - 2.0]
*/
PetscErrorCode FormFunctionGradient(Tao tao, Vec X, PetscReal *f, Vec G, void *ctx)
{
  PetscScalar       g;
  const PetscScalar *x;
  MPI_Comm          comm;
  PetscMPIInt       rank;
  PetscErrorCode    ierr;
  PetscReal         fin;
  AppCtx            *user=(AppCtx*)ctx;
  Vec               Xseq=user->Xseq;
  VecScatter        scat=user->scat;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)tao,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  ierr = VecScatterBegin(scat,X,Xseq,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(scat,X,Xseq,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

  fin = 0.0;
  if (!rank) {
    ierr = VecGetArrayRead(Xseq,&x);CHKERRQ(ierr);
    fin = (x[0]-2.0)*(x[0]-2.0) + (x[1]-2.0)*(x[1]-2.0) - 2.0*(x[0]+x[1]);
    g = 2.0*(x[0]-2.0) - 2.0;
    ierr = VecSetValue(G,0,g,INSERT_VALUES);CHKERRQ(ierr);
    g = 2.0*(x[1]-2.0) - 2.0;
    ierr = VecSetValue(G,1,g,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(Xseq,&x);CHKERRQ(ierr);
  }
  ierr = MPI_Allreduce(&fin,f,1,MPIU_REAL,MPIU_SUM,comm);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(G);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(G);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode FormHessian(Tao tao, Vec x,Mat H, Mat Hpre, void *ctx)
{
  AppCtx            *user=(AppCtx*)ctx;
  Vec               DE,DI;
  const PetscScalar *de, *di;
  PetscInt          zero=0,one=1;
  PetscScalar       two=2.0;
  PetscScalar       val=0.0;
  Vec               Deseq,Diseq=user->Xseq;
  VecScatter        Descat,Discat=user->scat;
  PetscMPIInt       rank;
  MPI_Comm          comm;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = TaoGetDualVariables(tao,&DE,&DI);CHKERRQ(ierr);

  ierr = PetscObjectGetComm((PetscObject)tao,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  if (!user->noeqflag){
   ierr = VecScatterCreateToZero(DE,&Descat,&Deseq);CHKERRQ(ierr);
   ierr = VecScatterBegin(Descat,DE,Deseq,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
   ierr = VecScatterEnd(Descat,DE,Deseq,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  }

  ierr = VecScatterBegin(Discat,DI,Diseq,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(Discat,DI,Diseq,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  

  if (!rank){
    if (!user->noeqflag){
      ierr = VecGetArrayRead(Deseq,&de);CHKERRQ(ierr);  /* places equality constraint dual into array */
    }

    ierr = VecGetArrayRead(Diseq,&di);CHKERRQ(ierr);  /* places inequality constraint dual into array */
    
    if (!user->noeqflag){
      val = 2.0 * (1 + de[0] + di[0] - di[1]);
      ierr = VecRestoreArrayRead(Deseq,&de);CHKERRQ(ierr);
    }else{
      val = 2.0 * (1 + di[0] - di[1]);
    }
    ierr = VecRestoreArrayRead(Diseq,&di);CHKERRQ(ierr);
    ierr = MatSetValues(H,1,&zero,1,&zero,&val,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatSetValues(H,1,&one,1,&one,&two,INSERT_VALUES);CHKERRQ(ierr);
  }

  ierr = MatAssemblyBegin(H,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(H,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (!user->noeqflag){
    ierr = VecScatterDestroy(&Descat);CHKERRQ(ierr);
    ierr = VecDestroy(&Deseq);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode FormInequalityConstraints(Tao tao,Vec X,Vec CI,void *ctx)
{
  const PetscScalar *x;
  PetscScalar       ci;
  PetscErrorCode    ierr;
  MPI_Comm          comm;
  PetscMPIInt       rank;
  AppCtx            *user=(AppCtx*)ctx;
  Vec               Xseq=user->Xseq;
  VecScatter        scat=user->scat;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)tao,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  ierr = VecScatterBegin(scat,X,Xseq,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(scat,X,Xseq,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

  if (!rank) {
    ierr = VecGetArrayRead(Xseq,&x);CHKERRQ(ierr);
    ci = x[0]*x[0] - x[1];
    ierr = VecSetValue(CI,0,ci,INSERT_VALUES);CHKERRQ(ierr);
    ci = -x[0]*x[0] + x[1] + 1.0;
    ierr = VecSetValue(CI,1,ci,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(Xseq,&x);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(CI);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(CI);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode FormEqualityConstraints(Tao tao,Vec X,Vec CE,void *ctx)
{
  const PetscScalar *x;
  PetscScalar       ce;
  PetscErrorCode    ierr;
  MPI_Comm          comm;
  PetscMPIInt       rank;
  AppCtx            *user=(AppCtx*)ctx;
  Vec               Xseq=user->Xseq;
  VecScatter        scat=user->scat;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)tao,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  ierr = VecScatterBegin(scat,X,Xseq,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(scat,X,Xseq,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

  if (!rank) {
    ierr = VecGetArrayRead(Xseq,&x);CHKERRQ(ierr);
    ce = x[0]*x[0] + x[1] - 2.0;
    ierr = VecSetValue(CE,0,ce,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(Xseq,&x);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(CE);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(CE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode FormInequalityJacobian(Tao tao, Vec X, Mat JI, Mat JIpre,  void *ctx)
{
  AppCtx            *user=(AppCtx*)ctx;
  PetscInt          cols[2],min,max,i;
  PetscScalar       vals[2];
  const PetscScalar *x;
  PetscErrorCode    ierr;
  Vec               Xseq=user->Xseq;
  VecScatter        scat=user->scat;
  MPI_Comm          comm;
  PetscMPIInt       rank;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)tao,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = VecScatterBegin(scat,X,Xseq,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(scat,X,Xseq,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

  ierr = VecGetArrayRead(Xseq,&x);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(JI,&min,&max);CHKERRQ(ierr);

  cols[0] = 0; cols[1] = 1;
  for (i=min;i<max;i++) {
    if (i==0){
      vals[0] = +2*x[0]; vals[1] = -1.0;
      ierr = MatSetValues(JI,1,&i,2,cols,vals,INSERT_VALUES);CHKERRQ(ierr);
    }
    if (i==1) {
      vals[0] = -2*x[0]; vals[1] = +1.0;
      ierr = MatSetValues(JI,1,&i,2,cols,vals,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = VecRestoreArrayRead(Xseq,&x);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(JI,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(JI,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode FormEqualityJacobian(Tao tao,Vec X,Mat JE,Mat JEpre,void *ctx)
{
  PetscInt          rows[2];
  PetscScalar       vals[2];
  const PetscScalar *x;
  PetscMPIInt       rank;
  MPI_Comm          comm;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)tao,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  if (!rank) {
    ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
    rows[0] = 0;       rows[1] = 1;
    vals[0] = 2*x[0];  vals[1] = 1.0;
    ierr = MatSetValues(JE,1,rows,2,rows,vals,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(JE,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(JE,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/*TEST

   build:
      requires: !complex !define(PETSC_USE_CXX)

   test:
      requires: superlu_dist
      args: -tao_converged_reason

   test:
      suffix: 2
      requires: superlu_dist
      nsize: 2
      args: -tao_converged_reason

   test:
      suffix: 3
      requires: superlu_dist
      args: -tao_converged_reason -no_eq

   test:
      suffix: 4
      requires: superlu_dist
      nsize: 2
      args: -tao_converged_reason -no_eq

TEST*/
