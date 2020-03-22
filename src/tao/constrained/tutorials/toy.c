/* Program usage: mpiexec -n 1 toy[-help] [all TAO options] */

/* ----------------------------------------------------------------------
min f=(x1-x2)^2 + (x2-2)^2 -2*x1-2*x2
s.t.     x1^2 + x2 = 2
      0 <= x1^2 - x2 <= 1
      -1 <= x1,x2 <= 2
---------------------------------------------------------------------- */

#include <petsctao.h>

static  char help[]="";

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
  PetscInt n; /* Length x */
  PetscInt ne; /* number of equality constraints */
  PetscInt ni; /* number of inequality constraints */
  Vec      x,xl,xu;
  Vec      ce,ci,bl,bu;
  Mat      Ae,Ai,H;
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
  PetscErrorCode     ierr;                /* used to check for functions returning nonzeros */
  Tao                tao;
  KSP                ksp;
  PC                 pc;
  AppCtx             user;                /* application context */

  ierr = PetscInitialize(&argc,&argv,(char *)0,help);if (ierr) return ierr;
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n---- TOY Problem -----\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Solution should be f(1,1)=-2\n");CHKERRQ(ierr);
  ierr = InitializeProblem(&user);CHKERRQ(ierr);
  ierr = TaoCreate(PETSC_COMM_WORLD,&tao);CHKERRQ(ierr);
  ierr = TaoSetType(tao,TAOIPM);CHKERRQ(ierr);
  ierr = TaoSetInitialVector(tao,user.x);CHKERRQ(ierr);
  ierr = TaoSetVariableBounds(tao,user.xl,user.xu);CHKERRQ(ierr);
  ierr = TaoSetObjectiveAndGradientRoutine(tao,FormFunctionGradient,(void*)&user);CHKERRQ(ierr);

  ierr = TaoSetEqualityConstraintsRoutine(tao,user.ce,FormEqualityConstraints,(void*)&user);CHKERRQ(ierr);
  ierr = TaoSetInequalityConstraintsRoutine(tao,user.ci,FormInequalityConstraints,(void*)&user);CHKERRQ(ierr);

  ierr = TaoSetJacobianEqualityRoutine(tao,user.Ae,user.Ae,FormEqualityJacobian,(void*)&user);CHKERRQ(ierr);
  ierr = TaoSetJacobianInequalityRoutine(tao,user.Ai,user.Ai,FormInequalityJacobian,(void*)&user);CHKERRQ(ierr);
  ierr = TaoSetHessianRoutine(tao,user.H,user.H,FormHessian,(void*)&user);CHKERRQ(ierr);
  /* ierr = TaoSetTolerances(tao,0,0,0);CHKERRQ(ierr); */

  ierr = TaoSetFromOptions(tao);CHKERRQ(ierr);

  ierr = TaoGetKSP(tao,&ksp);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCLU);CHKERRQ(ierr);
  /*
      This algorithm produces matrices with zeros along the diagonal therefore we need to use
    SuperLU which does partial pivoting
  */
  ierr = PCFactorSetMatSolverType(pc,MATSOLVERSUPERLU);CHKERRQ(ierr);
  ierr = KSPSetType(ksp,KSPPREONLY);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

  /* ierr = TaoSetTolerances(tao,0,0,0);CHKERRQ(ierr); */
  ierr = TaoSolve(tao);CHKERRQ(ierr);

  ierr = DestroyProblem(&user);CHKERRQ(ierr);
  ierr = TaoDestroy(&tao);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

PetscErrorCode InitializeProblem(AppCtx *user)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  user->n = 2;
  ierr = VecCreateSeq(PETSC_COMM_SELF,user->n,&user->x);CHKERRQ(ierr);
  ierr = VecDuplicate(user->x,&user->xl);CHKERRQ(ierr);
  ierr = VecDuplicate(user->x,&user->xu);CHKERRQ(ierr);
  ierr = VecSet(user->x,0.0);CHKERRQ(ierr);
  ierr = VecSet(user->xl,-1.0);CHKERRQ(ierr);
  ierr = VecSet(user->xu,2.0);CHKERRQ(ierr);

  user->ne = 1;
  ierr = VecCreateSeq(PETSC_COMM_SELF,user->ne,&user->ce);CHKERRQ(ierr);

  user->ni = 2;
  ierr = VecCreateSeq(PETSC_COMM_SELF,user->ni,&user->ci);CHKERRQ(ierr);

  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,user->ne,user->n,user->n,NULL,&user->Ae);CHKERRQ(ierr);
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,user->ni,user->n,user->n,NULL,&user->Ai);CHKERRQ(ierr);
  ierr = MatSetFromOptions(user->Ae);CHKERRQ(ierr);
  ierr = MatSetFromOptions(user->Ai);CHKERRQ(ierr);


  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,user->n,user->n,1,NULL,&user->H);CHKERRQ(ierr);
  ierr = MatSetFromOptions(user->H);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode DestroyProblem(AppCtx *user)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatDestroy(&user->Ae);CHKERRQ(ierr);
  ierr = MatDestroy(&user->Ai);CHKERRQ(ierr);
  ierr = MatDestroy(&user->H);CHKERRQ(ierr);

  ierr = VecDestroy(&user->x);CHKERRQ(ierr);
  ierr = VecDestroy(&user->ce);CHKERRQ(ierr);
  ierr = VecDestroy(&user->ci);CHKERRQ(ierr);
  ierr = VecDestroy(&user->xl);CHKERRQ(ierr);
  ierr = VecDestroy(&user->xu);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode FormFunctionGradient(Tao tao, Vec X, PetscReal *f, Vec G, void *ctx)
{
  PetscScalar       *g;
  const PetscScalar *x;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecGetArray(G,&g);CHKERRQ(ierr);
  *f = (x[0]-2.0)*(x[0]-2.0) + (x[1]-2.0)*(x[1]-2.0) - 2.0*(x[0]+x[1]);
  g[0] = 2.0*(x[0]-2.0) - 2.0;
  g[1] = 2.0*(x[1]-2.0) - 2.0;
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(G,&g);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode FormHessian(Tao tao, Vec x, Mat H, Mat Hpre, void *ctx)
{
  Vec               DE,DI;
  const PetscScalar *de, *di;
  PetscInt          zero=0,one=1;
  PetscScalar       two=2.0;
  PetscScalar       val;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = TaoGetDualVariables(tao,&DE,&DI);CHKERRQ(ierr);

  ierr = VecGetArrayRead(DE,&de);CHKERRQ(ierr);
  ierr = VecGetArrayRead(DI,&di);CHKERRQ(ierr);
  val=2.0 * (1 + de[0] + di[0] - di[1]);
  ierr = VecRestoreArrayRead(DE,&de);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(DI,&di);CHKERRQ(ierr);

  ierr = MatSetValues(H,1,&zero,1,&zero,&val,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatSetValues(H,1,&one,1,&one,&two,INSERT_VALUES);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(H,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(H,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode FormInequalityConstraints(Tao tao, Vec X, Vec CI, void *ctx)
{
  const PetscScalar *x;
  PetscScalar       *c;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecGetArray(CI,&c);CHKERRQ(ierr);
  c[0] = x[0]*x[0] - x[1];
  c[1] = -x[0]*x[0] + x[1] + 1.0;
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(CI,&c);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode FormEqualityConstraints(Tao tao, Vec X, Vec CE,void *ctx)
{
  PetscScalar    *x,*c;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  ierr = VecGetArray(CE,&c);CHKERRQ(ierr);
  c[0] = x[0]*x[0] + x[1] - 2.0;
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(CE,&c);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode FormInequalityJacobian(Tao tao, Vec X, Mat JI, Mat JIpre,  void *ctx)
{
  PetscInt          rows[2];
  PetscInt          cols[2];
  PetscScalar       vals[4];
  const PetscScalar *x;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  rows[0] = 0;       rows[1] = 1;
  cols[0] = 0;       cols[1] = 1;
  vals[0] = +2*x[0]; vals[1] = -1.0;
  vals[2] = -2*x[0]; vals[3] = +1.0;
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = MatSetValues(JI,2,rows,2,cols,vals,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(JI,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(JI,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode FormEqualityJacobian(Tao tao, Vec X, Mat JE, Mat JEpre, void *ctx)
{
  PetscInt          rows[2];
  PetscScalar       vals[2];
  const PetscScalar *x;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  rows[0] = 0;       rows[1] = 1;
  vals[0] = 2*x[0];  vals[1] = 1.0;
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = MatSetValues(JE,1,rows,2,rows,vals,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(JE,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(JE,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/*TEST

   build:
      requires: !complex !define(PETSC_USE_CXX)

   test:
      requires: superlu
      args: -tao_smonitor -tao_view -tao_gatol 1.e-5

TEST*/
