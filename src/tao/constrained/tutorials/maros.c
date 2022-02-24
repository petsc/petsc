/* Program usage: mpiexec -n 1 maros1 [-help] [all TAO options] */

/* ----------------------------------------------------------------------
TODO Explain maros example
---------------------------------------------------------------------- */

#include <petsctao.h>

static  char help[]="";

/*T
   Concepts: TAO^Solving an unconstrained minimization problem
   Routines: TaoCreate(); TaoSetType();
   Routines: TaoSetSolution();
   Routines: TaoSetObjectiveAndGradient();
   Routines: TaoSetEqualityConstraintsRoutine();
   Routines: TaoSetInequalityConstraintsRoutine();
   Routines: TaoSetEqualityJacobianRoutine();
   Routines: TaoSetInequalityJacobianRoutine();
   Routines: TaoSetHessian(); TaoSetFromOptions();
   Routines: TaoGetKSP(); TaoSolve();
   Routines: TaoGetConvergedReason();TaoDestroy();
   Processors: 1
T*/

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
  char     name[32];
  PetscInt n; /* Length x */
  PetscInt me; /* number of equality constraints */
  PetscInt mi; /* number of inequality constraints */
  PetscInt m;  /* me+mi */
  Mat      Aeq,Ain,H;
  Vec      beq,bin,d;
} AppCtx;

/* -------- User-defined Routines --------- */

PetscErrorCode InitializeProblem(AppCtx*);
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
  PetscMPIInt        size;
  Vec                x;                   /* solution */
  KSP                ksp;
  PC                 pc;
  Vec                ceq,cin;
  PetscBool          flg;                 /* A return value when checking for use options */
  Tao                tao;                 /* Tao solver context */
  TaoConvergedReason reason;
  AppCtx             user;                /* application context */

  /* Initialize TAO,PETSc */
  ierr = PetscInitialize(&argc,&argv,(char *)0,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  /* Specify default parameters for the problem, check for command-line overrides */
  CHKERRQ(PetscStrncpy(user.name,"HS21",sizeof(user.name)));
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-cutername",user.name,sizeof(user.name),&flg));

  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n---- MAROS Problem %s -----\n",user.name));
  CHKERRQ(InitializeProblem(&user));
  CHKERRQ(VecDuplicate(user.d,&x));
  CHKERRQ(VecDuplicate(user.beq,&ceq));
  CHKERRQ(VecDuplicate(user.bin,&cin));
  CHKERRQ(VecSet(x,1.0));

  CHKERRQ(TaoCreate(PETSC_COMM_WORLD,&tao));
  CHKERRQ(TaoSetType(tao,TAOIPM));
  CHKERRQ(TaoSetSolution(tao,x));
  CHKERRQ(TaoSetObjectiveAndGradient(tao,NULL,FormFunctionGradient,(void*)&user));
  CHKERRQ(TaoSetEqualityConstraintsRoutine(tao,ceq,FormEqualityConstraints,(void*)&user));
  CHKERRQ(TaoSetInequalityConstraintsRoutine(tao,cin,FormInequalityConstraints,(void*)&user));
  CHKERRQ(TaoSetInequalityBounds(tao,user.bin,NULL));
  CHKERRQ(TaoSetJacobianEqualityRoutine(tao,user.Aeq,user.Aeq,FormEqualityJacobian,(void*)&user));
  CHKERRQ(TaoSetJacobianInequalityRoutine(tao,user.Ain,user.Ain,FormInequalityJacobian,(void*)&user));
  CHKERRQ(TaoSetHessian(tao,user.H,user.H,FormHessian,(void*)&user));
  CHKERRQ(TaoGetKSP(tao,&ksp));
  CHKERRQ(KSPGetPC(ksp,&pc));
  CHKERRQ(PCSetType(pc,PCLU));
  /*
      This algorithm produces matrices with zeros along the diagonal therefore we need to use
    SuperLU which does partial pivoting
  */
  CHKERRQ(PCFactorSetMatSolverType(pc,MATSOLVERSUPERLU));
  CHKERRQ(KSPSetType(ksp,KSPPREONLY));
  CHKERRQ(TaoSetTolerances(tao,0,0,0));

  CHKERRQ(TaoSetFromOptions(tao));
  CHKERRQ(TaoSolve(tao));
  CHKERRQ(TaoGetConvergedReason(tao,&reason));
  if (reason < 0) {
    CHKERRQ(PetscPrintf(MPI_COMM_WORLD, "TAO failed to converge due to %s.\n",TaoConvergedReasons[reason]));
  } else {
    CHKERRQ(PetscPrintf(MPI_COMM_WORLD, "Optimization completed with status %s.\n",TaoConvergedReasons[reason]));
  }

  CHKERRQ(DestroyProblem(&user));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&ceq));
  CHKERRQ(VecDestroy(&cin));
  CHKERRQ(TaoDestroy(&tao));

  ierr = PetscFinalize();
  return ierr;
}

PetscErrorCode InitializeProblem(AppCtx *user)
{
  PetscViewer loader;
  MPI_Comm    comm;
  PetscInt    nrows,ncols,i;
  PetscScalar one = 1.0;
  char        filebase[128];
  char        filename[128];

  PetscFunctionBegin;
  comm = PETSC_COMM_WORLD;
  CHKERRQ(PetscStrncpy(filebase,user->name,sizeof(filebase)));
  CHKERRQ(PetscStrlcat(filebase,"/",sizeof(filebase)));
  CHKERRQ(PetscStrncpy(filename,filebase,sizeof(filename)));
  CHKERRQ(PetscStrlcat(filename,"f",sizeof(filename)));
  CHKERRQ(PetscViewerBinaryOpen(comm,filename,FILE_MODE_READ,&loader));

  CHKERRQ(VecCreate(comm,&user->d));
  CHKERRQ(VecLoad(user->d,loader));
  CHKERRQ(PetscViewerDestroy(&loader));
  CHKERRQ(VecGetSize(user->d,&nrows));
  CHKERRQ(VecSetFromOptions(user->d));
  user->n = nrows;

  CHKERRQ(PetscStrncpy(filename,filebase,sizeof(filename)));
  CHKERRQ(PetscStrlcat(filename,"H",sizeof(filename)));
  CHKERRQ(PetscViewerBinaryOpen(comm,filename,FILE_MODE_READ,&loader));

  CHKERRQ(MatCreate(comm,&user->H));
  CHKERRQ(MatSetSizes(user->H,PETSC_DECIDE,PETSC_DECIDE,nrows,nrows));
  CHKERRQ(MatLoad(user->H,loader));
  CHKERRQ(PetscViewerDestroy(&loader));
  CHKERRQ(MatGetSize(user->H,&nrows,&ncols));
  PetscCheck(nrows == user->n,comm,PETSC_ERR_ARG_SIZ,"H: nrows != n");
  PetscCheck(ncols == user->n,comm,PETSC_ERR_ARG_SIZ,"H: ncols != n");
  CHKERRQ(MatSetFromOptions(user->H));

  CHKERRQ(PetscStrncpy(filename,filebase,sizeof(filename)));
  CHKERRQ(PetscStrlcat(filename,"Aeq",sizeof(filename)));
  CHKERRQ(PetscViewerBinaryOpen(comm,filename,FILE_MODE_READ,&loader));
  CHKERRQ(MatCreate(comm,&user->Aeq));
  CHKERRQ(MatLoad(user->Aeq,loader));
  CHKERRQ(PetscViewerDestroy(&loader));
  CHKERRQ(MatGetSize(user->Aeq,&nrows,&ncols));
  PetscCheck(ncols == user->n,comm,PETSC_ERR_ARG_SIZ,"Aeq ncols != H nrows");
  CHKERRQ(MatSetFromOptions(user->Aeq));
  user->me = nrows;

  CHKERRQ(PetscStrncpy(filename,filebase,sizeof(filename)));
  CHKERRQ(PetscStrlcat(filename,"Beq",sizeof(filename)));
  CHKERRQ(PetscViewerBinaryOpen(comm,filename,FILE_MODE_READ,&loader));
  CHKERRQ(VecCreate(comm,&user->beq));
  CHKERRQ(VecLoad(user->beq,loader));
  CHKERRQ(PetscViewerDestroy(&loader));
  CHKERRQ(VecGetSize(user->beq,&nrows));
  PetscCheck(nrows == user->me,comm,PETSC_ERR_ARG_SIZ,"Aeq nrows != Beq n");
  CHKERRQ(VecSetFromOptions(user->beq));

  user->mi = user->n;
  /* Ain = eye(n,n) */
  CHKERRQ(MatCreate(comm,&user->Ain));
  CHKERRQ(MatSetType(user->Ain,MATAIJ));
  CHKERRQ(MatSetSizes(user->Ain,PETSC_DECIDE,PETSC_DECIDE,user->mi,user->mi));

  CHKERRQ(MatMPIAIJSetPreallocation(user->Ain,1,NULL,0,NULL));
  CHKERRQ(MatSeqAIJSetPreallocation(user->Ain,1,NULL));

  for (i=0;i<user->mi;i++) CHKERRQ(MatSetValues(user->Ain,1,&i,1,&i,&one,INSERT_VALUES));
  CHKERRQ(MatAssemblyBegin(user->Ain,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(user->Ain,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatSetFromOptions(user->Ain));

  /* bin = [0,0 ... 0]' */
  CHKERRQ(VecCreate(comm,&user->bin));
  CHKERRQ(VecSetType(user->bin,VECMPI));
  CHKERRQ(VecSetSizes(user->bin,PETSC_DECIDE,user->mi));
  CHKERRQ(VecSet(user->bin,0.0));
  CHKERRQ(VecSetFromOptions(user->bin));
  user->m = user->me + user->mi;
  PetscFunctionReturn(0);
}

PetscErrorCode DestroyProblem(AppCtx *user)
{
  PetscFunctionBegin;
  CHKERRQ(MatDestroy(&user->H));
  CHKERRQ(MatDestroy(&user->Aeq));
  CHKERRQ(MatDestroy(&user->Ain));
  CHKERRQ(VecDestroy(&user->beq));
  CHKERRQ(VecDestroy(&user->bin));
  CHKERRQ(VecDestroy(&user->d));
  PetscFunctionReturn(0);
}

PetscErrorCode FormFunctionGradient(Tao tao, Vec x, PetscReal *f, Vec g, void *ctx)
{
  AppCtx         *user = (AppCtx*)ctx;
  PetscScalar    xtHx;

  PetscFunctionBegin;
  CHKERRQ(MatMult(user->H,x,g));
  CHKERRQ(VecDot(x,g,&xtHx));
  CHKERRQ(VecDot(x,user->d,f));
  *f += 0.5*xtHx;
  CHKERRQ(VecAXPY(g,1.0,user->d));
  PetscFunctionReturn(0);
}

PetscErrorCode FormHessian(Tao tao, Vec x, Mat H, Mat Hpre, void *ctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

PetscErrorCode FormInequalityConstraints(Tao tao, Vec x, Vec ci, void *ctx)
{
  AppCtx         *user = (AppCtx*)ctx;

  PetscFunctionBegin;
  CHKERRQ(MatMult(user->Ain,x,ci));
  PetscFunctionReturn(0);
}

PetscErrorCode FormEqualityConstraints(Tao tao, Vec x, Vec ce,void *ctx)
{
  AppCtx         *user = (AppCtx*)ctx;

  PetscFunctionBegin;
  CHKERRQ(MatMult(user->Aeq,x,ce));
  CHKERRQ(VecAXPY(ce,-1.0,user->beq));
  PetscFunctionReturn(0);
}

PetscErrorCode FormInequalityJacobian(Tao tao, Vec x, Mat JI, Mat JIpre,  void *ctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

PetscErrorCode FormEqualityJacobian(Tao tao, Vec x, Mat JE, Mat JEpre, void *ctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

/*TEST

   build:
      requires: !complex

   test:
      requires: superlu
      localrunfiles: HS21

TEST*/
