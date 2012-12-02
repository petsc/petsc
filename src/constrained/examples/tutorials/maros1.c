/* Program usage: mpirun -np 1 maros1 [-help] [all TAO options] */

/* ----------------------------------------------------------------------
TODO Explain maros example
---------------------------------------------------------------------- */

#include "taosolver.h"


static  char help[]="";

/*T 
   Concepts: TAO - Solving an unconstrained minimization problem
   Routines: TaoInitialize(); TaoFinalize(); 
   Routines: TaoCreate(); TaoSetType();
   Routines: TaoSetInitialVector(); 
   Routines: TaoSetObjectiveAndGradientRoutine();
   Routiens: TaoSetEqualityConstraintsRoutine();
   Routines: TaoSetInequalityConstraintsRoutine();
   Routines: TaoSetEqualityJacobianRoutine();
   Routines: TaoSetInequalityJacobianRoutine();
   Routines: TaoSetHessianRoutine(); TaoSetFromOptions();
   Routines: TaoGetKSP(); TaoSolve();
   Routines: TaoGetTerminationReason(); TaoDestroy();
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
  char name[32];
  PetscInt n; /* Length x */
  PetscInt me; /* number of equality constraints */
  PetscInt mi; /* number of inequality constraints */
  PetscInt m;  /* me+mi */
  Mat Aeq,Ain,H;
  Vec beq,bin,d;
} AppCtx;

/* -------- User-defined Routines --------- */

PetscErrorCode InitializeProblem(AppCtx*);
PetscErrorCode FormFunctionGradient(TaoSolver,Vec,PetscReal *,Vec,void *);
PetscErrorCode FormHessian(TaoSolver,Vec,Mat*,Mat*, MatStructure *,void*);
PetscErrorCode FormInequalityConstraints(TaoSolver,Vec,Vec,void*);
PetscErrorCode FormEqualityConstraints(TaoSolver,Vec,Vec,void*);
PetscErrorCode FormInequalityJacobian(TaoSolver,Vec,Mat*,Mat*, MatStructure *,void*);
PetscErrorCode FormEqualityJacobian(TaoSolver,Vec,Mat*,Mat*, MatStructure *,void*);



#undef __FUNCT__
#define __FUNCT__ "main"
PetscErrorCode main(int argc,char **argv)
{
  PetscErrorCode ierr;                /* used to check for functions returning nonzeros */
  Vec         x;                   /* solution */
  Vec         ceq,cin;
  PetscBool   flg;                 /* A return value when checking for use options */
  TaoSolver   tao;                 /* TaoSolver solver context */
  Vec         constraintsE,constraintsI;
  TaoSolverTerminationReason reason;        
  AppCtx      user;                /* application context */

  /* Initialize TAO,PETSc */

  /* Specify default parameters for the problem, check for command-line overrides */
  ierr = PetscStrncpy(user.name,"LOTSCHD",8); CHKERRQ(ierr);
  ierr = PetscOptionsGetString(PETSC_NULL,"-cutername",&user.name,24,&flg);CHKERRQ(ierr);

  PetscInitialize(&argc,&argv,(char *)0,help);
  TaoInitialize(&argc,&argv,(char *)0,help);
  PetscPrintf(PETSC_COMM_SELF,"\n---- MAROS Problem %s -----\n",user.name);CHKERRQ(ierr);
  ierr = InitializeProblem(&user);CHKERRQ(ierr);
  ierr = VecDuplicate(user->d,&x);CHKERRQ(ierr);
  ierr = VecDuplicate(user->beq,&ceq);CHKERRQ(ierr);
  ierr = VecDuplicate(user->bin,&cin);CHKERRQ(ierr);
  ierr = VecSet(x,1.0);CHKERRQ(ierr);

  ierr = TaoCreate(PETSC_COMM_SELF,&tao);CHKERRQ(ierr);
  ierr = TaoSetType(tao,TAO_IPM);CHKERRQ(ierr);
  ierr = TaoSetInitialVector(tao,x);CHKERRQ(ierr);
  ierr = TaoSetObjectiveAndGradientRoutine(tao,FormObjectiveAndGradient,(void*)&user);CHKERRQ(ierr);
  ierr = TaoSetEqualityConstraintsRoutine(tao,ceq,FormEqualityConstraints,(void*)&user);CHKERRQ(ierr);
  ierr = TaoSetInequalityConstraintsRoutine(tao,cin,FormInequalityConstraints,(void*)&user);CHKERRQ(ierr);
  ierr = TaoSetEqualityJacobianRoutine(tao,user.Aeq,user.Aeq,FormEqualityJacobian,(void*)&user);CHKERRQ(ierr);
  ierr = TaoSetInequalityJacobianRoutine(tao,user.Ain,user.Ain,FormInequalityJacobian,(void*)&user);CHKERRQ(ierr);
  ierr = TaoSetHessianRoutine(tao,user.H,user.H,FormHessian,(void*)&user);CHKERRQ(ierr);
  ierr = TaoSetFromOptions(tao);CHKERRQ(ierr);

  /* Solve */
  ierr = TaoSolve(tao);CHKERRQ(ierr);


  /* Analyze solution */
  ierr = TaoGetTerminationReason(tao,&reason);CHKERRQ(ierr);
  if (reason < 0) {
    PetscPrintf(MPI_COMM_WORLD, "TAO failed to converge.\n");
  }
  else {
    PetscPrintf(MPI_COMM_WORLD, "Optimization terminated with status %D.\n", reason);
  }



  /* Finalize Memory */
  ierr = DestroyProblem(&user);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = TaoDestroy(&tao);CHKERRQ(ierr);
  /* Finalize TAO, PETSc */
  TaoFinalize();
  PetscFinalize();

  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "InitializeProblem"
PetscErrorCode InitializeProblem(AppCtx *user)
{
  PetscViewer loader;
  MPIComm     comm;
  PetscInt    nrows,ncols,i;
  PetscScalar one=1.0;
  char[128]   filebase;
  char[128]   filename;

  PetscFunctionBegin;
  comm = PETSC_COMM_SELF;
  ierr = PetscStrncpy(filebase,user.name,128);CHKERRQ(ierr);
  ierr = PetscStrncat(filebase,"/",1);CHKERRQ(ierr);



  ierr = PetscStrncpy(filename,filebase,128);CHKERRQ(ierr);
  ierr = PetscStrncat(filename,"f",3);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(comm,filename,FILE_MODE_READ,&loader);CHKERRQ(ierr);
  if (ierr) {
    SETERRQ(comm,"file 'f' not found");
  } else {
    ierr = VecCreate(comm,&user->d);CHKERRQ(ierr);
    ierr = VecLoad(user->d,loader);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&loader);CHKERRQ(ierr);
    ierr = VecGetSize(user->d,&nrows);CHKERRQ(ierr);
    user->n = nrows;
  }
    
  ierr = PetscStrncpy(filename,filebase,128);CHKERRQ(ierr);
  ierr = PetscStrncat(filename,"H",3);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(comm,filename,FILE_MODE_READ,&loader);CHKERRQ(ierr);
  user->n=0;
  if (ierr) {
    user->H = 0;
  } else {
    ierr = MatCreate(comm,&user->H);CHKERRQ(ierr);
    ierr = MatLoad(user->H,loader);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&loader);CHKERRQ(ierr);
    ierr = MatGetSize(user->H,&nrows,&ncols);CHKERRQ(ierr);
    if (nrows != user->n) {
      SETERRQ(comm,"H: nrows != n\n");
    }
    if (ncols != user->n) {
      SETERRQ(comm,"H: ncols != n\n");
    }

  }

  ierr = PetscStrncpy(filename,filebase,128);CHKERRQ(ierr);
  ierr = PetscStrncat(filename,"Aeq",3);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(comm,filename,FILE_MODE_READ,&loader);
  if (ierr) {
    user->Aeq=0;
    user->me=0;
  } else {
    ierr = MatCreate(comm,&user->Aeq);CHKERRQ(ierr);
    ierr = MatLoad(user->Aeq,loader);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&loader);CHKERRQ(ierr);
    ierr = MatGetSize(user->Aeq,&nrows,&ncols);CHKERRQ(ierr);
    if (ncols != user->n) {
      SETERRQ(comm,"Aeq ncols != H nrows\n");
    }
    user->me = nrows;
  }

  ierr = PetscStrncpy(filename,filebase,128);CHKERRQ(ierr);
  ierr = PetscStrncat(filename,"Beq",3);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(comm,filename,FILE_MODE_READ,&loader);CHKERRQ(ierr);
  if (ierr) {
    user->beq = 0;
  } else {
    ierr = VecCreate(comm,&user->beq);CHKERRQ(ierr);
    ierr = VecLoad(user->beq,loader);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&loader);CHKERRQ(ierr);
    ierr = VecGetSize(user->beq,&nrows);CHKERRQ(ierr);
    if (nrows != user->me) {
      SETERRQ(comm,"Aeq nrows != Beq n\n");
    }
  }

  user->mi = user->n;
  /* Ain = eye(n,n) */
  ierr = MatCreate(comm,&user->Ain);CHKERRQ(ierr);
  ierr = MatSetSizes(user->Ain,user->mi,user->mi,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(user->Ain,1,PETSC_NULL);CHKERRQ(ierr);
  for (i=0;i<user->mi;i++) {
    ierr = MatSetValues(user->Ain,1,&i,1,&i,&one,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(user->Ain,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(user->Ain,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* bin = [0,0 ... 0]' */
  ierr = VecCreate(comm,&user->bin);CHKERRQ(ierr);
  ierr = VecSetSize(user->bin,&user->mi,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSet(user->bin,0.0);CHKERRQ(ierr);
  user->m = user->me + user->mi;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DestroyProblem"
PetscErrorCode DestroyProblem(AppCtx *user)
{
  ierr = MatDestroy(&user->H);CHKERRQ(ierr);
  ierr = MatDestroy(&user->Aeq);CHKERRQ(ierr);
  ierr = MatDestroy(&user->Ain);CHKERRQ(ierr);
  ierr = VecDestroy(&user->beq);CHKERRQ(ierr);
  ierr = VecDestroy(&user->bin);CHKERRQ(ierr);
  ierr = VecDestroy(&user->d);CHKERRQ(ierr);
}
#undef __FUNCT__
#define __FUNCT__ "FormFunctionGradient"
PetscErrorCode FormFunctionGradient(TaoSolver tao, Vec x, PetscReal *f, Vec g, void *ctx)
{
  AppCtx *user = (AppCtx*)ctx;
  PetscScalar xtHx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMult(user->H,x,g);CHKERRQ(ierr);
  ierr = VecDot(x,g,&xtHx);CHKERRQ(ierr);
  ierr = VecDot(x,user->d,f);CHKERRQ(ierr);
  *f += 0.5*xtHx;
  ierr = VecAXPY(g,1.0,user->d);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormHessian"
PetscErrorCode FormHessian(TaoSolver tao, Vec x, Mat *H, Mat *Hpre, MatStructure *ms, void *ctx)
{
  AppCtx *user = (AppCtx*)ctx;

  PetscFunctionBegin;
  *H = user->H;
  *Hpre = user->H;
  *ms = SAME_NONZERO_PATTERN;
  PetscFunctionReturn(0);
}

#unedf __FUNCT__
#define __FUNCT__ "FormInequaliteConstraints"
PetscErrorCode FormInequalityConstraints(TaoSolver tao, Vec x, Vec ci, void *ctx)
{
  AppCtx *user = (AppCtx*)ctx;
  PetscFunctionBegin;
  /* 
  ierr = MatMult(user->Ain,x,ci);CHKERRQ(ierr);
  ierr = VecAXPY(ci,-1.0,user->bin);CHKERRQ(ierr);
  /*
  /* Special case -- Ain =I and bin =0 */
  ierr = VecCopy(x,ci);CHKERRQ(ierr);
  PetscFunctionReturn(0);
  
}

PetscErrorCode FormEqualityConstraints(TaoSolver tao, Vec x, Vec ce,void *ctx)
{
  AppCtx *user = (AppCtx*)ctx;
  PetscFunctionBegin;

  ierr = MatMult(user->Aeq,x,ce);CHKERRQ(ierr);
  ierr = VecAXPY(ce,-1.0,user->beq);CHKERRQ(ierr);

  PetscFunctionReturn(0);

}

PetscErrorCode FormInequalityJacobian(TaoSolver tao, Vec x, Mat *JI, Mat *JIpre,  MatStructure *ms, void *ctx)
{
  AppCtx *user = (AppCtx*)ctx;
  PetscFunctionBegin;
  *JI = user->Ain;
  *JIpre = user->Ain;
  *ms = SAME_NONZERO_PATTERN;
  PetscFunctionReturn(0);
}


PetscErrorCode FormEqualityJacobian(TaoSolver tao, Vec x, Mat *JE, Mat *JEpre, MatStructure *ms, void *ctx)
{
  AppCtx *user = (AppCtx*)ctx;
  PetscFunctionBegin;
  *JE = user->Aeq;
  *JEpre = user->Aeq;
  *ms = SAME_NONZERO_PATTERN;
  PetscFunctionReturn(0);
}
