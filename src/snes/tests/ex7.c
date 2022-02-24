
static char help[] = "Solves u`` + u^{2} = f with Newton-like methods. Using\n\
 matrix-free techniques with user-provided explicit preconditioner matrix.\n\n";

#include <petscsnes.h>

extern PetscErrorCode FormJacobian(SNES,Vec,Mat,Mat,void*);
extern PetscErrorCode FormJacobianNoMatrix(SNES,Vec,Mat,Mat,void*);
extern PetscErrorCode FormFunction(SNES,Vec,Vec,void*);
extern PetscErrorCode FormFunctioni(void *,PetscInt,Vec,PetscScalar *);
extern PetscErrorCode OtherFunctionForDifferencing(void*,Vec,Vec);
extern PetscErrorCode FormInitialGuess(SNES,Vec);
extern PetscErrorCode Monitor(SNES,PetscInt,PetscReal,void*);

typedef struct {
  PetscViewer viewer;
} MonitorCtx;

typedef struct {
  PetscBool variant;
} AppCtx;

int main(int argc,char **argv)
{
  SNES           snes;                 /* SNES context */
  SNESType       type = SNESNEWTONLS;        /* default nonlinear solution method */
  Vec            x,r,F,U;              /* vectors */
  Mat            J,B;                  /* Jacobian matrix-free, explicit preconditioner */
  AppCtx         user;                 /* user-defined work context */
  PetscScalar    h,xp = 0.0,v;
  PetscInt       its,n = 5,i;
  PetscErrorCode ierr;
  PetscBool      puremf = PETSC_FALSE;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-variant",&user.variant));
  h    = 1.0/(n-1);

  /* Set up data structures */
  CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,n,&x));
  CHKERRQ(PetscObjectSetName((PetscObject)x,"Approximate Solution"));
  CHKERRQ(VecDuplicate(x,&r));
  CHKERRQ(VecDuplicate(x,&F));
  CHKERRQ(VecDuplicate(x,&U));
  CHKERRQ(PetscObjectSetName((PetscObject)U,"Exact Solution"));

  /* create explicit matrix preconditioner */
  CHKERRQ(MatCreateSeqAIJ(PETSC_COMM_SELF,n,n,3,NULL,&B));

  /* Store right-hand-side of PDE and exact solution */
  for (i=0; i<n; i++) {
    v    = 6.0*xp + PetscPowScalar(xp+1.e-12,6.0); /* +1.e-12 is to prevent 0^6 */
    CHKERRQ(VecSetValues(F,1,&i,&v,INSERT_VALUES));
    v    = xp*xp*xp;
    CHKERRQ(VecSetValues(U,1,&i,&v,INSERT_VALUES));
    xp  += h;
  }

  /* Create nonlinear solver */
  CHKERRQ(SNESCreate(PETSC_COMM_WORLD,&snes));
  CHKERRQ(SNESSetType(snes,type));

  /* Set various routines and options */
  CHKERRQ(SNESSetFunction(snes,r,FormFunction,F));
  if (user.variant) {
    /* this approach is not normally needed, one should use the MatCreateSNESMF() below usually */
    CHKERRQ(MatCreateMFFD(PETSC_COMM_WORLD,n,n,n,n,&J));
    CHKERRQ(MatMFFDSetFunction(J,(PetscErrorCode (*)(void*, Vec, Vec))SNESComputeFunction,snes));
    CHKERRQ(MatMFFDSetFunctioni(J,FormFunctioni));
    /* Use the matrix free operator for both the Jacobian used to define the linear system and used to define the preconditioner */
    /* This tests MatGetDiagonal() for MATMFFD */
    CHKERRQ(PetscOptionsHasName(NULL,NULL,"-puremf",&puremf));
  } else {
    /* create matrix free matrix for Jacobian */
    CHKERRQ(MatCreateSNESMF(snes,&J));
    /* demonstrates differencing a different function than FormFunction() to apply a matrix operator */
    /* note we use the same context for this function as FormFunction, the F vector */
    CHKERRQ(MatMFFDSetFunction(J,OtherFunctionForDifferencing,F));
  }

  /* Set various routines and options */
  CHKERRQ(SNESSetJacobian(snes,J,puremf ? J : B,puremf ? FormJacobianNoMatrix : FormJacobian,&user));
  CHKERRQ(SNESSetFromOptions(snes));

  /* Solve nonlinear system */
  CHKERRQ(FormInitialGuess(snes,x));
  CHKERRQ(SNESSolve(snes,NULL,x));
  CHKERRQ(SNESGetIterationNumber(snes,&its));
  CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"number of SNES iterations = %D\n\n",its));

  /* Free data structures */
  CHKERRQ(VecDestroy(&x));  CHKERRQ(VecDestroy(&r));
  CHKERRQ(VecDestroy(&U));  CHKERRQ(VecDestroy(&F));
  CHKERRQ(MatDestroy(&J));  CHKERRQ(MatDestroy(&B));
  CHKERRQ(SNESDestroy(&snes));
  ierr = PetscFinalize();
  return ierr;
}
/* --------------------  Evaluate Function F(x) --------------------- */

PetscErrorCode  FormFunction(SNES snes,Vec x,Vec f,void *dummy)
{
  const PetscScalar *xx,*FF;
  PetscScalar       *ff,d;
  PetscInt          i,n;

  CHKERRQ(VecGetArrayRead(x,&xx));
  CHKERRQ(VecGetArray(f,&ff));
  CHKERRQ(VecGetArrayRead((Vec) dummy,&FF));
  CHKERRQ(VecGetSize(x,&n));
  d     = (PetscReal)(n - 1); d = d*d;
  ff[0] = xx[0];
  for (i=1; i<n-1; i++) ff[i] = d*(xx[i-1] - 2.0*xx[i] + xx[i+1]) + xx[i]*xx[i] - FF[i];
  ff[n-1] = xx[n-1] - 1.0;
  CHKERRQ(VecRestoreArrayRead(x,&xx));
  CHKERRQ(VecRestoreArray(f,&ff));
  CHKERRQ(VecRestoreArrayRead((Vec)dummy,&FF));
  return 0;
}

PetscErrorCode  FormFunctioni(void *dummy,PetscInt i,Vec x,PetscScalar *s)
{
  const PetscScalar *xx,*FF;
  PetscScalar       d;
  PetscInt          n;
  SNES              snes = (SNES) dummy;
  Vec               F;

  CHKERRQ(SNESGetFunction(snes,NULL,NULL,(void**)&F));
  CHKERRQ(VecGetArrayRead(x,&xx));
  CHKERRQ(VecGetArrayRead(F,&FF));
  CHKERRQ(VecGetSize(x,&n));
  d     = (PetscReal)(n - 1); d = d*d;
  if (i == 0) {
    *s = xx[0];
  } else if (i == n-1) {
    *s = xx[n-1] - 1.0;
  } else {
    *s = d*(xx[i-1] - 2.0*xx[i] + xx[i+1]) + xx[i]*xx[i] - FF[i];
  }
  CHKERRQ(VecRestoreArrayRead(x,&xx));
  CHKERRQ(VecRestoreArrayRead(F,&FF));
  return 0;
}

/*

   Example function that when differenced produces the same matrix free Jacobian as FormFunction()
   this is provided to show how a user can provide a different function
*/
PetscErrorCode  OtherFunctionForDifferencing(void *dummy,Vec x,Vec f)
{

  CHKERRQ(FormFunction(NULL,x,f,dummy));
  CHKERRQ(VecShift(f,1.0));
  return 0;
}

/* --------------------  Form initial approximation ----------------- */

PetscErrorCode  FormInitialGuess(SNES snes,Vec x)
{
  PetscScalar    pfive = .50;
  CHKERRQ(VecSet(x,pfive));
  return 0;
}
/* --------------------  Evaluate Jacobian F'(x) -------------------- */
/*  Evaluates a matrix that is used to precondition the matrix-free
    jacobian. In this case, the explicit preconditioner matrix is
    also EXACTLY the Jacobian. In general, it would be some lower
    order, simplified apprioximation */

PetscErrorCode  FormJacobian(SNES snes,Vec x,Mat jac,Mat B,void *dummy)
{
  const PetscScalar *xx;
  PetscScalar       A[3],d;
  PetscInt          i,n,j[3];
  AppCtx            *user = (AppCtx*) dummy;

  CHKERRQ(VecGetArrayRead(x,&xx));
  CHKERRQ(VecGetSize(x,&n));
  d    = (PetscReal)(n - 1); d = d*d;

  i    = 0; A[0] = 1.0;
  CHKERRQ(MatSetValues(B,1,&i,1,&i,&A[0],INSERT_VALUES));
  for (i=1; i<n-1; i++) {
    j[0] = i - 1; j[1] = i;                   j[2] = i + 1;
    A[0] = d;     A[1] = -2.0*d + 2.0*xx[i];  A[2] = d;
    CHKERRQ(MatSetValues(B,1,&i,3,j,A,INSERT_VALUES));
  }
  i     = n-1; A[0] = 1.0;
  CHKERRQ(MatSetValues(B,1,&i,1,&i,&A[0],INSERT_VALUES));
  CHKERRQ(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  CHKERRQ(VecRestoreArrayRead(x,&xx));

  if (user->variant) {
    CHKERRQ(MatMFFDSetBase(jac,x,NULL));
  }
  CHKERRQ(MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY));
  return 0;
}

PetscErrorCode  FormJacobianNoMatrix(SNES snes,Vec x,Mat jac,Mat B,void *dummy)
{
  AppCtx            *user = (AppCtx*) dummy;

  if (user->variant) {
    CHKERRQ(MatMFFDSetBase(jac,x,NULL));
  }
  CHKERRQ(MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY));
  return 0;
}

/* --------------------  User-defined monitor ----------------------- */

PetscErrorCode  Monitor(SNES snes,PetscInt its,PetscReal fnorm,void *dummy)
{
  MonitorCtx     *monP = (MonitorCtx*) dummy;
  Vec            x;
  MPI_Comm       comm;

  CHKERRQ(PetscObjectGetComm((PetscObject)snes,&comm));
  CHKERRQ(PetscFPrintf(comm,stdout,"iter = %D, SNES Function norm %g \n",its,(double)fnorm));
  CHKERRQ(SNESGetSolution(snes,&x));
  CHKERRQ(VecView(x,monP->viewer));
  return 0;
}

/*TEST

   test:
      args: -ksp_gmres_cgs_refinement_type refine_always -snes_monitor_short

   test:
      suffix: 2
      args: -variant -ksp_gmres_cgs_refinement_type refine_always  -snes_monitor_short
      output_file: output/ex7_1.out

   # uses AIJ matrix to define diagonal matrix for Jacobian preconditioning
   test:
      suffix: 3
      args: -variant -pc_type jacobi -snes_view -ksp_monitor

   # uses MATMFFD matrix to define diagonal matrix for Jacobian preconditioning
   test:
      suffix: 4
      args: -variant -pc_type jacobi -puremf  -snes_view -ksp_monitor

TEST*/
