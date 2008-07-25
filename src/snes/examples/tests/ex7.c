
static char help[] = "Solves u`` + u^{2} = f with Newton-like methods. Using\n\
 matrix-free techniques with user-provided explicit preconditioner matrix.\n\n";

#include "petscsnes.h"

extern PetscErrorCode   FormJacobian(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
extern PetscErrorCode   FormFunction(SNES,Vec,Vec,void*);
extern PetscErrorCode   FormInitialGuess(SNES,Vec);
extern PetscErrorCode   Monitor(SNES,PetscInt,PetscReal,void *);

typedef struct {
   PetscViewer viewer;
} MonitorCtx;

typedef struct {
  Mat        precond;
  PetscTruth variant;
} AppCtx;

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  SNES           snes;                 /* SNES context */
  const SNESType type = SNESLS;        /* default nonlinear solution method */
  Vec            x,r,F,U;              /* vectors */
  Mat            J,B;                  /* Jacobian matrix-free, explicit preconditioner */
  MonitorCtx     monP;                 /* monitoring context */
  AppCtx         user;                 /* user-defined work context */
  PetscScalar    h,xp = 0.0,v;
  PetscInt       its,n = 5,i;
  PetscErrorCode ierr;

  PetscInitialize(&argc,&argv,(char *)0,help);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL,"-variant",&user.variant);CHKERRQ(ierr);
  h = 1.0/(n-1);

  /* Set up data structures */
  ierr = PetscViewerDrawOpen(PETSC_COMM_SELF,0,0,0,0,400,400,&monP.viewer);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,n,&x);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)x,"Approximate Solution");CHKERRQ(ierr);
  ierr = VecDuplicate(x,&r);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&F);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&U);CHKERRQ(ierr); 
  ierr = PetscObjectSetName((PetscObject)U,"Exact Solution");CHKERRQ(ierr);

  /* create explict matrix preconditioner */
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,n,n,3,PETSC_NULL,&B);CHKERRQ(ierr);
  user.precond = B;

  /* Store right-hand-side of PDE and exact solution */
  for (i=0; i<n; i++) {
    v = 6.0*xp + PetscPowScalar(xp+1.e-12,6.0); /* +1.e-12 is to prevent 0^6 */
    ierr = VecSetValues(F,1,&i,&v,INSERT_VALUES);CHKERRQ(ierr);
    v= xp*xp*xp;
    ierr = VecSetValues(U,1,&i,&v,INSERT_VALUES);CHKERRQ(ierr);
    xp += h;
  }

  /* Create nonlinear solver */  
  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
  ierr = SNESSetType(snes,type);CHKERRQ(ierr);

  /* Set various routines and options */
  ierr = SNESSetFunction(snes,r,FormFunction,F);CHKERRQ(ierr);
  if (user.variant) {
    ierr = MatCreateMFFD(PETSC_COMM_WORLD,n,n,n,n,&J);CHKERRQ(ierr);
    ierr = MatMFFDSetFunction(J,(PetscErrorCode (*)(void*, Vec, Vec))SNESComputeFunction,snes);CHKERRQ(ierr);
  } else {
    /* create matrix free matrix for Jacobian */
    ierr = MatCreateSNESMF(snes,&J);CHKERRQ(ierr);
  }

  /* Set various routines and options */
  ierr = SNESSetJacobian(snes,J,B,FormJacobian,&user);CHKERRQ(ierr);
  ierr = SNESMonitorSet(snes,Monitor,&monP,0);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  /* Solve nonlinear system */
  ierr = FormInitialGuess(snes,x);CHKERRQ(ierr);
  ierr = SNESSolve(snes,PETSC_NULL,x);CHKERRQ(ierr);
  ierr = SNESGetIterationNumber(snes,&its);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"number of Newton iterations = %D\n\n",its);CHKERRQ(ierr);

  /* Free data structures */
  ierr = VecDestroy(x);CHKERRQ(ierr);  ierr = VecDestroy(r);CHKERRQ(ierr);
  ierr = VecDestroy(U);CHKERRQ(ierr);  ierr = VecDestroy(F);CHKERRQ(ierr);
  ierr = MatDestroy(J);CHKERRQ(ierr);  ierr = MatDestroy(B);CHKERRQ(ierr);
  ierr = SNESDestroy(snes);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(monP.viewer);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);

  return 0;
}
/* --------------------  Evaluate Function F(x) --------------------- */

PetscErrorCode  FormFunction(SNES snes,Vec x,Vec f,void *dummy)
{
  PetscScalar    *xx,*ff,*FF,d;
  PetscInt       i,n;
  PetscErrorCode ierr;

  ierr = VecGetArray(x,&xx);CHKERRQ(ierr);
  ierr = VecGetArray(f,&ff);CHKERRQ(ierr);
  ierr = VecGetArray((Vec) dummy,&FF);CHKERRQ(ierr);
  ierr = VecGetSize(x,&n);CHKERRQ(ierr);
  d = (PetscReal)(n - 1); d = d*d;
  ff[0]   = xx[0];
  for (i=1; i<n-1; i++) {
    ff[i] = d*(xx[i-1] - 2.0*xx[i] + xx[i+1]) + xx[i]*xx[i] - FF[i];
  }
  ff[n-1] = xx[n-1] - 1.0;
  ierr = VecRestoreArray(x,&xx);CHKERRQ(ierr);
  ierr = VecRestoreArray(f,&ff);CHKERRQ(ierr);
  ierr = VecRestoreArray((Vec)dummy,&FF);CHKERRQ(ierr);
  return 0;
}
/* --------------------  Form initial approximation ----------------- */

#undef __FUNCT__
#define __FUNCT__ "FormInitialGuess"
PetscErrorCode  FormInitialGuess(SNES snes,Vec x)
{
  PetscErrorCode     ierr;
  PetscScalar pfive = .50;
  ierr = VecSet(x,pfive);CHKERRQ(ierr);
  return 0;
}
#undef __FUNCT__
#define __FUNCT__ "FormJacobian"
/* --------------------  Evaluate Jacobian F'(x) -------------------- */
/*  Evaluates a matrix that is used to precondition the matrix-free
    jacobian. In this case, the explict preconditioner matrix is 
    also EXACTLY the Jacobian. In general, it would be some lower
    order, simplified apprioximation */

PetscErrorCode  FormJacobian(SNES snes,Vec x,Mat *jac,Mat *B,MatStructure*flag,void *dummy)
{
  PetscScalar    *xx,A[3],d;
  PetscInt       i,n,j[3],iter;
  PetscErrorCode ierr;
  AppCtx         *user = (AppCtx*) dummy;

  ierr = SNESGetIterationNumber(snes,&iter);CHKERRQ(ierr);

  if (iter%2 ==0) { /* Compute new preconditioner matrix */
    ierr = PetscPrintf(PETSC_COMM_SELF,"iter=%D, computing new preconditioning matrix\n",iter+1);CHKERRQ(ierr);
    *B = user->precond;
    ierr = VecGetArray(x,&xx);CHKERRQ(ierr);
    ierr = VecGetSize(x,&n);CHKERRQ(ierr);
    d = (PetscReal)(n - 1); d = d*d;

    i = 0; A[0] = 1.0; 
    ierr = MatSetValues(*B,1,&i,1,&i,&A[0],INSERT_VALUES);CHKERRQ(ierr);
    for (i=1; i<n-1; i++) {
      j[0] = i - 1; j[1] = i;                   j[2] = i + 1; 
      A[0] = d;     A[1] = -2.0*d + 2.0*xx[i];  A[2] = d; 
      ierr = MatSetValues(*B,1,&i,3,j,A,INSERT_VALUES);CHKERRQ(ierr);
    }
    i = n-1; A[0] = 1.0; 
    ierr = MatSetValues(*B,1,&i,1,&i,&A[0],INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = VecRestoreArray(x,&xx);CHKERRQ(ierr);
    *flag = SAME_NONZERO_PATTERN;
  }  else { /* reuse preconditioner from last iteration */
    ierr = PetscPrintf(PETSC_COMM_SELF,"iter=%D, using old preconditioning matrix\n",iter+1);CHKERRQ(ierr);
    *flag = SAME_PRECONDITIONER;
  }
  if (user->variant) {
    ierr = MatMFFDSetBase(*jac,x,PETSC_NULL);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(*jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  return 0;
}
/* --------------------  User-defined monitor ----------------------- */

#undef __FUNCT__
#define __FUNCT__ "Monitor"
PetscErrorCode  Monitor(SNES snes,PetscInt its,PetscReal fnorm,void *dummy)
{
  PetscErrorCode ierr;
  MonitorCtx     *monP = (MonitorCtx*) dummy;
  Vec            x;
  MPI_Comm       comm;

  ierr = PetscObjectGetComm((PetscObject)snes,&comm);CHKERRQ(ierr);
  ierr = PetscFPrintf(comm,stdout,"iter = %D, SNES Function norm %G \n",its,fnorm);CHKERRQ(ierr);
  ierr = SNESGetSolution(snes,&x);CHKERRQ(ierr);
  ierr = VecView(x,monP->viewer);CHKERRQ(ierr);
  return 0;
}
