
static char help[] = "u`` + u^{2} = f. \n\
Demonstrates the use of Newton-Krylov methods \n\
with different jacobian evaluation routines and matrix colorings. \n\
Modified from ex6.c \n\
Input arguments are:\n\
   -snes_jacobian_default : Jacobian using finite differences. Slow and expensive, not take advantage of sparsity \n\
   -fd_jacobian_coloring:   Jacobian using finite differences with matrix coloring\n\
   -my_jacobian_struct:     use user-provided Jacobian data structure to create matcoloring context \n\n";

/*
  Example: 
  ./ex45 -n 10 -snes_monitor -ksp_monitor
  ./ex45 -n 10 -snes_monitor -ksp_monitor -snes_jacobian_default -pc_type jacobi
  ./ex45 -n 10 -snes_monitor -ksp_monitor -snes_jacobian_default -pc_type ilu
  ./ex45 -n 10 -snes_jacobian_default -log_summary |grep SNESFunctionEval 
  ./ex45 -n 10 -snes_jacobian_default -fd_jacobian_coloring -my_jacobian_struct -log_summary |grep SNESFunctionEval 
 */

#include "petscsnes.h"

/* 
   User-defined routines
*/
PetscErrorCode MyJacobian(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
PetscErrorCode FormFunction(SNES,Vec,Vec,void*);
PetscErrorCode MyApproxJacobianStructure(Mat *,void *);

int main(int argc,char **argv)
{
  SNES           snes;                /* SNES context */
  Vec            x,r,F;               /* vectors */
  Mat            J,JPrec;             /* Jacobian,preconditioner matrices */
  PetscErrorCode ierr;
  PetscInt       it,n = 5,i;
  PetscMPIInt    size;
  PetscReal      h,xp = 0.0; 
  PetscScalar    v,pfive = .5;
  PetscTruth     flg;
  MatFDColoring  matfdcoloring = 0;
  PetscTruth     fd_jacobian_coloring;

  PetscInitialize(&argc,&argv,(char *)0,help);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size != 1) SETERRQ(1,"This is a uniprocessor example only!");
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);
  h = 1.0/(n-1);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create nonlinear solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create vector data structures; set function evaluation routine
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = VecCreate(PETSC_COMM_SELF,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,PETSC_DECIDE,n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&r);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&F);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize application:
     Store right-hand-side of PDE and exact solution
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  xp = 0.0;
  for (i=0; i<n; i++) {
    v = 6.0*xp + pow(xp+1.e-12,6.0); /* +1.e-12 is to prevent 0^6 */
    ierr = VecSetValues(F,1,&i,&v,INSERT_VALUES);CHKERRQ(ierr);
    xp += h;
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Evaluate initial guess; then solve nonlinear system
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = VecSet(x,pfive);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set Function evaluation routine
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = SNESSetFunction(snes,r,FormFunction,(void*)F);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create matrix data structures; set Jacobian evaluation routine
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,n,n,3,PETSC_NULL,&J);CHKERRQ(ierr);
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,n,n,1,PETSC_NULL,&JPrec);CHKERRQ(ierr);
  
  flg = PETSC_FALSE;
  ierr = PetscOptionsGetTruth(PETSC_NULL,"-snes_jacobian_default",&flg,PETSC_NULL);CHKERRQ(ierr);
  if (flg){ 
    /* Jacobian using finite differences. Slow and expensive, not take advantage of sparsity */ 
    ierr = SNESSetJacobian(snes,J,J,SNESDefaultComputeJacobian,PETSC_NULL);CHKERRQ(ierr);
  } else {
    /* User provided Jacobian and preconditioner(diagonal part of Jacobian) */
    ierr = SNESSetJacobian(snes,J,JPrec,MyJacobian,0);CHKERRQ(ierr);
  } 

  fd_jacobian_coloring = PETSC_FALSE;
  ierr = PetscOptionsGetTruth(PETSC_NULL,"-fd_jacobian_coloring",&fd_jacobian_coloring,PETSC_NULL);CHKERRQ(ierr);  
  if (fd_jacobian_coloring){ 
    /* Jacobian using finite differences with matfdcoloring based on the sparse structure.
     In this case, only three calls to FormFunction() for each Jacobian evaluation - very fast! */
    ISColoring    iscoloring;
    
    /* Get the data structure of J */
    flg = PETSC_FALSE;
    ierr = PetscOptionsGetTruth(PETSC_NULL,"-my_jacobian_struct",&flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg){ 
      /* use user-provided jacobian data structure */
      ierr = MyApproxJacobianStructure(&J,PETSC_NULL);CHKERRQ(ierr);
    } else {
      /* use previously defined jacobian: SNESDefaultComputeJacobian() or MyJacobian()  */
      MatStructure  flag;
      ierr = SNESComputeJacobian(snes,x,&J,&J,&flag);CHKERRQ(ierr);
    }

    /* Create coloring context */
    ierr = MatGetColoring(J,MATCOLORING_SL,&iscoloring);CHKERRQ(ierr);
    ierr = MatFDColoringCreate(J,iscoloring,&matfdcoloring);CHKERRQ(ierr);
    ierr = MatFDColoringSetFunction(matfdcoloring,(PetscErrorCode (*)(void))FormFunction,(void*)F);CHKERRQ(ierr);
    ierr = MatFDColoringSetFromOptions(matfdcoloring);CHKERRQ(ierr);
    /* ierr = MatFDColoringView(matfdcoloring,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); */
    
    /* Use SNESDefaultComputeJacobianColor() for Jacobian evaluation */
    ierr = SNESSetJacobian(snes,J,J,SNESDefaultComputeJacobianColor,matfdcoloring);CHKERRQ(ierr); 
    ierr = ISColoringDestroy(iscoloring);CHKERRQ(ierr);
  }

  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = SNESSolve(snes,PETSC_NULL,x);CHKERRQ(ierr);
  ierr = SNESGetIterationNumber(snes,&it);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"Newton iterations = %D\n\n",it);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = VecDestroy(x);CHKERRQ(ierr);     ierr = VecDestroy(r);CHKERRQ(ierr);
  ierr = VecDestroy(F);CHKERRQ(ierr);     ierr = MatDestroy(J);CHKERRQ(ierr);
  ierr = MatDestroy(JPrec);CHKERRQ(ierr); ierr = SNESDestroy(snes);CHKERRQ(ierr);
  if (fd_jacobian_coloring){
    ierr = MatFDColoringDestroy(matfdcoloring);CHKERRQ(ierr);
  }
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
/* ------------------------------------------------------------------- */
/* 
   FormInitialGuess - Forms initial approximation.

   Input Parameters:
   user - user-defined application context
   X - vector

   Output Parameter:
   X - vector
 */
PetscErrorCode FormFunction(SNES snes,Vec x,Vec f,void *dummy)
{
  PetscScalar    *xx,*ff,*FF,d;
  PetscErrorCode ierr;
  PetscInt       i,n;

  ierr = VecGetArray(x,&xx);CHKERRQ(ierr);
  ierr = VecGetArray(f,&ff);CHKERRQ(ierr);
  ierr = VecGetArray((Vec)dummy,&FF);CHKERRQ(ierr);
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
/* ------------------------------------------------------------------- */
/*
   MyJacobian - This routine demonstrates the use of different
   matrices for the Jacobian and preconditioner

   Input Parameters:
.  snes - the SNES context
.  x - input vector
.  ptr - optional user-defined context, as set by SNESSetJacobian()

   Output Parameters:
.  A - Jacobian matrix
.  B - different preconditioning matrix
.  flag - flag indicating matrix structure
*/
PetscErrorCode MyJacobian(SNES snes,Vec x,Mat *jac,Mat *prejac,MatStructure *flag,void *dummy)
{
  PetscScalar    *xx,A[3],d;
  PetscInt       i,n,j[3];
  PetscErrorCode ierr;

  ierr = VecGetArray(x,&xx);CHKERRQ(ierr);
  ierr = VecGetSize(x,&n);CHKERRQ(ierr);
  d = (PetscReal)(n - 1); d = d*d;

  /* Form Jacobian.  Also form a different preconditioning matrix that 
     has only the diagonal elements. */
  i = 0; A[0] = 1.0; 
  ierr = MatSetValues(*jac,1,&i,1,&i,&A[0],INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatSetValues(*prejac,1,&i,1,&i,&A[0],INSERT_VALUES);CHKERRQ(ierr);
  for (i=1; i<n-1; i++) {
    j[0] = i - 1; j[1] = i;                   j[2] = i + 1; 
    A[0] = d;     A[1] = -2.0*d + 2.0*xx[i];  A[2] = d; 
    ierr = MatSetValues(*jac,1,&i,3,j,A,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatSetValues(*prejac,1,&i,1,&i,&A[1],INSERT_VALUES);CHKERRQ(ierr);
  }
  i = n-1; A[0] = 1.0; 
  ierr = MatSetValues(*jac,1,&i,1,&i,&A[0],INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatSetValues(*prejac,1,&i,1,&i,&A[0],INSERT_VALUES);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(*jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*prejac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*prejac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = VecRestoreArray(x,&xx);CHKERRQ(ierr);
  *flag = SAME_NONZERO_PATTERN;
  return 0;
}

/* ------------------------------------------------------------------- */
/*
  Create an approximate data structure for Jacobian matrix to be used with matcoloring

   Input Parameters:
.    A - dummy jacobian matrix 

   Output Parameters:
     A -  jacobian matrix with assigned non-zero structure
 */
PetscErrorCode MyApproxJacobianStructure(Mat *jac,void *dummy)
{
  PetscScalar    zeros[3];
  PetscInt       i,n,j[3];
  PetscErrorCode ierr;

  ierr = MatGetSize(*jac,&n,&n);CHKERRQ(ierr);

  zeros[0] = zeros[1] = zeros[2] = 0.0;
  i = 0; 
  ierr = MatSetValues(*jac,1,&i,1,&i,zeros,INSERT_VALUES);CHKERRQ(ierr);
  
  for (i=1; i<n-1; i++) {
    j[0] = i - 1; j[1] = i; j[2] = i + 1; 
    ierr = MatSetValues(*jac,1,&i,3,j,zeros,INSERT_VALUES);CHKERRQ(ierr);
  }
  i = n-1; 
  ierr = MatSetValues(*jac,1,&i,1,&i,zeros,INSERT_VALUES);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(*jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  return 0;
}
