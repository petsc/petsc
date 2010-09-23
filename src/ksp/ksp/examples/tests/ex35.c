
/*
  Used for testing AIJ matrix with all zeros. 
*/

static char help[] = "Used for Solving a linear system where the matrix has all zeros.\n\n";

#include "petscda.h"
#include "petscksp.h"
#include "petscmg.h"

extern PetscErrorCode ComputeMatrix(DA,Mat);
extern PetscErrorCode ComputeRHS(DA,Vec);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  KSP            ksp;
  PC             pc;
  Vec            x,b;
  DA             da;
  Mat            A;
  PetscInt       dof=1;
  PetscTruth     flg;
  PetscScalar    zero=0.0;

  PetscInitialize(&argc,&argv,(char *)0,help);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-dof",&dof,PETSC_NULL);CHKERRQ(ierr);

  ierr = DACreate(PETSC_COMM_WORLD,&da);CHKERRQ(ierr);
  ierr = DASetDim(da,3);CHKERRQ(ierr);
  ierr = DASetPeriodicity(da,DA_NONPERIODIC);CHKERRQ(ierr);
  ierr = DASetStencilType(da,DA_STENCIL_STAR);CHKERRQ(ierr);
  ierr = DASetSizes(da,3,3,3);CHKERRQ(ierr);
  ierr = DASetNumProcs(da,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = DASetDof(da,dof);CHKERRQ(ierr);
  ierr = DASetStencilWidth(da,1);CHKERRQ(ierr);
  ierr = DASetOwnershipRanges(da,PETSC_NULL,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  ierr = DASetFromOptions(da);CHKERRQ(ierr);

  ierr = DACreateGlobalVector(da,&x);CHKERRQ(ierr);
  ierr = DACreateGlobalVector(da,&b);CHKERRQ(ierr);
  ierr = DAGetMatrix(da,MATAIJ,&A);CHKERRQ(ierr);
  ierr = VecSet(b,zero);CHKERRQ(ierr);

  /* Test sbaij matrix */
  flg = PETSC_FALSE;
  ierr = PetscOptionsGetTruth(PETSC_NULL,"-test_sbaij",&flg,PETSC_NULL);CHKERRQ(ierr);
  if (flg) {
    Mat sA;
    ierr = MatConvert(A,MATSBAIJ,MAT_INITIAL_MATRIX,&sA);CHKERRQ(ierr);
    ierr = MatDestroy(A);CHKERRQ(ierr);
    A = sA;
  }

  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetDM(pc,(DM)da);CHKERRQ(ierr);
 
  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);

  /* check final residual */
  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetTruth(PETSC_NULL, "-check_final_residual", &flg,PETSC_NULL);CHKERRQ(ierr);
  if (flg){
    Vec            b1;
    PetscReal      norm;
    ierr = KSPGetSolution(ksp,&x);CHKERRQ(ierr);
    ierr = VecDuplicate(b,&b1);CHKERRQ(ierr);
    ierr = MatMult(A,x,b1);CHKERRQ(ierr);
    ierr = VecAXPY(b1,-1.0,b);CHKERRQ(ierr);
    ierr = VecNorm(b1,NORM_2,&norm);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Final residual %g\n",norm);CHKERRQ(ierr);
    ierr = VecDestroy(b1);CHKERRQ(ierr);
  }
   
  ierr = KSPDestroy(ksp);CHKERRQ(ierr);
  ierr = VecDestroy(x);CHKERRQ(ierr);
  ierr = VecDestroy(b);CHKERRQ(ierr);
  ierr = MatDestroy(A);CHKERRQ(ierr);
  ierr = DADestroy(da);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}

