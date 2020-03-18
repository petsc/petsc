
static char help[] = "Tests MATSEQDENSECUDA\n\n";

#include <petscmat.h>

int main(int argc,char **argv)
{
  Mat            A,AC,B;
  PetscErrorCode ierr;
  PetscInt       m = 10,n = 10;
  PetscReal      r,tol = 10*PETSC_SMALL;

  ierr = PetscInitialize(&argc,&argv,(char*) 0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_SELF,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m,n);CHKERRQ(ierr);
  ierr = MatSetType(A,MATSEQDENSE);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSeqDenseSetPreallocation(A,NULL);CHKERRQ(ierr);
  ierr = MatSetRandom(A,NULL);CHKERRQ(ierr);

  /* Create a CUDA version of A */
  ierr = MatConvert(A,MATSEQDENSECUDA,MAT_INITIAL_MATRIX,&AC);CHKERRQ(ierr);
  ierr = MatDuplicate(AC,MAT_COPY_VALUES,&B);CHKERRQ(ierr);

  /* full CUDA AXPY */
  ierr = MatAXPY(B,-1.0,AC,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatNorm(B,NORM_INFINITY,&r);CHKERRQ(ierr);
  if (r != 0.0) SETERRQ1(PetscObjectComm((PetscObject)B),PETSC_ERR_PLIB,"Error MatDuplicate + MatCopy + MatAXPY %g",(double)r);

  /* test Copy */
  ierr = MatCopy(AC,B,SAME_NONZERO_PATTERN);CHKERRQ(ierr);

  /* call MatAXPY_Basic since B is CUDA, A is CPU,  */
  ierr = MatAXPY(B,-1.0,A,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatNorm(B,NORM_INFINITY,&r);CHKERRQ(ierr);
  if (r != 0.0) SETERRQ1(PetscObjectComm((PetscObject)B),PETSC_ERR_PLIB,"Error MatDuplicate + MatCopy + MatAXPY_Basic %g",(double)r);

  if (m == n) {
    Mat B1,B2;

    ierr = MatCopy(AC,B,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    /* full CUDA PtAP */
    ierr = MatPtAP(B,AC,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&B1);CHKERRQ(ierr);
    /* CPU PtAP since A is on the CPU only */
    ierr = MatPtAP(B,A,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&B2);CHKERRQ(ierr);
    ierr = MatAXPY(B2,-1.0,B1,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatNorm(B2,NORM_INFINITY,&r);CHKERRQ(ierr);
    if (r > tol) SETERRQ1(PetscObjectComm((PetscObject)B),PETSC_ERR_PLIB,"Error MatPtAP %g",(double)r);

    /* test reuse */
    ierr = MatPtAP(B,AC,MAT_REUSE_MATRIX,PETSC_DEFAULT,&B1);CHKERRQ(ierr);
    ierr = MatPtAP(B,A,MAT_REUSE_MATRIX,PETSC_DEFAULT,&B2);CHKERRQ(ierr);
    ierr = MatAXPY(B2,-1.0,B1,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatNorm(B2,NORM_INFINITY,&r);CHKERRQ(ierr);
    if (r > tol) SETERRQ1(PetscObjectComm((PetscObject)B),PETSC_ERR_PLIB,"Error MatPtAP %g",(double)r);

    ierr = MatDestroy(&B1);CHKERRQ(ierr);
    ierr = MatDestroy(&B2);CHKERRQ(ierr);
  }

  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = MatDestroy(&AC);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
     requires: cuda

   test:
     output_file: output/ex32_1.out
     args: -m {{3 5 12}} -n {{3 5 12}}
     suffix: seqdensecuda

TEST*/
