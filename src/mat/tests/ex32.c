
static char help[] = "Tests MATSEQDENSECUDA\n\n";

#include <petscmat.h>

int main(int argc,char **argv)
{
  Mat            A,AC,B;
  PetscErrorCode ierr;
  PetscInt       m = 10,n = 10;
  PetscReal      r,tol = 10*PETSC_SMALL;

  ierr = PetscInitialize(&argc,&argv,(char*) 0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(MatCreate(PETSC_COMM_SELF,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m,n));
  CHKERRQ(MatSetType(A,MATSEQDENSE));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSeqDenseSetPreallocation(A,NULL));
  CHKERRQ(MatSetRandom(A,NULL));
#if 0
  PetscInt       i,j;
  PetscScalar    val;
  for (i=0; i<m; i++) {
    for (j=0; j<n; j++) {
      val = (PetscScalar)(i+j);
      CHKERRQ(MatSetValues(A,1,&i,1,&j,&val,INSERT_VALUES));
    }
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
#endif

  /* Create a CUDA version of A */
#if defined(PETSC_HAVE_CUDA)
  CHKERRQ(MatConvert(A,MATSEQDENSECUDA,MAT_INITIAL_MATRIX,&AC));
#else
  CHKERRQ(MatDuplicate(A,MAT_COPY_VALUES,&AC));
#endif
  CHKERRQ(MatDuplicate(AC,MAT_COPY_VALUES,&B));

  /* full CUDA AXPY */
  CHKERRQ(MatAXPY(B,-1.0,AC,SAME_NONZERO_PATTERN));
  CHKERRQ(MatNorm(B,NORM_INFINITY,&r));
  PetscCheckFalse(r != 0.0,PetscObjectComm((PetscObject)B),PETSC_ERR_PLIB,"Error MatDuplicate + MatCopy + MatAXPY %g",(double)r);

  /* test Copy */
  CHKERRQ(MatCopy(AC,B,SAME_NONZERO_PATTERN));

  /* call MatAXPY_Basic since B is CUDA, A is CPU,  */
  CHKERRQ(MatAXPY(B,-1.0,A,SAME_NONZERO_PATTERN));
  CHKERRQ(MatNorm(B,NORM_INFINITY,&r));
  PetscCheckFalse(r != 0.0,PetscObjectComm((PetscObject)B),PETSC_ERR_PLIB,"Error MatDuplicate + MatCopy + MatAXPY_Basic %g",(double)r);

  if (m == n) {
    Mat B1,B2;

    CHKERRQ(MatCopy(AC,B,SAME_NONZERO_PATTERN));
    /* full CUDA PtAP */
    CHKERRQ(MatPtAP(B,AC,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&B1));

    /* CPU PtAP since A is on the CPU only */
    CHKERRQ(MatPtAP(B,A,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&B2));

    CHKERRQ(MatAXPY(B2,-1.0,B1,SAME_NONZERO_PATTERN));
    CHKERRQ(MatNorm(B2,NORM_INFINITY,&r));
    PetscCheckFalse(r > tol,PetscObjectComm((PetscObject)B),PETSC_ERR_PLIB,"Error MatPtAP %g",(double)r);

    /* test reuse */
    CHKERRQ(MatPtAP(B,AC,MAT_REUSE_MATRIX,PETSC_DEFAULT,&B1));
    CHKERRQ(MatPtAP(B,A,MAT_REUSE_MATRIX,PETSC_DEFAULT,&B2));
    CHKERRQ(MatAXPY(B2,-1.0,B1,SAME_NONZERO_PATTERN));
    CHKERRQ(MatNorm(B2,NORM_INFINITY,&r));
    PetscCheckFalse(r > tol,PetscObjectComm((PetscObject)B),PETSC_ERR_PLIB,"Error MatPtAP %g",(double)r);

    CHKERRQ(MatDestroy(&B1));
    CHKERRQ(MatDestroy(&B2));
  }

  CHKERRQ(MatDestroy(&B));
  CHKERRQ(MatDestroy(&AC));
  CHKERRQ(MatDestroy(&A));
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
