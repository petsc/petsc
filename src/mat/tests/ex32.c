
static char help[] = "Tests MATSEQDENSECUDA\n\n";

#include <petscmat.h>

int main(int argc,char **argv)
{
  Mat       A,AC,B;
  PetscInt  m     = 10,n = 10;
  PetscReal r,tol = 10*PETSC_SMALL;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,(char*) 0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  PetscCall(MatCreate(PETSC_COMM_SELF,&A));
  PetscCall(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m,n));
  PetscCall(MatSetType(A,MATSEQDENSE));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSeqDenseSetPreallocation(A,NULL));
  PetscCall(MatSetRandom(A,NULL));
#if 0
  PetscInt       i,j;
  PetscScalar    val;
  for (i=0; i<m; i++) {
    for (j=0; j<n; j++) {
      val = (PetscScalar)(i+j);
      PetscCall(MatSetValues(A,1,&i,1,&j,&val,INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
#endif

  /* Create a CUDA version of A */
#if defined(PETSC_HAVE_CUDA)
  PetscCall(MatConvert(A,MATSEQDENSECUDA,MAT_INITIAL_MATRIX,&AC));
#else
  PetscCall(MatDuplicate(A,MAT_COPY_VALUES,&AC));
#endif
  PetscCall(MatDuplicate(AC,MAT_COPY_VALUES,&B));

  /* full CUDA AXPY */
  PetscCall(MatAXPY(B,-1.0,AC,SAME_NONZERO_PATTERN));
  PetscCall(MatNorm(B,NORM_INFINITY,&r));
  PetscCheck(r == 0.0,PetscObjectComm((PetscObject)B),PETSC_ERR_PLIB,"Error MatDuplicate + MatCopy + MatAXPY %g",(double)r);

  /* test Copy */
  PetscCall(MatCopy(AC,B,SAME_NONZERO_PATTERN));

  /* call MatAXPY_Basic since B is CUDA, A is CPU,  */
  PetscCall(MatAXPY(B,-1.0,A,SAME_NONZERO_PATTERN));
  PetscCall(MatNorm(B,NORM_INFINITY,&r));
  PetscCheck(r == 0.0,PetscObjectComm((PetscObject)B),PETSC_ERR_PLIB,"Error MatDuplicate + MatCopy + MatAXPY_Basic %g",(double)r);

  if (m == n) {
    Mat B1,B2;

    PetscCall(MatCopy(AC,B,SAME_NONZERO_PATTERN));
    /* full CUDA PtAP */
    PetscCall(MatPtAP(B,AC,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&B1));

    /* CPU PtAP since A is on the CPU only */
    PetscCall(MatPtAP(B,A,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&B2));

    PetscCall(MatAXPY(B2,-1.0,B1,SAME_NONZERO_PATTERN));
    PetscCall(MatNorm(B2,NORM_INFINITY,&r));
    PetscCheck(r <= tol,PetscObjectComm((PetscObject)B),PETSC_ERR_PLIB,"Error MatPtAP %g",(double)r);

    /* test reuse */
    PetscCall(MatPtAP(B,AC,MAT_REUSE_MATRIX,PETSC_DEFAULT,&B1));
    PetscCall(MatPtAP(B,A,MAT_REUSE_MATRIX,PETSC_DEFAULT,&B2));
    PetscCall(MatAXPY(B2,-1.0,B1,SAME_NONZERO_PATTERN));
    PetscCall(MatNorm(B2,NORM_INFINITY,&r));
    PetscCheck(r <= tol,PetscObjectComm((PetscObject)B),PETSC_ERR_PLIB,"Error MatPtAP %g",(double)r);

    PetscCall(MatDestroy(&B1));
    PetscCall(MatDestroy(&B2));
  }

  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&AC));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
     requires: cuda

   test:
     output_file: output/ex32_1.out
     args: -m {{3 5 12}} -n {{3 5 12}}
     suffix: seqdensecuda

TEST*/
