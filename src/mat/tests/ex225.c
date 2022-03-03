
static char help[] = "Test Hypre matrix APIs\n";

#include <petscmathypre.h>

int main(int argc,char **args)
{
  Mat            A, B, C;
  PetscReal      err;
  PetscInt       i,j,M = 20;
  PetscErrorCode ierr;
  PetscMPIInt    NP;
  MPI_Comm       comm;
  PetscInt       *rows;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  CHKERRMPI(MPI_Comm_size(comm,&NP));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL));
  PetscCheckFalse(M < 6,PETSC_COMM_WORLD,PETSC_ERR_SUP,"Matrix has to have more than 6 columns");
  /* Hypre matrix */
  CHKERRQ(MatCreate(comm,&B));
  CHKERRQ(MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,M,M));
  CHKERRQ(MatSetType(B,MATHYPRE));
  CHKERRQ(MatHYPRESetPreallocation(B,9,NULL,9,NULL));

  /* PETSc AIJ matrix */
  CHKERRQ(MatCreate(comm,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,M,M));
  CHKERRQ(MatSetType(A,MATAIJ));
  CHKERRQ(MatSeqAIJSetPreallocation(A,9,NULL));
  CHKERRQ(MatMPIAIJSetPreallocation(A,9,NULL,9,NULL));

  /*Set Values */
  for (i=0; i<M; i++) {
    PetscInt    cols[] = {0,1,2,3,4,5};
    PetscScalar vals[6] = {0};
    PetscScalar value[] = {100};
    for (j=0; j<6; j++)
      vals[j] = ((PetscReal)j)/NP;

    CHKERRQ(MatSetValues(B,1,&i,6,cols,vals,ADD_VALUES));
    CHKERRQ(MatSetValues(B,1,&i,1,&i,value,ADD_VALUES));
    CHKERRQ(MatSetValues(A,1,&i,6,cols,vals,ADD_VALUES));
    CHKERRQ(MatSetValues(A,1,&i,1,&i,value,ADD_VALUES));
  }

  /* MAT_FLUSH_ASSEMBLY currently not supported */
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));

  /* Compare A and B */
  CHKERRQ(MatConvert(B,MATAIJ,MAT_INITIAL_MATRIX,&C));
  CHKERRQ(MatAXPY(C,-1.,A,SAME_NONZERO_PATTERN));
  CHKERRQ(MatNorm(C,NORM_INFINITY,&err));
  PetscCheckFalse(err > PETSC_SMALL,PetscObjectComm((PetscObject)B),PETSC_ERR_PLIB,"Error MatSetValues %g",err);
  CHKERRQ(MatDestroy(&C));

  /* MatZeroRows */
  CHKERRQ(PetscMalloc1(M, &rows));
  for (i=0; i<M; i++) rows[i] = i;
  CHKERRQ(MatZeroRows(B, M, rows, 10.0, NULL, NULL));
  CHKERRQ(MatSetOption(A, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE));
  CHKERRQ(MatZeroRows(A, M, rows, 10.0,NULL, NULL));
  CHKERRQ(MatConvert(B,MATAIJ,MAT_INITIAL_MATRIX,&C));
  CHKERRQ(MatAXPY(C,-1.,A,SAME_NONZERO_PATTERN));
  CHKERRQ(MatNorm(C,NORM_INFINITY,&err));
  PetscCheckFalse(err > PETSC_SMALL,PetscObjectComm((PetscObject)B),PETSC_ERR_PLIB,"Error MatZeroRows %g",err);
  CHKERRQ(MatDestroy(&C));
  CHKERRQ(PetscFree(rows));

  /* Test MatZeroEntries */
  CHKERRQ(MatZeroEntries(B));
  CHKERRQ(MatConvert(B,MATAIJ,MAT_INITIAL_MATRIX,&C));
  CHKERRQ(MatNorm(C,NORM_INFINITY,&err));
  PetscCheckFalse(err > PETSC_SMALL,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Error MatZeroEntries %g",err);
  CHKERRQ(MatDestroy(&C));

  /* Insert Values */
  for (i=0; i<M; i++) {
    PetscInt    cols[] = {0,1,2,3,4,5};
    PetscScalar vals[6] = {0};
    PetscScalar value[] = {100};

    for (j=0; j<6; j++)
      vals[j] = ((PetscReal)j)/NP;

    CHKERRQ(MatSetValues(B,1,&i,6,cols,vals,INSERT_VALUES));
    CHKERRQ(MatSetValues(B,1,&i,1,&i,value,INSERT_VALUES));
    CHKERRQ(MatSetValues(A,1,&i,6,cols,vals,INSERT_VALUES));
    CHKERRQ(MatSetValues(A,1,&i,1,&i,value,INSERT_VALUES));
  }

  /* MAT_FLUSH_ASSEMBLY currently not supported */
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));

  /* Rows are not sorted with HYPRE so we need an intermediate sort
     They use a temporary buffer, so we can sort inplace the const memory */
  {
    const PetscInt    *idxA,*idxB;
    const PetscScalar *vA, *vB;
    PetscInt          rstart, rend, nzA, nzB;
    PetscInt          cols[] = {0,1,2,3,4,-5};
    PetscInt          *rows;
    PetscScalar       *valuesA, *valuesB;
    PetscBool         flg;

    CHKERRQ(MatGetOwnershipRange(A,&rstart,&rend));
    for (i=rstart; i<rend; i++) {
      CHKERRQ(MatGetRow(A,i,&nzA,&idxA,&vA));
      CHKERRQ(MatGetRow(B,i,&nzB,&idxB,&vB));
      PetscCheckFalse(nzA!=nzB,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error MatGetRow %" PetscInt_FMT, nzA-nzB);
      CHKERRQ(PetscSortIntWithScalarArray(nzB,(PetscInt*)idxB,(PetscScalar*)vB));
      CHKERRQ(PetscArraycmp(idxA,idxB,nzA,&flg));
      PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error MatGetRow %" PetscInt_FMT " (indices)",i);
      CHKERRQ(PetscArraycmp(vA,vB,nzA,&flg));
      PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error MatGetRow %" PetscInt_FMT " (values)",i);
      CHKERRQ(MatRestoreRow(A,i,&nzA,&idxA,&vA));
      CHKERRQ(MatRestoreRow(B,i,&nzB,&idxB,&vB));
    }

    CHKERRQ(MatGetOwnershipRange(A,&rstart,&rend));
    CHKERRQ(PetscCalloc3((rend-rstart)*6,&valuesA,(rend-rstart)*6,&valuesB,rend-rstart,&rows));
    for (i=rstart; i<rend; i++) rows[i-rstart] =i;

    CHKERRQ(MatGetValues(A,rend-rstart,rows,6,cols,valuesA));
    CHKERRQ(MatGetValues(B,rend-rstart,rows,6,cols,valuesB));

    for (i=0; i<(rend-rstart); i++) {
      CHKERRQ(PetscArraycmp(valuesA + 6*i,valuesB + 6*i,6,&flg));
      PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error MatGetValues %" PetscInt_FMT,i + rstart);
    }
    CHKERRQ(PetscFree3(valuesA,valuesB,rows));
  }

  /* Compare A and B */
  CHKERRQ(MatConvert(B,MATAIJ,MAT_INITIAL_MATRIX,&C));
  CHKERRQ(MatAXPY(C,-1.,A,SAME_NONZERO_PATTERN));
  CHKERRQ(MatNorm(C,NORM_INFINITY,&err));
  PetscCheckFalse(err > PETSC_SMALL,PetscObjectComm((PetscObject)B),PETSC_ERR_PLIB,"Error MatSetValues with INSERT_VALUES %g",err);

  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&B));
  CHKERRQ(MatDestroy(&C));

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
      requires: hypre

   test:
      suffix: 1
      requires: !defined(PETSC_HAVE_HYPRE_DEVICE)

   test:
      suffix: 2
      requires: !defined(PETSC_HAVE_HYPRE_DEVICE)
      output_file: output/ex225_1.out
      nsize: 2

TEST*/
