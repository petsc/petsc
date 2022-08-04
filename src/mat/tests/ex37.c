
static char help[] = "Tests MatCopy() and MatStore/RetrieveValues().\n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            C,A;
  PetscInt       i, n = 10,midx[3],bs=1;
  PetscScalar    v[3];
  PetscBool      flg,isAIJ;
  MatType        type;
  PetscMPIInt    size;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-mat_block_size",&bs,NULL));

  PetscCall(MatCreate(PETSC_COMM_WORLD,&C));
  PetscCall(MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,n,n));
  PetscCall(MatSetType(C,MATAIJ));
  PetscCall(MatSetFromOptions(C));
  PetscCall(PetscObjectSetName((PetscObject)C,"initial"));

  PetscCall(MatGetType(C,&type));
  if (size == 1) {
    PetscCall(PetscObjectTypeCompare((PetscObject)C,MATSEQAIJ,&isAIJ));
  } else {
    PetscCall(PetscObjectTypeCompare((PetscObject)C,MATMPIAIJ,&isAIJ));
  }
  PetscCall(MatSeqAIJSetPreallocation(C,3,NULL));
  PetscCall(MatMPIAIJSetPreallocation(C,3,NULL,3,NULL));
  PetscCall(MatSeqBAIJSetPreallocation(C,bs,3,NULL));
  PetscCall(MatMPIBAIJSetPreallocation(C,bs,3,NULL,3,NULL));
  PetscCall(MatSeqSBAIJSetPreallocation(C,bs,3,NULL));
  PetscCall(MatMPISBAIJSetPreallocation(C,bs,3,NULL,3,NULL));

  v[0] = -1.; v[1] = 2.; v[2] = -1.;
  for (i=1; i<n-1; i++) {
    midx[2] = i-1; midx[1] = i; midx[0] = i+1;
    PetscCall(MatSetValues(C,1,&i,3,midx,v,INSERT_VALUES));
  }
  i    = 0; midx[0] = 0; midx[1] = 1;
  v[0] = 2.0; v[1] = -1.;
  PetscCall(MatSetValues(C,1,&i,2,midx,v,INSERT_VALUES));
  i    = n-1; midx[0] = n-2; midx[1] = n-1;
  v[0] = -1.0; v[1] = 2.;
  PetscCall(MatSetValues(C,1,&i,2,midx,v,INSERT_VALUES));

  PetscCall(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));
  PetscCall(MatView(C,NULL));
  PetscCall(MatViewFromOptions(C,NULL,"-view"));

  /* test matduplicate */
  PetscCall(MatDuplicate(C,MAT_COPY_VALUES,&A));
  PetscCall(PetscObjectSetName((PetscObject)A,"duplicate_copy"));
  PetscCall(MatViewFromOptions(A,NULL,"-view"));
  PetscCall(MatEqual(A,C,&flg));
  PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_SUP,"MatDuplicate(C,MAT_COPY_VALUES,): Matrices are NOT equal");
  PetscCall(MatDestroy(&A));

  /* test matrices with different nonzero patterns - Note: A is created with different nonzero pattern of C! */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  PetscCall(MatCopy(C,A,DIFFERENT_NONZERO_PATTERN));
  PetscCall(PetscObjectSetName((PetscObject)A,"copy_diffnnz"));
  PetscCall(MatViewFromOptions(A,NULL,"-view"));
  PetscCall(MatEqual(A,C,&flg));
  PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_SUP,"MatCopy(C,A,DIFFERENT_NONZERO_PATTERN): Matrices are NOT equal");

  /* test matrices with same nonzero pattern */
  PetscCall(MatDestroy(&A));
  PetscCall(MatDuplicate(C,MAT_DO_NOT_COPY_VALUES,&A));
  PetscCall(MatCopy(C,A,SAME_NONZERO_PATTERN));
  PetscCall(PetscObjectSetName((PetscObject)A,"copy_samennz"));
  PetscCall(MatViewFromOptions(A,NULL,"-view"));
  PetscCall(MatEqual(A,C,&flg));
  PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_SUP,"MatCopy(C,A,SAME_NONZERO_PATTERN): Matrices are NOT equal");

  /* test subset nonzero pattern */
  PetscCall(MatCopy(C,A,SUBSET_NONZERO_PATTERN));
  PetscCall(PetscObjectSetName((PetscObject)A,"copy_subnnz"));
  PetscCall(MatViewFromOptions(A,NULL,"-view"));
  PetscCall(MatEqual(A,C,&flg));
  PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_SUP,"MatCopy(C,A,SUBSET_NONZERO_PATTERN): Matrices are NOT equal");

  /* Test MatCopy on a matrix obtained after MatConvert from AIJ
     see https://lists.mcs.anl.gov/pipermail/petsc-dev/2019-April/024289.html */
  PetscCall(MatHasCongruentLayouts(C,&flg));
  if (flg) {
    Mat     Cs,Cse;
    MatType Ctype,Cstype;

    PetscCall(MatGetType(C,&Ctype));
    PetscCall(MatTranspose(C,MAT_INITIAL_MATRIX,&Cs));
    PetscCall(MatAXPY(Cs,1.0,C,DIFFERENT_NONZERO_PATTERN));
    PetscCall(MatConvert(Cs,MATAIJ,MAT_INPLACE_MATRIX,&Cs));
    PetscCall(MatSetOption(Cs,MAT_SYMMETRIC,PETSC_TRUE));
    PetscCall(MatGetType(Cs,&Cstype));

    PetscCall(PetscObjectSetName((PetscObject)Cs,"symm_initial"));
    PetscCall(MatViewFromOptions(Cs,NULL,"-view"));

    PetscCall(MatConvert(Cs,Ctype,MAT_INITIAL_MATRIX,&Cse));
    PetscCall(PetscObjectSetName((PetscObject)Cse,"symm_conv_init"));
    PetscCall(MatViewFromOptions(Cse,NULL,"-view"));
    PetscCall(MatMultEqual(Cs,Cse,5,&flg));
    PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatConvert MAT_INITIAL_MATRIX %s -> %s: Matrices are NOT multequal",Ctype,Cstype);

    PetscCall(MatConvert(Cs,Ctype,MAT_REUSE_MATRIX,&Cse));
    PetscCall(PetscObjectSetName((PetscObject)Cse,"symm_conv_reuse"));
    PetscCall(MatViewFromOptions(Cse,NULL,"-view"));
    PetscCall(MatMultEqual(Cs,Cse,5,&flg));
    PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatConvert MAT_REUSE_MATRIX %s -> %s: Matrices are NOT multequal",Ctype,Cstype);

    PetscCall(MatCopy(Cs,Cse,SAME_NONZERO_PATTERN));
    PetscCall(PetscObjectSetName((PetscObject)Cse,"symm_conv_copy_samennz"));
    PetscCall(MatViewFromOptions(Cse,NULL,"-view"));
    PetscCall(MatMultEqual(Cs,Cse,5,&flg));
    PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatCopy(...SAME_NONZERO_PATTERN) %s -> %s: Matrices are NOT multequal",Ctype,Cstype);

    PetscCall(MatCopy(Cs,Cse,SUBSET_NONZERO_PATTERN));
    PetscCall(PetscObjectSetName((PetscObject)Cse,"symm_conv_copy_subnnz"));
    PetscCall(MatViewFromOptions(Cse,NULL,"-view"));
    PetscCall(MatMultEqual(Cs,Cse,5,&flg));
    PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatCopy(...SUBSET_NONZERO_PATTERN) %s -> %s: Matrices are NOT multequal",Ctype,Cstype);

    PetscCall(MatCopy(Cs,Cse,DIFFERENT_NONZERO_PATTERN));
    PetscCall(PetscObjectSetName((PetscObject)Cse,"symm_conv_copy_diffnnz"));
    PetscCall(MatViewFromOptions(Cse,NULL,"-view"));
    PetscCall(MatMultEqual(Cs,Cse,5,&flg));
    PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatCopy(...DIFFERENT_NONZERO_PATTERN) %s -> %s: Matrices are NOT multequal",Ctype,Cstype);

    PetscCall(MatDestroy(&Cse));
    PetscCall(MatDestroy(&Cs));
  }

  /* test MatStore/RetrieveValues() */
  if (isAIJ) {
    PetscCall(MatSetOption(A,MAT_NEW_NONZERO_LOCATIONS,PETSC_FALSE));
    PetscCall(MatStoreValues(A));
    PetscCall(MatZeroEntries(A));
    PetscCall(MatRetrieveValues(A));
  }

  PetscCall(MatDestroy(&C));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   testset:
      nsize: {{1 2}separate output}
      args: -view ::ascii_info -mat_type {{aij baij sbaij mpiaij mpibaij mpisbaij}separate output} -mat_block_size {{1 2}separate output}

TEST*/
