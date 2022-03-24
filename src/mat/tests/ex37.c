
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

  CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-mat_block_size",&bs,NULL));

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&C));
  CHKERRQ(MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,n,n));
  CHKERRQ(MatSetType(C,MATAIJ));
  CHKERRQ(MatSetFromOptions(C));
  CHKERRQ(PetscObjectSetName((PetscObject)C,"initial"));

  CHKERRQ(MatGetType(C,&type));
  if (size == 1) {
    CHKERRQ(PetscObjectTypeCompare((PetscObject)C,MATSEQAIJ,&isAIJ));
  } else {
    CHKERRQ(PetscObjectTypeCompare((PetscObject)C,MATMPIAIJ,&isAIJ));
  }
  CHKERRQ(MatSeqAIJSetPreallocation(C,3,NULL));
  CHKERRQ(MatMPIAIJSetPreallocation(C,3,NULL,3,NULL));
  CHKERRQ(MatSeqBAIJSetPreallocation(C,bs,3,NULL));
  CHKERRQ(MatMPIBAIJSetPreallocation(C,bs,3,NULL,3,NULL));
  CHKERRQ(MatSeqSBAIJSetPreallocation(C,bs,3,NULL));
  CHKERRQ(MatMPISBAIJSetPreallocation(C,bs,3,NULL,3,NULL));

  v[0] = -1.; v[1] = 2.; v[2] = -1.;
  for (i=1; i<n-1; i++) {
    midx[2] = i-1; midx[1] = i; midx[0] = i+1;
    CHKERRQ(MatSetValues(C,1,&i,3,midx,v,INSERT_VALUES));
  }
  i    = 0; midx[0] = 0; midx[1] = 1;
  v[0] = 2.0; v[1] = -1.;
  CHKERRQ(MatSetValues(C,1,&i,2,midx,v,INSERT_VALUES));
  i    = n-1; midx[0] = n-2; midx[1] = n-1;
  v[0] = -1.0; v[1] = 2.;
  CHKERRQ(MatSetValues(C,1,&i,2,midx,v,INSERT_VALUES));

  CHKERRQ(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatView(C,NULL));
  CHKERRQ(MatViewFromOptions(C,NULL,"-view"));

  /* test matduplicate */
  CHKERRQ(MatDuplicate(C,MAT_COPY_VALUES,&A));
  CHKERRQ(PetscObjectSetName((PetscObject)A,"duplicate_copy"));
  CHKERRQ(MatViewFromOptions(A,NULL,"-view"));
  CHKERRQ(MatEqual(A,C,&flg));
  PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_SUP,"MatDuplicate(C,MAT_COPY_VALUES,): Matrices are NOT equal");
  CHKERRQ(MatDestroy(&A));

  /* test matrices with different nonzero patterns - Note: A is created with different nonzero pattern of C! */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  CHKERRQ(MatCopy(C,A,DIFFERENT_NONZERO_PATTERN));
  CHKERRQ(PetscObjectSetName((PetscObject)A,"copy_diffnnz"));
  CHKERRQ(MatViewFromOptions(A,NULL,"-view"));
  CHKERRQ(MatEqual(A,C,&flg));
  PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_SUP,"MatCopy(C,A,DIFFERENT_NONZERO_PATTERN): Matrices are NOT equal");

  /* test matrices with same nonzero pattern */
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDuplicate(C,MAT_DO_NOT_COPY_VALUES,&A));
  CHKERRQ(MatCopy(C,A,SAME_NONZERO_PATTERN));
  CHKERRQ(PetscObjectSetName((PetscObject)A,"copy_samennz"));
  CHKERRQ(MatViewFromOptions(A,NULL,"-view"));
  CHKERRQ(MatEqual(A,C,&flg));
  PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_SUP,"MatCopy(C,A,SAME_NONZERO_PATTERN): Matrices are NOT equal");

  /* test subset nonzero pattern */
  CHKERRQ(MatCopy(C,A,SUBSET_NONZERO_PATTERN));
  CHKERRQ(PetscObjectSetName((PetscObject)A,"copy_subnnz"));
  CHKERRQ(MatViewFromOptions(A,NULL,"-view"));
  CHKERRQ(MatEqual(A,C,&flg));
  PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_SUP,"MatCopy(C,A,SUBSET_NONZERO_PATTERN): Matrices are NOT equal");

  /* Test MatCopy on a matrix obtained after MatConvert from AIJ
     see https://lists.mcs.anl.gov/pipermail/petsc-dev/2019-April/024289.html */
  CHKERRQ(MatHasCongruentLayouts(C,&flg));
  if (flg) {
    Mat     Cs,Cse;
    MatType Ctype,Cstype;

    CHKERRQ(MatGetType(C,&Ctype));
    CHKERRQ(MatTranspose(C,MAT_INITIAL_MATRIX,&Cs));
    CHKERRQ(MatAXPY(Cs,1.0,C,DIFFERENT_NONZERO_PATTERN));
    CHKERRQ(MatConvert(Cs,MATAIJ,MAT_INPLACE_MATRIX,&Cs));
    CHKERRQ(MatSetOption(Cs,MAT_SYMMETRIC,PETSC_TRUE));
    CHKERRQ(MatGetType(Cs,&Cstype));

    CHKERRQ(PetscObjectSetName((PetscObject)Cs,"symm_initial"));
    CHKERRQ(MatViewFromOptions(Cs,NULL,"-view"));

    CHKERRQ(MatConvert(Cs,Ctype,MAT_INITIAL_MATRIX,&Cse));
    CHKERRQ(PetscObjectSetName((PetscObject)Cse,"symm_conv_init"));
    CHKERRQ(MatViewFromOptions(Cse,NULL,"-view"));
    CHKERRQ(MatMultEqual(Cs,Cse,5,&flg));
    PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatConvert MAT_INITIAL_MATRIX %s -> %s: Matrices are NOT multequal",Ctype,Cstype);

    CHKERRQ(MatConvert(Cs,Ctype,MAT_REUSE_MATRIX,&Cse));
    CHKERRQ(PetscObjectSetName((PetscObject)Cse,"symm_conv_reuse"));
    CHKERRQ(MatViewFromOptions(Cse,NULL,"-view"));
    CHKERRQ(MatMultEqual(Cs,Cse,5,&flg));
    PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatConvert MAT_REUSE_MATRIX %s -> %s: Matrices are NOT multequal",Ctype,Cstype);

    CHKERRQ(MatCopy(Cs,Cse,SAME_NONZERO_PATTERN));
    CHKERRQ(PetscObjectSetName((PetscObject)Cse,"symm_conv_copy_samennz"));
    CHKERRQ(MatViewFromOptions(Cse,NULL,"-view"));
    CHKERRQ(MatMultEqual(Cs,Cse,5,&flg));
    PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatCopy(...SAME_NONZERO_PATTERN) %s -> %s: Matrices are NOT multequal",Ctype,Cstype);

    CHKERRQ(MatCopy(Cs,Cse,SUBSET_NONZERO_PATTERN));
    CHKERRQ(PetscObjectSetName((PetscObject)Cse,"symm_conv_copy_subnnz"));
    CHKERRQ(MatViewFromOptions(Cse,NULL,"-view"));
    CHKERRQ(MatMultEqual(Cs,Cse,5,&flg));
    PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatCopy(...SUBSET_NONZERO_PATTERN) %s -> %s: Matrices are NOT multequal",Ctype,Cstype);

    CHKERRQ(MatCopy(Cs,Cse,DIFFERENT_NONZERO_PATTERN));
    CHKERRQ(PetscObjectSetName((PetscObject)Cse,"symm_conv_copy_diffnnz"));
    CHKERRQ(MatViewFromOptions(Cse,NULL,"-view"));
    CHKERRQ(MatMultEqual(Cs,Cse,5,&flg));
    PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatCopy(...DIFFERENT_NONZERO_PATTERN) %s -> %s: Matrices are NOT multequal",Ctype,Cstype);

    CHKERRQ(MatDestroy(&Cse));
    CHKERRQ(MatDestroy(&Cs));
  }

  /* test MatStore/RetrieveValues() */
  if (isAIJ) {
    CHKERRQ(MatSetOption(A,MAT_NEW_NONZERO_LOCATIONS,PETSC_FALSE));
    CHKERRQ(MatStoreValues(A));
    CHKERRQ(MatZeroEntries(A));
    CHKERRQ(MatRetrieveValues(A));
  }

  CHKERRQ(MatDestroy(&C));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   testset:
      nsize: {{1 2}separate output}
      args: -view ::ascii_info -mat_type {{aij baij sbaij mpiaij mpibaij mpisbaij}separate output} -mat_block_size {{1 2}separate output}

TEST*/
