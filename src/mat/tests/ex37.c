
static char help[] = "Tests MatCopy() and MatStore/RetrieveValues().\n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            C,A;
  PetscInt       i, n = 10,midx[3],bs=1;
  PetscErrorCode ierr;
  PetscScalar    v[3];
  PetscBool      flg,isAIJ;
  MatType        type;
  PetscMPIInt    size;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-mat_block_size",&bs,NULL);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&C);CHKERRQ(ierr);
  ierr = MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
  ierr = MatSetType(C,MATAIJ);CHKERRQ(ierr);
  ierr = MatSetFromOptions(C);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)C,"initial");CHKERRQ(ierr);

  ierr = MatGetType(C,&type);CHKERRQ(ierr);
  if (size == 1) {
    ierr = PetscObjectTypeCompare((PetscObject)C,MATSEQAIJ,&isAIJ);CHKERRQ(ierr);
  } else {
    ierr = PetscObjectTypeCompare((PetscObject)C,MATMPIAIJ,&isAIJ);CHKERRQ(ierr);
  }
  ierr = MatSeqAIJSetPreallocation(C,3,NULL);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(C,3,NULL,3,NULL);CHKERRQ(ierr);
  ierr = MatSeqBAIJSetPreallocation(C,bs,3,NULL);CHKERRQ(ierr);
  ierr = MatMPIBAIJSetPreallocation(C,bs,3,NULL,3,NULL);CHKERRQ(ierr);
  ierr = MatSeqSBAIJSetPreallocation(C,bs,3,NULL);CHKERRQ(ierr);
  ierr = MatMPISBAIJSetPreallocation(C,bs,3,NULL,3,NULL);CHKERRQ(ierr);

  v[0] = -1.; v[1] = 2.; v[2] = -1.;
  for (i=1; i<n-1; i++) {
    midx[2] = i-1; midx[1] = i; midx[0] = i+1;
    ierr    = MatSetValues(C,1,&i,3,midx,v,INSERT_VALUES);CHKERRQ(ierr);
  }
  i    = 0; midx[0] = 0; midx[1] = 1;
  v[0] = 2.0; v[1] = -1.;
  ierr = MatSetValues(C,1,&i,2,midx,v,INSERT_VALUES);CHKERRQ(ierr);
  i    = n-1; midx[0] = n-2; midx[1] = n-1;
  v[0] = -1.0; v[1] = 2.;
  ierr = MatSetValues(C,1,&i,2,midx,v,INSERT_VALUES);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatView(C,NULL);CHKERRQ(ierr);
  ierr = MatViewFromOptions(C,NULL,"-view");CHKERRQ(ierr);

  /* test matduplicate */
  ierr = MatDuplicate(C,MAT_COPY_VALUES,&A);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)A,"duplicate_copy");CHKERRQ(ierr);
  ierr = MatViewFromOptions(A,NULL,"-view");CHKERRQ(ierr);
  ierr = MatEqual(A,C,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"MatDuplicate(C,MAT_COPY_VALUES,): Matrices are NOT equal");
  ierr = MatDestroy(&A);CHKERRQ(ierr);

  /* test matrices with different nonzero patterns - Note: A is created with different nonzero pattern of C! */
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatCopy(C,A,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)A,"copy_diffnnz");CHKERRQ(ierr);
  ierr = MatViewFromOptions(A,NULL,"-view");CHKERRQ(ierr);
  ierr = MatEqual(A,C,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"MatCopy(C,A,DIFFERENT_NONZERO_PATTERN): Matrices are NOT equal");

  /* test matrices with same nonzero pattern */
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDuplicate(C,MAT_DO_NOT_COPY_VALUES,&A);CHKERRQ(ierr);
  ierr = MatCopy(C,A,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)A,"copy_samennz");CHKERRQ(ierr);
  ierr = MatViewFromOptions(A,NULL,"-view");CHKERRQ(ierr);
  ierr = MatEqual(A,C,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"MatCopy(C,A,SAME_NONZERO_PATTERN): Matrices are NOT equal");

  /* test subset nonzero pattern */
  ierr = MatCopy(C,A,SUBSET_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)A,"copy_subnnz");CHKERRQ(ierr);
  ierr = MatViewFromOptions(A,NULL,"-view");CHKERRQ(ierr);
  ierr = MatEqual(A,C,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"MatCopy(C,A,SUBSET_NONZERO_PATTERN): Matrices are NOT equal");

  /* Test MatCopy on a matrix obtained after MatConvert from AIJ
     see https://lists.mcs.anl.gov/pipermail/petsc-dev/2019-April/024289.html */
  ierr = MatHasCongruentLayouts(C,&flg);CHKERRQ(ierr);
  if (flg) {
    Mat     Cs,Cse;
    MatType Ctype,Cstype;

    ierr = MatGetType(C,&Ctype);CHKERRQ(ierr);
    ierr = MatTranspose(C,MAT_INITIAL_MATRIX,&Cs);CHKERRQ(ierr);
    ierr = MatAXPY(Cs,1.0,C,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatConvert(Cs,MATAIJ,MAT_INPLACE_MATRIX,&Cs);CHKERRQ(ierr);
    ierr = MatSetOption(Cs,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);
    ierr = MatGetType(Cs,&Cstype);CHKERRQ(ierr);

    ierr = PetscObjectSetName((PetscObject)Cs,"symm_initial");CHKERRQ(ierr);
    ierr = MatViewFromOptions(Cs,NULL,"-view");CHKERRQ(ierr);

    ierr = MatConvert(Cs,Ctype,MAT_INITIAL_MATRIX,&Cse);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)Cse,"symm_conv_init");CHKERRQ(ierr);
    ierr = MatViewFromOptions(Cse,NULL,"-view");CHKERRQ(ierr);
    ierr = MatMultEqual(Cs,Cse,5,&flg);CHKERRQ(ierr);
    if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatConvert MAT_INITIAL_MATRIX %s -> %s: Matrices are NOT multequal",Ctype,Cstype);

    ierr = MatConvert(Cs,Ctype,MAT_REUSE_MATRIX,&Cse);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)Cse,"symm_conv_reuse");CHKERRQ(ierr);
    ierr = MatViewFromOptions(Cse,NULL,"-view");CHKERRQ(ierr);
    ierr = MatMultEqual(Cs,Cse,5,&flg);CHKERRQ(ierr);
    if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatConvert MAT_REUSE_MATRIX %s -> %s: Matrices are NOT multequal",Ctype,Cstype);

    ierr = MatCopy(Cs,Cse,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)Cse,"symm_conv_copy_samennz");CHKERRQ(ierr);
    ierr = MatViewFromOptions(Cse,NULL,"-view");CHKERRQ(ierr);
    ierr = MatMultEqual(Cs,Cse,5,&flg);CHKERRQ(ierr);
    if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatCopy(...SAME_NONZERO_PATTERN) %s -> %s: Matrices are NOT multequal",Ctype,Cstype);

    ierr = MatCopy(Cs,Cse,SUBSET_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)Cse,"symm_conv_copy_subnnz");CHKERRQ(ierr);
    ierr = MatViewFromOptions(Cse,NULL,"-view");CHKERRQ(ierr);
    ierr = MatMultEqual(Cs,Cse,5,&flg);CHKERRQ(ierr);
    if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatCopy(...SUBSET_NONZERO_PATTERN) %s -> %s: Matrices are NOT multequal",Ctype,Cstype);

    ierr = MatCopy(Cs,Cse,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)Cse,"symm_conv_copy_diffnnz");CHKERRQ(ierr);
    ierr = MatViewFromOptions(Cse,NULL,"-view");CHKERRQ(ierr);
    ierr = MatMultEqual(Cs,Cse,5,&flg);CHKERRQ(ierr);
    if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatCopy(...DIFFERENT_NONZERO_PATTERN) %s -> %s: Matrices are NOT multequal",Ctype,Cstype);

    ierr = MatDestroy(&Cse);CHKERRQ(ierr);
    ierr = MatDestroy(&Cs);CHKERRQ(ierr);
  }

  /* test MatStore/RetrieveValues() */
  if (isAIJ) {
    ierr = MatSetOption(A,MAT_NEW_NONZERO_LOCATIONS,PETSC_FALSE);CHKERRQ(ierr);
    ierr = MatStoreValues(A);CHKERRQ(ierr);
    ierr = MatZeroEntries(A);CHKERRQ(ierr);
    ierr = MatRetrieveValues(A);CHKERRQ(ierr);
  }

  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   testset:
      nsize: {{1 2}separate output}
      args: -view ::ascii_info -mat_type {{aij baij sbaij mpiaij mpibaij mpisbaij}separate output} -mat_block_size {{1 2}separate output}

TEST*/
