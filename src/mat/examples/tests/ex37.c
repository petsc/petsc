
static char help[] = "Tests MatCopy() and MatStore/RetrieveValues().\n\n"; 

#include <petscmat.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat            C,A; 
  PetscInt       i, n = 10,midx[3],bs=1;
  PetscErrorCode ierr;
  PetscScalar    v[3];
  PetscBool      flg,isAIJ;
  const MatType  type;
  PetscMPIInt    size;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-mat_block_size",&bs,PETSC_NULL);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&C);CHKERRQ(ierr);
  ierr = MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
  ierr = MatSetType(C,MATAIJ);CHKERRQ(ierr);
  ierr = MatSetFromOptions(C);CHKERRQ(ierr);
  
  ierr = MatGetType(C,&type);CHKERRQ(ierr);
  if (size == 1){
    ierr = PetscTypeCompare((PetscObject)C,MATSEQAIJ,&isAIJ);CHKERRQ(ierr);
  } else {
    ierr = PetscTypeCompare((PetscObject)C,MATMPIAIJ,&isAIJ);CHKERRQ(ierr);
  }
  ierr = MatSeqAIJSetPreallocation(C,3,PETSC_NULL);
  ierr = MatMPIAIJSetPreallocation(C,3,PETSC_NULL,3,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatSeqBAIJSetPreallocation(C,bs,3,PETSC_NULL);
  ierr = MatMPIBAIJSetPreallocation(C,bs,3,PETSC_NULL,3,PETSC_NULL);CHKERRQ(ierr);

  v[0] = -1.; v[1] = 2.; v[2] = -1.;
  for (i=1; i<n-1; i++){
    midx[2] = i-1; midx[1] = i; midx[0] = i+1;
    ierr = MatSetValues(C,1,&i,3,midx,v,INSERT_VALUES);CHKERRQ(ierr);
  }
  i = 0; midx[0] = 0; midx[1] = 1;
  v[0] = 2.0; v[1] = -1.; 
  ierr = MatSetValues(C,1,&i,2,midx,v,INSERT_VALUES);CHKERRQ(ierr);
  i = n-1; midx[0] = n-2; midx[1] = n-1;
  v[0] = -1.0; v[1] = 2.; 
  ierr = MatSetValues(C,1,&i,2,midx,v,INSERT_VALUES);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* test matrices with different nonzero patterns - Note: A is created with different nonzero pattern of C! */
  ierr = MatCopy(C,A,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatEqual(A,C,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_SELF,1,"MatCopy(C,A,DIFFERENT_NONZERO_PATTERN): Matrices are NOT equal");
  
  ierr = PetscViewerSetFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"A is obtained with MatCopy(,,DIFFERENT_NONZERO_PATTERN):\n");CHKERRQ(ierr);
  ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);

  /* test matrices with same nonzero pattern */
  ierr = MatDuplicate(C,MAT_DO_NOT_COPY_VALUES,&A);CHKERRQ(ierr);
  ierr = MatCopy(C,A,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatEqual(A,C,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_SELF,1,"MatCopy(C,A,SAME_NONZERO_PATTERN): Matrices are NOT equal");
  
  ierr = PetscViewerSetFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nA is obtained with MatCopy(,,SAME_NONZERO_PATTERN):\n");CHKERRQ(ierr);
  ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = PetscViewerSetFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_COMMON);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"A:\n");CHKERRQ(ierr);
  ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  
  /* test MatStore/RetrieveValues() */
  if (isAIJ){
    ierr = MatSetOption(A,MAT_NEW_NONZERO_LOCATIONS,PETSC_FALSE);CHKERRQ(ierr);
    ierr = MatStoreValues(A);CHKERRQ(ierr);
    ierr = MatZeroEntries(A);CHKERRQ(ierr);
    ierr = MatRetrieveValues(A);CHKERRQ(ierr);
  }

  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}

 
