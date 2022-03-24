
static char help[] = "Creates a matrix, inserts some values, and tests MatCreateSubMatrices() and MatZeroEntries().\n\n";

#include <petscmat.h>

int main(int argc,char **argv)
{
  Mat            mat,submat,submat1,*submatrices;
  PetscInt       m = 10,n = 10,i = 4,tmp,rstart,rend;
  IS             irow,icol;
  PetscScalar    value = 1.0;
  PetscViewer    sviewer;
  PetscBool      allA = PETSC_FALSE;

  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));
  CHKERRQ(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_COMMON));
  CHKERRQ(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_SELF,PETSC_VIEWER_ASCII_COMMON));

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&mat));
  CHKERRQ(MatSetSizes(mat,PETSC_DECIDE,PETSC_DECIDE,m,n));
  CHKERRQ(MatSetFromOptions(mat));
  CHKERRQ(MatSetUp(mat));
  CHKERRQ(MatGetOwnershipRange(mat,&rstart,&rend));
  for (i=rstart; i<rend; i++) {
    value = (PetscReal)i+1; tmp = i % 5;
    CHKERRQ(MatSetValues(mat,1,&tmp,1,&i,&value,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY));
  CHKERRQ(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"Original matrix\n"));
  CHKERRQ(MatView(mat,PETSC_VIEWER_STDOUT_WORLD));

  /* Test MatCreateSubMatrix_XXX_All(), i.e., submatrix = A */
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-test_all",&allA,NULL));
  if (allA) {
    CHKERRQ(ISCreateStride(PETSC_COMM_SELF,m,0,1,&irow));
    CHKERRQ(ISCreateStride(PETSC_COMM_SELF,n,0,1,&icol));
    CHKERRQ(MatCreateSubMatrices(mat,1,&irow,&icol,MAT_INITIAL_MATRIX,&submatrices));
    CHKERRQ(MatCreateSubMatrices(mat,1,&irow,&icol,MAT_REUSE_MATRIX,&submatrices));
    submat = *submatrices;

    /* sviewer will cause the submatrices (one per processor) to be printed in the correct order */
    CHKERRQ(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"\nSubmatrices with all\n"));
    CHKERRQ(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"--------------------\n"));
    CHKERRQ(PetscViewerGetSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&sviewer));
    CHKERRQ(MatView(submat,sviewer));
    CHKERRQ(PetscViewerRestoreSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&sviewer));
    CHKERRQ(PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD));

    CHKERRQ(ISDestroy(&irow));
    CHKERRQ(ISDestroy(&icol));

    /* test getting a reference on a submat */
    CHKERRQ(PetscObjectReference((PetscObject)submat));
    CHKERRQ(MatDestroySubMatrices(1,&submatrices));
    CHKERRQ(MatDestroy(&submat));
  }

  /* Form submatrix with rows 2-4 and columns 4-8 */
  CHKERRQ(ISCreateStride(PETSC_COMM_SELF,3,2,1,&irow));
  CHKERRQ(ISCreateStride(PETSC_COMM_SELF,5,4,1,&icol));
  CHKERRQ(MatCreateSubMatrices(mat,1,&irow,&icol,MAT_INITIAL_MATRIX,&submatrices));
  submat = *submatrices;

  /* Test reuse submatrices */
  CHKERRQ(MatCreateSubMatrices(mat,1,&irow,&icol,MAT_REUSE_MATRIX,&submatrices));

  /* sviewer will cause the submatrices (one per processor) to be printed in the correct order */
  CHKERRQ(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"\nSubmatrices\n"));
  CHKERRQ(PetscViewerGetSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&sviewer));
  CHKERRQ(MatView(submat,sviewer));
  CHKERRQ(PetscViewerRestoreSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&sviewer));
  CHKERRQ(PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(PetscObjectReference((PetscObject)submat));
  CHKERRQ(MatDestroySubMatrices(1,&submatrices));
  CHKERRQ(MatDestroy(&submat));

  /* Form submatrix with rows 2-4 and all columns */
  CHKERRQ(ISDestroy(&icol));
  CHKERRQ(ISCreateStride(PETSC_COMM_SELF,10,0,1,&icol));
  CHKERRQ(MatCreateSubMatrices(mat,1,&irow,&icol,MAT_INITIAL_MATRIX,&submatrices));
  CHKERRQ(MatCreateSubMatrices(mat,1,&irow,&icol,MAT_REUSE_MATRIX,&submatrices));
  submat = *submatrices;

  CHKERRQ(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"\nSubmatrices with allcolumns\n"));
  CHKERRQ(PetscViewerGetSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&sviewer));
  CHKERRQ(MatView(submat,sviewer));
  CHKERRQ(PetscViewerRestoreSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&sviewer));
  CHKERRQ(PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD));

  /* Test MatDuplicate */
  CHKERRQ(MatDuplicate(submat,MAT_COPY_VALUES,&submat1));
  CHKERRQ(MatDestroy(&submat1));

  /* Zero the original matrix */
  CHKERRQ(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"Original zeroed matrix\n"));
  CHKERRQ(MatZeroEntries(mat));
  CHKERRQ(MatView(mat,PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(ISDestroy(&irow));
  CHKERRQ(ISDestroy(&icol));
  CHKERRQ(PetscObjectReference((PetscObject)submat));
  CHKERRQ(MatDestroySubMatrices(1,&submatrices));
  CHKERRQ(MatDestroy(&submat));
  CHKERRQ(MatDestroy(&mat));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:
      args: -mat_type aij

   test:
      suffix: 2
      args: -mat_type dense

   test:
      suffix: 3
      nsize: 3
      args: -mat_type aij

   test:
      suffix: 4
      nsize: 3
      args: -mat_type dense

   test:
      suffix: 5
      nsize: 3
      args: -mat_type aij -test_all

TEST*/
