
static char help[] = "Creates a matrix, inserts some values, and tests MatCreateSubMatrices() and MatZeroEntries().\n\n";

#include <petscmat.h>

int main(int argc,char **argv)
{
  Mat            mat,submat,submat1,*submatrices;
  PetscInt       m = 10,n = 10,i = 4,tmp,rstart,rend;
  PetscErrorCode ierr;
  IS             irow,icol;
  PetscScalar    value = 1.0;
  PetscViewer    sviewer;
  PetscBool      allA = PETSC_FALSE;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_COMMON);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_SELF,PETSC_VIEWER_ASCII_COMMON);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&mat);CHKERRQ(ierr);
  ierr = MatSetSizes(mat,PETSC_DECIDE,PETSC_DECIDE,m,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(mat);CHKERRQ(ierr);
  ierr = MatSetUp(mat);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(mat,&rstart,&rend);CHKERRQ(ierr);
  for (i=rstart; i<rend; i++) {
    value = (PetscReal)i+1; tmp = i % 5;
    ierr  = MatSetValues(mat,1,&tmp,1,&i,&value,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"Original matrix\n");CHKERRQ(ierr);
  ierr = MatView(mat,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* Test MatCreateSubMatrix_XXX_All(), i.e., submatrix = A */
  ierr = PetscOptionsGetBool(NULL,NULL,"-test_all",&allA,NULL);CHKERRQ(ierr);
  if (allA) {
    ierr   = ISCreateStride(PETSC_COMM_SELF,m,0,1,&irow);CHKERRQ(ierr);
    ierr   = ISCreateStride(PETSC_COMM_SELF,n,0,1,&icol);CHKERRQ(ierr);
    ierr   = MatCreateSubMatrices(mat,1,&irow,&icol,MAT_INITIAL_MATRIX,&submatrices);CHKERRQ(ierr);
    ierr   = MatCreateSubMatrices(mat,1,&irow,&icol,MAT_REUSE_MATRIX,&submatrices);CHKERRQ(ierr);
    submat = *submatrices;

    /* sviewer will cause the submatrices (one per processor) to be printed in the correct order */
    ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"\nSubmatrices with all\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"--------------------\n");CHKERRQ(ierr);
    ierr = PetscViewerGetSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&sviewer);CHKERRQ(ierr);
    ierr = MatView(submat,sviewer);CHKERRQ(ierr);
    ierr = PetscViewerRestoreSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&sviewer);CHKERRQ(ierr);
    ierr = PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

    ierr = ISDestroy(&irow);CHKERRQ(ierr);
    ierr = ISDestroy(&icol);CHKERRQ(ierr);

    /* test getting a reference on a submat */
    ierr = PetscObjectReference((PetscObject)submat);CHKERRQ(ierr);
    ierr = MatDestroySubMatrices(1,&submatrices);CHKERRQ(ierr);
    ierr = MatDestroy(&submat);CHKERRQ(ierr);
  }

  /* Form submatrix with rows 2-4 and columns 4-8 */
  ierr   = ISCreateStride(PETSC_COMM_SELF,3,2,1,&irow);CHKERRQ(ierr);
  ierr   = ISCreateStride(PETSC_COMM_SELF,5,4,1,&icol);CHKERRQ(ierr);
  ierr   = MatCreateSubMatrices(mat,1,&irow,&icol,MAT_INITIAL_MATRIX,&submatrices);CHKERRQ(ierr);
  submat = *submatrices;

  /* Test reuse submatrices */
  ierr = MatCreateSubMatrices(mat,1,&irow,&icol,MAT_REUSE_MATRIX,&submatrices);CHKERRQ(ierr);

  /* sviewer will cause the submatrices (one per processor) to be printed in the correct order */
  ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"\nSubmatrices\n");CHKERRQ(ierr);
  ierr = PetscViewerGetSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&sviewer);CHKERRQ(ierr);
  ierr = MatView(submat,sviewer);CHKERRQ(ierr);
  ierr = PetscViewerRestoreSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&sviewer);CHKERRQ(ierr);
  ierr = PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)submat);CHKERRQ(ierr);
  ierr = MatDestroySubMatrices(1,&submatrices);CHKERRQ(ierr);
  ierr = MatDestroy(&submat);CHKERRQ(ierr);

  /* Form submatrix with rows 2-4 and all columns */
  ierr   = ISDestroy(&icol);CHKERRQ(ierr);
  ierr   = ISCreateStride(PETSC_COMM_SELF,10,0,1,&icol);CHKERRQ(ierr);
  ierr   = MatCreateSubMatrices(mat,1,&irow,&icol,MAT_INITIAL_MATRIX,&submatrices);CHKERRQ(ierr);
  ierr   = MatCreateSubMatrices(mat,1,&irow,&icol,MAT_REUSE_MATRIX,&submatrices);CHKERRQ(ierr);
  submat = *submatrices;

  ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"\nSubmatrices with allcolumns\n");CHKERRQ(ierr);
  ierr = PetscViewerGetSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&sviewer);CHKERRQ(ierr);
  ierr = MatView(submat,sviewer);CHKERRQ(ierr);
  ierr = PetscViewerRestoreSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&sviewer);CHKERRQ(ierr);
  ierr = PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* Test MatDuplicate */
  ierr = MatDuplicate(submat,MAT_COPY_VALUES,&submat1);CHKERRQ(ierr);
  ierr = MatDestroy(&submat1);CHKERRQ(ierr);

  /* Zero the original matrix */
  ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"Original zeroed matrix\n");CHKERRQ(ierr);
  ierr = MatZeroEntries(mat);CHKERRQ(ierr);
  ierr = MatView(mat,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = ISDestroy(&irow);CHKERRQ(ierr);
  ierr = ISDestroy(&icol);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)submat);CHKERRQ(ierr);
  ierr = MatDestroySubMatrices(1,&submatrices);CHKERRQ(ierr);
  ierr = MatDestroy(&submat);CHKERRQ(ierr);
  ierr = MatDestroy(&mat);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
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
