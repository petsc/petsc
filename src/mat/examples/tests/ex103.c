static char help[] = "Test MatConvert() from MATDENSE to MATELEMENTAL. Modified from the code contributed by Yaning Liu @lbl.gov \n\n";
    
#include <petscmat.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char** argv)
{
  Mat            mat_dense,mat_elemental;
  PetscInt       i,j,nrow = 10,ncol = 5;
  PetscErrorCode ierr;
  PetscScalar    sum;

  PetscInitialize(&argc, &argv, (char*)0, help);
  ierr = MatCreate(PETSC_COMM_WORLD, &mat_dense);CHKERRQ(ierr);
  ierr = MatSetType(mat_dense, MATDENSE);CHKERRQ(ierr);
  ierr = MatSetSizes(mat_dense, PETSC_DECIDE, PETSC_DECIDE, nrow, ncol);CHKERRQ(ierr);
  ierr = MatSetUp(mat_dense);CHKERRQ(ierr);
  for (i=0; i<nrow; i++) {
    for (j=0; j<ncol; j++) {
      sum  = i+j;
      ierr = MatSetValues(mat_dense, 1, &i, 1, &j, &sum, INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(mat_dense, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat_dense, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  //ierr = MatView(mat_dense, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

#if defined(PETSC_HAVE_ELEMENTAL)
  ierr = MatConvert(mat_dense, MATELEMENTAL, MAT_INITIAL_MATRIX, &mat_elemental);CHKERRQ(ierr);
  ierr = MatDestroy(&mat_elemental);CHKERRQ(ierr); 
#endif    
  ierr = MatDestroy(&mat_dense);CHKERRQ(ierr);
  PetscFinalize();
  return 0;
}
