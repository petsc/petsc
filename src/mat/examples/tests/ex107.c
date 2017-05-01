
static char help[] = "Test MatCreate() with MAT_STRUCTURE_ONLY .\n\n";

#include <petscmat.h>

int main(int argc,char **argv)
{
  Mat            mat;
  PetscInt       m = 7,n,i,j,rstart,rend;
  PetscErrorCode ierr;
  PetscMPIInt    size,rank;
  PetscScalar    v;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_COMMON);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  n    = m;

  /* ------- Assemble matrix, test MatValid() --------- */
  ierr = MatCreate(PETSC_COMM_WORLD,&mat);CHKERRQ(ierr);
  ierr = MatSetSizes(mat,PETSC_DECIDE,PETSC_DECIDE,m,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(mat);CHKERRQ(ierr);
  ierr = MatSetOption(mat,MAT_STRUCTURE_ONLY,PETSC_TRUE);CHKERRQ(ierr);

  ierr = MatSetUp(mat);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(mat,&rstart,&rend);CHKERRQ(ierr);
  for (i=rstart; i<rend; i++) {
    for (j=0; j<n; j++) {
      v    = 10.0*i+j;
      ierr = MatSetValues(mat,1,&i,1,&j,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatView(mat,0);CHKERRQ(ierr);

  /* Free data structures */
  ierr = MatDestroy(&mat);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}



