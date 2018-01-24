
static char help[] = "Saves a dense matrix in a dense format (binary).\n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            C;
  PetscScalar    v;
  PetscInt       i,j,m = 4,n = 4;
  PetscErrorCode ierr;
  PetscMPIInt    rank,size;
  PetscViewer    viewer;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);

  /* PART 1:  Generate matrix, then write it in binary format */

  /* Generate matrix */
  ierr = MatCreateSeqDense(PETSC_COMM_WORLD,m,n,NULL,&C);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    for (j=0; j<n; j++) {
      v    = i*m+j;
      ierr = MatSetValues(C,1,&i,1,&j,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"matrix.dat",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = MatView(C,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:

TEST*/
