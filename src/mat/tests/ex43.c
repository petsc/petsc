
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
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));

  /* PART 1:  Generate matrix, then write it in binary format */

  /* Generate matrix */
  CHKERRQ(MatCreateSeqDense(PETSC_COMM_WORLD,m,n,NULL,&C));
  for (i=0; i<m; i++) {
    for (j=0; j<n; j++) {
      v    = i*m+j;
      CHKERRQ(MatSetValues(C,1,&i,1,&j,&v,INSERT_VALUES));
    }
  }
  CHKERRQ(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,"matrix.dat",FILE_MODE_WRITE,&viewer));
  CHKERRQ(MatView(C,viewer));
  CHKERRQ(PetscViewerDestroy(&viewer));
  CHKERRQ(MatDestroy(&C));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:

TEST*/
