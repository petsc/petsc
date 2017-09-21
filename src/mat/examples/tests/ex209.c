static char help[] = "Test MatTransposeMatMult() \n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            A,C,AtA,At;
  PetscViewer    fd;
  char           file[PETSC_MAX_PATH_LEN]="ceres_solver_iteration_001_A.petsc";
  PetscErrorCode ierr;
  PetscReal      fill = 4.0;
  PetscMPIInt    rank;
  PetscBool      equal;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-f",file,PETSC_MAX_PATH_LEN,NULL);CHKERRQ(ierr);

  /* Load matrices */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetType(A,MATAIJ);CHKERRQ(ierr);
  ierr = MatLoad(A,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);
  if (!rank) printf("A is loaded...\n");

  ierr = MatTransposeMatMult(A,A,MAT_INITIAL_MATRIX,fill,&C);CHKERRQ(ierr); /* C = A^T*A */
  if (!rank) printf("C = A^T*A is done...\n");

  ierr = MatTranspose(A,MAT_INITIAL_MATRIX,&At);CHKERRQ(ierr);
  if (!rank) printf("At is done...\n");
  ierr = MatMatMult(At,A,MAT_INITIAL_MATRIX,fill,&AtA);CHKERRQ(ierr);
  if (!rank) printf("AtA is done...\n");

  ierr = MatEqual(C,AtA,&equal);CHKERRQ(ierr);
  if (!equal) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"A^T*A != At*A");

  ierr = MatDestroy(&At);CHKERRQ(ierr);
  ierr = MatDestroy(&AtA);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
