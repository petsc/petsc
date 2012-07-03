 
static char help[] = "Tests MatTransposeMatMult() on MatLoad() matrix \n\n";

#include <petscmat.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat            A,C; 
  PetscErrorCode ierr;
  PetscViewer    fd;              /* viewer */
  char           file[PETSC_MAX_PATH_LEN]; /* input file name */
  PetscBool      flg;
  PetscMPIInt    rank;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  /* Determine file from which we read the matrix A */
  ierr = PetscOptionsGetString(PETSC_NULL,"-f",file,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,1,"Must indicate binary file with the -f option");

  /* Load matrix A */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatLoad(A,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr); 

  /* Print (for testing only) */
  if (!rank) printf("A:\n");
  ierr = MatView(A,0);CHKERRQ(ierr);

  /* Test MatTransposeMatMult() */
  PetscReal fill=1.0;
  ierr = MatTransposeMatMult(A,A,MAT_INITIAL_MATRIX,fill,&C);CHKERRQ(ierr);
  if (!rank) printf("\nC = A^T * A:\n");
  ierr = MatView(C,0);CHKERRQ(ierr);

  /* Free data structures */
  ierr = MatDestroy(&A);CHKERRQ(ierr); 
  ierr = MatDestroy(&C);CHKERRQ(ierr); 

  ierr = PetscFinalize();
  return 0;
}
