
static char help[] = "Tests MatMult() on MatLoad() matrix \n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            A;
  Vec            x,b;
  PetscErrorCode ierr;
  PetscViewer    fd;              /* viewer */
  char           file[PETSC_MAX_PATH_LEN]; /* input file name */
  PetscBool      flg;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  /* Determine file from which we read the matrix A */
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),&flg));
  PetscCheckFalse(!flg,PETSC_COMM_WORLD,PETSC_ERR_USER,"Must indicate binary file with the -f option");

  /* Load matrix A */
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatLoad(A,fd));
  flg  = PETSC_FALSE;
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&x));
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-vec",file,sizeof(file),&flg));
  if (flg) {
    if (file[0] == '0') {
      PetscInt    m;
      PetscScalar one = 1.0;
      CHKERRQ(PetscInfo(0,"Using vector of ones for RHS\n"));
      CHKERRQ(MatGetLocalSize(A,&m,NULL));
      CHKERRQ(VecSetSizes(x,m,PETSC_DECIDE));
      CHKERRQ(VecSetFromOptions(x));
      CHKERRQ(VecSet(x,one));
    }
  } else {
    CHKERRQ(VecLoad(x,fd));
    CHKERRQ(PetscViewerDestroy(&fd));
  }
  CHKERRQ(VecDuplicate(x,&b));
  CHKERRQ(MatMult(A,x,b));

  /* Print (for testing only) */
  CHKERRQ(MatView(A,0));
  CHKERRQ(VecView(b,0));
  /* Free data structures */
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&b));
  ierr = PetscFinalize();
  return ierr;
}
