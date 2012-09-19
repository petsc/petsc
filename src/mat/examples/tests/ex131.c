
static char help[] = "Tests MatMult() on MatLoad() matrix \n\n";

#include <petscmat.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat            A;
  Vec            x,b;
  PetscErrorCode ierr;
  PetscViewer    fd;              /* viewer */
  char           file[PETSC_MAX_PATH_LEN]; /* input file name */
  PetscBool      flg;

  PetscInitialize(&argc,&args,(char *)0,help);

  /* Determine file from which we read the matrix A */
  ierr = PetscOptionsGetString(PETSC_NULL,"-f",file,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,1,"Must indicate binary file with the -f option");

  /* Load matrix A */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatLoad(A,fd);CHKERRQ(ierr);
  flg = PETSC_FALSE;
  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(PETSC_NULL,"-vec",file,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  if (flg){
    if (file[0] == '0'){
      PetscInt    m;
      PetscScalar one = 1.0;
      ierr = PetscInfo(0,"Using vector of ones for RHS\n");CHKERRQ(ierr);
      ierr = MatGetLocalSize(A,&m,PETSC_NULL);CHKERRQ(ierr);
      ierr = VecSetSizes(x,m,PETSC_DECIDE);CHKERRQ(ierr);
      ierr = VecSetFromOptions(x);CHKERRQ(ierr);
      ierr = VecSet(x,one);CHKERRQ(ierr);
      }
  } else {
    ierr = VecLoad(x,fd);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);
  }
  ierr = VecDuplicate(x,&b);CHKERRQ(ierr);
  ierr = MatMult(A,x,b);CHKERRQ(ierr);

  /* Print (for testing only) */
  ierr = MatView(A,0);CHKERRQ(ierr);
  ierr = VecView(b,0);CHKERRQ(ierr);
  /* Free data structures */
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
