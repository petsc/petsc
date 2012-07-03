
static char help[] = "Reads a matrix and vector from a file and writes to another. Input options:\n\
  -fin <input_file> : file to load.  For example see $PETSC_DIR/share/petsc/datafiles/matrices\n\
  -fout <output_file> : file for saving output matrix and vector\n\n";

#include <petscmat.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  PetscErrorCode ierr;
  PetscBool   flg;
  Vec         x;
  Mat         A;
  char        file[256];
  PetscViewer fd;

  PetscInitialize(&argc,&args,(char *)0,help);

  /* Read matrix and RHS */
  ierr = PetscOptionsGetString(PETSC_NULL,"-fin",file,256,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,1,help);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetType(A,MATSEQAIJ);CHKERRQ(ierr);
  ierr = MatLoad(A,fd);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecLoad(x,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);

  /* Write matrix and vector */
  ierr = PetscOptionsGetString(PETSC_NULL,"-fout",file,256,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,1,help);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_WRITE,&fd);CHKERRQ(ierr);
  ierr = MatView(A,fd);CHKERRQ(ierr);
  ierr = VecView(x,fd);CHKERRQ(ierr);

  /* Free data structures */
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return 0;
}

