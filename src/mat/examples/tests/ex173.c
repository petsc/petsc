
static char help[] = "Tests MatConvert() from MATAIJ and MATDENSE to MATELEMENTAL.\n\n";

#include <petscmat.h>
#include <petscviewer.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat            A,Aelem;
  PetscErrorCode ierr;
  PetscViewer    view;
  char           file[PETSC_MAX_PATH_LEN];
  PetscBool      flg;

  PetscInitialize(&argc,&args,(char*)0,help);

  /* Now reload PETSc matrix and view it */
  ierr = PetscOptionsGetString(NULL,"-f",file,PETSC_MAX_PATH_LEN,NULL);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&view);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetType(A,MATAIJ);CHKERRQ(ierr); 
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatLoad(A,view);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&view);CHKERRQ(ierr);

  /* Convert A into a Elemental matrix */
  ierr = MatConvert(A, MATELEMENTAL, MAT_INITIAL_MATRIX, &Aelem);CHKERRQ(ierr);

  /* Test MAT_REUSE_MATRIX which is only supported for inplace conversion */
  ierr = MatConvert(A, MATELEMENTAL, MAT_REUSE_MATRIX, &A);CHKERRQ(ierr);

  /* Test accuracy */
  ierr = MatMultEqual(A,Aelem,10,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NOTSAMETYPE,"A != A_elemental.");

  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&Aelem);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
