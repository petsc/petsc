
static char help[] = "Tests C=A^T*B via MatTranspose() and MatMatMult(). \n\
                     Contributed by Alexander Grayver, Jan. 2012 \n\n";
/* Example:
  mpiexec -n <np> ./ex165 -fA A.dat -fB B.dat -view_C
 */

#include <petscmat.h>
int main(int argc,char **args)
{
  PetscErrorCode ierr;
  Mat            A,AT,B,C;
  PetscViewer    viewer;
  PetscBool      flg;
  char           file[PETSC_MAX_PATH_LEN];

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetString(NULL,NULL,"-fA",file,sizeof(file),&flg);CHKERRQ(ierr);
  PetscAssertFalse(!flg,PETSC_COMM_WORLD,PETSC_ERR_USER,"Input fileA not specified");
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetType(A,MATAIJ);CHKERRQ(ierr);
  ierr = MatLoad(A,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  ierr = PetscOptionsGetString(NULL,NULL,"-fB",file,sizeof(file),&flg);CHKERRQ(ierr);
  PetscAssertFalse(!flg,PETSC_COMM_WORLD,PETSC_ERR_USER,"Input fileB not specified");
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&B);CHKERRQ(ierr);
  ierr = MatSetType(B,MATDENSE);CHKERRQ(ierr);
  ierr = MatLoad(B,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  ierr = MatTranspose(A,MAT_INITIAL_MATRIX,&AT);CHKERRQ(ierr);
  ierr = MatMatMult(AT,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&C);CHKERRQ(ierr);

  ierr = PetscOptionsHasName(NULL,NULL,"-view_C",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"C.dat",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_NATIVE);CHKERRQ(ierr);
    ierr = MatView(C,viewer);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = MatDestroy(&AT);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
