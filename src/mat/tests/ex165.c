
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
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-fA",file,sizeof(file),&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_USER,"Input fileA not specified");
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&viewer));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetType(A,MATAIJ));
  CHKERRQ(MatLoad(A,viewer));
  CHKERRQ(PetscViewerDestroy(&viewer));

  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-fB",file,sizeof(file),&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_USER,"Input fileB not specified");
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&viewer));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&B));
  CHKERRQ(MatSetType(B,MATDENSE));
  CHKERRQ(MatLoad(B,viewer));
  CHKERRQ(PetscViewerDestroy(&viewer));

  CHKERRQ(MatTranspose(A,MAT_INITIAL_MATRIX,&AT));
  CHKERRQ(MatMatMult(AT,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&C));

  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-view_C",&flg));
  if (flg) {
    CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,"C.dat",FILE_MODE_WRITE,&viewer));
    CHKERRQ(PetscViewerPushFormat(viewer,PETSC_VIEWER_NATIVE));
    CHKERRQ(MatView(C,viewer));
    CHKERRQ(PetscViewerPopFormat(viewer));
    CHKERRQ(PetscViewerDestroy(&viewer));
  }
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&B));
  CHKERRQ(MatDestroy(&AT));
  CHKERRQ(MatDestroy(&C));
  ierr = PetscFinalize();
  return ierr;
}
