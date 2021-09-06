static char help[] = "Test MatrixMarket outputting.\n\n";

/*
  Include "petscmat.h" so that we can use matrices.
  automatically includes:
     petscsys.h    - base PETSc routines   petscvec.h    - vectors
     petscmat.h    - matrices
     petscis.h     - index sets            petscviewer.h - viewers
*/

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            A;
  PetscViewer    fd;                        /* viewer */
  char           file[PETSC_MAX_PATH_LEN];  /* input file name */
  PetscErrorCode ierr;
  PetscBool      flg;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetString(NULL,NULL,"-f0",file,sizeof(file),&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Must indicate binary file with the -f0 option");
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatLoad(A,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);

  ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_MATRIXMARKET);CHKERRQ(ierr);
  ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      args: -f0 ${wPETSC_DIR}/share/petsc/datafiles/matrices/ns-real-int32-float64
      requires: !complex double !defined(PETSC_USE_64BIT_INDICES)

TEST*/
