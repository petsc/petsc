
static char help[] = "Tests MatDiagonalSet() on MatLoad() matrix \n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            A;
  Vec            x;
  PetscErrorCode ierr;
  PetscViewer    fd;              /* viewer */
  char           file[PETSC_MAX_PATH_LEN]; /* input file name */
  PetscReal      norm;
  PetscBool      flg;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  /* Determine file from which we read the matrix A */
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_USER,"Must indicate binary file with the -f option");

  /* Load matrix A */
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatLoad(A,fd));
  CHKERRQ(PetscViewerDestroy(&fd));
  CHKERRQ(MatCreateVecs(A,&x,NULL));
  CHKERRQ(MatGetDiagonal(A,x));
  CHKERRQ(VecScale(x,-1.0));
  CHKERRQ(MatDiagonalSet(A,x,ADD_VALUES));
  CHKERRQ(MatGetDiagonal(A,x));
  CHKERRQ(VecNorm(x,NORM_2,&norm));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Norm %g\n",(double)norm));

  /* Free data structures */
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(VecDestroy(&x));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      nsize: 4
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -f ${wPETSC_DIR}/share/petsc/datafiles/matrices/ns-real-int32-float64 -malloc_dump

TEST*/
