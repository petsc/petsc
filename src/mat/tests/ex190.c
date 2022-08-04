static char help[] = "Tests MatLoad() with uneven dimensions set in program\n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            A;
  PetscViewer    fd;
  char           file[PETSC_MAX_PATH_LEN];
  PetscBool      flg;
  PetscMPIInt    rank;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  /* Determine files from which we read the matrix */
  PetscCall(PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_USER,"Must indicate binary file with the -f");

  /* Load matrices */
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd));
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetBlockSize(A,2));
  if (rank == 0) {
    PetscCall(MatSetSizes(A, 4, PETSC_DETERMINE, PETSC_DETERMINE,PETSC_DETERMINE));
  } else {
    PetscCall(MatSetSizes(A, 8, PETSC_DETERMINE, PETSC_DETERMINE,PETSC_DETERMINE));
  }
  PetscCall(MatLoad(A,fd));
  PetscCall(PetscViewerDestroy(&fd));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

      test:
         nsize: 2
         args: -mat_type aij -mat_view -f ${wPETSC_DIR}/share/petsc/datafiles/matrices/ns-real-int32-float64
         requires: double !complex !defined(PETSC_USE_64BIT_INDICES)

      test:
         suffix: 2
         nsize: 2
         args: -mat_type baij -mat_view -f ${wPETSC_DIR}/share/petsc/datafiles/matrices/ns-real-int32-float64
         requires: double !complex !defined(PETSC_USE_64BIT_INDICES)

      test:
         suffix: 3
         nsize: 2
         args: -mat_type sbaij -mat_view -f ${wPETSC_DIR}/share/petsc/datafiles/matrices/ns-real-int32-float64
         requires: double !complex !defined(PETSC_USE_64BIT_INDICES)

TEST*/
