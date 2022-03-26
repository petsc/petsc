static char help[] = "Tests MatLoad() with blocksize set in in program\n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            A;
  PetscViewer    fd;
  char           file[PETSC_MAX_PATH_LEN];
  PetscBool      flg;

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  /* Determine files from which we read the matrix */
  PetscCall(PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_USER,"Must indicate binary file with the -f");

  /* Load matrices */
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd));
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetType(A,MATSBAIJ));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetBlockSize(A,2));
  PetscCall(MatLoad(A,fd));
  PetscCall(PetscViewerDestroy(&fd));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

      test:
         args: -mat_type aij -f ${wPETSC_DIR}/share/petsc/datafiles/matrices/ns-real-int32-float64 -malloc_dump
         output_file: output/ex180_1.out
         requires: !complex double !defined(PETSC_USE_64BIT_INDICES)

      test:
         suffix: 2
         nsize: 2
         args: -mat_type aij -f ${wPETSC_DIR}/share/petsc/datafiles/matrices/ns-real-int32-float64 -malloc_dump
         output_file: output/ex180_1.out
         requires: !complex double !defined(PETSC_USE_64BIT_INDICES)

      test:
         suffix: 3
         args: -mat_type baij -f ${wPETSC_DIR}/share/petsc/datafiles/matrices/ns-real-int32-float64 -malloc_dump
         output_file: output/ex180_1.out
         requires: !complex double !defined(PETSC_USE_64BIT_INDICES)

      test:
         suffix: 4more
         nsize: 2
         args: -mat_type baij -f ${wPETSC_DIR}/share/petsc/datafiles/matrices/ns-real-int32-float64 -malloc_dump
         output_file: output/ex180_1.out
         requires: !complex double !defined(PETSC_USE_64BIT_INDICES)

      test:
         suffix: 5
         args: -mat_type sbaij -f ${wPETSC_DIR}/share/petsc/datafiles/matrices/ns-real-int32-float64 -malloc_dump
         output_file: output/ex180_1.out
         requires: !complex double !defined(PETSC_USE_64BIT_INDICES)

      test:
         suffix: 6
         nsize: 2
         args: -mat_type sbaij -f ${wPETSC_DIR}/share/petsc/datafiles/matrices/ns-real-int32-float64 -malloc_dump
         output_file: output/ex180_1.out
         requires: !complex double !defined(PETSC_USE_64BIT_INDICES)

      test:
         suffix: 7
         args: -mat_type sbaij -matload_block_size 4 -f ${wPETSC_DIR}/share/petsc/datafiles/matrices/ns-real-int32-float64 -malloc_dump
         output_file: output/ex180_1.out
         requires: !complex double !defined(PETSC_USE_64BIT_INDICES)

      test:
         suffix: 8
         nsize: 2
         args: -mat_type sbaij -matload_block_size 4 -f ${wPETSC_DIR}/share/petsc/datafiles/matrices/ns-real-int32-float64 -malloc_dump
         output_file: output/ex180_1.out
         requires: !complex double !defined(PETSC_USE_64BIT_INDICES)

      test:
         suffix: 9
         args: -mat_type baij -matload_block_size 4 -f ${wPETSC_DIR}/share/petsc/datafiles/matrices/ns-real-int32-float64 -malloc_dump
         output_file: output/ex180_1.out
         requires: !complex double !defined(PETSC_USE_64BIT_INDICES)

      test:
         suffix: 10
         nsize: 2
         args: -mat_type baij -matload_block_size 4 -f ${wPETSC_DIR}/share/petsc/datafiles/matrices/ns-real-int32-float64 -malloc_dump
         output_file: output/ex180_1.out
         requires: !complex double !defined(PETSC_USE_64BIT_INDICES)

TEST*/
