static char help[] = "Tests MatLoad() with blocksize set in in program\n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            A;
  PetscViewer    fd;
  char           file[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;
  PetscBool      flg;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  /* Determine files from which we read the matrix */
  ierr = PetscOptionsGetString(NULL,NULL,"-f",file,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Must indicate binary file with the -f");

  /* Load matrices */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetType(A,MATSBAIJ);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetBlockSize(A,2);CHKERRQ(ierr);
  ierr = MatLoad(A,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

      test:
         args: -mat_type aij -f ${wPETSC_DIR}/share/petsc/datafiles/matrices/ns-real-int32-float64 -malloc_dump
         output_file: output/ex180_1.out
         requires: !complex double !define(PETSC_USE_64BIT_INDICES)

      test:
         suffix: 2
         nsize: 2
         args: -mat_type aij -f ${wPETSC_DIR}/share/petsc/datafiles/matrices/ns-real-int32-float64 -malloc_dump
         output_file: output/ex180_1.out
         requires: !complex double !define(PETSC_USE_64BIT_INDICES)

      test:
         suffix: 3
         args: -mat_type baij -f ${wPETSC_DIR}/share/petsc/datafiles/matrices/ns-real-int32-float64 -malloc_dump
         output_file: output/ex180_1.out
         requires: !complex double !define(PETSC_USE_64BIT_INDICES)

      test:
         suffix: 4more
         nsize: 2
         args: -mat_type baij -f ${wPETSC_DIR}/share/petsc/datafiles/matrices/ns-real-int32-float64 -malloc_dump
         output_file: output/ex180_1.out
         requires: !complex double !define(PETSC_USE_64BIT_INDICES)

      test:
         suffix: 5
         args: -mat_type sbaij -f ${wPETSC_DIR}/share/petsc/datafiles/matrices/ns-real-int32-float64 -malloc_dump
         output_file: output/ex180_1.out
         requires: !complex double !define(PETSC_USE_64BIT_INDICES)

      test:
         suffix: 6
         nsize: 2
         args: -mat_type sbaij -f ${wPETSC_DIR}/share/petsc/datafiles/matrices/ns-real-int32-float64 -malloc_dump
         output_file: output/ex180_1.out
         requires: !complex double !define(PETSC_USE_64BIT_INDICES)

      test:
         suffix: 7
         args: -mat_type sbaij -matload_block_size 4 -f ${wPETSC_DIR}/share/petsc/datafiles/matrices/ns-real-int32-float64 -malloc_dump
         output_file: output/ex180_1.out
         requires: !complex double !define(PETSC_USE_64BIT_INDICES)

      test:
         suffix: 8
         nsize: 2
         args: -mat_type sbaij -matload_block_size 4 -f ${wPETSC_DIR}/share/petsc/datafiles/matrices/ns-real-int32-float64 -malloc_dump
         output_file: output/ex180_1.out
         requires: !complex double !define(PETSC_USE_64BIT_INDICES)

      test:
         suffix: 9
         args: -mat_type baij -matload_block_size 4 -f ${wPETSC_DIR}/share/petsc/datafiles/matrices/ns-real-int32-float64 -malloc_dump
         output_file: output/ex180_1.out
         requires: !complex double !define(PETSC_USE_64BIT_INDICES)

      test:
         suffix: 10
         nsize: 2
         args: -mat_type baij -matload_block_size 4 -f ${wPETSC_DIR}/share/petsc/datafiles/matrices/ns-real-int32-float64 -malloc_dump
         output_file: output/ex180_1.out
         requires: !complex double !define(PETSC_USE_64BIT_INDICES)

TEST*/
