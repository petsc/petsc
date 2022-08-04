
static char help[] = "Reads a PETSc matrix and computes the 2 norm of the columns\n\n";

/*
  Include "petscmat.h" so that we can use matrices.
  automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h    - vectors
     petscmat.h    - matrices
     petscis.h     - index sets            petscviewer.h - viewers
*/
#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            A;                       /* matrix */
  PetscViewer    fd;                      /* viewer */
  char           file[PETSC_MAX_PATH_LEN];            /* input file name */
  PetscReal      *norms;
  PetscInt       n,cstart,cend;
  PetscBool      flg;
  PetscViewerFormat format;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  /*
     Determine files from which we read the matrix
  */
  PetscCall(PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_USER,"Must indicate binary file with the -f option");

  /*
     Open binary file.  Note that we use FILE_MODE_READ to indicate
     reading from this file.
  */
  PetscCall(PetscViewerCreate(PETSC_COMM_WORLD,&fd));
  PetscCall(PetscViewerSetType(fd,PETSCVIEWERBINARY));
  PetscCall(PetscViewerSetFromOptions(fd));
  PetscCall(PetscOptionsGetEnum(NULL,NULL,"-viewer_format",PetscViewerFormats,(PetscEnum*)&format,&flg));
  if (flg) PetscCall(PetscViewerPushFormat(fd,format));
  PetscCall(PetscViewerFileSetMode(fd,FILE_MODE_READ));
  PetscCall(PetscViewerFileSetName(fd,file));

  /*
    Load the matrix; then destroy the viewer.
    Matrix type is set automatically but you can override it by MatSetType() prior to MatLoad().
    Do that only if you really insist on the given type.
  */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetOptionsPrefix(A,"a_"));
  PetscCall(PetscObjectSetName((PetscObject) A,"A"));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatLoad(A,fd));
  PetscCall(PetscViewerDestroy(&fd));

  PetscCall(MatGetSize(A,NULL,&n));
  PetscCall(MatGetOwnershipRangeColumn(A,&cstart,&cend));
  PetscCall(PetscMalloc1(n,&norms));
  PetscCall(MatGetColumnNorms(A,NORM_2,norms));
  PetscCall(PetscRealView(cend-cstart,norms+cstart,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscFree(norms));

  PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject)A,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(MatViewFromOptions(A,NULL,"-mat_view"));

  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      suffix: mpiaij
      nsize: 2
      requires: datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/small -a_mat_type mpiaij
      args: -a_matload_symmetric

   test:
      suffix: mpiaij_hdf5
      nsize: 2
      requires: datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES) hdf5 defined(PETSC_HDF5_HAVE_ZLIB)
      args: -f ${DATAFILESPATH}/matrices/matlab/small.mat -a_mat_type mpiaij -viewer_type hdf5 -viewer_format hdf5_mat
      args: -a_matload_symmetric

   test:
      suffix: mpiaij_rect_hdf5
      nsize: 2
      requires: datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES) hdf5 defined(PETSC_HDF5_HAVE_ZLIB)
      args: -f ${DATAFILESPATH}/matrices/matlab/small_rect.mat -a_mat_type mpiaij -viewer_type hdf5 -viewer_format hdf5_mat

   test:
      # test for more processes than rows
      suffix: mpiaij_hdf5_tiny
      nsize: 8
      requires: double !complex !defined(PETSC_USE_64BIT_INDICES) hdf5 defined(PETSC_HDF5_HAVE_ZLIB)
      args: -f ${wPETSC_DIR}/share/petsc/datafiles/matrices/tiny_system_with_x0.mat -a_mat_type mpiaij -viewer_type hdf5 -viewer_format hdf5_mat
      args: -a_matload_symmetric

   test:
      # test for more processes than rows, complex
      TODO: not yet implemented for MATLAB complex format
      suffix: mpiaij_hdf5_tiny_complex
      nsize: 8
      requires: double complex !defined(PETSC_USE_64BIT_INDICES) hdf5 defined(PETSC_HDF5_HAVE_ZLIB)
      args: -f ${wPETSC_DIR}/share/petsc/datafiles/matrices/tiny_system_with_x0_complex.mat -a_mat_type mpiaij -viewer_type hdf5 -viewer_format hdf5_mat
      args: -a_matload_symmetric

   test:
      TODO: mpibaij not supported yet
      suffix: mpibaij_hdf5
      nsize: 2
      requires: datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES) hdf5 defined(PETSC_HDF5_HAVE_ZLIB)
      args: -f ${DATAFILESPATH}/matrices/matlab/small.mat -a_mat_type mpibaij -a_mat_block_size 2 -viewer_type hdf5 -viewer_format hdf5_mat
      args: -a_matload_symmetric

   test:
      suffix: mpidense
      nsize: 2
      requires: datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/small -a_mat_type mpidense
      args: -a_matload_symmetric

   test:
      suffix: seqaij
      requires: datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/small -a_mat_type seqaij
      args: -a_matload_symmetric

   test:
      suffix: seqaij_hdf5
      requires: datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES) hdf5 defined(PETSC_HDF5_HAVE_ZLIB)
      args: -f ${DATAFILESPATH}/matrices/matlab/small.mat -a_mat_type seqaij -viewer_type hdf5 -viewer_format hdf5_mat
      args: -a_matload_symmetric

   test:
      suffix: seqaij_rect_hdf5
      requires: datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES) hdf5 defined(PETSC_HDF5_HAVE_ZLIB)
      args: -f ${DATAFILESPATH}/matrices/matlab/small_rect.mat -a_mat_type seqaij -viewer_type hdf5 -viewer_format hdf5_mat

   test:
      suffix: seqdense
      requires: datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/small -a_mat_type seqdense
      args: -a_matload_symmetric

   test:
      suffix: seqdense_hdf5
      requires: datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES) hdf5 defined(PETSC_HDF5_HAVE_ZLIB)
      args: -f ${DATAFILESPATH}/matrices/matlab/small_dense.mat -a_mat_type seqdense -viewer_type hdf5 -viewer_format hdf5_mat
      args: -a_matload_symmetric

   test:
      suffix: seqdense_rect_hdf5
      requires: datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES) hdf5 defined(PETSC_HDF5_HAVE_ZLIB)
      args: -f ${DATAFILESPATH}/matrices/matlab/small_rect_dense.mat -a_mat_type seqdense -viewer_type hdf5 -viewer_format hdf5_mat

   test:
      suffix: mpidense_hdf5
      nsize: 2
      requires: datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES) hdf5 defined(PETSC_HDF5_HAVE_ZLIB)
      args: -f ${DATAFILESPATH}/matrices/matlab/small_dense.mat -a_mat_type mpidense -viewer_type hdf5 -viewer_format hdf5_mat
      args: -a_matload_symmetric

   test:
      suffix: mpidense_rect_hdf5
      nsize: 2
      requires: datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES) hdf5 defined(PETSC_HDF5_HAVE_ZLIB)
      args: -f ${DATAFILESPATH}/matrices/matlab/small_rect_dense.mat -a_mat_type mpidense -viewer_type hdf5 -viewer_format hdf5_mat
TEST*/
