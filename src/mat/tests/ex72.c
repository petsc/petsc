static char help[] = "Read a non-complex sparse matrix from a Matrix Market (v. 2.0) file\n\
and write it to a file in petsc sparse binary format. If the matrix is symmetric, the binary file is in \n\
PETSc MATSBAIJ format, otherwise it is in MATAIJ format \n\
Usage:  ./ex72 -fin <infile> -fout <outfile> \n\
(See https://math.nist.gov/MatrixMarket/ for details.)\n\
The option -permute <natural,rcm,nd,...> permutes the matrix using the ordering type.\n\
The option -aij_only allows to use MATAIJ for all cases.\n\\n";

/*
   NOTES:

   1) Matrix Market files are always 1-based, i.e. the index of the first
      element of a matrix is (1,1), not (0,0) as in C.  ADJUST THESE
      OFFSETS ACCORDINGLY offsets accordingly when reading and writing
      to files.

   2) ANSI C requires one to use the "l" format modifier when reading
      double precision floating point numbers in scanf() and
      its variants.  For example, use "%lf", "%lg", or "%le"
      when reading doubles, otherwise errors will occur.
*/
#include <petscmat.h>
#include "mmloader.h"

int main(int argc, char **argv)
{
  MM_typecode matcode;
  FILE       *file;
  PetscInt    M, N, nz;
  Mat         A;
  char        filein[PETSC_MAX_PATH_LEN], fileout[PETSC_MAX_PATH_LEN];
  char        ordering[256] = MATORDERINGRCM;
  PetscViewer view;
  PetscBool   flag, symmetric = PETSC_FALSE, aijonly = PETSC_FALSE, permute = PETSC_FALSE;
  IS          rowperm = NULL, colperm = NULL;
  PetscMPIInt size;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "This is a uniprocessor example only!");

  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "Matrix Market example options", "");
  {
    PetscCall(PetscOptionsString("-fin", "Input Matrix Market file", "", filein, filein, sizeof(filein), &flag));
    PetscCheck(flag, PETSC_COMM_SELF, PETSC_ERR_USER_INPUT, "Please use -fin <filename> to specify the input file name!");
    PetscCall(PetscOptionsString("-fout", "Output file in petsc sparse binary format", "", fileout, fileout, sizeof(fileout), &flag));
    PetscCheck(flag, PETSC_COMM_SELF, PETSC_ERR_USER_INPUT, "Please use -fout <filename> to specify the output file name!");
    PetscCall(PetscOptionsBool("-aij_only", "Use MATAIJ for all cases", "", aijonly, &aijonly, NULL));
    PetscCall(PetscOptionsFList("-permute", "Permute matrix and vector to solving in new ordering", "", MatOrderingList, ordering, ordering, sizeof(ordering), &permute));
  }
  PetscOptionsEnd();

  PetscCall(MatCreateFromMTX(&A, filein, aijonly));
  PetscCall(PetscFOpen(PETSC_COMM_SELF, filein, "r", &file));
  PetscCallExternal(mm_read_banner, file, &matcode);
  if (mm_is_symmetric(matcode)) symmetric = PETSC_TRUE;
  PetscCallExternal(mm_write_banner, stdout, matcode);
  PetscCallExternal(mm_read_mtx_crd_size, file, &M, &N, &nz);
  PetscCall(PetscFClose(PETSC_COMM_SELF, file));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "M: %d, N: %d, nnz: %d\n", M, N, nz));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "Reading matrix completes.\n"));
  if (permute) {
    Mat Aperm;
    PetscCall(MatGetOrdering(A, ordering, &rowperm, &colperm));
    PetscCall(MatPermute(A, rowperm, colperm, &Aperm));
    PetscCall(MatDestroy(&A));
    A = Aperm; /* Replace original operator with permuted version */
  }

  /* Write out matrix */
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "Writing matrix to binary file %s using PETSc %s format ...\n", fileout, (symmetric && !aijonly) ? "SBAIJ" : "AIJ"));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_SELF, fileout, FILE_MODE_WRITE, &view));
  PetscCall(MatView(A, view));
  PetscCall(PetscViewerDestroy(&view));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "Writing matrix completes.\n"));

  PetscCall(MatDestroy(&A));
  PetscCall(ISDestroy(&rowperm));
  PetscCall(ISDestroy(&colperm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
      requires:  !complex double !defined(PETSC_USE_64BIT_INDICES)
      depends: mmloader.c mmio.c

   test:
      suffix: 1
      args: -fin ${wPETSC_DIR}/share/petsc/datafiles/matrices/amesos2_test_mat0.mtx -fout petscmat.aij
      output_file: output/ex72_1.out

   test:
      suffix: 2
      args: -fin ${wPETSC_DIR}/share/petsc/datafiles/matrices/LFAT5.mtx -fout petscmat.sbaij
      output_file: output/ex72_2.out

   test:
      suffix: 3
      args: -fin ${wPETSC_DIR}/share/petsc/datafiles/matrices/m_05_05_crk.mtx -fout petscmat2.aij
      output_file: output/ex72_3.out

   test:
      suffix: 4
      args: -fin ${wPETSC_DIR}/share/petsc/datafiles/matrices/amesos2_test_mat0.mtx -fout petscmat.aij -permute rcm
      output_file: output/ex72_4.out
TEST*/
