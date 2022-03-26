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
#include "ex72mmio.h"

int main(int argc,char **argv)
{
  MM_typecode matcode;
  FILE        *file;
  PetscInt    M, N, ninput;
  PetscInt    *ia, *ja;
  Mat         A;
  char        filein[PETSC_MAX_PATH_LEN],fileout[PETSC_MAX_PATH_LEN];
  char        ordering[256] = MATORDERINGRCM;
  PetscInt    i,j,nz,ierr,size,*rownz;
  PetscScalar *val,zero = 0.0;
  PetscViewer view;
  PetscBool   sametype,flag,symmetric = PETSC_FALSE,skew = PETSC_FALSE,real = PETSC_FALSE,pattern = PETSC_FALSE,aijonly = PETSC_FALSE, permute = PETSC_FALSE;
  IS          rowperm = NULL,colperm = NULL;

  PetscInitialize(&argc,&argv,(char *)0,help);
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheck(size == 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This is a uniprocessor example only!");

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Matrix Market example options","");PetscCall(ierr);
  {
    PetscCall(PetscOptionsString("-fin","Input Matrix Market file","",filein,filein,sizeof(filein),&flag));
    PetscCheck(flag,PETSC_COMM_SELF,PETSC_ERR_USER_INPUT,"Please use -fin <filename> to specify the input file name!");
    PetscCall(PetscOptionsString("-fout","Output file in petsc sparse binary format","",fileout,fileout,sizeof(fileout),&flag));
    PetscCheck(flag,PETSC_COMM_SELF,PETSC_ERR_USER_INPUT,"Please use -fout <filename> to specify the output file name!");
    PetscCall(PetscOptionsBool("-aij_only","Use MATAIJ for all cases","",aijonly,&aijonly,NULL));
    PetscCall(PetscOptionsFList("-permute","Permute matrix and vector to solving in new ordering","",MatOrderingList,ordering,ordering,sizeof(ordering),&permute));
  }
  ierr = PetscOptionsEnd();PetscCall(ierr);

  /* Read in matrix */
  PetscCall(PetscFOpen(PETSC_COMM_SELF,filein,"r",&file));

  PetscCheck(mm_read_banner(file, &matcode) == 0,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Could not process Matrix Market banner.");

  /*  This is how one can screen matrix types if their application */
  /*  only supports a subset of the Matrix Market data types.      */
  PetscCheck(mm_is_matrix(matcode) && mm_is_sparse(matcode),PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Input must be a sparse matrix. Market Market type: [%s]", mm_typecode_to_str(matcode));

  if (mm_is_symmetric(matcode)) symmetric = PETSC_TRUE;
  if (mm_is_skew(matcode)) skew = PETSC_TRUE;
  if (mm_is_real(matcode)) real = PETSC_TRUE;
  if (mm_is_pattern(matcode)) pattern = PETSC_TRUE;

  /* Find out size of sparse matrix .... */
  PetscCheck(mm_read_mtx_crd_size(file, &M, &N, &nz) == 0,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Size of sparse matrix is wrong.");

  PetscCall(mm_write_banner(stdout, matcode));
  PetscCall(PetscPrintf(PETSC_COMM_SELF,"M: %d, N: %d, nnz: %d\n",M,N,nz));

  /* Reseve memory for matrices */
  PetscCall(PetscMalloc4(nz,&ia,nz,&ja,nz,&val,M,&rownz));
  for (i=0; i<M; i++) rownz[i] = 1; /* Since we will add 0.0 to diagonal entries */

  /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
  /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
  /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

  for (i=0; i<nz; i++) {
    if (pattern) {
      ninput = fscanf(file, "%d %d\n", &ia[i], &ja[i]);
      PetscCheck(ninput >= 2,PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Badly formatted input file");
      val[i] = 1.0;
    } else if (real) {
      ninput = fscanf(file, "%d %d %lg\n", &ia[i], &ja[i], &val[i]);
      PetscCheck(ninput >= 3,PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Badly formatted input file");
    }
    ia[i]--; ja[i]--;     /* adjust from 1-based to 0-based */
    if (ia[i] != ja[i]) { /* already counted the diagonals above */
      if ((symmetric && aijonly) || skew) { /* transpose */
        rownz[ia[i]]++;
        rownz[ja[i]]++;
      } else rownz[ia[i]]++;
    }
  }
  PetscCall(PetscFClose(PETSC_COMM_SELF,file));
  PetscCall(PetscPrintf(PETSC_COMM_SELF,"Reading matrix completes.\n"));

  /* Create, preallocate, and then assemble the matrix */
  PetscCall(MatCreate(PETSC_COMM_SELF,&A));
  PetscCall(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,M,N));

  if (symmetric && !aijonly) {
    PetscCall(MatSetType(A,MATSEQSBAIJ));
    PetscCall(MatSetFromOptions(A));
    PetscCall(MatSetUp(A));
    PetscCall(MatSeqSBAIJSetPreallocation(A,1,0,rownz));
    PetscCall(PetscObjectTypeCompare((PetscObject)A,MATSEQSBAIJ,&sametype));
    PetscCheck(sametype,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Only AIJ and SBAIJ are supported. Your mattype is not supported");
  } else {
    PetscCall(MatSetType(A,MATSEQAIJ));
    PetscCall(MatSetFromOptions(A));
    PetscCall(MatSetUp(A));
    PetscCall(MatSeqAIJSetPreallocation(A,0,rownz));
    PetscCall(PetscObjectTypeCompare((PetscObject)A,MATSEQAIJ,&sametype));
    PetscCheck(sametype,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Only AIJ and SBAIJ are supported. Your mattype is not supported");
  }

  /* Add zero to diagonals, in case the matrix missing diagonals */
  for (j=0; j<M; j++)  PetscCall(MatSetValues(A,1,&j,1,&j,&zero,INSERT_VALUES));
  /* Add values to the matrix, these correspond to lower triangular part for symmetric or skew matrices */
  for (j=0; j<nz; j++) PetscCall(MatSetValues(A,1,&ia[j],1,&ja[j],&val[j],INSERT_VALUES));

  /* Add values to upper triangular part for some cases */
  if (symmetric && aijonly) {
    /* MatrixMarket matrix stores symm matrix in lower triangular part. Take its transpose */
    for (j=0; j<nz; j++) PetscCall(MatSetValues(A,1,&ja[j],1,&ia[j],&val[j],INSERT_VALUES));
  }
  if (skew) {
    for (j=0; j<nz; j++) {
      val[j] = -val[j];
      PetscCall(MatSetValues(A,1,&ja[j],1,&ia[j],&val[j],INSERT_VALUES));
    }
  }

  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  if (permute) {
    Mat Aperm;
    PetscCall(MatGetOrdering(A,ordering,&rowperm,&colperm));
    PetscCall(MatPermute(A,rowperm,colperm,&Aperm));
    PetscCall(MatDestroy(&A));
    A    = Aperm;               /* Replace original operator with permuted version */
  }

  /* Write out matrix */
  PetscCall(PetscPrintf(PETSC_COMM_SELF,"Writing matrix to binary file %s using PETSc %s format ...\n",fileout,(symmetric && !aijonly)?"SBAIJ":"AIJ"));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_SELF,fileout,FILE_MODE_WRITE,&view));
  PetscCall(MatView(A,view));
  PetscCall(PetscViewerDestroy(&view));
  PetscCall(PetscPrintf(PETSC_COMM_SELF,"Writing matrix completes.\n"));

  PetscCall(PetscFree4(ia,ja,val,rownz));
  PetscCall(MatDestroy(&A));
  PetscCall(ISDestroy(&rowperm));
  PetscCall(ISDestroy(&colperm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
      requires:  !complex double !defined(PETSC_USE_64BIT_INDICES)
      depends: ex72mmio.c

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
