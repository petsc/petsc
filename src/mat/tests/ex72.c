static char help[] = "Read a non-complex sparse matrix from a Matrix Market (v. 2.0) file\n\
and write it to a file in petsc sparse binary format. If the matrix is symmetric, the binary file is in \n\
PETSc MATSBAIJ format, otherwise it is in MATAIJ format \n\
Usage:  ./ex72 -fin <infile> -fout <outfile> \n\
(See https://math.nist.gov/MatrixMarket/ for details.)\n\
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
  PetscInt    i,j,nz,ierr,size,*rownz;
  PetscScalar *val,zero = 0.0;
  PetscViewer view;
  PetscBool   sametype,flag,symmetric = PETSC_FALSE,skew = PETSC_FALSE,real = PETSC_FALSE,pattern = PETSC_FALSE,aijonly = PETSC_FALSE;

  PetscInitialize(&argc,&argv,(char *)0,help);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  PetscAssertFalse(size != 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This is a uniprocessor example only!");

  ierr = PetscOptionsGetString(NULL,NULL,"-fin",filein,sizeof(filein),&flag);CHKERRQ(ierr);
  PetscAssertFalse(!flag,PETSC_COMM_SELF,PETSC_ERR_USER_INPUT,"Please use -fin <filename> to specify the input file name!");
  ierr = PetscOptionsGetString(NULL,NULL,"-fout",fileout,sizeof(fileout),&flag);CHKERRQ(ierr);
  PetscAssertFalse(!flag,PETSC_COMM_SELF,PETSC_ERR_USER_INPUT,"Please use -fout <filename> to specify the output file name!");
  ierr = PetscOptionsGetBool(NULL,NULL,"-aij_only",&aijonly,NULL);CHKERRQ(ierr);

  /* Read in matrix */
  ierr = PetscFOpen(PETSC_COMM_SELF,filein,"r",&file);CHKERRQ(ierr);

  if (mm_read_banner(file, &matcode) != 0)
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Could not process Matrix Market banner.");

  /*  This is how one can screen matrix types if their application */
  /*  only supports a subset of the Matrix Market data types.      */
  if (!mm_is_matrix(matcode) || !mm_is_sparse(matcode)) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Input must be a sparse matrix. Market Market type: [%s]", mm_typecode_to_str(matcode));
  }

  if (mm_is_symmetric(matcode)) symmetric = PETSC_TRUE;
  if (mm_is_skew(matcode)) skew = PETSC_TRUE;
  if (mm_is_real(matcode)) real = PETSC_TRUE;
  if (mm_is_pattern(matcode)) pattern = PETSC_TRUE;

  /* Find out size of sparse matrix .... */
  if (mm_read_mtx_crd_size(file, &M, &N, &nz))
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Size of sparse matrix is wrong.");

  ierr = mm_write_banner(stdout, matcode);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"M: %d, N: %d, nnz: %d\n",M,N,nz);CHKERRQ(ierr);

  /* Reseve memory for matrices */
  ierr = PetscMalloc4(nz,&ia,nz,&ja,nz,&val,M,&rownz);CHKERRQ(ierr);
  for (i=0; i<M; i++) rownz[i] = 1; /* Since we will add 0.0 to diagonal entries */

  /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
  /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
  /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

  for (i=0; i<nz; i++) {
    if (pattern) {
      ninput = fscanf(file, "%d %d\n", &ia[i], &ja[i]);
      PetscAssertFalse(ninput < 2,PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Badly formatted input file");
      val[i] = 1.0;
    } else if (real) {
      ninput = fscanf(file, "%d %d %lg\n", &ia[i], &ja[i], &val[i]);
      PetscAssertFalse(ninput < 3,PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Badly formatted input file");
    }
    ia[i]--; ja[i]--;     /* adjust from 1-based to 0-based */
    if (ia[i] != ja[i]) { /* already counted the diagonals above */
      if ((symmetric && aijonly) || skew) { /* transpose */
        rownz[ia[i]]++;
        rownz[ja[i]]++;
      } else rownz[ia[i]]++;
    }
  }
  ierr = PetscFClose(PETSC_COMM_SELF,file);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"Reading matrix completes.\n");CHKERRQ(ierr);

  /* Create, preallocate, and then assemble the matrix */
  ierr = MatCreate(PETSC_COMM_SELF,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,M,N);CHKERRQ(ierr);

  if (symmetric && !aijonly) {
    ierr = MatSetType(A,MATSEQSBAIJ);CHKERRQ(ierr);
    ierr = MatSetFromOptions(A);CHKERRQ(ierr);
    ierr = MatSetUp(A);CHKERRQ(ierr);
    ierr = MatSeqSBAIJSetPreallocation(A,1,0,rownz);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)A,MATSEQSBAIJ,&sametype);CHKERRQ(ierr);
    PetscAssertFalse(!sametype,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Only AIJ and SBAIJ are supported. Your mattype is not supported");
  } else {
    ierr = MatSetType(A,MATSEQAIJ);CHKERRQ(ierr);
    ierr = MatSetFromOptions(A);CHKERRQ(ierr);
    ierr = MatSetUp(A);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(A,0,rownz);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)A,MATSEQAIJ,&sametype);CHKERRQ(ierr);
    PetscAssertFalse(!sametype,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Only AIJ and SBAIJ are supported. Your mattype is not supported");
  }

  /* Add zero to diagonals, in case the matrix missing diagonals */
  for (j=0; j<M; j++)  {
    ierr = MatSetValues(A,1,&j,1,&j,&zero,INSERT_VALUES);CHKERRQ(ierr);
  }
  /* Add values to the matrix, these correspond to lower triangular part for symmetric or skew matrices */
  for (j=0; j<nz; j++) {
    ierr = MatSetValues(A,1,&ia[j],1,&ja[j],&val[j],INSERT_VALUES);CHKERRQ(ierr);
  }

  /* Add values to upper triangular part for some cases */
  if (symmetric && aijonly) {
    /* MatrixMarket matrix stores symm matrix in lower triangular part. Take its transpose */
    for (j=0; j<nz; j++) {
      ierr = MatSetValues(A,1,&ja[j],1,&ia[j],&val[j],INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  if (skew) {
    for (j=0; j<nz; j++) {
      val[j] = -val[j];
      ierr = MatSetValues(A,1,&ja[j],1,&ia[j],&val[j],INSERT_VALUES);CHKERRQ(ierr);
    }
  }

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Write out matrix */
  ierr = PetscPrintf(PETSC_COMM_SELF,"Writing matrix to binary file %s using PETSc %s format ...\n",fileout,(symmetric && !aijonly)?"SBAIJ":"AIJ");CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_SELF,fileout,FILE_MODE_WRITE,&view);CHKERRQ(ierr);
  ierr = MatView(A,view);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&view);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"Writing matrix completes.\n");CHKERRQ(ierr);

  ierr = PetscFree4(ia,ja,val,rownz);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
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
TEST*/
