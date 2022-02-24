static char help[] = "Test LAPACK routine DSYEV() or DSYEVX(). \n\
Reads PETSc matrix A \n\
then computes selected eigenvalues, and optionally, eigenvectors of \n\
a real generalized symmetric-definite eigenproblem \n\
 A*x = lambda*x \n\
Input parameters include\n\
  -f <input_file> : file to load\n\
e.g. ./ex116 -f $DATAFILESPATH/matrices/small  \n\n";

#include <petscmat.h>
#include <petscblaslapack.h>

extern PetscErrorCode CkEigenSolutions(PetscInt,Mat,PetscInt,PetscInt,PetscReal*,Vec*,PetscReal*);

int main(int argc,char **args)
{
  Mat            A,A_dense;
  Vec            *evecs;
  PetscViewer    fd;                /* viewer */
  char           file[1][PETSC_MAX_PATH_LEN];     /* input file name */
  PetscBool      flg,TestSYEVX=PETSC_TRUE;
  PetscErrorCode ierr;
  PetscBool      isSymmetric;
  PetscScalar    *arrayA,*evecs_array,*work,*evals;
  PetscMPIInt    size;
  PetscInt       m,n,i,cklvl=2;
  PetscBLASInt   nevs,il,iu,in;
  PetscReal      vl,vu,abstol=1.e-8;
  PetscBLASInt   *iwork,*ifail,lwork,lierr,bn;
  PetscReal      tols[2];

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheckFalse(size != 1,PETSC_COMM_WORLD,PETSC_ERR_SUP,"This is a uniprocessor example only!");

  CHKERRQ(PetscOptionsHasName(NULL,NULL, "-test_syev", &flg));
  if (flg) {
    TestSYEVX = PETSC_FALSE;
  }

  /* Determine files from which we read the two matrices */
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-f",file[0],sizeof(file[0]),&flg));

  /* Load matrix A */
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file[0],FILE_MODE_READ,&fd));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetType(A,MATSEQAIJ));
  CHKERRQ(MatLoad(A,fd));
  CHKERRQ(PetscViewerDestroy(&fd));
  CHKERRQ(MatGetSize(A,&m,&n));

  /* Check whether A is symmetric */
  CHKERRQ(PetscOptionsHasName(NULL,NULL, "-check_symmetry", &flg));
  if (flg) {
    Mat Trans;
    CHKERRQ(MatTranspose(A,MAT_INITIAL_MATRIX, &Trans));
    CHKERRQ(MatEqual(A, Trans, &isSymmetric));
    PetscCheckFalse(!isSymmetric,PETSC_COMM_SELF,PETSC_ERR_USER,"A must be symmetric");
    CHKERRQ(MatDestroy(&Trans));
  }

  /* Solve eigenvalue problem: A_dense*x = lambda*B*x */
  /*==================================================*/
  /* Convert aij matrix to MatSeqDense for LAPACK */
  CHKERRQ(MatConvert(A,MATSEQDENSE,MAT_INITIAL_MATRIX,&A_dense));

  CHKERRQ(PetscBLASIntCast(8*n,&lwork));
  CHKERRQ(PetscBLASIntCast(n,&bn));
  CHKERRQ(PetscMalloc1(n,&evals));
  CHKERRQ(PetscMalloc1(lwork,&work));
  CHKERRQ(MatDenseGetArray(A_dense,&arrayA));

  if (!TestSYEVX) { /* test syev() */
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF," LAPACKsyev: compute all %" PetscInt_FMT " eigensolutions...\n",m));
    LAPACKsyev_("V","U",&bn,arrayA,&bn,evals,work,&lwork,&lierr);
    evecs_array = arrayA;
    CHKERRQ(PetscBLASIntCast(m,&nevs));
    il          = 1;
    CHKERRQ(PetscBLASIntCast(m,&iu));
  } else { /* test syevx()  */
    il   = 1;
    CHKERRQ(PetscBLASIntCast(0.2*m,&iu));
    CHKERRQ(PetscBLASIntCast(n,&in));
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF," LAPACKsyevx: compute %" PetscBLASInt_FMT " to %" PetscBLASInt_FMT "-th eigensolutions...\n",il,iu));
    CHKERRQ(PetscMalloc1(m*n+1,&evecs_array));
    CHKERRQ(PetscMalloc1(6*n+1,&iwork));
    ifail = iwork + 5*n;

    /* in the case "I", vl and vu are not referenced */
    vl = 0.0; vu = 8.0;
    LAPACKsyevx_("V","I","U",&bn,arrayA,&bn,&vl,&vu,&il,&iu,&abstol,&nevs,evals,evecs_array,&in,work,&lwork,iwork,ifail,&lierr);
    CHKERRQ(PetscFree(iwork));
  }
  CHKERRQ(MatDenseRestoreArray(A_dense,&arrayA));
  PetscCheckFalse(nevs <= 0,PETSC_COMM_SELF,PETSC_ERR_CONV_FAILED, "nev=%" PetscBLASInt_FMT ", no eigensolution has found", nevs);

  /* View eigenvalues */
  CHKERRQ(PetscOptionsHasName(NULL,NULL, "-eig_view", &flg));
  if (flg) {
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF," %" PetscBLASInt_FMT " evals: \n",nevs));
    for (i=0; i<nevs; i++) CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"%" PetscInt_FMT "  %g\n",(PetscInt)(i+il),(double)evals[i]));
  }

  /* Check residuals and orthogonality */
  CHKERRQ(PetscMalloc1(nevs+1,&evecs));
  for (i=0; i<nevs; i++) {
    CHKERRQ(VecCreate(PETSC_COMM_SELF,&evecs[i]));
    CHKERRQ(VecSetSizes(evecs[i],PETSC_DECIDE,n));
    CHKERRQ(VecSetFromOptions(evecs[i]));
    CHKERRQ(VecPlaceArray(evecs[i],evecs_array+i*n));
  }

  tols[0] = tols[1] = PETSC_SQRT_MACHINE_EPSILON;
  CHKERRQ(CkEigenSolutions(cklvl,A,il-1,iu-1,evals,evecs,tols));

  /* Free work space. */
  for (i=0; i<nevs; i++) CHKERRQ(VecDestroy(&evecs[i]));
  CHKERRQ(PetscFree(evecs));
  CHKERRQ(MatDestroy(&A_dense));
  CHKERRQ(PetscFree(work));
  if (TestSYEVX) CHKERRQ(PetscFree(evecs_array));

  /* Compute SVD: A_dense = U*SIGMA*transpose(V),
     JOBU=JOBV='S':  the first min(m,n) columns of U and V are returned in the arrayU and arrayV; */
  /*==============================================================================================*/
  {
    /* Convert aij matrix to MatSeqDense for LAPACK */
    PetscScalar  *arrayU,*arrayVT,*arrayErr,alpha=1.0,beta=-1.0;
    Mat          Err;
    PetscBLASInt minMN,maxMN,im,in;
    PetscInt     j;
    PetscReal    norm;

    CHKERRQ(MatConvert(A,MATSEQDENSE,MAT_INITIAL_MATRIX,&A_dense));

    minMN = PetscMin(m,n);
    maxMN = PetscMax(m,n);
    lwork = 5*minMN + maxMN;
    CHKERRQ(PetscMalloc4(m*minMN,&arrayU,m*minMN,&arrayVT,m*minMN,&arrayErr,lwork,&work));

    /* Create matrix Err for checking error */
    CHKERRQ(MatCreate(PETSC_COMM_WORLD,&Err));
    CHKERRQ(MatSetSizes(Err,PETSC_DECIDE,PETSC_DECIDE,m,minMN));
    CHKERRQ(MatSetType(Err,MATSEQDENSE));
    CHKERRQ(MatSeqDenseSetPreallocation(Err,(PetscScalar*)arrayErr));

    /* Save A to arrayErr for checking accuracy later. arrayA will be destroyed by LAPACKgesvd_() */
    CHKERRQ(MatDenseGetArray(A_dense,&arrayA));
    CHKERRQ(PetscArraycpy(arrayErr,arrayA,m*minMN));

    CHKERRQ(PetscBLASIntCast(m,&im));
    CHKERRQ(PetscBLASIntCast(n,&in));
    /* Compute A = U*SIGMA*VT */
    LAPACKgesvd_("S","S",&im,&in,arrayA,&im,evals,arrayU,&minMN,arrayVT,&minMN,work,&lwork,&lierr);
    CHKERRQ(MatDenseRestoreArray(A_dense,&arrayA));
    if (!lierr) {
      CHKERRQ(PetscPrintf(PETSC_COMM_SELF," 1st 10 of %" PetscBLASInt_FMT " singular values: \n",minMN));
      for (i=0; i<10; i++) CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"%" PetscInt_FMT "  %g\n",i,(double)evals[i]));
    } else {
      CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"LAPACKgesvd_ fails!"));
    }

    /* Check Err = (U*Sigma*V^T - A) using BLASgemm() */
    /* U = U*Sigma */
    for (j=0; j<minMN; j++) { /* U[:,j] = sigma[j]*U[:,j] */
      for (i=0; i<m; i++) arrayU[j*m+i] *= evals[j];
    }
    /* Err = U*VT - A = alpha*U*VT + beta*Err */
    BLASgemm_("N","N",&im,&minMN,&minMN,&alpha,arrayU,&im,arrayVT,&minMN,&beta,arrayErr,&im);
    CHKERRQ(MatNorm(Err,NORM_FROBENIUS,&norm));
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF," || U*Sigma*VT - A || = %g\n",(double)norm));

    CHKERRQ(PetscFree4(arrayU,arrayVT,arrayErr,work));
    CHKERRQ(PetscFree(evals));
    CHKERRQ(MatDestroy(&A_dense));
    CHKERRQ(MatDestroy(&Err));
  }

  CHKERRQ(MatDestroy(&A));
  ierr = PetscFinalize();
  return ierr;
}
/*------------------------------------------------
  Check the accuracy of the eigen solution
  ----------------------------------------------- */
/*
  input:
     cklvl      - check level:
                    1: check residual
                    2: 1 and check B-orthogonality locally
     A          - matrix
     il,iu      - lower and upper index bound of eigenvalues
     eval, evec - eigenvalues and eigenvectors stored in this process
     tols[0]    - reporting tol_res: || A * evec[i] - eval[i]*evec[i] ||
     tols[1]    - reporting tol_orth: evec[i]^T*evec[j] - delta_ij
*/
PetscErrorCode CkEigenSolutions(PetscInt cklvl,Mat A,PetscInt il,PetscInt iu,PetscReal *eval,Vec *evec,PetscReal *tols)
{
  PetscInt  i,j,nev;
  Vec       vt1,vt2;    /* tmp vectors */
  PetscReal norm,tmp,dot,norm_max,dot_max;

  PetscFunctionBegin;
  nev = iu - il;
  if (nev <= 0) PetscFunctionReturn(0);

  /*ierr = VecView(evec[0],PETSC_VIEWER_STDOUT_WORLD);*/
  CHKERRQ(VecDuplicate(evec[0],&vt1));
  CHKERRQ(VecDuplicate(evec[0],&vt2));

  switch (cklvl) {
  case 2:
    dot_max = 0.0;
    for (i = il; i<iu; i++) {
      CHKERRQ(VecCopy(evec[i], vt1));
      for (j=il; j<iu; j++) {
        CHKERRQ(VecDot(evec[j],vt1,&dot));
        if (j == i) {
          dot = PetscAbsScalar(dot - 1);
        } else {
          dot = PetscAbsScalar(dot);
        }
        if (dot > dot_max) dot_max = dot;
        if (dot > tols[1]) {
          CHKERRQ(VecNorm(evec[i],NORM_INFINITY,&norm));
          CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"|delta(%" PetscInt_FMT ",%" PetscInt_FMT ")|: %g, norm: %g\n",i,j,(double)dot,(double)norm));
        }
      }
    }
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"    max|(x_j^T*x_i) - delta_ji|: %g\n",(double)dot_max));

  case 1:
    norm_max = 0.0;
    for (i = il; i< iu; i++) {
      CHKERRQ(MatMult(A, evec[i], vt1));
      CHKERRQ(VecCopy(evec[i], vt2));
      tmp  = -eval[i];
      CHKERRQ(VecAXPY(vt1,tmp,vt2));
      CHKERRQ(VecNorm(vt1, NORM_INFINITY, &norm));
      norm = PetscAbsScalar(norm);
      if (norm > norm_max) norm_max = norm;
      /* sniff, and bark if necessary */
      if (norm > tols[0]) {
        CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"  residual violation: %" PetscInt_FMT ", resi: %g\n",i, (double)norm));
      }
    }
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"    max_resi:                    %g\n", (double)norm_max));
    break;
  default:
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Error: cklvl=%" PetscInt_FMT " is not supported \n",cklvl));
  }
  CHKERRQ(VecDestroy(&vt2));
  CHKERRQ(VecDestroy(&vt1));
  PetscFunctionReturn(0);
}

/*TEST

   build:
      requires: !complex

   test:
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/small
      output_file: output/ex116_1.out

   test:
      suffix: 2
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/small -test_syev -check_symmetry

TEST*/
