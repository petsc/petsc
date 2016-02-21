static char help[] = "Test LAPACK routine DSYEV() or DSYEVX(). \n\
Reads PETSc matrix A \n\
then computes selected eigenvalues, and optionally, eigenvectors of \n\
a real generalized symmetric-definite eigenproblem \n\
 A*x = lambda*x \n\
Input parameters include\n\
  -f <input_file> : file to load\n\
e.g. ./ex116 -f /home/petsc/datafiles/matrices/small  \n\n";

#include <petscmat.h>
#include <petscblaslapack.h>

extern PetscErrorCode CkEigenSolutions(PetscInt,Mat,PetscInt,PetscInt,PetscReal*,Vec*,PetscReal*);

#undef __FUNCT__
#define __FUNCT__ "main"
PetscInt main(PetscInt argc,char **args)
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
  PetscInt       m,n,i,nevs,il,iu,cklvl=2;
  PetscReal      vl,vu,abstol=1.e-8;
  PetscBLASInt   *iwork,*ifail,lwork,lierr,bn;
  PetscReal      tols[2];

  PetscInitialize(&argc,&args,(char*)0,help);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size != 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"This is a uniprocessor example only!");

  ierr = PetscOptionsHasName(NULL,NULL, "-test_syev", &flg);CHKERRQ(ierr);
  if (flg) {
    TestSYEVX = PETSC_FALSE;
  }

  /* Determine files from which we read the two matrices */
  ierr = PetscOptionsGetString(NULL,NULL,"-f",file[0],PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);

  /* Load matrix A */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file[0],FILE_MODE_READ,&fd);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetType(A,MATSEQAIJ);CHKERRQ(ierr);
  ierr = MatLoad(A,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);
  ierr = MatGetSize(A,&m,&n);CHKERRQ(ierr);

  /* Check whether A is symmetric */
  ierr = PetscOptionsHasName(NULL,NULL, "-check_symmetry", &flg);CHKERRQ(ierr);
  if (flg) {
    Mat Trans;
    ierr = MatTranspose(A,MAT_INITIAL_MATRIX, &Trans);
    ierr = MatEqual(A, Trans, &isSymmetric);
    if (!isSymmetric) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"A must be symmetric");
    ierr = MatDestroy(&Trans);CHKERRQ(ierr);
  }

  /* Solve eigenvalue problem: A_dense*x = lambda*B*x */
  /*==================================================*/
  /* Convert aij matrix to MatSeqDense for LAPACK */
  ierr = MatConvert(A,MATSEQDENSE,MAT_INITIAL_MATRIX,&A_dense);CHKERRQ(ierr);

  ierr = PetscBLASIntCast(8*n,&lwork);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(n,&bn);CHKERRQ(ierr);
  ierr = PetscMalloc1(n,&evals);CHKERRQ(ierr);
  ierr = PetscMalloc1(lwork,&work);CHKERRQ(ierr);
  ierr = MatDenseGetArray(A_dense,&arrayA);CHKERRQ(ierr);

  if (!TestSYEVX) { /* test syev() */
    printf(" LAPACKsyev: compute all %D eigensolutions...\n",m);
    LAPACKsyev_("V","U",&bn,arrayA,&bn,evals,work,&lwork,&lierr);
    evecs_array = arrayA;
    ierr        = PetscBLASIntCast(m,&nevs);CHKERRQ(ierr);
    il          = 1;
    ierr        = PetscBLASIntCast(m,&iu);CHKERRQ(ierr);
  } else { /* test syevx()  */
    il   = 1;
    ierr = PetscBLASIntCast(0.2*m,&iu);CHKERRQ(ierr);
    printf(" LAPACKsyevx: compute %D to %D-th eigensolutions...\n",il,iu);
    ierr  = PetscMalloc1(m*n+1,&evecs_array);CHKERRQ(ierr);
    ierr  = PetscMalloc1(6*n+1,&iwork);CHKERRQ(ierr);
    ifail = iwork + 5*n;

    /* in the case "I", vl and vu are not referenced */
    vl = 0.0; vu = 8.0;
    LAPACKsyevx_("V","I","U",&bn,arrayA,&bn,&vl,&vu,&il,&iu,&abstol,&nevs,evals,evecs_array,&n,work,&lwork,iwork,ifail,&lierr);
    ierr = PetscFree(iwork);CHKERRQ(ierr);
  }
  ierr = MatDenseRestoreArray(A_dense,&arrayA);CHKERRQ(ierr);
  if (nevs <= 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_CONV_FAILED, "nev=%D, no eigensolution has found", nevs);

  /* View eigenvalues */
  ierr = PetscOptionsHasName(NULL,NULL, "-eig_view", &flg);CHKERRQ(ierr);
  if (flg) {
    printf(" %D evals: \n",nevs);
    for (i=0; i<nevs; i++) printf("%D  %g\n",i+il,(double)evals[i]);
  }

  /* Check residuals and orthogonality */
  ierr = PetscMalloc1(nevs+1,&evecs);CHKERRQ(ierr);
  for (i=0; i<nevs; i++) {
    ierr = VecCreate(PETSC_COMM_SELF,&evecs[i]);CHKERRQ(ierr);
    ierr = VecSetSizes(evecs[i],PETSC_DECIDE,n);CHKERRQ(ierr);
    ierr = VecSetFromOptions(evecs[i]);CHKERRQ(ierr);
    ierr = VecPlaceArray(evecs[i],evecs_array+i*n);CHKERRQ(ierr);
  }

  tols[0] = tols[1] = PETSC_SQRT_MACHINE_EPSILON;
  ierr    = CkEigenSolutions(cklvl,A,il-1,iu-1,evals,evecs,tols);CHKERRQ(ierr);

  /* Free work space. */
  for (i=0; i<nevs; i++) { ierr = VecDestroy(&evecs[i]);CHKERRQ(ierr);}
  ierr = PetscFree(evecs);CHKERRQ(ierr);
  ierr = MatDestroy(&A_dense);CHKERRQ(ierr);
  ierr = PetscFree(work);CHKERRQ(ierr);
  if (TestSYEVX) {ierr = PetscFree(evecs_array);CHKERRQ(ierr);}

  /* Compute SVD: A_dense = U*SIGMA*transpose(V),
     JOBU=JOBV='S':  the first min(m,n) columns of U and V are returned in the arrayU and arrayV; */
  /*==============================================================================================*/
  {
    /* Convert aij matrix to MatSeqDense for LAPACK */
    PetscScalar  *arrayU,*arrayVT,*arrayErr,alpha=1.0,beta=-1.0;
    Mat          Err;
    PetscBLASInt minMN,maxMN;
    PetscInt     j;
    PetscReal    norm;

    ierr = MatConvert(A,MATSEQDENSE,MAT_INITIAL_MATRIX,&A_dense);CHKERRQ(ierr);

    minMN = PetscMin(m,n);
    maxMN = PetscMax(m,n);
    lwork = 5*minMN + maxMN;
    ierr  = PetscMalloc4(m*minMN,&arrayU,m*minMN,&arrayVT,m*minMN,&arrayErr,lwork,&work);CHKERRQ(ierr);

    /* Create matrix Err for checking error */
    ierr = MatCreate(PETSC_COMM_WORLD,&Err);CHKERRQ(ierr);
    ierr = MatSetSizes(Err,PETSC_DECIDE,PETSC_DECIDE,m,minMN);CHKERRQ(ierr);
    ierr = MatSetType(Err,MATSEQDENSE);CHKERRQ(ierr);
    ierr = MatSeqDenseSetPreallocation(Err,(PetscScalar*)arrayErr);CHKERRQ(ierr);

    /* Save A to arrayErr for checking accuracy later. arrayA will be destroyed by LAPACKgesvd_() */
    ierr = MatDenseGetArray(A_dense,&arrayA);CHKERRQ(ierr);
    ierr = PetscMemcpy(arrayErr,arrayA,sizeof(PetscScalar)*m*minMN);CHKERRQ(ierr);

    /* Compute A = U*SIGMA*VT */
    LAPACKgesvd_("S","S",&m,&n,arrayA,&m,evals,arrayU,&minMN,arrayVT,&minMN,work,&lwork,&lierr);
    ierr = MatDenseRestoreArray(A_dense,&arrayA);CHKERRQ(ierr);
    if (!lierr) {
      printf(" 1st 10 of %d singular values: \n",minMN);
      for (i=0; i<10; i++) printf("%D  %g\n",i,(double)evals[i]);
    } else {
      printf("LAPACKgesvd_ fails!");
    }

    /* Check Err = (U*Sigma*V^T - A) using BLASgemm() */
    /* U = U*Sigma */
    for (j=0; j<minMN; j++) { /* U[:,j] = sigma[j]*U[:,j] */
      for (i=0; i<m; i++) arrayU[j*m+i] *= evals[j];
    }
    /* Err = U*VT - A = alpha*U*VT + beta*Err */
    BLASgemm_("N","N",&m,&minMN,&minMN,&alpha,arrayU,&m,arrayVT,&minMN,&beta,arrayErr,&m);
    /*printf(" Err:\n");*/
    /*ierr = MatView(Err,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);*/
    ierr = MatNorm(Err,NORM_FROBENIUS,&norm);CHKERRQ(ierr);
    printf(" || U*Sigma*VT - A || = %g\n",(double)norm);

    ierr = PetscFree4(arrayU,arrayVT,arrayErr,work);CHKERRQ(ierr);
    ierr = PetscFree(evals);CHKERRQ(ierr);
    ierr = MatDestroy(&A_dense);CHKERRQ(ierr);
    ierr = MatDestroy(&Err);CHKERRQ(ierr);
  }

  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
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
#undef DEBUG_CkEigenSolutions
#undef __FUNCT__
#define __FUNCT__ "CkEigenSolutions"
PetscErrorCode CkEigenSolutions(PetscInt cklvl,Mat A,PetscInt il,PetscInt iu,PetscReal *eval,Vec *evec,PetscReal *tols)
{
  PetscInt  ierr,i,j,nev;
  Vec       vt1,vt2;    /* tmp vectors */
  PetscReal norm,tmp,dot,norm_max,dot_max;

  PetscFunctionBegin;
  nev = iu - il;
  if (nev <= 0) PetscFunctionReturn(0);

  /*ierr = VecView(evec[0],PETSC_VIEWER_STDOUT_WORLD);*/
  ierr = VecDuplicate(evec[0],&vt1);
  ierr = VecDuplicate(evec[0],&vt2);

  switch (cklvl) {
  case 2:
    dot_max = 0.0;
    for (i = il; i<iu; i++) {
      ierr = VecCopy(evec[i], vt1);
      for (j=il; j<iu; j++) {
        ierr = VecDot(evec[j],vt1,&dot);
        if (j == i) {
          dot = PetscAbsScalar(dot - 1.0);
        } else {
          dot = PetscAbsScalar(dot);
        }
        if (dot > dot_max) dot_max = dot;
#if defined(DEBUG_CkEigenSolutions)
        if (dot > tols[1]) {
          ierr = VecNorm(evec[i],NORM_INFINITY,&norm);
          ierr = PetscPrintf(PETSC_COMM_SELF,"|delta(%D,%D)|: %g, norm: %g\n",i,j,(double)dot,(double)norm);
        }
#endif
      }
    }
    ierr = PetscPrintf(PETSC_COMM_SELF,"    max|(x_j^T*x_i) - delta_ji|: %g\n",(double)dot_max);

  case 1:
    norm_max = 0.0;
    for (i = il; i< iu; i++) {
      ierr = MatMult(A, evec[i], vt1);
      ierr = VecCopy(evec[i], vt2);
      tmp  = -eval[i];
      ierr = VecAXPY(vt1,tmp,vt2);
      ierr = VecNorm(vt1, NORM_INFINITY, &norm);
      norm = PetscAbsScalar(norm);
      if (norm > norm_max) norm_max = norm;
#if defined(DEBUG_CkEigenSolutions)
      /* sniff, and bark if necessary */
      if (norm > tols[0]) {
        printf("  residual violation: %D, resi: %g\n",i, norm);
      }
#endif
    }
    ierr = PetscPrintf(PETSC_COMM_SELF,"    max_resi:                    %g\n", (double)norm_max);
    break;
  default:
    ierr = PetscPrintf(PETSC_COMM_SELF,"Error: cklvl=%D is not supported \n",cklvl);
  }
  ierr = VecDestroy(&vt2);
  ierr = VecDestroy(&vt1);
  PetscFunctionReturn(0);
}
