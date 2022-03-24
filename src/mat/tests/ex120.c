static char help[] = "Test LAPACK routine ZHEEV, ZHEEVX, ZHEGV and ZHEGVX. \n\
ZHEEV computes all eigenvalues and, optionally, eigenvectors of a complex Hermitian matrix A. \n\n";

#include <petscmat.h>
#include <petscblaslapack.h>

extern PetscErrorCode CkEigenSolutions(PetscInt,Mat,PetscInt,PetscInt,PetscReal*,Vec*,PetscReal*);

int main(int argc,char **args)
{
  Mat            A,A_dense,B;
  Vec            *evecs;
  PetscBool      flg,TestZHEEV=PETSC_TRUE,TestZHEEVX=PETSC_FALSE,TestZHEGV=PETSC_FALSE,TestZHEGVX=PETSC_FALSE;
  PetscBool      isSymmetric;
  PetscScalar    *arrayA,*arrayB,*evecs_array=NULL,*work;
  PetscReal      *evals,*rwork;
  PetscMPIInt    size;
  PetscInt       m,i,j,cklvl=2;
  PetscReal      vl,vu,abstol=1.e-8;
  PetscBLASInt   nn,nevs,il,iu,*iwork,*ifail,lwork,lierr,bn,one=1;
  PetscReal      tols[2];
  PetscScalar    v,sigma2;
  PetscRandom    rctx;
  PetscReal      h2,sigma1 = 100.0;
  PetscInt       dim,Ii,J,n = 6,use_random;

  CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheck(size == 1,PETSC_COMM_WORLD,PETSC_ERR_SUP,"This is a uniprocessor example only!");

  CHKERRQ(PetscOptionsHasName(NULL,NULL, "-test_zheevx", &flg));
  if (flg) {
    TestZHEEV  = PETSC_FALSE;
    TestZHEEVX = PETSC_TRUE;
  }
  CHKERRQ(PetscOptionsHasName(NULL,NULL, "-test_zhegv", &flg));
  if (flg) {
    TestZHEEV = PETSC_FALSE;
    TestZHEGV = PETSC_TRUE;
  }
  CHKERRQ(PetscOptionsHasName(NULL,NULL, "-test_zhegvx", &flg));
  if (flg) {
    TestZHEEV  = PETSC_FALSE;
    TestZHEGVX = PETSC_TRUE;
  }

  CHKERRQ(PetscOptionsGetReal(NULL,NULL,"-sigma1",&sigma1,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  dim  = n*n;

  CHKERRQ(MatCreate(PETSC_COMM_SELF,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,dim,dim));
  CHKERRQ(MatSetType(A,MATSEQDENSE));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));

  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-norandom",&flg));
  if (flg) use_random = 0;
  else     use_random = 1;
  if (use_random) {
    CHKERRQ(PetscRandomCreate(PETSC_COMM_SELF,&rctx));
    CHKERRQ(PetscRandomSetFromOptions(rctx));
    CHKERRQ(PetscRandomSetInterval(rctx,0.0,PETSC_i));
  } else {
    sigma2 = 10.0*PETSC_i;
  }
  h2 = 1.0/((n+1)*(n+1));
  for (Ii=0; Ii<dim; Ii++) {
    v = -1.0; i = Ii/n; j = Ii - i*n;
    if (i>0) {
      J = Ii-n; CHKERRQ(MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES));
    }
    if (i<n-1) {
      J = Ii+n; CHKERRQ(MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES));
    }
    if (j>0) {
      J = Ii-1; CHKERRQ(MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES));
    }
    if (j<n-1) {
      J = Ii+1; CHKERRQ(MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES));
    }
    if (use_random) CHKERRQ(PetscRandomGetValue(rctx,&sigma2));
    v    = 4.0 - sigma1*h2;
    CHKERRQ(MatSetValues(A,1,&Ii,1,&Ii,&v,ADD_VALUES));
  }
  /* make A complex Hermitian */
  v    = sigma2*h2;
  Ii   = 0; J = 1;
  CHKERRQ(MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES));
  v    = -sigma2*h2;
  CHKERRQ(MatSetValues(A,1,&J,1,&Ii,&v,ADD_VALUES));
  if (use_random) CHKERRQ(PetscRandomDestroy(&rctx));
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  m    = n = dim;

  /* Check whether A is symmetric */
  CHKERRQ(PetscOptionsHasName(NULL,NULL, "-check_symmetry", &flg));
  if (flg) {
    Mat Trans;
    CHKERRQ(MatTranspose(A,MAT_INITIAL_MATRIX, &Trans));
    CHKERRQ(MatEqual(A, Trans, &isSymmetric));
    PetscCheck(isSymmetric,PETSC_COMM_SELF,PETSC_ERR_USER,"A must be symmetric");
    CHKERRQ(MatDestroy(&Trans));
  }

  /* Convert aij matrix to MatSeqDense for LAPACK */
  CHKERRQ(PetscObjectTypeCompare((PetscObject)A,MATSEQDENSE,&flg));
  if (flg) {
    CHKERRQ(MatDuplicate(A,MAT_COPY_VALUES,&A_dense));
  } else {
    CHKERRQ(MatConvert(A,MATSEQDENSE,MAT_INITIAL_MATRIX,&A_dense));
  }

  CHKERRQ(MatCreate(PETSC_COMM_SELF,&B));
  CHKERRQ(MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,dim,dim));
  CHKERRQ(MatSetType(B,MATSEQDENSE));
  CHKERRQ(MatSetFromOptions(B));
  CHKERRQ(MatSetUp(B));
  v    = 1.0;
  for (Ii=0; Ii<dim; Ii++) {
    CHKERRQ(MatSetValues(B,1,&Ii,1,&Ii,&v,ADD_VALUES));
  }

  /* Solve standard eigenvalue problem: A*x = lambda*x */
  /*===================================================*/
  CHKERRQ(PetscBLASIntCast(2*n,&lwork));
  CHKERRQ(PetscBLASIntCast(n,&bn));
  CHKERRQ(PetscMalloc1(n,&evals));
  CHKERRQ(PetscMalloc1(lwork,&work));
  CHKERRQ(MatDenseGetArray(A_dense,&arrayA));

  if (TestZHEEV) { /* test zheev() */
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," LAPACKsyev: compute all %" PetscInt_FMT " eigensolutions...\n",m));
    CHKERRQ(PetscMalloc1(3*n-2,&rwork));
    LAPACKsyev_("V","U",&bn,arrayA,&bn,evals,work,&lwork,rwork,&lierr);
    CHKERRQ(PetscFree(rwork));

    evecs_array = arrayA;
    nevs        = m;
    il          =1; iu=m;
  }
  if (TestZHEEVX) {
    il   = 1;
    CHKERRQ(PetscBLASIntCast((0.2*m),&iu));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," LAPACKsyevx: compute %d to %d-th eigensolutions...\n",il,iu));
    CHKERRQ(PetscMalloc1(m*n+1,&evecs_array));
    CHKERRQ(PetscMalloc1(7*n+1,&rwork));
    CHKERRQ(PetscMalloc1(5*n+1,&iwork));
    CHKERRQ(PetscMalloc1(n+1,&ifail));

    /* in the case "I", vl and vu are not referenced */
    vl = 0.0; vu = 8.0;
    CHKERRQ(PetscBLASIntCast(n,&nn));
    LAPACKsyevx_("V","I","U",&bn,arrayA,&bn,&vl,&vu,&il,&iu,&abstol,&nevs,evals,evecs_array,&nn,work,&lwork,rwork,iwork,ifail,&lierr);
    CHKERRQ(PetscFree(iwork));
    CHKERRQ(PetscFree(ifail));
    CHKERRQ(PetscFree(rwork));
  }
  if (TestZHEGV) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," LAPACKsygv: compute all %" PetscInt_FMT " eigensolutions...\n",m));
    CHKERRQ(PetscMalloc1(3*n+1,&rwork));
    CHKERRQ(MatDenseGetArray(B,&arrayB));
    LAPACKsygv_(&one,"V","U",&bn,arrayA,&bn,arrayB,&bn,evals,work,&lwork,rwork,&lierr);
    evecs_array = arrayA;
    nevs        = m;
    il          = 1; iu=m;
    CHKERRQ(MatDenseRestoreArray(B,&arrayB));
    CHKERRQ(PetscFree(rwork));
  }
  if (TestZHEGVX) {
    il   = 1;
    CHKERRQ(PetscBLASIntCast((0.2*m),&iu));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," LAPACKsygv: compute %d to %d-th eigensolutions...\n",il,iu));
    CHKERRQ(PetscMalloc1(m*n+1,&evecs_array));
    CHKERRQ(PetscMalloc1(6*n+1,&iwork));
    ifail = iwork + 5*n;
    CHKERRQ(PetscMalloc1(7*n+1,&rwork));
    CHKERRQ(MatDenseGetArray(B,&arrayB));
    vl    = 0.0; vu = 8.0;
    CHKERRQ(PetscBLASIntCast(n,&nn));
    LAPACKsygvx_(&one,"V","I","U",&bn,arrayA,&bn,arrayB,&bn,&vl,&vu,&il,&iu,&abstol,&nevs,evals,evecs_array,&nn,work,&lwork,rwork,iwork,ifail,&lierr);
    CHKERRQ(MatDenseRestoreArray(B,&arrayB));
    CHKERRQ(PetscFree(iwork));
    CHKERRQ(PetscFree(rwork));
  }
  CHKERRQ(MatDenseRestoreArray(A_dense,&arrayA));
  PetscCheck(nevs > 0,PETSC_COMM_SELF,PETSC_ERR_CONV_FAILED, "nev=%d, no eigensolution has found", nevs);

  /* View evals */
  CHKERRQ(PetscOptionsHasName(NULL,NULL, "-eig_view", &flg));
  if (flg) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," %d evals: \n",nevs));
    for (i=0; i<nevs; i++) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"%" PetscInt_FMT "  %g\n",i+il,(double)evals[i]));
  }

  /* Check residuals and orthogonality */
  CHKERRQ(PetscMalloc1(nevs+1,&evecs));
  for (i=0; i<nevs; i++) {
    CHKERRQ(VecCreate(PETSC_COMM_SELF,&evecs[i]));
    CHKERRQ(VecSetSizes(evecs[i],PETSC_DECIDE,n));
    CHKERRQ(VecSetFromOptions(evecs[i]));
    CHKERRQ(VecPlaceArray(evecs[i],evecs_array+i*n));
  }

  tols[0] = PETSC_SQRT_MACHINE_EPSILON;  tols[1] = PETSC_SQRT_MACHINE_EPSILON;
  CHKERRQ(CkEigenSolutions(cklvl,A,il-1,iu-1,evals,evecs,tols));
  for (i=0; i<nevs; i++) CHKERRQ(VecDestroy(&evecs[i]));
  CHKERRQ(PetscFree(evecs));

  /* Free work space. */
  if (TestZHEEVX || TestZHEGVX) {
    CHKERRQ(PetscFree(evecs_array));
  }
  CHKERRQ(PetscFree(evals));
  CHKERRQ(PetscFree(work));
  CHKERRQ(MatDestroy(&A_dense));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&B));
  CHKERRQ(PetscFinalize());
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
PetscErrorCode CkEigenSolutions(PetscInt cklvl,Mat A,PetscInt il,PetscInt iu,PetscReal *eval,Vec *evec,PetscReal *tols)
{
  PetscInt    i,j,nev;
  Vec         vt1,vt2;  /* tmp vectors */
  PetscReal   norm,tmp,norm_max,dot_max,rdot;
  PetscScalar dot;

  PetscFunctionBegin;
  nev = iu - il;
  if (nev <= 0) PetscFunctionReturn(0);

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
          rdot = PetscAbsScalar(dot - (PetscScalar)1.0);
        } else {
          rdot = PetscAbsScalar(dot);
        }
        if (rdot > dot_max) dot_max = rdot;
        if (rdot > tols[1]) {
          CHKERRQ(VecNorm(evec[i],NORM_INFINITY,&norm));
          CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"|delta(%" PetscInt_FMT ",%" PetscInt_FMT ")|: %g, norm: %g\n",i,j,(double)rdot,(double)norm));
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
      norm = PetscAbs(norm);
      if (norm > norm_max) norm_max = norm;
      /* sniff, and bark if necessary */
      if (norm > tols[0]) {
        CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  residual violation: %" PetscInt_FMT ", resi: %g\n",i, norm));
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
      requires: complex

   test:

   test:
      suffix: 2
      args: -test_zheevx

   test:
      suffix: 3
      args: -test_zhegv

   test:
      suffix: 4
      args: -test_zhegvx

TEST*/
