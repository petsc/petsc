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
  PetscErrorCode ierr;
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

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  PetscCheckFalse(size != 1,PETSC_COMM_WORLD,PETSC_ERR_SUP,"This is a uniprocessor example only!");

  ierr = PetscOptionsHasName(NULL,NULL, "-test_zheevx", &flg);CHKERRQ(ierr);
  if (flg) {
    TestZHEEV  = PETSC_FALSE;
    TestZHEEVX = PETSC_TRUE;
  }
  ierr = PetscOptionsHasName(NULL,NULL, "-test_zhegv", &flg);CHKERRQ(ierr);
  if (flg) {
    TestZHEEV = PETSC_FALSE;
    TestZHEGV = PETSC_TRUE;
  }
  ierr = PetscOptionsHasName(NULL,NULL, "-test_zhegvx", &flg);CHKERRQ(ierr);
  if (flg) {
    TestZHEEV  = PETSC_FALSE;
    TestZHEGVX = PETSC_TRUE;
  }

  ierr = PetscOptionsGetReal(NULL,NULL,"-sigma1",&sigma1,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  dim  = n*n;

  ierr = MatCreate(PETSC_COMM_SELF,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,dim,dim);CHKERRQ(ierr);
  ierr = MatSetType(A,MATSEQDENSE);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);

  ierr = PetscOptionsHasName(NULL,NULL,"-norandom",&flg);CHKERRQ(ierr);
  if (flg) use_random = 0;
  else     use_random = 1;
  if (use_random) {
    ierr = PetscRandomCreate(PETSC_COMM_SELF,&rctx);CHKERRQ(ierr);
    ierr = PetscRandomSetFromOptions(rctx);CHKERRQ(ierr);
    ierr = PetscRandomSetInterval(rctx,0.0,PETSC_i);CHKERRQ(ierr);
  } else {
    sigma2 = 10.0*PETSC_i;
  }
  h2 = 1.0/((n+1)*(n+1));
  for (Ii=0; Ii<dim; Ii++) {
    v = -1.0; i = Ii/n; j = Ii - i*n;
    if (i>0) {
      J = Ii-n; ierr = MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);
    }
    if (i<n-1) {
      J = Ii+n; ierr = MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);
    }
    if (j>0) {
      J = Ii-1; ierr = MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);
    }
    if (j<n-1) {
      J = Ii+1; ierr = MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);
    }
    if (use_random) {ierr = PetscRandomGetValue(rctx,&sigma2);CHKERRQ(ierr);}
    v    = 4.0 - sigma1*h2;
    ierr = MatSetValues(A,1,&Ii,1,&Ii,&v,ADD_VALUES);CHKERRQ(ierr);
  }
  /* make A complex Hermitian */
  v    = sigma2*h2;
  Ii   = 0; J = 1;
  ierr = MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);
  v    = -sigma2*h2;
  ierr = MatSetValues(A,1,&J,1,&Ii,&v,ADD_VALUES);CHKERRQ(ierr);
  if (use_random) {ierr = PetscRandomDestroy(&rctx);CHKERRQ(ierr);}
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  m    = n = dim;

  /* Check whether A is symmetric */
  ierr = PetscOptionsHasName(NULL,NULL, "-check_symmetry", &flg);CHKERRQ(ierr);
  if (flg) {
    Mat Trans;
    ierr = MatTranspose(A,MAT_INITIAL_MATRIX, &Trans);CHKERRQ(ierr);
    ierr = MatEqual(A, Trans, &isSymmetric);CHKERRQ(ierr);
    PetscCheckFalse(!isSymmetric,PETSC_COMM_SELF,PETSC_ERR_USER,"A must be symmetric");
    ierr = MatDestroy(&Trans);CHKERRQ(ierr);
  }

  /* Convert aij matrix to MatSeqDense for LAPACK */
  ierr = PetscObjectTypeCompare((PetscObject)A,MATSEQDENSE,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MatDuplicate(A,MAT_COPY_VALUES,&A_dense);CHKERRQ(ierr);
  } else {
    ierr = MatConvert(A,MATSEQDENSE,MAT_INITIAL_MATRIX,&A_dense);CHKERRQ(ierr);
  }

  ierr = MatCreate(PETSC_COMM_SELF,&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,dim,dim);CHKERRQ(ierr);
  ierr = MatSetType(B,MATSEQDENSE);CHKERRQ(ierr);
  ierr = MatSetFromOptions(B);CHKERRQ(ierr);
  ierr = MatSetUp(B);CHKERRQ(ierr);
  v    = 1.0;
  for (Ii=0; Ii<dim; Ii++) {
    ierr = MatSetValues(B,1,&Ii,1,&Ii,&v,ADD_VALUES);CHKERRQ(ierr);
  }

  /* Solve standard eigenvalue problem: A*x = lambda*x */
  /*===================================================*/
  ierr = PetscBLASIntCast(2*n,&lwork);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(n,&bn);CHKERRQ(ierr);
  ierr = PetscMalloc1(n,&evals);CHKERRQ(ierr);
  ierr = PetscMalloc1(lwork,&work);CHKERRQ(ierr);
  ierr = MatDenseGetArray(A_dense,&arrayA);CHKERRQ(ierr);

  if (TestZHEEV) { /* test zheev() */
    ierr = PetscPrintf(PETSC_COMM_WORLD," LAPACKsyev: compute all %" PetscInt_FMT " eigensolutions...\n",m);CHKERRQ(ierr);
    ierr = PetscMalloc1(3*n-2,&rwork);CHKERRQ(ierr);
    LAPACKsyev_("V","U",&bn,arrayA,&bn,evals,work,&lwork,rwork,&lierr);
    ierr = PetscFree(rwork);CHKERRQ(ierr);

    evecs_array = arrayA;
    nevs        = m;
    il          =1; iu=m;
  }
  if (TestZHEEVX) {
    il   = 1;
    ierr = PetscBLASIntCast((0.2*m),&iu);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD," LAPACKsyevx: compute %d to %d-th eigensolutions...\n",il,iu);CHKERRQ(ierr);
    ierr = PetscMalloc1(m*n+1,&evecs_array);CHKERRQ(ierr);
    ierr = PetscMalloc1(7*n+1,&rwork);CHKERRQ(ierr);
    ierr = PetscMalloc1(5*n+1,&iwork);CHKERRQ(ierr);
    ierr = PetscMalloc1(n+1,&ifail);CHKERRQ(ierr);

    /* in the case "I", vl and vu are not referenced */
    vl = 0.0; vu = 8.0;
    ierr = PetscBLASIntCast(n,&nn);CHKERRQ(ierr);
    LAPACKsyevx_("V","I","U",&bn,arrayA,&bn,&vl,&vu,&il,&iu,&abstol,&nevs,evals,evecs_array,&nn,work,&lwork,rwork,iwork,ifail,&lierr);
    ierr = PetscFree(iwork);CHKERRQ(ierr);
    ierr = PetscFree(ifail);CHKERRQ(ierr);
    ierr = PetscFree(rwork);CHKERRQ(ierr);
  }
  if (TestZHEGV) {
    ierr = PetscPrintf(PETSC_COMM_WORLD," LAPACKsygv: compute all %" PetscInt_FMT " eigensolutions...\n",m);CHKERRQ(ierr);
    ierr = PetscMalloc1(3*n+1,&rwork);CHKERRQ(ierr);
    ierr = MatDenseGetArray(B,&arrayB);CHKERRQ(ierr);
    LAPACKsygv_(&one,"V","U",&bn,arrayA,&bn,arrayB,&bn,evals,work,&lwork,rwork,&lierr);
    evecs_array = arrayA;
    nevs        = m;
    il          = 1; iu=m;
    ierr        = MatDenseRestoreArray(B,&arrayB);CHKERRQ(ierr);
    ierr        = PetscFree(rwork);CHKERRQ(ierr);
  }
  if (TestZHEGVX) {
    il   = 1;
    ierr = PetscBLASIntCast((0.2*m),&iu);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD," LAPACKsygv: compute %d to %d-th eigensolutions...\n",il,iu);CHKERRQ(ierr);
    ierr  = PetscMalloc1(m*n+1,&evecs_array);CHKERRQ(ierr);
    ierr  = PetscMalloc1(6*n+1,&iwork);CHKERRQ(ierr);
    ifail = iwork + 5*n;
    ierr  = PetscMalloc1(7*n+1,&rwork);CHKERRQ(ierr);
    ierr  = MatDenseGetArray(B,&arrayB);CHKERRQ(ierr);
    vl    = 0.0; vu = 8.0;
    ierr = PetscBLASIntCast(n,&nn);CHKERRQ(ierr);
    LAPACKsygvx_(&one,"V","I","U",&bn,arrayA,&bn,arrayB,&bn,&vl,&vu,&il,&iu,&abstol,&nevs,evals,evecs_array,&nn,work,&lwork,rwork,iwork,ifail,&lierr);
    ierr = MatDenseRestoreArray(B,&arrayB);CHKERRQ(ierr);
    ierr = PetscFree(iwork);CHKERRQ(ierr);
    ierr = PetscFree(rwork);CHKERRQ(ierr);
  }
  ierr = MatDenseRestoreArray(A_dense,&arrayA);CHKERRQ(ierr);
  PetscCheckFalse(nevs <= 0,PETSC_COMM_SELF,PETSC_ERR_CONV_FAILED, "nev=%d, no eigensolution has found", nevs);

  /* View evals */
  ierr = PetscOptionsHasName(NULL,NULL, "-eig_view", &flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscPrintf(PETSC_COMM_WORLD," %d evals: \n",nevs);CHKERRQ(ierr);
    for (i=0; i<nevs; i++) {ierr = PetscPrintf(PETSC_COMM_WORLD,"%" PetscInt_FMT "  %g\n",i+il,(double)evals[i]);CHKERRQ(ierr);}
  }

  /* Check residuals and orthogonality */
  ierr = PetscMalloc1(nevs+1,&evecs);CHKERRQ(ierr);
  for (i=0; i<nevs; i++) {
    ierr = VecCreate(PETSC_COMM_SELF,&evecs[i]);CHKERRQ(ierr);
    ierr = VecSetSizes(evecs[i],PETSC_DECIDE,n);CHKERRQ(ierr);
    ierr = VecSetFromOptions(evecs[i]);CHKERRQ(ierr);
    ierr = VecPlaceArray(evecs[i],evecs_array+i*n);CHKERRQ(ierr);
  }

  tols[0] = PETSC_SQRT_MACHINE_EPSILON;  tols[1] = PETSC_SQRT_MACHINE_EPSILON;
  ierr    = CkEigenSolutions(cklvl,A,il-1,iu-1,evals,evecs,tols);CHKERRQ(ierr);
  for (i=0; i<nevs; i++) { ierr = VecDestroy(&evecs[i]);CHKERRQ(ierr);}
  ierr = PetscFree(evecs);CHKERRQ(ierr);

  /* Free work space. */
  if (TestZHEEVX || TestZHEGVX) {
    ierr = PetscFree(evecs_array);CHKERRQ(ierr);
  }
  ierr = PetscFree(evals);CHKERRQ(ierr);
  ierr = PetscFree(work);CHKERRQ(ierr);
  ierr = MatDestroy(&A_dense);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
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
  PetscInt    ierr,i,j,nev;
  Vec         vt1,vt2;  /* tmp vectors */
  PetscReal   norm,tmp,norm_max,dot_max,rdot;
  PetscScalar dot;

  PetscFunctionBegin;
  nev = iu - il;
  if (nev <= 0) PetscFunctionReturn(0);

  ierr = VecDuplicate(evec[0],&vt1);CHKERRQ(ierr);
  ierr = VecDuplicate(evec[0],&vt2);CHKERRQ(ierr);

  switch (cklvl) {
  case 2:
    dot_max = 0.0;
    for (i = il; i<iu; i++) {
      ierr = VecCopy(evec[i], vt1);CHKERRQ(ierr);
      for (j=il; j<iu; j++) {
        ierr = VecDot(evec[j],vt1,&dot);CHKERRQ(ierr);
        if (j == i) {
          rdot = PetscAbsScalar(dot - (PetscScalar)1.0);
        } else {
          rdot = PetscAbsScalar(dot);
        }
        if (rdot > dot_max) dot_max = rdot;
        if (rdot > tols[1]) {
          ierr = VecNorm(evec[i],NORM_INFINITY,&norm);CHKERRQ(ierr);
          ierr = PetscPrintf(PETSC_COMM_SELF,"|delta(%" PetscInt_FMT ",%" PetscInt_FMT ")|: %g, norm: %g\n",i,j,(double)rdot,(double)norm);CHKERRQ(ierr);
        }
      }
    }
    ierr = PetscPrintf(PETSC_COMM_SELF,"    max|(x_j^T*x_i) - delta_ji|: %g\n",(double)dot_max);CHKERRQ(ierr);

  case 1:
    norm_max = 0.0;
    for (i = il; i< iu; i++) {
      ierr = MatMult(A, evec[i], vt1);CHKERRQ(ierr);
      ierr = VecCopy(evec[i], vt2);CHKERRQ(ierr);
      tmp  = -eval[i];
      ierr = VecAXPY(vt1,tmp,vt2);CHKERRQ(ierr);
      ierr = VecNorm(vt1, NORM_INFINITY, &norm);CHKERRQ(ierr);
      norm = PetscAbs(norm);
      if (norm > norm_max) norm_max = norm;
      /* sniff, and bark if necessary */
      if (norm > tols[0]) {
        ierr = PetscPrintf(PETSC_COMM_WORLD,"  residual violation: %" PetscInt_FMT ", resi: %g\n",i, norm);CHKERRQ(ierr);
      }
    }
    ierr = PetscPrintf(PETSC_COMM_SELF,"    max_resi:                    %g\n", (double)norm_max);CHKERRQ(ierr);
    break;
  default:
    ierr = PetscPrintf(PETSC_COMM_SELF,"Error: cklvl=%" PetscInt_FMT " is not supported \n",cklvl);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&vt2);CHKERRQ(ierr);
  ierr = VecDestroy(&vt1);CHKERRQ(ierr);
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
