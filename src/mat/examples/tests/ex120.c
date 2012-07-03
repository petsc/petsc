static char help[] = "Test LAPACK routine ZHEEV, ZHEEVX, ZHEGV and ZHEGVX. \n\
ZHEEV computes all eigenvalues and, optionally, eigenvectors of a complex Hermitian matrix A. \n\n";

#include <petscmat.h>
#include <petscblaslapack.h>

extern PetscErrorCode CkEigenSolutions(PetscInt,Mat,PetscInt,PetscInt,PetscReal*,Vec*,PetscReal*);

#undef __FUNCT__
#define __FUNCT__ "main"
PetscInt main(PetscInt argc,char **args)
{
  Mat            A,A_dense,B;    
  Vec            *evecs;
  PetscBool      flg,TestZHEEV=PETSC_TRUE,TestZHEEVX=PETSC_FALSE,TestZHEGV=PETSC_FALSE,TestZHEGVX=PETSC_FALSE; 
  PetscErrorCode ierr;
  PetscBool      isSymmetric;
  PetscScalar    sigma,*arrayA,*arrayB,*evecs_array=PETSC_NULL,*work;
  PetscReal      *evals,*rwork;
  PetscMPIInt    size;
  PetscInt       m,i,j,nevs,il,iu,cklvl=2; 
  PetscReal      vl,vu,abstol=1.e-8; 
  PetscBLASInt   *iwork,*ifail,lwork,lierr,bn;
  PetscReal      tols[2];
  PetscInt       nzeros[2],nz;
  PetscReal      ratio;
  PetscScalar    v,none = -1.0,sigma2,pfive = 0.5,*xa;
  PetscRandom    rctx;
  PetscReal      h2,sigma1 = 100.0;
  PetscInt       dim,Ii,J,Istart,Iend,n = 6,its,use_random,one=1;
  
  PetscInitialize(&argc,&args,(char *)0,help);
#if !defined(PETSC_USE_COMPLEX)
  SETERRQ(PETSC_COMM_WORLD,1,"This example requires complex numbers");
#endif
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size != 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"This is a uniprocessor example only!");

  ierr = PetscOptionsHasName(PETSC_NULL, "-test_zheevx", &flg);CHKERRQ(ierr);
  if (flg){
    TestZHEEV  = PETSC_FALSE;
    TestZHEEVX = PETSC_TRUE; 
  }
  ierr = PetscOptionsHasName(PETSC_NULL, "-test_zhegv", &flg);CHKERRQ(ierr);
  if (flg){
    TestZHEEV  = PETSC_FALSE;
    TestZHEGV= PETSC_TRUE; 
  }
  ierr = PetscOptionsHasName(PETSC_NULL, "-test_zhegvx", &flg);CHKERRQ(ierr);
  if (flg){
    TestZHEEV  = PETSC_FALSE;
    TestZHEGVX = PETSC_TRUE; 
  }

  ierr = PetscOptionsGetReal(PETSC_NULL,"-sigma1",&sigma1,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);
  dim = n*n;

  ierr = MatCreate(PETSC_COMM_SELF,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,dim,dim);CHKERRQ(ierr);
  ierr = MatSetType(A,MATSEQDENSE);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);

  ierr = PetscOptionsHasName(PETSC_NULL,"-norandom",&flg);CHKERRQ(ierr);
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
      J = Ii-n; ierr = MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
    if (i<n-1) {
      J = Ii+n; ierr = MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
    if (j>0) {
      J = Ii-1; ierr = MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
    if (j<n-1) {
      J = Ii+1; ierr = MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
    if (use_random) {ierr = PetscRandomGetValue(rctx,&sigma2);CHKERRQ(ierr);}
    v = 4.0 - sigma1*h2; 
    ierr = MatSetValues(A,1,&Ii,1,&Ii,&v,ADD_VALUES);CHKERRQ(ierr);
  }
  /* make A complex Hermitian */
  v = sigma2*h2;
  Ii = 0; J = 1;
  ierr = MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);
  v = -sigma2*h2;
  ierr = MatSetValues(A,1,&J,1,&Ii,&v,ADD_VALUES);CHKERRQ(ierr);
  if (use_random) {ierr = PetscRandomDestroy(&rctx);CHKERRQ(ierr);}
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  m = n = dim;

  /* Check whether A is symmetric */
  ierr = PetscOptionsHasName(PETSC_NULL, "-check_symmetry", &flg);CHKERRQ(ierr);
  if (flg) {
    Mat Trans;
    ierr = MatTranspose(A,MAT_INITIAL_MATRIX, &Trans);
    ierr = MatEqual(A, Trans, &isSymmetric);
    if (!isSymmetric) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"A must be symmetric");
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
  v    = 1.0;
  for (Ii=0; Ii<dim; Ii++) {
    ierr = MatSetValues(B,1,&Ii,1,&Ii,&v,ADD_VALUES);CHKERRQ(ierr);
  }

  /* Solve standard eigenvalue problem: A*x = lambda*x */
  /*===================================================*/
  lwork = PetscBLASIntCast(2*n);
  bn    = PetscBLASIntCast(n);
  ierr = PetscMalloc(n*sizeof(PetscReal),&evals);CHKERRQ(ierr);
  ierr = PetscMalloc(lwork*sizeof(PetscScalar),&work);CHKERRQ(ierr);
  ierr = MatGetArray(A_dense,&arrayA);CHKERRQ(ierr);

  if (TestZHEEV){ /* test zheev() */
    printf(" LAPACKsyev: compute all %d eigensolutions...\n",m);
    ierr = PetscMalloc((3*n-2)*sizeof(PetscReal),&rwork);CHKERRQ(ierr);
    LAPACKsyev_("V","U",&bn,arrayA,&bn,evals,work,&lwork,rwork,&lierr); 
    ierr = PetscFree(rwork);CHKERRQ(ierr);
    evecs_array = arrayA;
    nevs = m;
    il=1; iu=m; 
  } 
  if (TestZHEEVX){
    il = 1; iu=PetscBLASIntCast((0.2*m)); /* request 1 to 20%m evalues */
    printf(" LAPACKsyevx: compute %d to %d-th eigensolutions...\n",il,iu);
    ierr = PetscMalloc((m*n+1)*sizeof(PetscScalar),&evecs_array);CHKERRQ(ierr);
    ierr = PetscMalloc((7*n+1)*sizeof(PetscReal),&rwork);CHKERRQ(ierr);
    ierr = PetscMalloc((5*n+1)*sizeof(PetscBLASInt),&iwork);CHKERRQ(ierr);
    ierr = PetscMalloc((n+1)*sizeof(PetscBLASInt),&ifail);CHKERRQ(ierr);
      
    /* in the case "I", vl and vu are not referenced */
    vl = 0.0; vu = 8.0;
    LAPACKsyevx_("V","I","U",&bn,arrayA,&bn,&vl,&vu,&il,&iu,&abstol,&nevs,evals,evecs_array,&n,work,&lwork,rwork,iwork,ifail,&lierr);  
    ierr = PetscFree(iwork);CHKERRQ(ierr);
    ierr = PetscFree(ifail);CHKERRQ(ierr);
    ierr = PetscFree(rwork);CHKERRQ(ierr);
  }
  if (TestZHEGV){
    printf(" LAPACKsygv: compute all %d eigensolutions...\n",m);
    ierr = PetscMalloc((3*n+1)*sizeof(PetscReal),&rwork);CHKERRQ(ierr);
    ierr = MatGetArray(B,&arrayB);CHKERRQ(ierr);
    LAPACKsygv_(&one,"V","U",&bn,arrayA,&bn,arrayB,&bn,evals,work,&lwork,rwork,&lierr); 
    evecs_array = arrayA;
    nevs = m;
    il=1; iu=m;
    ierr = MatRestoreArray(B,&arrayB);CHKERRQ(ierr);
    ierr = PetscFree(rwork);CHKERRQ(ierr);
  }
  if (TestZHEGVX){
    il = 1; iu=PetscBLASIntCast((0.2*m)); /* request 1 to 20%m evalues */
    printf(" LAPACKsygv: compute %d to %d-th eigensolutions...\n",il,iu);
    ierr = PetscMalloc((m*n+1)*sizeof(PetscScalar),&evecs_array);CHKERRQ(ierr);    
    ierr = PetscMalloc((6*n+1)*sizeof(PetscBLASInt),&iwork);CHKERRQ(ierr);     
    ifail = iwork + 5*n;
    ierr = PetscMalloc((7*n+1)*sizeof(PetscReal),&rwork);CHKERRQ(ierr);
    ierr = MatGetArray(B,&arrayB);CHKERRQ(ierr);
    vl = 0.0; vu = 8.0;
    LAPACKsygvx_(&one,"V","I","U",&bn,arrayA,&bn,arrayB,&bn,&vl,&vu,&il,&iu,&abstol,&nevs,evals,evecs_array,&n,work,&lwork,rwork,iwork,ifail,&lierr);
    ierr = MatRestoreArray(B,&arrayB);CHKERRQ(ierr);
    ierr = PetscFree(iwork);CHKERRQ(ierr);
    ierr = PetscFree(rwork);CHKERRQ(ierr);
  }
  ierr = MatRestoreArray(A_dense,&arrayA);CHKERRQ(ierr);
  if (nevs <= 0 ) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_CONV_FAILED, "nev=%d, no eigensolution has found", nevs);

  /* View evals */
  ierr = PetscOptionsHasName(PETSC_NULL, "-eig_view", &flg);CHKERRQ(ierr);
  if (flg){
    printf(" %d evals: \n",nevs);
    for (i=0; i<nevs; i++) printf("%d  %G\n",i+il,evals[i]); 
  }

  /* Check residuals and orthogonality */   
  ierr = PetscMalloc((nevs+1)*sizeof(Vec),&evecs);CHKERRQ(ierr);
  for (i=0; i<nevs; i++){
    ierr = VecCreate(PETSC_COMM_SELF,&evecs[i]);CHKERRQ(ierr);
    ierr = VecSetSizes(evecs[i],PETSC_DECIDE,n);CHKERRQ(ierr);
    ierr = VecSetFromOptions(evecs[i]);CHKERRQ(ierr);
    ierr = VecPlaceArray(evecs[i],evecs_array+i*n);CHKERRQ(ierr);
  }
  
  tols[0] = 1.e-8;  tols[1] = 1.e-8;
  ierr = CkEigenSolutions(cklvl,A,il-1,iu-1,evals,evecs,tols);CHKERRQ(ierr);
  for (i=0; i<nevs; i++){ ierr = VecDestroy(&evecs[i]);CHKERRQ(ierr);}
  ierr = PetscFree(evecs);CHKERRQ(ierr);
    
  /* Free work space. */
  if (TestZHEEVX || TestZHEGVX){
    ierr = PetscFree(evecs_array);CHKERRQ(ierr);
  }  
  ierr = PetscFree(evals);CHKERRQ(ierr);
  ierr = PetscFree(work);CHKERRQ(ierr);
  ierr = MatDestroy(&A_dense);CHKERRQ(ierr); 
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
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
  PetscInt     ierr,i,j,nev; 
  Vec          vt1,vt2; /* tmp vectors */
  PetscReal    norm,tmp,norm_max,dot_max,rdot; 
  PetscScalar  dot;

  PetscFunctionBegin;
  nev = iu - il;
  if (nev <= 0) PetscFunctionReturn(0);

  //ierr = VecView(evec[0],PETSC_VIEWER_STDOUT_SELF);
  ierr = VecDuplicate(evec[0],&vt1);
  ierr = VecDuplicate(evec[0],&vt2);

  switch (cklvl){
  case 2:  
    dot_max = 0.0;
    for (i = il; i<iu; i++){
      //printf("ck %d-th\n",i);
      ierr = VecCopy(evec[i], vt1);
      for (j=il; j<iu; j++){ 
        ierr = VecDot(evec[j],vt1,&dot);
        if (j == i){
          rdot = PetscAbsScalar(dot - 1.0);
        } else {
          rdot = PetscAbsScalar(dot);
        }
        if (rdot > dot_max) dot_max = rdot;
#ifdef DEBUG_CkEigenSolutions
        if (rdot > tols[1] ) {
          ierr = VecNorm(evec[i],NORM_INFINITY,&norm);
          ierr = PetscPrintf(PETSC_COMM_SELF,"|delta(%d,%d)|: %G, norm: %G\n",i,j,dot,norm);
        } 
#endif
      } 
    } 
    ierr = PetscPrintf(PETSC_COMM_SELF,"    max|(x_j^T*x_i) - delta_ji|: %G\n",dot_max);

  case 1: 
    norm_max = 0.0;
    for (i = il; i< iu; i++){
      ierr = MatMult(A, evec[i], vt1);
      ierr = VecCopy(evec[i], vt2);
      tmp  = -eval[i];
      ierr = VecAXPY(vt1,tmp,vt2);
      ierr = VecNorm(vt1, NORM_INFINITY, &norm);
      norm = PetscAbsScalar(norm); 
      if (norm > norm_max) norm_max = norm;
#ifdef DEBUG_CkEigenSolutions
      /* sniff, and bark if necessary */
      if (norm > tols[0]){
        printf( "  residual violation: %d, resi: %g\n",i, norm);
      }
#endif
    }    
    ierr = PetscPrintf(PETSC_COMM_SELF,"    max_resi:                    %G\n", norm_max);
   break;
  default:
    ierr = PetscPrintf(PETSC_COMM_SELF,"Error: cklvl=%d is not supported \n",cklvl);
  }
  ierr = VecDestroy(&vt2); 
  ierr = VecDestroy(&vt1);
  PetscFunctionReturn(0);
}
