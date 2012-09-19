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
  PetscBool      flg,flgA=PETSC_FALSE,flgB=PETSC_FALSE,TestSYEVX=PETSC_TRUE;
  PetscErrorCode ierr;
  PetscBool      isSymmetric;
  PetscScalar    sigma,*arrayA,*arrayB,*evecs_array,*work,*evals;
  PetscMPIInt    size;
  PetscInt       m,n,i,j,nevs,il,iu,cklvl=2;
  PetscReal      vl,vu,abstol=1.e-8;
  PetscBLASInt   *iwork,*ifail,lwork,lierr,bn;
  PetscReal      tols[2];
  PetscInt       nzeros[2],nz;
  PetscReal      ratio;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size != 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"This is a uniprocessor example only!");

  ierr = PetscOptionsHasName(PETSC_NULL, "-test_syev", &flg);CHKERRQ(ierr);
  if (flg){
    TestSYEVX = PETSC_FALSE;
  }

  /* Determine files from which we read the two matrices */
  ierr = PetscOptionsGetString(PETSC_NULL,"-f",file[0],PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);

  /* Load matrix A */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file[0],FILE_MODE_READ,&fd);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetType(A,MATSEQAIJ);CHKERRQ(ierr);
  ierr = MatLoad(A,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);
  ierr = MatGetSize(A,&m,&n);CHKERRQ(ierr);

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
  if (!flg) {
    ierr = MatConvert(A,MATSEQDENSE,MAT_INITIAL_MATRIX,&A_dense);CHKERRQ(ierr);
  }

  /* Solve eigenvalue problem: A*x = lambda*B*x */
  /*============================================*/
  lwork = PetscBLASIntCast(8*n);
  bn    = PetscBLASIntCast(n);
  ierr = PetscMalloc(n*sizeof(PetscScalar),&evals);CHKERRQ(ierr);
  ierr = PetscMalloc(lwork*sizeof(PetscScalar),&work);CHKERRQ(ierr);
  ierr = MatDenseGetArray(A_dense,&arrayA);CHKERRQ(ierr);

  if (!TestSYEVX){ /* test syev() */
    printf(" LAPACKsyev: compute all %d eigensolutions...\n",m);
    LAPACKsyev_("V","U",&bn,arrayA,&bn,evals,work,&lwork,&lierr);
    evecs_array = arrayA;
    nevs = PetscBLASIntCast(m);
    il=1; iu=PetscBLASIntCast(m);
  } else { /* test syevx()  */
    il = 1; iu=PetscBLASIntCast((0.2*m)); /* request 1 to 20%m evalues */
    printf(" LAPACKsyevx: compute %d to %d-th eigensolutions...\n",il,iu);
    ierr = PetscMalloc((m*n+1)*sizeof(PetscScalar),&evecs_array);CHKERRQ(ierr);
    ierr = PetscMalloc((6*n+1)*sizeof(PetscBLASInt),&iwork);CHKERRQ(ierr);
    ifail = iwork + 5*n;

    /* in the case "I", vl and vu are not referenced */
    vl = 0.0; vu = 8.0;
    LAPACKsyevx_("V","I","U",&bn,arrayA,&bn,&vl,&vu,&il,&iu,&abstol,&nevs,evals,evecs_array,&n,work,&lwork,iwork,ifail,&lierr);
    ierr = PetscFree(iwork);CHKERRQ(ierr);
  }
  ierr = MatDenseRestoreArray(A,&arrayA);CHKERRQ(ierr);
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
  if (TestSYEVX){ierr = PetscFree(evecs_array);CHKERRQ(ierr);}

  ierr = PetscFree(evals);CHKERRQ(ierr);
  ierr = PetscFree(work);CHKERRQ(ierr);

  ierr = MatDestroy(&A_dense);CHKERRQ(ierr);
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
  PetscInt     ierr,i,j,nev;
  Vec          vt1,vt2; /* tmp vectors */
  PetscReal    norm,tmp,dot,norm_max,dot_max;

  PetscFunctionBegin;
  nev = iu - il;
  if (nev <= 0) PetscFunctionReturn(0);

  //ierr = VecView(evec[0],PETSC_VIEWER_STDOUT_WORLD);
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
          dot = PetscAbsScalar(dot - 1.0);
        } else {
          dot = PetscAbsScalar(dot);
        }
        if (dot > dot_max) dot_max = dot;
#ifdef DEBUG_CkEigenSolutions
        if (dot > tols[1] ) {
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
