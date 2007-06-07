static char help[] = "Test LAPACK routine DSTEBZ() and DTEIN().  \n\n";

#include "petscmat.h"
#include "petscblaslapack.h"

extern PetscErrorCode CkEigenSolutions(PetscInt,Mat,PetscInt,PetscInt,PetscReal*,Vec*,PetscReal*);

#undef __FUNCT__
#define __FUNCT__ "main"
PetscInt main(PetscInt argc,char **args)
{
  PetscErrorCode ierr;
  PetscReal      *D,*E,vl=0.0,vu=4.0,*evals,*work,tol=1.e-10,tols[2];
  PetscInt       i,j,n,il=1,iu=5,nsplit,*iblock,*isplit,*iwork,info,nevs,*ifail,cklvl=2;
  PetscMPIInt    size;
  PetscTruth     flg;
  Vec            *evecs;
  PetscScalar    *evecs_array;
  Mat            T;
  
  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size != 1) SETERRQ(PETSC_ERR_SUP,"This is a uniprocessor example only!");

  n    = 100;
  nevs = iu - il;
  ierr = PetscMalloc((3*n+1)*sizeof(PetscReal),&D);CHKERRQ(ierr);
  E     = D + n;
  evals = E + n;
  ierr = PetscMalloc((5*n+1)*sizeof(PetscReal),&work);CHKERRQ(ierr);
  ierr = PetscMalloc((3*n+1)*sizeof(PetscInt),&iwork);CHKERRQ(ierr);
  ierr = PetscMalloc((3*n+1)*sizeof(PetscInt),&iblock);CHKERRQ(ierr);
  isplit = iblock + n;

  /* Set symmetric tridiagonal matrix */
  for(i=0; i<n; i++){
    D[i] = 2.0;
    E[i] = 1.0;
  } 

  /* Solve eigenvalue problem: A*evec = eval*evec */
  printf(" LAPACKstebz_: compute %d eigenvalues...\n",nevs);    
  LAPACKstebz_("I","E",&n,&vl,&vu,&il,&iu,&tol,D,E,&nevs,&nsplit,evals,iblock,isplit,work,iwork,&info);
  if (info) SETERRQ1(PETSC_ERR_USER,"LAPACKstebz_ fails. info %d",info); 

  printf(" LAPACKstein_: compute %d eigenvectors...\n",nevs); 
  ierr = PetscMalloc(n*nevs*sizeof(PetscScalar),&evecs_array);CHKERRQ(ierr);
  ierr = PetscMalloc(nevs*sizeof(PetscInt),&ifail);CHKERRQ(ierr);
  LAPACKstein_(&n,D,E,&nevs,evals,iblock,isplit,evecs_array,&n,work,iwork,ifail,&info);
  if (info) SETERRQ1(PETSC_ERR_USER,"LAPACKstein_ fails. info %d",info); 

  /* View evals */
  ierr = PetscOptionsHasName(PETSC_NULL, "-eig_view", &flg);CHKERRQ(ierr);
  if (flg){
    printf(" %d evals: \n",nevs);
    for (i=0; i<nevs; i++) printf("%d  %G\n",i,evals[i]); 
  }

  /* Check residuals and orthogonality */ 
  ierr = MatCreate(PETSC_COMM_SELF,&T);CHKERRQ(ierr);
  ierr = MatSetSizes(T,PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
  ierr = MatSetType(T,MATSBAIJ);CHKERRQ(ierr);
  ierr = MatSetFromOptions(T);CHKERRQ(ierr);
  for (i=0; i<n; i++){
    ierr = MatSetValues(T,1,&i,1,&i,&D[i],INSERT_VALUES);CHKERRQ(ierr);
    if (i != n-1){
      j = i+1;
      ierr = MatSetValues(T,1,&i,1,&j,&E[i],INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(T,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(T,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = PetscMalloc((nevs+1)*sizeof(Vec),&evecs);CHKERRQ(ierr);
  for (i=0; i<nevs; i++){
    ierr = VecCreate(PETSC_COMM_SELF,&evecs[i]);CHKERRQ(ierr);
    ierr = VecSetSizes(evecs[i],PETSC_DECIDE,n);CHKERRQ(ierr);
    ierr = VecSetFromOptions(evecs[i]);CHKERRQ(ierr);
    ierr = VecPlaceArray(evecs[i],evecs_array+i*n);CHKERRQ(ierr);
  }
    
  tols[0] = 1.e-8;  tols[1] = 1.e-8;
  ierr = CkEigenSolutions(cklvl,T,il-1,iu-1,evals,evecs,tols);CHKERRQ(ierr);
  
  /* free space */
  ierr = MatDestroy(T);CHKERRQ(ierr);
  for (i=0; i<nevs; i++){ ierr = VecDestroy(evecs[i]);CHKERRQ(ierr);}
  ierr = PetscFree(evecs);CHKERRQ(ierr);
  ierr = PetscFree(D);CHKERRQ(ierr);
  ierr = PetscFree(work);CHKERRQ(ierr);
  ierr = PetscFree(iwork);CHKERRQ(ierr);
  ierr = PetscFree(iblock);CHKERRQ(ierr);
  ierr = PetscFree(evecs_array);CHKERRQ(ierr);
  ierr = PetscFree(ifail);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
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
  ierr = VecDestroy(vt2); 
  ierr = VecDestroy(vt1);
  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
