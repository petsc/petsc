static char help[] = "Test LAPACK routine DSTEBZ() and DTEIN().  \n\n";

#include <petscmat.h>
#include <petscblaslapack.h>

extern PetscErrorCode CkEigenSolutions(PetscInt,Mat,PetscInt,PetscInt,PetscScalar*,Vec*,PetscReal*);

int main(int argc,char **args)
{
  PetscErrorCode ierr;
#if defined(PETSC_USE_COMPLEX) || defined(PETSC_MISSING_LAPACK_STEBZ) || defined(PETSC_MISSING_LAPACK_STEIN)
  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP_SYS,"This example requires LAPACK routines dstebz and stien and real numbers");
#else
  PetscReal      *work,tols[2];
  PetscInt       i,j;
  PetscBLASInt   n,il=1,iu=5,*iblock,*isplit,*iwork,nevs,*ifail,cklvl=2;
  PetscMPIInt    size;
  PetscBool      flg;
  Vec            *evecs;
  PetscScalar    *evecs_array,*D,*E,*evals;
  Mat            T;
  PetscReal      vl=0.0,vu=4.0,tol= 1000*PETSC_MACHINE_EPSILON;
  PetscBLASInt   nsplit,info;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  PetscAssertFalse(size != 1,PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE,"This is a uniprocessor example only!");

  n      = 100;
  nevs   = iu - il;
  ierr   = PetscMalloc1(3*n+1,&D);CHKERRQ(ierr);
  E      = D + n;
  evals  = E + n;
  ierr   = PetscMalloc1(5*n+1,&work);CHKERRQ(ierr);
  ierr   = PetscMalloc1(3*n+1,&iwork);CHKERRQ(ierr);
  ierr   = PetscMalloc1(3*n+1,&iblock);CHKERRQ(ierr);
  isplit = iblock + n;

  /* Set symmetric tridiagonal matrix */
  for (i=0; i<n; i++) {
    D[i] = 2.0;
    E[i] = 1.0;
  }

  /* Solve eigenvalue problem: A*evec = eval*evec */
  ierr = PetscPrintf(PETSC_COMM_SELF," LAPACKstebz_: compute %d eigenvalues...\n",nevs);CHKERRQ(ierr);
  LAPACKstebz_("I","E",&n,&vl,&vu,&il,&iu,&tol,(PetscReal*)D,(PetscReal*)E,&nevs,&nsplit,(PetscReal*)evals,iblock,isplit,work,iwork,&info);
  PetscAssertFalse(info,PETSC_COMM_SELF,PETSC_ERR_USER,"LAPACKstebz_ fails. info %d",info);

  ierr = PetscPrintf(PETSC_COMM_SELF," LAPACKstein_: compute %d found eigenvectors...\n",nevs);CHKERRQ(ierr);
  ierr = PetscMalloc1(n*nevs,&evecs_array);CHKERRQ(ierr);
  ierr = PetscMalloc1(nevs,&ifail);CHKERRQ(ierr);
  LAPACKstein_(&n,(PetscReal*)D,(PetscReal*)E,&nevs,(PetscReal*)evals,iblock,isplit,evecs_array,&n,work,iwork,ifail,&info);
  PetscAssertFalse(info,PETSC_COMM_SELF,PETSC_ERR_USER,"LAPACKstein_ fails. info %d",info);
  /* View evals */
  ierr = PetscOptionsHasName(NULL,NULL, "-eig_view", &flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscPrintf(PETSC_COMM_SELF," %d evals: \n",nevs);CHKERRQ(ierr);
    for (i=0; i<nevs; i++) {ierr = PetscPrintf(PETSC_COMM_SELF,"%" PetscInt_FMT "  %g\n",i,(double)evals[i]);CHKERRQ(ierr);}
  }

  /* Check residuals and orthogonality */
  ierr = MatCreate(PETSC_COMM_SELF,&T);CHKERRQ(ierr);
  ierr = MatSetSizes(T,PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
  ierr = MatSetType(T,MATSBAIJ);CHKERRQ(ierr);
  ierr = MatSetFromOptions(T);CHKERRQ(ierr);
  ierr = MatSetUp(T);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    ierr = MatSetValues(T,1,&i,1,&i,&D[i],INSERT_VALUES);CHKERRQ(ierr);
    if (i != n-1) {
      j    = i+1;
      ierr = MatSetValues(T,1,&i,1,&j,&E[i],INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(T,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(T,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = PetscMalloc1(nevs+1,&evecs);CHKERRQ(ierr);
  for (i=0; i<nevs; i++) {
    ierr = VecCreate(PETSC_COMM_SELF,&evecs[i]);CHKERRQ(ierr);
    ierr = VecSetSizes(evecs[i],PETSC_DECIDE,n);CHKERRQ(ierr);
    ierr = VecSetFromOptions(evecs[i]);CHKERRQ(ierr);
    ierr = VecPlaceArray(evecs[i],evecs_array+i*n);CHKERRQ(ierr);
  }

  tols[0] = 1.e-8;  tols[1] = 1.e-8;
  ierr    = CkEigenSolutions(cklvl,T,il-1,iu-1,evals,evecs,tols);CHKERRQ(ierr);

  for (i=0; i<nevs; i++) {
    ierr = VecResetArray(evecs[i]);CHKERRQ(ierr);
  }

  /* free space */

  ierr = MatDestroy(&T);CHKERRQ(ierr);

  for (i=0; i<nevs; i++) { ierr = VecDestroy(&evecs[i]);CHKERRQ(ierr);}
  ierr = PetscFree(evecs);CHKERRQ(ierr);
  ierr = PetscFree(D);CHKERRQ(ierr);
  ierr = PetscFree(work);CHKERRQ(ierr);
  ierr = PetscFree(iwork);CHKERRQ(ierr);
  ierr = PetscFree(iblock);CHKERRQ(ierr);
  ierr = PetscFree(evecs_array);CHKERRQ(ierr);
  ierr = PetscFree(ifail);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
#endif
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
PetscErrorCode CkEigenSolutions(PetscInt cklvl,Mat A,PetscInt il,PetscInt iu,PetscScalar *eval,Vec *evec,PetscReal *tols)
{
  PetscInt    ierr,i,j,nev;
  Vec         vt1,vt2;  /* tmp vectors */
  PetscReal   norm,norm_max;
  PetscScalar dot,tmp;
  PetscReal   dot_max;

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
          dot = PetscAbsScalar(dot - (PetscScalar)1.0);
        } else {
          dot = PetscAbsScalar(dot);
        }
        if (PetscAbsScalar(dot) > dot_max) dot_max = PetscAbsScalar(dot);
#if defined(DEBUG_CkEigenSolutions)
        if (dot > tols[1]) {
          ierr = VecNorm(evec[i],NORM_INFINITY,&norm);CHKERRQ(ierr);
          ierr = PetscPrintf(PETSC_COMM_SELF,"|delta(%d,%d)|: %g, norm: %d\n",i,j,(double)dot,(double)norm);CHKERRQ(ierr);
        }
#endif
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
      norm = PetscAbsReal(norm);
      if (norm > norm_max) norm_max = norm;
#if defined(DEBUG_CkEigenSolutions)
      if (norm > tols[0]) {
        ierr = PetscPrintf(PETSC_COMM_SELF,"  residual violation: %d, resi: %g\n",i, norm);CHKERRQ(ierr);
      }
#endif
    }
    ierr = PetscPrintf(PETSC_COMM_SELF,"    max_resi:                    %g\n", (double)norm_max);CHKERRQ(ierr);
    break;
  default:
    ierr = PetscPrintf(PETSC_COMM_SELF,"Error: cklvl=%d is not supported \n",cklvl);CHKERRQ(ierr);
  }

  ierr = VecDestroy(&vt2);CHKERRQ(ierr);
  ierr = VecDestroy(&vt1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
