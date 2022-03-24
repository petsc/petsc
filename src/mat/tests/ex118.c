static char help[] = "Test LAPACK routine DSTEBZ() and DTEIN().  \n\n";

#include <petscmat.h>
#include <petscblaslapack.h>

extern PetscErrorCode CkEigenSolutions(PetscInt,Mat,PetscInt,PetscInt,PetscScalar*,Vec*,PetscReal*);

int main(int argc,char **args)
{
#if defined(PETSC_USE_COMPLEX) || defined(PETSC_MISSING_LAPACK_STEBZ) || defined(PETSC_MISSING_LAPACK_STEIN)
  CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));
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

  CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheckFalse(size != 1,PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE,"This is a uniprocessor example only!");

  n      = 100;
  nevs   = iu - il;
  CHKERRQ(PetscMalloc1(3*n+1,&D));
  E      = D + n;
  evals  = E + n;
  CHKERRQ(PetscMalloc1(5*n+1,&work));
  CHKERRQ(PetscMalloc1(3*n+1,&iwork));
  CHKERRQ(PetscMalloc1(3*n+1,&iblock));
  isplit = iblock + n;

  /* Set symmetric tridiagonal matrix */
  for (i=0; i<n; i++) {
    D[i] = 2.0;
    E[i] = 1.0;
  }

  /* Solve eigenvalue problem: A*evec = eval*evec */
  CHKERRQ(PetscPrintf(PETSC_COMM_SELF," LAPACKstebz_: compute %d eigenvalues...\n",nevs));
  LAPACKstebz_("I","E",&n,&vl,&vu,&il,&iu,&tol,(PetscReal*)D,(PetscReal*)E,&nevs,&nsplit,(PetscReal*)evals,iblock,isplit,work,iwork,&info);
  PetscCheck(!info,PETSC_COMM_SELF,PETSC_ERR_USER,"LAPACKstebz_ fails. info %d",info);

  CHKERRQ(PetscPrintf(PETSC_COMM_SELF," LAPACKstein_: compute %d found eigenvectors...\n",nevs));
  CHKERRQ(PetscMalloc1(n*nevs,&evecs_array));
  CHKERRQ(PetscMalloc1(nevs,&ifail));
  LAPACKstein_(&n,(PetscReal*)D,(PetscReal*)E,&nevs,(PetscReal*)evals,iblock,isplit,evecs_array,&n,work,iwork,ifail,&info);
  PetscCheck(!info,PETSC_COMM_SELF,PETSC_ERR_USER,"LAPACKstein_ fails. info %d",info);
  /* View evals */
  CHKERRQ(PetscOptionsHasName(NULL,NULL, "-eig_view", &flg));
  if (flg) {
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF," %d evals: \n",nevs));
    for (i=0; i<nevs; i++) CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"%" PetscInt_FMT "  %g\n",i,(double)evals[i]));
  }

  /* Check residuals and orthogonality */
  CHKERRQ(MatCreate(PETSC_COMM_SELF,&T));
  CHKERRQ(MatSetSizes(T,PETSC_DECIDE,PETSC_DECIDE,n,n));
  CHKERRQ(MatSetType(T,MATSBAIJ));
  CHKERRQ(MatSetFromOptions(T));
  CHKERRQ(MatSetUp(T));
  for (i=0; i<n; i++) {
    CHKERRQ(MatSetValues(T,1,&i,1,&i,&D[i],INSERT_VALUES));
    if (i != n-1) {
      j    = i+1;
      CHKERRQ(MatSetValues(T,1,&i,1,&j,&E[i],INSERT_VALUES));
    }
  }
  CHKERRQ(MatAssemblyBegin(T,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(T,MAT_FINAL_ASSEMBLY));

  CHKERRQ(PetscMalloc1(nevs+1,&evecs));
  for (i=0; i<nevs; i++) {
    CHKERRQ(VecCreate(PETSC_COMM_SELF,&evecs[i]));
    CHKERRQ(VecSetSizes(evecs[i],PETSC_DECIDE,n));
    CHKERRQ(VecSetFromOptions(evecs[i]));
    CHKERRQ(VecPlaceArray(evecs[i],evecs_array+i*n));
  }

  tols[0] = 1.e-8;  tols[1] = 1.e-8;
  CHKERRQ(CkEigenSolutions(cklvl,T,il-1,iu-1,evals,evecs,tols));

  for (i=0; i<nevs; i++) {
    CHKERRQ(VecResetArray(evecs[i]));
  }

  /* free space */

  CHKERRQ(MatDestroy(&T));

  for (i=0; i<nevs; i++) CHKERRQ(VecDestroy(&evecs[i]));
  CHKERRQ(PetscFree(evecs));
  CHKERRQ(PetscFree(D));
  CHKERRQ(PetscFree(work));
  CHKERRQ(PetscFree(iwork));
  CHKERRQ(PetscFree(iblock));
  CHKERRQ(PetscFree(evecs_array));
  CHKERRQ(PetscFree(ifail));
  CHKERRQ(PetscFinalize());
  return 0;
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
          dot = PetscAbsScalar(dot - (PetscScalar)1.0);
        } else {
          dot = PetscAbsScalar(dot);
        }
        if (PetscAbsScalar(dot) > dot_max) dot_max = PetscAbsScalar(dot);
#if defined(DEBUG_CkEigenSolutions)
        if (dot > tols[1]) {
          CHKERRQ(VecNorm(evec[i],NORM_INFINITY,&norm));
          CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"|delta(%d,%d)|: %g, norm: %d\n",i,j,(double)dot,(double)norm));
        }
#endif
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
      norm = PetscAbsReal(norm);
      if (norm > norm_max) norm_max = norm;
#if defined(DEBUG_CkEigenSolutions)
      if (norm > tols[0]) {
        CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"  residual violation: %d, resi: %g\n",i, norm));
      }
#endif
    }
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"    max_resi:                    %g\n", (double)norm_max));
    break;
  default:
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Error: cklvl=%d is not supported \n",cklvl));
  }

  CHKERRQ(VecDestroy(&vt2));
  CHKERRQ(VecDestroy(&vt1));
  PetscFunctionReturn(0);
}
