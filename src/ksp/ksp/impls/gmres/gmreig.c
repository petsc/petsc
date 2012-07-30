
#include <../src/ksp/ksp/impls/gmres/gmresimpl.h>
#include <petscblaslapack.h>

#undef __FUNCT__  
#define __FUNCT__ "KSPComputeExtremeSingularValues_GMRES"
PetscErrorCode KSPComputeExtremeSingularValues_GMRES(KSP ksp,PetscReal *emax,PetscReal *emin)
{
#if defined(PETSC_MISSING_LAPACK_GESVD) 
  PetscFunctionBegin;
  /*
      The Cray math libraries on T3D/T3E, and early versions of Intel Math Kernel Libraries (MKL)
      for PCs do not seem to have the DGESVD() lapack routines
  */
  SETERRQ(((PetscObject)ksp)->comm,PETSC_ERR_SUP,"GESVD - Lapack routine is unavailable\nNot able to provide singular value estimates.");
#else
  KSP_GMRES      *gmres = (KSP_GMRES*)ksp->data;
  PetscErrorCode ierr;
  PetscInt       n = gmres->it + 1,i,N = gmres->max_k + 2;
  PetscBLASInt   bn, bN ,lwork, idummy,lierr;
  PetscScalar    *R = gmres->Rsvd,*work = R + N*N,sdummy;
  PetscReal      *realpart = gmres->Dsvd;

  PetscFunctionBegin;
  bn = PetscBLASIntCast(n);
  bN = PetscBLASIntCast(N);
  lwork = PetscBLASIntCast(5*N);
  idummy = PetscBLASIntCast(N);
  if (n <= 0) {
    *emax = *emin = 1.0;
    PetscFunctionReturn(0);
  }
  /* copy R matrix to work space */
  ierr = PetscMemcpy(R,gmres->hh_origin,(gmres->max_k+2)*(gmres->max_k+1)*sizeof(PetscScalar));CHKERRQ(ierr);

  /* zero below diagonal garbage */
  for (i=0; i<n; i++) {
    R[i*N+i+1] = 0.0;
  }
  
  /* compute Singular Values */
  ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  LAPACKgesvd_("N","N",&bn,&bn,R,&bN,realpart,&sdummy,&idummy,&sdummy,&idummy,work,&lwork,&lierr);
#else
  LAPACKgesvd_("N","N",&bn,&bn,R,&bN,realpart,&sdummy,&idummy,&sdummy,&idummy,work,&lwork,realpart+N,&lierr);
#endif
  if (lierr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in SVD Lapack routine %d",(int)lierr);
  ierr = PetscFPTrapPop();CHKERRQ(ierr);

  *emin = realpart[n-1];
  *emax = realpart[0];
#endif
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------------ */
/* ESSL has a different calling sequence for dgeev() and zgeev() than standard LAPACK */
#undef __FUNCT__  
#define __FUNCT__ "KSPComputeEigenvalues_GMRES"
PetscErrorCode KSPComputeEigenvalues_GMRES(KSP ksp,PetscInt nmax,PetscReal *r,PetscReal *c,PetscInt *neig)
{
#if defined(PETSC_HAVE_ESSL)
  KSP_GMRES      *gmres = (KSP_GMRES*)ksp->data;
  PetscErrorCode ierr;
  PetscInt       n = gmres->it + 1,N = gmres->max_k + 1;
  PetscInt       i,*perm;
  PetscScalar    *R = gmres->Rsvd;
  PetscScalar    *cwork = R + N*N,sdummy;
  PetscReal      *work,*realpart = gmres->Dsvd ;
  PetscBLASInt   zero = 0,bn,bN,idummy,lwork;

  PetscFunctionBegin;
  bn = PetscBLASIntCast(n);
  bN = PetscBLASIntCast(N);
  idummy = -1;                  /* unused */
  lwork = PetscBLASIntCast(5*N);
  if (nmax < n) SETERRQ(((PetscObject)ksp)->comm,PETSC_ERR_ARG_SIZ,"Not enough room in work space r and c for eigenvalues");
  *neig = n;

  if (!n) {
    PetscFunctionReturn(0);
  }
  /* copy R matrix to work space */
  ierr = PetscMemcpy(R,gmres->hes_origin,N*N*sizeof(PetscScalar));CHKERRQ(ierr);

  /* compute eigenvalues */

  /* for ESSL version need really cwork of length N (complex), 2N
     (real); already at least 5N of space has been allocated */

  ierr = PetscMalloc(lwork*sizeof(PetscReal),&work);CHKERRQ(ierr);
  ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
  LAPACKgeev_(&zero,R,&bN,cwork,&sdummy,&idummy,&idummy,&bn,work,&lwork);
  ierr = PetscFPTrapPop();CHKERRQ(ierr);
  ierr = PetscFree(work);CHKERRQ(ierr);

  /* For now we stick with the convention of storing the real and imaginary
     components of evalues separately.  But is this what we really want? */
  ierr = PetscMalloc(n*sizeof(PetscInt),&perm);CHKERRQ(ierr);

#if !defined(PETSC_USE_COMPLEX)
  for (i=0; i<n; i++) {
    realpart[i] = cwork[2*i];
    perm[i]     = i;
  }
  ierr = PetscSortRealWithPermutation(n,realpart,perm);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    r[i] = cwork[2*perm[i]];
    c[i] = cwork[2*perm[i]+1];
  }
#else
  for (i=0; i<n; i++) {
    realpart[i] = PetscRealPart(cwork[i]);
    perm[i]     = i;
  }
  ierr = PetscSortRealWithPermutation(n,realpart,perm);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    r[i] = PetscRealPart(cwork[perm[i]]);
    c[i] = PetscImaginaryPart(cwork[perm[i]]);
  }
#endif
  ierr = PetscFree(perm);CHKERRQ(ierr);
#elif defined(PETSC_MISSING_LAPACK_GEEV) 
  PetscFunctionBegin;
  SETERRQ(((PetscObject)ksp)->comm,PETSC_ERR_SUP,"GEEV - Lapack routine is unavailable\nNot able to provide eigen values.");
#elif !defined(PETSC_USE_COMPLEX)
  KSP_GMRES      *gmres = (KSP_GMRES*)ksp->data;
  PetscErrorCode ierr;
  PetscInt       n = gmres->it + 1,N = gmres->max_k + 1,i,*perm;
  PetscBLASInt   bn, bN, lwork, idummy, lierr;
  PetscScalar    *R = gmres->Rsvd,*work = R + N*N;
  PetscScalar    *realpart = gmres->Dsvd,*imagpart = realpart + N,sdummy;

  PetscFunctionBegin;
  bn = PetscBLASIntCast(n);
  bN = PetscBLASIntCast(N);
  lwork = PetscBLASIntCast(5*N);
  idummy = PetscBLASIntCast(N);
  if (nmax < n) SETERRQ(((PetscObject)ksp)->comm,PETSC_ERR_ARG_SIZ,"Not enough room in work space r and c for eigenvalues");
  *neig = n;

  if (!n) {
    PetscFunctionReturn(0);
  }

  /* copy R matrix to work space */
  ierr = PetscMemcpy(R,gmres->hes_origin,N*N*sizeof(PetscScalar));CHKERRQ(ierr);

  /* compute eigenvalues */
  ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
  LAPACKgeev_("N","N",&bn,R,&bN,realpart,imagpart,&sdummy,&idummy,&sdummy,&idummy,work,&lwork,&lierr);
  if (lierr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in LAPACK routine %d",(int)lierr);
  ierr = PetscFPTrapPop();CHKERRQ(ierr);
  ierr = PetscMalloc(n*sizeof(PetscInt),&perm);CHKERRQ(ierr);
  for (i=0; i<n; i++) { perm[i] = i;}
  ierr = PetscSortRealWithPermutation(n,realpart,perm);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    r[i] = realpart[perm[i]];
    c[i] = imagpart[perm[i]];
  }
  ierr = PetscFree(perm);CHKERRQ(ierr);
#else
  KSP_GMRES      *gmres = (KSP_GMRES*)ksp->data;
  PetscErrorCode ierr;
  PetscInt       n = gmres->it + 1,N = gmres->max_k + 1,i,*perm;
  PetscScalar    *R = gmres->Rsvd,*work = R + N*N,*eigs = work + 5*N,sdummy;
  PetscBLASInt   bn,bN,lwork,idummy,lierr;

  PetscFunctionBegin;
  bn = PetscBLASIntCast(n);
  bN = PetscBLASIntCast(N);
  lwork = PetscBLASIntCast(5*N);
  idummy = PetscBLASIntCast(N);
  if (nmax < n) SETERRQ(((PetscObject)ksp)->comm,PETSC_ERR_ARG_SIZ,"Not enough room in work space r and c for eigenvalues");
  *neig = n;

  if (!n) {
    PetscFunctionReturn(0);
  }
  /* copy R matrix to work space */
  ierr = PetscMemcpy(R,gmres->hes_origin,N*N*sizeof(PetscScalar));CHKERRQ(ierr);

  /* compute eigenvalues */
  ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
  LAPACKgeev_("N","N",&bn,R,&bN,eigs,&sdummy,&idummy,&sdummy,&idummy,work,&lwork,gmres->Dsvd,&lierr);
  if (lierr) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in LAPACK routine");
  ierr = PetscFPTrapPop();CHKERRQ(ierr);
  ierr = PetscMalloc(n*sizeof(PetscInt),&perm);CHKERRQ(ierr);
  for (i=0; i<n; i++) { perm[i] = i;}
  for (i=0; i<n; i++) { r[i]    = PetscRealPart(eigs[i]);}
  ierr = PetscSortRealWithPermutation(n,r,perm);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    r[i] = PetscRealPart(eigs[perm[i]]);
    c[i] = PetscImaginaryPart(eigs[perm[i]]);
  }
  ierr = PetscFree(perm);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}




