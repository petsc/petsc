#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: gmreig.c,v 1.13 1999/05/12 03:31:46 bsmith Exp balay $";
#endif

#include "src/sles/ksp/impls/gmres/gmresp.h"
#include "pinclude/blaslapack.h"

#undef __FUNC__  
#define __FUNC__ "KSPComputeExtremeSingularValues_GMRES"
int KSPComputeExtremeSingularValues_GMRES(KSP ksp,double *emax,double *emin)
{
  KSP_GMRES *gmres = (KSP_GMRES *) ksp->data;
  int       n = gmres->it + 1, N = gmres->max_k + 2, ierr, lwork = 5*N;
  int       idummy = N, i;
  Scalar    *R = gmres->Rsvd;
  double    *realpart = gmres->Dsvd;
  Scalar    *work = R + N*N, sdummy;

  PetscFunctionBegin;
  if (n == 0) {
    *emax = *emin = 1.0;
    PetscFunctionReturn(0);
  }
  /* copy R matrix to work space */
  ierr = PetscMemcpy(R,gmres->hh_origin,N*N*sizeof(Scalar));CHKERRQ(ierr);

  /* zero below diagonal garbage */
  for ( i=0; i<n; i++ ) {
    R[i*N+i+1] = 0.0;
  }
  
  /* compute Singular Values */
  /*
      The Cray math libraries do not seem to have the DGESVD() lapack routines
  */
#if defined(PETSC_HAVE_MISSING_DGESVD) 
  SETERRQ(PETSC_ERR_SUP,0,"DGESVD not found on Cray T3D\nNot able to provide singular value estimates.");
#else
#if !defined(PETSC_USE_COMPLEX)
  LAgesvd_("N","N",&n,&n,R,&N,realpart,&sdummy,&idummy,&sdummy,&idummy,work,&lwork,&ierr);
#else
  LAgesvd_("N","N",&n,&n,R,&N,realpart,&sdummy,&idummy,&sdummy,&idummy,work,&lwork,realpart+N,&ierr);
#endif
  if (ierr) SETERRQ(PETSC_ERR_LIB,0,"Error in SVD Lapack routine");

  *emin = realpart[n-1];
  *emax = realpart[0];

  PetscFunctionReturn(0);
#endif
}
/* ------------------------------------------------------------------------ */
/* ESSL has a different calling sequence for dgeev() and zgeev() than standard LAPACK */
#if defined(PETSC_HAVE_ESSL)
#undef __FUNC__  
#define __FUNC__ "KSPComputeEigenvalues_GMRES"
int KSPComputeEigenvalues_GMRES(KSP ksp,int nmax,double *r,double *c,int *neig)
{
  KSP_GMRES *gmres = (KSP_GMRES *) ksp->data;
  int       n = gmres->it + 1, N = gmres->max_k + 1, ierr, lwork = 5*N;
  int       idummy = N, i,*perm, clen, zero;
  Scalar    *R = gmres->Rsvd;
  Scalar    *cwork = R + N*N;
  double    *work, *realpart = gmres->Dsvd, *imagpart = realpart + N ;
  Scalar    sdummy;

  PetscFunctionBegin;
  if (nmax < n) SETERRQ(PETSC_ERR_ARG_SIZ,0,"Not enough room in work space r and c for eigenvalues");
  *neig = n;

  if (n == 0) {
    PetscFunctionReturn(0);
  }
  /* copy R matrix to work space */
  ierr = PetscMemcpy(R,gmres->hes_origin,N*N*sizeof(Scalar));CHKERRQ(ierr);

  /* compute eigenvalues */

  /* for ESSL version need really cwork of length N (complex), 2N
     (real); already at least 5N of space has been allocated */

  work     = (double *) PetscMalloc( lwork*sizeof(double) );CHKPTRQ(work);
  zero     = 0;
  LAgeev_(&zero,R,&N,cwork,&sdummy,&idummy,&idummy,&n,work,&lwork);
  ierr = PetscFree(work);CHKERRQ(ierr);

  /* For now we stick with the convention of storing the real and imaginary
     components of evalues separately.  But is this what we really want? */
  perm = (int *) PetscMalloc( n*sizeof(int) );CHKPTRQ(perm);

#if !defined(PETSC_USE_COMPLEX)
  for ( i=0; i<n; i++ ) {
    realpart[i] = cwork[2*i];
    perm[i]     = i;
  }
  ierr = PetscSortDoubleWithPermutation(n,realpart,perm);CHKERRQ(ierr);
  for ( i=0; i<n; i++ ) {
    r[i] = cwork[2*perm[i]];
    c[i] = cwork[2*perm[i]+1];
  }
#else
  for ( i=0; i<n; i++ ) {
    realpart[i] = PetscReal(cwork[i]);
    perm[i]     = i;
  }
  ierr = PetscSortDoubleWithPermutation(n,realpart,perm);CHKERRQ(ierr);
  for ( i=0; i<n; i++ ) {
    r[i] = PetscReal(cwork[perm[i]]);
    c[i] = PetscImaginary(cwork[perm[i]]);
  }
#endif
  ierr = PetscFree(perm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#elif !defined(PETSC_USE_COMPLEX)
#undef __FUNC__  
#define __FUNC__ "KSPComputeEigenvalues_GMRES"
int KSPComputeEigenvalues_GMRES(KSP ksp,int nmax,double *r,double *c,int *neig)
{
  KSP_GMRES *gmres = (KSP_GMRES *) ksp->data;
  int       n = gmres->it + 1, N = gmres->max_k + 1, ierr, lwork = 5*N;
  int       idummy = N, i,*perm;
  Scalar    *R = gmres->Rsvd;
  Scalar    *work = R + N*N;
  Scalar    *realpart = gmres->Dsvd, *imagpart = realpart + N ;
  Scalar    sdummy;

  PetscFunctionBegin;
  if (nmax < n) SETERRQ(PETSC_ERR_ARG_SIZ,0,"Not enough room in work space r and c for eigenvalues");
  *neig = n;

  if (n == 0) {
    PetscFunctionReturn(0);
  }

  /* copy R matrix to work space */
  ierr = PetscMemcpy(R,gmres->hes_origin,N*N*sizeof(Scalar));CHKERRQ(ierr);

  /* compute eigenvalues */
  LAgeev_("N","N",&n,R,&N,realpart,imagpart,&sdummy,&idummy,&sdummy,&idummy,work,&lwork,&ierr);
  if (ierr) SETERRQ(PETSC_ERR_LIB,0,"Error in LAPACK routine");
  perm = (int *) PetscMalloc( n*sizeof(int) );CHKPTRQ(perm);
  for ( i=0; i<n; i++ ) { perm[i] = i;}
  ierr = PetscSortDoubleWithPermutation(n,realpart,perm);CHKERRQ(ierr);
  for ( i=0; i<n; i++ ) {
    r[i] = realpart[perm[i]];
    c[i] = imagpart[perm[i]];
  }
  ierr = PetscFree(perm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#else
#undef __FUNC__  
#define __FUNC__ "KSPComputeEigenvalues_GMRES"
int KSPComputeEigenvalues_GMRES(KSP ksp,int nmax,double *r,double *c,int *neig)
{
  KSP_GMRES *gmres = (KSP_GMRES *) ksp->data;
  int       n = gmres->it + 1, N = gmres->max_k + 1, ierr, lwork = 5*N;
  int       idummy = N, i,*perm;
  Scalar    *R = gmres->Rsvd;
  Scalar    *work = R + N*N;
  Scalar    *eigs = work + 5*N;
  Scalar    sdummy;

  PetscFunctionBegin;
  if (nmax < n) SETERRQ(PETSC_ERR_ARG_SIZ,0,"Not enough room in work space r and c for eigenvalues");
  *neig = n;

  if (n == 0) {
    PetscFunctionReturn(0);
  }
  /* copy R matrix to work space */
  ierr = PetscMemcpy(R,gmres->hes_origin,N*N*sizeof(Scalar));CHKERRQ(ierr);

  /* compute eigenvalues */
  LAgeev_("N","N",&n,R,&N,eigs,&sdummy,&idummy,&sdummy,&idummy,work,&lwork,gmres->Dsvd,&ierr);
  if (ierr) SETERRQ(PETSC_ERR_LIB,0,"Error in LAPACK routine");
  perm = (int *) PetscMalloc( n*sizeof(int) );CHKPTRQ(perm);
  for ( i=0; i<n; i++ ) { perm[i] = i;}
  for ( i=0; i<n; i++ ) { r[i]    = PetscReal(eigs[i]);}
  ierr = PetscSortDoubleWithPermutation(n,r,perm);CHKERRQ(ierr);
  for ( i=0; i<n; i++ ) {
    r[i] = PetscReal(eigs[perm[i]]);
    c[i] = PetscImaginary(eigs[perm[i]]);
  }
  ierr = PetscFree(perm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif




