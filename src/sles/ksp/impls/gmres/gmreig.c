#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: gmreig.c,v 1.4 1997/09/11 02:56:33 curfman Exp bsmith $";
#endif

#include "src/ksp/impls/gmres/gmresp.h"
#include <math.h>
#include "pinclude/plapack.h"

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
  PetscMemcpy(R,gmres->hh_origin,N*N*sizeof(Scalar));

  /* zero below diagonal garbage */
  for ( i=0; i<n; i++ ) {
    R[i*N+i+1] = 0.0;
  }
  
  /* compute Singular Values */
#if defined(PARCH_t3d)
  SETERRQ(1,0,"DGESVD not found on Cray T3D\n\
             Therefore not able to provide singular value estimates.");
#else
#if !defined(USE_PETSC_COMPLEX)
  LAgesvd_("N","N",&n,&n,R,&N,realpart,&sdummy,&idummy,&sdummy,&idummy,work,&lwork,&ierr);
#else
  LAgesvd_("N","N",&n,&n,R,&N,realpart,&sdummy,&idummy,&sdummy,&idummy,work,&lwork,realpart+N,&ierr);
#endif
  if (ierr) SETERRQ(1,0,"Error in SVD");

  *emin = realpart[n-1];
  *emax = realpart[0];

  PetscFunctionReturn(0);
#endif
}
/* ------------------------------------------------------------------------ */
/* ESSL has a different calling sequence for dgeev() and zgeev() than standard LAPACK */
#if defined(HAVE_ESSL)
#undef __FUNC__  
#define __FUNC__ "KSPComputeEigenvalues_GMRES"
int KSPComputeEigenvalues_GMRES(KSP ksp,int nmax,double *r,double *c)
{
  KSP_GMRES *gmres = (KSP_GMRES *) ksp->data;
  int       n = gmres->it + 1, N = gmres->max_k + 1, ierr, lwork = 5*N;
  int       idummy = N, i,*perm, clen, zero;
  Scalar    *R = gmres->Rsvd;
  Scalar    *cwork = R + N*N;
  double    *work, *realpart = gmres->Dsvd, *imagpart = realpart + N ;
  Scalar    sdummy;

  PetscFunctionBegin;
  if (nmax < n) SETERRQ(1,0,"Not enough room in r and c for eigenvalues");

  if (n == 0) {
    PetscFunctionReturn(0);
  }
  /* copy R matrix to work space */
  PetscMemcpy(R,gmres->hes_origin,N*N*sizeof(Scalar));

  /* compute eigenvalues */

  /* for ESSL version need really cwork of length N (complex), 2N
     (real); already at least 5N of space has been allocated */

  work     = (double *) PetscMalloc( lwork*sizeof(double) ); CHKPTRQ(work);
  zero     = 0;
  LAgeev_(&zero,R,&N,cwork,&sdummy,&idummy,&idummy,&n,work,&lwork);
  PetscFree(work);

  /* For now we stick with the convention of storing the real and imaginary
     components of evalues separately.  But is this what we really want? */
  perm = (int *) PetscMalloc( n*sizeof(int) ); CHKPTRQ(perm);

#if !defined(USE_PETSC_COMPLEX)
  for ( i=0; i<n; i++ ) {
    realpart[i] = cwork[2*i];
    perm[i]     = i;
  }
  ierr = PetscSortDoubleWithPermutation(n,realpart,perm); CHKERRQ(ierr);
  for ( i=0; i<n; i++ ) {
    r[i] = cwork[2*perm[i]];
    c[i] = cwork[2*perm[i]+1];
  }
#else
  for ( i=0; i<n; i++ ) {
    realpart[i] = PetscReal(cwork[i]);
    perm[i]     = i;
  }
  ierr = PetscSortDoubleWithPermutation(n,realpart,perm); CHKERRQ(ierr);
  for ( i=0; i<n; i++ ) {
    r[i] = PetscReal(cwork[perm[i]]);
    c[i] = PetscImaginary(cwork[perm[i]]);
  }
#endif
  PetscFree(perm);
  PetscFunctionReturn(0);
}
#elif !defined(USE_PETSC_COMPLEX)
#undef __FUNC__  
#define __FUNC__ "KSPComputeEigenvalues_GMRES"
int KSPComputeEigenvalues_GMRES(KSP ksp,int nmax,double *r,double *c)
{
  KSP_GMRES *gmres = (KSP_GMRES *) ksp->data;
  int       n = gmres->it + 1, N = gmres->max_k + 1, ierr, lwork = 5*N;
  int       idummy = N, i,*perm;
  Scalar    *R = gmres->Rsvd;
  Scalar    *work = R + N*N;
  Scalar    *realpart = gmres->Dsvd, *imagpart = realpart + N ;
  Scalar    sdummy;

  PetscFunctionBegin;
  if (nmax < n) SETERRQ(1,0,"Not enough room in r and c for eigenvalues");

  if (n == 0) {
    PetscFunctionReturn(0);
  }
  /* copy R matrix to work space */
  PetscMemcpy(R,gmres->hes_origin,N*N*sizeof(Scalar));

  /* compute eigenvalues */
  LAgeev_("N","N",&n,R,&N,realpart,imagpart,&sdummy,&idummy,&sdummy,&idummy,work,&lwork,&ierr);
  if (ierr) SETERRQ(1,0,"Error in LAPACK routine");
  perm = (int *) PetscMalloc( n*sizeof(int) ); CHKPTRQ(perm);
  for ( i=0; i<n; i++ ) { perm[i] = i;}
  ierr = PetscSortDoubleWithPermutation(n,realpart,perm); CHKERRQ(ierr);
  for ( i=0; i<n; i++ ) {
    r[i] = realpart[perm[i]];
    c[i] = imagpart[perm[i]];
  }
  PetscFree(perm);
  PetscFunctionReturn(0);
}
#else
#undef __FUNC__  
#define __FUNC__ "KSPComputeEigenvalues_GMRES"
int KSPComputeEigenvalues_GMRES(KSP ksp,int nmax,double *r,double *c)
{
  KSP_GMRES *gmres = (KSP_GMRES *) ksp->data;
  int       n = gmres->it + 1, N = gmres->max_k + 1, ierr, lwork = 5*N;
  int       idummy = N, i,*perm;
  Scalar    *R = gmres->Rsvd;
  Scalar    *work = R + N*N;
  Scalar    *eigs = work + 5*N;
  Scalar    sdummy;

  PetscFunctionBegin;
  if (nmax < n) SETERRQ(1,0,"Not enough room in r and c for eigenvalues");

  if (n == 0) {
    PetscFunctionReturn(0);
  }
  /* copy R matrix to work space */
  PetscMemcpy(R,gmres->hes_origin,N*N*sizeof(Scalar));

  /* compute eigenvalues */
  LAgeev_("N","N",&n,R,&N,eigs,&sdummy,&idummy,&sdummy,&idummy,work,&lwork,gmres->Dsvd,&ierr);
  if (ierr) SETERRQ(1,0,"Error in LAPACK routine");
  perm = (int *) PetscMalloc( n*sizeof(int) ); CHKPTRQ(perm);
  for ( i=0; i<n; i++ ) { perm[i] = i;}
  for ( i=0; i<n; i++ ) { r[i]    = real(eigs[i]);}
  ierr = PetscSortDoubleWithPermutation(n,r,perm); CHKERRQ(ierr);
  for ( i=0; i<n; i++ ) {
    r[i] = PetscReal(eigs[perm[i]]);
    c[i] = PetscImaginary(eigs[perm[i]]);
  }
  PetscFree(perm);
  PetscFunctionReturn(0);
}
#endif




