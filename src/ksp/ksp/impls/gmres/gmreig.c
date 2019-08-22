
#include <../src/ksp/ksp/impls/gmres/gmresimpl.h>
#include <petscblaslapack.h>

PetscErrorCode KSPComputeExtremeSingularValues_GMRES(KSP ksp,PetscReal *emax,PetscReal *emin)
{
#if defined(PETSC_MISSING_LAPACK_GESVD)
  PetscFunctionBegin;
  /*
      The Cray math libraries on T3D/T3E, and early versions of Intel Math Kernel Libraries (MKL)
      for PCs do not seem to have the DGESVD() lapack routines
  */
  SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"GESVD - Lapack routine is unavailable\nNot able to provide singular value estimates.");
#else
  KSP_GMRES      *gmres = (KSP_GMRES*)ksp->data;
  PetscErrorCode ierr;
  PetscInt       n = gmres->it + 1,i,N = gmres->max_k + 2;
  PetscBLASInt   bn, bN,lwork, idummy,lierr;
  PetscScalar    *R        = gmres->Rsvd,*work = R + N*N,sdummy = 0;
  PetscReal      *realpart = gmres->Dsvd;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(n,&bn);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(N,&bN);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(5*N,&lwork);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(N,&idummy);CHKERRQ(ierr);
  if (n <= 0) {
    *emax = *emin = 1.0;
    PetscFunctionReturn(0);
  }
  /* copy R matrix to work space */
  ierr = PetscArraycpy(R,gmres->hh_origin,(gmres->max_k+2)*(gmres->max_k+1));CHKERRQ(ierr);

  /* zero below diagonal garbage */
  for (i=0; i<n; i++) R[i*N+i+1] = 0.0;

  /* compute Singular Values */
  ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  PetscStackCallBLAS("LAPACKgesvd",LAPACKgesvd_("N","N",&bn,&bn,R,&bN,realpart,&sdummy,&idummy,&sdummy,&idummy,work,&lwork,&lierr));
#else
  PetscStackCallBLAS("LAPACKgesvd",LAPACKgesvd_("N","N",&bn,&bn,R,&bN,realpart,&sdummy,&idummy,&sdummy,&idummy,work,&lwork,realpart+N,&lierr));
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
PetscErrorCode KSPComputeEigenvalues_GMRES(KSP ksp,PetscInt nmax,PetscReal *r,PetscReal *c,PetscInt *neig)
{
#if defined(PETSC_HAVE_ESSL)
  KSP_GMRES      *gmres = (KSP_GMRES*)ksp->data;
  PetscErrorCode ierr;
  PetscInt       n = gmres->it + 1,N = gmres->max_k + 1;
  PetscInt       i,*perm;
  PetscScalar    *R     = gmres->Rsvd;
  PetscScalar    *cwork = R + N*N,sdummy = 0;
  PetscReal      *work,*realpart = gmres->Dsvd;
  PetscBLASInt   zero = 0,bn,bN,idummy = -1,lwork;

  PetscFunctionBegin;
  ierr   = PetscBLASIntCast(n,&bn);CHKERRQ(ierr);
  ierr   = PetscBLASIntCast(N,&bN);CHKERRQ(ierr);
  ierr   = PetscBLASIntCast(5*N,&lwork);CHKERRQ(ierr);
  if (nmax < n) SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_SIZ,"Not enough room in work space r and c for eigenvalues");
  *neig = n;

  if (!n) PetscFunctionReturn(0);

  /* copy R matrix to work space */
  ierr = PetscArraycpy(R,gmres->hes_origin,N*N);CHKERRQ(ierr);

  /* compute eigenvalues */

  /* for ESSL version need really cwork of length N (complex), 2N
     (real); already at least 5N of space has been allocated */

  ierr = PetscMalloc1(lwork,&work);CHKERRQ(ierr);
  ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
  PetscStackCallBLAS("LAPACKgeev",LAPACKgeev_(&zero,R,&bN,cwork,&sdummy,&idummy,&idummy,&bn,work,&lwork));
  ierr = PetscFPTrapPop();CHKERRQ(ierr);
  ierr = PetscFree(work);CHKERRQ(ierr);

  /* For now we stick with the convention of storing the real and imaginary
     components of evalues separately.  But is this what we really want? */
  ierr = PetscMalloc1(n,&perm);CHKERRQ(ierr);

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
  SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"GEEV - Lapack routine is unavailable\nNot able to provide eigen values.");
#elif !defined(PETSC_USE_COMPLEX)
  KSP_GMRES      *gmres = (KSP_GMRES*)ksp->data;
  PetscErrorCode ierr;
  PetscInt       n = gmres->it + 1,N = gmres->max_k + 1,i,*perm;
  PetscBLASInt   bn, bN, lwork, idummy, lierr = -1;
  PetscScalar    *R        = gmres->Rsvd,*work = R + N*N;
  PetscScalar    *realpart = gmres->Dsvd,*imagpart = realpart + N,sdummy = 0;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(n,&bn);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(N,&bN);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(5*N,&lwork);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(N,&idummy);CHKERRQ(ierr);
  if (nmax < n) SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_SIZ,"Not enough room in work space r and c for eigenvalues");
  *neig = n;

  if (!n) PetscFunctionReturn(0);

  /* copy R matrix to work space */
  ierr = PetscArraycpy(R,gmres->hes_origin,N*N);CHKERRQ(ierr);

  /* compute eigenvalues */
  ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
  PetscStackCallBLAS("LAPACKgeev",LAPACKgeev_("N","N",&bn,R,&bN,realpart,imagpart,&sdummy,&idummy,&sdummy,&idummy,work,&lwork,&lierr));
  if (lierr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in LAPACK routine %d",(int)lierr);
  ierr = PetscFPTrapPop();CHKERRQ(ierr);
  ierr = PetscMalloc1(n,&perm);CHKERRQ(ierr);
  for (i=0; i<n; i++) perm[i] = i;
  ierr = PetscSortRealWithPermutation(n,realpart,perm);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    r[i] = realpart[perm[i]];
    c[i] = imagpart[perm[i]];
  }
  ierr = PetscFree(perm);CHKERRQ(ierr);
#else
  KSP_GMRES      *gmres = (KSP_GMRES*)ksp->data;
  PetscErrorCode ierr;
  PetscInt       n  = gmres->it + 1,N = gmres->max_k + 1,i,*perm;
  PetscScalar    *R = gmres->Rsvd,*work = R + N*N,*eigs = work + 5*N,sdummy;
  PetscBLASInt   bn,bN,lwork,idummy,lierr = -1;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(n,&bn);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(N,&bN);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(5*N,&lwork);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(N,&idummy);CHKERRQ(ierr);
  if (nmax < n) SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_SIZ,"Not enough room in work space r and c for eigenvalues");
  *neig = n;

  if (!n) PetscFunctionReturn(0);

  /* copy R matrix to work space */
  ierr = PetscArraycpy(R,gmres->hes_origin,N*N);CHKERRQ(ierr);

  /* compute eigenvalues */
  ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
  PetscStackCallBLAS("LAPACKgeev",LAPACKgeev_("N","N",&bn,R,&bN,eigs,&sdummy,&idummy,&sdummy,&idummy,work,&lwork,gmres->Dsvd,&lierr));
  if (lierr) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in LAPACK routine");
  ierr = PetscFPTrapPop();CHKERRQ(ierr);
  ierr = PetscMalloc1(n,&perm);CHKERRQ(ierr);
  for (i=0; i<n; i++) perm[i] = i;
  for (i=0; i<n; i++) r[i] = PetscRealPart(eigs[i]);
  ierr = PetscSortRealWithPermutation(n,r,perm);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    r[i] = PetscRealPart(eigs[perm[i]]);
    c[i] = PetscImaginaryPart(eigs[perm[i]]);
  }
  ierr = PetscFree(perm);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

#if !defined(PETSC_USE_COMPLEX) && !defined(PETSC_HAVE_ESSL)
PetscErrorCode KSPComputeRitz_GMRES(KSP ksp,PetscBool ritz,PetscBool small,PetscInt *nrit,Vec S[],PetscReal *tetar,PetscReal *tetai)
{
  KSP_GMRES      *gmres = (KSP_GMRES*)ksp->data;
  PetscErrorCode ierr;
  PetscInt       n = gmres->it + 1,N = gmres->max_k + 1,NbrRitz,nb=0;
  PetscInt       i,j,*perm;
  PetscReal      *H,*Q,*Ht;              /* H Hessenberg Matrix and Q matrix of eigenvectors of H*/
  PetscReal      *wr,*wi,*modul;       /* Real and imaginary part and modul of the Ritz values*/
  PetscReal      *SR,*work;
  PetscBLASInt   bn,bN,lwork,idummy;
  PetscScalar    *t,sdummy = 0;

  PetscFunctionBegin;
  /* n: size of the Hessenberg matrix */
  if (gmres->fullcycle) n = N-1;
  /* NbrRitz: number of (harmonic) Ritz pairs to extract */ 
  NbrRitz = PetscMin(*nrit,n);

  /* Definition of PetscBLASInt for lapack routines*/
  ierr = PetscBLASIntCast(n,&bn);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(N,&bN);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(N,&idummy);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(5*N,&lwork);CHKERRQ(ierr);
  /* Memory allocation */
  ierr = PetscMalloc1(bN*bN,&H);CHKERRQ(ierr);
  ierr = PetscMalloc1(bn*bn,&Q);CHKERRQ(ierr);
  ierr = PetscMalloc1(lwork,&work);CHKERRQ(ierr);
  ierr = PetscMalloc1(n,&wr);CHKERRQ(ierr);
  ierr = PetscMalloc1(n,&wi);CHKERRQ(ierr);

  /* copy H matrix to work space */
  if (gmres->fullcycle) {
    ierr = PetscArraycpy(H,gmres->hes_ritz,bN*bN);CHKERRQ(ierr);
  } else {
    ierr = PetscArraycpy(H,gmres->hes_origin,bN*bN);CHKERRQ(ierr);
  }

  /* Modify H to compute Harmonic Ritz pairs H = H + H^{-T}*h^2_{m+1,m}e_m*e_m^T */
  if (!ritz) {
    /* Transpose the Hessenberg matrix => Ht */
    ierr = PetscMalloc1(bn*bn,&Ht);CHKERRQ(ierr);
    for (i=0; i<bn; i++) {
      for (j=0; j<bn; j++) {
        Ht[i*bn+j] = H[j*bN+i];
      }
    }
    /* Solve the system H^T*t = h^2_{m+1,m}e_m */
    ierr = PetscCalloc1(bn,&t);CHKERRQ(ierr);
    /* t = h^2_{m+1,m}e_m */
    if (gmres->fullcycle) {
      t[bn-1] = PetscSqr(gmres->hes_ritz[(bn-1)*bN+bn]); 
    } else {
      t[bn-1] = PetscSqr(gmres->hes_origin[(bn-1)*bN+bn]); 
    }
    /* Call the LAPACK routine dgesv to compute t = H^{-T}*t */
#if   defined(PETSC_MISSING_LAPACK_GESV)
    SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"GESV - Lapack routine is unavailable.");
#else
    {
      PetscBLASInt info;
      PetscBLASInt nrhs = 1;
      PetscBLASInt *ipiv;
      ierr = PetscMalloc1(bn,&ipiv);CHKERRQ(ierr);
      PetscStackCallBLAS("LAPACKgesv",LAPACKgesv_(&bn,&nrhs,Ht,&bn,ipiv,t,&bn,&info));
      if (info) SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_PLIB,"Error while calling the Lapack routine DGESV");
      ierr = PetscFree(ipiv);CHKERRQ(ierr);
      ierr = PetscFree(Ht);CHKERRQ(ierr);
    }
#endif
    /* Now form H + H^{-T}*h^2_{m+1,m}e_m*e_m^T */
    for (i=0; i<bn; i++) H[(bn-1)*bn+i] += t[i];
    ierr = PetscFree(t);CHKERRQ(ierr);
  }

  /* Compute (harmonic) Ritz pairs */
#if defined(PETSC_MISSING_LAPACK_HSEQR)
  SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"GEEV - Lapack routine is unavailable\nNot able to provide eigen values.");
#else
  {
    PetscBLASInt info;
    ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
    PetscStackCallBLAS("LAPACKgeev",LAPACKgeev_("N","V",&bn,H,&bN,wr,wi,&sdummy,&idummy,Q,&bn,work,&lwork,&info));
    if (info) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in LAPACK routine");
  }
#endif
  /* sort the (harmonic) Ritz values */
  ierr = PetscMalloc1(n,&modul);CHKERRQ(ierr);
  ierr = PetscMalloc1(n,&perm);CHKERRQ(ierr);
  for (i=0; i<n; i++) modul[i] = PetscSqrtReal(wr[i]*wr[i]+wi[i]*wi[i]);
  for (i=0; i<n; i++) perm[i] = i;
  ierr = PetscSortRealWithPermutation(n,modul,perm);CHKERRQ(ierr);
  /* count the number of extracted Ritz or Harmonic Ritz pairs (with complex conjugates) */
  if (small) {
    while (nb < NbrRitz) {
      if (!wi[perm[nb]]) nb += 1;
      else nb += 2;
    }
    ierr = PetscMalloc1(nb*n,&SR);CHKERRQ(ierr);
    for (i=0; i<nb; i++) {
      tetar[i] = wr[perm[i]];
      tetai[i] = wi[perm[i]];
      ierr = PetscArraycpy(&SR[i*n],&(Q[perm[i]*bn]),n);CHKERRQ(ierr);
    }
  } else {
    while (nb < NbrRitz) {
      if (wi[perm[n-nb-1]] == 0) nb += 1;
      else nb += 2;
    }
    ierr = PetscMalloc1(nb*n,&SR);CHKERRQ(ierr);
    for (i=0; i<nb; i++) {
      tetar[i] = wr[perm[n-nb+i]];
      tetai[i] = wi[perm[n-nb+i]];
      ierr = PetscArraycpy(&SR[i*n], &(Q[perm[n-nb+i]*bn]), n);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(modul);CHKERRQ(ierr);
  ierr = PetscFree(perm);CHKERRQ(ierr);  

  /* Form the Ritz or Harmonic Ritz vectors S=VV*Sr, 
    where the columns of VV correspond to the basis of the Krylov subspace */
  if (gmres->fullcycle) {
    for (j=0; j<nb; j++) {
      ierr = VecZeroEntries(S[j]);CHKERRQ(ierr);  
      ierr = VecMAXPY(S[j],n,&SR[j*n],gmres->vecb);CHKERRQ(ierr); 
    } 
  } else {
    for (j=0; j<nb; j++) {
      ierr = VecZeroEntries(S[j]);CHKERRQ(ierr);  
      ierr = VecMAXPY(S[j],n,&SR[j*n],&VEC_VV(0));CHKERRQ(ierr);
    }
  }
  *nrit = nb;
  ierr  = PetscFree(H);CHKERRQ(ierr);
  ierr  = PetscFree(Q);CHKERRQ(ierr);
  ierr  = PetscFree(SR);CHKERRQ(ierr);
  ierr  = PetscFree(wr);CHKERRQ(ierr);
  ierr  = PetscFree(wi);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif


