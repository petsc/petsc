
#include <../src/ksp/ksp/impls/gmres/gmresimpl.h>
#include <petscblaslapack.h>

PetscErrorCode KSPComputeExtremeSingularValues_GMRES(KSP ksp,PetscReal *emax,PetscReal *emin)
{
  KSP_GMRES      *gmres = (KSP_GMRES*)ksp->data;
  PetscInt       n = gmres->it + 1,i,N = gmres->max_k + 2;
  PetscBLASInt   bn, bN,lwork, idummy,lierr;
  PetscScalar    *R        = gmres->Rsvd,*work = R + N*N,sdummy = 0;
  PetscReal      *realpart = gmres->Dsvd;

  PetscFunctionBegin;
  PetscCall(PetscBLASIntCast(n,&bn));
  PetscCall(PetscBLASIntCast(N,&bN));
  PetscCall(PetscBLASIntCast(5*N,&lwork));
  PetscCall(PetscBLASIntCast(N,&idummy));
  if (n <= 0) {
    *emax = *emin = 1.0;
    PetscFunctionReturn(0);
  }
  /* copy R matrix to work space */
  PetscCall(PetscArraycpy(R,gmres->hh_origin,(gmres->max_k+2)*(gmres->max_k+1)));

  /* zero below diagonal garbage */
  for (i=0; i<n; i++) R[i*N+i+1] = 0.0;

  /* compute Singular Values */
  PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
#if !defined(PETSC_USE_COMPLEX)
  PetscCallBLAS("LAPACKgesvd",LAPACKgesvd_("N","N",&bn,&bn,R,&bN,realpart,&sdummy,&idummy,&sdummy,&idummy,work,&lwork,&lierr));
#else
  PetscCallBLAS("LAPACKgesvd",LAPACKgesvd_("N","N",&bn,&bn,R,&bN,realpart,&sdummy,&idummy,&sdummy,&idummy,work,&lwork,realpart+N,&lierr));
#endif
  PetscCheck(!lierr,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in SVD Lapack routine %d",(int)lierr);
  PetscCall(PetscFPTrapPop());

  *emin = realpart[n-1];
  *emax = realpart[0];
  PetscFunctionReturn(0);
}

PetscErrorCode KSPComputeEigenvalues_GMRES(KSP ksp,PetscInt nmax,PetscReal *r,PetscReal *c,PetscInt *neig)
{
#if !defined(PETSC_USE_COMPLEX)
  KSP_GMRES      *gmres = (KSP_GMRES*)ksp->data;
  PetscInt       n = gmres->it + 1,N = gmres->max_k + 1,i,*perm;
  PetscBLASInt   bn, bN, lwork, idummy, lierr = -1;
  PetscScalar    *R        = gmres->Rsvd,*work = R + N*N;
  PetscScalar    *realpart = gmres->Dsvd,*imagpart = realpart + N,sdummy = 0;

  PetscFunctionBegin;
  PetscCall(PetscBLASIntCast(n,&bn));
  PetscCall(PetscBLASIntCast(N,&bN));
  PetscCall(PetscBLASIntCast(5*N,&lwork));
  PetscCall(PetscBLASIntCast(N,&idummy));
  PetscCheck(nmax >= n,PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_SIZ,"Not enough room in work space r and c for eigenvalues");
  *neig = n;

  if (!n) PetscFunctionReturn(0);

  /* copy R matrix to work space */
  PetscCall(PetscArraycpy(R,gmres->hes_origin,N*N));

  /* compute eigenvalues */
  PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
  PetscCallBLAS("LAPACKgeev",LAPACKgeev_("N","N",&bn,R,&bN,realpart,imagpart,&sdummy,&idummy,&sdummy,&idummy,work,&lwork,&lierr));
  PetscCheck(!lierr,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in LAPACK routine %d",(int)lierr);
  PetscCall(PetscFPTrapPop());
  PetscCall(PetscMalloc1(n,&perm));
  for (i=0; i<n; i++) perm[i] = i;
  PetscCall(PetscSortRealWithPermutation(n,realpart,perm));
  for (i=0; i<n; i++) {
    r[i] = realpart[perm[i]];
    c[i] = imagpart[perm[i]];
  }
  PetscCall(PetscFree(perm));
#else
  KSP_GMRES      *gmres = (KSP_GMRES*)ksp->data;
  PetscInt       n  = gmres->it + 1,N = gmres->max_k + 1,i,*perm;
  PetscScalar    *R = gmres->Rsvd,*work = R + N*N,*eigs = work + 5*N,sdummy;
  PetscBLASInt   bn,bN,lwork,idummy,lierr = -1;

  PetscFunctionBegin;
  PetscCall(PetscBLASIntCast(n,&bn));
  PetscCall(PetscBLASIntCast(N,&bN));
  PetscCall(PetscBLASIntCast(5*N,&lwork));
  PetscCall(PetscBLASIntCast(N,&idummy));
  PetscCheck(nmax >= n,PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_SIZ,"Not enough room in work space r and c for eigenvalues");
  *neig = n;

  if (!n) PetscFunctionReturn(0);

  /* copy R matrix to work space */
  PetscCall(PetscArraycpy(R,gmres->hes_origin,N*N));

  /* compute eigenvalues */
  PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
  PetscCallBLAS("LAPACKgeev",LAPACKgeev_("N","N",&bn,R,&bN,eigs,&sdummy,&idummy,&sdummy,&idummy,work,&lwork,gmres->Dsvd,&lierr));
  PetscCheck(!lierr,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in LAPACK routine");
  PetscCall(PetscFPTrapPop());
  PetscCall(PetscMalloc1(n,&perm));
  for (i=0; i<n; i++) perm[i] = i;
  for (i=0; i<n; i++) r[i] = PetscRealPart(eigs[i]);
  PetscCall(PetscSortRealWithPermutation(n,r,perm));
  for (i=0; i<n; i++) {
    r[i] = PetscRealPart(eigs[perm[i]]);
    c[i] = PetscImaginaryPart(eigs[perm[i]]);
  }
  PetscCall(PetscFree(perm));
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode KSPComputeRitz_GMRES(KSP ksp,PetscBool ritz,PetscBool small,PetscInt *nrit,Vec S[],PetscReal *tetar,PetscReal *tetai)
{
  KSP_GMRES      *gmres = (KSP_GMRES*)ksp->data;
  PetscInt       NbrRitz,nb = 0,n;
  PetscInt       i,j,*perm;
  PetscScalar    *H,*Q,*Ht;            /* H Hessenberg matrix; Q matrix of eigenvectors of H */
  PetscScalar    *wr,*wi;              /* Real and imaginary part of the Ritz values */
  PetscScalar    *SR,*work;
  PetscReal      *modul;
  PetscBLASInt   bn,bN,lwork,idummy;
  PetscScalar    *t,sdummy = 0;
  Mat            A;

  PetscFunctionBegin;
  /* Express sizes in PetscBLASInt for LAPACK routines*/
  PetscCall(PetscBLASIntCast(gmres->fullcycle ? gmres->max_k : gmres->it + 1,&bn)); /* size of the Hessenberg matrix */
  PetscCall(PetscBLASIntCast(gmres->max_k + 1,&bN));                                /* LDA of the Hessenberg matrix */
  PetscCall(PetscBLASIntCast(gmres->max_k + 1,&idummy));
  PetscCall(PetscBLASIntCast(5*(gmres->max_k + 1)*(gmres->max_k + 1),&lwork));

  /* NbrRitz: number of (Harmonic) Ritz pairs to extract */
  NbrRitz = PetscMin(*nrit,bn);
  PetscCall(KSPGetOperators(ksp,&A,NULL));
  PetscCall(MatGetSize(A,&n,NULL));
  NbrRitz = PetscMin(NbrRitz,n);

  PetscCall(PetscMalloc4(bN*bN,&H,bn*bn,&Q,bn,&wr,bn,&wi));

  /* copy H matrix to work space */
  PetscCall(PetscArraycpy(H,gmres->fullcycle ? gmres->hes_ritz : gmres->hes_origin,bN*bN));

  /* Modify H to compute Harmonic Ritz pairs H = H + H^{-T}*h^2_{m+1,m}e_m*e_m^T */
  if (!ritz) {
    /* Transpose the Hessenberg matrix => Ht */
    PetscCall(PetscMalloc1(bn*bn,&Ht));
    for (i=0; i<bn; i++) {
      for (j=0; j<bn; j++) {
        Ht[i*bn+j] = PetscConj(H[j*bN+i]);
      }
    }
    /* Solve the system H^T*t = h^2_{m+1,m}e_m */
    PetscCall(PetscCalloc1(bn,&t));
    /* t = h^2_{m+1,m}e_m */
    if (gmres->fullcycle) t[bn-1] = PetscSqr(gmres->hes_ritz[(bn-1)*bN+bn]);
    else t[bn-1] = PetscSqr(gmres->hes_origin[(bn-1)*bN+bn]);

    /* Call the LAPACK routine dgesv to compute t = H^{-T}*t */
    {
      PetscBLASInt info;
      PetscBLASInt nrhs = 1;
      PetscBLASInt *ipiv;
      PetscCall(PetscMalloc1(bn,&ipiv));
      PetscCallBLAS("LAPACKgesv",LAPACKgesv_(&bn,&nrhs,Ht,&bn,ipiv,t,&bn,&info));
      PetscCheck(!info,PetscObjectComm((PetscObject)ksp),PETSC_ERR_PLIB,"Error while calling the Lapack routine DGESV");
      PetscCall(PetscFree(ipiv));
      PetscCall(PetscFree(Ht));
    }
    /* Form H + H^{-T}*h^2_{m+1,m}e_m*e_m^T */
    for (i=0; i<bn; i++) H[(bn-1)*bn+i] += t[i];
    PetscCall(PetscFree(t));
  }

  /*
    Compute (Harmonic) Ritz pairs;
    For a real Ritz eigenvector at wr(j)  Q(:,j) columns contain the real right eigenvector
    For a complex Ritz pair of eigenvectors at wr(j), wi(j), wr(j+1), and wi(j+1), Q(:,j) + i Q(:,j+1) and Q(:,j) - i Q(:,j+1) are the two eigenvectors
  */
  {
    PetscBLASInt info;
#if defined(PETSC_USE_COMPLEX)
    PetscReal    *rwork=NULL;
#endif
    PetscCall(PetscMalloc1(lwork,&work));
    PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
#if !defined(PETSC_USE_COMPLEX)
    PetscCallBLAS("LAPACKgeev",LAPACKgeev_("N","V",&bn,H,&bN,wr,wi,&sdummy,&idummy,Q,&bn,work,&lwork,&info));
#else
    PetscCall(PetscMalloc1(2*n,&rwork));
    PetscCallBLAS("LAPACKgeev",LAPACKgeev_("N","V",&bn,H,&bN,wr,&sdummy,&idummy,Q,&bn,work,&lwork,rwork,&info));
    PetscCall(PetscFree(rwork));
#endif
    PetscCheck(!info,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in LAPACK routine");
    PetscCall(PetscFPTrapPop());
    PetscCall(PetscFree(work));
  }
  /* sort the (Harmonic) Ritz values */
  PetscCall(PetscMalloc2(bn,&modul,bn,&perm));
#if defined(PETSC_USE_COMPLEX)
  for (i=0; i<bn; i++) modul[i] = PetscAbsScalar(wr[i]);
#else
  for (i=0; i<bn; i++) modul[i] = PetscSqrtReal(wr[i]*wr[i]+wi[i]*wi[i]);
#endif
  for (i=0; i<bn; i++) perm[i] = i;
  PetscCall(PetscSortRealWithPermutation(bn,modul,perm));

#if defined(PETSC_USE_COMPLEX)
  /* sort extracted (Harmonic) Ritz pairs */
  nb = NbrRitz;
  PetscCall(PetscMalloc1(nb*bn,&SR));
  for (i=0; i<nb; i++) {
    if (small) {
      tetar[i] = PetscRealPart(wr[perm[i]]);
      tetai[i] = PetscImaginaryPart(wr[perm[i]]);
      PetscCall(PetscArraycpy(&SR[i*bn],&(Q[perm[i]*bn]),bn));
    } else {
      tetar[i] = PetscRealPart(wr[perm[bn-nb+i]]);
      tetai[i] = PetscImaginaryPart(wr[perm[bn-nb+i]]);
      PetscCall(PetscArraycpy(&SR[i*bn],&(Q[perm[bn-nb+i]*bn]),bn)); /* permute columns of Q */
    }
  }
#else
  /* count the number of extracted (Harmonic) Ritz pairs (with complex conjugates) */
  if (small) {
    while (nb < NbrRitz) {
      if (!wi[perm[nb]]) nb += 1;
      else {
        if (nb < NbrRitz - 1) nb += 2;
        else break;
      }
    }
    PetscCall(PetscMalloc1(nb*bn,&SR));
    for (i=0; i<nb; i++) {
      tetar[i] = wr[perm[i]];
      tetai[i] = wi[perm[i]];
      PetscCall(PetscArraycpy(&SR[i*bn],&(Q[perm[i]*bn]),bn));
    }
  } else {
    while (nb < NbrRitz) {
      if (wi[perm[bn-nb-1]] == 0) nb += 1;
      else {
        if (nb < NbrRitz - 1) nb += 2;
        else break;
      }
    }
    PetscCall(PetscMalloc1(nb*bn,&SR)); /* bn rows, nb columns */
    for (i=0; i<nb; i++) {
      tetar[i] = wr[perm[bn-nb+i]];
      tetai[i] = wi[perm[bn-nb+i]];
      PetscCall(PetscArraycpy(&SR[i*bn], &(Q[perm[bn-nb+i]*bn]), bn)); /* permute columns of Q */
    }
  }
#endif
  PetscCall(PetscFree2(modul,perm));
  PetscCall(PetscFree4(H,Q,wr,wi));

  /* Form the (Harmonic) Ritz vectors S = SR*V, columns of VV correspond to the basis of the Krylov subspace */
  for (j=0; j<nb; j++) {
    PetscCall(VecZeroEntries(S[j]));
    PetscCall(VecMAXPY(S[j],bn,&SR[j*bn],gmres->fullcycle ? gmres->vecb : &VEC_VV(0)));
  }

  PetscCall(PetscFree(SR));
  *nrit = nb;
  PetscFunctionReturn(0);
}
