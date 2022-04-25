#include <petsc/private/kspimpl.h> /*I "petscksp.h" I*/
#include <petsc/private/matimpl.h>
#include <petscblaslapack.h>
static PetscBool  cited = PETSC_FALSE;
static const char citation[] =
"@phdthesis{zampini2010non,\n"
"  title={Non-overlapping Domain Decomposition Methods for Cardiac Reaction-Diffusion Models and Applications},\n"
"  author={Zampini, S},\n"
"  year={2010},\n"
"  school={PhD thesis, Universita degli Studi di Milano}\n"
"}\n";

typedef struct {
  PetscInt     maxn;             /* maximum number of snapshots */
  PetscInt     n;                /* number of active snapshots */
  PetscInt     curr;             /* current tip of snapshots set */
  Vec          *xsnap;           /* snapshots */
  Vec          *bsnap;           /* rhs snapshots */
  Vec          *work;            /* parallel work vectors */
  PetscScalar  *dots_iallreduce;
  MPI_Request  req_iallreduce;
  PetscInt     ndots_iallreduce; /* if we have iallreduce we can hide the VecMDot communications */
  PetscReal    tol;              /* relative tolerance to retain eigenvalues */
  PetscBool    Aspd;             /* if true, uses the SPD operator as inner product */
  PetscScalar  *corr;            /* correlation matrix */
  PetscReal    *eigs;            /* eigenvalues */
  PetscScalar  *eigv;            /* eigenvectors */
  PetscBLASInt nen;              /* dimension of lower dimensional system */
  PetscInt     st;               /* first eigenvector of correlation matrix to be retained */
  PetscBLASInt *iwork;           /* integer work vector */
  PetscScalar  *yhay;            /* Y^H * A * Y */
  PetscScalar  *low;             /* lower dimensional linear system */
#if defined(PETSC_USE_COMPLEX)
  PetscReal    *rwork;
#endif
  PetscBLASInt lwork;
  PetscScalar  *swork;
  PetscBool    monitor;
} KSPGuessPOD;

static PetscErrorCode KSPGuessReset_POD(KSPGuess guess)
{
  KSPGuessPOD    *pod = (KSPGuessPOD*)guess->data;
  PetscLayout    Alay = NULL,vlay = NULL;
  PetscBool      cong;

  PetscFunctionBegin;
  pod->nen  = 0;
  pod->n    = 0;
  pod->curr = 0;
  /* need to wait for completion of outstanding requests */
  if (pod->ndots_iallreduce) {
    PetscCallMPI(MPI_Wait(&pod->req_iallreduce,MPI_STATUS_IGNORE));
  }
  pod->ndots_iallreduce = 0;
  /* destroy vectors if the size of the linear system has changed */
  if (guess->A) {
    PetscCall(MatGetLayouts(guess->A,&Alay,NULL));
  }
  if (pod->xsnap) {
    PetscCall(VecGetLayout(pod->xsnap[0],&vlay));
  }
  cong = PETSC_FALSE;
  if (vlay && Alay) {
    PetscCall(PetscLayoutCompare(Alay,vlay,&cong));
  }
  if (!cong) {
    PetscCall(VecDestroyVecs(pod->maxn,&pod->xsnap));
    PetscCall(VecDestroyVecs(pod->maxn,&pod->bsnap));
    PetscCall(VecDestroyVecs(1,&pod->work));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPGuessSetUp_POD(KSPGuess guess)
{
  KSPGuessPOD    *pod = (KSPGuessPOD*)guess->data;

  PetscFunctionBegin;
  if (!pod->corr) {
    PetscScalar  sdummy;
    PetscReal    rdummy = 0;
    PetscBLASInt bN,lierr,idummy;

    PetscCall(PetscCalloc6(pod->maxn*pod->maxn,&pod->corr,pod->maxn,&pod->eigs,pod->maxn*pod->maxn,&pod->eigv,6*pod->maxn,&pod->iwork,pod->maxn*pod->maxn,&pod->yhay,pod->maxn*pod->maxn,&pod->low));
#if defined(PETSC_USE_COMPLEX)
    PetscCall(PetscMalloc1(7*pod->maxn,&pod->rwork));
#endif
#if defined(PETSC_HAVE_MPI_NONBLOCKING_COLLECTIVES)
    PetscCall(PetscMalloc1(3*pod->maxn,&pod->dots_iallreduce));
#endif
    pod->lwork = -1;
    PetscCall(PetscBLASIntCast(pod->maxn,&bN));
#if !defined(PETSC_USE_COMPLEX)
    PetscStackCallBLAS("LAPACKsyevx",LAPACKsyevx_("V","A","L",&bN,pod->corr,&bN,&rdummy,&rdummy,&idummy,&idummy,
                                                  &rdummy,&idummy,pod->eigs,pod->eigv,&bN,&sdummy,&pod->lwork,pod->iwork,pod->iwork+5*bN,&lierr));
#else
    PetscStackCallBLAS("LAPACKsyevx",LAPACKsyevx_("V","A","L",&bN,pod->corr,&bN,&rdummy,&rdummy,&idummy,&idummy,
                                                  &rdummy,&idummy,pod->eigs,pod->eigv,&bN,&sdummy,&pod->lwork,pod->rwork,pod->iwork,pod->iwork+5*bN,&lierr));
#endif
    PetscCheck(!lierr,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in query to SYEV Lapack routine %d",(int)lierr);
    PetscCall(PetscBLASIntCast((PetscInt)PetscRealPart(sdummy),&pod->lwork));
    PetscCall(PetscMalloc1(pod->lwork+PetscMax(bN*bN,6*bN),&pod->swork));
  }
  /* work vectors are sequential, we explicitly use MPI_Allreduce */
  if (!pod->xsnap) {
    VecType   type;
    Vec       *v,vseq;
    PetscInt  n;

    PetscCall(KSPCreateVecs(guess->ksp,1,&v,0,NULL));
    PetscCall(VecCreate(PETSC_COMM_SELF,&vseq));
    PetscCall(VecGetLocalSize(v[0],&n));
    PetscCall(VecSetSizes(vseq,n,n));
    PetscCall(VecGetType(v[0],&type));
    PetscCall(VecSetType(vseq,type));
    PetscCall(VecDestroyVecs(1,&v));
    PetscCall(VecDuplicateVecs(vseq,pod->maxn,&pod->xsnap));
    PetscCall(VecDestroy(&vseq));
    PetscCall(PetscLogObjectParents(guess,pod->maxn,pod->xsnap));
  }
  if (!pod->bsnap) {
    PetscCall(VecDuplicateVecs(pod->xsnap[0],pod->maxn,&pod->bsnap));
    PetscCall(PetscLogObjectParents(guess,pod->maxn,pod->bsnap));
  }
  if (!pod->work) {
    PetscCall(KSPCreateVecs(guess->ksp,1,&pod->work,0,NULL));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPGuessDestroy_POD(KSPGuess guess)
{
  KSPGuessPOD *pod = (KSPGuessPOD*)guess->data;

  PetscFunctionBegin;
  PetscCall(PetscFree6(pod->corr,pod->eigs,pod->eigv,pod->iwork,pod->yhay,pod->low));
#if defined(PETSC_USE_COMPLEX)
  PetscCall(PetscFree(pod->rwork));
#endif
  /* need to wait for completion before destroying dots_iallreduce */
  if (pod->ndots_iallreduce) {
    PetscCallMPI(MPI_Wait(&pod->req_iallreduce,MPI_STATUS_IGNORE));
  }
  PetscCall(PetscFree(pod->dots_iallreduce));
  PetscCall(PetscFree(pod->swork));
  PetscCall(VecDestroyVecs(pod->maxn,&pod->bsnap));
  PetscCall(VecDestroyVecs(pod->maxn,&pod->xsnap));
  PetscCall(VecDestroyVecs(1,&pod->work));
  PetscCall(PetscFree(pod));
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPGuessUpdate_POD(KSPGuess,Vec,Vec);

static PetscErrorCode KSPGuessFormGuess_POD(KSPGuess guess,Vec b,Vec x)
{
  KSPGuessPOD    *pod = (KSPGuessPOD*)guess->data;
  PetscScalar    one = 1, zero = 0, *array;
  PetscBLASInt   bN,ione = 1,bNen,lierr;
  PetscInt       i;

  PetscFunctionBegin;
  PetscCall(PetscCitationsRegister(citation,&cited));
  if (pod->ndots_iallreduce) { /* complete communication and project the linear system */
    PetscCall(KSPGuessUpdate_POD(guess,NULL,NULL));
  }
  if (!pod->nen) PetscFunctionReturn(0);
  /* b_low = S * V^T * X^T * b */
  PetscCall(VecGetArrayRead(b,(const PetscScalar**)&array));
  PetscCall(VecPlaceArray(pod->bsnap[pod->curr],array));
  PetscCall(VecRestoreArrayRead(b,(const PetscScalar**)&array));
  PetscCall(VecMDot(pod->bsnap[pod->curr],pod->n,pod->xsnap,pod->swork));
  PetscCall(VecResetArray(pod->bsnap[pod->curr]));
  PetscCall(MPIU_Allreduce(pod->swork,pod->swork + pod->n,pod->n,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)guess)));
  PetscCall(PetscBLASIntCast(pod->n,&bN));
  PetscCall(PetscBLASIntCast(pod->nen,&bNen));
  PetscStackCallBLAS("BLASgemv",BLASgemv_("T",&bN,&bNen,&one,pod->eigv+pod->st*pod->n,&bN,pod->swork+pod->n,&ione,&zero,pod->swork,&ione));
  if (pod->monitor) {
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)guess),"  KSPGuessPOD alphas = "));
    for (i=0; i<pod->nen; i++) {
#if defined(PETSC_USE_COMPLEX)
      PetscCall(PetscPrintf(PetscObjectComm((PetscObject)guess),"%g + %g i",(double)PetscRealPart(pod->swork[i]),(double)PetscImaginaryPart(pod->swork[i])));
#else
      PetscCall(PetscPrintf(PetscObjectComm((PetscObject)guess),"%g ",(double)pod->swork[i]));
#endif
    }
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)guess),"\n"));
  }
  /* A_low x_low = b_low */
  if (!pod->Aspd) { /* A is spd -> LOW = Identity */
    KSP       pksp = guess->ksp;
    PetscBool tsolve,symm;

    if (pod->monitor) {
      PetscMPIInt rank;
      Mat         L;

      PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)guess),&rank));
      PetscCall(MatCreateSeqDense(PETSC_COMM_SELF,pod->nen,pod->nen,pod->low,&L));
      if (rank == 0) {
        PetscCall(MatView(L,NULL));
      }
      PetscCall(MatDestroy(&L));
    }
    PetscCall(MatGetOption(guess->A,MAT_SYMMETRIC,&symm));
    tsolve = symm ? PETSC_FALSE : pksp->transpose_solve;
    PetscStackCallBLAS("LAPACKgetrf",LAPACKgetrf_(&bNen,&bNen,pod->low,&bNen,pod->iwork,&lierr));
    PetscCheck(!lierr,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in GETRF Lapack routine %d",(int)lierr);
    PetscStackCallBLAS("LAPACKgetrs",LAPACKgetrs_(tsolve ? "T" : "N",&bNen,&ione,pod->low,&bNen,pod->iwork,pod->swork,&bNen,&lierr));
    PetscCheck(!lierr,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in GETRS Lapack routine %d",(int)lierr);
  }
  /* x = X * V * S * x_low */
  PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&bN,&bNen,&one,pod->eigv+pod->st*pod->n,&bN,pod->swork,&ione,&zero,pod->swork+pod->n,&ione));
  if (pod->monitor) {
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)guess),"  KSPGuessPOD sol = "));
    for (i=0; i<pod->nen; i++) {
#if defined(PETSC_USE_COMPLEX)
      PetscCall(PetscPrintf(PetscObjectComm((PetscObject)guess),"%g + %g i",(double)PetscRealPart(pod->swork[i+pod->n]),(double)PetscImaginaryPart(pod->swork[i+pod->n])));
#else
      PetscCall(PetscPrintf(PetscObjectComm((PetscObject)guess),"%g ",(double)pod->swork[i+pod->n]));
#endif
    }
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)guess),"\n"));
  }
  PetscCall(VecGetArray(x,&array));
  PetscCall(VecPlaceArray(pod->bsnap[pod->curr],array));
  PetscCall(VecSet(pod->bsnap[pod->curr],0));
  PetscCall(VecMAXPY(pod->bsnap[pod->curr],pod->n,pod->swork+pod->n,pod->xsnap));
  PetscCall(VecResetArray(pod->bsnap[pod->curr]));
  PetscCall(VecRestoreArray(x,&array));
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPGuessUpdate_POD(KSPGuess guess, Vec b, Vec x)
{
  KSPGuessPOD    *pod = (KSPGuessPOD*)guess->data;
  PetscScalar    one = 1, zero = 0,*array;
  PetscReal      toten, parten, reps = 0; /* dlamch? */
  PetscBLASInt   bN,lierr,idummy;
  PetscInt       i;

  PetscFunctionBegin;
  if (pod->ndots_iallreduce) goto complete_request;
  pod->n = pod->n < pod->maxn ? pod->n+1 : pod->maxn;
  PetscCall(VecCopy(x,pod->xsnap[pod->curr]));
  PetscCall(VecGetArray(pod->bsnap[pod->curr],&array));
  PetscCall(VecPlaceArray(pod->work[0],array));
  PetscCall(KSP_MatMult(guess->ksp,guess->A,x,pod->work[0]));
  PetscCall(VecResetArray(pod->work[0]));
  PetscCall(VecRestoreArray(pod->bsnap[pod->curr],&array));
  if (pod->Aspd) {
    PetscCall(VecMDot(pod->xsnap[pod->curr],pod->n,pod->bsnap,pod->swork));
#if !defined(PETSC_HAVE_MPI_NONBLOCKING_COLLECTIVES)
    PetscCall(MPIU_Allreduce(pod->swork,pod->swork + 3*pod->n,pod->n,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)guess)));
#else
    PetscCallMPI(MPI_Iallreduce(pod->swork,pod->dots_iallreduce,pod->n,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)guess),&pod->req_iallreduce));
    pod->ndots_iallreduce = 1;
#endif
  } else {
    PetscInt  off;
    PetscBool herm;

#if defined(PETSC_USE_COMPLEX)
    PetscCall(MatGetOption(guess->A,MAT_HERMITIAN,&herm));
#else
    PetscCall(MatGetOption(guess->A,MAT_SYMMETRIC,&herm));
#endif
    off = (guess->ksp->transpose_solve && !herm) ? 2*pod->n : pod->n;

    /* TODO: we may want to use a user-defined dot for the correlation matrix */
    PetscCall(VecMDot(pod->xsnap[pod->curr],pod->n,pod->xsnap,pod->swork));
    PetscCall(VecMDot(pod->bsnap[pod->curr],pod->n,pod->xsnap,pod->swork + off));
    if (!herm) {
      off  = (off == pod->n) ? 2*pod->n : pod->n;
      PetscCall(VecMDot(pod->xsnap[pod->curr],pod->n,pod->bsnap,pod->swork + off));
#if !defined(PETSC_HAVE_MPI_NONBLOCKING_COLLECTIVES)
      PetscCall(MPIU_Allreduce(pod->swork,pod->swork + 3*pod->n,3*pod->n,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)guess)));
#else
      PetscCallMPI(MPI_Iallreduce(pod->swork,pod->dots_iallreduce,3*pod->n,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)guess),&pod->req_iallreduce));
      pod->ndots_iallreduce = 3;
#endif
    } else {
#if !defined(PETSC_HAVE_MPI_NONBLOCKING_COLLECTIVES)
      PetscCall(MPIU_Allreduce(pod->swork,pod->swork + 3*pod->n,2*pod->n,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)guess)));
      for (i=0;i<pod->n;i++) pod->swork[5*pod->n + i] = pod->swork[4*pod->n + i];
#else
      PetscCallMPI(MPI_Iallreduce(pod->swork,pod->dots_iallreduce,2*pod->n,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)guess),&pod->req_iallreduce));
      pod->ndots_iallreduce = 2;
#endif
    }
  }
  if (pod->ndots_iallreduce) PetscFunctionReturn(0);

complete_request:
  if (pod->ndots_iallreduce) {
    PetscCallMPI(MPI_Wait(&pod->req_iallreduce,MPI_STATUS_IGNORE));
    switch (pod->ndots_iallreduce) {
    case 3:
      for (i=0;i<pod->n;i++) pod->swork[3*pod->n + i] = pod->dots_iallreduce[         i];
      for (i=0;i<pod->n;i++) pod->swork[4*pod->n + i] = pod->dots_iallreduce[  pod->n+i];
      for (i=0;i<pod->n;i++) pod->swork[5*pod->n + i] = pod->dots_iallreduce[2*pod->n+i];
      break;
    case 2:
      for (i=0;i<pod->n;i++) pod->swork[3*pod->n + i] = pod->dots_iallreduce[       i];
      for (i=0;i<pod->n;i++) pod->swork[4*pod->n + i] = pod->dots_iallreduce[pod->n+i];
      for (i=0;i<pod->n;i++) pod->swork[5*pod->n + i] = pod->dots_iallreduce[pod->n+i];
      break;
    case 1:
      for (i=0;i<pod->n;i++) pod->swork[3*pod->n + i] = pod->dots_iallreduce[i];
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)guess),PETSC_ERR_PLIB,"Invalid number of outstanding dots operations: %" PetscInt_FMT,pod->ndots_iallreduce);
    }
  }
  pod->ndots_iallreduce = 0;

  /* correlation matrix and Y^H A Y (Galerkin) */
  for (i=0;i<pod->n;i++) {
    pod->corr[pod->curr*pod->maxn+i] = pod->swork[3*pod->n + i];
    pod->corr[i*pod->maxn+pod->curr] = PetscConj(pod->swork[3*pod->n + i]);
    if (!pod->Aspd) {
      pod->yhay[pod->curr*pod->maxn+i] = pod->swork[4*pod->n + i];
      pod->yhay[i*pod->maxn+pod->curr] = PetscConj(pod->swork[5*pod->n + i]);
    }
  }
  /* syevx change the input matrix */
  for (i=0;i<pod->n;i++) {
    PetscInt j;
    for (j=i;j<pod->n;j++) pod->swork[i*pod->n+j] = pod->corr[i*pod->maxn+j];
  }
  PetscCall(PetscBLASIntCast(pod->n,&bN));
#if !defined(PETSC_USE_COMPLEX)
  PetscStackCallBLAS("LAPACKsyevx",LAPACKsyevx_("V","A","L",&bN,pod->swork,&bN,
                                                &reps,&reps,&idummy,&idummy,
                                                &reps,&idummy,pod->eigs,pod->eigv,&bN,
                                                pod->swork+bN*bN,&pod->lwork,pod->iwork,pod->iwork+5*bN,&lierr));
#else
  PetscStackCallBLAS("LAPACKsyevx",LAPACKsyevx_("V","A","L",&bN,pod->swork,&bN,
                                                &reps,&reps,&idummy,&idummy,
                                                &reps,&idummy,pod->eigs,pod->eigv,&bN,
                                                pod->swork+bN*bN,&pod->lwork,pod->rwork,pod->iwork,pod->iwork+5*bN,&lierr));
#endif
  PetscCheck(lierr>=0,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in SYEV Lapack routine: illegal argument %d",-(int)lierr);
  else PetscCheck(!lierr,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in SYEV Lapack routine: %d eigenvectors failed to converge",(int)lierr);

  /* dimension of lower dimensional system */
  pod->st = -1;
  for (i=0,toten=0;i<pod->n;i++) {
    pod->eigs[i] = PetscMax(pod->eigs[i],0.0);
    toten += pod->eigs[i];
    if (!pod->eigs[i]) pod->st = i;
  }
  pod->nen = 0;
  for (i=pod->n-1,parten=0;i>pod->st && toten > 0;i--) {
    pod->nen++;
    parten += pod->eigs[i];
    if (parten + toten*pod->tol >= toten) break;
  }
  pod->st = pod->n - pod->nen;

  /* Compute eigv = V * S */
  for (i=pod->st;i<pod->n;i++) {
    const PetscReal v = 1.0/PetscSqrtReal(pod->eigs[i]);
    const PetscInt  st = pod->n*i;
    PetscInt        j;

    for (j=0;j<pod->n;j++) pod->eigv[st+j] *= v;
  }

  /* compute S * V^T * X^T * A * X * V * S if needed */
  if (pod->nen && !pod->Aspd) {
    PetscBLASInt bNen,bMaxN;
    PetscInt     st = pod->st*pod->n;
    PetscCall(PetscBLASIntCast(pod->nen,&bNen));
    PetscCall(PetscBLASIntCast(pod->maxn,&bMaxN));
    PetscStackCallBLAS("BLASgemm",BLASgemm_("T","N",&bNen,&bN,&bN,&one,pod->eigv+st,&bN,pod->yhay,&bMaxN,&zero,pod->swork,&bNen));
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&bNen,&bNen,&bN,&one,pod->swork,&bNen,pod->eigv+st,&bN,&zero,pod->low,&bNen));
  }

  if (pod->monitor) {
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)guess),"  KSPGuessPOD: basis %" PetscBLASInt_FMT ", energy fractions = ",pod->nen));
    for (i=pod->n-1;i>=0;i--) {
      PetscCall(PetscPrintf(PetscObjectComm((PetscObject)guess),"%1.6e (%d) ",(double)(pod->eigs[i]/toten),i >= pod->st ? 1 : 0));
    }
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)guess),"\n"));
    if (PetscDefined(USE_DEBUG)) {
      for (i=0;i<pod->n;i++) {
        Vec v;
        PetscInt j;
        PetscBLASInt bNen,ione = 1;

        PetscCall(VecDuplicate(pod->xsnap[i],&v));
        PetscCall(VecCopy(pod->xsnap[i],v));
        PetscCall(PetscBLASIntCast(pod->nen,&bNen));
        PetscStackCallBLAS("BLASgemv",BLASgemv_("T",&bN,&bNen,&one,pod->eigv+pod->st*pod->n,&bN,pod->corr+pod->maxn*i,&ione,&zero,pod->swork,&ione));
        PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&bN,&bNen,&one,pod->eigv+pod->st*pod->n,&bN,pod->swork,&ione,&zero,pod->swork+pod->n,&ione));
        for (j=0;j<pod->n;j++) pod->swork[j] = -pod->swork[pod->n+j];
        PetscCall(VecMAXPY(v,pod->n,pod->swork,pod->xsnap));
        PetscCall(VecDot(v,v,pod->swork));
        PetscCall(MPIU_Allreduce(pod->swork,pod->swork + 1,1,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)guess)));
        PetscCall(PetscPrintf(PetscObjectComm((PetscObject)guess),"  Error projection %" PetscInt_FMT ": %g (expected lower than %g)\n",i,(double)PetscRealPart(pod->swork[1]),(double)(toten-parten)));
        PetscCall(VecDestroy(&v));
      }
    }
  }
  /* new tip */
  pod->curr = (pod->curr+1)%pod->maxn;
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPGuessSetFromOptions_POD(KSPGuess guess)
{
  KSPGuessPOD    *pod = (KSPGuessPOD *)guess->data;

  PetscFunctionBegin;
  PetscOptionsBegin(PetscObjectComm((PetscObject)guess),((PetscObject)guess)->prefix,"POD initial guess options","KSPGuess");
  PetscCall(PetscOptionsInt("-ksp_guess_pod_size","Number of snapshots",NULL,pod->maxn,&pod->maxn,NULL));
  PetscCall(PetscOptionsBool("-ksp_guess_pod_monitor","Monitor initial guess generator",NULL,pod->monitor,&pod->monitor,NULL));
  PetscCall(PetscOptionsReal("-ksp_guess_pod_tol","Tolerance to retain eigenvectors","KSPGuessSetTolerance",pod->tol,&pod->tol,NULL));
  PetscCall(PetscOptionsBool("-ksp_guess_pod_Ainner","Use the operator as inner product (must be SPD)",NULL,pod->Aspd,&pod->Aspd,NULL));
  PetscOptionsEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPGuessSetTolerance_POD(KSPGuess guess,PetscReal tol)
{
  KSPGuessPOD *pod = (KSPGuessPOD *)guess->data;

  PetscFunctionBegin;
  pod->tol = tol;
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPGuessView_POD(KSPGuess guess,PetscViewer viewer)
{
  KSPGuessPOD    *pod = (KSPGuessPOD*)guess->data;
  PetscBool      isascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii));
  if (isascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer,"Max size %" PetscInt_FMT ", tolerance %g, Ainner %d\n",pod->maxn,(double)pod->tol,pod->Aspd));
  }
  PetscFunctionReturn(0);
}

/*
    KSPGUESSPOD - Implements a proper orthogonal decomposition based Galerkin scheme for repeated linear system solves.

  The initial guess is obtained by solving a small and dense linear system, obtained by Galerkin projection on a lower dimensional space generated by the previous solutions.
  The number of solutions to be retained and the energy tolerance to construct the lower dimensional basis can be specified at command line by -ksp_guess_pod_tol <real> and -ksp_guess_pod_size <int>.

  References:
. * - http://www.math.uni-konstanz.de/numerik/personen/volkwein/teaching/POD-Book.pdf

    Level: intermediate

.seealso: `KSPGuess`, `KSPGuessType`, `KSPGuessCreate()`, `KSPSetGuess()`, `KSPGetGuess()`
@*/
PetscErrorCode KSPGuessCreate_POD(KSPGuess guess)
{
  KSPGuessPOD    *pod;

  PetscFunctionBegin;
  PetscCall(PetscNewLog(guess,&pod));
  pod->maxn   = 10;
  pod->tol    = PETSC_MACHINE_EPSILON;
  guess->data = pod;

  guess->ops->setfromoptions = KSPGuessSetFromOptions_POD;
  guess->ops->destroy        = KSPGuessDestroy_POD;
  guess->ops->settolerance   = KSPGuessSetTolerance_POD;
  guess->ops->setup          = KSPGuessSetUp_POD;
  guess->ops->view           = KSPGuessView_POD;
  guess->ops->reset          = KSPGuessReset_POD;
  guess->ops->update         = KSPGuessUpdate_POD;
  guess->ops->formguess      = KSPGuessFormGuess_POD;
  PetscFunctionReturn(0);
}
