#include <../src/ksp/ksp/utils/lmvm/lmvm.h> /*I "petscksp.h" I*/

/*------------------------------------------------------------*/

PetscErrorCode MatReset_LMVM(Mat B, PetscBool destructive)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;

  PetscFunctionBegin;
  lmvm->k = -1;
  lmvm->prev_set = PETSC_FALSE;
  lmvm->shift = 0.0;
  if (destructive && lmvm->allocated) {
    PetscCall(MatLMVMClearJ0(B));
    B->rmap->n = B->rmap->N = B->cmap->n = B->cmap->N = 0;
    PetscCall(VecDestroyVecs(lmvm->m, &lmvm->S));
    PetscCall(VecDestroyVecs(lmvm->m, &lmvm->Y));
    PetscCall(VecDestroy(&lmvm->Xprev));
    PetscCall(VecDestroy(&lmvm->Fprev));
    lmvm->nupdates = 0;
    lmvm->nrejects = 0;
    lmvm->m_old = 0;
    lmvm->allocated = PETSC_FALSE;
    B->preallocated = PETSC_FALSE;
    B->assembled = PETSC_FALSE;
  }
  ++lmvm->nresets;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PetscErrorCode MatAllocate_LMVM(Mat B, Vec X, Vec F)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  PetscBool         same, allocate = PETSC_FALSE;
  PetscInt          m, n, M, N;
  VecType           type;

  PetscFunctionBegin;
  if (lmvm->allocated) {
    VecCheckMatCompatible(B, X, 2, F, 3);
    PetscCall(VecGetType(X, &type));
    PetscCall(PetscObjectTypeCompare((PetscObject)lmvm->Xprev, type, &same));
    if (!same) {
      /* Given X vector has a different type than allocated X-type data structures.
         We need to destroy all of this and duplicate again out of the given vector. */
      allocate = PETSC_TRUE;
      PetscCall(MatLMVMReset(B, PETSC_TRUE));
    }
  } else {
    allocate = PETSC_TRUE;
  }
  if (allocate) {
    PetscCall(VecGetLocalSize(X, &n));
    PetscCall(VecGetSize(X, &N));
    PetscCall(VecGetLocalSize(F, &m));
    PetscCall(VecGetSize(F, &M));
    B->rmap->n = m;
    B->cmap->n = n;
    B->rmap->N = M > -1 ? M : B->rmap->N;
    B->cmap->N = N > -1 ? N : B->cmap->N;
    PetscCall(VecDuplicate(X, &lmvm->Xprev));
    PetscCall(VecDuplicate(F, &lmvm->Fprev));
    if (lmvm->m > 0) {
      PetscCall(VecDuplicateVecs(lmvm->Xprev, lmvm->m, &lmvm->S));
      PetscCall(VecDuplicateVecs(lmvm->Fprev, lmvm->m, &lmvm->Y));
    }
    lmvm->m_old = lmvm->m;
    lmvm->allocated = PETSC_TRUE;
    B->preallocated = PETSC_TRUE;
    B->assembled = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PetscErrorCode MatUpdateKernel_LMVM(Mat B, Vec S, Vec Y)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  PetscInt          i;
  Vec               Stmp, Ytmp;

  PetscFunctionBegin;
  if (lmvm->k == lmvm->m-1) {
    /* We hit the memory limit, so shift all the vectors back one spot
       and shift the oldest to the front to receive the latest update. */
    Stmp = lmvm->S[0];
    Ytmp = lmvm->Y[0];
    for (i = 0; i < lmvm->k; ++i) {
      lmvm->S[i] = lmvm->S[i+1];
      lmvm->Y[i] = lmvm->Y[i+1];
    }
    lmvm->S[lmvm->k] = Stmp;
    lmvm->Y[lmvm->k] = Ytmp;
  } else {
    ++lmvm->k;
  }
  /* Put the precomputed update into the last vector */
  PetscCall(VecCopy(S, lmvm->S[lmvm->k]));
  PetscCall(VecCopy(Y, lmvm->Y[lmvm->k]));
  ++lmvm->nupdates;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PetscErrorCode MatUpdate_LMVM(Mat B, Vec X, Vec F)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;

  PetscFunctionBegin;
  if (!lmvm->m) PetscFunctionReturn(0);
  if (lmvm->prev_set) {
    /* Compute the new (S = X - Xprev) and (Y = F - Fprev) vectors */
    PetscCall(VecAXPBY(lmvm->Xprev, 1.0, -1.0, X));
    PetscCall(VecAXPBY(lmvm->Fprev, 1.0, -1.0, F));
    /* Update S and Y */
    PetscCall(MatUpdateKernel_LMVM(B, lmvm->Xprev, lmvm->Fprev));
  }

  /* Save the solution and function to be used in the next update */
  PetscCall(VecCopy(X, lmvm->Xprev));
  PetscCall(VecCopy(F, lmvm->Fprev));
  lmvm->prev_set = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatMultAdd_LMVM(Mat B, Vec X, Vec Y, Vec Z)
{
  PetscFunctionBegin;
  PetscCall(MatMult(B, X, Z));
  PetscCall(VecAXPY(Z, 1.0, Y));
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatMult_LMVM(Mat B, Vec X, Vec Y)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;

  PetscFunctionBegin;
  VecCheckSameSize(X, 2, Y, 3);
  VecCheckMatCompatible(B, X, 2, Y, 3);
  PetscCheck(lmvm->allocated,PetscObjectComm((PetscObject)B), PETSC_ERR_ORDER, "LMVM matrix must be allocated first");
  PetscCall((*lmvm->ops->mult)(B, X, Y));
  if (lmvm->shift != 0.0) {
    PetscCall(VecAXPY(Y, lmvm->shift, X));
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatCopy_LMVM(Mat B, Mat M, MatStructure str)
{
  Mat_LMVM          *bctx = (Mat_LMVM*)B->data;
  Mat_LMVM          *mctx;
  PetscInt          i;
  PetscBool         allocatedM;

  PetscFunctionBegin;
  if (str == DIFFERENT_NONZERO_PATTERN) {
    PetscCall(MatLMVMReset(M, PETSC_TRUE));
    PetscCall(MatLMVMAllocate(M, bctx->Xprev, bctx->Fprev));
  } else {
    PetscCall(MatLMVMIsAllocated(M, &allocatedM));
    PetscCheck(allocatedM,PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_WRONGSTATE, "Target matrix must be allocated first");
    MatCheckSameSize(B, 1, M, 2);
  }

  mctx = (Mat_LMVM*)M->data;
  if (bctx->user_pc) {
    PetscCall(MatLMVMSetJ0PC(M, bctx->J0pc));
  } else if (bctx->user_ksp) {
    PetscCall(MatLMVMSetJ0KSP(M, bctx->J0ksp));
  } else if (bctx->J0) {
    PetscCall(MatLMVMSetJ0(M, bctx->J0));
  } else if (bctx->user_scale) {
    if (bctx->J0diag) {
      PetscCall(MatLMVMSetJ0Diag(M, bctx->J0diag));
    } else {
      PetscCall(MatLMVMSetJ0Scale(M, bctx->J0scalar));
    }
  }
  mctx->nupdates = bctx->nupdates;
  mctx->nrejects = bctx->nrejects;
  mctx->k = bctx->k;
  for (i=0; i<=bctx->k; ++i) {
    PetscCall(VecCopy(bctx->S[i], mctx->S[i]));
    PetscCall(VecCopy(bctx->Y[i], mctx->Y[i]));
    PetscCall(VecCopy(bctx->Xprev, mctx->Xprev));
    PetscCall(VecCopy(bctx->Fprev, mctx->Fprev));
  }
  if (bctx->ops->copy) PetscCall((*bctx->ops->copy)(B, M, str));
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatDuplicate_LMVM(Mat B, MatDuplicateOption op, Mat *mat)
{
  Mat_LMVM          *bctx = (Mat_LMVM*)B->data;
  Mat_LMVM          *mctx;
  MatType           lmvmType;
  Mat               A;

  PetscFunctionBegin;
  PetscCall(MatGetType(B, &lmvmType));
  PetscCall(MatCreate(PetscObjectComm((PetscObject)B), mat));
  PetscCall(MatSetType(*mat, lmvmType));

  A = *mat;
  mctx = (Mat_LMVM*)A->data;
  mctx->m = bctx->m;
  mctx->ksp_max_it = bctx->ksp_max_it;
  mctx->ksp_rtol = bctx->ksp_rtol;
  mctx->ksp_atol = bctx->ksp_atol;
  mctx->shift = bctx->shift;
  PetscCall(KSPSetTolerances(mctx->J0ksp, mctx->ksp_rtol, mctx->ksp_atol, PETSC_DEFAULT, mctx->ksp_max_it));

  PetscCall(MatLMVMAllocate(*mat, bctx->Xprev, bctx->Fprev));
  if (op == MAT_COPY_VALUES) {
    PetscCall(MatCopy(B, *mat, SAME_NONZERO_PATTERN));
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatShift_LMVM(Mat B, PetscScalar a)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;

  PetscFunctionBegin;
  PetscCheck(lmvm->allocated,PetscObjectComm((PetscObject)B), PETSC_ERR_ORDER, "LMVM matrix must be allocated first");
  lmvm->shift += PetscRealPart(a);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatGetVecs_LMVM(Mat B, Vec *L, Vec *R)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;

  PetscFunctionBegin;
  PetscCheck(lmvm->allocated,PetscObjectComm((PetscObject)B), PETSC_ERR_ORDER, "LMVM matrix must be allocated first");
  PetscCall(VecDuplicate(lmvm->Xprev, L));
  PetscCall(VecDuplicate(lmvm->Fprev, R));
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PetscErrorCode MatView_LMVM(Mat B, PetscViewer pv)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  PetscBool         isascii;
  MatType           type;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)pv,PETSCVIEWERASCII,&isascii));
  if (isascii) {
    PetscCall(MatGetType(B, &type));
    PetscCall(PetscViewerASCIIPrintf(pv,"Max. storage: %" PetscInt_FMT "\n",lmvm->m));
    PetscCall(PetscViewerASCIIPrintf(pv,"Used storage: %" PetscInt_FMT "\n",lmvm->k+1));
    PetscCall(PetscViewerASCIIPrintf(pv,"Number of updates: %" PetscInt_FMT "\n",lmvm->nupdates));
    PetscCall(PetscViewerASCIIPrintf(pv,"Number of rejects: %" PetscInt_FMT "\n",lmvm->nrejects));
    PetscCall(PetscViewerASCIIPrintf(pv,"Number of resets: %" PetscInt_FMT "\n",lmvm->nresets));
    if (lmvm->J0) {
      PetscCall(PetscViewerASCIIPrintf(pv,"J0 Matrix:\n"));
      PetscCall(PetscViewerPushFormat(pv, PETSC_VIEWER_ASCII_INFO));
      PetscCall(MatView(lmvm->J0, pv));
      PetscCall(PetscViewerPopFormat(pv));
    }
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PetscErrorCode MatSetFromOptions_LMVM(PetscOptionItems *PetscOptionsObject, Mat B)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject,"Limited-memory Variable Metric matrix for approximating Jacobians");
  PetscCall(PetscOptionsInt("-mat_lmvm_hist_size","number of past updates kept in memory for the approximation","",lmvm->m,&lmvm->m,NULL));
  PetscCall(PetscOptionsInt("-mat_lmvm_ksp_its","(developer) fixed number of KSP iterations to take when inverting J0","",lmvm->ksp_max_it,&lmvm->ksp_max_it,NULL));
  PetscCall(PetscOptionsReal("-mat_lmvm_eps","(developer) machine zero definition","",lmvm->eps,&lmvm->eps,NULL));
  PetscOptionsHeadEnd();
  PetscCall(KSPSetFromOptions(lmvm->J0ksp));
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PetscErrorCode MatSetUp_LMVM(Mat B)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  PetscInt          m, n, M, N;
  PetscMPIInt       size;
  MPI_Comm          comm = PetscObjectComm((PetscObject)B);

  PetscFunctionBegin;
  PetscCall(MatGetSize(B, &M, &N));
  PetscCheck(M != 0 || N != 0,comm, PETSC_ERR_ORDER, "MatSetSizes() must be called before MatSetUp()");
  if (!lmvm->allocated) {
    PetscCallMPI(MPI_Comm_size(comm, &size));
    if (size == 1) {
      PetscCall(VecCreateSeq(comm, N, &lmvm->Xprev));
      PetscCall(VecCreateSeq(comm, M, &lmvm->Fprev));
    } else {
      PetscCall(MatGetLocalSize(B, &m, &n));
      PetscCall(VecCreateMPI(comm, n, N, &lmvm->Xprev));
      PetscCall(VecCreateMPI(comm, m, M, &lmvm->Fprev));
    }
    if (lmvm->m > 0) {
      PetscCall(VecDuplicateVecs(lmvm->Xprev, lmvm->m, &lmvm->S));
      PetscCall(VecDuplicateVecs(lmvm->Fprev, lmvm->m, &lmvm->Y));
    }
    lmvm->m_old = lmvm->m;
    lmvm->allocated = PETSC_TRUE;
    B->preallocated = PETSC_TRUE;
    B->assembled = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PetscErrorCode MatDestroy_LMVM(Mat B)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;

  PetscFunctionBegin;
  if (lmvm->allocated) {
    PetscCall(VecDestroyVecs(lmvm->m, &lmvm->S));
    PetscCall(VecDestroyVecs(lmvm->m, &lmvm->Y));
    PetscCall(VecDestroy(&lmvm->Xprev));
    PetscCall(VecDestroy(&lmvm->Fprev));
  }
  PetscCall(KSPDestroy(&lmvm->J0ksp));
  PetscCall(MatLMVMClearJ0(B));
  PetscCall(PetscFree(B->data));
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PetscErrorCode MatCreate_LMVM(Mat B)
{
  Mat_LMVM          *lmvm;

  PetscFunctionBegin;
  PetscCall(PetscNewLog(B, &lmvm));
  B->data = (void*)lmvm;

  lmvm->m_old = 0;
  lmvm->m = 5;
  lmvm->k = -1;
  lmvm->nupdates = 0;
  lmvm->nrejects = 0;
  lmvm->nresets = 0;

  lmvm->ksp_max_it = 20;
  lmvm->ksp_rtol = 0.0;
  lmvm->ksp_atol = 0.0;

  lmvm->shift = 0.0;

  lmvm->eps = PetscPowReal(PETSC_MACHINE_EPSILON, 2.0/3.0);
  lmvm->allocated = PETSC_FALSE;
  lmvm->prev_set = PETSC_FALSE;
  lmvm->user_scale = PETSC_FALSE;
  lmvm->user_pc = PETSC_FALSE;
  lmvm->user_ksp = PETSC_FALSE;
  lmvm->square = PETSC_FALSE;

  B->ops->destroy = MatDestroy_LMVM;
  B->ops->setfromoptions = MatSetFromOptions_LMVM;
  B->ops->view = MatView_LMVM;
  B->ops->setup = MatSetUp_LMVM;
  B->ops->getvecs = MatGetVecs_LMVM;
  B->ops->shift = MatShift_LMVM;
  B->ops->duplicate = MatDuplicate_LMVM;
  B->ops->mult = MatMult_LMVM;
  B->ops->multadd = MatMultAdd_LMVM;
  B->ops->copy = MatCopy_LMVM;

  lmvm->ops->update = MatUpdate_LMVM;
  lmvm->ops->allocate = MatAllocate_LMVM;
  lmvm->ops->reset = MatReset_LMVM;

  PetscCall(KSPCreate(PetscObjectComm((PetscObject)B), &lmvm->J0ksp));
  PetscCall(PetscObjectIncrementTabLevel((PetscObject)lmvm->J0ksp, (PetscObject)B, 1));
  PetscCall(KSPSetOptionsPrefix(lmvm->J0ksp, "mat_lmvm_"));
  PetscCall(KSPSetType(lmvm->J0ksp, KSPGMRES));
  PetscCall(KSPSetTolerances(lmvm->J0ksp, lmvm->ksp_rtol, lmvm->ksp_atol, PETSC_DEFAULT, lmvm->ksp_max_it));
  PetscFunctionReturn(0);
}
