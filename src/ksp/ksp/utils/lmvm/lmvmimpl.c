#include <../src/ksp/ksp/utils/lmvm/lmvm.h> /*I "petscksp.h" I*/

/*------------------------------------------------------------*/

PetscErrorCode MatReset_LMVM(Mat B, PetscBool destructive)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  lmvm->k = -1;
  lmvm->prev_set = PETSC_FALSE;
  lmvm->shift = 0.0;
  if (destructive && lmvm->allocated) {
    ierr = MatLMVMClearJ0(B);CHKERRQ(ierr);
    B->rmap->n = B->rmap->N = B->cmap->n = B->cmap->N = 0;
    ierr = VecDestroyVecs(lmvm->m, &lmvm->S);CHKERRQ(ierr);
    ierr = VecDestroyVecs(lmvm->m, &lmvm->Y);CHKERRQ(ierr);
    ierr = VecDestroy(&lmvm->Xprev);CHKERRQ(ierr);
    ierr = VecDestroy(&lmvm->Fprev);CHKERRQ(ierr);
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
  PetscErrorCode    ierr;
  PetscBool         same, allocate = PETSC_FALSE;
  PetscInt          m, n, M, N;
  VecType           type;

  PetscFunctionBegin;
  if (lmvm->allocated) {
    VecCheckMatCompatible(B, X, 2, F, 3);
    ierr = VecGetType(X, &type);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)lmvm->Xprev, type, &same);CHKERRQ(ierr);
    if (!same) {
      /* Given X vector has a different type than allocated X-type data structures.
         We need to destroy all of this and duplicate again out of the given vector. */
      allocate = PETSC_TRUE;
      ierr = MatLMVMReset(B, PETSC_TRUE);CHKERRQ(ierr);
    }
  } else {
    allocate = PETSC_TRUE;
  }
  if (allocate) {
    ierr = VecGetLocalSize(X, &n);CHKERRQ(ierr);
    ierr = VecGetSize(X, &N);CHKERRQ(ierr);
    ierr = VecGetLocalSize(F, &m);CHKERRQ(ierr);
    ierr = VecGetSize(F, &M);CHKERRQ(ierr);
    B->rmap->n = m;
    B->cmap->n = n;
    B->rmap->N = M > -1 ? M : B->rmap->N;
    B->cmap->N = N > -1 ? N : B->cmap->N;
    ierr = VecDuplicate(X, &lmvm->Xprev);CHKERRQ(ierr);
    ierr = VecDuplicate(F, &lmvm->Fprev);CHKERRQ(ierr);
    if (lmvm->m > 0) {
      ierr = VecDuplicateVecs(lmvm->Xprev, lmvm->m, &lmvm->S);CHKERRQ(ierr);
      ierr = VecDuplicateVecs(lmvm->Fprev, lmvm->m, &lmvm->Y);CHKERRQ(ierr);
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
  PetscErrorCode    ierr;
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
  ierr = VecCopy(S, lmvm->S[lmvm->k]);CHKERRQ(ierr);
  ierr = VecCopy(Y, lmvm->Y[lmvm->k]);CHKERRQ(ierr);
  ++lmvm->nupdates;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PetscErrorCode MatUpdate_LMVM(Mat B, Vec X, Vec F)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (!lmvm->m) PetscFunctionReturn(0);
  if (lmvm->prev_set) {
    /* Compute the new (S = X - Xprev) and (Y = F - Fprev) vectors */
    ierr = VecAXPBY(lmvm->Xprev, 1.0, -1.0, X);CHKERRQ(ierr);
    ierr = VecAXPBY(lmvm->Fprev, 1.0, -1.0, F);CHKERRQ(ierr);
    /* Update S and Y */
    ierr = MatUpdateKernel_LMVM(B, lmvm->Xprev, lmvm->Fprev);CHKERRQ(ierr);
  }

  /* Save the solution and function to be used in the next update */
  ierr = VecCopy(X, lmvm->Xprev);CHKERRQ(ierr);
  ierr = VecCopy(F, lmvm->Fprev);CHKERRQ(ierr);
  lmvm->prev_set = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatMultAdd_LMVM(Mat B, Vec X, Vec Y, Vec Z)
{
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MatMult(B, X, Z);CHKERRQ(ierr);
  ierr = VecAXPY(Z, 1.0, Y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatMult_LMVM(Mat B, Vec X, Vec Y)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  VecCheckSameSize(X, 2, Y, 3);
  VecCheckMatCompatible(B, X, 2, Y, 3);
  PetscCheckFalse(!lmvm->allocated,PetscObjectComm((PetscObject)B), PETSC_ERR_ORDER, "LMVM matrix must be allocated first");
  ierr = (*lmvm->ops->mult)(B, X, Y);CHKERRQ(ierr);
  if (lmvm->shift != 0.0) {
    ierr = VecAXPY(Y, lmvm->shift, X);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatCopy_LMVM(Mat B, Mat M, MatStructure str)
{
  Mat_LMVM          *bctx = (Mat_LMVM*)B->data;
  Mat_LMVM          *mctx;
  PetscErrorCode    ierr;
  PetscInt          i;
  PetscBool         allocatedM;

  PetscFunctionBegin;
  if (str == DIFFERENT_NONZERO_PATTERN) {
    ierr = MatLMVMReset(M, PETSC_TRUE);CHKERRQ(ierr);
    ierr = MatLMVMAllocate(M, bctx->Xprev, bctx->Fprev);CHKERRQ(ierr);
  } else {
    ierr = MatLMVMIsAllocated(M, &allocatedM);CHKERRQ(ierr);
    PetscCheckFalse(!allocatedM,PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_WRONGSTATE, "Target matrix must be allocated first");
    MatCheckSameSize(B, 1, M, 2);
  }

  mctx = (Mat_LMVM*)M->data;
  if (bctx->user_pc) {
    ierr = MatLMVMSetJ0PC(M, bctx->J0pc);CHKERRQ(ierr);
  } else if (bctx->user_ksp) {
    ierr = MatLMVMSetJ0KSP(M, bctx->J0ksp);CHKERRQ(ierr);
  } else if (bctx->J0) {
    ierr = MatLMVMSetJ0(M, bctx->J0);CHKERRQ(ierr);
  } else if (bctx->user_scale) {
    if (bctx->J0diag) {
      ierr = MatLMVMSetJ0Diag(M, bctx->J0diag);CHKERRQ(ierr);
    } else {
      ierr = MatLMVMSetJ0Scale(M, bctx->J0scalar);CHKERRQ(ierr);
    }
  }
  mctx->nupdates = bctx->nupdates;
  mctx->nrejects = bctx->nrejects;
  mctx->k = bctx->k;
  for (i=0; i<=bctx->k; ++i) {
    ierr = VecCopy(bctx->S[i], mctx->S[i]);CHKERRQ(ierr);
    ierr = VecCopy(bctx->Y[i], mctx->Y[i]);CHKERRQ(ierr);
    ierr = VecCopy(bctx->Xprev, mctx->Xprev);CHKERRQ(ierr);
    ierr = VecCopy(bctx->Fprev, mctx->Fprev);CHKERRQ(ierr);
  }
  if (bctx->ops->copy) {
    ierr = (*bctx->ops->copy)(B, M, str);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatDuplicate_LMVM(Mat B, MatDuplicateOption op, Mat *mat)
{
  Mat_LMVM          *bctx = (Mat_LMVM*)B->data;
  Mat_LMVM          *mctx;
  PetscErrorCode    ierr;
  MatType           lmvmType;
  Mat               A;

  PetscFunctionBegin;
  ierr = MatGetType(B, &lmvmType);CHKERRQ(ierr);
  ierr = MatCreate(PetscObjectComm((PetscObject)B), mat);CHKERRQ(ierr);
  ierr = MatSetType(*mat, lmvmType);CHKERRQ(ierr);

  A = *mat;
  mctx = (Mat_LMVM*)A->data;
  mctx->m = bctx->m;
  mctx->ksp_max_it = bctx->ksp_max_it;
  mctx->ksp_rtol = bctx->ksp_rtol;
  mctx->ksp_atol = bctx->ksp_atol;
  mctx->shift = bctx->shift;
  ierr = KSPSetTolerances(mctx->J0ksp, mctx->ksp_rtol, mctx->ksp_atol, PETSC_DEFAULT, mctx->ksp_max_it);CHKERRQ(ierr);

  ierr = MatLMVMAllocate(*mat, bctx->Xprev, bctx->Fprev);CHKERRQ(ierr);
  if (op == MAT_COPY_VALUES) {
    ierr = MatCopy(B, *mat, SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatShift_LMVM(Mat B, PetscScalar a)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;

  PetscFunctionBegin;
  PetscCheckFalse(!lmvm->allocated,PetscObjectComm((PetscObject)B), PETSC_ERR_ORDER, "LMVM matrix must be allocated first");
  lmvm->shift += PetscRealPart(a);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatGetVecs_LMVM(Mat B, Vec *L, Vec *R)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscCheckFalse(!lmvm->allocated,PetscObjectComm((PetscObject)B), PETSC_ERR_ORDER, "LMVM matrix must be allocated first");
  ierr = VecDuplicate(lmvm->Xprev, L);CHKERRQ(ierr);
  ierr = VecDuplicate(lmvm->Fprev, R);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PetscErrorCode MatView_LMVM(Mat B, PetscViewer pv)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  PetscErrorCode    ierr;
  PetscBool         isascii;
  MatType           type;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)pv,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = MatGetType(B, &type);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(pv,"Max. storage: %D\n",lmvm->m);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(pv,"Used storage: %D\n",lmvm->k+1);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(pv,"Number of updates: %D\n",lmvm->nupdates);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(pv,"Number of rejects: %D\n",lmvm->nrejects);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(pv,"Number of resets: %D\n",lmvm->nresets);CHKERRQ(ierr);
    if (lmvm->J0) {
      ierr = PetscViewerASCIIPrintf(pv,"J0 Matrix:\n");CHKERRQ(ierr);
      ierr = PetscViewerPushFormat(pv, PETSC_VIEWER_ASCII_INFO);CHKERRQ(ierr);
      ierr = MatView(lmvm->J0, pv);CHKERRQ(ierr);
      ierr = PetscViewerPopFormat(pv);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PetscErrorCode MatSetFromOptions_LMVM(PetscOptionItems *PetscOptionsObject, Mat B)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"Limited-memory Variable Metric matrix for approximating Jacobians");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-mat_lmvm_hist_size","number of past updates kept in memory for the approximation","",lmvm->m,&lmvm->m,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-mat_lmvm_ksp_its","(developer) fixed number of KSP iterations to take when inverting J0","",lmvm->ksp_max_it,&lmvm->ksp_max_it,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mat_lmvm_eps","(developer) machine zero definition","",lmvm->eps,&lmvm->eps,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  ierr = KSPSetFromOptions(lmvm->J0ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PetscErrorCode MatSetUp_LMVM(Mat B)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  PetscErrorCode    ierr;
  PetscInt          m, n, M, N;
  PetscMPIInt       size;
  MPI_Comm          comm = PetscObjectComm((PetscObject)B);

  PetscFunctionBegin;
  ierr = MatGetSize(B, &M, &N);CHKERRQ(ierr);
  PetscCheckFalse(M == 0 && N == 0,comm, PETSC_ERR_ORDER, "MatSetSizes() must be called before MatSetUp()");
  if (!lmvm->allocated) {
    ierr = MPI_Comm_size(comm, &size);CHKERRMPI(ierr);
    if (size == 1) {
      ierr = VecCreateSeq(comm, N, &lmvm->Xprev);CHKERRQ(ierr);
      ierr = VecCreateSeq(comm, M, &lmvm->Fprev);CHKERRQ(ierr);
    } else {
      ierr = MatGetLocalSize(B, &m, &n);CHKERRQ(ierr);
      ierr = VecCreateMPI(comm, n, N, &lmvm->Xprev);CHKERRQ(ierr);
      ierr = VecCreateMPI(comm, m, M, &lmvm->Fprev);CHKERRQ(ierr);
    }
    if (lmvm->m > 0) {
      ierr = VecDuplicateVecs(lmvm->Xprev, lmvm->m, &lmvm->S);CHKERRQ(ierr);
      ierr = VecDuplicateVecs(lmvm->Fprev, lmvm->m, &lmvm->Y);CHKERRQ(ierr);
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
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (lmvm->allocated) {
    ierr = VecDestroyVecs(lmvm->m, &lmvm->S);CHKERRQ(ierr);
    ierr = VecDestroyVecs(lmvm->m, &lmvm->Y);CHKERRQ(ierr);
    ierr = VecDestroy(&lmvm->Xprev);CHKERRQ(ierr);
    ierr = VecDestroy(&lmvm->Fprev);CHKERRQ(ierr);
  }
  ierr = KSPDestroy(&lmvm->J0ksp);CHKERRQ(ierr);
  ierr = MatLMVMClearJ0(B);CHKERRQ(ierr);
  ierr = PetscFree(B->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PetscErrorCode MatCreate_LMVM(Mat B)
{
  Mat_LMVM          *lmvm;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(B, &lmvm);CHKERRQ(ierr);
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

  ierr = KSPCreate(PetscObjectComm((PetscObject)B), &lmvm->J0ksp);CHKERRQ(ierr);
  ierr = PetscObjectIncrementTabLevel((PetscObject)lmvm->J0ksp, (PetscObject)B, 1);CHKERRQ(ierr);
  ierr = KSPSetOptionsPrefix(lmvm->J0ksp, "mat_lmvm_");CHKERRQ(ierr);
  ierr = KSPSetType(lmvm->J0ksp, KSPGMRES);CHKERRQ(ierr);
  ierr = KSPSetTolerances(lmvm->J0ksp, lmvm->ksp_rtol, lmvm->ksp_atol, PETSC_DEFAULT, lmvm->ksp_max_it);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
