/*
  Defines the matrix-matrix product C = A^T * B of an MPIAIJ matrix A and an MPIDENSE matrix B
*/
#include <../src/mat/impls/aij/seq/aij.h> /*I "petscmat.h" I*/
#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <../src/mat/impls/dense/mpi/mpidense.h>

// The work matrix for A_o^T B_loc and the strided PetscSFs, kept on C so that they are set up only once
typedef struct {
  Mat      workC; // A_o^T times one batch of columns of B_loc
  PetscSF  sf[2]; // reductions for batches of ncols[0] columns and for a smaller last batch of ncols[1] columns, NULL if unused
  PetscInt ncols[2];
  PetscInt clda;  // leading dimension of the local block of C that the PetscSFs are built for
  Mat      rootC; // has leading dimension clda, receives the reductions on a process where the local block of C no longer does
} MPIAIJ_MPIDense_AtB;

static PetscErrorCode MatProductCtxDestroy_MPIAIJ_MPIDense_AtB(PetscCtxRt ctx)
{
  MPIAIJ_MPIDense_AtB *data = *(MPIAIJ_MPIDense_AtB **)ctx;

  PetscFunctionBegin;
  PetscCall(MatDestroy(&data->workC));
  PetscCall(MatDestroy(&data->rootC));
  PetscCall(PetscSFDestroy(&data->sf[0]));
  PetscCall(PetscSFDestroy(&data->sf[1]));
  PetscCall(PetscFree(data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatTransposeMatMultNumeric_MPIAIJ_MPIDense(Mat, Mat, Mat);

PETSC_INTERN PetscErrorCode MatTransposeMatMultSymbolic_MPIAIJ_MPIDense(Mat A, Mat B, PetscReal fill, Mat C)
{
  Mat_MPIAIJ          *aij = (Mat_MPIAIJ *)A->data;
  MPIAIJ_MPIDense_AtB *data;
  VecScatter           ctx = aij->Mvctx;
  PetscInt             nz = aij->B->cmap->n, BN = B->cmap->N, Bbn, numBb, ncols;
  Mat                  bloc, cloc;
  PetscBool            cisdense;

  PetscFunctionBegin;
  MatCheckProduct(C, 4);
  PetscCheck(C->product->data == NULL, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Extra product struct not empty");

  /* create output dense matrix C = A^T*B */
  PetscCall(MatSetSizes(C, A->cmap->n, B->cmap->n, A->cmap->N, BN));
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)C, MATMPIDENSE, &cisdense));
  if (!cisdense) {
    PetscCall(MatSetType(C, ((PetscObject)B)->type_name));
    PetscCall(MatSetVecType(C, B->defaultvectype));
  }
  PetscCall(MatSetUp(C));
  bloc = ((Mat_MPIDense *)B->data)->A;
  cloc = ((Mat_MPIDense *)C->data)->A;

  PetscCall(PetscNew(&data));
  PetscCall(MatDenseGetLDA(C, &data->clda));
  PetscCall(MatMPIDenseGetBatchSize_Private(nz, A->rmap->n, B, C, &Bbn, &numBb));
  ncols = Bbn > 0 ? Bbn : BN;
  if (ncols > 0) {
    PetscInt wlda;

    PetscCall(MatCreate(PETSC_COMM_SELF, &data->workC));
    PetscCall(MatSetSizes(data->workC, nz, ncols, nz, ncols));
    PetscCall(MatSetType(data->workC, ((PetscObject)cloc)->type_name));
    PetscCall(MatSetUp(data->workC));
    PetscCall(MatDenseGetLDA(data->workC, &wlda));
    data->ncols[0] = ncols;
    PetscCall(PetscSFCreateStridedSF(ctx, ncols, data->clda, wlda, &data->sf[0]));
    if (numBb > 0 && BN % Bbn != 0) {
      data->ncols[1] = BN % Bbn;
      PetscCall(PetscSFCreateStridedSF(ctx, BN % Bbn, data->clda, wlda, &data->sf[1]));
    }
  }
  if (nz > 0 && ncols > 0) {
    Mat Bb = bloc, workC1;

    // set up the product with the matrix the numeric uses: the local block of B, or a sub-matrix of it if there are batches
    if (ncols != BN) PetscCall(MatDenseGetSubMatrix(bloc, PETSC_DECIDE, PETSC_DECIDE, 0, ncols, &Bb));
    PetscCall(MatProductCreateWithMat(aij->B, Bb, NULL, data->workC));
    PetscCall(MatProductSetType(data->workC, MATPRODUCT_AtB));
    PetscCall(MatProductSetFromOptions(data->workC));
    PetscCall(MatProductSymbolic(data->workC));
    if (ncols != BN) PetscCall(MatDenseRestoreSubMatrix(bloc, &Bb));
    if (numBb > 0 && BN % Bbn != 0) {
      // the last batch has fewer columns, so it needs its own product; the numeric gets the same sub-matrix of workC back, so the product stays on it
      PetscCall(MatDenseGetSubMatrix(bloc, PETSC_DECIDE, PETSC_DECIDE, 0, BN % Bbn, &Bb));
      PetscCall(MatDenseGetSubMatrix(data->workC, PETSC_DECIDE, PETSC_DECIDE, 0, BN % Bbn, &workC1));
      PetscCall(MatProductCreateWithMat(aij->B, Bb, NULL, workC1));
      PetscCall(MatProductSetType(workC1, MATPRODUCT_AtB));
      PetscCall(MatProductSetFromOptions(workC1));
      PetscCall(MatProductSymbolic(workC1));
      PetscCall(MatDenseRestoreSubMatrix(data->workC, &workC1));
      PetscCall(MatDenseRestoreSubMatrix(bloc, &Bb));
    }
  }
  if (ncols > 0) {
    PetscCall(MatProductCreateWithMat(aij->A, bloc, NULL, cloc));
    PetscCall(MatProductSetType(cloc, MATPRODUCT_AtB));
    PetscCall(MatProductSetFromOptions(cloc));
    PetscCall(MatProductSymbolic(cloc));
  }
  C->product->data    = data;
  C->product->destroy = MatProductCtxDestroy_MPIAIJ_MPIDense_AtB;

  C->ops->transposematmultnumeric = MatTransposeMatMultNumeric_MPIAIJ_MPIDense;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatTransposeMatMultNumeric_MPIAIJ_MPIDense(Mat A, Mat B, Mat C)
{
  Mat_MPIAIJ          *aij  = (Mat_MPIAIJ *)A->data;
  Mat                  bloc = ((Mat_MPIDense *)B->data)->A, cloc = ((Mat_MPIDense *)C->data)->A;
  MPIAIJ_MPIDense_AtB *data;
  PetscInt             BN = B->cmap->N, nz = aij->B->cmap->n, n, clda;
  PetscBool            clear = PETSC_FALSE, flg;

  PetscFunctionBegin;
  MatCheckProduct(C, 3);
  data = (MPIAIJ_MPIDense_AtB *)C->product->data;
  PetscCheck(data != NULL, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Missing product struct");
  if (BN == 0) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCheck(data->workC->rmap->n == nz, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Number of rows of workC %" PetscInt_FMT " not equal to columns of off-diagonal block %" PetscInt_FMT, data->workC->rmap->n, nz);
  PetscCall(MatDenseGetLDA(C, &clda));
  if (PetscDefined(HAVE_CUPM)) {
    PetscCall(PetscObjectTypeCompare((PetscObject)C, MATMPIDENSE, &flg));
    if (flg) PetscCall(PetscObjectTypeCompare((PetscObject)A, MATMPIAIJ, &flg));
    clear = (PetscBool)!flg; // if either A or C is a device Mat, make sure MatProductClear() is called
  }
  // A_d^T B_loc overwrites the whole local block of C, so the reductions below can add to it
  PetscCall(MatMPIAIJ_MPIDenseProductNumeric_Private(aij->A, bloc, cloc, MATPRODUCT_AtB, clear));
  n = data->workC->cmap->n;
  // rebuilding the PetscSFs would need all processes to agree that C changed, which takes a global reduction on every numeric
  if (clda != data->clda && data->rootC == NULL) {
    PetscCall(MatCreate(PETSC_COMM_SELF, &data->rootC));
    PetscCall(MatSetSizes(data->rootC, cloc->rmap->n, n, cloc->rmap->n, n));
    PetscCall(MatSetType(data->rootC, ((PetscObject)cloc)->type_name));
    PetscCall(MatDenseSetLDA(data->rootC, data->clda));
    PetscCall(MatSetUp(data->rootC));
  }
  for (PetscInt i = 0; i < BN; i += n) {
    PetscInt cols = PetscMin(n, BN - i), k = cols == data->ncols[0] ? 0 : 1;
    PetscSF  sf    = data->sf[k];
    Mat      workC = data->workC, Bb = bloc, Cb = cloc;

    PetscCheck(sf != NULL && cols == data->ncols[k], PETSC_COMM_SELF, PETSC_ERR_PLIB, "No reduction set up for %" PetscInt_FMT " columns", cols);
    if (cols != n) PetscCall(MatDenseGetSubMatrix(data->workC, PETSC_DECIDE, PETSC_DECIDE, 0, cols, &workC));
    // A_o^T times this batch of columns of B_loc, for rows of C owned by other processes
    if (nz > 0) {
      if (cols != BN) PetscCall(MatDenseGetSubMatrix(bloc, PETSC_DECIDE, PETSC_DECIDE, i, i + cols, &Bb));
      PetscCall(MatMPIAIJ_MPIDenseProductNumeric_Private(aij->B, Bb, workC, MATPRODUCT_AtB, clear));
      if (cols != BN) PetscCall(MatDenseRestoreSubMatrix(bloc, &Bb));
    }
    // a process with no off-diagonal columns still takes part: the reduction is collective and it may own rows of C that other processes add to
    if (cols != BN) PetscCall(MatDenseGetSubMatrix(cloc, PETSC_DECIDE, PETSC_DECIDE, i, i + cols, &Cb));
    if (clda == data->clda) PetscCall(MatDenseScatterWithSF_Private(sf, workC, Cb, MPIU_SUM, SCATTER_REVERSE, PETSC_FALSE));
    else {
      Mat Rb = data->rootC;

      if (cols != n) PetscCall(MatDenseGetSubMatrix(data->rootC, PETSC_DECIDE, PETSC_DECIDE, 0, cols, &Rb));
      PetscCall(MatZeroEntries(Rb));
      PetscCall(MatDenseScatterWithSF_Private(sf, workC, Rb, MPIU_SUM, SCATTER_REVERSE, PETSC_FALSE));
      PetscCall(MatAXPY(Cb, 1.0, Rb, SAME_NONZERO_PATTERN));
      if (cols != n) PetscCall(MatDenseRestoreSubMatrix(data->rootC, &Rb));
    }
    if (cols != BN) PetscCall(MatDenseRestoreSubMatrix(cloc, &Cb));
    if (cols != n) PetscCall(MatDenseRestoreSubMatrix(data->workC, &workC));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
