#include "symbrdn.h" /*I "petscksp.h" I*/
#include <petscblaslapack.h>

/* Compute the forward symmetric Broyden scaling parameter phi corresponding the already known value of the "bad"
   symmetric Broyden scaling parameter psi */
static inline PetscScalar PhiFromPsi(PetscScalar psi, PetscScalar yts, PetscScalar stBs, PetscScalar ytHy)
{
  PetscScalar numer = (1.0 - psi) * PetscRealPart(PetscConj(yts) * yts);
  PetscScalar phi   = numer / (numer + psi * stBs * ytHy);
  return phi;
}

/* The symmetric Broyden update can be written as

                   [         |     ] [ a_k | b_k ] [ s_k^T B_k ]
   B_{k+1} = B_k + [ B_k s_k | y_k ] [-----+-----] [-----------]
                   [         |     ] [ b_k | c_k ] [    y_k^T  ]

   We can unroll this as

                          [         |     ] [ a_i | b_i ] [ s_i^T B_i ]
   B_{k+1} = B_0 + \sum_i [ B_i s_i | y_i ] [-----+-----] [-----------]
                          [         |     ] [ b_i | c_i ] [    y_i^T  ]

   The a_i, b_i, and c_i values are stored in M00, M01, and M11, and are computed in
   SymBroydenRecursiveBasisUpdate() below
 */
static PetscErrorCode SymBroydenKernel_Recursive_Inner(Mat B, MatLMVMMode mode, PetscInt oldest, PetscInt next, Vec X, Vec B0X)
{
  Mat_LMVM        *lmvm = (Mat_LMVM *)B->data;
  Mat_SymBrdn     *lsb  = (Mat_SymBrdn *)lmvm->ctx;
  MatLMVMBasisType Y_t  = LMVMModeMap(LMBASIS_Y, mode);
  LMBasis          BkS  = lsb->basis[LMVMModeMap(SYMBROYDEN_BASIS_BKS, mode)];
  LMProducts       M00  = lsb->products[LMVMModeMap(SYMBROYDEN_PRODUCTS_M00, mode)];
  LMProducts       M01  = lsb->products[LMVMModeMap(SYMBROYDEN_PRODUCTS_M01, mode)];
  LMProducts       M11  = lsb->products[LMVMModeMap(SYMBROYDEN_PRODUCTS_M11, mode)];
  LMBasis          Y;
  Vec              StBkX, YtX, U, V;

  PetscFunctionBegin;
  PetscCall(MatLMVMGetUpdatedBasis(B, Y_t, &Y, NULL, NULL));
  PetscCall(MatLMVMGetWorkRow(B, &StBkX));
  PetscCall(MatLMVMGetWorkRow(B, &YtX));
  PetscCall(MatLMVMGetWorkRow(B, &U));
  PetscCall(MatLMVMGetWorkRow(B, &V));
  PetscCall(LMBasisGEMVH(BkS, oldest, next, 1.0, X, 0.0, StBkX));
  PetscCall(LMBasisGEMVH(Y, oldest, next, 1.0, X, 0.0, YtX));
  PetscCall(LMProductsMult(M00, oldest, next, 1.0, StBkX, 0.0, U, PETSC_FALSE));
  PetscCall(LMProductsMult(M01, oldest, next, 1.0, YtX, 1.0, U, PETSC_FALSE));
  PetscCall(LMProductsMult(M01, oldest, next, 1.0, StBkX, 0.0, V, PETSC_FALSE));
  PetscCall(LMProductsMult(M11, oldest, next, 1.0, YtX, 1.0, V, PETSC_FALSE));
  PetscCall(LMBasisGEMV(BkS, oldest, next, 1.0, U, 1.0, B0X));
  PetscCall(LMBasisGEMV(Y, oldest, next, 1.0, V, 1.0, B0X));
  PetscCall(MatLMVMRestoreWorkRow(B, &V));
  PetscCall(MatLMVMRestoreWorkRow(B, &U));
  PetscCall(MatLMVMRestoreWorkRow(B, &YtX));
  PetscCall(MatLMVMRestoreWorkRow(B, &StBkX));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatLMVMSymBroydenGetConvexFactor(Mat B, SymBroydenProductsType Phi_t, LMProducts *Phi)
{
  Mat_LMVM    *lmvm = (Mat_LMVM *)B->data;
  Mat_SymBrdn *lsb  = (Mat_SymBrdn *)lmvm->ctx;
  PetscScalar  phi  = Phi_t == SYMBROYDEN_PRODUCTS_PHI ? lsb->phi_scalar : lsb->psi_scalar;

  PetscFunctionBegin;
  if (!lsb->products[Phi_t]) PetscCall(MatLMVMCreateProducts(B, LMBLOCK_DIAGONAL, &lsb->products[Phi_t]));
  *Phi = lsb->products[Phi_t];
  if (phi != PETSC_DETERMINE) {
    PetscInt oldest, next, start;

    PetscCall(MatLMVMGetRange(B, &oldest, &next));
    start     = PetscMax((*Phi)->k, oldest);
    (*Phi)->k = start;
    for (PetscInt i = start; i < next; i++) PetscCall(LMProductsInsertNextDiagonalValue(*Phi, i, phi));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SymBroydenRecursiveBasisUpdate(Mat B, MatLMVMMode mode, PetscBool update_phi_from_psi)
{
  Mat_LMVM              *lmvm    = (Mat_LMVM *)B->data;
  Mat_SymBrdn           *lsb     = (Mat_SymBrdn *)lmvm->ctx;
  MatLMVMBasisType       S_t     = LMVMModeMap(LMBASIS_S, mode);
  MatLMVMBasisType       B0S_t   = LMVMModeMap(LMBASIS_B0S, mode);
  SymBroydenProductsType Phi_t   = LMVMModeMap(SYMBROYDEN_PRODUCTS_PHI, mode);
  SymBroydenProductsType Psi_t   = LMVMModeMap(SYMBROYDEN_PRODUCTS_PSI, mode);
  SymBroydenProductsType StBkS_t = LMVMModeMap(SYMBROYDEN_PRODUCTS_STBKS, mode);
  SymBroydenProductsType YtHkY_t = LMVMModeMap(SYMBROYDEN_PRODUCTS_YTHKY, mode);
  SymBroydenProductsType M00_t   = LMVMModeMap(SYMBROYDEN_PRODUCTS_M00, mode);
  SymBroydenProductsType M01_t   = LMVMModeMap(SYMBROYDEN_PRODUCTS_M01, mode);
  SymBroydenProductsType M11_t   = LMVMModeMap(SYMBROYDEN_PRODUCTS_M11, mode);
  SymBroydenProductsType M_t[3]  = {M00_t, M01_t, M11_t};
  LMProducts             M[3];
  LMProducts             Phi, Psi = NULL;
  SymBroydenBasisType    BkS_t = LMVMModeMap(SYMBROYDEN_BASIS_BKS, mode);
  LMBasis                BkS;
  LMProducts             StBkS, YtHkY = NULL;
  PetscInt               oldest, start, next;
  PetscInt               products_oldest;
  LMBasis                S;

  PetscFunctionBegin;
  PetscCall(MatLMVMGetRange(B, &oldest, &next));
  for (PetscInt i = 0; i < 3; i++) {
    if (lsb->products[M_t[i]] && lsb->products[M_t[i]]->block_type != LMBLOCK_DIAGONAL) PetscCall(LMProductsDestroy(&lsb->products[M_t[i]]));
    if (!lsb->products[M_t[i]]) PetscCall(MatLMVMCreateProducts(B, LMBLOCK_DIAGONAL, &lsb->products[M_t[i]]));
    M[i] = lsb->products[M_t[i]];
  }
  if (!lsb->basis[BkS_t]) PetscCall(LMBasisCreate(MatLMVMBasisSizeOf(B0S_t) == LMBASIS_S ? lmvm->Xprev : lmvm->Fprev, lmvm->m, &lsb->basis[BkS_t]));
  BkS = lsb->basis[BkS_t];
  PetscCall(MatLMVMSymBroydenGetConvexFactor(B, Phi_t, &Phi));
  if (!lsb->products[StBkS_t]) PetscCall(MatLMVMCreateProducts(B, LMBLOCK_DIAGONAL, &lsb->products[StBkS_t]));
  StBkS = lsb->products[StBkS_t];
  PetscCall(LMProductsPrepare(StBkS, lmvm->J0, oldest, next));
  products_oldest = PetscMax(0, StBkS->k - lmvm->m);
  if (oldest > products_oldest) {
    // recursion is starting from a different starting index, it must be recomputed
    StBkS->k = oldest;
  }
  BkS->k = start = StBkS->k;
  for (PetscInt i = 0; i < 3; i++) M[i]->k = start;
  if (start == next) PetscFunctionReturn(PETSC_SUCCESS);

  if (update_phi_from_psi) {
    Phi->k = start;
    // we have to first make sure that the inverse data is up to date
    PetscCall(SymBroydenRecursiveBasisUpdate(B, (MatLMVMMode)(mode ^ 1), PETSC_FALSE));
    PetscCall(MatLMVMSymBroydenGetConvexFactor(B, Psi_t, &Psi));
    YtHkY = lsb->products[YtHkY_t];
  }
  PetscCall(MatLMVMGetUpdatedBasis(B, S_t, &S, NULL, NULL));

  for (PetscInt j = start; j < next; j++) {
    Vec         p_j, s_j, B0s_j;
    PetscScalar alpha, sjtbjsj;
    PetscScalar m_00, m_01, m_11, yjtsj;
    PetscScalar phi_j;

    PetscCall(LMBasisGetWorkVec(BkS, &p_j));
    // p_j starts as B_0 * s_j
    PetscCall(MatLMVMBasisGetVecRead(B, B0S_t, j, &B0s_j, &alpha));
    PetscCall(VecAXPBY(p_j, alpha, 0.0, B0s_j));
    PetscCall(MatLMVMBasisRestoreVecRead(B, B0S_t, j, &B0s_j, &alpha));

    // Use the matmult kernel to compute p_j = B_j * p_j
    PetscCall(LMBasisGetVecRead(S, j, &s_j));
    if (j > oldest) PetscCall(SymBroydenKernel_Recursive_Inner(B, mode, oldest, j, s_j, p_j));
    PetscCall(VecDot(p_j, s_j, &sjtbjsj));
    PetscCall(LMBasisRestoreVecRead(S, j, &s_j));
    PetscCall(LMProductsInsertNextDiagonalValue(StBkS, j, sjtbjsj));
    PetscCall(LMBasisSetNextVec(BkS, p_j));
    PetscCall(LMBasisRestoreWorkVec(BkS, &p_j));

    PetscCall(MatLMVMProductsGetDiagonalValue(B, LMBASIS_Y, LMBASIS_S, j, &yjtsj));
    if (update_phi_from_psi) {
      PetscScalar psi_j;
      PetscScalar yjthjyj;

      PetscCall(LMProductsGetDiagonalValue(YtHkY, j, &yjthjyj));
      PetscCall(LMProductsGetDiagonalValue(Psi, j, &psi_j));

      phi_j = PhiFromPsi(psi_j, yjtsj, sjtbjsj, yjthjyj);
      PetscCall(LMProductsInsertNextDiagonalValue(Phi, j, phi_j));
    } else PetscCall(LMProductsGetDiagonalValue(Phi, j, &phi_j));

    /* The symmetric Broyden update can be represented as

       [     |     ][ (phi - 1) / s_j^T p_j |              -phi / y_j^T s_j                 ][ p_j^T ]
       [ p_j | y_j ][-----------------------+-----------------------------------------------][-------]
       [     |     ][    -phi / y_j^T s_j   | (y_j^T s_j + phi * s_j^T p_j) / (y_j^T s_j)^2 ][ y_j^T ]

       We store diagonal vectors with these values
     */

    m_00 = (phi_j - 1.0) / sjtbjsj;
    m_01 = -phi_j / yjtsj;
    m_11 = (yjtsj + phi_j * sjtbjsj) / (yjtsj * yjtsj);
    PetscCall(LMProductsInsertNextDiagonalValue(M[0], j, m_00));
    PetscCall(LMProductsInsertNextDiagonalValue(M[1], j, m_01));
    PetscCall(LMProductsInsertNextDiagonalValue(M[2], j, m_11));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode SymBroydenKernel_Recursive(Mat B, MatLMVMMode mode, Vec X, Vec Y, PetscBool update_phi_from_psi)
{
  PetscInt oldest, next;

  PetscFunctionBegin;
  PetscCall(MatLMVMApplyJ0Mode(mode)(B, X, Y));
  PetscCall(MatLMVMGetRange(B, &oldest, &next));
  if (next > oldest) {
    PetscCall(SymBroydenRecursiveBasisUpdate(B, mode, update_phi_from_psi));
    PetscCall(SymBroydenKernel_Recursive_Inner(B, mode, oldest, next, X, Y));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMult_LMVMSymBrdn_Recursive(Mat B, Vec X, Vec Y)
{
  Mat_LMVM    *lmvm = (Mat_LMVM *)B->data;
  Mat_SymBrdn *lsb  = (Mat_SymBrdn *)lmvm->ctx;

  PetscFunctionBegin;
  PetscCheck(lsb->phi_scalar != PETSC_DETERMINE, PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_WRONGSTATE, "Phi must first be set using MatLMVMSymBroydenSetPhi()");
  if (lsb->phi_scalar == 0.0) {
    PetscCall(BFGSKernel_Recursive(B, MATLMVM_MODE_PRIMAL, X, Y));
  } else if (lsb->phi_scalar == 1.0) {
    PetscCall(DFPKernel_Recursive(B, MATLMVM_MODE_PRIMAL, X, Y));
  } else {
    PetscCall(SymBroydenKernel_Recursive(B, MATLMVM_MODE_PRIMAL, X, Y, PETSC_FALSE));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSolve_LMVMSymBrdn_Recursive(Mat B, Vec X, Vec Y)
{
  Mat_LMVM    *lmvm = (Mat_LMVM *)B->data;
  Mat_SymBrdn *lsb  = (Mat_SymBrdn *)lmvm->ctx;

  PetscFunctionBegin;
  PetscCheck(lsb->phi_scalar != PETSC_DETERMINE, PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_WRONGSTATE, "Phi must first be set using MatLMVMSymBroydenSetPhi()");
  if (lsb->phi_scalar == 0.0) {
    PetscCall(DFPKernel_Recursive(B, MATLMVM_MODE_DUAL, X, Y));
  } else if (lsb->phi_scalar == 1.0) {
    PetscCall(BFGSKernel_Recursive(B, MATLMVM_MODE_DUAL, X, Y));
  } else {
    PetscCall(SymBroydenKernel_Recursive(B, MATLMVM_MODE_DUAL, X, Y, PETSC_TRUE));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscBool  ErwayMarciaCite       = PETSC_FALSE;
const char ErwayMarciaCitation[] = "@article{Erway2015,"
                                   "  title = {On Efficiently Computing the Eigenvalues of Limited-Memory Quasi-Newton Matrices},"
                                   "  volume = {36},"
                                   "  ISSN = {1095-7162},"
                                   "  url = {http://dx.doi.org/10.1137/140997737},"
                                   "  DOI = {10.1137/140997737},"
                                   "  number = {3},"
                                   "  journal = {SIAM Journal on Matrix Analysis and Applications},"
                                   "  publisher = {Society for Industrial & Applied Mathematics (SIAM)},"
                                   "  author = {Erway,  Jennifer B. and Marcia,  Roummel F.},"
                                   "  year = {2015},"
                                   "  month = jan,"
                                   "  pages = {1338-1359}"
                                   "}\n";

// TODO: on device (e.g. cuBLAS) implementation?
static PetscErrorCode SymBroydenCompactDenseUpdateArrays(PetscBLASInt m, PetscBLASInt oldest, PetscBLASInt next, PetscScalar M00[], PetscBLASInt lda00, PetscScalar M01[], PetscBLASInt lda01, PetscScalar M11[], PetscBLASInt lda11, const PetscScalar StB0S[], PetscBLASInt ldasbs, const PetscScalar YtS[], PetscBLASInt ldays, const PetscScalar Phi[], PetscScalar p0[], PetscScalar p1[], const PetscScalar Psi[], const PetscScalar YtHkY[], PetscScalar StBkS[])
{
  PetscBLASInt i;
  PetscScalar  alpha, beta, delta;
  PetscBLASInt ione  = 1;
  PetscScalar  sone  = 1.0;
  PetscScalar  szero = 0.0;
  PetscScalar  sBis;
  PetscScalar  yts;
  PetscScalar  phi;

  PetscFunctionBegin;
  if (next <= oldest || m <= 0) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscCitationsRegister(ErwayMarciaCitation, &ErwayMarciaCite));

  PetscCall(PetscArrayzero(M00, m * lda00));
  PetscCall(PetscArrayzero(M01, m * lda01));
  PetscCall(PetscArrayzero(M11, m * lda11));

  i = oldest % m;

  // Base case entries
  sBis = StB0S[i + i * ldasbs];
  if (StBkS) StBkS[i] = sBis;
  yts = YtS[i + i * ldays];
  if (Psi) {
    phi = PhiFromPsi(Psi[i], yts, sBis, YtHkY[i]);
  } else {
    phi = Phi[i];
  }
  alpha              = PetscRealPart(-(1.0 - phi) / sBis);
  beta               = -phi / yts;
  delta              = (1.0 + phi * sBis / yts) / yts;
  M00[i + i * lda00] = alpha;
  M01[i + i * lda01] = beta;
  M11[i + i * lda11] = delta;
  for (PetscBLASInt i_ = oldest + 1; i_ < next; i_++) {
    const PetscScalar *q0, *q1;
    PetscScalar        qp;

    i  = i_ % m;
    q0 = &StB0S[0 + i * ldasbs];
    q1 = &YtS[0 + i * ldays];

    // p_i <- M_{i-1} q_i

    PetscCallBLAS("BLASgemv", BLASgemv_("N", &m, &m, &sone, M00, &lda00, q0, &ione, &szero, p0, &ione));
    PetscCallBLAS("BLASgemv", BLASgemv_("N", &m, &m, &sone, M01, &lda01, q1, &ione, &sone, p0, &ione));
    PetscCallBLAS("BLASgemv", BLASgemv_("C", &m, &m, &sone, M01, &lda01, q0, &ione, &szero, p1, &ione));
    PetscCallBLAS("BLASgemv", BLASgemv_("N", &m, &m, &sone, M11, &lda11, q1, &ione, &sone, p1, &ione));

    // q'p
    PetscCallBLAS("BLASdot", qp = BLASdot_(&m, q0, &ione, p0, &ione));
    PetscCallBLAS("BLASdot", qp += BLASdot_(&m, q1, &ione, p1, &ione));

    sBis = StB0S[i + i * ldasbs] + qp;
    if (StBkS) StBkS[i] = sBis;
    yts = YtS[i + i * ldays];
    if (Psi) {
      phi = PhiFromPsi(Psi[i], yts, sBis, YtHkY[i]);
    } else {
      phi = Phi[i];
    }

    alpha = PetscRealPart(-(1.0 - phi) / sBis);
    beta  = -phi / yts;
    delta = (1.0 + phi * sBis / yts) / yts;

    PetscCallBLAS("LAPACKgerc", LAPACKgerc_(&m, &m, &alpha, p0, &ione, p0, &ione, M00, &lda00));
    for (PetscInt j = 0; j < m; j++) M00[j + i * lda00] = alpha * p0[j];
    for (PetscInt j = 0; j < m; j++) M00[i + j * lda00] = PetscConj(alpha * p0[j]);
    M00[i + i * lda00] = alpha;

    PetscCallBLAS("LAPACKgerc", LAPACKgerc_(&m, &m, &alpha, p0, &ione, p1, &ione, M01, &lda01));
    for (PetscBLASInt j = 0; j < m; j++) M01[j + i * lda01] = beta * p0[j];
    for (PetscBLASInt j = 0; j < m; j++) M01[i + j * lda01] = PetscConj(alpha * p1[j]);
    M01[i + i * lda01] = beta;

    PetscCallBLAS("LAPACKgerc", LAPACKgerc_(&m, &m, &alpha, p1, &ione, p1, &ione, M11, &lda11));
    for (PetscInt j = 0; j < m; j++) M11[j + i * lda11] = beta * p1[j];
    for (PetscInt j = 0; j < m; j++) M11[i + j * lda11] = PetscConj(beta * p1[j]);
    M11[i + i * lda11] = delta;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SymBroydenCompactProductsUpdate(Mat B, MatLMVMMode mode, PetscBool update_phi_from_psi)
{
  Mat_LMVM              *lmvm = (Mat_LMVM *)B->data;
  Mat_SymBrdn           *lsb  = (Mat_SymBrdn *)lmvm->ctx;
  PetscInt               oldest, next;
  MatLMVMBasisType       S_t     = LMVMModeMap(LMBASIS_S, mode);
  MatLMVMBasisType       B0S_t   = LMVMModeMap(LMBASIS_B0S, mode);
  MatLMVMBasisType       Y_t     = LMVMModeMap(LMBASIS_Y, mode);
  SymBroydenProductsType Phi_t   = LMVMModeMap(SYMBROYDEN_PRODUCTS_PHI, mode);
  SymBroydenProductsType Psi_t   = LMVMModeMap(SYMBROYDEN_PRODUCTS_PSI, mode);
  SymBroydenProductsType StBkS_t = LMVMModeMap(SYMBROYDEN_PRODUCTS_STBKS, mode);
  SymBroydenProductsType YtHkY_t = LMVMModeMap(SYMBROYDEN_PRODUCTS_YTHKY, mode);
  SymBroydenProductsType M00_t   = LMVMModeMap(SYMBROYDEN_PRODUCTS_M00, mode);
  SymBroydenProductsType M01_t   = LMVMModeMap(SYMBROYDEN_PRODUCTS_M01, mode);
  SymBroydenProductsType M11_t   = LMVMModeMap(SYMBROYDEN_PRODUCTS_M11, mode);
  SymBroydenProductsType M_t[3]  = {M00_t, M01_t, M11_t};
  Mat                    M_local[3];
  LMProducts             M[3], Phi, Psi, YtS, StB0S, StBkS, YtHkY;
  PetscInt               products_oldest, k;
  PetscBool              local_is_nonempty;

  PetscFunctionBegin;
  PetscCall(MatLMVMGetRange(B, &oldest, &next));
  for (PetscInt i = 0; i < 3; i++) {
    if (lsb->products[M_t[i]] && lsb->products[M_t[i]]->block_type != LMBLOCK_FULL) PetscCall(LMProductsDestroy(&lsb->products[M_t[i]]));
    if (!lsb->products[M_t[i]]) PetscCall(MatLMVMCreateProducts(B, LMBLOCK_FULL, &lsb->products[M_t[i]]));
    M[i] = lsb->products[M_t[i]];
  }
  PetscCall(MatLMVMSymBroydenGetConvexFactor(B, Phi_t, &Phi));
  PetscCall(MatLMVMSymBroydenGetConvexFactor(B, Psi_t, &Psi));
  if (!lsb->products[StBkS_t]) PetscCall(MatLMVMCreateProducts(B, LMBLOCK_DIAGONAL, &lsb->products[StBkS_t]));
  StBkS = lsb->products[StBkS_t];
  if (!lsb->products[YtHkY_t]) PetscCall(MatLMVMCreateProducts(B, LMBLOCK_DIAGONAL, &lsb->products[YtHkY_t]));
  YtHkY = lsb->products[YtHkY_t];
  PetscCall(LMProductsPrepare(M[0], lmvm->J0, oldest, next));
  PetscCall(LMProductsGetLocalMatrix(M[0], &M_local[0], &k, &local_is_nonempty));
  products_oldest = PetscMax(0, k - lmvm->m);
  if (products_oldest < oldest) k = oldest;
  if (k < next) {
    Mat StB0S_local, YtS_local;
    Vec Psi_local = NULL, Phi_local = NULL, YtHkY_local = NULL, StBkS_local = NULL;

    PetscCall(MatLMVMGetUpdatedProducts(B, Y_t, S_t, LMBLOCK_UPPER_TRIANGLE, &YtS));
    PetscCall(MatLMVMGetUpdatedProducts(B, S_t, B0S_t, LMBLOCK_UPPER_TRIANGLE, &StB0S));
    PetscCall(LMProductsGetLocalMatrix(StB0S, &StB0S_local, NULL, NULL));
    PetscCall(LMProductsGetLocalMatrix(YtS, &YtS_local, NULL, NULL));
    PetscCall(LMProductsGetLocalMatrix(M[1], &M_local[1], NULL, NULL));
    PetscCall(LMProductsGetLocalMatrix(M[2], &M_local[2], NULL, NULL));
    if (update_phi_from_psi) {
      PetscCall(LMProductsGetLocalDiagonal(Psi, &Psi_local));
      PetscCall(LMProductsGetLocalDiagonal(YtHkY, &YtHkY_local));
    } else {
      PetscCall(LMProductsGetLocalDiagonal(Phi, &Phi_local));
      PetscCall(LMProductsGetLocalDiagonal(StBkS, &StBkS_local));
    }
    if (local_is_nonempty) {
      PetscInt           lda;
      PetscBLASInt       M_lda[3], StB0S_lda, YtS_lda, m_blas, oldest_blas, next_blas;
      const PetscScalar *StB0S_;
      const PetscScalar *YtS_;
      PetscScalar       *M_[3];
      const PetscScalar *Phi_   = NULL;
      const PetscScalar *Psi_   = NULL;
      const PetscScalar *YtHkY_ = NULL;
      PetscScalar       *StBkS_ = NULL;
      PetscScalar       *p0, *p1;

      for (PetscInt i = 0; i < 3; i++) {
        PetscCall(MatDenseGetLDA(M_local[i], &lda));
        PetscCall(PetscBLASIntCast(lda, &M_lda[i]));
        PetscCall(MatDenseGetArrayWrite(M_local[i], &M_[i]));
      }

      PetscCall(MatDenseGetArrayRead(StB0S_local, &StB0S_));
      PetscCall(MatDenseGetLDA(StB0S_local, &lda));
      PetscCall(PetscBLASIntCast(lda, &StB0S_lda));

      PetscCall(MatDenseGetArrayRead(YtS_local, &YtS_));
      PetscCall(MatDenseGetLDA(YtS_local, &lda));
      PetscCall(PetscBLASIntCast(lda, &YtS_lda));

      PetscCall(PetscBLASIntCast(lmvm->m, &m_blas));
      PetscCall(PetscBLASIntCast(oldest, &oldest_blas));
      PetscCall(PetscBLASIntCast(next, &next_blas));

      if (update_phi_from_psi) {
        PetscCall(VecGetArrayRead(Psi_local, &Psi_));
        PetscCall(VecGetArrayRead(YtHkY_local, &YtHkY_));
      } else {
        PetscCall(VecGetArrayRead(Phi_local, &Phi_));
        PetscCall(VecGetArrayWrite(StBkS_local, &StBkS_));
      }

      PetscCall(PetscMalloc2(lmvm->m, &p0, lmvm->m, &p1));
      PetscCall(SymBroydenCompactDenseUpdateArrays(m_blas, oldest_blas, next_blas, M_[0], M_lda[0], M_[1], M_lda[1], M_[2], M_lda[2], StB0S_, StB0S_lda, YtS_, YtS_lda, Phi_, p0, p1, Psi_, YtHkY_, StBkS_));
      PetscCall(PetscFree2(p0, p1));

      if (update_phi_from_psi) {
        PetscCall(VecRestoreArrayRead(YtHkY_local, &YtHkY_));
        PetscCall(VecRestoreArrayRead(Psi_local, &Psi_));
      } else {
        PetscCall(VecRestoreArrayWrite(StBkS_local, &StBkS_));
        PetscCall(VecRestoreArrayRead(Phi_local, &Phi_));
      }

      for (PetscInt i = 0; i < 3; i++) PetscCall(MatDenseRestoreArrayWrite(M_local[i], &M_[i]));
      PetscCall(MatDenseRestoreArrayRead(YtS_local, &YtS_));
      PetscCall(MatDenseRestoreArrayRead(StB0S_local, &StB0S_));
    }
    if (update_phi_from_psi) {
      PetscCall(LMProductsRestoreLocalDiagonal(YtHkY, &YtHkY_local));
      PetscCall(LMProductsRestoreLocalDiagonal(Psi, &Psi_local));
    } else {
      PetscCall(LMProductsRestoreLocalDiagonal(StBkS, &StBkS_local));
      PetscCall(LMProductsRestoreLocalDiagonal(Phi, &Phi_local));
    }
    PetscCall(LMProductsRestoreLocalMatrix(M[2], &M_local[2], &next));
    PetscCall(LMProductsRestoreLocalMatrix(M[1], &M_local[1], &next));
    PetscCall(LMProductsRestoreLocalMatrix(YtS, &YtS_local, NULL));
    PetscCall(LMProductsRestoreLocalMatrix(StB0S, &StB0S_local, NULL));
  }
  PetscCall(LMProductsRestoreLocalMatrix(M[0], &M_local[0], &next));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode SymBroydenCompactDenseKernelUseB0S(Mat B, MatLMVMMode mode, Vec X, PetscBool *use_B0S)
{
  Mat_LMVM        *lmvm = (Mat_LMVM *)B->data;
  PetscBool        is_scalar;
  PetscScalar      J0_scale;
  LMBasis          B0S;
  Mat              J0 = lmvm->J0;
  PetscObjectId    id;
  PetscObjectState state;

  /*
     The one time where we would want to compute S^T B_0 X as (B_0 S)^T X instead of S^T (B_0 X)
     is if (B_0 S)^T X is cached.
   */
  PetscFunctionBegin;
  *use_B0S = PETSC_FALSE;
  PetscCall(MatLMVMGetJ0Scalar(B, &is_scalar, &J0_scale));
  B0S = lmvm->basis[is_scalar ? LMVMModeMap(LMBASIS_S, mode) : LMVMModeMap(LMBASIS_B0S, mode)];
  if ((B0S->k < lmvm->k) || (B0S->cached_product == NULL)) PetscFunctionReturn(PETSC_SUCCESS);
  if (!is_scalar) {
    PetscCall(PetscObjectGetId((PetscObject)J0, &id));
    PetscCall(PetscObjectStateGet((PetscObject)J0, &state));
    if (id != B0S->operator_id || state != B0S->operator_state) PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(PetscObjectGetId((PetscObject)X, &id));
  PetscCall(PetscObjectStateGet((PetscObject)X, &state));
  if (id == B0S->cached_vec_id && state == B0S->cached_vec_state) *use_B0S = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode SymBroydenKernel_CompactDense(Mat B, MatLMVMMode mode, Vec X, Vec BX, PetscBool update_phi_from_psi)
{
  PetscInt oldest, next;

  PetscFunctionBegin;
  PetscCall(MatLMVMApplyJ0Mode(mode)(B, X, BX));
  PetscCall(MatLMVMGetRange(B, &oldest, &next));
  if (next > oldest) {
    Mat_LMVM              *lmvm = (Mat_LMVM *)B->data;
    Mat_SymBrdn           *lsb  = (Mat_SymBrdn *)lmvm->ctx;
    Vec                    StB0X, YtX, u, v;
    MatLMVMBasisType       S_t   = LMVMModeMap(LMBASIS_S, mode);
    MatLMVMBasisType       Y_t   = LMVMModeMap(LMBASIS_Y, mode);
    MatLMVMBasisType       B0S_t = LMVMModeMap(LMBASIS_B0S, mode);
    SymBroydenProductsType M00_t = LMVMModeMap(SYMBROYDEN_PRODUCTS_M00, mode);
    SymBroydenProductsType M01_t = LMVMModeMap(SYMBROYDEN_PRODUCTS_M01, mode);
    SymBroydenProductsType M11_t = LMVMModeMap(SYMBROYDEN_PRODUCTS_M11, mode);
    LMProducts             M00, M01, M11;
    LMBasis                S, Y;
    PetscBool              use_B0S;

    if (update_phi_from_psi) PetscCall(SymBroydenCompactProductsUpdate(B, (MatLMVMMode)(mode ^ 1), PETSC_FALSE));
    PetscCall(SymBroydenCompactProductsUpdate(B, mode, update_phi_from_psi));
    M00 = lsb->products[M00_t];
    M01 = lsb->products[M01_t];
    M11 = lsb->products[M11_t];

    PetscCall(MatLMVMGetUpdatedBasis(B, S_t, &S, NULL, NULL));
    PetscCall(MatLMVMGetUpdatedBasis(B, Y_t, &Y, NULL, NULL));

    PetscCall(MatLMVMGetWorkRow(B, &StB0X));
    PetscCall(MatLMVMGetWorkRow(B, &YtX));
    PetscCall(MatLMVMGetWorkRow(B, &u));
    PetscCall(MatLMVMGetWorkRow(B, &v));

    PetscCall(SymBroydenCompactDenseKernelUseB0S(B, mode, X, &use_B0S));
    if (use_B0S) PetscCall(MatLMVMBasisGEMVH(B, B0S_t, oldest, next, 1.0, X, 0.0, StB0X));
    else PetscCall(LMBasisGEMVH(S, oldest, next, 1.0, BX, 0.0, StB0X));

    PetscCall(LMBasisGEMVH(Y, oldest, next, 1.0, X, 0.0, YtX));

    PetscCall(LMProductsMult(M00, oldest, next, 1.0, StB0X, 0.0, u, PETSC_FALSE));
    PetscCall(LMProductsMult(M01, oldest, next, 1.0, YtX, 1.0, u, PETSC_FALSE));
    PetscCall(LMProductsMult(M01, oldest, next, 1.0, StB0X, 0.0, v, PETSC_TRUE));
    PetscCall(LMProductsMult(M11, oldest, next, 1.0, YtX, 1.0, v, PETSC_FALSE));

    PetscCall(LMBasisGEMV(Y, oldest, next, 1.0, v, 1.0, BX));
    PetscCall(MatLMVMBasisGEMV(B, B0S_t, oldest, next, 1.0, u, 1.0, BX));

    PetscCall(MatLMVMRestoreWorkRow(B, &v));
    PetscCall(MatLMVMRestoreWorkRow(B, &u));
    PetscCall(MatLMVMRestoreWorkRow(B, &YtX));
    PetscCall(MatLMVMRestoreWorkRow(B, &StB0X));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMult_LMVMSymBrdn_CompactDense(Mat B, Vec X, Vec BX)
{
  Mat_LMVM    *lmvm = (Mat_LMVM *)B->data;
  Mat_SymBrdn *lsb  = (Mat_SymBrdn *)lmvm->ctx;

  PetscFunctionBegin;
  PetscCheck(lsb->phi_scalar != PETSC_DETERMINE, PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_WRONGSTATE, "Phi must first be set using MatLMVMSymBroydenSetPhi()");
  if (lsb->phi_scalar == 0.0) {
    PetscCall(BFGSKernel_CompactDense(B, MATLMVM_MODE_PRIMAL, X, BX));
  } else if (lsb->phi_scalar == 1.0) {
    PetscCall(DFPKernel_CompactDense(B, MATLMVM_MODE_PRIMAL, X, BX));
  } else {
    PetscCall(SymBroydenKernel_CompactDense(B, MATLMVM_MODE_PRIMAL, X, BX, PETSC_FALSE));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSolve_LMVMSymBrdn_CompactDense(Mat B, Vec X, Vec HX)
{
  Mat_LMVM    *lmvm = (Mat_LMVM *)B->data;
  Mat_SymBrdn *lsb  = (Mat_SymBrdn *)lmvm->ctx;

  PetscFunctionBegin;
  PetscCheck(lsb->phi_scalar != PETSC_DETERMINE, PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_WRONGSTATE, "Phi must first be set using MatLMVMSymBroydenSetPhi()");
  if (lsb->phi_scalar == 0.0) {
    PetscCall(DFPKernel_CompactDense(B, MATLMVM_MODE_DUAL, X, HX));
  } else if (lsb->phi_scalar == 1.0) {
    PetscCall(BFGSKernel_CompactDense(B, MATLMVM_MODE_DUAL, X, HX));
  } else {
    PetscCall(SymBroydenKernel_CompactDense(B, MATLMVM_MODE_DUAL, X, HX, PETSC_TRUE));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatUpdate_LMVMSymBrdn(Mat B, Vec X, Vec F)
{
  Mat_LMVM        *lmvm = (Mat_LMVM *)B->data;
  Mat_SymBrdn     *lsb  = (Mat_SymBrdn *)lmvm->ctx;
  PetscReal        curvtol;
  PetscScalar      curvature, ststmp;
  PetscInt         oldest, next;
  PetscBool        cache_StFprev   = (lmvm->mult_alg != MAT_LMVM_MULT_RECURSIVE) ? lmvm->cache_gradient_products : PETSC_FALSE;
  PetscBool        cache_YtH0Fprev = cache_StFprev;
  LMBasis          S = NULL, H0Y = NULL;
  PetscScalar      H0_alpha = 1.0;
  MatLMVMBasisType H0Y_t    = LMBASIS_H0Y;

  PetscFunctionBegin;
  if (!lmvm->m) PetscFunctionReturn(PETSC_SUCCESS);
  // BFGS using the dense algorithm does not need to cache YtH0F products
  if (lsb->phi_scalar == 0.0 && lmvm->mult_alg == MAT_LMVM_MULT_DENSE) cache_YtH0Fprev = PETSC_FALSE;
  // Caching is no use if we are diagonally updating
  if (lsb->rescale->scale_type == MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL) cache_YtH0Fprev = PETSC_FALSE;
  PetscCall(MatLMVMGetRange(B, &oldest, &next));
  if (lmvm->prev_set) {
    LMBasis Y         = NULL;
    Vec     Fprev_old = NULL;

    if (cache_StFprev || cache_YtH0Fprev) {
      PetscCall(MatLMVMGetUpdatedBasis(B, LMBASIS_Y, &Y, NULL, NULL));
      PetscCall(LMBasisGetWorkVec(Y, &Fprev_old));
      PetscCall(VecCopy(lmvm->Fprev, Fprev_old));
    }

    /* Compute the new (S = X - Xprev) and (Y = F - Fprev) vectors */
    PetscCall(VecAYPX(lmvm->Xprev, -1.0, X));
    PetscCall(VecAYPX(lmvm->Fprev, -1.0, F));

    /* Test if the updates can be accepted */
    {
      Vec         sy[2] = {lmvm->Xprev, lmvm->Fprev};
      PetscScalar stsy[2];

      PetscCall(VecMDot(lmvm->Xprev, 2, sy, stsy));
      ststmp    = stsy[0];
      curvature = stsy[1];
    }
    curvtol = lmvm->eps * PetscRealPart(ststmp);

    if (PetscRealPart(curvature) > curvtol) { /* Update is good, accept it */
      LMProducts StY           = NULL;
      LMProducts YtH0Y         = NULL;
      Vec        StFprev_old   = NULL;
      Vec        YtH0Fprev_old = NULL;
      PetscInt   oldest_new, next_new;

      lsb->watchdog = 0;
      if (cache_StFprev) {
        PetscCall(MatLMVMGetUpdatedBasis(B, LMBASIS_S, &S, NULL, NULL));
        if (!lsb->StFprev) PetscCall(LMBasisCreateRow(S, &lsb->StFprev));
        PetscCall(MatLMVMGetUpdatedProducts(B, LMBASIS_S, LMBASIS_Y, LMBLOCK_UPPER_TRIANGLE, &StY));
        PetscCall(LMProductsGetNextColumn(StY, &StFprev_old));
        PetscCall(VecCopy(lsb->StFprev, StFprev_old));
      }
      if (cache_YtH0Fprev) {
        PetscCall(MatLMVMGetUpdatedBasis(B, LMBASIS_H0Y, &H0Y, &H0Y_t, &H0_alpha));
        if (!lsb->YtH0Fprev) PetscCall(LMBasisCreateRow(H0Y, &lsb->YtH0Fprev));
        PetscCall(MatLMVMGetUpdatedProducts(B, LMBASIS_Y, H0Y_t, LMBLOCK_UPPER_TRIANGLE, &YtH0Y));
        PetscCall(LMProductsGetNextColumn(YtH0Y, &YtH0Fprev_old));
        if (lsb->YtH0Fprev == H0Y->cached_product) {
          PetscCall(VecCopy(lsb->YtH0Fprev, YtH0Fprev_old));
        } else {
          if (next > oldest) {
            // need to recalculate
            PetscCall(LMBasisGEMVH(H0Y, oldest, next, 1.0, Fprev_old, 0.0, YtH0Fprev_old));
          } else {
            PetscCall(VecZeroEntries(YtH0Fprev_old));
          }
        }
      }

      PetscCall(MatUpdateKernel_LMVM(B, lmvm->Xprev, lmvm->Fprev));
      PetscCall(MatLMVMGetRange(B, &oldest_new, &next_new));
      if (cache_StFprev) {
        // compute the one new s_i^T F_old value
        PetscCall(LMBasisGEMVH(S, next, next_new, 1.0, Fprev_old, 0.0, StFprev_old));
        PetscCall(LMBasisGEMVH(S, oldest_new, next_new, 1.0, F, 0.0, lsb->StFprev));
        PetscCall(LMBasisSetCachedProduct(S, F, lsb->StFprev));
        PetscCall(VecAXPBY(StFprev_old, 1.0, -1.0, lsb->StFprev));
        PetscCall(LMProductsRestoreNextColumn(StY, &StFprev_old));
      }
      if (cache_YtH0Fprev) {
        PetscCall(MatLMVMGetUpdatedBasis(B, LMBASIS_H0Y, &H0Y, &H0Y_t, &H0_alpha));
        // compute the one new (H_0 y_i)^T F_old value
        PetscCall(LMBasisGEMVH(H0Y, next, next_new, 1.0, Fprev_old, 0.0, YtH0Fprev_old));
        PetscCall(LMBasisGEMVH(H0Y, oldest_new, next_new, 1.0, F, 0.0, lsb->YtH0Fprev));
        PetscCall(LMBasisSetCachedProduct(H0Y, F, lsb->YtH0Fprev));
        PetscCall(VecAXPBY(YtH0Fprev_old, 1.0, -1.0, lsb->YtH0Fprev));
        PetscCall(LMProductsRestoreNextColumn(YtH0Y, &YtH0Fprev_old));
      }

      PetscCall(MatLMVMProductsInsertDiagonalValue(B, LMBASIS_Y, LMBASIS_S, next, PetscRealPart(curvature)));
      PetscCall(MatLMVMProductsInsertDiagonalValue(B, LMBASIS_S, LMBASIS_Y, next, PetscRealPart(curvature)));
      PetscCall(MatLMVMProductsInsertDiagonalValue(B, LMBASIS_S, LMBASIS_S, next, ststmp));
      PetscCall(SymBroydenRescaleUpdate(B, lsb->rescale));
    } else {
      /* Update is bad, skip it */
      PetscCall(PetscInfo(B, "Rejecting update: curvature %g, ||s||^2 %g\n", (double)PetscRealPart(curvature), (double)PetscRealPart(ststmp)));
      lmvm->nrejects++;
      lsb->watchdog++;
      if (cache_StFprev) {
        // we still need to update the cached product
        PetscCall(MatLMVMGetUpdatedBasis(B, LMBASIS_S, &S, NULL, NULL));
        PetscCall(LMBasisGEMVH(S, oldest, next, 1.0, F, 0.0, lsb->StFprev));
        PetscCall(LMBasisSetCachedProduct(S, F, lsb->StFprev));
      }
      if (cache_YtH0Fprev) {
        // we still need to update the cached product
        PetscCall(MatLMVMGetUpdatedBasis(B, LMBASIS_H0Y, &H0Y, &H0Y_t, &H0_alpha));
        PetscCall(LMBasisGEMVH(H0Y, oldest, next, 1.0, F, 0.0, lsb->YtH0Fprev));
        PetscCall(LMBasisSetCachedProduct(H0Y, F, lsb->StFprev));
      }
    }
    if (cache_StFprev || cache_YtH0Fprev) PetscCall(LMBasisRestoreWorkVec(Y, &Fprev_old));
  } else {
    if (cache_StFprev) {
      PetscCall(MatLMVMGetUpdatedBasis(B, LMBASIS_S, &S, NULL, NULL));
      if (!lsb->StFprev) PetscCall(LMBasisCreateRow(S, &lsb->StFprev));
    }
    if (cache_YtH0Fprev) {
      MatLMVMBasisType H0Y_t;
      PetscScalar      H0_alpha;

      PetscCall(MatLMVMGetUpdatedBasis(B, LMBASIS_H0Y, &H0Y, &H0Y_t, &H0_alpha));
      if (!lsb->YtH0Fprev) PetscCall(LMBasisCreateRow(H0Y, &lsb->YtH0Fprev));
    }
  }

  if (lsb->watchdog > lsb->max_seq_rejects) PetscCall(MatLMVMReset(B, PETSC_FALSE));

  /* Save the solution and function to be used in the next update */
  PetscCall(VecCopy(X, lmvm->Xprev));
  PetscCall(VecCopy(F, lmvm->Fprev));
  lmvm->prev_set = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatCopy_LMVMSymBrdn(Mat B, Mat M, MatStructure str)
{
  Mat_LMVM    *bdata = (Mat_LMVM *)B->data;
  Mat_SymBrdn *blsb  = (Mat_SymBrdn *)bdata->ctx;
  Mat_LMVM    *mdata = (Mat_LMVM *)M->data;
  Mat_SymBrdn *mlsb  = (Mat_SymBrdn *)mdata->ctx;

  PetscFunctionBegin;
  mlsb->phi_scalar      = blsb->phi_scalar;
  mlsb->psi_scalar      = blsb->psi_scalar;
  mlsb->watchdog        = blsb->watchdog;
  mlsb->max_seq_rejects = blsb->max_seq_rejects;
  PetscCall(SymBroydenRescaleCopy(blsb->rescale, mlsb->rescale));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatReset_LMVMSymBrdn_Internal(Mat B, MatLMVMResetMode mode)
{
  Mat_LMVM    *lmvm = (Mat_LMVM *)B->data;
  Mat_SymBrdn *lsb  = (Mat_SymBrdn *)lmvm->ctx;

  PetscFunctionBegin;
  if (MatLMVMResetClearsBases(mode)) {
    for (PetscInt i = 0; i < SYMBROYDEN_BASIS_COUNT; i++) PetscCall(LMBasisDestroy(&lsb->basis[i]));
    for (PetscInt i = 0; i < SYMBROYDEN_PRODUCTS_COUNT; i++) PetscCall(LMProductsDestroy(&lsb->products[i]));
    PetscCall(VecDestroy(&lsb->StFprev));
    PetscCall(VecDestroy(&lsb->YtH0Fprev));
  } else {
    for (PetscInt i = 0; i < SYMBROYDEN_BASIS_COUNT; i++) PetscCall(LMBasisReset(lsb->basis[i]));
    for (PetscInt i = 0; i < SYMBROYDEN_PRODUCTS_COUNT; i++) PetscCall(LMProductsReset(lsb->products[i]));
    if (lsb->StFprev) PetscCall(VecZeroEntries(lsb->StFprev));
    if (lsb->YtH0Fprev) PetscCall(VecZeroEntries(lsb->YtH0Fprev));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatReset_LMVMSymBrdn(Mat B, MatLMVMResetMode mode)
{
  Mat_LMVM    *lmvm = (Mat_LMVM *)B->data;
  Mat_SymBrdn *lsb  = (Mat_SymBrdn *)lmvm->ctx;

  PetscFunctionBegin;
  lsb->watchdog = 0;
  PetscCall(SymBroydenRescaleReset(B, lsb->rescale, mode));
  PetscCall(MatReset_LMVMSymBrdn_Internal(B, mode));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDestroy_LMVMSymBrdn(Mat B)
{
  Mat_LMVM    *lmvm = (Mat_LMVM *)B->data;
  Mat_SymBrdn *lsb  = (Mat_SymBrdn *)lmvm->ctx;

  PetscFunctionBegin;
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatLMVMSymBroydenGetPhi_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatLMVMSymBroydenSetPhi_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatLMVMSymBadBroydenGetPsi_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatLMVMSymBadBroydenSetPsi_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatLMVMSymBroydenSetDelta_C", NULL));
  PetscCall(SymBroydenRescaleDestroy(&lsb->rescale));
  PetscCall(MatReset_LMVMSymBrdn_Internal(B, MAT_LMVM_RESET_ALL));
  PetscCall(PetscFree(lmvm->ctx));
  PetscCall(MatDestroy_LMVM(B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSetUp_LMVMSymBrdn(Mat B)
{
  Mat_LMVM    *lmvm = (Mat_LMVM *)B->data;
  Mat_SymBrdn *lsb  = (Mat_SymBrdn *)lmvm->ctx;

  PetscFunctionBegin;
  PetscCall(MatSetUp_LMVM(B));
  PetscCall(SymBroydenRescaleInitializeJ0(B, lsb->rescale));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatView_LMVMSymBrdn(Mat B, PetscViewer pv)
{
  Mat_LMVM    *lmvm = (Mat_LMVM *)B->data;
  Mat_SymBrdn *lsb  = (Mat_SymBrdn *)lmvm->ctx;
  PetscBool    isascii;

  PetscFunctionBegin;
  PetscCall(MatView_LMVM(B, pv));
  PetscCall(PetscObjectTypeCompare((PetscObject)pv, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    PetscBool is_other;

    PetscCall(PetscObjectTypeCompareAny((PetscObject)B, &is_other, MATLMVMBFGS, MATLMVMDFP, ""));
    if (!is_other) {
      if (lsb->phi_scalar != PETSC_DETERMINE) PetscCall(PetscViewerASCIIPrintf(pv, "Convex factor phi = %g\n", (double)lsb->phi_scalar));
      if (lsb->psi_scalar != PETSC_DETERMINE) PetscCall(PetscViewerASCIIPrintf(pv, "Dual convex factor psi = %g\n", (double)lsb->psi_scalar));
    }
  }
  PetscCall(SymBroydenRescaleView(lsb->rescale, pv));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatLMVMSetMultAlgorithm_SymBrdn(Mat B)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;

  PetscFunctionBegin;
  switch (lmvm->mult_alg) {
  case MAT_LMVM_MULT_RECURSIVE:
    lmvm->ops->mult  = MatMult_LMVMSymBrdn_Recursive;
    lmvm->ops->solve = MatSolve_LMVMSymBrdn_Recursive;
    break;
  case MAT_LMVM_MULT_DENSE:
  case MAT_LMVM_MULT_COMPACT_DENSE:
    lmvm->ops->mult  = MatMult_LMVMSymBrdn_CompactDense;
    lmvm->ops->solve = MatSolve_LMVMSymBrdn_CompactDense;
    break;
  }
  lmvm->ops->multht  = lmvm->ops->mult;
  lmvm->ops->solveht = lmvm->ops->solve;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSetFromOptions_LMVMSymBrdn(Mat B, PetscOptionItems PetscOptionsObject)
{
  Mat_LMVM    *lmvm = (Mat_LMVM *)B->data;
  Mat_SymBrdn *lsb  = (Mat_SymBrdn *)lmvm->ctx;

  PetscFunctionBegin;
  PetscCall(MatSetFromOptions_LMVM(B, PetscOptionsObject));
  PetscOptionsHeadBegin(PetscOptionsObject, "Restricted/Symmetric Broyden method for approximating SPD Jacobian actions (MATLMVMSYMBRDN)");
  PetscCall(PetscOptionsReal("-mat_lmvm_phi", "convex ratio between BFGS and DFP components of the update", "", lsb->phi_scalar, &lsb->phi_scalar, NULL));
  PetscCheck(lsb->phi_scalar >= 0.0 && lsb->phi_scalar <= 1.0, PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_OUTOFRANGE, "convex ratio for the update formula cannot be outside the range of [0, 1]");
  PetscCall(SymBroydenRescaleSetFromOptions(B, lsb->rescale, PetscOptionsObject));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatLMVMSymBroydenGetPhi - Get the phi parameter for a Broyden class quasi-Newton update matrix

  Input Parameter:
. B - The matrix

  Output Parameter:
. phi - a number defining an update that is an affine combination of the BFGS update (phi = 0) and DFP update (phi = 1)

  Level: advanced

  Note:
  If `B` does not have a constant value of `phi` for all iterations this will
  return `phi` = `PETSC_DETERMINE` = -1, a negative value that `phi` cannot
  attain for a valid general Broyden update.
  This is the case if `B` is a `MATLMVMSYMBADBROYDEN`, where `phi`'s dual value
  `psi` is constant and `phi` changes from iteration to iteration.

.seealso: [](ch_ksp),
          `MATLMVMSYMBROYDEN`, `MATLMVMSYMBADBROYDEN`,
          `MATLMVMDFP`, `MATLMVMBFGS`,
          `MatLMVMSymBroydenSetPhi()`,
          `MatLMVMSymBadBroydenGetPsi()`, `MatLMVMSymBadBroydenSetPsi()`
@*/
PetscErrorCode MatLMVMSymBroydenGetPhi(Mat B, PetscReal *phi)
{
  PetscFunctionBegin;
  *phi = PETSC_DETERMINE;
  PetscUseMethod(B, "MatLMVMSymBroydenGetPhi_C", (Mat, PetscReal *), (B, phi));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatLMVMSymBroydenGetPhi_SymBrdn(Mat B, PetscReal *phi)
{
  Mat_LMVM    *lmvm = (Mat_LMVM *)B->data;
  Mat_SymBrdn *lsb  = (Mat_SymBrdn *)lmvm->ctx;

  PetscFunctionBegin;
  *phi = lsb->phi_scalar;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatLMVMSymBroydenSetPhi - Get the phi parameter for a Broyden class quasi-Newton update matrix

  Input Parameters:
+ B   - The matrix
- phi - a number defining an update that is a convex combination of the BFGS update (phi = 0) and DFP update (phi = 1)

  Level: advanced

  Note:
  If `B` cannot have a constant value of `phi` for all iterations this will be ignored.
  This is the case if `B` is a `MATLMVMSYMBADBROYDEN`, where `phi`'s dual value
  `psi` is constant and `phi` changes from iteration to iteration.

.seealso: [](ch_ksp),
          `MATLMVMSYMBROYDEN`, `MATLMVMSYMBADBROYDEN`,
          `MATLMVMDFP`, `MATLMVMBFGS`,
          `MatLMVMSymBroydenGetPhi()`,
          `MatLMVMSymBadBroydenGetPsi()`, `MatLMVMSymBadBroydenSetPsi()`
@*/
PetscErrorCode MatLMVMSymBroydenSetPhi(Mat B, PetscReal phi)
{
  PetscFunctionBegin;
  PetscTryMethod(B, "MatLMVMSymBroydenSetPhi_C", (Mat, PetscReal), (B, phi));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatLMVMSymBroydenSetPhi_SymBrdn(Mat B, PetscReal phi)
{
  Mat_LMVM    *lmvm = (Mat_LMVM *)B->data;
  Mat_SymBrdn *lsb  = (Mat_SymBrdn *)lmvm->ctx;

  PetscFunctionBegin;
  lsb->phi_scalar = phi;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatLMVMSymBadBroydenGetPsi - Get the psi parameter for a Broyden class quasi-Newton update matrix

  Input Parameter:
. B - The matrix

  Output Parameter:
. psi - a number defining an update that is an affine combination of the BFGS update (psi = 1) and DFP update (psi = 0)

  Level: advanced

  Note:
  If B does not have a constant value of `psi` for all iterations this  will
  return `psi` = `PETSC_DETERMINE` = -1, a negative value that `psi` cannot
  attain for a valid general Broyden update.
  This is the case if `B` is a `MATLMVMSYMBROYDEN`, where `psi`'s dual value
  `phi` is constant and `psi` changes from iteration to iteration.

.seealso: [](ch_ksp),
          `MATLMVMSYMBROYDEN`, `MATLMVMSYMBADBROYDEN`,
          `MATLMVMDFP`, `MATLMVMBFGS`,
          `MatLMVMSymBadBroydenSetPsi()`,
          `MatLMVMSymBroydenGetPhi()`, `MatLMVMSymBroydenSetPhi()`
@*/
PetscErrorCode MatLMVMSymBadBroydenGetPsi(Mat B, PetscReal *psi)
{
  PetscFunctionBegin;
  *psi = PETSC_DETERMINE;
  PetscTryMethod(B, "MatLMVMSymBadBroydenGetPsi_C", (Mat, PetscReal *), (B, psi));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatLMVMSymBadBroydenGetPsi_SymBrdn(Mat B, PetscReal *psi)
{
  Mat_LMVM    *lmvm = (Mat_LMVM *)B->data;
  Mat_SymBrdn *lsb  = (Mat_SymBrdn *)lmvm->ctx;

  PetscFunctionBegin;
  *psi = lsb->psi_scalar;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatLMVMSymBadBroydenSetPsi - Get the psi parameter for a Broyden class quasi-Newton update matrix

  Input Parameters:
+ B   - The matrix
- psi - a number defining an update that is a convex combination of the BFGS update (psi = 1) and DFP update (psi = 0)

  Level: developer

  Note:
  If `B` cannot have a constant value of `psi` for all iterations this will
  be ignored.
  This is the case if `B` is a `MATLMVMSYMBROYDEN`, where `psi`'s dual value
  `phi` is constant and `psi` changes from iteration to iteration.

.seealso: [](ch_ksp),
          `MATLMVMSYMBROYDEN`, `MATLMVMSYMBADBROYDEN`,
          `MATLMVMDFP`, `MATLMVMBFGS`,
          `MatLMVMSymBadBroydenGetPsi()`,
          `MatLMVMSymBroydenGetPhi()`, `MatLMVMSymBroydenSetPhi()`
@*/
PetscErrorCode MatLMVMSymBadBroydenSetPsi(Mat B, PetscReal psi)
{
  PetscFunctionBegin;
  PetscTryMethod(B, "MatLMVMSymBadBroydenSetPsi_C", (Mat, PetscReal), (B, psi));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatLMVMSymBadBroydenSetPsi_SymBrdn(Mat B, PetscReal psi)
{
  Mat_LMVM    *lmvm = (Mat_LMVM *)B->data;
  Mat_SymBrdn *lsb  = (Mat_SymBrdn *)lmvm->ctx;

  PetscFunctionBegin;
  lsb->psi_scalar = psi;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatLMVMSymBroydenSetDelta_SymBrdn(Mat B, PetscScalar delta)
{
  Mat_LMVM    *lmvm = (Mat_LMVM *)B->data;
  Mat_SymBrdn *lsb  = (Mat_SymBrdn *)lmvm->ctx;

  PetscFunctionBegin;
  PetscCall(SymBroydenRescaleSetDelta(B, lsb->rescale, PetscAbsReal(PetscRealPart(delta))));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatCreate_LMVMSymBrdn(Mat B)
{
  Mat_LMVM    *lmvm;
  Mat_SymBrdn *lsb;

  PetscFunctionBegin;
  PetscCall(MatCreate_LMVM(B));
  PetscCall(PetscObjectChangeTypeName((PetscObject)B, MATLMVMSYMBROYDEN));
  PetscCall(MatSetOption(B, MAT_HERMITIAN, PETSC_TRUE));
  PetscCall(MatSetOption(B, MAT_SPD, PETSC_TRUE)); // TODO: change to HPD when available
  PetscCall(MatSetOption(B, MAT_SPD_ETERNAL, PETSC_TRUE));
  B->ops->view           = MatView_LMVMSymBrdn;
  B->ops->setfromoptions = MatSetFromOptions_LMVMSymBrdn;
  B->ops->setup          = MatSetUp_LMVMSymBrdn;
  B->ops->destroy        = MatDestroy_LMVMSymBrdn;

  lmvm                          = (Mat_LMVM *)B->data;
  lmvm->ops->reset              = MatReset_LMVMSymBrdn;
  lmvm->ops->update             = MatUpdate_LMVMSymBrdn;
  lmvm->ops->copy               = MatCopy_LMVMSymBrdn;
  lmvm->ops->setmultalgorithm   = MatLMVMSetMultAlgorithm_SymBrdn;
  lmvm->cache_gradient_products = PETSC_TRUE;
  PetscCall(MatLMVMSetMultAlgorithm_SymBrdn(B));

  PetscCall(PetscNew(&lsb));
  lmvm->ctx            = (void *)lsb;
  lsb->phi_scalar      = 0.125;
  lsb->psi_scalar      = PETSC_DETERMINE;
  lsb->watchdog        = 0;
  lsb->max_seq_rejects = lmvm->m / 2;

  PetscCall(SymBroydenRescaleCreate(&lsb->rescale));
  lsb->rescale->theta = lsb->phi_scalar;
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatLMVMSymBroydenGetPhi_C", MatLMVMSymBroydenGetPhi_SymBrdn));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatLMVMSymBroydenSetPhi_C", MatLMVMSymBroydenSetPhi_SymBrdn));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatLMVMSymBadBroydenGetPsi_C", MatLMVMSymBadBroydenGetPsi_SymBrdn));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatLMVMSymBadBroydenSetPsi_C", MatLMVMSymBadBroydenSetPsi_SymBrdn));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatLMVMSymBroydenSetDelta_C", MatLMVMSymBroydenSetDelta_SymBrdn));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatLMVMSymBroydenSetDelta - Sets the starting value for the diagonal scaling vector computed
  in the SymBrdn approximations (also works for BFGS and DFP).

  Input Parameters:
+ B     - `MATLMVM` matrix
- delta - initial value for diagonal scaling

  Level: intermediate

.seealso: [](ch_ksp), `MATLMVMSYMBROYDEN`
@*/
PetscErrorCode MatLMVMSymBroydenSetDelta(Mat B, PetscScalar delta)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, MAT_CLASSID, 1);
  PetscTryMethod(B, "MatLMVMSymBroydenSetDelta_C", (Mat, PetscScalar), (B, delta));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatLMVMSymBroydenSetScaleType - Sets the scale type for symmetric Broyden-type updates.

  Input Parameters:
+ B     - the `MATLMVM` matrix
- stype - scale type, see `MatLMVMSymBroydenScaleType`

  Options Database Key:
. -mat_lmvm_scale_type <none,scalar,diagonal> - set the scaling type

  Level: intermediate

  MatLMVMSymBrdnScaleTypes\:
+   `MAT_LMVM_SYMBROYDEN_SCALE_NONE`     - use whatever initial Hessian is already there (will be the identity if the user does nothing)
.   `MAT_LMVM_SYMBROYDEN_SCALE_SCALAR`   - use the Shanno scalar as the initial Hessian
.   `MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL` - use a diagonalized BFGS update as the initial Hessian
.   `MAT_LMVM_SYMBROYDEN_SCALE_USER`     - same as `MAT_LMVM_SYMBROYDEN_NONE`
-   `MAT_LMVM_SYMBROYDEN_SCALE_DECIDE`   - let PETSc decide

.seealso: [](ch_ksp), `MATLMVMSYMBROYDEN`, `MatCreateLMVMSymBroyden()`, `MatLMVMSymBroydenScaleType`
@*/
PetscErrorCode MatLMVMSymBroydenSetScaleType(Mat B, MatLMVMSymBroydenScaleType stype)
{
  Mat_LMVM    *lmvm = (Mat_LMVM *)B->data;
  Mat_SymBrdn *lsb  = (Mat_SymBrdn *)lmvm->ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, MAT_CLASSID, 1);
  PetscCall(SymBroydenRescaleSetType(lsb->rescale, stype));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatCreateLMVMSymBroyden - Creates a limited-memory Symmetric Broyden-type matrix used
  for approximating Jacobians.

  Collective

  Input Parameters:
+ comm - MPI communicator, set to `PETSC_COMM_SELF`
. n    - number of local rows for storage vectors
- N    - global size of the storage vectors

  Output Parameter:
. B - the matrix

  Options Database Keys:
+ -mat_lmvm_hist_size         - the number of history vectors to keep
. -mat_lmvm_phi               - convex ratio between BFGS and DFP components of the update
. -mat_lmvm_scale_type        - type of scaling applied to J0 (none, scalar, diagonal)
. -mat_lmvm_mult_algorithm    - the algorithm to use for multiplication (recursive, dense, compact_dense)
. -mat_lmvm_cache_J0_products - whether products between the base Jacobian J0 and history vectors should be cached or recomputed
. -mat_lmvm_eps               - (developer) numerical zero tolerance for testing when an update should be skipped
. -mat_lmvm_debug             - (developer) perform internal debugging checks
. -mat_lmvm_theta             - (developer) convex ratio between BFGS and DFP components of the diagonal J0 scaling
. -mat_lmvm_rho               - (developer) update limiter for the J0 scaling
. -mat_lmvm_alpha             - (developer) coefficient factor for the quadratic subproblem in J0 scaling
. -mat_lmvm_beta              - (developer) exponential factor for the diagonal J0 scaling
- -mat_lmvm_sigma_hist        - (developer) number of past updates to use in J0 scaling

  Level: intermediate

  Notes:
  It is recommended that one use the `MatCreate()`, `MatSetType()` and/or `MatSetFromOptions()` paradigm instead of this
  routine directly.

  L-SymBrdn is a convex combination of L-DFP and L-BFGS such that $B = (1 - \phi)B_{\text{BFGS}} + \phi B_{\text{DFP}}$.
  The combination factor $\phi$ is restricted to the range $[0, 1]$, where the L-SymBrdn matrix is guaranteed to be
  symmetric positive-definite.

  To use the L-SymBrdn matrix with other vector types, the matrix must be created using `MatCreate()` and `MatSetType()`,
  followed by `MatLMVMAllocate()`.  This ensures that the internal storage and work vectors are duplicated from the
  correct type of vector.

.seealso: [](ch_ksp), `MatCreate()`, `MATLMVM`, `MATLMVMSYMBROYDEN`, `MatCreateLMVMDFP()`, `MatCreateLMVMSR1()`,
          `MatCreateLMVMBFGS()`, `MatCreateLMVMBroyden()`, `MatCreateLMVMBadBroyden()`
@*/
PetscErrorCode MatCreateLMVMSymBroyden(MPI_Comm comm, PetscInt n, PetscInt N, Mat *B)
{
  PetscFunctionBegin;
  PetscCall(KSPInitializePackage());
  PetscCall(MatCreate(comm, B));
  PetscCall(MatSetSizes(*B, n, n, N, N));
  PetscCall(MatSetType(*B, MATLMVMSYMBROYDEN));
  PetscCall(MatSetUp(*B));
  PetscFunctionReturn(PETSC_SUCCESS);
}
