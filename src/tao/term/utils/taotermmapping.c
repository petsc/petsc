#include <petsc/private/taoimpl.h>
#include <petsc/private/matimpl.h>

PETSC_INTERN PetscErrorCode TaoTermMappingSetData(TaoTermMapping *mt, const char *prefix, PetscReal scale, TaoTerm term, Mat map)
{
  PetscBool same_name;

  PetscFunctionBegin;
  PetscCall(PetscStrcmp(prefix, mt->prefix, &same_name));
  if (!same_name) {
    PetscCall(PetscFree(mt->prefix));
    PetscCall(PetscStrallocpy(prefix, &mt->prefix));
  }
  if (term != mt->term) {
    PetscCall(VecDestroy(&mt->_unmapped_gradient));
    PetscCall(MatDestroy(&mt->_unmapped_H));
    PetscCall(MatDestroy(&mt->_unmapped_Hpre));
    PetscCall(MatDestroy(&mt->_mapped_H));
    PetscCall(MatDestroy(&mt->_mapped_Hpre));
    PetscCall(MatDestroy(&mt->_mapped_H_work));
    PetscCall(MatDestroy(&mt->_mapped_Hpre_work));
  }
  PetscCall(PetscObjectReference((PetscObject)term));
  PetscCall(TaoTermDestroy(&mt->term));
  mt->term  = term;
  mt->scale = scale;
  if (map != mt->map) PetscCall(VecDestroy(&mt->_map_output));
  PetscCall(PetscObjectReference((PetscObject)map));
  PetscCall(MatDestroy(&mt->map));
  mt->map = map;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TaoTermMappingReset(TaoTermMapping *mt)
{
  PetscFunctionBegin;
  PetscCall(TaoTermMappingSetData(mt, NULL, 0.0, NULL, NULL));
  PetscCall(VecDestroy(&mt->_mapped_gradient));
  PetscCall(MatDestroy(&mt->_unmapped_H));
  PetscCall(MatDestroy(&mt->_unmapped_Hpre));
  PetscCall(MatDestroy(&mt->_mapped_H));
  PetscCall(MatDestroy(&mt->_mapped_Hpre));
  PetscCall(MatDestroy(&mt->_mapped_H_work));
  PetscCall(MatDestroy(&mt->_mapped_Hpre_work));
  mt->mask = TAOTERM_MASK_NONE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TaoTermMappingGetData(TaoTermMapping *mt, const char **prefix, PetscReal *scale, TaoTerm *term, Mat *map)
{
  PetscFunctionBegin;
  if (prefix) *prefix = mt->prefix;
  if (term) *term = mt->term;
  if (scale) *scale = mt->scale;
  if (map) *map = mt->map;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#define TaoTermMappingCheckInsertMode(mt, mode) \
  do { \
    PetscCheck((mt)->term, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "TaoTermMapping has no TaoTerm set"); \
    PetscCheck((mode) == INSERT_VALUES || (mode) == ADD_VALUES, PetscObjectComm((PetscObject)(mt)->term), PETSC_ERR_ARG_OUTOFRANGE, "insert mode must be INSERT_VALUES or ADD_VALUES"); \
  } while (0)

static PetscErrorCode TaoTermMappingMap(TaoTermMapping *mt, Vec x, Vec *Ax)
{
  PetscFunctionBegin;
  *Ax = x;
  if (mt->map) {
    if (!mt->_map_output) PetscCall(MatCreateVecs(mt->map, NULL, &mt->_map_output));
    PetscCall(MatMult(mt->map, x, mt->_map_output));
    *Ax = mt->_map_output;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TaoTermMappingComputeObjective(TaoTermMapping *mt, Vec x, Vec params, InsertMode mode, PetscReal *value)
{
  Vec       Ax;
  PetscReal v;

  PetscFunctionBegin;
  TaoTermMappingCheckInsertMode(mt, mode);
  if (TaoTermObjectiveMasked(mt->mask)) {
    if (mode == INSERT_VALUES) *value = 0.0;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(TaoTermMappingMap(mt, x, &Ax));
  PetscCall(TaoTermComputeObjective(mt->term, Ax, params, &v));
  if (mode == ADD_VALUES) *value += mt->scale * v;
  else *value = mt->scale * v;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermMappingGetGradients(TaoTermMapping *mt, InsertMode mode, Vec g, Vec *mapped_g, Vec *unmapped_g)
{
  PetscFunctionBegin;
  *mapped_g = g;
  if (mode == ADD_VALUES) {
    if (!mt->_mapped_gradient) PetscCall(VecDuplicate(g, &mt->_mapped_gradient));
    *mapped_g = mt->_mapped_gradient;
  }
  *unmapped_g = *mapped_g;
  if (mt->map) {
    if (!mt->_unmapped_gradient) PetscCall(TaoTermCreateSolutionVec(mt->term, &mt->_unmapped_gradient));
    *unmapped_g = mt->_unmapped_gradient;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermMappingSetGradients(TaoTermMapping *mt, InsertMode mode, Vec g, Vec mapped_g, Vec unmapped_g)
{
  PetscFunctionBegin;
  if (mt->map) PetscCall(MatMultHermitianTranspose(mt->map, unmapped_g, mapped_g));
  else PetscAssert(mapped_g == unmapped_g, PETSC_COMM_SELF, PETSC_ERR_PLIB, "gradient not written to the right place");
  if (mode == ADD_VALUES) PetscCall(VecAXPY(g, mt->scale, mapped_g));
  else {
    PetscAssert(mapped_g == g, PETSC_COMM_SELF, PETSC_ERR_PLIB, "gradient not written to the right place");
    if (mt->scale != 1.0) PetscCall(VecScale(g, mt->scale));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TaoTermMappingComputeGradient(TaoTermMapping *mt, Vec x, Vec params, InsertMode mode, Vec g)
{
  Vec Ax, mapped_g, unmapped_g = NULL;

  PetscFunctionBegin;
  TaoTermMappingCheckInsertMode(mt, mode);
  if (TaoTermGradientMasked(mt->mask)) {
    if (mode == INSERT_VALUES) PetscCall(VecZeroEntries(g));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(TaoTermMappingGetGradients(mt, mode, g, &mapped_g, &unmapped_g));
  PetscCall(TaoTermMappingMap(mt, x, &Ax));
  PetscCall(TaoTermComputeGradient(mt->term, Ax, params, unmapped_g));
  PetscCall(TaoTermMappingSetGradients(mt, mode, g, mapped_g, unmapped_g));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TaoTermMappingComputeObjectiveAndGradient(TaoTermMapping *mt, Vec x, Vec params, InsertMode mode, PetscReal *value, Vec g)
{
  Vec       Ax, mapped_g, unmapped_g = NULL;
  PetscReal v;

  PetscFunctionBegin;
  TaoTermMappingCheckInsertMode(mt, mode);
  if (TaoTermObjectiveMasked(mt->mask) && TaoTermGradientMasked(mt->mask)) {
    if (mode == INSERT_VALUES) {
      *value = 0.0;
      PetscCall(VecZeroEntries(g));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  if (TaoTermObjectiveMasked(mt->mask)) {
    if (mode == INSERT_VALUES) *value = 0.0;
    PetscCall(TaoTermMappingComputeGradient(mt, x, params, mode, g));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  if (TaoTermGradientMasked(mt->mask)) {
    if (mode == INSERT_VALUES) PetscCall(VecZeroEntries(g));
    PetscCall(TaoTermMappingComputeObjective(mt, x, params, mode, value));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(TaoTermMappingGetGradients(mt, mode, g, &mapped_g, &unmapped_g));
  PetscCall(TaoTermMappingMap(mt, x, &Ax));
  PetscCall(TaoTermComputeObjectiveAndGradient(mt->term, Ax, params, &v, unmapped_g));
  PetscCall(TaoTermMappingSetGradients(mt, mode, g, mapped_g, unmapped_g));
  if (mode == ADD_VALUES) *value += mt->scale * v;
  else *value = mt->scale * v;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermMappingMatPtAP(Mat unmapped_H, Mat map, Mat mapped_H, Mat work)
{
  PetscBool is_uH_diag, is_map_diag, is_uH_cdiag, is_map_cdiag;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)unmapped_H, MATDIAGONAL, &is_uH_diag));
  PetscCall(PetscObjectTypeCompare((PetscObject)unmapped_H, MATCONSTANTDIAGONAL, &is_uH_cdiag));
  PetscCall(PetscObjectTypeCompare((PetscObject)map, MATDIAGONAL, &is_map_diag));
  PetscCall(PetscObjectTypeCompare((PetscObject)map, MATCONSTANTDIAGONAL, &is_map_cdiag));

  if (is_map_diag) {
    Vec m_diag;

    PetscCall(MatDiagonalGetDiagonal(map, &m_diag));
    if (is_uH_cdiag) {
      Vec         mapped_diag;
      PetscScalar cc;

      // mapped_H \gets cc map * map
      PetscCall(MatConstantDiagonalGetConstant(unmapped_H, &cc));
      PetscCall(MatDiagonalGetDiagonal(mapped_H, &mapped_diag));
      PetscCall(VecPointwiseMult(mapped_diag, m_diag, m_diag));
      PetscCall(VecScale(mapped_diag, cc));
      PetscCall(MatDiagonalRestoreDiagonal(mapped_H, &mapped_diag));
    } else if (is_uH_diag) {
      Vec mapped_diag, unmapped_diag;

      PetscCall(MatDiagonalGetDiagonal(mapped_H, &mapped_diag));
      PetscCall(MatDiagonalGetDiagonal(unmapped_H, &unmapped_diag));
      PetscCall(VecPointwiseMult(mapped_diag, m_diag, m_diag));
      PetscCall(VecPointwiseMult(mapped_diag, unmapped_diag, mapped_diag));
      PetscCall(MatDiagonalRestoreDiagonal(mapped_H, &mapped_diag));
      PetscCall(MatDiagonalRestoreDiagonal(unmapped_H, &unmapped_diag));
    } else {
      PetscCall(MatCopy(unmapped_H, mapped_H, SAME_NONZERO_PATTERN));
      PetscCall(MatDiagonalScale(mapped_H, m_diag, m_diag));
    }
    PetscCall(MatDiagonalRestoreDiagonal(map, &m_diag));
  } else if (is_map_cdiag) {
    PetscScalar cc;

    PetscCall(MatConstantDiagonalGetConstant(map, &cc));
    PetscCall(MatCopy(unmapped_H, mapped_H, SAME_NONZERO_PATTERN));
    PetscCall(MatScale(mapped_H, cc * cc));
  } else if (is_uH_diag) {
    Vec unmapped_diag;

    // TODO inefficient. Remove when diag PtAP gets implemented
    PetscCall(MatDiagonalGetDiagonal(unmapped_H, &unmapped_diag));
    PetscCall(MatCopy(map, work, SAME_NONZERO_PATTERN));
    PetscCall(MatDiagonalScale(work, unmapped_diag, NULL));
    PetscCall(MatTransposeMatMult(map, work, MAT_REUSE_MATRIX, PETSC_DETERMINE, &mapped_H));
    PetscCall(MatDiagonalRestoreDiagonal(unmapped_H, &unmapped_diag));
  } else if (is_uH_cdiag) {
    // cc * A^T A
    PetscScalar cc;

    PetscCall(MatConstantDiagonalGetConstant(unmapped_H, &cc));
    PetscCall(MatTransposeMatMult(map, map, MAT_REUSE_MATRIX, PETSC_DETERMINE, &mapped_H));
    PetscCall(MatScale(mapped_H, cc));
  } else PetscCall(MatPtAP(unmapped_H, map, MAT_REUSE_MATRIX, PETSC_DETERMINE, &mapped_H));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermMappingGetHessians(TaoTermMapping *mt, InsertMode mode, Mat H, Mat Hpre, Mat *mapped_H, Mat *mapped_Hpre, Mat *unmapped_H, Mat *unmapped_Hpre)
{
  PetscFunctionBegin;
  *mapped_H    = H;
  *mapped_Hpre = Hpre;
  if (mode == ADD_VALUES || mt->map) {
    // we will need _unmapped_H / _unmapped_Hpre
    if (!mt->_unmapped_H) {
      PetscBool is_defined = PETSC_FALSE;

      PetscCall(TaoTermIsCreateHessianMatricesDefined(mt->term, &is_defined));
      if (is_defined) {
        PetscCall(MatDestroy(&mt->_unmapped_Hpre));
        PetscCall(TaoTermCreateHessianMatrices(mt->term, &mt->_unmapped_H, &mt->_unmapped_Hpre));
      }
      if (!mt->map) {
        PetscCall(PetscObjectReference((PetscObject)mt->_unmapped_H));
        PetscCall(MatDestroy(&mt->_mapped_H));
        mt->_mapped_H = mt->_unmapped_H;

        PetscCall(PetscObjectReference((PetscObject)mt->_unmapped_Hpre));
        PetscCall(MatDestroy(&mt->_mapped_Hpre));
        mt->_mapped_Hpre = mt->_unmapped_Hpre;
      }
    }
  }
  if (mode == ADD_VALUES) {
    if (H) {
      if (!mt->_mapped_H) PetscCall(MatDuplicate(H, MAT_DO_NOT_COPY_VALUES, &mt->_mapped_H));
      *mapped_H = mt->_mapped_H;
    }
    if (Hpre) {
      if (!mt->_mapped_Hpre) PetscCall(MatDuplicate(Hpre, MAT_DO_NOT_COPY_VALUES, &mt->_mapped_Hpre));
      *mapped_Hpre = mt->_mapped_Hpre;
    }
  }
  *unmapped_H    = *mapped_H;
  *unmapped_Hpre = *mapped_Hpre;
  if (mt->map) {
    if (H) *unmapped_H = mt->_unmapped_H;
    if (Hpre) *unmapped_Hpre = mt->_unmapped_Hpre;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// if (map) mapped_H \gets map^T @ unmapped_H @ map
// else (assumes that unmapped == mapped.
//
// if INSERT
//   H \gets mapped_H
// else if ADD
//   H \gets H + scale * mapped_H
static PetscErrorCode TaoTermMappingSetHessians(TaoTermMapping *mt, InsertMode mode, Mat H, Mat Hpre, Mat mapped_H, Mat mapped_Hpre, Mat unmapped_H, Mat unmapped_Hpre)
{
  PetscFunctionBegin;
  if (mt->map) {
    // currently only implements Gauss-Newton Hessian approximation
    if (mapped_H) PetscCall(TaoTermMappingMatPtAP(unmapped_H, mt->map, mapped_H, mt->_mapped_H_work));
    if (mapped_Hpre && (mapped_Hpre != mapped_H)) PetscCall(TaoTermMappingMatPtAP(unmapped_Hpre, mt->map, mapped_Hpre, mt->_mapped_Hpre_work));
  }
  if (mode == ADD_VALUES) {
    if (H) PetscCall(MatAXPY(H, mt->scale, mapped_H, UNKNOWN_NONZERO_PATTERN));
    if (Hpre) PetscCall(MatAXPY(Hpre, mt->scale, mapped_Hpre, UNKNOWN_NONZERO_PATTERN));
  } else {
    if (H) PetscCall(MatCopy(mapped_H, H, DIFFERENT_NONZERO_PATTERN));
    if (Hpre && (H != Hpre)) PetscCall(MatCopy(mapped_Hpre, Hpre, DIFFERENT_NONZERO_PATTERN));
    if (mt->scale != 1.0) {
      if (H) PetscCall(MatScale(H, mt->scale));
      if (Hpre && Hpre != H) PetscCall(MatScale(Hpre, mt->scale));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Either called by TaoComputeHessian (one term in Tao), or by TAOTERMSUM
//
// First case: (one term in Tao)
// TaoComputeHessian
//   -> TaoTermMappingComputeHessian
//      (unmapped_H == mapped_H)
//
// Second case: TAOTERMSUM, (more than one term in Tao)
// TaoComputeHessian
//   -> TaoTermMappingComputeHessian
//     -> (mt->_unmapped_H == mt->_mapped_H == tao->hessian) (SUM does not take mapping)
//     -> TaoTermComputeHessian
//       -> TaoTermComputeHessian_Sum
//         -> for(i:n_terms)
//         -> TaoTermMappingComputeHessian
//           -> (unmapped_H may not == mapped_H)
PETSC_INTERN PetscErrorCode TaoTermMappingComputeHessian(TaoTermMapping *mt, Vec x, Vec params, InsertMode mode, Mat H, Mat Hpre)
{
  Vec Ax;
  Mat mapped_H, mapped_Hpre, unmapped_H = NULL, unmapped_Hpre = NULL;

  PetscFunctionBegin;
  TaoTermMappingCheckInsertMode(mt, mode);
  if (TaoTermHessianMasked(mt->mask)) {
    if (mode == INSERT_VALUES) {
      if (H) {
        PetscCall(MatZeroEntries(H));
        PetscCall(MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY));
        PetscCall(MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY));
      }
      if (Hpre && Hpre != H) {
        PetscCall(MatZeroEntries(Hpre));
        PetscCall(MatAssemblyBegin(Hpre, MAT_FINAL_ASSEMBLY));
        PetscCall(MatAssemblyEnd(Hpre, MAT_FINAL_ASSEMBLY));
      }
    }
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(TaoTermMappingMap(mt, x, &Ax));
  PetscCall(TaoTermMappingGetHessians(mt, mode, H, Hpre, &mapped_H, &mapped_Hpre, &unmapped_H, &unmapped_Hpre));
  PetscCall(TaoTermComputeHessian(mt->term, Ax, params, unmapped_H, unmapped_Hpre));
  PetscCall(TaoTermMappingSetHessians(mt, mode, H, Hpre, mapped_H, mapped_Hpre, unmapped_H, unmapped_Hpre));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TaoTermMappingSetUp(TaoTermMapping *mt)
{
  PetscFunctionBegin;
  PetscCall(TaoTermSetUp(mt->term));
  if (mt->map) PetscCall(MatSetUp(mt->map));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TaoTermMappingCreateSolutionVec(TaoTermMapping *mt, Vec *solution)
{
  PetscFunctionBegin;
  if (mt->map) PetscCall(MatCreateVecs(mt->map, solution, NULL));
  else PetscCall(TaoTermCreateSolutionVec(mt->term, solution));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TaoTermMappingCreateParametersVec(TaoTermMapping *mt, Vec *params)
{
  PetscFunctionBegin;
  PetscCall(TaoTermCreateParametersVec(mt->term, params));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermMappingCreateAPWorkMatrix(Mat map, Mat unmapped, Mat *mapped_work)
{
  PetscBool is_uH_diag, is_map_diag;

  PetscFunctionBegin;
  PetscCall(PetscObjectBaseTypeCompareAny((PetscObject)map, &is_map_diag, MATDIAGONAL, MATCONSTANTDIAGONAL, ""));
  PetscCall(PetscObjectTypeCompare((PetscObject)unmapped, MATDIAGONAL, &is_uH_diag));
  if (is_uH_diag && !is_map_diag) PetscCall(MatDuplicate(map, MAT_DO_NOT_COPY_VALUES, mapped_work));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// This function takes in unmapped_H, map, and returns matrix for mapped_H, which is PtAP
static PetscErrorCode TaoTermMappingCreatePtAP(Mat unmapped_H, Mat map, Mat *H)
{
  PetscBool is_uH_diag, is_uH_cdiag, is_map_diag, is_map_cdiag;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)unmapped_H, MATDIAGONAL, &is_uH_diag));
  PetscCall(PetscObjectTypeCompare((PetscObject)unmapped_H, MATCONSTANTDIAGONAL, &is_uH_cdiag));
  PetscCall(PetscObjectTypeCompare((PetscObject)map, MATDIAGONAL, &is_map_diag));
  PetscCall(PetscObjectTypeCompare((PetscObject)map, MATCONSTANTDIAGONAL, &is_map_cdiag));

  // TODO support for PtAP with diagonal would be ideal
  if (is_map_diag && is_uH_diag) {
    PetscCall(MatDuplicate(unmapped_H, MAT_DO_NOT_COPY_VALUES, H));
  } else if (is_map_cdiag && is_uH_cdiag) {
    // MatDiagonal does not support setvalues, thus AIJ
    PetscLayout rlayout;
    PetscInt    m, M;

    PetscCall(MatGetLayouts(map, &rlayout, NULL));
    PetscCall(MatGetSize(map, &M, NULL));
    PetscCall(MatGetLocalSize(map, &m, NULL));
    PetscCall(MatCreate(PetscObjectComm((PetscObject)map), H));
    PetscCall(MatSetSizes(*H, m, m, M, M));
    PetscCall(MatSetLayouts(*H, rlayout, rlayout));
    PetscCall(MatSetType(*H, MATAIJ));
    PetscCall(MatSetUp(*H));
  } else if ((is_map_diag && !is_uH_diag && !is_uH_cdiag)) {
    PetscCall(MatDuplicate(unmapped_H, MAT_DO_NOT_COPY_VALUES, H));
  } else if (is_map_cdiag && is_uH_diag) {
    // MatDiagonal does not support setvalues, thus AIJ
    PetscLayout rlayout;
    PetscInt    m, M;

    PetscCall(MatGetLayouts(map, &rlayout, NULL));
    PetscCall(MatGetSize(map, &M, NULL));
    PetscCall(MatGetLocalSize(map, &m, NULL));
    PetscCall(MatCreate(PetscObjectComm((PetscObject)map), H));
    PetscCall(MatSetSizes(*H, m, m, M, M));
    PetscCall(MatSetLayouts(*H, rlayout, rlayout));
    PetscCall(MatSetType(*H, MATAIJ));
    PetscCall(MatSetUp(*H));
  } else if (is_map_diag && is_uH_cdiag) {
    PetscCall(MatDuplicate(map, MAT_DO_NOT_COPY_VALUES, H));
  } else if ((is_uH_diag && !is_map_diag && !is_map_cdiag) || (is_uH_cdiag && !is_map_diag && !is_map_cdiag)) {
    PetscCall(MatTransposeMatMult(map, map, MAT_INITIAL_MATRIX, PETSC_DETERMINE, H));
  } else if (!is_uH_diag && !is_uH_cdiag && is_map_cdiag) {
    PetscScalar cc;

    PetscCall(MatConstantDiagonalGetConstant(map, &cc));
    PetscCall(MatDuplicate(unmapped_H, MAT_COPY_VALUES, H));
    PetscCall(MatScale(*H, cc * cc));
  } else {
    PetscCall(MatProductCreate(unmapped_H, map, NULL, H));
    PetscCall(MatProductSetType(*H, MATPRODUCT_PtAP));
    PetscCall(MatProductSetFromOptions(*H));
    // TODO Some other default fallback?
    if ((*H)->ops->productsymbolic) PetscCall(MatProductSymbolic(*H));
    else SETERRQ(PetscObjectComm((PetscObject)map), PETSC_ERR_SUP, "Currently does not support PtAP routines for given pair of matrices");
    PetscCall(MatProductNumeric(*H));
    PetscCall(MatZeroEntries(*H));
    PetscCall(MatAssemblyBegin(*H, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(*H, MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
 * Internal function to create Hessian matrices for TaoTermMapping
 *
 * map: m x n
 *
 * This function will internally create unmapped, and mapped  H and Hpre,
 * and return H \gets mt->_mapped_H, and Hpre \gets mt->_mapped_Hpre.
 *
 * if (mt->map)
 *   It also creates internal work matrices to support PtAP with diagonal matrix, which is currently unsupported natively.
 *   mapped:   n x n
 *   unmapped: m x m
 *
 * else
 *   mapped:   n x n
 *   unmapped: n x n
 *
 */
PETSC_INTERN PetscErrorCode TaoTermMappingCreateHessianMatrices(TaoTermMapping *mt, Mat *H, Mat *Hpre)
{
  Mat       uH, uHpre, mH, mHpre;
  PetscBool is_sum;

  PetscFunctionBegin;
  uH    = mt->_unmapped_H;
  uHpre = mt->_unmapped_Hpre;
  mH    = mt->_mapped_H;
  mHpre = mt->_mapped_Hpre;
  PetscCall(PetscObjectTypeCompare((PetscObject)mt->term, TAOTERMSUM, &is_sum));
  if (is_sum && mt->map) PetscCall(PetscInfo(mt->term, "%s: TaoTermType is TAOTERMSUM, but Map is given. Ignoring it.\n", ((PetscObject)mt->term)->prefix));
  PetscCheck(H, PetscObjectComm((PetscObject)mt->term), PETSC_ERR_SUP, "TaoTermMappingCreateHessianMatrices does not take NULL input for H");
  PetscCheck(Hpre, PetscObjectComm((PetscObject)mt->term), PETSC_ERR_SUP, "TaoTermMappingCreateHessianMatrices does not take NULL input Hpre");
  if (!mt->map) {
    // mt->_unmapped_{H,Hpre} == mt->_unmapped_{H,Hpre}
    if (uH && mH) PetscCheck(uH == mH, PetscObjectComm((PetscObject)mt->term), PETSC_ERR_USER, "For unmapped TaoTerm, mapped Hessian and unmapped Hessian must be same");
    if (uHpre && mHpre) PetscCheck(uHpre == mHpre, PetscObjectComm((PetscObject)mt->term), PETSC_ERR_USER, "For unmapped TaoTerm, mapped Hessian preconditioner and unmapped Hessian preconditioner needs to be same");

    // If mapped matrices are present, it should be set to unmapped matrices
    if (mt->_mapped_H && !mt->_unmapped_H) {
      PetscCall(PetscObjectReference((PetscObject)mt->_mapped_H));
      mt->_unmapped_H = mt->_mapped_H;
    }
    if (mt->_mapped_Hpre && !mt->_unmapped_Hpre) {
      PetscCall(PetscObjectReference((PetscObject)mt->_mapped_Hpre));
      mt->_unmapped_Hpre = mt->_mapped_Hpre;
    }
    // create _unmapped only if they are empty
    PetscCall(TaoTermCreateHessianMatrices(mt->term, (mt->_unmapped_H) ? NULL : &mt->_unmapped_H, (mt->_unmapped_Hpre) ? NULL : &mt->_unmapped_Hpre));
    // If mapped matrices are NULL, it should be set to mapped matrices
    if (mt->_unmapped_H && !mt->_mapped_H) {
      PetscCall(PetscObjectReference((PetscObject)mt->_unmapped_H));
      mt->_mapped_H = mt->_unmapped_H;
    }
    if (mt->_unmapped_Hpre && !mt->_mapped_Hpre) {
      PetscCall(PetscObjectReference((PetscObject)mt->_unmapped_Hpre));
      mt->_mapped_Hpre = mt->_unmapped_Hpre;
    }

    // always returns Hpre, even if same as H
    if (*H != mt->_unmapped_H) PetscCall(PetscObjectReference((PetscObject)mt->_unmapped_H));
    if (*Hpre != mt->_unmapped_Hpre) PetscCall(PetscObjectReference((PetscObject)mt->_unmapped_Hpre));
    *H    = mt->_unmapped_H;
    *Hpre = mt->_unmapped_Hpre;
  } else {
    // create _unmapped only if they are empty
    PetscCall(TaoTermCreateHessianMatrices(mt->term, (mt->_unmapped_H) ? NULL : &mt->_unmapped_H, (mt->_unmapped_Hpre) ? NULL : &mt->_unmapped_Hpre));
    // Hack to support  AIJ.... TODO
    PetscCall(MatAssemblyBegin(mt->_unmapped_H, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(mt->_unmapped_H, MAT_FINAL_ASSEMBLY));
    PetscCall(MatShift(mt->_unmapped_H, 1.));
    // Create PtAP only if mt->_mapped_H is empty
    if (mt->_unmapped_H && !mt->_mapped_H) PetscCall(TaoTermMappingCreatePtAP(mt->_unmapped_H, mt->map, &mt->_mapped_H));
    // Creating expensive work matrix to store AP TODO remove when diag PtAP gets implemented
    if (!mt->_mapped_H_work) PetscCall(TaoTermMappingCreateAPWorkMatrix(mt->map, mt->_unmapped_H, &mt->_mapped_H_work));
    if (*H != mt->_mapped_H) PetscCall(PetscObjectReference((PetscObject)mt->_mapped_H));
    *H = mt->_mapped_H;
    if (mt->_unmapped_Hpre == mt->_unmapped_H) {
      // Hpre_is_H true, so mapped_H = mapped_Hpre
      if (!mt->_mapped_Hpre) {
        PetscCall(PetscObjectReference((PetscObject)mt->_mapped_H));
        mt->_mapped_Hpre = mt->_mapped_H;
      }
      if (*Hpre != mt->_mapped_Hpre) PetscCall(PetscObjectReference((PetscObject)*H));
      *Hpre = *H;
    } else {
      if (!mt->_mapped_Hpre) PetscCall(TaoTermMappingCreatePtAP(mt->_unmapped_Hpre, mt->map, &mt->_mapped_Hpre));
      if (!mt->_mapped_Hpre_work) PetscCall(TaoTermMappingCreateAPWorkMatrix(mt->map, mt->_unmapped_Hpre, &mt->_mapped_Hpre_work));
      if (*Hpre != mt->_mapped_Hpre) PetscCall(PetscObjectReference((PetscObject)mt->_mapped_Hpre));
      *Hpre = mt->_mapped_Hpre;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
