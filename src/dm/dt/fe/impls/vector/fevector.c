#include <petsc/private/petscfeimpl.h> /*I "petscfe.h" I*/
#include <petsc/private/petscimpl.h>

typedef struct _n_PetscFE_Vec {
  PetscFE   scalar_fe;
  PetscInt  num_copies;
  PetscBool interleave_basis;
  PetscBool interleave_components;
} PetscFE_Vec;

static PetscErrorCode PetscFEDestroy_Vector(PetscFE fe)
{
  PetscFE_Vec *v;

  PetscFunctionBegin;
  v = (PetscFE_Vec *)fe->data;
  PetscCall(PetscFEDestroy(&v->scalar_fe));
  PetscCall(PetscFree(v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscFEView_Vector_Ascii(PetscFE fe, PetscViewer v)
{
  PetscInt          dim, Nc, scalar_Nc;
  PetscSpace        basis = NULL;
  PetscDualSpace    dual  = NULL;
  PetscQuadrature   quad  = NULL;
  PetscFE_Vec      *vec;
  PetscViewerFormat fmt;

  PetscFunctionBegin;
  vec = (PetscFE_Vec *)fe->data;
  PetscCall(PetscFEGetSpatialDimension(fe, &dim));
  PetscCall(PetscFEGetNumComponents(fe, &Nc));
  PetscCall(PetscFEGetNumComponents(vec->scalar_fe, &scalar_Nc));
  PetscCall(PetscFEGetBasisSpace(fe, &basis));
  PetscCall(PetscFEGetDualSpace(fe, &dual));
  PetscCall(PetscFEGetQuadrature(fe, &quad));
  PetscCall(PetscViewerGetFormat(v, &fmt));
  PetscCall(PetscViewerASCIIPushTab(v));
  if (scalar_Nc == 1) {
    PetscCall(PetscViewerASCIIPrintf(v, "Vector Finite Element in %" PetscInt_FMT " dimensions with %" PetscInt_FMT " components\n", dim, Nc));
  } else {
    PetscCall(PetscViewerASCIIPrintf(v, "Vector Finite Element in %" PetscInt_FMT " dimensions with %" PetscInt_FMT " components (%" PetscInt_FMT " copies of finite element with %" PetscInt_FMT " components)\n", dim, Nc, vec->num_copies, scalar_Nc));
  }
  if (fmt == PETSC_VIEWER_ASCII_INFO_DETAIL) {
    PetscCall(PetscViewerASCIIPrintf(v, "Original finite element:\n"));
    PetscCall(PetscViewerASCIIPushTab(v));
    PetscCall(PetscFEView(vec->scalar_fe, v));
    PetscCall(PetscViewerASCIIPopTab(v));
  }
  if (basis) PetscCall(PetscSpaceView(basis, v));
  if (dual) PetscCall(PetscDualSpaceView(dual, v));
  if (quad) PetscCall(PetscQuadratureView(quad, v));
  PetscCall(PetscViewerASCIIPopTab(v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscFEView_Vector(PetscFE fe, PetscViewer v)
{
  PetscBool isascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)v, PETSCVIEWERASCII, &isascii));
  if (isascii) PetscCall(PetscFEView_Vector_Ascii(fe, v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscFESetUp_Vector(PetscFE fe)
{
  PetscFE_Vec        *v = (PetscFE_Vec *)fe->data;
  PetscDualSpace      dsp;
  PetscInt            n, Ncopies = v->num_copies;
  PetscInt            scalar_n;
  PetscInt           *d, *d_mapped;
  PetscDualSpace_Sum *sum;
  PetscBool           is_sum;

  PetscFunctionBegin;
  PetscCall(PetscFESetUp(v->scalar_fe));
  PetscCall(PetscFEGetDimension(v->scalar_fe, &scalar_n));
  PetscCall(PetscFEGetDualSpace(fe, &dsp));
  PetscCall(PetscObjectTypeCompare((PetscObject)dsp, PETSCDUALSPACESUM, &is_sum));
  PetscCheck(is_sum, PetscObjectComm((PetscObject)fe), PETSC_ERR_ARG_INCOMP, "Expected PETSCDUALSPACESUM dual space");
  sum = (PetscDualSpace_Sum *)dsp->data;
  n   = Ncopies * scalar_n;
  PetscCall(PetscCalloc1(n * n, &fe->invV));
  PetscCall(PetscMalloc2(scalar_n, &d, scalar_n, &d_mapped));
  for (PetscInt i = 0; i < scalar_n; i++) d[i] = i;
  for (PetscInt c = 0; c < Ncopies; c++) {
    PetscCall(ISLocalToGlobalMappingApply(sum->all_rows[c], scalar_n, d, d_mapped));
    for (PetscInt i = 0; i < scalar_n; i++) {
      PetscInt         iw      = d_mapped[i];
      PetscReal       *row_w   = &fe->invV[iw * n];
      const PetscReal *row_r   = &v->scalar_fe->invV[i * scalar_n];
      PetscInt         j0      = v->interleave_basis ? c : c * scalar_n;
      PetscInt         jstride = v->interleave_basis ? Ncopies : 1;

      for (PetscInt j = 0; j < scalar_n; j++) row_w[j0 + j * jstride] = row_r[j];
    }
  }
  PetscCall(PetscFree2(d, d_mapped));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscFEGetDimension_Vector(PetscFE fe, PetscInt *dim)
{
  PetscFE_Vec *v = (PetscFE_Vec *)fe->data;

  PetscFunctionBegin;
  PetscCall(PetscFEGetDimension(v->scalar_fe, dim));
  *dim *= v->num_copies;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscFEVectorInsertTabulation(PetscFE fe, PetscInt npoints, const PetscReal points[], PetscInt k, PetscInt scalar_Nb, PetscInt scalar_point_stride, const PetscReal scalar_Tk[], PetscReal Tk[])
{
  PetscFE_Vec *v = (PetscFE_Vec *)fe->data;
  PetscInt     scalar_Nc;
  PetscInt     Nc;
  PetscInt     cdim;
  PetscInt     dblock;
  PetscInt     block;
  PetscInt     scalar_block;

  PetscFunctionBegin;
  PetscCall(PetscFEGetNumComponents(v->scalar_fe, &scalar_Nc));
  Nc = scalar_Nc * v->num_copies;
  PetscCall(PetscFEGetSpatialDimension(v->scalar_fe, &cdim));
  dblock       = PetscPowInt(cdim, k);
  block        = Nc * dblock;
  scalar_block = scalar_Nc * dblock;
  for (PetscInt p = 0; p < npoints; p++) {
    const PetscReal *scalar_Tp = &scalar_Tk[p * scalar_point_stride];
    PetscReal       *Tp        = &Tk[p * scalar_point_stride * v->num_copies * v->num_copies];

    for (PetscInt j = 0; j < v->num_copies; j++) {
      for (PetscInt scalar_i = 0; scalar_i < scalar_Nb; scalar_i++) {
        PetscInt         i         = v->interleave_basis ? (scalar_i * v->num_copies + j) : (j * scalar_Nb + scalar_i);
        const PetscReal *scalar_Ti = &scalar_Tp[scalar_i * scalar_block];
        PetscReal       *Ti        = &Tp[i * block];

        for (PetscInt scalar_c = 0; scalar_c < scalar_Nc; scalar_c++) {
          PetscInt         c         = v->interleave_components ? (scalar_c * v->num_copies + j) : (j * scalar_Nc + scalar_c);
          const PetscReal *scalar_Tc = &scalar_Ti[scalar_c * dblock];
          PetscReal       *Tc        = &Ti[c * dblock];

          PetscCall(PetscArraycpy(Tc, scalar_Tc, dblock));
        }
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscFEComputeTabulation_Vector(PetscFE fe, PetscInt npoints, const PetscReal points[], PetscInt K, PetscTabulation T)
{
  PetscFE_Vec    *v = (PetscFE_Vec *)fe->data;
  PetscInt        scalar_Nc;
  PetscInt        scalar_Nb;
  PetscInt        cdim;
  PetscTabulation scalar_T;

  PetscFunctionBegin;
  PetscAssert(npoints == T->Nr * T->Np, PetscObjectComm((PetscObject)fe), PETSC_ERR_PLIB, "Expected to be able decode PetscFECreateTabulation() from PetscTabulation fields");
  PetscCall(PetscFECreateTabulation(v->scalar_fe, T->Nr, T->Np, points, K, &scalar_T));
  PetscCall(PetscFEGetNumComponents(v->scalar_fe, &scalar_Nc));
  PetscCall(PetscFEGetDimension(v->scalar_fe, &scalar_Nb));
  PetscCall(PetscFEGetSpatialDimension(v->scalar_fe, &cdim));
  for (PetscInt k = 0; k <= T->K; k++) {
    PetscReal       *Tk                  = T->T[k];
    const PetscReal *scalar_Tk           = scalar_T->T[k];
    PetscInt         dblock              = PetscPowInt(cdim, k);
    PetscInt         scalar_block        = scalar_Nc * dblock;
    PetscInt         scalar_point_stride = scalar_Nb * scalar_block;

    if (v->interleave_basis) {
      PetscCall(PetscFEVectorInsertTabulation(fe, npoints, points, k, scalar_Nb, scalar_point_stride, scalar_Tk, Tk));
    } else {
      PetscDualSpace scalar_dsp;
      PetscSection   ref_section;
      PetscInt       pStart, pEnd;

      PetscCall(PetscFEGetDualSpace(v->scalar_fe, &scalar_dsp));
      PetscCall(PetscDualSpaceGetSection(scalar_dsp, &ref_section));
      PetscCall(PetscSectionGetChart(ref_section, &pStart, &pEnd));
      for (PetscInt p = pStart; p < pEnd; p++) {
        PetscInt         dof, off;
        PetscReal       *Tp;
        const PetscReal *scalar_Tp;

        PetscCall(PetscSectionGetDof(ref_section, p, &dof));
        PetscCall(PetscSectionGetOffset(ref_section, p, &off));
        scalar_Tp = &scalar_Tk[off * scalar_block];
        Tp        = &Tk[off * scalar_block * v->num_copies * v->num_copies];
        PetscCall(PetscFEVectorInsertTabulation(fe, npoints, points, k, dof, scalar_point_stride, scalar_Tp, Tp));
      }
    }
  }
  PetscCall(PetscTabulationDestroy(&scalar_T));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscFECreatePointTrace_Vector(PetscFE fe, PetscInt refPoint, PetscFE *trFE)
{
  PetscFE_Vec *v = (PetscFE_Vec *)fe->data;
  PetscFE      scalar_trFE;
  const char  *name;

  PetscFunctionBegin;
  PetscCall(PetscFECreatePointTrace(v->scalar_fe, refPoint, &scalar_trFE));
  PetscCall(PetscFECreateVector(scalar_trFE, v->num_copies, v->interleave_basis, v->interleave_components, trFE));
  PetscCall(PetscFEDestroy(&scalar_trFE));
  PetscCall(PetscObjectGetName((PetscObject)fe, &name));
  if (name) PetscCall(PetscFESetName(*trFE, name));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscFEIntegrate_Basic(PetscDS, PetscInt, PetscInt, PetscFEGeom *, const PetscScalar[], PetscDS, const PetscScalar[], PetscScalar[]);
PETSC_INTERN PetscErrorCode PetscFEIntegrateBd_Basic(PetscDS, PetscInt, PetscBdPointFn *, PetscInt, PetscFEGeom *, const PetscScalar[], PetscDS, const PetscScalar[], PetscScalar[]);
PETSC_INTERN PetscErrorCode PetscFEIntegrateHybridResidual_Basic(PetscDS, PetscDS, PetscFormKey, PetscInt, PetscInt, PetscFEGeom *, PetscFEGeom *, const PetscScalar[], const PetscScalar[], PetscDS, const PetscScalar[], PetscReal, PetscScalar[]);
PETSC_INTERN PetscErrorCode PetscFEIntegrateBdJacobian_Basic(PetscDS, PetscWeakForm, PetscFEJacobianType, PetscFormKey, PetscInt, PetscFEGeom *, const PetscScalar[], const PetscScalar[], PetscDS, const PetscScalar[], PetscReal, PetscReal, PetscScalar[]);
PETSC_INTERN PetscErrorCode PetscFEIntegrateHybridJacobian_Basic(PetscDS, PetscDS, PetscFEJacobianType, PetscFormKey, PetscInt, PetscInt, PetscFEGeom *, PetscFEGeom *, const PetscScalar[], const PetscScalar[], PetscDS, const PetscScalar[], PetscReal, PetscReal, PetscScalar[]);

static PetscErrorCode PetscFEInitialize_Vector(PetscFE fe)
{
  PetscFunctionBegin;
  fe->ops->setfromoptions          = NULL;
  fe->ops->setup                   = PetscFESetUp_Vector;
  fe->ops->view                    = PetscFEView_Vector;
  fe->ops->destroy                 = PetscFEDestroy_Vector;
  fe->ops->getdimension            = PetscFEGetDimension_Vector;
  fe->ops->createpointtrace        = PetscFECreatePointTrace_Vector;
  fe->ops->computetabulation       = PetscFEComputeTabulation_Vector;
  fe->ops->integrate               = PetscFEIntegrate_Basic;
  fe->ops->integratebd             = PetscFEIntegrateBd_Basic;
  fe->ops->integrateresidual       = PetscFEIntegrateResidual_Basic;
  fe->ops->integratebdresidual     = PetscFEIntegrateBdResidual_Basic;
  fe->ops->integratehybridresidual = PetscFEIntegrateHybridResidual_Basic;
  fe->ops->integratejacobianaction = NULL;
  fe->ops->integratejacobian       = PetscFEIntegrateJacobian_Basic;
  fe->ops->integratebdjacobian     = PetscFEIntegrateBdJacobian_Basic;
  fe->ops->integratehybridjacobian = PetscFEIntegrateHybridJacobian_Basic;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  PETSCFEVECTOR = "vector" - A vector-valued `PetscFE` object that is repeated copies
  of the same underlying finite element.

  Level: intermediate

.seealso: `PetscFE`, `PetscFEType`, `PetscFECreate()`, `PetscFESetType()`, `PETSCFEBASIC`, `PetscFECreateVector()`
M*/
PETSC_EXTERN PetscErrorCode PetscFECreate_Vector(PetscFE fe)
{
  PetscFE_Vec *v;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fe, PETSCFE_CLASSID, 1);
  PetscCall(PetscNew(&v));
  fe->data = v;

  PetscCall(PetscFEInitialize_Vector(fe));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscFECreateVector - Create a vector-valued `PetscFE` from multiple copies of an underlying
  `PetscFE`.

  Collective

  Input Parameters:
+ scalar_fe             - a `PetscFE` finite element
. num_copies            - a positive integer
. interleave_basis      - if `PETSC_TRUE`, the first `num_copies` basis vectors
                          of the output finite element will be copies of the
                          first basis vector of `scalar_fe`, and so on for the
                          other basis vectors; otherwise all of the first-copy
                          basis vectors will come first, followed by all of the
                          second-copy, and so on.
- interleave_components - if `PETSC_TRUE`, the first `num_copies` components
                          of the output finite element will be copies of the
                          first component of `scalar_fe`, and so on for the
                          other components; otherwise all of the first-copy
                          components will come first, followed by all of the
                          second-copy, and so on.

  Output Parameter:
. vector_fe - a `PetscFE` of type `PETSCFEVECTOR` that represent a discretization space with `num_copies` copies of `scalar_fe`

  Level: intermediate

.seealso: `PetscFE`, `PetscFEType`, `PetscFECreate()`, `PetscFESetType()`, `PETSCFEBASIC`, `PETSCFEVECTOR`
@*/
PetscErrorCode PetscFECreateVector(PetscFE scalar_fe, PetscInt num_copies, PetscBool interleave_basis, PetscBool interleave_components, PetscFE *vector_fe)
{
  MPI_Comm        comm;
  PetscFE         fe_vec;
  PetscFE_Vec    *v;
  PetscInt        scalar_Nc;
  PetscQuadrature quad;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(scalar_fe, PETSCFE_CLASSID, 1);
  PetscCall(PetscObjectGetComm((PetscObject)scalar_fe, &comm));
  PetscCheck(num_copies >= 0, comm, PETSC_ERR_ARG_OUTOFRANGE, "Expected positive number of copies, got %" PetscInt_FMT, num_copies);
  PetscCall(PetscFECreate(comm, vector_fe));
  fe_vec = *vector_fe;
  PetscCall(PetscFESetType(fe_vec, PETSCFEVECTOR));
  v = (PetscFE_Vec *)fe_vec->data;
  PetscCall(PetscObjectReference((PetscObject)scalar_fe));
  v->scalar_fe             = scalar_fe;
  v->num_copies            = num_copies;
  v->interleave_basis      = interleave_basis;
  v->interleave_components = interleave_components;
  PetscCall(PetscFEGetNumComponents(scalar_fe, &scalar_Nc));
  PetscCall(PetscFESetNumComponents(fe_vec, scalar_Nc * num_copies));
  PetscCall(PetscFEGetQuadrature(scalar_fe, &quad));
  PetscCall(PetscFESetQuadrature(fe_vec, quad));
  PetscCall(PetscFEGetFaceQuadrature(scalar_fe, &quad));
  PetscCall(PetscFESetFaceQuadrature(fe_vec, quad));
  {
    PetscSpace  scalar_sp;
    PetscSpace *copies;
    PetscSpace  sp;

    PetscCall(PetscFEGetBasisSpace(scalar_fe, &scalar_sp));
    PetscCall(PetscMalloc1(num_copies, &copies));
    for (PetscInt i = 0; i < num_copies; i++) {
      PetscCall(PetscObjectReference((PetscObject)scalar_sp));
      copies[i] = scalar_sp;
    }
    PetscCall(PetscSpaceCreateSum(num_copies, copies, PETSC_TRUE, &sp));
    PetscCall(PetscSpaceSumSetInterleave(sp, interleave_basis, interleave_components));
    PetscCall(PetscSpaceSetUp(sp));
    PetscCall(PetscFESetBasisSpace(fe_vec, sp));
    PetscCall(PetscSpaceDestroy(&sp));
    for (PetscInt i = 0; i < num_copies; i++) PetscCall(PetscSpaceDestroy(&copies[i]));
    PetscCall(PetscFree(copies));
  }
  {
    PetscDualSpace  scalar_sp;
    PetscDualSpace *copies;
    PetscDualSpace  sp;

    PetscCall(PetscFEGetDualSpace(scalar_fe, &scalar_sp));
    PetscCall(PetscMalloc1(num_copies, &copies));
    for (PetscInt i = 0; i < num_copies; i++) {
      PetscCall(PetscObjectReference((PetscObject)scalar_sp));
      copies[i] = scalar_sp;
    }
    PetscCall(PetscDualSpaceCreateSum(num_copies, copies, PETSC_TRUE, &sp));
    PetscCall(PetscDualSpaceSumSetInterleave(sp, interleave_basis, interleave_components));
    PetscCall(PetscDualSpaceSetUp(sp));
    PetscCall(PetscFESetDualSpace(fe_vec, sp));
    PetscCall(PetscDualSpaceDestroy(&sp));
    for (PetscInt i = 0; i < num_copies; i++) PetscCall(PetscDualSpaceDestroy(&copies[i]));
    PetscCall(PetscFree(copies));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
