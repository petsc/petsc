#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/

#include <petsc/private/petscfeimpl.h>

/*@
  DMPlexGetActivePoint - Get the point on which projection is currently working

  Not collective

  Input Parameter:
. dm   - the DM

  Output Parameter:
. point - The mesh point involved in the current projection

  Level: developer

.seealso: `DMPlexSetActivePoint()`
@*/
PetscErrorCode DMPlexGetActivePoint(DM dm, PetscInt *point)
{
  PetscFunctionBeginHot;
  *point = ((DM_Plex *) dm->data)->activePoint;
  PetscFunctionReturn(0);
}

/*@
  DMPlexSetActivePoint - Set the point on which projection is currently working

  Not collective

  Input Parameters:
+ dm   - the DM
- point - The mesh point involved in the current projection

  Level: developer

.seealso: `DMPlexGetActivePoint()`
@*/
PetscErrorCode DMPlexSetActivePoint(DM dm, PetscInt point)
{
  PetscFunctionBeginHot;
  ((DM_Plex *) dm->data)->activePoint = point;
  PetscFunctionReturn(0);
}

/*
  DMProjectPoint_Func_Private - Interpolate the given function in the output basis on the given point

  Input Parameters:
+ dm     - The output DM
. ds     - The output DS
. dmIn   - The input DM
. dsIn   - The input DS
. time   - The time for this evaluation
. fegeom - The FE geometry for this point
. fvgeom - The FV geometry for this point
. isFE   - Flag indicating whether each output field has an FE discretization
. sp     - The output PetscDualSpace for each field
. funcs  - The evaluation function for each field
- ctxs   - The user context for each field

  Output Parameter:
. values - The value for each dual basis vector in the output dual space

  Level: developer

.seealso: `DMProjectPoint_Field_Private()`
*/
static PetscErrorCode DMProjectPoint_Func_Private(DM dm, PetscDS ds, DM dmIn, PetscDS dsIn, PetscReal time, PetscFEGeom *fegeom, PetscFVCellGeom *fvgeom, PetscBool isFE[], PetscDualSpace sp[],
                                                  PetscErrorCode (**funcs)(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void *), void **ctxs,
                                                  PetscScalar values[])
{
  PetscInt       coordDim, Nf, *Nc, f, spDim, d, v, tp;
  PetscBool      isAffine, isCohesive, transform;

  PetscFunctionBeginHot;
  PetscCall(DMGetCoordinateDim(dmIn, &coordDim));
  PetscCall(DMHasBasisTransform(dmIn, &transform));
  PetscCall(PetscDSGetNumFields(ds, &Nf));
  PetscCall(PetscDSGetComponents(ds, &Nc));
  PetscCall(PetscDSIsCohesive(ds, &isCohesive));
  /* Get values for closure */
  isAffine = fegeom->isAffine;
  for (f = 0, v = 0, tp = 0; f < Nf; ++f) {
    void * const ctx = ctxs ? ctxs[f] : NULL;
    PetscBool    cohesive;

    if (!sp[f]) continue;
    PetscCall(PetscDSGetCohesive(ds, f, &cohesive));
    PetscCall(PetscDualSpaceGetDimension(sp[f], &spDim));
    if (funcs[f]) {
      if (isFE[f]) {
        PetscQuadrature   allPoints;
        PetscInt          q, dim, numPoints;
        const PetscReal   *points;
        PetscScalar       *pointEval;
        PetscReal         *x;
        DM                rdm;

        PetscCall(PetscDualSpaceGetDM(sp[f],&rdm));
        PetscCall(PetscDualSpaceGetAllData(sp[f], &allPoints, NULL));
        PetscCall(PetscQuadratureGetData(allPoints,&dim,NULL,&numPoints,&points,NULL));
        PetscCall(DMGetWorkArray(rdm,numPoints*Nc[f],MPIU_SCALAR,&pointEval));
        PetscCall(DMGetWorkArray(rdm,coordDim,MPIU_REAL,&x));
        PetscCall(PetscArrayzero(pointEval, numPoints*Nc[f]));
        for (q = 0; q < numPoints; q++, tp++) {
          const PetscReal *v0;

          if (isAffine) {
            const PetscReal *refpoint = &points[q*dim];
            PetscReal        injpoint[3] = {0., 0., 0.};

            if (dim != fegeom->dim) {
              if (isCohesive) {
                /* We just need to inject into the higher dimensional space assuming the last dimension is collapsed */
                for (d = 0; d < dim; ++d) injpoint[d] = refpoint[d];
                refpoint = injpoint;
              } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Reference spatial dimension %" PetscInt_FMT " != %" PetscInt_FMT " dual basis spatial dimension", fegeom->dim, dim);
            }
            CoordinatesRefToReal(coordDim, fegeom->dim, fegeom->xi, fegeom->v, fegeom->J, refpoint, x);
            v0 = x;
          } else {
            v0 = &fegeom->v[tp*coordDim];
          }
          if (transform) {PetscCall(DMPlexBasisTransformApplyReal_Internal(dmIn, v0, PETSC_TRUE, coordDim, v0, x, dm->transformCtx)); v0 = x;}
          PetscCall((*funcs[f])(coordDim, time, v0, Nc[f], &pointEval[Nc[f]*q], ctx));
        }
        /* Transform point evaluations pointEval[q,c] */
        PetscCall(PetscDualSpacePullback(sp[f], fegeom, numPoints, Nc[f], pointEval));
        PetscCall(PetscDualSpaceApplyAll(sp[f], pointEval, &values[v]));
        PetscCall(DMRestoreWorkArray(rdm,coordDim,MPIU_REAL,&x));
        PetscCall(DMRestoreWorkArray(rdm,numPoints*Nc[f],MPIU_SCALAR,&pointEval));
        v += spDim;
        if (isCohesive && !cohesive) {
          for (d = 0; d < spDim; d++, v++) values[v] = values[v - spDim];
        }
      } else {
        for (d = 0; d < spDim; ++d, ++v) {
          PetscCall(PetscDualSpaceApplyFVM(sp[f], d, time, fvgeom, Nc[f], funcs[f], ctx, &values[v]));
        }
      }
    } else {
      for (d = 0; d < spDim; d++, v++) values[v] = 0.;
      if (isCohesive && !cohesive) {
        for (d = 0; d < spDim; d++, v++) values[v] = 0.;
      }
    }
  }
  PetscFunctionReturn(0);
}

/*
  DMProjectPoint_Field_Private - Interpolate a function of the given field, in the input basis, using the output basis on the given point

  Input Parameters:
+ dm             - The output DM
. ds             - The output DS
. dmIn           - The input DM
. dsIn           - The input DS
. dmAux          - The auxiliary DM, which is always for the input space
. dsAux          - The auxiliary DS, which is always for the input space
. time           - The time for this evaluation
. localU         - The local solution
. localA         - The local auziliary fields
. cgeom          - The FE geometry for this point
. sp             - The output PetscDualSpace for each field
. p              - The point in the output DM
. T              - Input basis and derivatives for each field tabulated on the quadrature points
. TAux           - Auxiliary basis and derivatives for each aux field tabulated on the quadrature points
. funcs          - The evaluation function for each field
- ctxs           - The user context for each field

  Output Parameter:
. values         - The value for each dual basis vector in the output dual space

  Note: Not supported for FV

  Level: developer

.seealso: `DMProjectPoint_Field_Private()`
*/
static PetscErrorCode DMProjectPoint_Field_Private(DM dm, PetscDS ds, DM dmIn, DMEnclosureType encIn, PetscDS dsIn, DM dmAux, DMEnclosureType encAux, PetscDS dsAux, PetscReal time, Vec localU, Vec localA, PetscFEGeom *cgeom, PetscDualSpace sp[], PetscInt p,
                                                   PetscTabulation *T, PetscTabulation *TAux,
                                                   void (**funcs)(PetscInt, PetscInt, PetscInt,
                                                                  const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                                  const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                                  PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]), void **ctxs,
                                                   PetscScalar values[])
{
  PetscSection       section, sectionAux = NULL;
  PetscScalar       *u, *u_t = NULL, *u_x, *a = NULL, *a_t = NULL, *a_x = NULL, *bc;
  PetscScalar       *coefficients   = NULL, *coefficientsAux   = NULL;
  PetscScalar       *coefficients_t = NULL, *coefficientsAux_t = NULL;
  const PetscScalar *constants;
  PetscReal         *x;
  PetscInt          *uOff, *uOff_x, *aOff = NULL, *aOff_x = NULL, *Nc;
  PetscFEGeom        fegeom;
  const PetscInt     dE = cgeom->dimEmbed;
  PetscInt           numConstants, Nf, NfIn, NfAux = 0, f, spDim, d, v, inp, tp = 0;
  PetscBool          isAffine, isCohesive, transform;

  PetscFunctionBeginHot;
  PetscCall(PetscDSGetNumFields(ds, &Nf));
  PetscCall(PetscDSGetComponents(ds, &Nc));
  PetscCall(PetscDSIsCohesive(ds, &isCohesive));
  PetscCall(PetscDSGetNumFields(dsIn, &NfIn));
  PetscCall(PetscDSGetComponentOffsets(dsIn, &uOff));
  PetscCall(PetscDSGetComponentDerivativeOffsets(dsIn, &uOff_x));
  PetscCall(PetscDSGetEvaluationArrays(dsIn, &u, &bc /*&u_t*/, &u_x));
  PetscCall(PetscDSGetWorkspace(dsIn, &x, NULL, NULL, NULL, NULL));
  PetscCall(PetscDSGetConstants(dsIn, &numConstants, &constants));
  PetscCall(DMHasBasisTransform(dmIn, &transform));
  PetscCall(DMGetLocalSection(dmIn, &section));
  PetscCall(DMGetEnclosurePoint(dmIn, dm, encIn, p, &inp));
  PetscCall(DMPlexVecGetClosure(dmIn, section, localU, inp, NULL, &coefficients));
  if (dmAux) {
    PetscInt subp;

    PetscCall(DMGetEnclosurePoint(dmAux, dm, encAux, p, &subp));
    PetscCall(PetscDSGetNumFields(dsAux, &NfAux));
    PetscCall(DMGetLocalSection(dmAux, &sectionAux));
    PetscCall(PetscDSGetComponentOffsets(dsAux, &aOff));
    PetscCall(PetscDSGetComponentDerivativeOffsets(dsAux, &aOff_x));
    PetscCall(PetscDSGetEvaluationArrays(dsAux, &a, NULL /*&a_t*/, &a_x));
    PetscCall(DMPlexVecGetClosure(dmAux, sectionAux, localA, subp, NULL, &coefficientsAux));
  }
  /* Get values for closure */
  isAffine = cgeom->isAffine;
  fegeom.dim      = cgeom->dim;
  fegeom.dimEmbed = cgeom->dimEmbed;
  if (isAffine) {
    fegeom.v    = x;
    fegeom.xi   = cgeom->xi;
    fegeom.J    = cgeom->J;
    fegeom.invJ = cgeom->invJ;
    fegeom.detJ = cgeom->detJ;
  }
  for (f = 0, v = 0; f < Nf; ++f) {
    PetscQuadrature  allPoints;
    PetscInt         q, dim, numPoints;
    const PetscReal *points;
    PetscScalar     *pointEval;
    PetscBool        cohesive;
    DM               dm;

    if (!sp[f]) continue;
    PetscCall(PetscDSGetCohesive(ds, f, &cohesive));
    PetscCall(PetscDualSpaceGetDimension(sp[f], &spDim));
    if (!funcs[f]) {
      for (d = 0; d < spDim; d++, v++) values[v] = 0.;
      if (isCohesive && !cohesive) {
        for (d = 0; d < spDim; d++, v++) values[v] = 0.;
      }
      continue;
    }
    PetscCall(PetscDualSpaceGetDM(sp[f],&dm));
    PetscCall(PetscDualSpaceGetAllData(sp[f], &allPoints, NULL));
    PetscCall(PetscQuadratureGetData(allPoints,&dim,NULL,&numPoints,&points,NULL));
    PetscCall(DMGetWorkArray(dm,numPoints*Nc[f],MPIU_SCALAR,&pointEval));
    for (q = 0; q < numPoints; ++q, ++tp) {
      if (isAffine) {
        CoordinatesRefToReal(dE, cgeom->dim, fegeom.xi, cgeom->v, fegeom.J, &points[q*dim], x);
      } else {
        fegeom.v    = &cgeom->v[tp*dE];
        fegeom.J    = &cgeom->J[tp*dE*dE];
        fegeom.invJ = &cgeom->invJ[tp*dE*dE];
        fegeom.detJ = &cgeom->detJ[tp];
      }
      PetscCall(PetscFEEvaluateFieldJets_Internal(dsIn, NfIn, 0, tp, T, &fegeom, coefficients, coefficients_t, u, u_x, u_t));
      if (dsAux) PetscCall(PetscFEEvaluateFieldJets_Internal(dsAux, NfAux, 0, tp, TAux, &fegeom, coefficientsAux, coefficientsAux_t, a, a_x, a_t));
      if (transform) PetscCall(DMPlexBasisTransformApplyReal_Internal(dmIn, fegeom.v, PETSC_TRUE, dE, fegeom.v, fegeom.v, dm->transformCtx));
      (*funcs[f])(dE, NfIn, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, a_t, a_x, time, fegeom.v, numConstants, constants, &pointEval[Nc[f]*q]);
    }
    PetscCall(PetscDualSpaceApplyAll(sp[f], pointEval, &values[v]));
    PetscCall(DMRestoreWorkArray(dm,numPoints*Nc[f],MPIU_SCALAR,&pointEval));
    v += spDim;
    /* TODO: For now, set both sides equal, but this should use info from other support cell */
    if (isCohesive && !cohesive) {
      for (d = 0; d < spDim; d++, v++) values[v] = values[v - spDim];
    }
  }
  PetscCall(DMPlexVecRestoreClosure(dmIn, section, localU, inp, NULL, &coefficients));
  if (dmAux) PetscCall(DMPlexVecRestoreClosure(dmAux, sectionAux, localA, p, NULL, &coefficientsAux));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMProjectPoint_BdField_Private(DM dm, PetscDS ds, DM dmIn, PetscDS dsIn, DM dmAux, DMEnclosureType encAux, PetscDS dsAux, PetscReal time, Vec localU, Vec localA, PetscFEGeom *fgeom, PetscDualSpace sp[], PetscInt p,
                                                     PetscTabulation *T, PetscTabulation *TAux,
                                                     void (**funcs)(PetscInt, PetscInt, PetscInt,
                                                                    const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                                    const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                                    PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]), void **ctxs,
                                                     PetscScalar values[])
{
  PetscSection       section, sectionAux = NULL;
  PetscScalar       *u, *u_t = NULL, *u_x, *a = NULL, *a_t = NULL, *a_x = NULL, *bc;
  PetscScalar       *coefficients   = NULL, *coefficientsAux   = NULL;
  PetscScalar       *coefficients_t = NULL, *coefficientsAux_t = NULL;
  const PetscScalar *constants;
  PetscReal         *x;
  PetscInt          *uOff, *uOff_x, *aOff = NULL, *aOff_x = NULL, *Nc;
  PetscFEGeom        fegeom, cgeom;
  const PetscInt     dE = fgeom->dimEmbed;
  PetscInt           numConstants, Nf, NfAux = 0, f, spDim, d, v, tp = 0;
  PetscBool          isAffine;

  PetscFunctionBeginHot;
  PetscCheck(dm == dmIn,PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "Not yet upgraded to use different input DM");
  PetscCall(PetscDSGetNumFields(ds, &Nf));
  PetscCall(PetscDSGetComponents(ds, &Nc));
  PetscCall(PetscDSGetComponentOffsets(ds, &uOff));
  PetscCall(PetscDSGetComponentDerivativeOffsets(ds, &uOff_x));
  PetscCall(PetscDSGetEvaluationArrays(ds, &u, &bc /*&u_t*/, &u_x));
  PetscCall(PetscDSGetWorkspace(ds, &x, NULL, NULL, NULL, NULL));
  PetscCall(PetscDSGetConstants(ds, &numConstants, &constants));
  PetscCall(DMGetLocalSection(dm, &section));
  PetscCall(DMPlexVecGetClosure(dmIn, section, localU, p, NULL, &coefficients));
  if (dmAux) {
    PetscInt subp;

    PetscCall(DMGetEnclosurePoint(dmAux, dm, encAux, p, &subp));
    PetscCall(PetscDSGetNumFields(dsAux, &NfAux));
    PetscCall(DMGetLocalSection(dmAux, &sectionAux));
    PetscCall(PetscDSGetComponentOffsets(dsAux, &aOff));
    PetscCall(PetscDSGetComponentDerivativeOffsets(dsAux, &aOff_x));
    PetscCall(PetscDSGetEvaluationArrays(dsAux, &a, NULL /*&a_t*/, &a_x));
    PetscCall(DMPlexVecGetClosure(dmAux, sectionAux, localA, subp, NULL, &coefficientsAux));
  }
  /* Get values for closure */
  isAffine = fgeom->isAffine;
  fegeom.n  = NULL;
  fegeom.J  = NULL;
  fegeom.v  = NULL;
  fegeom.xi = NULL;
  cgeom.dim      = fgeom->dim;
  cgeom.dimEmbed = fgeom->dimEmbed;
  if (isAffine) {
    fegeom.v    = x;
    fegeom.xi   = fgeom->xi;
    fegeom.J    = fgeom->J;
    fegeom.invJ = fgeom->invJ;
    fegeom.detJ = fgeom->detJ;
    fegeom.n    = fgeom->n;

    cgeom.J     = fgeom->suppJ[0];
    cgeom.invJ  = fgeom->suppInvJ[0];
    cgeom.detJ  = fgeom->suppDetJ[0];
  }
  for (f = 0, v = 0; f < Nf; ++f) {
    PetscQuadrature   allPoints;
    PetscInt          q, dim, numPoints;
    const PetscReal   *points;
    PetscScalar       *pointEval;
    DM                dm;

    if (!sp[f]) continue;
    PetscCall(PetscDualSpaceGetDimension(sp[f], &spDim));
    if (!funcs[f]) {
      for (d = 0; d < spDim; d++, v++) values[v] = 0.;
      continue;
    }
    PetscCall(PetscDualSpaceGetDM(sp[f],&dm));
    PetscCall(PetscDualSpaceGetAllData(sp[f], &allPoints, NULL));
    PetscCall(PetscQuadratureGetData(allPoints,&dim,NULL,&numPoints,&points,NULL));
    PetscCall(DMGetWorkArray(dm,numPoints*Nc[f],MPIU_SCALAR,&pointEval));
    for (q = 0; q < numPoints; ++q, ++tp) {
      if (isAffine) {
        CoordinatesRefToReal(dE, fgeom->dim, fegeom.xi, fgeom->v, fegeom.J, &points[q*dim], x);
      } else {
        fegeom.v    = &fgeom->v[tp*dE];
        fegeom.J    = &fgeom->J[tp*dE*dE];
        fegeom.invJ = &fgeom->invJ[tp*dE*dE];
        fegeom.detJ = &fgeom->detJ[tp];
        fegeom.n    = &fgeom->n[tp*dE];

        cgeom.J     = &fgeom->suppJ[0][tp*dE*dE];
        cgeom.invJ  = &fgeom->suppInvJ[0][tp*dE*dE];
        cgeom.detJ  = &fgeom->suppDetJ[0][tp];
      }
      /* TODO We should use cgeom here, instead of fegeom, however the geometry coming in through fgeom does not have the support cell geometry */
      PetscCall(PetscFEEvaluateFieldJets_Internal(ds, Nf, 0, tp, T, &cgeom, coefficients, coefficients_t, u, u_x, u_t));
      if (dsAux) PetscCall(PetscFEEvaluateFieldJets_Internal(dsAux, NfAux, 0, tp, TAux, &cgeom, coefficientsAux, coefficientsAux_t, a, a_x, a_t));
      (*funcs[f])(dE, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, a_t, a_x, time, fegeom.v, fegeom.n, numConstants, constants, &pointEval[Nc[f]*q]);
    }
    PetscCall(PetscDualSpaceApplyAll(sp[f], pointEval, &values[v]));
    PetscCall(DMRestoreWorkArray(dm,numPoints*Nc[f],MPIU_SCALAR,&pointEval));
    v += spDim;
  }
  PetscCall(DMPlexVecRestoreClosure(dmIn, section, localU, p, NULL, &coefficients));
  if (dmAux) PetscCall(DMPlexVecRestoreClosure(dmAux, sectionAux, localA, p, NULL, &coefficientsAux));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMProjectPoint_Private(DM dm, PetscDS ds, DM dmIn, DMEnclosureType encIn, PetscDS dsIn, DM dmAux, DMEnclosureType encAux, PetscDS dsAux, PetscFEGeom *fegeom, PetscInt effectiveHeight, PetscReal time, Vec localU, Vec localA, PetscBool hasFE, PetscBool hasFV, PetscBool isFE[],
                                             PetscDualSpace sp[], PetscInt p, PetscTabulation *T, PetscTabulation *TAux,
                                             DMBoundaryConditionType type, void (**funcs)(void), void **ctxs, PetscBool fieldActive[], PetscScalar values[])
{
  PetscFVCellGeom fvgeom;
  PetscInt        dim, dimEmbed;

  PetscFunctionBeginHot;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMGetCoordinateDim(dm, &dimEmbed));
  if (hasFV) PetscCall(DMPlexComputeCellGeometryFVM(dm, p, &fvgeom.volume, fvgeom.centroid, NULL));
  switch (type) {
  case DM_BC_ESSENTIAL:
  case DM_BC_NATURAL:
    PetscCall(DMProjectPoint_Func_Private(dm, ds, dmIn, dsIn, time, fegeom, &fvgeom, isFE, sp, (PetscErrorCode (**)(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void *)) funcs, ctxs, values));break;
  case DM_BC_ESSENTIAL_FIELD:
  case DM_BC_NATURAL_FIELD:
    PetscCall(DMProjectPoint_Field_Private(dm, ds, dmIn, encIn, dsIn, dmAux, encAux, dsAux, time, localU, localA, fegeom, sp, p, T, TAux,
                                           (void (**)(PetscInt, PetscInt, PetscInt,
                                                      const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                      const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                      PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[])) funcs, ctxs, values));
    break;
  case DM_BC_ESSENTIAL_BD_FIELD:
    PetscCall(DMProjectPoint_BdField_Private(dm, ds, dmIn, dsIn, dmAux, encAux, dsAux, time, localU, localA, fegeom, sp, p, T, TAux,
                                             (void (**)(PetscInt, PetscInt, PetscInt,
                                                        const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                        const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                        PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[])) funcs, ctxs, values));
    break;
  default: SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "Unknown boundary condition type: %d", (int) type);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceGetAllPointsUnion(PetscInt Nf, PetscDualSpace *sp, PetscInt dim, void (**funcs)(void), PetscQuadrature *allPoints)
{
  PetscReal      *points;
  PetscInt       f, numPoints;

  PetscFunctionBegin;
  if (!dim) {
    PetscCall(PetscQuadratureCreate(PETSC_COMM_SELF, allPoints));
    PetscFunctionReturn(0);
  }
  numPoints = 0;
  for (f = 0; f < Nf; ++f) {
    if (funcs[f]) {
      PetscQuadrature fAllPoints;
      PetscInt        fNumPoints;

      PetscCall(PetscDualSpaceGetAllData(sp[f],&fAllPoints, NULL));
      PetscCall(PetscQuadratureGetData(fAllPoints, NULL, NULL, &fNumPoints, NULL, NULL));
      numPoints += fNumPoints;
    }
  }
  PetscCall(PetscMalloc1(dim*numPoints,&points));
  numPoints = 0;
  for (f = 0; f < Nf; ++f) {
    if (funcs[f]) {
      PetscQuadrature fAllPoints;
      PetscInt        qdim, fNumPoints, q;
      const PetscReal *fPoints;

      PetscCall(PetscDualSpaceGetAllData(sp[f],&fAllPoints, NULL));
      PetscCall(PetscQuadratureGetData(fAllPoints, &qdim, NULL, &fNumPoints, &fPoints, NULL));
      PetscCheck(qdim == dim,PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Spatial dimension %" PetscInt_FMT " for dual basis does not match input dimension %" PetscInt_FMT, qdim, dim);
      for (q = 0; q < fNumPoints*dim; ++q) points[numPoints*dim+q] = fPoints[q];
      numPoints += fNumPoints;
    }
  }
  PetscCall(PetscQuadratureCreate(PETSC_COMM_SELF,allPoints));
  PetscCall(PetscQuadratureSetData(*allPoints,dim,0,numPoints,points,NULL));
  PetscFunctionReturn(0);
}

/*@C
  DMGetFirstLabeledPoint - Find first labeled point p_o in odm such that the corresponding point p in dm has the specified height. Return p and the corresponding ds.

  Input Parameters:
  dm - the DM
  odm - the enclosing DM
  label - label for DM domain, or NULL for whole domain
  numIds - the number of ids
  ids - An array of the label ids in sequence for the domain
  height - Height of target cells in DMPlex topology

  Output Parameters:
  point - the first labeled point
  ds - the ds corresponding to the first labeled point

  Level: developer
@*/
PetscErrorCode DMGetFirstLabeledPoint(DM dm, DM odm, DMLabel label, PetscInt numIds, const PetscInt ids[], PetscInt height, PetscInt *point, PetscDS *ds)
{
  DM              plex;
  DMEnclosureType enc;
  PetscInt        ls = -1;

  PetscFunctionBegin;
  if (point) *point = -1;
  if (!label) PetscFunctionReturn(0);
  PetscCall(DMGetEnclosureRelation(dm, odm, &enc));
  PetscCall(DMConvert(dm, DMPLEX, &plex));
  for (PetscInt i = 0; i < numIds; ++i) {
    IS       labelIS;
    PetscInt num_points, pStart, pEnd;
    PetscCall(DMLabelGetStratumIS(label, ids[i], &labelIS));
    if (!labelIS) continue; /* No points with that id on this process */
    PetscCall(DMPlexGetHeightStratum(plex, height, &pStart, &pEnd));
    PetscCall(ISGetSize(labelIS, &num_points));
    if (num_points) {
      const PetscInt *points;
      PetscCall(ISGetIndices(labelIS, &points));
      for (PetscInt i=0; i<num_points; i++) {
        PetscInt point;
        PetscCall(DMGetEnclosurePoint(dm, odm, enc, points[i], &point));
        if (pStart <= point && point < pEnd) {
          ls = point;
          if (ds) PetscCall(DMGetCellDS(dm, ls, ds));
        }
      }
      PetscCall(ISRestoreIndices(labelIS, &points));
    }
    PetscCall(ISDestroy(&labelIS));
    if (ls >= 0) break;
  }
  if (point) *point = ls;
  PetscCall(DMDestroy(&plex));
  PetscFunctionReturn(0);
}

/*
  This function iterates over a manifold, and interpolates the input function/field using the basis provided by the DS in our DM

  There are several different scenarios:

  1) Volumetric mesh with volumetric auxiliary data

     Here minHeight=0 since we loop over cells.

  2) Boundary mesh with boundary auxiliary data

     Here minHeight=1 since we loop over faces. This normally happens since we hang cells off of our boundary meshes to facilitate computation.

  3) Volumetric mesh with boundary auxiliary data

     Here minHeight=1 and auxbd=PETSC_TRUE since we loop over faces and use data only supported on those faces. This is common when imposing Dirichlet boundary conditions.

  4) Volumetric input mesh with boundary output mesh

     Here we must get a subspace for the input DS

  The maxHeight is used to support enforcement of constraints in DMForest.

  If localU is given and not equal to localX, we call DMPlexInsertBoundaryValues() to complete it.

  If we are using an input field (DM_BC_ESSENTIAL_FIELD or DM_BC_NATURAL_FIELD), we need to evaluate it at all the quadrature points of the dual basis functionals.
    - We use effectiveHeight to mean the height above our incoming DS. For example, if the DS is for a submesh then the effective height is zero, whereas if the DS
      is for the volumetric mesh, but we are iterating over a surface, then the effective height is nonzero. When the effective height is nonzero, we need to extract
      dual spaces for the boundary from our input spaces.
    - After extracting all quadrature points, we tabulate the input fields and auxiliary fields on them.

  We check that the #dof(closure(p)) == #dual basis functionals(p) for a representative p in the iteration

  If we have a label, we iterate over those points. This will probably break the maxHeight functionality since we do not check the height of those points.
*/
static PetscErrorCode DMProjectLocal_Generic_Plex(DM dm, PetscReal time, Vec localU,
                                                  PetscInt Ncc, const PetscInt comps[], DMLabel label, PetscInt numIds, const PetscInt ids[],
                                                  DMBoundaryConditionType type, void (**funcs)(void), void **ctxs,
                                                  InsertMode mode, Vec localX)
{
  DM                 plex, dmIn, plexIn, dmAux = NULL, plexAux = NULL, tdm;
  DMEnclosureType    encIn, encAux;
  PetscDS            ds = NULL, dsIn = NULL, dsAux = NULL;
  Vec                localA = NULL, tv;
  IS                 fieldIS;
  PetscSection       section;
  PetscDualSpace    *sp, *cellsp, *spIn, *cellspIn;
  PetscTabulation *T = NULL, *TAux = NULL;
  PetscInt          *Nc;
  PetscInt           dim, dimEmbed, depth, htInc = 0, htIncIn = 0, htIncAux = 0, minHeight, maxHeight, h, regionNum, Nf, NfIn, NfAux = 0, NfTot, f;
  PetscBool         *isFE, hasFE = PETSC_FALSE, hasFV = PETSC_FALSE, isCohesive = PETSC_FALSE, transform;
  DMField            coordField;
  DMLabel            depthLabel;
  PetscQuadrature    allPoints = NULL;

  PetscFunctionBegin;
  if (localU) PetscCall(VecGetDM(localU, &dmIn));
  else        {dmIn = dm;}
  PetscCall(DMGetAuxiliaryVec(dm, label, numIds ? ids[0] : 0, 0, &localA));
  if (localA) PetscCall(VecGetDM(localA, &dmAux)); else {dmAux = NULL;}
  PetscCall(DMConvert(dm, DMPLEX, &plex));
  PetscCall(DMConvert(dmIn, DMPLEX, &plexIn));
  PetscCall(DMGetEnclosureRelation(dmIn, dm, &encIn));
  PetscCall(DMGetEnclosureRelation(dmAux, dm, &encAux));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetVTKCellHeight(plex, &minHeight));
  PetscCall(DMGetBasisTransformDM_Internal(dm, &tdm));
  PetscCall(DMGetBasisTransformVec_Internal(dm, &tv));
  PetscCall(DMHasBasisTransform(dm, &transform));
  /* Auxiliary information can only be used with interpolation of field functions */
  if (dmAux) {
    PetscCall(DMConvert(dmAux, DMPLEX, &plexAux));
    if (type == DM_BC_ESSENTIAL_FIELD || type == DM_BC_ESSENTIAL_BD_FIELD || type == DM_BC_NATURAL_FIELD) {
      PetscCheck(localA,PETSC_COMM_SELF, PETSC_ERR_USER, "Missing localA vector");
    }
  }
  /* Determine height for iteration of all meshes */
  {
    DMPolytopeType ct, ctIn, ctAux;
    PetscInt       minHeightIn, minHeightAux, lStart, pStart, pEnd, p, pStartIn, pStartAux, pEndAux;
    PetscInt       dim = -1, dimIn = -1, dimAux = -1;

    PetscCall(DMPlexGetSimplexOrBoxCells(plex, minHeight, &pStart, &pEnd));
    if (pEnd > pStart) {
      PetscCall(DMGetFirstLabeledPoint(dm, dm, label, numIds, ids, minHeight, &lStart, NULL));
      p    = lStart < 0 ? pStart : lStart;
      PetscCall(DMPlexGetCellType(plex, p, &ct));
      dim  = DMPolytopeTypeGetDim(ct);
      PetscCall(DMPlexGetVTKCellHeight(plexIn, &minHeightIn));
      PetscCall(DMPlexGetSimplexOrBoxCells(plexIn, minHeightIn, &pStartIn, NULL));
      PetscCall(DMPlexGetCellType(plexIn, pStartIn, &ctIn));
      dimIn = DMPolytopeTypeGetDim(ctIn);
      if (dmAux) {
        PetscCall(DMPlexGetVTKCellHeight(plexAux, &minHeightAux));
        PetscCall(DMPlexGetSimplexOrBoxCells(plexAux, minHeightAux, &pStartAux, &pEndAux));
        if (pStartAux < pEndAux) {
          PetscCall(DMPlexGetCellType(plexAux, pStartAux, &ctAux));
          dimAux = DMPolytopeTypeGetDim(ctAux);
        }
      } else dimAux = dim;
    }
    if (dim < 0) {
      DMLabel spmap = NULL, spmapIn = NULL, spmapAux = NULL;

      /* Fall back to determination based on being a submesh */
      PetscCall(DMPlexGetSubpointMap(plex,   &spmap));
      PetscCall(DMPlexGetSubpointMap(plexIn, &spmapIn));
      if (plexAux) PetscCall(DMPlexGetSubpointMap(plexAux, &spmapAux));
      dim    = spmap    ? 1 : 0;
      dimIn  = spmapIn  ? 1 : 0;
      dimAux = spmapAux ? 1 : 0;
    }
    {
      PetscInt dimProj   = PetscMin(PetscMin(dim, dimIn), (dimAux < 0 ? PETSC_MAX_INT : dimAux));
      PetscInt dimAuxEff = dimAux < 0 ? dimProj : dimAux;

      PetscCheck(PetscAbsInt(dimProj - dim) <= 1 && PetscAbsInt(dimProj - dimIn) <= 1 && PetscAbsInt(dimProj - dimAuxEff) <= 1,PETSC_COMM_SELF, PETSC_ERR_SUP, "Do not currently support differences of more than 1 in dimension");
      if (dimProj < dim) minHeight = 1;
      htInc    =  dim       - dimProj;
      htIncIn  =  dimIn     - dimProj;
      htIncAux =  dimAuxEff - dimProj;
    }
  }
  PetscCall(DMPlexGetDepth(plex, &depth));
  PetscCall(DMPlexGetDepthLabel(plex, &depthLabel));
  PetscCall(DMPlexGetMaxProjectionHeight(plex, &maxHeight));
  maxHeight = PetscMax(maxHeight, minHeight);
  PetscCheck(maxHeight >= 0 && maxHeight <= dim,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Maximum projection height %" PetscInt_FMT " not in [0, %" PetscInt_FMT ")", maxHeight, dim);
  PetscCall(DMGetFirstLabeledPoint(dm, dm, label, numIds, ids, 0, NULL, &ds));
  if (!ds) PetscCall(DMGetDS(dm, &ds));
  PetscCall(DMGetFirstLabeledPoint(dmIn, dm, label, numIds, ids, 0, NULL, &dsIn));
  if (!dsIn) PetscCall(DMGetDS(dmIn, &dsIn));
  PetscCall(PetscDSGetNumFields(ds, &Nf));
  PetscCall(PetscDSGetNumFields(dsIn, &NfIn));
  PetscCall(DMGetNumFields(dm, &NfTot));
  PetscCall(DMFindRegionNum(dm, ds, &regionNum));
  PetscCall(DMGetRegionNumDS(dm, regionNum, NULL, &fieldIS, NULL));
  PetscCall(PetscDSIsCohesive(ds, &isCohesive));
  PetscCall(DMGetCoordinateDim(dm, &dimEmbed));
  PetscCall(DMGetLocalSection(dm, &section));
  if (dmAux) {
    PetscCall(DMGetDS(dmAux, &dsAux));
    PetscCall(PetscDSGetNumFields(dsAux, &NfAux));
  }
  PetscCall(PetscDSGetComponents(ds, &Nc));
  PetscCall(PetscMalloc3(Nf, &isFE, Nf, &sp, NfIn, &spIn));
  if (maxHeight > 0) PetscCall(PetscMalloc2(Nf, &cellsp, NfIn, &cellspIn));
  else               {cellsp = sp; cellspIn = spIn;}
  if (localU && localU != localX) PetscCall(DMPlexInsertBoundaryValues(plex, PETSC_TRUE, localU, time, NULL, NULL, NULL));
  /* Get cell dual spaces */
  for (f = 0; f < Nf; ++f) {
    PetscDiscType disctype;

    PetscCall(PetscDSGetDiscType_Internal(ds, f, &disctype));
    if (disctype == PETSC_DISC_FE) {
      PetscFE fe;

      isFE[f] = PETSC_TRUE;
      hasFE   = PETSC_TRUE;
      PetscCall(PetscDSGetDiscretization(ds, f, (PetscObject *) &fe));
      PetscCall(PetscFEGetDualSpace(fe, &cellsp[f]));
    } else if (disctype == PETSC_DISC_FV) {
      PetscFV fv;

      isFE[f] = PETSC_FALSE;
      hasFV   = PETSC_TRUE;
      PetscCall(PetscDSGetDiscretization(ds, f, (PetscObject *) &fv));
      PetscCall(PetscFVGetDualSpace(fv, &cellsp[f]));
    } else {
      isFE[f]   = PETSC_FALSE;
      cellsp[f] = NULL;
    }
  }
  for (f = 0; f < NfIn; ++f) {
    PetscDiscType disctype;

    PetscCall(PetscDSGetDiscType_Internal(dsIn, f, &disctype));
    if (disctype == PETSC_DISC_FE) {
      PetscFE fe;

      PetscCall(PetscDSGetDiscretization(dsIn, f, (PetscObject *) &fe));
      PetscCall(PetscFEGetDualSpace(fe, &cellspIn[f]));
    } else if (disctype == PETSC_DISC_FV) {
      PetscFV fv;

      PetscCall(PetscDSGetDiscretization(dsIn, f, (PetscObject *) &fv));
      PetscCall(PetscFVGetDualSpace(fv, &cellspIn[f]));
    } else {
      cellspIn[f] = NULL;
    }
  }
  PetscCall(DMGetCoordinateField(dm,&coordField));
  for (f = 0; f < Nf; ++f) {
    if (!htInc) {sp[f] = cellsp[f];}
    else        PetscCall(PetscDualSpaceGetHeightSubspace(cellsp[f], htInc, &sp[f]));
  }
  if (type == DM_BC_ESSENTIAL_FIELD || type == DM_BC_ESSENTIAL_BD_FIELD || type == DM_BC_NATURAL_FIELD) {
    PetscFE          fem, subfem;
    PetscDiscType    disctype;
    const PetscReal *points;
    PetscInt         numPoints;

    PetscCheck(maxHeight <= minHeight,PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "Field projection not supported for face interpolation");
    PetscCall(PetscDualSpaceGetAllPointsUnion(Nf, sp, dim-htInc, funcs, &allPoints));
    PetscCall(PetscQuadratureGetData(allPoints, NULL, NULL, &numPoints, &points, NULL));
    PetscCall(PetscMalloc2(NfIn, &T, NfAux, &TAux));
    for (f = 0; f < NfIn; ++f) {
      if (!htIncIn) {spIn[f] = cellspIn[f];}
      else          PetscCall(PetscDualSpaceGetHeightSubspace(cellspIn[f], htIncIn, &spIn[f]));

      PetscCall(PetscDSGetDiscType_Internal(dsIn, f, &disctype));
      if (disctype != PETSC_DISC_FE) continue;
      PetscCall(PetscDSGetDiscretization(dsIn, f, (PetscObject *) &fem));
      if (!htIncIn) {subfem = fem;}
      else          PetscCall(PetscFEGetHeightSubspace(fem, htIncIn, &subfem));
      PetscCall(PetscFECreateTabulation(subfem, 1, numPoints, points, 1, &T[f]));
    }
    for (f = 0; f < NfAux; ++f) {
      PetscCall(PetscDSGetDiscType_Internal(dsAux, f, &disctype));
      if (disctype != PETSC_DISC_FE) continue;
      PetscCall(PetscDSGetDiscretization(dsAux, f, (PetscObject *) &fem));
      if (!htIncAux) {subfem = fem;}
      else           PetscCall(PetscFEGetHeightSubspace(fem, htIncAux, &subfem));
      PetscCall(PetscFECreateTabulation(subfem, 1, numPoints, points, 1, &TAux[f]));
    }
  }
  /* Note: We make no attempt to optimize for height. Higher height things just overwrite the lower height results. */
  for (h = minHeight; h <= maxHeight; h++) {
    PetscInt     hEff     = h - minHeight + htInc;
    PetscInt     hEffIn   = h - minHeight + htIncIn;
    PetscInt     hEffAux  = h - minHeight + htIncAux;
    PetscDS      dsEff    = ds;
    PetscDS      dsEffIn  = dsIn;
    PetscDS      dsEffAux = dsAux;
    PetscScalar *values;
    PetscBool   *fieldActive;
    PetscInt     maxDegree;
    PetscInt     pStart, pEnd, p, lStart, spDim, totDim, numValues;
    IS           heightIS;

    if (h > minHeight) {
      for (f = 0; f < Nf; ++f) PetscCall(PetscDualSpaceGetHeightSubspace(cellsp[f], hEff, &sp[f]));
    }
    PetscCall(DMPlexGetSimplexOrBoxCells(plex, h, &pStart, &pEnd));
    PetscCall(DMGetFirstLabeledPoint(dm, dm, label, numIds, ids, h, &lStart, NULL));
    PetscCall(DMLabelGetStratumIS(depthLabel, depth - h, &heightIS));
    if (pEnd <= pStart) {
      PetscCall(ISDestroy(&heightIS));
      continue;
    }
    /* Compute totDim, the number of dofs in the closure of a point at this height */
    totDim = 0;
    for (f = 0; f < Nf; ++f) {
      PetscBool cohesive;

      if (!sp[f]) continue;
      PetscCall(PetscDSGetCohesive(ds, f, &cohesive));
      PetscCall(PetscDualSpaceGetDimension(sp[f], &spDim));
      totDim += spDim;
      if (isCohesive && !cohesive) totDim += spDim;
    }
    p    = lStart < 0 ? pStart : lStart;
    PetscCall(DMPlexVecGetClosure(plex, section, localX, p, &numValues, NULL));
    PetscCheck(numValues == totDim,PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "The output section point (%" PetscInt_FMT ") closure size %" PetscInt_FMT " != dual space dimension %" PetscInt_FMT " at height %" PetscInt_FMT " in [%" PetscInt_FMT ", %" PetscInt_FMT "]", p, numValues, totDim, h, minHeight, maxHeight);
    if (!totDim) {
      PetscCall(ISDestroy(&heightIS));
      continue;
    }
    if (htInc) PetscCall(PetscDSGetHeightSubspace(ds, hEff, &dsEff));
    /* Compute totDimIn, the number of dofs in the closure of a point at this height */
    if (localU) {
      PetscInt totDimIn, pIn, numValuesIn;

      totDimIn = 0;
      for (f = 0; f < NfIn; ++f) {
        PetscBool cohesive;

        if (!spIn[f]) continue;
        PetscCall(PetscDSGetCohesive(dsIn, f, &cohesive));
        PetscCall(PetscDualSpaceGetDimension(spIn[f], &spDim));
        totDimIn += spDim;
        if (isCohesive && !cohesive) totDimIn += spDim;
      }
      PetscCall(DMGetEnclosurePoint(dmIn, dm, encIn, lStart < 0 ? pStart : lStart, &pIn));
      PetscCall(DMPlexVecGetClosure(plexIn, NULL, localU, pIn, &numValuesIn, NULL));
      PetscCheck(numValuesIn == totDimIn,PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "The input section point (%" PetscInt_FMT ") closure size %" PetscInt_FMT " != dual space dimension %" PetscInt_FMT " at height %" PetscInt_FMT, pIn, numValuesIn, totDimIn, htIncIn);
      if (htIncIn) PetscCall(PetscDSGetHeightSubspace(dsIn, hEffIn, &dsEffIn));
    }
    if (htIncAux) PetscCall(PetscDSGetHeightSubspace(dsAux, hEffAux, &dsEffAux));
    /* Loop over points at this height */
    PetscCall(DMGetWorkArray(dm, numValues, MPIU_SCALAR, &values));
    PetscCall(DMGetWorkArray(dm, NfTot, MPI_INT, &fieldActive));
    {
      const PetscInt *fields;

      PetscCall(ISGetIndices(fieldIS, &fields));
      for (f = 0; f < NfTot; ++f) {fieldActive[f] = PETSC_FALSE;}
      for (f = 0; f < Nf; ++f) {fieldActive[fields[f]] = (funcs[f] && sp[f]) ? PETSC_TRUE : PETSC_FALSE;}
      PetscCall(ISRestoreIndices(fieldIS, &fields));
    }
    if (label) {
      PetscInt i;

      for (i = 0; i < numIds; ++i) {
        IS              pointIS, isectIS;
        const PetscInt *points;
        PetscInt        n;
        PetscFEGeom  *fegeom = NULL, *chunkgeom = NULL;
        PetscQuadrature quad = NULL;

        PetscCall(DMLabelGetStratumIS(label, ids[i], &pointIS));
        if (!pointIS) continue; /* No points with that id on this process */
        PetscCall(ISIntersect(pointIS,heightIS,&isectIS));
        PetscCall(ISDestroy(&pointIS));
        if (!isectIS) continue;
        PetscCall(ISGetLocalSize(isectIS, &n));
        PetscCall(ISGetIndices(isectIS, &points));
        PetscCall(DMFieldGetDegree(coordField,isectIS,NULL,&maxDegree));
        if (maxDegree <= 1) {
          PetscCall(DMFieldCreateDefaultQuadrature(coordField,isectIS,&quad));
        }
        if (!quad) {
          if (!h && allPoints) {
            quad = allPoints;
            allPoints = NULL;
          } else {
            PetscCall(PetscDualSpaceGetAllPointsUnion(Nf,sp,isCohesive ? dim-htInc-1 : dim-htInc,funcs,&quad));
          }
        }
        PetscCall(DMFieldCreateFEGeom(coordField, isectIS, quad, (htInc && h == minHeight) ? PETSC_TRUE : PETSC_FALSE, &fegeom));
        for (p = 0; p < n; ++p) {
          const PetscInt  point = points[p];

          PetscCall(PetscArrayzero(values, numValues));
          PetscCall(PetscFEGeomGetChunk(fegeom,p,p+1,&chunkgeom));
          PetscCall(DMPlexSetActivePoint(dm, point));
          PetscCall(DMProjectPoint_Private(dm, dsEff, plexIn, encIn, dsEffIn, plexAux, encAux, dsEffAux, chunkgeom, htInc, time, localU, localA, hasFE, hasFV, isFE, sp, point, T, TAux, type, funcs, ctxs, fieldActive, values));
          if (transform) PetscCall(DMPlexBasisTransformPoint_Internal(plex, tdm, tv, point, fieldActive, PETSC_FALSE, values));
          PetscCall(DMPlexVecSetFieldClosure_Internal(plex, section, localX, fieldActive, point, Ncc, comps, label, ids[i], values, mode));
        }
        PetscCall(PetscFEGeomRestoreChunk(fegeom,p,p+1,&chunkgeom));
        PetscCall(PetscFEGeomDestroy(&fegeom));
        PetscCall(PetscQuadratureDestroy(&quad));
        PetscCall(ISRestoreIndices(isectIS, &points));
        PetscCall(ISDestroy(&isectIS));
      }
    } else {
      PetscFEGeom    *fegeom = NULL, *chunkgeom = NULL;
      PetscQuadrature quad = NULL;
      IS              pointIS;

      PetscCall(ISCreateStride(PETSC_COMM_SELF,pEnd-pStart,pStart,1,&pointIS));
      PetscCall(DMFieldGetDegree(coordField,pointIS,NULL,&maxDegree));
      if (maxDegree <= 1) {
        PetscCall(DMFieldCreateDefaultQuadrature(coordField,pointIS,&quad));
      }
      if (!quad) {
        if (!h && allPoints) {
          quad = allPoints;
          allPoints = NULL;
        } else {
          PetscCall(PetscDualSpaceGetAllPointsUnion(Nf, sp, dim-htInc, funcs, &quad));
        }
      }
      PetscCall(DMFieldCreateFEGeom(coordField, pointIS, quad, (htInc && h == minHeight) ? PETSC_TRUE : PETSC_FALSE, &fegeom));
      for (p = pStart; p < pEnd; ++p) {
        PetscCall(PetscArrayzero(values, numValues));
        PetscCall(PetscFEGeomGetChunk(fegeom,p-pStart,p-pStart+1,&chunkgeom));
        PetscCall(DMPlexSetActivePoint(dm, p));
        PetscCall(DMProjectPoint_Private(dm, dsEff, plexIn, encIn, dsEffIn, plexAux, encAux, dsEffAux, chunkgeom, htInc, time, localU, localA, hasFE, hasFV, isFE, sp, p, T, TAux, type, funcs, ctxs, fieldActive, values));
        if (transform) PetscCall(DMPlexBasisTransformPoint_Internal(plex, tdm, tv, p, fieldActive, PETSC_FALSE, values));
        PetscCall(DMPlexVecSetFieldClosure_Internal(plex, section, localX, fieldActive, p, Ncc, comps, NULL, -1, values, mode));
      }
      PetscCall(PetscFEGeomRestoreChunk(fegeom,p-pStart,pStart-p+1,&chunkgeom));
      PetscCall(PetscFEGeomDestroy(&fegeom));
      PetscCall(PetscQuadratureDestroy(&quad));
      PetscCall(ISDestroy(&pointIS));
    }
    PetscCall(ISDestroy(&heightIS));
    PetscCall(DMRestoreWorkArray(dm, numValues, MPIU_SCALAR, &values));
    PetscCall(DMRestoreWorkArray(dm, Nf, MPI_INT, &fieldActive));
  }
  /* Cleanup */
  if (type == DM_BC_ESSENTIAL_FIELD || type == DM_BC_ESSENTIAL_BD_FIELD || type == DM_BC_NATURAL_FIELD) {
    for (f = 0; f < NfIn;  ++f) PetscCall(PetscTabulationDestroy(&T[f]));
    for (f = 0; f < NfAux; ++f) PetscCall(PetscTabulationDestroy(&TAux[f]));
    PetscCall(PetscFree2(T, TAux));
  }
  PetscCall(PetscQuadratureDestroy(&allPoints));
  PetscCall(PetscFree3(isFE, sp, spIn));
  if (maxHeight > 0) PetscCall(PetscFree2(cellsp, cellspIn));
  PetscCall(DMDestroy(&plex));
  PetscCall(DMDestroy(&plexIn));
  if (dmAux) PetscCall(DMDestroy(&plexAux));
  PetscFunctionReturn(0);
}

PetscErrorCode DMProjectFunctionLocal_Plex(DM dm, PetscReal time, PetscErrorCode (**funcs)(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void *), void **ctxs, InsertMode mode, Vec localX)
{
  PetscFunctionBegin;
  PetscCall(DMProjectLocal_Generic_Plex(dm, time, NULL, 0, NULL, NULL, 0, NULL, DM_BC_ESSENTIAL, (void (**)(void)) funcs, ctxs, mode, localX));
  PetscFunctionReturn(0);
}

PetscErrorCode DMProjectFunctionLabelLocal_Plex(DM dm, PetscReal time, DMLabel label, PetscInt numIds, const PetscInt ids[], PetscInt Ncc, const PetscInt comps[], PetscErrorCode (**funcs)(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void *), void **ctxs, InsertMode mode, Vec localX)
{
  PetscFunctionBegin;
  PetscCall(DMProjectLocal_Generic_Plex(dm, time, NULL, Ncc, comps, label, numIds, ids, DM_BC_ESSENTIAL, (void (**)(void)) funcs, ctxs, mode, localX));
  PetscFunctionReturn(0);
}

PetscErrorCode DMProjectFieldLocal_Plex(DM dm, PetscReal time, Vec localU,
                                        void (**funcs)(PetscInt, PetscInt, PetscInt,
                                                       const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                       const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                       PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                        InsertMode mode, Vec localX)
{
  PetscFunctionBegin;
  PetscCall(DMProjectLocal_Generic_Plex(dm, time, localU, 0, NULL, NULL, 0, NULL, DM_BC_ESSENTIAL_FIELD, (void (**)(void)) funcs, NULL, mode, localX));
  PetscFunctionReturn(0);
}

PetscErrorCode DMProjectFieldLabelLocal_Plex(DM dm, PetscReal time, DMLabel label, PetscInt numIds, const PetscInt ids[], PetscInt Ncc, const PetscInt comps[], Vec localU,
                                             void (**funcs)(PetscInt, PetscInt, PetscInt,
                                                            const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                            const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                            PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                             InsertMode mode, Vec localX)
{
  PetscFunctionBegin;
  PetscCall(DMProjectLocal_Generic_Plex(dm, time, localU, Ncc, comps, label, numIds, ids, DM_BC_ESSENTIAL_FIELD, (void (**)(void)) funcs, NULL, mode, localX));
  PetscFunctionReturn(0);
}

PetscErrorCode DMProjectBdFieldLabelLocal_Plex(DM dm, PetscReal time, DMLabel label, PetscInt numIds, const PetscInt ids[], PetscInt Ncc, const PetscInt comps[], Vec localU,
                                               void (**funcs)(PetscInt, PetscInt, PetscInt,
                                                              const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                              const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                              PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                               InsertMode mode, Vec localX)
{
  PetscFunctionBegin;
  PetscCall(DMProjectLocal_Generic_Plex(dm, time, localU, Ncc, comps, label, numIds, ids, DM_BC_ESSENTIAL_BD_FIELD, (void (**)(void)) funcs, NULL, mode, localX));
  PetscFunctionReturn(0);
}
