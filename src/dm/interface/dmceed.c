#include <petsc/private/dmimpl.h> /*I      "petscdm.h"          I*/
#include <petscdmceed.h>

#ifdef PETSC_HAVE_LIBCEED
  #include <petsc/private/dmpleximpl.h>
  #include <petscdmplexceed.h>
  #include <petscfeceed.h>

/*@C
  DMGetCeed - Get the LibCEED context associated with this `DM`

  Not Collective

  Input Parameter:
. DM   - The `DM`

  Output Parameter:
. ceed - The LibCEED context

  Level: intermediate

.seealso: `DM`, `DMCreate()`
@*/
PetscErrorCode DMGetCeed(DM dm, Ceed *ceed)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscAssertPointer(ceed, 2);
  if (!dm->ceed) {
    char        ceedresource[PETSC_MAX_PATH_LEN]; /* libCEED resource specifier */
    const char *prefix;

    PetscCall(PetscStrncpy(ceedresource, "/cpu/self", sizeof(ceedresource)));
    PetscCall(PetscObjectGetOptionsPrefix((PetscObject)dm, &prefix));
    PetscCall(PetscOptionsGetString(NULL, prefix, "-dm_ceed", ceedresource, sizeof(ceedresource), NULL));
    PetscCallCEED(CeedInit(ceedresource, &dm->ceed));
  }
  *ceed = dm->ceed;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static CeedMemType PetscMemType2Ceed(PetscMemType mem_type)
{
  return PetscMemTypeDevice(mem_type) ? CEED_MEM_DEVICE : CEED_MEM_HOST;
}

PetscErrorCode VecGetCeedVector(Vec X, Ceed ceed, CeedVector *cx)
{
  PetscMemType memtype;
  PetscScalar *x;
  PetscInt     n;

  PetscFunctionBegin;
  PetscCall(VecGetLocalSize(X, &n));
  PetscCall(VecGetArrayAndMemType(X, &x, &memtype));
  PetscCallCEED(CeedVectorCreate(ceed, n, cx));
  PetscCallCEED(CeedVectorSetArray(*cx, PetscMemType2Ceed(memtype), CEED_USE_POINTER, x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecRestoreCeedVector(Vec X, CeedVector *cx)
{
  PetscFunctionBegin;
  PetscCall(VecRestoreArrayAndMemType(X, NULL));
  PetscCallCEED(CeedVectorDestroy(cx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecGetCeedVectorRead(Vec X, Ceed ceed, CeedVector *cx)
{
  PetscMemType       memtype;
  const PetscScalar *x;
  PetscInt           n;

  PetscFunctionBegin;
  PetscCall(VecGetLocalSize(X, &n));
  PetscCall(VecGetArrayReadAndMemType(X, &x, &memtype));
  PetscCallCEED(CeedVectorCreate(ceed, n, cx));
  PetscCallCEED(CeedVectorSetArray(*cx, PetscMemType2Ceed(memtype), CEED_USE_POINTER, (PetscScalar *)x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecRestoreCeedVectorRead(Vec X, CeedVector *cx)
{
  PetscFunctionBegin;
  PetscCall(VecRestoreArrayReadAndMemType(X, NULL));
  PetscCallCEED(CeedVectorDestroy(cx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

CEED_QFUNCTION(Geometry2D)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out)
{
  const CeedScalar *x = in[0], *Jac = in[1], *w = in[2];
  CeedScalar       *qdata = out[0];

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; ++i)
  {
    const CeedScalar J[2][2] = {
      {Jac[i + Q * 0], Jac[i + Q * 2]},
      {Jac[i + Q * 1], Jac[i + Q * 3]}
    };
    const CeedScalar det = J[0][0] * J[1][1] - J[0][1] * J[1][0];

    qdata[i + Q * 0] = det * w[i];
    qdata[i + Q * 1] = x[i + Q * 0];
    qdata[i + Q * 2] = x[i + Q * 1];
    qdata[i + Q * 3] = J[1][1] / det;
    qdata[i + Q * 4] = -J[1][0] / det;
    qdata[i + Q * 5] = -J[0][1] / det;
    qdata[i + Q * 6] = J[0][0] / det;
  }
  return CEED_ERROR_SUCCESS;
}

CEED_QFUNCTION(Geometry3D)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out)
{
  const CeedScalar *Jac = in[1], *w = in[2];
  CeedScalar       *qdata = out[0];

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; ++i)
  {
    const CeedScalar J[3][3] = {
      {Jac[i + Q * 0], Jac[i + Q * 3], Jac[i + Q * 6]},
      {Jac[i + Q * 1], Jac[i + Q * 4], Jac[i + Q * 7]},
      {Jac[i + Q * 2], Jac[i + Q * 5], Jac[i + Q * 8]}
    };
    const CeedScalar det = J[0][0] * (J[1][1] * J[2][2] - J[1][2] * J[2][1]) + J[0][1] * (J[1][2] * J[2][0] - J[1][0] * J[2][2]) + J[0][2] * (J[1][0] * J[2][1] - J[1][1] * J[2][0]);

    qdata[i + Q * 0] = det * w[i]; /* det J * weight */
  }
  return CEED_ERROR_SUCCESS;
}

static PetscErrorCode DMCeedCreateGeometry(DM dm, IS cellIS, PetscInt *Nqdata, CeedElemRestriction *erq, CeedVector *qd, DMCeed *soldata)
{
  Ceed              ceed;
  DMCeed            sd;
  PetscDS           ds;
  PetscFE           fe;
  CeedQFunctionUser geom     = NULL;
  const char       *geomName = NULL;
  const PetscInt   *cells;
  PetscInt          dim, cdim, cStart, cEnd, Ncell;
  CeedInt           Nq;

  PetscFunctionBegin;
  PetscCall(PetscCalloc1(1, &sd));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMGetCoordinateDim(dm, &cdim));
  PetscCall(DMGetCeed(dm, &ceed));
  PetscCall(ISGetPointRange(cellIS, &cStart, &cEnd, &cells));
  Ncell = cEnd - cStart;

  PetscCall(DMGetCellDS(dm, cells ? cells[cStart] : cStart, &ds, NULL));
  PetscCall(PetscDSGetDiscretization(ds, 0, (PetscObject *)&fe));
  PetscCall(PetscFEGetCeedBasis(fe, &sd->basis));
  PetscCall(CeedBasisGetNumQuadraturePoints(sd->basis, &Nq));
  PetscCall(DMPlexGetCeedRestriction(dm, NULL, 0, 0, 0, &sd->er));

  *Nqdata = 1 + cdim + cdim * dim; // |J| * w_q, x, J^{-1}
  PetscCallCEED(CeedElemRestrictionCreateStrided(ceed, Ncell, Nq, *Nqdata, Ncell * Nq * (*Nqdata), CEED_STRIDES_BACKEND, erq));

  switch (dim) {
  case 2:
    geom     = Geometry2D;
    geomName = Geometry2D_loc;
    break;
  case 3:
    geom     = Geometry3D;
    geomName = Geometry3D_loc;
    break;
  }
  PetscCallCEED(CeedQFunctionCreateInterior(ceed, 1, geom, geomName, &sd->qf));
  PetscCallCEED(CeedQFunctionAddInput(sd->qf, "x", cdim, CEED_EVAL_INTERP));
  PetscCallCEED(CeedQFunctionAddInput(sd->qf, "dx", cdim * dim, CEED_EVAL_GRAD));
  PetscCallCEED(CeedQFunctionAddInput(sd->qf, "weight", 1, CEED_EVAL_WEIGHT));
  PetscCallCEED(CeedQFunctionAddOutput(sd->qf, "qdata", *Nqdata, CEED_EVAL_NONE));

  PetscCallCEED(CeedOperatorCreate(ceed, sd->qf, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &sd->op));
  PetscCallCEED(CeedOperatorSetField(sd->op, "x", sd->er, sd->basis, CEED_VECTOR_ACTIVE));
  PetscCallCEED(CeedOperatorSetField(sd->op, "dx", sd->er, sd->basis, CEED_VECTOR_ACTIVE));
  PetscCallCEED(CeedOperatorSetField(sd->op, "weight", CEED_ELEMRESTRICTION_NONE, sd->basis, CEED_VECTOR_NONE));
  PetscCallCEED(CeedOperatorSetField(sd->op, "qdata", *erq, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));

  PetscCallCEED(CeedElemRestrictionCreateVector(*erq, qd, NULL));
  *soldata = sd;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMRefineHook_Ceed(DM coarse, DM fine, void *ctx)
{
  PetscFunctionBegin;
  if (coarse->dmceed) PetscCall(DMCeedCreate(fine, coarse->dmceed->geom ? PETSC_TRUE : PETSC_FALSE, coarse->dmceed->func, coarse->dmceed->funcSource));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMCeedCreate_Internal(DM dm, IS cellIS, PetscBool createGeometry, CeedQFunctionUser func, const char *func_source, DMCeed *soldata)
{
  PetscDS  ds;
  PetscFE  fe;
  DMCeed   sd;
  Ceed     ceed;
  PetscInt dim, Nc, Nqdata = 0;
  CeedInt  Nq;

  PetscFunctionBegin;
  PetscCall(PetscCalloc1(1, &sd));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMGetCeed(dm, &ceed));
  PetscCall(DMGetDS(dm, &ds));
  PetscCall(PetscDSGetDiscretization(ds, 0, (PetscObject *)&fe));
  PetscCall(PetscFEGetCeedBasis(fe, &sd->basis));
  PetscCall(PetscFEGetNumComponents(fe, &Nc));
  PetscCall(CeedBasisGetNumQuadraturePoints(sd->basis, &Nq));
  PetscCall(DMPlexGetCeedRestriction(dm, NULL, 0, 0, 0, &sd->er));

  if (createGeometry) {
    DM cdm;

    PetscCall(DMGetCoordinateDM(dm, &cdm));
    PetscCall(DMCeedCreateGeometry(cdm, cellIS, &Nqdata, &sd->erq, &sd->qd, &sd->geom));
  }

  if (sd->geom) {
    PetscInt cdim;
    CeedInt  Nqx;

    PetscCallCEED(CeedBasisGetNumQuadraturePoints(sd->geom->basis, &Nqx));
    PetscCheck(Nqx == Nq, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_INCOMP, "Number of qpoints for solution %" CeedInt_FMT " != %" CeedInt_FMT " Number of qpoints for coordinates", Nq, Nqx);
    /* TODO Remove this limitation */
    PetscCall(DMGetCoordinateDim(dm, &cdim));
    PetscCheck(dim == cdim, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_INCOMP, "Topological dimension %" PetscInt_FMT " != %" PetscInt_FMT " embedding dimension", dim, cdim);
  }

  PetscCallCEED(CeedQFunctionCreateInterior(ceed, 1, func, func_source, &sd->qf));
  PetscCallCEED(CeedQFunctionAddInput(sd->qf, "u", Nc, CEED_EVAL_INTERP));
  PetscCallCEED(CeedQFunctionAddInput(sd->qf, "du", Nc * dim, CEED_EVAL_GRAD));
  PetscCallCEED(CeedQFunctionAddInput(sd->qf, "qdata", Nqdata, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddOutput(sd->qf, "v", Nc, CEED_EVAL_INTERP));
  PetscCallCEED(CeedQFunctionAddOutput(sd->qf, "dv", Nc * dim, CEED_EVAL_GRAD));

  PetscCallCEED(CeedOperatorCreate(ceed, sd->qf, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &sd->op));
  PetscCallCEED(CeedOperatorSetField(sd->op, "u", sd->er, sd->basis, CEED_VECTOR_ACTIVE));
  PetscCallCEED(CeedOperatorSetField(sd->op, "du", sd->er, sd->basis, CEED_VECTOR_ACTIVE));
  PetscCallCEED(CeedOperatorSetField(sd->op, "qdata", sd->erq, CEED_BASIS_NONE, sd->qd));
  PetscCallCEED(CeedOperatorSetField(sd->op, "v", sd->er, sd->basis, CEED_VECTOR_ACTIVE));
  PetscCallCEED(CeedOperatorSetField(sd->op, "dv", sd->er, sd->basis, CEED_VECTOR_ACTIVE));

  // Handle refinement
  sd->func = func;
  PetscCall(PetscStrallocpy(func_source, &sd->funcSource));
  PetscCall(DMRefineHookAdd(dm, DMRefineHook_Ceed, NULL, NULL));

  *soldata = sd;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMCeedCreate(DM dm, PetscBool createGeometry, CeedQFunctionUser func, const char *func_source)
{
  DM plex;
  IS cellIS;

  PetscFunctionBegin;
  PetscCall(DMConvert(dm, DMPLEX, &plex));
  PetscCall(DMPlexGetAllCells_Internal(plex, &cellIS));
  #ifdef PETSC_HAVE_LIBCEED
  PetscCall(DMCeedCreate_Internal(dm, cellIS, createGeometry, func, func_source, &dm->dmceed));
  #endif
  PetscCall(ISDestroy(&cellIS));
  PetscCall(DMDestroy(&plex));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMCeedCreateGeometryFVM(DM dm, IS faceIS, PetscInt *Nqdata, CeedElemRestriction *erq, CeedVector *qd, DMCeed *soldata)
{
  Ceed            ceed;
  DMCeed          sd;
  const PetscInt *faces;
  CeedInt         strides[3];
  PetscInt        dim, cdim, fStart, fEnd, Nface, Nq = 1;

  PetscFunctionBegin;
  PetscCall(PetscCalloc1(1, &sd));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMGetCoordinateDim(dm, &cdim));
  PetscCall(DMGetCeed(dm, &ceed));
  PetscCall(ISGetPointRange(faceIS, &fStart, &fEnd, &faces));
  Nface = fEnd - fStart;

  *Nqdata    = cdim + 2; // face normal and support cell volumes
  strides[0] = 1;
  strides[1] = Nq;
  strides[2] = Nq * (*Nqdata);
  PetscCallCEED(CeedElemRestrictionCreateStrided(ceed, Nface, Nq, *Nqdata, Nface * Nq * (*Nqdata), strides, erq));
  PetscCallCEED(CeedElemRestrictionCreateVector(*erq, qd, NULL));
  *soldata = sd;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMCeedCreateInfoFVM(DM dm, IS faceIS, PetscInt *Nqinfo, CeedElemRestriction *eri, CeedVector *qi, DMCeed *solinfo)
{
  Ceed            ceed;
  DMCeed          si;
  const PetscInt *faces;
  CeedInt         strides[3];
  PetscInt        dim, fStart, fEnd, Nface, Nq = 1;

  PetscFunctionBegin;
  PetscCall(PetscCalloc1(1, &si));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMGetCeed(dm, &ceed));
  PetscCall(ISGetPointRange(faceIS, &fStart, &fEnd, &faces));
  Nface = fEnd - fStart;

  *Nqinfo    = 3; // face number and support cell numbers
  strides[0] = 1;
  strides[1] = Nq;
  strides[2] = Nq * (*Nqinfo);
  PetscCallCEED(CeedElemRestrictionCreateStrided(ceed, Nface, Nq, *Nqinfo, Nface * Nq * (*Nqinfo), strides, eri));
  PetscCallCEED(CeedElemRestrictionCreateVector(*eri, qi, NULL));
  *solinfo = si;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMCeedCreateFVM_Internal(DM dm, IS faceIS, PetscBool createGeometry, PetscBool createInfo, CeedQFunctionUser func, const char *func_source, DMCeed *soldata, CeedQFunctionContext qfCtx)
{
  PetscDS  ds;
  PetscFV  fv;
  DMCeed   sd;
  Ceed     ceed;
  PetscInt dim, Nc, Nqdata = 0, Nqinfo = 0;

  PetscFunctionBegin;
  PetscCall(PetscCalloc1(1, &sd));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMGetCeed(dm, &ceed));
  PetscCall(DMGetDS(dm, &ds));
  PetscCall(PetscDSGetDiscretization(ds, 0, (PetscObject *)&fv));
  PetscCall(PetscFVGetNumComponents(fv, &Nc));
  PetscCall(DMPlexCreateCeedRestrictionFVM(dm, &sd->erL, &sd->erR));

  if (createGeometry) {
    DM cdm;

    PetscCall(DMGetCoordinateDM(dm, &cdm));
    PetscCall(DMCeedCreateGeometryFVM(cdm, faceIS, &Nqdata, &sd->erq, &sd->qd, &sd->geom));
  }

  if (createInfo) {
    DM cdm;

    PetscCall(DMGetCoordinateDM(dm, &cdm));
    PetscCall(DMCeedCreateInfoFVM(cdm, faceIS, &Nqinfo, &sd->eri, &sd->qi, &sd->info));
    PetscCall(DMCeedComputeInfo(dm, sd));
  }

  PetscCallCEED(CeedQFunctionCreateInterior(ceed, 1, func, func_source, &sd->qf));
  PetscCallCEED(CeedQFunctionAddInput(sd->qf, "uL", Nc, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddInput(sd->qf, "uR", Nc, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddInput(sd->qf, "geom", Nqdata, CEED_EVAL_NONE));
  if (createInfo) PetscCallCEED(CeedQFunctionAddInput(sd->qf, "info", Nqinfo, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddOutput(sd->qf, "cL", Nc, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddOutput(sd->qf, "cR", Nc, CEED_EVAL_NONE));

  PetscCallCEED(CeedQFunctionSetContext(sd->qf, qfCtx));

  PetscCallCEED(CeedOperatorCreate(ceed, sd->qf, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &sd->op));
  PetscCallCEED(CeedOperatorSetField(sd->op, "uL", sd->erL, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));
  PetscCallCEED(CeedOperatorSetField(sd->op, "uR", sd->erR, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));
  PetscCallCEED(CeedOperatorSetField(sd->op, "geom", sd->erq, CEED_BASIS_NONE, sd->qd));
  if (createInfo) PetscCallCEED(CeedOperatorSetField(sd->op, "info", sd->eri, CEED_BASIS_NONE, sd->qi));
  PetscCallCEED(CeedOperatorSetField(sd->op, "cL", sd->erL, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));
  PetscCallCEED(CeedOperatorSetField(sd->op, "cR", sd->erR, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));

  // Handle refinement
  sd->func = func;
  PetscCall(PetscStrallocpy(func_source, &sd->funcSource));
  PetscCall(DMRefineHookAdd(dm, DMRefineHook_Ceed, NULL, NULL));

  *soldata = sd;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMCeedCreateFVM(DM dm, PetscBool createGeometry, CeedQFunctionUser func, const char *func_source, CeedQFunctionContext qfCtx)
{
  DM plex;
  IS faceIS;

  PetscFunctionBegin;
  PetscCall(DMConvert(dm, DMPLEX, &plex));
  PetscCall(DMPlexGetAllFaces_Internal(plex, &faceIS));
  #ifdef PETSC_HAVE_LIBCEED
  PetscCall(DMCeedCreateFVM_Internal(dm, faceIS, createGeometry, PETSC_TRUE, func, func_source, &dm->dmceed, qfCtx));
  #endif
  PetscCall(ISDestroy(&faceIS));
  PetscCall(DMDestroy(&plex));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#endif

PetscErrorCode DMCeedDestroy(DMCeed *pceed)
{
  DMCeed p = *pceed;

  PetscFunctionBegin;
  if (!p) PetscFunctionReturn(PETSC_SUCCESS);
#ifdef PETSC_HAVE_LIBCEED
  PetscCall(PetscFree(p->funcSource));
  if (p->qd) PetscCallCEED(CeedVectorDestroy(&p->qd));
  if (p->qi) PetscCallCEED(CeedVectorDestroy(&p->qi));
  if (p->op) PetscCallCEED(CeedOperatorDestroy(&p->op));
  if (p->qf) PetscCallCEED(CeedQFunctionDestroy(&p->qf));
  if (p->erL) PetscCallCEED(CeedElemRestrictionDestroy(&p->erL));
  if (p->erR) PetscCallCEED(CeedElemRestrictionDestroy(&p->erR));
  if (p->erq) PetscCallCEED(CeedElemRestrictionDestroy(&p->erq));
  if (p->eri) PetscCallCEED(CeedElemRestrictionDestroy(&p->eri));
  PetscCall(DMCeedDestroy(&p->geom));
  PetscCall(DMCeedDestroy(&p->info));
#endif
  PetscCall(PetscFree(p));
  *pceed = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMCeedComputeGeometry(DM dm, DMCeed sd)
{
#ifdef PETSC_HAVE_LIBCEED
  Ceed       ceed;
  Vec        coords;
  CeedVector ccoords;
#endif

  PetscFunctionBegin;
#ifdef PETSC_HAVE_LIBCEED
  PetscCall(DMGetCeed(dm, &ceed));
  PetscCall(DMGetCoordinatesLocal(dm, &coords));
  PetscCall(VecGetCeedVectorRead(coords, ceed, &ccoords));
  if (sd->geom->op) PetscCallCEED(CeedOperatorApply(sd->geom->op, ccoords, sd->qd, CEED_REQUEST_IMMEDIATE));
  else PetscCall(DMPlexCeedComputeGeometryFVM(dm, sd->qd));
  //PetscCallCEED(CeedVectorView(sd->qd, "%g", stdout));
  PetscCall(VecRestoreCeedVectorRead(coords, &ccoords));
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMCeedComputeInfo(DM dm, DMCeed sd)
{
#ifdef PETSC_HAVE_LIBCEED
  CeedScalar *a;
#endif

  PetscFunctionBegin;
#ifdef PETSC_HAVE_LIBCEED
  PetscCallCEED(CeedVectorGetArrayWrite(sd->qi, CEED_MEM_HOST, &a));

  IS              iterIS;
  DMLabel         label = NULL;
  const PetscInt *indices;
  PetscInt        value = 0, height = 1, NfInt = 0, Nf = 0;

  PetscCall(DMGetPoints_Internal(dm, label, value, height, &iterIS));
  if (iterIS) {
    PetscCall(ISGetIndices(iterIS, &indices));
    PetscCall(ISGetLocalSize(iterIS, &Nf));
    for (PetscInt p = 0, Ns; p < Nf; ++p) {
      PetscCall(DMPlexGetSupportSize(dm, indices[p], &Ns));
      if (Ns == 2) ++NfInt;
    }
  } else {
    indices = NULL;
  }

  PetscInt infoOffset = 0;

  for (PetscInt p = 0; p < Nf; ++p) {
    const PetscInt  face = indices[p];
    const PetscInt *supp;
    PetscInt        Ns;

    PetscCall(DMPlexGetSupport(dm, face, &supp));
    PetscCall(DMPlexGetSupportSize(dm, face, &Ns));
    // Ignore boundary faces
    //   TODO check for face on parallel boundary
    if (Ns == 2) {
      a[infoOffset++] = face;
      a[infoOffset++] = supp[0];
      a[infoOffset++] = supp[1];
    }
  }
  PetscCheck(infoOffset == NfInt * 3, PETSC_COMM_SELF, PETSC_ERR_SUP, "Shape mismatch, info offsets array of shape (%" PetscInt_FMT ") initialized for %" PetscInt_FMT " nodes", infoOffset, NfInt * 3);
  if (iterIS) PetscCall(ISRestoreIndices(iterIS, &indices));
  PetscCall(ISDestroy(&iterIS));

  PetscCallCEED(CeedVectorRestoreArray(sd->qi, &a));
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}
