#include "petscsys.h"
#include <petsc/private/dmpleximpl.h> /*I      "petscdmplex.h"   I*/
#include <petsc/private/hashmapi.h>

/* We need to understand how to natively parse STEP files. There seems to be only one open-source implementation of
   the STEP parser contained in the OpenCASCADE package. It is enough to make a strong man weep:

     https://github.com/tpaviot/oce/tree/master/src/STEPControl

   The STEP, and inner EXPRESS, formats are ISO standards, so they are documented

     https://stackoverflow.com/questions/26774037/documentation-or-specification-for-step-and-stp-files
     http://stepmod.sourceforge.net/express_model_spec/

   but again it seems that there has been a deliberate effort at obfuscation, probably to raise the bar for entrants.
*/

#ifdef PETSC_HAVE_EGADS
  #include <petscdmplexegads.h>

PETSC_INTERN PetscErrorCode DMSnapToGeomModel_EGADS_Internal(DM, PetscInt, ego, PetscInt, PetscInt, PetscInt, const PetscScalar[], PetscScalar[], PetscBool);
PETSC_INTERN PetscErrorCode DMPlex_Geom_EDGE_XYZtoUV_Internal(const PetscScalar[], ego, const PetscScalar[], const PetscInt, const PetscInt, PetscScalar[], PetscBool);
PETSC_INTERN PetscErrorCode DMPlex_Geom_FACE_XYZtoUV_Internal(const PetscScalar[], ego, const PetscScalar[], const PetscInt, const PetscInt, PetscScalar[], PetscBool);

PetscErrorCode DMPlex_EGADS_GeomDecode_Internal(const PetscInt geomClass, const PetscInt geomType, char **retClass, char **retType)
{
  PetscFunctionBeginHot;
  /* EGADS Object Type */
  if (geomClass == CONTXT) { *retClass = (char *)"CONTEXT"; }
  if (geomClass == TRANSFORM) { *retClass = (char *)"TRANSFORM"; }
  if (geomClass == TESSELLATION) { *retClass = (char *)"TESSELLATION"; }
  if (geomClass == NIL) { *retClass = (char *)"NIL"; }
  if (geomClass == EMPTY) { *retClass = (char *)"EMPTY"; }
  if (geomClass == REFERENCE) { *retClass = (char *)"REFERENCE"; }
  if (geomClass == PCURVE) { *retClass = (char *)"PCURVE"; }
  if (geomClass == CURVE) { *retClass = (char *)"CURVE"; }
  if (geomClass == SURFACE) { *retClass = (char *)"SURFACE"; }
  if (geomClass == NODE) { *retClass = (char *)"NODE"; }
  if (geomClass == EDGE) { *retClass = (char *)"EDGE"; }
  if (geomClass == LOOP) { *retClass = (char *)"LOOP"; }
  if (geomClass == FACE) { *retClass = (char *)"FACE"; }
  if (geomClass == SHELL) { *retClass = (char *)"SHELL"; }
  if (geomClass == BODY) { *retClass = (char *)"BODY"; }
  if (geomClass == MODEL) { *retClass = (char *)"MODEL"; }

  /* PCURVES & CURVES */
  if (geomClass == PCURVE || geomClass == CURVE) {
    if (geomType == LINE) { *retType = (char *)"LINE"; }
    if (geomType == CIRCLE) { *retType = (char *)"CIRCLE"; }
    if (geomType == ELLIPSE) { *retType = (char *)"ELLIPSE"; }
    if (geomType == PARABOLA) { *retType = (char *)"PARABOLA"; }
    if (geomType == HYPERBOLA) { *retType = (char *)"HYPERBOLA"; }
    if (geomType == TRIMMED) { *retType = (char *)"TRIMMED"; }
    if (geomType == BEZIER) { *retType = (char *)"BEZIER"; }
    if (geomType == BSPLINE) { *retType = (char *)"BSPLINE"; }
    if (geomType == OFFSET) { *retType = (char *)"OFFSET"; }
  }

  /* SURFACE */
  if (geomClass == SURFACE) {
    if (geomType == PLANE) { *retType = (char *)"PLANE"; }
    if (geomType == SPHERICAL) { *retType = (char *)"SPHERICAL"; }
    if (geomType == CYLINDRICAL) { *retType = (char *)"CYLINDRICAL"; }
    if (geomType == REVOLUTION) { *retType = (char *)"REVOLUTION"; }
    if (geomType == TOROIDAL) { *retType = (char *)"TOROIDAL"; }
    if (geomType == CONICAL) { *retType = (char *)"CONICAL"; }
    if (geomType == EXTRUSION) { *retType = (char *)"EXTRUSION"; }
    if (geomType == BEZIER) { *retType = (char *)"BEZIER"; }
    if (geomType == BSPLINE) { *retType = (char *)"BSPLINE"; }
  }

  /* TOPOLOGY */
  if (geomClass == NODE || geomClass == EDGE || geomClass == LOOP || geomClass == FACE || geomClass == SHELL || geomClass == BODY || geomClass == MODEL) {
    if (geomType == SREVERSE) { *retType = (char *)"SREVERSE"; }
    if (geomType == NOMTYPE) { *retType = (char *)"NOMTYPE"; }
    if (geomType == SFORWARD && geomClass == FACE) { *retType = (char *)"SFORWARD"; }
    if (geomType == ONENODE && geomClass == EDGE) { *retType = (char *)"ONENODE"; }
    if (geomType == TWONODE) { *retType = (char *)"TWONODE"; }
    if (geomType == OPEN) { *retType = (char *)"OPEN"; }
    if (geomType == CLOSED) { *retType = (char *)"CLOSED"; }
    if (geomType == DEGENERATE) { *retType = (char *)"DEGENERATE"; }
    if (geomType == WIREBODY) { *retType = (char *)"WIREBODY"; }
    if (geomType == FACEBODY) { *retType = (char *)"FACEBODY"; }
    if (geomType == SHEETBODY) { *retType = (char *)"SHEETBODY"; }
    if (geomType == SOLIDBODY) { *retType = (char *)"SOLIDBODY"; }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMPlex_EGADS_EDGE_XYZtoUV_Internal(const PetscScalar coords[], ego obj, const PetscScalar range[], const PetscInt v, const PetscInt dE, PetscScalar paramsV[])
{
  //
  //
  // Depreciated. Changed all references to DMPlex_Geom_FACE_XYZtoUV_Internal()
  //
  //

  PetscInt    loopCntr = 0;
  PetscScalar dx, dy, dz, lambda, tolr, obj_old, obj_tmp, target;
  PetscScalar delta, A, b;
  PetscScalar ts[2], tt[2], eval[18], data[18];

  PetscFunctionBeginHot;
  /* Initialize Levenberg-Marquardt parameters */
  lambda = 1.0;
  tolr   = 1.0;
  target = 1.0E-20;
  ts[0]  = (range[0] + range[1]) / 2.;

  while (tolr >= target) {
    PetscCall(EG_evaluate(obj, ts, eval));
    dx      = coords[v * dE + 0] - eval[0];
    dy      = coords[v * dE + 1] - eval[1];
    dz      = coords[v * dE + 2] - eval[2];
    obj_old = dx * dx + dy * dy + dz * dz;

    if (obj_old < target) {
      tolr = obj_old;
      break;
    }

    A = (eval[3] * eval[3] + eval[4] * eval[4] + eval[5] * eval[5]) * (1.0 + lambda);
    if (A == 0.0) {
      PetscCall(PetscPrintf(PETSC_COMM_SELF, "A = 0.0 \n"));
      break;
    }
    b = eval[3] * dx + eval[4] * dy + eval[5] * dz;

    /* Solve A*delta = b */
    delta = b / A;

    /* Find a temp (u,v) and associated objective function */
    tt[0] = ts[0] + delta;
    if (tt[0] < range[0]) {
      tt[0] = range[0];
      delta = tt[0] - ts[0];
    }
    if (tt[0] > range[1]) {
      tt[0] = range[1];
      delta = tt[0] - ts[0];
    }

    PetscCall(EG_evaluate(obj, tt, data));

    obj_tmp = (coords[v * dE + 0] - data[0]) * (coords[v * dE + 0] - data[0]) + (coords[v * dE + 1] - data[1]) * (coords[v * dE + 1] - data[1]) + (coords[v * dE + 2] - data[2]) * (coords[v * dE + 2] - data[2]);

    /* If step is better, accept it and halve lambda (making it more Newton-like) */
    if (obj_tmp < obj_old) {
      obj_old = obj_tmp;
      ts[0]   = tt[0];
      for (int jj = 0; jj < 18; ++jj) eval[jj] = data[jj];
      lambda /= 2.0;
      if (lambda < 1.0E-14) lambda = 1.0E-14;
      if (obj_old < target) {
        tolr = obj_old;
        break;
      }
    } else {
      /* Otherwise reject it and double lambda (making it more gradient-descent like) */
      lambda *= 2.0;
    }

    if ((tt[0] == range[0]) || (tt[0] == range[1])) break;
    if (fabs(delta) < target) {
      tolr = obj_old;
      break;
    }

    tolr = obj_old;

    loopCntr += 1;
    if (loopCntr > 100) break;
  }
  paramsV[v * 3 + 0] = ts[0];
  paramsV[v * 3 + 1] = 0.;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMPlex_Geom_EDGE_XYZtoUV_Internal(const PetscScalar coords[], ego obj, const PetscScalar range[], const PetscInt v, const PetscInt dE, PetscScalar paramsV[], PetscBool islite)
{
  PetscInt    loopCntr = 0;
  PetscScalar dx, dy, dz, lambda, tolr, obj_old, obj_tmp, target;
  PetscScalar delta, A, b;
  PetscScalar ts[2], tt[2], eval[18], data[18];

  PetscFunctionBeginHot;
  /* Initialize Levenberg-Marquardt parameters */
  lambda = 1.0;
  tolr   = 1.0;
  target = 1.0E-20;
  ts[0]  = (range[0] + range[1]) / 2.;

  while (tolr >= target) {
    if (islite) {
      PetscCall(EGlite_evaluate(obj, ts, eval));
    } else {
      PetscCall(EG_evaluate(obj, ts, eval));
    }

    dx      = coords[v * dE + 0] - eval[0];
    dy      = coords[v * dE + 1] - eval[1];
    dz      = coords[v * dE + 2] - eval[2];
    obj_old = dx * dx + dy * dy + dz * dz;

    if (obj_old < target) {
      tolr = obj_old;
      break;
    }

    A = (eval[3] * eval[3] + eval[4] * eval[4] + eval[5] * eval[5]) * (1.0 + lambda);
    if (A == 0.0) {
      PetscCall(PetscPrintf(PETSC_COMM_SELF, "A = 0.0 \n"));
      break;
    }
    b = eval[3] * dx + eval[4] * dy + eval[5] * dz;

    /* Solve A*delta = b */
    delta = b / A;

    /* Find a temp (u,v) and associated objective function */
    tt[0] = ts[0] + delta;
    if (tt[0] < range[0]) {
      tt[0] = range[0];
      delta = tt[0] - ts[0];
    }
    if (tt[0] > range[1]) {
      tt[0] = range[1];
      delta = tt[0] - ts[0];
    }

    if (islite) {
      PetscCall(EGlite_evaluate(obj, tt, data));
    } else {
      PetscCall(EG_evaluate(obj, tt, data));
    }

    obj_tmp = (coords[v * dE + 0] - data[0]) * (coords[v * dE + 0] - data[0]) + (coords[v * dE + 1] - data[1]) * (coords[v * dE + 1] - data[1]) + (coords[v * dE + 2] - data[2]) * (coords[v * dE + 2] - data[2]);

    /* If step is better, accept it and halve lambda (making it more Newton-like) */
    if (obj_tmp < obj_old) {
      obj_old = obj_tmp;
      ts[0]   = tt[0];
      for (int jj = 0; jj < 18; ++jj) eval[jj] = data[jj];
      lambda /= 2.0;
      if (lambda < 1.0E-14) lambda = 1.0E-14;
      if (obj_old < target) {
        tolr = obj_old;
        break;
      }
    } else {
      /* Otherwise reject it and double lambda (making it more gradient-descent like) */
      lambda *= 2.0;
    }

    if ((tt[0] == range[0]) || (tt[0] == range[1])) break;
    if (fabs(delta) < target) {
      tolr = obj_old;
      break;
    }

    tolr = obj_old;

    loopCntr += 1;
    if (loopCntr > 100) break;
  }
  paramsV[v * 3 + 0] = ts[0];
  paramsV[v * 3 + 1] = 0.;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMPlex_EGADS_FACE_XYZtoUV_Internal(const PetscScalar coords[], ego obj, const PetscScalar range[], const PetscInt v, const PetscInt dE, PetscScalar paramsV[])
{
  //
  //
  // Depreciated. Changed all references to DMPlex_Geom_FACE_XYZtoUV_Internal()
  //
  //

  PetscInt    loopCntr = 0;
  PetscScalar dx, dy, dz, lambda, tolr, denom, obj_old, obj_tmp, target;
  PetscScalar uvs[2], uvt[2], delta[2], A[4], b[2], eval[18], data[18];

  PetscFunctionBeginHot;
  /* Initialize Levenberg-Marquardt parameters */
  lambda = 1.0;
  tolr   = 1.0;
  target = 1.0E-20;
  uvs[0] = (range[0] + range[1]) / 2.;
  uvs[1] = (range[2] + range[3]) / 2.;

  while (tolr >= target) {
    PetscCall(EG_evaluate(obj, uvs, eval));
    dx      = coords[v * dE + 0] - eval[0];
    dy      = coords[v * dE + 1] - eval[1];
    dz      = coords[v * dE + 2] - eval[2];
    obj_old = dx * dx + dy * dy + dz * dz;

    if (obj_old < target) {
      tolr = obj_old;
      break;
    }

    A[0] = (eval[3] * eval[3] + eval[4] * eval[4] + eval[5] * eval[5]) * (1.0 + lambda);
    A[1] = eval[3] * eval[6] + eval[4] * eval[7] + eval[5] * eval[8];
    A[2] = A[1];
    A[3] = (eval[6] * eval[6] + eval[7] * eval[7] + eval[8] * eval[8]) * (1.0 + lambda);

    b[0] = eval[3] * dx + eval[4] * dy + eval[5] * dz;
    b[1] = eval[6] * dx + eval[7] * dy + eval[8] * dz;

    /* Solve A*delta = b using Cramer's Rule */
    denom = A[0] * A[3] - A[2] * A[1];
    if (denom == 0.0) { PetscCall(PetscPrintf(PETSC_COMM_SELF, "denom = 0.0 \n")); }
    delta[0] = (b[0] * A[3] - b[1] * A[1]) / denom;
    delta[1] = (A[0] * b[1] - A[2] * b[0]) / denom;

    /* Find a temp (u,v) and associated objective function */
    uvt[0] = uvs[0] + delta[0];
    uvt[1] = uvs[1] + delta[1];

    if (uvt[0] < range[0]) {
      uvt[0]   = range[0];
      delta[0] = uvt[0] - uvs[0];
    }
    if (uvt[0] > range[1]) {
      uvt[0]   = range[1];
      delta[0] = uvt[0] - uvs[0];
    }
    if (uvt[1] < range[2]) {
      uvt[1]   = range[2];
      delta[1] = uvt[1] - uvs[1];
    }
    if (uvt[1] > range[3]) {
      uvt[1]   = range[3];
      delta[1] = uvt[1] - uvs[1];
    }

    PetscCall(EG_evaluate(obj, uvt, data));

    obj_tmp = (coords[v * dE + 0] - data[0]) * (coords[v * dE + 0] - data[0]) + (coords[v * dE + 1] - data[1]) * (coords[v * dE + 1] - data[1]) + (coords[v * dE + 2] - data[2]) * (coords[v * dE + 2] - data[2]);

    /* If step is better, accept it and halve lambda (making it more Newton-like) */
    if (obj_tmp < obj_old) {
      obj_old = obj_tmp;
      uvs[0]  = uvt[0];
      uvs[1]  = uvt[1];
      for (int jj = 0; jj < 18; ++jj) eval[jj] = data[jj];
      lambda /= 2.0;
      if (lambda < 1.0E-14) lambda = 1.0E-14;
      if (obj_old < target) {
        tolr = obj_old;
        break;
      }
    } else {
      /* Otherwise reject it and double lambda (making it more gradient-descent like) */
      lambda *= 2.0;
    }

    if (sqrt(delta[0] * delta[0] + delta[1] * delta[1]) < target) {
      tolr = obj_old;
      break;
    }

    tolr = obj_old;

    loopCntr += 1;
    if (loopCntr > 100) break;
  }
  paramsV[v * 3 + 0] = uvs[0];
  paramsV[v * 3 + 1] = uvs[1];
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMPlex_Geom_FACE_XYZtoUV_Internal(const PetscScalar coords[], ego obj, const PetscScalar range[], const PetscInt v, const PetscInt dE, PetscScalar paramsV[], PetscBool islite)
{
  PetscInt    loopCntr = 0;
  PetscScalar dx, dy, dz, lambda, tolr, denom, obj_old, obj_tmp, target;
  PetscScalar uvs[2], uvt[2], delta[2], A[4], b[2], eval[18], data[18];

  PetscFunctionBeginHot;
  /* Initialize Levenberg-Marquardt parameters */
  lambda = 1.0;
  tolr   = 1.0;
  target = 1.0E-20;
  uvs[0] = (range[0] + range[1]) / 2.;
  uvs[1] = (range[2] + range[3]) / 2.;

  while (tolr >= target) {
    if (islite) {
      PetscCallEGADS(EGlite_evaluate, (obj, uvs, eval));
    } else {
      PetscCallEGADS(EG_evaluate, (obj, uvs, eval));
    }

    dx      = coords[v * dE + 0] - eval[0];
    dy      = coords[v * dE + 1] - eval[1];
    dz      = coords[v * dE + 2] - eval[2];
    obj_old = dx * dx + dy * dy + dz * dz;

    if (obj_old < target) {
      tolr = obj_old;
      break;
    }

    A[0] = (eval[3] * eval[3] + eval[4] * eval[4] + eval[5] * eval[5]) * (1.0 + lambda);
    A[1] = eval[3] * eval[6] + eval[4] * eval[7] + eval[5] * eval[8];
    A[2] = A[1];
    A[3] = (eval[6] * eval[6] + eval[7] * eval[7] + eval[8] * eval[8]) * (1.0 + lambda);

    b[0] = eval[3] * dx + eval[4] * dy + eval[5] * dz;
    b[1] = eval[6] * dx + eval[7] * dy + eval[8] * dz;

    /* Solve A*delta = b using Cramer's Rule */
    denom = A[0] * A[3] - A[2] * A[1];
    if (denom == 0.0) { PetscCall(PetscPrintf(PETSC_COMM_SELF, "denom = 0.0 \n")); }
    delta[0] = (b[0] * A[3] - b[1] * A[1]) / denom;
    delta[1] = (A[0] * b[1] - A[2] * b[0]) / denom;

    /* Find a temp (u,v) and associated objective function */
    uvt[0] = uvs[0] + delta[0];
    uvt[1] = uvs[1] + delta[1];

    if (uvt[0] < range[0]) {
      uvt[0]   = range[0];
      delta[0] = uvt[0] - uvs[0];
    }
    if (uvt[0] > range[1]) {
      uvt[0]   = range[1];
      delta[0] = uvt[0] - uvs[0];
    }
    if (uvt[1] < range[2]) {
      uvt[1]   = range[2];
      delta[1] = uvt[1] - uvs[1];
    }
    if (uvt[1] > range[3]) {
      uvt[1]   = range[3];
      delta[1] = uvt[1] - uvs[1];
    }

    if (islite) {
      PetscCall(EGlite_evaluate(obj, uvt, data));
    } else {
      PetscCall(EG_evaluate(obj, uvt, data));
    }

    obj_tmp = (coords[v * dE + 0] - data[0]) * (coords[v * dE + 0] - data[0]) + (coords[v * dE + 1] - data[1]) * (coords[v * dE + 1] - data[1]) + (coords[v * dE + 2] - data[2]) * (coords[v * dE + 2] - data[2]);

    /* If step is better, accept it and halve lambda (making it more Newton-like) */
    if (obj_tmp < obj_old) {
      obj_old = obj_tmp;
      uvs[0]  = uvt[0];
      uvs[1]  = uvt[1];
      for (int jj = 0; jj < 18; ++jj) eval[jj] = data[jj];
      lambda /= 2.0;
      if (lambda < 1.0E-14) lambda = 1.0E-14;
      if (obj_old < target) {
        tolr = obj_old;
        break;
      }
    } else {
      /* Otherwise reject it and double lambda (making it more gradient-descent like) */
      lambda *= 2.0;
    }

    if (sqrt(delta[0] * delta[0] + delta[1] * delta[1]) < target) {
      tolr = obj_old;
      break;
    }

    tolr = obj_old;

    loopCntr += 1;
    if (loopCntr > 100) break;
  }
  paramsV[v * 3 + 0] = uvs[0];
  paramsV[v * 3 + 1] = uvs[1];
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMSnapToGeomModel_EGADS_Internal(DM dm, PetscInt p, ego model, PetscInt bodyID, PetscInt faceID, PetscInt edgeID, const PetscScalar mcoords[], PetscScalar gcoords[], PetscBool islite)
{
  /* PETSc Variables */
  DM   cdm;
  ego *bodies;
  ego  geom, body, obj;
  /* result has to hold derivatives, along with the value */
  double       params[3], result[18], paramsV[16 * 3], range[4];
  int          Nb, oclass, mtype, *senses, peri;
  Vec          coordinatesLocal;
  PetscScalar *coords = NULL;
  PetscInt     Nv, v, Np = 0, pm;
  PetscInt     dE, d;
  PetscReal    pTolr = 1.0e-14;

  PetscFunctionBeginHot;
  PetscCall(DMGetCoordinateDM(dm, &cdm));
  PetscCall(DMGetCoordinateDim(dm, &dE));
  PetscCall(DMGetCoordinatesLocal(dm, &coordinatesLocal));

  if (islite) {
    PetscCall(EGlite_getTopology(model, &geom, &oclass, &mtype, NULL, &Nb, &bodies, &senses));
  } else {
    PetscCall(EG_getTopology(model, &geom, &oclass, &mtype, NULL, &Nb, &bodies, &senses));
  }

  PetscCheck(bodyID < Nb, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Body %" PetscInt_FMT " is not in [0, %" PetscInt_FMT ")", bodyID, Nb);
  body = bodies[bodyID];

  if (edgeID >= 0) {
    if (islite) {
      PetscCall(EGlite_objectBodyTopo(body, EDGE, edgeID, &obj));
      Np = 1;
    } else {
      PetscCall(EG_objectBodyTopo(body, EDGE, edgeID, &obj));
      Np = 1;
    }
  } else if (faceID >= 0) {
    if (islite) {
      PetscCall(EGlite_objectBodyTopo(body, FACE, faceID, &obj));
      Np = 2;
    } else {
      PetscCall(EG_objectBodyTopo(body, FACE, faceID, &obj));
      Np = 2;
    }
  } else {
    for (d = 0; d < dE; ++d) gcoords[d] = mcoords[d];
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* Calculate parameters (t or u,v) for vertices */
  PetscCall(DMPlexVecGetClosure(cdm, NULL, coordinatesLocal, p, &Nv, &coords));
  Nv /= dE;
  if (Nv == 1) {
    PetscCall(DMPlexVecRestoreClosure(cdm, NULL, coordinatesLocal, p, &Nv, &coords));
    for (d = 0; d < dE; ++d) gcoords[d] = mcoords[d];
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCheck(Nv <= 16, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Cannot handle %" PetscInt_FMT " coordinates associated to point %" PetscInt_FMT, Nv, p);

  /* Correct EGADS/EGADSlite 2pi bug when calculating nearest point on Periodic Surfaces */
  if (islite) {
    PetscCall(EGlite_getRange(obj, range, &peri));
  } else {
    PetscCall(EG_getRange(obj, range, &peri));
  }

  for (v = 0; v < Nv; ++v) {
    if (edgeID > 0) {
      PetscCall(DMPlex_Geom_EDGE_XYZtoUV_Internal(coords, obj, range, v, dE, paramsV, islite));
    } else {
      PetscCall(DMPlex_Geom_FACE_XYZtoUV_Internal(coords, obj, range, v, dE, paramsV, islite));
    }
  }
  PetscCall(DMPlexVecRestoreClosure(cdm, NULL, coordinatesLocal, p, &Nv, &coords));

  /* Calculate parameters (t or u,v) for new vertex at edge midpoint */
  for (pm = 0; pm < Np; ++pm) {
    params[pm] = 0.;
    for (v = 0; v < Nv; ++v) params[pm] += paramsV[v * 3 + pm];
    params[pm] /= Nv;
  }
  PetscCheck((params[0] + pTolr >= range[0]) || (params[0] - pTolr <= range[1]), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Point %" PetscInt_FMT " had bad interpolation", p);
  PetscCheck(Np < 2 || ((params[1] + pTolr >= range[2]) || (params[1] - pTolr <= range[3])), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Point %d had bad interpolation on v", p);

  /* Put coordinates for new vertex in result[] */
  if (islite) {
    PetscCall(EGlite_evaluate(obj, params, result));
  } else {
    PetscCall(EG_evaluate(obj, params, result));
  }

  for (d = 0; d < dE; ++d) gcoords[d] = result[d];
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

PetscErrorCode DMSnapToGeomModel_EGADS(DM dm, PetscInt p, PetscInt dE, const PetscScalar mcoords[], PetscScalar gcoords[])
{
  PetscFunctionBeginHot;
#ifdef PETSC_HAVE_EGADS
  DMLabel        bodyLabel, faceLabel, edgeLabel;
  PetscInt       bodyID, faceID, edgeID;
  PetscContainer modelObj;
  ego            model;
  PetscBool      islite = PETSC_FALSE;

  // FIXME: Change -dm_plex_refine_without_snap_to_geom to DM to shut off snapping
  PetscCall(DMGetLabel(dm, "EGADS Body ID", &bodyLabel));
  PetscCall(DMGetLabel(dm, "EGADS Face ID", &faceLabel));
  PetscCall(DMGetLabel(dm, "EGADS Edge ID", &edgeLabel));
  PetscCheck(bodyLabel && faceLabel && edgeLabel, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "EGADS meshes must have body, face, and edge labels defined");
  PetscCall(PetscObjectQuery((PetscObject)dm, "EGADS Model", (PetscObject *)&modelObj));
  if (!modelObj) {
    PetscCall(PetscObjectQuery((PetscObject)dm, "EGADSlite Model", (PetscObject *)&modelObj));
    islite = PETSC_TRUE;
  }
  PetscCheck(modelObj, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "EGADS mesh missing model object");

  PetscCall(PetscContainerGetPointer(modelObj, (void **)&model));
  PetscCall(DMLabelGetValue(bodyLabel, p, &bodyID));
  PetscCall(DMLabelGetValue(faceLabel, p, &faceID));
  PetscCall(DMLabelGetValue(edgeLabel, p, &edgeID));
  /* Allows for "Connective" Plex Edges present in models with multiple non-touching Entities */
  if (bodyID < 0) {
    for (PetscInt d = 0; d < dE; ++d) gcoords[d] = mcoords[d];
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(DMSnapToGeomModel_EGADS_Internal(dm, p, model, bodyID, faceID, edgeID, mcoords, gcoords, islite));
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if defined(PETSC_HAVE_EGADS)
PetscErrorCode DMPlexGeomPrintModel_Internal(ego model, PetscBool islite)
{
  /* PETSc Variables */
  ego geom, *bodies, *nobjs, *mobjs, *lobjs, *shobjs, *fobjs, *eobjs;
  int oclass, mtype, *senses, *shsenses, *fsenses, *lsenses, *esenses;
  int Nb, b;

  PetscFunctionBeginUser;
  /* test bodyTopo functions */
  if (islite) {
    PetscCall(EGlite_getTopology(model, &geom, &oclass, &mtype, NULL, &Nb, &bodies, &senses));
  } else {
    PetscCall(EG_getTopology(model, &geom, &oclass, &mtype, NULL, &Nb, &bodies, &senses));
  }
  PetscCall(PetscPrintf(PETSC_COMM_SELF, " Number of BODIES (nbodies): %" PetscInt_FMT " \n", Nb));

  for (b = 0; b < Nb; ++b) {
    ego body = bodies[b];
    int id, sh, Nsh, f, Nf, l, Nl, e, Ne, v, Nv;

    /* List Topology of Bodies */
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "\n"));
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "   BODY %d TOPOLOGY SUMMARY \n", b));

    /* Output Basic Model Topology */
    if (islite) {
      PetscCall(EGlite_getBodyTopos(body, NULL, SHELL, &Nsh, &shobjs));
    } else {
      PetscCall(EG_getBodyTopos(body, NULL, SHELL, &Nsh, &shobjs));
    }
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "      Number of SHELLS: %d \n", Nsh));

    if (islite) {
      PetscCall(EGlite_getBodyTopos(body, NULL, FACE, &Nf, &fobjs));
    } else {
      PetscCall(EG_getBodyTopos(body, NULL, FACE, &Nf, &fobjs));
    }
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "      Number of FACES: %d \n", Nf));

    if (islite) {
      PetscCall(EGlite_getBodyTopos(body, NULL, LOOP, &Nl, &lobjs));
    } else {
      PetscCall(EG_getBodyTopos(body, NULL, LOOP, &Nl, &lobjs));
    }
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "      Number of LOOPS: %d \n", Nl));

    if (islite) {
      PetscCall(EGlite_getBodyTopos(body, NULL, EDGE, &Ne, &eobjs));
    } else {
      PetscCall(EG_getBodyTopos(body, NULL, EDGE, &Ne, &eobjs));
    }
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "      Number of EDGES: %d \n", Ne));

    if (islite) {
      PetscCall(EGlite_getBodyTopos(body, NULL, NODE, &Nv, &nobjs));
    } else {
      PetscCall(EG_getBodyTopos(body, NULL, NODE, &Nv, &nobjs));
    }
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "      Number of NODES: %d \n", Nv));

    if (islite) {
      EGlite_free(shobjs);
      EGlite_free(fobjs);
      EGlite_free(lobjs);
      EGlite_free(eobjs);
      EGlite_free(nobjs);
    } else {
      EG_free(shobjs);
      EG_free(fobjs);
      EG_free(lobjs);
      EG_free(eobjs);
      EG_free(nobjs);
    }

    /* List Topology of Bodies */
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "\n"));
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "      BODY %d TOPOLOGY DETAILS \n", b));

    /* Get SHELL info which associated with the current BODY */
    if (islite) {
      PetscCall(EGlite_getTopology(body, &geom, &oclass, &mtype, NULL, &Nsh, &shobjs, &shsenses));
    } else {
      PetscCall(EG_getTopology(body, &geom, &oclass, &mtype, NULL, &Nsh, &shobjs, &shsenses));
    }

    for (sh = 0; sh < Nsh; ++sh) {
      ego shell   = shobjs[sh];
      int shsense = shsenses[sh];

      if (islite) {
        id = EGlite_indexBodyTopo(body, shell);
      } else {
        id = EG_indexBodyTopo(body, shell);
      }
      PetscCall(PetscPrintf(PETSC_COMM_SELF, "         SHELL ID: %d :: sense = %d\n", id, shsense));

      /* Get FACE information associated with current SHELL */
      if (islite) {
        PetscCall(EGlite_getTopology(shell, &geom, &oclass, &mtype, NULL, &Nf, &fobjs, &fsenses));
      } else {
        PetscCall(EG_getTopology(shell, &geom, &oclass, &mtype, NULL, &Nf, &fobjs, &fsenses));
      }

      for (f = 0; f < Nf; ++f) {
        ego     face = fobjs[f];
        ego     gRef, gPrev, gNext;
        int     goclass, gmtype, *gpinfo;
        double *gprv;
        char   *gClass = (char *)"", *gType = (char *)"";
        double  fdata[4];
        ego     fRef, fPrev, fNext;
        int     foclass, fmtype;

        if (islite) {
          id = EGlite_indexBodyTopo(body, face);
        } else {
          id = EG_indexBodyTopo(body, face);
        }

        /* Get LOOP info associated with current FACE */
        if (islite) {
          PetscCall(EGlite_getTopology(face, &geom, &oclass, &mtype, fdata, &Nl, &lobjs, &lsenses));
          PetscCall(EGlite_getInfo(face, &foclass, &fmtype, &fRef, &fPrev, &fNext));
          PetscCall(EGlite_getGeometry(geom, &goclass, &gmtype, &gRef, &gpinfo, &gprv));
          PetscCall(EGlite_getInfo(geom, &goclass, &gmtype, &gRef, &gPrev, &gNext));
        } else {
          PetscCall(EG_getTopology(face, &geom, &oclass, &mtype, fdata, &Nl, &lobjs, &lsenses));
          PetscCall(EG_getInfo(face, &foclass, &fmtype, &fRef, &fPrev, &fNext));
          PetscCall(EG_getGeometry(geom, &goclass, &gmtype, &gRef, &gpinfo, &gprv));
          PetscCall(EG_getInfo(geom, &goclass, &gmtype, &gRef, &gPrev, &gNext));
        }

        PetscCall(DMPlex_EGADS_GeomDecode_Internal(goclass, gmtype, &gClass, &gType));
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "           FACE ID: %d :: sense = %d \n", id, fmtype));
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "             GEOMETRY CLASS: %s \n", gClass));
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "             GEOMETRY TYPE:  %s \n\n", gType));
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "             RANGE (umin, umax) = (%f, %f) \n", fdata[0], fdata[1]));
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "                   (vmin, vmax) = (%f, %f) \n\n", fdata[2], fdata[3]));

        for (l = 0; l < Nl; ++l) {
          ego loop   = lobjs[l];
          int lsense = lsenses[l];

          if (islite) {
            id = EGlite_indexBodyTopo(body, loop);
          } else {
            id = EG_indexBodyTopo(body, loop);
          }

          PetscCall(PetscPrintf(PETSC_COMM_SELF, "             LOOP ID: %d :: sense = %d\n", id, lsense));

          /* Get EDGE info associated with the current LOOP */
          if (islite) {
            PetscCall(EGlite_getTopology(loop, &geom, &oclass, &mtype, NULL, &Ne, &eobjs, &esenses));
          } else {
            PetscCall(EG_getTopology(loop, &geom, &oclass, &mtype, NULL, &Ne, &eobjs, &esenses));
          }

          for (e = 0; e < Ne; ++e) {
            ego    edge = eobjs[e];
            ego    topRef, prev, next;
            int    esense   = esenses[e];
            double range[4] = {0., 0., 0., 0.};
            int    peri;
            ego    gEref, gEprev, gEnext;
            int    gEoclass, gEmtype;
            char  *gEclass = (char *)"", *gEtype = (char *)"";

            if (islite) {
              PetscCall(EGlite_getTopology(edge, &geom, &oclass, &mtype, NULL, &Nv, &nobjs, &senses));
              if (mtype != DEGENERATE) { PetscCall(EGlite_getInfo(geom, &gEoclass, &gEmtype, &gEref, &gEprev, &gEnext)); }
            } else {
              PetscCall(EG_getTopology(edge, &geom, &oclass, &mtype, NULL, &Nv, &nobjs, &senses));
              PetscCall(EG_getInfo(geom, &gEoclass, &gEmtype, &gEref, &gEprev, &gEnext));
            }

            if (mtype != DEGENERATE) { PetscCall(DMPlex_EGADS_GeomDecode_Internal(gEoclass, gEmtype, &gEclass, &gEtype)); }

            if (islite) {
              id = EGlite_indexBodyTopo(body, edge);
              PetscCall(EGlite_getInfo(edge, &oclass, &mtype, &topRef, &prev, &next));
            } else {
              id = EG_indexBodyTopo(body, edge);
              PetscCall(EG_getInfo(edge, &oclass, &mtype, &topRef, &prev, &next));
            }

            PetscCall(PetscPrintf(PETSC_COMM_SELF, "               EDGE ID: %d :: sense = %d\n", id, esense));
            if (mtype != DEGENERATE) {
              PetscCall(PetscPrintf(PETSC_COMM_SELF, "                 GEOMETRY CLASS: %s \n", gEclass));
              PetscCall(PetscPrintf(PETSC_COMM_SELF, "                 GEOMETRY TYPE:  %s \n", gEtype));
            }

            if (mtype == DEGENERATE) { PetscCall(PetscPrintf(PETSC_COMM_SELF, "                 EDGE %d is DEGENERATE \n", id)); }

            if (islite) {
              PetscCall(EGlite_getRange(edge, range, &peri));
            } else {
              PetscCall(EG_getRange(edge, range, &peri));
            }

            PetscCall(PetscPrintf(PETSC_COMM_SELF, "                 Peri = %d :: Range = %lf, %lf, %lf, %lf \n", peri, range[0], range[1], range[2], range[3]));

            /* Get NODE info associated with the current EDGE */
            if (islite) {
              PetscCall(EGlite_getTopology(edge, &geom, &oclass, &mtype, NULL, &Nv, &nobjs, &senses));
            } else {
              PetscCall(EG_getTopology(edge, &geom, &oclass, &mtype, NULL, &Nv, &nobjs, &senses));
            }

            for (v = 0; v < Nv; ++v) {
              ego    vertex = nobjs[v];
              double limits[4];
              int    dummy;

              if (islite) {
                PetscCall(EGlite_getTopology(vertex, &geom, &oclass, &mtype, limits, &dummy, &mobjs, &senses));
                id = EGlite_indexBodyTopo(body, vertex);
              } else {
                PetscCall(EG_getTopology(vertex, &geom, &oclass, &mtype, limits, &dummy, &mobjs, &senses));
                id = EG_indexBodyTopo(body, vertex);
              }
              PetscCall(PetscPrintf(PETSC_COMM_SELF, "                 NODE ID: %d \n", id));
              PetscCall(PetscPrintf(PETSC_COMM_SELF, "                    (x, y, z) = (%lf, %lf, %lf) \n", limits[0], limits[1], limits[2]));
            }
          }
        }
      }
    }
  }
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "\n\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexEGADSDestroy_Private(void **context)
{
  if (*context) EG_deleteObject((ego)*context);
  return (PETSC_SUCCESS);
}

static PetscErrorCode DMPlexEGADSClose_Private(void **context)
{
  if (*context) EG_close((ego)*context);
  return (PETSC_SUCCESS);
}

PetscErrorCode DMPlexEGADSliteDestroy_Private(void **context)
{
  if (*context) EGlite_deleteObject((ego)*context);
  return 0;
}

PetscErrorCode DMPlexEGADSliteClose_Private(void **context)
{
  if (*context) EGlite_close((ego)*context);
  return 0;
}

PetscErrorCode DMPlexCreateGeom_Internal(MPI_Comm comm, ego context, ego model, DM *newdm, PetscBool islite)
{
  /* EGADS variables */
  ego geom, *bodies, *objs, *nobjs, *mobjs, *lobjs;
  int oclass, mtype, nbodies, *senses;
  int b;
  /* PETSc variables */
  DM          dm;
  DMLabel     bodyLabel, faceLabel, edgeLabel, vertexLabel;
  PetscHMapI  edgeMap = NULL;
  PetscInt    cStart, cEnd, c;
  PetscInt    dim = -1, cdim = -1, numCorners = 0, maxCorners = 0, numVertices = 0, newVertices = 0, numEdges = 0, numCells = 0, newCells = 0, numQuads = 0, cOff = 0, fOff = 0;
  PetscInt   *cells = NULL, *cone = NULL;
  PetscReal  *coords = NULL;
  PetscMPIInt rank;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  if (rank == 0) {
    const PetscInt debug = 0;

    /* ---------------------------------------------------------------------------------------------------
    Generate PETSc DMPlex
      Get all Nodes in model, record coordinates in a correctly formatted array
      Cycle through bodies, cycle through loops, recorde NODE IDs in a correctly formatted array
      We need to uniformly refine the initial geometry to guarantee a valid mesh
    */

    /* Calculate cell and vertex sizes */
    if (islite) {
      PetscCall(EGlite_getTopology(model, &geom, &oclass, &mtype, NULL, &nbodies, &bodies, &senses));
    } else {
      PetscCall(EG_getTopology(model, &geom, &oclass, &mtype, NULL, &nbodies, &bodies, &senses));
    }

    PetscCall(PetscHMapICreate(&edgeMap));
    numEdges = 0;
    for (b = 0; b < nbodies; ++b) {
      ego body = bodies[b];
      int id, Nl, l, Nv, v;

      if (islite) {
        PetscCall(EGlite_getBodyTopos(body, NULL, LOOP, &Nl, &lobjs));
      } else {
        PetscCall(EG_getBodyTopos(body, NULL, LOOP, &Nl, &lobjs));
      }

      for (l = 0; l < Nl; ++l) {
        ego loop = lobjs[l];
        int Ner  = 0, Ne, e, Nc;

        if (islite) {
          PetscCall(EGlite_getTopology(loop, &geom, &oclass, &mtype, NULL, &Ne, &objs, &senses));
        } else {
          PetscCall(EG_getTopology(loop, &geom, &oclass, &mtype, NULL, &Ne, &objs, &senses));
        }

        for (e = 0; e < Ne; ++e) {
          ego           edge = objs[e];
          int           Nv, id;
          PetscHashIter iter;
          PetscBool     found;

          if (islite) {
            PetscCall(EGlite_getTopology(edge, &geom, &oclass, &mtype, NULL, &Nv, &nobjs, &senses));
          } else {
            PetscCall(EG_getTopology(edge, &geom, &oclass, &mtype, NULL, &Nv, &nobjs, &senses));
          }

          if (mtype == DEGENERATE) continue;

          if (islite) {
            id = EGlite_indexBodyTopo(body, edge);
          } else {
            id = EG_indexBodyTopo(body, edge);
          }

          PetscCall(PetscHMapIFind(edgeMap, id - 1, &iter, &found));
          if (!found) { PetscCall(PetscHMapISet(edgeMap, id - 1, numEdges++)); }
          ++Ner;
        }
        if (Ner == 2) {
          Nc = 2;
        } else if (Ner == 3) {
          Nc = 4;
        } else if (Ner == 4) {
          Nc = 8;
          ++numQuads;
        } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Cannot support loop with %d edges", Ner);
        numCells += Nc;
        newCells += Nc - 1;
        maxCorners = PetscMax(Ner * 2 + 1, maxCorners);
      }
      if (islite) {
        PetscCall(EGlite_getBodyTopos(body, NULL, NODE, &Nv, &nobjs));
      } else {
        PetscCall(EG_getBodyTopos(body, NULL, NODE, &Nv, &nobjs));
      }

      for (v = 0; v < Nv; ++v) {
        ego vertex = nobjs[v];

        if (islite) {
          id = EGlite_indexBodyTopo(body, vertex);
        } else {
          id = EG_indexBodyTopo(body, vertex);
        }
        /* TODO: Instead of assuming contiguous ids, we could use a hash table */
        numVertices = PetscMax(id, numVertices);
      }
      if (islite) {
        EGlite_free(lobjs);
        EGlite_free(nobjs);
      } else {
        EG_free(lobjs);
        EG_free(nobjs);
      }
    }
    PetscCall(PetscHMapIGetSize(edgeMap, &numEdges));
    newVertices = numEdges + numQuads;
    numVertices += newVertices;

    dim        = 2; /* Assume 3D Models :: Need to update to handle 2D Models in the future */
    cdim       = 3; /* Assume 3D Models :: Need to update to handle 2D Models in the future */
    numCorners = 3; /* Split cells into triangles */
    PetscCall(PetscMalloc3(numVertices * cdim, &coords, numCells * numCorners, &cells, maxCorners, &cone));

    /* Get vertex coordinates */
    for (b = 0; b < nbodies; ++b) {
      ego body = bodies[b];
      int id, Nv, v;

      if (islite) {
        PetscCall(EGlite_getBodyTopos(body, NULL, NODE, &Nv, &nobjs));
      } else {
        PetscCall(EG_getBodyTopos(body, NULL, NODE, &Nv, &nobjs));
      }

      for (v = 0; v < Nv; ++v) {
        ego    vertex = nobjs[v];
        double limits[4];
        int    dummy;

        if (islite) {
          PetscCall(EGlite_getTopology(vertex, &geom, &oclass, &mtype, limits, &dummy, &mobjs, &senses));
          id = EGlite_indexBodyTopo(body, vertex);
        } else {
          PetscCall(EG_getTopology(vertex, &geom, &oclass, &mtype, limits, &dummy, &mobjs, &senses));
          id = EG_indexBodyTopo(body, vertex);
        }

        coords[(id - 1) * cdim + 0] = limits[0];
        coords[(id - 1) * cdim + 1] = limits[1];
        coords[(id - 1) * cdim + 2] = limits[2];
      }
      if (islite) {
        EGlite_free(nobjs);
      } else {
        EG_free(nobjs);
      }
    }
    PetscCall(PetscHMapIClear(edgeMap));
    fOff     = numVertices - newVertices + numEdges;
    numEdges = 0;
    numQuads = 0;
    for (b = 0; b < nbodies; ++b) {
      ego body = bodies[b];
      int Nl, l;

      if (islite) {
        PetscCall(EGlite_getBodyTopos(body, NULL, LOOP, &Nl, &lobjs));
      } else {
        PetscCall(EG_getBodyTopos(body, NULL, LOOP, &Nl, &lobjs));
      }

      for (l = 0; l < Nl; ++l) {
        ego loop = lobjs[l];
        int lid, Ner = 0, Ne, e;

        if (islite) {
          lid = EGlite_indexBodyTopo(body, loop);
          PetscCall(EGlite_getTopology(loop, &geom, &oclass, &mtype, NULL, &Ne, &objs, &senses));
        } else {
          lid = EG_indexBodyTopo(body, loop);
          PetscCall(EG_getTopology(loop, &geom, &oclass, &mtype, NULL, &Ne, &objs, &senses));
        }

        for (e = 0; e < Ne; ++e) {
          ego           edge = objs[e];
          int           eid, Nv;
          PetscHashIter iter;
          PetscBool     found;

          if (islite) {
            PetscCall(EGlite_getTopology(edge, &geom, &oclass, &mtype, NULL, &Nv, &nobjs, &senses));
          } else {
            PetscCall(EG_getTopology(edge, &geom, &oclass, &mtype, NULL, &Nv, &nobjs, &senses));
          }

          if (mtype == DEGENERATE) continue;
          ++Ner;

          if (islite) {
            eid = EGlite_indexBodyTopo(body, edge);
          } else {
            eid = EG_indexBodyTopo(body, edge);
          }

          PetscCall(PetscHMapIFind(edgeMap, eid - 1, &iter, &found));
          if (!found) {
            PetscInt v = numVertices - newVertices + numEdges;
            double   range[4], params[3] = {0., 0., 0.}, result[18];
            int      periodic[2];

            PetscCall(PetscHMapISet(edgeMap, eid - 1, numEdges++));

            if (islite) {
              PetscCall(EGlite_getRange(edge, range, periodic));
            } else {
              PetscCall(EG_getRange(edge, range, periodic));
            }

            params[0] = 0.5 * (range[0] + range[1]);
            if (islite) {
              PetscCall(EGlite_evaluate(edge, params, result));
            } else {
              PetscCall(EG_evaluate(edge, params, result));
            }
            coords[v * cdim + 0] = result[0];
            coords[v * cdim + 1] = result[1];
            coords[v * cdim + 2] = result[2];
          }
        }
        if (Ner == 4) {
          PetscInt v = fOff + numQuads++;
          ego     *fobjs, face;
          double   range[4], params[3] = {0., 0., 0.}, result[18];
          int      Nf, fid, periodic[2];

          if (islite) {
            PetscCall(EGlite_getBodyTopos(body, loop, FACE, &Nf, &fobjs));
          } else {
            PetscCall(EG_getBodyTopos(body, loop, FACE, &Nf, &fobjs));
          }
          face = fobjs[0];

          if (islite) {
            fid = EGlite_indexBodyTopo(body, face);
          } else {
            fid = EG_indexBodyTopo(body, face);
          }

          PetscCheck(Nf != 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Loop %d has %" PetscInt_FMT " faces, instead of 1 (%" PetscInt_FMT ")", lid - 1, Nf, fid);
          if (islite) {
            PetscCall(EGlite_getRange(face, range, periodic));
          } else {
            PetscCall(EG_getRange(face, range, periodic));
          }
          params[0] = 0.5 * (range[0] + range[1]);
          params[1] = 0.5 * (range[2] + range[3]);
          if (islite) {
            PetscCall(EGlite_evaluate(face, params, result));
          } else {
            PetscCall(EG_evaluate(face, params, result));
          }
          coords[v * cdim + 0] = result[0];
          coords[v * cdim + 1] = result[1];
          coords[v * cdim + 2] = result[2];
        }
      }
    }
    PetscCheck(numEdges + numQuads == newVertices, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Number of new vertices %d != %d previous count", numEdges + numQuads, newVertices);

    /* Get cell vertices by traversing loops */
    numQuads = 0;
    cOff     = 0;
    for (b = 0; b < nbodies; ++b) {
      ego body = bodies[b];
      int id, Nl, l;

      if (islite) {
        PetscCall(EGlite_getBodyTopos(body, NULL, LOOP, &Nl, &lobjs));
      } else {
        PetscCall(EG_getBodyTopos(body, NULL, LOOP, &Nl, &lobjs));
      }
      for (l = 0; l < Nl; ++l) {
        ego loop = lobjs[l];
        int lid, Ner = 0, Ne, e, nc = 0, c, Nt, t;

        if (islite) {
          lid = EGlite_indexBodyTopo(body, loop);
          PetscCall(EGlite_getTopology(loop, &geom, &oclass, &mtype, NULL, &Ne, &objs, &senses));
        } else {
          lid = EG_indexBodyTopo(body, loop);
          PetscCall(EG_getTopology(loop, &geom, &oclass, &mtype, NULL, &Ne, &objs, &senses));
        }

        for (e = 0; e < Ne; ++e) {
          ego edge = objs[e];
          int points[3];
          int eid, Nv, v, tmp;

          if (islite) {
            eid = EGlite_indexBodyTopo(body, edge);
            PetscCall(EGlite_getTopology(edge, &geom, &oclass, &mtype, NULL, &Nv, &nobjs, &senses));
          } else {
            eid = EG_indexBodyTopo(body, edge);
            PetscCall(EG_getTopology(edge, &geom, &oclass, &mtype, NULL, &Nv, &nobjs, &senses));
          }

          if (mtype == DEGENERATE) continue;
          else ++Ner;
          PetscCheck(Nv == 2, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Edge %" PetscInt_FMT " has %" PetscInt_FMT " vertices != 2", eid, Nv);

          for (v = 0; v < Nv; ++v) {
            ego vertex = nobjs[v];

            if (islite) {
              id = EGlite_indexBodyTopo(body, vertex);
            } else {
              id = EG_indexBodyTopo(body, vertex);
            }
            points[v * 2] = id - 1;
          }
          {
            PetscInt edgeNum;

            PetscCall(PetscHMapIGet(edgeMap, eid - 1, &edgeNum));
            points[1] = numVertices - newVertices + edgeNum;
          }
          /* EGADS loops are not oriented, but seem to be in order, so we must piece them together */
          if (!nc) {
            for (v = 0; v < Nv + 1; ++v) cone[nc++] = points[v];
          } else {
            if (cone[nc - 1] == points[0]) {
              cone[nc++] = points[1];
              if (cone[0] != points[2]) cone[nc++] = points[2];
            } else if (cone[nc - 1] == points[2]) {
              cone[nc++] = points[1];
              if (cone[0] != points[0]) cone[nc++] = points[0];
            } else if (cone[nc - 3] == points[0]) {
              tmp          = cone[nc - 3];
              cone[nc - 3] = cone[nc - 1];
              cone[nc - 1] = tmp;
              cone[nc++]   = points[1];
              if (cone[0] != points[2]) cone[nc++] = points[2];
            } else if (cone[nc - 3] == points[2]) {
              tmp          = cone[nc - 3];
              cone[nc - 3] = cone[nc - 1];
              cone[nc - 1] = tmp;
              cone[nc++]   = points[1];
              if (cone[0] != points[0]) cone[nc++] = points[0];
            } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Edge %d does not match its predecessor", eid);
          }
        }
        PetscCheck(nc == 2 * Ner, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Number of corners %" PetscInt_FMT " != %" PetscInt_FMT, nc, 2 * Ner);
        if (Ner == 4) { cone[nc++] = numVertices - newVertices + numEdges + numQuads++; }
        PetscCheck(nc <= maxCorners, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Number of corners %" PetscInt_FMT " > %" PetscInt_FMT " max", nc, maxCorners);
        /* Triangulate the loop */
        switch (Ner) {
        case 2: /* Bi-Segment -> 2 triangles */
          Nt                           = 2;
          cells[cOff * numCorners + 0] = cone[0];
          cells[cOff * numCorners + 1] = cone[1];
          cells[cOff * numCorners + 2] = cone[2];
          ++cOff;
          cells[cOff * numCorners + 0] = cone[0];
          cells[cOff * numCorners + 1] = cone[2];
          cells[cOff * numCorners + 2] = cone[3];
          ++cOff;
          break;
        case 3: /* Triangle   -> 4 triangles */
          Nt                           = 4;
          cells[cOff * numCorners + 0] = cone[0];
          cells[cOff * numCorners + 1] = cone[1];
          cells[cOff * numCorners + 2] = cone[5];
          ++cOff;
          cells[cOff * numCorners + 0] = cone[1];
          cells[cOff * numCorners + 1] = cone[2];
          cells[cOff * numCorners + 2] = cone[3];
          ++cOff;
          cells[cOff * numCorners + 0] = cone[5];
          cells[cOff * numCorners + 1] = cone[3];
          cells[cOff * numCorners + 2] = cone[4];
          ++cOff;
          cells[cOff * numCorners + 0] = cone[1];
          cells[cOff * numCorners + 1] = cone[3];
          cells[cOff * numCorners + 2] = cone[5];
          ++cOff;
          break;
        case 4: /* Quad       -> 8 triangles */
          Nt                           = 8;
          cells[cOff * numCorners + 0] = cone[0];
          cells[cOff * numCorners + 1] = cone[1];
          cells[cOff * numCorners + 2] = cone[7];
          ++cOff;
          cells[cOff * numCorners + 0] = cone[1];
          cells[cOff * numCorners + 1] = cone[2];
          cells[cOff * numCorners + 2] = cone[3];
          ++cOff;
          cells[cOff * numCorners + 0] = cone[3];
          cells[cOff * numCorners + 1] = cone[4];
          cells[cOff * numCorners + 2] = cone[5];
          ++cOff;
          cells[cOff * numCorners + 0] = cone[5];
          cells[cOff * numCorners + 1] = cone[6];
          cells[cOff * numCorners + 2] = cone[7];
          ++cOff;
          cells[cOff * numCorners + 0] = cone[8];
          cells[cOff * numCorners + 1] = cone[1];
          cells[cOff * numCorners + 2] = cone[3];
          ++cOff;
          cells[cOff * numCorners + 0] = cone[8];
          cells[cOff * numCorners + 1] = cone[3];
          cells[cOff * numCorners + 2] = cone[5];
          ++cOff;
          cells[cOff * numCorners + 0] = cone[8];
          cells[cOff * numCorners + 1] = cone[5];
          cells[cOff * numCorners + 2] = cone[7];
          ++cOff;
          cells[cOff * numCorners + 0] = cone[8];
          cells[cOff * numCorners + 1] = cone[7];
          cells[cOff * numCorners + 2] = cone[1];
          ++cOff;
          break;
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Loop %d has %d edges, which we do not support", lid, Ner);
        }
        if (debug) {
          for (t = 0; t < Nt; ++t) {
            PetscCall(PetscPrintf(PETSC_COMM_SELF, "  LOOP Corner NODEs Triangle %d (", t));
            for (c = 0; c < numCorners; ++c) {
              if (c > 0) { PetscCall(PetscPrintf(PETSC_COMM_SELF, ", ")); }
              PetscCall(PetscPrintf(PETSC_COMM_SELF, "%d", cells[(cOff - Nt + t) * numCorners + c]));
            }
            PetscCall(PetscPrintf(PETSC_COMM_SELF, ")\n"));
          }
        }
      }
      if (islite) {
        EGlite_free(lobjs);
      } else {
        EG_free(lobjs);
      }
    }
  }
  PetscCheck(cOff == numCells, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Count of total cells %d != %d previous count", cOff, numCells);
  PetscCall(DMPlexCreateFromCellListPetsc(PETSC_COMM_WORLD, dim, numCells, numVertices, numCorners, PETSC_TRUE, cells, cdim, coords, &dm));
  PetscCall(PetscFree3(coords, cells, cone));
  PetscCall(PetscInfo(dm, " Total Number of Unique Cells    = %d (%d)\n", numCells, newCells));
  PetscCall(PetscInfo(dm, " Total Number of Unique Vertices = %d (%d)\n", numVertices, newVertices));
  /* Embed EGADS model in DM */
  {
    PetscContainer modelObj, contextObj;

    PetscCall(PetscContainerCreate(PETSC_COMM_SELF, &modelObj));
    PetscCall(PetscContainerSetPointer(modelObj, model));
    PetscCall(PetscContainerSetCtxDestroy(modelObj, (PetscCtxDestroyFn *)DMPlexEGADSDestroy_Private));
    PetscCall(PetscObjectCompose((PetscObject)dm, "EGADS Model", (PetscObject)modelObj));
    PetscCall(PetscContainerDestroy(&modelObj));

    PetscCall(PetscContainerCreate(PETSC_COMM_SELF, &contextObj));
    PetscCall(PetscContainerSetPointer(contextObj, context));
    PetscCall(PetscContainerSetCtxDestroy(contextObj, (PetscCtxDestroyFn *)DMPlexEGADSClose_Private));
    PetscCall(PetscObjectCompose((PetscObject)dm, "EGADS Context", (PetscObject)contextObj));
    PetscCall(PetscContainerDestroy(&contextObj));
  }
  /* Label points */
  PetscCall(DMCreateLabel(dm, "EGADS Body ID"));
  PetscCall(DMGetLabel(dm, "EGADS Body ID", &bodyLabel));
  PetscCall(DMCreateLabel(dm, "EGADS Face ID"));
  PetscCall(DMGetLabel(dm, "EGADS Face ID", &faceLabel));
  PetscCall(DMCreateLabel(dm, "EGADS Edge ID"));
  PetscCall(DMGetLabel(dm, "EGADS Edge ID", &edgeLabel));
  PetscCall(DMCreateLabel(dm, "EGADS Vertex ID"));
  PetscCall(DMGetLabel(dm, "EGADS Vertex ID", &vertexLabel));
  cOff = 0;
  for (b = 0; b < nbodies; ++b) {
    ego body = bodies[b];
    int id, Nl, l;

    if (islite) {
      PetscCall(EGlite_getBodyTopos(body, NULL, LOOP, &Nl, &lobjs));
    } else {
      PetscCall(EG_getBodyTopos(body, NULL, LOOP, &Nl, &lobjs));
    }
    for (l = 0; l < Nl; ++l) {
      ego  loop = lobjs[l];
      ego *fobjs;
      int  lid, Nf, fid, Ner = 0, Ne, e, Nt = 0, t;

      if (islite) {
        lid = EGlite_indexBodyTopo(body, loop);
        PetscCall(EGlite_getBodyTopos(body, loop, FACE, &Nf, &fobjs));
      } else {
        lid = EG_indexBodyTopo(body, loop);
        PetscCall(EG_getBodyTopos(body, loop, FACE, &Nf, &fobjs));
      }

      PetscCheck(Nf <= 1, PETSC_COMM_SELF, PETSC_ERR_SUP, "Loop %d has %d > 1 faces, which is not supported", lid, Nf);
      if (islite) {
        fid = EGlite_indexBodyTopo(body, fobjs[0]);
        EGlite_free(fobjs);
        PetscCall(EGlite_getTopology(loop, &geom, &oclass, &mtype, NULL, &Ne, &objs, &senses));
      } else {
        fid = EG_indexBodyTopo(body, fobjs[0]);
        EG_free(fobjs);
        PetscCall(EG_getTopology(loop, &geom, &oclass, &mtype, NULL, &Ne, &objs, &senses));
      }

      for (e = 0; e < Ne; ++e) {
        ego             edge = objs[e];
        int             eid, Nv, v;
        PetscInt        points[3], support[2], numEdges, edgeNum;
        const PetscInt *edges;

        if (islite) {
          eid = EGlite_indexBodyTopo(body, edge);
          PetscCall(EGlite_getTopology(edge, &geom, &oclass, &mtype, NULL, &Nv, &nobjs, &senses));
        } else {
          eid = EG_indexBodyTopo(body, edge);
          PetscCall(EG_getTopology(edge, &geom, &oclass, &mtype, NULL, &Nv, &nobjs, &senses));
        }

        if (mtype == DEGENERATE) continue;
        else ++Ner;
        for (v = 0; v < Nv; ++v) {
          ego vertex = nobjs[v];

          if (islite) {
            id = EGlite_indexBodyTopo(body, vertex);
          } else {
            id = EG_indexBodyTopo(body, vertex);
          }

          PetscCall(DMLabelSetValue(edgeLabel, numCells + id - 1, eid));
          points[v * 2] = numCells + id - 1;
        }
        PetscCall(PetscHMapIGet(edgeMap, eid - 1, &edgeNum));
        points[1] = numCells + numVertices - newVertices + edgeNum;

        PetscCall(DMLabelSetValue(edgeLabel, points[1], eid));
        support[0] = points[0];
        support[1] = points[1];
        PetscCall(DMPlexGetJoin(dm, 2, support, &numEdges, &edges));
        PetscCheck(numEdges == 1, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Vertices (%d, %d) should only bound 1 edge, not %d", support[0], support[1], numEdges);
        PetscCall(DMLabelSetValue(edgeLabel, edges[0], eid));
        PetscCall(DMPlexRestoreJoin(dm, 2, support, &numEdges, &edges));
        support[0] = points[1];
        support[1] = points[2];
        PetscCall(DMPlexGetJoin(dm, 2, support, &numEdges, &edges));
        PetscCheck(numEdges == 1, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Vertices (%d, %d) should only bound 1 edge, not %d", support[0], support[1], numEdges);
        PetscCall(DMLabelSetValue(edgeLabel, edges[0], eid));
        PetscCall(DMPlexRestoreJoin(dm, 2, support, &numEdges, &edges));
      }
      switch (Ner) {
      case 2:
        Nt = 2;
        break;
      case 3:
        Nt = 4;
        break;
      case 4:
        Nt = 8;
        break;
      default:
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Loop with %d edges is unsupported", Ner);
      }
      for (t = 0; t < Nt; ++t) {
        PetscCall(DMLabelSetValue(bodyLabel, cOff + t, b));
        PetscCall(DMLabelSetValue(faceLabel, cOff + t, fid));
      }
      cOff += Nt;
    }
    if (islite) {
      EGlite_free(lobjs);
    } else {
      EG_free(lobjs);
    }
  }
  PetscCall(PetscHMapIDestroy(&edgeMap));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  for (c = cStart; c < cEnd; ++c) {
    PetscInt *closure = NULL;
    PetscInt  clSize, cl, bval, fval;

    PetscCall(DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &clSize, &closure));
    PetscCall(DMLabelGetValue(bodyLabel, c, &bval));
    PetscCall(DMLabelGetValue(faceLabel, c, &fval));
    for (cl = 0; cl < clSize * 2; cl += 2) {
      PetscCall(DMLabelSetValue(bodyLabel, closure[cl], bval));
      PetscCall(DMLabelSetValue(faceLabel, closure[cl], fval));
    }
    PetscCall(DMPlexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &clSize, &closure));
  }
  *newdm = dm;
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexCreateGeom(MPI_Comm comm, ego context, ego model, DM *newdm, PetscBool islite)
{
  // EGADS variables
  ego geom, *bodies, *mobjs, *fobjs, *lobjs, *eobjs, *nobjs;
  ego topRef, prev, next;
  int oclass, mtype, nbodies, *senses, *lSenses, *eSenses;
  int b;
  // PETSc variables
  DM              dm;
  DMLabel         bodyLabel, faceLabel, edgeLabel, vertexLabel;
  PetscHMapI      edgeMap = NULL, bodyIndexMap = NULL, bodyVertexMap = NULL, bodyEdgeMap = NULL, bodyFaceMap = NULL, bodyEdgeGlobalMap = NULL;
  PetscInt        dim = -1, cdim = -1, numCorners = 0, numVertices = 0, numEdges = 0, numFaces = 0, numCells = 0, edgeCntr = 0;
  PetscInt        cellCntr = 0, numPoints = 0;
  PetscInt       *cells  = NULL;
  const PetscInt *cone   = NULL;
  PetscReal      *coords = NULL;
  PetscMPIInt     rank;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  if (rank == 0) {
    // ---------------------------------------------------------------------------------------------------
    // Generate PETSc DMPlex
    //  Get all Nodes in model, record coordinates in a correctly formatted array
    //  Cycle through bodies, cycle through loops, recorde NODE IDs in a correctly formatted array
    //  We need to uniformly refine the initial geometry to guarantee a valid mesh

    // Calculate cell and vertex sizes
    if (islite) {
      PetscCall(EGlite_getTopology(model, &geom, &oclass, &mtype, NULL, &nbodies, &bodies, &senses));
    } else {
      PetscCall(EG_getTopology(model, &geom, &oclass, &mtype, NULL, &nbodies, &bodies, &senses));
    }
    PetscCall(PetscHMapICreate(&edgeMap));
    PetscCall(PetscHMapICreate(&bodyIndexMap));
    PetscCall(PetscHMapICreate(&bodyVertexMap));
    PetscCall(PetscHMapICreate(&bodyEdgeMap));
    PetscCall(PetscHMapICreate(&bodyEdgeGlobalMap));
    PetscCall(PetscHMapICreate(&bodyFaceMap));

    for (b = 0; b < nbodies; ++b) {
      ego           body = bodies[b];
      int           Nf, Ne, Nv;
      PetscHashIter BIiter, BViter, BEiter, BEGiter, BFiter, EMiter;
      PetscBool     BIfound, BVfound, BEfound, BEGfound, BFfound, EMfound;

      PetscCall(PetscHMapIFind(bodyIndexMap, b, &BIiter, &BIfound));
      PetscCall(PetscHMapIFind(bodyVertexMap, b, &BViter, &BVfound));
      PetscCall(PetscHMapIFind(bodyEdgeMap, b, &BEiter, &BEfound));
      PetscCall(PetscHMapIFind(bodyEdgeGlobalMap, b, &BEGiter, &BEGfound));
      PetscCall(PetscHMapIFind(bodyFaceMap, b, &BFiter, &BFfound));

      if (!BIfound) PetscCall(PetscHMapISet(bodyIndexMap, b, numFaces + numEdges + numVertices));
      if (!BVfound) PetscCall(PetscHMapISet(bodyVertexMap, b, numVertices));
      if (!BEfound) PetscCall(PetscHMapISet(bodyEdgeMap, b, numEdges));
      if (!BEGfound) PetscCall(PetscHMapISet(bodyEdgeGlobalMap, b, edgeCntr));
      if (!BFfound) PetscCall(PetscHMapISet(bodyFaceMap, b, numFaces));

      if (islite) {
        PetscCall(EGlite_getBodyTopos(body, NULL, FACE, &Nf, &fobjs));
        PetscCall(EGlite_getBodyTopos(body, NULL, EDGE, &Ne, &eobjs));
        PetscCall(EGlite_getBodyTopos(body, NULL, NODE, &Nv, &nobjs));
        EGlite_free(fobjs);
        EGlite_free(eobjs);
        EGlite_free(nobjs);
      } else {
        PetscCall(EG_getBodyTopos(body, NULL, FACE, &Nf, &fobjs));
        PetscCall(EG_getBodyTopos(body, NULL, EDGE, &Ne, &eobjs));
        PetscCall(EG_getBodyTopos(body, NULL, NODE, &Nv, &nobjs));
        EG_free(fobjs);
        EG_free(eobjs);
        EG_free(nobjs);
      }

      // Remove DEGENERATE EDGES from Edge count
      if (islite) {
        PetscCall(EGlite_getBodyTopos(body, NULL, EDGE, &Ne, &eobjs));
      } else {
        PetscCall(EG_getBodyTopos(body, NULL, EDGE, &Ne, &eobjs));
      }

      int Netemp = 0;
      for (int e = 0; e < Ne; ++e) {
        ego edge = eobjs[e];
        int eid;

        if (islite) {
          PetscCall(EGlite_getInfo(edge, &oclass, &mtype, &topRef, &prev, &next));
          eid = EGlite_indexBodyTopo(body, edge);
        } else {
          PetscCall(EG_getInfo(edge, &oclass, &mtype, &topRef, &prev, &next));
          eid = EG_indexBodyTopo(body, edge);
        }

        PetscCall(PetscHMapIFind(edgeMap, edgeCntr + eid - 1, &EMiter, &EMfound));
        if (mtype == DEGENERATE) {
          if (!EMfound) PetscCall(PetscHMapISet(edgeMap, edgeCntr + eid - 1, -1));
        } else {
          ++Netemp;
          if (!EMfound) PetscCall(PetscHMapISet(edgeMap, edgeCntr + eid - 1, Netemp));
        }
      }
      if (islite) {
        EGlite_free(eobjs);
      } else {
        EG_free(eobjs);
      }

      // Determine Number of Cells
      if (islite) {
        PetscCall(EGlite_getBodyTopos(body, NULL, FACE, &Nf, &fobjs));
      } else {
        PetscCall(EG_getBodyTopos(body, NULL, FACE, &Nf, &fobjs));
      }

      for (int f = 0; f < Nf; ++f) {
        ego face     = fobjs[f];
        int edgeTemp = 0;

        if (islite) {
          PetscCall(EGlite_getBodyTopos(body, face, EDGE, &Ne, &eobjs));
        } else {
          PetscCall(EG_getBodyTopos(body, face, EDGE, &Ne, &eobjs));
        }

        for (int e = 0; e < Ne; ++e) {
          ego edge = eobjs[e];

          if (islite) {
            PetscCall(EGlite_getInfo(edge, &oclass, &mtype, &topRef, &prev, &next));
          } else {
            PetscCall(EG_getInfo(edge, &oclass, &mtype, &topRef, &prev, &next));
          }
          if (mtype != DEGENERATE) ++edgeTemp;
        }
        numCells += (2 * edgeTemp);
        if (islite) {
          EGlite_free(eobjs);
        } else {
          EG_free(eobjs);
        }
      }
      if (islite) {
        EGlite_free(fobjs);
      } else {
        EG_free(fobjs);
      }

      numFaces += Nf;
      numEdges += Netemp;
      numVertices += Nv;
      edgeCntr += Ne;
    }

    // Set up basic DMPlex parameters
    dim        = 2;                                 // Assumes 3D Models :: Need to handle 2D models in the future
    cdim       = 3;                                 // Assumes 3D Models :: Need to update to handle 2D models in future
    numCorners = 3;                                 // Split Faces into triangles
    numPoints  = numVertices + numEdges + numFaces; // total number of coordinate points

    PetscCall(PetscMalloc2(numPoints * cdim, &coords, numCells * numCorners, &cells));

    // Get Vertex Coordinates and Set up Cells
    for (b = 0; b < nbodies; ++b) {
      ego           body = bodies[b];
      int           Nf, Ne, Nv;
      PetscInt      bodyVertexIndexStart, bodyEdgeIndexStart, bodyEdgeGlobalIndexStart, bodyFaceIndexStart;
      PetscHashIter BViter, BEiter, BEGiter, BFiter, EMiter;
      PetscBool     BVfound, BEfound, BEGfound, BFfound, EMfound;

      // Vertices on Current Body
      if (islite) {
        PetscCall(EGlite_getBodyTopos(body, NULL, NODE, &Nv, &nobjs));
      } else {
        PetscCall(EG_getBodyTopos(body, NULL, NODE, &Nv, &nobjs));
      }

      PetscCall(PetscHMapIFind(bodyVertexMap, b, &BViter, &BVfound));
      PetscCheck(BVfound, PETSC_COMM_SELF, PETSC_ERR_SUP, "Body %" PetscInt_FMT " not found in bodyVertexMap", b);
      PetscCall(PetscHMapIGet(bodyVertexMap, b, &bodyVertexIndexStart));

      for (int v = 0; v < Nv; ++v) {
        ego    vertex = nobjs[v];
        double limits[4];
        int    id, dummy;

        if (islite) {
          PetscCall(EGlite_getTopology(vertex, &geom, &oclass, &mtype, limits, &dummy, &mobjs, &senses));
          id = EGlite_indexBodyTopo(body, vertex);
        } else {
          PetscCall(EG_getTopology(vertex, &geom, &oclass, &mtype, limits, &dummy, &mobjs, &senses));
          id = EG_indexBodyTopo(body, vertex);
        }

        coords[(bodyVertexIndexStart + id - 1) * cdim + 0] = limits[0];
        coords[(bodyVertexIndexStart + id - 1) * cdim + 1] = limits[1];
        coords[(bodyVertexIndexStart + id - 1) * cdim + 2] = limits[2];
      }
      if (islite) {
        EGlite_free(nobjs);
      } else {
        EG_free(nobjs);
      }

      // Edge Midpoint Vertices on Current Body
      if (islite) {
        PetscCall(EGlite_getBodyTopos(body, NULL, EDGE, &Ne, &eobjs));
      } else {
        PetscCall(EG_getBodyTopos(body, NULL, EDGE, &Ne, &eobjs));
      }

      PetscCall(PetscHMapIFind(bodyEdgeMap, b, &BEiter, &BEfound));
      PetscCheck(BEfound, PETSC_COMM_SELF, PETSC_ERR_SUP, "Body %" PetscInt_FMT " not found in bodyEdgeMap", b);
      PetscCall(PetscHMapIGet(bodyEdgeMap, b, &bodyEdgeIndexStart));

      PetscCall(PetscHMapIFind(bodyEdgeGlobalMap, b, &BEGiter, &BEGfound));
      PetscCheck(BEGfound, PETSC_COMM_SELF, PETSC_ERR_SUP, "Body %" PetscInt_FMT " not found in bodyEdgeGlobalMap", b);
      PetscCall(PetscHMapIGet(bodyEdgeGlobalMap, b, &bodyEdgeGlobalIndexStart));

      for (int e = 0; e < Ne; ++e) {
        ego    edge = eobjs[e];
        double range[2], avgt[1], cntrPnt[9];
        int    eid, eOffset;
        int    periodic;

        if (islite) {
          PetscCall(EGlite_getInfo(edge, &oclass, &mtype, &topRef, &prev, &next));
        } else {
          PetscCall(EG_getInfo(edge, &oclass, &mtype, &topRef, &prev, &next));
        }
        if (mtype == DEGENERATE) continue;

        if (islite) {
          eid = EGlite_indexBodyTopo(body, edge);
        } else {
          eid = EG_indexBodyTopo(body, edge);
        }
        // get relative offset from globalEdgeID Vector
        PetscCall(PetscHMapIFind(edgeMap, bodyEdgeGlobalIndexStart + eid - 1, &EMiter, &EMfound));
        PetscCheck(EMfound, PETSC_COMM_SELF, PETSC_ERR_SUP, "Edge %" PetscInt_FMT " not found in edgeMap", bodyEdgeGlobalIndexStart + eid - 1);
        PetscCall(PetscHMapIGet(edgeMap, bodyEdgeGlobalIndexStart + eid - 1, &eOffset));

        if (islite) {
          PetscCall(EGlite_getRange(edge, range, &periodic));
        } else {
          PetscCall(EG_getRange(edge, range, &periodic));
        }
        avgt[0] = (range[0] + range[1]) / 2.;

        if (islite) {
          PetscCall(EGlite_evaluate(edge, avgt, cntrPnt));
        } else {
          PetscCall(EG_evaluate(edge, avgt, cntrPnt));
        }
        coords[(numVertices + bodyEdgeIndexStart + eOffset - 1) * cdim + 0] = cntrPnt[0];
        coords[(numVertices + bodyEdgeIndexStart + eOffset - 1) * cdim + 1] = cntrPnt[1];
        coords[(numVertices + bodyEdgeIndexStart + eOffset - 1) * cdim + 2] = cntrPnt[2];
      }
      if (islite) {
        EGlite_free(eobjs);
      } else {
        EG_free(eobjs);
      }
      // Face Midpoint Vertices on Current Body
      if (islite) {
        PetscCall(EGlite_getBodyTopos(body, NULL, FACE, &Nf, &fobjs));
      } else {
        PetscCall(EG_getBodyTopos(body, NULL, FACE, &Nf, &fobjs));
      }
      PetscCall(PetscHMapIFind(bodyFaceMap, b, &BFiter, &BFfound));
      PetscCheck(BFfound, PETSC_COMM_SELF, PETSC_ERR_SUP, "Body %d not found in bodyFaceMap", b);
      PetscCall(PetscHMapIGet(bodyFaceMap, b, &bodyFaceIndexStart));

      for (int f = 0; f < Nf; ++f) {
        ego    face = fobjs[f];
        double range[4], avgUV[2], cntrPnt[18];
        int    peri, id;

        if (islite) {
          id = EGlite_indexBodyTopo(body, face);
          PetscCall(EGlite_getRange(face, range, &peri));
        } else {
          id = EG_indexBodyTopo(body, face);
          PetscCall(EG_getRange(face, range, &peri));
        }

        avgUV[0] = (range[0] + range[1]) / 2.;
        avgUV[1] = (range[2] + range[3]) / 2.;

        if (islite) {
          PetscCall(EGlite_evaluate(face, avgUV, cntrPnt));
        } else {
          PetscCall(EG_evaluate(face, avgUV, cntrPnt));
        }

        coords[(numVertices + numEdges + bodyFaceIndexStart + id - 1) * cdim + 0] = cntrPnt[0];
        coords[(numVertices + numEdges + bodyFaceIndexStart + id - 1) * cdim + 1] = cntrPnt[1];
        coords[(numVertices + numEdges + bodyFaceIndexStart + id - 1) * cdim + 2] = cntrPnt[2];
      }
      if (islite) {
        EGlite_free(fobjs);
      } else {
        EG_free(fobjs);
      }

      // Define Cells :: Note - This could be incorporated in the Face Midpoint Vertices Loop but was kept separate for clarity
      if (islite) {
        PetscCall(EGlite_getBodyTopos(body, NULL, FACE, &Nf, &fobjs));
      } else {
        PetscCall(EG_getBodyTopos(body, NULL, FACE, &Nf, &fobjs));
      }
      for (int f = 0; f < Nf; ++f) {
        ego face = fobjs[f];
        int fID, midFaceID, midPntID, startID, endID, Nl;

        if (islite) {
          fID = EGlite_indexBodyTopo(body, face);
        } else {
          fID = EG_indexBodyTopo(body, face);
        }

        midFaceID = numVertices + numEdges + bodyFaceIndexStart + fID - 1;
        // Must Traverse Loop to ensure we have all necessary information like the sense (+/- 1) of the edges.
        // TODO :: Only handles single loop faces (No holes). The choices for handling multiloop faces are:
        //            1) Use the DMPlexCreateGeomFromFile() with the -dm_plex_geom_with_tess = 1 option.
        //               This will use a default EGADS tessellation as an initial surface mesh.
        //            2) Create the initial surface mesh via a 2D mesher :: Currently not available (?future?)
        //               May I suggest the XXXX as a starting point?

        if (islite) {
          PetscCall(EGlite_getTopology(face, &geom, &oclass, &mtype, NULL, &Nl, &lobjs, &lSenses));
        } else {
          PetscCall(EG_getTopology(face, &geom, &oclass, &mtype, NULL, &Nl, &lobjs, &lSenses));
        }

        PetscCheck(Nl == 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Face has %" PetscInt_FMT " Loops. Can only handle Faces with 1 Loop. Please use --dm_plex_geom_with_tess = 1 Option", Nl);
        for (int l = 0; l < Nl; ++l) {
          ego loop = lobjs[l];

          if (islite) {
            PetscCall(EGlite_getTopology(loop, &geom, &oclass, &mtype, NULL, &Ne, &eobjs, &eSenses));
          } else {
            PetscCall(EG_getTopology(loop, &geom, &oclass, &mtype, NULL, &Ne, &eobjs, &eSenses));
          }

          for (int e = 0; e < Ne; ++e) {
            ego edge = eobjs[e];
            int eid, eOffset;

            if (islite) {
              PetscCall(EGlite_getInfo(edge, &oclass, &mtype, &topRef, &prev, &next));
              eid = EGlite_indexBodyTopo(body, edge);
            } else {
              PetscCall(EG_getInfo(edge, &oclass, &mtype, &topRef, &prev, &next));
              eid = EG_indexBodyTopo(body, edge);
            }
            if (mtype == DEGENERATE) continue;

            // get relative offset from globalEdgeID Vector
            PetscCall(PetscHMapIFind(edgeMap, bodyEdgeGlobalIndexStart + eid - 1, &EMiter, &EMfound));
            PetscCheck(EMfound, PETSC_COMM_SELF, PETSC_ERR_SUP, "Edge %" PetscInt_FMT " of Body %" PetscInt_FMT " not found in edgeMap. Global Edge ID :: %" PetscInt_FMT, eid, b, bodyEdgeGlobalIndexStart + eid - 1);
            PetscCall(PetscHMapIGet(edgeMap, bodyEdgeGlobalIndexStart + eid - 1, &eOffset));

            midPntID = numVertices + bodyEdgeIndexStart + eOffset - 1;

            if (islite) {
              PetscCall(EGlite_getTopology(edge, &geom, &oclass, &mtype, NULL, &Nv, &nobjs, &senses));
            } else {
              PetscCall(EG_getTopology(edge, &geom, &oclass, &mtype, NULL, &Nv, &nobjs, &senses));
            }

            if (eSenses[e] > 0) {
              if (islite) {
                startID = EGlite_indexBodyTopo(body, nobjs[0]);
                endID   = EGlite_indexBodyTopo(body, nobjs[1]);
              } else {
                startID = EG_indexBodyTopo(body, nobjs[0]);
                endID   = EG_indexBodyTopo(body, nobjs[1]);
              }
            } else {
              if (islite) {
                startID = EGlite_indexBodyTopo(body, nobjs[1]);
                endID   = EGlite_indexBodyTopo(body, nobjs[0]);
              } else {
                startID = EG_indexBodyTopo(body, nobjs[1]);
                endID   = EG_indexBodyTopo(body, nobjs[0]);
              }
            }

            // Define 2 Cells per Edge with correct orientation
            cells[cellCntr * numCorners + 0] = midFaceID;
            cells[cellCntr * numCorners + 1] = bodyVertexIndexStart + startID - 1;
            cells[cellCntr * numCorners + 2] = midPntID;

            cells[cellCntr * numCorners + 3] = midFaceID;
            cells[cellCntr * numCorners + 4] = midPntID;
            cells[cellCntr * numCorners + 5] = bodyVertexIndexStart + endID - 1;

            cellCntr = cellCntr + 2;
          }
        }
      }
      if (islite) {
        EGlite_free(fobjs);
      } else {
        EG_free(fobjs);
      }
    }
  }

  // Generate DMPlex
  PetscCall(DMPlexCreateFromCellListPetsc(PETSC_COMM_WORLD, dim, numCells, numPoints, numCorners, PETSC_TRUE, cells, cdim, coords, &dm));
  PetscCall(PetscFree2(coords, cells));
  PetscCall(PetscInfo(dm, " Total Number of Unique Cells    = %" PetscInt_FMT " \n", numCells));
  PetscCall(PetscInfo(dm, " Total Number of Unique Vertices = %" PetscInt_FMT " \n", numVertices));

  // Embed EGADS model in DM
  {
    PetscContainer modelObj, contextObj;

    PetscCall(PetscContainerCreate(PETSC_COMM_SELF, &modelObj));
    PetscCall(PetscContainerSetPointer(modelObj, model));
    if (islite) {
      PetscCall(PetscContainerSetCtxDestroy(modelObj, DMPlexEGADSliteDestroy_Private));
      PetscCall(PetscObjectCompose((PetscObject)dm, "EGADSlite Model", (PetscObject)modelObj));
    } else {
      PetscCall(PetscContainerSetCtxDestroy(modelObj, DMPlexEGADSDestroy_Private));
      PetscCall(PetscObjectCompose((PetscObject)dm, "EGADS Model", (PetscObject)modelObj));
    }
    PetscCall(PetscContainerDestroy(&modelObj));

    PetscCall(PetscContainerCreate(PETSC_COMM_SELF, &contextObj));
    PetscCall(PetscContainerSetPointer(contextObj, context));

    if (islite) {
      PetscCall(PetscContainerSetCtxDestroy(contextObj, DMPlexEGADSliteClose_Private));
      PetscCall(PetscObjectCompose((PetscObject)dm, "EGADSlite Context", (PetscObject)contextObj));
    } else {
      PetscCall(PetscContainerSetCtxDestroy(contextObj, DMPlexEGADSClose_Private));
      PetscCall(PetscObjectCompose((PetscObject)dm, "EGADS Context", (PetscObject)contextObj));
    }
    PetscCall(PetscContainerDestroy(&contextObj));
  }
  // Label points
  PetscInt nStart, nEnd;

  PetscCall(DMCreateLabel(dm, "EGADS Body ID"));
  PetscCall(DMGetLabel(dm, "EGADS Body ID", &bodyLabel));
  PetscCall(DMCreateLabel(dm, "EGADS Face ID"));
  PetscCall(DMGetLabel(dm, "EGADS Face ID", &faceLabel));
  PetscCall(DMCreateLabel(dm, "EGADS Edge ID"));
  PetscCall(DMGetLabel(dm, "EGADS Edge ID", &edgeLabel));
  PetscCall(DMCreateLabel(dm, "EGADS Vertex ID"));
  PetscCall(DMGetLabel(dm, "EGADS Vertex ID", &vertexLabel));

  PetscCall(DMPlexGetHeightStratum(dm, 2, &nStart, &nEnd));

  cellCntr = 0;
  for (b = 0; b < nbodies; ++b) {
    ego           body = bodies[b];
    int           Nv, Ne, Nf;
    PetscInt      bodyVertexIndexStart, bodyEdgeIndexStart, bodyEdgeGlobalIndexStart, bodyFaceIndexStart;
    PetscHashIter BViter, BEiter, BEGiter, BFiter, EMiter;
    PetscBool     BVfound, BEfound, BEGfound, BFfound, EMfound;

    PetscCall(PetscHMapIFind(bodyVertexMap, b, &BViter, &BVfound));
    PetscCheck(BVfound, PETSC_COMM_SELF, PETSC_ERR_SUP, "Body %d not found in bodyVertexMap", b);
    PetscCall(PetscHMapIGet(bodyVertexMap, b, &bodyVertexIndexStart));

    PetscCall(PetscHMapIFind(bodyEdgeMap, b, &BEiter, &BEfound));
    PetscCheck(BEfound, PETSC_COMM_SELF, PETSC_ERR_SUP, "Body %d not found in bodyEdgeMap", b);
    PetscCall(PetscHMapIGet(bodyEdgeMap, b, &bodyEdgeIndexStart));

    PetscCall(PetscHMapIFind(bodyFaceMap, b, &BFiter, &BFfound));
    PetscCheck(BFfound, PETSC_COMM_SELF, PETSC_ERR_SUP, "Body %d not found in bodyFaceMap", b);
    PetscCall(PetscHMapIGet(bodyFaceMap, b, &bodyFaceIndexStart));

    PetscCall(PetscHMapIFind(bodyEdgeGlobalMap, b, &BEGiter, &BEGfound));
    PetscCheck(BEGfound, PETSC_COMM_SELF, PETSC_ERR_SUP, "Body %d not found in bodyEdgeGlobalMap", b);
    PetscCall(PetscHMapIGet(bodyEdgeGlobalMap, b, &bodyEdgeGlobalIndexStart));

    if (islite) {
      PetscCall(EGlite_getBodyTopos(body, NULL, FACE, &Nf, &fobjs));
    } else {
      PetscCall(EG_getBodyTopos(body, NULL, FACE, &Nf, &fobjs));
    }

    for (int f = 0; f < Nf; ++f) {
      ego face = fobjs[f];
      int fID, Nl;

      if (islite) {
        fID = EGlite_indexBodyTopo(body, face);
        PetscCall(EGlite_getBodyTopos(body, face, LOOP, &Nl, &lobjs));
      } else {
        fID = EG_indexBodyTopo(body, face);
        PetscCall(EG_getBodyTopos(body, face, LOOP, &Nl, &lobjs));
      }

      for (int l = 0; l < Nl; ++l) {
        ego loop = lobjs[l];
        int lid;

        if (islite) {
          lid = EGlite_indexBodyTopo(body, loop);
        } else {
          lid = EG_indexBodyTopo(body, loop);
        }

        PetscCheck(Nl == 1, PETSC_COMM_SELF, PETSC_ERR_SUP, "Loop %" PetscInt_FMT " has %" PetscInt_FMT " > 1 faces, which is not supported", lid, Nf);

        if (islite) {
          PetscCall(EGlite_getTopology(loop, &geom, &oclass, &mtype, NULL, &Ne, &eobjs, &eSenses));
        } else {
          PetscCall(EG_getTopology(loop, &geom, &oclass, &mtype, NULL, &Ne, &eobjs, &eSenses));
        }

        for (int e = 0; e < Ne; ++e) {
          ego edge = eobjs[e];
          int eid, eOffset;

          // Skip DEGENERATE Edges
          if (islite) {
            PetscCall(EGlite_getInfo(edge, &oclass, &mtype, &topRef, &prev, &next));
          } else {
            PetscCall(EG_getInfo(edge, &oclass, &mtype, &topRef, &prev, &next));
          }

          if (mtype == DEGENERATE) { continue; }

          if (islite) {
            eid = EGlite_indexBodyTopo(body, edge);
          } else {
            eid = EG_indexBodyTopo(body, edge);
          }

          // get relative offset from globalEdgeID Vector
          PetscCall(PetscHMapIFind(edgeMap, bodyEdgeGlobalIndexStart + eid - 1, &EMiter, &EMfound));
          PetscCheck(EMfound, PETSC_COMM_SELF, PETSC_ERR_SUP, "Edge %" PetscInt_FMT " of Body %" PetscInt_FMT " not found in edgeMap. Global Edge ID :: %" PetscInt_FMT, eid, b, bodyEdgeGlobalIndexStart + eid - 1);
          PetscCall(PetscHMapIGet(edgeMap, bodyEdgeGlobalIndexStart + eid - 1, &eOffset));

          if (islite) {
            PetscCall(EGlite_getBodyTopos(body, edge, NODE, &Nv, &nobjs));
          } else {
            PetscCall(EG_getBodyTopos(body, edge, NODE, &Nv, &nobjs));
          }

          for (int v = 0; v < Nv; ++v) {
            ego vertex = nobjs[v];
            int vID;

            if (islite) {
              vID = EGlite_indexBodyTopo(body, vertex);
            } else {
              vID = EG_indexBodyTopo(body, vertex);
            }

            PetscCall(DMLabelSetValue(bodyLabel, nStart + bodyVertexIndexStart + vID - 1, b));
            PetscCall(DMLabelSetValue(vertexLabel, nStart + bodyVertexIndexStart + vID - 1, vID));
          }
          if (islite) {
            EGlite_free(nobjs);
          } else {
            EG_free(nobjs);
          }

          PetscCall(DMLabelSetValue(bodyLabel, nStart + numVertices + bodyEdgeIndexStart + eOffset - 1, b));
          PetscCall(DMLabelSetValue(edgeLabel, nStart + numVertices + bodyEdgeIndexStart + eOffset - 1, eid));

          // Define Cell faces
          for (int jj = 0; jj < 2; ++jj) {
            PetscCall(DMLabelSetValue(bodyLabel, cellCntr, b));
            PetscCall(DMLabelSetValue(faceLabel, cellCntr, fID));
            PetscCall(DMPlexGetCone(dm, cellCntr, &cone));

            PetscCall(DMLabelSetValue(bodyLabel, cone[0], b));
            PetscCall(DMLabelSetValue(faceLabel, cone[0], fID));

            PetscCall(DMLabelSetValue(bodyLabel, cone[1], b));
            PetscCall(DMLabelSetValue(edgeLabel, cone[1], eid));

            PetscCall(DMLabelSetValue(bodyLabel, cone[2], b));
            PetscCall(DMLabelSetValue(faceLabel, cone[2], fID));

            cellCntr = cellCntr + 1;
          }
        }
      }
      if (islite) {
        EGlite_free(lobjs);
      } else {
        EG_free(lobjs);
      }

      PetscCall(DMLabelSetValue(bodyLabel, nStart + numVertices + numEdges + bodyFaceIndexStart + fID - 1, b));
      PetscCall(DMLabelSetValue(faceLabel, nStart + numVertices + numEdges + bodyFaceIndexStart + fID - 1, fID));
    }
    if (islite) {
      EGlite_free(fobjs);
    } else {
      EG_free(fobjs);
    }
  }

  PetscCall(PetscHMapIDestroy(&edgeMap));
  PetscCall(PetscHMapIDestroy(&bodyIndexMap));
  PetscCall(PetscHMapIDestroy(&bodyVertexMap));
  PetscCall(PetscHMapIDestroy(&bodyEdgeMap));
  PetscCall(PetscHMapIDestroy(&bodyEdgeGlobalMap));
  PetscCall(PetscHMapIDestroy(&bodyFaceMap));

  *newdm = dm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMPlexCreateGeom_Tess_Internal(MPI_Comm comm, ego context, ego model, DM *newdm, PetscBool islite)
{
  /* EGADSlite variables */
  ego    geom, *bodies, *fobjs;
  int    b, oclass, mtype, nbodies, *senses;
  int    totalNumTris = 0, totalNumPoints = 0;
  double boundBox[6] = {0., 0., 0., 0., 0., 0.}, tessSize;
  /* PETSc variables */
  DM              dm;
  DMLabel         bodyLabel, faceLabel, edgeLabel, vertexLabel;
  PetscHMapI      pointIndexStartMap = NULL, triIndexStartMap = NULL, pTypeLabelMap = NULL, pIndexLabelMap = NULL;
  PetscHMapI      pBodyIndexLabelMap = NULL, triFaceIDLabelMap = NULL, triBodyIDLabelMap = NULL;
  PetscInt        dim = -1, cdim = -1, numCorners = 0, counter = 0;
  PetscInt       *cells  = NULL;
  const PetscInt *cone   = NULL;
  PetscReal      *coords = NULL;
  PetscMPIInt     rank;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  if (rank == 0) {
    // ---------------------------------------------------------------------------------------------------
    // Generate PETSc DMPlex from EGADSlite created Tessellation of geometry
    // ---------------------------------------------------------------------------------------------------

    // Calculate cell and vertex sizes
    if (islite) {
      PetscCall(EGlite_getTopology(model, &geom, &oclass, &mtype, NULL, &nbodies, &bodies, &senses));
    } else {
      PetscCall(EG_getTopology(model, &geom, &oclass, &mtype, NULL, &nbodies, &bodies, &senses));
    }

    PetscCall(PetscHMapICreate(&pointIndexStartMap));
    PetscCall(PetscHMapICreate(&triIndexStartMap));
    PetscCall(PetscHMapICreate(&pTypeLabelMap));
    PetscCall(PetscHMapICreate(&pIndexLabelMap));
    PetscCall(PetscHMapICreate(&pBodyIndexLabelMap));
    PetscCall(PetscHMapICreate(&triFaceIDLabelMap));
    PetscCall(PetscHMapICreate(&triBodyIDLabelMap));

    /* Create Tessellation of Bodies */
    ego *tessArray;

    PetscCall(PetscMalloc1(nbodies, &tessArray));
    for (b = 0; b < nbodies; ++b) {
      ego           body      = bodies[b];
      double        params[3] = {0.0, 0.0, 0.0}; // Parameters for Tessellation
      int           Nf, bodyNumPoints = 0, bodyNumTris = 0;
      PetscHashIter PISiter, TISiter;
      PetscBool     PISfound, TISfound;

      /* Store Start Index for each Body's Point and Tris */
      PetscCall(PetscHMapIFind(pointIndexStartMap, b, &PISiter, &PISfound));
      PetscCall(PetscHMapIFind(triIndexStartMap, b, &TISiter, &TISfound));

      if (!PISfound) PetscCall(PetscHMapISet(pointIndexStartMap, b, totalNumPoints));
      if (!TISfound) PetscCall(PetscHMapISet(triIndexStartMap, b, totalNumTris));

      /* Calculate Tessellation parameters based on Bounding Box */
      /* Get Bounding Box Dimensions of the BODY */
      if (islite) {
        PetscCall(EGlite_getBoundingBox(body, boundBox));
      } else {
        PetscCall(EG_getBoundingBox(body, boundBox));
      }

      tessSize = boundBox[3] - boundBox[0];
      if (tessSize < boundBox[4] - boundBox[1]) tessSize = boundBox[4] - boundBox[1];
      if (tessSize < boundBox[5] - boundBox[2]) tessSize = boundBox[5] - boundBox[2];

      // TODO :: May want to give users tessellation parameter options //
      params[0] = 0.0250 * tessSize;
      params[1] = 0.0075 * tessSize;
      params[2] = 15.0;

      if (islite) {
        PetscCall(EGlite_makeTessBody(body, params, &tessArray[b]));
        PetscCall(EGlite_getBodyTopos(body, NULL, FACE, &Nf, &fobjs));
      } else {
        PetscCall(EG_makeTessBody(body, params, &tessArray[b]));
        PetscCall(EG_getBodyTopos(body, NULL, FACE, &Nf, &fobjs));
      }

      for (int f = 0; f < Nf; ++f) {
        ego           face = fobjs[f];
        int           len, fID, ntris;
        const int    *ptype, *pindex, *ptris, *ptric;
        const double *pxyz, *puv;

        // Get Face ID //
        if (islite) {
          fID = EGlite_indexBodyTopo(body, face);
        } else {
          fID = EG_indexBodyTopo(body, face);
        }

        // Checkout the Surface Tessellation //
        if (islite) {
          PetscCall(EGlite_getTessFace(tessArray[b], fID, &len, &pxyz, &puv, &ptype, &pindex, &ntris, &ptris, &ptric));
        } else {
          PetscCall(EG_getTessFace(tessArray[b], fID, &len, &pxyz, &puv, &ptype, &pindex, &ntris, &ptris, &ptric));
        }

        // Determine total number of triangle cells in the tessellation //
        bodyNumTris += (int)ntris;

        // Check out the point index and coordinate //
        for (int p = 0; p < len; ++p) {
          int global;

          if (islite) {
            PetscCall(EGlite_localToGlobal(tessArray[b], fID, p + 1, &global));
          } else {
            PetscCall(EG_localToGlobal(tessArray[b], fID, p + 1, &global));
          }

          // Determine the total number of points in the tessellation //
          bodyNumPoints = PetscMax(bodyNumPoints, global);
        }
      }
      if (islite) {
        EGlite_free(fobjs);
      } else {
        EG_free(fobjs);
      }

      totalNumPoints += bodyNumPoints;
      totalNumTris += bodyNumTris;
    }

    dim        = 2;
    cdim       = 3;
    numCorners = 3;

    /* NEED TO DEFINE MATRICES/VECTORS TO STORE GEOM REFERENCE DATA   */
    /* Fill in below and use to define DMLabels after DMPlex creation */
    PetscCall(PetscMalloc2(totalNumPoints * cdim, &coords, totalNumTris * numCorners, &cells));

    for (b = 0; b < nbodies; ++b) {
      ego           body = bodies[b];
      int           Nf;
      PetscInt      pointIndexStart;
      PetscHashIter PISiter;
      PetscBool     PISfound;

      PetscCall(PetscHMapIFind(pointIndexStartMap, b, &PISiter, &PISfound));
      PetscCheck(PISfound, PETSC_COMM_SELF, PETSC_ERR_SUP, "Body %" PetscInt_FMT " not found in pointIndexStartMap", b);
      PetscCall(PetscHMapIGet(pointIndexStartMap, b, &pointIndexStart));

      if (islite) {
        PetscCall(EGlite_getBodyTopos(body, NULL, FACE, &Nf, &fobjs));
      } else {
        PetscCall(EG_getBodyTopos(body, NULL, FACE, &Nf, &fobjs));
      }

      for (int f = 0; f < Nf; ++f) {
        /* Get Face Object */
        ego           face = fobjs[f];
        int           len, fID, ntris;
        const int    *ptype, *pindex, *ptris, *ptric;
        const double *pxyz, *puv;

        /* Get Face ID */
        if (islite) {
          fID = EGlite_indexBodyTopo(body, face);
        } else {
          fID = EG_indexBodyTopo(body, face);
        }

        /* Checkout the Surface Tessellation */
        if (islite) {
          PetscCall(EGlite_getTessFace(tessArray[b], fID, &len, &pxyz, &puv, &ptype, &pindex, &ntris, &ptris, &ptric));
        } else {
          PetscCall(EG_getTessFace(tessArray[b], fID, &len, &pxyz, &puv, &ptype, &pindex, &ntris, &ptris, &ptric));
        }

        /* Check out the point index and coordinate */
        for (int p = 0; p < len; ++p) {
          int           global;
          PetscHashIter PTLiter, PILiter, PBLiter;
          PetscBool     PTLfound, PILfound, PBLfound;

          if (islite) {
            PetscCall(EGlite_localToGlobal(tessArray[b], fID, p + 1, &global));
          } else {
            PetscCall(EG_localToGlobal(tessArray[b], fID, p + 1, &global));
          }

          /* Set the coordinates array for DAG */
          coords[((global - 1 + pointIndexStart) * 3) + 0] = pxyz[(p * 3) + 0];
          coords[((global - 1 + pointIndexStart) * 3) + 1] = pxyz[(p * 3) + 1];
          coords[((global - 1 + pointIndexStart) * 3) + 2] = pxyz[(p * 3) + 2];

          /* Store Geometry Label Information for DMLabel assignment later */
          PetscCall(PetscHMapIFind(pTypeLabelMap, global - 1 + pointIndexStart, &PTLiter, &PTLfound));
          PetscCall(PetscHMapIFind(pIndexLabelMap, global - 1 + pointIndexStart, &PILiter, &PILfound));
          PetscCall(PetscHMapIFind(pBodyIndexLabelMap, global - 1 + pointIndexStart, &PBLiter, &PBLfound));

          if (!PTLfound) PetscCall(PetscHMapISet(pTypeLabelMap, global - 1 + pointIndexStart, ptype[p]));
          if (!PILfound) PetscCall(PetscHMapISet(pIndexLabelMap, global - 1 + pointIndexStart, pindex[p]));
          if (!PBLfound) PetscCall(PetscHMapISet(pBodyIndexLabelMap, global - 1 + pointIndexStart, b));

          if (ptype[p] < 0) PetscCall(PetscHMapISet(pIndexLabelMap, global - 1 + pointIndexStart, fID));
        }

        for (int t = 0; t < (int)ntris; ++t) {
          int           global, globalA, globalB;
          PetscHashIter TFLiter, TBLiter;
          PetscBool     TFLfound, TBLfound;

          if (islite) {
            PetscCall(EGlite_localToGlobal(tessArray[b], fID, ptris[(t * 3) + 0], &global));
          } else {
            PetscCall(EG_localToGlobal(tessArray[b], fID, ptris[(t * 3) + 0], &global));
          }
          cells[(counter * 3) + 0] = global - 1 + pointIndexStart;

          if (islite) {
            PetscCall(EGlite_localToGlobal(tessArray[b], fID, ptris[(t * 3) + 1], &globalA));
          } else {
            PetscCall(EG_localToGlobal(tessArray[b], fID, ptris[(t * 3) + 1], &globalA));
          }
          cells[(counter * 3) + 1] = globalA - 1 + pointIndexStart;

          if (islite) {
            PetscCall(EGlite_localToGlobal(tessArray[b], fID, ptris[(t * 3) + 2], &globalB));
          } else {
            PetscCall(EG_localToGlobal(tessArray[b], fID, ptris[(t * 3) + 2], &globalB));
          }
          cells[(counter * 3) + 2] = globalB - 1 + pointIndexStart;

          PetscCall(PetscHMapIFind(triFaceIDLabelMap, counter, &TFLiter, &TFLfound));
          PetscCall(PetscHMapIFind(triBodyIDLabelMap, counter, &TBLiter, &TBLfound));

          if (!TFLfound) PetscCall(PetscHMapISet(triFaceIDLabelMap, counter, fID));
          if (!TBLfound) PetscCall(PetscHMapISet(triBodyIDLabelMap, counter, b));

          counter += 1;
        }
      }
      if (islite) {
        EGlite_free(fobjs);
      } else {
        EG_free(fobjs);
      }
    }
    PetscCall(PetscFree(tessArray));
  }

  //Build DMPlex
  PetscCall(DMPlexCreateFromCellListPetsc(PETSC_COMM_WORLD, dim, totalNumTris, totalNumPoints, numCorners, PETSC_TRUE, cells, cdim, coords, &dm));
  PetscCall(PetscFree2(coords, cells));

  // Embed EGADS model in DM
  {
    PetscContainer modelObj, contextObj;

    PetscCall(PetscContainerCreate(PETSC_COMM_SELF, &modelObj));
    PetscCall(PetscContainerSetPointer(modelObj, model));
    if (islite) {
      PetscCall(PetscContainerSetCtxDestroy(modelObj, (PetscCtxDestroyFn *)DMPlexEGADSliteDestroy_Private));
      PetscCall(PetscObjectCompose((PetscObject)dm, "EGADSlite Model", (PetscObject)modelObj));
    } else {
      PetscCall(PetscContainerSetCtxDestroy(modelObj, (PetscCtxDestroyFn *)DMPlexEGADSDestroy_Private));
      PetscCall(PetscObjectCompose((PetscObject)dm, "EGADS Model", (PetscObject)modelObj));
    }
    PetscCall(PetscContainerDestroy(&modelObj));

    PetscCall(PetscContainerCreate(PETSC_COMM_SELF, &contextObj));
    PetscCall(PetscContainerSetPointer(contextObj, context));

    if (islite) {
      PetscCall(PetscContainerSetCtxDestroy(contextObj, (PetscCtxDestroyFn *)DMPlexEGADSliteClose_Private));
      PetscCall(PetscObjectCompose((PetscObject)dm, "EGADSlite Context", (PetscObject)contextObj));
    } else {
      PetscCall(PetscContainerSetCtxDestroy(contextObj, (PetscCtxDestroyFn *)DMPlexEGADSClose_Private));
      PetscCall(PetscObjectCompose((PetscObject)dm, "EGADS Context", (PetscObject)contextObj));
    }
    PetscCall(PetscContainerDestroy(&contextObj));
  }

  // Label Points
  PetscCall(DMCreateLabel(dm, "EGADS Body ID"));
  PetscCall(DMGetLabel(dm, "EGADS Body ID", &bodyLabel));
  PetscCall(DMCreateLabel(dm, "EGADS Face ID"));
  PetscCall(DMGetLabel(dm, "EGADS Face ID", &faceLabel));
  PetscCall(DMCreateLabel(dm, "EGADS Edge ID"));
  PetscCall(DMGetLabel(dm, "EGADS Edge ID", &edgeLabel));
  PetscCall(DMCreateLabel(dm, "EGADS Vertex ID"));
  PetscCall(DMGetLabel(dm, "EGADS Vertex ID", &vertexLabel));

  /* Get Number of DAG Nodes at each level */
  int fStart, fEnd, eStart, eEnd, nStart, nEnd;

  PetscCall(DMPlexGetHeightStratum(dm, 0, &fStart, &fEnd));
  PetscCall(DMPlexGetHeightStratum(dm, 1, &eStart, &eEnd));
  PetscCall(DMPlexGetHeightStratum(dm, 2, &nStart, &nEnd));

  /* Set DMLabels for NODES */
  for (int n = nStart; n < nEnd; ++n) {
    int           pTypeVal, pIndexVal, pBodyVal;
    PetscHashIter PTLiter, PILiter, PBLiter;
    PetscBool     PTLfound, PILfound, PBLfound;

    //Converted to Hash Tables
    PetscCall(PetscHMapIFind(pTypeLabelMap, n - nStart, &PTLiter, &PTLfound));
    PetscCheck(PTLfound, PETSC_COMM_SELF, PETSC_ERR_SUP, "DAG Point %" PetscInt_FMT " not found in pTypeLabelMap", n);
    PetscCall(PetscHMapIGet(pTypeLabelMap, n - nStart, &pTypeVal));

    PetscCall(PetscHMapIFind(pIndexLabelMap, n - nStart, &PILiter, &PILfound));
    PetscCheck(PILfound, PETSC_COMM_SELF, PETSC_ERR_SUP, "DAG Point %" PetscInt_FMT " not found in pIndexLabelMap", n);
    PetscCall(PetscHMapIGet(pIndexLabelMap, n - nStart, &pIndexVal));

    PetscCall(PetscHMapIFind(pBodyIndexLabelMap, n - nStart, &PBLiter, &PBLfound));
    PetscCheck(PBLfound, PETSC_COMM_SELF, PETSC_ERR_SUP, "DAG Point %" PetscInt_FMT " not found in pBodyLabelMap", n);
    PetscCall(PetscHMapIGet(pBodyIndexLabelMap, n - nStart, &pBodyVal));

    PetscCall(DMLabelSetValue(bodyLabel, n, pBodyVal));
    if (pTypeVal == 0) PetscCall(DMLabelSetValue(vertexLabel, n, pIndexVal));
    if (pTypeVal > 0) PetscCall(DMLabelSetValue(edgeLabel, n, pIndexVal));
    if (pTypeVal < 0) PetscCall(DMLabelSetValue(faceLabel, n, pIndexVal));
  }

  /* Set DMLabels for Edges - Based on the DMLabels of the EDGE's NODES */
  for (int e = eStart; e < eEnd; ++e) {
    int bodyID_0, vertexID_0, vertexID_1, edgeID_0, edgeID_1, faceID_0, faceID_1;

    PetscCall(DMPlexGetCone(dm, e, &cone));
    PetscCall(DMLabelGetValue(bodyLabel, cone[0], &bodyID_0)); // Do I need to check the other end?
    PetscCall(DMLabelGetValue(vertexLabel, cone[0], &vertexID_0));
    PetscCall(DMLabelGetValue(vertexLabel, cone[1], &vertexID_1));
    PetscCall(DMLabelGetValue(edgeLabel, cone[0], &edgeID_0));
    PetscCall(DMLabelGetValue(edgeLabel, cone[1], &edgeID_1));
    PetscCall(DMLabelGetValue(faceLabel, cone[0], &faceID_0));
    PetscCall(DMLabelGetValue(faceLabel, cone[1], &faceID_1));

    PetscCall(DMLabelSetValue(bodyLabel, e, bodyID_0));

    if (edgeID_0 == edgeID_1) PetscCall(DMLabelSetValue(edgeLabel, e, edgeID_0));
    else if (vertexID_0 > 0 && edgeID_1 > 0) PetscCall(DMLabelSetValue(edgeLabel, e, edgeID_1));
    else if (vertexID_1 > 0 && edgeID_0 > 0) PetscCall(DMLabelSetValue(edgeLabel, e, edgeID_0));
    else { /* Do Nothing */ }
  }

  /* Set DMLabels for Cells */
  for (int f = fStart; f < fEnd; ++f) {
    int           edgeID_0;
    PetscInt      triBodyVal, triFaceVal;
    PetscHashIter TFLiter, TBLiter;
    PetscBool     TFLfound, TBLfound;

    // Convert to Hash Table
    PetscCall(PetscHMapIFind(triFaceIDLabelMap, f - fStart, &TFLiter, &TFLfound));
    PetscCheck(TFLfound, PETSC_COMM_SELF, PETSC_ERR_SUP, "DAG Point %" PetscInt_FMT " not found in triFaceIDLabelMap", f);
    PetscCall(PetscHMapIGet(triFaceIDLabelMap, f - fStart, &triFaceVal));

    PetscCall(PetscHMapIFind(triBodyIDLabelMap, f - fStart, &TBLiter, &TBLfound));
    PetscCheck(TBLfound, PETSC_COMM_SELF, PETSC_ERR_SUP, "DAG Point %" PetscInt_FMT " not found in triBodyIDLabelMap", f);
    PetscCall(PetscHMapIGet(triBodyIDLabelMap, f - fStart, &triBodyVal));

    PetscCall(DMLabelSetValue(bodyLabel, f, triBodyVal));
    PetscCall(DMLabelSetValue(faceLabel, f, triFaceVal));

    /* Finish Labeling previously unlabeled DMPlex Edges - Assumes Triangular Cell (3 Edges Max) */
    PetscCall(DMPlexGetCone(dm, f, &cone));

    for (int jj = 0; jj < 3; ++jj) {
      PetscCall(DMLabelGetValue(edgeLabel, cone[jj], &edgeID_0));

      if (edgeID_0 < 0) {
        PetscCall(DMLabelSetValue(bodyLabel, cone[jj], triBodyVal));
        PetscCall(DMLabelSetValue(faceLabel, cone[jj], triFaceVal));
      }
    }
  }

  *newdm = dm;
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

/*@C
  DMPlexInflateToGeomModelUseXYZ - Snaps the vertex coordinates of a `DMPLEX` object representing the mesh to its geometry if some vertices depart from the model. This usually happens with non-conforming refinement.

  Collective

  Input Parameter:
. dm - The uninflated `DM` object representing the mesh

  Level: intermediate

.seealso: [](ch_unstructured), `DM`, `DMPLEX`, `DMCreate()`, `DMPlexCreateEGADS()`
@*/
PetscErrorCode DMPlexInflateToGeomModelUseXYZ(DM dm) PeNS
{
  // please don't fucking write code like this with #ifdef all of the place!
#if defined(PETSC_HAVE_EGADS)
  /* EGADS Variables */
  ego    model, geom, body, face, edge, vertex;
  ego   *bodies;
  int    Nb, oclass, mtype, *senses;
  double result[4];
  /* PETSc Variables */
  DM             cdm;
  PetscContainer modelObj;
  DMLabel        bodyLabel, faceLabel, edgeLabel, vertexLabel;
  Vec            coordinates;
  PetscScalar   *coords;
  PetscInt       bodyID, faceID, edgeID, vertexID;
  PetscInt       cdim, d, vStart, vEnd, v;
  PetscBool      islite = PETSC_FALSE;
#endif

  PetscFunctionBegin;
#if defined(PETSC_HAVE_EGADS)
  PetscCall(PetscObjectQuery((PetscObject)dm, "EGADS Model", (PetscObject *)&modelObj));
  if (!modelObj) {
    PetscCall(PetscObjectQuery((PetscObject)dm, "EGADSlite Model", (PetscObject *)&modelObj));
    islite = PETSC_TRUE;
  }
  if (!modelObj) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(DMGetCoordinateDim(dm, &cdim));
  PetscCall(DMGetCoordinateDM(dm, &cdm));
  PetscCall(DMGetCoordinatesLocal(dm, &coordinates));
  PetscCall(DMGetLabel(dm, "EGADS Body ID", &bodyLabel));
  PetscCall(DMGetLabel(dm, "EGADS Face ID", &faceLabel));
  PetscCall(DMGetLabel(dm, "EGADS Edge ID", &edgeLabel));
  PetscCall(DMGetLabel(dm, "EGADS Vertex ID", &vertexLabel));

  PetscCall(PetscContainerGetPointer(modelObj, (void **)&model));

  if (islite) {
    PetscCall(EGlite_getTopology(model, &geom, &oclass, &mtype, NULL, &Nb, &bodies, &senses));
  } else {
    PetscCall(EG_getTopology(model, &geom, &oclass, &mtype, NULL, &Nb, &bodies, &senses));
  }

  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  PetscCall(VecGetArrayWrite(coordinates, &coords));
  for (v = vStart; v < vEnd; ++v) {
    PetscScalar *vcoords;

    PetscCall(DMLabelGetValue(bodyLabel, v, &bodyID));
    PetscCall(DMLabelGetValue(faceLabel, v, &faceID));
    PetscCall(DMLabelGetValue(edgeLabel, v, &edgeID));
    PetscCall(DMLabelGetValue(vertexLabel, v, &vertexID));

    PetscCheck(bodyID < Nb, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Body %" PetscInt_FMT " is not in [0, %" PetscInt_FMT ")", bodyID, Nb);
    body = bodies[bodyID];

    PetscCall(DMPlexPointLocalRef(cdm, v, coords, (void *)&vcoords));
    if (vertexID > 0) {
      if (islite) {
        PetscCall(EGlite_objectBodyTopo(body, NODE, vertexID, &vertex));
        PetscCall(EGlite_evaluate(vertex, NULL, result));
      } else {
        PetscCall(EG_objectBodyTopo(body, NODE, vertexID, &vertex));
        PetscCall(EG_evaluate(vertex, NULL, result));
      }
      for (d = 0; d < cdim; ++d) vcoords[d] = result[d];
    } else if (edgeID > 0) {
      /* Snap to EDGE at nearest location */
      double params[1];
      if (islite) {
        PetscCall(EGlite_objectBodyTopo(body, EDGE, edgeID, &edge));
        PetscCall(EGlite_invEvaluate(edge, vcoords, params, result));
      } // Get (x,y,z) of nearest point on EDGE
      else {
        PetscCall(EG_objectBodyTopo(body, EDGE, edgeID, &edge));
        PetscCall(EG_invEvaluate(edge, vcoords, params, result));
      }
      for (d = 0; d < cdim; ++d) vcoords[d] = result[d];
    } else if (faceID > 0) {
      /* Snap to FACE at nearest location */
      double params[2];
      if (islite) {
        PetscCall(EGlite_objectBodyTopo(body, FACE, faceID, &face));
        PetscCall(EGlite_invEvaluate(face, vcoords, params, result));
      } // Get (x,y,z) of nearest point on FACE
      else {
        PetscCall(EG_objectBodyTopo(body, FACE, faceID, &face));
        PetscCall(EG_invEvaluate(face, vcoords, params, result));
      }
      for (d = 0; d < cdim; ++d) vcoords[d] = result[d];
    }
  }
  PetscCall(VecRestoreArrayWrite(coordinates, &coords));
  /* Clear out global coordinates */
  PetscCall(VecDestroy(&dm->coordinates[0].x));
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if defined(PETSC_HAVE_EGADS)
// This replaces the model in-place
PetscErrorCode ConvertGeomModelToAllBSplines(PetscBool islite, ego *model) PeNS
{
  /* EGADS/EGADSlite Variables */
  ego  context = NULL, geom, *bodies, *fobjs;
  int  oclass, mtype;
  int *senses;
  int  Nb, Nf;

  PetscFunctionBegin;
  // Get the number of bodies and body objects in the model
  if (islite) PetscCallEGADS(EGlite_getTopology, (*model, &geom, &oclass, &mtype, NULL, &Nb, &bodies, &senses));
  else PetscCallEGADS(EG_getTopology, (*model, &geom, &oclass, &mtype, NULL, &Nb, &bodies, &senses));

  // Get all Faces on the body    <-- Only working with 1 body at the moment.
  ego body = bodies[0];
  if (islite) PetscCallEGADS(EGlite_getBodyTopos, (body, NULL, FACE, &Nf, &fobjs));
  else PetscCallEGADS(EG_getBodyTopos, (body, NULL, FACE, &Nf, &fobjs));
  ego newGeom[Nf];
  ego newFaces[Nf];

  // Convert the 1st Face to a BSpline Geometry
  for (int ii = 0; ii < Nf; ++ii) {
    ego     face = fobjs[ii];
    ego     gRef, gPrev, gNext, *lobjs;
    int     goclass, gmtype, *gpinfo;
    int     Nl, *lsenses;
    double *gprv;
    char   *gClass = (char *)"", *gType = (char *)"";

    /* Shape Optimization is NOT available for EGADSlite geometry files. */
    /*     Note :: islite options are left below in case future versions of EGADSlite includes this capability */
    PetscCheck(!islite, PETSC_COMM_SELF, PETSC_ERR_SUP, "Cannot convert geometric entities to all BSplines for geometries defined by EGADSlite (.egadslite)! Please use another geometry file format STEP, IGES, EGADS or BRep");

    if (islite) {
      PetscCallEGADS(EGlite_getTopology, (face, &geom, &oclass, &mtype, NULL, &Nl, &lobjs, &lsenses)); // Get FACES Geometry object (geom_
      PetscCallEGADS(EGlite_getGeometry, (geom, &goclass, &gmtype, &gRef, &gpinfo, &gprv));            // Get geometry object info
      PetscCallEGADS(EGlite_getInfo, (geom, &goclass, &gmtype, &gRef, &gPrev, &gNext));
    } // Get geometry info
    else {
      PetscCallEGADS(EG_getTopology, (face, &geom, &oclass, &mtype, NULL, &Nl, &lobjs, &lsenses)); // Get FACES Geometry object (geom_
      PetscCallEGADS(EG_getGeometry, (geom, &goclass, &gmtype, &gRef, &gpinfo, &gprv));            // Get geometry object info
      PetscCallEGADS(EG_getInfo, (geom, &goclass, &gmtype, &gRef, &gPrev, &gNext));
    } // Get geometry info

    PetscCall(DMPlex_EGADS_GeomDecode_Internal(goclass, gmtype, &gClass, &gType)); // Decode Geometry integers

    // Convert current FACE to a BSpline Surface
    ego     bspline;
    ego     bRef, bPrev, bNext;
    int     boclass, bmtype, *bpinfo;
    double *bprv;
    char   *bClass = (char *)"", *bType = (char *)"";

    PetscCallEGADS(EG_convertToBSpline, (face, &bspline)); // Does not have an EGlite_ version

    if (islite) {
      PetscCallEGADS(EGlite_getGeometry, (bspline, &boclass, &bmtype, &bRef, &bpinfo, &bprv)); // Get geometry object info
      PetscCallEGADS(EGlite_getInfo, (bspline, &boclass, &bmtype, &bRef, &bPrev, &bNext));
    } // Get geometry info
    else {
      PetscCallEGADS(EG_getGeometry, (bspline, &boclass, &bmtype, &bRef, &bpinfo, &bprv)); // Get geometry object info
      PetscCallEGADS(EG_getInfo, (bspline, &boclass, &bmtype, &bRef, &bPrev, &bNext));
    } // Get geometry info

    PetscCall(DMPlex_EGADS_GeomDecode_Internal(boclass, bmtype, &bClass, &bType)); // Decode Geometry integers

    // Get Context from FACE
    context = NULL;
    PetscCallEGADS(EG_getContext, (face, &context)); // Does not have an EGlite_ version

    // Silence WARNING Regarding OPENCASCADE 7.5
    if (islite) PetscCallEGADS(EGlite_setOutLevel, (context, 0));
    else PetscCallEGADS(EG_setOutLevel, (context, 0));

    ego newgeom;
    PetscCallEGADS(EG_makeGeometry, (context, SURFACE, BSPLINE, NULL, bpinfo, bprv, &newgeom)); // Does not have an EGlite_ version

    PetscCallEGADS(EG_deleteObject, (bspline));

    // Create new FACE based on new SURFACE geometry
    double data[4];
    int    periodic;
    if (islite) PetscCallEGADS(EGlite_getRange, (newgeom, data, &periodic));
    else PetscCallEGADS(EG_getRange, (newgeom, data, &periodic));

    ego newface;
    PetscCallEGADS(EG_makeFace, (newgeom, SFORWARD, data, &newface)); // Does not have an EGlite_ version
    //PetscCallEGADS(EG_deleteObject, (newgeom));
    //PetscCallEGADS(EG_deleteObject, (newface));
    newFaces[ii] = newface;
    newGeom[ii]  = newgeom;

    // Reinstate WARNING Regarding OPENCASCADE 7.5
    if (islite) PetscCallEGADS(EGlite_setOutLevel, (context, 1));
    else PetscCallEGADS(EG_setOutLevel, (context, 1));
  }

  // Sew New Faces together to get a new model
  ego newmodel;
  PetscCallEGADS(EG_sewFaces, (Nf, newFaces, 0.0, 0, &newmodel)); // Does not have an EGlite_ version
  for (int ii = 0; ii < Nf; ++ii) {
    PetscCallEGADS(EG_deleteObject, (newFaces[ii]));
    PetscCallEGADS(EG_deleteObject, (newGeom[ii]));
  }
  PetscCallEGADS(EG_deleteObject, (*model));
  *model = newmodel;
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

/*@C
  DMPlexCreateGeomFromFile - Create a `DMPLEX` mesh from an EGADS, IGES, or STEP file.

  Collective

  Input Parameters:
+ comm     - The MPI communicator
. filename - The name of the EGADS, IGES, or STEP file
- islite   - Flag for EGADSlite support

  Output Parameter:
. dm - The `DM` object representing the mesh

  Level: beginner

.seealso: [](ch_unstructured), `DM`, `DMPLEX`, `DMCreate()`, `DMPlexCreateEGADS()`, `DMPlexCreateEGADSliteFromFile()`
@*/
PetscErrorCode DMPlexCreateGeomFromFile(MPI_Comm comm, const char filename[], DM *dm, PetscBool islite) PeNS
{
  /* PETSc Variables */
  PetscMPIInt rank;
  PetscBool   printModel = PETSC_FALSE, tessModel = PETSC_FALSE, newModel = PETSC_FALSE;
  PetscBool   shapeOpt = PETSC_FALSE;

#if defined(PETSC_HAVE_EGADS)
  ego context = NULL, model = NULL;
#endif

  PetscFunctionBegin;
  PetscAssertPointer(filename, 2);
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-dm_plex_geom_print_model", &printModel, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-dm_plex_geom_tess_model", &tessModel, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-dm_plex_geom_new_model", &newModel, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-dm_plex_geom_shape_opt", &shapeOpt, NULL));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
#if defined(PETSC_HAVE_EGADS)
  if (rank == 0) {
    /* EGADSlite files cannot be used for Shape Optimization Work. It lacks the ability to make new geometry. */
    /* Must use EGADS, STEP, IGES or BRep files to perform this work.                                         */
    if (islite) {
      PetscCallEGADS(EGlite_open, (&context));
      PetscCallEGADS(EGlite_loadModel, (context, 0, filename, &model));
      if (shapeOpt) PetscCall(ConvertGeomModelToAllBSplines(islite, &model));
      if (printModel) PetscCall(DMPlexGeomPrintModel_Internal(model, islite));
    } else {
      PetscCallEGADS(EG_open, (&context));
      PetscCallEGADS(EG_loadModel, (context, 0, filename, &model));
      if (shapeOpt) PetscCall(ConvertGeomModelToAllBSplines(islite, &model));
      if (printModel) PetscCall(DMPlexGeomPrintModel_Internal(model, islite));
    }
  }
  if (tessModel) PetscCall(DMPlexCreateGeom_Tess_Internal(comm, context, model, dm, islite));
  else if (newModel) PetscCall(DMPlexCreateGeom_Internal(comm, context, model, dm, islite));
  else {
    PetscCall(DMPlexCreateGeom(comm, context, model, dm, islite));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
#else
  SETERRQ(comm, PETSC_ERR_SUP, "This method requires EGADS support. Reconfigure using --download-egads");
#endif
}

#if defined(PETSC_HAVE_EGADS)
/*@C
  DMPlex_Surface_Grad - Exposes the Geometry's Control Points and Weights and Calculates the Mesh Topology Boundary Nodes Gradient
                        with respect the associated geometry's Control Points and Weights.

                        // ----- Depreciated ---- See DMPlexGeomDataAndGrads ------ //

  Collective

  Input Parameters:
. dm      - The DM object representing the mesh with PetscContainer containing an EGADS geometry model

  Output Parameter:
. dm       - The DM object representing the mesh with PetscContainers containing the EGADS geometry model, Array-Hash Table Geometry Control Point Pair, Array-Hash Table Geometry Weights Pair and Matrix-Hash Table Surface Gradient Pair

  Level: intermediate

.seealso:
@*/
PetscErrorCode DMPlex_Surface_Grad(DM dm)
{
  ego            model, geom, *bodies, *fobjs;
  PetscContainer modelObj;
  int            oclass, mtype, *senses;
  int            Nb, Nf;
  PetscHMapI     faceCntrlPtRow_Start = NULL, faceCPWeightsRow_Start = NULL;
  PetscHMapI     pointSurfGradRow_Start = NULL;
  Mat            pointSurfGrad;
  IS             faceLabelValues, edgeLabelValues, vertexLabelValues;
  PetscInt       faceLabelSize, edgeLabelSize, vertexLabelSize;
  PetscBool      islite = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(PetscObjectQuery((PetscObject)dm, "EGADS Model", (PetscObject *)&modelObj));
  if (!modelObj) {
    PetscCall(PetscObjectQuery((PetscObject)dm, "EGADSlite Model", (PetscObject *)&modelObj));
    islite = PETSC_TRUE;
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, " Cannot provide geometric data or associated calculated gradients for geometries defined by EGADSlite (.egadslite)! \n Please use another geometry file format STEP, IGES, EGADS or BRep");
  }

  // Get attached EGADS model (pointer)
  PetscCall(PetscContainerGetPointer(modelObj, (void **)&model));

  // Get the bodies in the model
  if (islite) {
    PetscCall(EGlite_getTopology(model, &geom, &oclass, &mtype, NULL, &Nb, &bodies, &senses));
  } else {
    PetscCall(EG_getTopology(model, &geom, &oclass, &mtype, NULL, &Nb, &bodies, &senses));
  }

  ego body = bodies[0]; // Only operate on 1st body. Model should only have 1 body.

  // Get the total number of FACEs in the model
  if (islite) {
    PetscCall(EGlite_getBodyTopos(body, NULL, FACE, &Nf, &fobjs));
  } else {
    PetscCall(EG_getBodyTopos(body, NULL, FACE, &Nf, &fobjs));
  }

  // Get the total number of points and IDs in the DMPlex with a "EGADS Face Label"
  // This will provide the total number of DMPlex points on the boundary of the geometry
  PetscCall(DMGetLabelIdIS(dm, "EGADS Face ID", &faceLabelValues));
  PetscCall(DMGetLabelSize(dm, "EGADS Face ID", &faceLabelSize));

  PetscCall(DMGetLabelIdIS(dm, "EGADS Edge ID", &edgeLabelValues));
  PetscCall(DMGetLabelSize(dm, "EGADS Edge ID", &edgeLabelSize));

  PetscCall(DMGetLabelIdIS(dm, "EGADS Vertex ID", &vertexLabelValues));
  PetscCall(DMGetLabelSize(dm, "EGADS Vertex ID", &vertexLabelSize));

  const PetscInt *faceIndices, *edgeIndices, *vertexIndices;
  PetscCall(ISGetIndices(faceLabelValues, &faceIndices));
  PetscCall(ISGetIndices(edgeLabelValues, &edgeIndices));
  PetscCall(ISGetIndices(vertexLabelValues, &vertexIndices));

  // Get the points associated with each FACE, EDGE and VERTEX label in the DM
  PetscInt totalNumPoints = 0;
  for (int ii = 0; ii < faceLabelSize; ++ii) {
    // Cycle through FACE labels
    PetscInt size;
    PetscCall(DMGetStratumSize(dm, "EGADS Face ID", faceIndices[ii], &size));
    totalNumPoints += size;
  }
  PetscCall(ISRestoreIndices(faceLabelValues, &faceIndices));
  PetscCall(ISDestroy(&faceLabelValues));

  for (int ii = 0; ii < edgeLabelSize; ++ii) {
    // Cycle Through EDGE Labels
    PetscInt size;
    PetscCall(DMGetStratumSize(dm, "EGADS Edge ID", edgeIndices[ii], &size));
    totalNumPoints += size;
  }
  PetscCall(ISRestoreIndices(edgeLabelValues, &edgeIndices));
  PetscCall(ISDestroy(&edgeLabelValues));

  for (int ii = 0; ii < vertexLabelSize; ++ii) {
    // Cycle Through VERTEX Labels
    PetscInt size;
    PetscCall(DMGetStratumSize(dm, "EGADS Vertex ID", vertexIndices[ii], &size));
    totalNumPoints += size;
  }
  PetscCall(ISRestoreIndices(vertexLabelValues, &vertexIndices));
  PetscCall(ISDestroy(&vertexLabelValues));

  int     maxNumCPs   = 0;
  int     totalNumCPs = 0;
  ego     bRef, bPrev, bNext, fgeom, *lobjs;
  int     id, boclass, bmtype, *bpinfo;
  int     foclass, fmtype, Nl, *lsenses;
  double *bprv;
  double  fdata[4];

  // Create Hash Tables
  PetscInt cntr = 0, wcntr = 0;
  PetscCall(PetscHMapICreate(&faceCntrlPtRow_Start));
  PetscCall(PetscHMapICreate(&faceCPWeightsRow_Start));

  for (int ii = 0; ii < Nf; ++ii) {
    // Need to get the maximum number of Control Points defining the FACEs
    ego face = fobjs[ii];
    int maxNumCPs_temp;

    if (islite) {
      id = EGlite_indexBodyTopo(body, face);
      PetscCall(EGlite_getTopology(face, &fgeom, &foclass, &fmtype, fdata, &Nl, &lobjs, &lsenses));
      PetscCall(EGlite_getGeometry(fgeom, &boclass, &bmtype, &bRef, &bpinfo, &bprv));
      PetscCall(EGlite_getInfo(fgeom, &boclass, &bmtype, &bRef, &bPrev, &bNext));
    } else {
      id = EG_indexBodyTopo(body, face);
      PetscCall(EG_getTopology(face, &fgeom, &foclass, &fmtype, fdata, &Nl, &lobjs, &lsenses));
      PetscCall(EG_getGeometry(fgeom, &boclass, &bmtype, &bRef, &bpinfo, &bprv));
      PetscCall(EG_getInfo(fgeom, &boclass, &bmtype, &bRef, &bPrev, &bNext));
    }

    maxNumCPs_temp = bpinfo[2] * bpinfo[5];
    totalNumCPs += bpinfo[2] * bpinfo[5];

    if (maxNumCPs_temp > maxNumCPs) { maxNumCPs = maxNumCPs_temp; }
  }

  PetscInt *cpCoordDataLengthPtr, *wDataLengthPtr;
  PetscInt  cpCoordDataLength = 3 * totalNumCPs;
  PetscInt  wDataLength       = totalNumCPs;
  cpCoordDataLengthPtr        = &cpCoordDataLength;
  wDataLengthPtr              = &wDataLength;
  PetscScalar *cntrlPtCoords, *cntrlPtWeights;
  PetscMalloc1(cpCoordDataLength, &cntrlPtCoords);
  PetscMalloc1(wDataLength, &cntrlPtWeights);
  for (int ii = 0; ii < Nf; ++ii) {
    // Need to Populate Control Point Coordinates and Weight Vectors
    ego           face = fobjs[ii];
    PetscHashIter hashKeyIter, wHashKeyIter;
    PetscBool     hashKeyFound, wHashKeyFound;

    if (islite) {
      id = EGlite_indexBodyTopo(body, face);
      PetscCall(EGlite_getTopology(face, &fgeom, &foclass, &fmtype, fdata, &Nl, &lobjs, &lsenses));
      PetscCall(EGlite_getGeometry(fgeom, &boclass, &bmtype, &bRef, &bpinfo, &bprv));
      PetscCall(EGlite_getInfo(fgeom, &boclass, &bmtype, &bRef, &bPrev, &bNext));
    } else {
      id = EG_indexBodyTopo(body, face);
      PetscCall(EG_getTopology(face, &fgeom, &foclass, &fmtype, fdata, &Nl, &lobjs, &lsenses));
      PetscCall(EG_getGeometry(fgeom, &boclass, &bmtype, &bRef, &bpinfo, &bprv));
      PetscCall(EG_getInfo(fgeom, &boclass, &bmtype, &bRef, &bPrev, &bNext));
    }

    // Store Face ID to 1st Row of Control Point Vector
    PetscCall(PetscHMapIFind(faceCntrlPtRow_Start, id, &hashKeyIter, &hashKeyFound));

    if (!hashKeyFound) { PetscCall(PetscHMapISet(faceCntrlPtRow_Start, id, cntr)); }

    int offsetCoord = bpinfo[3] + bpinfo[6];
    for (int jj = 0; jj < 3 * bpinfo[2] * bpinfo[5]; ++jj) {
      cntrlPtCoords[cntr] = bprv[offsetCoord + jj];
      cntr += 1;
    }

    // Store Face ID to 1st Row of Control Point Weight Vector
    PetscCall(PetscHMapIFind(faceCPWeightsRow_Start, id, &wHashKeyIter, &wHashKeyFound));

    if (!wHashKeyFound) { PetscCall(PetscHMapISet(faceCPWeightsRow_Start, id, wcntr)); }

    int offsetWeight = bpinfo[3] + bpinfo[6] + (3 * bpinfo[2] * bpinfo[5]);
    for (int jj = 0; jj < bpinfo[2] * bpinfo[5]; ++jj) {
      cntrlPtWeights[wcntr] = bprv[offsetWeight + jj];
      wcntr += 1;
    }
  }

  // Attach Control Point and Weight Data to DM
  {
    PetscContainer cpOrgObj, cpCoordObj, cpCoordLengthObj;
    PetscContainer wOrgObj, wValObj, wDataLengthObj;

    PetscCall(PetscContainerCreate(PETSC_COMM_SELF, &cpOrgObj));
    PetscCall(PetscContainerSetPointer(cpOrgObj, faceCntrlPtRow_Start));
    PetscCall(PetscObjectCompose((PetscObject)dm, "Control Point Hash Table", (PetscObject)cpOrgObj));
    PetscCall(PetscContainerDestroy(&cpOrgObj));

    PetscCall(PetscContainerCreate(PETSC_COMM_SELF, &cpCoordObj));
    PetscCall(PetscContainerSetPointer(cpCoordObj, cntrlPtCoords));
    PetscCall(PetscObjectCompose((PetscObject)dm, "Control Point Coordinates", (PetscObject)cpCoordObj));
    PetscCall(PetscContainerDestroy(&cpCoordObj));

    PetscCall(PetscContainerCreate(PETSC_COMM_SELF, &cpCoordLengthObj));
    PetscCall(PetscContainerSetPointer(cpCoordLengthObj, cpCoordDataLengthPtr));
    PetscCall(PetscObjectCompose((PetscObject)dm, "Control Point Coordinate Data Length", (PetscObject)cpCoordLengthObj));
    PetscCall(PetscContainerDestroy(&cpCoordLengthObj));

    PetscCall(PetscContainerCreate(PETSC_COMM_SELF, &wOrgObj));
    PetscCall(PetscContainerSetPointer(wOrgObj, faceCPWeightsRow_Start));
    PetscCall(PetscObjectCompose((PetscObject)dm, "Control Point Weights Hash Table", (PetscObject)wOrgObj));
    PetscCall(PetscContainerDestroy(&wOrgObj));

    PetscCall(PetscContainerCreate(PETSC_COMM_SELF, &wValObj));
    PetscCall(PetscContainerSetPointer(wValObj, cntrlPtWeights));
    PetscCall(PetscObjectCompose((PetscObject)dm, "Control Point Weight Data", (PetscObject)wValObj));
    PetscCall(PetscContainerDestroy(&wValObj));

    PetscCall(PetscContainerCreate(PETSC_COMM_SELF, &wDataLengthObj));
    PetscCall(PetscContainerSetPointer(wDataLengthObj, wDataLengthPtr));
    PetscCall(PetscObjectCompose((PetscObject)dm, "Control Point Weight Data Length", (PetscObject)wDataLengthObj));
    PetscCall(PetscContainerDestroy(&wDataLengthObj));
  }

  // Define Matrix to store  Surface Gradient information dx_i/dCPj_i
  PetscInt       gcntr   = 0;
  const PetscInt rowSize = 3 * maxNumCPs * totalNumPoints;
  const PetscInt colSize = 4 * Nf;

  // Create Point Surface Gradient Matrix
  MatCreate(PETSC_COMM_WORLD, &pointSurfGrad);
  MatSetSizes(pointSurfGrad, PETSC_DECIDE, PETSC_DECIDE, rowSize, colSize);
  MatSetType(pointSurfGrad, MATAIJ);
  MatSetUp(pointSurfGrad);

  // Create Hash Table to store Point's stare row in surfaceGrad[][]
  PetscCall(PetscHMapICreate(&pointSurfGradRow_Start));

  // Get Coordinates for the DMPlex point
  DM           cdm;
  PetscInt     dE, Nv;
  Vec          coordinatesLocal;
  PetscScalar *coords = NULL;
  PetscCall(DMGetCoordinateDM(dm, &cdm));
  PetscCall(DMGetCoordinateDim(dm, &dE));
  PetscCall(DMGetCoordinatesLocal(dm, &coordinatesLocal));

  // CYCLE THROUGH FACEs
  for (int ii = 0; ii < Nf; ++ii) {
    ego             face = fobjs[ii];
    ego            *eobjs, *nobjs;
    PetscInt        fid, Ne, Nn;
    DMLabel         faceLabel, edgeLabel, nodeLabel;
    PetscHMapI      currFaceUniquePoints = NULL;
    IS              facePoints, edgePoints, nodePoints;
    const PetscInt *fIndices, *eIndices, *nIndices;
    PetscInt        fSize, eSize, nSize;
    PetscHashIter   fHashKeyIter, eHashKeyIter, nHashKeyIter, pHashKeyIter;
    PetscBool       fHashKeyFound, eHashKeyFound, nHashKeyFound, pHashKeyFound;
    PetscInt        cfCntr = 0;

    // Get Geometry Object for the Current FACE
    if (islite) {
      PetscCall(EGlite_getTopology(face, &fgeom, &foclass, &fmtype, fdata, &Nl, &lobjs, &lsenses));
      PetscCall(EGlite_getGeometry(fgeom, &boclass, &bmtype, &bRef, &bpinfo, &bprv));
    } else {
      PetscCall(EG_getTopology(face, &fgeom, &foclass, &fmtype, fdata, &Nl, &lobjs, &lsenses));
      PetscCall(EG_getGeometry(fgeom, &boclass, &bmtype, &bRef, &bpinfo, &bprv));
    }

    // Get all EDGE and NODE objects attached to the current FACE
    if (islite) {
      PetscCall(EGlite_getBodyTopos(body, face, EDGE, &Ne, &eobjs));
      PetscCall(EGlite_getBodyTopos(body, face, NODE, &Nn, &nobjs));
    } else {
      PetscCall(EG_getBodyTopos(body, face, EDGE, &Ne, &eobjs));
      PetscCall(EG_getBodyTopos(body, face, NODE, &Nn, &nobjs));
    }

    // Get all DMPlex Points that have DMLabel "EGADS Face ID" and store them in a Hash Table for later use
    if (islite) {
      fid = EGlite_indexBodyTopo(body, face);
    } else {
      fid = EG_indexBodyTopo(body, face);
    }

    PetscCall(DMGetLabel(dm, "EGADS Face ID", &faceLabel));
    PetscCall(DMLabelGetStratumIS(faceLabel, fid, &facePoints));
    PetscCall(ISGetIndices(facePoints, &fIndices));
    PetscCall(ISGetSize(facePoints, &fSize));

    PetscCall(PetscHMapICreate(&currFaceUniquePoints));

    for (int jj = 0; jj < fSize; ++jj) {
      PetscCall(PetscHMapIFind(currFaceUniquePoints, fIndices[jj], &fHashKeyIter, &fHashKeyFound));

      if (!fHashKeyFound) {
        PetscCall(PetscHMapISet(currFaceUniquePoints, fIndices[jj], cfCntr));
        cfCntr += 1;
      }

      PetscCall(PetscHMapIFind(pointSurfGradRow_Start, fIndices[jj], &pHashKeyIter, &pHashKeyFound));

      if (!pHashKeyFound) {
        PetscCall(PetscHMapISet(pointSurfGradRow_Start, fIndices[jj], gcntr));
        gcntr += 3 * maxNumCPs;
      }
    }
    PetscCall(ISRestoreIndices(facePoints, &fIndices));
    PetscCall(ISDestroy(&facePoints));

    // Get all DMPlex Points that have DMLable "EGADS Edge ID" attached to the current FACE and store them in a Hash Table for later use.
    for (int jj = 0; jj < Ne; ++jj) {
      ego       edge = eobjs[jj];
      PetscBool containLabelValue;

      if (islite) {
        id = EGlite_indexBodyTopo(body, edge);
      } else {
        id = EG_indexBodyTopo(body, edge);
      }

      PetscCall(DMGetLabel(dm, "EGADS Edge ID", &edgeLabel));
      PetscCall(DMLabelHasValue(edgeLabel, id, &containLabelValue));

      if (containLabelValue) {
        PetscCall(DMLabelGetStratumIS(edgeLabel, id, &edgePoints));
        PetscCall(ISGetIndices(edgePoints, &eIndices));
        PetscCall(ISGetSize(edgePoints, &eSize));

        for (int kk = 0; kk < eSize; ++kk) {
          PetscCall(PetscHMapIFind(currFaceUniquePoints, eIndices[kk], &eHashKeyIter, &eHashKeyFound));

          if (!eHashKeyFound) {
            PetscCall(PetscHMapISet(currFaceUniquePoints, eIndices[kk], cfCntr));
            cfCntr += 1;
          }

          PetscCall(PetscHMapIFind(pointSurfGradRow_Start, eIndices[kk], &pHashKeyIter, &pHashKeyFound));

          if (!pHashKeyFound) {
            PetscCall(PetscHMapISet(pointSurfGradRow_Start, eIndices[kk], gcntr));
            gcntr += 3 * maxNumCPs;
          }
        }
        PetscCall(ISRestoreIndices(edgePoints, &eIndices));
        PetscCall(ISDestroy(&edgePoints));
      }
    }

    // Get all DMPlex Points that have DMLabel "EGADS Vertex ID" attached to the current FACE and store them in a Hash Table for later use.
    for (int jj = 0; jj < Nn; ++jj) {
      ego node = nobjs[jj];

      if (islite) {
        id = EGlite_indexBodyTopo(body, node);
      } else {
        id = EG_indexBodyTopo(body, node);
      }

      PetscCall(DMGetLabel(dm, "EGADS Vertex ID", &nodeLabel));
      PetscCall(DMLabelGetStratumIS(nodeLabel, id, &nodePoints));
      PetscCall(ISGetIndices(nodePoints, &nIndices));
      PetscCall(ISGetSize(nodePoints, &nSize));

      for (int kk = 0; kk < nSize; ++kk) {
        PetscCall(PetscHMapIFind(currFaceUniquePoints, nIndices[kk], &nHashKeyIter, &nHashKeyFound));

        if (!nHashKeyFound) {
          PetscCall(PetscHMapISet(currFaceUniquePoints, nIndices[kk], cfCntr));
          cfCntr += 1;
        }

        PetscCall(PetscHMapIFind(pointSurfGradRow_Start, nIndices[kk], &pHashKeyIter, &pHashKeyFound));
        if (!pHashKeyFound) {
          PetscCall(PetscHMapISet(pointSurfGradRow_Start, nIndices[kk], gcntr));
          gcntr += 3 * maxNumCPs;
        }
      }
      PetscCall(ISRestoreIndices(nodePoints, &nIndices));
      PetscCall(ISDestroy(&nodePoints));
    }

    // Get the Total Number of entries in the Hash Table
    PetscInt currFaceUPSize;
    PetscCall(PetscHMapIGetSize(currFaceUniquePoints, &currFaceUPSize));

    // Get Keys
    PetscInt currFaceUPKeys[currFaceUPSize], off = 0;
    PetscCall(PetscHMapIGetKeys(currFaceUniquePoints, &off, currFaceUPKeys));

    // Cycle through all points on the current FACE
    for (int jj = 0; jj < currFaceUPSize; ++jj) {
      PetscInt currPointID = currFaceUPKeys[jj];
      PetscCall(DMPlexVecGetClosure(cdm, NULL, coordinatesLocal, currPointID, &Nv, &coords));

      // Get UV position of FACE
      double params[2], range[4], eval[18];
      int    peri;

      if (islite) {
        PetscCall(EGlite_getRange(face, range, &peri));
      } else {
        PetscCall(EG_getRange(face, range, &peri));
      }

      PetscCall(DMPlex_Geom_FACE_XYZtoUV_Internal(coords, face, range, 0, dE, params, islite));

      if (islite) {
        PetscCall(EGlite_evaluate(face, params, eval));
      } else {
        PetscCall(EG_evaluate(face, params, eval));
      }

      // Make a new SURFACE Geometry by changing the location of the Control Points
      int    prvSize = bpinfo[3] + bpinfo[6] + (4 * bpinfo[2] * bpinfo[5]);
      double nbprv[prvSize];

      // Cycle through each Control Point
      double deltaCoord = 1.0E-4;
      int    offset     = bpinfo[3] + bpinfo[6];
      int    wOffset    = offset + (3 * bpinfo[2] * bpinfo[5]);
      for (int ii = 0; ii < bpinfo[2] * bpinfo[5]; ++ii) {
        // Cycle through each direction (x, then y, then z)
        for (int kk = 0; kk < 4; ++kk) {
          // Reinitialize nbprv[] values because we only want to change one value at a time
          for (int mm = 0; mm < prvSize; ++mm) { nbprv[mm] = bprv[mm]; }

          if (kk == 0) { //X
            nbprv[offset + 0] = bprv[offset + 0] + deltaCoord;
            nbprv[offset + 1] = bprv[offset + 1];
            nbprv[offset + 2] = bprv[offset + 2];
          } else if (kk == 1) { //Y
            nbprv[offset + 0] = bprv[offset + 0];
            nbprv[offset + 1] = bprv[offset + 1] + deltaCoord;
            nbprv[offset + 2] = bprv[offset + 2];
          } else if (kk == 2) { //Z
            nbprv[offset + 0] = bprv[offset + 0];
            nbprv[offset + 1] = bprv[offset + 1];
            nbprv[offset + 2] = bprv[offset + 2] + deltaCoord;
          } else if (kk == 3) { // Weights
            nbprv[wOffset + ii] = bprv[wOffset + ii] + deltaCoord;
          } else {
            // currently do nothing
          }

          // Create New Surface Based on New Control Points or Weights
          ego newgeom, context;
          if (islite) {
            PetscCall(EGlite_open(&context));
            PetscCall(EGlite_setOutLevel(context, 0));
          } else {
            PetscCall(EG_open(&context));
            PetscCall(EG_setOutLevel(context, 0));
          }

          PetscCall(EG_makeGeometry(context, SURFACE, BSPLINE, NULL, bpinfo, nbprv, &newgeom)); // Does not have an EGlite_ version KNOWN_ISSUE

          if (islite) {
            PetscCall(EGlite_setOutLevel(context, 1));
          } else {
            PetscCall(EG_setOutLevel(context, 1));
          }

          // Evaluate new (x, y, z) Point Position based on new Surface Definition
          double newCoords[18];
          if (islite) {
            PetscCall(EGlite_getRange(newgeom, range, &peri));
          } else {
            PetscCall(EG_getRange(newgeom, range, &peri));
          }

          PetscCall(DMPlex_Geom_FACE_XYZtoUV_Internal(coords, newgeom, range, 0, dE, params, islite));

          if (islite) {
            PetscCall(EGlite_evaluate(newgeom, params, newCoords));
          } else {
            PetscCall(EG_evaluate(newgeom, params, newCoords));
          }

          // Now Calculate the Surface Gradient for the change in x-component Control Point
          PetscScalar dxdCx = (newCoords[0] - coords[0]) / deltaCoord;
          PetscScalar dxdCy = (newCoords[1] - coords[1]) / deltaCoord;
          PetscScalar dxdCz = (newCoords[2] - coords[2]) / deltaCoord;

          // Store Gradient Information in surfaceGrad[][] Matrix
          PetscInt startRow;
          PetscCall(PetscHMapIGet(pointSurfGradRow_Start, currPointID, &startRow));

          // Store Results in PETSc Mat
          PetscCall(MatSetValue(pointSurfGrad, startRow + (ii * 3) + 0, ((fid - 1) * 4) + kk, dxdCx, INSERT_VALUES));
          PetscCall(MatSetValue(pointSurfGrad, startRow + (ii * 3) + 1, ((fid - 1) * 4) + kk, dxdCy, INSERT_VALUES));
          PetscCall(MatSetValue(pointSurfGrad, startRow + (ii * 3) + 2, ((fid - 1) * 4) + kk, dxdCz, INSERT_VALUES));
        }
        offset += 3;
      }
      PetscCall(DMPlexVecRestoreClosure(cdm, NULL, coordinatesLocal, currPointID, &Nv, &coords));
    }
  }

  // Assemble Point Surface Grad Matrix
  MatAssemblyBegin(pointSurfGrad, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(pointSurfGrad, MAT_FINAL_ASSEMBLY);

  // Attach Surface Gradient Hash Table and Matrix to DM
  {
    PetscContainer surfGradOrgObj, surfGradObj;

    PetscCall(PetscContainerCreate(PETSC_COMM_SELF, &surfGradOrgObj));
    PetscCall(PetscContainerSetPointer(surfGradOrgObj, pointSurfGradRow_Start));
    PetscCall(PetscObjectCompose((PetscObject)dm, "Surface Gradient Hash Table", (PetscObject)surfGradOrgObj));
    PetscCall(PetscContainerDestroy(&surfGradOrgObj));

    PetscCall(PetscContainerCreate(PETSC_COMM_SELF, &surfGradObj));
    PetscCall(PetscContainerSetPointer(surfGradObj, pointSurfGrad));
    PetscCall(PetscObjectCompose((PetscObject)dm, "Surface Gradient Matrix", (PetscObject)surfGradObj));
    PetscCall(PetscContainerDestroy(&surfGradObj));
  }
  if (islite) EGlite_free(fobjs);
  else EG_free(fobjs);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DestroyHashMap(void **p)
{
  PetscFunctionBegin;
  PetscCall(PetscHMapIDestroy((PetscHMapI *)p));
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

/*@C
  DMPlexGeomDataAndGrads - Exposes Control Points and Control Point Weights defining the underlying geometry allowing user manipulation of the geometry.

  Collective

  Input Parameters:
+ dm           - The DM object representing the mesh with PetscContainer containing an EGADS geometry model
- fullGeomGrad - PetscBool flag. Determines how the Surface Area and Volume Gradients wrt to Control Points and Control Point Weights are calculated.
                      PETSC_FALSE :: Surface Area Gradient wrt Control Points and Control Point Weights are calculated using the change in the local
                                     FACE changes (not the entire body). Volume Gradients are not calculated. Faster computations.
                      PETSC_TRUE  :: Surface Area Gradietn wrt to Control Points and Control Point Weights are calculated using the change observed in
                                     the entire solid body. Volume Gradients are calculated. Slower computation due to the need to generate a new solid
                                     body geometry for every Control Point and Control Point Weight change.

  Output Parameter:
. dm - The updated DM object representing the mesh with PetscContainers containing the Control Point, Control Point Weight and Gradient Data.

  Level: intermediate

  Note:
  Calculates the DM Point location, surface area and volume gradients wrt to Control Point and Control Point Weights using Finite Difference (small perturbation of Control Point coordinates or Control Point Weight value).

.seealso: `DMPLEX`, `DMCreate()`, `DMPlexCreateGeom()`, `DMPlexModifyEGADSGeomModel()`
@*/
PetscErrorCode DMPlexGeomDataAndGrads(DM dm, PetscBool fullGeomGrad) PeNS
{
#if defined(PETSC_HAVE_EGADS)
  /* PETSc Variables */
  PetscContainer modelObj;
  PetscHMapI     faceCntrlPtRow_Start = NULL, faceCPWeightsRow_Start = NULL;
  PetscHMapI     pointSurfGradRow_Start = NULL;
  Mat            pointSurfGrad, cpEquiv;
  IS             faceLabelValues, edgeLabelValues, vertexLabelValues;
  PetscInt       faceLabelSize, edgeLabelSize, vertexLabelSize;
  PetscBool      islite = PETSC_FALSE;
  /* EGADS Variables */
  ego model, geom, *bodies, *fobjs = NULL;
  int oclass, mtype, *senses;
  int Nb, Nf;
#endif

  PetscFunctionBegin;
#if defined(PETSC_HAVE_EGADS)

  PetscCall(PetscObjectQuery((PetscObject)dm, "EGADS Model", (PetscObject *)&modelObj));
  if (!modelObj) {
    PetscCall(PetscObjectQuery((PetscObject)dm, "EGADSlite Model", (PetscObject *)&modelObj));
    PetscCheck(modelObj, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Input DM must have attached EGADS Geometry Model");
    islite = PETSC_TRUE;
    SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Cannot provide geometric data or associated calculated gradients for geometries defined by EGADSlite (.egadslite)!\nPlease use another geometry file format STEP, IGES, EGADS or BRep");
  }

  // Get attached EGADS model (pointer)
  PetscCall(PetscContainerGetPointer(modelObj, (void **)&model));

  // Get the bodies in the model
  if (islite) {
    PetscCall(EGlite_getTopology(model, &geom, &oclass, &mtype, NULL, &Nb, &bodies, &senses));
  } else {
    PetscCall(EG_getTopology(model, &geom, &oclass, &mtype, NULL, &Nb, &bodies, &senses));
  }

  ego body = bodies[0]; // Only operate on 1st body. Model should only have 1 body.

  // Get the total number of FACEs in the model
  if (islite) {
    PetscCall(EGlite_getBodyTopos(body, NULL, FACE, &Nf, &fobjs));
  } else {
    PetscCall(EG_getBodyTopos(body, NULL, FACE, &Nf, &fobjs));
  }

  // Get the total number of points and IDs in the DMPlex with a "EGADS Face Label"
  // This will provide the total number of DMPlex points on the boundary of the geometry
  PetscCall(DMGetLabelIdIS(dm, "EGADS Face ID", &faceLabelValues));
  PetscCall(DMGetLabelSize(dm, "EGADS Face ID", &faceLabelSize));

  PetscCall(DMGetLabelIdIS(dm, "EGADS Edge ID", &edgeLabelValues));
  PetscCall(DMGetLabelSize(dm, "EGADS Edge ID", &edgeLabelSize));

  PetscCall(DMGetLabelIdIS(dm, "EGADS Vertex ID", &vertexLabelValues));
  PetscCall(DMGetLabelSize(dm, "EGADS Vertex ID", &vertexLabelSize));

  const PetscInt *faceIndices, *edgeIndices, *vertexIndices;
  PetscCall(ISGetIndices(faceLabelValues, &faceIndices));
  PetscCall(ISGetIndices(edgeLabelValues, &edgeIndices));
  PetscCall(ISGetIndices(vertexLabelValues, &vertexIndices));

  // Get the points associated with each FACE, EDGE and VERTEX label in the DM
  PetscInt totalNumPoints = 0;
  for (int f = 0; f < faceLabelSize; ++f) {
    // Cycle through FACE labels
    PetscInt size;
    PetscCall(DMGetStratumSize(dm, "EGADS Face ID", faceIndices[f], &size));
    totalNumPoints += size;
  }
  PetscCall(ISRestoreIndices(faceLabelValues, &faceIndices));
  PetscCall(ISDestroy(&faceLabelValues));

  for (int e = 0; e < edgeLabelSize; ++e) {
    // Cycle Through EDGE Labels
    PetscInt size;
    PetscCall(DMGetStratumSize(dm, "EGADS Edge ID", edgeIndices[e], &size));
    totalNumPoints += size;
  }
  PetscCall(ISRestoreIndices(edgeLabelValues, &edgeIndices));
  PetscCall(ISDestroy(&edgeLabelValues));

  for (int ii = 0; ii < vertexLabelSize; ++ii) {
    // Cycle Through VERTEX Labels
    PetscInt size;
    PetscCall(DMGetStratumSize(dm, "EGADS Vertex ID", vertexIndices[ii], &size));
    totalNumPoints += size;
  }
  PetscCall(ISRestoreIndices(vertexLabelValues, &vertexIndices));
  PetscCall(ISDestroy(&vertexLabelValues));

  int     maxNumCPs   = 0;
  int     totalNumCPs = 0;
  ego     bRef, bPrev, bNext, fgeom, *lobjs;
  int     id, boclass, bmtype, *bpinfo;
  int     foclass, fmtype, Nl, *lsenses;
  double *bprv;
  double  fdata[4];

  // Create Hash Tables
  PetscInt cntr = 0, wcntr = 0, vcntr = 0;
  PetscCall(PetscHMapICreate(&faceCntrlPtRow_Start));
  PetscCall(PetscHMapICreate(&faceCPWeightsRow_Start));

  for (int f = 0; f < Nf; ++f) {
    // Need to get the maximum number of Control Points defining the FACEs
    ego face = fobjs[f];
    int maxNumCPs_temp;

    if (islite) {
      id = EGlite_indexBodyTopo(body, face);
      PetscCall(EGlite_getTopology(face, &fgeom, &foclass, &fmtype, fdata, &Nl, &lobjs, &lsenses));
      PetscCall(EGlite_getGeometry(fgeom, &boclass, &bmtype, &bRef, &bpinfo, &bprv));
      PetscCall(EGlite_getInfo(fgeom, &boclass, &bmtype, &bRef, &bPrev, &bNext));
    } else {
      id = EG_indexBodyTopo(body, face);
      PetscCall(EG_getTopology(face, &fgeom, &foclass, &fmtype, fdata, &Nl, &lobjs, &lsenses));
      PetscCall(EG_getGeometry(fgeom, &boclass, &bmtype, &bRef, &bpinfo, &bprv));
      PetscCall(EG_getInfo(fgeom, &boclass, &bmtype, &bRef, &bPrev, &bNext));
    }
    maxNumCPs_temp = bpinfo[2] * bpinfo[5];
    totalNumCPs += bpinfo[2] * bpinfo[5];

    if (maxNumCPs_temp > maxNumCPs) { maxNumCPs = maxNumCPs_temp; }
  }

  PetscInt *cpCoordDataLengthPtr, *wDataLengthPtr;
  PetscInt  cpCoordDataLength = 3 * totalNumCPs;
  PetscInt  wDataLength       = totalNumCPs;
  cpCoordDataLengthPtr        = &cpCoordDataLength;
  wDataLengthPtr              = &wDataLength;

  Vec          cntrlPtCoordsVec, cntrlPtWeightsVec;
  PetscScalar *cntrlPtCoords, *cntrlPtWeights;
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, cpCoordDataLength, &cntrlPtCoordsVec));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, wDataLength, &cntrlPtWeightsVec));

  // For dSA/dCPi
  Vec          gradSACPVec, gradSAWVec, gradVCPVec, gradVWVec;
  PetscScalar *gradSACP, *gradSAW, *gradVCP, *gradVW;
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, cpCoordDataLength, &gradSACPVec));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, wDataLength, &gradSAWVec));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, cpCoordDataLength, &gradVCPVec));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, wDataLength, &gradVWVec));

  // Control Point - Vertex/Edge/Face Relationship
  PetscInt *cp_vertex, *cp_edge, *cp_face;
  PetscInt *w_vertex, *w_edge, *w_face;
  PetscCall(PetscMalloc1(totalNumCPs, &cp_vertex));
  PetscCall(PetscMalloc1(totalNumCPs, &cp_edge));
  PetscCall(PetscMalloc1(totalNumCPs, &cp_face));
  PetscCall(PetscMalloc1(wDataLength, &w_vertex));
  PetscCall(PetscMalloc1(wDataLength, &w_edge));
  PetscCall(PetscMalloc1(wDataLength, &w_face));

  for (int f = 0; f < Nf; ++f) {
    // Need to Populate Control Point Coordinates and Weight Vectors
    ego           face = fobjs[f];
    ego          *vobjs, *eobjs;
    int           offsetCoord, offsetWeight;
    PetscInt      Nv, Ne, wRowStart = 0;
    PetscHashIter hashKeyIter, wHashKeyIter;
    PetscBool     hashKeyFound, wHashKeyFound;

    if (islite) {
      id = EGlite_indexBodyTopo(body, face);
      PetscCallEGADS(EGlite_getTopology, (face, &fgeom, &foclass, &fmtype, fdata, &Nl, &lobjs, &lsenses));
      PetscCallEGADS(EGlite_getGeometry, (fgeom, &boclass, &bmtype, &bRef, &bpinfo, &bprv));
      PetscCallEGADS(EGlite_getInfo, (fgeom, &boclass, &bmtype, &bRef, &bPrev, &bNext));
      PetscCallEGADS(EGlite_getBodyTopos, (body, face, NODE, &Nv, &vobjs));
    } else {
      id = EG_indexBodyTopo(body, face);
      PetscCallEGADS(EG_getTopology, (face, &fgeom, &foclass, &fmtype, fdata, &Nl, &lobjs, &lsenses));
      PetscCallEGADS(EG_getGeometry, (fgeom, &boclass, &bmtype, &bRef, &bpinfo, &bprv));
      PetscCallEGADS(EG_getInfo, (fgeom, &boclass, &bmtype, &bRef, &bPrev, &bNext));
      PetscCallEGADS(EG_getBodyTopos, (body, face, NODE, &Nv, &vobjs));
    }

    // Store Face ID to 1st Row of Control Point Vector
    PetscCall(PetscHMapIFind(faceCntrlPtRow_Start, id, &hashKeyIter, &hashKeyFound));

    if (!hashKeyFound) PetscCall(PetscHMapISet(faceCntrlPtRow_Start, id, cntr));

    PetscCall(VecGetArrayWrite(cntrlPtCoordsVec, &cntrlPtCoords));
    offsetCoord = bpinfo[3] + bpinfo[6];
    for (int jj = 0; jj < 3 * bpinfo[2] * bpinfo[5]; ++jj) {
      cntrlPtCoords[cntr] = bprv[offsetCoord + jj];
      cntr += 1;
    }

    // Store Face ID to 1st Row of Control Point Weight Vector
    PetscCall(PetscHMapIFind(faceCPWeightsRow_Start, id, &wHashKeyIter, &wHashKeyFound));

    if (!wHashKeyFound) {
      PetscCall(PetscHMapISet(faceCPWeightsRow_Start, id, wcntr));
      wRowStart = wcntr;
    }

    PetscCall(VecGetArrayWrite(cntrlPtWeightsVec, &cntrlPtWeights));
    offsetWeight = bpinfo[3] + bpinfo[6] + (3 * bpinfo[2] * bpinfo[5]);
    for (int jj = 0; jj < bpinfo[2] * bpinfo[5]; ++jj) {
      cntrlPtWeights[wcntr] = bprv[offsetWeight + jj];
      cp_face[wcntr]        = id;
      w_face[wcntr]         = id;
      wcntr += 1;
    }
    PetscCall(VecRestoreArrayWrite(cntrlPtWeightsVec, &cntrlPtWeights));

    // Associate Control Points with Vertex IDs
    PetscScalar xcp, ycp, zcp;
    offsetCoord = bpinfo[3] + bpinfo[6];
    for (int jj = 0; jj < 3 * bpinfo[2] * bpinfo[5]; jj += 3) {
      xcp = bprv[offsetCoord + jj + 0];
      ycp = bprv[offsetCoord + jj + 1];
      zcp = bprv[offsetCoord + jj + 2];

      //Initialize Control Point and Weight to Vertex ID relationship to -1
      cp_vertex[vcntr] = -1;
      w_vertex[vcntr]  = -1;
      cp_edge[vcntr]   = -1;
      w_edge[vcntr]    = -1;

      for (int kk = 0; kk < Nv; ++kk) {
        int         vid;
        double      vCoords[3];
        PetscScalar vDelta;
        ego         vertex = vobjs[kk];

        if (islite) {
          vid = EGlite_indexBodyTopo(body, vertex);
          PetscCallEGADS(EGlite_evaluate, (vertex, NULL, vCoords));
        } else {
          vid = EG_indexBodyTopo(body, vertex);
          PetscCallEGADS(EG_evaluate, (vertex, NULL, vCoords));
        }
        vDelta = PetscSqrtReal(PetscSqr(vCoords[0] - xcp) + PetscSqr(vCoords[1] - ycp) + PetscSqr(vCoords[2] - zcp));

        if (vDelta < 1.0E-15) {
          cp_vertex[vcntr] = vid;
          w_vertex[vcntr]  = vid;
        }
      }
      vcntr += 1;
    }
    // These two line could be replaced with DMPlexFreeGeomObject()
    if (islite) EGlite_free(vobjs);
    else EG_free(vobjs);

    // Associate Control Points with Edge IDs
    if (islite) PetscCallEGADS(EGlite_getBodyTopos, (body, face, EDGE, &Ne, &eobjs));
    else PetscCallEGADS(EG_getBodyTopos, (body, face, EDGE, &Ne, &eobjs));

    int cpV1, cpV2;
    int minID, maxID;

    // Along vmin axis
    minID = wRowStart;
    maxID = wRowStart + (bpinfo[2] - 1);
    cpV1  = cp_vertex[minID];
    cpV2  = cp_vertex[maxID];
    for (int jj = 0; jj < Ne; ++jj) {
      ego edge = eobjs[jj];
      ego egeom, *nobjs;
      int eoclass, emtype, Nn, *nsenses;
      int n1ID, n2ID, eid;

      if (islite) {
        eid = EGlite_indexBodyTopo(body, edge);
        PetscCallEGADS(EGlite_getTopology, (edge, &egeom, &eoclass, &emtype, NULL, &Nn, &nobjs, &nsenses));
      } else {
        eid = EG_indexBodyTopo(body, edge);
        PetscCallEGADS(EG_getTopology, (edge, &egeom, &eoclass, &emtype, NULL, &Nn, &nobjs, &nsenses));
      }

      if (emtype != DEGENERATE) {
        // Get IDs for current Edge's End Vertices
        if (islite) {
          n1ID = EGlite_indexBodyTopo(body, nobjs[0]);
          n2ID = EGlite_indexBodyTopo(body, nobjs[1]);
        } else {
          n1ID = EG_indexBodyTopo(body, nobjs[0]);
          n2ID = EG_indexBodyTopo(body, nobjs[1]);
        }

        if ((cpV1 == n1ID || cpV1 == n2ID) && (cpV2 == n1ID || cpV2 == n2ID)) {
          for (int kk = minID + 1; kk < maxID; ++kk) {
            cp_edge[kk] = eid;
            w_edge[kk]  = eid;
          }
        }
      }
    }

    // Along vmax axis
    minID = wRowStart + (bpinfo[2] * (bpinfo[5] - 1));
    maxID = wRowStart + (bpinfo[2] * bpinfo[5] - 1);

    cpV1 = cp_vertex[minID];
    cpV2 = cp_vertex[maxID];
    for (int jj = 0; jj < Ne; ++jj) {
      ego edge = eobjs[jj];
      ego egeom, *nobjs;
      int eoclass, emtype, Nn, *nsenses;
      int n1ID, n2ID, eid;

      if (islite) {
        eid = EGlite_indexBodyTopo(body, edge);
        PetscCallEGADS(EGlite_getTopology, (edge, &egeom, &eoclass, &emtype, NULL, &Nn, &nobjs, &nsenses));
      } else {
        eid = EG_indexBodyTopo(body, edge);
        PetscCallEGADS(EG_getTopology, (edge, &egeom, &eoclass, &emtype, NULL, &Nn, &nobjs, &nsenses));
      }

      if (emtype != DEGENERATE) {
        // Get IDs for current Edge's End Vertices
        if (islite) {
          n1ID = EGlite_indexBodyTopo(body, nobjs[0]);
          n2ID = EGlite_indexBodyTopo(body, nobjs[1]);
        } else {
          n1ID = EG_indexBodyTopo(body, nobjs[0]);
          n2ID = EG_indexBodyTopo(body, nobjs[1]);
        }

        if ((cpV1 == n1ID || cpV1 == n2ID) && (cpV2 == n1ID || cpV2 == n2ID)) {
          for (int kk = minID + 1; kk < maxID - 1; ++kk) {
            cp_edge[kk] = eid;
            w_edge[kk]  = eid;
          }
        }
      }
    }

    // Along umin axis
    minID = wRowStart;
    maxID = wRowStart + (bpinfo[2] * (bpinfo[5] - 1));

    cpV1 = cp_vertex[minID];
    cpV2 = cp_vertex[maxID];
    for (int jj = 0; jj < Ne; ++jj) {
      ego edge = eobjs[jj];
      ego egeom, *nobjs;
      int eoclass, emtype, Nn, *nsenses;
      int n1ID, n2ID, eid;

      if (islite) {
        eid = EGlite_indexBodyTopo(body, edge);
        PetscCallEGADS(EGlite_getTopology, (edge, &egeom, &eoclass, &emtype, NULL, &Nn, &nobjs, &nsenses));
      } else {
        eid = EG_indexBodyTopo(body, edge);
        PetscCallEGADS(EG_getTopology, (edge, &egeom, &eoclass, &emtype, NULL, &Nn, &nobjs, &nsenses));
      }

      if (emtype != DEGENERATE) {
        // Get IDs for current Edge's End Vertices
        if (islite) {
          n1ID = EGlite_indexBodyTopo(body, nobjs[0]);
          n2ID = EGlite_indexBodyTopo(body, nobjs[1]);
        } else {
          n1ID = EG_indexBodyTopo(body, nobjs[0]);
          n2ID = EG_indexBodyTopo(body, nobjs[1]);
        }

        if ((cpV1 == n1ID || cpV1 == n2ID) && (cpV2 == n1ID || cpV2 == n2ID)) {
          for (int kk = minID + bpinfo[2]; kk < maxID; kk += bpinfo[2]) {
            cp_edge[kk] = eid;
            w_edge[kk]  = eid;
          }
        }
      }
    }

    // Along umax axis
    minID = wRowStart + (bpinfo[2] - 1);
    maxID = wRowStart + (bpinfo[2] * bpinfo[5]) - 1;
    cpV1  = cp_vertex[minID];
    cpV2  = cp_vertex[maxID];
    for (int jj = 0; jj < Ne; ++jj) {
      ego edge = eobjs[jj];
      ego egeom, *nobjs;
      int eoclass, emtype, Nn, *nsenses;
      int n1ID, n2ID, eid;

      if (islite) {
        eid = EGlite_indexBodyTopo(body, edge);
        PetscCallEGADS(EGlite_getTopology, (edge, &egeom, &eoclass, &emtype, NULL, &Nn, &nobjs, &nsenses));
      } else {
        eid = EG_indexBodyTopo(body, edge);
        PetscCallEGADS(EG_getTopology, (edge, &egeom, &eoclass, &emtype, NULL, &Nn, &nobjs, &nsenses));
      }

      if (emtype != DEGENERATE) {
        // Get IDs for current Edge's End Vertices
        if (islite) {
          n1ID = EGlite_indexBodyTopo(body, nobjs[0]);
          n2ID = EGlite_indexBodyTopo(body, nobjs[1]);
        } else {
          n1ID = EG_indexBodyTopo(body, nobjs[0]);
          n2ID = EG_indexBodyTopo(body, nobjs[1]);
        }

        if ((cpV1 == n1ID || cpV1 == n2ID) && (cpV2 == n1ID || cpV2 == n2ID)) {
          for (int kk = minID + bpinfo[2]; kk < maxID; kk += bpinfo[2]) {
            cp_edge[kk] = eid;
            w_edge[kk]  = eid;
          }
        }
      }
    }
    // These two lines could be replaced with DMPlexFreeGeomObject()
    if (islite) EGlite_free(eobjs);
    else EG_free(eobjs);
  }

  // Determine Control Point Equivalence Matrix relating Control Points between Surfaces
  //     Note: The Weights will also be tied together in the same manner
  //           Also can use the Weight Hash Table for Row Start ID of each Face
  const PetscInt cpRowSize = totalNumCPs;
  const PetscInt cpColSize = cpRowSize;
  PetscInt      *maxNumRelatePtr;
  PetscInt       maxNumRelate = 0;

  // Create Point Surface Gradient Matrix
  PetscCall(MatCreate(PETSC_COMM_WORLD, &cpEquiv));
  PetscCall(MatSetSizes(cpEquiv, PETSC_DECIDE, PETSC_DECIDE, cpRowSize, cpColSize));
  PetscCall(MatSetType(cpEquiv, MATAIJ));
  PetscCall(MatSetUp(cpEquiv));

  for (int ii = 0; ii < totalNumCPs; ++ii) {
    PetscScalar x1, y1, z1;
    PetscInt    maxRelateTemp = 0;

    x1 = cntrlPtCoords[(3 * ii) + 0];
    y1 = cntrlPtCoords[(3 * ii) + 1];
    z1 = cntrlPtCoords[(3 * ii) + 2];

    for (int jj = 0; jj < totalNumCPs; ++jj) {
      PetscScalar x2, y2, z2;
      PetscScalar cpDelta, eqFactor;
      x2 = cntrlPtCoords[(3 * jj) + 0];
      y2 = cntrlPtCoords[(3 * jj) + 1];
      z2 = cntrlPtCoords[(3 * jj) + 2];

      cpDelta = PetscSqrtReal(PetscSqr(x2 - x1) + PetscSqr(y2 - y1) + PetscSqr(z2 - z1));
      if (cpDelta < 1.0E-15) {
        eqFactor = 1.0;
        maxRelateTemp += 1;
      } else {
        eqFactor = 0.0;
      }

      // Store Results in PETSc Mat
      PetscCall(MatSetValue(cpEquiv, ii, jj, eqFactor, INSERT_VALUES));
    }
    if (maxRelateTemp > maxNumRelate) maxNumRelate = maxRelateTemp;
  }
  maxNumRelatePtr = &maxNumRelate;
  PetscCall(VecRestoreArrayWrite(cntrlPtCoordsVec, &cntrlPtCoords));

  // Assemble Point Surface Grad Matrix
  PetscCall(MatAssemblyBegin(cpEquiv, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(cpEquiv, MAT_FINAL_ASSEMBLY));

  // Attach Control Point and Weight Data to DM
  {
    PetscContainer cpOrgObj, cpCoordLengthObj;
    PetscContainer wOrgObj, wDataLengthObj;
    PetscContainer cp_faceObj, cp_edgeObj, cp_vertexObj;
    PetscContainer w_faceObj, w_edgeObj, w_vertexObj;
    PetscContainer maxNumRelateObj;

    PetscCall(PetscObjectQuery((PetscObject)dm, "Control Point Hash Table", (PetscObject *)&cpOrgObj));
    if (!cpOrgObj) {
      PetscCall(PetscContainerCreate(PETSC_COMM_SELF, &cpOrgObj));
      PetscCall(PetscContainerSetPointer(cpOrgObj, faceCntrlPtRow_Start));
      PetscCall(PetscObjectCompose((PetscObject)dm, "Control Point Hash Table", (PetscObject)cpOrgObj));
      PetscCall(PetscContainerDestroy(&cpOrgObj));
    } else {
      PetscCall(PetscContainerSetPointer(cpOrgObj, faceCntrlPtRow_Start));
    }

    PetscCall(PetscObjectCompose((PetscObject)dm, "Control Point Coordinates", (PetscObject)cntrlPtCoordsVec));
    PetscCall(VecDestroy(&cntrlPtCoordsVec));

    PetscCall(PetscObjectQuery((PetscObject)dm, "Control Point Coordinate Data Length", (PetscObject *)&cpCoordLengthObj));
    if (!cpCoordLengthObj) {
      PetscCall(PetscContainerCreate(PETSC_COMM_SELF, &cpCoordLengthObj));
      PetscCall(PetscContainerSetPointer(cpCoordLengthObj, cpCoordDataLengthPtr));
      PetscCall(PetscObjectCompose((PetscObject)dm, "Control Point Coordinate Data Length", (PetscObject)cpCoordLengthObj));
      PetscCall(PetscContainerDestroy(&cpCoordLengthObj));
    } else {
      PetscCall(PetscContainerSetPointer(cpCoordLengthObj, cpCoordDataLengthPtr));
    }

    PetscCall(PetscObjectQuery((PetscObject)dm, "Control Point Weights Hash Table", (PetscObject *)&wOrgObj));
    if (!wOrgObj) {
      PetscCall(PetscContainerCreate(PETSC_COMM_SELF, &wOrgObj));
      PetscCall(PetscContainerSetPointer(wOrgObj, faceCPWeightsRow_Start));
      PetscCall(PetscObjectCompose((PetscObject)dm, "Control Point Weights Hash Table", (PetscObject)wOrgObj));
      PetscCall(PetscContainerDestroy(&wOrgObj));
    } else {
      PetscCall(PetscContainerSetPointer(wOrgObj, faceCPWeightsRow_Start));
    }

    PetscCall(PetscObjectCompose((PetscObject)dm, "Control Point Weight Data", (PetscObject)cntrlPtWeightsVec));
    PetscCall(VecDestroy(&cntrlPtWeightsVec));

    PetscCall(PetscObjectQuery((PetscObject)dm, "Control Point Weight Data Length", (PetscObject *)&wDataLengthObj));
    if (!wDataLengthObj) {
      PetscCall(PetscContainerCreate(PETSC_COMM_SELF, &wDataLengthObj));
      PetscCall(PetscContainerSetPointer(wDataLengthObj, wDataLengthPtr));
      PetscCall(PetscObjectCompose((PetscObject)dm, "Control Point Weight Data Length", (PetscObject)wDataLengthObj));
      PetscCall(PetscContainerDestroy(&wDataLengthObj));
    } else {
      PetscCall(PetscContainerSetPointer(wDataLengthObj, wDataLengthPtr));
    }

    PetscCall(PetscObjectCompose((PetscObject)dm, "Control Point Equivalency Matrix", (PetscObject)cpEquiv));

    PetscCall(PetscObjectQuery((PetscObject)dm, "Maximum Number Control Point Equivalency", (PetscObject *)&maxNumRelateObj));
    if (!maxNumRelateObj) {
      PetscCall(PetscContainerCreate(PETSC_COMM_SELF, &maxNumRelateObj));
      PetscCall(PetscContainerSetPointer(maxNumRelateObj, maxNumRelatePtr));
      PetscCall(PetscObjectCompose((PetscObject)dm, "Maximum Number Control Point Equivalency", (PetscObject)maxNumRelateObj));
      PetscCall(PetscContainerDestroy(&maxNumRelateObj));
    } else {
      PetscCall(PetscContainerSetPointer(maxNumRelateObj, maxNumRelatePtr));
    }

    PetscCall(PetscObjectQuery((PetscObject)dm, "Control Point - Face Map", (PetscObject *)&cp_faceObj));
    if (!cp_faceObj) {
      PetscCall(PetscContainerCreate(PETSC_COMM_SELF, &cp_faceObj));
      PetscCall(PetscContainerSetPointer(cp_faceObj, cp_face));
      PetscCall(PetscContainerSetCtxDestroy(cp_faceObj, PetscCtxDestroyDefault));
      PetscCall(PetscObjectCompose((PetscObject)dm, "Control Point - Face Map", (PetscObject)cp_faceObj));
      PetscCall(PetscContainerDestroy(&cp_faceObj));
    } else {
      void *tmp;

      PetscCall(PetscContainerGetPointer(cp_faceObj, &tmp));
      PetscCall(PetscFree(tmp));
      PetscCall(PetscContainerSetPointer(cp_faceObj, cp_face));
    }

    PetscCall(PetscObjectQuery((PetscObject)dm, "Control Point Weight - Face Map", (PetscObject *)&w_faceObj));
    if (!w_faceObj) {
      PetscCall(PetscContainerCreate(PETSC_COMM_SELF, &w_faceObj));
      PetscCall(PetscContainerSetPointer(w_faceObj, w_face));
      PetscCall(PetscContainerSetCtxDestroy(w_faceObj, PetscCtxDestroyDefault));
      PetscCall(PetscObjectCompose((PetscObject)dm, "Control Point Weight - Face Map", (PetscObject)w_faceObj));
      PetscCall(PetscContainerDestroy(&w_faceObj));
    } else {
      void *tmp;

      PetscCall(PetscContainerGetPointer(w_faceObj, &tmp));
      PetscCall(PetscFree(tmp));
      PetscCall(PetscContainerSetPointer(w_faceObj, w_face));
    }

    PetscCall(PetscObjectQuery((PetscObject)dm, "Control Point - Edge Map", (PetscObject *)&cp_edgeObj));
    if (!cp_edgeObj) {
      PetscCall(PetscContainerCreate(PETSC_COMM_SELF, &cp_edgeObj));
      PetscCall(PetscContainerSetPointer(cp_edgeObj, cp_edge));
      PetscCall(PetscContainerSetCtxDestroy(cp_edgeObj, PetscCtxDestroyDefault));
      PetscCall(PetscObjectCompose((PetscObject)dm, "Control Point - Edge Map", (PetscObject)cp_edgeObj));
      PetscCall(PetscContainerDestroy(&cp_edgeObj));
    } else {
      void *tmp;

      PetscCall(PetscContainerGetPointer(cp_edgeObj, &tmp));
      PetscCall(PetscFree(tmp));
      PetscCall(PetscContainerSetPointer(cp_edgeObj, cp_edge));
    }

    PetscCall(PetscObjectQuery((PetscObject)dm, "Control Point Weight - Edge Map", (PetscObject *)&w_edgeObj));
    if (!w_edgeObj) {
      PetscCall(PetscContainerCreate(PETSC_COMM_SELF, &w_edgeObj));
      PetscCall(PetscContainerSetPointer(w_edgeObj, w_edge));
      PetscCall(PetscContainerSetCtxDestroy(w_edgeObj, PetscCtxDestroyDefault));
      PetscCall(PetscObjectCompose((PetscObject)dm, "Control Point Weight - Edge Map", (PetscObject)w_edgeObj));
      PetscCall(PetscContainerDestroy(&w_edgeObj));
    } else {
      void *tmp;

      PetscCall(PetscContainerGetPointer(w_edgeObj, &tmp));
      PetscCall(PetscFree(tmp));
      PetscCall(PetscContainerSetPointer(w_edgeObj, w_edge));
    }

    PetscCall(PetscObjectQuery((PetscObject)dm, "Control Point - Vertex Map", (PetscObject *)&cp_vertexObj));
    if (!cp_vertexObj) {
      PetscCall(PetscContainerCreate(PETSC_COMM_SELF, &cp_vertexObj));
      PetscCall(PetscContainerSetPointer(cp_vertexObj, cp_vertex));
      PetscCall(PetscContainerSetCtxDestroy(cp_vertexObj, PetscCtxDestroyDefault));
      PetscCall(PetscObjectCompose((PetscObject)dm, "Control Point - Vertex Map", (PetscObject)cp_vertexObj));
      PetscCall(PetscContainerDestroy(&cp_vertexObj));
    } else {
      void *tmp;

      PetscCall(PetscContainerGetPointer(cp_vertexObj, &tmp));
      PetscCall(PetscFree(tmp));
      PetscCall(PetscContainerSetPointer(cp_vertexObj, cp_vertex));
    }

    PetscCall(PetscObjectQuery((PetscObject)dm, "Control Point Weight - Vertex Map", (PetscObject *)&w_vertexObj));
    if (!w_vertexObj) {
      PetscCall(PetscContainerCreate(PETSC_COMM_SELF, &w_vertexObj));
      PetscCall(PetscContainerSetPointer(w_vertexObj, w_vertex));
      PetscCall(PetscContainerSetCtxDestroy(w_vertexObj, PetscCtxDestroyDefault));
      PetscCall(PetscObjectCompose((PetscObject)dm, "Control Point Weight - Vertex Map", (PetscObject)w_vertexObj));
      PetscCall(PetscContainerDestroy(&w_vertexObj));
    } else {
      void *tmp;

      PetscCall(PetscContainerGetPointer(w_vertexObj, &tmp));
      PetscCall(PetscFree(tmp));
      PetscCall(PetscContainerSetPointer(w_vertexObj, w_vertex));
    }
  }

  // Define Matrix to store  Geometry Gradient information dGeom_i/dCPj_i
  PetscInt       gcntr   = 0;
  const PetscInt rowSize = 3 * maxNumCPs * totalNumPoints;
  const PetscInt colSize = 4 * Nf;

  // Create Point Surface Gradient Matrix
  PetscCall(MatCreate(PETSC_COMM_WORLD, &pointSurfGrad));
  PetscCall(MatSetSizes(pointSurfGrad, PETSC_DECIDE, PETSC_DECIDE, rowSize, colSize));
  PetscCall(MatSetType(pointSurfGrad, MATAIJ));
  PetscCall(MatSetUp(pointSurfGrad));

  // Create Hash Table to store Point's stare row in surfaceGrad[][]
  PetscCall(PetscHMapICreate(&pointSurfGradRow_Start));

  // Get Coordinates for the DMPlex point
  DM           cdm;
  PetscInt     dE, Nv;
  Vec          coordinatesLocal;
  PetscScalar *coords = NULL;

  PetscCall(DMGetCoordinateDM(dm, &cdm));
  PetscCall(DMGetCoordinateDim(dm, &dE));
  PetscCall(DMGetCoordinatesLocal(dm, &coordinatesLocal));

  // CYCLE THROUGH FACEs
  PetscScalar maxGrad = 0.;
  PetscCall(VecGetArrayWrite(gradSACPVec, &gradSACP));
  PetscCall(VecGetArrayWrite(gradSAWVec, &gradSAW));
  PetscCall(VecGetArrayWrite(gradVCPVec, &gradVCP));
  PetscCall(VecGetArrayWrite(gradVWVec, &gradVW));
  for (int f = 0; f < Nf; ++f) {
    ego             face = fobjs[f];
    ego            *eobjs, *nobjs;
    PetscInt        fid, Ne, Nn;
    DMLabel         faceLabel, edgeLabel, nodeLabel;
    PetscHMapI      currFaceUniquePoints = NULL;
    IS              facePoints, edgePoints, nodePoints;
    const PetscInt *fIndices, *eIndices, *nIndices;
    PetscInt        fSize, eSize, nSize;
    PetscHashIter   fHashKeyIter, eHashKeyIter, nHashKeyIter, pHashKeyIter;
    PetscBool       fHashKeyFound, eHashKeyFound, nHashKeyFound, pHashKeyFound;
    PetscInt        cfCntr = 0;

    // Get Geometry Object for the Current FACE
    if (islite) {
      PetscCall(EGlite_getTopology(face, &fgeom, &foclass, &fmtype, fdata, &Nl, &lobjs, &lsenses));
      PetscCall(EGlite_getGeometry(fgeom, &boclass, &bmtype, &bRef, &bpinfo, &bprv));
    } else {
      PetscCall(EG_getTopology(face, &fgeom, &foclass, &fmtype, fdata, &Nl, &lobjs, &lsenses));
      PetscCall(EG_getGeometry(fgeom, &boclass, &bmtype, &bRef, &bpinfo, &bprv));
    }

    // Get all EDGE and NODE objects attached to the current FACE
    if (islite) {
      PetscCall(EGlite_getBodyTopos(body, face, EDGE, &Ne, &eobjs));
      PetscCall(EGlite_getBodyTopos(body, face, NODE, &Nn, &nobjs));
    } else {
      PetscCall(EG_getBodyTopos(body, face, EDGE, &Ne, &eobjs));
      PetscCall(EG_getBodyTopos(body, face, NODE, &Nn, &nobjs));
    }

    // Get all DMPlex Points that have DMLabel "EGADS Face ID" and store them in a Hash Table for later use
    if (islite) {
      fid = EGlite_indexBodyTopo(body, face);
    } else {
      fid = EG_indexBodyTopo(body, face);
    }

    PetscCall(DMGetLabel(dm, "EGADS Face ID", &faceLabel));
    PetscCall(DMLabelGetStratumIS(faceLabel, fid, &facePoints));
    PetscCall(ISGetIndices(facePoints, &fIndices));
    PetscCall(ISGetSize(facePoints, &fSize));

    PetscCall(PetscHMapICreate(&currFaceUniquePoints));

    for (int jj = 0; jj < fSize; ++jj) {
      PetscCall(PetscHMapIFind(currFaceUniquePoints, fIndices[jj], &fHashKeyIter, &fHashKeyFound));

      if (!fHashKeyFound) {
        PetscCall(PetscHMapISet(currFaceUniquePoints, fIndices[jj], cfCntr));
        cfCntr += 1;
      }

      PetscCall(PetscHMapIFind(pointSurfGradRow_Start, fIndices[jj], &pHashKeyIter, &pHashKeyFound));

      if (!pHashKeyFound) {
        PetscCall(PetscHMapISet(pointSurfGradRow_Start, fIndices[jj], gcntr));
        gcntr += 3 * maxNumCPs;
      }
    }
    PetscCall(ISRestoreIndices(facePoints, &fIndices));
    PetscCall(ISDestroy(&facePoints));

    // Get all DMPlex Points that have DMLable "EGADS Edge ID" attached to the current FACE and store them in a Hash Table for later use.
    for (int jj = 0; jj < Ne; ++jj) {
      ego       edge = eobjs[jj];
      PetscBool containLabelValue;

      if (islite) {
        id = EGlite_indexBodyTopo(body, edge);
      } else {
        id = EG_indexBodyTopo(body, edge);
      }

      PetscCall(DMGetLabel(dm, "EGADS Edge ID", &edgeLabel));
      PetscCall(DMLabelHasValue(edgeLabel, id, &containLabelValue));

      if (containLabelValue) {
        PetscCall(DMLabelGetStratumIS(edgeLabel, id, &edgePoints));
        PetscCall(ISGetIndices(edgePoints, &eIndices));
        PetscCall(ISGetSize(edgePoints, &eSize));

        for (int kk = 0; kk < eSize; ++kk) {
          PetscCall(PetscHMapIFind(currFaceUniquePoints, eIndices[kk], &eHashKeyIter, &eHashKeyFound));

          if (!eHashKeyFound) {
            PetscCall(PetscHMapISet(currFaceUniquePoints, eIndices[kk], cfCntr));
            cfCntr += 1;
          }

          PetscCall(PetscHMapIFind(pointSurfGradRow_Start, eIndices[kk], &pHashKeyIter, &pHashKeyFound));

          if (!pHashKeyFound) {
            PetscCall(PetscHMapISet(pointSurfGradRow_Start, eIndices[kk], gcntr));
            gcntr += 3 * maxNumCPs;
          }
        }
        PetscCall(ISRestoreIndices(edgePoints, &eIndices));
        PetscCall(ISDestroy(&edgePoints));
      }
    }

    // Get all DMPlex Points that have DMLabel "EGADS Vertex ID" attached to the current FACE and store them in a Hash Table for later use.
    for (int jj = 0; jj < Nn; ++jj) {
      ego node = nobjs[jj];

      if (islite) {
        id = EGlite_indexBodyTopo(body, node);
      } else {
        id = EG_indexBodyTopo(body, node);
      }

      PetscCall(DMGetLabel(dm, "EGADS Vertex ID", &nodeLabel));
      PetscCall(DMLabelGetStratumIS(nodeLabel, id, &nodePoints));
      PetscCall(ISGetIndices(nodePoints, &nIndices));
      PetscCall(ISGetSize(nodePoints, &nSize));

      for (int kk = 0; kk < nSize; ++kk) {
        PetscCall(PetscHMapIFind(currFaceUniquePoints, nIndices[kk], &nHashKeyIter, &nHashKeyFound));

        if (!nHashKeyFound) {
          PetscCall(PetscHMapISet(currFaceUniquePoints, nIndices[kk], cfCntr));
          cfCntr += 1;
        }

        PetscCall(PetscHMapIFind(pointSurfGradRow_Start, nIndices[kk], &pHashKeyIter, &pHashKeyFound));
        if (!pHashKeyFound) {
          PetscCall(PetscHMapISet(pointSurfGradRow_Start, nIndices[kk], gcntr));
          gcntr += 3 * maxNumCPs;
        }
      }
      PetscCall(ISRestoreIndices(nodePoints, &nIndices));
      PetscCall(ISDestroy(&nodePoints));
    }

    // Get the Total Number of entries in the Hash Table
    PetscInt currFaceUPSize;
    PetscCall(PetscHMapIGetSize(currFaceUniquePoints, &currFaceUPSize));

    // Get Keys
    PetscInt currFaceUPKeys[currFaceUPSize], off = 0;
    PetscCall(PetscHMapIGetKeys(currFaceUniquePoints, &off, currFaceUPKeys));
    PetscCall(PetscHMapIDestroy(&currFaceUniquePoints));

    // Get Current Face Surface Area
    PetscScalar fSA, faceData[14];
    PetscCall(EG_getMassProperties(face, faceData)); // This doesn't have a EGlite version. Will it work for EGADSlite files??  KNOWN_ISSUE
    fSA = faceData[1];

    // Get Start Row in cpEquiv Matrix
    PetscHashIter Witer;
    PetscBool     Wfound;
    PetscInt      faceWStartRow;
    PetscCall(PetscHMapIFind(faceCPWeightsRow_Start, fid, &Witer, &Wfound));
    PetscCheck(Wfound, PETSC_COMM_SELF, PETSC_ERR_SUP, "FACE ID not found in Control Point Weights Hash Table");
    PetscCall(PetscHMapIGet(faceCPWeightsRow_Start, fid, &faceWStartRow));

    // Cycle through all points on the current FACE
    for (int jj = 0; jj < currFaceUPSize; ++jj) {
      PetscInt currPointID = currFaceUPKeys[jj];
      PetscCall(DMPlexVecGetClosure(cdm, NULL, coordinatesLocal, currPointID, &Nv, &coords));

      // Get UV position of FACE
      double params[2], range[4], eval[18];
      int    peri;

      if (islite) PetscCall(EGlite_getRange(face, range, &peri));
      else PetscCall(EG_getRange(face, range, &peri));

      PetscCall(DMPlex_Geom_FACE_XYZtoUV_Internal(coords, face, range, 0, dE, params, islite));

      if (islite) PetscCall(EGlite_evaluate(face, params, eval));
      else PetscCall(EG_evaluate(face, params, eval));

      // Make a new SURFACE Geometry by changing the location of the Control Points
      int    prvSize = bpinfo[3] + bpinfo[6] + (4 * bpinfo[2] * bpinfo[5]);
      double nbprv[prvSize];

      // Cycle through each Control Point
      double denomNew, denomOld;
      double deltaCoord = 1.0E-4;
      int    offset     = bpinfo[3] + bpinfo[6];
      int    wOffset    = offset + (3 * bpinfo[2] * bpinfo[5]);
      for (int ii = 0; ii < bpinfo[2] * bpinfo[5]; ++ii) {
        PetscCheck(face->blind, PETSC_COMM_SELF, PETSC_ERR_LIB, "Face %d is corrupted: %d %d", f, jj, ii);
  #if 0
        // Cycle through each direction (x, then y, then z)
        if (jj == 0) {
          // Get the Number Control Points that are the same as the current points
          //    We are looking for repeated Control Points
          PetscInt commonCPcntr = 0;
          for (int mm = 0; mm < bpinfo[2]*bpinfo[5]; ++mm) {
            PetscScalar matValue;
            PetscCall(MatGetValue(cpEquiv, faceWStartRow + ii, faceWStartRow + mm, &matValue));

            if (matValue > 0.0) commonCPcntr += 1;
          }
        }
  #endif

        for (int kk = 0; kk < 4; ++kk) {
          // Reinitialize nbprv[] values because we only want to change one value at a time
          for (int mm = 0; mm < prvSize; ++mm) { nbprv[mm] = bprv[mm]; }
          PetscCheck(face->blind, PETSC_COMM_SELF, PETSC_ERR_LIB, "Face %d is corrupted: %d %d %d", f, jj, ii, kk);

          if (kk == 0) { //X
            nbprv[offset + 0] = bprv[offset + 0] + deltaCoord;
            nbprv[offset + 1] = bprv[offset + 1];
            nbprv[offset + 2] = bprv[offset + 2];
            denomNew          = nbprv[offset + 0];
            denomOld          = bprv[offset + 0];
          } else if (kk == 1) { //Y
            nbprv[offset + 0] = bprv[offset + 0];
            nbprv[offset + 1] = bprv[offset + 1] + deltaCoord;
            nbprv[offset + 2] = bprv[offset + 2];
            denomNew          = nbprv[offset + 1];
            denomOld          = bprv[offset + 1];
          } else if (kk == 2) { //Z
            nbprv[offset + 0] = bprv[offset + 0];
            nbprv[offset + 1] = bprv[offset + 1];
            nbprv[offset + 2] = bprv[offset + 2] + deltaCoord;
            denomNew          = nbprv[offset + 2];
            denomOld          = bprv[offset + 2];
          } else if (kk == 3) { // Weights
            nbprv[wOffset + ii] = bprv[wOffset + ii] + deltaCoord;
            denomNew            = nbprv[wOffset + ii];
            denomOld            = bprv[wOffset + ii];
          } else {
            // currently do nothing
          }

          // Create New Surface Based on New Control Points or Weights
          ego newgeom, context;
          PetscCallEGADS(EG_getContext, (face, &context));                                             // This does not have an EGlite_ version KNOWN_ISSUE
          PetscCallEGADS(EG_makeGeometry, (context, SURFACE, BSPLINE, NULL, bpinfo, nbprv, &newgeom)); // This does not have an EGlite_ version KNOWN_ISSUE
          PetscCheck(face->blind, PETSC_COMM_SELF, PETSC_ERR_LIB, "Face %d is corrupted: %d %d %d", f, jj, ii, kk);

          // Evaluate new (x, y, z) Point Position based on new Surface Definition
          double newCoords[18];
          if (islite) PetscCall(EGlite_getRange(newgeom, range, &peri));
          else PetscCall(EG_getRange(newgeom, range, &peri));

          PetscCall(DMPlex_Geom_FACE_XYZtoUV_Internal(coords, face, range, 0, dE, params, islite));
          PetscCheck(face->blind, PETSC_COMM_SELF, PETSC_ERR_LIB, "Face %d is corrupted: %d %d %d", f, jj, ii, kk);

          if (islite) PetscCall(EGlite_evaluate(newgeom, params, newCoords));
          else PetscCall(EG_evaluate(newgeom, params, newCoords));

          // Calculate Surface Area Gradients wrt Control Points and Weights using the local discrete FACE only
          //      NOTE 1: Will not provide Volume Gradient wrt to Control Points and Weights.
          //      NOTE 2: This is faster than below where an entire new solid geometry is created for each
          //              Control Point and Weight gradient
          if (!fullGeomGrad) {
            // Create new FACE based on new SURFACE geometry
            if (jj == 0) { // only for 1st DMPlex Point because we only per CP or Weight
              double newFaceRange[4];
              int    newFacePeri;
              if (islite) PetscCall(EGlite_getRange(newgeom, newFaceRange, &newFacePeri));
              else PetscCall(EG_getRange(newgeom, newFaceRange, &newFacePeri));

              ego newface;
              PetscCallEGADS(EG_makeFace, (newgeom, SFORWARD, newFaceRange, &newface)); // Does not have EGlite version KNOWN_ISSUE
              PetscCheck(face->blind, PETSC_COMM_SELF, PETSC_ERR_LIB, "Face %d is corrupted: %d %d %d", f, jj, ii, kk);

              // Get New Face Surface Area
              PetscScalar newfSA, newFaceData[14];
              PetscCall(EG_getMassProperties(newface, newFaceData)); // Does not have EGlite version KNOWN_ISSUE
              newfSA = newFaceData[1];
              PetscCallEGADS(EG_deleteObject, (newface));
              PetscCheck(face->blind, PETSC_COMM_SELF, PETSC_ERR_LIB, "Face %d is corrupted: %d %d %d", f, jj, ii, kk);

              // Update Control Points
              PetscHashIter CPiter, Witer;
              PetscBool     CPfound, Wfound;
              PetscInt      faceCPStartRow, faceWStartRow;

              PetscScalar dSAdCPi;
              dSAdCPi = (newfSA - fSA) / (denomNew - denomOld);

              if (kk < 3) {
                PetscCall(PetscHMapIFind(faceCntrlPtRow_Start, fid, &CPiter, &CPfound));
                PetscCheck(CPfound, PETSC_COMM_SELF, PETSC_ERR_SUP, "FACE ID not found in Control Point Hash Table");
                PetscCall(PetscHMapIGet(faceCntrlPtRow_Start, fid, &faceCPStartRow));

                gradSACP[faceCPStartRow + (ii * 3) + kk] = dSAdCPi;

                if (PetscAbsReal(dSAdCPi) > maxGrad) maxGrad = PetscAbsReal(dSAdCPi);

              } else if (kk == 3) {
                PetscCall(PetscHMapIFind(faceCPWeightsRow_Start, fid, &Witer, &Wfound));
                PetscCheck(Wfound, PETSC_COMM_SELF, PETSC_ERR_SUP, "FACE ID not found in Control Point Hash Table");
                PetscCall(PetscHMapIGet(faceCPWeightsRow_Start, fid, &faceWStartRow));

                gradSAW[faceWStartRow + ii] = dSAdCPi;

              } else {
                // Do Nothing
              }
            }
          }
          PetscCallEGADS(EG_deleteObject, (newgeom));

          // Now Calculate the Surface Gradient for the change in x-component Control Point
          PetscScalar dxdCx = (newCoords[0] - coords[0]) / deltaCoord;
          PetscScalar dxdCy = (newCoords[1] - coords[1]) / deltaCoord;
          PetscScalar dxdCz = (newCoords[2] - coords[2]) / deltaCoord;

          // Store Gradient Information in surfaceGrad[][] Matrix
          PetscInt startRow;
          PetscCall(PetscHMapIGet(pointSurfGradRow_Start, currPointID, &startRow));

          // Store Results in PETSc Mat
          PetscCall(MatSetValue(pointSurfGrad, startRow + (ii * 3) + 0, ((fid - 1) * 4) + kk, dxdCx, INSERT_VALUES));
          PetscCall(MatSetValue(pointSurfGrad, startRow + (ii * 3) + 1, ((fid - 1) * 4) + kk, dxdCy, INSERT_VALUES));
          PetscCall(MatSetValue(pointSurfGrad, startRow + (ii * 3) + 2, ((fid - 1) * 4) + kk, dxdCz, INSERT_VALUES));

          //PetscCallEGADS(EG_deleteObject, (newgeom));
          PetscCheck(face->blind, PETSC_COMM_SELF, PETSC_ERR_LIB, "Face is corrupted");
        }
        offset += 3;
      }
      PetscCall(DMPlexVecRestoreClosure(cdm, NULL, coordinatesLocal, currPointID, &Nv, &coords));
    }
  }

  // Assemble Point Surface Grad Matrix
  PetscCall(MatAssemblyBegin(pointSurfGrad, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(pointSurfGrad, MAT_FINAL_ASSEMBLY));

  if (fullGeomGrad) {
    // Calculate Surface Area and Volume Control Point and Control Point Weight Gradients
    //    Note: This is much slower than above due to a new solid geometry being created for
    //          each change in Control Point and Control Point Weight. However, this method
    //          will provide the Volume Gradient.

    // Get Current Face Surface Area
    PetscScalar bodyVol, bodySA, bodyData[14];
    PetscCall(EG_getMassProperties(body, bodyData)); // Does not have an EGlite version KNOWN_ISSUE
    bodyVol = bodyData[0];
    bodySA  = bodyData[1];

    // Cycle through Control Points
    for (int ii = 0; ii < totalNumCPs; ++ii) { // ii should also be the row in cpEquiv for the Control Point
      // Cycle through X, Y, Z, W changes
      for (int jj = 0; jj < 4; ++jj) {
        // Cycle Through Faces
        double denomNew = 0.0, denomOld = 0.0;
        double deltaCoord = 1.0E-4;
        ego    newGeom[Nf];
        ego    newFaces[Nf];
        for (int kk = 0; kk < Nf; ++kk) {
          ego      face;
          PetscInt currFID = kk + 1;

          if (islite) {
            // Get Current FACE
            PetscCallEGADS(EGlite_objectBodyTopo, (body, FACE, currFID, &face));

            // Get Geometry Object for the Current FACE
            PetscCallEGADS(EGlite_getTopology, (face, &fgeom, &foclass, &fmtype, fdata, &Nl, &lobjs, &lsenses));
            PetscCallEGADS(EGlite_getGeometry, (fgeom, &boclass, &bmtype, &bRef, &bpinfo, &bprv));
          } else {
            // Get Current FACE
            PetscCallEGADS(EG_objectBodyTopo, (body, FACE, currFID, &face));

            // Get Geometry Object for the Current FACE
            PetscCallEGADS(EG_getTopology, (face, &fgeom, &foclass, &fmtype, fdata, &Nl, &lobjs, &lsenses));
            PetscCallEGADS(EG_getGeometry, (fgeom, &boclass, &bmtype, &bRef, &bpinfo, &bprv));
          }

          // Make a new SURFACE Geometry by changing the location of the Control Points
          int    prvSize = bpinfo[3] + bpinfo[6] + (4 * bpinfo[2] * bpinfo[5]);
          double nbprv[prvSize];

          // Reinitialize nbprv[] values because we only want to change one value at a time
          for (int mm = 0; mm < prvSize; ++mm) nbprv[mm] = bprv[mm];

          // Get Control Point Row and Column Start for cpEquiv
          PetscHashIter Witer;
          PetscBool     Wfound;
          PetscInt      faceWStartRow;
          PetscCall(PetscHMapIFind(faceCPWeightsRow_Start, currFID, &Witer, &Wfound));
          PetscCheck(Wfound, PETSC_COMM_SELF, PETSC_ERR_SUP, "FACE ID not found in Control Point Weights Hash Table");
          PetscCall(PetscHMapIGet(faceCPWeightsRow_Start, currFID, &faceWStartRow));

          // Modify the Current Control Point on this FACE and All Other FACES
          // IMPORTANT!!! If you do not move all identical Control Points on other FACES
          //              you will not generate a solid body. You will generate a set of
          //              disconnected surfaces that have gap(s) between them.
          int offset  = bpinfo[3] + bpinfo[6];
          int wOffset = offset + (3 * bpinfo[2] * bpinfo[5]);
          for (int mm = 0; mm < bpinfo[2] * bpinfo[5]; ++mm) {
            PetscScalar matValue;
            PetscCall(MatGetValue(cpEquiv, ii, faceWStartRow + mm, &matValue));

            if (matValue > 0.0) {
              if (jj == 0) { //X
                nbprv[offset + (3 * mm) + 0] = bprv[offset + (3 * mm) + 0] + deltaCoord;
                nbprv[offset + (3 * mm) + 1] = bprv[offset + (3 * mm) + 1];
                nbprv[offset + (3 * mm) + 2] = bprv[offset + (3 * mm) + 2];
                denomNew                     = nbprv[offset + (3 * mm) + 0];
                denomOld                     = bprv[offset + (3 * mm) + 0];
              } else if (jj == 1) { //Y
                nbprv[offset + (3 * mm) + 0] = bprv[offset + (3 * mm) + 0];
                nbprv[offset + (3 * mm) + 1] = bprv[offset + (3 * mm) + 1] + deltaCoord;
                nbprv[offset + (3 * mm) + 2] = bprv[offset + (3 * mm) + 2];
                denomNew                     = nbprv[offset + (3 * mm) + 1];
                denomOld                     = bprv[offset + (3 * mm) + 1];
              } else if (jj == 2) { //Z
                nbprv[offset + (3 * mm) + 0] = bprv[offset + (3 * mm) + 0];
                nbprv[offset + (3 * mm) + 1] = bprv[offset + (3 * mm) + 1];
                nbprv[offset + (3 * mm) + 2] = bprv[offset + (3 * mm) + 2] + deltaCoord;
                denomNew                     = nbprv[offset + (3 * mm) + 2];
                denomOld                     = bprv[offset + (3 * mm) + 2];
              } else if (jj == 3) { // Weights
                nbprv[wOffset + mm] = bprv[wOffset + mm] + deltaCoord;
                denomNew            = nbprv[wOffset + mm];
                denomOld            = bprv[wOffset + mm];
              } else {
                // currently do nothing
              }
            }
          }

          // Create New Surface Based on New Control Points or Weights
          ego newgeom, context;
          PetscCallEGADS(EG_getContext, (face, &context));                                             // Does not have an EGlite_ versions   KNOWN_ISSUE
          PetscCallEGADS(EG_makeGeometry, (context, SURFACE, BSPLINE, NULL, bpinfo, nbprv, &newgeom)); // Does not have an EGlite_ version KNOWN_ISSUE

          // Create New FACE based on modified geometry
          double newFaceRange[4];
          int    newFacePeri;
          if (islite) PetscCallEGADS(EGlite_getRange, (newgeom, newFaceRange, &newFacePeri));
          else PetscCallEGADS(EG_getRange, (newgeom, newFaceRange, &newFacePeri));

          ego newface;
          PetscCallEGADS(EG_makeFace, (newgeom, SFORWARD, newFaceRange, &newface)); // Does not have an EGlite_ version KNOWN_ISSUE

          // store new face for later assembly
          newGeom[kk]  = newgeom;
          newFaces[kk] = newface;
        }

        // X-WANT TO BUILD THE NEW GEOMETRY, X-GET NEW SA AND PERFORM dSA/dCPi CALCS HERE <---
        // Sew New Faces together to get a new model
        ego newmodel;
        PetscCall(EG_sewFaces(Nf, newFaces, 0.0, 0, &newmodel)); // Does not have an EGlite_ version KNOWN_ISSUE

        // Get Surface Area and Volume of New/Updated Solid Body
        PetscScalar newData[14];
        if (islite) PetscCallEGADS(EGlite_getTopology, (newmodel, &geom, &oclass, &mtype, NULL, &Nb, &bodies, &senses));
        else PetscCallEGADS(EG_getTopology, (newmodel, &geom, &oclass, &mtype, NULL, &Nb, &bodies, &senses));

        ego nbody = bodies[0];
        PetscCall(EG_getMassProperties(nbody, newData)); // Does not have an EGlite_ version   KNOWN_ISSUE

        PetscScalar dSAdCPi, dVdCPi;
        PetscScalar nbodyVol = newData[0], nbodySA = newData[1];

        // Calculate Gradients wrt to Control Points and Control Points Weights depending on jj value
        dSAdCPi = (nbodySA - bodySA) / (denomNew - denomOld);
        dVdCPi  = (nbodyVol - bodyVol) / (denomNew - denomOld);

        if (jj < 3) {
          // Gradienst wrt to Control Points
          gradSACP[(ii * 3) + jj] = dSAdCPi;
          gradVCP[(ii * 3) + jj]  = dVdCPi;
        } else if (jj == 3) {
          // Gradients wrt to Control Point Weights
          gradSAW[ii] = dSAdCPi;
          gradVW[ii]  = dVdCPi;
        } else {
          // Do Nothing
        }
        PetscCallEGADS(EG_deleteObject, (newmodel));
        for (int kk = 0; kk < Nf; ++kk) {
          PetscCallEGADS(EG_deleteObject, (newFaces[kk]));
          PetscCallEGADS(EG_deleteObject, (newGeom[kk]));
        }
      }
    }
  }
  PetscCall(VecRestoreArrayWrite(gradSACPVec, &gradSACP));
  PetscCall(VecRestoreArrayWrite(gradSAWVec, &gradSAW));
  PetscCall(VecRestoreArrayWrite(gradVCPVec, &gradVCP));
  PetscCall(VecRestoreArrayWrite(gradVWVec, &gradVW));
  PetscCall(MatDestroy(&cpEquiv));

  // Attach Surface Gradient Hash Table and Matrix to DM
  {
    PetscContainer surfGradOrgObj;

    PetscCall(PetscObjectQuery((PetscObject)dm, "Surface Gradient Hash Table", (PetscObject *)&surfGradOrgObj));
    if (!surfGradOrgObj) {
      PetscCall(PetscContainerCreate(PETSC_COMM_SELF, &surfGradOrgObj));
      PetscCall(PetscContainerSetPointer(surfGradOrgObj, pointSurfGradRow_Start));
      PetscCall(PetscContainerSetCtxDestroy(surfGradOrgObj, DestroyHashMap));
      PetscCall(PetscObjectCompose((PetscObject)dm, "Surface Gradient Hash Table", (PetscObject)surfGradOrgObj));
      PetscCall(PetscContainerDestroy(&surfGradOrgObj));
    } else {
      PetscCall(PetscContainerSetPointer(surfGradOrgObj, pointSurfGradRow_Start));
    }

    PetscCall(PetscObjectCompose((PetscObject)dm, "Surface Gradient Matrix", (PetscObject)pointSurfGrad));
    PetscCall(MatDestroy(&pointSurfGrad));

    PetscCall(PetscObjectCompose((PetscObject)dm, "Surface Area Control Point Gradient", (PetscObject)gradSACPVec));
    PetscCall(VecDestroy(&gradSACPVec));

    PetscCall(PetscObjectCompose((PetscObject)dm, "Surface Area Weights Gradient", (PetscObject)gradSAWVec));
    PetscCall(VecDestroy(&gradSAWVec));

    if (fullGeomGrad) {
      PetscCall(PetscObjectCompose((PetscObject)dm, "Volume Control Point Gradient", (PetscObject)gradVCPVec));
      PetscCall(PetscObjectCompose((PetscObject)dm, "Volume Weights Gradient", (PetscObject)gradVWVec));
    }
    PetscCall(VecDestroy(&gradVCPVec));
    PetscCall(VecDestroy(&gradVWVec));
  }

  // Could be replaced with DMPlexFreeGeomObject()
  if (islite) EGlite_free(fobjs);
  else EG_free(fobjs);
  PetscFunctionReturn(PETSC_SUCCESS);
#else
  SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "This method requires EGADS support. Reconfigure using --download-egads");
#endif
}

/*@C
  DMPlexModifyGeomModel - Generates a new EGADS geometry model based in user provided Control Points and Control Points Weights. Optionally, the function will inflate the DM to the new geometry and save the new geometry to a file.

  Collective

  Input Parameters:
+ dm          - The DM object representing the mesh with PetscContainer containing an EGADS geometry model
. comm        - MPI_Comm object
. newCP       - C Array of [x, y, z] New/Updated Control Point Coordinates defining the geometry (See DMPlexGeomDataAndGrads() for format)
. newW        - C Array of New/Updated Control Point Weights associated with the Control Points defining the new geometry (See DMPlexGemGrads() for format)
. autoInflate - PetscBool Flag denoting if the user would like to inflate the DM points to the new geometry.
. saveGeom    - PetscBool Flag denoting if the user would iike to save the new geometry to a file.
- stpName     - Char Array indicating the name of the file to save the new geometry to. Extension must be included and will denote type of file written.
                      *.stp or *.step = STEP File
                      *.igs or *.iges = IGES File
                              *.egads = EGADS File
                               *.brep = BRep File (OpenCASCADE File)

  Output Parameter:
. dm - The updated DM object representing the mesh with PetscContainers containing the updated/modified geometry

  Level: intermediate

  Note:
  Functionality not available for DMPlexes with attached EGADSlite geometry files (.egadslite).

.seealso: `DMPLEX`, `DMCreate()`, `DMPlexCreateGeom()`, `DMPlexGeomDataAndGrads()`
@*/
PetscErrorCode DMPlexModifyGeomModel(DM dm, MPI_Comm comm, PetscScalar newCP[], PetscScalar newW[], PetscBool autoInflate, PetscBool saveGeom, const char *stpName) PeNS
{
#if defined(PETSC_HAVE_EGADS)
  /* EGADS/EGADSlite variables */
  ego context, model, geom, *bodies, *lobjs, *fobjs;
  int oclass, mtype, *senses, *lsenses;
  int Nb, Nf, Nl, id;
  /* PETSc variables */
  DMLabel        bodyLabel, faceLabel, edgeLabel, vertexLabel;
  PetscContainer modelObj, cpHashTableObj, wHashTableObj;
  PetscHMapI     cpHashTable = NULL, wHashTable = NULL;
  PetscBool      islite = PETSC_FALSE;
#endif

#if defined(PETSC_HAVE_EGADS)
  PetscFunctionBegin;
  // Look to see if DM has a Container with either a EGADS or EGADSlite Model
  PetscCall(PetscObjectQuery((PetscObject)dm, "EGADS Model", (PetscObject *)&modelObj));
  if (!modelObj) {
    PetscCall(PetscObjectQuery((PetscObject)dm, "EGADSlite Model", (PetscObject *)&modelObj));
    islite = PETSC_TRUE;
  }
  PetscCheck(!islite, PETSC_COMM_SELF, PETSC_ERR_SUP, "Cannot modify geometries defined by EGADSlite (.egadslite)! Please use another geometry file format STEP, IGES, EGADS or BRep");
  PetscCheck(modelObj, PETSC_COMM_SELF, PETSC_ERR_SUP, "DM does not have a EGADS Geometry Model attached to it!");

  // Get attached EGADS model (pointer)
  PetscCall(PetscContainerGetPointer(modelObj, (void **)&model));

  // Look to see if DM has Container for Geometry Control Point Data
  PetscCall(PetscObjectQuery((PetscObject)dm, "Control Point Hash Table", (PetscObject *)&cpHashTableObj));
  PetscCall(PetscObjectQuery((PetscObject)dm, "Control Point Weights Hash Table", (PetscObject *)&wHashTableObj));

  PetscCheck(cpHashTableObj && wHashTableObj, PETSC_COMM_SELF, PETSC_ERR_SUP, "DM does not have required Geometry Data attached! Please run DMPlexGeomDataAndGrads() Function first.");

  // Get attached EGADS model Control Point and Weights Hash Tables and Data Arrays (pointer)
  PetscCall(PetscContainerGetPointer(cpHashTableObj, (void **)&cpHashTable));
  PetscCall(PetscContainerGetPointer(wHashTableObj, (void **)&wHashTable));

  // Get the number of bodies and body objects in the model
  if (islite) PetscCallEGADS(EGlite_getTopology, (model, &geom, &oclass, &mtype, NULL, &Nb, &bodies, &senses));
  else PetscCallEGADS(EG_getTopology, (model, &geom, &oclass, &mtype, NULL, &Nb, &bodies, &senses));

  // Get all Faces on the body
  ego body = bodies[0];
  if (islite) PetscCallEGADS(EGlite_getBodyTopos, (body, NULL, FACE, &Nf, &fobjs));
  else PetscCallEGADS(EG_getBodyTopos, (body, NULL, FACE, &Nf, &fobjs));

  ego newGeom[Nf];
  ego newFaces[Nf];

  // Update Control Point and Weight definitions for each surface
  for (int jj = 0; jj < Nf; ++jj) {
    ego     face = fobjs[jj];
    ego     bRef, bPrev, bNext;
    ego     fgeom;
    int     offset;
    int     boclass, bmtype, *bpinfo;
    double *bprv;

    // Get FACE ID and other Geometry Data
    if (islite) {
      id = EGlite_indexBodyTopo(body, face);
      PetscCallEGADS(EGlite_getTopology, (face, &fgeom, &oclass, &mtype, NULL, &Nl, &lobjs, &lsenses));
      PetscCallEGADS(EGlite_getGeometry, (fgeom, &boclass, &bmtype, &bRef, &bpinfo, &bprv));
      PetscCallEGADS(EGlite_getInfo, (fgeom, &boclass, &bmtype, &bRef, &bPrev, &bNext));
    } else {
      id = EG_indexBodyTopo(body, face);
      PetscCallEGADS(EG_getTopology, (face, &fgeom, &oclass, &mtype, NULL, &Nl, &lobjs, &lsenses));
      PetscCallEGADS(EG_getGeometry, (fgeom, &boclass, &bmtype, &bRef, &bpinfo, &bprv));
      PetscCallEGADS(EG_getInfo, (fgeom, &boclass, &bmtype, &bRef, &bPrev, &bNext));
    }

    // Update Control Points
    PetscHashIter CPiter, Witer;
    PetscBool     CPfound, Wfound;
    PetscInt      faceCPStartRow, faceWStartRow;

    PetscCall(PetscHMapIFind(cpHashTable, id, &CPiter, &CPfound));
    PetscCheck(CPfound, PETSC_COMM_SELF, PETSC_ERR_SUP, "FACE ID not found in Control Point Hash Table");
    PetscCall(PetscHMapIGet(cpHashTable, id, &faceCPStartRow));

    PetscCall(PetscHMapIFind(wHashTable, id, &Witer, &Wfound));
    PetscCheck(Wfound, PETSC_COMM_SELF, PETSC_ERR_SUP, "FACE ID not found in Control Point Weights Hash Table");
    PetscCall(PetscHMapIGet(wHashTable, id, &faceWStartRow));

    // UPDATE CONTROL POINTS Locations
    offset = bpinfo[3] + bpinfo[6];
    for (int ii = 0; ii < 3 * bpinfo[2] * bpinfo[5]; ++ii) { bprv[offset + ii] = newCP[faceCPStartRow + ii]; }

    // UPDATE CONTROL POINT WEIGHTS
    offset = bpinfo[3] + bpinfo[6] + 3 * bpinfo[2] * bpinfo[5];
    for (int ii = 0; ii < bpinfo[2] * bpinfo[5]; ++ii) { bprv[offset + ii] = newW[faceWStartRow + ii]; }

    // Get Context from FACE
    context = NULL;
    PetscCallEGADS(EG_getContext, (face, &context)); // Does not have an EGlite_ version  KNOWN_ISSUE

    // Create New Surface
    ego newgeom;
    PetscCallEGADS(EG_makeGeometry, (context, SURFACE, BSPLINE, NULL, bpinfo, bprv, &newgeom)); // Does not have an EGlite_ version KNOWN_ISSUE

    // Create new FACE based on new SURFACE geometry
    double data[4];
    int    periodic;
    if (islite) PetscCallEGADS(EGlite_getRange, (newgeom, data, &periodic));
    else PetscCallEGADS(EG_getRange, (newgeom, data, &periodic));

    ego newface;
    PetscCallEGADS(EG_makeFace, (newgeom, SFORWARD, data, &newface)); // Does not have an EGlite_ version KNOWN_ISSUE
    newGeom[jj]  = newgeom;
    newFaces[jj] = newface;
  }
  // Could be replaced by DMPlexFreeGeomObject
  if (islite) EGlite_free(fobjs);
  else EG_free(fobjs);

  // Sew New Faces together to get a new model
  ego newmodel;
  PetscCall(EG_sewFaces(Nf, newFaces, 0.0, 0, &newmodel)); // Does not have an EGlite_ version   KNOWN_ISSUE
  for (PetscInt f = 0; f < Nf; ++f) {
    PetscCallEGADS(EG_deleteObject, (newFaces[f]));
    PetscCallEGADS(EG_deleteObject, (newGeom[f]));
  }

  // Get the total number of NODEs on the original geometry. (This will be the same for the new geometry)
  int  totalNumNode;
  ego *nobjTotal;
  if (islite) {
    PetscCallEGADS(EGlite_getBodyTopos, (body, NULL, NODE, &totalNumNode, &nobjTotal));
    EGlite_free(nobjTotal);
  } else {
    PetscCallEGADS(EG_getBodyTopos, (body, NULL, NODE, &totalNumNode, &nobjTotal));
    EG_free(nobjTotal);
  } // Could be replaced with DMPlexFreeGeomObject

  // Initialize vector to store equivalent NODE indices between the 2 geometries
  // FORMAT :: vector index is the Original Geometry's NODE ID, the vector Value is the New Geometry's NODE ID
  int nodeIDEquiv[totalNumNode + 1];

  // Now we need to Map the NODE and EDGE IDs from each Model
  if (islite) PetscCallEGADS(EGlite_getBodyTopos, (body, NULL, FACE, &Nf, &fobjs));
  else PetscCallEGADS(EG_getBodyTopos, (body, NULL, FACE, &Nf, &fobjs));

  // New CAD
  ego *newbodies, newgeomtest, *nfobjs;
  int  nNf, newNb, newoclass, newmtype, *newsenses;
  if (islite) PetscCallEGADS(EGlite_getTopology, (newmodel, &newgeomtest, &newoclass, &newmtype, NULL, &newNb, &newbodies, &newsenses));
  else PetscCallEGADS(EG_getTopology, (newmodel, &newgeomtest, &newoclass, &newmtype, NULL, &newNb, &newbodies, &newsenses));

  ego newbody = newbodies[0];
  if (islite) PetscCallEGADS(EGlite_getBodyTopos, (newbody, NULL, FACE, &nNf, &nfobjs));
  else PetscCallEGADS(EG_getBodyTopos, (newbody, NULL, FACE, &nNf, &nfobjs));

  PetscCheck(newNb == 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "ERROR :: newNb > 1 || newNb = %d", newNb);

  // Find Equivalent Nodes
  for (int ii = 0; ii < Nf; ++ii) {
    double fdata[4];
    int    peri;

    // Get Current FACE [u, v] Ranges
    if (islite) PetscCallEGADS(EGlite_getRange, (fobjs[ii], fdata, &peri));
    else PetscCallEGADS(EG_getRange, (fobjs[ii], fdata, &peri));

    // Equate NODE IDs between 2 FACEs by working through (u, v) limits of FACE
    for (int jj = 0; jj < 2; ++jj) {
      for (int kk = 2; kk < 4; ++kk) {
        double params[2] = {fdata[jj], fdata[kk]};
        double eval[18];
        if (islite) PetscCallEGADS(EGlite_evaluate, (fobjs[ii], params, eval));
        else PetscCallEGADS(EG_evaluate, (fobjs[ii], params, eval));

        // Original Body
        ego *nobjsOrigFace;
        int  origNn;
        if (islite) PetscCallEGADS(EGlite_getBodyTopos, (body, fobjs[ii], NODE, &origNn, &nobjsOrigFace));
        else PetscCallEGADS(EG_getBodyTopos, (body, fobjs[ii], NODE, &origNn, &nobjsOrigFace));

        double minVal = 1.0E10;
        double evalCheck[18];
        int    equivOrigNodeID = -1;
        for (int mm = 0; mm < origNn; ++mm) {
          double delta = 1.0E10;
          if (islite) PetscCallEGADS(EGlite_evaluate, (nobjsOrigFace[mm], NULL, evalCheck));
          else PetscCallEGADS(EG_evaluate, (nobjsOrigFace[mm], NULL, evalCheck));

          delta = PetscSqrtReal(PetscSqr(evalCheck[0] - eval[0]) + PetscSqr(evalCheck[1] - eval[1]) + PetscSqr(evalCheck[2] - eval[2]));

          if (delta < minVal) {
            if (islite) equivOrigNodeID = EGlite_indexBodyTopo(body, nobjsOrigFace[mm]);
            else equivOrigNodeID = EG_indexBodyTopo(body, nobjsOrigFace[mm]);

            minVal = delta;
          }
        }
        // Could be replaced with DMPlexFreeGeomObject
        if (islite) EGlite_free(nobjsOrigFace);
        else EG_free(nobjsOrigFace);

        // New Body
        ego *nobjsNewFace;
        int  newNn;
        if (islite) PetscCallEGADS(EGlite_getBodyTopos, (newbody, nfobjs[ii], NODE, &newNn, &nobjsNewFace));
        else PetscCallEGADS(EG_getBodyTopos, (newbody, nfobjs[ii], NODE, &newNn, &nobjsNewFace));

        minVal             = 1.0E10;
        int equivNewNodeID = -1;
        for (int mm = 0; mm < newNn; ++mm) {
          double delta = 1.0E10;
          if (islite) PetscCallEGADS(EGlite_evaluate, (nobjsNewFace[mm], NULL, evalCheck));
          else PetscCallEGADS(EG_evaluate, (nobjsNewFace[mm], NULL, evalCheck));

          delta = PetscSqrtReal(PetscSqr(evalCheck[0] - eval[0]) + PetscSqr(evalCheck[1] - eval[1]) + PetscSqr(evalCheck[2] - eval[2]));

          if (delta < minVal) {
            if (islite) equivNewNodeID = EGlite_indexBodyTopo(newbody, nobjsNewFace[mm]);
            else equivNewNodeID = EG_indexBodyTopo(newbody, nobjsNewFace[mm]);

            minVal = delta;
          }
        }
        if (islite) EGlite_free(nobjsNewFace);
        else EG_free(nobjsNewFace);

        // Store equivalent NODE IDs
        nodeIDEquiv[equivOrigNodeID] = equivNewNodeID;
      }
    }
  }

  // Find Equivalent EDGEs
  //   Get total number of EDGEs on Original Geometry
  int  totalNumEdge;
  ego *eobjsOrig;
  if (islite) {
    PetscCallEGADS(EGlite_getBodyTopos, (body, NULL, EDGE, &totalNumEdge, &eobjsOrig));
    EGlite_free(eobjsOrig);
  } else {
    PetscCallEGADS(EG_getBodyTopos, (body, NULL, EDGE, &totalNumEdge, &eobjsOrig));
    EG_free(eobjsOrig);
  }

  //   Get total number of EDGEs on New Geometry
  int  totalNumEdgeNew;
  ego *eobjsNew;
  if (islite) {
    PetscCallEGADS(EGlite_getBodyTopos, (newbody, NULL, EDGE, &totalNumEdgeNew, &eobjsNew));
    EGlite_free(eobjsNew);
  } else {
    PetscCallEGADS(EG_getBodyTopos, (newbody, NULL, EDGE, &totalNumEdgeNew, &eobjsNew));
    EG_free(eobjsNew);
  }

  // Initialize EDGE ID equivalent vector
  // FORMAT :: vector index is the Original Geometry's EDGE ID, the vector Value is the New Geometry's EDGE ID
  int edgeIDEquiv[totalNumEdge + 1];

  // Find Equivalent EDGEs
  for (int ii = 0; ii < Nf; ++ii) {
    // Get Original Geometry EDGE's NODEs
    int numOrigEdge, numNewEdge;
    if (islite) {
      PetscCallEGADS(EGlite_getBodyTopos, (body, fobjs[ii], EDGE, &numOrigEdge, &eobjsOrig));
      PetscCallEGADS(EGlite_getBodyTopos, (newbody, nfobjs[ii], EDGE, &numNewEdge, &eobjsNew));
    } else {
      PetscCallEGADS(EG_getBodyTopos, (body, fobjs[ii], EDGE, &numOrigEdge, &eobjsOrig));
      PetscCallEGADS(EG_getBodyTopos, (newbody, nfobjs[ii], EDGE, &numNewEdge, &eobjsNew));
    }

    // new loop below
    for (int nn = 0; nn < numOrigEdge; ++nn) {
      ego origEdge = eobjsOrig[nn];
      ego geomEdgeOrig, *nobjsOrig;
      int oclassEdgeOrig, mtypeEdgeOrig;
      int NnOrig, *nsensesEdgeOrig;

      if (islite) PetscCallEGADS(EGlite_getTopology, (origEdge, &geomEdgeOrig, &oclassEdgeOrig, &mtypeEdgeOrig, NULL, &NnOrig, &nobjsOrig, &nsensesEdgeOrig));
      else PetscCallEGADS(EG_getTopology, (origEdge, &geomEdgeOrig, &oclassEdgeOrig, &mtypeEdgeOrig, NULL, &NnOrig, &nobjsOrig, &nsensesEdgeOrig));

      PetscBool isSame = PETSC_FALSE;
      for (int jj = 0; jj < numNewEdge; ++jj) {
        ego newEdge = eobjsNew[jj];
        ego geomEdgeNew, *nobjsNew;
        int oclassEdgeNew, mtypeEdgeNew;
        int NnNew, *nsensesEdgeNew;

        if (islite) PetscCallEGADS(EGlite_getTopology, (newEdge, &geomEdgeNew, &oclassEdgeNew, &mtypeEdgeNew, NULL, &NnNew, &nobjsNew, &nsensesEdgeNew));
        else PetscCallEGADS(EG_getTopology, (newEdge, &geomEdgeNew, &oclassEdgeNew, &mtypeEdgeNew, NULL, &NnNew, &nobjsNew, &nsensesEdgeNew));

        if (mtypeEdgeOrig == mtypeEdgeNew) {
          // Only operate if the EDGE types are the same
          for (int kk = 0; kk < NnNew; ++kk) {
            int nodeIDOrigGeom, nodeIDNewGeom;
            if (islite) {
              nodeIDOrigGeom = EGlite_indexBodyTopo(body, nobjsOrig[kk]);
              nodeIDNewGeom  = EGlite_indexBodyTopo(newbody, nobjsNew[kk]);
            } else {
              nodeIDOrigGeom = EG_indexBodyTopo(body, nobjsOrig[kk]);
              nodeIDNewGeom  = EG_indexBodyTopo(newbody, nobjsNew[kk]);
            }

            if (nodeIDNewGeom == nodeIDEquiv[nodeIDOrigGeom]) {
              isSame = PETSC_TRUE;
            } else {
              isSame = PETSC_FALSE;
              kk     = NnNew; // skip ahead because first NODE failed test and order is important
            }
          }

          if (isSame == PETSC_TRUE) {
            int edgeIDOrig, edgeIDNew;
            if (islite) {
              edgeIDOrig = EGlite_indexBodyTopo(body, origEdge);
              edgeIDNew  = EGlite_indexBodyTopo(newbody, newEdge);
            } else {
              edgeIDOrig = EG_indexBodyTopo(body, origEdge);
              edgeIDNew  = EG_indexBodyTopo(newbody, newEdge);
            }

            edgeIDEquiv[edgeIDOrig] = edgeIDNew;
            jj                      = numNewEdge;
          }
        }
      }
    }
    if (islite) {
      EGlite_free(eobjsOrig);
      EGlite_free(eobjsNew);
    } else {
      EG_free(eobjsOrig);
      EG_free(eobjsNew);
    }
  }
  if (islite) {
    EGlite_free(fobjs);
    EGlite_free(nfobjs);
  } else {
    EG_free(fobjs);
    EG_free(nfobjs);
  }

  // Modify labels to point to the IDs on the new Geometry
  IS isNodeID, isEdgeID;

  PetscCall(DMGetLabel(dm, "EGADS Body ID", &bodyLabel));
  PetscCall(DMGetLabel(dm, "EGADS Face ID", &faceLabel));
  PetscCall(DMGetLabel(dm, "EGADS Edge ID", &edgeLabel));
  PetscCall(DMGetLabel(dm, "EGADS Vertex ID", &vertexLabel));

  PetscCall(ISCreateGeneral(comm, totalNumNode + 1, nodeIDEquiv, PETSC_COPY_VALUES, &isNodeID));
  PetscCall(ISCreateGeneral(comm, totalNumEdge + 1, edgeIDEquiv, PETSC_COPY_VALUES, &isEdgeID));
  /* Do not perform check. Np may != Nv due to Degenerate Geometry which is not stored in labels.               */
  /* We do not know in advance which IDs have been omitted. This may also change due to geometry modifications. */
  PetscCall(DMLabelRewriteValues(vertexLabel, isNodeID));
  PetscCall(DMLabelRewriteValues(edgeLabel, isEdgeID));
  PetscCall(ISDestroy(&isNodeID));
  PetscCall(ISDestroy(&isEdgeID));

  // Attempt to point to the new geometry
  PetscCallEGADS(EG_deleteObject, (model));
  PetscCall(PetscContainerSetPointer(modelObj, newmodel));

  // save updated model to file
  if (saveGeom == PETSC_TRUE && stpName != NULL) PetscCall(EG_saveModel(newmodel, stpName));

  // Inflate Mesh to EGADS Model
  if (autoInflate == PETSC_TRUE) PetscCall(DMPlexInflateToGeomModel(dm, PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
#else
  SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "This method requires EGADS support. Reconfigure using --download-egads");
#endif
}

/*@C
  DMPlexGetGeomModelTUV - Gets the [t] (EDGES) and [u, v] (FACES) geometry parameters of DM points that are associated geometry relationships. Requires a DM with a EGADS model attached.

  Collective

  Input Parameter:
. dm - The DM object representing the mesh with PetscContainer containing an EGADS geometry model

  Level: intermediate

.seealso: `DMPLEX`, `DMCreate()`, `DMPlexCreateGeom()`, `DMPlexGeomDataAndGrads()`
@*/
PetscErrorCode DMPlexGetGeomModelTUV(DM dm) PeNS
{
#if defined(PETSC_HAVE_EGADS)
  /* EGADS Variables */
  ego    model, geom, body, face, edge;
  ego   *bodies;
  int    Nb, oclass, mtype, *senses;
  double result[4];
  /* PETSc Variables */
  DM             cdm;
  PetscContainer modelObj;
  DMLabel        bodyLabel, faceLabel, edgeLabel, vertexLabel;
  Vec            coordinates;
  PetscScalar   *coords;
  PetscInt       bodyID, faceID, edgeID, vertexID;
  PetscInt       cdim, vStart, vEnd, v;
  PetscBool      islite = PETSC_FALSE;
#endif

  PetscFunctionBegin;
#if defined(PETSC_HAVE_EGADS)
  PetscCall(PetscObjectQuery((PetscObject)dm, "EGADS Model", (PetscObject *)&modelObj));
  if (!modelObj) {
    PetscCall(PetscObjectQuery((PetscObject)dm, "EGADSlite Model", (PetscObject *)&modelObj));
    islite = PETSC_TRUE;
  }
  if (!modelObj) PetscFunctionReturn(0);

  PetscCall(DMGetCoordinateDim(dm, &cdim));
  PetscCall(DMGetCoordinateDM(dm, &cdm));
  PetscCall(DMGetCoordinatesLocal(dm, &coordinates));
  PetscCall(DMGetLabel(dm, "EGADS Body ID", &bodyLabel));
  PetscCall(DMGetLabel(dm, "EGADS Face ID", &faceLabel));
  PetscCall(DMGetLabel(dm, "EGADS Edge ID", &edgeLabel));
  PetscCall(DMGetLabel(dm, "EGADS Vertex ID", &vertexLabel));

  PetscCall(PetscContainerGetPointer(modelObj, (void **)&model));

  if (islite) PetscCall(EGlite_getTopology(model, &geom, &oclass, &mtype, NULL, &Nb, &bodies, &senses));
  else PetscCall(EG_getTopology(model, &geom, &oclass, &mtype, NULL, &Nb, &bodies, &senses));

  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  PetscCall(VecGetArrayWrite(coordinates, &coords));

  // Define t, u, v arrays to be stored in a PetscContainer after populated
  PetscScalar *t_point, *u_point, *v_point;
  PetscCall(PetscMalloc1(vEnd - vStart, &t_point));
  PetscCall(PetscMalloc1(vEnd - vStart, &u_point));
  PetscCall(PetscMalloc1(vEnd - vStart, &v_point));

  for (v = vStart; v < vEnd; ++v) {
    PetscScalar *vcoords;

    PetscCall(DMLabelGetValue(bodyLabel, v, &bodyID));
    PetscCall(DMLabelGetValue(faceLabel, v, &faceID));
    PetscCall(DMLabelGetValue(edgeLabel, v, &edgeID));
    PetscCall(DMLabelGetValue(vertexLabel, v, &vertexID));

    // TODO Figure out why this is unknown sometimes
    if (bodyID < 0 && Nb == 1) bodyID = 0;
    PetscCheck(bodyID >= 0 && bodyID < Nb, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Body %" PetscInt_FMT " for vertex %" PetscInt_FMT " is not in [0, %d)", bodyID, v, Nb);
    body = bodies[bodyID];

    PetscCall(DMPlexPointLocalRef(cdm, v, coords, (void *)&vcoords));
    if (edgeID > 0) {
      /* Snap to EDGE at nearest location */
      double params[1];

      if (islite) {
        PetscCall(EGlite_objectBodyTopo(body, EDGE, edgeID, &edge));
        PetscCall(EGlite_invEvaluate(edge, vcoords, params, result));
      } // Get (t) of nearest point on EDGE
      else {
        PetscCall(EG_objectBodyTopo(body, EDGE, edgeID, &edge));
        PetscCall(EG_invEvaluate(edge, vcoords, params, result));
      } // Get (t) of nearest point on EDGE

      t_point[v - vStart] = params[0];
      u_point[v - vStart] = 0.0;
      v_point[v - vStart] = 0.0;
    } else if (faceID > 0) {
      /* Snap to FACE at nearest location */
      double params[2];

      if (islite) {
        PetscCall(EGlite_objectBodyTopo(body, FACE, faceID, &face));
        PetscCall(EGlite_invEvaluate(face, vcoords, params, result));
      } // Get (x,y,z) of nearest point on FACE
      else {
        PetscCall(EG_objectBodyTopo(body, FACE, faceID, &face));
        PetscCall(EG_invEvaluate(face, vcoords, params, result));
      } // Get (x,y,z) of nearest point on FACE

      t_point[v - vStart] = 0.0;
      u_point[v - vStart] = params[0];
      v_point[v - vStart] = params[1];
    } else {
      t_point[v - vStart] = 0.0;
      u_point[v - vStart] = 0.0;
      v_point[v - vStart] = 0.0;
    }
  }
  PetscCall(VecRestoreArrayWrite(coordinates, &coords));
  /* Clear out global coordinates */
  PetscCall(VecDestroy(&dm->coordinates[0].x));

  /* Store in PetscContainters */
  {
    PetscContainer t_pointObj, u_pointObj, v_pointObj;

    PetscCall(PetscObjectQuery((PetscObject)dm, "Point - Edge t Parameter", (PetscObject *)&t_pointObj));
    if (!t_pointObj) {
      PetscCall(PetscContainerCreate(PETSC_COMM_SELF, &t_pointObj));
      PetscCall(PetscContainerSetPointer(t_pointObj, t_point));
      PetscCall(PetscObjectCompose((PetscObject)dm, "Point - Edge t Parameter", (PetscObject)t_pointObj));
      PetscCall(PetscContainerSetCtxDestroy(t_pointObj, PetscCtxDestroyDefault));
      PetscCall(PetscContainerDestroy(&t_pointObj));
    } else {
      void *old;

      PetscCall(PetscContainerGetPointer(t_pointObj, &old));
      PetscCall(PetscFree(old));
      PetscCall(PetscContainerSetPointer(t_pointObj, t_point));
    }

    PetscCall(PetscObjectQuery((PetscObject)dm, "Point - Face u Parameter", (PetscObject *)&u_pointObj));
    if (!u_pointObj) {
      PetscCall(PetscContainerCreate(PETSC_COMM_SELF, &u_pointObj));
      PetscCall(PetscContainerSetPointer(u_pointObj, u_point));
      PetscCall(PetscObjectCompose((PetscObject)dm, "Point - Face u Parameter", (PetscObject)u_pointObj));
      PetscCall(PetscContainerSetCtxDestroy(u_pointObj, PetscCtxDestroyDefault));
      PetscCall(PetscContainerDestroy(&u_pointObj));
    } else {
      void *old;

      PetscCall(PetscContainerGetPointer(u_pointObj, &old));
      PetscCall(PetscFree(old));
      PetscCall(PetscContainerSetPointer(u_pointObj, u_point));
    }

    PetscCall(PetscObjectQuery((PetscObject)dm, "Point - Face v Parameter", (PetscObject *)&v_pointObj));
    if (!v_pointObj) {
      PetscCall(PetscContainerCreate(PETSC_COMM_SELF, &v_pointObj));
      PetscCall(PetscContainerSetPointer(v_pointObj, v_point));
      PetscCall(PetscObjectCompose((PetscObject)dm, "Point - Face v Parameter", (PetscObject)v_pointObj));
      PetscCall(PetscContainerSetCtxDestroy(v_pointObj, PetscCtxDestroyDefault));
      PetscCall(PetscContainerDestroy(&v_pointObj));
    } else {
      void *old;

      PetscCall(PetscContainerGetPointer(v_pointObj, &old));
      PetscCall(PetscFree(old));
      PetscCall(PetscContainerSetPointer(v_pointObj, v_point));
    }
  }
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMPlexInflateToGeomModelUseTUV - Inflates the DM to the associated underlying geometry using the [t] {EDGES) and [u, v] (FACES} associated parameters. Requires a DM with an EGADS model attached and a previous call to DMPlexGetGeomModelTUV().

  Collective

  Input Parameter:
. dm - The DM object representing the mesh with PetscContainer containing an EGADS geometry model

  Level: intermediate

  Note:
  The updated DM object inflated to the associated underlying geometry. This updates the [x, y, z] coordinates of DM points associated with geometry.

.seealso: `DMPLEX`, `DMCreate()`, `DMPlexCreateGeom()`, `DMPlexGeomDataAndGrads()`, `DMPlexGetGeomModelTUV()`
@*/
PetscErrorCode DMPlexInflateToGeomModelUseTUV(DM dm) PeNS
{
#if defined(PETSC_HAVE_EGADS)
  /* EGADS Variables */
  ego    model, geom, body, face, edge, vertex;
  ego   *bodies;
  int    Nb, oclass, mtype, *senses;
  double result[18], params[2];
  /* PETSc Variables */
  DM             cdm;
  PetscContainer modelObj;
  PetscContainer t_pointObj, u_pointObj, v_pointObj;
  DMLabel        bodyLabel, faceLabel, edgeLabel, vertexLabel;
  Vec            coordinates;
  PetscScalar   *coords;
  PetscScalar   *t_point, *u_point, *v_point;
  PetscInt       bodyID, faceID, edgeID, vertexID;
  PetscInt       cdim, d, vStart, vEnd, v;
  PetscBool      islite = PETSC_FALSE;
#endif

  PetscFunctionBegin;
#if defined(PETSC_HAVE_EGADS)
  PetscCall(PetscObjectQuery((PetscObject)dm, "EGADS Model", (PetscObject *)&modelObj));
  if (!modelObj) {
    PetscCall(PetscObjectQuery((PetscObject)dm, "EGADSlite Model", (PetscObject *)&modelObj));
    islite = PETSC_TRUE;
  }

  PetscCall(PetscObjectQuery((PetscObject)dm, "Point - Edge t Parameter", (PetscObject *)&t_pointObj));
  PetscCall(PetscObjectQuery((PetscObject)dm, "Point - Face u Parameter", (PetscObject *)&u_pointObj));
  PetscCall(PetscObjectQuery((PetscObject)dm, "Point - Face v Parameter", (PetscObject *)&v_pointObj));

  if (!modelObj) PetscFunctionReturn(PETSC_SUCCESS);
  if (!t_pointObj) PetscFunctionReturn(PETSC_SUCCESS);
  if (!u_pointObj) PetscFunctionReturn(PETSC_SUCCESS);
  if (!v_pointObj) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(DMGetCoordinateDim(dm, &cdim));
  PetscCall(DMGetCoordinateDM(dm, &cdm));
  PetscCall(DMGetCoordinatesLocal(dm, &coordinates));
  PetscCall(DMGetLabel(dm, "EGADS Body ID", &bodyLabel));
  PetscCall(DMGetLabel(dm, "EGADS Face ID", &faceLabel));
  PetscCall(DMGetLabel(dm, "EGADS Edge ID", &edgeLabel));
  PetscCall(DMGetLabel(dm, "EGADS Vertex ID", &vertexLabel));

  PetscCall(PetscContainerGetPointer(t_pointObj, (void **)&t_point));
  PetscCall(PetscContainerGetPointer(u_pointObj, (void **)&u_point));
  PetscCall(PetscContainerGetPointer(v_pointObj, (void **)&v_point));

  PetscCall(PetscContainerGetPointer(modelObj, (void **)&model));

  if (islite) {
    PetscCall(EGlite_getTopology(model, &geom, &oclass, &mtype, NULL, &Nb, &bodies, &senses));
  } else {
    PetscCall(EG_getTopology(model, &geom, &oclass, &mtype, NULL, &Nb, &bodies, &senses));
  }

  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  PetscCall(VecGetArrayWrite(coordinates, &coords));

  for (v = vStart; v < vEnd; ++v) {
    PetscScalar *vcoords;

    PetscCall(DMLabelGetValue(bodyLabel, v, &bodyID));
    PetscCall(DMLabelGetValue(faceLabel, v, &faceID));
    PetscCall(DMLabelGetValue(edgeLabel, v, &edgeID));
    PetscCall(DMLabelGetValue(vertexLabel, v, &vertexID));

    // TODO Figure out why this is unknown sometimes
    if (bodyID < 0 && Nb == 1) bodyID = 0;
    PetscCheck(bodyID >= 0 && bodyID < Nb, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Body %" PetscInt_FMT " for vertex %" PetscInt_FMT " is not in [0, %d)", bodyID, v, Nb);
    body = bodies[bodyID];

    PetscCall(DMPlexPointLocalRef(cdm, v, coords, (void *)&vcoords));
    if (vertexID > 0) {
      /* Snap to Vertices */
      if (islite) {
        PetscCall(EGlite_objectBodyTopo(body, NODE, vertexID, &vertex));
        PetscCall(EGlite_evaluate(vertex, NULL, result));
      } else {
        PetscCall(EG_objectBodyTopo(body, NODE, vertexID, &vertex));
        PetscCall(EG_evaluate(vertex, NULL, result));
      }
      for (d = 0; d < cdim; ++d) vcoords[d] = result[d];
    } else if (edgeID > 0) {
      /* Snap to EDGE */
      params[0] = t_point[v - vStart];
      if (islite) {
        PetscCall(EGlite_objectBodyTopo(body, EDGE, edgeID, &edge));
        PetscCall(EGlite_evaluate(edge, params, result));
      } else {
        PetscCall(EG_objectBodyTopo(body, EDGE, edgeID, &edge));
        PetscCall(EG_evaluate(edge, params, result));
      }
      for (d = 0; d < cdim; ++d) vcoords[d] = result[d];
    } else if (faceID > 0) {
      /* Snap to FACE */
      params[0] = u_point[v - vStart];
      params[1] = v_point[v - vStart];
      if (islite) {
        PetscCall(EGlite_objectBodyTopo(body, FACE, faceID, &face));
        PetscCall(EGlite_evaluate(face, params, result));
      } else {
        PetscCall(EG_objectBodyTopo(body, FACE, faceID, &face));
        PetscCall(EG_evaluate(face, params, result));
      }
      for (d = 0; d < cdim; ++d) vcoords[d] = result[d];
    }
  }
  PetscCall(VecRestoreArrayWrite(coordinates, &coords));
  /* Clear out global coordinates */
  PetscCall(VecDestroy(&dm->coordinates[0].x));
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMPlexInflateToGeomModel - Wrapper function allowing two methods for inflating refined meshes to the underlying geometric domain.

  Collective

  Input Parameters:
+ dm     - The DMPlex object with an attached PetscContainer storing a CAD Geometry object
- useTUV - PetscBool indicating if the user would like to inflate the DMPlex to the underlying geometry
           using (t) for nodes on EDGEs and (u, v) for nodes on FACEs or using the nodes (x, y, z) coordinates
           and shortest distance routine.
            If useTUV = PETSC_TRUE, use the (t) or (u, v) parameters to inflate the DMPlex to the CAD geometry.
            If useTUV = PETSC_FALSE, use the nodes (x, y, z) coordinates and the shortest disctance routine.

  Notes:
  DM with nodal coordinates modified so that they lie on the EDGEs and FACEs of the underlying geometry.

  (t) and (u, v) parameters for all DMPlex nodes on EDGEs and FACEs are stored in arrays within PetscContainers attached to the DM.
  The containers have names "Point - Edge t Parameter", "Point - Face u Parameter", and "Point - Face v Parameter".
  The arrays are organized by Point 0-based ID (i.e. [v-vstart] as defined in the DMPlex.

  Level: intermediate

.seealso: `DMPlexGetGeomModelTUV()`, `DMPlexInflateToGeomModelUseTUV()`, `DMPlexInflateToGeomModelUseXYZ()`
@*/
PetscErrorCode DMPlexInflateToGeomModel(DM dm, PetscBool useTUV) PeNS
{
  PetscFunctionBeginHot;
  if (useTUV) {
    PetscCall(DMPlexGetGeomModelTUV(dm));
    PetscCall(DMPlexInflateToGeomModelUseTUV(dm));
  } else {
    PetscCall(DMPlexInflateToGeomModelUseXYZ(dm));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#ifdef PETSC_HAVE_EGADS
/*@C
  DMPlexGetGeomModelBodies - Returns an array of `PetscGeom` BODY objects attached to the referenced geometric model entity as well as the number of BODYs.

  Collective

  Input Parameter:
. dm - The DMPlex object with an attached PetscContainer storing a CAD Geometry object

  Output Parameters:
+ bodies    - Array of PetscGeom BODY objects referenced by the geometric model.
- numBodies - Number of BODYs referenced by the geometric model. Also the size of **bodies array.

  Level: intermediate

.seealso:
@*/
PetscErrorCode DMPlexGetGeomModelBodies(DM dm, PetscGeom **bodies, PetscInt *numBodies) PeNS
{
  PetscFunctionBeginHot;
  PetscContainer modelObj;
  PetscBool      islite = PETSC_FALSE;
  ego            model, geom;
  int            oclass, mtype;
  int           *senses;

  /* Determine which type of EGADS model is attached to the DM */
  PetscCall(PetscObjectQuery((PetscObject)dm, "EGADS Model", (PetscObject *)&modelObj));
  if (!modelObj) {
    PetscCall(PetscObjectQuery((PetscObject)dm, "EGADSlite Model", (PetscObject *)&modelObj));
    islite = PETSC_TRUE;
  }

  // Get attached EGADS or EGADSlite model (pointer)
  PetscCall(PetscContainerGetPointer(modelObj, (void **)&model));

  if (islite) {
    PetscCall(EGlite_getTopology(model, &geom, &oclass, &mtype, NULL, numBodies, bodies, &senses));
  } else {
    PetscCall(EG_getTopology(model, &geom, &oclass, &mtype, NULL, numBodies, bodies, &senses));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMPlexGetGeomModelBodyShells - Returns an array of `PetscGeom` SHELL objects attached to the referenced BODY geometric entity as well as the number of SHELLs.

  Collective

  Input Parameters:
+ dm   - The DMPlex object with an attached PetscContainer storing a CAD Geometry object
- body - PetscGeom BODY object containing the SHELL objects of interest.

  Output Parameters:
+ shells    - Array of PetscGeom SHELL objects referenced by the PetscGeom BODY object
- numShells - Number of SHELLs referenced by the PetscGeom BODY object. Also the size of **shells array.

  Level: intermediate

.seealso:
@*/
PetscErrorCode DMPlexGetGeomModelBodyShells(DM dm, PetscGeom body, PetscGeom **shells, PetscInt *numShells) PeNS
{
  PetscFunctionBeginHot;
  #ifdef PETSC_HAVE_EGADS
  PetscContainer modelObj;
  PetscBool      islite = PETSC_FALSE;

  /* Determine which type of EGADS model is attached to the DM */
  PetscCall(PetscObjectQuery((PetscObject)dm, "EGADS Model", (PetscObject *)&modelObj));
  if (!modelObj) {
    PetscCall(PetscObjectQuery((PetscObject)dm, "EGADSlite Model", (PetscObject *)&modelObj));
    islite = PETSC_TRUE;
  }

  if (islite) {
    PetscCall(EGlite_getBodyTopos(body, NULL, SHELL, numShells, shells));
  } else {
    PetscCall(EG_getBodyTopos(body, NULL, SHELL, numShells, shells));
  }
  #endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMPlexGetGeomModelBodyFaces - Returns an array of `PetscGeom` FACE objects attached to the referenced BODY geometric entity as well as the number of FACEs.

  Collective

  Input Parameters:
+ dm   - The DMPlex object with an attached PetscContainer storing a CAD Geometry object
- body - PetscGeom BODY object containing the FACE objects of interest.

  Output Parameters:
+ faces    - Array of PetscGeom FACE objects referenced by the PetscGeom BODY object
- numFaces - Number of FACEs referenced by the PetscGeom BODY object. Also the size of **faces array.

  Level: intermediate

.seealso:
@*/
PetscErrorCode DMPlexGetGeomModelBodyFaces(DM dm, PetscGeom body, PetscGeom **faces, PetscInt *numFaces) PeNS
{
  PetscFunctionBeginHot;
  #ifdef PETSC_HAVE_EGADS
  PetscContainer modelObj;
  PetscBool      islite = PETSC_FALSE;

  /* Determine which type of EGADS model is attached to the DM */
  PetscCall(PetscObjectQuery((PetscObject)dm, "EGADS Model", (PetscObject *)&modelObj));
  if (!modelObj) {
    PetscCall(PetscObjectQuery((PetscObject)dm, "EGADSlite Model", (PetscObject *)&modelObj));
    islite = PETSC_TRUE;
  }

  if (islite) {
    PetscCall(EGlite_getBodyTopos(body, NULL, FACE, numFaces, faces));
  } else {
    PetscCall(EG_getBodyTopos(body, NULL, FACE, numFaces, faces));
  }
  #endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMPlexGetGeomModelBodyLoops - Returns an array of `PetscGeom` Loop objects attached to the referenced BODY geometric entity as well as the number of LOOPs.

  Collective

  Input Parameters:
+ dm   - The DMPlex object with an attached PetscContainer storing a CAD Geometry object
- body - PetscGeom BODY object containing the LOOP objects of interest.

  Output Parameters:
+ loops    - Array of PetscGeom FACE objects referenced by the PetscGeom SHELL object
- numLoops - Number of LOOPs referenced by the PetscGeom BODY object. Also the size of **loops array.

  Level: intermediate

.seealso:
@*/
PetscErrorCode DMPlexGetGeomModelBodyLoops(DM dm, PetscGeom body, PetscGeom **loops, PetscInt *numLoops) PeNS
{
  PetscFunctionBeginHot;
  #ifdef PETSC_HAVE_EGADS
  PetscContainer modelObj;
  PetscBool      islite = PETSC_FALSE;

  /* Determine which type of EGADS model is attached to the DM */
  PetscCall(PetscObjectQuery((PetscObject)dm, "EGADS Model", (PetscObject *)&modelObj));
  if (!modelObj) {
    PetscCall(PetscObjectQuery((PetscObject)dm, "EGADSlite Model", (PetscObject *)&modelObj));
    islite = PETSC_TRUE;
  }

  if (islite) {
    PetscCall(EGlite_getBodyTopos(body, NULL, LOOP, numLoops, loops));
  } else {
    PetscCall(EG_getBodyTopos(body, NULL, LOOP, numLoops, loops));
  }
  #endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMPlexGetGeomModelShellFaces - Returns an array of `PetscGeom` FACE objects attached to the referenced SHELL geometric entity as well as the number of FACEs.

  Collective

  Input Parameters:
+ dm    - The DMPlex object with an attached PetscContainer storing a CAD Geometry object
. body  - PetscGeom BODY object containing the FACE objects of interest.
- shell - PetscGeom SHELL object with FACEs of interest.

  Output Parameters:
+ faces    - Array of PetscGeom FACE objects referenced by the PetscGeom SHELL object
- numFaces - Number of FACEs referenced by the PetscGeom SHELL object. Also the size of **faces array.

  Level: intermediate

.seealso:
@*/
PetscErrorCode DMPlexGetGeomModelShellFaces(DM dm, PetscGeom body, PetscGeom shell, PetscGeom **faces, PetscInt *numFaces) PeNS
{
  PetscFunctionBeginHot;
  #ifdef PETSC_HAVE_EGADS
  PetscContainer modelObj;
  PetscBool      islite = PETSC_FALSE;

  /* Determine which type of EGADS model is attached to the DM */
  PetscCall(PetscObjectQuery((PetscObject)dm, "EGADS Model", (PetscObject *)&modelObj));
  if (!modelObj) {
    PetscCall(PetscObjectQuery((PetscObject)dm, "EGADSlite Model", (PetscObject *)&modelObj));
    islite = PETSC_TRUE;
  }

  if (islite) {
    PetscCall(EGlite_getBodyTopos(body, shell, FACE, numFaces, faces));
  } else {
    PetscCall(EG_getBodyTopos(body, shell, FACE, numFaces, faces));
  }
  #endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMPlexGetGeomModelFaceLoops - Returns an array of `PetscGeom` LOOP objects attached to the referenced FACE geometric entity as well as the number of LOOPs.

  Collective

  Input Parameters:
+ dm   - The DMPlex object with an attached PetscContainer storing a CAD Geometry object
. body - PetscGeom BODY object containing the LOOP objects of interest.
- face - PetscGeom FACE object with LOOPs of interest.

  Output Parameters:
+ loops    - Array of PetscGeom LOOP objects referenced by the PetscGeom FACE object
- numLoops - Number of LOOPs referenced by the PetscGeom FACE object. Also the size of **loops array.

  Level: intermediate

.seealso:
@*/
PetscErrorCode DMPlexGetGeomModelFaceLoops(DM dm, PetscGeom body, PetscGeom face, PetscGeom **loops, PetscInt *numLoops) PeNS
{
  PetscFunctionBeginHot;
  #ifdef PETSC_HAVE_EGADS
  PetscContainer modelObj;
  PetscBool      islite = PETSC_FALSE;

  /* Determine which type of EGADS model is attached to the DM */
  PetscCall(PetscObjectQuery((PetscObject)dm, "EGADS Model", (PetscObject *)&modelObj));
  if (!modelObj) {
    PetscCall(PetscObjectQuery((PetscObject)dm, "EGADSlite Model", (PetscObject *)&modelObj));
    islite = PETSC_TRUE;
  }

  if (islite) {
    PetscCall(EGlite_getBodyTopos(body, face, LOOP, numLoops, loops));
  } else {
    PetscCall(EG_getBodyTopos(body, face, LOOP, numLoops, loops));
  }
  #endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMPlexGetGeomModelFaceEdges - Returns an array of `PetscGeom` EDGE objects attached to the referenced FACE geometric entity as well as the number of EDGEs.

  Collective

  Input Parameters:
+ dm   - The DMPlex object with an attached PetscContainer storing a CAD Geometry object
. body - PetscGeom Body object containing the EDGE objects of interest.
- face - PetscGeom FACE object with EDGEs of interest.

  Output Parameters:
+ edges    - Array of PetscGeom EDGE objects referenced by the PetscGeom FACE object
- numEdges - Number of EDGEs referenced by the PetscGeom FACE object. Also the size of **edges array.

  Level: intermediate

.seealso:
@*/
PetscErrorCode DMPlexGetGeomModelFaceEdges(DM dm, PetscGeom body, PetscGeom face, PetscGeom **edges, PetscInt *numEdges) PeNS
{
  PetscFunctionBeginHot;
  #ifdef PETSC_HAVE_EGADS
  PetscContainer modelObj;
  PetscBool      islite = PETSC_FALSE;

  /* Determine which type of EGADS model is attached to the DM */
  PetscCall(PetscObjectQuery((PetscObject)dm, "EGADS Model", (PetscObject *)&modelObj));
  if (!modelObj) {
    PetscCall(PetscObjectQuery((PetscObject)dm, "EGADSlite Model", (PetscObject *)&modelObj));
    islite = PETSC_TRUE;
  }

  if (islite) {
    PetscCall(EGlite_getBodyTopos(body, face, EDGE, numEdges, edges));
  } else {
    PetscCall(EG_getBodyTopos(body, face, EDGE, numEdges, edges));
  }
  #endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMPlexGetGeomModelBodyEdges - Returns an array of `PetscGeom` EDGE objects attached to the referenced BODY geometric entity as well as the number of EDGEs.

  Collective

  Input Parameters:
+ dm   - The DMPlex object with an attached PetscContainer storing a CAD Geometry object
- body - PetscGeom body object of interest.

  Output Parameters:
+ edges    - Array of PetscGeom EDGE objects referenced by the PetscGeom BODY object
- numEdges - Number of EDGEs referenced by the PetscGeom BODY object. Also the size of **edges array.

  Level: intermediate

.seealso:
@*/
PetscErrorCode DMPlexGetGeomModelBodyEdges(DM dm, PetscGeom body, PetscGeom **edges, PetscInt *numEdges) PeNS
{
  PetscFunctionBeginHot;
  #ifdef PETSC_HAVE_EGADS
  PetscContainer modelObj;
  PetscBool      islite = PETSC_FALSE;

  /* Determine which type of EGADS model is attached to the DM */
  PetscCall(PetscObjectQuery((PetscObject)dm, "EGADS Model", (PetscObject *)&modelObj));
  if (!modelObj) {
    PetscCall(PetscObjectQuery((PetscObject)dm, "EGADSlite Model", (PetscObject *)&modelObj));
    islite = PETSC_TRUE;
  }

  if (islite) {
    PetscCall(EGlite_getBodyTopos(body, NULL, EDGE, numEdges, edges));
  } else {
    PetscCall(EG_getBodyTopos(body, NULL, EDGE, numEdges, edges));
  }
  #endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMPlexGetGeomModelBodyNodes - Returns an array of `PetscGeom` NODE objects attached to the referenced BODY geometric entity as well as the number of NODES.

  Collective

  Input Parameters:
+ dm   - The DMPlex object with an attached PetscContainer storing a CAD Geometry object
- body - PetscGeom body object of interest.

  Output Parameters:
+ nodes    - Array of PetscGeom NODE objects referenced by the PetscGeom BODY object
- numNodes - Number of NODEs referenced by the PetscGeom BODY object. Also the size of **nodes array.

  Level: intermediate

.seealso:
@*/
PetscErrorCode DMPlexGetGeomModelBodyNodes(DM dm, PetscGeom body, PetscGeom **nodes, PetscInt *numNodes) PeNS
{
  PetscFunctionBeginHot;
  #ifdef PETSC_HAVE_EGADS
  PetscContainer modelObj;
  PetscBool      islite = PETSC_FALSE;

  /* Determine which type of EGADS model is attached to the DM */
  PetscCall(PetscObjectQuery((PetscObject)dm, "EGADS Model", (PetscObject *)&modelObj));
  if (!modelObj) {
    PetscCall(PetscObjectQuery((PetscObject)dm, "EGADSlite Model", (PetscObject *)&modelObj));
    islite = PETSC_TRUE;
  }

  if (islite) {
    PetscCall(EGlite_getBodyTopos(body, NULL, NODE, numNodes, nodes));
  } else {
    PetscCall(EG_getBodyTopos(body, NULL, NODE, numNodes, nodes));
  }
  #endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMPlexGetGeomModelEdgeNodes - Returns an array of `PetscGeom` NODE objects attached to the referenced EDGE geometric entity as well as the number of NODES.

  Collective

  Input Parameters:
+ dm   - The DMPlex object with an attached PetscContainer storing a CAD Geometry object
. body - PetscGeom body object containing the EDGE object of interest.
- edge - PetscGeom EDGE object with NODEs of interest.

  Output Parameters:
+ nodes    - Array of PetscGeom NODE objects referenced by the PetscGeom EDGE object
- numNodes - Number of Nodes referenced by the PetscGeom EDGE object. Also the size of **nodes array.

  Level: intermediate

.seealso:
@*/
PetscErrorCode DMPlexGetGeomModelEdgeNodes(DM dm, PetscGeom body, PetscGeom edge, PetscGeom **nodes, PetscInt *numNodes) PeNS
{
  PetscFunctionBeginHot;
  #ifdef PETSC_HAVE_EGADS
  PetscContainer modelObj;
  PetscBool      islite = PETSC_FALSE;

  /* Determine which type of EGADS model is attached to the DM */
  PetscCall(PetscObjectQuery((PetscObject)dm, "EGADS Model", (PetscObject *)&modelObj));
  if (!modelObj) {
    PetscCall(PetscObjectQuery((PetscObject)dm, "EGADSlite Model", (PetscObject *)&modelObj));
    islite = PETSC_TRUE;
  }

  if (islite) {
    PetscCall(EGlite_getBodyTopos(body, edge, NODE, numNodes, nodes));
  } else {
    PetscCall(EG_getBodyTopos(body, edge, NODE, numNodes, nodes));
  }
  #endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMPlexGetGeomID - Returns ID number of the entity in the geometric (CAD) model

  Collective

  Input Parameters:
+ dm      - The DMPlex object with an attached PetscContainer storing a CAD Geometry object
. body    - PetscGeom body object containing the lower level entity the ID number is being requested.
- topoObj - PetscGeom SHELL, FACE, LOOP, EDGE, or NODE object for which ID number is being requested.

  Output Parameter:
. id - ID number of the entity

  Level: intermediate

.seealso:
@*/
PetscErrorCode DMPlexGetGeomID(DM dm, PetscGeom body, PetscGeom topoObj, PetscInt *id) PeNS
{
  PetscFunctionBeginHot;
  #ifdef PETSC_HAVE_EGADS
  PetscContainer modelObj;
  PetscBool      islite = PETSC_FALSE;
  int            topoID;

  /* Determine which type of EGADS model is attached to the DM */
  PetscCall(PetscObjectQuery((PetscObject)dm, "EGADS Model", (PetscObject *)&modelObj));
  if (!modelObj) {
    PetscCall(PetscObjectQuery((PetscObject)dm, "EGADSlite Model", (PetscObject *)&modelObj));
    islite = PETSC_TRUE;
  }

  // Get Topology Object's ID
  if (islite) {
    topoID = EGlite_indexBodyTopo(body, topoObj);
  } else {
    topoID = EG_indexBodyTopo(body, topoObj);
  }

  *id = topoID;
  #endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMPlexGetGeomObject - Returns Geometry Object using the objects ID in the geometric (CAD) model

  Collective

  Input Parameters:
+ dm       - The DMPlex object with an attached PetscContainer storing a CAD Geometry object
. body     - PetscGeom body object containing the lower level entity the referenced by the ID.
. geomType - Keyword SHELL, FACE, LOOP, EDGE, or NODE of the geometry type for which ID number is being requested.
- geomID   - ID number of the geometry entity being requested.

  Output Parameter:
. geomObj - Geometry Object referenced by the ID number requested.

  Level: intermediate

.seealso:
@*/
PetscErrorCode DMPlexGetGeomObject(DM dm, PetscGeom body, PetscInt geomType, PetscInt geomID, PetscGeom *geomObj) PeNS
{
  PetscFunctionBeginHot;
  #ifdef PETSC_HAVE_EGADS
  PetscContainer modelObj;
  PetscBool      islite = PETSC_FALSE;

  /* Determine which type of EGADS model is attached to the DM */
  PetscCall(PetscObjectQuery((PetscObject)dm, "EGADS Model", (PetscObject *)&modelObj));
  if (!modelObj) {
    PetscCall(PetscObjectQuery((PetscObject)dm, "EGADSlite Model", (PetscObject *)&modelObj));
    islite = PETSC_TRUE;
  }

  // Get Topology Object's ID
  if (islite) {
    PetscCall(EGlite_objectBodyTopo(body, geomType, geomID, geomObj));
  } else {
    PetscCall(EG_objectBodyTopo(body, geomType, geomID, geomObj));
  }
  #endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMPlexGetGeomFaceNumOfControlPoints - Returns the total number of Control Points (and associated Weights) defining a FACE of a Geometry

  Not collective

  Input Parameters:
+ dm   - The DMPlex object with an attached PetscContainer storing a CAD Geometry object
- face - PetscGeom FACE object

  Output Parameter:
. numCntrlPnts - Number of Control Points (and Weights) defining the FACE

  Level: intermediate

.seealso:
@*/
PetscErrorCode DMPlexGetGeomFaceNumOfControlPoints(DM dm, PetscGeom face, PetscInt *numCntrlPnts) PeNS
{
  PetscFunctionBeginHot;
  #ifdef PETSC_HAVE_EGADS
  PetscContainer modelObj;
  PetscBool      islite = PETSC_FALSE;
  PetscGeom      geom, gRef;
  PetscGeom     *lobjs;
  int            Nl, oclass, mtype, goclass, gmtype;
  int           *lsenses, *gpinfo;
  double        *gprv;

  /* Determine which type of EGADS model is attached to the DM */
  PetscCall(PetscObjectQuery((PetscObject)dm, "EGADS Model", (PetscObject *)&modelObj));
  if (!modelObj) {
    PetscCall(PetscObjectQuery((PetscObject)dm, "EGADSlite Model", (PetscObject *)&modelObj));
    islite = PETSC_TRUE;
  }

  // Get Total Number of Control Points on FACE
  if (islite) {
    PetscCall(EGlite_getTopology(face, &geom, &oclass, &mtype, NULL, &Nl, &lobjs, &lsenses));
    PetscCall(EGlite_getGeometry(geom, &goclass, &gmtype, &gRef, &gpinfo, &gprv));
  } else {
    PetscCall(EG_getTopology(face, &geom, &oclass, &mtype, NULL, &Nl, &lobjs, &lsenses));
    PetscCall(EG_getGeometry(geom, &goclass, &gmtype, &gRef, &gpinfo, &gprv));
  }

  *numCntrlPnts = gpinfo[2] * gpinfo[5];
  #endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMPlexGetGeomBodyMassProperties - Returns the Volume, Surface Area, Center of Gravity, and Inertia about the Body's Center of Gravity

  Not collective

  Input Parameters:
+ dm   - The DMPlex object with an attached PetscContainer storing a CAD Geometry object
- body - PetscGeom BODY object

  Output Parameters:
+ volume           - Volume of the CAD Body attached to the DM Plex
. surfArea         - Surface Area of the CAD Body attached to the DM Plex
. centerOfGravity  - Array with the Center of Gravity coordinates of the CAD Body attached to the DM Plex [x, y, z]
. COGszie          - Size of centerOfGravity[] Array
. inertiaMatrixCOG - Array containing the Inertia about the Body's Center of Gravity [Ixx, Ixy, Ixz, Iyx, Iyy, Iyz, Izx, Izy, Izz]
- IMCOGsize        - Size of inertiaMatrixCOG[] Array

  Level: intermediate

.seealso:
@*/
PetscErrorCode DMPlexGetGeomBodyMassProperties(DM dm, PetscGeom body, PetscScalar *volume, PetscScalar *surfArea, PetscScalar **centerOfGravity, PetscInt *COGsize, PetscScalar **inertiaMatrixCOG, PetscInt *IMCOGsize) PeNS
{
  PetscFunctionBeginHot;
  #ifdef PETSC_HAVE_EGADS
  PetscContainer modelObj;
  PetscBool      islite = PETSC_FALSE;
  PetscScalar    geomData[14];

  /* Determine which type of EGADS model is attached to the DM */
  PetscCall(PetscObjectQuery((PetscObject)dm, "EGADS Model", (PetscObject *)&modelObj));
  if (!modelObj) {
    PetscCall(PetscObjectQuery((PetscObject)dm, "EGADSlite Model", (PetscObject *)&modelObj));
    islite = PETSC_TRUE;
    PetscCheck(modelObj, PETSC_COMM_SELF, PETSC_ERR_SUP, "Cannot provide geometric mass properties for geometries defined by EGADSlite (.egadslite)! Please use another geometry file format STEP, IGES, EGADS or BRep");
  }

  if (islite) {
    PetscCall(PetscPrintf(PETSC_COMM_SELF, " WARNING!! This functionality is not supported for EGADSlite files. \n"));
    PetscCall(PetscPrintf(PETSC_COMM_SELF, " All returned values are equal to 0 \n"));
  } else {
    PetscCall(EG_getMassProperties(body, geomData));
  }

  PetscCall(PetscMalloc2(3, centerOfGravity, 9, inertiaMatrixCOG));

  if (!islite) {
    *volume   = geomData[0];
    *surfArea = geomData[1];
    for (int ii = 2; ii < 5; ++ii) { (*centerOfGravity)[ii - 2] = geomData[ii]; }
    *COGsize = 3;
    for (int ii = 5; ii < 14; ++ii) { (*inertiaMatrixCOG)[ii - 5] = geomData[ii]; }
    *IMCOGsize = 9;
  } else {
    *volume   = 0.;
    *surfArea = 0.;
    for (int ii = 2; ii < 5; ++ii) { (*centerOfGravity)[ii - 2] = 0.; }
    *COGsize = 0;
    for (int ii = 5; ii < 14; ++ii) { (*inertiaMatrixCOG)[ii - 5] = 0.; }
    *IMCOGsize = 0;
  }
  #endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMPlexRestoreGeomBodyMassProperties(DM dm, PetscGeom body, PetscScalar *volume, PetscScalar *surfArea, PetscScalar **centerOfGravity, PetscInt *COGsize, PetscScalar **inertiaMatrixCOG, PetscInt *IMCOGsize) PeNS
{
  PetscFunctionBegin;
  PetscCall(PetscFree2(*centerOfGravity, *inertiaMatrixCOG));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMPlexFreeGeomObject - Frees PetscGeom Objects

  Not collective

  Input Parameters:
+ dm      - The DMPlex object with an attached PetscContainer storing a CAD Geometry object
- geomObj - PetscGeom object

  Level: intermediate

.seealso:
@*/
PetscErrorCode DMPlexFreeGeomObject(DM dm, PetscGeom *geomObj) PeNS
{
  PetscFunctionBeginHot;
  #ifdef PETSC_HAVE_EGADS
  PetscContainer modelObj;
  PetscBool      islite = PETSC_FALSE;

  /* Determine which type of EGADS model is attached to the DM */
  PetscCall(PetscObjectQuery((PetscObject)dm, "EGADS Model", (PetscObject *)&modelObj));
  if (!modelObj) {
    PetscCall(PetscObjectQuery((PetscObject)dm, "EGADSlite Model", (PetscObject *)&modelObj));
    islite = PETSC_TRUE;
  }

  if (islite) {
    EGlite_free(geomObj);
  } else {
    EG_free(geomObj);
  }
  #endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMPlexGetGeomCntrlPntAndWeightData - Gets Control Point and Associated Weight Data for the Geometry attached to the DMPlex

  Not collective

  Input Parameter:
. dm - The DMPlex object with an attached PetscContainer storing a CAD Geometry object

  Output Parameters:
+ cpHashTable       - Hash Table containing the relationship between FACE ID and Control Point IDs.
. cpCoordDataLength - Length of cpCoordData Array.
. cpCoordData       - Array holding the Geometry Control Point Coordinate Data.
. maxNumEquiv       - Maximum Number of Equivalent Control Points (Control Points with the same coordinates but different IDs).
. cpEquiv           - Matrix with a size(Number of Control Points, Number or Control Points) which stores a value of 1.0 in locations where Control Points with different IDS (row or column) have the same coordinates
. wHashTable        - Hash Table containing the relationship between FACE ID and Control Point Weight.
. wDataLength       - Length of wData Array.
- wData             - Array holding the Weight for an associated Geometry Control Point.

  Note:
  Must Call DMPLexGeomDataAndGrads() before calling this function.

  Level: intermediate

.seealso:
@*/
PetscErrorCode DMPlexGetGeomCntrlPntAndWeightData(DM dm, PetscHMapI *cpHashTable, PetscInt *cpCoordDataLength, PetscScalar **cpCoordData, PetscInt *maxNumEquiv, Mat *cpEquiv, PetscHMapI *wHashTable, PetscInt *wDataLength, PetscScalar **wData) PeNS
{
  PetscContainer modelObj, cpHashTableObj, wHashTableObj, cpCoordDataLengthObj, wDataLengthObj, maxNumRelateObj;
  Vec            cntrlPtCoordsVec, cntrlPtWeightsVec;
  PetscInt      *cpCoordDataLengthPtr, *wDataLengthPtr, *maxNumEquivPtr;
  PetscHMapI     cpHashTableTemp, wHashTableTemp;

  PetscFunctionBeginHot;
  /* Determine which type of EGADS model is attached to the DM */
  PetscCall(PetscObjectQuery((PetscObject)dm, "EGADS Model", (PetscObject *)&modelObj));
  if (!modelObj) { PetscCall(PetscObjectQuery((PetscObject)dm, "EGADSlite Model", (PetscObject *)&modelObj)); }

  if (!modelObj) { PetscFunctionReturn(PETSC_SUCCESS); }

  // Look to see if DM has Container for Geometry Control Point Data
  PetscCall(PetscObjectQuery((PetscObject)dm, "Control Point Hash Table", (PetscObject *)&cpHashTableObj));
  PetscCall(PetscObjectQuery((PetscObject)dm, "Control Point Coordinates", (PetscObject *)&cntrlPtCoordsVec));
  PetscCall(PetscObjectQuery((PetscObject)dm, "Control Point Coordinate Data Length", (PetscObject *)&cpCoordDataLengthObj));
  PetscCall(PetscObjectQuery((PetscObject)dm, "Control Point Weights Hash Table", (PetscObject *)&wHashTableObj));
  PetscCall(PetscObjectQuery((PetscObject)dm, "Control Point Weight Data", (PetscObject *)&cntrlPtWeightsVec));
  PetscCall(PetscObjectQuery((PetscObject)dm, "Control Point Weight Data Length", (PetscObject *)&wDataLengthObj));
  PetscCall(PetscObjectQuery((PetscObject)dm, "Control Point Equivalency Matrix", (PetscObject *)cpEquiv));
  PetscCall(PetscObjectQuery((PetscObject)dm, "Maximum Number Control Point Equivalency", (PetscObject *)&maxNumRelateObj));

  // Get attached EGADS model Control Point and Weights Hash Tables and Data Arrays (pointer)
  PetscCall(PetscContainerGetPointer(cpHashTableObj, (void **)&cpHashTableTemp));
  PetscCall(PetscContainerGetPointer(cpCoordDataLengthObj, (void **)&cpCoordDataLengthPtr));
  PetscCall(PetscContainerGetPointer(wHashTableObj, (void **)&wHashTableTemp));
  PetscCall(PetscContainerGetPointer(wDataLengthObj, (void **)&wDataLengthPtr));
  PetscCall(PetscContainerGetPointer(maxNumRelateObj, (void **)&maxNumEquivPtr));

  *cpCoordDataLength = *cpCoordDataLengthPtr;
  *wDataLength       = *wDataLengthPtr;
  *maxNumEquiv       = *maxNumEquivPtr;
  *cpHashTable       = cpHashTableTemp;
  *wHashTable        = wHashTableTemp;
  PetscCall(VecGetArrayWrite(cntrlPtCoordsVec, cpCoordData));
  PetscCall(VecGetArrayWrite(cntrlPtWeightsVec, wData));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMPlexRestoreGeomCntrlPntAndWeightData(DM dm, PetscHMapI *cpHashTable, PetscInt *cpCoordDataLength, PetscScalar **cpCoordData, PetscInt *maxNumEquiv, Mat *cpEquiv, PetscHMapI *wHashTable, PetscInt *wDataLength, PetscScalar **wData)
{
  Vec cntrlPtCoordsVec, cntrlPtWeightsVec;

  PetscFunctionBeginHot;
  PetscCall(PetscObjectQuery((PetscObject)dm, "Control Point Coordinates", (PetscObject *)&cntrlPtCoordsVec));
  PetscCall(VecRestoreArrayWrite(cntrlPtCoordsVec, cpCoordData));
  PetscCall(PetscObjectQuery((PetscObject)dm, "Control Point Weight Data", (PetscObject *)&cntrlPtWeightsVec));
  PetscCall(VecRestoreArrayWrite(cntrlPtWeightsVec, wData));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMPlexGetGeomGradData - Gets Point, Surface and Volume Gradients with respect to changes in Control Points and their associated Weights for the Geometry attached to the DMPlex .

  Not collective

  Input Parameter:
. dm - The DMPlex object with an attached PetscContainer storing a CAD Geometry object

  Output Parameters:
+ cpSurfGradHashTable - Hash Table Relating the Control Point ID to the the Row in the cpSurfGrad Matrix
. cpSurfGrad          - Matrix containing the Surface Gradient with respect to the Control Point Data. Data is ranged where the Row corresponds to Control Point ID and the Columns are associated with the Geometric FACE.
. cpArraySize         - The size of arrays gradSACP and gradVolCP and is equal to 3 * total number of Control Points in the Geometry
. gradSACP            - Array containing the Surface Area Gradient with respect to Control Point Data. Data is arranged by Control Point ID * 3 where 3 is for the coordinate dimension.
. gradVolCP           - Array containing the Volume Gradient with respect to Control Point Data. Data is arranged by Control Point ID * 3 where 3 is for the coordinate dimension.
. wArraySize          - The size of arrayws gradSAW and gradVolW and is equal to the total number of Control Points in the Geometry.
. gradSAW             - Array containing the Surface Area Gradient with respect to Control Point Weight. Data is arranged by Control Point ID.
- gradVolW            - Array containing the Volume Gradient with respect to Control Point Weight. Data is arranged by Control Point ID.

  Notes:
  Must Call DMPLexGeomDataAndGrads() before calling this function.

  gradVolCP and gradVolW are only available when DMPlexGeomDataAndGrads() is called with fullGeomGrad = PETSC_TRUE.

  Level: intermediate

.seealso: DMPlexGeomDataAndGrads
@*/
PetscErrorCode DMPlexGetGeomGradData(DM dm, PetscHMapI *cpSurfGradHashTable, Mat *cpSurfGrad, PetscInt *cpArraySize, PetscScalar **gradSACP, PetscScalar **gradVolCP, PetscInt *wArraySize, PetscScalar **gradSAW, PetscScalar **gradVolW)
{
  PetscContainer modelObj, cpSurfGradHashTableObj, cpArraySizeObj, wArraySizeObj;
  Vec            gradSACPVec, gradVolCPVec, gradSAWVec, gradVolWVec;
  PetscInt      *cpArraySizePtr, *wArraySizePtr;
  PetscHMapI     cpSurfGradHashTableTemp;

  PetscFunctionBeginHot;
  /* Determine which type of EGADS model is attached to the DM */
  PetscCall(PetscObjectQuery((PetscObject)dm, "EGADS Model", (PetscObject *)&modelObj));
  if (!modelObj) { PetscCall(PetscObjectQuery((PetscObject)dm, "EGADSlite Model", (PetscObject *)&modelObj)); }

  if (!modelObj) { PetscFunctionReturn(PETSC_SUCCESS); }

  // Look to see if DM has Container for Geometry Control Point Data
  PetscCall(PetscObjectQuery((PetscObject)dm, "Surface Gradient Hash Table", (PetscObject *)&cpSurfGradHashTableObj));
  PetscCall(PetscObjectQuery((PetscObject)dm, "Surface Gradient Matrix", (PetscObject *)cpSurfGrad));
  PetscCall(PetscObjectQuery((PetscObject)dm, "Control Point Coordinate Data Length", (PetscObject *)&cpArraySizeObj));
  PetscCall(PetscObjectQuery((PetscObject)dm, "Surface Area Control Point Gradient", (PetscObject *)&gradSACPVec));
  PetscCall(PetscObjectQuery((PetscObject)dm, "Volume Control Point Gradient", (PetscObject *)&gradVolCPVec));
  PetscCall(PetscObjectQuery((PetscObject)dm, "Control Point Weight Data Length", (PetscObject *)&wArraySizeObj));
  PetscCall(PetscObjectQuery((PetscObject)dm, "Surface Area Weights Gradient", (PetscObject *)&gradSAWVec));
  PetscCall(PetscObjectQuery((PetscObject)dm, "Volume Weights Gradient", (PetscObject *)&gradVolWVec));

  // Get attached EGADS model Control Point and Weights Hash Tables and Data Arrays (pointer)
  if (cpSurfGradHashTableObj) {
    PetscCall(PetscContainerGetPointer(cpSurfGradHashTableObj, (void **)&cpSurfGradHashTableTemp));
    *cpSurfGradHashTable = cpSurfGradHashTableTemp;
  }

  if (cpArraySizeObj) {
    PetscCall(PetscContainerGetPointer(cpArraySizeObj, (void **)&cpArraySizePtr));
    *cpArraySize = *cpArraySizePtr;
  }

  if (gradSACPVec) PetscCall(VecGetArrayWrite(gradSACPVec, gradSACP));
  if (gradVolCPVec) PetscCall(VecGetArrayWrite(gradVolCPVec, gradVolCP));
  if (gradSAWVec) PetscCall(VecGetArrayWrite(gradSAWVec, gradSAW));
  if (gradVolWVec) PetscCall(VecGetArrayWrite(gradVolWVec, gradVolW));

  if (wArraySizeObj) {
    PetscCall(PetscContainerGetPointer(wArraySizeObj, (void **)&wArraySizePtr));
    *wArraySize = *wArraySizePtr;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMPlexRestoreGeomGradData(DM dm, PetscHMapI *cpSurfGradHashTable, Mat *cpSurfGrad, PetscInt *cpArraySize, PetscScalar **gradSACP, PetscScalar **gradVolCP, PetscInt *wArraySize, PetscScalar **gradSAW, PetscScalar **gradVolW)
{
  Vec gradSACPVec, gradVolCPVec, gradSAWVec, gradVolWVec;

  PetscFunctionBegin;
  PetscCall(PetscObjectQuery((PetscObject)dm, "Surface Area Control Point Gradient", (PetscObject *)&gradSACPVec));
  PetscCall(PetscObjectQuery((PetscObject)dm, "Volume Control Point Gradient", (PetscObject *)&gradVolCPVec));
  PetscCall(PetscObjectQuery((PetscObject)dm, "Surface Area Weights Gradient", (PetscObject *)&gradSAWVec));
  PetscCall(PetscObjectQuery((PetscObject)dm, "Volume Weights Gradient", (PetscObject *)&gradVolWVec));

  if (gradSACPVec) PetscCall(VecRestoreArrayWrite(gradSACPVec, gradSACP));
  if (gradVolCPVec) PetscCall(VecRestoreArrayWrite(gradVolCPVec, gradVolCP));
  if (gradSAWVec) PetscCall(VecRestoreArrayWrite(gradSAWVec, gradSAW));
  if (gradVolWVec) PetscCall(VecRestoreArrayWrite(gradVolWVec, gradVolW));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMPlexGetGeomCntrlPntMaps - Gets arrays which maps Control Point IDs to their associated Geometry FACE, EDGE, and VERTEX.

  Not collective

  Input Parameter:
. dm - The DMPlex object with an attached PetscContainer storing a CAD Geometry object

  Output Parameters:
+ numCntrlPnts            - Number of Control Points defining the Geometry attached to the DMPlex
. cntrlPntFaceMap         - Array containing the FACE ID for the Control Point. Array index corresponds to Control Point ID.
. cntrlPntWeightFaceMap   - Array containing the FACE ID for the Control Point Weight. Array index corresponds to Control Point ID.
. cntrlPntEdgeMap         - Array containing the EDGE ID for the Control Point. Array index corresponds to Control Point ID.
. cntrlPntWeightEdgeMap   - Array containing the EDGE ID for the Control Point Weight. Array index corresponds to Control Point ID.
. cntrlPntVertexMap       - Array containing the VERTEX ID for the Control Point. Array index corresponds to Control Point ID.
- cntrlPntWeightVertexMap - Array containing the VERTEX ID for the Control Point Weight. Array index corresponds to Control Point ID.

  Note:
  Arrays are initialized to -1. Array elements with a -1 value indicates that the Control Point or Control Point Weight not associated with the referenced Geometric entity in the array name.

  Level: intermediate

.seealso: DMPlexGeomDataAndGrads
@*/
PetscErrorCode DMPlexGetGeomCntrlPntMaps(DM dm, PetscInt *numCntrlPnts, PetscInt **cntrlPntFaceMap, PetscInt **cntrlPntWeightFaceMap, PetscInt **cntrlPntEdgeMap, PetscInt **cntrlPntWeightEdgeMap, PetscInt **cntrlPntVertexMap, PetscInt **cntrlPntWeightVertexMap)
{
  PetscFunctionBeginHot;
  #ifdef PETSC_HAVE_EGADS
  PetscContainer modelObj, numCntrlPntsObj, cntrlPntFaceMapObj, cntrlPntWeightFaceMapObj, cntrlPntEdgeMapObj, cntrlPntWeightEdgeMapObj, cntrlPntVertexMapObj, cntrlPntWeightVertexMapObj;
  PetscInt      *numCntrlPntsPtr, *cntrlPntFaceMapPtr, *cntrlPntWeightFaceMapPtr, *cntrlPntEdgeMapPtr, *cntrlPntWeightEdgeMapPtr, *cntrlPntVertexMapPtr, *cntrlPntWeightVertexMapPtr;

  /* Determine which type of EGADS model is attached to the DM */
  PetscCall(PetscObjectQuery((PetscObject)dm, "EGADS Model", (PetscObject *)&modelObj));
  if (!modelObj) { PetscCall(PetscObjectQuery((PetscObject)dm, "EGADSlite Model", (PetscObject *)&modelObj)); }

  if (!modelObj) { PetscFunctionReturn(PETSC_SUCCESS); }

  // Look to see if DM has Container for Geometry Control Point Data
  PetscCall(PetscObjectQuery((PetscObject)dm, "Control Point Weight Data Length", (PetscObject *)&numCntrlPntsObj));
  PetscCall(PetscObjectQuery((PetscObject)dm, "Control Point - Face Map", (PetscObject *)&cntrlPntFaceMapObj));
  PetscCall(PetscObjectQuery((PetscObject)dm, "Control Point Weight - Face Map", (PetscObject *)&cntrlPntWeightFaceMapObj));
  PetscCall(PetscObjectQuery((PetscObject)dm, "Control Point - Edge Map", (PetscObject *)&cntrlPntEdgeMapObj));
  PetscCall(PetscObjectQuery((PetscObject)dm, "Control Point Weight - Edge Map", (PetscObject *)&cntrlPntWeightEdgeMapObj));
  PetscCall(PetscObjectQuery((PetscObject)dm, "Control Point - Vertex Map", (PetscObject *)&cntrlPntVertexMapObj));
  PetscCall(PetscObjectQuery((PetscObject)dm, "Control Point Weight - Vertex Map", (PetscObject *)&cntrlPntWeightVertexMapObj));

  // Get attached EGADS model Control Point and Weights Hash Tables and Data Arrays (pointer)
  if (numCntrlPntsObj) {
    PetscCall(PetscContainerGetPointer(numCntrlPntsObj, (void **)&numCntrlPntsPtr));
    *numCntrlPnts = *numCntrlPntsPtr;
  }

  if (cntrlPntFaceMapObj) {
    PetscCall(PetscContainerGetPointer(cntrlPntFaceMapObj, (void **)&cntrlPntFaceMapPtr));
    *cntrlPntFaceMap = cntrlPntFaceMapPtr;
  }

  if (cntrlPntWeightFaceMapObj) {
    PetscCall(PetscContainerGetPointer(cntrlPntWeightFaceMapObj, (void **)&cntrlPntWeightFaceMapPtr));
    *cntrlPntWeightFaceMap = cntrlPntWeightFaceMapPtr;
  }

  if (cntrlPntEdgeMapObj) {
    PetscCall(PetscContainerGetPointer(cntrlPntEdgeMapObj, (void **)&cntrlPntEdgeMapPtr));
    *cntrlPntEdgeMap = cntrlPntEdgeMapPtr;
  }

  if (cntrlPntWeightEdgeMapObj) {
    PetscCall(PetscContainerGetPointer(cntrlPntWeightEdgeMapObj, (void **)&cntrlPntWeightEdgeMapPtr));
    *cntrlPntWeightEdgeMap = cntrlPntWeightEdgeMapPtr;
  }

  if (cntrlPntVertexMapObj) {
    PetscCall(PetscContainerGetPointer(cntrlPntVertexMapObj, (void **)&cntrlPntVertexMapPtr));
    *cntrlPntVertexMap = cntrlPntVertexMapPtr;
  }

  if (cntrlPntWeightVertexMapObj) {
    PetscCall(PetscContainerGetPointer(cntrlPntWeightVertexMapObj, (void **)&cntrlPntWeightVertexMapPtr));
    *cntrlPntWeightVertexMap = cntrlPntWeightVertexMapPtr;
  }

  #endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

#endif
