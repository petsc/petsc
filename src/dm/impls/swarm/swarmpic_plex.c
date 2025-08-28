#include <petscdm.h>
#include <petscdmplex.h>
#include <petscdmswarm.h>
#include "../src/dm/impls/swarm/data_bucket.h"

PetscBool  SwarmProjcite       = PETSC_FALSE;
const char SwarmProjCitation[] = "@article{PusztayKnepleyAdams2022,\n"
                                 "title   = {Conservative Projection Between FEM and Particle Bases},\n"
                                 "author  = {Joseph V. Pusztay and Matthew G. Knepley and Mark F. Adams},\n"
                                 "journal = {SIAM Journal on Scientific Computing},\n"
                                 "volume  = {44},\n"
                                 "number  = {4},\n"
                                 "pages   = {C310--C319},\n"
                                 "doi     = {10.1137/21M145407},\n"
                                 "year    = {2022}\n}\n";

PetscErrorCode private_DMSwarmSetPointCoordinatesCellwise_PLEX(DM, DM, PetscInt, PetscReal *xi);

static PetscErrorCode private_PetscFECreateDefault_scalar_pk1(DM dm, PetscInt dim, PetscBool isSimplex, PetscInt qorder, PetscFE *fem)
{
  const PetscInt  Nc = 1;
  PetscQuadrature q, fq;
  DM              K;
  PetscSpace      P;
  PetscDualSpace  Q;
  PetscInt        order, quadPointsPerEdge;
  PetscBool       tensor = isSimplex ? PETSC_FALSE : PETSC_TRUE;

  PetscFunctionBegin;
  /* Create space */
  PetscCall(PetscSpaceCreate(PetscObjectComm((PetscObject)dm), &P));
  /* PetscCall(PetscObjectSetOptionsPrefix((PetscObject) P, prefix)); */
  PetscCall(PetscSpacePolynomialSetTensor(P, tensor));
  /* PetscCall(PetscSpaceSetFromOptions(P)); */
  PetscCall(PetscSpaceSetType(P, PETSCSPACEPOLYNOMIAL));
  PetscCall(PetscSpaceSetDegree(P, 1, PETSC_DETERMINE));
  PetscCall(PetscSpaceSetNumComponents(P, Nc));
  PetscCall(PetscSpaceSetNumVariables(P, dim));
  PetscCall(PetscSpaceSetUp(P));
  PetscCall(PetscSpaceGetDegree(P, &order, NULL));
  PetscCall(PetscSpacePolynomialGetTensor(P, &tensor));
  /* Create dual space */
  PetscCall(PetscDualSpaceCreate(PetscObjectComm((PetscObject)dm), &Q));
  PetscCall(PetscDualSpaceSetType(Q, PETSCDUALSPACELAGRANGE));
  /* PetscCall(PetscObjectSetOptionsPrefix((PetscObject) Q, prefix)); */
  PetscCall(DMPlexCreateReferenceCell(PETSC_COMM_SELF, DMPolytopeTypeSimpleShape(dim, isSimplex), &K));
  PetscCall(PetscDualSpaceSetDM(Q, K));
  PetscCall(DMDestroy(&K));
  PetscCall(PetscDualSpaceSetNumComponents(Q, Nc));
  PetscCall(PetscDualSpaceSetOrder(Q, order));
  PetscCall(PetscDualSpaceLagrangeSetTensor(Q, tensor));
  /* PetscCall(PetscDualSpaceSetFromOptions(Q)); */
  PetscCall(PetscDualSpaceSetType(Q, PETSCDUALSPACELAGRANGE));
  PetscCall(PetscDualSpaceSetUp(Q));
  /* Create element */
  PetscCall(PetscFECreate(PetscObjectComm((PetscObject)dm), fem));
  /* PetscCall(PetscObjectSetOptionsPrefix((PetscObject) *fem, prefix)); */
  /* PetscCall(PetscFESetFromOptions(*fem)); */
  PetscCall(PetscFESetType(*fem, PETSCFEBASIC));
  PetscCall(PetscFESetBasisSpace(*fem, P));
  PetscCall(PetscFESetDualSpace(*fem, Q));
  PetscCall(PetscFESetNumComponents(*fem, Nc));
  PetscCall(PetscFESetUp(*fem));
  PetscCall(PetscSpaceDestroy(&P));
  PetscCall(PetscDualSpaceDestroy(&Q));
  /* Create quadrature (with specified order if given) */
  qorder            = qorder >= 0 ? qorder : order;
  quadPointsPerEdge = PetscMax(qorder + 1, 1);
  if (isSimplex) {
    PetscCall(PetscDTStroudConicalQuadrature(dim, 1, quadPointsPerEdge, -1.0, 1.0, &q));
    PetscCall(PetscDTStroudConicalQuadrature(dim - 1, 1, quadPointsPerEdge, -1.0, 1.0, &fq));
  } else {
    PetscCall(PetscDTGaussTensorQuadrature(dim, 1, quadPointsPerEdge, -1.0, 1.0, &q));
    PetscCall(PetscDTGaussTensorQuadrature(dim - 1, 1, quadPointsPerEdge, -1.0, 1.0, &fq));
  }
  PetscCall(PetscFESetQuadrature(*fem, q));
  PetscCall(PetscFESetFaceQuadrature(*fem, fq));
  PetscCall(PetscQuadratureDestroy(&q));
  PetscCall(PetscQuadratureDestroy(&fq));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode private_DMSwarmInsertPointsUsingCellDM_PLEX_SubDivide(DM dm, DM dmc, PetscInt nsub)
{
  PetscInt         dim, nfaces, nbasis;
  PetscInt         q, npoints_q, e, nel, pcnt, ps, pe, d, k, r, Nfc;
  DMSwarmCellDM    celldm;
  PetscTabulation  T;
  Vec              coorlocal;
  PetscSection     coordSection;
  PetscScalar     *elcoor = NULL;
  PetscReal       *swarm_coor;
  PetscInt        *swarm_cellid;
  const PetscReal *xiq;
  PetscQuadrature  quadrature;
  PetscFE          fe, feRef;
  PetscBool        is_simplex;
  const char     **coordFields, *cellid;

  PetscFunctionBegin;
  PetscCall(DMGetDimension(dmc, &dim));
  is_simplex = PETSC_FALSE;
  PetscCall(DMPlexGetHeightStratum(dmc, 0, &ps, &pe));
  PetscCall(DMPlexGetConeSize(dmc, ps, &nfaces));
  if (nfaces == (dim + 1)) is_simplex = PETSC_TRUE;

  PetscCall(private_PetscFECreateDefault_scalar_pk1(dmc, dim, is_simplex, 0, &fe));

  for (r = 0; r < nsub; r++) {
    PetscCall(PetscFERefine(fe, &feRef));
    PetscCall(PetscFECopyQuadrature(feRef, fe));
    PetscCall(PetscFEDestroy(&feRef));
  }

  PetscCall(PetscFEGetQuadrature(fe, &quadrature));
  PetscCall(PetscQuadratureGetData(quadrature, NULL, NULL, &npoints_q, &xiq, NULL));
  PetscCall(PetscFEGetDimension(fe, &nbasis));
  PetscCall(PetscFEGetCellTabulation(fe, 1, &T));

  /* 0->cell, 1->edge, 2->vert */
  PetscCall(DMPlexGetHeightStratum(dmc, 0, &ps, &pe));
  nel = pe - ps;

  PetscCall(DMSwarmGetCellDMActive(dmc, &celldm));
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm, &Nfc, &coordFields));
  PetscCheck(Nfc == 1, PetscObjectComm((PetscObject)dmc), PETSC_ERR_SUP, "We only support a single coordinate field right now, not %" PetscInt_FMT, Nfc);
  PetscCall(DMSwarmCellDMGetCellID(celldm, &cellid));

  PetscCall(DMSwarmSetLocalSizes(dm, npoints_q * nel, -1));
  PetscCall(DMSwarmGetField(dm, coordFields[0], NULL, NULL, (void **)&swarm_coor));
  PetscCall(DMSwarmGetField(dm, cellid, NULL, NULL, (void **)&swarm_cellid));

  PetscCall(DMGetCoordinatesLocal(dmc, &coorlocal));
  PetscCall(DMGetCoordinateSection(dmc, &coordSection));

  pcnt = 0;
  for (e = 0; e < nel; e++) {
    PetscCall(DMPlexVecGetClosure(dmc, coordSection, coorlocal, ps + e, NULL, &elcoor));

    for (q = 0; q < npoints_q; q++) {
      for (d = 0; d < dim; d++) {
        swarm_coor[dim * pcnt + d] = 0.0;
        for (k = 0; k < nbasis; k++) swarm_coor[dim * pcnt + d] += T->T[0][q * nbasis + k] * PetscRealPart(elcoor[dim * k + d]);
      }
      swarm_cellid[pcnt] = e;
      pcnt++;
    }
    PetscCall(DMPlexVecRestoreClosure(dmc, coordSection, coorlocal, ps + e, NULL, &elcoor));
  }
  PetscCall(DMSwarmRestoreField(dm, cellid, NULL, NULL, (void **)&swarm_cellid));
  PetscCall(DMSwarmRestoreField(dm, coordFields[0], NULL, NULL, (void **)&swarm_coor));

  PetscCall(PetscFEDestroy(&fe));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode private_DMSwarmInsertPointsUsingCellDM_PLEX2D_Regular(DM dm, DM dmc, PetscInt npoints)
{
  PetscInt      dim;
  PetscInt      ii, jj, q, npoints_q, e, nel, npe, pcnt, ps, pe, d, k, nfaces, Nfc;
  PetscReal    *xi, ds, ds2;
  PetscReal   **basis;
  DMSwarmCellDM celldm;
  Vec           coorlocal;
  PetscSection  coordSection;
  PetscScalar  *elcoor = NULL;
  PetscReal    *swarm_coor;
  PetscInt     *swarm_cellid;
  PetscBool     is_simplex;
  const char  **coordFields, *cellid;

  PetscFunctionBegin;
  PetscCall(DMGetDimension(dmc, &dim));
  PetscCheck(dim == 2, PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Only 2D is supported");
  is_simplex = PETSC_FALSE;
  PetscCall(DMPlexGetHeightStratum(dmc, 0, &ps, &pe));
  PetscCall(DMPlexGetConeSize(dmc, ps, &nfaces));
  if (nfaces == (dim + 1)) is_simplex = PETSC_TRUE;
  PetscCheck(is_simplex, PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Only the simplex is supported");

  PetscCall(PetscMalloc1(dim * npoints * npoints, &xi));
  pcnt = 0;
  ds   = 1.0 / (PetscReal)(npoints - 1);
  ds2  = 1.0 / (PetscReal)npoints;
  for (jj = 0; jj < npoints; jj++) {
    for (ii = 0; ii < npoints - jj; ii++) {
      xi[dim * pcnt + 0] = ii * ds;
      xi[dim * pcnt + 1] = jj * ds;

      xi[dim * pcnt + 0] *= (1.0 - 1.2 * ds2);
      xi[dim * pcnt + 1] *= (1.0 - 1.2 * ds2);

      xi[dim * pcnt + 0] += 0.35 * ds2;
      xi[dim * pcnt + 1] += 0.35 * ds2;
      pcnt++;
    }
  }
  npoints_q = pcnt;

  npe = 3; /* nodes per element (triangle) */
  PetscCall(PetscMalloc1(npoints_q, &basis));
  for (q = 0; q < npoints_q; q++) {
    PetscCall(PetscMalloc1(npe, &basis[q]));

    basis[q][0] = 1.0 - xi[dim * q + 0] - xi[dim * q + 1];
    basis[q][1] = xi[dim * q + 0];
    basis[q][2] = xi[dim * q + 1];
  }

  /* 0->cell, 1->edge, 2->vert */
  PetscCall(DMPlexGetHeightStratum(dmc, 0, &ps, &pe));
  nel = pe - ps;

  PetscCall(DMSwarmGetCellDMActive(dmc, &celldm));
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm, &Nfc, &coordFields));
  PetscCheck(Nfc == 1, PetscObjectComm((PetscObject)dmc), PETSC_ERR_SUP, "We only support a single coordinate field right now, not %" PetscInt_FMT, Nfc);
  PetscCall(DMSwarmCellDMGetCellID(celldm, &cellid));

  PetscCall(DMSwarmSetLocalSizes(dm, npoints_q * nel, -1));
  PetscCall(DMSwarmGetField(dm, coordFields[0], NULL, NULL, (void **)&swarm_coor));
  PetscCall(DMSwarmGetField(dm, cellid, NULL, NULL, (void **)&swarm_cellid));

  PetscCall(DMGetCoordinatesLocal(dmc, &coorlocal));
  PetscCall(DMGetCoordinateSection(dmc, &coordSection));

  pcnt = 0;
  for (e = 0; e < nel; e++) {
    PetscCall(DMPlexVecGetClosure(dmc, coordSection, coorlocal, e, NULL, &elcoor));

    for (q = 0; q < npoints_q; q++) {
      for (d = 0; d < dim; d++) {
        swarm_coor[dim * pcnt + d] = 0.0;
        for (k = 0; k < npe; k++) swarm_coor[dim * pcnt + d] += basis[q][k] * PetscRealPart(elcoor[dim * k + d]);
      }
      swarm_cellid[pcnt] = e;
      pcnt++;
    }
    PetscCall(DMPlexVecRestoreClosure(dmc, coordSection, coorlocal, e, NULL, &elcoor));
  }
  PetscCall(DMSwarmRestoreField(dm, cellid, NULL, NULL, (void **)&swarm_cellid));
  PetscCall(DMSwarmRestoreField(dm, coordFields[0], NULL, NULL, (void **)&swarm_coor));

  PetscCall(PetscFree(xi));
  for (q = 0; q < npoints_q; q++) PetscCall(PetscFree(basis[q]));
  PetscCall(PetscFree(basis));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode private_DMSwarmInsertPointsUsingCellDM_PLEX(DM dm, DM celldm, DMSwarmPICLayoutType layout, PetscInt layout_param)
{
  PetscInt dim;

  PetscFunctionBegin;
  PetscCall(DMGetDimension(celldm, &dim));
  switch (layout) {
  case DMSWARMPIC_LAYOUT_REGULAR:
    PetscCheck(dim != 3, PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "No 3D support for REGULAR+PLEX");
    PetscCall(private_DMSwarmInsertPointsUsingCellDM_PLEX2D_Regular(dm, celldm, layout_param));
    break;
  case DMSWARMPIC_LAYOUT_GAUSS: {
    PetscQuadrature  quad, facequad;
    const PetscReal *xi;
    DMPolytopeType   ct;
    PetscInt         cStart, Nq;

    PetscCall(DMPlexGetHeightStratum(celldm, 0, &cStart, NULL));
    PetscCall(DMPlexGetCellType(celldm, cStart, &ct));
    PetscCall(PetscDTCreateDefaultQuadrature(ct, layout_param, &quad, &facequad));
    PetscCall(PetscQuadratureGetData(quad, NULL, NULL, &Nq, &xi, NULL));
    PetscCall(private_DMSwarmSetPointCoordinatesCellwise_PLEX(dm, celldm, Nq, (PetscReal *)xi));
    PetscCall(PetscQuadratureDestroy(&quad));
    PetscCall(PetscQuadratureDestroy(&facequad));
  } break;
  case DMSWARMPIC_LAYOUT_SUBDIVISION:
    PetscCall(private_DMSwarmInsertPointsUsingCellDM_PLEX_SubDivide(dm, celldm, layout_param));
    break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode private_DMSwarmSetPointCoordinatesCellwise_PLEX(DM dm, DM dmc, PetscInt npoints, PetscReal xi[])
{
  PetscBool     is_simplex, is_tensorcell;
  PetscInt      dim, ps, pe, nel, nfaces, Nfc;
  DMSwarmCellDM celldm;
  PetscReal    *swarm_coor;
  PetscInt     *swarm_cellid;
  const char  **coordFields, *cellid;

  PetscFunctionBegin;
  PetscCall(DMGetDimension(dmc, &dim));

  is_simplex    = PETSC_FALSE;
  is_tensorcell = PETSC_FALSE;
  PetscCall(DMPlexGetHeightStratum(dmc, 0, &ps, &pe));
  PetscCall(DMPlexGetConeSize(dmc, ps, &nfaces));

  if (nfaces == (dim + 1)) is_simplex = PETSC_TRUE;

  switch (dim) {
  case 2:
    if (nfaces == 4) is_tensorcell = PETSC_TRUE;
    break;
  case 3:
    if (nfaces == 6) is_tensorcell = PETSC_TRUE;
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Only support for 2D, 3D");
  }

  /* check points provided fail inside the reference cell */
  if (is_simplex) {
    for (PetscInt p = 0; p < npoints; p++) {
      PetscReal sum;
      for (PetscInt d = 0; d < dim; d++) PetscCheck(xi[dim * p + d] >= -1.0, PetscObjectComm((PetscObject)dm), PETSC_ERR_USER, "Points do not fail inside the simplex domain");
      sum = 0.0;
      for (PetscInt d = 0; d < dim; d++) sum += xi[dim * p + d];
      PetscCheck(sum <= 0.0, PetscObjectComm((PetscObject)dm), PETSC_ERR_USER, "Points do not fail inside the simplex domain");
    }
  } else if (is_tensorcell) {
    for (PetscInt p = 0; p < npoints; p++) {
      for (PetscInt d = 0; d < dim; d++) PetscCheck(PetscAbsReal(xi[dim * p + d]) <= 1.0, PetscObjectComm((PetscObject)dm), PETSC_ERR_USER, "Points do not fail inside the tensor domain [-1,1]^d");
    }
  } else SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Only support for d-simplex and d-tensorcell");

  PetscCall(DMPlexGetHeightStratum(dmc, 0, &ps, &pe));
  nel = pe - ps;

  PetscCall(DMSwarmGetCellDMActive(dm, &celldm));
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm, &Nfc, &coordFields));
  PetscCheck(Nfc == 1, PetscObjectComm((PetscObject)dmc), PETSC_ERR_SUP, "We only support a single coordinate field right now, not %" PetscInt_FMT, Nfc);
  PetscCall(DMSwarmCellDMGetCellID(celldm, &cellid));

  PetscCall(DMSwarmSetLocalSizes(dm, npoints * nel, PETSC_DECIDE));
  PetscCall(DMSwarmGetField(dm, coordFields[0], NULL, NULL, (void **)&swarm_coor));
  PetscCall(DMSwarmGetField(dm, cellid, NULL, NULL, (void **)&swarm_cellid));

  // Use DMPlexReferenceToCoordinates so that arbitrary discretizations work
  for (PetscInt e = 0; e < nel; e++) {
    PetscCall(DMPlexReferenceToCoordinates(dmc, e + ps, npoints, xi, &swarm_coor[npoints * dim * e]));
    for (PetscInt p = 0; p < npoints; p++) swarm_cellid[e * npoints + p] = e + ps;
  }
  PetscCall(DMSwarmRestoreField(dm, cellid, NULL, NULL, (void **)&swarm_cellid));
  PetscCall(DMSwarmRestoreField(dm, coordFields[0], NULL, NULL, (void **)&swarm_coor));
  PetscFunctionReturn(PETSC_SUCCESS);
}
