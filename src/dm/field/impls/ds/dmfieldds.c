#include <petsc/private/dmfieldimpl.h> /*I "petscdmfield.h" I*/
#include <petsc/private/petscfeimpl.h> /*I "petscdmfield.h" I*/
#include <petscfe.h>
#include <petscdmplex.h>
#include <petscds.h>

typedef struct _n_DMField_DS {
  PetscBool    multifieldVec;
  PetscInt     height;   /* Point height at which we want values and number of discretizations */
  PetscInt     fieldNum; /* Number in DS of field which we evaluate */
  PetscObject *disc;     /* Discretizations of this field at each height */
  Vec          vec;      /* Field values */
  DM           dmDG;     /* DM for the DG values */
  PetscObject *discDG;   /* DG Discretizations of this field at each height */
  Vec          vecDG;    /* DG Field values */
} DMField_DS;

static PetscErrorCode DMFieldDestroy_DS(DMField field)
{
  DMField_DS *dsfield = (DMField_DS *)field->data;
  PetscInt    i;

  PetscFunctionBegin;
  PetscCall(VecDestroy(&dsfield->vec));
  for (i = 0; i < dsfield->height; i++) PetscCall(PetscObjectDereference(dsfield->disc[i]));
  PetscCall(PetscFree(dsfield->disc));
  PetscCall(VecDestroy(&dsfield->vecDG));
  if (dsfield->discDG)
    for (i = 0; i < dsfield->height; i++) PetscCall(PetscObjectDereference(dsfield->discDG[i]));
  PetscCall(PetscFree(dsfield->discDG));
  PetscCall(PetscFree(dsfield));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMFieldView_DS(DMField field, PetscViewer viewer)
{
  DMField_DS *dsfield = (DMField_DS *)field->data;
  PetscObject disc;
  PetscBool   iascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  disc = dsfield->disc[0];
  if (iascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "PetscDS field %" PetscInt_FMT "\n", dsfield->fieldNum));
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscCall(PetscObjectView(disc, viewer));
    PetscCall(PetscViewerASCIIPopTab(viewer));
  }
  PetscCall(PetscViewerASCIIPushTab(viewer));
  PetscCheck(!dsfield->multifieldVec, PetscObjectComm((PetscObject)field), PETSC_ERR_SUP, "View of subfield not implemented yet");
  PetscCall(VecView(dsfield->vec, viewer));
  PetscCall(PetscViewerASCIIPopTab(viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMFieldDSGetHeightDisc(DMField field, PetscInt height, PetscObject discList[], PetscObject *disc)
{
  PetscFunctionBegin;
  PetscCheck(height >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Height %" PetscInt_FMT " must be non-negative", height);
  if (!discList[height]) {
    PetscClassId id;

    PetscCall(PetscObjectGetClassId(discList[0], &id));
    if (id == PETSCFE_CLASSID) PetscCall(PetscFECreateHeightTrace((PetscFE)discList[0], height, (PetscFE *)&discList[height]));
  }
  *disc = discList[height];
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* y[m,c] = A[m,n,c] . b[n] */
#define DMFieldDSdot(y, A, b, m, n, c, cast) \
  do { \
    PetscInt _i, _j, _k; \
    for (_i = 0; _i < (m); _i++) { \
      for (_k = 0; _k < (c); _k++) (y)[_i * (c) + _k] = 0.; \
      for (_j = 0; _j < (n); _j++) { \
        for (_k = 0; _k < (c); _k++) (y)[_i * (c) + _k] += (A)[(_i * (n) + _j) * (c) + _k] * cast((b)[_j]); \
      } \
    } \
  } while (0)

/*
  Since this is used for coordinates, we need to allow for the possibility that values come from multiple sections/Vecs, so that we can have DG version of the coordinates for periodicity. This reproduces DMPlexGetCellCoordinates_Internal().
*/
static PetscErrorCode DMFieldGetClosure_Internal(DMField field, PetscInt cell, PetscBool *isDG, PetscInt *Nc, const PetscScalar *array[], PetscScalar *values[])
{
  DMField_DS        *dsfield = (DMField_DS *)field->data;
  DM                 fdm     = dsfield->dmDG;
  PetscSection       s       = NULL;
  const PetscScalar *cvalues;
  PetscInt           pStart, pEnd;

  PetscFunctionBeginHot;
  *isDG   = PETSC_FALSE;
  *Nc     = 0;
  *array  = NULL;
  *values = NULL;
  /* Check for cellwise section */
  if (fdm) PetscCall(DMGetLocalSection(fdm, &s));
  if (!s) goto cg;
  /* Check that the cell exists in the cellwise section */
  PetscCall(PetscSectionGetChart(s, &pStart, &pEnd));
  if (cell < pStart || cell >= pEnd) goto cg;
  /* Check for cellwise coordinates for this cell */
  PetscCall(PetscSectionGetDof(s, cell, Nc));
  if (!*Nc) goto cg;
  /* Check for cellwise coordinates */
  if (!dsfield->vecDG) goto cg;
  /* Get cellwise coordinates */
  PetscCall(VecGetArrayRead(dsfield->vecDG, array));
  PetscCall(DMPlexPointLocalRead(fdm, cell, *array, &cvalues));
  PetscCall(DMGetWorkArray(fdm, *Nc, MPIU_SCALAR, values));
  PetscCall(PetscArraycpy(*values, cvalues, *Nc));
  PetscCall(VecRestoreArrayRead(dsfield->vecDG, array));
  *isDG = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
cg:
  /* Use continuous values */
  PetscCall(DMFieldGetDM(field, &fdm));
  PetscCall(DMGetLocalSection(fdm, &s));
  PetscCall(PetscSectionGetField(s, dsfield->fieldNum, &s));
  PetscCall(DMPlexVecGetClosure(fdm, s, dsfield->vec, cell, Nc, values));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMFieldRestoreClosure_Internal(DMField field, PetscInt cell, PetscBool *isDG, PetscInt *Nc, const PetscScalar *array[], PetscScalar *values[])
{
  DMField_DS  *dsfield = (DMField_DS *)field->data;
  DM           fdm;
  PetscSection s;

  PetscFunctionBeginHot;
  if (*isDG) {
    PetscCall(DMRestoreWorkArray(dsfield->dmDG, *Nc, MPIU_SCALAR, values));
  } else {
    PetscCall(DMFieldGetDM(field, &fdm));
    PetscCall(DMGetLocalSection(fdm, &s));
    PetscCall(DMPlexVecRestoreClosure(fdm, s, dsfield->vec, cell, Nc, (PetscScalar **)values));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* TODO: Reorganize interface so that I can reuse a tabulation rather than mallocing each time */
static PetscErrorCode DMFieldEvaluateFE_DS(DMField field, IS pointIS, PetscQuadrature quad, PetscDataType type, void *B, void *D, void *H)
{
  DMField_DS      *dsfield = (DMField_DS *)field->data;
  DM               dm;
  PetscObject      disc;
  PetscClassId     classid;
  PetscInt         nq, nc, dim, meshDim, numCells;
  PetscSection     section;
  const PetscReal *qpoints;
  PetscBool        isStride;
  const PetscInt  *points = NULL;
  PetscInt         sfirst = -1, stride = -1;

  PetscFunctionBeginHot;
  dm = field->dm;
  nc = field->numComponents;
  PetscCall(PetscQuadratureGetData(quad, &dim, NULL, &nq, &qpoints, NULL));
  PetscCall(DMFieldDSGetHeightDisc(field, dsfield->height - 1 - dim, dsfield->disc, &disc));
  PetscCall(DMGetDimension(dm, &meshDim));
  PetscCall(DMGetLocalSection(dm, &section));
  PetscCall(PetscSectionGetField(section, dsfield->fieldNum, &section));
  PetscCall(PetscObjectGetClassId(disc, &classid));
  /* TODO: batch */
  PetscCall(PetscObjectTypeCompare((PetscObject)pointIS, ISSTRIDE, &isStride));
  PetscCall(ISGetLocalSize(pointIS, &numCells));
  if (isStride) PetscCall(ISStrideGetInfo(pointIS, &sfirst, &stride));
  else PetscCall(ISGetIndices(pointIS, &points));
  if (classid == PETSCFE_CLASSID) {
    PetscFE         fe = (PetscFE)disc;
    PetscInt        feDim, i;
    PetscInt        K = H ? 2 : (D ? 1 : (B ? 0 : -1));
    PetscTabulation T;

    PetscCall(PetscFEGetDimension(fe, &feDim));
    PetscCall(PetscFECreateTabulation(fe, 1, nq, qpoints, K, &T));
    for (i = 0; i < numCells; i++) {
      PetscInt           c = isStride ? (sfirst + i * stride) : points[i];
      PetscInt           closureSize;
      const PetscScalar *array;
      PetscScalar       *elem = NULL;
      PetscBool          isDG;

      PetscCall(DMFieldGetClosure_Internal(field, c, &isDG, &closureSize, &array, &elem));
      if (B) {
        /* field[c] = T[q,b,c] . coef[b], so v[c] = T[q,b,c] . coords[b] */
        if (type == PETSC_SCALAR) {
          PetscScalar *cB = &((PetscScalar *)B)[nc * nq * i];

          DMFieldDSdot(cB, T->T[0], elem, nq, feDim, nc, (PetscScalar));
        } else {
          PetscReal *cB = &((PetscReal *)B)[nc * nq * i];

          DMFieldDSdot(cB, T->T[0], elem, nq, feDim, nc, PetscRealPart);
        }
      }
      if (D) {
        if (type == PETSC_SCALAR) {
          PetscScalar *cD = &((PetscScalar *)D)[nc * nq * dim * i];

          DMFieldDSdot(cD, T->T[1], elem, nq, feDim, (nc * dim), (PetscScalar));
        } else {
          PetscReal *cD = &((PetscReal *)D)[nc * nq * dim * i];

          DMFieldDSdot(cD, T->T[1], elem, nq, feDim, (nc * dim), PetscRealPart);
        }
      }
      if (H) {
        if (type == PETSC_SCALAR) {
          PetscScalar *cH = &((PetscScalar *)H)[nc * nq * dim * dim * i];

          DMFieldDSdot(cH, T->T[2], elem, nq, feDim, (nc * dim * dim), (PetscScalar));
        } else {
          PetscReal *cH = &((PetscReal *)H)[nc * nq * dim * dim * i];

          DMFieldDSdot(cH, T->T[2], elem, nq, feDim, (nc * dim * dim), PetscRealPart);
        }
      }
      PetscCall(DMFieldRestoreClosure_Internal(field, c, &isDG, &closureSize, &array, &elem));
    }
    PetscCall(PetscTabulationDestroy(&T));
  } else SETERRQ(PetscObjectComm((PetscObject)field), PETSC_ERR_SUP, "Not implemented");
  if (!isStride) PetscCall(ISRestoreIndices(pointIS, &points));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMFieldEvaluate_DS(DMField field, Vec points, PetscDataType datatype, void *B, void *D, void *H)
{
  DMField_DS        *dsfield = (DMField_DS *)field->data;
  PetscSF            cellSF  = NULL;
  const PetscSFNode *cells;
  PetscInt           c, nFound, numCells, feDim, nc;
  const PetscInt    *cellDegrees;
  const PetscScalar *pointsArray;
  PetscScalar       *cellPoints;
  PetscInt           gatherSize, gatherMax;
  PetscInt           dim, dimR, offset;
  MPI_Datatype       pointType;
  PetscObject        cellDisc;
  PetscFE            cellFE;
  PetscClassId       discID;
  PetscReal         *coordsReal, *coordsRef;
  PetscSection       section;
  PetscScalar       *cellBs = NULL, *cellDs = NULL, *cellHs = NULL;
  PetscReal         *cellBr = NULL, *cellDr = NULL, *cellHr = NULL;
  PetscReal         *v, *J, *invJ, *detJ;

  PetscFunctionBegin;
  nc = field->numComponents;
  PetscCall(DMGetLocalSection(field->dm, &section));
  PetscCall(DMFieldDSGetHeightDisc(field, 0, dsfield->disc, &cellDisc));
  PetscCall(PetscObjectGetClassId(cellDisc, &discID));
  PetscCheck(discID == PETSCFE_CLASSID, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Discretization type not supported");
  cellFE = (PetscFE)cellDisc;
  PetscCall(PetscFEGetDimension(cellFE, &feDim));
  PetscCall(DMGetCoordinateDim(field->dm, &dim));
  PetscCall(DMGetDimension(field->dm, &dimR));
  PetscCall(DMLocatePoints(field->dm, points, DM_POINTLOCATION_NONE, &cellSF));
  PetscCall(PetscSFGetGraph(cellSF, &numCells, &nFound, NULL, &cells));
  for (c = 0; c < nFound; c++) PetscCheck(cells[c].index >= 0, PetscObjectComm((PetscObject)points), PETSC_ERR_ARG_WRONG, "Point %" PetscInt_FMT " could not be located", c);
  PetscCall(PetscSFComputeDegreeBegin(cellSF, &cellDegrees));
  PetscCall(PetscSFComputeDegreeEnd(cellSF, &cellDegrees));
  for (c = 0, gatherSize = 0, gatherMax = 0; c < numCells; c++) {
    gatherMax = PetscMax(gatherMax, cellDegrees[c]);
    gatherSize += cellDegrees[c];
  }
  PetscCall(PetscMalloc3(gatherSize * dim, &cellPoints, gatherMax * dim, &coordsReal, gatherMax * dimR, &coordsRef));
  PetscCall(PetscMalloc4(gatherMax * dimR, &v, gatherMax * dimR * dimR, &J, gatherMax * dimR * dimR, &invJ, gatherMax, &detJ));
  if (datatype == PETSC_SCALAR) PetscCall(PetscMalloc3((B ? (size_t)nc * gatherSize : 0), &cellBs, (D ? (size_t)nc * dim * gatherSize : 0), &cellDs, (H ? (size_t)nc * dim * dim * gatherSize : 0), &cellHs));
  else PetscCall(PetscMalloc3((B ? (size_t)nc * gatherSize : 0), &cellBr, (D ? (size_t)nc * dim * gatherSize : 0), &cellDr, (H ? (size_t)nc * dim * dim * gatherSize : 0), &cellHr));

  PetscCallMPI(MPI_Type_contiguous((PetscMPIInt)dim, MPIU_SCALAR, &pointType));
  PetscCallMPI(MPI_Type_commit(&pointType));
  PetscCall(VecGetArrayRead(points, &pointsArray));
  PetscCall(PetscSFGatherBegin(cellSF, pointType, pointsArray, cellPoints));
  PetscCall(PetscSFGatherEnd(cellSF, pointType, pointsArray, cellPoints));
  PetscCall(VecRestoreArrayRead(points, &pointsArray));
  for (c = 0, offset = 0; c < numCells; c++) {
    PetscInt nq = cellDegrees[c], p;

    if (nq) {
      PetscInt           K = H ? 2 : (D ? 1 : (B ? 0 : -1));
      PetscTabulation    T;
      PetscQuadrature    quad;
      const PetscScalar *array;
      PetscScalar       *elem = NULL;
      PetscReal         *quadPoints;
      PetscBool          isDG;
      PetscInt           closureSize, d, e, f, g;

      for (p = 0; p < dim * nq; p++) coordsReal[p] = PetscRealPart(cellPoints[dim * offset + p]);
      PetscCall(DMPlexCoordinatesToReference(field->dm, c, nq, coordsReal, coordsRef));
      PetscCall(PetscFECreateTabulation(cellFE, 1, nq, coordsRef, K, &T));
      PetscCall(PetscQuadratureCreate(PETSC_COMM_SELF, &quad));
      PetscCall(PetscMalloc1(dimR * nq, &quadPoints));
      for (p = 0; p < dimR * nq; p++) quadPoints[p] = coordsRef[p];
      PetscCall(PetscQuadratureSetData(quad, dimR, 0, nq, quadPoints, NULL));
      PetscCall(DMPlexComputeCellGeometryFEM(field->dm, c, quad, v, J, invJ, detJ));
      PetscCall(PetscQuadratureDestroy(&quad));
      PetscCall(DMFieldGetClosure_Internal(field, c, &isDG, &closureSize, &array, &elem));
      if (B) {
        if (datatype == PETSC_SCALAR) {
          PetscScalar *cB = &cellBs[nc * offset];

          DMFieldDSdot(cB, T->T[0], elem, nq, feDim, nc, (PetscScalar));
        } else {
          PetscReal *cB = &cellBr[nc * offset];

          DMFieldDSdot(cB, T->T[0], elem, nq, feDim, nc, PetscRealPart);
        }
      }
      if (D) {
        if (datatype == PETSC_SCALAR) {
          PetscScalar *cD = &cellDs[nc * dim * offset];

          DMFieldDSdot(cD, T->T[1], elem, nq, feDim, (nc * dim), (PetscScalar));
          for (p = 0; p < nq; p++) {
            for (g = 0; g < nc; g++) {
              PetscScalar vs[3];

              for (d = 0; d < dimR; d++) {
                vs[d] = 0.;
                for (e = 0; e < dimR; e++) vs[d] += invJ[dimR * dimR * p + e * dimR + d] * cD[(nc * p + g) * dimR + e];
              }
              for (d = 0; d < dimR; d++) cD[(nc * p + g) * dimR + d] = vs[d];
            }
          }
        } else {
          PetscReal *cD = &cellDr[nc * dim * offset];

          DMFieldDSdot(cD, T->T[1], elem, nq, feDim, (nc * dim), PetscRealPart);
          for (p = 0; p < nq; p++) {
            for (g = 0; g < nc; g++) {
              for (d = 0; d < dimR; d++) {
                v[d] = 0.;
                for (e = 0; e < dimR; e++) v[d] += invJ[dimR * dimR * p + e * dimR + d] * cD[(nc * p + g) * dimR + e];
              }
              for (d = 0; d < dimR; d++) cD[(nc * p + g) * dimR + d] = v[d];
            }
          }
        }
      }
      if (H) {
        if (datatype == PETSC_SCALAR) {
          PetscScalar *cH = &cellHs[nc * dim * dim * offset];

          DMFieldDSdot(cH, T->T[2], elem, nq, feDim, (nc * dim * dim), (PetscScalar));
          for (p = 0; p < nq; p++) {
            for (g = 0; g < nc * dimR; g++) {
              PetscScalar vs[3];

              for (d = 0; d < dimR; d++) {
                vs[d] = 0.;
                for (e = 0; e < dimR; e++) vs[d] += invJ[dimR * dimR * p + e * dimR + d] * cH[(nc * dimR * p + g) * dimR + e];
              }
              for (d = 0; d < dimR; d++) cH[(nc * dimR * p + g) * dimR + d] = vs[d];
            }
            for (g = 0; g < nc; g++) {
              for (f = 0; f < dimR; f++) {
                PetscScalar vs[3];

                for (d = 0; d < dimR; d++) {
                  vs[d] = 0.;
                  for (e = 0; e < dimR; e++) vs[d] += invJ[dimR * dimR * p + e * dimR + d] * cH[((nc * p + g) * dimR + e) * dimR + f];
                }
                for (d = 0; d < dimR; d++) cH[((nc * p + g) * dimR + d) * dimR + f] = vs[d];
              }
            }
          }
        } else {
          PetscReal *cH = &cellHr[nc * dim * dim * offset];

          DMFieldDSdot(cH, T->T[2], elem, nq, feDim, (nc * dim * dim), PetscRealPart);
          for (p = 0; p < nq; p++) {
            for (g = 0; g < nc * dimR; g++) {
              for (d = 0; d < dimR; d++) {
                v[d] = 0.;
                for (e = 0; e < dimR; e++) v[d] += invJ[dimR * dimR * p + e * dimR + d] * cH[(nc * dimR * p + g) * dimR + e];
              }
              for (d = 0; d < dimR; d++) cH[(nc * dimR * p + g) * dimR + d] = v[d];
            }
            for (g = 0; g < nc; g++) {
              for (f = 0; f < dimR; f++) {
                for (d = 0; d < dimR; d++) {
                  v[d] = 0.;
                  for (e = 0; e < dimR; e++) v[d] += invJ[dimR * dimR * p + e * dimR + d] * cH[((nc * p + g) * dimR + e) * dimR + f];
                }
                for (d = 0; d < dimR; d++) cH[((nc * p + g) * dimR + d) * dimR + f] = v[d];
              }
            }
          }
        }
      }
      PetscCall(DMFieldRestoreClosure_Internal(field, c, &isDG, &closureSize, &array, &elem));
      PetscCall(PetscTabulationDestroy(&T));
    }
    offset += nq;
  }
  {
    MPI_Datatype origtype;

    if (datatype == PETSC_SCALAR) {
      origtype = MPIU_SCALAR;
    } else {
      origtype = MPIU_REAL;
    }
    if (B) {
      MPI_Datatype Btype;

      PetscCallMPI(MPI_Type_contiguous((PetscMPIInt)nc, origtype, &Btype));
      PetscCallMPI(MPI_Type_commit(&Btype));
      PetscCall(PetscSFScatterBegin(cellSF, Btype, (datatype == PETSC_SCALAR) ? (void *)cellBs : (void *)cellBr, B));
      PetscCall(PetscSFScatterEnd(cellSF, Btype, (datatype == PETSC_SCALAR) ? (void *)cellBs : (void *)cellBr, B));
      PetscCallMPI(MPI_Type_free(&Btype));
    }
    if (D) {
      MPI_Datatype Dtype;

      PetscCallMPI(MPI_Type_contiguous((PetscMPIInt)(nc * dim), origtype, &Dtype));
      PetscCallMPI(MPI_Type_commit(&Dtype));
      PetscCall(PetscSFScatterBegin(cellSF, Dtype, (datatype == PETSC_SCALAR) ? (void *)cellDs : (void *)cellDr, D));
      PetscCall(PetscSFScatterEnd(cellSF, Dtype, (datatype == PETSC_SCALAR) ? (void *)cellDs : (void *)cellDr, D));
      PetscCallMPI(MPI_Type_free(&Dtype));
    }
    if (H) {
      MPI_Datatype Htype;

      PetscCallMPI(MPI_Type_contiguous((PetscMPIInt)(nc * dim * dim), origtype, &Htype));
      PetscCallMPI(MPI_Type_commit(&Htype));
      PetscCall(PetscSFScatterBegin(cellSF, Htype, (datatype == PETSC_SCALAR) ? (void *)cellHs : (void *)cellHr, H));
      PetscCall(PetscSFScatterEnd(cellSF, Htype, (datatype == PETSC_SCALAR) ? (void *)cellHs : (void *)cellHr, H));
      PetscCallMPI(MPI_Type_free(&Htype));
    }
  }
  PetscCall(PetscFree4(v, J, invJ, detJ));
  PetscCall(PetscFree3(cellBr, cellDr, cellHr));
  PetscCall(PetscFree3(cellBs, cellDs, cellHs));
  PetscCall(PetscFree3(cellPoints, coordsReal, coordsRef));
  PetscCallMPI(MPI_Type_free(&pointType));
  PetscCall(PetscSFDestroy(&cellSF));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMFieldEvaluateFV_DS(DMField field, IS pointIS, PetscDataType type, void *B, void *D, void *H)
{
  DMField_DS      *dsfield = (DMField_DS *)field->data;
  PetscInt         h, imin;
  PetscInt         dim;
  PetscClassId     id;
  PetscQuadrature  quad = NULL;
  PetscInt         maxDegree;
  PetscFEGeom     *geom;
  PetscInt         Nq, Nc, dimC, qNc, N;
  PetscInt         numPoints;
  void            *qB = NULL, *qD = NULL, *qH = NULL;
  const PetscReal *weights;
  MPI_Datatype     mpitype = type == PETSC_SCALAR ? MPIU_SCALAR : MPIU_REAL;
  PetscObject      disc;
  DMField          coordField;

  PetscFunctionBegin;
  Nc = field->numComponents;
  PetscCall(DMGetCoordinateDim(field->dm, &dimC));
  PetscCall(DMGetDimension(field->dm, &dim));
  PetscCall(ISGetLocalSize(pointIS, &numPoints));
  PetscCall(ISGetMinMax(pointIS, &imin, NULL));
  for (h = 0; h < dsfield->height; h++) {
    PetscInt hEnd;

    PetscCall(DMPlexGetHeightStratum(field->dm, h, NULL, &hEnd));
    if (imin < hEnd) break;
  }
  dim -= h;
  PetscCall(DMFieldDSGetHeightDisc(field, h, dsfield->disc, &disc));
  PetscCall(PetscObjectGetClassId(disc, &id));
  PetscCheck(id == PETSCFE_CLASSID, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Discretization not supported");
  PetscCall(DMGetCoordinateField(field->dm, &coordField));
  PetscCall(DMFieldGetDegree(coordField, pointIS, NULL, &maxDegree));
  if (maxDegree <= 1) PetscCall(DMFieldCreateDefaultQuadrature(coordField, pointIS, &quad));
  if (!quad) PetscCall(DMFieldCreateDefaultQuadrature(field, pointIS, &quad));
  PetscCheck(quad, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Could not determine quadrature for cell averages");
  PetscCall(DMFieldCreateFEGeom(coordField, pointIS, quad, PETSC_FALSE, &geom));
  PetscCall(PetscQuadratureGetData(quad, NULL, &qNc, &Nq, NULL, &weights));
  PetscCheck(qNc == 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Expected scalar quadrature components");
  N = numPoints * Nq * Nc;
  if (B) PetscCall(DMGetWorkArray(field->dm, N, mpitype, &qB));
  if (D) PetscCall(DMGetWorkArray(field->dm, N * dimC, mpitype, &qD));
  if (H) PetscCall(DMGetWorkArray(field->dm, N * dimC * dimC, mpitype, &qH));
  PetscCall(DMFieldEvaluateFE(field, pointIS, quad, type, qB, qD, qH));
  if (B) {
    PetscInt i, j, k;

    if (type == PETSC_SCALAR) {
      PetscScalar *sB  = (PetscScalar *)B;
      PetscScalar *sqB = (PetscScalar *)qB;

      for (i = 0; i < numPoints; i++) {
        PetscReal vol = 0.;

        for (j = 0; j < Nc; j++) sB[i * Nc + j] = 0.;
        for (k = 0; k < Nq; k++) {
          vol += geom->detJ[i * Nq + k] * weights[k];
          for (j = 0; j < Nc; j++) sB[i * Nc + j] += geom->detJ[i * Nq + k] * weights[k] * sqB[(i * Nq + k) * Nc + j];
        }
        for (k = 0; k < Nc; k++) sB[i * Nc + k] /= vol;
      }
    } else {
      PetscReal *rB  = (PetscReal *)B;
      PetscReal *rqB = (PetscReal *)qB;

      for (i = 0; i < numPoints; i++) {
        PetscReal vol = 0.;

        for (j = 0; j < Nc; j++) rB[i * Nc + j] = 0.;
        for (k = 0; k < Nq; k++) {
          vol += geom->detJ[i * Nq + k] * weights[k];
          for (j = 0; j < Nc; j++) rB[i * Nc + j] += weights[k] * rqB[(i * Nq + k) * Nc + j];
        }
        for (k = 0; k < Nc; k++) rB[i * Nc + k] /= vol;
      }
    }
  }
  if (D) {
    PetscInt i, j, k, l, m;

    if (type == PETSC_SCALAR) {
      PetscScalar *sD  = (PetscScalar *)D;
      PetscScalar *sqD = (PetscScalar *)qD;

      for (i = 0; i < numPoints; i++) {
        PetscReal vol = 0.;

        for (j = 0; j < Nc * dimC; j++) sD[i * Nc * dimC + j] = 0.;
        for (k = 0; k < Nq; k++) {
          vol += geom->detJ[i * Nq + k] * weights[k];
          for (j = 0; j < Nc; j++) {
            PetscScalar pD[3] = {0., 0., 0.};

            for (l = 0; l < dimC; l++) {
              for (m = 0; m < dim; m++) pD[l] += geom->invJ[((i * Nq + k) * dimC + m) * dimC + l] * sqD[((i * Nq + k) * Nc + j) * dim + m];
            }
            for (l = 0; l < dimC; l++) sD[(i * Nc + j) * dimC + l] += geom->detJ[i * Nq + k] * weights[k] * pD[l];
          }
        }
        for (k = 0; k < Nc * dimC; k++) sD[i * Nc * dimC + k] /= vol;
      }
    } else {
      PetscReal *rD  = (PetscReal *)D;
      PetscReal *rqD = (PetscReal *)qD;

      for (i = 0; i < numPoints; i++) {
        PetscReal vol = 0.;

        for (j = 0; j < Nc * dimC; j++) rD[i * Nc * dimC + j] = 0.;
        for (k = 0; k < Nq; k++) {
          vol += geom->detJ[i * Nq + k] * weights[k];
          for (j = 0; j < Nc; j++) {
            PetscReal pD[3] = {0., 0., 0.};

            for (l = 0; l < dimC; l++) {
              for (m = 0; m < dim; m++) pD[l] += geom->invJ[((i * Nq + k) * dimC + m) * dimC + l] * rqD[((i * Nq + k) * Nc + j) * dim + m];
            }
            for (l = 0; l < dimC; l++) rD[(i * Nc + j) * dimC + l] += geom->detJ[i * Nq + k] * weights[k] * pD[l];
          }
        }
        for (k = 0; k < Nc * dimC; k++) rD[i * Nc * dimC + k] /= vol;
      }
    }
  }
  if (H) {
    PetscInt i, j, k, l, m, q, r;

    if (type == PETSC_SCALAR) {
      PetscScalar *sH  = (PetscScalar *)H;
      PetscScalar *sqH = (PetscScalar *)qH;

      for (i = 0; i < numPoints; i++) {
        PetscReal vol = 0.;

        for (j = 0; j < Nc * dimC * dimC; j++) sH[i * Nc * dimC * dimC + j] = 0.;
        for (k = 0; k < Nq; k++) {
          const PetscReal *invJ = &geom->invJ[(i * Nq + k) * dimC * dimC];

          vol += geom->detJ[i * Nq + k] * weights[k];
          for (j = 0; j < Nc; j++) {
            PetscScalar pH[3][3] = {
              {0., 0., 0.},
              {0., 0., 0.},
              {0., 0., 0.}
            };
            const PetscScalar *spH = &sqH[((i * Nq + k) * Nc + j) * dimC * dimC];

            for (l = 0; l < dimC; l++) {
              for (m = 0; m < dimC; m++) {
                for (q = 0; q < dim; q++) {
                  for (r = 0; r < dim; r++) pH[l][m] += invJ[q * dimC + l] * invJ[r * dimC + m] * spH[q * dim + r];
                }
              }
            }
            for (l = 0; l < dimC; l++) {
              for (m = 0; m < dimC; m++) sH[(i * Nc + j) * dimC * dimC + l * dimC + m] += geom->detJ[i * Nq + k] * weights[k] * pH[l][m];
            }
          }
        }
        for (k = 0; k < Nc * dimC * dimC; k++) sH[i * Nc * dimC * dimC + k] /= vol;
      }
    } else {
      PetscReal *rH  = (PetscReal *)H;
      PetscReal *rqH = (PetscReal *)qH;

      for (i = 0; i < numPoints; i++) {
        PetscReal vol = 0.;

        for (j = 0; j < Nc * dimC * dimC; j++) rH[i * Nc * dimC * dimC + j] = 0.;
        for (k = 0; k < Nq; k++) {
          const PetscReal *invJ = &geom->invJ[(i * Nq + k) * dimC * dimC];

          vol += geom->detJ[i * Nq + k] * weights[k];
          for (j = 0; j < Nc; j++) {
            PetscReal pH[3][3] = {
              {0., 0., 0.},
              {0., 0., 0.},
              {0., 0., 0.}
            };
            const PetscReal *rpH = &rqH[((i * Nq + k) * Nc + j) * dimC * dimC];

            for (l = 0; l < dimC; l++) {
              for (m = 0; m < dimC; m++) {
                for (q = 0; q < dim; q++) {
                  for (r = 0; r < dim; r++) pH[l][m] += invJ[q * dimC + l] * invJ[r * dimC + m] * rpH[q * dim + r];
                }
              }
            }
            for (l = 0; l < dimC; l++) {
              for (m = 0; m < dimC; m++) rH[(i * Nc + j) * dimC * dimC + l * dimC + m] += geom->detJ[i * Nq + k] * weights[k] * pH[l][m];
            }
          }
        }
        for (k = 0; k < Nc * dimC * dimC; k++) rH[i * Nc * dimC * dimC + k] /= vol;
      }
    }
  }
  if (B) PetscCall(DMRestoreWorkArray(field->dm, N, mpitype, &qB));
  if (D) PetscCall(DMRestoreWorkArray(field->dm, N * dimC, mpitype, &qD));
  if (H) PetscCall(DMRestoreWorkArray(field->dm, N * dimC * dimC, mpitype, &qH));
  PetscCall(PetscFEGeomDestroy(&geom));
  PetscCall(PetscQuadratureDestroy(&quad));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMFieldGetDegree_DS(DMField field, IS pointIS, PetscInt *minDegree, PetscInt *maxDegree)
{
  DMField_DS  *dsfield;
  PetscObject  disc;
  PetscInt     h, imin, imax;
  PetscClassId id;

  PetscFunctionBegin;
  dsfield = (DMField_DS *)field->data;
  PetscCall(ISGetMinMax(pointIS, &imin, &imax));
  if (imin >= imax) {
    h = 0;
  } else {
    for (h = 0; h < dsfield->height; h++) {
      PetscInt hEnd;

      PetscCall(DMPlexGetHeightStratum(field->dm, h, NULL, &hEnd));
      if (imin < hEnd) break;
    }
  }
  PetscCall(DMFieldDSGetHeightDisc(field, h, dsfield->disc, &disc));
  PetscCall(PetscObjectGetClassId(disc, &id));
  if (id == PETSCFE_CLASSID) {
    PetscFE    fe = (PetscFE)disc;
    PetscSpace sp;

    PetscCall(PetscFEGetBasisSpace(fe, &sp));
    PetscCall(PetscSpaceGetDegree(sp, minDegree, maxDegree));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMFieldGetFVQuadrature_Internal(DMField field, IS pointIS, PetscQuadrature *quad)
{
  DM              dm = field->dm;
  const PetscInt *points;
  DMPolytopeType  ct;
  PetscInt        dim, n;
  PetscBool       isplex;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)dm, DMPLEX, &isplex));
  PetscCall(ISGetLocalSize(pointIS, &n));
  if (isplex && n) {
    PetscCall(DMGetDimension(dm, &dim));
    PetscCall(ISGetIndices(pointIS, &points));
    PetscCall(DMPlexGetCellType(dm, points[0], &ct));
    switch (ct) {
    case DM_POLYTOPE_TRIANGLE:
    case DM_POLYTOPE_TETRAHEDRON:
      PetscCall(PetscDTStroudConicalQuadrature(dim, 1, 1, -1.0, 1.0, quad));
      break;
    default:
      PetscCall(PetscDTGaussTensorQuadrature(dim, 1, 1, -1.0, 1.0, quad));
    }
    PetscCall(ISRestoreIndices(pointIS, &points));
  } else PetscCall(DMFieldCreateDefaultQuadrature(field, pointIS, quad));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMFieldCreateDefaultQuadrature_DS(DMField field, IS pointIS, PetscQuadrature *quad)
{
  PetscInt     h, dim, imax, imin, cellHeight;
  DM           dm;
  DMField_DS  *dsfield;
  PetscObject  disc;
  PetscFE      fe;
  PetscClassId id;

  PetscFunctionBegin;
  dm      = field->dm;
  dsfield = (DMField_DS *)field->data;
  PetscCall(ISGetMinMax(pointIS, &imin, &imax));
  PetscCall(DMGetDimension(dm, &dim));
  for (h = 0; h <= dim; h++) {
    PetscInt hStart, hEnd;

    PetscCall(DMPlexGetHeightStratum(dm, h, &hStart, &hEnd));
    if (imax >= hStart && imin < hEnd) break;
  }
  PetscCall(DMPlexGetVTKCellHeight(dm, &cellHeight));
  h -= cellHeight;
  *quad = NULL;
  if (h < dsfield->height) {
    PetscCall(DMFieldDSGetHeightDisc(field, h, dsfield->disc, &disc));
    PetscCall(PetscObjectGetClassId(disc, &id));
    if (id != PETSCFE_CLASSID) PetscFunctionReturn(PETSC_SUCCESS);
    fe = (PetscFE)disc;
    PetscCall(PetscFEGetQuadrature(fe, quad));
    PetscCall(PetscObjectReference((PetscObject)*quad));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMFieldComputeFaceData_DS(DMField field, IS pointIS, PetscQuadrature quad, PetscFEGeom *geom)
{
  const PetscInt *points;
  PetscInt        p, dim, dE, numFaces, Nq;
  PetscInt        maxDegree;
  DMLabel         depthLabel;
  IS              cellIS;
  DM              dm = field->dm;

  PetscFunctionBegin;
  dim = geom->dim;
  dE  = geom->dimEmbed;
  PetscCall(DMPlexGetDepthLabel(dm, &depthLabel));
  PetscCall(DMLabelGetStratumIS(depthLabel, dim + 1, &cellIS));
  PetscCall(DMFieldGetDegree(field, cellIS, NULL, &maxDegree));
  PetscCall(ISGetIndices(pointIS, &points));
  numFaces = geom->numCells;
  Nq       = geom->numPoints;
  /* First, set local faces and flip normals so that they are outward for the first supporting cell */
  for (p = 0; p < numFaces; p++) {
    PetscInt        point = points[p];
    PetscInt        suppSize, s, coneSize, c, numChildren;
    const PetscInt *supp;

    PetscCall(DMPlexGetTreeChildren(dm, point, &numChildren, NULL));
    PetscCheck(!numChildren, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Face data not valid for facets with children");
    PetscCall(DMPlexGetSupportSize(dm, point, &suppSize));
    PetscCheck(suppSize <= 2, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %" PetscInt_FMT " has %" PetscInt_FMT " support, expected at most 2", point, suppSize);
    if (!suppSize) continue;
    PetscCall(DMPlexGetSupport(dm, point, &supp));
    for (s = 0; s < suppSize; ++s) {
      const PetscInt *cone, *ornt;

      PetscCall(DMPlexGetConeSize(dm, supp[s], &coneSize));
      PetscCall(DMPlexGetOrientedCone(dm, supp[s], &cone, &ornt));
      for (c = 0; c < coneSize; ++c)
        if (cone[c] == point) break;
      geom->face[p][s * 2 + 0] = c;
      geom->face[p][s * 2 + 1] = ornt[c];
      PetscCall(DMPlexRestoreOrientedCone(dm, supp[s], &cone, &ornt));
      PetscCheck(c != coneSize, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid connectivity: point %" PetscInt_FMT " not found in cone of support point %" PetscInt_FMT, point, supp[s]);
    }
    if (geom->face[p][1] < 0) {
      PetscInt Np = geom->numPoints, q, dE = geom->dimEmbed, d;

      for (q = 0; q < Np; ++q)
        for (d = 0; d < dE; ++d) geom->n[(p * Np + q) * dE + d] = -geom->n[(p * Np + q) * dE + d];
    }
  }
  if (maxDegree <= 1) {
    PetscQuadrature cellQuad = NULL;
    PetscInt        numCells, offset, *cells;
    PetscFEGeom    *cellGeom;
    IS              suppIS;

    if (quad) {
      DM         dm;
      PetscReal *points, *weights;
      PetscInt   tdim, Nc, Np;

      PetscCall(DMFieldGetDM(field, &dm));
      PetscCall(DMGetDimension(dm, &tdim));
      if (tdim > dim) {
        // Make a compatible cell quadrature (points don't matter since its affine)
        PetscCall(PetscQuadratureCreate(PETSC_COMM_SELF, &cellQuad));
        PetscCall(PetscQuadratureGetData(quad, NULL, &Nc, &Np, NULL, NULL));
        PetscCall(PetscCalloc1((dim + 1) * Np, &points));
        PetscCall(PetscCalloc1(Nc * Np, &weights));
        PetscCall(PetscQuadratureSetData(cellQuad, dim + 1, Nc, Np, points, weights));
      } else {
        // TODO J will be wrong here, but other things need to be fixed
        //   This path comes from calling DMProjectBdFieldLabelLocal() in Plex ex5
        PetscCall(PetscObjectReference((PetscObject)quad));
        cellQuad = quad;
      }
    }
    for (p = 0, numCells = 0; p < numFaces; p++) {
      PetscInt point = points[p];
      PetscInt numSupp, numChildren;

      PetscCall(DMPlexGetTreeChildren(dm, point, &numChildren, NULL));
      PetscCheck(!numChildren, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Face data not valid for facets with children");
      PetscCall(DMPlexGetSupportSize(dm, point, &numSupp));
      PetscCheck(numSupp <= 2, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %" PetscInt_FMT " has %" PetscInt_FMT " support, expected at most 2", point, numSupp);
      numCells += numSupp;
    }
    PetscCall(PetscMalloc1(numCells, &cells));
    for (p = 0, offset = 0; p < numFaces; p++) {
      PetscInt        point = points[p];
      PetscInt        numSupp, s;
      const PetscInt *supp;

      PetscCall(DMPlexGetSupportSize(dm, point, &numSupp));
      PetscCall(DMPlexGetSupport(dm, point, &supp));
      for (s = 0; s < numSupp; s++, offset++) cells[offset] = supp[s];
    }
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF, numCells, cells, PETSC_USE_POINTER, &suppIS));
    PetscCall(DMFieldCreateFEGeom(field, suppIS, cellQuad, PETSC_FALSE, &cellGeom));
    for (p = 0, offset = 0; p < numFaces; p++) {
      PetscInt        point = points[p];
      PetscInt        numSupp, s, q;
      const PetscInt *supp;

      PetscCall(DMPlexGetSupportSize(dm, point, &numSupp));
      PetscCall(DMPlexGetSupport(dm, point, &supp));
      for (s = 0; s < numSupp; s++, offset++) {
        for (q = 0; q < Nq * dE * dE; q++) {
          geom->suppJ[s][p * Nq * dE * dE + q]    = cellGeom->J[offset * Nq * dE * dE + q];
          geom->suppInvJ[s][p * Nq * dE * dE + q] = cellGeom->invJ[offset * Nq * dE * dE + q];
        }
        for (q = 0; q < Nq; q++) geom->suppDetJ[s][p * Nq + q] = cellGeom->detJ[offset * Nq + q];
      }
    }
    PetscCall(PetscFEGeomDestroy(&cellGeom));
    PetscCall(PetscQuadratureDestroy(&cellQuad));
    PetscCall(ISDestroy(&suppIS));
    PetscCall(PetscFree(cells));
  } else {
    DMField_DS    *dsfield = (DMField_DS *)field->data;
    PetscObject    faceDisc, cellDisc;
    PetscClassId   faceId, cellId;
    PetscDualSpace dsp;
    DM             K;
    DMPolytopeType ct;
    PetscInt(*co)[2][3];
    PetscInt        coneSize;
    PetscInt      **counts;
    PetscInt        f, i, o, q, s;
    PetscBool       found = PETSC_FALSE;
    const PetscInt *coneK;
    PetscInt        eStart, minOrient, maxOrient, numOrient;
    PetscInt       *orients;
    PetscReal     **orientPoints;
    PetscReal      *cellPoints;
    PetscReal      *dummyWeights;
    PetscQuadrature cellQuad = NULL;

    PetscCall(DMFieldDSGetHeightDisc(field, 1, dsfield->disc, &faceDisc));
    PetscCall(DMFieldDSGetHeightDisc(field, 0, dsfield->disc, &cellDisc));
    PetscCall(PetscObjectGetClassId(faceDisc, &faceId));
    PetscCall(PetscObjectGetClassId(cellDisc, &cellId));
    PetscCheck(faceId == PETSCFE_CLASSID && cellId == PETSCFE_CLASSID, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Not supported");
    PetscCall(PetscFEGetDualSpace((PetscFE)cellDisc, &dsp));
    PetscCall(PetscDualSpaceGetDM(dsp, &K));
    PetscCall(DMPlexGetHeightStratum(K, 1, &eStart, NULL));
    PetscCall(DMPlexGetCellType(K, eStart, &ct));
    PetscCall(DMPlexGetConeSize(K, 0, &coneSize));
    PetscCall(DMPlexGetCone(K, 0, &coneK));
    PetscCall(PetscMalloc2(numFaces, &co, coneSize, &counts));
    PetscCall(PetscMalloc1(dE * Nq, &cellPoints));
    PetscCall(PetscMalloc1(Nq, &dummyWeights));
    PetscCall(PetscQuadratureCreate(PETSC_COMM_SELF, &cellQuad));
    PetscCall(PetscQuadratureSetData(cellQuad, dE, 1, Nq, cellPoints, dummyWeights));
    minOrient = PETSC_INT_MAX;
    maxOrient = PETSC_INT_MIN;
    for (p = 0; p < numFaces; p++) { /* record the orientation of the facet wrt the support cells */
      PetscInt        point = points[p];
      PetscInt        numSupp, numChildren;
      const PetscInt *supp;

      PetscCall(DMPlexGetTreeChildren(dm, point, &numChildren, NULL));
      PetscCheck(!numChildren, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Face data not valid for facets with children");
      PetscCall(DMPlexGetSupportSize(dm, point, &numSupp));
      PetscCheck(numSupp <= 2, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %" PetscInt_FMT " has %" PetscInt_FMT " support, expected at most 2", point, numSupp);
      PetscCall(DMPlexGetSupport(dm, point, &supp));
      for (s = 0; s < numSupp; s++) {
        PetscInt        cell = supp[s];
        PetscInt        numCone;
        const PetscInt *cone, *orient;

        PetscCall(DMPlexGetConeSize(dm, cell, &numCone));
        // When we extract submeshes, we hang cells from the side that are not fully realized. We ignore these
        if (numCone == 1) {
          co[p][s][0] = -1;
          co[p][s][1] = -1;
          co[p][s][2] = -1;
          continue;
        }
        PetscCheck(numCone == coneSize, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Support point does not match reference element");
        PetscCall(DMPlexGetCone(dm, cell, &cone));
        PetscCall(DMPlexGetConeOrientation(dm, cell, &orient));
        for (f = 0; f < coneSize; f++) {
          if (cone[f] == point) break;
        }
        co[p][s][0] = f;
        co[p][s][1] = orient[f];
        co[p][s][2] = cell;
        minOrient   = PetscMin(minOrient, orient[f]);
        maxOrient   = PetscMax(maxOrient, orient[f]);
        found       = PETSC_TRUE;
      }
      for (; s < 2; s++) {
        co[p][s][0] = -1;
        co[p][s][1] = -1;
        co[p][s][2] = -1;
      }
    }
    numOrient = found ? maxOrient + 1 - minOrient : 0;
    PetscCall(DMPlexGetCone(K, 0, &coneK));
    /* count all (face,orientation) doubles that appear */
    PetscCall(PetscCalloc2(numOrient, &orients, numOrient, &orientPoints));
    for (f = 0; f < coneSize; f++) PetscCall(PetscCalloc1(numOrient + 1, &counts[f]));
    for (p = 0; p < numFaces; p++) {
      for (s = 0; s < 2; s++) {
        if (co[p][s][0] >= 0) {
          counts[co[p][s][0]][co[p][s][1] - minOrient]++;
          orients[co[p][s][1] - minOrient]++;
        }
      }
    }
    for (o = 0; o < numOrient; o++) {
      if (orients[o]) {
        PetscInt orient = o + minOrient;
        PetscInt q;

        PetscCall(PetscMalloc1(Nq * dim, &orientPoints[o]));
        /* rotate the quadrature points appropriately */
        switch (ct) {
        case DM_POLYTOPE_POINT:
          break;
        case DM_POLYTOPE_SEGMENT:
          if (orient == -2 || orient == 1) {
            for (q = 0; q < Nq; q++) orientPoints[o][q] = -geom->xi[q];
          } else {
            for (q = 0; q < Nq; q++) orientPoints[o][q] = geom->xi[q];
          }
          break;
        case DM_POLYTOPE_TRIANGLE:
          for (q = 0; q < Nq; q++) {
            PetscReal lambda[3];
            PetscReal lambdao[3];

            /* convert to barycentric */
            lambda[0] = -(geom->xi[2 * q] + geom->xi[2 * q + 1]) / 2.;
            lambda[1] = (geom->xi[2 * q] + 1.) / 2.;
            lambda[2] = (geom->xi[2 * q + 1] + 1.) / 2.;
            if (orient >= 0) {
              for (i = 0; i < 3; i++) lambdao[i] = lambda[(orient + i) % 3];
            } else {
              for (i = 0; i < 3; i++) lambdao[i] = lambda[(-(orient + i) + 3) % 3];
            }
            /* convert to coordinates */
            orientPoints[o][2 * q + 0] = -(lambdao[0] + lambdao[2]) + lambdao[1];
            orientPoints[o][2 * q + 1] = -(lambdao[0] + lambdao[1]) + lambdao[2];
          }
          break;
        case DM_POLYTOPE_QUADRILATERAL:
          for (q = 0; q < Nq; q++) {
            PetscReal xi[2], xio[2];
            PetscInt  oabs = (orient >= 0) ? orient : -(orient + 1);

            xi[0] = geom->xi[2 * q];
            xi[1] = geom->xi[2 * q + 1];
            switch (oabs) {
            case 1:
              xio[0] = xi[1];
              xio[1] = -xi[0];
              break;
            case 2:
              xio[0] = -xi[0];
              xio[1] = -xi[1];
              break;
            case 3:
              xio[0] = -xi[1];
              xio[1] = xi[0];
              break;
            case 0:
            default:
              xio[0] = xi[0];
              xio[1] = xi[1];
              break;
            }
            if (orient < 0) xio[0] = -xio[0];
            orientPoints[o][2 * q + 0] = xio[0];
            orientPoints[o][2 * q + 1] = xio[1];
          }
          break;
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Cell type %s not yet supported", DMPolytopeTypes[ct]);
        }
      }
    }
    for (f = 0; f < coneSize; f++) {
      PetscInt  face = coneK[f];
      PetscReal v0[3];
      PetscReal J[9], detJ;
      PetscInt  numCells, offset;
      PetscInt *cells;
      IS        suppIS;

      PetscCall(DMPlexComputeCellGeometryFEM(K, face, NULL, v0, J, NULL, &detJ));
      for (o = 0; o <= numOrient; o++) {
        PetscFEGeom *cellGeom;

        if (!counts[f][o]) continue;
        /* If this (face,orientation) double appears,
         * convert the face quadrature points into volume quadrature points */
        for (q = 0; q < Nq; q++) {
          PetscReal xi0[3] = {-1., -1., -1.};

          CoordinatesRefToReal(dE, dim, xi0, v0, J, &orientPoints[o][dim * q + 0], &cellPoints[dE * q + 0]);
        }
        for (p = 0, numCells = 0; p < numFaces; p++) {
          for (s = 0; s < 2; s++) {
            if (co[p][s][0] == f && co[p][s][1] == o + minOrient) numCells++;
          }
        }
        PetscCall(PetscMalloc1(numCells, &cells));
        for (p = 0, offset = 0; p < numFaces; p++) {
          for (s = 0; s < 2; s++) {
            if (co[p][s][0] == f && co[p][s][1] == o + minOrient) cells[offset++] = co[p][s][2];
          }
        }
        PetscCall(ISCreateGeneral(PETSC_COMM_SELF, numCells, cells, PETSC_USE_POINTER, &suppIS));
        PetscCall(DMFieldCreateFEGeom(field, suppIS, cellQuad, PETSC_FALSE, &cellGeom));
        for (p = 0, offset = 0; p < numFaces; p++) {
          for (s = 0; s < 2; s++) {
            if (co[p][s][0] == f && co[p][s][1] == o + minOrient) {
              for (q = 0; q < Nq * dE * dE; q++) {
                geom->suppJ[s][p * Nq * dE * dE + q]    = cellGeom->J[offset * Nq * dE * dE + q];
                geom->suppInvJ[s][p * Nq * dE * dE + q] = cellGeom->invJ[offset * Nq * dE * dE + q];
              }
              for (q = 0; q < Nq; q++) geom->suppDetJ[s][p * Nq + q] = cellGeom->detJ[offset * Nq + q];
              offset++;
            }
          }
        }
        PetscCall(PetscFEGeomDestroy(&cellGeom));
        PetscCall(ISDestroy(&suppIS));
        PetscCall(PetscFree(cells));
      }
    }
    for (o = 0; o < numOrient; o++) {
      if (orients[o]) PetscCall(PetscFree(orientPoints[o]));
    }
    PetscCall(PetscFree2(orients, orientPoints));
    PetscCall(PetscQuadratureDestroy(&cellQuad));
    for (f = 0; f < coneSize; f++) PetscCall(PetscFree(counts[f]));
    PetscCall(PetscFree2(co, counts));
  }
  PetscCall(ISRestoreIndices(pointIS, &points));
  PetscCall(ISDestroy(&cellIS));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMFieldInitialize_DS(DMField field)
{
  PetscFunctionBegin;
  field->ops->destroy                 = DMFieldDestroy_DS;
  field->ops->evaluate                = DMFieldEvaluate_DS;
  field->ops->evaluateFE              = DMFieldEvaluateFE_DS;
  field->ops->evaluateFV              = DMFieldEvaluateFV_DS;
  field->ops->getDegree               = DMFieldGetDegree_DS;
  field->ops->createDefaultQuadrature = DMFieldCreateDefaultQuadrature_DS;
  field->ops->view                    = DMFieldView_DS;
  field->ops->computeFaceData         = DMFieldComputeFaceData_DS;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode DMFieldCreate_DS(DMField field)
{
  DMField_DS *dsfield;

  PetscFunctionBegin;
  PetscCall(PetscNew(&dsfield));
  field->data = dsfield;
  PetscCall(DMFieldInitialize_DS(field));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMFieldCreateDSWithDG(DM dm, DM dmDG, PetscInt fieldNum, Vec vec, Vec vecDG, DMField *field)
{
  DMField      b;
  DMField_DS  *dsfield;
  PetscObject  disc = NULL, discDG = NULL;
  PetscSection section;
  PetscBool    isContainer   = PETSC_FALSE;
  PetscClassId id            = -1;
  PetscInt     numComponents = -1, dsNumFields;

  PetscFunctionBegin;
  PetscCall(DMGetLocalSection(dm, &section));
  PetscCall(PetscSectionGetFieldComponents(section, fieldNum, &numComponents));
  PetscCall(DMGetNumFields(dm, &dsNumFields));
  if (dsNumFields) PetscCall(DMGetField(dm, fieldNum, NULL, &disc));
  if (dsNumFields && dmDG) {
    PetscCall(DMGetField(dmDG, fieldNum, NULL, &discDG));
    PetscCall(PetscObjectReference(discDG));
  }
  if (disc) {
    PetscCall(PetscObjectGetClassId(disc, &id));
    isContainer = (id == PETSC_CONTAINER_CLASSID) ? PETSC_TRUE : PETSC_FALSE;
  }
  if (!disc || isContainer) {
    MPI_Comm       comm = PetscObjectComm((PetscObject)dm);
    PetscFE        fe;
    DMPolytopeType ct, locct = DM_POLYTOPE_UNKNOWN;
    PetscInt       dim, cStart, cEnd, cellHeight;

    PetscCall(DMPlexGetVTKCellHeight(dm, &cellHeight));
    PetscCall(DMGetDimension(dm, &dim));
    PetscCall(DMPlexGetHeightStratum(dm, cellHeight, &cStart, &cEnd));
    if (cEnd > cStart) PetscCall(DMPlexGetCellType(dm, cStart, &locct));
    PetscCallMPI(MPIU_Allreduce(&locct, &ct, 1, MPI_INT, MPI_MIN, comm));
    PetscCall(PetscFECreateLagrangeByCell(PETSC_COMM_SELF, dim, numComponents, ct, 1, PETSC_DETERMINE, &fe));
    PetscCall(PetscFEViewFromOptions(fe, NULL, "-field_fe_view"));
    disc = (PetscObject)fe;
  } else PetscCall(PetscObjectReference(disc));
  PetscCall(PetscObjectGetClassId(disc, &id));
  if (id == PETSCFE_CLASSID) PetscCall(PetscFEGetNumComponents((PetscFE)disc, &numComponents));
  else SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Cannot determine number of discretization components");
  PetscCall(DMFieldCreate(dm, numComponents, DMFIELD_VERTEX, &b));
  PetscCall(DMFieldSetType(b, DMFIELDDS));
  dsfield           = (DMField_DS *)b->data;
  dsfield->fieldNum = fieldNum;
  PetscCall(DMGetDimension(dm, &dsfield->height));
  dsfield->height++;
  PetscCall(PetscCalloc1(dsfield->height, &dsfield->disc));
  dsfield->disc[0] = disc;
  PetscCall(PetscObjectReference((PetscObject)vec));
  dsfield->vec = vec;
  if (dmDG) {
    dsfield->dmDG = dmDG;
    PetscCall(PetscCalloc1(dsfield->height, &dsfield->discDG));
    dsfield->discDG[0] = discDG;
    PetscCall(PetscObjectReference((PetscObject)vecDG));
    dsfield->vecDG = vecDG;
  }
  *field = b;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMFieldCreateDS(DM dm, PetscInt fieldNum, Vec vec, DMField *field)
{
  PetscFunctionBegin;
  PetscCall(DMFieldCreateDSWithDG(dm, NULL, fieldNum, vec, NULL, field));
  PetscFunctionReturn(PETSC_SUCCESS);
}
