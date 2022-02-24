#include <petsc/private/dmfieldimpl.h> /*I "petscdmfield.h" I*/
#include <petsc/private/petscfeimpl.h> /*I "petscdmfield.h" I*/
#include <petscfe.h>
#include <petscdmplex.h>
#include <petscds.h>

typedef struct _n_DMField_DS
{
  PetscInt    fieldNum;
  Vec         vec;
  PetscInt    height;
  PetscObject *disc;
  PetscBool   multifieldVec;
}
DMField_DS;

static PetscErrorCode DMFieldDestroy_DS(DMField field)
{
  DMField_DS     *dsfield;
  PetscInt       i;

  PetscFunctionBegin;
  dsfield = (DMField_DS *) field->data;
  CHKERRQ(VecDestroy(&dsfield->vec));
  for (i = 0; i < dsfield->height; i++) {
    CHKERRQ(PetscObjectDereference(dsfield->disc[i]));
  }
  CHKERRQ(PetscFree(dsfield->disc));
  CHKERRQ(PetscFree(dsfield));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMFieldView_DS(DMField field,PetscViewer viewer)
{
  DMField_DS     *dsfield = (DMField_DS *) field->data;
  PetscBool      iascii;
  PetscObject    disc;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  disc = dsfield->disc[0];
  if (iascii) {
    CHKERRQ(PetscViewerASCIIPrintf(viewer, "PetscDS field %D\n",dsfield->fieldNum));
    CHKERRQ(PetscViewerASCIIPushTab(viewer));
    CHKERRQ(PetscObjectView(disc,viewer));
    CHKERRQ(PetscViewerASCIIPopTab(viewer));
  }
  CHKERRQ(PetscViewerASCIIPushTab(viewer));
  PetscCheckFalse(dsfield->multifieldVec,PetscObjectComm((PetscObject)field),PETSC_ERR_SUP,"View of subfield not implemented yet");
  CHKERRQ(VecView(dsfield->vec,viewer));
  CHKERRQ(PetscViewerASCIIPopTab(viewer));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMFieldDSGetHeightDisc(DMField field, PetscInt height, PetscObject *disc)
{
  DMField_DS     *dsfield = (DMField_DS *) field->data;

  PetscFunctionBegin;
  if (!dsfield->disc[height]) {
    PetscClassId   id;

    CHKERRQ(PetscObjectGetClassId(dsfield->disc[0],&id));
    if (id == PETSCFE_CLASSID) {
      PetscFE fe = (PetscFE) dsfield->disc[0];

      CHKERRQ(PetscFECreateHeightTrace(fe,height,(PetscFE *)&dsfield->disc[height]));
    }
  }
  *disc = dsfield->disc[height];
  PetscFunctionReturn(0);
}

/* y[m,c] = A[m,n,c] . b[n] */
#define DMFieldDSdot(y,A,b,m,n,c,cast)                                           \
  do {                                                                           \
    PetscInt _i, _j, _k;                                                         \
    for (_i = 0; _i < (m); _i++) {                                               \
      for (_k = 0; _k < (c); _k++) {                                             \
        (y)[_i * (c) + _k] = 0.;                                                 \
      }                                                                          \
      for (_j = 0; _j < (n); _j++) {                                             \
        for (_k = 0; _k < (c); _k++) {                                           \
          (y)[_i * (c) + _k] += (A)[(_i * (n) + _j) * (c) + _k] * cast((b)[_j]); \
        }                                                                        \
      }                                                                          \
    }                                                                            \
  } while (0)

/* TODO: Reorganize interface so that I can reuse a tabulation rather than mallocing each time */
static PetscErrorCode DMFieldEvaluateFE_DS(DMField field, IS pointIS, PetscQuadrature quad, PetscDataType type, void *B, void *D, void *H)
{
  DMField_DS      *dsfield = (DMField_DS *) field->data;
  DM              dm;
  PetscObject     disc;
  PetscClassId    classid;
  PetscInt        nq, nc, dim, meshDim, numCells;
  PetscSection    section;
  const PetscReal *qpoints;
  PetscBool       isStride;
  const PetscInt  *points = NULL;
  PetscInt        sfirst = -1, stride = -1;

  PetscFunctionBeginHot;
  dm   = field->dm;
  nc   = field->numComponents;
  CHKERRQ(PetscQuadratureGetData(quad,&dim,NULL,&nq,&qpoints,NULL));
  CHKERRQ(DMFieldDSGetHeightDisc(field,dsfield->height - 1 - dim,&disc));
  CHKERRQ(DMGetDimension(dm,&meshDim));
  CHKERRQ(DMGetLocalSection(dm,&section));
  CHKERRQ(PetscSectionGetField(section,dsfield->fieldNum,&section));
  CHKERRQ(PetscObjectGetClassId(disc,&classid));
  /* TODO: batch */
  CHKERRQ(PetscObjectTypeCompare((PetscObject)pointIS,ISSTRIDE,&isStride));
  CHKERRQ(ISGetLocalSize(pointIS,&numCells));
  if (isStride) {
    CHKERRQ(ISStrideGetInfo(pointIS,&sfirst,&stride));
  } else {
    CHKERRQ(ISGetIndices(pointIS,&points));
  }
  if (classid == PETSCFE_CLASSID) {
    PetscFE      fe = (PetscFE) disc;
    PetscInt     feDim, i;
    PetscInt          K = H ? 2 : (D ? 1 : (B ? 0 : -1));
    PetscTabulation   T;

    CHKERRQ(PetscFEGetDimension(fe,&feDim));
    CHKERRQ(PetscFECreateTabulation(fe,1,nq,qpoints,K,&T));
    for (i = 0; i < numCells; i++) {
      PetscInt     c = isStride ? (sfirst + i * stride) : points[i];
      PetscInt     closureSize;
      PetscScalar *elem = NULL;

      CHKERRQ(DMPlexVecGetClosure(dm,section,dsfield->vec,c,&closureSize,&elem));
      if (B) {
        /* field[c] = T[q,b,c] . coef[b], so v[c] = T[q,b,c] . coords[b] */
        if (type == PETSC_SCALAR) {
          PetscScalar *cB = &((PetscScalar *) B)[nc * nq * i];

          DMFieldDSdot(cB,T->T[0],elem,nq,feDim,nc,(PetscScalar));
        } else {
          PetscReal *cB = &((PetscReal *) B)[nc * nq * i];

          DMFieldDSdot(cB,T->T[0],elem,nq,feDim,nc,PetscRealPart);
        }
      }
      if (D) {
        if (type == PETSC_SCALAR) {
          PetscScalar *cD = &((PetscScalar *) D)[nc * nq * dim * i];

          DMFieldDSdot(cD,T->T[1],elem,nq,feDim,(nc * dim),(PetscScalar));
        } else {
          PetscReal *cD = &((PetscReal *) D)[nc * nq * dim * i];

          DMFieldDSdot(cD,T->T[1],elem,nq,feDim,(nc * dim),PetscRealPart);
        }
      }
      if (H) {
        if (type == PETSC_SCALAR) {
          PetscScalar *cH = &((PetscScalar *) H)[nc * nq * dim * dim * i];

          DMFieldDSdot(cH,T->T[2],elem,nq,feDim,(nc * dim * dim),(PetscScalar));
        } else {
          PetscReal *cH = &((PetscReal *) H)[nc * nq * dim * dim * i];

          DMFieldDSdot(cH,T->T[2],elem,nq,feDim,(nc * dim * dim),PetscRealPart);
        }
      }
      CHKERRQ(DMPlexVecRestoreClosure(dm,section,dsfield->vec,c,&closureSize,&elem));
    }
    CHKERRQ(PetscTabulationDestroy(&T));
  } else SETERRQ(PetscObjectComm((PetscObject)field),PETSC_ERR_SUP,"Not implemented");
  if (!isStride) {
    CHKERRQ(ISRestoreIndices(pointIS,&points));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMFieldEvaluate_DS(DMField field, Vec points, PetscDataType datatype, void *B, void *D, void *H)
{
  DMField_DS        *dsfield = (DMField_DS *) field->data;
  PetscSF            cellSF = NULL;
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
  nc   = field->numComponents;
  CHKERRQ(DMGetLocalSection(field->dm,&section));
  CHKERRQ(DMFieldDSGetHeightDisc(field,0,&cellDisc));
  CHKERRQ(PetscObjectGetClassId(cellDisc, &discID));
  PetscCheckFalse(discID != PETSCFE_CLASSID,PETSC_COMM_SELF,PETSC_ERR_PLIB, "Discretization type not supported");
  cellFE = (PetscFE) cellDisc;
  CHKERRQ(PetscFEGetDimension(cellFE,&feDim));
  CHKERRQ(DMGetCoordinateDim(field->dm, &dim));
  CHKERRQ(DMGetDimension(field->dm, &dimR));
  CHKERRQ(DMLocatePoints(field->dm, points, DM_POINTLOCATION_NONE, &cellSF));
  CHKERRQ(PetscSFGetGraph(cellSF, &numCells, &nFound, NULL, &cells));
  for (c = 0; c < nFound; c++) {
    PetscCheckFalse(cells[c].index < 0,PetscObjectComm((PetscObject)points),PETSC_ERR_ARG_WRONG, "Point %D could not be located", c);
  }
  CHKERRQ(PetscSFComputeDegreeBegin(cellSF,&cellDegrees));
  CHKERRQ(PetscSFComputeDegreeEnd(cellSF,&cellDegrees));
  for (c = 0, gatherSize = 0, gatherMax = 0; c < numCells; c++) {
    gatherMax = PetscMax(gatherMax,cellDegrees[c]);
    gatherSize += cellDegrees[c];
  }
  CHKERRQ(PetscMalloc3(gatherSize*dim,&cellPoints,gatherMax*dim,&coordsReal,gatherMax*dimR,&coordsRef));
  CHKERRQ(PetscMalloc4(gatherMax*dimR,&v,gatherMax*dimR*dimR,&J,gatherMax*dimR*dimR,&invJ,gatherMax,&detJ));
  if (datatype == PETSC_SCALAR) {
    CHKERRQ(PetscMalloc3(B ? nc * gatherSize : 0, &cellBs, D ? nc * dim * gatherSize : 0, &cellDs, H ? nc * dim * dim * gatherSize : 0, &cellHs));
  } else {
    CHKERRQ(PetscMalloc3(B ? nc * gatherSize : 0, &cellBr, D ? nc * dim * gatherSize : 0, &cellDr, H ? nc * dim * dim * gatherSize : 0, &cellHr));
  }

  CHKERRMPI(MPI_Type_contiguous(dim,MPIU_SCALAR,&pointType));
  CHKERRMPI(MPI_Type_commit(&pointType));
  CHKERRQ(VecGetArrayRead(points,&pointsArray));
  CHKERRQ(PetscSFGatherBegin(cellSF, pointType, pointsArray, cellPoints));
  CHKERRQ(PetscSFGatherEnd(cellSF, pointType, pointsArray, cellPoints));
  CHKERRQ(VecRestoreArrayRead(points,&pointsArray));
  for (c = 0, offset = 0; c < numCells; c++) {
    PetscInt nq = cellDegrees[c], p;

    if (nq) {
      PetscInt          K = H ? 2 : (D ? 1 : (B ? 0 : -1));
      PetscTabulation   T;
      PetscInt     closureSize;
      PetscScalar *elem = NULL;
      PetscReal   *quadPoints;
      PetscQuadrature quad;
      PetscInt d, e, f, g;

      for (p = 0; p < dim * nq; p++) coordsReal[p] = PetscRealPart(cellPoints[dim * offset + p]);
      CHKERRQ(DMPlexCoordinatesToReference(field->dm, c, nq, coordsReal, coordsRef));
      CHKERRQ(PetscFECreateTabulation(cellFE,1,nq,coordsRef,K,&T));
      CHKERRQ(PetscQuadratureCreate(PETSC_COMM_SELF, &quad));
      CHKERRQ(PetscMalloc1(dimR * nq, &quadPoints));
      for (p = 0; p < dimR * nq; p++) quadPoints[p] = coordsRef[p];
      CHKERRQ(PetscQuadratureSetData(quad, dimR, 0, nq, quadPoints, NULL));
      CHKERRQ(DMPlexComputeCellGeometryFEM(field->dm, c, quad, v, J, invJ, detJ));
      CHKERRQ(PetscQuadratureDestroy(&quad));
      CHKERRQ(DMPlexVecGetClosure(field->dm,section,dsfield->vec,c,&closureSize,&elem));
      if (B) {
        if (datatype == PETSC_SCALAR) {
          PetscScalar *cB = &cellBs[nc * offset];

          DMFieldDSdot(cB,T->T[0],elem,nq,feDim,nc,(PetscScalar));
        } else {
          PetscReal *cB = &cellBr[nc * offset];

          DMFieldDSdot(cB,T->T[0],elem,nq,feDim,nc,PetscRealPart);
        }
      }
      if (D) {
        if (datatype == PETSC_SCALAR) {
          PetscScalar *cD = &cellDs[nc * dim * offset];

          DMFieldDSdot(cD,T->T[1],elem,nq,feDim,(nc * dim),(PetscScalar));
          for (p = 0; p < nq; p++) {
            for (g = 0; g < nc; g++) {
              PetscScalar vs[3];

              for (d = 0; d < dimR; d++) {
                vs[d] = 0.;
                for (e = 0; e < dimR; e++) {
                  vs[d] += invJ[dimR * dimR * p + e * dimR + d] * cD[(nc * p + g) * dimR + e];
                }
              }
              for (d = 0; d < dimR; d++) {
                cD[(nc * p + g) * dimR + d] = vs[d];
              }
            }
          }
        } else {
          PetscReal *cD = &cellDr[nc * dim * offset];

          DMFieldDSdot(cD,T->T[1],elem,nq,feDim,(nc * dim),PetscRealPart);
          for (p = 0; p < nq; p++) {
            for (g = 0; g < nc; g++) {
              for (d = 0; d < dimR; d++) {
                v[d] = 0.;
                for (e = 0; e < dimR; e++) {
                  v[d] += invJ[dimR * dimR * p + e * dimR + d] * cD[(nc * p + g) * dimR + e];
                }
              }
              for (d = 0; d < dimR; d++) {
                cD[(nc * p + g) * dimR + d] = v[d];
              }
            }
          }
        }
      }
      if (H) {
        if (datatype == PETSC_SCALAR) {
          PetscScalar *cH = &cellHs[nc * dim * dim * offset];

          DMFieldDSdot(cH,T->T[2],elem,nq,feDim,(nc * dim * dim),(PetscScalar));
          for (p = 0; p < nq; p++) {
            for (g = 0; g < nc * dimR; g++) {
              PetscScalar vs[3];

              for (d = 0; d < dimR; d++) {
                vs[d] = 0.;
                for (e = 0; e < dimR; e++) {
                  vs[d] += invJ[dimR * dimR * p + e * dimR + d] * cH[(nc * dimR * p + g) * dimR + e];
                }
              }
              for (d = 0; d < dimR; d++) {
                cH[(nc * dimR * p + g) * dimR + d] = vs[d];
              }
            }
            for (g = 0; g < nc; g++) {
              for (f = 0; f < dimR; f++) {
                PetscScalar vs[3];

                for (d = 0; d < dimR; d++) {
                  vs[d] = 0.;
                  for (e = 0; e < dimR; e++) {
                    vs[d] += invJ[dimR * dimR * p + e * dimR + d] * cH[((nc * p + g) * dimR + e) * dimR + f];
                  }
                }
                for (d = 0; d < dimR; d++) {
                  cH[((nc * p + g) * dimR + d) * dimR + f] = vs[d];
                }
              }
            }
          }
        } else {
          PetscReal *cH = &cellHr[nc * dim * dim * offset];

          DMFieldDSdot(cH,T->T[2],elem,nq,feDim,(nc * dim * dim),PetscRealPart);
          for (p = 0; p < nq; p++) {
            for (g = 0; g < nc * dimR; g++) {
              for (d = 0; d < dimR; d++) {
                v[d] = 0.;
                for (e = 0; e < dimR; e++) {
                  v[d] += invJ[dimR * dimR * p + e * dimR + d] * cH[(nc * dimR * p + g) * dimR + e];
                }
              }
              for (d = 0; d < dimR; d++) {
                cH[(nc * dimR * p + g) * dimR + d] = v[d];
              }
            }
            for (g = 0; g < nc; g++) {
              for (f = 0; f < dimR; f++) {
                for (d = 0; d < dimR; d++) {
                  v[d] = 0.;
                  for (e = 0; e < dimR; e++) {
                    v[d] += invJ[dimR * dimR * p + e * dimR + d] * cH[((nc * p + g) * dimR + e) * dimR + f];
                  }
                }
                for (d = 0; d < dimR; d++) {
                  cH[((nc * p + g) * dimR + d) * dimR + f] = v[d];
                }
              }
            }
          }
        }
      }
      CHKERRQ(DMPlexVecRestoreClosure(field->dm,section,dsfield->vec,c,&closureSize,&elem));
      CHKERRQ(PetscTabulationDestroy(&T));
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

      CHKERRMPI(MPI_Type_contiguous(nc, origtype, &Btype));
      CHKERRMPI(MPI_Type_commit(&Btype));
      CHKERRQ(PetscSFScatterBegin(cellSF,Btype,(datatype == PETSC_SCALAR) ? (void *) cellBs : (void *) cellBr, B));
      CHKERRQ(PetscSFScatterEnd(cellSF,Btype,(datatype == PETSC_SCALAR) ? (void *) cellBs : (void *) cellBr, B));
      CHKERRMPI(MPI_Type_free(&Btype));
    }
    if (D) {
      MPI_Datatype Dtype;

      CHKERRMPI(MPI_Type_contiguous(nc * dim, origtype, &Dtype));
      CHKERRMPI(MPI_Type_commit(&Dtype));
      CHKERRQ(PetscSFScatterBegin(cellSF,Dtype,(datatype == PETSC_SCALAR) ? (void *) cellDs : (void *) cellDr, D));
      CHKERRQ(PetscSFScatterEnd(cellSF,Dtype,(datatype == PETSC_SCALAR) ? (void *) cellDs : (void *) cellDr, D));
      CHKERRMPI(MPI_Type_free(&Dtype));
    }
    if (H) {
      MPI_Datatype Htype;

      CHKERRMPI(MPI_Type_contiguous(nc * dim * dim, origtype, &Htype));
      CHKERRMPI(MPI_Type_commit(&Htype));
      CHKERRQ(PetscSFScatterBegin(cellSF,Htype,(datatype == PETSC_SCALAR) ? (void *) cellHs : (void *) cellHr, H));
      CHKERRQ(PetscSFScatterEnd(cellSF,Htype,(datatype == PETSC_SCALAR) ? (void *) cellHs : (void *) cellHr, H));
      CHKERRMPI(MPI_Type_free(&Htype));
    }
  }
  CHKERRQ(PetscFree4(v,J,invJ,detJ));
  CHKERRQ(PetscFree3(cellBr, cellDr, cellHr));
  CHKERRQ(PetscFree3(cellBs, cellDs, cellHs));
  CHKERRQ(PetscFree3(cellPoints,coordsReal,coordsRef));
  CHKERRMPI(MPI_Type_free(&pointType));
  CHKERRQ(PetscSFDestroy(&cellSF));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMFieldEvaluateFV_DS(DMField field, IS pointIS, PetscDataType type, void *B, void *D, void *H)
{
  DMField_DS      *dsfield = (DMField_DS *) field->data;
  PetscInt         h, imin;
  PetscInt         dim;
  PetscClassId     id;
  PetscQuadrature  quad = NULL;
  PetscInt         maxDegree;
  PetscFEGeom      *geom;
  PetscInt         Nq, Nc, dimC, qNc, N;
  PetscInt         numPoints;
  void            *qB = NULL, *qD = NULL, *qH = NULL;
  const PetscReal *weights;
  MPI_Datatype     mpitype = type == PETSC_SCALAR ? MPIU_SCALAR : MPIU_REAL;
  PetscObject      disc;
  DMField          coordField;

  PetscFunctionBegin;
  Nc = field->numComponents;
  CHKERRQ(DMGetCoordinateDim(field->dm, &dimC));
  CHKERRQ(DMGetDimension(field->dm, &dim));
  CHKERRQ(ISGetLocalSize(pointIS, &numPoints));
  CHKERRQ(ISGetMinMax(pointIS,&imin,NULL));
  for (h = 0; h < dsfield->height; h++) {
    PetscInt hEnd;

    CHKERRQ(DMPlexGetHeightStratum(field->dm,h,NULL,&hEnd));
    if (imin < hEnd) break;
  }
  dim -= h;
  CHKERRQ(DMFieldDSGetHeightDisc(field,h,&disc));
  CHKERRQ(PetscObjectGetClassId(disc,&id));
  PetscCheckFalse(id != PETSCFE_CLASSID,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Discretization not supported");
  CHKERRQ(DMGetCoordinateField(field->dm, &coordField));
  CHKERRQ(DMFieldGetDegree(coordField, pointIS, NULL, &maxDegree));
  if (maxDegree <= 1) {
    CHKERRQ(DMFieldCreateDefaultQuadrature(coordField, pointIS, &quad));
  }
  if (!quad) CHKERRQ(DMFieldCreateDefaultQuadrature(field, pointIS, &quad));
  PetscCheckFalse(!quad,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Could not determine quadrature for cell averages");
  CHKERRQ(DMFieldCreateFEGeom(coordField,pointIS,quad,PETSC_FALSE,&geom));
  CHKERRQ(PetscQuadratureGetData(quad, NULL, &qNc, &Nq, NULL, &weights));
  PetscCheckFalse(qNc != 1,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Expected scalar quadrature components");
  N = numPoints * Nq * Nc;
  if (B) CHKERRQ(DMGetWorkArray(field->dm, N, mpitype, &qB));
  if (D) CHKERRQ(DMGetWorkArray(field->dm, N * dimC, mpitype, &qD));
  if (H) CHKERRQ(DMGetWorkArray(field->dm, N * dimC * dimC, mpitype, &qH));
  CHKERRQ(DMFieldEvaluateFE(field,pointIS,quad,type,qB,qD,qH));
  if (B) {
    PetscInt i, j, k;

    if (type == PETSC_SCALAR) {
      PetscScalar * sB  = (PetscScalar *) B;
      PetscScalar * sqB = (PetscScalar *) qB;

      for (i = 0; i < numPoints; i++) {
        PetscReal vol = 0.;

        for (j = 0; j < Nc; j++) {sB[i * Nc + j] = 0.;}
        for (k = 0; k < Nq; k++) {
          vol += geom->detJ[i * Nq + k] * weights[k];
          for (j = 0; j < Nc; j++) {
            sB[i * Nc + j] += geom->detJ[i * Nq + k] * weights[k] * sqB[ (i * Nq + k) * Nc + j];
          }
        }
        for (k = 0; k < Nc; k++) sB[i * Nc + k] /= vol;
      }
    } else {
      PetscReal * rB  = (PetscReal *) B;
      PetscReal * rqB = (PetscReal *) qB;

      for (i = 0; i < numPoints; i++) {
        PetscReal vol = 0.;

        for (j = 0; j < Nc; j++) {rB[i * Nc + j] = 0.;}
        for (k = 0; k < Nq; k++) {
          vol += geom->detJ[i * Nq + k] * weights[k];
          for (j = 0; j < Nc; j++) {
            rB[i * Nc + j] += weights[k] * rqB[ (i * Nq + k) * Nc + j];
          }
        }
        for (k = 0; k < Nc; k++) rB[i * Nc + k] /= vol;
      }
    }
  }
  if (D) {
    PetscInt i, j, k, l, m;

    if (type == PETSC_SCALAR) {
      PetscScalar * sD  = (PetscScalar *) D;
      PetscScalar * sqD = (PetscScalar *) qD;

      for (i = 0; i < numPoints; i++) {
        PetscReal vol = 0.;

        for (j = 0; j < Nc * dimC; j++) {sD[i * Nc * dimC + j] = 0.;}
        for (k = 0; k < Nq; k++) {
          vol += geom->detJ[i * Nq + k] * weights[k];
          for (j = 0; j < Nc; j++) {
            PetscScalar pD[3] = {0.,0.,0.};

            for (l = 0; l < dimC; l++) {
              for (m = 0; m < dim; m++) {
                pD[l] += geom->invJ[((i * Nq + k) * dimC + m) * dimC + l] * sqD[((i * Nq + k) * Nc + j) * dim + m];
              }
            }
            for (l = 0; l < dimC; l++) {
              sD[(i * Nc + j) * dimC + l] += geom->detJ[i * Nq + k] * weights[k] * pD[l];
            }
          }
        }
        for (k = 0; k < Nc * dimC; k++) sD[i * Nc * dimC + k] /= vol;
      }
    } else {
      PetscReal * rD  = (PetscReal *) D;
      PetscReal * rqD = (PetscReal *) qD;

      for (i = 0; i < numPoints; i++) {
        PetscReal vol = 0.;

        for (j = 0; j < Nc * dimC; j++) {rD[i * Nc * dimC + j] = 0.;}
        for (k = 0; k < Nq; k++) {
          vol += geom->detJ[i * Nq + k] * weights[k];
          for (j = 0; j < Nc; j++) {
            PetscReal pD[3] = {0.,0.,0.};

            for (l = 0; l < dimC; l++) {
              for (m = 0; m < dim; m++) {
                pD[l] += geom->invJ[((i * Nq + k) * dimC + m) * dimC + l] * rqD[((i * Nq + k) * Nc + j) * dim + m];
              }
            }
            for (l = 0; l < dimC; l++) {
              rD[(i * Nc + j) * dimC + l] += geom->detJ[i * Nq + k] * weights[k] * pD[l];
            }
          }
        }
        for (k = 0; k < Nc * dimC; k++) rD[i * Nc * dimC + k] /= vol;
      }
    }
  }
  if (H) {
    PetscInt i, j, k, l, m, q, r;

    if (type == PETSC_SCALAR) {
      PetscScalar * sH  = (PetscScalar *) H;
      PetscScalar * sqH = (PetscScalar *) qH;

      for (i = 0; i < numPoints; i++) {
        PetscReal vol = 0.;

        for (j = 0; j < Nc * dimC * dimC; j++) {sH[i * Nc * dimC * dimC + j] = 0.;}
        for (k = 0; k < Nq; k++) {
          const PetscReal *invJ = &geom->invJ[(i * Nq + k) * dimC * dimC];

          vol += geom->detJ[i * Nq + k] * weights[k];
          for (j = 0; j < Nc; j++) {
            PetscScalar pH[3][3] = {{0.,0.,0.},{0.,0.,0.},{0.,0.,0.}};
            const PetscScalar *spH = &sqH[((i * Nq + k) * Nc + j) * dimC * dimC];

            for (l = 0; l < dimC; l++) {
              for (m = 0; m < dimC; m++) {
                for (q = 0; q < dim; q++) {
                  for (r = 0; r < dim; r++) {
                    pH[l][m] += invJ[q * dimC + l] * invJ[r * dimC + m] * spH[q * dim + r];
                  }
                }
              }
            }
            for (l = 0; l < dimC; l++) {
              for (m = 0; m < dimC; m++) {
                sH[(i * Nc + j) * dimC * dimC + l * dimC + m] += geom->detJ[i * Nq + k] * weights[k] * pH[l][m];
              }
            }
          }
        }
        for (k = 0; k < Nc * dimC * dimC; k++) sH[i * Nc * dimC * dimC + k] /= vol;
      }
    } else {
      PetscReal * rH  = (PetscReal *) H;
      PetscReal * rqH = (PetscReal *) qH;

      for (i = 0; i < numPoints; i++) {
        PetscReal vol = 0.;

        for (j = 0; j < Nc * dimC * dimC; j++) {rH[i * Nc * dimC * dimC + j] = 0.;}
        for (k = 0; k < Nq; k++) {
          const PetscReal *invJ = &geom->invJ[(i * Nq + k) * dimC * dimC];

          vol += geom->detJ[i * Nq + k] * weights[k];
          for (j = 0; j < Nc; j++) {
            PetscReal pH[3][3] = {{0.,0.,0.},{0.,0.,0.},{0.,0.,0.}};
            const PetscReal *rpH = &rqH[((i * Nq + k) * Nc + j) * dimC * dimC];

            for (l = 0; l < dimC; l++) {
              for (m = 0; m < dimC; m++) {
                for (q = 0; q < dim; q++) {
                  for (r = 0; r < dim; r++) {
                    pH[l][m] += invJ[q * dimC + l] * invJ[r * dimC + m] * rpH[q * dim + r];
                  }
                }
              }
            }
            for (l = 0; l < dimC; l++) {
              for (m = 0; m < dimC; m++) {
                rH[(i * Nc + j) * dimC * dimC + l * dimC + m] += geom->detJ[i * Nq + k] * weights[k] * pH[l][m];
              }
            }
          }
        }
        for (k = 0; k < Nc * dimC * dimC; k++) rH[i * Nc * dimC * dimC + k] /= vol;
      }
    }
  }
  if (B) CHKERRQ(DMRestoreWorkArray(field->dm, N, mpitype, &qB));
  if (D) CHKERRQ(DMRestoreWorkArray(field->dm, N * dimC, mpitype, &qD));
  if (H) CHKERRQ(DMRestoreWorkArray(field->dm, N * dimC * dimC, mpitype, &qH));
  CHKERRQ(PetscFEGeomDestroy(&geom));
  CHKERRQ(PetscQuadratureDestroy(&quad));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMFieldGetDegree_DS(DMField field, IS pointIS, PetscInt *minDegree, PetscInt *maxDegree)
{
  DMField_DS     *dsfield;
  PetscObject    disc;
  PetscInt       h, imin, imax;
  PetscClassId   id;

  PetscFunctionBegin;
  dsfield = (DMField_DS *) field->data;
  CHKERRQ(ISGetMinMax(pointIS,&imin,&imax));
  if (imin >= imax) {
    h = 0;
  } else {
    for (h = 0; h < dsfield->height; h++) {
      PetscInt hEnd;

      CHKERRQ(DMPlexGetHeightStratum(field->dm,h,NULL,&hEnd));
      if (imin < hEnd) break;
    }
  }
  CHKERRQ(DMFieldDSGetHeightDisc(field,h,&disc));
  CHKERRQ(PetscObjectGetClassId(disc,&id));
  if (id == PETSCFE_CLASSID) {
    PetscFE    fe = (PetscFE) disc;
    PetscSpace sp;

    CHKERRQ(PetscFEGetBasisSpace(fe, &sp));
    CHKERRQ(PetscSpaceGetDegree(sp, minDegree, maxDegree));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMFieldGetFVQuadrature_Internal(DMField field, IS pointIS, PetscQuadrature *quad)
{
  DM              dm = field->dm;
  const PetscInt *points;
  DMPolytopeType  ct;
  PetscInt        dim, n;
  PetscBool       isplex;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject) dm, DMPLEX, &isplex));
  CHKERRQ(ISGetLocalSize(pointIS, &n));
  if (isplex && n) {
    CHKERRQ(DMGetDimension(dm, &dim));
    CHKERRQ(ISGetIndices(pointIS, &points));
    CHKERRQ(DMPlexGetCellType(dm, points[0], &ct));
    switch (ct) {
      case DM_POLYTOPE_TRIANGLE:
      case DM_POLYTOPE_TETRAHEDRON:
        CHKERRQ(PetscDTStroudConicalQuadrature(dim, 1, 1, -1.0, 1.0, quad));break;
      default: CHKERRQ(PetscDTGaussTensorQuadrature(dim, 1, 1, -1.0, 1.0, quad));
    }
    CHKERRQ(ISRestoreIndices(pointIS, &points));
  } else {
    CHKERRQ(DMFieldCreateDefaultQuadrature(field, pointIS, quad));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMFieldCreateDefaultQuadrature_DS(DMField field, IS pointIS, PetscQuadrature *quad)
{
  PetscInt       h, dim, imax, imin, cellHeight;
  DM             dm;
  DMField_DS     *dsfield;
  PetscObject    disc;
  PetscFE        fe;
  PetscClassId   id;

  PetscFunctionBegin;
  dm = field->dm;
  dsfield = (DMField_DS *) field->data;
  CHKERRQ(ISGetMinMax(pointIS,&imin,&imax));
  CHKERRQ(DMGetDimension(dm,&dim));
  for (h = 0; h <= dim; h++) {
    PetscInt hStart, hEnd;

    CHKERRQ(DMPlexGetHeightStratum(dm,h,&hStart,&hEnd));
    if (imax >= hStart && imin < hEnd) break;
  }
  CHKERRQ(DMPlexGetVTKCellHeight(dm, &cellHeight));
  h -= cellHeight;
  *quad = NULL;
  if (h < dsfield->height) {
    CHKERRQ(DMFieldDSGetHeightDisc(field,h,&disc));
    CHKERRQ(PetscObjectGetClassId(disc,&id));
    if (id != PETSCFE_CLASSID) PetscFunctionReturn(0);
    fe = (PetscFE) disc;
    CHKERRQ(PetscFEGetQuadrature(fe,quad));
    CHKERRQ(PetscObjectReference((PetscObject)*quad));
  }
  PetscFunctionReturn(0);
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
  CHKERRQ(DMPlexGetDepthLabel(dm, &depthLabel));
  CHKERRQ(DMLabelGetStratumIS(depthLabel, dim + 1, &cellIS));
  CHKERRQ(DMFieldGetDegree(field,cellIS,NULL,&maxDegree));
  CHKERRQ(ISGetIndices(pointIS, &points));
  numFaces = geom->numCells;
  Nq = geom->numPoints;
  /* First, set local faces and flip normals so that they are outward for the first supporting cell */
  for (p = 0; p < numFaces; p++) {
    PetscInt        point = points[p];
    PetscInt        suppSize, s, coneSize, c, numChildren;
    const PetscInt *supp, *cone, *ornt;

    CHKERRQ(DMPlexGetTreeChildren(dm, point, &numChildren, NULL));
    PetscCheckFalse(numChildren,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Face data not valid for facets with children");
    CHKERRQ(DMPlexGetSupportSize(dm, point, &suppSize));
    PetscCheckFalse(suppSize > 2,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D has %D support, expected at most 2", point, suppSize);
    if (!suppSize) continue;
    CHKERRQ(DMPlexGetSupport(dm, point, &supp));
    for (s = 0; s < suppSize; ++s) {
      CHKERRQ(DMPlexGetConeSize(dm, supp[s], &coneSize));
      CHKERRQ(DMPlexGetCone(dm, supp[s], &cone));
      for (c = 0; c < coneSize; ++c) if (cone[c] == point) break;
      PetscCheckFalse(c == coneSize,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid connectivity: point %D not found in cone of support point %D", point, supp[s]);
      geom->face[p][s] = c;
    }
    CHKERRQ(DMPlexGetConeOrientation(dm, supp[0], &ornt));
    if (ornt[geom->face[p][0]] < 0) {
      PetscInt Np = geom->numPoints, q, dE = geom->dimEmbed, d;

      for (q = 0; q < Np; ++q) for (d = 0; d < dE; ++d) geom->n[(p*Np + q)*dE + d] = -geom->n[(p*Np + q)*dE + d];
    }
  }
  if (maxDegree <= 1) {
    PetscInt        numCells, offset, *cells;
    PetscFEGeom     *cellGeom;
    IS              suppIS;

    for (p = 0, numCells = 0; p < numFaces; p++) {
      PetscInt        point = points[p];
      PetscInt        numSupp, numChildren;

      CHKERRQ(DMPlexGetTreeChildren(dm, point, &numChildren, NULL));
      PetscCheckFalse(numChildren,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Face data not valid for facets with children");
      CHKERRQ(DMPlexGetSupportSize(dm, point,&numSupp));
      PetscCheckFalse(numSupp > 2,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D has %D support, expected at most 2", point, numSupp);
      numCells += numSupp;
    }
    CHKERRQ(PetscMalloc1(numCells, &cells));
    for (p = 0, offset = 0; p < numFaces; p++) {
      PetscInt        point = points[p];
      PetscInt        numSupp, s;
      const PetscInt *supp;

      CHKERRQ(DMPlexGetSupportSize(dm, point,&numSupp));
      CHKERRQ(DMPlexGetSupport(dm, point, &supp));
      for (s = 0; s < numSupp; s++, offset++) {
        cells[offset] = supp[s];
      }
    }
    CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,numCells,cells,PETSC_USE_POINTER, &suppIS));
    CHKERRQ(DMFieldCreateFEGeom(field,suppIS,quad,PETSC_FALSE,&cellGeom));
    for (p = 0, offset = 0; p < numFaces; p++) {
      PetscInt        point = points[p];
      PetscInt        numSupp, s, q;
      const PetscInt *supp;

      CHKERRQ(DMPlexGetSupportSize(dm, point,&numSupp));
      CHKERRQ(DMPlexGetSupport(dm, point, &supp));
      for (s = 0; s < numSupp; s++, offset++) {
        for (q = 0; q < Nq * dE * dE; q++) {
          geom->suppJ[s][p * Nq * dE * dE + q]    = cellGeom->J[offset * Nq * dE * dE + q];
          geom->suppInvJ[s][p * Nq * dE * dE + q] = cellGeom->invJ[offset * Nq * dE * dE + q];
        }
        for (q = 0; q < Nq; q++) geom->suppDetJ[s][p * Nq + q] = cellGeom->detJ[offset * Nq + q];
      }
    }
    CHKERRQ(PetscFEGeomDestroy(&cellGeom));
    CHKERRQ(ISDestroy(&suppIS));
    CHKERRQ(PetscFree(cells));
  } else {
    PetscObject          faceDisc, cellDisc;
    PetscClassId         faceId, cellId;
    PetscDualSpace       dsp;
    DM                   K;
    DMPolytopeType       ct;
    PetscInt           (*co)[2][3];
    PetscInt             coneSize;
    PetscInt           **counts;
    PetscInt             f, i, o, q, s;
    const PetscInt      *coneK;
    PetscInt             eStart, minOrient, maxOrient, numOrient;
    PetscInt            *orients;
    PetscReal          **orientPoints;
    PetscReal           *cellPoints;
    PetscReal           *dummyWeights;
    PetscQuadrature      cellQuad = NULL;

    CHKERRQ(DMFieldDSGetHeightDisc(field, 1, &faceDisc));
    CHKERRQ(DMFieldDSGetHeightDisc(field, 0, &cellDisc));
    CHKERRQ(PetscObjectGetClassId(faceDisc,&faceId));
    CHKERRQ(PetscObjectGetClassId(cellDisc,&cellId));
    PetscCheckFalse(faceId != PETSCFE_CLASSID || cellId != PETSCFE_CLASSID,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Not supported");
    CHKERRQ(PetscFEGetDualSpace((PetscFE)cellDisc, &dsp));
    CHKERRQ(PetscDualSpaceGetDM(dsp, &K));
    CHKERRQ(DMPlexGetHeightStratum(K, 1, &eStart, NULL));
    CHKERRQ(DMPlexGetCellType(K, eStart, &ct));
    CHKERRQ(DMPlexGetConeSize(K,0,&coneSize));
    CHKERRQ(DMPlexGetCone(K,0,&coneK));
    CHKERRQ(PetscMalloc2(numFaces, &co, coneSize, &counts));
    CHKERRQ(PetscMalloc1(dE*Nq, &cellPoints));
    CHKERRQ(PetscMalloc1(Nq, &dummyWeights));
    CHKERRQ(PetscQuadratureCreate(PETSC_COMM_SELF, &cellQuad));
    CHKERRQ(PetscQuadratureSetData(cellQuad, dE, 1, Nq, cellPoints, dummyWeights));
    minOrient = PETSC_MAX_INT;
    maxOrient = PETSC_MIN_INT;
    for (p = 0; p < numFaces; p++) { /* record the orientation of the facet wrt the support cells */
      PetscInt        point = points[p];
      PetscInt        numSupp, numChildren;
      const PetscInt *supp;

      CHKERRQ(DMPlexGetTreeChildren(dm, point, &numChildren, NULL));
      PetscCheckFalse(numChildren,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Face data not valid for facets with children");
      CHKERRQ(DMPlexGetSupportSize(dm, point,&numSupp));
      PetscCheckFalse(numSupp > 2,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D has %D support, expected at most 2", point, numSupp);
      CHKERRQ(DMPlexGetSupport(dm, point, &supp));
      for (s = 0; s < numSupp; s++) {
        PetscInt        cell = supp[s];
        PetscInt        numCone;
        const PetscInt *cone, *orient;

        CHKERRQ(DMPlexGetConeSize(dm, cell, &numCone));
        PetscCheckFalse(numCone != coneSize,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Support point does not match reference element");
        CHKERRQ(DMPlexGetCone(dm, cell, &cone));
        CHKERRQ(DMPlexGetConeOrientation(dm, cell, &orient));
        for (f = 0; f < coneSize; f++) {
          if (cone[f] == point) break;
        }
        co[p][s][0] = f;
        co[p][s][1] = orient[f];
        co[p][s][2] = cell;
        minOrient = PetscMin(minOrient, orient[f]);
        maxOrient = PetscMax(maxOrient, orient[f]);
      }
      for (; s < 2; s++) {
        co[p][s][0] = -1;
        co[p][s][1] = -1;
        co[p][s][2] = -1;
      }
    }
    numOrient = maxOrient + 1 - minOrient;
    CHKERRQ(DMPlexGetCone(K,0,&coneK));
    /* count all (face,orientation) doubles that appear */
    CHKERRQ(PetscCalloc2(numOrient,&orients,numOrient,&orientPoints));
    for (f = 0; f < coneSize; f++) CHKERRQ(PetscCalloc1(numOrient+1, &counts[f]));
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

        CHKERRQ(PetscMalloc1(Nq * dim, &orientPoints[o]));
        /* rotate the quadrature points appropriately */
        switch (ct) {
        case DM_POLYTOPE_POINT: break;
        case DM_POLYTOPE_SEGMENT:
          if (orient == -2 || orient == 1) {
            for (q = 0; q < Nq; q++) {
              orientPoints[o][q] = -geom->xi[q];
            }
          } else {
            for (q = 0; q < Nq; q++) {
              orientPoints[o][q] = geom->xi[q];
            }
          }
          break;
        case DM_POLYTOPE_TRIANGLE:
          for (q = 0; q < Nq; q++) {
            PetscReal lambda[3];
            PetscReal lambdao[3];

            /* convert to barycentric */
            lambda[0] = - (geom->xi[2 * q] + geom->xi[2 * q + 1]) / 2.;
            lambda[1] = (geom->xi[2 * q] + 1.) / 2.;
            lambda[2] = (geom->xi[2 * q + 1] + 1.) / 2.;
            if (orient >= 0) {
              for (i = 0; i < 3; i++) {
                lambdao[i] = lambda[(orient + i) % 3];
              }
            } else {
              for (i = 0; i < 3; i++) {
                lambdao[i] = lambda[(-(orient + i) + 3) % 3];
              }
            }
            /* convert to coordinates */
            orientPoints[o][2 * q + 0] = -(lambdao[0] + lambdao[2]) + lambdao[1];
            orientPoints[o][2 * q + 1] = -(lambdao[0] + lambdao[1]) + lambdao[2];
          }
          break;
        case DM_POLYTOPE_QUADRILATERAL:
          for (q = 0; q < Nq; q++) {
            PetscReal xi[2], xio[2];
            PetscInt oabs = (orient >= 0) ? orient : -(orient + 1);

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
            case 3:
              xio[0] = -xi[1];
              xio[1] = xi[0];
            case 0:
            default:
              xio[0] = xi[0];
              xio[1] = xi[1];
              break;
            }
            if (orient < 0) {
              xio[0] = -xio[0];
            }
            orientPoints[o][2 * q + 0] = xio[0];
            orientPoints[o][2 * q + 1] = xio[1];
          }
          break;
        default: SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cell type %s not yet supported", DMPolytopeTypes[ct]);
        }
      }
    }
    for (f = 0; f < coneSize; f++) {
      PetscInt face = coneK[f];
      PetscReal v0[3];
      PetscReal J[9], detJ;
      PetscInt numCells, offset;
      PetscInt *cells;
      IS suppIS;

      CHKERRQ(DMPlexComputeCellGeometryFEM(K, face, NULL, v0, J, NULL, &detJ));
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
        CHKERRQ(PetscMalloc1(numCells, &cells));
        for (p = 0, offset = 0; p < numFaces; p++) {
          for (s = 0; s < 2; s++) {
            if (co[p][s][0] == f && co[p][s][1] == o + minOrient) {
              cells[offset++] = co[p][s][2];
            }
          }
        }
        CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,numCells,cells,PETSC_USE_POINTER, &suppIS));
        CHKERRQ(DMFieldCreateFEGeom(field,suppIS,cellQuad,PETSC_FALSE,&cellGeom));
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
        CHKERRQ(PetscFEGeomDestroy(&cellGeom));
        CHKERRQ(ISDestroy(&suppIS));
        CHKERRQ(PetscFree(cells));
      }
    }
    for (o = 0; o < numOrient; o++) {
      if (orients[o]) {
        CHKERRQ(PetscFree(orientPoints[o]));
      }
    }
    CHKERRQ(PetscFree2(orients,orientPoints));
    CHKERRQ(PetscQuadratureDestroy(&cellQuad));
    for (f = 0; f < coneSize; f++) CHKERRQ(PetscFree(counts[f]));
    CHKERRQ(PetscFree2(co,counts));
  }
  CHKERRQ(ISRestoreIndices(pointIS, &points));
  CHKERRQ(ISDestroy(&cellIS));
  PetscFunctionReturn(0);
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
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode DMFieldCreate_DS(DMField field)
{
  DMField_DS     *dsfield;

  PetscFunctionBegin;
  CHKERRQ(PetscNewLog(field,&dsfield));
  field->data = dsfield;
  CHKERRQ(DMFieldInitialize_DS(field));
  PetscFunctionReturn(0);
}

PetscErrorCode DMFieldCreateDS(DM dm, PetscInt fieldNum, Vec vec,DMField *field)
{
  DMField        b;
  DMField_DS     *dsfield;
  PetscObject    disc = NULL;
  PetscBool      isContainer = PETSC_FALSE;
  PetscClassId   id = -1;
  PetscInt       numComponents = -1, dsNumFields;
  PetscSection   section;

  PetscFunctionBegin;
  CHKERRQ(DMGetLocalSection(dm,&section));
  CHKERRQ(PetscSectionGetFieldComponents(section,fieldNum,&numComponents));
  CHKERRQ(DMGetNumFields(dm,&dsNumFields));
  if (dsNumFields) CHKERRQ(DMGetField(dm,fieldNum,NULL,&disc));
  if (disc) {
    CHKERRQ(PetscObjectGetClassId(disc,&id));
    isContainer = (id == PETSC_CONTAINER_CLASSID) ? PETSC_TRUE : PETSC_FALSE;
  }
  if (!disc || isContainer) {
    MPI_Comm       comm = PetscObjectComm((PetscObject) dm);
    PetscFE        fe;
    DMPolytopeType ct, locct = DM_POLYTOPE_UNKNOWN;
    PetscInt       dim, cStart, cEnd, cellHeight;

    CHKERRQ(DMPlexGetVTKCellHeight(dm, &cellHeight));
    CHKERRQ(DMGetDimension(dm, &dim));
    CHKERRQ(DMPlexGetHeightStratum(dm, cellHeight, &cStart, &cEnd));
    if (cEnd > cStart) CHKERRQ(DMPlexGetCellType(dm, cStart, &locct));
    CHKERRMPI(MPI_Allreduce(&locct, &ct, 1, MPI_INT, MPI_MIN, comm));
    CHKERRQ(PetscFECreateLagrangeByCell(PETSC_COMM_SELF, dim, numComponents, ct, 1, PETSC_DETERMINE, &fe));
    CHKERRQ(PetscFEViewFromOptions(fe, NULL, "-field_fe_view"));
    disc = (PetscObject) fe;
  } else {
    CHKERRQ(PetscObjectReference(disc));
  }
  CHKERRQ(PetscObjectGetClassId(disc,&id));
  if (id == PETSCFE_CLASSID) {
    PetscFE fe = (PetscFE) disc;

    CHKERRQ(PetscFEGetNumComponents(fe,&numComponents));
  } else SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Not implemented");
  CHKERRQ(DMFieldCreate(dm,numComponents,DMFIELD_VERTEX,&b));
  CHKERRQ(DMFieldSetType(b,DMFIELDDS));
  dsfield = (DMField_DS *) b->data;
  dsfield->fieldNum = fieldNum;
  CHKERRQ(DMGetDimension(dm,&dsfield->height));
  dsfield->height++;
  CHKERRQ(PetscCalloc1(dsfield->height,&dsfield->disc));
  dsfield->disc[0] = disc;
  CHKERRQ(PetscObjectReference((PetscObject)vec));
  dsfield->vec = vec;
  *field = b;

  PetscFunctionReturn(0);
}
