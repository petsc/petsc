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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  dsfield = (DMField_DS *) field->data;
  ierr = VecDestroy(&dsfield->vec);CHKERRQ(ierr);
  for (i = 0; i < dsfield->height; i++) {
    ierr = PetscObjectDereference(dsfield->disc[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(dsfield->disc);CHKERRQ(ierr);
  ierr = PetscFree(dsfield);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMFieldView_DS(DMField field,PetscViewer viewer)
{
  DMField_DS     *dsfield = (DMField_DS *) field->data;
  PetscBool      iascii;
  PetscObject    disc;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  disc = dsfield->disc[0];
  if (iascii) {
    PetscViewerASCIIPrintf(viewer, "PetscDS field %D\n",dsfield->fieldNum);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = PetscObjectView(disc,viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
  if (dsfield->multifieldVec) {
    SETERRQ(PetscObjectComm((PetscObject)field),PETSC_ERR_SUP,"View of subfield not implemented yet");
  } else {
    ierr = VecView(dsfield->vec,viewer);CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMFieldDSGetHeightDisc(DMField field, PetscInt height, PetscObject *disc)
{
  DMField_DS     *dsfield = (DMField_DS *) field->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!dsfield->disc[height]) {
    PetscClassId   id;

    ierr = PetscObjectGetClassId(dsfield->disc[0],&id);CHKERRQ(ierr);
    if (id == PETSCFE_CLASSID) {
      PetscFE fe = (PetscFE) dsfield->disc[0];

      ierr = PetscFECreateHeightTrace(fe,height,(PetscFE *)&dsfield->disc[height]);CHKERRQ(ierr);
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
  PetscErrorCode  ierr;

  PetscFunctionBeginHot;
  dm   = field->dm;
  nc   = field->numComponents;
  ierr = PetscQuadratureGetData(quad,&dim,NULL,&nq,&qpoints,NULL);CHKERRQ(ierr);
  ierr = DMFieldDSGetHeightDisc(field,dsfield->height - 1 - dim,&disc);CHKERRQ(ierr);
  ierr = DMGetDimension(dm,&meshDim);CHKERRQ(ierr);
  ierr = DMGetLocalSection(dm,&section);CHKERRQ(ierr);
  ierr = PetscSectionGetField(section,dsfield->fieldNum,&section);CHKERRQ(ierr);
  ierr = PetscObjectGetClassId(disc,&classid);CHKERRQ(ierr);
  /* TODO: batch */
  ierr = PetscObjectTypeCompare((PetscObject)pointIS,ISSTRIDE,&isStride);CHKERRQ(ierr);
  ierr = ISGetLocalSize(pointIS,&numCells);CHKERRQ(ierr);
  if (isStride) {
    ierr = ISStrideGetInfo(pointIS,&sfirst,&stride);CHKERRQ(ierr);
  } else {
    ierr = ISGetIndices(pointIS,&points);CHKERRQ(ierr);
  }
  if (classid == PETSCFE_CLASSID) {
    PetscFE      fe = (PetscFE) disc;
    PetscInt     feDim, i;
    PetscInt          K = H ? 2 : (D ? 1 : (B ? 0 : -1));
    PetscTabulation   T;

    ierr = PetscFEGetDimension(fe,&feDim);CHKERRQ(ierr);
    ierr = PetscFECreateTabulation(fe,1,nq,qpoints,K,&T);CHKERRQ(ierr);
    for (i = 0; i < numCells; i++) {
      PetscInt     c = isStride ? (sfirst + i * stride) : points[i];
      PetscInt     closureSize;
      PetscScalar *elem = NULL;

      ierr = DMPlexVecGetClosure(dm,section,dsfield->vec,c,&closureSize,&elem);CHKERRQ(ierr);
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
      ierr = DMPlexVecRestoreClosure(dm,section,dsfield->vec,c,&closureSize,&elem);CHKERRQ(ierr);
    }
    ierr = PetscTabulationDestroy(&T);CHKERRQ(ierr);
  } else SETERRQ(PetscObjectComm((PetscObject)field),PETSC_ERR_SUP,"Not implemented");
  if (!isStride) {
    ierr = ISRestoreIndices(pointIS,&points);CHKERRQ(ierr);
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
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  nc   = field->numComponents;
  ierr = DMGetLocalSection(field->dm,&section);CHKERRQ(ierr);
  ierr = DMFieldDSGetHeightDisc(field,0,&cellDisc);CHKERRQ(ierr);
  ierr = PetscObjectGetClassId(cellDisc, &discID);CHKERRQ(ierr);
  if (discID != PETSCFE_CLASSID) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB, "Discretization type not supported\n");
  cellFE = (PetscFE) cellDisc;
  ierr = PetscFEGetDimension(cellFE,&feDim);CHKERRQ(ierr);
  ierr = DMGetCoordinateDim(field->dm, &dim);CHKERRQ(ierr);
  ierr = DMGetDimension(field->dm, &dimR);CHKERRQ(ierr);
  ierr = DMLocatePoints(field->dm, points, DM_POINTLOCATION_NONE, &cellSF);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(cellSF, &numCells, &nFound, NULL, &cells);CHKERRQ(ierr);
  for (c = 0; c < nFound; c++) {
    if (cells[c].index < 0) SETERRQ1(PetscObjectComm((PetscObject)points),PETSC_ERR_ARG_WRONG, "Point %D could not be located\n", c);
  }
  ierr = PetscSFComputeDegreeBegin(cellSF,&cellDegrees);CHKERRQ(ierr);
  ierr = PetscSFComputeDegreeEnd(cellSF,&cellDegrees);CHKERRQ(ierr);
  for (c = 0, gatherSize = 0, gatherMax = 0; c < numCells; c++) {
    gatherMax = PetscMax(gatherMax,cellDegrees[c]);
    gatherSize += cellDegrees[c];
  }
  ierr = PetscMalloc3(gatherSize*dim,&cellPoints,gatherMax*dim,&coordsReal,gatherMax*dimR,&coordsRef);CHKERRQ(ierr);
  ierr = PetscMalloc4(gatherMax*dimR,&v,gatherMax*dimR*dimR,&J,gatherMax*dimR*dimR,&invJ,gatherMax,&detJ);CHKERRQ(ierr);
  if (datatype == PETSC_SCALAR) {
    ierr = PetscMalloc3(B ? nc * gatherSize : 0, &cellBs, D ? nc * dim * gatherSize : 0, &cellDs, H ? nc * dim * dim * gatherSize : 0, &cellHs);CHKERRQ(ierr);
  } else {
    ierr = PetscMalloc3(B ? nc * gatherSize : 0, &cellBr, D ? nc * dim * gatherSize : 0, &cellDr, H ? nc * dim * dim * gatherSize : 0, &cellHr);CHKERRQ(ierr);
  }

  ierr = MPI_Type_contiguous(dim,MPIU_SCALAR,&pointType);CHKERRMPI(ierr);
  ierr = MPI_Type_commit(&pointType);CHKERRMPI(ierr);
  ierr = VecGetArrayRead(points,&pointsArray);CHKERRQ(ierr);
  ierr = PetscSFGatherBegin(cellSF, pointType, pointsArray, cellPoints);CHKERRQ(ierr);
  ierr = PetscSFGatherEnd(cellSF, pointType, pointsArray, cellPoints);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(points,&pointsArray);CHKERRQ(ierr);
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
      ierr = DMPlexCoordinatesToReference(field->dm, c, nq, coordsReal, coordsRef);CHKERRQ(ierr);
      ierr = PetscFECreateTabulation(cellFE,1,nq,coordsRef,K,&T);CHKERRQ(ierr);
      ierr = PetscQuadratureCreate(PETSC_COMM_SELF, &quad);CHKERRQ(ierr);
      ierr = PetscMalloc1(dimR * nq, &quadPoints);CHKERRQ(ierr);
      for (p = 0; p < dimR * nq; p++) quadPoints[p] = coordsRef[p];
      ierr = PetscQuadratureSetData(quad, dimR, 0, nq, quadPoints, NULL);CHKERRQ(ierr);
      ierr = DMPlexComputeCellGeometryFEM(field->dm, c, quad, v, J, invJ, detJ);CHKERRQ(ierr);
      ierr = PetscQuadratureDestroy(&quad);CHKERRQ(ierr);
      ierr = DMPlexVecGetClosure(field->dm,section,dsfield->vec,c,&closureSize,&elem);CHKERRQ(ierr);
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
      ierr = DMPlexVecRestoreClosure(field->dm,section,dsfield->vec,c,&closureSize,&elem);CHKERRQ(ierr);
      ierr = PetscTabulationDestroy(&T);CHKERRQ(ierr);
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

      ierr = MPI_Type_contiguous(nc, origtype, &Btype);CHKERRMPI(ierr);
      ierr = MPI_Type_commit(&Btype);CHKERRMPI(ierr);
      ierr = PetscSFScatterBegin(cellSF,Btype,(datatype == PETSC_SCALAR) ? (void *) cellBs : (void *) cellBr, B);CHKERRQ(ierr);
      ierr = PetscSFScatterEnd(cellSF,Btype,(datatype == PETSC_SCALAR) ? (void *) cellBs : (void *) cellBr, B);CHKERRQ(ierr);
      ierr = MPI_Type_free(&Btype);CHKERRMPI(ierr);
    }
    if (D) {
      MPI_Datatype Dtype;

      ierr = MPI_Type_contiguous(nc * dim, origtype, &Dtype);CHKERRMPI(ierr);
      ierr = MPI_Type_commit(&Dtype);CHKERRMPI(ierr);
      ierr = PetscSFScatterBegin(cellSF,Dtype,(datatype == PETSC_SCALAR) ? (void *) cellDs : (void *) cellDr, D);CHKERRQ(ierr);
      ierr = PetscSFScatterEnd(cellSF,Dtype,(datatype == PETSC_SCALAR) ? (void *) cellDs : (void *) cellDr, D);CHKERRQ(ierr);
      ierr = MPI_Type_free(&Dtype);CHKERRMPI(ierr);
    }
    if (H) {
      MPI_Datatype Htype;

      ierr = MPI_Type_contiguous(nc * dim * dim, origtype, &Htype);CHKERRMPI(ierr);
      ierr = MPI_Type_commit(&Htype);CHKERRMPI(ierr);
      ierr = PetscSFScatterBegin(cellSF,Htype,(datatype == PETSC_SCALAR) ? (void *) cellHs : (void *) cellHr, H);CHKERRQ(ierr);
      ierr = PetscSFScatterEnd(cellSF,Htype,(datatype == PETSC_SCALAR) ? (void *) cellHs : (void *) cellHr, H);CHKERRQ(ierr);
      ierr = MPI_Type_free(&Htype);CHKERRMPI(ierr);
    }
  }
  ierr = PetscFree4(v,J,invJ,detJ);CHKERRQ(ierr);
  ierr = PetscFree3(cellBr, cellDr, cellHr);CHKERRQ(ierr);
  ierr = PetscFree3(cellBs, cellDs, cellHs);CHKERRQ(ierr);
  ierr = PetscFree3(cellPoints,coordsReal,coordsRef);CHKERRQ(ierr);
  ierr = MPI_Type_free(&pointType);CHKERRMPI(ierr);
  ierr = PetscSFDestroy(&cellSF);CHKERRQ(ierr);
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
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  Nc = field->numComponents;
  ierr = DMGetCoordinateDim(field->dm, &dimC);CHKERRQ(ierr);
  ierr = DMGetDimension(field->dm, &dim);CHKERRQ(ierr);
  ierr = ISGetLocalSize(pointIS, &numPoints);CHKERRQ(ierr);
  ierr = ISGetMinMax(pointIS,&imin,NULL);CHKERRQ(ierr);
  for (h = 0; h < dsfield->height; h++) {
    PetscInt hEnd;

    ierr = DMPlexGetHeightStratum(field->dm,h,NULL,&hEnd);CHKERRQ(ierr);
    if (imin < hEnd) break;
  }
  dim -= h;
  ierr = DMFieldDSGetHeightDisc(field,h,&disc);CHKERRQ(ierr);
  ierr = PetscObjectGetClassId(disc,&id);CHKERRQ(ierr);
  if (id != PETSCFE_CLASSID) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Discretization not supported\n");
  ierr = DMGetCoordinateField(field->dm, &coordField);CHKERRQ(ierr);
  ierr = DMFieldGetDegree(coordField, pointIS, NULL, &maxDegree);CHKERRQ(ierr);
  if (maxDegree <= 1) {
    ierr = DMFieldCreateDefaultQuadrature(coordField, pointIS, &quad);CHKERRQ(ierr);
  }
  if (!quad) {ierr = DMFieldCreateDefaultQuadrature(field, pointIS, &quad);CHKERRQ(ierr);}
  if (!quad) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Could not determine quadrature for cell averages\n");
  ierr = DMFieldCreateFEGeom(coordField,pointIS,quad,PETSC_FALSE,&geom);CHKERRQ(ierr);
  ierr = PetscQuadratureGetData(quad, NULL, &qNc, &Nq, NULL, &weights);CHKERRQ(ierr);
  if (qNc != 1) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Expected scalar quadrature components\n");
  N = numPoints * Nq * Nc;
  if (B) ierr = DMGetWorkArray(field->dm, N, mpitype, &qB);CHKERRQ(ierr);
  if (D) ierr = DMGetWorkArray(field->dm, N * dimC, mpitype, &qD);CHKERRQ(ierr);
  if (H) ierr = DMGetWorkArray(field->dm, N * dimC * dimC, mpitype, &qH);CHKERRQ(ierr);
  ierr = DMFieldEvaluateFE(field,pointIS,quad,type,qB,qD,qH);CHKERRQ(ierr);
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
  if (B) ierr = DMRestoreWorkArray(field->dm, N, mpitype, &qB);CHKERRQ(ierr);
  if (D) ierr = DMRestoreWorkArray(field->dm, N * dimC, mpitype, &qD);CHKERRQ(ierr);
  if (H) ierr = DMRestoreWorkArray(field->dm, N * dimC * dimC, mpitype, &qH);CHKERRQ(ierr);
  ierr = PetscFEGeomDestroy(&geom);CHKERRQ(ierr);
  ierr = PetscQuadratureDestroy(&quad);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMFieldGetDegree_DS(DMField field, IS pointIS, PetscInt *minDegree, PetscInt *maxDegree)
{
  DMField_DS     *dsfield;
  PetscObject    disc;
  PetscInt       h, imin, imax;
  PetscClassId   id;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  dsfield = (DMField_DS *) field->data;
  ierr = ISGetMinMax(pointIS,&imin,&imax);CHKERRQ(ierr);
  if (imin >= imax) {
    h = 0;
  } else {
    for (h = 0; h < dsfield->height; h++) {
      PetscInt hEnd;

      ierr = DMPlexGetHeightStratum(field->dm,h,NULL,&hEnd);CHKERRQ(ierr);
      if (imin < hEnd) break;
    }
  }
  ierr = DMFieldDSGetHeightDisc(field,h,&disc);CHKERRQ(ierr);
  ierr = PetscObjectGetClassId(disc,&id);CHKERRQ(ierr);
  if (id == PETSCFE_CLASSID) {
    PetscFE    fe = (PetscFE) disc;
    PetscSpace sp;

    ierr = PetscFEGetBasisSpace(fe, &sp);CHKERRQ(ierr);
    ierr = PetscSpaceGetDegree(sp, minDegree, maxDegree);CHKERRQ(ierr);
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
  PetscErrorCode ierr;


  PetscFunctionBegin;
  dm = field->dm;
  dsfield = (DMField_DS *) field->data;
  ierr = ISGetMinMax(pointIS,&imin,&imax);CHKERRQ(ierr);
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  for (h = 0; h <= dim; h++) {
    PetscInt hStart, hEnd;

    ierr = DMPlexGetHeightStratum(dm,h,&hStart,&hEnd);CHKERRQ(ierr);
    if (imax >= hStart && imin < hEnd) break;
  }
  ierr = DMPlexGetVTKCellHeight(dm, &cellHeight);CHKERRQ(ierr);
  h -= cellHeight;
  *quad = NULL;
  if (h < dsfield->height) {
    ierr = DMFieldDSGetHeightDisc(field,h,&disc);CHKERRQ(ierr);
    ierr = PetscObjectGetClassId(disc,&id);CHKERRQ(ierr);
    if (id != PETSCFE_CLASSID) PetscFunctionReturn(0);
    fe = (PetscFE) disc;
    ierr = PetscFEGetQuadrature(fe,quad);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)*quad);CHKERRQ(ierr);
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
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  dim = geom->dim;
  dE  = geom->dimEmbed;
  ierr = DMPlexGetDepthLabel(dm, &depthLabel);CHKERRQ(ierr);
  ierr = DMLabelGetStratumIS(depthLabel, dim + 1, &cellIS);CHKERRQ(ierr);
  ierr = DMFieldGetDegree(field,cellIS,NULL,&maxDegree);CHKERRQ(ierr);
  ierr = ISGetIndices(pointIS, &points);CHKERRQ(ierr);
  numFaces = geom->numCells;
  Nq = geom->numPoints;
  /* First, set local faces and flip normals so that they are outward for the first supporting cell */
  for (p = 0; p < numFaces; p++) {
    PetscInt        point = points[p];
    PetscInt        suppSize, s, coneSize, c, numChildren;
    const PetscInt *supp, *cone, *ornt;

    ierr = DMPlexGetTreeChildren(dm, point, &numChildren, NULL);CHKERRQ(ierr);
    if (numChildren) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Face data not valid for facets with children");
    ierr = DMPlexGetSupportSize(dm, point, &suppSize);CHKERRQ(ierr);
    if (suppSize > 2) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D has %D support, expected at most 2\n", point, suppSize);
    if (!suppSize) continue;
    ierr = DMPlexGetSupport(dm, point, &supp);CHKERRQ(ierr);
    for (s = 0; s < suppSize; ++s) {
      ierr = DMPlexGetConeSize(dm, supp[s], &coneSize);CHKERRQ(ierr);
      ierr = DMPlexGetCone(dm, supp[s], &cone);CHKERRQ(ierr);
      for (c = 0; c < coneSize; ++c) if (cone[c] == point) break;
      if (c == coneSize) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid connectivity: point %D not found in cone of support point %D", point, supp[s]);
      geom->face[p][s] = c;
    }
    ierr = DMPlexGetConeOrientation(dm, supp[0], &ornt);CHKERRQ(ierr);
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

      ierr = DMPlexGetTreeChildren(dm, point, &numChildren, NULL);CHKERRQ(ierr);
      if (numChildren) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Face data not valid for facets with children");
      ierr = DMPlexGetSupportSize(dm, point,&numSupp);CHKERRQ(ierr);
      if (numSupp > 2) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D has %D support, expected at most 2\n", point, numSupp);
      numCells += numSupp;
    }
    ierr = PetscMalloc1(numCells, &cells);CHKERRQ(ierr);
    for (p = 0, offset = 0; p < numFaces; p++) {
      PetscInt        point = points[p];
      PetscInt        numSupp, s;
      const PetscInt *supp;

      ierr = DMPlexGetSupportSize(dm, point,&numSupp);CHKERRQ(ierr);
      ierr = DMPlexGetSupport(dm, point, &supp);CHKERRQ(ierr);
      for (s = 0; s < numSupp; s++, offset++) {
        cells[offset] = supp[s];
      }
    }
    ierr = ISCreateGeneral(PETSC_COMM_SELF,numCells,cells,PETSC_USE_POINTER, &suppIS);CHKERRQ(ierr);
    ierr = DMFieldCreateFEGeom(field,suppIS,quad,PETSC_FALSE,&cellGeom);CHKERRQ(ierr);
    for (p = 0, offset = 0; p < numFaces; p++) {
      PetscInt        point = points[p];
      PetscInt        numSupp, s, q;
      const PetscInt *supp;

      ierr = DMPlexGetSupportSize(dm, point,&numSupp);CHKERRQ(ierr);
      ierr = DMPlexGetSupport(dm, point, &supp);CHKERRQ(ierr);
      for (s = 0; s < numSupp; s++, offset++) {
        for (q = 0; q < Nq * dE * dE; q++) {
          geom->suppJ[s][p * Nq * dE * dE + q]    = cellGeom->J[offset * Nq * dE * dE + q];
          geom->suppInvJ[s][p * Nq * dE * dE + q] = cellGeom->invJ[offset * Nq * dE * dE + q];
        }
        for (q = 0; q < Nq; q++) geom->suppDetJ[s][p * Nq + q] = cellGeom->detJ[offset * Nq + q];
      }
    }
    ierr = PetscFEGeomDestroy(&cellGeom);CHKERRQ(ierr);
    ierr = ISDestroy(&suppIS);CHKERRQ(ierr);
    ierr = PetscFree(cells);CHKERRQ(ierr);
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

    ierr = DMFieldDSGetHeightDisc(field, 1, &faceDisc);CHKERRQ(ierr);
    ierr = DMFieldDSGetHeightDisc(field, 0, &cellDisc);CHKERRQ(ierr);
    ierr = PetscObjectGetClassId(faceDisc,&faceId);CHKERRQ(ierr);
    ierr = PetscObjectGetClassId(cellDisc,&cellId);CHKERRQ(ierr);
    if (faceId != PETSCFE_CLASSID || cellId != PETSCFE_CLASSID) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Not supported\n");
    ierr = PetscFEGetDualSpace((PetscFE)cellDisc, &dsp);CHKERRQ(ierr);
    ierr = PetscDualSpaceGetDM(dsp, &K);CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(K, 1, &eStart, NULL);CHKERRQ(ierr);
    ierr = DMPlexGetCellType(K, eStart, &ct);CHKERRQ(ierr);
    ierr = DMPlexGetConeSize(K,0,&coneSize);CHKERRQ(ierr);
    ierr = DMPlexGetCone(K,0,&coneK);CHKERRQ(ierr);
    ierr = PetscMalloc2(numFaces, &co, coneSize, &counts);CHKERRQ(ierr);
    ierr = PetscMalloc1(dE*Nq, &cellPoints);CHKERRQ(ierr);
    ierr = PetscMalloc1(Nq, &dummyWeights);CHKERRQ(ierr);
    ierr = PetscQuadratureCreate(PETSC_COMM_SELF, &cellQuad);CHKERRQ(ierr);
    ierr = PetscQuadratureSetData(cellQuad, dE, 1, Nq, cellPoints, dummyWeights);CHKERRQ(ierr);
    minOrient = PETSC_MAX_INT;
    maxOrient = PETSC_MIN_INT;
    for (p = 0; p < numFaces; p++) { /* record the orientation of the facet wrt the support cells */
      PetscInt        point = points[p];
      PetscInt        numSupp, numChildren;
      const PetscInt *supp;

      ierr = DMPlexGetTreeChildren(dm, point, &numChildren, NULL);CHKERRQ(ierr);
      if (numChildren) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Face data not valid for facets with children");
      ierr = DMPlexGetSupportSize(dm, point,&numSupp);CHKERRQ(ierr);
      if (numSupp > 2) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Point %D has %D support, expected at most 2\n", point, numSupp);
      ierr = DMPlexGetSupport(dm, point, &supp);CHKERRQ(ierr);
      for (s = 0; s < numSupp; s++) {
        PetscInt        cell = supp[s];
        PetscInt        numCone;
        const PetscInt *cone, *orient;

        ierr = DMPlexGetConeSize(dm, cell, &numCone);CHKERRQ(ierr);
        if (numCone != coneSize) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Support point does not match reference element");
        ierr = DMPlexGetCone(dm, cell, &cone);CHKERRQ(ierr);
        ierr = DMPlexGetConeOrientation(dm, cell, &orient);CHKERRQ(ierr);
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
    ierr = DMPlexGetCone(K,0,&coneK);CHKERRQ(ierr);
    /* count all (face,orientation) doubles that appear */
    ierr = PetscCalloc2(numOrient,&orients,numOrient,&orientPoints);CHKERRQ(ierr);
    for (f = 0; f < coneSize; f++) {ierr = PetscCalloc1(numOrient+1, &counts[f]);CHKERRQ(ierr);}
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

        ierr = PetscMalloc1(Nq * dim, &orientPoints[o]);CHKERRQ(ierr);
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
        default: SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cell type %s not yet supported\n", DMPolytopeTypes[ct]);
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

      ierr = DMPlexComputeCellGeometryFEM(K, face, NULL, v0, J, NULL, &detJ);CHKERRQ(ierr);
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
        ierr = PetscMalloc1(numCells, &cells);CHKERRQ(ierr);
        for (p = 0, offset = 0; p < numFaces; p++) {
          for (s = 0; s < 2; s++) {
            if (co[p][s][0] == f && co[p][s][1] == o + minOrient) {
              cells[offset++] = co[p][s][2];
            }
          }
        }
        ierr = ISCreateGeneral(PETSC_COMM_SELF,numCells,cells,PETSC_USE_POINTER, &suppIS);CHKERRQ(ierr);
        ierr = DMFieldCreateFEGeom(field,suppIS,cellQuad,PETSC_FALSE,&cellGeom);CHKERRQ(ierr);
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
        ierr = PetscFEGeomDestroy(&cellGeom);CHKERRQ(ierr);
        ierr = ISDestroy(&suppIS);CHKERRQ(ierr);
        ierr = PetscFree(cells);CHKERRQ(ierr);
      }
    }
    for (o = 0; o < numOrient; o++) {
      if (orients[o]) {
        ierr = PetscFree(orientPoints[o]);CHKERRQ(ierr);
      }
    }
    ierr = PetscFree2(orients,orientPoints);CHKERRQ(ierr);
    ierr = PetscQuadratureDestroy(&cellQuad);CHKERRQ(ierr);
    for (f = 0; f < coneSize; f++) {ierr = PetscFree(counts[f]);CHKERRQ(ierr);}
    ierr = PetscFree2(co,counts);CHKERRQ(ierr);
  }
  ierr = ISRestoreIndices(pointIS, &points);CHKERRQ(ierr);
  ierr = ISDestroy(&cellIS);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(field,&dsfield);CHKERRQ(ierr);
  field->data = dsfield;
  ierr = DMFieldInitialize_DS(field);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetLocalSection(dm,&section);CHKERRQ(ierr);
  ierr = PetscSectionGetFieldComponents(section,fieldNum,&numComponents);CHKERRQ(ierr);
  ierr = DMGetNumFields(dm,&dsNumFields);CHKERRQ(ierr);
  if (dsNumFields) {ierr = DMGetField(dm,fieldNum,NULL,&disc);CHKERRQ(ierr);}
  if (disc) {
    ierr = PetscObjectGetClassId(disc,&id);CHKERRQ(ierr);
    isContainer = (id == PETSC_CONTAINER_CLASSID) ? PETSC_TRUE : PETSC_FALSE;
  }
  if (!disc || isContainer) {
    MPI_Comm        comm = PetscObjectComm((PetscObject) dm);
    PetscInt        cStart, cEnd, dim, cellHeight;
    PetscInt        localConeSize = 0, coneSize;
    PetscFE         fe;
    PetscDualSpace  Q;
    PetscSpace      P;
    DM              K;
    PetscQuadrature quad, fquad;
    PetscBool       isSimplex;

    ierr = DMPlexGetVTKCellHeight(dm, &cellHeight);CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(dm, cellHeight, &cStart, &cEnd);CHKERRQ(ierr);
    ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
    if (cEnd > cStart) {
      ierr = DMPlexGetConeSize(dm, cStart, &localConeSize);CHKERRQ(ierr);
    }
    ierr = MPI_Allreduce(&localConeSize,&coneSize,1,MPIU_INT,MPI_MAX,comm);CHKERRMPI(ierr);
    isSimplex = (coneSize == (dim + 1)) ? PETSC_TRUE : PETSC_FALSE;
    ierr = PetscSpaceCreate(PETSC_COMM_SELF, &P);CHKERRQ(ierr);
    ierr = PetscSpaceSetType(P,PETSCSPACEPOLYNOMIAL);CHKERRQ(ierr);
    ierr = PetscSpaceSetDegree(P, 1, PETSC_DETERMINE);CHKERRQ(ierr);
    ierr = PetscSpaceSetNumComponents(P, numComponents);CHKERRQ(ierr);
    ierr = PetscSpaceSetNumVariables(P, dim);CHKERRQ(ierr);
    ierr = PetscSpacePolynomialSetTensor(P, isSimplex ? PETSC_FALSE : PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscSpaceSetUp(P);CHKERRQ(ierr);
    ierr = PetscDualSpaceCreate(PETSC_COMM_SELF, &Q);CHKERRQ(ierr);
    ierr = PetscDualSpaceSetType(Q,PETSCDUALSPACELAGRANGE);CHKERRQ(ierr);
    ierr = PetscDualSpaceCreateReferenceCell(Q, dim, isSimplex, &K);CHKERRQ(ierr);
    ierr = PetscDualSpaceSetDM(Q, K);CHKERRQ(ierr);
    ierr = DMDestroy(&K);CHKERRQ(ierr);
    ierr = PetscDualSpaceSetNumComponents(Q, numComponents);CHKERRQ(ierr);
    ierr = PetscDualSpaceSetOrder(Q, 1);CHKERRQ(ierr);
    ierr = PetscDualSpaceLagrangeSetTensor(Q, isSimplex ? PETSC_FALSE : PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscDualSpaceSetUp(Q);CHKERRQ(ierr);
    ierr = PetscFECreate(PETSC_COMM_SELF, &fe);CHKERRQ(ierr);
    ierr = PetscFESetType(fe,PETSCFEBASIC);CHKERRQ(ierr);
    ierr = PetscFESetBasisSpace(fe, P);CHKERRQ(ierr);
    ierr = PetscFESetDualSpace(fe, Q);CHKERRQ(ierr);
    ierr = PetscFESetNumComponents(fe, numComponents);CHKERRQ(ierr);
    ierr = PetscFESetUp(fe);CHKERRQ(ierr);
    ierr = PetscSpaceDestroy(&P);CHKERRQ(ierr);
    ierr = PetscDualSpaceDestroy(&Q);CHKERRQ(ierr);
    if (isSimplex) {
      ierr = PetscDTStroudConicalQuadrature(dim,   1, 1, -1.0, 1.0, &quad);CHKERRQ(ierr);
      ierr = PetscDTStroudConicalQuadrature(dim-1, 1, 1, -1.0, 1.0, &fquad);CHKERRQ(ierr);
    }
    else {
      ierr = PetscDTGaussTensorQuadrature(dim,   1, 1, -1.0, 1.0, &quad);CHKERRQ(ierr);
      ierr = PetscDTGaussTensorQuadrature(dim-1, 1, 1, -1.0, 1.0, &fquad);CHKERRQ(ierr);
    }
    ierr = PetscFESetQuadrature(fe, quad);CHKERRQ(ierr);
    ierr = PetscFESetFaceQuadrature(fe, fquad);CHKERRQ(ierr);
    ierr = PetscQuadratureDestroy(&quad);CHKERRQ(ierr);
    ierr = PetscQuadratureDestroy(&fquad);CHKERRQ(ierr);
    disc = (PetscObject) fe;
  } else {
    ierr = PetscObjectReference(disc);CHKERRQ(ierr);
  }
  ierr = PetscObjectGetClassId(disc,&id);CHKERRQ(ierr);
  if (id == PETSCFE_CLASSID) {
    PetscFE fe = (PetscFE) disc;

    ierr = PetscFEGetNumComponents(fe,&numComponents);CHKERRQ(ierr);
  } else SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Not implemented");
  ierr = DMFieldCreate(dm,numComponents,DMFIELD_VERTEX,&b);CHKERRQ(ierr);
  ierr = DMFieldSetType(b,DMFIELDDS);CHKERRQ(ierr);
  dsfield = (DMField_DS *) b->data;
  dsfield->fieldNum = fieldNum;
  ierr = DMGetDimension(dm,&dsfield->height);CHKERRQ(ierr);
  dsfield->height++;
  ierr = PetscCalloc1(dsfield->height,&dsfield->disc);CHKERRQ(ierr);
  dsfield->disc[0] = disc;
  ierr = PetscObjectReference((PetscObject)vec);CHKERRQ(ierr);
  dsfield->vec = vec;
  *field = b;

  PetscFunctionReturn(0);
}
