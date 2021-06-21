#include <petsc/private/dmfieldimpl.h> /*I "petscdmfield.h" I*/
#include <petsc/private/dmimpl.h> /*I "petscdm.h" I*/
#include <petscdmda.h>

typedef struct _n_DMField_DA
{
  PetscScalar     *cornerVals;
  PetscScalar     *cornerCoeffs;
  PetscScalar     *work;
  PetscReal       coordRange[3][2];
}
DMField_DA;

static PetscErrorCode DMFieldDestroy_DA(DMField field)
{
  DMField_DA     *dafield;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  dafield = (DMField_DA *) field->data;
  ierr = PetscFree3(dafield->cornerVals,dafield->cornerCoeffs,dafield->work);CHKERRQ(ierr);
  ierr = PetscFree(dafield);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMFieldView_DA(DMField field,PetscViewer viewer)
{
  DMField_DA     *dafield = (DMField_DA *) field->data;
  PetscBool      iascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    PetscInt i, c, dim;
    PetscInt nc;
    DM       dm = field->dm;

    PetscViewerASCIIPrintf(viewer, "Field corner values:\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
    nc = field->numComponents;
    for (i = 0, c = 0; i < (1 << dim); i++) {
      PetscInt j;

      for (j = 0; j < nc; j++, c++) {
        PetscScalar val = dafield->cornerVals[nc * i + j];

#if !defined(PETSC_USE_COMPLEX)
        ierr = PetscViewerASCIIPrintf(viewer,"%g ",(double) val);CHKERRQ(ierr);
#else
        ierr = PetscViewerASCIIPrintf(viewer,"%g+i%g ",(double) PetscRealPart(val),(double) PetscImaginaryPart(val));CHKERRQ(ierr);
#endif
      }
      ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#define MEdot(y,A,x,m,c,cast)                          \
  do {                                                 \
    PetscInt _k, _l;                                   \
    for (_k = 0; _k < (c); _k++) (y)[_k] = 0.;         \
    for (_l = 0; _l < (m); _l++) {                     \
      for (_k = 0; _k < (c); _k++) {                   \
        (y)[_k] += cast((A)[(c) * _l + _k]) * (x)[_l]; \
      }                                                \
    }                                                  \
  } while (0)

#define MEHess(out,cf,etaB,etaD,dim,nc,cast)                      \
  do {                                                            \
    PetscInt _m, _j, _k;                                          \
    for (_m = 0; _m < (nc) * (dim) * (dim); _m++) (out)[_m] = 0.; \
    for (_j = 0; _j < (dim); _j++) {                              \
      for (_k = _j + 1; _k < (dim); _k++) {                       \
        PetscInt _ind = (1 << _j) + (1 << _k);                    \
        for (_m = 0; _m < (nc); _m++) {                           \
          PetscScalar c = (cf)[_m] * (etaB)[_ind] * (etaD)[_ind];   \
          (out)[(_m * (dim) + _k) * (dim) + _j] += cast(c);       \
          (out)[(_m * (dim) + _j) * (dim) + _k] += cast(c);       \
        }                                                         \
      }                                                           \
    }                                                             \
  } while (0)

static void MultilinearEvaluate(PetscInt dim, PetscReal (*coordRange)[2], PetscInt nc, PetscScalar *cf, PetscScalar *cfWork, PetscInt nPoints, const PetscScalar *points, PetscDataType datatype, void *B, void *D, void *H)
{
  PetscInt i, j, k, l, m;
  PetscInt  whol = 1 << dim;
  PetscInt  half = whol >> 1;

  PetscFunctionBeginHot;
  if (!B && !D && !H) PetscFunctionReturnVoid();
  for (i = 0; i < nPoints; i++) {
    const PetscScalar *point = &points[dim * i];
    PetscReal deta[3] = {0.};
    PetscReal etaB[8] = {1.,1.,1.,1.,1.,1.,1.,1.};
    PetscReal etaD[8] = {1.,1.,1.,1.,1.,1.,1.,1.};
    PetscReal work[8];

    for (j = 0; j < dim; j++) {
      PetscReal e, d;

      e = (PetscRealPart(point[j]) - coordRange[j][0]) / coordRange[j][1];
      deta[j] = d = 1. / coordRange[j][1];
      for (k = 0; k < whol; k++) {work[k] = etaB[k];}
      for (k = 0; k < half; k++) {
        etaB[k]        = work[2 * k] * e;
        etaB[k + half] = work[2 * k + 1];
      }
      if (H) {
        for (k = 0; k < whol; k++) {work[k] = etaD[k];}
        for (k = 0; k < half; k++) {
          etaD[k + half] = work[2 * k];
          etaD[k       ] = work[2 * k + 1] * d;
        }
      }
    }
    if (B) {
      if (datatype == PETSC_SCALAR) {
        PetscScalar *out = &((PetscScalar *)B)[nc * i];

        MEdot(out,cf,etaB,(1 << dim),nc,(PetscScalar));
      } else {
        PetscReal *out = &((PetscReal *)B)[nc * i];

        MEdot(out,cf,etaB,(1 << dim),nc,PetscRealPart);
      }
    }
    if (D) {
      if (datatype == PETSC_SCALAR) {
        PetscScalar *out = &((PetscScalar *)D)[nc * dim * i];

        for (m = 0; m < nc * dim; m++) out[m] = 0.;
      } else {
        PetscReal *out = &((PetscReal *)D)[nc * dim * i];

        for (m = 0; m < nc * dim; m++) out[m] = 0.;
      }
      for (j = 0; j < dim; j++) {
        PetscReal d = deta[j];

        for (k = 0; k < whol * nc; k++) {cfWork[k] = cf[k];}
        for (k = 0; k < whol; k++) {work[k] = etaB[k];}
        for (k = 0; k < half; k++) {
          PetscReal e;

          etaB[k]        =     work[2 * k];
          etaB[k + half] = e = work[2 * k + 1];

          for (l = 0; l < nc; l++) {
            cf[ k         * nc + l] = cfWork[ 2 * k      * nc + l];
            cf[(k + half) * nc + l] = cfWork[(2 * k + 1) * nc + l];
          }
          if (datatype == PETSC_SCALAR) {
            PetscScalar *out = &((PetscScalar *)D)[nc * dim * i];

            for (l = 0; l < nc; l++) {
              out[l * dim + j] += d * e * cf[k * nc + l];
            }
          } else {
            PetscReal *out = &((PetscReal *)D)[nc * dim * i];

            for (l = 0; l < nc; l++) {
              out[l * dim + j] += d * e * PetscRealPart(cf[k * nc + l]);
            }
          }
        }
      }
    }
    if (H) {
      if (datatype == PETSC_SCALAR) {
        PetscScalar *out = &((PetscScalar *)H)[nc * dim * dim * i];

        MEHess(out,cf,etaB,etaD,dim,nc,(PetscScalar));
      } else {
        PetscReal *out = &((PetscReal *)H)[nc * dim * dim * i];

        MEHess(out,cf,etaB,etaD,dim,nc,PetscRealPart);
      }
    }
  }
  PetscFunctionReturnVoid();
}

static PetscErrorCode DMFieldEvaluate_DA(DMField field, Vec points, PetscDataType datatype, void *B, void *D, void *H)
{
  DM             dm;
  DMField_DA     *dafield;
  PetscInt       dim;
  PetscInt       N, n, nc;
  const PetscScalar *array;
  PetscReal (*coordRange)[2];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  dm      = field->dm;
  nc      = field->numComponents;
  dafield = (DMField_DA *) field->data;
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  ierr = VecGetLocalSize(points,&N);CHKERRQ(ierr);
  if (N % dim) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Point vector size %D not divisible by coordinate dimension %D\n",N,dim);
  n = N / dim;
  coordRange = &(dafield->coordRange[0]);
  ierr = VecGetArrayRead(points,&array);CHKERRQ(ierr);
  MultilinearEvaluate(dim,coordRange,nc,dafield->cornerCoeffs,dafield->work,n,array,datatype,B,D,H);
  ierr = VecRestoreArrayRead(points,&array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMFieldEvaluateFE_DA(DMField field, IS cellIS, PetscQuadrature points, PetscDataType datatype, void *B, void *D, void *H)
{
  PetscInt       c, i, j, k, dim, cellsPer[3] = {0}, first[3] = {0}, whol, half;
  PetscReal      stepPer[3] = {0.};
  PetscReal      cellCoordRange[3][2] = {{0.,1.},{0.,1.},{0.,1.}};
  PetscScalar    *cellCoeffs, *work;
  DM             dm;
  DMDALocalInfo  info;
  PetscInt       cStart, cEnd;
  PetscInt       nq, nc;
  const PetscReal *q;
#if defined(PETSC_USE_COMPLEX)
  PetscScalar    *qs;
#else
  const PetscScalar *qs;
#endif
  DMField_DA     *dafield;
  PetscBool      isStride;
  const PetscInt *cells = NULL;
  PetscInt       sfirst = -1, stride = -1, nCells;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  dafield = (DMField_DA *) field->data;
  dm = field->dm;
  nc = field->numComponents;
  ierr = DMDAGetLocalInfo(dm,&info);CHKERRQ(ierr);
  dim = info.dim;
  work = dafield->work;
  stepPer[0] = 1./ info.mx;
  stepPer[1] = 1./ info.my;
  stepPer[2] = 1./ info.mz;
  first[0] = info.gxs;
  first[1] = info.gys;
  first[2] = info.gzs;
  cellsPer[0] = info.gxm;
  cellsPer[1] = info.gym;
  cellsPer[2] = info.gzm;
  /* TODO: probably take components into account */
  ierr = PetscQuadratureGetData(points, NULL, NULL, &nq, &q, NULL);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr = DMGetWorkArray(dm,nq * dim,MPIU_SCALAR,&qs);CHKERRQ(ierr);
  for (i = 0; i < nq * dim; i++) qs[i] = q[i];
#else
  qs = q;
#endif
  ierr = DMDAGetHeightStratum(dm,0,&cStart,&cEnd);CHKERRQ(ierr);
  ierr = DMGetWorkArray(dm,(1 << dim) * nc,MPIU_SCALAR,&cellCoeffs);CHKERRQ(ierr);
  whol = (1 << dim);
  half = whol >> 1;
  ierr = ISGetLocalSize(cellIS,&nCells);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)cellIS,ISSTRIDE,&isStride);CHKERRQ(ierr);
  if (isStride) {
    ierr = ISStrideGetInfo(cellIS,&sfirst,&stride);CHKERRQ(ierr);
  } else {
    ierr = ISGetIndices(cellIS,&cells);CHKERRQ(ierr);
  }
  for (c = 0; c < nCells; c++) {
    PetscInt  cell = isStride ? (sfirst + c * stride) : cells[c];
    PetscInt  rem  = cell;
    PetscInt  ijk[3] = {0};
    void *cB, *cD, *cH;

    if (datatype == PETSC_SCALAR) {
      cB = B ? &((PetscScalar *)B)[nc * nq * c] : NULL;
      cD = D ? &((PetscScalar *)D)[nc * nq * dim * c] : NULL;
      cH = H ? &((PetscScalar *)H)[nc * nq * dim * dim * c] : NULL;
    } else {
      cB = B ? &((PetscReal *)B)[nc * nq * c] : NULL;
      cD = D ? &((PetscReal *)D)[nc * nq * dim * c] : NULL;
      cH = H ? &((PetscReal *)H)[nc * nq * dim * dim * c] : NULL;
    }
    if (cell < cStart || cell >= cEnd) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Point %D not a cell [%D,%D), not implemented yet",cell,cStart,cEnd);
    for (i = 0; i < nc * whol; i++) {work[i] = dafield->cornerCoeffs[i];}
    for (j = 0; j < dim; j++) {
      PetscReal e, d;
      ijk[j] = (rem % cellsPer[j]);
      rem /= cellsPer[j];

      e = 2. * (ijk[j] + first[j] + 0.5) * stepPer[j] - 1.;
      d = stepPer[j];
      for (i = 0; i < half; i++) {
        for (k = 0; k < nc; k++) {
          cellCoeffs[ i         * nc + k] = work[ 2 * i * nc + k] * d;
          cellCoeffs[(i + half) * nc + k] = work[ 2 * i * nc + k] * e + work[(2 * i + 1) * nc + k];
        }
      }
      for (i = 0; i < whol * nc; i++) {work[i] = cellCoeffs[i];}
    }
    MultilinearEvaluate(dim,cellCoordRange,nc,cellCoeffs,dafield->work,nq,qs,datatype,cB,cD,cH);
  }
  if (!isStride) {
    ierr = ISRestoreIndices(cellIS,&cells);CHKERRQ(ierr);
  }
  ierr = DMRestoreWorkArray(dm,(1 << dim) * nc,MPIU_SCALAR,&cellCoeffs);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr = DMRestoreWorkArray(dm,nq * dim,MPIU_SCALAR,&qs);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

static PetscErrorCode DMFieldEvaluateFV_DA(DMField field, IS cellIS, PetscDataType datatype, void *B, void *D, void *H)
{
  PetscInt       c, i, dim, cellsPer[3] = {0}, first[3] = {0};
  PetscReal      stepPer[3] = {0.};
  DM             dm;
  DMDALocalInfo  info;
  PetscInt       cStart, cEnd, numCells;
  PetscInt       nc;
  PetscScalar    *points;
  DMField_DA     *dafield;
  PetscBool      isStride;
  const PetscInt *cells = NULL;
  PetscInt       sfirst = -1, stride = -1;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  dafield = (DMField_DA *) field->data;
  dm = field->dm;
  nc = field->numComponents;
  ierr = DMDAGetLocalInfo(dm,&info);CHKERRQ(ierr);
  dim = info.dim;
  stepPer[0] = 1./ info.mx;
  stepPer[1] = 1./ info.my;
  stepPer[2] = 1./ info.mz;
  first[0] = info.gxs;
  first[1] = info.gys;
  first[2] = info.gzs;
  cellsPer[0] = info.gxm;
  cellsPer[1] = info.gym;
  cellsPer[2] = info.gzm;
  ierr = DMDAGetHeightStratum(dm,0,&cStart,&cEnd);CHKERRQ(ierr);
  ierr = ISGetLocalSize(cellIS,&numCells);CHKERRQ(ierr);
  ierr = DMGetWorkArray(dm,dim * numCells,MPIU_SCALAR,&points);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)cellIS,ISSTRIDE,&isStride);CHKERRQ(ierr);
  if (isStride) {
    ierr = ISStrideGetInfo(cellIS,&sfirst,&stride);CHKERRQ(ierr);
  } else {
    ierr = ISGetIndices(cellIS,&cells);CHKERRQ(ierr);
  }
  for (c = 0; c < numCells; c++) {
    PetscInt  cell = isStride ? (sfirst + c * stride) : cells[c];
    PetscInt  rem  = cell;
    PetscInt  ijk[3] = {0};

    if (cell < cStart || cell >= cEnd) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Point %D not a cell [%D,%D), not implemented yet",cell,cStart,cEnd);
    for (i = 0; i < dim; i++) {
      ijk[i] = (rem % cellsPer[i]);
      rem /= cellsPer[i];
      points[dim * c + i] = (ijk[i] + first[i] + 0.5) * stepPer[i];
    }
  }
  if (!isStride) {
    ierr = ISRestoreIndices(cellIS,&cells);CHKERRQ(ierr);
  }
  MultilinearEvaluate(dim,dafield->coordRange,nc,dafield->cornerCoeffs,dafield->work,numCells,points,datatype,B,D,H);
  ierr = DMRestoreWorkArray(dm,dim * numCells,MPIU_SCALAR,&points);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMFieldGetDegree_DA(DMField field, IS pointIS, PetscInt *minDegree, PetscInt *maxDegree)
{
  DM             dm;
  PetscInt       dim, h, imin;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  dm = field->dm;
  ierr = ISGetMinMax(pointIS,&imin,NULL);CHKERRQ(ierr);
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  for (h = 0; h <= dim; h++) {
    PetscInt hEnd;

    ierr = DMDAGetHeightStratum(dm,h,NULL,&hEnd);CHKERRQ(ierr);
    if (imin < hEnd) break;
  }
  dim -= h;
  if (minDegree) *minDegree = 1;
  if (maxDegree) *maxDegree = dim;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMFieldCreateDefaultQuadrature_DA(DMField field, IS cellIS, PetscQuadrature *quad)
{
  PetscInt       h, dim, imax, imin;
  DM             dm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  dm = field->dm;
  ierr = ISGetMinMax(cellIS,&imin,&imax);CHKERRQ(ierr);
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  *quad = NULL;
  for (h = 0; h <= dim; h++) {
    PetscInt hStart, hEnd;

    ierr = DMDAGetHeightStratum(dm,h,&hStart,&hEnd);CHKERRQ(ierr);
    if (imin >= hStart && imax < hEnd) break;
  }
  dim -= h;
  if (dim > 0) {
    ierr = PetscDTGaussTensorQuadrature(dim, 1, 1, -1.0, 1.0, quad);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode DMFieldInitialize_DA(DMField field)
{
  DM             dm;
  Vec            coords = NULL;
  PetscInt       dim, i, j, k;
  DMField_DA     *dafield = (DMField_DA *) field->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  field->ops->destroy                 = DMFieldDestroy_DA;
  field->ops->evaluate                = DMFieldEvaluate_DA;
  field->ops->evaluateFE              = DMFieldEvaluateFE_DA;
  field->ops->evaluateFV              = DMFieldEvaluateFV_DA;
  field->ops->getDegree               = DMFieldGetDegree_DA;
  field->ops->createDefaultQuadrature = DMFieldCreateDefaultQuadrature_DA;
  field->ops->view                    = DMFieldView_DA;
  dm = field->dm;
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  if (dm->coordinates) coords = dm->coordinates;
  else if (dm->coordinatesLocal) coords = dm->coordinatesLocal;
  if (coords) {
    PetscInt          n;
    const PetscScalar *array;
    PetscReal         mins[3][2] = {{PETSC_MAX_REAL,PETSC_MAX_REAL},{PETSC_MAX_REAL,PETSC_MAX_REAL},{PETSC_MAX_REAL,PETSC_MAX_REAL}};

    ierr = VecGetLocalSize(coords,&n);CHKERRQ(ierr);
    n /= dim;
    ierr = VecGetArrayRead(coords,&array);CHKERRQ(ierr);
    for (i = 0, k = 0; i < n; i++) {
      for (j = 0; j < dim; j++, k++) {
        PetscReal val = PetscRealPart(array[k]);

        mins[j][0] = PetscMin(mins[j][0],val);
        mins[j][1] = PetscMin(mins[j][1],-val);
      }
    }
    ierr = VecRestoreArrayRead(coords,&array);CHKERRQ(ierr);
    ierr = MPIU_Allreduce((PetscReal *) mins,&(dafield->coordRange[0][0]),2*dim,MPIU_REAL,MPI_MIN,PetscObjectComm((PetscObject)dm));CHKERRMPI(ierr);
    for (j = 0; j < dim; j++) {
      dafield->coordRange[j][1] = -dafield->coordRange[j][1];
    }
  } else {
    for (j = 0; j < dim; j++) {
      dafield->coordRange[j][0] = 0.;
      dafield->coordRange[j][1] = 1.;
    }
  }
  for (j = 0; j < dim; j++) {
    PetscReal avg = 0.5 * (dafield->coordRange[j][1] + dafield->coordRange[j][0]);
    PetscReal dif = 0.5 * (dafield->coordRange[j][1] - dafield->coordRange[j][0]);

    dafield->coordRange[j][0] = avg;
    dafield->coordRange[j][1] = dif;
  }
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode DMFieldCreate_DA(DMField field)
{
  DMField_DA     *dafield;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(field,&dafield);CHKERRQ(ierr);
  field->data = dafield;
  ierr = DMFieldInitialize_DA(field);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMFieldCreateDA(DM dm, PetscInt nc, const PetscScalar *cornerValues,DMField *field)
{
  DMField        b;
  DMField_DA     *dafield;
  PetscInt       dim, nv, i, j, k;
  PetscInt       half;
  PetscScalar    *cv, *cf, *work;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMFieldCreate(dm,nc,DMFIELD_VERTEX,&b);CHKERRQ(ierr);
  ierr = DMFieldSetType(b,DMFIELDDA);CHKERRQ(ierr);
  dafield = (DMField_DA *) b->data;
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  nv = (1 << dim) * nc;
  ierr = PetscMalloc3(nv,&cv,nv,&cf,nv,&work);CHKERRQ(ierr);
  for (i = 0; i < nv; i++) cv[i] = cornerValues[i];
  for (i = 0; i < nv; i++) cf[i] = cv[i];
  dafield->cornerVals = cv;
  dafield->cornerCoeffs = cf;
  dafield->work = work;
  half = (1 << (dim - 1));
  for (i = 0; i < dim; i++) {
    PetscScalar *w;

    w = work;
    for (j = 0; j < half; j++) {
      for (k = 0; k < nc; k++) {
        w[j * nc + k] = 0.5 * (cf[(2 * j + 1) * nc + k] - cf[(2 * j) * nc + k]);
      }
    }
    w = &work[j * nc];
    for (j = 0; j < half; j++) {
      for (k = 0; k < nc; k++) {
        w[j * nc + k] = 0.5 * (cf[(2 * j) * nc + k] + cf[(2 * j + 1) * nc + k]);
      }
    }
    for (j = 0; j < nv; j++) {cf[j] = work[j];}
  }
  *field = b;
  PetscFunctionReturn(0);
}
