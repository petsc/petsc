
/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include <petsc/private/dmdaimpl.h>    /*I   "petscdmda.h"   I*/
#include <petscbt.h>
#include <petscsf.h>
#include <petscds.h>
#include <petscfe.h>

/*
   This allows the DMDA vectors to properly tell MATLAB their dimensions
*/
#if defined(PETSC_HAVE_MATLAB_ENGINE)
#include <engine.h>   /* MATLAB include file */
#include <mex.h>      /* MATLAB include file */
static PetscErrorCode  VecMatlabEnginePut_DA2d(PetscObject obj,void *mengine)
{
  PetscErrorCode ierr;
  PetscInt       n,m;
  Vec            vec = (Vec)obj;
  PetscScalar    *array;
  mxArray        *mat;
  DM             da;

  PetscFunctionBegin;
  ierr = VecGetDM(vec, &da);CHKERRQ(ierr);
  if (!da) SETERRQ(PetscObjectComm((PetscObject)vec),PETSC_ERR_ARG_WRONGSTATE,"Vector not associated with a DMDA");
  ierr = DMDAGetGhostCorners(da,0,0,0,&m,&n,0);CHKERRQ(ierr);

  ierr = VecGetArray(vec,&array);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  mat = mxCreateDoubleMatrix(m,n,mxREAL);
#else
  mat = mxCreateDoubleMatrix(m,n,mxCOMPLEX);
#endif
  ierr = PetscMemcpy(mxGetPr(mat),array,n*m*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscObjectName(obj);CHKERRQ(ierr);
  engPutVariable((Engine*)mengine,obj->name,mat);

  ierr = VecRestoreArray(vec,&array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif


PetscErrorCode  DMCreateLocalVector_DA(DM da,Vec *g)
{
  PetscErrorCode ierr;
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  PetscValidPointer(g,2);
  if (da->defaultSection) {
    ierr = DMCreateLocalVector_Section_Private(da,g);CHKERRQ(ierr);
  } else {
    ierr = VecCreate(PETSC_COMM_SELF,g);CHKERRQ(ierr);
    ierr = VecSetSizes(*g,dd->nlocal,PETSC_DETERMINE);CHKERRQ(ierr);
    ierr = VecSetBlockSize(*g,dd->w);CHKERRQ(ierr);
    ierr = VecSetType(*g,da->vectype);CHKERRQ(ierr);
    ierr = VecSetDM(*g, da);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MATLAB_ENGINE)
    if (dd->w == 1  && da->dim == 2) {
      ierr = PetscObjectComposeFunction((PetscObject)*g,"PetscMatlabEnginePut_C",VecMatlabEnginePut_DA2d);CHKERRQ(ierr);
    }
#endif
  }
  PetscFunctionReturn(0);
}

/*@
  DMDAGetNumCells - Get the number of cells in the local piece of the DMDA. This includes ghost cells.

  Input Parameter:
. dm - The DM object

  Output Parameters:
+ numCellsX - The number of local cells in the x-direction
. numCellsY - The number of local cells in the y-direction
. numCellsZ - The number of local cells in the z-direction
- numCells - The number of local cells

  Level: developer

.seealso: DMDAGetCellPoint()
@*/
PetscErrorCode DMDAGetNumCells(DM dm, PetscInt *numCellsX, PetscInt *numCellsY, PetscInt *numCellsZ, PetscInt *numCells)
{
  DM_DA         *da  = (DM_DA*) dm->data;
  const PetscInt dim = dm->dim;
  const PetscInt mx  = (da->Xe - da->Xs)/da->w, my = da->Ye - da->Ys, mz = da->Ze - da->Zs;
  const PetscInt nC  = (mx)*(dim > 1 ? (my)*(dim > 2 ? (mz) : 1) : 1);

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1,DMDA);
  if (numCellsX) {
    PetscValidIntPointer(numCellsX,2);
    *numCellsX = mx;
  }
  if (numCellsY) {
    PetscValidIntPointer(numCellsY,3);
    *numCellsY = my;
  }
  if (numCellsZ) {
    PetscValidIntPointer(numCellsZ,4);
    *numCellsZ = mz;
  }
  if (numCells) {
    PetscValidIntPointer(numCells,5);
    *numCells = nC;
  }
  PetscFunctionReturn(0);
}

/*@
  DMDAGetCellPoint - Get the DM point corresponding to the tuple (i, j, k) in the DMDA

  Input Parameters:
+ dm - The DM object
- i,j,k - The global indices for the cell

  Output Parameters:
. point - The local DM point

  Level: developer

.seealso: DMDAGetNumCells()
@*/
PetscErrorCode DMDAGetCellPoint(DM dm, PetscInt i, PetscInt j, PetscInt k, PetscInt *point)
{
  const PetscInt dim = dm->dim;
  DMDALocalInfo  info;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1,DMDA);
  PetscValidIntPointer(point,5);
  ierr = DMDAGetLocalInfo(dm, &info);CHKERRQ(ierr);
  if (dim > 0) {if ((i < info.gxs) || (i >= info.gxs+info.gxm)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "X index %d not in [%d, %d)", i, info.gxs, info.gxs+info.gxm);}
  if (dim > 1) {if ((j < info.gys) || (j >= info.gys+info.gym)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Y index %d not in [%d, %d)", j, info.gys, info.gys+info.gym);}
  if (dim > 2) {if ((k < info.gzs) || (k >= info.gzs+info.gzm)) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Z index %d not in [%d, %d)", k, info.gzs, info.gzs+info.gzm);}
  *point = i + (dim > 1 ? (j + (dim > 2 ? k*info.gym : 0))*info.gxm : 0);
  PetscFunctionReturn(0);
}

PetscErrorCode DMDAGetNumVertices(DM dm, PetscInt *numVerticesX, PetscInt *numVerticesY, PetscInt *numVerticesZ, PetscInt *numVertices)
{
  DM_DA          *da = (DM_DA*) dm->data;
  const PetscInt dim = dm->dim;
  const PetscInt mx  = (da->Xe - da->Xs)/da->w, my = da->Ye - da->Ys, mz = da->Ze - da->Zs;
  const PetscInt nVx = mx+1;
  const PetscInt nVy = dim > 1 ? (my+1) : 1;
  const PetscInt nVz = dim > 2 ? (mz+1) : 1;
  const PetscInt nV  = nVx*nVy*nVz;

  PetscFunctionBegin;
  if (numVerticesX) {
    PetscValidIntPointer(numVerticesX,2);
    *numVerticesX = nVx;
  }
  if (numVerticesY) {
    PetscValidIntPointer(numVerticesY,3);
    *numVerticesY = nVy;
  }
  if (numVerticesZ) {
    PetscValidIntPointer(numVerticesZ,4);
    *numVerticesZ = nVz;
  }
  if (numVertices) {
    PetscValidIntPointer(numVertices,5);
    *numVertices = nV;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMDAGetNumFaces(DM dm, PetscInt *numXFacesX, PetscInt *numXFaces, PetscInt *numYFacesY, PetscInt *numYFaces, PetscInt *numZFacesZ, PetscInt *numZFaces)
{
  DM_DA          *da = (DM_DA*) dm->data;
  const PetscInt dim = dm->dim;
  const PetscInt mx  = (da->Xe - da->Xs)/da->w, my = da->Ye - da->Ys, mz = da->Ze - da->Zs;
  const PetscInt nxF = (dim > 1 ? (my)*(dim > 2 ? (mz) : 1) : 1);
  const PetscInt nXF = (mx+1)*nxF;
  const PetscInt nyF = mx*(dim > 2 ? mz : 1);
  const PetscInt nYF = dim > 1 ? (my+1)*nyF : 0;
  const PetscInt nzF = mx*(dim > 1 ? my : 0);
  const PetscInt nZF = dim > 2 ? (mz+1)*nzF : 0;

  PetscFunctionBegin;
  if (numXFacesX) {
    PetscValidIntPointer(numXFacesX,2);
    *numXFacesX = nxF;
  }
  if (numXFaces) {
    PetscValidIntPointer(numXFaces,3);
    *numXFaces = nXF;
  }
  if (numYFacesY) {
    PetscValidIntPointer(numYFacesY,4);
    *numYFacesY = nyF;
  }
  if (numYFaces) {
    PetscValidIntPointer(numYFaces,5);
    *numYFaces = nYF;
  }
  if (numZFacesZ) {
    PetscValidIntPointer(numZFacesZ,6);
    *numZFacesZ = nzF;
  }
  if (numZFaces) {
    PetscValidIntPointer(numZFaces,7);
    *numZFaces = nZF;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMDAGetHeightStratum(DM dm, PetscInt height, PetscInt *pStart, PetscInt *pEnd)
{
  const PetscInt dim = dm->dim;
  PetscInt       nC, nV, nXF, nYF, nZF;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (pStart) PetscValidIntPointer(pStart,3);
  if (pEnd)   PetscValidIntPointer(pEnd,4);
  ierr = DMDAGetNumCells(dm, NULL, NULL, NULL, &nC);CHKERRQ(ierr);
  ierr = DMDAGetNumVertices(dm, NULL, NULL, NULL, &nV);CHKERRQ(ierr);
  ierr = DMDAGetNumFaces(dm, NULL, &nXF, NULL, &nYF, NULL, &nZF);CHKERRQ(ierr);
  if (height == 0) {
    /* Cells */
    if (pStart) *pStart = 0;
    if (pEnd)   *pEnd   = nC;
  } else if (height == 1) {
    /* Faces */
    if (pStart) *pStart = nC+nV;
    if (pEnd)   *pEnd   = nC+nV+nXF+nYF+nZF;
  } else if (height == dim) {
    /* Vertices */
    if (pStart) *pStart = nC;
    if (pEnd)   *pEnd   = nC+nV;
  } else if (height < 0) {
    /* All points */
    if (pStart) *pStart = 0;
    if (pEnd)   *pEnd   = nC+nV+nXF+nYF+nZF;
  } else SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "No points of height %d in the DA", height);
  PetscFunctionReturn(0);
}

PetscErrorCode DMDAGetDepthStratum(DM dm, PetscInt depth, PetscInt *pStart, PetscInt *pEnd)
{
  const PetscInt dim = dm->dim;
  PetscInt       nC, nV, nXF, nYF, nZF;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (pStart) PetscValidIntPointer(pStart,3);
  if (pEnd)   PetscValidIntPointer(pEnd,4);
  ierr = DMDAGetNumCells(dm, NULL, NULL, NULL, &nC);CHKERRQ(ierr);
  ierr = DMDAGetNumVertices(dm, NULL, NULL, NULL, &nV);CHKERRQ(ierr);
  ierr = DMDAGetNumFaces(dm, NULL, &nXF, NULL, &nYF, NULL, &nZF);CHKERRQ(ierr);
  if (depth == dim) {
    /* Cells */
    if (pStart) *pStart = 0;
    if (pEnd)   *pEnd   = nC;
  } else if (depth == dim-1) {
    /* Faces */
    if (pStart) *pStart = nC+nV;
    if (pEnd)   *pEnd   = nC+nV+nXF+nYF+nZF;
  } else if (depth == 0) {
    /* Vertices */
    if (pStart) *pStart = nC;
    if (pEnd)   *pEnd   = nC+nV;
  } else if (depth < 0) {
    /* All points */
    if (pStart) *pStart = 0;
    if (pEnd)   *pEnd   = nC+nV+nXF+nYF+nZF;
  } else SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "No points of depth %d in the DA", depth);
  PetscFunctionReturn(0);
}

PetscErrorCode DMDAGetConeSize(DM dm, PetscInt p, PetscInt *coneSize)
{
  const PetscInt dim = dm->dim;
  PetscInt       nC, nV, nXF, nYF, nZF;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *coneSize = 0;
  ierr = DMDAGetNumCells(dm, NULL, NULL, NULL, &nC);CHKERRQ(ierr);
  ierr = DMDAGetNumVertices(dm, NULL, NULL, NULL, &nV);CHKERRQ(ierr);
  ierr = DMDAGetNumFaces(dm, NULL, &nXF, NULL, &nYF, NULL, &nZF);CHKERRQ(ierr);
  switch (dim) {
  case 2:
    if (p >= 0) {
      if (p < nC) {
        *coneSize = 4;
      } else if (p < nC+nV) {
        *coneSize = 0;
      } else if (p < nC+nV+nXF+nYF+nZF) {
        *coneSize = 2;
      } else SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Point %d should be in [0, %d)", p, nC+nV+nXF+nYF+nZF);
    } else SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Negative point %d is invalid", p);
    break;
  case 3:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Too lazy to do 3D");
    break;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMDAGetCone(DM dm, PetscInt p, PetscInt *cone[])
{
  const PetscInt dim = dm->dim;
  PetscInt       nCx, nCy, nCz, nC, nVx, nVy, nVz, nV, nxF, nyF, nzF, nXF, nYF, nZF;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!cone) {ierr = DMGetWorkArray(dm, 6, MPIU_INT, cone);CHKERRQ(ierr);}
  ierr = DMDAGetNumCells(dm, &nCx, &nCy, &nCz, &nC);CHKERRQ(ierr);
  ierr = DMDAGetNumVertices(dm, &nVx, &nVy, &nVz, &nV);CHKERRQ(ierr);
  ierr = DMDAGetNumFaces(dm, &nxF, &nXF, &nyF, &nYF, &nzF, &nZF);CHKERRQ(ierr);
  switch (dim) {
  case 2:
    if (p >= 0) {
      if (p < nC) {
        const PetscInt cy = p / nCx;
        const PetscInt cx = p % nCx;

        (*cone)[0] = cy*nxF + cx + nC+nV;
        (*cone)[1] = cx*nyF + cy + nyF + nC+nV+nXF;
        (*cone)[2] = cy*nxF + cx + nxF + nC+nV;
        (*cone)[3] = cx*nyF + cy + nC+nV+nXF;
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Too lazy to do cell cones");
      } else if (p < nC+nV) {
      } else if (p < nC+nV+nXF) {
        const PetscInt fy = (p - nC+nV) / nxF;
        const PetscInt fx = (p - nC+nV) % nxF;

        (*cone)[0] = fy*nVx + fx + nC;
        (*cone)[1] = fy*nVx + fx + 1 + nC;
      } else if (p < nC+nV+nXF+nYF) {
        const PetscInt fx = (p - nC+nV+nXF) / nyF;
        const PetscInt fy = (p - nC+nV+nXF) % nyF;

        (*cone)[0] = fy*nVx + fx + nC;
        (*cone)[1] = fy*nVx + fx + nVx + nC;
      } else SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Point %d should be in [0, %d)", p, nC+nV+nXF+nYF+nZF);
    } else SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Negative point %d is invalid", p);
    break;
  case 3:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Too lazy to do 3D");
    break;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMDARestoreCone(DM dm, PetscInt p, PetscInt *cone[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetWorkArray(dm, 6, MPIU_INT, cone);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMDASetVertexCoordinates(DM dm, PetscReal xl, PetscReal xu, PetscReal yl, PetscReal yu, PetscReal zl, PetscReal zu)
{
  DM_DA         *da = (DM_DA *) dm->data;
  Vec            coordinates;
  PetscSection   section;
  PetscScalar   *coords;
  PetscReal      h[3];
  PetscInt       dim, size, M, N, P, nVx, nVy, nVz, nV, vStart, vEnd, v, i, j, k;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1,DMDA);
  ierr = DMDAGetInfo(dm, &dim, &M, &N, &P, 0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  if (dim > 3) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_PLIB,"The following code only works for dim <= 3");
  h[0] = (xu - xl)/M;
  h[1] = (yu - yl)/N;
  h[2] = (zu - zl)/P;
  ierr = DMDAGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMDAGetNumVertices(dm, &nVx, &nVy, &nVz, &nV);CHKERRQ(ierr);
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject) dm), &section);CHKERRQ(ierr);
  ierr = PetscSectionSetNumFields(section, 1);CHKERRQ(ierr);
  ierr = PetscSectionSetFieldComponents(section, 0, dim);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(section, vStart, vEnd);CHKERRQ(ierr);
  for (v = vStart; v < vEnd; ++v) {
    ierr = PetscSectionSetDof(section, v, dim);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(section);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(section, &size);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF, size, &coordinates);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)coordinates,"coordinates");CHKERRQ(ierr);
  ierr = VecGetArray(coordinates, &coords);CHKERRQ(ierr);
  for (k = 0; k < nVz; ++k) {
    PetscInt ind[3], d, off;

    ind[0] = 0;
    ind[1] = 0;
    ind[2] = k + da->zs;
    for (j = 0; j < nVy; ++j) {
      ind[1] = j + da->ys;
      for (i = 0; i < nVx; ++i) {
        const PetscInt vertex = (k*nVy + j)*nVx + i + vStart;

        ierr = PetscSectionGetOffset(section, vertex, &off);CHKERRQ(ierr);
        ind[0] = i + da->xs;
        for (d = 0; d < dim; ++d) {
          coords[off+d] = h[d]*ind[d];
        }
      }
    }
  }
  ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);
  ierr = DMSetCoordinateSection(dm, PETSC_DETERMINE, section);CHKERRQ(ierr);
  ierr = DMSetCoordinatesLocal(dm, coordinates);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&section);CHKERRQ(ierr);
  ierr = VecDestroy(&coordinates);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMProjectFunctionLocal_DA(DM dm, PetscReal time, PetscErrorCode (**funcs)(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void *), void **ctxs, InsertMode mode, Vec localX)
{
  PetscDS         prob;
  PetscFE         fe;
  PetscDualSpace  sp;
  PetscQuadrature q;
  PetscSection    section;
  PetscScalar    *values;
  PetscInt        numFields, Nc, dim, dimEmbed, spDim, totDim, numValues, cStart, cEnd, f, c, v, d;
  PetscFEGeom    *geom;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = PetscDSGetTotalDimension(prob, &totDim);CHKERRQ(ierr);
  ierr = PetscDSGetDiscretization(prob, 0, (PetscObject *) &fe);CHKERRQ(ierr);
  ierr = PetscFEGetQuadrature(fe, &q);CHKERRQ(ierr);
  ierr = DMGetSection(dm, &section);CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(section, &numFields);CHKERRQ(ierr);
  ierr = DMDAGetInfo(dm, &dim,0,0,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  ierr = DMGetCoordinateDim(dm, &dimEmbed);CHKERRQ(ierr);
  ierr = DMDAGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMDAVecGetClosure(dm, section, localX, cStart, &numValues, NULL);CHKERRQ(ierr);
  if (numValues != totDim) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "The section cell closure size %d != dual space dimension %d", numValues, totDim);
  ierr = DMGetWorkArray(dm, numValues, MPIU_SCALAR, &values);CHKERRQ(ierr);
  ierr = PetscFEGeomCreate(q,1,dimEmbed,PETSC_FALSE,&geom);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    ierr          = DMDAComputeCellGeometryFEM(dm, c, q, geom->v, geom->J, NULL, geom->detJ);CHKERRQ(ierr);
    for (f = 0, v = 0; f < numFields; ++f) {
      void * const ctx = ctxs ? ctxs[f] : NULL;

      ierr = PetscDSGetDiscretization(prob, f, (PetscObject *) &fe);CHKERRQ(ierr);
      ierr = PetscFEGetNumComponents(fe, &Nc);CHKERRQ(ierr);
      ierr = PetscFEGetDualSpace(fe, &sp);CHKERRQ(ierr);
      ierr = PetscDualSpaceGetDimension(sp, &spDim);CHKERRQ(ierr);
      for (d = 0; d < spDim; ++d) {
        ierr = PetscDualSpaceApply(sp, d, time, geom, Nc, funcs[f], ctx, &values[v]);CHKERRQ(ierr);
      }
    }
    ierr = DMDAVecSetClosure(dm, section, localX, c, values, mode);CHKERRQ(ierr);
  }
  ierr = PetscFEGeomDestroy(&geom);CHKERRQ(ierr);
  ierr = DMRestoreWorkArray(dm, numValues, MPIU_SCALAR, &values);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMComputeL2Diff_DA(DM dm, PetscReal time, PetscErrorCode (**funcs)(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void *), void **ctxs, Vec X, PetscReal *diff)
{
  const PetscInt  debug = 0;
  PetscDS         prob;
  PetscFE         fe;
  PetscQuadrature quad;
  PetscSection    section;
  Vec             localX;
  PetscScalar    *funcVal;
  PetscReal      *coords, *v0, *J, *invJ, detJ;
  PetscReal       localDiff = 0.0;
  PetscInt        dim, numFields, Nc, cStart, cEnd, c, field, fieldOffset, comp;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = DMDAGetInfo(dm, &dim,0,0,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = PetscDSGetTotalComponents(prob, &Nc);CHKERRQ(ierr);
  ierr = PetscDSGetDiscretization(prob, 0, (PetscObject *) &fe);CHKERRQ(ierr);
  ierr = PetscFEGetQuadrature(fe, &quad);CHKERRQ(ierr);
  ierr = DMGetSection(dm, &section);CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(section, &numFields);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm, &localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm, X, INSERT_VALUES, localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm, X, INSERT_VALUES, localX);CHKERRQ(ierr);
  /* There are no BC values in DAs right now: ierr = DMDAProjectFunctionLocal(dm, fe, funcs, INSERT_BC_VALUES, localX);CHKERRQ(ierr); */
  ierr = PetscMalloc5(Nc,&funcVal,dim,&coords,dim,&v0,dim*dim,&J,dim*dim,&invJ);CHKERRQ(ierr);
  ierr = DMDAGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    PetscScalar *x = NULL;
    PetscReal    elemDiff = 0.0;

    ierr = DMDAComputeCellGeometryFEM(dm, c, quad, v0, J, invJ, &detJ);CHKERRQ(ierr);
    if (detJ <= 0.0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %d", detJ, c);
    ierr = DMDAVecGetClosure(dm, NULL, localX, c, NULL, &x);CHKERRQ(ierr);

    for (field = 0, comp = 0, fieldOffset = 0; field < numFields; ++field) {
      void * const ctx = ctxs ? ctxs[field] : NULL;
      const PetscReal *quadPoints, *quadWeights;
      PetscReal       *basis;
      PetscInt         qNc, Nq, Nb, Nc, q, d, e, fc, f;

      ierr = PetscDSGetDiscretization(prob, field, (PetscObject *) &fe);CHKERRQ(ierr);
      ierr = PetscQuadratureGetData(quad, NULL, &qNc, &Nq, &quadPoints, &quadWeights);CHKERRQ(ierr);
      ierr = PetscFEGetDimension(fe, &Nb);CHKERRQ(ierr);
      ierr = PetscFEGetNumComponents(fe, &Nc);CHKERRQ(ierr);
      if (qNc != Nc) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_SUP, "Quadrature components %D != %D field components\n", qNc, Nc);
      ierr = PetscFEGetDefaultTabulation(fe, &basis, NULL, NULL);CHKERRQ(ierr);
      if (debug) {
        char title[1024];
        ierr = PetscSNPrintf(title, 1023, "Solution for Field %d", field);CHKERRQ(ierr);
        ierr = DMPrintCellVector(c, title, Nb, &x[fieldOffset]);CHKERRQ(ierr);
      }
      for (q = 0; q < Nq; ++q) {
        for (d = 0; d < dim; d++) {
          coords[d] = v0[d];
          for (e = 0; e < dim; e++) {
            coords[d] += J[d*dim+e]*(quadPoints[q*dim+e] + 1.0);
          }
        }
        ierr = (*funcs[field])(dim, time, coords, numFields, funcVal, ctx);CHKERRQ(ierr);
        for (fc = 0; fc < Nc; ++fc) {
          PetscScalar interpolant = 0.0;

          for (f = 0; f < Nb; ++f) {
            interpolant += x[fieldOffset+f]*basis[(q*Nb+f)*Nc+fc];
          }
          if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, "    elem %d field %d diff %g\n", c, field, PetscSqr(PetscRealPart(interpolant - funcVal[fc]))*quadWeights[q*Nc+fc]*detJ);CHKERRQ(ierr);}
          elemDiff += PetscSqr(PetscRealPart(interpolant - funcVal[fc]))*quadWeights[q*Nc+c]*detJ;
        }
      }
      comp        += Nc;
      fieldOffset += Nb;
    }
    ierr = DMDAVecRestoreClosure(dm, NULL, localX, c, NULL, &x);CHKERRQ(ierr);
    if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, "  elem %d diff %g\n", c, elemDiff);CHKERRQ(ierr);}
    localDiff += elemDiff;
  }
  ierr  = PetscFree5(funcVal,coords,v0,J,invJ);CHKERRQ(ierr);
  ierr  = DMRestoreLocalVector(dm, &localX);CHKERRQ(ierr);
  ierr  = MPIU_Allreduce(&localDiff, diff, 1, MPIU_REAL, MPIU_SUM, PetscObjectComm((PetscObject)dm));CHKERRQ(ierr);
  *diff = PetscSqrtReal(*diff);
  PetscFunctionReturn(0);
}

PetscErrorCode DMComputeL2GradientDiff_DA(DM dm, PetscReal time, PetscErrorCode (**funcs)(PetscInt, PetscReal, const PetscReal [], const PetscReal [], PetscInt, PetscScalar *, void *), void **ctxs, Vec X, const PetscReal n[], PetscReal *diff)
{
  const PetscInt  debug = 0;
  PetscDS         prob;
  PetscFE         fe;
  PetscQuadrature quad;
  PetscSection    section;
  Vec             localX;
  PetscScalar    *funcVal, *interpolantVec;
  PetscReal      *coords, *realSpaceDer, *v0, *J, *invJ, detJ;
  PetscReal       localDiff = 0.0;
  PetscInt        dim, numFields, Nc, cStart, cEnd, c, field, fieldOffset, comp;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = DMDAGetInfo(dm, &dim,0,0,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = PetscDSGetTotalComponents(prob, &Nc);CHKERRQ(ierr);
  ierr = PetscDSGetDiscretization(prob, 0, (PetscObject *) &fe);CHKERRQ(ierr);
  ierr = PetscFEGetQuadrature(fe, &quad);CHKERRQ(ierr);
  ierr = DMGetSection(dm, &section);CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(section, &numFields);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm, &localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm, X, INSERT_VALUES, localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm, X, INSERT_VALUES, localX);CHKERRQ(ierr);
  /* There are no BC values in DAs right now: ierr = DMDAProjectFunctionLocal(dm, fe, funcs, INSERT_BC_VALUES, localX);CHKERRQ(ierr); */
  ierr = PetscMalloc7(Nc,&funcVal,dim,&coords,dim,&realSpaceDer,dim,&v0,dim*dim,&J,dim*dim,&invJ,dim,&interpolantVec);CHKERRQ(ierr);
  ierr = DMDAGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    PetscScalar *x = NULL;
    PetscReal    elemDiff = 0.0;

    ierr = DMDAComputeCellGeometryFEM(dm, c, quad, v0, J, invJ, &detJ);CHKERRQ(ierr);
    if (detJ <= 0.0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %d", detJ, c);
    ierr = DMDAVecGetClosure(dm, NULL, localX, c, NULL, &x);CHKERRQ(ierr);

    for (field = 0, comp = 0, fieldOffset = 0; field < numFields; ++field) {
      void * const ctx = ctxs ? ctxs[field] : NULL;
      const PetscReal *quadPoints, *quadWeights;
      PetscReal       *basisDer;
      PetscInt         qNc, Nq, Nb, Nc, q, d, e, fc, f, g;

      ierr = PetscDSGetDiscretization(prob, field, (PetscObject *) &fe);CHKERRQ(ierr);
      ierr = PetscQuadratureGetData(quad, NULL, &qNc, &Nq, &quadPoints, &quadWeights);CHKERRQ(ierr);
      ierr = PetscFEGetDimension(fe, &Nb);CHKERRQ(ierr);
      ierr = PetscFEGetNumComponents(fe, &Nc);CHKERRQ(ierr);
      if (qNc != Nc) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_SUP, "Quadrature components %D != %D field components\n", qNc, Nc);
      ierr = PetscFEGetDefaultTabulation(fe, NULL, &basisDer, NULL);CHKERRQ(ierr);
      if (debug) {
        char title[1024];
        ierr = PetscSNPrintf(title, 1023, "Solution for Field %d", field);CHKERRQ(ierr);
        ierr = DMPrintCellVector(c, title, Nb*Nc, &x[fieldOffset]);CHKERRQ(ierr);
      }
      for (q = 0; q < Nq; ++q) {
        for (d = 0; d < dim; d++) {
          coords[d] = v0[d];
          for (e = 0; e < dim; e++) {
            coords[d] += J[d*dim+e]*(quadPoints[q*dim+e] + 1.0);
          }
        }
        ierr = (*funcs[field])(dim, time, coords, n, numFields, funcVal, ctx);CHKERRQ(ierr);
        for (fc = 0; fc < Nc; ++fc) {
          PetscScalar interpolant = 0.0;

          for (d = 0; d < dim; ++d) interpolantVec[d] = 0.0;
          for (f = 0; f < Nb; ++f) {
            for (d = 0; d < dim; ++d) {
              realSpaceDer[d] = 0.0;
              for (g = 0; g < dim; ++g) {
                realSpaceDer[d] += invJ[g*dim+d]*basisDer[(q*Nb+f)*dim+g];
              }
              interpolantVec[d] += x[fieldOffset+f]*realSpaceDer[d];
            }
          }
          for (d = 0; d < dim; ++d) interpolant += interpolantVec[d]*n[d];
          if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, "    elem %d fieldDer %d diff %g\n", c, field, PetscSqr(PetscRealPart(interpolant - funcVal[fc]))*quadWeights[q*Nc+c]*detJ);CHKERRQ(ierr);}
          elemDiff += PetscSqr(PetscRealPart(interpolant - funcVal[fc]))*quadWeights[q*Nc+c]*detJ;
        }
      }
      comp        += Nc;
      fieldOffset += Nb*Nc;
    }
    ierr = DMDAVecRestoreClosure(dm, NULL, localX, c, NULL, &x);CHKERRQ(ierr);
    if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, "  elem %d diff %g\n", c, elemDiff);CHKERRQ(ierr);}
    localDiff += elemDiff;
  }
  ierr  = PetscFree7(funcVal,coords,realSpaceDer,v0,J,invJ,interpolantVec);CHKERRQ(ierr);
  ierr  = DMRestoreLocalVector(dm, &localX);CHKERRQ(ierr);
  ierr  = MPIU_Allreduce(&localDiff, diff, 1, MPIU_REAL, MPIU_SUM, PetscObjectComm((PetscObject)dm));CHKERRQ(ierr);
  *diff = PetscSqrtReal(*diff);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */

/*@C
     DMDAGetArray - Gets a work array for a DMDA

    Input Parameter:
+    da - information about my local patch
-    ghosted - do you want arrays for the ghosted or nonghosted patch

    Output Parameters:
.    vptr - array data structured

    Note:  The vector values are NOT initialized and may have garbage in them, so you may need
           to zero them.

  Level: advanced

.seealso: DMDARestoreArray()

@*/
PetscErrorCode  DMDAGetArray(DM da,PetscBool ghosted,void *vptr)
{
  PetscErrorCode ierr;
  PetscInt       j,i,xs,ys,xm,ym,zs,zm;
  char           *iarray_start;
  void           **iptr = (void**)vptr;
  DM_DA          *dd    = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(da,DM_CLASSID,1,DMDA);
  if (ghosted) {
    for (i=0; i<DMDA_MAX_WORK_ARRAYS; i++) {
      if (dd->arrayghostedin[i]) {
        *iptr                 = dd->arrayghostedin[i];
        iarray_start          = (char*)dd->startghostedin[i];
        dd->arrayghostedin[i] = NULL;
        dd->startghostedin[i] = NULL;

        goto done;
      }
    }
    xs = dd->Xs;
    ys = dd->Ys;
    zs = dd->Zs;
    xm = dd->Xe-dd->Xs;
    ym = dd->Ye-dd->Ys;
    zm = dd->Ze-dd->Zs;
  } else {
    for (i=0; i<DMDA_MAX_WORK_ARRAYS; i++) {
      if (dd->arrayin[i]) {
        *iptr          = dd->arrayin[i];
        iarray_start   = (char*)dd->startin[i];
        dd->arrayin[i] = NULL;
        dd->startin[i] = NULL;

        goto done;
      }
    }
    xs = dd->xs;
    ys = dd->ys;
    zs = dd->zs;
    xm = dd->xe-dd->xs;
    ym = dd->ye-dd->ys;
    zm = dd->ze-dd->zs;
  }

  switch (da->dim) {
  case 1: {
    void *ptr;

    ierr = PetscMalloc(xm*sizeof(PetscScalar),&iarray_start);CHKERRQ(ierr);

    ptr   = (void*)(iarray_start - xs*sizeof(PetscScalar));
    *iptr = (void*)ptr;
    break;
  }
  case 2: {
    void **ptr;

    ierr = PetscMalloc((ym+1)*sizeof(void*)+xm*ym*sizeof(PetscScalar),&iarray_start);CHKERRQ(ierr);

    ptr = (void**)(iarray_start + xm*ym*sizeof(PetscScalar) - ys*sizeof(void*));
    for (j=ys; j<ys+ym; j++) ptr[j] = iarray_start + sizeof(PetscScalar)*(xm*(j-ys) - xs);
    *iptr = (void*)ptr;
    break;
  }
  case 3: {
    void ***ptr,**bptr;

    ierr = PetscMalloc((zm+1)*sizeof(void**)+(ym*zm+1)*sizeof(void*)+xm*ym*zm*sizeof(PetscScalar),&iarray_start);CHKERRQ(ierr);

    ptr  = (void***)(iarray_start + xm*ym*zm*sizeof(PetscScalar) - zs*sizeof(void*));
    bptr = (void**)(iarray_start + xm*ym*zm*sizeof(PetscScalar) + zm*sizeof(void**));
    for (i=zs; i<zs+zm; i++) ptr[i] = bptr + ((i-zs)*ym - ys);
    for (i=zs; i<zs+zm; i++) {
      for (j=ys; j<ys+ym; j++) {
        ptr[i][j] = iarray_start + sizeof(PetscScalar)*(xm*ym*(i-zs) + xm*(j-ys) - xs);
      }
    }
    *iptr = (void*)ptr;
    break;
  }
  default:
    SETERRQ1(PetscObjectComm((PetscObject)da),PETSC_ERR_SUP,"Dimension %D not supported",da->dim);
  }

done:
  /* add arrays to the checked out list */
  if (ghosted) {
    for (i=0; i<DMDA_MAX_WORK_ARRAYS; i++) {
      if (!dd->arrayghostedout[i]) {
        dd->arrayghostedout[i] = *iptr;
        dd->startghostedout[i] = iarray_start;
        break;
      }
    }
  } else {
    for (i=0; i<DMDA_MAX_WORK_ARRAYS; i++) {
      if (!dd->arrayout[i]) {
        dd->arrayout[i] = *iptr;
        dd->startout[i] = iarray_start;
        break;
      }
    }
  }
  PetscFunctionReturn(0);
}

/*@C
     DMDARestoreArray - Restores an array of derivative types for a DMDA

    Input Parameter:
+    da - information about my local patch
.    ghosted - do you want arrays for the ghosted or nonghosted patch
-    vptr - array data structured to be passed to ad_FormFunctionLocal()

     Level: advanced

.seealso: DMDAGetArray()

@*/
PetscErrorCode  DMDARestoreArray(DM da,PetscBool ghosted,void *vptr)
{
  PetscInt i;
  void     **iptr = (void**)vptr,*iarray_start = 0;
  DM_DA    *dd    = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(da,DM_CLASSID,1,DMDA);
  if (ghosted) {
    for (i=0; i<DMDA_MAX_WORK_ARRAYS; i++) {
      if (dd->arrayghostedout[i] == *iptr) {
        iarray_start           = dd->startghostedout[i];
        dd->arrayghostedout[i] = NULL;
        dd->startghostedout[i] = NULL;
        break;
      }
    }
    for (i=0; i<DMDA_MAX_WORK_ARRAYS; i++) {
      if (!dd->arrayghostedin[i]) {
        dd->arrayghostedin[i] = *iptr;
        dd->startghostedin[i] = iarray_start;
        break;
      }
    }
  } else {
    for (i=0; i<DMDA_MAX_WORK_ARRAYS; i++) {
      if (dd->arrayout[i] == *iptr) {
        iarray_start    = dd->startout[i];
        dd->arrayout[i] = NULL;
        dd->startout[i] = NULL;
        break;
      }
    }
    for (i=0; i<DMDA_MAX_WORK_ARRAYS; i++) {
      if (!dd->arrayin[i]) {
        dd->arrayin[i] = *iptr;
        dd->startin[i] = iarray_start;
        break;
      }
    }
  }
  PetscFunctionReturn(0);
}

