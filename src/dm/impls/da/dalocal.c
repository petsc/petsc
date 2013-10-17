
/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include <petsc-private/dmdaimpl.h>    /*I   "petscdmda.h"   I*/
#include <petscbt.h>
#include <petscsf.h>

/*
   This allows the DMDA vectors to properly tell MATLAB their dimensions
*/
#if defined(PETSC_HAVE_MATLAB_ENGINE)
#include <engine.h>   /* MATLAB include file */
#include <mex.h>      /* MATLAB include file */
#undef __FUNCT__
#define __FUNCT__ "VecMatlabEnginePut_DA2d"
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


#undef __FUNCT__
#define __FUNCT__ "DMCreateLocalVector_DA"
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
    if (dd->w == 1  && dd->dim == 2) {
      ierr = PetscObjectComposeFunction((PetscObject)*g,"PetscMatlabEnginePut_C",VecMatlabEnginePut_DA2d);CHKERRQ(ierr);
    }
#endif
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDAGetNumCells"
PetscErrorCode DMDAGetNumCells(DM dm, PetscInt *numCells)
{
  DM_DA          *da = (DM_DA*) dm->data;
  const PetscInt dim = da->dim;
  const PetscInt mx  = (da->Xe - da->Xs)/da->w, my = da->Ye - da->Ys, mz = da->Ze - da->Zs;
  const PetscInt nC  = (mx)*(dim > 1 ? (my)*(dim > 2 ? (mz) : 1) : 1);

  PetscFunctionBegin;
  if (numCells) {
    PetscValidIntPointer(numCells,2);
    *numCells = nC;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDAGetNumVertices"
PetscErrorCode DMDAGetNumVertices(DM dm, PetscInt *numVerticesX, PetscInt *numVerticesY, PetscInt *numVerticesZ, PetscInt *numVertices)
{
  DM_DA          *da = (DM_DA*) dm->data;
  const PetscInt dim = da->dim;
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

#undef __FUNCT__
#define __FUNCT__ "DMDAGetNumFaces"
PetscErrorCode DMDAGetNumFaces(DM dm, PetscInt *numXFacesX, PetscInt *numXFaces, PetscInt *numYFacesY, PetscInt *numYFaces, PetscInt *numZFacesZ, PetscInt *numZFaces)
{
  DM_DA          *da = (DM_DA*) dm->data;
  const PetscInt dim = da->dim;
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

#undef __FUNCT__
#define __FUNCT__ "DMDAGetHeightStratum"
PetscErrorCode DMDAGetHeightStratum(DM dm, PetscInt height, PetscInt *pStart, PetscInt *pEnd)
{
  DM_DA          *da = (DM_DA*) dm->data;
  const PetscInt dim = da->dim;
  PetscInt       nC, nV, nXF, nYF, nZF;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (pStart) PetscValidIntPointer(pStart,3);
  if (pEnd)   PetscValidIntPointer(pEnd,4);
  ierr = DMDAGetNumCells(dm, &nC);CHKERRQ(ierr);
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

#undef __FUNCT__
#define __FUNCT__ "DMDACreateSection"
/*@C
  DMDACreateSection - Create a PetscSection inside the DMDA that describes data layout. This allows multiple fields with
  different numbers of dofs on vertices, cells, and faces in each direction.

  Input Parameters:
+ dm- The DMDA
. numFields - The number of fields
. numComp - The number of components in each field, or NULL for 1
. numVertexDof - The number of dofs per vertex for each field, or NULL
. numFaceDof - The number of dofs per face for each field and direction, or NULL
- numCellDof - The number of dofs per cell for each field, or NULL

  Level: developer

  Note:
  The default DMDA numbering is as follows:

    - Cells:    [0,             nC)
    - Vertices: [nC,            nC+nV)
    - X-Faces:  [nC+nV,         nC+nV+nXF)         normal is +- x-dir
    - Y-Faces:  [nC+nV+nXF,     nC+nV+nXF+nYF)     normal is +- y-dir
    - Z-Faces:  [nC+nV+nXF+nYF, nC+nV+nXF+nYF+nZF) normal is +- z-dir

  We interpret the default DMDA partition as a cell partition, and the data assignment as a cell assignment.
@*/
PetscErrorCode DMDACreateSection(DM dm, PetscInt numComp[], PetscInt numVertexDof[], PetscInt numFaceDof[], PetscInt numCellDof[])
{
  DM_DA            *da  = (DM_DA*) dm->data;
  PetscSection      section;
  const PetscInt    dim = da->dim;
  PetscInt          numFields, numVertexTotDof = 0, numCellTotDof = 0, numFaceTotDof[3] = {0, 0, 0};
  PetscBT           isLeaf;
  PetscSF           sf;
  PetscMPIInt       rank;
  const PetscMPIInt *neighbors;
  PetscInt          *localPoints;
  PetscSFNode       *remotePoints;
  PetscInt          nleaves = 0,  nleavesCheck = 0, nL = 0;
  PetscInt          nC, nVx, nVy, nVz, nV, nxF, nXF, nyF, nYF, nzF, nZF;
  PetscInt          pStart, pEnd, cStart, cEnd, vStart, vEnd, fStart, fEnd, xfStart, xfEnd, yfStart, yfEnd, zfStart, zfEnd;
  PetscInt          f, v, c, xf, yf, zf, xn, yn, zn;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr    = MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank);CHKERRQ(ierr);
  ierr    = DMDAGetNumCells(dm, &nC);CHKERRQ(ierr);
  ierr    = DMDAGetNumVertices(dm, &nVx, &nVy, &nVz, &nV);CHKERRQ(ierr);
  ierr    = DMDAGetNumFaces(dm, &nxF, &nXF, &nyF, &nYF, &nzF, &nZF);CHKERRQ(ierr);
  ierr    = DMDAGetHeightStratum(dm, -1,  &pStart, &pEnd);CHKERRQ(ierr);
  ierr    = DMDAGetHeightStratum(dm, 0,   &cStart, &cEnd);CHKERRQ(ierr);
  ierr    = DMDAGetHeightStratum(dm, 1,   &fStart, &fEnd);CHKERRQ(ierr);
  ierr    = DMDAGetHeightStratum(dm, dim, &vStart, &vEnd);CHKERRQ(ierr);
  xfStart = vEnd;  xfEnd = xfStart+nXF;
  yfStart = xfEnd; yfEnd = yfStart+nYF;
  zfStart = yfEnd; zfEnd = zfStart+nZF;
  if (zfEnd != fEnd) SETERRQ2(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Invalid face end %d, should be %d", zfEnd, fEnd);
  /* Create local section */
  ierr = DMDAGetInfo(dm, 0,0,0,0,0,0,0, &numFields, 0,0,0,0,0);CHKERRQ(ierr);
  for (f = 0; f < numFields; ++f) {
    if (numVertexDof) numVertexTotDof  += numVertexDof[f];
    if (numCellDof)   numCellTotDof    += numCellDof[f];
    if (numFaceDof) {
      numFaceTotDof[0] += numFaceDof[f*dim+0];
      numFaceTotDof[1] += dim > 1 ? numFaceDof[f*dim+1] : 0;
      numFaceTotDof[2] += dim > 2 ? numFaceDof[f*dim+2] : 0;
    }
  }
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject)dm), &section);CHKERRQ(ierr);
  if (numFields > 1) {
    ierr = PetscSectionSetNumFields(section, numFields);CHKERRQ(ierr);
    for (f = 0; f < numFields; ++f) {
      const char *name;

      ierr = DMDAGetFieldName(dm, f, &name);CHKERRQ(ierr);
      ierr = PetscSectionSetFieldName(section, f, name);CHKERRQ(ierr);
      if (numComp) {
        ierr = PetscSectionSetFieldComponents(section, f, numComp[f]);CHKERRQ(ierr);
      }
    }
  } else {
    numFields = 0;
  }
  ierr = PetscSectionSetChart(section, pStart, pEnd);CHKERRQ(ierr);
  if (numVertexDof) {
    for (v = vStart; v < vEnd; ++v) {
      for (f = 0; f < numFields; ++f) {
        ierr = PetscSectionSetFieldDof(section, v, f, numVertexDof[f]);CHKERRQ(ierr);
      }
      ierr = PetscSectionSetDof(section, v, numVertexTotDof);CHKERRQ(ierr);
    }
  }
  if (numFaceDof) {
    for (xf = xfStart; xf < xfEnd; ++xf) {
      for (f = 0; f < numFields; ++f) {
        ierr = PetscSectionSetFieldDof(section, xf, f, numFaceDof[f*dim+0]);CHKERRQ(ierr);
      }
      ierr = PetscSectionSetDof(section, xf, numFaceTotDof[0]);CHKERRQ(ierr);
    }
    for (yf = yfStart; yf < yfEnd; ++yf) {
      for (f = 0; f < numFields; ++f) {
        ierr = PetscSectionSetFieldDof(section, yf, f, numFaceDof[f*dim+1]);CHKERRQ(ierr);
      }
      ierr = PetscSectionSetDof(section, yf, numFaceTotDof[1]);CHKERRQ(ierr);
    }
    for (zf = zfStart; zf < zfEnd; ++zf) {
      for (f = 0; f < numFields; ++f) {
        ierr = PetscSectionSetFieldDof(section, zf, f, numFaceDof[f*dim+2]);CHKERRQ(ierr);
      }
      ierr = PetscSectionSetDof(section, zf, numFaceTotDof[2]);CHKERRQ(ierr);
    }
  }
  if (numCellDof) {
    for (c = cStart; c < cEnd; ++c) {
      for (f = 0; f < numFields; ++f) {
        ierr = PetscSectionSetFieldDof(section, c, f, numCellDof[f]);CHKERRQ(ierr);
      }
      ierr = PetscSectionSetDof(section, c, numCellTotDof);CHKERRQ(ierr);
    }
  }
  ierr = PetscSectionSetUp(section);CHKERRQ(ierr);
  /* Create mesh point SF */
  ierr = PetscBTCreate(pEnd-pStart, &isLeaf);CHKERRQ(ierr);
  ierr = DMDAGetNeighbors(dm, &neighbors);CHKERRQ(ierr);
  for (zn = 0; zn < (dim > 2 ? 3 : 1); ++zn) {
    for (yn = 0; yn < (dim > 1 ? 3 : 1); ++yn) {
      for (xn = 0; xn < 3; ++xn) {
        const PetscInt xp       = xn-1, yp = dim > 1 ? yn-1 : 0, zp = dim > 2 ? zn-1 : 0;
        const PetscInt neighbor = neighbors[(zn*3+yn)*3+xn];
        PetscInt       xv, yv, zv;

        if (neighbor >= 0 && neighbor < rank) {
          if (xp < 0) { /* left */
            if (yp < 0) { /* bottom */
              if (zp < 0) { /* back */
                const PetscInt localVertex  = (      0*nVy +     0)*nVx +     0 + nC; /* left bottom back vertex */
                if (!PetscBTLookupSet(isLeaf, localVertex)) ++nleaves;
              } else if (zp > 0) { /* front */
                const PetscInt localVertex  = ((nVz-1)*nVy +     0)*nVx +     0 + nC; /* left bottom front vertex */
                if (!PetscBTLookupSet(isLeaf, localVertex)) ++nleaves;
              } else {
                for (zv = 0; zv < nVz; ++zv) {
                  const PetscInt localVertex  = (zv*nVy +     0)*nVx +     0 + nC; /* left bottom vertices */
                  if (!PetscBTLookupSet(isLeaf, localVertex)) ++nleaves;
                }
              }
            } else if (yp > 0) { /* top */
              if (zp < 0) { /* back */
                const PetscInt localVertex  = (      0*nVy + nVy-1)*nVx +     0 + nC; /* left top back vertex */
                if (!PetscBTLookupSet(isLeaf, localVertex)) ++nleaves;
              } else if (zp > 0) { /* front */
                const PetscInt localVertex  = ((nVz-1)*nVy + nVy-1)*nVx +     0 + nC; /* left top front vertex */
                if (!PetscBTLookupSet(isLeaf, localVertex)) ++nleaves;
              } else {
                for (zv = 0; zv < nVz; ++zv) {
                  const PetscInt localVertex  = (zv*nVy + nVy-1)*nVx +     0 + nC; /* left top vertices */
                  if (!PetscBTLookupSet(isLeaf, localVertex)) ++nleaves;
                }
              }
            } else {
              if (zp < 0) { /* back */
                for (yv = 0; yv < nVy; ++yv) {
                  const PetscInt localVertex  = (      0*nVy + yv)*nVx +     0 + nC; /* left back vertices */
                  if (!PetscBTLookupSet(isLeaf, localVertex)) ++nleaves;
                }
              } else if (zp > 0) { /* front */
                for (yv = 0; yv < nVy; ++yv) {
                  const PetscInt localVertex  = ((nVz-1)*nVy + yv)*nVx +     0 + nC; /* left front vertices */
                  if (!PetscBTLookupSet(isLeaf, localVertex)) ++nleaves;
                }
              } else {
                for (zv = 0; zv < nVz; ++zv) {
                  for (yv = 0; yv < nVy; ++yv) {
                    const PetscInt localVertex  = (zv*nVy + yv)*nVx +     0 + nC; /* left vertices */
                    if (!PetscBTLookupSet(isLeaf, localVertex)) ++nleaves;
                  }
                }
#if 0
                for (xf = 0; xf < nxF; ++xf) {
                  /* THIS IS WRONG */
                  const PetscInt localFace  = 0 + nC+nV; /* left faces */
                  if (!PetscBTLookupSet(isLeaf, localFace)) ++nleaves;
                }
#endif
              }
            }
          } else if (xp > 0) { /* right */
            if (yp < 0) { /* bottom */
              if (zp < 0) { /* back */
                const PetscInt localVertex  = (      0*nVy +     0)*nVx + nVx-1 + nC; /* right bottom back vertex */
                if (!PetscBTLookupSet(isLeaf, localVertex)) ++nleaves;
              } else if (zp > 0) { /* front */
                const PetscInt localVertex  = ((nVz-1)*nVy +     0)*nVx + nVx-1 + nC; /* right bottom front vertex */
                if (!PetscBTLookupSet(isLeaf, localVertex)) ++nleaves;
              } else {
                for (zv = 0; zv < nVz; ++zv) {
                  const PetscInt localVertex  = (zv*nVy +     0)*nVx + nVx-1 + nC; /* right bottom vertices */
                  if (!PetscBTLookupSet(isLeaf, localVertex)) ++nleaves;
                }
              }
            } else if (yp > 0) { /* top */
              if (zp < 0) { /* back */
                const PetscInt localVertex  = (      0*nVy + nVy-1)*nVx + nVx-1 + nC; /* right top back vertex */
                if (!PetscBTLookupSet(isLeaf, localVertex)) ++nleaves;
              } else if (zp > 0) { /* front */
                const PetscInt localVertex  = ((nVz-1)*nVy + nVy-1)*nVx + nVx-1 + nC; /* right top front vertex */
                if (!PetscBTLookupSet(isLeaf, localVertex)) ++nleaves;
              } else {
                for (zv = 0; zv < nVz; ++zv) {
                  const PetscInt localVertex  = (zv*nVy + nVy-1)*nVx + nVx-1 + nC; /* right top vertices */
                  if (!PetscBTLookupSet(isLeaf, localVertex)) ++nleaves;
                }
              }
            } else {
              if (zp < 0) { /* back */
                for (yv = 0; yv < nVy; ++yv) {
                  const PetscInt localVertex  = (      0*nVy + yv)*nVx + nVx-1 + nC; /* right back vertices */
                  if (!PetscBTLookupSet(isLeaf, localVertex)) ++nleaves;
                }
              } else if (zp > 0) { /* front */
                for (yv = 0; yv < nVy; ++yv) {
                  const PetscInt localVertex  = ((nVz-1)*nVy + yv)*nVx + nVx-1 + nC; /* right front vertices */
                  if (!PetscBTLookupSet(isLeaf, localVertex)) ++nleaves;
                }
              } else {
                for (zv = 0; zv < nVz; ++zv) {
                  for (yv = 0; yv < nVy; ++yv) {
                    const PetscInt localVertex  = (zv*nVy + yv)*nVx + nVx-1 + nC; /* right vertices */
                    if (!PetscBTLookupSet(isLeaf, localVertex)) ++nleaves;
                  }
                }
#if 0
                for (xf = 0; xf < nxF; ++xf) {
                  /* THIS IS WRONG */
                  const PetscInt localFace  = 0 + nC+nV; /* right faces */
                  if (!PetscBTLookupSet(isLeaf, localFace)) ++nleaves;
                }
#endif
              }
            }
          } else {
            if (yp < 0) { /* bottom */
              if (zp < 0) { /* back */
                for (xv = 0; xv < nVx; ++xv) {
                  const PetscInt localVertex  = (      0*nVy +     0)*nVx + xv + nC; /* bottom back vertices */
                  if (!PetscBTLookupSet(isLeaf, localVertex)) ++nleaves;
                }
              } else if (zp > 0) { /* front */
                for (xv = 0; xv < nVx; ++xv) {
                  const PetscInt localVertex  = ((nVz-1)*nVy +     0)*nVx + xv + nC; /* bottom front vertices */
                  if (!PetscBTLookupSet(isLeaf, localVertex)) ++nleaves;
                }
              } else {
                for (zv = 0; zv < nVz; ++zv) {
                  for (xv = 0; xv < nVx; ++xv) {
                    const PetscInt localVertex  = (zv*nVy +     0)*nVx + xv + nC; /* bottom vertices */
                    if (!PetscBTLookupSet(isLeaf, localVertex)) ++nleaves;
                  }
                }
#if 0
                for (yf = 0; yf < nyF; ++yf) {
                  /* THIS IS WRONG */
                  const PetscInt localFace  = 0 + nC+nV; /* bottom faces */
                  if (!PetscBTLookupSet(isLeaf, localVFace)) ++nleaves;
                }
#endif
              }
            } else if (yp > 0) { /* top */
              if (zp < 0) { /* back */
                for (xv = 0; xv < nVx; ++xv) {
                  const PetscInt localVertex  = (      0*nVy + nVy-1)*nVx + xv + nC; /* top back vertices */
                  if (!PetscBTLookupSet(isLeaf, localVertex)) ++nleaves;
                }
              } else if (zp > 0) { /* front */
                for (xv = 0; xv < nVx; ++xv) {
                  const PetscInt localVertex  = ((nVz-1)*nVy + nVy-1)*nVx + xv + nC; /* top front vertices */
                  if (!PetscBTLookupSet(isLeaf, localVertex)) ++nleaves;
                }
              } else {
                for (zv = 0; zv < nVz; ++zv) {
                  for (xv = 0; xv < nVx; ++xv) {
                    const PetscInt localVertex  = (zv*nVy + nVy-1)*nVx + xv + nC; /* top vertices */
                    if (!PetscBTLookupSet(isLeaf, localVertex)) ++nleaves;
                  }
                }
#if 0
                for (yf = 0; yf < nyF; ++yf) {
                  /* THIS IS WRONG */
                  const PetscInt localFace  = 0 + nC+nV; /* top faces */
                  if (!PetscBTLookupSet(isLeaf, localVFace)) ++nleaves;
                }
#endif
              }
            } else {
              if (zp < 0) { /* back */
                for (yv = 0; yv < nVy; ++yv) {
                  for (xv = 0; xv < nVx; ++xv) {
                    const PetscInt localVertex  = (      0*nVy + yv)*nVx + xv + nC; /* back vertices */
                    if (!PetscBTLookupSet(isLeaf, localVertex)) ++nleaves;
                  }
                }
#if 0
                for (zf = 0; zf < nzF; ++zf) {
                  /* THIS IS WRONG */
                  const PetscInt localFace  = 0 + nC+nV; /* back faces */
                  if (!PetscBTLookupSet(isLeaf, localFace)) ++nleaves;
                }
#endif
              } else if (zp > 0) { /* front */
                for (yv = 0; yv < nVy; ++yv) {
                  for (xv = 0; xv < nVx; ++xv) {
                    const PetscInt localVertex  = ((nVz-1)*nVy + yv)*nVx + xv + nC; /* front vertices */
                    if (!PetscBTLookupSet(isLeaf, localVertex)) ++nleaves;
                  }
                }
#if 0
                for (zf = 0; zf < nzF; ++zf) {
                  /* THIS IS WRONG */
                  const PetscInt localFace  = 0 + nC+nV; /* front faces */
                  if (!PetscBTLookupSet(isLeaf, localFace)) ++nleaves;
                }
#endif
              } else {
                /* Nothing is shared from the interior */
              }
            }
          }
        }
      }
    }
  }
  ierr = PetscBTMemzero(pEnd-pStart, isLeaf);CHKERRQ(ierr);
  ierr = PetscMalloc2(nleaves,PetscInt,&localPoints,nleaves,PetscSFNode,&remotePoints);CHKERRQ(ierr);
  for (zn = 0; zn < (dim > 2 ? 3 : 1); ++zn) {
    for (yn = 0; yn < (dim > 1 ? 3 : 1); ++yn) {
      for (xn = 0; xn < 3; ++xn) {
        const PetscInt xp       = xn-1, yp = dim > 1 ? yn-1 : 0, zp = dim > 2 ? zn-1 : 0;
        const PetscInt neighbor = neighbors[(zn*3+yn)*3+xn];
        PetscInt       xv, yv, zv;

        if (neighbor >= 0 && neighbor < rank) {
          if (xp < 0) { /* left */
            if (yp < 0) { /* bottom */
              if (zp < 0) { /* back */
                const PetscInt localVertex  = (      0*nVy +     0)*nVx +     0 + nC; /* left bottom back vertex */
                const PetscInt remoteVertex = ((nVz-1)*nVy + nVy-1)*nVx + nVx-1 + nC; /* TODO: Correct this for neighbor sizes */

                if (!PetscBTLookupSet(isLeaf, localVertex)) {
                  localPoints[nL]        = localVertex;
                  remotePoints[nL].rank  = neighbor;
                  remotePoints[nL].index = remoteVertex;
                  ++nL;
                }
              } else if (zp > 0) { /* front */
                const PetscInt localVertex  = ((nVz-1)*nVy +     0)*nVx +     0 + nC; /* left bottom front vertex */
                const PetscInt remoteVertex = (      0*nVy + nVy-1)*nVx + nVx-1 + nC; /* TODO: Correct this for neighbor sizes */

                if (!PetscBTLookupSet(isLeaf, localVertex)) {
                  localPoints[nL]        = localVertex;
                  remotePoints[nL].rank  = neighbor;
                  remotePoints[nL].index = remoteVertex;
                  ++nL;
                }
              } else {
                for (zv = 0; zv < nVz; ++zv) {
                  const PetscInt localVertex  = (zv*nVy +     0)*nVx +     0 + nC; /* left bottom vertices */
                  const PetscInt remoteVertex = (zv*nVy + nVy-1)*nVx + nVx-1 + nC; /* TODO: Correct this for neighbor sizes */

                  if (!PetscBTLookupSet(isLeaf, localVertex)) {
                    localPoints[nL]        = localVertex;
                    remotePoints[nL].rank  = neighbor;
                    remotePoints[nL].index = remoteVertex;
                    ++nL;
                  }
                }
              }
            } else if (yp > 0) { /* top */
              if (zp < 0) { /* back */
                const PetscInt localVertex  = (      0*nVy + nVy-1)*nVx +     0 + nC; /* left top back vertex */
                const PetscInt remoteVertex = ((nVz-1)*nVy +     0)*nVx + nVx-1 + nC; /* TODO: Correct this for neighbor sizes */

                if (!PetscBTLookupSet(isLeaf, localVertex)) {
                  localPoints[nL]        = localVertex;
                  remotePoints[nL].rank  = neighbor;
                  remotePoints[nL].index = remoteVertex;
                  ++nL;
                }
              } else if (zp > 0) { /* front */
                const PetscInt localVertex  = ((nVz-1)*nVy + nVy-1)*nVx +     0 + nC; /* left top front vertex */
                const PetscInt remoteVertex = (      0*nVy +     0)*nVx + nVx-1 + nC; /* TODO: Correct this for neighbor sizes */

                if (!PetscBTLookupSet(isLeaf, localVertex)) {
                  localPoints[nL]        = localVertex;
                  remotePoints[nL].rank  = neighbor;
                  remotePoints[nL].index = remoteVertex;
                  ++nL;
                }
              } else {
                for (zv = 0; zv < nVz; ++zv) {
                  const PetscInt localVertex  = (zv*nVy + nVy-1)*nVx +     0 + nC; /* left top vertices */
                  const PetscInt remoteVertex = (zv*nVy +     0)*nVx + nVx-1 + nC; /* TODO: Correct this for neighbor sizes */

                  if (!PetscBTLookupSet(isLeaf, localVertex)) {
                    localPoints[nL]        = localVertex;
                    remotePoints[nL].rank  = neighbor;
                    remotePoints[nL].index = remoteVertex;
                    ++nL;
                  }
                }
              }
            } else {
              if (zp < 0) { /* back */
                for (yv = 0; yv < nVy; ++yv) {
                  const PetscInt localVertex  = (      0*nVy + yv)*nVx +     0 + nC; /* left back vertices */
                  const PetscInt remoteVertex = ((nVz-1)*nVy + yv)*nVx + nVx-1 + nC; /* TODO: Correct this for neighbor sizes */

                  if (!PetscBTLookupSet(isLeaf, localVertex)) {
                    localPoints[nL]        = localVertex;
                    remotePoints[nL].rank  = neighbor;
                    remotePoints[nL].index = remoteVertex;
                    ++nL;
                  }
                }
              } else if (zp > 0) { /* front */
                for (yv = 0; yv < nVy; ++yv) {
                  const PetscInt localVertex  = ((nVz-1)*nVy + yv)*nVx +     0 + nC; /* left front vertices */
                  const PetscInt remoteVertex = (      0*nVy + yv)*nVx + nVx-1 + nC; /* TODO: Correct this for neighbor sizes */

                  if (!PetscBTLookupSet(isLeaf, localVertex)) {
                    localPoints[nL]        = localVertex;
                    remotePoints[nL].rank  = neighbor;
                    remotePoints[nL].index = remoteVertex;
                    ++nL;
                  }
                }
              } else {
                for (zv = 0; zv < nVz; ++zv) {
                  for (yv = 0; yv < nVy; ++yv) {
                    const PetscInt localVertex  = (zv*nVy + yv)*nVx +     0 + nC; /* left vertices */
                    const PetscInt remoteVertex = (zv*nVy + yv)*nVx + nVx-1 + nC; /* TODO: Correct this for neighbor sizes */

                    if (!PetscBTLookupSet(isLeaf, localVertex)) {
                      localPoints[nL]        = localVertex;
                      remotePoints[nL].rank  = neighbor;
                      remotePoints[nL].index = remoteVertex;
                      ++nL;
                    }
                  }
                }
#if 0
                for (xf = 0; xf < nxF; ++xf) {
                  /* THIS IS WRONG */
                  const PetscInt localFace  = 0 + nC+nV; /* left faces */
                  const PetscInt remoteFace = 0 + nC+nV;

                  if (!PetscBTLookupSet(isLeaf, localFace)) {
                    localPoints[nL]        = localFace;
                    remotePoints[nL].rank  = neighbor;
                    remotePoints[nL].index = remoteFace;
                  }
                }
#endif
              }
            }
          } else if (xp > 0) { /* right */
            if (yp < 0) { /* bottom */
              if (zp < 0) { /* back */
                const PetscInt localVertex  = (      0*nVy +     0)*nVx + nVx-1 + nC; /* right bottom back vertex */
                const PetscInt remoteVertex = ((nVz-1)*nVy + nVy-1)*nVx +     0 + nC; /* TODO: Correct this for neighbor sizes */

                if (!PetscBTLookupSet(isLeaf, localVertex)) {
                  localPoints[nL]        = localVertex;
                  remotePoints[nL].rank  = neighbor;
                  remotePoints[nL].index = remoteVertex;
                  ++nL;
                }
              } else if (zp > 0) { /* front */
                const PetscInt localVertex  = ((nVz-1)*nVy +     0)*nVx + nVx-1 + nC; /* right bottom front vertex */
                const PetscInt remoteVertex = (      0*nVy + nVy-1)*nVx +     0 + nC; /* TODO: Correct this for neighbor sizes */

                if (!PetscBTLookupSet(isLeaf, localVertex)) {
                  localPoints[nL]        = localVertex;
                  remotePoints[nL].rank  = neighbor;
                  remotePoints[nL].index = remoteVertex;
                  ++nL;
                }
              } else {
                nleavesCheck += nVz;
                for (zv = 0; zv < nVz; ++zv) {
                  const PetscInt localVertex  = (zv*nVy +     0)*nVx + nVx-1 + nC; /* right bottom vertices */
                  const PetscInt remoteVertex = (zv*nVy + nVy-1)*nVx +     0 + nC; /* TODO: Correct this for neighbor sizes */

                  if (!PetscBTLookupSet(isLeaf, localVertex)) {
                    localPoints[nL]        = localVertex;
                    remotePoints[nL].rank  = neighbor;
                    remotePoints[nL].index = remoteVertex;
                    ++nL;
                  }
                }
              }
            } else if (yp > 0) { /* top */
              if (zp < 0) { /* back */
                const PetscInt localVertex  = (      0*nVy + nVy-1)*nVx + nVx-1 + nC; /* right top back vertex */
                const PetscInt remoteVertex = ((nVz-1)*nVy +     0)*nVx +     0 + nC; /* TODO: Correct this for neighbor sizes */

                if (!PetscBTLookupSet(isLeaf, localVertex)) {
                  localPoints[nL]        = localVertex;
                  remotePoints[nL].rank  = neighbor;
                  remotePoints[nL].index = remoteVertex;
                  ++nL;
                }
              } else if (zp > 0) { /* front */
                const PetscInt localVertex  = ((nVz-1)*nVy + nVy-1)*nVx + nVx-1 + nC; /* right top front vertex */
                const PetscInt remoteVertex = (      0*nVy +     0)*nVx +     0 + nC; /* TODO: Correct this for neighbor sizes */

                if (!PetscBTLookupSet(isLeaf, localVertex)) {
                  localPoints[nL]        = localVertex;
                  remotePoints[nL].rank  = neighbor;
                  remotePoints[nL].index = remoteVertex;
                  ++nL;
                }
              } else {
                for (zv = 0; zv < nVz; ++zv) {
                  const PetscInt localVertex  = (zv*nVy + nVy-1)*nVx + nVx-1 + nC; /* right top vertices */
                  const PetscInt remoteVertex = (zv*nVy +     0)*nVx +     0 + nC; /* TODO: Correct this for neighbor sizes */

                  if (!PetscBTLookupSet(isLeaf, localVertex)) {
                    localPoints[nL]        = localVertex;
                    remotePoints[nL].rank  = neighbor;
                    remotePoints[nL].index = remoteVertex;
                    ++nL;
                  }
                }
              }
            } else {
              if (zp < 0) { /* back */
                for (yv = 0; yv < nVy; ++yv) {
                  const PetscInt localVertex  = (      0*nVy + yv)*nVx + nVx-1 + nC; /* right back vertices */
                  const PetscInt remoteVertex = ((nVz-1)*nVy + yv)*nVx +     0 + nC; /* TODO: Correct this for neighbor sizes */

                  if (!PetscBTLookupSet(isLeaf, localVertex)) {
                    localPoints[nL]        = localVertex;
                    remotePoints[nL].rank  = neighbor;
                    remotePoints[nL].index = remoteVertex;
                    ++nL;
                  }
                }
              } else if (zp > 0) { /* front */
                for (yv = 0; yv < nVy; ++yv) {
                  const PetscInt localVertex  = ((nVz-1)*nVy + yv)*nVx + nVx-1 + nC; /* right front vertices */
                  const PetscInt remoteVertex = (      0*nVy + yv)*nVx +     0 + nC; /* TODO: Correct this for neighbor sizes */

                  if (!PetscBTLookupSet(isLeaf, localVertex)) {
                    localPoints[nL]        = localVertex;
                    remotePoints[nL].rank  = neighbor;
                    remotePoints[nL].index = remoteVertex;
                    ++nL;
                  }
                }
              } else {
                for (zv = 0; zv < nVz; ++zv) {
                  for (yv = 0; yv < nVy; ++yv) {
                    const PetscInt localVertex  = (zv*nVy + yv)*nVx + nVx-1 + nC; /* right vertices */
                    const PetscInt remoteVertex = (zv*nVy + yv)*nVx + 0     + nC; /* TODO: Correct this for neighbor sizes */

                    if (!PetscBTLookupSet(isLeaf, localVertex)) {
                      localPoints[nL]        = localVertex;
                      remotePoints[nL].rank  = neighbor;
                      remotePoints[nL].index = remoteVertex;
                      ++nL;
                    }
                  }
                }
#if 0
                for (xf = 0; xf < nxF; ++xf) {
                  /* THIS IS WRONG */
                  const PetscInt localFace  = 0 + nC+nV; /* right faces */
                  const PetscInt remoteFace = 0 + nC+nV;

                  if (!PetscBTLookupSet(isLeaf, localFace)) {
                    localPoints[nL]        = localFace;
                    remotePoints[nL].rank  = neighbor;
                    remotePoints[nL].index = remoteFace;
                    ++nL;
                  }
                }
#endif
              }
            }
          } else {
            if (yp < 0) { /* bottom */
              if (zp < 0) { /* back */
                for (xv = 0; xv < nVx; ++xv) {
                  const PetscInt localVertex  = (      0*nVy +     0)*nVx + xv + nC; /* bottom back vertices */
                  const PetscInt remoteVertex = ((nVz-1)*nVy + nVy-1)*nVx + xv + nC; /* TODO: Correct this for neighbor sizes */

                  if (!PetscBTLookupSet(isLeaf, localVertex)) {
                    localPoints[nL]        = localVertex;
                    remotePoints[nL].rank  = neighbor;
                    remotePoints[nL].index = remoteVertex;
                    ++nL;
                  }
                }
              } else if (zp > 0) { /* front */
                for (xv = 0; xv < nVx; ++xv) {
                  const PetscInt localVertex  = ((nVz-1)*nVy +     0)*nVx + xv + nC; /* bottom front vertices */
                  const PetscInt remoteVertex = (      0*nVy + nVy-1)*nVx + xv + nC; /* TODO: Correct this for neighbor sizes */

                  if (!PetscBTLookupSet(isLeaf, localVertex)) {
                    localPoints[nL]        = localVertex;
                    remotePoints[nL].rank  = neighbor;
                    remotePoints[nL].index = remoteVertex;
                    ++nL;
                  }
                }
              } else {
                for (zv = 0; zv < nVz; ++zv) {
                  for (xv = 0; xv < nVx; ++xv) {
                    const PetscInt localVertex  = (zv*nVy +     0)*nVx + xv + nC; /* bottom vertices */
                    const PetscInt remoteVertex = (zv*nVy + nVy-1)*nVx + xv + nC; /* TODO: Correct this for neighbor sizes */

                    if (!PetscBTLookupSet(isLeaf, localVertex)) {
                      localPoints[nL]        = localVertex;
                      remotePoints[nL].rank  = neighbor;
                      remotePoints[nL].index = remoteVertex;
                      ++nL;
                    }
                  }
                }
#if 0
                for (yf = 0; yf < nyF; ++yf) {
                  /* THIS IS WRONG */
                  const PetscInt localFace  = 0 + nC+nV; /* bottom faces */
                  const PetscInt remoteFace = 0 + nC+nV;

                  if (!PetscBTLookupSet(isLeaf, localFace)) {
                    localPoints[nL]        = localFace;
                    remotePoints[nL].rank  = neighbor;
                    remotePoints[nL].index = remoteFace;
                    ++nL;
                  }
                }
#endif
              }
            } else if (yp > 0) { /* top */
              if (zp < 0) { /* back */
                for (xv = 0; xv < nVx; ++xv) {
                  const PetscInt localVertex  = (      0*nVy + nVy-1)*nVx + xv + nC; /* top back vertices */
                  const PetscInt remoteVertex = ((nVz-1)*nVy +     0)*nVx + xv + nC; /* TODO: Correct this for neighbor sizes */

                  if (!PetscBTLookupSet(isLeaf, localVertex)) {
                    localPoints[nL]        = localVertex;
                    remotePoints[nL].rank  = neighbor;
                    remotePoints[nL].index = remoteVertex;
                    ++nL;
                  }
                }
              } else if (zp > 0) { /* front */
                for (xv = 0; xv < nVx; ++xv) {
                  const PetscInt localVertex  = ((nVz-1)*nVy + nVy-1)*nVx + xv + nC; /* top front vertices */
                  const PetscInt remoteVertex = (      0*nVy +     0)*nVx + xv + nC; /* TODO: Correct this for neighbor sizes */

                  if (!PetscBTLookupSet(isLeaf, localVertex)) {
                    localPoints[nL]        = localVertex;
                    remotePoints[nL].rank  = neighbor;
                    remotePoints[nL].index = remoteVertex;
                    ++nL;
                  }
                }
              } else {
                for (zv = 0; zv < nVz; ++zv) {
                  for (xv = 0; xv < nVx; ++xv) {
                    const PetscInt localVertex  = (zv*nVy + nVy-1)*nVx + xv + nC; /* top vertices */
                    const PetscInt remoteVertex = (zv*nVy +     0)*nVx + xv + nC; /* TODO: Correct this for neighbor sizes */

                    if (!PetscBTLookupSet(isLeaf, localVertex)) {
                      localPoints[nL]        = localVertex;
                      remotePoints[nL].rank  = neighbor;
                      remotePoints[nL].index = remoteVertex;
                      ++nL;
                    }
                  }
                }
#if 0
                for (yf = 0; yf < nyF; ++yf) {
                  /* THIS IS WRONG */
                  const PetscInt localFace  = 0 + nC+nV; /* top faces */
                  const PetscInt remoteFace = 0 + nC+nV;

                  if (!PetscBTLookupSet(isLeaf, localFace)) {
                    localPoints[nL]        = localFace;
                    remotePoints[nL].rank  = neighbor;
                    remotePoints[nL].index = remoteFace;
                    ++nL;
                  }
                }
#endif
              }
            } else {
              if (zp < 0) { /* back */
                for (yv = 0; yv < nVy; ++yv) {
                  for (xv = 0; xv < nVx; ++xv) {
                    const PetscInt localVertex  = (      0*nVy + yv)*nVx + xv + nC; /* back vertices */
                    const PetscInt remoteVertex = ((nVz-1)*nVy + yv)*nVx + xv + nC; /* TODO: Correct this for neighbor sizes */

                    if (!PetscBTLookupSet(isLeaf, localVertex)) {
                      localPoints[nL]        = localVertex;
                      remotePoints[nL].rank  = neighbor;
                      remotePoints[nL].index = remoteVertex;
                      ++nL;
                    }
                  }
                }
#if 0
                for (zf = 0; zf < nzF; ++zf) {
                  /* THIS IS WRONG */
                  const PetscInt localFace  = 0 + nC+nV; /* back faces */
                  const PetscInt remoteFace = 0 + nC+nV;

                  if (!PetscBTLookupSet(isLeaf, localFace)) {
                    localPoints[nL]        = localFace;
                    remotePoints[nL].rank  = neighbor;
                    remotePoints[nL].index = remoteFace;
                    ++nL;
                  }
                }
#endif
              } else if (zp > 0) { /* front */
                for (yv = 0; yv < nVy; ++yv) {
                  for (xv = 0; xv < nVx; ++xv) {
                    const PetscInt localVertex  = ((nVz-1)*nVy + yv)*nVx + xv + nC; /* front vertices */
                    const PetscInt remoteVertex = (      0*nVy + yv)*nVx + xv + nC; /* TODO: Correct this for neighbor sizes */

                    if (!PetscBTLookupSet(isLeaf, localVertex)) {
                      localPoints[nL]        = localVertex;
                      remotePoints[nL].rank  = neighbor;
                      remotePoints[nL].index = remoteVertex;
                      ++nL;
                    }
                  }
                }
#if 0
                for (zf = 0; zf < nzF; ++zf) {
                  /* THIS IS WRONG */
                  const PetscInt localFace  = 0 + nC+nV; /* front faces */
                  const PetscInt remoteFace = 0 + nC+nV;

                  if (!PetscBTLookupSet(isLeaf, localFace)) {
                    localPoints[nL]        = localFace;
                    remotePoints[nL].rank  = neighbor;
                    remotePoints[nL].index = remoteFace;
                    ++nL;
                  }
                }
#endif
              } else {
                /* Nothing is shared from the interior */
              }
            }
          }
        }
      }
    }
  }
  ierr = PetscBTDestroy(&isLeaf);CHKERRQ(ierr);
  /* Remove duplication in leaf determination */
  if (nleaves != nL) SETERRQ2(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "The number of leaves %d did not match the number of remote leaves %d", nleaves, nleavesCheck);
  ierr = PetscSFCreate(PetscObjectComm((PetscObject)dm), &sf);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(sf, pEnd, nleaves, localPoints, PETSC_OWN_POINTER, remotePoints, PETSC_OWN_POINTER);CHKERRQ(ierr);
  ierr = DMSetPointSF(dm, sf);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);
  ierr = DMSetDefaultSection(dm, section);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&section);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */

#undef __FUNCT__
#define __FUNCT__ "DMDAGetArray"
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
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
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

  switch (dd->dim) {
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
    SETERRQ1(PetscObjectComm((PetscObject)da),PETSC_ERR_SUP,"Dimension %D not supported",dd->dim);
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

#undef __FUNCT__
#define __FUNCT__ "DMDARestoreArray"
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
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
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

