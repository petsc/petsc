#include <petsc-private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/
#include <../src/sys/utils/hash.h>

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetFaces_Internal"
/*
  DMPlexGetFaces_Internal - Gets groups of vertices that correspond to faces for the given cell
*/
PetscErrorCode DMPlexGetFaces_Internal(DM dm, PetscInt dim, PetscInt p, PetscInt *numFaces, PetscInt *faceSize, const PetscInt *faces[])
{
  const PetscInt *cone = NULL;
  PetscInt        maxConeSize, maxSupportSize, coneSize;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMPlexGetMaxSizes(dm, &maxConeSize, &maxSupportSize);CHKERRQ(ierr);
  ierr = DMPlexGetConeSize(dm, p, &coneSize);CHKERRQ(ierr);
  ierr = DMPlexGetCone(dm, p, &cone);CHKERRQ(ierr);
  ierr = DMPlexGetRawFaces_Internal(dm, dim, coneSize, cone, numFaces, faceSize, faces);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexRestoreFaces_Internal"
/*
  DMPlexRestoreFaces_Internal - Restores the array
*/
PetscErrorCode DMPlexRestoreFaces_Internal(DM dm, PetscInt dim, PetscInt p, PetscInt *numFaces, PetscInt *faceSize, const PetscInt *faces[])
{
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = DMRestoreWorkArray(dm, 0, PETSC_INT, (void *) faces);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetRawFaces_Internal"
/*
  DMPlexGetRawFaces_Internal - Gets groups of vertices that correspond to faces for the given cone
*/
PetscErrorCode DMPlexGetRawFaces_Internal(DM dm, PetscInt dim, PetscInt coneSize, const PetscInt cone[], PetscInt *numFaces, PetscInt *faceSize, const PetscInt *faces[])
{
  PetscInt       *facesTmp;
  PetscInt        maxConeSize, maxSupportSize;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMPlexGetMaxSizes(dm, &maxConeSize, &maxSupportSize);CHKERRQ(ierr);
  if (faces) {ierr = DMGetWorkArray(dm, PetscSqr(PetscMax(maxConeSize, maxSupportSize)), PETSC_INT, &facesTmp);CHKERRQ(ierr);}
  switch (dim) {
  case 2:
    switch (coneSize) {
    case 3:
      if (faces) {
        facesTmp[0] = cone[0]; facesTmp[1] = cone[1];
        facesTmp[2] = cone[1]; facesTmp[3] = cone[2];
        facesTmp[4] = cone[2]; facesTmp[5] = cone[0];
        *faces = facesTmp;
      }
      if (numFaces) *numFaces         = 3;
      if (faceSize) *faceSize         = 2;
      break;
    case 4:
      /* Vertices follow right hand rule */
      if (faces) {
        facesTmp[0] = cone[0]; facesTmp[1] = cone[1];
        facesTmp[2] = cone[1]; facesTmp[3] = cone[2];
        facesTmp[4] = cone[2]; facesTmp[5] = cone[3];
        facesTmp[6] = cone[3]; facesTmp[7] = cone[0];
        *faces = facesTmp;
      }
      if (numFaces) *numFaces         = 4;
      if (faceSize) *faceSize         = 2;
      break;
    default:
      SETERRQ2(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Cone size %D not supported for dimension %D", coneSize, dim);
    }
    break;
  case 3:
    switch (coneSize) {
    case 3:
      if (faces) {
        facesTmp[0] = cone[0]; facesTmp[1] = cone[1];
        facesTmp[2] = cone[1]; facesTmp[3] = cone[2];
        facesTmp[4] = cone[2]; facesTmp[5] = cone[0];
        *faces = facesTmp;
      }
      if (numFaces) *numFaces         = 3;
      if (faceSize) *faceSize         = 2;
      break;
    case 4:
      /* Vertices of first face follow right hand rule and normal points away from last vertex */
      if (faces) {
        facesTmp[0] = cone[0]; facesTmp[1]  = cone[1]; facesTmp[2]  = cone[2];
        facesTmp[3] = cone[0]; facesTmp[4]  = cone[3]; facesTmp[5]  = cone[1];
        facesTmp[6] = cone[0]; facesTmp[7]  = cone[2]; facesTmp[8]  = cone[3];
        facesTmp[9] = cone[2]; facesTmp[10] = cone[1]; facesTmp[11] = cone[3];
        *faces = facesTmp;
      }
      if (numFaces) *numFaces         = 4;
      if (faceSize) *faceSize         = 3;
      break;
    case 8:
      if (faces) {
        facesTmp[0]  = cone[0]; facesTmp[1]  = cone[1]; facesTmp[2]  = cone[2]; facesTmp[3]  = cone[3];
        facesTmp[4]  = cone[4]; facesTmp[5]  = cone[5]; facesTmp[6]  = cone[6]; facesTmp[7]  = cone[7];
        facesTmp[8]  = cone[0]; facesTmp[9]  = cone[3]; facesTmp[10] = cone[5]; facesTmp[11] = cone[4];
        facesTmp[12] = cone[2]; facesTmp[13] = cone[1]; facesTmp[14] = cone[7]; facesTmp[15] = cone[6];
        facesTmp[16] = cone[3]; facesTmp[17] = cone[2]; facesTmp[18] = cone[6]; facesTmp[19] = cone[5];
        facesTmp[20] = cone[0]; facesTmp[21] = cone[4]; facesTmp[22] = cone[7]; facesTmp[23] = cone[1];
        *faces = facesTmp;
      }
      if (numFaces) *numFaces         = 6;
      if (faceSize) *faceSize         = 4;
      break;
    default:
      SETERRQ2(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Cone size %D not supported for dimension %D", coneSize, dim);
    }
    break;
  default:
    SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Dimension %D not supported", dim);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexInterpolateFaces_Internal"
/* This interpolates faces for cells at some stratum */
static PetscErrorCode DMPlexInterpolateFaces_Internal(DM dm, PetscInt cellDepth, DM idm)
{
  DMLabel        subpointMap;
  PetscHashIJKL  faceTable;
  PetscInt      *pStart, *pEnd;
  PetscInt       cellDim, depth, faceDepth = cellDepth, numPoints = 0, faceSizeAll = 0, face, c, d;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetDimension(dm, &cellDim);CHKERRQ(ierr);
  /* HACK: I need a better way to determine face dimension, or an alternative to GetFaces() */
  ierr = DMPlexGetSubpointMap(dm, &subpointMap);CHKERRQ(ierr);
  if (subpointMap) ++cellDim;
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ++depth;
  ++cellDepth;
  cellDim -= depth - cellDepth;
  ierr = PetscMalloc2(depth+1,PetscInt,&pStart,depth+1,PetscInt,&pEnd);CHKERRQ(ierr);
  for (d = depth-1; d >= faceDepth; --d) {
    ierr = DMPlexGetDepthStratum(dm, d, &pStart[d+1], &pEnd[d+1]);CHKERRQ(ierr);
  }
  ierr = DMPlexGetDepthStratum(dm, -1, NULL, &pStart[faceDepth]);CHKERRQ(ierr);
  pEnd[faceDepth] = pStart[faceDepth];
  for (d = faceDepth-1; d >= 0; --d) {
    ierr = DMPlexGetDepthStratum(dm, d, &pStart[d], &pEnd[d]);CHKERRQ(ierr);
  }
  if (pEnd[cellDepth] > pStart[cellDepth]) {ierr = DMPlexGetFaces_Internal(dm, cellDim, pStart[cellDepth], NULL, &faceSizeAll, NULL);CHKERRQ(ierr);}
  if (faceSizeAll > 4) SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "Do not support interpolation of meshes with faces of %D vertices", faceSizeAll);
  ierr = PetscHashIJKLCreate(&faceTable);CHKERRQ(ierr);
  ierr = PetscHashIJKLSetMultivalued(faceTable, PETSC_FALSE);CHKERRQ(ierr);
  for (c = pStart[cellDepth], face = pStart[faceDepth]; c < pEnd[cellDepth]; ++c) {
    const PetscInt *cellFaces;
    PetscInt        numCellFaces, faceSize, cf, f;

    ierr = DMPlexGetFaces_Internal(dm, cellDim, c, &numCellFaces, &faceSize, &cellFaces);CHKERRQ(ierr);
    if (faceSize != faceSizeAll) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Inconsistent face for cell %D of size %D != %D", c, faceSize, faceSizeAll);
    for (cf = 0; cf < numCellFaces; ++cf) {
      const PetscInt  *cellFace = &cellFaces[cf*faceSize];
      PetscHashIJKLKey key;

      if (faceSize == 2) {
        key.i = PetscMin(cellFace[0], cellFace[1]);
        key.j = PetscMax(cellFace[0], cellFace[1]);
        key.k = 0;
        key.l = 0;
      } else {
        key.i = cellFace[0]; key.j = cellFace[1]; key.k = cellFace[2]; key.l = faceSize > 3 ? cellFace[3] : 0;
        ierr = PetscSortInt(faceSize, (PetscInt *) &key);
      }
      ierr  = PetscHashIJKLGet(faceTable, key, &f);CHKERRQ(ierr);
      if (f < 0) {
        ierr = PetscHashIJKLAdd(faceTable, key, face);CHKERRQ(ierr);
        f    = face++;
      }
    }
    ierr = DMPlexRestoreFaces_Internal(dm, cellDim, c, &numCellFaces, &faceSize, &cellFaces);CHKERRQ(ierr);
  }
  pEnd[faceDepth] = face;
  ierr = PetscHashIJKLDestroy(&faceTable);CHKERRQ(ierr);
  /* Count new points */
  for (d = 0; d <= depth; ++d) {
    numPoints += pEnd[d]-pStart[d];
  }
  ierr = DMPlexSetChart(idm, 0, numPoints);CHKERRQ(ierr);
  /* Set cone sizes */
  for (d = 0; d <= depth; ++d) {
    PetscInt coneSize, p;

    if (d == faceDepth) {
      for (p = pStart[d]; p < pEnd[d]; ++p) {
        /* I see no way to do this if we admit faces of different shapes */
        ierr = DMPlexSetConeSize(idm, p, faceSizeAll);CHKERRQ(ierr);
      }
    } else if (d == cellDepth) {
      for (p = pStart[d]; p < pEnd[d]; ++p) {
        /* Number of cell faces may be different from number of cell vertices*/
        ierr = DMPlexGetFaces_Internal(dm, cellDim, p, &coneSize, NULL, NULL);CHKERRQ(ierr);
        ierr = DMPlexSetConeSize(idm, p, coneSize);CHKERRQ(ierr);
      }
    } else {
      for (p = pStart[d]; p < pEnd[d]; ++p) {
        ierr = DMPlexGetConeSize(dm, p, &coneSize);CHKERRQ(ierr);
        ierr = DMPlexSetConeSize(idm, p, coneSize);CHKERRQ(ierr);
      }
    }
  }
  ierr = DMSetUp(idm);CHKERRQ(ierr);
  /* Get face cones from subsets of cell vertices */
  if (faceSizeAll > 4) SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "Do not support interpolation of meshes with faces of %D vertices", faceSizeAll);
  ierr = PetscHashIJKLCreate(&faceTable);CHKERRQ(ierr);
  ierr = PetscHashIJKLSetMultivalued(faceTable, PETSC_FALSE);CHKERRQ(ierr);
  for (d = depth; d > cellDepth; --d) {
    const PetscInt *cone;
    PetscInt        p;

    for (p = pStart[d]; p < pEnd[d]; ++p) {
      ierr = DMPlexGetCone(dm, p, &cone);CHKERRQ(ierr);
      ierr = DMPlexSetCone(idm, p, cone);CHKERRQ(ierr);
      ierr = DMPlexGetConeOrientation(dm, p, &cone);CHKERRQ(ierr);
      ierr = DMPlexSetConeOrientation(idm, p, cone);CHKERRQ(ierr);
    }
  }
  for (c = pStart[cellDepth], face = pStart[faceDepth]; c < pEnd[cellDepth]; ++c) {
    const PetscInt *cellFaces;
    PetscInt        numCellFaces, faceSize, cf, f;

    ierr = DMPlexGetFaces_Internal(dm, cellDim, c, &numCellFaces, &faceSize, &cellFaces);CHKERRQ(ierr);
    if (faceSize != faceSizeAll) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Inconsistent face for cell %D of size %D != %D", c, faceSize, faceSizeAll);
    for (cf = 0; cf < numCellFaces; ++cf) {
      const PetscInt  *cellFace = &cellFaces[cf*faceSize];
      PetscHashIJKLKey key;

      if (faceSize == 2) {
        key.i = PetscMin(cellFace[0], cellFace[1]);
        key.j = PetscMax(cellFace[0], cellFace[1]);
        key.k = 0;
        key.l = 0;
      } else {
        key.i = cellFace[0]; key.j = cellFace[1]; key.k = cellFace[2]; key.l = faceSize > 3 ? cellFace[3] : 0;
        ierr = PetscSortInt(faceSize, (PetscInt *) &key);
      }
      ierr  = PetscHashIJKLGet(faceTable, key, &f);CHKERRQ(ierr);
      if (f < 0) {
        ierr = DMPlexSetCone(idm, face, cellFace);CHKERRQ(ierr);
        ierr = PetscHashIJKLAdd(faceTable, key, face);CHKERRQ(ierr);
        f    = face++;
        ierr = DMPlexInsertCone(idm, c, cf, f);CHKERRQ(ierr);
      } else {
        const PetscInt *cone;
        PetscInt        coneSize, ornt, i, j;

        ierr = DMPlexInsertCone(idm, c, cf, f);CHKERRQ(ierr);
        /* Orient face: Do not allow reverse orientation at the first vertex */
        ierr = DMPlexGetConeSize(idm, f, &coneSize);CHKERRQ(ierr);
        ierr = DMPlexGetCone(idm, f, &cone);CHKERRQ(ierr);
        if (coneSize != faceSize) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid number of face vertices %D for face %D should be %D", coneSize, f, faceSize);
        /* - First find the initial vertex */
        for (i = 0; i < faceSize; ++i) if (cellFace[0] == cone[i]) break;
        /* - Try forward comparison */
        for (j = 0; j < faceSize; ++j) if (cellFace[j] != cone[(i+j)%faceSize]) break;
        if (j == faceSize) {
          if ((faceSize == 2) && (i == 1)) ornt = -2;
          else                             ornt = i;
        } else {
          /* - Try backward comparison */
          for (j = 0; j < faceSize; ++j) if (cellFace[j] != cone[(i+faceSize-j)%faceSize]) break;
          if (j == faceSize) {
            if (i == 0) ornt = -faceSize;
            else        ornt = -i;
          } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Could not determine face orientation");
        }
        ierr = DMPlexInsertConeOrientation(idm, c, cf, ornt);CHKERRQ(ierr);
      }
    }
    ierr = DMPlexRestoreFaces_Internal(dm, cellDim, c, &numCellFaces, &faceSize, &cellFaces);CHKERRQ(ierr);
  }
  if (face != pEnd[faceDepth]) SETERRQ2(PetscObjectComm((PetscObject) dm), PETSC_ERR_PLIB, "Invalid number of faces %D should be %D", face-pStart[faceDepth], pEnd[faceDepth]-pStart[faceDepth]);
  ierr = PetscFree2(pStart,pEnd);CHKERRQ(ierr);
  ierr = PetscHashIJKLDestroy(&faceTable);CHKERRQ(ierr);
  ierr = PetscFree2(pStart,pEnd);CHKERRQ(ierr);
  ierr = DMPlexSymmetrize(idm);CHKERRQ(ierr);
  ierr = DMPlexStratify(idm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexInterpolate"
/*@
  DMPlexInterpolate - Take in a cell-vertex mesh and return one with all intermediate faces, edges, etc.

  Collective on DM

  Input Parameter:
. dm - The DMPlex object with only cells and vertices

  Output Parameter:
. dmInt - The complete DMPlex object

  Level: intermediate

.keywords: mesh
.seealso: DMPlexUninterpolate(), DMPlexCreateFromCellList()
@*/
PetscErrorCode DMPlexInterpolate(DM dm, DM *dmInt)
{
  DM             idm, odm = dm;
  PetscInt       depth, dim, d;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
  if (dim <= 1) {
    ierr = PetscObjectReference((PetscObject) dm);CHKERRQ(ierr);
    idm  = dm;
  }
  for (d = 1; d < dim; ++d) {
    /* Create interpolated mesh */
    ierr = DMCreate(PetscObjectComm((PetscObject)dm), &idm);CHKERRQ(ierr);
    ierr = DMSetType(idm, DMPLEX);CHKERRQ(ierr);
    ierr = DMPlexSetDimension(idm, dim);CHKERRQ(ierr);
    if (depth > 0) {ierr = DMPlexInterpolateFaces_Internal(odm, 1, idm);CHKERRQ(ierr);}
    if (odm != dm) {ierr = DMDestroy(&odm);CHKERRQ(ierr);}
    odm  = idm;
  }
  *dmInt = idm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexCopyCoordinates"
/*@
  DMPlexCopyCoordinates - Copy coordinates from one mesh to another with the same vertices

  Collective on DM

  Input Parameter:
. dmA - The DMPlex object with initial coordinates

  Output Parameter:
. dmB - The DMPlex object with copied coordinates

  Level: intermediate

  Note: This is typically used when adding pieces other than vertices to a mesh

.keywords: mesh
.seealso: DMCopyLabels(), DMGetCoordinates(), DMGetCoordinatesLocal(), DMGetCoordinateDM(), DMPlexGetCoordinateSection()
@*/
PetscErrorCode DMPlexCopyCoordinates(DM dmA, DM dmB)
{
  Vec            coordinatesA, coordinatesB;
  PetscSection   coordSectionA, coordSectionB;
  PetscScalar   *coordsA, *coordsB;
  PetscInt       spaceDim, vStartA, vStartB, vEndA, vEndB, coordSizeB, v, d;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (dmA == dmB) PetscFunctionReturn(0);
  ierr = DMPlexGetDepthStratum(dmA, 0, &vStartA, &vEndA);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dmB, 0, &vStartB, &vEndB);CHKERRQ(ierr);
  if ((vEndA-vStartA) != (vEndB-vStartB)) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "The number of vertices in first DM %d != %d in the second DM", vEndA-vStartA, vEndB-vStartB);
  ierr = DMPlexGetCoordinateSection(dmA, &coordSectionA);CHKERRQ(ierr);
  ierr = DMPlexGetCoordinateSection(dmB, &coordSectionB);CHKERRQ(ierr);
  ierr = PetscSectionSetNumFields(coordSectionB, 1);CHKERRQ(ierr);
  ierr = PetscSectionGetFieldComponents(coordSectionA, 0, &spaceDim);CHKERRQ(ierr);
  ierr = PetscSectionSetFieldComponents(coordSectionB, 0, spaceDim);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(coordSectionB, vStartB, vEndB);CHKERRQ(ierr);
  for (v = vStartB; v < vEndB; ++v) {
    ierr = PetscSectionSetDof(coordSectionB, v, spaceDim);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldDof(coordSectionB, v, 0, spaceDim);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(coordSectionB);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(coordSectionB, &coordSizeB);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dmA, &coordinatesA);CHKERRQ(ierr);
  ierr = VecCreate(PetscObjectComm((PetscObject) dmB), &coordinatesB);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) coordinatesB, "coordinates");CHKERRQ(ierr);
  ierr = VecSetSizes(coordinatesB, coordSizeB, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetType(coordinatesB,VECSTANDARD);CHKERRQ(ierr);
  ierr = VecGetArray(coordinatesA, &coordsA);CHKERRQ(ierr);
  ierr = VecGetArray(coordinatesB, &coordsB);CHKERRQ(ierr);
  for (v = 0; v < vEndB-vStartB; ++v) {
    for (d = 0; d < spaceDim; ++d) {
      coordsB[v*spaceDim+d] = coordsA[v*spaceDim+d];
    }
  }
  ierr = VecRestoreArray(coordinatesA, &coordsA);CHKERRQ(ierr);
  ierr = VecRestoreArray(coordinatesB, &coordsB);CHKERRQ(ierr);
  ierr = DMSetCoordinatesLocal(dmB, coordinatesB);CHKERRQ(ierr);
  ierr = VecDestroy(&coordinatesB);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexCopyLabels"
/*@
  DMPlexCopyLabels - Copy labels from one mesh to another with a superset of the points

  Collective on DM

  Input Parameter:
. dmA - The DMPlex object with initial labels

  Output Parameter:
. dmB - The DMPlex object with copied labels

  Level: intermediate

  Note: This is typically used when interpolating or otherwise adding to a mesh

.keywords: mesh
.seealso: DMCopyCoordinates(), DMGetCoordinates(), DMGetCoordinatesLocal(), DMGetCoordinateDM(), DMPlexGetCoordinateSection()
@*/
PetscErrorCode DMPlexCopyLabels(DM dmA, DM dmB)
{
  PetscInt       numLabels, l;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (dmA == dmB) PetscFunctionReturn(0);
  ierr = DMPlexGetNumLabels(dmA, &numLabels);CHKERRQ(ierr);
  for (l = 0; l < numLabels; ++l) {
    DMLabel     label, labelNew;
    const char *name;
    PetscBool   flg;

    ierr = DMPlexGetLabelName(dmA, l, &name);CHKERRQ(ierr);
    ierr = PetscStrcmp(name, "depth", &flg);CHKERRQ(ierr);
    if (flg) continue;
    ierr = DMPlexGetLabel(dmA, name, &label);CHKERRQ(ierr);
    ierr = DMLabelDuplicate(label, &labelNew);CHKERRQ(ierr);
    ierr = DMPlexAddLabel(dmB, labelNew);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexUninterpolate"
/*@
  DMPlexUninterpolate - Take in a mesh with all intermediate faces, edges, etc. and return a cell-vertex mesh

  Collective on DM

  Input Parameter:
. dm - The complete DMPlex object

  Output Parameter:
. dmUnint - The DMPlex object with only cells and vertices

  Level: intermediate

.keywords: mesh
.seealso: DMPlexInterpolate(), DMPlexCreateFromCellList()
@*/
PetscErrorCode DMPlexUninterpolate(DM dm, DM *dmUnint)
{
  DM             udm;
  PetscInt       dim, vStart, vEnd, cStart, cEnd, c, maxConeSize = 0, *cone;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
  if (dim <= 1) {
    ierr = PetscObjectReference((PetscObject) dm);CHKERRQ(ierr);
    *dmUnint = dm;
    PetscFunctionReturn(0);
  }
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMCreate(PetscObjectComm((PetscObject) dm), &udm);CHKERRQ(ierr);
  ierr = DMSetType(udm, DMPLEX);CHKERRQ(ierr);
  ierr = DMPlexSetDimension(udm, dim);CHKERRQ(ierr);
  ierr = DMPlexSetChart(udm, cStart, vEnd);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    PetscInt *closure = NULL, closureSize, cl, coneSize = 0;

    ierr = DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    for (cl = 0; cl < closureSize*2; cl += 2) {
      const PetscInt p = closure[cl];

      if ((p >= vStart) && (p < vEnd)) ++coneSize;
    }
    ierr = DMPlexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    ierr = DMPlexSetConeSize(udm, c, coneSize);CHKERRQ(ierr);
    maxConeSize = PetscMax(maxConeSize, coneSize);
  }
  ierr = DMSetUp(udm);CHKERRQ(ierr);
  ierr = PetscMalloc(maxConeSize * sizeof(PetscInt), &cone);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    PetscInt *closure = NULL, closureSize, cl, coneSize = 0;

    ierr = DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    for (cl = 0; cl < closureSize*2; cl += 2) {
      const PetscInt p = closure[cl];

      if ((p >= vStart) && (p < vEnd)) cone[coneSize++] = p;
    }
    ierr = DMPlexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    ierr = DMPlexSetCone(udm, c, cone);CHKERRQ(ierr);
  }
  ierr = PetscFree(cone);CHKERRQ(ierr);
  ierr = DMPlexSymmetrize(udm);CHKERRQ(ierr);
  ierr = DMPlexStratify(udm);CHKERRQ(ierr);
  *dmUnint = udm;
  PetscFunctionReturn(0);
}
