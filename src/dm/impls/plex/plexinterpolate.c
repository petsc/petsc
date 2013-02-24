#include <petsc-private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/
#include <../src/sys/utils/hash.h>

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetFaces"
/*
  DMPlexGetFaces -

  Note: This will only work for cell-vertex meshes.
*/
static PetscErrorCode DMPlexGetFaces(DM dm, PetscInt p, PetscInt *numFaces, PetscInt *faceSize, const PetscInt *faces[])
{
  DM_Plex        *mesh = (DM_Plex*) dm->data;
  const PetscInt *cone = NULL;
  PetscInt        depth = 0, dim, coneSize;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  if (depth > 1) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Faces can only be returned for cell-vertex meshes.");
  if (!mesh->facesTmp) {ierr = PetscMalloc(PetscSqr(PetscMax(mesh->maxConeSize, mesh->maxSupportSize)) * sizeof(PetscInt), &mesh->facesTmp);CHKERRQ(ierr);}
  ierr = DMPlexGetConeSize(dm, p, &coneSize);CHKERRQ(ierr);
  ierr = DMPlexGetCone(dm, p, &cone);CHKERRQ(ierr);
  switch (dim) {
  case 2:
    switch (coneSize) {
    case 3:
      mesh->facesTmp[0] = cone[0]; mesh->facesTmp[1] = cone[1];
      mesh->facesTmp[2] = cone[1]; mesh->facesTmp[3] = cone[2];
      mesh->facesTmp[4] = cone[2]; mesh->facesTmp[5] = cone[0];
      *numFaces         = 3;
      *faceSize         = 2;
      *faces            = mesh->facesTmp;
      break;
    case 4:
      mesh->facesTmp[0] = cone[0]; mesh->facesTmp[1] = cone[1];
      mesh->facesTmp[2] = cone[1]; mesh->facesTmp[3] = cone[2];
      mesh->facesTmp[4] = cone[2]; mesh->facesTmp[5] = cone[3];
      mesh->facesTmp[6] = cone[3]; mesh->facesTmp[7] = cone[0];
      *numFaces         = 4;
      *faceSize         = 2;
      *faces            = mesh->facesTmp;
      break;
    default:
      SETERRQ2(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Cone size %D not supported for dimension %D", coneSize, dim);
    }
    break;
  case 3:
    switch (coneSize) {
    case 3:
      mesh->facesTmp[0] = cone[0]; mesh->facesTmp[1] = cone[1];
      mesh->facesTmp[2] = cone[1]; mesh->facesTmp[3] = cone[2];
      mesh->facesTmp[4] = cone[2]; mesh->facesTmp[5] = cone[0];
      *numFaces         = 3;
      *faceSize         = 2;
      *faces            = mesh->facesTmp;
      break;
    case 4:
      mesh->facesTmp[0] = cone[0]; mesh->facesTmp[1]  = cone[1]; mesh->facesTmp[2]  = cone[2];
      mesh->facesTmp[3] = cone[0]; mesh->facesTmp[4]  = cone[2]; mesh->facesTmp[5]  = cone[3];
      mesh->facesTmp[6] = cone[0]; mesh->facesTmp[7]  = cone[3]; mesh->facesTmp[8]  = cone[1];
      mesh->facesTmp[9] = cone[1]; mesh->facesTmp[10] = cone[3]; mesh->facesTmp[11] = cone[2];
      *numFaces         = 4;
      *faceSize         = 3;
      *faces            = mesh->facesTmp;
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
#define __FUNCT__ "DMPlexInterpolate_2D"
static PetscErrorCode DMPlexInterpolate_2D(DM dm, DM *dmInt)
{
  DM             idm;
  DM_Plex       *mesh;
  PetscHashIJ    edgeTable;
  PetscInt      *off;
  PetscInt       dim, numCells, cStart, cEnd, c, numVertices, vStart, vEnd;
  PetscInt       numEdges, firstEdge, edge, e;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr        = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr        = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr        = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  numCells    = cEnd - cStart;
  numVertices = vEnd - vStart;
  firstEdge   = numCells + numVertices;
  numEdges    = 0;
  /* Count edges using algorithm from CreateNeighborCSR */
  ierr = DMPlexCreateNeighborCSR(dm, NULL, &off, NULL);CHKERRQ(ierr);
  if (off) {
    PetscInt numCorners = 0;

    numEdges = off[numCells]/2;
#if 0
    /* Account for boundary edges: \sum_c 3 - neighbors = 3*numCells - totalNeighbors */
    numEdges += 3*numCells - off[numCells];
#else
    /* Account for boundary edges: \sum_c #faces - #neighbors = \sum_c #cellVertices - #neighbors = totalCorners - totalNeighbors */
    for (c = cStart; c < cEnd; ++c) {
      PetscInt coneSize;

      ierr        = DMPlexGetConeSize(dm, c, &coneSize);CHKERRQ(ierr);
      numCorners += coneSize;
    }
    numEdges += numCorners - off[numCells];
#endif
  }
#if 0
  /* Check Euler characteristic V - E + F = 1 */
  if (numVertices && (numVertices-numEdges+numCells != 1)) SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Euler characteristic of mesh is %d  != 1", numVertices-numEdges+numCells);
#endif
  /* Create interpolated mesh */
  ierr = DMCreate(PetscObjectComm((PetscObject)dm), &idm);CHKERRQ(ierr);
  ierr = DMSetType(idm, DMPLEX);CHKERRQ(ierr);
  ierr = DMPlexSetDimension(idm, dim);CHKERRQ(ierr);
  ierr = DMPlexSetChart(idm, 0, numCells+numVertices+numEdges);CHKERRQ(ierr);
  for (c = 0; c < numCells; ++c) {
    PetscInt numCorners;

    ierr = DMPlexGetConeSize(dm, c, &numCorners);CHKERRQ(ierr);
    ierr = DMPlexSetConeSize(idm, c, numCorners);CHKERRQ(ierr);
  }
  for (e = firstEdge; e < firstEdge+numEdges; ++e) {
    ierr = DMPlexSetConeSize(idm, e, 2);CHKERRQ(ierr);
  }
  ierr = DMSetUp(idm);CHKERRQ(ierr);
  /* Get edge cones from subsets of cell vertices */
  ierr = PetscHashIJCreate(&edgeTable);CHKERRQ(ierr);
  ierr = PetscHashIJSetMultivalued(edgeTable, PETSC_FALSE);CHKERRQ(ierr);

  for (c = 0, edge = firstEdge; c < numCells; ++c) {
    const PetscInt *cellFaces;
    PetscInt        numCellFaces, faceSize, cf;

    ierr = DMPlexGetFaces(dm, c, &numCellFaces, &faceSize, &cellFaces);CHKERRQ(ierr);
    if (faceSize != 2) SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Triangles cannot have face of size %D", faceSize);
    for (cf = 0; cf < numCellFaces; ++cf) {
#if 1
      PetscHashIJKey key;

      key.i = PetscMin(cellFaces[cf*faceSize+0], cellFaces[cf*faceSize+1]);
      key.j = PetscMax(cellFaces[cf*faceSize+0], cellFaces[cf*faceSize+1]);
      ierr  = PetscHashIJGet(edgeTable, key, &e);CHKERRQ(ierr);
      if (e < 0) {
        ierr = DMPlexSetCone(idm, edge, &cellFaces[cf*faceSize]);CHKERRQ(ierr);
        ierr = PetscHashIJAdd(edgeTable, key, edge);CHKERRQ(ierr);
        e    = edge++;
      }
#else
      PetscBool found = PETSC_FALSE;

      /* TODO Need join of vertices to check for existence of edges, which needs support (could set edge support), so just brute force for now */
      for (e = firstEdge; e < edge; ++e) {
        const PetscInt *cone;

        ierr = DMPlexGetCone(idm, e, &cone);CHKERRQ(ierr);
        if (((cellFaces[cf*faceSize+0] == cone[0]) && (cellFaces[cf*faceSize+1] == cone[1])) ||
            ((cellFaces[cf*faceSize+0] == cone[1]) && (cellFaces[cf*faceSize+1] == cone[0]))) {
          found = PETSC_TRUE;
          break;
        }
      }
      if (!found) {
        ierr = DMPlexSetCone(idm, edge, &cellFaces[cf*faceSize]);CHKERRQ(ierr);
        ++edge;
      }
#endif
      ierr = DMPlexInsertCone(idm, c, cf, e);CHKERRQ(ierr);
    }
  }
  if (edge != firstEdge+numEdges) SETERRQ2(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Invalid number of edges %D should be %D", edge-firstEdge, numEdges);
  ierr = PetscHashIJDestroy(&edgeTable);CHKERRQ(ierr);
  ierr = PetscFree(off);CHKERRQ(ierr);
  ierr = DMPlexSymmetrize(idm);CHKERRQ(ierr);
  ierr = DMPlexStratify(idm);CHKERRQ(ierr);
  mesh = (DM_Plex*) (idm)->data;
  /* Orient edges */
  for (c = 0; c < numCells; ++c) {
    const PetscInt *cone = NULL, *cellFaces;
    PetscInt        coneSize, coff, numCellFaces, faceSize, cf;

    ierr = DMPlexGetConeSize(idm, c, &coneSize);CHKERRQ(ierr);
    ierr = DMPlexGetCone(idm, c, &cone);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(mesh->coneSection, c, &coff);CHKERRQ(ierr);
    ierr = DMPlexGetFaces(dm, c, &numCellFaces, &faceSize, &cellFaces);CHKERRQ(ierr);
    if (coneSize != numCellFaces) SETERRQ3(PetscObjectComm((PetscObject)idm), PETSC_ERR_PLIB, "Invalid number of edges %D for cell %D should be %D", coneSize, c, numCellFaces);
    for (cf = 0; cf < numCellFaces; ++cf) {
      const PetscInt *econe = NULL;
      PetscInt        esize;

      ierr = DMPlexGetConeSize(idm, cone[cf], &esize);CHKERRQ(ierr);
      ierr = DMPlexGetCone(idm, cone[cf], &econe);CHKERRQ(ierr);
      if (esize != 2) SETERRQ2(PetscObjectComm((PetscObject)idm), PETSC_ERR_PLIB, "Invalid number of edge endpoints %D for edge %D should be 2", esize, cone[cf]);
      if ((cellFaces[cf*faceSize+0] == econe[0]) && (cellFaces[cf*faceSize+1] == econe[1])) {
        /* Correctly oriented */
        mesh->coneOrientations[coff+cf] = 0;
      } else if ((cellFaces[cf*faceSize+0] == econe[1]) && (cellFaces[cf*faceSize+1] == econe[0])) {
        /* Start at index 1, and reverse orientation */
        mesh->coneOrientations[coff+cf] = -(1+1);
      }
    }
  }
  *dmInt = idm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexInterpolate_3D"
static PetscErrorCode DMPlexInterpolate_3D(DM dm, DM *dmInt)
{
  DM             idm, fdm;
  DM_Plex       *mesh;
  PetscInt      *off;
  const PetscInt numCorners = 4;
  PetscInt       dim, numCells, cStart, cEnd, c, numVertices, vStart, vEnd;
  PetscInt       numFaces, firstFace, face, f, numEdges, firstEdge, edge, e;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  {
    ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_ASCII_INFO_DETAIL);CHKERRQ(ierr);
    ierr = DMView(dm, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  ierr        = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr        = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr        = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  numCells    = cEnd - cStart;
  numVertices = vEnd - vStart;
  firstFace   = numCells + numVertices;
  numFaces    = 0;
  /* Count faces using algorithm from CreateNeighborCSR */
  ierr = DMPlexCreateNeighborCSR(dm, NULL, &off, NULL);CHKERRQ(ierr);
  if (off) {
    numFaces = off[numCells]/2;
    /* Account for boundary faces: \sum_c 4 - neighbors = 4*numCells - totalNeighbors */
    numFaces += 4*numCells - off[numCells];
  }
  /* Use Euler characteristic to get edges V - E + F - C = 1 */
  firstEdge = firstFace + numFaces;
  numEdges  = numVertices + numFaces - numCells - 1;
  /* Create interpolated mesh */
  ierr = DMCreate(PetscObjectComm((PetscObject)dm), &idm);CHKERRQ(ierr);
  ierr = DMSetType(idm, DMPLEX);CHKERRQ(ierr);
  ierr = DMPlexSetDimension(idm, dim);CHKERRQ(ierr);
  ierr = DMPlexSetChart(idm, 0, numCells+numVertices+numFaces+numEdges);CHKERRQ(ierr);
  for (c = 0; c < numCells; ++c) {
    ierr = DMPlexSetConeSize(idm, c, numCorners);CHKERRQ(ierr);
  }
  for (f = firstFace; f < firstFace+numFaces; ++f) {
    ierr = DMPlexSetConeSize(idm, f, 3);CHKERRQ(ierr);
  }
  for (e = firstEdge; e < firstEdge+numEdges; ++e) {
    ierr = DMPlexSetConeSize(idm, e, 2);CHKERRQ(ierr);
  }
  ierr = DMSetUp(idm);CHKERRQ(ierr);
  /* Get face cones from subsets of cell vertices */
  ierr = DMCreate(PetscObjectComm((PetscObject)dm), &fdm);CHKERRQ(ierr);
  ierr = DMSetType(fdm, DMPLEX);CHKERRQ(ierr);
  ierr = DMPlexSetDimension(fdm, dim);CHKERRQ(ierr);
  ierr = DMPlexSetChart(fdm, numCells, firstFace+numFaces);CHKERRQ(ierr);
  for (f = firstFace; f < firstFace+numFaces; ++f) {
    ierr = DMPlexSetConeSize(fdm, f, 3);CHKERRQ(ierr);
  }
  ierr = DMSetUp(fdm);CHKERRQ(ierr);
  for (c = 0, face = firstFace; c < numCells; ++c) {
    const PetscInt *cellFaces;
    PetscInt        numCellFaces, faceSize, cf;

    ierr = DMPlexGetFaces(dm, c, &numCellFaces, &faceSize, &cellFaces);CHKERRQ(ierr);
    if (faceSize != 3) SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Tetrahedra cannot have face of size %D", faceSize);
    for (cf = 0; cf < numCellFaces; ++cf) {
      PetscBool found = PETSC_FALSE;

      /* TODO Need join of vertices to check for existence of edges, which needs support (could set edge support), so just brute force for now */
      for (f = firstFace; f < face; ++f) {
        const PetscInt *cone = NULL;

        ierr = DMPlexGetCone(idm, f, &cone);CHKERRQ(ierr);
        if (((cellFaces[cf*faceSize+0] == cone[0]) && (cellFaces[cf*faceSize+1] == cone[1]) && (cellFaces[cf*faceSize+2] == cone[2])) ||
            ((cellFaces[cf*faceSize+0] == cone[1]) && (cellFaces[cf*faceSize+1] == cone[2]) && (cellFaces[cf*faceSize+2] == cone[0])) ||
            ((cellFaces[cf*faceSize+0] == cone[2]) && (cellFaces[cf*faceSize+1] == cone[0]) && (cellFaces[cf*faceSize+2] == cone[1])) ||
            ((cellFaces[cf*faceSize+0] == cone[0]) && (cellFaces[cf*faceSize+1] == cone[2]) && (cellFaces[cf*faceSize+2] == cone[1])) ||
            ((cellFaces[cf*faceSize+0] == cone[2]) && (cellFaces[cf*faceSize+1] == cone[1]) && (cellFaces[cf*faceSize+2] == cone[0])) ||
            ((cellFaces[cf*faceSize+0] == cone[1]) && (cellFaces[cf*faceSize+1] == cone[0]) && (cellFaces[cf*faceSize+2] == cone[2]))) {
          found = PETSC_TRUE;
          break;
        }
      }
      if (!found) {
        ierr = DMPlexSetCone(idm, face, &cellFaces[cf*faceSize]);CHKERRQ(ierr);
        /* Save the vertices for orientation calculation */
        ierr = DMPlexSetCone(fdm, face, &cellFaces[cf*faceSize]);CHKERRQ(ierr);
        ++face;
      }
      ierr = DMPlexInsertCone(idm, c, cf, f);CHKERRQ(ierr);
    }
  }
  if (face != firstFace+numFaces) SETERRQ2(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Invalid number of faces %D should be %D", face-firstFace, numFaces);
  /* Get edge cones from subsets of face vertices */
  for (f = firstFace, edge = firstEdge; f < firstFace+numFaces; ++f) {
    const PetscInt *cellFaces;
    PetscInt        numCellFaces, faceSize, cf;

    ierr = DMPlexGetFaces(idm, f, &numCellFaces, &faceSize, &cellFaces);CHKERRQ(ierr);
    if (faceSize != 2) SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Triangles cannot have face of size %D", faceSize);
    for (cf = 0; cf < numCellFaces; ++cf) {
      PetscBool found = PETSC_FALSE;

      /* TODO Need join of vertices to check for existence of edges, which needs support (could set edge support), so just brute force for now */
      for (e = firstEdge; e < edge; ++e) {
        const PetscInt *cone = NULL;

        ierr = DMPlexGetCone(idm, e, &cone);CHKERRQ(ierr);
        if (((cellFaces[cf*faceSize+0] == cone[0]) && (cellFaces[cf*faceSize+1] == cone[1])) ||
            ((cellFaces[cf*faceSize+0] == cone[1]) && (cellFaces[cf*faceSize+1] == cone[0]))) {
          found = PETSC_TRUE;
          break;
        }
      }
      if (!found) {
        ierr = DMPlexSetCone(idm, edge, &cellFaces[cf*faceSize]);CHKERRQ(ierr);
        ++edge;
      }
      ierr = DMPlexInsertCone(idm, f, cf, e);CHKERRQ(ierr);
    }
  }
  if (edge != firstEdge+numEdges) SETERRQ2(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Invalid number of edges %D should be %D", edge-firstEdge, numEdges);
  ierr = PetscFree(off);CHKERRQ(ierr);
  ierr = DMPlexSymmetrize(idm);CHKERRQ(ierr);
  ierr = DMPlexStratify(idm);CHKERRQ(ierr);
  mesh = (DM_Plex*) (idm)->data;
  /* Orient edges */
  for (f = firstFace; f < firstFace+numFaces; ++f) {
    const PetscInt *cone, *cellFaces;
    PetscInt        coneSize, coff, numCellFaces, faceSize, cf;

    ierr = DMPlexGetConeSize(idm, f, &coneSize);CHKERRQ(ierr);
    ierr = DMPlexGetCone(idm, f, &cone);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(mesh->coneSection, f, &coff);CHKERRQ(ierr);
    ierr = DMPlexGetFaces(fdm, f, &numCellFaces, &faceSize, &cellFaces);CHKERRQ(ierr);
    if (coneSize != numCellFaces) SETERRQ3(PetscObjectComm((PetscObject)idm), PETSC_ERR_PLIB, "Invalid number of edges %D for face %D should be %D", coneSize, f, numCellFaces);
    for (cf = 0; cf < numCellFaces; ++cf) {
      const PetscInt *econe;
      PetscInt        esize;

      ierr = DMPlexGetConeSize(idm, cone[cf], &esize);CHKERRQ(ierr);
      ierr = DMPlexGetCone(idm, cone[cf], &econe);CHKERRQ(ierr);
      if (esize != 2) SETERRQ2(PetscObjectComm((PetscObject)idm), PETSC_ERR_PLIB, "Invalid number of edge endpoints %D for edge %D should be 2", esize, cone[cf]);
      if ((cellFaces[cf*faceSize+0] == econe[0]) && (cellFaces[cf*faceSize+1] == econe[1])) {
        /* Correctly oriented */
        mesh->coneOrientations[coff+cf] = 0;
      } else if ((cellFaces[cf*faceSize+0] == econe[1]) && (cellFaces[cf*faceSize+1] == econe[0])) {
        /* Start at index 1, and reverse orientation */
        mesh->coneOrientations[coff+cf] = -(1+1);
      }
    }
  }
  ierr = DMDestroy(&fdm);CHKERRQ(ierr);
  /* Orient faces */
  for (c = 0; c < numCells; ++c) {
    const PetscInt *cone, *cellFaces;
    PetscInt        coneSize, coff, numCellFaces, faceSize, cf;

    ierr = DMPlexGetConeSize(idm, c, &coneSize);CHKERRQ(ierr);
    ierr = DMPlexGetCone(idm, c, &cone);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(mesh->coneSection, c, &coff);CHKERRQ(ierr);
    ierr = DMPlexGetFaces(dm, c, &numCellFaces, &faceSize, &cellFaces);CHKERRQ(ierr);
    if (coneSize != numCellFaces) SETERRQ3(PetscObjectComm((PetscObject)idm), PETSC_ERR_PLIB, "Invalid number of edges %D for cell %D should be %D", coneSize, c, numCellFaces);
    for (cf = 0; cf < numCellFaces; ++cf) {
      PetscInt *origClosure = NULL, *closure;
      PetscInt closureSize, i;

      ierr = DMPlexGetTransitiveClosure(idm, cone[cf], PETSC_TRUE, &closureSize, &origClosure);CHKERRQ(ierr);
      if (closureSize != 7) SETERRQ2(PetscObjectComm((PetscObject)idm), PETSC_ERR_PLIB, "Invalid closure size %D for face %D should be 7", closureSize, cone[cf]);
      for (i = 4; i < 7; ++i) {
        if ((origClosure[i*2] < vStart) || (origClosure[i*2] >= vEnd)) SETERRQ3(PetscObjectComm((PetscObject)idm), PETSC_ERR_PLIB, "Invalid closure point %D should be a vertex in [%D, %D)", origClosure[i*2], vStart, vEnd);
      }
      closure = &origClosure[4*2];
      /* Remember that this is the orientation for edges, not vertices */
      if        ((cellFaces[cf*faceSize+0] == closure[0*2]) && (cellFaces[cf*faceSize+1] == closure[1*2]) && (cellFaces[cf*faceSize+2] == closure[2*2])) {
        /* Correctly oriented */
        mesh->coneOrientations[coff+cf] = 0;
      } else if ((cellFaces[cf*faceSize+0] == closure[1*2]) && (cellFaces[cf*faceSize+1] == closure[2*2]) && (cellFaces[cf*faceSize+2] == closure[0*2])) {
        /* Shifted by 1 */
        mesh->coneOrientations[coff+cf] = 1;
      } else if ((cellFaces[cf*faceSize+0] == closure[2*2]) && (cellFaces[cf*faceSize+1] == closure[0*2]) && (cellFaces[cf*faceSize+2] == closure[1*2])) {
        /* Shifted by 2 */
        mesh->coneOrientations[coff+cf] = 2;
      } else if ((cellFaces[cf*faceSize+0] == closure[2*2]) && (cellFaces[cf*faceSize+1] == closure[1*2]) && (cellFaces[cf*faceSize+2] == closure[0*2])) {
        /* Start at edge 1, and reverse orientation */
        mesh->coneOrientations[coff+cf] = -(1+1);
      } else if ((cellFaces[cf*faceSize+0] == closure[1*2]) && (cellFaces[cf*faceSize+1] == closure[0*2]) && (cellFaces[cf*faceSize+2] == closure[2*2])) {
        /* Start at index 0, and reverse orientation */
        mesh->coneOrientations[coff+cf] = -(0+1);
      } else if ((cellFaces[cf*faceSize+0] == closure[0*2]) && (cellFaces[cf*faceSize+1] == closure[2*2]) && (cellFaces[cf*faceSize+2] == closure[1*2])) {
        /* Start at index 2, and reverse orientation */
        mesh->coneOrientations[coff+cf] = -(2+1);
      } else SETERRQ3(PetscObjectComm((PetscObject)idm), PETSC_ERR_PLIB, "Face %D did not match local face %D in cell %D for any orientation", cone[cf], cf, c);
      ierr = DMPlexRestoreTransitiveClosure(idm, cone[cf], PETSC_TRUE, &closureSize, &origClosure);CHKERRQ(ierr);
    }
  }
  {
    ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_ASCII_INFO_DETAIL);CHKERRQ(ierr);
    ierr = DMView(idm, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  *dmInt = idm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexInterpolate"
PetscErrorCode DMPlexInterpolate(DM dm, DM *dmInt)
{
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
  switch (dim) {
  case 2:
    ierr = DMPlexInterpolate_2D(dm, dmInt);CHKERRQ(ierr);break;
  case 3:
    ierr = DMPlexInterpolate_3D(dm, dmInt);CHKERRQ(ierr);break;
  default:
    SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_OUTOFRANGE, "No mesh interpolation support for dimension %D", dim);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexCopyCoordinates"
PetscErrorCode DMPlexCopyCoordinates(DM dmA, DM dmB)
{
  Vec            coordinatesA, coordinatesB;
  PetscSection   coordSectionA, coordSectionB;
  PetscScalar   *coordsA, *coordsB;
  PetscInt       spaceDim, vStartA, vStartB, vEndA, vEndB, coordSizeB, v, d;
  PetscErrorCode ierr;

  PetscFunctionBegin;
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
  ierr = VecSetFromOptions(coordinatesB);CHKERRQ(ierr);
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
