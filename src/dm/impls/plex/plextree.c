#include <petsc-private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/
#include <../src/sys/utils/hash.h>
#include <petsc-private/isimpl.h>
#include <petscsf.h>

/** hierarchy routines */

#undef __FUNCT__
#define __FUNCT__ "DMPlexSetReferenceTree"
/*@
  DMPlexSetReferenceTree - set the reference tree for hierarchically non-conforming meshes.

  Not collective

  Input Parameters:
+ dm - The DMPlex object
- ref - The reference tree DMPlex object

  Level: intermediate

.seealso: DMPlexGetReferenceTree(), DMPlexCreateDefaultReferenceTree()
@*/
PetscErrorCode DMPlexSetReferenceTree(DM dm, DM ref)
{
  DM_Plex        *mesh = (DM_Plex *)dm->data;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(ref, DM_CLASSID, 2);
  ierr = PetscObjectReference((PetscObject)ref);CHKERRQ(ierr);
  ierr = DMDestroy(&mesh->referenceTree);CHKERRQ(ierr);
  mesh->referenceTree = ref;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetReferenceTree"
/*@
  DMPlexGetReferenceTree - get the reference tree for hierarchically non-conforming meshes.

  Not collective

  Input Parameters:
. dm - The DMPlex object

  Output Parameters
. ref - The reference tree DMPlex object

  Level: intermediate

.seealso: DMPlexSetReferenceTree(), DMPlexCreateDefaultReferenceTree()
@*/
PetscErrorCode DMPlexGetReferenceTree(DM dm, DM *ref)
{
  DM_Plex        *mesh = (DM_Plex *)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(ref,2);
  *ref = mesh->referenceTree;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexCreateDefaultReferenceTree"
/*@
  DMPlexCreateDefaultReferenceTree - create a reference tree for isotropic hierarchical mesh refinement.

  Collective on comm

  Input Parameters:
+ comm    - the MPI communicator
. dim     - the spatial dimension
- simplex - Flag for simplex, otherwise use a tensor-product cell

  Output Parameters:
. ref     - the reference tree DMPlex object

  Level: intermediate

.keywords: reference cell
.seealso: DMPlexSetReferenceTree(), DMPlexGetReferenceTree()
@*/
PetscErrorCode DMPlexCreateDefaultReferenceTree(MPI_Comm comm, PetscInt dim, PetscBool simplex, DM *ref)
{
  DM             K, Kref;
  PetscInt       p, pStart, pEnd, pRefStart, pRefEnd, d, offset, parentSize, *parents, *childIDs;
  PetscInt      *permvals, *unionCones, *coneSizes, *unionOrientations, numUnionPoints, *numDimPoints, numCones, numVerts;
  DMLabel        identity, identityRef;
  PetscSection   unionSection, unionConeSection, parentSection;
  PetscScalar   *unionCoords;
  IS             perm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* create a reference element */
  ierr = DMPlexCreateReferenceCell(comm, dim, simplex, &K);CHKERRQ(ierr);
  ierr = DMPlexCreateLabel(K, "identity");CHKERRQ(ierr);
  ierr = DMPlexGetLabel(K, "identity", &identity);CHKERRQ(ierr);
  ierr = DMPlexGetChart(K, &pStart, &pEnd);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; p++) {
    ierr = DMLabelSetValue(identity, p, p);CHKERRQ(ierr);
  }
  /* refine it */
  ierr = DMRefine(K,comm,&Kref);CHKERRQ(ierr);

  /* the reference tree is the union of these two, without duplicating
   * points that appear in both */
  ierr = DMPlexGetLabel(Kref, "identity", &identityRef);CHKERRQ(ierr);
  ierr = DMPlexGetChart(Kref, &pRefStart, &pRefEnd);CHKERRQ(ierr);
  ierr = PetscSectionCreate(comm, &unionSection);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(unionSection, 0, (pEnd - pStart) + (pRefEnd - pRefStart));CHKERRQ(ierr);
  /* count points that will go in the union */
  for (p = pStart; p < pEnd; p++) {
    ierr = PetscSectionSetDof(unionSection, p - pStart, 1);CHKERRQ(ierr);
  }
  for (p = pRefStart; p < pRefEnd; p++) {
    PetscInt q, qSize;
    ierr = DMLabelGetValue(identityRef, p, &q);CHKERRQ(ierr);
    ierr = DMLabelGetStratumSize(identityRef, q, &qSize);CHKERRQ(ierr);
    if (qSize > 1) {
      ierr = PetscSectionSetDof(unionSection, p - pRefStart + (pEnd - pStart), 1);CHKERRQ(ierr);
    }
  }
  ierr = PetscMalloc1((pEnd - pStart) + (pRefEnd - pRefStart),&permvals);CHKERRQ(ierr);
  offset = 0;
  /* stratify points in the union by topological dimension */
  for (d = 0; d <= dim; d++) {
    PetscInt cStart, cEnd, c;

    ierr = DMPlexGetHeightStratum(K, d, &cStart, &cEnd);CHKERRQ(ierr);
    for (c = cStart; c < cEnd; c++) {
      permvals[offset++] = c;
    }

    ierr = DMPlexGetHeightStratum(Kref, d, &cStart, &cEnd);CHKERRQ(ierr);
    for (c = cStart; c < cEnd; c++) {
      permvals[offset++] = c + (pEnd - pStart);
    }
  }
  ierr = ISCreateGeneral(comm, (pEnd - pStart) + (pRefEnd - pRefStart), permvals, PETSC_OWN_POINTER, &perm);CHKERRQ(ierr);
  ierr = PetscSectionSetPermutation(unionSection,perm);CHKERRQ(ierr);
  ierr = PetscSectionSetUp(unionSection);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(unionSection,&numUnionPoints);CHKERRQ(ierr);
  ierr = PetscMalloc2(numUnionPoints,&coneSizes,dim+1,&numDimPoints);CHKERRQ(ierr);
  /* count dimension points */
  for (d = 0; d <= dim; d++) {
    PetscInt cStart, cOff, cOff2;
    ierr = DMPlexGetHeightStratum(K,d,&cStart,NULL);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(unionSection,cStart-pStart,&cOff);CHKERRQ(ierr);
    if (d < dim) {
      ierr = DMPlexGetHeightStratum(K,d+1,&cStart,NULL);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(unionSection,cStart-pStart,&cOff2);CHKERRQ(ierr);
    }
    else {
      cOff2 = numUnionPoints;
    }
    numDimPoints[dim - d] = cOff2 - cOff;
  }
  ierr = PetscSectionCreate(comm, &unionConeSection);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(unionConeSection, 0, numUnionPoints);CHKERRQ(ierr);
  /* count the cones in the union */
  for (p = pStart; p < pEnd; p++) {
    PetscInt dof, uOff;

    ierr = DMPlexGetConeSize(K, p, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(unionSection, p - pStart,&uOff);CHKERRQ(ierr);
    ierr = PetscSectionSetDof(unionConeSection, uOff, dof);CHKERRQ(ierr);
    coneSizes[uOff] = dof;
  }
  for (p = pRefStart; p < pRefEnd; p++) {
    PetscInt dof, uDof, uOff;

    ierr = DMPlexGetConeSize(Kref, p, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetDof(unionSection, p - pRefStart + (pEnd - pStart),&uDof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(unionSection, p - pRefStart + (pEnd - pStart),&uOff);CHKERRQ(ierr);
    if (uDof) {
      ierr = PetscSectionSetDof(unionConeSection, uOff, dof);CHKERRQ(ierr);
      coneSizes[uOff] = dof;
    }
  }
  ierr = PetscSectionSetUp(unionConeSection);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(unionConeSection,&numCones);CHKERRQ(ierr);
  ierr = PetscMalloc2(numCones,&unionCones,numCones,&unionOrientations);CHKERRQ(ierr);
  /* write the cones in the union */
  for (p = pStart; p < pEnd; p++) {
    PetscInt dof, uOff, c, cOff;
    const PetscInt *cone, *orientation;

    ierr = DMPlexGetConeSize(K, p, &dof);CHKERRQ(ierr);
    ierr = DMPlexGetCone(K, p, &cone);CHKERRQ(ierr);
    ierr = DMPlexGetConeOrientation(K, p, &orientation);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(unionSection, p - pStart,&uOff);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(unionConeSection,uOff,&cOff);CHKERRQ(ierr);
    for (c = 0; c < dof; c++) {
      PetscInt e, eOff;
      e                           = cone[c];
      ierr                        = PetscSectionGetOffset(unionSection, e - pStart, &eOff);CHKERRQ(ierr);
      unionCones[cOff + c]        = eOff;
      unionOrientations[cOff + c] = orientation[c];
    }
  }
  for (p = pRefStart; p < pRefEnd; p++) {
    PetscInt dof, uDof, uOff, c, cOff;
    const PetscInt *cone, *orientation;

    ierr = DMPlexGetConeSize(Kref, p, &dof);CHKERRQ(ierr);
    ierr = DMPlexGetCone(Kref, p, &cone);CHKERRQ(ierr);
    ierr = DMPlexGetConeOrientation(Kref, p, &orientation);CHKERRQ(ierr);
    ierr = PetscSectionGetDof(unionSection, p - pRefStart + (pEnd - pStart),&uDof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(unionSection, p - pRefStart + (pEnd - pStart),&uOff);CHKERRQ(ierr);
    if (uDof) {
      ierr = PetscSectionGetOffset(unionConeSection,uOff,&cOff);CHKERRQ(ierr);
      for (c = 0; c < dof; c++) {
        PetscInt e, eOff, eDof;

        e    = cone[c];
        ierr = PetscSectionGetDof(unionSection, e - pRefStart + (pEnd - pStart),&eDof);CHKERRQ(ierr);
        if (eDof) {
          ierr = PetscSectionGetOffset(unionSection, e - pRefStart + (pEnd - pStart), &eOff);CHKERRQ(ierr);
        }
        else {
          ierr = DMLabelGetValue(identityRef, e, &e);CHKERRQ(ierr);
          ierr = PetscSectionGetOffset(unionSection, e - pStart, &eOff);CHKERRQ(ierr);
        }
        unionCones[cOff + c]        = eOff;
        unionOrientations[cOff + c] = orientation[c];
      }
    }
  }
  /* get the coordinates */
  {
    PetscInt vStart, vEnd, vRefStart, vRefEnd, v, vDof, vOff;
    PetscSection KcoordsSec, KrefCoordsSec;
    Vec      KcoordsVec, KrefCoordsVec;
    PetscScalar *Kcoords;

    DMGetCoordinateSection(K, &KcoordsSec);CHKERRQ(ierr);
    DMGetCoordinatesLocal(K, &KcoordsVec);CHKERRQ(ierr);
    DMGetCoordinateSection(Kref, &KrefCoordsSec);CHKERRQ(ierr);
    DMGetCoordinatesLocal(Kref, &KrefCoordsVec);CHKERRQ(ierr);

    numVerts = numDimPoints[0];
    ierr     = PetscMalloc1(numVerts * dim,&unionCoords);CHKERRQ(ierr);
    ierr     = DMPlexGetDepthStratum(K,0,&vStart,&vEnd);CHKERRQ(ierr);

    offset = 0;
    for (v = vStart; v < vEnd; v++) {
      ierr = PetscSectionGetOffset(unionSection,v - pStart,&vOff);CHKERRQ(ierr);
      ierr = VecGetValuesSection(KcoordsVec, KcoordsSec, v, &Kcoords);CHKERRQ(ierr);
      for (d = 0; d < dim; d++) {
        unionCoords[offset * dim + d] = Kcoords[d];
      }
      offset++;
    }
    ierr = DMPlexGetDepthStratum(Kref,0,&vRefStart,&vRefEnd);CHKERRQ(ierr);
    for (v = vRefStart; v < vRefEnd; v++) {
      ierr = PetscSectionGetDof(unionSection,v - pRefStart + (pEnd - pStart),&vDof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(unionSection,v - pRefStart + (pEnd - pStart),&vOff);CHKERRQ(ierr);
      ierr = VecGetValuesSection(KrefCoordsVec, KrefCoordsSec, v, &Kcoords);CHKERRQ(ierr);
      if (vDof) {
        for (d = 0; d < dim; d++) {
          unionCoords[offset * dim + d] = Kcoords[d];
        }
        offset++;
      }
    }
  }
  ierr = DMCreate(comm,ref);CHKERRQ(ierr);
  ierr = DMSetType(*ref,DMPLEX);CHKERRQ(ierr);
  ierr = DMPlexSetDimension(*ref,dim);CHKERRQ(ierr);
  ierr = DMPlexCreateFromDAG(*ref,dim,numDimPoints,coneSizes,unionCones,unionOrientations,unionCoords);CHKERRQ(ierr);
  /* set the tree */
  ierr = PetscSectionCreate(comm,&parentSection);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(parentSection,0,numUnionPoints);CHKERRQ(ierr);
  for (p = pRefStart; p < pRefEnd; p++) {
    PetscInt uDof, uOff;

    ierr = PetscSectionGetDof(unionSection, p - pRefStart + (pEnd - pStart),&uDof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(unionSection, p - pRefStart + (pEnd - pStart),&uOff);CHKERRQ(ierr);
    if (uDof) {
      PetscSectionSetDof(parentSection,uOff,1);CHKERRQ(ierr);
    }
  }
  ierr = PetscSectionSetUp(parentSection);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(parentSection,&parentSize);CHKERRQ(ierr);
  ierr = PetscMalloc2(parentSize,&parents,parentSize,&childIDs);CHKERRQ(ierr);
  for (p = pRefStart; p < pRefEnd; p++) {
    PetscInt uDof, uOff;

    ierr = PetscSectionGetDof(unionSection, p - pRefStart + (pEnd - pStart),&uDof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(unionSection, p - pRefStart + (pEnd - pStart),&uOff);CHKERRQ(ierr);
    if (uDof) {
      PetscInt pOff, parent, parentU;
      PetscSectionGetOffset(parentSection,uOff,&pOff);CHKERRQ(ierr);
      DMLabelGetValue(identityRef,p,&parent);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(unionSection, parent - pStart,&parentU);CHKERRQ(ierr);
      parents[pOff] = parentU;
      childIDs[pOff] = uOff;
    }
  }
  ierr = DMPlexSetTree(*ref,parentSection,parents,childIDs);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&parentSection);CHKERRQ(ierr);
  ierr = PetscFree2(parents,childIDs);CHKERRQ(ierr);

  /* clean up */
  ierr = PetscSectionDestroy(&unionSection);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&unionConeSection);CHKERRQ(ierr);
  ierr = ISDestroy(&perm);CHKERRQ(ierr);
  ierr = PetscFree(unionCoords);CHKERRQ(ierr);
  ierr = PetscFree2(unionCones,unionOrientations);CHKERRQ(ierr);
  ierr = PetscFree2(coneSizes,numDimPoints);CHKERRQ(ierr);
  ierr = DMDestroy(&K);CHKERRQ(ierr);
  ierr = DMDestroy(&Kref);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexTreeSymmetrize"
static PetscErrorCode DMPlexTreeSymmetrize(DM dm)
{
  DM_Plex        *mesh = (DM_Plex *)dm->data;
  PetscSection   childSec, pSec;
  PetscInt       p, pSize, cSize, parMax = PETSC_MIN_INT, parMin = PETSC_MAX_INT;
  PetscInt       *offsets, *children, pStart, pEnd;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = PetscSectionDestroy(&mesh->childSection);CHKERRQ(ierr);
  ierr = PetscFree(mesh->children);CHKERRQ(ierr);
  pSec = mesh->parentSection;
  if (!pSec) PetscFunctionReturn(0);
  ierr = PetscSectionGetStorageSize(pSec,&pSize);CHKERRQ(ierr);
  for (p = 0; p < pSize; p++) {
    PetscInt par = mesh->parents[p];

    parMax = PetscMax(parMax,par+1);
    parMin = PetscMin(parMin,par);
  }
  if (parMin > parMax) {
    parMin = -1;
    parMax = -1;
  }
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject)pSec),&childSec);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(childSec,parMin,parMax);CHKERRQ(ierr);
  for (p = 0; p < pSize; p++) {
    PetscInt par = mesh->parents[p];

    ierr = PetscSectionAddDof(childSec,par,1);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(childSec);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(childSec,&cSize);CHKERRQ(ierr);
  ierr = PetscMalloc1(cSize,&children);CHKERRQ(ierr);
  ierr = PetscCalloc1(parMax-parMin,&offsets);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(pSec,&pStart,&pEnd);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; p++) {
    PetscInt dof, off, i;

    ierr = PetscSectionGetDof(pSec,p,&dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(pSec,p,&off);CHKERRQ(ierr);
    for (i = 0; i < dof; i++) {
      PetscInt par = mesh->parents[off + i], cOff;

      ierr = PetscSectionGetOffset(childSec,par,&cOff);CHKERRQ(ierr);
      children[cOff + offsets[par-parMin]++] = p;
    }
  }
  mesh->childSection = childSec;
  mesh->children = children;
  ierr = PetscFree(offsets);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexComputeConstraints_Tree"
static PetscErrorCode DMPlexComputeConstraints_Tree(DM dm)
{
  PetscInt       p, pStart, pEnd, *anchors, size;
  PetscInt       aMin = PETSC_MAX_INT, aMax = PETSC_MIN_INT;
  PetscSection   aSec;
  IS             aIS;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMPlexGetChart(dm,&pStart,&pEnd);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; p++) {
    PetscInt parent;

    ierr = DMPlexGetTreeParent(dm,p,&parent,NULL);CHKERRQ(ierr);
    if (parent != p) {
      aMin = PetscMin(aMin,p);
      aMax = PetscMax(aMax,p+1);
    }
  }
  if (aMin > aMax) {
    aMin = -1;
    aMax = -1;
  }
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject)dm),&aSec);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(aSec,aMin,aMax);CHKERRQ(ierr);
  for (p = aMin; p < aMax; p++) {
    PetscInt parent, ancestor = p;

    ierr = DMPlexGetTreeParent(dm,p,&parent,NULL);CHKERRQ(ierr);
    while (parent != ancestor) {
      ancestor = parent;
      ierr     = DMPlexGetTreeParent(dm,ancestor,&parent,NULL);CHKERRQ(ierr);
    }
    if (ancestor != p) {
      PetscInt closureSize, *closure = NULL;

      ierr = DMPlexGetTransitiveClosure(dm,ancestor,PETSC_TRUE,&closureSize,&closure);CHKERRQ(ierr);
      ierr = PetscSectionSetDof(aSec,p,closureSize);CHKERRQ(ierr);
      ierr = DMPlexRestoreTransitiveClosure(dm,ancestor,PETSC_TRUE,&closureSize,&closure);CHKERRQ(ierr);
    }
  }
  ierr = PetscSectionSetUp(aSec);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(aSec,&size);CHKERRQ(ierr);
  ierr = PetscMalloc1(size,&anchors);CHKERRQ(ierr);
  for (p = aMin; p < aMax; p++) {
    PetscInt parent, ancestor = p;

    ierr = DMPlexGetTreeParent(dm,p,&parent,NULL);CHKERRQ(ierr);
    while (parent != ancestor) {
      ancestor = parent;
      ierr     = DMPlexGetTreeParent(dm,ancestor,&parent,NULL);CHKERRQ(ierr);
    }
    if (ancestor != p) {
      PetscInt j, closureSize, *closure = NULL, aOff;

      ierr = PetscSectionGetOffset(aSec,p,&aOff);CHKERRQ(ierr);

      ierr = DMPlexGetTransitiveClosure(dm,ancestor,PETSC_TRUE,&closureSize,&closure);CHKERRQ(ierr);
      for (j = 0; j < closureSize; j++) {
        anchors[aOff + j] = closure[2*j];
      }
      ierr = DMPlexRestoreTransitiveClosure(dm,ancestor,PETSC_TRUE,&closureSize,&closure);CHKERRQ(ierr);
    }
  }
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject)dm),size,anchors,PETSC_OWN_POINTER,&aIS);CHKERRQ(ierr);
  ierr = DMPlexSetConstraints(dm,aSec,aIS);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&aSec);CHKERRQ(ierr);
  ierr = ISDestroy(&aIS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexSetTree"
/*@
  DMPlexSetTree - set the tree that describes the hierarchy of non-conforming mesh points.  This routine also creates
  the point-to-point constraints determined by the tree: a point is constained to the points in the closure of its
  tree root.

  Collective on dm

  Input Parameters:
+ dm - the DMPlex object
. parentSection - a section describing the tree: a point has a parent if it has 1 dof in the section; the section
                  offset indexes the parent and childID list; the reference count of parentSection is incremented
. parents - a list of the point parents; copied, can be destroyed
- childIDs - identifies the relationship of the child point to the parent point; if there is a reference tree, then
             the child corresponds to the point in the reference tree with index childIDs; copied, can be destroyed

  Level: intermediate

.seealso: DMPlexGetTree(), DMPlexSetReferenceTree(), DMPlexSetConstraints(), DMPlexGetTreeParent(), DMPlexGetTreeChildren()
@*/
PetscErrorCode DMPlexSetTree(DM dm, PetscSection parentSection, PetscInt parents[], PetscInt childIDs[])
{
  DM_Plex       *mesh = (DM_Plex *)dm->data;
  PetscInt       size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(parentSection, PETSC_SECTION_CLASSID, 2);
  ierr = PetscObjectReference((PetscObject)parentSection);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&mesh->parentSection);CHKERRQ(ierr);
  mesh->parentSection = parentSection;
  ierr = PetscSectionGetStorageSize(parentSection,&size);CHKERRQ(ierr);
  if (parents != mesh->parents) {
    ierr = PetscFree(mesh->parents);CHKERRQ(ierr);
    ierr = PetscMalloc1(size,&mesh->parents);CHKERRQ(ierr);
    ierr = PetscMemcpy(mesh->parents, parents, size * sizeof(*parents));CHKERRQ(ierr);
  }
  if (childIDs != mesh->childIDs) {
    ierr = PetscFree(mesh->childIDs);CHKERRQ(ierr);
    ierr = PetscMalloc1(size,&mesh->childIDs);CHKERRQ(ierr);
    ierr = PetscMemcpy(mesh->childIDs, childIDs, size * sizeof(*childIDs));CHKERRQ(ierr);
  }
  ierr = DMPlexTreeSymmetrize(dm);CHKERRQ(ierr);
  ierr = DMPlexComputeConstraints_Tree(dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetTree"
/*@
  DMPlexGetTree - get the tree that describes the hierarchy of non-conforming mesh points.
  Collective on dm

  Input Parameters:
. dm - the DMPlex object

  Output Parameters:
+ parentSection - a section describing the tree: a point has a parent if it has 1 dof in the section; the section
                  offset indexes the parent and childID list
. parents - a list of the point parents
. childIDs - identifies the relationship of the child point to the parent point; if there is a reference tree, then
             the child corresponds to the point in the reference tree with index childID
. childSection - the inverse of the parent section
- children - a list of the point children

  Level: intermediate

.seealso: DMPlexSetTree(), DMPlexSetReferenceTree(), DMPlexSetConstraints(), DMPlexGetTreeParent(), DMPlexGetTreeChildren()
@*/
PetscErrorCode DMPlexGetTree(DM dm, PetscSection *parentSection, PetscInt *parents[], PetscInt *childIDs[], PetscSection *childSection, PetscInt *children[])
{
  DM_Plex        *mesh = (DM_Plex *)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (parentSection) *parentSection = mesh->parentSection;
  if (parents)       *parents       = mesh->parents;
  if (childIDs)      *childIDs      = mesh->childIDs;
  if (childSection)  *childSection  = mesh->childSection;
  if (children)      *children      = mesh->children;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetTreeParent"
/*@
  DMPlexGetTreeParent - get the parent of a point in the tree describing the point hierarchy (not the Sieve DAG)

  Input Parameters:
+ dm - the DMPlex object
- point - the query point

  Output Parameters:
+ parent - if not NULL, set to the parent of the point, or the point itself if the point does not have a parent
- childID - if not NULL, set to the child ID of the point with respect to its parent, or 0 if the point
            does not have a parent

  Level: intermediate

.seealso: DMPlexSetTree(), DMPlexGetTree(), DMPlexGetTreeChildren()
@*/
PetscErrorCode DMPlexGetTreeParent(DM dm, PetscInt point, PetscInt *parent, PetscInt *childID)
{
  DM_Plex       *mesh = (DM_Plex *)dm->data;
  PetscSection   pSec;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  pSec = mesh->parentSection;
  if (pSec && point >= pSec->pStart && point < pSec->pEnd) {
    PetscInt dof;

    ierr = PetscSectionGetDof (pSec, point, &dof);CHKERRQ(ierr);
    if (dof) {
      PetscInt off;

      ierr = PetscSectionGetOffset (pSec, point, &off);CHKERRQ(ierr);
      if (parent)  *parent = mesh->parents[off];
      if (childID) *childID = mesh->childIDs[off];
      PetscFunctionReturn(0);
    }
  }
  if (parent) {
    *parent = point;
  }
  if (childID) {
    *childID = 0;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetTreeChildren"
/*@C
  DMPlexGetTreeChildren - get the children of a point in the tree describing the point hierarchy (not the Sieve DAG)

  Input Parameters:
+ dm - the DMPlex object
- point - the query point

  Output Parameters:
+ numChildren - if not NULL, set to the number of children
- children - if not NULL, set to a list children, or set to NULL if the point has no children

  Level: intermediate

  Fortran Notes:
  Since it returns an array, this routine is only available in Fortran 90, and you must
  include petsc.h90 in your code.

.seealso: DMPlexSetTree(), DMPlexGetTree(), DMPlexGetTreeParent()
@*/
PetscErrorCode DMPlexGetTreeChildren(DM dm, PetscInt point, PetscInt *numChildren, const PetscInt *children[])
{
  DM_Plex       *mesh = (DM_Plex *)dm->data;
  PetscSection   childSec;
  PetscInt       dof = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  childSec = mesh->childSection;
  if (childSec && point >= childSec->pStart && point < childSec->pEnd) {
    ierr = PetscSectionGetDof (childSec, point, &dof);CHKERRQ(ierr);
  }
  if (numChildren) *numChildren = dof;
  if (children) {
    if (dof) {
      PetscInt off;

      ierr = PetscSectionGetOffset (childSec, point, &off);CHKERRQ(ierr);
      *children = &mesh->children[off];
    }
    else {
      *children = NULL;
    }
  }
  PetscFunctionReturn(0);
}
