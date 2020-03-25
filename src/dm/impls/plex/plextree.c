#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/
#include <petsc/private/isimpl.h>
#include <petsc/private/petscfeimpl.h>
#include <petscsf.h>
#include <petscds.h>

/** hierarchy routines */

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
  if (ref) {PetscValidHeaderSpecific(ref, DM_CLASSID, 2);}
  ierr = PetscObjectReference((PetscObject)ref);CHKERRQ(ierr);
  ierr = DMDestroy(&mesh->referenceTree);CHKERRQ(ierr);
  mesh->referenceTree = ref;
  PetscFunctionReturn(0);
}

/*@
  DMPlexGetReferenceTree - get the reference tree for hierarchically non-conforming meshes.

  Not collective

  Input Parameters:
. dm - The DMPlex object

  Output Parameters:
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

static PetscErrorCode DMPlexReferenceTreeGetChildSymmetry_Default(DM dm, PetscInt parent, PetscInt parentOrientA, PetscInt childOrientA, PetscInt childA, PetscInt parentOrientB, PetscInt *childOrientB, PetscInt *childB)
{
  PetscInt       coneSize, dStart, dEnd, dim, ABswap, oAvert, oBvert, ABswapVert;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (parentOrientA == parentOrientB) {
    if (childOrientB) *childOrientB = childOrientA;
    if (childB) *childB = childA;
    PetscFunctionReturn(0);
  }
  for (dim = 0; dim < 3; dim++) {
    ierr = DMPlexGetDepthStratum(dm,dim,&dStart,&dEnd);CHKERRQ(ierr);
    if (parent >= dStart && parent <= dEnd) {
      break;
    }
  }
  if (dim > 2) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot perform child symmetry for %d-cells",dim);
  if (!dim) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"A vertex has no children");
  if (childA < dStart || childA >= dEnd) {
    /* this is a lower-dimensional child: bootstrap */
    PetscInt size, i, sA = -1, sB, sOrientB, sConeSize;
    const PetscInt *supp, *coneA, *coneB, *oA, *oB;

    ierr = DMPlexGetSupportSize(dm,childA,&size);CHKERRQ(ierr);
    ierr = DMPlexGetSupport(dm,childA,&supp);CHKERRQ(ierr);

    /* find a point sA in supp(childA) that has the same parent */
    for (i = 0; i < size; i++) {
      PetscInt sParent;

      sA   = supp[i];
      if (sA == parent) continue;
      ierr = DMPlexGetTreeParent(dm,sA,&sParent,NULL);CHKERRQ(ierr);
      if (sParent == parent) {
        break;
      }
    }
    if (i == size) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"could not find support in children");
    /* find out which point sB is in an equivalent position to sA under
     * parentOrientB */
    ierr = DMPlexReferenceTreeGetChildSymmetry_Default(dm,parent,parentOrientA,0,sA,parentOrientB,&sOrientB,&sB);CHKERRQ(ierr);
    ierr = DMPlexGetConeSize(dm,sA,&sConeSize);CHKERRQ(ierr);
    ierr = DMPlexGetCone(dm,sA,&coneA);CHKERRQ(ierr);
    ierr = DMPlexGetCone(dm,sB,&coneB);CHKERRQ(ierr);
    ierr = DMPlexGetConeOrientation(dm,sA,&oA);CHKERRQ(ierr);
    ierr = DMPlexGetConeOrientation(dm,sB,&oB);CHKERRQ(ierr);
    /* step through the cone of sA in natural order */
    for (i = 0; i < sConeSize; i++) {
      if (coneA[i] == childA) {
        /* if childA is at position i in coneA,
         * then we want the point that is at sOrientB*i in coneB */
        PetscInt j = (sOrientB >= 0) ? ((sOrientB + i) % sConeSize) : ((sConeSize -(sOrientB+1) - i) % sConeSize);
        if (childB) *childB = coneB[j];
        if (childOrientB) {
          PetscInt oBtrue;

          ierr          = DMPlexGetConeSize(dm,childA,&coneSize);CHKERRQ(ierr);
          /* compose sOrientB and oB[j] */
          if (coneSize != 0 && coneSize != 2) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Expected a vertex or an edge");
          /* we may have to flip an edge */
          oBtrue        = coneSize ? ((sOrientB >= 0) ? oB[j] : -(oB[j] + 2)) : 0;
          ABswap        = DihedralSwap(coneSize,oA[i],oBtrue);
          *childOrientB = DihedralCompose(coneSize,childOrientA,ABswap);
        }
        break;
      }
    }
    if (i == sConeSize) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"support cone mismatch");
    PetscFunctionReturn(0);
  }
  /* get the cone size and symmetry swap */
  ierr   = DMPlexGetConeSize(dm,parent,&coneSize);CHKERRQ(ierr);
  ABswap = DihedralSwap(coneSize, parentOrientA, parentOrientB);
  if (dim == 2) {
    /* orientations refer to cones: we want them to refer to vertices:
     * if it's a rotation, they are the same, but if the order is reversed, a
     * permutation that puts side i first does *not* put vertex i first */
    oAvert     = (parentOrientA >= 0) ? parentOrientA : -((-parentOrientA % coneSize) + 1);
    oBvert     = (parentOrientB >= 0) ? parentOrientB : -((-parentOrientB % coneSize) + 1);
    ABswapVert = DihedralSwap(coneSize, oAvert, oBvert);
  } else {
    ABswapVert = ABswap;
  }
  if (childB) {
    /* assume that each child corresponds to a vertex, in the same order */
    PetscInt p, posA = -1, numChildren, i;
    const PetscInt *children;

    /* count which position the child is in */
    ierr = DMPlexGetTreeChildren(dm,parent,&numChildren,&children);CHKERRQ(ierr);
    for (i = 0; i < numChildren; i++) {
      p = children[i];
      if (p == childA) {
        posA = i;
        break;
      }
    }
    if (posA >= coneSize) {
      /* this is the triangle in the middle of a uniformly refined triangle: it
       * is invariant */
      if (dim != 2 || posA != 3) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Expected a middle triangle, got something else");
      *childB = childA;
    }
    else {
      /* figure out position B by applying ABswapVert */
      PetscInt posB;

      posB = (ABswapVert >= 0) ? ((ABswapVert + posA) % coneSize) : ((coneSize -(ABswapVert + 1) - posA) % coneSize);
      if (childB) *childB = children[posB];
    }
  }
  if (childOrientB) *childOrientB = DihedralCompose(coneSize,childOrientA,ABswap);
  PetscFunctionReturn(0);
}

/*@
  DMPlexReferenceTreeGetChildSymmetry - Given a reference tree, transform a childid and orientation from one parent frame to another

  Input Parameters:
+ dm - the reference tree DMPlex object
. parent - the parent point
. parentOrientA - the reference orientation for describing the parent
. childOrientA - the reference orientation for describing the child
. childA - the reference childID for describing the child
- parentOrientB - the new orientation for describing the parent

  Output Parameters:
+ childOrientB - if not NULL, set to the new oreintation for describing the child
- childB - if not NULL, the new childID for describing the child

  Level: developer

.seealso: DMPlexGetReferenceTree(), DMPlexSetReferenceTree(), DMPlexSetTree()
@*/
PetscErrorCode DMPlexReferenceTreeGetChildSymmetry(DM dm, PetscInt parent, PetscInt parentOrientA, PetscInt childOrientA, PetscInt childA, PetscInt parentOrientB, PetscInt *childOrientB, PetscInt *childB)
{
  DM_Plex        *mesh = (DM_Plex *)dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (!mesh->getchildsymmetry) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"DMPlexReferenceTreeGetChildSymmetry not implemented");
  ierr = mesh->getchildsymmetry(dm,parent,parentOrientA,childOrientA,childA,parentOrientB,childOrientB,childB);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexSetTree_Internal(DM,PetscSection,PetscInt*,PetscInt*,PetscBool,PetscBool);

PetscErrorCode DMPlexCreateReferenceTree_SetTree(DM dm, PetscSection parentSection, PetscInt parents[], PetscInt childIDs[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexSetTree_Internal(dm,parentSection,parents,childIDs,PETSC_TRUE,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexCreateReferenceTree_Union(DM K, DM Kref, const char *labelName, DM *ref)
{
  MPI_Comm       comm;
  PetscInt       dim, p, pStart, pEnd, pRefStart, pRefEnd, d, offset, parentSize, *parents, *childIDs;
  PetscInt      *permvals, *unionCones, *coneSizes, *unionOrientations, numUnionPoints, *numDimPoints, numCones, numVerts;
  DMLabel        identity, identityRef;
  PetscSection   unionSection, unionConeSection, parentSection;
  PetscScalar   *unionCoords;
  IS             perm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  comm = PetscObjectComm((PetscObject)K);
  ierr = DMGetDimension(K, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetChart(K, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = DMGetLabel(K, labelName, &identity);CHKERRQ(ierr);
  ierr = DMGetLabel(Kref, labelName, &identityRef);CHKERRQ(ierr);
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
  ierr = PetscMalloc1(pEnd - pStart + pRefEnd - pRefStart,&permvals);CHKERRQ(ierr);
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
    PetscInt     vStart, vEnd, vRefStart, vRefEnd, v, vDof, vOff;
    PetscSection KcoordsSec, KrefCoordsSec;
    Vec          KcoordsVec, KrefCoordsVec;
    PetscScalar *Kcoords;

    ierr = DMGetCoordinateSection(K, &KcoordsSec);CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocal(K, &KcoordsVec);CHKERRQ(ierr);
    ierr = DMGetCoordinateSection(Kref, &KrefCoordsSec);CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocal(Kref, &KrefCoordsVec);CHKERRQ(ierr);

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
  ierr = DMSetDimension(*ref,dim);CHKERRQ(ierr);
  ierr = DMPlexCreateFromDAG(*ref,dim,numDimPoints,coneSizes,unionCones,unionOrientations,unionCoords);CHKERRQ(ierr);
  /* set the tree */
  ierr = PetscSectionCreate(comm,&parentSection);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(parentSection,0,numUnionPoints);CHKERRQ(ierr);
  for (p = pRefStart; p < pRefEnd; p++) {
    PetscInt uDof, uOff;

    ierr = PetscSectionGetDof(unionSection, p - pRefStart + (pEnd - pStart),&uDof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(unionSection, p - pRefStart + (pEnd - pStart),&uOff);CHKERRQ(ierr);
    if (uDof) {
      ierr = PetscSectionSetDof(parentSection,uOff,1);CHKERRQ(ierr);
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
      ierr = PetscSectionGetOffset(parentSection,uOff,&pOff);CHKERRQ(ierr);
      ierr = DMLabelGetValue(identityRef,p,&parent);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(unionSection, parent - pStart,&parentU);CHKERRQ(ierr);
      parents[pOff] = parentU;
      childIDs[pOff] = uOff;
    }
  }
  ierr = DMPlexCreateReferenceTree_SetTree(*ref,parentSection,parents,childIDs);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&parentSection);CHKERRQ(ierr);
  ierr = PetscFree2(parents,childIDs);CHKERRQ(ierr);

  /* clean up */
  ierr = PetscSectionDestroy(&unionSection);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&unionConeSection);CHKERRQ(ierr);
  ierr = ISDestroy(&perm);CHKERRQ(ierr);
  ierr = PetscFree(unionCoords);CHKERRQ(ierr);
  ierr = PetscFree2(unionCones,unionOrientations);CHKERRQ(ierr);
  ierr = PetscFree2(coneSizes,numDimPoints);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMPlexCreateDefaultReferenceTree - create a reference tree for isotropic hierarchical mesh refinement.

  Collective

  Input Parameters:
+ comm    - the MPI communicator
. dim     - the spatial dimension
- simplex - Flag for simplex, otherwise use a tensor-product cell

  Output Parameters:
. ref     - the reference tree DMPlex object

  Level: intermediate

.seealso: DMPlexSetReferenceTree(), DMPlexGetReferenceTree()
@*/
PetscErrorCode DMPlexCreateDefaultReferenceTree(MPI_Comm comm, PetscInt dim, PetscBool simplex, DM *ref)
{
  DM_Plex       *mesh;
  DM             K, Kref;
  PetscInt       p, pStart, pEnd;
  DMLabel        identity;
  PetscErrorCode ierr;

  PetscFunctionBegin;
#if 1
  comm = PETSC_COMM_SELF;
#endif
  /* create a reference element */
  ierr = DMPlexCreateReferenceCell(comm, dim, simplex, &K);CHKERRQ(ierr);
  ierr = DMCreateLabel(K, "identity");CHKERRQ(ierr);
  ierr = DMGetLabel(K, "identity", &identity);CHKERRQ(ierr);
  ierr = DMPlexGetChart(K, &pStart, &pEnd);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; p++) {
    ierr = DMLabelSetValue(identity, p, p);CHKERRQ(ierr);
  }
  /* refine it */
  ierr = DMRefine(K,comm,&Kref);CHKERRQ(ierr);

  /* the reference tree is the union of these two, without duplicating
   * points that appear in both */
  ierr = DMPlexCreateReferenceTree_Union(K, Kref, "identity", ref);CHKERRQ(ierr);
  mesh = (DM_Plex *) (*ref)->data;
  mesh->getchildsymmetry = DMPlexReferenceTreeGetChildSymmetry_Default;
  ierr = DMDestroy(&K);CHKERRQ(ierr);
  ierr = DMDestroy(&Kref);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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

static PetscErrorCode AnchorsFlatten (PetscSection section, IS is, PetscSection *sectionNew, IS *isNew)
{
  PetscInt       pStart, pEnd, size, sizeNew, i, p, *valsNew = NULL;
  const PetscInt *vals;
  PetscSection   secNew;
  PetscBool      anyNew, globalAnyNew;
  PetscBool      compress;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSectionGetChart(section,&pStart,&pEnd);CHKERRQ(ierr);
  ierr = ISGetLocalSize(is,&size);CHKERRQ(ierr);
  ierr = ISGetIndices(is,&vals);CHKERRQ(ierr);
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject)section),&secNew);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(secNew,pStart,pEnd);CHKERRQ(ierr);
  for (i = 0; i < size; i++) {
    PetscInt dof;

    p = vals[i];
    if (p < pStart || p >= pEnd) continue;
    ierr = PetscSectionGetDof(section, p, &dof);CHKERRQ(ierr);
    if (dof) break;
  }
  if (i == size) {
    ierr     = PetscSectionSetUp(secNew);CHKERRQ(ierr);
    anyNew   = PETSC_FALSE;
    compress = PETSC_FALSE;
    sizeNew  = 0;
  }
  else {
    anyNew = PETSC_TRUE;
    for (p = pStart; p < pEnd; p++) {
      PetscInt dof, off;

      ierr = PetscSectionGetDof(section, p, &dof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(section, p, &off);CHKERRQ(ierr);
      for (i = 0; i < dof; i++) {
        PetscInt q = vals[off + i], qDof = 0;

        if (q >= pStart && q < pEnd) {
          ierr = PetscSectionGetDof(section, q, &qDof);CHKERRQ(ierr);
        }
        if (qDof) {
          ierr = PetscSectionAddDof(secNew, p, qDof);CHKERRQ(ierr);
        }
        else {
          ierr = PetscSectionAddDof(secNew, p, 1);CHKERRQ(ierr);
        }
      }
    }
    ierr = PetscSectionSetUp(secNew);CHKERRQ(ierr);
    ierr = PetscSectionGetStorageSize(secNew,&sizeNew);CHKERRQ(ierr);
    ierr = PetscMalloc1(sizeNew,&valsNew);CHKERRQ(ierr);
    compress = PETSC_FALSE;
    for (p = pStart; p < pEnd; p++) {
      PetscInt dof, off, count, offNew, dofNew;

      ierr  = PetscSectionGetDof(section, p, &dof);CHKERRQ(ierr);
      ierr  = PetscSectionGetOffset(section, p, &off);CHKERRQ(ierr);
      ierr  = PetscSectionGetDof(secNew, p, &dofNew);CHKERRQ(ierr);
      ierr  = PetscSectionGetOffset(secNew, p, &offNew);CHKERRQ(ierr);
      count = 0;
      for (i = 0; i < dof; i++) {
        PetscInt q = vals[off + i], qDof = 0, qOff = 0, j;

        if (q >= pStart && q < pEnd) {
          ierr = PetscSectionGetDof(section, q, &qDof);CHKERRQ(ierr);
          ierr = PetscSectionGetOffset(section, q, &qOff);CHKERRQ(ierr);
        }
        if (qDof) {
          PetscInt oldCount = count;

          for (j = 0; j < qDof; j++) {
            PetscInt k, r = vals[qOff + j];

            for (k = 0; k < oldCount; k++) {
              if (valsNew[offNew + k] == r) {
                break;
              }
            }
            if (k == oldCount) {
              valsNew[offNew + count++] = r;
            }
          }
        }
        else {
          PetscInt k, oldCount = count;

          for (k = 0; k < oldCount; k++) {
            if (valsNew[offNew + k] == q) {
              break;
            }
          }
          if (k == oldCount) {
            valsNew[offNew + count++] = q;
          }
        }
      }
      if (count < dofNew) {
        ierr = PetscSectionSetDof(secNew, p, count);CHKERRQ(ierr);
        compress = PETSC_TRUE;
      }
    }
  }
  ierr = ISRestoreIndices(is,&vals);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(&anyNew,&globalAnyNew,1,MPIU_BOOL,MPI_LOR,PetscObjectComm((PetscObject)secNew));CHKERRQ(ierr);
  if (!globalAnyNew) {
    ierr = PetscSectionDestroy(&secNew);CHKERRQ(ierr);
    *sectionNew = NULL;
    *isNew = NULL;
  }
  else {
    PetscBool globalCompress;

    ierr = MPIU_Allreduce(&compress,&globalCompress,1,MPIU_BOOL,MPI_LOR,PetscObjectComm((PetscObject)secNew));CHKERRQ(ierr);
    if (compress) {
      PetscSection secComp;
      PetscInt *valsComp = NULL;

      ierr = PetscSectionCreate(PetscObjectComm((PetscObject)section),&secComp);CHKERRQ(ierr);
      ierr = PetscSectionSetChart(secComp,pStart,pEnd);CHKERRQ(ierr);
      for (p = pStart; p < pEnd; p++) {
        PetscInt dof;

        ierr = PetscSectionGetDof(secNew, p, &dof);CHKERRQ(ierr);
        ierr = PetscSectionSetDof(secComp, p, dof);CHKERRQ(ierr);
      }
      ierr = PetscSectionSetUp(secComp);CHKERRQ(ierr);
      ierr = PetscSectionGetStorageSize(secComp,&sizeNew);CHKERRQ(ierr);
      ierr = PetscMalloc1(sizeNew,&valsComp);CHKERRQ(ierr);
      for (p = pStart; p < pEnd; p++) {
        PetscInt dof, off, offNew, j;

        ierr = PetscSectionGetDof(secNew, p, &dof);CHKERRQ(ierr);
        ierr = PetscSectionGetOffset(secNew, p, &off);CHKERRQ(ierr);
        ierr = PetscSectionGetOffset(secComp, p, &offNew);CHKERRQ(ierr);
        for (j = 0; j < dof; j++) {
          valsComp[offNew + j] = valsNew[off + j];
        }
      }
      ierr    = PetscSectionDestroy(&secNew);CHKERRQ(ierr);
      secNew  = secComp;
      ierr    = PetscFree(valsNew);CHKERRQ(ierr);
      valsNew = valsComp;
    }
    ierr = ISCreateGeneral(PetscObjectComm((PetscObject)is),sizeNew,valsNew,PETSC_OWN_POINTER,isNew);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexCreateAnchors_Tree(DM dm)
{
  PetscInt       p, pStart, pEnd, *anchors, size;
  PetscInt       aMin = PETSC_MAX_INT, aMax = PETSC_MIN_INT;
  PetscSection   aSec;
  DMLabel        canonLabel;
  IS             aIS;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMPlexGetChart(dm,&pStart,&pEnd);CHKERRQ(ierr);
  ierr = DMGetLabel(dm,"canonical",&canonLabel);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; p++) {
    PetscInt parent;

    if (canonLabel) {
      PetscInt canon;

      ierr = DMLabelGetValue(canonLabel,p,&canon);CHKERRQ(ierr);
      if (p != canon) continue;
    }
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
  ierr = PetscSectionCreate(PETSC_COMM_SELF,&aSec);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(aSec,aMin,aMax);CHKERRQ(ierr);
  for (p = aMin; p < aMax; p++) {
    PetscInt parent, ancestor = p;

    if (canonLabel) {
      PetscInt canon;

      ierr = DMLabelGetValue(canonLabel,p,&canon);CHKERRQ(ierr);
      if (p != canon) continue;
    }
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

    if (canonLabel) {
      PetscInt canon;

      ierr = DMLabelGetValue(canonLabel,p,&canon);CHKERRQ(ierr);
      if (p != canon) continue;
    }
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
  ierr = ISCreateGeneral(PETSC_COMM_SELF,size,anchors,PETSC_OWN_POINTER,&aIS);CHKERRQ(ierr);
  {
    PetscSection aSecNew = aSec;
    IS           aISNew  = aIS;

    ierr = PetscObjectReference((PetscObject)aSec);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)aIS);CHKERRQ(ierr);
    while (aSecNew) {
      ierr    = PetscSectionDestroy(&aSec);CHKERRQ(ierr);
      ierr    = ISDestroy(&aIS);CHKERRQ(ierr);
      aSec    = aSecNew;
      aIS     = aISNew;
      aSecNew = NULL;
      aISNew  = NULL;
      ierr    = AnchorsFlatten(aSec,aIS,&aSecNew,&aISNew);CHKERRQ(ierr);
    }
  }
  ierr = DMPlexSetAnchors(dm,aSec,aIS);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&aSec);CHKERRQ(ierr);
  ierr = ISDestroy(&aIS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexGetTrueSupportSize(DM dm,PetscInt p,PetscInt *dof,PetscInt *numTrueSupp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (numTrueSupp[p] == -1) {
    PetscInt i, alldof;
    const PetscInt *supp;
    PetscInt count = 0;

    ierr = DMPlexGetSupportSize(dm,p,&alldof);CHKERRQ(ierr);
    ierr = DMPlexGetSupport(dm,p,&supp);CHKERRQ(ierr);
    for (i = 0; i < alldof; i++) {
      PetscInt q = supp[i], numCones, j;
      const PetscInt *cone;

      ierr = DMPlexGetConeSize(dm,q,&numCones);CHKERRQ(ierr);
      ierr = DMPlexGetCone(dm,q,&cone);CHKERRQ(ierr);
      for (j = 0; j < numCones; j++) {
        if (cone[j] == p) break;
      }
      if (j < numCones) count++;
    }
    numTrueSupp[p] = count;
  }
  *dof = numTrueSupp[p];
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTreeExchangeSupports(DM dm)
{
  DM_Plex        *mesh = (DM_Plex *)dm->data;
  PetscSection   newSupportSection;
  PetscInt       newSize, *newSupports, pStart, pEnd, p, d, depth;
  PetscInt       *numTrueSupp;
  PetscInt       *offsets;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  /* symmetrize the hierarchy */
  ierr = DMPlexGetDepth(dm,&depth);CHKERRQ(ierr);
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject)(mesh->supportSection)),&newSupportSection);CHKERRQ(ierr);
  ierr = DMPlexGetChart(dm,&pStart,&pEnd);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(newSupportSection,pStart,pEnd);CHKERRQ(ierr);
  ierr = PetscCalloc1(pEnd,&offsets);CHKERRQ(ierr);
  ierr = PetscMalloc1(pEnd,&numTrueSupp);CHKERRQ(ierr);
  for (p = 0; p < pEnd; p++) numTrueSupp[p] = -1;
  /* if a point is in the (true) support of q, it should be in the support of
   * parent(q) */
  for (d = 0; d <= depth; d++) {
    ierr = DMPlexGetHeightStratum(dm,d,&pStart,&pEnd);CHKERRQ(ierr);
    for (p = pStart; p < pEnd; ++p) {
      PetscInt dof, q, qdof, parent;

      ierr = DMPlexGetTrueSupportSize(dm,p,&dof,numTrueSupp);CHKERRQ(ierr);
      ierr = PetscSectionAddDof(newSupportSection, p, dof);CHKERRQ(ierr);
      q    = p;
      ierr = DMPlexGetTreeParent(dm,q,&parent,NULL);CHKERRQ(ierr);
      while (parent != q && parent >= pStart && parent < pEnd) {
        q = parent;

        ierr = DMPlexGetTrueSupportSize(dm,q,&qdof,numTrueSupp);CHKERRQ(ierr);
        ierr = PetscSectionAddDof(newSupportSection,p,qdof);CHKERRQ(ierr);
        ierr = PetscSectionAddDof(newSupportSection,q,dof);CHKERRQ(ierr);
        ierr = DMPlexGetTreeParent(dm,q,&parent,NULL);CHKERRQ(ierr);
      }
    }
  }
  ierr = PetscSectionSetUp(newSupportSection);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(newSupportSection,&newSize);CHKERRQ(ierr);
  ierr = PetscMalloc1(newSize,&newSupports);CHKERRQ(ierr);
  for (d = 0; d <= depth; d++) {
    ierr = DMPlexGetHeightStratum(dm,d,&pStart,&pEnd);CHKERRQ(ierr);
    for (p = pStart; p < pEnd; p++) {
      PetscInt dof, off, q, qdof, qoff, newDof, newOff, newqOff, i, parent;

      ierr = PetscSectionGetDof(mesh->supportSection, p, &dof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(mesh->supportSection, p, &off);CHKERRQ(ierr);
      ierr = PetscSectionGetDof(newSupportSection, p, &newDof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(newSupportSection, p, &newOff);CHKERRQ(ierr);
      for (i = 0; i < dof; i++) {
        PetscInt numCones, j;
        const PetscInt *cone;
        PetscInt q = mesh->supports[off + i];

        ierr = DMPlexGetConeSize(dm,q,&numCones);CHKERRQ(ierr);
        ierr = DMPlexGetCone(dm,q,&cone);CHKERRQ(ierr);
        for (j = 0; j < numCones; j++) {
          if (cone[j] == p) break;
        }
        if (j < numCones) newSupports[newOff+offsets[p]++] = q;
      }
      mesh->maxSupportSize = PetscMax(mesh->maxSupportSize,newDof);

      q    = p;
      ierr = DMPlexGetTreeParent(dm,q,&parent,NULL);CHKERRQ(ierr);
      while (parent != q && parent >= pStart && parent < pEnd) {
        q = parent;
        ierr = PetscSectionGetDof(mesh->supportSection, q, &qdof);CHKERRQ(ierr);
        ierr = PetscSectionGetOffset(mesh->supportSection, q, &qoff);CHKERRQ(ierr);
        ierr = PetscSectionGetOffset(newSupportSection, q, &newqOff);CHKERRQ(ierr);
        for (i = 0; i < qdof; i++) {
          PetscInt numCones, j;
          const PetscInt *cone;
          PetscInt r = mesh->supports[qoff + i];

          ierr = DMPlexGetConeSize(dm,r,&numCones);CHKERRQ(ierr);
          ierr = DMPlexGetCone(dm,r,&cone);CHKERRQ(ierr);
          for (j = 0; j < numCones; j++) {
            if (cone[j] == q) break;
          }
          if (j < numCones) newSupports[newOff+offsets[p]++] = r;
        }
        for (i = 0; i < dof; i++) {
          PetscInt numCones, j;
          const PetscInt *cone;
          PetscInt r = mesh->supports[off + i];

          ierr = DMPlexGetConeSize(dm,r,&numCones);CHKERRQ(ierr);
          ierr = DMPlexGetCone(dm,r,&cone);CHKERRQ(ierr);
          for (j = 0; j < numCones; j++) {
            if (cone[j] == p) break;
          }
          if (j < numCones) newSupports[newqOff+offsets[q]++] = r;
        }
        ierr = DMPlexGetTreeParent(dm,q,&parent,NULL);CHKERRQ(ierr);
      }
    }
  }
  ierr = PetscSectionDestroy(&mesh->supportSection);CHKERRQ(ierr);
  mesh->supportSection = newSupportSection;
  ierr = PetscFree(mesh->supports);CHKERRQ(ierr);
  mesh->supports = newSupports;
  ierr = PetscFree(offsets);CHKERRQ(ierr);
  ierr = PetscFree(numTrueSupp);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexComputeAnchorMatrix_Tree_Direct(DM,PetscSection,PetscSection,Mat);
static PetscErrorCode DMPlexComputeAnchorMatrix_Tree_FromReference(DM,PetscSection,PetscSection,Mat);

static PetscErrorCode DMPlexSetTree_Internal(DM dm, PetscSection parentSection, PetscInt *parents, PetscInt *childIDs, PetscBool computeCanonical, PetscBool exchangeSupports)
{
  DM_Plex       *mesh = (DM_Plex *)dm->data;
  DM             refTree;
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
    ierr = PetscArraycpy(mesh->parents, parents, size);CHKERRQ(ierr);
  }
  if (childIDs != mesh->childIDs) {
    ierr = PetscFree(mesh->childIDs);CHKERRQ(ierr);
    ierr = PetscMalloc1(size,&mesh->childIDs);CHKERRQ(ierr);
    ierr = PetscArraycpy(mesh->childIDs, childIDs, size);CHKERRQ(ierr);
  }
  ierr = DMPlexGetReferenceTree(dm,&refTree);CHKERRQ(ierr);
  if (refTree) {
    DMLabel canonLabel;

    ierr = DMGetLabel(refTree,"canonical",&canonLabel);CHKERRQ(ierr);
    if (canonLabel) {
      PetscInt i;

      for (i = 0; i < size; i++) {
        PetscInt canon;
        ierr = DMLabelGetValue(canonLabel, mesh->childIDs[i], &canon);CHKERRQ(ierr);
        if (canon >= 0) {
          mesh->childIDs[i] = canon;
        }
      }
    }
    mesh->computeanchormatrix = DMPlexComputeAnchorMatrix_Tree_FromReference;
  } else {
    mesh->computeanchormatrix = DMPlexComputeAnchorMatrix_Tree_Direct;
  }
  ierr = DMPlexTreeSymmetrize(dm);CHKERRQ(ierr);
  if (computeCanonical) {
    PetscInt d, dim;

    /* add the canonical label */
    ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
    ierr = DMCreateLabel(dm,"canonical");CHKERRQ(ierr);
    for (d = 0; d <= dim; d++) {
      PetscInt p, dStart, dEnd, canon = -1, cNumChildren;
      const PetscInt *cChildren;

      ierr = DMPlexGetDepthStratum(dm,d,&dStart,&dEnd);CHKERRQ(ierr);
      for (p = dStart; p < dEnd; p++) {
        ierr = DMPlexGetTreeChildren(dm,p,&cNumChildren,&cChildren);CHKERRQ(ierr);
        if (cNumChildren) {
          canon = p;
          break;
        }
      }
      if (canon == -1) continue;
      for (p = dStart; p < dEnd; p++) {
        PetscInt numChildren, i;
        const PetscInt *children;

        ierr = DMPlexGetTreeChildren(dm,p,&numChildren,&children);CHKERRQ(ierr);
        if (numChildren) {
          if (numChildren != cNumChildren) SETERRQ2(PetscObjectComm((PetscObject)dm),PETSC_ERR_PLIB,"All parent points in a stratum should have the same number of children: %d != %d", numChildren, cNumChildren);
          ierr = DMSetLabelValue(dm,"canonical",p,canon);CHKERRQ(ierr);
          for (i = 0; i < numChildren; i++) {
            ierr = DMSetLabelValue(dm,"canonical",children[i],cChildren[i]);CHKERRQ(ierr);
          }
        }
      }
    }
  }
  if (exchangeSupports) {
    ierr = DMPlexTreeExchangeSupports(dm);CHKERRQ(ierr);
  }
  mesh->createanchors = DMPlexCreateAnchors_Tree;
  /* reset anchors */
  ierr = DMPlexSetAnchors(dm,NULL,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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

.seealso: DMPlexGetTree(), DMPlexSetReferenceTree(), DMPlexSetAnchors(), DMPlexGetTreeParent(), DMPlexGetTreeChildren()
@*/
PetscErrorCode DMPlexSetTree(DM dm, PetscSection parentSection, PetscInt parents[], PetscInt childIDs[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexSetTree_Internal(dm,parentSection,parents,childIDs,PETSC_FALSE,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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

.seealso: DMPlexSetTree(), DMPlexSetReferenceTree(), DMPlexSetAnchors(), DMPlexGetTreeParent(), DMPlexGetTreeChildren()
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

/*@
  DMPlexGetTreeParent - get the parent of a point in the tree describing the point hierarchy (not the DAG)

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

/*@C
  DMPlexGetTreeChildren - get the children of a point in the tree describing the point hierarchy (not the DAG)

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

static PetscErrorCode EvaluateBasis(PetscSpace space, PetscInt nBasis, PetscInt nFunctionals, PetscInt nComps, PetscInt nPoints, const PetscInt *pointsPerFn, const PetscReal *points, const PetscReal *weights, PetscReal *work, Mat basisAtPoints)
{
  PetscInt       f, b, p, c, offset, qPoints;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSpaceEvaluate(space,nPoints,points,work,NULL,NULL);CHKERRQ(ierr);
  for (f = 0, offset = 0; f < nFunctionals; f++) {
    qPoints = pointsPerFn[f];
    for (b = 0; b < nBasis; b++) {
      PetscScalar val = 0.;

      for (p = 0; p < qPoints; p++) {
        for (c = 0; c < nComps; c++) {
          val += work[((offset + p) * nBasis + b) * nComps + c] * weights[(offset + p) * nComps + c];
        }
      }
      ierr = MatSetValue(basisAtPoints,b,f,val,INSERT_VALUES);CHKERRQ(ierr);
    }
    offset += qPoints;
  }
  ierr = MatAssemblyBegin(basisAtPoints,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(basisAtPoints,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexComputeAnchorMatrix_Tree_Direct(DM dm, PetscSection section, PetscSection cSec, Mat cMat)
{
  PetscDS        ds;
  PetscInt       spdim;
  PetscInt       numFields, f, c, cStart, cEnd, pStart, pEnd, conStart, conEnd;
  const PetscInt *anchors;
  PetscSection   aSec;
  PetscReal      *v0, *v0parent, *vtmp, *J, *Jparent, *invJparent, detJ, detJparent;
  IS             aIS;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetChart(dm,&pStart,&pEnd);CHKERRQ(ierr);
  ierr = DMGetDS(dm,&ds);CHKERRQ(ierr);
  ierr = PetscDSGetNumFields(ds,&numFields);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetAnchors(dm,&aSec,&aIS);CHKERRQ(ierr);
  ierr = ISGetIndices(aIS,&anchors);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(cSec,&conStart,&conEnd);CHKERRQ(ierr);
  ierr = DMGetDimension(dm,&spdim);CHKERRQ(ierr);
  ierr = PetscMalloc6(spdim,&v0,spdim,&v0parent,spdim,&vtmp,spdim*spdim,&J,spdim*spdim,&Jparent,spdim*spdim,&invJparent);CHKERRQ(ierr);

  for (f = 0; f < numFields; f++) {
    PetscObject       disc;
    PetscClassId      id;
    PetscSpace        bspace;
    PetscDualSpace    dspace;
    PetscInt          i, j, k, nPoints, Nc, offset;
    PetscInt          fSize, maxDof;
    PetscReal         *weights, *pointsRef, *pointsReal, *work;
    PetscScalar       *scwork;
    const PetscScalar *X;
    PetscInt          *sizes, *workIndRow, *workIndCol;
    Mat               Amat, Bmat, Xmat;
    const PetscInt    *numDof  = NULL;
    const PetscInt    ***perms = NULL;
    const PetscScalar ***flips = NULL;

    ierr = PetscDSGetDiscretization(ds,f,&disc);CHKERRQ(ierr);
    ierr = PetscObjectGetClassId(disc,&id);CHKERRQ(ierr);
    if (id == PETSCFE_CLASSID) {
      PetscFE fe = (PetscFE) disc;

      ierr = PetscFEGetBasisSpace(fe,&bspace);CHKERRQ(ierr);
      ierr = PetscFEGetDualSpace(fe,&dspace);CHKERRQ(ierr);
      ierr = PetscDualSpaceGetDimension(dspace,&fSize);CHKERRQ(ierr);
      ierr = PetscFEGetNumComponents(fe,&Nc);CHKERRQ(ierr);
    }
    else if (id == PETSCFV_CLASSID) {
      PetscFV fv = (PetscFV) disc;

      ierr = PetscFVGetNumComponents(fv,&Nc);CHKERRQ(ierr);
      ierr = PetscSpaceCreate(PetscObjectComm((PetscObject)fv),&bspace);CHKERRQ(ierr);
      ierr = PetscSpaceSetType(bspace,PETSCSPACEPOLYNOMIAL);CHKERRQ(ierr);
      ierr = PetscSpaceSetDegree(bspace,0,PETSC_DETERMINE);CHKERRQ(ierr);
      ierr = PetscSpaceSetNumComponents(bspace,Nc);CHKERRQ(ierr);
      ierr = PetscSpaceSetNumVariables(bspace,spdim);CHKERRQ(ierr);
      ierr = PetscSpaceSetUp(bspace);CHKERRQ(ierr);
      ierr = PetscFVGetDualSpace(fv,&dspace);CHKERRQ(ierr);
      ierr = PetscDualSpaceGetDimension(dspace,&fSize);CHKERRQ(ierr);
    }
    else SETERRQ1(PetscObjectComm(disc),PETSC_ERR_ARG_UNKNOWN_TYPE, "PetscDS discretization id %d not recognized.", id);
    ierr = PetscDualSpaceGetNumDof(dspace,&numDof);CHKERRQ(ierr);
    for (i = 0, maxDof = 0; i <= spdim; i++) {maxDof = PetscMax(maxDof,numDof[i]);}
    ierr = PetscDualSpaceGetSymmetries(dspace,&perms,&flips);CHKERRQ(ierr);

    ierr = MatCreate(PETSC_COMM_SELF,&Amat);CHKERRQ(ierr);
    ierr = MatSetSizes(Amat,fSize,fSize,fSize,fSize);CHKERRQ(ierr);
    ierr = MatSetType(Amat,MATSEQDENSE);CHKERRQ(ierr);
    ierr = MatSetUp(Amat);CHKERRQ(ierr);
    ierr = MatDuplicate(Amat,MAT_DO_NOT_COPY_VALUES,&Bmat);CHKERRQ(ierr);
    ierr = MatDuplicate(Amat,MAT_DO_NOT_COPY_VALUES,&Xmat);CHKERRQ(ierr);
    nPoints = 0;
    for (i = 0; i < fSize; i++) {
      PetscInt        qPoints, thisNc;
      PetscQuadrature quad;

      ierr = PetscDualSpaceGetFunctional(dspace,i,&quad);CHKERRQ(ierr);
      ierr = PetscQuadratureGetData(quad,NULL,&thisNc,&qPoints,NULL,NULL);CHKERRQ(ierr);
      if (thisNc != Nc) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Functional dim %D does not much basis dim %D\n",thisNc,Nc);
      nPoints += qPoints;
    }
    ierr = PetscMalloc7(fSize,&sizes,nPoints*Nc,&weights,spdim*nPoints,&pointsRef,spdim*nPoints,&pointsReal,nPoints*fSize*Nc,&work,maxDof,&workIndRow,maxDof,&workIndCol);CHKERRQ(ierr);
    ierr = PetscMalloc1(maxDof * maxDof,&scwork);CHKERRQ(ierr);
    offset = 0;
    for (i = 0; i < fSize; i++) {
      PetscInt        qPoints;
      const PetscReal    *p, *w;
      PetscQuadrature quad;

      ierr = PetscDualSpaceGetFunctional(dspace,i,&quad);CHKERRQ(ierr);
      ierr = PetscQuadratureGetData(quad,NULL,NULL,&qPoints,&p,&w);CHKERRQ(ierr);
      ierr = PetscArraycpy(weights+Nc*offset,w,Nc*qPoints);CHKERRQ(ierr);
      ierr = PetscArraycpy(pointsRef+spdim*offset,p,spdim*qPoints);CHKERRQ(ierr);
      sizes[i] = qPoints;
      offset  += qPoints;
    }
    ierr = EvaluateBasis(bspace,fSize,fSize,Nc,nPoints,sizes,pointsRef,weights,work,Amat);CHKERRQ(ierr);
    ierr = MatLUFactor(Amat,NULL,NULL,NULL);CHKERRQ(ierr);
    for (c = cStart; c < cEnd; c++) {
      PetscInt        parent;
      PetscInt        closureSize, closureSizeP, *closure = NULL, *closureP = NULL;
      PetscInt        *childOffsets, *parentOffsets;

      ierr = DMPlexGetTreeParent(dm,c,&parent,NULL);CHKERRQ(ierr);
      if (parent == c) continue;
      ierr = DMPlexGetTransitiveClosure(dm,c,PETSC_TRUE,&closureSize,&closure);CHKERRQ(ierr);
      for (i = 0; i < closureSize; i++) {
        PetscInt p = closure[2*i];
        PetscInt conDof;

        if (p < conStart || p >= conEnd) continue;
        if (numFields) {
          ierr = PetscSectionGetFieldDof(cSec,p,f,&conDof);CHKERRQ(ierr);
        }
        else {
          ierr = PetscSectionGetDof(cSec,p,&conDof);CHKERRQ(ierr);
        }
        if (conDof) break;
      }
      if (i == closureSize) {
        ierr = DMPlexRestoreTransitiveClosure(dm,c,PETSC_TRUE,&closureSize,&closure);CHKERRQ(ierr);
        continue;
      }

      ierr = DMPlexComputeCellGeometryFEM(dm, c, NULL, v0, J, NULL, &detJ);CHKERRQ(ierr);
      ierr = DMPlexComputeCellGeometryFEM(dm, parent, NULL, v0parent, Jparent, invJparent, &detJparent);CHKERRQ(ierr);
      for (i = 0; i < nPoints; i++) {
        const PetscReal xi0[3] = {-1.,-1.,-1.};

        CoordinatesRefToReal(spdim, spdim, xi0, v0, J, &pointsRef[i*spdim],vtmp);
        CoordinatesRealToRef(spdim, spdim, xi0, v0parent, invJparent, vtmp, &pointsReal[i*spdim]);
      }
      ierr = EvaluateBasis(bspace,fSize,fSize,Nc,nPoints,sizes,pointsReal,weights,work,Bmat);CHKERRQ(ierr);
      ierr = MatMatSolve(Amat,Bmat,Xmat);CHKERRQ(ierr);
      ierr = MatDenseGetArrayRead(Xmat,&X);CHKERRQ(ierr);
      ierr = DMPlexGetTransitiveClosure(dm,parent,PETSC_TRUE,&closureSizeP,&closureP);CHKERRQ(ierr);
      ierr = PetscMalloc2(closureSize+1,&childOffsets,closureSizeP+1,&parentOffsets);CHKERRQ(ierr);
      childOffsets[0] = 0;
      for (i = 0; i < closureSize; i++) {
        PetscInt p = closure[2*i];
        PetscInt dof;

        if (numFields) {
          ierr = PetscSectionGetFieldDof(section,p,f,&dof);CHKERRQ(ierr);
        }
        else {
          ierr = PetscSectionGetDof(section,p,&dof);CHKERRQ(ierr);
        }
        childOffsets[i+1]=childOffsets[i]+dof;
      }
      parentOffsets[0] = 0;
      for (i = 0; i < closureSizeP; i++) {
        PetscInt p = closureP[2*i];
        PetscInt dof;

        if (numFields) {
          ierr = PetscSectionGetFieldDof(section,p,f,&dof);CHKERRQ(ierr);
        }
        else {
          ierr = PetscSectionGetDof(section,p,&dof);CHKERRQ(ierr);
        }
        parentOffsets[i+1]=parentOffsets[i]+dof;
      }
      for (i = 0; i < closureSize; i++) {
        PetscInt conDof, conOff, aDof, aOff, nWork;
        PetscInt p = closure[2*i];
        PetscInt o = closure[2*i+1];
        const PetscInt    *perm;
        const PetscScalar *flip;

        if (p < conStart || p >= conEnd) continue;
        if (numFields) {
          ierr = PetscSectionGetFieldDof(cSec,p,f,&conDof);CHKERRQ(ierr);
          ierr = PetscSectionGetFieldOffset(cSec,p,f,&conOff);CHKERRQ(ierr);
        }
        else {
          ierr = PetscSectionGetDof(cSec,p,&conDof);CHKERRQ(ierr);
          ierr = PetscSectionGetOffset(cSec,p,&conOff);CHKERRQ(ierr);
        }
        if (!conDof) continue;
        perm  = (perms && perms[i]) ? perms[i][o] : NULL;
        flip  = (flips && flips[i]) ? flips[i][o] : NULL;
        ierr  = PetscSectionGetDof(aSec,p,&aDof);CHKERRQ(ierr);
        ierr  = PetscSectionGetOffset(aSec,p,&aOff);CHKERRQ(ierr);
        nWork = childOffsets[i+1]-childOffsets[i];
        for (k = 0; k < aDof; k++) {
          PetscInt a = anchors[aOff + k];
          PetscInt aSecDof, aSecOff;

          if (numFields) {
            ierr = PetscSectionGetFieldDof(section,a,f,&aSecDof);CHKERRQ(ierr);
            ierr = PetscSectionGetFieldOffset(section,a,f,&aSecOff);CHKERRQ(ierr);
          }
          else {
            ierr = PetscSectionGetDof(section,a,&aSecDof);CHKERRQ(ierr);
            ierr = PetscSectionGetOffset(section,a,&aSecOff);CHKERRQ(ierr);
          }
          if (!aSecDof) continue;

          for (j = 0; j < closureSizeP; j++) {
            PetscInt q = closureP[2*j];
            PetscInt oq = closureP[2*j+1];

            if (q == a) {
              PetscInt           r, s, nWorkP;
              const PetscInt    *permP;
              const PetscScalar *flipP;

              permP  = (perms && perms[j]) ? perms[j][oq] : NULL;
              flipP  = (flips && flips[j]) ? flips[j][oq] : NULL;
              nWorkP = parentOffsets[j+1]-parentOffsets[j];
              /* get a copy of the child-to-anchor portion of the matrix, and transpose so that rows correspond to the
               * child and columns correspond to the anchor: BUT the maxrix returned by MatDenseGetArrayRead() is
               * column-major, so transpose-transpose = do nothing */
              for (r = 0; r < nWork; r++) {
                for (s = 0; s < nWorkP; s++) {
                  scwork[r * nWorkP + s] = X[fSize * (r + childOffsets[i]) + (s + parentOffsets[j])];
                }
              }
              for (r = 0; r < nWork; r++)  {workIndRow[perm  ? perm[r]  : r] = conOff  + r;}
              for (s = 0; s < nWorkP; s++) {workIndCol[permP ? permP[s] : s] = aSecOff + s;}
              if (flip) {
                for (r = 0; r < nWork; r++) {
                  for (s = 0; s < nWorkP; s++) {
                    scwork[r * nWorkP + s] *= flip[r];
                  }
                }
              }
              if (flipP) {
                for (r = 0; r < nWork; r++) {
                  for (s = 0; s < nWorkP; s++) {
                    scwork[r * nWorkP + s] *= flipP[s];
                  }
                }
              }
              ierr = MatSetValues(cMat,nWork,workIndRow,nWorkP,workIndCol,scwork,INSERT_VALUES);CHKERRQ(ierr);
              break;
            }
          }
        }
      }
      ierr = MatDenseRestoreArrayRead(Xmat,&X);CHKERRQ(ierr);
      ierr = PetscFree2(childOffsets,parentOffsets);CHKERRQ(ierr);
      ierr = DMPlexRestoreTransitiveClosure(dm,c,PETSC_TRUE,&closureSize,&closure);CHKERRQ(ierr);
      ierr = DMPlexRestoreTransitiveClosure(dm,parent,PETSC_TRUE,&closureSizeP,&closureP);CHKERRQ(ierr);
    }
    ierr = MatDestroy(&Amat);CHKERRQ(ierr);
    ierr = MatDestroy(&Bmat);CHKERRQ(ierr);
    ierr = MatDestroy(&Xmat);CHKERRQ(ierr);
    ierr = PetscFree(scwork);CHKERRQ(ierr);
    ierr = PetscFree7(sizes,weights,pointsRef,pointsReal,work,workIndRow,workIndCol);CHKERRQ(ierr);
    if (id == PETSCFV_CLASSID) {
      ierr = PetscSpaceDestroy(&bspace);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(cMat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(cMat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscFree6(v0,v0parent,vtmp,J,Jparent,invJparent);CHKERRQ(ierr);
  ierr = ISRestoreIndices(aIS,&anchors);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexReferenceTreeGetChildrenMatrices(DM refTree, PetscScalar ****childrenMats, PetscInt ***childrenN)
{
  Mat               refCmat;
  PetscDS           ds;
  PetscInt          numFields, maxFields, f, pRefStart, pRefEnd, p, *rows, *cols, maxDof, maxAnDof, **refPointFieldN;
  PetscScalar       ***refPointFieldMats;
  PetscSection      refConSec, refAnSec, refSection;
  IS                refAnIS;
  const PetscInt    *refAnchors;
  const PetscInt    **perms;
  const PetscScalar **flips;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = DMGetDS(refTree,&ds);CHKERRQ(ierr);
  ierr = PetscDSGetNumFields(ds,&numFields);CHKERRQ(ierr);
  maxFields = PetscMax(1,numFields);
  ierr = DMGetDefaultConstraints(refTree,&refConSec,&refCmat);CHKERRQ(ierr);
  ierr = DMPlexGetAnchors(refTree,&refAnSec,&refAnIS);CHKERRQ(ierr);
  ierr = ISGetIndices(refAnIS,&refAnchors);CHKERRQ(ierr);
  ierr = DMGetLocalSection(refTree,&refSection);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(refConSec,&pRefStart,&pRefEnd);CHKERRQ(ierr);
  ierr = PetscMalloc1(pRefEnd-pRefStart,&refPointFieldMats);CHKERRQ(ierr);
  ierr = PetscMalloc1(pRefEnd-pRefStart,&refPointFieldN);CHKERRQ(ierr);
  ierr = PetscSectionGetMaxDof(refConSec,&maxDof);CHKERRQ(ierr);
  ierr = PetscSectionGetMaxDof(refAnSec,&maxAnDof);CHKERRQ(ierr);
  ierr = PetscMalloc1(maxDof,&rows);CHKERRQ(ierr);
  ierr = PetscMalloc1(maxDof*maxAnDof,&cols);CHKERRQ(ierr);
  for (p = pRefStart; p < pRefEnd; p++) {
    PetscInt parent, closureSize, *closure = NULL, pDof;

    ierr = DMPlexGetTreeParent(refTree,p,&parent,NULL);CHKERRQ(ierr);
    ierr = PetscSectionGetDof(refConSec,p,&pDof);CHKERRQ(ierr);
    if (!pDof || parent == p) continue;

    ierr = PetscMalloc1(maxFields,&refPointFieldMats[p-pRefStart]);CHKERRQ(ierr);
    ierr = PetscCalloc1(maxFields,&refPointFieldN[p-pRefStart]);CHKERRQ(ierr);
    ierr = DMPlexGetTransitiveClosure(refTree,parent,PETSC_TRUE,&closureSize,&closure);CHKERRQ(ierr);
    for (f = 0; f < maxFields; f++) {
      PetscInt cDof, cOff, numCols, r, i;

      if (f < numFields) {
        ierr = PetscSectionGetFieldDof(refConSec,p,f,&cDof);CHKERRQ(ierr);
        ierr = PetscSectionGetFieldOffset(refConSec,p,f,&cOff);CHKERRQ(ierr);
        ierr = PetscSectionGetFieldPointSyms(refSection,f,closureSize,closure,&perms,&flips);CHKERRQ(ierr);
      } else {
        ierr = PetscSectionGetDof(refConSec,p,&cDof);CHKERRQ(ierr);
        ierr = PetscSectionGetOffset(refConSec,p,&cOff);CHKERRQ(ierr);
        ierr = PetscSectionGetPointSyms(refSection,closureSize,closure,&perms,&flips);CHKERRQ(ierr);
      }

      for (r = 0; r < cDof; r++) {
        rows[r] = cOff + r;
      }
      numCols = 0;
      for (i = 0; i < closureSize; i++) {
        PetscInt          q = closure[2*i];
        PetscInt          aDof, aOff, j;
        const PetscInt    *perm = perms ? perms[i] : NULL;

        if (numFields) {
          ierr = PetscSectionGetFieldDof(refSection,q,f,&aDof);CHKERRQ(ierr);
          ierr = PetscSectionGetFieldOffset(refSection,q,f,&aOff);CHKERRQ(ierr);
        }
        else {
          ierr = PetscSectionGetDof(refSection,q,&aDof);CHKERRQ(ierr);
          ierr = PetscSectionGetOffset(refSection,q,&aOff);CHKERRQ(ierr);
        }

        for (j = 0; j < aDof; j++) {
          cols[numCols++] = aOff + (perm ? perm[j] : j);
        }
      }
      refPointFieldN[p-pRefStart][f] = numCols;
      ierr = PetscMalloc1(cDof*numCols,&refPointFieldMats[p-pRefStart][f]);CHKERRQ(ierr);
      ierr = MatGetValues(refCmat,cDof,rows,numCols,cols,refPointFieldMats[p-pRefStart][f]);CHKERRQ(ierr);
      if (flips) {
        PetscInt colOff = 0;

        for (i = 0; i < closureSize; i++) {
          PetscInt          q = closure[2*i];
          PetscInt          aDof, aOff, j;
          const PetscScalar *flip = flips ? flips[i] : NULL;

          if (numFields) {
            ierr = PetscSectionGetFieldDof(refSection,q,f,&aDof);CHKERRQ(ierr);
            ierr = PetscSectionGetFieldOffset(refSection,q,f,&aOff);CHKERRQ(ierr);
          }
          else {
            ierr = PetscSectionGetDof(refSection,q,&aDof);CHKERRQ(ierr);
            ierr = PetscSectionGetOffset(refSection,q,&aOff);CHKERRQ(ierr);
          }
          if (flip) {
            PetscInt k;
            for (k = 0; k < cDof; k++) {
              for (j = 0; j < aDof; j++) {
                refPointFieldMats[p-pRefStart][f][k * numCols + colOff + j] *= flip[j];
              }
            }
          }
          colOff += aDof;
        }
      }
      if (numFields) {
        ierr = PetscSectionRestoreFieldPointSyms(refSection,f,closureSize,closure,&perms,&flips);CHKERRQ(ierr);
      } else {
        ierr = PetscSectionRestorePointSyms(refSection,closureSize,closure,&perms,&flips);CHKERRQ(ierr);
      }
    }
    ierr = DMPlexRestoreTransitiveClosure(refTree,parent,PETSC_TRUE,&closureSize,&closure);CHKERRQ(ierr);
  }
  *childrenMats = refPointFieldMats;
  *childrenN = refPointFieldN;
  ierr = ISRestoreIndices(refAnIS,&refAnchors);CHKERRQ(ierr);
  ierr = PetscFree(rows);CHKERRQ(ierr);
  ierr = PetscFree(cols);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexReferenceTreeRestoreChildrenMatrices(DM refTree, PetscScalar ****childrenMats, PetscInt ***childrenN)
{
  PetscDS        ds;
  PetscInt       **refPointFieldN;
  PetscScalar    ***refPointFieldMats;
  PetscInt       numFields, maxFields, pRefStart, pRefEnd, p, f;
  PetscSection   refConSec;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  refPointFieldN = *childrenN;
  *childrenN = NULL;
  refPointFieldMats = *childrenMats;
  *childrenMats = NULL;
  ierr = DMGetDS(refTree,&ds);CHKERRQ(ierr);
  ierr = PetscDSGetNumFields(ds,&numFields);CHKERRQ(ierr);
  maxFields = PetscMax(1,numFields);
  ierr = DMGetDefaultConstraints(refTree,&refConSec,NULL);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(refConSec,&pRefStart,&pRefEnd);CHKERRQ(ierr);
  for (p = pRefStart; p < pRefEnd; p++) {
    PetscInt parent, pDof;

    ierr = DMPlexGetTreeParent(refTree,p,&parent,NULL);CHKERRQ(ierr);
    ierr = PetscSectionGetDof(refConSec,p,&pDof);CHKERRQ(ierr);
    if (!pDof || parent == p) continue;

    for (f = 0; f < maxFields; f++) {
      PetscInt cDof;

      if (numFields) {
        ierr = PetscSectionGetFieldDof(refConSec,p,f,&cDof);CHKERRQ(ierr);
      }
      else {
        ierr = PetscSectionGetDof(refConSec,p,&cDof);CHKERRQ(ierr);
      }

      ierr = PetscFree(refPointFieldMats[p - pRefStart][f]);CHKERRQ(ierr);
    }
    ierr = PetscFree(refPointFieldMats[p - pRefStart]);CHKERRQ(ierr);
    ierr = PetscFree(refPointFieldN[p - pRefStart]);CHKERRQ(ierr);
  }
  ierr = PetscFree(refPointFieldMats);CHKERRQ(ierr);
  ierr = PetscFree(refPointFieldN);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexComputeAnchorMatrix_Tree_FromReference(DM dm, PetscSection section, PetscSection conSec, Mat cMat)
{
  DM             refTree;
  PetscDS        ds;
  Mat            refCmat;
  PetscInt       numFields, maxFields, f, pRefStart, pRefEnd, p, maxDof, maxAnDof, *perm, *iperm, pStart, pEnd, conStart, conEnd, **refPointFieldN;
  PetscScalar ***refPointFieldMats, *pointWork;
  PetscSection   refConSec, refAnSec, anSec;
  IS             refAnIS, anIS;
  const PetscInt *anchors;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMGetDS(dm,&ds);CHKERRQ(ierr);
  ierr = PetscDSGetNumFields(ds,&numFields);CHKERRQ(ierr);
  maxFields = PetscMax(1,numFields);
  ierr = DMPlexGetReferenceTree(dm,&refTree);CHKERRQ(ierr);
  ierr = DMCopyDisc(dm,refTree);CHKERRQ(ierr);
  ierr = DMGetDefaultConstraints(refTree,&refConSec,&refCmat);CHKERRQ(ierr);
  ierr = DMPlexGetAnchors(refTree,&refAnSec,&refAnIS);CHKERRQ(ierr);
  ierr = DMPlexGetAnchors(dm,&anSec,&anIS);CHKERRQ(ierr);
  ierr = ISGetIndices(anIS,&anchors);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(refConSec,&pRefStart,&pRefEnd);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(conSec,&conStart,&conEnd);CHKERRQ(ierr);
  ierr = PetscSectionGetMaxDof(refConSec,&maxDof);CHKERRQ(ierr);
  ierr = PetscSectionGetMaxDof(refAnSec,&maxAnDof);CHKERRQ(ierr);
  ierr = PetscMalloc1(maxDof*maxDof*maxAnDof,&pointWork);CHKERRQ(ierr);

  /* step 1: get submats for every constrained point in the reference tree */
  ierr = DMPlexReferenceTreeGetChildrenMatrices(refTree,&refPointFieldMats,&refPointFieldN);CHKERRQ(ierr);

  /* step 2: compute the preorder */
  ierr = DMPlexGetChart(dm,&pStart,&pEnd);CHKERRQ(ierr);
  ierr = PetscMalloc2(pEnd-pStart,&perm,pEnd-pStart,&iperm);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; p++) {
    perm[p - pStart] = p;
    iperm[p - pStart] = p-pStart;
  }
  for (p = 0; p < pEnd - pStart;) {
    PetscInt point = perm[p];
    PetscInt parent;

    ierr = DMPlexGetTreeParent(dm,point,&parent,NULL);CHKERRQ(ierr);
    if (parent == point) {
      p++;
    }
    else {
      PetscInt size, closureSize, *closure = NULL, i;

      ierr = DMPlexGetTransitiveClosure(dm,parent,PETSC_TRUE,&closureSize,&closure);CHKERRQ(ierr);
      for (i = 0; i < closureSize; i++) {
        PetscInt q = closure[2*i];
        if (iperm[q-pStart] > iperm[point-pStart]) {
          /* swap */
          perm[p]               = q;
          perm[iperm[q-pStart]] = point;
          iperm[point-pStart]   = iperm[q-pStart];
          iperm[q-pStart]       = p;
          break;
        }
      }
      size = closureSize;
      ierr = DMPlexRestoreTransitiveClosure(dm,parent,PETSC_TRUE,&closureSize,&closure);CHKERRQ(ierr);
      if (i == size) {
        p++;
      }
    }
  }

  /* step 3: fill the constraint matrix */
  /* we are going to use a preorder progressive fill strategy.  Mat doesn't
   * allow progressive fill without assembly, so we are going to set up the
   * values outside of the Mat first.
   */
  {
    PetscInt nRows, row, nnz;
    PetscBool done;
    const PetscInt *ia, *ja;
    PetscScalar *vals;

    ierr = MatGetRowIJ(cMat,0,PETSC_FALSE,PETSC_FALSE,&nRows,&ia,&ja,&done);CHKERRQ(ierr);
    if (!done) SETERRQ(PetscObjectComm((PetscObject)cMat),PETSC_ERR_PLIB,"Could not get RowIJ of constraint matrix");
    nnz  = ia[nRows];
    /* malloc and then zero rows right before we fill them: this way valgrind
     * can tell if we are doing progressive fill in the wrong order */
    ierr = PetscMalloc1(nnz,&vals);CHKERRQ(ierr);
    for (p = 0; p < pEnd - pStart; p++) {
      PetscInt        parent, childid, closureSize, *closure = NULL;
      PetscInt        point = perm[p], pointDof;

      ierr = DMPlexGetTreeParent(dm,point,&parent,&childid);CHKERRQ(ierr);
      if ((point < conStart) || (point >= conEnd) || (parent == point)) continue;
      ierr = PetscSectionGetDof(conSec,point,&pointDof);CHKERRQ(ierr);
      if (!pointDof) continue;
      ierr = DMPlexGetTransitiveClosure(dm,parent,PETSC_TRUE,&closureSize,&closure);CHKERRQ(ierr);
      for (f = 0; f < maxFields; f++) {
        PetscInt cDof, cOff, numCols, numFillCols, i, r, matOffset, offset;
        PetscScalar *pointMat;
        const PetscInt    **perms;
        const PetscScalar **flips;

        if (numFields) {
          ierr = PetscSectionGetFieldDof(conSec,point,f,&cDof);CHKERRQ(ierr);
          ierr = PetscSectionGetFieldOffset(conSec,point,f,&cOff);CHKERRQ(ierr);
        }
        else {
          ierr = PetscSectionGetDof(conSec,point,&cDof);CHKERRQ(ierr);
          ierr = PetscSectionGetOffset(conSec,point,&cOff);CHKERRQ(ierr);
        }
        if (!cDof) continue;
        if (numFields) {ierr = PetscSectionGetFieldPointSyms(section,f,closureSize,closure,&perms,&flips);CHKERRQ(ierr);}
        else           {ierr = PetscSectionGetPointSyms(section,closureSize,closure,&perms,&flips);CHKERRQ(ierr);}

        /* make sure that every row for this point is the same size */
#if defined(PETSC_USE_DEBUG)
        for (r = 0; r < cDof; r++) {
          if (cDof > 1 && r) {
            if ((ia[cOff+r+1]-ia[cOff+r]) != (ia[cOff+r]-ia[cOff+r-1])) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Two point rows have different nnz: %D vs. %D", (ia[cOff+r+1]-ia[cOff+r]), (ia[cOff+r]-ia[cOff+r-1]));
          }
        }
#endif
        /* zero rows */
        for (i = ia[cOff] ; i< ia[cOff+cDof];i++) {
          vals[i] = 0.;
        }
        matOffset = ia[cOff];
        numFillCols = ia[cOff+1] - matOffset;
        pointMat = refPointFieldMats[childid-pRefStart][f];
        numCols = refPointFieldN[childid-pRefStart][f];
        offset = 0;
        for (i = 0; i < closureSize; i++) {
          PetscInt q = closure[2*i];
          PetscInt aDof, aOff, j, k, qConDof, qConOff;
          const PetscInt    *perm = perms ? perms[i] : NULL;
          const PetscScalar *flip = flips ? flips[i] : NULL;

          qConDof = qConOff = 0;
          if (numFields) {
            ierr = PetscSectionGetFieldDof(section,q,f,&aDof);CHKERRQ(ierr);
            ierr = PetscSectionGetFieldOffset(section,q,f,&aOff);CHKERRQ(ierr);
            if (q >= conStart && q < conEnd) {
              ierr = PetscSectionGetFieldDof(conSec,q,f,&qConDof);CHKERRQ(ierr);
              ierr = PetscSectionGetFieldOffset(conSec,q,f,&qConOff);CHKERRQ(ierr);
            }
          }
          else {
            ierr = PetscSectionGetDof(section,q,&aDof);CHKERRQ(ierr);
            ierr = PetscSectionGetOffset(section,q,&aOff);CHKERRQ(ierr);
            if (q >= conStart && q < conEnd) {
              ierr = PetscSectionGetDof(conSec,q,&qConDof);CHKERRQ(ierr);
              ierr = PetscSectionGetOffset(conSec,q,&qConOff);CHKERRQ(ierr);
            }
          }
          if (!aDof) continue;
          if (qConDof) {
            /* this point has anchors: its rows of the matrix should already
             * be filled, thanks to preordering */
            /* first multiply into pointWork, then set in matrix */
            PetscInt aMatOffset = ia[qConOff];
            PetscInt aNumFillCols = ia[qConOff + 1] - aMatOffset;
            for (r = 0; r < cDof; r++) {
              for (j = 0; j < aNumFillCols; j++) {
                PetscScalar inVal = 0;
                for (k = 0; k < aDof; k++) {
                  PetscInt col = perm ? perm[k] : k;

                  inVal += pointMat[r * numCols + offset + col] * vals[aMatOffset + aNumFillCols * k + j] * (flip ? flip[col] : 1.);
                }
                pointWork[r * aNumFillCols + j] = inVal;
              }
            }
            /* assume that the columns are sorted, spend less time searching */
            for (j = 0, k = 0; j < aNumFillCols; j++) {
              PetscInt col = ja[aMatOffset + j];
              for (;k < numFillCols; k++) {
                if (ja[matOffset + k] == col) {
                  break;
                }
              }
              if (k == numFillCols) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"No nonzero space for (%d, %d)", cOff, col);
              for (r = 0; r < cDof; r++) {
                vals[matOffset + numFillCols * r + k] = pointWork[r * aNumFillCols + j];
              }
            }
          }
          else {
            /* find where to put this portion of pointMat into the matrix */
            for (k = 0; k < numFillCols; k++) {
              if (ja[matOffset + k] == aOff) {
                break;
              }
            }
            if (k == numFillCols) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"No nonzero space for (%d, %d)", cOff, aOff);
            for (r = 0; r < cDof; r++) {
              for (j = 0; j < aDof; j++) {
                PetscInt col = perm ? perm[j] : j;

                vals[matOffset + numFillCols * r + k + col] += pointMat[r * numCols + offset + j] * (flip ? flip[col] : 1.);
              }
            }
          }
          offset += aDof;
        }
        if (numFields) {
          ierr = PetscSectionRestoreFieldPointSyms(section,f,closureSize,closure,&perms,&flips);CHKERRQ(ierr);
        } else {
          ierr = PetscSectionRestorePointSyms(section,closureSize,closure,&perms,&flips);CHKERRQ(ierr);
        }
      }
      ierr = DMPlexRestoreTransitiveClosure(dm,parent,PETSC_TRUE,&closureSize,&closure);CHKERRQ(ierr);
    }
    for (row = 0; row < nRows; row++) {
      ierr = MatSetValues(cMat,1,&row,ia[row+1]-ia[row],&ja[ia[row]],&vals[ia[row]],INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = MatRestoreRowIJ(cMat,0,PETSC_FALSE,PETSC_FALSE,&nRows,&ia,&ja,&done);CHKERRQ(ierr);
    if (!done) SETERRQ(PetscObjectComm((PetscObject)cMat),PETSC_ERR_PLIB,"Could not restore RowIJ of constraint matrix");
    ierr = MatAssemblyBegin(cMat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(cMat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = PetscFree(vals);CHKERRQ(ierr);
  }

  /* clean up */
  ierr = ISRestoreIndices(anIS,&anchors);CHKERRQ(ierr);
  ierr = PetscFree2(perm,iperm);CHKERRQ(ierr);
  ierr = PetscFree(pointWork);CHKERRQ(ierr);
  ierr = DMPlexReferenceTreeRestoreChildrenMatrices(refTree,&refPointFieldMats,&refPointFieldN);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* refine a single cell on rank 0: this is not intended to provide good local refinement, only to create an example of
 * a non-conforming mesh.  Local refinement comes later */
PetscErrorCode DMPlexTreeRefineCell (DM dm, PetscInt cell, DM *ncdm)
{
  DM K;
  PetscMPIInt rank;
  PetscInt dim, *pNewStart, *pNewEnd, *pNewCount, *pOldStart, *pOldEnd, offset, d, pStart, pEnd;
  PetscInt numNewCones, *newConeSizes, *newCones, *newOrientations;
  PetscInt *Kembedding;
  PetscInt *cellClosure=NULL, nc;
  PetscScalar *newVertexCoords;
  PetscInt numPointsWithParents, *parents, *childIDs, *perm, *iperm, *preOrient, pOffset;
  PetscSection parentSection;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)dm),&rank);CHKERRQ(ierr);
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  ierr = DMPlexCreate(PetscObjectComm((PetscObject)dm), ncdm);CHKERRQ(ierr);
  ierr = DMSetDimension(*ncdm,dim);CHKERRQ(ierr);

  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject)dm),&parentSection);CHKERRQ(ierr);
  ierr = DMPlexGetReferenceTree(dm,&K);CHKERRQ(ierr);
  if (!rank) {
    /* compute the new charts */
    ierr = PetscMalloc5(dim+1,&pNewCount,dim+1,&pNewStart,dim+1,&pNewEnd,dim+1,&pOldStart,dim+1,&pOldEnd);CHKERRQ(ierr);
    offset = 0;
    for (d = 0; d <= dim; d++) {
      PetscInt pOldCount, kStart, kEnd, k;

      pNewStart[d] = offset;
      ierr = DMPlexGetHeightStratum(dm,d,&pOldStart[d],&pOldEnd[d]);CHKERRQ(ierr);
      ierr = DMPlexGetHeightStratum(K,d,&kStart,&kEnd);CHKERRQ(ierr);
      pOldCount = pOldEnd[d] - pOldStart[d];
      /* adding the new points */
      pNewCount[d] = pOldCount + kEnd - kStart;
      if (!d) {
        /* removing the cell */
        pNewCount[d]--;
      }
      for (k = kStart; k < kEnd; k++) {
        PetscInt parent;
        ierr = DMPlexGetTreeParent(K,k,&parent,NULL);CHKERRQ(ierr);
        if (parent == k) {
          /* avoid double counting points that won't actually be new */
          pNewCount[d]--;
        }
      }
      pNewEnd[d] = pNewStart[d] + pNewCount[d];
      offset = pNewEnd[d];

    }
    if (cell < pOldStart[0] || cell >= pOldEnd[0]) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"%d not in cell range [%d, %d)", cell, pOldStart[0], pOldEnd[0]);
    /* get the current closure of the cell that we are removing */
    ierr = DMPlexGetTransitiveClosure(dm,cell,PETSC_TRUE,&nc,&cellClosure);CHKERRQ(ierr);

    ierr = PetscMalloc1(pNewEnd[dim],&newConeSizes);CHKERRQ(ierr);
    {
      PetscInt kStart, kEnd, k, closureSizeK, *closureK = NULL, j;

      ierr = DMPlexGetChart(K,&kStart,&kEnd);CHKERRQ(ierr);
      ierr = PetscMalloc4(kEnd-kStart,&Kembedding,kEnd-kStart,&perm,kEnd-kStart,&iperm,kEnd-kStart,&preOrient);CHKERRQ(ierr);

      for (k = kStart; k < kEnd; k++) {
        perm[k - kStart] = k;
        iperm [k - kStart] = k - kStart;
        preOrient[k - kStart] = 0;
      }

      ierr = DMPlexGetTransitiveClosure(K,0,PETSC_TRUE,&closureSizeK,&closureK);CHKERRQ(ierr);
      for (j = 1; j < closureSizeK; j++) {
        PetscInt parentOrientA = closureK[2*j+1];
        PetscInt parentOrientB = cellClosure[2*j+1];
        PetscInt p, q;

        p = closureK[2*j];
        q = cellClosure[2*j];
        for (d = 0; d <= dim; d++) {
          if (q >= pOldStart[d] && q < pOldEnd[d]) {
            Kembedding[p] = (q - pOldStart[d]) + pNewStart[d];
          }
        }
        if (parentOrientA != parentOrientB) {
          PetscInt numChildren, i;
          const PetscInt *children;

          ierr = DMPlexGetTreeChildren(K,p,&numChildren,&children);CHKERRQ(ierr);
          for (i = 0; i < numChildren; i++) {
            PetscInt kPerm, oPerm;

            k    = children[i];
            ierr = DMPlexReferenceTreeGetChildSymmetry(K,p,parentOrientA,0,k,parentOrientB,&oPerm,&kPerm);CHKERRQ(ierr);
            /* perm = what refTree position I'm in */
            perm[kPerm-kStart]      = k;
            /* iperm = who is at this position */
            iperm[k-kStart]         = kPerm-kStart;
            preOrient[kPerm-kStart] = oPerm;
          }
        }
      }
      ierr = DMPlexRestoreTransitiveClosure(K,0,PETSC_TRUE,&closureSizeK,&closureK);CHKERRQ(ierr);
    }
    ierr = PetscSectionSetChart(parentSection,0,pNewEnd[dim]);CHKERRQ(ierr);
    offset = 0;
    numNewCones = 0;
    for (d = 0; d <= dim; d++) {
      PetscInt kStart, kEnd, k;
      PetscInt p;
      PetscInt size;

      for (p = pOldStart[d]; p < pOldEnd[d]; p++) {
        /* skip cell 0 */
        if (p == cell) continue;
        /* old cones to new cones */
        ierr = DMPlexGetConeSize(dm,p,&size);CHKERRQ(ierr);
        newConeSizes[offset++] = size;
        numNewCones += size;
      }

      ierr = DMPlexGetHeightStratum(K,d,&kStart,&kEnd);CHKERRQ(ierr);
      for (k = kStart; k < kEnd; k++) {
        PetscInt kParent;

        ierr = DMPlexGetTreeParent(K,k,&kParent,NULL);CHKERRQ(ierr);
        if (kParent != k) {
          Kembedding[k] = offset;
          ierr = DMPlexGetConeSize(K,k,&size);CHKERRQ(ierr);
          newConeSizes[offset++] = size;
          numNewCones += size;
          if (kParent != 0) {
            ierr = PetscSectionSetDof(parentSection,Kembedding[k],1);CHKERRQ(ierr);
          }
        }
      }
    }

    ierr = PetscSectionSetUp(parentSection);CHKERRQ(ierr);
    ierr = PetscSectionGetStorageSize(parentSection,&numPointsWithParents);CHKERRQ(ierr);
    ierr = PetscMalloc2(numNewCones,&newCones,numNewCones,&newOrientations);CHKERRQ(ierr);
    ierr = PetscMalloc2(numPointsWithParents,&parents,numPointsWithParents,&childIDs);CHKERRQ(ierr);

    /* fill new cones */
    offset = 0;
    for (d = 0; d <= dim; d++) {
      PetscInt kStart, kEnd, k, l;
      PetscInt p;
      PetscInt size;
      const PetscInt *cone, *orientation;

      for (p = pOldStart[d]; p < pOldEnd[d]; p++) {
        /* skip cell 0 */
        if (p == cell) continue;
        /* old cones to new cones */
        ierr = DMPlexGetConeSize(dm,p,&size);CHKERRQ(ierr);
        ierr = DMPlexGetCone(dm,p,&cone);CHKERRQ(ierr);
        ierr = DMPlexGetConeOrientation(dm,p,&orientation);CHKERRQ(ierr);
        for (l = 0; l < size; l++) {
          newCones[offset]          = (cone[l] - pOldStart[d + 1]) + pNewStart[d + 1];
          newOrientations[offset++] = orientation[l];
        }
      }

      ierr = DMPlexGetHeightStratum(K,d,&kStart,&kEnd);CHKERRQ(ierr);
      for (k = kStart; k < kEnd; k++) {
        PetscInt kPerm = perm[k], kParent;
        PetscInt preO  = preOrient[k];

        ierr = DMPlexGetTreeParent(K,k,&kParent,NULL);CHKERRQ(ierr);
        if (kParent != k) {
          /* embed new cones */
          ierr = DMPlexGetConeSize(K,k,&size);CHKERRQ(ierr);
          ierr = DMPlexGetCone(K,kPerm,&cone);CHKERRQ(ierr);
          ierr = DMPlexGetConeOrientation(K,kPerm,&orientation);CHKERRQ(ierr);
          for (l = 0; l < size; l++) {
            PetscInt q, m = (preO >= 0) ? ((preO + l) % size) : ((size -(preO + 1) - l) % size);
            PetscInt newO, lSize, oTrue;

            q                         = iperm[cone[m]];
            newCones[offset]          = Kembedding[q];
            ierr                      = DMPlexGetConeSize(K,q,&lSize);CHKERRQ(ierr);
            oTrue                     = orientation[m];
            oTrue                     = ((!lSize) || (preOrient[k] >= 0)) ? oTrue : -(oTrue + 2);
            newO                      = DihedralCompose(lSize,oTrue,preOrient[q]);
            newOrientations[offset++] = newO;
          }
          if (kParent != 0) {
            PetscInt newPoint = Kembedding[kParent];
            ierr              = PetscSectionGetOffset(parentSection,Kembedding[k],&pOffset);CHKERRQ(ierr);
            parents[pOffset]  = newPoint;
            childIDs[pOffset] = k;
          }
        }
      }
    }

    ierr = PetscMalloc1(dim*(pNewEnd[dim]-pNewStart[dim]),&newVertexCoords);CHKERRQ(ierr);

    /* fill coordinates */
    offset = 0;
    {
      PetscInt kStart, kEnd, l;
      PetscSection vSection;
      PetscInt v;
      Vec coords;
      PetscScalar *coordvals;
      PetscInt dof, off;
      PetscReal v0[3], J[9], detJ;

#if defined(PETSC_USE_DEBUG)
      {
        PetscInt k;
        ierr = DMPlexGetHeightStratum(K,0,&kStart,&kEnd);CHKERRQ(ierr);
        for (k = kStart; k < kEnd; k++) {
          ierr = DMPlexComputeCellGeometryFEM(K, k, NULL, v0, J, NULL, &detJ);CHKERRQ(ierr);
          if (detJ <= 0.) SETERRQ1 (PETSC_COMM_SELF,PETSC_ERR_PLIB,"reference tree cell %d has bad determinant",k);
        }
      }
#endif
      ierr = DMPlexComputeCellGeometryFEM(dm, cell, NULL, v0, J, NULL, &detJ);CHKERRQ(ierr);
      ierr = DMGetCoordinateSection(dm,&vSection);CHKERRQ(ierr);
      ierr = DMGetCoordinatesLocal(dm,&coords);CHKERRQ(ierr);
      ierr = VecGetArray(coords,&coordvals);CHKERRQ(ierr);
      for (v = pOldStart[dim]; v < pOldEnd[dim]; v++) {

        ierr = PetscSectionGetDof(vSection,v,&dof);CHKERRQ(ierr);
        ierr = PetscSectionGetOffset(vSection,v,&off);CHKERRQ(ierr);
        for (l = 0; l < dof; l++) {
          newVertexCoords[offset++] = coordvals[off + l];
        }
      }
      ierr = VecRestoreArray(coords,&coordvals);CHKERRQ(ierr);

      ierr = DMGetCoordinateSection(K,&vSection);CHKERRQ(ierr);
      ierr = DMGetCoordinatesLocal(K,&coords);CHKERRQ(ierr);
      ierr = VecGetArray(coords,&coordvals);CHKERRQ(ierr);
      ierr = DMPlexGetDepthStratum(K,0,&kStart,&kEnd);CHKERRQ(ierr);
      for (v = kStart; v < kEnd; v++) {
        PetscReal coord[3], newCoord[3];
        PetscInt  vPerm = perm[v];
        PetscInt  kParent;
        const PetscReal xi0[3] = {-1.,-1.,-1.};

        ierr = DMPlexGetTreeParent(K,v,&kParent,NULL);CHKERRQ(ierr);
        if (kParent != v) {
          /* this is a new vertex */
          ierr = PetscSectionGetOffset(vSection,vPerm,&off);CHKERRQ(ierr);
          for (l = 0; l < dim; ++l) coord[l] = PetscRealPart(coordvals[off+l]);
          CoordinatesRefToReal(dim, dim, xi0, v0, J, coord, newCoord);
          for (l = 0; l < dim; ++l) newVertexCoords[offset+l] = newCoord[l];
          offset += dim;
        }
      }
      ierr = VecRestoreArray(coords,&coordvals);CHKERRQ(ierr);
    }

    /* need to reverse the order of pNewCount: vertices first, cells last */
    for (d = 0; d < (dim + 1) / 2; d++) {
      PetscInt tmp;

      tmp = pNewCount[d];
      pNewCount[d] = pNewCount[dim - d];
      pNewCount[dim - d] = tmp;
    }

    ierr = DMPlexCreateFromDAG(*ncdm,dim,pNewCount,newConeSizes,newCones,newOrientations,newVertexCoords);CHKERRQ(ierr);
    ierr = DMPlexSetReferenceTree(*ncdm,K);CHKERRQ(ierr);
    ierr = DMPlexSetTree(*ncdm,parentSection,parents,childIDs);CHKERRQ(ierr);

    /* clean up */
    ierr = DMPlexRestoreTransitiveClosure(dm,cell,PETSC_TRUE,&nc,&cellClosure);CHKERRQ(ierr);
    ierr = PetscFree5(pNewCount,pNewStart,pNewEnd,pOldStart,pOldEnd);CHKERRQ(ierr);
    ierr = PetscFree(newConeSizes);CHKERRQ(ierr);
    ierr = PetscFree2(newCones,newOrientations);CHKERRQ(ierr);
    ierr = PetscFree(newVertexCoords);CHKERRQ(ierr);
    ierr = PetscFree2(parents,childIDs);CHKERRQ(ierr);
    ierr = PetscFree4(Kembedding,perm,iperm,preOrient);CHKERRQ(ierr);
  }
  else {
    PetscInt    p, counts[4];
    PetscInt    *coneSizes, *cones, *orientations;
    Vec         coordVec;
    PetscScalar *coords;

    for (d = 0; d <= dim; d++) {
      PetscInt dStart, dEnd;

      ierr = DMPlexGetDepthStratum(dm,d,&dStart,&dEnd);CHKERRQ(ierr);
      counts[d] = dEnd - dStart;
    }
    ierr = PetscMalloc1(pEnd-pStart,&coneSizes);CHKERRQ(ierr);
    for (p = pStart; p < pEnd; p++) {
      ierr = DMPlexGetConeSize(dm,p,&coneSizes[p-pStart]);CHKERRQ(ierr);
    }
    ierr = DMPlexGetCones(dm, &cones);CHKERRQ(ierr);
    ierr = DMPlexGetConeOrientations(dm, &orientations);CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocal(dm,&coordVec);CHKERRQ(ierr);
    ierr = VecGetArray(coordVec,&coords);CHKERRQ(ierr);

    ierr = PetscSectionSetChart(parentSection,pStart,pEnd);CHKERRQ(ierr);
    ierr = PetscSectionSetUp(parentSection);CHKERRQ(ierr);
    ierr = DMPlexCreateFromDAG(*ncdm,dim,counts,coneSizes,cones,orientations,NULL);CHKERRQ(ierr);
    ierr = DMPlexSetReferenceTree(*ncdm,K);CHKERRQ(ierr);
    ierr = DMPlexSetTree(*ncdm,parentSection,NULL,NULL);CHKERRQ(ierr);
    ierr = VecRestoreArray(coordVec,&coords);CHKERRQ(ierr);
  }
  ierr = PetscSectionDestroy(&parentSection);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexComputeInterpolatorTree(DM coarse, DM fine, PetscSF coarseToFine, PetscInt *childIds, Mat mat)
{
  PetscSF           coarseToFineEmbedded;
  PetscSection      globalCoarse, globalFine;
  PetscSection      localCoarse, localFine;
  PetscSection      aSec, cSec;
  PetscSection      rootIndicesSec, rootMatricesSec;
  PetscSection      leafIndicesSec, leafMatricesSec;
  PetscInt          *rootIndices, *leafIndices;
  PetscScalar       *rootMatrices, *leafMatrices;
  IS                aIS;
  const PetscInt    *anchors;
  Mat               cMat;
  PetscInt          numFields, maxFields;
  PetscInt          pStartC, pEndC, pStartF, pEndF, p;
  PetscInt          aStart, aEnd, cStart, cEnd;
  PetscInt          *maxChildIds;
  PetscInt          *offsets, *newOffsets, *offsetsCopy, *newOffsetsCopy, *rowOffsets, *numD, *numO;
  const PetscInt    ***perms;
  const PetscScalar ***flips;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetChart(coarse,&pStartC,&pEndC);CHKERRQ(ierr);
  ierr = DMPlexGetChart(fine,&pStartF,&pEndF);CHKERRQ(ierr);
  ierr = DMGetGlobalSection(fine,&globalFine);CHKERRQ(ierr);
  { /* winnow fine points that don't have global dofs out of the sf */
    PetscInt dof, cdof, numPointsWithDofs, offset, *pointsWithDofs, nleaves, l;
    const PetscInt *leaves;

    ierr = PetscSFGetGraph(coarseToFine,NULL,&nleaves,&leaves,NULL);CHKERRQ(ierr);
    for (l = 0, numPointsWithDofs = 0; l < nleaves; l++) {
      p = leaves ? leaves[l] : l;
      ierr = PetscSectionGetDof(globalFine,p,&dof);CHKERRQ(ierr);
      ierr = PetscSectionGetConstraintDof(globalFine,p,&cdof);CHKERRQ(ierr);
      if ((dof - cdof) > 0) {
        numPointsWithDofs++;
      }
    }
    ierr = PetscMalloc1(numPointsWithDofs,&pointsWithDofs);CHKERRQ(ierr);
    for (l = 0, offset = 0; l < nleaves; l++) {
      p = leaves ? leaves[l] : l;
      ierr = PetscSectionGetDof(globalFine,p,&dof);CHKERRQ(ierr);
      ierr = PetscSectionGetConstraintDof(globalFine,p,&cdof);CHKERRQ(ierr);
      if ((dof - cdof) > 0) {
        pointsWithDofs[offset++] = l;
      }
    }
    ierr = PetscSFCreateEmbeddedLeafSF(coarseToFine, numPointsWithDofs, pointsWithDofs, &coarseToFineEmbedded);CHKERRQ(ierr);
    ierr = PetscFree(pointsWithDofs);CHKERRQ(ierr);
  }
  /* communicate back to the coarse mesh which coarse points have children (that may require interpolation) */
  ierr = PetscMalloc1(pEndC-pStartC,&maxChildIds);CHKERRQ(ierr);
  for (p = pStartC; p < pEndC; p++) {
    maxChildIds[p - pStartC] = -2;
  }
  ierr = PetscSFReduceBegin(coarseToFineEmbedded,MPIU_INT,childIds,maxChildIds,MPIU_MAX);CHKERRQ(ierr);
  ierr = PetscSFReduceEnd(coarseToFineEmbedded,MPIU_INT,childIds,maxChildIds,MPIU_MAX);CHKERRQ(ierr);

  ierr = DMGetLocalSection(coarse,&localCoarse);CHKERRQ(ierr);
  ierr = DMGetGlobalSection(coarse,&globalCoarse);CHKERRQ(ierr);

  ierr = DMPlexGetAnchors(coarse,&aSec,&aIS);CHKERRQ(ierr);
  ierr = ISGetIndices(aIS,&anchors);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(aSec,&aStart,&aEnd);CHKERRQ(ierr);

  ierr = DMGetDefaultConstraints(coarse,&cSec,&cMat);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(cSec,&cStart,&cEnd);CHKERRQ(ierr);

  /* create sections that will send to children the indices and matrices they will need to construct the interpolator */
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject)coarse),&rootIndicesSec);CHKERRQ(ierr);
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject)coarse),&rootMatricesSec);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(rootIndicesSec,pStartC,pEndC);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(rootMatricesSec,pStartC,pEndC);CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(localCoarse,&numFields);CHKERRQ(ierr);
  maxFields = PetscMax(1,numFields);
  ierr = PetscMalloc7(maxFields+1,&offsets,maxFields+1,&offsetsCopy,maxFields+1,&newOffsets,maxFields+1,&newOffsetsCopy,maxFields+1,&rowOffsets,maxFields+1,&numD,maxFields+1,&numO);CHKERRQ(ierr);
  ierr = PetscMalloc2(maxFields+1,(PetscInt****)&perms,maxFields+1,(PetscScalar****)&flips);CHKERRQ(ierr);
  ierr = PetscMemzero((void *) perms, (maxFields+1) * sizeof(const PetscInt **));CHKERRQ(ierr);
  ierr = PetscMemzero((void *) flips, (maxFields+1) * sizeof(const PetscScalar **));CHKERRQ(ierr);

  for (p = pStartC; p < pEndC; p++) { /* count the sizes of the indices and matrices */
    PetscInt dof, matSize   = 0;
    PetscInt aDof           = 0;
    PetscInt cDof           = 0;
    PetscInt maxChildId     = maxChildIds[p - pStartC];
    PetscInt numRowIndices  = 0;
    PetscInt numColIndices  = 0;
    PetscInt f;

    ierr = PetscSectionGetDof(globalCoarse,p,&dof);CHKERRQ(ierr);
    if (dof < 0) {
      dof = -(dof + 1);
    }
    if (p >= aStart && p < aEnd) {
      ierr = PetscSectionGetDof(aSec,p,&aDof);CHKERRQ(ierr);
    }
    if (p >= cStart && p < cEnd) {
      ierr = PetscSectionGetDof(cSec,p,&cDof);CHKERRQ(ierr);
    }
    for (f = 0; f <= numFields; f++) offsets[f] = 0;
    for (f = 0; f <= numFields; f++) newOffsets[f] = 0;
    if (maxChildId >= 0) { /* this point has children (with dofs) that will need to be interpolated from the closure of p */
      PetscInt *closure = NULL, closureSize, cl;

      ierr = DMPlexGetTransitiveClosure(coarse,p,PETSC_TRUE,&closureSize,&closure);CHKERRQ(ierr);
      for (cl = 0; cl < closureSize; cl++) { /* get the closure */
        PetscInt c = closure[2 * cl], clDof;

        ierr = PetscSectionGetDof(localCoarse,c,&clDof);CHKERRQ(ierr);
        numRowIndices += clDof;
        for (f = 0; f < numFields; f++) {
          ierr = PetscSectionGetFieldDof(localCoarse,c,f,&clDof);CHKERRQ(ierr);
          offsets[f + 1] += clDof;
        }
      }
      for (f = 0; f < numFields; f++) {
        offsets[f + 1]   += offsets[f];
        newOffsets[f + 1] = offsets[f + 1];
      }
      /* get the number of indices needed and their field offsets */
      ierr = DMPlexAnchorsModifyMat(coarse,localCoarse,closureSize,numRowIndices,closure,NULL,NULL,NULL,&numColIndices,NULL,NULL,newOffsets,PETSC_FALSE);CHKERRQ(ierr);
      ierr = DMPlexRestoreTransitiveClosure(coarse,p,PETSC_TRUE,&closureSize,&closure);CHKERRQ(ierr);
      if (!numColIndices) { /* there are no hanging constraint modifications, so the matrix is just the identity: do not send it */
        numColIndices = numRowIndices;
        matSize = 0;
      }
      else if (numFields) { /* we send one submat for each field: sum their sizes */
        matSize = 0;
        for (f = 0; f < numFields; f++) {
          PetscInt numRow, numCol;

          numRow = offsets[f + 1] - offsets[f];
          numCol = newOffsets[f + 1] - newOffsets[f];
          matSize += numRow * numCol;
        }
      }
      else {
        matSize = numRowIndices * numColIndices;
      }
    } else if (maxChildId == -1) {
      if (cDof > 0) { /* this point's dofs are interpolated via cMat: get the submatrix of cMat */
        PetscInt aOff, a;

        ierr = PetscSectionGetOffset(aSec,p,&aOff);CHKERRQ(ierr);
        for (f = 0; f < numFields; f++) {
          PetscInt fDof;

          ierr = PetscSectionGetFieldDof(localCoarse,p,f,&fDof);CHKERRQ(ierr);
          offsets[f+1] = fDof;
        }
        for (a = 0; a < aDof; a++) {
          PetscInt anchor = anchors[a + aOff], aLocalDof;

          ierr = PetscSectionGetDof(localCoarse,anchor,&aLocalDof);CHKERRQ(ierr);
          numColIndices += aLocalDof;
          for (f = 0; f < numFields; f++) {
            PetscInt fDof;

            ierr = PetscSectionGetFieldDof(localCoarse,anchor,f,&fDof);CHKERRQ(ierr);
            newOffsets[f+1] += fDof;
          }
        }
        if (numFields) {
          matSize = 0;
          for (f = 0; f < numFields; f++) {
            matSize += offsets[f+1] * newOffsets[f+1];
          }
        }
        else {
          matSize = numColIndices * dof;
        }
      }
      else { /* no children, and no constraints on dofs: just get the global indices */
        numColIndices = dof;
        matSize       = 0;
      }
    }
    /* we will pack the column indices with the field offsets */
    ierr = PetscSectionSetDof(rootIndicesSec,p,numColIndices ? numColIndices+2*numFields : 0);CHKERRQ(ierr);
    ierr = PetscSectionSetDof(rootMatricesSec,p,matSize);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(rootIndicesSec);CHKERRQ(ierr);
  ierr = PetscSectionSetUp(rootMatricesSec);CHKERRQ(ierr);
  {
    PetscInt numRootIndices, numRootMatrices;

    ierr = PetscSectionGetStorageSize(rootIndicesSec,&numRootIndices);CHKERRQ(ierr);
    ierr = PetscSectionGetStorageSize(rootMatricesSec,&numRootMatrices);CHKERRQ(ierr);
    ierr = PetscMalloc2(numRootIndices,&rootIndices,numRootMatrices,&rootMatrices);CHKERRQ(ierr);
    for (p = pStartC; p < pEndC; p++) {
      PetscInt    numRowIndices, numColIndices, matSize, dof;
      PetscInt    pIndOff, pMatOff, f;
      PetscInt    *pInd;
      PetscInt    maxChildId = maxChildIds[p - pStartC];
      PetscScalar *pMat = NULL;

      ierr = PetscSectionGetDof(rootIndicesSec,p,&numColIndices);CHKERRQ(ierr);
      if (!numColIndices) {
        continue;
      }
      for (f = 0; f <= numFields; f++) {
        offsets[f]        = 0;
        newOffsets[f]     = 0;
        offsetsCopy[f]    = 0;
        newOffsetsCopy[f] = 0;
      }
      numColIndices -= 2 * numFields;
      ierr = PetscSectionGetOffset(rootIndicesSec,p,&pIndOff);CHKERRQ(ierr);
      pInd = &(rootIndices[pIndOff]);
      ierr = PetscSectionGetDof(rootMatricesSec,p,&matSize);CHKERRQ(ierr);
      if (matSize) {
        ierr = PetscSectionGetOffset(rootMatricesSec,p,&pMatOff);CHKERRQ(ierr);
        pMat = &rootMatrices[pMatOff];
      }
      ierr = PetscSectionGetDof(globalCoarse,p,&dof);CHKERRQ(ierr);
      if (dof < 0) {
        dof = -(dof + 1);
      }
      if (maxChildId >= 0) { /* build an identity matrix, apply matrix constraints on the right */
        PetscInt i, j;
        PetscInt numRowIndices = matSize / numColIndices;

        if (!numRowIndices) { /* don't need to calculate the mat, just the indices */
          PetscInt numIndices, *indices;
          ierr = DMPlexGetClosureIndices(coarse,localCoarse,globalCoarse,p,&numIndices,&indices,offsets);CHKERRQ(ierr);
          if (numIndices != numColIndices) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"mismatching constraint indices calculations");
          for (i = 0; i < numColIndices; i++) {
            pInd[i] = indices[i];
          }
          for (i = 0; i < numFields; i++) {
            pInd[numColIndices + i]             = offsets[i+1];
            pInd[numColIndices + numFields + i] = offsets[i+1];
          }
          ierr = DMPlexRestoreClosureIndices(coarse,localCoarse,globalCoarse,p,&numIndices,&indices,offsets);CHKERRQ(ierr);
        }
        else {
          PetscInt closureSize, *closure = NULL, cl;
          PetscScalar *pMatIn, *pMatModified;
          PetscInt numPoints,*points;

          ierr = DMGetWorkArray(coarse,numRowIndices * numRowIndices,MPIU_SCALAR,&pMatIn);CHKERRQ(ierr);
          for (i = 0; i < numRowIndices; i++) { /* initialize to the identity */
            for (j = 0; j < numRowIndices; j++) {
              pMatIn[i * numRowIndices + j] = (i == j) ? 1. : 0.;
            }
          }
          ierr = DMPlexGetTransitiveClosure(coarse, p, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
          for (f = 0; f < maxFields; f++) {
            if (numFields) {ierr = PetscSectionGetFieldPointSyms(localCoarse,f,closureSize,closure,&perms[f],&flips[f]);CHKERRQ(ierr);}
            else           {ierr = PetscSectionGetPointSyms(localCoarse,closureSize,closure,&perms[f],&flips[f]);CHKERRQ(ierr);}
          }
          if (numFields) {
            for (cl = 0; cl < closureSize; cl++) {
              PetscInt c = closure[2 * cl];

              for (f = 0; f < numFields; f++) {
                PetscInt fDof;

                ierr = PetscSectionGetFieldDof(localCoarse,c,f,&fDof);CHKERRQ(ierr);
                offsets[f + 1] += fDof;
              }
            }
            for (f = 0; f < numFields; f++) {
              offsets[f + 1]   += offsets[f];
              newOffsets[f + 1] = offsets[f + 1];
            }
          }
          /* TODO : flips here ? */
          /* apply hanging node constraints on the right, get the new points and the new offsets */
          ierr = DMPlexAnchorsModifyMat(coarse,localCoarse,closureSize,numRowIndices,closure,perms,pMatIn,&numPoints,NULL,&points,&pMatModified,newOffsets,PETSC_FALSE);CHKERRQ(ierr);
          for (f = 0; f < maxFields; f++) {
            if (numFields) {ierr = PetscSectionRestoreFieldPointSyms(localCoarse,f,closureSize,closure,&perms[f],&flips[f]);CHKERRQ(ierr);}
            else           {ierr = PetscSectionRestorePointSyms(localCoarse,closureSize,closure,&perms[f],&flips[f]);CHKERRQ(ierr);}
          }
          for (f = 0; f < maxFields; f++) {
            if (numFields) {ierr = PetscSectionGetFieldPointSyms(localCoarse,f,numPoints,points,&perms[f],&flips[f]);CHKERRQ(ierr);}
            else           {ierr = PetscSectionGetPointSyms(localCoarse,numPoints,points,&perms[f],&flips[f]);CHKERRQ(ierr);}
          }
          if (!numFields) {
            for (i = 0; i < numRowIndices * numColIndices; i++) {
              pMat[i] = pMatModified[i];
            }
          }
          else {
            PetscInt i, j, count;
            for (f = 0, count = 0; f < numFields; f++) {
              for (i = offsets[f]; i < offsets[f+1]; i++) {
                for (j = newOffsets[f]; j < newOffsets[f+1]; j++, count++) {
                  pMat[count] = pMatModified[i * numColIndices + j];
                }
              }
            }
          }
          ierr = DMRestoreWorkArray(coarse,numRowIndices * numColIndices,MPIU_SCALAR,&pMatModified);CHKERRQ(ierr);
          ierr = DMPlexRestoreTransitiveClosure(coarse, p, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
          ierr = DMRestoreWorkArray(coarse,numRowIndices * numColIndices,MPIU_SCALAR,&pMatIn);CHKERRQ(ierr);
          if (numFields) {
            for (f = 0; f < numFields; f++) {
              pInd[numColIndices + f]             = offsets[f+1];
              pInd[numColIndices + numFields + f] = newOffsets[f+1];
            }
            for (cl = 0; cl < numPoints; cl++) {
              PetscInt globalOff, c = points[2*cl];
              ierr = PetscSectionGetOffset(globalCoarse, c, &globalOff);CHKERRQ(ierr);
              ierr = DMPlexGetIndicesPointFields_Internal(localCoarse, PETSC_FALSE, c, globalOff < 0 ? -(globalOff+1) : globalOff, newOffsets, PETSC_FALSE, perms, cl, NULL, pInd);CHKERRQ(ierr);
            }
          } else {
            for (cl = 0; cl < numPoints; cl++) {
              PetscInt c = points[2*cl], globalOff;
              const PetscInt *perm = perms[0] ? perms[0][cl] : NULL;

              ierr = PetscSectionGetOffset(globalCoarse, c, &globalOff);CHKERRQ(ierr);
              ierr = DMPlexGetIndicesPoint_Internal(localCoarse, PETSC_FALSE, c, globalOff < 0 ? -(globalOff+1) : globalOff, newOffsets, PETSC_FALSE, perm, NULL, pInd);CHKERRQ(ierr);
            }
          }
          for (f = 0; f < maxFields; f++) {
            if (numFields) {ierr = PetscSectionRestoreFieldPointSyms(localCoarse,f,numPoints,points,&perms[f],&flips[f]);CHKERRQ(ierr);}
            else           {ierr = PetscSectionRestorePointSyms(localCoarse,numPoints,points,&perms[f],&flips[f]);CHKERRQ(ierr);}
          }
          ierr = DMRestoreWorkArray(coarse,numPoints,MPIU_SCALAR,&points);CHKERRQ(ierr);
        }
      }
      else if (matSize) {
        PetscInt cOff;
        PetscInt *rowIndices, *colIndices, a, aDof, aOff;

        numRowIndices = matSize / numColIndices;
        if (numRowIndices != dof) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Miscounted dofs");
        ierr = DMGetWorkArray(coarse,numRowIndices,MPIU_INT,&rowIndices);CHKERRQ(ierr);
        ierr = DMGetWorkArray(coarse,numColIndices,MPIU_INT,&colIndices);CHKERRQ(ierr);
        ierr = PetscSectionGetOffset(cSec,p,&cOff);CHKERRQ(ierr);
        ierr = PetscSectionGetDof(aSec,p,&aDof);CHKERRQ(ierr);
        ierr = PetscSectionGetOffset(aSec,p,&aOff);CHKERRQ(ierr);
        if (numFields) {
          for (f = 0; f < numFields; f++) {
            PetscInt fDof;

            ierr = PetscSectionGetFieldDof(cSec,p,f,&fDof);CHKERRQ(ierr);
            offsets[f + 1] = fDof;
            for (a = 0; a < aDof; a++) {
              PetscInt anchor = anchors[a + aOff];
              ierr = PetscSectionGetFieldDof(localCoarse,anchor,f,&fDof);CHKERRQ(ierr);
              newOffsets[f + 1] += fDof;
            }
          }
          for (f = 0; f < numFields; f++) {
            offsets[f + 1]       += offsets[f];
            offsetsCopy[f + 1]    = offsets[f + 1];
            newOffsets[f + 1]    += newOffsets[f];
            newOffsetsCopy[f + 1] = newOffsets[f + 1];
          }
          ierr = DMPlexGetIndicesPointFields_Internal(cSec,PETSC_TRUE,p,cOff,offsetsCopy,PETSC_TRUE,NULL,-1, NULL,rowIndices);CHKERRQ(ierr);
          for (a = 0; a < aDof; a++) {
            PetscInt anchor = anchors[a + aOff], lOff;
            ierr = PetscSectionGetOffset(localCoarse,anchor,&lOff);CHKERRQ(ierr);
            ierr = DMPlexGetIndicesPointFields_Internal(localCoarse,PETSC_TRUE,anchor,lOff,newOffsetsCopy,PETSC_TRUE,NULL,-1, NULL,colIndices);CHKERRQ(ierr);
          }
        }
        else {
          ierr = DMPlexGetIndicesPoint_Internal(cSec,PETSC_TRUE,p,cOff,offsetsCopy,PETSC_TRUE,NULL, NULL,rowIndices);CHKERRQ(ierr);
          for (a = 0; a < aDof; a++) {
            PetscInt anchor = anchors[a + aOff], lOff;
            ierr = PetscSectionGetOffset(localCoarse,anchor,&lOff);CHKERRQ(ierr);
            ierr = DMPlexGetIndicesPoint_Internal(localCoarse,PETSC_TRUE,anchor,lOff,newOffsetsCopy,PETSC_TRUE,NULL, NULL,colIndices);CHKERRQ(ierr);
          }
        }
        if (numFields) {
          PetscInt count, a;

          for (f = 0, count = 0; f < numFields; f++) {
            PetscInt iSize = offsets[f + 1] - offsets[f];
            PetscInt jSize = newOffsets[f + 1] - newOffsets[f];
            ierr = MatGetValues(cMat,iSize,&rowIndices[offsets[f]],jSize,&colIndices[newOffsets[f]],&pMat[count]);CHKERRQ(ierr);
            count += iSize * jSize;
            pInd[numColIndices + f]             = offsets[f+1];
            pInd[numColIndices + numFields + f] = newOffsets[f+1];
          }
          for (a = 0; a < aDof; a++) {
            PetscInt anchor = anchors[a + aOff];
            PetscInt gOff;
            ierr = PetscSectionGetOffset(globalCoarse,anchor,&gOff);CHKERRQ(ierr);
            ierr = DMPlexGetIndicesPointFields_Internal(localCoarse,PETSC_FALSE,anchor,gOff < 0 ? -(gOff + 1) : gOff,newOffsets,PETSC_FALSE,NULL,-1, NULL,pInd);CHKERRQ(ierr);
          }
        }
        else {
          PetscInt a;
          ierr = MatGetValues(cMat,numRowIndices,rowIndices,numColIndices,colIndices,pMat);CHKERRQ(ierr);
          for (a = 0; a < aDof; a++) {
            PetscInt anchor = anchors[a + aOff];
            PetscInt gOff;
            ierr = PetscSectionGetOffset(globalCoarse,anchor,&gOff);CHKERRQ(ierr);
            ierr = DMPlexGetIndicesPoint_Internal(localCoarse,PETSC_FALSE,anchor,gOff < 0 ? -(gOff + 1) : gOff,newOffsets,PETSC_FALSE,NULL, NULL,pInd);CHKERRQ(ierr);
          }
        }
        ierr = DMRestoreWorkArray(coarse,numColIndices,MPIU_INT,&colIndices);CHKERRQ(ierr);
        ierr = DMRestoreWorkArray(coarse,numRowIndices,MPIU_INT,&rowIndices);CHKERRQ(ierr);
      }
      else {
        PetscInt gOff;

        ierr = PetscSectionGetOffset(globalCoarse,p,&gOff);CHKERRQ(ierr);
        if (numFields) {
          for (f = 0; f < numFields; f++) {
            PetscInt fDof;
            ierr = PetscSectionGetFieldDof(localCoarse,p,f,&fDof);CHKERRQ(ierr);
            offsets[f + 1] = fDof + offsets[f];
          }
          for (f = 0; f < numFields; f++) {
            pInd[numColIndices + f]             = offsets[f+1];
            pInd[numColIndices + numFields + f] = offsets[f+1];
          }
          ierr = DMPlexGetIndicesPointFields_Internal(localCoarse,PETSC_FALSE,p,gOff < 0 ? -(gOff + 1) : gOff,offsets,PETSC_FALSE,NULL,-1, NULL,pInd);CHKERRQ(ierr);
        } else {
          ierr = DMPlexGetIndicesPoint_Internal(localCoarse,PETSC_FALSE,p,gOff < 0 ? -(gOff + 1) : gOff,offsets,PETSC_FALSE,NULL, NULL,pInd);CHKERRQ(ierr);
        }
      }
    }
    ierr = PetscFree(maxChildIds);CHKERRQ(ierr);
  }
  {
    PetscSF  indicesSF, matricesSF;
    PetscInt *remoteOffsetsIndices, *remoteOffsetsMatrices, numLeafIndices, numLeafMatrices;

    ierr = PetscSectionCreate(PetscObjectComm((PetscObject)fine),&leafIndicesSec);CHKERRQ(ierr);
    ierr = PetscSectionCreate(PetscObjectComm((PetscObject)fine),&leafMatricesSec);CHKERRQ(ierr);
    ierr = PetscSFDistributeSection(coarseToFineEmbedded,rootIndicesSec,&remoteOffsetsIndices,leafIndicesSec);CHKERRQ(ierr);
    ierr = PetscSFDistributeSection(coarseToFineEmbedded,rootMatricesSec,&remoteOffsetsMatrices,leafMatricesSec);CHKERRQ(ierr);
    ierr = PetscSFCreateSectionSF(coarseToFineEmbedded,rootIndicesSec,remoteOffsetsIndices,leafIndicesSec,&indicesSF);CHKERRQ(ierr);
    ierr = PetscSFCreateSectionSF(coarseToFineEmbedded,rootMatricesSec,remoteOffsetsMatrices,leafMatricesSec,&matricesSF);CHKERRQ(ierr);
    ierr = PetscSFDestroy(&coarseToFineEmbedded);CHKERRQ(ierr);
    ierr = PetscFree(remoteOffsetsIndices);CHKERRQ(ierr);
    ierr = PetscFree(remoteOffsetsMatrices);CHKERRQ(ierr);
    ierr = PetscSectionGetStorageSize(leafIndicesSec,&numLeafIndices);CHKERRQ(ierr);
    ierr = PetscSectionGetStorageSize(leafMatricesSec,&numLeafMatrices);CHKERRQ(ierr);
    ierr = PetscMalloc2(numLeafIndices,&leafIndices,numLeafMatrices,&leafMatrices);CHKERRQ(ierr);
    ierr = PetscSFBcastBegin(indicesSF,MPIU_INT,rootIndices,leafIndices);CHKERRQ(ierr);
    ierr = PetscSFBcastBegin(matricesSF,MPIU_SCALAR,rootMatrices,leafMatrices);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(indicesSF,MPIU_INT,rootIndices,leafIndices);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(matricesSF,MPIU_SCALAR,rootMatrices,leafMatrices);CHKERRQ(ierr);
    ierr = PetscSFDestroy(&matricesSF);CHKERRQ(ierr);
    ierr = PetscSFDestroy(&indicesSF);CHKERRQ(ierr);
    ierr = PetscFree2(rootIndices,rootMatrices);CHKERRQ(ierr);
    ierr = PetscSectionDestroy(&rootIndicesSec);CHKERRQ(ierr);
    ierr = PetscSectionDestroy(&rootMatricesSec);CHKERRQ(ierr);
  }
  /* count to preallocate */
  ierr = DMGetLocalSection(fine,&localFine);CHKERRQ(ierr);
  {
    PetscInt    nGlobal;
    PetscInt    *dnnz, *onnz;
    PetscLayout rowMap, colMap;
    PetscInt    rowStart, rowEnd, colStart, colEnd;
    PetscInt    maxDof;
    PetscInt    *rowIndices;
    DM           refTree;
    PetscInt     **refPointFieldN;
    PetscScalar  ***refPointFieldMats;
    PetscSection refConSec, refAnSec;
    PetscInt     pRefStart,pRefEnd,maxConDof,maxColumns,leafStart,leafEnd;
    PetscScalar  *pointWork;

    ierr = PetscSectionGetConstrainedStorageSize(globalFine,&nGlobal);CHKERRQ(ierr);
    ierr = PetscCalloc2(nGlobal,&dnnz,nGlobal,&onnz);CHKERRQ(ierr);
    ierr = MatGetLayouts(mat,&rowMap,&colMap);CHKERRQ(ierr);
    ierr = PetscLayoutSetUp(rowMap);CHKERRQ(ierr);
    ierr = PetscLayoutSetUp(colMap);CHKERRQ(ierr);
    ierr = PetscLayoutGetRange(rowMap,&rowStart,&rowEnd);CHKERRQ(ierr);
    ierr = PetscLayoutGetRange(colMap,&colStart,&colEnd);CHKERRQ(ierr);
    ierr = PetscSectionGetMaxDof(globalFine,&maxDof);CHKERRQ(ierr);
    ierr = PetscSectionGetChart(leafIndicesSec,&leafStart,&leafEnd);CHKERRQ(ierr);
    ierr = DMGetWorkArray(fine,maxDof,MPIU_INT,&rowIndices);CHKERRQ(ierr);
    for (p = leafStart; p < leafEnd; p++) {
      PetscInt    gDof, gcDof, gOff;
      PetscInt    numColIndices, pIndOff, *pInd;
      PetscInt    matSize;
      PetscInt    i;

      ierr = PetscSectionGetDof(globalFine,p,&gDof);CHKERRQ(ierr);
      ierr = PetscSectionGetConstraintDof(globalFine,p,&gcDof);CHKERRQ(ierr);
      if ((gDof - gcDof) <= 0) {
        continue;
      }
      ierr = PetscSectionGetOffset(globalFine,p,&gOff);CHKERRQ(ierr);
      if (gOff < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"I though having global dofs meant a non-negative offset");
      if ((gOff < rowStart) || ((gOff + gDof - gcDof) > rowEnd)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"I thought the row map would constrain the global dofs");
      ierr = PetscSectionGetDof(leafIndicesSec,p,&numColIndices);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(leafIndicesSec,p,&pIndOff);CHKERRQ(ierr);
      numColIndices -= 2 * numFields;
      if (numColIndices <= 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"global fine dof with no dofs to interpolate from");
      pInd = &leafIndices[pIndOff];
      offsets[0]        = 0;
      offsetsCopy[0]    = 0;
      newOffsets[0]     = 0;
      newOffsetsCopy[0] = 0;
      if (numFields) {
        PetscInt f;
        for (f = 0; f < numFields; f++) {
          PetscInt rowDof;

          ierr = PetscSectionGetFieldDof(localFine,p,f,&rowDof);CHKERRQ(ierr);
          offsets[f + 1]        = offsets[f] + rowDof;
          offsetsCopy[f + 1]    = offsets[f + 1];
          newOffsets[f + 1]     = pInd[numColIndices + numFields + f];
          numD[f] = 0;
          numO[f] = 0;
        }
        ierr = DMPlexGetIndicesPointFields_Internal(localFine,PETSC_FALSE,p,gOff,offsetsCopy,PETSC_FALSE,NULL,-1, NULL,rowIndices);CHKERRQ(ierr);
        for (f = 0; f < numFields; f++) {
          PetscInt colOffset    = newOffsets[f];
          PetscInt numFieldCols = newOffsets[f + 1] - newOffsets[f];

          for (i = 0; i < numFieldCols; i++) {
            PetscInt gInd = pInd[i + colOffset];

            if (gInd >= colStart && gInd < colEnd) {
              numD[f]++;
            }
            else if (gInd >= 0) { /* negative means non-entry */
              numO[f]++;
            }
          }
        }
      }
      else {
        ierr = DMPlexGetIndicesPoint_Internal(localFine,PETSC_FALSE,p,gOff,offsetsCopy,PETSC_FALSE,NULL, NULL,rowIndices);CHKERRQ(ierr);
        numD[0] = 0;
        numO[0] = 0;
        for (i = 0; i < numColIndices; i++) {
          PetscInt gInd = pInd[i];

          if (gInd >= colStart && gInd < colEnd) {
            numD[0]++;
          }
          else if (gInd >= 0) { /* negative means non-entry */
            numO[0]++;
          }
        }
      }
      ierr = PetscSectionGetDof(leafMatricesSec,p,&matSize);CHKERRQ(ierr);
      if (!matSize) { /* incoming matrix is identity */
        PetscInt childId;

        childId = childIds[p-pStartF];
        if (childId < 0) { /* no child interpolation: one nnz per */
          if (numFields) {
            PetscInt f;
            for (f = 0; f < numFields; f++) {
              PetscInt numRows = offsets[f+1] - offsets[f], row;
              for (row = 0; row < numRows; row++) {
                PetscInt gIndCoarse = pInd[newOffsets[f] + row];
                PetscInt gIndFine   = rowIndices[offsets[f] + row];
                if (gIndCoarse >= colStart && gIndCoarse < colEnd) { /* local */
                  if (gIndFine < rowStart || gIndFine >= rowEnd) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Mismatched number of constrained dofs");
                  dnnz[gIndFine - rowStart] = 1;
                }
                else if (gIndCoarse >= 0) { /* remote */
                  if (gIndFine < rowStart || gIndFine >= rowEnd) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Mismatched number of constrained dofs");
                  onnz[gIndFine - rowStart] = 1;
                }
                else { /* constrained */
                  if (gIndFine >= 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Mismatched number of constrained dofs");
                }
              }
            }
          }
          else {
            PetscInt i;
            for (i = 0; i < gDof; i++) {
              PetscInt gIndCoarse = pInd[i];
              PetscInt gIndFine   = rowIndices[i];
              if (gIndCoarse >= colStart && gIndCoarse < colEnd) { /* local */
                if (gIndFine < rowStart || gIndFine >= rowEnd) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Mismatched number of constrained dofs");
                dnnz[gIndFine - rowStart] = 1;
              }
              else if (gIndCoarse >= 0) { /* remote */
                if (gIndFine < rowStart || gIndFine >= rowEnd) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Mismatched number of constrained dofs");
                onnz[gIndFine - rowStart] = 1;
              }
              else { /* constrained */
                if (gIndFine >= 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Mismatched number of constrained dofs");
              }
            }
          }
        }
        else { /* interpolate from all */
          if (numFields) {
            PetscInt f;
            for (f = 0; f < numFields; f++) {
              PetscInt numRows = offsets[f+1] - offsets[f], row;
              for (row = 0; row < numRows; row++) {
                PetscInt gIndFine = rowIndices[offsets[f] + row];
                if (gIndFine >= 0) {
                  if (gIndFine < rowStart || gIndFine >= rowEnd) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Mismatched number of constrained dofs");
                  dnnz[gIndFine - rowStart] = numD[f];
                  onnz[gIndFine - rowStart] = numO[f];
                }
              }
            }
          }
          else {
            PetscInt i;
            for (i = 0; i < gDof; i++) {
              PetscInt gIndFine = rowIndices[i];
              if (gIndFine >= 0) {
                if (gIndFine < rowStart || gIndFine >= rowEnd) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Mismatched number of constrained dofs");
                dnnz[gIndFine - rowStart] = numD[0];
                onnz[gIndFine - rowStart] = numO[0];
              }
            }
          }
        }
      }
      else { /* interpolate from all */
        if (numFields) {
          PetscInt f;
          for (f = 0; f < numFields; f++) {
            PetscInt numRows = offsets[f+1] - offsets[f], row;
            for (row = 0; row < numRows; row++) {
              PetscInt gIndFine = rowIndices[offsets[f] + row];
              if (gIndFine >= 0) {
                if (gIndFine < rowStart || gIndFine >= rowEnd) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Mismatched number of constrained dofs");
                dnnz[gIndFine - rowStart] = numD[f];
                onnz[gIndFine - rowStart] = numO[f];
              }
            }
          }
        }
        else { /* every dof get a full row */
          PetscInt i;
          for (i = 0; i < gDof; i++) {
            PetscInt gIndFine = rowIndices[i];
            if (gIndFine >= 0) {
              if (gIndFine < rowStart || gIndFine >= rowEnd) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Mismatched number of constrained dofs");
              dnnz[gIndFine - rowStart] = numD[0];
              onnz[gIndFine - rowStart] = numO[0];
            }
          }
        }
      }
    }
    ierr = MatXAIJSetPreallocation(mat,1,dnnz,onnz,NULL,NULL);CHKERRQ(ierr);
    ierr = PetscFree2(dnnz,onnz);CHKERRQ(ierr);

    ierr = DMPlexGetReferenceTree(fine,&refTree);CHKERRQ(ierr);
    ierr = DMPlexReferenceTreeGetChildrenMatrices(refTree,&refPointFieldMats,&refPointFieldN);CHKERRQ(ierr);
    ierr = DMGetDefaultConstraints(refTree,&refConSec,NULL);CHKERRQ(ierr);
    ierr = DMPlexGetAnchors(refTree,&refAnSec,NULL);CHKERRQ(ierr);
    ierr = PetscSectionGetChart(refConSec,&pRefStart,&pRefEnd);CHKERRQ(ierr);
    ierr = PetscSectionGetMaxDof(refConSec,&maxConDof);CHKERRQ(ierr);
    ierr = PetscSectionGetMaxDof(leafIndicesSec,&maxColumns);CHKERRQ(ierr);
    ierr = PetscMalloc1(maxConDof*maxColumns,&pointWork);CHKERRQ(ierr);
    for (p = leafStart; p < leafEnd; p++) {
      PetscInt gDof, gcDof, gOff;
      PetscInt numColIndices, pIndOff, *pInd;
      PetscInt matSize;
      PetscInt childId;

      ierr = PetscSectionGetDof(globalFine,p,&gDof);CHKERRQ(ierr);
      ierr = PetscSectionGetConstraintDof(globalFine,p,&gcDof);CHKERRQ(ierr);
      if ((gDof - gcDof) <= 0) {
        continue;
      }
      childId = childIds[p-pStartF];
      ierr = PetscSectionGetOffset(globalFine,p,&gOff);CHKERRQ(ierr);
      ierr = PetscSectionGetDof(leafIndicesSec,p,&numColIndices);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(leafIndicesSec,p,&pIndOff);CHKERRQ(ierr);
      numColIndices -= 2 * numFields;
      pInd = &leafIndices[pIndOff];
      offsets[0]        = 0;
      offsetsCopy[0]    = 0;
      newOffsets[0]     = 0;
      newOffsetsCopy[0] = 0;
      rowOffsets[0]     = 0;
      if (numFields) {
        PetscInt f;
        for (f = 0; f < numFields; f++) {
          PetscInt rowDof;

          ierr = PetscSectionGetFieldDof(localFine,p,f,&rowDof);CHKERRQ(ierr);
          offsets[f + 1]     = offsets[f] + rowDof;
          offsetsCopy[f + 1] = offsets[f + 1];
          rowOffsets[f + 1]  = pInd[numColIndices + f];
          newOffsets[f + 1]  = pInd[numColIndices + numFields + f];
        }
        ierr = DMPlexGetIndicesPointFields_Internal(localFine,PETSC_FALSE,p,gOff,offsetsCopy,PETSC_FALSE,NULL,-1, NULL,rowIndices);CHKERRQ(ierr);
      }
      else {
        ierr = DMPlexGetIndicesPoint_Internal(localFine,PETSC_FALSE,p,gOff,offsetsCopy,PETSC_FALSE,NULL, NULL,rowIndices);CHKERRQ(ierr);
      }
      ierr = PetscSectionGetDof(leafMatricesSec,p,&matSize);CHKERRQ(ierr);
      if (!matSize) { /* incoming matrix is identity */
        if (childId < 0) { /* no child interpolation: scatter */
          if (numFields) {
            PetscInt f;
            for (f = 0; f < numFields; f++) {
              PetscInt numRows = offsets[f+1] - offsets[f], row;
              for (row = 0; row < numRows; row++) {
                ierr = MatSetValue(mat,rowIndices[offsets[f]+row],pInd[newOffsets[f]+row],1.,INSERT_VALUES);CHKERRQ(ierr);
              }
            }
          }
          else {
            PetscInt numRows = gDof, row;
            for (row = 0; row < numRows; row++) {
              ierr = MatSetValue(mat,rowIndices[row],pInd[row],1.,INSERT_VALUES);CHKERRQ(ierr);
            }
          }
        }
        else { /* interpolate from all */
          if (numFields) {
            PetscInt f;
            for (f = 0; f < numFields; f++) {
              PetscInt numRows = offsets[f+1] - offsets[f];
              PetscInt numCols = newOffsets[f + 1] - newOffsets[f];
              ierr = MatSetValues(mat,numRows,&rowIndices[offsets[f]],numCols,&pInd[newOffsets[f]],refPointFieldMats[childId - pRefStart][f],INSERT_VALUES);CHKERRQ(ierr);
            }
          }
          else {
            ierr = MatSetValues(mat,gDof,rowIndices,numColIndices,pInd,refPointFieldMats[childId - pRefStart][0],INSERT_VALUES);CHKERRQ(ierr);
          }
        }
      }
      else { /* interpolate from all */
        PetscInt    pMatOff;
        PetscScalar *pMat;

        ierr = PetscSectionGetOffset(leafMatricesSec,p,&pMatOff);CHKERRQ(ierr);
        pMat = &leafMatrices[pMatOff];
        if (childId < 0) { /* copy the incoming matrix */
          if (numFields) {
            PetscInt f, count;
            for (f = 0, count = 0; f < numFields; f++) {
              PetscInt numRows = offsets[f+1]-offsets[f];
              PetscInt numCols = newOffsets[f+1]-newOffsets[f];
              PetscInt numInRows = rowOffsets[f+1]-rowOffsets[f];
              PetscScalar *inMat = &pMat[count];

              ierr = MatSetValues(mat,numRows,&rowIndices[offsets[f]],numCols,&pInd[newOffsets[f]],inMat,INSERT_VALUES);CHKERRQ(ierr);
              count += numCols * numInRows;
            }
          }
          else {
            ierr = MatSetValues(mat,gDof,rowIndices,numColIndices,pInd,pMat,INSERT_VALUES);CHKERRQ(ierr);
          }
        }
        else { /* multiply the incoming matrix by the child interpolation */
          if (numFields) {
            PetscInt f, count;
            for (f = 0, count = 0; f < numFields; f++) {
              PetscInt numRows = offsets[f+1]-offsets[f];
              PetscInt numCols = newOffsets[f+1]-newOffsets[f];
              PetscInt numInRows = rowOffsets[f+1]-rowOffsets[f];
              PetscScalar *inMat = &pMat[count];
              PetscInt i, j, k;
              if (refPointFieldN[childId - pRefStart][f] != numInRows) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Point constraint matrix multiply dimension mismatch");
              for (i = 0; i < numRows; i++) {
                for (j = 0; j < numCols; j++) {
                  PetscScalar val = 0.;
                  for (k = 0; k < numInRows; k++) {
                    val += refPointFieldMats[childId - pRefStart][f][i * numInRows + k] * inMat[k * numCols + j];
                  }
                  pointWork[i * numCols + j] = val;
                }
              }
              ierr = MatSetValues(mat,numRows,&rowIndices[offsets[f]],numCols,&pInd[newOffsets[f]],pointWork,INSERT_VALUES);CHKERRQ(ierr);
              count += numCols * numInRows;
            }
          }
          else { /* every dof gets a full row */
            PetscInt numRows   = gDof;
            PetscInt numCols   = numColIndices;
            PetscInt numInRows = matSize / numColIndices;
            PetscInt i, j, k;
            if (refPointFieldN[childId - pRefStart][0] != numInRows) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Point constraint matrix multiply dimension mismatch");
            for (i = 0; i < numRows; i++) {
              for (j = 0; j < numCols; j++) {
                PetscScalar val = 0.;
                for (k = 0; k < numInRows; k++) {
                  val += refPointFieldMats[childId - pRefStart][0][i * numInRows + k] * pMat[k * numCols + j];
                }
                pointWork[i * numCols + j] = val;
              }
            }
            ierr = MatSetValues(mat,numRows,rowIndices,numCols,pInd,pointWork,INSERT_VALUES);CHKERRQ(ierr);
          }
        }
      }
    }
    ierr = DMPlexReferenceTreeRestoreChildrenMatrices(refTree,&refPointFieldMats,&refPointFieldN);CHKERRQ(ierr);
    ierr = DMRestoreWorkArray(fine,maxDof,MPIU_INT,&rowIndices);CHKERRQ(ierr);
    ierr = PetscFree(pointWork);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&leafIndicesSec);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&leafMatricesSec);CHKERRQ(ierr);
  ierr = PetscFree2(leafIndices,leafMatrices);CHKERRQ(ierr);
  ierr = PetscFree2(*(PetscInt****)&perms,*(PetscScalar****)&flips);CHKERRQ(ierr);
  ierr = PetscFree7(offsets,offsetsCopy,newOffsets,newOffsetsCopy,rowOffsets,numD,numO);CHKERRQ(ierr);
  ierr = ISRestoreIndices(aIS,&anchors);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
 * Assuming a nodal basis (w.r.t. the dual basis) basis:
 *
 * for each coarse dof \phi^c_i:
 *   for each quadrature point (w_l,x_l) in the dual basis definition of \phi^c_i:
 *     for each fine dof \phi^f_j;
 *       a_{i,j} = 0;
 *       for each fine dof \phi^f_k:
 *         a_{i,j} += interp_{i,k} * \phi^f_k(x_l) * \phi^f_j(x_l) * w_l
 *                    [^^^ this is = \phi^c_i ^^^]
 */
PetscErrorCode DMPlexComputeInjectorReferenceTree(DM refTree, Mat *inj)
{
  PetscDS        ds;
  PetscSection   section, cSection;
  DMLabel        canonical, depth;
  Mat            cMat, mat;
  PetscInt       *nnz;
  PetscInt       f, dim, numFields, numSecFields, p, pStart, pEnd, cStart, cEnd;
  PetscInt       m, n;
  PetscScalar    *pointScalar;
  PetscReal      *v0, *v0parent, *vtmp, *J, *Jparent, *invJ, *pointRef, detJ, detJparent;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetLocalSection(refTree,&section);CHKERRQ(ierr);
  ierr = DMGetDimension(refTree, &dim);CHKERRQ(ierr);
  ierr = PetscMalloc6(dim,&v0,dim,&v0parent,dim,&vtmp,dim*dim,&J,dim*dim,&Jparent,dim*dim,&invJ);CHKERRQ(ierr);
  ierr = PetscMalloc2(dim,&pointScalar,dim,&pointRef);CHKERRQ(ierr);
  ierr = DMGetDS(refTree,&ds);CHKERRQ(ierr);
  ierr = PetscDSGetNumFields(ds,&numFields);CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(section,&numSecFields);CHKERRQ(ierr);
  ierr = DMGetLabel(refTree,"canonical",&canonical);CHKERRQ(ierr);
  ierr = DMGetLabel(refTree,"depth",&depth);CHKERRQ(ierr);
  ierr = DMGetDefaultConstraints(refTree,&cSection,&cMat);CHKERRQ(ierr);
  ierr = DMPlexGetChart(refTree, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(refTree, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = MatGetSize(cMat,&n,&m);CHKERRQ(ierr); /* the injector has transpose sizes from the constraint matrix */
  /* Step 1: compute non-zero pattern.  A proper subset of constraint matrix non-zero */
  ierr = PetscCalloc1(m,&nnz);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; p++) { /* a point will have non-zeros if it is canonical, it has dofs, and its children have dofs */
    const PetscInt *children;
    PetscInt numChildren;
    PetscInt i, numChildDof, numSelfDof;

    if (canonical) {
      PetscInt pCanonical;
      ierr = DMLabelGetValue(canonical,p,&pCanonical);CHKERRQ(ierr);
      if (p != pCanonical) continue;
    }
    ierr = DMPlexGetTreeChildren(refTree,p,&numChildren,&children);CHKERRQ(ierr);
    if (!numChildren) continue;
    for (i = 0, numChildDof = 0; i < numChildren; i++) {
      PetscInt child = children[i];
      PetscInt dof;

      ierr = PetscSectionGetDof(section,child,&dof);CHKERRQ(ierr);
      numChildDof += dof;
    }
    ierr = PetscSectionGetDof(section,p,&numSelfDof);CHKERRQ(ierr);
    if (!numChildDof || !numSelfDof) continue;
    for (f = 0; f < numFields; f++) {
      PetscInt selfOff;

      if (numSecFields) { /* count the dofs for just this field */
        for (i = 0, numChildDof = 0; i < numChildren; i++) {
          PetscInt child = children[i];
          PetscInt dof;

          ierr = PetscSectionGetFieldDof(section,child,f,&dof);CHKERRQ(ierr);
          numChildDof += dof;
        }
        ierr = PetscSectionGetFieldDof(section,p,f,&numSelfDof);CHKERRQ(ierr);
        ierr = PetscSectionGetFieldOffset(section,p,f,&selfOff);CHKERRQ(ierr);
      }
      else {
        ierr = PetscSectionGetOffset(section,p,&selfOff);CHKERRQ(ierr);
      }
      for (i = 0; i < numSelfDof; i++) {
        nnz[selfOff + i] = numChildDof;
      }
    }
  }
  ierr = MatCreateAIJ(PETSC_COMM_SELF,m,n,m,n,-1,nnz,-1,NULL,&mat);CHKERRQ(ierr);
  ierr = PetscFree(nnz);CHKERRQ(ierr);
  /* Setp 2: compute entries */
  for (p = pStart; p < pEnd; p++) {
    const PetscInt *children;
    PetscInt numChildren;
    PetscInt i, numChildDof, numSelfDof;

    /* same conditions about when entries occur */
    if (canonical) {
      PetscInt pCanonical;
      ierr = DMLabelGetValue(canonical,p,&pCanonical);CHKERRQ(ierr);
      if (p != pCanonical) continue;
    }
    ierr = DMPlexGetTreeChildren(refTree,p,&numChildren,&children);CHKERRQ(ierr);
    if (!numChildren) continue;
    for (i = 0, numChildDof = 0; i < numChildren; i++) {
      PetscInt child = children[i];
      PetscInt dof;

      ierr = PetscSectionGetDof(section,child,&dof);CHKERRQ(ierr);
      numChildDof += dof;
    }
    ierr = PetscSectionGetDof(section,p,&numSelfDof);CHKERRQ(ierr);
    if (!numChildDof || !numSelfDof) continue;

    for (f = 0; f < numFields; f++) {
      PetscInt       pI = -1, cI = -1;
      PetscInt       selfOff, Nc, parentCell;
      PetscInt       cellShapeOff;
      PetscObject    disc;
      PetscDualSpace dsp;
      PetscClassId   classId;
      PetscScalar    *pointMat;
      PetscInt       *matRows, *matCols;
      PetscInt       pO = PETSC_MIN_INT;
      const PetscInt *depthNumDof;

      if (numSecFields) {
        for (i = 0, numChildDof = 0; i < numChildren; i++) {
          PetscInt child = children[i];
          PetscInt dof;

          ierr = PetscSectionGetFieldDof(section,child,f,&dof);CHKERRQ(ierr);
          numChildDof += dof;
        }
        ierr = PetscSectionGetFieldDof(section,p,f,&numSelfDof);CHKERRQ(ierr);
        ierr = PetscSectionGetFieldOffset(section,p,f,&selfOff);CHKERRQ(ierr);
      }
      else {
        ierr = PetscSectionGetOffset(section,p,&selfOff);CHKERRQ(ierr);
      }

      /* find a cell whose closure contains p */
      if (p >= cStart && p < cEnd) {
        parentCell = p;
      }
      else {
        PetscInt *star = NULL;
        PetscInt numStar;

        parentCell = -1;
        ierr = DMPlexGetTransitiveClosure(refTree,p,PETSC_FALSE,&numStar,&star);CHKERRQ(ierr);
        for (i = numStar - 1; i >= 0; i--) {
          PetscInt c = star[2 * i];

          if (c >= cStart && c < cEnd) {
            parentCell = c;
            break;
          }
        }
        ierr = DMPlexRestoreTransitiveClosure(refTree,p,PETSC_FALSE,&numStar,&star);CHKERRQ(ierr);
      }
      /* determine the offset of p's shape functions withing parentCell's shape functions */
      ierr = PetscDSGetDiscretization(ds,f,&disc);CHKERRQ(ierr);
      ierr = PetscObjectGetClassId(disc,&classId);CHKERRQ(ierr);
      if (classId == PETSCFE_CLASSID) {
        ierr = PetscFEGetDualSpace((PetscFE)disc,&dsp);CHKERRQ(ierr);
      }
      else if (classId == PETSCFV_CLASSID) {
        ierr = PetscFVGetDualSpace((PetscFV)disc,&dsp);CHKERRQ(ierr);
      }
      else {
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unsupported discretization object");
      }
      ierr = PetscDualSpaceGetNumDof(dsp,&depthNumDof);CHKERRQ(ierr);
      ierr = PetscDualSpaceGetNumComponents(dsp,&Nc);CHKERRQ(ierr);
      {
        PetscInt *closure = NULL;
        PetscInt numClosure;

        ierr = DMPlexGetTransitiveClosure(refTree,parentCell,PETSC_TRUE,&numClosure,&closure);CHKERRQ(ierr);
        for (i = 0, pI = -1, cellShapeOff = 0; i < numClosure; i++) {
          PetscInt point = closure[2 * i], pointDepth;

          pO = closure[2 * i + 1];
          if (point == p) {
            pI = i;
            break;
          }
          ierr = DMLabelGetValue(depth,point,&pointDepth);CHKERRQ(ierr);
          cellShapeOff += depthNumDof[pointDepth];
        }
        ierr = DMPlexRestoreTransitiveClosure(refTree,parentCell,PETSC_TRUE,&numClosure,&closure);CHKERRQ(ierr);
      }

      ierr = DMGetWorkArray(refTree, numSelfDof * numChildDof, MPIU_SCALAR,&pointMat);CHKERRQ(ierr);
      ierr = DMGetWorkArray(refTree, numSelfDof + numChildDof, MPIU_INT,&matRows);CHKERRQ(ierr);
      matCols = matRows + numSelfDof;
      for (i = 0; i < numSelfDof; i++) {
        matRows[i] = selfOff + i;
      }
      for (i = 0; i < numSelfDof * numChildDof; i++) pointMat[i] = 0.;
      {
        PetscInt colOff = 0;

        for (i = 0; i < numChildren; i++) {
          PetscInt child = children[i];
          PetscInt dof, off, j;

          if (numSecFields) {
            ierr = PetscSectionGetFieldDof(cSection,child,f,&dof);CHKERRQ(ierr);
            ierr = PetscSectionGetFieldOffset(cSection,child,f,&off);CHKERRQ(ierr);
          }
          else {
            ierr = PetscSectionGetDof(cSection,child,&dof);CHKERRQ(ierr);
            ierr = PetscSectionGetOffset(cSection,child,&off);CHKERRQ(ierr);
          }

          for (j = 0; j < dof; j++) {
            matCols[colOff++] = off + j;
          }
        }
      }
      if (classId == PETSCFE_CLASSID) {
        PetscFE        fe = (PetscFE) disc;
        PetscInt       fSize;
        const PetscInt ***perms;
        const PetscScalar ***flips;
        const PetscInt *pperms;


        ierr = PetscFEGetDualSpace(fe,&dsp);CHKERRQ(ierr);
        ierr = PetscDualSpaceGetDimension(dsp,&fSize);CHKERRQ(ierr);
        ierr = PetscDualSpaceGetSymmetries(dsp, &perms, &flips);CHKERRQ(ierr);
        pperms = perms ? perms[pI] ? perms[pI][pO] : NULL : NULL;
        for (i = 0; i < numSelfDof; i++) { /* for every shape function */
          PetscQuadrature q;
          PetscInt        dim, thisNc, numPoints, j, k;
          const PetscReal *points;
          const PetscReal *weights;
          PetscInt        *closure = NULL;
          PetscInt        numClosure;
          PetscInt        iCell = pperms ? pperms[i] : i;
          PetscInt        parentCellShapeDof = cellShapeOff + iCell;
          PetscTabulation Tparent;

          ierr = PetscDualSpaceGetFunctional(dsp,parentCellShapeDof,&q);CHKERRQ(ierr);
          ierr = PetscQuadratureGetData(q,&dim,&thisNc,&numPoints,&points,&weights);CHKERRQ(ierr);
          if (thisNc != Nc) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Functional dim %D does not much basis dim %D\n",thisNc,Nc);
          ierr = PetscFECreateTabulation(fe,1,numPoints,points,0,&Tparent);CHKERRQ(ierr); /* I'm expecting a nodal basis: weights[:]' * Bparent[:,cellShapeDof] = 1. */
          for (j = 0; j < numPoints; j++) {
            PetscInt          childCell = -1;
            PetscReal         *parentValAtPoint;
            const PetscReal   xi0[3] = {-1.,-1.,-1.};
            const PetscReal   *pointReal = &points[dim * j];
            const PetscScalar *point;
            PetscTabulation Tchild;
            PetscInt          childCellShapeOff, pointMatOff;
#if defined(PETSC_USE_COMPLEX)
            PetscInt          d;

            for (d = 0; d < dim; d++) {
              pointScalar[d] = points[dim * j + d];
            }
            point = pointScalar;
#else
            point = pointReal;
#endif

            parentValAtPoint = &Tparent->T[0][(fSize * j + parentCellShapeDof) * Nc];

            for (k = 0; k < numChildren; k++) { /* locate the point in a child's star cell*/
              PetscInt child = children[k];
              PetscInt *star = NULL;
              PetscInt numStar, s;

              ierr = DMPlexGetTransitiveClosure(refTree,child,PETSC_FALSE,&numStar,&star);CHKERRQ(ierr);
              for (s = numStar - 1; s >= 0; s--) {
                PetscInt c = star[2 * s];

                if (c < cStart || c >= cEnd) continue;
                ierr = DMPlexLocatePoint_Internal(refTree,dim,point,c,&childCell);CHKERRQ(ierr);
                if (childCell >= 0) break;
              }
              ierr = DMPlexRestoreTransitiveClosure(refTree,child,PETSC_FALSE,&numStar,&star);CHKERRQ(ierr);
              if (childCell >= 0) break;
            }
            if (childCell < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Could not locate quadrature point");
            ierr = DMPlexComputeCellGeometryFEM(refTree, childCell, NULL, v0, J, invJ, &detJ);CHKERRQ(ierr);
            ierr = DMPlexComputeCellGeometryFEM(refTree, parentCell, NULL, v0parent, Jparent, NULL, &detJparent);CHKERRQ(ierr);
            CoordinatesRefToReal(dim, dim, xi0, v0parent, Jparent, pointReal, vtmp);
            CoordinatesRealToRef(dim, dim, xi0, v0, invJ, vtmp, pointRef);

            ierr = PetscFECreateTabulation(fe,1,1,pointRef,0,&Tchild);CHKERRQ(ierr);
            ierr = DMPlexGetTransitiveClosure(refTree,childCell,PETSC_TRUE,&numClosure,&closure);CHKERRQ(ierr);
            for (k = 0, pointMatOff = 0; k < numChildren; k++) { /* point is located in cell => child dofs support at point are in closure of cell */
              PetscInt child = children[k], childDepth, childDof, childO = PETSC_MIN_INT;
              PetscInt l;
              const PetscInt *cperms;

              ierr = DMLabelGetValue(depth,child,&childDepth);CHKERRQ(ierr);
              childDof = depthNumDof[childDepth];
              for (l = 0, cI = -1, childCellShapeOff = 0; l < numClosure; l++) {
                PetscInt point = closure[2 * l];
                PetscInt pointDepth;

                childO = closure[2 * l + 1];
                if (point == child) {
                  cI = l;
                  break;
                }
                ierr = DMLabelGetValue(depth,point,&pointDepth);CHKERRQ(ierr);
                childCellShapeOff += depthNumDof[pointDepth];
              }
              if (l == numClosure) {
                pointMatOff += childDof;
                continue; /* child is not in the closure of the cell: has nothing to contribute to this point */
              }
              cperms = perms ? perms[cI] ? perms[cI][childO] : NULL : NULL;
              for (l = 0; l < childDof; l++) {
                PetscInt    lCell = cperms ? cperms[l] : l;
                PetscInt    childCellDof = childCellShapeOff + lCell;
                PetscReal   *childValAtPoint;
                PetscReal   val = 0.;

                childValAtPoint = &Tchild->T[0][childCellDof * Nc];
                for (m = 0; m < Nc; m++) {
                  val += weights[j * Nc + m] * parentValAtPoint[m] * childValAtPoint[m];
                }

                pointMat[i * numChildDof + pointMatOff + l] += val;
              }
              pointMatOff += childDof;
            }
            ierr = DMPlexRestoreTransitiveClosure(refTree,childCell,PETSC_TRUE,&numClosure,&closure);CHKERRQ(ierr);
            ierr = PetscTabulationDestroy(&Tchild);CHKERRQ(ierr);
          }
          ierr = PetscTabulationDestroy(&Tparent);CHKERRQ(ierr);
        }
      }
      else { /* just the volume-weighted averages of the children */
        PetscReal parentVol;
        PetscInt  childCell;

        ierr = DMPlexComputeCellGeometryFVM(refTree, p, &parentVol, NULL, NULL);CHKERRQ(ierr);
        for (i = 0, childCell = 0; i < numChildren; i++) {
          PetscInt  child = children[i], j;
          PetscReal childVol;

          if (child < cStart || child >= cEnd) continue;
          ierr = DMPlexComputeCellGeometryFVM(refTree, child, &childVol, NULL, NULL);CHKERRQ(ierr);
          for (j = 0; j < Nc; j++) {
            pointMat[j * numChildDof + Nc * childCell + j] = childVol / parentVol;
          }
          childCell++;
        }
      }
      /* Insert pointMat into mat */
      ierr = MatSetValues(mat,numSelfDof,matRows,numChildDof,matCols,pointMat,INSERT_VALUES);CHKERRQ(ierr);
      ierr = DMRestoreWorkArray(refTree, numSelfDof + numChildDof, MPIU_INT,&matRows);CHKERRQ(ierr);
      ierr = DMRestoreWorkArray(refTree, numSelfDof * numChildDof, MPIU_SCALAR,&pointMat);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree6(v0,v0parent,vtmp,J,Jparent,invJ);CHKERRQ(ierr);
  ierr = PetscFree2(pointScalar,pointRef);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  *inj = mat;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexReferenceTreeGetChildrenMatrices_Injection(DM refTree, Mat inj, PetscScalar ****childrenMats)
{
  PetscDS        ds;
  PetscInt       numFields, f, pRefStart, pRefEnd, p, *rows, *cols, maxDof;
  PetscScalar    ***refPointFieldMats;
  PetscSection   refConSec, refSection;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetDS(refTree,&ds);CHKERRQ(ierr);
  ierr = PetscDSGetNumFields(ds,&numFields);CHKERRQ(ierr);
  ierr = DMGetDefaultConstraints(refTree,&refConSec,NULL);CHKERRQ(ierr);
  ierr = DMGetLocalSection(refTree,&refSection);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(refConSec,&pRefStart,&pRefEnd);CHKERRQ(ierr);
  ierr = PetscMalloc1(pRefEnd-pRefStart,&refPointFieldMats);CHKERRQ(ierr);
  ierr = PetscSectionGetMaxDof(refConSec,&maxDof);CHKERRQ(ierr);
  ierr = PetscMalloc1(maxDof,&rows);CHKERRQ(ierr);
  ierr = PetscMalloc1(maxDof*maxDof,&cols);CHKERRQ(ierr);
  for (p = pRefStart; p < pRefEnd; p++) {
    PetscInt parent, pDof, parentDof;

    ierr = DMPlexGetTreeParent(refTree,p,&parent,NULL);CHKERRQ(ierr);
    ierr = PetscSectionGetDof(refConSec,p,&pDof);CHKERRQ(ierr);
    ierr = PetscSectionGetDof(refSection,parent,&parentDof);CHKERRQ(ierr);
    if (!pDof || !parentDof || parent == p) continue;

    ierr = PetscMalloc1(numFields,&refPointFieldMats[p-pRefStart]);CHKERRQ(ierr);
    for (f = 0; f < numFields; f++) {
      PetscInt cDof, cOff, numCols, r;

      if (numFields > 1) {
        ierr = PetscSectionGetFieldDof(refConSec,p,f,&cDof);CHKERRQ(ierr);
        ierr = PetscSectionGetFieldOffset(refConSec,p,f,&cOff);CHKERRQ(ierr);
      }
      else {
        ierr = PetscSectionGetDof(refConSec,p,&cDof);CHKERRQ(ierr);
        ierr = PetscSectionGetOffset(refConSec,p,&cOff);CHKERRQ(ierr);
      }

      for (r = 0; r < cDof; r++) {
        rows[r] = cOff + r;
      }
      numCols = 0;
      {
        PetscInt aDof, aOff, j;

        if (numFields > 1) {
          ierr = PetscSectionGetFieldDof(refSection,parent,f,&aDof);CHKERRQ(ierr);
          ierr = PetscSectionGetFieldOffset(refSection,parent,f,&aOff);CHKERRQ(ierr);
        }
        else {
          ierr = PetscSectionGetDof(refSection,parent,&aDof);CHKERRQ(ierr);
          ierr = PetscSectionGetOffset(refSection,parent,&aOff);CHKERRQ(ierr);
        }

        for (j = 0; j < aDof; j++) {
          cols[numCols++] = aOff + j;
        }
      }
      ierr = PetscMalloc1(cDof*numCols,&refPointFieldMats[p-pRefStart][f]);CHKERRQ(ierr);
      /* transpose of constraint matrix */
      ierr = MatGetValues(inj,numCols,cols,cDof,rows,refPointFieldMats[p-pRefStart][f]);CHKERRQ(ierr);
    }
  }
  *childrenMats = refPointFieldMats;
  ierr = PetscFree(rows);CHKERRQ(ierr);
  ierr = PetscFree(cols);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexReferenceTreeRestoreChildrenMatrices_Injection(DM refTree, Mat inj, PetscScalar ****childrenMats)
{
  PetscDS        ds;
  PetscScalar    ***refPointFieldMats;
  PetscInt       numFields, pRefStart, pRefEnd, p, f;
  PetscSection   refConSec, refSection;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  refPointFieldMats = *childrenMats;
  *childrenMats = NULL;
  ierr = DMGetDS(refTree,&ds);CHKERRQ(ierr);
  ierr = DMGetLocalSection(refTree,&refSection);CHKERRQ(ierr);
  ierr = PetscDSGetNumFields(ds,&numFields);CHKERRQ(ierr);
  ierr = DMGetDefaultConstraints(refTree,&refConSec,NULL);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(refConSec,&pRefStart,&pRefEnd);CHKERRQ(ierr);
  for (p = pRefStart; p < pRefEnd; p++) {
    PetscInt parent, pDof, parentDof;

    ierr = DMPlexGetTreeParent(refTree,p,&parent,NULL);CHKERRQ(ierr);
    ierr = PetscSectionGetDof(refConSec,p,&pDof);CHKERRQ(ierr);
    ierr = PetscSectionGetDof(refSection,parent,&parentDof);CHKERRQ(ierr);
    if (!pDof || !parentDof || parent == p) continue;

    for (f = 0; f < numFields; f++) {
      PetscInt cDof;

      if (numFields > 1) {
        ierr = PetscSectionGetFieldDof(refConSec,p,f,&cDof);CHKERRQ(ierr);
      }
      else {
        ierr = PetscSectionGetDof(refConSec,p,&cDof);CHKERRQ(ierr);
      }

      ierr = PetscFree(refPointFieldMats[p - pRefStart][f]);CHKERRQ(ierr);
    }
    ierr = PetscFree(refPointFieldMats[p - pRefStart]);CHKERRQ(ierr);
  }
  ierr = PetscFree(refPointFieldMats);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexReferenceTreeGetInjector(DM refTree,Mat *injRef)
{
  Mat            cMatRef;
  PetscObject    injRefObj;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetDefaultConstraints(refTree,NULL,&cMatRef);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject)cMatRef,"DMPlexComputeInjectorTree_refTree",&injRefObj);CHKERRQ(ierr);
  *injRef = (Mat) injRefObj;
  if (!*injRef) {
    ierr = DMPlexComputeInjectorReferenceTree(refTree,injRef);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject)cMatRef,"DMPlexComputeInjectorTree_refTree",(PetscObject)*injRef);CHKERRQ(ierr);
    /* there is now a reference in cMatRef, which should be the only one for symmetry with the above case */
    ierr = PetscObjectDereference((PetscObject)*injRef);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTransferInjectorTree(DM coarse, DM fine, PetscSF coarseToFine, const PetscInt *childIds, Vec fineVec, PetscInt numFields, PetscInt *offsets, PetscSection *rootMultiSec, PetscSection *multiLeafSec, PetscInt **gatheredIndices, PetscScalar **gatheredValues)
{
  PetscInt       pStartF, pEndF, pStartC, pEndC, p, maxDof, numMulti;
  PetscSection   globalCoarse, globalFine;
  PetscSection   localCoarse, localFine, leafIndicesSec;
  PetscSection   multiRootSec, rootIndicesSec;
  PetscInt       *leafInds, *rootInds = NULL;
  const PetscInt *rootDegrees;
  PetscScalar    *leafVals = NULL, *rootVals = NULL;
  PetscSF        coarseToFineEmbedded;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetChart(coarse,&pStartC,&pEndC);CHKERRQ(ierr);
  ierr = DMPlexGetChart(fine,&pStartF,&pEndF);CHKERRQ(ierr);
  ierr = DMGetLocalSection(fine,&localFine);CHKERRQ(ierr);
  ierr = DMGetGlobalSection(fine,&globalFine);CHKERRQ(ierr);
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject)fine),&leafIndicesSec);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(leafIndicesSec,pStartF, pEndF);CHKERRQ(ierr);
  ierr = PetscSectionGetMaxDof(localFine,&maxDof);CHKERRQ(ierr);
  { /* winnow fine points that don't have global dofs out of the sf */
    PetscInt l, nleaves, dof, cdof, numPointsWithDofs, offset, *pointsWithDofs, numIndices;
    const PetscInt *leaves;

    ierr = PetscSFGetGraph(coarseToFine,NULL,&nleaves,&leaves,NULL);CHKERRQ(ierr);
    for (l = 0, numPointsWithDofs = 0; l < nleaves; l++) {
      p    = leaves ? leaves[l] : l;
      ierr = PetscSectionGetDof(globalFine,p,&dof);CHKERRQ(ierr);
      ierr = PetscSectionGetConstraintDof(globalFine,p,&cdof);CHKERRQ(ierr);
      if ((dof - cdof) > 0) {
        numPointsWithDofs++;

        ierr = PetscSectionGetDof(localFine,p,&dof);CHKERRQ(ierr);
        ierr = PetscSectionSetDof(leafIndicesSec,p,dof + 1);CHKERRQ(ierr);
      }
    }
    ierr = PetscMalloc1(numPointsWithDofs,&pointsWithDofs);CHKERRQ(ierr);
    ierr = PetscSectionSetUp(leafIndicesSec);CHKERRQ(ierr);
    ierr = PetscSectionGetStorageSize(leafIndicesSec,&numIndices);CHKERRQ(ierr);
    ierr = PetscMalloc1(gatheredIndices ? numIndices : (maxDof + 1),&leafInds);CHKERRQ(ierr);
    if (gatheredValues)  {ierr = PetscMalloc1(numIndices,&leafVals);CHKERRQ(ierr);}
    for (l = 0, offset = 0; l < nleaves; l++) {
      p    = leaves ? leaves[l] : l;
      ierr = PetscSectionGetDof(globalFine,p,&dof);CHKERRQ(ierr);
      ierr = PetscSectionGetConstraintDof(globalFine,p,&cdof);CHKERRQ(ierr);
      if ((dof - cdof) > 0) {
        PetscInt    off, gOff;
        PetscInt    *pInd;
        PetscScalar *pVal = NULL;

        pointsWithDofs[offset++] = l;

        ierr = PetscSectionGetOffset(leafIndicesSec,p,&off);CHKERRQ(ierr);

        pInd = gatheredIndices ? (&leafInds[off + 1]) : leafInds;
        if (gatheredValues) {
          PetscInt i;

          pVal = &leafVals[off + 1];
          for (i = 0; i < dof; i++) pVal[i] = 0.;
        }
        ierr = PetscSectionGetOffset(globalFine,p,&gOff);CHKERRQ(ierr);

        offsets[0] = 0;
        if (numFields) {
          PetscInt f;

          for (f = 0; f < numFields; f++) {
            PetscInt fDof;
            ierr = PetscSectionGetFieldDof(localFine,p,f,&fDof);CHKERRQ(ierr);
            offsets[f + 1] = fDof + offsets[f];
          }
          ierr = DMPlexGetIndicesPointFields_Internal(localFine,PETSC_FALSE,p,gOff < 0 ? -(gOff + 1) : gOff,offsets,PETSC_FALSE,NULL,-1, NULL,pInd);CHKERRQ(ierr);
        } else {
          ierr = DMPlexGetIndicesPoint_Internal(localFine,PETSC_FALSE,p,gOff < 0 ? -(gOff + 1) : gOff,offsets,PETSC_FALSE,NULL, NULL,pInd);CHKERRQ(ierr);
        }
        if (gatheredValues) {ierr = VecGetValues(fineVec,dof,pInd,pVal);CHKERRQ(ierr);}
      }
    }
    ierr = PetscSFCreateEmbeddedLeafSF(coarseToFine, numPointsWithDofs, pointsWithDofs, &coarseToFineEmbedded);CHKERRQ(ierr);
    ierr = PetscFree(pointsWithDofs);CHKERRQ(ierr);
  }

  ierr = DMPlexGetChart(coarse,&pStartC,&pEndC);CHKERRQ(ierr);
  ierr = DMGetLocalSection(coarse,&localCoarse);CHKERRQ(ierr);
  ierr = DMGetGlobalSection(coarse,&globalCoarse);CHKERRQ(ierr);

  { /* there may be the case where an sf root has a parent: broadcast parents back to children */
    MPI_Datatype threeInt;
    PetscMPIInt  rank;
    PetscInt     (*parentNodeAndIdCoarse)[3];
    PetscInt     (*parentNodeAndIdFine)[3];
    PetscInt     p, nleaves, nleavesToParents;
    PetscSF      pointSF, sfToParents;
    const PetscInt *ilocal;
    const PetscSFNode *iremote;
    PetscSFNode  *iremoteToParents;
    PetscInt     *ilocalToParents;

    ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)coarse),&rank);CHKERRQ(ierr);
    ierr = MPI_Type_contiguous(3,MPIU_INT,&threeInt);CHKERRQ(ierr);
    ierr = MPI_Type_commit(&threeInt);CHKERRQ(ierr);
    ierr = PetscMalloc2(pEndC-pStartC,&parentNodeAndIdCoarse,pEndF-pStartF,&parentNodeAndIdFine);CHKERRQ(ierr);
    ierr = DMGetPointSF(coarse,&pointSF);CHKERRQ(ierr);
    ierr = PetscSFGetGraph(pointSF,NULL,&nleaves,&ilocal,&iremote);CHKERRQ(ierr);
    for (p = pStartC; p < pEndC; p++) {
      PetscInt parent, childId;
      ierr = DMPlexGetTreeParent(coarse,p,&parent,&childId);CHKERRQ(ierr);
      parentNodeAndIdCoarse[p - pStartC][0] = rank;
      parentNodeAndIdCoarse[p - pStartC][1] = parent - pStartC;
      parentNodeAndIdCoarse[p - pStartC][2] = (p == parent) ? -1 : childId;
      if (nleaves > 0) {
        PetscInt leaf = -1;

        if (ilocal) {
          ierr  = PetscFindInt(parent,nleaves,ilocal,&leaf);CHKERRQ(ierr);
        }
        else {
          leaf = p - pStartC;
        }
        if (leaf >= 0) {
          parentNodeAndIdCoarse[p - pStartC][0] = iremote[leaf].rank;
          parentNodeAndIdCoarse[p - pStartC][1] = iremote[leaf].index;
        }
      }
    }
    for (p = pStartF; p < pEndF; p++) {
      parentNodeAndIdFine[p - pStartF][0] = -1;
      parentNodeAndIdFine[p - pStartF][1] = -1;
      parentNodeAndIdFine[p - pStartF][2] = -1;
    }
    ierr = PetscSFBcastBegin(coarseToFineEmbedded,threeInt,parentNodeAndIdCoarse,parentNodeAndIdFine);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(coarseToFineEmbedded,threeInt,parentNodeAndIdCoarse,parentNodeAndIdFine);CHKERRQ(ierr);
    for (p = pStartF, nleavesToParents = 0; p < pEndF; p++) {
      PetscInt dof;

      ierr = PetscSectionGetDof(leafIndicesSec,p,&dof);CHKERRQ(ierr);
      if (dof) {
        PetscInt off;

        ierr = PetscSectionGetOffset(leafIndicesSec,p,&off);CHKERRQ(ierr);
        if (gatheredIndices) {
          leafInds[off] = PetscMax(childIds[p-pStartF],parentNodeAndIdFine[p-pStartF][2]);
        } else if (gatheredValues) {
          leafVals[off] = (PetscScalar) PetscMax(childIds[p-pStartF],parentNodeAndIdFine[p-pStartF][2]);
        }
      }
      if (parentNodeAndIdFine[p-pStartF][0] >= 0) {
        nleavesToParents++;
      }
    }
    ierr = PetscMalloc1(nleavesToParents,&ilocalToParents);CHKERRQ(ierr);
    ierr = PetscMalloc1(nleavesToParents,&iremoteToParents);CHKERRQ(ierr);
    for (p = pStartF, nleavesToParents = 0; p < pEndF; p++) {
      if (parentNodeAndIdFine[p-pStartF][0] >= 0) {
        ilocalToParents[nleavesToParents] = p - pStartF;
        iremoteToParents[nleavesToParents].rank  = parentNodeAndIdFine[p-pStartF][0];
        iremoteToParents[nleavesToParents].index = parentNodeAndIdFine[p-pStartF][1];
        nleavesToParents++;
      }
    }
    ierr = PetscSFCreate(PetscObjectComm((PetscObject)coarse),&sfToParents);CHKERRQ(ierr);
    ierr = PetscSFSetGraph(sfToParents,pEndC-pStartC,nleavesToParents,ilocalToParents,PETSC_OWN_POINTER,iremoteToParents,PETSC_OWN_POINTER);CHKERRQ(ierr);
    ierr = PetscSFDestroy(&coarseToFineEmbedded);CHKERRQ(ierr);

    coarseToFineEmbedded = sfToParents;

    ierr = PetscFree2(parentNodeAndIdCoarse,parentNodeAndIdFine);CHKERRQ(ierr);
    ierr = MPI_Type_free(&threeInt);CHKERRQ(ierr);
  }

  { /* winnow out coarse points that don't have dofs */
    PetscInt dof, cdof, numPointsWithDofs, offset, *pointsWithDofs;
    PetscSF  sfDofsOnly;

    for (p = pStartC, numPointsWithDofs = 0; p < pEndC; p++) {
      ierr = PetscSectionGetDof(globalCoarse,p,&dof);CHKERRQ(ierr);
      ierr = PetscSectionGetConstraintDof(globalCoarse,p,&cdof);CHKERRQ(ierr);
      if ((dof - cdof) > 0) {
        numPointsWithDofs++;
      }
    }
    ierr = PetscMalloc1(numPointsWithDofs,&pointsWithDofs);CHKERRQ(ierr);
    for (p = pStartC, offset = 0; p < pEndC; p++) {
      ierr = PetscSectionGetDof(globalCoarse,p,&dof);CHKERRQ(ierr);
      ierr = PetscSectionGetConstraintDof(globalCoarse,p,&cdof);CHKERRQ(ierr);
      if ((dof - cdof) > 0) {
        pointsWithDofs[offset++] = p - pStartC;
      }
    }
    ierr = PetscSFCreateEmbeddedSF(coarseToFineEmbedded, numPointsWithDofs, pointsWithDofs, &sfDofsOnly);CHKERRQ(ierr);
    ierr = PetscSFDestroy(&coarseToFineEmbedded);CHKERRQ(ierr);
    ierr = PetscFree(pointsWithDofs);CHKERRQ(ierr);
    coarseToFineEmbedded = sfDofsOnly;
  }

  /* communicate back to the coarse mesh which coarse points have children (that may require injection) */
  ierr = PetscSFComputeDegreeBegin(coarseToFineEmbedded,&rootDegrees);CHKERRQ(ierr);
  ierr = PetscSFComputeDegreeEnd(coarseToFineEmbedded,&rootDegrees);CHKERRQ(ierr);
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject)coarse),&multiRootSec);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(multiRootSec,pStartC,pEndC);CHKERRQ(ierr);
  for (p = pStartC; p < pEndC; p++) {
    ierr = PetscSectionSetDof(multiRootSec,p,rootDegrees[p-pStartC]);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(multiRootSec);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(multiRootSec,&numMulti);CHKERRQ(ierr);
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject)coarse),&rootIndicesSec);CHKERRQ(ierr);
  { /* distribute the leaf section */
    PetscSF multi, multiInv, indicesSF;
    PetscInt *remoteOffsets, numRootIndices;

    ierr = PetscSFGetMultiSF(coarseToFineEmbedded,&multi);CHKERRQ(ierr);
    ierr = PetscSFCreateInverseSF(multi,&multiInv);CHKERRQ(ierr);
    ierr = PetscSFDistributeSection(multiInv,leafIndicesSec,&remoteOffsets,rootIndicesSec);CHKERRQ(ierr);
    ierr = PetscSFCreateSectionSF(multiInv,leafIndicesSec,remoteOffsets,rootIndicesSec,&indicesSF);CHKERRQ(ierr);
    ierr = PetscFree(remoteOffsets);CHKERRQ(ierr);
    ierr = PetscSFDestroy(&multiInv);CHKERRQ(ierr);
    ierr = PetscSectionGetStorageSize(rootIndicesSec,&numRootIndices);CHKERRQ(ierr);
    if (gatheredIndices) {
      ierr = PetscMalloc1(numRootIndices,&rootInds);CHKERRQ(ierr);
      ierr = PetscSFBcastBegin(indicesSF,MPIU_INT,leafInds,rootInds);CHKERRQ(ierr);
      ierr = PetscSFBcastEnd(indicesSF,MPIU_INT,leafInds,rootInds);CHKERRQ(ierr);
    }
    if (gatheredValues) {
      ierr = PetscMalloc1(numRootIndices,&rootVals);CHKERRQ(ierr);
      ierr = PetscSFBcastBegin(indicesSF,MPIU_SCALAR,leafVals,rootVals);CHKERRQ(ierr);
      ierr = PetscSFBcastEnd(indicesSF,MPIU_SCALAR,leafVals,rootVals);CHKERRQ(ierr);
    }
    ierr = PetscSFDestroy(&indicesSF);CHKERRQ(ierr);
  }
  ierr = PetscSectionDestroy(&leafIndicesSec);CHKERRQ(ierr);
  ierr = PetscFree(leafInds);CHKERRQ(ierr);
  ierr = PetscFree(leafVals);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&coarseToFineEmbedded);CHKERRQ(ierr);
  *rootMultiSec = multiRootSec;
  *multiLeafSec = rootIndicesSec;
  if (gatheredIndices) *gatheredIndices = rootInds;
  if (gatheredValues)  *gatheredValues  = rootVals;
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexComputeInjectorTree(DM coarse, DM fine, PetscSF coarseToFine, PetscInt *childIds, Mat mat)
{
  DM             refTree;
  PetscSection   multiRootSec, rootIndicesSec;
  PetscSection   globalCoarse, globalFine;
  PetscSection   localCoarse, localFine;
  PetscSection   cSecRef;
  PetscInt       *rootIndices = NULL, *parentIndices, pRefStart, pRefEnd;
  Mat            injRef;
  PetscInt       numFields, maxDof;
  PetscInt       pStartC, pEndC, pStartF, pEndF, p;
  PetscInt       *offsets, *offsetsCopy, *rowOffsets;
  PetscLayout    rowMap, colMap;
  PetscInt       rowStart, rowEnd, colStart, colEnd, *nnzD, *nnzO;
  PetscScalar    ***childrenMats=NULL ; /* gcc -O gives 'may be used uninitialized' warning'. Initializing to suppress this warning */
  PetscErrorCode ierr;

  PetscFunctionBegin;

  /* get the templates for the fine-to-coarse injection from the reference tree */
  ierr = DMPlexGetReferenceTree(coarse,&refTree);CHKERRQ(ierr);
  ierr = DMGetDefaultConstraints(refTree,&cSecRef,NULL);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(cSecRef,&pRefStart,&pRefEnd);CHKERRQ(ierr);
  ierr = DMPlexReferenceTreeGetInjector(refTree,&injRef);CHKERRQ(ierr);

  ierr = DMPlexGetChart(fine,&pStartF,&pEndF);CHKERRQ(ierr);
  ierr = DMGetLocalSection(fine,&localFine);CHKERRQ(ierr);
  ierr = DMGetGlobalSection(fine,&globalFine);CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(localFine,&numFields);CHKERRQ(ierr);
  ierr = DMPlexGetChart(coarse,&pStartC,&pEndC);CHKERRQ(ierr);
  ierr = DMGetLocalSection(coarse,&localCoarse);CHKERRQ(ierr);
  ierr = DMGetGlobalSection(coarse,&globalCoarse);CHKERRQ(ierr);
  ierr = PetscSectionGetMaxDof(localCoarse,&maxDof);CHKERRQ(ierr);
  {
    PetscInt maxFields = PetscMax(1,numFields) + 1;
    ierr = PetscMalloc3(maxFields,&offsets,maxFields,&offsetsCopy,maxFields,&rowOffsets);CHKERRQ(ierr);
  }

  ierr = DMPlexTransferInjectorTree(coarse,fine,coarseToFine,childIds,NULL,numFields,offsets,&multiRootSec,&rootIndicesSec,&rootIndices,NULL);CHKERRQ(ierr);

  ierr = PetscMalloc1(maxDof,&parentIndices);CHKERRQ(ierr);

  /* count indices */
  ierr = MatGetLayouts(mat,&rowMap,&colMap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(rowMap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(colMap);CHKERRQ(ierr);
  ierr = PetscLayoutGetRange(rowMap,&rowStart,&rowEnd);CHKERRQ(ierr);
  ierr = PetscLayoutGetRange(colMap,&colStart,&colEnd);CHKERRQ(ierr);
  ierr = PetscCalloc2(rowEnd-rowStart,&nnzD,rowEnd-rowStart,&nnzO);CHKERRQ(ierr);
  for (p = pStartC; p < pEndC; p++) {
    PetscInt numLeaves, leafStart, leafEnd, l, dof, cdof, gOff;

    ierr = PetscSectionGetDof(globalCoarse,p,&dof);CHKERRQ(ierr);
    ierr = PetscSectionGetConstraintDof(globalCoarse,p,&cdof);CHKERRQ(ierr);
    if ((dof - cdof) <= 0) continue;
    ierr = PetscSectionGetOffset(globalCoarse,p,&gOff);CHKERRQ(ierr);

    rowOffsets[0] = 0;
    offsetsCopy[0] = 0;
    if (numFields) {
      PetscInt f;

      for (f = 0; f < numFields; f++) {
        PetscInt fDof;
        ierr = PetscSectionGetFieldDof(localCoarse,p,f,&fDof);CHKERRQ(ierr);
        rowOffsets[f + 1] = offsetsCopy[f + 1] = fDof + rowOffsets[f];
      }
      ierr = DMPlexGetIndicesPointFields_Internal(localCoarse,PETSC_FALSE,p,gOff < 0 ? -(gOff + 1) : gOff,offsetsCopy,PETSC_FALSE,NULL,-1, NULL,parentIndices);CHKERRQ(ierr);
    } else {
      ierr = DMPlexGetIndicesPoint_Internal(localCoarse,PETSC_FALSE,p,gOff < 0 ? -(gOff + 1) : gOff,offsetsCopy,PETSC_FALSE,NULL, NULL,parentIndices);CHKERRQ(ierr);
      rowOffsets[1] = offsetsCopy[0];
    }

    ierr = PetscSectionGetDof(multiRootSec,p,&numLeaves);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(multiRootSec,p,&leafStart);CHKERRQ(ierr);
    leafEnd = leafStart + numLeaves;
    for (l = leafStart; l < leafEnd; l++) {
      PetscInt numIndices, childId, offset;
      const PetscInt *childIndices;

      ierr = PetscSectionGetDof(rootIndicesSec,l,&numIndices);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(rootIndicesSec,l,&offset);CHKERRQ(ierr);
      childId = rootIndices[offset++];
      childIndices = &rootIndices[offset];
      numIndices--;

      if (childId == -1) { /* equivalent points: scatter */
        PetscInt i;

        for (i = 0; i < numIndices; i++) {
          PetscInt colIndex = childIndices[i];
          PetscInt rowIndex = parentIndices[i];
          if (rowIndex < 0) continue;
          if (colIndex < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Unconstrained fine and constrained coarse");
          if (colIndex >= colStart && colIndex < colEnd) {
            nnzD[rowIndex - rowStart] = 1;
          }
          else {
            nnzO[rowIndex - rowStart] = 1;
          }
        }
      }
      else {
        PetscInt parentId, f, lim;

        ierr = DMPlexGetTreeParent(refTree,childId,&parentId,NULL);CHKERRQ(ierr);

        lim = PetscMax(1,numFields);
        offsets[0] = 0;
        if (numFields) {
          PetscInt f;

          for (f = 0; f < numFields; f++) {
            PetscInt fDof;
            ierr = PetscSectionGetFieldDof(cSecRef,childId,f,&fDof);CHKERRQ(ierr);

            offsets[f + 1] = fDof + offsets[f];
          }
        }
        else {
          PetscInt cDof;

          ierr = PetscSectionGetDof(cSecRef,childId,&cDof);CHKERRQ(ierr);
          offsets[1] = cDof;
        }
        for (f = 0; f < lim; f++) {
          PetscInt parentStart = rowOffsets[f], parentEnd = rowOffsets[f + 1];
          PetscInt childStart = offsets[f], childEnd = offsets[f + 1];
          PetscInt i, numD = 0, numO = 0;

          for (i = childStart; i < childEnd; i++) {
            PetscInt colIndex = childIndices[i];

            if (colIndex < 0) continue;
            if (colIndex >= colStart && colIndex < colEnd) {
              numD++;
            }
            else {
              numO++;
            }
          }
          for (i = parentStart; i < parentEnd; i++) {
            PetscInt rowIndex = parentIndices[i];

            if (rowIndex < 0) continue;
            nnzD[rowIndex - rowStart] += numD;
            nnzO[rowIndex - rowStart] += numO;
          }
        }
      }
    }
  }
  /* preallocate */
  ierr = MatXAIJSetPreallocation(mat,1,nnzD,nnzO,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscFree2(nnzD,nnzO);CHKERRQ(ierr);
  /* insert values */
  ierr = DMPlexReferenceTreeGetChildrenMatrices_Injection(refTree,injRef,&childrenMats);CHKERRQ(ierr);
  for (p = pStartC; p < pEndC; p++) {
    PetscInt numLeaves, leafStart, leafEnd, l, dof, cdof, gOff;

    ierr = PetscSectionGetDof(globalCoarse,p,&dof);CHKERRQ(ierr);
    ierr = PetscSectionGetConstraintDof(globalCoarse,p,&cdof);CHKERRQ(ierr);
    if ((dof - cdof) <= 0) continue;
    ierr = PetscSectionGetOffset(globalCoarse,p,&gOff);CHKERRQ(ierr);

    rowOffsets[0] = 0;
    offsetsCopy[0] = 0;
    if (numFields) {
      PetscInt f;

      for (f = 0; f < numFields; f++) {
        PetscInt fDof;
        ierr = PetscSectionGetFieldDof(localCoarse,p,f,&fDof);CHKERRQ(ierr);
        rowOffsets[f + 1] = offsetsCopy[f + 1] = fDof + rowOffsets[f];
      }
      ierr = DMPlexGetIndicesPointFields_Internal(localCoarse,PETSC_FALSE,p,gOff < 0 ? -(gOff + 1) : gOff,offsetsCopy,PETSC_FALSE,NULL,-1, NULL,parentIndices);CHKERRQ(ierr);
    } else {
      ierr = DMPlexGetIndicesPoint_Internal(localCoarse,PETSC_FALSE,p,gOff < 0 ? -(gOff + 1) : gOff,offsetsCopy,PETSC_FALSE,NULL, NULL,parentIndices);CHKERRQ(ierr);
      rowOffsets[1] = offsetsCopy[0];
    }

    ierr = PetscSectionGetDof(multiRootSec,p,&numLeaves);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(multiRootSec,p,&leafStart);CHKERRQ(ierr);
    leafEnd = leafStart + numLeaves;
    for (l = leafStart; l < leafEnd; l++) {
      PetscInt numIndices, childId, offset;
      const PetscInt *childIndices;

      ierr = PetscSectionGetDof(rootIndicesSec,l,&numIndices);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(rootIndicesSec,l,&offset);CHKERRQ(ierr);
      childId = rootIndices[offset++];
      childIndices = &rootIndices[offset];
      numIndices--;

      if (childId == -1) { /* equivalent points: scatter */
        PetscInt i;

        for (i = 0; i < numIndices; i++) {
          ierr = MatSetValue(mat,parentIndices[i],childIndices[i],1.,INSERT_VALUES);CHKERRQ(ierr);
        }
      }
      else {
        PetscInt parentId, f, lim;

        ierr = DMPlexGetTreeParent(refTree,childId,&parentId,NULL);CHKERRQ(ierr);

        lim = PetscMax(1,numFields);
        offsets[0] = 0;
        if (numFields) {
          PetscInt f;

          for (f = 0; f < numFields; f++) {
            PetscInt fDof;
            ierr = PetscSectionGetFieldDof(cSecRef,childId,f,&fDof);CHKERRQ(ierr);

            offsets[f + 1] = fDof + offsets[f];
          }
        }
        else {
          PetscInt cDof;

          ierr = PetscSectionGetDof(cSecRef,childId,&cDof);CHKERRQ(ierr);
          offsets[1] = cDof;
        }
        for (f = 0; f < lim; f++) {
          PetscScalar    *childMat   = &childrenMats[childId - pRefStart][f][0];
          PetscInt       *rowIndices = &parentIndices[rowOffsets[f]];
          const PetscInt *colIndices = &childIndices[offsets[f]];

          ierr = MatSetValues(mat,rowOffsets[f+1]-rowOffsets[f],rowIndices,offsets[f+1]-offsets[f],colIndices,childMat,INSERT_VALUES);CHKERRQ(ierr);
        }
      }
    }
  }
  ierr = PetscSectionDestroy(&multiRootSec);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&rootIndicesSec);CHKERRQ(ierr);
  ierr = PetscFree(parentIndices);CHKERRQ(ierr);
  ierr = DMPlexReferenceTreeRestoreChildrenMatrices_Injection(refTree,injRef,&childrenMats);CHKERRQ(ierr);
  ierr = PetscFree(rootIndices);CHKERRQ(ierr);
  ierr = PetscFree3(offsets,offsetsCopy,rowOffsets);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTransferVecTree_Interpolate(DM coarse, Vec vecCoarseLocal, DM fine, Vec vecFine, PetscSF coarseToFine, PetscInt *cids, Vec grad, Vec cellGeom)
{
  PetscSF           coarseToFineEmbedded;
  PetscSection      globalCoarse, globalFine;
  PetscSection      localCoarse, localFine;
  PetscSection      aSec, cSec;
  PetscSection      rootValuesSec;
  PetscSection      leafValuesSec;
  PetscScalar       *rootValues, *leafValues;
  IS                aIS;
  const PetscInt    *anchors;
  Mat               cMat;
  PetscInt          numFields;
  PetscInt          pStartC, pEndC, pStartF, pEndF, p, cellStart, cellEnd, cellEndInterior;
  PetscInt          aStart, aEnd, cStart, cEnd;
  PetscInt          *maxChildIds;
  PetscInt          *offsets, *newOffsets, *offsetsCopy, *newOffsetsCopy, *rowOffsets, *numD, *numO;
  PetscFV           fv = NULL;
  PetscInt          dim, numFVcomps = -1, fvField = -1;
  DM                cellDM = NULL, gradDM = NULL;
  const PetscScalar *cellGeomArray = NULL;
  const PetscScalar *gradArray = NULL;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = VecSetOption(vecFine,VEC_IGNORE_NEGATIVE_INDICES,PETSC_TRUE);CHKERRQ(ierr);
  ierr = DMPlexGetChart(coarse,&pStartC,&pEndC);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(coarse,0,&cellStart,&cellEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHybridBounds(coarse,&cellEndInterior,NULL,NULL,NULL);CHKERRQ(ierr);
  cellEnd = (cellEndInterior < 0) ? cellEnd : cellEndInterior;
  ierr = DMPlexGetChart(fine,&pStartF,&pEndF);CHKERRQ(ierr);
  ierr = DMGetGlobalSection(fine,&globalFine);CHKERRQ(ierr);
  ierr = DMGetCoordinateDim(coarse,&dim);CHKERRQ(ierr);
  { /* winnow fine points that don't have global dofs out of the sf */
    PetscInt       nleaves, l;
    const PetscInt *leaves;
    PetscInt       dof, cdof, numPointsWithDofs, offset, *pointsWithDofs;

    ierr = PetscSFGetGraph(coarseToFine,NULL,&nleaves,&leaves,NULL);CHKERRQ(ierr);

    for (l = 0, numPointsWithDofs = 0; l < nleaves; l++) {
      PetscInt p = leaves ? leaves[l] : l;

      ierr = PetscSectionGetDof(globalFine,p,&dof);CHKERRQ(ierr);
      ierr = PetscSectionGetConstraintDof(globalFine,p,&cdof);CHKERRQ(ierr);
      if ((dof - cdof) > 0) {
        numPointsWithDofs++;
      }
    }
    ierr = PetscMalloc1(numPointsWithDofs,&pointsWithDofs);CHKERRQ(ierr);
    for (l = 0, offset = 0; l < nleaves; l++) {
      PetscInt p = leaves ? leaves[l] : l;

      ierr = PetscSectionGetDof(globalFine,p,&dof);CHKERRQ(ierr);
      ierr = PetscSectionGetConstraintDof(globalFine,p,&cdof);CHKERRQ(ierr);
      if ((dof - cdof) > 0) {
        pointsWithDofs[offset++] = l;
      }
    }
    ierr = PetscSFCreateEmbeddedLeafSF(coarseToFine, numPointsWithDofs, pointsWithDofs, &coarseToFineEmbedded);CHKERRQ(ierr);
    ierr = PetscFree(pointsWithDofs);CHKERRQ(ierr);
  }
  /* communicate back to the coarse mesh which coarse points have children (that may require interpolation) */
  ierr = PetscMalloc1(pEndC-pStartC,&maxChildIds);CHKERRQ(ierr);
  for (p = pStartC; p < pEndC; p++) {
    maxChildIds[p - pStartC] = -2;
  }
  ierr = PetscSFReduceBegin(coarseToFineEmbedded,MPIU_INT,cids,maxChildIds,MPIU_MAX);CHKERRQ(ierr);
  ierr = PetscSFReduceEnd(coarseToFineEmbedded,MPIU_INT,cids,maxChildIds,MPIU_MAX);CHKERRQ(ierr);

  ierr = DMGetLocalSection(coarse,&localCoarse);CHKERRQ(ierr);
  ierr = DMGetGlobalSection(coarse,&globalCoarse);CHKERRQ(ierr);

  ierr = DMPlexGetAnchors(coarse,&aSec,&aIS);CHKERRQ(ierr);
  ierr = ISGetIndices(aIS,&anchors);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(aSec,&aStart,&aEnd);CHKERRQ(ierr);

  ierr = DMGetDefaultConstraints(coarse,&cSec,&cMat);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(cSec,&cStart,&cEnd);CHKERRQ(ierr);

  /* create sections that will send to children the indices and matrices they will need to construct the interpolator */
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject)coarse),&rootValuesSec);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(rootValuesSec,pStartC,pEndC);CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(localCoarse,&numFields);CHKERRQ(ierr);
  {
    PetscInt maxFields = PetscMax(1,numFields) + 1;
    ierr = PetscMalloc7(maxFields,&offsets,maxFields,&offsetsCopy,maxFields,&newOffsets,maxFields,&newOffsetsCopy,maxFields,&rowOffsets,maxFields,&numD,maxFields,&numO);CHKERRQ(ierr);
  }
  if (grad) {
    PetscInt i;

    ierr = VecGetDM(cellGeom,&cellDM);CHKERRQ(ierr);
    ierr = VecGetArrayRead(cellGeom,&cellGeomArray);CHKERRQ(ierr);
    ierr = VecGetDM(grad,&gradDM);CHKERRQ(ierr);
    ierr = VecGetArrayRead(grad,&gradArray);CHKERRQ(ierr);
    for (i = 0; i < PetscMax(1,numFields); i++) {
      PetscObject  obj;
      PetscClassId id;

      ierr = DMGetField(coarse, i, NULL, &obj);CHKERRQ(ierr);
      ierr = PetscObjectGetClassId(obj,&id);CHKERRQ(ierr);
      if (id == PETSCFV_CLASSID) {
        fv      = (PetscFV) obj;
        ierr    = PetscFVGetNumComponents(fv,&numFVcomps);CHKERRQ(ierr);
        fvField = i;
        break;
      }
    }
  }

  for (p = pStartC; p < pEndC; p++) { /* count the sizes of the indices and matrices */
    PetscInt dof;
    PetscInt maxChildId     = maxChildIds[p - pStartC];
    PetscInt numValues      = 0;

    ierr = PetscSectionGetDof(globalCoarse,p,&dof);CHKERRQ(ierr);
    if (dof < 0) {
      dof = -(dof + 1);
    }
    offsets[0]    = 0;
    newOffsets[0] = 0;
    if (maxChildId >= 0) { /* this point has children (with dofs) that will need to be interpolated from the closure of p */
      PetscInt *closure = NULL, closureSize, cl;

      ierr = DMPlexGetTransitiveClosure(coarse,p,PETSC_TRUE,&closureSize,&closure);CHKERRQ(ierr);
      for (cl = 0; cl < closureSize; cl++) { /* get the closure */
        PetscInt c = closure[2 * cl], clDof;

        ierr = PetscSectionGetDof(localCoarse,c,&clDof);CHKERRQ(ierr);
        numValues += clDof;
      }
      ierr = DMPlexRestoreTransitiveClosure(coarse,p,PETSC_TRUE,&closureSize,&closure);CHKERRQ(ierr);
    }
    else if (maxChildId == -1) {
      ierr = PetscSectionGetDof(localCoarse,p,&numValues);CHKERRQ(ierr);
    }
    /* we will pack the column indices with the field offsets */
    if (maxChildId >= 0 && grad && p >= cellStart && p < cellEnd) {
      /* also send the centroid, and the gradient */
      numValues += dim * (1 + numFVcomps);
    }
    ierr = PetscSectionSetDof(rootValuesSec,p,numValues);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(rootValuesSec);CHKERRQ(ierr);
  {
    PetscInt          numRootValues;
    const PetscScalar *coarseArray;

    ierr = PetscSectionGetStorageSize(rootValuesSec,&numRootValues);CHKERRQ(ierr);
    ierr = PetscMalloc1(numRootValues,&rootValues);CHKERRQ(ierr);
    ierr = VecGetArrayRead(vecCoarseLocal,&coarseArray);CHKERRQ(ierr);
    for (p = pStartC; p < pEndC; p++) {
      PetscInt    numValues;
      PetscInt    pValOff;
      PetscScalar *pVal;
      PetscInt    maxChildId = maxChildIds[p - pStartC];

      ierr = PetscSectionGetDof(rootValuesSec,p,&numValues);CHKERRQ(ierr);
      if (!numValues) {
        continue;
      }
      ierr = PetscSectionGetOffset(rootValuesSec,p,&pValOff);CHKERRQ(ierr);
      pVal = &(rootValues[pValOff]);
      if (maxChildId >= 0) { /* build an identity matrix, apply matrix constraints on the right */
        PetscInt closureSize = numValues;
        ierr = DMPlexVecGetClosure(coarse,NULL,vecCoarseLocal,p,&closureSize,&pVal);CHKERRQ(ierr);
        if (grad && p >= cellStart && p < cellEnd) {
          PetscFVCellGeom *cg;
          PetscScalar     *gradVals = NULL;
          PetscInt        i;

          pVal += (numValues - dim * (1 + numFVcomps));

          ierr = DMPlexPointLocalRead(cellDM,p,cellGeomArray,(void *) &cg);CHKERRQ(ierr);
          for (i = 0; i < dim; i++) pVal[i] = cg->centroid[i];
          pVal += dim;
          ierr = DMPlexPointGlobalRead(gradDM,p,gradArray,(void *) &gradVals);CHKERRQ(ierr);
          for (i = 0; i < dim * numFVcomps; i++) pVal[i] = gradVals[i];
        }
      }
      else if (maxChildId == -1) {
        PetscInt lDof, lOff, i;

        ierr = PetscSectionGetDof(localCoarse,p,&lDof);CHKERRQ(ierr);
        ierr = PetscSectionGetOffset(localCoarse,p,&lOff);CHKERRQ(ierr);
        for (i = 0; i < lDof; i++) pVal[i] = coarseArray[lOff + i];
      }
    }
    ierr = VecRestoreArrayRead(vecCoarseLocal,&coarseArray);CHKERRQ(ierr);
    ierr = PetscFree(maxChildIds);CHKERRQ(ierr);
  }
  {
    PetscSF  valuesSF;
    PetscInt *remoteOffsetsValues, numLeafValues;

    ierr = PetscSectionCreate(PetscObjectComm((PetscObject)fine),&leafValuesSec);CHKERRQ(ierr);
    ierr = PetscSFDistributeSection(coarseToFineEmbedded,rootValuesSec,&remoteOffsetsValues,leafValuesSec);CHKERRQ(ierr);
    ierr = PetscSFCreateSectionSF(coarseToFineEmbedded,rootValuesSec,remoteOffsetsValues,leafValuesSec,&valuesSF);CHKERRQ(ierr);
    ierr = PetscSFDestroy(&coarseToFineEmbedded);CHKERRQ(ierr);
    ierr = PetscFree(remoteOffsetsValues);CHKERRQ(ierr);
    ierr = PetscSectionGetStorageSize(leafValuesSec,&numLeafValues);CHKERRQ(ierr);
    ierr = PetscMalloc1(numLeafValues,&leafValues);CHKERRQ(ierr);
    ierr = PetscSFBcastBegin(valuesSF,MPIU_SCALAR,rootValues,leafValues);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(valuesSF,MPIU_SCALAR,rootValues,leafValues);CHKERRQ(ierr);
    ierr = PetscSFDestroy(&valuesSF);CHKERRQ(ierr);
    ierr = PetscFree(rootValues);CHKERRQ(ierr);
    ierr = PetscSectionDestroy(&rootValuesSec);CHKERRQ(ierr);
  }
  ierr = DMGetLocalSection(fine,&localFine);CHKERRQ(ierr);
  {
    PetscInt    maxDof;
    PetscInt    *rowIndices;
    DM           refTree;
    PetscInt     **refPointFieldN;
    PetscScalar  ***refPointFieldMats;
    PetscSection refConSec, refAnSec;
    PetscInt     pRefStart,pRefEnd,leafStart,leafEnd;
    PetscScalar  *pointWork;

    ierr = PetscSectionGetMaxDof(localFine,&maxDof);CHKERRQ(ierr);
    ierr = DMGetWorkArray(fine,maxDof,MPIU_INT,&rowIndices);CHKERRQ(ierr);
    ierr = DMGetWorkArray(fine,maxDof,MPIU_SCALAR,&pointWork);CHKERRQ(ierr);
    ierr = DMPlexGetReferenceTree(fine,&refTree);CHKERRQ(ierr);
    ierr = DMCopyDisc(fine,refTree);CHKERRQ(ierr);
    ierr = DMPlexReferenceTreeGetChildrenMatrices(refTree,&refPointFieldMats,&refPointFieldN);CHKERRQ(ierr);
    ierr = DMGetDefaultConstraints(refTree,&refConSec,NULL);CHKERRQ(ierr);
    ierr = DMPlexGetAnchors(refTree,&refAnSec,NULL);CHKERRQ(ierr);
    ierr = PetscSectionGetChart(refConSec,&pRefStart,&pRefEnd);CHKERRQ(ierr);
    ierr = PetscSectionGetChart(leafValuesSec,&leafStart,&leafEnd);CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(fine,0,&cellStart,&cellEnd);CHKERRQ(ierr);
    ierr = DMPlexGetHybridBounds(fine,&cellEndInterior,NULL,NULL,NULL);CHKERRQ(ierr);
    cellEnd = (cellEndInterior < 0) ? cellEnd : cellEndInterior;
    for (p = leafStart; p < leafEnd; p++) {
      PetscInt          gDof, gcDof, gOff, lDof;
      PetscInt          numValues, pValOff;
      PetscInt          childId;
      const PetscScalar *pVal;
      const PetscScalar *fvGradData = NULL;

      ierr = PetscSectionGetDof(globalFine,p,&gDof);CHKERRQ(ierr);
      ierr = PetscSectionGetDof(localFine,p,&lDof);CHKERRQ(ierr);
      ierr = PetscSectionGetConstraintDof(globalFine,p,&gcDof);CHKERRQ(ierr);
      if ((gDof - gcDof) <= 0) {
        continue;
      }
      ierr = PetscSectionGetOffset(globalFine,p,&gOff);CHKERRQ(ierr);
      ierr = PetscSectionGetDof(leafValuesSec,p,&numValues);CHKERRQ(ierr);
      if (!numValues) continue;
      ierr = PetscSectionGetOffset(leafValuesSec,p,&pValOff);CHKERRQ(ierr);
      pVal = &leafValues[pValOff];
      offsets[0]        = 0;
      offsetsCopy[0]    = 0;
      newOffsets[0]     = 0;
      newOffsetsCopy[0] = 0;
      childId           = cids[p - pStartF];
      if (numFields) {
        PetscInt f;
        for (f = 0; f < numFields; f++) {
          PetscInt rowDof;

          ierr = PetscSectionGetFieldDof(localFine,p,f,&rowDof);CHKERRQ(ierr);
          offsets[f + 1]        = offsets[f] + rowDof;
          offsetsCopy[f + 1]    = offsets[f + 1];
          /* TODO: closure indices */
          newOffsets[f + 1]     = newOffsets[f] + ((childId == -1) ? rowDof : refPointFieldN[childId - pRefStart][f]);
        }
        ierr = DMPlexGetIndicesPointFields_Internal(localFine,PETSC_FALSE,p,gOff,offsetsCopy,PETSC_FALSE,NULL,-1,NULL,rowIndices);CHKERRQ(ierr);
      }
      else {
        offsets[0]    = 0;
        offsets[1]    = lDof;
        newOffsets[0] = 0;
        newOffsets[1] = (childId == -1) ? lDof : refPointFieldN[childId - pRefStart][0];
        ierr = DMPlexGetIndicesPoint_Internal(localFine,PETSC_FALSE,p,gOff,offsetsCopy,PETSC_FALSE,NULL,NULL,rowIndices);CHKERRQ(ierr);
      }
      if (childId == -1) { /* no child interpolation: one nnz per */
        ierr = VecSetValues(vecFine,numValues,rowIndices,pVal,INSERT_VALUES);CHKERRQ(ierr);
      } else {
        PetscInt f;

        if (grad && p >= cellStart && p < cellEnd) {
          numValues -= (dim * (1 + numFVcomps));
          fvGradData = &pVal[numValues];
        }
        for (f = 0; f < PetscMax(1,numFields); f++) {
          const PetscScalar *childMat = refPointFieldMats[childId - pRefStart][f];
          PetscInt numRows = offsets[f+1] - offsets[f];
          PetscInt numCols = newOffsets[f + 1] - newOffsets[f];
          const PetscScalar *cVal = &pVal[newOffsets[f]];
          PetscScalar *rVal = &pointWork[offsets[f]];
          PetscInt i, j;

#if 0
          ierr = PetscInfo5(coarse,"childId %D, numRows %D, numCols %D, refPointFieldN %D maxDof %D\n",childId,numRows,numCols,refPointFieldN[childId - pRefStart][f], maxDof);CHKERRQ(ierr);
#endif
          for (i = 0; i < numRows; i++) {
            PetscScalar val = 0.;
            for (j = 0; j < numCols; j++) {
              val += childMat[i * numCols + j] * cVal[j];
            }
            rVal[i] = val;
          }
          if (f == fvField && p >= cellStart && p < cellEnd) {
            PetscReal   centroid[3];
            PetscScalar diff[3];
            const PetscScalar *parentCentroid = &fvGradData[0];
            const PetscScalar *gradient       = &fvGradData[dim];

            ierr = DMPlexComputeCellGeometryFVM(fine,p,NULL,centroid,NULL);CHKERRQ(ierr);
            for (i = 0; i < dim; i++) {
              diff[i] = centroid[i] - parentCentroid[i];
            }
            for (i = 0; i < numFVcomps; i++) {
              PetscScalar val = 0.;

              for (j = 0; j < dim; j++) {
                val += gradient[dim * i + j] * diff[j];
              }
              rVal[i] += val;
            }
          }
          ierr = VecSetValues(vecFine,numRows,&rowIndices[offsets[f]],rVal,INSERT_VALUES);CHKERRQ(ierr);
        }
      }
    }
    ierr = DMPlexReferenceTreeRestoreChildrenMatrices(refTree,&refPointFieldMats,&refPointFieldN);CHKERRQ(ierr);
    ierr = DMRestoreWorkArray(fine,maxDof,MPIU_SCALAR,&pointWork);CHKERRQ(ierr);
    ierr = DMRestoreWorkArray(fine,maxDof,MPIU_INT,&rowIndices);CHKERRQ(ierr);
  }
  ierr = PetscFree(leafValues);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&leafValuesSec);CHKERRQ(ierr);
  ierr = PetscFree7(offsets,offsetsCopy,newOffsets,newOffsetsCopy,rowOffsets,numD,numO);CHKERRQ(ierr);
  ierr = ISRestoreIndices(aIS,&anchors);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTransferVecTree_Inject(DM fine, Vec vecFine, DM coarse, Vec vecCoarse, PetscSF coarseToFine, PetscInt *cids)
{
  DM             refTree;
  PetscSection   multiRootSec, rootIndicesSec;
  PetscSection   globalCoarse, globalFine;
  PetscSection   localCoarse, localFine;
  PetscSection   cSecRef;
  PetscInt       *parentIndices, pRefStart, pRefEnd;
  PetscScalar    *rootValues, *parentValues;
  Mat            injRef;
  PetscInt       numFields, maxDof;
  PetscInt       pStartC, pEndC, pStartF, pEndF, p;
  PetscInt       *offsets, *offsetsCopy, *rowOffsets;
  PetscLayout    rowMap, colMap;
  PetscInt       rowStart, rowEnd, colStart, colEnd;
  PetscScalar    ***childrenMats=NULL ; /* gcc -O gives 'may be used uninitialized' warning'. Initializing to suppress this warning */
  PetscErrorCode ierr;

  PetscFunctionBegin;

  /* get the templates for the fine-to-coarse injection from the reference tree */
  ierr = VecSetOption(vecFine,VEC_IGNORE_NEGATIVE_INDICES,PETSC_TRUE);CHKERRQ(ierr);
  ierr = VecSetOption(vecCoarse,VEC_IGNORE_NEGATIVE_INDICES,PETSC_TRUE);CHKERRQ(ierr);
  ierr = DMPlexGetReferenceTree(coarse,&refTree);CHKERRQ(ierr);
  ierr = DMCopyDisc(coarse,refTree);CHKERRQ(ierr);
  ierr = DMGetDefaultConstraints(refTree,&cSecRef,NULL);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(cSecRef,&pRefStart,&pRefEnd);CHKERRQ(ierr);
  ierr = DMPlexReferenceTreeGetInjector(refTree,&injRef);CHKERRQ(ierr);

  ierr = DMPlexGetChart(fine,&pStartF,&pEndF);CHKERRQ(ierr);
  ierr = DMGetLocalSection(fine,&localFine);CHKERRQ(ierr);
  ierr = DMGetGlobalSection(fine,&globalFine);CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(localFine,&numFields);CHKERRQ(ierr);
  ierr = DMPlexGetChart(coarse,&pStartC,&pEndC);CHKERRQ(ierr);
  ierr = DMGetLocalSection(coarse,&localCoarse);CHKERRQ(ierr);
  ierr = DMGetGlobalSection(coarse,&globalCoarse);CHKERRQ(ierr);
  ierr = PetscSectionGetMaxDof(localCoarse,&maxDof);CHKERRQ(ierr);
  {
    PetscInt maxFields = PetscMax(1,numFields) + 1;
    ierr = PetscMalloc3(maxFields,&offsets,maxFields,&offsetsCopy,maxFields,&rowOffsets);CHKERRQ(ierr);
  }

  ierr = DMPlexTransferInjectorTree(coarse,fine,coarseToFine,cids,vecFine,numFields,offsets,&multiRootSec,&rootIndicesSec,NULL,&rootValues);CHKERRQ(ierr);

  ierr = PetscMalloc2(maxDof,&parentIndices,maxDof,&parentValues);CHKERRQ(ierr);

  /* count indices */
  ierr = VecGetLayout(vecFine,&colMap);CHKERRQ(ierr);
  ierr = VecGetLayout(vecCoarse,&rowMap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(rowMap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(colMap);CHKERRQ(ierr);
  ierr = PetscLayoutGetRange(rowMap,&rowStart,&rowEnd);CHKERRQ(ierr);
  ierr = PetscLayoutGetRange(colMap,&colStart,&colEnd);CHKERRQ(ierr);
  /* insert values */
  ierr = DMPlexReferenceTreeGetChildrenMatrices_Injection(refTree,injRef,&childrenMats);CHKERRQ(ierr);
  for (p = pStartC; p < pEndC; p++) {
    PetscInt  numLeaves, leafStart, leafEnd, l, dof, cdof, gOff;
    PetscBool contribute = PETSC_FALSE;

    ierr = PetscSectionGetDof(globalCoarse,p,&dof);CHKERRQ(ierr);
    ierr = PetscSectionGetConstraintDof(globalCoarse,p,&cdof);CHKERRQ(ierr);
    if ((dof - cdof) <= 0) continue;
    ierr = PetscSectionGetDof(localCoarse,p,&dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(globalCoarse,p,&gOff);CHKERRQ(ierr);

    rowOffsets[0] = 0;
    offsetsCopy[0] = 0;
    if (numFields) {
      PetscInt f;

      for (f = 0; f < numFields; f++) {
        PetscInt fDof;
        ierr = PetscSectionGetFieldDof(localCoarse,p,f,&fDof);CHKERRQ(ierr);
        rowOffsets[f + 1] = offsetsCopy[f + 1] = fDof + rowOffsets[f];
      }
      ierr = DMPlexGetIndicesPointFields_Internal(localCoarse,PETSC_FALSE,p,gOff < 0 ? -(gOff + 1) : gOff,offsetsCopy,PETSC_FALSE,NULL,-1,NULL,parentIndices);CHKERRQ(ierr);
    } else {
      ierr = DMPlexGetIndicesPoint_Internal(localCoarse,PETSC_FALSE,p,gOff < 0 ? -(gOff + 1) : gOff,offsetsCopy,PETSC_FALSE,NULL,NULL,parentIndices);CHKERRQ(ierr);
      rowOffsets[1] = offsetsCopy[0];
    }

    ierr = PetscSectionGetDof(multiRootSec,p,&numLeaves);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(multiRootSec,p,&leafStart);CHKERRQ(ierr);
    leafEnd = leafStart + numLeaves;
    for (l = 0; l < dof; l++) parentValues[l] = 0.;
    for (l = leafStart; l < leafEnd; l++) {
      PetscInt numIndices, childId, offset;
      const PetscScalar *childValues;

      ierr = PetscSectionGetDof(rootIndicesSec,l,&numIndices);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(rootIndicesSec,l,&offset);CHKERRQ(ierr);
      childId = (PetscInt) PetscRealPart(rootValues[offset++]);
      childValues = &rootValues[offset];
      numIndices--;

      if (childId == -2) { /* skip */
        continue;
      } else if (childId == -1) { /* equivalent points: scatter */
        PetscInt m;

        contribute = PETSC_TRUE;
        for (m = 0; m < numIndices; m++) parentValues[m] = childValues[m];
      } else { /* contributions from children: sum with injectors from reference tree */
        PetscInt parentId, f, lim;

        contribute = PETSC_TRUE;
        ierr = DMPlexGetTreeParent(refTree,childId,&parentId,NULL);CHKERRQ(ierr);

        lim = PetscMax(1,numFields);
        offsets[0] = 0;
        if (numFields) {
          PetscInt f;

          for (f = 0; f < numFields; f++) {
            PetscInt fDof;
            ierr = PetscSectionGetFieldDof(cSecRef,childId,f,&fDof);CHKERRQ(ierr);

            offsets[f + 1] = fDof + offsets[f];
          }
        }
        else {
          PetscInt cDof;

          ierr = PetscSectionGetDof(cSecRef,childId,&cDof);CHKERRQ(ierr);
          offsets[1] = cDof;
        }
        for (f = 0; f < lim; f++) {
          PetscScalar       *childMat   = &childrenMats[childId - pRefStart][f][0];
          PetscInt          n           = offsets[f+1]-offsets[f];
          PetscInt          m           = rowOffsets[f+1]-rowOffsets[f];
          PetscInt          i, j;
          const PetscScalar *colValues  = &childValues[offsets[f]];

          for (i = 0; i < m; i++) {
            PetscScalar val = 0.;
            for (j = 0; j < n; j++) {
              val += childMat[n * i + j] * colValues[j];
            }
            parentValues[rowOffsets[f] + i] += val;
          }
        }
      }
    }
    if (contribute) {ierr = VecSetValues(vecCoarse,dof,parentIndices,parentValues,INSERT_VALUES);CHKERRQ(ierr);}
  }
  ierr = PetscSectionDestroy(&multiRootSec);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&rootIndicesSec);CHKERRQ(ierr);
  ierr = PetscFree2(parentIndices,parentValues);CHKERRQ(ierr);
  ierr = DMPlexReferenceTreeRestoreChildrenMatrices_Injection(refTree,injRef,&childrenMats);CHKERRQ(ierr);
  ierr = PetscFree(rootValues);CHKERRQ(ierr);
  ierr = PetscFree3(offsets,offsetsCopy,rowOffsets);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMPlexTransferVecTree - transfer a vector between two meshes that differ from each other by refinement/coarsening
  that can be represented by a common reference tree used by both.  This routine can be used for a combination of
  coarsening and refinement at the same time.

  collective

  Input Parameters:
+ dmIn        - The DMPlex mesh for the input vector
. vecIn       - The input vector
. sfRefine    - A star forest indicating points in the mesh dmIn (roots in the star forest) that are parents to points in
                the mesh dmOut (leaves in the star forest), i.e. where dmOut is more refined than dmIn
. sfCoarsen   - A star forest indicating points in the mesh dmOut (roots in the star forest) that are parents to points in
                the mesh dmIn (leaves in the star forest), i.e. where dmOut is more coarsened than dmIn
. cidsRefine  - The childIds of the points in dmOut.  These childIds relate back to the reference tree: childid[j] = k implies
                that mesh point j of dmOut was refined from a point in dmIn just as the mesh point k in the reference
                tree was refined from its parent.  childid[j] = -1 indicates that the point j in dmOut is exactly
                equivalent to its root in dmIn, so no interpolation is necessary.  childid[j] = -2 indicates that this
                point j in dmOut is not a leaf of sfRefine.
. cidsCoarsen - The childIds of the points in dmIn.  These childIds relate back to the reference tree: childid[j] = k implies
                that mesh point j of dmIn coarsens to a point in dmOut just as the mesh point k in the reference
                tree coarsens to its parent.  childid[j] = -2 indicates that point j in dmOut is not a leaf in sfCoarsen.
. useBCs      - PETSC_TRUE indicates that boundary values should be inserted into vecIn before transfer.
- time        - Used if boundary values are time dependent.

  Output Parameters:
. vecOut      - Using interpolation and injection operators calculated on the reference tree, the transferred
                projection of vecIn from dmIn to dmOut.  Note that any field discretized with a PetscFV finite volume
                method that uses gradient reconstruction will use reconstructed gradients when interpolating from
                coarse points to fine points.

  Level: developer

.seealso: DMPlexSetReferenceTree(), DMPlexGetReferenceTree(), PetscFVGetComputeGradients()
@*/
PetscErrorCode DMPlexTransferVecTree(DM dmIn, Vec vecIn, DM dmOut, Vec vecOut, PetscSF sfRefine, PetscSF sfCoarsen, PetscInt *cidsRefine, PetscInt *cidsCoarsen, PetscBool useBCs, PetscReal time)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecSet(vecOut,0.0);CHKERRQ(ierr);
  if (sfRefine) {
    Vec vecInLocal;
    DM  dmGrad = NULL;
    Vec faceGeom = NULL, cellGeom = NULL, grad = NULL;

    ierr = DMGetLocalVector(dmIn,&vecInLocal);CHKERRQ(ierr);
    ierr = VecSet(vecInLocal,0.0);CHKERRQ(ierr);
    {
      PetscInt  numFields, i;

      ierr = DMGetNumFields(dmIn, &numFields);CHKERRQ(ierr);
      for (i = 0; i < numFields; i++) {
        PetscObject  obj;
        PetscClassId classid;

        ierr = DMGetField(dmIn, i, NULL, &obj);CHKERRQ(ierr);
        ierr = PetscObjectGetClassId(obj, &classid);CHKERRQ(ierr);
        if (classid == PETSCFV_CLASSID) {
          ierr = DMPlexGetDataFVM(dmIn,(PetscFV)obj,&cellGeom,&faceGeom,&dmGrad);CHKERRQ(ierr);
          break;
        }
      }
    }
    if (useBCs) {
      ierr = DMPlexInsertBoundaryValues(dmIn,PETSC_TRUE,vecInLocal,time,faceGeom,cellGeom,NULL);CHKERRQ(ierr);
    }
    ierr = DMGlobalToLocalBegin(dmIn,vecIn,INSERT_VALUES,vecInLocal);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(dmIn,vecIn,INSERT_VALUES,vecInLocal);CHKERRQ(ierr);
    if (dmGrad) {
      ierr = DMGetGlobalVector(dmGrad,&grad);CHKERRQ(ierr);
      ierr = DMPlexReconstructGradientsFVM(dmIn,vecInLocal,grad);CHKERRQ(ierr);
    }
    ierr = DMPlexTransferVecTree_Interpolate(dmIn,vecInLocal,dmOut,vecOut,sfRefine,cidsRefine,grad,cellGeom);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dmIn,&vecInLocal);CHKERRQ(ierr);
    if (dmGrad) {
      ierr = DMRestoreGlobalVector(dmGrad,&grad);CHKERRQ(ierr);
    }
  }
  if (sfCoarsen) {
    ierr = DMPlexTransferVecTree_Inject(dmIn,vecIn,dmOut,vecOut,sfCoarsen,cidsCoarsen);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(vecOut);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(vecOut);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
