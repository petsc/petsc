#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/
#include <../src/sys/utils/hash.h>
#include <petsc/private/isimpl.h>
#include <petsc/private/petscfeimpl.h>
#include <petscsf.h>
#include <petscds.h>

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
  if (ref) {PetscValidHeaderSpecific(ref, DM_CLASSID, 2);}
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
#define __FUNCT__ "DMPlexReferenceTreeGetChildSymmetry_Default"
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
          ABswap        = DihedralSwap(coneSize,oA[i],oBtrue);CHKERRQ(ierr);
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
  }
  else {
    oAvert     = parentOrientA;
    oBvert     = parentOrientB;
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

#undef __FUNCT__
#define __FUNCT__ "DMPlexReferenceTreeGetChildSymmetry"
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
. childB - if not NULL, the new childID for describing the child

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

#undef __FUNCT__
#define __FUNCT__ "DMPlexCreateReferenceTree_Union"
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
  ierr = DMPlexSetTree_Internal(*ref,parentSection,parents,childIDs,PETSC_TRUE,PETSC_FALSE);CHKERRQ(ierr);
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
#define __FUNCT__ "AnchorsFlatten"
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

#undef __FUNCT__
#define __FUNCT__ "DMPlexCreateAnchors_Tree"
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

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetTrueSupportSize"
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

#undef __FUNCT__
#define __FUNCT__ "DMPlexTreeExchangeSupports"
static PetscErrorCode DMPlexTreeExchangeSupports(DM dm)
{
  DM_Plex *mesh = (DM_Plex *)dm->data;
  PetscSection newSupportSection;
  PetscInt newSize, *newSupports, pStart, pEnd, p, d, depth;
  PetscInt *numTrueSupp;
  PetscInt *offsets;
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

#undef __FUNCT__
#define __FUNCT__ "DMPlexSetTree_Internal"
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
    ierr = PetscMemcpy(mesh->parents, parents, size * sizeof(*parents));CHKERRQ(ierr);
  }
  if (childIDs != mesh->childIDs) {
    ierr = PetscFree(mesh->childIDs);CHKERRQ(ierr);
    ierr = PetscMalloc1(size,&mesh->childIDs);CHKERRQ(ierr);
    ierr = PetscMemcpy(mesh->childIDs, childIDs, size * sizeof(*childIDs));CHKERRQ(ierr);
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
  }
  else {
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

.seealso: DMPlexGetTree(), DMPlexSetReferenceTree(), DMPlexSetAnchors(), DMPlexGetTreeParent(), DMPlexGetTreeChildren()
@*/
PetscErrorCode DMPlexSetTree(DM dm, PetscSection parentSection, PetscInt parents[], PetscInt childIDs[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexSetTree_Internal(dm,parentSection,parents,childIDs,PETSC_FALSE,PETSC_TRUE);CHKERRQ(ierr);
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

#undef __FUNCT__
#define __FUNCT__ "DMPlexComputeAnchorMatrix_Tree_Direct"
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
    PetscObject disc;
    PetscFE fe = NULL;
    PetscFV fv = NULL;
    PetscClassId id;
    PetscDualSpace space;
    PetscInt i, j, k, nPoints, offset;
    PetscInt fSize, fComp;
    PetscReal *B = NULL;
    PetscReal *weights, *pointsRef, *pointsReal;
    Mat Amat, Bmat, Xmat;

    ierr = PetscDSGetDiscretization(ds,f,&disc);CHKERRQ(ierr);
    ierr = PetscObjectGetClassId(disc,&id);CHKERRQ(ierr);
    if (id == PETSCFE_CLASSID) {
      fe = (PetscFE) disc;
      ierr = PetscFEGetDualSpace(fe,&space);CHKERRQ(ierr);
      ierr = PetscDualSpaceGetDimension(space,&fSize);CHKERRQ(ierr);
      ierr = PetscFEGetNumComponents(fe,&fComp);CHKERRQ(ierr);
    }
    else if (id == PETSCFV_CLASSID) {
      fv = (PetscFV) disc;
      ierr = PetscFVGetDualSpace(fv,&space);CHKERRQ(ierr);
      ierr = PetscDualSpaceGetDimension(space,&fSize);CHKERRQ(ierr);
      ierr = PetscFVGetNumComponents(fv,&fComp);CHKERRQ(ierr);
    }
    else SETERRQ1(PetscObjectComm(disc),PETSC_ERR_ARG_UNKNOWN_TYPE, "PetscDS discretization id %d not recognized.", id);

    ierr = MatCreate(PETSC_COMM_SELF,&Amat);CHKERRQ(ierr);
    ierr = MatSetSizes(Amat,fSize,fSize,fSize,fSize);CHKERRQ(ierr);
    ierr = MatSetType(Amat,MATSEQDENSE);CHKERRQ(ierr);
    ierr = MatSetUp(Amat);CHKERRQ(ierr);
    ierr = MatDuplicate(Amat,MAT_DO_NOT_COPY_VALUES,&Bmat);CHKERRQ(ierr);
    ierr = MatDuplicate(Amat,MAT_DO_NOT_COPY_VALUES,&Xmat);CHKERRQ(ierr);
    nPoints = 0;
    for (i = 0; i < fSize; i++) {
      PetscInt        qPoints;
      PetscQuadrature quad;

      ierr = PetscDualSpaceGetFunctional(space,i,&quad);CHKERRQ(ierr);
      ierr = PetscQuadratureGetData(quad,NULL,&qPoints,NULL,NULL);CHKERRQ(ierr);
      nPoints += qPoints;
    }
    ierr = PetscMalloc3(nPoints,&weights,spdim*nPoints,&pointsRef,spdim*nPoints,&pointsReal);CHKERRQ(ierr);
    offset = 0;
    for (i = 0; i < fSize; i++) {
      PetscInt        qPoints;
      const PetscReal    *p, *w;
      PetscQuadrature quad;

      ierr = PetscDualSpaceGetFunctional(space,i,&quad);CHKERRQ(ierr);
      ierr = PetscQuadratureGetData(quad,NULL,&qPoints,&p,&w);CHKERRQ(ierr);
      ierr = PetscMemcpy(weights+offset,w,qPoints*sizeof(*w));CHKERRQ(ierr);
      ierr = PetscMemcpy(pointsRef+spdim*offset,p,spdim*qPoints*sizeof(*p));CHKERRQ(ierr);
      offset += qPoints;
    }
    if (id == PETSCFE_CLASSID) {
      ierr = PetscFEGetTabulation(fe,nPoints,pointsRef,&B,NULL,NULL);CHKERRQ(ierr);
    }
    else {
      ierr = PetscFVGetTabulation(fv,nPoints,pointsRef,&B,NULL,NULL);CHKERRQ(ierr);
    }
    offset = 0;
    for (i = 0; i < fSize; i++) {
      PetscInt        qPoints;
      PetscQuadrature quad;

      ierr = PetscDualSpaceGetFunctional(space,i,&quad);CHKERRQ(ierr);
      ierr = PetscQuadratureGetData(quad,NULL,&qPoints,NULL,NULL);CHKERRQ(ierr);
      for (j = 0; j < fSize; j++) {
        PetscScalar val = 0.;

        for (k = 0; k < qPoints; k++) {
          val += B[((offset + k) * fSize + j) * fComp] * weights[k];
        }
        ierr = MatSetValue(Amat,i,j,val,INSERT_VALUES);CHKERRQ(ierr);
      }
      offset += qPoints;
    }
    if (id == PETSCFE_CLASSID) {
      ierr = PetscFERestoreTabulation(fe,nPoints,pointsRef,&B,NULL,NULL);CHKERRQ(ierr);
    }
    else {
      ierr = PetscFVRestoreTabulation(fv,nPoints,pointsRef,&B,NULL,NULL);CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(Amat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Amat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatLUFactor(Amat,NULL,NULL,NULL);CHKERRQ(ierr);
    for (c = cStart; c < cEnd; c++) {
      PetscInt parent;
      PetscInt closureSize, closureSizeP, *closure = NULL, *closureP = NULL;
      PetscInt *childOffsets, *parentOffsets;

      ierr = DMPlexGetTreeParent(dm,c,&parent,NULL);CHKERRQ(ierr);
      if (parent == c) continue;
      ierr = DMPlexGetTransitiveClosure(dm,c,PETSC_TRUE,&closureSize,&closure);CHKERRQ(ierr);
      for (i = 0; i < closureSize; i++) {
        PetscInt p = closure[2*i];
        PetscInt conDof;

        if (p < conStart || p >= conEnd) continue;
        if (numFields > 1) {
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
        CoordinatesRefToReal(spdim, spdim, v0, J, &pointsRef[i*spdim],vtmp);CHKERRQ(ierr);
        CoordinatesRealToRef(spdim, spdim, v0parent, invJparent, vtmp, &pointsReal[i*spdim]);CHKERRQ(ierr);
      }
      if (id == PETSCFE_CLASSID) {
        ierr = PetscFEGetTabulation(fe,nPoints,pointsReal,&B,NULL,NULL);CHKERRQ(ierr);
      }
      else {
        ierr = PetscFVGetTabulation(fv,nPoints,pointsReal,&B,NULL,NULL);CHKERRQ(ierr);
      }
      offset = 0;
      for (i = 0; i < fSize; i++) {
        PetscInt        qPoints;
        PetscQuadrature quad;

        ierr = PetscDualSpaceGetFunctional(space,i,&quad);CHKERRQ(ierr);
        ierr = PetscQuadratureGetData(quad,NULL,&qPoints,NULL,NULL);CHKERRQ(ierr);
        for (j = 0; j < fSize; j++) {
          PetscScalar val = 0.;

          for (k = 0; k < qPoints; k++) {
            val += B[((offset + k) * fSize + j) * fComp] * weights[k];
          }
          MatSetValue(Bmat,i,j,val,INSERT_VALUES);CHKERRQ(ierr);
        }
        offset += qPoints;
      }
      if (id == PETSCFE_CLASSID) {
        ierr = PetscFERestoreTabulation(fe,nPoints,pointsReal,&B,NULL,NULL);CHKERRQ(ierr);
      }
      else {
        ierr = PetscFVRestoreTabulation(fv,nPoints,pointsReal,&B,NULL,NULL);CHKERRQ(ierr);
      }
      ierr = MatAssemblyBegin(Bmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(Bmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatMatSolve(Amat,Bmat,Xmat);CHKERRQ(ierr);
      ierr = DMPlexGetTransitiveClosure(dm,parent,PETSC_TRUE,&closureSizeP,&closureP);CHKERRQ(ierr);
      ierr = PetscMalloc2(closureSize+1,&childOffsets,closureSizeP+1,&parentOffsets);CHKERRQ(ierr);
      childOffsets[0] = 0;
      for (i = 0; i < closureSize; i++) {
        PetscInt p = closure[2*i];
        PetscInt dof;

        if (numFields > 1) {
          ierr = PetscSectionGetFieldDof(section,p,f,&dof);CHKERRQ(ierr);
        }
        else {
          ierr = PetscSectionGetDof(section,p,&dof);CHKERRQ(ierr);
        }
        childOffsets[i+1]=childOffsets[i]+dof / fComp;
      }
      parentOffsets[0] = 0;
      for (i = 0; i < closureSizeP; i++) {
        PetscInt p = closureP[2*i];
        PetscInt dof;

        if (numFields > 1) {
          ierr = PetscSectionGetFieldDof(section,p,f,&dof);CHKERRQ(ierr);
        }
        else {
          ierr = PetscSectionGetDof(section,p,&dof);CHKERRQ(ierr);
        }
        parentOffsets[i+1]=parentOffsets[i]+dof / fComp;
      }
      for (i = 0; i < closureSize; i++) {
        PetscInt conDof, conOff, aDof, aOff;
        PetscInt p = closure[2*i];
        PetscInt o = closure[2*i+1];

        if (p < conStart || p >= conEnd) continue;
        if (numFields > 1) {
          ierr = PetscSectionGetFieldDof(cSec,p,f,&conDof);CHKERRQ(ierr);
          ierr = PetscSectionGetFieldOffset(cSec,p,f,&conOff);CHKERRQ(ierr);
        }
        else {
          ierr = PetscSectionGetDof(cSec,p,&conDof);CHKERRQ(ierr);
          ierr = PetscSectionGetOffset(cSec,p,&conOff);CHKERRQ(ierr);
        }
        if (!conDof) continue;
        ierr = PetscSectionGetDof(aSec,p,&aDof);CHKERRQ(ierr);
        ierr = PetscSectionGetOffset(aSec,p,&aOff);CHKERRQ(ierr);
        for (k = 0; k < aDof; k++) {
          PetscInt a = anchors[aOff + k];
          PetscInt aSecDof, aSecOff;

          if (numFields > 1) {
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
              PetscInt r, s, t;

              for (r = childOffsets[i]; r < childOffsets[i+1]; r++) {
                for (s = parentOffsets[j]; s < parentOffsets[j+1]; s++) {
                  PetscScalar val;
                  PetscInt insertCol, insertRow;

                  ierr = MatGetValue(Xmat,r,s,&val);CHKERRQ(ierr);
                  if (o >= 0) {
                    insertRow = conOff + fComp * (r - childOffsets[i]);
                  }
                  else {
                    insertRow = conOff + fComp * (childOffsets[i + 1] - 1 - r);
                  }
                  if (oq >= 0) {
                    insertCol = aSecOff + fComp * (s - parentOffsets[j]);
                  }
                  else {
                    insertCol = aSecOff + fComp * (parentOffsets[j + 1] - 1 - s);
                  }
                  for (t = 0; t < fComp; t++) {
                    ierr = MatSetValue(cMat,insertRow + t,insertCol + t,val,INSERT_VALUES);CHKERRQ(ierr);
                  }
                }
              }
            }
          }
        }
      }
      ierr = PetscFree2(childOffsets,parentOffsets);CHKERRQ(ierr);
      ierr = DMPlexRestoreTransitiveClosure(dm,c,PETSC_TRUE,&closureSize,&closure);CHKERRQ(ierr);
      ierr = DMPlexRestoreTransitiveClosure(dm,parent,PETSC_TRUE,&closureSizeP,&closureP);CHKERRQ(ierr);
    }
    ierr = MatDestroy(&Amat);CHKERRQ(ierr);
    ierr = MatDestroy(&Bmat);CHKERRQ(ierr);
    ierr = MatDestroy(&Xmat);CHKERRQ(ierr);
    ierr = PetscFree3(weights,pointsRef,pointsReal);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(cMat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(cMat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscFree6(v0,v0parent,vtmp,J,Jparent,invJparent);CHKERRQ(ierr);
  ierr = ISRestoreIndices(aIS,&anchors);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexComputeAnchorMatrix_Tree_FromReference"
static PetscErrorCode DMPlexComputeAnchorMatrix_Tree_FromReference(DM dm, PetscSection section, PetscSection conSec, Mat cMat)
{
  DM             refTree;
  PetscDS        ds;
  Mat            refCmat;
  PetscInt       numFields, f, pRefStart, pRefEnd, p, *rows, *cols, maxDof, maxAnDof, *perm, *iperm, pStart, pEnd, conStart, conEnd, **refPointFieldN;
  PetscScalar ***refPointFieldMats, *pointWork;
  PetscSection   refConSec, refAnSec, anSec, refSection;
  IS             refAnIS, anIS;
  const PetscInt *refAnchors, *anchors;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMGetDS(dm,&ds);CHKERRQ(ierr);
  ierr = PetscDSGetNumFields(ds,&numFields);CHKERRQ(ierr);
  ierr = DMPlexGetReferenceTree(dm,&refTree);CHKERRQ(ierr);
  ierr = DMSetDS(refTree,ds);CHKERRQ(ierr);
  ierr = DMGetDefaultConstraints(refTree,&refConSec,&refCmat);CHKERRQ(ierr);
  ierr = DMPlexGetChart(refTree,&pRefStart,&pRefEnd);CHKERRQ(ierr);
  ierr = DMPlexGetAnchors(refTree,&refAnSec,&refAnIS);CHKERRQ(ierr);
  ierr = DMPlexGetAnchors(dm,&anSec,&anIS);CHKERRQ(ierr);
  ierr = ISGetIndices(refAnIS,&refAnchors);CHKERRQ(ierr);
  ierr = ISGetIndices(anIS,&anchors);CHKERRQ(ierr);
  ierr = DMGetDefaultSection(refTree,&refSection);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(refConSec,&pRefStart,&pRefEnd);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(conSec,&conStart,&conEnd);CHKERRQ(ierr);
  ierr = PetscMalloc1(pRefEnd-pRefStart,&refPointFieldMats);CHKERRQ(ierr);
  ierr = PetscMalloc1(pRefEnd-pRefStart,&refPointFieldN);CHKERRQ(ierr);
  ierr = PetscSectionGetMaxDof(refConSec,&maxDof);CHKERRQ(ierr);
  ierr = PetscSectionGetMaxDof(refAnSec,&maxAnDof);CHKERRQ(ierr);
  ierr = PetscMalloc1(maxDof,&rows);CHKERRQ(ierr);
  ierr = PetscMalloc1(maxDof*maxAnDof,&cols);CHKERRQ(ierr);
  ierr = PetscMalloc1(maxDof*maxDof*maxAnDof,&pointWork);CHKERRQ(ierr);

  /* step 1: get submats for every constrained point in the reference tree */
  for (p = pRefStart; p < pRefEnd; p++) {
    PetscInt parent, closureSize, *closure = NULL, pDof;

    ierr = DMPlexGetTreeParent(refTree,p,&parent,NULL);CHKERRQ(ierr);
    ierr = PetscSectionGetDof(refConSec,p,&pDof);CHKERRQ(ierr);
    if (!pDof || parent == p) continue;

    ierr = PetscMalloc1(numFields,&refPointFieldMats[p-pRefStart]);CHKERRQ(ierr);
    ierr = PetscCalloc1(numFields,&refPointFieldN[p-pRefStart]);CHKERRQ(ierr);
    ierr = DMPlexGetTransitiveClosure(refTree,parent,PETSC_TRUE,&closureSize,&closure);CHKERRQ(ierr);
    for (f = 0; f < numFields; f++) {
      PetscInt cDof, cOff, numCols, r, i, fComp;
      PetscObject disc;
      PetscClassId id;
      PetscFE fe = NULL;
      PetscFV fv = NULL;

      ierr = PetscDSGetDiscretization(ds,f,&disc);CHKERRQ(ierr);
      ierr = PetscObjectGetClassId(disc,&id);CHKERRQ(ierr);
      if (id == PETSCFE_CLASSID) {
        fe = (PetscFE) disc;
        ierr = PetscFEGetNumComponents(fe,&fComp);CHKERRQ(ierr);
      }
      else if (id == PETSCFV_CLASSID) {
        fv = (PetscFV) disc;
        ierr = PetscFVGetNumComponents(fv,&fComp);CHKERRQ(ierr);
      }
      else SETERRQ1(PetscObjectComm(disc),PETSC_ERR_ARG_UNKNOWN_TYPE, "PetscDS discretization id %d not recognized.", id);

      if (numFields > 1) {
        ierr = PetscSectionGetFieldDof(refConSec,p,f,&cDof);CHKERRQ(ierr);
        ierr = PetscSectionGetFieldOffset(refConSec,p,f,&cOff);CHKERRQ(ierr);
      }
      else {
        ierr = PetscSectionGetDof(refConSec,p,&cDof);CHKERRQ(ierr);
        ierr = PetscSectionGetOffset(refConSec,p,&cOff);CHKERRQ(ierr);
      }

      if (!cDof) continue;
      for (r = 0; r < cDof; r++) {
        rows[r] = cOff + r;
      }
      numCols = 0;
      for (i = 0; i < closureSize; i++) {
        PetscInt q = closure[2*i];
        PetscInt o = closure[2*i+1];
        PetscInt aDof, aOff, j;

        if (numFields > 1) {
          ierr = PetscSectionGetFieldDof(refSection,q,f,&aDof);CHKERRQ(ierr);
          ierr = PetscSectionGetFieldOffset(refSection,q,f,&aOff);CHKERRQ(ierr);
        }
        else {
          ierr = PetscSectionGetDof(refSection,q,&aDof);CHKERRQ(ierr);
          ierr = PetscSectionGetOffset(refSection,q,&aOff);CHKERRQ(ierr);
        }

        for (j = 0; j < aDof; j++) {
          PetscInt node = (o >= 0) ? (j / fComp) : ((aDof - 1 - j) / fComp);
          PetscInt comp = (j % fComp);

          cols[numCols++] = aOff + node * fComp + comp;
        }
      }
      refPointFieldN[p-pRefStart][f] = numCols;
      ierr = PetscMalloc1(cDof*numCols,&refPointFieldMats[p-pRefStart][f]);CHKERRQ(ierr);
      ierr = MatGetValues(refCmat,cDof,rows,numCols,cols,refPointFieldMats[p-pRefStart][f]);CHKERRQ(ierr);
    }
    ierr = DMPlexRestoreTransitiveClosure(refTree,parent,PETSC_TRUE,&closureSize,&closure);CHKERRQ(ierr);
  }

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
      PetscInt parent, childid, closureSize, *closure = NULL;
      PetscInt point = perm[p], pointDof;

      ierr = DMPlexGetTreeParent(dm,point,&parent,&childid);CHKERRQ(ierr);
      if ((point < conStart) || (point >= conEnd) || (parent == point)) continue;
      ierr = PetscSectionGetDof(conSec,point,&pointDof);CHKERRQ(ierr);
      if (!pointDof) continue;
      ierr = DMPlexGetTransitiveClosure(dm,parent,PETSC_TRUE,&closureSize,&closure);CHKERRQ(ierr);
      for (f = 0; f < numFields; f++) {
        PetscInt cDof, cOff, numCols, numFillCols, i, r, fComp, matOffset, offset;
        PetscScalar *pointMat;
        PetscObject disc;
        PetscClassId id;
        PetscFE fe = NULL;
        PetscFV fv = NULL;

        ierr = PetscDSGetDiscretization(ds,f,&disc);CHKERRQ(ierr);
        ierr = PetscObjectGetClassId(disc,&id);CHKERRQ(ierr);
        if (id == PETSCFE_CLASSID) {
          fe = (PetscFE) disc;
          ierr = PetscFEGetNumComponents(fe,&fComp);CHKERRQ(ierr);
        }
        else if (id == PETSCFV_CLASSID) {
          fv = (PetscFV) disc;
          ierr = PetscFVGetNumComponents(fv,&fComp);CHKERRQ(ierr);
        }

        if (numFields > 1) {
          ierr = PetscSectionGetFieldDof(conSec,point,f,&cDof);CHKERRQ(ierr);
          ierr = PetscSectionGetFieldOffset(conSec,point,f,&cOff);CHKERRQ(ierr);
        }
        else {
          ierr = PetscSectionGetDof(conSec,point,&cDof);CHKERRQ(ierr);
          ierr = PetscSectionGetOffset(conSec,point,&cOff);CHKERRQ(ierr);
        }
        if (!cDof) continue;

        /* make sure that every row for this point is the same size */
#if defined(PETSC_USE_DEBUG)
        for (r = 0; r < cDof; r++) {
          if (cDof > 1 && r) {
            if ((ia[cOff+r+1]-ia[cOff+r]) != (ia[cOff+r]-ia[cOff+r-1])) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Two point rows have different nnz: %D vs. %D", (ia[rows[r]+1]-ia[rows[r]]), (ia[rows[r]]-ia[rows[r]-1]));
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
          PetscInt o = closure[2*i+1];
          PetscInt aDof, aOff, j, k, qConDof, qConOff;

          qConDof = qConOff = 0;
          if (numFields > 1) {
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
                  PetscInt node = (o >= 0) ? (k / fComp) : ((aDof - 1 - k) / fComp);
                  PetscInt comp = (k % fComp);
                  PetscInt col  = node * fComp + comp;

                  inVal += pointMat[r * numCols + offset + col] * vals[aMatOffset + aNumFillCols * k + j];
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
            for (j = 0; j < aDof; j++) {
              PetscInt node = (o >= 0) ? (j / fComp) : ((aDof - 1 - j) / fComp);
              PetscInt comp = (j % fComp);
              PetscInt col  = node * fComp + comp;
              for (r = 0; r < cDof; r++) {
                vals[matOffset + numFillCols * r + k + col] += pointMat[r * numCols + offset + col];
              }
            }
          }
          offset += aDof;
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
  ierr = ISRestoreIndices(refAnIS,&refAnchors);CHKERRQ(ierr);
  ierr = ISRestoreIndices(anIS,&anchors);CHKERRQ(ierr);
  ierr = PetscFree2(perm,iperm);CHKERRQ(ierr);
  ierr = PetscFree(rows);CHKERRQ(ierr);
  ierr = PetscFree(cols);CHKERRQ(ierr);
  ierr = PetscFree(pointWork);CHKERRQ(ierr);
  for (p = pRefStart; p < pRefEnd; p++) {
    PetscInt parent, pDof;

    ierr = DMPlexGetTreeParent(refTree,p,&parent,NULL);CHKERRQ(ierr);
    ierr = PetscSectionGetDof(refConSec,p,&pDof);CHKERRQ(ierr);
    if (!pDof || parent == p) continue;

    for (f = 0; f < numFields; f++) {
      PetscInt cDof;

      if (numFields > 1) {
        ierr = PetscSectionGetFieldDof(refConSec,p,f,&cDof);CHKERRQ(ierr);
      }
      else {
        ierr = PetscSectionGetDof(refConSec,p,&cDof);CHKERRQ(ierr);
      }

      if (!cDof) continue;
      ierr = PetscFree(refPointFieldMats[p - pRefStart][f]);CHKERRQ(ierr);
    }
    ierr = PetscFree(refPointFieldMats[p - pRefStart]);CHKERRQ(ierr);
    ierr = PetscFree(refPointFieldN[p - pRefStart]);CHKERRQ(ierr);
  }
  ierr = PetscFree(refPointFieldMats);CHKERRQ(ierr);
  ierr = PetscFree(refPointFieldN);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexTreeRefineCell"
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

        ierr = DMPlexGetTreeParent(K,v,&kParent,NULL);CHKERRQ(ierr);
        if (kParent != v) {
          /* this is a new vertex */
          ierr = PetscSectionGetOffset(vSection,vPerm,&off);CHKERRQ(ierr);
          for (l = 0; l < dim; ++l) coord[l] = PetscRealPart(coordvals[off+l]);
          CoordinatesRefToReal(dim, dim, v0, J, coord, newCoord);CHKERRQ(ierr);
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
