#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/
#include <petsc/private/isimpl.h>
#include <petsc/private/petscfeimpl.h>
#include <petscsf.h>
#include <petscds.h>

/* hierarchy routines */

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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (ref) {PetscValidHeaderSpecific(ref, DM_CLASSID, 2);}
  CHKERRQ(PetscObjectReference((PetscObject)ref));
  CHKERRQ(DMDestroy(&mesh->referenceTree));
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

  PetscFunctionBegin;
  if (parentOrientA == parentOrientB) {
    if (childOrientB) *childOrientB = childOrientA;
    if (childB) *childB = childA;
    PetscFunctionReturn(0);
  }
  for (dim = 0; dim < 3; dim++) {
    CHKERRQ(DMPlexGetDepthStratum(dm,dim,&dStart,&dEnd));
    if (parent >= dStart && parent <= dEnd) {
      break;
    }
  }
  PetscCheckFalse(dim > 2,PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot perform child symmetry for %d-cells",dim);
  PetscCheckFalse(!dim,PETSC_COMM_SELF,PETSC_ERR_PLIB,"A vertex has no children");
  if (childA < dStart || childA >= dEnd) {
    /* this is a lower-dimensional child: bootstrap */
    PetscInt size, i, sA = -1, sB, sOrientB, sConeSize;
    const PetscInt *supp, *coneA, *coneB, *oA, *oB;

    CHKERRQ(DMPlexGetSupportSize(dm,childA,&size));
    CHKERRQ(DMPlexGetSupport(dm,childA,&supp));

    /* find a point sA in supp(childA) that has the same parent */
    for (i = 0; i < size; i++) {
      PetscInt sParent;

      sA   = supp[i];
      if (sA == parent) continue;
      CHKERRQ(DMPlexGetTreeParent(dm,sA,&sParent,NULL));
      if (sParent == parent) {
        break;
      }
    }
    PetscCheckFalse(i == size,PETSC_COMM_SELF,PETSC_ERR_PLIB,"could not find support in children");
    /* find out which point sB is in an equivalent position to sA under
     * parentOrientB */
    CHKERRQ(DMPlexReferenceTreeGetChildSymmetry_Default(dm,parent,parentOrientA,0,sA,parentOrientB,&sOrientB,&sB));
    CHKERRQ(DMPlexGetConeSize(dm,sA,&sConeSize));
    CHKERRQ(DMPlexGetCone(dm,sA,&coneA));
    CHKERRQ(DMPlexGetCone(dm,sB,&coneB));
    CHKERRQ(DMPlexGetConeOrientation(dm,sA,&oA));
    CHKERRQ(DMPlexGetConeOrientation(dm,sB,&oB));
    /* step through the cone of sA in natural order */
    for (i = 0; i < sConeSize; i++) {
      if (coneA[i] == childA) {
        /* if childA is at position i in coneA,
         * then we want the point that is at sOrientB*i in coneB */
        PetscInt j = (sOrientB >= 0) ? ((sOrientB + i) % sConeSize) : ((sConeSize -(sOrientB+1) - i) % sConeSize);
        if (childB) *childB = coneB[j];
        if (childOrientB) {
          DMPolytopeType ct;
          PetscInt       oBtrue;

          CHKERRQ(DMPlexGetConeSize(dm,childA,&coneSize));
          /* compose sOrientB and oB[j] */
          PetscCheckFalse(coneSize != 0 && coneSize != 2,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Expected a vertex or an edge");
          ct = coneSize ? DM_POLYTOPE_SEGMENT : DM_POLYTOPE_POINT;
          /* we may have to flip an edge */
          oBtrue        = (sOrientB >= 0) ? oB[j] : DMPolytopeTypeComposeOrientation(ct, -1, oB[j]);
          oBtrue        = DMPolytopeConvertNewOrientation_Internal(ct, oBtrue);
          ABswap        = DihedralSwap(coneSize,DMPolytopeConvertNewOrientation_Internal(ct, oA[i]),oBtrue);
          *childOrientB = DihedralCompose(coneSize,childOrientA,ABswap);
        }
        break;
      }
    }
    PetscCheckFalse(i == sConeSize,PETSC_COMM_SELF,PETSC_ERR_PLIB,"support cone mismatch");
    PetscFunctionReturn(0);
  }
  /* get the cone size and symmetry swap */
  CHKERRQ(DMPlexGetConeSize(dm,parent,&coneSize));
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
    CHKERRQ(DMPlexGetTreeChildren(dm,parent,&numChildren,&children));
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
      PetscCheckFalse(dim != 2 || posA != 3,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Expected a middle triangle, got something else");
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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCheckFalse(!mesh->getchildsymmetry,PETSC_COMM_SELF,PETSC_ERR_SUP,"DMPlexReferenceTreeGetChildSymmetry not implemented");
  CHKERRQ(mesh->getchildsymmetry(dm,parent,parentOrientA,childOrientA,childA,parentOrientB,childOrientB,childB));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexSetTree_Internal(DM,PetscSection,PetscInt*,PetscInt*,PetscBool,PetscBool);

PetscErrorCode DMPlexCreateReferenceTree_SetTree(DM dm, PetscSection parentSection, PetscInt parents[], PetscInt childIDs[])
{
  PetscFunctionBegin;
  CHKERRQ(DMPlexSetTree_Internal(dm,parentSection,parents,childIDs,PETSC_TRUE,PETSC_FALSE));
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

  PetscFunctionBegin;
  comm = PetscObjectComm((PetscObject)K);
  CHKERRQ(DMGetDimension(K, &dim));
  CHKERRQ(DMPlexGetChart(K, &pStart, &pEnd));
  CHKERRQ(DMGetLabel(K, labelName, &identity));
  CHKERRQ(DMGetLabel(Kref, labelName, &identityRef));
  CHKERRQ(DMPlexGetChart(Kref, &pRefStart, &pRefEnd));
  CHKERRQ(PetscSectionCreate(comm, &unionSection));
  CHKERRQ(PetscSectionSetChart(unionSection, 0, (pEnd - pStart) + (pRefEnd - pRefStart)));
  /* count points that will go in the union */
  for (p = pStart; p < pEnd; p++) {
    CHKERRQ(PetscSectionSetDof(unionSection, p - pStart, 1));
  }
  for (p = pRefStart; p < pRefEnd; p++) {
    PetscInt q, qSize;
    CHKERRQ(DMLabelGetValue(identityRef, p, &q));
    CHKERRQ(DMLabelGetStratumSize(identityRef, q, &qSize));
    if (qSize > 1) {
      CHKERRQ(PetscSectionSetDof(unionSection, p - pRefStart + (pEnd - pStart), 1));
    }
  }
  CHKERRQ(PetscMalloc1(pEnd - pStart + pRefEnd - pRefStart,&permvals));
  offset = 0;
  /* stratify points in the union by topological dimension */
  for (d = 0; d <= dim; d++) {
    PetscInt cStart, cEnd, c;

    CHKERRQ(DMPlexGetHeightStratum(K, d, &cStart, &cEnd));
    for (c = cStart; c < cEnd; c++) {
      permvals[offset++] = c;
    }

    CHKERRQ(DMPlexGetHeightStratum(Kref, d, &cStart, &cEnd));
    for (c = cStart; c < cEnd; c++) {
      permvals[offset++] = c + (pEnd - pStart);
    }
  }
  CHKERRQ(ISCreateGeneral(comm, (pEnd - pStart) + (pRefEnd - pRefStart), permvals, PETSC_OWN_POINTER, &perm));
  CHKERRQ(PetscSectionSetPermutation(unionSection,perm));
  CHKERRQ(PetscSectionSetUp(unionSection));
  CHKERRQ(PetscSectionGetStorageSize(unionSection,&numUnionPoints));
  CHKERRQ(PetscMalloc2(numUnionPoints,&coneSizes,dim+1,&numDimPoints));
  /* count dimension points */
  for (d = 0; d <= dim; d++) {
    PetscInt cStart, cOff, cOff2;
    CHKERRQ(DMPlexGetHeightStratum(K,d,&cStart,NULL));
    CHKERRQ(PetscSectionGetOffset(unionSection,cStart-pStart,&cOff));
    if (d < dim) {
      CHKERRQ(DMPlexGetHeightStratum(K,d+1,&cStart,NULL));
      CHKERRQ(PetscSectionGetOffset(unionSection,cStart-pStart,&cOff2));
    }
    else {
      cOff2 = numUnionPoints;
    }
    numDimPoints[dim - d] = cOff2 - cOff;
  }
  CHKERRQ(PetscSectionCreate(comm, &unionConeSection));
  CHKERRQ(PetscSectionSetChart(unionConeSection, 0, numUnionPoints));
  /* count the cones in the union */
  for (p = pStart; p < pEnd; p++) {
    PetscInt dof, uOff;

    CHKERRQ(DMPlexGetConeSize(K, p, &dof));
    CHKERRQ(PetscSectionGetOffset(unionSection, p - pStart,&uOff));
    CHKERRQ(PetscSectionSetDof(unionConeSection, uOff, dof));
    coneSizes[uOff] = dof;
  }
  for (p = pRefStart; p < pRefEnd; p++) {
    PetscInt dof, uDof, uOff;

    CHKERRQ(DMPlexGetConeSize(Kref, p, &dof));
    CHKERRQ(PetscSectionGetDof(unionSection, p - pRefStart + (pEnd - pStart),&uDof));
    CHKERRQ(PetscSectionGetOffset(unionSection, p - pRefStart + (pEnd - pStart),&uOff));
    if (uDof) {
      CHKERRQ(PetscSectionSetDof(unionConeSection, uOff, dof));
      coneSizes[uOff] = dof;
    }
  }
  CHKERRQ(PetscSectionSetUp(unionConeSection));
  CHKERRQ(PetscSectionGetStorageSize(unionConeSection,&numCones));
  CHKERRQ(PetscMalloc2(numCones,&unionCones,numCones,&unionOrientations));
  /* write the cones in the union */
  for (p = pStart; p < pEnd; p++) {
    PetscInt dof, uOff, c, cOff;
    const PetscInt *cone, *orientation;

    CHKERRQ(DMPlexGetConeSize(K, p, &dof));
    CHKERRQ(DMPlexGetCone(K, p, &cone));
    CHKERRQ(DMPlexGetConeOrientation(K, p, &orientation));
    CHKERRQ(PetscSectionGetOffset(unionSection, p - pStart,&uOff));
    CHKERRQ(PetscSectionGetOffset(unionConeSection,uOff,&cOff));
    for (c = 0; c < dof; c++) {
      PetscInt e, eOff;
      e                           = cone[c];
      CHKERRQ(PetscSectionGetOffset(unionSection, e - pStart, &eOff));
      unionCones[cOff + c]        = eOff;
      unionOrientations[cOff + c] = orientation[c];
    }
  }
  for (p = pRefStart; p < pRefEnd; p++) {
    PetscInt dof, uDof, uOff, c, cOff;
    const PetscInt *cone, *orientation;

    CHKERRQ(DMPlexGetConeSize(Kref, p, &dof));
    CHKERRQ(DMPlexGetCone(Kref, p, &cone));
    CHKERRQ(DMPlexGetConeOrientation(Kref, p, &orientation));
    CHKERRQ(PetscSectionGetDof(unionSection, p - pRefStart + (pEnd - pStart),&uDof));
    CHKERRQ(PetscSectionGetOffset(unionSection, p - pRefStart + (pEnd - pStart),&uOff));
    if (uDof) {
      CHKERRQ(PetscSectionGetOffset(unionConeSection,uOff,&cOff));
      for (c = 0; c < dof; c++) {
        PetscInt e, eOff, eDof;

        e    = cone[c];
        CHKERRQ(PetscSectionGetDof(unionSection, e - pRefStart + (pEnd - pStart),&eDof));
        if (eDof) {
          CHKERRQ(PetscSectionGetOffset(unionSection, e - pRefStart + (pEnd - pStart), &eOff));
        }
        else {
          CHKERRQ(DMLabelGetValue(identityRef, e, &e));
          CHKERRQ(PetscSectionGetOffset(unionSection, e - pStart, &eOff));
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

    CHKERRQ(DMGetCoordinateSection(K, &KcoordsSec));
    CHKERRQ(DMGetCoordinatesLocal(K, &KcoordsVec));
    CHKERRQ(DMGetCoordinateSection(Kref, &KrefCoordsSec));
    CHKERRQ(DMGetCoordinatesLocal(Kref, &KrefCoordsVec));

    numVerts = numDimPoints[0];
    CHKERRQ(PetscMalloc1(numVerts * dim,&unionCoords));
    CHKERRQ(DMPlexGetDepthStratum(K,0,&vStart,&vEnd));

    offset = 0;
    for (v = vStart; v < vEnd; v++) {
      CHKERRQ(PetscSectionGetOffset(unionSection,v - pStart,&vOff));
      CHKERRQ(VecGetValuesSection(KcoordsVec, KcoordsSec, v, &Kcoords));
      for (d = 0; d < dim; d++) {
        unionCoords[offset * dim + d] = Kcoords[d];
      }
      offset++;
    }
    CHKERRQ(DMPlexGetDepthStratum(Kref,0,&vRefStart,&vRefEnd));
    for (v = vRefStart; v < vRefEnd; v++) {
      CHKERRQ(PetscSectionGetDof(unionSection,v - pRefStart + (pEnd - pStart),&vDof));
      CHKERRQ(PetscSectionGetOffset(unionSection,v - pRefStart + (pEnd - pStart),&vOff));
      CHKERRQ(VecGetValuesSection(KrefCoordsVec, KrefCoordsSec, v, &Kcoords));
      if (vDof) {
        for (d = 0; d < dim; d++) {
          unionCoords[offset * dim + d] = Kcoords[d];
        }
        offset++;
      }
    }
  }
  CHKERRQ(DMCreate(comm,ref));
  CHKERRQ(DMSetType(*ref,DMPLEX));
  CHKERRQ(DMSetDimension(*ref,dim));
  CHKERRQ(DMPlexCreateFromDAG(*ref,dim,numDimPoints,coneSizes,unionCones,unionOrientations,unionCoords));
  /* set the tree */
  CHKERRQ(PetscSectionCreate(comm,&parentSection));
  CHKERRQ(PetscSectionSetChart(parentSection,0,numUnionPoints));
  for (p = pRefStart; p < pRefEnd; p++) {
    PetscInt uDof, uOff;

    CHKERRQ(PetscSectionGetDof(unionSection, p - pRefStart + (pEnd - pStart),&uDof));
    CHKERRQ(PetscSectionGetOffset(unionSection, p - pRefStart + (pEnd - pStart),&uOff));
    if (uDof) {
      CHKERRQ(PetscSectionSetDof(parentSection,uOff,1));
    }
  }
  CHKERRQ(PetscSectionSetUp(parentSection));
  CHKERRQ(PetscSectionGetStorageSize(parentSection,&parentSize));
  CHKERRQ(PetscMalloc2(parentSize,&parents,parentSize,&childIDs));
  for (p = pRefStart; p < pRefEnd; p++) {
    PetscInt uDof, uOff;

    CHKERRQ(PetscSectionGetDof(unionSection, p - pRefStart + (pEnd - pStart),&uDof));
    CHKERRQ(PetscSectionGetOffset(unionSection, p - pRefStart + (pEnd - pStart),&uOff));
    if (uDof) {
      PetscInt pOff, parent, parentU;
      CHKERRQ(PetscSectionGetOffset(parentSection,uOff,&pOff));
      CHKERRQ(DMLabelGetValue(identityRef,p,&parent));
      CHKERRQ(PetscSectionGetOffset(unionSection, parent - pStart,&parentU));
      parents[pOff] = parentU;
      childIDs[pOff] = uOff;
    }
  }
  CHKERRQ(DMPlexCreateReferenceTree_SetTree(*ref,parentSection,parents,childIDs));
  CHKERRQ(PetscSectionDestroy(&parentSection));
  CHKERRQ(PetscFree2(parents,childIDs));

  /* clean up */
  CHKERRQ(PetscSectionDestroy(&unionSection));
  CHKERRQ(PetscSectionDestroy(&unionConeSection));
  CHKERRQ(ISDestroy(&perm));
  CHKERRQ(PetscFree(unionCoords));
  CHKERRQ(PetscFree2(unionCones,unionOrientations));
  CHKERRQ(PetscFree2(coneSizes,numDimPoints));
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

  PetscFunctionBegin;
#if 1
  comm = PETSC_COMM_SELF;
#endif
  /* create a reference element */
  CHKERRQ(DMPlexCreateReferenceCell(comm, DMPolytopeTypeSimpleShape(dim, simplex), &K));
  CHKERRQ(DMCreateLabel(K, "identity"));
  CHKERRQ(DMGetLabel(K, "identity", &identity));
  CHKERRQ(DMPlexGetChart(K, &pStart, &pEnd));
  for (p = pStart; p < pEnd; p++) {
    CHKERRQ(DMLabelSetValue(identity, p, p));
  }
  /* refine it */
  CHKERRQ(DMRefine(K,comm,&Kref));

  /* the reference tree is the union of these two, without duplicating
   * points that appear in both */
  CHKERRQ(DMPlexCreateReferenceTree_Union(K, Kref, "identity", ref));
  mesh = (DM_Plex *) (*ref)->data;
  mesh->getchildsymmetry = DMPlexReferenceTreeGetChildSymmetry_Default;
  CHKERRQ(DMDestroy(&K));
  CHKERRQ(DMDestroy(&Kref));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTreeSymmetrize(DM dm)
{
  DM_Plex        *mesh = (DM_Plex *)dm->data;
  PetscSection   childSec, pSec;
  PetscInt       p, pSize, cSize, parMax = PETSC_MIN_INT, parMin = PETSC_MAX_INT;
  PetscInt       *offsets, *children, pStart, pEnd;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  CHKERRQ(PetscSectionDestroy(&mesh->childSection));
  CHKERRQ(PetscFree(mesh->children));
  pSec = mesh->parentSection;
  if (!pSec) PetscFunctionReturn(0);
  CHKERRQ(PetscSectionGetStorageSize(pSec,&pSize));
  for (p = 0; p < pSize; p++) {
    PetscInt par = mesh->parents[p];

    parMax = PetscMax(parMax,par+1);
    parMin = PetscMin(parMin,par);
  }
  if (parMin > parMax) {
    parMin = -1;
    parMax = -1;
  }
  CHKERRQ(PetscSectionCreate(PetscObjectComm((PetscObject)pSec),&childSec));
  CHKERRQ(PetscSectionSetChart(childSec,parMin,parMax));
  for (p = 0; p < pSize; p++) {
    PetscInt par = mesh->parents[p];

    CHKERRQ(PetscSectionAddDof(childSec,par,1));
  }
  CHKERRQ(PetscSectionSetUp(childSec));
  CHKERRQ(PetscSectionGetStorageSize(childSec,&cSize));
  CHKERRQ(PetscMalloc1(cSize,&children));
  CHKERRQ(PetscCalloc1(parMax-parMin,&offsets));
  CHKERRQ(PetscSectionGetChart(pSec,&pStart,&pEnd));
  for (p = pStart; p < pEnd; p++) {
    PetscInt dof, off, i;

    CHKERRQ(PetscSectionGetDof(pSec,p,&dof));
    CHKERRQ(PetscSectionGetOffset(pSec,p,&off));
    for (i = 0; i < dof; i++) {
      PetscInt par = mesh->parents[off + i], cOff;

      CHKERRQ(PetscSectionGetOffset(childSec,par,&cOff));
      children[cOff + offsets[par-parMin]++] = p;
    }
  }
  mesh->childSection = childSec;
  mesh->children = children;
  CHKERRQ(PetscFree(offsets));
  PetscFunctionReturn(0);
}

static PetscErrorCode AnchorsFlatten (PetscSection section, IS is, PetscSection *sectionNew, IS *isNew)
{
  PetscInt       pStart, pEnd, size, sizeNew, i, p, *valsNew = NULL;
  const PetscInt *vals;
  PetscSection   secNew;
  PetscBool      anyNew, globalAnyNew;
  PetscBool      compress;

  PetscFunctionBegin;
  CHKERRQ(PetscSectionGetChart(section,&pStart,&pEnd));
  CHKERRQ(ISGetLocalSize(is,&size));
  CHKERRQ(ISGetIndices(is,&vals));
  CHKERRQ(PetscSectionCreate(PetscObjectComm((PetscObject)section),&secNew));
  CHKERRQ(PetscSectionSetChart(secNew,pStart,pEnd));
  for (i = 0; i < size; i++) {
    PetscInt dof;

    p = vals[i];
    if (p < pStart || p >= pEnd) continue;
    CHKERRQ(PetscSectionGetDof(section, p, &dof));
    if (dof) break;
  }
  if (i == size) {
    CHKERRQ(PetscSectionSetUp(secNew));
    anyNew   = PETSC_FALSE;
    compress = PETSC_FALSE;
    sizeNew  = 0;
  }
  else {
    anyNew = PETSC_TRUE;
    for (p = pStart; p < pEnd; p++) {
      PetscInt dof, off;

      CHKERRQ(PetscSectionGetDof(section, p, &dof));
      CHKERRQ(PetscSectionGetOffset(section, p, &off));
      for (i = 0; i < dof; i++) {
        PetscInt q = vals[off + i], qDof = 0;

        if (q >= pStart && q < pEnd) {
          CHKERRQ(PetscSectionGetDof(section, q, &qDof));
        }
        if (qDof) {
          CHKERRQ(PetscSectionAddDof(secNew, p, qDof));
        }
        else {
          CHKERRQ(PetscSectionAddDof(secNew, p, 1));
        }
      }
    }
    CHKERRQ(PetscSectionSetUp(secNew));
    CHKERRQ(PetscSectionGetStorageSize(secNew,&sizeNew));
    CHKERRQ(PetscMalloc1(sizeNew,&valsNew));
    compress = PETSC_FALSE;
    for (p = pStart; p < pEnd; p++) {
      PetscInt dof, off, count, offNew, dofNew;

      CHKERRQ(PetscSectionGetDof(section, p, &dof));
      CHKERRQ(PetscSectionGetOffset(section, p, &off));
      CHKERRQ(PetscSectionGetDof(secNew, p, &dofNew));
      CHKERRQ(PetscSectionGetOffset(secNew, p, &offNew));
      count = 0;
      for (i = 0; i < dof; i++) {
        PetscInt q = vals[off + i], qDof = 0, qOff = 0, j;

        if (q >= pStart && q < pEnd) {
          CHKERRQ(PetscSectionGetDof(section, q, &qDof));
          CHKERRQ(PetscSectionGetOffset(section, q, &qOff));
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
        CHKERRQ(PetscSectionSetDof(secNew, p, count));
        compress = PETSC_TRUE;
      }
    }
  }
  CHKERRQ(ISRestoreIndices(is,&vals));
  CHKERRMPI(MPIU_Allreduce(&anyNew,&globalAnyNew,1,MPIU_BOOL,MPI_LOR,PetscObjectComm((PetscObject)secNew)));
  if (!globalAnyNew) {
    CHKERRQ(PetscSectionDestroy(&secNew));
    *sectionNew = NULL;
    *isNew = NULL;
  }
  else {
    PetscBool globalCompress;

    CHKERRMPI(MPIU_Allreduce(&compress,&globalCompress,1,MPIU_BOOL,MPI_LOR,PetscObjectComm((PetscObject)secNew)));
    if (compress) {
      PetscSection secComp;
      PetscInt *valsComp = NULL;

      CHKERRQ(PetscSectionCreate(PetscObjectComm((PetscObject)section),&secComp));
      CHKERRQ(PetscSectionSetChart(secComp,pStart,pEnd));
      for (p = pStart; p < pEnd; p++) {
        PetscInt dof;

        CHKERRQ(PetscSectionGetDof(secNew, p, &dof));
        CHKERRQ(PetscSectionSetDof(secComp, p, dof));
      }
      CHKERRQ(PetscSectionSetUp(secComp));
      CHKERRQ(PetscSectionGetStorageSize(secComp,&sizeNew));
      CHKERRQ(PetscMalloc1(sizeNew,&valsComp));
      for (p = pStart; p < pEnd; p++) {
        PetscInt dof, off, offNew, j;

        CHKERRQ(PetscSectionGetDof(secNew, p, &dof));
        CHKERRQ(PetscSectionGetOffset(secNew, p, &off));
        CHKERRQ(PetscSectionGetOffset(secComp, p, &offNew));
        for (j = 0; j < dof; j++) {
          valsComp[offNew + j] = valsNew[off + j];
        }
      }
      CHKERRQ(PetscSectionDestroy(&secNew));
      secNew  = secComp;
      CHKERRQ(PetscFree(valsNew));
      valsNew = valsComp;
    }
    CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)is),sizeNew,valsNew,PETSC_OWN_POINTER,isNew));
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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  CHKERRQ(DMPlexGetChart(dm,&pStart,&pEnd));
  CHKERRQ(DMGetLabel(dm,"canonical",&canonLabel));
  for (p = pStart; p < pEnd; p++) {
    PetscInt parent;

    if (canonLabel) {
      PetscInt canon;

      CHKERRQ(DMLabelGetValue(canonLabel,p,&canon));
      if (p != canon) continue;
    }
    CHKERRQ(DMPlexGetTreeParent(dm,p,&parent,NULL));
    if (parent != p) {
      aMin = PetscMin(aMin,p);
      aMax = PetscMax(aMax,p+1);
    }
  }
  if (aMin > aMax) {
    aMin = -1;
    aMax = -1;
  }
  CHKERRQ(PetscSectionCreate(PETSC_COMM_SELF,&aSec));
  CHKERRQ(PetscSectionSetChart(aSec,aMin,aMax));
  for (p = aMin; p < aMax; p++) {
    PetscInt parent, ancestor = p;

    if (canonLabel) {
      PetscInt canon;

      CHKERRQ(DMLabelGetValue(canonLabel,p,&canon));
      if (p != canon) continue;
    }
    CHKERRQ(DMPlexGetTreeParent(dm,p,&parent,NULL));
    while (parent != ancestor) {
      ancestor = parent;
      CHKERRQ(DMPlexGetTreeParent(dm,ancestor,&parent,NULL));
    }
    if (ancestor != p) {
      PetscInt closureSize, *closure = NULL;

      CHKERRQ(DMPlexGetTransitiveClosure(dm,ancestor,PETSC_TRUE,&closureSize,&closure));
      CHKERRQ(PetscSectionSetDof(aSec,p,closureSize));
      CHKERRQ(DMPlexRestoreTransitiveClosure(dm,ancestor,PETSC_TRUE,&closureSize,&closure));
    }
  }
  CHKERRQ(PetscSectionSetUp(aSec));
  CHKERRQ(PetscSectionGetStorageSize(aSec,&size));
  CHKERRQ(PetscMalloc1(size,&anchors));
  for (p = aMin; p < aMax; p++) {
    PetscInt parent, ancestor = p;

    if (canonLabel) {
      PetscInt canon;

      CHKERRQ(DMLabelGetValue(canonLabel,p,&canon));
      if (p != canon) continue;
    }
    CHKERRQ(DMPlexGetTreeParent(dm,p,&parent,NULL));
    while (parent != ancestor) {
      ancestor = parent;
      CHKERRQ(DMPlexGetTreeParent(dm,ancestor,&parent,NULL));
    }
    if (ancestor != p) {
      PetscInt j, closureSize, *closure = NULL, aOff;

      CHKERRQ(PetscSectionGetOffset(aSec,p,&aOff));

      CHKERRQ(DMPlexGetTransitiveClosure(dm,ancestor,PETSC_TRUE,&closureSize,&closure));
      for (j = 0; j < closureSize; j++) {
        anchors[aOff + j] = closure[2*j];
      }
      CHKERRQ(DMPlexRestoreTransitiveClosure(dm,ancestor,PETSC_TRUE,&closureSize,&closure));
    }
  }
  CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,size,anchors,PETSC_OWN_POINTER,&aIS));
  {
    PetscSection aSecNew = aSec;
    IS           aISNew  = aIS;

    CHKERRQ(PetscObjectReference((PetscObject)aSec));
    CHKERRQ(PetscObjectReference((PetscObject)aIS));
    while (aSecNew) {
      CHKERRQ(PetscSectionDestroy(&aSec));
      CHKERRQ(ISDestroy(&aIS));
      aSec    = aSecNew;
      aIS     = aISNew;
      aSecNew = NULL;
      aISNew  = NULL;
      CHKERRQ(AnchorsFlatten(aSec,aIS,&aSecNew,&aISNew));
    }
  }
  CHKERRQ(DMPlexSetAnchors(dm,aSec,aIS));
  CHKERRQ(PetscSectionDestroy(&aSec));
  CHKERRQ(ISDestroy(&aIS));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexGetTrueSupportSize(DM dm,PetscInt p,PetscInt *dof,PetscInt *numTrueSupp)
{
  PetscFunctionBegin;
  if (numTrueSupp[p] == -1) {
    PetscInt i, alldof;
    const PetscInt *supp;
    PetscInt count = 0;

    CHKERRQ(DMPlexGetSupportSize(dm,p,&alldof));
    CHKERRQ(DMPlexGetSupport(dm,p,&supp));
    for (i = 0; i < alldof; i++) {
      PetscInt q = supp[i], numCones, j;
      const PetscInt *cone;

      CHKERRQ(DMPlexGetConeSize(dm,q,&numCones));
      CHKERRQ(DMPlexGetCone(dm,q,&cone));
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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  /* symmetrize the hierarchy */
  CHKERRQ(DMPlexGetDepth(dm,&depth));
  CHKERRQ(PetscSectionCreate(PetscObjectComm((PetscObject)(mesh->supportSection)),&newSupportSection));
  CHKERRQ(DMPlexGetChart(dm,&pStart,&pEnd));
  CHKERRQ(PetscSectionSetChart(newSupportSection,pStart,pEnd));
  CHKERRQ(PetscCalloc1(pEnd,&offsets));
  CHKERRQ(PetscMalloc1(pEnd,&numTrueSupp));
  for (p = 0; p < pEnd; p++) numTrueSupp[p] = -1;
  /* if a point is in the (true) support of q, it should be in the support of
   * parent(q) */
  for (d = 0; d <= depth; d++) {
    CHKERRQ(DMPlexGetHeightStratum(dm,d,&pStart,&pEnd));
    for (p = pStart; p < pEnd; ++p) {
      PetscInt dof, q, qdof, parent;

      CHKERRQ(DMPlexGetTrueSupportSize(dm,p,&dof,numTrueSupp));
      CHKERRQ(PetscSectionAddDof(newSupportSection, p, dof));
      q    = p;
      CHKERRQ(DMPlexGetTreeParent(dm,q,&parent,NULL));
      while (parent != q && parent >= pStart && parent < pEnd) {
        q = parent;

        CHKERRQ(DMPlexGetTrueSupportSize(dm,q,&qdof,numTrueSupp));
        CHKERRQ(PetscSectionAddDof(newSupportSection,p,qdof));
        CHKERRQ(PetscSectionAddDof(newSupportSection,q,dof));
        CHKERRQ(DMPlexGetTreeParent(dm,q,&parent,NULL));
      }
    }
  }
  CHKERRQ(PetscSectionSetUp(newSupportSection));
  CHKERRQ(PetscSectionGetStorageSize(newSupportSection,&newSize));
  CHKERRQ(PetscMalloc1(newSize,&newSupports));
  for (d = 0; d <= depth; d++) {
    CHKERRQ(DMPlexGetHeightStratum(dm,d,&pStart,&pEnd));
    for (p = pStart; p < pEnd; p++) {
      PetscInt dof, off, q, qdof, qoff, newDof, newOff, newqOff, i, parent;

      CHKERRQ(PetscSectionGetDof(mesh->supportSection, p, &dof));
      CHKERRQ(PetscSectionGetOffset(mesh->supportSection, p, &off));
      CHKERRQ(PetscSectionGetDof(newSupportSection, p, &newDof));
      CHKERRQ(PetscSectionGetOffset(newSupportSection, p, &newOff));
      for (i = 0; i < dof; i++) {
        PetscInt numCones, j;
        const PetscInt *cone;
        PetscInt q = mesh->supports[off + i];

        CHKERRQ(DMPlexGetConeSize(dm,q,&numCones));
        CHKERRQ(DMPlexGetCone(dm,q,&cone));
        for (j = 0; j < numCones; j++) {
          if (cone[j] == p) break;
        }
        if (j < numCones) newSupports[newOff+offsets[p]++] = q;
      }
      mesh->maxSupportSize = PetscMax(mesh->maxSupportSize,newDof);

      q    = p;
      CHKERRQ(DMPlexGetTreeParent(dm,q,&parent,NULL));
      while (parent != q && parent >= pStart && parent < pEnd) {
        q = parent;
        CHKERRQ(PetscSectionGetDof(mesh->supportSection, q, &qdof));
        CHKERRQ(PetscSectionGetOffset(mesh->supportSection, q, &qoff));
        CHKERRQ(PetscSectionGetOffset(newSupportSection, q, &newqOff));
        for (i = 0; i < qdof; i++) {
          PetscInt numCones, j;
          const PetscInt *cone;
          PetscInt r = mesh->supports[qoff + i];

          CHKERRQ(DMPlexGetConeSize(dm,r,&numCones));
          CHKERRQ(DMPlexGetCone(dm,r,&cone));
          for (j = 0; j < numCones; j++) {
            if (cone[j] == q) break;
          }
          if (j < numCones) newSupports[newOff+offsets[p]++] = r;
        }
        for (i = 0; i < dof; i++) {
          PetscInt numCones, j;
          const PetscInt *cone;
          PetscInt r = mesh->supports[off + i];

          CHKERRQ(DMPlexGetConeSize(dm,r,&numCones));
          CHKERRQ(DMPlexGetCone(dm,r,&cone));
          for (j = 0; j < numCones; j++) {
            if (cone[j] == p) break;
          }
          if (j < numCones) newSupports[newqOff+offsets[q]++] = r;
        }
        CHKERRQ(DMPlexGetTreeParent(dm,q,&parent,NULL));
      }
    }
  }
  CHKERRQ(PetscSectionDestroy(&mesh->supportSection));
  mesh->supportSection = newSupportSection;
  CHKERRQ(PetscFree(mesh->supports));
  mesh->supports = newSupports;
  CHKERRQ(PetscFree(offsets));
  CHKERRQ(PetscFree(numTrueSupp));

  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexComputeAnchorMatrix_Tree_Direct(DM,PetscSection,PetscSection,Mat);
static PetscErrorCode DMPlexComputeAnchorMatrix_Tree_FromReference(DM,PetscSection,PetscSection,Mat);

static PetscErrorCode DMPlexSetTree_Internal(DM dm, PetscSection parentSection, PetscInt *parents, PetscInt *childIDs, PetscBool computeCanonical, PetscBool exchangeSupports)
{
  DM_Plex       *mesh = (DM_Plex *)dm->data;
  DM             refTree;
  PetscInt       size;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(parentSection, PETSC_SECTION_CLASSID, 2);
  CHKERRQ(PetscObjectReference((PetscObject)parentSection));
  CHKERRQ(PetscSectionDestroy(&mesh->parentSection));
  mesh->parentSection = parentSection;
  CHKERRQ(PetscSectionGetStorageSize(parentSection,&size));
  if (parents != mesh->parents) {
    CHKERRQ(PetscFree(mesh->parents));
    CHKERRQ(PetscMalloc1(size,&mesh->parents));
    CHKERRQ(PetscArraycpy(mesh->parents, parents, size));
  }
  if (childIDs != mesh->childIDs) {
    CHKERRQ(PetscFree(mesh->childIDs));
    CHKERRQ(PetscMalloc1(size,&mesh->childIDs));
    CHKERRQ(PetscArraycpy(mesh->childIDs, childIDs, size));
  }
  CHKERRQ(DMPlexGetReferenceTree(dm,&refTree));
  if (refTree) {
    DMLabel canonLabel;

    CHKERRQ(DMGetLabel(refTree,"canonical",&canonLabel));
    if (canonLabel) {
      PetscInt i;

      for (i = 0; i < size; i++) {
        PetscInt canon;
        CHKERRQ(DMLabelGetValue(canonLabel, mesh->childIDs[i], &canon));
        if (canon >= 0) {
          mesh->childIDs[i] = canon;
        }
      }
    }
    mesh->computeanchormatrix = DMPlexComputeAnchorMatrix_Tree_FromReference;
  } else {
    mesh->computeanchormatrix = DMPlexComputeAnchorMatrix_Tree_Direct;
  }
  CHKERRQ(DMPlexTreeSymmetrize(dm));
  if (computeCanonical) {
    PetscInt d, dim;

    /* add the canonical label */
    CHKERRQ(DMGetDimension(dm,&dim));
    CHKERRQ(DMCreateLabel(dm,"canonical"));
    for (d = 0; d <= dim; d++) {
      PetscInt p, dStart, dEnd, canon = -1, cNumChildren;
      const PetscInt *cChildren;

      CHKERRQ(DMPlexGetDepthStratum(dm,d,&dStart,&dEnd));
      for (p = dStart; p < dEnd; p++) {
        CHKERRQ(DMPlexGetTreeChildren(dm,p,&cNumChildren,&cChildren));
        if (cNumChildren) {
          canon = p;
          break;
        }
      }
      if (canon == -1) continue;
      for (p = dStart; p < dEnd; p++) {
        PetscInt numChildren, i;
        const PetscInt *children;

        CHKERRQ(DMPlexGetTreeChildren(dm,p,&numChildren,&children));
        if (numChildren) {
          PetscCheckFalse(numChildren != cNumChildren,PetscObjectComm((PetscObject)dm),PETSC_ERR_PLIB,"All parent points in a stratum should have the same number of children: %d != %d", numChildren, cNumChildren);
          CHKERRQ(DMSetLabelValue(dm,"canonical",p,canon));
          for (i = 0; i < numChildren; i++) {
            CHKERRQ(DMSetLabelValue(dm,"canonical",children[i],cChildren[i]));
          }
        }
      }
    }
  }
  if (exchangeSupports) {
    CHKERRQ(DMPlexTreeExchangeSupports(dm));
  }
  mesh->createanchors = DMPlexCreateAnchors_Tree;
  /* reset anchors */
  CHKERRQ(DMPlexSetAnchors(dm,NULL,NULL));
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
  PetscFunctionBegin;
  CHKERRQ(DMPlexSetTree_Internal(dm,parentSection,parents,childIDs,PETSC_FALSE,PETSC_TRUE));
  PetscFunctionReturn(0);
}

/*@
  DMPlexGetTree - get the tree that describes the hierarchy of non-conforming mesh points.
  Collective on dm

  Input Parameter:
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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  pSec = mesh->parentSection;
  if (pSec && point >= pSec->pStart && point < pSec->pEnd) {
    PetscInt dof;

    CHKERRQ(PetscSectionGetDof (pSec, point, &dof));
    if (dof) {
      PetscInt off;

      CHKERRQ(PetscSectionGetOffset (pSec, point, &off));
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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  childSec = mesh->childSection;
  if (childSec && point >= childSec->pStart && point < childSec->pEnd) {
    CHKERRQ(PetscSectionGetDof (childSec, point, &dof));
  }
  if (numChildren) *numChildren = dof;
  if (children) {
    if (dof) {
      PetscInt off;

      CHKERRQ(PetscSectionGetOffset (childSec, point, &off));
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

  PetscFunctionBegin;
  CHKERRQ(PetscSpaceEvaluate(space,nPoints,points,work,NULL,NULL));
  for (f = 0, offset = 0; f < nFunctionals; f++) {
    qPoints = pointsPerFn[f];
    for (b = 0; b < nBasis; b++) {
      PetscScalar val = 0.;

      for (p = 0; p < qPoints; p++) {
        for (c = 0; c < nComps; c++) {
          val += work[((offset + p) * nBasis + b) * nComps + c] * weights[(offset + p) * nComps + c];
        }
      }
      CHKERRQ(MatSetValue(basisAtPoints,b,f,val,INSERT_VALUES));
    }
    offset += qPoints;
  }
  CHKERRQ(MatAssemblyBegin(basisAtPoints,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(basisAtPoints,MAT_FINAL_ASSEMBLY));
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

  PetscFunctionBegin;
  CHKERRQ(DMPlexGetChart(dm,&pStart,&pEnd));
  CHKERRQ(DMGetDS(dm,&ds));
  CHKERRQ(PetscDSGetNumFields(ds,&numFields));
  CHKERRQ(DMPlexGetHeightStratum(dm,0,&cStart,&cEnd));
  CHKERRQ(DMPlexGetAnchors(dm,&aSec,&aIS));
  CHKERRQ(ISGetIndices(aIS,&anchors));
  CHKERRQ(PetscSectionGetChart(cSec,&conStart,&conEnd));
  CHKERRQ(DMGetDimension(dm,&spdim));
  CHKERRQ(PetscMalloc6(spdim,&v0,spdim,&v0parent,spdim,&vtmp,spdim*spdim,&J,spdim*spdim,&Jparent,spdim*spdim,&invJparent));

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

    CHKERRQ(PetscDSGetDiscretization(ds,f,&disc));
    CHKERRQ(PetscObjectGetClassId(disc,&id));
    if (id == PETSCFE_CLASSID) {
      PetscFE fe = (PetscFE) disc;

      CHKERRQ(PetscFEGetBasisSpace(fe,&bspace));
      CHKERRQ(PetscFEGetDualSpace(fe,&dspace));
      CHKERRQ(PetscDualSpaceGetDimension(dspace,&fSize));
      CHKERRQ(PetscFEGetNumComponents(fe,&Nc));
    }
    else if (id == PETSCFV_CLASSID) {
      PetscFV fv = (PetscFV) disc;

      CHKERRQ(PetscFVGetNumComponents(fv,&Nc));
      CHKERRQ(PetscSpaceCreate(PetscObjectComm((PetscObject)fv),&bspace));
      CHKERRQ(PetscSpaceSetType(bspace,PETSCSPACEPOLYNOMIAL));
      CHKERRQ(PetscSpaceSetDegree(bspace,0,PETSC_DETERMINE));
      CHKERRQ(PetscSpaceSetNumComponents(bspace,Nc));
      CHKERRQ(PetscSpaceSetNumVariables(bspace,spdim));
      CHKERRQ(PetscSpaceSetUp(bspace));
      CHKERRQ(PetscFVGetDualSpace(fv,&dspace));
      CHKERRQ(PetscDualSpaceGetDimension(dspace,&fSize));
    }
    else SETERRQ(PetscObjectComm(disc),PETSC_ERR_ARG_UNKNOWN_TYPE, "PetscDS discretization id %d not recognized.", id);
    CHKERRQ(PetscDualSpaceGetNumDof(dspace,&numDof));
    for (i = 0, maxDof = 0; i <= spdim; i++) {maxDof = PetscMax(maxDof,numDof[i]);}
    CHKERRQ(PetscDualSpaceGetSymmetries(dspace,&perms,&flips));

    CHKERRQ(MatCreate(PETSC_COMM_SELF,&Amat));
    CHKERRQ(MatSetSizes(Amat,fSize,fSize,fSize,fSize));
    CHKERRQ(MatSetType(Amat,MATSEQDENSE));
    CHKERRQ(MatSetUp(Amat));
    CHKERRQ(MatDuplicate(Amat,MAT_DO_NOT_COPY_VALUES,&Bmat));
    CHKERRQ(MatDuplicate(Amat,MAT_DO_NOT_COPY_VALUES,&Xmat));
    nPoints = 0;
    for (i = 0; i < fSize; i++) {
      PetscInt        qPoints, thisNc;
      PetscQuadrature quad;

      CHKERRQ(PetscDualSpaceGetFunctional(dspace,i,&quad));
      CHKERRQ(PetscQuadratureGetData(quad,NULL,&thisNc,&qPoints,NULL,NULL));
      PetscCheckFalse(thisNc != Nc,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Functional dim %D does not much basis dim %D",thisNc,Nc);
      nPoints += qPoints;
    }
    CHKERRQ(PetscMalloc7(fSize,&sizes,nPoints*Nc,&weights,spdim*nPoints,&pointsRef,spdim*nPoints,&pointsReal,nPoints*fSize*Nc,&work,maxDof,&workIndRow,maxDof,&workIndCol));
    CHKERRQ(PetscMalloc1(maxDof * maxDof,&scwork));
    offset = 0;
    for (i = 0; i < fSize; i++) {
      PetscInt        qPoints;
      const PetscReal    *p, *w;
      PetscQuadrature quad;

      CHKERRQ(PetscDualSpaceGetFunctional(dspace,i,&quad));
      CHKERRQ(PetscQuadratureGetData(quad,NULL,NULL,&qPoints,&p,&w));
      CHKERRQ(PetscArraycpy(weights+Nc*offset,w,Nc*qPoints));
      CHKERRQ(PetscArraycpy(pointsRef+spdim*offset,p,spdim*qPoints));
      sizes[i] = qPoints;
      offset  += qPoints;
    }
    CHKERRQ(EvaluateBasis(bspace,fSize,fSize,Nc,nPoints,sizes,pointsRef,weights,work,Amat));
    CHKERRQ(MatLUFactor(Amat,NULL,NULL,NULL));
    for (c = cStart; c < cEnd; c++) {
      PetscInt        parent;
      PetscInt        closureSize, closureSizeP, *closure = NULL, *closureP = NULL;
      PetscInt        *childOffsets, *parentOffsets;

      CHKERRQ(DMPlexGetTreeParent(dm,c,&parent,NULL));
      if (parent == c) continue;
      CHKERRQ(DMPlexGetTransitiveClosure(dm,c,PETSC_TRUE,&closureSize,&closure));
      for (i = 0; i < closureSize; i++) {
        PetscInt p = closure[2*i];
        PetscInt conDof;

        if (p < conStart || p >= conEnd) continue;
        if (numFields) {
          CHKERRQ(PetscSectionGetFieldDof(cSec,p,f,&conDof));
        }
        else {
          CHKERRQ(PetscSectionGetDof(cSec,p,&conDof));
        }
        if (conDof) break;
      }
      if (i == closureSize) {
        CHKERRQ(DMPlexRestoreTransitiveClosure(dm,c,PETSC_TRUE,&closureSize,&closure));
        continue;
      }

      CHKERRQ(DMPlexComputeCellGeometryFEM(dm, c, NULL, v0, J, NULL, &detJ));
      CHKERRQ(DMPlexComputeCellGeometryFEM(dm, parent, NULL, v0parent, Jparent, invJparent, &detJparent));
      for (i = 0; i < nPoints; i++) {
        const PetscReal xi0[3] = {-1.,-1.,-1.};

        CoordinatesRefToReal(spdim, spdim, xi0, v0, J, &pointsRef[i*spdim],vtmp);
        CoordinatesRealToRef(spdim, spdim, xi0, v0parent, invJparent, vtmp, &pointsReal[i*spdim]);
      }
      CHKERRQ(EvaluateBasis(bspace,fSize,fSize,Nc,nPoints,sizes,pointsReal,weights,work,Bmat));
      CHKERRQ(MatMatSolve(Amat,Bmat,Xmat));
      CHKERRQ(MatDenseGetArrayRead(Xmat,&X));
      CHKERRQ(DMPlexGetTransitiveClosure(dm,parent,PETSC_TRUE,&closureSizeP,&closureP));
      CHKERRQ(PetscMalloc2(closureSize+1,&childOffsets,closureSizeP+1,&parentOffsets));
      childOffsets[0] = 0;
      for (i = 0; i < closureSize; i++) {
        PetscInt p = closure[2*i];
        PetscInt dof;

        if (numFields) {
          CHKERRQ(PetscSectionGetFieldDof(section,p,f,&dof));
        }
        else {
          CHKERRQ(PetscSectionGetDof(section,p,&dof));
        }
        childOffsets[i+1]=childOffsets[i]+dof;
      }
      parentOffsets[0] = 0;
      for (i = 0; i < closureSizeP; i++) {
        PetscInt p = closureP[2*i];
        PetscInt dof;

        if (numFields) {
          CHKERRQ(PetscSectionGetFieldDof(section,p,f,&dof));
        }
        else {
          CHKERRQ(PetscSectionGetDof(section,p,&dof));
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
          CHKERRQ(PetscSectionGetFieldDof(cSec,p,f,&conDof));
          CHKERRQ(PetscSectionGetFieldOffset(cSec,p,f,&conOff));
        }
        else {
          CHKERRQ(PetscSectionGetDof(cSec,p,&conDof));
          CHKERRQ(PetscSectionGetOffset(cSec,p,&conOff));
        }
        if (!conDof) continue;
        perm  = (perms && perms[i]) ? perms[i][o] : NULL;
        flip  = (flips && flips[i]) ? flips[i][o] : NULL;
        CHKERRQ(PetscSectionGetDof(aSec,p,&aDof));
        CHKERRQ(PetscSectionGetOffset(aSec,p,&aOff));
        nWork = childOffsets[i+1]-childOffsets[i];
        for (k = 0; k < aDof; k++) {
          PetscInt a = anchors[aOff + k];
          PetscInt aSecDof, aSecOff;

          if (numFields) {
            CHKERRQ(PetscSectionGetFieldDof(section,a,f,&aSecDof));
            CHKERRQ(PetscSectionGetFieldOffset(section,a,f,&aSecOff));
          }
          else {
            CHKERRQ(PetscSectionGetDof(section,a,&aSecDof));
            CHKERRQ(PetscSectionGetOffset(section,a,&aSecOff));
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
              CHKERRQ(MatSetValues(cMat,nWork,workIndRow,nWorkP,workIndCol,scwork,INSERT_VALUES));
              break;
            }
          }
        }
      }
      CHKERRQ(MatDenseRestoreArrayRead(Xmat,&X));
      CHKERRQ(PetscFree2(childOffsets,parentOffsets));
      CHKERRQ(DMPlexRestoreTransitiveClosure(dm,c,PETSC_TRUE,&closureSize,&closure));
      CHKERRQ(DMPlexRestoreTransitiveClosure(dm,parent,PETSC_TRUE,&closureSizeP,&closureP));
    }
    CHKERRQ(MatDestroy(&Amat));
    CHKERRQ(MatDestroy(&Bmat));
    CHKERRQ(MatDestroy(&Xmat));
    CHKERRQ(PetscFree(scwork));
    CHKERRQ(PetscFree7(sizes,weights,pointsRef,pointsReal,work,workIndRow,workIndCol));
    if (id == PETSCFV_CLASSID) {
      CHKERRQ(PetscSpaceDestroy(&bspace));
    }
  }
  CHKERRQ(MatAssemblyBegin(cMat,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(cMat,MAT_FINAL_ASSEMBLY));
  CHKERRQ(PetscFree6(v0,v0parent,vtmp,J,Jparent,invJparent));
  CHKERRQ(ISRestoreIndices(aIS,&anchors));

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

  PetscFunctionBegin;
  CHKERRQ(DMGetDS(refTree,&ds));
  CHKERRQ(PetscDSGetNumFields(ds,&numFields));
  maxFields = PetscMax(1,numFields);
  CHKERRQ(DMGetDefaultConstraints(refTree,&refConSec,&refCmat,NULL));
  CHKERRQ(DMPlexGetAnchors(refTree,&refAnSec,&refAnIS));
  CHKERRQ(ISGetIndices(refAnIS,&refAnchors));
  CHKERRQ(DMGetLocalSection(refTree,&refSection));
  CHKERRQ(PetscSectionGetChart(refConSec,&pRefStart,&pRefEnd));
  CHKERRQ(PetscMalloc1(pRefEnd-pRefStart,&refPointFieldMats));
  CHKERRQ(PetscMalloc1(pRefEnd-pRefStart,&refPointFieldN));
  CHKERRQ(PetscSectionGetMaxDof(refConSec,&maxDof));
  CHKERRQ(PetscSectionGetMaxDof(refAnSec,&maxAnDof));
  CHKERRQ(PetscMalloc1(maxDof,&rows));
  CHKERRQ(PetscMalloc1(maxDof*maxAnDof,&cols));
  for (p = pRefStart; p < pRefEnd; p++) {
    PetscInt parent, closureSize, *closure = NULL, pDof;

    CHKERRQ(DMPlexGetTreeParent(refTree,p,&parent,NULL));
    CHKERRQ(PetscSectionGetDof(refConSec,p,&pDof));
    if (!pDof || parent == p) continue;

    CHKERRQ(PetscMalloc1(maxFields,&refPointFieldMats[p-pRefStart]));
    CHKERRQ(PetscCalloc1(maxFields,&refPointFieldN[p-pRefStart]));
    CHKERRQ(DMPlexGetTransitiveClosure(refTree,parent,PETSC_TRUE,&closureSize,&closure));
    for (f = 0; f < maxFields; f++) {
      PetscInt cDof, cOff, numCols, r, i;

      if (f < numFields) {
        CHKERRQ(PetscSectionGetFieldDof(refConSec,p,f,&cDof));
        CHKERRQ(PetscSectionGetFieldOffset(refConSec,p,f,&cOff));
        CHKERRQ(PetscSectionGetFieldPointSyms(refSection,f,closureSize,closure,&perms,&flips));
      } else {
        CHKERRQ(PetscSectionGetDof(refConSec,p,&cDof));
        CHKERRQ(PetscSectionGetOffset(refConSec,p,&cOff));
        CHKERRQ(PetscSectionGetPointSyms(refSection,closureSize,closure,&perms,&flips));
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
          CHKERRQ(PetscSectionGetFieldDof(refSection,q,f,&aDof));
          CHKERRQ(PetscSectionGetFieldOffset(refSection,q,f,&aOff));
        }
        else {
          CHKERRQ(PetscSectionGetDof(refSection,q,&aDof));
          CHKERRQ(PetscSectionGetOffset(refSection,q,&aOff));
        }

        for (j = 0; j < aDof; j++) {
          cols[numCols++] = aOff + (perm ? perm[j] : j);
        }
      }
      refPointFieldN[p-pRefStart][f] = numCols;
      CHKERRQ(PetscMalloc1(cDof*numCols,&refPointFieldMats[p-pRefStart][f]));
      CHKERRQ(MatGetValues(refCmat,cDof,rows,numCols,cols,refPointFieldMats[p-pRefStart][f]));
      if (flips) {
        PetscInt colOff = 0;

        for (i = 0; i < closureSize; i++) {
          PetscInt          q = closure[2*i];
          PetscInt          aDof, aOff, j;
          const PetscScalar *flip = flips ? flips[i] : NULL;

          if (numFields) {
            CHKERRQ(PetscSectionGetFieldDof(refSection,q,f,&aDof));
            CHKERRQ(PetscSectionGetFieldOffset(refSection,q,f,&aOff));
          }
          else {
            CHKERRQ(PetscSectionGetDof(refSection,q,&aDof));
            CHKERRQ(PetscSectionGetOffset(refSection,q,&aOff));
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
        CHKERRQ(PetscSectionRestoreFieldPointSyms(refSection,f,closureSize,closure,&perms,&flips));
      } else {
        CHKERRQ(PetscSectionRestorePointSyms(refSection,closureSize,closure,&perms,&flips));
      }
    }
    CHKERRQ(DMPlexRestoreTransitiveClosure(refTree,parent,PETSC_TRUE,&closureSize,&closure));
  }
  *childrenMats = refPointFieldMats;
  *childrenN = refPointFieldN;
  CHKERRQ(ISRestoreIndices(refAnIS,&refAnchors));
  CHKERRQ(PetscFree(rows));
  CHKERRQ(PetscFree(cols));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexReferenceTreeRestoreChildrenMatrices(DM refTree, PetscScalar ****childrenMats, PetscInt ***childrenN)
{
  PetscDS        ds;
  PetscInt       **refPointFieldN;
  PetscScalar    ***refPointFieldMats;
  PetscInt       numFields, maxFields, pRefStart, pRefEnd, p, f;
  PetscSection   refConSec;

  PetscFunctionBegin;
  refPointFieldN = *childrenN;
  *childrenN = NULL;
  refPointFieldMats = *childrenMats;
  *childrenMats = NULL;
  CHKERRQ(DMGetDS(refTree,&ds));
  CHKERRQ(PetscDSGetNumFields(ds,&numFields));
  maxFields = PetscMax(1,numFields);
  CHKERRQ(DMGetDefaultConstraints(refTree,&refConSec,NULL,NULL));
  CHKERRQ(PetscSectionGetChart(refConSec,&pRefStart,&pRefEnd));
  for (p = pRefStart; p < pRefEnd; p++) {
    PetscInt parent, pDof;

    CHKERRQ(DMPlexGetTreeParent(refTree,p,&parent,NULL));
    CHKERRQ(PetscSectionGetDof(refConSec,p,&pDof));
    if (!pDof || parent == p) continue;

    for (f = 0; f < maxFields; f++) {
      PetscInt cDof;

      if (numFields) {
        CHKERRQ(PetscSectionGetFieldDof(refConSec,p,f,&cDof));
      }
      else {
        CHKERRQ(PetscSectionGetDof(refConSec,p,&cDof));
      }

      CHKERRQ(PetscFree(refPointFieldMats[p - pRefStart][f]));
    }
    CHKERRQ(PetscFree(refPointFieldMats[p - pRefStart]));
    CHKERRQ(PetscFree(refPointFieldN[p - pRefStart]));
  }
  CHKERRQ(PetscFree(refPointFieldMats));
  CHKERRQ(PetscFree(refPointFieldN));
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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  CHKERRQ(DMGetDS(dm,&ds));
  CHKERRQ(PetscDSGetNumFields(ds,&numFields));
  maxFields = PetscMax(1,numFields);
  CHKERRQ(DMPlexGetReferenceTree(dm,&refTree));
  CHKERRQ(DMCopyDisc(dm,refTree));
  CHKERRQ(DMGetDefaultConstraints(refTree,&refConSec,&refCmat,NULL));
  CHKERRQ(DMPlexGetAnchors(refTree,&refAnSec,&refAnIS));
  CHKERRQ(DMPlexGetAnchors(dm,&anSec,&anIS));
  CHKERRQ(ISGetIndices(anIS,&anchors));
  CHKERRQ(PetscSectionGetChart(refConSec,&pRefStart,&pRefEnd));
  CHKERRQ(PetscSectionGetChart(conSec,&conStart,&conEnd));
  CHKERRQ(PetscSectionGetMaxDof(refConSec,&maxDof));
  CHKERRQ(PetscSectionGetMaxDof(refAnSec,&maxAnDof));
  CHKERRQ(PetscMalloc1(maxDof*maxDof*maxAnDof,&pointWork));

  /* step 1: get submats for every constrained point in the reference tree */
  CHKERRQ(DMPlexReferenceTreeGetChildrenMatrices(refTree,&refPointFieldMats,&refPointFieldN));

  /* step 2: compute the preorder */
  CHKERRQ(DMPlexGetChart(dm,&pStart,&pEnd));
  CHKERRQ(PetscMalloc2(pEnd-pStart,&perm,pEnd-pStart,&iperm));
  for (p = pStart; p < pEnd; p++) {
    perm[p - pStart] = p;
    iperm[p - pStart] = p-pStart;
  }
  for (p = 0; p < pEnd - pStart;) {
    PetscInt point = perm[p];
    PetscInt parent;

    CHKERRQ(DMPlexGetTreeParent(dm,point,&parent,NULL));
    if (parent == point) {
      p++;
    }
    else {
      PetscInt size, closureSize, *closure = NULL, i;

      CHKERRQ(DMPlexGetTransitiveClosure(dm,parent,PETSC_TRUE,&closureSize,&closure));
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
      CHKERRQ(DMPlexRestoreTransitiveClosure(dm,parent,PETSC_TRUE,&closureSize,&closure));
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

    CHKERRQ(MatGetRowIJ(cMat,0,PETSC_FALSE,PETSC_FALSE,&nRows,&ia,&ja,&done));
    PetscCheckFalse(!done,PetscObjectComm((PetscObject)cMat),PETSC_ERR_PLIB,"Could not get RowIJ of constraint matrix");
    nnz  = ia[nRows];
    /* malloc and then zero rows right before we fill them: this way valgrind
     * can tell if we are doing progressive fill in the wrong order */
    CHKERRQ(PetscMalloc1(nnz,&vals));
    for (p = 0; p < pEnd - pStart; p++) {
      PetscInt        parent, childid, closureSize, *closure = NULL;
      PetscInt        point = perm[p], pointDof;

      CHKERRQ(DMPlexGetTreeParent(dm,point,&parent,&childid));
      if ((point < conStart) || (point >= conEnd) || (parent == point)) continue;
      CHKERRQ(PetscSectionGetDof(conSec,point,&pointDof));
      if (!pointDof) continue;
      CHKERRQ(DMPlexGetTransitiveClosure(dm,parent,PETSC_TRUE,&closureSize,&closure));
      for (f = 0; f < maxFields; f++) {
        PetscInt cDof, cOff, numCols, numFillCols, i, r, matOffset, offset;
        PetscScalar *pointMat;
        const PetscInt    **perms;
        const PetscScalar **flips;

        if (numFields) {
          CHKERRQ(PetscSectionGetFieldDof(conSec,point,f,&cDof));
          CHKERRQ(PetscSectionGetFieldOffset(conSec,point,f,&cOff));
        }
        else {
          CHKERRQ(PetscSectionGetDof(conSec,point,&cDof));
          CHKERRQ(PetscSectionGetOffset(conSec,point,&cOff));
        }
        if (!cDof) continue;
        if (numFields) CHKERRQ(PetscSectionGetFieldPointSyms(section,f,closureSize,closure,&perms,&flips));
        else           CHKERRQ(PetscSectionGetPointSyms(section,closureSize,closure,&perms,&flips));

        /* make sure that every row for this point is the same size */
        if (PetscDefined(USE_DEBUG)) {
          for (r = 0; r < cDof; r++) {
            if (cDof > 1 && r) {
              PetscCheckFalse((ia[cOff+r+1]-ia[cOff+r]) != (ia[cOff+r]-ia[cOff+r-1]),PETSC_COMM_SELF,PETSC_ERR_PLIB,"Two point rows have different nnz: %D vs. %D", (ia[cOff+r+1]-ia[cOff+r]), (ia[cOff+r]-ia[cOff+r-1]));
            }
          }
        }
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
            CHKERRQ(PetscSectionGetFieldDof(section,q,f,&aDof));
            CHKERRQ(PetscSectionGetFieldOffset(section,q,f,&aOff));
            if (q >= conStart && q < conEnd) {
              CHKERRQ(PetscSectionGetFieldDof(conSec,q,f,&qConDof));
              CHKERRQ(PetscSectionGetFieldOffset(conSec,q,f,&qConOff));
            }
          }
          else {
            CHKERRQ(PetscSectionGetDof(section,q,&aDof));
            CHKERRQ(PetscSectionGetOffset(section,q,&aOff));
            if (q >= conStart && q < conEnd) {
              CHKERRQ(PetscSectionGetDof(conSec,q,&qConDof));
              CHKERRQ(PetscSectionGetOffset(conSec,q,&qConOff));
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
              PetscCheckFalse(k == numFillCols,PETSC_COMM_SELF,PETSC_ERR_PLIB,"No nonzero space for (%d, %d)", cOff, col);
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
            PetscCheckFalse(k == numFillCols,PETSC_COMM_SELF,PETSC_ERR_PLIB,"No nonzero space for (%d, %d)", cOff, aOff);
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
          CHKERRQ(PetscSectionRestoreFieldPointSyms(section,f,closureSize,closure,&perms,&flips));
        } else {
          CHKERRQ(PetscSectionRestorePointSyms(section,closureSize,closure,&perms,&flips));
        }
      }
      CHKERRQ(DMPlexRestoreTransitiveClosure(dm,parent,PETSC_TRUE,&closureSize,&closure));
    }
    for (row = 0; row < nRows; row++) {
      CHKERRQ(MatSetValues(cMat,1,&row,ia[row+1]-ia[row],&ja[ia[row]],&vals[ia[row]],INSERT_VALUES));
    }
    CHKERRQ(MatRestoreRowIJ(cMat,0,PETSC_FALSE,PETSC_FALSE,&nRows,&ia,&ja,&done));
    PetscCheckFalse(!done,PetscObjectComm((PetscObject)cMat),PETSC_ERR_PLIB,"Could not restore RowIJ of constraint matrix");
    CHKERRQ(MatAssemblyBegin(cMat,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(cMat,MAT_FINAL_ASSEMBLY));
    CHKERRQ(PetscFree(vals));
  }

  /* clean up */
  CHKERRQ(ISRestoreIndices(anIS,&anchors));
  CHKERRQ(PetscFree2(perm,iperm));
  CHKERRQ(PetscFree(pointWork));
  CHKERRQ(DMPlexReferenceTreeRestoreChildrenMatrices(refTree,&refPointFieldMats,&refPointFieldN));
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

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm),&rank));
  CHKERRQ(DMGetDimension(dm,&dim));
  CHKERRQ(DMPlexCreate(PetscObjectComm((PetscObject)dm), ncdm));
  CHKERRQ(DMSetDimension(*ncdm,dim));

  CHKERRQ(DMPlexGetChart(dm, &pStart, &pEnd));
  CHKERRQ(PetscSectionCreate(PetscObjectComm((PetscObject)dm),&parentSection));
  CHKERRQ(DMPlexGetReferenceTree(dm,&K));
  if (rank == 0) {
    /* compute the new charts */
    CHKERRQ(PetscMalloc5(dim+1,&pNewCount,dim+1,&pNewStart,dim+1,&pNewEnd,dim+1,&pOldStart,dim+1,&pOldEnd));
    offset = 0;
    for (d = 0; d <= dim; d++) {
      PetscInt pOldCount, kStart, kEnd, k;

      pNewStart[d] = offset;
      CHKERRQ(DMPlexGetHeightStratum(dm,d,&pOldStart[d],&pOldEnd[d]));
      CHKERRQ(DMPlexGetHeightStratum(K,d,&kStart,&kEnd));
      pOldCount = pOldEnd[d] - pOldStart[d];
      /* adding the new points */
      pNewCount[d] = pOldCount + kEnd - kStart;
      if (!d) {
        /* removing the cell */
        pNewCount[d]--;
      }
      for (k = kStart; k < kEnd; k++) {
        PetscInt parent;
        CHKERRQ(DMPlexGetTreeParent(K,k,&parent,NULL));
        if (parent == k) {
          /* avoid double counting points that won't actually be new */
          pNewCount[d]--;
        }
      }
      pNewEnd[d] = pNewStart[d] + pNewCount[d];
      offset = pNewEnd[d];

    }
    PetscCheckFalse(cell < pOldStart[0] || cell >= pOldEnd[0],PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"%d not in cell range [%d, %d)", cell, pOldStart[0], pOldEnd[0]);
    /* get the current closure of the cell that we are removing */
    CHKERRQ(DMPlexGetTransitiveClosure(dm,cell,PETSC_TRUE,&nc,&cellClosure));

    CHKERRQ(PetscMalloc1(pNewEnd[dim],&newConeSizes));
    {
      DMPolytopeType pct, qct;
      PetscInt kStart, kEnd, k, closureSizeK, *closureK = NULL, j;

      CHKERRQ(DMPlexGetChart(K,&kStart,&kEnd));
      CHKERRQ(PetscMalloc4(kEnd-kStart,&Kembedding,kEnd-kStart,&perm,kEnd-kStart,&iperm,kEnd-kStart,&preOrient));

      for (k = kStart; k < kEnd; k++) {
        perm[k - kStart] = k;
        iperm [k - kStart] = k - kStart;
        preOrient[k - kStart] = 0;
      }

      CHKERRQ(DMPlexGetTransitiveClosure(K,0,PETSC_TRUE,&closureSizeK,&closureK));
      for (j = 1; j < closureSizeK; j++) {
        PetscInt parentOrientA = closureK[2*j+1];
        PetscInt parentOrientB = cellClosure[2*j+1];
        PetscInt p, q;

        p = closureK[2*j];
        q = cellClosure[2*j];
        CHKERRQ(DMPlexGetCellType(K, p, &pct));
        CHKERRQ(DMPlexGetCellType(dm, q, &qct));
        for (d = 0; d <= dim; d++) {
          if (q >= pOldStart[d] && q < pOldEnd[d]) {
            Kembedding[p] = (q - pOldStart[d]) + pNewStart[d];
          }
        }
        parentOrientA = DMPolytopeConvertNewOrientation_Internal(pct, parentOrientA);
        parentOrientB = DMPolytopeConvertNewOrientation_Internal(qct, parentOrientB);
        if (parentOrientA != parentOrientB) {
          PetscInt numChildren, i;
          const PetscInt *children;

          CHKERRQ(DMPlexGetTreeChildren(K,p,&numChildren,&children));
          for (i = 0; i < numChildren; i++) {
            PetscInt kPerm, oPerm;

            k    = children[i];
            CHKERRQ(DMPlexReferenceTreeGetChildSymmetry(K,p,parentOrientA,0,k,parentOrientB,&oPerm,&kPerm));
            /* perm = what refTree position I'm in */
            perm[kPerm-kStart]      = k;
            /* iperm = who is at this position */
            iperm[k-kStart]         = kPerm-kStart;
            preOrient[kPerm-kStart] = oPerm;
          }
        }
      }
      CHKERRQ(DMPlexRestoreTransitiveClosure(K,0,PETSC_TRUE,&closureSizeK,&closureK));
    }
    CHKERRQ(PetscSectionSetChart(parentSection,0,pNewEnd[dim]));
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
        CHKERRQ(DMPlexGetConeSize(dm,p,&size));
        newConeSizes[offset++] = size;
        numNewCones += size;
      }

      CHKERRQ(DMPlexGetHeightStratum(K,d,&kStart,&kEnd));
      for (k = kStart; k < kEnd; k++) {
        PetscInt kParent;

        CHKERRQ(DMPlexGetTreeParent(K,k,&kParent,NULL));
        if (kParent != k) {
          Kembedding[k] = offset;
          CHKERRQ(DMPlexGetConeSize(K,k,&size));
          newConeSizes[offset++] = size;
          numNewCones += size;
          if (kParent != 0) {
            CHKERRQ(PetscSectionSetDof(parentSection,Kembedding[k],1));
          }
        }
      }
    }

    CHKERRQ(PetscSectionSetUp(parentSection));
    CHKERRQ(PetscSectionGetStorageSize(parentSection,&numPointsWithParents));
    CHKERRQ(PetscMalloc2(numNewCones,&newCones,numNewCones,&newOrientations));
    CHKERRQ(PetscMalloc2(numPointsWithParents,&parents,numPointsWithParents,&childIDs));

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
        CHKERRQ(DMPlexGetConeSize(dm,p,&size));
        CHKERRQ(DMPlexGetCone(dm,p,&cone));
        CHKERRQ(DMPlexGetConeOrientation(dm,p,&orientation));
        for (l = 0; l < size; l++) {
          newCones[offset]          = (cone[l] - pOldStart[d + 1]) + pNewStart[d + 1];
          newOrientations[offset++] = orientation[l];
        }
      }

      CHKERRQ(DMPlexGetHeightStratum(K,d,&kStart,&kEnd));
      for (k = kStart; k < kEnd; k++) {
        PetscInt kPerm = perm[k], kParent;
        PetscInt preO  = preOrient[k];

        CHKERRQ(DMPlexGetTreeParent(K,k,&kParent,NULL));
        if (kParent != k) {
          /* embed new cones */
          CHKERRQ(DMPlexGetConeSize(K,k,&size));
          CHKERRQ(DMPlexGetCone(K,kPerm,&cone));
          CHKERRQ(DMPlexGetConeOrientation(K,kPerm,&orientation));
          for (l = 0; l < size; l++) {
            PetscInt q, m = (preO >= 0) ? ((preO + l) % size) : ((size -(preO + 1) - l) % size);
            PetscInt newO, lSize, oTrue;
            DMPolytopeType ct = DM_NUM_POLYTOPES;

            q                         = iperm[cone[m]];
            newCones[offset]          = Kembedding[q];
            CHKERRQ(DMPlexGetConeSize(K,q,&lSize));
            if (lSize == 2) ct = DM_POLYTOPE_SEGMENT;
            else if (lSize == 4) ct = DM_POLYTOPE_QUADRILATERAL;
            oTrue                     = DMPolytopeConvertNewOrientation_Internal(ct, orientation[m]);
            oTrue                     = ((!lSize) || (preOrient[k] >= 0)) ? oTrue : -(oTrue + 2);
            newO                      = DihedralCompose(lSize,oTrue,preOrient[q]);
            newOrientations[offset++] = DMPolytopeConvertOldOrientation_Internal(ct, newO);
          }
          if (kParent != 0) {
            PetscInt newPoint = Kembedding[kParent];
            CHKERRQ(PetscSectionGetOffset(parentSection,Kembedding[k],&pOffset));
            parents[pOffset]  = newPoint;
            childIDs[pOffset] = k;
          }
        }
      }
    }

    CHKERRQ(PetscMalloc1(dim*(pNewEnd[dim]-pNewStart[dim]),&newVertexCoords));

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

      if (PetscDefined(USE_DEBUG)) {
        PetscInt k;
        CHKERRQ(DMPlexGetHeightStratum(K,0,&kStart,&kEnd));
        for (k = kStart; k < kEnd; k++) {
          CHKERRQ(DMPlexComputeCellGeometryFEM(K, k, NULL, v0, J, NULL, &detJ));
          PetscCheckFalse(detJ <= 0.,PETSC_COMM_SELF,PETSC_ERR_PLIB,"reference tree cell %d has bad determinant",k);
        }
      }
      CHKERRQ(DMPlexComputeCellGeometryFEM(dm, cell, NULL, v0, J, NULL, &detJ));
      CHKERRQ(DMGetCoordinateSection(dm,&vSection));
      CHKERRQ(DMGetCoordinatesLocal(dm,&coords));
      CHKERRQ(VecGetArray(coords,&coordvals));
      for (v = pOldStart[dim]; v < pOldEnd[dim]; v++) {

        CHKERRQ(PetscSectionGetDof(vSection,v,&dof));
        CHKERRQ(PetscSectionGetOffset(vSection,v,&off));
        for (l = 0; l < dof; l++) {
          newVertexCoords[offset++] = coordvals[off + l];
        }
      }
      CHKERRQ(VecRestoreArray(coords,&coordvals));

      CHKERRQ(DMGetCoordinateSection(K,&vSection));
      CHKERRQ(DMGetCoordinatesLocal(K,&coords));
      CHKERRQ(VecGetArray(coords,&coordvals));
      CHKERRQ(DMPlexGetDepthStratum(K,0,&kStart,&kEnd));
      for (v = kStart; v < kEnd; v++) {
        PetscReal coord[3], newCoord[3];
        PetscInt  vPerm = perm[v];
        PetscInt  kParent;
        const PetscReal xi0[3] = {-1.,-1.,-1.};

        CHKERRQ(DMPlexGetTreeParent(K,v,&kParent,NULL));
        if (kParent != v) {
          /* this is a new vertex */
          CHKERRQ(PetscSectionGetOffset(vSection,vPerm,&off));
          for (l = 0; l < dim; ++l) coord[l] = PetscRealPart(coordvals[off+l]);
          CoordinatesRefToReal(dim, dim, xi0, v0, J, coord, newCoord);
          for (l = 0; l < dim; ++l) newVertexCoords[offset+l] = newCoord[l];
          offset += dim;
        }
      }
      CHKERRQ(VecRestoreArray(coords,&coordvals));
    }

    /* need to reverse the order of pNewCount: vertices first, cells last */
    for (d = 0; d < (dim + 1) / 2; d++) {
      PetscInt tmp;

      tmp = pNewCount[d];
      pNewCount[d] = pNewCount[dim - d];
      pNewCount[dim - d] = tmp;
    }

    CHKERRQ(DMPlexCreateFromDAG(*ncdm,dim,pNewCount,newConeSizes,newCones,newOrientations,newVertexCoords));
    CHKERRQ(DMPlexSetReferenceTree(*ncdm,K));
    CHKERRQ(DMPlexSetTree(*ncdm,parentSection,parents,childIDs));

    /* clean up */
    CHKERRQ(DMPlexRestoreTransitiveClosure(dm,cell,PETSC_TRUE,&nc,&cellClosure));
    CHKERRQ(PetscFree5(pNewCount,pNewStart,pNewEnd,pOldStart,pOldEnd));
    CHKERRQ(PetscFree(newConeSizes));
    CHKERRQ(PetscFree2(newCones,newOrientations));
    CHKERRQ(PetscFree(newVertexCoords));
    CHKERRQ(PetscFree2(parents,childIDs));
    CHKERRQ(PetscFree4(Kembedding,perm,iperm,preOrient));
  }
  else {
    PetscInt    p, counts[4];
    PetscInt    *coneSizes, *cones, *orientations;
    Vec         coordVec;
    PetscScalar *coords;

    for (d = 0; d <= dim; d++) {
      PetscInt dStart, dEnd;

      CHKERRQ(DMPlexGetDepthStratum(dm,d,&dStart,&dEnd));
      counts[d] = dEnd - dStart;
    }
    CHKERRQ(PetscMalloc1(pEnd-pStart,&coneSizes));
    for (p = pStart; p < pEnd; p++) {
      CHKERRQ(DMPlexGetConeSize(dm,p,&coneSizes[p-pStart]));
    }
    CHKERRQ(DMPlexGetCones(dm, &cones));
    CHKERRQ(DMPlexGetConeOrientations(dm, &orientations));
    CHKERRQ(DMGetCoordinatesLocal(dm,&coordVec));
    CHKERRQ(VecGetArray(coordVec,&coords));

    CHKERRQ(PetscSectionSetChart(parentSection,pStart,pEnd));
    CHKERRQ(PetscSectionSetUp(parentSection));
    CHKERRQ(DMPlexCreateFromDAG(*ncdm,dim,counts,coneSizes,cones,orientations,NULL));
    CHKERRQ(DMPlexSetReferenceTree(*ncdm,K));
    CHKERRQ(DMPlexSetTree(*ncdm,parentSection,NULL,NULL));
    CHKERRQ(VecRestoreArray(coordVec,&coords));
  }
  CHKERRQ(PetscSectionDestroy(&parentSection));

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

  PetscFunctionBegin;
  CHKERRQ(DMPlexGetChart(coarse,&pStartC,&pEndC));
  CHKERRQ(DMPlexGetChart(fine,&pStartF,&pEndF));
  CHKERRQ(DMGetGlobalSection(fine,&globalFine));
  { /* winnow fine points that don't have global dofs out of the sf */
    PetscInt dof, cdof, numPointsWithDofs, offset, *pointsWithDofs, nleaves, l;
    const PetscInt *leaves;

    CHKERRQ(PetscSFGetGraph(coarseToFine,NULL,&nleaves,&leaves,NULL));
    for (l = 0, numPointsWithDofs = 0; l < nleaves; l++) {
      p = leaves ? leaves[l] : l;
      CHKERRQ(PetscSectionGetDof(globalFine,p,&dof));
      CHKERRQ(PetscSectionGetConstraintDof(globalFine,p,&cdof));
      if ((dof - cdof) > 0) {
        numPointsWithDofs++;
      }
    }
    CHKERRQ(PetscMalloc1(numPointsWithDofs,&pointsWithDofs));
    for (l = 0, offset = 0; l < nleaves; l++) {
      p = leaves ? leaves[l] : l;
      CHKERRQ(PetscSectionGetDof(globalFine,p,&dof));
      CHKERRQ(PetscSectionGetConstraintDof(globalFine,p,&cdof));
      if ((dof - cdof) > 0) {
        pointsWithDofs[offset++] = l;
      }
    }
    CHKERRQ(PetscSFCreateEmbeddedLeafSF(coarseToFine, numPointsWithDofs, pointsWithDofs, &coarseToFineEmbedded));
    CHKERRQ(PetscFree(pointsWithDofs));
  }
  /* communicate back to the coarse mesh which coarse points have children (that may require interpolation) */
  CHKERRQ(PetscMalloc1(pEndC-pStartC,&maxChildIds));
  for (p = pStartC; p < pEndC; p++) {
    maxChildIds[p - pStartC] = -2;
  }
  CHKERRQ(PetscSFReduceBegin(coarseToFineEmbedded,MPIU_INT,childIds,maxChildIds,MPIU_MAX));
  CHKERRQ(PetscSFReduceEnd(coarseToFineEmbedded,MPIU_INT,childIds,maxChildIds,MPIU_MAX));

  CHKERRQ(DMGetLocalSection(coarse,&localCoarse));
  CHKERRQ(DMGetGlobalSection(coarse,&globalCoarse));

  CHKERRQ(DMPlexGetAnchors(coarse,&aSec,&aIS));
  CHKERRQ(ISGetIndices(aIS,&anchors));
  CHKERRQ(PetscSectionGetChart(aSec,&aStart,&aEnd));

  CHKERRQ(DMGetDefaultConstraints(coarse,&cSec,&cMat,NULL));
  CHKERRQ(PetscSectionGetChart(cSec,&cStart,&cEnd));

  /* create sections that will send to children the indices and matrices they will need to construct the interpolator */
  CHKERRQ(PetscSectionCreate(PetscObjectComm((PetscObject)coarse),&rootIndicesSec));
  CHKERRQ(PetscSectionCreate(PetscObjectComm((PetscObject)coarse),&rootMatricesSec));
  CHKERRQ(PetscSectionSetChart(rootIndicesSec,pStartC,pEndC));
  CHKERRQ(PetscSectionSetChart(rootMatricesSec,pStartC,pEndC));
  CHKERRQ(PetscSectionGetNumFields(localCoarse,&numFields));
  maxFields = PetscMax(1,numFields);
  CHKERRQ(PetscMalloc7(maxFields+1,&offsets,maxFields+1,&offsetsCopy,maxFields+1,&newOffsets,maxFields+1,&newOffsetsCopy,maxFields+1,&rowOffsets,maxFields+1,&numD,maxFields+1,&numO));
  CHKERRQ(PetscMalloc2(maxFields+1,(PetscInt****)&perms,maxFields+1,(PetscScalar****)&flips));
  CHKERRQ(PetscMemzero((void *) perms, (maxFields+1) * sizeof(const PetscInt **)));
  CHKERRQ(PetscMemzero((void *) flips, (maxFields+1) * sizeof(const PetscScalar **)));

  for (p = pStartC; p < pEndC; p++) { /* count the sizes of the indices and matrices */
    PetscInt dof, matSize   = 0;
    PetscInt aDof           = 0;
    PetscInt cDof           = 0;
    PetscInt maxChildId     = maxChildIds[p - pStartC];
    PetscInt numRowIndices  = 0;
    PetscInt numColIndices  = 0;
    PetscInt f;

    CHKERRQ(PetscSectionGetDof(globalCoarse,p,&dof));
    if (dof < 0) {
      dof = -(dof + 1);
    }
    if (p >= aStart && p < aEnd) {
      CHKERRQ(PetscSectionGetDof(aSec,p,&aDof));
    }
    if (p >= cStart && p < cEnd) {
      CHKERRQ(PetscSectionGetDof(cSec,p,&cDof));
    }
    for (f = 0; f <= numFields; f++) offsets[f] = 0;
    for (f = 0; f <= numFields; f++) newOffsets[f] = 0;
    if (maxChildId >= 0) { /* this point has children (with dofs) that will need to be interpolated from the closure of p */
      PetscInt *closure = NULL, closureSize, cl;

      CHKERRQ(DMPlexGetTransitiveClosure(coarse,p,PETSC_TRUE,&closureSize,&closure));
      for (cl = 0; cl < closureSize; cl++) { /* get the closure */
        PetscInt c = closure[2 * cl], clDof;

        CHKERRQ(PetscSectionGetDof(localCoarse,c,&clDof));
        numRowIndices += clDof;
        for (f = 0; f < numFields; f++) {
          CHKERRQ(PetscSectionGetFieldDof(localCoarse,c,f,&clDof));
          offsets[f + 1] += clDof;
        }
      }
      for (f = 0; f < numFields; f++) {
        offsets[f + 1]   += offsets[f];
        newOffsets[f + 1] = offsets[f + 1];
      }
      /* get the number of indices needed and their field offsets */
      CHKERRQ(DMPlexAnchorsModifyMat(coarse,localCoarse,closureSize,numRowIndices,closure,NULL,NULL,NULL,&numColIndices,NULL,NULL,newOffsets,PETSC_FALSE));
      CHKERRQ(DMPlexRestoreTransitiveClosure(coarse,p,PETSC_TRUE,&closureSize,&closure));
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

        CHKERRQ(PetscSectionGetOffset(aSec,p,&aOff));
        for (f = 0; f < numFields; f++) {
          PetscInt fDof;

          CHKERRQ(PetscSectionGetFieldDof(localCoarse,p,f,&fDof));
          offsets[f+1] = fDof;
        }
        for (a = 0; a < aDof; a++) {
          PetscInt anchor = anchors[a + aOff], aLocalDof;

          CHKERRQ(PetscSectionGetDof(localCoarse,anchor,&aLocalDof));
          numColIndices += aLocalDof;
          for (f = 0; f < numFields; f++) {
            PetscInt fDof;

            CHKERRQ(PetscSectionGetFieldDof(localCoarse,anchor,f,&fDof));
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
    CHKERRQ(PetscSectionSetDof(rootIndicesSec,p,numColIndices ? numColIndices+2*numFields : 0));
    CHKERRQ(PetscSectionSetDof(rootMatricesSec,p,matSize));
  }
  CHKERRQ(PetscSectionSetUp(rootIndicesSec));
  CHKERRQ(PetscSectionSetUp(rootMatricesSec));
  {
    PetscInt numRootIndices, numRootMatrices;

    CHKERRQ(PetscSectionGetStorageSize(rootIndicesSec,&numRootIndices));
    CHKERRQ(PetscSectionGetStorageSize(rootMatricesSec,&numRootMatrices));
    CHKERRQ(PetscMalloc2(numRootIndices,&rootIndices,numRootMatrices,&rootMatrices));
    for (p = pStartC; p < pEndC; p++) {
      PetscInt    numRowIndices, numColIndices, matSize, dof;
      PetscInt    pIndOff, pMatOff, f;
      PetscInt    *pInd;
      PetscInt    maxChildId = maxChildIds[p - pStartC];
      PetscScalar *pMat = NULL;

      CHKERRQ(PetscSectionGetDof(rootIndicesSec,p,&numColIndices));
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
      CHKERRQ(PetscSectionGetOffset(rootIndicesSec,p,&pIndOff));
      pInd = &(rootIndices[pIndOff]);
      CHKERRQ(PetscSectionGetDof(rootMatricesSec,p,&matSize));
      if (matSize) {
        CHKERRQ(PetscSectionGetOffset(rootMatricesSec,p,&pMatOff));
        pMat = &rootMatrices[pMatOff];
      }
      CHKERRQ(PetscSectionGetDof(globalCoarse,p,&dof));
      if (dof < 0) {
        dof = -(dof + 1);
      }
      if (maxChildId >= 0) { /* build an identity matrix, apply matrix constraints on the right */
        PetscInt i, j;
        PetscInt numRowIndices = matSize / numColIndices;

        if (!numRowIndices) { /* don't need to calculate the mat, just the indices */
          PetscInt numIndices, *indices;
          CHKERRQ(DMPlexGetClosureIndices(coarse,localCoarse,globalCoarse,p,PETSC_TRUE,&numIndices,&indices,offsets,NULL));
          PetscCheckFalse(numIndices != numColIndices,PETSC_COMM_SELF,PETSC_ERR_PLIB,"mismatching constraint indices calculations");
          for (i = 0; i < numColIndices; i++) {
            pInd[i] = indices[i];
          }
          for (i = 0; i < numFields; i++) {
            pInd[numColIndices + i]             = offsets[i+1];
            pInd[numColIndices + numFields + i] = offsets[i+1];
          }
          CHKERRQ(DMPlexRestoreClosureIndices(coarse,localCoarse,globalCoarse,p,PETSC_TRUE,&numIndices,&indices,offsets,NULL));
        }
        else {
          PetscInt closureSize, *closure = NULL, cl;
          PetscScalar *pMatIn, *pMatModified;
          PetscInt numPoints,*points;

          CHKERRQ(DMGetWorkArray(coarse,numRowIndices * numRowIndices,MPIU_SCALAR,&pMatIn));
          for (i = 0; i < numRowIndices; i++) { /* initialize to the identity */
            for (j = 0; j < numRowIndices; j++) {
              pMatIn[i * numRowIndices + j] = (i == j) ? 1. : 0.;
            }
          }
          CHKERRQ(DMPlexGetTransitiveClosure(coarse, p, PETSC_TRUE, &closureSize, &closure));
          for (f = 0; f < maxFields; f++) {
            if (numFields) CHKERRQ(PetscSectionGetFieldPointSyms(localCoarse,f,closureSize,closure,&perms[f],&flips[f]));
            else           CHKERRQ(PetscSectionGetPointSyms(localCoarse,closureSize,closure,&perms[f],&flips[f]));
          }
          if (numFields) {
            for (cl = 0; cl < closureSize; cl++) {
              PetscInt c = closure[2 * cl];

              for (f = 0; f < numFields; f++) {
                PetscInt fDof;

                CHKERRQ(PetscSectionGetFieldDof(localCoarse,c,f,&fDof));
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
          CHKERRQ(DMPlexAnchorsModifyMat(coarse,localCoarse,closureSize,numRowIndices,closure,perms,pMatIn,&numPoints,NULL,&points,&pMatModified,newOffsets,PETSC_FALSE));
          for (f = 0; f < maxFields; f++) {
            if (numFields) CHKERRQ(PetscSectionRestoreFieldPointSyms(localCoarse,f,closureSize,closure,&perms[f],&flips[f]));
            else           CHKERRQ(PetscSectionRestorePointSyms(localCoarse,closureSize,closure,&perms[f],&flips[f]));
          }
          for (f = 0; f < maxFields; f++) {
            if (numFields) CHKERRQ(PetscSectionGetFieldPointSyms(localCoarse,f,numPoints,points,&perms[f],&flips[f]));
            else           CHKERRQ(PetscSectionGetPointSyms(localCoarse,numPoints,points,&perms[f],&flips[f]));
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
          CHKERRQ(DMRestoreWorkArray(coarse,numRowIndices * numColIndices,MPIU_SCALAR,&pMatModified));
          CHKERRQ(DMPlexRestoreTransitiveClosure(coarse, p, PETSC_TRUE, &closureSize, &closure));
          CHKERRQ(DMRestoreWorkArray(coarse,numRowIndices * numColIndices,MPIU_SCALAR,&pMatIn));
          if (numFields) {
            for (f = 0; f < numFields; f++) {
              pInd[numColIndices + f]             = offsets[f+1];
              pInd[numColIndices + numFields + f] = newOffsets[f+1];
            }
            for (cl = 0; cl < numPoints; cl++) {
              PetscInt globalOff, c = points[2*cl];
              CHKERRQ(PetscSectionGetOffset(globalCoarse, c, &globalOff));
              CHKERRQ(DMPlexGetIndicesPointFields_Internal(localCoarse, PETSC_FALSE, c, globalOff < 0 ? -(globalOff+1) : globalOff, newOffsets, PETSC_FALSE, perms, cl, NULL, pInd));
            }
          } else {
            for (cl = 0; cl < numPoints; cl++) {
              PetscInt c = points[2*cl], globalOff;
              const PetscInt *perm = perms[0] ? perms[0][cl] : NULL;

              CHKERRQ(PetscSectionGetOffset(globalCoarse, c, &globalOff));
              CHKERRQ(DMPlexGetIndicesPoint_Internal(localCoarse, PETSC_FALSE, c, globalOff < 0 ? -(globalOff+1) : globalOff, newOffsets, PETSC_FALSE, perm, NULL, pInd));
            }
          }
          for (f = 0; f < maxFields; f++) {
            if (numFields) CHKERRQ(PetscSectionRestoreFieldPointSyms(localCoarse,f,numPoints,points,&perms[f],&flips[f]));
            else           CHKERRQ(PetscSectionRestorePointSyms(localCoarse,numPoints,points,&perms[f],&flips[f]));
          }
          CHKERRQ(DMRestoreWorkArray(coarse,numPoints,MPIU_SCALAR,&points));
        }
      }
      else if (matSize) {
        PetscInt cOff;
        PetscInt *rowIndices, *colIndices, a, aDof, aOff;

        numRowIndices = matSize / numColIndices;
        PetscCheckFalse(numRowIndices != dof,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Miscounted dofs");
        CHKERRQ(DMGetWorkArray(coarse,numRowIndices,MPIU_INT,&rowIndices));
        CHKERRQ(DMGetWorkArray(coarse,numColIndices,MPIU_INT,&colIndices));
        CHKERRQ(PetscSectionGetOffset(cSec,p,&cOff));
        CHKERRQ(PetscSectionGetDof(aSec,p,&aDof));
        CHKERRQ(PetscSectionGetOffset(aSec,p,&aOff));
        if (numFields) {
          for (f = 0; f < numFields; f++) {
            PetscInt fDof;

            CHKERRQ(PetscSectionGetFieldDof(cSec,p,f,&fDof));
            offsets[f + 1] = fDof;
            for (a = 0; a < aDof; a++) {
              PetscInt anchor = anchors[a + aOff];
              CHKERRQ(PetscSectionGetFieldDof(localCoarse,anchor,f,&fDof));
              newOffsets[f + 1] += fDof;
            }
          }
          for (f = 0; f < numFields; f++) {
            offsets[f + 1]       += offsets[f];
            offsetsCopy[f + 1]    = offsets[f + 1];
            newOffsets[f + 1]    += newOffsets[f];
            newOffsetsCopy[f + 1] = newOffsets[f + 1];
          }
          CHKERRQ(DMPlexGetIndicesPointFields_Internal(cSec,PETSC_TRUE,p,cOff,offsetsCopy,PETSC_TRUE,NULL,-1, NULL,rowIndices));
          for (a = 0; a < aDof; a++) {
            PetscInt anchor = anchors[a + aOff], lOff;
            CHKERRQ(PetscSectionGetOffset(localCoarse,anchor,&lOff));
            CHKERRQ(DMPlexGetIndicesPointFields_Internal(localCoarse,PETSC_TRUE,anchor,lOff,newOffsetsCopy,PETSC_TRUE,NULL,-1, NULL,colIndices));
          }
        }
        else {
          CHKERRQ(DMPlexGetIndicesPoint_Internal(cSec,PETSC_TRUE,p,cOff,offsetsCopy,PETSC_TRUE,NULL, NULL,rowIndices));
          for (a = 0; a < aDof; a++) {
            PetscInt anchor = anchors[a + aOff], lOff;
            CHKERRQ(PetscSectionGetOffset(localCoarse,anchor,&lOff));
            CHKERRQ(DMPlexGetIndicesPoint_Internal(localCoarse,PETSC_TRUE,anchor,lOff,newOffsetsCopy,PETSC_TRUE,NULL, NULL,colIndices));
          }
        }
        if (numFields) {
          PetscInt count, a;

          for (f = 0, count = 0; f < numFields; f++) {
            PetscInt iSize = offsets[f + 1] - offsets[f];
            PetscInt jSize = newOffsets[f + 1] - newOffsets[f];
            CHKERRQ(MatGetValues(cMat,iSize,&rowIndices[offsets[f]],jSize,&colIndices[newOffsets[f]],&pMat[count]));
            count += iSize * jSize;
            pInd[numColIndices + f]             = offsets[f+1];
            pInd[numColIndices + numFields + f] = newOffsets[f+1];
          }
          for (a = 0; a < aDof; a++) {
            PetscInt anchor = anchors[a + aOff];
            PetscInt gOff;
            CHKERRQ(PetscSectionGetOffset(globalCoarse,anchor,&gOff));
            CHKERRQ(DMPlexGetIndicesPointFields_Internal(localCoarse,PETSC_FALSE,anchor,gOff < 0 ? -(gOff + 1) : gOff,newOffsets,PETSC_FALSE,NULL,-1, NULL,pInd));
          }
        }
        else {
          PetscInt a;
          CHKERRQ(MatGetValues(cMat,numRowIndices,rowIndices,numColIndices,colIndices,pMat));
          for (a = 0; a < aDof; a++) {
            PetscInt anchor = anchors[a + aOff];
            PetscInt gOff;
            CHKERRQ(PetscSectionGetOffset(globalCoarse,anchor,&gOff));
            CHKERRQ(DMPlexGetIndicesPoint_Internal(localCoarse,PETSC_FALSE,anchor,gOff < 0 ? -(gOff + 1) : gOff,newOffsets,PETSC_FALSE,NULL, NULL,pInd));
          }
        }
        CHKERRQ(DMRestoreWorkArray(coarse,numColIndices,MPIU_INT,&colIndices));
        CHKERRQ(DMRestoreWorkArray(coarse,numRowIndices,MPIU_INT,&rowIndices));
      }
      else {
        PetscInt gOff;

        CHKERRQ(PetscSectionGetOffset(globalCoarse,p,&gOff));
        if (numFields) {
          for (f = 0; f < numFields; f++) {
            PetscInt fDof;
            CHKERRQ(PetscSectionGetFieldDof(localCoarse,p,f,&fDof));
            offsets[f + 1] = fDof + offsets[f];
          }
          for (f = 0; f < numFields; f++) {
            pInd[numColIndices + f]             = offsets[f+1];
            pInd[numColIndices + numFields + f] = offsets[f+1];
          }
          CHKERRQ(DMPlexGetIndicesPointFields_Internal(localCoarse,PETSC_FALSE,p,gOff < 0 ? -(gOff + 1) : gOff,offsets,PETSC_FALSE,NULL,-1, NULL,pInd));
        } else {
          CHKERRQ(DMPlexGetIndicesPoint_Internal(localCoarse,PETSC_FALSE,p,gOff < 0 ? -(gOff + 1) : gOff,offsets,PETSC_FALSE,NULL, NULL,pInd));
        }
      }
    }
    CHKERRQ(PetscFree(maxChildIds));
  }
  {
    PetscSF  indicesSF, matricesSF;
    PetscInt *remoteOffsetsIndices, *remoteOffsetsMatrices, numLeafIndices, numLeafMatrices;

    CHKERRQ(PetscSectionCreate(PetscObjectComm((PetscObject)fine),&leafIndicesSec));
    CHKERRQ(PetscSectionCreate(PetscObjectComm((PetscObject)fine),&leafMatricesSec));
    CHKERRQ(PetscSFDistributeSection(coarseToFineEmbedded,rootIndicesSec,&remoteOffsetsIndices,leafIndicesSec));
    CHKERRQ(PetscSFDistributeSection(coarseToFineEmbedded,rootMatricesSec,&remoteOffsetsMatrices,leafMatricesSec));
    CHKERRQ(PetscSFCreateSectionSF(coarseToFineEmbedded,rootIndicesSec,remoteOffsetsIndices,leafIndicesSec,&indicesSF));
    CHKERRQ(PetscSFCreateSectionSF(coarseToFineEmbedded,rootMatricesSec,remoteOffsetsMatrices,leafMatricesSec,&matricesSF));
    CHKERRQ(PetscSFDestroy(&coarseToFineEmbedded));
    CHKERRQ(PetscFree(remoteOffsetsIndices));
    CHKERRQ(PetscFree(remoteOffsetsMatrices));
    CHKERRQ(PetscSectionGetStorageSize(leafIndicesSec,&numLeafIndices));
    CHKERRQ(PetscSectionGetStorageSize(leafMatricesSec,&numLeafMatrices));
    CHKERRQ(PetscMalloc2(numLeafIndices,&leafIndices,numLeafMatrices,&leafMatrices));
    CHKERRQ(PetscSFBcastBegin(indicesSF,MPIU_INT,rootIndices,leafIndices,MPI_REPLACE));
    CHKERRQ(PetscSFBcastBegin(matricesSF,MPIU_SCALAR,rootMatrices,leafMatrices,MPI_REPLACE));
    CHKERRQ(PetscSFBcastEnd(indicesSF,MPIU_INT,rootIndices,leafIndices,MPI_REPLACE));
    CHKERRQ(PetscSFBcastEnd(matricesSF,MPIU_SCALAR,rootMatrices,leafMatrices,MPI_REPLACE));
    CHKERRQ(PetscSFDestroy(&matricesSF));
    CHKERRQ(PetscSFDestroy(&indicesSF));
    CHKERRQ(PetscFree2(rootIndices,rootMatrices));
    CHKERRQ(PetscSectionDestroy(&rootIndicesSec));
    CHKERRQ(PetscSectionDestroy(&rootMatricesSec));
  }
  /* count to preallocate */
  CHKERRQ(DMGetLocalSection(fine,&localFine));
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

    CHKERRQ(PetscSectionGetConstrainedStorageSize(globalFine,&nGlobal));
    CHKERRQ(PetscCalloc2(nGlobal,&dnnz,nGlobal,&onnz));
    CHKERRQ(MatGetLayouts(mat,&rowMap,&colMap));
    CHKERRQ(PetscLayoutSetUp(rowMap));
    CHKERRQ(PetscLayoutSetUp(colMap));
    CHKERRQ(PetscLayoutGetRange(rowMap,&rowStart,&rowEnd));
    CHKERRQ(PetscLayoutGetRange(colMap,&colStart,&colEnd));
    CHKERRQ(PetscSectionGetMaxDof(localFine,&maxDof));
    CHKERRQ(PetscSectionGetChart(leafIndicesSec,&leafStart,&leafEnd));
    CHKERRQ(DMGetWorkArray(fine,maxDof,MPIU_INT,&rowIndices));
    for (p = leafStart; p < leafEnd; p++) {
      PetscInt    gDof, gcDof, gOff;
      PetscInt    numColIndices, pIndOff, *pInd;
      PetscInt    matSize;
      PetscInt    i;

      CHKERRQ(PetscSectionGetDof(globalFine,p,&gDof));
      CHKERRQ(PetscSectionGetConstraintDof(globalFine,p,&gcDof));
      if ((gDof - gcDof) <= 0) {
        continue;
      }
      CHKERRQ(PetscSectionGetOffset(globalFine,p,&gOff));
      PetscCheckFalse(gOff < 0,PETSC_COMM_SELF,PETSC_ERR_PLIB,"I though having global dofs meant a non-negative offset");
      PetscCheckFalse((gOff < rowStart) || ((gOff + gDof - gcDof) > rowEnd),PETSC_COMM_SELF,PETSC_ERR_PLIB,"I thought the row map would constrain the global dofs");
      CHKERRQ(PetscSectionGetDof(leafIndicesSec,p,&numColIndices));
      CHKERRQ(PetscSectionGetOffset(leafIndicesSec,p,&pIndOff));
      numColIndices -= 2 * numFields;
      PetscCheckFalse(numColIndices <= 0,PETSC_COMM_SELF,PETSC_ERR_PLIB,"global fine dof with no dofs to interpolate from");
      pInd = &leafIndices[pIndOff];
      offsets[0]        = 0;
      offsetsCopy[0]    = 0;
      newOffsets[0]     = 0;
      newOffsetsCopy[0] = 0;
      if (numFields) {
        PetscInt f;
        for (f = 0; f < numFields; f++) {
          PetscInt rowDof;

          CHKERRQ(PetscSectionGetFieldDof(localFine,p,f,&rowDof));
          offsets[f + 1]        = offsets[f] + rowDof;
          offsetsCopy[f + 1]    = offsets[f + 1];
          newOffsets[f + 1]     = pInd[numColIndices + numFields + f];
          numD[f] = 0;
          numO[f] = 0;
        }
        CHKERRQ(DMPlexGetIndicesPointFields_Internal(localFine,PETSC_FALSE,p,gOff,offsetsCopy,PETSC_FALSE,NULL,-1, NULL,rowIndices));
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
        CHKERRQ(DMPlexGetIndicesPoint_Internal(localFine,PETSC_FALSE,p,gOff,offsetsCopy,PETSC_FALSE,NULL, NULL,rowIndices));
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
      CHKERRQ(PetscSectionGetDof(leafMatricesSec,p,&matSize));
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
                  PetscCheckFalse(gIndFine < rowStart || gIndFine >= rowEnd,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Mismatched number of constrained dofs");
                  dnnz[gIndFine - rowStart] = 1;
                }
                else if (gIndCoarse >= 0) { /* remote */
                  PetscCheckFalse(gIndFine < rowStart || gIndFine >= rowEnd,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Mismatched number of constrained dofs");
                  onnz[gIndFine - rowStart] = 1;
                }
                else { /* constrained */
                  PetscCheckFalse(gIndFine >= 0,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Mismatched number of constrained dofs");
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
                PetscCheckFalse(gIndFine < rowStart || gIndFine >= rowEnd,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Mismatched number of constrained dofs");
                dnnz[gIndFine - rowStart] = 1;
              }
              else if (gIndCoarse >= 0) { /* remote */
                PetscCheckFalse(gIndFine < rowStart || gIndFine >= rowEnd,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Mismatched number of constrained dofs");
                onnz[gIndFine - rowStart] = 1;
              }
              else { /* constrained */
                PetscCheckFalse(gIndFine >= 0,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Mismatched number of constrained dofs");
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
                  PetscCheckFalse(gIndFine < rowStart || gIndFine >= rowEnd,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Mismatched number of constrained dofs");
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
                PetscCheckFalse(gIndFine < rowStart || gIndFine >= rowEnd,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Mismatched number of constrained dofs");
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
                PetscCheckFalse(gIndFine < rowStart || gIndFine >= rowEnd,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Mismatched number of constrained dofs");
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
              PetscCheckFalse(gIndFine < rowStart || gIndFine >= rowEnd,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Mismatched number of constrained dofs");
              dnnz[gIndFine - rowStart] = numD[0];
              onnz[gIndFine - rowStart] = numO[0];
            }
          }
        }
      }
    }
    CHKERRQ(MatXAIJSetPreallocation(mat,1,dnnz,onnz,NULL,NULL));
    CHKERRQ(PetscFree2(dnnz,onnz));

    CHKERRQ(DMPlexGetReferenceTree(fine,&refTree));
    CHKERRQ(DMPlexReferenceTreeGetChildrenMatrices(refTree,&refPointFieldMats,&refPointFieldN));
    CHKERRQ(DMGetDefaultConstraints(refTree,&refConSec,NULL,NULL));
    CHKERRQ(DMPlexGetAnchors(refTree,&refAnSec,NULL));
    CHKERRQ(PetscSectionGetChart(refConSec,&pRefStart,&pRefEnd));
    CHKERRQ(PetscSectionGetMaxDof(refConSec,&maxConDof));
    CHKERRQ(PetscSectionGetMaxDof(leafIndicesSec,&maxColumns));
    CHKERRQ(PetscMalloc1(maxConDof*maxColumns,&pointWork));
    for (p = leafStart; p < leafEnd; p++) {
      PetscInt gDof, gcDof, gOff;
      PetscInt numColIndices, pIndOff, *pInd;
      PetscInt matSize;
      PetscInt childId;

      CHKERRQ(PetscSectionGetDof(globalFine,p,&gDof));
      CHKERRQ(PetscSectionGetConstraintDof(globalFine,p,&gcDof));
      if ((gDof - gcDof) <= 0) {
        continue;
      }
      childId = childIds[p-pStartF];
      CHKERRQ(PetscSectionGetOffset(globalFine,p,&gOff));
      CHKERRQ(PetscSectionGetDof(leafIndicesSec,p,&numColIndices));
      CHKERRQ(PetscSectionGetOffset(leafIndicesSec,p,&pIndOff));
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

          CHKERRQ(PetscSectionGetFieldDof(localFine,p,f,&rowDof));
          offsets[f + 1]     = offsets[f] + rowDof;
          offsetsCopy[f + 1] = offsets[f + 1];
          rowOffsets[f + 1]  = pInd[numColIndices + f];
          newOffsets[f + 1]  = pInd[numColIndices + numFields + f];
        }
        CHKERRQ(DMPlexGetIndicesPointFields_Internal(localFine,PETSC_FALSE,p,gOff,offsetsCopy,PETSC_FALSE,NULL,-1, NULL,rowIndices));
      }
      else {
        CHKERRQ(DMPlexGetIndicesPoint_Internal(localFine,PETSC_FALSE,p,gOff,offsetsCopy,PETSC_FALSE,NULL, NULL,rowIndices));
      }
      CHKERRQ(PetscSectionGetDof(leafMatricesSec,p,&matSize));
      if (!matSize) { /* incoming matrix is identity */
        if (childId < 0) { /* no child interpolation: scatter */
          if (numFields) {
            PetscInt f;
            for (f = 0; f < numFields; f++) {
              PetscInt numRows = offsets[f+1] - offsets[f], row;
              for (row = 0; row < numRows; row++) {
                CHKERRQ(MatSetValue(mat,rowIndices[offsets[f]+row],pInd[newOffsets[f]+row],1.,INSERT_VALUES));
              }
            }
          }
          else {
            PetscInt numRows = gDof, row;
            for (row = 0; row < numRows; row++) {
              CHKERRQ(MatSetValue(mat,rowIndices[row],pInd[row],1.,INSERT_VALUES));
            }
          }
        }
        else { /* interpolate from all */
          if (numFields) {
            PetscInt f;
            for (f = 0; f < numFields; f++) {
              PetscInt numRows = offsets[f+1] - offsets[f];
              PetscInt numCols = newOffsets[f + 1] - newOffsets[f];
              CHKERRQ(MatSetValues(mat,numRows,&rowIndices[offsets[f]],numCols,&pInd[newOffsets[f]],refPointFieldMats[childId - pRefStart][f],INSERT_VALUES));
            }
          }
          else {
            CHKERRQ(MatSetValues(mat,gDof,rowIndices,numColIndices,pInd,refPointFieldMats[childId - pRefStart][0],INSERT_VALUES));
          }
        }
      }
      else { /* interpolate from all */
        PetscInt    pMatOff;
        PetscScalar *pMat;

        CHKERRQ(PetscSectionGetOffset(leafMatricesSec,p,&pMatOff));
        pMat = &leafMatrices[pMatOff];
        if (childId < 0) { /* copy the incoming matrix */
          if (numFields) {
            PetscInt f, count;
            for (f = 0, count = 0; f < numFields; f++) {
              PetscInt numRows = offsets[f+1]-offsets[f];
              PetscInt numCols = newOffsets[f+1]-newOffsets[f];
              PetscInt numInRows = rowOffsets[f+1]-rowOffsets[f];
              PetscScalar *inMat = &pMat[count];

              CHKERRQ(MatSetValues(mat,numRows,&rowIndices[offsets[f]],numCols,&pInd[newOffsets[f]],inMat,INSERT_VALUES));
              count += numCols * numInRows;
            }
          }
          else {
            CHKERRQ(MatSetValues(mat,gDof,rowIndices,numColIndices,pInd,pMat,INSERT_VALUES));
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
              PetscCheckFalse(refPointFieldN[childId - pRefStart][f] != numInRows,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Point constraint matrix multiply dimension mismatch");
              for (i = 0; i < numRows; i++) {
                for (j = 0; j < numCols; j++) {
                  PetscScalar val = 0.;
                  for (k = 0; k < numInRows; k++) {
                    val += refPointFieldMats[childId - pRefStart][f][i * numInRows + k] * inMat[k * numCols + j];
                  }
                  pointWork[i * numCols + j] = val;
                }
              }
              CHKERRQ(MatSetValues(mat,numRows,&rowIndices[offsets[f]],numCols,&pInd[newOffsets[f]],pointWork,INSERT_VALUES));
              count += numCols * numInRows;
            }
          }
          else { /* every dof gets a full row */
            PetscInt numRows   = gDof;
            PetscInt numCols   = numColIndices;
            PetscInt numInRows = matSize / numColIndices;
            PetscInt i, j, k;
            PetscCheckFalse(refPointFieldN[childId - pRefStart][0] != numInRows,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Point constraint matrix multiply dimension mismatch");
            for (i = 0; i < numRows; i++) {
              for (j = 0; j < numCols; j++) {
                PetscScalar val = 0.;
                for (k = 0; k < numInRows; k++) {
                  val += refPointFieldMats[childId - pRefStart][0][i * numInRows + k] * pMat[k * numCols + j];
                }
                pointWork[i * numCols + j] = val;
              }
            }
            CHKERRQ(MatSetValues(mat,numRows,rowIndices,numCols,pInd,pointWork,INSERT_VALUES));
          }
        }
      }
    }
    CHKERRQ(DMPlexReferenceTreeRestoreChildrenMatrices(refTree,&refPointFieldMats,&refPointFieldN));
    CHKERRQ(DMRestoreWorkArray(fine,maxDof,MPIU_INT,&rowIndices));
    CHKERRQ(PetscFree(pointWork));
  }
  CHKERRQ(MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY));
  CHKERRQ(PetscSectionDestroy(&leafIndicesSec));
  CHKERRQ(PetscSectionDestroy(&leafMatricesSec));
  CHKERRQ(PetscFree2(leafIndices,leafMatrices));
  CHKERRQ(PetscFree2(*(PetscInt****)&perms,*(PetscScalar****)&flips));
  CHKERRQ(PetscFree7(offsets,offsetsCopy,newOffsets,newOffsetsCopy,rowOffsets,numD,numO));
  CHKERRQ(ISRestoreIndices(aIS,&anchors));
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

  PetscFunctionBegin;
  CHKERRQ(DMGetLocalSection(refTree,&section));
  CHKERRQ(DMGetDimension(refTree, &dim));
  CHKERRQ(PetscMalloc6(dim,&v0,dim,&v0parent,dim,&vtmp,dim*dim,&J,dim*dim,&Jparent,dim*dim,&invJ));
  CHKERRQ(PetscMalloc2(dim,&pointScalar,dim,&pointRef));
  CHKERRQ(DMGetDS(refTree,&ds));
  CHKERRQ(PetscDSGetNumFields(ds,&numFields));
  CHKERRQ(PetscSectionGetNumFields(section,&numSecFields));
  CHKERRQ(DMGetLabel(refTree,"canonical",&canonical));
  CHKERRQ(DMGetLabel(refTree,"depth",&depth));
  CHKERRQ(DMGetDefaultConstraints(refTree,&cSection,&cMat,NULL));
  CHKERRQ(DMPlexGetChart(refTree, &pStart, &pEnd));
  CHKERRQ(DMPlexGetHeightStratum(refTree, 0, &cStart, &cEnd));
  CHKERRQ(MatGetSize(cMat,&n,&m)); /* the injector has transpose sizes from the constraint matrix */
  /* Step 1: compute non-zero pattern.  A proper subset of constraint matrix non-zero */
  CHKERRQ(PetscCalloc1(m,&nnz));
  for (p = pStart; p < pEnd; p++) { /* a point will have non-zeros if it is canonical, it has dofs, and its children have dofs */
    const PetscInt *children;
    PetscInt numChildren;
    PetscInt i, numChildDof, numSelfDof;

    if (canonical) {
      PetscInt pCanonical;
      CHKERRQ(DMLabelGetValue(canonical,p,&pCanonical));
      if (p != pCanonical) continue;
    }
    CHKERRQ(DMPlexGetTreeChildren(refTree,p,&numChildren,&children));
    if (!numChildren) continue;
    for (i = 0, numChildDof = 0; i < numChildren; i++) {
      PetscInt child = children[i];
      PetscInt dof;

      CHKERRQ(PetscSectionGetDof(section,child,&dof));
      numChildDof += dof;
    }
    CHKERRQ(PetscSectionGetDof(section,p,&numSelfDof));
    if (!numChildDof || !numSelfDof) continue;
    for (f = 0; f < numFields; f++) {
      PetscInt selfOff;

      if (numSecFields) { /* count the dofs for just this field */
        for (i = 0, numChildDof = 0; i < numChildren; i++) {
          PetscInt child = children[i];
          PetscInt dof;

          CHKERRQ(PetscSectionGetFieldDof(section,child,f,&dof));
          numChildDof += dof;
        }
        CHKERRQ(PetscSectionGetFieldDof(section,p,f,&numSelfDof));
        CHKERRQ(PetscSectionGetFieldOffset(section,p,f,&selfOff));
      }
      else {
        CHKERRQ(PetscSectionGetOffset(section,p,&selfOff));
      }
      for (i = 0; i < numSelfDof; i++) {
        nnz[selfOff + i] = numChildDof;
      }
    }
  }
  CHKERRQ(MatCreateAIJ(PETSC_COMM_SELF,m,n,m,n,-1,nnz,-1,NULL,&mat));
  CHKERRQ(PetscFree(nnz));
  /* Setp 2: compute entries */
  for (p = pStart; p < pEnd; p++) {
    const PetscInt *children;
    PetscInt numChildren;
    PetscInt i, numChildDof, numSelfDof;

    /* same conditions about when entries occur */
    if (canonical) {
      PetscInt pCanonical;
      CHKERRQ(DMLabelGetValue(canonical,p,&pCanonical));
      if (p != pCanonical) continue;
    }
    CHKERRQ(DMPlexGetTreeChildren(refTree,p,&numChildren,&children));
    if (!numChildren) continue;
    for (i = 0, numChildDof = 0; i < numChildren; i++) {
      PetscInt child = children[i];
      PetscInt dof;

      CHKERRQ(PetscSectionGetDof(section,child,&dof));
      numChildDof += dof;
    }
    CHKERRQ(PetscSectionGetDof(section,p,&numSelfDof));
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

          CHKERRQ(PetscSectionGetFieldDof(section,child,f,&dof));
          numChildDof += dof;
        }
        CHKERRQ(PetscSectionGetFieldDof(section,p,f,&numSelfDof));
        CHKERRQ(PetscSectionGetFieldOffset(section,p,f,&selfOff));
      }
      else {
        CHKERRQ(PetscSectionGetOffset(section,p,&selfOff));
      }

      /* find a cell whose closure contains p */
      if (p >= cStart && p < cEnd) {
        parentCell = p;
      }
      else {
        PetscInt *star = NULL;
        PetscInt numStar;

        parentCell = -1;
        CHKERRQ(DMPlexGetTransitiveClosure(refTree,p,PETSC_FALSE,&numStar,&star));
        for (i = numStar - 1; i >= 0; i--) {
          PetscInt c = star[2 * i];

          if (c >= cStart && c < cEnd) {
            parentCell = c;
            break;
          }
        }
        CHKERRQ(DMPlexRestoreTransitiveClosure(refTree,p,PETSC_FALSE,&numStar,&star));
      }
      /* determine the offset of p's shape functions within parentCell's shape functions */
      CHKERRQ(PetscDSGetDiscretization(ds,f,&disc));
      CHKERRQ(PetscObjectGetClassId(disc,&classId));
      if (classId == PETSCFE_CLASSID) {
        CHKERRQ(PetscFEGetDualSpace((PetscFE)disc,&dsp));
      }
      else if (classId == PETSCFV_CLASSID) {
        CHKERRQ(PetscFVGetDualSpace((PetscFV)disc,&dsp));
      }
      else {
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unsupported discretization object");
      }
      CHKERRQ(PetscDualSpaceGetNumDof(dsp,&depthNumDof));
      CHKERRQ(PetscDualSpaceGetNumComponents(dsp,&Nc));
      {
        PetscInt *closure = NULL;
        PetscInt numClosure;

        CHKERRQ(DMPlexGetTransitiveClosure(refTree,parentCell,PETSC_TRUE,&numClosure,&closure));
        for (i = 0, pI = -1, cellShapeOff = 0; i < numClosure; i++) {
          PetscInt point = closure[2 * i], pointDepth;

          pO = closure[2 * i + 1];
          if (point == p) {
            pI = i;
            break;
          }
          CHKERRQ(DMLabelGetValue(depth,point,&pointDepth));
          cellShapeOff += depthNumDof[pointDepth];
        }
        CHKERRQ(DMPlexRestoreTransitiveClosure(refTree,parentCell,PETSC_TRUE,&numClosure,&closure));
      }

      CHKERRQ(DMGetWorkArray(refTree, numSelfDof * numChildDof, MPIU_SCALAR,&pointMat));
      CHKERRQ(DMGetWorkArray(refTree, numSelfDof + numChildDof, MPIU_INT,&matRows));
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
            CHKERRQ(PetscSectionGetFieldDof(cSection,child,f,&dof));
            CHKERRQ(PetscSectionGetFieldOffset(cSection,child,f,&off));
          }
          else {
            CHKERRQ(PetscSectionGetDof(cSection,child,&dof));
            CHKERRQ(PetscSectionGetOffset(cSection,child,&off));
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

        CHKERRQ(PetscFEGetDualSpace(fe,&dsp));
        CHKERRQ(PetscDualSpaceGetDimension(dsp,&fSize));
        CHKERRQ(PetscDualSpaceGetSymmetries(dsp, &perms, &flips));
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

          CHKERRQ(PetscDualSpaceGetFunctional(dsp,parentCellShapeDof,&q));
          CHKERRQ(PetscQuadratureGetData(q,&dim,&thisNc,&numPoints,&points,&weights));
          PetscCheckFalse(thisNc != Nc,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Functional dim %D does not much basis dim %D",thisNc,Nc);
          CHKERRQ(PetscFECreateTabulation(fe,1,numPoints,points,0,&Tparent)); /* I'm expecting a nodal basis: weights[:]' * Bparent[:,cellShapeDof] = 1. */
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

              CHKERRQ(DMPlexGetTransitiveClosure(refTree,child,PETSC_FALSE,&numStar,&star));
              for (s = numStar - 1; s >= 0; s--) {
                PetscInt c = star[2 * s];

                if (c < cStart || c >= cEnd) continue;
                CHKERRQ(DMPlexLocatePoint_Internal(refTree,dim,point,c,&childCell));
                if (childCell >= 0) break;
              }
              CHKERRQ(DMPlexRestoreTransitiveClosure(refTree,child,PETSC_FALSE,&numStar,&star));
              if (childCell >= 0) break;
            }
            PetscCheckFalse(childCell < 0,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Could not locate quadrature point");
            CHKERRQ(DMPlexComputeCellGeometryFEM(refTree, childCell, NULL, v0, J, invJ, &detJ));
            CHKERRQ(DMPlexComputeCellGeometryFEM(refTree, parentCell, NULL, v0parent, Jparent, NULL, &detJparent));
            CoordinatesRefToReal(dim, dim, xi0, v0parent, Jparent, pointReal, vtmp);
            CoordinatesRealToRef(dim, dim, xi0, v0, invJ, vtmp, pointRef);

            CHKERRQ(PetscFECreateTabulation(fe,1,1,pointRef,0,&Tchild));
            CHKERRQ(DMPlexGetTransitiveClosure(refTree,childCell,PETSC_TRUE,&numClosure,&closure));
            for (k = 0, pointMatOff = 0; k < numChildren; k++) { /* point is located in cell => child dofs support at point are in closure of cell */
              PetscInt child = children[k], childDepth, childDof, childO = PETSC_MIN_INT;
              PetscInt l;
              const PetscInt *cperms;

              CHKERRQ(DMLabelGetValue(depth,child,&childDepth));
              childDof = depthNumDof[childDepth];
              for (l = 0, cI = -1, childCellShapeOff = 0; l < numClosure; l++) {
                PetscInt point = closure[2 * l];
                PetscInt pointDepth;

                childO = closure[2 * l + 1];
                if (point == child) {
                  cI = l;
                  break;
                }
                CHKERRQ(DMLabelGetValue(depth,point,&pointDepth));
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
            CHKERRQ(DMPlexRestoreTransitiveClosure(refTree,childCell,PETSC_TRUE,&numClosure,&closure));
            CHKERRQ(PetscTabulationDestroy(&Tchild));
          }
          CHKERRQ(PetscTabulationDestroy(&Tparent));
        }
      }
      else { /* just the volume-weighted averages of the children */
        PetscReal parentVol;
        PetscInt  childCell;

        CHKERRQ(DMPlexComputeCellGeometryFVM(refTree, p, &parentVol, NULL, NULL));
        for (i = 0, childCell = 0; i < numChildren; i++) {
          PetscInt  child = children[i], j;
          PetscReal childVol;

          if (child < cStart || child >= cEnd) continue;
          CHKERRQ(DMPlexComputeCellGeometryFVM(refTree, child, &childVol, NULL, NULL));
          for (j = 0; j < Nc; j++) {
            pointMat[j * numChildDof + Nc * childCell + j] = childVol / parentVol;
          }
          childCell++;
        }
      }
      /* Insert pointMat into mat */
      CHKERRQ(MatSetValues(mat,numSelfDof,matRows,numChildDof,matCols,pointMat,INSERT_VALUES));
      CHKERRQ(DMRestoreWorkArray(refTree, numSelfDof + numChildDof, MPIU_INT,&matRows));
      CHKERRQ(DMRestoreWorkArray(refTree, numSelfDof * numChildDof, MPIU_SCALAR,&pointMat));
    }
  }
  CHKERRQ(PetscFree6(v0,v0parent,vtmp,J,Jparent,invJ));
  CHKERRQ(PetscFree2(pointScalar,pointRef));
  CHKERRQ(MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY));
  *inj = mat;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexReferenceTreeGetChildrenMatrices_Injection(DM refTree, Mat inj, PetscScalar ****childrenMats)
{
  PetscDS        ds;
  PetscInt       numFields, f, pRefStart, pRefEnd, p, *rows, *cols, maxDof;
  PetscScalar    ***refPointFieldMats;
  PetscSection   refConSec, refSection;

  PetscFunctionBegin;
  CHKERRQ(DMGetDS(refTree,&ds));
  CHKERRQ(PetscDSGetNumFields(ds,&numFields));
  CHKERRQ(DMGetDefaultConstraints(refTree,&refConSec,NULL,NULL));
  CHKERRQ(DMGetLocalSection(refTree,&refSection));
  CHKERRQ(PetscSectionGetChart(refConSec,&pRefStart,&pRefEnd));
  CHKERRQ(PetscMalloc1(pRefEnd-pRefStart,&refPointFieldMats));
  CHKERRQ(PetscSectionGetMaxDof(refConSec,&maxDof));
  CHKERRQ(PetscMalloc1(maxDof,&rows));
  CHKERRQ(PetscMalloc1(maxDof*maxDof,&cols));
  for (p = pRefStart; p < pRefEnd; p++) {
    PetscInt parent, pDof, parentDof;

    CHKERRQ(DMPlexGetTreeParent(refTree,p,&parent,NULL));
    CHKERRQ(PetscSectionGetDof(refConSec,p,&pDof));
    CHKERRQ(PetscSectionGetDof(refSection,parent,&parentDof));
    if (!pDof || !parentDof || parent == p) continue;

    CHKERRQ(PetscMalloc1(numFields,&refPointFieldMats[p-pRefStart]));
    for (f = 0; f < numFields; f++) {
      PetscInt cDof, cOff, numCols, r;

      if (numFields > 1) {
        CHKERRQ(PetscSectionGetFieldDof(refConSec,p,f,&cDof));
        CHKERRQ(PetscSectionGetFieldOffset(refConSec,p,f,&cOff));
      }
      else {
        CHKERRQ(PetscSectionGetDof(refConSec,p,&cDof));
        CHKERRQ(PetscSectionGetOffset(refConSec,p,&cOff));
      }

      for (r = 0; r < cDof; r++) {
        rows[r] = cOff + r;
      }
      numCols = 0;
      {
        PetscInt aDof, aOff, j;

        if (numFields > 1) {
          CHKERRQ(PetscSectionGetFieldDof(refSection,parent,f,&aDof));
          CHKERRQ(PetscSectionGetFieldOffset(refSection,parent,f,&aOff));
        }
        else {
          CHKERRQ(PetscSectionGetDof(refSection,parent,&aDof));
          CHKERRQ(PetscSectionGetOffset(refSection,parent,&aOff));
        }

        for (j = 0; j < aDof; j++) {
          cols[numCols++] = aOff + j;
        }
      }
      CHKERRQ(PetscMalloc1(cDof*numCols,&refPointFieldMats[p-pRefStart][f]));
      /* transpose of constraint matrix */
      CHKERRQ(MatGetValues(inj,numCols,cols,cDof,rows,refPointFieldMats[p-pRefStart][f]));
    }
  }
  *childrenMats = refPointFieldMats;
  CHKERRQ(PetscFree(rows));
  CHKERRQ(PetscFree(cols));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexReferenceTreeRestoreChildrenMatrices_Injection(DM refTree, Mat inj, PetscScalar ****childrenMats)
{
  PetscDS        ds;
  PetscScalar    ***refPointFieldMats;
  PetscInt       numFields, pRefStart, pRefEnd, p, f;
  PetscSection   refConSec, refSection;

  PetscFunctionBegin;
  refPointFieldMats = *childrenMats;
  *childrenMats = NULL;
  CHKERRQ(DMGetDS(refTree,&ds));
  CHKERRQ(DMGetLocalSection(refTree,&refSection));
  CHKERRQ(PetscDSGetNumFields(ds,&numFields));
  CHKERRQ(DMGetDefaultConstraints(refTree,&refConSec,NULL,NULL));
  CHKERRQ(PetscSectionGetChart(refConSec,&pRefStart,&pRefEnd));
  for (p = pRefStart; p < pRefEnd; p++) {
    PetscInt parent, pDof, parentDof;

    CHKERRQ(DMPlexGetTreeParent(refTree,p,&parent,NULL));
    CHKERRQ(PetscSectionGetDof(refConSec,p,&pDof));
    CHKERRQ(PetscSectionGetDof(refSection,parent,&parentDof));
    if (!pDof || !parentDof || parent == p) continue;

    for (f = 0; f < numFields; f++) {
      PetscInt cDof;

      if (numFields > 1) {
        CHKERRQ(PetscSectionGetFieldDof(refConSec,p,f,&cDof));
      }
      else {
        CHKERRQ(PetscSectionGetDof(refConSec,p,&cDof));
      }

      CHKERRQ(PetscFree(refPointFieldMats[p - pRefStart][f]));
    }
    CHKERRQ(PetscFree(refPointFieldMats[p - pRefStart]));
  }
  CHKERRQ(PetscFree(refPointFieldMats));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexReferenceTreeGetInjector(DM refTree,Mat *injRef)
{
  Mat            cMatRef;
  PetscObject    injRefObj;

  PetscFunctionBegin;
  CHKERRQ(DMGetDefaultConstraints(refTree,NULL,&cMatRef,NULL));
  CHKERRQ(PetscObjectQuery((PetscObject)cMatRef,"DMPlexComputeInjectorTree_refTree",&injRefObj));
  *injRef = (Mat) injRefObj;
  if (!*injRef) {
    CHKERRQ(DMPlexComputeInjectorReferenceTree(refTree,injRef));
    CHKERRQ(PetscObjectCompose((PetscObject)cMatRef,"DMPlexComputeInjectorTree_refTree",(PetscObject)*injRef));
    /* there is now a reference in cMatRef, which should be the only one for symmetry with the above case */
    CHKERRQ(PetscObjectDereference((PetscObject)*injRef));
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

  PetscFunctionBegin;
  CHKERRQ(DMPlexGetChart(coarse,&pStartC,&pEndC));
  CHKERRQ(DMPlexGetChart(fine,&pStartF,&pEndF));
  CHKERRQ(DMGetLocalSection(fine,&localFine));
  CHKERRQ(DMGetGlobalSection(fine,&globalFine));
  CHKERRQ(PetscSectionCreate(PetscObjectComm((PetscObject)fine),&leafIndicesSec));
  CHKERRQ(PetscSectionSetChart(leafIndicesSec,pStartF, pEndF));
  CHKERRQ(PetscSectionGetMaxDof(localFine,&maxDof));
  { /* winnow fine points that don't have global dofs out of the sf */
    PetscInt l, nleaves, dof, cdof, numPointsWithDofs, offset, *pointsWithDofs, numIndices;
    const PetscInt *leaves;

    CHKERRQ(PetscSFGetGraph(coarseToFine,NULL,&nleaves,&leaves,NULL));
    for (l = 0, numPointsWithDofs = 0; l < nleaves; l++) {
      p    = leaves ? leaves[l] : l;
      CHKERRQ(PetscSectionGetDof(globalFine,p,&dof));
      CHKERRQ(PetscSectionGetConstraintDof(globalFine,p,&cdof));
      if ((dof - cdof) > 0) {
        numPointsWithDofs++;

        CHKERRQ(PetscSectionGetDof(localFine,p,&dof));
        CHKERRQ(PetscSectionSetDof(leafIndicesSec,p,dof + 1));
      }
    }
    CHKERRQ(PetscMalloc1(numPointsWithDofs,&pointsWithDofs));
    CHKERRQ(PetscSectionSetUp(leafIndicesSec));
    CHKERRQ(PetscSectionGetStorageSize(leafIndicesSec,&numIndices));
    CHKERRQ(PetscMalloc1(gatheredIndices ? numIndices : (maxDof + 1),&leafInds));
    if (gatheredValues)  CHKERRQ(PetscMalloc1(numIndices,&leafVals));
    for (l = 0, offset = 0; l < nleaves; l++) {
      p    = leaves ? leaves[l] : l;
      CHKERRQ(PetscSectionGetDof(globalFine,p,&dof));
      CHKERRQ(PetscSectionGetConstraintDof(globalFine,p,&cdof));
      if ((dof - cdof) > 0) {
        PetscInt    off, gOff;
        PetscInt    *pInd;
        PetscScalar *pVal = NULL;

        pointsWithDofs[offset++] = l;

        CHKERRQ(PetscSectionGetOffset(leafIndicesSec,p,&off));

        pInd = gatheredIndices ? (&leafInds[off + 1]) : leafInds;
        if (gatheredValues) {
          PetscInt i;

          pVal = &leafVals[off + 1];
          for (i = 0; i < dof; i++) pVal[i] = 0.;
        }
        CHKERRQ(PetscSectionGetOffset(globalFine,p,&gOff));

        offsets[0] = 0;
        if (numFields) {
          PetscInt f;

          for (f = 0; f < numFields; f++) {
            PetscInt fDof;
            CHKERRQ(PetscSectionGetFieldDof(localFine,p,f,&fDof));
            offsets[f + 1] = fDof + offsets[f];
          }
          CHKERRQ(DMPlexGetIndicesPointFields_Internal(localFine,PETSC_FALSE,p,gOff < 0 ? -(gOff + 1) : gOff,offsets,PETSC_FALSE,NULL,-1, NULL,pInd));
        } else {
          CHKERRQ(DMPlexGetIndicesPoint_Internal(localFine,PETSC_FALSE,p,gOff < 0 ? -(gOff + 1) : gOff,offsets,PETSC_FALSE,NULL, NULL,pInd));
        }
        if (gatheredValues) CHKERRQ(VecGetValues(fineVec,dof,pInd,pVal));
      }
    }
    CHKERRQ(PetscSFCreateEmbeddedLeafSF(coarseToFine, numPointsWithDofs, pointsWithDofs, &coarseToFineEmbedded));
    CHKERRQ(PetscFree(pointsWithDofs));
  }

  CHKERRQ(DMPlexGetChart(coarse,&pStartC,&pEndC));
  CHKERRQ(DMGetLocalSection(coarse,&localCoarse));
  CHKERRQ(DMGetGlobalSection(coarse,&globalCoarse));

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

    CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)coarse),&rank));
    CHKERRMPI(MPI_Type_contiguous(3,MPIU_INT,&threeInt));
    CHKERRMPI(MPI_Type_commit(&threeInt));
    CHKERRQ(PetscMalloc2(pEndC-pStartC,&parentNodeAndIdCoarse,pEndF-pStartF,&parentNodeAndIdFine));
    CHKERRQ(DMGetPointSF(coarse,&pointSF));
    CHKERRQ(PetscSFGetGraph(pointSF,NULL,&nleaves,&ilocal,&iremote));
    for (p = pStartC; p < pEndC; p++) {
      PetscInt parent, childId;
      CHKERRQ(DMPlexGetTreeParent(coarse,p,&parent,&childId));
      parentNodeAndIdCoarse[p - pStartC][0] = rank;
      parentNodeAndIdCoarse[p - pStartC][1] = parent - pStartC;
      parentNodeAndIdCoarse[p - pStartC][2] = (p == parent) ? -1 : childId;
      if (nleaves > 0) {
        PetscInt leaf = -1;

        if (ilocal) {
          CHKERRQ(PetscFindInt(parent,nleaves,ilocal,&leaf));
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
    CHKERRQ(PetscSFBcastBegin(coarseToFineEmbedded,threeInt,parentNodeAndIdCoarse,parentNodeAndIdFine,MPI_REPLACE));
    CHKERRQ(PetscSFBcastEnd(coarseToFineEmbedded,threeInt,parentNodeAndIdCoarse,parentNodeAndIdFine,MPI_REPLACE));
    for (p = pStartF, nleavesToParents = 0; p < pEndF; p++) {
      PetscInt dof;

      CHKERRQ(PetscSectionGetDof(leafIndicesSec,p,&dof));
      if (dof) {
        PetscInt off;

        CHKERRQ(PetscSectionGetOffset(leafIndicesSec,p,&off));
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
    CHKERRQ(PetscMalloc1(nleavesToParents,&ilocalToParents));
    CHKERRQ(PetscMalloc1(nleavesToParents,&iremoteToParents));
    for (p = pStartF, nleavesToParents = 0; p < pEndF; p++) {
      if (parentNodeAndIdFine[p-pStartF][0] >= 0) {
        ilocalToParents[nleavesToParents] = p - pStartF;
        iremoteToParents[nleavesToParents].rank  = parentNodeAndIdFine[p-pStartF][0];
        iremoteToParents[nleavesToParents].index = parentNodeAndIdFine[p-pStartF][1];
        nleavesToParents++;
      }
    }
    CHKERRQ(PetscSFCreate(PetscObjectComm((PetscObject)coarse),&sfToParents));
    CHKERRQ(PetscSFSetGraph(sfToParents,pEndC-pStartC,nleavesToParents,ilocalToParents,PETSC_OWN_POINTER,iremoteToParents,PETSC_OWN_POINTER));
    CHKERRQ(PetscSFDestroy(&coarseToFineEmbedded));

    coarseToFineEmbedded = sfToParents;

    CHKERRQ(PetscFree2(parentNodeAndIdCoarse,parentNodeAndIdFine));
    CHKERRMPI(MPI_Type_free(&threeInt));
  }

  { /* winnow out coarse points that don't have dofs */
    PetscInt dof, cdof, numPointsWithDofs, offset, *pointsWithDofs;
    PetscSF  sfDofsOnly;

    for (p = pStartC, numPointsWithDofs = 0; p < pEndC; p++) {
      CHKERRQ(PetscSectionGetDof(globalCoarse,p,&dof));
      CHKERRQ(PetscSectionGetConstraintDof(globalCoarse,p,&cdof));
      if ((dof - cdof) > 0) {
        numPointsWithDofs++;
      }
    }
    CHKERRQ(PetscMalloc1(numPointsWithDofs,&pointsWithDofs));
    for (p = pStartC, offset = 0; p < pEndC; p++) {
      CHKERRQ(PetscSectionGetDof(globalCoarse,p,&dof));
      CHKERRQ(PetscSectionGetConstraintDof(globalCoarse,p,&cdof));
      if ((dof - cdof) > 0) {
        pointsWithDofs[offset++] = p - pStartC;
      }
    }
    CHKERRQ(PetscSFCreateEmbeddedRootSF(coarseToFineEmbedded, numPointsWithDofs, pointsWithDofs, &sfDofsOnly));
    CHKERRQ(PetscSFDestroy(&coarseToFineEmbedded));
    CHKERRQ(PetscFree(pointsWithDofs));
    coarseToFineEmbedded = sfDofsOnly;
  }

  /* communicate back to the coarse mesh which coarse points have children (that may require injection) */
  CHKERRQ(PetscSFComputeDegreeBegin(coarseToFineEmbedded,&rootDegrees));
  CHKERRQ(PetscSFComputeDegreeEnd(coarseToFineEmbedded,&rootDegrees));
  CHKERRQ(PetscSectionCreate(PetscObjectComm((PetscObject)coarse),&multiRootSec));
  CHKERRQ(PetscSectionSetChart(multiRootSec,pStartC,pEndC));
  for (p = pStartC; p < pEndC; p++) {
    CHKERRQ(PetscSectionSetDof(multiRootSec,p,rootDegrees[p-pStartC]));
  }
  CHKERRQ(PetscSectionSetUp(multiRootSec));
  CHKERRQ(PetscSectionGetStorageSize(multiRootSec,&numMulti));
  CHKERRQ(PetscSectionCreate(PetscObjectComm((PetscObject)coarse),&rootIndicesSec));
  { /* distribute the leaf section */
    PetscSF multi, multiInv, indicesSF;
    PetscInt *remoteOffsets, numRootIndices;

    CHKERRQ(PetscSFGetMultiSF(coarseToFineEmbedded,&multi));
    CHKERRQ(PetscSFCreateInverseSF(multi,&multiInv));
    CHKERRQ(PetscSFDistributeSection(multiInv,leafIndicesSec,&remoteOffsets,rootIndicesSec));
    CHKERRQ(PetscSFCreateSectionSF(multiInv,leafIndicesSec,remoteOffsets,rootIndicesSec,&indicesSF));
    CHKERRQ(PetscFree(remoteOffsets));
    CHKERRQ(PetscSFDestroy(&multiInv));
    CHKERRQ(PetscSectionGetStorageSize(rootIndicesSec,&numRootIndices));
    if (gatheredIndices) {
      CHKERRQ(PetscMalloc1(numRootIndices,&rootInds));
      CHKERRQ(PetscSFBcastBegin(indicesSF,MPIU_INT,leafInds,rootInds,MPI_REPLACE));
      CHKERRQ(PetscSFBcastEnd(indicesSF,MPIU_INT,leafInds,rootInds,MPI_REPLACE));
    }
    if (gatheredValues) {
      CHKERRQ(PetscMalloc1(numRootIndices,&rootVals));
      CHKERRQ(PetscSFBcastBegin(indicesSF,MPIU_SCALAR,leafVals,rootVals,MPI_REPLACE));
      CHKERRQ(PetscSFBcastEnd(indicesSF,MPIU_SCALAR,leafVals,rootVals,MPI_REPLACE));
    }
    CHKERRQ(PetscSFDestroy(&indicesSF));
  }
  CHKERRQ(PetscSectionDestroy(&leafIndicesSec));
  CHKERRQ(PetscFree(leafInds));
  CHKERRQ(PetscFree(leafVals));
  CHKERRQ(PetscSFDestroy(&coarseToFineEmbedded));
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

  PetscFunctionBegin;

  /* get the templates for the fine-to-coarse injection from the reference tree */
  CHKERRQ(DMPlexGetReferenceTree(coarse,&refTree));
  CHKERRQ(DMGetDefaultConstraints(refTree,&cSecRef,NULL,NULL));
  CHKERRQ(PetscSectionGetChart(cSecRef,&pRefStart,&pRefEnd));
  CHKERRQ(DMPlexReferenceTreeGetInjector(refTree,&injRef));

  CHKERRQ(DMPlexGetChart(fine,&pStartF,&pEndF));
  CHKERRQ(DMGetLocalSection(fine,&localFine));
  CHKERRQ(DMGetGlobalSection(fine,&globalFine));
  CHKERRQ(PetscSectionGetNumFields(localFine,&numFields));
  CHKERRQ(DMPlexGetChart(coarse,&pStartC,&pEndC));
  CHKERRQ(DMGetLocalSection(coarse,&localCoarse));
  CHKERRQ(DMGetGlobalSection(coarse,&globalCoarse));
  CHKERRQ(PetscSectionGetMaxDof(localCoarse,&maxDof));
  {
    PetscInt maxFields = PetscMax(1,numFields) + 1;
    CHKERRQ(PetscMalloc3(maxFields,&offsets,maxFields,&offsetsCopy,maxFields,&rowOffsets));
  }

  CHKERRQ(DMPlexTransferInjectorTree(coarse,fine,coarseToFine,childIds,NULL,numFields,offsets,&multiRootSec,&rootIndicesSec,&rootIndices,NULL));

  CHKERRQ(PetscMalloc1(maxDof,&parentIndices));

  /* count indices */
  CHKERRQ(MatGetLayouts(mat,&rowMap,&colMap));
  CHKERRQ(PetscLayoutSetUp(rowMap));
  CHKERRQ(PetscLayoutSetUp(colMap));
  CHKERRQ(PetscLayoutGetRange(rowMap,&rowStart,&rowEnd));
  CHKERRQ(PetscLayoutGetRange(colMap,&colStart,&colEnd));
  CHKERRQ(PetscCalloc2(rowEnd-rowStart,&nnzD,rowEnd-rowStart,&nnzO));
  for (p = pStartC; p < pEndC; p++) {
    PetscInt numLeaves, leafStart, leafEnd, l, dof, cdof, gOff;

    CHKERRQ(PetscSectionGetDof(globalCoarse,p,&dof));
    CHKERRQ(PetscSectionGetConstraintDof(globalCoarse,p,&cdof));
    if ((dof - cdof) <= 0) continue;
    CHKERRQ(PetscSectionGetOffset(globalCoarse,p,&gOff));

    rowOffsets[0] = 0;
    offsetsCopy[0] = 0;
    if (numFields) {
      PetscInt f;

      for (f = 0; f < numFields; f++) {
        PetscInt fDof;
        CHKERRQ(PetscSectionGetFieldDof(localCoarse,p,f,&fDof));
        rowOffsets[f + 1] = offsetsCopy[f + 1] = fDof + rowOffsets[f];
      }
      CHKERRQ(DMPlexGetIndicesPointFields_Internal(localCoarse,PETSC_FALSE,p,gOff < 0 ? -(gOff + 1) : gOff,offsetsCopy,PETSC_FALSE,NULL,-1, NULL,parentIndices));
    } else {
      CHKERRQ(DMPlexGetIndicesPoint_Internal(localCoarse,PETSC_FALSE,p,gOff < 0 ? -(gOff + 1) : gOff,offsetsCopy,PETSC_FALSE,NULL, NULL,parentIndices));
      rowOffsets[1] = offsetsCopy[0];
    }

    CHKERRQ(PetscSectionGetDof(multiRootSec,p,&numLeaves));
    CHKERRQ(PetscSectionGetOffset(multiRootSec,p,&leafStart));
    leafEnd = leafStart + numLeaves;
    for (l = leafStart; l < leafEnd; l++) {
      PetscInt numIndices, childId, offset;
      const PetscInt *childIndices;

      CHKERRQ(PetscSectionGetDof(rootIndicesSec,l,&numIndices));
      CHKERRQ(PetscSectionGetOffset(rootIndicesSec,l,&offset));
      childId = rootIndices[offset++];
      childIndices = &rootIndices[offset];
      numIndices--;

      if (childId == -1) { /* equivalent points: scatter */
        PetscInt i;

        for (i = 0; i < numIndices; i++) {
          PetscInt colIndex = childIndices[i];
          PetscInt rowIndex = parentIndices[i];
          if (rowIndex < 0) continue;
          PetscCheckFalse(colIndex < 0,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Unconstrained fine and constrained coarse");
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

        CHKERRQ(DMPlexGetTreeParent(refTree,childId,&parentId,NULL));

        lim = PetscMax(1,numFields);
        offsets[0] = 0;
        if (numFields) {
          PetscInt f;

          for (f = 0; f < numFields; f++) {
            PetscInt fDof;
            CHKERRQ(PetscSectionGetFieldDof(cSecRef,childId,f,&fDof));

            offsets[f + 1] = fDof + offsets[f];
          }
        }
        else {
          PetscInt cDof;

          CHKERRQ(PetscSectionGetDof(cSecRef,childId,&cDof));
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
  CHKERRQ(MatXAIJSetPreallocation(mat,1,nnzD,nnzO,NULL,NULL));
  CHKERRQ(PetscFree2(nnzD,nnzO));
  /* insert values */
  CHKERRQ(DMPlexReferenceTreeGetChildrenMatrices_Injection(refTree,injRef,&childrenMats));
  for (p = pStartC; p < pEndC; p++) {
    PetscInt numLeaves, leafStart, leafEnd, l, dof, cdof, gOff;

    CHKERRQ(PetscSectionGetDof(globalCoarse,p,&dof));
    CHKERRQ(PetscSectionGetConstraintDof(globalCoarse,p,&cdof));
    if ((dof - cdof) <= 0) continue;
    CHKERRQ(PetscSectionGetOffset(globalCoarse,p,&gOff));

    rowOffsets[0] = 0;
    offsetsCopy[0] = 0;
    if (numFields) {
      PetscInt f;

      for (f = 0; f < numFields; f++) {
        PetscInt fDof;
        CHKERRQ(PetscSectionGetFieldDof(localCoarse,p,f,&fDof));
        rowOffsets[f + 1] = offsetsCopy[f + 1] = fDof + rowOffsets[f];
      }
      CHKERRQ(DMPlexGetIndicesPointFields_Internal(localCoarse,PETSC_FALSE,p,gOff < 0 ? -(gOff + 1) : gOff,offsetsCopy,PETSC_FALSE,NULL,-1, NULL,parentIndices));
    } else {
      CHKERRQ(DMPlexGetIndicesPoint_Internal(localCoarse,PETSC_FALSE,p,gOff < 0 ? -(gOff + 1) : gOff,offsetsCopy,PETSC_FALSE,NULL, NULL,parentIndices));
      rowOffsets[1] = offsetsCopy[0];
    }

    CHKERRQ(PetscSectionGetDof(multiRootSec,p,&numLeaves));
    CHKERRQ(PetscSectionGetOffset(multiRootSec,p,&leafStart));
    leafEnd = leafStart + numLeaves;
    for (l = leafStart; l < leafEnd; l++) {
      PetscInt numIndices, childId, offset;
      const PetscInt *childIndices;

      CHKERRQ(PetscSectionGetDof(rootIndicesSec,l,&numIndices));
      CHKERRQ(PetscSectionGetOffset(rootIndicesSec,l,&offset));
      childId = rootIndices[offset++];
      childIndices = &rootIndices[offset];
      numIndices--;

      if (childId == -1) { /* equivalent points: scatter */
        PetscInt i;

        for (i = 0; i < numIndices; i++) {
          CHKERRQ(MatSetValue(mat,parentIndices[i],childIndices[i],1.,INSERT_VALUES));
        }
      }
      else {
        PetscInt parentId, f, lim;

        CHKERRQ(DMPlexGetTreeParent(refTree,childId,&parentId,NULL));

        lim = PetscMax(1,numFields);
        offsets[0] = 0;
        if (numFields) {
          PetscInt f;

          for (f = 0; f < numFields; f++) {
            PetscInt fDof;
            CHKERRQ(PetscSectionGetFieldDof(cSecRef,childId,f,&fDof));

            offsets[f + 1] = fDof + offsets[f];
          }
        }
        else {
          PetscInt cDof;

          CHKERRQ(PetscSectionGetDof(cSecRef,childId,&cDof));
          offsets[1] = cDof;
        }
        for (f = 0; f < lim; f++) {
          PetscScalar    *childMat   = &childrenMats[childId - pRefStart][f][0];
          PetscInt       *rowIndices = &parentIndices[rowOffsets[f]];
          const PetscInt *colIndices = &childIndices[offsets[f]];

          CHKERRQ(MatSetValues(mat,rowOffsets[f+1]-rowOffsets[f],rowIndices,offsets[f+1]-offsets[f],colIndices,childMat,INSERT_VALUES));
        }
      }
    }
  }
  CHKERRQ(PetscSectionDestroy(&multiRootSec));
  CHKERRQ(PetscSectionDestroy(&rootIndicesSec));
  CHKERRQ(PetscFree(parentIndices));
  CHKERRQ(DMPlexReferenceTreeRestoreChildrenMatrices_Injection(refTree,injRef,&childrenMats));
  CHKERRQ(PetscFree(rootIndices));
  CHKERRQ(PetscFree3(offsets,offsetsCopy,rowOffsets));

  CHKERRQ(MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY));
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
  PetscInt          pStartC, pEndC, pStartF, pEndF, p, cellStart, cellEnd;
  PetscInt          aStart, aEnd, cStart, cEnd;
  PetscInt          *maxChildIds;
  PetscInt          *offsets, *newOffsets, *offsetsCopy, *newOffsetsCopy, *rowOffsets, *numD, *numO;
  PetscFV           fv = NULL;
  PetscInt          dim, numFVcomps = -1, fvField = -1;
  DM                cellDM = NULL, gradDM = NULL;
  const PetscScalar *cellGeomArray = NULL;
  const PetscScalar *gradArray = NULL;

  PetscFunctionBegin;
  CHKERRQ(VecSetOption(vecFine,VEC_IGNORE_NEGATIVE_INDICES,PETSC_TRUE));
  CHKERRQ(DMPlexGetChart(coarse,&pStartC,&pEndC));
  CHKERRQ(DMPlexGetSimplexOrBoxCells(coarse,0,&cellStart,&cellEnd));
  CHKERRQ(DMPlexGetChart(fine,&pStartF,&pEndF));
  CHKERRQ(DMGetGlobalSection(fine,&globalFine));
  CHKERRQ(DMGetCoordinateDim(coarse,&dim));
  { /* winnow fine points that don't have global dofs out of the sf */
    PetscInt       nleaves, l;
    const PetscInt *leaves;
    PetscInt       dof, cdof, numPointsWithDofs, offset, *pointsWithDofs;

    CHKERRQ(PetscSFGetGraph(coarseToFine,NULL,&nleaves,&leaves,NULL));

    for (l = 0, numPointsWithDofs = 0; l < nleaves; l++) {
      PetscInt p = leaves ? leaves[l] : l;

      CHKERRQ(PetscSectionGetDof(globalFine,p,&dof));
      CHKERRQ(PetscSectionGetConstraintDof(globalFine,p,&cdof));
      if ((dof - cdof) > 0) {
        numPointsWithDofs++;
      }
    }
    CHKERRQ(PetscMalloc1(numPointsWithDofs,&pointsWithDofs));
    for (l = 0, offset = 0; l < nleaves; l++) {
      PetscInt p = leaves ? leaves[l] : l;

      CHKERRQ(PetscSectionGetDof(globalFine,p,&dof));
      CHKERRQ(PetscSectionGetConstraintDof(globalFine,p,&cdof));
      if ((dof - cdof) > 0) {
        pointsWithDofs[offset++] = l;
      }
    }
    CHKERRQ(PetscSFCreateEmbeddedLeafSF(coarseToFine, numPointsWithDofs, pointsWithDofs, &coarseToFineEmbedded));
    CHKERRQ(PetscFree(pointsWithDofs));
  }
  /* communicate back to the coarse mesh which coarse points have children (that may require interpolation) */
  CHKERRQ(PetscMalloc1(pEndC-pStartC,&maxChildIds));
  for (p = pStartC; p < pEndC; p++) {
    maxChildIds[p - pStartC] = -2;
  }
  CHKERRQ(PetscSFReduceBegin(coarseToFineEmbedded,MPIU_INT,cids,maxChildIds,MPIU_MAX));
  CHKERRQ(PetscSFReduceEnd(coarseToFineEmbedded,MPIU_INT,cids,maxChildIds,MPIU_MAX));

  CHKERRQ(DMGetLocalSection(coarse,&localCoarse));
  CHKERRQ(DMGetGlobalSection(coarse,&globalCoarse));

  CHKERRQ(DMPlexGetAnchors(coarse,&aSec,&aIS));
  CHKERRQ(ISGetIndices(aIS,&anchors));
  CHKERRQ(PetscSectionGetChart(aSec,&aStart,&aEnd));

  CHKERRQ(DMGetDefaultConstraints(coarse,&cSec,&cMat,NULL));
  CHKERRQ(PetscSectionGetChart(cSec,&cStart,&cEnd));

  /* create sections that will send to children the indices and matrices they will need to construct the interpolator */
  CHKERRQ(PetscSectionCreate(PetscObjectComm((PetscObject)coarse),&rootValuesSec));
  CHKERRQ(PetscSectionSetChart(rootValuesSec,pStartC,pEndC));
  CHKERRQ(PetscSectionGetNumFields(localCoarse,&numFields));
  {
    PetscInt maxFields = PetscMax(1,numFields) + 1;
    CHKERRQ(PetscMalloc7(maxFields,&offsets,maxFields,&offsetsCopy,maxFields,&newOffsets,maxFields,&newOffsetsCopy,maxFields,&rowOffsets,maxFields,&numD,maxFields,&numO));
  }
  if (grad) {
    PetscInt i;

    CHKERRQ(VecGetDM(cellGeom,&cellDM));
    CHKERRQ(VecGetArrayRead(cellGeom,&cellGeomArray));
    CHKERRQ(VecGetDM(grad,&gradDM));
    CHKERRQ(VecGetArrayRead(grad,&gradArray));
    for (i = 0; i < PetscMax(1,numFields); i++) {
      PetscObject  obj;
      PetscClassId id;

      CHKERRQ(DMGetField(coarse, i, NULL, &obj));
      CHKERRQ(PetscObjectGetClassId(obj,&id));
      if (id == PETSCFV_CLASSID) {
        fv      = (PetscFV) obj;
        CHKERRQ(PetscFVGetNumComponents(fv,&numFVcomps));
        fvField = i;
        break;
      }
    }
  }

  for (p = pStartC; p < pEndC; p++) { /* count the sizes of the indices and matrices */
    PetscInt dof;
    PetscInt maxChildId     = maxChildIds[p - pStartC];
    PetscInt numValues      = 0;

    CHKERRQ(PetscSectionGetDof(globalCoarse,p,&dof));
    if (dof < 0) {
      dof = -(dof + 1);
    }
    offsets[0]    = 0;
    newOffsets[0] = 0;
    if (maxChildId >= 0) { /* this point has children (with dofs) that will need to be interpolated from the closure of p */
      PetscInt *closure = NULL, closureSize, cl;

      CHKERRQ(DMPlexGetTransitiveClosure(coarse,p,PETSC_TRUE,&closureSize,&closure));
      for (cl = 0; cl < closureSize; cl++) { /* get the closure */
        PetscInt c = closure[2 * cl], clDof;

        CHKERRQ(PetscSectionGetDof(localCoarse,c,&clDof));
        numValues += clDof;
      }
      CHKERRQ(DMPlexRestoreTransitiveClosure(coarse,p,PETSC_TRUE,&closureSize,&closure));
    }
    else if (maxChildId == -1) {
      CHKERRQ(PetscSectionGetDof(localCoarse,p,&numValues));
    }
    /* we will pack the column indices with the field offsets */
    if (maxChildId >= 0 && grad && p >= cellStart && p < cellEnd) {
      /* also send the centroid, and the gradient */
      numValues += dim * (1 + numFVcomps);
    }
    CHKERRQ(PetscSectionSetDof(rootValuesSec,p,numValues));
  }
  CHKERRQ(PetscSectionSetUp(rootValuesSec));
  {
    PetscInt          numRootValues;
    const PetscScalar *coarseArray;

    CHKERRQ(PetscSectionGetStorageSize(rootValuesSec,&numRootValues));
    CHKERRQ(PetscMalloc1(numRootValues,&rootValues));
    CHKERRQ(VecGetArrayRead(vecCoarseLocal,&coarseArray));
    for (p = pStartC; p < pEndC; p++) {
      PetscInt    numValues;
      PetscInt    pValOff;
      PetscScalar *pVal;
      PetscInt    maxChildId = maxChildIds[p - pStartC];

      CHKERRQ(PetscSectionGetDof(rootValuesSec,p,&numValues));
      if (!numValues) {
        continue;
      }
      CHKERRQ(PetscSectionGetOffset(rootValuesSec,p,&pValOff));
      pVal = &(rootValues[pValOff]);
      if (maxChildId >= 0) { /* build an identity matrix, apply matrix constraints on the right */
        PetscInt closureSize = numValues;
        CHKERRQ(DMPlexVecGetClosure(coarse,NULL,vecCoarseLocal,p,&closureSize,&pVal));
        if (grad && p >= cellStart && p < cellEnd) {
          PetscFVCellGeom *cg;
          PetscScalar     *gradVals = NULL;
          PetscInt        i;

          pVal += (numValues - dim * (1 + numFVcomps));

          CHKERRQ(DMPlexPointLocalRead(cellDM,p,cellGeomArray,(void *) &cg));
          for (i = 0; i < dim; i++) pVal[i] = cg->centroid[i];
          pVal += dim;
          CHKERRQ(DMPlexPointGlobalRead(gradDM,p,gradArray,(void *) &gradVals));
          for (i = 0; i < dim * numFVcomps; i++) pVal[i] = gradVals[i];
        }
      }
      else if (maxChildId == -1) {
        PetscInt lDof, lOff, i;

        CHKERRQ(PetscSectionGetDof(localCoarse,p,&lDof));
        CHKERRQ(PetscSectionGetOffset(localCoarse,p,&lOff));
        for (i = 0; i < lDof; i++) pVal[i] = coarseArray[lOff + i];
      }
    }
    CHKERRQ(VecRestoreArrayRead(vecCoarseLocal,&coarseArray));
    CHKERRQ(PetscFree(maxChildIds));
  }
  {
    PetscSF  valuesSF;
    PetscInt *remoteOffsetsValues, numLeafValues;

    CHKERRQ(PetscSectionCreate(PetscObjectComm((PetscObject)fine),&leafValuesSec));
    CHKERRQ(PetscSFDistributeSection(coarseToFineEmbedded,rootValuesSec,&remoteOffsetsValues,leafValuesSec));
    CHKERRQ(PetscSFCreateSectionSF(coarseToFineEmbedded,rootValuesSec,remoteOffsetsValues,leafValuesSec,&valuesSF));
    CHKERRQ(PetscSFDestroy(&coarseToFineEmbedded));
    CHKERRQ(PetscFree(remoteOffsetsValues));
    CHKERRQ(PetscSectionGetStorageSize(leafValuesSec,&numLeafValues));
    CHKERRQ(PetscMalloc1(numLeafValues,&leafValues));
    CHKERRQ(PetscSFBcastBegin(valuesSF,MPIU_SCALAR,rootValues,leafValues,MPI_REPLACE));
    CHKERRQ(PetscSFBcastEnd(valuesSF,MPIU_SCALAR,rootValues,leafValues,MPI_REPLACE));
    CHKERRQ(PetscSFDestroy(&valuesSF));
    CHKERRQ(PetscFree(rootValues));
    CHKERRQ(PetscSectionDestroy(&rootValuesSec));
  }
  CHKERRQ(DMGetLocalSection(fine,&localFine));
  {
    PetscInt    maxDof;
    PetscInt    *rowIndices;
    DM           refTree;
    PetscInt     **refPointFieldN;
    PetscScalar  ***refPointFieldMats;
    PetscSection refConSec, refAnSec;
    PetscInt     pRefStart,pRefEnd,leafStart,leafEnd;
    PetscScalar  *pointWork;

    CHKERRQ(PetscSectionGetMaxDof(localFine,&maxDof));
    CHKERRQ(DMGetWorkArray(fine,maxDof,MPIU_INT,&rowIndices));
    CHKERRQ(DMGetWorkArray(fine,maxDof,MPIU_SCALAR,&pointWork));
    CHKERRQ(DMPlexGetReferenceTree(fine,&refTree));
    CHKERRQ(DMCopyDisc(fine,refTree));
    CHKERRQ(DMPlexReferenceTreeGetChildrenMatrices(refTree,&refPointFieldMats,&refPointFieldN));
    CHKERRQ(DMGetDefaultConstraints(refTree,&refConSec,NULL,NULL));
    CHKERRQ(DMPlexGetAnchors(refTree,&refAnSec,NULL));
    CHKERRQ(PetscSectionGetChart(refConSec,&pRefStart,&pRefEnd));
    CHKERRQ(PetscSectionGetChart(leafValuesSec,&leafStart,&leafEnd));
    CHKERRQ(DMPlexGetSimplexOrBoxCells(fine,0,&cellStart,&cellEnd));
    for (p = leafStart; p < leafEnd; p++) {
      PetscInt          gDof, gcDof, gOff, lDof;
      PetscInt          numValues, pValOff;
      PetscInt          childId;
      const PetscScalar *pVal;
      const PetscScalar *fvGradData = NULL;

      CHKERRQ(PetscSectionGetDof(globalFine,p,&gDof));
      CHKERRQ(PetscSectionGetDof(localFine,p,&lDof));
      CHKERRQ(PetscSectionGetConstraintDof(globalFine,p,&gcDof));
      if ((gDof - gcDof) <= 0) {
        continue;
      }
      CHKERRQ(PetscSectionGetOffset(globalFine,p,&gOff));
      CHKERRQ(PetscSectionGetDof(leafValuesSec,p,&numValues));
      if (!numValues) continue;
      CHKERRQ(PetscSectionGetOffset(leafValuesSec,p,&pValOff));
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

          CHKERRQ(PetscSectionGetFieldDof(localFine,p,f,&rowDof));
          offsets[f + 1]        = offsets[f] + rowDof;
          offsetsCopy[f + 1]    = offsets[f + 1];
          /* TODO: closure indices */
          newOffsets[f + 1]     = newOffsets[f] + ((childId == -1) ? rowDof : refPointFieldN[childId - pRefStart][f]);
        }
        CHKERRQ(DMPlexGetIndicesPointFields_Internal(localFine,PETSC_FALSE,p,gOff,offsetsCopy,PETSC_FALSE,NULL,-1,NULL,rowIndices));
      }
      else {
        offsets[0]    = 0;
        offsets[1]    = lDof;
        newOffsets[0] = 0;
        newOffsets[1] = (childId == -1) ? lDof : refPointFieldN[childId - pRefStart][0];
        CHKERRQ(DMPlexGetIndicesPoint_Internal(localFine,PETSC_FALSE,p,gOff,offsetsCopy,PETSC_FALSE,NULL,NULL,rowIndices));
      }
      if (childId == -1) { /* no child interpolation: one nnz per */
        CHKERRQ(VecSetValues(vecFine,numValues,rowIndices,pVal,INSERT_VALUES));
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
          CHKERRQ(PetscInfo(coarse,"childId %D, numRows %D, numCols %D, refPointFieldN %D maxDof %D\n",childId,numRows,numCols,refPointFieldN[childId - pRefStart][f], maxDof));
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

            CHKERRQ(DMPlexComputeCellGeometryFVM(fine,p,NULL,centroid,NULL));
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
          CHKERRQ(VecSetValues(vecFine,numRows,&rowIndices[offsets[f]],rVal,INSERT_VALUES));
        }
      }
    }
    CHKERRQ(DMPlexReferenceTreeRestoreChildrenMatrices(refTree,&refPointFieldMats,&refPointFieldN));
    CHKERRQ(DMRestoreWorkArray(fine,maxDof,MPIU_SCALAR,&pointWork));
    CHKERRQ(DMRestoreWorkArray(fine,maxDof,MPIU_INT,&rowIndices));
  }
  CHKERRQ(PetscFree(leafValues));
  CHKERRQ(PetscSectionDestroy(&leafValuesSec));
  CHKERRQ(PetscFree7(offsets,offsetsCopy,newOffsets,newOffsetsCopy,rowOffsets,numD,numO));
  CHKERRQ(ISRestoreIndices(aIS,&anchors));
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

  PetscFunctionBegin;

  /* get the templates for the fine-to-coarse injection from the reference tree */
  CHKERRQ(VecSetOption(vecFine,VEC_IGNORE_NEGATIVE_INDICES,PETSC_TRUE));
  CHKERRQ(VecSetOption(vecCoarse,VEC_IGNORE_NEGATIVE_INDICES,PETSC_TRUE));
  CHKERRQ(DMPlexGetReferenceTree(coarse,&refTree));
  CHKERRQ(DMCopyDisc(coarse,refTree));
  CHKERRQ(DMGetDefaultConstraints(refTree,&cSecRef,NULL,NULL));
  CHKERRQ(PetscSectionGetChart(cSecRef,&pRefStart,&pRefEnd));
  CHKERRQ(DMPlexReferenceTreeGetInjector(refTree,&injRef));

  CHKERRQ(DMPlexGetChart(fine,&pStartF,&pEndF));
  CHKERRQ(DMGetLocalSection(fine,&localFine));
  CHKERRQ(DMGetGlobalSection(fine,&globalFine));
  CHKERRQ(PetscSectionGetNumFields(localFine,&numFields));
  CHKERRQ(DMPlexGetChart(coarse,&pStartC,&pEndC));
  CHKERRQ(DMGetLocalSection(coarse,&localCoarse));
  CHKERRQ(DMGetGlobalSection(coarse,&globalCoarse));
  CHKERRQ(PetscSectionGetMaxDof(localCoarse,&maxDof));
  {
    PetscInt maxFields = PetscMax(1,numFields) + 1;
    CHKERRQ(PetscMalloc3(maxFields,&offsets,maxFields,&offsetsCopy,maxFields,&rowOffsets));
  }

  CHKERRQ(DMPlexTransferInjectorTree(coarse,fine,coarseToFine,cids,vecFine,numFields,offsets,&multiRootSec,&rootIndicesSec,NULL,&rootValues));

  CHKERRQ(PetscMalloc2(maxDof,&parentIndices,maxDof,&parentValues));

  /* count indices */
  CHKERRQ(VecGetLayout(vecFine,&colMap));
  CHKERRQ(VecGetLayout(vecCoarse,&rowMap));
  CHKERRQ(PetscLayoutSetUp(rowMap));
  CHKERRQ(PetscLayoutSetUp(colMap));
  CHKERRQ(PetscLayoutGetRange(rowMap,&rowStart,&rowEnd));
  CHKERRQ(PetscLayoutGetRange(colMap,&colStart,&colEnd));
  /* insert values */
  CHKERRQ(DMPlexReferenceTreeGetChildrenMatrices_Injection(refTree,injRef,&childrenMats));
  for (p = pStartC; p < pEndC; p++) {
    PetscInt  numLeaves, leafStart, leafEnd, l, dof, cdof, gOff;
    PetscBool contribute = PETSC_FALSE;

    CHKERRQ(PetscSectionGetDof(globalCoarse,p,&dof));
    CHKERRQ(PetscSectionGetConstraintDof(globalCoarse,p,&cdof));
    if ((dof - cdof) <= 0) continue;
    CHKERRQ(PetscSectionGetDof(localCoarse,p,&dof));
    CHKERRQ(PetscSectionGetOffset(globalCoarse,p,&gOff));

    rowOffsets[0] = 0;
    offsetsCopy[0] = 0;
    if (numFields) {
      PetscInt f;

      for (f = 0; f < numFields; f++) {
        PetscInt fDof;
        CHKERRQ(PetscSectionGetFieldDof(localCoarse,p,f,&fDof));
        rowOffsets[f + 1] = offsetsCopy[f + 1] = fDof + rowOffsets[f];
      }
      CHKERRQ(DMPlexGetIndicesPointFields_Internal(localCoarse,PETSC_FALSE,p,gOff < 0 ? -(gOff + 1) : gOff,offsetsCopy,PETSC_FALSE,NULL,-1,NULL,parentIndices));
    } else {
      CHKERRQ(DMPlexGetIndicesPoint_Internal(localCoarse,PETSC_FALSE,p,gOff < 0 ? -(gOff + 1) : gOff,offsetsCopy,PETSC_FALSE,NULL,NULL,parentIndices));
      rowOffsets[1] = offsetsCopy[0];
    }

    CHKERRQ(PetscSectionGetDof(multiRootSec,p,&numLeaves));
    CHKERRQ(PetscSectionGetOffset(multiRootSec,p,&leafStart));
    leafEnd = leafStart + numLeaves;
    for (l = 0; l < dof; l++) parentValues[l] = 0.;
    for (l = leafStart; l < leafEnd; l++) {
      PetscInt numIndices, childId, offset;
      const PetscScalar *childValues;

      CHKERRQ(PetscSectionGetDof(rootIndicesSec,l,&numIndices));
      CHKERRQ(PetscSectionGetOffset(rootIndicesSec,l,&offset));
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
        CHKERRQ(DMPlexGetTreeParent(refTree,childId,&parentId,NULL));

        lim = PetscMax(1,numFields);
        offsets[0] = 0;
        if (numFields) {
          PetscInt f;

          for (f = 0; f < numFields; f++) {
            PetscInt fDof;
            CHKERRQ(PetscSectionGetFieldDof(cSecRef,childId,f,&fDof));

            offsets[f + 1] = fDof + offsets[f];
          }
        }
        else {
          PetscInt cDof;

          CHKERRQ(PetscSectionGetDof(cSecRef,childId,&cDof));
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
    if (contribute) CHKERRQ(VecSetValues(vecCoarse,dof,parentIndices,parentValues,INSERT_VALUES));
  }
  CHKERRQ(PetscSectionDestroy(&multiRootSec));
  CHKERRQ(PetscSectionDestroy(&rootIndicesSec));
  CHKERRQ(PetscFree2(parentIndices,parentValues));
  CHKERRQ(DMPlexReferenceTreeRestoreChildrenMatrices_Injection(refTree,injRef,&childrenMats));
  CHKERRQ(PetscFree(rootValues));
  CHKERRQ(PetscFree3(offsets,offsetsCopy,rowOffsets));
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
  PetscFunctionBegin;
  CHKERRQ(VecSet(vecOut,0.0));
  if (sfRefine) {
    Vec vecInLocal;
    DM  dmGrad = NULL;
    Vec faceGeom = NULL, cellGeom = NULL, grad = NULL;

    CHKERRQ(DMGetLocalVector(dmIn,&vecInLocal));
    CHKERRQ(VecSet(vecInLocal,0.0));
    {
      PetscInt  numFields, i;

      CHKERRQ(DMGetNumFields(dmIn, &numFields));
      for (i = 0; i < numFields; i++) {
        PetscObject  obj;
        PetscClassId classid;

        CHKERRQ(DMGetField(dmIn, i, NULL, &obj));
        CHKERRQ(PetscObjectGetClassId(obj, &classid));
        if (classid == PETSCFV_CLASSID) {
          CHKERRQ(DMPlexGetDataFVM(dmIn,(PetscFV)obj,&cellGeom,&faceGeom,&dmGrad));
          break;
        }
      }
    }
    if (useBCs) {
      CHKERRQ(DMPlexInsertBoundaryValues(dmIn,PETSC_TRUE,vecInLocal,time,faceGeom,cellGeom,NULL));
    }
    CHKERRQ(DMGlobalToLocalBegin(dmIn,vecIn,INSERT_VALUES,vecInLocal));
    CHKERRQ(DMGlobalToLocalEnd(dmIn,vecIn,INSERT_VALUES,vecInLocal));
    if (dmGrad) {
      CHKERRQ(DMGetGlobalVector(dmGrad,&grad));
      CHKERRQ(DMPlexReconstructGradientsFVM(dmIn,vecInLocal,grad));
    }
    CHKERRQ(DMPlexTransferVecTree_Interpolate(dmIn,vecInLocal,dmOut,vecOut,sfRefine,cidsRefine,grad,cellGeom));
    CHKERRQ(DMRestoreLocalVector(dmIn,&vecInLocal));
    if (dmGrad) {
      CHKERRQ(DMRestoreGlobalVector(dmGrad,&grad));
    }
  }
  if (sfCoarsen) {
    CHKERRQ(DMPlexTransferVecTree_Inject(dmIn,vecIn,dmOut,vecOut,sfCoarsen,cidsCoarsen));
  }
  CHKERRQ(VecAssemblyBegin(vecOut));
  CHKERRQ(VecAssemblyEnd(vecOut));
  PetscFunctionReturn(0);
}
