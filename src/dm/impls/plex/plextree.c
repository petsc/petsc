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

.seealso: `DMPlexGetReferenceTree()`, `DMPlexCreateDefaultReferenceTree()`
@*/
PetscErrorCode DMPlexSetReferenceTree(DM dm, DM ref)
{
  DM_Plex        *mesh = (DM_Plex *)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (ref) {PetscValidHeaderSpecific(ref, DM_CLASSID, 2);}
  PetscCall(PetscObjectReference((PetscObject)ref));
  PetscCall(DMDestroy(&mesh->referenceTree));
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

.seealso: `DMPlexSetReferenceTree()`, `DMPlexCreateDefaultReferenceTree()`
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
    PetscCall(DMPlexGetDepthStratum(dm,dim,&dStart,&dEnd));
    if (parent >= dStart && parent <= dEnd) {
      break;
    }
  }
  PetscCheck(dim <= 2,PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot perform child symmetry for %" PetscInt_FMT "-cells",dim);
  PetscCheck(dim,PETSC_COMM_SELF,PETSC_ERR_PLIB,"A vertex has no children");
  if (childA < dStart || childA >= dEnd) {
    /* this is a lower-dimensional child: bootstrap */
    PetscInt size, i, sA = -1, sB, sOrientB, sConeSize;
    const PetscInt *supp, *coneA, *coneB, *oA, *oB;

    PetscCall(DMPlexGetSupportSize(dm,childA,&size));
    PetscCall(DMPlexGetSupport(dm,childA,&supp));

    /* find a point sA in supp(childA) that has the same parent */
    for (i = 0; i < size; i++) {
      PetscInt sParent;

      sA   = supp[i];
      if (sA == parent) continue;
      PetscCall(DMPlexGetTreeParent(dm,sA,&sParent,NULL));
      if (sParent == parent) {
        break;
      }
    }
    PetscCheck(i != size,PETSC_COMM_SELF,PETSC_ERR_PLIB,"could not find support in children");
    /* find out which point sB is in an equivalent position to sA under
     * parentOrientB */
    PetscCall(DMPlexReferenceTreeGetChildSymmetry_Default(dm,parent,parentOrientA,0,sA,parentOrientB,&sOrientB,&sB));
    PetscCall(DMPlexGetConeSize(dm,sA,&sConeSize));
    PetscCall(DMPlexGetCone(dm,sA,&coneA));
    PetscCall(DMPlexGetCone(dm,sB,&coneB));
    PetscCall(DMPlexGetConeOrientation(dm,sA,&oA));
    PetscCall(DMPlexGetConeOrientation(dm,sB,&oB));
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

          PetscCall(DMPlexGetConeSize(dm,childA,&coneSize));
          /* compose sOrientB and oB[j] */
          PetscCheck(coneSize == 0 || coneSize == 2,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Expected a vertex or an edge");
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
    PetscCheck(i != sConeSize,PETSC_COMM_SELF,PETSC_ERR_PLIB,"support cone mismatch");
    PetscFunctionReturn(0);
  }
  /* get the cone size and symmetry swap */
  PetscCall(DMPlexGetConeSize(dm,parent,&coneSize));
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
    PetscCall(DMPlexGetTreeChildren(dm,parent,&numChildren,&children));
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
      PetscCheck(dim == 2 && posA == 3,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Expected a middle triangle, got something else");
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

.seealso: `DMPlexGetReferenceTree()`, `DMPlexSetReferenceTree()`, `DMPlexSetTree()`
@*/
PetscErrorCode DMPlexReferenceTreeGetChildSymmetry(DM dm, PetscInt parent, PetscInt parentOrientA, PetscInt childOrientA, PetscInt childA, PetscInt parentOrientB, PetscInt *childOrientB, PetscInt *childB)
{
  DM_Plex        *mesh = (DM_Plex *)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCheck(mesh->getchildsymmetry,PETSC_COMM_SELF,PETSC_ERR_SUP,"DMPlexReferenceTreeGetChildSymmetry not implemented");
  PetscCall(mesh->getchildsymmetry(dm,parent,parentOrientA,childOrientA,childA,parentOrientB,childOrientB,childB));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexSetTree_Internal(DM,PetscSection,PetscInt*,PetscInt*,PetscBool,PetscBool);

PetscErrorCode DMPlexCreateReferenceTree_SetTree(DM dm, PetscSection parentSection, PetscInt parents[], PetscInt childIDs[])
{
  PetscFunctionBegin;
  PetscCall(DMPlexSetTree_Internal(dm,parentSection,parents,childIDs,PETSC_TRUE,PETSC_FALSE));
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
  PetscCall(DMGetDimension(K, &dim));
  PetscCall(DMPlexGetChart(K, &pStart, &pEnd));
  PetscCall(DMGetLabel(K, labelName, &identity));
  PetscCall(DMGetLabel(Kref, labelName, &identityRef));
  PetscCall(DMPlexGetChart(Kref, &pRefStart, &pRefEnd));
  PetscCall(PetscSectionCreate(comm, &unionSection));
  PetscCall(PetscSectionSetChart(unionSection, 0, (pEnd - pStart) + (pRefEnd - pRefStart)));
  /* count points that will go in the union */
  for (p = pStart; p < pEnd; p++) {
    PetscCall(PetscSectionSetDof(unionSection, p - pStart, 1));
  }
  for (p = pRefStart; p < pRefEnd; p++) {
    PetscInt q, qSize;
    PetscCall(DMLabelGetValue(identityRef, p, &q));
    PetscCall(DMLabelGetStratumSize(identityRef, q, &qSize));
    if (qSize > 1) {
      PetscCall(PetscSectionSetDof(unionSection, p - pRefStart + (pEnd - pStart), 1));
    }
  }
  PetscCall(PetscMalloc1(pEnd - pStart + pRefEnd - pRefStart,&permvals));
  offset = 0;
  /* stratify points in the union by topological dimension */
  for (d = 0; d <= dim; d++) {
    PetscInt cStart, cEnd, c;

    PetscCall(DMPlexGetHeightStratum(K, d, &cStart, &cEnd));
    for (c = cStart; c < cEnd; c++) {
      permvals[offset++] = c;
    }

    PetscCall(DMPlexGetHeightStratum(Kref, d, &cStart, &cEnd));
    for (c = cStart; c < cEnd; c++) {
      permvals[offset++] = c + (pEnd - pStart);
    }
  }
  PetscCall(ISCreateGeneral(comm, (pEnd - pStart) + (pRefEnd - pRefStart), permvals, PETSC_OWN_POINTER, &perm));
  PetscCall(PetscSectionSetPermutation(unionSection,perm));
  PetscCall(PetscSectionSetUp(unionSection));
  PetscCall(PetscSectionGetStorageSize(unionSection,&numUnionPoints));
  PetscCall(PetscMalloc2(numUnionPoints,&coneSizes,dim+1,&numDimPoints));
  /* count dimension points */
  for (d = 0; d <= dim; d++) {
    PetscInt cStart, cOff, cOff2;
    PetscCall(DMPlexGetHeightStratum(K,d,&cStart,NULL));
    PetscCall(PetscSectionGetOffset(unionSection,cStart-pStart,&cOff));
    if (d < dim) {
      PetscCall(DMPlexGetHeightStratum(K,d+1,&cStart,NULL));
      PetscCall(PetscSectionGetOffset(unionSection,cStart-pStart,&cOff2));
    }
    else {
      cOff2 = numUnionPoints;
    }
    numDimPoints[dim - d] = cOff2 - cOff;
  }
  PetscCall(PetscSectionCreate(comm, &unionConeSection));
  PetscCall(PetscSectionSetChart(unionConeSection, 0, numUnionPoints));
  /* count the cones in the union */
  for (p = pStart; p < pEnd; p++) {
    PetscInt dof, uOff;

    PetscCall(DMPlexGetConeSize(K, p, &dof));
    PetscCall(PetscSectionGetOffset(unionSection, p - pStart,&uOff));
    PetscCall(PetscSectionSetDof(unionConeSection, uOff, dof));
    coneSizes[uOff] = dof;
  }
  for (p = pRefStart; p < pRefEnd; p++) {
    PetscInt dof, uDof, uOff;

    PetscCall(DMPlexGetConeSize(Kref, p, &dof));
    PetscCall(PetscSectionGetDof(unionSection, p - pRefStart + (pEnd - pStart),&uDof));
    PetscCall(PetscSectionGetOffset(unionSection, p - pRefStart + (pEnd - pStart),&uOff));
    if (uDof) {
      PetscCall(PetscSectionSetDof(unionConeSection, uOff, dof));
      coneSizes[uOff] = dof;
    }
  }
  PetscCall(PetscSectionSetUp(unionConeSection));
  PetscCall(PetscSectionGetStorageSize(unionConeSection,&numCones));
  PetscCall(PetscMalloc2(numCones,&unionCones,numCones,&unionOrientations));
  /* write the cones in the union */
  for (p = pStart; p < pEnd; p++) {
    PetscInt dof, uOff, c, cOff;
    const PetscInt *cone, *orientation;

    PetscCall(DMPlexGetConeSize(K, p, &dof));
    PetscCall(DMPlexGetCone(K, p, &cone));
    PetscCall(DMPlexGetConeOrientation(K, p, &orientation));
    PetscCall(PetscSectionGetOffset(unionSection, p - pStart,&uOff));
    PetscCall(PetscSectionGetOffset(unionConeSection,uOff,&cOff));
    for (c = 0; c < dof; c++) {
      PetscInt e, eOff;
      e                           = cone[c];
      PetscCall(PetscSectionGetOffset(unionSection, e - pStart, &eOff));
      unionCones[cOff + c]        = eOff;
      unionOrientations[cOff + c] = orientation[c];
    }
  }
  for (p = pRefStart; p < pRefEnd; p++) {
    PetscInt dof, uDof, uOff, c, cOff;
    const PetscInt *cone, *orientation;

    PetscCall(DMPlexGetConeSize(Kref, p, &dof));
    PetscCall(DMPlexGetCone(Kref, p, &cone));
    PetscCall(DMPlexGetConeOrientation(Kref, p, &orientation));
    PetscCall(PetscSectionGetDof(unionSection, p - pRefStart + (pEnd - pStart),&uDof));
    PetscCall(PetscSectionGetOffset(unionSection, p - pRefStart + (pEnd - pStart),&uOff));
    if (uDof) {
      PetscCall(PetscSectionGetOffset(unionConeSection,uOff,&cOff));
      for (c = 0; c < dof; c++) {
        PetscInt e, eOff, eDof;

        e    = cone[c];
        PetscCall(PetscSectionGetDof(unionSection, e - pRefStart + (pEnd - pStart),&eDof));
        if (eDof) {
          PetscCall(PetscSectionGetOffset(unionSection, e - pRefStart + (pEnd - pStart), &eOff));
        }
        else {
          PetscCall(DMLabelGetValue(identityRef, e, &e));
          PetscCall(PetscSectionGetOffset(unionSection, e - pStart, &eOff));
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

    PetscCall(DMGetCoordinateSection(K, &KcoordsSec));
    PetscCall(DMGetCoordinatesLocal(K, &KcoordsVec));
    PetscCall(DMGetCoordinateSection(Kref, &KrefCoordsSec));
    PetscCall(DMGetCoordinatesLocal(Kref, &KrefCoordsVec));

    numVerts = numDimPoints[0];
    PetscCall(PetscMalloc1(numVerts * dim,&unionCoords));
    PetscCall(DMPlexGetDepthStratum(K,0,&vStart,&vEnd));

    offset = 0;
    for (v = vStart; v < vEnd; v++) {
      PetscCall(PetscSectionGetOffset(unionSection,v - pStart,&vOff));
      PetscCall(VecGetValuesSection(KcoordsVec, KcoordsSec, v, &Kcoords));
      for (d = 0; d < dim; d++) {
        unionCoords[offset * dim + d] = Kcoords[d];
      }
      offset++;
    }
    PetscCall(DMPlexGetDepthStratum(Kref,0,&vRefStart,&vRefEnd));
    for (v = vRefStart; v < vRefEnd; v++) {
      PetscCall(PetscSectionGetDof(unionSection,v - pRefStart + (pEnd - pStart),&vDof));
      PetscCall(PetscSectionGetOffset(unionSection,v - pRefStart + (pEnd - pStart),&vOff));
      PetscCall(VecGetValuesSection(KrefCoordsVec, KrefCoordsSec, v, &Kcoords));
      if (vDof) {
        for (d = 0; d < dim; d++) {
          unionCoords[offset * dim + d] = Kcoords[d];
        }
        offset++;
      }
    }
  }
  PetscCall(DMCreate(comm,ref));
  PetscCall(DMSetType(*ref,DMPLEX));
  PetscCall(DMSetDimension(*ref,dim));
  PetscCall(DMPlexCreateFromDAG(*ref,dim,numDimPoints,coneSizes,unionCones,unionOrientations,unionCoords));
  /* set the tree */
  PetscCall(PetscSectionCreate(comm,&parentSection));
  PetscCall(PetscSectionSetChart(parentSection,0,numUnionPoints));
  for (p = pRefStart; p < pRefEnd; p++) {
    PetscInt uDof, uOff;

    PetscCall(PetscSectionGetDof(unionSection, p - pRefStart + (pEnd - pStart),&uDof));
    PetscCall(PetscSectionGetOffset(unionSection, p - pRefStart + (pEnd - pStart),&uOff));
    if (uDof) {
      PetscCall(PetscSectionSetDof(parentSection,uOff,1));
    }
  }
  PetscCall(PetscSectionSetUp(parentSection));
  PetscCall(PetscSectionGetStorageSize(parentSection,&parentSize));
  PetscCall(PetscMalloc2(parentSize,&parents,parentSize,&childIDs));
  for (p = pRefStart; p < pRefEnd; p++) {
    PetscInt uDof, uOff;

    PetscCall(PetscSectionGetDof(unionSection, p - pRefStart + (pEnd - pStart),&uDof));
    PetscCall(PetscSectionGetOffset(unionSection, p - pRefStart + (pEnd - pStart),&uOff));
    if (uDof) {
      PetscInt pOff, parent, parentU;
      PetscCall(PetscSectionGetOffset(parentSection,uOff,&pOff));
      PetscCall(DMLabelGetValue(identityRef,p,&parent));
      PetscCall(PetscSectionGetOffset(unionSection, parent - pStart,&parentU));
      parents[pOff] = parentU;
      childIDs[pOff] = uOff;
    }
  }
  PetscCall(DMPlexCreateReferenceTree_SetTree(*ref,parentSection,parents,childIDs));
  PetscCall(PetscSectionDestroy(&parentSection));
  PetscCall(PetscFree2(parents,childIDs));

  /* clean up */
  PetscCall(PetscSectionDestroy(&unionSection));
  PetscCall(PetscSectionDestroy(&unionConeSection));
  PetscCall(ISDestroy(&perm));
  PetscCall(PetscFree(unionCoords));
  PetscCall(PetscFree2(unionCones,unionOrientations));
  PetscCall(PetscFree2(coneSizes,numDimPoints));
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

.seealso: `DMPlexSetReferenceTree()`, `DMPlexGetReferenceTree()`
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
  PetscCall(DMPlexCreateReferenceCell(comm, DMPolytopeTypeSimpleShape(dim, simplex), &K));
  PetscCall(DMCreateLabel(K, "identity"));
  PetscCall(DMGetLabel(K, "identity", &identity));
  PetscCall(DMPlexGetChart(K, &pStart, &pEnd));
  for (p = pStart; p < pEnd; p++) {
    PetscCall(DMLabelSetValue(identity, p, p));
  }
  /* refine it */
  PetscCall(DMRefine(K,comm,&Kref));

  /* the reference tree is the union of these two, without duplicating
   * points that appear in both */
  PetscCall(DMPlexCreateReferenceTree_Union(K, Kref, "identity", ref));
  mesh = (DM_Plex *) (*ref)->data;
  mesh->getchildsymmetry = DMPlexReferenceTreeGetChildSymmetry_Default;
  PetscCall(DMDestroy(&K));
  PetscCall(DMDestroy(&Kref));
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
  PetscCall(PetscSectionDestroy(&mesh->childSection));
  PetscCall(PetscFree(mesh->children));
  pSec = mesh->parentSection;
  if (!pSec) PetscFunctionReturn(0);
  PetscCall(PetscSectionGetStorageSize(pSec,&pSize));
  for (p = 0; p < pSize; p++) {
    PetscInt par = mesh->parents[p];

    parMax = PetscMax(parMax,par+1);
    parMin = PetscMin(parMin,par);
  }
  if (parMin > parMax) {
    parMin = -1;
    parMax = -1;
  }
  PetscCall(PetscSectionCreate(PetscObjectComm((PetscObject)pSec),&childSec));
  PetscCall(PetscSectionSetChart(childSec,parMin,parMax));
  for (p = 0; p < pSize; p++) {
    PetscInt par = mesh->parents[p];

    PetscCall(PetscSectionAddDof(childSec,par,1));
  }
  PetscCall(PetscSectionSetUp(childSec));
  PetscCall(PetscSectionGetStorageSize(childSec,&cSize));
  PetscCall(PetscMalloc1(cSize,&children));
  PetscCall(PetscCalloc1(parMax-parMin,&offsets));
  PetscCall(PetscSectionGetChart(pSec,&pStart,&pEnd));
  for (p = pStart; p < pEnd; p++) {
    PetscInt dof, off, i;

    PetscCall(PetscSectionGetDof(pSec,p,&dof));
    PetscCall(PetscSectionGetOffset(pSec,p,&off));
    for (i = 0; i < dof; i++) {
      PetscInt par = mesh->parents[off + i], cOff;

      PetscCall(PetscSectionGetOffset(childSec,par,&cOff));
      children[cOff + offsets[par-parMin]++] = p;
    }
  }
  mesh->childSection = childSec;
  mesh->children = children;
  PetscCall(PetscFree(offsets));
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
  PetscCall(PetscSectionGetChart(section,&pStart,&pEnd));
  PetscCall(ISGetLocalSize(is,&size));
  PetscCall(ISGetIndices(is,&vals));
  PetscCall(PetscSectionCreate(PetscObjectComm((PetscObject)section),&secNew));
  PetscCall(PetscSectionSetChart(secNew,pStart,pEnd));
  for (i = 0; i < size; i++) {
    PetscInt dof;

    p = vals[i];
    if (p < pStart || p >= pEnd) continue;
    PetscCall(PetscSectionGetDof(section, p, &dof));
    if (dof) break;
  }
  if (i == size) {
    PetscCall(PetscSectionSetUp(secNew));
    anyNew   = PETSC_FALSE;
    compress = PETSC_FALSE;
    sizeNew  = 0;
  }
  else {
    anyNew = PETSC_TRUE;
    for (p = pStart; p < pEnd; p++) {
      PetscInt dof, off;

      PetscCall(PetscSectionGetDof(section, p, &dof));
      PetscCall(PetscSectionGetOffset(section, p, &off));
      for (i = 0; i < dof; i++) {
        PetscInt q = vals[off + i], qDof = 0;

        if (q >= pStart && q < pEnd) {
          PetscCall(PetscSectionGetDof(section, q, &qDof));
        }
        if (qDof) {
          PetscCall(PetscSectionAddDof(secNew, p, qDof));
        }
        else {
          PetscCall(PetscSectionAddDof(secNew, p, 1));
        }
      }
    }
    PetscCall(PetscSectionSetUp(secNew));
    PetscCall(PetscSectionGetStorageSize(secNew,&sizeNew));
    PetscCall(PetscMalloc1(sizeNew,&valsNew));
    compress = PETSC_FALSE;
    for (p = pStart; p < pEnd; p++) {
      PetscInt dof, off, count, offNew, dofNew;

      PetscCall(PetscSectionGetDof(section, p, &dof));
      PetscCall(PetscSectionGetOffset(section, p, &off));
      PetscCall(PetscSectionGetDof(secNew, p, &dofNew));
      PetscCall(PetscSectionGetOffset(secNew, p, &offNew));
      count = 0;
      for (i = 0; i < dof; i++) {
        PetscInt q = vals[off + i], qDof = 0, qOff = 0, j;

        if (q >= pStart && q < pEnd) {
          PetscCall(PetscSectionGetDof(section, q, &qDof));
          PetscCall(PetscSectionGetOffset(section, q, &qOff));
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
        PetscCall(PetscSectionSetDof(secNew, p, count));
        compress = PETSC_TRUE;
      }
    }
  }
  PetscCall(ISRestoreIndices(is,&vals));
  PetscCall(MPIU_Allreduce(&anyNew,&globalAnyNew,1,MPIU_BOOL,MPI_LOR,PetscObjectComm((PetscObject)secNew)));
  if (!globalAnyNew) {
    PetscCall(PetscSectionDestroy(&secNew));
    *sectionNew = NULL;
    *isNew = NULL;
  }
  else {
    PetscBool globalCompress;

    PetscCall(MPIU_Allreduce(&compress,&globalCompress,1,MPIU_BOOL,MPI_LOR,PetscObjectComm((PetscObject)secNew)));
    if (compress) {
      PetscSection secComp;
      PetscInt *valsComp = NULL;

      PetscCall(PetscSectionCreate(PetscObjectComm((PetscObject)section),&secComp));
      PetscCall(PetscSectionSetChart(secComp,pStart,pEnd));
      for (p = pStart; p < pEnd; p++) {
        PetscInt dof;

        PetscCall(PetscSectionGetDof(secNew, p, &dof));
        PetscCall(PetscSectionSetDof(secComp, p, dof));
      }
      PetscCall(PetscSectionSetUp(secComp));
      PetscCall(PetscSectionGetStorageSize(secComp,&sizeNew));
      PetscCall(PetscMalloc1(sizeNew,&valsComp));
      for (p = pStart; p < pEnd; p++) {
        PetscInt dof, off, offNew, j;

        PetscCall(PetscSectionGetDof(secNew, p, &dof));
        PetscCall(PetscSectionGetOffset(secNew, p, &off));
        PetscCall(PetscSectionGetOffset(secComp, p, &offNew));
        for (j = 0; j < dof; j++) {
          valsComp[offNew + j] = valsNew[off + j];
        }
      }
      PetscCall(PetscSectionDestroy(&secNew));
      secNew  = secComp;
      PetscCall(PetscFree(valsNew));
      valsNew = valsComp;
    }
    PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)is),sizeNew,valsNew,PETSC_OWN_POINTER,isNew));
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
  PetscCall(DMPlexGetChart(dm,&pStart,&pEnd));
  PetscCall(DMGetLabel(dm,"canonical",&canonLabel));
  for (p = pStart; p < pEnd; p++) {
    PetscInt parent;

    if (canonLabel) {
      PetscInt canon;

      PetscCall(DMLabelGetValue(canonLabel,p,&canon));
      if (p != canon) continue;
    }
    PetscCall(DMPlexGetTreeParent(dm,p,&parent,NULL));
    if (parent != p) {
      aMin = PetscMin(aMin,p);
      aMax = PetscMax(aMax,p+1);
    }
  }
  if (aMin > aMax) {
    aMin = -1;
    aMax = -1;
  }
  PetscCall(PetscSectionCreate(PETSC_COMM_SELF,&aSec));
  PetscCall(PetscSectionSetChart(aSec,aMin,aMax));
  for (p = aMin; p < aMax; p++) {
    PetscInt parent, ancestor = p;

    if (canonLabel) {
      PetscInt canon;

      PetscCall(DMLabelGetValue(canonLabel,p,&canon));
      if (p != canon) continue;
    }
    PetscCall(DMPlexGetTreeParent(dm,p,&parent,NULL));
    while (parent != ancestor) {
      ancestor = parent;
      PetscCall(DMPlexGetTreeParent(dm,ancestor,&parent,NULL));
    }
    if (ancestor != p) {
      PetscInt closureSize, *closure = NULL;

      PetscCall(DMPlexGetTransitiveClosure(dm,ancestor,PETSC_TRUE,&closureSize,&closure));
      PetscCall(PetscSectionSetDof(aSec,p,closureSize));
      PetscCall(DMPlexRestoreTransitiveClosure(dm,ancestor,PETSC_TRUE,&closureSize,&closure));
    }
  }
  PetscCall(PetscSectionSetUp(aSec));
  PetscCall(PetscSectionGetStorageSize(aSec,&size));
  PetscCall(PetscMalloc1(size,&anchors));
  for (p = aMin; p < aMax; p++) {
    PetscInt parent, ancestor = p;

    if (canonLabel) {
      PetscInt canon;

      PetscCall(DMLabelGetValue(canonLabel,p,&canon));
      if (p != canon) continue;
    }
    PetscCall(DMPlexGetTreeParent(dm,p,&parent,NULL));
    while (parent != ancestor) {
      ancestor = parent;
      PetscCall(DMPlexGetTreeParent(dm,ancestor,&parent,NULL));
    }
    if (ancestor != p) {
      PetscInt j, closureSize, *closure = NULL, aOff;

      PetscCall(PetscSectionGetOffset(aSec,p,&aOff));

      PetscCall(DMPlexGetTransitiveClosure(dm,ancestor,PETSC_TRUE,&closureSize,&closure));
      for (j = 0; j < closureSize; j++) {
        anchors[aOff + j] = closure[2*j];
      }
      PetscCall(DMPlexRestoreTransitiveClosure(dm,ancestor,PETSC_TRUE,&closureSize,&closure));
    }
  }
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF,size,anchors,PETSC_OWN_POINTER,&aIS));
  {
    PetscSection aSecNew = aSec;
    IS           aISNew  = aIS;

    PetscCall(PetscObjectReference((PetscObject)aSec));
    PetscCall(PetscObjectReference((PetscObject)aIS));
    while (aSecNew) {
      PetscCall(PetscSectionDestroy(&aSec));
      PetscCall(ISDestroy(&aIS));
      aSec    = aSecNew;
      aIS     = aISNew;
      aSecNew = NULL;
      aISNew  = NULL;
      PetscCall(AnchorsFlatten(aSec,aIS,&aSecNew,&aISNew));
    }
  }
  PetscCall(DMPlexSetAnchors(dm,aSec,aIS));
  PetscCall(PetscSectionDestroy(&aSec));
  PetscCall(ISDestroy(&aIS));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexGetTrueSupportSize(DM dm,PetscInt p,PetscInt *dof,PetscInt *numTrueSupp)
{
  PetscFunctionBegin;
  if (numTrueSupp[p] == -1) {
    PetscInt i, alldof;
    const PetscInt *supp;
    PetscInt count = 0;

    PetscCall(DMPlexGetSupportSize(dm,p,&alldof));
    PetscCall(DMPlexGetSupport(dm,p,&supp));
    for (i = 0; i < alldof; i++) {
      PetscInt q = supp[i], numCones, j;
      const PetscInt *cone;

      PetscCall(DMPlexGetConeSize(dm,q,&numCones));
      PetscCall(DMPlexGetCone(dm,q,&cone));
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
  PetscCall(DMPlexGetDepth(dm,&depth));
  PetscCall(PetscSectionCreate(PetscObjectComm((PetscObject)(mesh->supportSection)),&newSupportSection));
  PetscCall(DMPlexGetChart(dm,&pStart,&pEnd));
  PetscCall(PetscSectionSetChart(newSupportSection,pStart,pEnd));
  PetscCall(PetscCalloc1(pEnd,&offsets));
  PetscCall(PetscMalloc1(pEnd,&numTrueSupp));
  for (p = 0; p < pEnd; p++) numTrueSupp[p] = -1;
  /* if a point is in the (true) support of q, it should be in the support of
   * parent(q) */
  for (d = 0; d <= depth; d++) {
    PetscCall(DMPlexGetHeightStratum(dm,d,&pStart,&pEnd));
    for (p = pStart; p < pEnd; ++p) {
      PetscInt dof, q, qdof, parent;

      PetscCall(DMPlexGetTrueSupportSize(dm,p,&dof,numTrueSupp));
      PetscCall(PetscSectionAddDof(newSupportSection, p, dof));
      q    = p;
      PetscCall(DMPlexGetTreeParent(dm,q,&parent,NULL));
      while (parent != q && parent >= pStart && parent < pEnd) {
        q = parent;

        PetscCall(DMPlexGetTrueSupportSize(dm,q,&qdof,numTrueSupp));
        PetscCall(PetscSectionAddDof(newSupportSection,p,qdof));
        PetscCall(PetscSectionAddDof(newSupportSection,q,dof));
        PetscCall(DMPlexGetTreeParent(dm,q,&parent,NULL));
      }
    }
  }
  PetscCall(PetscSectionSetUp(newSupportSection));
  PetscCall(PetscSectionGetStorageSize(newSupportSection,&newSize));
  PetscCall(PetscMalloc1(newSize,&newSupports));
  for (d = 0; d <= depth; d++) {
    PetscCall(DMPlexGetHeightStratum(dm,d,&pStart,&pEnd));
    for (p = pStart; p < pEnd; p++) {
      PetscInt dof, off, q, qdof, qoff, newDof, newOff, newqOff, i, parent;

      PetscCall(PetscSectionGetDof(mesh->supportSection, p, &dof));
      PetscCall(PetscSectionGetOffset(mesh->supportSection, p, &off));
      PetscCall(PetscSectionGetDof(newSupportSection, p, &newDof));
      PetscCall(PetscSectionGetOffset(newSupportSection, p, &newOff));
      for (i = 0; i < dof; i++) {
        PetscInt numCones, j;
        const PetscInt *cone;
        PetscInt q = mesh->supports[off + i];

        PetscCall(DMPlexGetConeSize(dm,q,&numCones));
        PetscCall(DMPlexGetCone(dm,q,&cone));
        for (j = 0; j < numCones; j++) {
          if (cone[j] == p) break;
        }
        if (j < numCones) newSupports[newOff+offsets[p]++] = q;
      }

      q    = p;
      PetscCall(DMPlexGetTreeParent(dm,q,&parent,NULL));
      while (parent != q && parent >= pStart && parent < pEnd) {
        q = parent;
        PetscCall(PetscSectionGetDof(mesh->supportSection, q, &qdof));
        PetscCall(PetscSectionGetOffset(mesh->supportSection, q, &qoff));
        PetscCall(PetscSectionGetOffset(newSupportSection, q, &newqOff));
        for (i = 0; i < qdof; i++) {
          PetscInt numCones, j;
          const PetscInt *cone;
          PetscInt r = mesh->supports[qoff + i];

          PetscCall(DMPlexGetConeSize(dm,r,&numCones));
          PetscCall(DMPlexGetCone(dm,r,&cone));
          for (j = 0; j < numCones; j++) {
            if (cone[j] == q) break;
          }
          if (j < numCones) newSupports[newOff+offsets[p]++] = r;
        }
        for (i = 0; i < dof; i++) {
          PetscInt numCones, j;
          const PetscInt *cone;
          PetscInt r = mesh->supports[off + i];

          PetscCall(DMPlexGetConeSize(dm,r,&numCones));
          PetscCall(DMPlexGetCone(dm,r,&cone));
          for (j = 0; j < numCones; j++) {
            if (cone[j] == p) break;
          }
          if (j < numCones) newSupports[newqOff+offsets[q]++] = r;
        }
        PetscCall(DMPlexGetTreeParent(dm,q,&parent,NULL));
      }
    }
  }
  PetscCall(PetscSectionDestroy(&mesh->supportSection));
  mesh->supportSection = newSupportSection;
  PetscCall(PetscFree(mesh->supports));
  mesh->supports = newSupports;
  PetscCall(PetscFree(offsets));
  PetscCall(PetscFree(numTrueSupp));

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
  PetscCall(PetscObjectReference((PetscObject)parentSection));
  PetscCall(PetscSectionDestroy(&mesh->parentSection));
  mesh->parentSection = parentSection;
  PetscCall(PetscSectionGetStorageSize(parentSection,&size));
  if (parents != mesh->parents) {
    PetscCall(PetscFree(mesh->parents));
    PetscCall(PetscMalloc1(size,&mesh->parents));
    PetscCall(PetscArraycpy(mesh->parents, parents, size));
  }
  if (childIDs != mesh->childIDs) {
    PetscCall(PetscFree(mesh->childIDs));
    PetscCall(PetscMalloc1(size,&mesh->childIDs));
    PetscCall(PetscArraycpy(mesh->childIDs, childIDs, size));
  }
  PetscCall(DMPlexGetReferenceTree(dm,&refTree));
  if (refTree) {
    DMLabel canonLabel;

    PetscCall(DMGetLabel(refTree,"canonical",&canonLabel));
    if (canonLabel) {
      PetscInt i;

      for (i = 0; i < size; i++) {
        PetscInt canon;
        PetscCall(DMLabelGetValue(canonLabel, mesh->childIDs[i], &canon));
        if (canon >= 0) {
          mesh->childIDs[i] = canon;
        }
      }
    }
    mesh->computeanchormatrix = DMPlexComputeAnchorMatrix_Tree_FromReference;
  } else {
    mesh->computeanchormatrix = DMPlexComputeAnchorMatrix_Tree_Direct;
  }
  PetscCall(DMPlexTreeSymmetrize(dm));
  if (computeCanonical) {
    PetscInt d, dim;

    /* add the canonical label */
    PetscCall(DMGetDimension(dm,&dim));
    PetscCall(DMCreateLabel(dm,"canonical"));
    for (d = 0; d <= dim; d++) {
      PetscInt p, dStart, dEnd, canon = -1, cNumChildren;
      const PetscInt *cChildren;

      PetscCall(DMPlexGetDepthStratum(dm,d,&dStart,&dEnd));
      for (p = dStart; p < dEnd; p++) {
        PetscCall(DMPlexGetTreeChildren(dm,p,&cNumChildren,&cChildren));
        if (cNumChildren) {
          canon = p;
          break;
        }
      }
      if (canon == -1) continue;
      for (p = dStart; p < dEnd; p++) {
        PetscInt numChildren, i;
        const PetscInt *children;

        PetscCall(DMPlexGetTreeChildren(dm,p,&numChildren,&children));
        if (numChildren) {
          PetscCheck(numChildren == cNumChildren,PetscObjectComm((PetscObject)dm),PETSC_ERR_PLIB,"All parent points in a stratum should have the same number of children: %" PetscInt_FMT " != %" PetscInt_FMT, numChildren, cNumChildren);
          PetscCall(DMSetLabelValue(dm,"canonical",p,canon));
          for (i = 0; i < numChildren; i++) {
            PetscCall(DMSetLabelValue(dm,"canonical",children[i],cChildren[i]));
          }
        }
      }
    }
  }
  if (exchangeSupports) {
    PetscCall(DMPlexTreeExchangeSupports(dm));
  }
  mesh->createanchors = DMPlexCreateAnchors_Tree;
  /* reset anchors */
  PetscCall(DMPlexSetAnchors(dm,NULL,NULL));
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

.seealso: `DMPlexGetTree()`, `DMPlexSetReferenceTree()`, `DMPlexSetAnchors()`, `DMPlexGetTreeParent()`, `DMPlexGetTreeChildren()`
@*/
PetscErrorCode DMPlexSetTree(DM dm, PetscSection parentSection, PetscInt parents[], PetscInt childIDs[])
{
  PetscFunctionBegin;
  PetscCall(DMPlexSetTree_Internal(dm,parentSection,parents,childIDs,PETSC_FALSE,PETSC_TRUE));
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

.seealso: `DMPlexSetTree()`, `DMPlexSetReferenceTree()`, `DMPlexSetAnchors()`, `DMPlexGetTreeParent()`, `DMPlexGetTreeChildren()`
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

.seealso: `DMPlexSetTree()`, `DMPlexGetTree()`, `DMPlexGetTreeChildren()`
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

    PetscCall(PetscSectionGetDof (pSec, point, &dof));
    if (dof) {
      PetscInt off;

      PetscCall(PetscSectionGetOffset (pSec, point, &off));
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

.seealso: `DMPlexSetTree()`, `DMPlexGetTree()`, `DMPlexGetTreeParent()`
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
    PetscCall(PetscSectionGetDof (childSec, point, &dof));
  }
  if (numChildren) *numChildren = dof;
  if (children) {
    if (dof) {
      PetscInt off;

      PetscCall(PetscSectionGetOffset (childSec, point, &off));
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
  PetscCall(PetscSpaceEvaluate(space,nPoints,points,work,NULL,NULL));
  for (f = 0, offset = 0; f < nFunctionals; f++) {
    qPoints = pointsPerFn[f];
    for (b = 0; b < nBasis; b++) {
      PetscScalar val = 0.;

      for (p = 0; p < qPoints; p++) {
        for (c = 0; c < nComps; c++) {
          val += work[((offset + p) * nBasis + b) * nComps + c] * weights[(offset + p) * nComps + c];
        }
      }
      PetscCall(MatSetValue(basisAtPoints,b,f,val,INSERT_VALUES));
    }
    offset += qPoints;
  }
  PetscCall(MatAssemblyBegin(basisAtPoints,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(basisAtPoints,MAT_FINAL_ASSEMBLY));
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
  PetscCall(DMPlexGetChart(dm,&pStart,&pEnd));
  PetscCall(DMGetDS(dm,&ds));
  PetscCall(PetscDSGetNumFields(ds,&numFields));
  PetscCall(DMPlexGetHeightStratum(dm,0,&cStart,&cEnd));
  PetscCall(DMPlexGetAnchors(dm,&aSec,&aIS));
  PetscCall(ISGetIndices(aIS,&anchors));
  PetscCall(PetscSectionGetChart(cSec,&conStart,&conEnd));
  PetscCall(DMGetDimension(dm,&spdim));
  PetscCall(PetscMalloc6(spdim,&v0,spdim,&v0parent,spdim,&vtmp,spdim*spdim,&J,spdim*spdim,&Jparent,spdim*spdim,&invJparent));

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

    PetscCall(PetscDSGetDiscretization(ds,f,&disc));
    PetscCall(PetscObjectGetClassId(disc,&id));
    if (id == PETSCFE_CLASSID) {
      PetscFE fe = (PetscFE) disc;

      PetscCall(PetscFEGetBasisSpace(fe,&bspace));
      PetscCall(PetscFEGetDualSpace(fe,&dspace));
      PetscCall(PetscDualSpaceGetDimension(dspace,&fSize));
      PetscCall(PetscFEGetNumComponents(fe,&Nc));
    }
    else if (id == PETSCFV_CLASSID) {
      PetscFV fv = (PetscFV) disc;

      PetscCall(PetscFVGetNumComponents(fv,&Nc));
      PetscCall(PetscSpaceCreate(PetscObjectComm((PetscObject)fv),&bspace));
      PetscCall(PetscSpaceSetType(bspace,PETSCSPACEPOLYNOMIAL));
      PetscCall(PetscSpaceSetDegree(bspace,0,PETSC_DETERMINE));
      PetscCall(PetscSpaceSetNumComponents(bspace,Nc));
      PetscCall(PetscSpaceSetNumVariables(bspace,spdim));
      PetscCall(PetscSpaceSetUp(bspace));
      PetscCall(PetscFVGetDualSpace(fv,&dspace));
      PetscCall(PetscDualSpaceGetDimension(dspace,&fSize));
    }
    else SETERRQ(PetscObjectComm(disc),PETSC_ERR_ARG_UNKNOWN_TYPE, "PetscDS discretization id %d not recognized.", id);
    PetscCall(PetscDualSpaceGetNumDof(dspace,&numDof));
    for (i = 0, maxDof = 0; i <= spdim; i++) {maxDof = PetscMax(maxDof,numDof[i]);}
    PetscCall(PetscDualSpaceGetSymmetries(dspace,&perms,&flips));

    PetscCall(MatCreate(PETSC_COMM_SELF,&Amat));
    PetscCall(MatSetSizes(Amat,fSize,fSize,fSize,fSize));
    PetscCall(MatSetType(Amat,MATSEQDENSE));
    PetscCall(MatSetUp(Amat));
    PetscCall(MatDuplicate(Amat,MAT_DO_NOT_COPY_VALUES,&Bmat));
    PetscCall(MatDuplicate(Amat,MAT_DO_NOT_COPY_VALUES,&Xmat));
    nPoints = 0;
    for (i = 0; i < fSize; i++) {
      PetscInt        qPoints, thisNc;
      PetscQuadrature quad;

      PetscCall(PetscDualSpaceGetFunctional(dspace,i,&quad));
      PetscCall(PetscQuadratureGetData(quad,NULL,&thisNc,&qPoints,NULL,NULL));
      PetscCheck(thisNc == Nc,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Functional dim %" PetscInt_FMT " does not much basis dim %" PetscInt_FMT,thisNc,Nc);
      nPoints += qPoints;
    }
    PetscCall(PetscMalloc7(fSize,&sizes,nPoints*Nc,&weights,spdim*nPoints,&pointsRef,spdim*nPoints,&pointsReal,nPoints*fSize*Nc,&work,maxDof,&workIndRow,maxDof,&workIndCol));
    PetscCall(PetscMalloc1(maxDof * maxDof,&scwork));
    offset = 0;
    for (i = 0; i < fSize; i++) {
      PetscInt        qPoints;
      const PetscReal    *p, *w;
      PetscQuadrature quad;

      PetscCall(PetscDualSpaceGetFunctional(dspace,i,&quad));
      PetscCall(PetscQuadratureGetData(quad,NULL,NULL,&qPoints,&p,&w));
      PetscCall(PetscArraycpy(weights+Nc*offset,w,Nc*qPoints));
      PetscCall(PetscArraycpy(pointsRef+spdim*offset,p,spdim*qPoints));
      sizes[i] = qPoints;
      offset  += qPoints;
    }
    PetscCall(EvaluateBasis(bspace,fSize,fSize,Nc,nPoints,sizes,pointsRef,weights,work,Amat));
    PetscCall(MatLUFactor(Amat,NULL,NULL,NULL));
    for (c = cStart; c < cEnd; c++) {
      PetscInt        parent;
      PetscInt        closureSize, closureSizeP, *closure = NULL, *closureP = NULL;
      PetscInt        *childOffsets, *parentOffsets;

      PetscCall(DMPlexGetTreeParent(dm,c,&parent,NULL));
      if (parent == c) continue;
      PetscCall(DMPlexGetTransitiveClosure(dm,c,PETSC_TRUE,&closureSize,&closure));
      for (i = 0; i < closureSize; i++) {
        PetscInt p = closure[2*i];
        PetscInt conDof;

        if (p < conStart || p >= conEnd) continue;
        if (numFields) {
          PetscCall(PetscSectionGetFieldDof(cSec,p,f,&conDof));
        }
        else {
          PetscCall(PetscSectionGetDof(cSec,p,&conDof));
        }
        if (conDof) break;
      }
      if (i == closureSize) {
        PetscCall(DMPlexRestoreTransitiveClosure(dm,c,PETSC_TRUE,&closureSize,&closure));
        continue;
      }

      PetscCall(DMPlexComputeCellGeometryFEM(dm, c, NULL, v0, J, NULL, &detJ));
      PetscCall(DMPlexComputeCellGeometryFEM(dm, parent, NULL, v0parent, Jparent, invJparent, &detJparent));
      for (i = 0; i < nPoints; i++) {
        const PetscReal xi0[3] = {-1.,-1.,-1.};

        CoordinatesRefToReal(spdim, spdim, xi0, v0, J, &pointsRef[i*spdim],vtmp);
        CoordinatesRealToRef(spdim, spdim, xi0, v0parent, invJparent, vtmp, &pointsReal[i*spdim]);
      }
      PetscCall(EvaluateBasis(bspace,fSize,fSize,Nc,nPoints,sizes,pointsReal,weights,work,Bmat));
      PetscCall(MatMatSolve(Amat,Bmat,Xmat));
      PetscCall(MatDenseGetArrayRead(Xmat,&X));
      PetscCall(DMPlexGetTransitiveClosure(dm,parent,PETSC_TRUE,&closureSizeP,&closureP));
      PetscCall(PetscMalloc2(closureSize+1,&childOffsets,closureSizeP+1,&parentOffsets));
      childOffsets[0] = 0;
      for (i = 0; i < closureSize; i++) {
        PetscInt p = closure[2*i];
        PetscInt dof;

        if (numFields) {
          PetscCall(PetscSectionGetFieldDof(section,p,f,&dof));
        }
        else {
          PetscCall(PetscSectionGetDof(section,p,&dof));
        }
        childOffsets[i+1]=childOffsets[i]+dof;
      }
      parentOffsets[0] = 0;
      for (i = 0; i < closureSizeP; i++) {
        PetscInt p = closureP[2*i];
        PetscInt dof;

        if (numFields) {
          PetscCall(PetscSectionGetFieldDof(section,p,f,&dof));
        }
        else {
          PetscCall(PetscSectionGetDof(section,p,&dof));
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
          PetscCall(PetscSectionGetFieldDof(cSec,p,f,&conDof));
          PetscCall(PetscSectionGetFieldOffset(cSec,p,f,&conOff));
        }
        else {
          PetscCall(PetscSectionGetDof(cSec,p,&conDof));
          PetscCall(PetscSectionGetOffset(cSec,p,&conOff));
        }
        if (!conDof) continue;
        perm  = (perms && perms[i]) ? perms[i][o] : NULL;
        flip  = (flips && flips[i]) ? flips[i][o] : NULL;
        PetscCall(PetscSectionGetDof(aSec,p,&aDof));
        PetscCall(PetscSectionGetOffset(aSec,p,&aOff));
        nWork = childOffsets[i+1]-childOffsets[i];
        for (k = 0; k < aDof; k++) {
          PetscInt a = anchors[aOff + k];
          PetscInt aSecDof, aSecOff;

          if (numFields) {
            PetscCall(PetscSectionGetFieldDof(section,a,f,&aSecDof));
            PetscCall(PetscSectionGetFieldOffset(section,a,f,&aSecOff));
          }
          else {
            PetscCall(PetscSectionGetDof(section,a,&aSecDof));
            PetscCall(PetscSectionGetOffset(section,a,&aSecOff));
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
              PetscCall(MatSetValues(cMat,nWork,workIndRow,nWorkP,workIndCol,scwork,INSERT_VALUES));
              break;
            }
          }
        }
      }
      PetscCall(MatDenseRestoreArrayRead(Xmat,&X));
      PetscCall(PetscFree2(childOffsets,parentOffsets));
      PetscCall(DMPlexRestoreTransitiveClosure(dm,c,PETSC_TRUE,&closureSize,&closure));
      PetscCall(DMPlexRestoreTransitiveClosure(dm,parent,PETSC_TRUE,&closureSizeP,&closureP));
    }
    PetscCall(MatDestroy(&Amat));
    PetscCall(MatDestroy(&Bmat));
    PetscCall(MatDestroy(&Xmat));
    PetscCall(PetscFree(scwork));
    PetscCall(PetscFree7(sizes,weights,pointsRef,pointsReal,work,workIndRow,workIndCol));
    if (id == PETSCFV_CLASSID) {
      PetscCall(PetscSpaceDestroy(&bspace));
    }
  }
  PetscCall(MatAssemblyBegin(cMat,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(cMat,MAT_FINAL_ASSEMBLY));
  PetscCall(PetscFree6(v0,v0parent,vtmp,J,Jparent,invJparent));
  PetscCall(ISRestoreIndices(aIS,&anchors));

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
  PetscCall(DMGetDS(refTree,&ds));
  PetscCall(PetscDSGetNumFields(ds,&numFields));
  maxFields = PetscMax(1,numFields);
  PetscCall(DMGetDefaultConstraints(refTree,&refConSec,&refCmat,NULL));
  PetscCall(DMPlexGetAnchors(refTree,&refAnSec,&refAnIS));
  PetscCall(ISGetIndices(refAnIS,&refAnchors));
  PetscCall(DMGetLocalSection(refTree,&refSection));
  PetscCall(PetscSectionGetChart(refConSec,&pRefStart,&pRefEnd));
  PetscCall(PetscMalloc1(pRefEnd-pRefStart,&refPointFieldMats));
  PetscCall(PetscMalloc1(pRefEnd-pRefStart,&refPointFieldN));
  PetscCall(PetscSectionGetMaxDof(refConSec,&maxDof));
  PetscCall(PetscSectionGetMaxDof(refAnSec,&maxAnDof));
  PetscCall(PetscMalloc1(maxDof,&rows));
  PetscCall(PetscMalloc1(maxDof*maxAnDof,&cols));
  for (p = pRefStart; p < pRefEnd; p++) {
    PetscInt parent, closureSize, *closure = NULL, pDof;

    PetscCall(DMPlexGetTreeParent(refTree,p,&parent,NULL));
    PetscCall(PetscSectionGetDof(refConSec,p,&pDof));
    if (!pDof || parent == p) continue;

    PetscCall(PetscMalloc1(maxFields,&refPointFieldMats[p-pRefStart]));
    PetscCall(PetscCalloc1(maxFields,&refPointFieldN[p-pRefStart]));
    PetscCall(DMPlexGetTransitiveClosure(refTree,parent,PETSC_TRUE,&closureSize,&closure));
    for (f = 0; f < maxFields; f++) {
      PetscInt cDof, cOff, numCols, r, i;

      if (f < numFields) {
        PetscCall(PetscSectionGetFieldDof(refConSec,p,f,&cDof));
        PetscCall(PetscSectionGetFieldOffset(refConSec,p,f,&cOff));
        PetscCall(PetscSectionGetFieldPointSyms(refSection,f,closureSize,closure,&perms,&flips));
      } else {
        PetscCall(PetscSectionGetDof(refConSec,p,&cDof));
        PetscCall(PetscSectionGetOffset(refConSec,p,&cOff));
        PetscCall(PetscSectionGetPointSyms(refSection,closureSize,closure,&perms,&flips));
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
          PetscCall(PetscSectionGetFieldDof(refSection,q,f,&aDof));
          PetscCall(PetscSectionGetFieldOffset(refSection,q,f,&aOff));
        }
        else {
          PetscCall(PetscSectionGetDof(refSection,q,&aDof));
          PetscCall(PetscSectionGetOffset(refSection,q,&aOff));
        }

        for (j = 0; j < aDof; j++) {
          cols[numCols++] = aOff + (perm ? perm[j] : j);
        }
      }
      refPointFieldN[p-pRefStart][f] = numCols;
      PetscCall(PetscMalloc1(cDof*numCols,&refPointFieldMats[p-pRefStart][f]));
      PetscCall(MatGetValues(refCmat,cDof,rows,numCols,cols,refPointFieldMats[p-pRefStart][f]));
      if (flips) {
        PetscInt colOff = 0;

        for (i = 0; i < closureSize; i++) {
          PetscInt          q = closure[2*i];
          PetscInt          aDof, aOff, j;
          const PetscScalar *flip = flips ? flips[i] : NULL;

          if (numFields) {
            PetscCall(PetscSectionGetFieldDof(refSection,q,f,&aDof));
            PetscCall(PetscSectionGetFieldOffset(refSection,q,f,&aOff));
          }
          else {
            PetscCall(PetscSectionGetDof(refSection,q,&aDof));
            PetscCall(PetscSectionGetOffset(refSection,q,&aOff));
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
        PetscCall(PetscSectionRestoreFieldPointSyms(refSection,f,closureSize,closure,&perms,&flips));
      } else {
        PetscCall(PetscSectionRestorePointSyms(refSection,closureSize,closure,&perms,&flips));
      }
    }
    PetscCall(DMPlexRestoreTransitiveClosure(refTree,parent,PETSC_TRUE,&closureSize,&closure));
  }
  *childrenMats = refPointFieldMats;
  *childrenN = refPointFieldN;
  PetscCall(ISRestoreIndices(refAnIS,&refAnchors));
  PetscCall(PetscFree(rows));
  PetscCall(PetscFree(cols));
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
  PetscCall(DMGetDS(refTree,&ds));
  PetscCall(PetscDSGetNumFields(ds,&numFields));
  maxFields = PetscMax(1,numFields);
  PetscCall(DMGetDefaultConstraints(refTree,&refConSec,NULL,NULL));
  PetscCall(PetscSectionGetChart(refConSec,&pRefStart,&pRefEnd));
  for (p = pRefStart; p < pRefEnd; p++) {
    PetscInt parent, pDof;

    PetscCall(DMPlexGetTreeParent(refTree,p,&parent,NULL));
    PetscCall(PetscSectionGetDof(refConSec,p,&pDof));
    if (!pDof || parent == p) continue;

    for (f = 0; f < maxFields; f++) {
      PetscInt cDof;

      if (numFields) {
        PetscCall(PetscSectionGetFieldDof(refConSec,p,f,&cDof));
      }
      else {
        PetscCall(PetscSectionGetDof(refConSec,p,&cDof));
      }

      PetscCall(PetscFree(refPointFieldMats[p - pRefStart][f]));
    }
    PetscCall(PetscFree(refPointFieldMats[p - pRefStart]));
    PetscCall(PetscFree(refPointFieldN[p - pRefStart]));
  }
  PetscCall(PetscFree(refPointFieldMats));
  PetscCall(PetscFree(refPointFieldN));
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
  PetscCall(DMGetDS(dm,&ds));
  PetscCall(PetscDSGetNumFields(ds,&numFields));
  maxFields = PetscMax(1,numFields);
  PetscCall(DMPlexGetReferenceTree(dm,&refTree));
  PetscCall(DMCopyDisc(dm,refTree));
  PetscCall(DMGetDefaultConstraints(refTree,&refConSec,&refCmat,NULL));
  PetscCall(DMPlexGetAnchors(refTree,&refAnSec,&refAnIS));
  PetscCall(DMPlexGetAnchors(dm,&anSec,&anIS));
  PetscCall(ISGetIndices(anIS,&anchors));
  PetscCall(PetscSectionGetChart(refConSec,&pRefStart,&pRefEnd));
  PetscCall(PetscSectionGetChart(conSec,&conStart,&conEnd));
  PetscCall(PetscSectionGetMaxDof(refConSec,&maxDof));
  PetscCall(PetscSectionGetMaxDof(refAnSec,&maxAnDof));
  PetscCall(PetscMalloc1(maxDof*maxDof*maxAnDof,&pointWork));

  /* step 1: get submats for every constrained point in the reference tree */
  PetscCall(DMPlexReferenceTreeGetChildrenMatrices(refTree,&refPointFieldMats,&refPointFieldN));

  /* step 2: compute the preorder */
  PetscCall(DMPlexGetChart(dm,&pStart,&pEnd));
  PetscCall(PetscMalloc2(pEnd-pStart,&perm,pEnd-pStart,&iperm));
  for (p = pStart; p < pEnd; p++) {
    perm[p - pStart] = p;
    iperm[p - pStart] = p-pStart;
  }
  for (p = 0; p < pEnd - pStart;) {
    PetscInt point = perm[p];
    PetscInt parent;

    PetscCall(DMPlexGetTreeParent(dm,point,&parent,NULL));
    if (parent == point) {
      p++;
    }
    else {
      PetscInt size, closureSize, *closure = NULL, i;

      PetscCall(DMPlexGetTransitiveClosure(dm,parent,PETSC_TRUE,&closureSize,&closure));
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
      PetscCall(DMPlexRestoreTransitiveClosure(dm,parent,PETSC_TRUE,&closureSize,&closure));
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

    PetscCall(MatGetRowIJ(cMat,0,PETSC_FALSE,PETSC_FALSE,&nRows,&ia,&ja,&done));
    PetscCheck(done,PetscObjectComm((PetscObject)cMat),PETSC_ERR_PLIB,"Could not get RowIJ of constraint matrix");
    nnz  = ia[nRows];
    /* malloc and then zero rows right before we fill them: this way valgrind
     * can tell if we are doing progressive fill in the wrong order */
    PetscCall(PetscMalloc1(nnz,&vals));
    for (p = 0; p < pEnd - pStart; p++) {
      PetscInt        parent, childid, closureSize, *closure = NULL;
      PetscInt        point = perm[p], pointDof;

      PetscCall(DMPlexGetTreeParent(dm,point,&parent,&childid));
      if ((point < conStart) || (point >= conEnd) || (parent == point)) continue;
      PetscCall(PetscSectionGetDof(conSec,point,&pointDof));
      if (!pointDof) continue;
      PetscCall(DMPlexGetTransitiveClosure(dm,parent,PETSC_TRUE,&closureSize,&closure));
      for (f = 0; f < maxFields; f++) {
        PetscInt cDof, cOff, numCols, numFillCols, i, r, matOffset, offset;
        PetscScalar *pointMat;
        const PetscInt    **perms;
        const PetscScalar **flips;

        if (numFields) {
          PetscCall(PetscSectionGetFieldDof(conSec,point,f,&cDof));
          PetscCall(PetscSectionGetFieldOffset(conSec,point,f,&cOff));
        }
        else {
          PetscCall(PetscSectionGetDof(conSec,point,&cDof));
          PetscCall(PetscSectionGetOffset(conSec,point,&cOff));
        }
        if (!cDof) continue;
        if (numFields) PetscCall(PetscSectionGetFieldPointSyms(section,f,closureSize,closure,&perms,&flips));
        else           PetscCall(PetscSectionGetPointSyms(section,closureSize,closure,&perms,&flips));

        /* make sure that every row for this point is the same size */
        if (PetscDefined(USE_DEBUG)) {
          for (r = 0; r < cDof; r++) {
            if (cDof > 1 && r) {
              PetscCheck((ia[cOff+r+1]-ia[cOff+r]) == (ia[cOff+r]-ia[cOff+r-1]),PETSC_COMM_SELF,PETSC_ERR_PLIB,"Two point rows have different nnz: %" PetscInt_FMT " vs. %" PetscInt_FMT, (ia[cOff+r+1]-ia[cOff+r]), (ia[cOff+r]-ia[cOff+r-1]));
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
            PetscCall(PetscSectionGetFieldDof(section,q,f,&aDof));
            PetscCall(PetscSectionGetFieldOffset(section,q,f,&aOff));
            if (q >= conStart && q < conEnd) {
              PetscCall(PetscSectionGetFieldDof(conSec,q,f,&qConDof));
              PetscCall(PetscSectionGetFieldOffset(conSec,q,f,&qConOff));
            }
          }
          else {
            PetscCall(PetscSectionGetDof(section,q,&aDof));
            PetscCall(PetscSectionGetOffset(section,q,&aOff));
            if (q >= conStart && q < conEnd) {
              PetscCall(PetscSectionGetDof(conSec,q,&qConDof));
              PetscCall(PetscSectionGetOffset(conSec,q,&qConOff));
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
              PetscCheck(k != numFillCols,PETSC_COMM_SELF,PETSC_ERR_PLIB,"No nonzero space for (%" PetscInt_FMT ", %" PetscInt_FMT ")", cOff, col);
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
            PetscCheck(k != numFillCols,PETSC_COMM_SELF,PETSC_ERR_PLIB,"No nonzero space for (%" PetscInt_FMT ", %" PetscInt_FMT ")", cOff, aOff);
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
          PetscCall(PetscSectionRestoreFieldPointSyms(section,f,closureSize,closure,&perms,&flips));
        } else {
          PetscCall(PetscSectionRestorePointSyms(section,closureSize,closure,&perms,&flips));
        }
      }
      PetscCall(DMPlexRestoreTransitiveClosure(dm,parent,PETSC_TRUE,&closureSize,&closure));
    }
    for (row = 0; row < nRows; row++) {
      PetscCall(MatSetValues(cMat,1,&row,ia[row+1]-ia[row],&ja[ia[row]],&vals[ia[row]],INSERT_VALUES));
    }
    PetscCall(MatRestoreRowIJ(cMat,0,PETSC_FALSE,PETSC_FALSE,&nRows,&ia,&ja,&done));
    PetscCheck(done,PetscObjectComm((PetscObject)cMat),PETSC_ERR_PLIB,"Could not restore RowIJ of constraint matrix");
    PetscCall(MatAssemblyBegin(cMat,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(cMat,MAT_FINAL_ASSEMBLY));
    PetscCall(PetscFree(vals));
  }

  /* clean up */
  PetscCall(ISRestoreIndices(anIS,&anchors));
  PetscCall(PetscFree2(perm,iperm));
  PetscCall(PetscFree(pointWork));
  PetscCall(DMPlexReferenceTreeRestoreChildrenMatrices(refTree,&refPointFieldMats,&refPointFieldN));
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
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm),&rank));
  PetscCall(DMGetDimension(dm,&dim));
  PetscCall(DMPlexCreate(PetscObjectComm((PetscObject)dm), ncdm));
  PetscCall(DMSetDimension(*ncdm,dim));

  PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
  PetscCall(PetscSectionCreate(PetscObjectComm((PetscObject)dm),&parentSection));
  PetscCall(DMPlexGetReferenceTree(dm,&K));
  if (rank == 0) {
    /* compute the new charts */
    PetscCall(PetscMalloc5(dim+1,&pNewCount,dim+1,&pNewStart,dim+1,&pNewEnd,dim+1,&pOldStart,dim+1,&pOldEnd));
    offset = 0;
    for (d = 0; d <= dim; d++) {
      PetscInt pOldCount, kStart, kEnd, k;

      pNewStart[d] = offset;
      PetscCall(DMPlexGetHeightStratum(dm,d,&pOldStart[d],&pOldEnd[d]));
      PetscCall(DMPlexGetHeightStratum(K,d,&kStart,&kEnd));
      pOldCount = pOldEnd[d] - pOldStart[d];
      /* adding the new points */
      pNewCount[d] = pOldCount + kEnd - kStart;
      if (!d) {
        /* removing the cell */
        pNewCount[d]--;
      }
      for (k = kStart; k < kEnd; k++) {
        PetscInt parent;
        PetscCall(DMPlexGetTreeParent(K,k,&parent,NULL));
        if (parent == k) {
          /* avoid double counting points that won't actually be new */
          pNewCount[d]--;
        }
      }
      pNewEnd[d] = pNewStart[d] + pNewCount[d];
      offset = pNewEnd[d];

    }
    PetscCheck(cell >= pOldStart[0] && cell < pOldEnd[0],PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"%" PetscInt_FMT " not in cell range [%" PetscInt_FMT ", %" PetscInt_FMT ")", cell, pOldStart[0], pOldEnd[0]);
    /* get the current closure of the cell that we are removing */
    PetscCall(DMPlexGetTransitiveClosure(dm,cell,PETSC_TRUE,&nc,&cellClosure));

    PetscCall(PetscMalloc1(pNewEnd[dim],&newConeSizes));
    {
      DMPolytopeType pct, qct;
      PetscInt kStart, kEnd, k, closureSizeK, *closureK = NULL, j;

      PetscCall(DMPlexGetChart(K,&kStart,&kEnd));
      PetscCall(PetscMalloc4(kEnd-kStart,&Kembedding,kEnd-kStart,&perm,kEnd-kStart,&iperm,kEnd-kStart,&preOrient));

      for (k = kStart; k < kEnd; k++) {
        perm[k - kStart] = k;
        iperm [k - kStart] = k - kStart;
        preOrient[k - kStart] = 0;
      }

      PetscCall(DMPlexGetTransitiveClosure(K,0,PETSC_TRUE,&closureSizeK,&closureK));
      for (j = 1; j < closureSizeK; j++) {
        PetscInt parentOrientA = closureK[2*j+1];
        PetscInt parentOrientB = cellClosure[2*j+1];
        PetscInt p, q;

        p = closureK[2*j];
        q = cellClosure[2*j];
        PetscCall(DMPlexGetCellType(K, p, &pct));
        PetscCall(DMPlexGetCellType(dm, q, &qct));
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

          PetscCall(DMPlexGetTreeChildren(K,p,&numChildren,&children));
          for (i = 0; i < numChildren; i++) {
            PetscInt kPerm, oPerm;

            k    = children[i];
            PetscCall(DMPlexReferenceTreeGetChildSymmetry(K,p,parentOrientA,0,k,parentOrientB,&oPerm,&kPerm));
            /* perm = what refTree position I'm in */
            perm[kPerm-kStart]      = k;
            /* iperm = who is at this position */
            iperm[k-kStart]         = kPerm-kStart;
            preOrient[kPerm-kStart] = oPerm;
          }
        }
      }
      PetscCall(DMPlexRestoreTransitiveClosure(K,0,PETSC_TRUE,&closureSizeK,&closureK));
    }
    PetscCall(PetscSectionSetChart(parentSection,0,pNewEnd[dim]));
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
        PetscCall(DMPlexGetConeSize(dm,p,&size));
        newConeSizes[offset++] = size;
        numNewCones += size;
      }

      PetscCall(DMPlexGetHeightStratum(K,d,&kStart,&kEnd));
      for (k = kStart; k < kEnd; k++) {
        PetscInt kParent;

        PetscCall(DMPlexGetTreeParent(K,k,&kParent,NULL));
        if (kParent != k) {
          Kembedding[k] = offset;
          PetscCall(DMPlexGetConeSize(K,k,&size));
          newConeSizes[offset++] = size;
          numNewCones += size;
          if (kParent != 0) {
            PetscCall(PetscSectionSetDof(parentSection,Kembedding[k],1));
          }
        }
      }
    }

    PetscCall(PetscSectionSetUp(parentSection));
    PetscCall(PetscSectionGetStorageSize(parentSection,&numPointsWithParents));
    PetscCall(PetscMalloc2(numNewCones,&newCones,numNewCones,&newOrientations));
    PetscCall(PetscMalloc2(numPointsWithParents,&parents,numPointsWithParents,&childIDs));

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
        PetscCall(DMPlexGetConeSize(dm,p,&size));
        PetscCall(DMPlexGetCone(dm,p,&cone));
        PetscCall(DMPlexGetConeOrientation(dm,p,&orientation));
        for (l = 0; l < size; l++) {
          newCones[offset]          = (cone[l] - pOldStart[d + 1]) + pNewStart[d + 1];
          newOrientations[offset++] = orientation[l];
        }
      }

      PetscCall(DMPlexGetHeightStratum(K,d,&kStart,&kEnd));
      for (k = kStart; k < kEnd; k++) {
        PetscInt kPerm = perm[k], kParent;
        PetscInt preO  = preOrient[k];

        PetscCall(DMPlexGetTreeParent(K,k,&kParent,NULL));
        if (kParent != k) {
          /* embed new cones */
          PetscCall(DMPlexGetConeSize(K,k,&size));
          PetscCall(DMPlexGetCone(K,kPerm,&cone));
          PetscCall(DMPlexGetConeOrientation(K,kPerm,&orientation));
          for (l = 0; l < size; l++) {
            PetscInt q, m = (preO >= 0) ? ((preO + l) % size) : ((size -(preO + 1) - l) % size);
            PetscInt newO, lSize, oTrue;
            DMPolytopeType ct = DM_NUM_POLYTOPES;

            q                         = iperm[cone[m]];
            newCones[offset]          = Kembedding[q];
            PetscCall(DMPlexGetConeSize(K,q,&lSize));
            if (lSize == 2) ct = DM_POLYTOPE_SEGMENT;
            else if (lSize == 4) ct = DM_POLYTOPE_QUADRILATERAL;
            oTrue                     = DMPolytopeConvertNewOrientation_Internal(ct, orientation[m]);
            oTrue                     = ((!lSize) || (preOrient[k] >= 0)) ? oTrue : -(oTrue + 2);
            newO                      = DihedralCompose(lSize,oTrue,preOrient[q]);
            newOrientations[offset++] = DMPolytopeConvertOldOrientation_Internal(ct, newO);
          }
          if (kParent != 0) {
            PetscInt newPoint = Kembedding[kParent];
            PetscCall(PetscSectionGetOffset(parentSection,Kembedding[k],&pOffset));
            parents[pOffset]  = newPoint;
            childIDs[pOffset] = k;
          }
        }
      }
    }

    PetscCall(PetscMalloc1(dim*(pNewEnd[dim]-pNewStart[dim]),&newVertexCoords));

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
        PetscCall(DMPlexGetHeightStratum(K,0,&kStart,&kEnd));
        for (k = kStart; k < kEnd; k++) {
          PetscCall(DMPlexComputeCellGeometryFEM(K, k, NULL, v0, J, NULL, &detJ));
          PetscCheck(detJ > 0.,PETSC_COMM_SELF,PETSC_ERR_PLIB,"reference tree cell %" PetscInt_FMT " has bad determinant",k);
        }
      }
      PetscCall(DMPlexComputeCellGeometryFEM(dm, cell, NULL, v0, J, NULL, &detJ));
      PetscCall(DMGetCoordinateSection(dm,&vSection));
      PetscCall(DMGetCoordinatesLocal(dm,&coords));
      PetscCall(VecGetArray(coords,&coordvals));
      for (v = pOldStart[dim]; v < pOldEnd[dim]; v++) {

        PetscCall(PetscSectionGetDof(vSection,v,&dof));
        PetscCall(PetscSectionGetOffset(vSection,v,&off));
        for (l = 0; l < dof; l++) {
          newVertexCoords[offset++] = coordvals[off + l];
        }
      }
      PetscCall(VecRestoreArray(coords,&coordvals));

      PetscCall(DMGetCoordinateSection(K,&vSection));
      PetscCall(DMGetCoordinatesLocal(K,&coords));
      PetscCall(VecGetArray(coords,&coordvals));
      PetscCall(DMPlexGetDepthStratum(K,0,&kStart,&kEnd));
      for (v = kStart; v < kEnd; v++) {
        PetscReal coord[3], newCoord[3];
        PetscInt  vPerm = perm[v];
        PetscInt  kParent;
        const PetscReal xi0[3] = {-1.,-1.,-1.};

        PetscCall(DMPlexGetTreeParent(K,v,&kParent,NULL));
        if (kParent != v) {
          /* this is a new vertex */
          PetscCall(PetscSectionGetOffset(vSection,vPerm,&off));
          for (l = 0; l < dim; ++l) coord[l] = PetscRealPart(coordvals[off+l]);
          CoordinatesRefToReal(dim, dim, xi0, v0, J, coord, newCoord);
          for (l = 0; l < dim; ++l) newVertexCoords[offset+l] = newCoord[l];
          offset += dim;
        }
      }
      PetscCall(VecRestoreArray(coords,&coordvals));
    }

    /* need to reverse the order of pNewCount: vertices first, cells last */
    for (d = 0; d < (dim + 1) / 2; d++) {
      PetscInt tmp;

      tmp = pNewCount[d];
      pNewCount[d] = pNewCount[dim - d];
      pNewCount[dim - d] = tmp;
    }

    PetscCall(DMPlexCreateFromDAG(*ncdm,dim,pNewCount,newConeSizes,newCones,newOrientations,newVertexCoords));
    PetscCall(DMPlexSetReferenceTree(*ncdm,K));
    PetscCall(DMPlexSetTree(*ncdm,parentSection,parents,childIDs));

    /* clean up */
    PetscCall(DMPlexRestoreTransitiveClosure(dm,cell,PETSC_TRUE,&nc,&cellClosure));
    PetscCall(PetscFree5(pNewCount,pNewStart,pNewEnd,pOldStart,pOldEnd));
    PetscCall(PetscFree(newConeSizes));
    PetscCall(PetscFree2(newCones,newOrientations));
    PetscCall(PetscFree(newVertexCoords));
    PetscCall(PetscFree2(parents,childIDs));
    PetscCall(PetscFree4(Kembedding,perm,iperm,preOrient));
  }
  else {
    PetscInt    p, counts[4];
    PetscInt    *coneSizes, *cones, *orientations;
    Vec         coordVec;
    PetscScalar *coords;

    for (d = 0; d <= dim; d++) {
      PetscInt dStart, dEnd;

      PetscCall(DMPlexGetDepthStratum(dm,d,&dStart,&dEnd));
      counts[d] = dEnd - dStart;
    }
    PetscCall(PetscMalloc1(pEnd-pStart,&coneSizes));
    for (p = pStart; p < pEnd; p++) {
      PetscCall(DMPlexGetConeSize(dm,p,&coneSizes[p-pStart]));
    }
    PetscCall(DMPlexGetCones(dm, &cones));
    PetscCall(DMPlexGetConeOrientations(dm, &orientations));
    PetscCall(DMGetCoordinatesLocal(dm,&coordVec));
    PetscCall(VecGetArray(coordVec,&coords));

    PetscCall(PetscSectionSetChart(parentSection,pStart,pEnd));
    PetscCall(PetscSectionSetUp(parentSection));
    PetscCall(DMPlexCreateFromDAG(*ncdm,dim,counts,coneSizes,cones,orientations,NULL));
    PetscCall(DMPlexSetReferenceTree(*ncdm,K));
    PetscCall(DMPlexSetTree(*ncdm,parentSection,NULL,NULL));
    PetscCall(VecRestoreArray(coordVec,&coords));
  }
  PetscCall(PetscSectionDestroy(&parentSection));

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
  PetscCall(DMPlexGetChart(coarse,&pStartC,&pEndC));
  PetscCall(DMPlexGetChart(fine,&pStartF,&pEndF));
  PetscCall(DMGetGlobalSection(fine,&globalFine));
  { /* winnow fine points that don't have global dofs out of the sf */
    PetscInt dof, cdof, numPointsWithDofs, offset, *pointsWithDofs, nleaves, l;
    const PetscInt *leaves;

    PetscCall(PetscSFGetGraph(coarseToFine,NULL,&nleaves,&leaves,NULL));
    for (l = 0, numPointsWithDofs = 0; l < nleaves; l++) {
      p = leaves ? leaves[l] : l;
      PetscCall(PetscSectionGetDof(globalFine,p,&dof));
      PetscCall(PetscSectionGetConstraintDof(globalFine,p,&cdof));
      if ((dof - cdof) > 0) {
        numPointsWithDofs++;
      }
    }
    PetscCall(PetscMalloc1(numPointsWithDofs,&pointsWithDofs));
    for (l = 0, offset = 0; l < nleaves; l++) {
      p = leaves ? leaves[l] : l;
      PetscCall(PetscSectionGetDof(globalFine,p,&dof));
      PetscCall(PetscSectionGetConstraintDof(globalFine,p,&cdof));
      if ((dof - cdof) > 0) {
        pointsWithDofs[offset++] = l;
      }
    }
    PetscCall(PetscSFCreateEmbeddedLeafSF(coarseToFine, numPointsWithDofs, pointsWithDofs, &coarseToFineEmbedded));
    PetscCall(PetscFree(pointsWithDofs));
  }
  /* communicate back to the coarse mesh which coarse points have children (that may require interpolation) */
  PetscCall(PetscMalloc1(pEndC-pStartC,&maxChildIds));
  for (p = pStartC; p < pEndC; p++) {
    maxChildIds[p - pStartC] = -2;
  }
  PetscCall(PetscSFReduceBegin(coarseToFineEmbedded,MPIU_INT,childIds,maxChildIds,MPIU_MAX));
  PetscCall(PetscSFReduceEnd(coarseToFineEmbedded,MPIU_INT,childIds,maxChildIds,MPIU_MAX));

  PetscCall(DMGetLocalSection(coarse,&localCoarse));
  PetscCall(DMGetGlobalSection(coarse,&globalCoarse));

  PetscCall(DMPlexGetAnchors(coarse,&aSec,&aIS));
  PetscCall(ISGetIndices(aIS,&anchors));
  PetscCall(PetscSectionGetChart(aSec,&aStart,&aEnd));

  PetscCall(DMGetDefaultConstraints(coarse,&cSec,&cMat,NULL));
  PetscCall(PetscSectionGetChart(cSec,&cStart,&cEnd));

  /* create sections that will send to children the indices and matrices they will need to construct the interpolator */
  PetscCall(PetscSectionCreate(PetscObjectComm((PetscObject)coarse),&rootIndicesSec));
  PetscCall(PetscSectionCreate(PetscObjectComm((PetscObject)coarse),&rootMatricesSec));
  PetscCall(PetscSectionSetChart(rootIndicesSec,pStartC,pEndC));
  PetscCall(PetscSectionSetChart(rootMatricesSec,pStartC,pEndC));
  PetscCall(PetscSectionGetNumFields(localCoarse,&numFields));
  maxFields = PetscMax(1,numFields);
  PetscCall(PetscMalloc7(maxFields+1,&offsets,maxFields+1,&offsetsCopy,maxFields+1,&newOffsets,maxFields+1,&newOffsetsCopy,maxFields+1,&rowOffsets,maxFields+1,&numD,maxFields+1,&numO));
  PetscCall(PetscMalloc2(maxFields+1,(PetscInt****)&perms,maxFields+1,(PetscScalar****)&flips));
  PetscCall(PetscMemzero((void *) perms, (maxFields+1) * sizeof(const PetscInt **)));
  PetscCall(PetscMemzero((void *) flips, (maxFields+1) * sizeof(const PetscScalar **)));

  for (p = pStartC; p < pEndC; p++) { /* count the sizes of the indices and matrices */
    PetscInt dof, matSize   = 0;
    PetscInt aDof           = 0;
    PetscInt cDof           = 0;
    PetscInt maxChildId     = maxChildIds[p - pStartC];
    PetscInt numRowIndices  = 0;
    PetscInt numColIndices  = 0;
    PetscInt f;

    PetscCall(PetscSectionGetDof(globalCoarse,p,&dof));
    if (dof < 0) {
      dof = -(dof + 1);
    }
    if (p >= aStart && p < aEnd) {
      PetscCall(PetscSectionGetDof(aSec,p,&aDof));
    }
    if (p >= cStart && p < cEnd) {
      PetscCall(PetscSectionGetDof(cSec,p,&cDof));
    }
    for (f = 0; f <= numFields; f++) offsets[f] = 0;
    for (f = 0; f <= numFields; f++) newOffsets[f] = 0;
    if (maxChildId >= 0) { /* this point has children (with dofs) that will need to be interpolated from the closure of p */
      PetscInt *closure = NULL, closureSize, cl;

      PetscCall(DMPlexGetTransitiveClosure(coarse,p,PETSC_TRUE,&closureSize,&closure));
      for (cl = 0; cl < closureSize; cl++) { /* get the closure */
        PetscInt c = closure[2 * cl], clDof;

        PetscCall(PetscSectionGetDof(localCoarse,c,&clDof));
        numRowIndices += clDof;
        for (f = 0; f < numFields; f++) {
          PetscCall(PetscSectionGetFieldDof(localCoarse,c,f,&clDof));
          offsets[f + 1] += clDof;
        }
      }
      for (f = 0; f < numFields; f++) {
        offsets[f + 1]   += offsets[f];
        newOffsets[f + 1] = offsets[f + 1];
      }
      /* get the number of indices needed and their field offsets */
      PetscCall(DMPlexAnchorsModifyMat(coarse,localCoarse,closureSize,numRowIndices,closure,NULL,NULL,NULL,&numColIndices,NULL,NULL,newOffsets,PETSC_FALSE));
      PetscCall(DMPlexRestoreTransitiveClosure(coarse,p,PETSC_TRUE,&closureSize,&closure));
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

        PetscCall(PetscSectionGetOffset(aSec,p,&aOff));
        for (f = 0; f < numFields; f++) {
          PetscInt fDof;

          PetscCall(PetscSectionGetFieldDof(localCoarse,p,f,&fDof));
          offsets[f+1] = fDof;
        }
        for (a = 0; a < aDof; a++) {
          PetscInt anchor = anchors[a + aOff], aLocalDof;

          PetscCall(PetscSectionGetDof(localCoarse,anchor,&aLocalDof));
          numColIndices += aLocalDof;
          for (f = 0; f < numFields; f++) {
            PetscInt fDof;

            PetscCall(PetscSectionGetFieldDof(localCoarse,anchor,f,&fDof));
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
    PetscCall(PetscSectionSetDof(rootIndicesSec,p,numColIndices ? numColIndices+2*numFields : 0));
    PetscCall(PetscSectionSetDof(rootMatricesSec,p,matSize));
  }
  PetscCall(PetscSectionSetUp(rootIndicesSec));
  PetscCall(PetscSectionSetUp(rootMatricesSec));
  {
    PetscInt numRootIndices, numRootMatrices;

    PetscCall(PetscSectionGetStorageSize(rootIndicesSec,&numRootIndices));
    PetscCall(PetscSectionGetStorageSize(rootMatricesSec,&numRootMatrices));
    PetscCall(PetscMalloc2(numRootIndices,&rootIndices,numRootMatrices,&rootMatrices));
    for (p = pStartC; p < pEndC; p++) {
      PetscInt    numRowIndices, numColIndices, matSize, dof;
      PetscInt    pIndOff, pMatOff, f;
      PetscInt    *pInd;
      PetscInt    maxChildId = maxChildIds[p - pStartC];
      PetscScalar *pMat = NULL;

      PetscCall(PetscSectionGetDof(rootIndicesSec,p,&numColIndices));
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
      PetscCall(PetscSectionGetOffset(rootIndicesSec,p,&pIndOff));
      pInd = &(rootIndices[pIndOff]);
      PetscCall(PetscSectionGetDof(rootMatricesSec,p,&matSize));
      if (matSize) {
        PetscCall(PetscSectionGetOffset(rootMatricesSec,p,&pMatOff));
        pMat = &rootMatrices[pMatOff];
      }
      PetscCall(PetscSectionGetDof(globalCoarse,p,&dof));
      if (dof < 0) {
        dof = -(dof + 1);
      }
      if (maxChildId >= 0) { /* build an identity matrix, apply matrix constraints on the right */
        PetscInt i, j;
        PetscInt numRowIndices = matSize / numColIndices;

        if (!numRowIndices) { /* don't need to calculate the mat, just the indices */
          PetscInt numIndices, *indices;
          PetscCall(DMPlexGetClosureIndices(coarse,localCoarse,globalCoarse,p,PETSC_TRUE,&numIndices,&indices,offsets,NULL));
          PetscCheck(numIndices == numColIndices,PETSC_COMM_SELF,PETSC_ERR_PLIB,"mismatching constraint indices calculations");
          for (i = 0; i < numColIndices; i++) {
            pInd[i] = indices[i];
          }
          for (i = 0; i < numFields; i++) {
            pInd[numColIndices + i]             = offsets[i+1];
            pInd[numColIndices + numFields + i] = offsets[i+1];
          }
          PetscCall(DMPlexRestoreClosureIndices(coarse,localCoarse,globalCoarse,p,PETSC_TRUE,&numIndices,&indices,offsets,NULL));
        }
        else {
          PetscInt closureSize, *closure = NULL, cl;
          PetscScalar *pMatIn, *pMatModified;
          PetscInt numPoints,*points;

          PetscCall(DMGetWorkArray(coarse,numRowIndices * numRowIndices,MPIU_SCALAR,&pMatIn));
          for (i = 0; i < numRowIndices; i++) { /* initialize to the identity */
            for (j = 0; j < numRowIndices; j++) {
              pMatIn[i * numRowIndices + j] = (i == j) ? 1. : 0.;
            }
          }
          PetscCall(DMPlexGetTransitiveClosure(coarse, p, PETSC_TRUE, &closureSize, &closure));
          for (f = 0; f < maxFields; f++) {
            if (numFields) PetscCall(PetscSectionGetFieldPointSyms(localCoarse,f,closureSize,closure,&perms[f],&flips[f]));
            else           PetscCall(PetscSectionGetPointSyms(localCoarse,closureSize,closure,&perms[f],&flips[f]));
          }
          if (numFields) {
            for (cl = 0; cl < closureSize; cl++) {
              PetscInt c = closure[2 * cl];

              for (f = 0; f < numFields; f++) {
                PetscInt fDof;

                PetscCall(PetscSectionGetFieldDof(localCoarse,c,f,&fDof));
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
          PetscCall(DMPlexAnchorsModifyMat(coarse,localCoarse,closureSize,numRowIndices,closure,perms,pMatIn,&numPoints,NULL,&points,&pMatModified,newOffsets,PETSC_FALSE));
          for (f = 0; f < maxFields; f++) {
            if (numFields) PetscCall(PetscSectionRestoreFieldPointSyms(localCoarse,f,closureSize,closure,&perms[f],&flips[f]));
            else           PetscCall(PetscSectionRestorePointSyms(localCoarse,closureSize,closure,&perms[f],&flips[f]));
          }
          for (f = 0; f < maxFields; f++) {
            if (numFields) PetscCall(PetscSectionGetFieldPointSyms(localCoarse,f,numPoints,points,&perms[f],&flips[f]));
            else           PetscCall(PetscSectionGetPointSyms(localCoarse,numPoints,points,&perms[f],&flips[f]));
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
          PetscCall(DMRestoreWorkArray(coarse,numRowIndices * numColIndices,MPIU_SCALAR,&pMatModified));
          PetscCall(DMPlexRestoreTransitiveClosure(coarse, p, PETSC_TRUE, &closureSize, &closure));
          PetscCall(DMRestoreWorkArray(coarse,numRowIndices * numColIndices,MPIU_SCALAR,&pMatIn));
          if (numFields) {
            for (f = 0; f < numFields; f++) {
              pInd[numColIndices + f]             = offsets[f+1];
              pInd[numColIndices + numFields + f] = newOffsets[f+1];
            }
            for (cl = 0; cl < numPoints; cl++) {
              PetscInt globalOff, c = points[2*cl];
              PetscCall(PetscSectionGetOffset(globalCoarse, c, &globalOff));
              PetscCall(DMPlexGetIndicesPointFields_Internal(localCoarse, PETSC_FALSE, c, globalOff < 0 ? -(globalOff+1) : globalOff, newOffsets, PETSC_FALSE, perms, cl, NULL, pInd));
            }
          } else {
            for (cl = 0; cl < numPoints; cl++) {
              PetscInt c = points[2*cl], globalOff;
              const PetscInt *perm = perms[0] ? perms[0][cl] : NULL;

              PetscCall(PetscSectionGetOffset(globalCoarse, c, &globalOff));
              PetscCall(DMPlexGetIndicesPoint_Internal(localCoarse, PETSC_FALSE, c, globalOff < 0 ? -(globalOff+1) : globalOff, newOffsets, PETSC_FALSE, perm, NULL, pInd));
            }
          }
          for (f = 0; f < maxFields; f++) {
            if (numFields) PetscCall(PetscSectionRestoreFieldPointSyms(localCoarse,f,numPoints,points,&perms[f],&flips[f]));
            else           PetscCall(PetscSectionRestorePointSyms(localCoarse,numPoints,points,&perms[f],&flips[f]));
          }
          PetscCall(DMRestoreWorkArray(coarse,numPoints,MPIU_SCALAR,&points));
        }
      }
      else if (matSize) {
        PetscInt cOff;
        PetscInt *rowIndices, *colIndices, a, aDof, aOff;

        numRowIndices = matSize / numColIndices;
        PetscCheck(numRowIndices == dof,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Miscounted dofs");
        PetscCall(DMGetWorkArray(coarse,numRowIndices,MPIU_INT,&rowIndices));
        PetscCall(DMGetWorkArray(coarse,numColIndices,MPIU_INT,&colIndices));
        PetscCall(PetscSectionGetOffset(cSec,p,&cOff));
        PetscCall(PetscSectionGetDof(aSec,p,&aDof));
        PetscCall(PetscSectionGetOffset(aSec,p,&aOff));
        if (numFields) {
          for (f = 0; f < numFields; f++) {
            PetscInt fDof;

            PetscCall(PetscSectionGetFieldDof(cSec,p,f,&fDof));
            offsets[f + 1] = fDof;
            for (a = 0; a < aDof; a++) {
              PetscInt anchor = anchors[a + aOff];
              PetscCall(PetscSectionGetFieldDof(localCoarse,anchor,f,&fDof));
              newOffsets[f + 1] += fDof;
            }
          }
          for (f = 0; f < numFields; f++) {
            offsets[f + 1]       += offsets[f];
            offsetsCopy[f + 1]    = offsets[f + 1];
            newOffsets[f + 1]    += newOffsets[f];
            newOffsetsCopy[f + 1] = newOffsets[f + 1];
          }
          PetscCall(DMPlexGetIndicesPointFields_Internal(cSec,PETSC_TRUE,p,cOff,offsetsCopy,PETSC_TRUE,NULL,-1, NULL,rowIndices));
          for (a = 0; a < aDof; a++) {
            PetscInt anchor = anchors[a + aOff], lOff;
            PetscCall(PetscSectionGetOffset(localCoarse,anchor,&lOff));
            PetscCall(DMPlexGetIndicesPointFields_Internal(localCoarse,PETSC_TRUE,anchor,lOff,newOffsetsCopy,PETSC_TRUE,NULL,-1, NULL,colIndices));
          }
        }
        else {
          PetscCall(DMPlexGetIndicesPoint_Internal(cSec,PETSC_TRUE,p,cOff,offsetsCopy,PETSC_TRUE,NULL, NULL,rowIndices));
          for (a = 0; a < aDof; a++) {
            PetscInt anchor = anchors[a + aOff], lOff;
            PetscCall(PetscSectionGetOffset(localCoarse,anchor,&lOff));
            PetscCall(DMPlexGetIndicesPoint_Internal(localCoarse,PETSC_TRUE,anchor,lOff,newOffsetsCopy,PETSC_TRUE,NULL, NULL,colIndices));
          }
        }
        if (numFields) {
          PetscInt count, a;

          for (f = 0, count = 0; f < numFields; f++) {
            PetscInt iSize = offsets[f + 1] - offsets[f];
            PetscInt jSize = newOffsets[f + 1] - newOffsets[f];
            PetscCall(MatGetValues(cMat,iSize,&rowIndices[offsets[f]],jSize,&colIndices[newOffsets[f]],&pMat[count]));
            count += iSize * jSize;
            pInd[numColIndices + f]             = offsets[f+1];
            pInd[numColIndices + numFields + f] = newOffsets[f+1];
          }
          for (a = 0; a < aDof; a++) {
            PetscInt anchor = anchors[a + aOff];
            PetscInt gOff;
            PetscCall(PetscSectionGetOffset(globalCoarse,anchor,&gOff));
            PetscCall(DMPlexGetIndicesPointFields_Internal(localCoarse,PETSC_FALSE,anchor,gOff < 0 ? -(gOff + 1) : gOff,newOffsets,PETSC_FALSE,NULL,-1, NULL,pInd));
          }
        }
        else {
          PetscInt a;
          PetscCall(MatGetValues(cMat,numRowIndices,rowIndices,numColIndices,colIndices,pMat));
          for (a = 0; a < aDof; a++) {
            PetscInt anchor = anchors[a + aOff];
            PetscInt gOff;
            PetscCall(PetscSectionGetOffset(globalCoarse,anchor,&gOff));
            PetscCall(DMPlexGetIndicesPoint_Internal(localCoarse,PETSC_FALSE,anchor,gOff < 0 ? -(gOff + 1) : gOff,newOffsets,PETSC_FALSE,NULL, NULL,pInd));
          }
        }
        PetscCall(DMRestoreWorkArray(coarse,numColIndices,MPIU_INT,&colIndices));
        PetscCall(DMRestoreWorkArray(coarse,numRowIndices,MPIU_INT,&rowIndices));
      }
      else {
        PetscInt gOff;

        PetscCall(PetscSectionGetOffset(globalCoarse,p,&gOff));
        if (numFields) {
          for (f = 0; f < numFields; f++) {
            PetscInt fDof;
            PetscCall(PetscSectionGetFieldDof(localCoarse,p,f,&fDof));
            offsets[f + 1] = fDof + offsets[f];
          }
          for (f = 0; f < numFields; f++) {
            pInd[numColIndices + f]             = offsets[f+1];
            pInd[numColIndices + numFields + f] = offsets[f+1];
          }
          PetscCall(DMPlexGetIndicesPointFields_Internal(localCoarse,PETSC_FALSE,p,gOff < 0 ? -(gOff + 1) : gOff,offsets,PETSC_FALSE,NULL,-1, NULL,pInd));
        } else {
          PetscCall(DMPlexGetIndicesPoint_Internal(localCoarse,PETSC_FALSE,p,gOff < 0 ? -(gOff + 1) : gOff,offsets,PETSC_FALSE,NULL, NULL,pInd));
        }
      }
    }
    PetscCall(PetscFree(maxChildIds));
  }
  {
    PetscSF  indicesSF, matricesSF;
    PetscInt *remoteOffsetsIndices, *remoteOffsetsMatrices, numLeafIndices, numLeafMatrices;

    PetscCall(PetscSectionCreate(PetscObjectComm((PetscObject)fine),&leafIndicesSec));
    PetscCall(PetscSectionCreate(PetscObjectComm((PetscObject)fine),&leafMatricesSec));
    PetscCall(PetscSFDistributeSection(coarseToFineEmbedded,rootIndicesSec,&remoteOffsetsIndices,leafIndicesSec));
    PetscCall(PetscSFDistributeSection(coarseToFineEmbedded,rootMatricesSec,&remoteOffsetsMatrices,leafMatricesSec));
    PetscCall(PetscSFCreateSectionSF(coarseToFineEmbedded,rootIndicesSec,remoteOffsetsIndices,leafIndicesSec,&indicesSF));
    PetscCall(PetscSFCreateSectionSF(coarseToFineEmbedded,rootMatricesSec,remoteOffsetsMatrices,leafMatricesSec,&matricesSF));
    PetscCall(PetscSFDestroy(&coarseToFineEmbedded));
    PetscCall(PetscFree(remoteOffsetsIndices));
    PetscCall(PetscFree(remoteOffsetsMatrices));
    PetscCall(PetscSectionGetStorageSize(leafIndicesSec,&numLeafIndices));
    PetscCall(PetscSectionGetStorageSize(leafMatricesSec,&numLeafMatrices));
    PetscCall(PetscMalloc2(numLeafIndices,&leafIndices,numLeafMatrices,&leafMatrices));
    PetscCall(PetscSFBcastBegin(indicesSF,MPIU_INT,rootIndices,leafIndices,MPI_REPLACE));
    PetscCall(PetscSFBcastBegin(matricesSF,MPIU_SCALAR,rootMatrices,leafMatrices,MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(indicesSF,MPIU_INT,rootIndices,leafIndices,MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(matricesSF,MPIU_SCALAR,rootMatrices,leafMatrices,MPI_REPLACE));
    PetscCall(PetscSFDestroy(&matricesSF));
    PetscCall(PetscSFDestroy(&indicesSF));
    PetscCall(PetscFree2(rootIndices,rootMatrices));
    PetscCall(PetscSectionDestroy(&rootIndicesSec));
    PetscCall(PetscSectionDestroy(&rootMatricesSec));
  }
  /* count to preallocate */
  PetscCall(DMGetLocalSection(fine,&localFine));
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

    PetscCall(PetscSectionGetConstrainedStorageSize(globalFine,&nGlobal));
    PetscCall(PetscCalloc2(nGlobal,&dnnz,nGlobal,&onnz));
    PetscCall(MatGetLayouts(mat,&rowMap,&colMap));
    PetscCall(PetscLayoutSetUp(rowMap));
    PetscCall(PetscLayoutSetUp(colMap));
    PetscCall(PetscLayoutGetRange(rowMap,&rowStart,&rowEnd));
    PetscCall(PetscLayoutGetRange(colMap,&colStart,&colEnd));
    PetscCall(PetscSectionGetMaxDof(localFine,&maxDof));
    PetscCall(PetscSectionGetChart(leafIndicesSec,&leafStart,&leafEnd));
    PetscCall(DMGetWorkArray(fine,maxDof,MPIU_INT,&rowIndices));
    for (p = leafStart; p < leafEnd; p++) {
      PetscInt    gDof, gcDof, gOff;
      PetscInt    numColIndices, pIndOff, *pInd;
      PetscInt    matSize;
      PetscInt    i;

      PetscCall(PetscSectionGetDof(globalFine,p,&gDof));
      PetscCall(PetscSectionGetConstraintDof(globalFine,p,&gcDof));
      if ((gDof - gcDof) <= 0) {
        continue;
      }
      PetscCall(PetscSectionGetOffset(globalFine,p,&gOff));
      PetscCheck(gOff >= 0,PETSC_COMM_SELF,PETSC_ERR_PLIB,"I though having global dofs meant a non-negative offset");
      PetscCheck(gOff >= rowStart && (gOff + gDof - gcDof) <= rowEnd,PETSC_COMM_SELF,PETSC_ERR_PLIB,"I thought the row map would constrain the global dofs");
      PetscCall(PetscSectionGetDof(leafIndicesSec,p,&numColIndices));
      PetscCall(PetscSectionGetOffset(leafIndicesSec,p,&pIndOff));
      numColIndices -= 2 * numFields;
      PetscCheck(numColIndices > 0,PETSC_COMM_SELF,PETSC_ERR_PLIB,"global fine dof with no dofs to interpolate from");
      pInd = &leafIndices[pIndOff];
      offsets[0]        = 0;
      offsetsCopy[0]    = 0;
      newOffsets[0]     = 0;
      newOffsetsCopy[0] = 0;
      if (numFields) {
        PetscInt f;
        for (f = 0; f < numFields; f++) {
          PetscInt rowDof;

          PetscCall(PetscSectionGetFieldDof(localFine,p,f,&rowDof));
          offsets[f + 1]        = offsets[f] + rowDof;
          offsetsCopy[f + 1]    = offsets[f + 1];
          newOffsets[f + 1]     = pInd[numColIndices + numFields + f];
          numD[f] = 0;
          numO[f] = 0;
        }
        PetscCall(DMPlexGetIndicesPointFields_Internal(localFine,PETSC_FALSE,p,gOff,offsetsCopy,PETSC_FALSE,NULL,-1, NULL,rowIndices));
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
        PetscCall(DMPlexGetIndicesPoint_Internal(localFine,PETSC_FALSE,p,gOff,offsetsCopy,PETSC_FALSE,NULL, NULL,rowIndices));
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
      PetscCall(PetscSectionGetDof(leafMatricesSec,p,&matSize));
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
                  PetscCheck(gIndFine >= rowStart && gIndFine < rowEnd,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Mismatched number of constrained dofs");
                  dnnz[gIndFine - rowStart] = 1;
                }
                else if (gIndCoarse >= 0) { /* remote */
                  PetscCheck(gIndFine >= rowStart && gIndFine < rowEnd,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Mismatched number of constrained dofs");
                  onnz[gIndFine - rowStart] = 1;
                }
                else { /* constrained */
                  PetscCheck(gIndFine < 0,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Mismatched number of constrained dofs");
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
                PetscCheck(gIndFine >= rowStart && gIndFine < rowEnd,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Mismatched number of constrained dofs");
                dnnz[gIndFine - rowStart] = 1;
              }
              else if (gIndCoarse >= 0) { /* remote */
                PetscCheck(gIndFine >= rowStart && gIndFine < rowEnd,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Mismatched number of constrained dofs");
                onnz[gIndFine - rowStart] = 1;
              }
              else { /* constrained */
                PetscCheck(gIndFine < 0,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Mismatched number of constrained dofs");
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
                  PetscCheck(gIndFine >= rowStart && gIndFine < rowEnd,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Mismatched number of constrained dofs");
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
                PetscCheck(gIndFine >= rowStart && gIndFine < rowEnd,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Mismatched number of constrained dofs");
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
                PetscCheck(gIndFine >= rowStart && gIndFine < rowEnd,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Mismatched number of constrained dofs");
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
              PetscCheck(gIndFine >= rowStart && gIndFine < rowEnd,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Mismatched number of constrained dofs");
              dnnz[gIndFine - rowStart] = numD[0];
              onnz[gIndFine - rowStart] = numO[0];
            }
          }
        }
      }
    }
    PetscCall(MatXAIJSetPreallocation(mat,1,dnnz,onnz,NULL,NULL));
    PetscCall(PetscFree2(dnnz,onnz));

    PetscCall(DMPlexGetReferenceTree(fine,&refTree));
    PetscCall(DMPlexReferenceTreeGetChildrenMatrices(refTree,&refPointFieldMats,&refPointFieldN));
    PetscCall(DMGetDefaultConstraints(refTree,&refConSec,NULL,NULL));
    PetscCall(DMPlexGetAnchors(refTree,&refAnSec,NULL));
    PetscCall(PetscSectionGetChart(refConSec,&pRefStart,&pRefEnd));
    PetscCall(PetscSectionGetMaxDof(refConSec,&maxConDof));
    PetscCall(PetscSectionGetMaxDof(leafIndicesSec,&maxColumns));
    PetscCall(PetscMalloc1(maxConDof*maxColumns,&pointWork));
    for (p = leafStart; p < leafEnd; p++) {
      PetscInt gDof, gcDof, gOff;
      PetscInt numColIndices, pIndOff, *pInd;
      PetscInt matSize;
      PetscInt childId;

      PetscCall(PetscSectionGetDof(globalFine,p,&gDof));
      PetscCall(PetscSectionGetConstraintDof(globalFine,p,&gcDof));
      if ((gDof - gcDof) <= 0) {
        continue;
      }
      childId = childIds[p-pStartF];
      PetscCall(PetscSectionGetOffset(globalFine,p,&gOff));
      PetscCall(PetscSectionGetDof(leafIndicesSec,p,&numColIndices));
      PetscCall(PetscSectionGetOffset(leafIndicesSec,p,&pIndOff));
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

          PetscCall(PetscSectionGetFieldDof(localFine,p,f,&rowDof));
          offsets[f + 1]     = offsets[f] + rowDof;
          offsetsCopy[f + 1] = offsets[f + 1];
          rowOffsets[f + 1]  = pInd[numColIndices + f];
          newOffsets[f + 1]  = pInd[numColIndices + numFields + f];
        }
        PetscCall(DMPlexGetIndicesPointFields_Internal(localFine,PETSC_FALSE,p,gOff,offsetsCopy,PETSC_FALSE,NULL,-1, NULL,rowIndices));
      }
      else {
        PetscCall(DMPlexGetIndicesPoint_Internal(localFine,PETSC_FALSE,p,gOff,offsetsCopy,PETSC_FALSE,NULL, NULL,rowIndices));
      }
      PetscCall(PetscSectionGetDof(leafMatricesSec,p,&matSize));
      if (!matSize) { /* incoming matrix is identity */
        if (childId < 0) { /* no child interpolation: scatter */
          if (numFields) {
            PetscInt f;
            for (f = 0; f < numFields; f++) {
              PetscInt numRows = offsets[f+1] - offsets[f], row;
              for (row = 0; row < numRows; row++) {
                PetscCall(MatSetValue(mat,rowIndices[offsets[f]+row],pInd[newOffsets[f]+row],1.,INSERT_VALUES));
              }
            }
          }
          else {
            PetscInt numRows = gDof, row;
            for (row = 0; row < numRows; row++) {
              PetscCall(MatSetValue(mat,rowIndices[row],pInd[row],1.,INSERT_VALUES));
            }
          }
        }
        else { /* interpolate from all */
          if (numFields) {
            PetscInt f;
            for (f = 0; f < numFields; f++) {
              PetscInt numRows = offsets[f+1] - offsets[f];
              PetscInt numCols = newOffsets[f + 1] - newOffsets[f];
              PetscCall(MatSetValues(mat,numRows,&rowIndices[offsets[f]],numCols,&pInd[newOffsets[f]],refPointFieldMats[childId - pRefStart][f],INSERT_VALUES));
            }
          }
          else {
            PetscCall(MatSetValues(mat,gDof,rowIndices,numColIndices,pInd,refPointFieldMats[childId - pRefStart][0],INSERT_VALUES));
          }
        }
      }
      else { /* interpolate from all */
        PetscInt    pMatOff;
        PetscScalar *pMat;

        PetscCall(PetscSectionGetOffset(leafMatricesSec,p,&pMatOff));
        pMat = &leafMatrices[pMatOff];
        if (childId < 0) { /* copy the incoming matrix */
          if (numFields) {
            PetscInt f, count;
            for (f = 0, count = 0; f < numFields; f++) {
              PetscInt numRows = offsets[f+1]-offsets[f];
              PetscInt numCols = newOffsets[f+1]-newOffsets[f];
              PetscInt numInRows = rowOffsets[f+1]-rowOffsets[f];
              PetscScalar *inMat = &pMat[count];

              PetscCall(MatSetValues(mat,numRows,&rowIndices[offsets[f]],numCols,&pInd[newOffsets[f]],inMat,INSERT_VALUES));
              count += numCols * numInRows;
            }
          }
          else {
            PetscCall(MatSetValues(mat,gDof,rowIndices,numColIndices,pInd,pMat,INSERT_VALUES));
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
              PetscCheck(refPointFieldN[childId - pRefStart][f] == numInRows,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Point constraint matrix multiply dimension mismatch");
              for (i = 0; i < numRows; i++) {
                for (j = 0; j < numCols; j++) {
                  PetscScalar val = 0.;
                  for (k = 0; k < numInRows; k++) {
                    val += refPointFieldMats[childId - pRefStart][f][i * numInRows + k] * inMat[k * numCols + j];
                  }
                  pointWork[i * numCols + j] = val;
                }
              }
              PetscCall(MatSetValues(mat,numRows,&rowIndices[offsets[f]],numCols,&pInd[newOffsets[f]],pointWork,INSERT_VALUES));
              count += numCols * numInRows;
            }
          }
          else { /* every dof gets a full row */
            PetscInt numRows   = gDof;
            PetscInt numCols   = numColIndices;
            PetscInt numInRows = matSize / numColIndices;
            PetscInt i, j, k;
            PetscCheck(refPointFieldN[childId - pRefStart][0] == numInRows,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Point constraint matrix multiply dimension mismatch");
            for (i = 0; i < numRows; i++) {
              for (j = 0; j < numCols; j++) {
                PetscScalar val = 0.;
                for (k = 0; k < numInRows; k++) {
                  val += refPointFieldMats[childId - pRefStart][0][i * numInRows + k] * pMat[k * numCols + j];
                }
                pointWork[i * numCols + j] = val;
              }
            }
            PetscCall(MatSetValues(mat,numRows,rowIndices,numCols,pInd,pointWork,INSERT_VALUES));
          }
        }
      }
    }
    PetscCall(DMPlexReferenceTreeRestoreChildrenMatrices(refTree,&refPointFieldMats,&refPointFieldN));
    PetscCall(DMRestoreWorkArray(fine,maxDof,MPIU_INT,&rowIndices));
    PetscCall(PetscFree(pointWork));
  }
  PetscCall(MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY));
  PetscCall(PetscSectionDestroy(&leafIndicesSec));
  PetscCall(PetscSectionDestroy(&leafMatricesSec));
  PetscCall(PetscFree2(leafIndices,leafMatrices));
  PetscCall(PetscFree2(*(PetscInt****)&perms,*(PetscScalar****)&flips));
  PetscCall(PetscFree7(offsets,offsetsCopy,newOffsets,newOffsetsCopy,rowOffsets,numD,numO));
  PetscCall(ISRestoreIndices(aIS,&anchors));
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
  PetscCall(DMGetLocalSection(refTree,&section));
  PetscCall(DMGetDimension(refTree, &dim));
  PetscCall(PetscMalloc6(dim,&v0,dim,&v0parent,dim,&vtmp,dim*dim,&J,dim*dim,&Jparent,dim*dim,&invJ));
  PetscCall(PetscMalloc2(dim,&pointScalar,dim,&pointRef));
  PetscCall(DMGetDS(refTree,&ds));
  PetscCall(PetscDSGetNumFields(ds,&numFields));
  PetscCall(PetscSectionGetNumFields(section,&numSecFields));
  PetscCall(DMGetLabel(refTree,"canonical",&canonical));
  PetscCall(DMGetLabel(refTree,"depth",&depth));
  PetscCall(DMGetDefaultConstraints(refTree,&cSection,&cMat,NULL));
  PetscCall(DMPlexGetChart(refTree, &pStart, &pEnd));
  PetscCall(DMPlexGetHeightStratum(refTree, 0, &cStart, &cEnd));
  PetscCall(MatGetSize(cMat,&n,&m)); /* the injector has transpose sizes from the constraint matrix */
  /* Step 1: compute non-zero pattern.  A proper subset of constraint matrix non-zero */
  PetscCall(PetscCalloc1(m,&nnz));
  for (p = pStart; p < pEnd; p++) { /* a point will have non-zeros if it is canonical, it has dofs, and its children have dofs */
    const PetscInt *children;
    PetscInt numChildren;
    PetscInt i, numChildDof, numSelfDof;

    if (canonical) {
      PetscInt pCanonical;
      PetscCall(DMLabelGetValue(canonical,p,&pCanonical));
      if (p != pCanonical) continue;
    }
    PetscCall(DMPlexGetTreeChildren(refTree,p,&numChildren,&children));
    if (!numChildren) continue;
    for (i = 0, numChildDof = 0; i < numChildren; i++) {
      PetscInt child = children[i];
      PetscInt dof;

      PetscCall(PetscSectionGetDof(section,child,&dof));
      numChildDof += dof;
    }
    PetscCall(PetscSectionGetDof(section,p,&numSelfDof));
    if (!numChildDof || !numSelfDof) continue;
    for (f = 0; f < numFields; f++) {
      PetscInt selfOff;

      if (numSecFields) { /* count the dofs for just this field */
        for (i = 0, numChildDof = 0; i < numChildren; i++) {
          PetscInt child = children[i];
          PetscInt dof;

          PetscCall(PetscSectionGetFieldDof(section,child,f,&dof));
          numChildDof += dof;
        }
        PetscCall(PetscSectionGetFieldDof(section,p,f,&numSelfDof));
        PetscCall(PetscSectionGetFieldOffset(section,p,f,&selfOff));
      }
      else {
        PetscCall(PetscSectionGetOffset(section,p,&selfOff));
      }
      for (i = 0; i < numSelfDof; i++) {
        nnz[selfOff + i] = numChildDof;
      }
    }
  }
  PetscCall(MatCreateAIJ(PETSC_COMM_SELF,m,n,m,n,-1,nnz,-1,NULL,&mat));
  PetscCall(PetscFree(nnz));
  /* Setp 2: compute entries */
  for (p = pStart; p < pEnd; p++) {
    const PetscInt *children;
    PetscInt numChildren;
    PetscInt i, numChildDof, numSelfDof;

    /* same conditions about when entries occur */
    if (canonical) {
      PetscInt pCanonical;
      PetscCall(DMLabelGetValue(canonical,p,&pCanonical));
      if (p != pCanonical) continue;
    }
    PetscCall(DMPlexGetTreeChildren(refTree,p,&numChildren,&children));
    if (!numChildren) continue;
    for (i = 0, numChildDof = 0; i < numChildren; i++) {
      PetscInt child = children[i];
      PetscInt dof;

      PetscCall(PetscSectionGetDof(section,child,&dof));
      numChildDof += dof;
    }
    PetscCall(PetscSectionGetDof(section,p,&numSelfDof));
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

          PetscCall(PetscSectionGetFieldDof(section,child,f,&dof));
          numChildDof += dof;
        }
        PetscCall(PetscSectionGetFieldDof(section,p,f,&numSelfDof));
        PetscCall(PetscSectionGetFieldOffset(section,p,f,&selfOff));
      }
      else {
        PetscCall(PetscSectionGetOffset(section,p,&selfOff));
      }

      /* find a cell whose closure contains p */
      if (p >= cStart && p < cEnd) {
        parentCell = p;
      }
      else {
        PetscInt *star = NULL;
        PetscInt numStar;

        parentCell = -1;
        PetscCall(DMPlexGetTransitiveClosure(refTree,p,PETSC_FALSE,&numStar,&star));
        for (i = numStar - 1; i >= 0; i--) {
          PetscInt c = star[2 * i];

          if (c >= cStart && c < cEnd) {
            parentCell = c;
            break;
          }
        }
        PetscCall(DMPlexRestoreTransitiveClosure(refTree,p,PETSC_FALSE,&numStar,&star));
      }
      /* determine the offset of p's shape functions within parentCell's shape functions */
      PetscCall(PetscDSGetDiscretization(ds,f,&disc));
      PetscCall(PetscObjectGetClassId(disc,&classId));
      if (classId == PETSCFE_CLASSID) {
        PetscCall(PetscFEGetDualSpace((PetscFE)disc,&dsp));
      }
      else if (classId == PETSCFV_CLASSID) {
        PetscCall(PetscFVGetDualSpace((PetscFV)disc,&dsp));
      }
      else {
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unsupported discretization object");
      }
      PetscCall(PetscDualSpaceGetNumDof(dsp,&depthNumDof));
      PetscCall(PetscDualSpaceGetNumComponents(dsp,&Nc));
      {
        PetscInt *closure = NULL;
        PetscInt numClosure;

        PetscCall(DMPlexGetTransitiveClosure(refTree,parentCell,PETSC_TRUE,&numClosure,&closure));
        for (i = 0, pI = -1, cellShapeOff = 0; i < numClosure; i++) {
          PetscInt point = closure[2 * i], pointDepth;

          pO = closure[2 * i + 1];
          if (point == p) {
            pI = i;
            break;
          }
          PetscCall(DMLabelGetValue(depth,point,&pointDepth));
          cellShapeOff += depthNumDof[pointDepth];
        }
        PetscCall(DMPlexRestoreTransitiveClosure(refTree,parentCell,PETSC_TRUE,&numClosure,&closure));
      }

      PetscCall(DMGetWorkArray(refTree, numSelfDof * numChildDof, MPIU_SCALAR,&pointMat));
      PetscCall(DMGetWorkArray(refTree, numSelfDof + numChildDof, MPIU_INT,&matRows));
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
            PetscCall(PetscSectionGetFieldDof(cSection,child,f,&dof));
            PetscCall(PetscSectionGetFieldOffset(cSection,child,f,&off));
          }
          else {
            PetscCall(PetscSectionGetDof(cSection,child,&dof));
            PetscCall(PetscSectionGetOffset(cSection,child,&off));
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

        PetscCall(PetscFEGetDualSpace(fe,&dsp));
        PetscCall(PetscDualSpaceGetDimension(dsp,&fSize));
        PetscCall(PetscDualSpaceGetSymmetries(dsp, &perms, &flips));
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

          PetscCall(PetscDualSpaceGetFunctional(dsp,parentCellShapeDof,&q));
          PetscCall(PetscQuadratureGetData(q,&dim,&thisNc,&numPoints,&points,&weights));
          PetscCheck(thisNc == Nc,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Functional dim %" PetscInt_FMT " does not much basis dim %" PetscInt_FMT,thisNc,Nc);
          PetscCall(PetscFECreateTabulation(fe,1,numPoints,points,0,&Tparent)); /* I'm expecting a nodal basis: weights[:]' * Bparent[:,cellShapeDof] = 1. */
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

              PetscCall(DMPlexGetTransitiveClosure(refTree,child,PETSC_FALSE,&numStar,&star));
              for (s = numStar - 1; s >= 0; s--) {
                PetscInt c = star[2 * s];

                if (c < cStart || c >= cEnd) continue;
                PetscCall(DMPlexLocatePoint_Internal(refTree,dim,point,c,&childCell));
                if (childCell >= 0) break;
              }
              PetscCall(DMPlexRestoreTransitiveClosure(refTree,child,PETSC_FALSE,&numStar,&star));
              if (childCell >= 0) break;
            }
            PetscCheck(childCell >= 0,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Could not locate quadrature point");
            PetscCall(DMPlexComputeCellGeometryFEM(refTree, childCell, NULL, v0, J, invJ, &detJ));
            PetscCall(DMPlexComputeCellGeometryFEM(refTree, parentCell, NULL, v0parent, Jparent, NULL, &detJparent));
            CoordinatesRefToReal(dim, dim, xi0, v0parent, Jparent, pointReal, vtmp);
            CoordinatesRealToRef(dim, dim, xi0, v0, invJ, vtmp, pointRef);

            PetscCall(PetscFECreateTabulation(fe,1,1,pointRef,0,&Tchild));
            PetscCall(DMPlexGetTransitiveClosure(refTree,childCell,PETSC_TRUE,&numClosure,&closure));
            for (k = 0, pointMatOff = 0; k < numChildren; k++) { /* point is located in cell => child dofs support at point are in closure of cell */
              PetscInt child = children[k], childDepth, childDof, childO = PETSC_MIN_INT;
              PetscInt l;
              const PetscInt *cperms;

              PetscCall(DMLabelGetValue(depth,child,&childDepth));
              childDof = depthNumDof[childDepth];
              for (l = 0, cI = -1, childCellShapeOff = 0; l < numClosure; l++) {
                PetscInt point = closure[2 * l];
                PetscInt pointDepth;

                childO = closure[2 * l + 1];
                if (point == child) {
                  cI = l;
                  break;
                }
                PetscCall(DMLabelGetValue(depth,point,&pointDepth));
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
            PetscCall(DMPlexRestoreTransitiveClosure(refTree,childCell,PETSC_TRUE,&numClosure,&closure));
            PetscCall(PetscTabulationDestroy(&Tchild));
          }
          PetscCall(PetscTabulationDestroy(&Tparent));
        }
      }
      else { /* just the volume-weighted averages of the children */
        PetscReal parentVol;
        PetscInt  childCell;

        PetscCall(DMPlexComputeCellGeometryFVM(refTree, p, &parentVol, NULL, NULL));
        for (i = 0, childCell = 0; i < numChildren; i++) {
          PetscInt  child = children[i], j;
          PetscReal childVol;

          if (child < cStart || child >= cEnd) continue;
          PetscCall(DMPlexComputeCellGeometryFVM(refTree, child, &childVol, NULL, NULL));
          for (j = 0; j < Nc; j++) {
            pointMat[j * numChildDof + Nc * childCell + j] = childVol / parentVol;
          }
          childCell++;
        }
      }
      /* Insert pointMat into mat */
      PetscCall(MatSetValues(mat,numSelfDof,matRows,numChildDof,matCols,pointMat,INSERT_VALUES));
      PetscCall(DMRestoreWorkArray(refTree, numSelfDof + numChildDof, MPIU_INT,&matRows));
      PetscCall(DMRestoreWorkArray(refTree, numSelfDof * numChildDof, MPIU_SCALAR,&pointMat));
    }
  }
  PetscCall(PetscFree6(v0,v0parent,vtmp,J,Jparent,invJ));
  PetscCall(PetscFree2(pointScalar,pointRef));
  PetscCall(MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY));
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
  PetscCall(DMGetDS(refTree,&ds));
  PetscCall(PetscDSGetNumFields(ds,&numFields));
  PetscCall(DMGetDefaultConstraints(refTree,&refConSec,NULL,NULL));
  PetscCall(DMGetLocalSection(refTree,&refSection));
  PetscCall(PetscSectionGetChart(refConSec,&pRefStart,&pRefEnd));
  PetscCall(PetscMalloc1(pRefEnd-pRefStart,&refPointFieldMats));
  PetscCall(PetscSectionGetMaxDof(refConSec,&maxDof));
  PetscCall(PetscMalloc1(maxDof,&rows));
  PetscCall(PetscMalloc1(maxDof*maxDof,&cols));
  for (p = pRefStart; p < pRefEnd; p++) {
    PetscInt parent, pDof, parentDof;

    PetscCall(DMPlexGetTreeParent(refTree,p,&parent,NULL));
    PetscCall(PetscSectionGetDof(refConSec,p,&pDof));
    PetscCall(PetscSectionGetDof(refSection,parent,&parentDof));
    if (!pDof || !parentDof || parent == p) continue;

    PetscCall(PetscMalloc1(numFields,&refPointFieldMats[p-pRefStart]));
    for (f = 0; f < numFields; f++) {
      PetscInt cDof, cOff, numCols, r;

      if (numFields > 1) {
        PetscCall(PetscSectionGetFieldDof(refConSec,p,f,&cDof));
        PetscCall(PetscSectionGetFieldOffset(refConSec,p,f,&cOff));
      }
      else {
        PetscCall(PetscSectionGetDof(refConSec,p,&cDof));
        PetscCall(PetscSectionGetOffset(refConSec,p,&cOff));
      }

      for (r = 0; r < cDof; r++) {
        rows[r] = cOff + r;
      }
      numCols = 0;
      {
        PetscInt aDof, aOff, j;

        if (numFields > 1) {
          PetscCall(PetscSectionGetFieldDof(refSection,parent,f,&aDof));
          PetscCall(PetscSectionGetFieldOffset(refSection,parent,f,&aOff));
        }
        else {
          PetscCall(PetscSectionGetDof(refSection,parent,&aDof));
          PetscCall(PetscSectionGetOffset(refSection,parent,&aOff));
        }

        for (j = 0; j < aDof; j++) {
          cols[numCols++] = aOff + j;
        }
      }
      PetscCall(PetscMalloc1(cDof*numCols,&refPointFieldMats[p-pRefStart][f]));
      /* transpose of constraint matrix */
      PetscCall(MatGetValues(inj,numCols,cols,cDof,rows,refPointFieldMats[p-pRefStart][f]));
    }
  }
  *childrenMats = refPointFieldMats;
  PetscCall(PetscFree(rows));
  PetscCall(PetscFree(cols));
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
  PetscCall(DMGetDS(refTree,&ds));
  PetscCall(DMGetLocalSection(refTree,&refSection));
  PetscCall(PetscDSGetNumFields(ds,&numFields));
  PetscCall(DMGetDefaultConstraints(refTree,&refConSec,NULL,NULL));
  PetscCall(PetscSectionGetChart(refConSec,&pRefStart,&pRefEnd));
  for (p = pRefStart; p < pRefEnd; p++) {
    PetscInt parent, pDof, parentDof;

    PetscCall(DMPlexGetTreeParent(refTree,p,&parent,NULL));
    PetscCall(PetscSectionGetDof(refConSec,p,&pDof));
    PetscCall(PetscSectionGetDof(refSection,parent,&parentDof));
    if (!pDof || !parentDof || parent == p) continue;

    for (f = 0; f < numFields; f++) {
      PetscInt cDof;

      if (numFields > 1) {
        PetscCall(PetscSectionGetFieldDof(refConSec,p,f,&cDof));
      }
      else {
        PetscCall(PetscSectionGetDof(refConSec,p,&cDof));
      }

      PetscCall(PetscFree(refPointFieldMats[p - pRefStart][f]));
    }
    PetscCall(PetscFree(refPointFieldMats[p - pRefStart]));
  }
  PetscCall(PetscFree(refPointFieldMats));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexReferenceTreeGetInjector(DM refTree,Mat *injRef)
{
  Mat            cMatRef;
  PetscObject    injRefObj;

  PetscFunctionBegin;
  PetscCall(DMGetDefaultConstraints(refTree,NULL,&cMatRef,NULL));
  PetscCall(PetscObjectQuery((PetscObject)cMatRef,"DMPlexComputeInjectorTree_refTree",&injRefObj));
  *injRef = (Mat) injRefObj;
  if (!*injRef) {
    PetscCall(DMPlexComputeInjectorReferenceTree(refTree,injRef));
    PetscCall(PetscObjectCompose((PetscObject)cMatRef,"DMPlexComputeInjectorTree_refTree",(PetscObject)*injRef));
    /* there is now a reference in cMatRef, which should be the only one for symmetry with the above case */
    PetscCall(PetscObjectDereference((PetscObject)*injRef));
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
  PetscCall(DMPlexGetChart(coarse,&pStartC,&pEndC));
  PetscCall(DMPlexGetChart(fine,&pStartF,&pEndF));
  PetscCall(DMGetLocalSection(fine,&localFine));
  PetscCall(DMGetGlobalSection(fine,&globalFine));
  PetscCall(PetscSectionCreate(PetscObjectComm((PetscObject)fine),&leafIndicesSec));
  PetscCall(PetscSectionSetChart(leafIndicesSec,pStartF, pEndF));
  PetscCall(PetscSectionGetMaxDof(localFine,&maxDof));
  { /* winnow fine points that don't have global dofs out of the sf */
    PetscInt l, nleaves, dof, cdof, numPointsWithDofs, offset, *pointsWithDofs, numIndices;
    const PetscInt *leaves;

    PetscCall(PetscSFGetGraph(coarseToFine,NULL,&nleaves,&leaves,NULL));
    for (l = 0, numPointsWithDofs = 0; l < nleaves; l++) {
      p    = leaves ? leaves[l] : l;
      PetscCall(PetscSectionGetDof(globalFine,p,&dof));
      PetscCall(PetscSectionGetConstraintDof(globalFine,p,&cdof));
      if ((dof - cdof) > 0) {
        numPointsWithDofs++;

        PetscCall(PetscSectionGetDof(localFine,p,&dof));
        PetscCall(PetscSectionSetDof(leafIndicesSec,p,dof + 1));
      }
    }
    PetscCall(PetscMalloc1(numPointsWithDofs,&pointsWithDofs));
    PetscCall(PetscSectionSetUp(leafIndicesSec));
    PetscCall(PetscSectionGetStorageSize(leafIndicesSec,&numIndices));
    PetscCall(PetscMalloc1(gatheredIndices ? numIndices : (maxDof + 1),&leafInds));
    if (gatheredValues)  PetscCall(PetscMalloc1(numIndices,&leafVals));
    for (l = 0, offset = 0; l < nleaves; l++) {
      p    = leaves ? leaves[l] : l;
      PetscCall(PetscSectionGetDof(globalFine,p,&dof));
      PetscCall(PetscSectionGetConstraintDof(globalFine,p,&cdof));
      if ((dof - cdof) > 0) {
        PetscInt    off, gOff;
        PetscInt    *pInd;
        PetscScalar *pVal = NULL;

        pointsWithDofs[offset++] = l;

        PetscCall(PetscSectionGetOffset(leafIndicesSec,p,&off));

        pInd = gatheredIndices ? (&leafInds[off + 1]) : leafInds;
        if (gatheredValues) {
          PetscInt i;

          pVal = &leafVals[off + 1];
          for (i = 0; i < dof; i++) pVal[i] = 0.;
        }
        PetscCall(PetscSectionGetOffset(globalFine,p,&gOff));

        offsets[0] = 0;
        if (numFields) {
          PetscInt f;

          for (f = 0; f < numFields; f++) {
            PetscInt fDof;
            PetscCall(PetscSectionGetFieldDof(localFine,p,f,&fDof));
            offsets[f + 1] = fDof + offsets[f];
          }
          PetscCall(DMPlexGetIndicesPointFields_Internal(localFine,PETSC_FALSE,p,gOff < 0 ? -(gOff + 1) : gOff,offsets,PETSC_FALSE,NULL,-1, NULL,pInd));
        } else {
          PetscCall(DMPlexGetIndicesPoint_Internal(localFine,PETSC_FALSE,p,gOff < 0 ? -(gOff + 1) : gOff,offsets,PETSC_FALSE,NULL, NULL,pInd));
        }
        if (gatheredValues) PetscCall(VecGetValues(fineVec,dof,pInd,pVal));
      }
    }
    PetscCall(PetscSFCreateEmbeddedLeafSF(coarseToFine, numPointsWithDofs, pointsWithDofs, &coarseToFineEmbedded));
    PetscCall(PetscFree(pointsWithDofs));
  }

  PetscCall(DMPlexGetChart(coarse,&pStartC,&pEndC));
  PetscCall(DMGetLocalSection(coarse,&localCoarse));
  PetscCall(DMGetGlobalSection(coarse,&globalCoarse));

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

    PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)coarse),&rank));
    PetscCallMPI(MPI_Type_contiguous(3,MPIU_INT,&threeInt));
    PetscCallMPI(MPI_Type_commit(&threeInt));
    PetscCall(PetscMalloc2(pEndC-pStartC,&parentNodeAndIdCoarse,pEndF-pStartF,&parentNodeAndIdFine));
    PetscCall(DMGetPointSF(coarse,&pointSF));
    PetscCall(PetscSFGetGraph(pointSF,NULL,&nleaves,&ilocal,&iremote));
    for (p = pStartC; p < pEndC; p++) {
      PetscInt parent, childId;
      PetscCall(DMPlexGetTreeParent(coarse,p,&parent,&childId));
      parentNodeAndIdCoarse[p - pStartC][0] = rank;
      parentNodeAndIdCoarse[p - pStartC][1] = parent - pStartC;
      parentNodeAndIdCoarse[p - pStartC][2] = (p == parent) ? -1 : childId;
      if (nleaves > 0) {
        PetscInt leaf = -1;

        if (ilocal) {
          PetscCall(PetscFindInt(parent,nleaves,ilocal,&leaf));
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
    PetscCall(PetscSFBcastBegin(coarseToFineEmbedded,threeInt,parentNodeAndIdCoarse,parentNodeAndIdFine,MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(coarseToFineEmbedded,threeInt,parentNodeAndIdCoarse,parentNodeAndIdFine,MPI_REPLACE));
    for (p = pStartF, nleavesToParents = 0; p < pEndF; p++) {
      PetscInt dof;

      PetscCall(PetscSectionGetDof(leafIndicesSec,p,&dof));
      if (dof) {
        PetscInt off;

        PetscCall(PetscSectionGetOffset(leafIndicesSec,p,&off));
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
    PetscCall(PetscMalloc1(nleavesToParents,&ilocalToParents));
    PetscCall(PetscMalloc1(nleavesToParents,&iremoteToParents));
    for (p = pStartF, nleavesToParents = 0; p < pEndF; p++) {
      if (parentNodeAndIdFine[p-pStartF][0] >= 0) {
        ilocalToParents[nleavesToParents] = p - pStartF;
        iremoteToParents[nleavesToParents].rank  = parentNodeAndIdFine[p-pStartF][0];
        iremoteToParents[nleavesToParents].index = parentNodeAndIdFine[p-pStartF][1];
        nleavesToParents++;
      }
    }
    PetscCall(PetscSFCreate(PetscObjectComm((PetscObject)coarse),&sfToParents));
    PetscCall(PetscSFSetGraph(sfToParents,pEndC-pStartC,nleavesToParents,ilocalToParents,PETSC_OWN_POINTER,iremoteToParents,PETSC_OWN_POINTER));
    PetscCall(PetscSFDestroy(&coarseToFineEmbedded));

    coarseToFineEmbedded = sfToParents;

    PetscCall(PetscFree2(parentNodeAndIdCoarse,parentNodeAndIdFine));
    PetscCallMPI(MPI_Type_free(&threeInt));
  }

  { /* winnow out coarse points that don't have dofs */
    PetscInt dof, cdof, numPointsWithDofs, offset, *pointsWithDofs;
    PetscSF  sfDofsOnly;

    for (p = pStartC, numPointsWithDofs = 0; p < pEndC; p++) {
      PetscCall(PetscSectionGetDof(globalCoarse,p,&dof));
      PetscCall(PetscSectionGetConstraintDof(globalCoarse,p,&cdof));
      if ((dof - cdof) > 0) {
        numPointsWithDofs++;
      }
    }
    PetscCall(PetscMalloc1(numPointsWithDofs,&pointsWithDofs));
    for (p = pStartC, offset = 0; p < pEndC; p++) {
      PetscCall(PetscSectionGetDof(globalCoarse,p,&dof));
      PetscCall(PetscSectionGetConstraintDof(globalCoarse,p,&cdof));
      if ((dof - cdof) > 0) {
        pointsWithDofs[offset++] = p - pStartC;
      }
    }
    PetscCall(PetscSFCreateEmbeddedRootSF(coarseToFineEmbedded, numPointsWithDofs, pointsWithDofs, &sfDofsOnly));
    PetscCall(PetscSFDestroy(&coarseToFineEmbedded));
    PetscCall(PetscFree(pointsWithDofs));
    coarseToFineEmbedded = sfDofsOnly;
  }

  /* communicate back to the coarse mesh which coarse points have children (that may require injection) */
  PetscCall(PetscSFComputeDegreeBegin(coarseToFineEmbedded,&rootDegrees));
  PetscCall(PetscSFComputeDegreeEnd(coarseToFineEmbedded,&rootDegrees));
  PetscCall(PetscSectionCreate(PetscObjectComm((PetscObject)coarse),&multiRootSec));
  PetscCall(PetscSectionSetChart(multiRootSec,pStartC,pEndC));
  for (p = pStartC; p < pEndC; p++) {
    PetscCall(PetscSectionSetDof(multiRootSec,p,rootDegrees[p-pStartC]));
  }
  PetscCall(PetscSectionSetUp(multiRootSec));
  PetscCall(PetscSectionGetStorageSize(multiRootSec,&numMulti));
  PetscCall(PetscSectionCreate(PetscObjectComm((PetscObject)coarse),&rootIndicesSec));
  { /* distribute the leaf section */
    PetscSF multi, multiInv, indicesSF;
    PetscInt *remoteOffsets, numRootIndices;

    PetscCall(PetscSFGetMultiSF(coarseToFineEmbedded,&multi));
    PetscCall(PetscSFCreateInverseSF(multi,&multiInv));
    PetscCall(PetscSFDistributeSection(multiInv,leafIndicesSec,&remoteOffsets,rootIndicesSec));
    PetscCall(PetscSFCreateSectionSF(multiInv,leafIndicesSec,remoteOffsets,rootIndicesSec,&indicesSF));
    PetscCall(PetscFree(remoteOffsets));
    PetscCall(PetscSFDestroy(&multiInv));
    PetscCall(PetscSectionGetStorageSize(rootIndicesSec,&numRootIndices));
    if (gatheredIndices) {
      PetscCall(PetscMalloc1(numRootIndices,&rootInds));
      PetscCall(PetscSFBcastBegin(indicesSF,MPIU_INT,leafInds,rootInds,MPI_REPLACE));
      PetscCall(PetscSFBcastEnd(indicesSF,MPIU_INT,leafInds,rootInds,MPI_REPLACE));
    }
    if (gatheredValues) {
      PetscCall(PetscMalloc1(numRootIndices,&rootVals));
      PetscCall(PetscSFBcastBegin(indicesSF,MPIU_SCALAR,leafVals,rootVals,MPI_REPLACE));
      PetscCall(PetscSFBcastEnd(indicesSF,MPIU_SCALAR,leafVals,rootVals,MPI_REPLACE));
    }
    PetscCall(PetscSFDestroy(&indicesSF));
  }
  PetscCall(PetscSectionDestroy(&leafIndicesSec));
  PetscCall(PetscFree(leafInds));
  PetscCall(PetscFree(leafVals));
  PetscCall(PetscSFDestroy(&coarseToFineEmbedded));
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
  PetscCall(DMPlexGetReferenceTree(coarse,&refTree));
  PetscCall(DMGetDefaultConstraints(refTree,&cSecRef,NULL,NULL));
  PetscCall(PetscSectionGetChart(cSecRef,&pRefStart,&pRefEnd));
  PetscCall(DMPlexReferenceTreeGetInjector(refTree,&injRef));

  PetscCall(DMPlexGetChart(fine,&pStartF,&pEndF));
  PetscCall(DMGetLocalSection(fine,&localFine));
  PetscCall(DMGetGlobalSection(fine,&globalFine));
  PetscCall(PetscSectionGetNumFields(localFine,&numFields));
  PetscCall(DMPlexGetChart(coarse,&pStartC,&pEndC));
  PetscCall(DMGetLocalSection(coarse,&localCoarse));
  PetscCall(DMGetGlobalSection(coarse,&globalCoarse));
  PetscCall(PetscSectionGetMaxDof(localCoarse,&maxDof));
  {
    PetscInt maxFields = PetscMax(1,numFields) + 1;
    PetscCall(PetscMalloc3(maxFields,&offsets,maxFields,&offsetsCopy,maxFields,&rowOffsets));
  }

  PetscCall(DMPlexTransferInjectorTree(coarse,fine,coarseToFine,childIds,NULL,numFields,offsets,&multiRootSec,&rootIndicesSec,&rootIndices,NULL));

  PetscCall(PetscMalloc1(maxDof,&parentIndices));

  /* count indices */
  PetscCall(MatGetLayouts(mat,&rowMap,&colMap));
  PetscCall(PetscLayoutSetUp(rowMap));
  PetscCall(PetscLayoutSetUp(colMap));
  PetscCall(PetscLayoutGetRange(rowMap,&rowStart,&rowEnd));
  PetscCall(PetscLayoutGetRange(colMap,&colStart,&colEnd));
  PetscCall(PetscCalloc2(rowEnd-rowStart,&nnzD,rowEnd-rowStart,&nnzO));
  for (p = pStartC; p < pEndC; p++) {
    PetscInt numLeaves, leafStart, leafEnd, l, dof, cdof, gOff;

    PetscCall(PetscSectionGetDof(globalCoarse,p,&dof));
    PetscCall(PetscSectionGetConstraintDof(globalCoarse,p,&cdof));
    if ((dof - cdof) <= 0) continue;
    PetscCall(PetscSectionGetOffset(globalCoarse,p,&gOff));

    rowOffsets[0] = 0;
    offsetsCopy[0] = 0;
    if (numFields) {
      PetscInt f;

      for (f = 0; f < numFields; f++) {
        PetscInt fDof;
        PetscCall(PetscSectionGetFieldDof(localCoarse,p,f,&fDof));
        rowOffsets[f + 1] = offsetsCopy[f + 1] = fDof + rowOffsets[f];
      }
      PetscCall(DMPlexGetIndicesPointFields_Internal(localCoarse,PETSC_FALSE,p,gOff < 0 ? -(gOff + 1) : gOff,offsetsCopy,PETSC_FALSE,NULL,-1, NULL,parentIndices));
    } else {
      PetscCall(DMPlexGetIndicesPoint_Internal(localCoarse,PETSC_FALSE,p,gOff < 0 ? -(gOff + 1) : gOff,offsetsCopy,PETSC_FALSE,NULL, NULL,parentIndices));
      rowOffsets[1] = offsetsCopy[0];
    }

    PetscCall(PetscSectionGetDof(multiRootSec,p,&numLeaves));
    PetscCall(PetscSectionGetOffset(multiRootSec,p,&leafStart));
    leafEnd = leafStart + numLeaves;
    for (l = leafStart; l < leafEnd; l++) {
      PetscInt numIndices, childId, offset;
      const PetscInt *childIndices;

      PetscCall(PetscSectionGetDof(rootIndicesSec,l,&numIndices));
      PetscCall(PetscSectionGetOffset(rootIndicesSec,l,&offset));
      childId = rootIndices[offset++];
      childIndices = &rootIndices[offset];
      numIndices--;

      if (childId == -1) { /* equivalent points: scatter */
        PetscInt i;

        for (i = 0; i < numIndices; i++) {
          PetscInt colIndex = childIndices[i];
          PetscInt rowIndex = parentIndices[i];
          if (rowIndex < 0) continue;
          PetscCheck(colIndex >= 0,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Unconstrained fine and constrained coarse");
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

        PetscCall(DMPlexGetTreeParent(refTree,childId,&parentId,NULL));

        lim = PetscMax(1,numFields);
        offsets[0] = 0;
        if (numFields) {
          PetscInt f;

          for (f = 0; f < numFields; f++) {
            PetscInt fDof;
            PetscCall(PetscSectionGetFieldDof(cSecRef,childId,f,&fDof));

            offsets[f + 1] = fDof + offsets[f];
          }
        }
        else {
          PetscInt cDof;

          PetscCall(PetscSectionGetDof(cSecRef,childId,&cDof));
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
  PetscCall(MatXAIJSetPreallocation(mat,1,nnzD,nnzO,NULL,NULL));
  PetscCall(PetscFree2(nnzD,nnzO));
  /* insert values */
  PetscCall(DMPlexReferenceTreeGetChildrenMatrices_Injection(refTree,injRef,&childrenMats));
  for (p = pStartC; p < pEndC; p++) {
    PetscInt numLeaves, leafStart, leafEnd, l, dof, cdof, gOff;

    PetscCall(PetscSectionGetDof(globalCoarse,p,&dof));
    PetscCall(PetscSectionGetConstraintDof(globalCoarse,p,&cdof));
    if ((dof - cdof) <= 0) continue;
    PetscCall(PetscSectionGetOffset(globalCoarse,p,&gOff));

    rowOffsets[0] = 0;
    offsetsCopy[0] = 0;
    if (numFields) {
      PetscInt f;

      for (f = 0; f < numFields; f++) {
        PetscInt fDof;
        PetscCall(PetscSectionGetFieldDof(localCoarse,p,f,&fDof));
        rowOffsets[f + 1] = offsetsCopy[f + 1] = fDof + rowOffsets[f];
      }
      PetscCall(DMPlexGetIndicesPointFields_Internal(localCoarse,PETSC_FALSE,p,gOff < 0 ? -(gOff + 1) : gOff,offsetsCopy,PETSC_FALSE,NULL,-1, NULL,parentIndices));
    } else {
      PetscCall(DMPlexGetIndicesPoint_Internal(localCoarse,PETSC_FALSE,p,gOff < 0 ? -(gOff + 1) : gOff,offsetsCopy,PETSC_FALSE,NULL, NULL,parentIndices));
      rowOffsets[1] = offsetsCopy[0];
    }

    PetscCall(PetscSectionGetDof(multiRootSec,p,&numLeaves));
    PetscCall(PetscSectionGetOffset(multiRootSec,p,&leafStart));
    leafEnd = leafStart + numLeaves;
    for (l = leafStart; l < leafEnd; l++) {
      PetscInt numIndices, childId, offset;
      const PetscInt *childIndices;

      PetscCall(PetscSectionGetDof(rootIndicesSec,l,&numIndices));
      PetscCall(PetscSectionGetOffset(rootIndicesSec,l,&offset));
      childId = rootIndices[offset++];
      childIndices = &rootIndices[offset];
      numIndices--;

      if (childId == -1) { /* equivalent points: scatter */
        PetscInt i;

        for (i = 0; i < numIndices; i++) {
          PetscCall(MatSetValue(mat,parentIndices[i],childIndices[i],1.,INSERT_VALUES));
        }
      }
      else {
        PetscInt parentId, f, lim;

        PetscCall(DMPlexGetTreeParent(refTree,childId,&parentId,NULL));

        lim = PetscMax(1,numFields);
        offsets[0] = 0;
        if (numFields) {
          PetscInt f;

          for (f = 0; f < numFields; f++) {
            PetscInt fDof;
            PetscCall(PetscSectionGetFieldDof(cSecRef,childId,f,&fDof));

            offsets[f + 1] = fDof + offsets[f];
          }
        }
        else {
          PetscInt cDof;

          PetscCall(PetscSectionGetDof(cSecRef,childId,&cDof));
          offsets[1] = cDof;
        }
        for (f = 0; f < lim; f++) {
          PetscScalar    *childMat   = &childrenMats[childId - pRefStart][f][0];
          PetscInt       *rowIndices = &parentIndices[rowOffsets[f]];
          const PetscInt *colIndices = &childIndices[offsets[f]];

          PetscCall(MatSetValues(mat,rowOffsets[f+1]-rowOffsets[f],rowIndices,offsets[f+1]-offsets[f],colIndices,childMat,INSERT_VALUES));
        }
      }
    }
  }
  PetscCall(PetscSectionDestroy(&multiRootSec));
  PetscCall(PetscSectionDestroy(&rootIndicesSec));
  PetscCall(PetscFree(parentIndices));
  PetscCall(DMPlexReferenceTreeRestoreChildrenMatrices_Injection(refTree,injRef,&childrenMats));
  PetscCall(PetscFree(rootIndices));
  PetscCall(PetscFree3(offsets,offsetsCopy,rowOffsets));

  PetscCall(MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY));
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
  PetscCall(VecSetOption(vecFine,VEC_IGNORE_NEGATIVE_INDICES,PETSC_TRUE));
  PetscCall(DMPlexGetChart(coarse,&pStartC,&pEndC));
  PetscCall(DMPlexGetSimplexOrBoxCells(coarse,0,&cellStart,&cellEnd));
  PetscCall(DMPlexGetChart(fine,&pStartF,&pEndF));
  PetscCall(DMGetGlobalSection(fine,&globalFine));
  PetscCall(DMGetCoordinateDim(coarse,&dim));
  { /* winnow fine points that don't have global dofs out of the sf */
    PetscInt       nleaves, l;
    const PetscInt *leaves;
    PetscInt       dof, cdof, numPointsWithDofs, offset, *pointsWithDofs;

    PetscCall(PetscSFGetGraph(coarseToFine,NULL,&nleaves,&leaves,NULL));

    for (l = 0, numPointsWithDofs = 0; l < nleaves; l++) {
      PetscInt p = leaves ? leaves[l] : l;

      PetscCall(PetscSectionGetDof(globalFine,p,&dof));
      PetscCall(PetscSectionGetConstraintDof(globalFine,p,&cdof));
      if ((dof - cdof) > 0) {
        numPointsWithDofs++;
      }
    }
    PetscCall(PetscMalloc1(numPointsWithDofs,&pointsWithDofs));
    for (l = 0, offset = 0; l < nleaves; l++) {
      PetscInt p = leaves ? leaves[l] : l;

      PetscCall(PetscSectionGetDof(globalFine,p,&dof));
      PetscCall(PetscSectionGetConstraintDof(globalFine,p,&cdof));
      if ((dof - cdof) > 0) {
        pointsWithDofs[offset++] = l;
      }
    }
    PetscCall(PetscSFCreateEmbeddedLeafSF(coarseToFine, numPointsWithDofs, pointsWithDofs, &coarseToFineEmbedded));
    PetscCall(PetscFree(pointsWithDofs));
  }
  /* communicate back to the coarse mesh which coarse points have children (that may require interpolation) */
  PetscCall(PetscMalloc1(pEndC-pStartC,&maxChildIds));
  for (p = pStartC; p < pEndC; p++) {
    maxChildIds[p - pStartC] = -2;
  }
  PetscCall(PetscSFReduceBegin(coarseToFineEmbedded,MPIU_INT,cids,maxChildIds,MPIU_MAX));
  PetscCall(PetscSFReduceEnd(coarseToFineEmbedded,MPIU_INT,cids,maxChildIds,MPIU_MAX));

  PetscCall(DMGetLocalSection(coarse,&localCoarse));
  PetscCall(DMGetGlobalSection(coarse,&globalCoarse));

  PetscCall(DMPlexGetAnchors(coarse,&aSec,&aIS));
  PetscCall(ISGetIndices(aIS,&anchors));
  PetscCall(PetscSectionGetChart(aSec,&aStart,&aEnd));

  PetscCall(DMGetDefaultConstraints(coarse,&cSec,&cMat,NULL));
  PetscCall(PetscSectionGetChart(cSec,&cStart,&cEnd));

  /* create sections that will send to children the indices and matrices they will need to construct the interpolator */
  PetscCall(PetscSectionCreate(PetscObjectComm((PetscObject)coarse),&rootValuesSec));
  PetscCall(PetscSectionSetChart(rootValuesSec,pStartC,pEndC));
  PetscCall(PetscSectionGetNumFields(localCoarse,&numFields));
  {
    PetscInt maxFields = PetscMax(1,numFields) + 1;
    PetscCall(PetscMalloc7(maxFields,&offsets,maxFields,&offsetsCopy,maxFields,&newOffsets,maxFields,&newOffsetsCopy,maxFields,&rowOffsets,maxFields,&numD,maxFields,&numO));
  }
  if (grad) {
    PetscInt i;

    PetscCall(VecGetDM(cellGeom,&cellDM));
    PetscCall(VecGetArrayRead(cellGeom,&cellGeomArray));
    PetscCall(VecGetDM(grad,&gradDM));
    PetscCall(VecGetArrayRead(grad,&gradArray));
    for (i = 0; i < PetscMax(1,numFields); i++) {
      PetscObject  obj;
      PetscClassId id;

      PetscCall(DMGetField(coarse, i, NULL, &obj));
      PetscCall(PetscObjectGetClassId(obj,&id));
      if (id == PETSCFV_CLASSID) {
        fv      = (PetscFV) obj;
        PetscCall(PetscFVGetNumComponents(fv,&numFVcomps));
        fvField = i;
        break;
      }
    }
  }

  for (p = pStartC; p < pEndC; p++) { /* count the sizes of the indices and matrices */
    PetscInt dof;
    PetscInt maxChildId     = maxChildIds[p - pStartC];
    PetscInt numValues      = 0;

    PetscCall(PetscSectionGetDof(globalCoarse,p,&dof));
    if (dof < 0) {
      dof = -(dof + 1);
    }
    offsets[0]    = 0;
    newOffsets[0] = 0;
    if (maxChildId >= 0) { /* this point has children (with dofs) that will need to be interpolated from the closure of p */
      PetscInt *closure = NULL, closureSize, cl;

      PetscCall(DMPlexGetTransitiveClosure(coarse,p,PETSC_TRUE,&closureSize,&closure));
      for (cl = 0; cl < closureSize; cl++) { /* get the closure */
        PetscInt c = closure[2 * cl], clDof;

        PetscCall(PetscSectionGetDof(localCoarse,c,&clDof));
        numValues += clDof;
      }
      PetscCall(DMPlexRestoreTransitiveClosure(coarse,p,PETSC_TRUE,&closureSize,&closure));
    }
    else if (maxChildId == -1) {
      PetscCall(PetscSectionGetDof(localCoarse,p,&numValues));
    }
    /* we will pack the column indices with the field offsets */
    if (maxChildId >= 0 && grad && p >= cellStart && p < cellEnd) {
      /* also send the centroid, and the gradient */
      numValues += dim * (1 + numFVcomps);
    }
    PetscCall(PetscSectionSetDof(rootValuesSec,p,numValues));
  }
  PetscCall(PetscSectionSetUp(rootValuesSec));
  {
    PetscInt          numRootValues;
    const PetscScalar *coarseArray;

    PetscCall(PetscSectionGetStorageSize(rootValuesSec,&numRootValues));
    PetscCall(PetscMalloc1(numRootValues,&rootValues));
    PetscCall(VecGetArrayRead(vecCoarseLocal,&coarseArray));
    for (p = pStartC; p < pEndC; p++) {
      PetscInt    numValues;
      PetscInt    pValOff;
      PetscScalar *pVal;
      PetscInt    maxChildId = maxChildIds[p - pStartC];

      PetscCall(PetscSectionGetDof(rootValuesSec,p,&numValues));
      if (!numValues) {
        continue;
      }
      PetscCall(PetscSectionGetOffset(rootValuesSec,p,&pValOff));
      pVal = &(rootValues[pValOff]);
      if (maxChildId >= 0) { /* build an identity matrix, apply matrix constraints on the right */
        PetscInt closureSize = numValues;
        PetscCall(DMPlexVecGetClosure(coarse,NULL,vecCoarseLocal,p,&closureSize,&pVal));
        if (grad && p >= cellStart && p < cellEnd) {
          PetscFVCellGeom *cg;
          PetscScalar     *gradVals = NULL;
          PetscInt        i;

          pVal += (numValues - dim * (1 + numFVcomps));

          PetscCall(DMPlexPointLocalRead(cellDM,p,cellGeomArray,(void *) &cg));
          for (i = 0; i < dim; i++) pVal[i] = cg->centroid[i];
          pVal += dim;
          PetscCall(DMPlexPointGlobalRead(gradDM,p,gradArray,(void *) &gradVals));
          for (i = 0; i < dim * numFVcomps; i++) pVal[i] = gradVals[i];
        }
      }
      else if (maxChildId == -1) {
        PetscInt lDof, lOff, i;

        PetscCall(PetscSectionGetDof(localCoarse,p,&lDof));
        PetscCall(PetscSectionGetOffset(localCoarse,p,&lOff));
        for (i = 0; i < lDof; i++) pVal[i] = coarseArray[lOff + i];
      }
    }
    PetscCall(VecRestoreArrayRead(vecCoarseLocal,&coarseArray));
    PetscCall(PetscFree(maxChildIds));
  }
  {
    PetscSF  valuesSF;
    PetscInt *remoteOffsetsValues, numLeafValues;

    PetscCall(PetscSectionCreate(PetscObjectComm((PetscObject)fine),&leafValuesSec));
    PetscCall(PetscSFDistributeSection(coarseToFineEmbedded,rootValuesSec,&remoteOffsetsValues,leafValuesSec));
    PetscCall(PetscSFCreateSectionSF(coarseToFineEmbedded,rootValuesSec,remoteOffsetsValues,leafValuesSec,&valuesSF));
    PetscCall(PetscSFDestroy(&coarseToFineEmbedded));
    PetscCall(PetscFree(remoteOffsetsValues));
    PetscCall(PetscSectionGetStorageSize(leafValuesSec,&numLeafValues));
    PetscCall(PetscMalloc1(numLeafValues,&leafValues));
    PetscCall(PetscSFBcastBegin(valuesSF,MPIU_SCALAR,rootValues,leafValues,MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(valuesSF,MPIU_SCALAR,rootValues,leafValues,MPI_REPLACE));
    PetscCall(PetscSFDestroy(&valuesSF));
    PetscCall(PetscFree(rootValues));
    PetscCall(PetscSectionDestroy(&rootValuesSec));
  }
  PetscCall(DMGetLocalSection(fine,&localFine));
  {
    PetscInt    maxDof;
    PetscInt    *rowIndices;
    DM           refTree;
    PetscInt     **refPointFieldN;
    PetscScalar  ***refPointFieldMats;
    PetscSection refConSec, refAnSec;
    PetscInt     pRefStart,pRefEnd,leafStart,leafEnd;
    PetscScalar  *pointWork;

    PetscCall(PetscSectionGetMaxDof(localFine,&maxDof));
    PetscCall(DMGetWorkArray(fine,maxDof,MPIU_INT,&rowIndices));
    PetscCall(DMGetWorkArray(fine,maxDof,MPIU_SCALAR,&pointWork));
    PetscCall(DMPlexGetReferenceTree(fine,&refTree));
    PetscCall(DMCopyDisc(fine,refTree));
    PetscCall(DMPlexReferenceTreeGetChildrenMatrices(refTree,&refPointFieldMats,&refPointFieldN));
    PetscCall(DMGetDefaultConstraints(refTree,&refConSec,NULL,NULL));
    PetscCall(DMPlexGetAnchors(refTree,&refAnSec,NULL));
    PetscCall(PetscSectionGetChart(refConSec,&pRefStart,&pRefEnd));
    PetscCall(PetscSectionGetChart(leafValuesSec,&leafStart,&leafEnd));
    PetscCall(DMPlexGetSimplexOrBoxCells(fine,0,&cellStart,&cellEnd));
    for (p = leafStart; p < leafEnd; p++) {
      PetscInt          gDof, gcDof, gOff, lDof;
      PetscInt          numValues, pValOff;
      PetscInt          childId;
      const PetscScalar *pVal;
      const PetscScalar *fvGradData = NULL;

      PetscCall(PetscSectionGetDof(globalFine,p,&gDof));
      PetscCall(PetscSectionGetDof(localFine,p,&lDof));
      PetscCall(PetscSectionGetConstraintDof(globalFine,p,&gcDof));
      if ((gDof - gcDof) <= 0) {
        continue;
      }
      PetscCall(PetscSectionGetOffset(globalFine,p,&gOff));
      PetscCall(PetscSectionGetDof(leafValuesSec,p,&numValues));
      if (!numValues) continue;
      PetscCall(PetscSectionGetOffset(leafValuesSec,p,&pValOff));
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

          PetscCall(PetscSectionGetFieldDof(localFine,p,f,&rowDof));
          offsets[f + 1]        = offsets[f] + rowDof;
          offsetsCopy[f + 1]    = offsets[f + 1];
          /* TODO: closure indices */
          newOffsets[f + 1]     = newOffsets[f] + ((childId == -1) ? rowDof : refPointFieldN[childId - pRefStart][f]);
        }
        PetscCall(DMPlexGetIndicesPointFields_Internal(localFine,PETSC_FALSE,p,gOff,offsetsCopy,PETSC_FALSE,NULL,-1,NULL,rowIndices));
      }
      else {
        offsets[0]    = 0;
        offsets[1]    = lDof;
        newOffsets[0] = 0;
        newOffsets[1] = (childId == -1) ? lDof : refPointFieldN[childId - pRefStart][0];
        PetscCall(DMPlexGetIndicesPoint_Internal(localFine,PETSC_FALSE,p,gOff,offsetsCopy,PETSC_FALSE,NULL,NULL,rowIndices));
      }
      if (childId == -1) { /* no child interpolation: one nnz per */
        PetscCall(VecSetValues(vecFine,numValues,rowIndices,pVal,INSERT_VALUES));
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
          PetscCall(PetscInfo(coarse,"childId %" PetscInt_FMT ", numRows %" PetscInt_FMT ", numCols %" PetscInt_FMT ", refPointFieldN %" PetscInt_FMT " maxDof %" PetscInt_FMT "\n",childId,numRows,numCols,refPointFieldN[childId - pRefStart][f], maxDof));
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

            PetscCall(DMPlexComputeCellGeometryFVM(fine,p,NULL,centroid,NULL));
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
          PetscCall(VecSetValues(vecFine,numRows,&rowIndices[offsets[f]],rVal,INSERT_VALUES));
        }
      }
    }
    PetscCall(DMPlexReferenceTreeRestoreChildrenMatrices(refTree,&refPointFieldMats,&refPointFieldN));
    PetscCall(DMRestoreWorkArray(fine,maxDof,MPIU_SCALAR,&pointWork));
    PetscCall(DMRestoreWorkArray(fine,maxDof,MPIU_INT,&rowIndices));
  }
  PetscCall(PetscFree(leafValues));
  PetscCall(PetscSectionDestroy(&leafValuesSec));
  PetscCall(PetscFree7(offsets,offsetsCopy,newOffsets,newOffsetsCopy,rowOffsets,numD,numO));
  PetscCall(ISRestoreIndices(aIS,&anchors));
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
  PetscCall(VecSetOption(vecFine,VEC_IGNORE_NEGATIVE_INDICES,PETSC_TRUE));
  PetscCall(VecSetOption(vecCoarse,VEC_IGNORE_NEGATIVE_INDICES,PETSC_TRUE));
  PetscCall(DMPlexGetReferenceTree(coarse,&refTree));
  PetscCall(DMCopyDisc(coarse,refTree));
  PetscCall(DMGetDefaultConstraints(refTree,&cSecRef,NULL,NULL));
  PetscCall(PetscSectionGetChart(cSecRef,&pRefStart,&pRefEnd));
  PetscCall(DMPlexReferenceTreeGetInjector(refTree,&injRef));

  PetscCall(DMPlexGetChart(fine,&pStartF,&pEndF));
  PetscCall(DMGetLocalSection(fine,&localFine));
  PetscCall(DMGetGlobalSection(fine,&globalFine));
  PetscCall(PetscSectionGetNumFields(localFine,&numFields));
  PetscCall(DMPlexGetChart(coarse,&pStartC,&pEndC));
  PetscCall(DMGetLocalSection(coarse,&localCoarse));
  PetscCall(DMGetGlobalSection(coarse,&globalCoarse));
  PetscCall(PetscSectionGetMaxDof(localCoarse,&maxDof));
  {
    PetscInt maxFields = PetscMax(1,numFields) + 1;
    PetscCall(PetscMalloc3(maxFields,&offsets,maxFields,&offsetsCopy,maxFields,&rowOffsets));
  }

  PetscCall(DMPlexTransferInjectorTree(coarse,fine,coarseToFine,cids,vecFine,numFields,offsets,&multiRootSec,&rootIndicesSec,NULL,&rootValues));

  PetscCall(PetscMalloc2(maxDof,&parentIndices,maxDof,&parentValues));

  /* count indices */
  PetscCall(VecGetLayout(vecFine,&colMap));
  PetscCall(VecGetLayout(vecCoarse,&rowMap));
  PetscCall(PetscLayoutSetUp(rowMap));
  PetscCall(PetscLayoutSetUp(colMap));
  PetscCall(PetscLayoutGetRange(rowMap,&rowStart,&rowEnd));
  PetscCall(PetscLayoutGetRange(colMap,&colStart,&colEnd));
  /* insert values */
  PetscCall(DMPlexReferenceTreeGetChildrenMatrices_Injection(refTree,injRef,&childrenMats));
  for (p = pStartC; p < pEndC; p++) {
    PetscInt  numLeaves, leafStart, leafEnd, l, dof, cdof, gOff;
    PetscBool contribute = PETSC_FALSE;

    PetscCall(PetscSectionGetDof(globalCoarse,p,&dof));
    PetscCall(PetscSectionGetConstraintDof(globalCoarse,p,&cdof));
    if ((dof - cdof) <= 0) continue;
    PetscCall(PetscSectionGetDof(localCoarse,p,&dof));
    PetscCall(PetscSectionGetOffset(globalCoarse,p,&gOff));

    rowOffsets[0] = 0;
    offsetsCopy[0] = 0;
    if (numFields) {
      PetscInt f;

      for (f = 0; f < numFields; f++) {
        PetscInt fDof;
        PetscCall(PetscSectionGetFieldDof(localCoarse,p,f,&fDof));
        rowOffsets[f + 1] = offsetsCopy[f + 1] = fDof + rowOffsets[f];
      }
      PetscCall(DMPlexGetIndicesPointFields_Internal(localCoarse,PETSC_FALSE,p,gOff < 0 ? -(gOff + 1) : gOff,offsetsCopy,PETSC_FALSE,NULL,-1,NULL,parentIndices));
    } else {
      PetscCall(DMPlexGetIndicesPoint_Internal(localCoarse,PETSC_FALSE,p,gOff < 0 ? -(gOff + 1) : gOff,offsetsCopy,PETSC_FALSE,NULL,NULL,parentIndices));
      rowOffsets[1] = offsetsCopy[0];
    }

    PetscCall(PetscSectionGetDof(multiRootSec,p,&numLeaves));
    PetscCall(PetscSectionGetOffset(multiRootSec,p,&leafStart));
    leafEnd = leafStart + numLeaves;
    for (l = 0; l < dof; l++) parentValues[l] = 0.;
    for (l = leafStart; l < leafEnd; l++) {
      PetscInt numIndices, childId, offset;
      const PetscScalar *childValues;

      PetscCall(PetscSectionGetDof(rootIndicesSec,l,&numIndices));
      PetscCall(PetscSectionGetOffset(rootIndicesSec,l,&offset));
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
        PetscCall(DMPlexGetTreeParent(refTree,childId,&parentId,NULL));

        lim = PetscMax(1,numFields);
        offsets[0] = 0;
        if (numFields) {
          PetscInt f;

          for (f = 0; f < numFields; f++) {
            PetscInt fDof;
            PetscCall(PetscSectionGetFieldDof(cSecRef,childId,f,&fDof));

            offsets[f + 1] = fDof + offsets[f];
          }
        }
        else {
          PetscInt cDof;

          PetscCall(PetscSectionGetDof(cSecRef,childId,&cDof));
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
    if (contribute) PetscCall(VecSetValues(vecCoarse,dof,parentIndices,parentValues,INSERT_VALUES));
  }
  PetscCall(PetscSectionDestroy(&multiRootSec));
  PetscCall(PetscSectionDestroy(&rootIndicesSec));
  PetscCall(PetscFree2(parentIndices,parentValues));
  PetscCall(DMPlexReferenceTreeRestoreChildrenMatrices_Injection(refTree,injRef,&childrenMats));
  PetscCall(PetscFree(rootValues));
  PetscCall(PetscFree3(offsets,offsetsCopy,rowOffsets));
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

.seealso: `DMPlexSetReferenceTree()`, `DMPlexGetReferenceTree()`, `PetscFVGetComputeGradients()`
@*/
PetscErrorCode DMPlexTransferVecTree(DM dmIn, Vec vecIn, DM dmOut, Vec vecOut, PetscSF sfRefine, PetscSF sfCoarsen, PetscInt *cidsRefine, PetscInt *cidsCoarsen, PetscBool useBCs, PetscReal time)
{
  PetscFunctionBegin;
  PetscCall(VecSet(vecOut,0.0));
  if (sfRefine) {
    Vec vecInLocal;
    DM  dmGrad = NULL;
    Vec faceGeom = NULL, cellGeom = NULL, grad = NULL;

    PetscCall(DMGetLocalVector(dmIn,&vecInLocal));
    PetscCall(VecSet(vecInLocal,0.0));
    {
      PetscInt  numFields, i;

      PetscCall(DMGetNumFields(dmIn, &numFields));
      for (i = 0; i < numFields; i++) {
        PetscObject  obj;
        PetscClassId classid;

        PetscCall(DMGetField(dmIn, i, NULL, &obj));
        PetscCall(PetscObjectGetClassId(obj, &classid));
        if (classid == PETSCFV_CLASSID) {
          PetscCall(DMPlexGetDataFVM(dmIn,(PetscFV)obj,&cellGeom,&faceGeom,&dmGrad));
          break;
        }
      }
    }
    if (useBCs) {
      PetscCall(DMPlexInsertBoundaryValues(dmIn,PETSC_TRUE,vecInLocal,time,faceGeom,cellGeom,NULL));
    }
    PetscCall(DMGlobalToLocalBegin(dmIn,vecIn,INSERT_VALUES,vecInLocal));
    PetscCall(DMGlobalToLocalEnd(dmIn,vecIn,INSERT_VALUES,vecInLocal));
    if (dmGrad) {
      PetscCall(DMGetGlobalVector(dmGrad,&grad));
      PetscCall(DMPlexReconstructGradientsFVM(dmIn,vecInLocal,grad));
    }
    PetscCall(DMPlexTransferVecTree_Interpolate(dmIn,vecInLocal,dmOut,vecOut,sfRefine,cidsRefine,grad,cellGeom));
    PetscCall(DMRestoreLocalVector(dmIn,&vecInLocal));
    if (dmGrad) {
      PetscCall(DMRestoreGlobalVector(dmGrad,&grad));
    }
  }
  if (sfCoarsen) {
    PetscCall(DMPlexTransferVecTree_Inject(dmIn,vecIn,dmOut,vecOut,sfCoarsen,cidsCoarsen));
  }
  PetscCall(VecAssemblyBegin(vecOut));
  PetscCall(VecAssemblyEnd(vecOut));
  PetscFunctionReturn(0);
}
