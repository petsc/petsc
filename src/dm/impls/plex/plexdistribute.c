#include <petsc-private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/
#include <petscsf.h>

#undef __FUNCT__
#define __FUNCT__ "DMPlexSetAdjacencyUseCone"
/*@
  DMPlexSetAdjacencyUseCone - Define adjacency in the mesh using either the cone or the support first

  Input Parameters:
+ dm      - The DM object
- useCone - Flag to use the cone first

  Level: intermediate

  Notes:
$     FEM:   Two points p and q are adjacent if q \in closure(star(p)), useCone = PETSC_FALSE, useClosure = PETSC_TRUE
$     FVM:   Two points p and q are adjacent if q \in star(cone(p)),    useCone = PETSC_TRUE,  useClosure = PETSC_FALSE
$     FVM++: Two points p and q are adjacent if q \in star(closure(p)), useCone = PETSC_TRUE,  useClosure = PETSC_TRUE

.seealso: DMPlexGetAdjacencyUseCone(), DMPlexSetAdjacencyUseClosure(), DMPlexGetAdjacencyUseClosure(), DMPlexDistribute(), DMPlexPreallocateOperator()
@*/
PetscErrorCode DMPlexSetAdjacencyUseCone(DM dm, PetscBool useCone)
{
  DM_Plex *mesh = (DM_Plex *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  mesh->useCone = useCone;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetAdjacencyUseCone"
/*@
  DMPlexGetAdjacencyUseCone - Query whether adjacency in the mesh uses the cone or the support first

  Input Parameter:
. dm      - The DM object

  Output Parameter:
. useCone - Flag to use the cone first

  Level: intermediate

  Notes:
$     FEM:   Two points p and q are adjacent if q \in closure(star(p)), useCone = PETSC_FALSE, useClosure = PETSC_TRUE
$     FVM:   Two points p and q are adjacent if q \in star(cone(p)),    useCone = PETSC_TRUE,  useClosure = PETSC_FALSE
$     FVM++: Two points p and q are adjacent if q \in star(closure(p)), useCone = PETSC_TRUE,  useClosure = PETSC_TRUE

.seealso: DMPlexSetAdjacencyUseCone(), DMPlexSetAdjacencyUseClosure(), DMPlexGetAdjacencyUseClosure(), DMPlexDistribute(), DMPlexPreallocateOperator()
@*/
PetscErrorCode DMPlexGetAdjacencyUseCone(DM dm, PetscBool *useCone)
{
  DM_Plex *mesh = (DM_Plex *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidIntPointer(useCone, 2);
  *useCone = mesh->useCone;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexSetAdjacencyUseClosure"
/*@
  DMPlexSetAdjacencyUseClosure - Define adjacency in the mesh using the transitive closure

  Input Parameters:
+ dm      - The DM object
- useClosure - Flag to use the closure

  Level: intermediate

  Notes:
$     FEM:   Two points p and q are adjacent if q \in closure(star(p)), useCone = PETSC_FALSE, useClosure = PETSC_TRUE
$     FVM:   Two points p and q are adjacent if q \in star(cone(p)),    useCone = PETSC_TRUE,  useClosure = PETSC_FALSE
$     FVM++: Two points p and q are adjacent if q \in star(closure(p)), useCone = PETSC_TRUE,  useClosure = PETSC_TRUE

.seealso: DMPlexGetAdjacencyUseClosure(), DMPlexSetAdjacencyUseCone(), DMPlexGetAdjacencyUseCone(), DMPlexDistribute(), DMPlexPreallocateOperator()
@*/
PetscErrorCode DMPlexSetAdjacencyUseClosure(DM dm, PetscBool useClosure)
{
  DM_Plex *mesh = (DM_Plex *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  mesh->useClosure = useClosure;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetAdjacencyUseClosure"
/*@
  DMPlexGetAdjacencyUseClosure - Query whether adjacency in the mesh uses the transitive closure

  Input Parameter:
. dm      - The DM object

  Output Parameter:
. useClosure - Flag to use the closure

  Level: intermediate

  Notes:
$     FEM:   Two points p and q are adjacent if q \in closure(star(p)), useCone = PETSC_FALSE, useClosure = PETSC_TRUE
$     FVM:   Two points p and q are adjacent if q \in star(cone(p)),    useCone = PETSC_TRUE,  useClosure = PETSC_FALSE
$     FVM++: Two points p and q are adjacent if q \in star(closure(p)), useCone = PETSC_TRUE,  useClosure = PETSC_TRUE

.seealso: DMPlexSetAdjacencyUseClosure(), DMPlexSetAdjacencyUseCone(), DMPlexGetAdjacencyUseCone(), DMPlexDistribute(), DMPlexPreallocateOperator()
@*/
PetscErrorCode DMPlexGetAdjacencyUseClosure(DM dm, PetscBool *useClosure)
{
  DM_Plex *mesh = (DM_Plex *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidIntPointer(useClosure, 2);
  *useClosure = mesh->useClosure;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexSetAdjacencyUseAnchors"
/*@
  DMPlexSetAdjacencyUseAnchors - Define adjacency in the mesh using the point-to-point constraints.

  Input Parameters:
+ dm      - The DM object
- useAnchors - Flag to use the constraints.  If PETSC_TRUE, then constrained points are omitted from DMPlexGetAdjacency(), and their anchor points appear in their place.

  Level: intermediate

.seealso: DMPlexGetAdjacencyUseClosure(), DMPlexSetAdjacencyUseCone(), DMPlexGetAdjacencyUseCone(), DMPlexDistribute(), DMPlexPreallocateOperator(), DMPlexSetAnchors()
@*/
PetscErrorCode DMPlexSetAdjacencyUseAnchors(DM dm, PetscBool useAnchors)
{
  DM_Plex *mesh = (DM_Plex *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  mesh->useAnchors = useAnchors;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetAdjacencyUseAnchors"
/*@
  DMPlexGetAdjacencyUseAnchors - Query whether adjacency in the mesh uses the point-to-point constraints.

  Input Parameter:
. dm      - The DM object

  Output Parameter:
. useAnchors - Flag to use the closure.  If PETSC_TRUE, then constrained points are omitted from DMPlexGetAdjacency(), and their anchor points appear in their place.

  Level: intermediate

.seealso: DMPlexSetAdjacencyUseAnchors(), DMPlexSetAdjacencyUseCone(), DMPlexGetAdjacencyUseCone(), DMPlexDistribute(), DMPlexPreallocateOperator(), DMPlexSetAnchors()
@*/
PetscErrorCode DMPlexGetAdjacencyUseAnchors(DM dm, PetscBool *useAnchors)
{
  DM_Plex *mesh = (DM_Plex *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidIntPointer(useAnchors, 2);
  *useAnchors = mesh->useAnchors;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetAdjacency_Cone_Internal"
static PetscErrorCode DMPlexGetAdjacency_Cone_Internal(DM dm, PetscInt p, PetscInt *adjSize, PetscInt adj[])
{
  const PetscInt *cone = NULL;
  PetscInt        numAdj = 0, maxAdjSize = *adjSize, coneSize, c;
  PetscErrorCode  ierr;

  PetscFunctionBeginHot;
  ierr = DMPlexGetConeSize(dm, p, &coneSize);CHKERRQ(ierr);
  ierr = DMPlexGetCone(dm, p, &cone);CHKERRQ(ierr);
  for (c = 0; c < coneSize; ++c) {
    const PetscInt *support = NULL;
    PetscInt        supportSize, s, q;

    ierr = DMPlexGetSupportSize(dm, cone[c], &supportSize);CHKERRQ(ierr);
    ierr = DMPlexGetSupport(dm, cone[c], &support);CHKERRQ(ierr);
    for (s = 0; s < supportSize; ++s) {
      for (q = 0; q < numAdj || (adj[numAdj++] = support[s],0); ++q) {
        if (support[s] == adj[q]) break;
      }
      if (numAdj > maxAdjSize) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid mesh exceeded adjacency allocation (%D)", maxAdjSize);
    }
  }
  *adjSize = numAdj;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetAdjacency_Support_Internal"
static PetscErrorCode DMPlexGetAdjacency_Support_Internal(DM dm, PetscInt p, PetscInt *adjSize, PetscInt adj[])
{
  const PetscInt *support = NULL;
  PetscInt        numAdj   = 0, maxAdjSize = *adjSize, supportSize, s;
  PetscErrorCode  ierr;

  PetscFunctionBeginHot;
  ierr = DMPlexGetSupportSize(dm, p, &supportSize);CHKERRQ(ierr);
  ierr = DMPlexGetSupport(dm, p, &support);CHKERRQ(ierr);
  for (s = 0; s < supportSize; ++s) {
    const PetscInt *cone = NULL;
    PetscInt        coneSize, c, q;

    ierr = DMPlexGetConeSize(dm, support[s], &coneSize);CHKERRQ(ierr);
    ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
    for (c = 0; c < coneSize; ++c) {
      for (q = 0; q < numAdj || (adj[numAdj++] = cone[c],0); ++q) {
        if (cone[c] == adj[q]) break;
      }
      if (numAdj > maxAdjSize) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid mesh exceeded adjacency allocation (%D)", maxAdjSize);
    }
  }
  *adjSize = numAdj;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetAdjacency_Transitive_Internal"
static PetscErrorCode DMPlexGetAdjacency_Transitive_Internal(DM dm, PetscInt p, PetscBool useClosure, PetscInt *adjSize, PetscInt adj[])
{
  PetscInt      *star = NULL;
  PetscInt       numAdj = 0, maxAdjSize = *adjSize, starSize, s;
  PetscErrorCode ierr;

  PetscFunctionBeginHot;
  ierr = DMPlexGetTransitiveClosure(dm, p, useClosure, &starSize, &star);CHKERRQ(ierr);
  for (s = 0; s < starSize*2; s += 2) {
    const PetscInt *closure = NULL;
    PetscInt        closureSize, c, q;

    ierr = DMPlexGetTransitiveClosure(dm, star[s], (PetscBool)!useClosure, &closureSize, (PetscInt**) &closure);CHKERRQ(ierr);
    for (c = 0; c < closureSize*2; c += 2) {
      for (q = 0; q < numAdj || (adj[numAdj++] = closure[c],0); ++q) {
        if (closure[c] == adj[q]) break;
      }
      if (numAdj > maxAdjSize) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid mesh exceeded adjacency allocation (%D)", maxAdjSize);
    }
    ierr = DMPlexRestoreTransitiveClosure(dm, star[s], (PetscBool)!useClosure, &closureSize, (PetscInt**) &closure);CHKERRQ(ierr);
  }
  ierr = DMPlexRestoreTransitiveClosure(dm, p, useClosure, &starSize, &star);CHKERRQ(ierr);
  *adjSize = numAdj;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetAdjacency_Internal"
PetscErrorCode DMPlexGetAdjacency_Internal(DM dm, PetscInt p, PetscBool useCone, PetscBool useTransitiveClosure, PetscBool useAnchors, PetscInt *adjSize, PetscInt *adj[])
{
  static PetscInt asiz = 0;
  PetscInt maxAnchors = 1;
  PetscInt aStart = -1, aEnd = -1;
  PetscInt maxAdjSize;
  PetscSection aSec = NULL;
  IS aIS = NULL;
  const PetscInt *anchors;
  PetscErrorCode  ierr;

  PetscFunctionBeginHot;
  if (useAnchors) {
    ierr = DMPlexGetAnchors(dm,&aSec,&aIS);CHKERRQ(ierr);
    if (aSec) {
      ierr = PetscSectionGetMaxDof(aSec,&maxAnchors);CHKERRQ(ierr);
      maxAnchors = PetscMax(1,maxAnchors);
      ierr = PetscSectionGetChart(aSec,&aStart,&aEnd);CHKERRQ(ierr);
      ierr = ISGetIndices(aIS,&anchors);CHKERRQ(ierr);
    }
  }
  if (!*adj) {
    PetscInt depth, coneSeries, supportSeries, maxC, maxS, pStart, pEnd;

    ierr  = DMPlexGetChart(dm, &pStart,&pEnd);CHKERRQ(ierr);
    ierr  = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
    ierr  = DMPlexGetMaxSizes(dm, &maxC, &maxS);CHKERRQ(ierr);
    coneSeries    = (maxC > 1) ? ((PetscPowInt(maxC,depth+1)-1)/(maxC-1)) : depth+1;
    supportSeries = (maxS > 1) ? ((PetscPowInt(maxS,depth+1)-1)/(maxS-1)) : depth+1;
    asiz  = PetscMax(PetscPowInt(maxS,depth)*coneSeries,PetscPowInt(maxC,depth)*supportSeries);
    asiz *= maxAnchors;
    asiz  = PetscMin(asiz,pEnd-pStart);
    ierr  = PetscMalloc1(asiz,adj);CHKERRQ(ierr);
  }
  if (*adjSize < 0) *adjSize = asiz;
  maxAdjSize = *adjSize;
  if (useTransitiveClosure) {
    ierr = DMPlexGetAdjacency_Transitive_Internal(dm, p, useCone, adjSize, *adj);CHKERRQ(ierr);
  } else if (useCone) {
    ierr = DMPlexGetAdjacency_Cone_Internal(dm, p, adjSize, *adj);CHKERRQ(ierr);
  } else {
    ierr = DMPlexGetAdjacency_Support_Internal(dm, p, adjSize, *adj);CHKERRQ(ierr);
  }
  if (useAnchors && aSec) {
    PetscInt origSize = *adjSize;
    PetscInt numAdj = origSize;
    PetscInt i = 0, j;
    PetscInt *orig = *adj;

    while (i < origSize) {
      PetscInt p = orig[i];
      PetscInt aDof = 0;

      if (p >= aStart && p < aEnd) {
        ierr = PetscSectionGetDof(aSec,p,&aDof);CHKERRQ(ierr);
      }
      if (aDof) {
        PetscInt aOff;
        PetscInt s, q;

        for (j = i + 1; j < numAdj; j++) {
          orig[j - 1] = orig[j];
        }
        origSize--;
        numAdj--;
        ierr = PetscSectionGetOffset(aSec,p,&aOff);CHKERRQ(ierr);
        for (s = 0; s < aDof; ++s) {
          for (q = 0; q < numAdj || (orig[numAdj++] = anchors[aOff+s],0); ++q) {
            if (anchors[aOff+s] == orig[q]) break;
          }
          if (numAdj > maxAdjSize) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid mesh exceeded adjacency allocation (%D)", maxAdjSize);
        }
      }
      else {
        i++;
      }
    }
    *adjSize = numAdj;
    ierr = ISRestoreIndices(aIS,&anchors);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetAdjacency"
/*@
  DMPlexGetAdjacency - Return all points adjacent to the given point

  Input Parameters:
+ dm - The DM object
. p  - The point
. adjSize - The maximum size of adj if it is non-NULL, or PETSC_DETERMINE
- adj - Either NULL so that the array is allocated, or an existing array with size adjSize

  Output Parameters:
+ adjSize - The number of adjacent points
- adj - The adjacent points

  Level: advanced

  Notes: The user must PetscFree the adj array if it was not passed in.

.seealso: DMPlexSetAdjacencyUseCone(), DMPlexSetAdjacencyUseClosure(), DMPlexDistribute(), DMCreateMatrix(), DMPlexPreallocateOperator()
@*/
PetscErrorCode DMPlexGetAdjacency(DM dm, PetscInt p, PetscInt *adjSize, PetscInt *adj[])
{
  DM_Plex       *mesh = (DM_Plex *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(adjSize,3);
  PetscValidPointer(adj,4);
  ierr = DMPlexGetAdjacency_Internal(dm, p, mesh->useCone, mesh->useClosure, mesh->useAnchors, adjSize, adj);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "DMPlexCreateTwoSidedProcessSF"
/*@
  DMPlexCreateTwoSidedProcessSF - Create an SF which just has process connectivity

  Collective on DM

  Input Parameters:
+ dm      - The DM
- sfPoint - The PetscSF which encodes point connectivity

  Output Parameters:
+ processRanks - A list of process neighbors, or NULL
- sfProcess    - An SF encoding the two-sided process connectivity, or NULL

  Level: developer

.seealso: PetscSFCreate(), DMPlexCreateProcessSF()
@*/
PetscErrorCode DMPlexCreateTwoSidedProcessSF(DM dm, PetscSF sfPoint, PetscSection rootRankSection, IS rootRanks, PetscSection leafRankSection, IS leafRanks, IS *processRanks, PetscSF *sfProcess)
{
  const PetscSFNode *remotePoints;
  PetscInt          *localPointsNew;
  PetscSFNode       *remotePointsNew;
  const PetscInt    *nranks;
  PetscInt          *ranksNew;
  PetscBT            neighbors;
  PetscInt           pStart, pEnd, p, numLeaves, l, numNeighbors, n;
  PetscMPIInt        numProcs, proc, rank;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(sfPoint, PETSCSF_CLASSID, 2);
  if (processRanks) {PetscValidPointer(processRanks, 3);}
  if (sfProcess)    {PetscValidPointer(sfProcess, 4);}
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject) dm), &numProcs);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject) dm), &rank);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(sfPoint, NULL, &numLeaves, NULL, &remotePoints);CHKERRQ(ierr);
  ierr = PetscBTCreate(numProcs, &neighbors);CHKERRQ(ierr);
  ierr = PetscBTMemzero(numProcs, neighbors);CHKERRQ(ierr);
  /* Compute root-to-leaf process connectivity */
  ierr = PetscSectionGetChart(rootRankSection, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = ISGetIndices(rootRanks, &nranks);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {
    PetscInt ndof, noff, n;

    ierr = PetscSectionGetDof(rootRankSection, p, &ndof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(rootRankSection, p, &noff);CHKERRQ(ierr);
    for (n = 0; n < ndof; ++n) {ierr = PetscBTSet(neighbors, nranks[noff+n]);}
  }
  ierr = ISRestoreIndices(rootRanks, &nranks);CHKERRQ(ierr);
  /* Compute leaf-to-neighbor process connectivity */
  ierr = PetscSectionGetChart(leafRankSection, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = ISGetIndices(leafRanks, &nranks);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {
    PetscInt ndof, noff, n;

    ierr = PetscSectionGetDof(leafRankSection, p, &ndof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(leafRankSection, p, &noff);CHKERRQ(ierr);
    for (n = 0; n < ndof; ++n) {ierr = PetscBTSet(neighbors, nranks[noff+n]);}
  }
  ierr = ISRestoreIndices(leafRanks, &nranks);CHKERRQ(ierr);
  /* Compute leaf-to-root process connectivity */
  for (l = 0; l < numLeaves; ++l) {PetscBTSet(neighbors, remotePoints[l].rank);}
  /* Calculate edges */
  PetscBTClear(neighbors, rank);
  for(proc = 0, numNeighbors = 0; proc < numProcs; ++proc) {if (PetscBTLookup(neighbors, proc)) ++numNeighbors;}
  ierr = PetscMalloc1(numNeighbors, &ranksNew);CHKERRQ(ierr);
  ierr = PetscMalloc1(numNeighbors, &localPointsNew);CHKERRQ(ierr);
  ierr = PetscMalloc1(numNeighbors, &remotePointsNew);CHKERRQ(ierr);
  for(proc = 0, n = 0; proc < numProcs; ++proc) {
    if (PetscBTLookup(neighbors, proc)) {
      ranksNew[n]              = proc;
      localPointsNew[n]        = proc;
      remotePointsNew[n].index = rank;
      remotePointsNew[n].rank  = proc;
      ++n;
    }
  }
  ierr = PetscBTDestroy(&neighbors);CHKERRQ(ierr);
  if (processRanks) {ierr = ISCreateGeneral(PetscObjectComm((PetscObject)dm), numNeighbors, ranksNew, PETSC_OWN_POINTER, processRanks);CHKERRQ(ierr);}
  else              {ierr = PetscFree(ranksNew);CHKERRQ(ierr);}
  if (sfProcess) {
    ierr = PetscSFCreate(PetscObjectComm((PetscObject)dm), sfProcess);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) *sfProcess, "Two-Sided Process SF");CHKERRQ(ierr);
    ierr = PetscSFSetFromOptions(*sfProcess);CHKERRQ(ierr);
    ierr = PetscSFSetGraph(*sfProcess, numProcs, numNeighbors, localPointsNew, PETSC_OWN_POINTER, remotePointsNew, PETSC_OWN_POINTER);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexDistributeOwnership"
/*@
  DMPlexDistributeOwnership - Compute owner information for shared points. This basically gets two-sided for an SF.

  Collective on DM

  Input Parameter:
. dm - The DM

  Output Parameters:
+ rootSection - The number of leaves for a given root point
. rootrank    - The rank of each edge into the root point
. leafSection - The number of processes sharing a given leaf point
- leafrank    - The rank of each process sharing a leaf point

  Level: developer

.seealso: DMPlexCreateOverlap()
@*/
PetscErrorCode DMPlexDistributeOwnership(DM dm, PetscSection rootSection, IS *rootrank, PetscSection leafSection, IS *leafrank)
{
  MPI_Comm        comm;
  PetscSF         sfPoint;
  const PetscInt *rootdegree;
  PetscInt       *myrank, *remoterank;
  PetscInt        pStart, pEnd, p, nedges;
  PetscMPIInt     rank;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = DMGetPointSF(dm, &sfPoint);CHKERRQ(ierr);
  /* Compute number of leaves for each root */
  ierr = PetscObjectSetName((PetscObject) rootSection, "Root Section");CHKERRQ(ierr);
  ierr = PetscSectionSetChart(rootSection, pStart, pEnd);CHKERRQ(ierr);
  ierr = PetscSFComputeDegreeBegin(sfPoint, &rootdegree);CHKERRQ(ierr);
  ierr = PetscSFComputeDegreeEnd(sfPoint, &rootdegree);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {ierr = PetscSectionSetDof(rootSection, p, rootdegree[p-pStart]);CHKERRQ(ierr);}
  ierr = PetscSectionSetUp(rootSection);CHKERRQ(ierr);
  /* Gather rank of each leaf to root */
  ierr = PetscSectionGetStorageSize(rootSection, &nedges);CHKERRQ(ierr);
  ierr = PetscMalloc1(pEnd-pStart, &myrank);CHKERRQ(ierr);
  ierr = PetscMalloc1(nedges,  &remoterank);CHKERRQ(ierr);
  for (p = 0; p < pEnd-pStart; ++p) myrank[p] = rank;
  ierr = PetscSFGatherBegin(sfPoint, MPIU_INT, myrank, remoterank);CHKERRQ(ierr);
  ierr = PetscSFGatherEnd(sfPoint, MPIU_INT, myrank, remoterank);CHKERRQ(ierr);
  ierr = PetscFree(myrank);CHKERRQ(ierr);
  ierr = ISCreateGeneral(comm, nedges, remoterank, PETSC_OWN_POINTER, rootrank);CHKERRQ(ierr);
  /* Distribute remote ranks to leaves */
  ierr = PetscObjectSetName((PetscObject) leafSection, "Leaf Section");CHKERRQ(ierr);
  ierr = DMPlexDistributeFieldIS(dm, sfPoint, rootSection, *rootrank, leafSection, leafrank);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexCreateOverlap"
/*@C
  DMPlexCreateOverlap - Compute owner information for shared points. This basically gets two-sided for an SF.

  Collective on DM

  Input Parameters:
+ dm          - The DM
. rootSection - The number of leaves for a given root point
. rootrank    - The rank of each edge into the root point
. leafSection - The number of processes sharing a given leaf point
- leafrank    - The rank of each process sharing a leaf point

  Output Parameters:
+ ovRootSection - The number of new overlap points for each neighboring process
. ovRootPoints  - The new overlap points for each neighboring process
. ovLeafSection - The number of new overlap points from each neighboring process
- ovLeafPoints  - The new overlap points from each neighboring process

  Level: developer

.seealso: DMPlexDistributeOwnership()
@*/
PetscErrorCode DMPlexCreateOverlap(DM dm, PetscSection rootSection, IS rootrank, PetscSection leafSection, IS leafrank, PetscSection ovRootSection, PetscSFNode **ovRootPoints, PetscSection ovLeafSection, PetscSFNode **ovLeafPoints)
{
  DMLabel            ovAdjByRank; /* A DMLabel containing all points adjacent to shared points, separated by rank (value in label) */
  PetscSF            sfPoint, sfProc;
  IS                 valueIS;
  const PetscSFNode *remote;
  const PetscInt    *local;
  const PetscInt    *nrank, *rrank, *neighbors;
  PetscInt          *adj = NULL;
  PetscInt           pStart, pEnd, p, sStart, sEnd, nleaves, l, numNeighbors, n, ovSize;
  PetscMPIInt        rank, numProcs;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject) dm), &numProcs);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject) dm), &rank);CHKERRQ(ierr);
  ierr = DMGetPointSF(dm, &sfPoint);CHKERRQ(ierr);
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(leafSection, &sStart, &sEnd);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(sfPoint, NULL, &nleaves, &local, &remote);CHKERRQ(ierr);
  ierr = DMLabelCreate("Overlap adjacency", &ovAdjByRank);CHKERRQ(ierr);
  /* Handle leaves: shared with the root point */
  for (l = 0; l < nleaves; ++l) {
    PetscInt adjSize = PETSC_DETERMINE, a;

    ierr = DMPlexGetAdjacency(dm, local[l], &adjSize, &adj);CHKERRQ(ierr);
    for (a = 0; a < adjSize; ++a) {ierr = DMLabelSetValue(ovAdjByRank, adj[a], remote[l].rank);CHKERRQ(ierr);}
  }
  ierr = ISGetIndices(rootrank, &rrank);CHKERRQ(ierr);
  ierr = ISGetIndices(leafrank, &nrank);CHKERRQ(ierr);
  /* Handle roots */
  for (p = pStart; p < pEnd; ++p) {
    PetscInt adjSize = PETSC_DETERMINE, neighbors = 0, noff, n, a;

    if ((p >= sStart) && (p < sEnd)) {
      /* Some leaves share a root with other leaves on different processes */
      ierr = PetscSectionGetDof(leafSection, p, &neighbors);CHKERRQ(ierr);
      if (neighbors) {
        ierr = PetscSectionGetOffset(leafSection, p, &noff);CHKERRQ(ierr);
        ierr = DMPlexGetAdjacency(dm, p, &adjSize, &adj);CHKERRQ(ierr);
        for (n = 0; n < neighbors; ++n) {
          const PetscInt remoteRank = nrank[noff+n];

          if (remoteRank == rank) continue;
          for (a = 0; a < adjSize; ++a) {ierr = DMLabelSetValue(ovAdjByRank, adj[a], remoteRank);CHKERRQ(ierr);}
        }
      }
    }
    /* Roots are shared with leaves */
    ierr = PetscSectionGetDof(rootSection, p, &neighbors);CHKERRQ(ierr);
    if (!neighbors) continue;
    ierr = PetscSectionGetOffset(rootSection, p, &noff);CHKERRQ(ierr);
    ierr = DMPlexGetAdjacency(dm, p, &adjSize, &adj);CHKERRQ(ierr);
    for (n = 0; n < neighbors; ++n) {
      const PetscInt remoteRank = rrank[noff+n];

      if (remoteRank == rank) continue;
      for (a = 0; a < adjSize; ++a) {ierr = DMLabelSetValue(ovAdjByRank, adj[a], remoteRank);CHKERRQ(ierr);}
    }
  }
  ierr = PetscFree(adj);CHKERRQ(ierr);
  ierr = ISRestoreIndices(rootrank, &rrank);CHKERRQ(ierr);
  ierr = ISRestoreIndices(leafrank, &nrank);CHKERRQ(ierr);
  {
    ierr = PetscViewerASCIISynchronizedAllow(PETSC_VIEWER_STDOUT_WORLD, PETSC_TRUE);CHKERRQ(ierr);
    ierr = DMLabelView(ovAdjByRank, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  /* Convert to (point, rank) and use actual owners */
  ierr = PetscSectionSetChart(ovRootSection, 0, numProcs);CHKERRQ(ierr);
  ierr = DMLabelGetValueIS(ovAdjByRank, &valueIS);CHKERRQ(ierr);
  ierr = ISGetLocalSize(valueIS, &numNeighbors);CHKERRQ(ierr);
  ierr = ISGetIndices(valueIS, &neighbors);CHKERRQ(ierr);
  for (n = 0; n < numNeighbors; ++n) {
    PetscInt numPoints;

    ierr = DMLabelGetStratumSize(ovAdjByRank, neighbors[n], &numPoints);CHKERRQ(ierr);
    ierr = PetscSectionAddDof(ovRootSection, neighbors[n], numPoints);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(ovRootSection);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(ovRootSection, &ovSize);CHKERRQ(ierr);
  ierr = PetscMalloc1(ovSize, ovRootPoints);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(sfPoint, NULL, &nleaves, &local, &remote);CHKERRQ(ierr);
  for (n = 0; n < numNeighbors; ++n) {
    IS              pointIS;
    const PetscInt *points;
    PetscInt        off, numPoints, p;

    ierr = PetscSectionGetOffset(ovRootSection, neighbors[n], &off);CHKERRQ(ierr);
    ierr = DMLabelGetStratumIS(ovAdjByRank, neighbors[n], &pointIS);CHKERRQ(ierr);
    ierr = ISGetLocalSize(pointIS, &numPoints);CHKERRQ(ierr);
    ierr = ISGetIndices(pointIS, &points);CHKERRQ(ierr);
    for (p = 0; p < numPoints; ++p) {
      ierr = PetscFindInt(points[p], nleaves, local, &l);CHKERRQ(ierr);
      if (l >= 0) {(*ovRootPoints)[off+p] = remote[l];}
      else        {(*ovRootPoints)[off+p].index = points[p]; (*ovRootPoints)[off+p].rank = rank;}
    }
    ierr = ISRestoreIndices(pointIS, &points);CHKERRQ(ierr);
    ierr = ISDestroy(&pointIS);CHKERRQ(ierr);
  }
  ierr = ISRestoreIndices(valueIS, &neighbors);CHKERRQ(ierr);
  ierr = ISDestroy(&valueIS);CHKERRQ(ierr);
  ierr = DMLabelDestroy(&ovAdjByRank);CHKERRQ(ierr);
  /* Make process SF */
  ierr = DMPlexCreateTwoSidedProcessSF(dm, sfPoint, rootSection, rootrank, leafSection, leafrank, NULL, &sfProc);CHKERRQ(ierr);
  {
    ierr = PetscSFView(sfProc, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  /* Communicate overlap */
  ierr = DMPlexDistributeData(dm, sfProc, ovRootSection, MPIU_2INT, (void *) *ovRootPoints, ovLeafSection, (void **) ovLeafPoints);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sfProc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexDistributeField"
/*@
  DMPlexDistributeField - Distribute field data to match a given PetscSF, usually the SF from mesh distribution

  Collective on DM

  Input Parameters:
+ dm - The DMPlex object
. pointSF - The PetscSF describing the communication pattern
. originalSection - The PetscSection for existing data layout
- originalVec - The existing data

  Output Parameters:
+ newSection - The PetscSF describing the new data layout
- newVec - The new data

  Level: developer

.seealso: DMPlexDistribute(), DMPlexDistributeFieldIS(), DMPlexDistributeData()
@*/
PetscErrorCode DMPlexDistributeField(DM dm, PetscSF pointSF, PetscSection originalSection, Vec originalVec, PetscSection newSection, Vec newVec)
{
  PetscSF        fieldSF;
  PetscInt      *remoteOffsets, fieldSize;
  PetscScalar   *originalValues, *newValues;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(DMPLEX_DistributeField,dm,0,0,0);CHKERRQ(ierr);
  ierr = PetscSFDistributeSection(pointSF, originalSection, &remoteOffsets, newSection);CHKERRQ(ierr);

  ierr = PetscSectionGetStorageSize(newSection, &fieldSize);CHKERRQ(ierr);
  ierr = VecSetSizes(newVec, fieldSize, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetType(newVec,dm->vectype);CHKERRQ(ierr);

  ierr = VecGetArray(originalVec, &originalValues);CHKERRQ(ierr);
  ierr = VecGetArray(newVec, &newValues);CHKERRQ(ierr);
  ierr = PetscSFCreateSectionSF(pointSF, originalSection, remoteOffsets, newSection, &fieldSF);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(fieldSF, MPIU_SCALAR, originalValues, newValues);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(fieldSF, MPIU_SCALAR, originalValues, newValues);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&fieldSF);CHKERRQ(ierr);
  ierr = VecRestoreArray(newVec, &newValues);CHKERRQ(ierr);
  ierr = VecRestoreArray(originalVec, &originalValues);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(DMPLEX_DistributeField,dm,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexDistributeFieldIS"
/*@
  DMPlexDistributeFieldIS - Distribute field data to match a given PetscSF, usually the SF from mesh distribution

  Collective on DM

  Input Parameters:
+ dm - The DMPlex object
. pointSF - The PetscSF describing the communication pattern
. originalSection - The PetscSection for existing data layout
- originalIS - The existing data

  Output Parameters:
+ newSection - The PetscSF describing the new data layout
- newIS - The new data

  Level: developer

.seealso: DMPlexDistribute(), DMPlexDistributeField(), DMPlexDistributeData()
@*/
PetscErrorCode DMPlexDistributeFieldIS(DM dm, PetscSF pointSF, PetscSection originalSection, IS originalIS, PetscSection newSection, IS *newIS)
{
  PetscSF         fieldSF;
  PetscInt       *newValues, *remoteOffsets, fieldSize;
  const PetscInt *originalValues;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(DMPLEX_DistributeField,dm,0,0,0);CHKERRQ(ierr);
  ierr = PetscSFDistributeSection(pointSF, originalSection, &remoteOffsets, newSection);CHKERRQ(ierr);

  ierr = PetscSectionGetStorageSize(newSection, &fieldSize);CHKERRQ(ierr);
  ierr = PetscMalloc(fieldSize * sizeof(PetscInt), &newValues);CHKERRQ(ierr);

  ierr = ISGetIndices(originalIS, &originalValues);CHKERRQ(ierr);
  ierr = PetscSFCreateSectionSF(pointSF, originalSection, remoteOffsets, newSection, &fieldSF);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(fieldSF, MPIU_INT, originalValues, newValues);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(fieldSF, MPIU_INT, originalValues, newValues);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&fieldSF);CHKERRQ(ierr);
  ierr = ISRestoreIndices(originalIS, &originalValues);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject) pointSF), fieldSize, newValues, PETSC_OWN_POINTER, newIS);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(DMPLEX_DistributeField,dm,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexDistributeData"
/*@
  DMPlexDistributeData - Distribute field data to match a given PetscSF, usually the SF from mesh distribution

  Collective on DM

  Input Parameters:
+ dm - The DMPlex object
. pointSF - The PetscSF describing the communication pattern
. originalSection - The PetscSection for existing data layout
. datatype - The type of data
- originalData - The existing data

  Output Parameters:
+ newSection - The PetscSection describing the new data layout
- newData - The new data

  Level: developer

.seealso: DMPlexDistribute(), DMPlexDistributeField()
@*/
PetscErrorCode DMPlexDistributeData(DM dm, PetscSF pointSF, PetscSection originalSection, MPI_Datatype datatype, void *originalData, PetscSection newSection, void **newData)
{
  PetscSF        fieldSF;
  PetscInt      *remoteOffsets, fieldSize;
  PetscMPIInt    dataSize;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(DMPLEX_DistributeData,dm,0,0,0);CHKERRQ(ierr);
  ierr = PetscSFDistributeSection(pointSF, originalSection, &remoteOffsets, newSection);CHKERRQ(ierr);

  ierr = PetscSectionGetStorageSize(newSection, &fieldSize);CHKERRQ(ierr);
  ierr = MPI_Type_size(datatype, &dataSize);CHKERRQ(ierr);
  ierr = PetscMalloc(fieldSize * dataSize, newData);CHKERRQ(ierr);

  ierr = PetscSFCreateSectionSF(pointSF, originalSection, remoteOffsets, newSection, &fieldSF);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(fieldSF, datatype, originalData, *newData);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(fieldSF, datatype, originalData, *newData);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&fieldSF);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(DMPLEX_DistributeData,dm,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexDistributeCones"
PetscErrorCode DMPlexDistributeCones(DM dm, PetscSF migrationSF, ISLocalToGlobalMapping renumbering, DM dmParallel)
{
  DM_Plex               *pmesh   = (DM_Plex*) (dmParallel)->data;
  MPI_Comm               comm;
  PetscSF                coneSF;
  PetscSection           originalConeSection, newConeSection;
  PetscInt              *remoteOffsets, *cones, *newCones, newConesSize;
  PetscBool              flg;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(dmParallel,4);
  ierr = PetscLogEventBegin(DMPLEX_DistributeCones,dm,0,0,0);CHKERRQ(ierr);

  /* Distribute cone section */
  ierr = PetscObjectGetComm((PetscObject)dm, &comm);CHKERRQ(ierr);
  ierr = DMPlexGetConeSection(dm, &originalConeSection);CHKERRQ(ierr);
  ierr = DMPlexGetConeSection(dmParallel, &newConeSection);CHKERRQ(ierr);
  ierr = PetscSFDistributeSection(migrationSF, originalConeSection, &remoteOffsets, newConeSection);CHKERRQ(ierr);
  ierr = DMSetUp(dmParallel);CHKERRQ(ierr);
  {
    PetscInt pStart, pEnd, p;

    ierr = PetscSectionGetChart(newConeSection, &pStart, &pEnd);CHKERRQ(ierr);
    for (p = pStart; p < pEnd; ++p) {
      PetscInt coneSize;
      ierr               = PetscSectionGetDof(newConeSection, p, &coneSize);CHKERRQ(ierr);
      pmesh->maxConeSize = PetscMax(pmesh->maxConeSize, coneSize);
    }
  }
  /* Communicate and renumber cones */
  ierr = PetscSFCreateSectionSF(migrationSF, originalConeSection, remoteOffsets, newConeSection, &coneSF);CHKERRQ(ierr);
  ierr = DMPlexGetCones(dm, &cones);CHKERRQ(ierr);
  ierr = DMPlexGetCones(dmParallel, &newCones);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(coneSF, MPIU_INT, cones, newCones);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(coneSF, MPIU_INT, cones, newCones);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(newConeSection, &newConesSize);CHKERRQ(ierr);
  ierr = ISGlobalToLocalMappingApplyBlock(renumbering, IS_GTOLM_MASK, newConesSize, newCones, NULL, newCones);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(((PetscObject) dm)->prefix, "-cones_view", &flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscPrintf(comm, "Serial Cone Section:\n");CHKERRQ(ierr);
    ierr = PetscSectionView(originalConeSection, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "Parallel Cone Section:\n");CHKERRQ(ierr);
    ierr = PetscSectionView(newConeSection, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscSFView(coneSF, NULL);CHKERRQ(ierr);
  }
  ierr = DMPlexGetConeOrientations(dm, &cones);CHKERRQ(ierr);
  ierr = DMPlexGetConeOrientations(dmParallel, &newCones);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(coneSF, MPIU_INT, cones, newCones);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(coneSF, MPIU_INT, cones, newCones);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&coneSF);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(DMPLEX_DistributeCones,dm,0,0,0);CHKERRQ(ierr);
  /* Create supports and stratify sieve */
  {
    PetscInt pStart, pEnd;

    ierr = PetscSectionGetChart(pmesh->coneSection, &pStart, &pEnd);CHKERRQ(ierr);
    ierr = PetscSectionSetChart(pmesh->supportSection, pStart, pEnd);CHKERRQ(ierr);
  }
  ierr = DMPlexSymmetrize(dmParallel);CHKERRQ(ierr);
  ierr = DMPlexStratify(dmParallel);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexDistribute"
/*@C
  DMPlexDistribute - Distributes the mesh and any associated sections.

  Not Collective

  Input Parameter:
+ dm  - The original DMPlex object
. partitioner - The partitioning package, or NULL for the default
- overlap - The overlap of partitions, 0 is the default

  Output Parameter:
+ sf - The PetscSF used for point distribution
- parallelMesh - The distributed DMPlex object, or NULL

  Note: If the mesh was not distributed, the return value is NULL.

  The user can control the definition of adjacency for the mesh using DMPlexGetAdjacencyUseCone() and
  DMPlexSetAdjacencyUseClosure(). They should choose the combination appropriate for the function
  representation on the mesh.

  Level: intermediate

.keywords: mesh, elements
.seealso: DMPlexCreate(), DMPlexDistributeByFace(), DMPlexSetAdjacencyUseCone(), DMPlexSetAdjacencyUseClosure()
@*/
PetscErrorCode DMPlexDistribute(DM dm, const char partitioner[], PetscInt overlap, PetscSF *sf, DM *dmParallel)
{
  DM_Plex               *mesh   = (DM_Plex*) dm->data, *pmesh;
  MPI_Comm               comm;
  PetscInt               dim, numRemoteRanks;
  IS                     origCellPart,        origPart,        cellPart,        part;
  PetscSection           origCellPartSection, origPartSection, cellPartSection, partSection;
  PetscSFNode           *remoteRanks;
  PetscSF                partSF, pointSF;
  ISLocalToGlobalMapping renumbering;
  PetscBool              flg;
  PetscMPIInt            rank, numProcs, p;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (sf) PetscValidPointer(sf,4);
  PetscValidPointer(dmParallel,5);

  ierr = PetscLogEventBegin(DMPLEX_Distribute,dm,0,0,0);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &numProcs);CHKERRQ(ierr);

  *dmParallel = NULL;
  if (numProcs == 1) PetscFunctionReturn(0);

  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  /* Create cell partition - We need to rewrite to use IS, use the MatPartition stuff */
  ierr = PetscLogEventBegin(PETSCPARTITIONER_Partition,dm,0,0,0);CHKERRQ(ierr);
  if (overlap > 1) SETERRQ(comm, PETSC_ERR_SUP, "Overlap > 1 not yet implemented");
  ierr = PetscSectionCreate(comm, &cellPartSection);CHKERRQ(ierr);
  ierr = PetscSectionCreate(comm, &origCellPartSection);CHKERRQ(ierr);
  ierr = PetscPartitionerPartition(mesh->partitioner, dm, overlap > 0 ? PETSC_TRUE : PETSC_FALSE, cellPartSection, &cellPart, origCellPartSection, &origCellPart);CHKERRQ(ierr);
  /* Create SF assuming a serial partition for all processes: Could check for IS length here */
  if (!rank) numRemoteRanks = numProcs;
  else       numRemoteRanks = 0;
  ierr = PetscMalloc1(numRemoteRanks, &remoteRanks);CHKERRQ(ierr);
  for (p = 0; p < numRemoteRanks; ++p) {
    remoteRanks[p].rank  = p;
    remoteRanks[p].index = 0;
  }
  ierr = PetscSFCreate(comm, &partSF);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(partSF, 1, numRemoteRanks, NULL, PETSC_OWN_POINTER, remoteRanks, PETSC_OWN_POINTER);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(((PetscObject) dm)->prefix, "-partition_view", &flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscPrintf(comm, "Cell Partition:\n");CHKERRQ(ierr);
    ierr = PetscSectionView(cellPartSection, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = ISView(cellPart, NULL);CHKERRQ(ierr);
    if (origCellPart) {
      ierr = PetscPrintf(comm, "Original Cell Partition:\n");CHKERRQ(ierr);
      ierr = PetscSectionView(origCellPartSection, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
      ierr = ISView(origCellPart, NULL);CHKERRQ(ierr);
    }
    ierr = PetscSFView(partSF, NULL);CHKERRQ(ierr);
  }
  /* Close the partition over the mesh */
  ierr = DMPlexCreatePartitionClosure(dm, cellPartSection, cellPart, &partSection, &part);CHKERRQ(ierr);
  ierr = ISDestroy(&cellPart);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&cellPartSection);CHKERRQ(ierr);
  /* Create new mesh */
  ierr  = DMPlexCreate(comm, dmParallel);CHKERRQ(ierr);
  ierr  = DMSetDimension(*dmParallel, dim);CHKERRQ(ierr);
  ierr  = PetscObjectSetName((PetscObject) *dmParallel, "Parallel Mesh");CHKERRQ(ierr);
  pmesh = (DM_Plex*) (*dmParallel)->data;
  /* Distribute sieve points and the global point numbering (replaces creating remote bases) */
  ierr = PetscSFConvertPartition(partSF, partSection, part, &renumbering, &pointSF);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscPrintf(comm, "Point Partition:\n");CHKERRQ(ierr);
    ierr = PetscSectionView(partSection, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = ISView(part, NULL);CHKERRQ(ierr);
    ierr = PetscSFView(pointSF, NULL);CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "Point Renumbering after partition:\n");CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingView(renumbering, NULL);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(PETSCPARTITIONER_Partition,dm,0,0,0);CHKERRQ(ierr);

  ierr = DMPlexDistributeCones(dm, pointSF, renumbering, *dmParallel);CHKERRQ(ierr);

  /* Create point SF for parallel mesh */
  ierr = PetscLogEventBegin(DMPLEX_DistributeSF,dm,0,0,0);CHKERRQ(ierr);
  {
    const PetscInt *leaves;
    PetscSFNode    *remotePoints, *rowners, *lowners;
    PetscInt        numRoots, numLeaves, numGhostPoints = 0, p, gp, *ghostPoints;
    PetscInt        pStart, pEnd;

    ierr = DMPlexGetChart(*dmParallel, &pStart, &pEnd);CHKERRQ(ierr);
    ierr = PetscSFGetGraph(pointSF, &numRoots, &numLeaves, &leaves, NULL);CHKERRQ(ierr);
    ierr = PetscMalloc2(numRoots,&rowners,numLeaves,&lowners);CHKERRQ(ierr);
    for (p=0; p<numRoots; p++) {
      rowners[p].rank  = -1;
      rowners[p].index = -1;
    }
    if (origCellPart) {
      /* Make sure points in the original partition are not assigned to other procs */
      const PetscInt *origPoints;

      ierr = DMPlexCreatePartitionClosure(dm, origCellPartSection, origCellPart, &origPartSection, &origPart);CHKERRQ(ierr);
      ierr = ISGetIndices(origPart, &origPoints);CHKERRQ(ierr);
      for (p = 0; p < numProcs; ++p) {
        PetscInt dof, off, d;

        ierr = PetscSectionGetDof(origPartSection, p, &dof);CHKERRQ(ierr);
        ierr = PetscSectionGetOffset(origPartSection, p, &off);CHKERRQ(ierr);
        for (d = off; d < off+dof; ++d) {
          rowners[origPoints[d]].rank = p;
        }
      }
      ierr = ISRestoreIndices(origPart, &origPoints);CHKERRQ(ierr);
      ierr = ISDestroy(&origPart);CHKERRQ(ierr);
      ierr = PetscSectionDestroy(&origPartSection);CHKERRQ(ierr);
    }
    ierr = ISDestroy(&origCellPart);CHKERRQ(ierr);
    ierr = PetscSectionDestroy(&origCellPartSection);CHKERRQ(ierr);

    ierr = PetscSFBcastBegin(pointSF, MPIU_2INT, rowners, lowners);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(pointSF, MPIU_2INT, rowners, lowners);CHKERRQ(ierr);
    for (p = 0; p < numLeaves; ++p) {
      if (lowners[p].rank < 0 || lowners[p].rank == rank) { /* Either put in a bid or we know we own it */
        lowners[p].rank  = rank;
        lowners[p].index = leaves ? leaves[p] : p;
      } else if (lowners[p].rank >= 0) { /* Point already claimed so flag so that MAXLOC does not listen to us */
        lowners[p].rank  = -2;
        lowners[p].index = -2;
      }
    }
    for (p=0; p<numRoots; p++) { /* Root must not participate in the rediction, flag so that MAXLOC does not use */
      rowners[p].rank  = -3;
      rowners[p].index = -3;
    }
    ierr = PetscSFReduceBegin(pointSF, MPIU_2INT, lowners, rowners, MPI_MAXLOC);CHKERRQ(ierr);
    ierr = PetscSFReduceEnd(pointSF, MPIU_2INT, lowners, rowners, MPI_MAXLOC);CHKERRQ(ierr);
    ierr = PetscSFBcastBegin(pointSF, MPIU_2INT, rowners, lowners);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(pointSF, MPIU_2INT, rowners, lowners);CHKERRQ(ierr);
    for (p = 0; p < numLeaves; ++p) {
      if (lowners[p].rank < 0 || lowners[p].index < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Cell partition corrupt: point not claimed");
      if (lowners[p].rank != rank) ++numGhostPoints;
    }
    ierr = PetscMalloc1(numGhostPoints,    &ghostPoints);CHKERRQ(ierr);
    ierr = PetscMalloc1(numGhostPoints, &remotePoints);CHKERRQ(ierr);
    for (p = 0, gp = 0; p < numLeaves; ++p) {
      if (lowners[p].rank != rank) {
        ghostPoints[gp]        = leaves ? leaves[p] : p;
        remotePoints[gp].rank  = lowners[p].rank;
        remotePoints[gp].index = lowners[p].index;
        ++gp;
      }
    }
    ierr = PetscFree2(rowners,lowners);CHKERRQ(ierr);
    ierr = PetscSFSetGraph((*dmParallel)->sf, pEnd - pStart, numGhostPoints, ghostPoints, PETSC_OWN_POINTER, remotePoints, PETSC_OWN_POINTER);CHKERRQ(ierr);
    ierr = PetscSFSetFromOptions((*dmParallel)->sf);CHKERRQ(ierr);
  }
  pmesh->useCone    = mesh->useCone;
  pmesh->useClosure = mesh->useClosure;
  pmesh->useAnchors = mesh->useAnchors;
  ierr = PetscLogEventEnd(DMPLEX_DistributeSF,dm,0,0,0);CHKERRQ(ierr);
  /* Distribute Coordinates */
  {
    PetscSection     originalCoordSection, newCoordSection;
    Vec              originalCoordinates, newCoordinates;
    PetscInt         bs;
    const char      *name;
    const PetscReal *maxCell, *L;

    ierr = DMGetCoordinateSection(dm, &originalCoordSection);CHKERRQ(ierr);
    ierr = DMGetCoordinateSection(*dmParallel, &newCoordSection);CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocal(dm, &originalCoordinates);CHKERRQ(ierr);
    if (originalCoordinates) {
      ierr = VecCreate(comm, &newCoordinates);CHKERRQ(ierr);
      ierr = PetscObjectGetName((PetscObject) originalCoordinates, &name);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject) newCoordinates, name);CHKERRQ(ierr);

      ierr = DMPlexDistributeField(dm, pointSF, originalCoordSection, originalCoordinates, newCoordSection, newCoordinates);CHKERRQ(ierr);
      ierr = DMSetCoordinatesLocal(*dmParallel, newCoordinates);CHKERRQ(ierr);
      ierr = VecGetBlockSize(originalCoordinates, &bs);CHKERRQ(ierr);
      ierr = VecSetBlockSize(newCoordinates, bs);CHKERRQ(ierr);
      ierr = VecDestroy(&newCoordinates);CHKERRQ(ierr);
    }
    ierr = DMGetPeriodicity(dm, &maxCell, &L);CHKERRQ(ierr);
    if (L) {ierr = DMSetPeriodicity(*dmParallel, maxCell, L);CHKERRQ(ierr);}
  }
  /* Distribute labels */
  ierr = PetscLogEventBegin(DMPLEX_DistributeLabels,dm,0,0,0);CHKERRQ(ierr);
  {
    DMLabel  next      = mesh->labels, newNext = pmesh->labels;
    PetscInt numLabels = 0, l;

    /* Bcast number of labels */
    while (next) {++numLabels; next = next->next;}
    ierr = MPI_Bcast(&numLabels, 1, MPIU_INT, 0, comm);CHKERRQ(ierr);
    next = mesh->labels;
    for (l = 0; l < numLabels; ++l) {
      DMLabel   labelNew;
      PetscBool isdepth;

      /* Skip "depth" because it is recreated */
      if (!rank) {ierr = PetscStrcmp(next->name, "depth", &isdepth);CHKERRQ(ierr);}
      ierr = MPI_Bcast(&isdepth, 1, MPIU_BOOL, 0, comm);CHKERRQ(ierr);
      if (isdepth) {if (!rank) next = next->next; continue;}
      ierr = DMLabelDistribute(next, partSection, part, renumbering, &labelNew);CHKERRQ(ierr);
      /* Insert into list */
      if (newNext) newNext->next = labelNew;
      else         pmesh->labels = labelNew;
      newNext = labelNew;
      if (!rank) next = next->next;
    }
  }
  ierr = PetscLogEventEnd(DMPLEX_DistributeLabels,dm,0,0,0);CHKERRQ(ierr);
  /* Setup hybrid structure */
  {
    const PetscInt *gpoints;
    PetscInt        depth, n, d;

    for (d = 0; d <= dim; ++d) {pmesh->hybridPointMax[d] = mesh->hybridPointMax[d];}
    ierr = MPI_Bcast(pmesh->hybridPointMax, dim+1, MPIU_INT, 0, comm);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingGetSize(renumbering, &n);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingGetIndices(renumbering, &gpoints);CHKERRQ(ierr);
    ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
    for (d = 0; d <= dim; ++d) {
      PetscInt pmax = pmesh->hybridPointMax[d], newmax = 0, pEnd, stratum[2], p;

      if (pmax < 0) continue;
      ierr = DMPlexGetDepthStratum(dm, d > depth ? depth : d, &stratum[0], &stratum[1]);CHKERRQ(ierr);
      ierr = DMPlexGetDepthStratum(*dmParallel, d, NULL, &pEnd);CHKERRQ(ierr);
      ierr = MPI_Bcast(stratum, 2, MPIU_INT, 0, comm);CHKERRQ(ierr);
      for (p = 0; p < n; ++p) {
        const PetscInt point = gpoints[p];

        if ((point >= stratum[0]) && (point < stratum[1]) && (point >= pmax)) ++newmax;
      }
      if (newmax > 0) pmesh->hybridPointMax[d] = pEnd - newmax;
      else            pmesh->hybridPointMax[d] = -1;
    }
    ierr = ISLocalToGlobalMappingRestoreIndices(renumbering, &gpoints);CHKERRQ(ierr);
  }
  /* Set up tree */
  {
    DM              refTree;
    PetscSection    origParentSection, newParentSection;
    PetscInt        *origParents, *origChildIDs;

    ierr = DMPlexGetReferenceTree(dm,&refTree);CHKERRQ(ierr);
    ierr = DMPlexSetReferenceTree(*dmParallel,refTree);CHKERRQ(ierr);
    ierr = DMPlexGetTree(dm,&origParentSection,&origParents,&origChildIDs,NULL,NULL);CHKERRQ(ierr);
    if (origParentSection) {
      PetscInt        pStart, pEnd;
      PetscInt        *newParents, *newChildIDs;
      PetscInt        *remoteOffsetsParents, newParentSize;
      PetscSF         parentSF;

      ierr = DMPlexGetChart(*dmParallel, &pStart, &pEnd);CHKERRQ(ierr);
      ierr = PetscSectionCreate(PetscObjectComm((PetscObject)*dmParallel),&newParentSection);CHKERRQ(ierr);
      ierr = PetscSectionSetChart(newParentSection,pStart,pEnd);CHKERRQ(ierr);
      ierr = PetscSFDistributeSection(pointSF, origParentSection, &remoteOffsetsParents, newParentSection);CHKERRQ(ierr);
      ierr = PetscSFCreateSectionSF(pointSF, origParentSection, remoteOffsetsParents, newParentSection, &parentSF);CHKERRQ(ierr);
      ierr = PetscSectionGetStorageSize(newParentSection,&newParentSize);CHKERRQ(ierr);
      ierr = PetscMalloc2(newParentSize,&newParents,newParentSize,&newChildIDs);CHKERRQ(ierr);
      ierr = PetscSFBcastBegin(parentSF, MPIU_INT, origParents, newParents);CHKERRQ(ierr);
      ierr = PetscSFBcastEnd(parentSF, MPIU_INT, origParents, newParents);CHKERRQ(ierr);
      ierr = PetscSFBcastBegin(parentSF, MPIU_INT, origChildIDs, newChildIDs);CHKERRQ(ierr);
      ierr = PetscSFBcastEnd(parentSF, MPIU_INT, origChildIDs, newChildIDs);CHKERRQ(ierr);
      ierr = ISGlobalToLocalMappingApplyBlock(renumbering,IS_GTOLM_MASK, newParentSize, newParents, NULL, newParents);CHKERRQ(ierr);
      ierr = PetscOptionsHasName(((PetscObject) dm)->prefix, "-parents_view", &flg);CHKERRQ(ierr);
      if (flg) {
        ierr = PetscPrintf(comm, "Serial Parent Section: \n");CHKERRQ(ierr);
        ierr = PetscSectionView(origParentSection, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
        ierr = PetscPrintf(comm, "Parallel Parent Section: \n");CHKERRQ(ierr);
        ierr = PetscSectionView(newParentSection, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
        ierr = PetscSFView(parentSF, NULL);CHKERRQ(ierr);
      }
      ierr = DMPlexSetTree(*dmParallel,newParentSection,newParents,newChildIDs);CHKERRQ(ierr);
      ierr = PetscSectionDestroy(&newParentSection);CHKERRQ(ierr);
      ierr = PetscFree2(newParents,newChildIDs);CHKERRQ(ierr);
      ierr = PetscSFDestroy(&parentSF);CHKERRQ(ierr);
    }
  }
  /* Cleanup Partition */
  ierr = ISLocalToGlobalMappingDestroy(&renumbering);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&partSF);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&partSection);CHKERRQ(ierr);
  ierr = ISDestroy(&part);CHKERRQ(ierr);
  /* Copy BC */
  ierr = DMPlexCopyBoundary(dm, *dmParallel);CHKERRQ(ierr);
  /* Cleanup */
  if (sf) {*sf = pointSF;}
  else    {ierr = PetscSFDestroy(&pointSF);CHKERRQ(ierr);}
  ierr = DMSetFromOptions(*dmParallel);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(DMPLEX_Distribute,dm,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
