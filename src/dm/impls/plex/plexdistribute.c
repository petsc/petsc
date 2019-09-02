#include <petsc/private/dmpleximpl.h>    /*I      "petscdmplex.h"   I*/
#include <petsc/private/dmlabelimpl.h>   /*I      "petscdmlabel.h"  I*/

/*@C
  DMPlexSetAdjacencyUser - Define adjacency in the mesh using a user-provided callback

  Input Parameters:
+ dm      - The DM object
. user    - The user callback, may be NULL (to clear the callback)
- ctx     - context for callback evaluation, may be NULL

  Level: advanced

  Notes:
     The caller of DMPlexGetAdjacency may need to arrange that a large enough array is available for the adjacency.

     Any setting here overrides other configuration of DMPlex adjacency determination.

.seealso: DMSetAdjacency(), DMPlexDistribute(), DMPlexPreallocateOperator(), DMPlexGetAdjacency(), DMPlexGetAdjacencyUser()
@*/
PetscErrorCode DMPlexSetAdjacencyUser(DM dm,PetscErrorCode (*user)(DM,PetscInt,PetscInt*,PetscInt[],void*),void *ctx)
{
  DM_Plex *mesh = (DM_Plex *)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  mesh->useradjacency = user;
  mesh->useradjacencyctx = ctx;
  PetscFunctionReturn(0);
}

/*@C
  DMPlexGetAdjacencyUser - get the user-defined adjacency callback

  Input Parameter:
. dm      - The DM object

  Output Parameters:
- user    - The user callback
- ctx     - context for callback evaluation

  Level: advanced

.seealso: DMSetAdjacency(), DMPlexDistribute(), DMPlexPreallocateOperator(), DMPlexGetAdjacency(), DMPlexSetAdjacencyUser()
@*/
PetscErrorCode DMPlexGetAdjacencyUser(DM dm, PetscErrorCode (**user)(DM,PetscInt,PetscInt*,PetscInt[],void*), void **ctx)
{
  DM_Plex *mesh = (DM_Plex *)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (user) *user = mesh->useradjacency;
  if (ctx) *ctx = mesh->useradjacencyctx;
  PetscFunctionReturn(0);
}

/*@
  DMPlexSetAdjacencyUseAnchors - Define adjacency in the mesh using the point-to-point constraints.

  Input Parameters:
+ dm      - The DM object
- useAnchors - Flag to use the constraints.  If PETSC_TRUE, then constrained points are omitted from DMPlexGetAdjacency(), and their anchor points appear in their place.

  Level: intermediate

.seealso: DMGetAdjacency(), DMSetAdjacency(), DMPlexDistribute(), DMPlexPreallocateOperator(), DMPlexSetAnchors()
@*/
PetscErrorCode DMPlexSetAdjacencyUseAnchors(DM dm, PetscBool useAnchors)
{
  DM_Plex *mesh = (DM_Plex *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  mesh->useAnchors = useAnchors;
  PetscFunctionReturn(0);
}

/*@
  DMPlexGetAdjacencyUseAnchors - Query whether adjacency in the mesh uses the point-to-point constraints.

  Input Parameter:
. dm      - The DM object

  Output Parameter:
. useAnchors - Flag to use the closure.  If PETSC_TRUE, then constrained points are omitted from DMPlexGetAdjacency(), and their anchor points appear in their place.

  Level: intermediate

.seealso: DMPlexSetAdjacencyUseAnchors(), DMSetAdjacency(), DMGetAdjacency(), DMPlexDistribute(), DMPlexPreallocateOperator(), DMPlexSetAnchors()
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

static PetscErrorCode DMPlexGetAdjacency_Cone_Internal(DM dm, PetscInt p, PetscInt *adjSize, PetscInt adj[])
{
  const PetscInt *cone = NULL;
  PetscInt        numAdj = 0, maxAdjSize = *adjSize, coneSize, c;
  PetscErrorCode  ierr;

  PetscFunctionBeginHot;
  ierr = DMPlexGetConeSize(dm, p, &coneSize);CHKERRQ(ierr);
  ierr = DMPlexGetCone(dm, p, &cone);CHKERRQ(ierr);
  for (c = 0; c <= coneSize; ++c) {
    const PetscInt  point   = !c ? p : cone[c-1];
    const PetscInt *support = NULL;
    PetscInt        supportSize, s, q;

    ierr = DMPlexGetSupportSize(dm, point, &supportSize);CHKERRQ(ierr);
    ierr = DMPlexGetSupport(dm, point, &support);CHKERRQ(ierr);
    for (s = 0; s < supportSize; ++s) {
      for (q = 0; q < numAdj || ((void)(adj[numAdj++] = support[s]),0); ++q) {
        if (support[s] == adj[q]) break;
      }
      if (numAdj > maxAdjSize) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid mesh exceeded adjacency allocation (%D)", maxAdjSize);
    }
  }
  *adjSize = numAdj;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexGetAdjacency_Support_Internal(DM dm, PetscInt p, PetscInt *adjSize, PetscInt adj[])
{
  const PetscInt *support = NULL;
  PetscInt        numAdj   = 0, maxAdjSize = *adjSize, supportSize, s;
  PetscErrorCode  ierr;

  PetscFunctionBeginHot;
  ierr = DMPlexGetSupportSize(dm, p, &supportSize);CHKERRQ(ierr);
  ierr = DMPlexGetSupport(dm, p, &support);CHKERRQ(ierr);
  for (s = 0; s <= supportSize; ++s) {
    const PetscInt  point = !s ? p : support[s-1];
    const PetscInt *cone  = NULL;
    PetscInt        coneSize, c, q;

    ierr = DMPlexGetConeSize(dm, point, &coneSize);CHKERRQ(ierr);
    ierr = DMPlexGetCone(dm, point, &cone);CHKERRQ(ierr);
    for (c = 0; c < coneSize; ++c) {
      for (q = 0; q < numAdj || ((void)(adj[numAdj++] = cone[c]),0); ++q) {
        if (cone[c] == adj[q]) break;
      }
      if (numAdj > maxAdjSize) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid mesh exceeded adjacency allocation (%D)", maxAdjSize);
    }
  }
  *adjSize = numAdj;
  PetscFunctionReturn(0);
}

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
      for (q = 0; q < numAdj || ((void)(adj[numAdj++] = closure[c]),0); ++q) {
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

PetscErrorCode DMPlexGetAdjacency_Internal(DM dm, PetscInt p, PetscBool useCone, PetscBool useTransitiveClosure, PetscBool useAnchors, PetscInt *adjSize, PetscInt *adj[])
{
  static PetscInt asiz = 0;
  PetscInt maxAnchors = 1;
  PetscInt aStart = -1, aEnd = -1;
  PetscInt maxAdjSize;
  PetscSection aSec = NULL;
  IS aIS = NULL;
  const PetscInt *anchors;
  DM_Plex *mesh = (DM_Plex *)dm->data;
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
  if (mesh->useradjacency) {
    ierr = mesh->useradjacency(dm, p, adjSize, *adj, mesh->useradjacencyctx);CHKERRQ(ierr);
  } else if (useTransitiveClosure) {
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
          for (q = 0; q < numAdj || ((void)(orig[numAdj++] = anchors[aOff+s]),0); ++q) {
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

  Notes:
    The user must PetscFree the adj array if it was not passed in.

.seealso: DMSetAdjacency(), DMPlexDistribute(), DMCreateMatrix(), DMPlexPreallocateOperator()
@*/
PetscErrorCode DMPlexGetAdjacency(DM dm, PetscInt p, PetscInt *adjSize, PetscInt *adj[])
{
  PetscBool      useCone, useClosure, useAnchors;
  PetscErrorCode ierr;

  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(adjSize,3);
  PetscValidPointer(adj,4);
  ierr = DMGetBasicAdjacency(dm, &useCone, &useClosure);CHKERRQ(ierr);
  ierr = DMPlexGetAdjacencyUseAnchors(dm, &useAnchors);CHKERRQ(ierr);
  ierr = DMPlexGetAdjacency_Internal(dm, p, useCone, useClosure, useAnchors, adjSize, adj);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMPlexCreateTwoSidedProcessSF - Create an SF which just has process connectivity

  Collective on dm

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
  PetscMPIInt        size, proc, rank;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(sfPoint, PETSCSF_CLASSID, 2);
  if (processRanks) {PetscValidPointer(processRanks, 3);}
  if (sfProcess)    {PetscValidPointer(sfProcess, 4);}
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject) dm), &size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject) dm), &rank);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(sfPoint, NULL, &numLeaves, NULL, &remotePoints);CHKERRQ(ierr);
  ierr = PetscBTCreate(size, &neighbors);CHKERRQ(ierr);
  ierr = PetscBTMemzero(size, neighbors);CHKERRQ(ierr);
  /* Compute root-to-leaf process connectivity */
  ierr = PetscSectionGetChart(rootRankSection, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = ISGetIndices(rootRanks, &nranks);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {
    PetscInt ndof, noff, n;

    ierr = PetscSectionGetDof(rootRankSection, p, &ndof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(rootRankSection, p, &noff);CHKERRQ(ierr);
    for (n = 0; n < ndof; ++n) {ierr = PetscBTSet(neighbors, nranks[noff+n]);CHKERRQ(ierr);}
  }
  ierr = ISRestoreIndices(rootRanks, &nranks);CHKERRQ(ierr);
  /* Compute leaf-to-neighbor process connectivity */
  ierr = PetscSectionGetChart(leafRankSection, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = ISGetIndices(leafRanks, &nranks);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {
    PetscInt ndof, noff, n;

    ierr = PetscSectionGetDof(leafRankSection, p, &ndof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(leafRankSection, p, &noff);CHKERRQ(ierr);
    for (n = 0; n < ndof; ++n) {ierr = PetscBTSet(neighbors, nranks[noff+n]);CHKERRQ(ierr);}
  }
  ierr = ISRestoreIndices(leafRanks, &nranks);CHKERRQ(ierr);
  /* Compute leaf-to-root process connectivity */
  for (l = 0; l < numLeaves; ++l) {PetscBTSet(neighbors, remotePoints[l].rank);}
  /* Calculate edges */
  PetscBTClear(neighbors, rank);
  for(proc = 0, numNeighbors = 0; proc < size; ++proc) {if (PetscBTLookup(neighbors, proc)) ++numNeighbors;}
  ierr = PetscMalloc1(numNeighbors, &ranksNew);CHKERRQ(ierr);
  ierr = PetscMalloc1(numNeighbors, &localPointsNew);CHKERRQ(ierr);
  ierr = PetscMalloc1(numNeighbors, &remotePointsNew);CHKERRQ(ierr);
  for(proc = 0, n = 0; proc < size; ++proc) {
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
    ierr = PetscSFSetGraph(*sfProcess, size, numNeighbors, localPointsNew, PETSC_OWN_POINTER, remotePointsNew, PETSC_OWN_POINTER);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
  DMPlexDistributeOwnership - Compute owner information for shared points. This basically gets two-sided for an SF.

  Collective on dm

  Input Parameter:
. dm - The DM

  Output Parameters:
+ rootSection - The number of leaves for a given root point
. rootrank    - The rank of each edge into the root point
. leafSection - The number of processes sharing a given leaf point
- leafrank    - The rank of each process sharing a leaf point

  Level: developer

.seealso: DMPlexCreateOverlapLabel(), DMPlexDistribute(), DMPlexDistributeOverlap()
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

/*@C
  DMPlexCreateOverlapLabel - Compute owner information for shared points. This basically gets two-sided for an SF.

  Collective on dm

  Input Parameters:
+ dm          - The DM
. levels      - Number of overlap levels
. rootSection - The number of leaves for a given root point
. rootrank    - The rank of each edge into the root point
. leafSection - The number of processes sharing a given leaf point
- leafrank    - The rank of each process sharing a leaf point

  Output Parameters:
. ovLabel     - DMLabel containing remote overlap contributions as point/rank pairings

  Level: developer

.seealso: DMPlexDistributeOwnership(), DMPlexDistribute()
@*/
PetscErrorCode DMPlexCreateOverlapLabel(DM dm, PetscInt levels, PetscSection rootSection, IS rootrank, PetscSection leafSection, IS leafrank, DMLabel *ovLabel)
{
  MPI_Comm           comm;
  DMLabel            ovAdjByRank; /* A DMLabel containing all points adjacent to shared points, separated by rank (value in label) */
  PetscSF            sfPoint;
  const PetscSFNode *remote;
  const PetscInt    *local;
  const PetscInt    *nrank, *rrank;
  PetscInt          *adj = NULL;
  PetscInt           pStart, pEnd, p, sStart, sEnd, nleaves, l;
  PetscMPIInt        rank, size;
  PetscBool          flg;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = DMGetPointSF(dm, &sfPoint);CHKERRQ(ierr);
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(leafSection, &sStart, &sEnd);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(sfPoint, NULL, &nleaves, &local, &remote);CHKERRQ(ierr);
  ierr = DMLabelCreate(PETSC_COMM_SELF, "Overlap adjacency", &ovAdjByRank);CHKERRQ(ierr);
  /* Handle leaves: shared with the root point */
  for (l = 0; l < nleaves; ++l) {
    PetscInt adjSize = PETSC_DETERMINE, a;

    ierr = DMPlexGetAdjacency(dm, local ? local[l] : l, &adjSize, &adj);CHKERRQ(ierr);
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
  /* Add additional overlap levels */
  for (l = 1; l < levels; l++) {
    /* Propagate point donations over SF to capture remote connections */
    ierr = DMPlexPartitionLabelPropagate(dm, ovAdjByRank);CHKERRQ(ierr);
    /* Add next level of point donations to the label */
    ierr = DMPlexPartitionLabelAdjacency(dm, ovAdjByRank);CHKERRQ(ierr);
  }
  /* We require the closure in the overlap */
  ierr = DMPlexPartitionLabelClosure(dm, ovAdjByRank);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(((PetscObject) dm)->options,((PetscObject) dm)->prefix, "-overlap_view", &flg);CHKERRQ(ierr);
  if (flg) {
    PetscViewer viewer;
    ierr = PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)dm), &viewer);CHKERRQ(ierr);
    ierr = DMLabelView(ovAdjByRank, viewer);CHKERRQ(ierr);
  }
  /* Invert sender to receiver label */
  ierr = DMLabelCreate(PETSC_COMM_SELF, "Overlap label", ovLabel);CHKERRQ(ierr);
  ierr = DMPlexPartitionLabelInvert(dm, ovAdjByRank, NULL, *ovLabel);CHKERRQ(ierr);
  /* Add owned points, except for shared local points */
  for (p = pStart; p < pEnd; ++p) {ierr = DMLabelSetValue(*ovLabel, p, rank);CHKERRQ(ierr);}
  for (l = 0; l < nleaves; ++l) {
    ierr = DMLabelClearValue(*ovLabel, local[l], rank);CHKERRQ(ierr);
    ierr = DMLabelSetValue(*ovLabel, remote[l].index, remote[l].rank);CHKERRQ(ierr);
  }
  /* Clean up */
  ierr = DMLabelDestroy(&ovAdjByRank);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMPlexCreateOverlapMigrationSF - Create an SF describing the new mesh distribution to make the overlap described by the input SF

  Collective on dm

  Input Parameters:
+ dm          - The DM
- overlapSF   - The SF mapping ghost points in overlap to owner points on other processes

  Output Parameters:
. migrationSF - An SF that maps original points in old locations to points in new locations

  Level: developer

.seealso: DMPlexCreateOverlapLabel(), DMPlexDistributeOverlap(), DMPlexDistribute()
@*/
PetscErrorCode DMPlexCreateOverlapMigrationSF(DM dm, PetscSF overlapSF, PetscSF *migrationSF)
{
  MPI_Comm           comm;
  PetscMPIInt        rank, size;
  PetscInt           d, dim, p, pStart, pEnd, nroots, nleaves, newLeaves, point, numSharedPoints;
  PetscInt          *pointDepths, *remoteDepths, *ilocal;
  PetscInt          *depthRecv, *depthShift, *depthIdx;
  PetscSFNode       *iremote;
  PetscSF            pointSF;
  const PetscInt    *sharedLocal;
  const PetscSFNode *overlapRemote, *sharedRemote;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = PetscObjectGetComm((PetscObject)dm, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);

  /* Before building the migration SF we need to know the new stratum offsets */
  ierr = PetscSFGetGraph(overlapSF, &nroots, &nleaves, NULL, &overlapRemote);CHKERRQ(ierr);
  ierr = PetscMalloc2(nroots, &pointDepths, nleaves, &remoteDepths);CHKERRQ(ierr);
  for (d=0; d<dim+1; d++) {
    ierr = DMPlexGetDepthStratum(dm, d, &pStart, &pEnd);CHKERRQ(ierr);
    for (p=pStart; p<pEnd; p++) pointDepths[p] = d;
  }
  for (p=0; p<nleaves; p++) remoteDepths[p] = -1;
  ierr = PetscSFBcastBegin(overlapSF, MPIU_INT, pointDepths, remoteDepths);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(overlapSF, MPIU_INT, pointDepths, remoteDepths);CHKERRQ(ierr);

  /* Count recevied points in each stratum and compute the internal strata shift */
  ierr = PetscMalloc3(dim+1, &depthRecv, dim+1, &depthShift, dim+1, &depthIdx);CHKERRQ(ierr);
  for (d=0; d<dim+1; d++) depthRecv[d]=0;
  for (p=0; p<nleaves; p++) depthRecv[remoteDepths[p]]++;
  depthShift[dim] = 0;
  for (d=0; d<dim; d++) depthShift[d] = depthRecv[dim];
  for (d=1; d<dim; d++) depthShift[d] += depthRecv[0];
  for (d=dim-2; d>0; d--) depthShift[d] += depthRecv[d+1];
  for (d=0; d<dim+1; d++) {
    ierr = DMPlexGetDepthStratum(dm, d, &pStart, &pEnd);CHKERRQ(ierr);
    depthIdx[d] = pStart + depthShift[d];
  }

  /* Form the overlap SF build an SF that describes the full overlap migration SF */
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  newLeaves = pEnd - pStart + nleaves;
  ierr = PetscMalloc1(newLeaves, &ilocal);CHKERRQ(ierr);
  ierr = PetscMalloc1(newLeaves, &iremote);CHKERRQ(ierr);
  /* First map local points to themselves */
  for (d=0; d<dim+1; d++) {
    ierr = DMPlexGetDepthStratum(dm, d, &pStart, &pEnd);CHKERRQ(ierr);
    for (p=pStart; p<pEnd; p++) {
      point = p + depthShift[d];
      ilocal[point] = point;
      iremote[point].index = p;
      iremote[point].rank = rank;
      depthIdx[d]++;
    }
  }

  /* Add in the remote roots for currently shared points */
  ierr = DMGetPointSF(dm, &pointSF);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(pointSF, NULL, &numSharedPoints, &sharedLocal, &sharedRemote);CHKERRQ(ierr);
  for (d=0; d<dim+1; d++) {
    ierr = DMPlexGetDepthStratum(dm, d, &pStart, &pEnd);CHKERRQ(ierr);
    for (p=0; p<numSharedPoints; p++) {
      if (pStart <= sharedLocal[p] && sharedLocal[p] < pEnd) {
        point = sharedLocal[p] + depthShift[d];
        iremote[point].index = sharedRemote[p].index;
        iremote[point].rank = sharedRemote[p].rank;
      }
    }
  }

  /* Now add the incoming overlap points */
  for (p=0; p<nleaves; p++) {
    point = depthIdx[remoteDepths[p]];
    ilocal[point] = point;
    iremote[point].index = overlapRemote[p].index;
    iremote[point].rank = overlapRemote[p].rank;
    depthIdx[remoteDepths[p]]++;
  }
  ierr = PetscFree2(pointDepths,remoteDepths);CHKERRQ(ierr);

  ierr = PetscSFCreate(comm, migrationSF);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *migrationSF, "Overlap Migration SF");CHKERRQ(ierr);
  ierr = PetscSFSetFromOptions(*migrationSF);CHKERRQ(ierr);
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(*migrationSF, pEnd-pStart, newLeaves, ilocal, PETSC_OWN_POINTER, iremote, PETSC_OWN_POINTER);CHKERRQ(ierr);

  ierr = PetscFree3(depthRecv, depthShift, depthIdx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMPlexStratifyMigrationSF - Rearrange the leaves of a migration sf for stratification.

  Input Parameter:
+ dm          - The DM
- sf          - A star forest with non-ordered leaves, usually defining a DM point migration

  Output Parameter:
. migrationSF - A star forest with added leaf indirection that ensures the resulting DM is stratified

  Level: developer

.seealso: DMPlexPartitionLabelCreateSF(), DMPlexDistribute(), DMPlexDistributeOverlap()
@*/
PetscErrorCode DMPlexStratifyMigrationSF(DM dm, PetscSF sf, PetscSF *migrationSF)
{
  MPI_Comm           comm;
  PetscMPIInt        rank, size;
  PetscInt           d, ldepth, depth, p, pStart, pEnd, nroots, nleaves;
  PetscInt          *pointDepths, *remoteDepths, *ilocal;
  PetscInt          *depthRecv, *depthShift, *depthIdx;
  PetscInt           hybEnd[4];
  const PetscSFNode *iremote;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  ierr = DMPlexGetDepth(dm, &ldepth);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(&ldepth, &depth, 1, MPIU_INT, MPI_MAX, comm);CHKERRQ(ierr);
  if ((ldepth >= 0) && (depth != ldepth)) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Inconsistent Plex depth %d != %d", ldepth, depth);
  ierr = PetscLogEventBegin(DMPLEX_PartStratSF,dm,0,0,0);CHKERRQ(ierr);

  /* Before building the migration SF we need to know the new stratum offsets */
  ierr = PetscSFGetGraph(sf, &nroots, &nleaves, NULL, &iremote);CHKERRQ(ierr);
  ierr = PetscMalloc2(nroots, &pointDepths, nleaves, &remoteDepths);CHKERRQ(ierr);
  ierr = DMPlexGetHybridBounds(dm,&hybEnd[depth],&hybEnd[PetscMax(depth-1,0)],&hybEnd[1],&hybEnd[0]);CHKERRQ(ierr);
  for (d = 0; d < depth+1; ++d) {
    ierr = DMPlexGetDepthStratum(dm, d, &pStart, &pEnd);CHKERRQ(ierr);
    for (p = pStart; p < pEnd; ++p) {
      if (hybEnd[d] >= 0 && p >= hybEnd[d]) { /* put in a separate value for hybrid points */
        pointDepths[p] = 2 * d;
      } else {
        pointDepths[p] = 2 * d + 1;
      }
    }
  }
  for (p = 0; p < nleaves; ++p) remoteDepths[p] = -1;
  ierr = PetscSFBcastBegin(sf, MPIU_INT, pointDepths, remoteDepths);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(sf, MPIU_INT, pointDepths, remoteDepths);CHKERRQ(ierr);
  /* Count received points in each stratum and compute the internal strata shift */
  ierr = PetscMalloc3(2*(depth+1), &depthRecv, 2*(depth+1), &depthShift, 2*(depth+1), &depthIdx);CHKERRQ(ierr);
  for (d = 0; d < 2*(depth+1); ++d) depthRecv[d] = 0;
  for (p = 0; p < nleaves; ++p) depthRecv[remoteDepths[p]]++;
  depthShift[2*depth+1] = 0;
  for (d = 0; d < 2*depth+1; ++d) depthShift[d] = depthRecv[2 * depth + 1];
  for (d = 0; d < 2*depth; ++d) depthShift[d] += depthRecv[2 * depth];
  depthShift[0] += depthRecv[1];
  for (d = 2; d < 2*depth; ++d) depthShift[d] += depthRecv[1];
  for (d = 2; d < 2*depth; ++d) depthShift[d] += depthRecv[0];
  for (d = 2 * depth-1; d > 2; --d) {
    PetscInt e;

    for (e = d -1; e > 1; --e) depthShift[e] += depthRecv[d];
  }
  for (d = 0; d < 2*(depth+1); ++d) {depthIdx[d] = 0;}
  /* Derive a new local permutation based on stratified indices */
  ierr = PetscMalloc1(nleaves, &ilocal);CHKERRQ(ierr);
  for (p = 0; p < nleaves; ++p) {
    const PetscInt dep = remoteDepths[p];

    ilocal[p] = depthShift[dep] + depthIdx[dep];
    depthIdx[dep]++;
  }
  ierr = PetscSFCreate(comm, migrationSF);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *migrationSF, "Migration SF");CHKERRQ(ierr);
  ierr = PetscSFSetGraph(*migrationSF, nroots, nleaves, ilocal, PETSC_OWN_POINTER, iremote, PETSC_COPY_VALUES);CHKERRQ(ierr);
  ierr = PetscFree2(pointDepths,remoteDepths);CHKERRQ(ierr);
  ierr = PetscFree3(depthRecv, depthShift, depthIdx);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(DMPLEX_PartStratSF,dm,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMPlexDistributeField - Distribute field data to match a given PetscSF, usually the SF from mesh distribution

  Collective on dm

  Input Parameters:
+ dm - The DMPlex object
. pointSF - The PetscSF describing the communication pattern
. originalSection - The PetscSection for existing data layout
- originalVec - The existing data in a local vector

  Output Parameters:
+ newSection - The PetscSF describing the new data layout
- newVec - The new data in a local vector

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
  ierr = PetscFree(remoteOffsets);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(fieldSF, MPIU_SCALAR, originalValues, newValues);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(fieldSF, MPIU_SCALAR, originalValues, newValues);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&fieldSF);CHKERRQ(ierr);
  ierr = VecRestoreArray(newVec, &newValues);CHKERRQ(ierr);
  ierr = VecRestoreArray(originalVec, &originalValues);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(DMPLEX_DistributeField,dm,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMPlexDistributeFieldIS - Distribute field data to match a given PetscSF, usually the SF from mesh distribution

  Collective on dm

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
  ierr = PetscMalloc1(fieldSize, &newValues);CHKERRQ(ierr);

  ierr = ISGetIndices(originalIS, &originalValues);CHKERRQ(ierr);
  ierr = PetscSFCreateSectionSF(pointSF, originalSection, remoteOffsets, newSection, &fieldSF);CHKERRQ(ierr);
  ierr = PetscFree(remoteOffsets);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(fieldSF, MPIU_INT, (PetscInt *) originalValues, newValues);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(fieldSF, MPIU_INT, (PetscInt *) originalValues, newValues);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&fieldSF);CHKERRQ(ierr);
  ierr = ISRestoreIndices(originalIS, &originalValues);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject) pointSF), fieldSize, newValues, PETSC_OWN_POINTER, newIS);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(DMPLEX_DistributeField,dm,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMPlexDistributeData - Distribute field data to match a given PetscSF, usually the SF from mesh distribution

  Collective on dm

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
  ierr = PetscFree(remoteOffsets);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(fieldSF, datatype, originalData, *newData);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(fieldSF, datatype, originalData, *newData);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&fieldSF);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(DMPLEX_DistributeData,dm,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexDistributeCones(DM dm, PetscSF migrationSF, ISLocalToGlobalMapping original, ISLocalToGlobalMapping renumbering, DM dmParallel)
{
  DM_Plex               *pmesh = (DM_Plex*) (dmParallel)->data;
  MPI_Comm               comm;
  PetscSF                coneSF;
  PetscSection           originalConeSection, newConeSection;
  PetscInt              *remoteOffsets, *cones, *globCones, *newCones, newConesSize;
  PetscBool              flg;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(dmParallel, DM_CLASSID, 5);
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
  ierr = PetscFree(remoteOffsets);CHKERRQ(ierr);
  ierr = DMPlexGetCones(dm, &cones);CHKERRQ(ierr);
  if (original) {
    PetscInt numCones;

    ierr = PetscSectionGetStorageSize(originalConeSection,&numCones);CHKERRQ(ierr);
    ierr = PetscMalloc1(numCones,&globCones);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingApplyBlock(original, numCones, cones, globCones);CHKERRQ(ierr);
  } else {
    globCones = cones;
  }
  ierr = DMPlexGetCones(dmParallel, &newCones);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(coneSF, MPIU_INT, globCones, newCones);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(coneSF, MPIU_INT, globCones, newCones);CHKERRQ(ierr);
  if (original) {
    ierr = PetscFree(globCones);CHKERRQ(ierr);
  }
  ierr = PetscSectionGetStorageSize(newConeSection, &newConesSize);CHKERRQ(ierr);
  ierr = ISGlobalToLocalMappingApplyBlock(renumbering, IS_GTOLM_MASK, newConesSize, newCones, NULL, newCones);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
  {
    PetscInt  p;
    PetscBool valid = PETSC_TRUE;
    for (p = 0; p < newConesSize; ++p) {
      if (newCones[p] < 0) {valid = PETSC_FALSE; ierr = PetscPrintf(PETSC_COMM_SELF, "[%d] Point %D not in overlap SF\n", PetscGlobalRank,p);CHKERRQ(ierr);}
    }
    if (!valid) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid global to local map");
  }
#endif
  ierr = PetscOptionsHasName(((PetscObject) dm)->options,((PetscObject) dm)->prefix, "-cones_view", &flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscPrintf(comm, "Serial Cone Section:\n");CHKERRQ(ierr);
    ierr = PetscSectionView(originalConeSection, PETSC_VIEWER_STDOUT_(comm));CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "Parallel Cone Section:\n");CHKERRQ(ierr);
    ierr = PetscSectionView(newConeSection, PETSC_VIEWER_STDOUT_(comm));CHKERRQ(ierr);
    ierr = PetscSFView(coneSF, NULL);CHKERRQ(ierr);
  }
  ierr = DMPlexGetConeOrientations(dm, &cones);CHKERRQ(ierr);
  ierr = DMPlexGetConeOrientations(dmParallel, &newCones);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(coneSF, MPIU_INT, cones, newCones);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(coneSF, MPIU_INT, cones, newCones);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&coneSF);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(DMPLEX_DistributeCones,dm,0,0,0);CHKERRQ(ierr);
  /* Create supports and stratify DMPlex */
  {
    PetscInt pStart, pEnd;

    ierr = PetscSectionGetChart(pmesh->coneSection, &pStart, &pEnd);CHKERRQ(ierr);
    ierr = PetscSectionSetChart(pmesh->supportSection, pStart, pEnd);CHKERRQ(ierr);
  }
  ierr = DMPlexSymmetrize(dmParallel);CHKERRQ(ierr);
  ierr = DMPlexStratify(dmParallel);CHKERRQ(ierr);
  {
    PetscBool useCone, useClosure, useAnchors;

    ierr = DMGetBasicAdjacency(dm, &useCone, &useClosure);CHKERRQ(ierr);
    ierr = DMSetBasicAdjacency(dmParallel, useCone, useClosure);CHKERRQ(ierr);
    ierr = DMPlexGetAdjacencyUseAnchors(dm, &useAnchors);CHKERRQ(ierr);
    ierr = DMPlexSetAdjacencyUseAnchors(dmParallel, useAnchors);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexDistributeCoordinates(DM dm, PetscSF migrationSF, DM dmParallel)
{
  MPI_Comm         comm;
  PetscSection     originalCoordSection, newCoordSection;
  Vec              originalCoordinates, newCoordinates;
  PetscInt         bs;
  PetscBool        isper;
  const char      *name;
  const PetscReal *maxCell, *L;
  const DMBoundaryType *bd;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(dmParallel, DM_CLASSID, 3);

  ierr = PetscObjectGetComm((PetscObject)dm, &comm);CHKERRQ(ierr);
  ierr = DMGetCoordinateSection(dm, &originalCoordSection);CHKERRQ(ierr);
  ierr = DMGetCoordinateSection(dmParallel, &newCoordSection);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &originalCoordinates);CHKERRQ(ierr);
  if (originalCoordinates) {
    ierr = VecCreate(PETSC_COMM_SELF, &newCoordinates);CHKERRQ(ierr);
    ierr = PetscObjectGetName((PetscObject) originalCoordinates, &name);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) newCoordinates, name);CHKERRQ(ierr);

    ierr = DMPlexDistributeField(dm, migrationSF, originalCoordSection, originalCoordinates, newCoordSection, newCoordinates);CHKERRQ(ierr);
    ierr = DMSetCoordinatesLocal(dmParallel, newCoordinates);CHKERRQ(ierr);
    ierr = VecGetBlockSize(originalCoordinates, &bs);CHKERRQ(ierr);
    ierr = VecSetBlockSize(newCoordinates, bs);CHKERRQ(ierr);
    ierr = VecDestroy(&newCoordinates);CHKERRQ(ierr);
  }
  ierr = DMGetPeriodicity(dm, &isper, &maxCell, &L, &bd);CHKERRQ(ierr);
  ierr = DMSetPeriodicity(dmParallel, isper, maxCell, L, bd);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexDistributeLabels(DM dm, PetscSF migrationSF, DM dmParallel)
{
  DM_Plex         *mesh = (DM_Plex*) dm->data;
  MPI_Comm         comm;
  DMLabel          depthLabel;
  PetscMPIInt      rank;
  PetscInt         depth, d, numLabels, numLocalLabels, l;
  PetscBool        hasLabels = PETSC_FALSE, lsendDepth, sendDepth;
  PetscObjectState depthState = -1;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(dmParallel, DM_CLASSID, 3);

  ierr = PetscLogEventBegin(DMPLEX_DistributeLabels,dm,0,0,0);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)dm, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);

  /* If the user has changed the depth label, communicate it instead */
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = DMPlexGetDepthLabel(dm, &depthLabel);CHKERRQ(ierr);
  if (depthLabel) {ierr = PetscObjectStateGet((PetscObject) depthLabel, &depthState);CHKERRQ(ierr);}
  lsendDepth = mesh->depthState != depthState ? PETSC_TRUE : PETSC_FALSE;
  ierr = MPIU_Allreduce(&lsendDepth, &sendDepth, 1, MPIU_BOOL, MPI_LOR, comm);CHKERRQ(ierr);
  if (sendDepth) {
    ierr = DMPlexGetDepthLabel(dmParallel, &dmParallel->depthLabel);CHKERRQ(ierr);
    ierr = DMRemoveLabelBySelf(dmParallel, &dmParallel->depthLabel, PETSC_FALSE);CHKERRQ(ierr);
  }
  /* Everyone must have either the same number of labels, or none */
  ierr = DMGetNumLabels(dm, &numLocalLabels);CHKERRQ(ierr);
  numLabels = numLocalLabels;
  ierr = MPI_Bcast(&numLabels, 1, MPIU_INT, 0, comm);CHKERRQ(ierr);
  if (numLabels == numLocalLabels) hasLabels = PETSC_TRUE;
  for (l = numLabels-1; l >= 0; --l) {
    DMLabel     label = NULL, labelNew = NULL;
    PetscBool   isDepth, lisOutput = PETSC_TRUE, isOutput;
    const char *name = NULL;

    if (hasLabels) {
      ierr = DMGetLabelByNum(dm, l, &label);CHKERRQ(ierr);
      /* Skip "depth" because it is recreated */
      ierr = PetscObjectGetName((PetscObject) label, &name);CHKERRQ(ierr);
      ierr = PetscStrcmp(name, "depth", &isDepth);CHKERRQ(ierr);
    }
    ierr = MPI_Bcast(&isDepth, 1, MPIU_BOOL, 0, comm);CHKERRQ(ierr);
    if (isDepth && !sendDepth) continue;
    ierr = DMLabelDistribute(label, migrationSF, &labelNew);CHKERRQ(ierr);
    if (isDepth) {
      /* Put in any missing strata which can occur if users are managing the depth label themselves */
      PetscInt gdepth;

      ierr = MPIU_Allreduce(&depth, &gdepth, 1, MPIU_INT, MPI_MAX, comm);CHKERRQ(ierr);
      if ((depth >= 0) && (gdepth != depth)) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Inconsistent Plex depth %d != %d", depth, gdepth);
      for (d = 0; d <= gdepth; ++d) {
        PetscBool has;

        ierr = DMLabelHasStratum(labelNew, d, &has);CHKERRQ(ierr);
        if (!has) {ierr = DMLabelAddStratum(labelNew, d);CHKERRQ(ierr);}
      }
    }
    ierr = DMAddLabel(dmParallel, labelNew);CHKERRQ(ierr);
    /* Put the output flag in the new label */
    if (hasLabels) {ierr = DMGetLabelOutput(dm, name, &lisOutput);CHKERRQ(ierr);}
    ierr = MPIU_Allreduce(&lisOutput, &isOutput, 1, MPIU_BOOL, MPI_LAND, comm);CHKERRQ(ierr);
    ierr = PetscObjectGetName((PetscObject) labelNew, &name);CHKERRQ(ierr);
    ierr = DMSetLabelOutput(dmParallel, name, isOutput);CHKERRQ(ierr);
    ierr = DMLabelDestroy(&labelNew);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(DMPLEX_DistributeLabels,dm,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexDistributeSetupHybrid(DM dm, PetscSF migrationSF, ISLocalToGlobalMapping renumbering, DM dmParallel)
{
  DM_Plex        *mesh  = (DM_Plex*) dm->data;
  DM_Plex        *pmesh = (DM_Plex*) (dmParallel)->data;
  PetscBool      *isHybrid, *isHybridParallel;
  PetscInt        dim, depth, d;
  PetscInt        pStart, pEnd, pStartP, pEndP;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(dmParallel, DM_CLASSID, 3);

  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = DMPlexGetChart(dm,&pStart,&pEnd);CHKERRQ(ierr);
  ierr = DMPlexGetChart(dmParallel,&pStartP,&pEndP);CHKERRQ(ierr);
  ierr = PetscCalloc2(pEnd-pStart,&isHybrid,pEndP-pStartP,&isHybridParallel);CHKERRQ(ierr);
  for (d = 0; d <= depth; d++) {
    PetscInt hybridMax = (depth == 1 && d == 1) ? mesh->hybridPointMax[dim] : mesh->hybridPointMax[d];

    if (hybridMax >= 0) {
      PetscInt sStart, sEnd, p;

      ierr = DMPlexGetDepthStratum(dm,d,&sStart,&sEnd);CHKERRQ(ierr);
      for (p = hybridMax; p < sEnd; p++) isHybrid[p-pStart] = PETSC_TRUE;
    }
  }
  ierr = PetscSFBcastBegin(migrationSF,MPIU_BOOL,isHybrid,isHybridParallel);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(migrationSF,MPIU_BOOL,isHybrid,isHybridParallel);CHKERRQ(ierr);
  for (d = 0; d <= dim; d++) pmesh->hybridPointMax[d] = -1;
  for (d = 0; d <= depth; d++) {
    PetscInt sStart, sEnd, p, dd;

    ierr = DMPlexGetDepthStratum(dmParallel,d,&sStart,&sEnd);CHKERRQ(ierr);
    dd = (depth == 1 && d == 1) ? dim : d;
    for (p = sStart; p < sEnd; p++) {
      if (isHybridParallel[p-pStartP]) {
        pmesh->hybridPointMax[dd] = p;
        break;
      }
    }
  }
  ierr = PetscFree2(isHybrid,isHybridParallel);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexDistributeSetupTree(DM dm, PetscSF migrationSF, ISLocalToGlobalMapping original, ISLocalToGlobalMapping renumbering, DM dmParallel)
{
  DM_Plex        *mesh  = (DM_Plex*) dm->data;
  DM_Plex        *pmesh = (DM_Plex*) (dmParallel)->data;
  MPI_Comm        comm;
  DM              refTree;
  PetscSection    origParentSection, newParentSection;
  PetscInt        *origParents, *origChildIDs;
  PetscBool       flg;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(dmParallel, DM_CLASSID, 5);
  ierr = PetscObjectGetComm((PetscObject)dm, &comm);CHKERRQ(ierr);

  /* Set up tree */
  ierr = DMPlexGetReferenceTree(dm,&refTree);CHKERRQ(ierr);
  ierr = DMPlexSetReferenceTree(dmParallel,refTree);CHKERRQ(ierr);
  ierr = DMPlexGetTree(dm,&origParentSection,&origParents,&origChildIDs,NULL,NULL);CHKERRQ(ierr);
  if (origParentSection) {
    PetscInt        pStart, pEnd;
    PetscInt        *newParents, *newChildIDs, *globParents;
    PetscInt        *remoteOffsetsParents, newParentSize;
    PetscSF         parentSF;

    ierr = DMPlexGetChart(dmParallel, &pStart, &pEnd);CHKERRQ(ierr);
    ierr = PetscSectionCreate(PetscObjectComm((PetscObject)dmParallel),&newParentSection);CHKERRQ(ierr);
    ierr = PetscSectionSetChart(newParentSection,pStart,pEnd);CHKERRQ(ierr);
    ierr = PetscSFDistributeSection(migrationSF, origParentSection, &remoteOffsetsParents, newParentSection);CHKERRQ(ierr);
    ierr = PetscSFCreateSectionSF(migrationSF, origParentSection, remoteOffsetsParents, newParentSection, &parentSF);CHKERRQ(ierr);
    ierr = PetscFree(remoteOffsetsParents);CHKERRQ(ierr);
    ierr = PetscSectionGetStorageSize(newParentSection,&newParentSize);CHKERRQ(ierr);
    ierr = PetscMalloc2(newParentSize,&newParents,newParentSize,&newChildIDs);CHKERRQ(ierr);
    if (original) {
      PetscInt numParents;

      ierr = PetscSectionGetStorageSize(origParentSection,&numParents);CHKERRQ(ierr);
      ierr = PetscMalloc1(numParents,&globParents);CHKERRQ(ierr);
      ierr = ISLocalToGlobalMappingApplyBlock(original, numParents, origParents, globParents);CHKERRQ(ierr);
    }
    else {
      globParents = origParents;
    }
    ierr = PetscSFBcastBegin(parentSF, MPIU_INT, globParents, newParents);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(parentSF, MPIU_INT, globParents, newParents);CHKERRQ(ierr);
    if (original) {
      ierr = PetscFree(globParents);CHKERRQ(ierr);
    }
    ierr = PetscSFBcastBegin(parentSF, MPIU_INT, origChildIDs, newChildIDs);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(parentSF, MPIU_INT, origChildIDs, newChildIDs);CHKERRQ(ierr);
    ierr = ISGlobalToLocalMappingApplyBlock(renumbering,IS_GTOLM_MASK, newParentSize, newParents, NULL, newParents);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
    {
      PetscInt  p;
      PetscBool valid = PETSC_TRUE;
      for (p = 0; p < newParentSize; ++p) {
        if (newParents[p] < 0) {valid = PETSC_FALSE; ierr = PetscPrintf(PETSC_COMM_SELF, "Point %d not in overlap SF\n", p);CHKERRQ(ierr);}
      }
      if (!valid) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid global to local map");
    }
#endif
    ierr = PetscOptionsHasName(((PetscObject) dm)->options,((PetscObject) dm)->prefix, "-parents_view", &flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PetscPrintf(comm, "Serial Parent Section: \n");CHKERRQ(ierr);
      ierr = PetscSectionView(origParentSection, PETSC_VIEWER_STDOUT_(comm));CHKERRQ(ierr);
      ierr = PetscPrintf(comm, "Parallel Parent Section: \n");CHKERRQ(ierr);
      ierr = PetscSectionView(newParentSection, PETSC_VIEWER_STDOUT_(comm));CHKERRQ(ierr);
      ierr = PetscSFView(parentSF, NULL);CHKERRQ(ierr);
    }
    ierr = DMPlexSetTree(dmParallel,newParentSection,newParents,newChildIDs);CHKERRQ(ierr);
    ierr = PetscSectionDestroy(&newParentSection);CHKERRQ(ierr);
    ierr = PetscFree2(newParents,newChildIDs);CHKERRQ(ierr);
    ierr = PetscSFDestroy(&parentSF);CHKERRQ(ierr);
  }
  pmesh->useAnchors = mesh->useAnchors;
  PetscFunctionReturn(0);
}

PETSC_UNUSED static PetscErrorCode DMPlexDistributeSF(DM dm, PetscSF migrationSF, DM dmParallel)
{
  PetscMPIInt            rank, size;
  MPI_Comm               comm;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(dmParallel, DM_CLASSID, 3);

  /* Create point SF for parallel mesh */
  ierr = PetscLogEventBegin(DMPLEX_DistributeSF,dm,0,0,0);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)dm, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  {
    const PetscInt *leaves;
    PetscSFNode    *remotePoints, *rowners, *lowners;
    PetscInt        numRoots, numLeaves, numGhostPoints = 0, p, gp, *ghostPoints;
    PetscInt        pStart, pEnd;

    ierr = DMPlexGetChart(dmParallel, &pStart, &pEnd);CHKERRQ(ierr);
    ierr = PetscSFGetGraph(migrationSF, &numRoots, &numLeaves, &leaves, NULL);CHKERRQ(ierr);
    ierr = PetscMalloc2(numRoots,&rowners,numLeaves,&lowners);CHKERRQ(ierr);
    for (p=0; p<numRoots; p++) {
      rowners[p].rank  = -1;
      rowners[p].index = -1;
    }
    ierr = PetscSFBcastBegin(migrationSF, MPIU_2INT, rowners, lowners);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(migrationSF, MPIU_2INT, rowners, lowners);CHKERRQ(ierr);
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
    ierr = PetscSFReduceBegin(migrationSF, MPIU_2INT, lowners, rowners, MPI_MAXLOC);CHKERRQ(ierr);
    ierr = PetscSFReduceEnd(migrationSF, MPIU_2INT, lowners, rowners, MPI_MAXLOC);CHKERRQ(ierr);
    ierr = PetscSFBcastBegin(migrationSF, MPIU_2INT, rowners, lowners);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(migrationSF, MPIU_2INT, rowners, lowners);CHKERRQ(ierr);
    for (p = 0; p < numLeaves; ++p) {
      if (lowners[p].rank < 0 || lowners[p].index < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Cell partition corrupt: point not claimed");
      if (lowners[p].rank != rank) ++numGhostPoints;
    }
    ierr = PetscMalloc1(numGhostPoints, &ghostPoints);CHKERRQ(ierr);
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
    ierr = PetscSFSetGraph((dmParallel)->sf, pEnd - pStart, numGhostPoints, ghostPoints, PETSC_OWN_POINTER, remotePoints, PETSC_OWN_POINTER);CHKERRQ(ierr);
    ierr = PetscSFSetFromOptions((dmParallel)->sf);CHKERRQ(ierr);
  }
  {
    PetscBool useCone, useClosure, useAnchors;

    ierr = DMGetBasicAdjacency(dm, &useCone, &useClosure);CHKERRQ(ierr);
    ierr = DMSetBasicAdjacency(dmParallel, useCone, useClosure);CHKERRQ(ierr);
    ierr = DMPlexGetAdjacencyUseAnchors(dm, &useAnchors);CHKERRQ(ierr);
    ierr = DMPlexSetAdjacencyUseAnchors(dmParallel, useAnchors);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(DMPLEX_DistributeSF,dm,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMPlexSetPartitionBalance - Should distribution of the DM attempt to balance the shared point partition?

  Input Parameters:
+ dm - The DMPlex object
- flg - Balance the partition?

  Level: intermediate

.seealso: DMPlexDistribute(), DMPlexGetPartitionBalance()
@*/
PetscErrorCode DMPlexSetPartitionBalance(DM dm, PetscBool flg)
{
  DM_Plex *mesh = (DM_Plex *)dm->data;

  PetscFunctionBegin;
  mesh->partitionBalance = flg;
  PetscFunctionReturn(0);
}

/*@
  DMPlexGetPartitionBalance - Does distribution of the DM attempt to balance the shared point partition?

  Input Parameter:
. dm - The DMPlex object

  Output Parameter:
. flg - Balance the partition?

  Level: intermediate

.seealso: DMPlexDistribute(), DMPlexSetPartitionBalance()
@*/
PetscErrorCode DMPlexGetPartitionBalance(DM dm, PetscBool *flg)
{
  DM_Plex *mesh = (DM_Plex *)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidBoolPointer(flg, 2);
  *flg = mesh->partitionBalance;
  PetscFunctionReturn(0);
}

typedef struct {
  PetscInt vote, rank, index;
} Petsc3Int;

/* MaxLoc, but carry a third piece of information around */
static void MaxLocCarry(void *in_, void *inout_, PetscMPIInt *len_, MPI_Datatype *dtype)
{
  Petsc3Int *a = (Petsc3Int *)inout_;
  Petsc3Int *b = (Petsc3Int *)in_;
  PetscInt i, len = *len_;
  for (i = 0; i < len; i++) {
    if (a[i].vote < b[i].vote) {
      a[i].vote = b[i].vote;
      a[i].rank = b[i].rank;
      a[i].index = b[i].index;
    } else if (a[i].vote <= b[i].vote) {
      if (a[i].rank >= b[i].rank) {
        a[i].rank = b[i].rank;
        a[i].index = b[i].index;
      }
    }
  }
}

/*@C
  DMPlexCreatePointSF - Build a point SF from an SF describing a point migration

  Input Parameter:
+ dm          - The source DMPlex object
. migrationSF - The star forest that describes the parallel point remapping
. ownership   - Flag causing a vote to determine point ownership

  Output Parameter:
- pointSF     - The star forest describing the point overlap in the remapped DM

  Level: developer

.seealso: DMPlexDistribute(), DMPlexDistributeOverlap()
@*/
PetscErrorCode DMPlexCreatePointSF(DM dm, PetscSF migrationSF, PetscBool ownership, PetscSF *pointSF)
{
  PetscMPIInt        rank, size;
  PetscInt           p, nroots, nleaves, idx, npointLeaves;
  PetscInt          *pointLocal;
  const PetscInt    *leaves;
  const PetscSFNode *roots;
  PetscSFNode       *rootNodes, *leafNodes, *pointRemote;
  Vec                shifts;
  const PetscInt     numShifts = 13759;
  const PetscScalar *shift = NULL;
  const PetscBool    shiftDebug = PETSC_FALSE;
  PetscBool          balance;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject) dm), &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject) dm), &size);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(DMPLEX_CreatePointSF,dm,0,0,0);CHKERRQ(ierr);

  ierr = DMPlexGetPartitionBalance(dm, &balance);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(migrationSF, &nroots, &nleaves, &leaves, &roots);CHKERRQ(ierr);
  ierr = PetscMalloc2(nroots, &rootNodes, nleaves, &leafNodes);CHKERRQ(ierr);
  if (ownership) {
    MPI_Op       op;
    MPI_Datatype datatype;
    Petsc3Int   *rootVote = NULL, *leafVote = NULL;
    /* If balancing, we compute a random cyclic shift of the rank for each remote point. That way, the max will evenly distribute among ranks. */
    if (balance) {
      PetscRandom r;

      ierr = PetscRandomCreate(PETSC_COMM_SELF, &r);CHKERRQ(ierr);
      ierr = PetscRandomSetInterval(r, 0, 2467*size);CHKERRQ(ierr);
      ierr = VecCreate(PETSC_COMM_SELF, &shifts);CHKERRQ(ierr);
      ierr = VecSetSizes(shifts, numShifts, numShifts);CHKERRQ(ierr);
      ierr = VecSetType(shifts, VECSTANDARD);CHKERRQ(ierr);
      ierr = VecSetRandom(shifts, r);CHKERRQ(ierr);
      ierr = PetscRandomDestroy(&r);CHKERRQ(ierr);
      ierr = VecGetArrayRead(shifts, &shift);CHKERRQ(ierr);
    }

    ierr = PetscMalloc1(nroots, &rootVote);CHKERRQ(ierr);
    ierr = PetscMalloc1(nleaves, &leafVote);CHKERRQ(ierr);
    /* Point ownership vote: Process with highest rank owns shared points */
    for (p = 0; p < nleaves; ++p) {
      if (shiftDebug) {
        ierr = PetscSynchronizedPrintf(PetscObjectComm((PetscObject) dm), "[%d] Point %D RemotePoint %D Shift %D MyRank %D\n", rank, leaves ? leaves[p] : p, roots[p].index, (PetscInt) PetscRealPart(shift[roots[p].index%numShifts]), (rank + (shift ? (PetscInt) PetscRealPart(shift[roots[p].index%numShifts]) : 0))%size);CHKERRQ(ierr);
      }
      /* Either put in a bid or we know we own it */
      leafVote[p].vote  = (rank + (shift ? (PetscInt) PetscRealPart(shift[roots[p].index%numShifts]) : 0))%size;
      leafVote[p].rank = rank;
      leafVote[p].index = p;
    }
    for (p = 0; p < nroots; p++) {
      /* Root must not participate in the reduction, flag so that MAXLOC does not use */
      rootVote[p].vote  = -3;
      rootVote[p].rank  = -3;
      rootVote[p].index = -3;
    }
    ierr = MPI_Type_contiguous(3, MPIU_INT, &datatype);CHKERRQ(ierr);
    ierr = MPI_Type_commit(&datatype);CHKERRQ(ierr);
    ierr = MPI_Op_create(&MaxLocCarry, 1, &op);CHKERRQ(ierr);
    ierr = PetscSFReduceBegin(migrationSF, datatype, leafVote, rootVote, op);CHKERRQ(ierr);
    ierr = PetscSFReduceEnd(migrationSF, datatype, leafVote, rootVote, op);CHKERRQ(ierr);
    ierr = MPI_Op_free(&op);CHKERRQ(ierr);
    ierr = MPI_Type_free(&datatype);CHKERRQ(ierr);
    for (p = 0; p < nroots; p++) {
      rootNodes[p].rank = rootVote[p].rank;
      rootNodes[p].index = rootVote[p].index;
    }
    ierr = PetscFree(leafVote);CHKERRQ(ierr);
    ierr = PetscFree(rootVote);CHKERRQ(ierr);
  } else {
    for (p = 0; p < nroots; p++) {
      rootNodes[p].index = -1;
      rootNodes[p].rank = rank;
    }
    for (p = 0; p < nleaves; p++) {
      /* Write new local id into old location */
      if (roots[p].rank == rank) {
        rootNodes[roots[p].index].index = leaves ? leaves[p] : p;
      }
    }
  }
  ierr = PetscSFBcastBegin(migrationSF, MPIU_2INT, rootNodes, leafNodes);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(migrationSF, MPIU_2INT, rootNodes, leafNodes);CHKERRQ(ierr);

  for (npointLeaves = 0, p = 0; p < nleaves; p++) {
    if (leafNodes[p].rank != rank) npointLeaves++;
  }
  ierr = PetscMalloc1(npointLeaves, &pointLocal);CHKERRQ(ierr);
  ierr = PetscMalloc1(npointLeaves, &pointRemote);CHKERRQ(ierr);
  for (idx = 0, p = 0; p < nleaves; p++) {
    if (leafNodes[p].rank != rank) {
      pointLocal[idx] = p;
      pointRemote[idx] = leafNodes[p];
      idx++;
    }
  }
  if (shift) {
    ierr = VecRestoreArrayRead(shifts, &shift);CHKERRQ(ierr);
    ierr = VecDestroy(&shifts);CHKERRQ(ierr);
  }
  if (shiftDebug) {ierr = PetscSynchronizedFlush(PetscObjectComm((PetscObject) dm), PETSC_STDOUT);CHKERRQ(ierr);}
  ierr = PetscSFCreate(PetscObjectComm((PetscObject) dm), pointSF);CHKERRQ(ierr);
  ierr = PetscSFSetFromOptions(*pointSF);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(*pointSF, nleaves, npointLeaves, pointLocal, PETSC_OWN_POINTER, pointRemote, PETSC_OWN_POINTER);CHKERRQ(ierr);
  ierr = PetscFree2(rootNodes, leafNodes);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(DMPLEX_CreatePointSF,dm,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMPlexMigrate  - Migrates internal DM data over the supplied star forest

  Collective on dm

  Input Parameter:
+ dm       - The source DMPlex object
. sf       - The star forest communication context describing the migration pattern

  Output Parameter:
- targetDM - The target DMPlex object

  Level: intermediate

.seealso: DMPlexDistribute(), DMPlexDistributeOverlap()
@*/
PetscErrorCode DMPlexMigrate(DM dm, PetscSF sf, DM targetDM)
{
  MPI_Comm               comm;
  PetscInt               dim, cdim, nroots;
  PetscSF                sfPoint;
  ISLocalToGlobalMapping ltogMigration;
  ISLocalToGlobalMapping ltogOriginal = NULL;
  PetscBool              flg;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = PetscLogEventBegin(DMPLEX_Migrate, dm, 0, 0, 0);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMSetDimension(targetDM, dim);CHKERRQ(ierr);
  ierr = DMGetCoordinateDim(dm, &cdim);CHKERRQ(ierr);
  ierr = DMSetCoordinateDim(targetDM, cdim);CHKERRQ(ierr);

  /* Check for a one-to-all distribution pattern */
  ierr = DMGetPointSF(dm, &sfPoint);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(sfPoint, &nroots, NULL, NULL, NULL);CHKERRQ(ierr);
  if (nroots >= 0) {
    IS        isOriginal;
    PetscInt  n, size, nleaves;
    PetscInt  *numbering_orig, *numbering_new;

    /* Get the original point numbering */
    ierr = DMPlexCreatePointNumbering(dm, &isOriginal);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingCreateIS(isOriginal, &ltogOriginal);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingGetSize(ltogOriginal, &size);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingGetBlockIndices(ltogOriginal, (const PetscInt**)&numbering_orig);CHKERRQ(ierr);
    /* Convert to positive global numbers */
    for (n=0; n<size; n++) {if (numbering_orig[n] < 0) numbering_orig[n] = -(numbering_orig[n]+1);}
    /* Derive the new local-to-global mapping from the old one */
    ierr = PetscSFGetGraph(sf, NULL, &nleaves, NULL, NULL);CHKERRQ(ierr);
    ierr = PetscMalloc1(nleaves, &numbering_new);CHKERRQ(ierr);
    ierr = PetscSFBcastBegin(sf, MPIU_INT, numbering_orig, numbering_new);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(sf, MPIU_INT, numbering_orig, numbering_new);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingCreate(comm, 1, nleaves, numbering_new, PETSC_OWN_POINTER, &ltogMigration);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingRestoreIndices(ltogOriginal, (const PetscInt**)&numbering_orig);CHKERRQ(ierr);
    ierr = ISDestroy(&isOriginal);CHKERRQ(ierr);
  } else {
    /* One-to-all distribution pattern: We can derive LToG from SF */
    ierr = ISLocalToGlobalMappingCreateSF(sf, 0, &ltogMigration);CHKERRQ(ierr);
  }
  ierr = PetscOptionsHasName(((PetscObject) dm)->options,((PetscObject) dm)->prefix, "-partition_view", &flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscPrintf(comm, "Point renumbering for DM migration:\n");CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingView(ltogMigration, NULL);CHKERRQ(ierr);
  }
  /* Migrate DM data to target DM */
  ierr = DMPlexDistributeCones(dm, sf, ltogOriginal, ltogMigration, targetDM);CHKERRQ(ierr);
  ierr = DMPlexDistributeLabels(dm, sf, targetDM);CHKERRQ(ierr);
  ierr = DMPlexDistributeCoordinates(dm, sf, targetDM);CHKERRQ(ierr);
  ierr = DMPlexDistributeSetupHybrid(dm, sf, ltogMigration, targetDM);CHKERRQ(ierr);
  ierr = DMPlexDistributeSetupTree(dm, sf, ltogOriginal, ltogMigration, targetDM);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&ltogOriginal);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&ltogMigration);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(DMPLEX_Migrate, dm, 0, 0, 0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMPlexDistribute - Distributes the mesh and any associated sections.

  Collective on dm

  Input Parameter:
+ dm  - The original DMPlex object
- overlap - The overlap of partitions, 0 is the default

  Output Parameter:
+ sf - The PetscSF used for point distribution, or NULL if not needed
- dmParallel - The distributed DMPlex object

  Note: If the mesh was not distributed, the output dmParallel will be NULL.

  The user can control the definition of adjacency for the mesh using DMSetAdjacency(). They should choose the combination appropriate for the function
  representation on the mesh.

  Level: intermediate

.seealso: DMPlexCreate(), DMSetAdjacency(), DMPlexGetOverlap()
@*/
PetscErrorCode DMPlexDistribute(DM dm, PetscInt overlap, PetscSF *sf, DM *dmParallel)
{
  MPI_Comm               comm;
  PetscPartitioner       partitioner;
  IS                     cellPart;
  PetscSection           cellPartSection;
  DM                     dmCoord;
  DMLabel                lblPartition, lblMigration;
  PetscSF                sfMigration, sfStratified, sfPoint;
  PetscBool              flg, balance;
  PetscMPIInt            rank, size;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidLogicalCollectiveInt(dm, overlap, 2);
  if (sf) PetscValidPointer(sf,3);
  PetscValidPointer(dmParallel,4);

  if (sf) *sf = NULL;
  *dmParallel = NULL;
  ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  if (size == 1) PetscFunctionReturn(0);

  ierr = PetscLogEventBegin(DMPLEX_Distribute,dm,0,0,0);CHKERRQ(ierr);
  /* Create cell partition */
  ierr = PetscLogEventBegin(DMPLEX_Partition,dm,0,0,0);CHKERRQ(ierr);
  ierr = PetscSectionCreate(comm, &cellPartSection);CHKERRQ(ierr);
  ierr = DMPlexGetPartitioner(dm, &partitioner);CHKERRQ(ierr);
  ierr = PetscPartitionerPartition(partitioner, dm, cellPartSection, &cellPart);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(DMPLEX_PartSelf,dm,0,0,0);CHKERRQ(ierr);
  {
    /* Convert partition to DMLabel */
    IS             is;
    PetscHSetI     ht;
    const PetscInt *points;
    PetscInt       *iranks;
    PetscInt       pStart, pEnd, proc, npoints, poff = 0, nranks;

    ierr = DMLabelCreate(PETSC_COMM_SELF, "Point Partition", &lblPartition);CHKERRQ(ierr);
    /* Preallocate strata */
    ierr = PetscHSetICreate(&ht);CHKERRQ(ierr);
    ierr = PetscSectionGetChart(cellPartSection, &pStart, &pEnd);CHKERRQ(ierr);
    for (proc = pStart; proc < pEnd; proc++) {
      ierr = PetscSectionGetDof(cellPartSection, proc, &npoints);CHKERRQ(ierr);
      if (npoints) {ierr = PetscHSetIAdd(ht, proc);CHKERRQ(ierr);}
    }
    ierr = PetscHSetIGetSize(ht, &nranks);CHKERRQ(ierr);
    ierr = PetscMalloc1(nranks, &iranks);CHKERRQ(ierr);
    ierr = PetscHSetIGetElems(ht, &poff, iranks);CHKERRQ(ierr);
    ierr = PetscHSetIDestroy(&ht);CHKERRQ(ierr);
    ierr = DMLabelAddStrata(lblPartition, nranks, iranks);CHKERRQ(ierr);
    ierr = PetscFree(iranks);CHKERRQ(ierr);
    /* Inline DMPlexPartitionLabelClosure() */
    ierr = ISGetIndices(cellPart, &points);CHKERRQ(ierr);
    ierr = PetscSectionGetChart(cellPartSection, &pStart, &pEnd);CHKERRQ(ierr);
    for (proc = pStart; proc < pEnd; proc++) {
      ierr = PetscSectionGetDof(cellPartSection, proc, &npoints);CHKERRQ(ierr);
      if (!npoints) continue;
      ierr = PetscSectionGetOffset(cellPartSection, proc, &poff);CHKERRQ(ierr);
      ierr = DMPlexClosurePoints_Private(dm, npoints, points+poff, &is);CHKERRQ(ierr);
      ierr = DMLabelSetStratumIS(lblPartition, proc, is);CHKERRQ(ierr);
      ierr = ISDestroy(&is);CHKERRQ(ierr);
    }
    ierr = ISRestoreIndices(cellPart, &points);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(DMPLEX_PartSelf,dm,0,0,0);CHKERRQ(ierr);

  ierr = DMLabelCreate(PETSC_COMM_SELF, "Point migration", &lblMigration);CHKERRQ(ierr);
  ierr = DMPlexPartitionLabelInvert(dm, lblPartition, NULL, lblMigration);CHKERRQ(ierr);
  ierr = DMPlexPartitionLabelCreateSF(dm, lblMigration, &sfMigration);CHKERRQ(ierr);
  ierr = DMPlexStratifyMigrationSF(dm, sfMigration, &sfStratified);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sfMigration);CHKERRQ(ierr);
  sfMigration = sfStratified;
  ierr = PetscSFSetUp(sfMigration);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(DMPLEX_Partition,dm,0,0,0);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(((PetscObject) dm)->options,((PetscObject) dm)->prefix, "-partition_view", &flg);CHKERRQ(ierr);
  if (flg) {
    ierr = DMLabelView(lblPartition, PETSC_VIEWER_STDOUT_(comm));CHKERRQ(ierr);
    ierr = PetscSFView(sfMigration, PETSC_VIEWER_STDOUT_(comm));CHKERRQ(ierr);
  }

  /* Create non-overlapping parallel DM and migrate internal data */
  ierr = DMPlexCreate(comm, dmParallel);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *dmParallel, "Parallel Mesh");CHKERRQ(ierr);
  ierr = DMPlexMigrate(dm, sfMigration, *dmParallel);CHKERRQ(ierr);

  /* Build the point SF without overlap */
  ierr = DMPlexGetPartitionBalance(dm, &balance);CHKERRQ(ierr);
  ierr = DMPlexSetPartitionBalance(*dmParallel, balance);CHKERRQ(ierr);
  ierr = DMPlexCreatePointSF(*dmParallel, sfMigration, PETSC_TRUE, &sfPoint);CHKERRQ(ierr);
  ierr = DMSetPointSF(*dmParallel, sfPoint);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(*dmParallel, &dmCoord);CHKERRQ(ierr);
  if (dmCoord) {ierr = DMSetPointSF(dmCoord, sfPoint);CHKERRQ(ierr);}
  if (flg) {ierr = PetscSFView(sfPoint, NULL);CHKERRQ(ierr);}

  if (overlap > 0) {
    DM                 dmOverlap;
    PetscInt           nroots, nleaves, noldleaves, l;
    const PetscInt    *oldLeaves;
    PetscSFNode       *newRemote, *permRemote;
    const PetscSFNode *oldRemote;
    PetscSF            sfOverlap, sfOverlapPoint;

    /* Add the partition overlap to the distributed DM */
    ierr = DMPlexDistributeOverlap(*dmParallel, overlap, &sfOverlap, &dmOverlap);CHKERRQ(ierr);
    ierr = DMDestroy(dmParallel);CHKERRQ(ierr);
    *dmParallel = dmOverlap;
    if (flg) {
      ierr = PetscPrintf(comm, "Overlap Migration SF:\n");CHKERRQ(ierr);
      ierr = PetscSFView(sfOverlap, NULL);CHKERRQ(ierr);
    }

    /* Re-map the migration SF to establish the full migration pattern */
    ierr = PetscSFGetGraph(sfMigration, &nroots, &noldleaves, &oldLeaves, &oldRemote);CHKERRQ(ierr);
    ierr = PetscSFGetGraph(sfOverlap, NULL, &nleaves, NULL, NULL);CHKERRQ(ierr);
    ierr = PetscMalloc1(nleaves, &newRemote);CHKERRQ(ierr);
    /* oldRemote: original root point mapping to original leaf point
       newRemote: original leaf point mapping to overlapped leaf point */
    if (oldLeaves) {
      /* After stratification, the migration remotes may not be in root (canonical) order, so we reorder using the leaf numbering */
      ierr = PetscMalloc1(noldleaves, &permRemote);CHKERRQ(ierr);
      for (l = 0; l < noldleaves; ++l) permRemote[oldLeaves[l]] = oldRemote[l];
      oldRemote = permRemote;
    }
    ierr = PetscSFBcastBegin(sfOverlap, MPIU_2INT, oldRemote, newRemote);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(sfOverlap, MPIU_2INT, oldRemote, newRemote);CHKERRQ(ierr);
    if (oldLeaves) {ierr = PetscFree(oldRemote);CHKERRQ(ierr);}
    ierr = PetscSFCreate(comm, &sfOverlapPoint);CHKERRQ(ierr);
    ierr = PetscSFSetGraph(sfOverlapPoint, nroots, nleaves, NULL, PETSC_OWN_POINTER, newRemote, PETSC_OWN_POINTER);CHKERRQ(ierr);
    ierr = PetscSFDestroy(&sfOverlap);CHKERRQ(ierr);
    ierr = PetscSFDestroy(&sfMigration);CHKERRQ(ierr);
    sfMigration = sfOverlapPoint;
  }
  /* Cleanup Partition */
  ierr = DMLabelDestroy(&lblPartition);CHKERRQ(ierr);
  ierr = DMLabelDestroy(&lblMigration);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&cellPartSection);CHKERRQ(ierr);
  ierr = ISDestroy(&cellPart);CHKERRQ(ierr);
  /* Copy BC */
  ierr = DMCopyBoundary(dm, *dmParallel);CHKERRQ(ierr);
  /* Create sfNatural */
  if (dm->useNatural) {
    PetscSection section;

    ierr = DMGetLocalSection(dm, &section);CHKERRQ(ierr);
    ierr = DMPlexCreateGlobalToNaturalSF(*dmParallel, section, sfMigration, &(*dmParallel)->sfNatural);CHKERRQ(ierr);
    ierr = DMSetUseNatural(*dmParallel, PETSC_TRUE);CHKERRQ(ierr);
  }
  /* Cleanup */
  if (sf) {*sf = sfMigration;}
  else    {ierr = PetscSFDestroy(&sfMigration);CHKERRQ(ierr);}
  ierr = PetscSFDestroy(&sfPoint);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(DMPLEX_Distribute,dm,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMPlexDistributeOverlap - Add partition overlap to a distributed non-overlapping DM.

  Collective on dm

  Input Parameter:
+ dm  - The non-overlapping distrbuted DMPlex object
- overlap - The overlap of partitions

  Output Parameter:
+ sf - The PetscSF used for point distribution
- dmOverlap - The overlapping distributed DMPlex object, or NULL

  Notes:
  If the mesh was not distributed, the return value is NULL.

  The user can control the definition of adjacency for the mesh using DMSetAdjacency(). They should choose the combination appropriate for the function
  representation on the mesh.

  Level: advanced

.seealso: DMPlexCreate(), DMSetAdjacency(), DMPlexDistribute(), DMPlexCreateOverlapLabel(), DMPlexGetOverlap()
@*/
PetscErrorCode DMPlexDistributeOverlap(DM dm, PetscInt overlap, PetscSF *sf, DM *dmOverlap)
{
  MPI_Comm               comm;
  PetscMPIInt            size, rank;
  PetscSection           rootSection, leafSection;
  IS                     rootrank, leafrank;
  DM                     dmCoord;
  DMLabel                lblOverlap;
  PetscSF                sfOverlap, sfStratified, sfPoint;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (sf) PetscValidPointer(sf, 3);
  PetscValidPointer(dmOverlap, 4);

  if (sf) *sf = NULL;
  *dmOverlap  = NULL;
  ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  if (size == 1) PetscFunctionReturn(0);

  ierr = PetscLogEventBegin(DMPLEX_DistributeOverlap, dm, 0, 0, 0);CHKERRQ(ierr);
  /* Compute point overlap with neighbouring processes on the distributed DM */
  ierr = PetscLogEventBegin(DMPLEX_Partition,dm,0,0,0);CHKERRQ(ierr);
  ierr = PetscSectionCreate(comm, &rootSection);CHKERRQ(ierr);
  ierr = PetscSectionCreate(comm, &leafSection);CHKERRQ(ierr);
  ierr = DMPlexDistributeOwnership(dm, rootSection, &rootrank, leafSection, &leafrank);CHKERRQ(ierr);
  ierr = DMPlexCreateOverlapLabel(dm, overlap, rootSection, rootrank, leafSection, leafrank, &lblOverlap);CHKERRQ(ierr);
  /* Convert overlap label to stratified migration SF */
  ierr = DMPlexPartitionLabelCreateSF(dm, lblOverlap, &sfOverlap);CHKERRQ(ierr);
  ierr = DMPlexStratifyMigrationSF(dm, sfOverlap, &sfStratified);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sfOverlap);CHKERRQ(ierr);
  sfOverlap = sfStratified;
  ierr = PetscObjectSetName((PetscObject) sfOverlap, "Overlap SF");CHKERRQ(ierr);
  ierr = PetscSFSetFromOptions(sfOverlap);CHKERRQ(ierr);

  ierr = PetscSectionDestroy(&rootSection);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&leafSection);CHKERRQ(ierr);
  ierr = ISDestroy(&rootrank);CHKERRQ(ierr);
  ierr = ISDestroy(&leafrank);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(DMPLEX_Partition,dm,0,0,0);CHKERRQ(ierr);

  /* Build the overlapping DM */
  ierr = DMPlexCreate(comm, dmOverlap);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *dmOverlap, "Parallel Mesh");CHKERRQ(ierr);
  ierr = DMPlexMigrate(dm, sfOverlap, *dmOverlap);CHKERRQ(ierr);
  /* Store the overlap in the new DM */
  ((DM_Plex*)(*dmOverlap)->data)->overlap = overlap + ((DM_Plex*)dm->data)->overlap;
  /* Build the new point SF */
  ierr = DMPlexCreatePointSF(*dmOverlap, sfOverlap, PETSC_FALSE, &sfPoint);CHKERRQ(ierr);
  ierr = DMSetPointSF(*dmOverlap, sfPoint);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(*dmOverlap, &dmCoord);CHKERRQ(ierr);
  if (dmCoord) {ierr = DMSetPointSF(dmCoord, sfPoint);CHKERRQ(ierr);}
  ierr = PetscSFDestroy(&sfPoint);CHKERRQ(ierr);
  /* Cleanup overlap partition */
  ierr = DMLabelDestroy(&lblOverlap);CHKERRQ(ierr);
  if (sf) *sf = sfOverlap;
  else    {ierr = PetscSFDestroy(&sfOverlap);CHKERRQ(ierr);}
  ierr = PetscLogEventEnd(DMPLEX_DistributeOverlap, dm, 0, 0, 0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexGetOverlap_Plex(DM dm, PetscInt *overlap)
{
  DM_Plex        *mesh  = (DM_Plex*) dm->data;

  PetscFunctionBegin;
  *overlap = mesh->overlap;
  PetscFunctionReturn(0);
}

/*@
  DMPlexGetOverlap - Get the DMPlex partition overlap.

  Not collective

  Input Parameter:
. dm - The DM

  Output Parameters:
. overlap - The overlap of this DM

  Level: intermediate

.seealso: DMPlexDistribute(), DMPlexDistributeOverlap(), DMPlexCreateOverlapLabel()
@*/
PetscErrorCode DMPlexGetOverlap(DM dm, PetscInt *overlap)
{
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = PetscUseMethod(dm,"DMPlexGetOverlap_C",(DM,PetscInt*),(dm,overlap));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/*@C
  DMPlexGetGatherDM - Get a copy of the DMPlex that gathers all points on the
  root process of the original's communicator.

  Collective on dm

  Input Parameters:
. dm - the original DMPlex object

  Output Parameters:
+ sf - the PetscSF used for point distribution (optional)
- gatherMesh - the gathered DM object, or NULL

  Level: intermediate

.seealso: DMPlexDistribute(), DMPlexGetRedundantDM()
@*/
PetscErrorCode DMPlexGetGatherDM(DM dm, PetscSF *sf, DM *gatherMesh)
{
  MPI_Comm       comm;
  PetscMPIInt    size;
  PetscPartitioner oldPart, gatherPart;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(gatherMesh,2);
  *gatherMesh = NULL;
  if (sf) *sf = NULL;
  comm = PetscObjectComm((PetscObject)dm);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (size == 1) PetscFunctionReturn(0);
  ierr = DMPlexGetPartitioner(dm,&oldPart);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)oldPart);CHKERRQ(ierr);
  ierr = PetscPartitionerCreate(comm,&gatherPart);CHKERRQ(ierr);
  ierr = PetscPartitionerSetType(gatherPart,PETSCPARTITIONERGATHER);CHKERRQ(ierr);
  ierr = DMPlexSetPartitioner(dm,gatherPart);CHKERRQ(ierr);
  ierr = DMPlexDistribute(dm,0,sf,gatherMesh);CHKERRQ(ierr);

  ierr = DMPlexSetPartitioner(dm,oldPart);CHKERRQ(ierr);
  ierr = PetscPartitionerDestroy(&gatherPart);CHKERRQ(ierr);
  ierr = PetscPartitionerDestroy(&oldPart);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMPlexGetRedundantDM - Get a copy of the DMPlex that is completely copied on each process.

  Collective on dm

  Input Parameters:
. dm - the original DMPlex object

  Output Parameters:
+ sf - the PetscSF used for point distribution (optional)
- redundantMesh - the redundant DM object, or NULL

  Level: intermediate

.seealso: DMPlexDistribute(), DMPlexGetGatherDM()
@*/
PetscErrorCode DMPlexGetRedundantDM(DM dm, PetscSF *sf, DM *redundantMesh)
{
  MPI_Comm       comm;
  PetscMPIInt    size, rank;
  PetscInt       pStart, pEnd, p;
  PetscInt       numPoints = -1;
  PetscSF        migrationSF, sfPoint, gatherSF;
  DM             gatherDM, dmCoord;
  PetscSFNode    *points;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(redundantMesh,2);
  *redundantMesh = NULL;
  comm = PetscObjectComm((PetscObject)dm);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (size == 1) {
    ierr = PetscObjectReference((PetscObject) dm);CHKERRQ(ierr);
    *redundantMesh = dm;
    if (sf) *sf = NULL;
    PetscFunctionReturn(0);
  }
  ierr = DMPlexGetGatherDM(dm,&gatherSF,&gatherDM);CHKERRQ(ierr);
  if (!gatherDM) PetscFunctionReturn(0);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = DMPlexGetChart(gatherDM,&pStart,&pEnd);CHKERRQ(ierr);
  numPoints = pEnd - pStart;
  ierr = MPI_Bcast(&numPoints,1,MPIU_INT,0,comm);CHKERRQ(ierr);
  ierr = PetscMalloc1(numPoints,&points);CHKERRQ(ierr);
  ierr = PetscSFCreate(comm,&migrationSF);CHKERRQ(ierr);
  for (p = 0; p < numPoints; p++) {
    points[p].index = p;
    points[p].rank  = 0;
  }
  ierr = PetscSFSetGraph(migrationSF,pEnd-pStart,numPoints,NULL,PETSC_OWN_POINTER,points,PETSC_OWN_POINTER);CHKERRQ(ierr);
  ierr = DMPlexCreate(comm, redundantMesh);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *redundantMesh, "Redundant Mesh");CHKERRQ(ierr);
  ierr = DMPlexMigrate(gatherDM, migrationSF, *redundantMesh);CHKERRQ(ierr);
  ierr = DMPlexCreatePointSF(*redundantMesh, migrationSF, PETSC_FALSE, &sfPoint);CHKERRQ(ierr);
  ierr = DMSetPointSF(*redundantMesh, sfPoint);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(*redundantMesh, &dmCoord);CHKERRQ(ierr);
  if (dmCoord) {ierr = DMSetPointSF(dmCoord, sfPoint);CHKERRQ(ierr);}
  ierr = PetscSFDestroy(&sfPoint);CHKERRQ(ierr);
  if (sf) {
    PetscSF tsf;

    ierr = PetscSFCompose(gatherSF,migrationSF,&tsf);CHKERRQ(ierr);
    ierr = DMPlexStratifyMigrationSF(dm, tsf, sf);CHKERRQ(ierr);
    ierr = PetscSFDestroy(&tsf);CHKERRQ(ierr);
  }
  ierr = PetscSFDestroy(&migrationSF);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&gatherSF);CHKERRQ(ierr);
  ierr = DMDestroy(&gatherDM);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
