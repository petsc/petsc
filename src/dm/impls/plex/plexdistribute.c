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
+ user    - The user callback
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
  PetscValidBoolPointer(useAnchors, 2);
  *useAnchors = mesh->useAnchors;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexGetAdjacency_Cone_Internal(DM dm, PetscInt p, PetscInt *adjSize, PetscInt adj[])
{
  const PetscInt *cone = NULL;
  PetscInt        numAdj = 0, maxAdjSize = *adjSize, coneSize, c;

  PetscFunctionBeginHot;
  PetscCall(DMPlexGetConeSize(dm, p, &coneSize));
  PetscCall(DMPlexGetCone(dm, p, &cone));
  for (c = 0; c <= coneSize; ++c) {
    const PetscInt  point   = !c ? p : cone[c-1];
    const PetscInt *support = NULL;
    PetscInt        supportSize, s, q;

    PetscCall(DMPlexGetSupportSize(dm, point, &supportSize));
    PetscCall(DMPlexGetSupport(dm, point, &support));
    for (s = 0; s < supportSize; ++s) {
      for (q = 0; q < numAdj || ((void)(adj[numAdj++] = support[s]),0); ++q) {
        if (support[s] == adj[q]) break;
      }
      PetscCheckFalse(numAdj > maxAdjSize,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid mesh exceeded adjacency allocation (%D)", maxAdjSize);
    }
  }
  *adjSize = numAdj;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexGetAdjacency_Support_Internal(DM dm, PetscInt p, PetscInt *adjSize, PetscInt adj[])
{
  const PetscInt *support = NULL;
  PetscInt        numAdj   = 0, maxAdjSize = *adjSize, supportSize, s;

  PetscFunctionBeginHot;
  PetscCall(DMPlexGetSupportSize(dm, p, &supportSize));
  PetscCall(DMPlexGetSupport(dm, p, &support));
  for (s = 0; s <= supportSize; ++s) {
    const PetscInt  point = !s ? p : support[s-1];
    const PetscInt *cone  = NULL;
    PetscInt        coneSize, c, q;

    PetscCall(DMPlexGetConeSize(dm, point, &coneSize));
    PetscCall(DMPlexGetCone(dm, point, &cone));
    for (c = 0; c < coneSize; ++c) {
      for (q = 0; q < numAdj || ((void)(adj[numAdj++] = cone[c]),0); ++q) {
        if (cone[c] == adj[q]) break;
      }
      PetscCheckFalse(numAdj > maxAdjSize,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid mesh exceeded adjacency allocation (%D)", maxAdjSize);
    }
  }
  *adjSize = numAdj;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexGetAdjacency_Transitive_Internal(DM dm, PetscInt p, PetscBool useClosure, PetscInt *adjSize, PetscInt adj[])
{
  PetscInt      *star = NULL;
  PetscInt       numAdj = 0, maxAdjSize = *adjSize, starSize, s;

  PetscFunctionBeginHot;
  PetscCall(DMPlexGetTransitiveClosure(dm, p, useClosure, &starSize, &star));
  for (s = 0; s < starSize*2; s += 2) {
    const PetscInt *closure = NULL;
    PetscInt        closureSize, c, q;

    PetscCall(DMPlexGetTransitiveClosure(dm, star[s], (PetscBool)!useClosure, &closureSize, (PetscInt**) &closure));
    for (c = 0; c < closureSize*2; c += 2) {
      for (q = 0; q < numAdj || ((void)(adj[numAdj++] = closure[c]),0); ++q) {
        if (closure[c] == adj[q]) break;
      }
      PetscCheckFalse(numAdj > maxAdjSize,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid mesh exceeded adjacency allocation (%D)", maxAdjSize);
    }
    PetscCall(DMPlexRestoreTransitiveClosure(dm, star[s], (PetscBool)!useClosure, &closureSize, (PetscInt**) &closure));
  }
  PetscCall(DMPlexRestoreTransitiveClosure(dm, p, useClosure, &starSize, &star));
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

  PetscFunctionBeginHot;
  if (useAnchors) {
    PetscCall(DMPlexGetAnchors(dm,&aSec,&aIS));
    if (aSec) {
      PetscCall(PetscSectionGetMaxDof(aSec,&maxAnchors));
      maxAnchors = PetscMax(1,maxAnchors);
      PetscCall(PetscSectionGetChart(aSec,&aStart,&aEnd));
      PetscCall(ISGetIndices(aIS,&anchors));
    }
  }
  if (!*adj) {
    PetscInt depth, coneSeries, supportSeries, maxC, maxS, pStart, pEnd;

    PetscCall(DMPlexGetChart(dm, &pStart,&pEnd));
    PetscCall(DMPlexGetDepth(dm, &depth));
    depth = PetscMax(depth, -depth);
    PetscCall(DMPlexGetMaxSizes(dm, &maxC, &maxS));
    coneSeries    = (maxC > 1) ? ((PetscPowInt(maxC,depth+1)-1)/(maxC-1)) : depth+1;
    supportSeries = (maxS > 1) ? ((PetscPowInt(maxS,depth+1)-1)/(maxS-1)) : depth+1;
    asiz  = PetscMax(PetscPowInt(maxS,depth)*coneSeries,PetscPowInt(maxC,depth)*supportSeries);
    asiz *= maxAnchors;
    asiz  = PetscMin(asiz,pEnd-pStart);
    PetscCall(PetscMalloc1(asiz,adj));
  }
  if (*adjSize < 0) *adjSize = asiz;
  maxAdjSize = *adjSize;
  if (mesh->useradjacency) {
    PetscCall((*mesh->useradjacency)(dm, p, adjSize, *adj, mesh->useradjacencyctx));
  } else if (useTransitiveClosure) {
    PetscCall(DMPlexGetAdjacency_Transitive_Internal(dm, p, useCone, adjSize, *adj));
  } else if (useCone) {
    PetscCall(DMPlexGetAdjacency_Cone_Internal(dm, p, adjSize, *adj));
  } else {
    PetscCall(DMPlexGetAdjacency_Support_Internal(dm, p, adjSize, *adj));
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
        PetscCall(PetscSectionGetDof(aSec,p,&aDof));
      }
      if (aDof) {
        PetscInt aOff;
        PetscInt s, q;

        for (j = i + 1; j < numAdj; j++) {
          orig[j - 1] = orig[j];
        }
        origSize--;
        numAdj--;
        PetscCall(PetscSectionGetOffset(aSec,p,&aOff));
        for (s = 0; s < aDof; ++s) {
          for (q = 0; q < numAdj || ((void)(orig[numAdj++] = anchors[aOff+s]),0); ++q) {
            if (anchors[aOff+s] == orig[q]) break;
          }
          PetscCheckFalse(numAdj > maxAdjSize,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid mesh exceeded adjacency allocation (%D)", maxAdjSize);
        }
      }
      else {
        i++;
      }
    }
    *adjSize = numAdj;
    PetscCall(ISRestoreIndices(aIS,&anchors));
  }
  PetscFunctionReturn(0);
}

/*@
  DMPlexGetAdjacency - Return all points adjacent to the given point

  Input Parameters:
+ dm - The DM object
- p  - The point

  Input/Output Parameters:
+ adjSize - The maximum size of adj if it is non-NULL, or PETSC_DETERMINE;
            on output the number of adjacent points
- adj - Either NULL so that the array is allocated, or an existing array with size adjSize;
        on output contains the adjacent points

  Level: advanced

  Notes:
    The user must PetscFree the adj array if it was not passed in.

.seealso: DMSetAdjacency(), DMPlexDistribute(), DMCreateMatrix(), DMPlexPreallocateOperator()
@*/
PetscErrorCode DMPlexGetAdjacency(DM dm, PetscInt p, PetscInt *adjSize, PetscInt *adj[])
{
  PetscBool      useCone, useClosure, useAnchors;

  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidIntPointer(adjSize,3);
  PetscValidPointer(adj,4);
  PetscCall(DMGetBasicAdjacency(dm, &useCone, &useClosure));
  PetscCall(DMPlexGetAdjacencyUseAnchors(dm, &useAnchors));
  PetscCall(DMPlexGetAdjacency_Internal(dm, p, useCone, useClosure, useAnchors, adjSize, adj));
  PetscFunctionReturn(0);
}

/*@
  DMPlexCreateTwoSidedProcessSF - Create an SF which just has process connectivity

  Collective on dm

  Input Parameters:
+ dm      - The DM
. sfPoint - The PetscSF which encodes point connectivity
. rootRankSection -
. rootRanks -
. leftRankSection -
- leafRanks -

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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(sfPoint, PETSCSF_CLASSID, 2);
  if (processRanks) {PetscValidPointer(processRanks, 7);}
  if (sfProcess)    {PetscValidPointer(sfProcess, 8);}
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject) dm), &size));
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject) dm), &rank));
  PetscCall(PetscSFGetGraph(sfPoint, NULL, &numLeaves, NULL, &remotePoints));
  PetscCall(PetscBTCreate(size, &neighbors));
  PetscCall(PetscBTMemzero(size, neighbors));
  /* Compute root-to-leaf process connectivity */
  PetscCall(PetscSectionGetChart(rootRankSection, &pStart, &pEnd));
  PetscCall(ISGetIndices(rootRanks, &nranks));
  for (p = pStart; p < pEnd; ++p) {
    PetscInt ndof, noff, n;

    PetscCall(PetscSectionGetDof(rootRankSection, p, &ndof));
    PetscCall(PetscSectionGetOffset(rootRankSection, p, &noff));
    for (n = 0; n < ndof; ++n) PetscCall(PetscBTSet(neighbors, nranks[noff+n]));
  }
  PetscCall(ISRestoreIndices(rootRanks, &nranks));
  /* Compute leaf-to-neighbor process connectivity */
  PetscCall(PetscSectionGetChart(leafRankSection, &pStart, &pEnd));
  PetscCall(ISGetIndices(leafRanks, &nranks));
  for (p = pStart; p < pEnd; ++p) {
    PetscInt ndof, noff, n;

    PetscCall(PetscSectionGetDof(leafRankSection, p, &ndof));
    PetscCall(PetscSectionGetOffset(leafRankSection, p, &noff));
    for (n = 0; n < ndof; ++n) PetscCall(PetscBTSet(neighbors, nranks[noff+n]));
  }
  PetscCall(ISRestoreIndices(leafRanks, &nranks));
  /* Compute leaf-to-root process connectivity */
  for (l = 0; l < numLeaves; ++l) {PetscBTSet(neighbors, remotePoints[l].rank);}
  /* Calculate edges */
  PetscBTClear(neighbors, rank);
  for (proc = 0, numNeighbors = 0; proc < size; ++proc) {if (PetscBTLookup(neighbors, proc)) ++numNeighbors;}
  PetscCall(PetscMalloc1(numNeighbors, &ranksNew));
  PetscCall(PetscMalloc1(numNeighbors, &localPointsNew));
  PetscCall(PetscMalloc1(numNeighbors, &remotePointsNew));
  for (proc = 0, n = 0; proc < size; ++proc) {
    if (PetscBTLookup(neighbors, proc)) {
      ranksNew[n]              = proc;
      localPointsNew[n]        = proc;
      remotePointsNew[n].index = rank;
      remotePointsNew[n].rank  = proc;
      ++n;
    }
  }
  PetscCall(PetscBTDestroy(&neighbors));
  if (processRanks) PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)dm), numNeighbors, ranksNew, PETSC_OWN_POINTER, processRanks));
  else              PetscCall(PetscFree(ranksNew));
  if (sfProcess) {
    PetscCall(PetscSFCreate(PetscObjectComm((PetscObject)dm), sfProcess));
    PetscCall(PetscObjectSetName((PetscObject) *sfProcess, "Two-Sided Process SF"));
    PetscCall(PetscSFSetFromOptions(*sfProcess));
    PetscCall(PetscSFSetGraph(*sfProcess, size, numNeighbors, localPointsNew, PETSC_OWN_POINTER, remotePointsNew, PETSC_OWN_POINTER));
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

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject) dm, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
  PetscCall(DMGetPointSF(dm, &sfPoint));
  /* Compute number of leaves for each root */
  PetscCall(PetscObjectSetName((PetscObject) rootSection, "Root Section"));
  PetscCall(PetscSectionSetChart(rootSection, pStart, pEnd));
  PetscCall(PetscSFComputeDegreeBegin(sfPoint, &rootdegree));
  PetscCall(PetscSFComputeDegreeEnd(sfPoint, &rootdegree));
  for (p = pStart; p < pEnd; ++p) PetscCall(PetscSectionSetDof(rootSection, p, rootdegree[p-pStart]));
  PetscCall(PetscSectionSetUp(rootSection));
  /* Gather rank of each leaf to root */
  PetscCall(PetscSectionGetStorageSize(rootSection, &nedges));
  PetscCall(PetscMalloc1(pEnd-pStart, &myrank));
  PetscCall(PetscMalloc1(nedges,  &remoterank));
  for (p = 0; p < pEnd-pStart; ++p) myrank[p] = rank;
  PetscCall(PetscSFGatherBegin(sfPoint, MPIU_INT, myrank, remoterank));
  PetscCall(PetscSFGatherEnd(sfPoint, MPIU_INT, myrank, remoterank));
  PetscCall(PetscFree(myrank));
  PetscCall(ISCreateGeneral(comm, nedges, remoterank, PETSC_OWN_POINTER, rootrank));
  /* Distribute remote ranks to leaves */
  PetscCall(PetscObjectSetName((PetscObject) leafSection, "Leaf Section"));
  PetscCall(DMPlexDistributeFieldIS(dm, sfPoint, rootSection, *rootrank, leafSection, leafrank));
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

  Output Parameter:
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

  PetscFunctionBegin;
  *ovLabel = NULL;
  PetscCall(PetscObjectGetComm((PetscObject) dm, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  if (size == 1) PetscFunctionReturn(0);
  PetscCall(DMGetPointSF(dm, &sfPoint));
  PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
  if (!levels) {
    /* Add owned points */
    PetscCall(DMLabelCreate(PETSC_COMM_SELF, "Overlap label", ovLabel));
    for (p = pStart; p < pEnd; ++p) PetscCall(DMLabelSetValue(*ovLabel, p, rank));
    PetscFunctionReturn(0);
  }
  PetscCall(PetscSectionGetChart(leafSection, &sStart, &sEnd));
  PetscCall(PetscSFGetGraph(sfPoint, NULL, &nleaves, &local, &remote));
  PetscCall(DMLabelCreate(PETSC_COMM_SELF, "Overlap adjacency", &ovAdjByRank));
  /* Handle leaves: shared with the root point */
  for (l = 0; l < nleaves; ++l) {
    PetscInt adjSize = PETSC_DETERMINE, a;

    PetscCall(DMPlexGetAdjacency(dm, local ? local[l] : l, &adjSize, &adj));
    for (a = 0; a < adjSize; ++a) PetscCall(DMLabelSetValue(ovAdjByRank, adj[a], remote[l].rank));
  }
  PetscCall(ISGetIndices(rootrank, &rrank));
  PetscCall(ISGetIndices(leafrank, &nrank));
  /* Handle roots */
  for (p = pStart; p < pEnd; ++p) {
    PetscInt adjSize = PETSC_DETERMINE, neighbors = 0, noff, n, a;

    if ((p >= sStart) && (p < sEnd)) {
      /* Some leaves share a root with other leaves on different processes */
      PetscCall(PetscSectionGetDof(leafSection, p, &neighbors));
      if (neighbors) {
        PetscCall(PetscSectionGetOffset(leafSection, p, &noff));
        PetscCall(DMPlexGetAdjacency(dm, p, &adjSize, &adj));
        for (n = 0; n < neighbors; ++n) {
          const PetscInt remoteRank = nrank[noff+n];

          if (remoteRank == rank) continue;
          for (a = 0; a < adjSize; ++a) PetscCall(DMLabelSetValue(ovAdjByRank, adj[a], remoteRank));
        }
      }
    }
    /* Roots are shared with leaves */
    PetscCall(PetscSectionGetDof(rootSection, p, &neighbors));
    if (!neighbors) continue;
    PetscCall(PetscSectionGetOffset(rootSection, p, &noff));
    PetscCall(DMPlexGetAdjacency(dm, p, &adjSize, &adj));
    for (n = 0; n < neighbors; ++n) {
      const PetscInt remoteRank = rrank[noff+n];

      if (remoteRank == rank) continue;
      for (a = 0; a < adjSize; ++a) PetscCall(DMLabelSetValue(ovAdjByRank, adj[a], remoteRank));
    }
  }
  PetscCall(PetscFree(adj));
  PetscCall(ISRestoreIndices(rootrank, &rrank));
  PetscCall(ISRestoreIndices(leafrank, &nrank));
  /* Add additional overlap levels */
  for (l = 1; l < levels; l++) {
    /* Propagate point donations over SF to capture remote connections */
    PetscCall(DMPlexPartitionLabelPropagate(dm, ovAdjByRank));
    /* Add next level of point donations to the label */
    PetscCall(DMPlexPartitionLabelAdjacency(dm, ovAdjByRank));
  }
  /* We require the closure in the overlap */
  PetscCall(DMPlexPartitionLabelClosure(dm, ovAdjByRank));
  PetscCall(PetscOptionsHasName(((PetscObject) dm)->options,((PetscObject) dm)->prefix, "-overlap_view", &flg));
  if (flg) {
    PetscViewer viewer;
    PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)dm), &viewer));
    PetscCall(DMLabelView(ovAdjByRank, viewer));
  }
  /* Invert sender to receiver label */
  PetscCall(DMLabelCreate(PETSC_COMM_SELF, "Overlap label", ovLabel));
  PetscCall(DMPlexPartitionLabelInvert(dm, ovAdjByRank, NULL, *ovLabel));
  /* Add owned points, except for shared local points */
  for (p = pStart; p < pEnd; ++p) PetscCall(DMLabelSetValue(*ovLabel, p, rank));
  for (l = 0; l < nleaves; ++l) {
    PetscCall(DMLabelClearValue(*ovLabel, local[l], rank));
    PetscCall(DMLabelSetValue(*ovLabel, remote[l].index, remote[l].rank));
  }
  /* Clean up */
  PetscCall(DMLabelDestroy(&ovAdjByRank));
  PetscFunctionReturn(0);
}

/*@C
  DMPlexCreateOverlapMigrationSF - Create an SF describing the new mesh distribution to make the overlap described by the input SF

  Collective on dm

  Input Parameters:
+ dm          - The DM
- overlapSF   - The SF mapping ghost points in overlap to owner points on other processes

  Output Parameter:
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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCall(DMGetDimension(dm, &dim));

  /* Before building the migration SF we need to know the new stratum offsets */
  PetscCall(PetscSFGetGraph(overlapSF, &nroots, &nleaves, NULL, &overlapRemote));
  PetscCall(PetscMalloc2(nroots, &pointDepths, nleaves, &remoteDepths));
  for (d=0; d<dim+1; d++) {
    PetscCall(DMPlexGetDepthStratum(dm, d, &pStart, &pEnd));
    for (p=pStart; p<pEnd; p++) pointDepths[p] = d;
  }
  for (p=0; p<nleaves; p++) remoteDepths[p] = -1;
  PetscCall(PetscSFBcastBegin(overlapSF, MPIU_INT, pointDepths, remoteDepths,MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(overlapSF, MPIU_INT, pointDepths, remoteDepths,MPI_REPLACE));

  /* Count received points in each stratum and compute the internal strata shift */
  PetscCall(PetscMalloc3(dim+1, &depthRecv, dim+1, &depthShift, dim+1, &depthIdx));
  for (d=0; d<dim+1; d++) depthRecv[d]=0;
  for (p=0; p<nleaves; p++) depthRecv[remoteDepths[p]]++;
  depthShift[dim] = 0;
  for (d=0; d<dim; d++) depthShift[d] = depthRecv[dim];
  for (d=1; d<dim; d++) depthShift[d] += depthRecv[0];
  for (d=dim-2; d>0; d--) depthShift[d] += depthRecv[d+1];
  for (d=0; d<dim+1; d++) {
    PetscCall(DMPlexGetDepthStratum(dm, d, &pStart, &pEnd));
    depthIdx[d] = pStart + depthShift[d];
  }

  /* Form the overlap SF build an SF that describes the full overlap migration SF */
  PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
  newLeaves = pEnd - pStart + nleaves;
  PetscCall(PetscMalloc1(newLeaves, &ilocal));
  PetscCall(PetscMalloc1(newLeaves, &iremote));
  /* First map local points to themselves */
  for (d=0; d<dim+1; d++) {
    PetscCall(DMPlexGetDepthStratum(dm, d, &pStart, &pEnd));
    for (p=pStart; p<pEnd; p++) {
      point = p + depthShift[d];
      ilocal[point] = point;
      iremote[point].index = p;
      iremote[point].rank = rank;
      depthIdx[d]++;
    }
  }

  /* Add in the remote roots for currently shared points */
  PetscCall(DMGetPointSF(dm, &pointSF));
  PetscCall(PetscSFGetGraph(pointSF, NULL, &numSharedPoints, &sharedLocal, &sharedRemote));
  for (d=0; d<dim+1; d++) {
    PetscCall(DMPlexGetDepthStratum(dm, d, &pStart, &pEnd));
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
  PetscCall(PetscFree2(pointDepths,remoteDepths));

  PetscCall(PetscSFCreate(comm, migrationSF));
  PetscCall(PetscObjectSetName((PetscObject) *migrationSF, "Overlap Migration SF"));
  PetscCall(PetscSFSetFromOptions(*migrationSF));
  PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
  PetscCall(PetscSFSetGraph(*migrationSF, pEnd-pStart, newLeaves, ilocal, PETSC_OWN_POINTER, iremote, PETSC_OWN_POINTER));

  PetscCall(PetscFree3(depthRecv, depthShift, depthIdx));
  PetscFunctionReturn(0);
}

/*@
  DMPlexStratifyMigrationSF - Rearrange the leaves of a migration sf for stratification.

  Input Parameters:
+ dm          - The DM
- sf          - A star forest with non-ordered leaves, usually defining a DM point migration

  Output Parameter:
. migrationSF - A star forest with added leaf indirection that ensures the resulting DM is stratified

  Note:
  This lexicographically sorts by (depth, cellType)

  Level: developer

.seealso: DMPlexPartitionLabelCreateSF(), DMPlexDistribute(), DMPlexDistributeOverlap()
@*/
PetscErrorCode DMPlexStratifyMigrationSF(DM dm, PetscSF sf, PetscSF *migrationSF)
{
  MPI_Comm           comm;
  PetscMPIInt        rank, size;
  PetscInt           d, ldepth, depth, dim, p, pStart, pEnd, nroots, nleaves;
  PetscSFNode       *pointDepths, *remoteDepths;
  PetscInt          *ilocal;
  PetscInt          *depthRecv, *depthShift, *depthIdx;
  PetscInt          *ctRecv,    *ctShift,    *ctIdx;
  const PetscSFNode *iremote;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(PetscObjectGetComm((PetscObject) dm, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCall(DMPlexGetDepth(dm, &ldepth));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(MPIU_Allreduce(&ldepth, &depth, 1, MPIU_INT, MPI_MAX, comm));
  PetscCheckFalse((ldepth >= 0) && (depth != ldepth),PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Inconsistent Plex depth %d != %d", ldepth, depth);
  PetscCall(PetscLogEventBegin(DMPLEX_PartStratSF,dm,0,0,0));

  /* Before building the migration SF we need to know the new stratum offsets */
  PetscCall(PetscSFGetGraph(sf, &nroots, &nleaves, NULL, &iremote));
  PetscCall(PetscMalloc2(nroots, &pointDepths, nleaves, &remoteDepths));
  for (d = 0; d < depth+1; ++d) {
    PetscCall(DMPlexGetDepthStratum(dm, d, &pStart, &pEnd));
    for (p = pStart; p < pEnd; ++p) {
      DMPolytopeType ct;

      PetscCall(DMPlexGetCellType(dm, p, &ct));
      pointDepths[p].index = d;
      pointDepths[p].rank  = ct;
    }
  }
  for (p = 0; p < nleaves; ++p) {remoteDepths[p].index = -1; remoteDepths[p].rank = -1;}
  PetscCall(PetscSFBcastBegin(sf, MPIU_2INT, pointDepths, remoteDepths,MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(sf, MPIU_2INT, pointDepths, remoteDepths,MPI_REPLACE));
  /* Count received points in each stratum and compute the internal strata shift */
  PetscCall(PetscCalloc6(depth+1, &depthRecv, depth+1, &depthShift, depth+1, &depthIdx, DM_NUM_POLYTOPES, &ctRecv, DM_NUM_POLYTOPES, &ctShift, DM_NUM_POLYTOPES, &ctIdx));
  for (p = 0; p < nleaves; ++p) {
    if (remoteDepths[p].rank < 0) {
      ++depthRecv[remoteDepths[p].index];
    } else {
      ++ctRecv[remoteDepths[p].rank];
    }
  }
  {
    PetscInt depths[4], dims[4], shift = 0, i, c;

    /* Cells (depth), Vertices (0), Faces (depth-1), Edges (1)
         Consider DM_POLYTOPE_FV_GHOST and DM_POLYTOPE_INTERIOR_GHOST as cells
     */
    depths[0] = depth; depths[1] = 0; depths[2] = depth-1; depths[3] = 1;
    dims[0]   = dim;   dims[1]   = 0; dims[2]   = dim-1;   dims[3]   = 1;
    for (i = 0; i <= depth; ++i) {
      const PetscInt dep = depths[i];
      const PetscInt dim = dims[i];

      for (c = 0; c < DM_NUM_POLYTOPES; ++c) {
        if (DMPolytopeTypeGetDim((DMPolytopeType) c) != dim && !(i == 0 && (c == DM_POLYTOPE_FV_GHOST || c == DM_POLYTOPE_INTERIOR_GHOST))) continue;
        ctShift[c] = shift;
        shift     += ctRecv[c];
      }
      depthShift[dep] = shift;
      shift          += depthRecv[dep];
    }
    for (c = 0; c < DM_NUM_POLYTOPES; ++c) {
      const PetscInt ctDim = DMPolytopeTypeGetDim((DMPolytopeType) c);

      if ((ctDim < 0 || ctDim > dim) && (c != DM_POLYTOPE_FV_GHOST && c != DM_POLYTOPE_INTERIOR_GHOST)) {
        ctShift[c] = shift;
        shift     += ctRecv[c];
      }
    }
  }
  /* Derive a new local permutation based on stratified indices */
  PetscCall(PetscMalloc1(nleaves, &ilocal));
  for (p = 0; p < nleaves; ++p) {
    const PetscInt       dep = remoteDepths[p].index;
    const DMPolytopeType ct  = (DMPolytopeType) remoteDepths[p].rank;

    if ((PetscInt) ct < 0) {
      ilocal[p] = depthShift[dep] + depthIdx[dep];
      ++depthIdx[dep];
    } else {
      ilocal[p] = ctShift[ct] + ctIdx[ct];
      ++ctIdx[ct];
    }
  }
  PetscCall(PetscSFCreate(comm, migrationSF));
  PetscCall(PetscObjectSetName((PetscObject) *migrationSF, "Migration SF"));
  PetscCall(PetscSFSetGraph(*migrationSF, nroots, nleaves, ilocal, PETSC_OWN_POINTER, (PetscSFNode*)iremote, PETSC_COPY_VALUES));
  PetscCall(PetscFree2(pointDepths,remoteDepths));
  PetscCall(PetscFree6(depthRecv, depthShift, depthIdx, ctRecv, ctShift, ctIdx));
  PetscCall(PetscLogEventEnd(DMPLEX_PartStratSF,dm,0,0,0));
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

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(DMPLEX_DistributeField,dm,0,0,0));
  PetscCall(PetscSFDistributeSection(pointSF, originalSection, &remoteOffsets, newSection));

  PetscCall(PetscSectionGetStorageSize(newSection, &fieldSize));
  PetscCall(VecSetSizes(newVec, fieldSize, PETSC_DETERMINE));
  PetscCall(VecSetType(newVec,dm->vectype));

  PetscCall(VecGetArray(originalVec, &originalValues));
  PetscCall(VecGetArray(newVec, &newValues));
  PetscCall(PetscSFCreateSectionSF(pointSF, originalSection, remoteOffsets, newSection, &fieldSF));
  PetscCall(PetscFree(remoteOffsets));
  PetscCall(PetscSFBcastBegin(fieldSF, MPIU_SCALAR, originalValues, newValues,MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(fieldSF, MPIU_SCALAR, originalValues, newValues,MPI_REPLACE));
  PetscCall(PetscSFDestroy(&fieldSF));
  PetscCall(VecRestoreArray(newVec, &newValues));
  PetscCall(VecRestoreArray(originalVec, &originalValues));
  PetscCall(PetscLogEventEnd(DMPLEX_DistributeField,dm,0,0,0));
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

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(DMPLEX_DistributeField,dm,0,0,0));
  PetscCall(PetscSFDistributeSection(pointSF, originalSection, &remoteOffsets, newSection));

  PetscCall(PetscSectionGetStorageSize(newSection, &fieldSize));
  PetscCall(PetscMalloc1(fieldSize, &newValues));

  PetscCall(ISGetIndices(originalIS, &originalValues));
  PetscCall(PetscSFCreateSectionSF(pointSF, originalSection, remoteOffsets, newSection, &fieldSF));
  PetscCall(PetscFree(remoteOffsets));
  PetscCall(PetscSFBcastBegin(fieldSF, MPIU_INT, (PetscInt *) originalValues, newValues,MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(fieldSF, MPIU_INT, (PetscInt *) originalValues, newValues,MPI_REPLACE));
  PetscCall(PetscSFDestroy(&fieldSF));
  PetscCall(ISRestoreIndices(originalIS, &originalValues));
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject) pointSF), fieldSize, newValues, PETSC_OWN_POINTER, newIS));
  PetscCall(PetscLogEventEnd(DMPLEX_DistributeField,dm,0,0,0));
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

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(DMPLEX_DistributeData,dm,0,0,0));
  PetscCall(PetscSFDistributeSection(pointSF, originalSection, &remoteOffsets, newSection));

  PetscCall(PetscSectionGetStorageSize(newSection, &fieldSize));
  PetscCallMPI(MPI_Type_size(datatype, &dataSize));
  PetscCall(PetscMalloc(fieldSize * dataSize, newData));

  PetscCall(PetscSFCreateSectionSF(pointSF, originalSection, remoteOffsets, newSection, &fieldSF));
  PetscCall(PetscFree(remoteOffsets));
  PetscCall(PetscSFBcastBegin(fieldSF, datatype, originalData, *newData,MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(fieldSF, datatype, originalData, *newData,MPI_REPLACE));
  PetscCall(PetscSFDestroy(&fieldSF));
  PetscCall(PetscLogEventEnd(DMPLEX_DistributeData,dm,0,0,0));
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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(dmParallel, DM_CLASSID, 5);
  PetscCall(PetscLogEventBegin(DMPLEX_DistributeCones,dm,0,0,0));
  /* Distribute cone section */
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCall(DMPlexGetConeSection(dm, &originalConeSection));
  PetscCall(DMPlexGetConeSection(dmParallel, &newConeSection));
  PetscCall(PetscSFDistributeSection(migrationSF, originalConeSection, &remoteOffsets, newConeSection));
  PetscCall(DMSetUp(dmParallel));
  /* Communicate and renumber cones */
  PetscCall(PetscSFCreateSectionSF(migrationSF, originalConeSection, remoteOffsets, newConeSection, &coneSF));
  PetscCall(PetscFree(remoteOffsets));
  PetscCall(DMPlexGetCones(dm, &cones));
  if (original) {
    PetscInt numCones;

    PetscCall(PetscSectionGetStorageSize(originalConeSection,&numCones));
    PetscCall(PetscMalloc1(numCones,&globCones));
    PetscCall(ISLocalToGlobalMappingApplyBlock(original, numCones, cones, globCones));
  } else {
    globCones = cones;
  }
  PetscCall(DMPlexGetCones(dmParallel, &newCones));
  PetscCall(PetscSFBcastBegin(coneSF, MPIU_INT, globCones, newCones,MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(coneSF, MPIU_INT, globCones, newCones,MPI_REPLACE));
  if (original) {
    PetscCall(PetscFree(globCones));
  }
  PetscCall(PetscSectionGetStorageSize(newConeSection, &newConesSize));
  PetscCall(ISGlobalToLocalMappingApplyBlock(renumbering, IS_GTOLM_MASK, newConesSize, newCones, NULL, newCones));
  if (PetscDefined(USE_DEBUG)) {
    PetscInt  p;
    PetscBool valid = PETSC_TRUE;
    for (p = 0; p < newConesSize; ++p) {
      if (newCones[p] < 0) {valid = PETSC_FALSE; PetscCall(PetscPrintf(PETSC_COMM_SELF, "[%d] Point %D not in overlap SF\n", PetscGlobalRank,p));}
    }
    PetscCheck(valid,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid global to local map");
  }
  PetscCall(PetscOptionsHasName(((PetscObject) dm)->options,((PetscObject) dm)->prefix, "-cones_view", &flg));
  if (flg) {
    PetscCall(PetscPrintf(comm, "Serial Cone Section:\n"));
    PetscCall(PetscSectionView(originalConeSection, PETSC_VIEWER_STDOUT_(comm)));
    PetscCall(PetscPrintf(comm, "Parallel Cone Section:\n"));
    PetscCall(PetscSectionView(newConeSection, PETSC_VIEWER_STDOUT_(comm)));
    PetscCall(PetscSFView(coneSF, NULL));
  }
  PetscCall(DMPlexGetConeOrientations(dm, &cones));
  PetscCall(DMPlexGetConeOrientations(dmParallel, &newCones));
  PetscCall(PetscSFBcastBegin(coneSF, MPIU_INT, cones, newCones,MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(coneSF, MPIU_INT, cones, newCones,MPI_REPLACE));
  PetscCall(PetscSFDestroy(&coneSF));
  PetscCall(PetscLogEventEnd(DMPLEX_DistributeCones,dm,0,0,0));
  /* Create supports and stratify DMPlex */
  {
    PetscInt pStart, pEnd;

    PetscCall(PetscSectionGetChart(pmesh->coneSection, &pStart, &pEnd));
    PetscCall(PetscSectionSetChart(pmesh->supportSection, pStart, pEnd));
  }
  PetscCall(DMPlexSymmetrize(dmParallel));
  PetscCall(DMPlexStratify(dmParallel));
  {
    PetscBool useCone, useClosure, useAnchors;

    PetscCall(DMGetBasicAdjacency(dm, &useCone, &useClosure));
    PetscCall(DMSetBasicAdjacency(dmParallel, useCone, useClosure));
    PetscCall(DMPlexGetAdjacencyUseAnchors(dm, &useAnchors));
    PetscCall(DMPlexSetAdjacencyUseAnchors(dmParallel, useAnchors));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexDistributeCoordinates(DM dm, PetscSF migrationSF, DM dmParallel)
{
  MPI_Comm         comm;
  DM               cdm, cdmParallel;
  PetscSection     originalCoordSection, newCoordSection;
  Vec              originalCoordinates, newCoordinates;
  PetscInt         bs;
  PetscBool        isper;
  const char      *name;
  const PetscReal *maxCell, *L;
  const DMBoundaryType *bd;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(dmParallel, DM_CLASSID, 3);

  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCall(DMGetCoordinateSection(dm, &originalCoordSection));
  PetscCall(DMGetCoordinateSection(dmParallel, &newCoordSection));
  PetscCall(DMGetCoordinatesLocal(dm, &originalCoordinates));
  if (originalCoordinates) {
    PetscCall(VecCreate(PETSC_COMM_SELF, &newCoordinates));
    PetscCall(PetscObjectGetName((PetscObject) originalCoordinates, &name));
    PetscCall(PetscObjectSetName((PetscObject) newCoordinates, name));

    PetscCall(DMPlexDistributeField(dm, migrationSF, originalCoordSection, originalCoordinates, newCoordSection, newCoordinates));
    PetscCall(DMSetCoordinatesLocal(dmParallel, newCoordinates));
    PetscCall(VecGetBlockSize(originalCoordinates, &bs));
    PetscCall(VecSetBlockSize(newCoordinates, bs));
    PetscCall(VecDestroy(&newCoordinates));
  }
  PetscCall(DMGetPeriodicity(dm, &isper, &maxCell, &L, &bd));
  PetscCall(DMSetPeriodicity(dmParallel, isper, maxCell, L, bd));
  PetscCall(DMGetCoordinateDM(dm, &cdm));
  PetscCall(DMGetCoordinateDM(dmParallel, &cdmParallel));
  PetscCall(DMCopyDisc(cdm, cdmParallel));
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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(dmParallel, DM_CLASSID, 3);

  PetscCall(PetscLogEventBegin(DMPLEX_DistributeLabels,dm,0,0,0));
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));

  /* If the user has changed the depth label, communicate it instead */
  PetscCall(DMPlexGetDepth(dm, &depth));
  PetscCall(DMPlexGetDepthLabel(dm, &depthLabel));
  if (depthLabel) PetscCall(PetscObjectStateGet((PetscObject) depthLabel, &depthState));
  lsendDepth = mesh->depthState != depthState ? PETSC_TRUE : PETSC_FALSE;
  PetscCall(MPIU_Allreduce(&lsendDepth, &sendDepth, 1, MPIU_BOOL, MPI_LOR, comm));
  if (sendDepth) {
    PetscCall(DMPlexGetDepthLabel(dmParallel, &dmParallel->depthLabel));
    PetscCall(DMRemoveLabelBySelf(dmParallel, &dmParallel->depthLabel, PETSC_FALSE));
  }
  /* Everyone must have either the same number of labels, or none */
  PetscCall(DMGetNumLabels(dm, &numLocalLabels));
  numLabels = numLocalLabels;
  PetscCallMPI(MPI_Bcast(&numLabels, 1, MPIU_INT, 0, comm));
  if (numLabels == numLocalLabels) hasLabels = PETSC_TRUE;
  for (l = 0; l < numLabels; ++l) {
    DMLabel     label = NULL, labelNew = NULL;
    PetscBool   isDepth, lisOutput = PETSC_TRUE, isOutput;
    const char *name = NULL;

    if (hasLabels) {
      PetscCall(DMGetLabelByNum(dm, l, &label));
      /* Skip "depth" because it is recreated */
      PetscCall(PetscObjectGetName((PetscObject) label, &name));
      PetscCall(PetscStrcmp(name, "depth", &isDepth));
    }
    PetscCallMPI(MPI_Bcast(&isDepth, 1, MPIU_BOOL, 0, comm));
    if (isDepth && !sendDepth) continue;
    PetscCall(DMLabelDistribute(label, migrationSF, &labelNew));
    if (isDepth) {
      /* Put in any missing strata which can occur if users are managing the depth label themselves */
      PetscInt gdepth;

      PetscCall(MPIU_Allreduce(&depth, &gdepth, 1, MPIU_INT, MPI_MAX, comm));
      PetscCheckFalse((depth >= 0) && (gdepth != depth),PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Inconsistent Plex depth %d != %d", depth, gdepth);
      for (d = 0; d <= gdepth; ++d) {
        PetscBool has;

        PetscCall(DMLabelHasStratum(labelNew, d, &has));
        if (!has) PetscCall(DMLabelAddStratum(labelNew, d));
      }
    }
    PetscCall(DMAddLabel(dmParallel, labelNew));
    /* Put the output flag in the new label */
    if (hasLabels) PetscCall(DMGetLabelOutput(dm, name, &lisOutput));
    PetscCall(MPIU_Allreduce(&lisOutput, &isOutput, 1, MPIU_BOOL, MPI_LAND, comm));
    PetscCall(PetscObjectGetName((PetscObject) labelNew, &name));
    PetscCall(DMSetLabelOutput(dmParallel, name, isOutput));
    PetscCall(DMLabelDestroy(&labelNew));
  }
  PetscCall(PetscLogEventEnd(DMPLEX_DistributeLabels,dm,0,0,0));
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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(dmParallel, DM_CLASSID, 5);
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));

  /* Set up tree */
  PetscCall(DMPlexGetReferenceTree(dm,&refTree));
  PetscCall(DMPlexSetReferenceTree(dmParallel,refTree));
  PetscCall(DMPlexGetTree(dm,&origParentSection,&origParents,&origChildIDs,NULL,NULL));
  if (origParentSection) {
    PetscInt        pStart, pEnd;
    PetscInt        *newParents, *newChildIDs, *globParents;
    PetscInt        *remoteOffsetsParents, newParentSize;
    PetscSF         parentSF;

    PetscCall(DMPlexGetChart(dmParallel, &pStart, &pEnd));
    PetscCall(PetscSectionCreate(PetscObjectComm((PetscObject)dmParallel),&newParentSection));
    PetscCall(PetscSectionSetChart(newParentSection,pStart,pEnd));
    PetscCall(PetscSFDistributeSection(migrationSF, origParentSection, &remoteOffsetsParents, newParentSection));
    PetscCall(PetscSFCreateSectionSF(migrationSF, origParentSection, remoteOffsetsParents, newParentSection, &parentSF));
    PetscCall(PetscFree(remoteOffsetsParents));
    PetscCall(PetscSectionGetStorageSize(newParentSection,&newParentSize));
    PetscCall(PetscMalloc2(newParentSize,&newParents,newParentSize,&newChildIDs));
    if (original) {
      PetscInt numParents;

      PetscCall(PetscSectionGetStorageSize(origParentSection,&numParents));
      PetscCall(PetscMalloc1(numParents,&globParents));
      PetscCall(ISLocalToGlobalMappingApplyBlock(original, numParents, origParents, globParents));
    }
    else {
      globParents = origParents;
    }
    PetscCall(PetscSFBcastBegin(parentSF, MPIU_INT, globParents, newParents,MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(parentSF, MPIU_INT, globParents, newParents,MPI_REPLACE));
    if (original) {
      PetscCall(PetscFree(globParents));
    }
    PetscCall(PetscSFBcastBegin(parentSF, MPIU_INT, origChildIDs, newChildIDs,MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(parentSF, MPIU_INT, origChildIDs, newChildIDs,MPI_REPLACE));
    PetscCall(ISGlobalToLocalMappingApplyBlock(renumbering,IS_GTOLM_MASK, newParentSize, newParents, NULL, newParents));
    if (PetscDefined(USE_DEBUG)) {
      PetscInt  p;
      PetscBool valid = PETSC_TRUE;
      for (p = 0; p < newParentSize; ++p) {
        if (newParents[p] < 0) {valid = PETSC_FALSE; PetscCall(PetscPrintf(PETSC_COMM_SELF, "Point %d not in overlap SF\n", p));}
      }
      PetscCheck(valid,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid global to local map");
    }
    PetscCall(PetscOptionsHasName(((PetscObject) dm)->options,((PetscObject) dm)->prefix, "-parents_view", &flg));
    if (flg) {
      PetscCall(PetscPrintf(comm, "Serial Parent Section: \n"));
      PetscCall(PetscSectionView(origParentSection, PETSC_VIEWER_STDOUT_(comm)));
      PetscCall(PetscPrintf(comm, "Parallel Parent Section: \n"));
      PetscCall(PetscSectionView(newParentSection, PETSC_VIEWER_STDOUT_(comm)));
      PetscCall(PetscSFView(parentSF, NULL));
    }
    PetscCall(DMPlexSetTree(dmParallel,newParentSection,newParents,newChildIDs));
    PetscCall(PetscSectionDestroy(&newParentSection));
    PetscCall(PetscFree2(newParents,newChildIDs));
    PetscCall(PetscSFDestroy(&parentSF));
  }
  pmesh->useAnchors = mesh->useAnchors;
  PetscFunctionReturn(0);
}

PETSC_UNUSED static PetscErrorCode DMPlexDistributeSF(DM dm, PetscSF migrationSF, DM dmParallel)
{
  PetscMPIInt            rank, size;
  MPI_Comm               comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(dmParallel, DM_CLASSID, 3);

  /* Create point SF for parallel mesh */
  PetscCall(PetscLogEventBegin(DMPLEX_DistributeSF,dm,0,0,0));
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  {
    const PetscInt *leaves;
    PetscSFNode    *remotePoints, *rowners, *lowners;
    PetscInt        numRoots, numLeaves, numGhostPoints = 0, p, gp, *ghostPoints;
    PetscInt        pStart, pEnd;

    PetscCall(DMPlexGetChart(dmParallel, &pStart, &pEnd));
    PetscCall(PetscSFGetGraph(migrationSF, &numRoots, &numLeaves, &leaves, NULL));
    PetscCall(PetscMalloc2(numRoots,&rowners,numLeaves,&lowners));
    for (p=0; p<numRoots; p++) {
      rowners[p].rank  = -1;
      rowners[p].index = -1;
    }
    PetscCall(PetscSFBcastBegin(migrationSF, MPIU_2INT, rowners, lowners,MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(migrationSF, MPIU_2INT, rowners, lowners,MPI_REPLACE));
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
    PetscCall(PetscSFReduceBegin(migrationSF, MPIU_2INT, lowners, rowners, MPI_MAXLOC));
    PetscCall(PetscSFReduceEnd(migrationSF, MPIU_2INT, lowners, rowners, MPI_MAXLOC));
    PetscCall(PetscSFBcastBegin(migrationSF, MPIU_2INT, rowners, lowners,MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(migrationSF, MPIU_2INT, rowners, lowners,MPI_REPLACE));
    for (p = 0; p < numLeaves; ++p) {
      PetscCheckFalse(lowners[p].rank < 0 || lowners[p].index < 0,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Cell partition corrupt: point not claimed");
      if (lowners[p].rank != rank) ++numGhostPoints;
    }
    PetscCall(PetscMalloc1(numGhostPoints, &ghostPoints));
    PetscCall(PetscMalloc1(numGhostPoints, &remotePoints));
    for (p = 0, gp = 0; p < numLeaves; ++p) {
      if (lowners[p].rank != rank) {
        ghostPoints[gp]        = leaves ? leaves[p] : p;
        remotePoints[gp].rank  = lowners[p].rank;
        remotePoints[gp].index = lowners[p].index;
        ++gp;
      }
    }
    PetscCall(PetscFree2(rowners,lowners));
    PetscCall(PetscSFSetGraph((dmParallel)->sf, pEnd - pStart, numGhostPoints, ghostPoints, PETSC_OWN_POINTER, remotePoints, PETSC_OWN_POINTER));
    PetscCall(PetscSFSetFromOptions((dmParallel)->sf));
  }
  {
    PetscBool useCone, useClosure, useAnchors;

    PetscCall(DMGetBasicAdjacency(dm, &useCone, &useClosure));
    PetscCall(DMSetBasicAdjacency(dmParallel, useCone, useClosure));
    PetscCall(DMPlexGetAdjacencyUseAnchors(dm, &useAnchors));
    PetscCall(DMPlexSetAdjacencyUseAnchors(dmParallel, useAnchors));
  }
  PetscCall(PetscLogEventEnd(DMPLEX_DistributeSF,dm,0,0,0));
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
static void MPIAPI MaxLocCarry(void *in_, void *inout_, PetscMPIInt *len_, MPI_Datatype *dtype)
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

  Input Parameters:
+ dm          - The source DMPlex object
. migrationSF - The star forest that describes the parallel point remapping
- ownership   - Flag causing a vote to determine point ownership

  Output Parameter:
. pointSF     - The star forest describing the point overlap in the remapped DM

  Notes:
  Output pointSF is guaranteed to have the array of local indices (leaves) sorted.

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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject) dm), &rank));
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject) dm), &size));
  PetscCall(PetscLogEventBegin(DMPLEX_CreatePointSF,dm,0,0,0));

  PetscCall(DMPlexGetPartitionBalance(dm, &balance));
  PetscCall(PetscSFGetGraph(migrationSF, &nroots, &nleaves, &leaves, &roots));
  PetscCall(PetscMalloc2(nroots, &rootNodes, nleaves, &leafNodes));
  if (ownership) {
    MPI_Op       op;
    MPI_Datatype datatype;
    Petsc3Int   *rootVote = NULL, *leafVote = NULL;
    /* If balancing, we compute a random cyclic shift of the rank for each remote point. That way, the max will evenly distribute among ranks. */
    if (balance) {
      PetscRandom r;

      PetscCall(PetscRandomCreate(PETSC_COMM_SELF, &r));
      PetscCall(PetscRandomSetInterval(r, 0, 2467*size));
      PetscCall(VecCreate(PETSC_COMM_SELF, &shifts));
      PetscCall(VecSetSizes(shifts, numShifts, numShifts));
      PetscCall(VecSetType(shifts, VECSTANDARD));
      PetscCall(VecSetRandom(shifts, r));
      PetscCall(PetscRandomDestroy(&r));
      PetscCall(VecGetArrayRead(shifts, &shift));
    }

    PetscCall(PetscMalloc1(nroots, &rootVote));
    PetscCall(PetscMalloc1(nleaves, &leafVote));
    /* Point ownership vote: Process with highest rank owns shared points */
    for (p = 0; p < nleaves; ++p) {
      if (shiftDebug) {
        PetscCall(PetscSynchronizedPrintf(PetscObjectComm((PetscObject) dm), "[%d] Point %D RemotePoint %D Shift %D MyRank %D\n", rank, leaves ? leaves[p] : p, roots[p].index, (PetscInt) PetscRealPart(shift[roots[p].index%numShifts]), (rank + (shift ? (PetscInt) PetscRealPart(shift[roots[p].index%numShifts]) : 0))%size));
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
    PetscCallMPI(MPI_Type_contiguous(3, MPIU_INT, &datatype));
    PetscCallMPI(MPI_Type_commit(&datatype));
    PetscCallMPI(MPI_Op_create(&MaxLocCarry, 1, &op));
    PetscCall(PetscSFReduceBegin(migrationSF, datatype, leafVote, rootVote, op));
    PetscCall(PetscSFReduceEnd(migrationSF, datatype, leafVote, rootVote, op));
    PetscCallMPI(MPI_Op_free(&op));
    PetscCallMPI(MPI_Type_free(&datatype));
    for (p = 0; p < nroots; p++) {
      rootNodes[p].rank = rootVote[p].rank;
      rootNodes[p].index = rootVote[p].index;
    }
    PetscCall(PetscFree(leafVote));
    PetscCall(PetscFree(rootVote));
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
  PetscCall(PetscSFBcastBegin(migrationSF, MPIU_2INT, rootNodes, leafNodes,MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(migrationSF, MPIU_2INT, rootNodes, leafNodes,MPI_REPLACE));

  for (npointLeaves = 0, p = 0; p < nleaves; p++) {
    if (leafNodes[p].rank != rank) npointLeaves++;
  }
  PetscCall(PetscMalloc1(npointLeaves, &pointLocal));
  PetscCall(PetscMalloc1(npointLeaves, &pointRemote));
  for (idx = 0, p = 0; p < nleaves; p++) {
    if (leafNodes[p].rank != rank) {
      /* Note that pointLocal is automatically sorted as it is sublist of 0, ..., nleaves-1 */
      pointLocal[idx] = p;
      pointRemote[idx] = leafNodes[p];
      idx++;
    }
  }
  if (shift) {
    PetscCall(VecRestoreArrayRead(shifts, &shift));
    PetscCall(VecDestroy(&shifts));
  }
  if (shiftDebug) PetscCall(PetscSynchronizedFlush(PetscObjectComm((PetscObject) dm), PETSC_STDOUT));
  PetscCall(PetscSFCreate(PetscObjectComm((PetscObject) dm), pointSF));
  PetscCall(PetscSFSetFromOptions(*pointSF));
  PetscCall(PetscSFSetGraph(*pointSF, nleaves, npointLeaves, pointLocal, PETSC_OWN_POINTER, pointRemote, PETSC_OWN_POINTER));
  PetscCall(PetscFree2(rootNodes, leafNodes));
  PetscCall(PetscLogEventEnd(DMPLEX_CreatePointSF,dm,0,0,0));
  PetscFunctionReturn(0);
}

/*@C
  DMPlexMigrate  - Migrates internal DM data over the supplied star forest

  Collective on dm

  Input Parameters:
+ dm       - The source DMPlex object
- sf       - The star forest communication context describing the migration pattern

  Output Parameter:
. targetDM - The target DMPlex object

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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(PetscLogEventBegin(DMPLEX_Migrate, dm, 0, 0, 0));
  PetscCall(PetscObjectGetComm((PetscObject) dm, &comm));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMSetDimension(targetDM, dim));
  PetscCall(DMGetCoordinateDim(dm, &cdim));
  PetscCall(DMSetCoordinateDim(targetDM, cdim));

  /* Check for a one-to-all distribution pattern */
  PetscCall(DMGetPointSF(dm, &sfPoint));
  PetscCall(PetscSFGetGraph(sfPoint, &nroots, NULL, NULL, NULL));
  if (nroots >= 0) {
    IS        isOriginal;
    PetscInt  n, size, nleaves;
    PetscInt  *numbering_orig, *numbering_new;

    /* Get the original point numbering */
    PetscCall(DMPlexCreatePointNumbering(dm, &isOriginal));
    PetscCall(ISLocalToGlobalMappingCreateIS(isOriginal, &ltogOriginal));
    PetscCall(ISLocalToGlobalMappingGetSize(ltogOriginal, &size));
    PetscCall(ISLocalToGlobalMappingGetBlockIndices(ltogOriginal, (const PetscInt**)&numbering_orig));
    /* Convert to positive global numbers */
    for (n=0; n<size; n++) {if (numbering_orig[n] < 0) numbering_orig[n] = -(numbering_orig[n]+1);}
    /* Derive the new local-to-global mapping from the old one */
    PetscCall(PetscSFGetGraph(sf, NULL, &nleaves, NULL, NULL));
    PetscCall(PetscMalloc1(nleaves, &numbering_new));
    PetscCall(PetscSFBcastBegin(sf, MPIU_INT, numbering_orig, numbering_new,MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(sf, MPIU_INT, numbering_orig, numbering_new,MPI_REPLACE));
    PetscCall(ISLocalToGlobalMappingCreate(comm, 1, nleaves, numbering_new, PETSC_OWN_POINTER, &ltogMigration));
    PetscCall(ISLocalToGlobalMappingRestoreIndices(ltogOriginal, (const PetscInt**)&numbering_orig));
    PetscCall(ISDestroy(&isOriginal));
  } else {
    /* One-to-all distribution pattern: We can derive LToG from SF */
    PetscCall(ISLocalToGlobalMappingCreateSF(sf, 0, &ltogMigration));
  }
  PetscCall(PetscOptionsHasName(((PetscObject) dm)->options,((PetscObject) dm)->prefix, "-partition_view", &flg));
  if (flg) {
    PetscCall(PetscPrintf(comm, "Point renumbering for DM migration:\n"));
    PetscCall(ISLocalToGlobalMappingView(ltogMigration, NULL));
  }
  /* Migrate DM data to target DM */
  PetscCall(DMPlexDistributeCones(dm, sf, ltogOriginal, ltogMigration, targetDM));
  PetscCall(DMPlexDistributeLabels(dm, sf, targetDM));
  PetscCall(DMPlexDistributeCoordinates(dm, sf, targetDM));
  PetscCall(DMPlexDistributeSetupTree(dm, sf, ltogOriginal, ltogMigration, targetDM));
  PetscCall(ISLocalToGlobalMappingDestroy(&ltogOriginal));
  PetscCall(ISLocalToGlobalMappingDestroy(&ltogMigration));
  PetscCall(PetscLogEventEnd(DMPLEX_Migrate, dm, 0, 0, 0));
  PetscFunctionReturn(0);
}

/*@C
  DMPlexDistribute - Distributes the mesh and any associated sections.

  Collective on dm

  Input Parameters:
+ dm  - The original DMPlex object
- overlap - The overlap of partitions, 0 is the default

  Output Parameters:
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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidLogicalCollectiveInt(dm, overlap, 2);
  if (sf) PetscValidPointer(sf,3);
  PetscValidPointer(dmParallel,4);

  if (sf) *sf = NULL;
  *dmParallel = NULL;
  PetscCall(PetscObjectGetComm((PetscObject)dm,&comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  if (size == 1) PetscFunctionReturn(0);

  PetscCall(PetscLogEventBegin(DMPLEX_Distribute,dm,0,0,0));
  /* Create cell partition */
  PetscCall(PetscLogEventBegin(DMPLEX_Partition,dm,0,0,0));
  PetscCall(PetscSectionCreate(comm, &cellPartSection));
  PetscCall(DMPlexGetPartitioner(dm, &partitioner));
  PetscCall(PetscPartitionerDMPlexPartition(partitioner, dm, NULL, cellPartSection, &cellPart));
  PetscCall(PetscLogEventBegin(DMPLEX_PartSelf,dm,0,0,0));
  {
    /* Convert partition to DMLabel */
    IS             is;
    PetscHSetI     ht;
    const PetscInt *points;
    PetscInt       *iranks;
    PetscInt       pStart, pEnd, proc, npoints, poff = 0, nranks;

    PetscCall(DMLabelCreate(PETSC_COMM_SELF, "Point Partition", &lblPartition));
    /* Preallocate strata */
    PetscCall(PetscHSetICreate(&ht));
    PetscCall(PetscSectionGetChart(cellPartSection, &pStart, &pEnd));
    for (proc = pStart; proc < pEnd; proc++) {
      PetscCall(PetscSectionGetDof(cellPartSection, proc, &npoints));
      if (npoints) PetscCall(PetscHSetIAdd(ht, proc));
    }
    PetscCall(PetscHSetIGetSize(ht, &nranks));
    PetscCall(PetscMalloc1(nranks, &iranks));
    PetscCall(PetscHSetIGetElems(ht, &poff, iranks));
    PetscCall(PetscHSetIDestroy(&ht));
    PetscCall(DMLabelAddStrata(lblPartition, nranks, iranks));
    PetscCall(PetscFree(iranks));
    /* Inline DMPlexPartitionLabelClosure() */
    PetscCall(ISGetIndices(cellPart, &points));
    PetscCall(PetscSectionGetChart(cellPartSection, &pStart, &pEnd));
    for (proc = pStart; proc < pEnd; proc++) {
      PetscCall(PetscSectionGetDof(cellPartSection, proc, &npoints));
      if (!npoints) continue;
      PetscCall(PetscSectionGetOffset(cellPartSection, proc, &poff));
      PetscCall(DMPlexClosurePoints_Private(dm, npoints, points+poff, &is));
      PetscCall(DMLabelSetStratumIS(lblPartition, proc, is));
      PetscCall(ISDestroy(&is));
    }
    PetscCall(ISRestoreIndices(cellPart, &points));
  }
  PetscCall(PetscLogEventEnd(DMPLEX_PartSelf,dm,0,0,0));

  PetscCall(DMLabelCreate(PETSC_COMM_SELF, "Point migration", &lblMigration));
  PetscCall(DMPlexPartitionLabelInvert(dm, lblPartition, NULL, lblMigration));
  PetscCall(DMPlexPartitionLabelCreateSF(dm, lblMigration, &sfMigration));
  PetscCall(DMPlexStratifyMigrationSF(dm, sfMigration, &sfStratified));
  PetscCall(PetscSFDestroy(&sfMigration));
  sfMigration = sfStratified;
  PetscCall(PetscSFSetUp(sfMigration));
  PetscCall(PetscLogEventEnd(DMPLEX_Partition,dm,0,0,0));
  PetscCall(PetscOptionsHasName(((PetscObject) dm)->options,((PetscObject) dm)->prefix, "-partition_view", &flg));
  if (flg) {
    PetscCall(DMLabelView(lblPartition, PETSC_VIEWER_STDOUT_(comm)));
    PetscCall(PetscSFView(sfMigration, PETSC_VIEWER_STDOUT_(comm)));
  }

  /* Create non-overlapping parallel DM and migrate internal data */
  PetscCall(DMPlexCreate(comm, dmParallel));
  PetscCall(PetscObjectSetName((PetscObject) *dmParallel, "Parallel Mesh"));
  PetscCall(DMPlexMigrate(dm, sfMigration, *dmParallel));

  /* Build the point SF without overlap */
  PetscCall(DMPlexGetPartitionBalance(dm, &balance));
  PetscCall(DMPlexSetPartitionBalance(*dmParallel, balance));
  PetscCall(DMPlexCreatePointSF(*dmParallel, sfMigration, PETSC_TRUE, &sfPoint));
  PetscCall(DMSetPointSF(*dmParallel, sfPoint));
  PetscCall(DMGetCoordinateDM(*dmParallel, &dmCoord));
  if (dmCoord) PetscCall(DMSetPointSF(dmCoord, sfPoint));
  if (flg) PetscCall(PetscSFView(sfPoint, NULL));

  if (overlap > 0) {
    DM                 dmOverlap;
    PetscInt           nroots, nleaves, noldleaves, l;
    const PetscInt    *oldLeaves;
    PetscSFNode       *newRemote, *permRemote;
    const PetscSFNode *oldRemote;
    PetscSF            sfOverlap, sfOverlapPoint;

    /* Add the partition overlap to the distributed DM */
    PetscCall(DMPlexDistributeOverlap(*dmParallel, overlap, &sfOverlap, &dmOverlap));
    PetscCall(DMDestroy(dmParallel));
    *dmParallel = dmOverlap;
    if (flg) {
      PetscCall(PetscPrintf(comm, "Overlap Migration SF:\n"));
      PetscCall(PetscSFView(sfOverlap, NULL));
    }

    /* Re-map the migration SF to establish the full migration pattern */
    PetscCall(PetscSFGetGraph(sfMigration, &nroots, &noldleaves, &oldLeaves, &oldRemote));
    PetscCall(PetscSFGetGraph(sfOverlap, NULL, &nleaves, NULL, NULL));
    PetscCall(PetscMalloc1(nleaves, &newRemote));
    /* oldRemote: original root point mapping to original leaf point
       newRemote: original leaf point mapping to overlapped leaf point */
    if (oldLeaves) {
      /* After stratification, the migration remotes may not be in root (canonical) order, so we reorder using the leaf numbering */
      PetscCall(PetscMalloc1(noldleaves, &permRemote));
      for (l = 0; l < noldleaves; ++l) permRemote[oldLeaves[l]] = oldRemote[l];
      oldRemote = permRemote;
    }
    PetscCall(PetscSFBcastBegin(sfOverlap, MPIU_2INT, oldRemote, newRemote,MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(sfOverlap, MPIU_2INT, oldRemote, newRemote,MPI_REPLACE));
    if (oldLeaves) PetscCall(PetscFree(oldRemote));
    PetscCall(PetscSFCreate(comm, &sfOverlapPoint));
    PetscCall(PetscSFSetGraph(sfOverlapPoint, nroots, nleaves, NULL, PETSC_OWN_POINTER, newRemote, PETSC_OWN_POINTER));
    PetscCall(PetscSFDestroy(&sfOverlap));
    PetscCall(PetscSFDestroy(&sfMigration));
    sfMigration = sfOverlapPoint;
  }
  /* Cleanup Partition */
  PetscCall(DMLabelDestroy(&lblPartition));
  PetscCall(DMLabelDestroy(&lblMigration));
  PetscCall(PetscSectionDestroy(&cellPartSection));
  PetscCall(ISDestroy(&cellPart));
  /* Copy discretization */
  PetscCall(DMCopyDisc(dm, *dmParallel));
  /* Create sfNatural */
  if (dm->useNatural) {
    PetscSection section;

    PetscCall(DMGetLocalSection(dm, &section));
    PetscCall(DMPlexCreateGlobalToNaturalSF(*dmParallel, section, sfMigration, &(*dmParallel)->sfNatural));
    PetscCall(DMSetUseNatural(*dmParallel, PETSC_TRUE));
  }
  PetscCall(DMPlexCopy_Internal(dm, PETSC_TRUE, *dmParallel));
  /* Cleanup */
  if (sf) {*sf = sfMigration;}
  else    PetscCall(PetscSFDestroy(&sfMigration));
  PetscCall(PetscSFDestroy(&sfPoint));
  PetscCall(PetscLogEventEnd(DMPLEX_Distribute,dm,0,0,0));
  PetscFunctionReturn(0);
}

/*@C
  DMPlexDistributeOverlap - Add partition overlap to a distributed non-overlapping DM.

  Collective on dm

  Input Parameters:
+ dm  - The non-overlapping distributed DMPlex object
- overlap - The overlap of partitions (the same on all ranks)

  Output Parameters:
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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidLogicalCollectiveInt(dm, overlap, 2);
  if (sf) PetscValidPointer(sf, 3);
  PetscValidPointer(dmOverlap, 4);

  if (sf) *sf = NULL;
  *dmOverlap  = NULL;
  PetscCall(PetscObjectGetComm((PetscObject)dm,&comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  if (size == 1) PetscFunctionReturn(0);

  PetscCall(PetscLogEventBegin(DMPLEX_DistributeOverlap, dm, 0, 0, 0));
  /* Compute point overlap with neighbouring processes on the distributed DM */
  PetscCall(PetscLogEventBegin(DMPLEX_Partition,dm,0,0,0));
  PetscCall(PetscSectionCreate(comm, &rootSection));
  PetscCall(PetscSectionCreate(comm, &leafSection));
  PetscCall(DMPlexDistributeOwnership(dm, rootSection, &rootrank, leafSection, &leafrank));
  PetscCall(DMPlexCreateOverlapLabel(dm, overlap, rootSection, rootrank, leafSection, leafrank, &lblOverlap));
  /* Convert overlap label to stratified migration SF */
  PetscCall(DMPlexPartitionLabelCreateSF(dm, lblOverlap, &sfOverlap));
  PetscCall(DMPlexStratifyMigrationSF(dm, sfOverlap, &sfStratified));
  PetscCall(PetscSFDestroy(&sfOverlap));
  sfOverlap = sfStratified;
  PetscCall(PetscObjectSetName((PetscObject) sfOverlap, "Overlap SF"));
  PetscCall(PetscSFSetFromOptions(sfOverlap));

  PetscCall(PetscSectionDestroy(&rootSection));
  PetscCall(PetscSectionDestroy(&leafSection));
  PetscCall(ISDestroy(&rootrank));
  PetscCall(ISDestroy(&leafrank));
  PetscCall(PetscLogEventEnd(DMPLEX_Partition,dm,0,0,0));

  /* Build the overlapping DM */
  PetscCall(DMPlexCreate(comm, dmOverlap));
  PetscCall(PetscObjectSetName((PetscObject) *dmOverlap, "Parallel Mesh"));
  PetscCall(DMPlexMigrate(dm, sfOverlap, *dmOverlap));
  /* Store the overlap in the new DM */
  ((DM_Plex*)(*dmOverlap)->data)->overlap = overlap + ((DM_Plex*)dm->data)->overlap;
  /* Build the new point SF */
  PetscCall(DMPlexCreatePointSF(*dmOverlap, sfOverlap, PETSC_FALSE, &sfPoint));
  PetscCall(DMSetPointSF(*dmOverlap, sfPoint));
  PetscCall(DMGetCoordinateDM(*dmOverlap, &dmCoord));
  if (dmCoord) PetscCall(DMSetPointSF(dmCoord, sfPoint));
  PetscCall(PetscSFDestroy(&sfPoint));
  /* Cleanup overlap partition */
  PetscCall(DMLabelDestroy(&lblOverlap));
  if (sf) *sf = sfOverlap;
  else    PetscCall(PetscSFDestroy(&sfOverlap));
  PetscCall(PetscLogEventEnd(DMPLEX_DistributeOverlap, dm, 0, 0, 0));
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

  Output Parameter:
. overlap - The overlap of this DM

  Level: intermediate

.seealso: DMPlexDistribute(), DMPlexDistributeOverlap(), DMPlexCreateOverlapLabel()
@*/
PetscErrorCode DMPlexGetOverlap(DM dm, PetscInt *overlap)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscUseMethod(dm,"DMPlexGetOverlap_C",(DM,PetscInt*),(dm,overlap));
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexDistributeSetDefault_Plex(DM dm, PetscBool dist)
{
  DM_Plex *mesh = (DM_Plex *) dm->data;

  PetscFunctionBegin;
  mesh->distDefault = dist;
  PetscFunctionReturn(0);
}

/*@
  DMPlexDistributeSetDefault - Set flag indicating whether the DM should be distributed by default

  Logically collective

  Input Parameters:
+ dm   - The DM
- dist - Flag for distribution

  Level: intermediate

.seealso: DMPlexDistributeGetDefault(), DMPlexDistribute()
@*/
PetscErrorCode DMPlexDistributeSetDefault(DM dm, PetscBool dist)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidLogicalCollectiveBool(dm, dist, 2);
  PetscTryMethod(dm,"DMPlexDistributeSetDefault_C",(DM,PetscBool),(dm,dist));
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexDistributeGetDefault_Plex(DM dm, PetscBool *dist)
{
  DM_Plex *mesh = (DM_Plex *) dm->data;

  PetscFunctionBegin;
  *dist = mesh->distDefault;
  PetscFunctionReturn(0);
}

/*@
  DMPlexDistributeGetDefault - Get flag indicating whether the DM should be distributed by default

  Not collective

  Input Parameter:
. dm   - The DM

  Output Parameter:
. dist - Flag for distribution

  Level: intermediate

.seealso: DMPlexDistributeSetDefault(), DMPlexDistribute()
@*/
PetscErrorCode DMPlexDistributeGetDefault(DM dm, PetscBool *dist)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidBoolPointer(dist, 2);
  PetscUseMethod(dm,"DMPlexDistributeGetDefault_C",(DM,PetscBool*),(dm,dist));
  PetscFunctionReturn(0);
}

/*@C
  DMPlexGetGatherDM - Get a copy of the DMPlex that gathers all points on the
  root process of the original's communicator.

  Collective on dm

  Input Parameter:
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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(gatherMesh,3);
  *gatherMesh = NULL;
  if (sf) *sf = NULL;
  comm = PetscObjectComm((PetscObject)dm);
  PetscCallMPI(MPI_Comm_size(comm,&size));
  if (size == 1) PetscFunctionReturn(0);
  PetscCall(DMPlexGetPartitioner(dm,&oldPart));
  PetscCall(PetscObjectReference((PetscObject)oldPart));
  PetscCall(PetscPartitionerCreate(comm,&gatherPart));
  PetscCall(PetscPartitionerSetType(gatherPart,PETSCPARTITIONERGATHER));
  PetscCall(DMPlexSetPartitioner(dm,gatherPart));
  PetscCall(DMPlexDistribute(dm,0,sf,gatherMesh));

  PetscCall(DMPlexSetPartitioner(dm,oldPart));
  PetscCall(PetscPartitionerDestroy(&gatherPart));
  PetscCall(PetscPartitionerDestroy(&oldPart));
  PetscFunctionReturn(0);
}

/*@C
  DMPlexGetRedundantDM - Get a copy of the DMPlex that is completely copied on each process.

  Collective on dm

  Input Parameter:
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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(redundantMesh,3);
  *redundantMesh = NULL;
  comm = PetscObjectComm((PetscObject)dm);
  PetscCallMPI(MPI_Comm_size(comm,&size));
  if (size == 1) {
    PetscCall(PetscObjectReference((PetscObject) dm));
    *redundantMesh = dm;
    if (sf) *sf = NULL;
    PetscFunctionReturn(0);
  }
  PetscCall(DMPlexGetGatherDM(dm,&gatherSF,&gatherDM));
  if (!gatherDM) PetscFunctionReturn(0);
  PetscCallMPI(MPI_Comm_rank(comm,&rank));
  PetscCall(DMPlexGetChart(gatherDM,&pStart,&pEnd));
  numPoints = pEnd - pStart;
  PetscCallMPI(MPI_Bcast(&numPoints,1,MPIU_INT,0,comm));
  PetscCall(PetscMalloc1(numPoints,&points));
  PetscCall(PetscSFCreate(comm,&migrationSF));
  for (p = 0; p < numPoints; p++) {
    points[p].index = p;
    points[p].rank  = 0;
  }
  PetscCall(PetscSFSetGraph(migrationSF,pEnd-pStart,numPoints,NULL,PETSC_OWN_POINTER,points,PETSC_OWN_POINTER));
  PetscCall(DMPlexCreate(comm, redundantMesh));
  PetscCall(PetscObjectSetName((PetscObject) *redundantMesh, "Redundant Mesh"));
  PetscCall(DMPlexMigrate(gatherDM, migrationSF, *redundantMesh));
  PetscCall(DMPlexCreatePointSF(*redundantMesh, migrationSF, PETSC_FALSE, &sfPoint));
  PetscCall(DMSetPointSF(*redundantMesh, sfPoint));
  PetscCall(DMGetCoordinateDM(*redundantMesh, &dmCoord));
  if (dmCoord) PetscCall(DMSetPointSF(dmCoord, sfPoint));
  PetscCall(PetscSFDestroy(&sfPoint));
  if (sf) {
    PetscSF tsf;

    PetscCall(PetscSFCompose(gatherSF,migrationSF,&tsf));
    PetscCall(DMPlexStratifyMigrationSF(dm, tsf, sf));
    PetscCall(PetscSFDestroy(&tsf));
  }
  PetscCall(PetscSFDestroy(&migrationSF));
  PetscCall(PetscSFDestroy(&gatherSF));
  PetscCall(DMDestroy(&gatherDM));
  PetscCall(DMCopyDisc(dm, *redundantMesh));
  PetscCall(DMPlexCopy_Internal(dm, PETSC_TRUE, *redundantMesh));
  PetscFunctionReturn(0);
}

/*@
  DMPlexIsDistributed - Find out whether this DM is distributed, i.e. more than one rank owns some points.

  Collective

  Input Parameter:
. dm      - The DM object

  Output Parameter:
. distributed - Flag whether the DM is distributed

  Level: intermediate

  Notes:
  This currently finds out whether at least two ranks have any DAG points.
  This involves MPI_Allreduce() with one integer.
  The result is currently not stashed so every call to this routine involves this global communication.

.seealso: DMPlexDistribute(), DMPlexGetOverlap(), DMPlexIsInterpolated()
@*/
PetscErrorCode DMPlexIsDistributed(DM dm, PetscBool *distributed)
{
  PetscInt          pStart, pEnd, count;
  MPI_Comm          comm;
  PetscMPIInt       size;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidBoolPointer(distributed,2);
  PetscCall(PetscObjectGetComm((PetscObject)dm,&comm));
  PetscCallMPI(MPI_Comm_size(comm,&size));
  if (size == 1) { *distributed = PETSC_FALSE; PetscFunctionReturn(0); }
  PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
  count = (pEnd - pStart) > 0 ? 1 : 0;
  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &count, 1, MPIU_INT, MPI_SUM, comm));
  *distributed = count > 1 ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(0);
}
