/*
  PT-SCOTCH graph partitioning shared by the MATPARTITIONINGPTSCOTCH and PETSCPARTITIONERPTSCOTCH
  implementations.
*/

#include <petsc/private/matimpl.h>

EXTERN_C_BEGIN
#include <ptscotch.h>
EXTERN_C_END

/*
  PetscPTSCOTCHPartitionGraph_Private - Partitions a graph distributed over comm with PT-SCOTCH

  Collective

  Input Parameters:
+ comm       - the communicator the graph is distributed over, which PT-SCOTCH caches attributes on
. ranges     - the first vertex owned by each process, of length one more than the size of comm, the
               same on every process
. xadj       - adjacency offsets of the local vertices, of length one more than the local vertices
. adjncy     - adjacency of the local vertices, in global vertex numbers
. vwgt       - weight of each local vertex, or NULL for unit weights
. adjwgt     - weight of each local edge, or NULL when this process supplies none
. useadjwgt  - whether edge weights are in use, which must be the same on every process, and may be
               true where adjwgt is NULL because a process owning no edges allocates nothing
. nparts     - number of parts to partition into
. tpwgts     - target weight of each part, of length nparts, or NULL for equally weighted parts
. strategy   - PT-SCOTCH strategy flags
- imbalance  - allowed load imbalance ratio

  Output Parameter:
. assignment - part each local vertex is assigned to, of length the local vertices

  Notes:
  PT-SCOTCH requires every process of the communicator it is given to own at least one vertex, and
  SCOTCH_dgraphBuild() requires those processes to agree on whether each optional weight array is
  null. Both requirements are met by partitioning on the processes that own vertices; a process
  owning none has nothing to assign and returns without calling PT-SCOTCH. The graph itself is
  unchanged, so the partition does not depend on how many processes own no vertices.

  A process that owns vertices can still own no edges, and then its zero-size edge weight allocation
  returns null through no fault of the caller, so useadjwgt reports globally what adjwgt cannot, and
  such a process is given a stand-in array. PT-SCOTCH reads one entry per local edge and that process
  has none, so a single element it never looks at is enough. Uneven vertex weights are left to
  PT-SCOTCH to report, since every process here owns vertices and so has nothing to excuse a null
  array.

  The weight arrays are handed to PT-SCOTCH without a cast, so that the compiler rejects the call if
  SCOTCH_Num ever stops matching PetscInt, which PETSc's configure arranges for a downloaded
  PT-SCOTCH and assumes for a prebuilt one.

.seealso: `PetscCommCreateNonempty()`, `MATPARTITIONINGPTSCOTCH`, `PETSCPARTITIONERPTSCOTCH`
*/
PetscErrorCode PetscPTSCOTCHPartitionGraph_Private(MPI_Comm comm, const PetscInt ranges[], PetscInt xadj[], PetscInt adjncy[], PetscInt vwgt[], PetscInt adjwgt[], PetscBool useadjwgt, PetscInt nparts, PetscInt tpwgts[], PetscInt strategy, double imbalance, PetscInt assignment[])
{
  MPI_Comm     gcomm;
  SCOTCH_Arch  archdat;
  SCOTCH_Strat stradat;
  SCOTCH_Num   vertnbr, edgenbr, archnbr;
  PetscInt    *velotab = vwgt, *edlotab = adjwgt, *edlotabempty = NULL;
  double       kbalval = imbalance;
  PetscMPIInt  rank, size;

  PetscFunctionBegin;
  PetscCall(PetscCommCreateNonempty(comm, ranges, &gcomm));
  if (gcomm == MPI_COMM_NULL) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCallMPI(MPI_Comm_size(gcomm, &size));
  vertnbr = ranges[rank + 1] - ranges[rank];
  edgenbr = xadj[vertnbr];

  PetscCheck(!useadjwgt || adjwgt || edgenbr == 0, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Edge weights are in use but none were supplied for the %" PetscInt_FMT " edges of this process", (PetscInt)edgenbr);
  if (useadjwgt && !adjwgt) {
    PetscCall(PetscCalloc1(1, &edlotabempty));
    edlotab = edlotabempty;
  }

  PetscCallExternal(SCOTCH_stratInit, &stradat);
  PetscCallExternal(SCOTCH_archInit, &archdat);
  if (size == 1) {
    SCOTCH_Graph grafdat;

    /* The whole graph is on this process, so partition it sequentially. Asking for more parts than
       there are vertices leaves the surplus parts empty, and PT-SCOTCH divides by a per-part
       average load that truncates to zero once the graph is smaller than the number of parts. */
    archnbr = PetscMin(nparts, vertnbr);
    PetscCallExternal(SCOTCH_stratGraphMapBuild, &stradat, strategy, nparts, kbalval);
    PetscCallExternal(SCOTCH_graphInit, &grafdat);
    PetscCallExternal(SCOTCH_graphBuild, &grafdat, 0, vertnbr, xadj, xadj + 1, velotab, NULL, edgenbr, adjncy, edlotab);
    if (PetscDefined(USE_DEBUG)) PetscCallExternal(SCOTCH_graphCheck, &grafdat);
    if (tpwgts) PetscCallExternal(SCOTCH_archCmpltw, &archdat, archnbr, tpwgts);
    else PetscCallExternal(SCOTCH_archCmplt, &archdat, archnbr);
    PetscCallExternal(SCOTCH_graphMap, &grafdat, &archdat, &stradat, assignment);
    PetscCallExternalVoid("SCOTCH_graphExit", SCOTCH_graphExit(&grafdat));
  } else {
    SCOTCH_Dgraph   grafdat;
    SCOTCH_Dmapping mappdat;

    archnbr = nparts;
    PetscCallExternal(SCOTCH_stratDgraphMapBuild, &stradat, strategy, size, nparts, kbalval);
    PetscCallExternal(SCOTCH_dgraphInit, &grafdat, gcomm);
    PetscCallExternal(SCOTCH_dgraphBuild, &grafdat, 0, vertnbr, vertnbr, xadj, xadj + 1, velotab, NULL, edgenbr, edgenbr, adjncy, NULL, edlotab);
    if (PetscDefined(USE_DEBUG)) PetscCallExternal(SCOTCH_dgraphCheck, &grafdat);
    if (tpwgts) PetscCallExternal(SCOTCH_archCmpltw, &archdat, archnbr, tpwgts);
    else PetscCallExternal(SCOTCH_archCmplt, &archdat, archnbr);
    PetscCallExternal(SCOTCH_dgraphMapInit, &grafdat, &mappdat, &archdat, assignment);
    PetscCallExternal(SCOTCH_dgraphMapCompute, &grafdat, &mappdat, &stradat);
    PetscCallExternalVoid("SCOTCH_dgraphMapExit", SCOTCH_dgraphMapExit(&grafdat, &mappdat));
    PetscCallExternalVoid("SCOTCH_dgraphExit", SCOTCH_dgraphExit(&grafdat));
  }
  PetscCallExternalVoid("SCOTCH_archExit", SCOTCH_archExit(&archdat));
  PetscCallExternalVoid("SCOTCH_stratExit", SCOTCH_stratExit(&stradat));
  PetscCall(PetscFree(edlotabempty));
  PetscCall(PetscCommDestroyNonempty(comm, &gcomm));
  PetscFunctionReturn(PETSC_SUCCESS);
}
