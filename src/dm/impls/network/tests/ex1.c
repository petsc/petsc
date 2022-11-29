static char help[] = "Check if DMClone for DMNetwork Correctly Shallow Clones Topology Only \n\n";

#include <petscdmnetwork.h>

/*
CreateStarGraphEdgeList - Create a k-Star Graph Edgelist on current processor
  Not Collective

  Input Parameters:
. k    - order of the star graph (number of edges)
. directin - if true direction of edges is towards the center vertex, otherwise they are directed out of the center vertex.

  Output Parameters:
.  ne - number of edges of this star graph
.  edgelist - list of edges for this star graph, this is a one dimensional array with pairs of entries being the two vertices (in global numbering of the vertices) of each edge,
              [first vertex of first edge, second vertex of first edge, first vertex of second edge, second vertex of second edge, etc].

              User is responsible for deallocating this memory.
*/
PetscErrorCode CreateStarGraphEdgeList(PetscInt k, PetscBool directin, PetscInt *ne, PetscInt *edgelist[])
{
  PetscInt i;

  PetscFunctionBegin;
  *ne = k;
  PetscCall(PetscCalloc1(2 * k, edgelist));

  if (directin) {
    for (i = 0; i < k; i++) {
      (*edgelist)[2 * i]     = i + 1;
      (*edgelist)[2 * i + 1] = 0;
    }
  } else {
    for (i = 0; i < k; i++) {
      (*edgelist)[2 * i]     = 0;
      (*edgelist)[2 * i + 1] = i + 1;
    }
  }
  PetscFunctionReturn(0);
}

/*
CreateSimpleStarGraph - Create a Distributed k-Star Graph DMNetwork with a single PetscInt component on
all edges and vertices, aselectable number of dofs on vertices and edges. Intended mostly to be used for testing purposes.

  Input Parameters:
. comm       - the communicator of the dm
. numdofvert - number of degrees of freedom (dofs) on vertices
. numdofedge - number of degrees of freedom (dofs) on edges
. k          - order of the star graph (number of edges)
. directin   - if true direction of edges is towards the center vertex, otherwise they are directed out of the center vertex

  Output Parameters:
. newdm       - The created and distributed simple Star Graph
*/
PetscErrorCode CreateSimpleStarGraph(MPI_Comm comm, PetscInt numdofvert, PetscInt numdofedge, PetscInt k, PetscBool directin, DM *newdm)
{
  DM          dm;
  PetscMPIInt rank;
  PetscInt    ne       = 0, compkey, eStart, eEnd, vStart, vEnd, e, v;
  PetscInt   *edgelist = NULL, *compedge, *compvert;

  PetscFunctionBegin;
  PetscCall(DMNetworkCreate(comm, &dm));
  PetscCall(DMNetworkSetNumSubNetworks(dm, PETSC_DECIDE, 1));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  if (rank == 0) PetscCall(CreateStarGraphEdgeList(k, directin, &ne, &edgelist));
  PetscCall(DMNetworkAddSubnetwork(dm, "Main", ne, edgelist, NULL));
  PetscCall(DMNetworkRegisterComponent(dm, "dummy", sizeof(PetscInt), &compkey));
  PetscCall(DMNetworkLayoutSetUp(dm));
  PetscCall(PetscFree(edgelist));
  PetscCall(DMNetworkGetEdgeRange(dm, &eStart, &eEnd));
  PetscCall(DMNetworkGetVertexRange(dm, &vStart, &vEnd));
  PetscCall(PetscMalloc2(eEnd - eStart, &compedge, vEnd - vStart, &compvert));
  for (e = eStart; e < eEnd; e++) {
    compedge[e - eStart] = e;
    PetscCall(DMNetworkAddComponent(dm, e, compkey, &compedge[e - eStart], numdofedge));
  }
  for (v = vStart; v < vEnd; v++) {
    compvert[v - vStart] = v;
    PetscCall(DMNetworkAddComponent(dm, v, compkey, &compvert[v - vStart], numdofvert));
  }
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));
  PetscCall(DMSetUp(dm));
  PetscCall(PetscFree2(compedge, compvert));
  PetscCall(DMNetworkDistribute(&dm, 0));
  *newdm = dm;
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM           dm, dmclone, plex;
  PetscInt     e, eStart, eEnd, ndofs, ndofsprev;
  PetscInt    *compprev, *comp, compkey;
  PetscInt     dofv = 1, dofe = 1, ne = 1;
  PetscSection sec;
  Vec          vec;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  /* create a distributed k-Star graph DMNetwork */
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-dofv", &dofv, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-dofe", &dofe, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-ne", &ne, NULL));
  PetscCall(CreateSimpleStarGraph(PETSC_COMM_WORLD, dofv, dofe, ne, PETSC_TRUE, &dm));
  PetscCall(DMNetworkGetEdgeRange(dm, &eStart, &eEnd));

  /* check if cloning changed any componenent */
  if (eStart < eEnd) PetscCall(DMNetworkGetComponent(dm, eStart, 0, NULL, (void **)&compprev, &ndofsprev));
  PetscCall(DMClone(dm, &dmclone));
  if (eStart < eEnd) {
    PetscCall(DMNetworkGetComponent(dm, eStart, 0, NULL, (void **)&comp, &ndofs));
    PetscCheck(*comp == *compprev, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Cloning changed the Original, comp (previous) : %" PetscInt_FMT " comp (now) : %" PetscInt_FMT, *compprev, *comp);
    PetscCheck(ndofsprev == ndofs, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Cloning changed the Original, ndofs (previous) : %" PetscInt_FMT " ndofs (now) : %" PetscInt_FMT, ndofsprev, ndofs);
  }

  /* register new components to the clone and add a dummy component to every point */
  PetscCall(DMNetworkRegisterComponent(dmclone, "dummyclone", sizeof(PetscInt), &compkey));
  PetscCall(DMNetworkGetEdgeRange(dmclone, &eStart, &eEnd));
  PetscCall(PetscMalloc1(eEnd - eStart, &comp));
  for (e = eStart; e < eEnd; e++) {
    comp[e - eStart] = e;
    PetscCall(DMNetworkAddComponent(dmclone, e, compkey, &comp[e - eStart], 2));
  }
  PetscCall(DMNetworkFinalizeComponents(dmclone));
  PetscCall(PetscFree(comp));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, " dm: \n"));
  PetscCall(DMView(dm, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(DMNetworkGetPlex(dm, &plex));
  PetscCall(DMGetLocalSection(plex, &sec));
  PetscCall(PetscSectionView(sec, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n dmclone: \n"));
  PetscCall(DMView(dmclone, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(DMNetworkGetPlex(dmclone, &plex));
  PetscCall(DMGetLocalSection(plex, &sec));
  PetscCall(PetscSectionView(sec, PETSC_VIEWER_STDOUT_WORLD));

  /* create Vectors */
  PetscCall(DMCreateGlobalVector(dm, &vec));
  PetscCall(VecSet(vec, 1.0));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n dm vec:\n"));
  PetscCall(VecView(vec, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(VecDestroy(&vec));

  PetscCall(DMCreateGlobalVector(dmclone, &vec));
  PetscCall(VecSet(vec, 2.0));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n dmclone vec:\n"));
  PetscCall(VecView(vec, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(VecDestroy(&vec));

  PetscCall(DMDestroy(&dm));
  PetscCall(DMDestroy(&dmclone));
  PetscCall(PetscFinalize());
}

/*TEST

  test:
    suffix: 0
    args:

  test:
    suffix: 1
    nsize: 2
    args: -dofv 2 -dofe 2 -ne 2

 TEST*/
