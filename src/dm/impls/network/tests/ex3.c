static char help[] = "Test query functions for DMNetwork \n\n";

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
PetscErrorCode StarGraphCreateEdgeList(PetscInt k, PetscBool directin, PetscInt *ne, PetscInt *edgelist[])
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
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
CreateSimpleStarGraph - Create a Distributed k-Star Graph DMNetwork with a single PetscInt component on
all edges and vertices, a selectable number of dofs on vertices and edges. Intended mostly to be used for testing purposes.

  Input Parameters:
. comm       - the communicator of the dm
. numdofvert - number of degrees of freedom (dofs) on vertices
. numdofedge - number of degrees of freedom (dofs) on edges
. k          - order of the star graph (number of edges)
. directin   - if true direction of edges is towards the center vertex, otherwise they are directed out of the center vertex

  Output Parameter:
. newdm       - The created and distributed simple Star Graph
*/
PetscErrorCode StarGraphCreate(MPI_Comm comm, PetscInt numdofvert, PetscInt numdofedge, PetscInt k, PetscBool directin, DM *newdm)
{
  DM          dm;
  PetscMPIInt rank;
  PetscInt    ne       = 0, compkey, eStart, eEnd, vStart, vEnd, e, v;
  PetscInt   *edgelist = NULL, *compedge, *compvert;

  PetscFunctionBegin;
  PetscCall(DMNetworkCreate(comm, &dm));
  PetscCall(DMNetworkSetNumSubNetworks(dm, PETSC_DECIDE, 1));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  if (rank == 0) PetscCall(StarGraphCreateEdgeList(k, directin, &ne, &edgelist));
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
  *newdm = dm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode StarGraphTestQuery(DM dm, PetscInt ne)
{
  PetscInt globalnumvert, localnumvert, globalnumedge, localnumedge;

  PetscFunctionBegin;
  PetscCall(DMNetworkGetNumEdges(dm, &localnumedge, &globalnumedge));
  PetscCall(DMNetworkGetNumVertices(dm, &localnumvert, &globalnumvert));

  PetscCheck(globalnumedge == ne, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Global number of edges should be %" PetscInt_FMT "instead was %" PetscInt_FMT, ne, globalnumedge);
  PetscCheck(globalnumvert == ne + 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Global number of vertices should be %" PetscInt_FMT "instead was %" PetscInt_FMT, ne + 1, globalnumvert);
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM          dm;
  PetscInt    ne = 1;
  PetscMPIInt rank;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  /* create a distributed k-Star graph DMNetwork */
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-ne", &ne, NULL));
  PetscCall(StarGraphCreate(PETSC_COMM_WORLD, 1, 0, ne, PETSC_TRUE, &dm));
  PetscCall(DMNetworkDistribute(&dm, 0));
  /* Test if query functions for DMNetwork run successfully */
  PetscCall(StarGraphTestQuery(dm, ne));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
}

/*TEST
  test:
    suffix: 0
    args:  -ne 5
  test:
    suffix: 1
    nsize: 2
    args:  -ne 5 -petscpartitioner_type simple
 TEST*/
