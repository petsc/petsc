static char help[] = "Test DMCreateCoordinateDM_Network, and related functions \n\n";

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
  PetscCall(DMSetUp(dm));
  PetscCall(PetscFree2(compedge, compvert));
  *newdm = dm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Simple Circular embedding of the star graph */
PetscErrorCode StarGraphSetCoordinates(DM dm)
{
  DM           cdm;
  Vec          Coord;
  PetscScalar *coord;
  PetscInt     vStart, vEnd, v, vglobal, compkey, off, NVert;
  PetscReal    theta;

  PetscFunctionBegin;
  PetscCall(DMGetCoordinateDM(dm, &cdm));
  PetscCall(DMSetCoordinateDim(dm, 2));
  PetscCall(DMNetworkGetVertexRange(cdm, &vStart, &vEnd));
  PetscCall(DMNetworkRegisterComponent(cdm, "coordinates", 0, &compkey));
  for (v = vStart; v < vEnd; v++) PetscCall(DMNetworkAddComponent(cdm, v, compkey, NULL, 2));
  PetscCall(DMNetworkFinalizeComponents(cdm));

  PetscCall(DMCreateLocalVector(cdm, &Coord));
  PetscCall(VecGetArray(Coord, &coord));
  PetscCall(DMNetworkGetNumVertices(cdm, NULL, &NVert));
  theta = 2 * PETSC_PI / (NVert - 1);
  for (v = vStart; v < vEnd; v++) {
    PetscCall(DMNetworkGetGlobalVertexIndex(cdm, v, &vglobal));
    PetscCall(DMNetworkGetLocalVecOffset(cdm, v, 0, &off));
    if (vglobal == 0) {
      coord[off]     = 0.0;
      coord[off + 1] = 0.0;
    } else {
      /* embed on the unit circle */
      coord[off]     = PetscCosReal(theta * (vglobal - 1));
      coord[off + 1] = PetscSinReal(theta * (vglobal - 1));
    }
  }
  PetscCall(VecRestoreArray(Coord, &coord));
  PetscCall(DMSetCoordinatesLocal(dm, Coord));
  PetscCall(VecDestroy(&Coord));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* This subroutine is used in petsc/src/snes/tutorials/network/ex1.c */
static PetscErrorCode CoordinatePrint(DM dm)
{
  DM                 dmclone;
  PetscInt           cdim, v, off, vglobal, vStart, vEnd;
  const PetscScalar *carray;
  Vec                coords;
  MPI_Comm           comm;
  PetscMPIInt        rank;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));

  PetscCall(DMGetCoordinateDM(dm, &dmclone));
  PetscCall(DMNetworkGetVertexRange(dm, &vStart, &vEnd));
  PetscCall(DMGetCoordinatesLocal(dm, &coords));

  PetscCall(DMGetCoordinateDim(dm, &cdim));
  PetscCall(VecGetArrayRead(coords, &carray));

  PetscCall(PetscPrintf(MPI_COMM_WORLD, "\nCoordinatePrint, cdim %" PetscInt_FMT ":\n", cdim));
  PetscCall(PetscSynchronizedPrintf(MPI_COMM_WORLD, "[%i]\n", rank));
  for (v = vStart; v < vEnd; v++) {
    PetscCall(DMNetworkGetLocalVecOffset(dmclone, v, 0, &off));
    PetscCall(DMNetworkGetGlobalVertexIndex(dmclone, v, &vglobal));
    switch (cdim) {
    case 2:
      PetscCall(PetscSynchronizedPrintf(MPI_COMM_WORLD, "Vertex: %" PetscInt_FMT ", x =  %f y = %f \n", vglobal, (double)PetscRealPart(carray[off]), (double)PetscRealPart(carray[off + 1])));
      break;
    default:
      PetscCheck(cdim == 2, MPI_COMM_WORLD, PETSC_ERR_SUP, "Only supports Network embedding dimension of 2, not supplied  %" PetscInt_FMT, cdim);
      break;
    }
  }
  PetscCall(PetscSynchronizedFlush(MPI_COMM_WORLD, NULL));
  PetscCall(VecRestoreArrayRead(coords, &carray));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM          dm;
  PetscInt    dofv = 1, dofe = 1, ne = 1;
  PetscMPIInt rank;
  PetscBool   testdistribute = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

  /* create a distributed k-Star graph DMNetwork */
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-dofv", &dofv, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-dofe", &dofe, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-ne", &ne, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-testdistribute", &testdistribute, NULL));
  PetscCall(StarGraphCreate(PETSC_COMM_WORLD, dofv, dofe, ne, PETSC_TRUE, &dm));

  /* setup a quick R^2 embedding of the star graph */
  PetscCall(StarGraphSetCoordinates(dm));

  if (testdistribute) {
    PetscCall(DMNetworkDistribute(&dm, 0));
    PetscCall(DMView(dm, PETSC_VIEWER_STDOUT_WORLD));
  }

  /* print or view the coordinates of each vertex */
  PetscCall(CoordinatePrint(dm));

  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
}

/*TEST

  test:
    suffix: 0
    args: -ne 4 -testdistribute

  test:
    suffix: 1
    nsize: 2
    args: -ne 4 -testdistribute -petscpartitioner_type simple

 TEST*/
