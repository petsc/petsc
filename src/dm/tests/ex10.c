/*
    Simple example demonstrating creating a one sub-network DMNetwork in parallel.

    In this example vertices 0 and 1 are not connected to any edges.
*/

#include <petscdmnetwork.h>

int main(int argc, char **argv)
{
  DM              network;
  PetscMPIInt     size, rank;
  MPI_Comm        comm;
  PetscInt        e, ne, nv, v, ecompkey, vcompkey;
  PetscInt       *edgelist = NULL;
  const PetscInt *nodes, *edges;
  DM              plex;
  PetscSection    section;
  PetscInt        Ne, Ni;
  PetscInt        nodeOffset, k = 2, nedge;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
  PetscCall(PetscOptionsSetValue(NULL, "-petscpartitioner_use_vertex_weights", "No"));
  comm = PETSC_COMM_WORLD;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCallMPI(MPI_Comm_size(comm, &size));

  PetscCall(DMNetworkCreate(PETSC_COMM_WORLD, &network));

  /* Register zero size components to get compkeys to be used by DMNetworkAddComponent() */
  PetscCall(DMNetworkRegisterComponent(network, "ecomp", 0, &ecompkey));
  PetscCall(DMNetworkRegisterComponent(network, "vcomp", 0, &vcompkey));

  Ne         = 2;
  Ni         = 1;
  nodeOffset = (Ne + Ni) * rank; /* The global node index of the first node defined on this process */

  /* There are three nodes on each rank and two edges. The edges only connect nodes on the given rank */
  nedge = k * Ni;

  if (rank == 0) {
    nedge = 1;
    PetscCall(PetscCalloc1(2 * nedge, &edgelist));
    edgelist[0] = nodeOffset + 2;
    edgelist[1] = nodeOffset + 3;
  } else {
    nedge = 2;
    PetscCall(PetscCalloc1(2 * nedge, &edgelist));
    edgelist[0] = nodeOffset + 0;
    edgelist[1] = nodeOffset + 2;
    edgelist[2] = nodeOffset + 1;
    edgelist[3] = nodeOffset + 2;
  }

  PetscCall(DMNetworkSetNumSubNetworks(network, PETSC_DECIDE, 1));
  PetscCall(DMNetworkAddSubnetwork(network, "Subnetwork 1", nedge, edgelist, NULL));
  PetscCall(DMNetworkLayoutSetUp(network));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Network after DMNetworkLayoutSetUp:\n"));
  PetscCall(DMView(network, PETSC_VIEWER_STDOUT_WORLD));

  /* Add components and variables for the network */
  PetscCall(DMNetworkGetSubnetwork(network, 0, &nv, &ne, &nodes, &edges));
  for (e = 0; e < ne; e++) {
    /* The edges have no degrees of freedom */
    PetscCall(DMNetworkAddComponent(network, edges[e], ecompkey, NULL, 1));
  }
  for (v = 0; v < nv; v++) PetscCall(DMNetworkAddComponent(network, nodes[v], vcompkey, NULL, 2));

  PetscCall(DMSetUp(network));
  PetscCall(DMNetworkGetPlex(network, &plex));
  /* PetscCall(DMView(plex,PETSC_VIEWER_STDOUT_WORLD)); */
  PetscCall(DMGetLocalSection(plex, &section));
  PetscCall(PetscSectionView(section, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(PetscFree(edgelist));

  PetscCall(DMNetworkDistribute(&network, 0));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nNetwork after DMNetworkDistribute:\n"));
  PetscCall(DMView(network, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(DMNetworkGetPlex(network, &plex));
  /* PetscCall(DMView(plex,PETSC_VIEWER_STDOUT_WORLD)); */
  PetscCall(DMGetLocalSection(plex, &section));
  PetscCall(PetscSectionView(section, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(DMDestroy(&network));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
      requires: !complex double

   test:
      nsize: 2
      args: -petscpartitioner_type simple

TEST*/
