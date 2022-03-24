/*
    Simple example demonstrating creating a one sub-network DMNetwork in parallel.

    In this example vertices 0 and 1 are not connected to any edges.
*/

#include <petscdmnetwork.h>

int main(int argc,char ** argv)
{
  DM                network;
  PetscMPIInt       size,rank;
  MPI_Comm          comm;
  PetscInt          e,ne,nv,v,ecompkey,vcompkey;
  PetscInt          *edgelist = NULL;
  const PetscInt    *nodes,*edges;
  DM                plex;
  PetscSection      section;
  PetscInt          Ne,Ni;
  PetscInt          nodeOffset,k = 2,nedge;

  CHKERRQ(PetscInitialize(&argc,&argv,NULL,NULL));
  CHKERRQ(PetscOptionsSetValue(NULL,"-petscpartitioner_use_vertex_weights","No"));
  comm = PETSC_COMM_WORLD;
  CHKERRMPI(MPI_Comm_rank(comm,&rank));
  CHKERRMPI(MPI_Comm_size(comm,&size));

  CHKERRQ(DMNetworkCreate(PETSC_COMM_WORLD,&network));

  /* Register zero size componets to get compkeys to be used by DMNetworkAddComponent() */
  CHKERRQ(DMNetworkRegisterComponent(network,"ecomp",0,&ecompkey));
  CHKERRQ(DMNetworkRegisterComponent(network,"vcomp",0,&vcompkey));

  Ne = 2;
  Ni = 1;
  nodeOffset = (Ne+Ni)*rank;   /* The global node index of the first node defined on this process */

  /* There are three nodes on each rank and two edges. The edges only connect nodes on the given rank */
  nedge = k * Ni;

  if (rank == 0) {
    nedge = 1;
    CHKERRQ(PetscCalloc1(2*nedge,&edgelist));
    edgelist[0] = nodeOffset + 2;
    edgelist[1] = nodeOffset + 3;
  } else {
    nedge = 2;
    CHKERRQ(PetscCalloc1(2*nedge,&edgelist));
    edgelist[0] = nodeOffset + 0;
    edgelist[1] = nodeOffset + 2;
    edgelist[2] = nodeOffset + 1;
    edgelist[3] = nodeOffset + 2;
  }

  CHKERRQ(DMNetworkSetNumSubNetworks(network,PETSC_DECIDE,1));
  CHKERRQ(DMNetworkAddSubnetwork(network,"Subnetwork 1",nedge,edgelist,NULL));
  CHKERRQ(DMNetworkLayoutSetUp(network));

  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Network after DMNetworkLayoutSetUp:\n"));
  CHKERRQ(DMView(network,PETSC_VIEWER_STDOUT_WORLD));

  /* Add components and variables for the network */
  CHKERRQ(DMNetworkGetSubnetwork(network,0,&nv,&ne,&nodes,&edges));
  for (e = 0; e < ne; e++) {
    /* The edges have no degrees of freedom */
    CHKERRQ(DMNetworkAddComponent(network,edges[e],ecompkey,NULL,1));
  }
  for (v = 0; v < nv; v++) {
    CHKERRQ(DMNetworkAddComponent(network,nodes[v],vcompkey,NULL,2));
  }

  CHKERRQ(DMSetUp(network));
  CHKERRQ(DMNetworkGetPlex(network,&plex));
  /* CHKERRQ(DMView(plex,PETSC_VIEWER_STDOUT_WORLD)); */
  CHKERRQ(DMGetLocalSection(plex,&section));
  CHKERRQ(PetscSectionView(section,PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(PetscFree(edgelist));

  CHKERRQ(DMNetworkDistribute(&network,0));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nNetwork after DMNetworkDistribute:\n"));
  CHKERRQ(DMView(network,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(DMNetworkGetPlex(network,&plex));
  /* CHKERRQ(DMView(plex,PETSC_VIEWER_STDOUT_WORLD)); */
  CHKERRQ(DMGetLocalSection(plex,&section));
  CHKERRQ(PetscSectionView(section,PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(DMDestroy(&network));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   build:
      requires: !complex double

   test:
      nsize: 2
      args: -petscpartitioner_type simple

TEST*/
