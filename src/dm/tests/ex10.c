/*
    Simple example demonstrating creating a one sub-network DMNetwork in parallel.

    In this example vertices 0 and 1 are not connected to any edges.
*/

#include <petscdmnetwork.h>

int main(int argc,char ** argv)
{
  PetscErrorCode    ierr;
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

  ierr = PetscInitialize(&argc,&argv,NULL,NULL);if (ierr) return ierr;
  ierr = PetscOptionsSetValue(NULL,"-petscpartitioner_use_vertex_weights","No");CHKERRQ(ierr);
  comm = PETSC_COMM_WORLD;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);

  ierr = DMNetworkCreate(PETSC_COMM_WORLD,&network);CHKERRQ(ierr);

  /* Register zero size componets to get compkeys to be used by DMNetworkAddComponent() */
  ierr = DMNetworkRegisterComponent(network,"ecomp",0,&ecompkey);CHKERRQ(ierr);
  ierr = DMNetworkRegisterComponent(network,"vcomp",0,&vcompkey);CHKERRQ(ierr);

  Ne = 2;
  Ni = 1;
  nodeOffset = (Ne+Ni)*rank;   /* The global node index of the first node defined on this process */

  /* There are three nodes on each rank and two edges. The edges only connect nodes on the given rank */
  nedge = k * Ni;

  if (rank == 0) {
    nedge = 1;
    ierr = PetscCalloc1(2*nedge,&edgelist);CHKERRQ(ierr);
    edgelist[0] = nodeOffset + 2;
    edgelist[1] = nodeOffset + 3;
  } else {
    nedge = 2;
    ierr = PetscCalloc1(2*nedge,&edgelist);CHKERRQ(ierr);
    edgelist[0] = nodeOffset + 0;
    edgelist[1] = nodeOffset + 2;
    edgelist[2] = nodeOffset + 1;
    edgelist[3] = nodeOffset + 2;
  }

  ierr = DMNetworkSetNumSubNetworks(network,PETSC_DECIDE,1);CHKERRQ(ierr);
  ierr = DMNetworkAddSubnetwork(network,"Subnetwork 1",nedge,edgelist,NULL);CHKERRQ(ierr);
  ierr = DMNetworkLayoutSetUp(network);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"Network after DMNetworkLayoutSetUp:\n");CHKERRQ(ierr);
  ierr = DMView(network,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* Add components and variables for the network */
  ierr = DMNetworkGetSubnetwork(network,0,&nv,&ne,&nodes,&edges);CHKERRQ(ierr);
  for (e = 0; e < ne; e++) {
    /* The edges have no degrees of freedom */
    ierr = DMNetworkAddComponent(network,edges[e],ecompkey,NULL,1);CHKERRQ(ierr);
  }
  for (v = 0; v < nv; v++) {
    ierr = DMNetworkAddComponent(network,nodes[v],vcompkey,NULL,2);CHKERRQ(ierr);
  }

  ierr = DMSetUp(network);CHKERRQ(ierr);
  ierr = DMNetworkGetPlex(network,&plex);CHKERRQ(ierr);
  /* ierr = DMView(plex,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); */
  ierr = DMGetLocalSection(plex,&section);CHKERRQ(ierr);
  ierr = PetscSectionView(section,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = PetscFree(edgelist);CHKERRQ(ierr);

  ierr = DMNetworkDistribute(&network,0);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nNetwork after DMNetworkDistribute:\n");CHKERRQ(ierr);
  ierr = DMView(network,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = DMNetworkGetPlex(network,&plex);CHKERRQ(ierr);
  /* ierr = DMView(plex,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); */
  ierr = DMGetLocalSection(plex,&section);CHKERRQ(ierr);
  ierr = PetscSectionView(section,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = DMDestroy(&network);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
      requires: !complex double

   test:
      nsize: 2
      args: -petscpartitioner_type simple

TEST*/
