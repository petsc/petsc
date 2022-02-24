static char help[] = "This example tests subnetwork coupling with zero size components. \n\n";

#include <petscdmnetwork.h>

int main(int argc,char ** argv)
{
  PetscErrorCode ierr;
  PetscMPIInt    size,rank;
  DM             dmnetwork;
  PetscInt       i,j,net,Nsubnet,ne,nv,nvar,v,goffset,row,compkey0,compkey1,compkey;
  PetscInt       *numEdges,**edgelist,asvtx[2],bsvtx[2];
  const PetscInt *vtx,*edges;
  PetscBool      ghost,distribute=PETSC_TRUE,sharedv;
  Vec            X;
  PetscScalar    val;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));

  /* Create a network of subnetworks */
  if (size == 1) Nsubnet = 2;
  else Nsubnet = (PetscInt)size;
  CHKERRQ(PetscCalloc2(Nsubnet,&numEdges,Nsubnet,&edgelist));

  /* when size>1, process[i] creates subnetwork[i] */
  for (i=0; i<Nsubnet; i++) {
    if (i == 0 && (size == 1 || (rank == i && size >1))) {
      numEdges[i] = 3;
      CHKERRQ(PetscMalloc1(2*numEdges[i],&edgelist[i]));
      edgelist[i][0] = 0; edgelist[i][1] = 1;
      edgelist[i][2] = 1; edgelist[i][3] = 2;
      edgelist[i][4] = 2; edgelist[i][5] = 3;

    } else if (i == 1 && (size == 1 || (rank == i && size >1))) {
      numEdges[i] = 3;
      CHKERRQ(PetscMalloc1(2*numEdges[i],&edgelist[i]));
      edgelist[i][0] = 0; edgelist[i][1] = 1;
      edgelist[i][2] = 1; edgelist[i][3] = 2;
      edgelist[i][4] = 2; edgelist[i][5] = 3;

    } else if (i>1 && (size == 1 || (rank == i && size >1))) {
      numEdges[i] = 3;
      CHKERRQ(PetscMalloc1(2*numEdges[i],&edgelist[i]));
      for (j=0; j< numEdges[i]; j++) {
        edgelist[i][2*j] = j; edgelist[i][2*j+1] = j+1;
      }
    }
  }

  /* Create a dmnetwork */
  CHKERRQ(DMNetworkCreate(PETSC_COMM_WORLD,&dmnetwork));

  /* Register zero size componets to get compkeys to be used by DMNetworkAddComponent() */
  CHKERRQ(DMNetworkRegisterComponent(dmnetwork,"comp0",0,&compkey0));
  CHKERRQ(DMNetworkRegisterComponent(dmnetwork,"comp1",0,&compkey1));

  /* Set number of subnetworks, numbers of vertices and edges over each subnetwork */
  CHKERRQ(DMNetworkSetNumSubNetworks(dmnetwork,PETSC_DECIDE,Nsubnet));

  for (i=0; i<Nsubnet; i++) {
    PetscInt netNum = -1;
    CHKERRQ(DMNetworkAddSubnetwork(dmnetwork,NULL,numEdges[i],edgelist[i],&netNum));
  }

  /* Add shared vertices -- all processes hold this info at current implementation
       net[0].0 -> net[j].0, j=0,...,Nsubnet-1
       net[0].1 -> net[j].1, j=0,...,Nsubnet-1 */
  asvtx[0] = bsvtx[0] = 0;
  asvtx[1] = bsvtx[1] = 1;
  for (j=Nsubnet-1; j>=1; j--) {
    CHKERRQ(DMNetworkAddSharedVertices(dmnetwork,0,j,2,asvtx,bsvtx));
  }

  /* Setup the network layout */
  CHKERRQ(DMNetworkLayoutSetUp(dmnetwork));

  /* Get Subnetwork(); Add nvar=1 to subnet[0] and nvar=2 to other subnets */
  for (net=0; net<Nsubnet; net++) {
    CHKERRQ(DMNetworkGetSubnetwork(dmnetwork,net,&nv,&ne,&vtx,&edges));
    for (v=0; v<nv; v++) {
      CHKERRQ(DMNetworkIsSharedVertex(dmnetwork,vtx[v],&sharedv));
      if (sharedv) continue;

      if (!net) {
        /* Set nvar = 2 for subnet0 */
        CHKERRQ(DMNetworkAddComponent(dmnetwork,vtx[v],compkey0,NULL,2));
      } else {
        /* Set nvar = 1 for other subnets */
        CHKERRQ(DMNetworkAddComponent(dmnetwork,vtx[v],compkey1,NULL,1));
      }
    }
  }

  /* Add nvar to shared vertex -- owning and all ghost ranks must call DMNetworkAddComponent() */
  CHKERRQ(DMNetworkGetSharedVertices(dmnetwork,&nv,&vtx));
  for (v=0; v<nv; v++) {
    CHKERRQ(DMNetworkAddComponent(dmnetwork,vtx[v],compkey0,NULL,2));
    CHKERRQ(DMNetworkAddComponent(dmnetwork,vtx[v],compkey1,NULL,1));
  }

  /* Enable runtime option of graph partition type -- must be called before DMSetUp() */
  if (size > 1) {
    DM               plexdm;
    PetscPartitioner part;
    CHKERRQ(DMNetworkGetPlex(dmnetwork,&plexdm));
    CHKERRQ(DMPlexGetPartitioner(plexdm, &part));
    CHKERRQ(PetscPartitionerSetType(part,PETSCPARTITIONERSIMPLE));
    CHKERRQ(PetscOptionsSetValue(NULL,"-dm_plex_csr_alg","mat")); /* for parmetis */
  }

  /* Setup dmnetwork */
  CHKERRQ(DMSetUp(dmnetwork));

  /* Redistribute the network layout; use '-distribute false' to skip */
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-distribute",&distribute,NULL));
  if (distribute) {
    CHKERRQ(DMNetworkDistribute(&dmnetwork,0));
    CHKERRQ(DMView(dmnetwork,PETSC_VIEWER_STDOUT_WORLD));
  }

  /* Create a global vector */
  CHKERRQ(DMCreateGlobalVector(dmnetwork,&X));
  CHKERRQ(VecSet(X,0.0));

  /* Set X values at shared vertex */
  CHKERRQ(DMNetworkGetSharedVertices(dmnetwork,&nv,&vtx));
  for (v=0; v<nv; v++) {
    CHKERRQ(DMNetworkIsGhostVertex(dmnetwork,vtx[v],&ghost));
    if (ghost) continue;

    /* only one process holds a non-ghost vertex */
    CHKERRQ(DMNetworkGetComponent(dmnetwork,vtx[v],ALL_COMPONENTS,NULL,NULL,&nvar));
    CHKERRQ(DMNetworkGetGlobalVecOffset(dmnetwork,vtx[v],ALL_COMPONENTS,&goffset));
    for (i=0; i<nvar; i++) {
      row = goffset + i;
      val = (PetscScalar)rank + 1.0;
      CHKERRQ(VecSetValues(X,1,&row,&val,ADD_VALUES));
    }

    CHKERRQ(DMNetworkGetComponent(dmnetwork,vtx[v],1,&compkey,NULL,&nvar));
    CHKERRQ(DMNetworkGetGlobalVecOffset(dmnetwork,vtx[v],compkey,&goffset));
    for (i=0; i<nvar; i++) {
      row = goffset + i;
      val = 1.0;
      CHKERRQ(VecSetValues(X,1,&row,&val,ADD_VALUES));
    }
  }
  CHKERRQ(VecAssemblyBegin(X));
  CHKERRQ(VecAssemblyEnd(X));
  CHKERRQ(VecView(X,PETSC_VIEWER_STDOUT_WORLD));

  /* Free work space */
  CHKERRQ(VecDestroy(&X));
  for (i=0; i<Nsubnet; i++) {
    if (size == 1 || rank == i) CHKERRQ(PetscFree(edgelist[i]));
  }
  CHKERRQ(PetscFree2(numEdges,edgelist));
  CHKERRQ(DMDestroy(&dmnetwork));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
      requires: !single double defined(PETSC_HAVE_ATTRIBUTEALIGNED)

   test:
      args:

   test:
      suffix: 2
      nsize: 2
      args: -options_left no -petscpartitioner_type simple

   test:
      suffix: 3
      nsize: 4
      args: -options_left no -petscpartitioner_type simple

TEST*/
