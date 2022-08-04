static char help[] = "This example tests subnetwork coupling with zero size components. \n\n";

#include <petscdmnetwork.h>

int main(int argc,char ** argv)
{
  PetscMPIInt    size,rank;
  DM             dmnetwork;
  PetscInt       i,j,net,Nsubnet,ne,nv,nvar,v,goffset,row,compkey0,compkey1,compkey;
  PetscInt       *numEdges,**edgelist,asvtx[2],bsvtx[2];
  const PetscInt *vtx,*edges;
  PetscBool      ghost,distribute=PETSC_TRUE,sharedv;
  Vec            X;
  PetscScalar    val;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));

  /* Create a network of subnetworks */
  if (size == 1) Nsubnet = 2;
  else Nsubnet = (PetscInt)size;
  PetscCall(PetscCalloc2(Nsubnet,&numEdges,Nsubnet,&edgelist));

  /* when size>1, process[i] creates subnetwork[i] */
  for (i=0; i<Nsubnet; i++) {
    if (i == 0 && (size == 1 || (rank == i && size >1))) {
      numEdges[i] = 3;
      PetscCall(PetscMalloc1(2*numEdges[i],&edgelist[i]));
      edgelist[i][0] = 0; edgelist[i][1] = 1;
      edgelist[i][2] = 1; edgelist[i][3] = 2;
      edgelist[i][4] = 2; edgelist[i][5] = 3;

    } else if (i == 1 && (size == 1 || (rank == i && size >1))) {
      numEdges[i] = 3;
      PetscCall(PetscMalloc1(2*numEdges[i],&edgelist[i]));
      edgelist[i][0] = 0; edgelist[i][1] = 1;
      edgelist[i][2] = 1; edgelist[i][3] = 2;
      edgelist[i][4] = 2; edgelist[i][5] = 3;

    } else if (i>1 && (size == 1 || (rank == i && size >1))) {
      numEdges[i] = 3;
      PetscCall(PetscMalloc1(2*numEdges[i],&edgelist[i]));
      for (j=0; j< numEdges[i]; j++) {
        edgelist[i][2*j] = j; edgelist[i][2*j+1] = j+1;
      }
    }
  }

  /* Create a dmnetwork */
  PetscCall(DMNetworkCreate(PETSC_COMM_WORLD,&dmnetwork));

  /* Register zero size components to get compkeys to be used by DMNetworkAddComponent() */
  PetscCall(DMNetworkRegisterComponent(dmnetwork,"comp0",0,&compkey0));
  PetscCall(DMNetworkRegisterComponent(dmnetwork,"comp1",0,&compkey1));

  /* Set number of subnetworks, numbers of vertices and edges over each subnetwork */
  PetscCall(DMNetworkSetNumSubNetworks(dmnetwork,PETSC_DECIDE,Nsubnet));

  for (i=0; i<Nsubnet; i++) {
    PetscInt netNum = -1;
    PetscCall(DMNetworkAddSubnetwork(dmnetwork,NULL,numEdges[i],edgelist[i],&netNum));
  }

  /* Add shared vertices -- all processes hold this info at current implementation
       net[0].0 -> net[j].0, j=0,...,Nsubnet-1
       net[0].1 -> net[j].1, j=0,...,Nsubnet-1 */
  asvtx[0] = bsvtx[0] = 0;
  asvtx[1] = bsvtx[1] = 1;
  for (j=Nsubnet-1; j>=1; j--) {
    PetscCall(DMNetworkAddSharedVertices(dmnetwork,0,j,2,asvtx,bsvtx));
  }

  /* Setup the network layout */
  PetscCall(DMNetworkLayoutSetUp(dmnetwork));

  /* Get Subnetwork(); Add nvar=1 to subnet[0] and nvar=2 to other subnets */
  for (net=0; net<Nsubnet; net++) {
    PetscCall(DMNetworkGetSubnetwork(dmnetwork,net,&nv,&ne,&vtx,&edges));
    for (v=0; v<nv; v++) {
      PetscCall(DMNetworkIsSharedVertex(dmnetwork,vtx[v],&sharedv));
      if (sharedv) continue;

      if (!net) {
        /* Set nvar = 2 for subnet0 */
        PetscCall(DMNetworkAddComponent(dmnetwork,vtx[v],compkey0,NULL,2));
      } else {
        /* Set nvar = 1 for other subnets */
        PetscCall(DMNetworkAddComponent(dmnetwork,vtx[v],compkey1,NULL,1));
      }
    }
  }

  /* Add nvar to shared vertex -- owning and all ghost ranks must call DMNetworkAddComponent() */
  PetscCall(DMNetworkGetSharedVertices(dmnetwork,&nv,&vtx));
  for (v=0; v<nv; v++) {
    PetscCall(DMNetworkAddComponent(dmnetwork,vtx[v],compkey0,NULL,2));
    PetscCall(DMNetworkAddComponent(dmnetwork,vtx[v],compkey1,NULL,1));
  }

  /* Enable runtime option of graph partition type -- must be called before DMSetUp() */
  if (size > 1) {
    DM               plexdm;
    PetscPartitioner part;
    PetscCall(DMNetworkGetPlex(dmnetwork,&plexdm));
    PetscCall(DMPlexGetPartitioner(plexdm, &part));
    PetscCall(PetscPartitionerSetType(part,PETSCPARTITIONERSIMPLE));
    PetscCall(PetscOptionsSetValue(NULL,"-dm_plex_csr_alg","mat")); /* for parmetis */
  }

  /* Setup dmnetwork */
  PetscCall(DMSetUp(dmnetwork));

  /* Redistribute the network layout; use '-distribute false' to skip */
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-distribute",&distribute,NULL));
  if (distribute) {
    PetscCall(DMNetworkDistribute(&dmnetwork,0));
    PetscCall(DMView(dmnetwork,PETSC_VIEWER_STDOUT_WORLD));
  }

  /* Create a global vector */
  PetscCall(DMCreateGlobalVector(dmnetwork,&X));
  PetscCall(VecSet(X,0.0));

  /* Set X values at shared vertex */
  PetscCall(DMNetworkGetSharedVertices(dmnetwork,&nv,&vtx));
  for (v=0; v<nv; v++) {
    PetscCall(DMNetworkIsGhostVertex(dmnetwork,vtx[v],&ghost));
    if (ghost) continue;

    /* only one process holds a non-ghost vertex */
    PetscCall(DMNetworkGetComponent(dmnetwork,vtx[v],ALL_COMPONENTS,NULL,NULL,&nvar));
    PetscCall(DMNetworkGetGlobalVecOffset(dmnetwork,vtx[v],ALL_COMPONENTS,&goffset));
    for (i=0; i<nvar; i++) {
      row = goffset + i;
      val = (PetscScalar)rank + 1.0;
      PetscCall(VecSetValues(X,1,&row,&val,ADD_VALUES));
    }

    PetscCall(DMNetworkGetComponent(dmnetwork,vtx[v],1,&compkey,NULL,&nvar));
    PetscCall(DMNetworkGetGlobalVecOffset(dmnetwork,vtx[v],compkey,&goffset));
    for (i=0; i<nvar; i++) {
      row = goffset + i;
      val = 1.0;
      PetscCall(VecSetValues(X,1,&row,&val,ADD_VALUES));
    }
  }
  PetscCall(VecAssemblyBegin(X));
  PetscCall(VecAssemblyEnd(X));
  PetscCall(VecView(X,PETSC_VIEWER_STDOUT_WORLD));

  /* Free work space */
  PetscCall(VecDestroy(&X));
  for (i=0; i<Nsubnet; i++) {
    if (size == 1 || rank == i) PetscCall(PetscFree(edgelist[i]));
  }
  PetscCall(PetscFree2(numEdges,edgelist));
  PetscCall(DMDestroy(&dmnetwork));
  PetscCall(PetscFinalize());
  return 0;
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
