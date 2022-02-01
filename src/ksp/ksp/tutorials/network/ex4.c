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
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);

  /* Create a network of subnetworks */
  if (size == 1) Nsubnet = 2;
  else Nsubnet = (PetscInt)size;
  ierr = PetscCalloc2(Nsubnet,&numEdges,Nsubnet,&edgelist);CHKERRQ(ierr);

  /* when size>1, process[i] creates subnetwork[i] */
  for (i=0; i<Nsubnet; i++) {
    if (i == 0 && (size == 1 || (rank == i && size >1))) {
      numEdges[i] = 3;
      ierr = PetscMalloc1(2*numEdges[i],&edgelist[i]);CHKERRQ(ierr);
      edgelist[i][0] = 0; edgelist[i][1] = 1;
      edgelist[i][2] = 1; edgelist[i][3] = 2;
      edgelist[i][4] = 2; edgelist[i][5] = 3;

    } else if (i == 1 && (size == 1 || (rank == i && size >1))) {
      numEdges[i] = 3;
      ierr = PetscMalloc1(2*numEdges[i],&edgelist[i]);CHKERRQ(ierr);
      edgelist[i][0] = 0; edgelist[i][1] = 1;
      edgelist[i][2] = 1; edgelist[i][3] = 2;
      edgelist[i][4] = 2; edgelist[i][5] = 3;

    } else if (i>1 && (size == 1 || (rank == i && size >1))) {
      numEdges[i] = 3;
      ierr = PetscMalloc1(2*numEdges[i],&edgelist[i]);CHKERRQ(ierr);
      for (j=0; j< numEdges[i]; j++) {
        edgelist[i][2*j] = j; edgelist[i][2*j+1] = j+1;
      }
    }
  }

  /* Create a dmnetwork */
  ierr = DMNetworkCreate(PETSC_COMM_WORLD,&dmnetwork);CHKERRQ(ierr);

  /* Register zero size componets to get compkeys to be used by DMNetworkAddComponent() */
  ierr = DMNetworkRegisterComponent(dmnetwork,"comp0",0,&compkey0);CHKERRQ(ierr);
  ierr = DMNetworkRegisterComponent(dmnetwork,"comp1",0,&compkey1);CHKERRQ(ierr);

  /* Set number of subnetworks, numbers of vertices and edges over each subnetwork */
  ierr = DMNetworkSetNumSubNetworks(dmnetwork,PETSC_DECIDE,Nsubnet);CHKERRQ(ierr);

  for (i=0; i<Nsubnet; i++) {
    PetscInt netNum = -1;
    ierr = DMNetworkAddSubnetwork(dmnetwork,NULL,numEdges[i],edgelist[i],&netNum);CHKERRQ(ierr);
  }

  /* Add shared vertices -- all processes hold this info at current implementation
       net[0].0 -> net[j].0, j=0,...,Nsubnet-1
       net[0].1 -> net[j].1, j=0,...,Nsubnet-1 */
  asvtx[0] = bsvtx[0] = 0;
  asvtx[1] = bsvtx[1] = 1;
  for (j=Nsubnet-1; j>=1; j--) {
    ierr = DMNetworkAddSharedVertices(dmnetwork,0,j,2,asvtx,bsvtx);CHKERRQ(ierr);
  }

  /* Setup the network layout */
  ierr = DMNetworkLayoutSetUp(dmnetwork);CHKERRQ(ierr);

  /* Get Subnetwork(); Add nvar=1 to subnet[0] and nvar=2 to other subnets */
  for (net=0; net<Nsubnet; net++) {
    ierr = DMNetworkGetSubnetwork(dmnetwork,net,&nv,&ne,&vtx,&edges);CHKERRQ(ierr);
    for (v=0; v<nv; v++) {
      ierr = DMNetworkIsSharedVertex(dmnetwork,vtx[v],&sharedv);CHKERRQ(ierr);
      if (sharedv) continue;

      if (!net) {
        /* Set nvar = 2 for subnet0 */
        ierr = DMNetworkAddComponent(dmnetwork,vtx[v],compkey0,NULL,2);CHKERRQ(ierr);
      } else {
        /* Set nvar = 1 for other subnets */
        ierr = DMNetworkAddComponent(dmnetwork,vtx[v],compkey1,NULL,1);CHKERRQ(ierr);
      }
    }
  }

  /* Add nvar to shared vertex -- owning and all ghost ranks must call DMNetworkAddComponent() */
  ierr = DMNetworkGetSharedVertices(dmnetwork,&nv,&vtx);CHKERRQ(ierr);
  for (v=0; v<nv; v++) {
    ierr = DMNetworkAddComponent(dmnetwork,vtx[v],compkey0,NULL,2);CHKERRQ(ierr);
    ierr = DMNetworkAddComponent(dmnetwork,vtx[v],compkey1,NULL,1);CHKERRQ(ierr);
  }

  /* Enable runtime option of graph partition type -- must be called before DMSetUp() */
  if (size > 1) {
    DM               plexdm;
    PetscPartitioner part;
    ierr = DMNetworkGetPlex(dmnetwork,&plexdm);CHKERRQ(ierr);
    ierr = DMPlexGetPartitioner(plexdm, &part);CHKERRQ(ierr);
    ierr = PetscPartitionerSetType(part,PETSCPARTITIONERSIMPLE);CHKERRQ(ierr);
    ierr = PetscOptionsSetValue(NULL,"-dm_plex_csr_alg","mat");CHKERRQ(ierr); /* for parmetis */
  }

  /* Setup dmnetwork */
  ierr = DMSetUp(dmnetwork);CHKERRQ(ierr);

  /* Redistribute the network layout; use '-distribute false' to skip */
  ierr = PetscOptionsGetBool(NULL,NULL,"-distribute",&distribute,NULL);CHKERRQ(ierr);
  if (distribute) {
    ierr = DMNetworkDistribute(&dmnetwork,0);CHKERRQ(ierr);
    ierr = DMView(dmnetwork,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  /* Create a global vector */
  ierr = DMCreateGlobalVector(dmnetwork,&X);CHKERRQ(ierr);
  ierr = VecSet(X,0.0);CHKERRQ(ierr);

  /* Set X values at shared vertex */
  ierr = DMNetworkGetSharedVertices(dmnetwork,&nv,&vtx);CHKERRQ(ierr);
  for (v=0; v<nv; v++) {
    ierr = DMNetworkIsGhostVertex(dmnetwork,vtx[v],&ghost);CHKERRQ(ierr);
    if (ghost) continue;

    /* only one process holds a non-ghost vertex */
    ierr = DMNetworkGetComponent(dmnetwork,vtx[v],ALL_COMPONENTS,NULL,NULL,&nvar);CHKERRQ(ierr);
    ierr = DMNetworkGetGlobalVecOffset(dmnetwork,vtx[v],ALL_COMPONENTS,&goffset);CHKERRQ(ierr);
    for (i=0; i<nvar; i++) {
      row = goffset + i;
      val = (PetscScalar)rank + 1.0;
      ierr = VecSetValues(X,1,&row,&val,ADD_VALUES);CHKERRQ(ierr);
    }

    ierr = DMNetworkGetComponent(dmnetwork,vtx[v],1,&compkey,NULL,&nvar);CHKERRQ(ierr);
    ierr = DMNetworkGetGlobalVecOffset(dmnetwork,vtx[v],compkey,&goffset);CHKERRQ(ierr);
    for (i=0; i<nvar; i++) {
      row = goffset + i;
      val = 1.0;
      ierr = VecSetValues(X,1,&row,&val,ADD_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = VecAssemblyBegin(X);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(X);CHKERRQ(ierr);
  ierr = VecView(X,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* Free work space */
  ierr = VecDestroy(&X);CHKERRQ(ierr);
  for (i=0; i<Nsubnet; i++) {
    if (size == 1 || rank == i) {ierr = PetscFree(edgelist[i]);CHKERRQ(ierr);}
  }
  ierr = PetscFree2(numEdges,edgelist);CHKERRQ(ierr);
  ierr = DMDestroy(&dmnetwork);CHKERRQ(ierr);
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
