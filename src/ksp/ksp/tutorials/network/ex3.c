static char help[] = "This example tests subnetwork coupling. \n\
              \n\n";

/* T
  Concepts: DMNetwork
*/
#include <petscdmnetwork.h>

typedef struct{
  PetscInt id;
} Comp0;

typedef struct{
  PetscScalar val;
} Comp1;

int main(int argc,char ** argv)
{
  PetscMPIInt    size,rank;
  DM             dmnetwork;
  PetscInt       i,j,net,Nsubnet,nsubnet,ne,nv,nvar,v,ncomp,compkey0,compkey1,compkey,goffset,row;
  PetscInt       numEdges[10],*edgelist[10],asvtx,bsvtx;
  const PetscInt *vtx,*edges;
  PetscBool      sharedv,ghost,distribute=PETSC_TRUE,test=PETSC_FALSE;
  Vec            X;
  Comp0          comp0;
  Comp1          comp1;
  PetscScalar    val;

  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));

  /* Create a network of subnetworks */
  nsubnet = 1;
  if (size == 1) nsubnet = 2;

  /* Create a dmnetwork and register components */
  CHKERRQ(DMNetworkCreate(PETSC_COMM_WORLD,&dmnetwork));
  CHKERRQ(DMNetworkRegisterComponent(dmnetwork,"comp0",sizeof(Comp0),&compkey0));
  CHKERRQ(DMNetworkRegisterComponent(dmnetwork,"comp1",sizeof(Comp1),&compkey1));

  /* Set componnet values - intentionally take rank-dependent value for test */
  comp0.id  = rank;
  comp1.val = 10.0*rank;

  /* Set number of subnetworks, numbers of vertices and edges over each subnetwork */
  CHKERRQ(DMNetworkSetNumSubNetworks(dmnetwork,nsubnet,PETSC_DECIDE));
  CHKERRQ(DMNetworkGetNumSubNetworks(dmnetwork,NULL,&Nsubnet));

  /* Input subnetworks; when size>1, process[i] creates subnetwork[i] */
  for (i=0; i<Nsubnet; i++) numEdges[i] = 0;
  for (i=0; i<Nsubnet; i++) {
    if (i == 0 && (size == 1 || (rank == i && size >1))) {
      numEdges[i] = 3;
      CHKERRQ(PetscMalloc1(2*numEdges[i],&edgelist[i]));
      edgelist[i][0] = 0; edgelist[i][1] = 2;
      edgelist[i][2] = 2; edgelist[i][3] = 1;
      edgelist[i][4] = 1; edgelist[i][5] = 3;

    } else if (i == 1 && (size == 1 || (rank == i && size >1))) {
      numEdges[i] = 3;
      CHKERRQ(PetscMalloc1(2*numEdges[i],&edgelist[i]));
      edgelist[i][0] = 0; edgelist[i][1] = 3;
      edgelist[i][2] = 3; edgelist[i][3] = 2;
      edgelist[i][4] = 2; edgelist[i][5] = 1;

    } else if (i>1 && (size == 1 || (rank == i && size >1))) {
      numEdges[i] = 3;
      CHKERRQ(PetscMalloc1(2*numEdges[i],&edgelist[i]));
      for (j=0; j< numEdges[i]; j++) {
        edgelist[i][2*j] = j; edgelist[i][2*j+1] = j+1;
      }
    }
  }

  /* Add subnetworks */
  for (i=0; i<Nsubnet; i++) {
    PetscInt netNum = -1;
    CHKERRQ(DMNetworkAddSubnetwork(dmnetwork,NULL,numEdges[i],edgelist[i],&netNum));
  }

  /* Add shared vertices -- all processes hold this info at current implementation */
  asvtx = bsvtx = 0;
  for (j=1; j<Nsubnet; j++) {
    /* vertex subnet[0].0 shares with vertex subnet[j].0 */
    CHKERRQ(DMNetworkAddSharedVertices(dmnetwork,0,j,1,&asvtx,&bsvtx));
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
        CHKERRQ(DMNetworkAddComponent(dmnetwork,vtx[v],compkey0,&comp0,1));
      } else {
        CHKERRQ(DMNetworkAddComponent(dmnetwork,vtx[v],compkey1,&comp1,2));
      }
    }
  }

  /* Add components and nvar to shared vertex -- owning and all ghost ranks must call DMNetworkAddComponent() */
  CHKERRQ(DMNetworkGetSharedVertices(dmnetwork,&nv,&vtx));
  for (v=0; v<nv; v++) {
    CHKERRQ(DMNetworkAddComponent(dmnetwork,vtx[v],compkey0,&comp0,1));
    CHKERRQ(DMNetworkAddComponent(dmnetwork,vtx[v],compkey1,&comp1,2));
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
    CHKERRQ(DMNetworkGetNumComponents(dmnetwork,vtx[v],&ncomp));
    /* CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"[%d] shared v %D: nvar %D, ncomp %D\n",rank,vtx[v],nvar,ncomp)); */
    for (j=0; j<ncomp; j++) {
      CHKERRQ(DMNetworkGetComponent(dmnetwork,vtx[v],j,&compkey,NULL,&nvar));
      CHKERRQ(DMNetworkGetGlobalVecOffset(dmnetwork,vtx[v],j,&goffset));
      for (i=0; i<nvar; i++) {
        row = goffset + i;
        val = compkey + 1.0;
        CHKERRQ(VecSetValues(X,1,&row,&val,INSERT_VALUES));
      }
    }
  }
  CHKERRQ(VecAssemblyBegin(X));
  CHKERRQ(VecAssemblyEnd(X));
  CHKERRQ(VecView(X,PETSC_VIEWER_STDOUT_WORLD));

  /* Test DMNetworkGetSubnetwork() */
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-test_getsubnet",&test,NULL));
  if (test) {
    net = 0;
    CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-subnet",&net,NULL));
    CHKERRQ(DMNetworkGetSubnetwork(dmnetwork,net,&nv,&ne,&vtx,&edges));
    CHKERRQ(PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d] subnet %D: nv %D, ne %D\n",rank,net,nv,ne));
    CHKERRQ(PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT));
    CHKERRMPI(MPI_Barrier(PETSC_COMM_WORLD));

    for (i=0; i<nv; i++) {
      CHKERRQ(DMNetworkIsGhostVertex(dmnetwork,vtx[i],&ghost));
      CHKERRQ(DMNetworkIsSharedVertex(dmnetwork,vtx[i],&sharedv));

      CHKERRQ(DMNetworkGetNumComponents(dmnetwork,vtx[i],&ncomp));
      if (sharedv || ghost) {
        CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"  [%d] v %D is shared %d, is ghost %d, ncomp %D\n",rank,vtx[i],sharedv,ghost,ncomp));
      }

      for (j=0; j<ncomp; j++) {
        void* component;
        CHKERRQ(DMNetworkGetComponent(dmnetwork,vtx[i],j,&compkey,(void**)&component,NULL));
        if (compkey == 0) {
          Comp0  *mycomp0;
          mycomp0 = (Comp0*)component;
          CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"  [%d] v %D compkey %D, mycomp0->id %D\n",rank,vtx[i],compkey,mycomp0->id));
        } else if (compkey == 1) {
          Comp1  *mycomp1;
          mycomp1 = (Comp1*)component;
          CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"  [%d] v %D compkey %D, mycomp1->val %g\n",rank,vtx[i],compkey,mycomp1->val));
        }
      }
    }
  }

  /* Free work space */
  CHKERRQ(VecDestroy(&X));
  for (i=0; i<Nsubnet; i++) {
    if (size == 1 || rank == i) CHKERRQ(PetscFree(edgelist[i]));
  }

  CHKERRQ(DMDestroy(&dmnetwork));
  CHKERRQ(PetscFinalize());
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
      args: -options_left no

   test:
      suffix: 3
      nsize: 4
      args: -options_left no

TEST*/
