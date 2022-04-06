static char help[] = "This example tests subnetwork coupling. \n\
              \n\n";

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

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));

  /* Create a network of subnetworks */
  nsubnet = 1;
  if (size == 1) nsubnet = 2;

  /* Create a dmnetwork and register components */
  PetscCall(DMNetworkCreate(PETSC_COMM_WORLD,&dmnetwork));
  PetscCall(DMNetworkRegisterComponent(dmnetwork,"comp0",sizeof(Comp0),&compkey0));
  PetscCall(DMNetworkRegisterComponent(dmnetwork,"comp1",sizeof(Comp1),&compkey1));

  /* Set componnet values - intentionally take rank-dependent value for test */
  comp0.id  = rank;
  comp1.val = 10.0*rank;

  /* Set number of subnetworks, numbers of vertices and edges over each subnetwork */
  PetscCall(DMNetworkSetNumSubNetworks(dmnetwork,nsubnet,PETSC_DECIDE));
  PetscCall(DMNetworkGetNumSubNetworks(dmnetwork,NULL,&Nsubnet));

  /* Input subnetworks; when size>1, process[i] creates subnetwork[i] */
  for (i=0; i<Nsubnet; i++) numEdges[i] = 0;
  for (i=0; i<Nsubnet; i++) {
    if (i == 0 && (size == 1 || (rank == i && size >1))) {
      numEdges[i] = 3;
      PetscCall(PetscMalloc1(2*numEdges[i],&edgelist[i]));
      edgelist[i][0] = 0; edgelist[i][1] = 2;
      edgelist[i][2] = 2; edgelist[i][3] = 1;
      edgelist[i][4] = 1; edgelist[i][5] = 3;

    } else if (i == 1 && (size == 1 || (rank == i && size >1))) {
      numEdges[i] = 3;
      PetscCall(PetscMalloc1(2*numEdges[i],&edgelist[i]));
      edgelist[i][0] = 0; edgelist[i][1] = 3;
      edgelist[i][2] = 3; edgelist[i][3] = 2;
      edgelist[i][4] = 2; edgelist[i][5] = 1;

    } else if (i>1 && (size == 1 || (rank == i && size >1))) {
      numEdges[i] = 3;
      PetscCall(PetscMalloc1(2*numEdges[i],&edgelist[i]));
      for (j=0; j< numEdges[i]; j++) {
        edgelist[i][2*j] = j; edgelist[i][2*j+1] = j+1;
      }
    }
  }

  /* Add subnetworks */
  for (i=0; i<Nsubnet; i++) {
    PetscInt netNum = -1;
    PetscCall(DMNetworkAddSubnetwork(dmnetwork,NULL,numEdges[i],edgelist[i],&netNum));
  }

  /* Add shared vertices -- all processes hold this info at current implementation */
  asvtx = bsvtx = 0;
  for (j=1; j<Nsubnet; j++) {
    /* vertex subnet[0].0 shares with vertex subnet[j].0 */
    PetscCall(DMNetworkAddSharedVertices(dmnetwork,0,j,1,&asvtx,&bsvtx));
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
        PetscCall(DMNetworkAddComponent(dmnetwork,vtx[v],compkey0,&comp0,1));
      } else {
        PetscCall(DMNetworkAddComponent(dmnetwork,vtx[v],compkey1,&comp1,2));
      }
    }
  }

  /* Add components and nvar to shared vertex -- owning and all ghost ranks must call DMNetworkAddComponent() */
  PetscCall(DMNetworkGetSharedVertices(dmnetwork,&nv,&vtx));
  for (v=0; v<nv; v++) {
    PetscCall(DMNetworkAddComponent(dmnetwork,vtx[v],compkey0,&comp0,1));
    PetscCall(DMNetworkAddComponent(dmnetwork,vtx[v],compkey1,&comp1,2));
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
    PetscCall(DMNetworkGetNumComponents(dmnetwork,vtx[v],&ncomp));
    /* PetscCall(PetscPrintf(PETSC_COMM_SELF,"[%d] shared v %D: nvar %D, ncomp %D\n",rank,vtx[v],nvar,ncomp)); */
    for (j=0; j<ncomp; j++) {
      PetscCall(DMNetworkGetComponent(dmnetwork,vtx[v],j,&compkey,NULL,&nvar));
      PetscCall(DMNetworkGetGlobalVecOffset(dmnetwork,vtx[v],j,&goffset));
      for (i=0; i<nvar; i++) {
        row = goffset + i;
        val = compkey + 1.0;
        PetscCall(VecSetValues(X,1,&row,&val,INSERT_VALUES));
      }
    }
  }
  PetscCall(VecAssemblyBegin(X));
  PetscCall(VecAssemblyEnd(X));
  PetscCall(VecView(X,PETSC_VIEWER_STDOUT_WORLD));

  /* Test DMNetworkGetSubnetwork() */
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-test_getsubnet",&test,NULL));
  if (test) {
    net = 0;
    PetscCall(PetscOptionsGetInt(NULL,NULL,"-subnet",&net,NULL));
    PetscCall(DMNetworkGetSubnetwork(dmnetwork,net,&nv,&ne,&vtx,&edges));
    PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d] subnet %D: nv %D, ne %D\n",rank,net,nv,ne));
    PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT));
    PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));

    for (i=0; i<nv; i++) {
      PetscCall(DMNetworkIsGhostVertex(dmnetwork,vtx[i],&ghost));
      PetscCall(DMNetworkIsSharedVertex(dmnetwork,vtx[i],&sharedv));

      PetscCall(DMNetworkGetNumComponents(dmnetwork,vtx[i],&ncomp));
      if (sharedv || ghost) {
        PetscCall(PetscPrintf(PETSC_COMM_SELF,"  [%d] v %D is shared %d, is ghost %d, ncomp %D\n",rank,vtx[i],sharedv,ghost,ncomp));
      }

      for (j=0; j<ncomp; j++) {
        void* component;
        PetscCall(DMNetworkGetComponent(dmnetwork,vtx[i],j,&compkey,(void**)&component,NULL));
        if (compkey == 0) {
          Comp0  *mycomp0;
          mycomp0 = (Comp0*)component;
          PetscCall(PetscPrintf(PETSC_COMM_SELF,"  [%d] v %D compkey %D, mycomp0->id %D\n",rank,vtx[i],compkey,mycomp0->id));
        } else if (compkey == 1) {
          Comp1  *mycomp1;
          mycomp1 = (Comp1*)component;
          PetscCall(PetscPrintf(PETSC_COMM_SELF,"  [%d] v %D compkey %D, mycomp1->val %g\n",rank,vtx[i],compkey,mycomp1->val));
        }
      }
    }
  }

  /* Free work space */
  PetscCall(VecDestroy(&X));
  for (i=0; i<Nsubnet; i++) {
    if (size == 1 || rank == i) PetscCall(PetscFree(edgelist[i]));
  }

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
      args: -options_left no

   test:
      suffix: 3
      nsize: 4
      args: -options_left no

TEST*/
