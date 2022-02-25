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
  PetscErrorCode ierr;
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

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);

  /* Create a network of subnetworks */
  nsubnet = 1;
  if (size == 1) nsubnet = 2;

  /* Create a dmnetwork and register components */
  ierr = DMNetworkCreate(PETSC_COMM_WORLD,&dmnetwork);CHKERRQ(ierr);
  ierr = DMNetworkRegisterComponent(dmnetwork,"comp0",sizeof(Comp0),&compkey0);CHKERRQ(ierr);
  ierr = DMNetworkRegisterComponent(dmnetwork,"comp1",sizeof(Comp1),&compkey1);CHKERRQ(ierr);

  /* Set componnet values - intentionally take rank-dependent value for test */
  comp0.id  = rank;
  comp1.val = 10.0*rank;

  /* Set number of subnetworks, numbers of vertices and edges over each subnetwork */
  ierr = DMNetworkSetNumSubNetworks(dmnetwork,nsubnet,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = DMNetworkGetNumSubNetworks(dmnetwork,NULL,&Nsubnet);CHKERRQ(ierr);

  /* Input subnetworks; when size>1, process[i] creates subnetwork[i] */
  for (i=0; i<Nsubnet; i++) numEdges[i] = 0;
  for (i=0; i<Nsubnet; i++) {
    if (i == 0 && (size == 1 || (rank == i && size >1))) {
      numEdges[i] = 3;
      ierr = PetscMalloc1(2*numEdges[i],&edgelist[i]);CHKERRQ(ierr);
      edgelist[i][0] = 0; edgelist[i][1] = 2;
      edgelist[i][2] = 2; edgelist[i][3] = 1;
      edgelist[i][4] = 1; edgelist[i][5] = 3;

    } else if (i == 1 && (size == 1 || (rank == i && size >1))) {
      numEdges[i] = 3;
      ierr = PetscMalloc1(2*numEdges[i],&edgelist[i]);CHKERRQ(ierr);
      edgelist[i][0] = 0; edgelist[i][1] = 3;
      edgelist[i][2] = 3; edgelist[i][3] = 2;
      edgelist[i][4] = 2; edgelist[i][5] = 1;

    } else if (i>1 && (size == 1 || (rank == i && size >1))) {
      numEdges[i] = 3;
      ierr = PetscMalloc1(2*numEdges[i],&edgelist[i]);CHKERRQ(ierr);
      for (j=0; j< numEdges[i]; j++) {
        edgelist[i][2*j] = j; edgelist[i][2*j+1] = j+1;
      }
    }
  }

  /* Add subnetworks */
  for (i=0; i<Nsubnet; i++) {
    PetscInt netNum = -1;
    ierr = DMNetworkAddSubnetwork(dmnetwork,NULL,numEdges[i],edgelist[i],&netNum);CHKERRQ(ierr);
  }

  /* Add shared vertices -- all processes hold this info at current implementation */
  asvtx = bsvtx = 0;
  for (j=1; j<Nsubnet; j++) {
    /* vertex subnet[0].0 shares with vertex subnet[j].0 */
    ierr = DMNetworkAddSharedVertices(dmnetwork,0,j,1,&asvtx,&bsvtx);CHKERRQ(ierr);
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
        ierr = DMNetworkAddComponent(dmnetwork,vtx[v],compkey0,&comp0,1);CHKERRQ(ierr);
      } else {
        ierr = DMNetworkAddComponent(dmnetwork,vtx[v],compkey1,&comp1,2);CHKERRQ(ierr);
      }
    }
  }

  /* Add components and nvar to shared vertex -- owning and all ghost ranks must call DMNetworkAddComponent() */
  ierr = DMNetworkGetSharedVertices(dmnetwork,&nv,&vtx);CHKERRQ(ierr);
  for (v=0; v<nv; v++) {
    ierr = DMNetworkAddComponent(dmnetwork,vtx[v],compkey0,&comp0,1);CHKERRQ(ierr);
    ierr = DMNetworkAddComponent(dmnetwork,vtx[v],compkey1,&comp1,2);CHKERRQ(ierr);
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
    ierr = DMNetworkGetNumComponents(dmnetwork,vtx[v],&ncomp);CHKERRQ(ierr);
    /* ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] shared v %D: nvar %D, ncomp %D\n",rank,vtx[v],nvar,ncomp);CHKERRQ(ierr); */
    for (j=0; j<ncomp; j++) {
      ierr = DMNetworkGetComponent(dmnetwork,vtx[v],j,&compkey,NULL,&nvar);CHKERRQ(ierr);
      ierr = DMNetworkGetGlobalVecOffset(dmnetwork,vtx[v],j,&goffset);CHKERRQ(ierr);
      for (i=0; i<nvar; i++) {
        row = goffset + i;
        val = compkey + 1.0;
        ierr = VecSetValues(X,1,&row,&val,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }
  ierr = VecAssemblyBegin(X);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(X);CHKERRQ(ierr);
  ierr = VecView(X,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* Test DMNetworkGetSubnetwork() */
  ierr = PetscOptionsGetBool(NULL,NULL,"-test_getsubnet",&test,NULL);CHKERRQ(ierr);
  if (test) {
    net = 0;
    ierr = PetscOptionsGetInt(NULL,NULL,"-subnet",&net,NULL);CHKERRQ(ierr);
    ierr = DMNetworkGetSubnetwork(dmnetwork,net,&nv,&ne,&vtx,&edges);CHKERRQ(ierr);
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d] subnet %D: nv %D, ne %D\n",rank,net,nv,ne);CHKERRQ(ierr);
    ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);CHKERRQ(ierr);
    ierr = MPI_Barrier(PETSC_COMM_WORLD);CHKERRMPI(ierr);

    for (i=0; i<nv; i++) {
      ierr = DMNetworkIsGhostVertex(dmnetwork,vtx[i],&ghost);CHKERRQ(ierr);
      ierr = DMNetworkIsSharedVertex(dmnetwork,vtx[i],&sharedv);CHKERRQ(ierr);

      ierr = DMNetworkGetNumComponents(dmnetwork,vtx[i],&ncomp);CHKERRQ(ierr);
      if (sharedv || ghost) {
        ierr = PetscPrintf(PETSC_COMM_SELF,"  [%d] v %D is shared %d, is ghost %d, ncomp %D\n",rank,vtx[i],sharedv,ghost,ncomp);CHKERRQ(ierr);
      }

      for (j=0; j<ncomp; j++) {
        void* component;
        ierr = DMNetworkGetComponent(dmnetwork,vtx[i],j,&compkey,(void**)&component,NULL);CHKERRQ(ierr);
        if (compkey == 0) {
          Comp0  *mycomp0;
          mycomp0 = (Comp0*)component;
          ierr = PetscPrintf(PETSC_COMM_SELF,"  [%d] v %D compkey %D, mycomp0->id %D\n",rank,vtx[i],compkey,mycomp0->id);CHKERRQ(ierr);
        } else if (compkey == 1) {
          Comp1  *mycomp1;
          mycomp1 = (Comp1*)component;
          ierr = PetscPrintf(PETSC_COMM_SELF,"  [%d] v %D compkey %D, mycomp1->val %g\n",rank,vtx[i],compkey,mycomp1->val);CHKERRQ(ierr);
        }
      }
    }
  }

  /* Free work space */
  ierr = VecDestroy(&X);CHKERRQ(ierr);
  for (i=0; i<Nsubnet; i++) {
    if (size == 1 || rank == i) {ierr = PetscFree(edgelist[i]);CHKERRQ(ierr);}
  }

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
      args: -options_left no

   test:
      suffix: 3
      nsize: 4
      args: -options_left no

TEST*/
