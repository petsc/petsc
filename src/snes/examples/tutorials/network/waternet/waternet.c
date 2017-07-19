static char help[] = "This example demonstrates the use of DMNetwork interface for solving a steady-state water network model.\n\
                      The water network equations followed those used for the package EPANET, i.e., unknown head at nodes.\n\
                      The data file format used is from the EPANET package (https://www.epa.gov/water-research/epanet).\n\
                      Run this program: mpiexec -n <n> ./waternet\n\
                      mpiexec -n <n> ./waternet \n";

/* T
   Concepts: DMNetwork
   Concepts: PETSc SNES solver
*/

#include "waternet.h"
#include <petscdmnetwork.h>

#undef __FUNCT__
#define __FUNCT__ "FormFunction"
PetscErrorCode FormFunction(SNES snes,Vec X, Vec F, void *user)
{
  PetscErrorCode ierr;
  DM             networkdm;
  Vec           localX,localF;
  PetscInt      nv,ne;
  const PetscInt *vtx,*edges;

  PetscFunctionBegin;
  ierr = SNESGetDM(snes,&networkdm);CHKERRQ(ierr);
  ierr = DMGetLocalVector(networkdm,&localX);CHKERRQ(ierr);
  ierr = DMGetLocalVector(networkdm,&localF);CHKERRQ(ierr);
  ierr = VecSet(F,0.0);CHKERRQ(ierr);

  ierr = DMGlobalToLocalBegin(networkdm,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(networkdm,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  
  ierr = DMGlobalToLocalBegin(networkdm,F,INSERT_VALUES,localF);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(networkdm,F,INSERT_VALUES,localF);CHKERRQ(ierr);
  
  /* Form Function for first subnetwork */
  ierr = DMNetworkGetSubnetworkInfo(networkdm,0,&nv,&ne,&vtx,&edges);CHKERRQ(ierr);
  
  ierr = DMRestoreLocalVector(networkdm,&localX);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(networkdm,localF,ADD_VALUES,F);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(networkdm,localF,ADD_VALUES,F);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(networkdm,&localF);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetInitialGuess"
PetscErrorCode SetInitialGuess(DM networkdm,Vec X)
{
  PetscErrorCode ierr;
  PetscInt       nv,ne;
  const PetscInt *vtx,*edges;
  Vec            localX;
  PetscScalar    *xarr;
  DMNetworkComponentGenericDataType *arr;
  PetscInt       key,i,offset,offsetd;
  VERTEXDATA     vertex;

  PetscFunctionBegin;

  ierr = DMGetLocalVector(networkdm,&localX);CHKERRQ(ierr);

  ierr = VecSet(X,0.0);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(networkdm,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(networkdm,X,INSERT_VALUES,localX);CHKERRQ(ierr);

  /* Get subnetwork info */
  ierr = DMNetworkGetSubnetworkInfo(networkdm,0,&nv,&ne,&vtx,&edges);CHKERRQ(ierr);

  ierr = VecGetArray(localX,&xarr);CHKERRQ(ierr);
  ierr = DMNetworkGetComponentDataArray(networkdm,&arr);CHKERRQ(ierr);
  for(i=0; i < nv; i++) {
    ierr = DMNetworkGetVariableOffset(networkdm,vtx[i],&offset);CHKERRQ(ierr);
    ierr = DMNetworkGetComponentKeyOffset(networkdm,vtx[i],0,&key,&offsetd);CHKERRQ(ierr);
    vertex = (VERTEXDATA)(arr+offsetd);

    if(vertex->type == VERTEX_TYPE_JUNCTION) {
      xarr[i] = 100;
    } else if(vertex->type == VERTEX_TYPE_RESERVOIR) {
      xarr[i] = vertex->res.head;
    } else {
      xarr[i] = vertex->tank.initlvl + vertex->tank.elev;
    }
  }

  ierr = DMLocalToGlobalBegin(networkdm,localX,ADD_VALUES,X);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(networkdm,localX,ADD_VALUES,X);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(networkdm,&localX);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "GetListofEdges"
PetscErrorCode GetListofEdges(WATERNETDATA *waternet,int *edgelist)
{
  PetscInt       i,j;
  Pipe           *pipe;
  Pump           *pump;
  PetscInt       node1,node2;

  PetscFunctionBegin;
  for(i=0; i < waternet->nedge; i++) {
    if(waternet->edge[i].type == EDGE_TYPE_PIPE) {
      pipe = &waternet->edge[i].pipe;
      node1 = pipe->node1;
      node2 = pipe->node2;
    } else {
      pump = &waternet->edge[i].pump;
      node1 = pump->node1;
      node2 = pump->node2;
    }

    for(j=0; j < waternet->nvertex; j++) {
      if(waternet->vertex[j].id == node1) {
	edgelist[2*i] = j;
	break;
      }
    }

    for(j=0; j < waternet->nvertex; j++) {
      if(waternet->vertex[j].id == node2) {
	edgelist[2*i+1] = j;
	break;
      }
    }
  }

  PetscFunctionReturn(0);
}

int main(int argc,char ** argv)
{
  PetscErrorCode   ierr;
  char             waternetdata_file[PETSC_MAX_PATH_LEN]="datafiles/sample1.inp";
  WATERNETDATA     *waternetdata;
  PetscInt         componentkey[2];
  PetscLogStage    stage1,stage2;
  PetscMPIInt      crank;
  DM               networkdm;
  int              *edgelist = NULL;
  PetscInt         nv,ne,i;
  const PetscInt   *vtx,*edges;
  Vec              X,F;
  SNES             snes;

  ierr = PetscInitialize(&argc,&argv,"pfoptions",help);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&crank);CHKERRQ(ierr);

  /* Create an empty network object */
  ierr = DMNetworkCreate(PETSC_COMM_WORLD,&networkdm);CHKERRQ(ierr);
  /* Register the components in the network */
  ierr = DMNetworkRegisterComponent(networkdm,"edgestruct",sizeof(struct _p_EDGEDATA),&componentkey[0]);CHKERRQ(ierr);
  ierr = DMNetworkRegisterComponent(networkdm,"busstruct",sizeof(struct _p_VERTEXDATA),&componentkey[1]);CHKERRQ(ierr);
  
  ierr = PetscLogStageRegister("Read Data",&stage1);CHKERRQ(ierr);
  PetscLogStagePush(stage1);
  /* READ THE DATA */
  if (!crank) {
    /*    READ DATA */
    /* Only rank 0 reads the data */
    ierr = PetscOptionsGetString(NULL,NULL,"-waternetdata",waternetdata_file,PETSC_MAX_PATH_LEN-1,NULL);CHKERRQ(ierr);
    ierr = PetscNew(&waternetdata);CHKERRQ(ierr);
    ierr = WaterNetReadData(waternetdata,waternetdata_file);CHKERRQ(ierr);
    
    ierr = PetscCalloc1(2*waternetdata->nedge,&edgelist);CHKERRQ(ierr);
    ierr = GetListofEdges(waternetdata,edgelist);CHKERRQ(ierr);
  }

  PetscLogStagePop();
  ierr = PetscLogStageRegister("Create network",&stage2);CHKERRQ(ierr);
  PetscLogStagePush(stage2);

  /* Set number of nodes/edges */
  ierr = DMNetworkSetSizes(networkdm,1,&waternetdata->nvertex,&waternetdata->nedge,&waternetdata->nvertex,&waternetdata->nedge);CHKERRQ(ierr);
  /* Add edge connectivity */
  ierr = DMNetworkSetEdgeList(networkdm,&edgelist);CHKERRQ(ierr);
  /* Set up the network layout */
  ierr = DMNetworkLayoutSetUp(networkdm);CHKERRQ(ierr);

  if (!crank) {
    ierr = PetscFree(edgelist);CHKERRQ(ierr);
  }

  /* ADD VARIABLES AND COMPONENTS FOR THE FIRST SUBNETWORK */
  ierr = DMNetworkGetSubnetworkInfo(networkdm,0,&nv,&ne,&vtx,&edges);CHKERRQ(ierr);

  for (i = 0; i < ne; i++) {
    ierr = DMNetworkAddComponent(networkdm,edges[i],componentkey[0],&waternetdata->edge[i]);CHKERRQ(ierr);
  }

  for (i = 0; i < nv; i++) {
    ierr = DMNetworkAddComponent(networkdm,vtx[i],componentkey[1],&waternetdata->vertex[i]);CHKERRQ(ierr);
    /* Add number of variables */
    ierr = DMNetworkAddNumVariables(networkdm,vtx[i],1);CHKERRQ(ierr);
  }

  /* Set up DM for use */
  ierr = DMSetUp(networkdm);CHKERRQ(ierr);

  /* Distribute networkdm to multiple processes */
  ierr = DMNetworkDistribute(&networkdm,0);CHKERRQ(ierr);
  
  PetscLogStagePop();

  ierr = DMCreateGlobalVector(networkdm,&X);CHKERRQ(ierr);
  ierr = VecDuplicate(X,&F);CHKERRQ(ierr);
  
  /* HOOK UP SOLVER */
  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
  ierr = SNESSetDM(snes,networkdm);CHKERRQ(ierr);
  ierr = SNESSetFunction(snes,F,FormFunction,NULL);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
  
  ierr = SetInitialGuess(networkdm,X);CHKERRQ(ierr);

  ierr = SNESSolve(snes,NULL,X);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}
