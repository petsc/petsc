static char help[] = "This example demonstrates the use of DMNetwork interface for solving a steady-state water network model.\n\
                      The water network equations follow those used for the package EPANET\n\
                      The data file format used is from the EPANET package (https://www.epa.gov/water-research/epanet).\n\
                      Run this program: mpiexec -n <n> ./waternet\n\\n";

/* T
   Concepts: DMNetwork
   Concepts: PETSc SNES solver
*/

#include "waternet.h"
#include <petscdmnetwork.h>

PetscScalar Flow_Pipe(Pipe *pipe,PetscScalar hf,PetscScalar ht)
{
  PetscScalar flow_pipe;

  flow_pipe = PetscSign(hf-ht)*PetscPowScalar(PetscAbsScalar(hf - ht)/pipe->k,(1/pipe->n));
  return flow_pipe;
}

PetscScalar Flow_Pump(Pump *pump,PetscScalar hf, PetscScalar ht)
{
  PetscScalar flow_pump;
  flow_pump = PetscPowScalar((hf - ht + pump->h0)/pump->r,(1/pump->n));
  return flow_pump;
}

#undef __FUNCT__
#define __FUNCT__ "FormFunction"
PetscErrorCode FormFunction(SNES snes,Vec X, Vec F, void *user)
{
  PetscErrorCode    ierr;
  DM                networkdm;
  Vec               localX,localF;
  const PetscInt    *v,*e;
  const PetscScalar *xarr;
  PetscScalar       *farr,hf,ht,flow;
  PetscInt          nv,ne,key,i,offset,vnode1,vnode2;
  EDGEDATA          edge;
  VERTEXDATA        vertex,vertexnode1,vertexnode2;
  const PetscInt    *cone;
  PetscInt          offsetnode1,offsetnode2;
  Pipe              *pipe;
  Pump              *pump;
  Reservoir         *reservoir;
  Tank              *tank;
  PetscBool         ghostvtex;

  PetscFunctionBegin;
  /* Get the DM attached with the SNES */
  ierr = SNESGetDM(snes,&networkdm);CHKERRQ(ierr);

  /* Get vertices/edges */
  ierr = DMNetworkGetSubnetworkInfo(networkdm,0,&nv,&ne,&v,&e);CHKERRQ(ierr);

  /* Get local vectors */
  ierr = DMGetLocalVector(networkdm,&localX);CHKERRQ(ierr);
  ierr = DMGetLocalVector(networkdm,&localF);CHKERRQ(ierr);

  /* Scatter values from global to local vector */
  ierr = DMGlobalToLocalBegin(networkdm,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(networkdm,X,INSERT_VALUES,localX);CHKERRQ(ierr);

  /* Initialize residual */
  ierr = VecSet(F,0.0);CHKERRQ(ierr);
  ierr = VecSet(localF,0.0);CHKERRQ(ierr);

  /* Get arrays for the vectors */
  ierr = VecGetArrayRead(localX,&xarr);CHKERRQ(ierr);
  ierr = VecGetArray(localF,&farr);CHKERRQ(ierr);

  for(i=0; i < ne; i++) {
    /* Get the offset and the key for the component for edge number e[i] */
    ierr = DMNetworkGetComponent(networkdm,e[i],0,&key,(void**)&edge);CHKERRQ(ierr);

    /* Get the numbers for the vertices covering this edge */
    ierr = DMNetworkGetConnectedVertices(networkdm,e[i],&cone);CHKERRQ(ierr);
    vnode1 = cone[0];
    vnode2 = cone[1];

    /* Get the components at the two vertices */
    ierr = DMNetworkGetComponent(networkdm,vnode1,0,&key,(void**)&vertexnode1);CHKERRQ(ierr);
    ierr = DMNetworkGetComponent(networkdm,vnode2,0,&key,(void**)&vertexnode2);CHKERRQ(ierr);

    /* Get the variable offset (the starting location for the variables in the farr array) for node1 and node2 */
    ierr = DMNetworkGetVariableOffset(networkdm,vnode1,&offsetnode1);CHKERRQ(ierr);
    ierr = DMNetworkGetVariableOffset(networkdm,vnode2,&offsetnode2);CHKERRQ(ierr);

    /* Variables at node1 and node 2 */
    hf = xarr[offsetnode1];
    ht = xarr[offsetnode2];

    if (edge->type == EDGE_TYPE_PIPE) {
      pipe = &edge->pipe;
      flow = Flow_Pipe(pipe,hf,ht);
    } else if (edge->type == EDGE_TYPE_PUMP) {
      pump = &edge->pump;
      flow = Flow_Pump(pump,hf,ht);
    }
    /* Convention: Node 1 has outgoing flow and Node 2 has incoming flow */
    if(vertexnode1->type == VERTEX_TYPE_JUNCTION) farr[offsetnode1] -= flow;
    if(vertexnode2->type == VERTEX_TYPE_JUNCTION) farr[offsetnode2] += flow;
  }

  /* Subtract Demand flows at the vertices */
  for(i=0; i < nv; i++) {
    ierr = DMNetworkIsGhostVertex(networkdm,v[i],&ghostvtex);CHKERRQ(ierr);
    if(ghostvtex) continue;

    ierr = DMNetworkGetVariableOffset(networkdm,v[i],&offset);CHKERRQ(ierr);
    ierr = DMNetworkGetComponent(networkdm,v[i],0,&key,(void**)&vertex);CHKERRQ(ierr);

    if(vertex->type == VERTEX_TYPE_JUNCTION) {
      farr[offset] -= vertex->junc.demand;
    } else if (vertex->type == VERTEX_TYPE_RESERVOIR) {
      reservoir = &vertex->res;
      farr[offset] = xarr[offset] - reservoir->head;
    } else if(vertex->type == VERTEX_TYPE_TANK) {
      tank = &vertex->tank;
      farr[offset] = xarr[offset] - (tank->elev + tank->initlvl);
    }
  }

  ierr = VecRestoreArrayRead(localX,&xarr);CHKERRQ(ierr);
  ierr = VecRestoreArray(localF,&farr);CHKERRQ(ierr);
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
  PetscInt       nv,ne,key,i,offset;
  const PetscInt *vtx,*edges;
  Vec            localX;
  PetscScalar    *xarr;
  VERTEXDATA     vertex;
  PetscBool      ghostvtex;

  PetscFunctionBegin;
  ierr = DMGetLocalVector(networkdm,&localX);CHKERRQ(ierr);

  ierr = VecSet(X,0.0);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(networkdm,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(networkdm,X,INSERT_VALUES,localX);CHKERRQ(ierr);

  /* Get subnetwork info */
  ierr = DMNetworkGetSubnetworkInfo(networkdm,0,&nv,&ne,&vtx,&edges);CHKERRQ(ierr);

  ierr = VecGetArray(localX,&xarr);CHKERRQ(ierr);

  for(i=0; i < nv; i++) {
    ierr = DMNetworkIsGhostVertex(networkdm,vtx[i],&ghostvtex);CHKERRQ(ierr);
    if(ghostvtex) continue;
    ierr = DMNetworkGetVariableOffset(networkdm,vtx[i],&offset);CHKERRQ(ierr);
    ierr = DMNetworkGetComponent(networkdm,vtx[i],0,&key,(void**)&vertex);CHKERRQ(ierr);

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
  PetscInt i,j,node1,node2;
  Pipe     *pipe;
  Pump     *pump;

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
  char             waternetdata_file[PETSC_MAX_PATH_LEN]="sample1.inp";
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
  PetscInt         ngvtx=PETSC_DETERMINE,ngedge=PETSC_DETERMINE;
  SNESConvergedReason reason;

  ierr = PetscInitialize(&argc,&argv,"waternetoptions",help);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&crank);CHKERRQ(ierr);

  /* Create an empty network object */
  ierr = DMNetworkCreate(PETSC_COMM_WORLD,&networkdm);CHKERRQ(ierr);

  /* Register the components in the network */
  ierr = DMNetworkRegisterComponent(networkdm,"edgestruct",sizeof(struct _p_EDGEDATA),&componentkey[0]);CHKERRQ(ierr);
  ierr = DMNetworkRegisterComponent(networkdm,"busstruct",sizeof(struct _p_VERTEXDATA),&componentkey[1]);CHKERRQ(ierr);

  ierr = PetscLogStageRegister("Read Data",&stage1);CHKERRQ(ierr);
  PetscLogStagePush(stage1);
  ierr = PetscNew(&waternetdata);CHKERRQ(ierr);

  /* READ THE DATA */
  if (!crank) {
    /* READ DATA. Only rank 0 reads the data */
    ierr = PetscOptionsGetString(NULL,NULL,"-waternetdata",waternetdata_file,PETSC_MAX_PATH_LEN-1,NULL);CHKERRQ(ierr);
    ierr = WaterNetReadData(waternetdata,waternetdata_file);CHKERRQ(ierr);

    ierr = PetscCalloc1(2*waternetdata->nedge,&edgelist);CHKERRQ(ierr);
    ierr = GetListofEdges(waternetdata,edgelist);CHKERRQ(ierr);
  }
  PetscLogStagePop();

  ierr = PetscLogStageRegister("Create network",&stage2);CHKERRQ(ierr);
  PetscLogStagePush(stage2);

  /* Set numbers of nodes and edges */
  ierr = DMNetworkSetSizes(networkdm,1,&waternetdata->nvertex,&waternetdata->nedge,&ngvtx,&ngedge);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"[%D] waternet nvertices %D, nedges %D\n",crank,waternetdata->nvertex,waternetdata->nedge);CHKERRQ(ierr);

  /* Add edge connectivity */
  ierr = DMNetworkSetEdgeList(networkdm,&edgelist);CHKERRQ(ierr);

  /* Set up the network layout */
  ierr = DMNetworkLayoutSetUp(networkdm);CHKERRQ(ierr);

  if (!crank) {
    ierr = PetscFree(edgelist);CHKERRQ(ierr);
  }

  /* ADD VARIABLES AND COMPONENTS FOR THE NETWORK */
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

  if (!crank) {
    ierr = PetscFree(waternetdata->vertex);CHKERRQ(ierr);
    ierr = PetscFree(waternetdata->edge);CHKERRQ(ierr);
  }
  ierr = PetscFree(waternetdata);CHKERRQ(ierr);

  /* Distribute networkdm to multiple processes */
  ierr = DMNetworkDistribute(&networkdm,0);CHKERRQ(ierr);

  PetscLogStagePop();

  ierr = DMCreateGlobalVector(networkdm,&X);CHKERRQ(ierr);
  ierr = VecDuplicate(X,&F);CHKERRQ(ierr);

  /* HOOK UP SOLVER */
  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
  ierr = SNESSetDM(snes,networkdm);CHKERRQ(ierr);
  ierr = SNESSetOptionsPrefix(snes,"waternet_");CHKERRQ(ierr);
  ierr = SNESSetFunction(snes,F,FormFunction,NULL);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  ierr = SetInitialGuess(networkdm,X);CHKERRQ(ierr);

  ierr = SNESSolve(snes,NULL,X);CHKERRQ(ierr);
  ierr = SNESGetConvergedReason(snes,&reason);CHKERRQ(ierr);
  if(reason < 0) {
    SETERRQ(PETSC_COMM_SELF,0,"No solution found for the water network");
  }
  ierr = VecView(X,0);CHKERRQ(ierr);

  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = VecDestroy(&F);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = DMDestroy(&networkdm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
