/* function subroutines used by water.c */

#include "water.h"
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

PetscErrorCode FormFunction_Water(DM networkdm,Vec localX,Vec localF,PetscInt nv,PetscInt ne,const PetscInt* vtx,const PetscInt* edges,void* appctx)
{
  PetscErrorCode    ierr;
  const PetscScalar *xarr;
  const PetscInt    *cone;
  PetscScalar       *farr,hf,ht,flow;
  PetscInt          i,key,vnode1,vnode2,offsetnode1,offsetnode2,offset;
  PetscBool         ghostvtex;
  VERTEX_Water      vertex,vertexnode1,vertexnode2;
  EDGE_Water        edge;
  Pipe              *pipe;
  Pump              *pump;
  Reservoir         *reservoir;
  Tank              *tank;

  PetscFunctionBegin;
  /* Get arrays for the vectors */
  ierr = VecGetArrayRead(localX,&xarr);CHKERRQ(ierr);
  ierr = VecGetArray(localF,&farr);CHKERRQ(ierr);

  for (i=0; i<ne; i++) { /* for each edge */
    /* Get the offset and the key for the component for edge number e[i] */
    ierr = DMNetworkGetComponent(networkdm,edges[i],0,&key,(void**)&edge);CHKERRQ(ierr);

    /* Get the numbers for the vertices covering this edge */
    ierr = DMNetworkGetConnectedVertices(networkdm,edges[i],&cone);CHKERRQ(ierr);
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

    flow = 0.0;
    if (edge->type == EDGE_TYPE_PIPE) {
      pipe = &edge->pipe;
      flow = Flow_Pipe(pipe,hf,ht);
    } else if (edge->type == EDGE_TYPE_PUMP) {
      pump = &edge->pump;
      flow = Flow_Pump(pump,hf,ht);
    }
    /* Convention: Node 1 has outgoing flow and Node 2 has incoming flow */
    if (vertexnode1->type == VERTEX_TYPE_JUNCTION) farr[offsetnode1] -= flow;
    if (vertexnode2->type == VERTEX_TYPE_JUNCTION) farr[offsetnode2] += flow;
  }

  /* Subtract Demand flows at the vertices */
  for (i=0; i<nv; i++) {
    ierr = DMNetworkIsGhostVertex(networkdm,vtx[i],&ghostvtex);CHKERRQ(ierr);
    if (ghostvtex) continue;

    ierr = DMNetworkGetVariableOffset(networkdm,vtx[i],&offset);CHKERRQ(ierr);
    ierr = DMNetworkGetComponent(networkdm,vtx[i],0,&key,(void**)&vertex);CHKERRQ(ierr);

    if (vertex->type == VERTEX_TYPE_JUNCTION) {
      farr[offset] -= vertex->junc.demand;
    } else if (vertex->type == VERTEX_TYPE_RESERVOIR) {
      reservoir = &vertex->res;
      farr[offset] = xarr[offset] - reservoir->head;
    } else if (vertex->type == VERTEX_TYPE_TANK) {
      tank = &vertex->tank;
      farr[offset] = xarr[offset] - (tank->elev + tank->initlvl);
    }
  }

  ierr = VecRestoreArrayRead(localX,&xarr);CHKERRQ(ierr);
  ierr = VecRestoreArray(localF,&farr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode WaterFormFunction(SNES snes,Vec X, Vec F, void *user)
{
  PetscErrorCode ierr;
  DM             networkdm;
  Vec            localX,localF;
  const PetscInt *v,*e;
  PetscInt       nv,ne;

  PetscFunctionBegin;
  /* Get the DM attached with the SNES */
  ierr = SNESGetDM(snes,&networkdm);CHKERRQ(ierr);

  /* Get local vertices and edges */
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

  ierr = FormFunction_Water(networkdm,localX,localF,nv,ne,v,e,NULL);CHKERRQ(ierr);

  ierr = DMRestoreLocalVector(networkdm,&localX);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(networkdm,localF,ADD_VALUES,F);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(networkdm,localF,ADD_VALUES,F);CHKERRQ(ierr);

  ierr = DMRestoreLocalVector(networkdm,&localF);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode WaterSetInitialGuess(DM networkdm,Vec X)
{
  PetscErrorCode ierr;
  PetscInt       nv,ne;
  const PetscInt *vtx,*edges;
  Vec            localX;

  PetscFunctionBegin;
  ierr = DMGetLocalVector(networkdm,&localX);CHKERRQ(ierr);

  ierr = VecSet(X,0.0);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(networkdm,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(networkdm,X,INSERT_VALUES,localX);CHKERRQ(ierr);

  /* Get subnetwork info */
  ierr = DMNetworkGetSubnetworkInfo(networkdm,0,&nv,&ne,&vtx,&edges);CHKERRQ(ierr);
  ierr = SetInitialGuess_Water(networkdm,localX,nv,ne,vtx,edges,NULL);CHKERRQ(ierr);

  ierr = DMLocalToGlobalBegin(networkdm,localX,ADD_VALUES,X);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(networkdm,localX,ADD_VALUES,X);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(networkdm,&localX);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode GetListofEdges_Water(WATERDATA *water,PetscInt *edgelist)
{
  PetscErrorCode ierr;
  PetscInt       i,j,node1,node2;
  Pipe           *pipe;
  Pump           *pump;
  PetscBool      netview=PETSC_FALSE;

  PetscFunctionBegin;
  ierr = PetscOptionsHasName(NULL,NULL, "-water_view",&netview);CHKERRQ(ierr);
  for (i=0; i < water->nedge; i++) {
    if (water->edge[i].type == EDGE_TYPE_PIPE) {
      pipe  = &water->edge[i].pipe;
      node1 = pipe->node1;
      node2 = pipe->node2;
      if (netview) {
        ierr = PetscPrintf(PETSC_COMM_SELF,"edge %d, pipe v[%d] -> v[%d]\n",i,node1,node2);CHKERRQ(ierr);
      }
    } else {
      pump  = &water->edge[i].pump;
      node1 = pump->node1;
      node2 = pump->node2;
      if (netview) {
        ierr = PetscPrintf(PETSC_COMM_SELF,"edge %d, pump v[%d] -> v[%d]\n",i,node1,node2);CHKERRQ(ierr);
      }
    }

    for (j=0; j < water->nvertex; j++) {
      if (water->vertex[j].id == node1) {
        edgelist[2*i] = j;
        break;
      }
    }

    for (j=0; j < water->nvertex; j++) {
      if (water->vertex[j].id == node2) {
        edgelist[2*i+1] = j;
        break;
      }
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode SetInitialGuess_Water(DM networkdm,Vec localX,PetscInt nv,PetscInt ne, const PetscInt *vtx, const PetscInt *edges,void* appctx)
{
  PetscErrorCode ierr;
  PetscInt       i,offset,key;
  PetscBool      ghostvtex;
  VERTEX_Water   vertex;
  PetscScalar    *xarr;

  PetscFunctionBegin;
  ierr = VecGetArray(localX,&xarr);CHKERRQ(ierr);
  for (i=0; i < nv; i++) {
    ierr = DMNetworkIsGhostVertex(networkdm,vtx[i],&ghostvtex);CHKERRQ(ierr);
    if (ghostvtex) continue;
    ierr = DMNetworkGetVariableOffset(networkdm,vtx[i],&offset);CHKERRQ(ierr);
    ierr = DMNetworkGetComponent(networkdm,vtx[i],0,&key,(void**)&vertex);CHKERRQ(ierr);

    if (vertex->type == VERTEX_TYPE_JUNCTION) {
      xarr[offset] = 100;
    } else if (vertex->type == VERTEX_TYPE_RESERVOIR) {
      xarr[offset] = vertex->res.head;
    } else {
      xarr[offset] = vertex->tank.initlvl + vertex->tank.elev;
    }
  }
  ierr = VecRestoreArray(localX,&xarr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
