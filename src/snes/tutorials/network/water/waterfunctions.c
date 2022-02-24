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
  const PetscScalar *xarr;
  const PetscInt    *cone;
  PetscScalar       *farr,hf,ht,flow;
  PetscInt          i,key,vnode1,vnode2,offsetnode1,offsetnode2,offset,ncomp;
  PetscBool         ghostvtex;
  VERTEX_Water      vertex,vertexnode1,vertexnode2;
  EDGE_Water        edge;
  Pipe              *pipe;
  Pump              *pump;
  Reservoir         *reservoir;
  Tank              *tank;

  PetscFunctionBegin;
  /* Get arrays for the vectors */
  CHKERRQ(VecGetArrayRead(localX,&xarr));
  CHKERRQ(VecGetArray(localF,&farr));

  for (i=0; i<ne; i++) { /* for each edge */
    /* Get the offset and the key for the component for edge number e[i] */
    CHKERRQ(DMNetworkGetComponent(networkdm,edges[i],0,&key,(void**)&edge,NULL));

    /* Get the numbers for the vertices covering this edge */
    CHKERRQ(DMNetworkGetConnectedVertices(networkdm,edges[i],&cone));
    vnode1 = cone[0];
    vnode2 = cone[1];

    /* Get the components at the two vertices, their variable offsets */
    CHKERRQ(DMNetworkGetNumComponents(networkdm,vnode1,&ncomp));
    CHKERRQ(DMNetworkGetComponent(networkdm,vnode1,ncomp-1,&key,(void**)&vertexnode1,NULL));
    CHKERRQ(DMNetworkGetLocalVecOffset(networkdm,vnode1,ncomp-1,&offsetnode1));

    CHKERRQ(DMNetworkGetNumComponents(networkdm,vnode2,&ncomp));
    CHKERRQ(DMNetworkGetComponent(networkdm,vnode2,ncomp-1,&key,(void**)&vertexnode2,NULL));
    CHKERRQ(DMNetworkGetLocalVecOffset(networkdm,vnode2,ncomp-1,&offsetnode2));

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
    CHKERRQ(DMNetworkIsGhostVertex(networkdm,vtx[i],&ghostvtex));
    if (ghostvtex) continue;

    CHKERRQ(DMNetworkGetLocalVecOffset(networkdm,vtx[i],ALL_COMPONENTS,&offset));
    CHKERRQ(DMNetworkGetNumComponents(networkdm,vtx[i],&ncomp));
    CHKERRQ(DMNetworkGetComponent(networkdm,vtx[i],ncomp-1,&key,(void**)&vertex,NULL));

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

  CHKERRQ(VecRestoreArrayRead(localX,&xarr));
  CHKERRQ(VecRestoreArray(localF,&farr));
  PetscFunctionReturn(0);
}

PetscErrorCode WaterFormFunction(SNES snes,Vec X, Vec F, void *user)
{
  DM             networkdm;
  Vec            localX,localF;
  const PetscInt *v,*e;
  PetscInt       nv,ne;

  PetscFunctionBegin;
  /* Get the DM attached with the SNES */
  CHKERRQ(SNESGetDM(snes,&networkdm));

  /* Get local vertices and edges */
  CHKERRQ(DMNetworkGetSubnetwork(networkdm,0,&nv,&ne,&v,&e));

  /* Get local vectors */
  CHKERRQ(DMGetLocalVector(networkdm,&localX));
  CHKERRQ(DMGetLocalVector(networkdm,&localF));

  /* Scatter values from global to local vector */
  CHKERRQ(DMGlobalToLocalBegin(networkdm,X,INSERT_VALUES,localX));
  CHKERRQ(DMGlobalToLocalEnd(networkdm,X,INSERT_VALUES,localX));

  /* Initialize residual */
  CHKERRQ(VecSet(F,0.0));
  CHKERRQ(VecSet(localF,0.0));

  CHKERRQ(FormFunction_Water(networkdm,localX,localF,nv,ne,v,e,NULL));

  CHKERRQ(DMRestoreLocalVector(networkdm,&localX));
  CHKERRQ(DMLocalToGlobalBegin(networkdm,localF,ADD_VALUES,F));
  CHKERRQ(DMLocalToGlobalEnd(networkdm,localF,ADD_VALUES,F));

  CHKERRQ(DMRestoreLocalVector(networkdm,&localF));
  PetscFunctionReturn(0);
}

PetscErrorCode WaterSetInitialGuess(DM networkdm,Vec X)
{
  PetscInt       nv,ne;
  const PetscInt *vtx,*edges;
  Vec            localX;

  PetscFunctionBegin;
  CHKERRQ(DMGetLocalVector(networkdm,&localX));

  CHKERRQ(VecSet(X,0.0));
  CHKERRQ(DMGlobalToLocalBegin(networkdm,X,INSERT_VALUES,localX));
  CHKERRQ(DMGlobalToLocalEnd(networkdm,X,INSERT_VALUES,localX));

  /* Get subnetwork */
  CHKERRQ(DMNetworkGetSubnetwork(networkdm,0,&nv,&ne,&vtx,&edges));
  CHKERRQ(SetInitialGuess_Water(networkdm,localX,nv,ne,vtx,edges,NULL));

  CHKERRQ(DMLocalToGlobalBegin(networkdm,localX,ADD_VALUES,X));
  CHKERRQ(DMLocalToGlobalEnd(networkdm,localX,ADD_VALUES,X));
  CHKERRQ(DMRestoreLocalVector(networkdm,&localX));
  PetscFunctionReturn(0);
}

PetscErrorCode GetListofEdges_Water(WATERDATA *water,PetscInt *edgelist)
{
  PetscInt       i,j,node1,node2;
  Pipe           *pipe;
  Pump           *pump;
  PetscBool      netview=PETSC_FALSE;

  PetscFunctionBegin;
  CHKERRQ(PetscOptionsHasName(NULL,NULL, "-water_view",&netview));
  for (i=0; i < water->nedge; i++) {
    if (water->edge[i].type == EDGE_TYPE_PIPE) {
      pipe  = &water->edge[i].pipe;
      node1 = pipe->node1;
      node2 = pipe->node2;
      if (netview) {
        CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"edge %d, pipe v[%d] -> v[%d]\n",i,node1,node2));
      }
    } else {
      pump  = &water->edge[i].pump;
      node1 = pump->node1;
      node2 = pump->node2;
      if (netview) {
        CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"edge %d, pump v[%d] -> v[%d]\n",i,node1,node2));
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
  PetscInt       i,offset,key;
  PetscBool      ghostvtex,sharedv;
  VERTEX_Water   vertex;
  PetscScalar    *xarr;

  PetscFunctionBegin;
  CHKERRQ(VecGetArray(localX,&xarr));
  for (i=0; i < nv; i++) {
    CHKERRQ(DMNetworkIsGhostVertex(networkdm,vtx[i],&ghostvtex));
    CHKERRQ(DMNetworkIsSharedVertex(networkdm,vtx[i],&sharedv));
    if (ghostvtex || sharedv) continue;

    CHKERRQ(DMNetworkGetComponent(networkdm,vtx[i],0,&key,(void**)&vertex,NULL));
    CHKERRQ(DMNetworkGetLocalVecOffset(networkdm,vtx[i],0,&offset));
    if (vertex->type == VERTEX_TYPE_JUNCTION) {
      xarr[offset] = 100;
    } else if (vertex->type == VERTEX_TYPE_RESERVOIR) {
      xarr[offset] = vertex->res.head;
    } else {
      xarr[offset] = vertex->tank.initlvl + vertex->tank.elev;
    }
  }
  CHKERRQ(VecRestoreArray(localX,&xarr));
  PetscFunctionReturn(0);
}
