static char help[] = "This example demonstrates DMNetwork. It is used for testing parallel generation of dmnetwork, then redistribute. \n\\n";
/*
  Example: mpiexec -n <np> ./pipes -ts_max_steps 10
*/

#include "wash.h"

/*
  WashNetworkDistribute - proc[0] distributes sequential wash object
   Input Parameters:
.  comm - MPI communicator
.  wash - wash context with all network data in proc[0]

   Output Parameter:
.  wash - wash context with nedge, nvertex and edgelist distributed

   Note: The routine is used for testing parallel generation of dmnetwork, then redistribute.
*/
PetscErrorCode WashNetworkDistribute(MPI_Comm comm,Wash wash)
{
  PetscMPIInt    rank,size,tag=0;
  PetscInt       i,e,v,numEdges,numVertices,nedges,*eowners=NULL,estart,eend,*vtype=NULL,nvertices;
  PetscInt       *edgelist = wash->edgelist,*nvtx=NULL,*vtxDone=NULL;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_size(comm,&size));
  if (size == 1) PetscFunctionReturn(0);

  CHKERRMPI(MPI_Comm_rank(comm,&rank));
  numEdges    = wash->nedge;
  numVertices = wash->nvertex;

  /* (1) all processes get global and local number of edges */
  CHKERRMPI(MPI_Bcast(&numEdges,1,MPIU_INT,0,comm));
  nedges = numEdges/size; /* local nedges */
  if (rank == 0) {
    nedges += numEdges - size*(numEdges/size);
  }
  wash->Nedge = numEdges;
  wash->nedge = nedges;
  /* CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"[%d] nedges %d, numEdges %d\n",rank,nedges,numEdges)); */

  CHKERRQ(PetscCalloc3(size+1,&eowners,size,&nvtx,numVertices,&vtxDone));
  CHKERRMPI(MPI_Allgather(&nedges,1,MPIU_INT,eowners+1,1,MPIU_INT,PETSC_COMM_WORLD));
  eowners[0] = 0;
  for (i=2; i<=size; i++) {
    eowners[i] += eowners[i-1];
  }

  estart = eowners[rank];
  eend   = eowners[rank+1];
  /* CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"[%d] own lists row %d - %d\n",rank,estart,eend)); */

  /* (2) distribute row block edgelist to all processors */
  if (rank == 0) {
    vtype = wash->vtype;
    for (i=1; i<size; i++) {
      /* proc[0] sends edgelist to proc[i] */
      CHKERRMPI(MPI_Send(edgelist+2*eowners[i],2*(eowners[i+1]-eowners[i]),MPIU_INT,i,tag,comm));

      /* proc[0] sends vtype to proc[i] */
      CHKERRMPI(MPI_Send(vtype+2*eowners[i],2*(eowners[i+1]-eowners[i]),MPIU_INT,i,tag,comm));
    }
  } else {
    MPI_Status      status;
    CHKERRQ(PetscMalloc1(2*(eend-estart),&vtype));
    CHKERRQ(PetscMalloc1(2*(eend-estart),&edgelist));

    CHKERRMPI(MPI_Recv(edgelist,2*(eend-estart),MPIU_INT,0,tag,comm,&status));
    CHKERRMPI(MPI_Recv(vtype,2*(eend-estart),MPIU_INT,0,tag,comm,&status));
  }

  wash->edgelist = edgelist;

  /* (3) all processes get global and local number of vertices, without ghost vertices */
  if (rank == 0) {
    for (i=0; i<size; i++) {
      for (e=eowners[i]; e<eowners[i+1]; e++) {
        v = edgelist[2*e];
        if (!vtxDone[v]) {
          nvtx[i]++; vtxDone[v] = 1;
        }
        v = edgelist[2*e+1];
        if (!vtxDone[v]) {
          nvtx[i]++; vtxDone[v] = 1;
        }
      }
    }
  }
  CHKERRMPI(MPI_Bcast(&numVertices,1,MPIU_INT,0,PETSC_COMM_WORLD));
  CHKERRMPI(MPI_Scatter(nvtx,1,MPIU_INT,&nvertices,1,MPIU_INT,0,PETSC_COMM_WORLD));
  CHKERRQ(PetscFree3(eowners,nvtx,vtxDone));

  wash->Nvertex = numVertices;
  wash->nvertex = nvertices;
  wash->vtype   = vtype;
  PetscFunctionReturn(0);
}

PetscErrorCode WASHIFunction(TS ts,PetscReal t,Vec X,Vec Xdot,Vec F,void* ctx)
{
  Wash           wash=(Wash)ctx;
  DM             networkdm;
  Vec            localX,localXdot,localF, localXold;
  const PetscInt *cone;
  PetscInt       vfrom,vto,offsetfrom,offsetto,varoffset;
  PetscInt       v,vStart,vEnd,e,eStart,eEnd;
  PetscInt       nend,type;
  PetscBool      ghost;
  PetscScalar    *farr,*juncf, *pipef;
  PetscReal      dt;
  Pipe           pipe;
  PipeField      *pipex,*pipexdot,*juncx;
  Junction       junction;
  DMDALocalInfo  info;
  const PetscScalar *xarr,*xdotarr, *xoldarr;

  PetscFunctionBegin;
  localX    = wash->localX;
  localXdot = wash->localXdot;

  CHKERRQ(TSGetSolution(ts,&localXold));
  CHKERRQ(TSGetDM(ts,&networkdm));
  CHKERRQ(TSGetTimeStep(ts,&dt));
  CHKERRQ(DMGetLocalVector(networkdm,&localF));

  /* Set F and localF as zero */
  CHKERRQ(VecSet(F,0.0));
  CHKERRQ(VecSet(localF,0.0));

  /* Update ghost values of locaX and locaXdot */
  CHKERRQ(DMGlobalToLocalBegin(networkdm,X,INSERT_VALUES,localX));
  CHKERRQ(DMGlobalToLocalEnd(networkdm,X,INSERT_VALUES,localX));

  CHKERRQ(DMGlobalToLocalBegin(networkdm,Xdot,INSERT_VALUES,localXdot));
  CHKERRQ(DMGlobalToLocalEnd(networkdm,Xdot,INSERT_VALUES,localXdot));

  CHKERRQ(VecGetArrayRead(localX,&xarr));
  CHKERRQ(VecGetArrayRead(localXdot,&xdotarr));
  CHKERRQ(VecGetArrayRead(localXold,&xoldarr));
  CHKERRQ(VecGetArray(localF,&farr));

   /* junction->type == JUNCTION:
           juncf[0] = -qJ + sum(qin); juncf[1] = qJ - sum(qout)
       junction->type == RESERVOIR (upper stream):
           juncf[0] = -hJ + H0; juncf[1] = qJ - sum(qout)
       junction->type == VALVE (down stream):
           juncf[0] =  -qJ + sum(qin); juncf[1] = qJ
  */
  /* Vertex/junction initialization */
  CHKERRQ(DMNetworkGetVertexRange(networkdm,&vStart,&vEnd));
  for (v=vStart; v<vEnd; v++) {
    CHKERRQ(DMNetworkIsGhostVertex(networkdm,v,&ghost));
    if (ghost) continue;

    CHKERRQ(DMNetworkGetComponent(networkdm,v,0,&type,(void**)&junction,NULL));
    CHKERRQ(DMNetworkGetLocalVecOffset(networkdm,v,ALL_COMPONENTS,&varoffset));
    juncx      = (PipeField*)(xarr + varoffset);
    juncf      = (PetscScalar*)(farr + varoffset);

    juncf[0] = -juncx[0].q;
    juncf[1] =  juncx[0].q;

    if (junction->type == RESERVOIR) { /* upstream reservoir */
      juncf[0] = juncx[0].h - wash->H0;
    }
  }

  /* Edge/pipe */
  CHKERRQ(DMNetworkGetEdgeRange(networkdm,&eStart,&eEnd));
  for (e=eStart; e<eEnd; e++) {
    CHKERRQ(DMNetworkGetComponent(networkdm,e,0,&type,(void**)&pipe,NULL));
    CHKERRQ(DMNetworkGetLocalVecOffset(networkdm,e,ALL_COMPONENTS,&varoffset));
    pipex    = (PipeField*)(xarr + varoffset);
    pipexdot = (PipeField*)(xdotarr + varoffset);
    pipef    = (PetscScalar*)(farr + varoffset);

    /* Get some data into the pipe structure: note, some of these operations
     * might be redundant. Will it consume too much time? */
    pipe->dt   = dt;
    pipe->xold = (PipeField*)(xoldarr + varoffset);

    /* Evaluate F over this edge/pipe: pipef[1], ...,pipef[2*nend] */
    CHKERRQ(DMDAGetLocalInfo(pipe->da,&info));
    CHKERRQ(PipeIFunctionLocal_Lax(&info,t,pipex,pipexdot,pipef,pipe));

    /* Get boundary values from connected vertices */
    CHKERRQ(DMNetworkGetConnectedVertices(networkdm,e,&cone));
    vfrom = cone[0]; /* local ordering */
    vto   = cone[1];
    CHKERRQ(DMNetworkGetLocalVecOffset(networkdm,vfrom,ALL_COMPONENTS,&offsetfrom));
    CHKERRQ(DMNetworkGetLocalVecOffset(networkdm,vto,ALL_COMPONENTS,&offsetto));

    /* Evaluate upstream boundary */
    CHKERRQ(DMNetworkGetComponent(networkdm,vfrom,0,&type,(void**)&junction,NULL));
    PetscCheckFalse(junction->type != JUNCTION && junction->type != RESERVOIR,PETSC_COMM_SELF,PETSC_ERR_SUP,"junction type is not supported");
    juncx = (PipeField*)(xarr + offsetfrom);
    juncf = (PetscScalar*)(farr + offsetfrom);

    pipef[0] = pipex[0].h - juncx[0].h;
    juncf[1] -= pipex[0].q;

    /* Evaluate downstream boundary */
    CHKERRQ(DMNetworkGetComponent(networkdm,vto,0,&type,(void**)&junction,NULL));
    PetscCheckFalse(junction->type != JUNCTION && junction->type != VALVE,PETSC_COMM_SELF,PETSC_ERR_SUP,"junction type is not supported");
    juncx = (PipeField*)(xarr + offsetto);
    juncf = (PetscScalar*)(farr + offsetto);
    nend  = pipe->nnodes - 1;

    pipef[2*nend + 1] = pipex[nend].h - juncx[0].h;
    juncf[0] += pipex[nend].q;
  }

  CHKERRQ(VecRestoreArrayRead(localX,&xarr));
  CHKERRQ(VecRestoreArrayRead(localXdot,&xdotarr));
  CHKERRQ(VecRestoreArray(localF,&farr));

  CHKERRQ(DMLocalToGlobalBegin(networkdm,localF,ADD_VALUES,F));
  CHKERRQ(DMLocalToGlobalEnd(networkdm,localF,ADD_VALUES,F));
  CHKERRQ(DMRestoreLocalVector(networkdm,&localF));
  /*
   CHKERRQ(PetscPrintf(PETSC_COMM_WORLD("F:\n"));
   CHKERRQ(VecView(F,PETSC_VIEWER_STDOUT_WORLD));
   */
  PetscFunctionReturn(0);
}

PetscErrorCode WASHSetInitialSolution(DM networkdm,Vec X,Wash wash)
{
  PetscInt       k,nx,vkey,vfrom,vto,offsetfrom,offsetto;
  PetscInt       type,varoffset;
  PetscInt       e,eStart,eEnd;
  Vec            localX;
  PetscScalar    *xarr;
  Pipe           pipe;
  Junction       junction;
  const PetscInt *cone;
  const PetscScalar *xarray;

  PetscFunctionBegin;
  CHKERRQ(VecSet(X,0.0));
  CHKERRQ(DMGetLocalVector(networkdm,&localX));
  CHKERRQ(VecGetArray(localX,&xarr));

  /* Edge */
  CHKERRQ(DMNetworkGetEdgeRange(networkdm,&eStart,&eEnd));
  for (e=eStart; e<eEnd; e++) {
    CHKERRQ(DMNetworkGetLocalVecOffset(networkdm,e,ALL_COMPONENTS,&varoffset));
    CHKERRQ(DMNetworkGetComponent(networkdm,e,0,&type,(void**)&pipe,NULL));

    /* set initial values for this pipe */
    CHKERRQ(PipeComputeSteadyState(pipe,wash->Q0,wash->H0));
    CHKERRQ(VecGetSize(pipe->x,&nx));

    CHKERRQ(VecGetArrayRead(pipe->x,&xarray));
    /* copy pipe->x to xarray */
    for (k=0; k<nx; k++) {
      (xarr+varoffset)[k] = xarray[k];
    }

    /* set boundary values into vfrom and vto */
    CHKERRQ(DMNetworkGetConnectedVertices(networkdm,e,&cone));
    vfrom = cone[0]; /* local ordering */
    vto   = cone[1];
    CHKERRQ(DMNetworkGetLocalVecOffset(networkdm,vfrom,ALL_COMPONENTS,&offsetfrom));
    CHKERRQ(DMNetworkGetLocalVecOffset(networkdm,vto,ALL_COMPONENTS,&offsetto));

    /* if vform is a head vertex: */
    CHKERRQ(DMNetworkGetComponent(networkdm,vfrom,0,&vkey,(void**)&junction,NULL));
    if (junction->type == RESERVOIR) {
      (xarr+offsetfrom)[1] = wash->H0; /* 1st H */
    }

    /* if vto is an end vertex: */
    CHKERRQ(DMNetworkGetComponent(networkdm,vto,0,&vkey,(void**)&junction,NULL));
    if (junction->type == VALVE) {
      (xarr+offsetto)[0] = wash->QL; /* last Q */
    }
    CHKERRQ(VecRestoreArrayRead(pipe->x,&xarray));
  }

  CHKERRQ(VecRestoreArray(localX,&xarr));
  CHKERRQ(DMLocalToGlobalBegin(networkdm,localX,ADD_VALUES,X));
  CHKERRQ(DMLocalToGlobalEnd(networkdm,localX,ADD_VALUES,X));
  CHKERRQ(DMRestoreLocalVector(networkdm,&localX));

#if 0
  PetscInt N;
  CHKERRQ(VecGetSize(X,&N));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"initial solution %d:\n",N));
  CHKERRQ(VecView(X,PETSC_VIEWER_STDOUT_WORLD));
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode TSDMNetworkMonitor(TS ts, PetscInt step, PetscReal t, Vec x, void *context)
{
  DMNetworkMonitor   monitor;

  PetscFunctionBegin;
  monitor = (DMNetworkMonitor)context;
  CHKERRQ(DMNetworkMonitorView(monitor,x));
  PetscFunctionReturn(0);
}

PetscErrorCode PipesView(DM networkdm,PetscInt KeyPipe,Vec X)
{
  PetscInt       i,numkeys=1,*blocksize,*numselectedvariable,**selectedvariables,n;
  IS             isfrom_q,isfrom_h,isfrom;
  Vec            Xto;
  VecScatter     ctx;
  MPI_Comm       comm;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)networkdm,&comm));

  /* 1. Create isfrom_q for q-variable of pipes */
  CHKERRQ(PetscMalloc3(numkeys,&blocksize,numkeys,&numselectedvariable,numkeys,&selectedvariables));
  for (i=0; i<numkeys; i++) {
    blocksize[i]           = 2;
    numselectedvariable[i] = 1;
    CHKERRQ(PetscMalloc1(numselectedvariable[i],&selectedvariables[i]));
    selectedvariables[i][0] = 0; /* q-variable */
  }
  CHKERRQ(DMNetworkCreateIS(networkdm,numkeys,&KeyPipe,blocksize,numselectedvariable,selectedvariables,&isfrom_q));

  /* 2. Create Xto and isto */
  CHKERRQ(ISGetLocalSize(isfrom_q, &n));
  CHKERRQ(VecCreate(comm,&Xto));
  CHKERRQ(VecSetSizes(Xto,n,PETSC_DECIDE));
  CHKERRQ(VecSetFromOptions(Xto));
  CHKERRQ(VecSet(Xto,0.0));

  /* 3. Create scatter */
  CHKERRQ(VecScatterCreate(X,isfrom_q,Xto,NULL,&ctx));

  /* 4. Scatter to Xq */
  CHKERRQ(VecScatterBegin(ctx,X,Xto,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(ctx,X,Xto,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterDestroy(&ctx));
  CHKERRQ(ISDestroy(&isfrom_q));

  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Xq:\n"));
  CHKERRQ(VecView(Xto,PETSC_VIEWER_STDOUT_WORLD));

  /* 5. Create isfrom_h for h-variable of pipes; Create scatter; Scatter to Xh */
  for (i=0; i<numkeys; i++) {
    selectedvariables[i][0] = 1; /* h-variable */
  }
  CHKERRQ(DMNetworkCreateIS(networkdm,numkeys,&KeyPipe,blocksize,numselectedvariable,selectedvariables,&isfrom_h));

  CHKERRQ(VecScatterCreate(X,isfrom_h,Xto,NULL,&ctx));
  CHKERRQ(VecScatterBegin(ctx,X,Xto,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(ctx,X,Xto,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterDestroy(&ctx));
  CHKERRQ(ISDestroy(&isfrom_h));

  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Xh:\n"));
  CHKERRQ(VecView(Xto,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(VecDestroy(&Xto));

  /* 6. Create isfrom for all pipe variables; Create scatter; Scatter to Xpipes */
  for (i=0; i<numkeys; i++) {
    blocksize[i] = -1; /* select all the variables of the i-th component */
  }
  CHKERRQ(DMNetworkCreateIS(networkdm,numkeys,&KeyPipe,blocksize,NULL,NULL,&isfrom));
  CHKERRQ(ISDestroy(&isfrom));
  CHKERRQ(DMNetworkCreateIS(networkdm,numkeys,&KeyPipe,NULL,NULL,NULL,&isfrom));

  CHKERRQ(ISGetLocalSize(isfrom, &n));
  CHKERRQ(VecCreate(comm,&Xto));
  CHKERRQ(VecSetSizes(Xto,n,PETSC_DECIDE));
  CHKERRQ(VecSetFromOptions(Xto));
  CHKERRQ(VecSet(Xto,0.0));

  CHKERRQ(VecScatterCreate(X,isfrom,Xto,NULL,&ctx));
  CHKERRQ(VecScatterBegin(ctx,X,Xto,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(ctx,X,Xto,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterDestroy(&ctx));
  CHKERRQ(ISDestroy(&isfrom));

  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Xpipes:\n"));
  CHKERRQ(VecView(Xto,PETSC_VIEWER_STDOUT_WORLD));

  /* 7. Free spaces */
  for (i=0; i<numkeys; i++) {
    CHKERRQ(PetscFree(selectedvariables[i]));
  }
  CHKERRQ(PetscFree3(blocksize,numselectedvariable,selectedvariables));
  CHKERRQ(VecDestroy(&Xto));
  PetscFunctionReturn(0);
}

PetscErrorCode ISJunctionsView(DM networkdm,PetscInt KeyJunc)
{
  PetscInt       numkeys=1;
  IS             isfrom;
  MPI_Comm       comm;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)networkdm,&comm));
  CHKERRMPI(MPI_Comm_rank(comm,&rank));

  /* Create a global isfrom for all junction variables */
  CHKERRQ(DMNetworkCreateIS(networkdm,numkeys,&KeyJunc,NULL,NULL,NULL,&isfrom));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"ISJunctions:\n"));
  CHKERRQ(ISView(isfrom,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(ISDestroy(&isfrom));

  /* Create a local isfrom for all junction variables */
  CHKERRQ(DMNetworkCreateLocalIS(networkdm,numkeys,&KeyJunc,NULL,NULL,NULL,&isfrom));
  if (!rank) {
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"[%d] ISLocalJunctions:\n",rank));
    CHKERRQ(ISView(isfrom,PETSC_VIEWER_STDOUT_SELF));
  }
  CHKERRQ(ISDestroy(&isfrom));
  PetscFunctionReturn(0);
}

PetscErrorCode WashNetworkCleanUp(Wash wash)
{
  PetscMPIInt    rank;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_rank(wash->comm,&rank));
  CHKERRQ(PetscFree(wash->edgelist));
  CHKERRQ(PetscFree(wash->vtype));
  if (rank == 0) {
    CHKERRQ(PetscFree2(wash->junction,wash->pipe));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode WashNetworkCreate(MPI_Comm comm,PetscInt pipesCase,Wash *wash_ptr)
{
  PetscInt       npipes;
  PetscMPIInt    rank;
  Wash           wash=NULL;
  PetscInt       i,numVertices,numEdges,*vtype;
  PetscInt       *edgelist;
  Junction       junctions=NULL;
  Pipe           pipes=NULL;
  PetscBool      washdist=PETSC_TRUE;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_rank(comm,&rank));

  CHKERRQ(PetscCalloc1(1,&wash));
  wash->comm = comm;
  *wash_ptr  = wash;
  wash->Q0   = 0.477432; /* RESERVOIR */
  wash->H0   = 150.0;
  wash->HL   = 143.488;  /* VALVE */
  wash->QL   = 0.0;
  wash->nnodes_loc = 0;

  numVertices = 0;
  numEdges    = 0;
  edgelist    = NULL;

  /* proc[0] creates a sequential wash and edgelist */
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Setup pipesCase %D\n",pipesCase));

  /* Set global number of pipes, edges, and junctions */
  /*-------------------------------------------------*/
  switch (pipesCase) {
  case 0:
    /* pipeCase 0: */
    /* =================================================
    (RESERVOIR) v0 --E0--> v1--E1--> v2 --E2-->v3 (VALVE)
    ====================================================  */
    npipes = 3;
    CHKERRQ(PetscOptionsGetInt(NULL,NULL, "-npipes", &npipes, NULL));
    wash->nedge   = npipes;
    wash->nvertex = npipes + 1;

    /* Set local edges and vertices -- proc[0] sets entire network, then distributes */
    numVertices = 0;
    numEdges    = 0;
    edgelist    = NULL;
    if (rank == 0) {
      numVertices = wash->nvertex;
      numEdges    = wash->nedge;

      CHKERRQ(PetscCalloc1(2*numEdges,&edgelist));
      for (i=0; i<numEdges; i++) {
        edgelist[2*i] = i; edgelist[2*i+1] = i+1;
      }

      /* Add network components */
      /*------------------------*/
      CHKERRQ(PetscCalloc2(numVertices,&junctions,numEdges,&pipes));

      /* vertex */
      for (i=0; i<numVertices; i++) {
        junctions[i].id = i;
        junctions[i].type = JUNCTION;
      }

      junctions[0].type             = RESERVOIR;
      junctions[numVertices-1].type = VALVE;
    }
    break;
  case 1:
    /* pipeCase 1: */
    /* ==========================
                v2 (VALVE)
                ^
                |
               E2
                |
    v0 --E0--> v3--E1--> v1
  (RESERVOIR)            (RESERVOIR)
    =============================  */
    npipes = 3;
    wash->nedge   = npipes;
    wash->nvertex = npipes + 1;

    /* Set local edges and vertices -- proc[0] sets entire network, then distributes */
    if (rank == 0) {
      numVertices = wash->nvertex;
      numEdges    = wash->nedge;

      CHKERRQ(PetscCalloc1(2*numEdges,&edgelist));
      edgelist[0] = 0; edgelist[1] = 3;  /* edge[0] */
      edgelist[2] = 3; edgelist[3] = 1;  /* edge[1] */
      edgelist[4] = 3; edgelist[5] = 2;  /* edge[2] */

      /* Add network components */
      /*------------------------*/
      CHKERRQ(PetscCalloc2(numVertices,&junctions,numEdges,&pipes));
      /* vertex */
      for (i=0; i<numVertices; i++) {
        junctions[i].id   = i;
        junctions[i].type = JUNCTION;
      }

      junctions[0].type = RESERVOIR;
      junctions[1].type = VALVE;
      junctions[2].type = VALVE;
    }
    break;
  case 2:
    /* pipeCase 2: */
    /* ==========================
    (RESERVOIR)  v2--> E2
                       |
            v0 --E0--> v3--E1--> v1
    (RESERVOIR)               (VALVE)
    =============================  */

    /* Set application parameters -- to be used in function evalutions */
    npipes = 3;
    wash->nedge   = npipes;
    wash->nvertex = npipes + 1;

    /* Set local edges and vertices -- proc[0] sets entire network, then distributes */
    if (rank == 0) {
      numVertices = wash->nvertex;
      numEdges    = wash->nedge;

      CHKERRQ(PetscCalloc1(2*numEdges,&edgelist));
      edgelist[0] = 0; edgelist[1] = 3;  /* edge[0] */
      edgelist[2] = 3; edgelist[3] = 1;  /* edge[1] */
      edgelist[4] = 2; edgelist[5] = 3;  /* edge[2] */

      /* Add network components */
      /*------------------------*/
      CHKERRQ(PetscCalloc2(numVertices,&junctions,numEdges,&pipes));
      /* vertex */
      for (i=0; i<numVertices; i++) {
        junctions[i].id = i;
        junctions[i].type = JUNCTION;
      }

      junctions[0].type = RESERVOIR;
      junctions[1].type = VALVE;
      junctions[2].type = RESERVOIR;
    }
    break;
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"not done yet");
  }

  /* set edge global id */
  for (i=0; i<numEdges; i++) pipes[i].id = i;

  if (rank == 0) { /* set vtype for proc[0] */
    PetscInt v;
    CHKERRQ(PetscMalloc1(2*numEdges,&vtype));
    for (i=0; i<2*numEdges; i++) {
      v        = edgelist[i];
      vtype[i] = junctions[v].type;
    }
    wash->vtype = vtype;
  }

  *wash_ptr      = wash;
  wash->nedge    = numEdges;
  wash->nvertex  = numVertices;
  wash->edgelist = edgelist;
  wash->junction = junctions;
  wash->pipe     = pipes;

  /* Distribute edgelist to other processors */
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-wash_distribute",&washdist,NULL));
  if (washdist) {
    /*
     CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Distribute sequential wash ...\n"));
     */
    CHKERRQ(WashNetworkDistribute(comm,wash));
  }
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------- */
int main(int argc,char ** argv)
{
  PetscErrorCode    ierr;
  Wash              wash;
  Junction          junctions,junction;
  Pipe              pipe,pipes;
  PetscInt          KeyPipe,KeyJunction,*edgelist = NULL,*vtype = NULL;
  PetscInt          i,e,v,eStart,eEnd,vStart,vEnd,key,vkey,type;
  PetscInt          steps=1,nedges,nnodes=6;
  const PetscInt    *cone;
  DM                networkdm;
  PetscMPIInt       size,rank;
  PetscReal         ftime;
  Vec               X;
  TS                ts;
  TSConvergedReason reason;
  PetscBool         viewpipes,viewjuncs,monipipes=PETSC_FALSE,userJac=PETSC_TRUE,viewdm=PETSC_FALSE,viewX=PETSC_FALSE;
  PetscInt          pipesCase=0;
  DMNetworkMonitor  monitor;
  MPI_Comm          comm;

  ierr = PetscInitialize(&argc,&argv,"pOption",help);if (ierr) return ierr;

  /* Read runtime options */
  CHKERRQ(PetscOptionsGetInt(NULL,NULL, "-case", &pipesCase, NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-user_Jac",&userJac,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-pipe_monitor",&monipipes,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-viewdm",&viewdm,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-viewX",&viewX,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL, "-npipenodes", &nnodes, NULL));

  /* Create networkdm */
  /*------------------*/
  CHKERRQ(DMNetworkCreate(PETSC_COMM_WORLD,&networkdm));
  CHKERRQ(PetscObjectGetComm((PetscObject)networkdm,&comm));
  CHKERRMPI(MPI_Comm_rank(comm,&rank));
  CHKERRMPI(MPI_Comm_size(comm,&size));

  if (size == 1 && monipipes) {
    CHKERRQ(DMNetworkMonitorCreate(networkdm,&monitor));
  }

  /* Register the components in the network */
  CHKERRQ(DMNetworkRegisterComponent(networkdm,"junctionstruct",sizeof(struct _p_Junction),&KeyJunction));
  CHKERRQ(DMNetworkRegisterComponent(networkdm,"pipestruct",sizeof(struct _p_Pipe),&KeyPipe));

  /* Create a distributed wash network (user-specific) */
  CHKERRQ(WashNetworkCreate(comm,pipesCase,&wash));
  nedges      = wash->nedge;
  edgelist    = wash->edgelist;
  vtype       = wash->vtype;
  junctions   = wash->junction;
  pipes       = wash->pipe;

  /* Set up the network layout */
  CHKERRQ(DMNetworkSetNumSubNetworks(networkdm,PETSC_DECIDE,1));
  CHKERRQ(DMNetworkAddSubnetwork(networkdm,NULL,nedges,edgelist,NULL));

  CHKERRQ(DMNetworkLayoutSetUp(networkdm));

  CHKERRQ(DMNetworkGetEdgeRange(networkdm,&eStart,&eEnd));
  CHKERRQ(DMNetworkGetVertexRange(networkdm,&vStart,&vEnd));
  /* CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"[%d] eStart/End: %d - %d; vStart/End: %d - %d\n",rank,eStart,eEnd,vStart,vEnd)); */

  if (rank) { /* junctions[] and pipes[] for proc[0] are allocated in WashNetworkCreate() */
    /* vEnd - vStart = nvertices + number of ghost vertices! */
    CHKERRQ(PetscCalloc2(vEnd - vStart,&junctions,nedges,&pipes));
  }

  /* Add Pipe component and number of variables to all local edges */
  for (e = eStart; e < eEnd; e++) {
    pipes[e-eStart].nnodes = nnodes;
    CHKERRQ(DMNetworkAddComponent(networkdm,e,KeyPipe,&pipes[e-eStart],2*pipes[e-eStart].nnodes));

    if (size == 1 && monipipes) { /* Add monitor -- show Q_{pipes[e-eStart].id}? */
      pipes[e-eStart].length = 600.0;
      CHKERRQ(DMNetworkMonitorAdd(monitor, "Pipe Q", e, pipes[e-eStart].nnodes, 0, 2, 0.0,pipes[e-eStart].length, -0.8, 0.8, PETSC_TRUE));
      CHKERRQ(DMNetworkMonitorAdd(monitor, "Pipe H", e, pipes[e-eStart].nnodes, 1, 2, 0.0,pipes[e-eStart].length, -400.0, 800.0, PETSC_TRUE));
    }
  }

  /* Add Junction component and number of variables to all local vertices */
  for (v = vStart; v < vEnd; v++) {
    CHKERRQ(DMNetworkAddComponent(networkdm,v,KeyJunction,&junctions[v-vStart],2));
  }

  if (size > 1) {  /* must be called before DMSetUp()???. Other partitioners do not work yet??? -- cause crash in proc[0]! */
    DM               plexdm;
    PetscPartitioner part;
    CHKERRQ(DMNetworkGetPlex(networkdm,&plexdm));
    CHKERRQ(DMPlexGetPartitioner(plexdm, &part));
    CHKERRQ(PetscPartitionerSetType(part,PETSCPARTITIONERSIMPLE));
    CHKERRQ(PetscOptionsSetValue(NULL,"-dm_plex_csr_alg","mat")); /* for parmetis */
  }

  /* Set up DM for use */
  CHKERRQ(DMSetUp(networkdm));
  if (viewdm) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nOriginal networkdm, DMView:\n"));
    CHKERRQ(DMView(networkdm,PETSC_VIEWER_STDOUT_WORLD));
  }

  /* Set user physical parameters to the components */
  for (e = eStart; e < eEnd; e++) {
    CHKERRQ(DMNetworkGetConnectedVertices(networkdm,e,&cone));
    /* vfrom */
    CHKERRQ(DMNetworkGetComponent(networkdm,cone[0],0,&vkey,(void**)&junction,NULL));
    junction->type = (VertexType)vtype[2*e];

    /* vto */
    CHKERRQ(DMNetworkGetComponent(networkdm,cone[1],0,&vkey,(void**)&junction,NULL));
    junction->type = (VertexType)vtype[2*e+1];
  }

  CHKERRQ(WashNetworkCleanUp(wash));

  /* Network partitioning and distribution of data */
  CHKERRQ(DMNetworkDistribute(&networkdm,0));
  if (viewdm) {
    PetscPrintf(PETSC_COMM_WORLD,"\nAfter DMNetworkDistribute, DMView:\n");CHKERRQ(ierr);
    CHKERRQ(DMView(networkdm,PETSC_VIEWER_STDOUT_WORLD));
  }

  /* create vectors */
  CHKERRQ(DMCreateGlobalVector(networkdm,&X));
  CHKERRQ(DMCreateLocalVector(networkdm,&wash->localX));
  CHKERRQ(DMCreateLocalVector(networkdm,&wash->localXdot));

  /* PipeSetUp -- each process only sets its own pipes */
  /*---------------------------------------------------*/
  CHKERRQ(DMNetworkGetVertexRange(networkdm,&vStart,&vEnd));

  userJac = PETSC_TRUE;
  CHKERRQ(DMNetworkHasJacobian(networkdm,userJac,userJac));
  CHKERRQ(DMNetworkGetEdgeRange(networkdm,&eStart,&eEnd));
  for (e=eStart; e<eEnd; e++) { /* each edge has only one component, pipe */
    CHKERRQ(DMNetworkGetComponent(networkdm,e,0,&type,(void**)&pipe,NULL));

    wash->nnodes_loc += pipe->nnodes; /* local total number of nodes, will be used by PipesView() */
    ierr = PipeSetParameters(pipe,
                             600.0,          /* length */
                             0.5,            /* diameter */
                             1200.0,         /* a */
                             0.018);CHKERRQ(ierr);    /* friction */
    CHKERRQ(PipeSetUp(pipe));

    if (userJac) {
      /* Create Jacobian matrix structures for a Pipe */
      Mat            *J;
      CHKERRQ(PipeCreateJacobian(pipe,NULL,&J));
      CHKERRQ(DMNetworkEdgeSetMatrix(networkdm,e,J));
    }
  }

  if (userJac) {
    CHKERRQ(DMNetworkGetVertexRange(networkdm,&vStart,&vEnd));
    for (v=vStart; v<vEnd; v++) {
      Mat            *J;
      CHKERRQ(JunctionCreateJacobian(networkdm,v,NULL,&J));
      CHKERRQ(DMNetworkVertexSetMatrix(networkdm,v,J));

      CHKERRQ(DMNetworkGetComponent(networkdm,v,0,&vkey,(void**)&junction,NULL));
      junction->jacobian = J;
    }
  }

  /* Setup solver                                           */
  /*--------------------------------------------------------*/
  CHKERRQ(TSCreate(PETSC_COMM_WORLD,&ts));

  CHKERRQ(TSSetDM(ts,(DM)networkdm));
  CHKERRQ(TSSetIFunction(ts,NULL,WASHIFunction,wash));

  CHKERRQ(TSSetMaxSteps(ts,steps));
  CHKERRQ(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER));
  CHKERRQ(TSSetTimeStep(ts,0.1));
  CHKERRQ(TSSetType(ts,TSBEULER));
  if (size == 1 && monipipes) {
    CHKERRQ(TSMonitorSet(ts, TSDMNetworkMonitor, monitor, NULL));
  }
  CHKERRQ(TSSetFromOptions(ts));

  CHKERRQ(WASHSetInitialSolution(networkdm,X,wash));

  CHKERRQ(TSSolve(ts,X));

  CHKERRQ(TSGetSolveTime(ts,&ftime));
  CHKERRQ(TSGetStepNumber(ts,&steps));
  CHKERRQ(TSGetConvergedReason(ts,&reason));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"%s at time %g after %D steps\n",TSConvergedReasons[reason],(double)ftime,steps));
  if (viewX) {
    CHKERRQ(VecView(X,PETSC_VIEWER_STDOUT_WORLD));
  }

  viewpipes = PETSC_FALSE;
  CHKERRQ(PetscOptionsGetBool(NULL,NULL, "-Jac_view", &viewpipes,NULL));
  if (viewpipes) {
    SNES snes;
    Mat  Jac;
    CHKERRQ(TSGetSNES(ts,&snes));
    CHKERRQ(SNESGetJacobian(snes,&Jac,NULL,NULL,NULL));
    CHKERRQ(MatView(Jac,PETSC_VIEWER_DRAW_WORLD));
  }

  /* View solutions */
  /* -------------- */
  viewpipes = PETSC_FALSE;
  CHKERRQ(PetscOptionsGetBool(NULL,NULL, "-pipe_view", &viewpipes,NULL));
  if (viewpipes) {
    CHKERRQ(PipesView(networkdm,KeyPipe,X));
  }

  /* Test IS */
  viewjuncs = PETSC_FALSE;
  CHKERRQ(PetscOptionsGetBool(NULL,NULL, "-isjunc_view", &viewjuncs,NULL));
  if (viewjuncs) {
    CHKERRQ(ISJunctionsView(networkdm,KeyJunction));
  }

  /* Free spaces */
  /* ----------- */
  CHKERRQ(TSDestroy(&ts));
  CHKERRQ(VecDestroy(&X));
  CHKERRQ(VecDestroy(&wash->localX));
  CHKERRQ(VecDestroy(&wash->localXdot));

  /* Destroy objects from each pipe that are created in PipeSetUp() */
  CHKERRQ(DMNetworkGetEdgeRange(networkdm,&eStart, &eEnd));
  for (i = eStart; i < eEnd; i++) {
    CHKERRQ(DMNetworkGetComponent(networkdm,i,0,&key,(void**)&pipe,NULL));
    CHKERRQ(PipeDestroy(&pipe));
  }
  if (userJac) {
    for (v=vStart; v<vEnd; v++) {
      CHKERRQ(DMNetworkGetComponent(networkdm,v,0,&vkey,(void**)&junction,NULL));
      CHKERRQ(JunctionDestroyJacobian(networkdm,v,junction));
    }
  }

  if (size == 1 && monipipes) {
    CHKERRQ(DMNetworkMonitorDestroy(&monitor));
  }
  CHKERRQ(DMDestroy(&networkdm));
  CHKERRQ(PetscFree(wash));

  if (rank) {
    CHKERRQ(PetscFree2(junctions,pipes));
  }
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
     depends: pipeInterface.c pipeImpls.c
     requires: mumps

   test:
      args: -ts_monitor -case 1 -ts_max_steps 1 -pc_factor_mat_solver_type mumps -options_left no -viewX
      localrunfiles: pOption
      output_file: output/pipes_1.out

   test:
      suffix: 2
      nsize: 2
      args: -ts_monitor -case 1 -ts_max_steps 1 -pc_factor_mat_solver_type mumps -petscpartitioner_type simple -options_left no -viewX
      localrunfiles: pOption
      output_file: output/pipes_2.out

   test:
      suffix: 3
      nsize: 2
      args: -ts_monitor -case 0 -ts_max_steps 1 -pc_factor_mat_solver_type mumps -petscpartitioner_type simple -options_left no -viewX
      localrunfiles: pOption
      output_file: output/pipes_3.out

   test:
      suffix: 4
      args: -ts_monitor -case 2 -ts_max_steps 1 -pc_factor_mat_solver_type mumps -options_left no -viewX
      localrunfiles: pOption
      output_file: output/pipes_4.out

   test:
      suffix: 5
      nsize: 3
      args: -ts_monitor -case 2 -ts_max_steps 10 -pc_factor_mat_solver_type mumps -petscpartitioner_type simple -options_left no -viewX
      localrunfiles: pOption
      output_file: output/pipes_5.out

   test:
      suffix: 6
      nsize: 2
      args: -ts_monitor -case 1 -ts_max_steps 1 -pc_factor_mat_solver_type mumps -petscpartitioner_type simple -options_left no -wash_distribute 0 -viewX
      localrunfiles: pOption
      output_file: output/pipes_6.out

   test:
      suffix: 7
      nsize: 2
      args: -ts_monitor -case 2 -ts_max_steps 1 -pc_factor_mat_solver_type mumps -petscpartitioner_type simple -options_left no -wash_distribute 0 -viewX
      localrunfiles: pOption
      output_file: output/pipes_7.out

   test:
      suffix: 8
      nsize: 2
      requires: parmetis
      args: -ts_monitor -case 2 -ts_max_steps 1 -pc_factor_mat_solver_type mumps -petscpartitioner_type parmetis -options_left no -wash_distribute 1
      localrunfiles: pOption
      output_file: output/pipes_8.out

   test:
      suffix: 9
      nsize: 2
      args: -case 0 -ts_max_steps 1 -pc_factor_mat_solver_type mumps -petscpartitioner_type simple -options_left no -wash_distribute 0 -pipe_view
      localrunfiles: pOption
      output_file: output/pipes_9.out

   test:
      suffix: 10
      nsize: 2
      args: -case 0 -ts_max_steps 1 -pc_factor_mat_solver_type mumps -petscpartitioner_type simple -options_left no -wash_distribute 0 -isjunc_view
      localrunfiles: pOption
      output_file: output/pipes_10.out

TEST*/
