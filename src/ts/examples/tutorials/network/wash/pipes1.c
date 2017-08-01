     static char help[] = "This example demonstrates the use of DMNetwork \n\\n";

/*
  Example: mpiexec -n <np> ./pipes1 -ts_max_steps 10
*/

#include "wash.h"
#include <petscdmnetwork.h>

PetscErrorCode WASHIFunction(TS ts,PetscReal t,Vec X,Vec Xdot,Vec F,void* ctx)
{
  PetscErrorCode ierr;
  Wash           wash=(Wash)ctx;
  DM             networkdm;
  Vec            localX,localXdot,localF;
  const PetscInt *cone;
  PetscInt       vfrom,vto,offsetfrom,offsetto,type,varoffset,voffset;
  PetscInt       v,vStart,vEnd,e,eStart,eEnd,pipeoffset;
  PetscBool      ghost;
  PetscScalar    *farr,*vf,*juncx,*juncf;
  Pipe           pipe;
  PipeField      *pipex,*pipexdot,*pipef;
  DMDALocalInfo  info;
  Junction       junction;
  MPI_Comm       comm;
  PetscMPIInt    rank,size;
  const PetscScalar *xarr,*xdotarr;
  DMNetworkComponentGenericDataType *nwarr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)ts,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);

  ierr = VecSet(F,0.0);CHKERRQ(ierr);

  localX    = wash->localX;
  localXdot = wash->localXdot;

  ierr = TSGetDM(ts,&networkdm);CHKERRQ(ierr);
  ierr = DMGetLocalVector(networkdm,&localF);CHKERRQ(ierr);
  ierr = VecSet(localF,0.0);CHKERRQ(ierr);

  /* update ghost values of locaX and locaXdot */
  ierr = DMGlobalToLocalBegin(networkdm,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(networkdm,X,INSERT_VALUES,localX);CHKERRQ(ierr);

  ierr = DMGlobalToLocalBegin(networkdm,Xdot,INSERT_VALUES,localXdot);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(networkdm,Xdot,INSERT_VALUES,localXdot);CHKERRQ(ierr);

  ierr = VecGetArrayRead(localX,&xarr);CHKERRQ(ierr);
  ierr = VecGetArrayRead(localXdot,&xdotarr);CHKERRQ(ierr);
  ierr = VecGetArray(localF,&farr);CHKERRQ(ierr);

  /* Initialize localF = localX at non-ghost vertices */
  ierr = DMNetworkGetVertexRange(networkdm,&vStart,&vEnd);CHKERRQ(ierr);
  for (v=vStart; v<vEnd; v++) {
    ierr = DMNetworkIsGhostVertex(networkdm,v,&ghost);CHKERRQ(ierr);
    if (!ghost) {
      ierr = DMNetworkGetVariableOffset(networkdm,v,&varoffset);CHKERRQ(ierr);
      juncx  = (PetscScalar*)(xarr+varoffset);
      juncf  = (PetscScalar*)(farr+varoffset);
      juncf[0] = juncx[0];
      juncf[1] = juncx[1];
    }
  }

  /* Get component(application) data array */
  ierr = DMNetworkGetComponentDataArray(networkdm,&nwarr);CHKERRQ(ierr);

  /* Edge */
  ierr = DMNetworkGetEdgeRange(networkdm,&eStart,&eEnd);CHKERRQ(ierr);
  for (e=eStart; e<eEnd; e++) {
    ierr = DMNetworkGetComponentKeyOffset(networkdm,e,0,&type,&pipeoffset);CHKERRQ(ierr);
    ierr = DMNetworkGetVariableOffset(networkdm,e,&varoffset);CHKERRQ(ierr);
    pipe     = (Pipe)(nwarr + pipeoffset);
    pipex    = (PipeField*)(xarr + varoffset);
    pipexdot = (PipeField*)(xdotarr + varoffset);
    pipef    = (PipeField*)(farr + varoffset);

    /* Get boundary values H0 and QL from connected vertices */
    ierr = DMNetworkGetConnectedVertices(networkdm,e,&cone);CHKERRQ(ierr);
    vfrom = cone[0]; /* local ordering */
    vto   = cone[1];
    ierr = DMNetworkGetVariableOffset(networkdm,vfrom,&offsetfrom);CHKERRQ(ierr);
    ierr = DMNetworkGetVariableOffset(networkdm,vto,&offsetto);CHKERRQ(ierr);
    if (pipe->boundary.Q0 == PIPE_CHARACTERISTIC) {
      pipe->boundary.H0 = (xarr+offsetfrom)[1]; /* h_from */
    } else {
      pipe->boundary.Q0 = (xarr+offsetfrom)[0]; /* q_from */
    }
    if (pipe->boundary.HL == PIPE_CHARACTERISTIC) {
      pipe->boundary.QL = (xarr+offsetto)[0];   /* q_to */
    } else {
      pipe->boundary.HL = (xarr+offsetto)[1];   /* h_to */
    }

    /* Evaluate PipeIFunctionLocal() */
    ierr = DMDAGetLocalInfo(pipe->da,&info);CHKERRQ(ierr);
    ierr = PipeIFunctionLocal(&info, t, pipex, pipexdot, pipef, pipe);CHKERRQ(ierr);

    /* Set F at vfrom */
    vf = (PetscScalar*)(farr+offsetfrom);
    if (pipe->boundary.Q0 == PIPE_CHARACTERISTIC) {
      vf[0] -= pipex[0].q; /* q_vfrom - q[0] */
    } else {
      vf[1] -= pipex[0].h; /* h_vfrom - h[0] */
    }

    /* Set F at vto */
    vf = (PetscScalar*)(farr+offsetto);
    if (pipe->boundary.HL == PIPE_CHARACTERISTIC) {
      vf[1] -= pipex[pipe->nnodes-1].h; /* h_vto - h[last] */
    } else {
      vf[0] -= pipex[pipe->nnodes-1].q; /* q_vto - q[last] */
    }
  }

  /* Set F at boundary vertices */
  for (v=vStart; v<vEnd; v++) {
    ierr = DMNetworkGetComponentKeyOffset(networkdm,v,0,&type,&voffset);CHKERRQ(ierr);
    ierr = DMNetworkGetVariableOffset(networkdm,v,&varoffset);CHKERRQ(ierr);
    junction = (Junction)(nwarr + voffset);
    juncf = (PetscScalar *)(farr + varoffset);
    if (junction->isEnd == -1) {
      juncf[1] -= wash->H0;
      } else if (junction->isEnd == 1) {
      juncf[0] -= wash->QL;
    }
  }

  ierr = VecRestoreArrayRead(localX,&xarr);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(localXdot,&xdotarr);CHKERRQ(ierr);
  ierr = VecRestoreArray(localF,&farr);CHKERRQ(ierr);

  ierr = DMLocalToGlobalBegin(networkdm,localF,ADD_VALUES,F);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(networkdm,localF,ADD_VALUES,F);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(networkdm,&localF);CHKERRQ(ierr);
  /* ierr = VecView(F,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); */
  PetscFunctionReturn(0);
}

PetscErrorCode WASHSetInitialSolution(DM networkdm,Vec X,Wash wash)
{
  PetscErrorCode ierr;
  PetscInt       k,nx,vkey,vOffset,vfrom,vto,offsetfrom,offsetto;
  PetscInt       type,varoffset;
  PetscInt       e,eStart,eEnd,pipeoffset;
  Vec            localX;
  PetscScalar    *xarr;
  Pipe           pipe;
  Junction       junction;
  const PetscInt *cone;
  const PetscScalar *xarray;
  DMNetworkComponentGenericDataType *nwarr;

  PetscFunctionBegin;
  ierr = VecSet(X,0.0);CHKERRQ(ierr);
  ierr = DMGetLocalVector(networkdm,&localX);CHKERRQ(ierr);
  ierr = VecGetArray(localX,&xarr);CHKERRQ(ierr);

   /* Get component(application) data array */
  ierr = DMNetworkGetComponentDataArray(networkdm,&nwarr);CHKERRQ(ierr);

  /* Edge */
  ierr = DMNetworkGetEdgeRange(networkdm,&eStart,&eEnd);CHKERRQ(ierr);
  for (e=eStart; e<eEnd; e++) {
    ierr = DMNetworkGetVariableOffset(networkdm,e,&varoffset);CHKERRQ(ierr);
    ierr = DMNetworkGetComponentKeyOffset(networkdm,e,0,&type,&pipeoffset);CHKERRQ(ierr);

    /* get from and to vertices */
    ierr = DMNetworkGetConnectedVertices(networkdm,e,&cone);CHKERRQ(ierr);
    vfrom = cone[0]; /* local ordering */
    vto   = cone[1];

    ierr = DMNetworkGetVariableOffset(networkdm,vfrom,&offsetfrom);CHKERRQ(ierr);
    ierr = DMNetworkGetVariableOffset(networkdm,vto,&offsetto);CHKERRQ(ierr);

    /* set initial values for this pipe */
    /* Q0=0.477432; H0=150.0; needs to be updated by its succeeding pipe. Use SNESSolve()? */
    pipe     = (Pipe)(nwarr + pipeoffset);
    ierr = PipeComputeSteadyState(pipe, 0.477432, wash->H0);CHKERRQ(ierr);
    ierr = VecGetSize(pipe->x,&nx);CHKERRQ(ierr);

    ierr = VecGetArrayRead(pipe->x,&xarray);CHKERRQ(ierr);
    /* copy pipe->x to xarray */
    for (k=0; k<nx; k++) {
      (xarr+varoffset)[k] = xarray[k];
    }

    /* set boundary values into vfrom and vto */
    if (pipe->boundary.Q0 == PIPE_CHARACTERISTIC) {
      (xarr+offsetfrom)[0] += xarray[0];    /* Q0 -> vfrom[0] */
    } else {
      (xarr+offsetfrom)[1] += xarray[1];    /* H0 -> vfrom[1] */
    }

    if (pipe->boundary.HL == PIPE_CHARACTERISTIC) {
      (xarr+offsetto)[1]   += xarray[nx-1]; /* HL -> vto[1]   */
    } else {
      (xarr+offsetto)[0]   += xarray[nx-2]; /* QL -> vto[0]   */
    }

    /* if vform is a head vertex: */
    ierr = DMNetworkGetComponentKeyOffset(networkdm,vfrom,0,&vkey,&vOffset);CHKERRQ(ierr);
    junction = (Junction)(nwarr+vOffset);
    if (junction->isEnd == -1) { /* head junction */
      (xarr+offsetfrom)[0] = 0.0;      /* 1st Q -- not used */
      (xarr+offsetfrom)[1] = wash->H0; /* 1st H */
    }

    /* if vto is an end vertex: */
    ierr = DMNetworkGetComponentKeyOffset(networkdm,vto,0,&vkey,&vOffset);CHKERRQ(ierr);
    junction = (Junction)(nwarr+vOffset);
    if (junction->isEnd == 1) { /* end junction */
      (xarr+offsetto)[0] = wash->QL; /* last Q */
      (xarr+offsetto)[1] = 0.0;      /* last H -- not used */
    }
    ierr = VecRestoreArrayRead(pipe->x,&xarray);CHKERRQ(ierr);
  }

  ierr = VecRestoreArray(localX,&xarr);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(networkdm,localX,ADD_VALUES,X);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(networkdm,localX,ADD_VALUES,X);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(networkdm,&localX);CHKERRQ(ierr);

#if 0
  PetscInt N;
  ierr = VecGetSize(X,&N);CHKERRQ(ierr);
  printf("initial solution %d:\n",N);
  ierr = VecView(X,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode TSDMNetworkMonitor(TS ts, PetscInt step, PetscReal t, Vec x, void *context)
{
  PetscErrorCode     ierr;
  DMNetworkMonitor   monitor;

  PetscFunctionBegin;
  monitor = (DMNetworkMonitor)context;
  ierr = DMNetworkMonitorView(monitor,x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PipesView(Vec X,DM networkdm,Wash wash)
{
  PetscErrorCode       ierr;
  Pipe                 pipe;
  DMNetworkComponentGenericDataType *nwarr;
  PetscInt             pipeOffset,key,Start,End;
  PetscMPIInt          rank;
  PetscInt             nx,nnodes,nidx,*idx1,*idx2,*idx1_h,*idx2_h,idx_start,i,k,k1,xstart,j1;
  Vec                  Xq,Xh,localX;
  IS                   is1_q,is2_q,is1_h,is2_h;
  VecScatter           ctx_q,ctx_h;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  /* get num of local and global total nnodes */
  nidx = wash->nnodes_loc;
  ierr = MPIU_Allreduce(&nidx,&nx,1,MPIU_INT,MPI_SUM,PETSC_COMM_WORLD);CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD,&Xq);CHKERRQ(ierr);
  if (rank == 0) { /* all entries of Xq are in proc[0] */
    ierr = VecSetSizes(Xq,nx,PETSC_DECIDE);CHKERRQ(ierr);
  } else {
    ierr = VecSetSizes(Xq,0,PETSC_DECIDE);CHKERRQ(ierr);
  }
  ierr = VecSetFromOptions(Xq);CHKERRQ(ierr);
  ierr = VecSet(Xq,0.0);CHKERRQ(ierr);
  ierr = VecDuplicate(Xq,&Xh);CHKERRQ(ierr);

  ierr = DMGetLocalVector(networkdm,&localX);CHKERRQ(ierr);

  /* set idx1 and idx2 */
  ierr = PetscCalloc4(nidx,&idx1,nidx,&idx2,nidx,&idx1_h,nidx,&idx2_h);CHKERRQ(ierr);

  ierr = DMNetworkGetComponentDataArray(networkdm,&nwarr);CHKERRQ(ierr);
  ierr = DMNetworkGetEdgeRange(networkdm,&Start, &End);CHKERRQ(ierr);

  ierr = VecGetOwnershipRange(X,&xstart,NULL);CHKERRQ(ierr);
  k1 = 0;
  j1 = 0;
  for (i = Start; i < End; i++) {
    ierr = DMNetworkGetComponentKeyOffset(networkdm,i,0,&key,&pipeOffset);CHKERRQ(ierr);
    pipe = (Pipe)(nwarr+pipeOffset);
    nnodes = pipe->nnodes;
    idx_start = pipe->id*nnodes;
    for (k=0; k<nnodes; k++) {
      idx1[k1] = xstart + j1*2*nnodes + 2*k;
      idx2[k1] = idx_start + k;

      idx1_h[k1] = xstart + j1*2*nnodes + 2*k + 1;
      idx2_h[k1] = idx_start + k;
      k1++;
    }
    j1++;
  }

  ierr = ISCreateGeneral(PETSC_COMM_SELF,nidx,idx1,PETSC_COPY_VALUES,&is1_q);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF,nidx,idx2,PETSC_COPY_VALUES,&is2_q);CHKERRQ(ierr);
  ierr = VecScatterCreate(X,is1_q,Xq,is2_q,&ctx_q);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx_q,X,Xq,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx_q,X,Xq,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

  ierr = ISCreateGeneral(PETSC_COMM_SELF,nidx,idx1_h,PETSC_COPY_VALUES,&is1_h);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF,nidx,idx2_h,PETSC_COPY_VALUES,&is2_h);CHKERRQ(ierr);
  ierr = VecScatterCreate(X,is1_h,Xh,is2_h,&ctx_h);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx_h,X,Xh,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx_h,X,Xh,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

  if (!rank) printf("Xq: \n");
  ierr = VecView(Xq,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  if (!rank) printf("Xh: \n");
  ierr = VecView(Xh,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = VecScatterDestroy(&ctx_q);CHKERRQ(ierr);
  ierr = PetscFree4(idx1,idx2,idx1_h,idx2_h);CHKERRQ(ierr);
  ierr = ISDestroy(&is1_q);CHKERRQ(ierr);
  ierr = ISDestroy(&is2_q);CHKERRQ(ierr);

  ierr = VecScatterDestroy(&ctx_h);CHKERRQ(ierr);
  ierr = ISDestroy(&is1_h);CHKERRQ(ierr);
  ierr = ISDestroy(&is2_h);CHKERRQ(ierr);

  ierr = VecDestroy(&Xq);CHKERRQ(ierr);
  ierr = VecDestroy(&Xh);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(networkdm,&localX);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode WashNetworkCleanUp(Wash wash,int *edgelist)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(wash->comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    ierr = PetscFree(edgelist);CHKERRQ(ierr);
    ierr = PetscFree2(wash->junction,wash->pipe);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode WashNetworkCreate(MPI_Comm comm,PetscInt pipesCase,Wash *wash_ptr,int **elist)
{
  PetscErrorCode ierr;
  PetscInt       nnodes,npipes;
  PetscMPIInt    rank;
  Wash           wash;
  PetscInt       i,numVertices,numEdges;
  int            *edgelist;
  Junction       junctions=NULL;
  Pipe           pipes=NULL;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  ierr = PetscCalloc1(1,&wash);CHKERRQ(ierr);
  wash->comm = comm;
  *wash_ptr  = wash;
  wash->Q0   = 0.477432; /* copied from initial soluiton */
  wash->H0   = 150.0;
  wash->HL   = 143.488; /* copied from initial soluiton */
  wash->nnodes_loc = 0;

  numVertices = 0;
  numEdges    = 0;
  edgelist    = NULL;

  if (!rank) {
    ierr = PetscPrintf(PETSC_COMM_SELF,"Setup pipesCase %D\n",pipesCase);CHKERRQ(ierr);
  }
  nnodes = 6;
  ierr = PetscOptionsGetInt(NULL,NULL, "-npipenodes", &nnodes, NULL);CHKERRQ(ierr);

  /* Set global number of pipes, edges, and junctions */
  /*-------------------------------------------------*/
  switch (pipesCase) {
  case 0:
    /* pipeCase 0: */
    /* =============================
    v0 --E0--> v1--E1--> v2 --E2-->v3
    ================================  */
    npipes = 3;
    ierr = PetscOptionsGetInt(NULL,NULL, "-npipes", &npipes, NULL);CHKERRQ(ierr);
    wash->nedge   = npipes;
    wash->nvertex = npipes + 1;

    /* Set local edges and vertices -- proc[0] sets entire network, then distributes */
    numVertices = 0;
    numEdges    = 0;
    edgelist    = NULL;
    if (!rank) {
      numVertices = wash->nvertex;
      numEdges    = wash->nedge;

      ierr = PetscCalloc1(2*numEdges,&edgelist);CHKERRQ(ierr);
      for (i=0; i<numEdges; i++) {
        edgelist[2*i] = i; edgelist[2*i+1] = i+1;
      }

      /* Add network components */
      /*------------------------*/
      ierr = PetscCalloc2(numVertices,&junctions,numEdges,&pipes);CHKERRQ(ierr);
      /* vertex */
      for (i=0; i<numVertices; i++) {
        junctions[i].id = i;
        junctions[i].isEnd = 0;
        junctions[i].nedges_in = 1; junctions[i].nedges_out = 1;

        /* Set GPS data */
        junctions[i].latitude  = 0.0;
        junctions[i].longitude = 0.0;
      }
      junctions[0].isEnd                  = -1;
      junctions[0].nedges_in              =  0;
      junctions[numVertices-1].isEnd      =  1;
      junctions[numVertices-1].nedges_out =  0;

      /* edge and pipe */
      for (i=0; i<numEdges; i++) {
        pipes[i].id   = i;
        pipes[i].nnodes = nnodes;
      }
    }
    break;
  case 1:
    /* pipeCase 1: */
    /* ==========================
                v2
                ^
                |
               E2
                |
    v0 --E0--> v3--E1--> v1
    =============================  */
    npipes = 3;
    wash->nedge   = npipes;
    wash->nvertex = npipes + 1;

    /* Set local edges and vertices -- proc[0] sets entire network, then distributes */
    if (!rank) {
      numVertices = wash->nvertex;
      numEdges    = wash->nedge;

      ierr = PetscCalloc1(2*numEdges,&edgelist);CHKERRQ(ierr);
      edgelist[0] = 0; edgelist[1] = 3;  /* edge[0] */
      edgelist[2] = 3; edgelist[3] = 1;  /* edge[1] */
      edgelist[4] = 3; edgelist[5] = 2;  /* edge[2] */

      /* Add network components */
      /*------------------------*/
      ierr = PetscCalloc2(numVertices,&junctions,numEdges,&pipes);CHKERRQ(ierr);
      /* vertex */
      for (i=0; i<numVertices; i++) {
        junctions[i].id = i;

        /* Set GPS data */
        junctions[i].latitude  = 0.0;
        junctions[i].longitude = 0.0;
      }
      junctions[0].isEnd = -1; junctions[0].nedges_in = 0; junctions[0].nedges_out = 1;
      junctions[1].isEnd =  1; junctions[1].nedges_in = 1; junctions[1].nedges_out = 0;
      junctions[2].isEnd =  1; junctions[2].nedges_in = 1; junctions[2].nedges_out = 0;
      junctions[3].isEnd =  0; junctions[3].nedges_in = 1; junctions[3].nedges_out = 2;

      /* edge and pipe */
      for (i=0; i<numEdges; i++) {
        pipes[i].id     = i;
        pipes[i].nnodes = nnodes;
      }
    }
    break;
  case 2:
    /* pipeCase 2: */
    /* ==========================
         v2--> E2
                |
    v0 --E0--> v3--E1--> v1
    =============================  */

    /* Set application parameters -- to be used in function evalutions */
    npipes = 3;
    wash->nedge   = npipes;
    wash->nvertex = npipes + 1;

    /* Set local edges and vertices -- proc[0] sets entire network, then distributes */
    if (!rank) {
      numVertices = wash->nvertex;
      numEdges    = wash->nedge;

      ierr = PetscCalloc1(2*numEdges,&edgelist);CHKERRQ(ierr);
      edgelist[0] = 0; edgelist[1] = 3;  /* edge[0] */
      edgelist[2] = 3; edgelist[3] = 1;  /* edge[1] */
      edgelist[4] = 2; edgelist[5] = 3;  /* edge[2] */

      /* Add network components */
      /*------------------------*/
      ierr = PetscCalloc2(numVertices,&junctions,numEdges,&pipes);CHKERRQ(ierr);
      /* vertex */
      for (i=0; i<numVertices; i++) {
        junctions[i].id = i;

        /* Set GPS data */
        junctions[i].latitude  = 0.0;
        junctions[i].longitude = 0.0;
      }
      junctions[0].isEnd = -1; junctions[0].nedges_in = 0; junctions[0].nedges_out = 1;
      junctions[1].isEnd =  1; junctions[1].nedges_in = 1; junctions[1].nedges_out = 0;
      junctions[2].isEnd = -1; junctions[2].nedges_in = 0; junctions[2].nedges_out = 1;
      junctions[3].isEnd =  0; junctions[3].nedges_in = 2; junctions[3].nedges_out = 1;

      /* edge and pipe */
      for (i=0; i<numEdges; i++) {
        pipes[i].id     = i;
        pipes[i].nnodes = nnodes;
      }
    }
    break;
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"not done yet");
  }

  *wash_ptr      = wash;
  wash->nedge    = numEdges;
  wash->nvertex  = numVertices;
  *elist         = edgelist;
  wash->junction = junctions;
  wash->pipe     = pipes;
  PetscFunctionReturn(0);
}
/* ------------------------------------------------------- */
int main(int argc,char ** argv)
{
  PetscErrorCode    ierr;
  Wash              wash;
  Junction          junctions,junction;
  Pipe              pipe,pipes;
  PetscInt          numEdges,numVertices,KeyPipe,KeyJunction;
  int               *edgelist = NULL;
  PetscInt          i,e,v,eStart,eEnd,vStart,vEnd,pipeOffset,key,frombType,tobType;
  PetscInt          vfrom,vto,vkey,fromOffset,toOffset,type,varoffset,pipeoffset;
  PetscInt          from_nedge_in,from_nedge_out,to_nedge_in;
  const PetscInt    *cone;
  DM                networkdm;
  PetscMPIInt       size,rank;
  PetscReal         ftime = 2500.0;
  Vec               X;
  TS                ts;
  PetscInt          steps;
  TSConvergedReason reason;
  PetscBool         viewpipes;
  PetscInt          pipesCase;
  DMNetworkMonitor  monitor;
  DMNetworkComponentGenericDataType *nwarr;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);

  /* Create and setup network */
  /*--------------------------*/
  ierr = DMNetworkCreate(PETSC_COMM_WORLD,&networkdm);CHKERRQ(ierr);
  if (size == 1) {
    ierr = DMNetworkMonitorCreate(networkdm,&monitor);CHKERRQ(ierr);
  }
  /* Register the components in the network */
  ierr = DMNetworkRegisterComponent(networkdm,"junctionstruct",sizeof(struct _p_Junction),&KeyJunction);CHKERRQ(ierr);
  ierr = DMNetworkRegisterComponent(networkdm,"pipestruct",sizeof(struct _p_Pipe),&KeyPipe);CHKERRQ(ierr);

  /* Set global number of pipes, edges, and vertices */
  pipesCase = 2;
  ierr = PetscOptionsGetInt(NULL,NULL, "-case", &pipesCase, NULL);CHKERRQ(ierr);

  ierr = WashNetworkCreate(PETSC_COMM_WORLD,pipesCase,&wash,&edgelist);CHKERRQ(ierr);
  numEdges    = wash->nedge;
  numVertices = wash->nvertex;
  junctions    = wash->junction;
  pipes       = wash->pipe;

  /* Set number of vertices and edges */
  ierr = DMNetworkSetSizes(networkdm,numVertices,numEdges,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  /* Add edge connectivity */
  ierr = DMNetworkSetEdgeList(networkdm,edgelist);CHKERRQ(ierr); 
  /* Set up the network layout */
  ierr = DMNetworkLayoutSetUp(networkdm);CHKERRQ(ierr);

  /* Add EDGEDATA component to all edges -- currently networkdm is a sequential network */
  ierr = DMNetworkGetEdgeRange(networkdm,&eStart,&eEnd);CHKERRQ(ierr);
  ierr = DMNetworkGetVertexRange(networkdm,&vStart,&vEnd);CHKERRQ(ierr);

  for (e = eStart; e < eEnd; e++) {
    /* Add Pipe component to all edges -- create pipe here */
    ierr = DMNetworkAddComponent(networkdm,e,KeyPipe,&pipes[e-eStart]);CHKERRQ(ierr);

    /* Add number of variables to each edge */
    ierr = DMNetworkAddNumVariables(networkdm,e,2*pipes[e-eStart].nnodes);CHKERRQ(ierr);

    if (size == 1) { /* Add monitor -- show Q_{pipes[e-eStart].id}? */
      ierr = DMNetworkMonitorAdd(monitor, "Pipe Q", e, pipes[e-eStart].nnodes, 0, 2, -0.8, 0.8, PETSC_TRUE);CHKERRQ(ierr);
      ierr = DMNetworkMonitorAdd(monitor, "Pipe H", e, pipes[e-eStart].nnodes, 1, 2, -400.0, 800.0, PETSC_TRUE);CHKERRQ(ierr);
    }
  }

  /* Add Junction component to all vertices */
  for (v = vStart; v < vEnd; v++) {
    ierr = DMNetworkAddComponent(networkdm,v,KeyJunction,&junctions[v-vStart]);CHKERRQ(ierr);

    /* Add number of variables to vertex */
    ierr = DMNetworkAddNumVariables(networkdm,v,2);CHKERRQ(ierr);
  }

  /* Set up DM for use */
  ierr = DMSetUp(networkdm);CHKERRQ(ierr);
  ierr = WashNetworkCleanUp(wash,edgelist);CHKERRQ(ierr);

  /* Network partitioning and distribution of data */
  ierr = DMNetworkDistribute(&networkdm,0);CHKERRQ(ierr);

  /* PipeSetUp -- each process only sets its own pipes */
  /*---------------------------------------------------*/
  ierr = DMNetworkGetComponentDataArray(networkdm,&nwarr);CHKERRQ(ierr);

  ierr = DMNetworkGetEdgeRange(networkdm,&eStart,&eEnd);CHKERRQ(ierr);
  for (e=eStart; e<eEnd; e++) { /* each edge has only one component, pipe */
    ierr = DMNetworkGetComponentKeyOffset(networkdm,e,0,&type,&pipeoffset);CHKERRQ(ierr);
    ierr = DMNetworkGetVariableOffset(networkdm,e,&varoffset);CHKERRQ(ierr);
    pipe = (Pipe)(nwarr + pipeoffset);

    /* Setup conntected vertices */
    ierr = DMNetworkGetConnectedVertices(networkdm,e,&cone);CHKERRQ(ierr);
    vfrom = cone[0]; /* local ordering */
    vto   = cone[1];

    /* vfrom */
    ierr = DMNetworkGetComponentKeyOffset(networkdm,vfrom,0,&vkey,&fromOffset);CHKERRQ(ierr);
    junction = (Junction)(nwarr+fromOffset);
    from_nedge_in  = junction->nedges_in;
    from_nedge_out = junction->nedges_out;

    /* vto */
    ierr = DMNetworkGetComponentKeyOffset(networkdm,vto,0,&vkey,&toOffset);CHKERRQ(ierr);
    junction    = (Junction)(nwarr+toOffset);
    to_nedge_in = junction->nedges_in;

    pipe->comm = PETSC_COMM_SELF; /* must be set here, otherwise crashes in my mac??? */
    wash->nnodes_loc += pipe->nnodes; /* local total num of nodes, will be used by PipesView() */
    ierr = PipeSetParameters(pipe,
                             600.0,          /* length */
                             pipe->nnodes,   /* nnodes -- rm from PipeSetParameters */
                             0.5,            /* diameter */
                             1200.0,         /* a */
                             0.018);CHKERRQ(ierr);    /* friction */

    /* set boundary conditions for this pipe */
    if (from_nedge_in <= 1 && from_nedge_out > 0) {
      frombType = 0;
    } else {
      frombType = 1;
    }

    if (to_nedge_in == 1) {
      tobType = 0;
    } else {
      tobType = 1;
    }

    if (frombType == 0) {
      pipe->boundary.Q0 = PIPE_CHARACTERISTIC; /* will be obtained from characteristic */
      pipe->boundary.H0 = wash->H0;
    } else {
      pipe->boundary.Q0 = wash->Q0;
      pipe->boundary.H0 = PIPE_CHARACTERISTIC; /* will be obtained from characteristic */
    }
    if (tobType == 0) {
      pipe->boundary.QL = wash->QL;
      pipe->boundary.HL = PIPE_CHARACTERISTIC; /* will be obtained from characteristic */
    } else {
      pipe->boundary.QL = PIPE_CHARACTERISTIC; /* will be obtained from characteristic */
      pipe->boundary.HL = wash->HL;
    }

    ierr = PipeSetUp(pipe);CHKERRQ(ierr);
  }

  /* create vectors */
  ierr = DMCreateGlobalVector(networkdm,&X);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(networkdm,&wash->localX);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(networkdm,&wash->localXdot);CHKERRQ(ierr);

  /* Setup solver                                           */
  /*--------------------------------------------------------*/
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);

  ierr = TSSetDM(ts,(DM)networkdm);CHKERRQ(ierr);
  ierr = TSSetIFunction(ts,NULL,WASHIFunction,wash);CHKERRQ(ierr);

  ierr = TSSetMaxTime(ts,ftime);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,0.1);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSBEULER);CHKERRQ(ierr);
  if (size == 1) {
    ierr = TSMonitorSet(ts, TSDMNetworkMonitor, monitor, NULL);CHKERRQ(ierr);
  }
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  ierr = WASHSetInitialSolution(networkdm,X,wash);CHKERRQ(ierr);

  ierr = TSSolve(ts,X);CHKERRQ(ierr);

  ierr = TSGetSolveTime(ts,&ftime);CHKERRQ(ierr);
  ierr = TSGetStepNumber(ts,&steps);CHKERRQ(ierr);
  ierr = TSGetConvergedReason(ts,&reason);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"%s at time %g after %D steps\n",TSConvergedReasons[reason],(double)ftime,steps);CHKERRQ(ierr);
  /* ierr = VecView(X,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); */

  /* View solution q and h */
  /* --------------------- */
  viewpipes = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL, "-pipe_view", &viewpipes,NULL);CHKERRQ(ierr);
  if (viewpipes) {
    ierr = PipesView(X,networkdm,wash);CHKERRQ(ierr);
  }

  /* Free spaces */
  /* ----------- */
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = VecDestroy(&wash->localX);CHKERRQ(ierr);
  ierr = VecDestroy(&wash->localXdot);CHKERRQ(ierr);

  /* Destroy objects from each pipe that are created in PipeSetUp() */
  ierr = DMNetworkGetEdgeRange(networkdm,&eStart, &eEnd);CHKERRQ(ierr);
  for (i = eStart; i < eEnd; i++) {
    ierr = DMNetworkGetComponentKeyOffset(networkdm,i,0,&key,&pipeOffset);CHKERRQ(ierr);
    pipe = (Pipe)(nwarr+pipeOffset);
    ierr = DMDestroy(&(pipe->da));CHKERRQ(ierr);
    ierr = VecDestroy(&pipe->x);CHKERRQ(ierr);
  }
  if (size == 1) {
    ierr = DMNetworkMonitorDestroy(&monitor);CHKERRQ(ierr);
  }
  ierr = DMDestroy(&networkdm);CHKERRQ(ierr);
  ierr = PetscFree(wash);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
