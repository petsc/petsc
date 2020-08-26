static char help[] = "This example demonstrates the use of DMNetwork interface with subnetworks for solving a coupled nonlinear \n\
                      electric power grid and water pipe problem.\n\
                      The available solver options are in the ex1options file \n\
                      and the data files are in the datafiles of subdirectories.\n\
                      This example shows the use of subnetwork feature in DMNetwork. \n\
                      Run this program: mpiexec -n <n> ./ex1 \n\\n";

/* T
   Concepts: DMNetwork
   Concepts: PETSc SNES solver
*/

#include "power/power.h"
#include "water/water.h"

typedef struct{
  UserCtx_Power appctx_power;
  AppCtx_Water  appctx_water;
  PetscInt      subsnes_id; /* snes solver id */
  PetscInt      it;         /* iteration number */
  Vec           localXold;  /* store previous solution, used by FormFunction_Dummy() */
} UserCtx;

PetscErrorCode UserMonitor(SNES snes,PetscInt its,PetscReal fnorm ,void *appctx)
{
  PetscErrorCode ierr;
  UserCtx        *user = (UserCtx*)appctx;
  Vec            X,localXold=user->localXold;
  DM             networkdm;
  PetscMPIInt    rank;
  MPI_Comm       comm;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)snes,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
#if 0
  if (!rank) {
    PetscInt       subsnes_id=user->subsnes_id;
    if (subsnes_id == 2) {
      ierr = PetscPrintf(PETSC_COMM_SELF," it %d, subsnes_id %d, fnorm %g\n",user->it,user->subsnes_id,fnorm);CHKERRQ(ierr);
    } else {
      ierr = PetscPrintf(PETSC_COMM_SELF,"       subsnes_id %d, fnorm %g\n",user->subsnes_id,fnorm);CHKERRQ(ierr);
    }
  }
#endif
  ierr = SNESGetSolution(snes,&X);CHKERRQ(ierr);
  ierr = SNESGetDM(snes,&networkdm);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(networkdm,X,INSERT_VALUES,localXold);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(networkdm,X,INSERT_VALUES,localXold);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode FormJacobian_subPower(SNES snes,Vec X, Mat J,Mat Jpre,void *appctx)
{
  PetscErrorCode ierr;
  DM             networkdm;
  Vec            localX;
  PetscInt       nv,ne,i,j,offset,nvar,row;
  const PetscInt *vtx,*edges;
  PetscBool      ghostvtex;
  PetscScalar    one = 1.0;
  PetscMPIInt    rank;
  MPI_Comm       comm;

  PetscFunctionBegin;
  ierr = SNESGetDM(snes,&networkdm);CHKERRQ(ierr);
  ierr = DMGetLocalVector(networkdm,&localX);CHKERRQ(ierr);

  ierr = PetscObjectGetComm((PetscObject)networkdm,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  ierr = DMGlobalToLocalBegin(networkdm,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(networkdm,X,INSERT_VALUES,localX);CHKERRQ(ierr);

  ierr = MatZeroEntries(J);CHKERRQ(ierr);

  /* Power subnetwork: copied from power/FormJacobian_Power() */
  ierr = DMNetworkGetSubnetworkInfo(networkdm,0,&nv,&ne,&vtx,&edges);CHKERRQ(ierr);
  ierr = FormJacobian_Power_private(networkdm,localX,J,nv,ne,vtx,edges,appctx);CHKERRQ(ierr);

  /* Water subnetwork: Identity */
  ierr = DMNetworkGetSubnetworkInfo(networkdm,1,&nv,&ne,&vtx,&edges);CHKERRQ(ierr);
  for (i=0; i<nv; i++) {
    ierr = DMNetworkIsGhostVertex(networkdm,vtx[i],&ghostvtex);CHKERRQ(ierr);
    if (ghostvtex) continue;

    ierr = DMNetworkGetVariableGlobalOffset(networkdm,vtx[i],&offset);CHKERRQ(ierr);
    ierr = DMNetworkGetNumVariables(networkdm,vtx[i],&nvar);CHKERRQ(ierr);
    for (j=0; j<nvar; j++) {
      row = offset + j;
      ierr = MatSetValues(J,1,&row,1,&row,&one,ADD_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = DMRestoreLocalVector(networkdm,&localX);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Dummy equation localF(X) = localX - localXold */
PetscErrorCode FormFunction_Dummy(DM networkdm,Vec localX, Vec localF,PetscInt nv,PetscInt ne,const PetscInt* vtx,const PetscInt* edges,void* appctx)
{
  PetscErrorCode    ierr;
  const PetscScalar *xarr,*xoldarr;
  PetscScalar       *farr;
  PetscInt          i,j,offset,nvar;
  PetscBool         ghostvtex;
  UserCtx           *user = (UserCtx*)appctx;
  Vec               localXold = user->localXold;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(localX,&xarr);CHKERRQ(ierr);
  ierr = VecGetArrayRead(localXold,&xoldarr);CHKERRQ(ierr);
  ierr = VecGetArray(localF,&farr);CHKERRQ(ierr);

  for (i=0; i<nv; i++) {
    ierr = DMNetworkIsGhostVertex(networkdm,vtx[i],&ghostvtex);CHKERRQ(ierr);
    if (ghostvtex) continue;

    ierr = DMNetworkGetVariableOffset(networkdm,vtx[i],&offset);CHKERRQ(ierr);
    ierr = DMNetworkGetNumVariables(networkdm,vtx[i],&nvar);CHKERRQ(ierr);
    for (j=0; j<nvar; j++) {
      farr[offset+j] = xarr[offset+j] - xoldarr[offset+j];
    }
  }

  ierr = VecRestoreArrayRead(localX,&xarr);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(localXold,&xoldarr);CHKERRQ(ierr);
  ierr = VecRestoreArray(localF,&farr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode FormFunction(SNES snes,Vec X,Vec F,void *appctx)
{
  PetscErrorCode ierr;
  DM             networkdm;
  Vec            localX,localF;
  PetscInt       nv,ne;
  const PetscInt *vtx,*edges;
  PetscMPIInt    rank;
  MPI_Comm       comm;
  UserCtx        *user = (UserCtx*)appctx;
  UserCtx_Power  appctx_power = (*user).appctx_power;

  PetscFunctionBegin;
  ierr = SNESGetDM(snes,&networkdm);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)networkdm,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  ierr = DMGetLocalVector(networkdm,&localX);CHKERRQ(ierr);
  ierr = DMGetLocalVector(networkdm,&localF);CHKERRQ(ierr);
  ierr = VecSet(F,0.0);CHKERRQ(ierr);
  ierr = VecSet(localF,0.0);CHKERRQ(ierr);

  ierr = DMGlobalToLocalBegin(networkdm,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(networkdm,X,INSERT_VALUES,localX);CHKERRQ(ierr);

  /* Form Function for power subnetwork */
  ierr = DMNetworkGetSubnetworkInfo(networkdm,0,&nv,&ne,&vtx,&edges);CHKERRQ(ierr);
  if (user->subsnes_id == 1) { /* snes_water only */
    ierr = FormFunction_Dummy(networkdm,localX,localF,nv,ne,vtx,edges,user);CHKERRQ(ierr);
  } else {
    ierr = FormFunction_Power(networkdm,localX,localF,nv,ne,vtx,edges,&appctx_power);CHKERRQ(ierr);
  }

  /* Form Function for water subnetwork */
  ierr = DMNetworkGetSubnetworkInfo(networkdm,1,&nv,&ne,&vtx,&edges);CHKERRQ(ierr);
  if (user->subsnes_id == 0) { /* snes_power only */
    ierr = FormFunction_Dummy(networkdm,localX,localF,nv,ne,vtx,edges,user);CHKERRQ(ierr);
  } else {
    ierr = FormFunction_Water(networkdm,localX,localF,nv,ne,vtx,edges,NULL);CHKERRQ(ierr);
  }

#if 0
  /* Form Function for the coupling subnetwork -- experimental, not done yet */
  ierr = DMNetworkGetSubnetworkCoupleInfo(networkdm,0,&ne,&edges);CHKERRQ(ierr);
  if (ne) {
    const PetscInt *cone,*connedges,*econe;
    PetscInt       key,vid[2],nconnedges,k,e,keye;
    void*          component;
    AppCtx_Water   appctx_water = (*user).appctx_water;

    ierr = DMNetworkGetConnectedVertices(networkdm,edges[0],&cone);CHKERRQ(ierr);

    ierr = DMNetworkGetGlobalVertexIndex(networkdm,cone[0],&vid[0]);CHKERRQ(ierr);
    ierr = DMNetworkGetGlobalVertexIndex(networkdm,cone[1],&vid[1]);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] Formfunction, coupling subnetwork: ne %d; connected vertices %d %d\n",rank,ne,vid[0],vid[1]);CHKERRQ(ierr);

    /* Get coupling powernet load vertex */
    ierr = DMNetworkGetComponent(networkdm,cone[0],1,&key,&component);CHKERRQ(ierr);
    if (key != appctx_power.compkey_load) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Not a power load vertex");

    /* Get coupling water vertex and pump edge */
    ierr = DMNetworkGetComponent(networkdm,cone[1],0,&key,&component);CHKERRQ(ierr);
    if (key != appctx_water.compkey_vtx) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Not a water vertex");

    /* get its supporting edges */
    ierr = DMNetworkGetSupportingEdges(networkdm,cone[1],&nconnedges,&connedges);CHKERRQ(ierr);

    for (k=0; k<nconnedges; k++) {
      e = connedges[k];
      ierr = DMNetworkGetComponent(networkdm,e,0,&keye,&component);CHKERRQ(ierr);

      if (keye == appctx_water.compkey_edge) { /* water_compkey_edge */
        EDGE_Water        edge=(EDGE_Water)component;
        if (edge->type == EDGE_TYPE_PUMP) {
          /* compute flow_pump */
          PetscInt offsetnode1,offsetnode2,key_0,key_1;
          const PetscScalar *xarr;
          PetscScalar       *farr;
          VERTEX_Water      vertexnode1,vertexnode2;

          ierr = DMNetworkGetConnectedVertices(networkdm,e,&econe);CHKERRQ(ierr);
          ierr = DMNetworkGetGlobalVertexIndex(networkdm,econe[0],&vid[0]);CHKERRQ(ierr);
          ierr = DMNetworkGetGlobalVertexIndex(networkdm,econe[1],&vid[1]);CHKERRQ(ierr);

          ierr = VecGetArray(localF,&farr);CHKERRQ(ierr);
          ierr = DMNetworkGetVariableOffset(networkdm,econe[0],&offsetnode1);CHKERRQ(ierr);
          ierr = DMNetworkGetVariableOffset(networkdm,econe[1],&offsetnode2);CHKERRQ(ierr);

          ierr = VecGetArrayRead(localX,&xarr);CHKERRQ(ierr);
#if 0
          PetscScalar  hf,ht;
          Pump         *pump;
          pump = &edge->pump;
          hf = xarr[offsetnode1];
          ht = xarr[offsetnode2];

          PetscScalar flow = Flow_Pump(pump,hf,ht);
          PetscScalar Hp = 0.1; /* load->pl */
          PetscScalar flow_couple = 8.81*Hp*1.e6/(ht-hf);     /* pump->h0; */
          /* ierr = PetscPrintf(PETSC_COMM_SELF,"pump %d: connected vtx %d %d; flow_pump %g flow_couple %g; offset %d %d\n",e,vid[0],vid[1],flow,flow_couple,offsetnode1,offsetnode2);CHKERRQ(ierr); */
#endif
          /* Get the components at the two vertices */
          ierr = DMNetworkGetComponent(networkdm,econe[0],0,&key_0,(void**)&vertexnode1);CHKERRQ(ierr);
          if (key_0 != appctx_water.compkey_vtx) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Not a water vertex");
          ierr = DMNetworkGetComponent(networkdm,econe[1],0,&key_1,(void**)&vertexnode2);CHKERRQ(ierr);
          if (key_1 != appctx_water.compkey_vtx) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Not a water vertex");
#if 0
          /* subtract flow_pump computed in FormFunction_Water() and add flow_couple to connected nodes */
          if (vertexnode1->type == VERTEX_TYPE_JUNCTION) {
            farr[offsetnode1] += flow;
            farr[offsetnode1] -= flow_couple;
          }
          if (vertexnode2->type == VERTEX_TYPE_JUNCTION) {
            farr[offsetnode2] -= flow;
            farr[offsetnode2] += flow_couple;
          }
#endif
          ierr = VecRestoreArrayRead(localX,&xarr);CHKERRQ(ierr);
          ierr = VecRestoreArray(localF,&farr);CHKERRQ(ierr);
        }
      }
    }
  }
#endif

  ierr = DMRestoreLocalVector(networkdm,&localX);CHKERRQ(ierr);

  ierr = DMLocalToGlobalBegin(networkdm,localF,ADD_VALUES,F);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(networkdm,localF,ADD_VALUES,F);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(networkdm,&localF);CHKERRQ(ierr);
  /* ierr = VecView(F,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); */
  PetscFunctionReturn(0);
}

PetscErrorCode SetInitialGuess(DM networkdm,Vec X,void* appctx)
{
  PetscErrorCode ierr;
  PetscInt       nv,ne;
  const PetscInt *vtx,*edges;
  UserCtx        *user = (UserCtx*)appctx;
  Vec            localX = user->localXold;
  UserCtx_Power  appctx_power = (*user).appctx_power;

  PetscFunctionBegin;
  ierr = VecSet(X,0.0);CHKERRQ(ierr);
  ierr = VecSet(localX,0.0);CHKERRQ(ierr);

  /* Set initial guess for power subnetwork */
  ierr = DMNetworkGetSubnetworkInfo(networkdm,0,&nv,&ne,&vtx,&edges);CHKERRQ(ierr);
  ierr = SetInitialGuess_Power(networkdm,localX,nv,ne,vtx,edges,&appctx_power);CHKERRQ(ierr);

  /* Set initial guess for water subnetwork */
  ierr = DMNetworkGetSubnetworkInfo(networkdm,1,&nv,&ne,&vtx,&edges);CHKERRQ(ierr);
  ierr = SetInitialGuess_Water(networkdm,localX,nv,ne,vtx,edges,NULL);CHKERRQ(ierr);

#if 0
  /* Set initial guess for the coupling subnet */
  ierr = DMNetworkGetSubnetworkInfo(networkdm,2,&nv,&ne,&vtx,&edges);CHKERRQ(ierr);
  if (ne) {
    const PetscInt *cone;
    ierr = DMNetworkGetConnectedVertices(networkdm,edges[0],&cone);CHKERRQ(ierr);
  }
#endif

  ierr = DMLocalToGlobalBegin(networkdm,localX,ADD_VALUES,X);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(networkdm,localX,ADD_VALUES,X);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  PetscErrorCode   ierr;
  DM               networkdm;
  PetscLogStage    stage[4];
  PetscMPIInt      rank,size;
  PetscInt         nsubnet=2,nsubnetCouple=0,numVertices[2],numEdges[2],numEdgesCouple[1];
  PetscInt         i,j,nv,ne;
  PetscInt         *edgelist[2];
  const PetscInt   *vtx,*edges;
  Vec              X,F;
  SNES             snes,snes_power,snes_water;
  Mat              Jac;
  PetscBool        viewJ=PETSC_FALSE,viewX=PETSC_FALSE,viewDM=PETSC_FALSE,test=PETSC_FALSE,distribute=PETSC_TRUE;
  UserCtx          user;
  PetscInt         it_max=10;
  SNESConvergedReason reason;

  /* Power subnetwork */
  UserCtx_Power    *appctx_power = &user.appctx_power;
  char             pfdata_file[PETSC_MAX_PATH_LEN]="power/case9.m";
  PFDATA           *pfdata=NULL;
  PetscInt         genj,loadj;
  PetscInt         *edgelist_power=NULL;
  PetscScalar      Sbase=0.0;

  /* Water subnetwork */
  AppCtx_Water     *appctx_water = &user.appctx_water;
  WATERDATA        *waterdata=NULL;
  char             waterdata_file[PETSC_MAX_PATH_LEN]="water/sample1.inp";
  PetscInt         *edgelist_water=NULL;

  /* Coupling subnetwork */
  PetscInt         *edgelist_couple=NULL;

  ierr = PetscInitialize(&argc,&argv,"ex1options",help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);

  /* (1) Read Data - Only rank 0 reads the data */
  /*--------------------------------------------*/
  ierr = PetscLogStageRegister("Read Data",&stage[0]);CHKERRQ(ierr);
  PetscLogStagePush(stage[0]);

  for (i=0; i<nsubnet; i++) {
    numVertices[i] = 0;
    numEdges[i]    = 0;
  }
  numEdgesCouple[0] = 0;

  /* proc[0] READ THE DATA FOR THE FIRST SUBNETWORK: Electric Power Grid */
  if (rank == 0) {
    ierr = PetscOptionsGetString(NULL,NULL,"-pfdata",pfdata_file,sizeof(pfdata_file),NULL);CHKERRQ(ierr);
    ierr = PetscNew(&pfdata);CHKERRQ(ierr);
    ierr = PFReadMatPowerData(pfdata,pfdata_file);CHKERRQ(ierr);
    Sbase = pfdata->sbase;

    numEdges[0]    = pfdata->nbranch;
    numVertices[0] = pfdata->nbus;

    ierr = PetscMalloc1(2*numEdges[0],&edgelist_power);CHKERRQ(ierr);
    ierr = GetListofEdges_Power(pfdata,edgelist_power);CHKERRQ(ierr);
  }
  /* Broadcast power Sbase to all processors */
  ierr = MPI_Bcast(&Sbase,1,MPIU_SCALAR,0,PETSC_COMM_WORLD);CHKERRQ(ierr);
  appctx_power->Sbase = Sbase;
  appctx_power->jac_error = PETSC_FALSE;
  /* If external option activated. Introduce error in jacobian */
  ierr = PetscOptionsHasName(NULL,NULL, "-jac_error", &appctx_power->jac_error);CHKERRQ(ierr);

  /* proc[1] GET DATA FOR THE SECOND SUBNETWORK: Water */
  if (size == 1 || (size > 1 && rank == 1)) {
    ierr = PetscNew(&waterdata);CHKERRQ(ierr);
    ierr = PetscOptionsGetString(NULL,NULL,"-waterdata",waterdata_file,sizeof(waterdata_file),NULL);CHKERRQ(ierr);
    ierr = WaterReadData(waterdata,waterdata_file);CHKERRQ(ierr);

    ierr = PetscCalloc1(2*waterdata->nedge,&edgelist_water);CHKERRQ(ierr);
    ierr = GetListofEdges_Water(waterdata,edgelist_water);CHKERRQ(ierr);
    numEdges[1]    = waterdata->nedge;
    numVertices[1] = waterdata->nvertex;
  }

  /* Get data for the coupling subnetwork */
  if (size == 1) { /* TODO: for size > 1, parallel processing coupling is buggy */
    nsubnetCouple = 1;
    numEdgesCouple[0] = 1;

    ierr = PetscMalloc1(4*numEdgesCouple[0],&edgelist_couple);CHKERRQ(ierr);
    edgelist_couple[0] = 0; edgelist_couple[1] = 4; /* from node: net[0] vertex[4] */
    edgelist_couple[2] = 1; edgelist_couple[3] = 0; /* to node:   net[1] vertex[0] */
  }
  PetscLogStagePop();

  /* (2) Create network */
  /*--------------------*/
  ierr = MPI_Barrier(PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = PetscLogStageRegister("Net Setup",&stage[1]);CHKERRQ(ierr);
  PetscLogStagePush(stage[1]);

  ierr = PetscOptionsGetBool(NULL,NULL,"-viewDM",&viewDM,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-test",&test,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-distribute",&distribute,NULL);CHKERRQ(ierr);

  /* Create an empty network object */
  ierr = DMNetworkCreate(PETSC_COMM_WORLD,&networkdm);CHKERRQ(ierr);

  /* Register the components in the network */
  ierr = DMNetworkRegisterComponent(networkdm,"branchstruct",sizeof(struct _p_EDGE_Power),&appctx_power->compkey_branch);CHKERRQ(ierr);
  ierr = DMNetworkRegisterComponent(networkdm,"busstruct",sizeof(struct _p_VERTEX_Power),&appctx_power->compkey_bus);CHKERRQ(ierr);
  ierr = DMNetworkRegisterComponent(networkdm,"genstruct",sizeof(struct _p_GEN),&appctx_power->compkey_gen);CHKERRQ(ierr);
  ierr = DMNetworkRegisterComponent(networkdm,"loadstruct",sizeof(struct _p_LOAD),&appctx_power->compkey_load);CHKERRQ(ierr);

  ierr = DMNetworkRegisterComponent(networkdm,"edge_water",sizeof(struct _p_EDGE_Water),&appctx_water->compkey_edge);CHKERRQ(ierr);
  ierr = DMNetworkRegisterComponent(networkdm,"vertex_water",sizeof(struct _p_VERTEX_Water),&appctx_water->compkey_vtx);CHKERRQ(ierr);

  ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d] Total local nvertices %D + %D = %D, nedges %D + %D + %D = %D\n",rank,numVertices[0],numVertices[1],numVertices[0]+numVertices[1],numEdges[0],numEdges[1],numEdgesCouple[0],numEdges[0]+numEdges[1]+numEdgesCouple[0]);CHKERRQ(ierr);
  ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);CHKERRQ(ierr);

  ierr = DMNetworkSetSizes(networkdm,nsubnet,numVertices,numEdges,nsubnetCouple,numEdgesCouple);CHKERRQ(ierr);

  /* Add edge connectivity */
  edgelist[0] = edgelist_power;
  edgelist[1] = edgelist_water;
  ierr = DMNetworkSetEdgeList(networkdm,edgelist,&edgelist_couple);CHKERRQ(ierr);

  /* Set up the network layout */
  ierr = DMNetworkLayoutSetUp(networkdm);CHKERRQ(ierr);

  /* Add network components - only process[0] has any data to add */
  /* ADD VARIABLES AND COMPONENTS FOR THE POWER SUBNETWORK */
  genj = 0; loadj = 0;
  if (rank == 0) {
    ierr = DMNetworkGetSubnetworkInfo(networkdm,0,&nv,&ne,&vtx,&edges);CHKERRQ(ierr);
    /* ierr = PetscPrintf(PETSC_COMM_SELF,"Power network: nv %D, ne %D\n",nv,ne);CHKERRQ(ierr); */
    for (i = 0; i < ne; i++) {
      ierr = DMNetworkAddComponent(networkdm,edges[i],appctx_power->compkey_branch,&pfdata->branch[i]);CHKERRQ(ierr);
    }

    for (i = 0; i < nv; i++) {
      ierr = DMNetworkAddComponent(networkdm,vtx[i],appctx_power->compkey_bus,&pfdata->bus[i]);CHKERRQ(ierr);
      if (pfdata->bus[i].ngen) {
        for (j = 0; j < pfdata->bus[i].ngen; j++) {
          ierr = DMNetworkAddComponent(networkdm,vtx[i],appctx_power->compkey_gen,&pfdata->gen[genj++]);CHKERRQ(ierr);
        }
      }
      if (pfdata->bus[i].nload) {
        for (j=0; j < pfdata->bus[i].nload; j++) {
          ierr = DMNetworkAddComponent(networkdm,vtx[i],appctx_power->compkey_load,&pfdata->load[loadj++]);CHKERRQ(ierr);
        }
      }
      /* Add number of variables */
      ierr = DMNetworkAddNumVariables(networkdm,vtx[i],2);CHKERRQ(ierr);
    }
  }

  /* ADD VARIABLES AND COMPONENTS FOR THE WATER SUBNETWORK */
  if (size == 1 || (size > 1 && rank == 1)) {
    ierr = DMNetworkGetSubnetworkInfo(networkdm,1,&nv,&ne,&vtx,&edges);CHKERRQ(ierr);
    /* ierr = PetscPrintf(PETSC_COMM_SELF,"Water network: nv %D, ne %D\n",nv,ne);CHKERRQ(ierr); */
    for (i = 0; i < ne; i++) {
      ierr = DMNetworkAddComponent(networkdm,edges[i],appctx_water->compkey_edge,&waterdata->edge[i]);CHKERRQ(ierr);
    }

    for (i = 0; i < nv; i++) {
      ierr = DMNetworkAddComponent(networkdm,vtx[i],appctx_water->compkey_vtx,&waterdata->vertex[i]);CHKERRQ(ierr);
      /* Add number of variables */
      ierr = DMNetworkAddNumVariables(networkdm,vtx[i],1);CHKERRQ(ierr);
    }
  }

  /* Set up DM for use */
  ierr = DMSetUp(networkdm);CHKERRQ(ierr);
  if (viewDM) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\nAfter DMSetUp, DMView:\n");CHKERRQ(ierr);
    ierr = DMView(networkdm,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  ierr = DMNetworkGetSubnetworkCoupleInfo(networkdm,0,&ne,&edges);CHKERRQ(ierr);
  if (ne && viewDM) {
    const PetscInt *cone;
    PetscInt       vid[2];
    ierr = DMNetworkGetConnectedVertices(networkdm,edges[0],&cone);CHKERRQ(ierr);

    ierr = DMNetworkGetGlobalVertexIndex(networkdm,cone[0],&vid[0]);CHKERRQ(ierr);
    ierr = DMNetworkGetGlobalVertexIndex(networkdm,cone[1],&vid[1]);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] Coupling subnetwork: ne %d; connected vertices %d %d\n",rank,ne,vid[0],vid[1]);CHKERRQ(ierr);
  }

  /* Free user objects */
  if (!rank) {
    ierr = PetscFree(edgelist_power);CHKERRQ(ierr);
    ierr = PetscFree(pfdata->bus);CHKERRQ(ierr);
    ierr = PetscFree(pfdata->gen);CHKERRQ(ierr);
    ierr = PetscFree(pfdata->branch);CHKERRQ(ierr);
    ierr = PetscFree(pfdata->load);CHKERRQ(ierr);
    ierr = PetscFree(pfdata);CHKERRQ(ierr);
  }
  if (size == 1 || (size > 1 && rank == 1)) {
    ierr = PetscFree(edgelist_water);CHKERRQ(ierr);
    ierr = PetscFree(waterdata->vertex);CHKERRQ(ierr);
    ierr = PetscFree(waterdata->edge);CHKERRQ(ierr);
    ierr = PetscFree(waterdata);CHKERRQ(ierr);
  }
  if (size == 1) {
    ierr = PetscFree(edgelist_couple);CHKERRQ(ierr);
  }

  /* Re-distribute networkdm to multiple processes for better job balance */
  if (distribute) {
    ierr = DMNetworkDistribute(&networkdm,0);CHKERRQ(ierr);
    if (viewDM) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"\nAfter DMNetworkDistribute, DMView:\n");CHKERRQ(ierr);
      ierr = DMView(networkdm,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    }
  }

  /* Test DMNetworkGetSubnetworkCoupleInfo() */
  ierr = MPI_Barrier(PETSC_COMM_WORLD);CHKERRQ(ierr);
  if (test) {
    for (i=0; i<nsubnetCouple; i++) {
      ierr = DMNetworkGetSubnetworkCoupleInfo(networkdm,0,&ne,&edges);CHKERRQ(ierr);
      if (ne && viewDM) {
        const PetscInt *cone;
        PetscInt       vid[2];
        ierr = DMNetworkGetConnectedVertices(networkdm,edges[0],&cone);CHKERRQ(ierr);

        ierr = DMNetworkGetGlobalVertexIndex(networkdm,cone[0],&vid[0]);CHKERRQ(ierr);
        ierr = DMNetworkGetGlobalVertexIndex(networkdm,cone[1],&vid[1]);CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] After DMNetworkDistribute(), subnetworkCouple[%d]: ne %d; connected vertices %d %d\n",rank,i,ne,vid[0],vid[1]);CHKERRQ(ierr);
      }
    }
  }

  ierr = DMCreateGlobalVector(networkdm,&X);CHKERRQ(ierr);
  ierr = VecDuplicate(X,&F);CHKERRQ(ierr);
  ierr = DMGetLocalVector(networkdm,&user.localXold);CHKERRQ(ierr);

  PetscLogStagePop();

  /* (3) Setup Solvers */
  /*-------------------*/
  ierr = PetscOptionsGetBool(NULL,NULL,"-viewJ",&viewJ,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-viewX",&viewX,NULL);CHKERRQ(ierr);

  ierr = PetscLogStageRegister("SNES Setup",&stage[2]);CHKERRQ(ierr);
  PetscLogStagePush(stage[2]);

  ierr = SetInitialGuess(networkdm,X,&user);CHKERRQ(ierr);
  /* ierr = VecView(X,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); */

  /* Create coupled snes */
  /*-------------------- */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"SNES_coupled setup ......\n");CHKERRQ(ierr);
  user.subsnes_id = nsubnet;
  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
  ierr = SNESSetDM(snes,networkdm);CHKERRQ(ierr);
  ierr = SNESSetOptionsPrefix(snes,"coupled_");CHKERRQ(ierr);
  ierr = SNESSetFunction(snes,F,FormFunction,&user);CHKERRQ(ierr);
  ierr = SNESMonitorSet(snes,UserMonitor,&user,NULL);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  if (viewJ) {
    /* View Jac structure */
    ierr = SNESGetJacobian(snes,&Jac,NULL,NULL,NULL);CHKERRQ(ierr);
    ierr = MatView(Jac,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
  }

  if (viewX) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Solution:\n");CHKERRQ(ierr);
    ierr = VecView(X,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  if (viewJ) {
    /* View assembled Jac */
    ierr = MatView(Jac,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
  }

  /* Create snes_power */
  /*-------------------*/
  ierr = PetscPrintf(PETSC_COMM_WORLD,"SNES_power setup ......\n");CHKERRQ(ierr);
  ierr = SetInitialGuess(networkdm,X,&user);CHKERRQ(ierr);

  user.subsnes_id = 0;
  ierr = SNESCreate(PETSC_COMM_WORLD,&snes_power);CHKERRQ(ierr);
  ierr = SNESSetDM(snes_power,networkdm);CHKERRQ(ierr);
  ierr = SNESSetOptionsPrefix(snes_power,"power_");CHKERRQ(ierr);
  ierr = SNESSetFunction(snes_power,F,FormFunction,&user);CHKERRQ(ierr);
  ierr = SNESMonitorSet(snes_power,UserMonitor,&user,NULL);CHKERRQ(ierr);

  /* Use user-provide Jacobian */
  ierr = DMCreateMatrix(networkdm,&Jac);CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes_power,Jac,Jac,FormJacobian_subPower,&user);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes_power);CHKERRQ(ierr);

  if (viewX) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Power Solution:\n");CHKERRQ(ierr);
    ierr = VecView(X,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  if (viewJ) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Power Jac:\n");CHKERRQ(ierr);
    ierr = SNESGetJacobian(snes_power,&Jac,NULL,NULL,NULL);CHKERRQ(ierr);
    ierr = MatView(Jac,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
    /* ierr = MatView(Jac,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); */
  }

  /* Create snes_water */
  /*-------------------*/
  ierr = PetscPrintf(PETSC_COMM_WORLD,"SNES_water setup......\n");CHKERRQ(ierr);
  ierr = SetInitialGuess(networkdm,X,&user);CHKERRQ(ierr);

  user.subsnes_id = 1;
  ierr = SNESCreate(PETSC_COMM_WORLD,&snes_water);CHKERRQ(ierr);
  ierr = SNESSetDM(snes_water,networkdm);CHKERRQ(ierr);
  ierr = SNESSetOptionsPrefix(snes_water,"water_");CHKERRQ(ierr);
  ierr = SNESSetFunction(snes_water,F,FormFunction,&user);CHKERRQ(ierr);
  ierr = SNESMonitorSet(snes_water,UserMonitor,&user,NULL);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes_water);CHKERRQ(ierr);

  if (viewX) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Water Solution:\n");CHKERRQ(ierr);
    ierr = VecView(X,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  PetscLogStagePop();

  /* (4) Solve */
  /*-----------*/
  ierr = PetscLogStageRegister("SNES Solve",&stage[3]);CHKERRQ(ierr);
  PetscLogStagePush(stage[3]);
  user.it = 0;
  reason  = SNES_DIVERGED_DTOL;
  while (user.it < it_max && (PetscInt)reason<0) {
#if 0
    user.subsnes_id = 0;
    ierr = SNESSolve(snes_power,NULL,X);CHKERRQ(ierr);

    user.subsnes_id = 1;
    ierr = SNESSolve(snes_water,NULL,X);CHKERRQ(ierr);
#endif
    user.subsnes_id = nsubnet;
    ierr = SNESSolve(snes,NULL,X);CHKERRQ(ierr);

    ierr = SNESGetConvergedReason(snes,&reason);CHKERRQ(ierr);
    user.it++;
  }
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Coupled_SNES converged in %D iterations\n",user.it);CHKERRQ(ierr);
  if (viewX) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Final Solution:\n");CHKERRQ(ierr);
    ierr = VecView(X,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
   }
  PetscLogStagePop();

  /* Free objects */
  /* -------------*/
  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = VecDestroy(&F);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(networkdm,&user.localXold);CHKERRQ(ierr);

  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = MatDestroy(&Jac);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes_power);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes_water);CHKERRQ(ierr);

  ierr = DMDestroy(&networkdm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
     requires: !complex double define(PETSC_HAVE_ATTRIBUTEALIGNED)
     depends: power/PFReadData.c power/pffunctions.c water/waterreaddata.c water/waterfunctions.c

   test:
      args: -coupled_snes_converged_reason -options_left no
      localrunfiles: ex1options power/case9.m water/sample1.inp
      output_file: output/ex1.out
      requires: double !complex define(PETSC_HAVE_ATTRIBUTEALIGNED)

   test:
      suffix: 2
      nsize: 3
      args: -coupled_snes_converged_reason -options_left no
      localrunfiles: ex1options power/case9.m water/sample1.inp
      output_file: output/ex1_2.out
      requires: double !complex define(PETSC_HAVE_ATTRIBUTEALIGNED)

   test:
      suffix: 3
      nsize: 3
      args: -coupled_snes_converged_reason -options_left no -distribute false
      localrunfiles: ex1options power/case9.m water/sample1.inp
      output_file: output/ex1_2.out
      requires: double !complex define(PETSC_HAVE_ATTRIBUTEALIGNED)

TEST*/
