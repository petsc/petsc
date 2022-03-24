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
  UserCtx        *user = (UserCtx*)appctx;
  Vec            X,localXold = user->localXold;
  DM             networkdm;
  PetscMPIInt    rank;
  MPI_Comm       comm;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)snes,&comm));
  CHKERRMPI(MPI_Comm_rank(comm,&rank));
#if 0
  if (rank == 0) {
    PetscInt       subsnes_id = user->subsnes_id;
    if (subsnes_id == 2) {
      CHKERRQ(PetscPrintf(PETSC_COMM_SELF," it %D, subsnes_id %D, fnorm %g\n",user->it,user->subsnes_id,(double)fnorm));
    } else {
      CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"       subsnes_id %D, fnorm %g\n",user->subsnes_id,(double)fnorm));
    }
  }
#endif
  CHKERRQ(SNESGetSolution(snes,&X));
  CHKERRQ(SNESGetDM(snes,&networkdm));
  CHKERRQ(DMGlobalToLocalBegin(networkdm,X,INSERT_VALUES,localXold));
  CHKERRQ(DMGlobalToLocalEnd(networkdm,X,INSERT_VALUES,localXold));
  PetscFunctionReturn(0);
}

PetscErrorCode FormJacobian_subPower(SNES snes,Vec X, Mat J,Mat Jpre,void *appctx)
{
  DM             networkdm;
  Vec            localX;
  PetscInt       nv,ne,i,j,offset,nvar,row;
  const PetscInt *vtx,*edges;
  PetscBool      ghostvtex;
  PetscScalar    one = 1.0;
  PetscMPIInt    rank;
  MPI_Comm       comm;

  PetscFunctionBegin;
  CHKERRQ(SNESGetDM(snes,&networkdm));
  CHKERRQ(DMGetLocalVector(networkdm,&localX));

  CHKERRQ(PetscObjectGetComm((PetscObject)networkdm,&comm));
  CHKERRMPI(MPI_Comm_rank(comm,&rank));

  CHKERRQ(DMGlobalToLocalBegin(networkdm,X,INSERT_VALUES,localX));
  CHKERRQ(DMGlobalToLocalEnd(networkdm,X,INSERT_VALUES,localX));

  CHKERRQ(MatZeroEntries(J));

  /* Power subnetwork: copied from power/FormJacobian_Power() */
  CHKERRQ(DMNetworkGetSubnetwork(networkdm,0,&nv,&ne,&vtx,&edges));
  CHKERRQ(FormJacobian_Power_private(networkdm,localX,J,nv,ne,vtx,edges,appctx));

  /* Water subnetwork: Identity */
  CHKERRQ(DMNetworkGetSubnetwork(networkdm,1,&nv,&ne,&vtx,&edges));
  for (i=0; i<nv; i++) {
    CHKERRQ(DMNetworkIsGhostVertex(networkdm,vtx[i],&ghostvtex));
    if (ghostvtex) continue;

    CHKERRQ(DMNetworkGetGlobalVecOffset(networkdm,vtx[i],ALL_COMPONENTS,&offset));
    CHKERRQ(DMNetworkGetComponent(networkdm,vtx[i],ALL_COMPONENTS,NULL,NULL,&nvar));
    for (j=0; j<nvar; j++) {
      row = offset + j;
      CHKERRQ(MatSetValues(J,1,&row,1,&row,&one,ADD_VALUES));
    }
  }
  CHKERRQ(MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY));

  CHKERRQ(DMRestoreLocalVector(networkdm,&localX));
  PetscFunctionReturn(0);
}

/* Dummy equation localF(X) = localX - localXold */
PetscErrorCode FormFunction_Dummy(DM networkdm,Vec localX, Vec localF,PetscInt nv,PetscInt ne,const PetscInt* vtx,const PetscInt* edges,void* appctx)
{
  const PetscScalar *xarr,*xoldarr;
  PetscScalar       *farr;
  PetscInt          i,j,offset,nvar;
  PetscBool         ghostvtex;
  UserCtx           *user = (UserCtx*)appctx;
  Vec               localXold = user->localXold;

  PetscFunctionBegin;
  CHKERRQ(VecGetArrayRead(localX,&xarr));
  CHKERRQ(VecGetArrayRead(localXold,&xoldarr));
  CHKERRQ(VecGetArray(localF,&farr));

  for (i=0; i<nv; i++) {
    CHKERRQ(DMNetworkIsGhostVertex(networkdm,vtx[i],&ghostvtex));
    if (ghostvtex) continue;

    CHKERRQ(DMNetworkGetLocalVecOffset(networkdm,vtx[i],ALL_COMPONENTS,&offset));
    CHKERRQ(DMNetworkGetComponent(networkdm,vtx[i],ALL_COMPONENTS,NULL,NULL,&nvar));
    for (j=0; j<nvar; j++) {
      farr[offset+j] = xarr[offset+j] - xoldarr[offset+j];
    }
  }

  CHKERRQ(VecRestoreArrayRead(localX,&xarr));
  CHKERRQ(VecRestoreArrayRead(localXold,&xoldarr));
  CHKERRQ(VecRestoreArray(localF,&farr));
  PetscFunctionReturn(0);
}

PetscErrorCode FormFunction(SNES snes,Vec X,Vec F,void *appctx)
{
  DM             networkdm;
  Vec            localX,localF;
  PetscInt       nv,ne,v;
  const PetscInt *vtx,*edges;
  PetscMPIInt    rank;
  MPI_Comm       comm;
  UserCtx        *user = (UserCtx*)appctx;
  UserCtx_Power  appctx_power = (*user).appctx_power;
  AppCtx_Water   appctx_water = (*user).appctx_water;

  PetscFunctionBegin;
  CHKERRQ(SNESGetDM(snes,&networkdm));
  CHKERRQ(PetscObjectGetComm((PetscObject)networkdm,&comm));
  CHKERRMPI(MPI_Comm_rank(comm,&rank));

  CHKERRQ(DMGetLocalVector(networkdm,&localX));
  CHKERRQ(DMGetLocalVector(networkdm,&localF));
  CHKERRQ(VecSet(F,0.0));
  CHKERRQ(VecSet(localF,0.0));

  CHKERRQ(DMGlobalToLocalBegin(networkdm,X,INSERT_VALUES,localX));
  CHKERRQ(DMGlobalToLocalEnd(networkdm,X,INSERT_VALUES,localX));

  /* Form Function for power subnetwork */
  CHKERRQ(DMNetworkGetSubnetwork(networkdm,0,&nv,&ne,&vtx,&edges));
  if (user->subsnes_id == 1) { /* snes_water only */
    CHKERRQ(FormFunction_Dummy(networkdm,localX,localF,nv,ne,vtx,edges,user));
  } else {
    CHKERRQ(FormFunction_Power(networkdm,localX,localF,nv,ne,vtx,edges,&appctx_power));
  }

  /* Form Function for water subnetwork */
  CHKERRQ(DMNetworkGetSubnetwork(networkdm,1,&nv,&ne,&vtx,&edges));
  if (user->subsnes_id == 0) { /* snes_power only */
    CHKERRQ(FormFunction_Dummy(networkdm,localX,localF,nv,ne,vtx,edges,user));
  } else {
    CHKERRQ(FormFunction_Water(networkdm,localX,localF,nv,ne,vtx,edges,NULL));
  }

  /* Illustrate how to access the coupling vertex of the subnetworks without doing anything to F yet */
  CHKERRQ(DMNetworkGetSharedVertices(networkdm,&nv,&vtx));
  for (v=0; v<nv; v++) {
    PetscInt       key,ncomp,nvar,nconnedges,k,e,keye,goffset[3];
    void*          component;
    const PetscInt *connedges;

    CHKERRQ(DMNetworkGetComponent(networkdm,vtx[v],ALL_COMPONENTS,NULL,NULL,&nvar));
    CHKERRQ(DMNetworkGetNumComponents(networkdm,vtx[v],&ncomp));
    /* printf("  [%d] coupling vertex[%D]: v %D, ncomp %D; nvar %D\n",rank,v,vtx[v], ncomp,nvar); */

    for (k=0; k<ncomp; k++) {
      CHKERRQ(DMNetworkGetComponent(networkdm,vtx[v],k,&key,&component,&nvar));
      CHKERRQ(DMNetworkGetGlobalVecOffset(networkdm,vtx[v],k,&goffset[k]));

      /* Verify the coupling vertex is a powernet load vertex or a water vertex */
      switch (k) {
      case 0:
        PetscCheckFalse(key != appctx_power.compkey_bus || nvar != 2,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"key %D not a power bus vertex or nvar %D != 2",key,nvar);
        break;
      case 1:
        PetscCheckFalse(key != appctx_power.compkey_load || nvar != 0 || goffset[1] != goffset[0]+2,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Not a power load vertex");
        break;
      case 2:
        PetscCheckFalse(key != appctx_water.compkey_vtx || nvar != 1 || goffset[2] != goffset[1],PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Not a water vertex");
        break;
      default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "k %D is wrong",k);
      }
      /* printf("  [%d] coupling vertex[%D]: key %D; nvar %D, goffset %D\n",rank,v,key,nvar,goffset[k]); */
    }

    /* Get its supporting edges */
    CHKERRQ(DMNetworkGetSupportingEdges(networkdm,vtx[v],&nconnedges,&connedges));
    /* printf("\n[%d] coupling vertex: nconnedges %D\n",rank,nconnedges);CHKERRQ(ierr); */
    for (k=0; k<nconnedges; k++) {
      e = connedges[k];
      CHKERRQ(DMNetworkGetNumComponents(networkdm,e,&ncomp));
      /* printf("\n  [%d] connected edge[%D]=%D has ncomp %D\n",rank,k,e,ncomp); */
      CHKERRQ(DMNetworkGetComponent(networkdm,e,0,&keye,&component,NULL));
      if (keye == appctx_water.compkey_edge) { /* water_compkey_edge */
        EDGE_Water        edge=(EDGE_Water)component;
        if (edge->type == EDGE_TYPE_PUMP) {
          /* printf("  connected edge[%D]=%D has keye=%D, is appctx_water.compkey_edge with EDGE_TYPE_PUMP\n",k,e,keye); */
        }
      } else { /* ower->compkey_branch */
        PetscCheckFalse(keye != appctx_power.compkey_branch,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Not a power branch");
      }
    }
  }

  CHKERRQ(DMRestoreLocalVector(networkdm,&localX));

  CHKERRQ(DMLocalToGlobalBegin(networkdm,localF,ADD_VALUES,F));
  CHKERRQ(DMLocalToGlobalEnd(networkdm,localF,ADD_VALUES,F));
  CHKERRQ(DMRestoreLocalVector(networkdm,&localF));
#if 0
  if (rank == 0) printf("F:\n");
  CHKERRQ(VecView(F,PETSC_VIEWER_STDOUT_WORLD));
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode SetInitialGuess(DM networkdm,Vec X,void* appctx)
{
  PetscInt       nv,ne,i,j,ncomp,offset,key;
  const PetscInt *vtx,*edges;
  UserCtx        *user = (UserCtx*)appctx;
  Vec            localX = user->localXold;
  UserCtx_Power  appctx_power = (*user).appctx_power;
  AppCtx_Water   appctx_water = (*user).appctx_water;
  PetscBool      ghost;
  PetscScalar    *xarr;
  VERTEX_Power   bus;
  VERTEX_Water   vertex;
  void*          component;
  GEN            gen;

  PetscFunctionBegin;
  CHKERRQ(VecSet(X,0.0));
  CHKERRQ(VecSet(localX,0.0));

  /* Set initial guess for power subnetwork */
  CHKERRQ(DMNetworkGetSubnetwork(networkdm,0,&nv,&ne,&vtx,&edges));
  CHKERRQ(SetInitialGuess_Power(networkdm,localX,nv,ne,vtx,edges,&appctx_power));

  /* Set initial guess for water subnetwork */
  CHKERRQ(DMNetworkGetSubnetwork(networkdm,1,&nv,&ne,&vtx,&edges));
  CHKERRQ(SetInitialGuess_Water(networkdm,localX,nv,ne,vtx,edges,NULL));

  /* Set initial guess at the coupling vertex */
  CHKERRQ(VecGetArray(localX,&xarr));
  CHKERRQ(DMNetworkGetSharedVertices(networkdm,&nv,&vtx));
  for (i=0; i<nv; i++) {
    CHKERRQ(DMNetworkIsGhostVertex(networkdm,vtx[i],&ghost));
    if (ghost) continue;

    CHKERRQ(DMNetworkGetNumComponents(networkdm,vtx[i],&ncomp));
    for (j=0; j<ncomp; j++) {
      CHKERRQ(DMNetworkGetLocalVecOffset(networkdm,vtx[i],j,&offset));
      CHKERRQ(DMNetworkGetComponent(networkdm,vtx[i],j,&key,(void**)&component,NULL));
      if (key == appctx_power.compkey_bus) {
        bus = (VERTEX_Power)(component);
        xarr[offset]   = bus->va*PETSC_PI/180.0;
        xarr[offset+1] = bus->vm;
      } else if (key == appctx_power.compkey_gen) {
        gen = (GEN)(component);
        if (!gen->status) continue;
        xarr[offset+1] = gen->vs;
      } else if (key == appctx_water.compkey_vtx) {
        vertex = (VERTEX_Water)(component);
        if (vertex->type == VERTEX_TYPE_JUNCTION) {
          xarr[offset] = 100;
        } else if (vertex->type == VERTEX_TYPE_RESERVOIR) {
          xarr[offset] = vertex->res.head;
        } else {
          xarr[offset] = vertex->tank.initlvl + vertex->tank.elev;
        }
      }
    }
  }
  CHKERRQ(VecRestoreArray(localX,&xarr));

  CHKERRQ(DMLocalToGlobalBegin(networkdm,localX,ADD_VALUES,X));
  CHKERRQ(DMLocalToGlobalEnd(networkdm,localX,ADD_VALUES,X));
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  DM               networkdm;
  PetscLogStage    stage[4];
  PetscMPIInt      rank,size;
  PetscInt         Nsubnet=2,numVertices[2],numEdges[2],i,j,nv,ne,it_max=10;
  const PetscInt   *vtx,*edges;
  Vec              X,F;
  SNES             snes,snes_power,snes_water;
  Mat              Jac;
  PetscBool        ghost,viewJ=PETSC_FALSE,viewX=PETSC_FALSE,viewDM=PETSC_FALSE,test=PETSC_FALSE,distribute=PETSC_TRUE,flg;
  UserCtx          user;
  SNESConvergedReason reason;

  /* Power subnetwork */
  UserCtx_Power       *appctx_power  = &user.appctx_power;
  char                pfdata_file[PETSC_MAX_PATH_LEN] = "power/case9.m";
  PFDATA              *pfdata = NULL;
  PetscInt            genj,loadj,*edgelist_power = NULL,power_netnum;
  PetscScalar         Sbase = 0.0;

  /* Water subnetwork */
  AppCtx_Water        *appctx_water = &user.appctx_water;
  WATERDATA           *waterdata = NULL;
  char                waterdata_file[PETSC_MAX_PATH_LEN] = "water/sample1.inp";
  PetscInt            *edgelist_water = NULL,water_netnum;

  /* Shared vertices between subnetworks */
  PetscInt           power_svtx,water_svtx;

  CHKERRQ(PetscInitialize(&argc,&argv,"ex1options",help));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));

  /* (1) Read Data - Only rank 0 reads the data */
  /*--------------------------------------------*/
  CHKERRQ(PetscLogStageRegister("Read Data",&stage[0]));
  CHKERRQ(PetscLogStagePush(stage[0]));

  for (i=0; i<Nsubnet; i++) {
    numVertices[i] = 0;
    numEdges[i]    = 0;
  }

  /* All processes READ THE DATA FOR THE FIRST SUBNETWORK: Electric Power Grid */
  /* Used for shared vertex, because currently the coupling info must be available in all processes!!! */
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-pfdata",pfdata_file,PETSC_MAX_PATH_LEN-1,NULL));
  CHKERRQ(PetscNew(&pfdata));
  CHKERRQ(PFReadMatPowerData(pfdata,pfdata_file));
  if (rank == 0) {
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Power network: nb = %D, ngen = %D, nload = %D, nbranch = %D\n",pfdata->nbus,pfdata->ngen,pfdata->nload,pfdata->nbranch));
  }
  Sbase = pfdata->sbase;
  if (rank == 0) { /* proc[0] will create Electric Power Grid */
    numEdges[0]    = pfdata->nbranch;
    numVertices[0] = pfdata->nbus;

    CHKERRQ(PetscMalloc1(2*numEdges[0],&edgelist_power));
    CHKERRQ(GetListofEdges_Power(pfdata,edgelist_power));
  }
  /* Broadcast power Sbase to all processors */
  CHKERRMPI(MPI_Bcast(&Sbase,1,MPIU_SCALAR,0,PETSC_COMM_WORLD));
  appctx_power->Sbase     = Sbase;
  appctx_power->jac_error = PETSC_FALSE;
  /* If external option activated. Introduce error in jacobian */
  CHKERRQ(PetscOptionsHasName(NULL,NULL, "-jac_error", &appctx_power->jac_error));

  /* All processes READ THE DATA FOR THE SECOND SUBNETWORK: Water */
  /* Used for shared vertex, because currently the coupling info must be available in all processes!!! */
  CHKERRQ(PetscNew(&waterdata));
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-waterdata",waterdata_file,PETSC_MAX_PATH_LEN-1,NULL));
  CHKERRQ(WaterReadData(waterdata,waterdata_file));
  if (size == 1 || (size > 1 && rank == 1)) {
    CHKERRQ(PetscCalloc1(2*waterdata->nedge,&edgelist_water));
    CHKERRQ(GetListofEdges_Water(waterdata,edgelist_water));
    numEdges[1]    = waterdata->nedge;
    numVertices[1] = waterdata->nvertex;
  }
  PetscLogStagePop();

  /* (2) Create a network consist of two subnetworks */
  /*-------------------------------------------------*/
  CHKERRQ(PetscLogStageRegister("Net Setup",&stage[1]));
  CHKERRQ(PetscLogStagePush(stage[1]));

  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-viewDM",&viewDM,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-test",&test,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-distribute",&distribute,NULL));

  /* Create an empty network object */
  CHKERRQ(DMNetworkCreate(PETSC_COMM_WORLD,&networkdm));

  /* Register the components in the network */
  CHKERRQ(DMNetworkRegisterComponent(networkdm,"branchstruct",sizeof(struct _p_EDGE_Power),&appctx_power->compkey_branch));
  CHKERRQ(DMNetworkRegisterComponent(networkdm,"busstruct",sizeof(struct _p_VERTEX_Power),&appctx_power->compkey_bus));
  CHKERRQ(DMNetworkRegisterComponent(networkdm,"genstruct",sizeof(struct _p_GEN),&appctx_power->compkey_gen));
  CHKERRQ(DMNetworkRegisterComponent(networkdm,"loadstruct",sizeof(struct _p_LOAD),&appctx_power->compkey_load));

  CHKERRQ(DMNetworkRegisterComponent(networkdm,"edge_water",sizeof(struct _p_EDGE_Water),&appctx_water->compkey_edge));
  CHKERRQ(DMNetworkRegisterComponent(networkdm,"vertex_water",sizeof(struct _p_VERTEX_Water),&appctx_water->compkey_vtx));
#if 0
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"power->compkey_branch %d\n",appctx_power->compkey_branch));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"power->compkey_bus    %d\n",appctx_power->compkey_bus));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"power->compkey_gen    %d\n",appctx_power->compkey_gen));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"power->compkey_load   %d\n",appctx_power->compkey_load));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"water->compkey_edge   %d\n",appctx_water->compkey_edge));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"water->compkey_vtx    %d\n",appctx_water->compkey_vtx));
#endif
  CHKERRQ(PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d] Total local nvertices %D + %D = %D, nedges %D + %D = %D\n",rank,numVertices[0],numVertices[1],numVertices[0]+numVertices[1],numEdges[0],numEdges[1],numEdges[0]+numEdges[1]));
  CHKERRQ(PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT));

  CHKERRQ(DMNetworkSetNumSubNetworks(networkdm,PETSC_DECIDE,Nsubnet));
  CHKERRQ(DMNetworkAddSubnetwork(networkdm,"power",numEdges[0],edgelist_power,&power_netnum));
  CHKERRQ(DMNetworkAddSubnetwork(networkdm,"water",numEdges[1],edgelist_water,&water_netnum));

  /* vertex subnet[0].4 shares with vertex subnet[1].0 */
  power_svtx = 4; water_svtx = 0;
  CHKERRQ(DMNetworkAddSharedVertices(networkdm,power_netnum,water_netnum,1,&power_svtx,&water_svtx));

  /* Set up the network layout */
  CHKERRQ(DMNetworkLayoutSetUp(networkdm));

  /* ADD VARIABLES AND COMPONENTS FOR THE POWER SUBNETWORK */
  /*-------------------------------------------------------*/
  genj = 0; loadj = 0;
  CHKERRQ(DMNetworkGetSubnetwork(networkdm,power_netnum,&nv,&ne,&vtx,&edges));

  for (i = 0; i < ne; i++) {
    CHKERRQ(DMNetworkAddComponent(networkdm,edges[i],appctx_power->compkey_branch,&pfdata->branch[i],0));
  }

  for (i = 0; i < nv; i++) {
    CHKERRQ(DMNetworkIsSharedVertex(networkdm,vtx[i],&flg));
    if (flg) continue;

    CHKERRQ(DMNetworkAddComponent(networkdm,vtx[i],appctx_power->compkey_bus,&pfdata->bus[i],2));
    if (pfdata->bus[i].ngen) {
      for (j = 0; j < pfdata->bus[i].ngen; j++) {
        CHKERRQ(DMNetworkAddComponent(networkdm,vtx[i],appctx_power->compkey_gen,&pfdata->gen[genj++],0));
      }
    }
    if (pfdata->bus[i].nload) {
      for (j=0; j < pfdata->bus[i].nload; j++) {
        CHKERRQ(DMNetworkAddComponent(networkdm,vtx[i],appctx_power->compkey_load,&pfdata->load[loadj++],0));
      }
    }
  }

  /* ADD VARIABLES AND COMPONENTS FOR THE WATER SUBNETWORK */
  /*-------------------------------------------------------*/
  CHKERRQ(DMNetworkGetSubnetwork(networkdm,water_netnum,&nv,&ne,&vtx,&edges));
  for (i = 0; i < ne; i++) {
    CHKERRQ(DMNetworkAddComponent(networkdm,edges[i],appctx_water->compkey_edge,&waterdata->edge[i],0));
  }

  for (i = 0; i < nv; i++) {
    CHKERRQ(DMNetworkIsSharedVertex(networkdm,vtx[i],&flg));
    if (flg) continue;

    CHKERRQ(DMNetworkAddComponent(networkdm,vtx[i],appctx_water->compkey_vtx,&waterdata->vertex[i],1));
  }

  /* ADD VARIABLES AND COMPONENTS AT THE SHARED VERTEX: net[0].4 coupls with net[1].0 -- owning and all ghost ranks of the vertex do this */
  /*----------------------------------------------------------------------------------------------------------------------------*/
  CHKERRQ(DMNetworkGetSharedVertices(networkdm,&nv,&vtx));
  for (i = 0; i < nv; i++) {
    /* power */
    CHKERRQ(DMNetworkAddComponent(networkdm,vtx[i],appctx_power->compkey_bus,&pfdata->bus[4],2));
    /* bus[4] is a load, add its component */
    CHKERRQ(DMNetworkAddComponent(networkdm,vtx[i],appctx_power->compkey_load,&pfdata->load[0],0));

    /* water */
    CHKERRQ(DMNetworkAddComponent(networkdm,vtx[i],appctx_water->compkey_vtx,&waterdata->vertex[0],1));
  }

  /* Set up DM for use */
  CHKERRQ(DMSetUp(networkdm));
  if (viewDM) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nAfter DMSetUp, DMView:\n"));
    CHKERRQ(DMView(networkdm,PETSC_VIEWER_STDOUT_WORLD));
  }

  /* Free user objects */
  CHKERRQ(PetscFree(edgelist_power));
  CHKERRQ(PetscFree(pfdata->bus));
  CHKERRQ(PetscFree(pfdata->gen));
  CHKERRQ(PetscFree(pfdata->branch));
  CHKERRQ(PetscFree(pfdata->load));
  CHKERRQ(PetscFree(pfdata));

  CHKERRQ(PetscFree(edgelist_water));
  CHKERRQ(PetscFree(waterdata->vertex));
  CHKERRQ(PetscFree(waterdata->edge));
  CHKERRQ(PetscFree(waterdata));

  /* Re-distribute networkdm to multiple processes for better job balance */
  if (size >1 && distribute) {
    CHKERRQ(DMNetworkDistribute(&networkdm,0));
    if (viewDM) {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nAfter DMNetworkDistribute, DMView:\n"));
      CHKERRQ(DMView(networkdm,PETSC_VIEWER_STDOUT_WORLD));
    }
  }

  /* Test DMNetworkGetSubnetwork() and DMNetworkGetSubnetworkSharedVertices() */
  if (test) {
    PetscInt  v,gidx;
    CHKERRMPI(MPI_Barrier(PETSC_COMM_WORLD));
    for (i=0; i<Nsubnet; i++) {
      CHKERRQ(DMNetworkGetSubnetwork(networkdm,i,&nv,&ne,&vtx,&edges));
      CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"[%d] After distribute, subnet[%d] ne %d, nv %d\n",rank,i,ne,nv));
      CHKERRMPI(MPI_Barrier(PETSC_COMM_WORLD));

      for (v=0; v<nv; v++) {
        CHKERRQ(DMNetworkIsGhostVertex(networkdm,vtx[v],&ghost));
        CHKERRQ(DMNetworkGetGlobalVertexIndex(networkdm,vtx[v],&gidx));
        CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"[%d] subnet[%d] v %d %d; ghost %d\n",rank,i,vtx[v],gidx,ghost));
      }
      CHKERRMPI(MPI_Barrier(PETSC_COMM_WORLD));
    }
    CHKERRMPI(MPI_Barrier(PETSC_COMM_WORLD));

    CHKERRQ(DMNetworkGetSharedVertices(networkdm,&nv,&vtx));
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"[%d] After distribute, num of shared vertices nsv = %d\n",rank,nv));
    for (v=0; v<nv; v++) {
      CHKERRQ(DMNetworkGetGlobalVertexIndex(networkdm,vtx[v],&gidx));
      CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"[%d] sv %d, gidx=%d\n",rank,vtx[v],gidx));
    }
    CHKERRMPI(MPI_Barrier(PETSC_COMM_WORLD));
  }

  /* Create solution vector X */
  CHKERRQ(DMCreateGlobalVector(networkdm,&X));
  CHKERRQ(VecDuplicate(X,&F));
  CHKERRQ(DMGetLocalVector(networkdm,&user.localXold));
  PetscLogStagePop();

  /* (3) Setup Solvers */
  /*-------------------*/
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-viewJ",&viewJ,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-viewX",&viewX,NULL));

  CHKERRQ(PetscLogStageRegister("SNES Setup",&stage[2]));
  CHKERRQ(PetscLogStagePush(stage[2]));

  CHKERRQ(SetInitialGuess(networkdm,X,&user));

  /* Create coupled snes */
  /*-------------------- */
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"SNES_coupled setup ......\n"));
  user.subsnes_id = Nsubnet;
  CHKERRQ(SNESCreate(PETSC_COMM_WORLD,&snes));
  CHKERRQ(SNESSetDM(snes,networkdm));
  CHKERRQ(SNESSetOptionsPrefix(snes,"coupled_"));
  CHKERRQ(SNESSetFunction(snes,F,FormFunction,&user));
  CHKERRQ(SNESMonitorSet(snes,UserMonitor,&user,NULL));
  CHKERRQ(SNESSetFromOptions(snes));

  if (viewJ) {
    /* View Jac structure */
    CHKERRQ(SNESGetJacobian(snes,&Jac,NULL,NULL,NULL));
    CHKERRQ(MatView(Jac,PETSC_VIEWER_DRAW_WORLD));
  }

  if (viewX) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Solution:\n"));
    CHKERRQ(VecView(X,PETSC_VIEWER_STDOUT_WORLD));
  }

  if (viewJ) {
    /* View assembled Jac */
    CHKERRQ(MatView(Jac,PETSC_VIEWER_DRAW_WORLD));
  }

  /* Create snes_power */
  /*-------------------*/
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"SNES_power setup ......\n"));

  user.subsnes_id = 0;
  CHKERRQ(SNESCreate(PETSC_COMM_WORLD,&snes_power));
  CHKERRQ(SNESSetDM(snes_power,networkdm));
  CHKERRQ(SNESSetOptionsPrefix(snes_power,"power_"));
  CHKERRQ(SNESSetFunction(snes_power,F,FormFunction,&user));
  CHKERRQ(SNESMonitorSet(snes_power,UserMonitor,&user,NULL));

  /* Use user-provide Jacobian */
  CHKERRQ(DMCreateMatrix(networkdm,&Jac));
  CHKERRQ(SNESSetJacobian(snes_power,Jac,Jac,FormJacobian_subPower,&user));
  CHKERRQ(SNESSetFromOptions(snes_power));

  if (viewX) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Power Solution:\n"));
    CHKERRQ(VecView(X,PETSC_VIEWER_STDOUT_WORLD));
  }
  if (viewJ) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Power Jac:\n"));
    CHKERRQ(SNESGetJacobian(snes_power,&Jac,NULL,NULL,NULL));
    CHKERRQ(MatView(Jac,PETSC_VIEWER_DRAW_WORLD));
    /* CHKERRQ(MatView(Jac,PETSC_VIEWER_STDOUT_WORLD)); */
  }

  /* Create snes_water */
  /*-------------------*/
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"SNES_water setup......\n"));

  user.subsnes_id = 1;
  CHKERRQ(SNESCreate(PETSC_COMM_WORLD,&snes_water));
  CHKERRQ(SNESSetDM(snes_water,networkdm));
  CHKERRQ(SNESSetOptionsPrefix(snes_water,"water_"));
  CHKERRQ(SNESSetFunction(snes_water,F,FormFunction,&user));
  CHKERRQ(SNESMonitorSet(snes_water,UserMonitor,&user,NULL));
  CHKERRQ(SNESSetFromOptions(snes_water));

  if (viewX) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Water Solution:\n"));
    CHKERRQ(VecView(X,PETSC_VIEWER_STDOUT_WORLD));
  }
  CHKERRQ(PetscLogStagePop());

  /* (4) Solve */
  /*-----------*/
  CHKERRQ(PetscLogStageRegister("SNES Solve",&stage[3]));
  CHKERRQ(PetscLogStagePush(stage[3]));
  user.it = 0;
  reason  = SNES_DIVERGED_DTOL;
  while (user.it < it_max && (PetscInt)reason<0) {
#if 0
    user.subsnes_id = 0;
    CHKERRQ(SNESSolve(snes_power,NULL,X));

    user.subsnes_id = 1;
    CHKERRQ(SNESSolve(snes_water,NULL,X));
#endif
    user.subsnes_id = Nsubnet;
    CHKERRQ(SNESSolve(snes,NULL,X));

    CHKERRQ(SNESGetConvergedReason(snes,&reason));
    user.it++;
  }
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Coupled_SNES converged in %D iterations\n",user.it));
  if (viewX) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Final Solution:\n"));
    CHKERRQ(VecView(X,PETSC_VIEWER_STDOUT_WORLD));
  }
  CHKERRQ(PetscLogStagePop());

  /* Free objects */
  /* -------------*/
  CHKERRQ(VecDestroy(&X));
  CHKERRQ(VecDestroy(&F));
  CHKERRQ(DMRestoreLocalVector(networkdm,&user.localXold));

  CHKERRQ(SNESDestroy(&snes));
  CHKERRQ(MatDestroy(&Jac));
  CHKERRQ(SNESDestroy(&snes_power));
  CHKERRQ(SNESDestroy(&snes_water));

  CHKERRQ(DMDestroy(&networkdm));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   build:
     requires: !complex double defined(PETSC_HAVE_ATTRIBUTEALIGNED)
     depends: power/PFReadData.c power/pffunctions.c water/waterreaddata.c water/waterfunctions.c

   test:
      args: -coupled_snes_converged_reason -options_left no -viewDM
      localrunfiles: ex1options power/case9.m water/sample1.inp
      output_file: output/ex1.out

   test:
      suffix: 2
      nsize: 3
      args: -coupled_snes_converged_reason -options_left no -petscpartitioner_type parmetis
      localrunfiles: ex1options power/case9.m water/sample1.inp
      output_file: output/ex1_2.out
      requires: parmetis

#   test:
#      suffix: 3
#      nsize: 3
#      args: -coupled_snes_converged_reason -options_left no -distribute false
#      localrunfiles: ex1options power/case9.m water/sample1.inp
#      output_file: output/ex1_2.out

   test:
      suffix: 4
      nsize: 4
      args: -coupled_snes_converged_reason -options_left no -petscpartitioner_type simple -viewDM
      localrunfiles: ex1options power/case9.m water/sample1.inp
      output_file: output/ex1_4.out

TEST*/
