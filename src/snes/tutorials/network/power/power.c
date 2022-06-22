static char help[] = "This example demonstrates the use of DMNetwork interface for solving a nonlinear electric power grid problem.\n\
                      The available solver options are in the poweroptions file and the data files are in the datafiles directory.\n\
                      See 'Evaluation of overlapping restricted additive schwarz preconditioning for parallel solution \n\
                          of very large power flow problems' https://dl.acm.org/citation.cfm?id=2536784).\n\
                      The data file format used is from the MatPower package (http://www.pserc.cornell.edu//matpower/).\n\
                      Run this program: mpiexec -n <n> ./pf\n\
                      mpiexec -n <n> ./pfc \n";

#include "power.h"
#include <petscdmnetwork.h>

PetscErrorCode FormFunction(SNES snes,Vec X, Vec F,void *appctx)
{
  DM             networkdm;
  UserCtx_Power  *User=(UserCtx_Power*)appctx;
  Vec            localX,localF;
  PetscInt       nv,ne;
  const PetscInt *vtx,*edges;

  PetscFunctionBegin;
  PetscCall(SNESGetDM(snes,&networkdm));
  PetscCall(DMGetLocalVector(networkdm,&localX));
  PetscCall(DMGetLocalVector(networkdm,&localF));
  PetscCall(VecSet(F,0.0));
  PetscCall(VecSet(localF,0.0));

  PetscCall(DMGlobalToLocalBegin(networkdm,X,INSERT_VALUES,localX));
  PetscCall(DMGlobalToLocalEnd(networkdm,X,INSERT_VALUES,localX));

  PetscCall(DMNetworkGetSubnetwork(networkdm,0,&nv,&ne,&vtx,&edges));
  PetscCall(FormFunction_Power(networkdm,localX,localF,nv,ne,vtx,edges,User));

  PetscCall(DMRestoreLocalVector(networkdm,&localX));

  PetscCall(DMLocalToGlobalBegin(networkdm,localF,ADD_VALUES,F));
  PetscCall(DMLocalToGlobalEnd(networkdm,localF,ADD_VALUES,F));
  PetscCall(DMRestoreLocalVector(networkdm,&localF));
  PetscFunctionReturn(0);
}

PetscErrorCode SetInitialValues(DM networkdm,Vec X,void* appctx)
{
  PetscInt       vStart,vEnd,nv,ne;
  const PetscInt *vtx,*edges;
  Vec            localX;
  UserCtx_Power  *user_power=(UserCtx_Power*)appctx;

  PetscFunctionBegin;
  PetscCall(DMNetworkGetVertexRange(networkdm,&vStart, &vEnd));

  PetscCall(DMGetLocalVector(networkdm,&localX));

  PetscCall(VecSet(X,0.0));
  PetscCall(DMGlobalToLocalBegin(networkdm,X,INSERT_VALUES,localX));
  PetscCall(DMGlobalToLocalEnd(networkdm,X,INSERT_VALUES,localX));

  PetscCall(DMNetworkGetSubnetwork(networkdm,0,&nv,&ne,&vtx,&edges));
  PetscCall(SetInitialGuess_Power(networkdm,localX,nv,ne,vtx,edges,user_power));

  PetscCall(DMLocalToGlobalBegin(networkdm,localX,ADD_VALUES,X));
  PetscCall(DMLocalToGlobalEnd(networkdm,localX,ADD_VALUES,X));
  PetscCall(DMRestoreLocalVector(networkdm,&localX));
  PetscFunctionReturn(0);
}

int main(int argc,char ** argv)
{
  char             pfdata_file[PETSC_MAX_PATH_LEN]="case9.m";
  PFDATA           *pfdata;
  PetscInt         numEdges=0;
  PetscInt         *edges = NULL;
  PetscInt         i;
  DM               networkdm;
  UserCtx_Power    User;
#if defined(PETSC_USE_LOG)
  PetscLogStage    stage1,stage2;
#endif
  PetscMPIInt      rank;
  PetscInt         eStart, eEnd, vStart, vEnd,j;
  PetscInt         genj,loadj;
  Vec              X,F;
  Mat              J;
  SNES             snes;

  PetscCall(PetscInitialize(&argc,&argv,"poweroptions",help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  {
    /* introduce the const crank so the clang static analyzer realizes that if it enters any of the if (crank) then it must have entered the first */
    /* this is an experiment to see how the analyzer reacts */
    const PetscMPIInt crank = rank;

    /* Create an empty network object */
    PetscCall(DMNetworkCreate(PETSC_COMM_WORLD,&networkdm));
    /* Register the components in the network */
    PetscCall(DMNetworkRegisterComponent(networkdm,"branchstruct",sizeof(struct _p_EDGE_Power),&User.compkey_branch));
    PetscCall(DMNetworkRegisterComponent(networkdm,"busstruct",sizeof(struct _p_VERTEX_Power),&User.compkey_bus));
    PetscCall(DMNetworkRegisterComponent(networkdm,"genstruct",sizeof(struct _p_GEN),&User.compkey_gen));
    PetscCall(DMNetworkRegisterComponent(networkdm,"loadstruct",sizeof(struct _p_LOAD),&User.compkey_load));

    PetscCall(PetscLogStageRegister("Read Data",&stage1));
    PetscLogStagePush(stage1);
    /* READ THE DATA */
    if (!crank) {
      /*    READ DATA */
      /* Only rank 0 reads the data */
      PetscCall(PetscOptionsGetString(NULL,NULL,"-pfdata",pfdata_file,sizeof(pfdata_file),NULL));
      PetscCall(PetscNew(&pfdata));
      PetscCall(PFReadMatPowerData(pfdata,pfdata_file));
      User.Sbase = pfdata->sbase;

      numEdges = pfdata->nbranch;
      PetscCall(PetscMalloc1(2*numEdges,&edges));
      PetscCall(GetListofEdges_Power(pfdata,edges));
    }

    /* If external option activated. Introduce error in jacobian */
    PetscCall(PetscOptionsHasName(NULL,NULL, "-jac_error", &User.jac_error));

    PetscLogStagePop();
    PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
    PetscCall(PetscLogStageRegister("Create network",&stage2));
    PetscLogStagePush(stage2);
    /* Set number of nodes/edges */
    PetscCall(DMNetworkSetNumSubNetworks(networkdm,PETSC_DECIDE,1));
    PetscCall(DMNetworkAddSubnetwork(networkdm,"",numEdges,edges,NULL));

    /* Set up the network layout */
    PetscCall(DMNetworkLayoutSetUp(networkdm));

    if (!crank) {
      PetscCall(PetscFree(edges));
    }

    /* Add network components only process 0 has any data to add */
    if (!crank) {
      genj=0; loadj=0;
      PetscCall(DMNetworkGetEdgeRange(networkdm,&eStart,&eEnd));
      for (i = eStart; i < eEnd; i++) {
        PetscCall(DMNetworkAddComponent(networkdm,i,User.compkey_branch,&pfdata->branch[i-eStart],0));
      }
      PetscCall(DMNetworkGetVertexRange(networkdm,&vStart,&vEnd));
      for (i = vStart; i < vEnd; i++) {
        PetscCall(DMNetworkAddComponent(networkdm,i,User.compkey_bus,&pfdata->bus[i-vStart],2));
        if (pfdata->bus[i-vStart].ngen) {
          for (j = 0; j < pfdata->bus[i-vStart].ngen; j++) {
            PetscCall(DMNetworkAddComponent(networkdm,i,User.compkey_gen,&pfdata->gen[genj++],0));
          }
        }
        if (pfdata->bus[i-vStart].nload) {
          for (j=0; j < pfdata->bus[i-vStart].nload; j++) {
            PetscCall(DMNetworkAddComponent(networkdm,i,User.compkey_load,&pfdata->load[loadj++],0));
          }
        }
      }
    }

    /* Set up DM for use */
    PetscCall(DMSetUp(networkdm));

    if (!crank) {
      PetscCall(PetscFree(pfdata->bus));
      PetscCall(PetscFree(pfdata->gen));
      PetscCall(PetscFree(pfdata->branch));
      PetscCall(PetscFree(pfdata->load));
      PetscCall(PetscFree(pfdata));
    }

    /* Distribute networkdm to multiple processes */
    PetscCall(DMNetworkDistribute(&networkdm,0));

    PetscLogStagePop();
    PetscCall(DMNetworkGetEdgeRange(networkdm,&eStart,&eEnd));
    PetscCall(DMNetworkGetVertexRange(networkdm,&vStart,&vEnd));

#if 0
    EDGE_Power     edge;
    PetscInt       key,kk,numComponents;
    VERTEX_Power   bus;
    GEN            gen;
    LOAD           load;

    for (i = eStart; i < eEnd; i++) {
      PetscCall(DMNetworkGetComponent(networkdm,i,0,&key,(void**)&edge));
      PetscCall(DMNetworkGetNumComponents(networkdm,i,&numComponents));
      PetscCall(PetscPrintf(PETSC_COMM_SELF,"Rank %d ncomps = %d Line %d ---- %d\n",crank,numComponents,edge->internal_i,edge->internal_j));
    }

    for (i = vStart; i < vEnd; i++) {
      PetscCall(DMNetworkGetNumComponents(networkdm,i,&numComponents));
      for (kk=0; kk < numComponents; kk++) {
        PetscCall(DMNetworkGetComponent(networkdm,i,kk,&key,&component));
        if (key == 1) {
          bus = (VERTEX_Power)(component);
          PetscCall(PetscPrintf(PETSC_COMM_SELF,"Rank %d ncomps = %d Bus %d\n",crank,numComponents,bus->internal_i));
        } else if (key == 2) {
          gen = (GEN)(component);
          PetscCall(PetscPrintf(PETSC_COMM_SELF,"Rank %d Gen pg = %f qg = %f\n",crank,(double)gen->pg,(double)gen->qg));
        } else if (key == 3) {
          load = (LOAD)(component);
          PetscCall(PetscPrintf(PETSC_COMM_SELF,"Rank %d Load pl = %f ql = %f\n",crank,(double)load->pl,(double)load->ql));
        }
      }
    }
#endif
    /* Broadcast Sbase to all processors */
    PetscCallMPI(MPI_Bcast(&User.Sbase,1,MPIU_SCALAR,0,PETSC_COMM_WORLD));

    PetscCall(DMCreateGlobalVector(networkdm,&X));
    PetscCall(VecDuplicate(X,&F));

    PetscCall(DMCreateMatrix(networkdm,&J));
    PetscCall(MatSetOption(J,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE));

    PetscCall(SetInitialValues(networkdm,X,&User));

    /* HOOK UP SOLVER */
    PetscCall(SNESCreate(PETSC_COMM_WORLD,&snes));
    PetscCall(SNESSetDM(snes,networkdm));
    PetscCall(SNESSetFunction(snes,F,FormFunction,&User));
    PetscCall(SNESSetJacobian(snes,J,J,FormJacobian_Power,&User));
    PetscCall(SNESSetFromOptions(snes));

    PetscCall(SNESSolve(snes,NULL,X));
    /* PetscCall(VecView(X,PETSC_VIEWER_STDOUT_WORLD)); */

    PetscCall(VecDestroy(&X));
    PetscCall(VecDestroy(&F));
    PetscCall(MatDestroy(&J));

    PetscCall(SNESDestroy(&snes));
    PetscCall(DMDestroy(&networkdm));
  }
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
     depends: PFReadData.c pffunctions.c
     requires: !complex double defined(PETSC_HAVE_ATTRIBUTEALIGNED)

   test:
     args: -snes_rtol 1.e-3
     localrunfiles: poweroptions case9.m
     output_file: output/power_1.out

   test:
     suffix: 2
     args: -snes_rtol 1.e-3 -petscpartitioner_type simple
     nsize: 4
     localrunfiles: poweroptions case9.m
     output_file: output/power_1.out

TEST*/
