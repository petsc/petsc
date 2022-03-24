static char help[] = "This example demonstrates the use of DMNetwork interface for solving a nonlinear electric power grid problem.\n\
                      The available solver options are in the poweroptions file and the data files are in the datafiles directory.\n\
                      See 'Evaluation of overlapping restricted additive schwarz preconditioning for parallel solution \n\
                          of very large power flow problems' https://dl.acm.org/citation.cfm?id=2536784).\n\
                      The data file format used is from the MatPower package (http://www.pserc.cornell.edu//matpower/).\n\
                      Run this program: mpiexec -n <n> ./pf\n\
                      mpiexec -n <n> ./pfc \n";

/* T
   Concepts: DMNetwork
   Concepts: PETSc SNES solver
*/

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
  CHKERRQ(SNESGetDM(snes,&networkdm));
  CHKERRQ(DMGetLocalVector(networkdm,&localX));
  CHKERRQ(DMGetLocalVector(networkdm,&localF));
  CHKERRQ(VecSet(F,0.0));
  CHKERRQ(VecSet(localF,0.0));

  CHKERRQ(DMGlobalToLocalBegin(networkdm,X,INSERT_VALUES,localX));
  CHKERRQ(DMGlobalToLocalEnd(networkdm,X,INSERT_VALUES,localX));

  CHKERRQ(DMNetworkGetSubnetwork(networkdm,0,&nv,&ne,&vtx,&edges));
  CHKERRQ(FormFunction_Power(networkdm,localX,localF,nv,ne,vtx,edges,User));

  CHKERRQ(DMRestoreLocalVector(networkdm,&localX));

  CHKERRQ(DMLocalToGlobalBegin(networkdm,localF,ADD_VALUES,F));
  CHKERRQ(DMLocalToGlobalEnd(networkdm,localF,ADD_VALUES,F));
  CHKERRQ(DMRestoreLocalVector(networkdm,&localF));
  PetscFunctionReturn(0);
}

PetscErrorCode SetInitialValues(DM networkdm,Vec X,void* appctx)
{
  PetscInt       vStart,vEnd,nv,ne;
  const PetscInt *vtx,*edges;
  Vec            localX;
  UserCtx_Power  *user_power=(UserCtx_Power*)appctx;

  PetscFunctionBegin;
  CHKERRQ(DMNetworkGetVertexRange(networkdm,&vStart, &vEnd));

  CHKERRQ(DMGetLocalVector(networkdm,&localX));

  CHKERRQ(VecSet(X,0.0));
  CHKERRQ(DMGlobalToLocalBegin(networkdm,X,INSERT_VALUES,localX));
  CHKERRQ(DMGlobalToLocalEnd(networkdm,X,INSERT_VALUES,localX));

  CHKERRQ(DMNetworkGetSubnetwork(networkdm,0,&nv,&ne,&vtx,&edges));
  CHKERRQ(SetInitialGuess_Power(networkdm,localX,nv,ne,vtx,edges,user_power));

  CHKERRQ(DMLocalToGlobalBegin(networkdm,localX,ADD_VALUES,X));
  CHKERRQ(DMLocalToGlobalEnd(networkdm,localX,ADD_VALUES,X));
  CHKERRQ(DMRestoreLocalVector(networkdm,&localX));
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

  CHKERRQ(PetscInitialize(&argc,&argv,"poweroptions",help));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  {
    /* introduce the const crank so the clang static analyzer realizes that if it enters any of the if (crank) then it must have entered the first */
    /* this is an experiment to see how the analyzer reacts */
    const PetscMPIInt crank = rank;

    /* Create an empty network object */
    CHKERRQ(DMNetworkCreate(PETSC_COMM_WORLD,&networkdm));
    /* Register the components in the network */
    CHKERRQ(DMNetworkRegisterComponent(networkdm,"branchstruct",sizeof(struct _p_EDGE_Power),&User.compkey_branch));
    CHKERRQ(DMNetworkRegisterComponent(networkdm,"busstruct",sizeof(struct _p_VERTEX_Power),&User.compkey_bus));
    CHKERRQ(DMNetworkRegisterComponent(networkdm,"genstruct",sizeof(struct _p_GEN),&User.compkey_gen));
    CHKERRQ(DMNetworkRegisterComponent(networkdm,"loadstruct",sizeof(struct _p_LOAD),&User.compkey_load));

    CHKERRQ(PetscLogStageRegister("Read Data",&stage1));
    PetscLogStagePush(stage1);
    /* READ THE DATA */
    if (!crank) {
      /*    READ DATA */
      /* Only rank 0 reads the data */
      CHKERRQ(PetscOptionsGetString(NULL,NULL,"-pfdata",pfdata_file,sizeof(pfdata_file),NULL));
      CHKERRQ(PetscNew(&pfdata));
      CHKERRQ(PFReadMatPowerData(pfdata,pfdata_file));
      User.Sbase = pfdata->sbase;

      numEdges = pfdata->nbranch;
      CHKERRQ(PetscMalloc1(2*numEdges,&edges));
      CHKERRQ(GetListofEdges_Power(pfdata,edges));
    }

    /* If external option activated. Introduce error in jacobian */
    CHKERRQ(PetscOptionsHasName(NULL,NULL, "-jac_error", &User.jac_error));

    PetscLogStagePop();
    CHKERRMPI(MPI_Barrier(PETSC_COMM_WORLD));
    CHKERRQ(PetscLogStageRegister("Create network",&stage2));
    PetscLogStagePush(stage2);
    /* Set number of nodes/edges */
    CHKERRQ(DMNetworkSetNumSubNetworks(networkdm,PETSC_DECIDE,1));
    CHKERRQ(DMNetworkAddSubnetwork(networkdm,"",numEdges,edges,NULL));

    /* Set up the network layout */
    CHKERRQ(DMNetworkLayoutSetUp(networkdm));

    if (!crank) {
      CHKERRQ(PetscFree(edges));
    }

    /* Add network components only process 0 has any data to add */
    if (!crank) {
      genj=0; loadj=0;
      CHKERRQ(DMNetworkGetEdgeRange(networkdm,&eStart,&eEnd));
      for (i = eStart; i < eEnd; i++) {
        CHKERRQ(DMNetworkAddComponent(networkdm,i,User.compkey_branch,&pfdata->branch[i-eStart],0));
      }
      CHKERRQ(DMNetworkGetVertexRange(networkdm,&vStart,&vEnd));
      for (i = vStart; i < vEnd; i++) {
        CHKERRQ(DMNetworkAddComponent(networkdm,i,User.compkey_bus,&pfdata->bus[i-vStart],2));
        if (pfdata->bus[i-vStart].ngen) {
          for (j = 0; j < pfdata->bus[i-vStart].ngen; j++) {
            CHKERRQ(DMNetworkAddComponent(networkdm,i,User.compkey_gen,&pfdata->gen[genj++],0));
          }
        }
        if (pfdata->bus[i-vStart].nload) {
          for (j=0; j < pfdata->bus[i-vStart].nload; j++) {
            CHKERRQ(DMNetworkAddComponent(networkdm,i,User.compkey_load,&pfdata->load[loadj++],0));
          }
        }
      }
    }

    /* Set up DM for use */
    CHKERRQ(DMSetUp(networkdm));

    if (!crank) {
      CHKERRQ(PetscFree(pfdata->bus));
      CHKERRQ(PetscFree(pfdata->gen));
      CHKERRQ(PetscFree(pfdata->branch));
      CHKERRQ(PetscFree(pfdata->load));
      CHKERRQ(PetscFree(pfdata));
    }

    /* Distribute networkdm to multiple processes */
    CHKERRQ(DMNetworkDistribute(&networkdm,0));

    PetscLogStagePop();
    CHKERRQ(DMNetworkGetEdgeRange(networkdm,&eStart,&eEnd));
    CHKERRQ(DMNetworkGetVertexRange(networkdm,&vStart,&vEnd));

#if 0
    EDGE_Power     edge;
    PetscInt       key,kk,numComponents;
    VERTEX_Power   bus;
    GEN            gen;
    LOAD           load;

    for (i = eStart; i < eEnd; i++) {
      CHKERRQ(DMNetworkGetComponent(networkdm,i,0,&key,(void**)&edge));
      CHKERRQ(DMNetworkGetNumComponents(networkdm,i,&numComponents));
      CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Rank %d ncomps = %d Line %d ---- %d\n",crank,numComponents,edge->internal_i,edge->internal_j));
    }

    for (i = vStart; i < vEnd; i++) {
      CHKERRQ(DMNetworkGetNumComponents(networkdm,i,&numComponents));
      for (kk=0; kk < numComponents; kk++) {
        CHKERRQ(DMNetworkGetComponent(networkdm,i,kk,&key,&component));
        if (key == 1) {
          bus = (VERTEX_Power)(component);
          CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Rank %d ncomps = %d Bus %d\n",crank,numComponents,bus->internal_i));
        } else if (key == 2) {
          gen = (GEN)(component);
          CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Rank %d Gen pg = %f qg = %f\n",crank,gen->pg,gen->qg));
        } else if (key == 3) {
          load = (LOAD)(component);
          CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Rank %d Load pl = %f ql = %f\n",crank,load->pl,load->ql));
        }
      }
    }
#endif
    /* Broadcast Sbase to all processors */
    CHKERRMPI(MPI_Bcast(&User.Sbase,1,MPIU_SCALAR,0,PETSC_COMM_WORLD));

    CHKERRQ(DMCreateGlobalVector(networkdm,&X));
    CHKERRQ(VecDuplicate(X,&F));

    CHKERRQ(DMCreateMatrix(networkdm,&J));
    CHKERRQ(MatSetOption(J,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE));

    CHKERRQ(SetInitialValues(networkdm,X,&User));

    /* HOOK UP SOLVER */
    CHKERRQ(SNESCreate(PETSC_COMM_WORLD,&snes));
    CHKERRQ(SNESSetDM(snes,networkdm));
    CHKERRQ(SNESSetFunction(snes,F,FormFunction,&User));
    CHKERRQ(SNESSetJacobian(snes,J,J,FormJacobian_Power,&User));
    CHKERRQ(SNESSetFromOptions(snes));

    CHKERRQ(SNESSolve(snes,NULL,X));
    /* CHKERRQ(VecView(X,PETSC_VIEWER_STDOUT_WORLD)); */

    CHKERRQ(VecDestroy(&X));
    CHKERRQ(VecDestroy(&F));
    CHKERRQ(MatDestroy(&J));

    CHKERRQ(SNESDestroy(&snes));
    CHKERRQ(DMDestroy(&networkdm));
  }
  CHKERRQ(PetscFinalize());
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
