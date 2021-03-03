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
  PetscErrorCode ierr;
  DM             networkdm;
  UserCtx_Power  *User=(UserCtx_Power*)appctx;
  Vec            localX,localF;
  PetscInt       nv,ne;
  const PetscInt *vtx,*edges;

  PetscFunctionBegin;
  ierr = SNESGetDM(snes,&networkdm);CHKERRQ(ierr);
  ierr = DMGetLocalVector(networkdm,&localX);CHKERRQ(ierr);
  ierr = DMGetLocalVector(networkdm,&localF);CHKERRQ(ierr);
  ierr = VecSet(F,0.0);CHKERRQ(ierr);
  ierr = VecSet(localF,0.0);CHKERRQ(ierr);

  ierr = DMGlobalToLocalBegin(networkdm,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(networkdm,X,INSERT_VALUES,localX);CHKERRQ(ierr);

  ierr = DMNetworkGetSubnetwork(networkdm,0,&nv,&ne,&vtx,&edges);CHKERRQ(ierr);
  ierr = FormFunction_Power(networkdm,localX,localF,nv,ne,vtx,edges,User);CHKERRQ(ierr);

  ierr = DMRestoreLocalVector(networkdm,&localX);CHKERRQ(ierr);

  ierr = DMLocalToGlobalBegin(networkdm,localF,ADD_VALUES,F);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(networkdm,localF,ADD_VALUES,F);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(networkdm,&localF);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SetInitialValues(DM networkdm,Vec X,void* appctx)
{
  PetscErrorCode ierr;
  PetscInt       vStart,vEnd,nv,ne;
  const PetscInt *vtx,*edges;
  Vec            localX;
  UserCtx_Power  *user_power=(UserCtx_Power*)appctx;

  PetscFunctionBegin;
  ierr = DMNetworkGetVertexRange(networkdm,&vStart, &vEnd);CHKERRQ(ierr);

  ierr = DMGetLocalVector(networkdm,&localX);CHKERRQ(ierr);

  ierr = VecSet(X,0.0);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(networkdm,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(networkdm,X,INSERT_VALUES,localX);CHKERRQ(ierr);

  ierr = DMNetworkGetSubnetwork(networkdm,0,&nv,&ne,&vtx,&edges);CHKERRQ(ierr);
  ierr = SetInitialGuess_Power(networkdm,localX,nv,ne,vtx,edges,user_power);CHKERRQ(ierr);

  ierr = DMLocalToGlobalBegin(networkdm,localX,ADD_VALUES,X);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(networkdm,localX,ADD_VALUES,X);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(networkdm,&localX);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc,char ** argv)
{
  PetscErrorCode   ierr;
  char             pfdata_file[PETSC_MAX_PATH_LEN]="case9.m";
  PFDATA           *pfdata;
  PetscInt         numEdges=0,numVertices=0;
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

  ierr = PetscInitialize(&argc,&argv,"poweroptions",help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
  {
    /* introduce the const crank so the clang static analyzer realizes that if it enters any of the if (crank) then it must have entered the first */
    /* this is an experiment to see how the analyzer reacts */
    const PetscMPIInt crank = rank;

    /* Create an empty network object */
    ierr = DMNetworkCreate(PETSC_COMM_WORLD,&networkdm);CHKERRQ(ierr);
    /* Register the components in the network */
    ierr = DMNetworkRegisterComponent(networkdm,"branchstruct",sizeof(struct _p_EDGE_Power),&User.compkey_branch);CHKERRQ(ierr);
    ierr = DMNetworkRegisterComponent(networkdm,"busstruct",sizeof(struct _p_VERTEX_Power),&User.compkey_bus);CHKERRQ(ierr);
    ierr = DMNetworkRegisterComponent(networkdm,"genstruct",sizeof(struct _p_GEN),&User.compkey_gen);CHKERRQ(ierr);
    ierr = DMNetworkRegisterComponent(networkdm,"loadstruct",sizeof(struct _p_LOAD),&User.compkey_load);CHKERRQ(ierr);

    ierr = PetscLogStageRegister("Read Data",&stage1);CHKERRQ(ierr);
    PetscLogStagePush(stage1);
    /* READ THE DATA */
    if (!crank) {
      /*    READ DATA */
      /* Only rank 0 reads the data */
      ierr = PetscOptionsGetString(NULL,NULL,"-pfdata",pfdata_file,sizeof(pfdata_file),NULL);CHKERRQ(ierr);
      ierr = PetscNew(&pfdata);CHKERRQ(ierr);
      ierr = PFReadMatPowerData(pfdata,pfdata_file);CHKERRQ(ierr);
      User.Sbase = pfdata->sbase;

      numEdges = pfdata->nbranch;
      numVertices = pfdata->nbus;

      ierr = PetscMalloc1(2*numEdges,&edges);CHKERRQ(ierr);
      ierr = GetListofEdges_Power(pfdata,edges);CHKERRQ(ierr);
    }

    /* If external option activated. Introduce error in jacobian */
    ierr = PetscOptionsHasName(NULL,NULL, "-jac_error", &User.jac_error);CHKERRQ(ierr);

    PetscLogStagePop();
    ierr = MPI_Barrier(PETSC_COMM_WORLD);CHKERRMPI(ierr);
    ierr = PetscLogStageRegister("Create network",&stage2);CHKERRQ(ierr);
    PetscLogStagePush(stage2);
    /* Set number of nodes/edges */
    ierr = DMNetworkSetNumSubNetworks(networkdm,PETSC_DECIDE,1);CHKERRQ(ierr);
    ierr = DMNetworkAddSubnetwork(networkdm,"",numVertices,numEdges,edges,NULL);CHKERRQ(ierr);

    /* Set up the network layout */
    ierr = DMNetworkLayoutSetUp(networkdm);CHKERRQ(ierr);

    if (!crank) {
      ierr = PetscFree(edges);CHKERRQ(ierr);
    }

    /* Add network components only process 0 has any data to add */
    if (!crank) {
      genj=0; loadj=0;
      ierr = DMNetworkGetEdgeRange(networkdm,&eStart,&eEnd);CHKERRQ(ierr);
      for (i = eStart; i < eEnd; i++) {
        ierr = DMNetworkAddComponent(networkdm,i,User.compkey_branch,&pfdata->branch[i-eStart],0);CHKERRQ(ierr);
      }
      ierr = DMNetworkGetVertexRange(networkdm,&vStart,&vEnd);CHKERRQ(ierr);
      for (i = vStart; i < vEnd; i++) {
        ierr = DMNetworkAddComponent(networkdm,i,User.compkey_bus,&pfdata->bus[i-vStart],2);CHKERRQ(ierr);
        if (pfdata->bus[i-vStart].ngen) {
          for (j = 0; j < pfdata->bus[i-vStart].ngen; j++) {
            ierr = DMNetworkAddComponent(networkdm,i,User.compkey_gen,&pfdata->gen[genj++],0);CHKERRQ(ierr);
          }
        }
        if (pfdata->bus[i-vStart].nload) {
          for (j=0; j < pfdata->bus[i-vStart].nload; j++) {
            ierr = DMNetworkAddComponent(networkdm,i,User.compkey_load,&pfdata->load[loadj++],0);CHKERRQ(ierr);
          }
        }
      }
    }

    /* Set up DM for use */
    ierr = DMSetUp(networkdm);CHKERRQ(ierr);

    if (!crank) {
      ierr = PetscFree(pfdata->bus);CHKERRQ(ierr);
      ierr = PetscFree(pfdata->gen);CHKERRQ(ierr);
      ierr = PetscFree(pfdata->branch);CHKERRQ(ierr);
      ierr = PetscFree(pfdata->load);CHKERRQ(ierr);
      ierr = PetscFree(pfdata);CHKERRQ(ierr);
    }

    /* Distribute networkdm to multiple processes */
    ierr = DMNetworkDistribute(&networkdm,0);CHKERRQ(ierr);

    PetscLogStagePop();
    ierr = DMNetworkGetEdgeRange(networkdm,&eStart,&eEnd);CHKERRQ(ierr);
    ierr = DMNetworkGetVertexRange(networkdm,&vStart,&vEnd);CHKERRQ(ierr);

#if 0
    EDGE_Power     edge;
    PetscInt       key,kk,numComponents;
    VERTEX_Power   bus;
    GEN            gen;
    LOAD           load;

    for (i = eStart; i < eEnd; i++) {
      ierr = DMNetworkGetComponent(networkdm,i,0,&key,(void**)&edge);CHKERRQ(ierr);
      ierr = DMNetworkGetNumComponents(networkdm,i,&numComponents);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_SELF,"Rank %d ncomps = %d Line %d ---- %d\n",crank,numComponents,edge->internal_i,edge->internal_j);CHKERRQ(ierr);
    }

    for (i = vStart; i < vEnd; i++) {
      ierr = DMNetworkGetNumComponents(networkdm,i,&numComponents);CHKERRQ(ierr);
      for (kk=0; kk < numComponents; kk++) {
        ierr = DMNetworkGetComponent(networkdm,i,kk,&key,&component);CHKERRQ(ierr);
        if (key == 1) {
          bus = (VERTEX_Power)(component);
          ierr = PetscPrintf(PETSC_COMM_SELF,"Rank %d ncomps = %d Bus %d\n",crank,numComponents,bus->internal_i);CHKERRQ(ierr);
        } else if (key == 2) {
          gen = (GEN)(component);
          ierr = PetscPrintf(PETSC_COMM_SELF,"Rank %d Gen pg = %f qg = %f\n",crank,gen->pg,gen->qg);CHKERRQ(ierr);
        } else if (key == 3) {
          load = (LOAD)(component);
          ierr = PetscPrintf(PETSC_COMM_SELF,"Rank %d Load pl = %f ql = %f\n",crank,load->pl,load->ql);CHKERRQ(ierr);
        }
      }
    }
#endif
    /* Broadcast Sbase to all processors */
    ierr = MPI_Bcast(&User.Sbase,1,MPIU_SCALAR,0,PETSC_COMM_WORLD);CHKERRMPI(ierr);

    ierr = DMCreateGlobalVector(networkdm,&X);CHKERRQ(ierr);
    ierr = VecDuplicate(X,&F);CHKERRQ(ierr);

    ierr = DMCreateMatrix(networkdm,&J);CHKERRQ(ierr);
    ierr = MatSetOption(J,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);CHKERRQ(ierr);

    ierr = SetInitialValues(networkdm,X,&User);CHKERRQ(ierr);

    /* HOOK UP SOLVER */
    ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
    ierr = SNESSetDM(snes,networkdm);CHKERRQ(ierr);
    ierr = SNESSetFunction(snes,F,FormFunction,&User);CHKERRQ(ierr);
    ierr = SNESSetJacobian(snes,J,J,FormJacobian_Power,&User);CHKERRQ(ierr);
    ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

    ierr = SNESSolve(snes,NULL,X);CHKERRQ(ierr);
    /* ierr = VecView(X,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); */

    ierr = VecDestroy(&X);CHKERRQ(ierr);
    ierr = VecDestroy(&F);CHKERRQ(ierr);
    ierr = MatDestroy(&J);CHKERRQ(ierr);

    ierr = SNESDestroy(&snes);CHKERRQ(ierr);
    ierr = DMDestroy(&networkdm);CHKERRQ(ierr);
  }
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
     depends: PFReadData.c pffunctions.c
     requires: !complex double define(PETSC_HAVE_ATTRIBUTEALIGNED)

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
