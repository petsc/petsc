static char help[] = "This example demonstrates the use of DMNetwork interface for solving a steady-state water network model.\n\
                      The water network equations follow those used for the package EPANET\n\
                      The data file format used is from the EPANET package (https://www.epa.gov/water-research/epanet).\n\
                      Run this program: mpiexec -n <n> ./water\n\\n";

/* T
   Concepts: DMNetwork
   Concepts: PETSc SNES solver
*/

#include "water.h"
#include <petscdmnetwork.h>

int main(int argc,char ** argv)
{
  PetscErrorCode   ierr;
  char             waterdata_file[PETSC_MAX_PATH_LEN] = "sample1.inp";
  WATERDATA        *waterdata;
  AppCtx_Water     appctx;
#if defined(PETSC_USE_LOG)
  PetscLogStage    stage1,stage2;
#endif
  PetscMPIInt      crank;
  DM               networkdm;
  PetscInt         *edgelist = NULL;
  PetscInt         nv,ne,i;
  const PetscInt   *vtx,*edges;
  Vec              X,F;
  SNES             snes;
  SNESConvergedReason reason;

  ierr = PetscInitialize(&argc,&argv,"wateroptions",help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&crank));

  /* Create an empty network object */
  CHKERRQ(DMNetworkCreate(PETSC_COMM_WORLD,&networkdm));

  /* Register the components in the network */
  CHKERRQ(DMNetworkRegisterComponent(networkdm,"edgestruct",sizeof(struct _p_EDGE_Water),&appctx.compkey_edge));
  CHKERRQ(DMNetworkRegisterComponent(networkdm,"busstruct",sizeof(struct _p_VERTEX_Water),&appctx.compkey_vtx));

  CHKERRQ(PetscLogStageRegister("Read Data",&stage1));
  CHKERRQ(PetscLogStagePush(stage1));
  CHKERRQ(PetscNew(&waterdata));

  /* READ THE DATA */
  if (!crank) {
    /* READ DATA. Only rank 0 reads the data */
    CHKERRQ(PetscOptionsGetString(NULL,NULL,"-waterdata",waterdata_file,sizeof(waterdata_file),NULL));
    CHKERRQ(WaterReadData(waterdata,waterdata_file));

    CHKERRQ(PetscCalloc1(2*waterdata->nedge,&edgelist));
    CHKERRQ(GetListofEdges_Water(waterdata,edgelist));
  }
  CHKERRQ(PetscLogStagePop());

  CHKERRQ(PetscLogStageRegister("Create network",&stage2));
  CHKERRQ(PetscLogStagePush(stage2));

  /* Set numbers of nodes and edges */
  CHKERRQ(DMNetworkSetNumSubNetworks(networkdm,PETSC_DECIDE,1));
  CHKERRQ(DMNetworkAddSubnetwork(networkdm,"",waterdata->nedge,edgelist,NULL));
  if (!crank) {
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"water nvertices %D, nedges %D\n",waterdata->nvertex,waterdata->nedge));
  }

  /* Set up the network layout */
  CHKERRQ(DMNetworkLayoutSetUp(networkdm));

  if (!crank) {
    CHKERRQ(PetscFree(edgelist));
  }

  /* ADD VARIABLES AND COMPONENTS FOR THE NETWORK */
  CHKERRQ(DMNetworkGetSubnetwork(networkdm,0,&nv,&ne,&vtx,&edges));

  for (i = 0; i < ne; i++) {
    CHKERRQ(DMNetworkAddComponent(networkdm,edges[i],appctx.compkey_edge,&waterdata->edge[i],0));
  }

  for (i = 0; i < nv; i++) {
    CHKERRQ(DMNetworkAddComponent(networkdm,vtx[i],appctx.compkey_vtx,&waterdata->vertex[i],1));
  }

  /* Set up DM for use */
  CHKERRQ(DMSetUp(networkdm));

  if (!crank) {
    CHKERRQ(PetscFree(waterdata->vertex));
    CHKERRQ(PetscFree(waterdata->edge));
  }
  CHKERRQ(PetscFree(waterdata));

  /* Distribute networkdm to multiple processes */
  CHKERRQ(DMNetworkDistribute(&networkdm,0));

  CHKERRQ(PetscLogStagePop());

  CHKERRQ(DMCreateGlobalVector(networkdm,&X));
  CHKERRQ(VecDuplicate(X,&F));

  /* HOOK UP SOLVER */
  CHKERRQ(SNESCreate(PETSC_COMM_WORLD,&snes));
  CHKERRQ(SNESSetDM(snes,networkdm));
  CHKERRQ(SNESSetOptionsPrefix(snes,"water_"));
  CHKERRQ(SNESSetFunction(snes,F,WaterFormFunction,NULL));
  CHKERRQ(SNESSetFromOptions(snes));

  CHKERRQ(WaterSetInitialGuess(networkdm,X));
  /* CHKERRQ(VecView(X,PETSC_VIEWER_STDOUT_WORLD)); */

  CHKERRQ(SNESSolve(snes,NULL,X));
  CHKERRQ(SNESGetConvergedReason(snes,&reason));

  PetscCheckFalse(reason < 0,PETSC_COMM_SELF,PETSC_ERR_CONV_FAILED,"No solution found for the water network");
  /* CHKERRQ(VecView(X,PETSC_VIEWER_STDOUT_WORLD)); */

  CHKERRQ(VecDestroy(&X));
  CHKERRQ(VecDestroy(&F));
  CHKERRQ(SNESDestroy(&snes));
  CHKERRQ(DMDestroy(&networkdm));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
      depends: waterreaddata.c waterfunctions.c
      requires: !complex double defined(PETSC_HAVE_ATTRIBUTEALIGNED)

   test:
      args: -water_snes_converged_reason -options_left no
      localrunfiles: wateroptions sample1.inp
      output_file: output/water.out
      requires: double !complex defined(PETSC_HAVE_ATTRIBUTEALIGNED)

   test:
      suffix: 2
      nsize: 3
      args: -water_snes_converged_reason -options_left no
      localrunfiles: wateroptions sample1.inp
      output_file: output/water.out
      requires: double !complex defined(PETSC_HAVE_ATTRIBUTEALIGNED)

TEST*/
