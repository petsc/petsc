static char help[] = "This example demonstrates the use of DMNetwork interface for solving a steady-state water network model.\n\
                      The water network equations follow those used for the package EPANET\n\
                      The data file format used is from the EPANET package (https://www.epa.gov/water-research/epanet).\n\
                      Run this program: mpiexec -n <n> ./water\n\\n";

#include "water.h"
#include <petscdmnetwork.h>

int main(int argc,char ** argv)
{
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

  PetscCall(PetscInitialize(&argc,&argv,"wateroptions",help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&crank));

  /* Create an empty network object */
  PetscCall(DMNetworkCreate(PETSC_COMM_WORLD,&networkdm));

  /* Register the components in the network */
  PetscCall(DMNetworkRegisterComponent(networkdm,"edgestruct",sizeof(struct _p_EDGE_Water),&appctx.compkey_edge));
  PetscCall(DMNetworkRegisterComponent(networkdm,"busstruct",sizeof(struct _p_VERTEX_Water),&appctx.compkey_vtx));

  PetscCall(PetscLogStageRegister("Read Data",&stage1));
  PetscCall(PetscLogStagePush(stage1));
  PetscCall(PetscNew(&waterdata));

  /* READ THE DATA */
  if (!crank) {
    /* READ DATA. Only rank 0 reads the data */
    PetscCall(PetscOptionsGetString(NULL,NULL,"-waterdata",waterdata_file,sizeof(waterdata_file),NULL));
    PetscCall(WaterReadData(waterdata,waterdata_file));

    PetscCall(PetscCalloc1(2*waterdata->nedge,&edgelist));
    PetscCall(GetListofEdges_Water(waterdata,edgelist));
  }
  PetscCall(PetscLogStagePop());

  PetscCall(PetscLogStageRegister("Create network",&stage2));
  PetscCall(PetscLogStagePush(stage2));

  /* Set numbers of nodes and edges */
  PetscCall(DMNetworkSetNumSubNetworks(networkdm,PETSC_DECIDE,1));
  PetscCall(DMNetworkAddSubnetwork(networkdm,"",waterdata->nedge,edgelist,NULL));
  if (!crank) {
    PetscCall(PetscPrintf(PETSC_COMM_SELF,"water nvertices %" PetscInt_FMT ", nedges %" PetscInt_FMT "\n",waterdata->nvertex,waterdata->nedge));
  }

  /* Set up the network layout */
  PetscCall(DMNetworkLayoutSetUp(networkdm));

  if (!crank) {
    PetscCall(PetscFree(edgelist));
  }

  /* ADD VARIABLES AND COMPONENTS FOR THE NETWORK */
  PetscCall(DMNetworkGetSubnetwork(networkdm,0,&nv,&ne,&vtx,&edges));

  for (i = 0; i < ne; i++) {
    PetscCall(DMNetworkAddComponent(networkdm,edges[i],appctx.compkey_edge,&waterdata->edge[i],0));
  }

  for (i = 0; i < nv; i++) {
    PetscCall(DMNetworkAddComponent(networkdm,vtx[i],appctx.compkey_vtx,&waterdata->vertex[i],1));
  }

  /* Set up DM for use */
  PetscCall(DMSetUp(networkdm));

  if (!crank) {
    PetscCall(PetscFree(waterdata->vertex));
    PetscCall(PetscFree(waterdata->edge));
  }
  PetscCall(PetscFree(waterdata));

  /* Distribute networkdm to multiple processes */
  PetscCall(DMNetworkDistribute(&networkdm,0));

  PetscCall(PetscLogStagePop());

  PetscCall(DMCreateGlobalVector(networkdm,&X));
  PetscCall(VecDuplicate(X,&F));

  /* HOOK UP SOLVER */
  PetscCall(SNESCreate(PETSC_COMM_WORLD,&snes));
  PetscCall(SNESSetDM(snes,networkdm));
  PetscCall(SNESSetOptionsPrefix(snes,"water_"));
  PetscCall(SNESSetFunction(snes,F,WaterFormFunction,NULL));
  PetscCall(SNESSetFromOptions(snes));

  PetscCall(WaterSetInitialGuess(networkdm,X));
  /* PetscCall(VecView(X,PETSC_VIEWER_STDOUT_WORLD)); */

  PetscCall(SNESSolve(snes,NULL,X));
  PetscCall(SNESGetConvergedReason(snes,&reason));

  PetscCheck(reason >= 0,PETSC_COMM_SELF,PETSC_ERR_CONV_FAILED,"No solution found for the water network");
  /* PetscCall(VecView(X,PETSC_VIEWER_STDOUT_WORLD)); */

  PetscCall(VecDestroy(&X));
  PetscCall(VecDestroy(&F));
  PetscCall(SNESDestroy(&snes));
  PetscCall(DMDestroy(&networkdm));
  PetscCall(PetscFinalize());
  return 0;
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
