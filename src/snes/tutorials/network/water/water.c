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
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&crank);CHKERRQ(ierr);

  /* Create an empty network object */
  ierr = DMNetworkCreate(PETSC_COMM_WORLD,&networkdm);CHKERRQ(ierr);

  /* Register the components in the network */
  ierr = DMNetworkRegisterComponent(networkdm,"edgestruct",sizeof(struct _p_EDGE_Water),&appctx.compkey_edge);CHKERRQ(ierr);
  ierr = DMNetworkRegisterComponent(networkdm,"busstruct",sizeof(struct _p_VERTEX_Water),&appctx.compkey_vtx);CHKERRQ(ierr);

  ierr = PetscLogStageRegister("Read Data",&stage1);CHKERRQ(ierr);
  ierr = PetscLogStagePush(stage1);CHKERRQ(ierr);
  ierr = PetscNew(&waterdata);CHKERRQ(ierr);

  /* READ THE DATA */
  if (!crank) {
    /* READ DATA. Only rank 0 reads the data */
    ierr = PetscOptionsGetString(NULL,NULL,"-waterdata",waterdata_file,sizeof(waterdata_file),NULL);CHKERRQ(ierr);
    ierr = WaterReadData(waterdata,waterdata_file);CHKERRQ(ierr);

    ierr = PetscCalloc1(2*waterdata->nedge,&edgelist);CHKERRQ(ierr);
    ierr = GetListofEdges_Water(waterdata,edgelist);CHKERRQ(ierr);
  }
  ierr = PetscLogStagePop();CHKERRQ(ierr);

  ierr = PetscLogStageRegister("Create network",&stage2);CHKERRQ(ierr);
  ierr = PetscLogStagePush(stage2);CHKERRQ(ierr);

  /* Set numbers of nodes and edges */
  ierr = DMNetworkSetSizes(networkdm,1,&waterdata->nvertex,&waterdata->nedge,0,NULL);CHKERRQ(ierr);
  if (!crank) {
    ierr = PetscPrintf(PETSC_COMM_SELF,"water nvertices %D, nedges %D\n",waterdata->nvertex,waterdata->nedge);CHKERRQ(ierr);
  }

  /* Add edge connectivity */
  ierr = DMNetworkSetEdgeList(networkdm,&edgelist,NULL);CHKERRQ(ierr);

  /* Set up the network layout */
  ierr = DMNetworkLayoutSetUp(networkdm);CHKERRQ(ierr);

  if (!crank) {
    ierr = PetscFree(edgelist);CHKERRQ(ierr);
  }

  /* ADD VARIABLES AND COMPONENTS FOR THE NETWORK */
  ierr = DMNetworkGetSubnetworkInfo(networkdm,0,&nv,&ne,&vtx,&edges);CHKERRQ(ierr);

  for (i = 0; i < ne; i++) {
    ierr = DMNetworkAddComponent(networkdm,edges[i],appctx.compkey_edge,&waterdata->edge[i]);CHKERRQ(ierr);
  }

  for (i = 0; i < nv; i++) {
    ierr = DMNetworkAddComponent(networkdm,vtx[i],appctx.compkey_vtx,&waterdata->vertex[i]);CHKERRQ(ierr);
    /* Add number of variables */
    ierr = DMNetworkAddNumVariables(networkdm,vtx[i],1);CHKERRQ(ierr);
  }

  /* Set up DM for use */
  ierr = DMSetUp(networkdm);CHKERRQ(ierr);

  if (!crank) {
    ierr = PetscFree(waterdata->vertex);CHKERRQ(ierr);
    ierr = PetscFree(waterdata->edge);CHKERRQ(ierr);
  }
  ierr = PetscFree(waterdata);CHKERRQ(ierr);

  /* Distribute networkdm to multiple processes */
  ierr = DMNetworkDistribute(&networkdm,0);CHKERRQ(ierr);

  ierr = PetscLogStagePop();CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(networkdm,&X);CHKERRQ(ierr);
  ierr = VecDuplicate(X,&F);CHKERRQ(ierr);

  /* HOOK UP SOLVER */
  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
  ierr = SNESSetDM(snes,networkdm);CHKERRQ(ierr);
  ierr = SNESSetOptionsPrefix(snes,"water_");CHKERRQ(ierr);
  ierr = SNESSetFunction(snes,F,WaterFormFunction,NULL);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  ierr = WaterSetInitialGuess(networkdm,X);CHKERRQ(ierr);
  /* ierr = VecView(X,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); */

  ierr = SNESSolve(snes,NULL,X);CHKERRQ(ierr);
  ierr = SNESGetConvergedReason(snes,&reason);CHKERRQ(ierr);
  if (reason < 0) SETERRQ(PETSC_COMM_SELF,0,"No solution found for the water network");
  /* ierr = VecView(X,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); */

  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = VecDestroy(&F);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = DMDestroy(&networkdm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
      depends: waterreaddata.c waterfunctions.c
      requires: !complex double define(PETSC_HAVE_ATTRIBUTEALIGNED)

   test:
      args: -water_snes_converged_reason -options_left no
      localrunfiles: wateroptions sample1.inp
      output_file: output/water.out
      requires: double !complex define(PETSC_HAVE_ATTRIBUTEALIGNED)

   test:
      suffix: 2
      nsize: 3
      args: -water_snes_converged_reason -options_left no
      localrunfiles: wateroptions sample1.inp
      output_file: output/water.out
      requires: double !complex define(PETSC_HAVE_ATTRIBUTEALIGNED)

TEST*/
