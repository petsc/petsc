static char help[] = "Create a mesh, refine and coarsen simultaneously, and transfer a field\n\n";

#include <petscdmplex.h>
#include <petscdmforest.h>
#include <petscoptions.h>

#undef __FUNCT__
#define __FUNCT__ "AddIdentityLabel"
static PetscErrorCode AddIdentityLabel(DM dm)
{
  PetscInt       pStart,pEnd,p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMCreateLabel(dm, "identity");CHKERRQ(ierr);
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; p++) {ierr = DMSetLabelValue(dm, "identity", p, p);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "AddAdaptivityLabel"
static PetscErrorCode AddAdaptivityLabel(DM forest,const char name[])
{
  DMLabel        adaptLabel,identLabel;
  PetscInt       cStart, cEnd, c;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMCreateLabel(forest,name);CHKERRQ(ierr);
  ierr = DMGetLabel(forest,name,&adaptLabel);CHKERRQ(ierr);
  ierr = DMLabelSetDefaultValue(adaptLabel,DM_FOREST_COARSEN);CHKERRQ(ierr);
  ierr = DMGetLabel(forest,"identity",&identLabel);CHKERRQ(ierr);
  ierr = DMForestGetCellChart(forest,&cStart,&cEnd);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; c++) {
    PetscInt basePoint;

    ierr = DMLabelGetValue(identLabel,c,&basePoint);CHKERRQ(ierr);
    if (!basePoint) {ierr = DMLabelSetValue(adaptLabel,c,DM_FOREST_REFINE);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MultiaffineFunction"
static PetscErrorCode MultiaffineFunction(PetscInt dim,PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar u[], void *ctx)
{
  PetscFunctionBeginUser;
  u[0] = (x[0] * 1.0 + 2.0) * (x[1] * 3.0 - 4.0) * ((dim == 3) ? (x[2] * 5.0 + 6.0) : 1.);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  MPI_Comm       comm;
  DM             base, preForest, postForest;
  PetscInt       dim = 2;
  PetscFE        fe;
  Vec            preVec, postVecTransfer, postVecExact;
  PetscErrorCode (*funcs[1]) (PetscInt,PetscReal,const PetscReal [],PetscInt,PetscScalar [], void *) = {MultiaffineFunction};
  void           *ctxs[1] = {NULL};
  const PetscInt cells[] = {3, 3, 3};
  PetscReal      diff, tol = PETSC_SMALL;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = PetscOptionsBegin(comm, "", "DMForestTransfer() Test Options", "DMFOREST");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "The dimension (2 or 3)", "ex2.c", dim, &dim, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  /* the base mesh */
  ierr = DMPlexCreateHexBoxMesh(comm, dim, cells, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, &base);CHKERRQ(ierr);
  ierr = AddIdentityLabel(base);CHKERRQ(ierr);
  ierr = DMViewFromOptions(base,NULL,"-dm_base_view");CHKERRQ(ierr);

  /* the pre adaptivity forest */
  ierr = DMCreate(comm,&preForest);CHKERRQ(ierr);
  ierr = DMSetType(preForest,(dim == 2) ? DMP4EST : DMP8EST);CHKERRQ(ierr);
  ierr = DMForestSetBaseDM(preForest,base);CHKERRQ(ierr);
  ierr = DMForestSetInitialRefinement(preForest,1);CHKERRQ(ierr);
  ierr = DMSetUp(preForest);CHKERRQ(ierr);
  ierr = DMViewFromOptions(preForest,NULL,"-dm_pre_view");CHKERRQ(ierr);

  /* the pre adaptivity field */
  ierr = PetscFECreateDefault(preForest,dim,1,PETSC_FALSE,NULL,PETSC_DEFAULT,&fe);CHKERRQ(ierr);
  ierr = DMSetField(preForest,0,(PetscObject)fe);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(preForest,&preVec);CHKERRQ(ierr);
  ierr = DMProjectFunction(preForest,0.,funcs,ctxs,INSERT_VALUES,preVec);CHKERRQ(ierr);
  ierr = VecViewFromOptions(preVec,NULL,"-vec_pre_view");CHKERRQ(ierr);

  /* adapt */
  ierr = AddAdaptivityLabel(preForest,"adapt");CHKERRQ(ierr);
  ierr = DMForestTemplate(preForest,comm,&postForest);CHKERRQ(ierr);
  ierr = DMForestSetAdaptivityLabel(postForest,"adapt");CHKERRQ(ierr);
  ierr = DMSetUp(postForest);CHKERRQ(ierr);
  ierr = DMViewFromOptions(postForest,NULL,"-dm_post_view");CHKERRQ(ierr);

  /* transfer */
  ierr = DMCreateGlobalVector(postForest,&postVecTransfer);CHKERRQ(ierr);
  ierr = DMForestTransferVec(preForest,preVec,postForest,postVecTransfer);CHKERRQ(ierr);
  ierr = VecViewFromOptions(postVecTransfer,NULL,"-vec_post_transfer_view");CHKERRQ(ierr);

  /* the exact post adaptivity field */
  ierr = DMCreateGlobalVector(postForest,&postVecExact);CHKERRQ(ierr);
  ierr = DMProjectFunction(postForest,0.,funcs,ctxs,INSERT_VALUES,postVecExact);CHKERRQ(ierr);
  ierr = VecViewFromOptions(postVecExact,NULL,"-vec_post_exact_view");CHKERRQ(ierr);

  /* compare */
  ierr = VecAXPY(postVecTransfer,-1.,postVecExact);CHKERRQ(ierr);
  ierr = VecViewFromOptions(postVecTransfer,NULL,"-vec_diff_view");CHKERRQ(ierr);
  ierr = VecNorm(postVecTransfer,NORM_2,&diff);CHKERRQ(ierr);

  /* output */
  if (diff < tol) {
    ierr = PetscPrintf(comm,"DMForestTransfer() passes.\n");CHKERRQ(ierr);
  } else {
    ierr = PetscPrintf(comm,"DMForestTransfer() fails with error %g and tolerance %g\n",diff,tol);CHKERRQ(ierr);
  }

  /* cleanup */
  ierr = VecDestroy(&postVecExact);CHKERRQ(ierr);
  ierr = VecDestroy(&postVecTransfer);CHKERRQ(ierr);
  ierr = DMDestroy(&postForest);CHKERRQ(ierr);
  ierr = VecDestroy(&preVec);CHKERRQ(ierr);
  ierr = DMDestroy(&preForest);CHKERRQ(ierr);
  ierr = DMDestroy(&base);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
