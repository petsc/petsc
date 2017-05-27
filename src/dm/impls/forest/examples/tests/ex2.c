static char help[] = "Create a mesh, refine and coarsen simultaneously, and transfer a field\n\n";

#include <petscds.h>
#include <petscdmplex.h>
#include <petscdmforest.h>
#include <petscoptions.h>

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

static PetscErrorCode CreateAdaptivityLabel(DM forest,DMLabel *adaptLabel)
{
  DMLabel        identLabel;
  PetscInt       cStart, cEnd, c;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMLabelCreate("adapt",adaptLabel);CHKERRQ(ierr);
  ierr = DMLabelSetDefaultValue(*adaptLabel,DM_ADAPT_COARSEN);CHKERRQ(ierr);
  ierr = DMGetLabel(forest,"identity",&identLabel);CHKERRQ(ierr);
  ierr = DMForestGetCellChart(forest,&cStart,&cEnd);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; c++) {
    PetscInt basePoint;

    ierr = DMLabelGetValue(identLabel,c,&basePoint);CHKERRQ(ierr);
    if (!basePoint) {ierr = DMLabelSetValue(*adaptLabel,c,DM_ADAPT_REFINE);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode LinearFunction(PetscInt dim,PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar u[], void *ctx)
{
  PetscFunctionBeginUser;
  u[0] = (x[0] * 2.0 + 1.) + (x[1] * 20.0 + 10.) + ((dim == 3) ? (x[2] * 200.0 + 100.) : 0.);
  PetscFunctionReturn(0);
}

static PetscErrorCode MultiaffineFunction(PetscInt dim,PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar u[], void *ctx)
{
  PetscFunctionBeginUser;
  u[0] = (x[0] * 1.0 + 2.0) * (x[1] * 3.0 - 4.0) * ((dim == 3) ? (x[2] * 5.0 + 6.0) : 1.);
  PetscFunctionReturn(0);
}

typedef struct _bc_func_ctx
{
  PetscErrorCode (*func) (PetscInt,PetscReal,const PetscReal [], PetscInt, PetscScalar [], void *);
  PetscInt dim;
  PetscInt Nf;
  void *ctx;
}
bc_func_ctx;

static PetscErrorCode bc_func_fv (PetscReal time, const PetscReal *c, const PetscReal *n, const PetscScalar *xI, PetscScalar *xG, void *ctx)
{
  bc_func_ctx    *bcCtx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  bcCtx = (bc_func_ctx *) ctx;
  ierr = (bcCtx->func)(bcCtx->dim,time,c,bcCtx->Nf,xG,bcCtx->ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  MPI_Comm       comm;
  DM             base, preForest, postForest;
  PetscInt       dim = 2;
  PetscInt       preCount, postCount;
  Vec            preVec, postVecTransfer, postVecExact;
  PetscErrorCode (*funcs[1]) (PetscInt,PetscReal,const PetscReal [],PetscInt,PetscScalar [], void *) = {MultiaffineFunction};
  void           *ctxs[1] = {NULL};
  const PetscInt cells[] = {3, 3, 3};
  PetscReal      diff, tol = PETSC_SMALL;
  PetscBool      linear = PETSC_FALSE;
  PetscBool      useFV = PETSC_FALSE;
  PetscDS        ds;
  bc_func_ctx    bcCtx;
  DMLabel        adaptLabel;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = PetscOptionsBegin(comm, "", "DMForestTransferVec() Test Options", "DMFOREST");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "The dimension (2 or 3)", "ex2.c", dim, &dim, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-linear","Transfer a simple linear function", "ex2.c", linear, &linear, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-use_fv","Use a finite volume approximation", "ex2.c", useFV, &useFV, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  if (linear) {
    funcs[0] = LinearFunction;
  }

  bcCtx.func = funcs[0];
  bcCtx.dim  = dim;
  bcCtx.Nf   = 1;
  bcCtx.ctx  = NULL;

  /* the base mesh */
  ierr = DMPlexCreateHexBoxMesh(comm, dim, cells, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, &base);CHKERRQ(ierr);
  if (useFV) {
    PetscFV      fv;
    PetscLimiter limiter;
    DM           baseFV;

    ierr = DMPlexConstructGhostCells(base,NULL,NULL,&baseFV);CHKERRQ(ierr);
    ierr = DMDestroy(&base);CHKERRQ(ierr);
    base = baseFV;
    ierr = PetscFVCreate(comm, &fv);CHKERRQ(ierr);
    ierr = PetscFVSetSpatialDimension(fv,dim);CHKERRQ(ierr);
    ierr = PetscFVSetType(fv,PETSCFVLEASTSQUARES);CHKERRQ(ierr);
    ierr = PetscFVSetNumComponents(fv,1);CHKERRQ(ierr);
    ierr = PetscLimiterCreate(comm,&limiter);CHKERRQ(ierr);
    ierr = PetscLimiterSetType(limiter,PETSCLIMITERNONE);CHKERRQ(ierr);
    ierr = PetscFVSetLimiter(fv,limiter);CHKERRQ(ierr);
    ierr = PetscLimiterDestroy(&limiter);CHKERRQ(ierr);
    ierr = PetscFVSetFromOptions(fv);CHKERRQ(ierr);
    ierr = DMSetField(base,0,(PetscObject)fv);CHKERRQ(ierr);
    ierr = PetscFVDestroy(&fv);CHKERRQ(ierr);
  } else {
    PetscFE fe;
    ierr = PetscFECreateDefault(base,dim,1,PETSC_FALSE,NULL,PETSC_DEFAULT,&fe);CHKERRQ(ierr);
    ierr = DMSetField(base,0,(PetscObject)fe);CHKERRQ(ierr);
    ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
  }
  {
    PetscDS  prob;
    PetscInt comps[] = {0};
    PetscInt ids[]   = {1, 2, 3, 4, 5, 6};

    ierr = DMGetDS(base,&prob);CHKERRQ(ierr);
    ierr = PetscDSAddBoundary(prob,PETSC_TRUE, "bc", "marker", 0, 1, comps, useFV ? (void(*)()) bc_func_fv : (void(*)()) funcs[0], 2 * dim, ids, useFV ? (void *) &bcCtx : NULL);CHKERRQ(ierr);
  }
  ierr = AddIdentityLabel(base);CHKERRQ(ierr);
  ierr = DMViewFromOptions(base,NULL,"-dm_base_view");CHKERRQ(ierr);

  /* the pre adaptivity forest */
  ierr = DMCreate(comm,&preForest);CHKERRQ(ierr);
  ierr = DMSetType(preForest,(dim == 2) ? DMP4EST : DMP8EST);CHKERRQ(ierr);
  ierr = DMGetDS(base,&ds);CHKERRQ(ierr);
  ierr = DMSetDS(preForest,ds);CHKERRQ(ierr);
  ierr = DMForestSetBaseDM(preForest,base);CHKERRQ(ierr);
  ierr = DMForestSetMinimumRefinement(preForest,1);CHKERRQ(ierr);
  ierr = DMForestSetInitialRefinement(preForest,1);CHKERRQ(ierr);
  ierr = DMSetFromOptions(preForest);CHKERRQ(ierr);
  ierr = DMSetUp(preForest);CHKERRQ(ierr);
  ierr = DMViewFromOptions(preForest,NULL,"-dm_pre_view");CHKERRQ(ierr);

  /* the pre adaptivity field */
  ierr = DMCreateGlobalVector(preForest,&preVec);CHKERRQ(ierr);
  ierr = DMProjectFunction(preForest,0.,funcs,ctxs,INSERT_VALUES,preVec);CHKERRQ(ierr);
  ierr = VecViewFromOptions(preVec,NULL,"-vec_pre_view");CHKERRQ(ierr);

  ierr = PetscObjectGetReference((PetscObject)preForest,&preCount);CHKERRQ(ierr);

  /* adapt */
  ierr = CreateAdaptivityLabel(preForest,&adaptLabel);CHKERRQ(ierr);
  ierr = DMForestTemplate(preForest,comm,&postForest);CHKERRQ(ierr);
  ierr = DMForestSetMinimumRefinement(postForest,0);CHKERRQ(ierr);
  ierr = DMForestSetInitialRefinement(postForest,0);CHKERRQ(ierr);
  ierr = DMForestSetAdaptivityLabel(postForest,adaptLabel);CHKERRQ(ierr);
  ierr = DMLabelDestroy(&adaptLabel);CHKERRQ(ierr);
  ierr = DMSetUp(postForest);CHKERRQ(ierr);
  ierr = DMViewFromOptions(postForest,NULL,"-dm_post_view");CHKERRQ(ierr);

  /* transfer */
  ierr = DMCreateGlobalVector(postForest,&postVecTransfer);CHKERRQ(ierr);
  ierr = DMForestTransferVec(preForest,preVec,postForest,postVecTransfer,PETSC_TRUE,0.0);CHKERRQ(ierr);
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
    ierr = PetscPrintf(comm,"DMForestTransferVec() passes.\n");CHKERRQ(ierr);
  } else {
    ierr = PetscPrintf(comm,"DMForestTransferVec() fails with error %g and tolerance %g\n",diff,tol);CHKERRQ(ierr);
  }

  /* disconnect preForest from postForest */
  ierr = DMForestSetAdaptivityForest(postForest,NULL);CHKERRQ(ierr);
  ierr = PetscObjectGetReference((PetscObject)preForest,&postCount);CHKERRQ(ierr);
  if (postCount != preCount) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Adaptation not memory neutral: reference count increase from %d to %d\n",preCount,postCount);

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
