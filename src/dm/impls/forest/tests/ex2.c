static char help[] = "Create a mesh, refine and coarsen simultaneously, and transfer a field\n\n";

#include <petscds.h>
#include <petscdmplex.h>
#include <petscdmforest.h>
#include <petscoptions.h>

static PetscErrorCode AddIdentityLabel(DM dm)
{
  PetscInt       pStart,pEnd,p;

  PetscFunctionBegin;
  CHKERRQ(DMCreateLabel(dm, "identity"));
  CHKERRQ(DMPlexGetChart(dm, &pStart, &pEnd));
  for (p = pStart; p < pEnd; p++) CHKERRQ(DMSetLabelValue(dm, "identity", p, p));
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateAdaptivityLabel(DM forest,DMLabel *adaptLabel)
{
  DMLabel        identLabel;
  PetscInt       cStart, cEnd, c;

  PetscFunctionBegin;
  CHKERRQ(DMLabelCreate(PETSC_COMM_SELF,"adapt",adaptLabel));
  CHKERRQ(DMLabelSetDefaultValue(*adaptLabel,DM_ADAPT_COARSEN));
  CHKERRQ(DMGetLabel(forest,"identity",&identLabel));
  CHKERRQ(DMForestGetCellChart(forest,&cStart,&cEnd));
  for (c = cStart; c < cEnd; c++) {
    PetscInt basePoint;

    CHKERRQ(DMLabelGetValue(identLabel,c,&basePoint));
    if (!basePoint) CHKERRQ(DMLabelSetValue(*adaptLabel,c,DM_ADAPT_REFINE));
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

static PetscErrorCode CoordsFunction(PetscInt dim,PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar u[], void *ctx)
{
  PetscInt f;

  PetscFunctionBeginUser;
  for (f=0;f<Nf;f++) u[f] = x[f];
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

  PetscFunctionBegin;
  bcCtx = (bc_func_ctx *) ctx;
  CHKERRQ((bcCtx->func)(bcCtx->dim,time,c,bcCtx->Nf,xG,bcCtx->ctx));
  PetscFunctionReturn(0);
}

static PetscErrorCode IdentifyBadPoints (DM dm, Vec vec, PetscReal tol)
{
  DM             dmplex;
  PetscInt       p, pStart, pEnd, maxDof;
  Vec            vecLocal;
  DMLabel        depthLabel;
  PetscSection   section;

  PetscFunctionBegin;
  CHKERRQ(DMCreateLocalVector(dm, &vecLocal));
  CHKERRQ(DMGlobalToLocalBegin(dm, vec, INSERT_VALUES, vecLocal));
  CHKERRQ(DMGlobalToLocalEnd(dm, vec, INSERT_VALUES, vecLocal));
  CHKERRQ(DMConvert(dm ,DMPLEX, &dmplex));
  CHKERRQ(DMPlexGetChart(dmplex, &pStart, &pEnd));
  CHKERRQ(DMPlexGetDepthLabel(dmplex, &depthLabel));
  CHKERRQ(DMGetLocalSection(dmplex, &section));
  CHKERRQ(PetscSectionGetMaxDof(section, &maxDof));
  for (p = pStart; p < pEnd; p++) {
    PetscInt     s, c, cSize, parent, childID, numChildren;
    PetscInt     cl, closureSize, *closure = NULL;
    PetscScalar *values = NULL;
    PetscBool    bad = PETSC_FALSE;

    CHKERRQ(VecGetValuesSection(vecLocal, section, p, &values));
    CHKERRQ(PetscSectionGetDof(section, p, &cSize));
    for (c = 0; c < cSize; c++) {
      PetscReal absDiff = PetscAbsScalar(values[c]);
      if (absDiff > tol) {bad = PETSC_TRUE; break;}
    }
    if (!bad) continue;
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "Bad point %D\n", p));
    CHKERRQ(DMLabelGetValue(depthLabel, p, &s));
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "  Depth %D\n", s));
    CHKERRQ(DMPlexGetTransitiveClosure(dmplex, p, PETSC_TRUE, &closureSize, &closure));
    for (cl = 0; cl < closureSize; cl++) {
      PetscInt cp = closure[2 * cl];
      CHKERRQ(DMPlexGetTreeParent(dmplex, cp, &parent, &childID));
      if (parent != cp) {
        CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "  Closure point %D (%D) child of %D (ID %D)\n", cl, cp, parent, childID));
      }
      CHKERRQ(DMPlexGetTreeChildren(dmplex, cp, &numChildren, NULL));
      if (numChildren) {
        CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "  Closure point %D (%D) is parent\n", cl, cp));
      }
    }
    CHKERRQ(DMPlexRestoreTransitiveClosure(dmplex, p, PETSC_TRUE, &closureSize, &closure));
    for (c = 0; c < cSize; c++) {
      PetscReal absDiff = PetscAbsScalar(values[c]);
      if (absDiff > tol) {
        CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "  Bad dof %D\n", c));
      }
    }
  }
  CHKERRQ(DMDestroy(&dmplex));
  CHKERRQ(VecDestroy(&vecLocal));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  MPI_Comm       comm;
  DM             base, preForest, postForest;
  PetscInt       dim, Nf = 1;
  PetscInt       step, adaptSteps = 1;
  PetscInt       preCount, postCount;
  Vec            preVec, postVecTransfer, postVecExact;
  PetscErrorCode (*funcs[1]) (PetscInt,PetscReal,const PetscReal [],PetscInt,PetscScalar [], void *) = {MultiaffineFunction};
  void           *ctxs[1] = {NULL};
  PetscReal      diff, tol = PETSC_SMALL;
  PetscBool      linear = PETSC_FALSE;
  PetscBool      coords = PETSC_FALSE;
  PetscBool      useFV = PETSC_FALSE;
  PetscBool      conv = PETSC_FALSE;
  PetscBool      transfer_from_base[2] = {PETSC_TRUE,PETSC_FALSE};
  PetscBool      use_bcs = PETSC_TRUE;
  bc_func_ctx    bcCtx;
  DMLabel        adaptLabel;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = PetscOptionsBegin(comm, "", "DMForestTransferVec() Test Options", "DMFOREST");CHKERRQ(ierr);
  CHKERRQ(PetscOptionsBool("-linear","Transfer a simple linear function", "ex2.c", linear, &linear, NULL));
  CHKERRQ(PetscOptionsBool("-coords","Transfer a simple coordinate function", "ex2.c", coords, &coords, NULL));
  CHKERRQ(PetscOptionsBool("-use_fv","Use a finite volume approximation", "ex2.c", useFV, &useFV, NULL));
  CHKERRQ(PetscOptionsBool("-test_convert","Test conversion to DMPLEX",NULL,conv,&conv,NULL));
  CHKERRQ(PetscOptionsBool("-transfer_from_base","Transfer a vector from base DM to DMForest", "ex2.c", transfer_from_base[0], &transfer_from_base[0], NULL));
  transfer_from_base[1] = transfer_from_base[0];
  CHKERRQ(PetscOptionsBool("-transfer_from_base_steps","Transfer a vector from base DM to the latest DMForest after the adaptivity steps", "ex2.c", transfer_from_base[1], &transfer_from_base[1], NULL));
  CHKERRQ(PetscOptionsBool("-use_bcs","Use dirichlet boundary conditions", "ex2.c", use_bcs, &use_bcs, NULL));
  CHKERRQ(PetscOptionsBoundedInt("-adapt_steps","Number of adaptivity steps", "ex2.c", adaptSteps, &adaptSteps, NULL,0));
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  tol = PetscMax(1.e-10,tol); /* XXX fix for quadruple precision -> why do I need to do this? */

  /* the base mesh */
  CHKERRQ(DMCreate(comm, &base));
  CHKERRQ(DMSetType(base, DMPLEX));
  CHKERRQ(DMSetFromOptions(base));

  CHKERRQ(AddIdentityLabel(base));
  CHKERRQ(DMGetDimension(base, &dim));

  if (linear) {
    funcs[0] = LinearFunction;
  }
  if (coords) {
    funcs[0] = CoordsFunction;
    Nf = dim;
  }

  bcCtx.func = funcs[0];
  bcCtx.dim  = dim;
  bcCtx.Nf   = Nf;
  bcCtx.ctx  = NULL;

  if (useFV) {
    PetscFV      fv;
    PetscLimiter limiter;
    DM           baseFV;

    CHKERRQ(DMPlexConstructGhostCells(base,NULL,NULL,&baseFV));
    CHKERRQ(DMViewFromOptions(baseFV, NULL, "-fv_dm_view"));
    CHKERRQ(DMDestroy(&base));
    base = baseFV;
    CHKERRQ(PetscFVCreate(comm, &fv));
    CHKERRQ(PetscFVSetSpatialDimension(fv,dim));
    CHKERRQ(PetscFVSetType(fv,PETSCFVLEASTSQUARES));
    CHKERRQ(PetscFVSetNumComponents(fv,Nf));
    CHKERRQ(PetscLimiterCreate(comm,&limiter));
    CHKERRQ(PetscLimiterSetType(limiter,PETSCLIMITERNONE));
    CHKERRQ(PetscFVSetLimiter(fv,limiter));
    CHKERRQ(PetscLimiterDestroy(&limiter));
    CHKERRQ(PetscFVSetFromOptions(fv));
    CHKERRQ(DMSetField(base,0,NULL,(PetscObject)fv));
    CHKERRQ(PetscFVDestroy(&fv));
  } else {
    PetscFE fe;

    CHKERRQ(PetscFECreateDefault(comm,dim,Nf,PETSC_FALSE,NULL,PETSC_DEFAULT,&fe));
    CHKERRQ(DMSetField(base,0,NULL,(PetscObject)fe));
    CHKERRQ(PetscFEDestroy(&fe));
  }
  CHKERRQ(DMCreateDS(base));

  if (use_bcs) {
    PetscInt ids[] = {1, 2, 3, 4, 5, 6};
    DMLabel  label;

    CHKERRQ(DMGetLabel(base, "marker", &label));
    CHKERRQ(DMAddBoundary(base,DM_BC_ESSENTIAL, "bc", label, 2 * dim, ids, 0, 0, NULL, useFV ? (void(*)(void)) bc_func_fv : (void(*)(void)) funcs[0], NULL, useFV ? (void *) &bcCtx : NULL, NULL));
  }
  CHKERRQ(DMViewFromOptions(base,NULL,"-dm_base_view"));

  /* the pre adaptivity forest */
  CHKERRQ(DMCreate(comm,&preForest));
  CHKERRQ(DMSetType(preForest,(dim == 2) ? DMP4EST : DMP8EST));
  CHKERRQ(DMCopyDisc(base,preForest));
  CHKERRQ(DMForestSetBaseDM(preForest,base));
  CHKERRQ(DMForestSetMinimumRefinement(preForest,0));
  CHKERRQ(DMForestSetInitialRefinement(preForest,1));
  CHKERRQ(DMSetFromOptions(preForest));
  CHKERRQ(DMSetUp(preForest));
  CHKERRQ(DMViewFromOptions(preForest,NULL,"-dm_pre_view"));

  /* the pre adaptivity field */
  CHKERRQ(DMCreateGlobalVector(preForest,&preVec));
  CHKERRQ(DMProjectFunction(preForest,0.,funcs,ctxs,INSERT_VALUES,preVec));
  CHKERRQ(VecViewFromOptions(preVec,NULL,"-vec_pre_view"));

  /* communicate between base and pre adaptivity forest */
  if (transfer_from_base[0]) {
    Vec baseVec, baseVecMapped;

    CHKERRQ(DMGetGlobalVector(base,&baseVec));
    CHKERRQ(DMProjectFunction(base,0.,funcs,ctxs,INSERT_VALUES,baseVec));
    CHKERRQ(PetscObjectSetName((PetscObject)baseVec,"Function Base"));
    CHKERRQ(VecViewFromOptions(baseVec,NULL,"-vec_base_view"));

    CHKERRQ(DMGetGlobalVector(preForest,&baseVecMapped));
    CHKERRQ(DMForestTransferVecFromBase(preForest,baseVec,baseVecMapped));
    CHKERRQ(VecViewFromOptions(baseVecMapped,NULL,"-vec_map_base_view"));

    /* compare */
    CHKERRQ(VecAXPY(baseVecMapped,-1.,preVec));
    CHKERRQ(VecViewFromOptions(baseVecMapped,NULL,"-vec_map_diff_view"));
    CHKERRQ(VecNorm(baseVecMapped,NORM_2,&diff));

    /* output */
    if (diff < tol) {
      CHKERRQ(PetscPrintf(comm,"DMForestTransferVecFromBase() passes.\n"));
    } else {
      CHKERRQ(PetscPrintf(comm,"DMForestTransferVecFromBase() fails with error %g and tolerance %g\n",(double)diff,(double)tol));
    }

    CHKERRQ(DMRestoreGlobalVector(base,&baseVec));
    CHKERRQ(DMRestoreGlobalVector(preForest,&baseVecMapped));
  }

  for (step = 0; step < adaptSteps; ++step) {

    if (!transfer_from_base[1]) {
      CHKERRQ(PetscObjectGetReference((PetscObject)preForest,&preCount));
    }

    /* adapt */
    CHKERRQ(CreateAdaptivityLabel(preForest,&adaptLabel));
    CHKERRQ(DMForestTemplate(preForest,comm,&postForest));
    if (step) CHKERRQ(DMForestSetAdaptivityLabel(postForest,adaptLabel));
    CHKERRQ(DMLabelDestroy(&adaptLabel));
    CHKERRQ(DMSetUp(postForest));
    CHKERRQ(DMViewFromOptions(postForest,NULL,"-dm_post_view"));

    /* transfer */
    CHKERRQ(DMCreateGlobalVector(postForest,&postVecTransfer));
    CHKERRQ(DMForestTransferVec(preForest,preVec,postForest,postVecTransfer,PETSC_TRUE,0.0));
    CHKERRQ(VecViewFromOptions(postVecTransfer,NULL,"-vec_post_transfer_view"));

    /* the exact post adaptivity field */
    CHKERRQ(DMCreateGlobalVector(postForest,&postVecExact));
    CHKERRQ(DMProjectFunction(postForest,0.,funcs,ctxs,INSERT_VALUES,postVecExact));
    CHKERRQ(VecViewFromOptions(postVecExact,NULL,"-vec_post_exact_view"));

    /* compare */
    CHKERRQ(VecAXPY(postVecExact,-1.,postVecTransfer));
    CHKERRQ(VecViewFromOptions(postVecExact,NULL,"-vec_diff_view"));
    CHKERRQ(VecNorm(postVecExact,NORM_2,&diff));

    /* output */
    if (diff < tol) {
      CHKERRQ(PetscPrintf(comm,"DMForestTransferVec() passes.\n"));
    } else {
      CHKERRQ(PetscPrintf(comm,"DMForestTransferVec() fails with error %g and tolerance %g\n",(double)diff,(double)tol));
      CHKERRQ(IdentifyBadPoints(postForest, postVecExact, tol));
    }
    CHKERRQ(VecDestroy(&postVecExact));

    /* disconnect preForest from postForest if we don't test the transfer throughout the entire refinement process */
    if (!transfer_from_base[1]) {
      CHKERRQ(DMForestSetAdaptivityForest(postForest,NULL));
      CHKERRQ(PetscObjectGetReference((PetscObject)preForest,&postCount));
      PetscCheckFalse(postCount != preCount,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Adaptation not memory neutral: reference count increase from %d to %d",preCount,postCount);
    }

    if (conv) {
      DM dmConv;

      CHKERRQ(DMConvert(postForest,DMPLEX,&dmConv));
      CHKERRQ(DMViewFromOptions(dmConv,NULL,"-dm_conv_view"));
      CHKERRQ(DMPlexCheckCellShape(dmConv,PETSC_TRUE,PETSC_DETERMINE));
      CHKERRQ(DMDestroy(&dmConv));
    }

    CHKERRQ(VecDestroy(&preVec));
    CHKERRQ(DMDestroy(&preForest));

    preVec    = postVecTransfer;
    preForest = postForest;
  }

  if (transfer_from_base[1]) {
    Vec baseVec, baseVecMapped;

    /* communicate between base and last adapted forest */
    CHKERRQ(DMGetGlobalVector(base,&baseVec));
    CHKERRQ(DMProjectFunction(base,0.,funcs,ctxs,INSERT_VALUES,baseVec));
    CHKERRQ(PetscObjectSetName((PetscObject)baseVec,"Function Base"));
    CHKERRQ(VecViewFromOptions(baseVec,NULL,"-vec_base_view"));

    CHKERRQ(DMGetGlobalVector(preForest,&baseVecMapped));
    CHKERRQ(DMForestTransferVecFromBase(preForest,baseVec,baseVecMapped));
    CHKERRQ(VecViewFromOptions(baseVecMapped,NULL,"-vec_map_base_view"));

    /* compare */
    CHKERRQ(VecAXPY(baseVecMapped,-1.,preVec));
    CHKERRQ(VecViewFromOptions(baseVecMapped,NULL,"-vec_map_diff_view"));
    CHKERRQ(VecNorm(baseVecMapped,NORM_2,&diff));

    /* output */
    if (diff < tol) {
      CHKERRQ(PetscPrintf(comm,"DMForestTransferVecFromBase() passes.\n"));
    } else {
      CHKERRQ(PetscPrintf(comm,"DMForestTransferVecFromBase() fails with error %g and tolerance %g\n",(double)diff,(double)tol));
    }

    CHKERRQ(DMRestoreGlobalVector(base,&baseVec));
    CHKERRQ(DMRestoreGlobalVector(preForest,&baseVecMapped));
  }

  /* cleanup */
  CHKERRQ(VecDestroy(&preVec));
  CHKERRQ(DMDestroy(&preForest));
  CHKERRQ(DMDestroy(&base));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST
  testset:
    args: -dm_plex_simplex 0 -dm_plex_box_faces 3,3,3 -petscspace_type tensor

    test:
      output_file: output/ex2_2d.out
      suffix: p4est_2d
      args: -petscspace_degree 2
      nsize: 3
      requires: p4est !single

    test:
      output_file: output/ex2_2d.out
      suffix: p4est_2d_deg4
      args: -petscspace_degree 4
      requires: p4est !single

    test:
      output_file: output/ex2_2d.out
      suffix: p4est_2d_deg8
      args: -petscspace_degree 8
      requires: p4est !single

    test:
      output_file: output/ex2_steps2.out
      suffix: p4est_2d_deg2_steps2
      args: -petscspace_degree 2 -coords -adapt_steps 2
      nsize: 3
      requires: p4est !single

    test:
      output_file: output/ex2_steps3.out
      suffix: p4est_2d_deg3_steps3
      args: -petscspace_degree 3 -coords -adapt_steps 3 -petscdualspace_lagrange_node_type equispaced -petscdualspace_lagrange_node_endpoints 1
      nsize: 3
      requires: p4est !single

    test:
      output_file: output/ex2_steps3.out
      suffix: p4est_2d_deg3_steps3_L2_periodic
      args: -petscspace_degree 3 -petscdualspace_lagrange_continuity 0 -coords -adapt_steps 3 -dm_plex_box_bd periodic,periodic -use_bcs 0 -petscdualspace_lagrange_node_type equispaced
      nsize: 3
      requires: p4est !single

    test:
      output_file: output/ex2_steps3.out
      suffix: p4est_3d_deg2_steps3_L2_periodic
      args: -dm_plex_dim 3 -petscspace_degree 2 -petscdualspace_lagrange_continuity 0 -coords -adapt_steps 3 -dm_plex_box_bd periodic,periodic,periodic -use_bcs 0
      nsize: 3
      requires: p4est !single

    test:
      output_file: output/ex2_steps2.out
      suffix: p4est_3d_deg2_steps2
      args: -dm_plex_dim 3 -petscspace_degree 2 -coords -adapt_steps 2
      nsize: 3
      requires: p4est !single

    test:
      output_file: output/ex2_steps3.out
      suffix: p4est_3d_deg3_steps3
      args: -dm_plex_dim 3 -petscspace_degree 3 -coords -adapt_steps 3 -petscdualspace_lagrange_node_type equispaced -petscdualspace_lagrange_node_endpoints 1
      nsize: 3
      requires: p4est !single

    test:
      output_file: output/ex2_3d.out
      suffix: p4est_3d
      args: -dm_plex_dim 3 -petscspace_degree 1
      nsize: 3
      requires: p4est !single

    test:
      output_file: output/ex2_3d.out
      suffix: p4est_3d_deg3
      args: -dm_plex_dim 3 -petscspace_degree 3
      nsize: 3
      requires: p4est !single

    test:
      output_file: output/ex2_2d.out
      suffix: p4est_2d_deg2_coords
      args: -petscspace_degree 2 -coords
      nsize: 3
      requires: p4est !single

    test:
      output_file: output/ex2_3d.out
      suffix: p4est_3d_deg2_coords
      args: -dm_plex_dim 3 -petscspace_degree 2 -coords
      nsize: 3
      requires: p4est !single

    test:
      suffix: p4est_3d_nans
      args: -dm_plex_dim 3 -dm_forest_partition_overlap 1 -test_convert -petscspace_degree 1
      nsize: 2
      requires: p4est !single

    test:
      TODO: not broken, but the 3D case below is broken, so I do not trust this one
      output_file: output/ex2_steps2.out
      suffix: p4est_2d_tfb_distributed_nc
      args: -petscspace_degree 3 -dm_forest_maximum_refinement 2 -dm_p4est_refine_pattern hash -use_bcs 0 -coords -adapt_steps 2 -petscpartitioner_type shell -petscpartitioner_shell_random
      nsize: 3
      requires: p4est !single

    test:
      TODO: broken
      output_file: output/ex2_steps2.out
      suffix: p4est_3d_tfb_distributed_nc
      args: -dm_plex_dim 3 -petscspace_degree 2 -dm_forest_maximum_refinement 2 -dm_p4est_refine_pattern hash -use_bcs 0 -coords -adapt_steps 2 -petscpartitioner_type shell -petscpartitioner_shell_random
      nsize: 3
      requires: p4est !single

  testset:
    args: -petscspace_type tensor -dm_coord_space 0 -dm_plex_transform_type refine_tobox

    test:
      TODO: broken
      output_file: output/ex2_3d.out
      suffix: p4est_3d_transfer_fails
      args: -petscspace_degree 1 -dm_plex_filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/doublet-tet.msh -adapt_steps 1 -dm_forest_initial_refinement 1 -use_bcs 0 -dm_refine
      requires: p4est !single

    test:
      TODO: broken
      output_file: output/ex2_steps2_notfb.out
      suffix: p4est_3d_transfer_fails_2
      args: -petscspace_degree 1 -dm_plex_filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/doublet-tet.msh -adapt_steps 2 -dm_forest_initial_refinement 0 -transfer_from_base 0 -use_bcs 0 -dm_refine
      requires: p4est !single

    test:
      output_file: output/ex2_steps2.out
      suffix: p4est_3d_multi_transfer_s2t
      args: -petscspace_degree 3 -dm_plex_filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/doublet-tet.msh -adapt_steps 2 -dm_forest_initial_refinement 1 -petscdualspace_lagrange_continuity 0 -use_bcs 0 -dm_refine 1
      requires: p4est !single

    test:
      output_file: output/ex2_steps2.out
      suffix: p4est_3d_coords_transfer_s2t
      args: -petscspace_degree 3 -dm_plex_filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/doublet-tet.msh -adapt_steps 2 -dm_forest_initial_refinement 1 -petscdualspace_lagrange_continuity 0 -coords -use_bcs 0 -dm_refine 1
      requires: p4est !single

  testset:
    args: -dm_plex_simplex 0 -dm_plex_box_faces 3,3,3

    test:
      output_file: output/ex2_2d_fv.out
      suffix: p4est_2d_fv
      args: -transfer_from_base 0 -use_fv -linear -dm_forest_partition_overlap 1
      nsize: 3
      requires: p4est !single

    test:
      TODO: broken (codimension adjacency)
      output_file: output/ex2_2d_fv.out
      suffix: p4est_2d_fv_adjcodim
      args: -transfer_from_base 0 -use_fv -linear -dm_forest_partition_overlap 1 -dm_forest_adjacency_codimension 1
      nsize: 2
      requires: p4est !single

    test:
      TODO: broken (dimension adjacency)
      output_file: output/ex2_2d_fv.out
      suffix: p4est_2d_fv_adjdim
      args: -transfer_from_base 0 -use_fv -linear -dm_forest_partition_overlap 1 -dm_forest_adjacency_dimension 1
      nsize: 2
      requires: p4est !single

    test:
      output_file: output/ex2_2d_fv.out
      suffix: p4est_2d_fv_zerocells
      args: -transfer_from_base 0 -use_fv -linear -dm_forest_partition_overlap 1
      nsize: 10
      requires: p4est !single

    test:
      output_file: output/ex2_3d_fv.out
      suffix: p4est_3d_fv
      args: -dm_plex_dim 3 -transfer_from_base 0 -use_fv -linear -dm_forest_partition_overlap 1
      nsize: 3
      requires: p4est !single

TEST*/
