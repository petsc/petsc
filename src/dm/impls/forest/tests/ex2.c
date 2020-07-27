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
  ierr = DMLabelCreate(PETSC_COMM_SELF,"adapt",adaptLabel);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  bcCtx = (bc_func_ctx *) ctx;
  ierr = (bcCtx->func)(bcCtx->dim,time,c,bcCtx->Nf,xG,bcCtx->ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode IdentifyBadPoints (DM dm, Vec vec, PetscReal tol)
{
  DM             dmplex;
  PetscInt       p, pStart, pEnd, maxDof;
  Vec            vecLocal;
  DMLabel        depthLabel;
  PetscSection   section;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMCreateLocalVector(dm, &vecLocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm, vec, INSERT_VALUES, vecLocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm, vec, INSERT_VALUES, vecLocal);CHKERRQ(ierr);
  ierr = DMConvert(dm ,DMPLEX, &dmplex);CHKERRQ(ierr);
  ierr = DMPlexGetChart(dmplex, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = DMPlexGetDepthLabel(dmplex, &depthLabel);CHKERRQ(ierr);
  ierr = DMGetLocalSection(dmplex, &section);CHKERRQ(ierr);
  ierr = PetscSectionGetMaxDof(section, &maxDof);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; p++) {
    PetscInt     s, c, cSize, parent, childID, numChildren;
    PetscInt     cl, closureSize, *closure = NULL;
    PetscScalar *values = NULL;
    PetscBool    bad = PETSC_FALSE;

    ierr = VecGetValuesSection(vecLocal, section, p, &values);CHKERRQ(ierr);
    ierr = PetscSectionGetDof(section, p, &cSize);CHKERRQ(ierr);
    for (c = 0; c < cSize; c++) {
      PetscReal absDiff = PetscAbsScalar(values[c]);CHKERRQ(ierr);
      if (absDiff > tol) {bad = PETSC_TRUE; break;}
    }
    if (!bad) continue;
    ierr = PetscPrintf(PETSC_COMM_SELF, "Bad point %D\n", p);CHKERRQ(ierr);
    ierr = DMLabelGetValue(depthLabel, p, &s);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF, "  Depth %D\n", s);CHKERRQ(ierr);
    ierr = DMPlexGetTransitiveClosure(dmplex, p, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    for (cl = 0; cl < closureSize; cl++) {
      PetscInt cp = closure[2 * cl];
      ierr = DMPlexGetTreeParent(dmplex, cp, &parent, &childID);CHKERRQ(ierr);
      if (parent != cp) {
        ierr = PetscPrintf(PETSC_COMM_SELF, "  Closure point %D (%D) child of %D (ID %D)\n", cl, cp, parent, childID);CHKERRQ(ierr);
      }
      ierr = DMPlexGetTreeChildren(dmplex, cp, &numChildren, NULL);CHKERRQ(ierr);
      if (numChildren) {
        ierr = PetscPrintf(PETSC_COMM_SELF, "  Closure point %D (%D) is parent\n", cl, cp);CHKERRQ(ierr);
      }
    }
    ierr = DMPlexRestoreTransitiveClosure(dmplex, p, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    for (c = 0; c < cSize; c++) {
      PetscReal absDiff = PetscAbsScalar(values[c]);CHKERRQ(ierr);
      if (absDiff > tol) {
        ierr = PetscPrintf(PETSC_COMM_SELF, "  Bad dof %D\n", c);CHKERRQ(ierr);
      }
    }
  }
  ierr = DMDestroy(&dmplex);CHKERRQ(ierr);
  ierr = VecDestroy(&vecLocal);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  MPI_Comm       comm;
  DM             base, preForest, postForest;
  PetscInt       dim = 2, Nf = 1;
  PetscInt       step, adaptSteps = 1;
  PetscInt       preCount, postCount;
  Vec            preVec, postVecTransfer, postVecExact;
  PetscErrorCode (*funcs[1]) (PetscInt,PetscReal,const PetscReal [],PetscInt,PetscScalar [], void *) = {MultiaffineFunction};
  char           filename[PETSC_MAX_PATH_LEN];
  void           *ctxs[1] = {NULL};
  const PetscInt cells[] = {3, 3, 3};
  PetscReal      diff, tol = PETSC_SMALL;
  DMBoundaryType periodicity[] = {DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE};
  PetscBool      linear = PETSC_FALSE;
  PetscBool      coords = PETSC_FALSE;
  PetscBool      useFV = PETSC_FALSE;
  PetscBool      conv = PETSC_FALSE;
  PetscBool      distribute_base = PETSC_FALSE;
  PetscBool      transfer_from_base[2] = {PETSC_TRUE,PETSC_FALSE};
  PetscBool      use_bcs = PETSC_TRUE;
  PetscBool      periodic = PETSC_FALSE;
  bc_func_ctx    bcCtx;
  DMLabel        adaptLabel;
  size_t         len;
  PetscErrorCode ierr;

  filename[0] = '\0';
  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = PetscOptionsBegin(comm, "", "DMForestTransferVec() Test Options", "DMFOREST");CHKERRQ(ierr);
  ierr = PetscOptionsRangeInt("-dim", "The dimension (2 or 3)", "ex2.c", dim, &dim, NULL,2,3);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-linear","Transfer a simple linear function", "ex2.c", linear, &linear, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-coords","Transfer a simple coordinate function", "ex2.c", coords, &coords, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-use_fv","Use a finite volume approximation", "ex2.c", useFV, &useFV, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-test_convert","Test conversion to DMPLEX",NULL,conv,&conv,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-filename", "Read the base mesh from file", "ex2.c", filename, filename, sizeof(filename), NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-distribute_base","Distribute base DM", "ex2.c", distribute_base, &distribute_base, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-transfer_from_base","Transfer a vector from base DM to DMForest", "ex2.c", transfer_from_base[0], &transfer_from_base[0], NULL);CHKERRQ(ierr);
  transfer_from_base[1] = transfer_from_base[0];
  ierr = PetscOptionsBool("-transfer_from_base_steps","Transfer a vector from base DM to the latest DMForest after the adaptivity steps", "ex2.c", transfer_from_base[1], &transfer_from_base[1], NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-use_bcs","Use dirichlet boundary conditions", "ex2.c", use_bcs, &use_bcs, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBoundedInt("-adapt_steps","Number of adaptivity steps", "ex2.c", adaptSteps, &adaptSteps, NULL,0);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-periodic","Use periodic box mesh", "ex2.c", periodic, &periodic, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  tol = PetscMax(1.e-10,tol); /* XXX fix for quadruple precision -> why do I need to do this? */

  if (periodic) periodicity[0] = periodicity[1] = periodicity[2] = DM_BOUNDARY_PERIODIC;

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

  /* the base mesh */
  ierr = PetscStrlen(filename, &len);CHKERRQ(ierr);
  if (!len) {
    ierr = DMPlexCreateBoxMesh(comm,dim,PETSC_FALSE,cells,NULL,NULL,periodicity,PETSC_TRUE,&base);CHKERRQ(ierr);
  } else {
    DM tdm = NULL;

    ierr = DMPlexCreateFromFile(comm,filename,PETSC_TRUE,&base);CHKERRQ(ierr);
    ierr = DMGetDimension(base,&dim);CHKERRQ(ierr);
    ierr = DMPlexSetCellRefinerType(base, DM_REFINER_TO_BOX);CHKERRQ(ierr);
    ierr = DMRefine(base, PETSC_COMM_WORLD, &tdm);CHKERRQ(ierr);
    if (tdm) {
      ierr = DMDestroy(&base);CHKERRQ(ierr);
      base = tdm;
    }
    use_bcs = PETSC_FALSE;
  }
  ierr = AddIdentityLabel(base);CHKERRQ(ierr);
  if (distribute_base) {
    DM               baseParallel;
    PetscPartitioner part;

    ierr = DMPlexGetPartitioner(base,&part);CHKERRQ(ierr);
    ierr = PetscPartitionerSetFromOptions(part);CHKERRQ(ierr);
    ierr = DMPlexDistribute(base,0,NULL,&baseParallel);CHKERRQ(ierr);
    if (baseParallel) {
      ierr = DMDestroy(&base);CHKERRQ(ierr);
      base = baseParallel;
    }
  }
  if (useFV) {
    PetscFV      fv;
    PetscLimiter limiter;
    DM           baseFV;

    ierr = DMPlexConstructGhostCells(base,NULL,NULL,&baseFV);CHKERRQ(ierr);
    ierr = DMViewFromOptions(baseFV, NULL, "-fv_dm_view");CHKERRQ(ierr);
    ierr = DMDestroy(&base);CHKERRQ(ierr);
    base = baseFV;
    ierr = PetscFVCreate(comm, &fv);CHKERRQ(ierr);
    ierr = PetscFVSetSpatialDimension(fv,dim);CHKERRQ(ierr);
    ierr = PetscFVSetType(fv,PETSCFVLEASTSQUARES);CHKERRQ(ierr);
    ierr = PetscFVSetNumComponents(fv,Nf);CHKERRQ(ierr);
    ierr = PetscLimiterCreate(comm,&limiter);CHKERRQ(ierr);
    ierr = PetscLimiterSetType(limiter,PETSCLIMITERNONE);CHKERRQ(ierr);
    ierr = PetscFVSetLimiter(fv,limiter);CHKERRQ(ierr);
    ierr = PetscLimiterDestroy(&limiter);CHKERRQ(ierr);
    ierr = PetscFVSetFromOptions(fv);CHKERRQ(ierr);
    ierr = DMSetField(base,0,NULL,(PetscObject)fv);CHKERRQ(ierr);
    ierr = PetscFVDestroy(&fv);CHKERRQ(ierr);
  } else {
    PetscFE fe;

    ierr = PetscFECreateDefault(comm,dim,Nf,PETSC_FALSE,NULL,PETSC_DEFAULT,&fe);CHKERRQ(ierr);
    ierr = DMSetField(base,0,NULL,(PetscObject)fe);CHKERRQ(ierr);
    ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
  }
  ierr = DMCreateDS(base);CHKERRQ(ierr);

  if (use_bcs) {
    PetscInt ids[] = {1, 2, 3, 4, 5, 6};

    ierr = DMAddBoundary(base,DM_BC_ESSENTIAL, "bc", "marker", 0, 0, NULL, useFV ? (void(*)(void)) bc_func_fv : (void(*)(void)) funcs[0], NULL, 2 * dim, ids, useFV ? (void *) &bcCtx : NULL);CHKERRQ(ierr);
  }
  ierr = DMViewFromOptions(base,NULL,"-dm_base_view");CHKERRQ(ierr);

  /* the pre adaptivity forest */
  ierr = DMCreate(comm,&preForest);CHKERRQ(ierr);
  ierr = DMSetType(preForest,(dim == 2) ? DMP4EST : DMP8EST);CHKERRQ(ierr);
  ierr = DMCopyDisc(base,preForest);CHKERRQ(ierr);
  ierr = DMForestSetBaseDM(preForest,base);CHKERRQ(ierr);
  ierr = DMForestSetMinimumRefinement(preForest,0);CHKERRQ(ierr);
  ierr = DMForestSetInitialRefinement(preForest,1);CHKERRQ(ierr);
  ierr = DMSetFromOptions(preForest);CHKERRQ(ierr);
  ierr = DMSetUp(preForest);CHKERRQ(ierr);
  ierr = DMViewFromOptions(preForest,NULL,"-dm_pre_view");CHKERRQ(ierr);

  /* the pre adaptivity field */
  ierr = DMCreateGlobalVector(preForest,&preVec);CHKERRQ(ierr);
  ierr = DMProjectFunction(preForest,0.,funcs,ctxs,INSERT_VALUES,preVec);CHKERRQ(ierr);
  ierr = VecViewFromOptions(preVec,NULL,"-vec_pre_view");CHKERRQ(ierr);

  /* communicate between base and pre adaptivity forest */
  if (transfer_from_base[0]) {
    Vec baseVec, baseVecMapped;

    ierr = DMGetGlobalVector(base,&baseVec);CHKERRQ(ierr);
    ierr = DMProjectFunction(base,0.,funcs,ctxs,INSERT_VALUES,baseVec);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)baseVec,"Function Base");CHKERRQ(ierr);
    ierr = VecViewFromOptions(baseVec,NULL,"-vec_base_view");CHKERRQ(ierr);

    ierr = DMGetGlobalVector(preForest,&baseVecMapped);CHKERRQ(ierr);
    ierr = DMForestTransferVecFromBase(preForest,baseVec,baseVecMapped);CHKERRQ(ierr);
    ierr = VecViewFromOptions(baseVecMapped,NULL,"-vec_map_base_view");CHKERRQ(ierr);

    /* compare */
    ierr = VecAXPY(baseVecMapped,-1.,preVec);CHKERRQ(ierr);
    ierr = VecViewFromOptions(baseVecMapped,NULL,"-vec_map_diff_view");CHKERRQ(ierr);
    ierr = VecNorm(baseVecMapped,NORM_2,&diff);CHKERRQ(ierr);

    /* output */
    if (diff < tol) {
      ierr = PetscPrintf(comm,"DMForestTransferVecFromBase() passes.\n");CHKERRQ(ierr);
    } else {
      ierr = PetscPrintf(comm,"DMForestTransferVecFromBase() fails with error %g and tolerance %g\n",(double)diff,(double)tol);CHKERRQ(ierr);
    }

    ierr = DMRestoreGlobalVector(base,&baseVec);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(preForest,&baseVecMapped);CHKERRQ(ierr);
  }

  for (step = 0; step < adaptSteps; ++step) {

    if (!transfer_from_base[1]) {
      ierr = PetscObjectGetReference((PetscObject)preForest,&preCount);CHKERRQ(ierr);
    }

    /* adapt */
    ierr = CreateAdaptivityLabel(preForest,&adaptLabel);CHKERRQ(ierr);
    ierr = DMForestTemplate(preForest,comm,&postForest);CHKERRQ(ierr);
    if (step) { ierr = DMForestSetAdaptivityLabel(postForest,adaptLabel);CHKERRQ(ierr); }
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
    ierr = VecAXPY(postVecExact,-1.,postVecTransfer);CHKERRQ(ierr);
    ierr = VecViewFromOptions(postVecExact,NULL,"-vec_diff_view");CHKERRQ(ierr);
    ierr = VecNorm(postVecExact,NORM_2,&diff);CHKERRQ(ierr);

    /* output */
    if (diff < tol) {
      ierr = PetscPrintf(comm,"DMForestTransferVec() passes.\n");CHKERRQ(ierr);
    } else {
      ierr = PetscPrintf(comm,"DMForestTransferVec() fails with error %g and tolerance %g\n",(double)diff,(double)tol);CHKERRQ(ierr);
      ierr = IdentifyBadPoints(postForest, postVecExact, tol);CHKERRQ(ierr);
    }
    ierr = VecDestroy(&postVecExact);CHKERRQ(ierr);

    /* disconnect preForest from postForest if we don't test the transfer throughout the entire refinement process */
    if (!transfer_from_base[1]) {
      ierr = DMForestSetAdaptivityForest(postForest,NULL);CHKERRQ(ierr);
      ierr = PetscObjectGetReference((PetscObject)preForest,&postCount);CHKERRQ(ierr);
      if (postCount != preCount) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Adaptation not memory neutral: reference count increase from %d to %d\n",preCount,postCount);
    }

    if (conv) {
      DM dmConv;

      ierr = DMConvert(postForest,DMPLEX,&dmConv);CHKERRQ(ierr);
      ierr = DMViewFromOptions(dmConv,NULL,"-dm_conv_view");CHKERRQ(ierr);
      ierr = DMPlexCheckCellShape(dmConv,PETSC_TRUE,PETSC_DETERMINE);CHKERRQ(ierr);
      ierr = DMDestroy(&dmConv);CHKERRQ(ierr);
    }

    ierr = VecDestroy(&preVec);CHKERRQ(ierr);
    ierr = DMDestroy(&preForest);CHKERRQ(ierr);

    preVec    = postVecTransfer;
    preForest = postForest;
  }

  if (transfer_from_base[1]) {
    Vec baseVec, baseVecMapped;

    /* communicate between base and last adapted forest */
    ierr = DMGetGlobalVector(base,&baseVec);CHKERRQ(ierr);
    ierr = DMProjectFunction(base,0.,funcs,ctxs,INSERT_VALUES,baseVec);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)baseVec,"Function Base");CHKERRQ(ierr);
    ierr = VecViewFromOptions(baseVec,NULL,"-vec_base_view");CHKERRQ(ierr);

    ierr = DMGetGlobalVector(preForest,&baseVecMapped);CHKERRQ(ierr);
    ierr = DMForestTransferVecFromBase(preForest,baseVec,baseVecMapped);CHKERRQ(ierr);
    ierr = VecViewFromOptions(baseVecMapped,NULL,"-vec_map_base_view");CHKERRQ(ierr);

    /* compare */
    ierr = VecAXPY(baseVecMapped,-1.,preVec);CHKERRQ(ierr);
    ierr = VecViewFromOptions(baseVecMapped,NULL,"-vec_map_diff_view");CHKERRQ(ierr);
    ierr = VecNorm(baseVecMapped,NORM_2,&diff);CHKERRQ(ierr);

    /* output */
    if (diff < tol) {
      ierr = PetscPrintf(comm,"DMForestTransferVecFromBase() passes.\n");CHKERRQ(ierr);
    } else {
      ierr = PetscPrintf(comm,"DMForestTransferVecFromBase() fails with error %g and tolerance %g\n",(double)diff,(double)tol);CHKERRQ(ierr);
    }

    ierr = DMRestoreGlobalVector(base,&baseVec);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(preForest,&baseVecMapped);CHKERRQ(ierr);
  }

  /* cleanup */
  ierr = VecDestroy(&preVec);CHKERRQ(ierr);
  ierr = DMDestroy(&preForest);CHKERRQ(ierr);
  ierr = DMDestroy(&base);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

     test:
       output_file: output/ex2_2d.out
       suffix: p4est_2d
       args: -petscspace_type tensor -petscspace_degree 2 -dim 2
       nsize: 3
       requires: p4est !single

     test:
       output_file: output/ex2_2d.out
       suffix: p4est_2d_deg4
       args: -petscspace_type tensor -petscspace_degree 4 -dim 2
       requires: p4est !single

     test:
       output_file: output/ex2_2d.out
       suffix: p4est_2d_deg8
       args: -petscspace_type tensor -petscspace_degree 8 -dim 2
       requires: p4est !single

     test:
       output_file: output/ex2_steps2.out
       suffix: p4est_2d_deg2_steps2
       args: -petscspace_type tensor -petscspace_degree 2 -dim 2 -coords -adapt_steps 2
       nsize: 3
       requires: p4est !single

     test:
       output_file: output/ex2_steps3.out
       suffix: p4est_2d_deg3_steps3
       args: -petscspace_type tensor -petscspace_degree 3 -dim 2 -coords -adapt_steps 3 -petscdualspace_lagrange_node_type equispaced -petscdualspace_lagrange_node_endpoints 1
       nsize: 3
       requires: p4est !single

     test:
       output_file: output/ex2_steps3.out
       suffix: p4est_2d_deg3_steps3_L2_periodic
       args: -petscspace_type tensor -petscspace_degree 3 -petscdualspace_lagrange_continuity 0 -dim 2 -coords -adapt_steps 3 -periodic -use_bcs 0 -petscdualspace_lagrange_node_type equispaced
       nsize: 3
       requires: p4est !single

     test:
       output_file: output/ex2_steps3.out
       suffix: p4est_3d_deg2_steps3_L2_periodic
       args: -petscspace_type tensor -petscspace_degree 2 -petscdualspace_lagrange_continuity 0 -dim 3 -coords -adapt_steps 3 -periodic -use_bcs 0
       nsize: 3
       requires: p4est !single

     test:
       output_file: output/ex2_steps2.out
       suffix: p4est_3d_deg2_steps2
       args: -petscspace_type tensor -petscspace_degree 2 -dim 3 -coords -adapt_steps 2
       nsize: 3
       requires: p4est !single

     test:
       output_file: output/ex2_steps3.out
       suffix: p4est_3d_deg3_steps3
       args: -petscspace_type tensor -petscspace_degree 3 -dim 3 -coords -adapt_steps 3 -petscdualspace_lagrange_node_type equispaced -petscdualspace_lagrange_node_endpoints 1
       nsize: 3
       requires: p4est !single

     test:
       output_file: output/ex2_2d_fv.out
       suffix: p4est_2d_fv
       args: -transfer_from_base 0 -use_fv -linear -dim 2 -dm_forest_partition_overlap 1
       nsize: 3
       requires: p4est !single

     test:
       TODO: broken (codimension adjacency)
       output_file: output/ex2_2d_fv.out
       suffix: p4est_2d_fv_adjcodim
       args: -transfer_from_base 0 -use_fv -linear -dim 2 -dm_forest_partition_overlap 1 -dm_forest_adjacency_codimension 1
       nsize: 2
       requires: p4est !single

     test:
       TODO: broken (dimension adjacency)
       output_file: output/ex2_2d_fv.out
       suffix: p4est_2d_fv_adjdim
       args: -transfer_from_base 0 -use_fv -linear -dim 2 -dm_forest_partition_overlap 1 -dm_forest_adjacency_dimension 1
       nsize: 2
       requires: p4est !single

     test:
       output_file: output/ex2_2d_fv.out
       suffix: p4est_2d_fv_zerocells
       args: -transfer_from_base 0 -use_fv -linear -dim 2 -dm_forest_partition_overlap 1
       nsize: 10
       requires: p4est !single

     test:
       output_file: output/ex2_3d.out
       suffix: p4est_3d
       args: -petscspace_type tensor -petscspace_degree 1 -dim 3
       nsize: 3
       requires: p4est !single

     test:
       output_file: output/ex2_3d.out
       suffix: p4est_3d_deg3
       args: -petscspace_type tensor -petscspace_degree 3 -dim 3
       nsize: 3
       requires: p4est !single

     test:
       output_file: output/ex2_2d.out
       suffix: p4est_2d_deg2_coords
       args: -petscspace_type tensor -petscspace_degree 2 -dim 2 -coords
       nsize: 3
       requires: p4est !single

     test:
       output_file: output/ex2_3d.out
       suffix: p4est_3d_deg2_coords
       args: -petscspace_type tensor -petscspace_degree 2 -dim 3 -coords
       nsize: 3
       requires: p4est !single

     test:
       output_file: output/ex2_3d_fv.out
       suffix: p4est_3d_fv
       args: -transfer_from_base 0 -use_fv -linear -dim 3 -dm_forest_partition_overlap 1
       nsize: 3
       requires: p4est !single

     test:
       suffix: p4est_3d_nans
       args: -dim 3 -dm_forest_partition_overlap 1 -test_convert -petscspace_type tensor -petscspace_degree 1
       nsize: 2
       requires: p4est !single

     test:
       TODO: not broken, but the 3D case below is broken, so I do not trust this one
       output_file: output/ex2_steps2.out
       suffix: p4est_2d_tfb_distributed_nc
       args: -petscspace_type tensor -petscspace_degree 3 -dim 2 -dm_forest_maximum_refinement 2 -dm_p4est_refine_pattern hash -use_bcs 0 -coords -adapt_steps 2 -distribute_base -petscpartitioner_type shell -petscpartitioner_shell_random
       nsize: 3
       requires: p4est !single

     test:
       TODO: broken
       output_file: output/ex2_steps2.out
       suffix: p4est_3d_tfb_distributed_nc
       args: -petscspace_type tensor -petscspace_degree 2 -dim 3 -dm_forest_maximum_refinement 2 -dm_p4est_refine_pattern hash -use_bcs 0 -coords -adapt_steps 2 -distribute_base -petscpartitioner_type shell -petscpartitioner_shell_random
       nsize: 3
       requires: p4est !single

     test:
       TODO: broken
       output_file: output/ex2_3d.out
       suffix: p4est_3d_transfer_fails
       args: -petscspace_type tensor -petscspace_degree 1 -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/doublet-tet.msh -adapt_steps 1 -dm_forest_initial_refinement 1
       requires: p4est !single

     test:
       TODO: broken
       output_file: output/ex2_steps2_notfb.out
       suffix: p4est_3d_transfer_fails_2
       args: -petscspace_type tensor -petscspace_degree 1 -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/doublet-tet.msh -adapt_steps 2 -dm_forest_initial_refinement 0 -transfer_from_base 0
       requires: p4est !single

     test:
       output_file: output/ex2_steps2.out
       suffix: p4est_3d_multi_transfer_s2t
       args: -petscspace_type tensor -petscspace_degree 3 -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/doublet-tet.msh -adapt_steps 2 -dm_forest_initial_refinement 1 -petscdualspace_lagrange_continuity 0 -use_bcs 0
       requires: p4est !single

     test:
       output_file: output/ex2_steps2.out
       suffix: p4est_3d_coords_transfer_s2t
       args: -petscspace_type tensor -petscspace_degree 3 -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/doublet-tet.msh -adapt_steps 2 -dm_forest_initial_refinement 1 -petscdualspace_lagrange_continuity 0 -coords -use_bcs 0
       requires: p4est !single

TEST*/
