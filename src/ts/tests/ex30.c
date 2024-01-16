static char help[] = "Grid based Landau collision operator with PIC interface with OpenMP setup. (one species per grid)\n";

/*
   Support 2.5V with axisymmetric coordinates
     - r,z coordinates
     - Domain and species data input by Landau operator
     - "radius" for each grid, normalized with electron thermal velocity
     - Domain: (0,radius) x (-radius,radius), thus first coordinate x[0] is perpendicular velocity and 2pi*x[0] term is added for axisymmetric
   Supports full 3V

 */

#include <petscdmplex.h>
#include <petscds.h>
#include <petscdmswarm.h>
#include <petscksp.h>
#include <petsc/private/petscimpl.h>
#if defined(PETSC_HAVE_OPENMP) && defined(PETSC_HAVE_THREADSAFETY)
  #include <omp.h>
#endif
#include <petsclandau.h>
#include <petscdmcomposite.h>

typedef struct {
  Mat MpTrans;
  Mat Mp;
  Vec ff;
  Vec uu;
} MatShellCtx;

typedef struct {
  PetscInt   v_target;
  DM        *globSwarmArray;
  LandauCtx *ctx;
  PetscInt  *nTargetP;
  PetscReal  N_inv;
  DM        *grid_dm;
  Mat       *g_Mass;
  Mat       *globMpArray;
  Vec       *globXArray;
  PetscBool  print;
} PrintCtx;

PetscErrorCode MatMultMtM_SeqAIJ(Mat MtM, Vec xx, Vec yy)
{
  MatShellCtx *matshellctx;

  PetscFunctionBeginUser;
  PetscCall(MatShellGetContext(MtM, &matshellctx));
  PetscCheck(matshellctx, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "No context");
  PetscCall(MatMult(matshellctx->Mp, xx, matshellctx->ff));
  PetscCall(MatMult(matshellctx->MpTrans, matshellctx->ff, yy));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatMultAddMtM_SeqAIJ(Mat MtM, Vec xx, Vec yy, Vec zz)
{
  MatShellCtx *matshellctx;

  PetscFunctionBeginUser;
  PetscCall(MatShellGetContext(MtM, &matshellctx));
  PetscCheck(matshellctx, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "No context");
  PetscCall(MatMult(matshellctx->Mp, xx, matshellctx->ff));
  PetscCall(MatMultAdd(matshellctx->MpTrans, matshellctx->ff, yy, zz));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode createSwarm(const DM dm, PetscInt dim, DM *sw)
{
  PetscInt Nc = 1;

  PetscFunctionBeginUser;
  PetscCall(DMCreate(PETSC_COMM_SELF, sw));
  PetscCall(DMSetType(*sw, DMSWARM));
  PetscCall(DMSetDimension(*sw, dim));
  PetscCall(DMSwarmSetType(*sw, DMSWARM_PIC));
  PetscCall(DMSwarmSetCellDM(*sw, dm));
  PetscCall(DMSwarmRegisterPetscDatatypeField(*sw, "w_q", Nc, PETSC_REAL));
  PetscCall(DMSwarmFinalizeFieldRegister(*sw));
  PetscCall(DMSetFromOptions(*sw));
  PetscCall(PetscObjectSetName((PetscObject)*sw, "Particle Grid"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode makeSwarm(DM sw, const PetscInt dim, const PetscInt Np, const PetscReal xx[], const PetscReal yy[], const PetscReal zz[])
{
  PetscReal    *coords;
  PetscDataType dtype;
  PetscInt      bs, p, zero = 0;
  PetscFunctionBeginUser;

  PetscCall(DMSwarmSetLocalSizes(sw, Np, zero));
  PetscCall(DMSwarmGetField(sw, "DMSwarmPIC_coor", &bs, &dtype, (void **)&coords));
  for (p = 0; p < Np; p++) {
    coords[p * dim + 0] = xx[p];
    coords[p * dim + 1] = yy[p];
    if (dim == 3) coords[p * dim + 2] = zz[p];
  }
  PetscCall(DMSwarmRestoreField(sw, "DMSwarmPIC_coor", &bs, &dtype, (void **)&coords));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode createMp(const DM dm, DM sw, Mat *Mp_out)
{
  PetscBool removePoints = PETSC_TRUE;
  Mat       M_p;
  PetscFunctionBeginUser;
  // migrate after coords are set
  PetscCall(DMSwarmMigrate(sw, removePoints));
  PetscCall(PetscObjectSetName((PetscObject)sw, "Particle Grid"));
  /* This gives M f = \int_\Omega \phi f, which looks like a rhs for a PDE */
  PetscCall(DMCreateMassMatrix(sw, dm, &M_p));
  PetscCall(DMViewFromOptions(sw, NULL, "-ex30_sw_view"));
  // output
  *Mp_out = M_p;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode particlesToGrid(const DM dm, DM sw, const PetscInt Np, const PetscInt a_tid, const PetscInt dim, const PetscReal a_wp[], Vec rho, Mat M_p)
{
  PetscReal    *wq;
  PetscDataType dtype;
  Vec           ff;
  PetscInt      bs, p;

  PetscFunctionBeginUser;
  PetscCall(DMSwarmGetField(sw, "w_q", &bs, &dtype, (void **)&wq));
  for (p = 0; p < Np; p++) wq[p] = a_wp[p];
  PetscCall(DMSwarmRestoreField(sw, "w_q", &bs, &dtype, (void **)&wq));
  PetscCall(PetscObjectSetName((PetscObject)rho, "rho"));
  PetscCall(DMSwarmCreateGlobalVectorFromField(sw, "w_q", &ff));
  PetscCall(PetscObjectSetName((PetscObject)ff, "weights"));
  PetscCall(MatMultTranspose(M_p, ff, rho));
  PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, "w_q", &ff));
  PetscFunctionReturn(PETSC_SUCCESS);
}

//
// add grid to arg 'sw.w_q'
//
PetscErrorCode gridToParticles(const DM dm, DM sw, const Vec rhs, Vec work, Mat M_p, Mat Mass)
{
  PetscBool    is_lsqr;
  KSP          ksp;
  Mat          PM_p = NULL, MtM, D;
  Vec          ff;
  PetscInt     N, M, nzl;
  MatShellCtx *matshellctx;

  PetscFunctionBeginUser;
  PetscCall(MatMult(Mass, rhs, work));
  // pseudo-inverse
  PetscCall(KSPCreate(PETSC_COMM_SELF, &ksp));
  PetscCall(KSPSetOptionsPrefix(ksp, "ftop_"));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(PetscObjectTypeCompare((PetscObject)ksp, KSPLSQR, &is_lsqr));
  if (!is_lsqr) {
    PetscCall(MatGetLocalSize(M_p, &M, &N));
    if (N > M) {
      PC pc;
      PetscCall(PetscInfo(ksp, " M (%" PetscInt_FMT ") < M (%" PetscInt_FMT ") -- skip revert to lsqr\n", M, N));
      is_lsqr = PETSC_TRUE;
      PetscCall(KSPSetType(ksp, KSPLSQR));
      PetscCall(KSPGetPC(ksp, &pc));
      PetscCall(PCSetType(pc, PCNONE)); // could put in better solver -ftop_pc_type bjacobi -ftop_sub_pc_type lu -ftop_sub_pc_factor_shift_type nonzero
    } else {
      PetscCall(PetscNew(&matshellctx));
      PetscCall(MatCreateShell(PetscObjectComm((PetscObject)dm), N, N, PETSC_DECIDE, PETSC_DECIDE, matshellctx, &MtM));
      PetscCall(MatTranspose(M_p, MAT_INITIAL_MATRIX, &matshellctx->MpTrans));
      matshellctx->Mp = M_p;
      PetscCall(MatShellSetOperation(MtM, MATOP_MULT, (void (*)(void))MatMultMtM_SeqAIJ));
      PetscCall(MatShellSetOperation(MtM, MATOP_MULT_ADD, (void (*)(void))MatMultAddMtM_SeqAIJ));
      PetscCall(MatCreateVecs(M_p, &matshellctx->uu, &matshellctx->ff));
      PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, N, N, 1, NULL, &D));
      PetscCall(MatViewFromOptions(matshellctx->MpTrans, NULL, "-ftop2_Mp_mat_view"));
      for (int i = 0; i < N; i++) {
        const PetscScalar *vals;
        const PetscInt    *cols;
        PetscScalar        dot = 0;
        PetscCall(MatGetRow(matshellctx->MpTrans, i, &nzl, &cols, &vals));
        for (int ii = 0; ii < nzl; ii++) dot += PetscSqr(vals[ii]);
        PetscCheck(dot != 0.0, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Row %d is empty", i);
        PetscCall(MatSetValue(D, i, i, dot, INSERT_VALUES));
      }
      PetscCall(MatAssemblyBegin(D, MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(D, MAT_FINAL_ASSEMBLY));
      PetscCall(PetscInfo(M_p, "createMtMKSP Have %" PetscInt_FMT " eqs, nzl = %" PetscInt_FMT "\n", N, nzl));
      PetscCall(KSPSetOperators(ksp, MtM, D));
      PetscCall(MatViewFromOptions(D, NULL, "-ftop2_D_mat_view"));
      PetscCall(MatViewFromOptions(M_p, NULL, "-ftop2_Mp_mat_view"));
      PetscCall(MatViewFromOptions(matshellctx->MpTrans, NULL, "-ftop2_MpTranspose_mat_view"));
      PetscCall(MatViewFromOptions(MtM, NULL, "-ftop2_MtM_mat_view"));
    }
  }
  if (is_lsqr) {
    PC        pc;
    PetscBool is_bjac;
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PetscObjectTypeCompare((PetscObject)pc, PCBJACOBI, &is_bjac));
    if (is_bjac) {
      PetscCall(DMSwarmCreateMassMatrixSquare(sw, dm, &PM_p));
      PetscCall(KSPSetOperators(ksp, M_p, PM_p));
    } else {
      PetscCall(KSPSetOperators(ksp, M_p, M_p));
    }
  }
  PetscCall(DMSwarmCreateGlobalVectorFromField(sw, "w_q", &ff)); // this grabs access
  if (!is_lsqr) {
    PetscCall(KSPSolve(ksp, work, matshellctx->uu));
    PetscCall(MatMult(M_p, matshellctx->uu, ff));
    PetscCall(MatDestroy(&matshellctx->MpTrans));
    PetscCall(VecDestroy(&matshellctx->ff));
    PetscCall(VecDestroy(&matshellctx->uu));
    PetscCall(MatDestroy(&D));
    PetscCall(MatDestroy(&MtM));
    PetscCall(PetscFree(matshellctx));
  } else {
    PetscCall(KSPSolveTranspose(ksp, work, ff));
  }
  PetscCall(KSPDestroy(&ksp));
  /* Visualize particle field */
  PetscCall(VecViewFromOptions(ff, NULL, "-weights_view"));
  PetscCall(MatDestroy(&PM_p));
  PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, "w_q", &ff));

  PetscFunctionReturn(PETSC_SUCCESS);
}

#define EX30_MAX_NUM_THRDS 12
#define EX30_MAX_BATCH_SZ  1024
//
// add grid to arg 'globSwarmArray[].w_q'
//
PetscErrorCode gridToParticles_private(DM grid_dm[], DM globSwarmArray[], const PetscInt dim, const PetscInt v_target, const PetscInt numthreads, const PetscInt num_vertices, const PetscInt global_vertex_id, Mat globMpArray[], Mat g_Mass[], Vec t_fhat[][EX30_MAX_NUM_THRDS], PetscReal moments[], Vec globXArray[], LandauCtx *ctx)
{
  PetscErrorCode ierr = (PetscErrorCode)0; // used for inside thread loops

  PetscFunctionBeginUser;
  // map back to particles
  for (PetscInt v_id_0 = 0; v_id_0 < ctx->batch_sz; v_id_0 += numthreads) {
    PetscCall(PetscInfo(grid_dm[0], "g2p: global batch %" PetscInt_FMT " of %" PetscInt_FMT ", Landau batch %" PetscInt_FMT " of %" PetscInt_FMT ": map back to particles\n", global_vertex_id + 1, num_vertices, v_id_0 + 1, ctx->batch_sz));
    //PetscPragmaOMP(parallel for)
    for (int tid = 0; tid < numthreads; tid++) {
      const PetscInt v_id = v_id_0 + tid, glb_v_id = global_vertex_id + v_id;
      if (glb_v_id < num_vertices) {
        for (PetscInt grid = 0; grid < ctx->num_grids; grid++) { // add same particels for all grids
          PetscErrorCode ierr_t;
          ierr_t = PetscInfo(grid_dm[0], "gridToParticles: global batch %" PetscInt_FMT ", local batch b=%" PetscInt_FMT ", grid g=%" PetscInt_FMT ", index(b,g) %" PetscInt_FMT "\n", global_vertex_id, v_id, grid, LAND_PACK_IDX(v_id, grid));
          ierr_t = gridToParticles(grid_dm[grid], globSwarmArray[LAND_PACK_IDX(v_id, grid)], globXArray[LAND_PACK_IDX(v_id, grid)], t_fhat[grid][tid], globMpArray[LAND_PACK_IDX(v_id, grid)], g_Mass[grid]);
          if (ierr_t) ierr = ierr_t;
        }
      }
    }
    PetscCheck(!ierr, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Error in OMP loop. ierr = %d", (int)ierr);
    /* Get moments */
    PetscCall(PetscInfo(grid_dm[0], "Cleanup batches %" PetscInt_FMT " to %" PetscInt_FMT "\n", v_id_0, v_id_0 + numthreads));
    for (int tid = 0; tid < numthreads; tid++) {
      const PetscInt v_id = v_id_0 + tid, glb_v_id = global_vertex_id + v_id;
      if (glb_v_id == v_target) {
        for (PetscInt grid = 0; grid < ctx->num_grids; grid++) {
          PetscDataType dtype;
          PetscReal    *wp, *coords;
          DM            sw = globSwarmArray[LAND_PACK_IDX(v_id, grid)];
          PetscInt      npoints, bs = 1;
          PetscCall(DMSwarmGetField(sw, "w_q", &bs, &dtype, (void **)&wp)); // take data out here
          PetscCall(DMSwarmGetField(sw, "DMSwarmPIC_coor", &bs, &dtype, (void **)&coords));
          PetscCall(DMSwarmGetLocalSize(sw, &npoints));
          for (int p = 0; p < npoints; p++) {
            PetscReal v2 = 0, fact = (dim == 2) ? 2.0 * PETSC_PI * coords[p * dim + 0] : 1, w = fact * wp[p] * ctx->n_0 * ctx->masses[ctx->species_offset[grid]];
            for (int i = 0; i < dim; ++i) v2 += PetscSqr(coords[p * dim + i]);
            moments[0] += w;
            moments[1] += w * ctx->v_0 * coords[p * dim + 1]; // z-momentum
            moments[2] += w * ctx->v_0 * ctx->v_0 * v2;
          }
          PetscCall(DMSwarmRestoreField(sw, "w_q", &bs, &dtype, (void **)&wp));
          PetscCall(DMSwarmRestoreField(sw, "DMSwarmPIC_coor", &bs, &dtype, (void **)&coords));
        }
        const PetscReal N_inv = 1 / moments[0];
        for (PetscInt grid = 0; grid < ctx->num_grids; grid++) {
          PetscDataType dtype;
          PetscReal    *wp, *coords;
          DM            sw = globSwarmArray[LAND_PACK_IDX(v_id, grid)];
          PetscInt      npoints, bs = 1;
          PetscCall(DMSwarmGetField(sw, "w_q", &bs, &dtype, (void **)&wp)); // take data out here
          PetscCall(DMSwarmGetField(sw, "DMSwarmPIC_coor", &bs, &dtype, (void **)&coords));
          PetscCall(DMSwarmGetLocalSize(sw, &npoints));
          for (int p = 0; p < npoints; p++) {
            const PetscReal fact = dim == 2 ? 2.0 * PETSC_PI * coords[p * dim + 0] : 1, w = fact * wp[p] * ctx->n_0 * ctx->masses[ctx->species_offset[grid]], ww = w * N_inv;
            if (ww > PETSC_REAL_MIN) {
              moments[3] -= ww * PetscLogReal(ww);
              PetscCheck(ww < 1 - PETSC_MACHINE_EPSILON, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "ww (%g) > 1", (double)ww);
            }
          }
          PetscCall(DMSwarmRestoreField(sw, "w_q", &bs, &dtype, (void **)&wp));
          PetscCall(DMSwarmRestoreField(sw, "DMSwarmPIC_coor", &bs, &dtype, (void **)&coords));
        }
      }
    } // thread batch
  }   // batch
  PetscFunctionReturn(PETSC_SUCCESS);
}

static void maxwellian(PetscInt dim, const PetscReal x[], PetscReal kt_m, PetscReal n, PetscReal shift, PetscScalar *u)
{
  PetscInt  i;
  PetscReal v2 = 0, theta = 2.0 * kt_m; /* theta = 2kT/mc^2 */

  /* compute the exponents, v^2 */
  for (i = 0; i < dim; ++i) v2 += x[i] * x[i];
  /* evaluate the Maxwellian */
  u[0] = n * PetscPowReal(PETSC_PI * theta, -1.5) * (PetscExpReal(-v2 / theta));
  if (shift != 0.) {
    v2 = 0;
    for (i = 0; i < dim - 1; ++i) v2 += x[i] * x[i];
    v2 += (x[dim - 1] - shift) * (x[dim - 1] - shift);
    /* evaluate the shifted Maxwellian */
    u[0] += n * PetscPowReal(PETSC_PI * theta, -1.5) * (PetscExpReal(-v2 / theta));
  }
}

static PetscErrorCode PostStep(TS ts)
{
  PetscInt   n, dim, nDMs;
  PetscReal  t;
  LandauCtx *ctx;
  Vec        X;
  PrintCtx  *printCtx;
  PetscReal  moments[4];
  DM         pack;

  PetscFunctionBeginUser;
  PetscCall(TSGetApplicationContext(ts, &printCtx));
  if (!printCtx->print) PetscFunctionReturn(PETSC_SUCCESS);

  for (int i = 0; i < 4; i++) moments[i] = 0;
  ctx = printCtx->ctx;
  PetscCall(TSGetDM(ts, &pack));
  PetscCall(DMGetDimension(pack, &dim));
  PetscCall(DMCompositeGetNumberDM(pack, &nDMs)); // number of vertices * number of grids
  PetscCall(TSGetSolution(ts, &X));
  const PetscInt v_id = printCtx->v_target % ctx->batch_sz;
  PetscCall(TSGetSolution(ts, &X));
  PetscCall(TSGetStepNumber(ts, &n));
  PetscCall(TSGetTime(ts, &t));
  PetscCall(DMCompositeGetAccessArray(pack, X, nDMs, NULL, printCtx->globXArray));
  for (PetscInt grid = 0; grid < ctx->num_grids; grid++) {
    PetscDataType dtype;
    PetscReal    *wp, *coords;
    DM            sw = printCtx->globSwarmArray[LAND_PACK_IDX(v_id, grid)];
    Vec           work, subX = printCtx->globXArray[LAND_PACK_IDX(v_id, grid)];
    PetscInt      bs, NN     = printCtx->nTargetP[grid];
    // C-G moments
    PetscCall(VecDuplicate(subX, &work));
    PetscCall(gridToParticles(printCtx->grid_dm[grid], sw, subX, work, printCtx->globMpArray[LAND_PACK_IDX(v_id, grid)], printCtx->g_Mass[grid]));
    PetscCall(VecDestroy(&work));
    // moments
    PetscCall(DMSwarmGetField(sw, "DMSwarmPIC_coor", &bs, &dtype, (void **)&coords));
    PetscCall(DMSwarmGetField(sw, "w_q", &bs, &dtype, (void **)&wp)); // could get NN from sw - todo
    for (int pp = 0; pp < NN; pp++) {
      PetscReal v2 = 0, fact = (dim == 2) ? 2.0 * PETSC_PI * coords[pp * dim + 0] : 1, w = fact * wp[pp] * ctx->n_0 * ctx->masses[ctx->species_offset[grid]], ww = w * printCtx->N_inv;
      for (int i = 0; i < dim; ++i) v2 += PetscSqr(coords[pp * dim + i]);
      moments[0] += w;
      moments[1] += w * ctx->v_0 * coords[pp * dim + 1]; // z-momentum
      moments[2] += w * ctx->v_0 * ctx->v_0 * v2;
      if (ww > PETSC_REAL_MIN) {
        moments[3] -= ww * PetscLogReal(ww);
        PetscCheck(ww < 1 - PETSC_MACHINE_EPSILON, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "ww (%g) > 1", (double)ww);
      }
    }
    PetscCall(DMSwarmRestoreField(sw, "DMSwarmPIC_coor", &bs, &dtype, (void **)&coords));
    PetscCall(DMSwarmRestoreField(sw, "w_q", &bs, &dtype, (void **)&wp));
  }
  PetscCall(DMCompositeRestoreAccessArray(pack, X, nDMs, NULL, printCtx->globXArray));
  PetscCall(PetscInfo(X, "%4d) time %e, Landau moments: %18.12e %19.12e %18.12e %e\n", (int)n, (double)t, (double)moments[0], (double)moments[1], (double)moments[2], (double)moments[3]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode go(TS ts, Vec X, const PetscInt num_vertices, const PetscInt a_Np, const PetscInt dim, const PetscInt v_target, const PetscInt g_target, PetscReal shift)
{
  DM             pack, *globSwarmArray, grid_dm[LANDAU_MAX_GRIDS];
  Mat           *globMpArray, g_Mass[LANDAU_MAX_GRIDS];
  KSP            t_ksp[LANDAU_MAX_GRIDS][EX30_MAX_NUM_THRDS];
  Vec            t_fhat[LANDAU_MAX_GRIDS][EX30_MAX_NUM_THRDS];
  PetscInt       nDMs, nTargetP[LANDAU_MAX_GRIDS];
  PetscErrorCode ierr = (PetscErrorCode)0; // used for inside thread loops
#if defined(PETSC_HAVE_OPENMP) && defined(PETSC_HAVE_THREADSAFETY)
  PetscInt numthreads = PetscNumOMPThreads;
#else
  PetscInt numthreads = 1;
#endif
  LandauCtx *ctx;
  Vec       *globXArray;
  PetscReal  moments_0[4], moments_1a[4], moments_1b[4], dt_init;
  PrintCtx  *printCtx;

  PetscFunctionBeginUser;
  PetscCheck(numthreads <= EX30_MAX_NUM_THRDS, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Too many threads %" PetscInt_FMT " > %d", numthreads, EX30_MAX_NUM_THRDS);
  PetscCheck(numthreads > 0, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Number threads %" PetscInt_FMT " > %d", numthreads, EX30_MAX_NUM_THRDS);
  PetscCall(TSGetDM(ts, &pack));
  PetscCall(DMGetApplicationContext(pack, &ctx));
  PetscCheck(ctx->batch_sz % numthreads == 0, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "batch size (-dm_landau_batch_size) %" PetscInt_FMT "  mod #threads %" PetscInt_FMT " must equal zero", ctx->batch_sz, numthreads);
  PetscCall(DMCompositeGetNumberDM(pack, &nDMs)); // number of vertices * number of grids
  PetscCall(PetscInfo(pack, "Have %" PetscInt_FMT " total grids, with %" PetscInt_FMT " Landau local batched and %" PetscInt_FMT " global items (vertices)\n", ctx->num_grids, ctx->batch_sz, num_vertices));
  PetscCall(PetscMalloc(sizeof(*globXArray) * nDMs, &globXArray));
  PetscCall(PetscMalloc(sizeof(*globMpArray) * nDMs, &globMpArray));
  PetscCall(PetscMalloc(sizeof(*globSwarmArray) * nDMs, &globSwarmArray));
  // print ctx
  PetscCall(PetscNew(&printCtx));
  PetscCall(TSSetApplicationContext(ts, printCtx));
  printCtx->v_target       = v_target;
  printCtx->ctx            = ctx;
  printCtx->nTargetP       = nTargetP;
  printCtx->globSwarmArray = globSwarmArray;
  printCtx->grid_dm        = grid_dm;
  printCtx->globMpArray    = globMpArray;
  printCtx->g_Mass         = g_Mass;
  printCtx->globXArray     = globXArray;
  // view
  PetscCall(DMViewFromOptions(ctx->plex[g_target], NULL, "-ex30_dm_view"));
  // create mesh mass matrices
  PetscCall(VecZeroEntries(X));
  PetscCall(DMCompositeGetAccessArray(pack, X, nDMs, NULL, globXArray)); // just to duplicate
  for (PetscInt grid = 0; grid < ctx->num_grids; grid++) {               // add same particels for all grids
    Vec          subX = globXArray[LAND_PACK_IDX(0, grid)];
    DM           dm   = ctx->plex[grid];
    PetscSection s;
    grid_dm[grid] = dm;
    PetscCall(DMCreateMassMatrix(dm, dm, &g_Mass[grid]));
    //
    PetscCall(DMGetLocalSection(dm, &s));
    PetscCall(DMPlexCreateClosureIndex(dm, s));
    for (int tid = 0; tid < numthreads; tid++) {
      PetscCall(VecDuplicate(subX, &t_fhat[grid][tid]));
      PetscCall(KSPCreate(PETSC_COMM_SELF, &t_ksp[grid][tid]));
      PetscCall(KSPSetOptionsPrefix(t_ksp[grid][tid], "ptof_"));
      PetscCall(KSPSetOperators(t_ksp[grid][tid], g_Mass[grid], g_Mass[grid]));
      PetscCall(KSPSetFromOptions(t_ksp[grid][tid]));
    }
  }
  PetscCall(DMCompositeRestoreAccessArray(pack, X, nDMs, NULL, globXArray));
  for (int i = 0; i < 4; i++) moments_0[i] = moments_1a[i] = moments_1b[i] = 0;
  PetscCall(TSGetTimeStep(ts, &dt_init)); // we could have an adaptive time stepper
  // loop over all vertices in chucks that are batched for TSSolve
  for (PetscInt global_vertex_id_0 = 0; global_vertex_id_0 < num_vertices; global_vertex_id_0 += ctx->batch_sz, shift /= 2) { // outer vertex loop
    PetscCall(TSSetTime(ts, 0));
    PetscCall(TSSetStepNumber(ts, 0));
    PetscCall(TSSetTimeStep(ts, dt_init));
    PetscCall(DMCompositeGetAccessArray(pack, X, nDMs, NULL, globXArray));
    if (v_target >= global_vertex_id_0 && v_target < global_vertex_id_0 + ctx->batch_sz) {
      PetscCall(PetscObjectSetName((PetscObject)globXArray[LAND_PACK_IDX(v_target % ctx->batch_sz, g_target)], "rho"));
      printCtx->print = PETSC_TRUE;
    } else printCtx->print = PETSC_FALSE;
    // create fake particles in batches with threads
    for (PetscInt v_id_0 = 0; v_id_0 < ctx->batch_sz; v_id_0 += numthreads) {
      PetscReal *xx_t[LANDAU_MAX_GRIDS][EX30_MAX_NUM_THRDS], *yy_t[LANDAU_MAX_GRIDS][EX30_MAX_NUM_THRDS], *zz_t[LANDAU_MAX_GRIDS][EX30_MAX_NUM_THRDS], *wp_t[LANDAU_MAX_GRIDS][EX30_MAX_NUM_THRDS];
      PetscInt   Np_t[LANDAU_MAX_GRIDS][EX30_MAX_NUM_THRDS];
      // make particles
      for (int tid = 0; tid < numthreads; tid++) {
        const PetscInt v_id = v_id_0 + tid, glb_v_id = global_vertex_id_0 + v_id;
        if (glb_v_id < num_vertices) {                                                                                                                                            // the ragged edge (in last batch)
          PetscInt Npp0 = a_Np + (glb_v_id % (a_Np / 10 + 1)), NN;                                                                                                                // number of particels in each dimension with add some load imbalance
          for (PetscInt grid = 0; grid < ctx->num_grids; grid++) {                                                                                                                // add same particels for all grids
            const PetscReal kT_m  = ctx->k * ctx->thermal_temps[ctx->species_offset[grid]] / ctx->masses[ctx->species_offset[grid]] / (ctx->v_0 * ctx->v_0);                      /* theta = 2kT/mc^2 per species */
            PetscReal       lo[3] = {-ctx->radius[grid], -ctx->radius[grid], -ctx->radius[grid]}, hi[3] = {ctx->radius[grid], ctx->radius[grid], ctx->radius[grid]}, hp[3], vole; // would be nice to get box from DM
            PetscInt        Npi = Npp0, Npj = 2 * Npp0, Npk = 1;
            if (dim == 2) lo[0] = 0; // Landau coordinate (r,z)
            else Npi = Npj = Npk = Npp0;
            // User: use glb_v_id to index into your data
            NN              = Npi * Npj * Npk; // make a regular grid of particles Npp x Npp
            Np_t[grid][tid] = NN;
            if (glb_v_id == v_target) nTargetP[grid] = NN;
            PetscCall(PetscMalloc4(NN, &xx_t[grid][tid], NN, &yy_t[grid][tid], NN, &wp_t[grid][tid], dim == 2 ? 1 : NN, &zz_t[grid][tid]));
            hp[0] = (hi[0] - lo[0]) / Npi;
            hp[1] = (hi[1] - lo[1]) / Npj;
            hp[2] = (hi[2] - lo[2]) / Npk;
            if (dim == 2) hp[2] = 1;
            PetscCall(PetscInfo(pack, " lo = %14.7e, hi = %14.7e; hp = %14.7e, %14.7e; kT_m = %g; \n", (double)lo[1], (double)hi[1], (double)hp[0], (double)hp[1], (double)kT_m)); // temp
            vole = hp[0] * hp[1] * hp[2] * ctx->n[grid];                                                                                                                           // fix for multi-species
            PetscCall(PetscInfo(pack, "Vertex %" PetscInt_FMT ", grid %" PetscInt_FMT " with %" PetscInt_FMT " particles (diagnostic target = %" PetscInt_FMT ")\n", glb_v_id, grid, NN, v_target));
            for (int pj = 0, pp = 0; pj < Npj; pj++) {
              for (int pk = 0; pk < Npk; pk++) {
                for (int pi = 0; pi < Npi; pi++, pp++) {
                  xx_t[grid][tid][pp] = lo[0] + hp[0] / 2.0 + pi * hp[0];
                  yy_t[grid][tid][pp] = lo[1] + hp[1] / 2.0 + pj * hp[1];
                  if (dim == 3) zz_t[grid][tid][pp] = lo[2] + hp[2] / 2.0 + pk * hp[2];
                  {
                    PetscReal x[] = {xx_t[grid][tid][pp], yy_t[grid][tid][pp], dim == 2 ? 0 : zz_t[grid][tid][pp]};
                    maxwellian(dim, x, kT_m, vole, shift, &wp_t[grid][tid][pp]);
                    // PetscCall(PetscInfo(pack,"%" PetscInt_FMT ") x = %14.7e, %14.7e, %14.7e, n = %14.7e, w = %14.7e\n", pp, x[0], x[1], dim==2 ? 0 : x[2], ctx->n[grid], wp_t[grid][tid][pp])); // temp
                    if (glb_v_id == v_target) {
                      PetscReal v2 = 0, fact = dim == 2 ? 2.0 * PETSC_PI * x[0] : 1, w = fact * wp_t[grid][tid][pp] * ctx->n_0 * ctx->masses[ctx->species_offset[grid]];
                      for (int i = 0; i < dim; ++i) v2 += PetscSqr(x[i]);
                      moments_0[0] += w;                   // not thread safe
                      moments_0[1] += w * ctx->v_0 * x[1]; // z-momentum
                      moments_0[2] += w * ctx->v_0 * ctx->v_0 * v2;
                    }
                  }
                }
              }
            }
          }
          // entropy
          if (glb_v_id == v_target) {
            printCtx->N_inv = 1 / moments_0[0];
            PetscCall(PetscInfo(pack, "Target %" PetscInt_FMT " with %" PetscInt_FMT " particels\n", glb_v_id, nTargetP[0]));
            for (PetscInt grid = 0; grid < ctx->num_grids; grid++) {
              NN = nTargetP[grid];
              for (int pp = 0; pp < NN; pp++) {
                const PetscReal fact = dim == 2 ? 2.0 * PETSC_PI * xx_t[grid][tid][pp] : 1, w = fact * ctx->n_0 * ctx->masses[ctx->species_offset[grid]] * wp_t[grid][tid][pp], ww = w * printCtx->N_inv;
                if (ww > PETSC_REAL_MIN) {
                  moments_0[3] -= ww * PetscLogReal(ww);
                  PetscCheck(ww < 1 - PETSC_MACHINE_EPSILON, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "ww (%g) > 1", (double)ww);
                }
              }
            } // diagnostics
          }   // grid
        }     // active
      }       // threads
      /* Create particle swarm */
      for (int tid = 0; tid < numthreads; tid++) {
        const PetscInt v_id = v_id_0 + tid, glb_v_id = global_vertex_id_0 + v_id;
        if (glb_v_id < num_vertices) {                             // the ragged edge of the last batch
          for (PetscInt grid = 0; grid < ctx->num_grids; grid++) { // add same particels for all grids
            PetscErrorCode ierr_t;
            PetscSection   section;
            PetscInt       Nf;
            DM             dm = grid_dm[grid];
            ierr_t            = DMGetLocalSection(dm, &section);
            ierr_t            = PetscSectionGetNumFields(section, &Nf);
            if (Nf != 1) ierr_t = (PetscErrorCode)9999;
            else {
              ierr_t = DMViewFromOptions(dm, NULL, "-dm_view");
              ierr_t = PetscInfo(pack, "call createSwarm [%" PetscInt_FMT ".%" PetscInt_FMT "] local batch index %" PetscInt_FMT "\n", v_id, grid, LAND_PACK_IDX(v_id, grid));
              ierr_t = createSwarm(dm, dim, &globSwarmArray[LAND_PACK_IDX(v_id, grid)]);
            }
            if (ierr_t) ierr = ierr_t;
          }
        } // active
      }   // threads
      PetscCheck(ierr != 9999, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Only support one species per grid");
      PetscCheck(!ierr, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Error in OMP loop. ierr = %d", (int)ierr);
      // make globMpArray
      PetscPragmaOMP(parallel for)
      for (int tid = 0; tid < numthreads; tid++) {
        const PetscInt v_id = v_id_0 + tid, glb_v_id = global_vertex_id_0 + v_id;
        if (glb_v_id < num_vertices) {
          for (PetscInt grid = 0; grid < ctx->num_grids; grid++) { // add same particels for all grids
            PetscErrorCode ierr_t;
            DM             sw = globSwarmArray[LAND_PACK_IDX(v_id, grid)];
            ierr_t            = PetscInfo(pack, "makeSwarm %" PetscInt_FMT ".%" PetscInt_FMT ") for batch %" PetscInt_FMT "\n", global_vertex_id_0, grid, LAND_PACK_IDX(v_id, grid));
            ierr_t            = makeSwarm(sw, dim, Np_t[grid][tid], xx_t[grid][tid], yy_t[grid][tid], zz_t[grid][tid]);
            if (ierr_t) ierr = ierr_t;
          }
        }
      }
      //PetscPragmaOMP(parallel for)
      for (int tid = 0; tid < numthreads; tid++) {
        const PetscInt v_id = v_id_0 + tid, glb_v_id = global_vertex_id_0 + v_id;
        if (glb_v_id < num_vertices) {
          for (PetscInt grid = 0; grid < ctx->num_grids; grid++) { // add same particels for all grids
            PetscErrorCode ierr_t;
            DM             dm = grid_dm[grid];
            DM             sw = globSwarmArray[LAND_PACK_IDX(v_id, grid)];
            ierr_t            = PetscInfo(pack, "createMp %" PetscInt_FMT ".%" PetscInt_FMT ") for batch %" PetscInt_FMT "\n", global_vertex_id_0, grid, LAND_PACK_IDX(v_id, grid));
            ierr_t            = createMp(dm, sw, &globMpArray[LAND_PACK_IDX(v_id, grid)]);
            if (ierr_t) ierr = ierr_t;
          }
        }
      }
      PetscCheck(!ierr, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Error in OMP loop. ierr = %d", (int)ierr);
      // p --> g: set X
      // PetscPragmaOMP(parallel for)
      for (int tid = 0; tid < numthreads; tid++) {
        const PetscInt v_id = v_id_0 + tid, glb_v_id = global_vertex_id_0 + v_id;
        if (glb_v_id < num_vertices) {
          for (PetscInt grid = 0; grid < ctx->num_grids; grid++) { // add same particels for all grids
            PetscErrorCode ierr_t;
            DM             dm   = grid_dm[grid];
            DM             sw   = globSwarmArray[LAND_PACK_IDX(v_id, grid)];
            Vec            subX = globXArray[LAND_PACK_IDX(v_id, grid)], work = t_fhat[grid][tid];
            ierr_t = PetscInfo(pack, "particlesToGrid %" PetscInt_FMT ".%" PetscInt_FMT ") for local batch %" PetscInt_FMT "\n", global_vertex_id_0, grid, LAND_PACK_IDX(v_id, grid));
            ierr_t = particlesToGrid(dm, sw, Np_t[grid][tid], tid, dim, wp_t[grid][tid], subX, globMpArray[LAND_PACK_IDX(v_id, grid)]);
            if (ierr_t) ierr = ierr_t;
            // u = M^_1 f_w
            ierr_t = VecCopy(subX, work);
            ierr_t = KSPSolve(t_ksp[grid][tid], work, subX);
            if (ierr_t) ierr = ierr_t;
          }
        }
      }
      PetscCheck(!ierr, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Error in OMP loop. ierr = %d", (int)ierr);
      /* Cleanup */
      for (int tid = 0; tid < numthreads; tid++) {
        const PetscInt v_id = v_id_0 + tid, glb_v_id = global_vertex_id_0 + v_id;
        if (glb_v_id < num_vertices) {
          PetscCall(PetscInfo(pack, "Free for global batch %" PetscInt_FMT " of %" PetscInt_FMT "\n", glb_v_id + 1, num_vertices));
          for (PetscInt grid = 0; grid < ctx->num_grids; grid++) { // add same particels for all grids
            PetscCall(PetscFree4(xx_t[grid][tid], yy_t[grid][tid], wp_t[grid][tid], zz_t[grid][tid]));
          }
        } // active
      }   // threads
    }     // (fake) particle loop
    // standard view of initial conditions
    if (v_target >= global_vertex_id_0 && v_target < global_vertex_id_0 + ctx->batch_sz) {
      PetscCall(DMSetOutputSequenceNumber(ctx->plex[g_target], 0, 0.0));
      PetscCall(VecViewFromOptions(globXArray[LAND_PACK_IDX(v_target % ctx->batch_sz, g_target)], NULL, "-ex30_vec_view"));
    }
    // coarse graining moments
    if (v_target >= global_vertex_id_0 && v_target < global_vertex_id_0 + ctx->batch_sz) {
      const PetscInt v_id = v_target % ctx->batch_sz;
      for (PetscInt grid = 0; grid < ctx->num_grids; grid++) {
        PetscDataType dtype;
        PetscReal    *wp, *coords;
        DM            sw = globSwarmArray[LAND_PACK_IDX(v_id, grid)];
        Vec           work, subX = globXArray[LAND_PACK_IDX(v_id, grid)];
        PetscInt      bs, NN     = nTargetP[grid];
        // C-G moments
        PetscCall(VecDuplicate(subX, &work));
        PetscCall(gridToParticles(grid_dm[grid], sw, subX, work, globMpArray[LAND_PACK_IDX(v_id, grid)], g_Mass[grid]));
        PetscCall(VecDestroy(&work));
        // moments
        PetscCall(DMSwarmGetField(sw, "DMSwarmPIC_coor", &bs, &dtype, (void **)&coords));
        PetscCall(DMSwarmGetField(sw, "w_q", &bs, &dtype, (void **)&wp)); // could get NN from sw - todo
        for (int pp = 0; pp < NN; pp++) {
          PetscReal v2 = 0, fact = (dim == 2) ? 2.0 * PETSC_PI * coords[pp * dim + 0] : 1, w = fact * wp[pp] * ctx->n_0 * ctx->masses[ctx->species_offset[grid]], ww = w * printCtx->N_inv;
          for (int i = 0; i < dim; ++i) v2 += PetscSqr(coords[pp * dim + i]);
          moments_1a[0] += w;
          moments_1a[1] += w * ctx->v_0 * coords[pp * dim + 1]; // z-momentum
          moments_1a[2] += w * ctx->v_0 * ctx->v_0 * v2;
          if (ww > PETSC_REAL_MIN) {
            moments_1a[3] -= ww * PetscLogReal(ww);
            PetscCheck(ww < 1 - PETSC_MACHINE_EPSILON, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "ww (%g) > 1", (double)ww);
          }
        }
        PetscCall(DMSwarmRestoreField(sw, "DMSwarmPIC_coor", &bs, &dtype, (void **)&coords));
        PetscCall(DMSwarmRestoreField(sw, "w_q", &bs, &dtype, (void **)&wp));
      }
    }
    // restore vector
    PetscCall(DMCompositeRestoreAccessArray(pack, X, nDMs, NULL, globXArray));
    // view
    PetscCall(DMPlexLandauPrintNorms(X, 0));
    // advance
    PetscCall(TSSetSolution(ts, X));
    PetscCall(PetscInfo(pack, "Advance vertex %" PetscInt_FMT " to %" PetscInt_FMT " (with padding)\n", global_vertex_id_0, global_vertex_id_0 + ctx->batch_sz));
    PetscCall(TSSetPostStep(ts, PostStep));
    PetscCall(PostStep(ts));
    PetscCall(TSSolve(ts, X));
    // view
    PetscCall(DMPlexLandauPrintNorms(X, 1));
    PetscCall(DMCompositeGetAccessArray(pack, X, nDMs, NULL, globXArray));
    if (v_target >= global_vertex_id_0 && v_target < global_vertex_id_0 + ctx->batch_sz) {
      PetscCall(DMSetOutputSequenceNumber(ctx->plex[g_target], 1, dt_init));
      PetscCall(VecViewFromOptions(globXArray[LAND_PACK_IDX(v_target % ctx->batch_sz, g_target)], NULL, "-ex30_vec_view"));
    }
    // particles to grid, compute moments and entropy
    PetscCall(gridToParticles_private(grid_dm, globSwarmArray, dim, v_target, numthreads, num_vertices, global_vertex_id_0, globMpArray, g_Mass, t_fhat, moments_1b, globXArray, ctx));
    // restore vector
    PetscCall(DMCompositeRestoreAccessArray(pack, X, nDMs, NULL, globXArray));
    // cleanup
    for (PetscInt v_id_0 = 0; v_id_0 < ctx->batch_sz; v_id_0 += numthreads) {
      for (int tid = 0; tid < numthreads; tid++) {
        const PetscInt v_id = v_id_0 + tid, glb_v_id = global_vertex_id_0 + v_id;
        if (glb_v_id < num_vertices) {
          for (PetscInt grid = 0; grid < ctx->num_grids; grid++) {
            PetscCall(DMDestroy(&globSwarmArray[LAND_PACK_IDX(v_id, grid)]));
            PetscCall(MatDestroy(&globMpArray[LAND_PACK_IDX(v_id, grid)]));
          }
        }
      }
    }
  } // user batch
  /* Cleanup */
  PetscCall(PetscFree(globXArray));
  PetscCall(PetscFree(globSwarmArray));
  PetscCall(PetscFree(globMpArray));
  PetscCall(PetscFree(printCtx));
  // clean up mass matrices
  for (PetscInt grid = 0; grid < ctx->num_grids; grid++) { // add same particels for all grids
    PetscCall(MatDestroy(&g_Mass[grid]));
    for (int tid = 0; tid < numthreads; tid++) {
      PetscCall(VecDestroy(&t_fhat[grid][tid]));
      PetscCall(KSPDestroy(&t_ksp[grid][tid]));
    }
  }
  PetscCall(PetscInfo(X, "Moments:\t         number density      x-momentum          energy             entropy : # OMP threads %g\n", (double)numthreads));
  PetscCall(PetscInfo(X, "\tInitial:         %18.12e %19.12e %18.12e %e\n", (double)moments_0[0], (double)moments_0[1], (double)moments_0[2], (double)moments_0[3]));
  PetscCall(PetscInfo(X, "\tCoarse-graining: %18.12e %19.12e %18.12e %e\n", (double)moments_1a[0], (double)moments_1a[1], (double)moments_1a[2], (double)moments_1a[3]));
  PetscCall(PetscInfo(X, "\tLandau:          %18.12e %19.12e %18.12e %e\n", (double)moments_1b[0], (double)moments_1b[1], (double)moments_1b[2], (double)moments_1b[3]));
  PetscCall(PetscInfo(X, "Coarse-graining entropy generation = %e ; Landau entropy generation = %e\n", (double)(moments_1a[3] - moments_0[3]), (double)(moments_1b[3] - moments_0[3])));
  PetscCall(PetscInfo(X, "(relative) energy conservation: Coarse-graining = %e ; Landau = %e\n", (double)(moments_1a[2] - moments_0[2]) / (double)moments_0[2], (double)(moments_1b[2] - moments_0[2]) / (double)moments_0[2]));

  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM         pack;
  Vec        X;
  PetscInt   dim = 2, num_vertices = 1, Np = 10, v_target = 0, gtarget = 0;
  TS         ts;
  Mat        J;
  LandauCtx *ctx;
  PetscReal  shift = 0;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  // process args
  PetscOptionsBegin(PETSC_COMM_SELF, "", "Collision Options", "DMPLEX");
  PetscCall(PetscOptionsInt("-number_spatial_vertices", "Number of user spatial vertices to be batched for Landau", "ex30.c", num_vertices, &num_vertices, NULL));
  PetscCall(PetscOptionsInt("-dim", "Velocity space dimension", "ex30.c", dim, &dim, NULL));
  PetscCall(PetscOptionsInt("-number_particles_per_dimension", "Number of particles per grid, with slight modification per spatial vertex, in each dimension of base Cartesian grid", "ex30.c", Np, &Np, NULL));
  PetscCall(PetscOptionsInt("-vertex_view_target", "Vertex to view with diagnostics", "ex30.c", v_target, &v_target, NULL));
  PetscCall(PetscOptionsReal("-e_shift", "Bim-Maxwellian shift", "ex30.c", shift, &shift, NULL));
  PetscCheck(v_target < num_vertices, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Batch to view %" PetscInt_FMT " should be < number of vertices %" PetscInt_FMT, v_target, num_vertices);
  PetscCall(PetscOptionsInt("-grid_view_target", "Grid to view with diagnostics", "ex30.c", gtarget, &gtarget, NULL));
  PetscOptionsEnd();
  /* Create a mesh */
  PetscCall(DMPlexLandauCreateVelocitySpace(PETSC_COMM_SELF, dim, "", &X, &J, &pack));
  PetscCall(DMSetUp(pack));
  PetscCall(DMSetOutputSequenceNumber(pack, 0, 0.0));
  PetscCall(DMGetApplicationContext(pack, &ctx));
  PetscCheck(gtarget < ctx->num_grids, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Grid to view %" PetscInt_FMT " should be < number of grids %" PetscInt_FMT, gtarget, ctx->num_grids);
  PetscCheck(num_vertices >= ctx->batch_sz, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Number of vertices %" PetscInt_FMT " should be <= batch size %" PetscInt_FMT, num_vertices, ctx->batch_sz);
  /* Create timestepping solver context */
  PetscCall(TSCreate(PETSC_COMM_SELF, &ts));
  PetscCall(TSSetDM(ts, pack));
  PetscCall(TSSetIFunction(ts, NULL, DMPlexLandauIFunction, NULL));
  PetscCall(TSSetIJacobian(ts, J, J, DMPlexLandauIJacobian, NULL));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER));
  PetscCall(TSSetFromOptions(ts));
  PetscCall(PetscObjectSetName((PetscObject)X, "X"));
  // do particle advance
  PetscCall(go(ts, X, num_vertices, Np, dim, v_target, gtarget, shift));
  PetscCall(MatZeroEntries(J)); // need to zero out so as to not reuse it in Landau's logic
  /* clean up */
  PetscCall(DMPlexLandauDestroyVelocitySpace(&pack));
  PetscCall(TSDestroy(&ts));
  PetscCall(VecDestroy(&X));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  build:
    requires: !complex

  testset:
    requires: double defined(PETSC_USE_DMLANDAU_2D)
    output_file: output/ex30_0.out
    args: -dim 2 -petscspace_degree 3 -dm_landau_num_species_grid 1,1,1 -dm_refine 1 -number_particles_per_dimension 10 -dm_plex_hash_location \
          -dm_landau_batch_size 4 -number_spatial_vertices 5 -dm_landau_batch_view_idx 1 -vertex_view_target 2 -grid_view_target 1 \
          -dm_landau_n 1.000018,1,1e-6 -dm_landau_thermal_temps 2,1,1 -dm_landau_ion_masses 2,180 -dm_landau_ion_charges 1,18 \
          -ftop_ksp_rtol 1e-10 -ftop_ksp_type lsqr -ftop_pc_type bjacobi -ftop_sub_pc_factor_shift_type nonzero -ftop_sub_pc_type lu \
          -ksp_type preonly -pc_type lu -dm_landau_verbose 4 \
          -ptof_ksp_type cg -ptof_pc_type jacobi -ptof_ksp_rtol 1e-12\
          -snes_converged_reason -snes_monitor -snes_rtol 1e-14 -snes_stol 1e-14\
          -ts_dt 0.01 -ts_rtol 1e-1 -ts_exact_final_time stepover -ts_max_snes_failures -1 -ts_max_steps 1 -ts_monitor -ts_type beuler

    test:
      suffix: cpu
      args: -dm_landau_device_type cpu
    test:
      suffix: kokkos
      requires: kokkos_kernels !openmp
      args: -dm_landau_device_type kokkos -dm_mat_type aijkokkos -dm_vec_type kokkos -pc_type bjkokkos -pc_bjkokkos_ksp_type tfqmr -pc_bjkokkos_pc_type jacobi

  testset:
    requires: double !defined(PETSC_USE_DMLANDAU_2D)
    output_file: output/ex30_3d.out
    args: -dim 3 -petscspace_degree 2 -dm_landau_num_species_grid 1,1,1 -dm_refine 0 -number_particles_per_dimension 5 -dm_plex_hash_location \
          -dm_landau_batch_size 1 -number_spatial_vertices 1 -dm_landau_batch_view_idx 0 -vertex_view_target 0 -grid_view_target 0 \
          -dm_landau_n 1.000018,1,1e-6 -dm_landau_thermal_temps 2,1,1 -dm_landau_ion_masses 2,180 -dm_landau_ion_charges 1,18 \
          -ftop_ksp_rtol 1e-12 -ftop_ksp_type cg -ftop_pc_type jacobi \
          -ksp_type preonly -pc_type lu \
          -ptof_ksp_type cg -ptof_pc_type jacobi -ptof_ksp_rtol 1e-12\
          -snes_converged_reason -snes_monitor -snes_rtol 1e-12 -snes_stol 1e-12\
          -ts_dt 0.1 -ts_exact_final_time stepover -ts_max_snes_failures -1 -ts_max_steps 1 -ts_monitor -ts_type beuler

    test:
      suffix: cpu_3d
      args: -dm_landau_device_type cpu
    test:
      suffix: kokkos_3d
      requires: kokkos_kernels !openmp
      args: -dm_landau_device_type kokkos -dm_mat_type aijkokkos -dm_vec_type kokkos -pc_type bjkokkos -pc_bjkokkos_ksp_type tfqmr -pc_bjkokkos_pc_type jacobi

  testset:
    requires: !complex double defined(PETSC_USE_DMLANDAU_2D) !cuda
    args: -dm_landau_domain_radius 6 -dm_refine 2 -dm_landau_num_species_grid 1 -dm_landau_thermal_temps 1 -petscspace_degree 3 -snes_converged_reason -ts_type beuler -ts_dt 1 -ts_max_steps 1 -ksp_type preonly -pc_type lu -snes_rtol 1e-12 -snes_stol 1e-12 -dm_landau_device_type cpu -number_particles_per_dimension 30 -e_shift 3 -ftop_ksp_rtol 1e-12 -ptof_ksp_rtol 1e-12 -dm_landau_batch_size 4 -number_spatial_vertices 4 -grid_view_target 0 -vertex_view_target 1
    test:
      suffix: simple
      args: -ex30_dm_view
    test:
      requires: hdf5
      suffix: simple_hdf5
      args: -ex30_dm_view hdf5:sol_e.h5 -ex30_vec_view hdf5:sol_e.h5::append

TEST*/
