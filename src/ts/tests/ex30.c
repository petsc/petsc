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
  PetscInt   g_target;
  PetscInt   global_vertex_id_0;
  DM        *globSwarmArray;
  LandauCtx *ctx;
  DM        *grid_dm;
  Mat       *g_Mass;
  Mat       *globMpArray;
  Vec       *globXArray;
  PetscBool  print;
  PetscBool  print_entropy;
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
  PetscCall(DMSwarmVectorDefineField(sw, "w_q"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode createMp(const DM dm, DM sw, Mat *Mp_out)
{
  PetscBool removePoints = PETSC_TRUE;
  Mat       M_p;

  PetscFunctionBeginUser;
  // migrate after coords are set
  PetscCall(DMSwarmMigrate(sw, removePoints));
  //
  PetscCall(PetscObjectSetName((PetscObject)sw, "Particle Grid"));

  /* PetscInt  N,*count,nmin=10000,nmax=0,ntot=0; */
  /* // count */
  /* PetscCall(DMSwarmCreatePointPerCellCount(sw, &N, &count)); */
  /* for (int i=0, n; i< N ; i++) { */
  /*   if ((n=count[i]) > nmax) nmax = n; */
  /*   if (n < nmin) nmin = n; */
  /*   PetscCall(PetscInfo(dm, " %d) %d particles\n", i, n)); */
  /*   ntot += n; */
  /* } */
  /* PetscCall(PetscFree(count)); */
  /* PetscCall(PetscInfo(dm, " %" PetscInt_FMT " max particle / cell, and %" PetscInt_FMT " min, ratio = %g,  %" PetscInt_FMT " total\n", nmax, nmin, (double)nmax/(double)nmin,ntot)); */

  /* This gives M f = \int_\Omega \phi f, which looks like a rhs for a PDE */
  PetscCall(DMCreateMassMatrix(sw, dm, &M_p));
  PetscCall(DMViewFromOptions(sw, NULL, "-ex30_sw_view"));
  // output
  *Mp_out = M_p;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode particlesToGrid(const DM dm, DM sw, const PetscInt a_tid, const PetscInt dim, const PetscReal a_wp[], Vec rho, Mat M_p)
{
  PetscReal    *wq;
  PetscDataType dtype;
  Vec           ff;
  PetscInt      bs, p, Np;

  PetscFunctionBeginUser;
  PetscCall(DMSwarmGetField(sw, "w_q", &bs, &dtype, (void **)&wq));
  PetscCall(DMSwarmGetLocalSize(sw, &Np));
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
PetscErrorCode gridToParticles(const DM dm, DM sw, const Vec rhs, Vec work_ferhs, Mat M_p, Mat Mass)
{
  PetscBool    is_lsqr;
  KSP          ksp;
  Mat          PM_p = NULL, MtM, D = NULL;
  Vec          ff;
  PetscInt     N, M, nzl;
  MatShellCtx *matshellctx = NULL;
  PC           pc;

  PetscFunctionBeginUser;
  // 1) apply M in, for Moore-Penrose with mass: Mp (Mp' Mp)^-1 M
  PetscCall(MatMult(Mass, rhs, work_ferhs));
  // 2) pseudo-inverse, first part: (Mp' Mp)^-1
  PetscCall(KSPCreate(PETSC_COMM_SELF, &ksp));
  PetscCall(KSPSetType(ksp, KSPCG));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCJACOBI));
  PetscCall(KSPSetOptionsPrefix(ksp, "ftop_"));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(PetscObjectTypeCompare((PetscObject)ksp, KSPLSQR, &is_lsqr));
  if (!is_lsqr) {
    PetscCall(MatGetLocalSize(M_p, &M, &N));
    if (N > M) {
      PetscCall(PetscInfo(ksp, " M (%" PetscInt_FMT ") < M (%" PetscInt_FMT ") more vertices than particles: revert to lsqr\n", M, N));
      is_lsqr = PETSC_TRUE;
      PetscCall(KSPSetType(ksp, KSPLSQR));
      PetscCall(PCSetType(pc, PCNONE)); // should not happen, but could solve stable (Mp^T Mp), move projection Mp before solve
    } else {
      PetscCall(PetscNew(&matshellctx));
      PetscCall(MatCreateVecs(M_p, &matshellctx->uu, &matshellctx->ff));
      if (0) {
        PetscCall(MatTransposeMatMult(M_p, M_p, MAT_INITIAL_MATRIX, 4, &MtM));
        PetscCall(KSPSetOperators(ksp, MtM, MtM));
        PetscCall(PetscInfo(M_p, "createMtM KSP with explicit Mp'Mp\n"));
        PetscCall(MatViewFromOptions(MtM, NULL, "-ftop2_MtM_mat_view"));
      } else {
        PetscCall(MatCreateShell(PetscObjectComm((PetscObject)dm), N, N, PETSC_DECIDE, PETSC_DECIDE, matshellctx, &MtM));
        PetscCall(MatTranspose(M_p, MAT_INITIAL_MATRIX, &matshellctx->MpTrans));
        matshellctx->Mp = M_p;
        PetscCall(MatShellSetOperation(MtM, MATOP_MULT, (PetscErrorCodeFn *)MatMultMtM_SeqAIJ));
        PetscCall(MatShellSetOperation(MtM, MATOP_MULT_ADD, (PetscErrorCodeFn *)MatMultAddMtM_SeqAIJ));
        PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, N, N, 1, NULL, &D));
        PetscCall(MatViewFromOptions(matshellctx->MpTrans, NULL, "-ftop2_MpT_mat_view"));
        for (PetscInt i = 0; i < N; i++) {
          const PetscScalar *vals;
          const PetscInt    *cols;
          PetscScalar        dot = 0;
          PetscCall(MatGetRow(matshellctx->MpTrans, i, &nzl, &cols, &vals));
          for (PetscInt ii = 0; ii < nzl; ii++) dot += PetscSqr(vals[ii]);
          if (dot < PETSC_MACHINE_EPSILON) {
            PetscCall(PetscInfo(ksp, "empty row in pseudo-inverse %d\n", (int)i));
            is_lsqr = PETSC_TRUE; // empty rows
            PetscCall(KSPSetType(ksp, KSPLSQR));
            PetscCall(PCSetType(pc, PCNONE)); // should not happen, but could solve stable (Mp Mp^T), move projection Mp before solve
            // clean up
            PetscCall(MatDestroy(&matshellctx->MpTrans));
            PetscCall(VecDestroy(&matshellctx->ff));
            PetscCall(VecDestroy(&matshellctx->uu));
            PetscCall(MatDestroy(&D));
            PetscCall(MatDestroy(&MtM));
            PetscCall(PetscFree(matshellctx));
            D = NULL;
            break;
          }
          PetscCall(MatSetValue(D, i, i, dot, INSERT_VALUES));
        }
        if (D) {
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
    }
  }
  if (is_lsqr) {
    PC        pc2;
    PetscBool is_bjac;
    PetscCall(KSPGetPC(ksp, &pc2));
    PetscCall(PetscObjectTypeCompare((PetscObject)pc2, PCBJACOBI, &is_bjac));
    if (is_bjac) {
      PetscCall(DMSwarmCreateMassMatrixSquare(sw, dm, &PM_p));
      PetscCall(KSPSetOperators(ksp, M_p, PM_p));
    } else {
      PetscCall(KSPSetOperators(ksp, M_p, M_p));
    }
    PetscCall(MatViewFromOptions(M_p, NULL, "-ftop2_Mp_mat_view"));
  }
  PetscCall(DMSwarmCreateGlobalVectorFromField(sw, "w_q", &ff)); // this grabs access
  if (!is_lsqr) {
    PetscCall(KSPSolve(ksp, work_ferhs, matshellctx->uu));
    // 3) with Moore-Penrose apply Mp: M_p (Mp' Mp)^-1 M
    PetscCall(MatMult(M_p, matshellctx->uu, ff));
    if (D) PetscCall(MatDestroy(&D));
    PetscCall(MatDestroy(&MtM));
    if (matshellctx->MpTrans) PetscCall(MatDestroy(&matshellctx->MpTrans));
    PetscCall(VecDestroy(&matshellctx->ff));
    PetscCall(VecDestroy(&matshellctx->uu));
    PetscCall(PetscFree(matshellctx));
  } else {
    // finally with LSQR apply M_p^\dagger
    PetscCall(KSPSolveTranspose(ksp, work_ferhs, ff));
  }
  PetscCall(KSPDestroy(&ksp));
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
    for (PetscInt tid = 0; tid < numthreads; tid++) {
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
    PetscCheck(!ierr, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Error in OMP loop. ierr = %d", (int)ierr);
    /* Get moments */
    PetscCall(PetscInfo(grid_dm[0], "Cleanup batches %" PetscInt_FMT " to %" PetscInt_FMT "\n", v_id_0, v_id_0 + numthreads));
    for (PetscInt tid = 0; tid < numthreads; tid++) {
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
          for (PetscInt p = 0; p < npoints; p++) {
            PetscReal v2 = 0, fact = (dim == 2) ? 2.0 * PETSC_PI * coords[p * dim + 0] : 1, w = fact * wp[p] * ctx->n_0 * ctx->masses[ctx->species_offset[grid]];
            for (PetscInt i = 0; i < dim; ++i) v2 += PetscSqr(coords[p * dim + i]);
            moments[0] += w;
            moments[1] += w * ctx->v_0 * coords[p * dim + 1]; // z-momentum
            moments[2] += w * 0.5 * ctx->v_0 * ctx->v_0 * v2;
          }
          PetscCall(DMSwarmRestoreField(sw, "w_q", &bs, &dtype, (void **)&wp));
          PetscCall(DMSwarmRestoreField(sw, "DMSwarmPIC_coor", &bs, &dtype, (void **)&coords));
        }
        const PetscReal N_inv = 1 / moments[0];
        PetscCall(PetscInfo(grid_dm[0], "gridToParticles_private [%" PetscInt_FMT "], n = %g\n", v_id, (double)moments[0]));
        for (PetscInt grid = 0; grid < ctx->num_grids; grid++) {
          PetscDataType dtype;
          PetscReal    *wp, *coords;
          DM            sw = globSwarmArray[LAND_PACK_IDX(v_id, grid)];
          PetscInt      npoints, bs = 1;
          PetscCall(DMSwarmGetField(sw, "w_q", &bs, &dtype, (void **)&wp)); // take data out here
          PetscCall(DMSwarmGetField(sw, "DMSwarmPIC_coor", &bs, &dtype, (void **)&coords));
          PetscCall(DMSwarmGetLocalSize(sw, &npoints));
          for (PetscInt p = 0; p < npoints; p++) {
            const PetscReal fact = dim == 2 ? 2.0 * PETSC_PI * coords[p * dim + 0] : 1, w = fact * wp[p] * ctx->n_0 * ctx->masses[ctx->species_offset[grid]], ww = w * N_inv;
            if (w > PETSC_REAL_MIN) {
              moments[3] -= ww * PetscLogReal(ww);
              PetscCheck(ww < 1 - PETSC_MACHINE_EPSILON, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "ww (%g) > 1", (double)ww);
            } else moments[4] -= w; // keep track of density that is lost
          }
          PetscCall(DMSwarmRestoreField(sw, "w_q", &bs, &dtype, (void **)&wp));
          PetscCall(DMSwarmRestoreField(sw, "DMSwarmPIC_coor", &bs, &dtype, (void **)&coords));
        }
      }
    } // thread batch
  } // batch
  PetscFunctionReturn(PETSC_SUCCESS);
}

static void maxwellian(PetscInt dim, const PetscReal x[], PetscReal kt_m, PetscReal n, PetscReal shift, PetscScalar *u)
{
  PetscInt  i;
  PetscReal v2 = 0, theta = 2.0 * kt_m; /* theta = 2kT/mc^2 */

  if (shift != 0.) {
    v2 = 0;
    for (i = 0; i < dim - 1; ++i) v2 += x[i] * x[i];
    v2 += (x[dim - 1] - shift) * (x[dim - 1] - shift);
    /* evaluate the shifted Maxwellian */
    u[0] += n * PetscPowReal(PETSC_PI * theta, -1.5) * (PetscExpReal(-v2 / theta));
  } else {
    /* compute the exponents, v^2 */
    for (i = 0; i < dim; ++i) v2 += x[i] * x[i];
    /* evaluate the Maxwellian */
    u[0] += n * PetscPowReal(PETSC_PI * theta, -1.5) * (PetscExpReal(-v2 / theta));
  }
}

static PetscErrorCode PostStep(TS ts)
{
  PetscInt   n, dim, nDMs, v_id;
  PetscReal  t;
  LandauCtx *ctx;
  Vec        X;
  PrintCtx  *printCtx;
  DM         pack;
  PetscReal  moments[5], e_grid[LANDAU_MAX_GRIDS];

  PetscFunctionBeginUser;
  PetscCall(TSGetApplicationContext(ts, &printCtx));
  if (!printCtx->print && !printCtx->print_entropy) PetscFunctionReturn(PETSC_SUCCESS);
  ctx = printCtx->ctx;
  if (printCtx->v_target < printCtx->global_vertex_id_0 || printCtx->v_target >= printCtx->global_vertex_id_0 + ctx->batch_sz) PetscFunctionReturn(PETSC_SUCCESS);
  for (PetscInt i = 0; i < 5; i++) moments[i] = 0;
  for (PetscInt i = 0; i < LANDAU_MAX_GRIDS; i++) e_grid[i] = 0;
  v_id = printCtx->v_target % ctx->batch_sz;
  PetscCall(TSGetDM(ts, &pack));
  PetscCall(DMGetDimension(pack, &dim));
  PetscCall(DMCompositeGetNumberDM(pack, &nDMs)); // number of vertices * number of grids
  PetscCall(TSGetSolution(ts, &X));
  PetscCall(TSGetStepNumber(ts, &n));
  PetscCall(TSGetTime(ts, &t));
  PetscCall(DMCompositeGetAccessArray(pack, X, nDMs, NULL, printCtx->globXArray));
  if (printCtx->print_entropy && printCtx->v_target >= 0 && 0) {
    for (PetscInt grid = 0; grid < ctx->num_grids; grid++) {
      PetscDataType dtype;
      PetscReal    *wp, *coords;
      DM            sw = printCtx->globSwarmArray[LAND_PACK_IDX(v_id, grid)];
      Vec           work, subX = printCtx->globXArray[LAND_PACK_IDX(v_id, grid)];
      PetscInt      bs, NN;
      // C-G moments
      PetscCall(VecDuplicate(subX, &work));
      PetscCall(gridToParticles(printCtx->grid_dm[grid], sw, subX, work, printCtx->globMpArray[LAND_PACK_IDX(v_id, grid)], printCtx->g_Mass[grid]));
      PetscCall(VecDestroy(&work));
      // moments
      PetscCall(DMSwarmGetField(sw, "DMSwarmPIC_coor", &bs, &dtype, (void **)&coords));
      PetscCall(DMSwarmGetLocalSize(sw, &NN));
      PetscCall(DMSwarmGetField(sw, "w_q", &bs, &dtype, (void **)&wp));
      for (PetscInt pp = 0; pp < NN; pp++) {
        PetscReal v2 = 0, fact = (dim == 2) ? 2.0 * PETSC_PI * coords[pp * dim + 0] : 1, w = fact * wp[pp] * ctx->n_0 * ctx->masses[ctx->species_offset[grid]];
        for (PetscInt i = 0; i < dim; ++i) v2 += PetscSqr(coords[pp * dim + i]);
        moments[0] += w;
        moments[1] += w * ctx->v_0 * coords[pp * dim + 1]; // z-momentum
        moments[2] += w * 0.5 * ctx->v_0 * ctx->v_0 * v2;
        e_grid[grid] += w * 0.5 * ctx->v_0 * ctx->v_0 * v2;
      }
      PetscCall(DMSwarmRestoreField(sw, "DMSwarmPIC_coor", &bs, &dtype, (void **)&coords));
      PetscCall(DMSwarmRestoreField(sw, "w_q", &bs, &dtype, (void **)&wp));
    }
    // entropy
    const PetscReal N_inv = 1 / moments[0];
    for (PetscInt grid = 0; grid < ctx->num_grids; grid++) {
      PetscDataType dtype;
      PetscReal    *wp, *coords;
      DM            sw = printCtx->globSwarmArray[LAND_PACK_IDX(v_id, grid)];
      PetscInt      bs, NN;
      PetscCall(DMSwarmGetField(sw, "DMSwarmPIC_coor", &bs, &dtype, (void **)&coords));
      PetscCall(DMSwarmGetLocalSize(sw, &NN));
      PetscCall(DMSwarmGetField(sw, "w_q", &bs, &dtype, (void **)&wp));
      for (PetscInt pp = 0; pp < NN; pp++) {
        PetscReal fact = (dim == 2) ? 2.0 * PETSC_PI * coords[pp * dim + 0] : 1, w = fact * wp[pp] * ctx->n_0 * ctx->masses[ctx->species_offset[grid]], ww = w * N_inv;
        if (w > PETSC_REAL_MIN) {
          moments[3] -= ww * PetscLogReal(ww);
        } else moments[4] -= w;
      }
      PetscCall(DMSwarmRestoreField(sw, "DMSwarmPIC_coor", &bs, &dtype, (void **)&coords));
      PetscCall(DMSwarmRestoreField(sw, "w_q", &bs, &dtype, (void **)&wp));
    }
    PetscCall(PetscInfo(X, "%4d) time %e, Landau particle moments: 0: %18.12e 1: %19.12e 2: %18.12e entropy: %e loss %e. energy = %e + %e + %e\n", (int)n, (double)t, (double)moments[0], (double)moments[1], (double)moments[2], (double)moments[3], (double)(moments[4] / moments[0]), (double)e_grid[0], (double)e_grid[1], (double)e_grid[2]));
  }
  if (printCtx->print && printCtx->g_target >= 0) {
    PetscInt         grid   = printCtx->g_target, id;
    static PetscReal last_t = -100000, period = .5;
    if (last_t == -100000) last_t = -period + t;
    if (t >= last_t + period) {
      last_t = t;
      PetscCall(DMGetOutputSequenceNumber(ctx->plex[grid], &id, NULL));
      PetscCall(DMSetOutputSequenceNumber(ctx->plex[grid], id + 1, t));
      PetscCall(VecViewFromOptions(printCtx->globXArray[LAND_PACK_IDX(v_id % ctx->batch_sz, grid)], NULL, "-ex30_vec_view"));
      if (ctx->num_grids > grid + 1) {
        PetscCall(DMSetOutputSequenceNumber(ctx->plex[grid + 1], id + 1, t));
        PetscCall(VecViewFromOptions(printCtx->globXArray[LAND_PACK_IDX(v_id % ctx->batch_sz, grid + 1)], NULL, "-ex30_vec_view2"));
      }
      PetscCall(PetscInfo(X, "%4d) time %e View\n", (int)n, (double)t));
    }
  }
  PetscCall(DMCompositeRestoreAccessArray(pack, X, nDMs, NULL, printCtx->globXArray));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode go(TS ts, Vec X, const PetscInt num_vertices, const PetscInt a_Np, const PetscInt dim, const PetscInt v_target, const PetscInt g_target, PetscReal shift, PetscBool use_uniform_particle_grid)
{
  DM             pack, *globSwarmArray, grid_dm[LANDAU_MAX_GRIDS];
  Mat           *globMpArray, g_Mass[LANDAU_MAX_GRIDS];
  KSP            t_ksp[LANDAU_MAX_GRIDS][EX30_MAX_NUM_THRDS];
  Vec            t_fhat[LANDAU_MAX_GRIDS][EX30_MAX_NUM_THRDS];
  PetscInt       nDMs;
  PetscErrorCode ierr = (PetscErrorCode)0; // used for inside thread loops
#if defined(PETSC_HAVE_OPENMP) && defined(PETSC_HAVE_THREADSAFETY)
  PetscInt numthreads = PetscNumOMPThreads;
#else
  PetscInt numthreads = 1;
#endif
  LandauCtx *ctx;
  Vec       *globXArray;
  PetscReal  moments_0[5], moments_1a[5], moments_1b[5], dt_init;
  PrintCtx  *printCtx;

  PetscFunctionBeginUser;
  PetscCheck(numthreads <= EX30_MAX_NUM_THRDS, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Too many threads %" PetscInt_FMT " > %d", numthreads, EX30_MAX_NUM_THRDS);
  PetscCheck(numthreads > 0, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Number threads %" PetscInt_FMT " > %d", numthreads, EX30_MAX_NUM_THRDS);
  PetscCall(TSGetDM(ts, &pack));
  PetscCall(DMGetApplicationContext(pack, &ctx));
  PetscCheck(ctx->batch_sz % numthreads == 0, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "batch size (-dm_landau_batch_size) %" PetscInt_FMT "  mod #threads %" PetscInt_FMT " must equal zero", ctx->batch_sz, numthreads);
  PetscCall(DMCompositeGetNumberDM(pack, &nDMs)); // number of vertices * number of grids
  PetscCall(PetscInfo(pack, "Have %" PetscInt_FMT " total grids, with %" PetscInt_FMT " Landau local batched and %" PetscInt_FMT " global items (vertices) %d DMs\n", ctx->num_grids, ctx->batch_sz, num_vertices, (int)nDMs));
  PetscCall(PetscMalloc(sizeof(*globXArray) * nDMs, &globXArray));
  PetscCall(PetscMalloc(sizeof(*globMpArray) * nDMs, &globMpArray));
  PetscCall(PetscMalloc(sizeof(*globSwarmArray) * nDMs, &globSwarmArray));
  // print ctx
  PetscCall(PetscNew(&printCtx));
  PetscCall(TSSetApplicationContext(ts, printCtx));
  printCtx->v_target       = v_target;
  printCtx->g_target       = g_target;
  printCtx->ctx            = ctx;
  printCtx->globSwarmArray = globSwarmArray;
  printCtx->grid_dm        = grid_dm;
  printCtx->globMpArray    = globMpArray;
  printCtx->g_Mass         = g_Mass;
  printCtx->globXArray     = globXArray;
  printCtx->print_entropy  = PETSC_FALSE;
  PetscOptionsBegin(PETSC_COMM_SELF, "", "Print Options", "DMPLEX");
  PetscCall(PetscOptionsBool("-print_entropy", "Print entropy and moments at each time step", "ex30.c", printCtx->print_entropy, &printCtx->print_entropy, NULL));
  PetscOptionsEnd();
  // view
  PetscCall(DMViewFromOptions(ctx->plex[g_target], NULL, "-ex30_dm_view"));
  if (ctx->num_grids > g_target + 1) PetscCall(DMViewFromOptions(ctx->plex[g_target + 1], NULL, "-ex30_dm_view2"));
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
    for (PetscInt tid = 0; tid < numthreads; tid++) {
      PC pc;
      PetscCall(VecDuplicate(subX, &t_fhat[grid][tid]));
      PetscCall(KSPCreate(PETSC_COMM_SELF, &t_ksp[grid][tid]));
      PetscCall(KSPSetType(t_ksp[grid][tid], KSPCG));
      PetscCall(KSPGetPC(t_ksp[grid][tid], &pc));
      PetscCall(PCSetType(pc, PCJACOBI));
      PetscCall(KSPSetOptionsPrefix(t_ksp[grid][tid], "ptof_"));
      PetscCall(KSPSetOperators(t_ksp[grid][tid], g_Mass[grid], g_Mass[grid]));
      PetscCall(KSPSetFromOptions(t_ksp[grid][tid]));
    }
  }
  PetscCall(DMCompositeRestoreAccessArray(pack, X, nDMs, NULL, globXArray));
  PetscCall(TSGetTimeStep(ts, &dt_init)); // we could have an adaptive time stepper
  // loop over all vertices in chucks that are batched for TSSolve
  for (PetscInt i = 0; i < 5; i++) moments_0[i] = moments_1a[i] = moments_1b[i] = 0;
  for (PetscInt global_vertex_id_0 = 0; global_vertex_id_0 < num_vertices; global_vertex_id_0 += ctx->batch_sz, shift /= 2) { // outer vertex loop
    PetscCall(TSSetTime(ts, 0));
    PetscCall(TSSetStepNumber(ts, 0));
    PetscCall(TSSetTimeStep(ts, dt_init));
    PetscCall(DMCompositeGetAccessArray(pack, X, nDMs, NULL, globXArray));
    printCtx->global_vertex_id_0 = global_vertex_id_0;
    if (v_target >= global_vertex_id_0 && v_target < global_vertex_id_0 + ctx->batch_sz) {
      PetscCall(PetscObjectSetName((PetscObject)globXArray[LAND_PACK_IDX(v_target % ctx->batch_sz, g_target)], "rho"));
      printCtx->print = PETSC_TRUE;
    } else printCtx->print = PETSC_FALSE;
    // create fake particles in batches with threads
    for (PetscInt v_id_0 = 0; v_id_0 < ctx->batch_sz; v_id_0 += numthreads) {
      PetscReal *xx_t[LANDAU_MAX_GRIDS][EX30_MAX_NUM_THRDS], *yy_t[LANDAU_MAX_GRIDS][EX30_MAX_NUM_THRDS], *zz_t[LANDAU_MAX_GRIDS][EX30_MAX_NUM_THRDS], *wp_t[LANDAU_MAX_GRIDS][EX30_MAX_NUM_THRDS] /* , radiuses[80000] */;
      PetscInt   Np_t[LANDAU_MAX_GRIDS][EX30_MAX_NUM_THRDS];
      // make particles
      for (PetscInt tid = 0; tid < numthreads; tid++) {
        const PetscInt v_id = v_id_0 + tid, glb_v_id = global_vertex_id_0 + v_id;
        if (glb_v_id < num_vertices) {                                                     // the ragged edge (in last batch)
          PetscInt Npp0 = a_Np + (glb_v_id % (a_Np / 10 + 1)), nTargetP[LANDAU_MAX_GRIDS]; // n of particels in each dim with load imbalance
          for (PetscInt grid = 0; grid < ctx->num_grids; grid++) {                         // add same particels for all grids
            // for (PetscInt sp = ctx->species_offset[grid], i0 = 0; sp < ctx->species_offset[grid + 1]; sp++, i0++) {
            const PetscReal kT_m  = ctx->k * ctx->thermal_temps[ctx->species_offset[grid]] / ctx->masses[ctx->species_offset[grid]] / (ctx->v_0 * ctx->v_0);                      /* theta = 2kT/mc^2 per species */
            PetscReal       lo[3] = {-ctx->radius[grid], -ctx->radius[grid], -ctx->radius[grid]}, hi[3] = {ctx->radius[grid], ctx->radius[grid], ctx->radius[grid]}, hp[3], vole; // would be nice to get box from DM
            PetscInt        Npi = Npp0, Npj = 2 * Npp0, Npk = 1;
            PetscRandom     rand;
            PetscReal       sigma = ctx->thermal_speed[grid] / ctx->thermal_speed[0], p2_shift = grid == 0 ? shift : -shift; // symmetric shift of e vs ions
            PetscCall(PetscRandomCreate(PETSC_COMM_SELF, &rand));
            PetscCall(PetscRandomSetInterval(rand, 0., 1.));
            PetscCall(PetscRandomSetFromOptions(rand));
            if (dim == 2) lo[0] = 0; // Landau coordinate (r,z)
            else Npi = Npj = Npk = Npp0;
            // User: use glb_v_id to index into your data
            const PetscInt NNreal = Npi * Npj * Npk, NN = NNreal + (dim == 2 ? 3 : 6); // make room for bounding box
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
            for (PetscInt pj = 0, pp = 0; pj < Npj; pj++) {
              for (PetscInt pk = 0; pk < Npk; pk++) {
                for (PetscInt pi = 0; pi < Npi; pi++, pp++) {
                  PetscReal p_shift   = p2_shift;
                  wp_t[grid][tid][pp] = 0;
                  if (use_uniform_particle_grid) {
                    xx_t[grid][tid][pp] = lo[0] + hp[0] / 2.0 + pi * hp[0];
                    yy_t[grid][tid][pp] = lo[1] + hp[1] / 2.0 + pj * hp[1];
                    if (dim == 3) zz_t[grid][tid][pp] = lo[2] + hp[2] / 2.0 + pk * hp[2];
                    PetscReal x[] = {xx_t[grid][tid][pp], yy_t[grid][tid][pp], dim == 2 ? 0 : zz_t[grid][tid][pp]};
                    p_shift *= ctx->thermal_speed[grid] / ctx->v_0;
                    if (ctx->sphere && PetscSqrtReal(PetscSqr(xx_t[grid][tid][pp]) + PetscSqr(yy_t[grid][tid][pp])) > 0.92 * hi[0]) {
                      wp_t[grid][tid][pp] = 0;
                    } else {
                      maxwellian(dim, x, kT_m, vole, p_shift, &wp_t[grid][tid][pp]);
                      if (ctx->num_grids == 1 && shift != 0) {                          // bi-maxwellian, electron plasma
                        maxwellian(dim, x, kT_m, vole, -p_shift, &wp_t[grid][tid][pp]); // symmetric shift of electron plasma
                      }
                    }
                  } else {
                    PetscReal u1, u2;
                    do {
                      do {
                        PetscCall(PetscRandomGetValueReal(rand, &u1));
                      } while (u1 == 0);
                      PetscCall(PetscRandomGetValueReal(rand, &u2));
                      //compute z0 and z1
                      PetscReal mag       = sigma * PetscSqrtReal(-2.0 * PetscLogReal(u1)); // is this the same scale grid Maxwellian? t_therm = sigma
                      xx_t[grid][tid][pp] = mag * PetscCosReal(2.0 * PETSC_PI * u2);
                      yy_t[grid][tid][pp] = mag * PetscSinReal(2.0 * PETSC_PI * u2);
                      if (dim == 2 && xx_t[grid][tid][pp] < lo[0]) xx_t[grid][tid][pp] = -xx_t[grid][tid][pp];
                      if (dim == 3) zz_t[grid][tid][pp] = lo[2] + hp[2] / 2.0 + pk * hp[2];
                      if (!ctx->sphere) {
                        if (dim == 2 && xx_t[grid][tid][pp] < 0) xx_t[grid][tid][pp] = -xx_t[grid][tid][pp]; // ???
                        else if (dim == 3) {
                          while (zz_t[grid][tid][pp] >= hi[2] || zz_t[grid][tid][pp] <= lo[2]) zz_t[grid][tid][pp] *= .9;
                        }
                        while (xx_t[grid][tid][pp] >= hi[0] || xx_t[grid][tid][pp] <= lo[0]) xx_t[grid][tid][pp] *= .9;
                        while (yy_t[grid][tid][pp] >= hi[1] || yy_t[grid][tid][pp] <= lo[1]) yy_t[grid][tid][pp] *= .9;
                      } else { // 2D
                        //if (glb_v_id == v_target && pp < 80000) radiuses[pp] = PetscSqrtReal(PetscSqr(xx_t[grid][tid][pp]) + PetscSqr(yy_t[grid][tid][pp]));
                        while (PetscSqrtReal(PetscSqr(xx_t[grid][tid][pp]) + PetscSqr(yy_t[grid][tid][pp])) > 0.92 * hi[0]) { // safety factor for facets of sphere
                          xx_t[grid][tid][pp] *= .9;
                          yy_t[grid][tid][pp] *= .9;
                        }
                      }
                      if (ctx->num_grids == 1 && pp % 2 == 0) p_shift = 0; // one species, split bi-max
                      p_shift *= ctx->thermal_speed[grid] / ctx->v_0;
                      if (dim == 3) zz_t[grid][tid][pp] += p_shift;
                      else yy_t[grid][tid][pp] += p_shift;
                      wp_t[grid][tid][pp] += ctx->n[grid] / NNreal * PetscSqrtReal(ctx->masses[ctx->species_offset[grid]] / ctx->masses[0]);
                      if (p_shift <= 0) break; // add bi-max for electron plasma only
                      p_shift = -p_shift;
                    } while (ctx->num_grids == 1); // add bi-max for electron plasma only
                  }
                  {
                    if (glb_v_id == v_target) {
                      PetscReal x[] = {xx_t[grid][tid][pp], yy_t[grid][tid][pp], dim == 2 ? 0 : zz_t[grid][tid][pp]};
                      PetscReal v2 = 0, fact = dim == 2 ? 2.0 * PETSC_PI * x[0] : 1, w = fact * wp_t[grid][tid][pp] * ctx->n_0 * ctx->masses[ctx->species_offset[grid]];
                      for (PetscInt i = 0; i < dim; ++i) v2 += PetscSqr(x[i]);
                      moments_0[0] += w;                   // not thread safe
                      moments_0[1] += w * ctx->v_0 * x[1]; // z-momentum
                      moments_0[2] += w * 0.5 * ctx->v_0 * ctx->v_0 * v2;
                    }
                  }
                }
              }
            }
            if (dim == 2) { // fix bounding box
              PetscInt pp           = NNreal;
              wp_t[grid][tid][pp]   = 0;
              xx_t[grid][tid][pp]   = 1.e-7;
              yy_t[grid][tid][pp++] = hi[1] - 5.e-7;
              wp_t[grid][tid][pp]   = 0;
              xx_t[grid][tid][pp]   = hi[0] - 5.e-7;
              yy_t[grid][tid][pp++] = 0;
              wp_t[grid][tid][pp]   = 0;
              xx_t[grid][tid][pp]   = 1.e-7;
              yy_t[grid][tid][pp++] = lo[1] + 5.e-7;
            } else {
              const PetscInt p0 = NNreal;
              for (PetscInt pj = 0; pj < 6; pj++) xx_t[grid][tid][p0 + pj] = yy_t[grid][tid][p0 + pj] = zz_t[grid][tid][p0 + pj] = wp_t[grid][tid][p0 + pj] = 0;
              xx_t[grid][tid][p0 + 0] = lo[0];
              xx_t[grid][tid][p0 + 1] = hi[0];
              yy_t[grid][tid][p0 + 2] = lo[1];
              yy_t[grid][tid][p0 + 3] = hi[1];
              zz_t[grid][tid][p0 + 4] = lo[2];
              zz_t[grid][tid][p0 + 5] = hi[2];
            }
            PetscCall(PetscRandomDestroy(&rand));
          }
          // entropy init, need global n
          if (glb_v_id == v_target) {
            const PetscReal N_inv = 1 / moments_0[0];
            PetscCall(PetscInfo(pack, "Target %" PetscInt_FMT " with %" PetscInt_FMT " particels\n", glb_v_id, nTargetP[0]));
            for (PetscInt grid = 0; grid < ctx->num_grids; grid++) {
              const PetscInt NN = nTargetP[grid];
              for (PetscInt pp = 0; pp < NN; pp++) {
                const PetscReal fact = dim == 2 ? 2.0 * PETSC_PI * xx_t[grid][tid][pp] : 1, w = fact * ctx->n_0 * ctx->masses[ctx->species_offset[grid]] * wp_t[grid][tid][pp], ww = w * N_inv;
                if (w > PETSC_REAL_MIN) {
                  moments_0[3] -= ww * PetscLogReal(ww);
                  PetscCheck(ww < 1 - PETSC_MACHINE_EPSILON, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "ww (%g) > 1", (double)ww);
                } else moments_0[4] -= w;
              }
            } // grid
          } // target
        } // active
      } // threads
      /* Create particle swarm */
      for (PetscInt tid = 0; tid < numthreads; tid++) {
        const PetscInt v_id = v_id_0 + tid, glb_v_id = global_vertex_id_0 + v_id;
        if (glb_v_id < num_vertices) {                             // the ragged edge of the last batch
          for (PetscInt grid = 0; grid < ctx->num_grids; grid++) { // add same particels for all grids
            PetscSection section;
            PetscInt     Nf;
            DM           dm = grid_dm[grid];
            PetscCall(DMGetLocalSection(dm, &section));
            PetscCall(PetscSectionGetNumFields(section, &Nf));
            PetscCheck(Nf == 1, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Only one species per grid supported -- todo");
            PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));
            PetscCall(PetscInfo(pack, "call createSwarm [%" PetscInt_FMT ".%" PetscInt_FMT "] local block index %" PetscInt_FMT "\n", v_id, grid, LAND_PACK_IDX(v_id, grid)));
            PetscCall(createSwarm(dm, dim, &globSwarmArray[LAND_PACK_IDX(v_id, grid)]));
          }
        } // active
      } // threads
      PetscCheck(ierr != 9999, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Only support one species per grid");
      // make globMpArray
      PetscPragmaOMP(parallel for)
      for (PetscInt tid = 0; tid < numthreads; tid++) {
        const PetscInt v_id = v_id_0 + tid, glb_v_id = global_vertex_id_0 + v_id;
        if (glb_v_id < num_vertices) {
          for (PetscInt grid = 0; grid < ctx->num_grids; grid++) { // add same particels for all grids
            // for (PetscInt sp = ctx->species_offset[grid], i0 = 0; sp < ctx->species_offset[grid + 1]; sp++, i0++) -- loop over species for Nf > 1 -- TODO
            PetscErrorCode ierr_t;
            DM             sw = globSwarmArray[LAND_PACK_IDX(v_id, grid)];
            ierr_t            = PetscInfo(pack, "makeSwarm %" PetscInt_FMT ".%" PetscInt_FMT ") for block %" PetscInt_FMT "\n", v_id, grid, LAND_PACK_IDX(v_id, grid));
            ierr_t            = makeSwarm(sw, dim, Np_t[grid][tid], xx_t[grid][tid], yy_t[grid][tid], zz_t[grid][tid]);
            if (ierr_t) ierr = ierr_t;
          }
        }
      }
      for (PetscInt tid = 0; tid < numthreads; tid++) {
        const PetscInt v_id = v_id_0 + tid, glb_v_id = global_vertex_id_0 + v_id;
        if (glb_v_id < num_vertices) {
          for (PetscInt grid = 0; grid < ctx->num_grids; grid++) { // add same particels for all grids
            DM dm = grid_dm[grid];
            DM sw = globSwarmArray[LAND_PACK_IDX(v_id, grid)];
            PetscCall(PetscInfo(pack, "createMp %" PetscInt_FMT ".%" PetscInt_FMT ") for block %" PetscInt_FMT "\n", v_id, grid, LAND_PACK_IDX(v_id, grid)));
            PetscCall(createMp(dm, sw, &globMpArray[LAND_PACK_IDX(v_id, grid)]));
            PetscCall(MatViewFromOptions(globMpArray[LAND_PACK_IDX(v_id, grid)], NULL, "-mp_mat_view"));
          }
        }
      }
      // p --> g: set X
      // PetscPragmaOMP(parallel for)
      for (PetscInt tid = 0; tid < numthreads; tid++) {
        const PetscInt v_id = v_id_0 + tid, glb_v_id = global_vertex_id_0 + v_id;
        if (glb_v_id < num_vertices) {
          for (PetscInt grid = 0; grid < ctx->num_grids; grid++) { // add same particels for all grids
            PetscErrorCode ierr_t;
            DM             dm   = grid_dm[grid];
            DM             sw   = globSwarmArray[LAND_PACK_IDX(v_id, grid)];
            Vec            subX = globXArray[LAND_PACK_IDX(v_id, grid)], work = t_fhat[grid][tid];
            ierr_t = PetscInfo(pack, "particlesToGrid %" PetscInt_FMT ".%" PetscInt_FMT ") for block %" PetscInt_FMT "\n", v_id, grid, LAND_PACK_IDX(v_id, grid));
            ierr_t = particlesToGrid(dm, sw, tid, dim, wp_t[grid][tid], subX, globMpArray[LAND_PACK_IDX(v_id, grid)]);
            if (ierr_t) ierr = ierr_t;
            // u = M^_1 f_w
            ierr_t = VecCopy(subX, work);
            ierr_t = KSPSolve(t_ksp[grid][tid], work, subX);
            if (ierr_t) ierr = ierr_t;
          }
        }
      }
      PetscCheck(!ierr, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Error in OMP loop. ierr = %d", (int)ierr);
      /* Cleanup */
      for (PetscInt tid = 0; tid < numthreads; tid++) {
        const PetscInt v_id = v_id_0 + tid, glb_v_id = global_vertex_id_0 + v_id;
        if (glb_v_id < num_vertices) {
          for (PetscInt grid = 0; grid < ctx->num_grids; grid++) { // add same particels for all grids
            PetscCall(PetscFree4(xx_t[grid][tid], yy_t[grid][tid], wp_t[grid][tid], zz_t[grid][tid]));
          }
        } // active
      } // threads
    } // (fake) particle loop
    // standard view of initial conditions
    if (v_target >= global_vertex_id_0 && v_target < global_vertex_id_0 + ctx->batch_sz) {
      PetscCall(DMSetOutputSequenceNumber(ctx->plex[g_target], 0, 0.0));
      PetscCall(VecViewFromOptions(globXArray[LAND_PACK_IDX(v_target % ctx->batch_sz, g_target)], NULL, "-ex30_vec_view"));
      if (ctx->num_grids > g_target + 1) {
        PetscCall(DMSetOutputSequenceNumber(ctx->plex[g_target + 1], 0, 0.0));
        PetscCall(VecViewFromOptions(globXArray[LAND_PACK_IDX(v_target % ctx->batch_sz, g_target + 1)], NULL, "-ex30_vec_view2"));
      }
      PetscCall(MatViewFromOptions(globMpArray[LAND_PACK_IDX(v_target % ctx->batch_sz, g_target)], NULL, "-ex30_mass_mat_view"));
      PetscCall(DMViewFromOptions(globSwarmArray[LAND_PACK_IDX(v_target % ctx->batch_sz, g_target)], NULL, "-ex30_sw_view"));
      PetscCall(DMSwarmViewXDMF(globSwarmArray[LAND_PACK_IDX(v_target % ctx->batch_sz, g_target)], "initial_swarm.xmf")); // writes a file by default!!!
    }
    // coarse graining moments_1a, bring f back from grid before advance
    if (v_target >= global_vertex_id_0 && v_target < global_vertex_id_0 + ctx->batch_sz && printCtx->print_entropy) {
      const PetscInt v_id = v_target % ctx->batch_sz;
      for (PetscInt grid = 0; grid < ctx->num_grids; grid++) {
        PetscDataType dtype;
        PetscReal    *wp, *coords;
        DM            sw = globSwarmArray[LAND_PACK_IDX(v_id, grid)];
        Vec           work, subX = globXArray[LAND_PACK_IDX(v_id, grid)];
        PetscInt      bs, NN;
        // C-G moments
        PetscCall(VecDuplicate(subX, &work));
        PetscCall(gridToParticles(grid_dm[grid], sw, subX, work, globMpArray[LAND_PACK_IDX(v_id, grid)], g_Mass[grid]));
        PetscCall(VecDestroy(&work));
        // moments
        PetscCall(DMSwarmGetField(sw, "DMSwarmPIC_coor", &bs, &dtype, (void **)&coords));
        PetscCall(DMSwarmGetLocalSize(sw, &NN));
        PetscCall(DMSwarmGetField(sw, "w_q", &bs, &dtype, (void **)&wp));
        for (PetscInt pp = 0; pp < NN; pp++) {
          PetscReal v2 = 0, fact = (dim == 2) ? 2.0 * PETSC_PI * coords[pp * dim + 0] : 1, w = fact * wp[pp] * ctx->n_0 * ctx->masses[ctx->species_offset[grid]];
          for (PetscInt i = 0; i < dim; ++i) v2 += PetscSqr(coords[pp * dim + i]);
          moments_1a[0] += w;
          moments_1a[1] += w * ctx->v_0 * coords[pp * dim + 1]; // z-momentum
          moments_1a[2] += w * 0.5 * ctx->v_0 * ctx->v_0 * v2;
        }
        PetscCall(DMSwarmRestoreField(sw, "DMSwarmPIC_coor", &bs, &dtype, (void **)&coords));
        PetscCall(DMSwarmRestoreField(sw, "w_q", &bs, &dtype, (void **)&wp));
      }
      // entropy
      const PetscReal N_inv = 1 / moments_1a[0];
      PetscCall(PetscInfo(pack, "Entropy batch %" PetscInt_FMT " of %" PetscInt_FMT ", n = %g\n", v_target, num_vertices, (double)(1 / N_inv)));
      for (PetscInt grid = 0; grid < ctx->num_grids; grid++) {
        PetscDataType dtype;
        PetscReal    *wp, *coords;
        DM            sw = globSwarmArray[LAND_PACK_IDX(v_id, grid)];
        PetscInt      bs, NN;
        PetscCall(DMSwarmGetLocalSize(sw, &NN));
        PetscCall(DMSwarmGetField(sw, "w_q", &bs, &dtype, (void **)&wp));
        PetscCall(DMSwarmGetField(sw, "DMSwarmPIC_coor", &bs, &dtype, (void **)&coords));
        for (PetscInt pp = 0; pp < NN; pp++) {
          PetscReal fact = (dim == 2) ? 2.0 * PETSC_PI * coords[pp * dim + 0] : 1, w = fact * wp[pp] * ctx->n_0 * ctx->masses[ctx->species_offset[grid]], ww = w * N_inv;
          if (w > PETSC_REAL_MIN) {
            moments_1a[3] -= ww * PetscLogReal(ww);
            PetscCheck(ww < 1 - PETSC_MACHINE_EPSILON, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "ww (%g) > 1", (double)ww);
          } else moments_1a[4] -= w;
        }
        PetscCall(DMSwarmRestoreField(sw, "w_q", &bs, &dtype, (void **)&wp));
        PetscCall(DMSwarmRestoreField(sw, "DMSwarmPIC_coor", &bs, &dtype, (void **)&coords));
      }
    }
    // restore vector
    PetscCall(DMCompositeRestoreAccessArray(pack, X, nDMs, NULL, globXArray));
    // view initial grid
    if (v_target >= global_vertex_id_0 && v_target < global_vertex_id_0 + ctx->batch_sz) PetscCall(DMPlexLandauPrintNorms(X, 0));
    // advance
    PetscCall(TSSetSolution(ts, X));
    PetscCall(PetscInfo(pack, "Advance vertex %" PetscInt_FMT " to %" PetscInt_FMT "\n", global_vertex_id_0, global_vertex_id_0 + ctx->batch_sz));
    PetscCall(TSSetPostStep(ts, PostStep));
    PetscCall(PostStep(ts));
    PetscCall(TSSolve(ts, X));
    // view
    PetscCall(DMCompositeGetAccessArray(pack, X, nDMs, NULL, globXArray));
    if (v_target >= global_vertex_id_0 && v_target < global_vertex_id_0 + ctx->batch_sz) {
      /* Visualize original particle field */
      DM  sw = globSwarmArray[LAND_PACK_IDX(v_target % ctx->batch_sz, g_target)];
      Vec f;
      PetscCall(DMSetOutputSequenceNumber(sw, 0, 0.0));
      PetscCall(DMViewFromOptions(grid_dm[g_target], NULL, "-weights_dm_view"));
      PetscCall(DMViewFromOptions(sw, NULL, "-weights_sw_view"));
      PetscCall(DMSwarmCreateGlobalVectorFromField(sw, "w_q", &f));
      PetscCall(PetscObjectSetName((PetscObject)f, "weights"));
      PetscCall(VecViewFromOptions(f, NULL, "-weights_vec_view"));
      PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, "w_q", &f));
      //
      PetscCall(DMPlexLandauPrintNorms(X, 1));
    }
    if (!use_uniform_particle_grid) { // resample to uniform grid
      for (PetscInt v_id_0 = 0; v_id_0 < ctx->batch_sz; v_id_0 += numthreads) {
        PetscReal *xx_t[LANDAU_MAX_GRIDS][EX30_MAX_NUM_THRDS], *yy_t[LANDAU_MAX_GRIDS][EX30_MAX_NUM_THRDS], *zz_t[LANDAU_MAX_GRIDS][EX30_MAX_NUM_THRDS], *wp_t[LANDAU_MAX_GRIDS][EX30_MAX_NUM_THRDS];
        PetscInt   Np_t[LANDAU_MAX_GRIDS][EX30_MAX_NUM_THRDS];
        for (PetscInt tid = 0; tid < numthreads; tid++) {
          const PetscInt v_id = v_id_0 + tid, glb_v_id = global_vertex_id_0 + v_id;
          if (glb_v_id < num_vertices) {
            // create uniform grid w/o weights & smaller
            PetscInt Npp0 = (a_Np + (glb_v_id % (a_Np / 10 + 1))) / 2, Nv; // 1/2 of uniform particle grid size
            for (PetscInt grid = 0; grid < ctx->num_grids; grid++) {
              // for (PetscInt sp = ctx->species_offset[grid], i0 = 0; sp < ctx->species_offset[grid + 1]; sp++, i0++)
              PetscReal lo[3] = {-ctx->radius[grid], -ctx->radius[grid], -ctx->radius[grid]}, hi[3] = {ctx->radius[grid], ctx->radius[grid], ctx->radius[grid]}, hp[3];
              PetscInt  Npi = Npp0, Npj = 2 * Npp0, Npk = 1, NN;
              // delete old particles and particle mass matrix
              PetscCall(DMDestroy(&globSwarmArray[LAND_PACK_IDX(v_id, grid)]));
              PetscCall(MatDestroy(&globMpArray[LAND_PACK_IDX(v_id, grid)]));
              // create fake particles in batches with threads
              PetscCall(MatGetLocalSize(g_Mass[grid], &Nv, NULL));
              if (dim == 2) lo[0] = 0;
              else Npi = Npj = Npk = Npp0;
              NN = Npi * Npj * Npk + (dim == 2 ? 3 : 6); // make a regular grid of particles Npp x Npp
              while (Npi * Npj * Npk < Nv) {             // make stable - no LS
                Npi++;
                Npj++;
                Npk++;
                NN = Npi * Npj * Npk + (dim == 2 ? 3 : 6);
              }
              Np_t[grid][tid] = NN;
              PetscCall(PetscMalloc4(NN, &xx_t[grid][tid], NN, &yy_t[grid][tid], NN, &wp_t[grid][tid], dim == 2 ? 1 : NN, &zz_t[grid][tid]));
              hp[0] = (hi[0] - lo[0]) / Npi;
              hp[1] = (hi[1] - lo[1]) / Npj;
              hp[2] = (hi[2] - lo[2]) / Npk;
              if (dim == 2) hp[2] = 1;
              PetscCall(PetscInfo(pack, "Resampling %d particles, %d vertices\n", (int)NN, (int)Nv)); // temp
              for (PetscInt pj = 0, pp = 0; pj < Npj; pj++) {
                for (PetscInt pk = 0; pk < Npk; pk++) {
                  for (PetscInt pi = 0; pi < Npi; pi++, pp++) {
                    wp_t[grid][tid][pp] = 0;
                    xx_t[grid][tid][pp] = lo[0] + hp[0] / 2.0 + pi * hp[0];
                    yy_t[grid][tid][pp] = lo[1] + hp[1] / 2.0 + pj * hp[1];
                    if (dim == 3) zz_t[grid][tid][pp] = lo[2] + hp[2] / 2.0 + pk * hp[2];
                  }
                }
              }
              if (dim == 2) { // fix bounding box
                PetscInt pp           = NN - 3;
                wp_t[grid][tid][pp]   = 0;
                xx_t[grid][tid][pp]   = 1.e-7;
                yy_t[grid][tid][pp++] = hi[1] - 5.e-7;
                wp_t[grid][tid][pp]   = 0;
                xx_t[grid][tid][pp]   = hi[0] - 5.e-7;
                yy_t[grid][tid][pp++] = 0;
                wp_t[grid][tid][pp]   = 0;
                xx_t[grid][tid][pp]   = 1.e-7;
                yy_t[grid][tid][pp++] = lo[1] + 5.e-7;
              } else {
                const PetscInt p0 = NN - 6;
                for (PetscInt pj = 0; pj < 6; pj++) xx_t[grid][tid][p0 + pj] = yy_t[grid][tid][p0 + pj] = zz_t[grid][tid][p0 + pj] = wp_t[grid][tid][p0 + pj] = 0;
                xx_t[grid][tid][p0 + 0] = lo[0];
                xx_t[grid][tid][p0 + 1] = hi[0];
                yy_t[grid][tid][p0 + 2] = lo[1];
                yy_t[grid][tid][p0 + 3] = hi[1];
                zz_t[grid][tid][p0 + 4] = lo[2];
                zz_t[grid][tid][p0 + 5] = hi[2];
              }
            }
          } // active
        } // threads
        /* Create particle swarm */
        for (PetscInt tid = 0; tid < numthreads; tid++) {
          const PetscInt v_id = v_id_0 + tid, glb_v_id = global_vertex_id_0 + v_id;
          if (glb_v_id < num_vertices) {                             // the ragged edge of the last batch
            for (PetscInt grid = 0; grid < ctx->num_grids; grid++) { // add same particels for all grids
              // for (PetscInt sp = ctx->species_offset[grid], i0 = 0; sp < ctx->species_offset[grid + 1]; sp++, i0++) -- loop over species for Nf > 1 -- TODO
              PetscErrorCode ierr_t;
              PetscSection   section;
              PetscInt       Nf;
              DM             dm = grid_dm[grid];
              ierr_t            = DMGetLocalSection(dm, &section);
              ierr_t            = PetscSectionGetNumFields(section, &Nf);
              if (Nf != 1) ierr_t = (PetscErrorCode)9999;
              else {
                ierr_t = DMViewFromOptions(dm, NULL, "-dm_view");
                ierr_t = PetscInfo(pack, "call createSwarm [%" PetscInt_FMT ".%" PetscInt_FMT "] local block index %" PetscInt_FMT "\n", v_id, grid, LAND_PACK_IDX(v_id, grid));
                ierr_t = createSwarm(dm, dim, &globSwarmArray[LAND_PACK_IDX(v_id, grid)]);
              }
              if (ierr_t) ierr = ierr_t;
            }
          } // active
        } // threads
        PetscCheck(ierr != 9999, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Only support one species per grid");
        PetscCheck(!ierr, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Error in OMP loop. ierr = %d", (int)ierr);
        // make globMpArray
        PetscPragmaOMP(parallel for)
        for (PetscInt tid = 0; tid < numthreads; tid++) {
          const PetscInt v_id = v_id_0 + tid, glb_v_id = global_vertex_id_0 + v_id;
          if (glb_v_id < num_vertices) {
            for (PetscInt grid = 0; grid < ctx->num_grids; grid++) { // add same particels for all grids
              // for (PetscInt sp = ctx->species_offset[grid], i0 = 0; sp < ctx->species_offset[grid + 1]; sp++, i0++) -- loop over species for Nf > 1 -- TODO
              PetscErrorCode ierr_t;
              DM             sw = globSwarmArray[LAND_PACK_IDX(v_id, grid)];
              ierr_t            = PetscInfo(pack, "makeSwarm %" PetscInt_FMT ".%" PetscInt_FMT ") for block %" PetscInt_FMT "\n", v_id, grid, LAND_PACK_IDX(v_id, grid));
              ierr_t            = makeSwarm(sw, dim, Np_t[grid][tid], xx_t[grid][tid], yy_t[grid][tid], zz_t[grid][tid]);
              if (ierr_t) ierr = ierr_t;
            }
          } // active
        } // threads
        // create particle mass matrices
        //PetscPragmaOMP(parallel for)
        for (PetscInt tid = 0; tid < numthreads; tid++) {
          const PetscInt v_id = v_id_0 + tid, glb_v_id = global_vertex_id_0 + v_id;
          if (glb_v_id < num_vertices) {
            for (PetscInt grid = 0; grid < ctx->num_grids; grid++) { // add same particels for all grids
              PetscErrorCode ierr_t;
              DM             dm = grid_dm[grid];
              DM             sw = globSwarmArray[LAND_PACK_IDX(v_id, grid)];
              ierr_t            = PetscInfo(pack, "createMp %" PetscInt_FMT ".%" PetscInt_FMT ") for block %" PetscInt_FMT "\n", v_id, grid, LAND_PACK_IDX(v_id, grid));
              ierr_t            = createMp(dm, sw, &globMpArray[LAND_PACK_IDX(v_id, grid)]);
              if (ierr_t) ierr = ierr_t;
            }
          } // active
        } // threads
        PetscCheck(!ierr, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Error in OMP loop. ierr = %d", (int)ierr);
        /* Cleanup */
        for (PetscInt tid = 0; tid < numthreads; tid++) {
          const PetscInt v_id = v_id_0 + tid, glb_v_id = global_vertex_id_0 + v_id;
          if (glb_v_id < num_vertices) {
            for (PetscInt grid = 0; grid < ctx->num_grids; grid++) { // add same particels for all grids
              PetscCall(PetscFree4(xx_t[grid][tid], yy_t[grid][tid], wp_t[grid][tid], zz_t[grid][tid]));
            }
          } // active
        } // threads
      } // batch
      // view
      if (v_target >= global_vertex_id_0 && v_target < global_vertex_id_0 + ctx->batch_sz) {
        /* Visualize particle field */
        DM  sw = globSwarmArray[LAND_PACK_IDX(v_target % ctx->batch_sz, g_target)];
        Vec f;
        PetscCall(DMSetOutputSequenceNumber(sw, 0, 0.0));
        PetscCall(DMViewFromOptions(sw, NULL, "-resampled_weights_sw_view"));
        PetscCall(DMSwarmCreateGlobalVectorFromField(sw, "w_q", &f));
        PetscCall(PetscObjectSetName((PetscObject)f, "resampled_weights"));
        PetscCall(VecViewFromOptions(f, NULL, "-resampled_weights_vec_view"));
        PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, "w_q", &f));
        PetscCall(DMSwarmViewXDMF(sw, "resampled.xmf"));
      }
    } // !uniform
    // particles to grid, compute moments and entropy, for target vertex only
    if (v_target >= global_vertex_id_0 && v_target < global_vertex_id_0 + ctx->batch_sz && printCtx->print_entropy) {
      PetscReal energy_error_rel;
      PetscCall(gridToParticles_private(grid_dm, globSwarmArray, dim, v_target, numthreads, num_vertices, global_vertex_id_0, globMpArray, g_Mass, t_fhat, moments_1b, globXArray, ctx));
      energy_error_rel = PetscAbsReal(moments_1b[2] - moments_0[2]) / moments_0[2];
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Particle Moments:\t number density      momentum (par)     energy             entropy            negative weights  : # OMP threads %g\n", (double)numthreads));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\tInitial:         %18.12e %19.12e %18.12e %18.12e %g %%\n", (double)moments_0[0], (double)moments_0[1], (double)moments_0[2], (double)moments_0[3], 100 * (double)(moments_0[4] / moments_0[0])));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\tCoarse-graining: %18.12e %19.12e %18.12e %18.12e %g %%\n", (double)moments_1a[0], (double)moments_1a[1], (double)moments_1a[2], (double)moments_1a[3], 100 * (double)(moments_1a[4] / moments_0[0])));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\tLandau:          %18.12e %19.12e %18.12e %18.12e %g %%\n", (double)moments_1b[0], (double)moments_1b[1], (double)moments_1b[2], (double)moments_1b[3], 100 * (double)(moments_1b[4] / moments_0[0])));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Coarse-graining entropy generation = %e ; Landau entropy generation = %e\n", (double)(moments_1a[3] - moments_0[3]), (double)(moments_1b[3] - moments_0[3])));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "(relative) energy conservation: Coarse-graining = %e, Landau = %e (%g %d)\n", (double)(moments_1a[2] - moments_0[2]) / (double)moments_0[2], (double)energy_error_rel, (double)PetscLog10Real(energy_error_rel), (int)(PetscLog10Real(energy_error_rel) + .5)));
    }
    // restore vector
    PetscCall(DMCompositeRestoreAccessArray(pack, X, nDMs, NULL, globXArray));
    // cleanup
    for (PetscInt v_id_0 = 0; v_id_0 < ctx->batch_sz; v_id_0 += numthreads) {
      for (PetscInt tid = 0; tid < numthreads; tid++) {
        const PetscInt v_id = v_id_0 + tid, glb_v_id = global_vertex_id_0 + v_id;
        if (glb_v_id < num_vertices) {
          for (PetscInt grid = 0; grid < ctx->num_grids; grid++) {
            PetscCall(DMDestroy(&globSwarmArray[LAND_PACK_IDX(v_id, grid)]));
            PetscCall(MatDestroy(&globMpArray[LAND_PACK_IDX(v_id, grid)]));
          }
        }
      }
    }
  } // user batch, not used
  /* Cleanup */
  PetscCall(PetscFree(globXArray));
  PetscCall(PetscFree(globSwarmArray));
  PetscCall(PetscFree(globMpArray));
  PetscCall(PetscFree(printCtx));
  // clean up mass matrices
  for (PetscInt grid = 0; grid < ctx->num_grids; grid++) { // add same particels for all grids
    PetscCall(MatDestroy(&g_Mass[grid]));
    for (PetscInt tid = 0; tid < numthreads; tid++) {
      PetscCall(VecDestroy(&t_fhat[grid][tid]));
      PetscCall(KSPDestroy(&t_ksp[grid][tid]));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM         pack;
  Vec        X;
  PetscInt   dim = 2, num_vertices = 1, Np = 10, v_target = 0, g_target = 0;
  TS         ts;
  Mat        J;
  LandauCtx *ctx;
  PetscReal  shift                     = 0;
  PetscBool  use_uniform_particle_grid = PETSC_TRUE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  // process args
  PetscOptionsBegin(PETSC_COMM_SELF, "", "Collision Options", "DMPLEX");
  PetscCall(PetscOptionsInt("-dim", "Velocity space dimension", "ex30.c", dim, &dim, NULL));
  PetscCall(PetscOptionsInt("-number_spatial_vertices", "Number of user spatial vertices to be batched for Landau", "ex30.c", num_vertices, &num_vertices, NULL));
  PetscCall(PetscOptionsInt("-number_particles_per_dimension", "Number of particles per grid, with slight modification per spatial vertex, in each dimension of base Cartesian grid", "ex30.c", Np, &Np, NULL));
  PetscCall(PetscOptionsBool("-use_uniform_particle_grid", "Use uniform particle grid", "ex30.c", use_uniform_particle_grid, &use_uniform_particle_grid, NULL));
  PetscCall(PetscOptionsInt("-vertex_view_target", "Global vertex for diagnostics", "ex30.c", v_target, &v_target, NULL));
  PetscCall(PetscOptionsReal("-e_shift", "Bi-Maxwellian shift", "ex30.c", shift, &shift, NULL));
  PetscCheck(v_target < num_vertices, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Batch to view %" PetscInt_FMT " should be < number of vertices %" PetscInt_FMT, v_target, num_vertices);
  PetscCall(PetscOptionsInt("-grid_view_target", "Grid to view with diagnostics", "ex30.c", g_target, &g_target, NULL));
  PetscOptionsEnd();
  /* Create a mesh */
  PetscCall(DMPlexLandauCreateVelocitySpace(PETSC_COMM_SELF, dim, "", &X, &J, &pack));
  PetscCall(DMGetApplicationContext(pack, &ctx));
  PetscCall(DMSetUp(pack));
  PetscCall(DMSetOutputSequenceNumber(pack, 0, 0.0));
  PetscCheck(g_target < ctx->num_grids, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Grid to view %" PetscInt_FMT " should be < number of grids %" PetscInt_FMT, g_target, ctx->num_grids);
  PetscCheck(ctx->batch_view_idx == v_target % ctx->batch_sz, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Global view index %" PetscInt_FMT " mode batch size %" PetscInt_FMT " != ctx->batch_view_idx %" PetscInt_FMT, v_target, ctx->batch_sz, ctx->batch_view_idx);
  // PetscCheck(!use_uniform_particle_grid || !ctx->sphere, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Can not use -use_uniform_particle_grid and -dm_landau_sphere");
  /* Create timestepping solver context */
  PetscCall(TSCreate(PETSC_COMM_SELF, &ts));
  PetscCall(TSSetDM(ts, pack));
  PetscCall(TSSetIFunction(ts, NULL, DMPlexLandauIFunction, NULL));
  PetscCall(TSSetIJacobian(ts, J, J, DMPlexLandauIJacobian, NULL));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER));
  PetscCall(TSSetFromOptions(ts));
  PetscCall(PetscObjectSetName((PetscObject)X, "X"));
  // do particle advance
  PetscCall(go(ts, X, num_vertices, Np, dim, v_target, g_target, shift, use_uniform_particle_grid));
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
    args: -dim 2 -petscspace_degree 3 -dm_landau_num_species_grid 1,1,1 -dm_refine 1 -number_particles_per_dimension 20 \
          -dm_landau_batch_size 4 -number_spatial_vertices 6 -vertex_view_target 5 -grid_view_target 1 -dm_landau_batch_view_idx 1 \
          -dm_landau_n 1.000018,1,1e-6 -dm_landau_thermal_temps 2,1,1 -dm_landau_ion_masses 2,180 -dm_landau_ion_charges 1,18 \
          -ftop_ksp_rtol 1e-10 -ftop_ksp_type lsqr -ftop_pc_type bjacobi -ftop_sub_pc_factor_shift_type nonzero -ftop_sub_pc_type lu -ftop_ksp_error_if_not_converged \
          -ksp_type gmres -ksp_error_if_not_converged -dm_landau_verbose 4 -print_entropy \
          -ptof_ksp_type cg -ptof_pc_type jacobi -ptof_ksp_rtol 1e-12 -ptof_ksp_error_if_not_converged\
          -snes_converged_reason -snes_monitor -snes_rtol 1e-12 -snes_stol 1e-12 \
          -ts_dt 0.01 -ts_rtol 1e-1 -ts_exact_final_time stepover -ts_max_snes_failures -1 -ts_max_steps 1 -ts_monitor -ts_type beuler
    test:
      suffix: cpu
      args: -dm_landau_device_type cpu -pc_type jacobi
    test:
      suffix: kokkos
      # failed on Sunspot@ALCF with sycl
      requires: kokkos_kernels !openmp !sycl
      args: -dm_landau_device_type kokkos -dm_mat_type aijkokkos -dm_vec_type kokkos -pc_type bjkokkos -pc_bjkokkos_ksp_type tfqmr -pc_bjkokkos_pc_type jacobi

  testset:
    requires: double !defined(PETSC_USE_DMLANDAU_2D)
    output_file: output/ex30_3d.out
    args: -dim 3 -petscspace_degree 2 -dm_landau_num_species_grid 1,1 -dm_refine 0 -number_particles_per_dimension 10 -dm_plex_hash_location \
          -dm_landau_batch_size 1 -number_spatial_vertices 1 -vertex_view_target 0 -grid_view_target 0 -dm_landau_batch_view_idx 0 \
          -dm_landau_n 1.000018,1 -dm_landau_thermal_temps 2,1 -dm_landau_ion_masses 2 -dm_landau_ion_charges 1 \
          -ftop_ksp_type cg -ftop_pc_type jacobi -ftop_ksp_rtol 1e-12 -ftop_ksp_error_if_not_converged -ksp_type preonly -pc_type lu -ksp_error_if_not_converged \
          -ptof_ksp_type cg -ptof_pc_type jacobi -ptof_ksp_rtol 1e-12 -ptof_ksp_error_if_not_converged \
          -snes_converged_reason -snes_monitor -snes_rtol 1e-12 -snes_stol 1e-12 \
          -ts_dt 0.1 -ts_exact_final_time stepover -ts_max_snes_failures -1 -ts_max_steps 1 -ts_monitor -ts_type beuler -print_entropy
    test:
      suffix: cpu_3d
      args: -dm_landau_device_type cpu
    test:
      suffix: kokkos_3d
      requires: kokkos_kernels !openmp
      args: -dm_landau_device_type kokkos -dm_mat_type aijkokkos -dm_vec_type kokkos -pc_type bjkokkos -pc_bjkokkos_ksp_type tfqmr -pc_bjkokkos_pc_type jacobi

  test:
    suffix: conserve
    requires: !complex double defined(PETSC_USE_DMLANDAU_2D) !cuda
    args: -dm_landau_batch_size 4 -dm_refine 0 -dm_landau_num_species_grid 1 -dm_landau_thermal_temps 1 -petscspace_degree 3 -snes_converged_reason -ts_type beuler -ts_dt .1 \
          -ts_max_steps 1 -ksp_type preonly -ksp_error_if_not_converged -snes_rtol 1e-14 -snes_stol 1e-14 -dm_landau_device_type cpu -number_particles_per_dimension 20 \
          -ptof_ksp_type cg -ptof_pc_type jacobi -ptof_ksp_rtol 1e-14 -ptof_ksp_error_if_not_converged -pc_type lu -dm_landau_simplex 1 -use_uniform_particle_grid false -dm_landau_sphere -print_entropy -number_particles_per_dimension 50 -ftop_ksp_type cg -ftop_pc_type jacobi -ftop_ksp_rtol 1e-14

TEST*/
