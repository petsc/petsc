static char help[] = "Grid based Landau collision operator with PIC interface with OpenMP setup. (one species per grid)\n";

/*
   Support 2.5V with axisymmetric coordinates
     - r,z coordinates
     - Domain and species data input by Landau operator
     - "radius" for each grid, normalized with electron thermal velocity
     - Domain: (0,radius) x (-radius,radius), thus first coordinate x[0] is perpendicular velocity and 2pi*x[0] term is added for axisymmetric
   Supports full 3V

 */

#include "petscdmplex.h"
#include "petscds.h"
#include "petscdmswarm.h"
#include "petscksp.h"
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

PetscErrorCode MatMultMtM_SeqAIJ(Mat MtM,Vec xx,Vec yy)
{
  MatShellCtx    *matshellctx;

  PetscFunctionBeginUser;
  PetscCall(MatShellGetContext(MtM,&matshellctx));
  PetscCheck(matshellctx,PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "No context");
  PetscCall(MatMult(matshellctx->Mp, xx, matshellctx->ff));
  PetscCall(MatMult(matshellctx->MpTrans, matshellctx->ff, yy));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultAddMtM_SeqAIJ(Mat MtM,Vec xx, Vec yy, Vec zz)
{
  MatShellCtx    *matshellctx;

  PetscFunctionBeginUser;
  PetscCall(MatShellGetContext(MtM,&matshellctx));
  PetscCheck(matshellctx,PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "No context");
  PetscCall(MatMult(matshellctx->Mp, xx, matshellctx->ff));
  PetscCall(MatMultAdd(matshellctx->MpTrans, matshellctx->ff, yy, zz));
  PetscFunctionReturn(0);
}

PetscErrorCode createSwarm(const DM dm, PetscInt dim, DM *sw)
{
  PetscInt       Nc = 1;

  PetscFunctionBeginUser;
  PetscCall(DMCreate(PETSC_COMM_SELF, sw));
  PetscCall(DMSetType(*sw, DMSWARM));
  PetscCall(DMSetDimension(*sw, dim));
  PetscCall(DMSwarmSetType(*sw, DMSWARM_PIC));
  PetscCall(DMSwarmSetCellDM(*sw, dm));
  PetscCall(DMSwarmRegisterPetscDatatypeField(*sw, "w_q", Nc, PETSC_SCALAR));
  PetscCall(DMSwarmFinalizeFieldRegister(*sw));
  PetscCall(DMSetFromOptions(*sw));
  PetscFunctionReturn(0);
}

PetscErrorCode gridToParticles(const DM dm, DM sw, Vec rhs, Vec work, Mat M_p, Mat Mass)
{
  PetscBool      is_lsqr;
  KSP            ksp;
  Mat            PM_p=NULL,MtM,D;
  Vec            ff;
  PetscInt       N, M, nzl;
  MatShellCtx    *matshellctx;

  PetscFunctionBeginUser;
  PetscCall(MatMult(Mass, rhs, work));
  PetscCall(VecCopy(work, rhs));
  // pseudo-inverse
  PetscCall(KSPCreate(PETSC_COMM_SELF, &ksp));
  PetscCall(KSPSetOptionsPrefix(ksp, "ftop_"));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(PetscObjectTypeCompare((PetscObject)ksp,KSPLSQR,&is_lsqr));
  if (!is_lsqr) {
    PetscCall(MatGetLocalSize(M_p, &M, &N));
    if (N>M) {
      PC        pc;
      PetscCall(PetscInfo(ksp, " M (%" PetscInt_FMT ") < M (%" PetscInt_FMT ") -- skip revert to lsqr\n",M,N));
      is_lsqr = PETSC_TRUE;
      PetscCall(KSPSetType(ksp,KSPLSQR));
      PetscCall(KSPGetPC(ksp,&pc));
      PetscCall(PCSetType(pc,PCNONE)); // could put in better solver -ftop_pc_type bjacobi -ftop_sub_pc_type lu -ftop_sub_pc_factor_shift_type nonzero
    } else {
      PetscCall(PetscNew(&matshellctx));
      PetscCall(MatCreateShell(PetscObjectComm((PetscObject)dm),N,N,PETSC_DECIDE,PETSC_DECIDE,matshellctx,&MtM));
      PetscCall(MatTranspose(M_p,MAT_INITIAL_MATRIX,&matshellctx->MpTrans));
      matshellctx->Mp = M_p;
      PetscCall(MatShellSetOperation(MtM, MATOP_MULT, (void (*)(void))MatMultMtM_SeqAIJ));
      PetscCall(MatShellSetOperation(MtM, MATOP_MULT_ADD, (void (*)(void))MatMultAddMtM_SeqAIJ));
      PetscCall(MatCreateVecs(M_p,&matshellctx->uu,&matshellctx->ff));
      PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF,N,N,1,NULL,&D));
      PetscCall(MatViewFromOptions(matshellctx->MpTrans,NULL,"-ftop2_Mp_mat_view"));
      for (int i=0 ; i<N ; i++) {
        const PetscScalar *vals;
        const PetscInt    *cols;
        PetscScalar dot = 0;
        PetscCall(MatGetRow(matshellctx->MpTrans,i,&nzl,&cols,&vals));
        for (int ii=0 ; ii<nzl ; ii++) dot += PetscSqr(vals[ii]);
        PetscCheck(dot!=0.0,PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Row %" PetscInt_FMT " is empty", i);
        PetscCall(MatSetValue(D,i,i,dot,INSERT_VALUES));
      }
      PetscCall(MatAssemblyBegin(D, MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(D, MAT_FINAL_ASSEMBLY));
      PetscCall(PetscInfo(M_p,"createMtMKSP Have %" PetscInt_FMT " eqs, nzl = %" PetscInt_FMT "\n",N,nzl));
      PetscCall(KSPSetOperators(ksp, MtM, D));
      PetscCall(MatViewFromOptions(D,NULL,"-ftop2_D_mat_view"));
      PetscCall(MatViewFromOptions(M_p,NULL,"-ftop2_Mp_mat_view"));
      PetscCall(MatViewFromOptions(matshellctx->MpTrans,NULL,"-ftop2_MpTranspose_mat_view"));
    }
  }
  if (is_lsqr) {
    PC        pc;
    PetscBool is_bjac;
    PetscCall(KSPGetPC(ksp,&pc));
    PetscCall(PetscObjectTypeCompare((PetscObject)pc,PCBJACOBI,&is_bjac));
    if (is_bjac) {
      PetscCall(DMSwarmCreateMassMatrixSquare(sw, dm, &PM_p));
      PetscCall(KSPSetOperators(ksp, M_p, PM_p));
    } else {
      PetscCall(KSPSetOperators(ksp, M_p, M_p));
    }
  }
  PetscCall(DMSwarmCreateGlobalVectorFromField(sw, "w_q", &ff)); // this grabs access
  if (!is_lsqr) {
    PetscCall(KSPSolve(ksp, rhs, matshellctx->uu));
    PetscCall(MatMult(M_p, matshellctx->uu, ff));
    PetscCall(MatDestroy(&matshellctx->MpTrans));
    PetscCall(VecDestroy(&matshellctx->ff));
    PetscCall(VecDestroy(&matshellctx->uu));
    PetscCall(MatDestroy(&D));
    PetscCall(MatDestroy(&MtM));
    PetscCall(PetscFree(matshellctx));
  } else {
    PetscCall(KSPSolveTranspose(ksp, rhs, ff));
  }
  PetscCall(KSPDestroy(&ksp));
  /* Visualize particle field */
  PetscCall(VecViewFromOptions(ff, NULL, "-weights_view"));
  PetscCall(MatDestroy(&PM_p));
  PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, "w_q", &ff));

  PetscFunctionReturn(0);
}

PetscErrorCode particlesToGrid(const DM dm, DM sw, const PetscInt Np, const PetscInt a_tid, const PetscInt dim,
                               const PetscReal xx[], const PetscReal yy[], const PetscReal zz[], const PetscReal a_wp[], Vec rho, Mat *Mp_out)
{

  PetscBool      removePoints = PETSC_TRUE;
  PetscReal      *wq, *coords;
  PetscDataType  dtype;
  Mat            M_p;
  Vec            ff;
  PetscInt       bs,p,zero=0;

  PetscFunctionBeginUser;
  PetscCall(DMSwarmSetLocalSizes(sw, Np, zero));
  PetscCall(DMSwarmGetField(sw, "w_q", &bs, &dtype, (void**)&wq));
  PetscCall(DMSwarmGetField(sw, "DMSwarmPIC_coor", &bs, &dtype, (void**)&coords));
  for (p=0;p<Np;p++) {
    coords[p*dim+0]  = xx[p];
    coords[p*dim+1]  = yy[p];
    wq[p]          = a_wp[p];
    if (dim==3) coords[p*dim+2]  = zz[p];
  }
  PetscCall(DMSwarmRestoreField(sw, "DMSwarmPIC_coor", &bs, &dtype, (void**)&coords));
  PetscCall(DMSwarmRestoreField(sw, "w_q", &bs, &dtype, (void**)&wq));
  PetscCall(DMSwarmMigrate(sw, removePoints));
  PetscCall(PetscObjectSetName((PetscObject)sw, "Particle Grid"));

  /* This gives M f = \int_\Omega \phi f, which looks like a rhs for a PDE */
  PetscCall(DMCreateMassMatrix(sw, dm, &M_p));

  PetscCall(PetscObjectSetName((PetscObject)rho, "rho"));
  PetscCall(DMSwarmCreateGlobalVectorFromField(sw, "w_q", &ff));
  PetscCall(PetscObjectSetName((PetscObject)ff, "weights"));
  PetscCall(MatMultTranspose(M_p, ff, rho));
  PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, "w_q", &ff));

  // output
  *Mp_out = M_p;

  PetscFunctionReturn(0);
}
static void maxwellian(PetscInt dim, const PetscReal x[], PetscReal kt_m, PetscReal n, PetscScalar *u)
{
  PetscInt      i;
  PetscReal     v2 = 0, theta = 2.0*kt_m; /* theta = 2kT/mc^2 */

  /* compute the exponents, v^2 */
  for (i = 0; i < dim; ++i) v2 += x[i]*x[i];
  /* evaluate the Maxwellian */
  u[0] = n*PetscPowReal(PETSC_PI*theta,-1.5)*(PetscExpReal(-v2/theta));
}

#define MAX_NUM_THRDS 12
PetscErrorCode go(TS ts, Vec X, const PetscInt NUserV, const PetscInt a_Np, const PetscInt dim, const PetscInt b_target, const PetscInt g_target)
{
  DM              pack, *globSwarmArray, grid_dm[LANDAU_MAX_GRIDS];
  Mat             *globMpArray, g_Mass[LANDAU_MAX_GRIDS];
  KSP             t_ksp[LANDAU_MAX_GRIDS][MAX_NUM_THRDS];
  Vec             t_fhat[LANDAU_MAX_GRIDS][MAX_NUM_THRDS];
  PetscInt        nDMs, glb_b_id, nTargetP=0;
  PetscErrorCode  ierr = 0;
#if defined(PETSC_HAVE_OPENMP) && defined(PETSC_HAVE_THREADSAFETY)
  PetscInt        numthreads = PetscNumOMPThreads;
#else
  PetscInt        numthreads = 1;
#endif
  LandauCtx      *ctx;
  Vec            *globXArray;
  PetscReal       moments_0[3], moments_1[3], dt_init;

  PetscFunctionBeginUser;
  PetscCheck(numthreads<=MAX_NUM_THRDS,PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Too many threads %" PetscInt_FMT " > %" PetscInt_FMT "", numthreads, MAX_NUM_THRDS);
  PetscCheck(numthreads>0,PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Number threads %" PetscInt_FMT " > %" PetscInt_FMT " ", numthreads,  MAX_NUM_THRDS);
  PetscCall(TSGetDM(ts,&pack));
  PetscCall(DMGetApplicationContext(pack, &ctx));
  PetscCheck(ctx->batch_sz%numthreads==0,PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "batch size (-dm_landau_batch_size) %" PetscInt_FMT "  mod #threads %" PetscInt_FMT " must equal zero", ctx->batch_sz, numthreads);
  PetscCall(DMCompositeGetNumberDM(pack,&nDMs));
  PetscCall(PetscInfo(pack,"Have %" PetscInt_FMT " total grids, with %" PetscInt_FMT " Landau local batched and %" PetscInt_FMT " global items (vertices)\n",ctx->num_grids,ctx->batch_sz,NUserV));
  PetscCall(PetscMalloc(sizeof(*globXArray)*nDMs, &globXArray));
  PetscCall(PetscMalloc(sizeof(*globMpArray)*nDMs, &globMpArray));
  PetscCall(PetscMalloc(sizeof(*globSwarmArray)*nDMs, &globSwarmArray));
  PetscCall(DMViewFromOptions(ctx->plex[g_target],NULL,"-ex30_dm_view"));
  // create mass matrices
  PetscCall(DMCompositeGetAccessArray(pack, X, nDMs, NULL, globXArray)); // just to duplicate
  for (PetscInt grid=0 ; grid<ctx->num_grids ; grid++) { // add same particels for all grids
    Vec  subX = globXArray[LAND_PACK_IDX(0,grid)];
    DM   dm = ctx->plex[grid];
    PetscSection s;
    grid_dm[grid] = dm;
    PetscCall(DMCreateMassMatrix(dm,dm, &g_Mass[grid]));
    //
    PetscCall(DMGetLocalSection(dm, &s));
    PetscCall(DMPlexCreateClosureIndex(dm, s));
    for (int tid=0; tid<numthreads; tid++) {
      PetscCall(VecDuplicate(subX,&t_fhat[grid][tid]));
      PetscCall(KSPCreate(PETSC_COMM_SELF, &t_ksp[grid][tid]));
      PetscCall(KSPSetOptionsPrefix(t_ksp[grid][tid], "ptof_"));
      PetscCall(KSPSetOperators(t_ksp[grid][tid], g_Mass[grid], g_Mass[grid]));
      PetscCall(KSPSetFromOptions(t_ksp[grid][tid]));
    }
  }
  PetscCall(DMCompositeRestoreAccessArray(pack, X, nDMs, NULL, globXArray));
  // create particle raw data. could use OMP with a thread safe malloc, but this is just the fake user
  for (int i=0;i<3;i++) moments_0[i] = moments_1[i] = 0;
  PetscCall(TSGetTimeStep(ts,&dt_init)); // we could have an adaptive time stepper
  for (PetscInt global_batch_id=0 ; global_batch_id < NUserV ; global_batch_id += ctx->batch_sz) {
    ierr = TSSetTime(ts, 0);CHKERRQ(ierr);
    ierr = TSSetStepNumber(ts, 0);CHKERRQ(ierr);
    ierr = TSSetTimeStep(ts, dt_init);CHKERRQ(ierr);
    PetscCall(VecZeroEntries(X));
    PetscCall(DMCompositeGetAccessArray(pack, X, nDMs, NULL, globXArray));
    if (b_target >= global_batch_id && b_target < global_batch_id+ctx->batch_sz) {
      PetscCall(PetscObjectSetName((PetscObject)globXArray[LAND_PACK_IDX(b_target%ctx->batch_sz,g_target)], "rho"));
    }
    // create fake particles
    for (PetscInt b_id_0 = 0 ; b_id_0 < ctx->batch_sz ; b_id_0 += numthreads) {
      PetscReal *xx_t[LANDAU_MAX_GRIDS][MAX_NUM_THRDS], *yy_t[LANDAU_MAX_GRIDS][MAX_NUM_THRDS], *zz_t[LANDAU_MAX_GRIDS][MAX_NUM_THRDS], *wp_t[LANDAU_MAX_GRIDS][MAX_NUM_THRDS];
      PetscInt  Np_t[LANDAU_MAX_GRIDS][MAX_NUM_THRDS];
      // make particles
      for (int tid=0; tid<numthreads; tid++) {
        const PetscInt b_id = b_id_0 + tid;
        if ((glb_b_id = global_batch_id + b_id) < NUserV) { // the ragged edge of the last batch
          PetscInt Npp0 = a_Np + (glb_b_id%a_Np), NN; // fake user: number of particels in each dimension with add some load imbalance and diff (<2x)
          for (PetscInt grid=0 ; grid<ctx->num_grids ; grid++) { // add same particels for all grids
            const PetscReal kT_m = ctx->k*ctx->thermal_temps[ctx->species_offset[grid]]/ctx->masses[ctx->species_offset[grid]]/(ctx->v_0*ctx->v_0); /* theta = 2kT/mc^2 per species -- TODO */;
            PetscReal       lo[3] = {-ctx->radius[grid],-ctx->radius[grid],-ctx->radius[grid]}, hi[3] = {ctx->radius[grid],ctx->radius[grid],ctx->radius[grid]}, hp[3], vole; // would be nice to get box from DM
            PetscInt  Npi=Npp0,Npj=2*Npp0,Npk=1;
            if (dim==2) lo[0] = 0; // Landau coordinate (r,z)
            else Npi = Npj = Npk = Npp0;
            // User: use glb_b_id to index into your data
            NN = Npi*Npj*Npk; // make a regular grid of particles Npp x Npp
            if (glb_b_id==b_target) {
              nTargetP = NN;
              PetscCall(PetscInfo(pack,"Target %" PetscInt_FMT " with %" PetscInt_FMT " particels\n",glb_b_id,NN));
            }
            Np_t[grid][tid] = NN;
            PetscCall(PetscMalloc4(NN,&xx_t[grid][tid],NN,&yy_t[grid][tid],NN,&wp_t[grid][tid], dim==2 ? 1 : NN, &zz_t[grid][tid]));
            hp[0] = (hi[0] - lo[0])/Npi;
            hp[1] = (hi[1] - lo[1])/Npj;
            hp[2] = (hi[2] - lo[2])/Npk;
            if (dim==2) hp[2] = 1;
            PetscCall(PetscInfo(pack," lo = %14.7e, hi = %14.7e; hp = %14.7e, %14.7e; kT_m = %g; \n",lo[1], hi[1], hp[0], hp[1], kT_m)); // temp
            vole = hp[0]*hp[1]*hp[2]*ctx->n[grid]; // fix for multi-species
            PetscCall(PetscInfo(pack,"Vertex %" PetscInt_FMT ", grid %" PetscInt_FMT " with %" PetscInt_FMT " particles (diagnostic target = %" PetscInt_FMT ")\n",glb_b_id,grid,NN,b_target));
            for (int pj=0, pp=0 ; pj < Npj ; pj++) {
              for (int pk=0 ; pk < Npk ; pk++) {
                for (int pi=0 ; pi < Npi ; pi++, pp++) {
                  xx_t[grid][tid][pp] = lo[0] + hp[0]/2.0 + pi*hp[0];
                  yy_t[grid][tid][pp] = lo[1] + hp[1]/2.0 + pj*hp[1];
                  if (dim==3) zz_t[grid][tid][pp] = lo[2] + hp[2]/2.0 + pk*hp[2];
                  {
                    PetscReal x[] = {xx_t[grid][tid][pp], yy_t[grid][tid][pp], dim==2 ? 0 : zz_t[grid][tid][pp]};
                    maxwellian(dim, x, kT_m, vole, &wp_t[grid][tid][pp]);
                    //PetscCall(PetscInfo(pack,"%" PetscInt_FMT ") x = %14.7e, %14.7e, %14.7e, n = %14.7e, w = %14.7e\n", pp, x[0], x[1], dim==2 ? 0 : x[2], ctx->n[grid], wp_t[grid][tid][pp])); // temp
                    if (glb_b_id==b_target) {
                      PetscReal v2=0, fact = dim==2 ? 2.0*PETSC_PI*x[0] : 1;
                      for (int i = 0; i < dim; ++i) v2 += PetscSqr(x[i]);
                      moments_0[0] += fact*wp_t[grid][tid][pp]*ctx->n_0                  *ctx->masses[ctx->species_offset[grid]];
                      moments_0[1] += fact*wp_t[grid][tid][pp]*ctx->n_0*ctx->v_0         *ctx->masses[ctx->species_offset[grid]] * x[1]; // z-momentum
                      moments_0[2] += fact*wp_t[grid][tid][pp]*ctx->n_0*ctx->v_0*ctx->v_0*ctx->masses[ctx->species_offset[grid]] * v2;
                    }
                  }
                }
              }
            }
          } // grid
        } // active
      } // fake threads
      /* Create particle swarm */
      PetscPragmaOMP(parallel for)
        for (int tid=0; tid<numthreads; tid++) {
          const PetscInt b_id = b_id_0 + tid;
          if ((glb_b_id = global_batch_id + b_id) < NUserV) { // the ragged edge of the last batch
            //PetscCall(PetscInfo(pack,"Create swarms for 'glob' index %" PetscInt_FMT " create swarm\n",glb_b_id));
            for (PetscInt grid=0 ; grid<ctx->num_grids ; grid++) { // add same particels for all grids
              PetscErrorCode  ierr_t;
              PetscSection    section;
              PetscInt        Nf;
              DM              dm = grid_dm[grid];
              ierr_t = DMGetLocalSection(dm, &section);
              ierr_t = PetscSectionGetNumFields(section, &Nf);
              if (Nf != 1) ierr_t = 9999;
              else {
                ierr_t = DMViewFromOptions(dm,NULL,"-dm_view");
                ierr_t = PetscInfo(pack,"call createSwarm [%" PetscInt_FMT ".%" PetscInt_FMT "] local batch index %" PetscInt_FMT "\n",b_id,grid, LAND_PACK_IDX(b_id,grid));
                ierr_t = createSwarm(dm, dim, &globSwarmArray[LAND_PACK_IDX(b_id,grid)]);
              }
              if (ierr_t) ierr = ierr_t;
            }
          } // active
        }
      PetscCheckFalse(ierr == 9999, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Only support one species per grid");
      PetscCall(ierr);
      // p --> g: make globMpArray & set X
      PetscPragmaOMP(parallel for)
        for (int tid=0; tid<numthreads; tid++) {
          const PetscInt b_id = b_id_0 + tid;
          if ((glb_b_id = global_batch_id + b_id) < NUserV) {
            for (PetscInt grid=0 ; grid<ctx->num_grids ; grid++) { // add same particels for all grids
              PetscErrorCode ierr_t;
              DM             dm = grid_dm[grid];
              DM             sw = globSwarmArray[LAND_PACK_IDX(b_id,grid)];
              Vec            subX = globXArray[LAND_PACK_IDX(b_id,grid)], work = t_fhat[grid][tid];
              PetscInfo(pack,"particlesToGrid %" PetscInt_FMT ".%" PetscInt_FMT ") particlesToGrid for local batch %" PetscInt_FMT "\n",global_batch_id,grid,LAND_PACK_IDX(b_id,grid));
              ierr_t = particlesToGrid(dm, sw, Np_t[grid][tid], tid, dim, xx_t[grid][tid], yy_t[grid][tid], zz_t[grid][tid], wp_t[grid][tid], subX, &globMpArray[LAND_PACK_IDX(b_id,grid)]);
              if (ierr_t) ierr = ierr_t;
              // u = M^_1 f_w
              ierr_t = VecCopy(subX, work);
              ierr_t = KSPSolve(t_ksp[grid][tid], work, subX);
              if (ierr_t) ierr = ierr_t;
            }
          }
        }
      PetscCall(ierr);
      /* Cleanup */
      for (int tid=0; tid<numthreads; tid++) {
        const PetscInt b_id = b_id_0 + tid;
        if ((glb_b_id = global_batch_id + b_id) < NUserV) {
          PetscCall(PetscInfo(pack,"Free for global batch %" PetscInt_FMT " of %" PetscInt_FMT "\n",glb_b_id+1,NUserV));
          for (PetscInt grid=0 ; grid<ctx->num_grids ; grid++) { // add same particels for all grids
            PetscCall(PetscFree4(xx_t[grid][tid],yy_t[grid][tid],wp_t[grid][tid],zz_t[grid][tid]));
          }
        } // active
      }
    } // Landau
    if (b_target >= global_batch_id && b_target < global_batch_id+ctx->batch_sz) {
      PetscCall(VecViewFromOptions(globXArray[LAND_PACK_IDX(b_target%ctx->batch_sz,g_target)],NULL,"-ex30_vec_view"));
    }
    PetscCall(DMCompositeRestoreAccessArray(pack, X, nDMs, NULL, globXArray));
    PetscCall(DMPlexLandauPrintNorms(X,0));
    // advance
    PetscCall(TSSetSolution(ts,X));
    PetscCall(PetscInfo(pack,"Advance vertex %" PetscInt_FMT " to %" PetscInt_FMT " (with padding)\n",global_batch_id, global_batch_id+ctx->batch_sz));
    PetscCall(TSSolve(ts,X));
    PetscCall(DMPlexLandauPrintNorms(X,1));
    PetscCall(DMCompositeGetAccessArray(pack, X, nDMs, NULL, globXArray));
    // map back to particles
    for (PetscInt b_id_0 = 0 ; b_id_0 < ctx->batch_sz ; b_id_0 += numthreads) {
      PetscCall(PetscInfo(pack,"g2p: global batch %" PetscInt_FMT " of %" PetscInt_FMT ", Landau batch %" PetscInt_FMT " of %" PetscInt_FMT ": map back to particles\n",global_batch_id+1,NUserV,b_id_0+1,ctx->batch_sz));
      PetscPragmaOMP(parallel for)
        for (int tid=0; tid<numthreads; tid++) {
          const PetscInt b_id = b_id_0 + tid;
          if ((glb_b_id = global_batch_id + b_id) < NUserV) {
            for (PetscInt grid=0 ; grid<ctx->num_grids ; grid++) { // add same particels for all grids
              PetscErrorCode  ierr_t;
              PetscInfo(pack,"gridToParticles: global batch %" PetscInt_FMT ", local batch b=%" PetscInt_FMT ", grid g=%" PetscInt_FMT ", index(b,g) %" PetscInt_FMT "\n",global_batch_id,b_id,grid,LAND_PACK_IDX(b_id,grid));
              ierr_t = gridToParticles(grid_dm[grid], globSwarmArray[LAND_PACK_IDX(b_id,grid)], globXArray[LAND_PACK_IDX(b_id,grid)], t_fhat[grid][tid], globMpArray[LAND_PACK_IDX(b_id,grid)], g_Mass[grid]);
              if (ierr_t) ierr = ierr_t;
            }
          }
        }
      PetscCall(ierr);
      /* Cleanup, and get data */
      PetscCall(PetscInfo(pack,"Cleanup batches %" PetscInt_FMT " to %" PetscInt_FMT "\n",b_id_0,b_id_0+numthreads));
      for (int tid=0; tid<numthreads; tid++) {
        const PetscInt b_id = b_id_0 + tid;
        if ((glb_b_id = global_batch_id + b_id) < NUserV) {
          for (PetscInt grid=0 ; grid<ctx->num_grids ; grid++) {
            PetscDataType dtype;
            PetscReal     *wp,*coords;
            DM            sw = globSwarmArray[LAND_PACK_IDX(b_id,grid)];
            PetscInt      npoints,bs=1;
            PetscCall(DMSwarmGetField(sw, "w_q", &bs, &dtype, (void**)&wp)); // take data out here
            if (glb_b_id==b_target) {
              PetscCall(DMSwarmGetField(sw, "DMSwarmPIC_coor", &bs, &dtype, (void**)&coords));
              PetscCall(DMSwarmGetLocalSize(sw,&npoints));
              for (int p=0;p<npoints;p++) {
              PetscReal v2 = 0, fact = dim==2 ? 2.0*PETSC_PI*coords[p*dim+0] : 1;
              for (int i = 0; i < dim; ++i) v2 += PetscSqr(coords[p*dim+i]);
              moments_1[0] += fact*wp[p]*ctx->n_0                  *ctx->masses[ctx->species_offset[grid]];
              moments_1[1] += fact*wp[p]*ctx->n_0*ctx->v_0         *ctx->masses[ctx->species_offset[grid]] * coords[p*dim+1]; // z-momentum
              moments_1[2] += fact*wp[p]*ctx->n_0*ctx->v_0*ctx->v_0*ctx->masses[ctx->species_offset[grid]] * v2;
              }
              PetscCall(DMSwarmRestoreField(sw, "DMSwarmPIC_coor", &bs, &dtype, (void**)&coords));
            }
            PetscCall(DMSwarmRestoreField(sw, "w_q", &bs, &dtype, (void**)&wp));
            PetscCall(DMDestroy(&globSwarmArray[LAND_PACK_IDX(b_id,grid)]));
            PetscCall(MatDestroy(&globMpArray[LAND_PACK_IDX(b_id,grid)]));
          }
        }
      }
    } // thread batch
    PetscCall(DMCompositeRestoreAccessArray(pack, X, nDMs, NULL, globXArray));
  } // user batch
  /* Cleanup */
  PetscCall(PetscFree(globXArray));
  PetscCall(PetscFree(globSwarmArray));
  PetscCall(PetscFree(globMpArray));
  // clean up mass matrices
  for (PetscInt grid=0 ; grid<ctx->num_grids ; grid++) { // add same particels for all grids
    PetscCall(MatDestroy(&g_Mass[grid]));
    for (int tid=0; tid<numthreads; tid++) {
      PetscCall(VecDestroy(&t_fhat[grid][tid]));
      PetscCall(KSPDestroy(&t_ksp[grid][tid]));
    }
  }
  PetscCall(PetscInfo(X,"Total number density: %20.12e (%20.12e); x-momentum = %20.12e (%20.12e); energy = %20.12e (%20.12e) error = %e (log10 of error = %" PetscInt_FMT "), %" PetscInt_FMT " particles. Use %" PetscInt_FMT " threads\n",
                      moments_1[0], moments_0[0], moments_1[1], moments_0[1], moments_1[2],  moments_0[2], (moments_1[2]-moments_0[2])/moments_0[2], (PetscInt)PetscLog10Real(PetscAbsReal((moments_1[2]-moments_0[2])/moments_0[2])), nTargetP, numthreads));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM               pack;
  Vec              X;
  PetscInt         dim=2,nvert=1,Np=10,btarget=0,gtarget=0;
  TS               ts;
  Mat              J;
  LandauCtx        *ctx;
  PetscErrorCode   ierr;
#if defined(PETSC_USE_LOG)
  PetscLogStage    stage;
#endif

  PetscCall(PetscInitialize(&argc, &argv, NULL,help));
  // process args
  ierr = PetscOptionsBegin(PETSC_COMM_SELF, "", "Collision Options", "DMPLEX");PetscCall(ierr);
  PetscCall(PetscOptionsInt("-number_spatial_vertices", "Number of user spatial vertices to be batched for Landau", "ex30.c", nvert, &nvert, NULL));
  PetscCall(PetscOptionsInt("-dim", "Velocity space dimension", "ex30.c", dim, &dim, NULL));
  PetscCall(PetscOptionsInt("-number_particles_per_dimension", "Number of particles per grid, with slight modification per spatial vertex, in each dimension of base Cartesian grid", "ex30.c", Np, &Np, NULL));
  PetscCall(PetscOptionsInt("-view_vertex_target", "Batch to view with diagnostics", "ex30.c", btarget, &btarget, NULL));
  PetscCheck(btarget < nvert, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Batch to view %" PetscInt_FMT " should be < number of vertices %" PetscInt_FMT,btarget,nvert);
  PetscCall(PetscOptionsInt("-view_grid_target", "Grid to view with diagnostics", "ex30.c", gtarget, &gtarget, NULL));
  ierr = PetscOptionsEnd();PetscCall(ierr);
  /* Create a mesh */
  PetscCall(DMPlexLandauCreateVelocitySpace(PETSC_COMM_SELF, dim, "", &X, &J, &pack));
  PetscCall(DMSetUp(pack));
  PetscCall(DMSetOutputSequenceNumber(pack, 0, 0.0));
  PetscCall(DMGetApplicationContext(pack, &ctx));
  PetscCheck(gtarget < ctx->num_grids, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Grid to view %" PetscInt_FMT " should be < number of grids %" PetscInt_FMT,gtarget,ctx->num_grids);
  PetscCheck(nvert >= ctx->batch_sz, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Number of vertices %" PetscInt_FMT " should be <= batch size %" PetscInt_FMT,nvert,ctx->batch_sz);
  /* Create timestepping solver context */
  PetscCall(TSCreate(PETSC_COMM_SELF,&ts));
  PetscCall(TSSetDM(ts,pack));
  PetscCall(TSSetIFunction(ts,NULL,DMPlexLandauIFunction,NULL));
  PetscCall(TSSetIJacobian(ts,J,J,DMPlexLandauIJacobian,NULL));
  PetscCall(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER));
  PetscCall(TSSetFromOptions(ts));
  PetscCall(PetscObjectSetName((PetscObject)X, "X"));
  // do particle advance, warmup
  PetscCall(go(ts,X,nvert,Np,dim,btarget,gtarget));
  PetscCall(MatZeroEntries(J)); // need to zero out so as to not reuse it in Landau's logic
  // hot
  PetscCall(PetscLogStageRegister("ex30 hot solve", &stage));
  PetscCall(PetscLogStagePush(stage));
  PetscCall(go(ts,X,nvert,Np,dim,btarget,gtarget));
  PetscCall(PetscLogStagePop());
  /* clean up */
  PetscCall(DMPlexLandauDestroyVelocitySpace(&pack));
  PetscCall(TSDestroy(&ts));
  PetscCall(VecDestroy(&X));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  build:
    requires: !complex p4est

  testset:
    requires: double defined(PETSC_USE_DMLANDAU_2D)
    output_file: output/ex30_0.out
    args: -dim 2 -petscspace_degree 3 -dm_landau_type p4est -dm_landau_num_species_grid 1,1,1 -dm_landau_amr_levels_max 0,0,0 \
          -dm_landau_amr_post_refine 1 -number_particles_per_dimension 10 -dm_plex_hash_location \
          -dm_landau_batch_size 2 -number_spatial_vertices 3 -dm_landau_batch_view_idx 1 -view_vertex_target 2 -view_grid_target 1 \
          -dm_landau_n 1.000018,1,1e-6 -dm_landau_thermal_temps 2,1,1 -dm_landau_ion_masses 2,180 -dm_landau_ion_charges 1,18 \
          -ftop_ksp_converged_reason -ftop_ksp_rtol 1e-10 -ftop_ksp_type lsqr -ftop_pc_type bjacobi -ftop_sub_pc_factor_shift_type nonzero -ftop_sub_pc_type lu \
          -ksp_type preonly -pc_type lu \
          -ptof_ksp_type cg -ptof_pc_type jacobi -ptof_ksp_converged_reason -ptof_ksp_rtol 1e-12\
          -snes_converged_reason -snes_monitor -snes_rtol 1e-14 -snes_stol 1e-14\
          -ts_dt 0.01 -ts_rtol 1e-1 -ts_exact_final_time stepover -ts_max_snes_failures -1 -ts_max_steps 1 -ts_monitor -ts_type beuler -info :vec

    test:
      suffix: cpu
      args: -dm_landau_device_type cpu
    test:
      suffix: kokkos
      requires: kokkos_kernels
      args: -dm_landau_device_type kokkos -dm_mat_type aijkokkos -dm_vec_type kokkos
    test:
      suffix: cuda
      requires: cuda
      args: -dm_landau_device_type cuda -dm_mat_type aijcusparse -dm_vec_type cuda

  testset:
    requires: double !defined(PETSC_USE_DMLANDAU_2D)
    output_file: output/ex30_3d.out
    args: -dim 3 -petscspace_degree 2 -dm_landau_type p8est -dm_landau_num_species_grid 1,1,1 -dm_landau_amr_levels_max 0,0,0 \
          -dm_landau_amr_post_refine 0 -number_particles_per_dimension 5 -dm_plex_hash_location \
          -dm_landau_batch_size 1 -number_spatial_vertices 1 -dm_landau_batch_view_idx 0 -view_vertex_target 0 -view_grid_target 0 \
          -dm_landau_n 1.000018,1,1e-6 -dm_landau_thermal_temps 2,1,1 -dm_landau_ion_masses 2,180 -dm_landau_ion_charges 1,18 \
          -ftop_ksp_converged_reason -ftop_ksp_rtol 1e-12 -ftop_ksp_type cg -ftop_pc_type jacobi \
          -ksp_type preonly -pc_type lu \
          -ptof_ksp_type cg -ptof_pc_type jacobi -ptof_ksp_converged_reason -ptof_ksp_rtol 1e-12\
          -snes_converged_reason -snes_monitor -snes_rtol 1e-12 -snes_stol 1e-12\
          -ts_dt 0.1 -ts_exact_final_time stepover -ts_max_snes_failures -1 -ts_max_steps 1 -ts_monitor -ts_type beuler -info :vec

    test:
      suffix: cpu_3d
      args: -dm_landau_device_type cpu
    test:
      suffix: kokkos_3d
      requires: kokkos_kernels
      args: -dm_landau_device_type kokkos -dm_mat_type aijkokkos -dm_vec_type kokkos
    test:
      suffix: cuda_3d
      requires: cuda
      args: -dm_landau_device_type cuda -dm_mat_type aijcusparse -dm_vec_type cuda

TEST*/
