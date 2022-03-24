static char help[] = "Example program demonstrating projection between particle and finite element spaces using OpenMP in 2D cylindrical coordinates\n";

#include "petscdmplex.h"
#include "petscds.h"
#include "petscdmswarm.h"
#include "petscksp.h"
#include <petsc/private/petscimpl.h>
#if defined(PETSC_HAVE_OPENMP) && defined(PETSC_HAVE_THREADSAFETY)
#include <omp.h>
#endif

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
  CHKERRQ(MatShellGetContext(MtM,&matshellctx));
  PetscCheck(matshellctx,PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "No context");
  CHKERRQ(MatMult(matshellctx->Mp, xx, matshellctx->ff));
  CHKERRQ(MatMult(matshellctx->MpTrans, matshellctx->ff, yy));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultAddMtM_SeqAIJ(Mat MtM,Vec xx, Vec yy, Vec zz)
{
  MatShellCtx    *matshellctx;

  PetscFunctionBeginUser;
  CHKERRQ(MatShellGetContext(MtM,&matshellctx));
  PetscCheck(matshellctx,PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "No context");
  CHKERRQ(MatMult(matshellctx->Mp, xx, matshellctx->ff));
  CHKERRQ(MatMultAdd(matshellctx->MpTrans, matshellctx->ff, yy, zz));
  PetscFunctionReturn(0);
}

PetscErrorCode createSwarm(const DM dm, DM *sw)
{
  PetscInt Nc = 1, dim = 2;

  PetscFunctionBeginUser;
  CHKERRQ(DMCreate(PETSC_COMM_SELF, sw));
  CHKERRQ(DMSetType(*sw, DMSWARM));
  CHKERRQ(DMSetDimension(*sw, dim));
  CHKERRQ(DMSwarmSetType(*sw, DMSWARM_PIC));
  CHKERRQ(DMSwarmSetCellDM(*sw, dm));
  CHKERRQ(DMSwarmRegisterPetscDatatypeField(*sw, "w_q", Nc, PETSC_SCALAR));
  CHKERRQ(DMSwarmFinalizeFieldRegister(*sw));
  CHKERRQ(DMSetFromOptions(*sw));
  PetscFunctionReturn(0);
}

PetscErrorCode gridToParticles(const DM dm, DM sw, PetscReal *moments, Vec rhs, Mat M_p)
{
  PetscBool      is_lsqr;
  KSP            ksp;
  Mat            PM_p=NULL,MtM,D;
  Vec            ff;
  PetscInt       Np, timestep = 0, bs, N, M, nzl;
  PetscReal      time = 0.0;
  PetscDataType  dtype;
  MatShellCtx    *matshellctx;

  PetscFunctionBeginUser;
  CHKERRQ(KSPCreate(PETSC_COMM_SELF, &ksp));
  CHKERRQ(KSPSetOptionsPrefix(ksp, "ftop_"));
  CHKERRQ(KSPSetFromOptions(ksp));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)ksp,KSPLSQR,&is_lsqr));
  if (!is_lsqr) {
    CHKERRQ(MatGetLocalSize(M_p, &M, &N));
    if (N>M) {
      PC        pc;
      CHKERRQ(PetscInfo(ksp, " M (%D) < M (%D) -- skip revert to lsqr\n",M,N));
      is_lsqr = PETSC_TRUE;
      CHKERRQ(KSPSetType(ksp,KSPLSQR));
      CHKERRQ(KSPGetPC(ksp,&pc));
      CHKERRQ(PCSetType(pc,PCNONE)); // could put in better solver -ftop_pc_type bjacobi -ftop_sub_pc_type lu -ftop_sub_pc_factor_shift_type nonzero
    } else {
      CHKERRQ(PetscNew(&matshellctx));
      CHKERRQ(MatCreateShell(PetscObjectComm((PetscObject)dm),N,N,PETSC_DECIDE,PETSC_DECIDE,matshellctx,&MtM));
      CHKERRQ(MatTranspose(M_p,MAT_INITIAL_MATRIX,&matshellctx->MpTrans));
      matshellctx->Mp = M_p;
      CHKERRQ(MatShellSetOperation(MtM, MATOP_MULT, (void (*)(void))MatMultMtM_SeqAIJ));
      CHKERRQ(MatShellSetOperation(MtM, MATOP_MULT_ADD, (void (*)(void))MatMultAddMtM_SeqAIJ));
      CHKERRQ(MatCreateVecs(M_p,&matshellctx->uu,&matshellctx->ff));
      CHKERRQ(MatCreateSeqAIJ(PETSC_COMM_SELF,N,N,1,NULL,&D));
      for (int i=0 ; i<N ; i++) {
        const PetscScalar *vals;
        const PetscInt    *cols;
        PetscScalar dot = 0;
        CHKERRQ(MatGetRow(matshellctx->MpTrans,i,&nzl,&cols,&vals));
        for (int ii=0 ; ii<nzl ; ii++) dot += PetscSqr(vals[ii]);
        PetscCheckFalse(dot==0.0,PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Row %D is empty", i);
        CHKERRQ(MatSetValue(D,i,i,dot,INSERT_VALUES));
      }
      CHKERRQ(MatAssemblyBegin(D, MAT_FINAL_ASSEMBLY));
      CHKERRQ(MatAssemblyEnd(D, MAT_FINAL_ASSEMBLY));
      PetscInfo(M_p,"createMtMKSP Have %D eqs, nzl = %D\n",N,nzl);
      CHKERRQ(KSPSetOperators(ksp, MtM, D));
      CHKERRQ(MatViewFromOptions(D,NULL,"-ftop2_D_mat_view"));
      CHKERRQ(MatViewFromOptions(M_p,NULL,"-ftop2_Mp_mat_view"));
      CHKERRQ(MatViewFromOptions(matshellctx->MpTrans,NULL,"-ftop2_MpTranspose_mat_view"));
    }
  }
  if (is_lsqr) {
    PC        pc;
    PetscBool is_bjac;
    CHKERRQ(KSPGetPC(ksp,&pc));
    CHKERRQ(PetscObjectTypeCompare((PetscObject)pc,PCBJACOBI,&is_bjac));
    if (is_bjac) {
      CHKERRQ(DMSwarmCreateMassMatrixSquare(sw, dm, &PM_p));
      CHKERRQ(KSPSetOperators(ksp, M_p, PM_p));
    } else {
      CHKERRQ(KSPSetOperators(ksp, M_p, M_p));
    }
  }
  CHKERRQ(DMSwarmCreateGlobalVectorFromField(sw, "w_q", &ff)); // this grabs access !!!!!
  if (!is_lsqr) {
    CHKERRQ(KSPSolve(ksp, rhs, matshellctx->uu));
    CHKERRQ(MatMult(M_p, matshellctx->uu, ff));
    CHKERRQ(MatDestroy(&matshellctx->MpTrans));
    CHKERRQ(VecDestroy(&matshellctx->ff));
    CHKERRQ(VecDestroy(&matshellctx->uu));
    CHKERRQ(MatDestroy(&D));
    CHKERRQ(MatDestroy(&MtM));
    CHKERRQ(PetscFree(matshellctx));
  } else {
    CHKERRQ(KSPSolveTranspose(ksp, rhs, ff));
  }
  CHKERRQ(KSPDestroy(&ksp));
  /* Visualize particle field */
  CHKERRQ(DMSetOutputSequenceNumber(sw, timestep, time));
  CHKERRQ(VecViewFromOptions(ff, NULL, "-weights_view"));
  CHKERRQ(DMSwarmDestroyGlobalVectorFromField(sw, "w_q", &ff));

  /* compute energy */
  if (moments) {
    PetscReal *wq, *coords;
    CHKERRQ(DMSwarmGetLocalSize(sw,&Np));
    CHKERRQ(DMSwarmGetField(sw, "w_q", &bs, &dtype, (void**)&wq));
    CHKERRQ(DMSwarmGetField(sw, "DMSwarmPIC_coor", &bs, &dtype, (void**)&coords));
    moments[0] = moments[1] = moments[2] = 0;
    for (int p=0;p<Np;p++) {
      moments[0] += wq[p];
      moments[1] += wq[p] * coords[p*2+0]; // x-momentum
      moments[2] += wq[p] * (PetscSqr(coords[p*2+0])+PetscSqr(coords[p*2+1]));
    }
    CHKERRQ(DMSwarmRestoreField(sw, "DMSwarmPIC_coor", &bs, &dtype, (void**)&coords));
    CHKERRQ(DMSwarmRestoreField(sw, "w_q", &bs, &dtype, (void**)&wq));
  }
  CHKERRQ(MatDestroy(&PM_p));
  PetscFunctionReturn(0);
}

PetscErrorCode particlesToGrid(const DM dm, DM sw, const PetscInt Np, const PetscInt a_tid, const PetscInt dim, const PetscInt target,
                               const PetscReal xx[], const PetscReal yy[], const PetscReal a_wp[], Vec rho, Mat *Mp_out)
{

  PetscBool      removePoints = PETSC_TRUE;
  PetscReal      *wq, *coords;
  PetscDataType  dtype;
  Mat            M_p;
  Vec            ff;
  PetscInt       bs,p,zero=0;

  PetscFunctionBeginUser;
  CHKERRQ(DMSwarmSetLocalSizes(sw, Np, zero));
  CHKERRQ(DMSwarmGetField(sw, "w_q", &bs, &dtype, (void**)&wq));
  CHKERRQ(DMSwarmGetField(sw, "DMSwarmPIC_coor", &bs, &dtype, (void**)&coords));
  for (p=0;p<Np;p++) {
    coords[p*2+0]  = xx[p];
    coords[p*2+1]  = yy[p];
    wq[p]          = a_wp[p];
  }
  CHKERRQ(DMSwarmRestoreField(sw, "DMSwarmPIC_coor", &bs, &dtype, (void**)&coords));
  CHKERRQ(DMSwarmRestoreField(sw, "w_q", &bs, &dtype, (void**)&wq));
  CHKERRQ(DMSwarmMigrate(sw, removePoints));
  CHKERRQ(PetscObjectSetName((PetscObject)sw, "Particle Grid"));
  if (a_tid==target) CHKERRQ(DMViewFromOptions(sw, NULL, "-swarm_view"));

  /* Project particles to field */
  /* This gives M f = \int_\Omega \phi f, which looks like a rhs for a PDE */
  CHKERRQ(DMCreateMassMatrix(sw, dm, &M_p));
  CHKERRQ(PetscObjectSetName((PetscObject)rho, "rho"));

  CHKERRQ(DMSwarmCreateGlobalVectorFromField(sw, "w_q", &ff)); // this grabs access !!!!!
  CHKERRQ(PetscObjectSetName((PetscObject)ff, "weights"));
  CHKERRQ(MatMultTranspose(M_p, ff, rho));
  CHKERRQ(DMSwarmDestroyGlobalVectorFromField(sw, "w_q", &ff));

  /* Visualize mesh field */
  if (a_tid==target) CHKERRQ(VecViewFromOptions(rho, NULL, "-rho_view"));
  // output
  *Mp_out = M_p;

  PetscFunctionReturn(0);
}
static PetscErrorCode maxwellian(PetscInt dim, const PetscReal x[], PetscReal kt_m, PetscReal n, PetscScalar *u)
{
  PetscInt      i;
  PetscReal     v2 = 0, theta = 2*kt_m; /* theta = 2kT/mc^2 */

  PetscFunctionBegin;
  /* compute the exponents, v^2 */
  for (i = 0; i < dim; ++i) v2 += x[i]*x[i];
  /* evaluate the Maxwellian */
  u[0] = n*PetscPowReal(PETSC_PI*theta,-1.5)*(PetscExpReal(-v2/theta)) * 2.*PETSC_PI*x[1]; // radial term for 2D axi-sym.
  PetscFunctionReturn(0);
}
#define NUM_SOLVE_LOOPS 100
#define MAX_NUM_THRDS 12
PetscErrorCode go()
{
  DM              dm_t[MAX_NUM_THRDS], sw_t[MAX_NUM_THRDS];
  PetscFE         fe;
  PetscInt        dim = 2, Nc = 1, timestep = 0, i, faces[3];
  PetscInt        Np[2] = {10,10}, Np2[2], field = 0, target = 0, Np_t[MAX_NUM_THRDS];
  PetscReal       time = 0.0, moments_0[3], moments_1[3], vol;
  PetscReal       lo[3] = {-5,0,-5}, hi[3] = {5,5,5}, h[3], hp[3], *xx_t[MAX_NUM_THRDS], *yy_t[MAX_NUM_THRDS], *wp_t[MAX_NUM_THRDS], solve_time = 0;
  Vec             rho_t[MAX_NUM_THRDS], rhs_t[MAX_NUM_THRDS];
  Mat             M_p_t[MAX_NUM_THRDS];
#if defined PETSC_USE_LOG
  PetscLogStage   stage;
  PetscLogEvent   swarm_create_ev, solve_ev, solve_loop_ev;
#endif
#if defined(PETSC_HAVE_OPENMP) && defined(PETSC_HAVE_THREADSAFETY)
  PetscInt        numthreads = PetscNumOMPThreads;
  double          starttime, endtime;
#else
  PetscInt        numthreads = 1;
#endif

  PetscFunctionBeginUser;
#if defined(PETSC_HAVE_OPENMP) && defined(PETSC_HAVE_THREADSAFETY)
  PetscCheckFalse(numthreads>MAX_NUM_THRDS,PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Too many threads %D > %D", numthreads, MAX_NUM_THRDS);
  PetscCheckFalse(numthreads<=0,PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "No threads %D > %D ", numthreads,  MAX_NUM_THRDS);
#endif
  if (target >= numthreads) target = numthreads-1;
  CHKERRQ(PetscLogEventRegister("Create Swarm", DM_CLASSID, &swarm_create_ev));
  CHKERRQ(PetscLogEventRegister("Single solve", DM_CLASSID, &solve_ev));
  CHKERRQ(PetscLogEventRegister("Solve loop", DM_CLASSID, &solve_loop_ev));
  CHKERRQ(PetscLogStageRegister("Solve", &stage));
  i    = dim;
  CHKERRQ(PetscOptionsGetIntArray(NULL, NULL, "-dm_plex_box_faces", faces, &i, NULL));
  i    = dim;
  CHKERRQ(PetscOptionsGetIntArray(NULL, NULL, "-np", Np,  &i, NULL));
  /* Create thread meshes */
  for (int tid=0; tid<numthreads; tid++) {
    // setup mesh dm_t, could use PETSc's Landau create velocity space mesh here to get dm_t[tid]
    CHKERRQ(DMCreate(PETSC_COMM_SELF, &dm_t[tid]));
    CHKERRQ(DMSetType(dm_t[tid], DMPLEX));
    CHKERRQ(DMSetFromOptions(dm_t[tid]));
    CHKERRQ(PetscFECreateDefault(PETSC_COMM_SELF, dim, Nc, PETSC_FALSE, "", PETSC_DECIDE, &fe));
    CHKERRQ(PetscFESetFromOptions(fe));
    CHKERRQ(PetscObjectSetName((PetscObject)fe, "fe"));
    CHKERRQ(DMSetField(dm_t[tid], field, NULL, (PetscObject)fe));
    CHKERRQ(DMCreateDS(dm_t[tid]));
    CHKERRQ(PetscFEDestroy(&fe));
    // helper vectors
    CHKERRQ(DMSetOutputSequenceNumber(dm_t[tid], timestep, time)); // not used
    CHKERRQ(DMCreateGlobalVector(dm_t[tid], &rho_t[tid]));
    CHKERRQ(DMCreateGlobalVector(dm_t[tid], &rhs_t[tid]));
    // this mimics application code
    CHKERRQ(DMGetBoundingBox(dm_t[tid], lo, hi));
    if (tid==target) {
      CHKERRQ(DMViewFromOptions(dm_t[tid], NULL, "-dm_view"));
      for (i=0,vol=1;i<dim;i++) {
        h[i] = (hi[i] - lo[i])/faces[i];
        hp[i] = (hi[i] - lo[i])/Np[i];
        vol *= (hi[i] - lo[i]);
        CHKERRQ(PetscInfo(dm_t[tid]," lo = %g hi = %g n = %D h = %g hp = %g\n",lo[i],hi[i],faces[i],h[i],hp[i]));
      }
    }
  }
  // prepare particle data for problems. This mimics application code
  CHKERRQ(PetscLogEventBegin(swarm_create_ev,0,0,0,0));
  Np2[0] = Np[0]; Np2[1] = Np[1];
  for (int tid=0; tid<numthreads; tid++) { // change size of particle list a little
    Np_t[tid] = Np2[0]*Np2[1];
    CHKERRQ(PetscMalloc3(Np_t[tid],&xx_t[tid],Np_t[tid],&yy_t[tid],Np_t[tid],&wp_t[tid]));
    if (tid==target) {moments_0[0] = moments_0[1] = moments_0[2] = 0;}
    for (int pi=0, pp=0;pi<Np2[0];pi++) {
      for (int pj=0;pj<Np2[1];pj++,pp++) {
        xx_t[tid][pp] = lo[0] + hp[0]/2. + pi*hp[0];
        yy_t[tid][pp] = lo[1] + hp[1]/2. + pj*hp[1];
        {
          PetscReal x[] = {xx_t[tid][pp],yy_t[tid][pp]};
          CHKERRQ(maxwellian(2, x, 1.0, vol/(PetscReal)Np_t[tid], &wp_t[tid][pp]));
        }
        if (tid==target) { //energy_0 += wp_t[tid][pp]*(PetscSqr(xx_t[tid][pp])+PetscSqr(yy_t[tid][pp]));
          moments_0[0] += wp_t[tid][pp];
          moments_0[1] += wp_t[tid][pp] * xx_t[tid][pp]; // x-momentum
          moments_0[2] += wp_t[tid][pp] * (PetscSqr(xx_t[tid][pp]) + PetscSqr(yy_t[tid][pp]));
        }
      }
    }
    Np2[0]++; Np2[1]++;
  }
  CHKERRQ(PetscLogEventEnd(swarm_create_ev,0,0,0,0));
  CHKERRQ(PetscLogEventBegin(solve_ev,0,0,0,0));
  /* Create particle swarm */
  PetscPragmaOMP(parallel for)
  for (int tid=0; tid<numthreads; tid++) {
    CHKERRABORT(PETSC_COMM_SELF,createSwarm(dm_t[tid], &sw_t[tid]));
  }
  PetscPragmaOMP(parallel for)
  for (int tid=0; tid<numthreads; tid++) {
    CHKERRABORT(PETSC_COMM_SELF,particlesToGrid(dm_t[tid], sw_t[tid], Np_t[tid], tid, dim, target, xx_t[tid], yy_t[tid], wp_t[tid], rho_t[tid], &M_p_t[tid]));
  }
  /* Project field to particles */
  /*   This gives f_p = M_p^+ M f */
  PetscPragmaOMP(parallel for)
  for (int tid=0; tid<numthreads; tid++) {
    CHKERRABORT(PETSC_COMM_SELF,VecCopy(rho_t[tid], rhs_t[tid])); /* Identity: M^1 M rho */
  }
  PetscPragmaOMP(parallel for)
  for (int tid=0; tid<numthreads; tid++) {
    CHKERRABORT(PETSC_COMM_SELF,gridToParticles(dm_t[tid], sw_t[tid], (tid==target) ?  moments_1 : NULL, rhs_t[tid], M_p_t[tid]));
  }
  /* Cleanup */
  for (int tid=0; tid<numthreads; tid++) {
    CHKERRQ(MatDestroy(&M_p_t[tid]));
    CHKERRQ(DMDestroy(&sw_t[tid]));
  }
  CHKERRQ(PetscLogEventEnd(solve_ev,0,0,0,0));
  /* for timing */
  CHKERRQ(PetscOptionsClearValue(NULL,"-ftop_ksp_converged_reason"));
  CHKERRQ(PetscOptionsClearValue(NULL,"-ftop_ksp_monitor"));
  CHKERRQ(PetscOptionsClearValue(NULL,"-ftop_ksp_view"));
  CHKERRQ(PetscOptionsClearValue(NULL,"-info"));
  // repeat solve many times to get warmed up data
  CHKERRQ(PetscLogStagePush(stage));
#if defined(PETSC_HAVE_OPENMP) && defined(PETSC_HAVE_THREADSAFETY)
  starttime = MPI_Wtime();
#endif
  CHKERRQ(PetscLogEventBegin(solve_loop_ev,0,0,0,0));
  for (int d=0; d<NUM_SOLVE_LOOPS; d++) {
  /* Create particle swarm */
    PetscPragmaOMP(parallel for)
    for (int tid=0; tid<numthreads; tid++) {
      CHKERRABORT(PETSC_COMM_SELF,createSwarm(dm_t[tid], &sw_t[tid]));
    }
    PetscPragmaOMP(parallel for)
    for (int tid=0; tid<numthreads; tid++) {
      CHKERRABORT(PETSC_COMM_SELF,particlesToGrid(dm_t[tid], sw_t[tid], Np_t[tid], tid, dim, target, xx_t[tid], yy_t[tid], wp_t[tid], rho_t[tid], &M_p_t[tid]));
    }
    PetscPragmaOMP(parallel for)
    for (int tid=0; tid<numthreads; tid++) {
      CHKERRABORT(PETSC_COMM_SELF,VecCopy(rho_t[tid], rhs_t[tid])); /* Identity: M^1 M rho */
    }
    PetscPragmaOMP(parallel for)
    for (int tid=0; tid<numthreads; tid++) {
      CHKERRABORT(PETSC_COMM_SELF,gridToParticles(dm_t[tid], sw_t[tid], NULL, rhs_t[tid], M_p_t[tid]));
    }
    /* Cleanup */
    for (int tid=0; tid<numthreads; tid++) {
      CHKERRQ(MatDestroy(&M_p_t[tid]));
      CHKERRQ(DMDestroy(&sw_t[tid]));
    }
  }
  CHKERRQ(PetscLogEventEnd(solve_loop_ev,0,0,0,0));
#if defined(PETSC_HAVE_OPENMP) && defined(PETSC_HAVE_THREADSAFETY)
  endtime = MPI_Wtime();
  solve_time += (endtime - starttime);
#endif
  CHKERRQ(PetscLogStagePop());
  //
  CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Total number density: %20.12e (%20.12e); x-momentum = %g (%g); energy = %g error = %e, %D particles. Use %D threads, Solve time: %g\n", moments_1[0], moments_0[0], moments_1[1], moments_0[1], moments_1[2], (moments_1[2]-moments_0[2])/moments_0[2],Np[0]*Np[1],numthreads,solve_time));
  /* Cleanup */
  for (int tid=0; tid<numthreads; tid++) {
    CHKERRQ(VecDestroy(&rho_t[tid]));
    CHKERRQ(VecDestroy(&rhs_t[tid]));
    CHKERRQ(DMDestroy(&dm_t[tid]));
    CHKERRQ(PetscFree3(xx_t[tid],yy_t[tid],wp_t[tid]));
  }
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  CHKERRQ(PetscInitialize(&argc, &argv, NULL,help));
  CHKERRQ(go());
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

  build:
    requires: !complex

  test:
    suffix: 0
    requires: double triangle
    args: -dm_plex_simplex 0 -dm_plex_box_faces 4,2 -dm_plex_box_lower -2.0,0.0 -dm_plex_box_upper 2.0,2.0 -petscspace_degree 2 -ftop_ksp_type lsqr -ftop_pc_type none -dm_view -ftop_ksp_converged_reason -ftop_ksp_rtol 1.e-14
    filter: grep -v DM_ | grep -v atomic

  test:
    suffix: 1
    requires: double triangle
    args: -dm_plex_simplex 0 -dm_plex_box_faces 4,2 -dm_plex_box_lower -2.0,0.0 -dm_plex_box_upper 2.0,2.0 -petscspace_degree 2 -dm_plex_hash_location -ftop_ksp_type lsqr -ftop_pc_type bjacobi -ftop_sub_pc_type lu -ftop_sub_pc_factor_shift_type nonzero -dm_view -ftop_ksp_converged_reason -ftop_ksp_rtol 1.e-14
    filter: grep -v DM_ | grep -v atomic

  test:
    suffix: 2
    requires: double triangle
    args: -dm_plex_simplex 0 -dm_plex_box_faces 4,2 -dm_plex_box_lower -2.0,0.0 -dm_plex_box_upper 2.0,2.0 -petscspace_degree 2 -dm_plex_hash_location -ftop_ksp_type cg -ftop_pc_type jacobi -dm_view -ftop_ksp_converged_reason -ftop_ksp_rtol 1.e-14
    filter: grep -v DM_ | grep -v atomic

TEST*/
