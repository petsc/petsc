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
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = MatShellGetContext(MtM,&matshellctx);CHKERRQ(ierr);
  if (!matshellctx) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "No context");
  ierr = MatMult(matshellctx->Mp, xx, matshellctx->ff);CHKERRQ(ierr);
  ierr = MatMult(matshellctx->MpTrans, matshellctx->ff, yy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultAddMtM_SeqAIJ(Mat MtM,Vec xx, Vec yy, Vec zz)
{
  MatShellCtx    *matshellctx;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = MatShellGetContext(MtM,&matshellctx);CHKERRQ(ierr);
  if (!matshellctx) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "No context");
  ierr = MatMult(matshellctx->Mp, xx, matshellctx->ff);CHKERRQ(ierr);
  ierr = MatMultAdd(matshellctx->MpTrans, matshellctx->ff, yy, zz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode createSwarm(const DM dm, DM *sw)
{
  PetscErrorCode ierr;
  PetscInt       Nc = 1, dim = 2;

  PetscFunctionBeginUser;
  ierr = DMCreate(PETSC_COMM_SELF, sw);CHKERRQ(ierr);
  ierr = DMSetType(*sw, DMSWARM);CHKERRQ(ierr);
  ierr = DMSetDimension(*sw, dim);CHKERRQ(ierr);
  ierr = DMSwarmSetType(*sw, DMSWARM_PIC);CHKERRQ(ierr);
  ierr = DMSwarmSetCellDM(*sw, dm);CHKERRQ(ierr);
  ierr = DMSwarmRegisterPetscDatatypeField(*sw, "w_q", Nc, PETSC_SCALAR);CHKERRQ(ierr);
  ierr = DMSwarmFinalizeFieldRegister(*sw);CHKERRQ(ierr);
  ierr = DMSetFromOptions(*sw);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode gridToParticles(const DM dm, DM sw, PetscReal *moments, Vec rhs, Mat M_p)
{
  PetscBool      is_lsqr;
  KSP            ksp;
  Mat            PM_p=NULL,MtM,D;
  Vec            ff;
  PetscErrorCode ierr;
  PetscInt       Np, timestep = 0, bs, N, M, nzl;
  PetscReal      time = 0.0;
  PetscDataType  dtype;
  MatShellCtx    *matshellctx;

  PetscFunctionBeginUser;
  ierr = KSPCreate(PETSC_COMM_SELF, &ksp);CHKERRQ(ierr);
  ierr = KSPSetOptionsPrefix(ksp, "ftop_");CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)ksp,KSPLSQR,&is_lsqr);
  if (!is_lsqr) {
    ierr = MatGetLocalSize(M_p, &M, &N);CHKERRQ(ierr);
    if (N>M) {
      PC        pc;
      ierr = PetscInfo2(ksp, " M (%D) < M (%D) -- skip revert to lsqr\n",M,N);CHKERRQ(ierr);
      is_lsqr = PETSC_TRUE;
      ierr = KSPSetType(ksp,KSPLSQR);CHKERRQ(ierr);
      ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
      ierr = PCSetType(pc,PCNONE);CHKERRQ(ierr); // could put in better solver -ftop_pc_type bjacobi -ftop_sub_pc_type lu -ftop_sub_pc_factor_shift_type nonzero
    } else {
      ierr = PetscNew(&matshellctx);CHKERRQ(ierr);
      ierr = MatCreateShell(PetscObjectComm((PetscObject)dm),N,N,PETSC_DECIDE,PETSC_DECIDE,matshellctx,&MtM);CHKERRQ(ierr);
      ierr = MatTranspose(M_p,MAT_INITIAL_MATRIX,&matshellctx->MpTrans);CHKERRQ(ierr);
      matshellctx->Mp = M_p;
      ierr = MatShellSetOperation(MtM, MATOP_MULT, (void (*)(void))MatMultMtM_SeqAIJ);CHKERRQ(ierr);
      ierr = MatShellSetOperation(MtM, MATOP_MULT_ADD, (void (*)(void))MatMultAddMtM_SeqAIJ);CHKERRQ(ierr);
      ierr = MatCreateVecs(M_p,&matshellctx->uu,&matshellctx->ff);CHKERRQ(ierr);
      ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,N,N,1,NULL,&D);CHKERRQ(ierr);
      for (int i=0 ; i<N ; i++) {
        const PetscScalar *vals;
        const PetscInt    *cols;
        PetscScalar dot = 0;
        ierr = MatGetRow(matshellctx->MpTrans,i,&nzl,&cols,&vals);CHKERRQ(ierr);
        for (int ii=0 ; ii<nzl ; ii++) dot += PetscSqr(vals[ii]);
        if (dot==0.0) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Row %D is empty", i);
        ierr = MatSetValue(D,i,i,dot,INSERT_VALUES);
      }
      ierr = MatAssemblyBegin(D, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(D, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      PetscInfo2(M_p,"createMtMKSP Have %D eqs, nzl = %D\n",N,nzl);
      ierr = KSPSetOperators(ksp, MtM, D);CHKERRQ(ierr);
      ierr = MatViewFromOptions(D,NULL,"-ftop2_D_mat_view");CHKERRQ(ierr);
      ierr = MatViewFromOptions(M_p,NULL,"-ftop2_Mp_mat_view");CHKERRQ(ierr);
      ierr = MatViewFromOptions(matshellctx->MpTrans,NULL,"-ftop2_MpTranspose_mat_view");CHKERRQ(ierr);
    }
  }
  if (is_lsqr) {
    PC        pc;
    PetscBool is_bjac;
    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)pc,PCBJACOBI,&is_bjac);
    if (is_bjac) {
      ierr = DMSwarmCreateMassMatrixSquare(sw, dm, &PM_p);CHKERRQ(ierr);
      ierr = KSPSetOperators(ksp, M_p, PM_p);CHKERRQ(ierr);
    } else {
      ierr = KSPSetOperators(ksp, M_p, M_p);CHKERRQ(ierr);
    }
  }
  ierr = DMSwarmCreateGlobalVectorFromField(sw, "w_q", &ff);CHKERRQ(ierr); // this grabs access !!!!!
  if (!is_lsqr) {
    ierr = KSPSolve(ksp, rhs, matshellctx->uu);CHKERRQ(ierr);
    ierr = MatMult(M_p, matshellctx->uu, ff);CHKERRQ(ierr);
    ierr = MatDestroy(&matshellctx->MpTrans);CHKERRQ(ierr);
    ierr = VecDestroy(&matshellctx->ff);CHKERRQ(ierr);
    ierr = VecDestroy(&matshellctx->uu);CHKERRQ(ierr);
    ierr = MatDestroy(&D);CHKERRQ(ierr);
    ierr = MatDestroy(&MtM);CHKERRQ(ierr);
    ierr = PetscFree(matshellctx);CHKERRQ(ierr);
  } else {
    ierr = KSPSolveTranspose(ksp, rhs, ff);CHKERRQ(ierr);
  }
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  /* Visualize particle field */
  ierr = DMSetOutputSequenceNumber(sw, timestep, time);CHKERRQ(ierr);
  ierr = VecViewFromOptions(ff, NULL, "-weights_view");CHKERRQ(ierr);
  ierr = DMSwarmDestroyGlobalVectorFromField(sw, "w_q", &ff);CHKERRQ(ierr);

  /* compute energy */
  if (moments) {
    PetscReal *wq, *coords;
    ierr = DMSwarmGetLocalSize(sw,&Np);CHKERRQ(ierr);
    ierr = DMSwarmGetField(sw, "w_q", &bs, &dtype, (void**)&wq);CHKERRQ(ierr);
    ierr = DMSwarmGetField(sw, "DMSwarmPIC_coor", &bs, &dtype, (void**)&coords);CHKERRQ(ierr);
    moments[0] = moments[1] = moments[2] = 0;
    for (int p=0;p<Np;p++) {
      moments[0] += wq[p];
      moments[1] += wq[p] * coords[p*2+0]; // x-momentum
      moments[2] += wq[p] * (PetscSqr(coords[p*2+0])+PetscSqr(coords[p*2+1]));
    }
    ierr = DMSwarmRestoreField(sw, "DMSwarmPIC_coor", &bs, &dtype, (void**)&coords);CHKERRQ(ierr);
    ierr = DMSwarmRestoreField(sw, "w_q", &bs, &dtype, (void**)&wq);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&PM_p);

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
  PetscErrorCode ierr;
  PetscInt       bs,p,zero=0;

  PetscFunctionBeginUser;
  ierr = DMSwarmSetLocalSizes(sw, Np, zero);CHKERRQ(ierr);
  ierr = DMSwarmGetField(sw, "w_q", &bs, &dtype, (void**)&wq);CHKERRQ(ierr);
  ierr = DMSwarmGetField(sw, "DMSwarmPIC_coor", &bs, &dtype, (void**)&coords);CHKERRQ(ierr);
  for (p=0;p<Np;p++) {
    coords[p*2+0]  = xx[p];
    coords[p*2+1]  = yy[p];
    wq[p]          = a_wp[p];
  }
  ierr = DMSwarmRestoreField(sw, "DMSwarmPIC_coor", &bs, &dtype, (void**)&coords);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(sw, "w_q", &bs, &dtype, (void**)&wq);CHKERRQ(ierr);
  ierr = DMSwarmMigrate(sw, removePoints);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)sw, "Particle Grid");CHKERRQ(ierr);
  if (a_tid==target) {ierr = DMViewFromOptions(sw, NULL, "-swarm_view");CHKERRQ(ierr);}

  /* Project particles to field */
  /* This gives M f = \int_\Omega \phi f, which looks like a rhs for a PDE */
  ierr = DMCreateMassMatrix(sw, dm, &M_p);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)rho, "rho");CHKERRQ(ierr);

  ierr = DMSwarmCreateGlobalVectorFromField(sw, "w_q", &ff);CHKERRQ(ierr); // this grabs access !!!!!
  ierr = PetscObjectSetName((PetscObject)ff, "weights");CHKERRQ(ierr);
  ierr = MatMultTranspose(M_p, ff, rho);CHKERRQ(ierr);
  ierr = DMSwarmDestroyGlobalVectorFromField(sw, "w_q", &ff);CHKERRQ(ierr);

  /* Visualize mesh field */
  if (a_tid==target) {ierr = VecViewFromOptions(rho, NULL, "-rho_view");CHKERRQ(ierr);}
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
  PetscErrorCode  ierr;
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
  if (numthreads>MAX_NUM_THRDS) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Too many threads %D > %D", numthreads, MAX_NUM_THRDS);
  if (numthreads<=0) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "No threads %D > %D ", numthreads,  MAX_NUM_THRDS);
#endif
  if (target >= numthreads) target = numthreads-1;
  ierr = PetscLogEventRegister("Create Swarm", DM_CLASSID, &swarm_create_ev);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("Single solve", DM_CLASSID, &solve_ev);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("Solve loop", DM_CLASSID, &solve_loop_ev);CHKERRQ(ierr);
  ierr = PetscLogStageRegister("Solve", &stage);CHKERRQ(ierr);
  i    = dim;
  ierr = PetscOptionsGetIntArray(NULL, NULL, "-dm_plex_box_faces", faces, &i, NULL);CHKERRQ(ierr);
  i    = dim;
  ierr = PetscOptionsGetIntArray(NULL, NULL, "-np", Np,  &i, NULL);CHKERRQ(ierr);
  /* Create thread meshes */
  for (int tid=0; tid<numthreads; tid++) {
    // setup mesh dm_t, could use PETSc's Landau create velocity space mesh here to get dm_t[tid]
    ierr = DMCreate(PETSC_COMM_SELF, &dm_t[tid]);CHKERRQ(ierr);
    ierr = DMSetType(dm_t[tid], DMPLEX);CHKERRQ(ierr);
    ierr = DMSetFromOptions(dm_t[tid]);CHKERRQ(ierr);
    ierr = PetscFECreateDefault(PETSC_COMM_SELF, dim, Nc, PETSC_FALSE, "", PETSC_DECIDE, &fe);CHKERRQ(ierr);
    ierr = PetscFESetFromOptions(fe);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)fe, "fe");CHKERRQ(ierr);
    ierr = DMSetField(dm_t[tid], field, NULL, (PetscObject)fe);CHKERRQ(ierr);
    ierr = DMCreateDS(dm_t[tid]);CHKERRQ(ierr);
    ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
    // helper vectors
    ierr = DMSetOutputSequenceNumber(dm_t[tid], timestep, time);CHKERRQ(ierr); // not used
    ierr = DMCreateGlobalVector(dm_t[tid], &rho_t[tid]);CHKERRQ(ierr);
    ierr = DMCreateGlobalVector(dm_t[tid], &rhs_t[tid]);CHKERRQ(ierr);
    // this mimics application code
    ierr = DMGetBoundingBox(dm_t[tid], lo, hi);CHKERRQ(ierr);
    if (tid==target) {
      ierr = DMViewFromOptions(dm_t[tid], NULL, "-dm_view");CHKERRQ(ierr);
      for (i=0,vol=1;i<dim;i++) {
        h[i] = (hi[i] - lo[i])/faces[i];
        hp[i] = (hi[i] - lo[i])/Np[i];
        vol *= (hi[i] - lo[i]);
        ierr = PetscInfo5(dm_t[tid]," lo = %g hi = %g n = %D h = %g hp = %g\n",lo[i],hi[i],faces[i],h[i],hp[i]);CHKERRQ(ierr);
      }
    }
  }
  // prepare particle data for problems. This mimics application code
  ierr = PetscLogEventBegin(swarm_create_ev,0,0,0,0);CHKERRQ(ierr);
  Np2[0] = Np[0]; Np2[1] = Np[1];
  for (int tid=0; tid<numthreads; tid++) { // change size of particle list a little
    Np_t[tid] = Np2[0]*Np2[1];
    ierr = PetscMalloc3(Np_t[tid],&xx_t[tid],Np_t[tid],&yy_t[tid],Np_t[tid],&wp_t[tid]);CHKERRQ(ierr);
    if (tid==target) {moments_0[0] = moments_0[1] = moments_0[2] = 0;}
    for (int pi=0, pp=0;pi<Np2[0];pi++) {
      for (int pj=0;pj<Np2[1];pj++,pp++) {
        xx_t[tid][pp] = lo[0] + hp[0]/2. + pi*hp[0];
        yy_t[tid][pp] = lo[1] + hp[1]/2. + pj*hp[1];
        {
          PetscReal x[] = {xx_t[tid][pp],yy_t[tid][pp]};
          ierr = maxwellian(2, x, 1.0, vol/(PetscReal)Np_t[tid], &wp_t[tid][pp]);
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
  ierr = PetscLogEventEnd(swarm_create_ev,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(solve_ev,0,0,0,0);CHKERRQ(ierr);
  /* Create particle swarm */
  PetscPragmaOMP(parallel for)
  for (int tid=0; tid<numthreads; tid++) {
    PetscErrorCode  ierr_t;
    ierr_t = createSwarm(dm_t[tid], &sw_t[tid]);
    if (ierr_t) ierr = ierr_t;
  }
  CHKERRQ(ierr);
  PetscPragmaOMP(parallel for)
  for (int tid=0; tid<numthreads; tid++) {
    PetscErrorCode  ierr_t;
    ierr_t = particlesToGrid(dm_t[tid], sw_t[tid], Np_t[tid], tid, dim, target, xx_t[tid], yy_t[tid], wp_t[tid], rho_t[tid], &M_p_t[tid]);
    if (ierr_t) ierr = ierr_t;
  }
  CHKERRQ(ierr);
  /* Project field to particles */
  /*   This gives f_p = M_p^+ M f */
  PetscPragmaOMP(parallel for)
  for (int tid=0; tid<numthreads; tid++) {
    PetscErrorCode  ierr_t;
    ierr_t = VecCopy(rho_t[tid], rhs_t[tid]); /* Identity: M^1 M rho */
    if (ierr_t) ierr = ierr_t;
  }
  CHKERRQ(ierr);
  PetscPragmaOMP(parallel for)
  for (int tid=0; tid<numthreads; tid++) {
    PetscErrorCode  ierr_t;
    ierr_t = gridToParticles(dm_t[tid], sw_t[tid], (tid==target) ?  moments_1 : NULL, rhs_t[tid], M_p_t[tid]);
    if (ierr_t) ierr = ierr_t;
  }
  CHKERRQ(ierr);
  /* Cleanup */
  for (int tid=0; tid<numthreads; tid++) {
    ierr = MatDestroy(&M_p_t[tid]);CHKERRQ(ierr);
    ierr = DMDestroy(&sw_t[tid]);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(solve_ev,0,0,0,0);CHKERRQ(ierr);
  /* for timing */
  ierr = PetscOptionsClearValue(NULL,"-ftop_ksp_converged_reason");CHKERRQ(ierr);
  ierr = PetscOptionsClearValue(NULL,"-ftop_ksp_monitor");CHKERRQ(ierr);
  ierr = PetscOptionsClearValue(NULL,"-ftop_ksp_view");CHKERRQ(ierr);
  ierr = PetscOptionsClearValue(NULL,"-info");CHKERRQ(ierr);
  // repeat solve many times to get warmed up data
  ierr = PetscLogStagePush(stage);CHKERRQ(ierr);
#if defined(PETSC_HAVE_OPENMP) && defined(PETSC_HAVE_THREADSAFETY)
  starttime = MPI_Wtime();
#endif
  ierr = PetscLogEventBegin(solve_loop_ev,0,0,0,0);CHKERRQ(ierr);
  for (int d=0; d<NUM_SOLVE_LOOPS; d++) {
  /* Create particle swarm */
    PetscPragmaOMP(parallel for)
    for (int tid=0; tid<numthreads; tid++) {
      PetscErrorCode  ierr_t;
      ierr_t = createSwarm(dm_t[tid], &sw_t[tid]);
      if (ierr_t) ierr = ierr_t;
    }
    CHKERRQ(ierr);
    PetscPragmaOMP(parallel for)
    for (int tid=0; tid<numthreads; tid++) {
      PetscErrorCode  ierr_t;
      ierr_t = particlesToGrid(dm_t[tid], sw_t[tid], Np_t[tid], tid, dim, target, xx_t[tid], yy_t[tid], wp_t[tid], rho_t[tid], &M_p_t[tid]);
      if (ierr_t) ierr = ierr_t;
    }
    CHKERRQ(ierr);
    PetscPragmaOMP(parallel for)
    for (int tid=0; tid<numthreads; tid++) {
      PetscErrorCode  ierr_t;
      ierr_t = VecCopy(rho_t[tid], rhs_t[tid]); /* Identity: M^1 M rho */
      if (ierr_t) ierr = ierr_t;
    }
    CHKERRQ(ierr);
    PetscPragmaOMP(parallel for)
    for (int tid=0; tid<numthreads; tid++) {
      PetscErrorCode  ierr_t;
      ierr_t = gridToParticles(dm_t[tid], sw_t[tid], NULL, rhs_t[tid], M_p_t[tid]);
      if (ierr_t) ierr = ierr_t;
    }
    CHKERRQ(ierr);
    /* Cleanup */
    for (int tid=0; tid<numthreads; tid++) {
      ierr = MatDestroy(&M_p_t[tid]);CHKERRQ(ierr);
      ierr = DMDestroy(&sw_t[tid]);CHKERRQ(ierr);
    }
  }
  ierr = PetscLogEventEnd(solve_loop_ev,0,0,0,0);CHKERRQ(ierr);
#if defined(PETSC_HAVE_OPENMP) && defined(PETSC_HAVE_THREADSAFETY)
  endtime = MPI_Wtime();
  solve_time += (endtime - starttime);
#endif
  ierr = PetscLogStagePop();CHKERRQ(ierr);
  //
  ierr = PetscPrintf(PETSC_COMM_SELF,"Total number density: %20.12e (%20.12e); x-momentum = %g (%g); energy = %g error = %e, %D particles. Use %D threads, Solve time: %g\n", moments_1[0], moments_0[0], moments_1[1], moments_0[1], moments_1[2], (moments_1[2]-moments_0[2])/moments_0[2],Np[0]*Np[1],numthreads,solve_time);CHKERRQ(ierr);
  /* Cleanup */
  for (int tid=0; tid<numthreads; tid++) {
    ierr = VecDestroy(&rho_t[tid]);CHKERRQ(ierr);
    ierr = VecDestroy(&rhs_t[tid]);CHKERRQ(ierr);
    ierr = DMDestroy(&dm_t[tid]);CHKERRQ(ierr);
    ierr = PetscFree3(xx_t[tid],yy_t[tid],wp_t[tid]);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  PetscErrorCode  ierr;
  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  ierr = go();CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
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
