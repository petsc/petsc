#include <../src/mat/impls/aij/seq/aij.h>
#include <petsc/private/dmpleximpl.h>   /*I "petscdmplex.h" I*/
#include <petsclandau.h>                /*I "petsclandau.h"   I*/
#include <petscts.h>
#include <petscdmforest.h>
#include <petscdmcomposite.h>

/* Landau collision operator */

/* relativistic terms */
#if defined(PETSC_USE_REAL_SINGLE)
#define SPEED_OF_LIGHT 2.99792458e8F
#define C_0(v0) (SPEED_OF_LIGHT/v0) /* needed for relativistic tensor on all architectures */
#else
#define SPEED_OF_LIGHT 2.99792458e8
#define C_0(v0) (SPEED_OF_LIGHT/v0) /* needed for relativistic tensor on all architectures */
#endif

#define PETSC_THREAD_SYNC
#include "land_tensors.h"

#if defined(PETSC_HAVE_OPENMP)
#include <omp.h>
#endif

/* vector padding not supported */
#define LANDAU_VL  1

static PetscErrorCode LandauMatMult(Mat A, Vec x, Vec y)
{
  PetscErrorCode  ierr;
  LandauCtx       *ctx;
  PetscContainer  container;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject) A, "LandauCtx", (PetscObject *) &container);CHKERRQ(ierr);
  if (container) {
    ierr = PetscContainerGetPointer(container, (void **) &ctx);CHKERRQ(ierr);
    ierr = VecScatterBegin(ctx->plex_batch,x,ctx->work_vec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(ctx->plex_batch,x,ctx->work_vec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = (*ctx->seqaij_mult)(A,ctx->work_vec,y);CHKERRQ(ierr);
    ierr = VecCopy(y, ctx->work_vec);CHKERRQ(ierr);
    ierr = VecScatterBegin(ctx->plex_batch,ctx->work_vec,y,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(ctx->plex_batch,ctx->work_vec,y,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = MatMult(A,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

// Computes v3 = v2 + A * v1.
static PetscErrorCode LandauMatMultAdd(Mat A,Vec v1,Vec v2,Vec v3)
{
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "?????");
  ierr = LandauMatMult(A,v1,v3);CHKERRQ(ierr);
  ierr = VecAYPX(v3,1,v2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode LandauMatMultTranspose(Mat A, Vec x, Vec y)
{
  PetscErrorCode  ierr;
  LandauCtx       *ctx;
  PetscContainer  container;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject) A, "LandauCtx", (PetscObject *) &container);CHKERRQ(ierr);
  if (container) {
    ierr = PetscContainerGetPointer(container, (void **) &ctx);CHKERRQ(ierr);
    ierr = VecScatterBegin(ctx->plex_batch,x,ctx->work_vec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(ctx->plex_batch,x,ctx->work_vec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = (*ctx->seqaij_multtranspose)(A,ctx->work_vec,y);CHKERRQ(ierr);
    ierr = VecCopy(y, ctx->work_vec);CHKERRQ(ierr);
    ierr = VecScatterBegin(ctx->plex_batch,ctx->work_vec,y,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(ctx->plex_batch,ctx->work_vec,y,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = MatMultTranspose(A,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode LandauMatGetDiagonal(Mat A,Vec x)
{
  PetscErrorCode  ierr;
  LandauCtx       *ctx;
  PetscContainer  container;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject) A, "LandauCtx", (PetscObject *) &container);CHKERRQ(ierr);
  if (container) {
    ierr = PetscContainerGetPointer(container, (void **) &ctx);CHKERRQ(ierr);
    ierr = (*ctx->seqaij_getdiagonal)(A,ctx->work_vec);CHKERRQ(ierr);
    ierr = VecScatterBegin(ctx->plex_batch,ctx->work_vec,x,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(ctx->plex_batch,ctx->work_vec,x,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = MatGetDiagonal(A, x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode LandauGPUMapsDestroy(void *ptr)
{
  P4estVertexMaps *maps = (P4estVertexMaps*)ptr;
  PetscErrorCode  ierr;
  PetscFunctionBegin;
  // free device data
  if (maps[0].deviceType != LANDAU_CPU) {
#if defined(PETSC_HAVE_KOKKOS_KERNELS)
    if (maps[0].deviceType == LANDAU_KOKKOS) {
      ierr = LandauKokkosDestroyMatMaps(maps,  maps[0].numgrids);CHKERRQ(ierr); // imples Kokkos does
    } // else could be CUDA
#elif defined(PETSC_HAVE_CUDA)
    if (maps[0].deviceType == LANDAU_CUDA) {
      ierr = LandauCUDADestroyMatMaps(maps, maps[0].numgrids);CHKERRQ(ierr);
    } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "maps->deviceType %D ?????",maps->deviceType);
#endif
  }
  // free host data
  for (PetscInt grid=0 ; grid < maps[0].numgrids ; grid++) {
    ierr = PetscFree(maps[grid].c_maps);CHKERRQ(ierr);
    ierr = PetscFree(maps[grid].gIdx);CHKERRQ(ierr);
  }
  ierr = PetscFree(maps);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
static PetscErrorCode energy_f(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf_dummy, PetscScalar *u, void *actx)
{
  PetscReal     v2 = 0;
  PetscFunctionBegin;
  /* compute v^2 / 2 */
  for (int i = 0; i < dim; ++i) v2 += x[i]*x[i];
  /* evaluate the Maxwellian */
  u[0] = v2/2;
  PetscFunctionReturn(0);
}

/* needs double */
static PetscErrorCode gamma_m1_f(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf_dummy, PetscScalar *u, void *actx)
{
  PetscReal     *c2_0_arr = ((PetscReal*)actx);
  double        u2 = 0, c02 = (double)*c2_0_arr, xx;

  PetscFunctionBegin;
  /* compute u^2 / 2 */
  for (int i = 0; i < dim; ++i) u2 += x[i]*x[i];
  /* gamma - 1 = g_eps, for conditioning and we only take derivatives */
  xx = u2/c02;
#if defined(PETSC_USE_DEBUG)
  u[0] = PetscSqrtReal(1. + xx);
#else
  u[0] = xx/(PetscSqrtReal(1. + xx) + 1.) - 1.; // better conditioned. -1 might help condition and only used for derivative
#endif
  PetscFunctionReturn(0);
}

/*
 LandauFormJacobian_Internal - Evaluates Jacobian matrix.

 Input Parameters:
 .  globX - input vector
 .  actx - optional user-defined context
 .  dim - dimension

 Output Parameters:
 .  J0acP - Jacobian matrix filled, not created
 */
static PetscErrorCode LandauFormJacobian_Internal(Vec a_X, Mat JacP, const PetscInt dim, PetscReal shift, void *a_ctx)
{
  LandauCtx         *ctx = (LandauCtx*)a_ctx;
  PetscErrorCode    ierr;
  PetscInt          numCells[LANDAU_MAX_GRIDS],Nq,Nb;
  PetscQuadrature   quad;
  PetscReal         Eq_m[LANDAU_MAX_SPECIES]; // could be static data w/o quench (ex2)
  PetscScalar       *cellClosure=NULL;
  const PetscScalar *xdata=NULL;
  PetscDS           prob;
  PetscContainer    container;
  P4estVertexMaps   *maps;
  Mat               subJ[LANDAU_MAX_GRIDS*LANDAU_MAX_BATCH_SZ];

  PetscFunctionBegin;
  PetscValidHeaderSpecific(a_X,VEC_CLASSID,1);
  PetscValidHeaderSpecific(JacP,MAT_CLASSID,2);
  PetscValidPointer(ctx,5);
  /* check for matrix container for GPU assembly. Support CPU assembly for debugging */
  PetscCheckFalse(ctx->plex[0] == NULL,ctx->comm,PETSC_ERR_ARG_WRONG,"Plex not created");
  ierr = PetscLogEventBegin(ctx->events[10],0,0,0,0);CHKERRQ(ierr);
  ierr = DMGetDS(ctx->plex[0], &prob);CHKERRQ(ierr); // same DS for all grids
  ierr = PetscObjectQuery((PetscObject) JacP, "assembly_maps", (PetscObject *) &container);CHKERRQ(ierr);
  if (container) {
    PetscCheck(ctx->gpu_assembly,ctx->comm,PETSC_ERR_ARG_WRONG,"maps but no GPU assembly");
    ierr = PetscContainerGetPointer(container, (void **) &maps);CHKERRQ(ierr);
    PetscCheckFalse(!maps,ctx->comm,PETSC_ERR_ARG_WRONG,"empty GPU matrix container");
    for (PetscInt i=0;i<ctx->num_grids*ctx->batch_sz;i++) subJ[i] = NULL;
  } else {
    PetscCheck(!ctx->gpu_assembly,ctx->comm,PETSC_ERR_ARG_WRONG,"No maps but GPU assembly");
    for (PetscInt tid=0 ; tid<ctx->batch_sz ; tid++) {
      for (PetscInt grid=0;grid<ctx->num_grids;grid++) {
        ierr = DMCreateMatrix(ctx->plex[grid], &subJ[ LAND_PACK_IDX(tid,grid) ]);CHKERRQ(ierr);
      }
    }
    maps = NULL;
  }
  // get dynamic data (Eq is odd, for quench and Spitzer test) for CPU assembly and raw data for Jacobian GPU assembly. Get host numCells[], Nq (yuck)
  ierr = PetscFEGetQuadrature(ctx->fe[0], &quad);CHKERRQ(ierr);
  ierr = PetscQuadratureGetData(quad, NULL, NULL, &Nq, NULL, NULL);CHKERRQ(ierr); Nb = Nq;
  PetscCheckFalse(Nq >LANDAU_MAX_NQ,ctx->comm,PETSC_ERR_ARG_WRONG,"Order too high. Nq = %D > LANDAU_MAX_NQ (%D)",Nq,LANDAU_MAX_NQ);
  // get metadata for collecting dynamic data
  for (PetscInt grid=0;grid<ctx->num_grids;grid++) {
    PetscInt cStart, cEnd;
    PetscCheckFalse(ctx->plex[grid] == NULL,ctx->comm,PETSC_ERR_ARG_WRONG,"Plex not created");
    ierr = DMPlexGetHeightStratum(ctx->plex[grid], 0, &cStart, &cEnd);CHKERRQ(ierr);
    numCells[grid] = cEnd - cStart; // grids can have different topology
  }
  ierr = PetscLogEventEnd(ctx->events[10],0,0,0,0);CHKERRQ(ierr);
  if (shift==0) { /* create dynamic point data: f_alpha for closure of each cell (cellClosure[nbatch,ngrids,ncells[g],f[Nb,ns[g]]]) or xdata */
    DM pack;
    ierr = VecGetDM(a_X, &pack);CHKERRQ(ierr);
    PetscCheckFalse(!pack,PETSC_COMM_SELF, PETSC_ERR_PLIB, "pack has no DM");
    ierr = PetscLogEventBegin(ctx->events[1],0,0,0,0);CHKERRQ(ierr);
    ierr = MatZeroEntries(JacP);CHKERRQ(ierr);
    for (PetscInt fieldA=0;fieldA<ctx->num_species;fieldA++) {
      Eq_m[fieldA] = ctx->Ez * ctx->t_0 * ctx->charges[fieldA] / (ctx->v_0 * ctx->masses[fieldA]); /* normalize dimensionless */
      if (dim==2) Eq_m[fieldA] *=  2 * PETSC_PI; /* add the 2pi term that is not in Landau */
    }
    if (!ctx->gpu_assembly) {
      Vec          *locXArray,*globXArray;
      PetscScalar  *cellClosure_it;
      PetscInt     cellClosure_sz=0,nDMs,Nf[LANDAU_MAX_GRIDS];
      PetscSection section[LANDAU_MAX_GRIDS],globsection[LANDAU_MAX_GRIDS];
      for (PetscInt grid=0;grid<ctx->num_grids;grid++) {
        ierr = DMGetLocalSection(ctx->plex[grid], &section[grid]);CHKERRQ(ierr);
        ierr = DMGetGlobalSection(ctx->plex[grid], &globsection[grid]);CHKERRQ(ierr);
        ierr = PetscSectionGetNumFields(section[grid], &Nf[grid]);CHKERRQ(ierr);
      }
      /* count cellClosure size */
      ierr = DMCompositeGetNumberDM(pack,&nDMs);CHKERRQ(ierr);
      for (PetscInt grid=0 ; grid<ctx->num_grids ; grid++) cellClosure_sz += Nb*Nf[grid]*numCells[grid];
      ierr = PetscMalloc1(cellClosure_sz*ctx->batch_sz,&cellClosure);CHKERRQ(ierr);
      cellClosure_it = cellClosure;
      ierr = PetscMalloc(sizeof(*locXArray)*nDMs, &locXArray);CHKERRQ(ierr);
      ierr = PetscMalloc(sizeof(*globXArray)*nDMs, &globXArray);CHKERRQ(ierr);
      ierr = DMCompositeGetLocalAccessArray(pack, a_X, nDMs, NULL, locXArray);CHKERRQ(ierr);
      ierr = DMCompositeGetAccessArray(pack, a_X, nDMs, NULL, globXArray);CHKERRQ(ierr);
      for (PetscInt b_id = 0 ; b_id < ctx->batch_sz ; b_id++) { // OpenMP (once)
        for (PetscInt grid=0 ; grid<ctx->num_grids ; grid++) {
          Vec         locX = locXArray[ LAND_PACK_IDX(b_id,grid) ], globX = globXArray[ LAND_PACK_IDX(b_id,grid) ], locX2;
          PetscInt    cStart, cEnd, ei;
          ierr = VecDuplicate(locX,&locX2);CHKERRQ(ierr);
          ierr = DMGlobalToLocalBegin(ctx->plex[grid], globX, INSERT_VALUES, locX2);CHKERRQ(ierr);
          ierr = DMGlobalToLocalEnd  (ctx->plex[grid], globX, INSERT_VALUES, locX2);CHKERRQ(ierr);
          ierr = DMPlexGetHeightStratum(ctx->plex[grid], 0, &cStart, &cEnd);CHKERRQ(ierr);
          for (ei = cStart ; ei < cEnd; ++ei) {
            PetscScalar *coef = NULL;
            ierr = DMPlexVecGetClosure(ctx->plex[grid], section[grid], locX2, ei, NULL, &coef);CHKERRQ(ierr);
            ierr = PetscMemcpy(cellClosure_it,coef,Nb*Nf[grid]*sizeof(*cellClosure_it));CHKERRQ(ierr); /* change if LandauIPReal != PetscScalar */
            ierr = DMPlexVecRestoreClosure(ctx->plex[grid], section[grid], locX2, ei, NULL, &coef);CHKERRQ(ierr);
            cellClosure_it += Nb*Nf[grid];
          }
          ierr = VecDestroy(&locX2);CHKERRQ(ierr);
        }
      }
      PetscCheckFalse(cellClosure_it-cellClosure != cellClosure_sz*ctx->batch_sz,PETSC_COMM_SELF, PETSC_ERR_PLIB, "iteration wrong %D != cellClosure_sz = %D",cellClosure_it-cellClosure,cellClosure_sz*ctx->batch_sz);
      ierr = DMCompositeRestoreLocalAccessArray(pack, a_X, nDMs, NULL, locXArray);CHKERRQ(ierr);
      ierr = DMCompositeRestoreAccessArray(pack, a_X, nDMs, NULL, globXArray);CHKERRQ(ierr);
      ierr = PetscFree(locXArray);CHKERRQ(ierr);
      ierr = PetscFree(globXArray);CHKERRQ(ierr);
      xdata = NULL;
    } else {
      PetscMemType mtype;
      if (ctx->jacobian_field_major_order) { // get data in batch ordering
        ierr = VecScatterBegin(ctx->plex_batch,a_X,ctx->work_vec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
        ierr = VecScatterEnd(ctx->plex_batch,a_X,ctx->work_vec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
        ierr = VecGetArrayReadAndMemType(ctx->work_vec,&xdata,&mtype);CHKERRQ(ierr);
      } else {
        ierr = VecGetArrayReadAndMemType(a_X,&xdata,&mtype);CHKERRQ(ierr);
      }
      if (mtype!=PETSC_MEMTYPE_HOST && ctx->deviceType == LANDAU_CPU) {
        SETERRQ(ctx->comm,PETSC_ERR_ARG_WRONG,"CPU run with device data: use -mat_type aij");
      }
      cellClosure = NULL;
    }
    ierr = PetscLogEventEnd(ctx->events[1],0,0,0,0);CHKERRQ(ierr);
  } else xdata = cellClosure = NULL;

  /* do it */
  if (ctx->deviceType == LANDAU_CUDA || ctx->deviceType == LANDAU_KOKKOS) {
    if (ctx->deviceType == LANDAU_CUDA) {
#if defined(PETSC_HAVE_CUDA)
      ierr = LandauCUDAJacobian(ctx->plex,Nq,ctx->batch_sz,ctx->num_grids,numCells,Eq_m,cellClosure,xdata,&ctx->SData_d,ctx->subThreadBlockSize,shift,ctx->events,ctx->mat_offset, ctx->species_offset, subJ, JacP);CHKERRQ(ierr);
#else
      SETERRQ(ctx->comm,PETSC_ERR_ARG_WRONG,"-landau_device_type %s not built","cuda");
#endif
    } else if (ctx->deviceType == LANDAU_KOKKOS) {
#if defined(PETSC_HAVE_KOKKOS_KERNELS)
      ierr = LandauKokkosJacobian(ctx->plex,Nq,ctx->batch_sz,ctx->num_grids,numCells,Eq_m,cellClosure,xdata,&ctx->SData_d,ctx->subThreadBlockSize,shift,ctx->events,ctx->mat_offset, ctx->species_offset, subJ,JacP);CHKERRQ(ierr);
#else
      SETERRQ(ctx->comm,PETSC_ERR_ARG_WRONG,"-landau_device_type %s not built","kokkos");
#endif
    }
  } else {   /* CPU version */
    PetscTabulation *Tf; // used for CPU and print info. Same on all grids and all species
    PetscInt        ip_offset[LANDAU_MAX_GRIDS+1], ipf_offset[LANDAU_MAX_GRIDS+1], elem_offset[LANDAU_MAX_GRIDS+1],IPf_sz_glb,IPf_sz_tot,num_grids=ctx->num_grids,Nf[LANDAU_MAX_GRIDS];
    PetscReal       *ff, *dudx, *dudy, *dudz, *invJ_a = (PetscReal*)ctx->SData_d.invJ, *xx = (PetscReal*)ctx->SData_d.x, *yy = (PetscReal*)ctx->SData_d.y, *zz = (PetscReal*)ctx->SData_d.z, *ww = (PetscReal*)ctx->SData_d.w;
    PetscReal       Eq_m[LANDAU_MAX_SPECIES], invMass[LANDAU_MAX_SPECIES], nu_alpha[LANDAU_MAX_SPECIES], nu_beta[LANDAU_MAX_SPECIES];
    PetscSection    section[LANDAU_MAX_GRIDS],globsection[LANDAU_MAX_GRIDS];
    for (PetscInt grid=0;grid<ctx->num_grids;grid++) {
      ierr = DMGetLocalSection(ctx->plex[grid], &section[grid]);CHKERRQ(ierr);
      ierr = DMGetGlobalSection(ctx->plex[grid], &globsection[grid]);CHKERRQ(ierr);
      ierr = PetscSectionGetNumFields(section[grid], &Nf[grid]);CHKERRQ(ierr);
    }
    /* count IPf size, etc */
    ierr = PetscDSGetTabulation(prob, &Tf);CHKERRQ(ierr); // Bf, &Df same for all grids
    const PetscReal *const BB = Tf[0]->T[0], * const DD = Tf[0]->T[1];
    ip_offset[0] = ipf_offset[0] = elem_offset[0] = 0;
    for (PetscInt grid=0 ; grid<num_grids ; grid++) {
      PetscInt nfloc = ctx->species_offset[grid+1] - ctx->species_offset[grid];
      elem_offset[grid+1] = elem_offset[grid] + numCells[grid];
      ip_offset[grid+1]   = ip_offset[grid]   + numCells[grid]*Nq;
      ipf_offset[grid+1]  = ipf_offset[grid]  + Nq*nfloc*numCells[grid];
    }
    IPf_sz_glb = ipf_offset[num_grids];
    IPf_sz_tot = IPf_sz_glb*ctx->batch_sz;
    if (shift==0.0) { /* compute dynamic data f and df and init data for Jacobian */
#if defined(PETSC_HAVE_THREADSAFETY)
      double         starttime, endtime;
      starttime = MPI_Wtime();
#endif
      ierr = PetscLogEventBegin(ctx->events[8],0,0,0,0);CHKERRQ(ierr);
      for (PetscInt fieldA=0;fieldA<ctx->num_species;fieldA++) {
        invMass[fieldA] = ctx->m_0/ctx->masses[fieldA];
        Eq_m[fieldA] = ctx->Ez * ctx->t_0 * ctx->charges[fieldA] / (ctx->v_0 * ctx->masses[fieldA]); /* normalize dimensionless */
        if (dim==2) Eq_m[fieldA] *=  2 * PETSC_PI; /* add the 2pi term that is not in Landau */
        nu_alpha[fieldA] = PetscSqr(ctx->charges[fieldA]/ctx->m_0)*ctx->m_0/ctx->masses[fieldA];
        nu_beta[fieldA] = PetscSqr(ctx->charges[fieldA]/ctx->epsilon0)*ctx->lnLam / (8*PETSC_PI) * ctx->t_0*ctx->n_0/PetscPowReal(ctx->v_0,3);
      }
      ierr = PetscMalloc4(IPf_sz_tot, &ff, IPf_sz_tot, &dudx, IPf_sz_tot, &dudy, dim==3 ? IPf_sz_tot : 0, &dudz);CHKERRQ(ierr);
      // F df/dx
      for (PetscInt tid = 0 ; tid < ctx->batch_sz*elem_offset[num_grids] ; tid++) { // for each element
        const PetscInt b_Nelem = elem_offset[num_grids], b_elem_idx = tid%b_Nelem, b_id = tid/b_Nelem; // b_id == OMP thd_id in batch
        // find my grid:
        PetscInt       grid = 0;
        while (b_elem_idx >= elem_offset[grid+1]) grid++; // yuck search for grid
        {
          const PetscInt     loc_nip = numCells[grid]*Nq, loc_Nf = ctx->species_offset[grid+1] - ctx->species_offset[grid], loc_elem = b_elem_idx - elem_offset[grid];
          const PetscInt     moffset = LAND_MOFFSET(b_id,grid,ctx->batch_sz,ctx->num_grids,ctx->mat_offset); //b_id*b_N + ctx->mat_offset[grid];
          PetscScalar        *coef, coef_buff[LANDAU_MAX_SPECIES*LANDAU_MAX_NQ];
          PetscReal          *invJe = &invJ_a[(ip_offset[grid] + loc_elem*Nq)*dim*dim]; // ingJ is static data on batch 0
          PetscInt           b,f,q;
          if (cellClosure) {
            coef = &cellClosure[b_id*IPf_sz_glb + ipf_offset[grid] + loc_elem*Nb*loc_Nf]; // this is const
          } else {
            coef = coef_buff;
            for (f = 0; f < loc_Nf; ++f) {
              LandauIdx *const Idxs = &maps[grid].gIdx[loc_elem][f][0];
              for (b = 0; b < Nb; ++b) {
                PetscInt idx = Idxs[b];
                if (idx >= 0) {
                  coef[f*Nb+b] = xdata[idx+moffset];
                } else {
                  idx = -idx - 1;
                  coef[f*Nb+b] = 0;
                  for (q = 0; q < maps[grid].num_face; q++) {
                    PetscInt    id = maps[grid].c_maps[idx][q].gid;
                    PetscScalar scale = maps[grid].c_maps[idx][q].scale;
                    coef[f*Nb+b] += scale*xdata[id+moffset];
                  }
                }
              }
            }
          }
          /* get f and df */
          for (PetscInt qi = 0; qi < Nq; qi++) {
            const PetscReal  *invJ = &invJe[qi*dim*dim];
            const PetscReal  *Bq = &BB[qi*Nb];
            const PetscReal  *Dq = &DD[qi*Nb*dim];
            PetscReal        u_x[LANDAU_DIM];
            /* get f & df */
            for (f = 0; f < loc_Nf; ++f) {
              const PetscInt idx = b_id*IPf_sz_glb + ipf_offset[grid] + f*loc_nip + loc_elem*Nq + qi;
              PetscInt       b, e;
              PetscReal      refSpaceDer[LANDAU_DIM];
              ff[idx] = 0.0;
              for (int d = 0; d < LANDAU_DIM; ++d) refSpaceDer[d] = 0.0;
              for (b = 0; b < Nb; ++b) {
                const PetscInt    cidx = b;
                ff[idx] += Bq[cidx]*PetscRealPart(coef[f*Nb+cidx]);
                for (int d = 0; d < dim; ++d) {
                  refSpaceDer[d] += Dq[cidx*dim+d]*PetscRealPart(coef[f*Nb+cidx]);
                }
              }
              for (int d = 0; d < LANDAU_DIM; ++d) {
                for (e = 0, u_x[d] = 0.0; e < LANDAU_DIM; ++e) {
                  u_x[d] += invJ[e*dim+d]*refSpaceDer[e];
                }
              }
              dudx[idx] = u_x[0];
              dudy[idx] = u_x[1];
 #if LANDAU_DIM==3
              dudz[idx] = u_x[2];
#endif
            }
          } // q
        } // grid
      } // grid*batch
      ierr = PetscLogEventEnd(ctx->events[8],0,0,0,0);CHKERRQ(ierr);
#if defined(PETSC_HAVE_THREADSAFETY)
      endtime = MPI_Wtime();
      if (ctx->stage) ctx->times[LANDAU_F_DF] += (endtime - starttime);
#endif
    } // Jacobian setup
    /* doit it */
    for (PetscInt tid = 0 ; tid < ctx->batch_sz*elem_offset[num_grids] ; tid++) { // for each element
      const PetscInt b_Nelem = elem_offset[num_grids];
      const PetscInt b_elem_idx = tid%b_Nelem, b_id = tid/b_Nelem;
      PetscInt       grid = 0;
#if defined(PETSC_HAVE_THREADSAFETY)
      double         starttime, endtime;
      starttime = MPI_Wtime();
#endif
      while (b_elem_idx >= elem_offset[grid+1]) grid++;
      {
        const PetscInt     loc_Nf = ctx->species_offset[grid+1] - ctx->species_offset[grid], loc_elem = b_elem_idx - elem_offset[grid];
        const PetscInt     moffset = LAND_MOFFSET(b_id,grid,ctx->batch_sz,ctx->num_grids,ctx->mat_offset)/* ; b_id*b_N + ctx->mat_offset[grid] */, totDim = loc_Nf*Nq, elemMatSize = totDim*totDim;
        PetscScalar        *elemMat;
         const PetscReal   *invJe = &invJ_a[(ip_offset[grid] + loc_elem*Nq)*dim*dim];
        ierr = PetscMalloc1(elemMatSize, &elemMat);CHKERRQ(ierr);
        ierr = PetscMemzero(elemMat, elemMatSize*sizeof(*elemMat));CHKERRQ(ierr);
        if (shift==0.0) { // Jacobian
          ierr = PetscLogEventBegin(ctx->events[4],0,0,0,0);CHKERRQ(ierr);
        } else {
          ierr = PetscLogEventBegin(ctx->events[16],0,0,0,0);CHKERRQ(ierr);
        }
        for (PetscInt qj = 0; qj < Nq; ++qj) {
          const PetscInt   jpidx_glb = ip_offset[grid] + qj + loc_elem * Nq;
          PetscReal        g0[LANDAU_MAX_SPECIES], g2[LANDAU_MAX_SPECIES][LANDAU_DIM], g3[LANDAU_MAX_SPECIES][LANDAU_DIM][LANDAU_DIM]; // could make a LANDAU_MAX_SPECIES_GRID ~ number of ions - 1
          PetscInt         d,d2,dp,d3,IPf_idx;
          if (shift==0.0) { // Jacobian
            const PetscReal * const invJj = &invJe[qj*dim*dim];
            PetscReal               gg2[LANDAU_MAX_SPECIES][LANDAU_DIM],gg3[LANDAU_MAX_SPECIES][LANDAU_DIM][LANDAU_DIM], gg2_temp[LANDAU_DIM], gg3_temp[LANDAU_DIM][LANDAU_DIM];
            const PetscReal         vj[3] = {xx[jpidx_glb], yy[jpidx_glb], zz ? zz[jpidx_glb] : 0}, wj = ww[jpidx_glb];
            // create g2 & g3
            for (d=0;d<LANDAU_DIM;d++) { // clear accumulation data D & K
              gg2_temp[d] = 0;
              for (d2=0;d2<LANDAU_DIM;d2++) gg3_temp[d][d2] = 0;
            }
            /* inner beta reduction */
            IPf_idx = 0;
            for (PetscInt grid_r = 0, f_off = 0, ipidx = 0; grid_r < ctx->num_grids ; grid_r++, f_off = ctx->species_offset[grid_r]) { // IPf_idx += nip_loc_r*Nfloc_r
              PetscInt  nip_loc_r = numCells[grid_r]*Nq, Nfloc_r = Nf[grid_r];
              for (PetscInt ei_r = 0, loc_fdf_idx = 0; ei_r < numCells[grid_r]; ++ei_r) {
                for (PetscInt qi = 0; qi < Nq; qi++, ipidx++, loc_fdf_idx++) {
                  const PetscReal wi = ww[ipidx], x = xx[ipidx], y = yy[ipidx];
                  PetscReal       temp1[3] = {0, 0, 0}, temp2 = 0;
#if LANDAU_DIM==2
                  PetscReal       Ud[2][2], Uk[2][2], mask = (PetscAbs(vj[0]-x) < 100*PETSC_SQRT_MACHINE_EPSILON && PetscAbs(vj[1]-y) < 100*PETSC_SQRT_MACHINE_EPSILON) ? 0. : 1.;
                  LandauTensor2D(vj, x, y, Ud, Uk, mask);
#else
                  PetscReal U[3][3], z = zz[ipidx], mask = (PetscAbs(vj[0]-x) < 100*PETSC_SQRT_MACHINE_EPSILON && PetscAbs(vj[1]-y) < 100*PETSC_SQRT_MACHINE_EPSILON && PetscAbs(vj[2]-z) < 100*PETSC_SQRT_MACHINE_EPSILON) ? 0. : 1.;
                  if (ctx->use_relativistic_corrections) {
                    LandauTensor3DRelativistic(vj, x, y, z, U, mask, C_0(ctx->v_0));
                  } else {
                    LandauTensor3D(vj, x, y, z, U, mask);
                  }
#endif
                  for (int f = 0; f < Nfloc_r ; ++f) {
                    const PetscInt idx = b_id*IPf_sz_glb + ipf_offset[grid_r] + f*nip_loc_r + ei_r*Nq + qi;  // IPf_idx + f*nip_loc_r + loc_fdf_idx;
                    temp1[0] += dudx[idx]*nu_beta[f+f_off]*invMass[f+f_off];
                    temp1[1] += dudy[idx]*nu_beta[f+f_off]*invMass[f+f_off];
#if LANDAU_DIM==3
                    temp1[2] += dudz[idx]*nu_beta[f+f_off]*invMass[f+f_off];
#endif
                    temp2    += ff[idx]*nu_beta[f+f_off];
                  }
                  temp1[0] *= wi;
                  temp1[1] *= wi;
#if LANDAU_DIM==3
                  temp1[2] *= wi;
#endif
                  temp2    *= wi;
#if LANDAU_DIM==2
                  for (d2 = 0; d2 < 2; d2++) {
                    for (d3 = 0; d3 < 2; ++d3) {
                      /* K = U * grad(f): g2=e: i,A */
                      gg2_temp[d2] += Uk[d2][d3]*temp1[d3];
                      /* D = -U * (I \kron (fx)): g3=f: i,j,A */
                      gg3_temp[d2][d3] += Ud[d2][d3]*temp2;
                    }
                  }
#else
                  for (d2 = 0; d2 < 3; ++d2) {
                    for (d3 = 0; d3 < 3; ++d3) {
                      /* K = U * grad(f): g2 = e: i,A */
                      gg2_temp[d2] += U[d2][d3]*temp1[d3];
                      /* D = -U * (I \kron (fx)): g3 = f: i,j,A */
                      gg3_temp[d2][d3] += U[d2][d3]*temp2;
                    }
                  }
#endif
                } // qi
              } // ei_r
              IPf_idx += nip_loc_r*Nfloc_r;
            } /* grid_r - IPs */
            PetscCheckFalse(IPf_idx != IPf_sz_glb,PETSC_COMM_SELF, PETSC_ERR_PLIB, "IPf_idx != IPf_sz %D %D",IPf_idx,IPf_sz_glb);
            // add alpha and put in gg2/3
            for (PetscInt fieldA = 0, f_off = ctx->species_offset[grid]; fieldA < loc_Nf; ++fieldA) {
              for (d2 = 0; d2 < dim; d2++) {
                gg2[fieldA][d2] = gg2_temp[d2]*nu_alpha[fieldA+f_off];
                for (d3 = 0; d3 < dim; d3++) {
                  gg3[fieldA][d2][d3] = -gg3_temp[d2][d3]*nu_alpha[fieldA+f_off]*invMass[fieldA+f_off];
                }
              }
            }
            /* add electric field term once per IP */
            for (PetscInt fieldA = 0, f_off = ctx->species_offset[grid] ; fieldA < loc_Nf; ++fieldA) {
              gg2[fieldA][dim-1] += Eq_m[fieldA+f_off];
            }
            /* Jacobian transform - g2, g3 */
            for (PetscInt fieldA = 0; fieldA < loc_Nf; ++fieldA) {
              for (d = 0; d < dim; ++d) {
                g2[fieldA][d] = 0.0;
                for (d2 = 0; d2 < dim; ++d2) {
                  g2[fieldA][d] += invJj[d*dim+d2]*gg2[fieldA][d2];
                  g3[fieldA][d][d2] = 0.0;
                  for (d3 = 0; d3 < dim; ++d3) {
                    for (dp = 0; dp < dim; ++dp) {
                      g3[fieldA][d][d2] += invJj[d*dim + d3]*gg3[fieldA][d3][dp]*invJj[d2*dim + dp];
                    }
                  }
                  g3[fieldA][d][d2] *= wj;
                }
                g2[fieldA][d] *= wj;
              }
            }
          } else { // mass
            PetscReal wj = ww[jpidx_glb];
            /* Jacobian transform - g0 */
            for (PetscInt fieldA = 0; fieldA < loc_Nf ; ++fieldA) {
              if (dim==2) {
                g0[fieldA] = wj * shift * 2. * PETSC_PI; // move this to below and remove g0
              } else {
                g0[fieldA] = wj * shift; // move this to below and remove g0
              }
            }
          }
          /* FE matrix construction */
          {
            PetscInt  fieldA,d,f,d2,g;
            const PetscReal *BJq = &BB[qj*Nb], *DIq = &DD[qj*Nb*dim];
            /* assemble - on the diagonal (I,I) */
            for (fieldA = 0; fieldA < loc_Nf ; fieldA++) {
              for (f = 0; f < Nb ; f++) {
                const PetscInt i = fieldA*Nb + f; /* Element matrix row */
                for (g = 0; g < Nb; ++g) {
                  const PetscInt j    = fieldA*Nb + g; /* Element matrix column */
                  const PetscInt fOff = i*totDim + j;
                  if (shift==0.0) {
                    for (d = 0; d < dim; ++d) {
                      elemMat[fOff] += DIq[f*dim+d]*g2[fieldA][d]*BJq[g];
                      for (d2 = 0; d2 < dim; ++d2) {
                        elemMat[fOff] += DIq[f*dim + d]*g3[fieldA][d][d2]*DIq[g*dim + d2];
                      }
                    }
                  } else { // mass
                    elemMat[fOff] += BJq[f]*g0[fieldA]*BJq[g];
                  }
                }
              }
            }
          }
        } /* qj loop */
        if (shift==0.0) { // Jacobian
          ierr = PetscLogEventEnd(ctx->events[4],0,0,0,0);CHKERRQ(ierr);
        } else {
          ierr = PetscLogEventEnd(ctx->events[16],0,0,0,0);CHKERRQ(ierr);
        }
#if defined(PETSC_HAVE_THREADSAFETY)
        endtime = MPI_Wtime();
        if (ctx->stage) ctx->times[LANDAU_KERNEL] += (endtime - starttime);
#endif
        /* assemble matrix */
        if (!container) {
          PetscInt cStart;
          ierr = PetscLogEventBegin(ctx->events[6],0,0,0,0);CHKERRQ(ierr);
          ierr = DMPlexGetHeightStratum(ctx->plex[grid], 0, &cStart, NULL);CHKERRQ(ierr);
          ierr = DMPlexMatSetClosure(ctx->plex[grid], section[grid], globsection[grid], subJ[ LAND_PACK_IDX(b_id,grid) ], loc_elem + cStart, elemMat, ADD_VALUES);CHKERRQ(ierr);
          ierr = PetscLogEventEnd(ctx->events[6],0,0,0,0);CHKERRQ(ierr);
        } else {  // GPU like assembly for debugging
          PetscInt      fieldA,idx,q,f,g,d,nr,nc,rows0[LANDAU_MAX_Q_FACE]={0},cols0[LANDAU_MAX_Q_FACE]={0},rows[LANDAU_MAX_Q_FACE],cols[LANDAU_MAX_Q_FACE];
          PetscScalar   vals[LANDAU_MAX_Q_FACE*LANDAU_MAX_Q_FACE]={0},row_scale[LANDAU_MAX_Q_FACE]={0},col_scale[LANDAU_MAX_Q_FACE]={0};
          /* assemble - from the diagonal (I,I) in this format for DMPlexMatSetClosure */
          for (fieldA = 0; fieldA < loc_Nf ; fieldA++) {
            LandauIdx *const Idxs = &maps[grid].gIdx[loc_elem][fieldA][0];
            for (f = 0; f < Nb ; f++) {
              idx = Idxs[f];
              if (idx >= 0) {
                nr = 1;
                rows0[0] = idx;
                row_scale[0] = 1.;
              } else {
                idx = -idx - 1;
                nr = maps[grid].num_face;
                for (q = 0; q < maps[grid].num_face; q++) {
                  rows0[q]     = maps[grid].c_maps[idx][q].gid;
                  row_scale[q] = maps[grid].c_maps[idx][q].scale;
                }
              }
              for (g = 0; g < Nb; ++g) {
                idx = Idxs[g];
                if (idx >= 0) {
                  nc = 1;
                  cols0[0] = idx;
                  col_scale[0] = 1.;
                } else {
                  idx = -idx - 1;
                  nc = maps[grid].num_face;
                  for (q = 0; q < maps[grid].num_face; q++) {
                    cols0[q]     = maps[grid].c_maps[idx][q].gid;
                    col_scale[q] = maps[grid].c_maps[idx][q].scale;
                  }
                }
                const PetscInt    i = fieldA*Nb + f; /* Element matrix row */
                const PetscInt    j = fieldA*Nb + g; /* Element matrix column */
                const PetscScalar Aij = elemMat[i*totDim + j];
                for (q = 0; q < nr; q++) rows[q] = rows0[q] + moffset;
                for (d = 0; d < nc; d++) cols[d] = cols0[d] + moffset;
                for (q = 0; q < nr; q++) {
                  for (d = 0; d < nc; d++) {
                    vals[q*nc + d] = row_scale[q]*col_scale[d]*Aij;
                  }
                }
                ierr = MatSetValues(JacP,nr,rows,nc,cols,vals,ADD_VALUES);CHKERRQ(ierr);
              }
            }
          }
        }
        if (loc_elem==-1) {
          PetscErrorCode    ierr2;
          ierr2 = PetscPrintf(ctx->comm,"CPU Element matrix\n");CHKERRQ(ierr2);
          for (int d = 0; d < totDim; ++d) {
            for (int f = 0; f < totDim; ++f) {ierr2 = PetscPrintf(ctx->comm," %12.5e",  PetscRealPart(elemMat[d*totDim + f]));CHKERRQ(ierr2);}
            ierr2 = PetscPrintf(ctx->comm,"\n");CHKERRQ(ierr2);
          }
          exit(12);
        }
        ierr = PetscFree(elemMat);CHKERRQ(ierr);
      } /* grid */
    } /* outer element & batch loop */
    if (shift==0.0) { // mass
      ierr = PetscFree4(ff, dudx, dudy, dudz);CHKERRQ(ierr);
    }
    if (!container) {   // move nest matrix to global JacP
      for (PetscInt b_id = 0 ; b_id < ctx->batch_sz ; b_id++) { // OpenMP
        for (PetscInt grid=0 ; grid<ctx->num_grids ; grid++) {
          const PetscInt    moffset = LAND_MOFFSET(b_id,grid,ctx->batch_sz,ctx->num_grids,ctx->mat_offset); // b_id*b_N + ctx->mat_offset[grid];
          PetscInt          nloc, nzl, colbuf[1024], row;
          const PetscInt    *cols;
          const PetscScalar *vals;
          Mat               B = subJ[ LAND_PACK_IDX(b_id,grid) ];
          ierr = MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
          ierr = MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
          ierr = MatGetSize(B, &nloc, NULL);CHKERRQ(ierr);
          for (int i=0 ; i<nloc ; i++) {
            ierr = MatGetRow(B,i,&nzl,&cols,&vals);CHKERRQ(ierr);
            PetscCheck(nzl<=1024,PetscObjectComm((PetscObject) B), PETSC_ERR_PLIB, "Row too big: %D",nzl);
            for (int j=0; j<nzl; j++) colbuf[j] = moffset + cols[j];
            row = moffset + i;
            ierr = MatSetValues(JacP,1,&row,nzl,colbuf,vals,ADD_VALUES);CHKERRQ(ierr);
            ierr = MatRestoreRow(B,i,&nzl,&cols,&vals);CHKERRQ(ierr);
          }
          ierr = MatDestroy(&B);CHKERRQ(ierr);
        }
      }
    }
  } /* CPU version */
  ierr = MatAssemblyBegin(JacP, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(JacP, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  /* clean up */
  if (cellClosure) {
    ierr = PetscFree(cellClosure);CHKERRQ(ierr);
  }
  if (xdata) {
    ierr = VecRestoreArrayReadAndMemType(a_X,&xdata);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#if defined(LANDAU_ADD_BCS)
static void zero_bc(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                    const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                    PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar uexact[])
{
  uexact[0] = 0;
}
#endif

#define MATVEC2(__a,__x,__p) {int i,j; for (i=0.; i<2; i++) {__p[i] = 0; for (j=0.; j<2; j++) __p[i] += __a[i][j]*__x[j]; }}
static void CircleInflate(PetscReal r1, PetscReal r2, PetscReal r0, PetscInt num_sections, PetscReal x, PetscReal y,
                          PetscReal *outX, PetscReal *outY)
{
  PetscReal rr = PetscSqrtReal(x*x + y*y), outfact, efact;
  if (rr < r1 + PETSC_SQRT_MACHINE_EPSILON) {
    *outX = x; *outY = y;
  } else {
    const PetscReal xy[2] = {x,y}, sinphi=y/rr, cosphi=x/rr;
    PetscReal       cth,sth,xyprime[2],Rth[2][2],rotcos,newrr;
    if (num_sections==2) {
      rotcos = 0.70710678118654;
      outfact = 1.5; efact = 2.5;
      /* rotate normalized vector into [-pi/4,pi/4) */
      if (sinphi >= 0.) {         /* top cell, -pi/2 */
        cth = 0.707106781186548; sth = -0.707106781186548;
      } else {                    /* bottom cell -pi/8 */
        cth = 0.707106781186548; sth = .707106781186548;
      }
    } else if (num_sections==3) {
      rotcos = 0.86602540378443;
      outfact = 1.5; efact = 2.5;
      /* rotate normalized vector into [-pi/6,pi/6) */
      if (sinphi >= 0.5) {         /* top cell, -pi/3 */
        cth = 0.5; sth = -0.866025403784439;
      } else if (sinphi >= -.5) {  /* mid cell 0 */
        cth = 1.; sth = .0;
      } else { /* bottom cell +pi/3 */
        cth = 0.5; sth = 0.866025403784439;
      }
    } else if (num_sections==4) {
      rotcos = 0.9238795325112;
      outfact = 1.5; efact = 3;
      /* rotate normalized vector into [-pi/8,pi/8) */
      if (sinphi >= 0.707106781186548) {         /* top cell, -3pi/8 */
        cth = 0.38268343236509; sth = -0.923879532511287;
      } else if (sinphi >= 0.) {                 /* mid top cell -pi/8 */
        cth = 0.923879532511287; sth = -.38268343236509;
      } else if (sinphi >= -0.707106781186548) { /* mid bottom cell + pi/8 */
        cth = 0.923879532511287; sth = 0.38268343236509;
      } else {                                   /* bottom cell + 3pi/8 */
        cth = 0.38268343236509; sth = .923879532511287;
      }
    } else {
      cth = 0.; sth = 0.; rotcos = 0; efact = 0;
    }
    Rth[0][0] = cth; Rth[0][1] =-sth;
    Rth[1][0] = sth; Rth[1][1] = cth;
    MATVEC2(Rth,xy,xyprime);
    if (num_sections==2) {
      newrr = xyprime[0]/rotcos;
    } else {
      PetscReal newcosphi=xyprime[0]/rr, rin = r1, rout = rr - rin;
      PetscReal routmax = r0*rotcos/newcosphi - rin, nroutmax = r0 - rin, routfrac = rout/routmax;
      newrr = rin + routfrac*nroutmax;
    }
    *outX = cosphi*newrr; *outY = sinphi*newrr;
    /* grade */
    PetscReal fact,tt,rs,re, rr = PetscSqrtReal(PetscSqr(*outX) + PetscSqr(*outY));
    if (rr > r2) { rs = r2; re = r0; fact = outfact;} /* outer zone */
    else {         rs = r1; re = r2; fact = efact;} /* electron zone */
    tt = (rs + PetscPowReal((rr - rs)/(re - rs),fact) * (re-rs)) / rr;
    *outX *= tt;
    *outY *= tt;
  }
}

static PetscErrorCode GeometryDMLandau(DM base, PetscInt point, PetscInt dim, const PetscReal abc[], PetscReal xyz[], void *a_ctx)
{
  LandauCtx   *ctx = (LandauCtx*)a_ctx;
  PetscReal   r = abc[0], z = abc[1];
  if (ctx->inflate) {
    PetscReal absR, absZ;
    absR = PetscAbs(r);
    absZ = PetscAbs(z);
    CircleInflate(ctx->i_radius[0],ctx->e_radius,ctx->radius[0],ctx->num_sections,absR,absZ,&absR,&absZ); // wrong: how do I know what grid I am on?
    r = (r > 0) ? absR : -absR;
    z = (z > 0) ? absZ : -absZ;
  }
  xyz[0] = r;
  xyz[1] = z;
  if (dim==3) xyz[2] = abc[2];

  PetscFunctionReturn(0);
}

/* create DMComposite of meshes for each species group */
static PetscErrorCode LandauDMCreateVMeshes(MPI_Comm comm_self, const PetscInt dim, const char prefix[], LandauCtx *ctx, DM pack)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  { /* p4est, quads */
    /* Create plex mesh of Landau domain */
    for (PetscInt grid=0;grid<ctx->num_grids;grid++) {
      PetscReal radius = ctx->radius[grid];
      if (!ctx->sphere) {
        PetscInt       cells[] = {2,2,2};
        PetscReal      lo[] = {-radius,-radius,-radius}, hi[] = {radius,radius,radius};
        DMBoundaryType periodicity[3] = {DM_BOUNDARY_NONE, dim==2 ? DM_BOUNDARY_NONE : DM_BOUNDARY_NONE, DM_BOUNDARY_NONE};
        if (dim==2) { lo[0] = 0; cells[0] /* = cells[1] */ = 1; }
        ierr = DMPlexCreateBoxMesh(comm_self, dim, PETSC_FALSE, cells, lo, hi, periodicity, PETSC_TRUE, &ctx->plex[grid]);CHKERRQ(ierr); // todo: make composite and create dm[grid] here
        ierr = DMLocalizeCoordinates(ctx->plex[grid]);CHKERRQ(ierr); /* needed for periodic */
        if (dim==3) {ierr = PetscObjectSetName((PetscObject) ctx->plex[grid], "cube");CHKERRQ(ierr);}
        else {ierr = PetscObjectSetName((PetscObject) ctx->plex[grid], "half-plane");CHKERRQ(ierr);}
      } else if (dim==2) { // sphere is all wrong. should just have one inner radius
        PetscInt       numCells,cells[16][4],i,j;
        PetscInt       numVerts;
        PetscReal      inner_radius1 = ctx->i_radius[grid], inner_radius2 = ctx->e_radius;
        PetscReal      *flatCoords = NULL;
        PetscInt       *flatCells = NULL, *pcell;
        if (ctx->num_sections==2) {
#if 1
          numCells = 5;
          numVerts = 10;
          int cells2[][4] = { {0,1,4,3},
                              {1,2,5,4},
                              {3,4,7,6},
                              {4,5,8,7},
                              {6,7,8,9} };
          for (i = 0; i < numCells; i++) for (j = 0; j < 4; j++) cells[i][j] = cells2[i][j];
          ierr = PetscMalloc2(numVerts * 2, &flatCoords, numCells * 4, &flatCells);CHKERRQ(ierr);
          {
            PetscReal (*coords)[2] = (PetscReal (*) [2]) flatCoords;
            for (j = 0; j < numVerts-1; j++) {
              PetscReal z, r, theta = -PETSC_PI/2 + (j%3) * PETSC_PI/2;
              PetscReal rad = (j >= 6) ? inner_radius1 : (j >= 3) ? inner_radius2 : ctx->radius[grid];
              z = rad * PetscSinReal(theta);
              coords[j][1] = z;
              r = rad * PetscCosReal(theta);
              coords[j][0] = r;
            }
            coords[numVerts-1][0] = coords[numVerts-1][1] = 0;
          }
#else
          numCells = 4;
          numVerts = 8;
          static int     cells2[][4] = {{0,1,2,3},
                                        {4,5,1,0},
                                        {5,6,2,1},
                                        {6,7,3,2}};
          for (i = 0; i < numCells; i++) for (j = 0; j < 4; j++) cells[i][j] = cells2[i][j];
          ierr = loc2(numVerts * 2, &flatCoords, numCells * 4, &flatCells);CHKERRQ(ierr);
          {
            PetscReal (*coords)[2] = (PetscReal (*) [2]) flatCoords;
            PetscInt j;
            for (j = 0; j < 8; j++) {
              PetscReal z, r;
              PetscReal theta = -PETSC_PI/2 + (j%4) * PETSC_PI/3.;
              PetscReal rad = ctx->radius[grid] * ((j < 4) ? 0.5 : 1.0);
              z = rad * PetscSinReal(theta);
              coords[j][1] = z;
              r = rad * PetscCosReal(theta);
              coords[j][0] = r;
            }
          }
#endif
        } else if (ctx->num_sections==3) {
          numCells = 7;
          numVerts = 12;
          int cells2[][4] = { {0,1,5,4},
                              {1,2,6,5},
                              {2,3,7,6},
                              {4,5,9,8},
                              {5,6,10,9},
                              {6,7,11,10},
                              {8,9,10,11} };
          for (i = 0; i < numCells; i++) for (j = 0; j < 4; j++) cells[i][j] = cells2[i][j];
          ierr = PetscMalloc2(numVerts * 2, &flatCoords, numCells * 4, &flatCells);CHKERRQ(ierr);
          {
            PetscReal (*coords)[2] = (PetscReal (*) [2]) flatCoords;
            for (j = 0; j < numVerts; j++) {
              PetscReal z, r, theta = -PETSC_PI/2 + (j%4) * PETSC_PI/3;
              PetscReal rad = (j >= 8) ? inner_radius1 : (j >= 4) ? inner_radius2 : ctx->radius[grid];
              z = rad * PetscSinReal(theta);
              coords[j][1] = z;
              r = rad * PetscCosReal(theta);
              coords[j][0] = r;
            }
          }
        } else if (ctx->num_sections==4) {
          numCells = 10;
          numVerts = 16;
          int cells2[][4] = { {0,1,6,5},
                              {1,2,7,6},
                              {2,3,8,7},
                              {3,4,9,8},
                              {5,6,11,10},
                              {6,7,12,11},
                              {7,8,13,12},
                              {8,9,14,13},
                              {10,11,12,15},
                              {12,13,14,15}};
          for (i = 0; i < numCells; i++) for (j = 0; j < 4; j++) cells[i][j] = cells2[i][j];
          ierr = PetscMalloc2(numVerts * 2, &flatCoords, numCells * 4, &flatCells);CHKERRQ(ierr);
          {
            PetscReal (*coords)[2] = (PetscReal (*) [2]) flatCoords;
            for (j = 0; j < numVerts-1; j++) {
              PetscReal z, r, theta = -PETSC_PI/2 + (j%5) * PETSC_PI/4;
              PetscReal rad = (j >= 10) ? inner_radius1 : (j >= 5) ? inner_radius2 : ctx->radius[grid];
              z = rad * PetscSinReal(theta);
              coords[j][1] = z;
              r = rad * PetscCosReal(theta);
              coords[j][0] = r;
            }
            coords[numVerts-1][0] = coords[numVerts-1][1] = 0;
          }
        } else {
          numCells = 0;
          numVerts = 0;
        }
        for (j = 0, pcell = flatCells; j < numCells; j++, pcell += 4) {
          pcell[0] = cells[j][0]; pcell[1] = cells[j][1];
          pcell[2] = cells[j][2]; pcell[3] = cells[j][3];
        }
        ierr = DMPlexCreateFromCellListPetsc(comm_self,2,numCells,numVerts,4,ctx->interpolate,flatCells,2,flatCoords,&ctx->plex[grid]);CHKERRQ(ierr);
        ierr = PetscFree2(flatCoords,flatCells);CHKERRQ(ierr);
        ierr = PetscObjectSetName((PetscObject) ctx->plex[grid], "semi-circle");CHKERRQ(ierr);
      } else SETERRQ(ctx->comm, PETSC_ERR_PLIB, "Velocity space meshes does not support cubed sphere");

      ierr = DMSetFromOptions(ctx->plex[grid]);CHKERRQ(ierr);
    } // grid loop
    ierr = PetscObjectSetOptionsPrefix((PetscObject)pack,prefix);CHKERRQ(ierr);
    ierr = DMSetFromOptions(pack);CHKERRQ(ierr);

    { /* convert to p4est (or whatever), wait for discretization to create pack */
      char      convType[256];
      PetscBool flg;
      ierr = PetscOptionsBegin(ctx->comm, prefix, "Mesh conversion options", "DMPLEX");CHKERRQ(ierr);
      ierr = PetscOptionsFList("-dm_landau_type","Convert DMPlex to another format (p4est)","plexland.c",DMList,DMPLEX,convType,256,&flg);CHKERRQ(ierr);
      ierr = PetscOptionsEnd();CHKERRQ(ierr);
      if (flg) {
        ctx->use_p4est = PETSC_TRUE; /* flag for Forest */
        for (PetscInt grid=0;grid<ctx->num_grids;grid++) {
          DM dmforest;
          ierr = DMConvert(ctx->plex[grid],convType,&dmforest);CHKERRQ(ierr);
          if (dmforest) {
            PetscBool isForest;
            ierr = PetscObjectSetOptionsPrefix((PetscObject)dmforest,prefix);CHKERRQ(ierr);
            ierr = DMIsForest(dmforest,&isForest);CHKERRQ(ierr);
            if (isForest) {
              if (ctx->sphere && ctx->inflate) {
                ierr = DMForestSetBaseCoordinateMapping(dmforest,GeometryDMLandau,ctx);CHKERRQ(ierr);
              }
              ierr = DMDestroy(&ctx->plex[grid]);CHKERRQ(ierr);
              ctx->plex[grid] = dmforest; // Forest for adaptivity
            } else SETERRQ(ctx->comm, PETSC_ERR_PLIB, "Converted to non Forest?");
          } else SETERRQ(ctx->comm, PETSC_ERR_PLIB, "Convert failed?");
        }
      } else ctx->use_p4est = PETSC_FALSE; /* flag for Forest */
    }
  } /* non-file */
  ierr = DMSetDimension(pack, dim);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) pack, "Mesh");CHKERRQ(ierr);
  ierr = DMSetApplicationContext(pack, ctx);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode SetupDS(DM pack, PetscInt dim, PetscInt grid, LandauCtx *ctx)
{
  PetscErrorCode  ierr;
  PetscInt        ii,i0;
  char            buf[256];
  PetscSection    section;

  PetscFunctionBegin;
  for (ii = ctx->species_offset[grid], i0 = 0 ; ii < ctx->species_offset[grid+1] ; ii++, i0++) {
    if (ii==0) ierr = PetscSNPrintf(buf, 256, "e");
    else {ierr = PetscSNPrintf(buf, 256, "i%D", ii);CHKERRQ(ierr);}
    /* Setup Discretization - FEM */
    ierr = PetscFECreateDefault(PETSC_COMM_SELF, dim, 1, PETSC_FALSE, NULL, PETSC_DECIDE, &ctx->fe[ii]);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) ctx->fe[ii], buf);CHKERRQ(ierr);
    ierr = DMSetField(ctx->plex[grid], i0, NULL, (PetscObject) ctx->fe[ii]);CHKERRQ(ierr);
  }
  ierr = DMCreateDS(ctx->plex[grid]);CHKERRQ(ierr);
  ierr = DMGetSection(ctx->plex[grid], &section);CHKERRQ(ierr);
  for (PetscInt ii = ctx->species_offset[grid], i0 = 0 ; ii < ctx->species_offset[grid+1] ; ii++, i0++) {
    if (ii==0) ierr = PetscSNPrintf(buf, 256, "se");
    else ierr = PetscSNPrintf(buf, 256, "si%D", ii);
    ierr = PetscSectionSetComponentName(section, i0, 0, buf);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* Define a Maxwellian function for testing out the operator. */

/* Using cartesian velocity space coordinates, the particle */
/* density, [1/m^3], is defined according to */

/* $$ n=\int_{R^3} dv^3 \left(\frac{m}{2\pi T}\right)^{3/2}\exp [- mv^2/(2T)] $$ */

/* Using some constant, c, we normalize the velocity vector into a */
/* dimensionless variable according to v=c*x. Thus the density, $n$, becomes */

/* $$ n=\int_{R^3} dx^3 \left(\frac{mc^2}{2\pi T}\right)^{3/2}\exp [- mc^2/(2T)*x^2] $$ */

/* Defining $\theta=2T/mc^2$, we thus find that the probability density */
/* for finding the particle within the interval in a box dx^3 around x is */

/* f(x;\theta)=\left(\frac{1}{\pi\theta}\right)^{3/2} \exp [ -x^2/\theta ] */

typedef struct {
  PetscReal v_0;
  PetscReal kT_m;
  PetscReal n;
  PetscReal shift;
} MaxwellianCtx;

static PetscErrorCode maxwellian(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf_dummy, PetscScalar *u, void *actx)
{
  MaxwellianCtx *mctx = (MaxwellianCtx*)actx;
  PetscInt      i;
  PetscReal     v2 = 0, theta = 2*mctx->kT_m/(mctx->v_0*mctx->v_0); /* theta = 2kT/mc^2 */
  PetscFunctionBegin;
  /* compute the exponents, v^2 */
  for (i = 0; i < dim; ++i) v2 += x[i]*x[i];
  /* evaluate the Maxwellian */
  u[0] = mctx->n*PetscPowReal(PETSC_PI*theta,-1.5)*(PetscExpReal(-v2/theta));
  if (mctx->shift!=0.) {
    v2 = 0;
    for (i = 0; i < dim-1; ++i) v2 += x[i]*x[i];
    v2 += (x[dim-1]-mctx->shift)*(x[dim-1]-mctx->shift);
    /* evaluate the shifted Maxwellian */
    u[0] += mctx->n*PetscPowReal(PETSC_PI*theta,-1.5)*(PetscExpReal(-v2/theta));
  }
  PetscFunctionReturn(0);
}

/*@
 LandauAddMaxwellians - Add a Maxwellian distribution to a state

 Collective on X

 Input Parameters:
 .   dm - The mesh (local)
 +   time - Current time
 -   temps - Temperatures of each species (global)
 .   ns - Number density of each species (global)
 -   grid - index into current grid - just used for offset into temp and ns
 +   actx - Landau context

 Output Parameter:
 .   X  - The state (local to this grid)

 Level: beginner

 .keywords: mesh
 .seealso: LandauCreateVelocitySpace()
 @*/
PetscErrorCode LandauAddMaxwellians(DM dm, Vec X, PetscReal time, PetscReal temps[], PetscReal ns[], PetscInt grid, PetscInt b_id, void *actx)
{
  LandauCtx      *ctx = (LandauCtx*)actx;
  PetscErrorCode (*initu[LANDAU_MAX_SPECIES])(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar [], void *);
  PetscErrorCode ierr,ii,i0;
  PetscInt       dim;
  MaxwellianCtx  *mctxs[LANDAU_MAX_SPECIES], data[LANDAU_MAX_SPECIES];

  PetscFunctionBegin;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  if (!ctx) { ierr = DMGetApplicationContext(dm, &ctx);CHKERRQ(ierr); }
  for (ii = ctx->species_offset[grid], i0 = 0 ; ii < ctx->species_offset[grid+1] ; ii++, i0++) {
    mctxs[i0] = &data[i0];
    data[i0].v_0 = ctx->v_0; // v_0 same for all grids
    data[i0].kT_m = ctx->k*temps[ii]/ctx->masses[ii]; /* kT/m */
    data[i0].n = ns[ii] * (1+(double)b_id/100.0); // make solves a little different to mimic application, n[0] use for Conner-Hastie
    initu[i0] = maxwellian;
    data[i0].shift = 0;
  }
  data[0].shift = ctx->electronShift;
  /* need to make ADD_ALL_VALUES work - TODO */
  ierr = DMProjectFunction(dm, time, initu, (void**)mctxs, INSERT_ALL_VALUES, X);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
 LandauSetInitialCondition - Addes Maxwellians with context

 Collective on X

 Input Parameters:
 .   dm - The mesh
 -   grid - index into current grid - just used for offset into temp and ns
 +   actx - Landau context with T and n

 Output Parameter:
 .   X  - The state

 Level: beginner

 .keywords: mesh
 .seealso: LandauCreateVelocitySpace(), LandauAddMaxwellians()
 */
static PetscErrorCode LandauSetInitialCondition(DM dm, Vec X, PetscInt grid, PetscInt b_id, void *actx)
{
  LandauCtx        *ctx = (LandauCtx*)actx;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (!ctx) { ierr = DMGetApplicationContext(dm, &ctx);CHKERRQ(ierr); }
  ierr = VecZeroEntries(X);CHKERRQ(ierr);
  ierr = LandauAddMaxwellians(dm, X, 0.0, ctx->thermal_temps, ctx->n, grid, b_id, ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

// adapt a level once. Forest in/out
static PetscErrorCode adaptToleranceFEM(PetscFE fem, Vec sol, PetscInt type, PetscInt grid, LandauCtx *ctx, DM *newForest)
{
  DM               forest, plex, adaptedDM = NULL;
  PetscDS          prob;
  PetscBool        isForest;
  PetscQuadrature  quad;
  PetscInt         Nq, *Nb, cStart, cEnd, c, dim, qj, k;
  DMLabel          adaptLabel = NULL;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  forest = ctx->plex[grid];
  ierr = DMCreateDS(forest);CHKERRQ(ierr);
  ierr = DMGetDS(forest, &prob);CHKERRQ(ierr);
  ierr = DMGetDimension(forest, &dim);CHKERRQ(ierr);
  ierr = DMIsForest(forest, &isForest);CHKERRQ(ierr);
  PetscCheckFalse(!isForest,ctx->comm,PETSC_ERR_ARG_WRONG,"! Forest");
  ierr = DMConvert(forest, DMPLEX, &plex);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(plex,0,&cStart,&cEnd);CHKERRQ(ierr);
  ierr = DMLabelCreate(PETSC_COMM_SELF,"adapt",&adaptLabel);CHKERRQ(ierr);
  ierr = PetscFEGetQuadrature(fem, &quad);CHKERRQ(ierr);
  ierr = PetscQuadratureGetData(quad, NULL, NULL, &Nq, NULL, NULL);CHKERRQ(ierr);
  PetscCheckFalse(Nq >LANDAU_MAX_NQ,ctx->comm,PETSC_ERR_ARG_WRONG,"Order too high. Nq = %D > LANDAU_MAX_NQ (%D)",Nq,LANDAU_MAX_NQ);
  ierr = PetscDSGetDimensions(prob, &Nb);CHKERRQ(ierr);
  if (type==4) {
    for (c = cStart; c < cEnd; c++) {
      ierr = DMLabelSetValue(adaptLabel, c, DM_ADAPT_REFINE);CHKERRQ(ierr);
    }
    ierr = PetscInfo(sol, "Phase:%s: Uniform refinement\n","adaptToleranceFEM");CHKERRQ(ierr);
  } else if (type==2) {
    PetscInt  rCellIdx[8], eCellIdx[64], iCellIdx[64], eMaxIdx = -1, iMaxIdx = -1, nr = 0, nrmax = (dim==3) ? 8 : 2;
    PetscReal minRad = PETSC_INFINITY, r, eMinRad = PETSC_INFINITY, iMinRad = PETSC_INFINITY;
    for (c = 0; c < 64; c++) { eCellIdx[c] = iCellIdx[c] = -1; }
    for (c = cStart; c < cEnd; c++) {
      PetscReal    tt, v0[LANDAU_MAX_NQ*3], detJ[LANDAU_MAX_NQ];
      ierr = DMPlexComputeCellGeometryFEM(plex, c, quad, v0, NULL, NULL, detJ);CHKERRQ(ierr);
      for (qj = 0; qj < Nq; ++qj) {
        tt = PetscSqr(v0[dim*qj+0]) + PetscSqr(v0[dim*qj+1]) + PetscSqr(((dim==3) ? v0[dim*qj+2] : 0));
        r = PetscSqrtReal(tt);
        if (r < minRad - PETSC_SQRT_MACHINE_EPSILON*10.) {
          minRad = r;
          nr = 0;
          rCellIdx[nr++]= c;
          ierr = PetscInfo(sol, "\t\tPhase: adaptToleranceFEM Found first inner r=%e, cell %D, qp %D/%D\n", r, c, qj+1, Nq);CHKERRQ(ierr);
        } else if ((r-minRad) < PETSC_SQRT_MACHINE_EPSILON*100. && nr < nrmax) {
          for (k=0;k<nr;k++) if (c == rCellIdx[k]) break;
          if (k==nr) {
            rCellIdx[nr++]= c;
            ierr = PetscInfo(sol, "\t\t\tPhase: adaptToleranceFEM Found another inner r=%e, cell %D, qp %D/%D, d=%e\n", r, c, qj+1, Nq, r-minRad);CHKERRQ(ierr);
          }
        }
        if (ctx->sphere) {
          if ((tt=r-ctx->e_radius) > 0) {
            PetscInfo(sol, "\t\t\t %D cell r=%g\n",c,tt);
            if (tt < eMinRad - PETSC_SQRT_MACHINE_EPSILON*100.) {
              eMinRad = tt;
              eMaxIdx = 0;
              eCellIdx[eMaxIdx++] = c;
            } else if (eMaxIdx > 0 && (tt-eMinRad) <= PETSC_SQRT_MACHINE_EPSILON && c != eCellIdx[eMaxIdx-1]) {
              eCellIdx[eMaxIdx++] = c;
            }
          }
          if ((tt=r-ctx->i_radius[grid]) > 0) {
            if (tt < iMinRad - 1.e-5) {
              iMinRad = tt;
              iMaxIdx = 0;
              iCellIdx[iMaxIdx++] = c;
            } else if (iMaxIdx > 0 && (tt-iMinRad) <= PETSC_SQRT_MACHINE_EPSILON && c != iCellIdx[iMaxIdx-1]) {
              iCellIdx[iMaxIdx++] = c;
            }
          }
        }
      }
    }
    for (k=0;k<nr;k++) {
      ierr = DMLabelSetValue(adaptLabel, rCellIdx[k], DM_ADAPT_REFINE);CHKERRQ(ierr);
    }
    if (ctx->sphere) {
      for (c = 0; c < eMaxIdx; c++) {
        ierr = DMLabelSetValue(adaptLabel, eCellIdx[c], DM_ADAPT_REFINE);CHKERRQ(ierr);
        ierr = PetscInfo(sol, "\t\tPhase:%s: refine sphere e cell %D r=%g\n","adaptToleranceFEM",eCellIdx[c],eMinRad);CHKERRQ(ierr);
      }
      for (c = 0; c < iMaxIdx; c++) {
        ierr = DMLabelSetValue(adaptLabel, iCellIdx[c], DM_ADAPT_REFINE);CHKERRQ(ierr);
        ierr = PetscInfo(sol, "\t\tPhase:%s: refine sphere i cell %D r=%g\n","adaptToleranceFEM",iCellIdx[c],iMinRad);CHKERRQ(ierr);
      }
    }
    ierr = PetscInfo(sol, "Phase:%s: Adaptive refine origin cells %D,%D r=%g\n","adaptToleranceFEM",rCellIdx[0],rCellIdx[1],minRad);CHKERRQ(ierr);
  } else if (type==0 || type==1 || type==3) { /* refine along r=0 axis */
    PetscScalar  *coef = NULL;
    Vec          coords;
    PetscInt     csize,Nv,d,nz;
    DM           cdm;
    PetscSection cs;
    ierr = DMGetCoordinatesLocal(forest, &coords);CHKERRQ(ierr);
    ierr = DMGetCoordinateDM(forest, &cdm);CHKERRQ(ierr);
    ierr = DMGetLocalSection(cdm, &cs);CHKERRQ(ierr);
    for (c = cStart; c < cEnd; c++) {
      PetscInt doit = 0, outside = 0;
      ierr = DMPlexVecGetClosure(cdm, cs, coords, c, &csize, &coef);CHKERRQ(ierr);
      Nv = csize/dim;
      for (nz = d = 0; d < Nv; d++) {
        PetscReal z = PetscRealPart(coef[d*dim + (dim-1)]), x = PetscSqr(PetscRealPart(coef[d*dim + 0])) + ((dim==3) ? PetscSqr(PetscRealPart(coef[d*dim + 1])) : 0);
        x = PetscSqrtReal(x);
        if (x < PETSC_MACHINE_EPSILON*10. && PetscAbs(z)<PETSC_MACHINE_EPSILON*10.) doit = 1;             /* refine origin */
        else if (type==0 && (z < -PETSC_MACHINE_EPSILON*10. || z > ctx->re_radius+PETSC_MACHINE_EPSILON*10.)) outside++;   /* first pass don't refine bottom */
        else if (type==1 && (z > ctx->vperp0_radius1 || z < -ctx->vperp0_radius1)) outside++; /* don't refine outside electron refine radius */
        else if (type==3 && (z > ctx->vperp0_radius2 || z < -ctx->vperp0_radius2)) outside++; /* don't refine outside ion refine radius */
        if (x < PETSC_MACHINE_EPSILON*10.) nz++;
      }
      ierr = DMPlexVecRestoreClosure(cdm, cs, coords, c, &csize, &coef);CHKERRQ(ierr);
      if (doit || (outside<Nv && nz)) {
        ierr = DMLabelSetValue(adaptLabel, c, DM_ADAPT_REFINE);CHKERRQ(ierr);
      }
    }
    ierr = PetscInfo(sol, "Phase:%s: RE refinement\n","adaptToleranceFEM");CHKERRQ(ierr);
  }
  ierr = DMDestroy(&plex);CHKERRQ(ierr);
  ierr = DMAdaptLabel(forest, adaptLabel, &adaptedDM);CHKERRQ(ierr);
  ierr = DMLabelDestroy(&adaptLabel);CHKERRQ(ierr);
  *newForest = adaptedDM;
  if (adaptedDM) {
    if (isForest) {
      ierr = DMForestSetAdaptivityForest(adaptedDM,NULL);CHKERRQ(ierr); // ????
    } else exit(33); // ???????
    ierr = DMConvert(adaptedDM, DMPLEX, &plex);CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(plex,0,&cStart,&cEnd);CHKERRQ(ierr);
    ierr = PetscInfo(sol, "\tPhase: adaptToleranceFEM: %D cells, %d total quadrature points\n",cEnd-cStart,Nq*(cEnd-cStart));CHKERRQ(ierr);
    ierr = DMDestroy(&plex);CHKERRQ(ierr);
  } else  *newForest = NULL;
  PetscFunctionReturn(0);
}

// forest goes in (ctx->plex[grid]), plex comes out
static PetscErrorCode adapt(PetscInt grid, LandauCtx *ctx, Vec *uu)
{
  PetscErrorCode  ierr;
  PetscInt        adaptIter;

  PetscFunctionBegin;
  PetscInt  type, limits[5] = {(grid==0) ? ctx->numRERefine : 0, (grid==0) ? ctx->nZRefine1 : 0, ctx->numAMRRefine[grid], (grid==0) ? ctx->nZRefine2 : 0,ctx->postAMRRefine[grid]};
  for (type=0;type<5;type++) {
    for (adaptIter = 0; adaptIter<limits[type];adaptIter++) {
      DM  newForest = NULL;
      ierr = adaptToleranceFEM(ctx->fe[0], *uu, type, grid, ctx, &newForest);CHKERRQ(ierr);
      if (newForest)  {
        ierr = DMDestroy(&ctx->plex[grid]);CHKERRQ(ierr);
        ierr = VecDestroy(uu);CHKERRQ(ierr);
        ierr = DMCreateGlobalVector(newForest,uu);CHKERRQ(ierr);
        ierr = PetscObjectSetName((PetscObject) *uu, "uAMR");CHKERRQ(ierr);
        ierr = LandauSetInitialCondition(newForest, *uu, grid, 0, ctx);CHKERRQ(ierr);
        ctx->plex[grid] = newForest;
      } else {
        exit(4); // can happen with no AMR and post refinement
      }
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode ProcessOptions(LandauCtx *ctx, const char prefix[])
{
  PetscErrorCode    ierr;
  PetscBool         flg, sph_flg;
  PetscInt          ii,nt,nm,nc,num_species_grid[LANDAU_MAX_GRIDS];
  PetscReal         v0_grid[LANDAU_MAX_GRIDS];
  DM                dummy;

  PetscFunctionBegin;
  ierr = DMCreate(ctx->comm,&dummy);CHKERRQ(ierr);
  /* get options - initialize context */
  ctx->verbose = 1; // should be 0 for silent compliance
#if defined(PETSC_HAVE_THREADSAFETY)
  ctx->batch_sz = PetscNumOMPThreads;
#else
  ctx->batch_sz = 1;
#endif
  ctx->batch_view_idx = 0;
  ctx->interpolate = PETSC_TRUE;
  ctx->gpu_assembly = PETSC_TRUE;
  ctx->aux_bool = PETSC_FALSE;
  ctx->electronShift = 0;
  ctx->M = NULL;
  ctx->J = NULL;
  /* geometry and grids */
  ctx->sphere = PETSC_FALSE;
  ctx->inflate = PETSC_FALSE;
  ctx->aux_bool = PETSC_FALSE;
  ctx->use_p4est = PETSC_FALSE;
  ctx->num_sections = 3; /* 2, 3 or 4 */
  for (PetscInt grid=0;grid<LANDAU_MAX_GRIDS;grid++) {
    ctx->radius[grid] = 5.; /* thermal radius (velocity) */
    ctx->numAMRRefine[grid] = 5;
    ctx->postAMRRefine[grid] = 0;
    ctx->species_offset[grid+1] = 1; // one species default
    num_species_grid[grid] = 0;
    ctx->plex[grid] = NULL;     /* cache as expensive to Convert */
  }
  ctx->species_offset[0] = 0;
  ctx->re_radius = 0.;
  ctx->vperp0_radius1 = 0;
  ctx->vperp0_radius2 = 0;
  ctx->nZRefine1 = 0;
  ctx->nZRefine2 = 0;
  ctx->numRERefine = 0;
  num_species_grid[0] = 1; // one species default
  /* species - [0] electrons, [1] one ion species eg, duetarium, [2] heavy impurity ion, ... */
  ctx->charges[0] = -1;  /* electron charge (MKS) */
  ctx->masses[0] = 1/1835.469965278441013; /* temporary value in proton mass */
  ctx->n[0] = 1;
  ctx->v_0 = 1; /* thermal velocity, we could start with a scale != 1 */
  ctx->thermal_temps[0] = 1;
  /* constants, etc. */
  ctx->epsilon0 = 8.8542e-12; /* permittivity of free space (MKS) F/m */
  ctx->k = 1.38064852e-23; /* Boltzmann constant (MKS) J/K */
  ctx->lnLam = 10;         /* cross section ratio large - small angle collisions */
  ctx->n_0 = 1.e20;        /* typical plasma n, but could set it to 1 */
  ctx->Ez = 0;
  ctx->subThreadBlockSize = 1; /* for device and maybe OMP */
  for (PetscInt grid=0;grid<LANDAU_NUM_TIMERS;grid++) ctx->times[grid] = 0;
  ctx->use_matrix_mass = PETSC_FALSE; /* fast but slightly fragile */
  ctx->use_relativistic_corrections = PETSC_FALSE;
  ctx->use_energy_tensor_trick = PETSC_FALSE; /* Use Eero's trick for energy conservation v --> grad(v^2/2) */
  ctx->SData_d.w = NULL;
  ctx->SData_d.x = NULL;
  ctx->SData_d.y = NULL;
  ctx->SData_d.z = NULL;
  ctx->SData_d.invJ = NULL;
  ctx->jacobian_field_major_order = PETSC_FALSE;
  ierr = PetscOptionsBegin(ctx->comm, prefix, "Options for Fokker-Plank-Landau collision operator", "none");CHKERRQ(ierr);
  {
    char opstring[256];
#if defined(PETSC_HAVE_KOKKOS_KERNELS)
    ctx->deviceType = LANDAU_KOKKOS;
    ierr = PetscStrcpy(opstring,"kokkos");CHKERRQ(ierr);
#if defined(PETSC_HAVE_CUDA)
    ctx->subThreadBlockSize = 16;
#endif
#elif defined(PETSC_HAVE_CUDA)
    ctx->deviceType = LANDAU_CUDA;
    ierr = PetscStrcpy(opstring,"cuda");CHKERRQ(ierr);
#else
    ctx->deviceType = LANDAU_CPU;
    ierr = PetscStrcpy(opstring,"cpu");CHKERRQ(ierr);
    ctx->subThreadBlockSize = 0;
#endif
    ierr = PetscOptionsString("-dm_landau_device_type","Use kernels on 'cpu', 'cuda', or 'kokkos'","plexland.c",opstring,opstring,256,NULL);CHKERRQ(ierr);
    ierr = PetscStrcmp("cpu",opstring,&flg);CHKERRQ(ierr);
    if (flg) {
      ctx->deviceType = LANDAU_CPU;
      ctx->subThreadBlockSize = 0;
    } else {
      ierr = PetscStrcmp("cuda",opstring,&flg);CHKERRQ(ierr);
      if (flg) {
        ctx->deviceType = LANDAU_CUDA;
        ctx->subThreadBlockSize = 0;
      } else {
        ierr = PetscStrcmp("kokkos",opstring,&flg);CHKERRQ(ierr);
        if (flg) ctx->deviceType = LANDAU_KOKKOS;
        else SETERRQ(ctx->comm,PETSC_ERR_ARG_WRONG,"-dm_landau_device_type %s",opstring);
      }
    }
  }
  ierr = PetscOptionsReal("-dm_landau_electron_shift","Shift in thermal velocity of electrons","none",ctx->electronShift,&ctx->electronShift, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dm_landau_verbose", "Level of verbosity output", "plexland.c", ctx->verbose, &ctx->verbose, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dm_landau_batch_size", "Number of 'vertices' to batch", "ex2.c", ctx->batch_sz, &ctx->batch_sz, NULL);CHKERRQ(ierr);
  PetscCheckFalse(LANDAU_MAX_BATCH_SZ < ctx->batch_sz,ctx->comm,PETSC_ERR_ARG_WRONG,"LANDAU_MAX_BATCH_SZ %D < ctx->batch_sz %D",LANDAU_MAX_BATCH_SZ,ctx->batch_sz);
  ierr = PetscOptionsInt("-dm_landau_batch_view_idx", "Index of batch for diagnostics like plotting", "ex2.c", ctx->batch_view_idx, &ctx->batch_view_idx, NULL);CHKERRQ(ierr);
  PetscCheckFalse(ctx->batch_view_idx >= ctx->batch_sz,ctx->comm,PETSC_ERR_ARG_WRONG,"-ctx->batch_view_idx %D > ctx->batch_sz %D",ctx->batch_view_idx,ctx->batch_sz);
  ierr = PetscOptionsReal("-dm_landau_Ez","Initial parallel electric field in unites of Conner-Hastie critical field","plexland.c",ctx->Ez,&ctx->Ez, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-dm_landau_n_0","Normalization constant for number density","plexland.c",ctx->n_0,&ctx->n_0, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-dm_landau_ln_lambda","Cross section parameter","plexland.c",ctx->lnLam,&ctx->lnLam, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-dm_landau_use_mataxpy_mass", "Use fast but slightly fragile MATAXPY to add mass term", "plexland.c", ctx->use_matrix_mass, &ctx->use_matrix_mass, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-dm_landau_use_relativistic_corrections", "Use relativistic corrections", "plexland.c", ctx->use_relativistic_corrections, &ctx->use_relativistic_corrections, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-dm_landau_use_energy_tensor_trick", "Use Eero's trick of using grad(v^2/2) instead of v as args to Landau tensor to conserve energy with relativistic corrections and Q1 elements", "plexland.c", ctx->use_energy_tensor_trick, &ctx->use_energy_tensor_trick, NULL);CHKERRQ(ierr);

  /* get num species with temperature, set defaults */
  for (ii=1;ii<LANDAU_MAX_SPECIES;ii++) {
    ctx->thermal_temps[ii] = 1;
    ctx->charges[ii] = 1;
    ctx->masses[ii] = 1;
    ctx->n[ii] = 1;
  }
  nt = LANDAU_MAX_SPECIES;
  ierr = PetscOptionsRealArray("-dm_landau_thermal_temps", "Temperature of each species [e,i_0,i_1,...] in keV (must be set to set number of species)", "plexland.c", ctx->thermal_temps, &nt, &flg);CHKERRQ(ierr);
  if (flg) {
    PetscInfo(dummy, "num_species set to number of thermal temps provided (%D)\n",nt);
    ctx->num_species = nt;
  } else SETERRQ(ctx->comm,PETSC_ERR_ARG_WRONG,"-dm_landau_thermal_temps ,t1,t2,.. must be provided to set the number of species");
  for (ii=0;ii<ctx->num_species;ii++) ctx->thermal_temps[ii] *= 1.1604525e7; /* convert to Kelvin */
  nm = LANDAU_MAX_SPECIES-1;
  ierr = PetscOptionsRealArray("-dm_landau_ion_masses", "Mass of each species in units of proton mass [i_0=2,i_1=40...]", "plexland.c", &ctx->masses[1], &nm, &flg);CHKERRQ(ierr);
  if (flg && nm != ctx->num_species-1) {
    SETERRQ(ctx->comm,PETSC_ERR_ARG_WRONG,"num ion masses %D != num species %D",nm,ctx->num_species-1);
  }
  nm = LANDAU_MAX_SPECIES;
  ierr = PetscOptionsRealArray("-dm_landau_n", "Number density of each species = n_s * n_0", "plexland.c", ctx->n, &nm, &flg);CHKERRQ(ierr);
  PetscCheckFalse(flg && nm != ctx->num_species,ctx->comm,PETSC_ERR_ARG_WRONG,"wrong num n: %D != num species %D",nm,ctx->num_species);
  for (ii=0;ii<LANDAU_MAX_SPECIES;ii++) ctx->masses[ii] *= 1.6720e-27; /* scale by proton mass kg */
  ctx->masses[0] = 9.10938356e-31; /* electron mass kg (should be about right already) */
  ctx->m_0 = ctx->masses[0]; /* arbitrary reference mass, electrons */
  nc = LANDAU_MAX_SPECIES-1;
  ierr = PetscOptionsRealArray("-dm_landau_ion_charges", "Charge of each species in units of proton charge [i_0=2,i_1=18,...]", "plexland.c", &ctx->charges[1], &nc, &flg);CHKERRQ(ierr);
  PetscCheckFalse(flg && nc != ctx->num_species-1,ctx->comm,PETSC_ERR_ARG_WRONG,"num charges %D != num species %D",nc,ctx->num_species-1);
  for (ii=0;ii<LANDAU_MAX_SPECIES;ii++) ctx->charges[ii] *= 1.6022e-19; /* electron/proton charge (MKS) */
  /* geometry and grids */
  nt = LANDAU_MAX_GRIDS;
  ierr = PetscOptionsIntArray("-dm_landau_num_species_grid","Number of species on each grid: [ 1, ....] or [S, 0 ....] for single grid","plexland.c", num_species_grid, &nt, &flg);CHKERRQ(ierr);
  if (flg) {
    ctx->num_grids = nt;
    for (ii=nt=0;ii<ctx->num_grids;ii++) nt += num_species_grid[ii];
    PetscCheckFalse(ctx->num_species != nt,ctx->comm,PETSC_ERR_ARG_WRONG,"-dm_landau_num_species_grid: sum %D != num_species = %D. %D grids (check that number of grids <= LANDAU_MAX_GRIDS = %D)",nt,ctx->num_species,ctx->num_grids,LANDAU_MAX_GRIDS);
  } else {
    ctx->num_grids = 1; // go back to a single grid run
    num_species_grid[0] = ctx->num_species;
  }
  for (ctx->species_offset[0] = ii = 0; ii < ctx->num_grids ; ii++) ctx->species_offset[ii+1] = ctx->species_offset[ii] + num_species_grid[ii];
  PetscCheckFalse(ctx->species_offset[ctx->num_grids] != ctx->num_species,ctx->comm,PETSC_ERR_ARG_WRONG,"ctx->species_offset[ctx->num_grids] %D != ctx->num_species = %D ???????????",ctx->species_offset[ctx->num_grids],ctx->num_species);
  for (PetscInt grid = 0; grid < ctx->num_grids ; grid++) {
    int iii = ctx->species_offset[grid]; // normalize with first (arbitrary) species on grid
    v0_grid[grid] = PetscSqrtReal(ctx->k*ctx->thermal_temps[iii]/ctx->masses[iii]); /* arbitrary units for non-dimensionalization: mean velocity in 1D of first species on grid */
  }
  ii = 0;
  ierr = PetscOptionsInt("-dm_landau_v0_grid", "Index of grid to use for setting v_0 (electrons are default). Not recommended to change", "plexland.c", ii, &ii, NULL);CHKERRQ(ierr);
  ctx->v_0 = v0_grid[ii]; /* arbitrary units for non dimensionalization: global mean velocity in 1D of electrons */
  ctx->t_0 = 8*PETSC_PI*PetscSqr(ctx->epsilon0*ctx->m_0/PetscSqr(ctx->charges[0]))/ctx->lnLam/ctx->n_0*PetscPowReal(ctx->v_0,3); /* note, this t_0 makes nu[0,0]=1 */
  /* domain */
  nt = LANDAU_MAX_GRIDS;
  ierr = PetscOptionsRealArray("-dm_landau_domain_radius","Phase space size in units of thermal velocity of grid","plexland.c",ctx->radius,&nt, &flg);CHKERRQ(ierr);
  PetscCheckFalse(flg && nt < ctx->num_grids,ctx->comm,PETSC_ERR_ARG_WRONG,"-dm_landau_domain_radius: given %D radius != number grids %D",nt,ctx->num_grids);
  for (PetscInt grid = 0; grid < ctx->num_grids ; grid++) {
    if (flg && ctx->radius[grid] <= 0) { /* negative is ratio of c */
      if (ctx->radius[grid] == 0) ctx->radius[grid] = 0.75;
      else ctx->radius[grid] = -ctx->radius[grid];
      ctx->radius[grid] = ctx->radius[grid]*SPEED_OF_LIGHT/ctx->v_0; // use any species on grid to normalize (v_0 same for all on grid)
      ierr = PetscInfo(dummy, "Change domain radius to %g for grid %D\n",ctx->radius[grid],grid);CHKERRQ(ierr);
    }
    ctx->radius[grid] *= v0_grid[grid]/ctx->v_0; // scale domain by thermal radius relative to v_0
  }
  /* amr parametres */
  nt = LANDAU_MAX_GRIDS;
  ierr = PetscOptionsIntArray("-dm_landau_amr_levels_max", "Number of AMR levels of refinement around origin, after (RE) refinements along z", "plexland.c", ctx->numAMRRefine, &nt, &flg);CHKERRQ(ierr);
  PetscCheckFalse(flg && nt < ctx->num_grids,ctx->comm,PETSC_ERR_ARG_WRONG,"-dm_landau_amr_levels_max: given %D != number grids %D",nt,ctx->num_grids);
  nt = LANDAU_MAX_GRIDS;
  ierr = PetscOptionsIntArray("-dm_landau_amr_post_refine", "Number of levels to uniformly refine after AMR", "plexland.c", ctx->postAMRRefine, &nt, &flg);CHKERRQ(ierr);
  for (ii=1;ii<ctx->num_grids;ii++)  ctx->postAMRRefine[ii] = ctx->postAMRRefine[0]; // all grids the same now
  ierr = PetscOptionsInt("-dm_landau_amr_re_levels", "Number of levels to refine along v_perp=0, z>0", "plexland.c", ctx->numRERefine, &ctx->numRERefine, &flg);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dm_landau_amr_z_refine1",  "Number of levels to refine along v_perp=0", "plexland.c", ctx->nZRefine1, &ctx->nZRefine1, &flg);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dm_landau_amr_z_refine2",  "Number of levels to refine along v_perp=0", "plexland.c", ctx->nZRefine2, &ctx->nZRefine2, &flg);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-dm_landau_re_radius","velocity range to refine on positive (z>0) r=0 axis for runaways","plexland.c",ctx->re_radius,&ctx->re_radius, &flg);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-dm_landau_z_radius1","velocity range to refine r=0 axis (for electrons)","plexland.c",ctx->vperp0_radius1,&ctx->vperp0_radius1, &flg);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-dm_landau_z_radius2","velocity range to refine r=0 axis (for ions) after origin AMR","plexland.c",ctx->vperp0_radius2, &ctx->vperp0_radius2, &flg);CHKERRQ(ierr);
  /* spherical domain (not used) */
  ierr = PetscOptionsInt("-dm_landau_num_sections", "Number of tangential section in (2D) grid, 2, 3, of 4", "plexland.c", ctx->num_sections, &ctx->num_sections, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-dm_landau_sphere", "use sphere/semi-circle domain instead of rectangle", "plexland.c", ctx->sphere, &ctx->sphere, &sph_flg);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-dm_landau_inflate", "With sphere, inflate for curved edges", "plexland.c", ctx->inflate, &ctx->inflate, &flg);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-dm_landau_e_radius","Electron thermal velocity, used for circular meshes","plexland.c",ctx->e_radius, &ctx->e_radius, &flg);CHKERRQ(ierr);
  if (flg && !sph_flg) ctx->sphere = PETSC_TRUE; /* you gave me an e radius but did not set sphere, user error really */
  if (!flg) {
    ctx->e_radius = 1.5*PetscSqrtReal(8*ctx->k*ctx->thermal_temps[0]/ctx->masses[0]/PETSC_PI)/ctx->v_0;
  }
  nt = LANDAU_MAX_GRIDS;
  ierr = PetscOptionsRealArray("-dm_landau_i_radius","Ion thermal velocity, used for circular meshes","plexland.c",ctx->i_radius, &nt, &flg);CHKERRQ(ierr);
  if (flg && !sph_flg) ctx->sphere = PETSC_TRUE;
  if (!flg) {
    ctx->i_radius[0] = 1.5*PetscSqrtReal(8*ctx->k*ctx->thermal_temps[1]/ctx->masses[1]/PETSC_PI)/ctx->v_0; // need to correct for ion grid domain
  }
  PetscCheckFalse(flg && ctx->num_grids != nt,ctx->comm,PETSC_ERR_ARG_WRONG,"-dm_landau_i_radius: %D != num_species = %D",nt,ctx->num_grids);
  PetscCheckFalse(ctx->sphere && ctx->e_radius <= ctx->i_radius[0],ctx->comm,PETSC_ERR_ARG_WRONG,"bad radii: %g < %g < %g",ctx->i_radius[0],ctx->e_radius,ctx->radius[0]);
  /* processing options */
  ierr = PetscOptionsInt("-dm_landau_sub_thread_block_size", "Number of threads in Kokkos integration point subblock", "plexland.c", ctx->subThreadBlockSize, &ctx->subThreadBlockSize, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-dm_landau_gpu_assembly", "Assemble Jacobian on GPU", "plexland.c", ctx->gpu_assembly, &ctx->gpu_assembly, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-dm_landau_jacobian_field_major_order", "Reorder Jacobian for GPU assembly with field major, or block diagonal, ordering", "plexland.c", ctx->jacobian_field_major_order, &ctx->jacobian_field_major_order, NULL);CHKERRQ(ierr);
  if (ctx->jacobian_field_major_order && !ctx->gpu_assembly) {
    SETERRQ(ctx->comm,PETSC_ERR_ARG_WRONG,"-dm_landau_jacobian_field_major_order requires -dm_landau_gpu_assembly");
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  for (ii=ctx->num_species;ii<LANDAU_MAX_SPECIES;ii++) ctx->masses[ii] = ctx->thermal_temps[ii]  = ctx->charges[ii] = 0;
  if (ctx->verbose > 0) {
    ierr = PetscPrintf(ctx->comm, "masses:        e=%10.3e; ions in proton mass units:   %10.3e %10.3e ...\n",ctx->masses[0],ctx->masses[1]/1.6720e-27,ctx->num_species>2 ? ctx->masses[2]/1.6720e-27 : 0);CHKERRQ(ierr);
    ierr = PetscPrintf(ctx->comm, "charges:       e=%10.3e; charges in elementary units: %10.3e %10.3e\n", ctx->charges[0],-ctx->charges[1]/ctx->charges[0],ctx->num_species>2 ? -ctx->charges[2]/ctx->charges[0] : 0);CHKERRQ(ierr);
    ierr = PetscPrintf(ctx->comm, "n:             e: %10.3e                           i: %10.3e %10.3e\n", ctx->n[0],ctx->n[1],ctx->num_species>2 ? ctx->n[2] : 0);CHKERRQ(ierr);
    ierr = PetscPrintf(ctx->comm, "thermal T (K): e=%10.3e i=%10.3e %10.3e. v_0=%10.3e (%10.3ec) n_0=%10.3e t_0=%10.3e, %s, %s, %D batched\n", ctx->thermal_temps[0], ctx->thermal_temps[1], (ctx->num_species>2) ? ctx->thermal_temps[2] : 0, ctx->v_0, ctx->v_0/SPEED_OF_LIGHT, ctx->n_0, ctx->t_0, ctx->use_relativistic_corrections ? "relativistic" : "classical", ctx->use_energy_tensor_trick ? "Use trick" : "Intuitive",ctx->batch_sz);CHKERRQ(ierr);
    ierr = PetscPrintf(ctx->comm, "Domain radius (AMR levels) grid %D: %g (%D) ",0,ctx->radius[0],ctx->numAMRRefine[0]);CHKERRQ(ierr);
    for (ii=1;ii<ctx->num_grids;ii++) PetscPrintf(ctx->comm, ", %D: %10.3e (%D) ",ii,ctx->radius[ii],ctx->numAMRRefine[ii]);
    ierr = PetscPrintf(ctx->comm,"\n");CHKERRQ(ierr);
    if (ctx->jacobian_field_major_order) {
      ierr = PetscPrintf(ctx->comm,"Using field major order for GPU Jacobian\n");CHKERRQ(ierr);
    } else {
      ierr = PetscPrintf(ctx->comm,"Using default Plex order for all matrices\n");CHKERRQ(ierr);
    }
  }
  ierr = DMDestroy(&dummy);CHKERRQ(ierr);
  {
    PetscMPIInt    rank;
    ierr = MPI_Comm_rank(ctx->comm, &rank);CHKERRMPI(ierr);
    ctx->stage = 0;
    ierr = PetscLogEventRegister("Landau Create", DM_CLASSID, &ctx->events[13]);CHKERRQ(ierr); /* 13 */
    ierr = PetscLogEventRegister(" GPU ass. setup", DM_CLASSID, &ctx->events[2]);CHKERRQ(ierr); /* 2 */
    ierr = PetscLogEventRegister(" Build matrix", DM_CLASSID, &ctx->events[12]);CHKERRQ(ierr); /* 12 */
    ierr = PetscLogEventRegister(" Assembly maps", DM_CLASSID, &ctx->events[15]);CHKERRQ(ierr); /* 15 */
    ierr = PetscLogEventRegister("Landau Mass mat", DM_CLASSID, &ctx->events[14]);CHKERRQ(ierr); /* 14 */
    ierr = PetscLogEventRegister("Landau Operator", DM_CLASSID, &ctx->events[11]);CHKERRQ(ierr); /* 11 */
    ierr = PetscLogEventRegister("Landau Jacobian", DM_CLASSID, &ctx->events[0]);CHKERRQ(ierr); /* 0 */
    ierr = PetscLogEventRegister("Landau Mass", DM_CLASSID, &ctx->events[9]);CHKERRQ(ierr); /* 9 */
    ierr = PetscLogEventRegister(" Preamble", DM_CLASSID, &ctx->events[10]);CHKERRQ(ierr); /* 10 */
    ierr = PetscLogEventRegister(" static IP Data", DM_CLASSID, &ctx->events[7]);CHKERRQ(ierr); /* 7 */
    ierr = PetscLogEventRegister(" dynamic IP-Jac", DM_CLASSID, &ctx->events[1]);CHKERRQ(ierr); /* 1 */
    ierr = PetscLogEventRegister(" Kernel-init", DM_CLASSID, &ctx->events[3]);CHKERRQ(ierr); /* 3 */
    ierr = PetscLogEventRegister(" Jac-f-df (GPU)", DM_CLASSID, &ctx->events[8]);CHKERRQ(ierr); /* 8 */
    ierr = PetscLogEventRegister(" J Kernel (GPU)", DM_CLASSID, &ctx->events[4]);CHKERRQ(ierr); /* 4 */
    ierr = PetscLogEventRegister(" M Kernel (GPU)", DM_CLASSID, &ctx->events[16]);CHKERRQ(ierr); /* 16 */
    ierr = PetscLogEventRegister(" Copy to CPU", DM_CLASSID, &ctx->events[5]);CHKERRQ(ierr); /* 5 */
    ierr = PetscLogEventRegister(" CPU assemble", DM_CLASSID, &ctx->events[6]);CHKERRQ(ierr); /* 6 */

    if (rank) { /* turn off output stuff for duplicate runs - do we need to add the prefix to all this? */
      ierr = PetscOptionsClearValue(NULL,"-snes_converged_reason");CHKERRQ(ierr);
      ierr = PetscOptionsClearValue(NULL,"-ksp_converged_reason");CHKERRQ(ierr);
      ierr = PetscOptionsClearValue(NULL,"-snes_monitor");CHKERRQ(ierr);
      ierr = PetscOptionsClearValue(NULL,"-ksp_monitor");CHKERRQ(ierr);
      ierr = PetscOptionsClearValue(NULL,"-ts_monitor");CHKERRQ(ierr);
      ierr = PetscOptionsClearValue(NULL,"-ts_view");CHKERRQ(ierr);
      ierr = PetscOptionsClearValue(NULL,"-ts_adapt_monitor");CHKERRQ(ierr);
      ierr = PetscOptionsClearValue(NULL,"-dm_landau_amr_dm_view");CHKERRQ(ierr);
      ierr = PetscOptionsClearValue(NULL,"-dm_landau_amr_vec_view");CHKERRQ(ierr);
      ierr = PetscOptionsClearValue(NULL,"-dm_landau_mass_dm_view");CHKERRQ(ierr);
      ierr = PetscOptionsClearValue(NULL,"-dm_landau_mass_view");CHKERRQ(ierr);
      ierr = PetscOptionsClearValue(NULL,"-dm_landau_jacobian_view");CHKERRQ(ierr);
      ierr = PetscOptionsClearValue(NULL,"-dm_landau_mat_view");CHKERRQ(ierr);
      ierr = PetscOptionsClearValue(NULL,"-");CHKERRQ(ierr);
      ierr = PetscOptionsClearValue(NULL,"-info");CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateStaticGPUData(PetscInt dim, IS grid_batch_is_inv[], LandauCtx *ctx)
{
  PetscErrorCode    ierr;
  PetscSection      section[LANDAU_MAX_GRIDS],globsection[LANDAU_MAX_GRIDS];
  PetscQuadrature   quad;
  const PetscReal   *quadWeights;
  PetscInt          q,eidx,fieldA,numCells[LANDAU_MAX_GRIDS],Nq,Nb,Nf[LANDAU_MAX_GRIDS];
  PetscTabulation   *Tf;
  PetscDS           prob;

  PetscFunctionBegin;
  ierr = DMGetDS(ctx->plex[0], &prob);CHKERRQ(ierr); // same DS for all grids
  ierr = PetscDSGetTabulation(prob, &Tf);CHKERRQ(ierr); // Bf, &Df same for all grids
  /* DS, Tab and quad is same on all grids */
  PetscCheckFalse(ctx->plex[0] == NULL,ctx->comm,PETSC_ERR_ARG_WRONG,"Plex not created");
  ierr = PetscFEGetQuadrature(ctx->fe[0], &quad);CHKERRQ(ierr);
  ierr = PetscQuadratureGetData(quad, NULL, NULL, &Nq, NULL,  &quadWeights);CHKERRQ(ierr); Nb = Nq;
  PetscCheckFalse(Nq >LANDAU_MAX_NQ,ctx->comm,PETSC_ERR_ARG_WRONG,"Order too high. Nq = %D > LANDAU_MAX_NQ (%D)",Nq,LANDAU_MAX_NQ);
  /* setup each grid */
  for (PetscInt grid=0;grid<ctx->num_grids;grid++) {
    PetscInt cStart, cEnd;
    PetscCheckFalse(ctx->plex[grid] == NULL,ctx->comm,PETSC_ERR_ARG_WRONG,"Plex not created");
    ierr = DMPlexGetHeightStratum(ctx->plex[grid], 0, &cStart, &cEnd);CHKERRQ(ierr);
    numCells[grid] = cEnd - cStart; // grids can have different topology
    ierr = DMGetLocalSection(ctx->plex[grid], &section[grid]);CHKERRQ(ierr);
    ierr = DMGetGlobalSection(ctx->plex[grid], &globsection[grid]);CHKERRQ(ierr);
    ierr = PetscSectionGetNumFields(section[grid], &Nf[grid]);CHKERRQ(ierr);
  }
#define MAP_BF_SIZE (64*LANDAU_DIM*LANDAU_DIM*LANDAU_MAX_Q_FACE*LANDAU_MAX_SPECIES)
  /* create GPU assembly data */
  if (ctx->gpu_assembly) { /* we need GPU object with GPU assembly */
    PetscContainer          container;
    PetscScalar             elemMatrix[LANDAU_MAX_NQ*LANDAU_MAX_NQ*LANDAU_MAX_SPECIES*LANDAU_MAX_SPECIES], *elMat;
    pointInterpolationP4est pointMaps[MAP_BF_SIZE][LANDAU_MAX_Q_FACE];
    P4estVertexMaps         *maps;
    const PetscInt          *plex_batch=NULL;

    /* create GPU asssembly data */
    ierr = PetscInfo(ctx->plex[0], "Make GPU maps %D\n",1);CHKERRQ(ierr);
    ierr = PetscLogEventBegin(ctx->events[2],0,0,0,0);CHKERRQ(ierr);
    ierr = PetscMalloc(sizeof(*maps)*ctx->num_grids, &maps);CHKERRQ(ierr);
    ierr = PetscContainerCreate(PETSC_COMM_SELF, &container);CHKERRQ(ierr);
    ierr = PetscContainerSetPointer(container, (void *)maps);CHKERRQ(ierr);
    ierr = PetscContainerSetUserDestroy(container, LandauGPUMapsDestroy);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject) ctx->J, "assembly_maps", (PetscObject) container);CHKERRQ(ierr);
    ierr = PetscContainerDestroy(&container);CHKERRQ(ierr);

    for (PetscInt grid=0;grid<ctx->num_grids;grid++) {
      PetscInt cStart, cEnd, ej, Nfloc = Nf[grid], totDim = Nfloc*Nq;
      if (grid_batch_is_inv[grid]) {
        ierr = ISGetIndices(grid_batch_is_inv[grid], &plex_batch);CHKERRQ(ierr);
      }
      ierr = DMPlexGetHeightStratum(ctx->plex[grid], 0, &cStart, &cEnd);CHKERRQ(ierr);
      // make maps
      maps[grid].d_self = NULL;
      maps[grid].num_elements = numCells[grid];
      maps[grid].num_face = (PetscInt)(pow(Nq,1./((double)dim))+.001); // Q
      maps[grid].num_face = (PetscInt)(pow(maps[grid].num_face,(double)(dim-1))+.001); // Q^2
      maps[grid].num_reduced = 0;
      maps[grid].deviceType = ctx->deviceType;
      maps[grid].numgrids = ctx->num_grids;
      // count reduced and get
      ierr = PetscMalloc(maps[grid].num_elements * sizeof(*maps[grid].gIdx), &maps[grid].gIdx);CHKERRQ(ierr);
      for (fieldA=0;fieldA<Nf[grid];fieldA++) {
        for (ej = cStart, eidx = 0 ; ej < cEnd; ++ej, ++eidx) {
          for (q = 0; q < Nb; ++q) {
            PetscInt    numindices,*indices;
            PetscScalar *valuesOrig = elMat = elemMatrix;
            ierr = PetscMemzero(elMat, totDim*totDim*sizeof(*elMat));CHKERRQ(ierr);
            elMat[ (fieldA*Nb + q)*totDim + fieldA*Nb + q] = 1;
            ierr = DMPlexGetClosureIndices(ctx->plex[grid], section[grid], globsection[grid], ej, PETSC_TRUE, &numindices, &indices, NULL, (PetscScalar **) &elMat);CHKERRQ(ierr);
            for (PetscInt f = 0 ; f < numindices ; ++f) { // look for a non-zero on the diagonal
              if (PetscAbs(PetscRealPart(elMat[f*numindices + f])) > PETSC_MACHINE_EPSILON) {
                // found it
                if (PetscAbs(PetscRealPart(elMat[f*numindices + f] - 1.)) < PETSC_MACHINE_EPSILON) { // normal vertex 1.0
                  if (plex_batch) {
                    maps[grid].gIdx[eidx][fieldA][q] = (LandauIdx) plex_batch[indices[f]];
                  } else {
                    maps[grid].gIdx[eidx][fieldA][q] = (LandauIdx)indices[f];
                  }
                } else { //found a constraint
                  int       jj = 0;
                  PetscReal sum = 0;
                  const PetscInt ff = f;
                  maps[grid].gIdx[eidx][fieldA][q] = -maps[grid].num_reduced - 1; // store (-)index: id = -(idx+1): idx = -id - 1
                  do {  // constraints are continous in Plex - exploit that here
                    int ii; // get 'scale'
                    for (ii = 0, pointMaps[maps[grid].num_reduced][jj].scale = 0; ii < maps[grid].num_face; ii++) { // sum row of outer product to recover vector value
                      if (ff + ii < numindices) { // 3D has Q and Q^2 interps so might run off end. We could test that elMat[f*numindices + ff + ii] > 0, and break if not
                        pointMaps[maps[grid].num_reduced][jj].scale += PetscRealPart(elMat[f*numindices + ff + ii]);
                      }
                    }
                    sum += pointMaps[maps[grid].num_reduced][jj].scale; // diagnostic
                    // get 'gid'
                    if (pointMaps[maps[grid].num_reduced][jj].scale == 0) pointMaps[maps[grid].num_reduced][jj].gid = -1; // 3D has Q and Q^2 interps
                    else {
                      if (plex_batch) {
                        pointMaps[maps[grid].num_reduced][jj].gid = plex_batch[indices[f]];
                      } else {
                        pointMaps[maps[grid].num_reduced][jj].gid = indices[f];
                      }
                    }
                  } while (++jj < maps[grid].num_face && ++f < numindices); // jj is incremented if we hit the end
                  while (jj++ < maps[grid].num_face) {
                    pointMaps[maps[grid].num_reduced][jj].scale = 0;
                    pointMaps[maps[grid].num_reduced][jj].gid = -1;
                  }
                  if (PetscAbs(sum-1.0) > 10*PETSC_MACHINE_EPSILON) { // debug
                    int       d,f;
                    PetscReal tmp = 0;
                    PetscPrintf(PETSC_COMM_SELF,"\t\t%D.%D.%D) ERROR total I = %22.16e (LANDAU_MAX_Q_FACE=%d, #face=%D)\n",eidx,q,fieldA,sum,LANDAU_MAX_Q_FACE,maps[grid].num_face);
                    for (d = 0, tmp = 0; d < numindices; ++d) {
                      if (tmp!=0 && PetscAbs(tmp-1.0) > 10*PETSC_MACHINE_EPSILON) {ierr = PetscPrintf(PETSC_COMM_WORLD,"%3D) %3D: ",d,indices[d]);CHKERRQ(ierr);}
                      for (f = 0; f < numindices; ++f) {
                        tmp += PetscRealPart(elMat[d*numindices + f]);
                      }
                      if (tmp!=0) {ierr = PetscPrintf(ctx->comm," | %22.16e\n",tmp);CHKERRQ(ierr);}
                    }
                  }
                  maps[grid].num_reduced++;
                  PetscCheckFalse(maps[grid].num_reduced>=MAP_BF_SIZE,PETSC_COMM_SELF, PETSC_ERR_PLIB, "maps[grid].num_reduced %d > %d",maps[grid].num_reduced,MAP_BF_SIZE);
                }
                break;
              }
            }
            // cleanup
            ierr = DMPlexRestoreClosureIndices(ctx->plex[grid], section[grid], globsection[grid], ej, PETSC_TRUE, &numindices, &indices, NULL, (PetscScalar **) &elMat);CHKERRQ(ierr);
            if (elMat != valuesOrig) {ierr = DMRestoreWorkArray(ctx->plex[grid], numindices*numindices, MPIU_SCALAR, &elMat);CHKERRQ(ierr);}
          }
        } // cell
      } // field
      // allocate and copy point data maps[grid].gIdx[eidx][field][q]
      ierr = PetscMalloc(maps[grid].num_reduced * sizeof(*maps[grid].c_maps), &maps[grid].c_maps);CHKERRQ(ierr);
      for (ej = 0; ej < maps[grid].num_reduced; ++ej) {
        for (q = 0; q < maps[grid].num_face; ++q) {
          maps[grid].c_maps[ej][q].scale = pointMaps[ej][q].scale;
          maps[grid].c_maps[ej][q].gid   = pointMaps[ej][q].gid;
        }
      }
#if defined(PETSC_HAVE_KOKKOS_KERNELS)
      if (ctx->deviceType == LANDAU_KOKKOS) {
        ierr = LandauKokkosCreateMatMaps(maps, pointMaps, Nf, Nq, grid);CHKERRQ(ierr); // imples Kokkos does
      } // else could be CUDA
#endif
#if defined(PETSC_HAVE_CUDA)
      if (ctx->deviceType == LANDAU_CUDA) {
        ierr = LandauCUDACreateMatMaps(maps, pointMaps, Nf, Nq, grid);CHKERRQ(ierr);
      }
#endif
      if (plex_batch) {
        ierr = ISRestoreIndices(grid_batch_is_inv[grid], &plex_batch);CHKERRQ(ierr);
        ierr = ISDestroy(&grid_batch_is_inv[grid]);CHKERRQ(ierr); // we are done with this
      }
    } /* grids */
    ierr = PetscLogEventEnd(ctx->events[2],0,0,0,0);CHKERRQ(ierr);
  } // end GPU assembly
  { /* create static point data, Jacobian called first, only one vertex copy */
    PetscReal       *invJe,*ww,*xx,*yy,*zz=NULL,*invJ_a;
    PetscInt        outer_ipidx, outer_ej,grid, nip_glb = 0;
    PetscFE         fe;

    ierr = PetscLogEventBegin(ctx->events[7],0,0,0,0);CHKERRQ(ierr);
    ierr = PetscInfo(ctx->plex[0], "Initialize static data\n");CHKERRQ(ierr);
    for (PetscInt grid=0;grid<ctx->num_grids;grid++) nip_glb += Nq*numCells[grid];
    /* collect f data, first time is for Jacobian, but make mass now */
    if (ctx->verbose > 0) {
      PetscInt ncells = 0, N;
      ierr = MatGetSize(ctx->J,&N,NULL);CHKERRQ(ierr);
      for (PetscInt grid=0;grid<ctx->num_grids;grid++) ncells += numCells[grid];
      ierr = PetscPrintf(ctx->comm,"%D) %s %D IPs, %D cells total, Nb=%D, Nq=%D, dim=%D, Tab: Nb=%D Nf=%D Np=%D cdim=%D N=%D\n",
                         0,"FormLandau",nip_glb,ncells, Nb, Nq, dim, Nb, ctx->num_species, Nb, dim, N);CHKERRQ(ierr);
    }
    ierr = PetscMalloc4(nip_glb,&ww,nip_glb,&xx,nip_glb,&yy,nip_glb*dim*dim,&invJ_a);CHKERRQ(ierr);
    if (dim==3) {
      ierr = PetscMalloc1(nip_glb,&zz);CHKERRQ(ierr);
    }
    if (ctx->use_energy_tensor_trick) {
      ierr = PetscFECreateDefault(PETSC_COMM_SELF, dim, 1, PETSC_FALSE, NULL, PETSC_DECIDE, &fe);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject) fe, "energy");CHKERRQ(ierr);
    }
    /* init each grids static data - no batch */
    for (grid=0, outer_ipidx=0, outer_ej=0 ; grid < ctx->num_grids ; grid++) { // OpenMP (once)
      Vec             v2_2 = NULL; // projected function: v^2/2 for non-relativistic, gamma... for relativistic
      PetscSection    e_section;
      DM              dmEnergy;
      PetscInt        cStart, cEnd, ej;

      ierr = DMPlexGetHeightStratum(ctx->plex[grid], 0, &cStart, &cEnd);CHKERRQ(ierr);
      // prep energy trick, get v^2 / 2 vector
      if (ctx->use_energy_tensor_trick) {
        PetscErrorCode (*energyf[1])(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar [], void *) = {ctx->use_relativistic_corrections ? gamma_m1_f : energy_f};
        Vec            glob_v2;
        PetscReal      *c2_0[1], data[1] = {PetscSqr(C_0(ctx->v_0))};

        ierr = DMClone(ctx->plex[grid], &dmEnergy);CHKERRQ(ierr);
        ierr = PetscObjectSetName((PetscObject) dmEnergy, "energy");CHKERRQ(ierr);
        ierr = DMSetField(dmEnergy, 0, NULL, (PetscObject)fe);CHKERRQ(ierr);
        ierr = DMCreateDS(dmEnergy);CHKERRQ(ierr);
        ierr = DMGetSection(dmEnergy, &e_section);CHKERRQ(ierr);
        ierr = DMGetGlobalVector(dmEnergy,&glob_v2);CHKERRQ(ierr);
        ierr = PetscObjectSetName((PetscObject) glob_v2, "trick");CHKERRQ(ierr);
        c2_0[0] = &data[0];
        ierr = DMProjectFunction(dmEnergy, 0., energyf, (void**)c2_0, INSERT_ALL_VALUES, glob_v2);CHKERRQ(ierr);
        ierr = DMGetLocalVector(dmEnergy, &v2_2);CHKERRQ(ierr);
        ierr = VecZeroEntries(v2_2);CHKERRQ(ierr); /* zero BCs so don't set */
        ierr = DMGlobalToLocalBegin(dmEnergy, glob_v2, INSERT_VALUES, v2_2);CHKERRQ(ierr);
        ierr = DMGlobalToLocalEnd  (dmEnergy, glob_v2, INSERT_VALUES, v2_2);CHKERRQ(ierr);
        ierr = DMViewFromOptions(dmEnergy,NULL, "-energy_dm_view");CHKERRQ(ierr);
        ierr = VecViewFromOptions(glob_v2,NULL, "-energy_vec_view");CHKERRQ(ierr);
        ierr = DMRestoreGlobalVector(dmEnergy, &glob_v2);CHKERRQ(ierr);
      }
      /* append part of the IP data for each grid */
      for (ej = 0 ; ej < numCells[grid]; ++ej, ++outer_ej) {
        PetscScalar *coefs = NULL;
        PetscReal    vj[LANDAU_MAX_NQ*LANDAU_DIM],detJj[LANDAU_MAX_NQ], Jdummy[LANDAU_MAX_NQ*LANDAU_DIM*LANDAU_DIM], c0 = C_0(ctx->v_0), c02 = PetscSqr(c0);
        invJe = invJ_a + outer_ej*Nq*dim*dim;
        ierr = DMPlexComputeCellGeometryFEM(ctx->plex[grid], ej+cStart, quad, vj, Jdummy, invJe, detJj);CHKERRQ(ierr);
        if (ctx->use_energy_tensor_trick) {
          ierr = DMPlexVecGetClosure(dmEnergy, e_section, v2_2, ej+cStart, NULL, &coefs);CHKERRQ(ierr);
        }
        /* create static point data */
        for (PetscInt qj = 0; qj < Nq; qj++, outer_ipidx++) {
          const PetscInt  gidx = outer_ipidx;
          const PetscReal *invJ = &invJe[qj*dim*dim];
          ww    [gidx] = detJj[qj] * quadWeights[qj];
          if (dim==2) ww    [gidx] *=              vj[qj * dim + 0];  /* cylindrical coordinate, w/o 2pi */
          // get xx, yy, zz
          if (ctx->use_energy_tensor_trick) {
            double                  refSpaceDer[3],eGradPhi[3];
            const PetscReal * const DD = Tf[0]->T[1];
            const PetscReal         *Dq = &DD[qj*Nb*dim];
            for (int d = 0; d < 3; ++d) refSpaceDer[d] = eGradPhi[d] = 0.0;
            for (int b = 0; b < Nb; ++b) {
              for (int d = 0; d < dim; ++d) refSpaceDer[d] += Dq[b*dim+d]*PetscRealPart(coefs[b]);
            }
            xx[gidx] = 1e10;
            if (ctx->use_relativistic_corrections) {
              double dg2_c2 = 0;
              //for (int d = 0; d < dim; ++d) refSpaceDer[d] *= c02;
              for (int d = 0; d < dim; ++d) dg2_c2 += PetscSqr(refSpaceDer[d]);
              dg2_c2 *= (double)c02;
              if (dg2_c2 >= .999) {
                xx[gidx] = vj[qj * dim + 0]; /* coordinate */
                yy[gidx] = vj[qj * dim + 1];
                if (dim==3) zz[gidx] = vj[qj * dim + 2];
                PetscPrintf(ctx->comm,"Error: %12.5e %D.%D) dg2/c02 = %12.5e x= %12.5e %12.5e %12.5e\n",PetscSqrtReal(xx[gidx]*xx[gidx] + yy[gidx]*yy[gidx] + zz[gidx]*zz[gidx]), ej, qj, dg2_c2, xx[gidx],yy[gidx],zz[gidx]);
              } else {
                PetscReal fact = c02/PetscSqrtReal(1. - dg2_c2);
                for (int d = 0; d < dim; ++d) refSpaceDer[d] *= fact;
                // could test with other point u' that (grad - grad') * U (refSpaceDer, refSpaceDer') == 0
              }
            }
            if (xx[gidx] == 1e10) {
              for (int d = 0; d < dim; ++d) {
                for (int e = 0 ; e < dim; ++e) {
                  eGradPhi[d] += invJ[e*dim+d]*refSpaceDer[e];
                }
              }
              xx[gidx] = eGradPhi[0];
              yy[gidx] = eGradPhi[1];
              if (dim==3) zz[gidx] = eGradPhi[2];
            }
          } else {
            xx[gidx] = vj[qj * dim + 0]; /* coordinate */
            yy[gidx] = vj[qj * dim + 1];
            if (dim==3) zz[gidx] = vj[qj * dim + 2];
          }
        } /* q */
        if (ctx->use_energy_tensor_trick) {
          ierr = DMPlexVecRestoreClosure(dmEnergy, e_section, v2_2, ej+cStart, NULL, &coefs);CHKERRQ(ierr);
        }
      } /* ej */
      if (ctx->use_energy_tensor_trick) {
        ierr = DMRestoreLocalVector(dmEnergy, &v2_2);CHKERRQ(ierr);
        ierr = DMDestroy(&dmEnergy);CHKERRQ(ierr);
      }
    } /* grid */
    if (ctx->use_energy_tensor_trick) {
      ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
    }
    /* cache static data */
    if (ctx->deviceType == LANDAU_CUDA || ctx->deviceType == LANDAU_KOKKOS) {
#if defined(PETSC_HAVE_CUDA) || defined(PETSC_HAVE_KOKKOS_KERNELS)
      PetscReal invMass[LANDAU_MAX_SPECIES],nu_alpha[LANDAU_MAX_SPECIES], nu_beta[LANDAU_MAX_SPECIES];
      for (PetscInt grid = 0; grid < ctx->num_grids ; grid++) {
        for (PetscInt ii=ctx->species_offset[grid];ii<ctx->species_offset[grid+1];ii++) {
          invMass[ii] = ctx->m_0/ctx->masses[ii];
          nu_alpha[ii] = PetscSqr(ctx->charges[ii]/ctx->m_0)*ctx->m_0/ctx->masses[ii];
          nu_beta[ii] = PetscSqr(ctx->charges[ii]/ctx->epsilon0)*ctx->lnLam / (8*PETSC_PI) * ctx->t_0*ctx->n_0/PetscPowReal(ctx->v_0,3);
        }
      }
      if (ctx->deviceType == LANDAU_CUDA) {
#if defined(PETSC_HAVE_CUDA)
        ierr = LandauCUDAStaticDataSet(ctx->plex[0], Nq, ctx->batch_sz, ctx->num_grids, numCells, ctx->species_offset, ctx->mat_offset,
                                       nu_alpha, nu_beta, invMass, invJ_a, xx, yy, zz, ww, &ctx->SData_d);CHKERRQ(ierr);
#else
        SETERRQ(ctx->comm,PETSC_ERR_ARG_WRONG,"-landau_device_type %s not built","cuda");
#endif
      } else if (ctx->deviceType == LANDAU_KOKKOS) {
#if defined(PETSC_HAVE_KOKKOS_KERNELS)
        ierr = LandauKokkosStaticDataSet(ctx->plex[0], Nq, ctx->batch_sz, ctx->num_grids, numCells, ctx->species_offset, ctx->mat_offset,
                                         nu_alpha, nu_beta, invMass,invJ_a,xx,yy,zz,ww,&ctx->SData_d);CHKERRQ(ierr);
#else
        SETERRQ(ctx->comm,PETSC_ERR_ARG_WRONG,"-landau_device_type %s not built","kokkos");
#endif
      }
#endif
      /* free */
      ierr = PetscFree4(ww,xx,yy,invJ_a);CHKERRQ(ierr);
      if (dim==3) {
        ierr = PetscFree(zz);CHKERRQ(ierr);
      }
    } else { /* CPU version, just copy in, only use part */
      ctx->SData_d.w = (void*)ww;
      ctx->SData_d.x = (void*)xx;
      ctx->SData_d.y = (void*)yy;
      ctx->SData_d.z = (void*)zz;
      ctx->SData_d.invJ = (void*)invJ_a;
    }
    ierr = PetscLogEventEnd(ctx->events[7],0,0,0,0);CHKERRQ(ierr);
  } // initialize
  PetscFunctionReturn(0);
}

/* < v, u > */
static void g0_1(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, PetscReal u_tShift, const PetscReal x[],  PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  g0[0] = 1.;
}

/* < v, u > */
static void g0_fake(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, PetscReal u_tShift, const PetscReal x[],  PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  static double ttt = 1;
  g0[0] = ttt++;
}

/* < v, u > */
static void g0_r(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, PetscReal u_tShift, const PetscReal x[],  PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  g0[0] = 2.*PETSC_PI*x[0];
}

static PetscErrorCode MatrixNfDestroy(void *ptr)
{
  PetscInt *nf = (PetscInt *)ptr;
  PetscErrorCode  ierr;
  PetscFunctionBegin;
  ierr = PetscFree(nf);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode LandauCreateMatrix(MPI_Comm comm, Vec X, IS grid_batch_is_inv[LANDAU_MAX_GRIDS], LandauCtx *ctx)
{
  PetscErrorCode ierr;
  PetscInt       *idxs=NULL;
  Mat            subM[LANDAU_MAX_GRIDS];

  PetscFunctionBegin;
  if (!ctx->gpu_assembly) { /* we need GPU object with GPU assembly */
    PetscFunctionReturn(0);
  }
  // get the RCM for this grid to separate out species into blocks -- create 'idxs' & 'ctx->batch_is'
  if (ctx->gpu_assembly && ctx->jacobian_field_major_order) {
    ierr = PetscMalloc1(ctx->mat_offset[ctx->num_grids]*ctx->batch_sz, &idxs);CHKERRQ(ierr);
  }
  for (PetscInt grid=0 ; grid < ctx->num_grids ; grid++) {
    const PetscInt *values, n = ctx->mat_offset[grid+1] - ctx->mat_offset[grid];
    Mat             gMat;
    DM              massDM;
    PetscDS         prob;
    Vec             tvec;
    // get "mass" matrix for reordering
    ierr = DMClone(ctx->plex[grid], &massDM);CHKERRQ(ierr);
    ierr = DMCopyFields(ctx->plex[grid], massDM);CHKERRQ(ierr);
    ierr = DMCreateDS(massDM);CHKERRQ(ierr);
    ierr = DMGetDS(massDM, &prob);CHKERRQ(ierr);
    for (int ix=0, ii=ctx->species_offset[grid];ii<ctx->species_offset[grid+1];ii++,ix++) {
      ierr = PetscDSSetJacobian(prob, ix, ix, g0_fake, NULL, NULL, NULL);CHKERRQ(ierr);
    }
    ierr = PetscOptionsInsertString(NULL,"-dm_preallocate_only");
    ierr = DMSetFromOptions(massDM);CHKERRQ(ierr);
    ierr = DMCreateMatrix(massDM, &gMat);CHKERRQ(ierr);
    ierr = PetscOptionsInsertString(NULL,"-dm_preallocate_only false");
    ierr = MatSetOption(gMat, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE);CHKERRQ(ierr);
    ierr = DMCreateLocalVector(ctx->plex[grid],&tvec);CHKERRQ(ierr);
    ierr = DMPlexSNESComputeJacobianFEM(massDM, tvec, gMat, gMat, ctx);CHKERRQ(ierr);
    ierr = DMDestroy(&massDM);CHKERRQ(ierr);
    ierr = VecDestroy(&tvec);CHKERRQ(ierr);
    subM[grid] = gMat;
    if (ctx->gpu_assembly && ctx->jacobian_field_major_order) {
      MatOrderingType rtype = MATORDERINGRCM;
      IS              isrow,isicol;
      ierr = MatGetOrdering(gMat,rtype,&isrow,&isicol);CHKERRQ(ierr);
      ierr = ISInvertPermutation(isrow,PETSC_DECIDE,&grid_batch_is_inv[grid]);CHKERRQ(ierr);
      ierr = ISGetIndices(isrow, &values);CHKERRQ(ierr);
      for (PetscInt b_id=0 ; b_id < ctx->batch_sz ; b_id++) { // add batch size DMs for this species grid
#if !defined(LANDAU_SPECIES_MAJOR)
        PetscInt N = ctx->mat_offset[ctx->num_grids], n0 = ctx->mat_offset[grid] + b_id*N;
        for (int ii = 0; ii < n; ++ii) idxs[n0+ii] = values[ii] + n0;
#else
        PetscInt n0 = ctx->mat_offset[grid]*ctx->batch_sz + b_id*n;
        for (int ii = 0; ii < n; ++ii) idxs[n0+ii] = values[ii] + n0;
#endif
      }
      ierr = ISRestoreIndices(isrow, &values);CHKERRQ(ierr);
      ierr = ISDestroy(&isrow);CHKERRQ(ierr);
      ierr = ISDestroy(&isicol);CHKERRQ(ierr);
    }
  }
  if (ctx->gpu_assembly && ctx->jacobian_field_major_order) {
    ierr = ISCreateGeneral(comm,ctx->mat_offset[ctx->num_grids]*ctx->batch_sz,idxs,PETSC_OWN_POINTER,&ctx->batch_is);CHKERRQ(ierr);
  }
  // get a block matrix
  for (PetscInt grid=0 ; grid<ctx->num_grids ; grid++) {
    Mat               B = subM[grid];
    PetscInt          nloc, nzl, colbuf[1024], row;
    ierr = MatGetSize(B, &nloc, NULL);CHKERRQ(ierr);
    for (PetscInt b_id = 0 ; b_id < ctx->batch_sz ; b_id++) {
      const PetscInt    moffset = LAND_MOFFSET(b_id,grid,ctx->batch_sz,ctx->num_grids,ctx->mat_offset);
      const PetscInt    *cols;
      const PetscScalar *vals;
      for (int i=0 ; i<nloc ; i++) {
        ierr = MatGetRow(B,i,&nzl,&cols,&vals);CHKERRQ(ierr);
        PetscCheck(nzl<=1024,comm, PETSC_ERR_PLIB, "Row too big: %D",nzl);
        for (int j=0; j<nzl; j++) colbuf[j] = cols[j] + moffset;
        row = i + moffset;
        ierr = MatSetValues(ctx->J,1,&row,nzl,colbuf,vals,INSERT_VALUES);CHKERRQ(ierr);
        ierr = MatRestoreRow(B,i,&nzl,&cols,&vals);CHKERRQ(ierr);
      }
    }
  }
  for (PetscInt grid=0 ; grid<ctx->num_grids ; grid++) {
    ierr = MatDestroy(&subM[grid]);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(ctx->J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(ctx->J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  if (ctx->gpu_assembly && ctx->jacobian_field_major_order) {
    Mat            mat_block_order;
    PetscContainer container;
    ierr = MatCreateSubMatrix(ctx->J,ctx->batch_is,ctx->batch_is,MAT_INITIAL_MATRIX,&mat_block_order);CHKERRQ(ierr); // use MatPermute
    ierr = MatDestroy(&ctx->J);CHKERRQ(ierr);
    ctx->J = mat_block_order;
    // cache ctx for KSP with batch/field major Jacobian ordering -ksp_type gmres/etc -dm_landau_jacobian_field_major_order
    ierr = PetscContainerCreate(PETSC_COMM_SELF, &container);CHKERRQ(ierr);
    ierr = PetscContainerSetPointer(container, (void *)ctx);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject) ctx->J, "LandauCtx", (PetscObject) container);CHKERRQ(ierr);
    ierr = PetscContainerDestroy(&container);CHKERRQ(ierr);
    // override ops to make KSP work in field major space
    ctx->seqaij_mult = mat_block_order->ops->mult;
    mat_block_order->ops->mult = LandauMatMult;
    mat_block_order->ops->multadd = LandauMatMultAdd;
    ctx->seqaij_solve = NULL;
    ctx->seqaij_getdiagonal = mat_block_order->ops->getdiagonal;
    mat_block_order->ops->getdiagonal = LandauMatGetDiagonal;
    ctx->seqaij_multtranspose = mat_block_order->ops->multtranspose;
    mat_block_order->ops->multtranspose = LandauMatMultTranspose;
    ierr = VecDuplicate(X,&ctx->work_vec);CHKERRQ(ierr);
    ierr = VecScatterCreate(X, ctx->batch_is, ctx->work_vec, NULL, &ctx->plex_batch);CHKERRQ(ierr);
    // batch solvers need to map -- can batch solvers work
    ierr = PetscContainerCreate(PETSC_COMM_SELF, &container);CHKERRQ(ierr);
    ierr = PetscContainerSetPointer(container, (void *)ctx->plex_batch);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject) ctx->J, "plex_batch_is", (PetscObject) container);CHKERRQ(ierr);
    ierr = PetscContainerDestroy(&container);CHKERRQ(ierr);
  }
  {
    PetscContainer  container;
    PetscInt        *pNf;
    ierr = PetscContainerCreate(PETSC_COMM_SELF, &container);CHKERRQ(ierr);
    ierr = PetscMalloc(sizeof(*pNf), &pNf);CHKERRQ(ierr);
    *pNf = ctx->batch_sz;
    ierr = PetscContainerSetPointer(container, (void *)pNf);CHKERRQ(ierr);
    ierr = PetscContainerSetUserDestroy(container, MatrixNfDestroy);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject)ctx->J, "batch size", (PetscObject) container);CHKERRQ(ierr);
    ierr = PetscContainerDestroy(&container);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

PetscErrorCode LandauCreateMassMatrix(DM pack, Mat *Amat);
/*@C
 LandauCreateVelocitySpace - Create a DMPlex velocity space mesh

 Collective on comm

 Input Parameters:
 +   comm  - The MPI communicator
 .   dim - velocity space dimension (2 for axisymmetric, 3 for full 3X + 3V solver)
 -   prefix - prefix for options (not tested)

 Output Parameter:
 .   pack  - The DM object representing the mesh
 +   X - A vector (user destroys)
 -   J - Optional matrix (object destroys)

 Level: beginner

 .keywords: mesh
 .seealso: DMPlexCreate(), LandauDestroyVelocitySpace()
 @*/
PetscErrorCode LandauCreateVelocitySpace(MPI_Comm comm, PetscInt dim, const char prefix[], Vec *X, Mat *J, DM *pack)
{
  PetscErrorCode ierr;
  LandauCtx      *ctx;
  Vec            Xsub[LANDAU_MAX_GRIDS];
  IS             grid_batch_is_inv[LANDAU_MAX_GRIDS];

  PetscFunctionBegin;
  PetscCheckFalse(dim!=2 && dim!=3,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Only 2D and 3D supported");
  PetscCheckFalse(LANDAU_DIM != dim,PETSC_COMM_SELF, PETSC_ERR_PLIB, "dim %D != LANDAU_DIM %d",dim,LANDAU_DIM);
  ierr = PetscNew(&ctx);CHKERRQ(ierr);
  ctx->comm = comm; /* used for diagnostics and global errors */
  /* process options */
  ierr = ProcessOptions(ctx,prefix);CHKERRQ(ierr);
  if (dim==2) ctx->use_relativistic_corrections = PETSC_FALSE;
  /* Create Mesh */
  ierr = DMCompositeCreate(PETSC_COMM_SELF,pack);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(ctx->events[13],0,0,0,0);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(ctx->events[15],0,0,0,0);CHKERRQ(ierr);
  ierr = LandauDMCreateVMeshes(PETSC_COMM_SELF, dim, prefix, ctx, *pack);CHKERRQ(ierr); // creates grids (Forest of AMR)
  for (PetscInt grid=0;grid<ctx->num_grids;grid++) {
    /* create FEM */
    ierr = SetupDS(ctx->plex[grid],dim,grid,ctx);CHKERRQ(ierr);
    /* set initial state */
    ierr = DMCreateGlobalVector(ctx->plex[grid],&Xsub[grid]);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) Xsub[grid], "u_orig");CHKERRQ(ierr);
    /* initial static refinement, no solve */
    ierr = LandauSetInitialCondition(ctx->plex[grid], Xsub[grid], grid, 0, ctx);CHKERRQ(ierr);
    /* forest refinement - forest goes in (if forest), plex comes out */
    if (ctx->use_p4est) {
      DM plex;
      ierr = adapt(grid,ctx,&Xsub[grid]);CHKERRQ(ierr); // forest goes in, plex comes out
      ierr = DMViewFromOptions(ctx->plex[grid],NULL,"-dm_landau_amr_dm_view");CHKERRQ(ierr); // need to differentiate - todo
      ierr = VecViewFromOptions(Xsub[grid], NULL, "-dm_landau_amr_vec_view");CHKERRQ(ierr);
      // convert to plex, all done with this level
      ierr = DMConvert(ctx->plex[grid], DMPLEX, &plex);CHKERRQ(ierr);
      ierr = DMDestroy(&ctx->plex[grid]);CHKERRQ(ierr);
      ctx->plex[grid] = plex;
    }
#if !defined(LANDAU_SPECIES_MAJOR)
    ierr = DMCompositeAddDM(*pack,ctx->plex[grid]);CHKERRQ(ierr);
#else
    for (PetscInt b_id=0;b_id<ctx->batch_sz;b_id++) { // add batch size DMs for this species grid
      ierr = DMCompositeAddDM(*pack,ctx->plex[grid]);CHKERRQ(ierr);
    }
#endif
    ierr = DMSetApplicationContext(ctx->plex[grid], ctx);CHKERRQ(ierr);
  }
#if !defined(LANDAU_SPECIES_MAJOR)
  // stack the batched DMs, could do it all here!!! b_id=0
  for (PetscInt b_id=1;b_id<ctx->batch_sz;b_id++) {
    for (PetscInt grid=0;grid<ctx->num_grids;grid++) {
      ierr = DMCompositeAddDM(*pack,ctx->plex[grid]);CHKERRQ(ierr);
    }
  }
#endif
  // create ctx->mat_offset
  ctx->mat_offset[0] = 0;
  for (PetscInt grid=0 ; grid < ctx->num_grids ; grid++) {
    PetscInt    n;
    ierr = VecGetLocalSize(Xsub[grid],&n);CHKERRQ(ierr);
    ctx->mat_offset[grid+1] = ctx->mat_offset[grid] + n;
  }
  // creat DM & Jac
  ierr = DMSetApplicationContext(*pack, ctx);CHKERRQ(ierr);
  ierr = PetscOptionsInsertString(NULL,"-dm_preallocate_only");
  ierr = DMSetFromOptions(*pack);CHKERRQ(ierr);
  ierr = DMCreateMatrix(*pack, &ctx->J);CHKERRQ(ierr);
  ierr = PetscOptionsInsertString(NULL,"-dm_preallocate_only false");
  ierr = MatSetOption(ctx->J, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatSetOption(ctx->J, MAT_STRUCTURALLY_SYMMETRIC, PETSC_TRUE);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)ctx->J, "Jac");CHKERRQ(ierr);
  // construct initial conditions in X
  ierr = DMCreateGlobalVector(*pack,X);CHKERRQ(ierr);
  for (PetscInt grid=0 ; grid < ctx->num_grids ; grid++) {
    PetscInt n;
    ierr = VecGetLocalSize(Xsub[grid],&n);CHKERRQ(ierr);
    for (PetscInt b_id = 0 ; b_id < ctx->batch_sz ; b_id++) {
      PetscScalar const *values;
      const PetscInt    moffset = LAND_MOFFSET(b_id,grid,ctx->batch_sz,ctx->num_grids,ctx->mat_offset);
      ierr = LandauSetInitialCondition(ctx->plex[grid], Xsub[grid], grid, b_id, ctx);CHKERRQ(ierr);
      ierr = VecGetArrayRead(Xsub[grid],&values);CHKERRQ(ierr);
      for (int i=0, idx = moffset; i<n; i++, idx++) {
        ierr = VecSetValue(*X,idx,values[i],INSERT_VALUES);CHKERRQ(ierr);
      }
      ierr = VecRestoreArrayRead(Xsub[grid],&values);CHKERRQ(ierr);
    }
  }
  // cleanup
  for (PetscInt grid=0 ; grid < ctx->num_grids ; grid++) {
    ierr = VecDestroy(&Xsub[grid]);CHKERRQ(ierr);
  }
  /* check for correct matrix type */
  if (ctx->gpu_assembly) { /* we need GPU object with GPU assembly */
    PetscBool flg;
    if (ctx->deviceType == LANDAU_CUDA) {
      ierr = PetscObjectTypeCompareAny((PetscObject)ctx->J,&flg,MATSEQAIJCUSPARSE,MATMPIAIJCUSPARSE,MATAIJCUSPARSE,"");CHKERRQ(ierr);
      PetscCheckFalse(!flg,ctx->comm,PETSC_ERR_ARG_WRONG,"must use '-dm_mat_type aijcusparse -dm_vec_type cuda' for GPU assembly and Cuda or use '-dm_landau_device_type cpu'");
    } else if (ctx->deviceType == LANDAU_KOKKOS) {
      ierr = PetscObjectTypeCompareAny((PetscObject)ctx->J,&flg,MATSEQAIJKOKKOS,MATMPIAIJKOKKOS,MATAIJKOKKOS,"");CHKERRQ(ierr);
#if defined(PETSC_HAVE_KOKKOS_KERNELS)
      PetscCheckFalse(!flg,ctx->comm,PETSC_ERR_ARG_WRONG,"must use '-dm_mat_type aijkokkos -dm_vec_type kokkos' for GPU assembly and Kokkos or use '-dm_landau_device_type cpu'");
#else
      PetscCheckFalse(!flg,ctx->comm,PETSC_ERR_ARG_WRONG,"must configure with '--download-kokkos-kernels' for GPU assembly and Kokkos or use '-dm_landau_device_type cpu'");
#endif
    }
  }
  ierr = PetscLogEventEnd(ctx->events[15],0,0,0,0);CHKERRQ(ierr);
  // create field major ordering

  ctx->work_vec = NULL;
  ctx->plex_batch = NULL;
  ctx->batch_is = NULL;
  for (int i=0;i<LANDAU_MAX_GRIDS;i++) grid_batch_is_inv[i] = NULL;
  ierr = PetscLogEventBegin(ctx->events[12],0,0,0,0);CHKERRQ(ierr);
  ierr = LandauCreateMatrix(comm, *X, grid_batch_is_inv, ctx);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(ctx->events[12],0,0,0,0);CHKERRQ(ierr);

  // create AMR GPU assembly maps and static GPU data
  ierr = CreateStaticGPUData(dim,grid_batch_is_inv,ctx);CHKERRQ(ierr);

  ierr = PetscLogEventEnd(ctx->events[13],0,0,0,0);CHKERRQ(ierr);

  // create mass matrix
  ierr = LandauCreateMassMatrix(*pack, NULL);CHKERRQ(ierr);

  if (J) *J = ctx->J;

  PetscFunctionReturn(0);
}

/*@
 LandauDestroyVelocitySpace - Destroy a DMPlex velocity space mesh

 Collective on dm

 Input/Output Parameters:
 .   dm - the dm to destroy

 Level: beginner

 .keywords: mesh
 .seealso: LandauCreateVelocitySpace()
 @*/
PetscErrorCode LandauDestroyVelocitySpace(DM *dm)
{
  PetscErrorCode ierr,ii;
  LandauCtx      *ctx;
  PetscFunctionBegin;
  ierr = DMGetApplicationContext(*dm, &ctx);CHKERRQ(ierr);
  ierr = MatDestroy(&ctx->M);CHKERRQ(ierr);
  ierr = MatDestroy(&ctx->J);CHKERRQ(ierr);
  for (ii=0;ii<ctx->num_species;ii++) {
    ierr = PetscFEDestroy(&ctx->fe[ii]);CHKERRQ(ierr);
  }
  ierr = ISDestroy(&ctx->batch_is);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->work_vec);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&ctx->plex_batch);CHKERRQ(ierr);
  if (ctx->deviceType == LANDAU_CUDA) {
#if defined(PETSC_HAVE_CUDA)
    ierr = LandauCUDAStaticDataClear(&ctx->SData_d);CHKERRQ(ierr);
#else
    SETERRQ(ctx->comm,PETSC_ERR_ARG_WRONG,"-landau_device_type %s not built","cuda");
#endif
  } else if (ctx->deviceType == LANDAU_KOKKOS) {
#if defined(PETSC_HAVE_KOKKOS_KERNELS)
    ierr = LandauKokkosStaticDataClear(&ctx->SData_d);CHKERRQ(ierr);
#else
    SETERRQ(ctx->comm,PETSC_ERR_ARG_WRONG,"-landau_device_type %s not built","kokkos");
#endif
  } else {
    if (ctx->SData_d.x) { /* in a CPU run */
      PetscReal *invJ = (PetscReal*)ctx->SData_d.invJ, *xx = (PetscReal*)ctx->SData_d.x, *yy = (PetscReal*)ctx->SData_d.y, *zz = (PetscReal*)ctx->SData_d.z, *ww = (PetscReal*)ctx->SData_d.w;
      ierr = PetscFree4(ww,xx,yy,invJ);CHKERRQ(ierr);
      if (zz) {
        ierr = PetscFree(zz);CHKERRQ(ierr);
      }
    }
  }

  if (ctx->times[LANDAU_MATRIX_TOTAL] > 0) { // OMP timings
    ierr = PetscPrintf(ctx->comm, "TSStep               N  1.0 %10.3e\n",ctx->times[LANDAU_EX2_TSSOLVE]);CHKERRQ(ierr);
    ierr = PetscPrintf(ctx->comm, "2:           Solve:  %10.3e with %D threads\n",ctx->times[LANDAU_EX2_TSSOLVE] - ctx->times[LANDAU_MATRIX_TOTAL],ctx->batch_sz);CHKERRQ(ierr);
    ierr = PetscPrintf(ctx->comm, "3:          Landau:  %10.3e\n",ctx->times[LANDAU_MATRIX_TOTAL]);CHKERRQ(ierr);
    ierr = PetscPrintf(ctx->comm, "Landau Jacobian       %D 1.0 %10.3e\n",(PetscInt)ctx->times[LANDAU_JACOBIAN_COUNT],ctx->times[LANDAU_JACOBIAN]);CHKERRQ(ierr);
    ierr = PetscPrintf(ctx->comm, "Landau Operator       N 1.0  %10.3e\n",ctx->times[LANDAU_OPERATOR]);CHKERRQ(ierr);
    ierr = PetscPrintf(ctx->comm, "Landau Mass           N 1.0  %10.3e\n",ctx->times[LANDAU_MASS]);CHKERRQ(ierr);
    ierr = PetscPrintf(ctx->comm, " Jac-f-df (GPU)       N 1.0  %10.3e\n",ctx->times[LANDAU_F_DF]);CHKERRQ(ierr);
    ierr = PetscPrintf(ctx->comm, " Kernel (GPU)         N 1.0  %10.3e\n",ctx->times[LANDAU_KERNEL]);CHKERRQ(ierr);
    ierr = PetscPrintf(ctx->comm, "MatLUFactorNum        X 1.0 %10.3e\n",ctx->times[KSP_FACTOR]);CHKERRQ(ierr);
    ierr = PetscPrintf(ctx->comm, "MatSolve              X 1.0 %10.3e\n",ctx->times[KSP_SOLVE]);CHKERRQ(ierr);
  }
  for (PetscInt grid=0 ; grid < ctx->num_grids ; grid++) {
    ierr = DMDestroy(&ctx->plex[grid]);CHKERRQ(ierr);
  }
  PetscFree(ctx);
  ierr = DMDestroy(dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* < v, ru > */
static void f0_s_den(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                     const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                     const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                     PetscReal t, const PetscReal x[],  PetscInt numConstants, const PetscScalar constants[], PetscScalar *f0)
{
  PetscInt ii = (PetscInt)PetscRealPart(constants[0]);
  f0[0] = u[ii];
}

/* < v, ru > */
static void f0_s_mom(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                     const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                     const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                     PetscReal t, const PetscReal x[],  PetscInt numConstants, const PetscScalar constants[], PetscScalar *f0)
{
  PetscInt ii = (PetscInt)PetscRealPart(constants[0]), jj = (PetscInt)PetscRealPart(constants[1]);
  f0[0] = x[jj]*u[ii]; /* x momentum */
}

static void f0_s_v2(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                    const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                    PetscReal t, const PetscReal x[],  PetscInt numConstants, const PetscScalar constants[], PetscScalar *f0)
{
  PetscInt i, ii = (PetscInt)PetscRealPart(constants[0]);
  double tmp1 = 0.;
  for (i = 0; i < dim; ++i) tmp1 += x[i]*x[i];
  f0[0] = tmp1*u[ii];
}

static PetscErrorCode gamma_n_f(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *actx)
{
  const PetscReal *c2_0_arr = ((PetscReal*)actx);
  const PetscReal c02 = c2_0_arr[0];

  PetscFunctionBegin;
  for (int s = 0 ; s < Nf ; s++) {
    PetscReal tmp1 = 0.;
    for (int i = 0; i < dim; ++i) tmp1 += x[i]*x[i];
#if defined(PETSC_USE_DEBUG)
    u[s] = PetscSqrtReal(1. + tmp1/c02);//  u[0] = PetscSqrtReal(1. + xx);
#else
    {
      PetscReal xx = tmp1/c02;
      u[s] = xx/(PetscSqrtReal(1. + xx) + 1.); // better conditioned = xx/(PetscSqrtReal(1. + xx) + 1.)
    }
#endif
  }
  PetscFunctionReturn(0);
}

/* < v, ru > */
static void f0_s_rden(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                      const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                      const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                      PetscReal t, const PetscReal x[],  PetscInt numConstants, const PetscScalar constants[], PetscScalar *f0)
{
  PetscInt ii = (PetscInt)PetscRealPart(constants[0]);
  f0[0] = 2.*PETSC_PI*x[0]*u[ii];
}

/* < v, ru > */
static void f0_s_rmom(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                      const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                      const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                      PetscReal t, const PetscReal x[],  PetscInt numConstants, const PetscScalar constants[], PetscScalar *f0)
{
  PetscInt ii = (PetscInt)PetscRealPart(constants[0]);
  f0[0] = 2.*PETSC_PI*x[0]*x[1]*u[ii];
}

static void f0_s_rv2(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                     const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                     const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                     PetscReal t, const PetscReal x[],  PetscInt numConstants, const PetscScalar constants[], PetscScalar *f0)
{
  PetscInt ii = (PetscInt)PetscRealPart(constants[0]);
  f0[0] =  2.*PETSC_PI*x[0]*(x[0]*x[0] + x[1]*x[1])*u[ii];
}

/*@
 LandauPrintNorms - collects moments and prints them

 Collective on dm

 Input Parameters:
 +   X  - the state
 -   stepi - current step to print

 Level: beginner

 .keywords: mesh
 .seealso: LandauCreateVelocitySpace()
 @*/
PetscErrorCode LandauPrintNorms(Vec X, PetscInt stepi)
{
  PetscErrorCode ierr;
  LandauCtx      *ctx;
  PetscDS        prob;
  DM             pack;
  PetscInt       cStart, cEnd, dim, ii, i0, nDMs;
  PetscScalar    xmomentumtot=0, ymomentumtot=0, zmomentumtot=0, energytot=0, densitytot=0, tt[LANDAU_MAX_SPECIES];
  PetscScalar    xmomentum[LANDAU_MAX_SPECIES],  ymomentum[LANDAU_MAX_SPECIES],  zmomentum[LANDAU_MAX_SPECIES], energy[LANDAU_MAX_SPECIES], density[LANDAU_MAX_SPECIES];
  Vec            *globXArray;

  PetscFunctionBegin;
  ierr = VecGetDM(X, &pack);CHKERRQ(ierr);
  PetscCheckFalse(!pack,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Vector has no DM");
  ierr = DMGetDimension(pack, &dim);CHKERRQ(ierr);
  PetscCheckFalse(dim!=2 && dim!=3,PETSC_COMM_SELF, PETSC_ERR_PLIB, "dim= %D",dim);
  ierr = DMGetApplicationContext(pack, &ctx);CHKERRQ(ierr);
  PetscCheckFalse(!ctx,PETSC_COMM_SELF, PETSC_ERR_PLIB, "no context");
  /* print momentum and energy */
  ierr = DMCompositeGetNumberDM(pack,&nDMs);CHKERRQ(ierr);
  PetscCheckFalse(nDMs != ctx->num_grids*ctx->batch_sz,PETSC_COMM_WORLD, PETSC_ERR_PLIB, "#DM wrong %D %D",nDMs,ctx->num_grids*ctx->batch_sz);
  ierr = PetscMalloc(sizeof(*globXArray)*nDMs, &globXArray);CHKERRQ(ierr);
  ierr = DMCompositeGetAccessArray(pack, X, nDMs, NULL, globXArray);CHKERRQ(ierr);
  for (PetscInt grid = 0; grid < ctx->num_grids ; grid++) {
    Vec Xloc = globXArray[ LAND_PACK_IDX(ctx->batch_view_idx,grid) ];
    ierr = DMGetDS(ctx->plex[grid], &prob);CHKERRQ(ierr);
    for (ii=ctx->species_offset[grid],i0=0;ii<ctx->species_offset[grid+1];ii++,i0++) {
      PetscScalar user[2] = { (PetscScalar)i0, (PetscScalar)ctx->charges[ii]};
      ierr = PetscDSSetConstants(prob, 2, user);CHKERRQ(ierr);
      if (dim==2) { /* 2/3X + 3V (cylindrical coordinates) */
        ierr = PetscDSSetObjective(prob, 0, &f0_s_rden);CHKERRQ(ierr);
        ierr = DMPlexComputeIntegralFEM(ctx->plex[grid],Xloc,tt,ctx);CHKERRQ(ierr);
        density[ii] = tt[0]*ctx->n_0*ctx->charges[ii];
        ierr = PetscDSSetObjective(prob, 0, &f0_s_rmom);CHKERRQ(ierr);
        ierr = DMPlexComputeIntegralFEM(ctx->plex[grid],Xloc,tt,ctx);CHKERRQ(ierr);
        zmomentum[ii] = tt[0]*ctx->n_0*ctx->v_0*ctx->masses[ii];
        ierr = PetscDSSetObjective(prob, 0, &f0_s_rv2);CHKERRQ(ierr);
        ierr = DMPlexComputeIntegralFEM(ctx->plex[grid],Xloc,tt,ctx);CHKERRQ(ierr);
        energy[ii] = tt[0]*0.5*ctx->n_0*ctx->v_0*ctx->v_0*ctx->masses[ii];
        zmomentumtot += zmomentum[ii];
        energytot  += energy[ii];
        densitytot += density[ii];
        ierr = PetscPrintf(ctx->comm, "%3D) species-%D: charge density= %20.13e z-momentum= %20.13e energy= %20.13e",stepi,ii,PetscRealPart(density[ii]),PetscRealPart(zmomentum[ii]),PetscRealPart(energy[ii]));CHKERRQ(ierr);
      } else { /* 2/3Xloc + 3V */
        ierr = PetscDSSetObjective(prob, 0, &f0_s_den);CHKERRQ(ierr);
        ierr = DMPlexComputeIntegralFEM(ctx->plex[grid],Xloc,tt,ctx);CHKERRQ(ierr);
        density[ii] = tt[0]*ctx->n_0*ctx->charges[ii];
        ierr = PetscDSSetObjective(prob, 0, &f0_s_mom);CHKERRQ(ierr);
        user[1] = 0;
        ierr = PetscDSSetConstants(prob, 2, user);CHKERRQ(ierr);
        ierr = DMPlexComputeIntegralFEM(ctx->plex[grid],Xloc,tt,ctx);CHKERRQ(ierr);
        xmomentum[ii]  = tt[0]*ctx->n_0*ctx->v_0*ctx->masses[ii];
        user[1] = 1;
        ierr = PetscDSSetConstants(prob, 2, user);CHKERRQ(ierr);
        ierr = DMPlexComputeIntegralFEM(ctx->plex[grid],Xloc,tt,ctx);CHKERRQ(ierr);
        ymomentum[ii] = tt[0]*ctx->n_0*ctx->v_0*ctx->masses[ii];
        user[1] = 2;
        ierr = PetscDSSetConstants(prob, 2, user);CHKERRQ(ierr);
        ierr = DMPlexComputeIntegralFEM(ctx->plex[grid],Xloc,tt,ctx);CHKERRQ(ierr);
        zmomentum[ii] = tt[0]*ctx->n_0*ctx->v_0*ctx->masses[ii];
        if (ctx->use_relativistic_corrections) {
          /* gamma * M * f */
          if (ii==0 && grid==0) { // do all at once
            Vec            Mf, globGamma, *globMfArray, *globGammaArray;
            PetscErrorCode (*gammaf[1])(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar [], void *) = {gamma_n_f};
            PetscReal      *c2_0[1], data[1];

            ierr = VecDuplicate(X,&globGamma);CHKERRQ(ierr);
            ierr = VecDuplicate(X,&Mf);CHKERRQ(ierr);
            ierr = PetscMalloc(sizeof(*globMfArray)*nDMs, &globMfArray);CHKERRQ(ierr);
            ierr = PetscMalloc(sizeof(*globMfArray)*nDMs, &globGammaArray);CHKERRQ(ierr);
            /* M * f */
            ierr = MatMult(ctx->M,X,Mf);CHKERRQ(ierr);
            /* gamma */
            ierr = DMCompositeGetAccessArray(pack, globGamma, nDMs, NULL, globGammaArray);CHKERRQ(ierr);
            for (PetscInt grid = 0; grid < ctx->num_grids ; grid++) { // yes a grid loop in a grid loop to print nice, need to fix for batching
              Vec v1 = globGammaArray[ LAND_PACK_IDX(ctx->batch_view_idx,grid) ];
              data[0] = PetscSqr(C_0(ctx->v_0));
              c2_0[0] = &data[0];
              ierr = DMProjectFunction(ctx->plex[grid], 0., gammaf, (void**)c2_0, INSERT_ALL_VALUES, v1);CHKERRQ(ierr);
            }
            ierr = DMCompositeRestoreAccessArray(pack, globGamma, nDMs, NULL, globGammaArray);CHKERRQ(ierr);
            /* gamma * Mf */
            ierr = DMCompositeGetAccessArray(pack, globGamma, nDMs, NULL, globGammaArray);CHKERRQ(ierr);
            ierr = DMCompositeGetAccessArray(pack, Mf, nDMs, NULL, globMfArray);CHKERRQ(ierr);
            for (PetscInt grid = 0; grid < ctx->num_grids ; grid++) { // yes a grid loop in a grid loop to print nice
              PetscInt Nf = ctx->species_offset[grid+1] - ctx->species_offset[grid], N, bs;
              Vec      Mfsub = globMfArray[ LAND_PACK_IDX(ctx->batch_view_idx,grid) ], Gsub = globGammaArray[ LAND_PACK_IDX(ctx->batch_view_idx,grid) ], v1, v2;
              // get each component
              ierr = VecGetSize(Mfsub,&N);CHKERRQ(ierr);
              ierr = VecCreate(ctx->comm,&v1);CHKERRQ(ierr);
              ierr = VecSetSizes(v1,PETSC_DECIDE,N/Nf);CHKERRQ(ierr);
              ierr = VecCreate(ctx->comm,&v2);CHKERRQ(ierr);
              ierr = VecSetSizes(v2,PETSC_DECIDE,N/Nf);CHKERRQ(ierr);
              ierr = VecSetFromOptions(v1);CHKERRQ(ierr); // ???
              ierr = VecSetFromOptions(v2);CHKERRQ(ierr);
              // get each component
              ierr = VecGetBlockSize(Gsub,&bs);CHKERRQ(ierr);
              PetscCheckFalse(bs != Nf,PETSC_COMM_SELF, PETSC_ERR_PLIB, "bs %D != num_species %D in Gsub",bs,Nf);
              ierr = VecGetBlockSize(Mfsub,&bs);CHKERRQ(ierr);
              PetscCheckFalse(bs != Nf,PETSC_COMM_SELF, PETSC_ERR_PLIB, "bs %D != num_species %D",bs,Nf);
              for (int i=0, ix=ctx->species_offset[grid] ; i<Nf ; i++, ix++) {
                PetscScalar val;
                ierr = VecStrideGather(Gsub,i,v1,INSERT_VALUES);CHKERRQ(ierr);
                ierr = VecStrideGather(Mfsub,i,v2,INSERT_VALUES);CHKERRQ(ierr);
                ierr = VecDot(v1,v2,&val);CHKERRQ(ierr);
                energy[ix] = PetscRealPart(val)*ctx->n_0*ctx->v_0*ctx->v_0*ctx->masses[ix];
              }
              ierr = VecDestroy(&v1);CHKERRQ(ierr);
              ierr = VecDestroy(&v2);CHKERRQ(ierr);
            } /* grids */
            ierr = DMCompositeRestoreAccessArray(pack, globGamma, nDMs, NULL, globGammaArray);CHKERRQ(ierr);
            ierr = DMCompositeRestoreAccessArray(pack, Mf, nDMs, NULL, globMfArray);CHKERRQ(ierr);
            ierr = PetscFree(globGammaArray);CHKERRQ(ierr);
            ierr = PetscFree(globMfArray);CHKERRQ(ierr);
            ierr = VecDestroy(&globGamma);CHKERRQ(ierr);
            ierr = VecDestroy(&Mf);CHKERRQ(ierr);
          }
        } else {
          ierr = PetscDSSetObjective(prob, 0, &f0_s_v2);CHKERRQ(ierr);
          ierr = DMPlexComputeIntegralFEM(ctx->plex[grid],Xloc,tt,ctx);CHKERRQ(ierr);
          energy[ii]    = 0.5*tt[0]*ctx->n_0*ctx->v_0*ctx->v_0*ctx->masses[ii];
        }
        ierr = PetscPrintf(ctx->comm, "%3D) species %D: density=%20.13e, x-momentum=%20.13e, y-momentum=%20.13e, z-momentum=%20.13e, energy=%21.13e",
                           stepi,ii,PetscRealPart(density[ii]),PetscRealPart(xmomentum[ii]),PetscRealPart(ymomentum[ii]),PetscRealPart(zmomentum[ii]),PetscRealPart(energy[ii]));CHKERRQ(ierr);
        xmomentumtot += xmomentum[ii];
        ymomentumtot += ymomentum[ii];
        zmomentumtot += zmomentum[ii];
        energytot  += energy[ii];
        densitytot += density[ii];
      }
      if (ctx->num_species>1) PetscPrintf(ctx->comm, "\n");
    }
  }
  ierr = DMCompositeRestoreAccessArray(pack, X, nDMs, NULL, globXArray);CHKERRQ(ierr);
  ierr = PetscFree(globXArray);CHKERRQ(ierr);
  /* totals */
  ierr = DMPlexGetHeightStratum(ctx->plex[0],0,&cStart,&cEnd);CHKERRQ(ierr);
  if (ctx->num_species>1) {
    if (dim==2) {
      ierr = PetscPrintf(ctx->comm, "\t%3D) Total: charge density=%21.13e, momentum=%21.13e, energy=%21.13e (m_i[0]/m_e = %g, %D cells on electron grid)",
                         stepi,(double)PetscRealPart(densitytot),(double)PetscRealPart(zmomentumtot),(double)PetscRealPart(energytot),(double)(ctx->masses[1]/ctx->masses[0]),cEnd-cStart);CHKERRQ(ierr);
    } else {
      ierr = PetscPrintf(ctx->comm, "\t%3D) Total: charge density=%21.13e, x-momentum=%21.13e, y-momentum=%21.13e, z-momentum=%21.13e, energy=%21.13e (m_i[0]/m_e = %g, %D cells)",
                         stepi,(double)PetscRealPart(densitytot),(double)PetscRealPart(xmomentumtot),(double)PetscRealPart(ymomentumtot),(double)PetscRealPart(zmomentumtot),(double)PetscRealPart(energytot),(double)(ctx->masses[1]/ctx->masses[0]),cEnd-cStart);CHKERRQ(ierr);
    }
  } else {
    ierr = PetscPrintf(ctx->comm, " -- %D cells",cEnd-cStart);CHKERRQ(ierr);
  }
  if (ctx->verbose > 1) {ierr = PetscPrintf(ctx->comm,", %D sub (vector) threads\n",ctx->subThreadBlockSize);CHKERRQ(ierr);}
  else {ierr = PetscPrintf(ctx->comm,"\n");CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/*@
 LandauCreateMassMatrix - Create mass matrix for Landau in Plex space (not field major order of Jacobian)

 Collective on pack

 Input Parameters:
 . pack     - the DM object

 Output Parameters:
 . Amat - The mass matrix (optional), mass matrix is added to the DM context

 Level: beginner

 .keywords: mesh
 .seealso: LandauCreateVelocitySpace()
 @*/
PetscErrorCode LandauCreateMassMatrix(DM pack, Mat *Amat)
{
  DM             mass_pack,massDM[LANDAU_MAX_GRIDS];
  PetscDS        prob;
  PetscInt       ii,dim,N1=1,N2;
  PetscErrorCode ierr;
  LandauCtx      *ctx;
  Mat            packM,subM[LANDAU_MAX_GRIDS];

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pack,DM_CLASSID,1);
  if (Amat) PetscValidPointer(Amat,2);
  ierr = DMGetApplicationContext(pack, &ctx);CHKERRQ(ierr);
  PetscCheck(ctx,PETSC_COMM_SELF, PETSC_ERR_PLIB, "no context");
  ierr = PetscLogEventBegin(ctx->events[14],0,0,0,0);CHKERRQ(ierr);
  ierr = DMGetDimension(pack, &dim);CHKERRQ(ierr);
  ierr = DMCompositeCreate(PetscObjectComm((PetscObject) pack),&mass_pack);CHKERRQ(ierr);
  /* create pack mass matrix */
  for (PetscInt grid=0, ix=0 ; grid<ctx->num_grids ; grid++) {
    ierr = DMClone(ctx->plex[grid], &massDM[grid]);CHKERRQ(ierr);
    ierr = DMCopyFields(ctx->plex[grid], massDM[grid]);CHKERRQ(ierr);
    ierr = DMCreateDS(massDM[grid]);CHKERRQ(ierr);
    ierr = DMGetDS(massDM[grid], &prob);CHKERRQ(ierr);
    for (ix=0, ii=ctx->species_offset[grid];ii<ctx->species_offset[grid+1];ii++,ix++) {
      if (dim==3) {ierr = PetscDSSetJacobian(prob, ix, ix, g0_1, NULL, NULL, NULL);CHKERRQ(ierr);}
      else        {ierr = PetscDSSetJacobian(prob, ix, ix, g0_r, NULL, NULL, NULL);CHKERRQ(ierr);}
    }
#if !defined(LANDAU_SPECIES_MAJOR)
    ierr = DMCompositeAddDM(mass_pack,massDM[grid]);CHKERRQ(ierr);
#else
    for (PetscInt b_id=0;b_id<ctx->batch_sz;b_id++) { // add batch size DMs for this species grid
      ierr = DMCompositeAddDM(mass_pack,massDM[grid]);CHKERRQ(ierr);
    }
#endif
    ierr = DMCreateMatrix(massDM[grid], &subM[grid]);CHKERRQ(ierr);
  }
#if !defined(LANDAU_SPECIES_MAJOR)
  // stack the batched DMs
  for (PetscInt b_id=1;b_id<ctx->batch_sz;b_id++) {
    for (PetscInt grid=0;grid<ctx->num_grids;grid++) {
      ierr = DMCompositeAddDM(mass_pack, massDM[grid]);CHKERRQ(ierr);
    }
  }
#endif
  ierr = PetscOptionsInsertString(NULL,"-dm_preallocate_only");
  ierr = DMSetFromOptions(mass_pack);CHKERRQ(ierr);
  ierr = DMCreateMatrix(mass_pack, &packM);CHKERRQ(ierr);
  ierr = PetscOptionsInsertString(NULL,"-dm_preallocate_only false");
  ierr = MatSetOption(packM, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatSetOption(packM, MAT_STRUCTURALLY_SYMMETRIC, PETSC_TRUE);CHKERRQ(ierr);
  ierr = DMDestroy(&mass_pack);CHKERRQ(ierr);
  /* make mass matrix for each block */
  for (PetscInt grid=0;grid<ctx->num_grids;grid++) {
    Vec locX;
    DM  plex = massDM[grid];
    ierr = DMGetLocalVector(plex, &locX);CHKERRQ(ierr);
    /* Mass matrix is independent of the input, so no need to fill locX */
    ierr = DMPlexSNESComputeJacobianFEM(plex, locX, subM[grid], subM[grid], ctx);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(plex, &locX);CHKERRQ(ierr);
    ierr = DMDestroy(&massDM[grid]);CHKERRQ(ierr);
  }
  ierr = MatGetSize(ctx->J, &N1, NULL);CHKERRQ(ierr);
  ierr = MatGetSize(packM, &N2, NULL);CHKERRQ(ierr);
  PetscCheckFalse(N1 != N2,PetscObjectComm((PetscObject) pack), PETSC_ERR_PLIB, "Incorrect matrix sizes: |Jacobian| = %D, |Mass|=%D",N1,N2);
  /* assemble block diagonals */
  for (PetscInt grid=0 ; grid<ctx->num_grids ; grid++) {
    Mat               B = subM[grid];
    PetscInt          nloc, nzl, colbuf[1024], row;
    ierr = MatGetSize(B, &nloc, NULL);CHKERRQ(ierr);
    for (PetscInt b_id = 0 ; b_id < ctx->batch_sz ; b_id++) {
      const PetscInt    moffset = LAND_MOFFSET(b_id,grid,ctx->batch_sz,ctx->num_grids,ctx->mat_offset);
      const PetscInt    *cols;
      const PetscScalar *vals;
      for (int i=0 ; i<nloc ; i++) {
        ierr = MatGetRow(B,i,&nzl,&cols,&vals);CHKERRQ(ierr);
        PetscCheckFalse(nzl>1024,PetscObjectComm((PetscObject) pack), PETSC_ERR_PLIB, "Row too big: %D",nzl);
        for (int j=0; j<nzl; j++) colbuf[j] = cols[j] + moffset;
        row = i + moffset;
        ierr = MatSetValues(packM,1,&row,nzl,colbuf,vals,INSERT_VALUES);CHKERRQ(ierr);
        ierr = MatRestoreRow(B,i,&nzl,&cols,&vals);CHKERRQ(ierr);
      }
    }
  }
  // cleanup
  for (PetscInt grid=0 ; grid<ctx->num_grids ; grid++) {
    ierr = MatDestroy(&subM[grid]);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(packM,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(packM,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)packM, "mass");CHKERRQ(ierr);
  ierr = MatViewFromOptions(packM,NULL,"-dm_landau_mass_view");CHKERRQ(ierr);
  ctx->M = packM;
  if (Amat) *Amat = packM;
  ierr = PetscLogEventEnd(ctx->events[14],0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
 LandauIFunction - TS residual calculation

 Collective on ts

 Input Parameters:
 +   TS  - The time stepping context
 .   time_dummy - current time (not used)
 -   X - Current state
 +   X_t - Time derivative of current state
 .   actx - Landau context

 Output Parameter:
 .   F  - The residual

 Level: beginner

 .keywords: mesh
 .seealso: LandauCreateVelocitySpace(), LandauIJacobian()
 @*/
PetscErrorCode LandauIFunction(TS ts, PetscReal time_dummy, Vec X, Vec X_t, Vec F, void *actx)
{
  PetscErrorCode ierr;
  LandauCtx      *ctx=(LandauCtx*)actx;
  PetscInt       dim;
  DM             pack;
#if defined(PETSC_HAVE_THREADSAFETY)
  double         starttime, endtime;
#endif

  PetscFunctionBegin;
  ierr = TSGetDM(ts,&pack);CHKERRQ(ierr);
  ierr = DMGetApplicationContext(pack, &ctx);CHKERRQ(ierr);
  PetscCheckFalse(!ctx,PETSC_COMM_SELF, PETSC_ERR_PLIB, "no context");
  if (ctx->stage) {
    ierr = PetscLogStagePush(ctx->stage);CHKERRQ(ierr);
  }
  ierr = PetscLogEventBegin(ctx->events[11],0,0,0,0);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(ctx->events[0],0,0,0,0);CHKERRQ(ierr);
#if defined(PETSC_HAVE_THREADSAFETY)
  starttime = MPI_Wtime();
#endif
  ierr = DMGetDimension(pack, &dim);CHKERRQ(ierr);
  if (!ctx->aux_bool) {
    ierr = PetscInfo(ts, "Create Landau Jacobian t=%g X=%p %s\n",time_dummy,X_t,ctx->aux_bool ? " -- seems to be in line search" : "");CHKERRQ(ierr);
    ierr = LandauFormJacobian_Internal(X,ctx->J,dim,0.0,(void*)ctx);CHKERRQ(ierr);
    ierr = MatViewFromOptions(ctx->J, NULL, "-dm_landau_jacobian_view");CHKERRQ(ierr);
    ctx->aux_bool = PETSC_TRUE;
  } else {
    ierr = PetscInfo(ts, "Skip forming Jacobian, has not changed (should check norm)\n");CHKERRQ(ierr);
  }
  /* mat vec for op */
  ierr = MatMult(ctx->J,X,F);CHKERRQ(ierr);CHKERRQ(ierr); /* C*f */
  /* add time term */
  if (X_t) {
    ierr = MatMultAdd(ctx->M,X_t,F,F);CHKERRQ(ierr);
  }
#if defined(PETSC_HAVE_THREADSAFETY)
  if (ctx->stage) {
    endtime = MPI_Wtime();
    ctx->times[LANDAU_OPERATOR] += (endtime - starttime);
    ctx->times[LANDAU_JACOBIAN] += (endtime - starttime);
    ctx->times[LANDAU_JACOBIAN_COUNT] += 1;
  }
#endif
  ierr = PetscLogEventEnd(ctx->events[0],0,0,0,0);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(ctx->events[11],0,0,0,0);CHKERRQ(ierr);
  if (ctx->stage) {
    ierr = PetscLogStagePop();CHKERRQ(ierr);
#if defined(PETSC_HAVE_THREADSAFETY)
    ctx->times[LANDAU_MATRIX_TOTAL] += (endtime - starttime);
#endif
  }
  PetscFunctionReturn(0);
}

/*@
 LandauIJacobian - TS Jacobian construction

 Collective on ts

 Input Parameters:
 +   TS  - The time stepping context
 .   time_dummy - current time (not used)
 -   X - Current state
 +   U_tdummy - Time derivative of current state (not used)
 .   shift - shift for du/dt term
 -   actx - Landau context

 Output Parameter:
 .   Amat  - Jacobian
 +   Pmat  - same as Amat

 Level: beginner

 .keywords: mesh
 .seealso: LandauCreateVelocitySpace(), LandauIFunction()
 @*/
PetscErrorCode LandauIJacobian(TS ts, PetscReal time_dummy, Vec X, Vec U_tdummy, PetscReal shift, Mat Amat, Mat Pmat, void *actx)
{
  PetscErrorCode ierr;
  LandauCtx      *ctx=NULL;
  PetscInt       dim;
  DM             pack;
#if defined(PETSC_HAVE_THREADSAFETY)
  double         starttime, endtime;
#endif
  PetscFunctionBegin;
  ierr = TSGetDM(ts,&pack);CHKERRQ(ierr);
  ierr = DMGetApplicationContext(pack, &ctx);CHKERRQ(ierr);
  PetscCheckFalse(!ctx,PETSC_COMM_SELF, PETSC_ERR_PLIB, "no context");
  PetscCheckFalse(Amat!=Pmat || Amat!=ctx->J,ctx->comm, PETSC_ERR_PLIB, "Amat!=Pmat || Amat!=ctx->J");
  ierr = DMGetDimension(pack, &dim);CHKERRQ(ierr);
  /* get collision Jacobian into A */
  if (ctx->stage) {
    ierr = PetscLogStagePush(ctx->stage);CHKERRQ(ierr);
  }
  ierr = PetscLogEventBegin(ctx->events[11],0,0,0,0);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(ctx->events[9],0,0,0,0);CHKERRQ(ierr);
#if defined(PETSC_HAVE_THREADSAFETY)
  starttime = MPI_Wtime();
#endif
  ierr = PetscInfo(ts, "Adding just mass to Jacobian t=%g, shift=%g\n",(double)time_dummy,(double)shift);CHKERRQ(ierr);
  PetscCheckFalse(shift==0.0,ctx->comm, PETSC_ERR_PLIB, "zero shift");
  PetscCheckFalse(!ctx->aux_bool,ctx->comm, PETSC_ERR_PLIB, "wrong state");
  if (!ctx->use_matrix_mass) {
    ierr = LandauFormJacobian_Internal(X,ctx->J,dim,shift,(void*)ctx);CHKERRQ(ierr);
    ierr = MatViewFromOptions(ctx->J, NULL, "-dm_landau_mat_view");CHKERRQ(ierr);
  } else { /* add mass */
    ierr = MatAXPY(Pmat,shift,ctx->M,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  }
  ctx->aux_bool = PETSC_FALSE;
#if defined(PETSC_HAVE_THREADSAFETY)
  if (ctx->stage) {
    endtime = MPI_Wtime();
    ctx->times[LANDAU_OPERATOR] += (endtime - starttime);
    ctx->times[LANDAU_MASS] += (endtime - starttime);
  }
#endif
  ierr = PetscLogEventEnd(ctx->events[9],0,0,0,0);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(ctx->events[11],0,0,0,0);CHKERRQ(ierr);
  if (ctx->stage) {
    ierr = PetscLogStagePop();CHKERRQ(ierr);
#if defined(PETSC_HAVE_THREADSAFETY)
    ctx->times[LANDAU_MATRIX_TOTAL] += (endtime - starttime);
#endif
  }
  PetscFunctionReturn(0);
}
