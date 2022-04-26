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

static PetscErrorCode LandauGPUMapsDestroy(void *ptr)
{
  P4estVertexMaps *maps = (P4estVertexMaps*)ptr;
  PetscFunctionBegin;
  // free device data
  if (maps[0].deviceType != LANDAU_CPU) {
#if defined(PETSC_HAVE_KOKKOS_KERNELS)
    if (maps[0].deviceType == LANDAU_KOKKOS) {
      PetscCall(LandauKokkosDestroyMatMaps(maps,  maps[0].numgrids)); // imples Kokkos does
    } // else could be CUDA
#elif defined(PETSC_HAVE_CUDA)
    if (maps[0].deviceType == LANDAU_CUDA) {
      PetscCall(LandauCUDADestroyMatMaps(maps, maps[0].numgrids));
    } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "maps->deviceType %d ?????",maps->deviceType);
#endif
  }
  // free host data
  for (PetscInt grid=0 ; grid < maps[0].numgrids ; grid++) {
    PetscCall(PetscFree(maps[grid].c_maps));
    PetscCall(PetscFree(maps[grid].gIdx));
  }
  PetscCall(PetscFree(maps));

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
  PetscCheck(ctx->plex[0] != NULL,ctx->comm,PETSC_ERR_ARG_WRONG,"Plex not created");
  PetscCall(PetscLogEventBegin(ctx->events[10],0,0,0,0));
  PetscCall(DMGetDS(ctx->plex[0], &prob)); // same DS for all grids
  PetscCall(PetscObjectQuery((PetscObject) JacP, "assembly_maps", (PetscObject *) &container));
  if (container) {
    PetscCheck(ctx->gpu_assembly,ctx->comm,PETSC_ERR_ARG_WRONG,"maps but no GPU assembly");
    PetscCall(PetscContainerGetPointer(container, (void **) &maps));
    PetscCheck(maps,ctx->comm,PETSC_ERR_ARG_WRONG,"empty GPU matrix container");
    for (PetscInt i=0;i<ctx->num_grids*ctx->batch_sz;i++) subJ[i] = NULL;
  } else {
    PetscCheck(!ctx->gpu_assembly,ctx->comm,PETSC_ERR_ARG_WRONG,"No maps but GPU assembly");
    for (PetscInt tid=0 ; tid<ctx->batch_sz ; tid++) {
      for (PetscInt grid=0;grid<ctx->num_grids;grid++) {
        PetscCall(DMCreateMatrix(ctx->plex[grid], &subJ[ LAND_PACK_IDX(tid,grid) ]));
      }
    }
    maps = NULL;
  }
  // get dynamic data (Eq is odd, for quench and Spitzer test) for CPU assembly and raw data for Jacobian GPU assembly. Get host numCells[], Nq (yuck)
  PetscCall(PetscFEGetQuadrature(ctx->fe[0], &quad));
  PetscCall(PetscQuadratureGetData(quad, NULL, NULL, &Nq, NULL, NULL)); Nb = Nq;
  PetscCheck(Nq <=LANDAU_MAX_NQ,ctx->comm,PETSC_ERR_ARG_WRONG,"Order too high. Nq = %" PetscInt_FMT " > LANDAU_MAX_NQ (%d)",Nq,LANDAU_MAX_NQ);
  // get metadata for collecting dynamic data
  for (PetscInt grid=0;grid<ctx->num_grids;grid++) {
    PetscInt cStart, cEnd;
    PetscCheck(ctx->plex[grid] != NULL,ctx->comm,PETSC_ERR_ARG_WRONG,"Plex not created");
    PetscCall(DMPlexGetHeightStratum(ctx->plex[grid], 0, &cStart, &cEnd));
    numCells[grid] = cEnd - cStart; // grids can have different topology
  }
  PetscCall(PetscLogEventEnd(ctx->events[10],0,0,0,0));
  if (shift==0) { /* create dynamic point data: f_alpha for closure of each cell (cellClosure[nbatch,ngrids,ncells[g],f[Nb,ns[g]]]) or xdata */
    DM pack;
    PetscCall(VecGetDM(a_X, &pack));
    PetscCheck(pack,PETSC_COMM_SELF, PETSC_ERR_PLIB, "pack has no DM");
    PetscCall(PetscLogEventBegin(ctx->events[1],0,0,0,0));
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
        PetscCall(DMGetLocalSection(ctx->plex[grid], &section[grid]));
        PetscCall(DMGetGlobalSection(ctx->plex[grid], &globsection[grid]));
        PetscCall(PetscSectionGetNumFields(section[grid], &Nf[grid]));
      }
      /* count cellClosure size */
      PetscCall(DMCompositeGetNumberDM(pack,&nDMs));
      for (PetscInt grid=0 ; grid<ctx->num_grids ; grid++) cellClosure_sz += Nb*Nf[grid]*numCells[grid];
      PetscCall(PetscMalloc1(cellClosure_sz*ctx->batch_sz,&cellClosure));
      cellClosure_it = cellClosure;
      PetscCall(PetscMalloc(sizeof(*locXArray)*nDMs, &locXArray));
      PetscCall(PetscMalloc(sizeof(*globXArray)*nDMs, &globXArray));
      PetscCall(DMCompositeGetLocalAccessArray(pack, a_X, nDMs, NULL, locXArray));
      PetscCall(DMCompositeGetAccessArray(pack, a_X, nDMs, NULL, globXArray));
      for (PetscInt b_id = 0 ; b_id < ctx->batch_sz ; b_id++) { // OpenMP (once)
        for (PetscInt grid=0 ; grid<ctx->num_grids ; grid++) {
          Vec         locX = locXArray[ LAND_PACK_IDX(b_id,grid) ], globX = globXArray[ LAND_PACK_IDX(b_id,grid) ], locX2;
          PetscInt    cStart, cEnd, ei;
          PetscCall(VecDuplicate(locX,&locX2));
          PetscCall(DMGlobalToLocalBegin(ctx->plex[grid], globX, INSERT_VALUES, locX2));
          PetscCall(DMGlobalToLocalEnd  (ctx->plex[grid], globX, INSERT_VALUES, locX2));
          PetscCall(DMPlexGetHeightStratum(ctx->plex[grid], 0, &cStart, &cEnd));
          for (ei = cStart ; ei < cEnd; ++ei) {
            PetscScalar *coef = NULL;
            PetscCall(DMPlexVecGetClosure(ctx->plex[grid], section[grid], locX2, ei, NULL, &coef));
            PetscCall(PetscMemcpy(cellClosure_it,coef,Nb*Nf[grid]*sizeof(*cellClosure_it))); /* change if LandauIPReal != PetscScalar */
            PetscCall(DMPlexVecRestoreClosure(ctx->plex[grid], section[grid], locX2, ei, NULL, &coef));
            cellClosure_it += Nb*Nf[grid];
          }
          PetscCall(VecDestroy(&locX2));
        }
      }
      PetscCheck(cellClosure_it-cellClosure == cellClosure_sz*ctx->batch_sz,PETSC_COMM_SELF, PETSC_ERR_PLIB, "iteration wrong %" PetscCount_FMT " != cellClosure_sz = %" PetscInt_FMT,(PetscCount)(cellClosure_it-cellClosure),cellClosure_sz*ctx->batch_sz);
      PetscCall(DMCompositeRestoreLocalAccessArray(pack, a_X, nDMs, NULL, locXArray));
      PetscCall(DMCompositeRestoreAccessArray(pack, a_X, nDMs, NULL, globXArray));
      PetscCall(PetscFree(locXArray));
      PetscCall(PetscFree(globXArray));
      xdata = NULL;
    } else {
      PetscMemType mtype;
      if (ctx->jacobian_field_major_order) { // get data in batch ordering
        PetscCall(VecScatterBegin(ctx->plex_batch,a_X,ctx->work_vec,INSERT_VALUES,SCATTER_FORWARD));
        PetscCall(VecScatterEnd(ctx->plex_batch,a_X,ctx->work_vec,INSERT_VALUES,SCATTER_FORWARD));
        PetscCall(VecGetArrayReadAndMemType(ctx->work_vec,&xdata,&mtype));
      } else {
        PetscCall(VecGetArrayReadAndMemType(a_X,&xdata,&mtype));
      }
      if (mtype!=PETSC_MEMTYPE_HOST && ctx->deviceType == LANDAU_CPU) {
        SETERRQ(ctx->comm,PETSC_ERR_ARG_WRONG,"CPU run with device data: use -mat_type aij");
      }
      cellClosure = NULL;
    }
    PetscCall(PetscLogEventEnd(ctx->events[1],0,0,0,0));
  } else xdata = cellClosure = NULL;

  /* do it */
  if (ctx->deviceType == LANDAU_CUDA || ctx->deviceType == LANDAU_KOKKOS) {
    if (ctx->deviceType == LANDAU_CUDA) {
#if defined(PETSC_HAVE_CUDA)
      PetscCall(LandauCUDAJacobian(ctx->plex,Nq,ctx->batch_sz,ctx->num_grids,numCells,Eq_m,cellClosure,xdata,&ctx->SData_d,shift,ctx->events,ctx->mat_offset, ctx->species_offset, subJ, JacP));
#else
      SETERRQ(ctx->comm,PETSC_ERR_ARG_WRONG,"-landau_device_type %s not built","cuda");
#endif
    } else if (ctx->deviceType == LANDAU_KOKKOS) {
#if defined(PETSC_HAVE_KOKKOS_KERNELS)
      PetscCall(LandauKokkosJacobian(ctx->plex,Nq,ctx->batch_sz,ctx->num_grids,numCells,Eq_m,cellClosure,xdata,&ctx->SData_d,shift,ctx->events,ctx->mat_offset, ctx->species_offset, subJ,JacP));
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
    PetscScalar     *coo_vals=NULL;
    for (PetscInt grid=0;grid<ctx->num_grids;grid++) {
      PetscCall(DMGetLocalSection(ctx->plex[grid], &section[grid]));
      PetscCall(DMGetGlobalSection(ctx->plex[grid], &globsection[grid]));
      PetscCall(PetscSectionGetNumFields(section[grid], &Nf[grid]));
    }
    /* count IPf size, etc */
    PetscCall(PetscDSGetTabulation(prob, &Tf)); // Bf, &Df same for all grids
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
    // prep COO
    if (ctx->coo_assembly) {
      PetscCall(PetscMalloc1(ctx->SData_d.coo_size,&coo_vals)); // allocate every time?
      PetscCall(PetscInfo(ctx->plex[0], "COO Allocate %" PetscInt_FMT " values\n",(PetscInt)ctx->SData_d.coo_size));
    }
    if (shift==0.0) { /* compute dynamic data f and df and init data for Jacobian */
#if defined(PETSC_HAVE_THREADSAFETY)
      double         starttime, endtime;
      starttime = MPI_Wtime();
#endif
      PetscCall(PetscLogEventBegin(ctx->events[8],0,0,0,0));
      for (PetscInt fieldA=0;fieldA<ctx->num_species;fieldA++) {
        invMass[fieldA]  = ctx->m_0/ctx->masses[fieldA];
        Eq_m[fieldA]     = ctx->Ez * ctx->t_0 * ctx->charges[fieldA] / (ctx->v_0 * ctx->masses[fieldA]); /* normalize dimensionless */
        if (dim==2) Eq_m[fieldA] *=  2 * PETSC_PI; /* add the 2pi term that is not in Landau */
        nu_alpha[fieldA] = PetscSqr(ctx->charges[fieldA]/ctx->m_0)*ctx->m_0/ctx->masses[fieldA];
        nu_beta[fieldA]  = PetscSqr(ctx->charges[fieldA]/ctx->epsilon0)*ctx->lnLam / (8*PETSC_PI) * ctx->t_0*ctx->n_0/PetscPowReal(ctx->v_0,3);
      }
      PetscCall(PetscMalloc4(IPf_sz_tot, &ff, IPf_sz_tot, &dudx, IPf_sz_tot, &dudy, dim==3 ? IPf_sz_tot : 0, &dudz));
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
          PetscReal          *invJe  = &invJ_a[(ip_offset[grid] + loc_elem*Nq)*dim*dim]; // ingJ is static data on batch 0
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
                    PetscInt    id    = maps[grid].c_maps[idx][q].gid;
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
            const PetscReal  *Bq   = &BB[qi*Nb];
            const PetscReal  *Dq   = &DD[qi*Nb*dim];
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
      PetscCall(PetscLogEventEnd(ctx->events[8],0,0,0,0));
#if defined(PETSC_HAVE_THREADSAFETY)
      endtime = MPI_Wtime();
      if (ctx->stage) ctx->times[LANDAU_F_DF] += (endtime - starttime);
#endif
    } // Jacobian setup
    // assemble Jacobian (or mass)
    for (PetscInt tid = 0 ; tid < ctx->batch_sz*elem_offset[num_grids] ; tid++) { // for each element
      const PetscInt b_Nelem      = elem_offset[num_grids];
      const PetscInt glb_elem_idx = tid%b_Nelem, b_id = tid/b_Nelem;
      PetscInt       grid         = 0;
#if defined(PETSC_HAVE_THREADSAFETY)
      double         starttime, endtime;
      starttime                   = MPI_Wtime();
#endif
      while (glb_elem_idx >= elem_offset[grid+1]) grid++;
      {
        const PetscInt     loc_Nf  = ctx->species_offset[grid+1] - ctx->species_offset[grid], loc_elem = glb_elem_idx - elem_offset[grid];
        const PetscInt     moffset = LAND_MOFFSET(b_id,grid,ctx->batch_sz,ctx->num_grids,ctx->mat_offset), totDim = loc_Nf*Nq, elemMatSize = totDim*totDim;
        PetscScalar        *elemMat;
         const PetscReal   *invJe  = &invJ_a[(ip_offset[grid] + loc_elem*Nq)*dim*dim];
        PetscCall(PetscMalloc1(elemMatSize, &elemMat));
        PetscCall(PetscMemzero(elemMat, elemMatSize*sizeof(*elemMat)));
        if (shift==0.0) { // Jacobian
          PetscCall(PetscLogEventBegin(ctx->events[4],0,0,0,0));
        } else {          // mass
          PetscCall(PetscLogEventBegin(ctx->events[16],0,0,0,0));
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
                  const PetscReal wi       = ww[ipidx], x = xx[ipidx], y = yy[ipidx];
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
            PetscCheck(IPf_idx == IPf_sz_glb,PETSC_COMM_SELF, PETSC_ERR_PLIB, "IPf_idx != IPf_sz %" PetscInt_FMT " %" PetscInt_FMT,IPf_idx,IPf_sz_glb);
            // add alpha and put in gg2/3
            for (PetscInt fieldA = 0, f_off = ctx->species_offset[grid]; fieldA < loc_Nf; ++fieldA) {
              for (d2 = 0; d2 < LANDAU_DIM; d2++) {
                gg2[fieldA][d2] = gg2_temp[d2]*nu_alpha[fieldA+f_off];
                for (d3 = 0; d3 < LANDAU_DIM; d3++) {
                  gg3[fieldA][d2][d3] = -gg3_temp[d2][d3]*nu_alpha[fieldA+f_off]*invMass[fieldA+f_off];
                }
              }
            }
            /* add electric field term once per IP */
            for (PetscInt fieldA = 0, f_off = ctx->species_offset[grid] ; fieldA < loc_Nf; ++fieldA) {
              gg2[fieldA][LANDAU_DIM-1] += Eq_m[fieldA+f_off];
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
          PetscCall(PetscLogEventEnd(ctx->events[4],0,0,0,0));
        } else {
          PetscCall(PetscLogEventEnd(ctx->events[16],0,0,0,0));
        }
#if defined(PETSC_HAVE_THREADSAFETY)
        endtime = MPI_Wtime();
        if (ctx->stage) ctx->times[LANDAU_KERNEL] += (endtime - starttime);
#endif
        /* assemble matrix */
        if (!container) {
          PetscInt cStart;
          PetscCall(PetscLogEventBegin(ctx->events[6],0,0,0,0));
          PetscCall(DMPlexGetHeightStratum(ctx->plex[grid], 0, &cStart, NULL));
          PetscCall(DMPlexMatSetClosure(ctx->plex[grid], section[grid], globsection[grid], subJ[ LAND_PACK_IDX(b_id,grid) ], loc_elem + cStart, elemMat, ADD_VALUES));
          PetscCall(PetscLogEventEnd(ctx->events[6],0,0,0,0));
        } else {  // GPU like assembly for debugging
          PetscInt      fieldA,q,f,g,d,nr,nc,rows0[LANDAU_MAX_Q_FACE]={0},cols0[LANDAU_MAX_Q_FACE]={0},rows[LANDAU_MAX_Q_FACE],cols[LANDAU_MAX_Q_FACE];
          PetscScalar   vals[LANDAU_MAX_Q_FACE*LANDAU_MAX_Q_FACE]={0},row_scale[LANDAU_MAX_Q_FACE]={0},col_scale[LANDAU_MAX_Q_FACE]={0};
          LandauIdx     *coo_elem_offsets = (LandauIdx*)ctx->SData_d.coo_elem_offsets, *coo_elem_fullNb = (LandauIdx*)ctx->SData_d.coo_elem_fullNb, (*coo_elem_point_offsets)[LANDAU_MAX_NQ+1] = (LandauIdx (*)[LANDAU_MAX_NQ+1])ctx->SData_d.coo_elem_point_offsets;
          /* assemble - from the diagonal (I,I) in this format for DMPlexMatSetClosure */
          for (fieldA = 0; fieldA < loc_Nf ; fieldA++) {
            LandauIdx *const Idxs = &maps[grid].gIdx[loc_elem][fieldA][0];
            for (f = 0; f < Nb ; f++) {
              PetscInt idx = Idxs[f];
              if (idx >= 0) {
                nr           = 1;
                rows0[0]     = idx;
                row_scale[0] = 1.;
              } else {
                idx = -idx - 1;
                for (q = 0, nr = 0; q < maps[grid].num_face; q++, nr++) {
                  if (maps[grid].c_maps[idx][q].gid < 0) break;
                  rows0[q]     = maps[grid].c_maps[idx][q].gid;
                  row_scale[q] = maps[grid].c_maps[idx][q].scale;
                }
              }
              for (g = 0; g < Nb; ++g) {
                idx = Idxs[g];
                if (idx >= 0) {
                  nc = 1;
                  cols0[0]     = idx;
                  col_scale[0] = 1.;
                } else {
                  idx = -idx - 1;
                  nc = maps[grid].num_face;
                  for (q = 0, nc = 0; q < maps[grid].num_face; q++, nc++) {
                    if (maps[grid].c_maps[idx][q].gid < 0) break;
                    cols0[q]     = maps[grid].c_maps[idx][q].gid;
                    col_scale[q] = maps[grid].c_maps[idx][q].scale;
                  }
                }
                const PetscInt    i   = fieldA*Nb + f; /* Element matrix row */
                const PetscInt    j   = fieldA*Nb + g; /* Element matrix column */
                const PetscScalar Aij = elemMat[i*totDim + j];
                if (coo_vals) { // mirror (i,j) in CreateStaticGPUData
                  const int fullNb = coo_elem_fullNb[glb_elem_idx],fullNb2=fullNb*fullNb;
                  const int idx0   = b_id*coo_elem_offsets[elem_offset[num_grids]] + coo_elem_offsets[glb_elem_idx] + fieldA*fullNb2 + fullNb * coo_elem_point_offsets[glb_elem_idx][f] + nr * coo_elem_point_offsets[glb_elem_idx][g];
                  for (int q = 0, idx2 = idx0; q < nr; q++) {
                    for (int d = 0; d < nc; d++, idx2++) {
                      coo_vals[idx2] = row_scale[q]*col_scale[d]*Aij;
                    }
                  }
                } else {
                  for (q = 0; q < nr; q++) rows[q] = rows0[q] + moffset;
                  for (d = 0; d < nc; d++) cols[d] = cols0[d] + moffset;
                  for (q = 0; q < nr; q++) {
                    for (d = 0; d < nc; d++) {
                      vals[q*nc + d] = row_scale[q]*col_scale[d]*Aij;
                    }
                  }
                  PetscCall(MatSetValues(JacP,nr,rows,nc,cols,vals,ADD_VALUES));
                }
              }
            }
          }
        }
        if (loc_elem==-1) {
          PetscCall(PetscPrintf(ctx->comm,"CPU Element matrix\n"));
          for (int d = 0; d < totDim; ++d) {
            for (int f = 0; f < totDim; ++f) PetscCall(PetscPrintf(ctx->comm," %12.5e",  (double)PetscRealPart(elemMat[d*totDim + f])));
            PetscCall(PetscPrintf(ctx->comm,"\n"));
          }
          exit(12);
        }
        PetscCall(PetscFree(elemMat));
      } /* grid */
    } /* outer element & batch loop */
    if (shift==0.0) { // mass
      PetscCall(PetscFree4(ff, dudx, dudy, dudz));
    }
    if (!container) {   // 'CPU' assembly move nest matrix to global JacP
      for (PetscInt b_id = 0 ; b_id < ctx->batch_sz ; b_id++) { // OpenMP
        for (PetscInt grid=0 ; grid<ctx->num_grids ; grid++) {
          const PetscInt    moffset = LAND_MOFFSET(b_id,grid,ctx->batch_sz,ctx->num_grids,ctx->mat_offset); // b_id*b_N + ctx->mat_offset[grid];
          PetscInt          nloc, nzl, colbuf[1024], row;
          const PetscInt    *cols;
          const PetscScalar *vals;
          Mat               B = subJ[ LAND_PACK_IDX(b_id,grid) ];
          PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
          PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));
          PetscCall(MatGetSize(B, &nloc, NULL));
          for (int i=0 ; i<nloc ; i++) {
            PetscCall(MatGetRow(B,i,&nzl,&cols,&vals));
            PetscCheck(nzl<=1024,PetscObjectComm((PetscObject) B), PETSC_ERR_PLIB, "Row too big: %" PetscInt_FMT,nzl);
            for (int j=0; j<nzl; j++) colbuf[j] = moffset + cols[j];
            row  = moffset + i;
            PetscCall(MatSetValues(JacP,1,&row,nzl,colbuf,vals,ADD_VALUES));
            PetscCall(MatRestoreRow(B,i,&nzl,&cols,&vals));
          }
          PetscCall(MatDestroy(&B));
        }
      }
    }
    if (coo_vals) {
      PetscCall(MatSetValuesCOO(JacP,coo_vals,ADD_VALUES));
      PetscCall(PetscFree(coo_vals));
    }
  } /* CPU version */
  PetscCall(MatAssemblyBegin(JacP, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(JacP, MAT_FINAL_ASSEMBLY));
  /* clean up */
  if (cellClosure) PetscCall(PetscFree(cellClosure));
  if (xdata) {
    PetscCall(VecRestoreArrayReadAndMemType(a_X,&xdata));
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
      rotcos  = 0.70710678118654;
      outfact = 1.5; efact = 2.5;
      /* rotate normalized vector into [-pi/4,pi/4) */
      if (sinphi >= 0.) {         /* top cell, -pi/2 */
        cth = 0.707106781186548; sth = -0.707106781186548;
      } else {                    /* bottom cell -pi/8 */
        cth = 0.707106781186548; sth = .707106781186548;
      }
    } else if (num_sections==3) {
      rotcos  = 0.86602540378443;
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
      rotcos  = 0.9238795325112;
      outfact = 1.5; efact = 3;
      /* rotate normalized vector into [-pi/8,pi/8) */
      if (sinphi >= 0.707106781186548) {         /* top cell, -3pi/8 */
        cth = 0.38268343236509;  sth = -0.923879532511287;
      } else if (sinphi >= 0.) {                 /* mid top cell -pi/8 */
        cth = 0.923879532511287; sth = -.38268343236509;
      } else if (sinphi >= -0.707106781186548) { /* mid bottom cell + pi/8 */
        cth = 0.923879532511287; sth = 0.38268343236509;
      } else {                                   /* bottom cell + 3pi/8 */
        cth = 0.38268343236509;  sth = .923879532511287;
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
        PetscCall(DMPlexCreateBoxMesh(comm_self, dim, PETSC_FALSE, cells, lo, hi, periodicity, PETSC_TRUE, &ctx->plex[grid])); // todo: make composite and create dm[grid] here
        PetscCall(DMLocalizeCoordinates(ctx->plex[grid])); /* needed for periodic */
        if (dim==3) PetscCall(PetscObjectSetName((PetscObject) ctx->plex[grid], "cube"));
        else PetscCall(PetscObjectSetName((PetscObject) ctx->plex[grid], "half-plane"));
      } else if (dim==2) { // sphere is all wrong. should just have one inner radius
        PetscInt       numCells,cells[16][4],i,j;
        PetscInt       numVerts;
        PetscReal      inner_radius1 = ctx->i_radius[grid], inner_radius2 = ctx->e_radius;
        PetscReal      *flatCoords   = NULL;
        PetscInt       *flatCells    = NULL, *pcell;
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
          PetscCall(PetscMalloc2(numVerts * 2, &flatCoords, numCells * 4, &flatCells));
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
          PetscCall(loc2(numVerts * 2, &flatCoords, numCells * 4, &flatCells));
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
          PetscCall(PetscMalloc2(numVerts * 2, &flatCoords, numCells * 4, &flatCells));
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
          PetscCall(PetscMalloc2(numVerts * 2, &flatCoords, numCells * 4, &flatCells));
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
        PetscCall(DMPlexCreateFromCellListPetsc(comm_self,2,numCells,numVerts,4,ctx->interpolate,flatCells,2,flatCoords,&ctx->plex[grid]));
        PetscCall(PetscFree2(flatCoords,flatCells));
        PetscCall(PetscObjectSetName((PetscObject) ctx->plex[grid], "semi-circle"));
      } else SETERRQ(ctx->comm, PETSC_ERR_PLIB, "Velocity space meshes does not support cubed sphere");

      PetscCall(DMSetFromOptions(ctx->plex[grid]));
    } // grid loop
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)pack,prefix));
    PetscCall(DMSetFromOptions(pack));

    { /* convert to p4est (or whatever), wait for discretization to create pack */
      char           convType[256];
      PetscBool      flg;

      PetscOptionsBegin(ctx->comm, prefix, "Mesh conversion options", "DMPLEX");
      PetscCall(PetscOptionsFList("-dm_landau_type","Convert DMPlex to another format (p4est)","plexland.c",DMList,DMPLEX,convType,256,&flg));
      PetscOptionsEnd();
      if (flg) {
        ctx->use_p4est = PETSC_TRUE; /* flag for Forest */
        for (PetscInt grid=0;grid<ctx->num_grids;grid++) {
          DM dmforest;
          PetscCall(DMConvert(ctx->plex[grid],convType,&dmforest));
          if (dmforest) {
            PetscBool isForest;
            PetscCall(PetscObjectSetOptionsPrefix((PetscObject)dmforest,prefix));
            PetscCall(DMIsForest(dmforest,&isForest));
            if (isForest) {
              if (ctx->sphere && ctx->inflate) {
                PetscCall(DMForestSetBaseCoordinateMapping(dmforest,GeometryDMLandau,ctx));
              }
              PetscCall(DMDestroy(&ctx->plex[grid]));
              ctx->plex[grid] = dmforest; // Forest for adaptivity
            } else SETERRQ(ctx->comm, PETSC_ERR_PLIB, "Converted to non Forest?");
          } else SETERRQ(ctx->comm, PETSC_ERR_PLIB, "Convert failed?");
        }
      } else ctx->use_p4est = PETSC_FALSE; /* flag for Forest */
    }
  } /* non-file */
  PetscCall(DMSetDimension(pack, dim));
  PetscCall(PetscObjectSetName((PetscObject) pack, "Mesh"));
  PetscCall(DMSetApplicationContext(pack, ctx));

  PetscFunctionReturn(0);
}

static PetscErrorCode SetupDS(DM pack, PetscInt dim, PetscInt grid, LandauCtx *ctx)
{
  PetscInt        ii,i0;
  char            buf[256];
  PetscSection    section;

  PetscFunctionBegin;
  for (ii = ctx->species_offset[grid], i0 = 0 ; ii < ctx->species_offset[grid+1] ; ii++, i0++) {
    if (ii==0) PetscCall(PetscSNPrintf(buf, sizeof(buf), "e"));
    else PetscCall(PetscSNPrintf(buf, sizeof(buf), "i%" PetscInt_FMT, ii));
    /* Setup Discretization - FEM */
    PetscCall(PetscFECreateDefault(PETSC_COMM_SELF, dim, 1, PETSC_FALSE, NULL, PETSC_DECIDE, &ctx->fe[ii]));
    PetscCall(PetscObjectSetName((PetscObject) ctx->fe[ii], buf));
    PetscCall(DMSetField(ctx->plex[grid], i0, NULL, (PetscObject) ctx->fe[ii]));
  }
  PetscCall(DMCreateDS(ctx->plex[grid]));
  PetscCall(DMGetSection(ctx->plex[grid], &section));
  for (PetscInt ii = ctx->species_offset[grid], i0 = 0 ; ii < ctx->species_offset[grid+1] ; ii++, i0++) {
    if (ii==0) PetscCall(PetscSNPrintf(buf, sizeof(buf), "se"));
    else PetscCall(PetscSNPrintf(buf, sizeof(buf), "si%" PetscInt_FMT, ii));
    PetscCall(PetscSectionSetComponentName(section, i0, 0, buf));
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
 DMPlexLandauAddMaxwellians - Add a Maxwellian distribution to a state

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
 .seealso: `DMPlexLandauCreateVelocitySpace()`
 @*/
PetscErrorCode DMPlexLandauAddMaxwellians(DM dm, Vec X, PetscReal time, PetscReal temps[], PetscReal ns[], PetscInt grid, PetscInt b_id, PetscInt n_batch, void *actx)
{
  LandauCtx      *ctx = (LandauCtx*)actx;
  PetscErrorCode (*initu[LANDAU_MAX_SPECIES])(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar [], void *);
  PetscInt       dim;
  MaxwellianCtx  *mctxs[LANDAU_MAX_SPECIES], data[LANDAU_MAX_SPECIES];

  PetscFunctionBegin;
  PetscCall(DMGetDimension(dm, &dim));
  if (!ctx) PetscCall(DMGetApplicationContext(dm, &ctx));
  for (PetscInt ii = ctx->species_offset[grid], i0 = 0 ; ii < ctx->species_offset[grid+1] ; ii++, i0++) {
    mctxs[i0]      = &data[i0];
    data[i0].v_0   = ctx->v_0; // v_0 same for all grids
    data[i0].kT_m  = ctx->k*temps[ii]/ctx->masses[ii]; /* kT/m */
    data[i0].n     = ns[ii] * (1+0.1*(double)b_id/(double)n_batch); // ramp density up 10% to mimic application, n[0] use for Conner-Hastie
    initu[i0]      = maxwellian;
    data[i0].shift = 0;
  }
  data[0].shift = ctx->electronShift;
  /* need to make ADD_ALL_VALUES work - TODO */
  PetscCall(DMProjectFunction(dm, time, initu, (void**)mctxs, INSERT_ALL_VALUES, X));
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
 .seealso: `DMPlexLandauCreateVelocitySpace()`, `DMPlexLandauAddMaxwellians()`
 */
static PetscErrorCode LandauSetInitialCondition(DM dm, Vec X, PetscInt grid, PetscInt b_id, PetscInt n_batch, void *actx)
{
  LandauCtx        *ctx = (LandauCtx*)actx;
  PetscFunctionBegin;
  if (!ctx) PetscCall(DMGetApplicationContext(dm, &ctx));
  PetscCall(VecZeroEntries(X));
  PetscCall(DMPlexLandauAddMaxwellians(dm, X, 0.0, ctx->thermal_temps, ctx->n, grid, b_id, n_batch, ctx));
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

  PetscFunctionBegin;
  forest = ctx->plex[grid];
  PetscCall(DMCreateDS(forest));
  PetscCall(DMGetDS(forest, &prob));
  PetscCall(DMGetDimension(forest, &dim));
  PetscCall(DMIsForest(forest, &isForest));
  PetscCheck(isForest,ctx->comm,PETSC_ERR_ARG_WRONG,"! Forest");
  PetscCall(DMConvert(forest, DMPLEX, &plex));
  PetscCall(DMPlexGetHeightStratum(plex,0,&cStart,&cEnd));
  PetscCall(DMLabelCreate(PETSC_COMM_SELF,"adapt",&adaptLabel));
  PetscCall(PetscFEGetQuadrature(fem, &quad));
  PetscCall(PetscQuadratureGetData(quad, NULL, NULL, &Nq, NULL, NULL));
  PetscCheck(Nq <=LANDAU_MAX_NQ,ctx->comm,PETSC_ERR_ARG_WRONG,"Order too high. Nq = %" PetscInt_FMT " > LANDAU_MAX_NQ (%d)",Nq,LANDAU_MAX_NQ);
  PetscCall(PetscDSGetDimensions(prob, &Nb));
  if (type==4) {
    for (c = cStart; c < cEnd; c++) {
      PetscCall(DMLabelSetValue(adaptLabel, c, DM_ADAPT_REFINE));
    }
    PetscCall(PetscInfo(sol, "Phase:%s: Uniform refinement\n","adaptToleranceFEM"));
  } else if (type==2) {
    PetscInt  rCellIdx[8], eCellIdx[64], iCellIdx[64], eMaxIdx = -1, iMaxIdx = -1, nr = 0, nrmax = (dim==3) ? 8 : 2;
    PetscReal minRad = PETSC_INFINITY, r, eMinRad = PETSC_INFINITY, iMinRad = PETSC_INFINITY;
    for (c = 0; c < 64; c++) { eCellIdx[c] = iCellIdx[c] = -1; }
    for (c = cStart; c < cEnd; c++) {
      PetscReal    tt, v0[LANDAU_MAX_NQ*3], detJ[LANDAU_MAX_NQ];
      PetscCall(DMPlexComputeCellGeometryFEM(plex, c, quad, v0, NULL, NULL, detJ));
      for (qj = 0; qj < Nq; ++qj) {
        tt = PetscSqr(v0[dim*qj+0]) + PetscSqr(v0[dim*qj+1]) + PetscSqr(((dim==3) ? v0[dim*qj+2] : 0));
        r  = PetscSqrtReal(tt);
        if (r < minRad - PETSC_SQRT_MACHINE_EPSILON*10.) {
          minRad = r;
          nr     = 0;
          rCellIdx[nr++]= c;
          PetscCall(PetscInfo(sol, "\t\tPhase: adaptToleranceFEM Found first inner r=%e, cell %" PetscInt_FMT ", qp %" PetscInt_FMT "/%" PetscInt_FMT "\n", (double)r, c, qj+1, Nq));
        } else if ((r-minRad) < PETSC_SQRT_MACHINE_EPSILON*100. && nr < nrmax) {
          for (k=0;k<nr;k++) if (c == rCellIdx[k]) break;
          if (k==nr) {
            rCellIdx[nr++]= c;
            PetscCall(PetscInfo(sol, "\t\t\tPhase: adaptToleranceFEM Found another inner r=%e, cell %" PetscInt_FMT ", qp %" PetscInt_FMT "/%" PetscInt_FMT ", d=%e\n", (double)r, c, qj+1, Nq, (double)(r-minRad)));
          }
        }
        if (ctx->sphere) {
          if ((tt=r-ctx->e_radius) > 0) {
            PetscCall(PetscInfo(sol, "\t\t\t %" PetscInt_FMT " cell r=%g\n",c,(double)tt));
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
      PetscCall(DMLabelSetValue(adaptLabel, rCellIdx[k], DM_ADAPT_REFINE));
    }
    if (ctx->sphere) {
      for (c = 0; c < eMaxIdx; c++) {
        PetscCall(DMLabelSetValue(adaptLabel, eCellIdx[c], DM_ADAPT_REFINE));
        PetscCall(PetscInfo(sol, "\t\tPhase:%s: refine sphere e cell %" PetscInt_FMT " r=%g\n","adaptToleranceFEM",eCellIdx[c],(double)eMinRad));
      }
      for (c = 0; c < iMaxIdx; c++) {
        PetscCall(DMLabelSetValue(adaptLabel, iCellIdx[c], DM_ADAPT_REFINE));
        PetscCall(PetscInfo(sol, "\t\tPhase:%s: refine sphere i cell %" PetscInt_FMT " r=%g\n","adaptToleranceFEM",iCellIdx[c],(double)iMinRad));
      }
    }
    PetscCall(PetscInfo(sol, "Phase:%s: Adaptive refine origin cells %" PetscInt_FMT ",%" PetscInt_FMT " r=%g\n","adaptToleranceFEM",rCellIdx[0],rCellIdx[1],(double)minRad));
  } else if (type==0 || type==1 || type==3) { /* refine along r=0 axis */
    PetscScalar  *coef = NULL;
    Vec          coords;
    PetscInt     csize,Nv,d,nz;
    DM           cdm;
    PetscSection cs;
    PetscCall(DMGetCoordinatesLocal(forest, &coords));
    PetscCall(DMGetCoordinateDM(forest, &cdm));
    PetscCall(DMGetLocalSection(cdm, &cs));
    for (c = cStart; c < cEnd; c++) {
      PetscInt doit = 0, outside = 0;
      PetscCall(DMPlexVecGetClosure(cdm, cs, coords, c, &csize, &coef));
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
      PetscCall(DMPlexVecRestoreClosure(cdm, cs, coords, c, &csize, &coef));
      if (doit || (outside<Nv && nz)) {
        PetscCall(DMLabelSetValue(adaptLabel, c, DM_ADAPT_REFINE));
      }
    }
    PetscCall(PetscInfo(sol, "Phase:%s: RE refinement\n","adaptToleranceFEM"));
  }
  PetscCall(DMDestroy(&plex));
  PetscCall(DMAdaptLabel(forest, adaptLabel, &adaptedDM));
  PetscCall(DMLabelDestroy(&adaptLabel));
  *newForest = adaptedDM;
  if (adaptedDM) {
    if (isForest) {
      PetscCall(DMForestSetAdaptivityForest(adaptedDM,NULL)); // ????
    } else exit(33); // ???????
    PetscCall(DMConvert(adaptedDM, DMPLEX, &plex));
    PetscCall(DMPlexGetHeightStratum(plex,0,&cStart,&cEnd));
    PetscCall(PetscInfo(sol, "\tPhase: adaptToleranceFEM: %" PetscInt_FMT " cells, %" PetscInt_FMT " total quadrature points\n",cEnd-cStart,Nq*(cEnd-cStart)));
    PetscCall(DMDestroy(&plex));
  } else *newForest = NULL;
  PetscFunctionReturn(0);
}

// forest goes in (ctx->plex[grid]), plex comes out
static PetscErrorCode adapt(PetscInt grid, LandauCtx *ctx, Vec *uu)
{
  PetscInt        adaptIter;

  PetscFunctionBegin;
  PetscInt  type, limits[5] = {(grid==0) ? ctx->numRERefine : 0, (grid==0) ? ctx->nZRefine1 : 0, ctx->numAMRRefine[grid], (grid==0) ? ctx->nZRefine2 : 0,ctx->postAMRRefine[grid]};
  for (type=0;type<5;type++) {
    for (adaptIter = 0; adaptIter<limits[type];adaptIter++) {
      DM  newForest = NULL;
      PetscCall(adaptToleranceFEM(ctx->fe[0], *uu, type, grid, ctx, &newForest));
      if (newForest)  {
        PetscCall(DMDestroy(&ctx->plex[grid]));
        PetscCall(VecDestroy(uu));
        PetscCall(DMCreateGlobalVector(newForest,uu));
        PetscCall(PetscObjectSetName((PetscObject) *uu, "uAMR"));
        PetscCall(LandauSetInitialCondition(newForest, *uu, grid, 0, 1, ctx));
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
  PetscBool         flg, sph_flg;
  PetscInt          ii,nt,nm,nc,num_species_grid[LANDAU_MAX_GRIDS];
  PetscReal         v0_grid[LANDAU_MAX_GRIDS];
  DM                dummy;

  PetscFunctionBegin;
  PetscCall(DMCreate(ctx->comm,&dummy));
  /* get options - initialize context */
  ctx->verbose = 1; // should be 0 for silent compliance
#if defined(PETSC_HAVE_THREADSAFETY)
  ctx->batch_sz = PetscNumOMPThreads;
#else
  ctx->batch_sz = 1;
#endif
  ctx->batch_view_idx = 0;
  ctx->interpolate    = PETSC_TRUE;
  ctx->gpu_assembly   = PETSC_TRUE;
  ctx->norm_state     = 0;
  ctx->electronShift  = 0;
  ctx->M              = NULL;
  ctx->J              = NULL;
  /* geometry and grids */
  ctx->sphere         = PETSC_FALSE;
  ctx->inflate        = PETSC_FALSE;
  ctx->use_p4est      = PETSC_FALSE;
  ctx->num_sections   = 3; /* 2, 3 or 4 */
  for (PetscInt grid=0;grid<LANDAU_MAX_GRIDS;grid++) {
    ctx->radius[grid]           = 5.; /* thermal radius (velocity) */
    ctx->numAMRRefine[grid]     = 5;
    ctx->postAMRRefine[grid]    = 0;
    ctx->species_offset[grid+1] = 1; // one species default
    num_species_grid[grid]      = 0;
    ctx->plex[grid] = NULL;     /* cache as expensive to Convert */
  }
  ctx->species_offset[0] = 0;
  ctx->re_radius         = 0.;
  ctx->vperp0_radius1    = 0;
  ctx->vperp0_radius2    = 0;
  ctx->nZRefine1         = 0;
  ctx->nZRefine2         = 0;
  ctx->numRERefine       = 0;
  num_species_grid[0]    = 1; // one species default
  /* species - [0] electrons, [1] one ion species eg, duetarium, [2] heavy impurity ion, ... */
  ctx->charges[0]        = -1;  /* electron charge (MKS) */
  ctx->masses[0]         = 1/1835.469965278441013; /* temporary value in proton mass */
  ctx->n[0]              = 1;
  ctx->v_0               = 1; /* thermal velocity, we could start with a scale != 1 */
  ctx->thermal_temps[0]  = 1;
  /* constants, etc. */
  ctx->epsilon0          = 8.8542e-12; /* permittivity of free space (MKS) F/m */
  ctx->k                 = 1.38064852e-23; /* Boltzmann constant (MKS) J/K */
  ctx->lnLam             = 10;         /* cross section ratio large - small angle collisions */
  ctx->n_0               = 1.e20;        /* typical plasma n, but could set it to 1 */
  ctx->Ez                = 0;
  for (PetscInt grid=0;grid<LANDAU_NUM_TIMERS;grid++) ctx->times[grid] = 0;
  ctx->use_matrix_mass   =  PETSC_FALSE;
  ctx->use_relativistic_corrections = PETSC_FALSE;
  ctx->use_energy_tensor_trick      = PETSC_FALSE; /* Use Eero's trick for energy conservation v --> grad(v^2/2) */
  ctx->SData_d.w         = NULL;
  ctx->SData_d.x         = NULL;
  ctx->SData_d.y         = NULL;
  ctx->SData_d.z         = NULL;
  ctx->SData_d.invJ      = NULL;
  ctx->jacobian_field_major_order     = PETSC_FALSE;
  ctx->SData_d.coo_elem_offsets       = NULL;
  ctx->SData_d.coo_elem_point_offsets = NULL;
  ctx->coo_assembly                   = PETSC_FALSE;
  ctx->SData_d.coo_elem_fullNb        = NULL;
  ctx->SData_d.coo_size               = 0;
  PetscOptionsBegin(ctx->comm, prefix, "Options for Fokker-Plank-Landau collision operator", "none");
  {
    char opstring[256];
#if defined(PETSC_HAVE_KOKKOS_KERNELS)
    ctx->deviceType = LANDAU_KOKKOS;
    PetscCall(PetscStrcpy(opstring,"kokkos"));
#elif defined(PETSC_HAVE_CUDA)
    ctx->deviceType = LANDAU_CUDA;
    PetscCall(PetscStrcpy(opstring,"cuda"));
#else
    ctx->deviceType = LANDAU_CPU;
    PetscCall(PetscStrcpy(opstring,"cpu"));
#endif
    PetscCall(PetscOptionsString("-dm_landau_device_type","Use kernels on 'cpu', 'cuda', or 'kokkos'","plexland.c",opstring,opstring,sizeof(opstring),NULL));
    PetscCall(PetscStrcmp("cpu",opstring,&flg));
    if (flg) {
      ctx->deviceType = LANDAU_CPU;
    } else {
      PetscCall(PetscStrcmp("cuda",opstring,&flg));
      if (flg) {
        ctx->deviceType = LANDAU_CUDA;
      } else {
        PetscCall(PetscStrcmp("kokkos",opstring,&flg));
        if (flg) ctx->deviceType = LANDAU_KOKKOS;
        else SETERRQ(ctx->comm,PETSC_ERR_ARG_WRONG,"-dm_landau_device_type %s",opstring);
      }
    }
  }
  PetscCall(PetscOptionsReal("-dm_landau_electron_shift","Shift in thermal velocity of electrons","none",ctx->electronShift,&ctx->electronShift, NULL));
  PetscCall(PetscOptionsInt("-dm_landau_verbose", "Level of verbosity output", "plexland.c", ctx->verbose, &ctx->verbose, NULL));
  PetscCall(PetscOptionsInt("-dm_landau_batch_size", "Number of 'vertices' to batch", "ex2.c", ctx->batch_sz, &ctx->batch_sz, NULL));
  PetscCheck(LANDAU_MAX_BATCH_SZ >= ctx->batch_sz,ctx->comm,PETSC_ERR_ARG_WRONG,"LANDAU_MAX_BATCH_SZ %" PetscInt_FMT " < ctx->batch_sz %" PetscInt_FMT,(PetscInt)LANDAU_MAX_BATCH_SZ,ctx->batch_sz);
  PetscCall(PetscOptionsInt("-dm_landau_batch_view_idx", "Index of batch for diagnostics like plotting", "ex2.c", ctx->batch_view_idx, &ctx->batch_view_idx, NULL));
  PetscCheck(ctx->batch_view_idx < ctx->batch_sz,ctx->comm,PETSC_ERR_ARG_WRONG,"-ctx->batch_view_idx %" PetscInt_FMT " > ctx->batch_sz %" PetscInt_FMT,ctx->batch_view_idx,ctx->batch_sz);
  PetscCall(PetscOptionsReal("-dm_landau_Ez","Initial parallel electric field in unites of Conner-Hastie critical field","plexland.c",ctx->Ez,&ctx->Ez, NULL));
  PetscCall(PetscOptionsReal("-dm_landau_n_0","Normalization constant for number density","plexland.c",ctx->n_0,&ctx->n_0, NULL));
  PetscCall(PetscOptionsReal("-dm_landau_ln_lambda","Cross section parameter","plexland.c",ctx->lnLam,&ctx->lnLam, NULL));
  PetscCall(PetscOptionsBool("-dm_landau_use_mataxpy_mass", "Use fast but slightly fragile MATAXPY to add mass term", "plexland.c", ctx->use_matrix_mass, &ctx->use_matrix_mass, NULL));
  PetscCall(PetscOptionsBool("-dm_landau_use_relativistic_corrections", "Use relativistic corrections", "plexland.c", ctx->use_relativistic_corrections, &ctx->use_relativistic_corrections, NULL));
  PetscCall(PetscOptionsBool("-dm_landau_use_energy_tensor_trick", "Use Eero's trick of using grad(v^2/2) instead of v as args to Landau tensor to conserve energy with relativistic corrections and Q1 elements", "plexland.c", ctx->use_energy_tensor_trick, &ctx->use_energy_tensor_trick, NULL));

  /* get num species with temperature, set defaults */
  for (ii=1;ii<LANDAU_MAX_SPECIES;ii++) {
    ctx->thermal_temps[ii] = 1;
    ctx->charges[ii]       = 1;
    ctx->masses[ii]        = 1;
    ctx->n[ii]             = 1;
  }
  nt = LANDAU_MAX_SPECIES;
  PetscCall(PetscOptionsRealArray("-dm_landau_thermal_temps", "Temperature of each species [e,i_0,i_1,...] in keV (must be set to set number of species)", "plexland.c", ctx->thermal_temps, &nt, &flg));
  if (flg) {
    PetscCall(PetscInfo(dummy, "num_species set to number of thermal temps provided (%" PetscInt_FMT ")\n",nt));
    ctx->num_species = nt;
  } else SETERRQ(ctx->comm,PETSC_ERR_ARG_WRONG,"-dm_landau_thermal_temps ,t1,t2,.. must be provided to set the number of species");
  for (ii=0;ii<ctx->num_species;ii++) ctx->thermal_temps[ii] *= 1.1604525e7; /* convert to Kelvin */
  nm = LANDAU_MAX_SPECIES-1;
  PetscCall(PetscOptionsRealArray("-dm_landau_ion_masses", "Mass of each species in units of proton mass [i_0=2,i_1=40...]", "plexland.c", &ctx->masses[1], &nm, &flg));
  if (flg && nm != ctx->num_species-1) {
    SETERRQ(ctx->comm,PETSC_ERR_ARG_WRONG,"num ion masses %" PetscInt_FMT " != num species %" PetscInt_FMT,nm,ctx->num_species-1);
  }
  nm = LANDAU_MAX_SPECIES;
  PetscCall(PetscOptionsRealArray("-dm_landau_n", "Number density of each species = n_s * n_0", "plexland.c", ctx->n, &nm, &flg));
  PetscCheck(!flg || nm == ctx->num_species,ctx->comm,PETSC_ERR_ARG_WRONG,"wrong num n: %" PetscInt_FMT " != num species %" PetscInt_FMT,nm,ctx->num_species);
  for (ii=0;ii<LANDAU_MAX_SPECIES;ii++) ctx->masses[ii] *= 1.6720e-27; /* scale by proton mass kg */
  ctx->masses[0] = 9.10938356e-31; /* electron mass kg (should be about right already) */
  ctx->m_0 = ctx->masses[0]; /* arbitrary reference mass, electrons */
  nc = LANDAU_MAX_SPECIES-1;
  PetscCall(PetscOptionsRealArray("-dm_landau_ion_charges", "Charge of each species in units of proton charge [i_0=2,i_1=18,...]", "plexland.c", &ctx->charges[1], &nc, &flg));
  if (flg) PetscCheck(nc == ctx->num_species-1,ctx->comm,PETSC_ERR_ARG_WRONG,"num charges %" PetscInt_FMT " != num species %" PetscInt_FMT,nc,ctx->num_species-1);
  for (ii=0;ii<LANDAU_MAX_SPECIES;ii++) ctx->charges[ii] *= 1.6022e-19; /* electron/proton charge (MKS) */
  /* geometry and grids */
  nt = LANDAU_MAX_GRIDS;
  PetscCall(PetscOptionsIntArray("-dm_landau_num_species_grid","Number of species on each grid: [ 1, ....] or [S, 0 ....] for single grid","plexland.c", num_species_grid, &nt, &flg));
  if (flg) {
    ctx->num_grids = nt;
    for (ii=nt=0;ii<ctx->num_grids;ii++) nt += num_species_grid[ii];
    PetscCheck(ctx->num_species == nt,ctx->comm,PETSC_ERR_ARG_WRONG,"-dm_landau_num_species_grid: sum %" PetscInt_FMT " != num_species = %" PetscInt_FMT ". %" PetscInt_FMT " grids (check that number of grids <= LANDAU_MAX_GRIDS = %d)",nt,ctx->num_species,ctx->num_grids,LANDAU_MAX_GRIDS);
  } else {
    ctx->num_grids = 1; // go back to a single grid run
    num_species_grid[0] = ctx->num_species;
  }
  for (ctx->species_offset[0] = ii = 0; ii < ctx->num_grids ; ii++) ctx->species_offset[ii+1] = ctx->species_offset[ii] + num_species_grid[ii];
  PetscCheck(ctx->species_offset[ctx->num_grids] == ctx->num_species,ctx->comm,PETSC_ERR_ARG_WRONG,"ctx->species_offset[ctx->num_grids] %" PetscInt_FMT " != ctx->num_species = %" PetscInt_FMT " ???????????",ctx->species_offset[ctx->num_grids],ctx->num_species);
  for (PetscInt grid = 0; grid < ctx->num_grids ; grid++) {
    int iii = ctx->species_offset[grid]; // normalize with first (arbitrary) species on grid
    v0_grid[grid] = PetscSqrtReal(ctx->k*ctx->thermal_temps[iii]/ctx->masses[iii]); /* arbitrary units for non-dimensionalization: mean velocity in 1D of first species on grid */
  }
  ii = 0;
  PetscCall(PetscOptionsInt("-dm_landau_v0_grid", "Index of grid to use for setting v_0 (electrons are default). Not recommended to change", "plexland.c", ii, &ii, NULL));
  ctx->v_0 = v0_grid[ii]; /* arbitrary units for non dimensionalization: global mean velocity in 1D of electrons */
  ctx->t_0 = 8*PETSC_PI*PetscSqr(ctx->epsilon0*ctx->m_0/PetscSqr(ctx->charges[0]))/ctx->lnLam/ctx->n_0*PetscPowReal(ctx->v_0,3); /* note, this t_0 makes nu[0,0]=1 */
  /* domain */
  nt = LANDAU_MAX_GRIDS;
  PetscCall(PetscOptionsRealArray("-dm_landau_domain_radius","Phase space size in units of thermal velocity of grid","plexland.c",ctx->radius,&nt, &flg));
  if (flg) PetscCheck(nt >= ctx->num_grids,ctx->comm,PETSC_ERR_ARG_WRONG,"-dm_landau_domain_radius: given %" PetscInt_FMT " radius != number grids %" PetscInt_FMT,nt,ctx->num_grids);
  for (PetscInt grid = 0; grid < ctx->num_grids ; grid++) {
    if (flg && ctx->radius[grid] <= 0) { /* negative is ratio of c */
      if (ctx->radius[grid] == 0) ctx->radius[grid] = 0.75;
      else ctx->radius[grid] = -ctx->radius[grid];
      ctx->radius[grid] = ctx->radius[grid]*SPEED_OF_LIGHT/ctx->v_0; // use any species on grid to normalize (v_0 same for all on grid)
      PetscCall(PetscInfo(dummy, "Change domain radius to %g for grid %" PetscInt_FMT "\n",(double)ctx->radius[grid],grid));
    }
    ctx->radius[grid] *= v0_grid[grid]/ctx->v_0; // scale domain by thermal radius relative to v_0
  }
  /* amr parametres */
  nt = LANDAU_MAX_GRIDS;
  PetscCall(PetscOptionsIntArray("-dm_landau_amr_levels_max", "Number of AMR levels of refinement around origin, after (RE) refinements along z", "plexland.c", ctx->numAMRRefine, &nt, &flg));
  PetscCheck(!flg || nt >= ctx->num_grids,ctx->comm,PETSC_ERR_ARG_WRONG,"-dm_landau_amr_levels_max: given %" PetscInt_FMT " != number grids %" PetscInt_FMT,nt,ctx->num_grids);
  nt = LANDAU_MAX_GRIDS;
  PetscCall(PetscOptionsIntArray("-dm_landau_amr_post_refine", "Number of levels to uniformly refine after AMR", "plexland.c", ctx->postAMRRefine, &nt, &flg));
  for (ii=1;ii<ctx->num_grids;ii++)  ctx->postAMRRefine[ii] = ctx->postAMRRefine[0]; // all grids the same now
  PetscCall(PetscOptionsInt("-dm_landau_amr_re_levels", "Number of levels to refine along v_perp=0, z>0", "plexland.c", ctx->numRERefine, &ctx->numRERefine, &flg));
  PetscCall(PetscOptionsInt("-dm_landau_amr_z_refine1",  "Number of levels to refine along v_perp=0", "plexland.c", ctx->nZRefine1, &ctx->nZRefine1, &flg));
  PetscCall(PetscOptionsInt("-dm_landau_amr_z_refine2",  "Number of levels to refine along v_perp=0", "plexland.c", ctx->nZRefine2, &ctx->nZRefine2, &flg));
  PetscCall(PetscOptionsReal("-dm_landau_re_radius","velocity range to refine on positive (z>0) r=0 axis for runaways","plexland.c",ctx->re_radius,&ctx->re_radius, &flg));
  PetscCall(PetscOptionsReal("-dm_landau_z_radius1","velocity range to refine r=0 axis (for electrons)","plexland.c",ctx->vperp0_radius1,&ctx->vperp0_radius1, &flg));
  PetscCall(PetscOptionsReal("-dm_landau_z_radius2","velocity range to refine r=0 axis (for ions) after origin AMR","plexland.c",ctx->vperp0_radius2, &ctx->vperp0_radius2, &flg));
  /* spherical domain (not used) */
  PetscCall(PetscOptionsInt("-dm_landau_num_sections", "Number of tangential section in (2D) grid, 2, 3, of 4", "plexland.c", ctx->num_sections, &ctx->num_sections, NULL));
  PetscCall(PetscOptionsBool("-dm_landau_sphere", "use sphere/semi-circle domain instead of rectangle", "plexland.c", ctx->sphere, &ctx->sphere, &sph_flg));
  PetscCall(PetscOptionsBool("-dm_landau_inflate", "With sphere, inflate for curved edges", "plexland.c", ctx->inflate, &ctx->inflate, &flg));
  PetscCall(PetscOptionsReal("-dm_landau_e_radius","Electron thermal velocity, used for circular meshes","plexland.c",ctx->e_radius, &ctx->e_radius, &flg));
  if (flg && !sph_flg) ctx->sphere = PETSC_TRUE; /* you gave me an e radius but did not set sphere, user error really */
  if (!flg) {
    ctx->e_radius = 1.5*PetscSqrtReal(8*ctx->k*ctx->thermal_temps[0]/ctx->masses[0]/PETSC_PI)/ctx->v_0;
  }
  nt = LANDAU_MAX_GRIDS;
  PetscCall(PetscOptionsRealArray("-dm_landau_i_radius","Ion thermal velocity, used for circular meshes","plexland.c",ctx->i_radius, &nt, &flg));
  if (flg && !sph_flg) ctx->sphere = PETSC_TRUE;
  if (!flg) {
    ctx->i_radius[0] = 1.5*PetscSqrtReal(8*ctx->k*ctx->thermal_temps[1]/ctx->masses[1]/PETSC_PI)/ctx->v_0; // need to correct for ion grid domain
  }
  if (flg) PetscCheck(ctx->num_grids == nt,ctx->comm,PETSC_ERR_ARG_WRONG,"-dm_landau_i_radius: %" PetscInt_FMT " != num_species = %" PetscInt_FMT,nt,ctx->num_grids);
  if (ctx->sphere) PetscCheck(ctx->e_radius > ctx->i_radius[0],ctx->comm,PETSC_ERR_ARG_WRONG,"bad radii: %g < %g < %g",(double)ctx->i_radius[0],(double)ctx->e_radius,(double)ctx->radius[0]);
  /* processing options */
  PetscCall(PetscOptionsBool("-dm_landau_gpu_assembly", "Assemble Jacobian on GPU", "plexland.c", ctx->gpu_assembly, &ctx->gpu_assembly, NULL));
  if (ctx->deviceType == LANDAU_CPU || ctx->deviceType == LANDAU_KOKKOS) { // make Kokkos
    PetscCall(PetscOptionsBool("-dm_landau_coo_assembly", "Assemble Jacobian with Kokkos on 'device'", "plexland.c", ctx->coo_assembly, &ctx->coo_assembly, NULL));
    if (ctx->coo_assembly) PetscCheck(ctx->gpu_assembly,ctx->comm,PETSC_ERR_ARG_WRONG,"COO assembly requires 'gpu assembly' even if Kokkos 'CPU' back-end %d",ctx->coo_assembly);
  }
  PetscCall(PetscOptionsBool("-dm_landau_jacobian_field_major_order", "Reorder Jacobian for GPU assembly with field major, or block diagonal, ordering (DEPRECATED)", "plexland.c", ctx->jacobian_field_major_order, &ctx->jacobian_field_major_order, NULL));
  if (ctx->jacobian_field_major_order) PetscCheck(ctx->gpu_assembly,ctx->comm,PETSC_ERR_ARG_WRONG,"-dm_landau_jacobian_field_major_order requires -dm_landau_gpu_assembly");
  PetscCheck(!ctx->jacobian_field_major_order,ctx->comm,PETSC_ERR_ARG_WRONG,"-dm_landau_jacobian_field_major_order DEPRECATED");
  PetscOptionsEnd();

  for (ii=ctx->num_species;ii<LANDAU_MAX_SPECIES;ii++) ctx->masses[ii] = ctx->thermal_temps[ii]  = ctx->charges[ii] = 0;
  if (ctx->verbose > 0) {
    PetscCall(PetscPrintf(ctx->comm, "masses:        e=%10.3e; ions in proton mass units:   %10.3e %10.3e ...\n",(double)ctx->masses[0],(double)(ctx->masses[1]/1.6720e-27),(double)(ctx->num_species>2 ? ctx->masses[2]/1.6720e-27 : 0)));
    PetscCall(PetscPrintf(ctx->comm, "charges:       e=%10.3e; charges in elementary units: %10.3e %10.3e\n", (double)ctx->charges[0],(double)(-ctx->charges[1]/ctx->charges[0]),(double)(ctx->num_species>2 ? -ctx->charges[2]/ctx->charges[0] : 0)));
    PetscCall(PetscPrintf(ctx->comm, "n:             e: %10.3e                           i: %10.3e %10.3e\n", (double)ctx->n[0],(double)ctx->n[1],(double)(ctx->num_species>2 ? ctx->n[2] : 0)));
    PetscCall(PetscPrintf(ctx->comm, "thermal T (K): e=%10.3e i=%10.3e %10.3e. v_0=%10.3e (%10.3ec) n_0=%10.3e t_0=%10.3e, %s, %s, %" PetscInt_FMT " batched\n", (double)ctx->thermal_temps[0], (double)ctx->thermal_temps[1], (double)((ctx->num_species>2) ? ctx->thermal_temps[2] : 0), (double)ctx->v_0, (double)(ctx->v_0/SPEED_OF_LIGHT), (double)ctx->n_0, (double)ctx->t_0, ctx->use_relativistic_corrections ? "relativistic" : "classical", ctx->use_energy_tensor_trick ? "Use trick" : "Intuitive",ctx->batch_sz));
    PetscCall(PetscPrintf(ctx->comm, "Domain radius (AMR levels) grid %d: %10.3e (%" PetscInt_FMT ") ",0,(double)ctx->radius[0],ctx->numAMRRefine[0]));
    for (ii=1;ii<ctx->num_grids;ii++) PetscCall(PetscPrintf(ctx->comm, ", %" PetscInt_FMT ": %10.3e (%" PetscInt_FMT ") ",ii,(double)ctx->radius[ii],ctx->numAMRRefine[ii]));
    PetscCall(PetscPrintf(ctx->comm,"\n"));
    if (ctx->jacobian_field_major_order) {
      PetscCall(PetscPrintf(ctx->comm,"Using field major order for GPU Jacobian\n"));
    } else {
      PetscCall(PetscPrintf(ctx->comm,"Using default Plex order for all matrices\n"));
    }
  }
  PetscCall(DMDestroy(&dummy));
  {
    PetscMPIInt    rank;
    PetscCallMPI(MPI_Comm_rank(ctx->comm, &rank));
    ctx->stage = 0;
    PetscCall(PetscLogEventRegister("Landau Create", DM_CLASSID, &ctx->events[13])); /* 13 */
    PetscCall(PetscLogEventRegister(" GPU ass. setup", DM_CLASSID, &ctx->events[2])); /* 2 */
    PetscCall(PetscLogEventRegister(" Build matrix", DM_CLASSID, &ctx->events[12])); /* 12 */
    PetscCall(PetscLogEventRegister(" Assembly maps", DM_CLASSID, &ctx->events[15])); /* 15 */
    PetscCall(PetscLogEventRegister("Landau Mass mat", DM_CLASSID, &ctx->events[14])); /* 14 */
    PetscCall(PetscLogEventRegister("Landau Operator", DM_CLASSID, &ctx->events[11])); /* 11 */
    PetscCall(PetscLogEventRegister("Landau Jacobian", DM_CLASSID, &ctx->events[0])); /* 0 */
    PetscCall(PetscLogEventRegister("Landau Mass", DM_CLASSID, &ctx->events[9])); /* 9 */
    PetscCall(PetscLogEventRegister(" Preamble", DM_CLASSID, &ctx->events[10])); /* 10 */
    PetscCall(PetscLogEventRegister(" static IP Data", DM_CLASSID, &ctx->events[7])); /* 7 */
    PetscCall(PetscLogEventRegister(" dynamic IP-Jac", DM_CLASSID, &ctx->events[1])); /* 1 */
    PetscCall(PetscLogEventRegister(" Kernel-init", DM_CLASSID, &ctx->events[3])); /* 3 */
    PetscCall(PetscLogEventRegister(" Jac-f-df (GPU)", DM_CLASSID, &ctx->events[8])); /* 8 */
    PetscCall(PetscLogEventRegister(" J Kernel (GPU)", DM_CLASSID, &ctx->events[4])); /* 4 */
    PetscCall(PetscLogEventRegister(" M Kernel (GPU)", DM_CLASSID, &ctx->events[16])); /* 16 */
    PetscCall(PetscLogEventRegister(" Copy to CPU", DM_CLASSID, &ctx->events[5])); /* 5 */
    PetscCall(PetscLogEventRegister(" CPU assemble", DM_CLASSID, &ctx->events[6])); /* 6 */

    if (rank) { /* turn off output stuff for duplicate runs - do we need to add the prefix to all this? */
      PetscCall(PetscOptionsClearValue(NULL,"-snes_converged_reason"));
      PetscCall(PetscOptionsClearValue(NULL,"-ksp_converged_reason"));
      PetscCall(PetscOptionsClearValue(NULL,"-snes_monitor"));
      PetscCall(PetscOptionsClearValue(NULL,"-ksp_monitor"));
      PetscCall(PetscOptionsClearValue(NULL,"-ts_monitor"));
      PetscCall(PetscOptionsClearValue(NULL,"-ts_view"));
      PetscCall(PetscOptionsClearValue(NULL,"-ts_adapt_monitor"));
      PetscCall(PetscOptionsClearValue(NULL,"-dm_landau_amr_dm_view"));
      PetscCall(PetscOptionsClearValue(NULL,"-dm_landau_amr_vec_view"));
      PetscCall(PetscOptionsClearValue(NULL,"-dm_landau_mass_dm_view"));
      PetscCall(PetscOptionsClearValue(NULL,"-dm_landau_mass_view"));
      PetscCall(PetscOptionsClearValue(NULL,"-dm_landau_jacobian_view"));
      PetscCall(PetscOptionsClearValue(NULL,"-dm_landau_mat_view"));
      PetscCall(PetscOptionsClearValue(NULL,"-pc_bjkokkos_ksp_converged_reason"));
      PetscCall(PetscOptionsClearValue(NULL,"-pc_bjkokkos_ksp_monitor"));
      PetscCall(PetscOptionsClearValue(NULL,"-"));
      PetscCall(PetscOptionsClearValue(NULL,"-info"));
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateStaticGPUData(PetscInt dim, IS grid_batch_is_inv[], LandauCtx *ctx)
{
  PetscSection      section[LANDAU_MAX_GRIDS],globsection[LANDAU_MAX_GRIDS];
  PetscQuadrature   quad;
  const PetscReal   *quadWeights;
  PetscInt          numCells[LANDAU_MAX_GRIDS],Nq,Nf[LANDAU_MAX_GRIDS], ncellsTot=0, MAP_BF_SIZE = 64*LANDAU_DIM*LANDAU_DIM*LANDAU_MAX_Q_FACE*LANDAU_MAX_SPECIES;
  PetscTabulation   *Tf;
  PetscDS           prob;

  PetscFunctionBegin;
  PetscCall(DMGetDS(ctx->plex[0], &prob)); // same DS for all grids
  PetscCall(PetscDSGetTabulation(prob, &Tf)); // Bf, &Df same for all grids
  /* DS, Tab and quad is same on all grids */
  PetscCheck(ctx->plex[0],ctx->comm,PETSC_ERR_ARG_WRONG,"Plex not created");
  PetscCall(PetscFEGetQuadrature(ctx->fe[0], &quad));
  PetscCall(PetscQuadratureGetData(quad, NULL, NULL, &Nq, NULL,  &quadWeights));
  PetscCheck(Nq <= LANDAU_MAX_NQ,ctx->comm,PETSC_ERR_ARG_WRONG,"Order too high. Nq = %" PetscInt_FMT " > LANDAU_MAX_NQ (%d)",Nq,LANDAU_MAX_NQ);
  /* setup each grid */
  for (PetscInt grid=0;grid<ctx->num_grids;grid++) {
    PetscInt cStart, cEnd;
    PetscCheck(ctx->plex[grid] != NULL,ctx->comm,PETSC_ERR_ARG_WRONG,"Plex not created");
    PetscCall(DMPlexGetHeightStratum(ctx->plex[grid], 0, &cStart, &cEnd));
    numCells[grid] = cEnd - cStart; // grids can have different topology
    PetscCall(DMGetLocalSection(ctx->plex[grid], &section[grid]));
    PetscCall(DMGetGlobalSection(ctx->plex[grid], &globsection[grid]));
    PetscCall(PetscSectionGetNumFields(section[grid], &Nf[grid]));
    ncellsTot += numCells[grid];
  }
  /* create GPU assembly data */
  if (ctx->gpu_assembly) { /* we need GPU object with GPU assembly */
    PetscContainer          container;
    PetscScalar             elemMatrix[LANDAU_MAX_NQ*LANDAU_MAX_NQ*LANDAU_MAX_SPECIES*LANDAU_MAX_SPECIES], *elMat;
    pointInterpolationP4est (*pointMaps)[LANDAU_MAX_Q_FACE];
    P4estVertexMaps         *maps;
    const PetscInt          *plex_batch=NULL,Nb=Nq; // tensor elements;
    LandauIdx               *coo_elem_offsets=NULL, *coo_elem_fullNb=NULL, (*coo_elem_point_offsets)[LANDAU_MAX_NQ+1] = NULL;
    /* create GPU asssembly data */
    PetscCall(PetscInfo(ctx->plex[0], "Make GPU maps %d\n",1));
    PetscCall(PetscLogEventBegin(ctx->events[2],0,0,0,0));
    PetscCall(PetscMalloc(sizeof(*maps)*ctx->num_grids, &maps));
    PetscCall(PetscMalloc(sizeof(*pointMaps)*MAP_BF_SIZE, &pointMaps));

    if (ctx->coo_assembly) { // setup COO assembly -- put COO metadata directly in ctx->SData_d
      PetscCall(PetscMalloc3(ncellsTot+1,&coo_elem_offsets,ncellsTot,&coo_elem_fullNb,ncellsTot, &coo_elem_point_offsets)); // array of integer pointers
      coo_elem_offsets[0] = 0; // finish later
      PetscCall(PetscInfo(ctx->plex[0], "COO initialization, %" PetscInt_FMT " cells\n",ncellsTot));
      ctx->SData_d.coo_n_cellsTot         = ncellsTot;
      ctx->SData_d.coo_elem_offsets       = (void*)coo_elem_offsets;
      ctx->SData_d.coo_elem_fullNb        = (void*)coo_elem_fullNb;
      ctx->SData_d.coo_elem_point_offsets = (void*)coo_elem_point_offsets;
    } else {
      ctx->SData_d.coo_elem_offsets       = ctx->SData_d.coo_elem_fullNb = NULL;
      ctx->SData_d.coo_elem_point_offsets = NULL;
      ctx->SData_d.coo_n_cellsTot         = 0;
    }

    ctx->SData_d.coo_max_fullnb = 0;
    for (PetscInt grid=0,glb_elem_idx=0;grid<ctx->num_grids;grid++) {
      PetscInt cStart, cEnd, Nfloc = Nf[grid], totDim = Nfloc*Nq;
      if (grid_batch_is_inv[grid]) {
        PetscCall(ISGetIndices(grid_batch_is_inv[grid], &plex_batch));
      }
      PetscCall(DMPlexGetHeightStratum(ctx->plex[grid], 0, &cStart, &cEnd));
      // make maps
      maps[grid].d_self       = NULL;
      maps[grid].num_elements = numCells[grid];
      maps[grid].num_face = (PetscInt)(pow(Nq,1./((double)dim))+.001); // Q
      maps[grid].num_face = (PetscInt)(pow(maps[grid].num_face,(double)(dim-1))+.001); // Q^2
      maps[grid].num_reduced  = 0;
      maps[grid].deviceType   = ctx->deviceType;
      maps[grid].numgrids     = ctx->num_grids;
      // count reduced and get
      PetscCall(PetscMalloc(maps[grid].num_elements * sizeof(*maps[grid].gIdx), &maps[grid].gIdx));
      for (int ej = cStart, eidx = 0 ; ej < cEnd; ++ej, ++eidx, glb_elem_idx++) {
        if (coo_elem_offsets) coo_elem_offsets[glb_elem_idx+1] = coo_elem_offsets[glb_elem_idx]; // start with last one, then add
        for (int fieldA=0;fieldA<Nf[grid];fieldA++) {
          int fullNb = 0;
          for (int q = 0; q < Nb; ++q) {
            PetscInt    numindices,*indices;
            PetscScalar *valuesOrig = elMat = elemMatrix;
            PetscCall(PetscArrayzero(elMat, totDim*totDim));
            elMat[ (fieldA*Nb + q)*totDim + fieldA*Nb + q] = 1;
            PetscCall(DMPlexGetClosureIndices(ctx->plex[grid], section[grid], globsection[grid], ej, PETSC_TRUE, &numindices, &indices, NULL, (PetscScalar **) &elMat));
            for (PetscInt f = 0 ; f < numindices ; ++f) { // look for a non-zero on the diagonal
              if (PetscAbs(PetscRealPart(elMat[f*numindices + f])) > PETSC_MACHINE_EPSILON) {
                // found it
                if (PetscAbs(PetscRealPart(elMat[f*numindices + f] - 1.)) < PETSC_MACHINE_EPSILON) { // normal vertex 1.0
                  if (plex_batch) {
                    maps[grid].gIdx[eidx][fieldA][q] = (LandauIdx) plex_batch[indices[f]];
                  } else {
                    maps[grid].gIdx[eidx][fieldA][q] = (LandauIdx)indices[f];
                  }
                  fullNb++;
                } else { //found a constraint
                  int       jj      = 0;
                  PetscReal sum     = 0;
                  const PetscInt ff = f;
                  maps[grid].gIdx[eidx][fieldA][q] = -maps[grid].num_reduced - 1; // store (-)index: id = -(idx+1): idx = -id - 1

                  do {  // constraints are continuous in Plex - exploit that here
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
                      fullNb++;
                    }
                  } while (++jj < maps[grid].num_face && ++f < numindices); // jj is incremented if we hit the end
                  while (jj < maps[grid].num_face) {
                    pointMaps[maps[grid].num_reduced][jj].scale = 0;
                    pointMaps[maps[grid].num_reduced][jj].gid = -1;
                    jj++;
                  }
                  if (PetscAbs(sum-1.0) > 10*PETSC_MACHINE_EPSILON) { // debug
                    int       d,f;
                    PetscReal tmp = 0;
                    PetscCall(PetscPrintf(PETSC_COMM_SELF,"\t\t%d.%d.%d) ERROR total I = %22.16e (LANDAU_MAX_Q_FACE=%d, #face=%d)\n",eidx,q,fieldA,(double)sum,LANDAU_MAX_Q_FACE,maps[grid].num_face));
                    for (d = 0, tmp = 0; d < numindices; ++d) {
                      if (tmp!=0 && PetscAbs(tmp-1.0) > 10*PETSC_MACHINE_EPSILON) PetscCall(PetscPrintf(PETSC_COMM_WORLD,"%3d) %3" PetscInt_FMT ": ",d,indices[d]));
                      for (f = 0; f < numindices; ++f) {
                        tmp += PetscRealPart(elMat[d*numindices + f]);
                      }
                      if (tmp!=0) PetscCall(PetscPrintf(ctx->comm," | %22.16e\n",(double)tmp));
                    }
                  }
                  maps[grid].num_reduced++;
                  PetscCheck(maps[grid].num_reduced<MAP_BF_SIZE,PETSC_COMM_SELF, PETSC_ERR_PLIB, "maps[grid].num_reduced %d > %" PetscInt_FMT,maps[grid].num_reduced,MAP_BF_SIZE);
                }
                break;
              }
            }
            // cleanup
            PetscCall(DMPlexRestoreClosureIndices(ctx->plex[grid], section[grid], globsection[grid], ej, PETSC_TRUE, &numindices, &indices, NULL, (PetscScalar **) &elMat));
            if (elMat != valuesOrig) PetscCall(DMRestoreWorkArray(ctx->plex[grid], numindices*numindices, MPIU_SCALAR, &elMat));
          }
          if (ctx->coo_assembly) { // setup COO assembly
            coo_elem_offsets[glb_elem_idx+1] += fullNb*fullNb; // one species block, adds a block for each species, on this element in this grid
            if (fieldA==0) { // cache full Nb for this element, on this grid per species
              coo_elem_fullNb[glb_elem_idx] = fullNb;
              if (fullNb>ctx->SData_d.coo_max_fullnb) ctx->SData_d.coo_max_fullnb = fullNb;
            } else PetscCheck(coo_elem_fullNb[glb_elem_idx] == fullNb,PETSC_COMM_SELF, PETSC_ERR_PLIB, "full element size change with species %d %d",coo_elem_fullNb[glb_elem_idx],fullNb);
          }
        } // field
      } // cell
      // allocate and copy point data maps[grid].gIdx[eidx][field][q]
      PetscCall(PetscMalloc(maps[grid].num_reduced * sizeof(*maps[grid].c_maps), &maps[grid].c_maps));
      for (int ej = 0; ej < maps[grid].num_reduced; ++ej) {
        for (int q = 0; q < maps[grid].num_face; ++q) {
          maps[grid].c_maps[ej][q].scale = pointMaps[ej][q].scale;
          maps[grid].c_maps[ej][q].gid   = pointMaps[ej][q].gid;
        }
      }
#if defined(PETSC_HAVE_KOKKOS_KERNELS)
      if (ctx->deviceType == LANDAU_KOKKOS) {
        PetscCall(LandauKokkosCreateMatMaps(maps, pointMaps, Nf, Nq, grid)); // imples Kokkos does
      } // else could be CUDA
#endif
#if defined(PETSC_HAVE_CUDA)
      if (ctx->deviceType == LANDAU_CUDA) {
        PetscCall(LandauCUDACreateMatMaps(maps, pointMaps, Nf, Nq, grid));
      }
#endif
      if (plex_batch) {
        PetscCall(ISRestoreIndices(grid_batch_is_inv[grid], &plex_batch));
        PetscCall(ISDestroy(&grid_batch_is_inv[grid])); // we are done with this
      }
    } /* grids */
    // finish COO
    if (ctx->coo_assembly) { // setup COO assembly
      PetscInt *oor, *ooc;
      ctx->SData_d.coo_size = coo_elem_offsets[ncellsTot]*ctx->batch_sz;
      PetscCall(PetscMalloc2(ctx->SData_d.coo_size,&oor,ctx->SData_d.coo_size,&ooc));
      for (int i=0;i<ctx->SData_d.coo_size;i++) oor[i] = ooc[i] = -1;
      // get
      for (int grid=0,glb_elem_idx=0;grid<ctx->num_grids;grid++) {
        for (int ej = 0 ; ej < numCells[grid] ; ++ej, glb_elem_idx++) {
          const int              fullNb = coo_elem_fullNb[glb_elem_idx];
          const LandauIdx *const Idxs = &maps[grid].gIdx[ej][0][0]; // just use field-0 maps, They should be the same but this is just for COO storage
          coo_elem_point_offsets[glb_elem_idx][0] = 0;
          for (int f=0, cnt2=0;f<Nb;f++) {
            int idx = Idxs[f];
            coo_elem_point_offsets[glb_elem_idx][f+1] = coo_elem_point_offsets[glb_elem_idx][f]; // start at last
            if (idx >= 0) {
              cnt2++;
              coo_elem_point_offsets[glb_elem_idx][f+1]++; // inc
            } else {
              idx = -idx - 1;
              for (int q = 0 ; q < maps[grid].num_face; q++) {
                if (maps[grid].c_maps[idx][q].gid < 0) break;
                cnt2++;
                coo_elem_point_offsets[glb_elem_idx][f+1]++; // inc
              }
            }
            PetscCheck(cnt2 <= fullNb,PETSC_COMM_SELF, PETSC_ERR_PLIB, "wrong count %d < %d",fullNb,cnt2);
          }
          PetscCheck(coo_elem_point_offsets[glb_elem_idx][Nb]==fullNb,PETSC_COMM_SELF, PETSC_ERR_PLIB, "coo_elem_point_offsets size %d != fullNb=%d",coo_elem_point_offsets[glb_elem_idx][Nb],fullNb);
        }
      }
      // set
      for (PetscInt b_id = 0 ; b_id < ctx->batch_sz ; b_id++) {
        for (int grid=0,glb_elem_idx=0;grid<ctx->num_grids;grid++) {
          const PetscInt moffset = LAND_MOFFSET(b_id,grid,ctx->batch_sz,ctx->num_grids,ctx->mat_offset);
          for (int ej = 0 ; ej < numCells[grid] ; ++ej, glb_elem_idx++) {
            const int  fullNb = coo_elem_fullNb[glb_elem_idx],fullNb2=fullNb*fullNb;
            // set (i,j)
            for (int fieldA=0;fieldA<Nf[grid];fieldA++) {
              const LandauIdx *const Idxs = &maps[grid].gIdx[ej][fieldA][0];
              int                    rows[LANDAU_MAX_Q_FACE],cols[LANDAU_MAX_Q_FACE];
              for (int f = 0; f < Nb; ++f) {
                const int nr =  coo_elem_point_offsets[glb_elem_idx][f+1] - coo_elem_point_offsets[glb_elem_idx][f];
                if (nr==1) rows[0] = Idxs[f];
                else {
                  const int idx = -Idxs[f] - 1;
                  for (int q = 0; q < nr; q++) {
                    rows[q] = maps[grid].c_maps[idx][q].gid;
                  }
                }
                for (int g = 0; g < Nb; ++g) {
                  const int nc =  coo_elem_point_offsets[glb_elem_idx][g+1] - coo_elem_point_offsets[glb_elem_idx][g];
                  if (nc==1) cols[0] = Idxs[g];
                  else {
                    const int idx = -Idxs[g] - 1;
                    for (int q = 0; q < nc; q++) {
                      cols[q] = maps[grid].c_maps[idx][q].gid;
                    }
                  }
                  const int idx0 = b_id*coo_elem_offsets[ncellsTot] + coo_elem_offsets[glb_elem_idx] + fieldA*fullNb2 + fullNb * coo_elem_point_offsets[glb_elem_idx][f] + nr * coo_elem_point_offsets[glb_elem_idx][g];
                  for (int q = 0, idx = idx0; q < nr; q++) {
                    for (int d = 0; d < nc; d++, idx++) {
                      oor[idx] = rows[q] + moffset;
                      ooc[idx] = cols[d] + moffset;
                    }
                  }
                }
              }
            }
          } // cell
        } // grid
      } // batch
      PetscCall(MatSetPreallocationCOO(ctx->J,ctx->SData_d.coo_size,oor,ooc));
      PetscCall(PetscFree2(oor,ooc));
    }
    PetscCall(PetscFree(pointMaps));
    PetscCall(PetscContainerCreate(PETSC_COMM_SELF, &container));
    PetscCall(PetscContainerSetPointer(container, (void *)maps));
    PetscCall(PetscContainerSetUserDestroy(container, LandauGPUMapsDestroy));
    PetscCall(PetscObjectCompose((PetscObject) ctx->J, "assembly_maps", (PetscObject) container));
    PetscCall(PetscContainerDestroy(&container));
    PetscCall(PetscLogEventEnd(ctx->events[2],0,0,0,0));
  } // end GPU assembly
  { /* create static point data, Jacobian called first, only one vertex copy */
    PetscReal      *invJe,*ww,*xx,*yy,*zz=NULL,*invJ_a;
    PetscInt       outer_ipidx, outer_ej,grid, nip_glb = 0;
    PetscFE        fe;
    const PetscInt Nb = Nq;
    PetscCall(PetscLogEventBegin(ctx->events[7],0,0,0,0));
    PetscCall(PetscInfo(ctx->plex[0], "Initialize static data\n"));
    for (PetscInt grid=0;grid<ctx->num_grids;grid++) nip_glb += Nq*numCells[grid];
    /* collect f data, first time is for Jacobian, but make mass now */
    if (ctx->verbose > 0) {
      PetscInt ncells = 0, N;
      PetscCall(MatGetSize(ctx->J,&N,NULL));
      for (PetscInt grid=0;grid<ctx->num_grids;grid++) ncells += numCells[grid];
      PetscCall(PetscPrintf(ctx->comm,"%d) %s %" PetscInt_FMT " IPs, %" PetscInt_FMT " cells total, Nb=%" PetscInt_FMT ", Nq=%" PetscInt_FMT ", dim=%" PetscInt_FMT ", Tab: Nb=%" PetscInt_FMT " Nf=%" PetscInt_FMT " Np=%" PetscInt_FMT " cdim=%" PetscInt_FMT " N=%" PetscInt_FMT "\n",0,"FormLandau",nip_glb,ncells, Nb, Nq, dim, Nb, ctx->num_species, Nb, dim, N));
    }
    PetscCall(PetscMalloc4(nip_glb,&ww,nip_glb,&xx,nip_glb,&yy,nip_glb*dim*dim,&invJ_a));
    if (dim==3) {
      PetscCall(PetscMalloc1(nip_glb,&zz));
    }
    if (ctx->use_energy_tensor_trick) {
      PetscCall(PetscFECreateDefault(PETSC_COMM_SELF, dim, 1, PETSC_FALSE, NULL, PETSC_DECIDE, &fe));
      PetscCall(PetscObjectSetName((PetscObject) fe, "energy"));
    }
    /* init each grids static data - no batch */
    for (grid=0, outer_ipidx=0, outer_ej=0 ; grid < ctx->num_grids ; grid++) { // OpenMP (once)
      Vec             v2_2 = NULL; // projected function: v^2/2 for non-relativistic, gamma... for relativistic
      PetscSection    e_section;
      DM              dmEnergy;
      PetscInt        cStart, cEnd, ej;

      PetscCall(DMPlexGetHeightStratum(ctx->plex[grid], 0, &cStart, &cEnd));
      // prep energy trick, get v^2 / 2 vector
      if (ctx->use_energy_tensor_trick) {
        PetscErrorCode (*energyf[1])(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar [], void *) = {ctx->use_relativistic_corrections ? gamma_m1_f : energy_f};
        Vec            glob_v2;
        PetscReal      *c2_0[1], data[1] = {PetscSqr(C_0(ctx->v_0))};

        PetscCall(DMClone(ctx->plex[grid], &dmEnergy));
        PetscCall(PetscObjectSetName((PetscObject) dmEnergy, "energy"));
        PetscCall(DMSetField(dmEnergy, 0, NULL, (PetscObject)fe));
        PetscCall(DMCreateDS(dmEnergy));
        PetscCall(DMGetSection(dmEnergy, &e_section));
        PetscCall(DMGetGlobalVector(dmEnergy,&glob_v2));
        PetscCall(PetscObjectSetName((PetscObject) glob_v2, "trick"));
        c2_0[0] = &data[0];
        PetscCall(DMProjectFunction(dmEnergy, 0., energyf, (void**)c2_0, INSERT_ALL_VALUES, glob_v2));
        PetscCall(DMGetLocalVector(dmEnergy, &v2_2));
        PetscCall(VecZeroEntries(v2_2)); /* zero BCs so don't set */
        PetscCall(DMGlobalToLocalBegin(dmEnergy, glob_v2, INSERT_VALUES, v2_2));
        PetscCall(DMGlobalToLocalEnd  (dmEnergy, glob_v2, INSERT_VALUES, v2_2));
        PetscCall(DMViewFromOptions(dmEnergy,NULL, "-energy_dm_view"));
        PetscCall(VecViewFromOptions(glob_v2,NULL, "-energy_vec_view"));
        PetscCall(DMRestoreGlobalVector(dmEnergy, &glob_v2));
      }
      /* append part of the IP data for each grid */
      for (ej = 0 ; ej < numCells[grid]; ++ej, ++outer_ej) {
        PetscScalar *coefs = NULL;
        PetscReal    vj[LANDAU_MAX_NQ*LANDAU_DIM],detJj[LANDAU_MAX_NQ], Jdummy[LANDAU_MAX_NQ*LANDAU_DIM*LANDAU_DIM], c0 = C_0(ctx->v_0), c02 = PetscSqr(c0);
        invJe = invJ_a + outer_ej*Nq*dim*dim;
        PetscCall(DMPlexComputeCellGeometryFEM(ctx->plex[grid], ej+cStart, quad, vj, Jdummy, invJe, detJj));
        if (ctx->use_energy_tensor_trick) {
          PetscCall(DMPlexVecGetClosure(dmEnergy, e_section, v2_2, ej+cStart, NULL, &coefs));
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
                PetscCall(PetscPrintf(ctx->comm,"Error: %12.5e %" PetscInt_FMT ".%" PetscInt_FMT ") dg2/c02 = %12.5e x= %12.5e %12.5e %12.5e\n",(double)PetscSqrtReal(xx[gidx]*xx[gidx] + yy[gidx]*yy[gidx] + zz[gidx]*zz[gidx]), ej, qj, dg2_c2, (double)xx[gidx], (double)yy[gidx], (double)zz[gidx]));
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
          PetscCall(DMPlexVecRestoreClosure(dmEnergy, e_section, v2_2, ej+cStart, NULL, &coefs));
        }
      } /* ej */
      if (ctx->use_energy_tensor_trick) {
        PetscCall(DMRestoreLocalVector(dmEnergy, &v2_2));
        PetscCall(DMDestroy(&dmEnergy));
      }
    } /* grid */
    if (ctx->use_energy_tensor_trick) {
      PetscCall(PetscFEDestroy(&fe));
    }
    /* cache static data */
    if (ctx->deviceType == LANDAU_CUDA || ctx->deviceType == LANDAU_KOKKOS) {
#if defined(PETSC_HAVE_CUDA) || defined(PETSC_HAVE_KOKKOS_KERNELS)
      PetscReal invMass[LANDAU_MAX_SPECIES],nu_alpha[LANDAU_MAX_SPECIES], nu_beta[LANDAU_MAX_SPECIES];
      for (PetscInt grid = 0; grid < ctx->num_grids ; grid++) {
        for (PetscInt ii=ctx->species_offset[grid];ii<ctx->species_offset[grid+1];ii++) {
          invMass[ii]  = ctx->m_0/ctx->masses[ii];
          nu_alpha[ii] = PetscSqr(ctx->charges[ii]/ctx->m_0)*ctx->m_0/ctx->masses[ii];
          nu_beta[ii]  = PetscSqr(ctx->charges[ii]/ctx->epsilon0)*ctx->lnLam / (8*PETSC_PI) * ctx->t_0*ctx->n_0/PetscPowReal(ctx->v_0,3);
        }
      }
      if (ctx->deviceType == LANDAU_CUDA) {
#if defined(PETSC_HAVE_CUDA)
        PetscCall(LandauCUDAStaticDataSet(ctx->plex[0], Nq, ctx->batch_sz, ctx->num_grids, numCells, ctx->species_offset, ctx->mat_offset,
                                        nu_alpha, nu_beta, invMass, invJ_a, xx, yy, zz, ww, &ctx->SData_d));
#else
        SETERRQ(ctx->comm,PETSC_ERR_ARG_WRONG,"-landau_device_type cuda not built");
#endif
      } else if (ctx->deviceType == LANDAU_KOKKOS) {
#if defined(PETSC_HAVE_KOKKOS_KERNELS)
        PetscCall(LandauKokkosStaticDataSet(ctx->plex[0], Nq, ctx->batch_sz, ctx->num_grids, numCells, ctx->species_offset, ctx->mat_offset,
                                          nu_alpha, nu_beta, invMass,invJ_a,xx,yy,zz,ww,&ctx->SData_d));
#else
        SETERRQ(ctx->comm,PETSC_ERR_ARG_WRONG,"-landau_device_type kokkos not built");
#endif
      }
#endif
      /* free */
      PetscCall(PetscFree4(ww,xx,yy,invJ_a));
      if (dim==3) PetscCall(PetscFree(zz));
    } else { /* CPU version, just copy in, only use part */
      ctx->SData_d.w = (void*)ww;
      ctx->SData_d.x = (void*)xx;
      ctx->SData_d.y = (void*)yy;
      ctx->SData_d.z = (void*)zz;
      ctx->SData_d.invJ = (void*)invJ_a;
    }
    PetscCall(PetscLogEventEnd(ctx->events[7],0,0,0,0));
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
  PetscFunctionBegin;
  PetscCall(PetscFree(nf));
  PetscFunctionReturn(0);
}

static PetscErrorCode LandauCreateMatrix(MPI_Comm comm, Vec X, IS grid_batch_is_inv[LANDAU_MAX_GRIDS], LandauCtx *ctx)
{
  PetscInt       *idxs=NULL;
  Mat            subM[LANDAU_MAX_GRIDS];

  PetscFunctionBegin;
  if (!ctx->gpu_assembly) { /* we need GPU object with GPU assembly */
    PetscFunctionReturn(0);
  }
  // get the RCM for this grid to separate out species into blocks -- create 'idxs' & 'ctx->batch_is'
  if (ctx->gpu_assembly && ctx->jacobian_field_major_order) {
    PetscCall(PetscMalloc1(ctx->mat_offset[ctx->num_grids]*ctx->batch_sz, &idxs));
  }
  for (PetscInt grid=0 ; grid < ctx->num_grids ; grid++) {
    const PetscInt *values, n = ctx->mat_offset[grid+1] - ctx->mat_offset[grid];
    Mat             gMat;
    DM              massDM;
    PetscDS         prob;
    Vec             tvec;
    // get "mass" matrix for reordering
    PetscCall(DMClone(ctx->plex[grid], &massDM));
    PetscCall(DMCopyFields(ctx->plex[grid], massDM));
    PetscCall(DMCreateDS(massDM));
    PetscCall(DMGetDS(massDM, &prob));
    for (int ix=0, ii=ctx->species_offset[grid];ii<ctx->species_offset[grid+1];ii++,ix++) {
      PetscCall(PetscDSSetJacobian(prob, ix, ix, g0_fake, NULL, NULL, NULL));
    }
    PetscCall(PetscOptionsInsertString(NULL,"-dm_preallocate_only"));
    PetscCall(DMSetFromOptions(massDM));
    PetscCall(DMCreateMatrix(massDM, &gMat));
    PetscCall(PetscOptionsInsertString(NULL,"-dm_preallocate_only false"));
    PetscCall(MatSetOption(gMat,MAT_STRUCTURALLY_SYMMETRIC, PETSC_TRUE));
    PetscCall(MatSetOption(gMat,MAT_IGNORE_ZERO_ENTRIES,PETSC_TRUE));
    PetscCall(DMCreateLocalVector(ctx->plex[grid],&tvec));
    PetscCall(DMPlexSNESComputeJacobianFEM(massDM, tvec, gMat, gMat, ctx));
    PetscCall(MatViewFromOptions(gMat, NULL, "-dm_landau_reorder_mat_view"));
    PetscCall(DMDestroy(&massDM));
    PetscCall(VecDestroy(&tvec));
    subM[grid] = gMat;
    if (ctx->gpu_assembly && ctx->jacobian_field_major_order) {
      MatOrderingType rtype = MATORDERINGRCM;
      IS              isrow,isicol;
      PetscCall(MatGetOrdering(gMat,rtype,&isrow,&isicol));
      PetscCall(ISInvertPermutation(isrow,PETSC_DECIDE,&grid_batch_is_inv[grid]));
      PetscCall(ISGetIndices(isrow, &values));
      for (PetscInt b_id=0 ; b_id < ctx->batch_sz ; b_id++) { // add batch size DMs for this species grid
#if !defined(LANDAU_SPECIES_MAJOR)
        PetscInt N = ctx->mat_offset[ctx->num_grids], n0 = ctx->mat_offset[grid] + b_id*N;
        for (int ii = 0; ii < n; ++ii) idxs[n0+ii] = values[ii] + n0;
#else
        PetscInt n0 = ctx->mat_offset[grid]*ctx->batch_sz + b_id*n;
        for (int ii = 0; ii < n; ++ii) idxs[n0+ii] = values[ii] + n0;
#endif
      }
      PetscCall(ISRestoreIndices(isrow, &values));
      PetscCall(ISDestroy(&isrow));
      PetscCall(ISDestroy(&isicol));
    }
  }
  if (ctx->gpu_assembly && ctx->jacobian_field_major_order) {
    PetscCall(ISCreateGeneral(comm,ctx->mat_offset[ctx->num_grids]*ctx->batch_sz,idxs,PETSC_OWN_POINTER,&ctx->batch_is));
  }
  // get a block matrix
  for (PetscInt grid=0 ; grid<ctx->num_grids ; grid++) {
    Mat               B = subM[grid];
    PetscInt          nloc, nzl, colbuf[1024], row;
    PetscCall(MatGetSize(B, &nloc, NULL));
    for (PetscInt b_id = 0 ; b_id < ctx->batch_sz ; b_id++) {
      const PetscInt    moffset = LAND_MOFFSET(b_id,grid,ctx->batch_sz,ctx->num_grids,ctx->mat_offset);
      const PetscInt    *cols;
      const PetscScalar *vals;
      for (int i=0 ; i<nloc ; i++) {
        PetscCall(MatGetRow(B,i,&nzl,&cols,&vals));
        PetscCheck(nzl<=1024,comm, PETSC_ERR_PLIB, "Row too big: %" PetscInt_FMT,nzl);
        for (int j=0; j<nzl; j++) colbuf[j] = cols[j] + moffset;
        row = i + moffset;
        PetscCall(MatSetValues(ctx->J,1,&row,nzl,colbuf,vals,INSERT_VALUES));
        PetscCall(MatRestoreRow(B,i,&nzl,&cols,&vals));
      }
    }
  }
  for (PetscInt grid=0 ; grid<ctx->num_grids ; grid++) {
    PetscCall(MatDestroy(&subM[grid]));
  }
  PetscCall(MatAssemblyBegin(ctx->J,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(ctx->J,MAT_FINAL_ASSEMBLY));

  // debug
  if (ctx->gpu_assembly && ctx->jacobian_field_major_order) {
    Mat mat_block_order;
    PetscCall(MatCreateSubMatrix(ctx->J,ctx->batch_is,ctx->batch_is,MAT_INITIAL_MATRIX,&mat_block_order)); // use MatPermute
    PetscCall(MatViewFromOptions(ctx->J, NULL, "-dm_landau_mat_view"));
    PetscCall(MatViewFromOptions(mat_block_order, NULL, "-dm_landau_mat_view"));
    PetscCall(MatDestroy(&mat_block_order));
    PetscCall(VecScatterCreate(X, ctx->batch_is, X, NULL, &ctx->plex_batch));
    PetscCall(VecDuplicate(X,&ctx->work_vec));
  }

  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexLandauCreateMassMatrix(DM pack, Mat *Amat);
/*@C
 DMPlexLandauCreateVelocitySpace - Create a DMPlex velocity space mesh

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
 .seealso: `DMPlexCreate()`, `DMPlexLandauDestroyVelocitySpace()`
 @*/
PetscErrorCode DMPlexLandauCreateVelocitySpace(MPI_Comm comm, PetscInt dim, const char prefix[], Vec *X, Mat *J, DM *pack)
{
  LandauCtx      *ctx;
  Vec            Xsub[LANDAU_MAX_GRIDS];
  IS             grid_batch_is_inv[LANDAU_MAX_GRIDS];

  PetscFunctionBegin;
  PetscCheck(dim == 2 || dim == 3,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Only 2D and 3D supported");
  PetscCheck(LANDAU_DIM == dim,PETSC_COMM_SELF, PETSC_ERR_PLIB, "dim %" PetscInt_FMT " != LANDAU_DIM %d",dim,LANDAU_DIM);
  PetscCall(PetscNew(&ctx));
  ctx->comm = comm; /* used for diagnostics and global errors */
  /* process options */
  PetscCall(ProcessOptions(ctx,prefix));
  if (dim==2) ctx->use_relativistic_corrections = PETSC_FALSE;
  /* Create Mesh */
  PetscCall(DMCompositeCreate(PETSC_COMM_SELF,pack));
  PetscCall(PetscLogEventBegin(ctx->events[13],0,0,0,0));
  PetscCall(PetscLogEventBegin(ctx->events[15],0,0,0,0));
  PetscCall(LandauDMCreateVMeshes(PETSC_COMM_SELF, dim, prefix, ctx, *pack)); // creates grids (Forest of AMR)
  for (PetscInt grid=0;grid<ctx->num_grids;grid++) {
    /* create FEM */
    PetscCall(SetupDS(ctx->plex[grid],dim,grid,ctx));
    /* set initial state */
    PetscCall(DMCreateGlobalVector(ctx->plex[grid],&Xsub[grid]));
    PetscCall(PetscObjectSetName((PetscObject) Xsub[grid], "u_orig"));
    /* initial static refinement, no solve */
    PetscCall(LandauSetInitialCondition(ctx->plex[grid], Xsub[grid], grid, 0, 1, ctx));
    /* forest refinement - forest goes in (if forest), plex comes out */
    if (ctx->use_p4est) {
      DM plex;
      PetscCall(adapt(grid,ctx,&Xsub[grid])); // forest goes in, plex comes out
      PetscCall(DMViewFromOptions(ctx->plex[grid],NULL,"-dm_landau_amr_dm_view")); // need to differentiate - todo
      PetscCall(VecViewFromOptions(Xsub[grid], NULL, "-dm_landau_amr_vec_view"));
      // convert to plex, all done with this level
      PetscCall(DMConvert(ctx->plex[grid], DMPLEX, &plex));
      PetscCall(DMDestroy(&ctx->plex[grid]));
      ctx->plex[grid] = plex;
    }
#if !defined(LANDAU_SPECIES_MAJOR)
    PetscCall(DMCompositeAddDM(*pack,ctx->plex[grid]));
#else
    for (PetscInt b_id=0;b_id<ctx->batch_sz;b_id++) { // add batch size DMs for this species grid
      PetscCall(DMCompositeAddDM(*pack,ctx->plex[grid]));
    }
#endif
    PetscCall(DMSetApplicationContext(ctx->plex[grid], ctx));
  }
#if !defined(LANDAU_SPECIES_MAJOR)
  // stack the batched DMs, could do it all here!!! b_id=0
  for (PetscInt b_id=1;b_id<ctx->batch_sz;b_id++) {
    for (PetscInt grid=0;grid<ctx->num_grids;grid++) {
      PetscCall(DMCompositeAddDM(*pack,ctx->plex[grid]));
    }
  }
#endif
  // create ctx->mat_offset
  ctx->mat_offset[0] = 0;
  for (PetscInt grid=0 ; grid < ctx->num_grids ; grid++) {
    PetscInt    n;
    PetscCall(VecGetLocalSize(Xsub[grid],&n));
    ctx->mat_offset[grid+1] = ctx->mat_offset[grid] + n;
  }
  // creat DM & Jac
  PetscCall(DMSetApplicationContext(*pack, ctx));
  PetscCall(PetscOptionsInsertString(NULL,"-dm_preallocate_only"));
  PetscCall(DMSetFromOptions(*pack));
  PetscCall(DMCreateMatrix(*pack, &ctx->J));
  PetscCall(PetscOptionsInsertString(NULL,"-dm_preallocate_only false"));
  PetscCall(MatSetOption(ctx->J,MAT_STRUCTURALLY_SYMMETRIC, PETSC_TRUE));
  PetscCall(MatSetOption(ctx->J,MAT_IGNORE_ZERO_ENTRIES,PETSC_TRUE));
  PetscCall(PetscObjectSetName((PetscObject)ctx->J, "Jac"));
  // construct initial conditions in X
  PetscCall(DMCreateGlobalVector(*pack,X));
  for (PetscInt grid=0 ; grid < ctx->num_grids ; grid++) {
    PetscInt n;
    PetscCall(VecGetLocalSize(Xsub[grid],&n));
    for (PetscInt b_id = 0 ; b_id < ctx->batch_sz ; b_id++) {
      PetscScalar const *values;
      const PetscInt    moffset = LAND_MOFFSET(b_id,grid,ctx->batch_sz,ctx->num_grids,ctx->mat_offset);
      PetscCall(LandauSetInitialCondition(ctx->plex[grid], Xsub[grid], grid, b_id, ctx->batch_sz, ctx));
      PetscCall(VecGetArrayRead(Xsub[grid],&values));
      for (int i=0, idx = moffset; i<n; i++, idx++) {
        PetscCall(VecSetValue(*X,idx,values[i],INSERT_VALUES));
      }
      PetscCall(VecRestoreArrayRead(Xsub[grid],&values));
    }
  }
  // cleanup
  for (PetscInt grid=0 ; grid < ctx->num_grids ; grid++) {
    PetscCall(VecDestroy(&Xsub[grid]));
  }
  /* check for correct matrix type */
  if (ctx->gpu_assembly) { /* we need GPU object with GPU assembly */
    PetscBool flg;
    if (ctx->deviceType == LANDAU_CUDA) {
      PetscCall(PetscObjectTypeCompareAny((PetscObject)ctx->J,&flg,MATSEQAIJCUSPARSE,MATMPIAIJCUSPARSE,MATAIJCUSPARSE,""));
      PetscCheck(flg,ctx->comm,PETSC_ERR_ARG_WRONG,"must use '-dm_mat_type aijcusparse -dm_vec_type cuda' for GPU assembly and Cuda or use '-dm_landau_device_type cpu'");
    } else if (ctx->deviceType == LANDAU_KOKKOS) {
      PetscCall(PetscObjectTypeCompareAny((PetscObject)ctx->J,&flg,MATSEQAIJKOKKOS,MATMPIAIJKOKKOS,MATAIJKOKKOS,""));
#if defined(PETSC_HAVE_KOKKOS_KERNELS)
      PetscCheck(flg,ctx->comm,PETSC_ERR_ARG_WRONG,"must use '-dm_mat_type aijkokkos -dm_vec_type kokkos' for GPU assembly and Kokkos or use '-dm_landau_device_type cpu'");
#else
      PetscCheck(flg,ctx->comm,PETSC_ERR_ARG_WRONG,"must configure with '--download-kokkos-kernels' for GPU assembly and Kokkos or use '-dm_landau_device_type cpu'");
#endif
    }
  }
  PetscCall(PetscLogEventEnd(ctx->events[15],0,0,0,0));
  // create field major ordering

  ctx->work_vec   = NULL;
  ctx->plex_batch = NULL;
  ctx->batch_is   = NULL;
  for (int i=0;i<LANDAU_MAX_GRIDS;i++) grid_batch_is_inv[i] = NULL;
  PetscCall(PetscLogEventBegin(ctx->events[12],0,0,0,0));
  PetscCall(LandauCreateMatrix(comm, *X, grid_batch_is_inv, ctx));
  PetscCall(PetscLogEventEnd(ctx->events[12],0,0,0,0));

  // create AMR GPU assembly maps and static GPU data
  PetscCall(CreateStaticGPUData(dim,grid_batch_is_inv,ctx));

  PetscCall(PetscLogEventEnd(ctx->events[13],0,0,0,0));

  // create mass matrix
  PetscCall(DMPlexLandauCreateMassMatrix(*pack, NULL));

  if (J) *J = ctx->J;

  if (ctx->gpu_assembly && ctx->jacobian_field_major_order) {
    PetscContainer container;
    // cache ctx for KSP with batch/field major Jacobian ordering -ksp_type gmres/etc -dm_landau_jacobian_field_major_order
    PetscCall(PetscContainerCreate(PETSC_COMM_SELF, &container));
    PetscCall(PetscContainerSetPointer(container, (void *)ctx));
    PetscCall(PetscObjectCompose((PetscObject) ctx->J, "LandauCtx", (PetscObject) container));
    PetscCall(PetscContainerDestroy(&container));
    // batch solvers need to map -- can batch solvers work
    PetscCall(PetscContainerCreate(PETSC_COMM_SELF, &container));
    PetscCall(PetscContainerSetPointer(container, (void *)ctx->plex_batch));
    PetscCall(PetscObjectCompose((PetscObject) ctx->J, "plex_batch_is", (PetscObject) container));
    PetscCall(PetscContainerDestroy(&container));
  }
  // for batch solvers
  {
    PetscContainer  container;
    PetscInt        *pNf;
    PetscCall(PetscContainerCreate(PETSC_COMM_SELF, &container));
    PetscCall(PetscMalloc1(sizeof(*pNf), &pNf));
    *pNf = ctx->batch_sz;
    PetscCall(PetscContainerSetPointer(container, (void *)pNf));
    PetscCall(PetscContainerSetUserDestroy(container, MatrixNfDestroy));
    PetscCall(PetscObjectCompose((PetscObject)ctx->J, "batch size", (PetscObject) container));
    PetscCall(PetscContainerDestroy(&container));
  }

  PetscFunctionReturn(0);
}

/*@
 DMPlexLandauDestroyVelocitySpace - Destroy a DMPlex velocity space mesh

 Collective on dm

 Input/Output Parameters:
 .   dm - the dm to destroy

 Level: beginner

 .keywords: mesh
 .seealso: `DMPlexLandauCreateVelocitySpace()`
 @*/
PetscErrorCode DMPlexLandauDestroyVelocitySpace(DM *dm)
{
  LandauCtx      *ctx;
  PetscFunctionBegin;
  PetscCall(DMGetApplicationContext(*dm, &ctx));
  PetscCall(MatDestroy(&ctx->M));
  PetscCall(MatDestroy(&ctx->J));
  for (PetscInt ii=0;ii<ctx->num_species;ii++) PetscCall(PetscFEDestroy(&ctx->fe[ii]));
  PetscCall(ISDestroy(&ctx->batch_is));
  PetscCall(VecDestroy(&ctx->work_vec));
  PetscCall(VecScatterDestroy(&ctx->plex_batch));
  if (ctx->deviceType == LANDAU_CUDA) {
#if defined(PETSC_HAVE_CUDA)
    PetscCall(LandauCUDAStaticDataClear(&ctx->SData_d));
#else
    SETERRQ(ctx->comm,PETSC_ERR_ARG_WRONG,"-landau_device_type %s not built","cuda");
#endif
  } else if (ctx->deviceType == LANDAU_KOKKOS) {
#if defined(PETSC_HAVE_KOKKOS_KERNELS)
    PetscCall(LandauKokkosStaticDataClear(&ctx->SData_d));
#else
    SETERRQ(ctx->comm,PETSC_ERR_ARG_WRONG,"-landau_device_type %s not built","kokkos");
#endif
  } else {
    if (ctx->SData_d.x) { /* in a CPU run */
      PetscReal *invJ = (PetscReal*)ctx->SData_d.invJ, *xx = (PetscReal*)ctx->SData_d.x, *yy = (PetscReal*)ctx->SData_d.y, *zz = (PetscReal*)ctx->SData_d.z, *ww = (PetscReal*)ctx->SData_d.w;
      LandauIdx *coo_elem_offsets = (LandauIdx*)ctx->SData_d.coo_elem_offsets, *coo_elem_fullNb = (LandauIdx*)ctx->SData_d.coo_elem_fullNb, (*coo_elem_point_offsets)[LANDAU_MAX_NQ+1] = (LandauIdx (*)[LANDAU_MAX_NQ+1])ctx->SData_d.coo_elem_point_offsets;
      PetscCall(PetscFree4(ww,xx,yy,invJ));
      if (zz) PetscCall(PetscFree(zz));
      if (coo_elem_offsets) {
        PetscCall(PetscFree3(coo_elem_offsets,coo_elem_fullNb,coo_elem_point_offsets)); // could be NULL
      }
    }
  }

  if (ctx->times[LANDAU_MATRIX_TOTAL] > 0) { // OMP timings
    PetscCall(PetscPrintf(ctx->comm, "TSStep               N  1.0 %10.3e\n",ctx->times[LANDAU_EX2_TSSOLVE]));
    PetscCall(PetscPrintf(ctx->comm, "2:           Solve:  %10.3e with %" PetscInt_FMT " threads\n",ctx->times[LANDAU_EX2_TSSOLVE] - ctx->times[LANDAU_MATRIX_TOTAL],ctx->batch_sz));
    PetscCall(PetscPrintf(ctx->comm, "3:          Landau:  %10.3e\n",ctx->times[LANDAU_MATRIX_TOTAL]));
    PetscCall(PetscPrintf(ctx->comm, "Landau Jacobian       %" PetscInt_FMT " 1.0 %10.3e\n",(PetscInt)ctx->times[LANDAU_JACOBIAN_COUNT],ctx->times[LANDAU_JACOBIAN]));
    PetscCall(PetscPrintf(ctx->comm, "Landau Operator       N 1.0  %10.3e\n",ctx->times[LANDAU_OPERATOR]));
    PetscCall(PetscPrintf(ctx->comm, "Landau Mass           N 1.0  %10.3e\n",ctx->times[LANDAU_MASS]));
    PetscCall(PetscPrintf(ctx->comm, " Jac-f-df (GPU)       N 1.0  %10.3e\n",ctx->times[LANDAU_F_DF]));
    PetscCall(PetscPrintf(ctx->comm, " Kernel (GPU)         N 1.0  %10.3e\n",ctx->times[LANDAU_KERNEL]));
    PetscCall(PetscPrintf(ctx->comm, "MatLUFactorNum        X 1.0 %10.3e\n",ctx->times[KSP_FACTOR]));
    PetscCall(PetscPrintf(ctx->comm, "MatSolve              X 1.0 %10.3e\n",ctx->times[KSP_SOLVE]));
  }
  for (PetscInt grid=0 ; grid < ctx->num_grids ; grid++) {
    PetscCall(DMDestroy(&ctx->plex[grid]));
  }
  PetscFree(ctx);
  PetscCall(DMDestroy(dm));
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
 DMPlexLandauPrintNorms - collects moments and prints them

 Collective on dm

 Input Parameters:
 +   X  - the state
 -   stepi - current step to print

 Level: beginner

 .keywords: mesh
 .seealso: `DMPlexLandauCreateVelocitySpace()`
 @*/
PetscErrorCode DMPlexLandauPrintNorms(Vec X, PetscInt stepi)
{
  LandauCtx      *ctx;
  PetscDS        prob;
  DM             pack;
  PetscInt       cStart, cEnd, dim, ii, i0, nDMs;
  PetscScalar    xmomentumtot=0, ymomentumtot=0, zmomentumtot=0, energytot=0, densitytot=0, tt[LANDAU_MAX_SPECIES];
  PetscScalar    xmomentum[LANDAU_MAX_SPECIES],  ymomentum[LANDAU_MAX_SPECIES],  zmomentum[LANDAU_MAX_SPECIES], energy[LANDAU_MAX_SPECIES], density[LANDAU_MAX_SPECIES];
  Vec            *globXArray;

  PetscFunctionBegin;
  PetscCall(VecGetDM(X, &pack));
  PetscCheck(pack,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Vector has no DM");
  PetscCall(DMGetDimension(pack, &dim));
  PetscCheck(dim == 2 || dim == 3,PETSC_COMM_SELF, PETSC_ERR_PLIB, "dim %" PetscInt_FMT " not in [2,3]",dim);
  PetscCall(DMGetApplicationContext(pack, &ctx));
  PetscCheck(ctx,PETSC_COMM_SELF, PETSC_ERR_PLIB, "no context");
  /* print momentum and energy */
  PetscCall(DMCompositeGetNumberDM(pack,&nDMs));
  PetscCheck(nDMs == ctx->num_grids*ctx->batch_sz,PETSC_COMM_WORLD, PETSC_ERR_PLIB, "#DM wrong %" PetscInt_FMT " %" PetscInt_FMT,nDMs,ctx->num_grids*ctx->batch_sz);
  PetscCall(PetscMalloc(sizeof(*globXArray)*nDMs, &globXArray));
  PetscCall(DMCompositeGetAccessArray(pack, X, nDMs, NULL, globXArray));
  for (PetscInt grid = 0; grid < ctx->num_grids ; grid++) {
    Vec Xloc = globXArray[ LAND_PACK_IDX(ctx->batch_view_idx,grid) ];
    PetscCall(DMGetDS(ctx->plex[grid], &prob));
    for (ii=ctx->species_offset[grid],i0=0;ii<ctx->species_offset[grid+1];ii++,i0++) {
      PetscScalar user[2] = { (PetscScalar)i0, (PetscScalar)ctx->charges[ii]};
      PetscCall(PetscDSSetConstants(prob, 2, user));
      if (dim==2) { /* 2/3X + 3V (cylindrical coordinates) */
        PetscCall(PetscDSSetObjective(prob, 0, &f0_s_rden));
        PetscCall(DMPlexComputeIntegralFEM(ctx->plex[grid],Xloc,tt,ctx));
        density[ii] = tt[0]*ctx->n_0*ctx->charges[ii];
        PetscCall(PetscDSSetObjective(prob, 0, &f0_s_rmom));
        PetscCall(DMPlexComputeIntegralFEM(ctx->plex[grid],Xloc,tt,ctx));
        zmomentum[ii] = tt[0]*ctx->n_0*ctx->v_0*ctx->masses[ii];
        PetscCall(PetscDSSetObjective(prob, 0, &f0_s_rv2));
        PetscCall(DMPlexComputeIntegralFEM(ctx->plex[grid],Xloc,tt,ctx));
        energy[ii] = tt[0]*0.5*ctx->n_0*ctx->v_0*ctx->v_0*ctx->masses[ii];
        zmomentumtot += zmomentum[ii];
        energytot  += energy[ii];
        densitytot += density[ii];
        PetscCall(PetscPrintf(ctx->comm, "%3" PetscInt_FMT ") species-%" PetscInt_FMT ": charge density= %20.13e z-momentum= %20.13e energy= %20.13e",stepi,ii,(double)PetscRealPart(density[ii]),(double)PetscRealPart(zmomentum[ii]),(double)PetscRealPart(energy[ii])));
      } else { /* 2/3Xloc + 3V */
        PetscCall(PetscDSSetObjective(prob, 0, &f0_s_den));
        PetscCall(DMPlexComputeIntegralFEM(ctx->plex[grid],Xloc,tt,ctx));
        density[ii] = tt[0]*ctx->n_0*ctx->charges[ii];
        PetscCall(PetscDSSetObjective(prob, 0, &f0_s_mom));
        user[1] = 0;
        PetscCall(PetscDSSetConstants(prob, 2, user));
        PetscCall(DMPlexComputeIntegralFEM(ctx->plex[grid],Xloc,tt,ctx));
        xmomentum[ii]  = tt[0]*ctx->n_0*ctx->v_0*ctx->masses[ii];
        user[1] = 1;
        PetscCall(PetscDSSetConstants(prob, 2, user));
        PetscCall(DMPlexComputeIntegralFEM(ctx->plex[grid],Xloc,tt,ctx));
        ymomentum[ii] = tt[0]*ctx->n_0*ctx->v_0*ctx->masses[ii];
        user[1] = 2;
        PetscCall(PetscDSSetConstants(prob, 2, user));
        PetscCall(DMPlexComputeIntegralFEM(ctx->plex[grid],Xloc,tt,ctx));
        zmomentum[ii] = tt[0]*ctx->n_0*ctx->v_0*ctx->masses[ii];
        if (ctx->use_relativistic_corrections) {
          /* gamma * M * f */
          if (ii==0 && grid==0) { // do all at once
            Vec            Mf, globGamma, *globMfArray, *globGammaArray;
            PetscErrorCode (*gammaf[1])(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar [], void *) = {gamma_n_f};
            PetscReal      *c2_0[1], data[1];

            PetscCall(VecDuplicate(X,&globGamma));
            PetscCall(VecDuplicate(X,&Mf));
            PetscCall(PetscMalloc(sizeof(*globMfArray)*nDMs, &globMfArray));
            PetscCall(PetscMalloc(sizeof(*globMfArray)*nDMs, &globGammaArray));
            /* M * f */
            PetscCall(MatMult(ctx->M,X,Mf));
            /* gamma */
            PetscCall(DMCompositeGetAccessArray(pack, globGamma, nDMs, NULL, globGammaArray));
            for (PetscInt grid = 0; grid < ctx->num_grids ; grid++) { // yes a grid loop in a grid loop to print nice, need to fix for batching
              Vec v1 = globGammaArray[ LAND_PACK_IDX(ctx->batch_view_idx,grid) ];
              data[0] = PetscSqr(C_0(ctx->v_0));
              c2_0[0] = &data[0];
              PetscCall(DMProjectFunction(ctx->plex[grid], 0., gammaf, (void**)c2_0, INSERT_ALL_VALUES, v1));
            }
            PetscCall(DMCompositeRestoreAccessArray(pack, globGamma, nDMs, NULL, globGammaArray));
            /* gamma * Mf */
            PetscCall(DMCompositeGetAccessArray(pack, globGamma, nDMs, NULL, globGammaArray));
            PetscCall(DMCompositeGetAccessArray(pack, Mf, nDMs, NULL, globMfArray));
            for (PetscInt grid = 0; grid < ctx->num_grids ; grid++) { // yes a grid loop in a grid loop to print nice
              PetscInt Nf = ctx->species_offset[grid+1] - ctx->species_offset[grid], N, bs;
              Vec      Mfsub = globMfArray[ LAND_PACK_IDX(ctx->batch_view_idx,grid) ], Gsub = globGammaArray[ LAND_PACK_IDX(ctx->batch_view_idx,grid) ], v1, v2;
              // get each component
              PetscCall(VecGetSize(Mfsub,&N));
              PetscCall(VecCreate(ctx->comm,&v1));
              PetscCall(VecSetSizes(v1,PETSC_DECIDE,N/Nf));
              PetscCall(VecCreate(ctx->comm,&v2));
              PetscCall(VecSetSizes(v2,PETSC_DECIDE,N/Nf));
              PetscCall(VecSetFromOptions(v1)); // ???
              PetscCall(VecSetFromOptions(v2));
              // get each component
              PetscCall(VecGetBlockSize(Gsub,&bs));
              PetscCheck(bs == Nf,PETSC_COMM_SELF, PETSC_ERR_PLIB, "bs %" PetscInt_FMT " != num_species %" PetscInt_FMT " in Gsub",bs,Nf);
              PetscCall(VecGetBlockSize(Mfsub,&bs));
              PetscCheck(bs == Nf,PETSC_COMM_SELF, PETSC_ERR_PLIB, "bs %" PetscInt_FMT " != num_species %" PetscInt_FMT,bs,Nf);
              for (int i=0, ix=ctx->species_offset[grid] ; i<Nf ; i++, ix++) {
                PetscScalar val;
                PetscCall(VecStrideGather(Gsub,i,v1,INSERT_VALUES));
                PetscCall(VecStrideGather(Mfsub,i,v2,INSERT_VALUES));
                PetscCall(VecDot(v1,v2,&val));
                energy[ix] = PetscRealPart(val)*ctx->n_0*ctx->v_0*ctx->v_0*ctx->masses[ix];
              }
              PetscCall(VecDestroy(&v1));
              PetscCall(VecDestroy(&v2));
            } /* grids */
            PetscCall(DMCompositeRestoreAccessArray(pack, globGamma, nDMs, NULL, globGammaArray));
            PetscCall(DMCompositeRestoreAccessArray(pack, Mf, nDMs, NULL, globMfArray));
            PetscCall(PetscFree(globGammaArray));
            PetscCall(PetscFree(globMfArray));
            PetscCall(VecDestroy(&globGamma));
            PetscCall(VecDestroy(&Mf));
          }
        } else {
          PetscCall(PetscDSSetObjective(prob, 0, &f0_s_v2));
          PetscCall(DMPlexComputeIntegralFEM(ctx->plex[grid],Xloc,tt,ctx));
          energy[ii]    = 0.5*tt[0]*ctx->n_0*ctx->v_0*ctx->v_0*ctx->masses[ii];
        }
        PetscCall(PetscPrintf(ctx->comm, "%3" PetscInt_FMT ") species %" PetscInt_FMT ": density=%20.13e, x-momentum=%20.13e, y-momentum=%20.13e, z-momentum=%20.13e, energy=%21.13e",stepi,ii,(double)PetscRealPart(density[ii]),(double)PetscRealPart(xmomentum[ii]),(double)PetscRealPart(ymomentum[ii]),(double)PetscRealPart(zmomentum[ii]),(double)PetscRealPart(energy[ii])));
        xmomentumtot += xmomentum[ii];
        ymomentumtot += ymomentum[ii];
        zmomentumtot += zmomentum[ii];
        energytot    += energy[ii];
        densitytot   += density[ii];
      }
      if (ctx->num_species>1) PetscPrintf(ctx->comm, "\n");
    }
  }
  PetscCall(DMCompositeRestoreAccessArray(pack, X, nDMs, NULL, globXArray));
  PetscCall(PetscFree(globXArray));
  /* totals */
  PetscCall(DMPlexGetHeightStratum(ctx->plex[0],0,&cStart,&cEnd));
  if (ctx->num_species>1) {
    if (dim==2) {
      PetscCall(PetscPrintf(ctx->comm, "\t%3" PetscInt_FMT ") Total: charge density=%21.13e, momentum=%21.13e, energy=%21.13e (m_i[0]/m_e = %g, %" PetscInt_FMT " cells on electron grid)",stepi,(double)PetscRealPart(densitytot),(double)PetscRealPart(zmomentumtot),(double)PetscRealPart(energytot),(double)(ctx->masses[1]/ctx->masses[0]),cEnd-cStart));
    } else {
      PetscCall(PetscPrintf(ctx->comm, "\t%3" PetscInt_FMT ") Total: charge density=%21.13e, x-momentum=%21.13e, y-momentum=%21.13e, z-momentum=%21.13e, energy=%21.13e (m_i[0]/m_e = %g, %" PetscInt_FMT " cells)",stepi,(double)PetscRealPart(densitytot),(double)PetscRealPart(xmomentumtot),(double)PetscRealPart(ymomentumtot),(double)PetscRealPart(zmomentumtot),(double)PetscRealPart(energytot),(double)(ctx->masses[1]/ctx->masses[0]),cEnd-cStart));
    }
  } else PetscCall(PetscPrintf(ctx->comm, " -- %" PetscInt_FMT " cells",cEnd-cStart));
  PetscCall(PetscPrintf(ctx->comm,"\n"));
  PetscFunctionReturn(0);
}

/*@
 DMPlexLandauCreateMassMatrix - Create mass matrix for Landau in Plex space (not field major order of Jacobian)

 Collective on pack

 Input Parameters:
. pack     - the DM object

 Output Parameters:
. Amat - The mass matrix (optional), mass matrix is added to the DM context

 Level: beginner

 .keywords: mesh
 .seealso: `DMPlexLandauCreateVelocitySpace()`
 @*/
PetscErrorCode DMPlexLandauCreateMassMatrix(DM pack, Mat *Amat)
{
  DM             mass_pack,massDM[LANDAU_MAX_GRIDS];
  PetscDS        prob;
  PetscInt       ii,dim,N1=1,N2;
  LandauCtx      *ctx;
  Mat            packM,subM[LANDAU_MAX_GRIDS];

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pack,DM_CLASSID,1);
  if (Amat) PetscValidPointer(Amat,2);
  PetscCall(DMGetApplicationContext(pack, &ctx));
  PetscCheck(ctx,PETSC_COMM_SELF, PETSC_ERR_PLIB, "no context");
  PetscCall(PetscLogEventBegin(ctx->events[14],0,0,0,0));
  PetscCall(DMGetDimension(pack, &dim));
  PetscCall(DMCompositeCreate(PetscObjectComm((PetscObject) pack),&mass_pack));
  /* create pack mass matrix */
  for (PetscInt grid=0, ix=0 ; grid<ctx->num_grids ; grid++) {
    PetscCall(DMClone(ctx->plex[grid], &massDM[grid]));
    PetscCall(DMCopyFields(ctx->plex[grid], massDM[grid]));
    PetscCall(DMCreateDS(massDM[grid]));
    PetscCall(DMGetDS(massDM[grid], &prob));
    for (ix=0, ii=ctx->species_offset[grid];ii<ctx->species_offset[grid+1];ii++,ix++) {
      if (dim==3) PetscCall(PetscDSSetJacobian(prob, ix, ix, g0_1, NULL, NULL, NULL));
      else        PetscCall(PetscDSSetJacobian(prob, ix, ix, g0_r, NULL, NULL, NULL));
    }
#if !defined(LANDAU_SPECIES_MAJOR)
    PetscCall(DMCompositeAddDM(mass_pack,massDM[grid]));
#else
    for (PetscInt b_id=0;b_id<ctx->batch_sz;b_id++) { // add batch size DMs for this species grid
      PetscCall(DMCompositeAddDM(mass_pack,massDM[grid]));
    }
#endif
    PetscCall(DMCreateMatrix(massDM[grid], &subM[grid]));
  }
#if !defined(LANDAU_SPECIES_MAJOR)
  // stack the batched DMs
  for (PetscInt b_id=1;b_id<ctx->batch_sz;b_id++) {
    for (PetscInt grid=0;grid<ctx->num_grids;grid++) {
      PetscCall(DMCompositeAddDM(mass_pack, massDM[grid]));
    }
  }
#endif
  PetscCall(PetscOptionsInsertString(NULL,"-dm_preallocate_only"));
  PetscCall(DMSetFromOptions(mass_pack));
  PetscCall(DMCreateMatrix(mass_pack, &packM));
  PetscCall(PetscOptionsInsertString(NULL,"-dm_preallocate_only false"));
  PetscCall(MatSetOption(packM,MAT_STRUCTURALLY_SYMMETRIC, PETSC_TRUE));
  PetscCall(MatSetOption(packM,MAT_IGNORE_ZERO_ENTRIES,PETSC_TRUE));
  PetscCall(DMDestroy(&mass_pack));
  /* make mass matrix for each block */
  for (PetscInt grid=0;grid<ctx->num_grids;grid++) {
    Vec locX;
    DM  plex = massDM[grid];
    PetscCall(DMGetLocalVector(plex, &locX));
    /* Mass matrix is independent of the input, so no need to fill locX */
    PetscCall(DMPlexSNESComputeJacobianFEM(plex, locX, subM[grid], subM[grid], ctx));
    PetscCall(DMRestoreLocalVector(plex, &locX));
    PetscCall(DMDestroy(&massDM[grid]));
  }
  PetscCall(MatGetSize(ctx->J, &N1, NULL));
  PetscCall(MatGetSize(packM, &N2, NULL));
  PetscCheck(N1 == N2,PetscObjectComm((PetscObject) pack), PETSC_ERR_PLIB, "Incorrect matrix sizes: |Jacobian| = %" PetscInt_FMT ", |Mass|=%" PetscInt_FMT,N1,N2);
  /* assemble block diagonals */
  for (PetscInt grid=0 ; grid<ctx->num_grids ; grid++) {
    Mat               B = subM[grid];
    PetscInt          nloc, nzl, *colbuf, COL_BF_SIZE=1024, row;
    PetscCall(PetscMalloc(sizeof(*colbuf)*COL_BF_SIZE, &colbuf));
    PetscCall(MatGetSize(B, &nloc, NULL));
    for (PetscInt b_id = 0 ; b_id < ctx->batch_sz ; b_id++) {
      const PetscInt    moffset = LAND_MOFFSET(b_id,grid,ctx->batch_sz,ctx->num_grids,ctx->mat_offset);
      const PetscInt    *cols;
      const PetscScalar *vals;
      for (int i=0 ; i<nloc ; i++) {
        PetscCall(MatGetRow(B,i,&nzl,NULL,NULL));
        if (nzl>COL_BF_SIZE) {
          PetscCall(PetscFree(colbuf));
          PetscCall(PetscInfo(pack, "Realloc buffer %" PetscInt_FMT " to %" PetscInt_FMT " (row size %" PetscInt_FMT ") \n",COL_BF_SIZE,2*COL_BF_SIZE,nzl));
          COL_BF_SIZE = 2*nzl;
          PetscCall(PetscMalloc(sizeof(*colbuf)*COL_BF_SIZE, &colbuf));
        }
        PetscCall(MatGetRow(B,i,&nzl,&cols,&vals));
        for (int j=0; j<nzl; j++) colbuf[j] = cols[j] + moffset;
        row = i + moffset;
        PetscCall(MatSetValues(packM,1,&row,nzl,colbuf,vals,INSERT_VALUES));
        PetscCall(MatRestoreRow(B,i,&nzl,&cols,&vals));
      }
    }
    PetscCall(PetscFree(colbuf));
  }
  // cleanup
  for (PetscInt grid=0 ; grid<ctx->num_grids ; grid++) {
    PetscCall(MatDestroy(&subM[grid]));
  }
  PetscCall(MatAssemblyBegin(packM,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(packM,MAT_FINAL_ASSEMBLY));
  PetscCall(PetscObjectSetName((PetscObject)packM, "mass"));
  PetscCall(MatViewFromOptions(packM,NULL,"-dm_landau_mass_view"));
  ctx->M = packM;
  if (Amat) *Amat = packM;
  PetscCall(PetscLogEventEnd(ctx->events[14],0,0,0,0));
  PetscFunctionReturn(0);
}

/*@
 DMPlexLandauIFunction - TS residual calculation, confusingly this computes the Jacobian w/o mass

 Collective on ts

 Input Parameters:
+   TS  - The time stepping context
.   time_dummy - current time (not used)
.   X - Current state
.   X_t - Time derivative of current state
-   actx - Landau context

 Output Parameter:
.   F  - The residual

 Level: beginner

 .keywords: mesh
 .seealso: `DMPlexLandauCreateVelocitySpace()`, `DMPlexLandauIJacobian()`
 @*/
PetscErrorCode DMPlexLandauIFunction(TS ts, PetscReal time_dummy, Vec X, Vec X_t, Vec F, void *actx)
{
  LandauCtx        *ctx=(LandauCtx*)actx;
  PetscInt         dim;
  DM               pack;
#if defined(PETSC_HAVE_THREADSAFETY)
  double           starttime, endtime;
#endif
  PetscObjectState state;

  PetscFunctionBegin;
  PetscCall(TSGetDM(ts,&pack));
  PetscCall(DMGetApplicationContext(pack, &ctx));
  PetscCheck(ctx,PETSC_COMM_SELF, PETSC_ERR_PLIB, "no context");
  if (ctx->stage) PetscCall(PetscLogStagePush(ctx->stage));
  PetscCall(PetscLogEventBegin(ctx->events[11],0,0,0,0));
  PetscCall(PetscLogEventBegin(ctx->events[0],0,0,0,0));
#if defined(PETSC_HAVE_THREADSAFETY)
  starttime = MPI_Wtime();
#endif
  PetscCall(DMGetDimension(pack, &dim));
  PetscCall(PetscObjectStateGet((PetscObject)ctx->J,&state));
  if (state != ctx->norm_state) {
    PetscCall(PetscInfo(ts, "Create Landau Jacobian t=%g J.state %" PetscInt64_FMT " --> %" PetscInt64_FMT "\n",(double)time_dummy, ctx->norm_state, state));
    PetscCall(MatZeroEntries(ctx->J));
    PetscCall(LandauFormJacobian_Internal(X,ctx->J,dim,0.0,(void*)ctx));
    PetscCall(MatViewFromOptions(ctx->J, NULL, "-dm_landau_jacobian_view"));
    PetscCall(PetscObjectStateGet((PetscObject)ctx->J,&state));
    ctx->norm_state = state;
  } else {
    PetscCall(PetscInfo(ts, "WARNING Skip forming Jacobian, has not changed %" PetscInt64_FMT "\n",state));
  }
  /* mat vec for op */
  PetscCall(MatMult(ctx->J,X,F)); /* C*f */
  /* add time term */
  if (X_t) PetscCall(MatMultAdd(ctx->M,X_t,F,F));
#if defined(PETSC_HAVE_THREADSAFETY)
  if (ctx->stage) {
    endtime = MPI_Wtime();
    ctx->times[LANDAU_OPERATOR] += (endtime - starttime);
    ctx->times[LANDAU_JACOBIAN] += (endtime - starttime);
    ctx->times[LANDAU_JACOBIAN_COUNT] += 1;
  }
#endif
  PetscCall(PetscLogEventEnd(ctx->events[0],0,0,0,0));
  PetscCall(PetscLogEventEnd(ctx->events[11],0,0,0,0));
  if (ctx->stage) {
    PetscCall(PetscLogStagePop());
#if defined(PETSC_HAVE_THREADSAFETY)
    ctx->times[LANDAU_MATRIX_TOTAL] += (endtime - starttime);
#endif
  }
  PetscFunctionReturn(0);
}

/*@
 DMPlexLandauIJacobian - TS Jacobian construction, confusingly this adds mass

 Collective on ts

 Input Parameters:
+   TS  - The time stepping context
.   time_dummy - current time (not used)
.   X - Current state
.   U_tdummy - Time derivative of current state (not used)
.   shift - shift for du/dt term
-   actx - Landau context

 Output Parameters:
+   Amat  - Jacobian
-   Pmat  - same as Amat

 Level: beginner

 .keywords: mesh
 .seealso: `DMPlexLandauCreateVelocitySpace()`, `DMPlexLandauIFunction()`
 @*/
PetscErrorCode DMPlexLandauIJacobian(TS ts, PetscReal time_dummy, Vec X, Vec U_tdummy, PetscReal shift, Mat Amat, Mat Pmat, void *actx)
{
  LandauCtx        *ctx=NULL;
  PetscInt         dim;
  DM               pack;
#if defined(PETSC_HAVE_THREADSAFETY)
  double           starttime, endtime;
#endif
  PetscObjectState state;

  PetscFunctionBegin;
  PetscCall(TSGetDM(ts,&pack));
  PetscCall(DMGetApplicationContext(pack, &ctx));
  PetscCheck(ctx,PETSC_COMM_SELF, PETSC_ERR_PLIB, "no context");
  PetscCheck(Amat == Pmat && Amat == ctx->J,ctx->comm, PETSC_ERR_PLIB, "Amat!=Pmat || Amat!=ctx->J");
  PetscCall(DMGetDimension(pack, &dim));
  /* get collision Jacobian into A */
  if (ctx->stage) PetscCall(PetscLogStagePush(ctx->stage));
  PetscCall(PetscLogEventBegin(ctx->events[11],0,0,0,0));
  PetscCall(PetscLogEventBegin(ctx->events[9],0,0,0,0));
#if defined(PETSC_HAVE_THREADSAFETY)
  starttime = MPI_Wtime();
#endif
  PetscCall(PetscInfo(ts, "Adding mass to Jacobian t=%g, shift=%g\n",(double)time_dummy,(double)shift));
  PetscCheck(shift!=0.0,ctx->comm, PETSC_ERR_PLIB, "zero shift");
  PetscCall(PetscObjectStateGet((PetscObject)ctx->J,&state));
  PetscCheck(state == ctx->norm_state,ctx->comm, PETSC_ERR_PLIB, "wrong state, %" PetscInt64_FMT " %" PetscInt64_FMT "",ctx->norm_state,state);
  if (!ctx->use_matrix_mass) {
    PetscCall(LandauFormJacobian_Internal(X,ctx->J,dim,shift,(void*)ctx));
    PetscCall(MatViewFromOptions(ctx->J, NULL, "-dm_landau_mat_view"));
  } else { /* add mass */
    PetscCall(MatAXPY(Pmat,shift,ctx->M,SAME_NONZERO_PATTERN));
  }
#if defined(PETSC_HAVE_THREADSAFETY)
  if (ctx->stage) {
    endtime = MPI_Wtime();
    ctx->times[LANDAU_OPERATOR] += (endtime - starttime);
    ctx->times[LANDAU_MASS] += (endtime - starttime);
  }
#endif
  PetscCall(PetscLogEventEnd(ctx->events[9],0,0,0,0));
  PetscCall(PetscLogEventEnd(ctx->events[11],0,0,0,0));
  if (ctx->stage) {
    PetscCall(PetscLogStagePop());
#if defined(PETSC_HAVE_THREADSAFETY)
    ctx->times[LANDAU_MATRIX_TOTAL] += (endtime - starttime);
#endif
  }
  PetscFunctionReturn(0);
}
