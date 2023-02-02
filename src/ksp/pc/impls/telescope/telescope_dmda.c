
#include <petsc/private/matimpl.h>
#include <petsc/private/pcimpl.h>
#include <petsc/private/dmimpl.h>
#include <petscksp.h> /*I "petscksp.h" I*/
#include <petscdm.h>
#include <petscdmda.h>

#include "../src/ksp/pc/impls/telescope/telescope.h"

static PetscBool  cited      = PETSC_FALSE;
static const char citation[] = "@inproceedings{MaySananRuppKnepleySmith2016,\n"
                               "  title     = {Extreme-Scale Multigrid Components within PETSc},\n"
                               "  author    = {Dave A. May and Patrick Sanan and Karl Rupp and Matthew G. Knepley and Barry F. Smith},\n"
                               "  booktitle = {Proceedings of the Platform for Advanced Scientific Computing Conference},\n"
                               "  series    = {PASC '16},\n"
                               "  isbn      = {978-1-4503-4126-4},\n"
                               "  location  = {Lausanne, Switzerland},\n"
                               "  pages     = {5:1--5:12},\n"
                               "  articleno = {5},\n"
                               "  numpages  = {12},\n"
                               "  url       = {https://doi.acm.org/10.1145/2929908.2929913},\n"
                               "  doi       = {10.1145/2929908.2929913},\n"
                               "  acmid     = {2929913},\n"
                               "  publisher = {ACM},\n"
                               "  address   = {New York, NY, USA},\n"
                               "  keywords  = {GPU, HPC, agglomeration, coarse-level solver, multigrid, parallel computing, preconditioning},\n"
                               "  year      = {2016}\n"
                               "}\n";

static PetscErrorCode _DMDADetermineRankFromGlobalIJK(PetscInt dim, PetscInt i, PetscInt j, PetscInt k, PetscInt Mp, PetscInt Np, PetscInt Pp, PetscInt start_i[], PetscInt start_j[], PetscInt start_k[], PetscInt span_i[], PetscInt span_j[], PetscInt span_k[], PetscMPIInt *_pi, PetscMPIInt *_pj, PetscMPIInt *_pk, PetscMPIInt *rank_re)
{
  PetscInt pi, pj, pk, n;

  PetscFunctionBegin;
  *rank_re = -1;
  if (_pi) *_pi = -1;
  if (_pj) *_pj = -1;
  if (_pk) *_pk = -1;
  pi = pj = pk = -1;
  if (_pi) {
    for (n = 0; n < Mp; n++) {
      if ((i >= start_i[n]) && (i < start_i[n] + span_i[n])) {
        pi = n;
        break;
      }
    }
    PetscCheck(pi != -1, PETSC_COMM_SELF, PETSC_ERR_USER, "[dmda-ijk] pi cannot be determined : range %" PetscInt_FMT ", val %" PetscInt_FMT, Mp, i);
    *_pi = pi;
  }

  if (_pj) {
    for (n = 0; n < Np; n++) {
      if ((j >= start_j[n]) && (j < start_j[n] + span_j[n])) {
        pj = n;
        break;
      }
    }
    PetscCheck(pj != -1, PETSC_COMM_SELF, PETSC_ERR_USER, "[dmda-ijk] pj cannot be determined : range %" PetscInt_FMT ", val %" PetscInt_FMT, Np, j);
    *_pj = pj;
  }

  if (_pk) {
    for (n = 0; n < Pp; n++) {
      if ((k >= start_k[n]) && (k < start_k[n] + span_k[n])) {
        pk = n;
        break;
      }
    }
    PetscCheck(pk != -1, PETSC_COMM_SELF, PETSC_ERR_USER, "[dmda-ijk] pk cannot be determined : range %" PetscInt_FMT ", val %" PetscInt_FMT, Pp, k);
    *_pk = pk;
  }

  switch (dim) {
  case 1:
    *rank_re = pi;
    break;
  case 2:
    *rank_re = pi + pj * Mp;
    break;
  case 3:
    *rank_re = pi + pj * Mp + pk * (Mp * Np);
    break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode _DMDADetermineGlobalS0(PetscInt dim, PetscMPIInt rank_re, PetscInt Mp_re, PetscInt Np_re, PetscInt Pp_re, PetscInt range_i_re[], PetscInt range_j_re[], PetscInt range_k_re[], PetscInt *s0)
{
  PetscInt i, j, k, start_IJK = 0;
  PetscInt rank_ijk;

  PetscFunctionBegin;
  switch (dim) {
  case 1:
    for (i = 0; i < Mp_re; i++) {
      rank_ijk = i;
      if (rank_ijk < rank_re) start_IJK += range_i_re[i];
    }
    break;
  case 2:
    for (j = 0; j < Np_re; j++) {
      for (i = 0; i < Mp_re; i++) {
        rank_ijk = i + j * Mp_re;
        if (rank_ijk < rank_re) start_IJK += range_i_re[i] * range_j_re[j];
      }
    }
    break;
  case 3:
    for (k = 0; k < Pp_re; k++) {
      for (j = 0; j < Np_re; j++) {
        for (i = 0; i < Mp_re; i++) {
          rank_ijk = i + j * Mp_re + k * Mp_re * Np_re;
          if (rank_ijk < rank_re) start_IJK += range_i_re[i] * range_j_re[j] * range_k_re[k];
        }
      }
    }
    break;
  }
  *s0 = start_IJK;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCTelescopeSetUp_dmda_repart_coors2d(PC_Telescope sred, DM dm, DM subdm)
{
  DM         cdm;
  Vec        coor, coor_natural, perm_coors;
  PetscInt   i, j, si, sj, ni, nj, M, N, Ml, Nl, c, nidx;
  PetscInt  *fine_indices;
  IS         is_fine, is_local;
  VecScatter sctx;

  PetscFunctionBegin;
  PetscCall(DMGetCoordinates(dm, &coor));
  if (!coor) PetscFunctionReturn(PETSC_SUCCESS);
  if (PCTelescope_isActiveRank(sred)) PetscCall(DMDASetUniformCoordinates(subdm, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0));
  /* Get the coordinate vector from the distributed array */
  PetscCall(DMGetCoordinateDM(dm, &cdm));
  PetscCall(DMDACreateNaturalVector(cdm, &coor_natural));

  PetscCall(DMDAGlobalToNaturalBegin(cdm, coor, INSERT_VALUES, coor_natural));
  PetscCall(DMDAGlobalToNaturalEnd(cdm, coor, INSERT_VALUES, coor_natural));

  /* get indices of the guys I want to grab */
  PetscCall(DMDAGetInfo(dm, NULL, &M, &N, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
  if (PCTelescope_isActiveRank(sred)) {
    PetscCall(DMDAGetCorners(subdm, &si, &sj, NULL, &ni, &nj, NULL));
    Ml = ni;
    Nl = nj;
  } else {
    si = sj = 0;
    ni = nj = 0;
    Ml = Nl = 0;
  }

  PetscCall(PetscMalloc1(Ml * Nl * 2, &fine_indices));
  c = 0;
  if (PCTelescope_isActiveRank(sred)) {
    for (j = sj; j < sj + nj; j++) {
      for (i = si; i < si + ni; i++) {
        nidx                = (i) + (j)*M;
        fine_indices[c]     = 2 * nidx;
        fine_indices[c + 1] = 2 * nidx + 1;
        c                   = c + 2;
      }
    }
    PetscCheck(c == Ml * Nl * 2, PETSC_COMM_SELF, PETSC_ERR_PLIB, "c %" PetscInt_FMT " should equal 2 * Ml %" PetscInt_FMT " * Nl %" PetscInt_FMT, c, Ml, Nl);
  }

  /* generate scatter */
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)dm), Ml * Nl * 2, fine_indices, PETSC_USE_POINTER, &is_fine));
  PetscCall(ISCreateStride(PETSC_COMM_SELF, Ml * Nl * 2, 0, 1, &is_local));

  /* scatter */
  PetscCall(VecCreate(PETSC_COMM_SELF, &perm_coors));
  PetscCall(VecSetSizes(perm_coors, PETSC_DECIDE, Ml * Nl * 2));
  PetscCall(VecSetType(perm_coors, VECSEQ));

  PetscCall(VecScatterCreate(coor_natural, is_fine, perm_coors, is_local, &sctx));
  PetscCall(VecScatterBegin(sctx, coor_natural, perm_coors, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(sctx, coor_natural, perm_coors, INSERT_VALUES, SCATTER_FORWARD));
  /* access */
  if (PCTelescope_isActiveRank(sred)) {
    Vec                _coors;
    const PetscScalar *LA_perm;
    PetscScalar       *LA_coors;

    PetscCall(DMGetCoordinates(subdm, &_coors));
    PetscCall(VecGetArrayRead(perm_coors, &LA_perm));
    PetscCall(VecGetArray(_coors, &LA_coors));
    for (i = 0; i < Ml * Nl * 2; i++) LA_coors[i] = LA_perm[i];
    PetscCall(VecRestoreArray(_coors, &LA_coors));
    PetscCall(VecRestoreArrayRead(perm_coors, &LA_perm));
  }

  /* update local coords */
  if (PCTelescope_isActiveRank(sred)) {
    DM  _dmc;
    Vec _coors, _coors_local;
    PetscCall(DMGetCoordinateDM(subdm, &_dmc));
    PetscCall(DMGetCoordinates(subdm, &_coors));
    PetscCall(DMGetCoordinatesLocal(subdm, &_coors_local));
    PetscCall(DMGlobalToLocalBegin(_dmc, _coors, INSERT_VALUES, _coors_local));
    PetscCall(DMGlobalToLocalEnd(_dmc, _coors, INSERT_VALUES, _coors_local));
  }
  PetscCall(VecScatterDestroy(&sctx));
  PetscCall(ISDestroy(&is_fine));
  PetscCall(PetscFree(fine_indices));
  PetscCall(ISDestroy(&is_local));
  PetscCall(VecDestroy(&perm_coors));
  PetscCall(VecDestroy(&coor_natural));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCTelescopeSetUp_dmda_repart_coors3d(PC_Telescope sred, DM dm, DM subdm)
{
  DM         cdm;
  Vec        coor, coor_natural, perm_coors;
  PetscInt   i, j, k, si, sj, sk, ni, nj, nk, M, N, P, Ml, Nl, Pl, c, nidx;
  PetscInt  *fine_indices;
  IS         is_fine, is_local;
  VecScatter sctx;

  PetscFunctionBegin;
  PetscCall(DMGetCoordinates(dm, &coor));
  if (!coor) PetscFunctionReturn(PETSC_SUCCESS);

  if (PCTelescope_isActiveRank(sred)) PetscCall(DMDASetUniformCoordinates(subdm, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0));

  /* Get the coordinate vector from the distributed array */
  PetscCall(DMGetCoordinateDM(dm, &cdm));
  PetscCall(DMDACreateNaturalVector(cdm, &coor_natural));
  PetscCall(DMDAGlobalToNaturalBegin(cdm, coor, INSERT_VALUES, coor_natural));
  PetscCall(DMDAGlobalToNaturalEnd(cdm, coor, INSERT_VALUES, coor_natural));

  /* get indices of the guys I want to grab */
  PetscCall(DMDAGetInfo(dm, NULL, &M, &N, &P, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));

  if (PCTelescope_isActiveRank(sred)) {
    PetscCall(DMDAGetCorners(subdm, &si, &sj, &sk, &ni, &nj, &nk));
    Ml = ni;
    Nl = nj;
    Pl = nk;
  } else {
    si = sj = sk = 0;
    ni = nj = nk = 0;
    Ml = Nl = Pl = 0;
  }

  PetscCall(PetscMalloc1(Ml * Nl * Pl * 3, &fine_indices));

  c = 0;
  if (PCTelescope_isActiveRank(sred)) {
    for (k = sk; k < sk + nk; k++) {
      for (j = sj; j < sj + nj; j++) {
        for (i = si; i < si + ni; i++) {
          nidx                = (i) + (j)*M + (k)*M * N;
          fine_indices[c]     = 3 * nidx;
          fine_indices[c + 1] = 3 * nidx + 1;
          fine_indices[c + 2] = 3 * nidx + 2;
          c                   = c + 3;
        }
      }
    }
  }

  /* generate scatter */
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)dm), Ml * Nl * Pl * 3, fine_indices, PETSC_USE_POINTER, &is_fine));
  PetscCall(ISCreateStride(PETSC_COMM_SELF, Ml * Nl * Pl * 3, 0, 1, &is_local));

  /* scatter */
  PetscCall(VecCreate(PETSC_COMM_SELF, &perm_coors));
  PetscCall(VecSetSizes(perm_coors, PETSC_DECIDE, Ml * Nl * Pl * 3));
  PetscCall(VecSetType(perm_coors, VECSEQ));
  PetscCall(VecScatterCreate(coor_natural, is_fine, perm_coors, is_local, &sctx));
  PetscCall(VecScatterBegin(sctx, coor_natural, perm_coors, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(sctx, coor_natural, perm_coors, INSERT_VALUES, SCATTER_FORWARD));

  /* access */
  if (PCTelescope_isActiveRank(sred)) {
    Vec                _coors;
    const PetscScalar *LA_perm;
    PetscScalar       *LA_coors;

    PetscCall(DMGetCoordinates(subdm, &_coors));
    PetscCall(VecGetArrayRead(perm_coors, &LA_perm));
    PetscCall(VecGetArray(_coors, &LA_coors));
    for (i = 0; i < Ml * Nl * Pl * 3; i++) LA_coors[i] = LA_perm[i];
    PetscCall(VecRestoreArray(_coors, &LA_coors));
    PetscCall(VecRestoreArrayRead(perm_coors, &LA_perm));
  }

  /* update local coords */
  if (PCTelescope_isActiveRank(sred)) {
    DM  _dmc;
    Vec _coors, _coors_local;

    PetscCall(DMGetCoordinateDM(subdm, &_dmc));
    PetscCall(DMGetCoordinates(subdm, &_coors));
    PetscCall(DMGetCoordinatesLocal(subdm, &_coors_local));
    PetscCall(DMGlobalToLocalBegin(_dmc, _coors, INSERT_VALUES, _coors_local));
    PetscCall(DMGlobalToLocalEnd(_dmc, _coors, INSERT_VALUES, _coors_local));
  }

  PetscCall(VecScatterDestroy(&sctx));
  PetscCall(ISDestroy(&is_fine));
  PetscCall(PetscFree(fine_indices));
  PetscCall(ISDestroy(&is_local));
  PetscCall(VecDestroy(&perm_coors));
  PetscCall(VecDestroy(&coor_natural));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCTelescopeSetUp_dmda_repart_coors(PC pc, PC_Telescope sred, PC_Telescope_DMDACtx *ctx)
{
  PetscInt     dim;
  DM           dm, subdm;
  PetscSubcomm psubcomm;
  MPI_Comm     comm;
  Vec          coor;

  PetscFunctionBegin;
  PetscCall(PCGetDM(pc, &dm));
  PetscCall(DMGetCoordinates(dm, &coor));
  if (!coor) PetscFunctionReturn(PETSC_SUCCESS);
  psubcomm = sred->psubcomm;
  comm     = PetscSubcommParent(psubcomm);
  subdm    = ctx->dmrepart;

  PetscCall(PetscInfo(pc, "PCTelescope: setting up the coordinates (DMDA)\n"));
  PetscCall(DMDAGetInfo(dm, &dim, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
  switch (dim) {
  case 1:
    SETERRQ(comm, PETSC_ERR_SUP, "Telescope: DMDA (1D) repartitioning not provided");
  case 2:
    PetscCall(PCTelescopeSetUp_dmda_repart_coors2d(sred, dm, subdm));
    break;
  case 3:
    PetscCall(PCTelescopeSetUp_dmda_repart_coors3d(sred, dm, subdm));
    break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* setup repartitioned dm */
PetscErrorCode PCTelescopeSetUp_dmda_repart(PC pc, PC_Telescope sred, PC_Telescope_DMDACtx *ctx)
{
  DM                    dm;
  PetscInt              dim, nx, ny, nz, ndof, nsw, sum, k;
  DMBoundaryType        bx, by, bz;
  DMDAStencilType       stencil;
  const PetscInt       *_range_i_re;
  const PetscInt       *_range_j_re;
  const PetscInt       *_range_k_re;
  DMDAInterpolationType itype;
  PetscInt              refine_x, refine_y, refine_z;
  MPI_Comm              comm, subcomm;
  const char           *prefix;

  PetscFunctionBegin;
  comm    = PetscSubcommParent(sred->psubcomm);
  subcomm = PetscSubcommChild(sred->psubcomm);
  PetscCall(PCGetDM(pc, &dm));

  PetscCall(DMDAGetInfo(dm, &dim, &nx, &ny, &nz, NULL, NULL, NULL, &ndof, &nsw, &bx, &by, &bz, &stencil));
  PetscCall(DMDAGetInterpolationType(dm, &itype));
  PetscCall(DMDAGetRefinementFactor(dm, &refine_x, &refine_y, &refine_z));

  ctx->dmrepart = NULL;
  _range_i_re = _range_j_re = _range_k_re = NULL;
  /* Create DMDA on the child communicator */
  if (PCTelescope_isActiveRank(sred)) {
    switch (dim) {
    case 1:
      PetscCall(PetscInfo(pc, "PCTelescope: setting up the DMDA on comm subset (1D)\n"));
      /* PetscCall(DMDACreate1d(subcomm,bx,nx,ndof,nsw,NULL,&ctx->dmrepart)); */
      ny = nz = 1;
      by = bz = DM_BOUNDARY_NONE;
      break;
    case 2:
      PetscCall(PetscInfo(pc, "PCTelescope: setting up the DMDA on comm subset (2D)\n"));
      /* PetscCall(DMDACreate2d(subcomm,bx,by,stencil,nx,ny, PETSC_DECIDE,PETSC_DECIDE,
         ndof,nsw, NULL,NULL,&ctx->dmrepart)); */
      nz = 1;
      bz = DM_BOUNDARY_NONE;
      break;
    case 3:
      PetscCall(PetscInfo(pc, "PCTelescope: setting up the DMDA on comm subset (3D)\n"));
      /* PetscCall(DMDACreate3d(subcomm,bx,by,bz,stencil,nx,ny,nz,
         PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE, ndof,nsw, NULL,NULL,NULL,&ctx->dmrepart)); */
      break;
    }
    /*
     The API DMDACreate1d(), DMDACreate2d(), DMDACreate3d() does not allow us to set/append
     a unique option prefix for the DM, thus I prefer to expose the contents of these API's here.
     This allows users to control the partitioning of the subDM.
    */
    PetscCall(DMDACreate(subcomm, &ctx->dmrepart));
    /* Set unique option prefix name */
    PetscCall(KSPGetOptionsPrefix(sred->ksp, &prefix));
    PetscCall(DMSetOptionsPrefix(ctx->dmrepart, prefix));
    PetscCall(DMAppendOptionsPrefix(ctx->dmrepart, "repart_"));
    /* standard setup from DMDACreate{1,2,3}d() */
    PetscCall(DMSetDimension(ctx->dmrepart, dim));
    PetscCall(DMDASetSizes(ctx->dmrepart, nx, ny, nz));
    PetscCall(DMDASetNumProcs(ctx->dmrepart, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE));
    PetscCall(DMDASetBoundaryType(ctx->dmrepart, bx, by, bz));
    PetscCall(DMDASetDof(ctx->dmrepart, ndof));
    PetscCall(DMDASetStencilType(ctx->dmrepart, stencil));
    PetscCall(DMDASetStencilWidth(ctx->dmrepart, nsw));
    PetscCall(DMDASetOwnershipRanges(ctx->dmrepart, NULL, NULL, NULL));
    PetscCall(DMSetFromOptions(ctx->dmrepart));
    PetscCall(DMSetUp(ctx->dmrepart));
    /* Set refinement factors and interpolation type from the partent */
    PetscCall(DMDASetRefinementFactor(ctx->dmrepart, refine_x, refine_y, refine_z));
    PetscCall(DMDASetInterpolationType(ctx->dmrepart, itype));

    PetscCall(DMDAGetInfo(ctx->dmrepart, NULL, NULL, NULL, NULL, &ctx->Mp_re, &ctx->Np_re, &ctx->Pp_re, NULL, NULL, NULL, NULL, NULL, NULL));
    PetscCall(DMDAGetOwnershipRanges(ctx->dmrepart, &_range_i_re, &_range_j_re, &_range_k_re));

    ctx->dmrepart->ops->creatematrix              = dm->ops->creatematrix;
    ctx->dmrepart->ops->createdomaindecomposition = dm->ops->createdomaindecomposition;
  }

  /* generate ranges for repartitioned dm */
  /* note - assume rank 0 always participates */
  /* TODO: use a single MPI call */
  PetscCallMPI(MPI_Bcast(&ctx->Mp_re, 1, MPIU_INT, 0, comm));
  PetscCallMPI(MPI_Bcast(&ctx->Np_re, 1, MPIU_INT, 0, comm));
  PetscCallMPI(MPI_Bcast(&ctx->Pp_re, 1, MPIU_INT, 0, comm));

  PetscCall(PetscCalloc3(ctx->Mp_re, &ctx->range_i_re, ctx->Np_re, &ctx->range_j_re, ctx->Pp_re, &ctx->range_k_re));

  if (_range_i_re) PetscCall(PetscArraycpy(ctx->range_i_re, _range_i_re, ctx->Mp_re));
  if (_range_j_re) PetscCall(PetscArraycpy(ctx->range_j_re, _range_j_re, ctx->Np_re));
  if (_range_k_re) PetscCall(PetscArraycpy(ctx->range_k_re, _range_k_re, ctx->Pp_re));

  /* TODO: use a single MPI call */
  PetscCallMPI(MPI_Bcast(ctx->range_i_re, ctx->Mp_re, MPIU_INT, 0, comm));
  PetscCallMPI(MPI_Bcast(ctx->range_j_re, ctx->Np_re, MPIU_INT, 0, comm));
  PetscCallMPI(MPI_Bcast(ctx->range_k_re, ctx->Pp_re, MPIU_INT, 0, comm));

  PetscCall(PetscMalloc3(ctx->Mp_re, &ctx->start_i_re, ctx->Np_re, &ctx->start_j_re, ctx->Pp_re, &ctx->start_k_re));

  sum = 0;
  for (k = 0; k < ctx->Mp_re; k++) {
    ctx->start_i_re[k] = sum;
    sum += ctx->range_i_re[k];
  }

  sum = 0;
  for (k = 0; k < ctx->Np_re; k++) {
    ctx->start_j_re[k] = sum;
    sum += ctx->range_j_re[k];
  }

  sum = 0;
  for (k = 0; k < ctx->Pp_re; k++) {
    ctx->start_k_re[k] = sum;
    sum += ctx->range_k_re[k];
  }

  /* attach repartitioned dm to child ksp */
  {
    PetscErrorCode (*dmksp_func)(KSP, Mat, Mat, void *);
    void *dmksp_ctx;

    PetscCall(DMKSPGetComputeOperators(dm, &dmksp_func, &dmksp_ctx));

    /* attach dm to ksp on sub communicator */
    if (PCTelescope_isActiveRank(sred)) {
      PetscCall(KSPSetDM(sred->ksp, ctx->dmrepart));

      if (!dmksp_func || sred->ignore_kspcomputeoperators) {
        PetscCall(KSPSetDMActive(sred->ksp, PETSC_FALSE));
      } else {
        /* sub ksp inherits dmksp_func and context provided by user */
        PetscCall(KSPSetComputeOperators(sred->ksp, dmksp_func, dmksp_ctx));
        PetscCall(KSPSetDMActive(sred->ksp, PETSC_TRUE));
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCTelescopeSetUp_dmda_permutation_3d(PC pc, PC_Telescope sred, PC_Telescope_DMDACtx *ctx)
{
  DM       dm;
  MPI_Comm comm;
  Mat      Pscalar, P;
  PetscInt ndof;
  PetscInt i, j, k, location, startI[3], endI[3], lenI[3], nx, ny, nz;
  PetscInt sr, er, Mr;
  Vec      V;

  PetscFunctionBegin;
  PetscCall(PetscInfo(pc, "PCTelescope: setting up the permutation matrix (DMDA-3D)\n"));
  PetscCall(PetscObjectGetComm((PetscObject)pc, &comm));

  PetscCall(PCGetDM(pc, &dm));
  PetscCall(DMDAGetInfo(dm, NULL, &nx, &ny, &nz, NULL, NULL, NULL, &ndof, NULL, NULL, NULL, NULL, NULL));

  PetscCall(DMGetGlobalVector(dm, &V));
  PetscCall(VecGetSize(V, &Mr));
  PetscCall(VecGetOwnershipRange(V, &sr, &er));
  PetscCall(DMRestoreGlobalVector(dm, &V));
  sr = sr / ndof;
  er = er / ndof;
  Mr = Mr / ndof;

  PetscCall(MatCreate(comm, &Pscalar));
  PetscCall(MatSetSizes(Pscalar, (er - sr), (er - sr), Mr, Mr));
  PetscCall(MatSetType(Pscalar, MATAIJ));
  PetscCall(MatSeqAIJSetPreallocation(Pscalar, 1, NULL));
  PetscCall(MatMPIAIJSetPreallocation(Pscalar, 1, NULL, 1, NULL));

  PetscCall(DMDAGetCorners(dm, NULL, NULL, NULL, &lenI[0], &lenI[1], &lenI[2]));
  PetscCall(DMDAGetCorners(dm, &startI[0], &startI[1], &startI[2], &endI[0], &endI[1], &endI[2]));
  endI[0] += startI[0];
  endI[1] += startI[1];
  endI[2] += startI[2];

  for (k = startI[2]; k < endI[2]; k++) {
    for (j = startI[1]; j < endI[1]; j++) {
      for (i = startI[0]; i < endI[0]; i++) {
        PetscMPIInt rank_ijk_re, rank_reI[3];
        PetscInt    s0_re;
        PetscInt    ii, jj, kk, local_ijk_re, mapped_ijk;
        PetscInt    lenI_re[3];

        location = (i - startI[0]) + (j - startI[1]) * lenI[0] + (k - startI[2]) * lenI[0] * lenI[1];
        PetscCall(_DMDADetermineRankFromGlobalIJK(3, i, j, k, ctx->Mp_re, ctx->Np_re, ctx->Pp_re, ctx->start_i_re, ctx->start_j_re, ctx->start_k_re, ctx->range_i_re, ctx->range_j_re, ctx->range_k_re, &rank_reI[0], &rank_reI[1], &rank_reI[2], &rank_ijk_re));
        PetscCall(_DMDADetermineGlobalS0(3, rank_ijk_re, ctx->Mp_re, ctx->Np_re, ctx->Pp_re, ctx->range_i_re, ctx->range_j_re, ctx->range_k_re, &s0_re));
        ii = i - ctx->start_i_re[rank_reI[0]];
        PetscCheck(ii >= 0, PETSC_COMM_SELF, PETSC_ERR_USER, "[dmdarepart-perm3d] index error ii");
        jj = j - ctx->start_j_re[rank_reI[1]];
        PetscCheck(jj >= 0, PETSC_COMM_SELF, PETSC_ERR_USER, "[dmdarepart-perm3d] index error jj");
        kk = k - ctx->start_k_re[rank_reI[2]];
        PetscCheck(kk >= 0, PETSC_COMM_SELF, PETSC_ERR_USER, "[dmdarepart-perm3d] index error kk");
        lenI_re[0]   = ctx->range_i_re[rank_reI[0]];
        lenI_re[1]   = ctx->range_j_re[rank_reI[1]];
        lenI_re[2]   = ctx->range_k_re[rank_reI[2]];
        local_ijk_re = ii + jj * lenI_re[0] + kk * lenI_re[0] * lenI_re[1];
        mapped_ijk   = s0_re + local_ijk_re;
        PetscCall(MatSetValue(Pscalar, sr + location, mapped_ijk, 1.0, INSERT_VALUES));
      }
    }
  }
  PetscCall(MatAssemblyBegin(Pscalar, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Pscalar, MAT_FINAL_ASSEMBLY));
  PetscCall(MatCreateMAIJ(Pscalar, ndof, &P));
  PetscCall(MatDestroy(&Pscalar));
  ctx->permutation = P;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCTelescopeSetUp_dmda_permutation_2d(PC pc, PC_Telescope sred, PC_Telescope_DMDACtx *ctx)
{
  DM       dm;
  MPI_Comm comm;
  Mat      Pscalar, P;
  PetscInt ndof;
  PetscInt i, j, location, startI[2], endI[2], lenI[2], nx, ny, nz;
  PetscInt sr, er, Mr;
  Vec      V;

  PetscFunctionBegin;
  PetscCall(PetscInfo(pc, "PCTelescope: setting up the permutation matrix (DMDA-2D)\n"));
  PetscCall(PetscObjectGetComm((PetscObject)pc, &comm));
  PetscCall(PCGetDM(pc, &dm));
  PetscCall(DMDAGetInfo(dm, NULL, &nx, &ny, &nz, NULL, NULL, NULL, &ndof, NULL, NULL, NULL, NULL, NULL));
  PetscCall(DMGetGlobalVector(dm, &V));
  PetscCall(VecGetSize(V, &Mr));
  PetscCall(VecGetOwnershipRange(V, &sr, &er));
  PetscCall(DMRestoreGlobalVector(dm, &V));
  sr = sr / ndof;
  er = er / ndof;
  Mr = Mr / ndof;

  PetscCall(MatCreate(comm, &Pscalar));
  PetscCall(MatSetSizes(Pscalar, (er - sr), (er - sr), Mr, Mr));
  PetscCall(MatSetType(Pscalar, MATAIJ));
  PetscCall(MatSeqAIJSetPreallocation(Pscalar, 1, NULL));
  PetscCall(MatMPIAIJSetPreallocation(Pscalar, 1, NULL, 1, NULL));

  PetscCall(DMDAGetCorners(dm, NULL, NULL, NULL, &lenI[0], &lenI[1], NULL));
  PetscCall(DMDAGetCorners(dm, &startI[0], &startI[1], NULL, &endI[0], &endI[1], NULL));
  endI[0] += startI[0];
  endI[1] += startI[1];

  for (j = startI[1]; j < endI[1]; j++) {
    for (i = startI[0]; i < endI[0]; i++) {
      PetscMPIInt rank_ijk_re, rank_reI[3];
      PetscInt    s0_re;
      PetscInt    ii, jj, local_ijk_re, mapped_ijk;
      PetscInt    lenI_re[3];

      location = (i - startI[0]) + (j - startI[1]) * lenI[0];
      PetscCall(_DMDADetermineRankFromGlobalIJK(2, i, j, 0, ctx->Mp_re, ctx->Np_re, ctx->Pp_re, ctx->start_i_re, ctx->start_j_re, ctx->start_k_re, ctx->range_i_re, ctx->range_j_re, ctx->range_k_re, &rank_reI[0], &rank_reI[1], NULL, &rank_ijk_re));

      PetscCall(_DMDADetermineGlobalS0(2, rank_ijk_re, ctx->Mp_re, ctx->Np_re, ctx->Pp_re, ctx->range_i_re, ctx->range_j_re, ctx->range_k_re, &s0_re));

      ii = i - ctx->start_i_re[rank_reI[0]];
      PetscCheck(ii >= 0, PETSC_COMM_SELF, PETSC_ERR_USER, "[dmdarepart-perm2d] index error ii");
      jj = j - ctx->start_j_re[rank_reI[1]];
      PetscCheck(jj >= 0, PETSC_COMM_SELF, PETSC_ERR_USER, "[dmdarepart-perm2d] index error jj");

      lenI_re[0]   = ctx->range_i_re[rank_reI[0]];
      lenI_re[1]   = ctx->range_j_re[rank_reI[1]];
      local_ijk_re = ii + jj * lenI_re[0];
      mapped_ijk   = s0_re + local_ijk_re;
      PetscCall(MatSetValue(Pscalar, sr + location, mapped_ijk, 1.0, INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(Pscalar, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Pscalar, MAT_FINAL_ASSEMBLY));
  PetscCall(MatCreateMAIJ(Pscalar, ndof, &P));
  PetscCall(MatDestroy(&Pscalar));
  ctx->permutation = P;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCTelescopeSetUp_dmda_scatters(PC pc, PC_Telescope sred, PC_Telescope_DMDACtx *ctx)
{
  Vec        xred, yred, xtmp, x, xp;
  VecScatter scatter;
  IS         isin;
  Mat        B;
  PetscInt   m, bs, st, ed;
  MPI_Comm   comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)pc, &comm));
  PetscCall(PCGetOperators(pc, NULL, &B));
  PetscCall(MatCreateVecs(B, &x, NULL));
  PetscCall(MatGetBlockSize(B, &bs));
  PetscCall(VecDuplicate(x, &xp));
  m    = 0;
  xred = NULL;
  yred = NULL;
  if (PCTelescope_isActiveRank(sred)) {
    PetscCall(DMCreateGlobalVector(ctx->dmrepart, &xred));
    PetscCall(VecDuplicate(xred, &yred));
    PetscCall(VecGetOwnershipRange(xred, &st, &ed));
    PetscCall(ISCreateStride(comm, ed - st, st, 1, &isin));
    PetscCall(VecGetLocalSize(xred, &m));
  } else {
    PetscCall(VecGetOwnershipRange(x, &st, &ed));
    PetscCall(ISCreateStride(comm, 0, st, 1, &isin));
  }
  PetscCall(ISSetBlockSize(isin, bs));
  PetscCall(VecCreate(comm, &xtmp));
  PetscCall(VecSetSizes(xtmp, m, PETSC_DECIDE));
  PetscCall(VecSetBlockSize(xtmp, bs));
  PetscCall(VecSetType(xtmp, ((PetscObject)x)->type_name));
  PetscCall(VecScatterCreate(x, isin, xtmp, NULL, &scatter));
  sred->xred    = xred;
  sred->yred    = yred;
  sred->isin    = isin;
  sred->scatter = scatter;
  sred->xtmp    = xtmp;

  ctx->xp = xp;
  PetscCall(VecDestroy(&x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCTelescopeSetUp_dmda(PC pc, PC_Telescope sred)
{
  PC_Telescope_DMDACtx *ctx;
  PetscInt              dim;
  DM                    dm;
  MPI_Comm              comm;

  PetscFunctionBegin;
  PetscCall(PetscInfo(pc, "PCTelescope: setup (DMDA)\n"));
  PetscCall(PetscNew(&ctx));
  sred->dm_ctx = (void *)ctx;

  PetscCall(PetscObjectGetComm((PetscObject)pc, &comm));
  PetscCall(PCGetDM(pc, &dm));
  PetscCall(DMDAGetInfo(dm, &dim, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));

  PetscCall(PCTelescopeSetUp_dmda_repart(pc, sred, ctx));
  PetscCall(PCTelescopeSetUp_dmda_repart_coors(pc, sred, ctx));
  switch (dim) {
  case 1:
    SETERRQ(comm, PETSC_ERR_SUP, "Telescope: DMDA (1D) repartitioning not provided");
  case 2:
    PetscCall(PCTelescopeSetUp_dmda_permutation_2d(pc, sred, ctx));
    break;
  case 3:
    PetscCall(PCTelescopeSetUp_dmda_permutation_3d(pc, sred, ctx));
    break;
  }
  PetscCall(PCTelescopeSetUp_dmda_scatters(pc, sred, ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCTelescopeMatCreate_dmda_dmactivefalse(PC pc, PC_Telescope sred, MatReuse reuse, Mat *A)
{
  PC_Telescope_DMDACtx *ctx;
  MPI_Comm              comm, subcomm;
  Mat                   Bperm, Bred, B, P;
  PetscInt              nr, nc;
  IS                    isrow, iscol;
  Mat                   Blocal, *_Blocal;

  PetscFunctionBegin;
  PetscCall(PetscInfo(pc, "PCTelescope: updating the redundant preconditioned operator (DMDA)\n"));
  PetscCall(PetscObjectGetComm((PetscObject)pc, &comm));
  subcomm = PetscSubcommChild(sred->psubcomm);
  ctx     = (PC_Telescope_DMDACtx *)sred->dm_ctx;

  PetscCall(PCGetOperators(pc, NULL, &B));
  PetscCall(MatGetSize(B, &nr, &nc));

  P = ctx->permutation;
  PetscCall(MatPtAP(B, P, MAT_INITIAL_MATRIX, 1.1, &Bperm));

  /* Get submatrices */
  isrow = sred->isin;
  PetscCall(ISCreateStride(comm, nc, 0, 1, &iscol));

  PetscCall(MatCreateSubMatrices(Bperm, 1, &isrow, &iscol, MAT_INITIAL_MATRIX, &_Blocal));
  Blocal = *_Blocal;
  Bred   = NULL;
  if (PCTelescope_isActiveRank(sred)) {
    PetscInt mm;

    if (reuse != MAT_INITIAL_MATRIX) Bred = *A;
    PetscCall(MatGetSize(Blocal, &mm, NULL));
    /* PetscCall(MatCreateMPIMatConcatenateSeqMat(subcomm,Blocal,PETSC_DECIDE,reuse,&Bred)); */
    PetscCall(MatCreateMPIMatConcatenateSeqMat(subcomm, Blocal, mm, reuse, &Bred));
  }
  *A = Bred;

  PetscCall(ISDestroy(&iscol));
  PetscCall(MatDestroy(&Bperm));
  PetscCall(MatDestroyMatrices(1, &_Blocal));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCTelescopeMatCreate_dmda(PC pc, PC_Telescope sred, MatReuse reuse, Mat *A)
{
  DM dm;
  PetscErrorCode (*dmksp_func)(KSP, Mat, Mat, void *);
  void *dmksp_ctx;

  PetscFunctionBegin;
  PetscCall(PCGetDM(pc, &dm));
  PetscCall(DMKSPGetComputeOperators(dm, &dmksp_func, &dmksp_ctx));
  /* We assume that dmksp_func = NULL, is equivalent to dmActive = PETSC_FALSE */
  if (dmksp_func && !sred->ignore_kspcomputeoperators) {
    DM  dmrepart;
    Mat Ak;

    *A = NULL;
    if (PCTelescope_isActiveRank(sred)) {
      PetscCall(KSPGetDM(sred->ksp, &dmrepart));
      if (reuse == MAT_INITIAL_MATRIX) {
        PetscCall(DMCreateMatrix(dmrepart, &Ak));
        *A = Ak;
      } else if (reuse == MAT_REUSE_MATRIX) {
        Ak = *A;
      }
      /*
       There is no need to explicitly assemble the operator now,
       the sub-KSP will call the method provided to KSPSetComputeOperators() during KSPSetUp()
      */
    }
  } else {
    PetscCall(PCTelescopeMatCreate_dmda_dmactivefalse(pc, sred, reuse, A));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCTelescopeSubNullSpaceCreate_dmda_Telescope(PC pc, PC_Telescope sred, MatNullSpace nullspace, MatNullSpace *sub_nullspace)
{
  PetscBool             has_const;
  PetscInt              i, k, n = 0;
  const Vec            *vecs;
  Vec                  *sub_vecs = NULL;
  MPI_Comm              subcomm;
  PC_Telescope_DMDACtx *ctx;

  PetscFunctionBegin;
  ctx     = (PC_Telescope_DMDACtx *)sred->dm_ctx;
  subcomm = PetscSubcommChild(sred->psubcomm);
  PetscCall(MatNullSpaceGetVecs(nullspace, &has_const, &n, &vecs));

  if (PCTelescope_isActiveRank(sred)) {
    /* create new vectors */
    if (n) PetscCall(VecDuplicateVecs(sred->xred, n, &sub_vecs));
  }

  /* copy entries */
  for (k = 0; k < n; k++) {
    const PetscScalar *x_array;
    PetscScalar       *LA_sub_vec;
    PetscInt           st, ed;

    /* permute vector into ordering associated with re-partitioned dmda */
    PetscCall(MatMultTranspose(ctx->permutation, vecs[k], ctx->xp));

    /* pull in vector x->xtmp */
    PetscCall(VecScatterBegin(sred->scatter, ctx->xp, sred->xtmp, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(sred->scatter, ctx->xp, sred->xtmp, INSERT_VALUES, SCATTER_FORWARD));

    /* copy vector entries into xred */
    PetscCall(VecGetArrayRead(sred->xtmp, &x_array));
    if (sub_vecs) {
      if (sub_vecs[k]) {
        PetscCall(VecGetOwnershipRange(sub_vecs[k], &st, &ed));
        PetscCall(VecGetArray(sub_vecs[k], &LA_sub_vec));
        for (i = 0; i < ed - st; i++) LA_sub_vec[i] = x_array[i];
        PetscCall(VecRestoreArray(sub_vecs[k], &LA_sub_vec));
      }
    }
    PetscCall(VecRestoreArrayRead(sred->xtmp, &x_array));
  }

  if (PCTelescope_isActiveRank(sred)) {
    /* create new (near) nullspace for redundant object */
    PetscCall(MatNullSpaceCreate(subcomm, has_const, n, sub_vecs, sub_nullspace));
    PetscCall(VecDestroyVecs(n, &sub_vecs));
    PetscCheck(!nullspace->remove, PetscObjectComm((PetscObject)pc), PETSC_ERR_SUP, "Propagation of custom remove callbacks not supported when propagating (near) nullspaces with PCTelescope");
    PetscCheck(!nullspace->rmctx, PetscObjectComm((PetscObject)pc), PETSC_ERR_SUP, "Propagation of custom remove callback context not supported when propagating (near) nullspaces with PCTelescope");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCTelescopeMatNullSpaceCreate_dmda(PC pc, PC_Telescope sred, Mat sub_mat)
{
  Mat B;

  PetscFunctionBegin;
  PetscCall(PCGetOperators(pc, NULL, &B));
  {
    MatNullSpace nullspace, sub_nullspace;
    PetscCall(MatGetNullSpace(B, &nullspace));
    if (nullspace) {
      PetscCall(PetscInfo(pc, "PCTelescope: generating nullspace (DMDA)\n"));
      PetscCall(PCTelescopeSubNullSpaceCreate_dmda_Telescope(pc, sred, nullspace, &sub_nullspace));
      if (PCTelescope_isActiveRank(sred)) {
        PetscCall(MatSetNullSpace(sub_mat, sub_nullspace));
        PetscCall(MatNullSpaceDestroy(&sub_nullspace));
      }
    }
  }
  {
    MatNullSpace nearnullspace, sub_nearnullspace;
    PetscCall(MatGetNearNullSpace(B, &nearnullspace));
    if (nearnullspace) {
      PetscCall(PetscInfo(pc, "PCTelescope: generating near nullspace (DMDA)\n"));
      PetscCall(PCTelescopeSubNullSpaceCreate_dmda_Telescope(pc, sred, nearnullspace, &sub_nearnullspace));
      if (PCTelescope_isActiveRank(sred)) {
        PetscCall(MatSetNearNullSpace(sub_mat, sub_nearnullspace));
        PetscCall(MatNullSpaceDestroy(&sub_nearnullspace));
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCApply_Telescope_dmda(PC pc, Vec x, Vec y)
{
  PC_Telescope          sred = (PC_Telescope)pc->data;
  Mat                   perm;
  Vec                   xtmp, xp, xred, yred;
  PetscInt              i, st, ed;
  VecScatter            scatter;
  PetscScalar          *array;
  const PetscScalar    *x_array;
  PC_Telescope_DMDACtx *ctx;

  ctx     = (PC_Telescope_DMDACtx *)sred->dm_ctx;
  xtmp    = sred->xtmp;
  scatter = sred->scatter;
  xred    = sred->xred;
  yred    = sred->yred;
  perm    = ctx->permutation;
  xp      = ctx->xp;

  PetscFunctionBegin;
  PetscCall(PetscCitationsRegister(citation, &cited));

  /* permute vector into ordering associated with re-partitioned dmda */
  PetscCall(MatMultTranspose(perm, x, xp));

  /* pull in vector x->xtmp */
  PetscCall(VecScatterBegin(scatter, xp, xtmp, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(scatter, xp, xtmp, INSERT_VALUES, SCATTER_FORWARD));

  /* copy vector entries into xred */
  PetscCall(VecGetArrayRead(xtmp, &x_array));
  if (xred) {
    PetscScalar *LA_xred;
    PetscCall(VecGetOwnershipRange(xred, &st, &ed));

    PetscCall(VecGetArray(xred, &LA_xred));
    for (i = 0; i < ed - st; i++) LA_xred[i] = x_array[i];
    PetscCall(VecRestoreArray(xred, &LA_xred));
  }
  PetscCall(VecRestoreArrayRead(xtmp, &x_array));

  /* solve */
  if (PCTelescope_isActiveRank(sred)) {
    PetscCall(KSPSolve(sred->ksp, xred, yred));
    PetscCall(KSPCheckSolve(sred->ksp, pc, yred));
  }

  /* return vector */
  PetscCall(VecGetArray(xtmp, &array));
  if (yred) {
    const PetscScalar *LA_yred;
    PetscCall(VecGetOwnershipRange(yred, &st, &ed));
    PetscCall(VecGetArrayRead(yred, &LA_yred));
    for (i = 0; i < ed - st; i++) array[i] = LA_yred[i];
    PetscCall(VecRestoreArrayRead(yred, &LA_yred));
  }
  PetscCall(VecRestoreArray(xtmp, &array));
  PetscCall(VecScatterBegin(scatter, xtmp, xp, INSERT_VALUES, SCATTER_REVERSE));
  PetscCall(VecScatterEnd(scatter, xtmp, xp, INSERT_VALUES, SCATTER_REVERSE));
  PetscCall(MatMult(perm, xp, y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCApplyRichardson_Telescope_dmda(PC pc, Vec x, Vec y, Vec w, PetscReal rtol, PetscReal abstol, PetscReal dtol, PetscInt its, PetscBool zeroguess, PetscInt *outits, PCRichardsonConvergedReason *reason)
{
  PC_Telescope          sred = (PC_Telescope)pc->data;
  Mat                   perm;
  Vec                   xtmp, xp, yred;
  PetscInt              i, st, ed;
  VecScatter            scatter;
  const PetscScalar    *x_array;
  PetscBool             default_init_guess_value = PETSC_FALSE;
  PC_Telescope_DMDACtx *ctx;

  PetscFunctionBegin;
  ctx     = (PC_Telescope_DMDACtx *)sred->dm_ctx;
  xtmp    = sred->xtmp;
  scatter = sred->scatter;
  yred    = sred->yred;
  perm    = ctx->permutation;
  xp      = ctx->xp;

  PetscCheck(its <= 1, PetscObjectComm((PetscObject)pc), PETSC_ERR_SUP, "PCApplyRichardson_Telescope_dmda only supports max_it = 1");
  *reason = (PCRichardsonConvergedReason)0;

  if (!zeroguess) {
    PetscCall(PetscInfo(pc, "PCTelescopeDMDA: Scattering y for non-zero-initial guess\n"));
    /* permute vector into ordering associated with re-partitioned dmda */
    PetscCall(MatMultTranspose(perm, y, xp));

    /* pull in vector x->xtmp */
    PetscCall(VecScatterBegin(scatter, xp, xtmp, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(scatter, xp, xtmp, INSERT_VALUES, SCATTER_FORWARD));

    /* copy vector entries into xred */
    PetscCall(VecGetArrayRead(xtmp, &x_array));
    if (yred) {
      PetscScalar *LA_yred;
      PetscCall(VecGetOwnershipRange(yred, &st, &ed));
      PetscCall(VecGetArray(yred, &LA_yred));
      for (i = 0; i < ed - st; i++) LA_yred[i] = x_array[i];
      PetscCall(VecRestoreArray(yred, &LA_yred));
    }
    PetscCall(VecRestoreArrayRead(xtmp, &x_array));
  }

  if (PCTelescope_isActiveRank(sred)) {
    PetscCall(KSPGetInitialGuessNonzero(sred->ksp, &default_init_guess_value));
    if (!zeroguess) PetscCall(KSPSetInitialGuessNonzero(sred->ksp, PETSC_TRUE));
  }

  PetscCall(PCApply_Telescope_dmda(pc, x, y));

  if (PCTelescope_isActiveRank(sred)) PetscCall(KSPSetInitialGuessNonzero(sred->ksp, default_init_guess_value));

  if (!*reason) *reason = PCRICHARDSON_CONVERGED_ITS;
  *outits = 1;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCReset_Telescope_dmda(PC pc)
{
  PC_Telescope          sred = (PC_Telescope)pc->data;
  PC_Telescope_DMDACtx *ctx;

  PetscFunctionBegin;
  ctx = (PC_Telescope_DMDACtx *)sred->dm_ctx;
  PetscCall(VecDestroy(&ctx->xp));
  PetscCall(MatDestroy(&ctx->permutation));
  PetscCall(DMDestroy(&ctx->dmrepart));
  PetscCall(PetscFree3(ctx->range_i_re, ctx->range_j_re, ctx->range_k_re));
  PetscCall(PetscFree3(ctx->start_i_re, ctx->start_j_re, ctx->start_k_re));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMView_DA_Short_3d(DM dm, PetscViewer v)
{
  PetscInt    M, N, P, m, n, p, ndof, nsw;
  MPI_Comm    comm;
  PetscMPIInt size;
  const char *prefix;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCall(DMGetOptionsPrefix(dm, &prefix));
  PetscCall(DMDAGetInfo(dm, NULL, &M, &N, &P, &m, &n, &p, &ndof, &nsw, NULL, NULL, NULL, NULL));
  if (prefix) PetscCall(PetscViewerASCIIPrintf(v, "DMDA Object:    (%s)    %d MPI processes\n", prefix, size));
  else PetscCall(PetscViewerASCIIPrintf(v, "DMDA Object:    %d MPI processes\n", size));
  PetscCall(PetscViewerASCIIPrintf(v, "  M %" PetscInt_FMT " N %" PetscInt_FMT " P %" PetscInt_FMT " m %" PetscInt_FMT " n %" PetscInt_FMT " p %" PetscInt_FMT " dof %" PetscInt_FMT " overlap %" PetscInt_FMT "\n", M, N, P, m, n, p, ndof, nsw));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMView_DA_Short_2d(DM dm, PetscViewer v)
{
  PetscInt    M, N, m, n, ndof, nsw;
  MPI_Comm    comm;
  PetscMPIInt size;
  const char *prefix;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCall(DMGetOptionsPrefix(dm, &prefix));
  PetscCall(DMDAGetInfo(dm, NULL, &M, &N, NULL, &m, &n, NULL, &ndof, &nsw, NULL, NULL, NULL, NULL));
  if (prefix) PetscCall(PetscViewerASCIIPrintf(v, "DMDA Object:    (%s)    %d MPI processes\n", prefix, size));
  else PetscCall(PetscViewerASCIIPrintf(v, "DMDA Object:    %d MPI processes\n", size));
  PetscCall(PetscViewerASCIIPrintf(v, "  M %" PetscInt_FMT " N %" PetscInt_FMT " m %" PetscInt_FMT " n %" PetscInt_FMT " dof %" PetscInt_FMT " overlap %" PetscInt_FMT "\n", M, N, m, n, ndof, nsw));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMView_DA_Short(DM dm, PetscViewer v)
{
  PetscInt dim;

  PetscFunctionBegin;
  PetscCall(DMDAGetInfo(dm, &dim, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
  switch (dim) {
  case 2:
    PetscCall(DMView_DA_Short_2d(dm, v));
    break;
  case 3:
    PetscCall(DMView_DA_Short_3d(dm, v));
    break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
