
#include <petsc/private/matimpl.h>
#include <petsc/private/pcimpl.h>
#include <petsc/private/dmimpl.h>
#include <petscksp.h>           /*I "petscksp.h" I*/
#include <petscdm.h>
#include <petscdmda.h>

#include "../src/ksp/pc/impls/telescope/telescope.h"

static PetscBool  cited = PETSC_FALSE;
static const char citation[] =
"@inproceedings{MaySananRuppKnepleySmith2016,\n"
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

static PetscErrorCode _DMDADetermineRankFromGlobalIJK(PetscInt dim,PetscInt i,PetscInt j,PetscInt k,
                                                      PetscInt Mp,PetscInt Np,PetscInt Pp,
                                                      PetscInt start_i[],PetscInt start_j[],PetscInt start_k[],
                                                      PetscInt span_i[],PetscInt span_j[],PetscInt span_k[],
                                                      PetscMPIInt *_pi,PetscMPIInt *_pj,PetscMPIInt *_pk,PetscMPIInt *rank_re)
{
  PetscInt pi,pj,pk,n;

  PetscFunctionBegin;
  *rank_re = -1;
  if (_pi) *_pi = -1;
  if (_pj) *_pj = -1;
  if (_pk) *_pk = -1;
  pi = pj = pk = -1;
  if (_pi) {
    for (n=0; n<Mp; n++) {
      if ((i >= start_i[n]) && (i < start_i[n]+span_i[n])) {
        pi = n;
        break;
      }
    }
    PetscCheckFalse(pi == -1,PETSC_COMM_SELF,PETSC_ERR_USER,"[dmda-ijk] pi cannot be determined : range %D, val %D",Mp,i);
    *_pi = pi;
  }

  if (_pj) {
    for (n=0; n<Np; n++) {
      if ((j >= start_j[n]) && (j < start_j[n]+span_j[n])) {
        pj = n;
        break;
      }
    }
    PetscCheckFalse(pj == -1,PETSC_COMM_SELF,PETSC_ERR_USER,"[dmda-ijk] pj cannot be determined : range %D, val %D",Np,j);
    *_pj = pj;
  }

  if (_pk) {
    for (n=0; n<Pp; n++) {
      if ((k >= start_k[n]) && (k < start_k[n]+span_k[n])) {
        pk = n;
        break;
      }
    }
    PetscCheckFalse(pk == -1,PETSC_COMM_SELF,PETSC_ERR_USER,"[dmda-ijk] pk cannot be determined : range %D, val %D",Pp,k);
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
    *rank_re = pi + pj * Mp + pk * (Mp*Np);
    break;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode _DMDADetermineGlobalS0(PetscInt dim,PetscMPIInt rank_re,PetscInt Mp_re,PetscInt Np_re,PetscInt Pp_re,
                                      PetscInt range_i_re[],PetscInt range_j_re[],PetscInt range_k_re[],PetscInt *s0)
{
  PetscInt i,j,k,start_IJK = 0;
  PetscInt rank_ijk;

  PetscFunctionBegin;
  switch (dim) {
  case 1:
    for (i=0; i<Mp_re; i++) {
      rank_ijk = i;
      if (rank_ijk < rank_re) {
        start_IJK += range_i_re[i];
      }
    }
    break;
    case 2:
    for (j=0; j<Np_re; j++) {
      for (i=0; i<Mp_re; i++) {
        rank_ijk = i + j*Mp_re;
        if (rank_ijk < rank_re) {
          start_IJK += range_i_re[i]*range_j_re[j];
        }
      }
    }
    break;
    case 3:
    for (k=0; k<Pp_re; k++) {
      for (j=0; j<Np_re; j++) {
        for (i=0; i<Mp_re; i++) {
          rank_ijk = i + j*Mp_re + k*Mp_re*Np_re;
          if (rank_ijk < rank_re) {
            start_IJK += range_i_re[i]*range_j_re[j]*range_k_re[k];
          }
        }
      }
    }
    break;
  }
  *s0 = start_IJK;
  PetscFunctionReturn(0);
}

static PetscErrorCode PCTelescopeSetUp_dmda_repart_coors2d(PC_Telescope sred,DM dm,DM subdm)
{
  DM             cdm;
  Vec            coor,coor_natural,perm_coors;
  PetscInt       i,j,si,sj,ni,nj,M,N,Ml,Nl,c,nidx;
  PetscInt       *fine_indices;
  IS             is_fine,is_local;
  VecScatter     sctx;

  PetscFunctionBegin;
  CHKERRQ(DMGetCoordinates(dm,&coor));
  if (!coor) return(0);
  if (PCTelescope_isActiveRank(sred)) {
    CHKERRQ(DMDASetUniformCoordinates(subdm,0.0,1.0,0.0,1.0,0.0,1.0));
  }
  /* Get the coordinate vector from the distributed array */
  CHKERRQ(DMGetCoordinateDM(dm,&cdm));
  CHKERRQ(DMDACreateNaturalVector(cdm,&coor_natural));

  CHKERRQ(DMDAGlobalToNaturalBegin(cdm,coor,INSERT_VALUES,coor_natural));
  CHKERRQ(DMDAGlobalToNaturalEnd(cdm,coor,INSERT_VALUES,coor_natural));

  /* get indices of the guys I want to grab */
  CHKERRQ(DMDAGetInfo(dm,NULL,&M,&N,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL));
  if (PCTelescope_isActiveRank(sred)) {
    CHKERRQ(DMDAGetCorners(subdm,&si,&sj,NULL,&ni,&nj,NULL));
    Ml = ni;
    Nl = nj;
  } else {
    si = sj = 0;
    ni = nj = 0;
    Ml = Nl = 0;
  }

  CHKERRQ(PetscMalloc1(Ml*Nl*2,&fine_indices));
  c = 0;
  if (PCTelescope_isActiveRank(sred)) {
    for (j=sj; j<sj+nj; j++) {
      for (i=si; i<si+ni; i++) {
        nidx = (i) + (j)*M;
        fine_indices[c  ] = 2 * nidx     ;
        fine_indices[c+1] = 2 * nidx + 1 ;
        c = c + 2;
      }
    }
    PetscCheckFalse(c != Ml*Nl*2,PETSC_COMM_SELF,PETSC_ERR_PLIB,"c %D should equal 2 * Ml %D * Nl %D",c,Ml,Nl);
  }

  /* generate scatter */
  CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)dm),Ml*Nl*2,fine_indices,PETSC_USE_POINTER,&is_fine));
  CHKERRQ(ISCreateStride(PETSC_COMM_SELF,Ml*Nl*2,0,1,&is_local));

  /* scatter */
  CHKERRQ(VecCreate(PETSC_COMM_SELF,&perm_coors));
  CHKERRQ(VecSetSizes(perm_coors,PETSC_DECIDE,Ml*Nl*2));
  CHKERRQ(VecSetType(perm_coors,VECSEQ));

  CHKERRQ(VecScatterCreate(coor_natural,is_fine,perm_coors,is_local,&sctx));
  CHKERRQ(VecScatterBegin(sctx,coor_natural,perm_coors,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(  sctx,coor_natural,perm_coors,INSERT_VALUES,SCATTER_FORWARD));
  /* access */
  if (PCTelescope_isActiveRank(sred)) {
    Vec               _coors;
    const PetscScalar *LA_perm;
    PetscScalar       *LA_coors;

    CHKERRQ(DMGetCoordinates(subdm,&_coors));
    CHKERRQ(VecGetArrayRead(perm_coors,&LA_perm));
    CHKERRQ(VecGetArray(_coors,&LA_coors));
    for (i=0; i<Ml*Nl*2; i++) {
      LA_coors[i] = LA_perm[i];
    }
    CHKERRQ(VecRestoreArray(_coors,&LA_coors));
    CHKERRQ(VecRestoreArrayRead(perm_coors,&LA_perm));
  }

  /* update local coords */
  if (PCTelescope_isActiveRank(sred)) {
    DM  _dmc;
    Vec _coors,_coors_local;
    CHKERRQ(DMGetCoordinateDM(subdm,&_dmc));
    CHKERRQ(DMGetCoordinates(subdm,&_coors));
    CHKERRQ(DMGetCoordinatesLocal(subdm,&_coors_local));
    CHKERRQ(DMGlobalToLocalBegin(_dmc,_coors,INSERT_VALUES,_coors_local));
    CHKERRQ(DMGlobalToLocalEnd(_dmc,_coors,INSERT_VALUES,_coors_local));
  }
  CHKERRQ(VecScatterDestroy(&sctx));
  CHKERRQ(ISDestroy(&is_fine));
  CHKERRQ(PetscFree(fine_indices));
  CHKERRQ(ISDestroy(&is_local));
  CHKERRQ(VecDestroy(&perm_coors));
  CHKERRQ(VecDestroy(&coor_natural));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCTelescopeSetUp_dmda_repart_coors3d(PC_Telescope sred,DM dm,DM subdm)
{
  DM             cdm;
  Vec            coor,coor_natural,perm_coors;
  PetscInt       i,j,k,si,sj,sk,ni,nj,nk,M,N,P,Ml,Nl,Pl,c,nidx;
  PetscInt       *fine_indices;
  IS             is_fine,is_local;
  VecScatter     sctx;

  PetscFunctionBegin;
  CHKERRQ(DMGetCoordinates(dm,&coor));
  if (!coor) PetscFunctionReturn(0);

  if (PCTelescope_isActiveRank(sred)) {
    CHKERRQ(DMDASetUniformCoordinates(subdm,0.0,1.0,0.0,1.0,0.0,1.0));
  }

  /* Get the coordinate vector from the distributed array */
  CHKERRQ(DMGetCoordinateDM(dm,&cdm));
  CHKERRQ(DMDACreateNaturalVector(cdm,&coor_natural));
  CHKERRQ(DMDAGlobalToNaturalBegin(cdm,coor,INSERT_VALUES,coor_natural));
  CHKERRQ(DMDAGlobalToNaturalEnd(cdm,coor,INSERT_VALUES,coor_natural));

  /* get indices of the guys I want to grab */
  CHKERRQ(DMDAGetInfo(dm,NULL,&M,&N,&P,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL));

  if (PCTelescope_isActiveRank(sred)) {
    CHKERRQ(DMDAGetCorners(subdm,&si,&sj,&sk,&ni,&nj,&nk));
    Ml = ni;
    Nl = nj;
    Pl = nk;
  } else {
    si = sj = sk = 0;
    ni = nj = nk = 0;
    Ml = Nl = Pl = 0;
  }

  CHKERRQ(PetscMalloc1(Ml*Nl*Pl*3,&fine_indices));

  c = 0;
  if (PCTelescope_isActiveRank(sred)) {
    for (k=sk; k<sk+nk; k++) {
      for (j=sj; j<sj+nj; j++) {
        for (i=si; i<si+ni; i++) {
          nidx = (i) + (j)*M + (k)*M*N;
          fine_indices[c  ] = 3 * nidx     ;
          fine_indices[c+1] = 3 * nidx + 1 ;
          fine_indices[c+2] = 3 * nidx + 2 ;
          c = c + 3;
        }
      }
    }
  }

  /* generate scatter */
  CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)dm),Ml*Nl*Pl*3,fine_indices,PETSC_USE_POINTER,&is_fine));
  CHKERRQ(ISCreateStride(PETSC_COMM_SELF,Ml*Nl*Pl*3,0,1,&is_local));

  /* scatter */
  CHKERRQ(VecCreate(PETSC_COMM_SELF,&perm_coors));
  CHKERRQ(VecSetSizes(perm_coors,PETSC_DECIDE,Ml*Nl*Pl*3));
  CHKERRQ(VecSetType(perm_coors,VECSEQ));
  CHKERRQ(VecScatterCreate(coor_natural,is_fine,perm_coors,is_local,&sctx));
  CHKERRQ(VecScatterBegin(sctx,coor_natural,perm_coors,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(  sctx,coor_natural,perm_coors,INSERT_VALUES,SCATTER_FORWARD));

  /* access */
  if (PCTelescope_isActiveRank(sred)) {
    Vec               _coors;
    const PetscScalar *LA_perm;
    PetscScalar       *LA_coors;

    CHKERRQ(DMGetCoordinates(subdm,&_coors));
    CHKERRQ(VecGetArrayRead(perm_coors,&LA_perm));
    CHKERRQ(VecGetArray(_coors,&LA_coors));
    for (i=0; i<Ml*Nl*Pl*3; i++) {
      LA_coors[i] = LA_perm[i];
    }
    CHKERRQ(VecRestoreArray(_coors,&LA_coors));
    CHKERRQ(VecRestoreArrayRead(perm_coors,&LA_perm));
  }

  /* update local coords */
  if (PCTelescope_isActiveRank(sred)) {
    DM  _dmc;
    Vec _coors,_coors_local;

    CHKERRQ(DMGetCoordinateDM(subdm,&_dmc));
    CHKERRQ(DMGetCoordinates(subdm,&_coors));
    CHKERRQ(DMGetCoordinatesLocal(subdm,&_coors_local));
    CHKERRQ(DMGlobalToLocalBegin(_dmc,_coors,INSERT_VALUES,_coors_local));
    CHKERRQ(DMGlobalToLocalEnd(_dmc,_coors,INSERT_VALUES,_coors_local));
  }

  CHKERRQ(VecScatterDestroy(&sctx));
  CHKERRQ(ISDestroy(&is_fine));
  CHKERRQ(PetscFree(fine_indices));
  CHKERRQ(ISDestroy(&is_local));
  CHKERRQ(VecDestroy(&perm_coors));
  CHKERRQ(VecDestroy(&coor_natural));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCTelescopeSetUp_dmda_repart_coors(PC pc,PC_Telescope sred,PC_Telescope_DMDACtx *ctx)
{
  PetscInt       dim;
  DM             dm,subdm;
  PetscSubcomm   psubcomm;
  MPI_Comm       comm;
  Vec            coor;

  PetscFunctionBegin;
  CHKERRQ(PCGetDM(pc,&dm));
  CHKERRQ(DMGetCoordinates(dm,&coor));
  if (!coor) PetscFunctionReturn(0);
  psubcomm = sred->psubcomm;
  comm = PetscSubcommParent(psubcomm);
  subdm = ctx->dmrepart;

  CHKERRQ(PetscInfo(pc,"PCTelescope: setting up the coordinates (DMDA)\n"));
  CHKERRQ(DMDAGetInfo(dm,&dim,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL));
  switch (dim) {
  case 1:
    SETERRQ(comm,PETSC_ERR_SUP,"Telescope: DMDA (1D) repartitioning not provided");
  case 2: CHKERRQ(PCTelescopeSetUp_dmda_repart_coors2d(sred,dm,subdm));
    break;
  case 3: CHKERRQ(PCTelescopeSetUp_dmda_repart_coors3d(sred,dm,subdm));
    break;
  }
  PetscFunctionReturn(0);
}

/* setup repartitioned dm */
PetscErrorCode PCTelescopeSetUp_dmda_repart(PC pc,PC_Telescope sred,PC_Telescope_DMDACtx *ctx)
{
  DM                    dm;
  PetscInt              dim,nx,ny,nz,ndof,nsw,sum,k;
  DMBoundaryType        bx,by,bz;
  DMDAStencilType       stencil;
  const PetscInt        *_range_i_re;
  const PetscInt        *_range_j_re;
  const PetscInt        *_range_k_re;
  DMDAInterpolationType itype;
  PetscInt              refine_x,refine_y,refine_z;
  MPI_Comm              comm,subcomm;
  const char            *prefix;

  PetscFunctionBegin;
  comm = PetscSubcommParent(sred->psubcomm);
  subcomm = PetscSubcommChild(sred->psubcomm);
  CHKERRQ(PCGetDM(pc,&dm));

  CHKERRQ(DMDAGetInfo(dm,&dim,&nx,&ny,&nz,NULL,NULL,NULL,&ndof,&nsw,&bx,&by,&bz,&stencil));
  CHKERRQ(DMDAGetInterpolationType(dm,&itype));
  CHKERRQ(DMDAGetRefinementFactor(dm,&refine_x,&refine_y,&refine_z));

  ctx->dmrepart = NULL;
  _range_i_re = _range_j_re = _range_k_re = NULL;
  /* Create DMDA on the child communicator */
  if (PCTelescope_isActiveRank(sred)) {
    switch (dim) {
    case 1:
      CHKERRQ(PetscInfo(pc,"PCTelescope: setting up the DMDA on comm subset (1D)\n"));
      /*CHKERRQ(DMDACreate1d(subcomm,bx,nx,ndof,nsw,NULL,&ctx->dmrepart));*/
      ny = nz = 1;
      by = bz = DM_BOUNDARY_NONE;
      break;
    case 2:
      CHKERRQ(PetscInfo(pc,"PCTelescope: setting up the DMDA on comm subset (2D)\n"));
      /*CHKERRQ(DMDACreate2d(subcomm,bx,by,stencil,nx,ny, PETSC_DECIDE,PETSC_DECIDE, ndof,nsw, NULL,NULL,&ctx->dmrepart));*/
      nz = 1;
      bz = DM_BOUNDARY_NONE;
      break;
    case 3:
      CHKERRQ(PetscInfo(pc,"PCTelescope: setting up the DMDA on comm subset (3D)\n"));
      /*CHKERRQ(DMDACreate3d(subcomm,bx,by,bz,stencil,nx,ny,nz, PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE, ndof,nsw, NULL,NULL,NULL,&ctx->dmrepart));*/
      break;
    }
    /*
     The API DMDACreate1d(), DMDACreate2d(), DMDACreate3d() does not allow us to set/append
     a unique option prefix for the DM, thus I prefer to expose the contents of these API's here.
     This allows users to control the partitioning of the subDM.
    */
    CHKERRQ(DMDACreate(subcomm,&ctx->dmrepart));
    /* Set unique option prefix name */
    CHKERRQ(KSPGetOptionsPrefix(sred->ksp,&prefix));
    CHKERRQ(DMSetOptionsPrefix(ctx->dmrepart,prefix));
    CHKERRQ(DMAppendOptionsPrefix(ctx->dmrepart,"repart_"));
    /* standard setup from DMDACreate{1,2,3}d() */
    CHKERRQ(DMSetDimension(ctx->dmrepart,dim));
    CHKERRQ(DMDASetSizes(ctx->dmrepart,nx,ny,nz));
    CHKERRQ(DMDASetNumProcs(ctx->dmrepart,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE));
    CHKERRQ(DMDASetBoundaryType(ctx->dmrepart,bx,by,bz));
    CHKERRQ(DMDASetDof(ctx->dmrepart,ndof));
    CHKERRQ(DMDASetStencilType(ctx->dmrepart,stencil));
    CHKERRQ(DMDASetStencilWidth(ctx->dmrepart,nsw));
    CHKERRQ(DMDASetOwnershipRanges(ctx->dmrepart,NULL,NULL,NULL));
    CHKERRQ(DMSetFromOptions(ctx->dmrepart));
    CHKERRQ(DMSetUp(ctx->dmrepart));
    /* Set refinement factors and interpolation type from the partent */
    CHKERRQ(DMDASetRefinementFactor(ctx->dmrepart,refine_x,refine_y,refine_z));
    CHKERRQ(DMDASetInterpolationType(ctx->dmrepart,itype));

    CHKERRQ(DMDAGetInfo(ctx->dmrepart,NULL,NULL,NULL,NULL,&ctx->Mp_re,&ctx->Np_re,&ctx->Pp_re,NULL,NULL,NULL,NULL,NULL,NULL));
    CHKERRQ(DMDAGetOwnershipRanges(ctx->dmrepart,&_range_i_re,&_range_j_re,&_range_k_re));

    ctx->dmrepart->ops->creatematrix = dm->ops->creatematrix;
    ctx->dmrepart->ops->createdomaindecomposition = dm->ops->createdomaindecomposition;
  }

  /* generate ranges for repartitioned dm */
  /* note - assume rank 0 always participates */
  /* TODO: use a single MPI call */
  CHKERRMPI(MPI_Bcast(&ctx->Mp_re,1,MPIU_INT,0,comm));
  CHKERRMPI(MPI_Bcast(&ctx->Np_re,1,MPIU_INT,0,comm));
  CHKERRMPI(MPI_Bcast(&ctx->Pp_re,1,MPIU_INT,0,comm));

  CHKERRQ(PetscCalloc3(ctx->Mp_re,&ctx->range_i_re,ctx->Np_re,&ctx->range_j_re,ctx->Pp_re,&ctx->range_k_re));

  if (_range_i_re) CHKERRQ(PetscArraycpy(ctx->range_i_re,_range_i_re,ctx->Mp_re));
  if (_range_j_re) CHKERRQ(PetscArraycpy(ctx->range_j_re,_range_j_re,ctx->Np_re));
  if (_range_k_re) CHKERRQ(PetscArraycpy(ctx->range_k_re,_range_k_re,ctx->Pp_re));

  /* TODO: use a single MPI call */
  CHKERRMPI(MPI_Bcast(ctx->range_i_re,ctx->Mp_re,MPIU_INT,0,comm));
  CHKERRMPI(MPI_Bcast(ctx->range_j_re,ctx->Np_re,MPIU_INT,0,comm));
  CHKERRMPI(MPI_Bcast(ctx->range_k_re,ctx->Pp_re,MPIU_INT,0,comm));

  CHKERRQ(PetscMalloc3(ctx->Mp_re,&ctx->start_i_re,ctx->Np_re,&ctx->start_j_re,ctx->Pp_re,&ctx->start_k_re));

  sum = 0;
  for (k=0; k<ctx->Mp_re; k++) {
    ctx->start_i_re[k] = sum;
    sum += ctx->range_i_re[k];
  }

  sum = 0;
  for (k=0; k<ctx->Np_re; k++) {
    ctx->start_j_re[k] = sum;
    sum += ctx->range_j_re[k];
  }

  sum = 0;
  for (k=0; k<ctx->Pp_re; k++) {
    ctx->start_k_re[k] = sum;
    sum += ctx->range_k_re[k];
  }

  /* attach repartitioned dm to child ksp */
  {
    PetscErrorCode (*dmksp_func)(KSP,Mat,Mat,void*);
    void           *dmksp_ctx;

    CHKERRQ(DMKSPGetComputeOperators(dm,&dmksp_func,&dmksp_ctx));

    /* attach dm to ksp on sub communicator */
    if (PCTelescope_isActiveRank(sred)) {
      CHKERRQ(KSPSetDM(sred->ksp,ctx->dmrepart));

      if (!dmksp_func || sred->ignore_kspcomputeoperators) {
        CHKERRQ(KSPSetDMActive(sred->ksp,PETSC_FALSE));
      } else {
        /* sub ksp inherits dmksp_func and context provided by user */
        CHKERRQ(KSPSetComputeOperators(sred->ksp,dmksp_func,dmksp_ctx));
        CHKERRQ(KSPSetDMActive(sred->ksp,PETSC_TRUE));
      }
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PCTelescopeSetUp_dmda_permutation_3d(PC pc,PC_Telescope sred,PC_Telescope_DMDACtx *ctx)
{
  PetscErrorCode ierr;
  DM             dm;
  MPI_Comm       comm;
  Mat            Pscalar,P;
  PetscInt       ndof;
  PetscInt       i,j,k,location,startI[3],endI[3],lenI[3],nx,ny,nz;
  PetscInt       sr,er,Mr;
  Vec            V;

  PetscFunctionBegin;
  CHKERRQ(PetscInfo(pc,"PCTelescope: setting up the permutation matrix (DMDA-3D)\n"));
  CHKERRQ(PetscObjectGetComm((PetscObject)pc,&comm));

  CHKERRQ(PCGetDM(pc,&dm));
  CHKERRQ(DMDAGetInfo(dm,NULL,&nx,&ny,&nz,NULL,NULL,NULL,&ndof,NULL,NULL,NULL,NULL,NULL));

  CHKERRQ(DMGetGlobalVector(dm,&V));
  CHKERRQ(VecGetSize(V,&Mr));
  CHKERRQ(VecGetOwnershipRange(V,&sr,&er));
  CHKERRQ(DMRestoreGlobalVector(dm,&V));
  sr = sr / ndof;
  er = er / ndof;
  Mr = Mr / ndof;

  CHKERRQ(MatCreate(comm,&Pscalar));
  CHKERRQ(MatSetSizes(Pscalar,(er-sr),(er-sr),Mr,Mr));
  CHKERRQ(MatSetType(Pscalar,MATAIJ));
  CHKERRQ(MatSeqAIJSetPreallocation(Pscalar,1,NULL));
  CHKERRQ(MatMPIAIJSetPreallocation(Pscalar,1,NULL,1,NULL));

  CHKERRQ(DMDAGetCorners(dm,NULL,NULL,NULL,&lenI[0],&lenI[1],&lenI[2]));
  CHKERRQ(DMDAGetCorners(dm,&startI[0],&startI[1],&startI[2],&endI[0],&endI[1],&endI[2]));
  endI[0] += startI[0];
  endI[1] += startI[1];
  endI[2] += startI[2];

  for (k=startI[2]; k<endI[2]; k++) {
    for (j=startI[1]; j<endI[1]; j++) {
      for (i=startI[0]; i<endI[0]; i++) {
        PetscMPIInt rank_ijk_re,rank_reI[3];
        PetscInt    s0_re;
        PetscInt    ii,jj,kk,local_ijk_re,mapped_ijk;
        PetscInt    lenI_re[3];

        location = (i - startI[0]) + (j - startI[1])*lenI[0] + (k - startI[2])*lenI[0]*lenI[1];
        ierr = _DMDADetermineRankFromGlobalIJK(3,i,j,k,   ctx->Mp_re,ctx->Np_re,ctx->Pp_re,
                                               ctx->start_i_re,ctx->start_j_re,ctx->start_k_re,
                                               ctx->range_i_re,ctx->range_j_re,ctx->range_k_re,
                                               &rank_reI[0],&rank_reI[1],&rank_reI[2],&rank_ijk_re);CHKERRQ(ierr);
        CHKERRQ(_DMDADetermineGlobalS0(3,rank_ijk_re, ctx->Mp_re,ctx->Np_re,ctx->Pp_re, ctx->range_i_re,ctx->range_j_re,ctx->range_k_re, &s0_re));
        ii = i - ctx->start_i_re[ rank_reI[0] ];
        PetscCheckFalse(ii < 0,PETSC_COMM_SELF,PETSC_ERR_USER,"[dmdarepart-perm3d] index error ii");
        jj = j - ctx->start_j_re[ rank_reI[1] ];
        PetscCheckFalse(jj < 0,PETSC_COMM_SELF,PETSC_ERR_USER,"[dmdarepart-perm3d] index error jj");
        kk = k - ctx->start_k_re[ rank_reI[2] ];
        PetscCheckFalse(kk < 0,PETSC_COMM_SELF,PETSC_ERR_USER,"[dmdarepart-perm3d] index error kk");
        lenI_re[0] = ctx->range_i_re[ rank_reI[0] ];
        lenI_re[1] = ctx->range_j_re[ rank_reI[1] ];
        lenI_re[2] = ctx->range_k_re[ rank_reI[2] ];
        local_ijk_re = ii + jj * lenI_re[0] + kk * lenI_re[0] * lenI_re[1];
        mapped_ijk = s0_re + local_ijk_re;
        CHKERRQ(MatSetValue(Pscalar,sr+location,mapped_ijk,1.0,INSERT_VALUES));
      }
    }
  }
  CHKERRQ(MatAssemblyBegin(Pscalar,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(Pscalar,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatCreateMAIJ(Pscalar,ndof,&P));
  CHKERRQ(MatDestroy(&Pscalar));
  ctx->permutation = P;
  PetscFunctionReturn(0);
}

PetscErrorCode PCTelescopeSetUp_dmda_permutation_2d(PC pc,PC_Telescope sred,PC_Telescope_DMDACtx *ctx)
{
  PetscErrorCode ierr;
  DM             dm;
  MPI_Comm       comm;
  Mat            Pscalar,P;
  PetscInt       ndof;
  PetscInt       i,j,location,startI[2],endI[2],lenI[2],nx,ny,nz;
  PetscInt       sr,er,Mr;
  Vec            V;

  PetscFunctionBegin;
  CHKERRQ(PetscInfo(pc,"PCTelescope: setting up the permutation matrix (DMDA-2D)\n"));
  CHKERRQ(PetscObjectGetComm((PetscObject)pc,&comm));
  CHKERRQ(PCGetDM(pc,&dm));
  CHKERRQ(DMDAGetInfo(dm,NULL,&nx,&ny,&nz,NULL,NULL,NULL,&ndof,NULL,NULL,NULL,NULL,NULL));
  CHKERRQ(DMGetGlobalVector(dm,&V));
  CHKERRQ(VecGetSize(V,&Mr));
  CHKERRQ(VecGetOwnershipRange(V,&sr,&er));
  CHKERRQ(DMRestoreGlobalVector(dm,&V));
  sr = sr / ndof;
  er = er / ndof;
  Mr = Mr / ndof;

  CHKERRQ(MatCreate(comm,&Pscalar));
  CHKERRQ(MatSetSizes(Pscalar,(er-sr),(er-sr),Mr,Mr));
  CHKERRQ(MatSetType(Pscalar,MATAIJ));
  CHKERRQ(MatSeqAIJSetPreallocation(Pscalar,1,NULL));
  CHKERRQ(MatMPIAIJSetPreallocation(Pscalar,1,NULL,1,NULL));

  CHKERRQ(DMDAGetCorners(dm,NULL,NULL,NULL,&lenI[0],&lenI[1],NULL));
  CHKERRQ(DMDAGetCorners(dm,&startI[0],&startI[1],NULL,&endI[0],&endI[1],NULL));
  endI[0] += startI[0];
  endI[1] += startI[1];

  for (j=startI[1]; j<endI[1]; j++) {
    for (i=startI[0]; i<endI[0]; i++) {
      PetscMPIInt rank_ijk_re,rank_reI[3];
      PetscInt    s0_re;
      PetscInt    ii,jj,local_ijk_re,mapped_ijk;
      PetscInt    lenI_re[3];

      location = (i - startI[0]) + (j - startI[1])*lenI[0];
      ierr = _DMDADetermineRankFromGlobalIJK(2,i,j,0,   ctx->Mp_re,ctx->Np_re,ctx->Pp_re,
                                             ctx->start_i_re,ctx->start_j_re,ctx->start_k_re,
                                             ctx->range_i_re,ctx->range_j_re,ctx->range_k_re,
                                             &rank_reI[0],&rank_reI[1],NULL,&rank_ijk_re);CHKERRQ(ierr);

      CHKERRQ(_DMDADetermineGlobalS0(2,rank_ijk_re, ctx->Mp_re,ctx->Np_re,ctx->Pp_re, ctx->range_i_re,ctx->range_j_re,ctx->range_k_re, &s0_re));

      ii = i - ctx->start_i_re[ rank_reI[0] ];
      PetscCheckFalse(ii < 0,PETSC_COMM_SELF,PETSC_ERR_USER,"[dmdarepart-perm2d] index error ii");
      jj = j - ctx->start_j_re[ rank_reI[1] ];
      PetscCheckFalse(jj < 0,PETSC_COMM_SELF,PETSC_ERR_USER,"[dmdarepart-perm2d] index error jj");

      lenI_re[0] = ctx->range_i_re[ rank_reI[0] ];
      lenI_re[1] = ctx->range_j_re[ rank_reI[1] ];
      local_ijk_re = ii + jj * lenI_re[0];
      mapped_ijk = s0_re + local_ijk_re;
      CHKERRQ(MatSetValue(Pscalar,sr+location,mapped_ijk,1.0,INSERT_VALUES));
    }
  }
  CHKERRQ(MatAssemblyBegin(Pscalar,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(Pscalar,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatCreateMAIJ(Pscalar,ndof,&P));
  CHKERRQ(MatDestroy(&Pscalar));
  ctx->permutation = P;
  PetscFunctionReturn(0);
}

PetscErrorCode PCTelescopeSetUp_dmda_scatters(PC pc,PC_Telescope sred,PC_Telescope_DMDACtx *ctx)
{
  Vec            xred,yred,xtmp,x,xp;
  VecScatter     scatter;
  IS             isin;
  Mat            B;
  PetscInt       m,bs,st,ed;
  MPI_Comm       comm;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)pc,&comm));
  CHKERRQ(PCGetOperators(pc,NULL,&B));
  CHKERRQ(MatCreateVecs(B,&x,NULL));
  CHKERRQ(MatGetBlockSize(B,&bs));
  CHKERRQ(VecDuplicate(x,&xp));
  m = 0;
  xred = NULL;
  yred = NULL;
  if (PCTelescope_isActiveRank(sred)) {
    CHKERRQ(DMCreateGlobalVector(ctx->dmrepart,&xred));
    CHKERRQ(VecDuplicate(xred,&yred));
    CHKERRQ(VecGetOwnershipRange(xred,&st,&ed));
    CHKERRQ(ISCreateStride(comm,ed-st,st,1,&isin));
    CHKERRQ(VecGetLocalSize(xred,&m));
  } else {
    CHKERRQ(VecGetOwnershipRange(x,&st,&ed));
    CHKERRQ(ISCreateStride(comm,0,st,1,&isin));
  }
  CHKERRQ(ISSetBlockSize(isin,bs));
  CHKERRQ(VecCreate(comm,&xtmp));
  CHKERRQ(VecSetSizes(xtmp,m,PETSC_DECIDE));
  CHKERRQ(VecSetBlockSize(xtmp,bs));
  CHKERRQ(VecSetType(xtmp,((PetscObject)x)->type_name));
  CHKERRQ(VecScatterCreate(x,isin,xtmp,NULL,&scatter));
  sred->xred    = xred;
  sred->yred    = yred;
  sred->isin    = isin;
  sred->scatter = scatter;
  sred->xtmp    = xtmp;

  ctx->xp       = xp;
  CHKERRQ(VecDestroy(&x));
  PetscFunctionReturn(0);
}

PetscErrorCode PCTelescopeSetUp_dmda(PC pc,PC_Telescope sred)
{
  PC_Telescope_DMDACtx *ctx;
  PetscInt             dim;
  DM                   dm;
  MPI_Comm             comm;

  PetscFunctionBegin;
  CHKERRQ(PetscInfo(pc,"PCTelescope: setup (DMDA)\n"));
  CHKERRQ(PetscNew(&ctx));
  sred->dm_ctx = (void*)ctx;

  CHKERRQ(PetscObjectGetComm((PetscObject)pc,&comm));
  CHKERRQ(PCGetDM(pc,&dm));
  CHKERRQ(DMDAGetInfo(dm,&dim,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL));

  PCTelescopeSetUp_dmda_repart(pc,sred,ctx);
  PCTelescopeSetUp_dmda_repart_coors(pc,sred,ctx);
  switch (dim) {
  case 1:
    SETERRQ(comm,PETSC_ERR_SUP,"Telescope: DMDA (1D) repartitioning not provided");
  case 2: CHKERRQ(PCTelescopeSetUp_dmda_permutation_2d(pc,sred,ctx));
    break;
  case 3: CHKERRQ(PCTelescopeSetUp_dmda_permutation_3d(pc,sred,ctx));
    break;
  }
  CHKERRQ(PCTelescopeSetUp_dmda_scatters(pc,sred,ctx));
  PetscFunctionReturn(0);
}

PetscErrorCode PCTelescopeMatCreate_dmda_dmactivefalse(PC pc,PC_Telescope sred,MatReuse reuse,Mat *A)
{
  PC_Telescope_DMDACtx *ctx;
  MPI_Comm             comm,subcomm;
  Mat                  Bperm,Bred,B,P;
  PetscInt             nr,nc;
  IS                   isrow,iscol;
  Mat                  Blocal,*_Blocal;

  PetscFunctionBegin;
  CHKERRQ(PetscInfo(pc,"PCTelescope: updating the redundant preconditioned operator (DMDA)\n"));
  CHKERRQ(PetscObjectGetComm((PetscObject)pc,&comm));
  subcomm = PetscSubcommChild(sred->psubcomm);
  ctx = (PC_Telescope_DMDACtx*)sred->dm_ctx;

  CHKERRQ(PCGetOperators(pc,NULL,&B));
  CHKERRQ(MatGetSize(B,&nr,&nc));

  P = ctx->permutation;
  CHKERRQ(MatPtAP(B,P,MAT_INITIAL_MATRIX,1.1,&Bperm));

  /* Get submatrices */
  isrow = sred->isin;
  CHKERRQ(ISCreateStride(comm,nc,0,1,&iscol));

  CHKERRQ(MatCreateSubMatrices(Bperm,1,&isrow,&iscol,MAT_INITIAL_MATRIX,&_Blocal));
  Blocal = *_Blocal;
  Bred = NULL;
  if (PCTelescope_isActiveRank(sred)) {
    PetscInt mm;

    if (reuse != MAT_INITIAL_MATRIX) {Bred = *A;}
    CHKERRQ(MatGetSize(Blocal,&mm,NULL));
    /* CHKERRQ(MatCreateMPIMatConcatenateSeqMat(subcomm,Blocal,PETSC_DECIDE,reuse,&Bred)); */
    CHKERRQ(MatCreateMPIMatConcatenateSeqMat(subcomm,Blocal,mm,reuse,&Bred));
  }
  *A = Bred;

  CHKERRQ(ISDestroy(&iscol));
  CHKERRQ(MatDestroy(&Bperm));
  CHKERRQ(MatDestroyMatrices(1,&_Blocal));
  PetscFunctionReturn(0);
}

PetscErrorCode PCTelescopeMatCreate_dmda(PC pc,PC_Telescope sred,MatReuse reuse,Mat *A)
{
  DM             dm;
  PetscErrorCode (*dmksp_func)(KSP,Mat,Mat,void*);
  void           *dmksp_ctx;

  PetscFunctionBegin;
  CHKERRQ(PCGetDM(pc,&dm));
  CHKERRQ(DMKSPGetComputeOperators(dm,&dmksp_func,&dmksp_ctx));
  /* We assume that dmksp_func = NULL, is equivalent to dmActive = PETSC_FALSE */
  if (dmksp_func && !sred->ignore_kspcomputeoperators) {
    DM  dmrepart;
    Mat Ak;

    *A = NULL;
    if (PCTelescope_isActiveRank(sred)) {
      CHKERRQ(KSPGetDM(sred->ksp,&dmrepart));
      if (reuse == MAT_INITIAL_MATRIX) {
        CHKERRQ(DMCreateMatrix(dmrepart,&Ak));
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
    CHKERRQ(PCTelescopeMatCreate_dmda_dmactivefalse(pc,sred,reuse,A));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PCTelescopeSubNullSpaceCreate_dmda_Telescope(PC pc,PC_Telescope sred,MatNullSpace nullspace,MatNullSpace *sub_nullspace)
{
  PetscBool            has_const;
  PetscInt             i,k,n = 0;
  const Vec            *vecs;
  Vec                  *sub_vecs = NULL;
  MPI_Comm             subcomm;
  PC_Telescope_DMDACtx *ctx;

  PetscFunctionBegin;
  ctx = (PC_Telescope_DMDACtx*)sred->dm_ctx;
  subcomm = PetscSubcommChild(sred->psubcomm);
  CHKERRQ(MatNullSpaceGetVecs(nullspace,&has_const,&n,&vecs));

  if (PCTelescope_isActiveRank(sred)) {
    /* create new vectors */
    if (n) {
      CHKERRQ(VecDuplicateVecs(sred->xred,n,&sub_vecs));
    }
  }

  /* copy entries */
  for (k=0; k<n; k++) {
    const PetscScalar *x_array;
    PetscScalar       *LA_sub_vec;
    PetscInt          st,ed;

    /* permute vector into ordering associated with re-partitioned dmda */
    CHKERRQ(MatMultTranspose(ctx->permutation,vecs[k],ctx->xp));

    /* pull in vector x->xtmp */
    CHKERRQ(VecScatterBegin(sred->scatter,ctx->xp,sred->xtmp,INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecScatterEnd(sred->scatter,ctx->xp,sred->xtmp,INSERT_VALUES,SCATTER_FORWARD));

    /* copy vector entries into xred */
    CHKERRQ(VecGetArrayRead(sred->xtmp,&x_array));
    if (sub_vecs) {
      if (sub_vecs[k]) {
        CHKERRQ(VecGetOwnershipRange(sub_vecs[k],&st,&ed));
        CHKERRQ(VecGetArray(sub_vecs[k],&LA_sub_vec));
        for (i=0; i<ed-st; i++) {
          LA_sub_vec[i] = x_array[i];
        }
        CHKERRQ(VecRestoreArray(sub_vecs[k],&LA_sub_vec));
      }
    }
    CHKERRQ(VecRestoreArrayRead(sred->xtmp,&x_array));
  }

  if (PCTelescope_isActiveRank(sred)) {
    /* create new (near) nullspace for redundant object */
    CHKERRQ(MatNullSpaceCreate(subcomm,has_const,n,sub_vecs,sub_nullspace));
    CHKERRQ(VecDestroyVecs(n,&sub_vecs));
    PetscCheck(!nullspace->remove,PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"Propagation of custom remove callbacks not supported when propagating (near) nullspaces with PCTelescope");
    PetscCheck(!nullspace->rmctx,PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"Propagation of custom remove callback context not supported when propagating (near) nullspaces with PCTelescope");
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PCTelescopeMatNullSpaceCreate_dmda(PC pc,PC_Telescope sred,Mat sub_mat)
{
  Mat            B;

  PetscFunctionBegin;
  CHKERRQ(PCGetOperators(pc,NULL,&B));
  {
    MatNullSpace nullspace,sub_nullspace;
    CHKERRQ(MatGetNullSpace(B,&nullspace));
    if (nullspace) {
      CHKERRQ(PetscInfo(pc,"PCTelescope: generating nullspace (DMDA)\n"));
      CHKERRQ(PCTelescopeSubNullSpaceCreate_dmda_Telescope(pc,sred,nullspace,&sub_nullspace));
      if (PCTelescope_isActiveRank(sred)) {
        CHKERRQ(MatSetNullSpace(sub_mat,sub_nullspace));
        CHKERRQ(MatNullSpaceDestroy(&sub_nullspace));
      }
    }
  }
  {
    MatNullSpace nearnullspace,sub_nearnullspace;
    CHKERRQ(MatGetNearNullSpace(B,&nearnullspace));
    if (nearnullspace) {
      CHKERRQ(PetscInfo(pc,"PCTelescope: generating near nullspace (DMDA)\n"));
      CHKERRQ(PCTelescopeSubNullSpaceCreate_dmda_Telescope(pc,sred,nearnullspace,&sub_nearnullspace));
      if (PCTelescope_isActiveRank(sred)) {
        CHKERRQ(MatSetNearNullSpace(sub_mat,sub_nearnullspace));
        CHKERRQ(MatNullSpaceDestroy(&sub_nearnullspace));
      }
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PCApply_Telescope_dmda(PC pc,Vec x,Vec y)
{
  PC_Telescope         sred = (PC_Telescope)pc->data;
  Mat                  perm;
  Vec                  xtmp,xp,xred,yred;
  PetscInt             i,st,ed;
  VecScatter           scatter;
  PetscScalar          *array;
  const PetscScalar    *x_array;
  PC_Telescope_DMDACtx *ctx;

  ctx = (PC_Telescope_DMDACtx*)sred->dm_ctx;
  xtmp    = sred->xtmp;
  scatter = sred->scatter;
  xred    = sred->xred;
  yred    = sred->yred;
  perm    = ctx->permutation;
  xp      = ctx->xp;

  PetscFunctionBegin;
  CHKERRQ(PetscCitationsRegister(citation,&cited));

  /* permute vector into ordering associated with re-partitioned dmda */
  CHKERRQ(MatMultTranspose(perm,x,xp));

  /* pull in vector x->xtmp */
  CHKERRQ(VecScatterBegin(scatter,xp,xtmp,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(scatter,xp,xtmp,INSERT_VALUES,SCATTER_FORWARD));

  /* copy vector entries into xred */
  CHKERRQ(VecGetArrayRead(xtmp,&x_array));
  if (xred) {
    PetscScalar *LA_xred;
    CHKERRQ(VecGetOwnershipRange(xred,&st,&ed));

    CHKERRQ(VecGetArray(xred,&LA_xred));
    for (i=0; i<ed-st; i++) {
      LA_xred[i] = x_array[i];
    }
    CHKERRQ(VecRestoreArray(xred,&LA_xred));
  }
  CHKERRQ(VecRestoreArrayRead(xtmp,&x_array));

  /* solve */
  if (PCTelescope_isActiveRank(sred)) {
    CHKERRQ(KSPSolve(sred->ksp,xred,yred));
    CHKERRQ(KSPCheckSolve(sred->ksp,pc,yred));
  }

  /* return vector */
  CHKERRQ(VecGetArray(xtmp,&array));
  if (yred) {
    const PetscScalar *LA_yred;
    CHKERRQ(VecGetOwnershipRange(yred,&st,&ed));
    CHKERRQ(VecGetArrayRead(yred,&LA_yred));
    for (i=0; i<ed-st; i++) {
      array[i] = LA_yred[i];
    }
    CHKERRQ(VecRestoreArrayRead(yred,&LA_yred));
  }
  CHKERRQ(VecRestoreArray(xtmp,&array));
  CHKERRQ(VecScatterBegin(scatter,xtmp,xp,INSERT_VALUES,SCATTER_REVERSE));
  CHKERRQ(VecScatterEnd(scatter,xtmp,xp,INSERT_VALUES,SCATTER_REVERSE));
  CHKERRQ(MatMult(perm,xp,y));
  PetscFunctionReturn(0);
}

PetscErrorCode PCApplyRichardson_Telescope_dmda(PC pc,Vec x,Vec y,Vec w,PetscReal rtol,PetscReal abstol, PetscReal dtol,PetscInt its,PetscBool zeroguess,PetscInt *outits,PCRichardsonConvergedReason *reason)
{
  PC_Telescope         sred = (PC_Telescope)pc->data;
  Mat                  perm;
  Vec                  xtmp,xp,yred;
  PetscInt             i,st,ed;
  VecScatter           scatter;
  const PetscScalar    *x_array;
  PetscBool            default_init_guess_value = PETSC_FALSE;
  PC_Telescope_DMDACtx *ctx;

  PetscFunctionBegin;
  ctx = (PC_Telescope_DMDACtx*)sred->dm_ctx;
  xtmp    = sred->xtmp;
  scatter = sred->scatter;
  yred    = sred->yred;
  perm    = ctx->permutation;
  xp      = ctx->xp;

  PetscCheckFalse(its > 1,PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"PCApplyRichardson_Telescope_dmda only supports max_it = 1");
  *reason = (PCRichardsonConvergedReason)0;

  if (!zeroguess) {
    CHKERRQ(PetscInfo(pc,"PCTelescopeDMDA: Scattering y for non-zero-initial guess\n"));
    /* permute vector into ordering associated with re-partitioned dmda */
    CHKERRQ(MatMultTranspose(perm,y,xp));

    /* pull in vector x->xtmp */
    CHKERRQ(VecScatterBegin(scatter,xp,xtmp,INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecScatterEnd(scatter,xp,xtmp,INSERT_VALUES,SCATTER_FORWARD));

    /* copy vector entries into xred */
    CHKERRQ(VecGetArrayRead(xtmp,&x_array));
    if (yred) {
      PetscScalar *LA_yred;
      CHKERRQ(VecGetOwnershipRange(yred,&st,&ed));
      CHKERRQ(VecGetArray(yred,&LA_yred));
      for (i=0; i<ed-st; i++) {
        LA_yred[i] = x_array[i];
      }
      CHKERRQ(VecRestoreArray(yred,&LA_yred));
    }
    CHKERRQ(VecRestoreArrayRead(xtmp,&x_array));
  }

  if (PCTelescope_isActiveRank(sred)) {
    CHKERRQ(KSPGetInitialGuessNonzero(sred->ksp,&default_init_guess_value));
    if (!zeroguess) CHKERRQ(KSPSetInitialGuessNonzero(sred->ksp,PETSC_TRUE));
  }

  CHKERRQ(PCApply_Telescope_dmda(pc,x,y));

  if (PCTelescope_isActiveRank(sred)) {
    CHKERRQ(KSPSetInitialGuessNonzero(sred->ksp,default_init_guess_value));
  }

  if (!*reason) *reason = PCRICHARDSON_CONVERGED_ITS;
  *outits = 1;
  PetscFunctionReturn(0);
}

PetscErrorCode PCReset_Telescope_dmda(PC pc)
{
  PC_Telescope         sred = (PC_Telescope)pc->data;
  PC_Telescope_DMDACtx *ctx;

  PetscFunctionBegin;
  ctx = (PC_Telescope_DMDACtx*)sred->dm_ctx;
  CHKERRQ(VecDestroy(&ctx->xp));
  CHKERRQ(MatDestroy(&ctx->permutation));
  CHKERRQ(DMDestroy(&ctx->dmrepart));
  CHKERRQ(PetscFree3(ctx->range_i_re,ctx->range_j_re,ctx->range_k_re));
  CHKERRQ(PetscFree3(ctx->start_i_re,ctx->start_j_re,ctx->start_k_re));
  PetscFunctionReturn(0);
}

PetscErrorCode DMView_DA_Short_3d(DM dm,PetscViewer v)
{
  PetscInt       M,N,P,m,n,p,ndof,nsw;
  MPI_Comm       comm;
  PetscMPIInt    size;
  const char*    prefix;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)dm,&comm));
  CHKERRMPI(MPI_Comm_size(comm,&size));
  CHKERRQ(DMGetOptionsPrefix(dm,&prefix));
  CHKERRQ(DMDAGetInfo(dm,NULL,&M,&N,&P,&m,&n,&p,&ndof,&nsw,NULL,NULL,NULL,NULL));
  if (prefix) CHKERRQ(PetscViewerASCIIPrintf(v,"DMDA Object:    (%s)    %d MPI processes\n",prefix,size));
  else CHKERRQ(PetscViewerASCIIPrintf(v,"DMDA Object:    %d MPI processes\n",size));
  CHKERRQ(PetscViewerASCIIPrintf(v,"  M %D N %D P %D m %D n %D p %D dof %D overlap %D\n",M,N,P,m,n,p,ndof,nsw));
  PetscFunctionReturn(0);
}

PetscErrorCode DMView_DA_Short_2d(DM dm,PetscViewer v)
{
  PetscInt       M,N,m,n,ndof,nsw;
  MPI_Comm       comm;
  PetscMPIInt    size;
  const char*    prefix;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)dm,&comm));
  CHKERRMPI(MPI_Comm_size(comm,&size));
  CHKERRQ(DMGetOptionsPrefix(dm,&prefix));
  CHKERRQ(DMDAGetInfo(dm,NULL,&M,&N,NULL,&m,&n,NULL,&ndof,&nsw,NULL,NULL,NULL,NULL));
  if (prefix) CHKERRQ(PetscViewerASCIIPrintf(v,"DMDA Object:    (%s)    %d MPI processes\n",prefix,size));
  else CHKERRQ(PetscViewerASCIIPrintf(v,"DMDA Object:    %d MPI processes\n",size));
  CHKERRQ(PetscViewerASCIIPrintf(v,"  M %D N %D m %D n %D dof %D overlap %D\n",M,N,m,n,ndof,nsw));
  PetscFunctionReturn(0);
}

PetscErrorCode DMView_DA_Short(DM dm,PetscViewer v)
{
  PetscInt       dim;

  PetscFunctionBegin;
  CHKERRQ(DMDAGetInfo(dm,&dim,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL));
  switch (dim) {
  case 2: CHKERRQ(DMView_DA_Short_2d(dm,v));
    break;
  case 3: CHKERRQ(DMView_DA_Short_3d(dm,v));
    break;
  }
  PetscFunctionReturn(0);
}
