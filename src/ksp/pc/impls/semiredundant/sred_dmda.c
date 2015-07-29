
#include <petsc/private/pcimpl.h>
#include <petscksp.h>           /*I "petscksp.h" I*/
#include <petscdm.h>
#include <petscdmda.h>

#include "semiredundant.h"

#undef __FUNCT__
#define __FUNCT__ "_DMDADetermineRankFromGlobalIJK"
PetscErrorCode _DMDADetermineRankFromGlobalIJK(PetscInt dim,PetscInt i,PetscInt j,PetscInt k,
                                               PetscInt Mp,PetscInt Np,PetscInt Pp,
                                               PetscInt start_i[],PetscInt start_j[],PetscInt start_k[],
                                               PetscInt span_i[],PetscInt span_j[],PetscInt span_k[],
                                               PetscMPIInt *_pi,PetscMPIInt *_pj,PetscMPIInt *_pk,PetscMPIInt *rank_re)
{
  PetscInt pi,pj,pk,n;
  
  pi = pj = pk = -1;
  
  if (_pi) {
    for (n=0; n<Mp; n++) {
      if ( (i >= start_i[n]) && (i < start_i[n]+span_i[n]) ) {
        pi = n;
        break;
      }
    }
    if (pi == -1) { SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"  [dmdarepart][pi] cannot be determined : range %D, val %D",Mp,i); }
    *_pi = pi;
  }
  
  if (_pj) {
    for (n=0; n<Np; n++) {
      if ( (j >= start_j[n]) && (j < start_j[n]+span_j[n]) ) {
        pj = n;
        break;
      }
    }
    if (pj == -1) { SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"  [dmdarepart][pj] cannot be determined : range %D, val %D",Np,j); }
    *_pj = pj;
  }
  
  if (_pk) {
    for (n=0; n<Pp; n++) {
      if ( (k >= start_k[n]) && (k < start_k[n]+span_k[n]) ) {
        pk = n;
        break;
      }
    }
    if (pk == -1) { SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"  [dmdarepart][pk] cannot be determined : range %D, val %D",Pp,k); }
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
  return(0);
}

#undef __FUNCT__
#define __FUNCT__ "_DMDADetermineGlobalS0"
PetscErrorCode _DMDADetermineGlobalS0(PetscInt dim,PetscMPIInt rank_re,PetscInt Mp_re,PetscInt Np_re,PetscInt Pp_re,
                                      PetscInt range_i_re[],PetscInt range_j_re[],PetscInt range_k_re[],PetscInt *s0)
{
  PetscInt i,j,k,start_IJK;
  PetscInt rank_ijk;
  
  switch (dim) {
    case 1:
      start_IJK = 0;
      for (i=0; i<Mp_re; i++) {
        rank_ijk = i;
        if (rank_ijk < rank_re) {
          start_IJK += range_i_re[i];
        }
      }
      break;
      
    case 2:
      start_IJK = 0;
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
      start_IJK = 0;
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
  return(0);
}

/* setup repartitioned dm */
#undef __FUNCT__
#define __FUNCT__ "PCSemiRedundantSetUp_dmda_repart"
PetscErrorCode PCSemiRedundantSetUp_dmda_repart(PC pc,PC_SemiRedundant *sred,PC_SemiRedundant_DMDACtx *ctx)
{
  PetscErrorCode  ierr;
  DM              dm;
  PetscInt        dim,nx,ny,nz,ndof,nsw,sum,k;
  DMBoundaryType  bx,by,bz;
  DMDAStencilType stencil;
  const PetscInt  *_range_i_re;
  const PetscInt  *_range_j_re;
  const PetscInt  *_range_k_re;
  DMDAInterpolationType itype;
  PetscInt refine_x,refine_y,refine_z;
  MPI_Comm comm,subcomm;
  
  comm = PetscSubcommParent(sred->psubcomm);
  subcomm = PetscSubcommChild(sred->psubcomm);
  ierr = PCGetDM(pc,&dm);CHKERRQ(ierr);
  
  ierr = DMDAGetInfo(dm,&dim,&nx,&ny,&nz,NULL,NULL,NULL,&ndof,&nsw,&bx,&by,&bz,&stencil);CHKERRQ(ierr);
  ierr = DMDAGetInterpolationType(dm,&itype);CHKERRQ(ierr);
  ierr = DMDAGetRefinementFactor(dm,&refine_x,&refine_y,&refine_z);CHKERRQ(ierr);
  
  ctx->dmrepart = NULL;
  _range_i_re = _range_j_re = _range_k_re = NULL;
  
  /* Create DMDA on the child communicator */
  if (isActiveRank(sred->psubcomm)) {
    
    switch (dim) {
      case 1:
        PetscInfo(pc,"PCSemiRedundant: setting up the DMDA on comm subset (1D)\n");
        ierr = DMDACreate1d(subcomm,bx,nx,ndof,nsw,NULL,&ctx->dmrepart);CHKERRQ(ierr);
        break;
      case 2:
        PetscInfo(pc,"PCSemiRedundant: setting up the DMDA on comm subset (2D)\n");
        ierr = DMDACreate2d(subcomm,bx,by,stencil,nx,ny, PETSC_DECIDE,PETSC_DECIDE, ndof,nsw, NULL,NULL,&ctx->dmrepart);CHKERRQ(ierr);
        break;
      case 3:
        PetscInfo(pc,"PCSemiRedundant: setting up the DMDA on comm subset (3D)\n");
        ierr = DMDACreate3d(subcomm,bx,by,bz,stencil,nx,ny,nz, PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE, ndof,nsw, NULL,NULL,NULL,&ctx->dmrepart);CHKERRQ(ierr);
        break;
    }
    
    ierr = DMSetOptionsPrefix(ctx->dmrepart,"repart_");CHKERRQ(ierr);
    
    ierr = DMDASetRefinementFactor(ctx->dmrepart,refine_x,refine_y,refine_z);CHKERRQ(ierr);
    ierr = DMDASetInterpolationType(ctx->dmrepart,itype);CHKERRQ(ierr);
    
    ierr = DMDAGetInfo(ctx->dmrepart,NULL,NULL,NULL,NULL,&ctx->Mp_re,&ctx->Np_re,&ctx->Pp_re,NULL,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
    ierr = DMDAGetOwnershipRanges(ctx->dmrepart,&_range_i_re,&_range_j_re,&_range_k_re);CHKERRQ(ierr);
  }
  
  /* generate ranges for repartitioned dm */
  /* note - assume rank 0 always participates */
  ierr = MPI_Bcast(&ctx->Mp_re,1,MPIU_INT,0,comm);CHKERRQ(ierr);
  ierr = MPI_Bcast(&ctx->Np_re,1,MPIU_INT,0,comm);CHKERRQ(ierr);
  ierr = MPI_Bcast(&ctx->Pp_re,1,MPIU_INT,0,comm);CHKERRQ(ierr);
  
  PetscMalloc(sizeof(PetscInt)*ctx->Mp_re,&ctx->range_i_re);
  PetscMalloc(sizeof(PetscInt)*ctx->Np_re,&ctx->range_j_re);
  PetscMalloc(sizeof(PetscInt)*ctx->Pp_re,&ctx->range_k_re);
  
  if (_range_i_re != NULL) { PetscMemcpy(ctx->range_i_re,_range_i_re,sizeof(PetscInt)*ctx->Mp_re); }
  if (_range_j_re != NULL) { PetscMemcpy(ctx->range_j_re,_range_j_re,sizeof(PetscInt)*ctx->Np_re); }
  if (_range_k_re != NULL) { PetscMemcpy(ctx->range_k_re,_range_k_re,sizeof(PetscInt)*ctx->Pp_re); }
  
  ierr = MPI_Bcast(ctx->range_i_re,ctx->Mp_re,MPIU_INT,0,comm);CHKERRQ(ierr);
  ierr = MPI_Bcast(ctx->range_j_re,ctx->Np_re,MPIU_INT,0,comm);CHKERRQ(ierr);
  ierr = MPI_Bcast(ctx->range_k_re,ctx->Pp_re,MPIU_INT,0,comm);CHKERRQ(ierr);
  
  PetscMalloc(sizeof(PetscInt)*ctx->Mp_re,&ctx->start_i_re);
  PetscMalloc(sizeof(PetscInt)*ctx->Np_re,&ctx->start_j_re);
  PetscMalloc(sizeof(PetscInt)*ctx->Pp_re,&ctx->start_k_re);
  
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
  
  /* attach dm to ksp on sub communicator */
  if (isActiveRank(sred->psubcomm)) {
    ierr = KSPSetDM(sred->ksp,ctx->dmrepart);CHKERRQ(ierr);
    ierr = KSPSetDMActive(sred->ksp,PETSC_FALSE);CHKERRQ(ierr);
  }
  return(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCSemiRedundantSetUp_dmda_permutation_3d"
PetscErrorCode PCSemiRedundantSetUp_dmda_permutation_3d(PC pc,PC_SemiRedundant *sred,PC_SemiRedundant_DMDACtx *ctx)
{
  PetscErrorCode ierr;
  DM             dm;
  MPI_Comm       comm;
  Mat            Pscalar,P;
  PetscInt       ndof;
  PetscInt       i,j,k,location,startI[3],endI[3],lenI[3],nx,ny,nz;
  PetscInt       sr,er,Mr;
  Vec            V;
  
  PetscInfo(pc,"PCSemiRedundant: setting up the permutation matrix (DMDA-3D)\n");
  ierr = PetscObjectGetComm((PetscObject)pc,&comm);CHKERRQ(ierr);
  
  ierr = PCGetDM(pc,&dm);CHKERRQ(ierr);
  ierr = DMDAGetInfo(dm,NULL,&nx,&ny,&nz,NULL,NULL,NULL,&ndof,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  
  ierr = DMGetGlobalVector(dm,&V);CHKERRQ(ierr);
  ierr = VecGetSize(V,&Mr);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(V,&sr,&er);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm,&V);CHKERRQ(ierr);
  sr = sr / ndof;
  er = er / ndof;
  Mr = Mr / ndof;
  
  ierr = MatCreate(comm,&Pscalar);CHKERRQ(ierr);
  ierr = MatSetSizes(Pscalar,(er-sr),(er-sr),Mr,Mr);CHKERRQ(ierr);
  ierr = MatSetType(Pscalar,MATAIJ);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(Pscalar,2,NULL);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(Pscalar,2,NULL,2,NULL);CHKERRQ(ierr);
  
  ierr = DMDAGetCorners(dm,NULL,NULL,NULL,&lenI[0],&lenI[1],&lenI[2]);CHKERRQ(ierr);
  ierr = DMDAGetCorners(dm,&startI[0],&startI[1],&startI[2],&endI[0],&endI[1],&endI[2]);CHKERRQ(ierr);
  endI[0] += startI[0];
  endI[1] += startI[1];
  endI[2] += startI[2];
  
  for (k=startI[2]; k<endI[2]; k++) {
    for (j=startI[1]; j<endI[1]; j++) {
      for (i=startI[0]; i<endI[0]; i++) {
        PetscMPIInt rank_ijk_re,rank_reI[3];
        PetscInt    s0_re;
        PetscInt    ii,jj,kk,local_ijk_re,mapped_ijk,natural_ijk;
        PetscInt    lenI_re[3];
        
        location = (i - startI[0]) + (j - startI[1])*lenI[0] + (k - startI[2])*lenI[0]*lenI[1];
        
        ierr = _DMDADetermineRankFromGlobalIJK(3,i,j,k,   ctx->Mp_re,ctx->Np_re,ctx->Pp_re,
                                               ctx->start_i_re,ctx->start_j_re,ctx->start_k_re,
                                               ctx->range_i_re,ctx->range_j_re,ctx->range_k_re,
                                               &rank_reI[0],&rank_reI[1],&rank_reI[2],&rank_ijk_re);
        
        ierr = _DMDADetermineGlobalS0(3,rank_ijk_re, ctx->Mp_re,ctx->Np_re,ctx->Pp_re, ctx->range_i_re,ctx->range_j_re,ctx->range_k_re, &s0_re);CHKERRQ(ierr);
        
        ii = i - ctx->start_i_re[ rank_reI[0] ];
        if (ii < 0) { SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"  [dmdarepart] index error ii \n"); }
        
        jj = j - ctx->start_j_re[ rank_reI[1] ];
        if (jj < 0) { SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"  [dmdarepart] index error jj \n"); }
        
        kk = k - ctx->start_k_re[ rank_reI[2] ];
        if (kk < 0) { SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"  [dmdarepart] index error kk \n"); }
        
        lenI_re[0] = ctx->range_i_re[ rank_reI[0] ];
        lenI_re[1] = ctx->range_j_re[ rank_reI[1] ];
        lenI_re[2] = ctx->range_k_re[ rank_reI[2] ];
        
        local_ijk_re = ii + jj * lenI_re[0] + kk * lenI_re[0] * lenI_re[1];
        mapped_ijk = s0_re + local_ijk_re;
        natural_ijk = i + j*nx + k*nx*ny;
        
        ierr = MatSetValue(Pscalar,sr+location,mapped_ijk,1.0,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }
  ierr = MatAssemblyBegin(Pscalar,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Pscalar,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  
  ierr = MatCreateMAIJ(Pscalar,ndof,&P);CHKERRQ(ierr);
  ierr = MatDestroy(&Pscalar);CHKERRQ(ierr);
  
  ctx->permutation = P;
  return(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCSemiRedundantSetUp_dmda_permutation_2d"
PetscErrorCode PCSemiRedundantSetUp_dmda_permutation_2d(PC pc,PC_SemiRedundant *sred,PC_SemiRedundant_DMDACtx *ctx)
{
  PetscErrorCode ierr;
  DM             dm;
  MPI_Comm       comm;
  Mat            Pscalar,P;
  PetscInt       ndof;
  PetscInt       i,j,location,startI[2],endI[2],lenI[2],nx,ny,nz;
  PetscInt       sr,er,Mr;
  Vec            V;
  
  PetscInfo(pc,"PCSemiRedundant: setting up the permutation matrix (DMDA-2D)\n");
  ierr = PetscObjectGetComm((PetscObject)pc,&comm);CHKERRQ(ierr);
  
  ierr = PCGetDM(pc,&dm);CHKERRQ(ierr);
  ierr = DMDAGetInfo(dm,NULL,&nx,&ny,&nz,NULL,NULL,NULL,&ndof,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  
  ierr = DMGetGlobalVector(dm,&V);CHKERRQ(ierr);
  ierr = VecGetSize(V,&Mr);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(V,&sr,&er);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm,&V);CHKERRQ(ierr);
  sr = sr / ndof;
  er = er / ndof;
  Mr = Mr / ndof;
  
  ierr = MatCreate(comm,&Pscalar);CHKERRQ(ierr);
  ierr = MatSetSizes(Pscalar,(er-sr),(er-sr),Mr,Mr);CHKERRQ(ierr);
  ierr = MatSetType(Pscalar,MATAIJ);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(Pscalar,2,NULL);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(Pscalar,2,NULL,2,NULL);CHKERRQ(ierr);
  
  ierr = DMDAGetCorners(dm,NULL,NULL,NULL,&lenI[0],&lenI[1],NULL);CHKERRQ(ierr);
  ierr = DMDAGetCorners(dm,&startI[0],&startI[1],NULL,&endI[0],&endI[1],NULL);CHKERRQ(ierr);
  endI[0] += startI[0];
  endI[1] += startI[1];
  
  for (j=startI[1]; j<endI[1]; j++) {
    for (i=startI[0]; i<endI[0]; i++) {
      PetscMPIInt rank_ijk_re,rank_reI[3];
      PetscInt    s0_re;
      PetscInt    ii,jj,local_ijk_re,mapped_ijk,natural_ijk;
      PetscInt    lenI_re[3];
      
      location = (i - startI[0]) + (j - startI[1])*lenI[0];
      
      ierr = _DMDADetermineRankFromGlobalIJK(2,i,j,0,   ctx->Mp_re,ctx->Np_re,ctx->Pp_re,
                                             ctx->start_i_re,ctx->start_j_re,ctx->start_k_re,
                                             ctx->range_i_re,ctx->range_j_re,ctx->range_k_re,
                                             &rank_reI[0],&rank_reI[1],NULL,&rank_ijk_re);
      
      ierr = _DMDADetermineGlobalS0(2,rank_ijk_re, ctx->Mp_re,ctx->Np_re,ctx->Pp_re, ctx->range_i_re,ctx->range_j_re,ctx->range_k_re, &s0_re);CHKERRQ(ierr);
      
      ii = i - ctx->start_i_re[ rank_reI[0] ];
      if (ii < 0) { SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"  [dmdarepart] index error ii \n"); }
      
      jj = j - ctx->start_j_re[ rank_reI[1] ];
      if (jj < 0) { SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"  [dmdarepart] index error jj \n"); }
      
      lenI_re[0] = ctx->range_i_re[ rank_reI[0] ];
      lenI_re[1] = ctx->range_j_re[ rank_reI[1] ];
      
      local_ijk_re = ii + jj * lenI_re[0];
      mapped_ijk = s0_re + local_ijk_re;
      natural_ijk = i + j*nx;
      
      ierr = MatSetValue(Pscalar,sr+location,mapped_ijk,1.0,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(Pscalar,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Pscalar,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  
  ierr = MatCreateMAIJ(Pscalar,ndof,&P);CHKERRQ(ierr);
  ierr = MatDestroy(&Pscalar);CHKERRQ(ierr);
  
  ctx->permutation = P;
  return(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCSemiRedundantSetUp_dmda_scatters"
PetscErrorCode PCSemiRedundantSetUp_dmda_scatters(PC pc,PC_SemiRedundant *sred,PC_SemiRedundant_DMDACtx *ctx)
{
  PetscErrorCode ierr;
  Vec xred,yred,xtmp,x,xp;
  VecScatter scatter;
  IS isin;
  Mat B;
  PetscInt m,bs,st,ed;
  MPI_Comm comm;
  
  ierr = PetscObjectGetComm((PetscObject)pc,&comm);CHKERRQ(ierr);
  ierr = PCGetOperators(pc,NULL,&B);CHKERRQ(ierr);
  ierr = MatCreateVecs(B,&x,NULL);CHKERRQ(ierr);
  ierr = MatGetBlockSize(B,&bs);CHKERRQ(ierr);
  
  ierr = VecDuplicate(x,&xp);CHKERRQ(ierr);
  
  m = bs;
  xred = NULL;
  yred = NULL;
  if (isActiveRank(sred->psubcomm)) {
    ierr = DMCreateGlobalVector(ctx->dmrepart,&xred);CHKERRQ(ierr);
    ierr = VecDuplicate(xred,&yred);CHKERRQ(ierr);
    
    ierr = VecGetOwnershipRange(xred,&st,&ed);CHKERRQ(ierr);
    ierr = ISCreateStride(comm,ed-st,st,1,&isin);CHKERRQ(ierr);
    
    ierr = VecGetLocalSize(xred,&m);
  } else {
    /* fetch some local owned data - just to deal with avoiding zero length ownership on range */
    ierr = VecGetOwnershipRange(x,&st,&ed);CHKERRQ(ierr);
    ierr = ISCreateStride(comm,bs,st,1,&isin);CHKERRQ(ierr);
  }
  ierr = ISSetBlockSize(isin,bs);CHKERRQ(ierr);
  
  ierr = VecCreate(comm,&xtmp);CHKERRQ(ierr);
  ierr = VecSetSizes(xtmp,m,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetBlockSize(xtmp,bs);CHKERRQ(ierr);
  ierr = VecSetType(xtmp,((PetscObject)x)->type_name);CHKERRQ(ierr);
  
  ierr = VecScatterCreate(x,isin,xtmp,NULL,&scatter);CHKERRQ(ierr);
  
  sred->xred    = xred;
  sred->yred    = yred;
  
  sred->isin    = isin;
  sred->scatter = scatter;
  sred->xtmp    = xtmp;
  
  ctx->xp       = xp;
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  return(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCSemiRedundantSetUp_dmda"
PetscErrorCode PCSemiRedundantSetUp_dmda(PC pc,PC_SemiRedundant *sred)
{
  
  PC_SemiRedundant_DMDACtx *ctx;
  PetscInt dim;
  DM dm;
  MPI_Comm comm;
  PetscErrorCode ierr;
  
  PetscInfo(pc,"PCSemiRedundant: setup (DMDA)\n");
  PetscMalloc(sizeof(PC_SemiRedundant_DMDACtx),&ctx);
  PetscMemzero(ctx,sizeof(PC_SemiRedundant_DMDACtx));
  sred->dm_ctx = (void*)ctx;
  
  ierr = PetscObjectGetComm((PetscObject)pc,&comm);CHKERRQ(ierr);
  ierr = PCGetDM(pc,&dm);CHKERRQ(ierr);
  ierr = DMDAGetInfo(dm,&dim,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  
  PCSemiRedundantSetUp_dmda_repart(pc,sred,ctx);
  switch (dim) {
    case 1:
      SETERRQ(comm,PETSC_ERR_SUP,"SemiRedundant: DMDA (1D) repartitioning not provided");
      break;
    case 2: PCSemiRedundantSetUp_dmda_permutation_2d(pc,sred,ctx);
      break;
    case 3: PCSemiRedundantSetUp_dmda_permutation_3d(pc,sred,ctx);
      break;
  }
  PCSemiRedundantSetUp_dmda_scatters(pc,sred,ctx);
  return(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCSemiRedundantMatCreate_dmda"
PetscErrorCode PCSemiRedundantMatCreate_dmda(PC pc,PC_SemiRedundant *sred,MatReuse reuse,Mat *A)
{
  PetscErrorCode ierr;
  PC_SemiRedundant_DMDACtx *ctx;
  MPI_Comm       comm,subcomm;
  Mat            Bperm,Bred,B,P;
  PetscInt       nr,nc;
  IS             isrow,iscol;
  Mat            Blocal,*_Blocal;
  
  PetscInfo(pc,"PCSemiRedundant: updating the redundant preconditioned operator (DMDA)\n");
  ierr = PetscObjectGetComm((PetscObject)pc,&comm);CHKERRQ(ierr);
  subcomm = PetscSubcommChild(sred->psubcomm);
  ctx = (PC_SemiRedundant_DMDACtx*)sred->dm_ctx;
  
  ierr = PCGetOperators(pc,NULL,&B);CHKERRQ(ierr);
  ierr = MatGetSize(B,&nr,&nc);CHKERRQ(ierr);
  
  P = ctx->permutation;
  ierr = MatPtAP(B,P,MAT_INITIAL_MATRIX,1.1,&Bperm);CHKERRQ(ierr);
  
  /* Get submatrices */
  isrow = sred->isin;
  ierr = ISCreateStride(comm,nc,0,1,&iscol);CHKERRQ(ierr);
  
  ierr = MatGetSubMatrices(Bperm,1,&isrow,&iscol,MAT_INITIAL_MATRIX,&_Blocal);CHKERRQ(ierr);
  Blocal = *_Blocal;
  
  Bred = NULL;
  if (isActiveRank(sred->psubcomm)) {
    PetscInt mm;
    
    ierr = MatGetSize(Blocal,&mm,NULL);CHKERRQ(ierr);
    //ierr = MatCreateMPIMatConcatenateSeqMat(subcomm,Blocal,PETSC_DECIDE,reuse,&Bred);CHKERRQ(ierr);
    ierr = MatCreateMPIMatConcatenateSeqMat(subcomm,Blocal,mm,reuse,&Bred);CHKERRQ(ierr);
  }
  *A = Bred;
  
  ierr = ISDestroy(&iscol);CHKERRQ(ierr);
  ierr = MatDestroy(&Bperm);CHKERRQ(ierr);
  ierr = MatDestroyMatrices(1,&_Blocal);CHKERRQ(ierr);
  return(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCSemiRedundantMatNullSpaceCreate_dmda"
PetscErrorCode PCSemiRedundantMatNullSpaceCreate_dmda(PC pc,PC_SemiRedundant *sred,Mat sub_mat)
{
  PetscErrorCode   ierr;
  MatNullSpace     nullspace,sub_nullspace;
  Mat              A,B;
  PetscBool        has_const;
  PetscInt         i,k,n;
  const Vec        *vecs;
  Vec              *sub_vecs;
  MPI_Comm         subcomm;
  PC_SemiRedundant_DMDACtx *ctx;
  
  ierr = PCGetOperators(pc,&A,&B);CHKERRQ(ierr);
  ierr = MatGetNullSpace(B,&nullspace);CHKERRQ(ierr);
  if (!nullspace) return(0);
  
  PetscInfo(pc,"PCSemiRedundant: generating nullspace (DMDA)\n");
  ctx = (PC_SemiRedundant_DMDACtx*)sred->dm_ctx;
  subcomm = PetscSubcommChild(sred->psubcomm);
  ierr = MatNullSpaceGetVecs(nullspace,&has_const,&n,&vecs);CHKERRQ(ierr);
  
  if (isActiveRank(sred->psubcomm)) {
    sub_vecs = NULL;
    
    /* create new vectors */
    if (n != 0) {
      PetscMalloc(sizeof(Vec)*n,&sub_vecs);
      for (k=0; k<n; k++) {
        ierr = VecDuplicate(sred->xred,&sub_vecs[k]);CHKERRQ(ierr);
      }
    }
  }
  
  /* copy entries */
  for (k=0; k<n; k++) {
    const PetscScalar *x_array;
    PetscScalar *LA_sub_vec;
    PetscInt st,ed,bs;
    
    /* permute vector into ordering associated with re-partitioned dmda */
    ierr = MatMultTranspose(ctx->permutation,vecs[k],ctx->xp);CHKERRQ(ierr);
    
    /* pull in vector x->xtmp */
    ierr = VecScatterBegin(sred->scatter,ctx->xp,sred->xtmp,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(sred->scatter,ctx->xp,sred->xtmp,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    
    /* copy vector entires into xred */
    ierr = VecGetBlockSize(sred->xtmp,&bs);CHKERRQ(ierr);
    ierr = VecGetArrayRead(sred->xtmp,&x_array);CHKERRQ(ierr);
    if (sub_vecs[k]) {
      ierr = VecGetOwnershipRange(sub_vecs[k],&st,&ed);CHKERRQ(ierr);
      
      ierr = VecGetArray(sub_vecs[k],&LA_sub_vec);CHKERRQ(ierr);
      for (i=0; i<ed-st; i++) {
        LA_sub_vec[i] = x_array[i];
      }
      ierr = VecRestoreArray(sub_vecs[k],&LA_sub_vec);CHKERRQ(ierr);
    }
    ierr = VecRestoreArrayRead(sred->xtmp,&x_array);CHKERRQ(ierr);
  }
  
  if (isActiveRank(sred->psubcomm)) {
    /* create new nullspace for redundant object */
    ierr = MatNullSpaceCreate(subcomm,has_const,n,sub_vecs,&sub_nullspace);CHKERRQ(ierr);
    
    /* attach redundant nullspace to Bred */
    ierr = MatSetNullSpace(sub_mat,sub_nullspace);CHKERRQ(ierr);
  }
  return(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCApply_SemiRedundant_dmda"
PetscErrorCode PCApply_SemiRedundant_dmda(PC pc,Vec x,Vec y)
{
  PC_SemiRedundant  *sred = (PC_SemiRedundant*)pc->data;
  PetscErrorCode    ierr;
  Mat               perm;
  Vec               xtmp,xp,xred,yred;
  PetscInt          i,st,ed,bs;
  VecScatter        scatter;
  PetscScalar       *array;
  const PetscScalar *x_array;
  PC_SemiRedundant_DMDACtx *ctx;
  
  ctx = (PC_SemiRedundant_DMDACtx*)sred->dm_ctx;
  xtmp    = sred->xtmp;
  scatter = sred->scatter;
  xred    = sred->xred;
  yred    = sred->yred;
  perm  = ctx->permutation;
  xp    = ctx->xp;
  
  /* permute vector into ordering associated with re-partitioned dmda */
  ierr = MatMultTranspose(perm,x,xp);CHKERRQ(ierr);
  
  /* pull in vector x->xtmp */
  ierr = VecScatterBegin(scatter,xp,xtmp,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(scatter,xp,xtmp,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  
  /* copy vector entires into xred */
  ierr = VecGetBlockSize(xtmp,&bs);CHKERRQ(ierr);
  ierr = VecGetArrayRead(xtmp,&x_array);CHKERRQ(ierr);
  if (xred) {
    PetscScalar *LA_xred;
    
    ierr = VecGetOwnershipRange(xred,&st,&ed);CHKERRQ(ierr);
    
    ierr = VecGetArray(xred,&LA_xred);CHKERRQ(ierr);
    for (i=0; i<ed-st; i++) {
      LA_xred[i] = x_array[i];
    }
    ierr = VecRestoreArray(xred,&LA_xred);CHKERRQ(ierr);
  }
  ierr = VecRestoreArrayRead(xtmp,&x_array);CHKERRQ(ierr);
  
  /* solve */
  if (isActiveRank(sred->psubcomm)) {
    ierr = KSPSolve(sred->ksp,xred,yred);CHKERRQ(ierr);
  }
  
  /* return vector */
  ierr = VecGetBlockSize(xtmp,&bs);CHKERRQ(ierr);
  ierr = VecGetArray(xtmp,&array);CHKERRQ(ierr);
  if (yred) {
    const PetscScalar *LA_yred;
    
    ierr = VecGetOwnershipRange(yred,&st,&ed);CHKERRQ(ierr);
    
    ierr = VecGetArrayRead(yred,&LA_yred);CHKERRQ(ierr);
    for (i=0; i<ed-st; i++) {
      array[i] = LA_yred[i];
    }
    ierr = VecRestoreArrayRead(yred,&LA_yred);CHKERRQ(ierr);
  } else {
    for (i=0; i<bs; i++) {
      array[i] = 0.0;
    }
  }
  ierr = VecRestoreArray(xtmp,&array);CHKERRQ(ierr);
  
  ierr = VecScatterBegin(scatter,xtmp,xp,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(scatter,xtmp,xp,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  
  ierr = MatMult(perm,xp,y);CHKERRQ(ierr);
  return(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCReset_SemiRedundant_dmda"
PetscErrorCode PCReset_SemiRedundant_dmda(PC pc)
{
  PetscErrorCode   ierr;
  PC_SemiRedundant *sred = (PC_SemiRedundant*)pc->data;
  PC_SemiRedundant_DMDACtx *ctx;
  
  ctx = (PC_SemiRedundant_DMDACtx*)sred->dm_ctx;
  ierr = VecDestroy(&ctx->xp);CHKERRQ(ierr);
  if (ctx->permutation) { ierr = MatDestroy(&ctx->permutation);CHKERRQ(ierr);}
  if (ctx->dmrepart) { ierr = DMDestroy(&ctx->dmrepart);CHKERRQ(ierr); }
  if (ctx->range_i_re) PetscFree(ctx->range_i_re);
  if (ctx->range_j_re) PetscFree(ctx->range_j_re);
  if (ctx->range_k_re) PetscFree(ctx->range_k_re);
  if (ctx->start_i_re) PetscFree(ctx->start_i_re);
  if (ctx->start_j_re) PetscFree(ctx->start_j_re);
  if (ctx->start_k_re) PetscFree(ctx->start_k_re);
	return(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMView_DMDAShort_3d"
PetscErrorCode DMView_DMDAShort_3d(DM dm,PetscViewer v)
{
  PetscInt       M,N,P,m,n,p,ndof,nsw;
  MPI_Comm       comm;
  PetscMPIInt    size;
  PetscErrorCode ierr;
  
  ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = DMDAGetInfo(dm,NULL,&M,&N,&P,&m,&n,&p,&ndof,&nsw,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  PetscViewerASCIIPrintf(v,"DMDA Object: %d MPI processes\n",(int)size);
  PetscViewerASCIIPrintf(v,"  M %D N %D P %D m %D n %D p %D dof %D overlap %D\n",M,N,P,m,n,p,ndof,nsw);
	return(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMView_DMDAShort_2d"
PetscErrorCode DMView_DMDAShort_2d(DM dm,PetscViewer v)
{
  PetscInt       M,N,m,n,ndof,nsw;
  MPI_Comm       comm;
  PetscMPIInt    size;
  PetscErrorCode ierr;
  
  ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = DMDAGetInfo(dm,NULL,&M,&N,NULL,&m,&n,NULL,&ndof,&nsw,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  PetscViewerASCIIPrintf(v,"DMDA Object: %d MPI processes\n",(int)size);
  PetscViewerASCIIPrintf(v,"  M %D N %D m %D n %D dof %D overlap %D\n",M,N,m,n,ndof,nsw);
	return(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMView_DMDAShort"
PetscErrorCode DMView_DMDAShort(DM dm,PetscViewer v)
{
  PetscErrorCode ierr;
  PetscInt       dim;
  
  ierr = DMDAGetInfo(dm,&dim,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  switch (dim) {
    case 2: ierr = DMView_DMDAShort_2d(dm,v);CHKERRQ(ierr);
      break;
    case 3: ierr = DMView_DMDAShort_3d(dm,v);CHKERRQ(ierr);
      break;
  }
	return(0);
}

