
/*
     Code for creating scatters between vectors. This file
  includes the code for scattering between sequential vectors and
  some special cases for parallel scatters.
*/

#include <petsc/private/isimpl.h>
#include <petsc/private/vecimpl.h>    /*I   "petscvec.h"    I*/

#if defined(PETSC_HAVE_CUSP)
PETSC_INTERN PetscErrorCode VecScatterCUSPIndicesCreate_PtoP(PetscInt,PetscInt*,PetscInt,PetscInt*,PetscCUSPIndices*);
PETSC_INTERN PetscErrorCode VecScatterCUSPIndicesCreate_StoS(PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt*,PetscInt*,PetscCUSPIndices*);
PETSC_INTERN PetscErrorCode VecScatterCUSPIndicesDestroy(PetscCUSPIndices*);
PETSC_INTERN PetscErrorCode VecScatterCUSP_StoS(Vec,Vec,PetscCUSPIndices,InsertMode,ScatterMode);
#elif defined(PETSC_HAVE_VECCUDA)
PETSC_INTERN PetscErrorCode VecScatterCUDAIndicesCreate_PtoP(PetscInt,PetscInt*,PetscInt,PetscInt*,PetscCUDAIndices*);
PETSC_INTERN PetscErrorCode VecScatterCUDAIndicesCreate_StoS(PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt*,PetscInt*,PetscCUDAIndices*);
PETSC_INTERN PetscErrorCode VecScatterCUDAIndicesDestroy(PetscCUDAIndices*);
PETSC_INTERN PetscErrorCode VecScatterCUDA_StoS(Vec,Vec,PetscCUDAIndices,InsertMode,ScatterMode);
#endif

/* -------------------------------------------------------------------------*/

extern PetscErrorCode VecScatterCreate_PtoS_MPI1(PetscInt,const PetscInt*,PetscInt,const PetscInt*,Vec,Vec,PetscInt,VecScatter);
extern PetscErrorCode VecScatterCreate_PtoP_MPI1(PetscInt,const PetscInt*,PetscInt,const PetscInt*,Vec,Vec,PetscInt,VecScatter);
extern PetscErrorCode VecScatterCreate_StoP_MPI1(PetscInt,const PetscInt*,PetscInt,const PetscInt*,Vec,Vec,PetscInt,VecScatter);

/* =======================================================================*/
#define VEC_SEQ_ID 0
#define VEC_MPI_ID 1
#define IS_GENERAL_ID 0
#define IS_STRIDE_ID  1
#define IS_BLOCK_ID   2

/*
   Blocksizes we have optimized scatters for
*/

#define VecScatterOptimizedBS(mbs) (2 <= mbs)

PetscErrorCode VecScatterCreate_MPI1(Vec xin,IS ix,Vec yin,IS iy,VecScatter *newctx)
{
  VecScatter        ctx;
  PetscErrorCode    ierr;
  PetscMPIInt       size;
  PetscInt          xin_type = VEC_SEQ_ID,yin_type = VEC_SEQ_ID,*range;
  PetscInt          ix_type  = IS_GENERAL_ID,iy_type = IS_GENERAL_ID;
  MPI_Comm          comm,ycomm;
  PetscBool         totalv,ixblock,iyblock,iystride,islocal,cando,flag;
  IS                tix = 0,tiy = 0;

  PetscFunctionBegin;
  if (!ix && !iy) SETERRQ(PetscObjectComm((PetscObject)xin),PETSC_ERR_SUP,"Cannot pass default in for both input and output indices");
  //printf("VecScatterCreate_MPI1...\n");
  /*
      Determine if the vectors are "parallel", ie. it shares a comm with other processors, or
      sequential (it does not share a comm). The difference is that parallel vectors treat the
      index set as providing indices in the global parallel numbering of the vector, with
      sequential vectors treat the index set as providing indices in the local sequential
      numbering
  */
  ierr = PetscObjectGetComm((PetscObject)xin,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (size > 1) xin_type = VEC_MPI_ID;

  ierr = PetscObjectGetComm((PetscObject)yin,&ycomm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(ycomm,&size);CHKERRQ(ierr);
  if (size > 1) {comm = ycomm; yin_type = VEC_MPI_ID;}

  /* generate the Scatter context */
  ierr = PetscHeaderCreate(ctx,VEC_SCATTER_CLASSID,"VecScatter","VecScatter","Vec",comm,VecScatterDestroy,VecScatterViewMPI1);CHKERRQ(ierr);
  ctx->inuse               = PETSC_FALSE;

  ctx->beginandendtogether = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-vecscatter_merge",&ctx->beginandendtogether,NULL);CHKERRQ(ierr);
  if (ctx->beginandendtogether) {
    ierr = PetscInfo(ctx,"Using combined (merged) vector scatter begin and end\n");CHKERRQ(ierr);
  }
  ctx->packtogether = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-vecscatter_packtogether",&ctx->packtogether,NULL);CHKERRQ(ierr);
  if (ctx->packtogether) {
    ierr = PetscInfo(ctx,"Pack all messages before sending\n");CHKERRQ(ierr);
  }

  ierr = VecGetLocalSize(xin,&ctx->from_n);CHKERRQ(ierr);
  ierr = VecGetLocalSize(yin,&ctx->to_n);CHKERRQ(ierr);

  /*
      if ix or iy is not included; assume just grabbing entire vector
  */
  if (!ix && xin_type == VEC_SEQ_ID) {
    ierr = ISCreateStride(comm,ctx->from_n,0,1,&ix);CHKERRQ(ierr);
    tix  = ix;
  } else if (!ix && xin_type == VEC_MPI_ID) {
    if (yin_type == VEC_MPI_ID) {
      PetscInt ntmp, low;
      ierr = VecGetLocalSize(xin,&ntmp);CHKERRQ(ierr);
      ierr = VecGetOwnershipRange(xin,&low,NULL);CHKERRQ(ierr);
      ierr = ISCreateStride(comm,ntmp,low,1,&ix);CHKERRQ(ierr);
    } else {
      PetscInt Ntmp;
      ierr = VecGetSize(xin,&Ntmp);CHKERRQ(ierr);
      ierr = ISCreateStride(comm,Ntmp,0,1,&ix);CHKERRQ(ierr);
    }
    tix = ix;
  } else if (!ix) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"ix not given, but not Seq or MPI vector");

  if (!iy && yin_type == VEC_SEQ_ID) {
    ierr = ISCreateStride(comm,ctx->to_n,0,1,&iy);CHKERRQ(ierr);
    tiy  = iy;
  } else if (!iy && yin_type == VEC_MPI_ID) {
    if (xin_type == VEC_MPI_ID) {
      PetscInt ntmp, low;
      ierr = VecGetLocalSize(yin,&ntmp);CHKERRQ(ierr);
      ierr = VecGetOwnershipRange(yin,&low,NULL);CHKERRQ(ierr);
      ierr = ISCreateStride(comm,ntmp,low,1,&iy);CHKERRQ(ierr);
    } else {
      PetscInt Ntmp;
      ierr = VecGetSize(yin,&Ntmp);CHKERRQ(ierr);
      ierr = ISCreateStride(comm,Ntmp,0,1,&iy);CHKERRQ(ierr);
    }
    tiy = iy;
  } else if (!iy) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"iy not given, but not Seq or MPI vector");

  /*
     Determine types of index sets
  */
  ierr = PetscObjectTypeCompare((PetscObject)ix,ISBLOCK,&flag);CHKERRQ(ierr);
  if (flag) ix_type = IS_BLOCK_ID;
  ierr = PetscObjectTypeCompare((PetscObject)iy,ISBLOCK,&flag);CHKERRQ(ierr);
  if (flag) iy_type = IS_BLOCK_ID;
  ierr = PetscObjectTypeCompare((PetscObject)ix,ISSTRIDE,&flag);CHKERRQ(ierr);
  if (flag) ix_type = IS_STRIDE_ID;
  ierr = PetscObjectTypeCompare((PetscObject)iy,ISSTRIDE,&flag);CHKERRQ(ierr);
  if (flag) iy_type = IS_STRIDE_ID;

  /* ===========================================================================================================
        Check for special cases
     ==========================================================================================================*/
  /* ---------------------------------------------------------------------------*/
  if (xin_type == VEC_SEQ_ID && yin_type == VEC_SEQ_ID) {
    if (ix_type == IS_GENERAL_ID && iy_type == IS_GENERAL_ID) {
      PetscInt               nx,ny;
      const PetscInt         *idx,*idy;
      VecScatter_Seq_General *to = NULL,*from = NULL;

      ierr = ISGetLocalSize(ix,&nx);CHKERRQ(ierr);
      ierr = ISGetLocalSize(iy,&ny);CHKERRQ(ierr);
      if (nx != ny) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local scatter sizes don't match");
      ierr  = ISGetIndices(ix,&idx);CHKERRQ(ierr);
      ierr  = ISGetIndices(iy,&idy);CHKERRQ(ierr);
      ierr  = PetscMalloc2(1,&to,1,&from);CHKERRQ(ierr);
      ierr  = PetscMalloc2(nx,&to->vslots,nx,&from->vslots);CHKERRQ(ierr);
      to->n = nx;
#if defined(PETSC_USE_DEBUG)
      ierr = VecScatterCheckIndices_Private(ctx->to_n,ny,idy);CHKERRQ(ierr);
#endif
      ierr    = PetscMemcpy(to->vslots,idy,nx*sizeof(PetscInt));CHKERRQ(ierr);
      from->n = nx;
#if defined(PETSC_USE_DEBUG)
      ierr = VecScatterCheckIndices_Private(ctx->from_n,nx,idx);CHKERRQ(ierr);
#endif
      ierr              =  PetscMemcpy(from->vslots,idx,nx*sizeof(PetscInt));CHKERRQ(ierr);
      to->type          = VEC_SCATTER_SEQ_GENERAL;
      from->type        = VEC_SCATTER_SEQ_GENERAL;
      ctx->todata       = (void*)to;
      ctx->fromdata     = (void*)from;
      ctx->ops->begin   = VecScatterBegin_SGToSG;
      ctx->ops->end     = 0;
      ctx->ops->destroy = VecScatterDestroy_SGToSG;
      ctx->ops->copy    = VecScatterCopy_SGToSG;
      ctx->ops->view    = VecScatterView_SGToSG;
      ierr              = PetscInfo(xin,"Special case: sequential vector general scatter\n");CHKERRQ(ierr);
      goto functionend;
    } else if (ix_type == IS_STRIDE_ID &&  iy_type == IS_STRIDE_ID) {
      PetscInt              nx,ny,to_first,to_step,from_first,from_step;
      VecScatter_Seq_Stride *from8 = NULL,*to8 = NULL;

      ierr = ISGetLocalSize(ix,&nx);CHKERRQ(ierr);
      ierr = ISGetLocalSize(iy,&ny);CHKERRQ(ierr);
      if (nx != ny) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local scatter sizes don't match");
      ierr              = ISStrideGetInfo(iy,&to_first,&to_step);CHKERRQ(ierr);
      ierr              = ISStrideGetInfo(ix,&from_first,&from_step);CHKERRQ(ierr);
      ierr              = PetscMalloc2(1,&to8,1,&from8);CHKERRQ(ierr);
      to8->n            = nx;
      to8->first        = to_first;
      to8->step         = to_step;
      from8->n          = nx;
      from8->first      = from_first;
      from8->step       = from_step;
      to8->type         = VEC_SCATTER_SEQ_STRIDE;
      from8->type       = VEC_SCATTER_SEQ_STRIDE;
      ctx->todata       = (void*)to8;
      ctx->fromdata     = (void*)from8;
      ctx->ops->begin   = VecScatterBegin_SSToSS;
      ctx->ops->end     = 0;
      ctx->ops->destroy = VecScatterDestroy_SSToSS;
      ctx->ops->copy    = VecScatterCopy_SSToSS;
      ctx->ops->view    = VecScatterView_SSToSS;
      ierr          = PetscInfo(xin,"Special case: sequential vector stride to stride\n");CHKERRQ(ierr);
      goto functionend;
    } else if (ix_type == IS_GENERAL_ID && iy_type == IS_STRIDE_ID) {
      PetscInt               nx,ny,first,step;
      const PetscInt         *idx;
      VecScatter_Seq_General *from9 = NULL;
      VecScatter_Seq_Stride  *to9   = NULL;

      ierr = ISGetLocalSize(ix,&nx);CHKERRQ(ierr);
      ierr = ISGetIndices(ix,&idx);CHKERRQ(ierr);
      ierr = ISGetLocalSize(iy,&ny);CHKERRQ(ierr);
      ierr = ISStrideGetInfo(iy,&first,&step);CHKERRQ(ierr);
      if (nx != ny) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local scatter sizes don't match");
      ierr       = PetscMalloc2(1,&to9,1,&from9);CHKERRQ(ierr);
      ierr       = PetscMalloc1(nx,&from9->vslots);CHKERRQ(ierr);
      to9->n     = nx;
      to9->first = first;
      to9->step  = step;
      from9->n   = nx;
#if defined(PETSC_USE_DEBUG)
      ierr           = VecScatterCheckIndices_Private(ctx->from_n,nx,idx);CHKERRQ(ierr);
#endif
      ierr           = PetscMemcpy(from9->vslots,idx,nx*sizeof(PetscInt));CHKERRQ(ierr);
      ctx->todata    = (void*)to9; ctx->fromdata = (void*)from9;
      if (step == 1)  ctx->ops->begin = VecScatterBegin_SGToSS_Stride1;
      else            ctx->ops->begin = VecScatterBegin_SGToSS;
      ctx->ops->destroy   = VecScatterDestroy_SGToSS;
      ctx->ops->end       = 0;
      ctx->ops->copy      = VecScatterCopy_SGToSS;
      ctx->ops->view      = VecScatterView_SGToSS;
      to9->type      = VEC_SCATTER_SEQ_STRIDE;
      from9->type    = VEC_SCATTER_SEQ_GENERAL;
      ierr = PetscInfo(xin,"Special case: sequential vector general to stride\n");CHKERRQ(ierr);
      goto functionend;
    } else if (ix_type == IS_STRIDE_ID && iy_type == IS_GENERAL_ID) {
      PetscInt               nx,ny,first,step;
      const PetscInt         *idy;
      VecScatter_Seq_General *to10 = NULL;
      VecScatter_Seq_Stride  *from10 = NULL;

      ierr = ISGetLocalSize(ix,&nx);CHKERRQ(ierr);
      ierr = ISGetIndices(iy,&idy);CHKERRQ(ierr);
      ierr = ISGetLocalSize(iy,&ny);CHKERRQ(ierr);
      ierr = ISStrideGetInfo(ix,&first,&step);CHKERRQ(ierr);
      if (nx != ny) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local scatter sizes don't match");
      ierr = PetscMalloc2(1,&to10,1,&from10);CHKERRQ(ierr);
      ierr = PetscMalloc1(nx,&to10->vslots);CHKERRQ(ierr);
      from10->n     = nx;
      from10->first = first;
      from10->step  = step;
      to10->n       = nx;
#if defined(PETSC_USE_DEBUG)
      ierr = VecScatterCheckIndices_Private(ctx->to_n,ny,idy);CHKERRQ(ierr);
#endif
      ierr = PetscMemcpy(to10->vslots,idy,nx*sizeof(PetscInt));CHKERRQ(ierr);
      ctx->todata   = (void*)to10;
      ctx->fromdata = (void*)from10;
      if (step == 1) ctx->ops->begin = VecScatterBegin_SSToSG_Stride1;
      else           ctx->ops->begin = VecScatterBegin_SSToSG;
      ctx->ops->destroy = VecScatterDestroy_SSToSG;
      ctx->ops->end     = 0;
      ctx->ops->copy    = 0;
      ctx->ops->view    = VecScatterView_SSToSG;
      to10->type   = VEC_SCATTER_SEQ_GENERAL;
      from10->type = VEC_SCATTER_SEQ_STRIDE;
      ierr = PetscInfo(xin,"Special case: sequential vector stride to general\n");CHKERRQ(ierr);
      goto functionend;
    } else {
      PetscInt               nx,ny;
      const PetscInt         *idx,*idy;
      VecScatter_Seq_General *to11 = NULL,*from11 = NULL;
      PetscBool              idnx,idny;

      ierr = ISGetLocalSize(ix,&nx);CHKERRQ(ierr);
      ierr = ISGetLocalSize(iy,&ny);CHKERRQ(ierr);
      if (nx != ny) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local scatter sizes don't match, in %D out %D",nx,ny);

      ierr = ISIdentity(ix,&idnx);CHKERRQ(ierr);
      ierr = ISIdentity(iy,&idny);CHKERRQ(ierr);
      if (idnx && idny) {
        VecScatter_Seq_Stride *to13 = NULL,*from13 = NULL;
        ierr          = PetscMalloc2(1,&to13,1,&from13);CHKERRQ(ierr);
        to13->n       = nx;
        to13->first   = 0;
        to13->step    = 1;
        from13->n     = nx;
        from13->first = 0;
        from13->step  = 1;
        to13->type    = VEC_SCATTER_SEQ_STRIDE;
        from13->type  = VEC_SCATTER_SEQ_STRIDE;
        ctx->todata   = (void*)to13;
        ctx->fromdata = (void*)from13;
        ctx->ops->begin    = VecScatterBegin_SSToSS;
        ctx->ops->end      = 0;
        ctx->ops->destroy  = VecScatterDestroy_SSToSS;
        ctx->ops->copy     = VecScatterCopy_SSToSS;
        ctx->ops->view     = VecScatterView_SSToSS;
        ierr = PetscInfo(xin,"Special case: sequential copy\n");CHKERRQ(ierr);
        goto functionend;
      }

      ierr = ISGetIndices(iy,&idy);CHKERRQ(ierr);
      ierr = ISGetIndices(ix,&idx);CHKERRQ(ierr);
      ierr = PetscMalloc2(1,&to11,1,&from11);CHKERRQ(ierr);
      ierr = PetscMalloc2(nx,&to11->vslots,nx,&from11->vslots);CHKERRQ(ierr);
      to11->n = nx;
#if defined(PETSC_USE_DEBUG)
      ierr = VecScatterCheckIndices_Private(ctx->to_n,ny,idy);CHKERRQ(ierr);
#endif
      ierr = PetscMemcpy(to11->vslots,idy,nx*sizeof(PetscInt));CHKERRQ(ierr);
      from11->n = nx;
#if defined(PETSC_USE_DEBUG)
      ierr = VecScatterCheckIndices_Private(ctx->from_n,nx,idx);CHKERRQ(ierr);
#endif
      ierr = PetscMemcpy(from11->vslots,idx,nx*sizeof(PetscInt));CHKERRQ(ierr);
      to11->type        = VEC_SCATTER_SEQ_GENERAL;
      from11->type      = VEC_SCATTER_SEQ_GENERAL;
      ctx->todata       = (void*)to11;
      ctx->fromdata     = (void*)from11;
      ctx->ops->begin   = VecScatterBegin_SGToSG;
      ctx->ops->end     = 0;
      ctx->ops->destroy = VecScatterDestroy_SGToSG;
      ctx->ops->copy    = VecScatterCopy_SGToSG;
      ctx->ops->view    = VecScatterView_SGToSG;
      ierr = ISRestoreIndices(ix,&idx);CHKERRQ(ierr);
      ierr = ISRestoreIndices(iy,&idy);CHKERRQ(ierr);
      ierr = PetscInfo(xin,"Sequential vector scatter with block indices\n");CHKERRQ(ierr);
      goto functionend;
    }
  }
  /* ---------------------------------------------------------------------------*/
  if (xin_type == VEC_MPI_ID && yin_type == VEC_SEQ_ID) {

    /* ===========================================================================================================
          Check for special cases
       ==========================================================================================================*/
    islocal = PETSC_FALSE;
    /* special case extracting (subset of) local portion */
    if (ix_type == IS_STRIDE_ID && iy_type == IS_STRIDE_ID) {
      PetscInt              nx,ny,to_first,to_step,from_first,from_step;
      PetscInt              start,end,min,max;
      VecScatter_Seq_Stride *from12 = NULL,*to12 = NULL;

      ierr = VecGetOwnershipRange(xin,&start,&end);CHKERRQ(ierr);
      ierr = ISGetLocalSize(ix,&nx);CHKERRQ(ierr);
      ierr = ISStrideGetInfo(ix,&from_first,&from_step);CHKERRQ(ierr);
      ierr = ISGetLocalSize(iy,&ny);CHKERRQ(ierr);
      ierr = ISStrideGetInfo(iy,&to_first,&to_step);CHKERRQ(ierr);
      if (nx != ny) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local scatter sizes don't match");
      ierr = ISGetMinMax(ix,&min,&max);CHKERRQ(ierr);
      if (min >= start && max < end) islocal = PETSC_TRUE;
      else islocal = PETSC_FALSE;
      /* cannot use MPIU_Allreduce() since this call matches with the MPI_Allreduce() in the else statement below */
      ierr = MPI_Allreduce(&islocal,&cando,1,MPIU_BOOL,MPI_LAND,PetscObjectComm((PetscObject)xin));CHKERRQ(ierr);
      if (cando) {
        ierr               = PetscMalloc2(1,&to12,1,&from12);CHKERRQ(ierr);
        to12->n            = nx;
        to12->first        = to_first;
        to12->step         = to_step;
        from12->n          = nx;
        from12->first      = from_first-start;
        from12->step       = from_step;
        to12->type         = VEC_SCATTER_SEQ_STRIDE;
        from12->type       = VEC_SCATTER_SEQ_STRIDE;
        ctx->todata        = (void*)to12;
        ctx->fromdata      = (void*)from12;
        ctx->ops->begin    = VecScatterBegin_SSToSS;
        ctx->ops->end      = 0;
        ctx->ops->destroy  = VecScatterDestroy_SSToSS;
        ctx->ops->copy     = VecScatterCopy_SSToSS;
        ctx->ops->view     = VecScatterView_SSToSS;
        ierr = PetscInfo(xin,"Special case: processors only getting local values\n");CHKERRQ(ierr);
        goto functionend;
      }
    } else {
      ierr = MPI_Allreduce(&islocal,&cando,1,MPIU_BOOL,MPI_LAND,PetscObjectComm((PetscObject)xin));CHKERRQ(ierr);
    }

    /* test for special case of all processors getting entire vector */
    /* contains check that PetscMPIInt can handle the sizes needed */
    totalv = PETSC_FALSE;
    if (ix_type == IS_STRIDE_ID && iy_type == IS_STRIDE_ID) {
      PetscInt             i,nx,ny,to_first,to_step,from_first,from_step,N;
      PetscMPIInt          *count = NULL,*displx;
      VecScatter_MPI_ToAll *sto   = NULL;

      ierr = ISGetLocalSize(ix,&nx);CHKERRQ(ierr);
      ierr = ISStrideGetInfo(ix,&from_first,&from_step);CHKERRQ(ierr);
      ierr = ISGetLocalSize(iy,&ny);CHKERRQ(ierr);
      ierr = ISStrideGetInfo(iy,&to_first,&to_step);CHKERRQ(ierr);
      if (nx != ny) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local scatter sizes don't match");
      ierr = VecGetSize(xin,&N);CHKERRQ(ierr);
      if (nx != N) totalv = PETSC_FALSE;
      else if (from_first == 0 && from_step == 1 && from_first == to_first && from_step == to_step) totalv = PETSC_TRUE;
      else totalv = PETSC_FALSE;
      /* cannot use MPIU_Allreduce() since this call matches with the MPI_Allreduce() in the else statement below */
      ierr = MPI_Allreduce(&totalv,&cando,1,MPIU_BOOL,MPI_LAND,PetscObjectComm((PetscObject)xin));CHKERRQ(ierr);

#if defined(PETSC_USE_64BIT_INDICES)
      if (cando && (yin->map->N < PETSC_MPI_INT_MAX)) {
#else
      if (cando) {
#endif
        ierr  = MPI_Comm_size(PetscObjectComm((PetscObject)ctx),&size);CHKERRQ(ierr);
        ierr  = PetscMalloc3(1,&sto,size,&count,size,&displx);CHKERRQ(ierr);
        range = xin->map->range;
        for (i=0; i<size; i++) {
          ierr = PetscMPIIntCast(range[i+1] - range[i],count+i);CHKERRQ(ierr);
          ierr = PetscMPIIntCast(range[i],displx+i);CHKERRQ(ierr);
        }
        sto->count        = count;
        sto->displx       = displx;
        sto->work1        = 0;
        sto->work2        = 0;
        sto->type         = VEC_SCATTER_MPI_TOALL;
        ctx->todata       = (void*)sto;
        ctx->fromdata     = 0;
        ctx->ops->begin   = VecScatterBegin_MPI_ToAll;
        ctx->ops->end     = 0;
        ctx->ops->destroy = VecScatterDestroy_MPI_ToAll;
        ctx->ops->copy    = VecScatterCopy_MPI_ToAll;
        ctx->ops->view    = VecScatterView_MPI_ToAll;
        ierr = PetscInfo(xin,"Special case: all processors get entire parallel vector\n");CHKERRQ(ierr);
        goto functionend;
      }
    } else {
      ierr = MPI_Allreduce(&totalv,&cando,1,MPIU_BOOL,MPI_LAND,PetscObjectComm((PetscObject)xin));CHKERRQ(ierr);
    }

    /* test for special case of processor 0 getting entire vector */
    /* contains check that PetscMPIInt can handle the sizes needed */
    totalv = PETSC_FALSE;
    if (ix_type == IS_STRIDE_ID && iy_type == IS_STRIDE_ID) {
      PetscInt             i,nx,ny,to_first,to_step,from_first,from_step,N;
      PetscMPIInt          rank,*count = NULL,*displx;
      VecScatter_MPI_ToAll *sto = NULL;

      ierr = PetscObjectGetComm((PetscObject)xin,&comm);CHKERRQ(ierr);
      ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
      ierr = ISGetLocalSize(ix,&nx);CHKERRQ(ierr);
      ierr = ISStrideGetInfo(ix,&from_first,&from_step);CHKERRQ(ierr);
      ierr = ISGetLocalSize(iy,&ny);CHKERRQ(ierr);
      ierr = ISStrideGetInfo(iy,&to_first,&to_step);CHKERRQ(ierr);
      if (nx != ny) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local scatter sizes don't match");
      if (!rank) {
        ierr = VecGetSize(xin,&N);CHKERRQ(ierr);
        if (nx != N) totalv = PETSC_FALSE;
        else if (from_first == 0        && from_step == 1 &&
                 from_first == to_first && from_step == to_step) totalv = PETSC_TRUE;
        else totalv = PETSC_FALSE;
      } else {
        if (!nx) totalv = PETSC_TRUE;
        else     totalv = PETSC_FALSE;
      }
      /* cannot use MPIU_Allreduce() since this call matches with the MPI_Allreduce() in the else statement below */
      ierr = MPI_Allreduce(&totalv,&cando,1,MPIU_BOOL,MPI_LAND,PetscObjectComm((PetscObject)xin));CHKERRQ(ierr);

#if defined(PETSC_USE_64BIT_INDICES)
      if (cando && (yin->map->N < PETSC_MPI_INT_MAX)) {
#else
      if (cando) {
#endif
        ierr  = MPI_Comm_size(PetscObjectComm((PetscObject)ctx),&size);CHKERRQ(ierr);
        ierr  = PetscMalloc3(1,&sto,size,&count,size,&displx);CHKERRQ(ierr);
        range = xin->map->range;
        for (i=0; i<size; i++) {
          ierr = PetscMPIIntCast(range[i+1] - range[i],count+i);CHKERRQ(ierr);
          ierr = PetscMPIIntCast(range[i],displx+i);CHKERRQ(ierr);
        }
        sto->count        = count;
        sto->displx       = displx;
        sto->work1        = 0;
        sto->work2        = 0;
        sto->type         = VEC_SCATTER_MPI_TOONE;
        ctx->todata       = (void*)sto;
        ctx->fromdata     = 0;
        ctx->ops->begin   = VecScatterBegin_MPI_ToOne;
        ctx->ops->end     = 0;
        ctx->ops->destroy = VecScatterDestroy_MPI_ToAll;
        ctx->ops->copy    = VecScatterCopy_MPI_ToAll;
        ctx->ops->view    = VecScatterView_MPI_ToAll;
        ierr = PetscInfo(xin,"Special case: processor zero gets entire parallel vector, rest get none\n");CHKERRQ(ierr);
        goto functionend;
      }
    } else {
      ierr = MPI_Allreduce(&totalv,&cando,1,MPIU_BOOL,MPI_LAND,PetscObjectComm((PetscObject)xin));CHKERRQ(ierr);
    }

    ierr = PetscObjectTypeCompare((PetscObject)ix,ISBLOCK,&ixblock);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)iy,ISBLOCK,&iyblock);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)iy,ISSTRIDE,&iystride);CHKERRQ(ierr);
    if (ixblock) {
      /* special case block to block */
      if (iyblock) {
        PetscInt       nx,ny,bsx,bsy;
        const PetscInt *idx,*idy;
        ierr = ISGetBlockSize(iy,&bsy);CHKERRQ(ierr);
        ierr = ISGetBlockSize(ix,&bsx);CHKERRQ(ierr);
        if (bsx == bsy && VecScatterOptimizedBS(bsx)) {
          ierr = ISBlockGetLocalSize(ix,&nx);CHKERRQ(ierr);
          ierr = ISBlockGetIndices(ix,&idx);CHKERRQ(ierr);
          ierr = ISBlockGetLocalSize(iy,&ny);CHKERRQ(ierr);
          ierr = ISBlockGetIndices(iy,&idy);CHKERRQ(ierr);
          if (nx != ny) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local scatter sizes don't match");
          ierr = VecScatterCreate_PtoS_MPI1(nx,idx,ny,idy,xin,yin,bsx,ctx);CHKERRQ(ierr);
          ierr = ISBlockRestoreIndices(ix,&idx);CHKERRQ(ierr);
          ierr = ISBlockRestoreIndices(iy,&idy);CHKERRQ(ierr);
          ierr = PetscInfo(xin,"Special case: blocked indices\n");CHKERRQ(ierr);
          goto functionend;
        }
        /* special case block to stride */
      } else if (iystride) {
        PetscInt ystart,ystride,ysize,bsx;
        ierr = ISStrideGetInfo(iy,&ystart,&ystride);CHKERRQ(ierr);
        ierr = ISGetLocalSize(iy,&ysize);CHKERRQ(ierr);
        ierr = ISGetBlockSize(ix,&bsx);CHKERRQ(ierr);
        /* see if stride index set is equivalent to block index set */
        if (VecScatterOptimizedBS(bsx) && ((ystart % bsx) == 0) && (ystride == 1) && ((ysize % bsx) == 0)) {
          PetscInt       nx,il,*idy;
          const PetscInt *idx;
          ierr = ISBlockGetLocalSize(ix,&nx);CHKERRQ(ierr);
          ierr = ISBlockGetIndices(ix,&idx);CHKERRQ(ierr);
          if (ysize != bsx*nx) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local scatter sizes don't match");
          ierr = PetscMalloc1(nx,&idy);CHKERRQ(ierr);
          if (nx) {
            idy[0] = ystart/bsx;
            for (il=1; il<nx; il++) idy[il] = idy[il-1] + 1;
          }
          ierr = VecScatterCreate_PtoS_MPI1(nx,idx,nx,idy,xin,yin,bsx,ctx);CHKERRQ(ierr);
          ierr = PetscFree(idy);CHKERRQ(ierr);
          ierr = ISBlockRestoreIndices(ix,&idx);CHKERRQ(ierr);
          ierr = PetscInfo(xin,"Special case: blocked indices to stride\n");CHKERRQ(ierr);
          goto functionend;
        }
      }
    }
    /* left over general case */
    {
      PetscInt       nx,ny;
      const PetscInt *idx,*idy;
      ierr = ISGetLocalSize(ix,&nx);CHKERRQ(ierr);
      ierr = ISGetIndices(ix,&idx);CHKERRQ(ierr);
      ierr = ISGetLocalSize(iy,&ny);CHKERRQ(ierr);
      ierr = ISGetIndices(iy,&idy);CHKERRQ(ierr);
      if (nx != ny) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local scatter sizes don't match (%d %d)",nx,ny);

      ierr = VecScatterCreate_PtoS_MPI1(nx,idx,ny,idy,xin,yin,1,ctx);CHKERRQ(ierr);
      ierr = ISRestoreIndices(ix,&idx);CHKERRQ(ierr);
      ierr = ISRestoreIndices(iy,&idy);CHKERRQ(ierr);
      ierr = PetscInfo(xin,"General case: MPI to Seq\n");CHKERRQ(ierr);
      goto functionend;
    }
  }
  /* ---------------------------------------------------------------------------*/
  if (xin_type == VEC_SEQ_ID && yin_type == VEC_MPI_ID) {
    /* ===========================================================================================================
          Check for special cases
       ==========================================================================================================*/
    /* special case local copy portion */
    islocal = PETSC_FALSE;
    if (ix_type == IS_STRIDE_ID && iy_type == IS_STRIDE_ID) {
      PetscInt              nx,ny,to_first,to_step,from_step,start,end,from_first,min,max;
      VecScatter_Seq_Stride *from = NULL,*to = NULL;

      ierr = VecGetOwnershipRange(yin,&start,&end);CHKERRQ(ierr);
      ierr = ISGetLocalSize(ix,&nx);CHKERRQ(ierr);
      ierr = ISStrideGetInfo(ix,&from_first,&from_step);CHKERRQ(ierr);
      ierr = ISGetLocalSize(iy,&ny);CHKERRQ(ierr);
      ierr = ISStrideGetInfo(iy,&to_first,&to_step);CHKERRQ(ierr);
      if (nx != ny) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local scatter sizes don't match");
      ierr = ISGetMinMax(iy,&min,&max);CHKERRQ(ierr);
      if (min >= start && max < end) islocal = PETSC_TRUE;
      else islocal = PETSC_FALSE;
      /* cannot use MPIU_Allreduce() since this call matches with the MPI_Allreduce() in the else statement below */
      ierr = MPI_Allreduce(&islocal,&cando,1,MPIU_BOOL,MPI_LAND,PetscObjectComm((PetscObject)yin));CHKERRQ(ierr);
      if (cando) {
        ierr              = PetscMalloc2(1,&to,1,&from);CHKERRQ(ierr);
        to->n             = nx;
        to->first         = to_first-start;
        to->step          = to_step;
        from->n           = nx;
        from->first       = from_first;
        from->step        = from_step;
        to->type          = VEC_SCATTER_SEQ_STRIDE;
        from->type        = VEC_SCATTER_SEQ_STRIDE;
        ctx->todata       = (void*)to;
        ctx->fromdata     = (void*)from;
        ctx->ops->begin   = VecScatterBegin_SSToSS;
        ctx->ops->end     = 0;
        ctx->ops->destroy = VecScatterDestroy_SSToSS;
        ctx->ops->copy    = VecScatterCopy_SSToSS;
        ctx->ops->view    = VecScatterView_SSToSS;
        ierr          = PetscInfo(xin,"Special case: sequential stride to MPI stride\n");CHKERRQ(ierr);
        goto functionend;
      }
    } else {
      ierr = MPI_Allreduce(&islocal,&cando,1,MPIU_BOOL,MPI_LAND,PetscObjectComm((PetscObject)yin));CHKERRQ(ierr);
    }
    /* special case block to stride */
    if (ix_type == IS_BLOCK_ID && iy_type == IS_STRIDE_ID) {
      PetscInt ystart,ystride,ysize,bsx;
      ierr = ISStrideGetInfo(iy,&ystart,&ystride);CHKERRQ(ierr);
      ierr = ISGetLocalSize(iy,&ysize);CHKERRQ(ierr);
      ierr = ISGetBlockSize(ix,&bsx);CHKERRQ(ierr);
      /* see if stride index set is equivalent to block index set */
      if (VecScatterOptimizedBS(bsx) && ((ystart % bsx) == 0) && (ystride == 1) && ((ysize % bsx) == 0)) {
        PetscInt       nx,il,*idy;
        const PetscInt *idx;
        ierr = ISBlockGetLocalSize(ix,&nx);CHKERRQ(ierr);
        ierr = ISBlockGetIndices(ix,&idx);CHKERRQ(ierr);
        if (ysize != bsx*nx) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local scatter sizes don't match");
        ierr = PetscMalloc1(nx,&idy);CHKERRQ(ierr);
        if (nx) {
          idy[0] = ystart/bsx;
          for (il=1; il<nx; il++) idy[il] = idy[il-1] + 1;
        }
        ierr = VecScatterCreate_StoP_MPI1(nx,idx,nx,idy,xin,yin,bsx,ctx);CHKERRQ(ierr);
        ierr = PetscFree(idy);CHKERRQ(ierr);
        ierr = ISBlockRestoreIndices(ix,&idx);CHKERRQ(ierr);
        ierr = PetscInfo(xin,"Special case: Blocked indices to stride\n");CHKERRQ(ierr);
        goto functionend;
      }
    }

    /* general case */
    {
      PetscInt       nx,ny;
      const PetscInt *idx,*idy;
      ierr = ISGetLocalSize(ix,&nx);CHKERRQ(ierr);
      ierr = ISGetIndices(ix,&idx);CHKERRQ(ierr);
      ierr = ISGetLocalSize(iy,&ny);CHKERRQ(ierr);
      ierr = ISGetIndices(iy,&idy);CHKERRQ(ierr);
      if (nx != ny) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local scatter sizes don't match");
      ierr = VecScatterCreate_StoP_MPI1(nx,idx,ny,idy,xin,yin,1,ctx);CHKERRQ(ierr);
      ierr = ISRestoreIndices(ix,&idx);CHKERRQ(ierr);
      ierr = ISRestoreIndices(iy,&idy);CHKERRQ(ierr);
      ierr = PetscInfo(xin,"General case: Seq to MPI\n");CHKERRQ(ierr);
      goto functionend;
    }
  }
  /* ---------------------------------------------------------------------------*/
  if (xin_type == VEC_MPI_ID && yin_type == VEC_MPI_ID) {
    /* no special cases for now */
    PetscInt       nx,ny;
    const PetscInt *idx,*idy;
    ierr = ISGetLocalSize(ix,&nx);CHKERRQ(ierr);
    ierr = ISGetIndices(ix,&idx);CHKERRQ(ierr);
    ierr = ISGetLocalSize(iy,&ny);CHKERRQ(ierr);
    ierr = ISGetIndices(iy,&idy);CHKERRQ(ierr);
    if (nx != ny) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local scatter sizes don't match");
    ierr = VecScatterCreate_PtoP_MPI1(nx,idx,ny,idy,xin,yin,1,ctx);CHKERRQ(ierr);
    ierr = ISRestoreIndices(ix,&idx);CHKERRQ(ierr);
    ierr = ISRestoreIndices(iy,&idy);CHKERRQ(ierr);
    ierr = PetscInfo(xin,"General case: MPI to MPI\n");CHKERRQ(ierr);
    goto functionend;
  }

  functionend:
  *newctx = ctx;
  ierr = ISDestroy(&tix);CHKERRQ(ierr);
  ierr = ISDestroy(&tiy);CHKERRQ(ierr);
  ierr = VecScatterViewFromOptions(ctx,NULL,"-vecscatter_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------*/
/*@
   VecScatterGetMergedMPI1 - Returns true if the scatter is completed in the VecScatterBeginMPI1()
      and the VecScatterEnd() does nothing

   Not Collective

   Input Parameter:
.   ctx - scatter context created with VecScatterCreateMPI1()

   Output Parameter:
.   flg - PETSC_TRUE if the VecScatterBeginMPI1/End() are all done during the VecScatterBeginMPI1()

   Level: developer

.seealso: VecScatterCreate(), VecScatterEnd(), VecScatterBeginMPI1()
@*/
PetscErrorCode  VecScatterGetMergedMPI1(VecScatter ctx,PetscBool  *flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ctx,VEC_SCATTER_CLASSID,1);
  *flg = ctx->beginandendtogether;
  PetscFunctionReturn(0);
}

/*@
   VecScatterBeginMPI1 - Begins a generalized scatter from one vector to
   another. Complete the scattering phase with VecScatterEnd().

   Neighbor-wise Collective on VecScatter and Vec

   Input Parameters:
+  inctx - scatter context generated by VecScatterCreate_MPI1()
.  x - the vector from which we scatter
.  y - the vector to which we scatter
.  addv - either ADD_VALUES or INSERT_VALUES, with INSERT_VALUES mode any location
          not scattered to retains its old value; i.e. the vector is NOT first zeroed.
-  mode - the scattering mode, usually SCATTER_FORWARD.  The available modes are:
    SCATTER_FORWARD or SCATTER_REVERSE


   Level: intermediate

   Options Database: See VecScatterCreate()

   Notes:
   The vectors x and y need not be the same vectors used in the call
   to VecScatterCreate(), but x must have the same parallel data layout
   as that passed in as the x to VecScatterCreate(), similarly for the y.
   Most likely they have been obtained from VecDuplicate().

   You cannot change the values in the input vector between the calls to VecScatterBegin()
   and VecScatterEnd().

   If you use SCATTER_REVERSE the two arguments x and y should be reversed, from
   the SCATTER_FORWARD.

   y[iy[i]] = x[ix[i]], for i=0,...,ni-1

   This scatter is far more general than the conventional
   scatter, since it can be a gather or a scatter or a combination,
   depending on the indices ix and iy.  If x is a parallel vector and y
   is sequential, VecScatterBeginMPI1() can serve to gather values to a
   single processor.  Similarly, if y is parallel and x sequential, the
   routine can scatter from one processor to many processors.

   Concepts: scatter^between vectors
   Concepts: gather^between vectors

.seealso: VecScatterCreate(), VecScatterEnd()
@*/
PetscErrorCode  VecScatterBeginMPI1(VecScatter inctx,Vec x,Vec y,InsertMode addv,ScatterMode mode)
{
  PetscErrorCode ierr;
#if defined(PETSC_USE_DEBUG)
  PetscInt       to_n,from_n;
#endif
  PetscFunctionBegin;
  PetscValidHeaderSpecific(inctx,VEC_SCATTER_CLASSID,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidHeaderSpecific(y,VEC_CLASSID,3);
  if (inctx->inuse) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE," Scatter ctx already in use");

#if defined(PETSC_USE_DEBUG)
  /*
     Error checking to make sure these vectors match the vectors used
   to create the vector scatter context. -1 in the from_n and to_n indicate the
   vector lengths are unknown (for example with mapped scatters) and thus
   no error checking is performed.
  */
  if (inctx->from_n >= 0 && inctx->to_n >= 0) {
    ierr = VecGetLocalSize(x,&from_n);CHKERRQ(ierr);
    ierr = VecGetLocalSize(y,&to_n);CHKERRQ(ierr);
    if (mode & SCATTER_REVERSE) {
      if (to_n != inctx->from_n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Vector wrong size %D for scatter %D (scatter reverse and vector to != ctx from size)",to_n,inctx->from_n);
      if (from_n != inctx->to_n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Vector wrong size %D for scatter %D (scatter reverse and vector from != ctx to size)",from_n,inctx->to_n);
    } else {
      if (to_n != inctx->to_n)     SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Vector wrong size %D for scatter %D (scatter forward and vector to != ctx to size)",to_n,inctx->to_n);
      if (from_n != inctx->from_n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Vector wrong size %D for scatter %D (scatter forward and vector from != ctx from size)",from_n,inctx->from_n);
    }
  }
#endif

  inctx->inuse = PETSC_TRUE;
  ierr = PetscLogEventBarrierBegin(VEC_ScatterBarrier,0,0,0,0,PetscObjectComm((PetscObject)inctx));CHKERRQ(ierr);
  ierr = (*inctx->ops->begin)(inctx,x,y,addv,mode);CHKERRQ(ierr);
  if (inctx->beginandendtogether && inctx->ops->end) {
    inctx->inuse = PETSC_FALSE;
    ierr = (*inctx->ops->end)(inctx,x,y,addv,mode);CHKERRQ(ierr);
  }
  ierr = PetscLogEventBarrierEnd(VEC_ScatterBarrier,0,0,0,0,PetscObjectComm((PetscObject)inctx));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------*/
/*@
   VecScatterEndMPI1 - Ends a generalized scatter from one vector to another.  Call
   after first calling VecScatterBeginMPI1().

   Neighbor-wise Collective on VecScatter and Vec

   Input Parameters:
+  ctx - scatter context generated by VecScatterCreate()
.  x - the vector from which we scatter
.  y - the vector to which we scatter
.  addv - either ADD_VALUES or INSERT_VALUES.
-  mode - the scattering mode, usually SCATTER_FORWARD.  The available modes are:
     SCATTER_FORWARD, SCATTER_REVERSE

   Level: intermediate

   Notes:
   If you use SCATTER_REVERSE the arguments x and y should be reversed, from the SCATTER_FORWARD.

   y[iy[i]] = x[ix[i]], for i=0,...,ni-1

.seealso: VecScatterBeginMPI1(), VecScatterCreate()
@*/
PetscErrorCode  VecScatterEndMPI1(VecScatter ctx,Vec x,Vec y,InsertMode addv,ScatterMode mode)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ctx,VEC_SCATTER_CLASSID,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidHeaderSpecific(y,VEC_CLASSID,3);
  ctx->inuse = PETSC_FALSE;
  if (!ctx->ops->end) PetscFunctionReturn(0);
  if (!ctx->beginandendtogether) {
    ierr = PetscLogEventBegin(VEC_ScatterEnd,ctx,x,y,0);CHKERRQ(ierr);
    ierr = (*(ctx)->ops->end)(ctx,x,y,addv,mode);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(VEC_ScatterEnd,ctx,x,y,0);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
   VecScatterCopyMPI1 - Makes a copy of a scatter context.

   Collective on VecScatter

   Input Parameter:
.  sctx - the scatter context

   Output Parameter:
.  ctx - the context copy

   Level: advanced

.seealso: VecScatterCreate(), VecScatterDestroy()
@*/
PetscErrorCode  VecScatterCopyMPI1(VecScatter sctx,VecScatter *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sctx,VEC_SCATTER_CLASSID,1);
  PetscValidPointer(ctx,2);
  if (!sctx->ops->copy) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot copy this type");
  ierr = PetscHeaderCreate(*ctx,VEC_SCATTER_CLASSID,"VecScatter","VecScatter","Vec",PetscObjectComm((PetscObject)sctx),VecScatterDestroy,VecScatterViewMPI1);CHKERRQ(ierr);
  (*ctx)->to_n   = sctx->to_n;
  (*ctx)->from_n = sctx->from_n;
  ierr = (*sctx->ops->copy)(sctx,*ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/* ------------------------------------------------------------------*/
/*@C
   VecScatterViewMPI1 - Views a vector scatter context.

   Collective on VecScatter

   Input Parameters:
+  ctx - the scatter context
-  viewer - the viewer for displaying the context

   Level: intermediate

C@*/
PetscErrorCode  VecScatterViewMPI1(VecScatter ctx,PetscViewer viewer)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ctx,VEC_SCATTER_CLASSID,1);
  if (!viewer) {
    ierr = PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)ctx),&viewer);CHKERRQ(ierr);
  }
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  if (ctx->ops->view) {
    ierr = (*ctx->ops->view)(ctx,viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
   VecScatterRemapMPI1 - Remaps the "from" and "to" indices in a
   vector scatter context. FOR EXPERTS ONLY!

   Collective on VecScatter

   Input Parameters:
+  scat - vector scatter context
.  from - remapping for "from" indices (may be NULL)
-  to   - remapping for "to" indices (may be NULL)

   Level: developer

   Notes: In the parallel case the todata is actually the indices
          from which the data is TAKEN! The from stuff is where the
          data is finally put. This is VERY VERY confusing!

          In the sequential case the todata is the indices where the
          data is put and the fromdata is where it is taken from.
          This is backwards from the paralllel case! CRY! CRY! CRY!

@*/
PetscErrorCode VecScatterRemapMPI1(VecScatter scat,PetscInt *rto,PetscInt *rfrom)
{
  VecScatter_Seq_General *to,*from;
  VecScatter_MPI_General *mto;
  PetscInt               i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(scat,VEC_SCATTER_CLASSID,1);
  if (rto)   PetscValidIntPointer(rto,2);
  if (rfrom) PetscValidIntPointer(rfrom,3);

  from = (VecScatter_Seq_General*)scat->fromdata;
  mto  = (VecScatter_MPI_General*)scat->todata;

  if (mto->type == VEC_SCATTER_MPI_TOALL) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Not for to all scatter");

  if (rto) {
    if (mto->type == VEC_SCATTER_MPI_GENERAL) {
      /* handle off processor parts */
      for (i=0; i<mto->starts[mto->n]; i++) mto->indices[i] = rto[mto->indices[i]];

      /* handle local part */
      to = &mto->local;
      for (i=0; i<to->n; i++) to->vslots[i] = rto[to->vslots[i]];
    } else if (from->type == VEC_SCATTER_SEQ_GENERAL) {
      for (i=0; i<from->n; i++) from->vslots[i] = rto[from->vslots[i]];
    } else if (from->type == VEC_SCATTER_SEQ_STRIDE) {
      VecScatter_Seq_Stride *sto = (VecScatter_Seq_Stride*)from;

      /* if the remapping is the identity and stride is identity then skip remap */
      if (sto->step == 1 && sto->first == 0) {
        for (i=0; i<sto->n; i++) {
          if (rto[i] != i) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Unable to remap such scatters");
        }
      } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Unable to remap such scatters");
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Unable to remap such scatters");
  }

  if (rfrom) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unable to remap the FROM in scatters yet");

  /*
     Mark then vector lengths as unknown because we do not know the
   lengths of the remapped vectors
  */
  scat->from_n = -1;
  scat->to_n   = -1;
  PetscFunctionReturn(0);
}

#if 0  // rm VecScatterGetTypes_Private_MPI1

/*
 VecScatterGetTypes_Private - Returns the scatter types.

 scatter - The scatter.
 from    - Upon exit this contains the type of the from scatter.
 to      - Upon exit this contains the type of the to scatter.
*/
PetscErrorCode VecScatterGetTypes_Private_MPI1(VecScatter scatter,VecScatterType *from,VecScatterType *to)
{
  VecScatter_Common* fromdata = (VecScatter_Common*)scatter->fromdata;
  VecScatter_Common* todata   = (VecScatter_Common*)scatter->todata;

  PetscFunctionBegin;
  *from = fromdata->type;
  *to = todata->type;
  PetscFunctionReturn(0);
}

/*
  VecScatterIsSequential_Private - Returns true if the scatter is sequential.

  scatter - The scatter.
  flag    - Upon exit flag is true if the scatter is of type VecScatter_Seq_General 
            or VecScatter_Seq_Stride; otherwise flag is false.
*/
PetscErrorCode VecScatterIsSequential_Private_MPI1(VecScatter_Common *scatter,PetscBool *flag)
{
  VecScatterType scatterType = scatter->type;

  PetscFunctionBegin;
  if (scatterType == VEC_SCATTER_SEQ_GENERAL || scatterType == VEC_SCATTER_SEQ_STRIDE) {
    *flag = PETSC_TRUE;
  } else {
    *flag = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}
#endif

#if defined(PETSC_HAVE_CUSP) || defined(PETSC_HAVE_VECCUDA)

/*@C
   VecScatterInitializeForGPU - Initializes a generalized scatter from one vector
   to another for GPU based computation.

   Input Parameters:
+  inctx - scatter context generated by VecScatterCreate()
.  x - the vector from which we scatter
-  mode - the scattering mode, usually SCATTER_FORWARD.  The available modes are:
    SCATTER_FORWARD or SCATTER_REVERSE

  Level: intermediate

  Notes:
   Effectively, this function creates all the necessary indexing buffers and work
   vectors needed to move data only those data points in a vector which need to
   be communicated across ranks. This is done at the first time this function is
   called. Currently, this only used in the context of the parallel SpMV call in
   MatMult_MPIAIJCUSP or MatMult_MPIAIJCUSPARSE.

   This function is executed before the call to MatMult. This enables the memory
   transfers to be overlapped with the MatMult SpMV kernel call.

.seealso: VecScatterFinalizeForGPU(), VecScatterCreate(), VecScatterEnd()
@*/
PETSC_EXTERN PetscErrorCode VecScatterInitializeForGPU_MPI1(VecScatter inctx,Vec x,ScatterMode mode)
{
  PetscFunctionBegin;
  VecScatter_MPI_General *to,*from;
  PetscErrorCode         ierr;
  PetscInt               i,*indices,*sstartsSends,*sstartsRecvs,nrecvs,nsends,bs;
  PetscBool              isSeq1,isSeq2;

  PetscFunctionBegin;
  ierr = VecScatterIsSequential_Private((VecScatter_Common*)inctx->fromdata,&isSeq1);CHKERRQ(ierr);
  ierr = VecScatterIsSequential_Private((VecScatter_Common*)inctx->todata,&isSeq2);CHKERRQ(ierr);
  if (isSeq1 || isSeq2) {
    PetscFunctionReturn(0);
  }
  if (mode & SCATTER_REVERSE) {
    to     = (VecScatter_MPI_General*)inctx->fromdata;
    from   = (VecScatter_MPI_General*)inctx->todata;
  } else {
    to     = (VecScatter_MPI_General*)inctx->todata;
    from   = (VecScatter_MPI_General*)inctx->fromdata;
  }
  bs           = to->bs;
  nrecvs       = from->n;
  nsends       = to->n;
  indices      = to->indices;
  sstartsSends = to->starts;
  sstartsRecvs = from->starts;
#if defined(PETSC_HAVE_CUSP)
  if (x->valid_GPU_array != PETSC_CUSP_UNALLOCATED && (nsends>0 || nrecvs>0)) {
#else
  if (x->valid_GPU_array != PETSC_CUDA_UNALLOCATED && (nsends>0 || nrecvs>0)) {
#endif
    if (!inctx->spptr) {
      PetscInt k,*tindicesSends,*sindicesSends,*tindicesRecvs,*sindicesRecvs;
      PetscInt ns = sstartsSends[nsends],nr = sstartsRecvs[nrecvs];
      /* Here we create indices for both the senders and receivers. */
      ierr = PetscMalloc1(ns,&tindicesSends);CHKERRQ(ierr);
      ierr = PetscMalloc1(nr,&tindicesRecvs);CHKERRQ(ierr);

      ierr = PetscMemcpy(tindicesSends,indices,ns*sizeof(PetscInt));CHKERRQ(ierr);
      ierr = PetscMemcpy(tindicesRecvs,from->indices,nr*sizeof(PetscInt));CHKERRQ(ierr);

      ierr = PetscSortRemoveDupsInt(&ns,tindicesSends);CHKERRQ(ierr);
      ierr = PetscSortRemoveDupsInt(&nr,tindicesRecvs);CHKERRQ(ierr);

      ierr = PetscMalloc1(bs*ns,&sindicesSends);CHKERRQ(ierr);
      ierr = PetscMalloc1(from->bs*nr,&sindicesRecvs);CHKERRQ(ierr);

      /* sender indices */
      for (i=0; i<ns; i++) {
        for (k=0; k<bs; k++) sindicesSends[i*bs+k] = tindicesSends[i]+k;
      }
      ierr = PetscFree(tindicesSends);CHKERRQ(ierr);

      /* receiver indices */
      for (i=0; i<nr; i++) {
        for (k=0; k<from->bs; k++) sindicesRecvs[i*from->bs+k] = tindicesRecvs[i]+k;
      }
      ierr = PetscFree(tindicesRecvs);CHKERRQ(ierr);

      /* create GPU indices, work vectors, ... */
#if defined(PETSC_HAVE_CUSP)
      ierr = VecScatterCUSPIndicesCreate_PtoP(ns*bs,sindicesSends,nr*from->bs,sindicesRecvs,(PetscCUSPIndices*)&inctx->spptr);CHKERRQ(ierr);
#else
      ierr = VecScatterCUDAIndicesCreate_PtoP(ns*bs,sindicesSends,nr*from->bs,sindicesRecvs,(PetscCUDAIndices*)&inctx->spptr);CHKERRQ(ierr);
#endif
      ierr = PetscFree(sindicesSends);CHKERRQ(ierr);
      ierr = PetscFree(sindicesRecvs);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*@C
   VecScatterFinalizeForGPU - Finalizes a generalized scatter from one vector to
   another for GPU based computation.

   Input Parameter:
+  inctx - scatter context generated by VecScatterCreate()

  Level: intermediate

  Notes:
   Effectively, this function resets the temporary buffer flags. Currently, this
   only used in the context of the parallel SpMV call in in MatMult_MPIAIJCUDA
   or MatMult_MPIAIJCUDAARSE. Once the MatMultAdd is finished, the GPU temporary
   buffers used for messaging are no longer valid.

.seealso: VecScatterInitializeForGPU(), VecScatterCreate(), VecScatterEnd()
@*/
PETSC_EXTERN PetscErrorCode VecScatterFinalizeForGPU_MPI1(VecScatter inctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#endif

