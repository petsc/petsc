
/*
     Higher level code for creating scatters between vectors called by some of the implementations.

     The routines check for special cases and then call the implementation function for the general cases
*/

#include <petsc/private/vecscatterimpl.h>    /*I   "petscvec.h"    I*/

#if defined(PETSC_HAVE_CUDA)
#include <petsc/private/cudavecimpl.h>
#endif

/*
      This is special scatter code for when the entire parallel vector is copied to each processor.

   This code was written by Cameron Cooper, Occidental College, Fall 1995,
   will working at ANL as a SERS student.
*/
static PetscErrorCode VecScatterBegin_MPI_ToAll(VecScatter ctx,Vec x,Vec y,InsertMode addv,ScatterMode mode)
{
  PetscErrorCode ierr;
  PetscInt       yy_n,xx_n;
  PetscScalar    *xv,*yv;

  PetscFunctionBegin;
  ierr = VecGetLocalSize(y,&yy_n);CHKERRQ(ierr);
  ierr = VecGetLocalSize(x,&xx_n);CHKERRQ(ierr);
  ierr = VecGetArrayPair(x,y,&xv,&yv);CHKERRQ(ierr);

  if (mode & SCATTER_REVERSE) {
    PetscScalar          *xvt,*xvt2;
    VecScatter_MPI_ToAll *scat = (VecScatter_MPI_ToAll*)ctx->todata;
    PetscInt             i;
    PetscMPIInt          *disply = scat->displx;

    if (addv == INSERT_VALUES) {
      PetscInt rstart,rend;
      /*
         copy the correct part of the local vector into the local storage of
         the MPI one.  Note: this operation only makes sense if all the local
         vectors have the same values
      */
      ierr = VecGetOwnershipRange(y,&rstart,&rend);CHKERRQ(ierr);
      ierr = PetscArraycpy(yv,xv+rstart,yy_n);CHKERRQ(ierr);
    } else {
      MPI_Comm    comm;
      PetscMPIInt rank;
      ierr = PetscObjectGetComm((PetscObject)y,&comm);CHKERRQ(ierr);
      ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
      if (scat->work1) xvt = scat->work1;
      else {
        ierr        = PetscMalloc1(xx_n,&xvt);CHKERRQ(ierr);
        scat->work1 = xvt;
      }
      if (!rank) { /* I am the zeroth processor, values are accumulated here */
        if   (scat->work2) xvt2 = scat->work2;
        else {
          ierr        = PetscMalloc1(xx_n,&xvt2);CHKERRQ(ierr);
          scat->work2 = xvt2;
        }
        ierr = MPI_Gatherv(yv,yy_n,MPIU_SCALAR,xvt2,scat->count,disply,MPIU_SCALAR,0,PetscObjectComm((PetscObject)ctx));CHKERRQ(ierr);
        ierr = MPI_Reduce(xv,xvt,xx_n,MPIU_SCALAR,MPIU_SUM,0,PetscObjectComm((PetscObject)ctx));CHKERRQ(ierr);
        if (addv == ADD_VALUES) {
          for (i=0; i<xx_n; i++) xvt[i] += xvt2[i];
#if !defined(PETSC_USE_COMPLEX)
        } else if (addv == MAX_VALUES) {
          for (i=0; i<xx_n; i++) xvt[i] = PetscMax(xvt[i],xvt2[i]);
#endif
        } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Wrong insert option");
        ierr = MPI_Scatterv(xvt,scat->count,disply,MPIU_SCALAR,yv,yy_n,MPIU_SCALAR,0,PetscObjectComm((PetscObject)ctx));CHKERRQ(ierr);
      } else {
        ierr = MPI_Gatherv(yv,yy_n,MPIU_SCALAR,0, 0,0,MPIU_SCALAR,0,PetscObjectComm((PetscObject)ctx));CHKERRQ(ierr);
        ierr = MPI_Reduce(xv,xvt,xx_n,MPIU_SCALAR,MPIU_SUM,0,PetscObjectComm((PetscObject)ctx));CHKERRQ(ierr);
        ierr = MPI_Scatterv(0,scat->count,disply,MPIU_SCALAR,yv,yy_n,MPIU_SCALAR,0,PetscObjectComm((PetscObject)ctx));CHKERRQ(ierr);
      }
    }
  } else {
    PetscScalar          *yvt;
    VecScatter_MPI_ToAll *scat = (VecScatter_MPI_ToAll*)ctx->todata;
    PetscInt             i;
    PetscMPIInt          *displx = scat->displx;

    if (addv == INSERT_VALUES) {
      ierr = MPI_Allgatherv(xv,xx_n,MPIU_SCALAR,yv,scat->count,displx,MPIU_SCALAR,PetscObjectComm((PetscObject)ctx));CHKERRQ(ierr);
    } else {
      if (scat->work1) yvt = scat->work1;
      else {
        ierr        = PetscMalloc1(yy_n,&yvt);CHKERRQ(ierr);
        scat->work1 = yvt;
      }
      ierr = MPI_Allgatherv(xv,xx_n,MPIU_SCALAR,yvt,scat->count,displx,MPIU_SCALAR,PetscObjectComm((PetscObject)ctx));CHKERRQ(ierr);
      if (addv == ADD_VALUES) {
        for (i=0; i<yy_n; i++) yv[i] += yvt[i];
#if !defined(PETSC_USE_COMPLEX)
      } else if (addv == MAX_VALUES) {
        for (i=0; i<yy_n; i++) yv[i] = PetscMax(yv[i],yvt[i]);
#endif
      } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Wrong insert option");
    }
  }
  ierr = VecRestoreArrayPair(x,y,&xv,&yv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecScatterView_MPI_ToAll(VecScatter in,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      isascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"Entire parallel vector is copied to each process\n");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
      This is special scatter code for when the entire parallel vector is  copied to processor 0.

*/
static PetscErrorCode VecScatterBegin_MPI_ToOne(VecScatter ctx,Vec x,Vec y,InsertMode addv,ScatterMode mode)
{
  PetscErrorCode    ierr;
  PetscMPIInt       rank;
  PetscInt          yy_n,xx_n;
  PetscScalar       *yv;
  const PetscScalar *xv;
  MPI_Comm          comm;

  PetscFunctionBegin;
  ierr = VecGetLocalSize(y,&yy_n);CHKERRQ(ierr);
  ierr = VecGetLocalSize(x,&xx_n);CHKERRQ(ierr);
  ierr = VecGetArrayRead(x,&xv);CHKERRQ(ierr);
  ierr = VecGetArray(y,&yv);CHKERRQ(ierr);

  ierr = PetscObjectGetComm((PetscObject)x,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  /* --------  Reverse scatter; spread from processor 0 to other processors */
  if (mode & SCATTER_REVERSE) {
    PetscScalar          *yvt;
    VecScatter_MPI_ToAll *scat = (VecScatter_MPI_ToAll*)ctx->todata;
    PetscInt             i;
    PetscMPIInt          *disply = scat->displx;

    if (addv == INSERT_VALUES) {
      ierr = MPI_Scatterv((PetscScalar*)xv,scat->count,disply,MPIU_SCALAR,yv,yy_n,MPIU_SCALAR,0,PetscObjectComm((PetscObject)ctx));CHKERRQ(ierr);
    } else {
      if (scat->work2) yvt = scat->work2;
      else {
        PetscInt xx_nt;
        ierr = MPI_Allreduce(&xx_n,&xx_nt,1,MPIU_INT,MPI_MAX,PetscObjectComm((PetscObject)y));CHKERRQ(ierr);
        ierr        = PetscMalloc1(xx_nt,&yvt);CHKERRQ(ierr);
        scat->work2 = yvt;
      }
      ierr = MPI_Scatterv((PetscScalar*)xv,scat->count,disply,MPIU_SCALAR,yvt,yy_n,MPIU_SCALAR,0,PetscObjectComm((PetscObject)ctx));CHKERRQ(ierr);
      if (addv == ADD_VALUES) {
        for (i=0; i<yy_n; i++) yv[i] += yvt[i];
#if !defined(PETSC_USE_COMPLEX)
      } else if (addv == MAX_VALUES) {
        for (i=0; i<yy_n; i++) yv[i] = PetscMax(yv[i],yvt[i]);
#endif
      } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Wrong insert option");
    }
  /* ---------  Forward scatter; gather all values onto processor 0 */
  } else {
    PetscScalar          *yvt  = 0;
    VecScatter_MPI_ToAll *scat = (VecScatter_MPI_ToAll*)ctx->todata;
    PetscInt             i;
    PetscMPIInt          *displx = scat->displx;

    if (addv == INSERT_VALUES) {
      ierr = MPI_Gatherv((PetscScalar*)xv,xx_n,MPIU_SCALAR,yv,scat->count,displx,MPIU_SCALAR,0,PetscObjectComm((PetscObject)ctx));CHKERRQ(ierr);
    } else {
      if (!rank) {
        if (scat->work1) yvt = scat->work1;
        else {
          ierr        = PetscMalloc1(yy_n,&yvt);CHKERRQ(ierr);
          scat->work1 = yvt;
        }
      }
      ierr = MPI_Gatherv((PetscScalar*)xv,xx_n,MPIU_SCALAR,yvt,scat->count,displx,MPIU_SCALAR,0,PetscObjectComm((PetscObject)ctx));CHKERRQ(ierr);
      if (!rank) {
        if (addv == ADD_VALUES) {
          for (i=0; i<yy_n; i++) yv[i] += yvt[i];
#if !defined(PETSC_USE_COMPLEX)
        } else if (addv == MAX_VALUES) {
          for (i=0; i<yy_n; i++) yv[i] = PetscMax(yv[i],yvt[i]);
#endif
        }  else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Wrong insert option");
      }
    }
  }
  ierr = VecRestoreArrayRead(x,&xv);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&yv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
       The follow to are used for both VecScatterBegin_MPI_ToAll() and VecScatterBegin_MPI_ToOne()
*/
static PetscErrorCode VecScatterDestroy_MPI_ToAll(VecScatter ctx)
{
  VecScatter_MPI_ToAll *scat = (VecScatter_MPI_ToAll*)ctx->todata;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  ierr = PetscFree(scat->work1);CHKERRQ(ierr);
  ierr = PetscFree(scat->work2);CHKERRQ(ierr);
  ierr = PetscFree3(ctx->todata,scat->count,scat->displx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------*/
static PetscErrorCode VecScatterCopy_MPI_ToAll(VecScatter in,VecScatter out)
{
  VecScatter_MPI_ToAll *in_to = (VecScatter_MPI_ToAll*)in->todata,*sto;
  PetscErrorCode       ierr;
  PetscMPIInt          size,*count,*displx;
  PetscInt             i;

  PetscFunctionBegin;
  out->ops->begin   = in->ops->begin;
  out->ops->end     = in->ops->end;
  out->ops->copy    = in->ops->copy;
  out->ops->destroy = in->ops->destroy;
  out->ops->view    = in->ops->view;

  ierr        = MPI_Comm_size(PetscObjectComm((PetscObject)out),&size);CHKERRQ(ierr);
  ierr        = PetscMalloc3(1,&sto,size,&count,size,&displx);CHKERRQ(ierr);
  sto->format = in_to->format;
  sto->count  = count;
  sto->displx = displx;
  for (i=0; i<size; i++) {
    sto->count[i]  = in_to->count[i];
    sto->displx[i] = in_to->displx[i];
  }
  sto->work1    = 0;
  sto->work2    = 0;
  out->todata   = (void*)sto;
  out->fromdata = (void*)0;
  PetscFunctionReturn(0);
}

/*
  Create memcpy optimization plans based on indices of vector entries we want to scatter

   Input Parameters:
  +  n       - number of target processors
  .  starts  - [n+1] for the i-th processor, its associated indices are indices[starts[i], starts[i+1])
  .  indices - [] array storing indices. Its length is starts[n+1]
  -  bs      - block size

   Output Parameters:
  +  plan    - the memcpy plan
*/
PetscErrorCode VecScatterMemcpyPlanCreate_Index(PetscInt n,const PetscInt *starts,const PetscInt *indices,PetscInt bs,VecScatterMemcpyPlan *plan)
{
  PetscErrorCode ierr;
  PetscInt       i,j,k,my_copies,n_copies=0,step;
  PetscBool      strided,has_strided;

  PetscFunctionBegin;
  ierr    = PetscMemzero(plan,sizeof(VecScatterMemcpyPlan));CHKERRQ(ierr);
  plan->n = n;
  ierr    = PetscMalloc2(n,&plan->optimized,n+1,&plan->copy_offsets);CHKERRQ(ierr);

  /* check if each remote part of the scatter is made of copies, and count total_copies */
  for (i=0; i<n; i++) { /* for each target processor procs[i] */
    my_copies = 1; /* count num. of copies for procs[i] */
    for (j=starts[i]; j<starts[i+1]-1; j++) { /* go through indices from the first to the second to last */
      if (indices[j]+bs != indices[j+1]) my_copies++;
    }
    if (bs*(starts[i+1]-starts[i])*sizeof(PetscScalar)/my_copies >= 256) { /* worth using memcpy? */
      plan->optimized[i] = PETSC_TRUE;
      n_copies += my_copies;
    } else {
      plan->optimized[i] = PETSC_FALSE;
    }
  }

  /* do malloc with the recently known n_copies */
  ierr = PetscMalloc2(n_copies,&plan->copy_starts,n_copies,&plan->copy_lengths);CHKERRQ(ierr);

  /* analyze the total_copies one by one */
  k                     = 0; /* k-th copy */
  plan->copy_offsets[0] = 0;
  for (i=0; i<n; i++) { /* for each target processor procs[i] */
    if (plan->optimized[i]) {
      my_copies            = 1;
      plan->copy_starts[k] = indices[starts[i]];
      for (j=starts[i]; j<starts[i+1]-1; j++) {
        if (indices[j]+bs != indices[j+1]) { /* meet end of a copy (and next copy must exist) */
          my_copies++;
          plan->copy_starts[k+1] = indices[j+1];
          plan->copy_lengths[k]  = indices[j]+bs-plan->copy_starts[k];
          k++;
        }
      }
      /* set copy length of the last copy for this remote proc */
      plan->copy_lengths[k] = indices[j]+bs-plan->copy_starts[k];
      k++;
    }

    /* set offset for next proc. When optimized[i] is false, copy_offsets[i] = copy_offsets[i+1] */
    plan->copy_offsets[i+1] = k;
  }

  /* try the last chance to optimize. If a scatter is not memory copies, then is it strided? */
  has_strided = PETSC_FALSE;
  ierr = PetscMalloc3(n,&plan->stride_first,n,&plan->stride_step,n,&plan->stride_n);CHKERRQ(ierr);
  for (i=0; i<n; i++) { /* for each target processor procs[i] */
    if (!plan->optimized[i] && starts[i+1] - starts[i] >= 16) { /* few indices (<16) are not worth striding */
      strided = PETSC_TRUE;
      step    = indices[starts[i]+1] - indices[starts[i]];
      for (j=starts[i]; j<starts[i+1]-1; j++) {
        if (indices[j]+step != indices[j+1]) { strided = PETSC_FALSE; break; }
      }
      if (strided) {
        plan->optimized[i]    = PETSC_TRUE;
        plan->stride_first[i] = indices[starts[i]];
        plan->stride_step[i]  = step;
        plan->stride_n[i]     = starts[i+1] - starts[i];
        has_strided           = PETSC_TRUE;
      }
    }
  }
  /* if none is strided, free the arrays to save memory here and also in plan copying */
  if (!has_strided) { ierr = PetscFree3(plan->stride_first,plan->stride_step,plan->stride_n);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}

/* Copy the memcpy plan from in to out */
PetscErrorCode VecScatterMemcpyPlanCopy(const VecScatterMemcpyPlan *in,VecScatterMemcpyPlan *out)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr   = PetscMemzero(out,sizeof(VecScatterMemcpyPlan));CHKERRQ(ierr);
  out->n = in->n;
  ierr   = PetscMalloc2(in->n,&out->optimized,in->n+1,&out->copy_offsets);CHKERRQ(ierr);
  ierr   = PetscMalloc2(in->copy_offsets[in->n],&out->copy_starts,in->copy_offsets[in->n],&out->copy_lengths);CHKERRQ(ierr);
  ierr   = PetscArraycpy(out->optimized,in->optimized,in->n);CHKERRQ(ierr);
  ierr   = PetscArraycpy(out->copy_offsets,in->copy_offsets,in->n+1);CHKERRQ(ierr);
  ierr   = PetscArraycpy(out->copy_starts,in->copy_starts,in->copy_offsets[in->n]);CHKERRQ(ierr);
  ierr   = PetscArraycpy(out->copy_lengths,in->copy_lengths,in->copy_offsets[in->n]);CHKERRQ(ierr);
  if (in->stride_first) {
    ierr = PetscMalloc3(in->n,&out->stride_first,in->n,&out->stride_step,in->n,&out->stride_n);CHKERRQ(ierr);
    ierr = PetscArraycpy(out->stride_first,in->stride_first,in->n);CHKERRQ(ierr);
    ierr = PetscArraycpy(out->stride_step,in->stride_step,in->n);CHKERRQ(ierr);
    ierr = PetscArraycpy(out->stride_n,in->stride_n,in->n);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* Destroy the vecscatter memcpy plan */
PetscErrorCode VecScatterMemcpyPlanDestroy(VecScatterMemcpyPlan *plan)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree2(plan->optimized,plan->copy_offsets);CHKERRQ(ierr);
  ierr = PetscFree2(plan->copy_starts,plan->copy_lengths);CHKERRQ(ierr);
  ierr = PetscFree3(plan->stride_first,plan->stride_step,plan->stride_n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* =======================================================================*/
/*
   Blocksizes we have optimized scatters for
*/

#define VecScatterOptimizedBS(mbs) (2 <= mbs)


static PetscErrorCode VecScatterCreate_PtoS(VecScatter ctx,PetscErrorCode (*vecscattercreatelocal_ptos)(PetscInt,const PetscInt*,PetscInt,const PetscInt*,Vec,Vec,PetscInt,VecScatter))
{
  PetscErrorCode    ierr;
  PetscMPIInt       size;
  PetscInt          ix_type=-1,iy_type=-1,*range;
  MPI_Comm          comm;
  IS                tix=NULL,tiy=NULL,ix=ctx->from_is,iy=ctx->to_is;
  Vec               xin=ctx->from_v,yin=ctx->to_v;
  PetscBool         totalv,ixblock,iyblock,iystride,islocal,cando;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)ctx,&comm);CHKERRQ(ierr);
  ierr = GetInputISType_private(ctx,VEC_MPI_ID,VEC_SEQ_ID,&ix_type,&tix,&iy_type,&tiy);CHKERRQ(ierr);
  if (tix) ix = tix;
  if (tiy) iy = tiy;

  islocal = PETSC_FALSE;
  /* special case extracting (subset of) local portion */
  if (ix_type == IS_STRIDE_ID && iy_type == IS_STRIDE_ID) {
    /* Case (2-a) */
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
      to12->format         = VEC_SCATTER_SEQ_STRIDE;
      from12->format       = VEC_SCATTER_SEQ_STRIDE;
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
    /* Case (2-b) */
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
      sto->format         = VEC_SCATTER_MPI_TOALL;
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
    /* Case (2-c) */
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
      sto->format         = VEC_SCATTER_MPI_TOONE;
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

  /* Case 2-d */
  ierr = PetscObjectTypeCompare((PetscObject)ix,ISBLOCK,&ixblock);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)iy,ISBLOCK,&iyblock);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)iy,ISSTRIDE,&iystride);CHKERRQ(ierr);
  if (ixblock) {
    /* special case block to block */
    if (iyblock) {
      PetscInt       nx,ny,bsx,bsy,min,max;
      const PetscInt *idx,*idy;
      ierr = ISGetBlockSize(iy,&bsy);CHKERRQ(ierr);
      ierr = ISGetBlockSize(ix,&bsx);CHKERRQ(ierr);
      min  = PetscMin(bsx,bsy);
      max  = PetscMax(bsx,bsy);
      ierr = MPIU_Allreduce(MPI_IN_PLACE,&min,1,MPIU_INT,MPI_MIN,comm);CHKERRQ(ierr);
      ierr = MPIU_Allreduce(MPI_IN_PLACE,&max,1,MPIU_INT,MPI_MAX,comm);CHKERRQ(ierr);
      if (min == max && VecScatterOptimizedBS(bsx)) {
        ierr = ISBlockGetLocalSize(ix,&nx);CHKERRQ(ierr);
        ierr = ISBlockGetIndices(ix,&idx);CHKERRQ(ierr);
        ierr = ISBlockGetLocalSize(iy,&ny);CHKERRQ(ierr);
        ierr = ISBlockGetIndices(iy,&idy);CHKERRQ(ierr);
        if (nx != ny) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local scatter sizes don't match");
        ierr = (*vecscattercreatelocal_ptos)(nx,idx,ny,idy,xin,yin,bsx,ctx);CHKERRQ(ierr);
        ierr = ISBlockRestoreIndices(ix,&idx);CHKERRQ(ierr);
        ierr = ISBlockRestoreIndices(iy,&idy);CHKERRQ(ierr);
        ierr = PetscInfo(xin,"Special case: blocked indices\n");CHKERRQ(ierr);
        goto functionend;
      }
      /* special case block to stride */
    } else if (iystride) {
      /* Case (2-e) */
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
        ierr = (*vecscattercreatelocal_ptos)(nx,idx,nx,idy,xin,yin,bsx,ctx);CHKERRQ(ierr);
        ierr = PetscFree(idy);CHKERRQ(ierr);
        ierr = ISBlockRestoreIndices(ix,&idx);CHKERRQ(ierr);
        ierr = PetscInfo(xin,"Special case: blocked indices to stride\n");CHKERRQ(ierr);
        goto functionend;
      }
    }
  }

  /* left over general case (2-f) */
  {
    PetscInt       nx,ny;
    const PetscInt *idx,*idy;
    ierr = ISGetLocalSize(ix,&nx);CHKERRQ(ierr);
    ierr = ISGetIndices(ix,&idx);CHKERRQ(ierr);
    ierr = ISGetLocalSize(iy,&ny);CHKERRQ(ierr);
    ierr = ISGetIndices(iy,&idy);CHKERRQ(ierr);
    if (nx != ny) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local scatter sizes don't match (%D %D)",nx,ny);
    ierr = (*vecscattercreatelocal_ptos)(nx,idx,ny,idy,xin,yin,1,ctx);CHKERRQ(ierr);
    ierr = ISRestoreIndices(ix,&idx);CHKERRQ(ierr);
    ierr = ISRestoreIndices(iy,&idy);CHKERRQ(ierr);
    ierr = PetscInfo(xin,"General case: MPI to Seq\n");CHKERRQ(ierr);
    goto functionend;
  }
  functionend:
  ierr = ISDestroy(&tix);CHKERRQ(ierr);
  ierr = ISDestroy(&tiy);CHKERRQ(ierr);
  ierr = VecScatterViewFromOptions(ctx,NULL,"-vecscatter_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

  static PetscErrorCode VecScatterCreate_StoP(VecScatter ctx,PetscErrorCode (*vecscattercreatelocal_stop)(PetscInt,const PetscInt*,PetscInt,const PetscInt*,Vec,Vec,PetscInt,VecScatter))
{
  PetscErrorCode    ierr;
  PetscInt          ix_type=-1,iy_type=-1;
  IS                tix=NULL,tiy=NULL,ix=ctx->from_is,iy=ctx->to_is;
  Vec               xin=ctx->from_v,yin=ctx->to_v;
  PetscBool         islocal,cando;

  PetscFunctionBegin;
  ierr = GetInputISType_private(ctx,VEC_MPI_ID,VEC_SEQ_ID,&ix_type,&tix,&iy_type,&tiy);CHKERRQ(ierr);
  if (tix) ix = tix;
  if (tiy) iy = tiy;

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
      to->format          = VEC_SCATTER_SEQ_STRIDE;
      from->format        = VEC_SCATTER_SEQ_STRIDE;
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
      ierr = (*vecscattercreatelocal_stop)(nx,idx,nx,idy,xin,yin,bsx,ctx);CHKERRQ(ierr);
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
    ierr = (*vecscattercreatelocal_stop)(nx,idx,ny,idy,xin,yin,1,ctx);CHKERRQ(ierr);
    ierr = ISRestoreIndices(ix,&idx);CHKERRQ(ierr);
    ierr = ISRestoreIndices(iy,&idy);CHKERRQ(ierr);
    ierr = PetscInfo(xin,"General case: Seq to MPI\n");CHKERRQ(ierr);
    goto functionend;
  }
  functionend:
  ierr = ISDestroy(&tix);CHKERRQ(ierr);
  ierr = ISDestroy(&tiy);CHKERRQ(ierr);
  ierr = VecScatterViewFromOptions(ctx,NULL,"-vecscatter_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode VecScatterCreate_PtoP(VecScatter ctx,PetscErrorCode (*vecscattercreatelocal_ptop)(PetscInt,const PetscInt*,PetscInt,const PetscInt*,Vec,Vec,PetscInt,VecScatter))
{
  PetscErrorCode    ierr;
  PetscInt          ix_type=-1,iy_type=-1;
  IS                tix=NULL,tiy=NULL,ix=ctx->from_is,iy=ctx->to_is;
  Vec               xin=ctx->from_v,yin=ctx->to_v;
  PetscInt          nx,ny;
  const PetscInt    *idx,*idy;

  PetscFunctionBegin;
  ierr = GetInputISType_private(ctx,VEC_MPI_ID,VEC_MPI_ID,&ix_type,&tix,&iy_type,&tiy);CHKERRQ(ierr);
  if (tix) ix = tix;
  if (tiy) iy = tiy;

  /* no special cases for now */
  ierr = ISGetLocalSize(ix,&nx);CHKERRQ(ierr);
  ierr = ISGetIndices(ix,&idx);CHKERRQ(ierr);
  ierr = ISGetLocalSize(iy,&ny);CHKERRQ(ierr);
  ierr = ISGetIndices(iy,&idy);CHKERRQ(ierr);
  if (nx != ny) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local scatter sizes don't match");
  ierr = (*vecscattercreatelocal_ptop)(nx,idx,ny,idy,xin,yin,1,ctx);CHKERRQ(ierr);
  ierr = ISRestoreIndices(ix,&idx);CHKERRQ(ierr);
  ierr = ISRestoreIndices(iy,&idy);CHKERRQ(ierr);
  ierr = PetscInfo(xin,"General case: MPI to MPI\n");CHKERRQ(ierr);

  ierr = ISDestroy(&tix);CHKERRQ(ierr);
  ierr = ISDestroy(&tiy);CHKERRQ(ierr);
  ierr = VecScatterViewFromOptions(ctx,NULL,"-vecscatter_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode VecScatterGetInputVecType_private(VecScatter ctx,PetscInt *xin_type1,PetscInt *yin_type1)
{
  PetscErrorCode    ierr;
  MPI_Comm          comm,ycomm;
  PetscMPIInt       size;
  Vec               xin = ctx->from_v,yin = ctx->to_v;
  PetscInt          xin_type,yin_type;

  PetscFunctionBegin;
  /*
      Determine if the vectors are "parallel", ie. it shares a comm with other processors, or
      sequential (it does not share a comm). The difference is that parallel vectors treat the
      index set as providing indices in the global parallel numbering of the vector, with
      sequential vectors treat the index set as providing indices in the local sequential
      numbering
  */
  ierr = PetscObjectGetComm((PetscObject)xin,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (size > 1) {
    xin_type = VEC_MPI_ID;
  } else xin_type = VEC_SEQ_ID;
  *xin_type1 = xin_type;

  ierr = PetscObjectGetComm((PetscObject)yin,&ycomm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(ycomm,&size);CHKERRQ(ierr);
  if (size > 1) {
    yin_type = VEC_MPI_ID;
  } else yin_type = VEC_SEQ_ID;
  *yin_type1 = yin_type;
  PetscFunctionReturn(0);
}

PetscErrorCode GetInputISType_private(VecScatter ctx,PetscInt xin_type,PetscInt yin_type,PetscInt *ix_type1,IS *tix1,PetscInt *iy_type1,IS *tiy1)
{
  PetscErrorCode    ierr;
  MPI_Comm          comm;
  Vec               xin = ctx->from_v,yin = ctx->to_v;
  IS                tix = 0,tiy = 0,ix = ctx->from_is, iy = ctx->to_is;
  PetscInt          ix_type  = IS_GENERAL_ID,iy_type = IS_GENERAL_ID;
  PetscBool         flag;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)ctx,&comm);CHKERRQ(ierr);

  /* if ix or iy is not included; assume just grabbing entire vector */
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

  /* Determine types of index sets */
  ierr = PetscObjectTypeCompare((PetscObject)ix,ISBLOCK,&flag);CHKERRQ(ierr);
  if (flag) ix_type = IS_BLOCK_ID;
  ierr = PetscObjectTypeCompare((PetscObject)iy,ISBLOCK,&flag);CHKERRQ(ierr);
  if (flag) iy_type = IS_BLOCK_ID;
  ierr = PetscObjectTypeCompare((PetscObject)ix,ISSTRIDE,&flag);CHKERRQ(ierr);
  if (flag) ix_type = IS_STRIDE_ID;
  ierr = PetscObjectTypeCompare((PetscObject)iy,ISSTRIDE,&flag);CHKERRQ(ierr);
  if (flag) iy_type = IS_STRIDE_ID;

  if (ix_type1) *ix_type1 = ix_type;
  if (iy_type1) *iy_type1 = iy_type;
  if (tix1)     *tix1     = tix;
  if (tiy1)     *tiy1     = tiy;
  PetscFunctionReturn(0);
}

PetscErrorCode VecScatterSetUp_vectype_private(VecScatter ctx,PetscErrorCode (*vecscattercreatelocal_ptos)(PetscInt,const PetscInt*,PetscInt,const PetscInt*,Vec,Vec,PetscInt,VecScatter),PetscErrorCode (*vecscattercreatelocal_stop)(PetscInt,const PetscInt*,PetscInt,const PetscInt*,Vec,Vec,PetscInt,VecScatter),PetscErrorCode (*vecscattercreatelocal_ptop)(PetscInt,const PetscInt*,PetscInt,const PetscInt*,Vec,Vec,PetscInt,VecScatter))
{
  PetscErrorCode    ierr;
  PetscInt          xin_type=-1,yin_type=-1;

  PetscFunctionBegin;
  ierr = VecScatterGetInputVecType_private(ctx,&xin_type,&yin_type);CHKERRQ(ierr);
  if (xin_type == VEC_MPI_ID && yin_type == VEC_SEQ_ID) {
    ierr = VecScatterCreate_PtoS(ctx,vecscattercreatelocal_ptos);CHKERRQ(ierr);
  } else if (xin_type == VEC_SEQ_ID && yin_type == VEC_MPI_ID) {
    ierr = VecScatterCreate_StoP(ctx,vecscattercreatelocal_stop);CHKERRQ(ierr);
  } else if (xin_type == VEC_MPI_ID && yin_type == VEC_MPI_ID) {
    ierr = VecScatterCreate_PtoP(ctx,vecscattercreatelocal_ptop);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

