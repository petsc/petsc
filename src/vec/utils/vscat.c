
/*
       This code does not use data encapsulation. The reason is the 
   we need to scatter between different vector types (now chiefly 
   between sequential and MPI parallel). 

       This code knows about the different implementations
*/

#include "comm.h"
#include "is/isimpl.h"
#include "vecimpl.h"                     /*I "vec.h" I*/
#include "impls/dvecimpl.h"
#include "impls/mpi/pvecimpl.h"

/*
    Sequential general to general scatter 
*/
static int SGtoSG(Vec x,Vec y,VecScatterCtx ctx,InsertMode addv)
{
  VecScatterGeneral *gen_to = (VecScatterGeneral *) ctx->todata;
  VecScatterGeneral *gen_from = (VecScatterGeneral *) ctx->fromdata;
  int               i, n = gen_from->n, *fslots = gen_from->slots;
  int               *tslots = gen_to->slots;
  DvVector          *xx = (DvVector *) x->data,*yy = (DvVector *) y->data;
  Scalar            *xv = xx->array, *yv = yy->array;
  
  if (addv == InsertValues) {
    for ( i=0; i<n; i++ ) {yv[tslots[i]] = xv[fslots[i]];}
  }
  else {
    for ( i=0; i<n; i++ ) {yv[tslots[i]] += xv[fslots[i]];}
  }
  return 0;
}
static int SGtoSS(Vec x,Vec y,VecScatterCtx ctx,InsertMode addv)
{
  VecScatterStride  *gen_to = (VecScatterStride *) ctx->todata;
  VecScatterGeneral *gen_from = (VecScatterGeneral *) ctx->fromdata;
  int               i, n = gen_from->n, *fslots = gen_from->slots;
  int               first = gen_to->first,step = gen_to->step;
  DvVector          *xx = (DvVector *) x->data,*yy = (DvVector *) y->data;
  Scalar            *xv = xx->array, *yv = yy->array;
  
  if (addv == InsertValues) {
    for ( i=0; i<n; i++ ) {yv[first + i*step] = xv[fslots[i]];}
  }
  else {
    for ( i=0; i<n; i++ ) {yv[first + i*step] += xv[fslots[i]];}
  }
  return 0;
}

static int SStoSG(Vec x,Vec y,VecScatterCtx ctx,InsertMode addv)
{
  VecScatterStride  *gen_from = (VecScatterStride *) ctx->fromdata;
  VecScatterGeneral *gen_to = (VecScatterGeneral *) ctx->todata;
  int               i, n = gen_from->n, *fslots = gen_to->slots;
  int               first = gen_from->first,step = gen_from->step;
  DvVector          *xx = (DvVector *) x->data,*yy = (DvVector *) y->data;
  Scalar            *xv = xx->array, *yv = yy->array;
  
  if (addv == InsertValues) {
    for ( i=0; i<n; i++ ) {yv[fslots[i]] = xv[first + i*step];}
  }
  else {
    for ( i=0; i<n; i++ ) {yv[fslots[i]] += xv[first + i*step];}
  }
  return 0;
}

static int SStoSS(Vec x,Vec y,VecScatterCtx ctx,InsertMode addv)
{
  VecScatterStride  *gen_to = (VecScatterStride *) ctx->todata;
  VecScatterStride  *gen_from = (VecScatterStride *) ctx->fromdata;
  int               i, n = gen_from->n;
  int               to_first = gen_to->first,to_step = gen_to->step;
  int               from_first = gen_from->first,from_step = gen_from->step;
  DvVector          *xx = (DvVector *) x->data,*yy = (DvVector *) y->data;
  Scalar            *xv = xx->array, *yv = yy->array;
  
  if (addv == InsertValues) {
    if (to_step == 1 && from_step == 1) {
      MEMCPY(yv+to_first,xv+from_first,n*sizeof(Scalar));
    }
    else {
      for ( i=0; i<n; i++ ) {
        yv[to_first + i*to_step] = xv[from_first+i*from_step];
      }
    }
  }
  else {
    if (to_step == 1 && from_step == 1) {
      yv += to_first; xv += from_first;
      for ( i=0; i<n; i++ ) {
        yv[i] += xv[i];
      }
    }
    else {
      for ( i=0; i<n; i++ ) {
        yv[to_first + i*to_step] += xv[from_first+i*from_step];
      }
    }
  }
  return 0;
}
static int SGtoSGDestroy(PetscObject obj)
{
  VecScatterCtx ctx = (VecScatterCtx) obj;
  FREE(ctx->todata); FREE(ctx->fromdata); FREE(ctx);
  return 0;
}

int PtoSScatterCtxCreate(int,int *,int,int *,Vec,VecScatterCtx);
int StoPScatterCtxCreate(int,int *,int,int *,Vec,VecScatterCtx);
/* --------------------------------------------------------------*/
/*@
     VecScatterCtxCreate - Creates a vector scatter context. This should 
        be called you need to create a vector scatter context, but not 
        actually use it at creation. 

  Input Parameters:
.  xin the vector to scatter from
.  yin - the vector to scatter to
.  ix - the indices in xin to scatter
.  iy - tte indices in yin to put results

  Output Parameters:
.  newctx - location to store the scatter context

  Keywords: scatter, context, create
@*/
int VecScatterCtxCreate(Vec xin,IS ix,Vec yin,IS iy,VecScatterCtx *newctx)
{
  VecScatterCtx ctx;
  int           len; 

  /* generate the Scatter context */
  CREATEHEADER(ctx,_VecScatterCtx);
  ctx->cookie = VEC_SCATTER_COOKIE;

  if (xin->type == SEQVECTOR && yin->type == SEQVECTOR) {

    if (ix->type == ISGENERALSEQUENTIAL && iy->type == ISGENERALSEQUENTIAL){
      int               nx,ny,*idx,*idy;
      VecScatterGeneral *to,*from;
      ISGetLocalSize(ix,&nx); ISGetIndices(ix,&idx);
      ISGetLocalSize(iy,&ny); ISGetIndices(iy,&idy);
      if (nx != ny) SETERR(1,"Local scatter sizes don't match");
      len = sizeof(VecScatterGeneral) + nx*sizeof(int);
      to = (VecScatterGeneral *) MALLOC(len); CHKPTR(to);
      to->slots = (int *) (to + 1); to->n = nx; 
      MEMCPY(to->slots,idy,nx*sizeof(int));
      from = (VecScatterGeneral *) MALLOC(len); CHKPTR(from);
      from->slots = (int *) (from + 1); from->n = nx; 
      MEMCPY(from->slots,idx,nx*sizeof(int));
      ctx->todata = (void *) to; ctx->fromdata = (void *) from;
      ctx->begin = SGtoSG; ctx->destroy = SGtoSGDestroy;
      ctx->end = 0;
      *newctx = ctx;
      return 0;
    }
    else if (ix->type == ISSTRIDESEQUENTIAL && 
                                            iy->type == ISSTRIDESEQUENTIAL){
      int               nx,ny,to_first,to_step,from_first,from_step;
      VecScatterStride  *from,*to;
      ISGetLocalSize(ix,&nx); ISStrideGetInfo(ix,&from_first,&from_step);
      ISGetLocalSize(iy,&ny); ISStrideGetInfo(iy,&to_first,&to_step);
      if (nx != ny) SETERR(1,"Local scatter sizes don't match");
      to = (VecScatterStride *) MALLOC(sizeof(VecScatterStride)); CHKPTR(to);
      to->n = nx; to->first = to_first; to->step = to_step;
      from = (VecScatterStride *) MALLOC(sizeof(VecScatterStride));
      CHKPTR(from);
      from->n = nx; from->first = from_first; from->step = from_step;
      ctx->todata = (void *) to; ctx->fromdata = (void *) from;
      ctx->begin = SStoSS; ctx->destroy = SGtoSGDestroy;
      ctx->end = 0;
      *newctx = ctx;
      return 0;
    }
    else if (ix->type == ISGENERALSEQUENTIAL &&
                                            iy->type == ISSTRIDESEQUENTIAL){
      int               nx,ny,*idx,first,step;
      VecScatterGeneral *from;
      VecScatterStride  *to;
      ISGetLocalSize(ix,&nx); ISGetIndices(ix,&idx);
      ISGetLocalSize(iy,&ny); ISStrideGetInfo(iy,&first,&step);
      if (nx != ny) SETERR(1,"Local scatter sizes don't match");
      to = (VecScatterStride *) MALLOC(sizeof(VecScatterStride)); CHKPTR(to);
      to->n = nx; to->first = first; to->step = step;
      len = sizeof(VecScatterGeneral) + nx*sizeof(int);
      from = (VecScatterGeneral *) MALLOC(len); CHKPTR(from);
      from->slots = (int *) (from + 1); from->n = nx; 
      MEMCPY(from->slots,idx,nx*sizeof(int));
      ctx->todata = (void *) to; ctx->fromdata = (void *) from;
      ctx->begin = SGtoSS; ctx->destroy = SGtoSGDestroy;
      ctx->end = 0;
      *newctx = ctx;
      return 0;
    }
    else if (ix->type == ISSTRIDESEQUENTIAL &&
                                        iy->type == ISGENERALSEQUENTIAL){
      int               nx,ny,*idx,first,step;
      VecScatterGeneral *to;
      VecScatterStride  *from;
      ISGetLocalSize(ix,&nx); ISGetIndices(iy,&idx);
      ISGetLocalSize(iy,&ny); ISStrideGetInfo(ix,&first,&step);
      if (nx != ny) SETERR(1,"Local scatter sizes don't match");
      from = (VecScatterStride *) MALLOC(sizeof(VecScatterStride)); 
      CHKPTR(from);
      from->n = nx; from->first = first; from->step = step;
      len = sizeof(VecScatterGeneral) + nx*sizeof(int);
      to = (VecScatterGeneral *) MALLOC(len); CHKPTR(to);
      to->slots = (int *) (to + 1); to->n = nx; 
      MEMCPY(to->slots,idx,nx*sizeof(int));
      ctx->todata = (void *) to; ctx->fromdata = (void *) from;
      ctx->begin = SStoSG; ctx->destroy = SGtoSGDestroy;
      ctx->end = 0;
      *newctx = ctx;
      return 0;
    }
    else {
      SETERR(1,"Cannot generate such Scatter Context yet");
    }
  }
  if (xin->type == MPIVECTOR && yin->type == SEQVECTOR) {
    /* special case extracting (subset of) local portion */ 
    if (ix->type == ISSTRIDESEQUENTIAL && iy->type == ISSTRIDESEQUENTIAL){
      DvPVector         *x = (DvPVector *)xin->data;
      int               nx,ny,to_first,to_step,from_first,from_step;
      int               start = x->ownership[x->mytid];
      int               end = x->ownership[x->mytid+1],islocal,cando;
      VecScatterStride  *from,*to;
      ISGetLocalSize(ix,&nx); ISStrideGetInfo(ix,&from_first,&from_step);
      ISGetLocalSize(iy,&ny); ISStrideGetInfo(iy,&to_first,&to_step);
      if (nx != ny) SETERR(1,"Local scatter sizes don't match");
      if (ix->min >= start && ix->max < end ) islocal = 1; else islocal = 0;
      MPI_Allreduce((void *) &islocal,(void *) &cando,1,MPI_INT,
                    MPI_LAND,x->comm);
      if (cando) {
        to = (VecScatterStride *) MALLOC(sizeof(VecScatterStride)); 
        CHKPTR(to);
        to->n = nx; to->first = to_first; to->step = to_step;
        from = (VecScatterStride *) MALLOC(sizeof(VecScatterStride));
        CHKPTR(from);
        from->n = nx; from->first = from_first-start; from->step = from_step;
        ctx->todata = (void *) to; ctx->fromdata = (void *) from;
        ctx->begin = SStoSS; ctx->destroy = SGtoSGDestroy;
        ctx->end = 0;
        *newctx = ctx;
        return 0;
      }
    }
    {
      int ierr,nx,ny,*idx,*idy;
      ISGetLocalSize(ix,&nx); ISGetIndices(ix,&idx);
      ISGetLocalSize(iy,&ny); ISGetIndices(iy,&idy);
      if (nx != ny) SETERR(1,"Local scatter sizes don't match");
      ierr = PtoSScatterCtxCreate(nx,idx,ny,idy,xin,ctx); CHKERR(ierr);
      ISRestoreIndices(ix,&idx); ISRestoreIndices(iy,&idy);
      *newctx = ctx;
      return 0;
    }
  }
  if (xin->type == SEQVECTOR && yin->type == MPIVECTOR) {
    /* special case local copy portion */ 
    if (ix->type == ISSTRIDESEQUENTIAL && iy->type == ISSTRIDESEQUENTIAL){
      DvPVector         *y = (DvPVector *)yin->data;
      int               nx,ny,to_first,to_step,from_first,from_step;
      int               start = y->ownership[y->mytid];
      int               end = y->ownership[y->mytid+1],islocal,cando;
      VecScatterStride  *from,*to;
      ISGetLocalSize(ix,&nx); ISStrideGetInfo(ix,&from_first,&from_step);
      ISGetLocalSize(iy,&ny); ISStrideGetInfo(iy,&to_first,&to_step);
      if (nx != ny) SETERR(1,"Local scatter sizes don't match");
      if (iy->min >= start && iy->max < end ) islocal = 1; else islocal = 0;
      MPI_Allreduce((void *) &islocal,(void *) &cando,1,MPI_INT,
                    MPI_LAND,y->comm);
      if (cando) {
        to = (VecScatterStride *) MALLOC(sizeof(VecScatterStride)); 
        CHKPTR(to);
        to->n = nx; to->first = to_first-start; to->step = to_step;
        from = (VecScatterStride *) MALLOC(sizeof(VecScatterStride));
        CHKPTR(from);
        from->n = nx; from->first = from_first; from->step = from_step;
        ctx->todata = (void *) to; ctx->fromdata = (void *) from;
        ctx->begin = SStoSS; ctx->destroy = SGtoSGDestroy;
        ctx->end = 0;
        *newctx = ctx;
        return 0;
      }
    }
    {
      int ierr,nx,ny,*idx,*idy;
      ISGetLocalSize(ix,&nx); ISGetIndices(ix,&idx);
      ISGetLocalSize(iy,&ny); ISGetIndices(iy,&idy);
      if (nx != ny) SETERR(1,"Local scatter sizes don't match");
      ierr = StoPScatterCtxCreate(nx,idx,ny,idy,yin,ctx); CHKERR(ierr);
      ISRestoreIndices(ix,&idx); ISRestoreIndices(iy,&idy);
      *newctx = ctx;
      return 0;
    }
  }
  else {
    SETERR(1,"Cannot generate such Scatter Context yet");
  }
  return 0;
}

/* ------------------------------------------------------------------*/
/*@
     VecScatterBegin  -  Scatters from one vector into another.
                         This is far more general than the usual
                         scatter. Depending on ix, iy it can be 
                         a gather or a scatter or a combination.
                         If x is a parallel vector and y sequential
                         it can serve to gather values to a single
                         processor. Similar if y is parallel and 
                         x sequential it can scatter from one processor
                         to many.

  Input Parameters:
.  x - vector to scatter from
.  ix - indices of elements in x to take
.  iy - indices of locations in y to insert 
.  inctx - it not pointing to null, is used to coordinate communication
.  addv - either AddValues or InsertValues

  Output Parameters:
.  y - vector to scatter to 
.  inctx - if points to null then it is set and used

  Notes:
.   y[iy[i]] = x[ix[i]], for i=0,...,ni-1
.   destroy inctx with VecScatterCtxDestroy() when no longer needed.
@*/
int VecScatterBegin(Vec x,IS ix,Vec y,IS iy,InsertMode addv, 
                                                   VecScatterCtx *inctx)
{
  int           ierr;
  VecScatterCtx ctx;
  VALIDHEADER(x,VEC_COOKIE); VALIDHEADER(y,VEC_COOKIE);
  VALIDHEADER(ix,IS_COOKIE); VALIDHEADER(iy,IS_COOKIE);

  if (!*inctx) {
    ierr = VecScatterCtxCreate(x,ix,y,iy,inctx); CHKERR(ierr);
  }
  return (*(*inctx)->begin)(x,y,*inctx,addv);
}

/* --------------------------------------------------------------------*/
/*@
     VecScatterEnd  -  End scatter from one vector into another.
            Call after call to VecScatterBegin().

  Input Parameters:
.  x - vector to scatter from
.  ix - indices of elements in x to take
.  iy - indices of locations in y to insert 
.  addv - either AddValues or InsertValues

  Output Parameters:
.  y - vector to scatter to 

  Notes:
.   y[iy[i]] = x[ix[i]], for i=0,...,ni-1
@*/
int VecScatterEnd(Vec x,IS ix,Vec y,IS iy,InsertMode addv,VecScatterCtx *ctx)
{
  VALIDHEADER(x,VEC_COOKIE); VALIDHEADER(y,VEC_COOKIE);
  VALIDHEADER(ix,IS_COOKIE); VALIDHEADER(iy,IS_COOKIE);
  VALIDHEADER(*ctx,VEC_SCATTER_COOKIE);
  if ((*ctx)->end) return (*(*ctx)->end)(x,y,*ctx,addv);
  else return 0;
}


/*@
     VecScatterCtxDestroy - Destroys a scatter context created by 
              VecScatter(), VecScatterAdd() or VecScatterCtxCreate().

  Input Parameters:
.   sctx - the scatter context

  Keywords: vector, scatter,
@*/
int VecScatterCtxDestroy( VecScatterCtx ctx )
{
  return (*ctx->destroy)((PetscObject)ctx);
}


