
/*
       This code does not use data encapsulation. The reason is the 
   we need to scatter between different vector types (now chiefly 
   between sequential and MPI parallel). 

       This code knows about the different implementations
*/

#include "is/isimpl.h"
#include "vecimpl.h"                     /*I "vec.h" I*/
#include "impls/dvecimpl.h"
#include "impls/mpi/pvecimpl.h"

/*
     This creates a scatter context. 
*/
int VecScatterCtxCreate(Vec x,IS ix,Vec y,IS iy,VecScatterCtx *newctx)
{
  VecScatterCtx ctx;
  Scatter       *to,*from;
  if (x->type == SEQVECTOR && y->type == SEQVECTOR) {
    DvVector *xx = (DvVector *) x->data;
    DvVector *yy = (DvVector *) y->data;
    int      nx,ny,*idx,*idy;
    /* Index sets must be sequential to match */
    if (ix->type != ISGENERALSEQUENTIAL || iy->type != ISGENERALSEQUENTIAL){
      SETERR(1,"Sequential vector in VecScatter requires sequential IS");
    }
    ISGetLocalSize(ix,&nx); ISGetIndices(ix,&idx);
    ISGetLocalSize(iy,&ny); ISGetIndices(iy,&idy
);
    if (nx != ny) SETERR(1,"Local scatter sizes don't match");

    /* generate the Scatter context */
    CREATEHEADER(ctx,_VecScatterCtx);
    ctx->nto    = ctx->nfrom   = 1;
    ctx->lento  = ctx->lenfrom = nx;
    ctx->cookie = VEC_SCATTER_COOKIE;
    to = (Scatter *) MALLOC(nx*sizeof(int) + sizeof(Scatter)); CHKPTR(to);
    to->proc  = 0; 
    to->n     = nx;
    to->next  = 0;
    to->slots = (int *) (to + 1);
    MEMCPY(to->slots,idy,nx*sizeof(int));
    from = (Scatter *) MALLOC(nx*sizeof(int) + sizeof(Scatter)); CHKPTR(from);
    from->proc  = 0; 
    from->n     = nx;
    from->next  = 0;
    from->slots = (int *) (from + 1);
    MEMCPY(from->slots,idx,nx*sizeof(int)); 
    ctx->to = to; ctx->from = from;
    *newctx = ctx;
    return 0;
  }
  else {
    SETERR(1,"Cannot generate such Scatter Context yet");
  }
}

/* ------------------------------------------------------------------*/
static int VecScatterOrAddBegin(Vec x,IS ix,Vec y,IS iy,VecScatterCtx *inctx,
                                int addv)
{
  int           ierr,i;
  VecScatterCtx ctx;
  Scatter       *from,*to;
  VALIDHEADER(x,VEC_COOKIE); VALIDHEADER(y,VEC_COOKIE);
  VALIDHEADER(ix,IS_COOKIE); VALIDHEADER(iy,IS_COOKIE);

  if (*inctx) ctx = *inctx;
  else {ierr = VecScatterCtxCreate(x,ix,y,iy,&ctx); CHKERR(ierr);}

  if (x->type == SEQVECTOR && y->type == SEQVECTOR) {
    DvVector *xx = (DvVector *) x->data;
    DvVector *yy = (DvVector *) y->data;
    Scalar   *xv = xx->array, *yv = yy->array;
    if (ctx->nto != 1 || ctx->nfrom != 1) {
      SETERR(1,"Wrong scatter context for Vector scatter");
    }
    from = ctx->from; to = ctx->to;
    if (addv == INSERTVALUES) {
      for ( i=0; i<from->n; i++ ) {
        yv[to->slots[i]] = xv[from->slots[i]];
      }
    }
    else {
      for ( i=0; i<from->n; i++ ) {
        yv[to->slots[i]] += xv[from->slots[i]];
      }
    }
  }
  else {
    SETERR(1,"Cannot handle that scatter yet");
  }

  *inctx = ctx;
  return 0;
}

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

  Output Parameters:
.  y - vector to scatter to 
.  inctx - if points to null then it is set and used

  Notes:
.   y[iy[i]] = x[ix[i]], for i=0,...,ni-1
.   destroy inctx with VecScatterCtxDestroy() when no longer needed.
@*/
int VecScatterBegin(Vec x,IS ix,Vec y,IS iy,VecScatterCtx *inctx)
{
  return VecScatterOrAddBegin(x,ix,y,iy,inctx,INSERTVALUES);
}

/*@
     VecScatterAddBegin  -  Scatters from one vector into another.
                            See VecScatterBegin().
  Input Parameters:
.  x - vector to scatter from
.  ix - indices of elements in x to take
.  iy - indices of locations in y to insert 

  Output Parameters:
.  y - vector to scatter to 

  Notes:
.   y[iy[i]] += x[ix[i]], for i=0,...,ni-1
@*/
int VecScatterAddBegin(Vec x,IS ix,Vec y,IS iy,VecScatterCtx *ctx)
{
  return VecScatterOrAddBegin(x,ix,y,iy,ctx,ADDVALUES);
}

/* --------------------------------------------------------------------*/
static int VecScatterOrAddEnd(Vec x,IS ix,Vec y,IS iy,VecScatterCtx *ctx,
                              int addv)
{
  return 0;
}

/*@
     VecScatterEnd  -  End scatter from one vector into another.
            Call after call to VecScatterBegin().

  Input Parameters:
.  x - vector to scatter from
.  ix - indices of elements in x to take
.  iy - indices of locations in y to insert 

  Output Parameters:
.  y - vector to scatter to 

  Notes:
.   y[iy[i]] = x[ix[i]], for i=0,...,ni-1
@*/
int VecScatterEnd(Vec x,IS ix,Vec y,IS iy,VecScatterCtx *ctx)
{
  return VecScatterOrAddEnd(x,ix,y,iy,ctx,INSERTVALUES);
}

/*@
     VecScatterAddEnd  -  End scatter from one vector into another.
            Call after call to VecScatterAddBegin().

  Input Parameters:
.  x - vector to scatter from
.  ix - indices of elements in x to take
.  iy - indices of locations in y to insert 

  Output Parameters:
.  y - vector to scatter to 

  Notes:
.   y[iy[i]] += x[ix[i]], for i=0,...,ni-1
@*/
int VecScatterAddEnd(Vec x,IS ix,Vec y,IS iy,VecScatterCtx *ctx)
{
  return VecScatterOrAddEnd(x,ix,y,iy,ctx,ADDVALUES);
}

/*@
     VecScatterCtxDestroy - Destroys a scatter context created by 
  either VecScatter() or VecScatterAdd().

  Input Parameters:
.   sctx - the scatter context

  Keywords: vector, scatter,
@*/
int VecScatterCtxDestroy( VecScatterCtx ctx )
{
  return 0;
}
