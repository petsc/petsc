
/* cannot have vcid because included in other files */

/*
     These are routines shared by sequential vectors and BLAS sequential 
   vectors.
*/

#include "inline/dot.h"
#include "inline/vmult.h"
#include "inline/setval.h"
#include "inline/copy.h"
#include "inline/axpy.h"
#include <math.h>
#include "vecimpl.h"             
#include "dvecimpl.h"   
#include "draw.h"          

static int VecGetOwnershipRange_Seq(Vec xin, int *low,int *high )
{
  Vec_Seq *x = (Vec_Seq *) xin->data;
  *low = 0; *high = x->n;
  return 0;
}
#include "viewer.h"

static int VecView_Seq(PetscObject obj,Viewer ptr)
{
  Vec         xin = (Vec) obj;
  Vec_Seq    *x = (Vec_Seq *)xin->data;
  PetscObject vobj = (PetscObject) ptr;
  int         i, n = x->n, ierr;
  FILE        *fd;

  if (!ptr) { /* so that viewers may be used from debuggers */
    ptr = STDOUT_VIEWER; vobj = (PetscObject) ptr;
  }

  if (vobj->cookie == VIEWER_COOKIE && ((vobj->type == FILE_VIEWER) ||
                                       (vobj->type == FILES_VIEWER)))  {
    fd = ViewerFileGetPointer(ptr);
    for (i=0; i<n; i++ ) {
#if defined(PETSC_COMPLEX)
      fprintf(fd,"%g + %gi\n",real(x->array[i]),imag(x->array[i]));
#else
      fprintf(fd,"%g\n",x->array[i]);
#endif
    }
  }
#if !defined(PETSC_COMPLEX)
  else if (vobj->cookie == LG_COOKIE){
    DrawLGCtx lg = (DrawLGCtx) ptr;
    DrawCtx   win;
    double    *xx;
    DrawLGGetDrawCtx(lg,&win);
    DrawLGReset(lg);
    xx = (double *) MALLOC( n*sizeof(double) ); CHKPTR(xx);
    for ( i=0; i<n; i++ ) {
      xx[i] = (double) i;
    }
    DrawLGAddPoints(lg,n,&xx,&x->array);
    FREE(xx);
    DrawLG(lg);
    DrawSyncFlush(win);
  }
  else if (vobj->cookie == DRAW_COOKIE) {
    DrawCtx   win = (DrawCtx) ptr;
    DrawLGCtx lg;
    ierr = DrawLGCreate(win,1,&lg); CHKERR(ierr);
    ierr = VecView(xin,(Viewer) lg); CHKERR(ierr);
    DrawLGDestroy(lg);
  }
  else if (vobj->cookie == VIEWER_COOKIE && vobj->type == MATLAB_VIEWER) {
    return ViewerMatlabPutArray(ptr,x->n,1,x->array); 
  }
#endif
  return 0;
}
static int VecMDot_Seq(int nv,Vec xin,Vec *y, Scalar *z )
{
  Vec_Seq *x = (Vec_Seq *)xin->data;
  register int n = x->n;
  register Scalar sum;
  Scalar   *xx = x->array, *yy;
  int      i;
  /* This could be unrolled to reuse x[j] values */
  for (i=0; i<nv; i++) {
    sum = 0.0;
    yy = ((Vec_Seq *)(y[i]->data))->array;
    DOT(sum,xx,yy,n);
    z[i] = sum;
  }
  return 0;
}

static int VecAMax_Seq(Vec xin,int* idx,double * z )
{
  Vec_Seq          *x = (Vec_Seq *) xin->data;
  register int i, j=0, n = x->n;
  register double max = 0.0, tmp;
  Scalar   *xx = x->array;
  for (i=0; i<n; i++) {
#if defined(PETSC_COMPLEX)
    if ((tmp = abs(*xx++)) > max) max = tmp;
#else
    if ( (tmp = *xx++) > 0.0 ) { if (tmp > max) { j = i; max = tmp; } }
    else                       { if (-tmp > max) { j = i; max = -tmp; } }
#endif
  }
  *z   = max;
  if (idx) *idx = j;
  return 0;
}

static int VecMax_Seq(Vec xin,int* idx,double * z )
{
  Vec_Seq          *x = (Vec_Seq *) xin->data;
  register int i, j=0, n = x->n;
  register double max = -1.e40, tmp;
  Scalar    *xx = x->array;
  for (i=0; i<n; i++) {
#if defined(PETSC_COMPLEX)
    IF ((tmp = real(*xx++)) > max) { j = i; max = tmp;}
#else
    if ((tmp = *xx++) > max) { j = i; max = tmp; } 
#endif
  }
  *z   = max;
  if (idx) *idx = j;
  return 0;
}

static int VecMin_Seq(Vec xin,int* idx,double * z )
{
  Vec_Seq          *x = (Vec_Seq *) xin->data;
  register int i, j=0, n = x->n;
  register double min = 1.e40, tmp;
  Scalar   *xx = x->array;
  for ( i=0; i<n; i++ ) {

#if defined(PETSC_COMPLEX)
    if ((tmp = real(*xx++)) < min) { j = i; min = tmp;}
#else
    if ((tmp = *xx++) < min) { j = i; min = tmp; } 
#endif
  }
  *z   = min;
  if (idx) *idx = j;
  return 0;
}


static int VecSet_Seq(Scalar* alpha,Vec xin )
{
  Vec_Seq  *x = (Vec_Seq *)xin->data;
  register int n = x->n;
  Scalar   *xx = x->array;
  SET(xx,n,*alpha);
  return 0;
}

static int VecMAXPY_Seq( int nv, Scalar *alpha, Vec yin, Vec *x )
{
  Vec_Seq *y = (Vec_Seq *) yin->data;
  register int n = y->n;
  Scalar *yy = y->array, *xx;
  int      j;
  for (j=0; j<nv; j++) {
    xx = ((Vec_Seq *)(x[j]->data))->array;
    /* This should really look at the case alpha = +1 as well */
    if (alpha[j] == -1.0) {
	YMX(yy,xx,n);
    }
    else {
	APXY(yy,alpha[j],xx,n);
    }
  }
  return 0;
}

static int VecAYPX_Seq(Scalar *alpha, Vec xin, Vec yin )
{
  Vec_Seq *x = (Vec_Seq *)xin->data, *y = (Vec_Seq *)yin->data;
  register int n = x->n;
  Scalar   *xx = x->array, *yy = y->array;
  AYPX(yy,*alpha,xx,n);
  return 0;
}

static int VecWAXPY_Seq(Scalar* alpha,Vec xin,Vec yin,Vec win )
{
  Vec_Seq *w = (Vec_Seq *)win->data, *x = (Vec_Seq *)xin->data;
  Vec_Seq *y = (Vec_Seq *)yin->data;
  register int i, n = x->n;
  Scalar   *xx = x->array, *yy = y->array, *ww = w->array;
  if (*alpha == 1.0) {
    for (i=0; i<n; i++) ww[i] = yy[i] + xx[i];
  }
  else {
    for (i=0; i<n; i++) ww[i] = yy[i] + (*alpha) * xx[i];
  }
  return 0;
}

static int VecPMult_Seq( Vec xin, Vec yin, Vec win )
{
  Vec_Seq *w = (Vec_Seq *)win->data, *x = (Vec_Seq *)xin->data;
  Vec_Seq *y = (Vec_Seq *)yin->data;
  register int n = x->n, i;
  Scalar   *xx = x->array, *yy = y->array, *ww = w->array;
  for (i=0; i<n; i++) ww[i] = xx[i] * yy[i];
  return 0;
}

static int VecPDiv_Seq(Vec xin,Vec yin,Vec win )
{
  Vec_Seq *w = (Vec_Seq *)win->data, *x = (Vec_Seq *)xin->data;
  Vec_Seq *y = (Vec_Seq *)yin->data;
  register int n = x->n, i;
  Scalar   *xx = x->array, *yy = y->array, *ww = w->array;
  for (i=0; i<n; i++) ww[i] = xx[i] / yy[i];
  return 0;
}

#include "inline/spops.h"
static int VecSetValues_Seq(Vec xin, int ni, int *ix,Scalar* y,InsertMode m)
{
  Vec_Seq *x = (Vec_Seq *)xin->data;
  Scalar   *xx = x->array;
  int      i;

  if (m == InsertValues) {
    for ( i=0; i<ni; i++ ) {
#if defined(PETSC_DEBUG)
      if (ix[i] < 0 || ix[i] >= x->n) SETERR(1,"Index out of range");
#endif
      xx[ix[i]] = y[i];
    }
  }
  else {
    for ( i=0; i<ni; i++ ) {
#if defined(PETSC_DEBUG)
      if (ix[i] < 0 || ix[i] >= x->n) SETERR(1,"Index out of range");
#endif
      xx[ix[i]] += y[i];
    }  
  }  
  return 0;
}

static int VecGetArray_Seq(Vec vin,Scalar **a)
{
  Vec_Seq *v = (Vec_Seq *)vin->data;
  *a =  v->array; return 0;
}

static int VecGetSize_Seq(Vec vin,int *size)
{
  Vec_Seq *v = (Vec_Seq *)vin->data;
  *size = v->n; return 0;
}

static int VecDestroy_Seq(PetscObject obj )
{
  Vec      v = (Vec ) obj;
#if defined(PETSC_LOG)
  PLogObjectState(obj,"Rows %d",((Vec_Seq *)v->data)->n);
#endif
  FREE(v->data);
  PLogObjectDestroy(v);
  PETSCHEADERDESTROY(v); 
  return 0;
}
 
