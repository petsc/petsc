/* $Id: dvec2.c,v 1.16 1995/09/30 19:26:35 bsmith Exp bsmith $ */

#include "inline/dot.h"
#include "inline/vmult.h"
#include "inline/setval.h"
#include "inline/copy.h"
#include "inline/axpy.h"
#include "vecimpl.h"             
#include "dvecimpl.h"   
#include "draw.h"          
#include "pinclude/pviewer.h"

static int VecMDot_Seq(int nv,Vec xin,Vec *y, Scalar *z )
{
  Vec_Seq *x = (Vec_Seq *)xin->data;
  register int n = x->n, i;
  Scalar   sum,*xx = x->array, *yy;

  /* This could be unrolled to reuse x[j] values */
  for (i=0; i<nv; i++) {
    sum = 0.0;
    yy = ((Vec_Seq *)(y[i]->data))->array;
    DOT(sum,xx,yy,n);
    z[i] = sum;
  }
  PLogFlops(nv*(2*x->n-1));
  return 0;
}

static int VecAMax_Seq(Vec xin,int* idx,double * z )
{
  Vec_Seq         *x = (Vec_Seq *) xin->data;
  register int    i, j=0, n = x->n;
  register double max = 0.0, tmp;
  Scalar          *xx = x->array;

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
  Vec_Seq         *x = (Vec_Seq *) xin->data;
  register int    i, j=0, n = x->n;
  register double max = -1.e40, tmp;
  Scalar          *xx = x->array;

  for (i=0; i<n; i++) {
#if defined(PETSC_COMPLEX)
    if ((tmp = real(*xx++)) > max) { j = i; max = tmp;}
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
  Vec_Seq         *x = (Vec_Seq *) xin->data;
  register int    i, j=0, n = x->n;
  register double min = 1.e40, tmp;
  Scalar          *xx = x->array;

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
  Vec_Seq      *x = (Vec_Seq *)xin->data;
  register int n = x->n;
  Scalar       *xx = x->array;

  SET(xx,n,*alpha);
  return 0;
}

static int VecMAXPY_Seq( int nv, Scalar *alpha, Vec yin, Vec *x )
{
  Vec_Seq      *y = (Vec_Seq *) yin->data;
  register int n = y->n;
  Scalar       *yy = y->array, *xx;
  int          j;

  PLogFlops(nv*2*n);
  for (j=0; j<nv; j++) {
    xx = ((Vec_Seq *)(x[j]->data))->array;
    if (alpha[j] == -1.0) {
      YMX(yy,xx,n);
    }
    else if (alpha[j] == 1.0) {
      YPX(yy,xx,n);
    }
    else if (alpha[j] != 0.0) {
      APXY(yy,alpha[j],xx,n);
    }
  }
  return 0;
}

static int VecAYPX_Seq(Scalar *alpha, Vec xin, Vec yin )
{
  Vec_Seq      *x = (Vec_Seq *)xin->data, *y = (Vec_Seq *)yin->data;
  register int n = x->n;
  Scalar       *xx = x->array, *yy = y->array;

  PLogFlops(2*n);
  AYPX(yy,*alpha,xx,n);
  return 0;
}

static int VecWAXPY_Seq(Scalar* alpha,Vec xin,Vec yin,Vec win )
{
  Vec_Seq      *w = (Vec_Seq *)win->data, *x = (Vec_Seq *)xin->data;
  Vec_Seq      *y = (Vec_Seq *)yin->data;
  register int i, n = x->n;
  Scalar       *xx = x->array, *yy = y->array, *ww = w->array;

  if (*alpha == 1.0) {
    PLogFlops(n);
    /* could call BLAS axpy after call to memcopy, but may be slower */
    for (i=0; i<n; i++) ww[i] = yy[i] + xx[i];
  }
  else if (*alpha == -1.0) {
    PLogFlops(n);
    for (i=0; i<n; i++) ww[i] = yy[i] - xx[i];
  }
  else {
    PLogFlops(2*n);
    for (i=0; i<n; i++) ww[i] = yy[i] + (*alpha) * xx[i];
  }
  return 0;
}

static int VecPMult_Seq( Vec xin, Vec yin, Vec win )
{
  Vec_Seq      *w = (Vec_Seq *)win->data, *x = (Vec_Seq *)xin->data;
  Vec_Seq      *y = (Vec_Seq *)yin->data;
  register int n = x->n, i;
  Scalar       *xx = x->array, *yy = y->array, *ww = w->array;

  PLogFlops(n);
  for (i=0; i<n; i++) ww[i] = xx[i] * yy[i];
  return 0;
}

static int VecPDiv_Seq(Vec xin,Vec yin,Vec win )
{
  Vec_Seq      *w = (Vec_Seq *)win->data, *x = (Vec_Seq *)xin->data;
  Vec_Seq      *y = (Vec_Seq *)yin->data;
  register int n = x->n, i;
  Scalar       *xx = x->array, *yy = y->array, *ww = w->array;

  PLogFlops(n);
  for (i=0; i<n; i++) ww[i] = xx[i] / yy[i];
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


 



