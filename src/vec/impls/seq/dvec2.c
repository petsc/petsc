
/* cannot have vcid because included in other files */


/*
   This file defines the vector operations in the simplest way possible.
   These routines are for standard double precision serial vectors.
 */

#include "sys/flog.h"
#include "inline/dot.h"
#include "inline/vmult.h"
#include "inline/setval.h"
#include "inline/copy.h"
#include "inline/axpy.h"
#include <math.h>
#include "vecimpl.h"             
#include "dvecimpl.h"   
#include "draw.h"          

static int VeiDVrange(Vec xin, int *low,int *high )
{
  DvVector *x = (DvVector *) xin->data;
  *low = 0; *high = x->n;
  return 0;
}
#include "viewer.h"

static int VeiDVview(PetscObject obj,Viewer ptr)
{
  Vec         xin = (Vec) obj;
  DvVector    *x = (DvVector *)xin->data;
  PetscObject vobj = (PetscObject) ptr;
  int         i, n = x->n, ierr;

  if (!vobj) {
    for (i=0; i<n; i++ ) {
#if defined(PETSC_COMPLEX)
      printf("%g + %gi\n",real(x->array[i]),imag(x->array[i]));
#else
      printf("%g\n",x->array[i]);
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
static int VeiDVmdot(int nv,Vec xin,Vec *y, Scalar *z )
{
  DvVector *x = (DvVector *)xin->data;
  register int n = x->n;
  register Scalar sum;
  Scalar   *xx = x->array, *yy;
  int      i;
  /* This could be unrolled to reuse x[j] values */
  for (i=0; i<nv; i++) {
    sum = 0.0;
    yy = ((DvVector *)(y[i]->data))->array;
    DOT(sum,xx,yy,n);
    z[i] = sum;
  }
  return 0;
}

static int VeiDVmax(Vec xin,int* idx,double * z )
{
  DvVector          *x = (DvVector *) xin->data;
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


static int VeiDVset(Scalar* alpha,Vec xin )
{
  DvVector  *x = (DvVector *)xin->data;
  register int n = x->n;
  Scalar   *xx = x->array;
  SET(xx,n,*alpha);
  return 0;
}

static int VeiDVmaxpy( int nv, Scalar *alpha, Vec yin, Vec *x )
{
  DvVector *y = (DvVector *) yin->data;
  register int n = y->n;
  Scalar *yy = y->array, *xx;
  int      j;
  for (j=0; j<nv; j++) {
    xx = ((DvVector *)(x[j]->data))->array;
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

static int VeiDVaypx(Scalar *alpha, Vec xin, Vec yin )
{
  DvVector *x = (DvVector *)xin->data, *y = (DvVector *)yin->data;
  register int n = x->n;
  Scalar   *xx = x->array, *yy = y->array;
  AYPX(yy,*alpha,xx,n);
  return 0;
}

static int VeiDVwaxpy(Scalar* alpha,Vec xin,Vec yin,Vec win )
{
  DvVector *w = (DvVector *)win->data, *x = (DvVector *)xin->data;
  DvVector *y = (DvVector *)yin->data;
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

static int VeiDVpmult( Vec xin, Vec yin, Vec win )
{
  DvVector *w = (DvVector *)win->data, *x = (DvVector *)xin->data;
  DvVector *y = (DvVector *)yin->data;
  register int n = x->n, i;
  Scalar   *xx = x->array, *yy = y->array, *ww = w->array;
  for (i=0; i<n; i++) ww[i] = xx[i] * yy[i];
  return 0;
}

static int VeiDVpdiv(Vec xin,Vec yin,Vec win )
{
  DvVector *w = (DvVector *)win->data, *x = (DvVector *)xin->data;
  DvVector *y = (DvVector *)yin->data;
  register int n = x->n, i;
  Scalar   *xx = x->array, *yy = y->array, *ww = w->array;
  for (i=0; i<n; i++) ww[i] = xx[i] / yy[i];
  return 0;
}

#include "inline/spops.h"
static int VeiDVinsertvalues(Vec xin, int ni, int *ix,Scalar* y,InsertMode m)
{
  DvVector *x = (DvVector *)xin->data;
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

static int VeiDVgetarray(Vec vin,Scalar **a)
{
  DvVector *v = (DvVector *)vin->data;
  *a =  v->array; return 0;
}

static int VeiDVsize(Vec vin,int *size)
{
  DvVector *v = (DvVector *)vin->data;
  *size = v->n; return 0;
}
