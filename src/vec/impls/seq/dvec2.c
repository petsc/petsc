/* $Id: dvec2.c,v 1.29 1996/05/08 01:02:35 balay Exp balay $ */

/* 
   Defines some vector operation functions that are shared by 
  sequential and parallel vectors.
*/
#include "src/inline/dot.h"
#include "src/inline/setval.h"
#include "src/inline/axpy.h"
#include "vecimpl.h"             
#include "dvecimpl.h"   
#include "draw.h"          
#include "pinclude/pviewer.h"

/*
static int VecMDot_Seq(int nv,Vec xin,Vec *y, Scalar *z )
{
  Vec_Seq *x = (Vec_Seq *)xin->data;
  register int n = x->n, i;
  Scalar   sum,*xx = x->array, *yy;

  for (i=0; i<nv; i++) {
    sum = 0.0;
    yy = ((Vec_Seq *)(y[i]->data))->array;
    DOT(sum,xx,yy,n);
    z[i] = sum;
  }
  PLogFlops(nv*(2*x->n-1));
  return 0;
}
*/
static int VecMDot_Seq(int nv,Vec xin,Vec *yin, Scalar *z )
{
  Vec_Seq *xv = (Vec_Seq *)xin->data;
  register int n = xv->n,i,j,nv_rem,j_rem;
  Scalar   sum0,sum1,sum2,sum3,*y0,*y1,*y2,*y3,x0,x1,x2,x3,*x;
  Vec      *yy;
  
  /*
  for (i=0; i<nv_rem; i++) {
    sum = 0.0;
    y = ((Vec_Seq *)(yin[i]->data))->array;
    DOT(sum,x,y,n);
    *z++ = sum;
  } 
  i  = nv - nv_rem;
  yy = yin + nv_rem;
*/

  sum0 = 0;
  sum1 = 0;
  sum2 = 0;

  i      = nv;
  nv_rem = nv&0x3;
  yy     = yin;
  j    = n;
  x    = xv->array;

  switch (nv_rem) {
  case 3:
  y0   = ((Vec_Seq *)(yy[0]->data))->array;
  y1   = ((Vec_Seq *)(yy[1]->data))->array;
  y2   = ((Vec_Seq *)(yy[2]->data))->array;
    switch (j_rem=j&0x3) {
    case 3: 
      x2 = x[2]; 
      sum0 += x2*y0[2]; sum1 += x2*y1[2]; 
      sum2 += x2*y2[2]; 
    case 2: 
      x1 = x[1]; 
      sum0 += x1*y0[1]; sum1 += x1*y1[1]; 
      sum2 += x1*y2[1]; 
    case 1: 
      x0 = x[0]; 
      sum0 += x0*y0[0]; sum1 += x0*y1[0]; 
      sum2 += x0*y2[0]; 
    case 0: 
      x  += j_rem;
      y0 += j_rem;
      y1 += j_rem;
      y2 += j_rem;
      j  -= j_rem;
      break;
    }
    while (j>0) {
      x0 = x[0];
      x1 = x[1];
      x2 = x[2];
      x3 = x[3];
      x += 4;
      
      sum0 += x0*y0[0] + x1*y0[1] + x2*y0[2] + x3*y0[3]; y0+=4;
      sum1 += x0*y1[0] + x1*y1[1] + x2*y1[2] + x3*y1[3]; y1+=4;
      sum2 += x0*y2[0] + x1*y2[1] + x2*y2[2] + x3*y2[3]; y2+=4;
      j -= 4;
    }
    z[0] = sum0;
    z[1] = sum1;
    z[2] = sum2;
    break;
  case 2:
  y0   = ((Vec_Seq *)(yy[0]->data))->array;
  y1   = ((Vec_Seq *)(yy[1]->data))->array;
    switch (j_rem=j&0x3) {
    case 3: 
      x2 = x[2]; 
      sum0 += x2*y0[2]; sum1 += x2*y1[2]; 
    case 2: 
      x1 = x[1]; 
      sum0 += x1*y0[1]; sum1 += x1*y1[1]; 
    case 1: 
      x0 = x[0]; 
      sum0 += x0*y0[0]; sum1 += x0*y1[0]; 
    case 0: 
      x  += j_rem;
      y0 += j_rem;
      y1 += j_rem;
      j  -= j_rem;
      break;
    }
    while (j>0) {
      x0 = x[0];
      x1 = x[1];
      x2 = x[2];
      x3 = x[3];
      x += 4;
      
      sum0 += x0*y0[0] + x1*y0[1] + x2*y0[2] + x3*y0[3]; y0+=4;
      sum1 += x0*y1[0] + x1*y1[1] + x2*y1[2] + x3*y1[3]; y1+=4;
      j -= 4;
    }
    z[0] = sum0;
    z[1] = sum1;
 
    break;
  case 1:
  y0   = ((Vec_Seq *)(yy[0]->data))->array;
    switch (j_rem=j&0x3) {
    case 3: 
      x2 = x[2]; sum0 += x2*y0[2];
    case 2: 
      x1 = x[1]; sum0 += x1*y0[1];
    case 1: 
      x0 = x[0]; sum0 += x0*y0[0];
    case 0: 
      x  += j_rem;
      y0 += j_rem;
      j  -= j_rem;
      break;
    }
    while (j>0) {
      sum0 += x[0]*y0[0] + x[1]*y0[1] + x[2]*y0[2] + x[3]*y0[3]; y0+=4;
      j -= 4; x+=4;
    }
    z[0] = sum0;

    break;
  case 0:
    break;
  }
  z  += nv_rem;
  i  -= nv_rem;
  yy += nv_rem;

  while (i >0) {
    sum0 = 0;
    sum1 = 0;
    sum2 = 0;
    sum3 = 0;
    y0   = ((Vec_Seq *)(yy[0]->data))->array;
    y1   = ((Vec_Seq *)(yy[1]->data))->array;
    y2   = ((Vec_Seq *)(yy[2]->data))->array;
    y3   = ((Vec_Seq *)(yy[3]->data))->array;
    yy  += 4;

    j = n;
    x = xv->array;
    switch (j_rem=j&0x3) {
    case 3: 
      x2 = x[2]; 
      sum0 += x2*y0[2]; sum1 += x2*y1[2]; 
      sum2 += x2*y2[2]; sum3 += x2*y3[2];
    case 2: 
      x1 = x[1]; 
      sum0 += x1*y0[1]; sum1 += x1*y1[1]; 
      sum2 += x1*y2[1]; sum3 += x1*y3[1];
    case 1: 
      x0 = x[0]; 
      sum0 += x0*y0[0]; sum1 += x0*y1[0]; 
      sum2 += x0*y2[0]; sum3 += x0*y3[0];
    case 0: 
      x  += j_rem;
      y0 += j_rem;
      y1 += j_rem;
      y2 += j_rem;
      y3 += j_rem;
      j  -= j_rem;
      break;
    }
    while (j>0) {
      x0 = x[0];
      x1 = x[1];
      x2 = x[2];
      x3 = x[3];
      x += 4;
      
      sum0 += x0*y0[0] + x1*y0[1] + x2*y0[2] + x3*y0[3]; y0+=4;
      sum1 += x0*y1[0] + x1*y1[1] + x2*y1[2] + x3*y1[3]; y1+=4;
      sum2 += x0*y2[0] + x1*y2[1] + x2*y2[2] + x3*y2[3]; y2+=4;
      sum3 += x0*y3[0] + x1*y3[1] + x2*y3[2] + x3*y3[3]; y3+=4;
      j -= 4;
    }
    z[0] = sum0;
    z[1] = sum1;
    z[2] = sum2;
    z[3] = sum3;
    z   += 4;
    i   -= 4;
  }
  PLogFlops(nv*(2*x->n-1));
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

static int VecSet_Seq(Scalar* alpha,Vec xin)
{
  Vec_Seq      *x = (Vec_Seq *)xin->data;
  register int n = x->n;
  Scalar       *xx = x->array, oalpha = *alpha;

  if (oalpha == 0.0) {
    PetscMemzero(xx,n*sizeof(Scalar));
  }
  else {
    SET(xx,n,oalpha);
  }
  return 0;
}

static int VecSetRandom_Seq(PetscRandom r,Vec xin)
{
  Vec_Seq      *x = (Vec_Seq *)xin->data;
  register int n = x->n;
  int          i, ierr;
  Scalar       *xx = x->array;

  for (i=0; i<n; i++) {ierr = PetscRandomGetValue(r,&xx[i]); CHKERRQ(ierr);}
  return 0;
}

static int VecMAXPY_Seq( int nv, Scalar *alpha, Vec yin, Vec *x )
{
  Vec_Seq      *y = (Vec_Seq *) yin->data;
  register int n = y->n;
  int          j;
  Scalar       *yy = y->array, *xx,oalpha;

  PLogFlops(nv*2*n);
  for (j=0; j<nv; j++) {
    xx     = ((Vec_Seq *)(x[j]->data))->array;
    oalpha = alpha[j];
    if (oalpha == -1.0) {
      YMX(yy,xx,n);
    }
    else if (oalpha == 1.0) {
      YPX(yy,xx,n);
    }
    else if (oalpha != 0.0) {
      APXY(yy,oalpha,xx,n);
    }
  }
  return 0;
}

static int VecAYPX_Seq(Scalar *alpha, Vec xin, Vec yin )
{
  Vec_Seq      *x = (Vec_Seq *)xin->data, *y = (Vec_Seq *)yin->data;
  register int i,n = x->n;
  Scalar       *xx = x->array, *yy = y->array, oalpha = *alpha;

  PLogFlops(2*n);
  for ( i=0; i<n; i++ ) {
    yy[i] = xx[i] + oalpha*yy[i];
  }
  return 0;
}

/*
   IBM ESSL contains a routine dzaxpy() that is our WAXPY() but it appears
  to be slower then a regular C loop. Hence we do not include it.
  void ?zaxpy(int*,Scalar*,Scalar*,int*,Scalar*,int*,Scalar*,int*);
*/

static int VecWAXPY_Seq(Scalar* alpha,Vec xin,Vec yin,Vec win )
{
  Vec_Seq      *w = (Vec_Seq *)win->data, *x = (Vec_Seq *)xin->data;
  Vec_Seq      *y = (Vec_Seq *)yin->data;
  register int i, n = x->n;
  Scalar       *xx = x->array, *yy = y->array, *ww = w->array, oalpha = *alpha;

  if (oalpha == 1.0) {
    PLogFlops(n);
    /* could call BLAS axpy after call to memcopy, but may be slower */
    for (i=0; i<n; i++) ww[i] = yy[i] + xx[i];
  }
  else if (oalpha == -1.0) {
    PLogFlops(n);
    for (i=0; i<n; i++) ww[i] = yy[i] - xx[i];
  }
  else if (oalpha == 0.0) {
    PetscMemcpy(ww,yy,n*sizeof(Scalar));
  }
  else {
    for (i=0; i<n; i++) ww[i] = yy[i] + oalpha * xx[i];
    PLogFlops(2*n);
  }
  return 0;
}

static int VecPointwiseMult_Seq( Vec xin, Vec yin, Vec win )
{
  Vec_Seq      *w = (Vec_Seq *)win->data, *x = (Vec_Seq *)xin->data;
  Vec_Seq      *y = (Vec_Seq *)yin->data;
  register int n = x->n, i;
  Scalar       *xx = x->array, *yy = y->array, *ww = w->array;

  PLogFlops(n);
  for (i=0; i<n; i++) ww[i] = xx[i] * yy[i];
  return 0;
}

static int VecPointwiseDivide_Seq(Vec xin,Vec yin,Vec win )
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


 



