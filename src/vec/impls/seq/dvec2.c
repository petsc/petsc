#ifdef PETSC_RCS_HEADER
static char vcid[] =  "$Id: dvec2.c,v 1.60 1999/03/09 23:13:39 bsmith Exp bsmith $"
#endif

/* 
   Defines some vector operation functions that are shared by 
  sequential and parallel vectors.
*/
#include "src/vec/impls/dvecimpl.h"   
#include "src/inline/dot.h"
#include "src/inline/setval.h"
#include "src/inline/axpy.h"

#undef __FUNC__  
#define __FUNC__ "VecMDot_Seq"
#if defined(USE_FORTRAN_KERNEL_MDOT)
int VecMDot_Seq(int nv,Vec xin,const Vec yin[], Scalar *z )
{
  Vec_Seq *xv = (Vec_Seq *)xin->data;
  register int i,nv_rem;
  int      n = xv->n;
  Scalar   sum0,sum1,sum2,sum3,*yy0,*yy1,*yy2,*yy3,x0,x1,x2,x3,*x;
  Vec      *yy;

  PetscFunctionBegin;
  sum0 = 0;
  sum1 = 0;
  sum2 = 0;

  i      = nv;
  nv_rem = nv&0x3;
  yy     = (Vec *) yin;
  x      = xv->array;

  switch (nv_rem) {
  case 3:
    yy0   = ((Vec_Seq *)(yy[0]->data))->array;
    yy1   = ((Vec_Seq *)(yy[1]->data))->array;
    yy2   = ((Vec_Seq *)(yy[2]->data))->array;
    fortranmdot3_(x,yy0,yy1,yy2,&n,&sum0,&sum1,&sum2);
    z[0] = sum0;
    z[1] = sum1;
    z[2] = sum2;
    break;
  case 2:
    yy0   = ((Vec_Seq *)(yy[0]->data))->array;
    yy1   = ((Vec_Seq *)(yy[1]->data))->array;
    fortranmdot2_(x,yy0,yy1,&n,&sum0,&sum1);
    z[0] = sum0;
    z[1] = sum1;
    break;
  case 1:
    yy0   = ((Vec_Seq *)(yy[0]->data))->array;
    fortranmdot1_(x,yy0,&n,&sum0);
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
    yy0   = ((Vec_Seq *)(yy[0]->data))->array;
    yy1   = ((Vec_Seq *)(yy[1]->data))->array;
    yy2   = ((Vec_Seq *)(yy[2]->data))->array;
    yy3   = ((Vec_Seq *)(yy[3]->data))->array;
    yy  += 4;
    fortranmdot4_(x,yy0,yy1,yy2,yy3,&n,&sum0,&sum1,&sum2,&sum3);
    z[0] = sum0;
    z[1] = sum1;
    z[2] = sum2;
    z[3] = sum3;
    z   += 4;
    i   -= 4;
  }
  PLogFlops(nv*(2*xv->n-1));
  PetscFunctionReturn(0);
}

#else
int VecMDot_Seq(int nv,Vec xin,const Vec yin[], Scalar * restrict z )
{
  Vec_Seq *xv = (Vec_Seq *)xin->data;
  register int n = xv->n,i,j,nv_rem,j_rem;
  Scalar   sum0,sum1,sum2,sum3,x0,x1,x2,x3,* restrict x;
  Scalar   * restrict yy0, * restrict yy1, * restrict yy2, *restrict yy3; 
  Vec      *yy;
  
/*
  for (i=0; i<nv; i++) {
    sum = 0.0;
    yy = ((Vec_Seq *)(y[i]->data))->array;
    DOT(sum,xx,yy,n);
    z[i] = sum;
  }
*/

  PetscFunctionBegin;
  sum0 = 0;
  sum1 = 0;
  sum2 = 0;

  i      = nv;
  nv_rem = nv&0x3;
  yy     = (Vec *)yin;
  j    = n;
  x    = xv->array;

  switch (nv_rem) {
  case 3:
  yy0   = ((Vec_Seq *)(yy[0]->data))->array;
  yy1   = ((Vec_Seq *)(yy[1]->data))->array;
  yy2   = ((Vec_Seq *)(yy[2]->data))->array;
    switch (j_rem=j&0x3) {
    case 3: 
      x2 = x[2]; 
      sum0 += x2*PetscConj(yy0[2]); sum1 += x2*PetscConj(yy1[2]); 
      sum2 += x2*PetscConj(yy2[2]); 
    case 2: 
      x1 = x[1]; 
      sum0 += x1*PetscConj(yy0[1]); sum1 += x1*PetscConj(yy1[1]); 
      sum2 += x1*PetscConj(yy2[1]); 
    case 1: 
      x0 = x[0]; 
      sum0 += x0*PetscConj(yy0[0]); sum1 += x0*PetscConj(yy1[0]); 
      sum2 += x0*PetscConj(yy2[0]); 
    case 0: 
      x   += j_rem;
      yy0 += j_rem;
      yy1 += j_rem;
      yy2 += j_rem;
      j   -= j_rem;
      break;
    }
    while (j>0) {
      x0 = x[0];
      x1 = x[1];
      x2 = x[2];
      x3 = x[3];
      x += 4;
      
      sum0 += x0*PetscConj(yy0[0]) + x1*PetscConj(yy0[1]) + x2*PetscConj(yy0[2]) + x3*PetscConj(yy0[3]); yy0+=4;
      sum1 += x0*PetscConj(yy1[0]) + x1*PetscConj(yy1[1]) + x2*PetscConj(yy1[2]) + x3*PetscConj(yy1[3]); yy1+=4;
      sum2 += x0*PetscConj(yy2[0]) + x1*PetscConj(yy2[1]) + x2*PetscConj(yy2[2]) + x3*PetscConj(yy2[3]); yy2+=4;
      j -= 4;
    }
    z[0] = sum0;
    z[1] = sum1;
    z[2] = sum2;
    break;
  case 2:
  yy0   = ((Vec_Seq *)(yy[0]->data))->array;
  yy1   = ((Vec_Seq *)(yy[1]->data))->array;
    switch (j_rem=j&0x3) {
    case 3: 
      x2 = x[2]; 
      sum0 += x2*PetscConj(yy0[2]); sum1 += x2*PetscConj(yy1[2]); 
    case 2: 
      x1 = x[1]; 
      sum0 += x1*PetscConj(yy0[1]); sum1 += x1*PetscConj(yy1[1]); 
    case 1: 
      x0 = x[0]; 
      sum0 += x0*PetscConj(yy0[0]); sum1 += x0*PetscConj(yy1[0]); 
    case 0: 
      x   += j_rem;
      yy0 += j_rem;
      yy1 += j_rem;
      j   -= j_rem;
      break;
    }
    while (j>0) {
      x0 = x[0];
      x1 = x[1];
      x2 = x[2];
      x3 = x[3];
      x += 4;
      
      sum0 += x0*PetscConj(yy0[0]) + x1*PetscConj(yy0[1]) + x2*PetscConj(yy0[2]) + x3*PetscConj(yy0[3]); yy0+=4;
      sum1 += x0*PetscConj(yy1[0]) + x1*PetscConj(yy1[1]) + x2*PetscConj(yy1[2]) + x3*PetscConj(yy1[3]); yy1+=4;
      j -= 4;
    }
    z[0] = sum0;
    z[1] = sum1;
 
    break;
  case 1:
  yy0   = ((Vec_Seq *)(yy[0]->data))->array;
    switch (j_rem=j&0x3) {
    case 3: 
      x2 = x[2]; sum0 += x2*PetscConj(yy0[2]);
    case 2: 
      x1 = x[1]; sum0 += x1*PetscConj(yy0[1]);
    case 1: 
      x0 = x[0]; sum0 += x0*PetscConj(yy0[0]);
    case 0: 
      x   += j_rem;
      yy0 += j_rem;
      j   -= j_rem;
      break;
    }
    while (j>0) {
      sum0 += x[0]*PetscConj(yy0[0]) + x[1]*PetscConj(yy0[1])
            + x[2]*PetscConj(yy0[2]) + x[3]*PetscConj(yy0[3]); 
      yy0+=4;
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
    yy0   = ((Vec_Seq *)(yy[0]->data))->array;
    yy1   = ((Vec_Seq *)(yy[1]->data))->array;
    yy2   = ((Vec_Seq *)(yy[2]->data))->array;
    yy3   = ((Vec_Seq *)(yy[3]->data))->array;
    yy  += 4;

    j = n;
    x = xv->array;
    switch (j_rem=j&0x3) {
    case 3: 
      x2 = x[2]; 
      sum0 += x2*PetscConj(yy0[2]); sum1 += x2*PetscConj(yy1[2]); 
      sum2 += x2*PetscConj(yy2[2]); sum3 += x2*PetscConj(yy3[2]);
    case 2: 
      x1 = x[1]; 
      sum0 += x1*PetscConj(yy0[1]); sum1 += x1*PetscConj(yy1[1]); 
      sum2 += x1*PetscConj(yy2[1]); sum3 += x1*PetscConj(yy3[1]);
    case 1: 
      x0 = x[0]; 
      sum0 += x0*PetscConj(yy0[0]); sum1 += x0*PetscConj(yy1[0]); 
      sum2 += x0*PetscConj(yy2[0]); sum3 += x0*PetscConj(yy3[0]);
    case 0: 
      x   += j_rem;
      yy0 += j_rem;
      yy1 += j_rem;
      yy2 += j_rem;
      yy3 += j_rem;
      j   -= j_rem;
      break;
    }
    while (j>0) {
      x0 = x[0];
      x1 = x[1];
      x2 = x[2];
      x3 = x[3];
      x += 4;
      
      sum0 += x0*PetscConj(yy0[0]) + x1*PetscConj(yy0[1]) + x2*PetscConj(yy0[2]) + x3*PetscConj(yy0[3]); yy0+=4;
      sum1 += x0*PetscConj(yy1[0]) + x1*PetscConj(yy1[1]) + x2*PetscConj(yy1[2]) + x3*PetscConj(yy1[3]); yy1+=4;
      sum2 += x0*PetscConj(yy2[0]) + x1*PetscConj(yy2[1]) + x2*PetscConj(yy2[2]) + x3*PetscConj(yy2[3]); yy2+=4;
      sum3 += x0*PetscConj(yy3[0]) + x1*PetscConj(yy3[1]) + x2*PetscConj(yy3[2]) + x3*PetscConj(yy3[3]); yy3+=4;
      j -= 4;
    }
    z[0] = sum0;
    z[1] = sum1;
    z[2] = sum2;
    z[3] = sum3;
    z   += 4;
    i   -= 4;
  }
  PLogFlops(nv*(2*xv->n-1));
  PetscFunctionReturn(0);
}
#endif

/* ----------------------------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ "VecMTDot_Seq"
int VecMTDot_Seq(int nv,Vec xin,const Vec yin[], Scalar *z )
{
  Vec_Seq *xv = (Vec_Seq *)xin->data;
  register int n = xv->n,i,j,nv_rem,j_rem;
  Scalar   sum0,sum1,sum2,sum3,*yy0,*yy1,*yy2,*yy3,x0,x1,x2,x3,*x;
  Vec      *yy;
  
  PetscFunctionBegin;
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
  yy     = (Vec*)yin;
  j    = n;
  x    = xv->array;

  switch (nv_rem) {
  case 3:
  yy0   = ((Vec_Seq *)(yy[0]->data))->array;
  yy1   = ((Vec_Seq *)(yy[1]->data))->array;
  yy2   = ((Vec_Seq *)(yy[2]->data))->array;
    switch (j_rem=j&0x3) {
    case 3: 
      x2 = x[2]; 
      sum0 += x2*yy0[2]; sum1 += x2*yy1[2]; 
      sum2 += x2*yy2[2]; 
    case 2: 
      x1 = x[1]; 
      sum0 += x1*yy0[1]; sum1 += x1*yy1[1]; 
      sum2 += x1*yy2[1]; 
    case 1: 
      x0 = x[0]; 
      sum0 += x0*yy0[0]; sum1 += x0*yy1[0]; 
      sum2 += x0*yy2[0]; 
    case 0: 
      x  += j_rem;
      yy0 += j_rem;
      yy1 += j_rem;
      yy2 += j_rem;
      j  -= j_rem;
      break;
    }
    while (j>0) {
      x0 = x[0];
      x1 = x[1];
      x2 = x[2];
      x3 = x[3];
      x += 4;
      
      sum0 += x0*yy0[0] + x1*yy0[1] + x2*yy0[2] + x3*yy0[3]; yy0+=4;
      sum1 += x0*yy1[0] + x1*yy1[1] + x2*yy1[2] + x3*yy1[3]; yy1+=4;
      sum2 += x0*yy2[0] + x1*yy2[1] + x2*yy2[2] + x3*yy2[3]; yy2+=4;
      j -= 4;
    }
    z[0] = sum0;
    z[1] = sum1;
    z[2] = sum2;
    break;
  case 2:
  yy0   = ((Vec_Seq *)(yy[0]->data))->array;
  yy1   = ((Vec_Seq *)(yy[1]->data))->array;
    switch (j_rem=j&0x3) {
    case 3: 
      x2 = x[2]; 
      sum0 += x2*yy0[2]; sum1 += x2*yy1[2]; 
    case 2: 
      x1 = x[1]; 
      sum0 += x1*yy0[1]; sum1 += x1*yy1[1]; 
    case 1: 
      x0 = x[0]; 
      sum0 += x0*yy0[0]; sum1 += x0*yy1[0]; 
    case 0: 
      x  += j_rem;
      yy0 += j_rem;
      yy1 += j_rem;
      j  -= j_rem;
      break;
    }
    while (j>0) {
      x0 = x[0];
      x1 = x[1];
      x2 = x[2];
      x3 = x[3];
      x += 4;
      
      sum0 += x0*yy0[0] + x1*yy0[1] + x2*yy0[2] + x3*yy0[3]; yy0+=4;
      sum1 += x0*yy1[0] + x1*yy1[1] + x2*yy1[2] + x3*yy1[3]; yy1+=4;
      j -= 4;
    }
    z[0] = sum0;
    z[1] = sum1;
 
    break;
  case 1:
  yy0   = ((Vec_Seq *)(yy[0]->data))->array;
    switch (j_rem=j&0x3) {
    case 3: 
      x2 = x[2]; sum0 += x2*yy0[2];
    case 2: 
      x1 = x[1]; sum0 += x1*yy0[1];
    case 1: 
      x0 = x[0]; sum0 += x0*yy0[0];
    case 0: 
      x  += j_rem;
      yy0 += j_rem;
      j  -= j_rem;
      break;
    }
    while (j>0) {
      sum0 += x[0]*yy0[0] + x[1]*yy0[1] + x[2]*yy0[2] + x[3]*yy0[3]; yy0+=4;
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
    yy0   = ((Vec_Seq *)(yy[0]->data))->array;
    yy1   = ((Vec_Seq *)(yy[1]->data))->array;
    yy2   = ((Vec_Seq *)(yy[2]->data))->array;
    yy3   = ((Vec_Seq *)(yy[3]->data))->array;
    yy  += 4;

    j = n;
    x = xv->array;
    switch (j_rem=j&0x3) {
    case 3: 
      x2 = x[2]; 
      sum0 += x2*yy0[2]; sum1 += x2*yy1[2]; 
      sum2 += x2*yy2[2]; sum3 += x2*yy3[2];
    case 2: 
      x1 = x[1]; 
      sum0 += x1*yy0[1]; sum1 += x1*yy1[1]; 
      sum2 += x1*yy2[1]; sum3 += x1*yy3[1];
    case 1: 
      x0 = x[0]; 
      sum0 += x0*yy0[0]; sum1 += x0*yy1[0]; 
      sum2 += x0*yy2[0]; sum3 += x0*yy3[0];
    case 0: 
      x  += j_rem;
      yy0 += j_rem;
      yy1 += j_rem;
      yy2 += j_rem;
      yy3 += j_rem;
      j  -= j_rem;
      break;
    }
    while (j>0) {
      x0 = x[0];
      x1 = x[1];
      x2 = x[2];
      x3 = x[3];
      x += 4;
      
      sum0 += x0*yy0[0] + x1*yy0[1] + x2*yy0[2] + x3*yy0[3]; yy0+=4;
      sum1 += x0*yy1[0] + x1*yy1[1] + x2*yy1[2] + x3*yy1[3]; yy1+=4;
      sum2 += x0*yy2[0] + x1*yy2[1] + x2*yy2[2] + x3*yy2[3]; yy2+=4;
      sum3 += x0*yy3[0] + x1*yy3[1] + x2*yy3[2] + x3*yy3[3]; yy3+=4;
      j -= 4;
    }
    z[0] = sum0;
    z[1] = sum1;
    z[2] = sum2;
    z[3] = sum3;
    z   += 4;
    i   -= 4;
  }
  PLogFlops(nv*(2*xv->n-1));
  PetscFunctionReturn(0);
}
    

#undef __FUNC__  
#define __FUNC__ "VecMax_Seq"
int VecMax_Seq(Vec xin,int* idx,double * z )
{ 
  Vec_Seq         *x = (Vec_Seq *) xin->data;
  register int    i, j=0, n = x->n;
  register double max, tmp;
  Scalar          *xx = x->array;

  PetscFunctionBegin;
  if (!n) {
    max = PETSC_MIN;
    j   = -1;
  } else {
#if defined(USE_PETSC_COMPLEX)
      max = PetscReal(*xx++); j = 0;
#else
      max = *xx++; j = 0;
#endif
    for (i=1; i<n; i++) {
#if defined(USE_PETSC_COMPLEX)
      if ((tmp = PetscReal(*xx++)) > max) { j = i; max = tmp;}
#else
      if ((tmp = *xx++) > max) { j = i; max = tmp; } 
#endif
    }
  }
  *z   = max;
  if (idx) *idx = j;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecMin_Seq"
int VecMin_Seq(Vec xin,int* idx,double * z )
{
  Vec_Seq         *x = (Vec_Seq *) xin->data;
  register int    i, j=0, n = x->n;
  register double min, tmp;
  Scalar          *xx = x->array;

  PetscFunctionBegin;
  if (!n) {
    min = PETSC_MAX;
    j   = -1;
  } else {
#if defined(USE_PETSC_COMPLEX)
    min = PetscReal(*xx++); j = 0;
#else
    min = *xx++; j = 0;
#endif
    for ( i=1; i<n; i++ ) {
#if defined(USE_PETSC_COMPLEX)
      if ((tmp = PetscReal(*xx++)) < min) { j = i; min = tmp;}
#else
      if ((tmp = *xx++) < min) { j = i; min = tmp; } 
#endif
    }
  }
  *z   = min;
  if (idx) *idx = j;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecSet_Seq"
int VecSet_Seq(const Scalar* alpha,Vec xin)
{
  Vec_Seq      *x = (Vec_Seq *)xin->data;
  register int n = x->n;
  Scalar       *xx = x->array, oalpha = *alpha;

  PetscFunctionBegin;
  if (oalpha == 0.0) {
    PetscMemzero(xx,n*sizeof(Scalar));
  }
  else {
    SET(xx,n,oalpha);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecSetRandom_Seq"
int VecSetRandom_Seq(PetscRandom r,Vec xin)
{
  Vec_Seq      *x = (Vec_Seq *)xin->data;
  register int n = x->n;
  int          i, ierr;
  Scalar       *xx = x->array;

  PetscFunctionBegin;
  for (i=0; i<n; i++) {ierr = PetscRandomGetValue(r,&xx[i]); CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/* int VecMAXPY_Seq( int nv, Scalar *alpha, Vec xin, Vec *y )
{
  Vec_Seq      *x = (Vec_Seq *) xin->data;
  register int n = x->n;
  int          j;
  Scalar       *xx = x->array, *yy,oalpha;

  PetscFunctionBegin;
  PLogFlops(nv*2*n);
  for (j=0; j<nv; j++) {
    yy     = ((Vec_Seq *)(y[j]->data))->array;
    oalpha = alpha[j];
    if (oalpha == -1.0) {
      YMX(xx,yy,n);
    } else if (oalpha == 1.0) {
      YPX(xx,yy,n);
    } else if (oalpha != 0.0) {
      APXY(xx,oalpha,yy,n);
    }
  }
  PetscFunctionReturn(0);
} */


#undef __FUNC__  
#define __FUNC__ "VecMAXPY_Seq"
int VecMAXPY_Seq(int nv,const Scalar *alpha, Vec xin, Vec *y)
{
  Vec_Seq      *xdata = (Vec_Seq *) xin->data;
  int          n = xdata->n;
  int          j,j_rem;
  Scalar       *xx,*yy0,*yy1,*yy2,*yy3,alpha0,alpha1,alpha2,alpha3;

#if defined(HAVE_PRAGMA_DISJOINT)
#pragma disjoint(*xx,*yy0,*yy1,*yy2,*yy3,*alpha)
#endif

  PetscFunctionBegin;
  PLogFlops(nv*2*n);

  xx = xdata->array;  
  switch (j_rem=nv&0x3) {
  case 3: 
    yy0 = ((Vec_Seq *)(y[0]->data))->array;
    yy1 = ((Vec_Seq *)(y[1]->data))->array;
    yy2 = ((Vec_Seq *)(y[2]->data))->array;
    alpha0 = alpha[0]; 
    alpha1 = alpha[1]; 
    alpha2 = alpha[2]; 
    y     += 3;
    alpha += 3;
    APXY3(xx,alpha0,alpha1,alpha2,yy0,yy1,yy2,n);
    break;
  case 2: 
    yy0 = ((Vec_Seq *)(y[0]->data))->array;
    yy1 = ((Vec_Seq *)(y[1]->data))->array;
    alpha0 = alpha[0]; 
    alpha1 = alpha[1]; 
    y     +=2;
    alpha +=2;
    APXY2(xx,alpha0,alpha1,yy0,yy1,n);
    break;
  case 1: 
    yy0 = ((Vec_Seq *)(y[0]->data))->array; y++; 
    alpha0 = *alpha++; APXY(xx,alpha0,yy0,n);
    break;
  }
  for (j=j_rem; j<nv; j+=4 ) {
    yy0     = ((Vec_Seq *)(y[0]->data))->array;
    yy1     = ((Vec_Seq *)(y[1]->data))->array;
    yy2     = ((Vec_Seq *)(y[2]->data))->array;
    yy3     = ((Vec_Seq *)(y[3]->data))->array;
    alpha0 = alpha[0];
    alpha1 = alpha[1];
    alpha2 = alpha[2];
    alpha3 = alpha[3];
    y      += 4;
    alpha  += 4;

    APXY4(xx,alpha0,alpha1,alpha2,alpha3,yy0,yy1,yy2,yy3,n);
  }
  PetscFunctionReturn(0);
} 

#undef __FUNC__  
#define __FUNC__ "VecAYPX_Seq"
int VecAYPX_Seq(const Scalar *alpha, Vec xin, Vec yin )
{
  Vec_Seq      *x = (Vec_Seq *)xin->data, *y = (Vec_Seq *)yin->data;
  register int i,n = x->n;
  Scalar       *xx = x->array, *yy = y->array, oalpha = *alpha;

  PetscFunctionBegin;
  PLogFlops(2*n);
  for ( i=0; i<n; i++ ) {
    yy[i] = xx[i] + oalpha*yy[i];
  }
  PetscFunctionReturn(0);
}

/*
   IBM ESSL contains a routine dzaxpy() that is our WAXPY() but it appears
  to be slower than a regular C loop.  Hence, we do not include it.
  void ?zaxpy(int*,Scalar*,Scalar*,int*,Scalar*,int*,Scalar*,int*);
*/

#undef __FUNC__  
#define __FUNC__ "VecWAXPY_Seq"
int VecWAXPY_Seq(const Scalar* alpha,Vec xin,Vec yin,Vec win )
{
  Vec_Seq      *w = (Vec_Seq *)win->data, *x = (Vec_Seq *)xin->data;
  Vec_Seq      *y = (Vec_Seq *)yin->data;
  register int i, n = x->n;
  Scalar       *xx = x->array, *yy = y->array, *ww = w->array, oalpha = *alpha;

  PetscFunctionBegin;
  if (oalpha == 1.0) {
    PLogFlops(n);
    /* could call BLAS axpy after call to memcopy, but may be slower */
    for (i=0; i<n; i++) ww[i] = yy[i] + xx[i];
  } else if (oalpha == -1.0) {
    PLogFlops(n);
    for (i=0; i<n; i++) ww[i] = yy[i] - xx[i];
  } else if (oalpha == 0.0) {
    PetscMemcpy(ww,yy,n*sizeof(Scalar));
  } else {
    for (i=0; i<n; i++) ww[i] = yy[i] + oalpha * xx[i];
    PLogFlops(2*n);
  }
  PetscFunctionReturn(0);
}

#if defined(USE_FORTRAN_KERNEL_XTIMESY)
#ifdef HAVE_FORTRAN_CAPS
#define fortranxtimesy_ FORTRANXTIMESY
#elif !defined(HAVE_FORTRAN_UNDERSCORE)
#define fortranxtimesy_ fortranxtimesy
#endif
extern void fortranxtimesy_(Scalar *,Scalar *,Scalar *,int *);
#endif

#undef __FUNC__  
#define __FUNC__ "VecPointwiseMult_Seq"
int VecPointwiseMult_Seq( Vec xin, Vec yin, Vec win )
{
  Vec_Seq         *w = (Vec_Seq *)win->data, *x = (Vec_Seq *)xin->data;
  Vec_Seq         *y = (Vec_Seq *)yin->data;
  int             n = x->n, i;
  register Scalar *xx = x->array, *yy = y->array, *ww = w->array;

  PetscFunctionBegin;
  if (ww == xx) {
    for (i=0; i<n; i++) ww[i] *= yy[i];
  } else if (ww == yy) {
    for (i=0; i<n; i++) ww[i] *= xx[i];
  } else {
    /*    double * __restrict www = ww;
          double * __restrict yyy = yy;
          double * __restrict xxx = xx;
          for (i=0; i<n; i++) www[i] = xxx[i] * yyy[i];
    */
#if defined(USE_FORTRAN_KERNEL_XTIMESY)
    fortranxtimesy_(xx,yy,ww,&n);
#else
    for (i=0; i<n; i++) ww[i] = xx[i] * yy[i];
#endif
  }
  PLogFlops(n);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecPointwiseDivide_Seq"
int VecPointwiseDivide_Seq(Vec xin,Vec yin,Vec win )
{
  Vec_Seq      *w = (Vec_Seq *)win->data, *x = (Vec_Seq *)xin->data;
  Vec_Seq      *y = (Vec_Seq *)yin->data;
  register int n = x->n, i;
  Scalar       *xx = x->array, *yy = y->array, *ww = w->array;

  PetscFunctionBegin;
  PLogFlops(n);
  for (i=0; i<n; i++) ww[i] = xx[i] / yy[i];
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecGetArray_Seq"
int VecGetArray_Seq(Vec vin,Scalar *a[])
{
  Vec_Seq *v = (Vec_Seq *)vin->data;

  PetscFunctionBegin;

  /*  {
    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
    if (!rank) PetscStackView(0);
  } */

  if (vin->array_gotten) {
    SETERRQ(1,1,"Array has already been gotten for this vector, you may\n\
    have forgotten a call to VecRestoreArray()");
  }
  vin->array_gotten = PETSC_TRUE;

  *a =  v->array;
  PetscAMSTakeAccess(vin)
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecRestoreArray_Seq"
int VecRestoreArray_Seq(Vec vin,Scalar *a[])
{
  PetscFunctionBegin;

  /* {
    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
    if (!rank) PetscStackView(0);
  } */

  if (!vin->array_gotten) {
    SETERRQ(1,1,"Array has not been gotten for this vector, you may\n\
    have forgotten a call to VecGetArray()");
  }
  vin->array_gotten = PETSC_FALSE;

  PetscAMSGrantAccess(vin) 
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecPlaceArray_Seq"
int VecPlaceArray_Seq(Vec vin,const Scalar *a)
{
  Vec_Seq *v = (Vec_Seq *)vin->data;

  PetscFunctionBegin;
  v->array = (Scalar *)a;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecGetSize_Seq"
int VecGetSize_Seq(Vec vin,int *size)
{
  Vec_Seq *v = (Vec_Seq *)vin->data;

  PetscFunctionBegin;
  *size = v->n;
  PetscFunctionReturn(0);
}


 



