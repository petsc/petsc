
/*$Id: dvec2.c,v 1.91 2001/09/11 18:14:31 bsmith Exp $*/

/* 
   Defines some vector operation functions that are shared by 
  sequential and parallel vectors.
*/
#include "src/vec/impls/dvecimpl.h"   
#include "src/inline/dot.h"
#include "src/inline/setval.h"
#include "src/inline/axpy.h"

#if defined(PETSC_USE_FORTRAN_KERNEL_MDOT)
#undef __FUNCT__  
#define __FUNCT__ "VecMDot_Seq"
int VecMDot_Seq(int nv,Vec xin,const Vec yin[],PetscScalar *z)
{
  Vec_Seq     *xv = (Vec_Seq *)xin->data;
  int         i,nv_rem,n = xin->n,ierr;
  PetscScalar sum0,sum1,sum2,sum3,*yy0,*yy1,*yy2,*yy3,*x;
  Vec         *yy;

  PetscFunctionBegin;
  sum0 = 0;
  sum1 = 0;
  sum2 = 0;

  i      = nv;
  nv_rem = nv&0x3;
  yy     = (Vec*)yin;
  x      = xv->array;

  switch (nv_rem) {
  case 3:
    ierr = VecGetArrayFast(yy[0],&yy0);CHKERRQ(ierr);
    ierr = VecGetArrayFast(yy[1],&yy1);CHKERRQ(ierr);
    ierr = VecGetArrayFast(yy[2],&yy2);CHKERRQ(ierr);
    fortranmdot3_(x,yy0,yy1,yy2,&n,&sum0,&sum1,&sum2);
    ierr = VecRestoreArrayFast(yy[0],&yy0);CHKERRQ(ierr);
    ierr = VecRestoreArrayFast(yy[1],&yy1);CHKERRQ(ierr);
    ierr = VecRestoreArrayFast(yy[2],&yy2);CHKERRQ(ierr);
    z[0] = sum0;
    z[1] = sum1;
    z[2] = sum2;
    break;
  case 2:
    ierr = VecGetArrayFast(yy[0],&yy0);CHKERRQ(ierr);
    ierr = VecGetArrayFast(yy[1],&yy1);CHKERRQ(ierr);
    fortranmdot2_(x,yy0,yy1,&n,&sum0,&sum1);
    ierr = VecRestoreArrayFast(yy[0],&yy0);CHKERRQ(ierr);
    ierr = VecRestoreArrayFast(yy[1],&yy1);CHKERRQ(ierr);
    z[0] = sum0;
    z[1] = sum1;
    break;
  case 1:
    ierr = VecGetArrayFast(yy[0],&yy0);CHKERRQ(ierr);
    fortranmdot1_(x,yy0,&n,&sum0);
    ierr = VecRestoreArrayFast(yy[0],&yy0);CHKERRQ(ierr);
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
    ierr = VecGetArrayFast(yy[0],&yy0);CHKERRQ(ierr);
    ierr = VecGetArrayFast(yy[1],&yy1);CHKERRQ(ierr);
    ierr = VecGetArrayFast(yy[2],&yy2);CHKERRQ(ierr);
    ierr = VecGetArrayFast(yy[3],&yy3);CHKERRQ(ierr);
    fortranmdot4_(x,yy0,yy1,yy2,yy3,&n,&sum0,&sum1,&sum2,&sum3);
    ierr = VecRestoreArrayFast(yy[0],&yy0);CHKERRQ(ierr);
    ierr = VecRestoreArrayFast(yy[1],&yy1);CHKERRQ(ierr);
    ierr = VecRestoreArrayFast(yy[2],&yy2);CHKERRQ(ierr);
    ierr = VecRestoreArrayFast(yy[3],&yy3);CHKERRQ(ierr);
    yy  += 4;
    z[0] = sum0;
    z[1] = sum1;
    z[2] = sum2;
    z[3] = sum3;
    z   += 4;
    i   -= 4;
  }
  PetscLogFlops(nv*(2*xin->n-1));
  PetscFunctionReturn(0);
}

#else
#undef __FUNCT__  
#define __FUNCT__ "VecMDot_Seq"
int VecMDot_Seq(int nv,Vec xin,const Vec yin[],PetscScalar * restrict z)
{
  Vec_Seq     *xv = (Vec_Seq *)xin->data;
  int          n = xin->n,i,j,nv_rem,j_rem,ierr;
  PetscScalar  sum0,sum1,sum2,sum3,x0,x1,x2,x3,* restrict x;
  PetscScalar  * restrict yy0,* restrict yy1,* restrict yy2,*restrict yy3; 
  Vec          *yy;

  PetscFunctionBegin;
  sum0 = 0;
  sum1 = 0;
  sum2 = 0;

  i      = nv;
  nv_rem = nv&0x3;
  yy     = (Vec *)yin;
  j      = n;
  x      = xv->array;

  switch (nv_rem) {
  case 3:
    ierr = VecGetArrayFast(yy[0],&yy0);CHKERRQ(ierr);
    ierr = VecGetArrayFast(yy[1],&yy1);CHKERRQ(ierr);
    ierr = VecGetArrayFast(yy[2],&yy2);CHKERRQ(ierr);
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
    ierr = VecRestoreArrayFast(yy[0],&yy0);CHKERRQ(ierr);
    ierr = VecRestoreArrayFast(yy[1],&yy1);CHKERRQ(ierr);
    ierr = VecRestoreArrayFast(yy[2],&yy2);CHKERRQ(ierr);
    break;
  case 2:
    ierr = VecGetArrayFast(yy[0],&yy0);CHKERRQ(ierr);
    ierr = VecGetArrayFast(yy[1],&yy1);CHKERRQ(ierr);
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
 
    ierr = VecRestoreArrayFast(yy[0],&yy0);CHKERRQ(ierr);
    ierr = VecRestoreArrayFast(yy[1],&yy1);CHKERRQ(ierr);
    break;
  case 1:
    ierr = VecGetArrayFast(yy[0],&yy0);CHKERRQ(ierr);
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

    ierr = VecRestoreArrayFast(yy[0],&yy0);CHKERRQ(ierr);
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
    ierr = VecGetArrayFast(yy[0],&yy0);CHKERRQ(ierr);
    ierr = VecGetArrayFast(yy[1],&yy1);CHKERRQ(ierr);
    ierr = VecGetArrayFast(yy[2],&yy2);CHKERRQ(ierr);
    ierr = VecGetArrayFast(yy[3],&yy3);CHKERRQ(ierr);

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
    ierr = VecRestoreArrayFast(yy[0],&yy0);CHKERRQ(ierr);
    ierr = VecRestoreArrayFast(yy[1],&yy1);CHKERRQ(ierr);
    ierr = VecRestoreArrayFast(yy[2],&yy2);CHKERRQ(ierr);
    ierr = VecRestoreArrayFast(yy[3],&yy3);CHKERRQ(ierr);
    yy  += 4;
  }
  PetscLogFlops(nv*(2*xin->n-1));
  PetscFunctionReturn(0);
}
#endif

/* ----------------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "VecMTDot_Seq"
int VecMTDot_Seq(int nv,Vec xin,const Vec yin[],PetscScalar *z)
{
  Vec_Seq      *xv = (Vec_Seq *)xin->data;
  int          n = xin->n,i,j,nv_rem,j_rem,ierr;
  PetscScalar  sum0,sum1,sum2,sum3,*yy0,*yy1,*yy2,*yy3,x0,x1,x2,x3,*x;
  Vec          *yy;
  
  PetscFunctionBegin;

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
    ierr = VecGetArrayFast(yy[0],&yy0);CHKERRQ(ierr);
    ierr = VecGetArrayFast(yy[1],&yy1);CHKERRQ(ierr);
    ierr = VecGetArrayFast(yy[2],&yy2);CHKERRQ(ierr);
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
    ierr = VecRestoreArrayFast(yy[0],&yy0);CHKERRQ(ierr);
    ierr = VecRestoreArrayFast(yy[1],&yy1);CHKERRQ(ierr);
    ierr = VecRestoreArrayFast(yy[2],&yy2);CHKERRQ(ierr);
    break;
  case 2:
    ierr = VecGetArrayFast(yy[0],&yy0);CHKERRQ(ierr);
    ierr = VecGetArrayFast(yy[1],&yy1);CHKERRQ(ierr);
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
 
    ierr = VecRestoreArrayFast(yy[0],&yy0);CHKERRQ(ierr);
    ierr = VecRestoreArrayFast(yy[1],&yy1);CHKERRQ(ierr);
    break;
  case 1:
    ierr = VecGetArrayFast(yy[0],&yy0);CHKERRQ(ierr);
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

    ierr = VecRestoreArrayFast(yy[0],&yy0);CHKERRQ(ierr);
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
    ierr = VecGetArrayFast(yy[0],&yy0);CHKERRQ(ierr);
    ierr = VecGetArrayFast(yy[1],&yy1);CHKERRQ(ierr);
    ierr = VecGetArrayFast(yy[2],&yy2);CHKERRQ(ierr);
    ierr = VecGetArrayFast(yy[3],&yy3);CHKERRQ(ierr);

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
    ierr = VecRestoreArrayFast(yy[0],&yy0);CHKERRQ(ierr);
    ierr = VecRestoreArrayFast(yy[1],&yy1);CHKERRQ(ierr);
    ierr = VecRestoreArrayFast(yy[2],&yy2);CHKERRQ(ierr);
    ierr = VecRestoreArrayFast(yy[3],&yy3);CHKERRQ(ierr);
    yy  += 4;
  }
  PetscLogFlops(nv*(2*xin->n-1));
  PetscFunctionReturn(0);
}
    

#undef __FUNCT__  
#define __FUNCT__ "VecMax_Seq"
int VecMax_Seq(Vec xin,int* idx,PetscReal * z)
{ 
  Vec_Seq      *x = (Vec_Seq*)xin->data;
  int          i,j=0,n = xin->n;
  PetscReal    max,tmp;
  PetscScalar  *xx = x->array;

  PetscFunctionBegin;
  if (!n) {
    max = PETSC_MIN;
    j   = -1;
  } else {
#if defined(PETSC_USE_COMPLEX)
      max = PetscRealPart(*xx++); j = 0;
#else
      max = *xx++; j = 0;
#endif
    for (i=1; i<n; i++) {
#if defined(PETSC_USE_COMPLEX)
      if ((tmp = PetscRealPart(*xx++)) > max) { j = i; max = tmp;}
#else
      if ((tmp = *xx++) > max) { j = i; max = tmp; } 
#endif
    }
  }
  *z   = max;
  if (idx) *idx = j;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecMin_Seq"
int VecMin_Seq(Vec xin,int* idx,PetscReal * z)
{
  Vec_Seq      *x = (Vec_Seq*)xin->data;
  int          i,j=0,n = xin->n;
  PetscReal    min,tmp;
  PetscScalar  *xx = x->array;

  PetscFunctionBegin;
  if (!n) {
    min = PETSC_MAX;
    j   = -1;
  } else {
#if defined(PETSC_USE_COMPLEX)
    min = PetscRealPart(*xx++); j = 0;
#else
    min = *xx++; j = 0;
#endif
    for (i=1; i<n; i++) {
#if defined(PETSC_USE_COMPLEX)
      if ((tmp = PetscRealPart(*xx++)) < min) { j = i; min = tmp;}
#else
      if ((tmp = *xx++) < min) { j = i; min = tmp; } 
#endif
    }
  }
  *z   = min;
  if (idx) *idx = j;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecSet_Seq"
int VecSet_Seq(const PetscScalar* alpha,Vec xin)
{
  Vec_Seq      *x = (Vec_Seq *)xin->data;
  int          n = xin->n,ierr;
  PetscScalar  *xx = x->array,oalpha = *alpha;

  PetscFunctionBegin;
  if (oalpha == 0.0) {
    ierr = PetscMemzero(xx,n*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  else {
    SET(xx,n,oalpha);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecSetRandom_Seq"
int VecSetRandom_Seq(PetscRandom r,Vec xin)
{
  int          n = xin->n,i,ierr;
  PetscScalar  *xx;

  PetscFunctionBegin;
  ierr = VecGetArrayFast(xin,&xx);CHKERRQ(ierr);
  for (i=0; i<n; i++) {ierr = PetscRandomGetValue(r,&xx[i]);CHKERRQ(ierr);}
  ierr = VecRestoreArrayFast(xin,&xx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecMAXPY_Seq"
int VecMAXPY_Seq(int nv,const PetscScalar *alpha,Vec xin,Vec *y)
{
  Vec_Seq      *xdata = (Vec_Seq*)xin->data;
  int          n = xin->n,ierr,j,j_rem;
  PetscScalar  *xx,*yy0,*yy1,*yy2,*yy3,alpha0,alpha1,alpha2,alpha3;

#if defined(PETSC_HAVE_PRAGMA_DISJOINT)
#pragma disjoint(*xx,*yy0,*yy1,*yy2,*yy3,*alpha)
#endif

  PetscFunctionBegin;
  PetscLogFlops(nv*2*n);

  xx = xdata->array;  
  switch (j_rem=nv&0x3) {
  case 3: 
    ierr = VecGetArrayFast(y[0],&yy0);CHKERRQ(ierr);
    ierr = VecGetArrayFast(y[1],&yy1);CHKERRQ(ierr);
    ierr = VecGetArrayFast(y[2],&yy2);CHKERRQ(ierr);
    alpha0 = alpha[0]; 
    alpha1 = alpha[1]; 
    alpha2 = alpha[2]; 
    alpha += 3;
    APXY3(xx,alpha0,alpha1,alpha2,yy0,yy1,yy2,n);
    ierr = VecRestoreArrayFast(y[0],&yy0);CHKERRQ(ierr);
    ierr = VecRestoreArrayFast(y[1],&yy1);CHKERRQ(ierr);
    ierr = VecRestoreArrayFast(y[2],&yy2);CHKERRQ(ierr);
    y     += 3;
    break;
  case 2: 
    ierr = VecGetArrayFast(y[0],&yy0);CHKERRQ(ierr);
    ierr = VecGetArrayFast(y[1],&yy1);CHKERRQ(ierr);
    alpha0 = alpha[0]; 
    alpha1 = alpha[1]; 
    alpha +=2;
    APXY2(xx,alpha0,alpha1,yy0,yy1,n);
    ierr = VecRestoreArrayFast(y[0],&yy0);CHKERRQ(ierr);
    ierr = VecRestoreArrayFast(y[1],&yy1);CHKERRQ(ierr);
    y     +=2;
    break;
  case 1: 
    ierr = VecGetArrayFast(y[0],&yy0);CHKERRQ(ierr);
    alpha0 = *alpha++; APXY(xx,alpha0,yy0,n);
    ierr = VecRestoreArrayFast(y[0],&yy0);CHKERRQ(ierr);
    y     +=1;
    break;
  }
  for (j=j_rem; j<nv; j+=4) {
    ierr = VecGetArrayFast(y[0],&yy0);CHKERRQ(ierr);
    ierr = VecGetArrayFast(y[1],&yy1);CHKERRQ(ierr);
    ierr = VecGetArrayFast(y[2],&yy2);CHKERRQ(ierr);
    ierr = VecGetArrayFast(y[3],&yy3);CHKERRQ(ierr);
    alpha0 = alpha[0];
    alpha1 = alpha[1];
    alpha2 = alpha[2];
    alpha3 = alpha[3];
    alpha  += 4;

    APXY4(xx,alpha0,alpha1,alpha2,alpha3,yy0,yy1,yy2,yy3,n);
    ierr = VecRestoreArrayFast(y[0],&yy0);CHKERRQ(ierr);
    ierr = VecRestoreArrayFast(y[1],&yy1);CHKERRQ(ierr);
    ierr = VecRestoreArrayFast(y[2],&yy2);CHKERRQ(ierr);
    ierr = VecRestoreArrayFast(y[3],&yy3);CHKERRQ(ierr);
    y      += 4;
  }
  PetscFunctionReturn(0);
} 

#undef __FUNCT__  
#define __FUNCT__ "VecAYPX_Seq"
int VecAYPX_Seq(const PetscScalar *alpha,Vec xin,Vec yin)
{
  Vec_Seq      *x = (Vec_Seq *)xin->data;
  int          n = xin->n,ierr;
  PetscScalar  *xx = x->array,*yy;

  PetscFunctionBegin;
  ierr = VecGetArrayFast(yin,&yy);CHKERRQ(ierr);
#if defined(PETSC_USE_FORTRAN_KERNEL_AYPX)
  fortranaypx_(&n,alpha,xx,yy);
#else
  {
    int i;
    PetscScalar oalpha = *alpha;
    for (i=0; i<n; i++) {
      yy[i] = xx[i] + oalpha*yy[i];
    }
  }
#endif
  ierr = VecRestoreArrayFast(yin,&yy);CHKERRQ(ierr);
  PetscLogFlops(2*n);
  PetscFunctionReturn(0);
}

/*
   IBM ESSL contains a routine dzaxpy() that is our WAXPY() but it appears
  to be slower than a regular C loop.  Hence,we do not include it.
  void ?zaxpy(int*,PetscScalar*,PetscScalar*,int*,PetscScalar*,int*,PetscScalar*,int*);
*/

#undef __FUNCT__  
#define __FUNCT__ "VecWAXPY_Seq"
int VecWAXPY_Seq(const PetscScalar* alpha,Vec xin,Vec yin,Vec win)
{
  Vec_Seq      *x = (Vec_Seq *)xin->data;
  int          i,n = xin->n,ierr;
  PetscScalar  *xx = x->array,*yy,*ww,oalpha = *alpha;

  PetscFunctionBegin;
  ierr = VecGetArrayFast(yin,&yy);CHKERRQ(ierr);
  ierr = VecGetArrayFast(win,&ww);CHKERRQ(ierr);
  if (oalpha == 1.0) {
    PetscLogFlops(n);
    /* could call BLAS axpy after call to memcopy, but may be slower */
    for (i=0; i<n; i++) ww[i] = yy[i] + xx[i];
  } else if (oalpha == -1.0) {
    PetscLogFlops(n);
    for (i=0; i<n; i++) ww[i] = yy[i] - xx[i];
  } else if (oalpha == 0.0) {
    ierr = PetscMemcpy(ww,yy,n*sizeof(PetscScalar));CHKERRQ(ierr);
  } else {
#if defined(PETSC_USE_FORTRAN_KERNEL_WAXPY)
    fortranwaxpy_(&n,alpha,xx,yy,ww);
#else
    for (i=0; i<n; i++) ww[i] = yy[i] + oalpha * xx[i];
#endif
    PetscLogFlops(2*n);
  }
  ierr = VecRestoreArrayFast(yin,&yy);CHKERRQ(ierr);
  ierr = VecRestoreArrayFast(win,&ww);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecPointwiseMult_Seq"
int VecPointwiseMult_Seq(Vec xin,Vec yin,Vec win)
{
  Vec_Seq      *x = (Vec_Seq *)xin->data;
  int          n = xin->n,i,ierr;
  PetscScalar  *xx = x->array,*yy,*ww;

  PetscFunctionBegin;
  ierr = VecGetArrayFast(yin,&yy);CHKERRQ(ierr);
  if (yin != win) {ierr = VecGetArrayFast(win,&ww);CHKERRQ(ierr);}
  else ww = yy;

  if (ww == xx) {
    for (i=0; i<n; i++) ww[i] *= yy[i];
  } else if (ww == yy) {
    for (i=0; i<n; i++) ww[i] *= xx[i];
  } else {
    /*  This was suppose to help on SGI but didn't really seem to
          PetscReal * __restrict www = ww;
          PetscReal * __restrict yyy = yy;
          PetscReal * __restrict xxx = xx;
          for (i=0; i<n; i++) www[i] = xxx[i] * yyy[i];
    */
#if defined(PETSC_USE_FORTRAN_KERNEL_XTIMESY)
    fortranxtimesy_(xx,yy,ww,&n);
#else
    for (i=0; i<n; i++) ww[i] = xx[i] * yy[i];
#endif
  }
  ierr = VecRestoreArrayFast(yin,&yy);CHKERRQ(ierr);
  if (yin != win) {ierr = VecRestoreArrayFast(win,&ww);CHKERRQ(ierr);}
  PetscLogFlops(n);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecPointwiseDivide_Seq"
int VecPointwiseDivide_Seq(Vec xin,Vec yin,Vec win)
{
  Vec_Seq      *x = (Vec_Seq *)xin->data;
  int          n = xin->n,i,ierr;
  PetscScalar  *xx = x->array,*yy,*ww;

  PetscFunctionBegin;
  ierr = VecGetArrayFast(yin,&yy);CHKERRQ(ierr);
  if (yin != win) {ierr = VecGetArrayFast(win,&ww);CHKERRQ(ierr);}
  else {ww = yy;}
  for (i=0; i<n; i++) ww[i] = xx[i] / yy[i];
  ierr = VecRestoreArrayFast(yin,&yy);CHKERRQ(ierr);
  if (yin != win) {ierr = VecRestoreArrayFast(win,&ww);CHKERRQ(ierr);}
  PetscLogFlops(n);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecMaxPointwiseDivide_Seq"
int VecMaxPointwiseDivide_Seq(Vec xin,Vec yin,PetscReal *max)
{
  Vec_Seq      *x = (Vec_Seq *)xin->data;
  int          n = xin->n,i,ierr;
  PetscScalar  *xx = x->array,*yy;
  PetscReal    m = 0.0;

  PetscFunctionBegin;
  ierr = VecGetArrayFast(yin,&yy);CHKERRQ(ierr);
  for(i = 0; i < n; i++) {
    if (yy[i] != 0.0) {
      m = PetscMax(PetscAbsScalar(xx[i]/yy[i]), m);
    } else {
      m = PetscMax(PetscAbsScalar(xx[i]), m);
    }
  }
  ierr = MPI_Allreduce(&m,max,1,MPIU_REAL,MPI_MAX,xin->comm);CHKERRQ(ierr);
  ierr = VecRestoreArrayFast(yin,&yy);CHKERRQ(ierr);
  PetscLogFlops(n);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecGetArray_Seq"
int VecGetArray_Seq(Vec vin,PetscScalar *a[])
{
  Vec_Seq *v = (Vec_Seq *)vin->data;
  int     ierr;

  PetscFunctionBegin;
  if (vin->array_gotten) {
    SETERRQ(1,"Array has already been gotten for this vector,you may\n\
    have forgotten a call to VecRestoreArray()");
  }
  vin->array_gotten = PETSC_TRUE;

  *a =  v->array;
  ierr = PetscObjectTakeAccess(vin);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecRestoreArray_Seq"
int VecRestoreArray_Seq(Vec vin,PetscScalar *a[])
{
  int ierr;

  PetscFunctionBegin;

  if (!vin->array_gotten) {
    SETERRQ(1,"Array has not been gotten for this vector, you may\n\
    have forgotten a call to VecGetArray()");
  }
  vin->array_gotten = PETSC_FALSE;
  if (a) *a         = 0; /* now user cannot accidently use it again */

  ierr = PetscObjectGrantAccess(vin);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecResetArray_Seq"
int VecResetArray_Seq(Vec vin)
{
  Vec_Seq *v = (Vec_Seq *)vin->data;

  PetscFunctionBegin;
  v->array = v->array_allocated;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecPlaceArray_Seq"
int VecPlaceArray_Seq(Vec vin,const PetscScalar *a)
{
  Vec_Seq *v = (Vec_Seq *)vin->data;

  PetscFunctionBegin;
  v->array = (PetscScalar *)a;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecReplaceArray_Seq"
int VecReplaceArray_Seq(Vec vin,const PetscScalar *a)
{
  Vec_Seq *v = (Vec_Seq *)vin->data;
  int     ierr;

  PetscFunctionBegin;
  if (v->array_allocated) {ierr = PetscFree(v->array_allocated);CHKERRQ(ierr);}
  v->array_allocated = v->array = (PetscScalar *)a;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecGetSize_Seq"
int VecGetSize_Seq(Vec vin,int *size)
{
  PetscFunctionBegin;
  *size = vin->n;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecConjugate_Seq"
int VecConjugate_Seq(Vec xin)
{
  PetscScalar *x = ((Vec_Seq *)xin->data)->array;
  int         n = xin->n;

  PetscFunctionBegin;
  while (n-->0) {
    *x = PetscConj(*x);
    x++;
  }
  PetscFunctionReturn(0);
}
 



