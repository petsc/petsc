
/*
   Defines some vector operation functions that are shared by
   sequential and parallel vectors.
*/
#include <../src/vec/vec/impls/dvecimpl.h>
#include <petsc/private/kernels/petscaxpy.h>

#if defined(PETSC_USE_FORTRAN_KERNEL_MDOT)
#include <../src/vec/vec/impls/seq/ftn-kernels/fmdot.h>
PetscErrorCode VecMDot_Seq(Vec xin,PetscInt nv,const Vec yin[],PetscScalar *z)
{
  PetscInt          i,nv_rem,n = xin->map->n;
  PetscScalar       sum0,sum1,sum2,sum3;
  const PetscScalar *yy0,*yy1,*yy2,*yy3,*x;
  Vec               *yy;

  PetscFunctionBegin;
  sum0 = 0.0;
  sum1 = 0.0;
  sum2 = 0.0;

  i      = nv;
  nv_rem = nv&0x3;
  yy     = (Vec*)yin;
  PetscCall(VecGetArrayRead(xin,&x));

  switch (nv_rem) {
  case 3:
    PetscCall(VecGetArrayRead(yy[0],&yy0));
    PetscCall(VecGetArrayRead(yy[1],&yy1));
    PetscCall(VecGetArrayRead(yy[2],&yy2));
    fortranmdot3_(x,yy0,yy1,yy2,&n,&sum0,&sum1,&sum2);
    PetscCall(VecRestoreArrayRead(yy[0],&yy0));
    PetscCall(VecRestoreArrayRead(yy[1],&yy1));
    PetscCall(VecRestoreArrayRead(yy[2],&yy2));
    z[0] = sum0;
    z[1] = sum1;
    z[2] = sum2;
    break;
  case 2:
    PetscCall(VecGetArrayRead(yy[0],&yy0));
    PetscCall(VecGetArrayRead(yy[1],&yy1));
    fortranmdot2_(x,yy0,yy1,&n,&sum0,&sum1);
    PetscCall(VecRestoreArrayRead(yy[0],&yy0));
    PetscCall(VecRestoreArrayRead(yy[1],&yy1));
    z[0] = sum0;
    z[1] = sum1;
    break;
  case 1:
    PetscCall(VecGetArrayRead(yy[0],&yy0));
    fortranmdot1_(x,yy0,&n,&sum0);
    PetscCall(VecRestoreArrayRead(yy[0],&yy0));
    z[0] = sum0;
    break;
  case 0:
    break;
  }
  z  += nv_rem;
  i  -= nv_rem;
  yy += nv_rem;

  while (i >0) {
    sum0 = 0.;
    sum1 = 0.;
    sum2 = 0.;
    sum3 = 0.;
    PetscCall(VecGetArrayRead(yy[0],&yy0));
    PetscCall(VecGetArrayRead(yy[1],&yy1));
    PetscCall(VecGetArrayRead(yy[2],&yy2));
    PetscCall(VecGetArrayRead(yy[3],&yy3));
    fortranmdot4_(x,yy0,yy1,yy2,yy3,&n,&sum0,&sum1,&sum2,&sum3);
    PetscCall(VecRestoreArrayRead(yy[0],&yy0));
    PetscCall(VecRestoreArrayRead(yy[1],&yy1));
    PetscCall(VecRestoreArrayRead(yy[2],&yy2));
    PetscCall(VecRestoreArrayRead(yy[3],&yy3));
    yy  += 4;
    z[0] = sum0;
    z[1] = sum1;
    z[2] = sum2;
    z[3] = sum3;
    z   += 4;
    i   -= 4;
  }
  PetscCall(VecRestoreArrayRead(xin,&x));
  PetscCall(PetscLogFlops(PetscMax(nv*(2.0*xin->map->n-1),0.0)));
  PetscFunctionReturn(0);
}

#else
PetscErrorCode VecMDot_Seq(Vec xin,PetscInt nv,const Vec yin[],PetscScalar *z)
{
  PetscInt          n = xin->map->n,i,j,nv_rem,j_rem;
  PetscScalar       sum0,sum1,sum2,sum3,x0,x1,x2,x3;
  const PetscScalar *yy0,*yy1,*yy2,*yy3,*x,*xbase;
  Vec               *yy;

  PetscFunctionBegin;
  sum0 = 0.;
  sum1 = 0.;
  sum2 = 0.;

  i      = nv;
  nv_rem = nv&0x3;
  yy     = (Vec*)yin;
  j      = n;
  PetscCall(VecGetArrayRead(xin,&xbase));
  x      = xbase;

  switch (nv_rem) {
  case 3:
    PetscCall(VecGetArrayRead(yy[0],&yy0));
    PetscCall(VecGetArrayRead(yy[1],&yy1));
    PetscCall(VecGetArrayRead(yy[2],&yy2));
    switch (j_rem=j&0x3) {
    case 3:
      x2    = x[2];
      sum0 += x2*PetscConj(yy0[2]); sum1 += x2*PetscConj(yy1[2]);
      sum2 += x2*PetscConj(yy2[2]);
    case 2:
      x1    = x[1];
      sum0 += x1*PetscConj(yy0[1]); sum1 += x1*PetscConj(yy1[1]);
      sum2 += x1*PetscConj(yy2[1]);
    case 1:
      x0    = x[0];
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
      j    -= 4;
    }
    z[0] = sum0;
    z[1] = sum1;
    z[2] = sum2;
    PetscCall(VecRestoreArrayRead(yy[0],&yy0));
    PetscCall(VecRestoreArrayRead(yy[1],&yy1));
    PetscCall(VecRestoreArrayRead(yy[2],&yy2));
    break;
  case 2:
    PetscCall(VecGetArrayRead(yy[0],&yy0));
    PetscCall(VecGetArrayRead(yy[1],&yy1));
    switch (j_rem=j&0x3) {
    case 3:
      x2    = x[2];
      sum0 += x2*PetscConj(yy0[2]); sum1 += x2*PetscConj(yy1[2]);
    case 2:
      x1    = x[1];
      sum0 += x1*PetscConj(yy0[1]); sum1 += x1*PetscConj(yy1[1]);
    case 1:
      x0    = x[0];
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
      j    -= 4;
    }
    z[0] = sum0;
    z[1] = sum1;

    PetscCall(VecRestoreArrayRead(yy[0],&yy0));
    PetscCall(VecRestoreArrayRead(yy[1],&yy1));
    break;
  case 1:
    PetscCall(VecGetArrayRead(yy[0],&yy0));
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
      yy0  +=4;
      j    -= 4; x+=4;
    }
    z[0] = sum0;

    PetscCall(VecRestoreArrayRead(yy[0],&yy0));
    break;
  case 0:
    break;
  }
  z  += nv_rem;
  i  -= nv_rem;
  yy += nv_rem;

  while (i >0) {
    sum0 = 0.;
    sum1 = 0.;
    sum2 = 0.;
    sum3 = 0.;
    PetscCall(VecGetArrayRead(yy[0],&yy0));
    PetscCall(VecGetArrayRead(yy[1],&yy1));
    PetscCall(VecGetArrayRead(yy[2],&yy2));
    PetscCall(VecGetArrayRead(yy[3],&yy3));

    j = n;
    x = xbase;
    switch (j_rem=j&0x3) {
    case 3:
      x2    = x[2];
      sum0 += x2*PetscConj(yy0[2]); sum1 += x2*PetscConj(yy1[2]);
      sum2 += x2*PetscConj(yy2[2]); sum3 += x2*PetscConj(yy3[2]);
    case 2:
      x1    = x[1];
      sum0 += x1*PetscConj(yy0[1]); sum1 += x1*PetscConj(yy1[1]);
      sum2 += x1*PetscConj(yy2[1]); sum3 += x1*PetscConj(yy3[1]);
    case 1:
      x0    = x[0];
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
      j    -= 4;
    }
    z[0] = sum0;
    z[1] = sum1;
    z[2] = sum2;
    z[3] = sum3;
    z   += 4;
    i   -= 4;
    PetscCall(VecRestoreArrayRead(yy[0],&yy0));
    PetscCall(VecRestoreArrayRead(yy[1],&yy1));
    PetscCall(VecRestoreArrayRead(yy[2],&yy2));
    PetscCall(VecRestoreArrayRead(yy[3],&yy3));
    yy  += 4;
  }
  PetscCall(VecRestoreArrayRead(xin,&xbase));
  PetscCall(PetscLogFlops(PetscMax(nv*(2.0*xin->map->n-1),0.0)));
  PetscFunctionReturn(0);
}
#endif

/* ----------------------------------------------------------------------------*/
PetscErrorCode VecMTDot_Seq(Vec xin,PetscInt nv,const Vec yin[],PetscScalar *z)
{
  PetscInt          n = xin->map->n,i,j,nv_rem,j_rem;
  PetscScalar       sum0,sum1,sum2,sum3,x0,x1,x2,x3;
  const PetscScalar *yy0,*yy1,*yy2,*yy3,*x,*xbase;
  Vec               *yy;

  PetscFunctionBegin;
  sum0 = 0.;
  sum1 = 0.;
  sum2 = 0.;

  i      = nv;
  nv_rem = nv&0x3;
  yy     = (Vec*)yin;
  j      = n;
  PetscCall(VecGetArrayRead(xin,&xbase));
  x      = xbase;

  switch (nv_rem) {
  case 3:
    PetscCall(VecGetArrayRead(yy[0],&yy0));
    PetscCall(VecGetArrayRead(yy[1],&yy1));
    PetscCall(VecGetArrayRead(yy[2],&yy2));
    switch (j_rem=j&0x3) {
    case 3:
      x2    = x[2];
      sum0 += x2*yy0[2]; sum1 += x2*yy1[2];
      sum2 += x2*yy2[2];
    case 2:
      x1    = x[1];
      sum0 += x1*yy0[1]; sum1 += x1*yy1[1];
      sum2 += x1*yy2[1];
    case 1:
      x0    = x[0];
      sum0 += x0*yy0[0]; sum1 += x0*yy1[0];
      sum2 += x0*yy2[0];
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

      sum0 += x0*yy0[0] + x1*yy0[1] + x2*yy0[2] + x3*yy0[3]; yy0+=4;
      sum1 += x0*yy1[0] + x1*yy1[1] + x2*yy1[2] + x3*yy1[3]; yy1+=4;
      sum2 += x0*yy2[0] + x1*yy2[1] + x2*yy2[2] + x3*yy2[3]; yy2+=4;
      j    -= 4;
    }
    z[0] = sum0;
    z[1] = sum1;
    z[2] = sum2;
    PetscCall(VecRestoreArrayRead(yy[0],&yy0));
    PetscCall(VecRestoreArrayRead(yy[1],&yy1));
    PetscCall(VecRestoreArrayRead(yy[2],&yy2));
    break;
  case 2:
    PetscCall(VecGetArrayRead(yy[0],&yy0));
    PetscCall(VecGetArrayRead(yy[1],&yy1));
    switch (j_rem=j&0x3) {
    case 3:
      x2    = x[2];
      sum0 += x2*yy0[2]; sum1 += x2*yy1[2];
    case 2:
      x1    = x[1];
      sum0 += x1*yy0[1]; sum1 += x1*yy1[1];
    case 1:
      x0    = x[0];
      sum0 += x0*yy0[0]; sum1 += x0*yy1[0];
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

      sum0 += x0*yy0[0] + x1*yy0[1] + x2*yy0[2] + x3*yy0[3]; yy0+=4;
      sum1 += x0*yy1[0] + x1*yy1[1] + x2*yy1[2] + x3*yy1[3]; yy1+=4;
      j    -= 4;
    }
    z[0] = sum0;
    z[1] = sum1;

    PetscCall(VecRestoreArrayRead(yy[0],&yy0));
    PetscCall(VecRestoreArrayRead(yy[1],&yy1));
    break;
  case 1:
    PetscCall(VecGetArrayRead(yy[0],&yy0));
    switch (j_rem=j&0x3) {
    case 3:
      x2 = x[2]; sum0 += x2*yy0[2];
    case 2:
      x1 = x[1]; sum0 += x1*yy0[1];
    case 1:
      x0 = x[0]; sum0 += x0*yy0[0];
    case 0:
      x   += j_rem;
      yy0 += j_rem;
      j   -= j_rem;
      break;
    }
    while (j>0) {
      sum0 += x[0]*yy0[0] + x[1]*yy0[1] + x[2]*yy0[2] + x[3]*yy0[3]; yy0+=4;
      j    -= 4; x+=4;
    }
    z[0] = sum0;

    PetscCall(VecRestoreArrayRead(yy[0],&yy0));
    break;
  case 0:
    break;
  }
  z  += nv_rem;
  i  -= nv_rem;
  yy += nv_rem;

  while (i >0) {
    sum0 = 0.;
    sum1 = 0.;
    sum2 = 0.;
    sum3 = 0.;
    PetscCall(VecGetArrayRead(yy[0],&yy0));
    PetscCall(VecGetArrayRead(yy[1],&yy1));
    PetscCall(VecGetArrayRead(yy[2],&yy2));
    PetscCall(VecGetArrayRead(yy[3],&yy3));
    x    = xbase;

    j = n;
    switch (j_rem=j&0x3) {
    case 3:
      x2    = x[2];
      sum0 += x2*yy0[2]; sum1 += x2*yy1[2];
      sum2 += x2*yy2[2]; sum3 += x2*yy3[2];
    case 2:
      x1    = x[1];
      sum0 += x1*yy0[1]; sum1 += x1*yy1[1];
      sum2 += x1*yy2[1]; sum3 += x1*yy3[1];
    case 1:
      x0    = x[0];
      sum0 += x0*yy0[0]; sum1 += x0*yy1[0];
      sum2 += x0*yy2[0]; sum3 += x0*yy3[0];
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

      sum0 += x0*yy0[0] + x1*yy0[1] + x2*yy0[2] + x3*yy0[3]; yy0+=4;
      sum1 += x0*yy1[0] + x1*yy1[1] + x2*yy1[2] + x3*yy1[3]; yy1+=4;
      sum2 += x0*yy2[0] + x1*yy2[1] + x2*yy2[2] + x3*yy2[3]; yy2+=4;
      sum3 += x0*yy3[0] + x1*yy3[1] + x2*yy3[2] + x3*yy3[3]; yy3+=4;
      j    -= 4;
    }
    z[0] = sum0;
    z[1] = sum1;
    z[2] = sum2;
    z[3] = sum3;
    z   += 4;
    i   -= 4;
    PetscCall(VecRestoreArrayRead(yy[0],&yy0));
    PetscCall(VecRestoreArrayRead(yy[1],&yy1));
    PetscCall(VecRestoreArrayRead(yy[2],&yy2));
    PetscCall(VecRestoreArrayRead(yy[3],&yy3));
    yy  += 4;
  }
  PetscCall(VecRestoreArrayRead(xin,&xbase));
  PetscCall(PetscLogFlops(PetscMax(nv*(2.0*xin->map->n-1),0.0)));
  PetscFunctionReturn(0);
}

PetscErrorCode VecMax_Seq(Vec xin,PetscInt *idx,PetscReal *z)
{
  PetscInt          i,j=0,n = xin->map->n;
  PetscReal         max,tmp;
  const PetscScalar *xx;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(xin,&xx));
  if (!n) {
    max = PETSC_MIN_REAL;
    j   = -1;
  } else {
    max = PetscRealPart(*xx++); j = 0;
    for (i=1; i<n; i++) {
      if ((tmp = PetscRealPart(*xx++)) > max) { j = i; max = tmp;}
    }
  }
  *z = max;
  if (idx) *idx = j;
  PetscCall(VecRestoreArrayRead(xin,&xx));
  PetscFunctionReturn(0);
}

PetscErrorCode VecMin_Seq(Vec xin,PetscInt *idx,PetscReal *z)
{
  PetscInt          i,j=0,n = xin->map->n;
  PetscReal         min,tmp;
  const PetscScalar *xx;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(xin,&xx));
  if (!n) {
    min = PETSC_MAX_REAL;
    j   = -1;
  } else {
    min = PetscRealPart(*xx++); j = 0;
    for (i=1; i<n; i++) {
      if ((tmp = PetscRealPart(*xx++)) < min) { j = i; min = tmp;}
    }
  }
  *z = min;
  if (idx) *idx = j;
  PetscCall(VecRestoreArrayRead(xin,&xx));
  PetscFunctionReturn(0);
}

PetscErrorCode VecSet_Seq(Vec xin,PetscScalar alpha)
{
  PetscInt       i,n = xin->map->n;
  PetscScalar    *xx;

  PetscFunctionBegin;
  PetscCall(VecGetArrayWrite(xin,&xx));
  if (alpha == (PetscScalar)0.0) {
    PetscCall(PetscArrayzero(xx,n));
  } else {
    for (i=0; i<n; i++) xx[i] = alpha;
  }
  PetscCall(VecRestoreArrayWrite(xin,&xx));
  PetscFunctionReturn(0);
}

PetscErrorCode VecMAXPY_Seq(Vec xin, PetscInt nv,const PetscScalar *alpha,Vec *y)
{
  PetscInt          n = xin->map->n,j,j_rem;
  const PetscScalar *yy0,*yy1,*yy2,*yy3;
  PetscScalar       *xx,alpha0,alpha1,alpha2,alpha3;

#if defined(PETSC_HAVE_PRAGMA_DISJOINT)
#pragma disjoint(*xx,*yy0,*yy1,*yy2,*yy3,*alpha)
#endif

  PetscFunctionBegin;
  PetscCall(PetscLogFlops(nv*2.0*n));
  PetscCall(VecGetArray(xin,&xx));
  switch (j_rem=nv&0x3) {
  case 3:
    PetscCall(VecGetArrayRead(y[0],&yy0));
    PetscCall(VecGetArrayRead(y[1],&yy1));
    PetscCall(VecGetArrayRead(y[2],&yy2));
    alpha0 = alpha[0];
    alpha1 = alpha[1];
    alpha2 = alpha[2];
    alpha += 3;
    PetscKernelAXPY3(xx,alpha0,alpha1,alpha2,yy0,yy1,yy2,n);
    PetscCall(VecRestoreArrayRead(y[0],&yy0));
    PetscCall(VecRestoreArrayRead(y[1],&yy1));
    PetscCall(VecRestoreArrayRead(y[2],&yy2));
    y   += 3;
    break;
  case 2:
    PetscCall(VecGetArrayRead(y[0],&yy0));
    PetscCall(VecGetArrayRead(y[1],&yy1));
    alpha0 = alpha[0];
    alpha1 = alpha[1];
    alpha +=2;
    PetscKernelAXPY2(xx,alpha0,alpha1,yy0,yy1,n);
    PetscCall(VecRestoreArrayRead(y[0],&yy0));
    PetscCall(VecRestoreArrayRead(y[1],&yy1));
    y   +=2;
    break;
  case 1:
    PetscCall(VecGetArrayRead(y[0],&yy0));
    alpha0 = *alpha++;
    PetscKernelAXPY(xx,alpha0,yy0,n);
    PetscCall(VecRestoreArrayRead(y[0],&yy0));
    y   +=1;
    break;
  }
  for (j=j_rem; j<nv; j+=4) {
    PetscCall(VecGetArrayRead(y[0],&yy0));
    PetscCall(VecGetArrayRead(y[1],&yy1));
    PetscCall(VecGetArrayRead(y[2],&yy2));
    PetscCall(VecGetArrayRead(y[3],&yy3));
    alpha0 = alpha[0];
    alpha1 = alpha[1];
    alpha2 = alpha[2];
    alpha3 = alpha[3];
    alpha += 4;

    PetscKernelAXPY4(xx,alpha0,alpha1,alpha2,alpha3,yy0,yy1,yy2,yy3,n);
    PetscCall(VecRestoreArrayRead(y[0],&yy0));
    PetscCall(VecRestoreArrayRead(y[1],&yy1));
    PetscCall(VecRestoreArrayRead(y[2],&yy2));
    PetscCall(VecRestoreArrayRead(y[3],&yy3));
    y   += 4;
  }
  PetscCall(VecRestoreArray(xin,&xx));
  PetscFunctionReturn(0);
}

#include <../src/vec/vec/impls/seq/ftn-kernels/faypx.h>

PetscErrorCode VecAYPX_Seq(Vec yin,PetscScalar alpha,Vec xin)
{
  PetscInt          n = yin->map->n;
  PetscScalar       *yy;
  const PetscScalar *xx;

  PetscFunctionBegin;
  if (alpha == (PetscScalar)0.0) {
    PetscCall(VecCopy(xin,yin));
  } else if (alpha == (PetscScalar)1.0) {
    PetscCall(VecAXPY_Seq(yin,alpha,xin));
  } else if (alpha == (PetscScalar)-1.0) {
    PetscInt i;
    PetscCall(VecGetArrayRead(xin,&xx));
    PetscCall(VecGetArray(yin,&yy));

    for (i=0; i<n; i++) yy[i] = xx[i] - yy[i];

    PetscCall(VecRestoreArrayRead(xin,&xx));
    PetscCall(VecRestoreArray(yin,&yy));
    PetscCall(PetscLogFlops(1.0*n));
  } else {
    PetscCall(VecGetArrayRead(xin,&xx));
    PetscCall(VecGetArray(yin,&yy));
#if defined(PETSC_USE_FORTRAN_KERNEL_AYPX)
    {
      PetscScalar oalpha = alpha;
      fortranaypx_(&n,&oalpha,xx,yy);
    }
#else
    {
      PetscInt i;

      for (i=0; i<n; i++) yy[i] = xx[i] + alpha*yy[i];
    }
#endif
    PetscCall(VecRestoreArrayRead(xin,&xx));
    PetscCall(VecRestoreArray(yin,&yy));
    PetscCall(PetscLogFlops(2.0*n));
  }
  PetscFunctionReturn(0);
}

#include <../src/vec/vec/impls/seq/ftn-kernels/fwaxpy.h>
/*
   IBM ESSL contains a routine dzaxpy() that is our WAXPY() but it appears
   to be slower than a regular C loop.  Hence,we do not include it.
   void ?zaxpy(int*,PetscScalar*,PetscScalar*,int*,PetscScalar*,int*,PetscScalar*,int*);
*/

PetscErrorCode VecWAXPY_Seq(Vec win, PetscScalar alpha,Vec xin,Vec yin)
{
  PetscInt           i,n = win->map->n;
  PetscScalar        *ww;
  const PetscScalar  *yy,*xx;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(xin,&xx));
  PetscCall(VecGetArrayRead(yin,&yy));
  PetscCall(VecGetArray(win,&ww));
  if (alpha == (PetscScalar)1.0) {
    PetscCall(PetscLogFlops(n));
    /* could call BLAS axpy after call to memcopy, but may be slower */
    for (i=0; i<n; i++) ww[i] = yy[i] + xx[i];
  } else if (alpha == (PetscScalar)-1.0) {
    PetscCall(PetscLogFlops(n));
    for (i=0; i<n; i++) ww[i] = yy[i] - xx[i];
  } else if (alpha == (PetscScalar)0.0) {
    PetscCall(PetscArraycpy(ww,yy,n));
  } else {
    PetscScalar oalpha = alpha;
#if defined(PETSC_USE_FORTRAN_KERNEL_WAXPY)
    fortranwaxpy_(&n,&oalpha,xx,yy,ww);
#else
    for (i=0; i<n; i++) ww[i] = yy[i] + oalpha * xx[i];
#endif
    PetscCall(PetscLogFlops(2.0*n));
  }
  PetscCall(VecRestoreArrayRead(xin,&xx));
  PetscCall(VecRestoreArrayRead(yin,&yy));
  PetscCall(VecRestoreArray(win,&ww));
  PetscFunctionReturn(0);
}

PetscErrorCode VecMaxPointwiseDivide_Seq(Vec xin,Vec yin,PetscReal *max)
{
  PetscInt          n = xin->map->n,i;
  const PetscScalar *xx,*yy;
  PetscReal         m = 0.0;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(xin,&xx));
  PetscCall(VecGetArrayRead(yin,&yy));
  for (i = 0; i < n; i++) {
    if (yy[i] != (PetscScalar)0.0) {
      m = PetscMax(PetscAbsScalar(xx[i]/yy[i]), m);
    } else {
      m = PetscMax(PetscAbsScalar(xx[i]), m);
    }
  }
  PetscCall(VecRestoreArrayRead(xin,&xx));
  PetscCall(VecRestoreArrayRead(yin,&yy));
  PetscCall(MPIU_Allreduce(&m,max,1,MPIU_REAL,MPIU_MAX,PetscObjectComm((PetscObject)xin)));
  PetscCall(PetscLogFlops(n));
  PetscFunctionReturn(0);
}

PetscErrorCode VecPlaceArray_Seq(Vec vin,const PetscScalar *a)
{
  Vec_Seq *v = (Vec_Seq*)vin->data;

  PetscFunctionBegin;
  PetscCheck(!v->unplacedarray,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"VecPlaceArray() was already called on this vector, without a call to VecResetArray()");
  v->unplacedarray = v->array;  /* save previous array so reset can bring it back */
  v->array         = (PetscScalar*)a;
  PetscFunctionReturn(0);
}

PetscErrorCode VecReplaceArray_Seq(Vec vin,const PetscScalar *a)
{
  Vec_Seq        *v = (Vec_Seq*)vin->data;

  PetscFunctionBegin;
  PetscCall(PetscFree(v->array_allocated));
  v->array_allocated = v->array = (PetscScalar*)a;
  PetscFunctionReturn(0);
}
