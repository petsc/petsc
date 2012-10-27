
/*
   Defines some vector operation functions that are shared by
  sequential and parallel vectors.
*/
#include <../src/vec/vec/impls/dvecimpl.h>
#include <petsc-private/petscaxpy.h>
#include <petscthreadcomm.h>

#if defined(PETSC_THREADCOMM_ACTIVE)
PetscErrorCode VecMDot_kernel(PetscInt thread_id,Vec xin,PetscInt *nvp,Vec *yvec,PetscThreadCommReduction red)
{
  PetscErrorCode       ierr;
  PetscInt             *trstarts=xin->map->trstarts;
  PetscInt             start,end,nv=*nvp,i,j;
  PetscScalar          *xx,*yy;
  PetscScalar          sum;

  start = trstarts[thread_id];
  end   = trstarts[thread_id+1];
  ierr = VecGetArray(xin,&xx);CHKERRQ(ierr);
  for(i=0;i<nv;i++) {
    sum = 0.;
    ierr = VecGetArray(yvec[i],&yy);CHKERRQ(ierr);
    for(j=start;j<end;j++) sum += xx[j]*PetscConj(yy[j]);
    ierr = PetscThreadReductionKernelPost(thread_id,red,&sum);CHKERRQ(ierr);
    ierr = VecRestoreArray(yvec[i],&yy);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(xin,&xx);CHKERRQ(ierr);

  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "VecMDot_Seq"
PetscErrorCode VecMDot_Seq(Vec xin,PetscInt nv,const Vec yin[],PetscScalar *z)
{
  PetscErrorCode           ierr;
  PetscThreadCommReduction red;
  PetscInt                 *nvp,nreds,i,j;

  PetscFunctionBegin;

  ierr = PetscThreadCommGetInts(((PetscObject)xin)->comm,&nvp,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  for (i=0; i<nv; i+=nreds) {
    nreds = PetscMin(nv-i,PETSC_REDUCTIONS_MAX);
    ierr = PetscThreadReductionBegin(((PetscObject)xin)->comm,THREADCOMM_SUM,PETSC_SCALAR,nreds,&red);CHKERRQ(ierr);
    *nvp = nreds;
    ierr = PetscThreadCommRunKernel4(((PetscObject)xin)->comm,(PetscThreadKernel)VecMDot_kernel,xin,nvp,(Vec*)yin+i,red);CHKERRQ(ierr);
    for(j=0; j<nreds; j++) {
      ierr = PetscThreadReductionEnd(red,&z[i+j]);CHKERRQ(ierr);
    }
  }
  ierr = PetscLogFlops(PetscMax(nv*(2.0*xin->map->n-1),0.0));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#else
#if defined(PETSC_USE_FORTRAN_KERNEL_MDOT)
#include <../src/vec/vec/impls/seq/ftn-kernels/fmdot.h>
#undef __FUNCT__
#define __FUNCT__ "VecMDot_Seq"
PetscErrorCode VecMDot_Seq(Vec xin,PetscInt nv,const Vec yin[],PetscScalar *z)
{
  PetscErrorCode    ierr;
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
  ierr   = VecGetArrayRead(xin,&x);CHKERRQ(ierr);

  switch (nv_rem) {
  case 3:
    ierr = VecGetArrayRead(yy[0],&yy0);CHKERRQ(ierr);
    ierr = VecGetArrayRead(yy[1],&yy1);CHKERRQ(ierr);
    ierr = VecGetArrayRead(yy[2],&yy2);CHKERRQ(ierr);
    fortranmdot3_(x,yy0,yy1,yy2,&n,&sum0,&sum1,&sum2);
    ierr = VecRestoreArrayRead(yy[0],&yy0);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(yy[1],&yy1);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(yy[2],&yy2);CHKERRQ(ierr);
    z[0] = sum0;
    z[1] = sum1;
    z[2] = sum2;
    break;
  case 2:
    ierr = VecGetArrayRead(yy[0],&yy0);CHKERRQ(ierr);
    ierr = VecGetArrayRead(yy[1],&yy1);CHKERRQ(ierr);
    fortranmdot2_(x,yy0,yy1,&n,&sum0,&sum1);
    ierr = VecRestoreArrayRead(yy[0],&yy0);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(yy[1],&yy1);CHKERRQ(ierr);
    z[0] = sum0;
    z[1] = sum1;
    break;
  case 1:
    ierr = VecGetArrayRead(yy[0],&yy0);CHKERRQ(ierr);
    fortranmdot1_(x,yy0,&n,&sum0);
    ierr = VecRestoreArrayRead(yy[0],&yy0);CHKERRQ(ierr);
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
    ierr = VecGetArrayRead(yy[0],&yy0);CHKERRQ(ierr);
    ierr = VecGetArrayRead(yy[1],&yy1);CHKERRQ(ierr);
    ierr = VecGetArrayRead(yy[2],&yy2);CHKERRQ(ierr);
    ierr = VecGetArrayRead(yy[3],&yy3);CHKERRQ(ierr);
    fortranmdot4_(x,yy0,yy1,yy2,yy3,&n,&sum0,&sum1,&sum2,&sum3);
    ierr = VecRestoreArrayRead(yy[0],&yy0);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(yy[1],&yy1);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(yy[2],&yy2);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(yy[3],&yy3);CHKERRQ(ierr);
    yy  += 4;
    z[0] = sum0;
    z[1] = sum1;
    z[2] = sum2;
    z[3] = sum3;
    z   += 4;
    i   -= 4;
  }
  ierr = VecRestoreArrayRead(xin,&x);CHKERRQ(ierr);
  ierr = PetscLogFlops(PetscMax(nv*(2.0*xin->map->n-1),0.0));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#else
#undef __FUNCT__
#define __FUNCT__ "VecMDot_Seq"
PetscErrorCode VecMDot_Seq(Vec xin,PetscInt nv,const Vec yin[],PetscScalar *z)
{
  PetscErrorCode    ierr;
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
  yy     = (Vec *)yin;
  j      = n;
  ierr   = VecGetArrayRead(xin,&xbase);CHKERRQ(ierr);
  x      = xbase;

  switch (nv_rem) {
  case 3:
    ierr = VecGetArrayRead(yy[0],&yy0);CHKERRQ(ierr);
    ierr = VecGetArrayRead(yy[1],&yy1);CHKERRQ(ierr);
    ierr = VecGetArrayRead(yy[2],&yy2);CHKERRQ(ierr);
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
    ierr = VecRestoreArrayRead(yy[0],&yy0);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(yy[1],&yy1);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(yy[2],&yy2);CHKERRQ(ierr);
    break;
  case 2:
    ierr = VecGetArrayRead(yy[0],&yy0);CHKERRQ(ierr);
    ierr = VecGetArrayRead(yy[1],&yy1);CHKERRQ(ierr);
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

    ierr = VecRestoreArrayRead(yy[0],&yy0);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(yy[1],&yy1);CHKERRQ(ierr);
    break;
  case 1:
    ierr = VecGetArrayRead(yy[0],&yy0);CHKERRQ(ierr);
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

    ierr = VecRestoreArrayRead(yy[0],&yy0);CHKERRQ(ierr);
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
    ierr = VecGetArrayRead(yy[0],&yy0);CHKERRQ(ierr);
    ierr = VecGetArrayRead(yy[1],&yy1);CHKERRQ(ierr);
    ierr = VecGetArrayRead(yy[2],&yy2);CHKERRQ(ierr);
    ierr = VecGetArrayRead(yy[3],&yy3);CHKERRQ(ierr);

    j = n;
    x = xbase;
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
    ierr = VecRestoreArrayRead(yy[0],&yy0);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(yy[1],&yy1);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(yy[2],&yy2);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(yy[3],&yy3);CHKERRQ(ierr);
    yy  += 4;
  }
  ierr = VecRestoreArrayRead(xin,&xbase);CHKERRQ(ierr);
  ierr = PetscLogFlops(PetscMax(nv*(2.0*xin->map->n-1),0.0));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif
#endif
/* ----------------------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "VecMTDot_Seq"
PetscErrorCode VecMTDot_Seq(Vec xin,PetscInt nv,const Vec yin[],PetscScalar *z)
{
  PetscErrorCode    ierr;
  PetscInt          n = xin->map->n,i,j,nv_rem,j_rem;
  PetscScalar       sum0,sum1,sum2,sum3,x0,x1,x2,x3;
  const PetscScalar *yy0,*yy1,*yy2,*yy3,*x;
  Vec               *yy;

  PetscFunctionBegin;

  sum0 = 0.;
  sum1 = 0.;
  sum2 = 0.;

  i      = nv;
  nv_rem = nv&0x3;
  yy     = (Vec*)yin;
  j      = n;
  ierr   = VecGetArrayRead(xin,&x);CHKERRQ(ierr);

  switch (nv_rem) {
  case 3:
    ierr = VecGetArrayRead(yy[0],&yy0);CHKERRQ(ierr);
    ierr = VecGetArrayRead(yy[1],&yy1);CHKERRQ(ierr);
    ierr = VecGetArrayRead(yy[2],&yy2);CHKERRQ(ierr);
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
    ierr = VecRestoreArrayRead(yy[0],&yy0);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(yy[1],&yy1);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(yy[2],&yy2);CHKERRQ(ierr);
    break;
  case 2:
    ierr = VecGetArrayRead(yy[0],&yy0);CHKERRQ(ierr);
    ierr = VecGetArrayRead(yy[1],&yy1);CHKERRQ(ierr);
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

    ierr = VecRestoreArrayRead(yy[0],&yy0);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(yy[1],&yy1);CHKERRQ(ierr);
    break;
  case 1:
    ierr = VecGetArrayRead(yy[0],&yy0);CHKERRQ(ierr);
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

    ierr = VecRestoreArrayRead(yy[0],&yy0);CHKERRQ(ierr);
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
    ierr = VecGetArrayRead(yy[0],&yy0);CHKERRQ(ierr);
    ierr = VecGetArrayRead(yy[1],&yy1);CHKERRQ(ierr);
    ierr = VecGetArrayRead(yy[2],&yy2);CHKERRQ(ierr);
    ierr = VecGetArrayRead(yy[3],&yy3);CHKERRQ(ierr);

    j = n;
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
    ierr = VecRestoreArrayRead(yy[0],&yy0);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(yy[1],&yy1);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(yy[2],&yy2);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(yy[3],&yy3);CHKERRQ(ierr);
    yy  += 4;
  }
  ierr = VecRestoreArrayRead(xin,&x);CHKERRQ(ierr);
  ierr = PetscLogFlops(PetscMax(nv*(2.0*xin->map->n-1),0.0));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if defined(PETSC_THREADCOMM_ACTIVE)
PetscErrorCode VecMax_kernel(PetscInt thread_id,Vec xin,PetscThreadCommReduction red)
{
  PetscErrorCode ierr;
  PetscInt       *trstarts=xin->map->trstarts;
  PetscInt       start,end,i;
  const PetscScalar *xx;
  PetscReal      lred[2],tmp;

  start = trstarts[thread_id];
  end   = trstarts[thread_id+1];
  ierr = VecGetArrayRead(xin,&xx);CHKERRQ(ierr);
  lred[0] = xx[start]; lred[1] = start;
  for(i=start+1;i<end;i++) {
#if defined(PETSC_USE_COMPLEX)
    if((tmp = PetscRealPart(xx[i])) > lred[0]) { lred[0] = tmp; lred[1] = i;}
#else
    if((tmp = xx[i]) > lred[0]) {lred[0] = tmp; lred[1] = i;}
#endif
  }
  ierr = VecRestoreArrayRead(xin,&xx);CHKERRQ(ierr);
  ierr = PetscThreadReductionKernelPost(thread_id,red,lred);CHKERRQ(ierr);
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "VecMax_Seq"
PetscErrorCode VecMax_Seq(Vec xin,PetscInt* idx,PetscReal *z)
{
  PetscErrorCode           ierr;
  PetscInt                 n=xin->map->n;
  PetscThreadCommReduction red;
  PetscReal                out[2];

  PetscFunctionBegin;
  if(!n) {
    *z = PETSC_MIN_REAL;
    *idx = -1;
  } else {
    ierr = PetscThreadReductionBegin(((PetscObject)xin)->comm,THREADCOMM_MAXLOC,PETSC_REAL,1,&red);CHKERRQ(ierr);
    ierr = PetscThreadCommRunKernel2(((PetscObject)xin)->comm,(PetscThreadKernel)VecMax_kernel,xin,red);CHKERRQ(ierr);
    ierr = PetscThreadReductionEnd(red,out);CHKERRQ(ierr);
    *z = out[0];
    if(idx) *idx = (PetscInt)out[1];
  }
  PetscFunctionReturn(0);
}
#else
#undef __FUNCT__
#define __FUNCT__ "VecMax_Seq"
PetscErrorCode VecMax_Seq(Vec xin,PetscInt* idx,PetscReal * z)
{
  PetscInt          i,j=0,n = xin->map->n;
  PetscReal         max,tmp;
  const PetscScalar *xx;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(xin,&xx);CHKERRQ(ierr);
  if (!n) {
    max = PETSC_MIN_REAL;
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
  ierr = VecRestoreArrayRead(xin,&xx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

#if defined(PETSC_THREADCOMM_ACTIVE)
PetscErrorCode VecMin_kernel(PetscInt thread_id,Vec xin,PetscThreadCommReduction red)
{
  PetscErrorCode ierr;
  PetscInt       *trstarts=xin->map->trstarts;
  PetscInt       start,end,i;
  const PetscScalar *xx;
  PetscReal      lred[2],tmp;

  start = trstarts[thread_id];
  end   = trstarts[thread_id+1];
  ierr = VecGetArrayRead(xin,&xx);CHKERRQ(ierr);
  lred[0] = xx[start]; lred[1] = start;
  for(i=start+1;i<end;i++) {
#if defined(PETSC_USE_COMPLEX)
    if((tmp = PetscRealPart(xx[i])) < lred[0]) { lred[0] = tmp; lred[1] = i;}
#else
    if((tmp = xx[i]) < lred[0]) {lred[0] = tmp; lred[1] = i;}
#endif
  }
  ierr = VecRestoreArrayRead(xin,&xx);CHKERRQ(ierr);
  ierr = PetscThreadReductionKernelPost(thread_id,red,lred);CHKERRQ(ierr);
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "VecMin_Seq"
PetscErrorCode VecMin_Seq(Vec xin,PetscInt* idx,PetscReal *z)
{
  PetscErrorCode           ierr;
  PetscInt                 n=xin->map->n;
  PetscThreadCommReduction red;
  PetscReal                out[2];

  PetscFunctionBegin;
  if(!n) {
    *z = PETSC_MAX_REAL;
    *idx = -1;
  } else {
    ierr = PetscThreadReductionBegin(((PetscObject)xin)->comm,THREADCOMM_MINLOC,PETSC_REAL,1,&red);CHKERRQ(ierr);
    ierr = PetscThreadCommRunKernel2(((PetscObject)xin)->comm,(PetscThreadKernel)VecMin_kernel,xin,red);CHKERRQ(ierr);
    ierr = PetscThreadReductionEnd(red,out);CHKERRQ(ierr);
    *z = out[0];
    if(idx) *idx = (PetscInt)out[1];
  }
  PetscFunctionReturn(0);
}
#else
#undef __FUNCT__
#define __FUNCT__ "VecMin_Seq"
PetscErrorCode VecMin_Seq(Vec xin,PetscInt* idx,PetscReal * z)
{
  PetscInt          i,j=0,n = xin->map->n;
  PetscReal         min,tmp;
  const PetscScalar *xx;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(xin,&xx);CHKERRQ(ierr);
  if (!n) {
    min = PETSC_MAX_REAL;
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
  ierr = VecGetArrayRead(xin,&xx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

#if defined(PETSC_THREADCOMM_ACTIVE)
PetscErrorCode VecSet_kernel(PetscInt thread_id,Vec xin,PetscScalar *alpha_p)
{
  PetscErrorCode ierr;
  PetscScalar    *xx;
  PetscInt       i,start,end;
  PetscInt       *trstarts=xin->map->trstarts;
  PetscScalar    alpha = *alpha_p;

  start = trstarts[thread_id];
  end   = trstarts[thread_id+1];
  ierr = VecGetArray(xin,&xx);CHKERRQ(ierr);
  if (alpha == (PetscScalar)0.0) {
    ierr = PetscMemzero(xx+start,(end-start)*sizeof(PetscScalar));CHKERRQ(ierr);
  } else {
    for (i=start;i<end;i++) xx[i] = alpha;
  }
  ierr = VecRestoreArray(xin,&xx);CHKERRQ(ierr);
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "VecSet_Seq"
PetscErrorCode VecSet_Seq(Vec xin,PetscScalar alpha)
{
  PetscErrorCode ierr;
  PetscScalar    *scalar;

  PetscFunctionBegin;
  ierr = PetscThreadCommGetScalars(((PetscObject)xin)->comm,&scalar,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  *scalar = alpha;
  ierr = PetscThreadCommRunKernel2(((PetscObject)xin)->comm,(PetscThreadKernel)VecSet_kernel,xin,scalar);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#else
#undef __FUNCT__
#define __FUNCT__ "VecSet_Seq"
PetscErrorCode VecSet_Seq(Vec xin,PetscScalar alpha)
{
  PetscInt       i,n = xin->map->n;
  PetscScalar    *xx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecGetArray(xin,&xx);CHKERRQ(ierr);
  if (alpha == (PetscScalar)0.0) {
    ierr = PetscMemzero(xx,n*sizeof(PetscScalar));CHKERRQ(ierr);
  } else {
    for (i=0; i<n; i++) xx[i] = alpha;
  }
  ierr = VecRestoreArray(xin,&xx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

#if defined(PETSC_THREADCOMM_ACTIVE)
PetscErrorCode VecMAXPY_kernel(PetscInt thread_id,Vec xin,PetscInt *nv_p,const PetscScalar *alpha,Vec *y)
{
  PetscErrorCode    ierr;
  PetscInt          *trstarts=xin->map->trstarts,j,j_rem,nv=*nv_p;
  PetscInt          start,end,n;
  const PetscScalar *yy0,*yy1,*yy2,*yy3;
  PetscScalar       *xx,*x,alpha0,alpha1,alpha2,alpha3;
  const PetscScalar *y0,*y1,*y2,*y3;


  start = trstarts[thread_id];
  end   = trstarts[thread_id+1];
  n     = end-start;
#if defined(PETSC_HAVE_PRAGMA_DISJOINT)
#pragma disjoint(*xx,*yy0,*yy1,*yy2,*yy3,*alpha)
#endif

  ierr = VecGetArray(xin,&xx);CHKERRQ(ierr);
  x = xx+start;
  switch (j_rem=nv&0x3) {
  case 3: 
    ierr = VecGetArrayRead(y[0],&yy0);CHKERRQ(ierr);
    ierr = VecGetArrayRead(y[1],&yy1);CHKERRQ(ierr);
    ierr = VecGetArrayRead(y[2],&yy2);CHKERRQ(ierr);
    alpha0 = alpha[0]; 
    alpha1 = alpha[1]; 
    alpha2 = alpha[2]; 
    alpha += 3;
    y0 = yy0 + start; y1 = yy1+start; y2 = yy2+start;
    PetscAXPY3(x,alpha0,alpha1,alpha2,y0,y1,y2,n);
    ierr = VecRestoreArrayRead(y[0],&yy0);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(y[1],&yy1);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(y[2],&yy2);CHKERRQ(ierr);
    y     += 3;
    break;
  case 2: 
    ierr = VecGetArrayRead(y[0],&yy0);CHKERRQ(ierr);
    ierr = VecGetArrayRead(y[1],&yy1);CHKERRQ(ierr);
    alpha0 = alpha[0]; 
    alpha1 = alpha[1]; 
    alpha +=2;
    y0 = yy0+start; y1=yy1+start;
    PetscAXPY2(x,alpha0,alpha1,y0,y1,n);
    ierr = VecRestoreArrayRead(y[0],&yy0);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(y[1],&yy1);CHKERRQ(ierr);
    y     +=2;
    break;
  case 1: 
    ierr = VecGetArrayRead(y[0],&yy0);CHKERRQ(ierr);
    alpha0 = *alpha++; 
    y0 = yy0 + start;
    PetscAXPY(x,alpha0,y0,n);
    ierr = VecRestoreArrayRead(y[0],&yy0);CHKERRQ(ierr);
    y     +=1;
    break;
  }
  for (j=j_rem; j<nv; j+=4) {
    ierr = VecGetArrayRead(y[0],&yy0);CHKERRQ(ierr);
    ierr = VecGetArrayRead(y[1],&yy1);CHKERRQ(ierr);
    ierr = VecGetArrayRead(y[2],&yy2);CHKERRQ(ierr);
    ierr = VecGetArrayRead(y[3],&yy3);CHKERRQ(ierr);
    alpha0 = alpha[0];
    alpha1 = alpha[1];
    alpha2 = alpha[2];
    alpha3 = alpha[3];
    alpha  += 4;
    y0 = yy0+start; y1=yy1+start; y2=yy2+start; y3=yy3+start;
    PetscAXPY4(x,alpha0,alpha1,alpha2,alpha3,y0,y1,y2,y3,n);
    ierr = VecRestoreArrayRead(y[0],&yy0);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(y[1],&yy1);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(y[2],&yy2);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(y[3],&yy3);CHKERRQ(ierr);
    y      += 4;
  }
  ierr = VecRestoreArray(xin,&xx);CHKERRQ(ierr);
  return 0;
} 

#undef __FUNCT__
#define __FUNCT__ "VecMAXPY_Seq"
PetscErrorCode VecMAXPY_Seq(Vec xin,PetscInt nv,const PetscScalar *alpha,Vec *y)
{
  PetscErrorCode ierr;
  PetscInt       *int1;
  
  PetscFunctionBegin;
  ierr = PetscLogFlops(nv*2.0*xin->map->n);
  ierr = PetscThreadCommGetInts(((PetscObject)xin)->comm,&int1,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  *int1 = nv;
  ierr = PetscThreadCommRunKernel4(((PetscObject)xin)->comm,(PetscThreadKernel)VecMAXPY_kernel,xin,int1,(void*)alpha,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#else
#undef __FUNCT__
#define __FUNCT__ "VecMAXPY_Seq"
PetscErrorCode VecMAXPY_Seq(Vec xin, PetscInt nv,const PetscScalar *alpha,Vec *y)
{
  PetscErrorCode    ierr;
  PetscInt          n = xin->map->n,j,j_rem;
  const PetscScalar *yy0,*yy1,*yy2,*yy3;
  PetscScalar       *xx,alpha0,alpha1,alpha2,alpha3;

#if defined(PETSC_HAVE_PRAGMA_DISJOINT)
#pragma disjoint(*xx,*yy0,*yy1,*yy2,*yy3,*alpha)
#endif

  PetscFunctionBegin;
  ierr = PetscLogFlops(nv*2.0*n);CHKERRQ(ierr);
  ierr = VecGetArray(xin,&xx);CHKERRQ(ierr);
  switch (j_rem=nv&0x3) {
  case 3:
    ierr = VecGetArrayRead(y[0],&yy0);CHKERRQ(ierr);
    ierr = VecGetArrayRead(y[1],&yy1);CHKERRQ(ierr);
    ierr = VecGetArrayRead(y[2],&yy2);CHKERRQ(ierr);
    alpha0 = alpha[0];
    alpha1 = alpha[1];
    alpha2 = alpha[2];
    alpha += 3;
    PetscAXPY3(xx,alpha0,alpha1,alpha2,yy0,yy1,yy2,n);
    ierr = VecRestoreArrayRead(y[0],&yy0);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(y[1],&yy1);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(y[2],&yy2);CHKERRQ(ierr);
    y     += 3;
    break;
  case 2:
    ierr = VecGetArrayRead(y[0],&yy0);CHKERRQ(ierr);
    ierr = VecGetArrayRead(y[1],&yy1);CHKERRQ(ierr);
    alpha0 = alpha[0];
    alpha1 = alpha[1];
    alpha +=2;
    PetscAXPY2(xx,alpha0,alpha1,yy0,yy1,n);
    ierr = VecRestoreArrayRead(y[0],&yy0);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(y[1],&yy1);CHKERRQ(ierr);
    y     +=2;
    break;
  case 1:
    ierr = VecGetArrayRead(y[0],&yy0);CHKERRQ(ierr);
    alpha0 = *alpha++;
    PetscAXPY(xx,alpha0,yy0,n);
    ierr = VecRestoreArrayRead(y[0],&yy0);CHKERRQ(ierr);
    y     +=1;
    break;
  }
  for (j=j_rem; j<nv; j+=4) {
    ierr = VecGetArrayRead(y[0],&yy0);CHKERRQ(ierr);
    ierr = VecGetArrayRead(y[1],&yy1);CHKERRQ(ierr);
    ierr = VecGetArrayRead(y[2],&yy2);CHKERRQ(ierr);
    ierr = VecGetArrayRead(y[3],&yy3);CHKERRQ(ierr);
    alpha0 = alpha[0];
    alpha1 = alpha[1];
    alpha2 = alpha[2];
    alpha3 = alpha[3];
    alpha  += 4;

    PetscAXPY4(xx,alpha0,alpha1,alpha2,alpha3,yy0,yy1,yy2,yy3,n);
    ierr = VecRestoreArrayRead(y[0],&yy0);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(y[1],&yy1);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(y[2],&yy2);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(y[3],&yy3);CHKERRQ(ierr);
    y      += 4;
  }
  ierr = VecRestoreArray(xin,&xx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

#include <../src/vec/vec/impls/seq/ftn-kernels/faypx.h>
#if defined(PETSC_THREADCOMM_ACTIVE)
PetscErrorCode VecAYPX_kernel(PetscInt thread_id,Vec yin,PetscScalar *alpha_p,Vec xin)
{
  PetscErrorCode    ierr;
  PetscScalar       *yy;
  const PetscScalar *xx;
  PetscInt          *trstarts=yin->map->trstarts,i;
  PetscScalar       alpha = *alpha_p;

  ierr = VecGetArrayRead(xin,&xx);CHKERRQ(ierr);
  ierr = VecGetArray(yin,&yy);CHKERRQ(ierr);

  if (alpha == (PetscScalar)-1.0) {
    for(i=trstarts[thread_id]; i < trstarts[thread_id+1]; i++) {
      yy[i] = xx[i] - yy[i];
    }
  } else
#if defined(PETSC_USE_FORTRAN_KERNEL_AYPX)
  {
      PetscInt start=trstarts[thread_id],end=trstarts[thread_id+1],n;
      PetscScalar oalpha = alpha;
      n = end - start;
      fortranaypx_(&n,&oalpha,xx+start,yy+start);
  }
#else
  {
    for(i=trstarts[thread_id]; i < trstarts[thread_id+1];i++) {
      yy[i] = xx[i] + alpha*yy[i];
    }
  }
#endif
  ierr = VecRestoreArrayRead(xin,&xx);CHKERRQ(ierr);
  ierr = VecRestoreArray(yin,&yy);CHKERRQ(ierr);
  return 0;
}
#undef __FUNCT__  
#define __FUNCT__ "VecAYPX_Seq"
PetscErrorCode VecAYPX_Seq(Vec yin,PetscScalar alpha,Vec xin)
{
  PetscErrorCode    ierr;
  PetscInt          n=yin->map->n;

  PetscFunctionBegin;
  if (alpha == (PetscScalar)0.0) {
    ierr = VecCopy(xin,yin);CHKERRQ(ierr);
  } else if (alpha == (PetscScalar)1.0) {
    ierr = VecAXPY_Seq(yin,alpha,xin);CHKERRQ(ierr);
  } else {
    PetscScalar *scal1;
    ierr = PetscThreadCommGetScalars(((PetscObject)yin)->comm,&scal1,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    *scal1 = alpha;
    ierr = PetscThreadCommRunKernel3(((PetscObject)yin)->comm,(PetscThreadKernel)VecAYPX_kernel,yin,scal1,xin);CHKERRQ(ierr);
    if (alpha == (PetscScalar)-1.0) {
      ierr = PetscLogFlops(1.0*n);CHKERRQ(ierr);
    } else {
      ierr = PetscLogFlops(2.0*n);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}
#else
#undef __FUNCT__
#define __FUNCT__ "VecAYPX_Seq"
PetscErrorCode VecAYPX_Seq(Vec yin,PetscScalar alpha,Vec xin)
{
  PetscErrorCode    ierr;
  PetscInt          n = yin->map->n;
  PetscScalar       *yy;
  const PetscScalar *xx;

  PetscFunctionBegin;
  if (alpha == (PetscScalar)0.0) {
    ierr = VecCopy(xin,yin);CHKERRQ(ierr);
  } else if (alpha == (PetscScalar)1.0) {
    ierr = VecAXPY_Seq(yin,alpha,xin);CHKERRQ(ierr);
  } else if (alpha == (PetscScalar)-1.0) {
    PetscInt i;
    ierr = VecGetArrayRead(xin,&xx);CHKERRQ(ierr);
    ierr = VecGetArray(yin,&yy);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      yy[i] = xx[i] - yy[i];
    }
    ierr = VecRestoreArrayRead(xin,&xx);CHKERRQ(ierr);
    ierr = VecRestoreArray(yin,&yy);CHKERRQ(ierr);
    ierr = PetscLogFlops(1.0*n);CHKERRQ(ierr);
  } else {
    ierr = VecGetArrayRead(xin,&xx);CHKERRQ(ierr);
    ierr = VecGetArray(yin,&yy);CHKERRQ(ierr);
#if defined(PETSC_USE_FORTRAN_KERNEL_AYPX)
    {
      PetscScalar oalpha = alpha;
      fortranaypx_(&n,&oalpha,xx,yy);
    }
#else
    {
      PetscInt i;
      for (i=0; i<n; i++) {
	yy[i] = xx[i] + alpha*yy[i];
      }
    }
#endif
    ierr = VecRestoreArrayRead(xin,&xx);CHKERRQ(ierr);
    ierr = VecRestoreArray(yin,&yy);CHKERRQ(ierr);
    ierr = PetscLogFlops(2.0*n);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
#endif

#include <../src/vec/vec/impls/seq/ftn-kernels/fwaxpy.h>
/*
   IBM ESSL contains a routine dzaxpy() that is our WAXPY() but it appears
  to be slower than a regular C loop.  Hence,we do not include it.
  void ?zaxpy(int*,PetscScalar*,PetscScalar*,int*,PetscScalar*,int*,PetscScalar*,int*);
*/
#if defined(PETSC_THREADCOMM_ACTIVE)
PetscErrorCode VecWAXPY_kernel(PetscInt thread_id,Vec win,PetscScalar *alpha_p,Vec xin,Vec yin)
{
  PetscErrorCode    ierr;
  PetscScalar       alpha=*alpha_p;
  PetscInt          *trstarts=win->map->trstarts,start,end,i;
  const PetscScalar *xx,*yy;
  PetscScalar       *ww;

  start = trstarts[thread_id];
  end   = trstarts[thread_id+1];

  ierr = VecGetArrayRead(xin,&xx);CHKERRQ(ierr);
  ierr = VecGetArrayRead(yin,&yy);CHKERRQ(ierr);
  ierr = VecGetArray(win,&ww);CHKERRQ(ierr);
  if (alpha == (PetscScalar)1.0) {
    for(i=start; i < end; i++) ww[i] = yy[i] + xx[i];
  } else if (alpha == (PetscScalar)-1.0) {
    for(i=start; i < end; i++) ww[i] = yy[i] - xx[i];
  } else if (alpha == (PetscScalar)0.0) {
    ierr = PetscMemcpy(ww+start,yy+start,(end-start)*sizeof(PetscScalar));CHKERRQ(ierr);
  } else {
    PetscScalar oalpha = alpha;
#if defined(PETSC_USE_FORTRAN_KERNEL_WAXPY)
    fortranwaxpy_(&n,&oalpha,xx+start,yy+start,ww+start);
#else
    for(i=start; i < end; i++) ww[i] = yy[i] + oalpha*xx[i];
#endif
  }
  ierr = VecRestoreArrayRead(xin,&xx);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(yin,&yy);CHKERRQ(ierr);
  ierr = VecRestoreArray(win,&ww);CHKERRQ(ierr);
  return 0;
}

#undef __FUNCT__  
#define __FUNCT__ "VecWAXPY_Seq" 
PetscErrorCode VecWAXPY_Seq(Vec win, PetscScalar alpha,Vec xin,Vec yin)
{
  PetscErrorCode     ierr;
  PetscInt           n = win->map->n;
  PetscScalar        *scal1;

  PetscFunctionBegin;
  ierr = PetscThreadCommGetScalars(((PetscObject)win)->comm,&scal1,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  *scal1 = alpha;
  ierr = PetscThreadCommRunKernel4(((PetscObject)win)->comm,(PetscThreadKernel)VecWAXPY_kernel,win,scal1,xin,yin);CHKERRQ(ierr);
  if (alpha == (PetscScalar)1.0) {
    ierr = PetscLogFlops(n);CHKERRQ(ierr);
  } else if (alpha == (PetscScalar)-1.0) {
    ierr = PetscLogFlops(n);CHKERRQ(ierr);
  } else {
    ierr = PetscLogFlops(2.0*n);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
#else
#undef __FUNCT__
#define __FUNCT__ "VecWAXPY_Seq"
PetscErrorCode VecWAXPY_Seq(Vec win, PetscScalar alpha,Vec xin,Vec yin)
{
  PetscErrorCode     ierr;
  PetscInt           i,n = win->map->n;
  PetscScalar        *ww;
  const PetscScalar  *yy,*xx;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(xin,&xx);CHKERRQ(ierr);
  ierr = VecGetArrayRead(yin,&yy);CHKERRQ(ierr);
  ierr = VecGetArray(win,&ww);CHKERRQ(ierr);
  if (alpha == (PetscScalar)1.0) {
    ierr = PetscLogFlops(n);CHKERRQ(ierr);
    /* could call BLAS axpy after call to memcopy, but may be slower */
    for (i=0; i<n; i++) ww[i] = yy[i] + xx[i];
  } else if (alpha == (PetscScalar)-1.0) {
    ierr = PetscLogFlops(n);CHKERRQ(ierr);
    for (i=0; i<n; i++) ww[i] = yy[i] - xx[i];
  } else if (alpha == (PetscScalar)0.0) {
    ierr = PetscMemcpy(ww,yy,n*sizeof(PetscScalar));CHKERRQ(ierr);
  } else {
    PetscScalar oalpha = alpha;
#if defined(PETSC_USE_FORTRAN_KERNEL_WAXPY)
    fortranwaxpy_(&n,&oalpha,xx,yy,ww);
#else
    for (i=0; i<n; i++) ww[i] = yy[i] + oalpha * xx[i];
#endif
    ierr = PetscLogFlops(2.0*n);CHKERRQ(ierr);
  }
  ierr = VecRestoreArrayRead(xin,&xx);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(yin,&yy);CHKERRQ(ierr);
  ierr = VecRestoreArray(win,&ww);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

#undef __FUNCT__
#define __FUNCT__ "VecMaxPointwiseDivide_Seq"
PetscErrorCode VecMaxPointwiseDivide_Seq(Vec xin,Vec yin,PetscReal *max)
{
  PetscErrorCode    ierr;
  PetscInt          n = xin->map->n,i;
  const PetscScalar *xx,*yy;
  PetscReal         m = 0.0;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(xin,&xx);CHKERRQ(ierr);
  ierr = VecGetArrayRead(yin,&yy);CHKERRQ(ierr);
  for (i = 0; i < n; i++) {
    if (yy[i] != (PetscScalar)0.0) {
      m = PetscMax(PetscAbsScalar(xx[i]/yy[i]), m);
    } else {
      m = PetscMax(PetscAbsScalar(xx[i]), m);
    }
  }
  ierr = VecRestoreArrayRead(xin,&xx);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(yin,&yy);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&m,max,1,MPIU_REAL,MPIU_MAX,((PetscObject)xin)->comm);CHKERRQ(ierr);
  ierr = PetscLogFlops(n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecPlaceArray_Seq"
PetscErrorCode VecPlaceArray_Seq(Vec vin,const PetscScalar *a)
{
  Vec_Seq *v = (Vec_Seq *)vin->data;

  PetscFunctionBegin;
  if (v->unplacedarray) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"VecPlaceArray() was already called on this vector, without a call to VecResetArray()");
  v->unplacedarray = v->array;  /* save previous array so reset can bring it back */
  v->array = (PetscScalar *)a;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecReplaceArray_Seq"
PetscErrorCode VecReplaceArray_Seq(Vec vin,const PetscScalar *a)
{
  Vec_Seq        *v = (Vec_Seq *)vin->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(v->array_allocated);CHKERRQ(ierr);
  v->array_allocated = v->array = (PetscScalar *)a;
  PetscFunctionReturn(0);
}
