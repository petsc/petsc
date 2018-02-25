
/*
   Defines some vector operation functions that are shared by
   sequential and parallel vectors.
 */
#include <../src/vec/vec/impls/dvecimpl.h>
#include <petsc/private/kernels/petscaxpy.h>

#if defined(PETSC_HAVE_IMMINTRIN_H)
#include <immintrin.h>
#define _MM256_TRANSPOSE4_PD(row0, row1, row2, row3) {      \
  __m256d tmp3, tmp2, tmp1, tmp0;                         \
  \
  tmp0   = _mm256_unpacklo_pd((row0), (row1));            \
  tmp2   = _mm256_unpacklo_pd((row2), (row3));            \
  tmp1   = _mm256_unpackhi_pd((row0), (row1));            \
  tmp3   = _mm256_unpackhi_pd((row2), (row3));            \
  \
  (row0) = _mm256_permute2f128_pd(tmp0, tmp2, 0x20);      \
  (row2) = _mm256_permute2f128_pd(tmp2, tmp0, 0x13);      \
  (row1) = _mm256_permute2f128_pd(tmp1, tmp3, 0x20);      \
  (row3) = _mm256_permute2f128_pd(tmp3, tmp1, 0x13);      \
}
#endif

#if defined(PETSC_USE_FORTRAN_KERNEL_MDOT)
#include <../src/vec/vec/impls/seq/ftn-kernels/fmdot.h>
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
#if defined(PETSC_HAVE_IMMINTRIN_H) && defined(__AVX2__) && defined(PETSC_USE_REAL_DOUBLE) && !defined(PETSC_USE_COMPLEX)
PetscErrorCode VecMDot_Seq(Vec xin,PetscInt nv,const Vec yin[],PetscScalar *z)
{
  PetscErrorCode    ierr;
  PetscInt          n = xin->map->n,i,j,nv_rem,j_rem;
  PetscScalar       sum0,sum1,sum2,sum3,x0,x1,x2,x3;
  PetscScalar       sum4,sum5,sum6,sum7,x4,x5,x6,x7;
  PetscScalar       sum8,sum9,sum10,sum11,x8,x9,x10,x11;
  PetscScalar       sum12,sum13,sum14,sum15,x12,x13,x14,x15;
  const PetscScalar *yy0,*yy1,*yy2,*yy3,*x,*xbase;
  const PetscScalar *yy4,*yy5,*yy6,*yy7,*yy8,*yy9,*yy10,*yy11,*yy12,*yy13,*yy14,*yy15;
  Vec               *yy;
  __m256i mask;
  __m256d s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15;
  __m256d v0,v1,v2,v3;
  __m256d y0,y1,y2,y3,y4,y5,y6,y7,y8,y9,y10,y11,y12,y13,y14,y15;
  __m256d t0,t1,t2,t3,t4,t5,t6,t7;

  PetscFunctionBegin;
  sum0 = 0.;
  sum1 = 0.;
  sum2 = 0.;

  i      = nv;
  nv_rem = nv&0x3;
  yy     = (Vec*)yin;
  j      = n;
  ierr   = VecGetArrayRead(xin,&xbase);CHKERRQ(ierr);
  x      = xbase;

  switch (nv_rem) {
    case 3:
      sum0 = 0.; sum1 = 0.; sum2 = 0.;
      ierr = VecGetArrayRead(yy[0],&yy0);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[1],&yy1);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[2],&yy2);CHKERRQ(ierr);
      j = n;
      x = xbase;
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
          yy3 += j_rem;
          j   -= j_rem;
          break;
      }

      s0 = _mm256_set_pd(0., 0., 0., sum0); s1 = _mm256_set_pd(0., 0., 0., sum1);
      s2 = _mm256_set_pd(0., 0., 0., sum2); t3 = _mm256_setzero_pd();

      while (j>0) {
        v0 = _mm256_loadu_pd(&x[0]);
        x += 4;
        y0 = _mm256_loadu_pd(&yy0[0]);
        y1 = _mm256_loadu_pd(&yy1[0]);
        y2 = _mm256_loadu_pd(&yy2[0]);
        yy0+=4; yy1+=4; yy2+=4;
        s0 = _mm256_fmadd_pd(v0,y0,s0);
        s1 = _mm256_fmadd_pd(v0,y1,s1);
        s2 = _mm256_fmadd_pd(v0,y2,s2);
        j -= 4;
      } /* End while loop on j (vector length) */

           _MM256_TRANSPOSE4_PD(s0, s1, s2, s3);
      t0 = _mm256_add_pd(s0,s1);
      t1 = _mm256_add_pd(s2,s3);
      s0 = _mm256_add_pd(t0,t1);
           _mm256_maskstore_pd (&z[0],_mm256_set_epi64x(0LL,1LL<<63,1LL<<63,1LL<<63),s0);
      z   += 3;
      i   -= 3;
      ierr = VecRestoreArrayRead(yy[0],&yy0);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[1],&yy1);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[2],&yy2);CHKERRQ(ierr);
      yy  += 3;
      break;
    case 2:
      ierr = VecGetArrayRead(yy[0],&yy0);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[1],&yy1);CHKERRQ(ierr);
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

      ierr = VecRestoreArrayRead(yy[0],&yy0);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[1],&yy1);CHKERRQ(ierr);
      z  += nv_rem; i  -= nv_rem; yy += nv_rem;
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
          +  x[2]*PetscConj(yy0[2]) + x[3]*PetscConj(yy0[3]);
        yy0  +=4;
        j    -= 4; x+=4;
      }
      z[0] = sum0;

      ierr = VecRestoreArrayRead(yy[0],&yy0);CHKERRQ(ierr);
      z  += nv_rem; i  -= nv_rem; yy += nv_rem;
      break;
    case 0:
      break;
  }

  /* i is now zero or an integer multiple of 4 */
  switch (i) {
    case 32:
      sum0  = 0.; sum1  = 0.; sum2  = 0.; sum3  = 0.;
      sum4  = 0.; sum5  = 0.; sum6  = 0.; sum7  = 0.;
      sum8  = 0.; sum9  = 0.; sum10 = 0.; sum11 = 0.;
      sum12 = 0.; sum13 = 0.; sum14 = 0.; sum15 = 0.;

      ierr = VecGetArrayRead(yy[0 ],&yy0);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[1 ],&yy1);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[2 ],&yy2);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[3 ],&yy3);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[4 ],&yy4);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[5 ],&yy5);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[6 ],&yy6);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[7 ],&yy7);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[8 ],&yy8);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[9 ],&yy9);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[10],&yy10);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[11],&yy11);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[12],&yy12);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[13],&yy13);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[14],&yy14);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[15],&yy15);CHKERRQ(ierr);
      j = n;
      x = xbase;
      switch (j_rem=j&0x3) {
        case 3:
          x2    = x[2];
          sum0 += x2*PetscConj(yy0 [2]); sum1 += x2*PetscConj(yy1 [2]);
          sum2 += x2*PetscConj(yy2 [2]); sum3 += x2*PetscConj(yy3 [2]);
          sum4 += x2*PetscConj(yy4 [2]); sum5 += x2*PetscConj(yy5 [2]);
          sum6 += x2*PetscConj(yy6 [2]); sum7 += x2*PetscConj(yy7 [2]);
          sum8 += x2*PetscConj(yy8 [2]); sum9 += x2*PetscConj(yy9 [2]);
          sum10+= x2*PetscConj(yy10[2]); sum11+= x2*PetscConj(yy11[2]);
          sum12+= x2*PetscConj(yy12[2]); sum13+= x2*PetscConj(yy13[2]);
          sum14+= x2*PetscConj(yy14[2]); sum15+= x2*PetscConj(yy15[2]);
        case 2:
          x1    = x[1];
          sum0 += x1*PetscConj(yy0 [1]); sum1 += x1*PetscConj(yy1 [1]);
          sum2 += x1*PetscConj(yy2 [1]); sum3 += x1*PetscConj(yy3 [1]);
          sum4 += x1*PetscConj(yy4 [1]); sum5 += x1*PetscConj(yy5 [1]);
          sum6 += x1*PetscConj(yy6 [1]); sum7 += x1*PetscConj(yy7 [1]);
          sum8 += x1*PetscConj(yy8 [1]); sum9 += x1*PetscConj(yy9 [1]);
          sum10+= x1*PetscConj(yy10[1]); sum11+= x1*PetscConj(yy11[1]);
          sum12+= x1*PetscConj(yy12[1]); sum13+= x1*PetscConj(yy13[1]);
          sum14+= x1*PetscConj(yy14[1]); sum15+= x1*PetscConj(yy15[1]);
        case 1:
          x0    = x[0];
          sum0 += x0*PetscConj(yy0 [0]); sum1 += x0*PetscConj(yy1 [0]);
          sum2 += x0*PetscConj(yy2 [0]); sum3 += x0*PetscConj(yy3 [0]);
          sum4 += x0*PetscConj(yy4 [0]); sum5 += x0*PetscConj(yy5 [0]);
          sum6 += x0*PetscConj(yy6 [0]); sum7 += x0*PetscConj(yy7 [0]);
          sum8 += x0*PetscConj(yy8 [0]); sum9 += x0*PetscConj(yy9 [0]);
          sum10+= x0*PetscConj(yy10[0]); sum11+= x0*PetscConj(yy11[0]);
          sum12+= x0*PetscConj(yy12[0]); sum13+= x0*PetscConj(yy13[0]);
          sum14+= x0*PetscConj(yy14[0]); sum15+= x0*PetscConj(yy15[0]);
        case 0:
          x   += j_rem;
          yy0 += j_rem;
          yy1 += j_rem;
          yy2 += j_rem;
          yy3 += j_rem;
          j   -= j_rem;
          break;
      }

      s0  = _mm256_set_pd(0., 0., 0., sum0);   s1 = _mm256_set_pd(0., 0., 0., sum1);
      s2  = _mm256_set_pd(0., 0., 0., sum2);   s3 = _mm256_set_pd(0., 0., 0., sum3);
      s4  = _mm256_set_pd(0., 0., 0., sum4);   s5 = _mm256_set_pd(0., 0., 0., sum5);
      s6  = _mm256_set_pd(0., 0., 0., sum6);   s7 = _mm256_set_pd(0., 0., 0., sum7);
      s8  = _mm256_set_pd(0., 0., 0., sum8);   s9 = _mm256_set_pd(0., 0., 0., sum9);
      s10 = _mm256_set_pd(0., 0., 0., sum10); s11 = _mm256_set_pd(0., 0., 0., sum11);
      s12 = _mm256_set_pd(0., 0., 0., sum12); s13 = _mm256_set_pd(0., 0., 0., sum13);
      s14 = _mm256_set_pd(0., 0., 0., sum14); s15 = _mm256_set_pd(0., 0., 0., sum15);

      while (j>0) {
        v0 = _mm256_loadu_pd(&x[0]);
        x += 4;
        y0 = _mm256_loadu_pd(&yy0[0]);
        y1 = _mm256_loadu_pd(&yy1[0]);
        y2 = _mm256_loadu_pd(&yy2[0]);
        y3 = _mm256_loadu_pd(&yy3[0]);
        yy0+=4; yy1+=4; yy2+=4; yy3+=4;

        s0 = _mm256_fmadd_pd(v0,y0,s0);
        s1 = _mm256_fmadd_pd(v0,y1,s1);
        s2 = _mm256_fmadd_pd(v0,y2,s2);
        s3 = _mm256_fmadd_pd(v0,y3,s3);

        y4 = _mm256_loadu_pd(&yy4[0]);
        y5 = _mm256_loadu_pd(&yy5[0]);
        y6 = _mm256_loadu_pd(&yy6[0]);
        y7 = _mm256_loadu_pd(&yy7[0]);
        yy4+=4; yy5+=4; yy6+=4; yy7+=4;

        s4 = _mm256_fmadd_pd(v0,y4,s4);
        s5 = _mm256_fmadd_pd(v0,y5,s5);
        s6 = _mm256_fmadd_pd(v0,y6,s6);
        s7 = _mm256_fmadd_pd(v0,y7,s7);

        y8  = _mm256_loadu_pd(&yy8[0]);
        y9  = _mm256_loadu_pd(&yy9[0]);
        y10 = _mm256_loadu_pd(&yy10[0]);
        y11 = _mm256_loadu_pd(&yy11[0]);
        yy8+=4; yy9+=4; yy10+=4; yy11+=4;

        s8  = _mm256_fmadd_pd(v0,y8,s8);
        s9  = _mm256_fmadd_pd(v0,y9,s9);
        s10 = _mm256_fmadd_pd(v0,y10,s10);
        s11 = _mm256_fmadd_pd(v0,y11,s11);

        y12 = _mm256_loadu_pd(&yy12[0]);
        y13 = _mm256_loadu_pd(&yy13[0]);
        y14 = _mm256_loadu_pd(&yy14[0]);
        y15 = _mm256_loadu_pd(&yy15[0]);
        yy12+=4; yy13+=4; yy14+=4; yy15+=4;

        s12 = _mm256_fmadd_pd(v0,y12,s12);
        s13 = _mm256_fmadd_pd(v0,y13,s13);
        s14 = _mm256_fmadd_pd(v0,y14,s14);
        s15 = _mm256_fmadd_pd(v0,y15,s15);
        j -= 4;
      } /* End while loop on j (vector length) */

           _MM256_TRANSPOSE4_PD(s0, s1, s2, s3);
      t0 = _mm256_add_pd(s0,s1);
      t1 = _mm256_add_pd(s2,s3);
      s0 = _mm256_add_pd(t0,t1);
           _mm256_store_pd(&z[0], s0);

           _MM256_TRANSPOSE4_PD(s4, s5, s6, s7);
      t2 = _mm256_add_pd(s4,s5);
      t3 = _mm256_add_pd(s6,s7);
      s4 = _mm256_add_pd(t2,t3);
           _mm256_store_pd(&z[4], s4);

           _MM256_TRANSPOSE4_PD(s8, s9, s10, s11);
      t4 = _mm256_add_pd(s8,s9);
      t5 = _mm256_add_pd(s10,s11);
      s8 = _mm256_add_pd(t4,t5);
           _mm256_store_pd(&z[8], s8);

           _MM256_TRANSPOSE4_PD(s12, s13, s14, s15);
      t6 = _mm256_add_pd(s12,s13);
      t7 = _mm256_add_pd(s14,s15);
      s12= _mm256_add_pd(t6,t7);
           _mm256_store_pd(&z[12], s12);

      z   += 16;
      i   -= 16;
      ierr = VecRestoreArrayRead(yy[0 ],&yy0 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[1 ],&yy1 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[2 ],&yy2 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[3 ],&yy3 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[4 ],&yy4 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[5 ],&yy5 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[6 ],&yy6 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[7 ],&yy7 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[8 ],&yy8 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[9 ],&yy9 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[10],&yy10);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[11],&yy11);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[12],&yy12);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[13],&yy13);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[14],&yy14);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[15],&yy15);CHKERRQ(ierr);
      yy  += 16;
      sum0  = 0.; sum1  = 0.; sum2  = 0.; sum3  = 0.;
      sum4  = 0.; sum5  = 0.; sum6  = 0.; sum7  = 0.;
      sum8  = 0.; sum9  = 0.; sum10 = 0.; sum11 = 0.;
      sum12 = 0.; sum13 = 0.; sum14 = 0.; sum15 = 0.;

      ierr = VecGetArrayRead(yy[0 ],&yy0);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[1 ],&yy1);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[2 ],&yy2);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[3 ],&yy3);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[4 ],&yy4);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[5 ],&yy5);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[6 ],&yy6);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[7 ],&yy7);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[8 ],&yy8);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[9 ],&yy9);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[10],&yy10);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[11],&yy11);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[12],&yy12);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[13],&yy13);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[14],&yy14);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[15],&yy15);CHKERRQ(ierr);
      j = n;
      x = xbase;
      switch (j_rem=j&0x3) {
        case 3:
          x2    = x[2];
          sum0 += x2*PetscConj(yy0 [2]); sum1 += x2*PetscConj(yy1 [2]);
          sum2 += x2*PetscConj(yy2 [2]); sum3 += x2*PetscConj(yy3 [2]);
          sum4 += x2*PetscConj(yy4 [2]); sum5 += x2*PetscConj(yy5 [2]);
          sum6 += x2*PetscConj(yy6 [2]); sum7 += x2*PetscConj(yy7 [2]);
          sum8 += x2*PetscConj(yy8 [2]); sum9 += x2*PetscConj(yy9 [2]);
          sum10+= x2*PetscConj(yy10[2]); sum11+= x2*PetscConj(yy11[2]);
          sum12+= x2*PetscConj(yy12[2]); sum13+= x2*PetscConj(yy13[2]);
          sum14+= x2*PetscConj(yy14[2]); sum15+= x2*PetscConj(yy15[2]);
        case 2:
          x1    = x[1];
          sum0 += x1*PetscConj(yy0 [1]); sum1 += x1*PetscConj(yy1 [1]);
          sum2 += x1*PetscConj(yy2 [1]); sum3 += x1*PetscConj(yy3 [1]);
          sum4 += x1*PetscConj(yy4 [1]); sum5 += x1*PetscConj(yy5 [1]);
          sum6 += x1*PetscConj(yy6 [1]); sum7 += x1*PetscConj(yy7 [1]);
          sum8 += x1*PetscConj(yy8 [1]); sum9 += x1*PetscConj(yy9 [1]);
          sum10+= x1*PetscConj(yy10[1]); sum11+= x1*PetscConj(yy11[1]);
          sum12+= x1*PetscConj(yy12[1]); sum13+= x1*PetscConj(yy13[1]);
          sum14+= x1*PetscConj(yy14[1]); sum15+= x1*PetscConj(yy15[1]);
        case 1:
          x0    = x[0];
          sum0 += x0*PetscConj(yy0 [0]); sum1 += x0*PetscConj(yy1 [0]);
          sum2 += x0*PetscConj(yy2 [0]); sum3 += x0*PetscConj(yy3 [0]);
          sum4 += x0*PetscConj(yy4 [0]); sum5 += x0*PetscConj(yy5 [0]);
          sum6 += x0*PetscConj(yy6 [0]); sum7 += x0*PetscConj(yy7 [0]);
          sum8 += x0*PetscConj(yy8 [0]); sum9 += x0*PetscConj(yy9 [0]);
          sum10+= x0*PetscConj(yy10[0]); sum11+= x0*PetscConj(yy11[0]);
          sum12+= x0*PetscConj(yy12[0]); sum13+= x0*PetscConj(yy13[0]);
          sum14+= x0*PetscConj(yy14[0]); sum15+= x0*PetscConj(yy15[0]);
        case 0:
          x   += j_rem;
          yy0 += j_rem;
          yy1 += j_rem;
          yy2 += j_rem;
          yy3 += j_rem;
          j   -= j_rem;
          break;
      }

      s0  = _mm256_set_pd(0., 0., 0., sum0);   s1 = _mm256_set_pd(0., 0., 0., sum1);
      s2  = _mm256_set_pd(0., 0., 0., sum2);   s3 = _mm256_set_pd(0., 0., 0., sum3);
      s4  = _mm256_set_pd(0., 0., 0., sum4);   s5 = _mm256_set_pd(0., 0., 0., sum5);
      s6  = _mm256_set_pd(0., 0., 0., sum6);   s7 = _mm256_set_pd(0., 0., 0., sum7);
      s8  = _mm256_set_pd(0., 0., 0., sum8);   s9 = _mm256_set_pd(0., 0., 0., sum9);
      s10 = _mm256_set_pd(0., 0., 0., sum10); s11 = _mm256_set_pd(0., 0., 0., sum11);
      s12 = _mm256_set_pd(0., 0., 0., sum12); s13 = _mm256_set_pd(0., 0., 0., sum13);
      s14 = _mm256_set_pd(0., 0., 0., sum14); s15 = _mm256_set_pd(0., 0., 0., sum15);

      while (j>0) {
        v0 = _mm256_loadu_pd(&x[0]);
        x += 4;
        y0 = _mm256_loadu_pd(&yy0[0]);
        y1 = _mm256_loadu_pd(&yy1[0]);
        y2 = _mm256_loadu_pd(&yy2[0]);
        y3 = _mm256_loadu_pd(&yy3[0]);
        yy0+=4; yy1+=4; yy2+=4; yy3+=4;

        s0 = _mm256_fmadd_pd(v0,y0,s0);
        s1 = _mm256_fmadd_pd(v0,y1,s1);
        s2 = _mm256_fmadd_pd(v0,y2,s2);
        s3 = _mm256_fmadd_pd(v0,y3,s3);

        y4 = _mm256_loadu_pd(&yy4[0]);
        y5 = _mm256_loadu_pd(&yy5[0]);
        y6 = _mm256_loadu_pd(&yy6[0]);
        y7 = _mm256_loadu_pd(&yy7[0]);
        yy4+=4; yy5+=4; yy6+=4; yy7+=4;

        s4 = _mm256_fmadd_pd(v0,y4,s4);
        s5 = _mm256_fmadd_pd(v0,y5,s5);
        s6 = _mm256_fmadd_pd(v0,y6,s6);
        s7 = _mm256_fmadd_pd(v0,y7,s7);

        y8  = _mm256_loadu_pd(&yy8[0]);
        y9  = _mm256_loadu_pd(&yy9[0]);
        y10 = _mm256_loadu_pd(&yy10[0]);
        y11 = _mm256_loadu_pd(&yy11[0]);
        yy8+=4; yy9+=4; yy10+=4; yy11+=4;

        s8  = _mm256_fmadd_pd(v0,y8,s8);
        s9  = _mm256_fmadd_pd(v0,y9,s9);
        s10 = _mm256_fmadd_pd(v0,y10,s10);
        s11 = _mm256_fmadd_pd(v0,y11,s11);

        y12 = _mm256_loadu_pd(&yy12[0]);
        y13 = _mm256_loadu_pd(&yy13[0]);
        y14 = _mm256_loadu_pd(&yy14[0]);
        y15 = _mm256_loadu_pd(&yy15[0]);
        yy12+=4; yy13+=4; yy14+=4; yy15+=4;

        s12 = _mm256_fmadd_pd(v0,y12,s12);
        s13 = _mm256_fmadd_pd(v0,y13,s13);
        s14 = _mm256_fmadd_pd(v0,y14,s14);
        s15 = _mm256_fmadd_pd(v0,y15,s15);
        j -= 4;
      } /* End while loop on j (vector length) */

           _MM256_TRANSPOSE4_PD(s0, s1, s2, s3);
      t0 = _mm256_add_pd(s0,s1);
      t1 = _mm256_add_pd(s2,s3);
      s0 = _mm256_add_pd(t0,t1);
           _mm256_store_pd(&z[0], s0);

           _MM256_TRANSPOSE4_PD(s4, s5, s6, s7);
      t2 = _mm256_add_pd(s4,s5);
      t3 = _mm256_add_pd(s6,s7);
      s4 = _mm256_add_pd(t2,t3);
           _mm256_store_pd(&z[4], s4);

           _MM256_TRANSPOSE4_PD(s8, s9, s10, s11);
      t4 = _mm256_add_pd(s8,s9);
      t5 = _mm256_add_pd(s10,s11);
      s8 = _mm256_add_pd(t4,t5);
           _mm256_store_pd(&z[8], s8);

           _MM256_TRANSPOSE4_PD(s12, s13, s14, s15);
      t6 = _mm256_add_pd(s12,s13);
      t7 = _mm256_add_pd(s14,s15);
      s12= _mm256_add_pd(t6,t7);
           _mm256_store_pd(&z[12], s12);

      z   += 16;
      i   -= 16;
      ierr = VecRestoreArrayRead(yy[0 ],&yy0 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[1 ],&yy1 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[2 ],&yy2 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[3 ],&yy3 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[4 ],&yy4 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[5 ],&yy5 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[6 ],&yy6 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[7 ],&yy7 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[8 ],&yy8 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[9 ],&yy9 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[10],&yy10);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[11],&yy11);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[12],&yy12);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[13],&yy13);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[14],&yy14);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[15],&yy15);CHKERRQ(ierr);
      yy  += 16;
      break;
    case 28:
      sum0 = 0.; sum1 = 0.; sum2  = 0.; sum3  = 0.;
      sum4 = 0.; sum5 = 0.; sum6  = 0.; sum7  = 0.;
      sum8 = 0.; sum9 = 0.; sum10 = 0.; sum11 = 0.;
      ierr = VecGetArrayRead(yy[0 ],&yy0);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[1 ],&yy1);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[2 ],&yy2);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[3 ],&yy3);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[4 ],&yy4);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[5 ],&yy5);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[6 ],&yy6);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[7 ],&yy7);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[8 ],&yy8);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[9 ],&yy9);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[10],&yy10);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[11],&yy11);CHKERRQ(ierr);
      j = n;
      x = xbase;
      switch (j_rem=j&0x3) {
        case 3:
          x2    = x[2];
          sum0 += x2*PetscConj(yy0 [2]); sum1 += x2*PetscConj(yy1 [2]);
          sum2 += x2*PetscConj(yy2 [2]); sum3 += x2*PetscConj(yy3 [2]);
          sum4 += x2*PetscConj(yy4 [2]); sum5 += x2*PetscConj(yy5 [2]);
          sum6 += x2*PetscConj(yy6 [2]); sum7 += x2*PetscConj(yy7 [2]);
          sum8 += x2*PetscConj(yy8 [2]); sum9 += x2*PetscConj(yy9 [2]);
          sum10+= x2*PetscConj(yy10[2]); sum11+= x2*PetscConj(yy11[2]);
        case 2:
          x1    = x[1];
          sum0 += x1*PetscConj(yy0 [1]); sum1 += x1*PetscConj(yy1 [1]);
          sum2 += x1*PetscConj(yy2 [1]); sum3 += x1*PetscConj(yy3 [1]);
          sum4 += x1*PetscConj(yy4 [1]); sum5 += x1*PetscConj(yy5 [1]);
          sum6 += x1*PetscConj(yy6 [1]); sum7 += x1*PetscConj(yy7 [1]);
          sum8 += x1*PetscConj(yy8 [1]); sum9 += x1*PetscConj(yy9 [1]);
          sum10+= x1*PetscConj(yy10[1]); sum11+= x1*PetscConj(yy11[1]);
        case 1:
          x0    = x[0];
          sum0 += x0*PetscConj(yy0 [0]); sum1 += x0*PetscConj(yy1 [0]);
          sum2 += x0*PetscConj(yy2 [0]); sum3 += x0*PetscConj(yy3 [0]);
          sum4 += x0*PetscConj(yy4 [0]); sum5 += x0*PetscConj(yy5 [0]);
          sum6 += x0*PetscConj(yy6 [0]); sum7 += x0*PetscConj(yy7 [0]);
          sum8 += x0*PetscConj(yy8 [0]); sum9 += x0*PetscConj(yy9 [0]);
          sum10+= x0*PetscConj(yy10[0]); sum11+= x0*PetscConj(yy11[0]);
        case 0:
          x   += j_rem;
          yy0 += j_rem;
          yy1 += j_rem;
          yy2 += j_rem;
          yy3 += j_rem;
          j   -= j_rem;
          break;
      }

      s0  = _mm256_set_pd(0., 0., 0., sum0);   s1 = _mm256_set_pd(0., 0., 0., sum1);
      s2  = _mm256_set_pd(0., 0., 0., sum2);   s3 = _mm256_set_pd(0., 0., 0., sum3);
      s4  = _mm256_set_pd(0., 0., 0., sum4);   s5 = _mm256_set_pd(0., 0., 0., sum5);
      s6  = _mm256_set_pd(0., 0., 0., sum6);   s7 = _mm256_set_pd(0., 0., 0., sum7);
      s8  = _mm256_set_pd(0., 0., 0., sum8);   s9 = _mm256_set_pd(0., 0., 0., sum9);
      s10 = _mm256_set_pd(0., 0., 0., sum10); s11 = _mm256_set_pd(0., 0., 0., sum11);

      while (j>0) {
        v0 = _mm256_loadu_pd(&x[0]);
        x += 4;
        y0 = _mm256_loadu_pd(&yy0[0]);
        y1 = _mm256_loadu_pd(&yy1[0]);
        y2 = _mm256_loadu_pd(&yy2[0]);
        y3 = _mm256_loadu_pd(&yy3[0]);
        yy0+=4; yy1+=4; yy2+=4; yy3+=4;

        s0 = _mm256_fmadd_pd(v0,y0,s0);
        s1 = _mm256_fmadd_pd(v0,y1,s1);
        s2 = _mm256_fmadd_pd(v0,y2,s2);
        s3 = _mm256_fmadd_pd(v0,y3,s3);

        y4 = _mm256_loadu_pd(&yy4[0]);
        y5 = _mm256_loadu_pd(&yy5[0]);
        y6 = _mm256_loadu_pd(&yy6[0]);
        y7 = _mm256_loadu_pd(&yy7[0]);
        yy4+=4; yy5+=4; yy6+=4; yy7+=4;

        s4 = _mm256_fmadd_pd(v0,y4,s4);
        s5 = _mm256_fmadd_pd(v0,y5,s5);
        s6 = _mm256_fmadd_pd(v0,y6,s6);
        s7 = _mm256_fmadd_pd(v0,y7,s7);

        y8  = _mm256_loadu_pd(&yy8[0]);
        y9  = _mm256_loadu_pd(&yy9[0]);
        y10 = _mm256_loadu_pd(&yy10[0]);
        y11 = _mm256_loadu_pd(&yy11[0]);
        yy8+=4; yy9+=4; yy10+=4; yy11+=4;

        s8  = _mm256_fmadd_pd(v0,y8,s8);
        s9  = _mm256_fmadd_pd(v0,y9,s9);
        s10 = _mm256_fmadd_pd(v0,y10,s10);
        s11 = _mm256_fmadd_pd(v0,y11,s11);
        j -= 4;
      } /* End while loop on j (vector length) */

           _MM256_TRANSPOSE4_PD(s0, s1, s2, s3);
      t0 = _mm256_add_pd(s0,s1);
      t1 = _mm256_add_pd(s2,s3);
      s0 = _mm256_add_pd(t0,t1);
           _mm256_store_pd(&z[0], s0);

           _MM256_TRANSPOSE4_PD(s4, s5, s6, s7);
      t2 = _mm256_add_pd(s4,s5);
      t3 = _mm256_add_pd(s6,s7);
      s4 = _mm256_add_pd(t2,t3);
           _mm256_store_pd(&z[4], s4);

           _MM256_TRANSPOSE4_PD(s8, s9, s10, s11);
      t4 = _mm256_add_pd(s8,s9);
      t5 = _mm256_add_pd(s10,s11);
      s8 = _mm256_add_pd(t4,t5);
           _mm256_store_pd(&z[8], s8);

      z   += 12;
      i   -= 12;
      ierr = VecRestoreArrayRead(yy[0 ],&yy0 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[1 ],&yy1 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[2 ],&yy2 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[3 ],&yy3 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[4 ],&yy4 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[5 ],&yy5 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[6 ],&yy6 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[7 ],&yy7 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[8 ],&yy8 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[9 ],&yy9 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[10],&yy10);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[11],&yy11);CHKERRQ(ierr);
      yy  += 12;
      sum0 = 0.; sum1 = 0.; sum2  = 0.; sum3  = 0.;
      sum4 = 0.; sum5 = 0.; sum6  = 0.; sum7  = 0.;
      sum8 = 0.; sum9 = 0.; sum10 = 0.; sum11 = 0.;
      ierr = VecGetArrayRead(yy[0 ],&yy0);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[1 ],&yy1);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[2 ],&yy2);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[3 ],&yy3);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[4 ],&yy4);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[5 ],&yy5);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[6 ],&yy6);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[7 ],&yy7);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[8 ],&yy8);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[9 ],&yy9);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[10],&yy10);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[11],&yy11);CHKERRQ(ierr);
      j = n;
      x = xbase;
      switch (j_rem=j&0x3) {
        case 3:
          x2    = x[2];
          sum0 += x2*PetscConj(yy0 [2]); sum1 += x2*PetscConj(yy1 [2]);
          sum2 += x2*PetscConj(yy2 [2]); sum3 += x2*PetscConj(yy3 [2]);
          sum4 += x2*PetscConj(yy4 [2]); sum5 += x2*PetscConj(yy5 [2]);
          sum6 += x2*PetscConj(yy6 [2]); sum7 += x2*PetscConj(yy7 [2]);
          sum8 += x2*PetscConj(yy8 [2]); sum9 += x2*PetscConj(yy9 [2]);
          sum10+= x2*PetscConj(yy10[2]); sum11+= x2*PetscConj(yy11[2]);
        case 2:
          x1    = x[1];
          sum0 += x1*PetscConj(yy0 [1]); sum1 += x1*PetscConj(yy1 [1]);
          sum2 += x1*PetscConj(yy2 [1]); sum3 += x1*PetscConj(yy3 [1]);
          sum4 += x1*PetscConj(yy4 [1]); sum5 += x1*PetscConj(yy5 [1]);
          sum6 += x1*PetscConj(yy6 [1]); sum7 += x1*PetscConj(yy7 [1]);
          sum8 += x1*PetscConj(yy8 [1]); sum9 += x1*PetscConj(yy9 [1]);
          sum10+= x1*PetscConj(yy10[1]); sum11+= x1*PetscConj(yy11[1]);
        case 1:
          x0    = x[0];
          sum0 += x0*PetscConj(yy0 [0]); sum1 += x0*PetscConj(yy1 [0]);
          sum2 += x0*PetscConj(yy2 [0]); sum3 += x0*PetscConj(yy3 [0]);
          sum4 += x0*PetscConj(yy4 [0]); sum5 += x0*PetscConj(yy5 [0]);
          sum6 += x0*PetscConj(yy6 [0]); sum7 += x0*PetscConj(yy7 [0]);
          sum8 += x0*PetscConj(yy8 [0]); sum9 += x0*PetscConj(yy9 [0]);
          sum10+= x0*PetscConj(yy10[0]); sum11+= x0*PetscConj(yy11[0]);
        case 0:
          x   += j_rem;
          yy0 += j_rem;
          yy1 += j_rem;
          yy2 += j_rem;
          yy3 += j_rem;
          j   -= j_rem;
          break;
      }

      s0 = _mm256_set_pd(0., 0., 0., sum0);   s1 = _mm256_set_pd(0., 0., 0., sum1);
      s2 = _mm256_set_pd(0., 0., 0., sum2);   s3 = _mm256_set_pd(0., 0., 0., sum3);
      s4 = _mm256_set_pd(0., 0., 0., sum4);   s5 = _mm256_set_pd(0., 0., 0., sum5);
      s6 = _mm256_set_pd(0., 0., 0., sum6);   s7 = _mm256_set_pd(0., 0., 0., sum7);
      s8 = _mm256_set_pd(0., 0., 0., sum8);   s9 = _mm256_set_pd(0., 0., 0., sum9);
      s10 = _mm256_set_pd(0., 0., 0., sum10); s11 = _mm256_set_pd(0., 0., 0., sum11);

      while (j>0) {
        v0 = _mm256_loadu_pd(&x[0]);
        x += 4;
        y0 = _mm256_loadu_pd(&yy0[0]);
        y1 = _mm256_loadu_pd(&yy1[0]);
        y2 = _mm256_loadu_pd(&yy2[0]);
        y3 = _mm256_loadu_pd(&yy3[0]);
        yy0+=4; yy1+=4; yy2+=4; yy3+=4;

        s0 = _mm256_fmadd_pd(v0,y0,s0);
        s1 = _mm256_fmadd_pd(v0,y1,s1);
        s2 = _mm256_fmadd_pd(v0,y2,s2);
        s3 = _mm256_fmadd_pd(v0,y3,s3);

        y4 = _mm256_loadu_pd(&yy4[0]);
        y5 = _mm256_loadu_pd(&yy5[0]);
        y6 = _mm256_loadu_pd(&yy6[0]);
        y7 = _mm256_loadu_pd(&yy7[0]);
        yy4+=4; yy5+=4; yy6+=4; yy7+=4;

        s4 = _mm256_fmadd_pd(v0,y4,s4);
        s5 = _mm256_fmadd_pd(v0,y5,s5);
        s6 = _mm256_fmadd_pd(v0,y6,s6);
        s7 = _mm256_fmadd_pd(v0,y7,s7);

        y8 = _mm256_loadu_pd(&yy8[0]);
        y9 = _mm256_loadu_pd(&yy9[0]);
        y10= _mm256_loadu_pd(&yy10[0]);
        y11= _mm256_loadu_pd(&yy11[0]);
        yy8+=4; yy9+=4; yy10+=4; yy11+=4;

        s8 = _mm256_fmadd_pd(v0,y8,s8);
        s9 = _mm256_fmadd_pd(v0,y9,s9);
        s10= _mm256_fmadd_pd(v0,y10,s10);
        s11= _mm256_fmadd_pd(v0,y11,s11);
        j -= 4;
      } /* End while loop on j (vector length) */

           _MM256_TRANSPOSE4_PD(s0, s1, s2, s3);
      t0 = _mm256_add_pd(s0,s1);
      t1 = _mm256_add_pd(s2,s3);
      s0 = _mm256_add_pd(t0,t1);
           _mm256_store_pd(&z[0], s0);

           _MM256_TRANSPOSE4_PD(s4, s5, s6, s7);
      t2 = _mm256_add_pd(s4,s5);
      t3 = _mm256_add_pd(s6,s7);
      s4 = _mm256_add_pd(t2,t3);
           _mm256_store_pd(&z[4], s4);

           _MM256_TRANSPOSE4_PD(s8, s9, s10, s11);
      t4 = _mm256_add_pd(s8,s9);
      t5 = _mm256_add_pd(s10,s11);
      s8 = _mm256_add_pd(t4,t5);
           _mm256_store_pd(&z[8], s8);

      z   += 12;
      i   -= 12;
      ierr = VecRestoreArrayRead(yy[0 ],&yy0 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[1 ],&yy1 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[2 ],&yy2 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[3 ],&yy3 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[4 ],&yy4 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[5 ],&yy5 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[6 ],&yy6 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[7 ],&yy7 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[8 ],&yy8 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[9 ],&yy9 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[10],&yy10);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[11],&yy11);CHKERRQ(ierr);
      yy  += 12;

      sum0 = 0.; sum1 = 0.; sum2 = 0.; sum3 = 0.;
      ierr = VecGetArrayRead(yy[0],&yy0);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[1],&yy1);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[2],&yy2);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[3],&yy3);CHKERRQ(ierr);
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

      s0 = _mm256_set_pd(0., 0., 0., sum0); s1 = _mm256_set_pd(0., 0., 0., sum1);
      s2 = _mm256_set_pd(0., 0., 0., sum2); s3 = _mm256_set_pd(0., 0., 0., sum3);

      while (j>0) {
        v0 = _mm256_loadu_pd(&x[0]);
        x += 4;
        y0 = _mm256_loadu_pd(&yy0[0]);
        y1 = _mm256_loadu_pd(&yy1[0]);
        y2 = _mm256_loadu_pd(&yy2[0]);
        y3 = _mm256_loadu_pd(&yy3[0]);
        yy0+=4; yy1+=4; yy2+=4; yy3+=4;
        s0 = _mm256_fmadd_pd(v0,y0,s0);
        s1 = _mm256_fmadd_pd(v0,y1,s1);
        s2 = _mm256_fmadd_pd(v0,y2,s2);
        s3 = _mm256_fmadd_pd(v0,y3,s3);
        j -= 4;
      } /* End while loop on j (vector length) */

           _MM256_TRANSPOSE4_PD(s0, s1, s2, s3);
      t0 = _mm256_add_pd(s0,s1);
      t1 = _mm256_add_pd(s2,s3);
      s0 = _mm256_add_pd(t0,t1);
           _mm256_store_pd(&z[0], s0);

      z   += 4;
      i   -= 4;
      ierr = VecRestoreArrayRead(yy[0],&yy0);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[1],&yy1);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[2],&yy2);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[3],&yy3);CHKERRQ(ierr);
      yy  += 4;
      break;
    case 24:
      sum0 = 0.; sum1 = 0.; sum2  = 0.; sum3  = 0.;
      sum4 = 0.; sum5 = 0.; sum6  = 0.; sum7  = 0.;
      sum8 = 0.; sum9 = 0.; sum10 = 0.; sum11 = 0.;
      ierr = VecGetArrayRead(yy[0 ],&yy0);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[1 ],&yy1);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[2 ],&yy2);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[3 ],&yy3);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[4 ],&yy4);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[5 ],&yy5);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[6 ],&yy6);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[7 ],&yy7);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[8 ],&yy8);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[9 ],&yy9);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[10],&yy10);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[11],&yy11);CHKERRQ(ierr);
      j = n;
      x = xbase;
      switch (j_rem=j&0x3) {
        case 3:
          x2    = x[2];
          sum0 += x2*PetscConj(yy0 [2]); sum1 += x2*PetscConj(yy1 [2]);
          sum2 += x2*PetscConj(yy2 [2]); sum3 += x2*PetscConj(yy3 [2]);
          sum4 += x2*PetscConj(yy4 [2]); sum5 += x2*PetscConj(yy5 [2]);
          sum6 += x2*PetscConj(yy6 [2]); sum7 += x2*PetscConj(yy7 [2]);
          sum8 += x2*PetscConj(yy8 [2]); sum9 += x2*PetscConj(yy9 [2]);
          sum10+= x2*PetscConj(yy10[2]); sum11+= x2*PetscConj(yy11[2]);
        case 2:
          x1    = x[1];
          sum0 += x1*PetscConj(yy0 [1]); sum1 += x1*PetscConj(yy1 [1]);
          sum2 += x1*PetscConj(yy2 [1]); sum3 += x1*PetscConj(yy3 [1]);
          sum4 += x1*PetscConj(yy4 [1]); sum5 += x1*PetscConj(yy5 [1]);
          sum6 += x1*PetscConj(yy6 [1]); sum7 += x1*PetscConj(yy7 [1]);
          sum8 += x1*PetscConj(yy8 [1]); sum9 += x1*PetscConj(yy9 [1]);
          sum10+= x1*PetscConj(yy10[1]); sum11+= x1*PetscConj(yy11[1]);
        case 1:
          x0    = x[0];
          sum0 += x0*PetscConj(yy0 [0]); sum1 += x0*PetscConj(yy1 [0]);
          sum2 += x0*PetscConj(yy2 [0]); sum3 += x0*PetscConj(yy3 [0]);
          sum4 += x0*PetscConj(yy4 [0]); sum5 += x0*PetscConj(yy5 [0]);
          sum6 += x0*PetscConj(yy6 [0]); sum7 += x0*PetscConj(yy7 [0]);
          sum8 += x0*PetscConj(yy8 [0]); sum9 += x0*PetscConj(yy9 [0]);
          sum10+= x0*PetscConj(yy10[0]); sum11+= x0*PetscConj(yy11[0]);
        case 0:
          x   += j_rem;
          yy0 += j_rem;
          yy1 += j_rem;
          yy2 += j_rem;
          yy3 += j_rem;
          j   -= j_rem;
          break;
      }

      s0 = _mm256_set_pd(0., 0., 0., sum0);   s1 = _mm256_set_pd(0., 0., 0., sum1);
      s2 = _mm256_set_pd(0., 0., 0., sum2);   s3 = _mm256_set_pd(0., 0., 0., sum3);
      s4 = _mm256_set_pd(0., 0., 0., sum4);   s5 = _mm256_set_pd(0., 0., 0., sum5);
      s6 = _mm256_set_pd(0., 0., 0., sum6);   s7 = _mm256_set_pd(0., 0., 0., sum7);
      s8 = _mm256_set_pd(0., 0., 0., sum8);   s9 = _mm256_set_pd(0., 0., 0., sum9);
      s10= _mm256_set_pd(0., 0., 0., sum10); s11 = _mm256_set_pd(0., 0., 0., sum11);

      while (j>0) {
        v0 = _mm256_loadu_pd(&x[0]);
        x += 4;
        y0 = _mm256_loadu_pd(&yy0[0]);
        y1 = _mm256_loadu_pd(&yy1[0]);
        y2 = _mm256_loadu_pd(&yy2[0]);
        y3 = _mm256_loadu_pd(&yy3[0]);
        yy0+=4; yy1+=4; yy2+=4; yy3+=4;

        s0 = _mm256_fmadd_pd(v0,y0,s0);
        s1 = _mm256_fmadd_pd(v0,y1,s1);
        s2 = _mm256_fmadd_pd(v0,y2,s2);
        s3 = _mm256_fmadd_pd(v0,y3,s3);

        y4 = _mm256_loadu_pd(&yy4[0]);
        y5 = _mm256_loadu_pd(&yy5[0]);
        y6 = _mm256_loadu_pd(&yy6[0]);
        y7 = _mm256_loadu_pd(&yy7[0]);
        yy4+=4; yy5+=4; yy6+=4; yy7+=4;

        s4 = _mm256_fmadd_pd(v0,y4,s4);
        s5 = _mm256_fmadd_pd(v0,y5,s5);
        s6 = _mm256_fmadd_pd(v0,y6,s6);
        s7 = _mm256_fmadd_pd(v0,y7,s7);

        y8 = _mm256_loadu_pd(&yy8[0]);
        y9 = _mm256_loadu_pd(&yy9[0]);
        y10 = _mm256_loadu_pd(&yy10[0]);
        y11 = _mm256_loadu_pd(&yy11[0]);
        yy8+=4; yy9+=4; yy10+=4; yy11+=4;

        s8 = _mm256_fmadd_pd(v0,y8,s8);
        s9 = _mm256_fmadd_pd(v0,y9,s9);
        s10 = _mm256_fmadd_pd(v0,y10,s10);
        s11 = _mm256_fmadd_pd(v0,y11,s11);
        j -= 4;
      } /* End while loop on j (vector length) */

           _MM256_TRANSPOSE4_PD(s0, s1, s2, s3);
      t0 = _mm256_add_pd(s0,s1);
      t1 = _mm256_add_pd(s2,s3);
      s0 = _mm256_add_pd(t0,t1);
           _mm256_store_pd(&z[0], s0);

           _MM256_TRANSPOSE4_PD(s4, s5, s6, s7);
      t2 = _mm256_add_pd(s4,s5);
      t3 = _mm256_add_pd(s6,s7);
      s4 = _mm256_add_pd(t2,t3);
           _mm256_store_pd(&z[4], s4);

           _MM256_TRANSPOSE4_PD(s8, s9, s10, s11);
      t4 = _mm256_add_pd(s8,s9);
      t5 = _mm256_add_pd(s10,s11);
      s8 = _mm256_add_pd(t4,t5);
           _mm256_store_pd(&z[8], s8);

      z   += 12;
      i   -= 12;
      ierr = VecRestoreArrayRead(yy[0 ],&yy0 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[1 ],&yy1 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[2 ],&yy2 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[3 ],&yy3 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[4 ],&yy4 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[5 ],&yy5 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[6 ],&yy6 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[7 ],&yy7 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[8 ],&yy8 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[9 ],&yy9 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[10],&yy10);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[11],&yy11);CHKERRQ(ierr);
      yy  += 12;
      sum0 = 0.; sum1 = 0.; sum2  = 0.; sum3  = 0.;
      sum4 = 0.; sum5 = 0.; sum6  = 0.; sum7  = 0.;
      sum8 = 0.; sum9 = 0.; sum10 = 0.; sum11 = 0.;
      ierr = VecGetArrayRead(yy[0 ],&yy0);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[1 ],&yy1);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[2 ],&yy2);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[3 ],&yy3);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[4 ],&yy4);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[5 ],&yy5);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[6 ],&yy6);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[7 ],&yy7);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[8 ],&yy8);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[9 ],&yy9);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[10],&yy10);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[11],&yy11);CHKERRQ(ierr);
      j = n;
      x = xbase;
      switch (j_rem=j&0x3) {
        case 3:
          x2    = x[2];
          sum0 += x2*PetscConj(yy0 [2]); sum1 += x2*PetscConj(yy1 [2]);
          sum2 += x2*PetscConj(yy2 [2]); sum3 += x2*PetscConj(yy3 [2]);
          sum4 += x2*PetscConj(yy4 [2]); sum5 += x2*PetscConj(yy5 [2]);
          sum6 += x2*PetscConj(yy6 [2]); sum7 += x2*PetscConj(yy7 [2]);
          sum8 += x2*PetscConj(yy8 [2]); sum9 += x2*PetscConj(yy9 [2]);
          sum10+= x2*PetscConj(yy10[2]); sum11+= x2*PetscConj(yy11[2]);
        case 2:
          x1    = x[1];
          sum0 += x1*PetscConj(yy0 [1]); sum1 += x1*PetscConj(yy1 [1]);
          sum2 += x1*PetscConj(yy2 [1]); sum3 += x1*PetscConj(yy3 [1]);
          sum4 += x1*PetscConj(yy4 [1]); sum5 += x1*PetscConj(yy5 [1]);
          sum6 += x1*PetscConj(yy6 [1]); sum7 += x1*PetscConj(yy7 [1]);
          sum8 += x1*PetscConj(yy8 [1]); sum9 += x1*PetscConj(yy9 [1]);
          sum10+= x1*PetscConj(yy10[1]); sum11+= x1*PetscConj(yy11[1]);
        case 1:
          x0    = x[0];
          sum0 += x0*PetscConj(yy0 [0]); sum1 += x0*PetscConj(yy1 [0]);
          sum2 += x0*PetscConj(yy2 [0]); sum3 += x0*PetscConj(yy3 [0]);
          sum4 += x0*PetscConj(yy4 [0]); sum5 += x0*PetscConj(yy5 [0]);
          sum6 += x0*PetscConj(yy6 [0]); sum7 += x0*PetscConj(yy7 [0]);
          sum8 += x0*PetscConj(yy8 [0]); sum9 += x0*PetscConj(yy9 [0]);
          sum10+= x0*PetscConj(yy10[0]); sum11+= x0*PetscConj(yy11[0]);
        case 0:
          x   += j_rem;
          yy0 += j_rem;
          yy1 += j_rem;
          yy2 += j_rem;
          yy3 += j_rem;
          j   -= j_rem;
          break;
      }

      s0 = _mm256_set_pd(0., 0., 0., sum0);   s1 = _mm256_set_pd(0., 0., 0., sum1);
      s2 = _mm256_set_pd(0., 0., 0., sum2);   s3 = _mm256_set_pd(0., 0., 0., sum3);
      s4 = _mm256_set_pd(0., 0., 0., sum4);   s5 = _mm256_set_pd(0., 0., 0., sum5);
      s6 = _mm256_set_pd(0., 0., 0., sum6);   s7 = _mm256_set_pd(0., 0., 0., sum7);
      s8 = _mm256_set_pd(0., 0., 0., sum8);   s9 = _mm256_set_pd(0., 0., 0., sum9);
      s10= _mm256_set_pd(0., 0., 0., sum10); s11 = _mm256_set_pd(0., 0., 0., sum11);

      while (j>0) {
        v0 = _mm256_loadu_pd(&x[0]);
        x += 4;
        y0 = _mm256_loadu_pd(&yy0[0]);
        y1 = _mm256_loadu_pd(&yy1[0]);
        y2 = _mm256_loadu_pd(&yy2[0]);
        y3 = _mm256_loadu_pd(&yy3[0]);
        yy0+=4; yy1+=4; yy2+=4; yy3+=4;

        s0 = _mm256_fmadd_pd(v0,y0,s0);
        s1 = _mm256_fmadd_pd(v0,y1,s1);
        s2 = _mm256_fmadd_pd(v0,y2,s2);
        s3 = _mm256_fmadd_pd(v0,y3,s3);

        y4 = _mm256_loadu_pd(&yy4[0]);
        y5 = _mm256_loadu_pd(&yy5[0]);
        y6 = _mm256_loadu_pd(&yy6[0]);
        y7 = _mm256_loadu_pd(&yy7[0]);
        yy4+=4; yy5+=4; yy6+=4; yy7+=4;

        s4 = _mm256_fmadd_pd(v0,y4,s4);
        s5 = _mm256_fmadd_pd(v0,y5,s5);
        s6 = _mm256_fmadd_pd(v0,y6,s6);
        s7 = _mm256_fmadd_pd(v0,y7,s7);

        y8 = _mm256_loadu_pd(&yy8[0]);
        y9 = _mm256_loadu_pd(&yy9[0]);
        y10 = _mm256_loadu_pd(&yy10[0]);
        y11 = _mm256_loadu_pd(&yy11[0]);
        yy8+=4; yy9+=4; yy10+=4; yy11+=4;

        s8 = _mm256_fmadd_pd(v0,y8,s8);
        s9 = _mm256_fmadd_pd(v0,y9,s9);
        s10 = _mm256_fmadd_pd(v0,y10,s10);
        s11 = _mm256_fmadd_pd(v0,y11,s11);
        j -= 4;
      } /* End while loop on j (vector length) */

           _MM256_TRANSPOSE4_PD(s0, s1, s2, s3);
      t0 = _mm256_add_pd(s0,s1);
      t1 = _mm256_add_pd(s2,s3);
      s0 = _mm256_add_pd(t0,t1);
           _mm256_store_pd(&z[0], s0);

           _MM256_TRANSPOSE4_PD(s4, s5, s6, s7);
      t2 = _mm256_add_pd(s4,s5);
      t3 = _mm256_add_pd(s6,s7);
      s4 = _mm256_add_pd(t2,t3);
           _mm256_store_pd(&z[4], s4);

           _MM256_TRANSPOSE4_PD(s8, s9, s10, s11);
      t4 = _mm256_add_pd(s8,s9);
      t5 = _mm256_add_pd(s10,s11);
      s8 = _mm256_add_pd(t4,t5);
           _mm256_store_pd(&z[8], s8);

      z   += 12;
      i   -= 12;
      ierr = VecRestoreArrayRead(yy[0 ],&yy0 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[1 ],&yy1 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[2 ],&yy2 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[3 ],&yy3 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[4 ],&yy4 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[5 ],&yy5 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[6 ],&yy6 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[7 ],&yy7 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[8 ],&yy8 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[9 ],&yy9 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[10],&yy10);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[11],&yy11);CHKERRQ(ierr);
      yy  += 12;
      break;
    case 20:
      sum0 = 0.; sum1 = 0.; sum2  = 0.; sum3  = 0.;
      sum4 = 0.; sum5 = 0.; sum6  = 0.; sum7  = 0.;
      sum8 = 0.; sum9 = 0.; sum10 = 0.; sum11 = 0.;
      ierr = VecGetArrayRead(yy[0 ],&yy0);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[1 ],&yy1);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[2 ],&yy2);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[3 ],&yy3);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[4 ],&yy4);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[5 ],&yy5);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[6 ],&yy6);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[7 ],&yy7);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[8 ],&yy8);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[9 ],&yy9);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[10],&yy10);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[11],&yy11);CHKERRQ(ierr);
      j = n;
      x = xbase;
      switch (j_rem=j&0x3) {
        case 3:
          x2    = x[2];
          sum0 += x2*PetscConj(yy0 [2]); sum1 += x2*PetscConj(yy1 [2]);
          sum2 += x2*PetscConj(yy2 [2]); sum3 += x2*PetscConj(yy3 [2]);
          sum4 += x2*PetscConj(yy4 [2]); sum5 += x2*PetscConj(yy5 [2]);
          sum6 += x2*PetscConj(yy6 [2]); sum7 += x2*PetscConj(yy7 [2]);
          sum8 += x2*PetscConj(yy8 [2]); sum9 += x2*PetscConj(yy9 [2]);
          sum10+= x2*PetscConj(yy10[2]); sum11+= x2*PetscConj(yy11[2]);
        case 2:
          x1    = x[1];
          sum0 += x1*PetscConj(yy0 [1]); sum1 += x1*PetscConj(yy1 [1]);
          sum2 += x1*PetscConj(yy2 [1]); sum3 += x1*PetscConj(yy3 [1]);
          sum4 += x1*PetscConj(yy4 [1]); sum5 += x1*PetscConj(yy5 [1]);
          sum6 += x1*PetscConj(yy6 [1]); sum7 += x1*PetscConj(yy7 [1]);
          sum8 += x1*PetscConj(yy8 [1]); sum9 += x1*PetscConj(yy9 [1]);
          sum10+= x1*PetscConj(yy10[1]); sum11+= x1*PetscConj(yy11[1]);
        case 1:
          x0    = x[0];
          sum0 += x0*PetscConj(yy0 [0]); sum1 += x0*PetscConj(yy1 [0]);
          sum2 += x0*PetscConj(yy2 [0]); sum3 += x0*PetscConj(yy3 [0]);
          sum4 += x0*PetscConj(yy4 [0]); sum5 += x0*PetscConj(yy5 [0]);
          sum6 += x0*PetscConj(yy6 [0]); sum7 += x0*PetscConj(yy7 [0]);
          sum8 += x0*PetscConj(yy8 [0]); sum9 += x0*PetscConj(yy9 [0]);
          sum10+= x0*PetscConj(yy10[0]); sum11+= x0*PetscConj(yy11[0]);
        case 0:
          x   += j_rem;
          yy0 += j_rem;
          yy1 += j_rem;
          yy2 += j_rem;
          yy3 += j_rem;
          j   -= j_rem;
          break;
      }

      s0 = _mm256_set_pd(0., 0., 0., sum0);   s1 = _mm256_set_pd(0., 0., 0., sum1);
      s2 = _mm256_set_pd(0., 0., 0., sum2);   s3 = _mm256_set_pd(0., 0., 0., sum3);
      s4 = _mm256_set_pd(0., 0., 0., sum4);   s5 = _mm256_set_pd(0., 0., 0., sum5);
      s6 = _mm256_set_pd(0., 0., 0., sum6);   s7 = _mm256_set_pd(0., 0., 0., sum7);
      s8 = _mm256_set_pd(0., 0., 0., sum8);   s9 = _mm256_set_pd(0., 0., 0., sum9);
      s10= _mm256_set_pd(0., 0., 0., sum10); s11 = _mm256_set_pd(0., 0., 0., sum11);

      while (j>0) {
        v0 = _mm256_loadu_pd(&x[0]);
        x += 4;
        y0 = _mm256_loadu_pd(&yy0[0]);
        y1 = _mm256_loadu_pd(&yy1[0]);
        y2 = _mm256_loadu_pd(&yy2[0]);
        y3 = _mm256_loadu_pd(&yy3[0]);
        yy0+=4; yy1+=4; yy2+=4; yy3+=4;

        s0 = _mm256_fmadd_pd(v0,y0,s0);
        s1 = _mm256_fmadd_pd(v0,y1,s1);
        s2 = _mm256_fmadd_pd(v0,y2,s2);
        s3 = _mm256_fmadd_pd(v0,y3,s3);

        y4 = _mm256_loadu_pd(&yy4[0]);
        y5 = _mm256_loadu_pd(&yy5[0]);
        y6 = _mm256_loadu_pd(&yy6[0]);
        y7 = _mm256_loadu_pd(&yy7[0]);
        yy4+=4; yy5+=4; yy6+=4; yy7+=4;

        s4 = _mm256_fmadd_pd(v0,y4,s4);
        s5 = _mm256_fmadd_pd(v0,y5,s5);
        s6 = _mm256_fmadd_pd(v0,y6,s6);
        s7 = _mm256_fmadd_pd(v0,y7,s7);

        y8 = _mm256_loadu_pd(&yy8[0]);
        y9 = _mm256_loadu_pd(&yy9[0]);
        y10 = _mm256_loadu_pd(&yy10[0]);
        y11 = _mm256_loadu_pd(&yy11[0]);
        yy8+=4; yy9+=4; yy10+=4; yy11+=4;

        s8 = _mm256_fmadd_pd(v0,y8,s8);
        s9 = _mm256_fmadd_pd(v0,y9,s9);
        s10 = _mm256_fmadd_pd(v0,y10,s10);
        s11 = _mm256_fmadd_pd(v0,y11,s11);
        j -= 4;
      } /* End while loop on j (vector length) */

           _MM256_TRANSPOSE4_PD(s0, s1, s2, s3);
      t0 = _mm256_add_pd(s0,s1);
      t1 = _mm256_add_pd(s2,s3);
      s0 = _mm256_add_pd(t0,t1);
           _mm256_store_pd(&z[0], s0);

           _MM256_TRANSPOSE4_PD(s4, s5, s6, s7);
      t2 = _mm256_add_pd(s4,s5);
      t3 = _mm256_add_pd(s6,s7);
      s4 = _mm256_add_pd(t2,t3);
           _mm256_store_pd(&z[4], s4);

           _MM256_TRANSPOSE4_PD(s8, s9, s10, s11);
      t4 = _mm256_add_pd(s8,s9);
      t5 = _mm256_add_pd(s10,s11);
      s8 = _mm256_add_pd(t4,t5);
           _mm256_store_pd(&z[8], s8);

      z   += 12;
      i   -= 12;
      ierr = VecRestoreArrayRead(yy[0 ],&yy0 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[1 ],&yy1 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[2 ],&yy2 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[3 ],&yy3 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[4 ],&yy4 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[5 ],&yy5 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[6 ],&yy6 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[7 ],&yy7 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[8 ],&yy8 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[9 ],&yy9 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[10],&yy10);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[11],&yy11);CHKERRQ(ierr);
      yy  += 12;

      sum0 = 0.; sum1 = 0.; sum2 = 0.; sum3 = 0.;
      sum4 = 0.; sum5 = 0.; sum6 = 0.; sum7 = 0.;
      ierr = VecGetArrayRead(yy[0],&yy0);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[1],&yy1);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[2],&yy2);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[3],&yy3);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[4],&yy4);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[5],&yy5);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[6],&yy6);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[7],&yy7);CHKERRQ(ierr);
      j = n;
      x = xbase;
      switch (j_rem=j&0x3) {
        case 3:
          x2    = x[2];
          sum0 += x2*PetscConj(yy0[2]); sum1 += x2*PetscConj(yy1[2]);
          sum2 += x2*PetscConj(yy2[2]); sum3 += x2*PetscConj(yy3[2]);
          sum4 += x2*PetscConj(yy4[2]); sum5 += x2*PetscConj(yy5[2]);
          sum6 += x2*PetscConj(yy6[2]); sum7 += x2*PetscConj(yy7[2]);
        case 2:
          x1    = x[1];
          sum0 += x1*PetscConj(yy0[1]); sum1 += x1*PetscConj(yy1[1]);
          sum2 += x1*PetscConj(yy2[1]); sum3 += x1*PetscConj(yy3[1]);
          sum4 += x1*PetscConj(yy4[1]); sum5 += x1*PetscConj(yy5[1]);
          sum6 += x1*PetscConj(yy6[1]); sum7 += x1*PetscConj(yy7[1]);
        case 1:
          x0    = x[0];
          sum0 += x0*PetscConj(yy0[0]); sum1 += x0*PetscConj(yy1[0]);
          sum2 += x0*PetscConj(yy2[0]); sum3 += x0*PetscConj(yy3[0]);
          sum4 += x0*PetscConj(yy4[0]); sum5 += x0*PetscConj(yy5[0]);
          sum6 += x0*PetscConj(yy6[0]); sum7 += x0*PetscConj(yy7[0]);
        case 0:
          x   += j_rem;
          yy0 += j_rem;
          yy1 += j_rem;
          yy2 += j_rem;
          yy3 += j_rem;
          j   -= j_rem;
          break;
      }

      s0 = _mm256_set_pd(0., 0., 0., sum0); s1 = _mm256_set_pd(0., 0., 0., sum1);
      s2 = _mm256_set_pd(0., 0., 0., sum2); s3 = _mm256_set_pd(0., 0., 0., sum3);
      s4 = _mm256_set_pd(0., 0., 0., sum4); s5 = _mm256_set_pd(0., 0., 0., sum5);
      s6 = _mm256_set_pd(0., 0., 0., sum6); s7 = _mm256_set_pd(0., 0., 0., sum7);

      while (j>0) {
        v0 = _mm256_loadu_pd(&x[0]);
        x += 4;
        y0 = _mm256_loadu_pd(&yy0[0]);
        y1 = _mm256_loadu_pd(&yy1[0]);
        y2 = _mm256_loadu_pd(&yy2[0]);
        y3 = _mm256_loadu_pd(&yy3[0]);
        yy0+=4; yy1+=4; yy2+=4; yy3+=4;

        y4 = _mm256_loadu_pd(&yy4[0]);
        y5 = _mm256_loadu_pd(&yy5[0]);
        y6 = _mm256_loadu_pd(&yy6[0]);
        y7 = _mm256_loadu_pd(&yy7[0]);
        yy4+=4; yy5+=4; yy6+=4; yy7+=4;

        s0 = _mm256_fmadd_pd(v0,y0,s0);
        s1 = _mm256_fmadd_pd(v0,y1,s1);
        s2 = _mm256_fmadd_pd(v0,y2,s2);
        s3 = _mm256_fmadd_pd(v0,y3,s3);

        s4 = _mm256_fmadd_pd(v0,y4,s4);
        s5 = _mm256_fmadd_pd(v0,y5,s5);
        s6 = _mm256_fmadd_pd(v0,y6,s6);
        s7 = _mm256_fmadd_pd(v0,y7,s7);
        j -= 4;
      } /* End while loop on j (vector length) */

           _MM256_TRANSPOSE4_PD(s0, s1, s2, s3);
      t0 = _mm256_add_pd(s0,s1);
      t1 = _mm256_add_pd(s2,s3);
      s0 = _mm256_add_pd(t0,t1);
           _mm256_store_pd(&z[0], s0);

           _MM256_TRANSPOSE4_PD(s4, s5, s6, s7);
      t2 = _mm256_add_pd(s4,s5);
      t3 = _mm256_add_pd(s6,s7);
      s4 = _mm256_add_pd(t2,t3);
           _mm256_store_pd(&z[4], s4);

      z   += 8;
      i   -= 8;
      ierr = VecRestoreArrayRead(yy[0],&yy0);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[1],&yy1);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[2],&yy2);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[3],&yy3);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[4],&yy4);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[5],&yy5);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[6],&yy6);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[7],&yy7);CHKERRQ(ierr);
      yy  += 8;
      break;
    case 16:
       sum0 = 0.; sum1 = 0.; sum2  = 0.; sum3  = 0.;
       sum4 = 0.; sum5 = 0.; sum6  = 0.; sum7  = 0.;
       sum8 = 0.; sum9 = 0.; sum10 = 0.; sum11 = 0.;
       sum12= 0.; sum13= 0.; sum14 = 0.; sum15 = 0.;

       ierr = VecGetArrayRead(yy[0 ],&yy0);CHKERRQ(ierr);
       ierr = VecGetArrayRead(yy[1 ],&yy1);CHKERRQ(ierr);
       ierr = VecGetArrayRead(yy[2 ],&yy2);CHKERRQ(ierr);
       ierr = VecGetArrayRead(yy[3 ],&yy3);CHKERRQ(ierr);
       ierr = VecGetArrayRead(yy[4 ],&yy4);CHKERRQ(ierr);
       ierr = VecGetArrayRead(yy[5 ],&yy5);CHKERRQ(ierr);
       ierr = VecGetArrayRead(yy[6 ],&yy6);CHKERRQ(ierr);
       ierr = VecGetArrayRead(yy[7 ],&yy7);CHKERRQ(ierr);
       ierr = VecGetArrayRead(yy[8 ],&yy8);CHKERRQ(ierr);
       ierr = VecGetArrayRead(yy[9 ],&yy9);CHKERRQ(ierr);
       ierr = VecGetArrayRead(yy[10],&yy10);CHKERRQ(ierr);
       ierr = VecGetArrayRead(yy[11],&yy11);CHKERRQ(ierr);
       ierr = VecGetArrayRead(yy[12],&yy12);CHKERRQ(ierr);
       ierr = VecGetArrayRead(yy[13],&yy13);CHKERRQ(ierr);
       ierr = VecGetArrayRead(yy[14],&yy14);CHKERRQ(ierr);
       ierr = VecGetArrayRead(yy[15],&yy15);CHKERRQ(ierr);
       j = n;
       x = xbase;
       switch (j_rem=j&0x3) {
         case 3:
           x2    = x[2];
           sum0 += x2*PetscConj(yy0 [2]); sum1 += x2*PetscConj(yy1 [2]);
           sum2 += x2*PetscConj(yy2 [2]); sum3 += x2*PetscConj(yy3 [2]);
           sum4 += x2*PetscConj(yy4 [2]); sum5 += x2*PetscConj(yy5 [2]);
           sum6 += x2*PetscConj(yy6 [2]); sum7 += x2*PetscConj(yy7 [2]);
           sum8 += x2*PetscConj(yy8 [2]); sum9 += x2*PetscConj(yy9 [2]);
           sum10+= x2*PetscConj(yy10[2]); sum11+= x2*PetscConj(yy11[2]);
           sum12+= x2*PetscConj(yy12[2]); sum13+= x2*PetscConj(yy13[2]);
           sum14+= x2*PetscConj(yy14[2]); sum15+= x2*PetscConj(yy15[2]);
         case 2:
           x1    = x[1];
           sum0 += x1*PetscConj(yy0 [1]); sum1 += x1*PetscConj(yy1 [1]);
           sum2 += x1*PetscConj(yy2 [1]); sum3 += x1*PetscConj(yy3 [1]);
           sum4 += x1*PetscConj(yy4 [1]); sum5 += x1*PetscConj(yy5 [1]);
           sum6 += x1*PetscConj(yy6 [1]); sum7 += x1*PetscConj(yy7 [1]);
           sum8 += x1*PetscConj(yy8 [1]); sum9 += x1*PetscConj(yy9 [1]);
           sum10+= x1*PetscConj(yy10[1]); sum11+= x1*PetscConj(yy11[1]);
           sum12+= x1*PetscConj(yy12[1]); sum13+= x1*PetscConj(yy13[1]);
           sum14+= x1*PetscConj(yy14[1]); sum15+= x1*PetscConj(yy15[1]);
         case 1:
           x0    = x[0];
           sum0 += x0*PetscConj(yy0 [0]); sum1 += x0*PetscConj(yy1 [0]);
           sum2 += x0*PetscConj(yy2 [0]); sum3 += x0*PetscConj(yy3 [0]);
           sum4 += x0*PetscConj(yy4 [0]); sum5 += x0*PetscConj(yy5 [0]);
           sum6 += x0*PetscConj(yy6 [0]); sum7 += x0*PetscConj(yy7 [0]);
           sum8 += x0*PetscConj(yy8 [0]); sum9 += x0*PetscConj(yy9 [0]);
           sum10+= x0*PetscConj(yy10[0]); sum11+= x0*PetscConj(yy11[0]);
           sum12+= x0*PetscConj(yy12[0]); sum13+= x0*PetscConj(yy13[0]);
           sum14+= x0*PetscConj(yy14[0]); sum15+= x0*PetscConj(yy15[0]);
         case 0:
           x   += j_rem;
           yy0 += j_rem;
           yy1 += j_rem;
           yy2 += j_rem;
           yy3 += j_rem;
           j   -= j_rem;
           break;
       }

       s0 = _mm256_set_pd(0., 0., 0., sum0);   s1 = _mm256_set_pd(0., 0., 0., sum1);
       s2 = _mm256_set_pd(0., 0., 0., sum2);   s3 = _mm256_set_pd(0., 0., 0., sum3);
       s4 = _mm256_set_pd(0., 0., 0., sum4);   s5 = _mm256_set_pd(0., 0., 0., sum5);
       s6 = _mm256_set_pd(0., 0., 0., sum6);   s7 = _mm256_set_pd(0., 0., 0., sum7);
       s8 = _mm256_set_pd(0., 0., 0., sum8);   s9 = _mm256_set_pd(0., 0., 0., sum9);
       s10= _mm256_set_pd(0., 0., 0., sum10); s11 = _mm256_set_pd(0., 0., 0., sum11);
       s12= _mm256_set_pd(0., 0., 0., sum12); s13 = _mm256_set_pd(0., 0., 0., sum13);
       s14= _mm256_set_pd(0., 0., 0., sum14); s15 = _mm256_set_pd(0., 0., 0., sum15);

       while (j>0) {
         v0 = _mm256_loadu_pd(&x[0]);
         x += 4;
         y0 = _mm256_loadu_pd(&yy0[0]);
         y1 = _mm256_loadu_pd(&yy1[0]);
         y2 = _mm256_loadu_pd(&yy2[0]);
         y3 = _mm256_loadu_pd(&yy3[0]);
         yy0+=4; yy1+=4; yy2+=4; yy3+=4;

         s0 = _mm256_fmadd_pd(v0,y0,s0);
         s1 = _mm256_fmadd_pd(v0,y1,s1);
         s2 = _mm256_fmadd_pd(v0,y2,s2);
         s3 = _mm256_fmadd_pd(v0,y3,s3);

         y4 = _mm256_loadu_pd(&yy4[0]);
         y5 = _mm256_loadu_pd(&yy5[0]);
         y6 = _mm256_loadu_pd(&yy6[0]);
         y7 = _mm256_loadu_pd(&yy7[0]);
         yy4+=4; yy5+=4; yy6+=4; yy7+=4;

         s4 = _mm256_fmadd_pd(v0,y4,s4);
         s5 = _mm256_fmadd_pd(v0,y5,s5);
         s6 = _mm256_fmadd_pd(v0,y6,s6);
         s7 = _mm256_fmadd_pd(v0,y7,s7);

         y8 = _mm256_loadu_pd(&yy8[0]);
         y9 = _mm256_loadu_pd(&yy9[0]);
         y10= _mm256_loadu_pd(&yy10[0]);
         y11= _mm256_loadu_pd(&yy11[0]);
         yy8+=4; yy9+=4; yy10+=4; yy11+=4;

         s8 = _mm256_fmadd_pd(v0,y8,s8);
         s9 = _mm256_fmadd_pd(v0,y9,s9);
         s10= _mm256_fmadd_pd(v0,y10,s10);
         s11= _mm256_fmadd_pd(v0,y11,s11);

         y12 = _mm256_loadu_pd(&yy12[0]);
         y13 = _mm256_loadu_pd(&yy13[0]);
         y14 = _mm256_loadu_pd(&yy14[0]);
         y15 = _mm256_loadu_pd(&yy15[0]);
         yy12+=4; yy13+=4; yy14+=4; yy15+=4;

         s12 = _mm256_fmadd_pd(v0,y12,s12);
         s13 = _mm256_fmadd_pd(v0,y13,s13);
         s14 = _mm256_fmadd_pd(v0,y14,s14);
         s15 = _mm256_fmadd_pd(v0,y15,s15);
         j -= 4;
       } /* End while loop on j (vector length) */

            _MM256_TRANSPOSE4_PD(s0, s1, s2, s3);
       t0 = _mm256_add_pd(s0,s1);
       t1 = _mm256_add_pd(s2,s3);
       s0 = _mm256_add_pd(t0,t1);
            _mm256_store_pd(&z[0], s0);

            _MM256_TRANSPOSE4_PD(s4, s5, s6, s7);
       t2 = _mm256_add_pd(s4,s5);
       t3 = _mm256_add_pd(s6,s7);
       s4 = _mm256_add_pd(t2,t3);
            _mm256_store_pd(&z[4], s4);

            _MM256_TRANSPOSE4_PD(s8, s9, s10, s11);
       t4 = _mm256_add_pd(s8,s9);
       t5 = _mm256_add_pd(s10,s11);
       s8 = _mm256_add_pd(t4,t5);
            _mm256_store_pd(&z[8], s8);

            _MM256_TRANSPOSE4_PD(s12, s13, s14, s15);
       t6 = _mm256_add_pd(s12,s13);
       t7 = _mm256_add_pd(s14,s15);
       s12= _mm256_add_pd(t6,t7);
            _mm256_store_pd(&z[12], s12);

       z   += 16;
       i   -= 16;
       ierr = VecRestoreArrayRead(yy[0 ],&yy0 );CHKERRQ(ierr);
       ierr = VecRestoreArrayRead(yy[1 ],&yy1 );CHKERRQ(ierr);
       ierr = VecRestoreArrayRead(yy[2 ],&yy2 );CHKERRQ(ierr);
       ierr = VecRestoreArrayRead(yy[3 ],&yy3 );CHKERRQ(ierr);
       ierr = VecRestoreArrayRead(yy[4 ],&yy4 );CHKERRQ(ierr);
       ierr = VecRestoreArrayRead(yy[5 ],&yy5 );CHKERRQ(ierr);
       ierr = VecRestoreArrayRead(yy[6 ],&yy6 );CHKERRQ(ierr);
       ierr = VecRestoreArrayRead(yy[7 ],&yy7 );CHKERRQ(ierr);
       ierr = VecRestoreArrayRead(yy[8 ],&yy8 );CHKERRQ(ierr);
       ierr = VecRestoreArrayRead(yy[9 ],&yy9 );CHKERRQ(ierr);
       ierr = VecRestoreArrayRead(yy[10],&yy10);CHKERRQ(ierr);
       ierr = VecRestoreArrayRead(yy[11],&yy11);CHKERRQ(ierr);
       ierr = VecRestoreArrayRead(yy[12],&yy12);CHKERRQ(ierr);
       ierr = VecRestoreArrayRead(yy[13],&yy13);CHKERRQ(ierr);
       ierr = VecRestoreArrayRead(yy[14],&yy14);CHKERRQ(ierr);
       ierr = VecRestoreArrayRead(yy[15],&yy15);CHKERRQ(ierr);
       yy  += 16;
      break;
    case 12:
       sum0 = 0.; sum1 = 0.; sum2  = 0.; sum3  = 0.;
       sum4 = 0.; sum5 = 0.; sum6  = 0.; sum7  = 0.;
       sum8 = 0.; sum9 = 0.; sum10 = 0.; sum11 = 0.;
       ierr = VecGetArrayRead(yy[0 ],&yy0);CHKERRQ(ierr);
       ierr = VecGetArrayRead(yy[1 ],&yy1);CHKERRQ(ierr);
       ierr = VecGetArrayRead(yy[2 ],&yy2);CHKERRQ(ierr);
       ierr = VecGetArrayRead(yy[3 ],&yy3);CHKERRQ(ierr);
       ierr = VecGetArrayRead(yy[4 ],&yy4);CHKERRQ(ierr);
       ierr = VecGetArrayRead(yy[5 ],&yy5);CHKERRQ(ierr);
       ierr = VecGetArrayRead(yy[6 ],&yy6);CHKERRQ(ierr);
       ierr = VecGetArrayRead(yy[7 ],&yy7);CHKERRQ(ierr);
       ierr = VecGetArrayRead(yy[8 ],&yy8);CHKERRQ(ierr);
       ierr = VecGetArrayRead(yy[9 ],&yy9);CHKERRQ(ierr);
       ierr = VecGetArrayRead(yy[10],&yy10);CHKERRQ(ierr);
       ierr = VecGetArrayRead(yy[11],&yy11);CHKERRQ(ierr);
       j = n;
       x = xbase;
       switch (j_rem=j&0x3) {
         case 3:
           x2    = x[2];
           sum0 += x2*PetscConj(yy0 [2]); sum1 += x2*PetscConj(yy1 [2]);
           sum2 += x2*PetscConj(yy2 [2]); sum3 += x2*PetscConj(yy3 [2]);
           sum4 += x2*PetscConj(yy4 [2]); sum5 += x2*PetscConj(yy5 [2]);
           sum6 += x2*PetscConj(yy6 [2]); sum7 += x2*PetscConj(yy7 [2]);
           sum8 += x2*PetscConj(yy8 [2]); sum9 += x2*PetscConj(yy9 [2]);
           sum10+= x2*PetscConj(yy10[2]); sum11+= x2*PetscConj(yy11[2]);
         case 2:
           x1    = x[1];
           sum0 += x1*PetscConj(yy0 [1]); sum1 += x1*PetscConj(yy1 [1]);
           sum2 += x1*PetscConj(yy2 [1]); sum3 += x1*PetscConj(yy3 [1]);
           sum4 += x1*PetscConj(yy4 [1]); sum5 += x1*PetscConj(yy5 [1]);
           sum6 += x1*PetscConj(yy6 [1]); sum7 += x1*PetscConj(yy7 [1]);
           sum8 += x1*PetscConj(yy8 [1]); sum9 += x1*PetscConj(yy9 [1]);
           sum10+= x1*PetscConj(yy10[1]); sum11+= x1*PetscConj(yy11[1]);
         case 1:
           x0    = x[0];
           sum0 += x0*PetscConj(yy0 [0]); sum1 += x0*PetscConj(yy1 [0]);
           sum2 += x0*PetscConj(yy2 [0]); sum3 += x0*PetscConj(yy3 [0]);
           sum4 += x0*PetscConj(yy4 [0]); sum5 += x0*PetscConj(yy5 [0]);
           sum6 += x0*PetscConj(yy6 [0]); sum7 += x0*PetscConj(yy7 [0]);
           sum8 += x0*PetscConj(yy8 [0]); sum9 += x0*PetscConj(yy9 [0]);
           sum10+= x0*PetscConj(yy10[0]); sum11+= x0*PetscConj(yy11[0]);
         case 0:
           x   += j_rem;
           yy0 += j_rem;
           yy1 += j_rem;
           yy2 += j_rem;
           yy3 += j_rem;
           j   -= j_rem;
           break;
       }

       s0 = _mm256_set_pd(0., 0., 0., sum0);   s1 = _mm256_set_pd(0., 0., 0., sum1);
       s2 = _mm256_set_pd(0., 0., 0., sum2);   s3 = _mm256_set_pd(0., 0., 0., sum3);
       s4 = _mm256_set_pd(0., 0., 0., sum4);   s5 = _mm256_set_pd(0., 0., 0., sum5);
       s6 = _mm256_set_pd(0., 0., 0., sum6);   s7 = _mm256_set_pd(0., 0., 0., sum7);
       s8 = _mm256_set_pd(0., 0., 0., sum8);   s9 = _mm256_set_pd(0., 0., 0., sum9);
       s10= _mm256_set_pd(0., 0., 0., sum10); s11 = _mm256_set_pd(0., 0., 0., sum11);

       while (j>0) {
         v0 = _mm256_loadu_pd(&x[0]);
         x += 4;
         y0 = _mm256_loadu_pd(&yy0[0]);
         y1 = _mm256_loadu_pd(&yy1[0]);
         y2 = _mm256_loadu_pd(&yy2[0]);
         y3 = _mm256_loadu_pd(&yy3[0]);
         yy0+=4; yy1+=4; yy2+=4; yy3+=4;

         s0 = _mm256_fmadd_pd(v0,y0,s0);
         s1 = _mm256_fmadd_pd(v0,y1,s1);
         s2 = _mm256_fmadd_pd(v0,y2,s2);
         s3 = _mm256_fmadd_pd(v0,y3,s3);

         y4 = _mm256_loadu_pd(&yy4[0]);
         y5 = _mm256_loadu_pd(&yy5[0]);
         y6 = _mm256_loadu_pd(&yy6[0]);
         y7 = _mm256_loadu_pd(&yy7[0]);
         yy4+=4; yy5+=4; yy6+=4; yy7+=4;

         s4 = _mm256_fmadd_pd(v0,y4,s4);
         s5 = _mm256_fmadd_pd(v0,y5,s5);
         s6 = _mm256_fmadd_pd(v0,y6,s6);
         s7 = _mm256_fmadd_pd(v0,y7,s7);

         y8 = _mm256_loadu_pd(&yy8[0]);
         y9 = _mm256_loadu_pd(&yy9[0]);
         y10= _mm256_loadu_pd(&yy10[0]);
         y11= _mm256_loadu_pd(&yy11[0]);
         yy8+=4; yy9+=4; yy10+=4; yy11+=4;

         s8 = _mm256_fmadd_pd(v0,y8,s8);
         s9 = _mm256_fmadd_pd(v0,y9,s9);
         s10= _mm256_fmadd_pd(v0,y10,s10);
         s11= _mm256_fmadd_pd(v0,y11,s11);
         j -= 4;
       } /* End while loop on j (vector length) */

            _MM256_TRANSPOSE4_PD(s0, s1, s2, s3);
       t0 = _mm256_add_pd(s0,s1);
       t1 = _mm256_add_pd(s2,s3);
       s0 = _mm256_add_pd(t0,t1);
            _mm256_store_pd(&z[0], s0);

            _MM256_TRANSPOSE4_PD(s4, s5, s6, s7);
       t2 = _mm256_add_pd(s4,s5);
       t3 = _mm256_add_pd(s6,s7);
       s4 = _mm256_add_pd(t2,t3);
            _mm256_store_pd(&z[4], s4);

            _MM256_TRANSPOSE4_PD(s8, s9, s10, s11);
       t4 = _mm256_add_pd(s8,s9);
       t5 = _mm256_add_pd(s10,s11);
       s8 = _mm256_add_pd(t4,t5);
            _mm256_store_pd(&z[8], s8);

       z   += 12;
       i   -= 12;
       ierr = VecRestoreArrayRead(yy[0 ],&yy0 );CHKERRQ(ierr);
       ierr = VecRestoreArrayRead(yy[1 ],&yy1 );CHKERRQ(ierr);
       ierr = VecRestoreArrayRead(yy[2 ],&yy2 );CHKERRQ(ierr);
       ierr = VecRestoreArrayRead(yy[3 ],&yy3 );CHKERRQ(ierr);
       ierr = VecRestoreArrayRead(yy[4 ],&yy4 );CHKERRQ(ierr);
       ierr = VecRestoreArrayRead(yy[5 ],&yy5 );CHKERRQ(ierr);
       ierr = VecRestoreArrayRead(yy[6 ],&yy6 );CHKERRQ(ierr);
       ierr = VecRestoreArrayRead(yy[7 ],&yy7 );CHKERRQ(ierr);
       ierr = VecRestoreArrayRead(yy[8 ],&yy8 );CHKERRQ(ierr);
       ierr = VecRestoreArrayRead(yy[9 ],&yy9 );CHKERRQ(ierr);
       ierr = VecRestoreArrayRead(yy[10],&yy10);CHKERRQ(ierr);
       ierr = VecRestoreArrayRead(yy[11],&yy11);CHKERRQ(ierr);
       yy  += 12;
      break;
    case  8:
      sum0 = 0.; sum1 = 0.; sum2 = 0.; sum3 = 0.;
      sum4 = 0.; sum5 = 0.; sum6 = 0.; sum7 = 0.;
      ierr = VecGetArrayRead(yy[0],&yy0);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[1],&yy1);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[2],&yy2);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[3],&yy3);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[4],&yy4);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[5],&yy5);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[6],&yy6);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[7],&yy7);CHKERRQ(ierr);
      j = n;
      x = xbase;
      switch (j_rem=j&0x3) {
        case 3:
          x2    = x[2];
          sum0 += x2*PetscConj(yy0[2]); sum1 += x2*PetscConj(yy1[2]);
          sum2 += x2*PetscConj(yy2[2]); sum3 += x2*PetscConj(yy3[2]);
          sum4 += x2*PetscConj(yy4[2]); sum5 += x2*PetscConj(yy5[2]);
          sum6 += x2*PetscConj(yy6[2]); sum7 += x2*PetscConj(yy7[2]);
        case 2:
          x1    = x[1];
          sum0 += x1*PetscConj(yy0[1]); sum1 += x1*PetscConj(yy1[1]);
          sum2 += x1*PetscConj(yy2[1]); sum3 += x1*PetscConj(yy3[1]);
          sum4 += x1*PetscConj(yy4[1]); sum5 += x1*PetscConj(yy5[1]);
          sum6 += x1*PetscConj(yy6[1]); sum7 += x1*PetscConj(yy7[1]);
        case 1:
          x0    = x[0];
          sum0 += x0*PetscConj(yy0[0]); sum1 += x0*PetscConj(yy1[0]);
          sum2 += x0*PetscConj(yy2[0]); sum3 += x0*PetscConj(yy3[0]);
          sum4 += x0*PetscConj(yy4[0]); sum5 += x0*PetscConj(yy5[0]);
          sum6 += x0*PetscConj(yy6[0]); sum7 += x0*PetscConj(yy7[0]);
        case 0:
          x   += j_rem;
          yy0 += j_rem;
          yy1 += j_rem;
          yy2 += j_rem;
          yy3 += j_rem;
          j   -= j_rem;
          break;
      }

      s0 = _mm256_set_pd(0., 0., 0., sum0); s1 = _mm256_set_pd(0., 0., 0., sum1);
      s2 = _mm256_set_pd(0., 0., 0., sum2); s3 = _mm256_set_pd(0., 0., 0., sum3);
      s4 = _mm256_set_pd(0., 0., 0., sum4); s5 = _mm256_set_pd(0., 0., 0., sum5);
      s6 = _mm256_set_pd(0., 0., 0., sum6); s7 = _mm256_set_pd(0., 0., 0., sum7);

      while (j>0) {
        v0 = _mm256_loadu_pd(&x[0]);
        x += 4;
        y0 = _mm256_loadu_pd(&yy0[0]);
        y1 = _mm256_loadu_pd(&yy1[0]);
        y2 = _mm256_loadu_pd(&yy2[0]);
        y3 = _mm256_loadu_pd(&yy3[0]);
        yy0+=4; yy1+=4; yy2+=4; yy3+=4;

        y4 = _mm256_loadu_pd(&yy4[0]);
        y5 = _mm256_loadu_pd(&yy5[0]);
        y6 = _mm256_loadu_pd(&yy6[0]);
        y7 = _mm256_loadu_pd(&yy7[0]);
        yy4+=4; yy5+=4; yy6+=4; yy7+=4;

        s0 = _mm256_fmadd_pd(v0,y0,s0);
        s1 = _mm256_fmadd_pd(v0,y1,s1);
        s2 = _mm256_fmadd_pd(v0,y2,s2);
        s3 = _mm256_fmadd_pd(v0,y3,s3);

        s4 = _mm256_fmadd_pd(v0,y4,s4);
        s5 = _mm256_fmadd_pd(v0,y5,s5);
        s6 = _mm256_fmadd_pd(v0,y6,s6);
        s7 = _mm256_fmadd_pd(v0,y7,s7);
        j -= 4;
      } /* End while loop on j (vector length) */

           _MM256_TRANSPOSE4_PD(s0, s1, s2, s3);
      t0 = _mm256_add_pd(s0,s1);
      t1 = _mm256_add_pd(s2,s3);
      s0 = _mm256_add_pd(t0,t1);
           _mm256_store_pd(&z[0], s0);

           _MM256_TRANSPOSE4_PD(s4, s5, s6, s7);
      t2 = _mm256_add_pd(s4,s5);
      t3 = _mm256_add_pd(s6,s7);
      s4 = _mm256_add_pd(t2,t3);
           _mm256_store_pd(&z[4], s4);

      z   += 8;
      i   -= 8;
      ierr = VecRestoreArrayRead(yy[0],&yy0);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[1],&yy1);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[2],&yy2);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[3],&yy3);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[4],&yy4);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[5],&yy5);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[6],&yy6);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[7],&yy7);CHKERRQ(ierr);
      yy  += 8;
      break;
    case  4:
      sum0 = 0.; sum1 = 0.; sum2 = 0.; sum3 = 0.;
      ierr = VecGetArrayRead(yy[0],&yy0);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[1],&yy1);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[2],&yy2);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[3],&yy3);CHKERRQ(ierr);
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

      s0 = _mm256_set_pd(0., 0., 0., sum0); s1 = _mm256_set_pd(0., 0., 0., sum1);
      s2 = _mm256_set_pd(0., 0., 0., sum2); s3 = _mm256_set_pd(0., 0., 0., sum3);

      while (j>0) {
        v0 = _mm256_loadu_pd(&x[0]);
        x += 4;
        y0 = _mm256_loadu_pd(&yy0[0]);
        y1 = _mm256_loadu_pd(&yy1[0]);
        y2 = _mm256_loadu_pd(&yy2[0]);
        y3 = _mm256_loadu_pd(&yy3[0]);
        yy0+=4; yy1+=4; yy2+=4; yy3+=4;
        s0 = _mm256_fmadd_pd(v0,y0,s0);
        s1 = _mm256_fmadd_pd(v0,y1,s1);
        s2 = _mm256_fmadd_pd(v0,y2,s2);
        s3 = _mm256_fmadd_pd(v0,y3,s3);
        j -= 4;
      } /* End while loop on j (vector length) */

           _MM256_TRANSPOSE4_PD(s0, s1, s2, s3);
      t0 = _mm256_add_pd(s0,s1);
      t1 = _mm256_add_pd(s2,s3);
      s0 = _mm256_add_pd(t0,t1);
           _mm256_store_pd(&z[0], s0);

      z   += 4;
      i   -= 4;
      ierr = VecRestoreArrayRead(yy[0],&yy0);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[1],&yy1);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[2],&yy2);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[3],&yy3);CHKERRQ(ierr);
      yy  += 4;
      break;
    default:
      while (i >0) {
        sum0 = 0.; sum1 = 0.; sum2 = 0.; sum3 = 0.;
        ierr = VecGetArrayRead(yy[0],&yy0);CHKERRQ(ierr);
        ierr = VecGetArrayRead(yy[1],&yy1);CHKERRQ(ierr);
        ierr = VecGetArrayRead(yy[2],&yy2);CHKERRQ(ierr);
        ierr = VecGetArrayRead(yy[3],&yy3);CHKERRQ(ierr);

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
        /* j is now zero or an integer multiple of 4 */

        s0 = _mm256_set_pd(sum3, sum2, sum1, sum0);

        while (j>0) {
          v0 = _mm256_loadu_pd(&x[0]);
          x += 4;
          y0 = _mm256_loadu_pd(&yy0[0]);
          y1 = _mm256_loadu_pd(&yy1[0]);
          y2 = _mm256_loadu_pd(&yy2[0]);
          y3 = _mm256_loadu_pd(&yy3[0]);
          yy0+=4; yy1+=4; yy2+=4; yy3+=4;
          t0 = _mm256_mul_pd (v0,y0);
          t1 = _mm256_mul_pd (v0,y1);
          t2 = _mm256_mul_pd (v0,y2);
          t3 = _mm256_mul_pd (v0,y3);
               _MM256_TRANSPOSE4_PD(t0, t1, t2, t3);
          t4 = _mm256_add_pd(t0,t1);
          t5 = _mm256_add_pd(t2,t3);
          s0 = _mm256_add_pd(s0,_mm256_add_pd(t4,t5));
          j -= 4;
        } /* End while loop on j (vector length) */

        _mm256_storeu_pd(&z[0], s0);

        z   += 4;
        i   -= 4;
        ierr = VecRestoreArrayRead(yy[0],&yy0);CHKERRQ(ierr);
        ierr = VecRestoreArrayRead(yy[1],&yy1);CHKERRQ(ierr);
        ierr = VecRestoreArrayRead(yy[2],&yy2);CHKERRQ(ierr);
        ierr = VecRestoreArrayRead(yy[3],&yy3);CHKERRQ(ierr);
        yy  += 4;
      } /* End while loop on i */
  } /* End i switch */
  ierr = VecRestoreArrayRead(xin,&xbase);CHKERRQ(ierr);
  ierr = PetscLogFlops(PetscMax(nv*(2.0*xin->map->n-1),0.0));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#else
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
  yy     = (Vec*)yin;
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
    ierr = VecRestoreArrayRead(yy[0],&yy0);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(yy[1],&yy1);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(yy[2],&yy2);CHKERRQ(ierr);
    break;
  case 2:
    ierr = VecGetArrayRead(yy[0],&yy0);CHKERRQ(ierr);
    ierr = VecGetArrayRead(yy[1],&yy1);CHKERRQ(ierr);
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
      yy0  +=4;
      j    -= 4; x+=4;
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
PetscErrorCode VecMTDot_Seq(Vec xin,PetscInt nv,const Vec yin[],PetscScalar *z)
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
  yy     = (Vec*)yin;
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
      ierr = VecRestoreArrayRead(yy[0],&yy0);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[1],&yy1);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(yy[2],&yy2);CHKERRQ(ierr);
      break;
    case 2:
      ierr = VecGetArrayRead(yy[0],&yy0);CHKERRQ(ierr);
      ierr = VecGetArrayRead(yy[1],&yy1);CHKERRQ(ierr);
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


PetscErrorCode VecMax_Seq(Vec xin,PetscInt *idx,PetscReal *z)
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
    max = PetscRealPart(*xx++); j = 0;
    for (i=1; i<n; i++) {
      if ((tmp = PetscRealPart(*xx++)) > max) { j = i; max = tmp;}
    }
  }
  *z = max;
  if (idx) *idx = j;
  ierr = VecRestoreArrayRead(xin,&xx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecMin_Seq(Vec xin,PetscInt *idx,PetscReal *z)
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
    min = PetscRealPart(*xx++); j = 0;
    for (i=1; i<n; i++) {
      if ((tmp = PetscRealPart(*xx++)) < min) { j = i; min = tmp;}
    }
  }
  *z = min;
  if (idx) *idx = j;
  ierr = VecRestoreArrayRead(xin,&xx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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

PetscErrorCode VecMAXPY_Seq(Vec xin, PetscInt nv,const PetscScalar *alpha,Vec *y)
{
  PetscErrorCode    ierr;
  PetscInt          n = xin->map->n,j,j_rem;
  const PetscScalar *yy0,*yy1,*yy2,*yy3,*yy4,*yy5,*yy6,*yy7,*yy8,*yy9,*yy10,*yy11,*yy12,*yy13,*yy14,*yy15;
  PetscScalar       *xx,alpha0,alpha1,alpha2,alpha3,alpha4,alpha5,alpha6,alpha7,alpha8,alpha9,alpha10,alpha11,alpha12,alpha13,alpha14,alpha15;

#if defined(PETSC_HAVE_PRAGMA_DISJOINT)
#pragma disjoint(*xx,*yy0,*yy1,*yy2,*yy3,*alpha)
#endif

  PetscFunctionBegin;
  ierr = PetscLogFlops(nv*2.0*n);CHKERRQ(ierr);
  ierr = VecGetArray(xin,&xx);CHKERRQ(ierr);
  switch (j_rem=nv&0x3) {
    case 3:
      ierr   = VecGetArrayRead(y[0],&yy0);CHKERRQ(ierr);
      ierr   = VecGetArrayRead(y[1],&yy1);CHKERRQ(ierr);
      ierr   = VecGetArrayRead(y[2],&yy2);CHKERRQ(ierr);
      alpha0 = alpha[0];
      alpha1 = alpha[1];
      alpha2 = alpha[2];
      alpha += 3;
      PetscKernelAXPY3(xx,alpha0,alpha1,alpha2,yy0,yy1,yy2,n);
      ierr = VecRestoreArrayRead(y[0],&yy0);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(y[1],&yy1);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(y[2],&yy2);CHKERRQ(ierr);
      y   += 3;
      break;
    case 2:
      ierr   = VecGetArrayRead(y[0],&yy0);CHKERRQ(ierr);
      ierr   = VecGetArrayRead(y[1],&yy1);CHKERRQ(ierr);
      alpha0 = alpha[0];
      alpha1 = alpha[1];
      alpha +=2;
      PetscKernelAXPY2(xx,alpha0,alpha1,yy0,yy1,n);
      ierr = VecRestoreArrayRead(y[0],&yy0);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(y[1],&yy1);CHKERRQ(ierr);
      y   +=2;
      break;
    case 1:
      ierr   = VecGetArrayRead(y[0],&yy0);CHKERRQ(ierr);
      alpha0 = *alpha++;
      PetscKernelAXPY(xx,alpha0,yy0,n);
      ierr = VecRestoreArrayRead(y[0],&yy0);CHKERRQ(ierr);
      y   +=1;
      break;
  }

  /* (nv-j_rem) is now zero or an integer multiple of 4 */
  switch (nv-j_rem) {
    case 32:
      ierr   = VecGetArrayRead(y[ 0],&yy0 );CHKERRQ(ierr); ierr = VecGetArrayRead(y[1 ],&yy1 );CHKERRQ(ierr);
      ierr   = VecGetArrayRead(y[ 2],&yy2 );CHKERRQ(ierr); ierr = VecGetArrayRead(y[3 ],&yy3 );CHKERRQ(ierr);
      ierr   = VecGetArrayRead(y[ 4],&yy4 );CHKERRQ(ierr); ierr = VecGetArrayRead(y[5 ],&yy5 );CHKERRQ(ierr);
      ierr   = VecGetArrayRead(y[ 6],&yy6 );CHKERRQ(ierr); ierr = VecGetArrayRead(y[7 ],&yy7 );CHKERRQ(ierr);
      ierr   = VecGetArrayRead(y[ 8],&yy8 );CHKERRQ(ierr); ierr = VecGetArrayRead(y[9 ],&yy9 );CHKERRQ(ierr);
      ierr   = VecGetArrayRead(y[10],&yy10);CHKERRQ(ierr); ierr = VecGetArrayRead(y[11],&yy11);CHKERRQ(ierr);
      ierr   = VecGetArrayRead(y[12],&yy12);CHKERRQ(ierr); ierr = VecGetArrayRead(y[13],&yy13);CHKERRQ(ierr);
      ierr   = VecGetArrayRead(y[14],&yy14);CHKERRQ(ierr); ierr = VecGetArrayRead(y[15],&yy15);CHKERRQ(ierr);

      alpha0 = alpha[ 0]; alpha1  = alpha[ 1]; alpha2 = alpha[ 2]; alpha3 = alpha[ 3];
      alpha4 = alpha[ 4]; alpha5  = alpha[ 5]; alpha6 = alpha[ 6]; alpha7 = alpha[ 7];
      alpha8 = alpha[ 8]; alpha9  = alpha[ 9]; alpha10= alpha[10]; alpha11= alpha[11];
      alpha12= alpha[12]; alpha13 = alpha[13]; alpha14= alpha[14]; alpha15= alpha[15];
      alpha += 16;

#if defined(__INTEL_COMPILER)
#pragma ivdep
#pragma unroll(2)
#endif
      {PetscInt __i;
        for (__i=0; __i<n; __i++) {
          xx[__i] += alpha0 *yy0 [__i] + alpha1 *yy1 [__i] + alpha2 *yy2 [__i] + alpha3 *yy3 [__i]
            +  alpha4 *yy4 [__i] + alpha5 *yy5 [__i] + alpha6 *yy6 [__i] + alpha7 *yy7 [__i]
            +  alpha8 *yy8 [__i] + alpha9 *yy9 [__i] + alpha10*yy10[__i] + alpha11*yy11[__i]
            +  alpha12*yy12[__i] + alpha13*yy13[__i] + alpha14*yy14[__i] + alpha15*yy15[__i];
        };}
      ierr = VecRestoreArrayRead(y[ 0],&yy0 );CHKERRQ(ierr); ierr = VecRestoreArrayRead(y[ 1],&yy1 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(y[ 2],&yy2 );CHKERRQ(ierr); ierr = VecRestoreArrayRead(y[ 3],&yy3 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(y[ 4],&yy4 );CHKERRQ(ierr); ierr = VecRestoreArrayRead(y[ 5],&yy5 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(y[ 6],&yy6 );CHKERRQ(ierr); ierr = VecRestoreArrayRead(y[ 7],&yy7 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(y[ 8],&yy8 );CHKERRQ(ierr); ierr = VecRestoreArrayRead(y[ 9],&yy9 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(y[10],&yy10);CHKERRQ(ierr); ierr = VecRestoreArrayRead(y[11],&yy11);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(y[12],&yy12);CHKERRQ(ierr); ierr = VecRestoreArrayRead(y[13],&yy13);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(y[14],&yy14);CHKERRQ(ierr); ierr = VecRestoreArrayRead(y[15],&yy15);CHKERRQ(ierr);
      y   += 16;

      ierr   = VecGetArrayRead(y[ 0],&yy0 );CHKERRQ(ierr); ierr = VecGetArrayRead(y[1 ],&yy1 );CHKERRQ(ierr);
      ierr   = VecGetArrayRead(y[ 2],&yy2 );CHKERRQ(ierr); ierr = VecGetArrayRead(y[3 ],&yy3 );CHKERRQ(ierr);
      ierr   = VecGetArrayRead(y[ 4],&yy4 );CHKERRQ(ierr); ierr = VecGetArrayRead(y[5 ],&yy5 );CHKERRQ(ierr);
      ierr   = VecGetArrayRead(y[ 6],&yy6 );CHKERRQ(ierr); ierr = VecGetArrayRead(y[7 ],&yy7 );CHKERRQ(ierr);
      ierr   = VecGetArrayRead(y[ 8],&yy8 );CHKERRQ(ierr); ierr = VecGetArrayRead(y[9 ],&yy9 );CHKERRQ(ierr);
      ierr   = VecGetArrayRead(y[10],&yy10);CHKERRQ(ierr); ierr = VecGetArrayRead(y[11],&yy11);CHKERRQ(ierr);
      ierr   = VecGetArrayRead(y[12],&yy12);CHKERRQ(ierr); ierr = VecGetArrayRead(y[13],&yy13);CHKERRQ(ierr);
      ierr   = VecGetArrayRead(y[14],&yy14);CHKERRQ(ierr); ierr = VecGetArrayRead(y[15],&yy15);CHKERRQ(ierr);

      alpha0 = alpha[ 0]; alpha1  = alpha[ 1]; alpha2 = alpha[ 2]; alpha3 = alpha[ 3];
      alpha4 = alpha[ 4]; alpha5  = alpha[ 5]; alpha6 = alpha[ 6]; alpha7 = alpha[ 7];
      alpha8 = alpha[ 8]; alpha9  = alpha[ 9]; alpha10= alpha[10]; alpha11= alpha[11];
      alpha12= alpha[12]; alpha13 = alpha[13]; alpha14= alpha[14]; alpha15= alpha[15];
      alpha += 16;

#if defined(__INTEL_COMPILER)
#pragma ivdep
#pragma unroll(2)
#endif
      {PetscInt __i;
        for (__i=0; __i<n; __i++) {
          xx[__i] += alpha0 *yy0 [__i] + alpha1 *yy1 [__i] + alpha2 *yy2 [__i] + alpha3 *yy3 [__i]
            +  alpha4 *yy4 [__i] + alpha5 *yy5 [__i] + alpha6 *yy6 [__i] + alpha7 *yy7 [__i]
            +  alpha8 *yy8 [__i] + alpha9 *yy9 [__i] + alpha10*yy10[__i] + alpha11*yy11[__i]
            +  alpha12*yy12[__i] + alpha13*yy13[__i] + alpha14*yy14[__i] + alpha15*yy15[__i];
        };}
      ierr = VecRestoreArrayRead(y[ 0],&yy0 );CHKERRQ(ierr); ierr = VecRestoreArrayRead(y[ 1],&yy1 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(y[ 2],&yy2 );CHKERRQ(ierr); ierr = VecRestoreArrayRead(y[ 3],&yy3 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(y[ 4],&yy4 );CHKERRQ(ierr); ierr = VecRestoreArrayRead(y[ 5],&yy5 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(y[ 6],&yy6 );CHKERRQ(ierr); ierr = VecRestoreArrayRead(y[ 7],&yy7 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(y[ 8],&yy8 );CHKERRQ(ierr); ierr = VecRestoreArrayRead(y[ 9],&yy9 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(y[10],&yy10);CHKERRQ(ierr); ierr = VecRestoreArrayRead(y[11],&yy11);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(y[12],&yy12);CHKERRQ(ierr); ierr = VecRestoreArrayRead(y[13],&yy13);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(y[14],&yy14);CHKERRQ(ierr); ierr = VecRestoreArrayRead(y[15],&yy15);CHKERRQ(ierr);
      y   += 16;
      break;

    case 28:
      ierr   = VecGetArrayRead(y[ 0],&yy0 );CHKERRQ(ierr); ierr = VecGetArrayRead(y[1 ],&yy1 );CHKERRQ(ierr);
      ierr   = VecGetArrayRead(y[ 2],&yy2 );CHKERRQ(ierr); ierr = VecGetArrayRead(y[3 ],&yy3 );CHKERRQ(ierr);
      ierr   = VecGetArrayRead(y[ 4],&yy4 );CHKERRQ(ierr); ierr = VecGetArrayRead(y[5 ],&yy5 );CHKERRQ(ierr);
      ierr   = VecGetArrayRead(y[ 6],&yy6 );CHKERRQ(ierr); ierr = VecGetArrayRead(y[7 ],&yy7 );CHKERRQ(ierr);
      ierr   = VecGetArrayRead(y[ 8],&yy8 );CHKERRQ(ierr); ierr = VecGetArrayRead(y[9 ],&yy9 );CHKERRQ(ierr);
      ierr   = VecGetArrayRead(y[10],&yy10);CHKERRQ(ierr); ierr = VecGetArrayRead(y[11],&yy11);CHKERRQ(ierr);
      ierr   = VecGetArrayRead(y[12],&yy12);CHKERRQ(ierr); ierr = VecGetArrayRead(y[13],&yy13);CHKERRQ(ierr);
      ierr   = VecGetArrayRead(y[14],&yy14);CHKERRQ(ierr); ierr = VecGetArrayRead(y[15],&yy15);CHKERRQ(ierr);

      alpha0 = alpha[ 0]; alpha1  = alpha[ 1]; alpha2 = alpha[ 2]; alpha3 = alpha[ 3];
      alpha4 = alpha[ 4]; alpha5  = alpha[ 5]; alpha6 = alpha[ 6]; alpha7 = alpha[ 7];
      alpha8 = alpha[ 8]; alpha9  = alpha[ 9]; alpha10= alpha[10]; alpha11= alpha[11];
      alpha12= alpha[12]; alpha13 = alpha[13]; alpha14= alpha[14]; alpha15= alpha[15];
      alpha += 16;

#if defined(__INTEL_COMPILER)
#pragma ivdep
#pragma unroll(2)
#endif
      {PetscInt __i;
        for (__i=0; __i<n; __i++) {
          xx[__i] += alpha0 *yy0 [__i] + alpha1 *yy1 [__i] + alpha2 *yy2 [__i] + alpha3 *yy3 [__i]
            +  alpha4 *yy4 [__i] + alpha5 *yy5 [__i] + alpha6 *yy6 [__i] + alpha7 *yy7 [__i]
            +  alpha8 *yy8 [__i] + alpha9 *yy9 [__i] + alpha10*yy10[__i] + alpha11*yy11[__i]
            +  alpha12*yy12[__i] + alpha13*yy13[__i] + alpha14*yy14[__i] + alpha15*yy15[__i];
        };}
      ierr = VecRestoreArrayRead(y[ 0],&yy0 );CHKERRQ(ierr); ierr = VecRestoreArrayRead(y[ 1],&yy1 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(y[ 2],&yy2 );CHKERRQ(ierr); ierr = VecRestoreArrayRead(y[ 3],&yy3 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(y[ 4],&yy4 );CHKERRQ(ierr); ierr = VecRestoreArrayRead(y[ 5],&yy5 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(y[ 6],&yy6 );CHKERRQ(ierr); ierr = VecRestoreArrayRead(y[ 7],&yy7 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(y[ 8],&yy8 );CHKERRQ(ierr); ierr = VecRestoreArrayRead(y[ 9],&yy9 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(y[10],&yy10);CHKERRQ(ierr); ierr = VecRestoreArrayRead(y[11],&yy11);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(y[12],&yy12);CHKERRQ(ierr); ierr = VecRestoreArrayRead(y[13],&yy13);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(y[14],&yy14);CHKERRQ(ierr); ierr = VecRestoreArrayRead(y[15],&yy15);CHKERRQ(ierr);
      y   += 16;

      ierr   = VecGetArrayRead(y[ 0],&yy0 );CHKERRQ(ierr); ierr = VecGetArrayRead(y[1 ],&yy1 );CHKERRQ(ierr);
      ierr   = VecGetArrayRead(y[ 2],&yy2 );CHKERRQ(ierr); ierr = VecGetArrayRead(y[3 ],&yy3 );CHKERRQ(ierr);
      ierr   = VecGetArrayRead(y[ 4],&yy4 );CHKERRQ(ierr); ierr = VecGetArrayRead(y[5 ],&yy5 );CHKERRQ(ierr);
      ierr   = VecGetArrayRead(y[ 6],&yy6 );CHKERRQ(ierr); ierr = VecGetArrayRead(y[7 ],&yy7 );CHKERRQ(ierr);
      ierr   = VecGetArrayRead(y[ 8],&yy8 );CHKERRQ(ierr); ierr = VecGetArrayRead(y[9 ],&yy9 );CHKERRQ(ierr);
      ierr   = VecGetArrayRead(y[10],&yy10);CHKERRQ(ierr); ierr = VecGetArrayRead(y[11],&yy11);CHKERRQ(ierr);

      alpha0 = alpha[ 0]; alpha1  = alpha[ 1]; alpha2 = alpha[ 2]; alpha3 = alpha[ 3];
      alpha4 = alpha[ 4]; alpha5  = alpha[ 5]; alpha6 = alpha[ 6]; alpha7 = alpha[ 7];
      alpha8 = alpha[ 8]; alpha9  = alpha[ 9]; alpha10= alpha[10]; alpha11= alpha[11];
      alpha += 12;

#if defined(__INTEL_COMPILER)
#pragma ivdep
#pragma unroll(2)
#endif
      {PetscInt __i;
        for (__i=0; __i<n; __i++) {
          xx[__i] += alpha0 *yy0 [__i] + alpha1 *yy1 [__i] + alpha2 *yy2 [__i] + alpha3 *yy3 [__i]
            +  alpha4 *yy4 [__i] + alpha5 *yy5 [__i] + alpha6 *yy6 [__i] + alpha7 *yy7 [__i]
            +  alpha8 *yy8 [__i] + alpha9 *yy9 [__i] + alpha10*yy10[__i] + alpha11*yy11[__i];
        };}
      ierr = VecRestoreArrayRead(y[ 0],&yy0 );CHKERRQ(ierr); ierr = VecRestoreArrayRead(y[ 1],&yy1 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(y[ 2],&yy2 );CHKERRQ(ierr); ierr = VecRestoreArrayRead(y[ 3],&yy3 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(y[ 4],&yy4 );CHKERRQ(ierr); ierr = VecRestoreArrayRead(y[ 5],&yy5 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(y[ 6],&yy6 );CHKERRQ(ierr); ierr = VecRestoreArrayRead(y[ 7],&yy7 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(y[ 8],&yy8 );CHKERRQ(ierr); ierr = VecRestoreArrayRead(y[ 9],&yy9 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(y[10],&yy10);CHKERRQ(ierr); ierr = VecRestoreArrayRead(y[11],&yy11);CHKERRQ(ierr);
      y   += 12;
      break;

    case 24:
      ierr   = VecGetArrayRead(y[ 0],&yy0 );CHKERRQ(ierr); ierr = VecGetArrayRead(y[1 ],&yy1 );CHKERRQ(ierr);
      ierr   = VecGetArrayRead(y[ 2],&yy2 );CHKERRQ(ierr); ierr = VecGetArrayRead(y[3 ],&yy3 );CHKERRQ(ierr);
      ierr   = VecGetArrayRead(y[ 4],&yy4 );CHKERRQ(ierr); ierr = VecGetArrayRead(y[5 ],&yy5 );CHKERRQ(ierr);
      ierr   = VecGetArrayRead(y[ 6],&yy6 );CHKERRQ(ierr); ierr = VecGetArrayRead(y[7 ],&yy7 );CHKERRQ(ierr);
      ierr   = VecGetArrayRead(y[ 8],&yy8 );CHKERRQ(ierr); ierr = VecGetArrayRead(y[9 ],&yy9 );CHKERRQ(ierr);
      ierr   = VecGetArrayRead(y[10],&yy10);CHKERRQ(ierr); ierr = VecGetArrayRead(y[11],&yy11);CHKERRQ(ierr);

      alpha0 = alpha[0]; alpha1 = alpha[1]; alpha2 = alpha[ 2]; alpha3 = alpha[ 3];
      alpha4 = alpha[4]; alpha5 = alpha[5]; alpha6 = alpha[ 6]; alpha7 = alpha[ 7];
      alpha8 = alpha[8]; alpha9 = alpha[9]; alpha10= alpha[10]; alpha11= alpha[11];
      alpha += 12;

#if defined(__INTEL_COMPILER)
#pragma ivdep
#pragma unroll(2)
#endif
      {PetscInt __i;
        for (__i=0; __i<n; __i++) {
          xx[__i] += alpha0*yy0[__i] + alpha1*yy1[__i] + alpha2 *yy2 [__i] + alpha3 *yy3 [__i]
            +  alpha4*yy4[__i] + alpha5*yy5[__i] + alpha6 *yy6 [__i] + alpha7 *yy7 [__i]
            +  alpha8*yy8[__i] + alpha9*yy9[__i] + alpha10*yy10[__i] + alpha11*yy11[__i];
        };}
      ierr = VecRestoreArrayRead(y[ 0],&yy0 );CHKERRQ(ierr); ierr = VecRestoreArrayRead(y[ 1],&yy1 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(y[ 2],&yy2 );CHKERRQ(ierr); ierr = VecRestoreArrayRead(y[ 3],&yy3 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(y[ 4],&yy4 );CHKERRQ(ierr); ierr = VecRestoreArrayRead(y[ 5],&yy5 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(y[ 6],&yy6 );CHKERRQ(ierr); ierr = VecRestoreArrayRead(y[ 7],&yy7 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(y[ 8],&yy8 );CHKERRQ(ierr); ierr = VecRestoreArrayRead(y[ 9],&yy9 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(y[10],&yy10);CHKERRQ(ierr); ierr = VecRestoreArrayRead(y[11],&yy11);CHKERRQ(ierr);
      y   += 12;

      ierr   = VecGetArrayRead(y[ 0],&yy0 );CHKERRQ(ierr); ierr = VecGetArrayRead(y[1 ],&yy1 );CHKERRQ(ierr);
      ierr   = VecGetArrayRead(y[ 2],&yy2 );CHKERRQ(ierr); ierr = VecGetArrayRead(y[3 ],&yy3 );CHKERRQ(ierr);
      ierr   = VecGetArrayRead(y[ 4],&yy4 );CHKERRQ(ierr); ierr = VecGetArrayRead(y[5 ],&yy5 );CHKERRQ(ierr);
      ierr   = VecGetArrayRead(y[ 6],&yy6 );CHKERRQ(ierr); ierr = VecGetArrayRead(y[7 ],&yy7 );CHKERRQ(ierr);
      ierr   = VecGetArrayRead(y[ 8],&yy8 );CHKERRQ(ierr); ierr = VecGetArrayRead(y[9 ],&yy9 );CHKERRQ(ierr);
      ierr   = VecGetArrayRead(y[10],&yy10);CHKERRQ(ierr); ierr = VecGetArrayRead(y[11],&yy11);CHKERRQ(ierr);

      alpha0 = alpha[0]; alpha1 = alpha[1]; alpha2 = alpha[ 2]; alpha3 = alpha[ 3];
      alpha4 = alpha[4]; alpha5 = alpha[5]; alpha6 = alpha[ 6]; alpha7 = alpha[ 7];
      alpha8 = alpha[8]; alpha9 = alpha[9]; alpha10= alpha[10]; alpha11= alpha[11];
      alpha += 12;

#if defined(__INTEL_COMPILER)
#pragma ivdep
#pragma unroll(2)
#endif
      {PetscInt __i;
        for (__i=0; __i<n; __i++) {
          xx[__i] += alpha0*yy0[__i] + alpha1*yy1[__i] + alpha2 *yy2 [__i] + alpha3 *yy3 [__i]
            +  alpha4*yy4[__i] + alpha5*yy5[__i] + alpha6 *yy6 [__i] + alpha7 *yy7 [__i]
            +  alpha8*yy8[__i] + alpha9*yy9[__i] + alpha10*yy10[__i] + alpha11*yy11[__i];
        };}
      ierr = VecRestoreArrayRead(y[ 0],&yy0 );CHKERRQ(ierr); ierr = VecRestoreArrayRead(y[ 1],&yy1 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(y[ 2],&yy2 );CHKERRQ(ierr); ierr = VecRestoreArrayRead(y[ 3],&yy3 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(y[ 4],&yy4 );CHKERRQ(ierr); ierr = VecRestoreArrayRead(y[ 5],&yy5 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(y[ 6],&yy6 );CHKERRQ(ierr); ierr = VecRestoreArrayRead(y[ 7],&yy7 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(y[ 8],&yy8 );CHKERRQ(ierr); ierr = VecRestoreArrayRead(y[ 9],&yy9 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(y[10],&yy10);CHKERRQ(ierr); ierr = VecRestoreArrayRead(y[11],&yy11);CHKERRQ(ierr);
      y   += 12;
      break;

    case 20:
      ierr   = VecGetArrayRead(y[0 ],&yy0 );CHKERRQ(ierr); ierr = VecGetArrayRead(y[1 ],&yy1 );CHKERRQ(ierr);
      ierr   = VecGetArrayRead(y[2 ],&yy2 );CHKERRQ(ierr); ierr = VecGetArrayRead(y[3 ],&yy3 );CHKERRQ(ierr);
      ierr   = VecGetArrayRead(y[4 ],&yy4 );CHKERRQ(ierr); ierr = VecGetArrayRead(y[5 ],&yy5 );CHKERRQ(ierr);
      ierr   = VecGetArrayRead(y[6 ],&yy6 );CHKERRQ(ierr); ierr = VecGetArrayRead(y[7 ],&yy7 );CHKERRQ(ierr);
      ierr   = VecGetArrayRead(y[8 ],&yy8 );CHKERRQ(ierr); ierr = VecGetArrayRead(y[9 ],&yy9 );CHKERRQ(ierr);
      ierr   = VecGetArrayRead(y[10],&yy10);CHKERRQ(ierr); ierr = VecGetArrayRead(y[11],&yy11);CHKERRQ(ierr);
      ierr   = VecGetArrayRead(y[12],&yy12);CHKERRQ(ierr); ierr = VecGetArrayRead(y[13],&yy13);CHKERRQ(ierr);
      ierr   = VecGetArrayRead(y[14],&yy14);CHKERRQ(ierr); ierr = VecGetArrayRead(y[15],&yy15);CHKERRQ(ierr);

      alpha0 = alpha[ 0]; alpha1  = alpha[ 1]; alpha2 = alpha[ 2]; alpha3 = alpha[ 3];
      alpha4 = alpha[ 4]; alpha5  = alpha[ 5]; alpha6 = alpha[ 6]; alpha7 = alpha[ 7];
      alpha8 = alpha[ 8]; alpha9  = alpha[ 9]; alpha10= alpha[10]; alpha11= alpha[11];
      alpha12= alpha[12]; alpha13 = alpha[13]; alpha14= alpha[14]; alpha15= alpha[15];
      alpha += 16;

#if defined(__INTEL_COMPILER)
#pragma ivdep
#pragma unroll(2)
#endif
      {PetscInt __i;
        for (__i=0; __i<n; __i++) {
          xx[__i] += alpha0 *yy0 [__i] + alpha1 *yy1 [__i] + alpha2 *yy2 [__i] + alpha3 *yy3 [__i]
            +  alpha4 *yy4 [__i] + alpha5 *yy5 [__i] + alpha6 *yy6 [__i] + alpha7 *yy7 [__i]
            +  alpha8 *yy8 [__i] + alpha9 *yy9 [__i] + alpha10*yy10[__i] + alpha11*yy11[__i]
            +  alpha12*yy12[__i] + alpha13*yy13[__i] + alpha14*yy14[__i] + alpha15*yy15[__i];
        };}
      ierr = VecRestoreArrayRead(y[ 0],&yy0 );CHKERRQ(ierr); ierr = VecRestoreArrayRead(y[ 1],&yy1 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(y[ 2],&yy2 );CHKERRQ(ierr); ierr = VecRestoreArrayRead(y[ 3],&yy3 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(y[ 4],&yy4 );CHKERRQ(ierr); ierr = VecRestoreArrayRead(y[ 5],&yy5 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(y[ 6],&yy6 );CHKERRQ(ierr); ierr = VecRestoreArrayRead(y[ 7],&yy7 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(y[ 8],&yy8 );CHKERRQ(ierr); ierr = VecRestoreArrayRead(y[ 9],&yy9 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(y[10],&yy10);CHKERRQ(ierr); ierr = VecRestoreArrayRead(y[11],&yy11);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(y[12],&yy12);CHKERRQ(ierr); ierr = VecRestoreArrayRead(y[13],&yy13);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(y[14],&yy14);CHKERRQ(ierr); ierr = VecRestoreArrayRead(y[15],&yy15);CHKERRQ(ierr);
      y   += 16;

      ierr   = VecGetArrayRead(y[0],&yy0);CHKERRQ(ierr);
      ierr   = VecGetArrayRead(y[1],&yy1);CHKERRQ(ierr);
      ierr   = VecGetArrayRead(y[2],&yy2);CHKERRQ(ierr);
      ierr   = VecGetArrayRead(y[3],&yy3);CHKERRQ(ierr);

      alpha0 = alpha[0]; alpha1 = alpha[1]; alpha2 = alpha[2]; alpha3 = alpha[3];
      alpha += 4;

#if defined(__INTEL_COMPILER)
#pragma ivdep
#pragma unroll(8)
#endif
      {PetscInt __i;
        for (__i=0; __i<n; __i++) {
          xx[__i] += alpha0 *yy0 [__i] + alpha1 *yy1 [__i] + alpha2 *yy2 [__i] + alpha3 *yy3 [__i];
        };}
      ierr = VecRestoreArrayRead(y[0],&yy0);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(y[1],&yy1);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(y[2],&yy2);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(y[3],&yy3);CHKERRQ(ierr);
      y   += 4;
      break;

    case 16:
      ierr   = VecGetArrayRead(y[ 0],&yy0 );CHKERRQ(ierr); ierr = VecGetArrayRead(y[1 ],&yy1 );CHKERRQ(ierr);
      ierr   = VecGetArrayRead(y[ 2],&yy2 );CHKERRQ(ierr); ierr = VecGetArrayRead(y[3 ],&yy3 );CHKERRQ(ierr);
      ierr   = VecGetArrayRead(y[ 4],&yy4 );CHKERRQ(ierr); ierr = VecGetArrayRead(y[5 ],&yy5 );CHKERRQ(ierr);
      ierr   = VecGetArrayRead(y[ 6],&yy6 );CHKERRQ(ierr); ierr = VecGetArrayRead(y[7 ],&yy7 );CHKERRQ(ierr);
      ierr   = VecGetArrayRead(y[ 8],&yy8 );CHKERRQ(ierr); ierr = VecGetArrayRead(y[9 ],&yy9 );CHKERRQ(ierr);
      ierr   = VecGetArrayRead(y[10],&yy10);CHKERRQ(ierr); ierr = VecGetArrayRead(y[11],&yy11);CHKERRQ(ierr);
      ierr   = VecGetArrayRead(y[12],&yy12);CHKERRQ(ierr); ierr = VecGetArrayRead(y[13],&yy13);CHKERRQ(ierr);
      ierr   = VecGetArrayRead(y[14],&yy14);CHKERRQ(ierr); ierr = VecGetArrayRead(y[15],&yy15);CHKERRQ(ierr);

      alpha0 = alpha[ 0]; alpha1  = alpha[ 1]; alpha2 = alpha[ 2]; alpha3 = alpha[ 3];
      alpha4 = alpha[ 4]; alpha5  = alpha[ 5]; alpha6 = alpha[ 6]; alpha7 = alpha[ 7];
      alpha8 = alpha[ 8]; alpha9  = alpha[ 9]; alpha10= alpha[10]; alpha11= alpha[11];
      alpha12= alpha[12]; alpha13 = alpha[13]; alpha14= alpha[14]; alpha15= alpha[15];
      alpha += 16;

#if defined(__INTEL_COMPILER)
#pragma ivdep
#pragma unroll(2)
#pragma vector unaligned
      __assume(n%4==0);
#endif
      {PetscInt __i;
        for (__i=0; __i<n; __i++) {
          xx[__i] += alpha0 *yy0 [__i] + alpha1 *yy1 [__i] + alpha2 *yy2 [__i] + alpha3 *yy3 [__i]
            +  alpha4 *yy4 [__i] + alpha5 *yy5 [__i] + alpha6 *yy6 [__i] + alpha7 *yy7 [__i]
            +  alpha8 *yy8 [__i] + alpha9 *yy9 [__i] + alpha10*yy10[__i] + alpha11*yy11[__i]
            +  alpha12*yy12[__i] + alpha13*yy13[__i] + alpha14*yy14[__i] + alpha15*yy15[__i];
        };}
      ierr = VecRestoreArrayRead(y[ 0],&yy0 );CHKERRQ(ierr); ierr = VecRestoreArrayRead(y[ 1],&yy1 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(y[ 2],&yy2 );CHKERRQ(ierr); ierr = VecRestoreArrayRead(y[ 3],&yy3 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(y[ 4],&yy4 );CHKERRQ(ierr); ierr = VecRestoreArrayRead(y[ 5],&yy5 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(y[ 6],&yy6 );CHKERRQ(ierr); ierr = VecRestoreArrayRead(y[ 7],&yy7 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(y[ 8],&yy8 );CHKERRQ(ierr); ierr = VecRestoreArrayRead(y[ 9],&yy9 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(y[10],&yy10);CHKERRQ(ierr); ierr = VecRestoreArrayRead(y[11],&yy11);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(y[12],&yy12);CHKERRQ(ierr); ierr = VecRestoreArrayRead(y[13],&yy13);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(y[14],&yy14);CHKERRQ(ierr); ierr = VecRestoreArrayRead(y[15],&yy15);CHKERRQ(ierr);
      y   += 16;
      break;

    case 12:
      ierr   = VecGetArrayRead(y[ 0],&yy0 );CHKERRQ(ierr); ierr = VecGetArrayRead(y[1 ],&yy1 );CHKERRQ(ierr);
      ierr   = VecGetArrayRead(y[ 2],&yy2 );CHKERRQ(ierr); ierr = VecGetArrayRead(y[3 ],&yy3 );CHKERRQ(ierr);
      ierr   = VecGetArrayRead(y[ 4],&yy4 );CHKERRQ(ierr); ierr = VecGetArrayRead(y[5 ],&yy5 );CHKERRQ(ierr);
      ierr   = VecGetArrayRead(y[ 6],&yy6 );CHKERRQ(ierr); ierr = VecGetArrayRead(y[7 ],&yy7 );CHKERRQ(ierr);
      ierr   = VecGetArrayRead(y[ 8],&yy8 );CHKERRQ(ierr); ierr = VecGetArrayRead(y[9 ],&yy9 );CHKERRQ(ierr);
      ierr   = VecGetArrayRead(y[10],&yy10);CHKERRQ(ierr); ierr = VecGetArrayRead(y[11],&yy11);CHKERRQ(ierr);

      alpha0 = alpha[0]; alpha1 = alpha[1]; alpha2 = alpha[ 2]; alpha3 = alpha[ 3];
      alpha4 = alpha[4]; alpha5 = alpha[5]; alpha6 = alpha[ 6]; alpha7 = alpha[ 7];
      alpha8 = alpha[8]; alpha9 = alpha[9]; alpha10= alpha[10]; alpha11= alpha[11];
      alpha += 12;

#if defined(__INTEL_COMPILER)
#pragma ivdep
#pragma unroll(2)
#pragma vector unaligned
      __assume(n%4==0);
#endif
      {PetscInt __i;
        for (__i=0; __i<n; __i++) {
          xx[__i] += alpha0*yy0[__i] + alpha1*yy1[__i] + alpha2 *yy2 [__i] + alpha3 *yy3 [__i]
            +  alpha4*yy4[__i] + alpha5*yy5[__i] + alpha6 *yy6 [__i] + alpha7 *yy7 [__i]
            +  alpha8*yy8[__i] + alpha9*yy9[__i] + alpha10*yy10[__i] + alpha11*yy11[__i];
        };}
      ierr = VecRestoreArrayRead(y[ 0],&yy0 );CHKERRQ(ierr); ierr = VecRestoreArrayRead(y[ 1],&yy1 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(y[ 2],&yy2 );CHKERRQ(ierr); ierr = VecRestoreArrayRead(y[ 3],&yy3 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(y[ 4],&yy4 );CHKERRQ(ierr); ierr = VecRestoreArrayRead(y[ 5],&yy5 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(y[ 6],&yy6 );CHKERRQ(ierr); ierr = VecRestoreArrayRead(y[ 7],&yy7 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(y[ 8],&yy8 );CHKERRQ(ierr); ierr = VecRestoreArrayRead(y[ 9],&yy9 );CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(y[10],&yy10);CHKERRQ(ierr); ierr = VecRestoreArrayRead(y[11],&yy11);CHKERRQ(ierr);
      y   += 12;
      break;

    case  8:
      ierr   = VecGetArrayRead(y[0],&yy0);CHKERRQ(ierr);
      ierr   = VecGetArrayRead(y[1],&yy1);CHKERRQ(ierr);
      ierr   = VecGetArrayRead(y[2],&yy2);CHKERRQ(ierr);
      ierr   = VecGetArrayRead(y[3],&yy3);CHKERRQ(ierr);
      ierr   = VecGetArrayRead(y[4],&yy4);CHKERRQ(ierr);
      ierr   = VecGetArrayRead(y[5],&yy5);CHKERRQ(ierr);
      ierr   = VecGetArrayRead(y[6],&yy6);CHKERRQ(ierr);
      ierr   = VecGetArrayRead(y[7],&yy7);CHKERRQ(ierr);

      alpha0 = alpha[0]; alpha1 = alpha[1]; alpha2 = alpha[2]; alpha3 = alpha[3];
      alpha4 = alpha[4]; alpha5 = alpha[5]; alpha6 = alpha[6]; alpha7 = alpha[7];
      alpha += 8;

#if defined(__INTEL_COMPILER)
#pragma ivdep
#pragma unroll(4)
#endif
      {PetscInt __i;
        for (__i=0; __i<n; __i++) {
          xx[__i] += alpha0*yy0[__i] + alpha1*yy1[__i] + alpha2*yy2[__i] + alpha3*yy3[__i]
            +  alpha4*yy4[__i] + alpha5*yy5[__i] + alpha6*yy6[__i] + alpha7*yy7[__i];
        };}
      ierr = VecRestoreArrayRead(y[0],&yy0);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(y[1],&yy1);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(y[2],&yy2);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(y[3],&yy3);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(y[4],&yy4);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(y[5],&yy5);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(y[6],&yy6);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(y[7],&yy7);CHKERRQ(ierr);
      y   += 8;
      break;

    default:
      for (j=j_rem; j<nv; j+=4) {
        ierr   = VecGetArrayRead(y[0],&yy0);CHKERRQ(ierr);
        ierr   = VecGetArrayRead(y[1],&yy1);CHKERRQ(ierr);
        ierr   = VecGetArrayRead(y[2],&yy2);CHKERRQ(ierr);
        ierr   = VecGetArrayRead(y[3],&yy3);CHKERRQ(ierr);
        alpha0 = alpha[0]; alpha1 = alpha[1]; alpha2 = alpha[2]; alpha3 = alpha[3];
        alpha += 4;
#if defined(__INTEL_COMPILER)
#pragma ivdep
#endif
        PetscKernelAXPY4(xx,alpha0,alpha1,alpha2,alpha3,yy0,yy1,yy2,yy3,n);
        ierr = VecRestoreArrayRead(y[0],&yy0);CHKERRQ(ierr);
        ierr = VecRestoreArrayRead(y[1],&yy1);CHKERRQ(ierr);
        ierr = VecRestoreArrayRead(y[2],&yy2);CHKERRQ(ierr);
        ierr = VecRestoreArrayRead(y[3],&yy3);CHKERRQ(ierr);
        y   += 4;
      }
  }
  ierr = VecRestoreArray(xin,&xx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#include <../src/vec/vec/impls/seq/ftn-kernels/faypx.h>

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

    for (i=0; i<n; i++) yy[i] = xx[i] - yy[i];

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

      for (i=0; i<n; i++) yy[i] = xx[i] + alpha*yy[i];
    }
#endif
    ierr = VecRestoreArrayRead(xin,&xx);CHKERRQ(ierr);
    ierr = VecRestoreArray(yin,&yy);CHKERRQ(ierr);
    ierr = PetscLogFlops(2.0*n);CHKERRQ(ierr);
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
  ierr = MPIU_Allreduce(&m,max,1,MPIU_REAL,MPIU_MAX,PetscObjectComm((PetscObject)xin));CHKERRQ(ierr);
  ierr = PetscLogFlops(n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecPlaceArray_Seq(Vec vin,const PetscScalar *a)
{
  Vec_Seq *v = (Vec_Seq*)vin->data;

  PetscFunctionBegin;
  if (v->unplacedarray) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"VecPlaceArray() was already called on this vector, without a call to VecResetArray()");
  v->unplacedarray = v->array;  /* save previous array so reset can bring it back */
  v->array         = (PetscScalar*)a;
  PetscFunctionReturn(0);
}

PetscErrorCode VecReplaceArray_Seq(Vec vin,const PetscScalar *a)
{
  Vec_Seq        *v = (Vec_Seq*)vin->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(v->array_allocated);CHKERRQ(ierr);
  v->array_allocated = v->array = (PetscScalar*)a;
  PetscFunctionReturn(0);
}
