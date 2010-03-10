#define PETSCMAT_DLL

/*
    Factorization code for BAIJ format. 
*/
#include "../src/mat/impls/baij/seq/baij.h"
#include "../src/mat/blockinvert.h"

/* ------------------------------------------------------------*/
/*
      Version for when blocks are 6 by 6
*/
#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorNumeric_SeqBAIJ_6_inplace"
PetscErrorCode MatLUFactorNumeric_SeqBAIJ_6_inplace(Mat C,Mat A,const MatFactorInfo *info)
{
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data,*b = (Mat_SeqBAIJ *)C->data;
  IS             isrow = b->row,isicol = b->icol;
  PetscErrorCode ierr;
  const PetscInt *ajtmpold,*ajtmp,*diag_offset = b->diag,*r,*ic,*bi = b->i,*bj = b->j,*ai=a->i,*aj=a->j,*pj;
  PetscInt       nz,row,i,j,n = a->mbs,idx;
  MatScalar      *pv,*v,*rtmp,*pc,*w,*x;
  MatScalar      p1,p2,p3,p4,m1,m2,m3,m4,m5,m6,m7,m8,m9,x1,x2,x3,x4;
  MatScalar      p5,p6,p7,p8,p9,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16;
  MatScalar      x17,x18,x19,x20,x21,x22,x23,x24,x25,p10,p11,p12,p13,p14;
  MatScalar      p15,p16,p17,p18,p19,p20,p21,p22,p23,p24,p25,m10,m11,m12;
  MatScalar      m13,m14,m15,m16,m17,m18,m19,m20,m21,m22,m23,m24,m25;
  MatScalar      p26,p27,p28,p29,p30,p31,p32,p33,p34,p35,p36;
  MatScalar      x26,x27,x28,x29,x30,x31,x32,x33,x34,x35,x36;
  MatScalar      m26,m27,m28,m29,m30,m31,m32,m33,m34,m35,m36;
  MatScalar      *ba = b->a,*aa = a->a;
  PetscReal      shift = info->shiftamount;

  PetscFunctionBegin;
  ierr = ISGetIndices(isrow,&r);CHKERRQ(ierr);
  ierr = ISGetIndices(isicol,&ic);CHKERRQ(ierr);
  ierr = PetscMalloc(36*(n+1)*sizeof(MatScalar),&rtmp);CHKERRQ(ierr);

  for (i=0; i<n; i++) {
    nz    = bi[i+1] - bi[i];
    ajtmp = bj + bi[i];
    for  (j=0; j<nz; j++) {
      x = rtmp+36*ajtmp[j]; 
      x[0] = x[1] = x[2] = x[3] = x[4] = x[5] = x[6] = x[7] = x[8] = x[9] = 0.0;
      x[10] = x[11] = x[12] = x[13] = x[14] = x[15] = x[16] = x[17] = 0.0;
      x[18] = x[19] = x[20] = x[21] = x[22] = x[23] = x[24] = x[25] = 0.0 ;
      x[26] = x[27] = x[28] = x[29] = x[30] = x[31] = x[32] = x[33] = 0.0 ;
      x[34] = x[35] = 0.0 ;
    }
    /* load in initial (unfactored row) */
    idx      = r[i];
    nz       = ai[idx+1] - ai[idx];
    ajtmpold = aj + ai[idx];
    v        = aa + 36*ai[idx];
    for (j=0; j<nz; j++) {
      x    = rtmp+36*ic[ajtmpold[j]];
      x[0] =  v[0];  x[1] =  v[1];  x[2] =  v[2];  x[3] =  v[3];
      x[4] =  v[4];  x[5] =  v[5];  x[6] =  v[6];  x[7] =  v[7]; 
      x[8] =  v[8];  x[9] =  v[9];  x[10] = v[10]; x[11] = v[11]; 
      x[12] = v[12]; x[13] = v[13]; x[14] = v[14]; x[15] = v[15]; 
      x[16] = v[16]; x[17] = v[17]; x[18] = v[18]; x[19] = v[19]; 
      x[20] = v[20]; x[21] = v[21]; x[22] = v[22]; x[23] = v[23]; 
      x[24] = v[24]; x[25] = v[25]; x[26] = v[26]; x[27] = v[27]; 
      x[28] = v[28]; x[29] = v[29]; x[30] = v[30]; x[31] = v[31]; 
      x[32] = v[32]; x[33] = v[33]; x[34] = v[34]; x[35] = v[35]; 
      v    += 36;
    }
    row = *ajtmp++;
    while (row < i) {
      pc  =  rtmp + 36*row;
      p1  = pc[0];  p2  = pc[1];  p3  = pc[2];  p4  = pc[3];
      p5  = pc[4];  p6  = pc[5];  p7  = pc[6];  p8  = pc[7];
      p9  = pc[8];  p10 = pc[9];  p11 = pc[10]; p12 = pc[11]; 
      p13 = pc[12]; p14 = pc[13]; p15 = pc[14]; p16 = pc[15]; 
      p17 = pc[16]; p18 = pc[17]; p19 = pc[18]; p20 = pc[19];
      p21 = pc[20]; p22 = pc[21]; p23 = pc[22]; p24 = pc[23];
      p25 = pc[24]; p26 = pc[25]; p27 = pc[26]; p28 = pc[27];
      p29 = pc[28]; p30 = pc[29]; p31 = pc[30]; p32 = pc[31];
      p33 = pc[32]; p34 = pc[33]; p35 = pc[34]; p36 = pc[35];
      if (p1  != 0.0 || p2  != 0.0 || p3  != 0.0 || p4  != 0.0 || 
          p5  != 0.0 || p6  != 0.0 || p7  != 0.0 || p8  != 0.0 ||
          p9  != 0.0 || p10 != 0.0 || p11 != 0.0 || p12 != 0.0 || 
          p13 != 0.0 || p14 != 0.0 || p15 != 0.0 || p16 != 0.0 ||
          p17 != 0.0 || p18 != 0.0 || p19 != 0.0 || p20 != 0.0 ||
          p21 != 0.0 || p22 != 0.0 || p23 != 0.0 || p24 != 0.0 ||
          p25 != 0.0 || p26 != 0.0 || p27 != 0.0 || p28 != 0.0 ||
          p29 != 0.0 || p30 != 0.0 || p31 != 0.0 || p32 != 0.0 ||
          p33 != 0.0 || p34 != 0.0 || p35 != 0.0 || p36 != 0.0) { 
        pv = ba + 36*diag_offset[row];
        pj = bj + diag_offset[row] + 1;
	x1  = pv[0];  x2  = pv[1];  x3  = pv[2];  x4  = pv[3];
	x5  = pv[4];  x6  = pv[5];  x7  = pv[6];  x8  = pv[7];
	x9  = pv[8];  x10 = pv[9];  x11 = pv[10]; x12 = pv[11]; 
	x13 = pv[12]; x14 = pv[13]; x15 = pv[14]; x16 = pv[15]; 
	x17 = pv[16]; x18 = pv[17]; x19 = pv[18]; x20 = pv[19];
	x21 = pv[20]; x22 = pv[21]; x23 = pv[22]; x24 = pv[23];
	x25 = pv[24]; x26 = pv[25]; x27 = pv[26]; x28 = pv[27];
	x29 = pv[28]; x30 = pv[29]; x31 = pv[30]; x32 = pv[31];
	x33 = pv[32]; x34 = pv[33]; x35 = pv[34]; x36 = pv[35];
        pc[0]  = m1  = p1*x1  + p7*x2   + p13*x3  + p19*x4  + p25*x5  + p31*x6;
        pc[1]  = m2  = p2*x1  + p8*x2   + p14*x3  + p20*x4  + p26*x5  + p32*x6;
        pc[2]  = m3  = p3*x1  + p9*x2   + p15*x3  + p21*x4  + p27*x5  + p33*x6;
        pc[3]  = m4  = p4*x1  + p10*x2  + p16*x3  + p22*x4  + p28*x5  + p34*x6;
        pc[4]  = m5  = p5*x1  + p11*x2  + p17*x3  + p23*x4  + p29*x5  + p35*x6;
        pc[5]  = m6  = p6*x1  + p12*x2  + p18*x3  + p24*x4  + p30*x5  + p36*x6;

        pc[6]  = m7  = p1*x7  + p7*x8   + p13*x9  + p19*x10 + p25*x11 + p31*x12;
        pc[7]  = m8  = p2*x7  + p8*x8   + p14*x9  + p20*x10 + p26*x11 + p32*x12;
        pc[8]  = m9  = p3*x7  + p9*x8   + p15*x9  + p21*x10 + p27*x11 + p33*x12;
        pc[9]  = m10 = p4*x7  + p10*x8  + p16*x9  + p22*x10 + p28*x11 + p34*x12;
        pc[10] = m11 = p5*x7  + p11*x8  + p17*x9  + p23*x10 + p29*x11 + p35*x12;
        pc[11] = m12 = p6*x7  + p12*x8  + p18*x9  + p24*x10 + p30*x11 + p36*x12;

        pc[12] = m13 = p1*x13 + p7*x14  + p13*x15 + p19*x16 + p25*x17 + p31*x18;
        pc[13] = m14 = p2*x13 + p8*x14  + p14*x15 + p20*x16 + p26*x17 + p32*x18;
        pc[14] = m15 = p3*x13 + p9*x14  + p15*x15 + p21*x16 + p27*x17 + p33*x18;
        pc[15] = m16 = p4*x13 + p10*x14 + p16*x15 + p22*x16 + p28*x17 + p34*x18;
        pc[16] = m17 = p5*x13 + p11*x14 + p17*x15 + p23*x16 + p29*x17 + p35*x18;
        pc[17] = m18 = p6*x13 + p12*x14 + p18*x15 + p24*x16 + p30*x17 + p36*x18;

        pc[18] = m19 = p1*x19 + p7*x20  + p13*x21 + p19*x22 + p25*x23 + p31*x24;
        pc[19] = m20 = p2*x19 + p8*x20  + p14*x21 + p20*x22 + p26*x23 + p32*x24;
        pc[20] = m21 = p3*x19 + p9*x20  + p15*x21 + p21*x22 + p27*x23 + p33*x24;
        pc[21] = m22 = p4*x19 + p10*x20 + p16*x21 + p22*x22 + p28*x23 + p34*x24;
        pc[22] = m23 = p5*x19 + p11*x20 + p17*x21 + p23*x22 + p29*x23 + p35*x24;
        pc[23] = m24 = p6*x19 + p12*x20 + p18*x21 + p24*x22 + p30*x23 + p36*x24;

        pc[24] = m25 = p1*x25 + p7*x26  + p13*x27 + p19*x28 + p25*x29 + p31*x30;
        pc[25] = m26 = p2*x25 + p8*x26  + p14*x27 + p20*x28 + p26*x29 + p32*x30;
        pc[26] = m27 = p3*x25 + p9*x26  + p15*x27 + p21*x28 + p27*x29 + p33*x30;
        pc[27] = m28 = p4*x25 + p10*x26 + p16*x27 + p22*x28 + p28*x29 + p34*x30;
        pc[28] = m29 = p5*x25 + p11*x26 + p17*x27 + p23*x28 + p29*x29 + p35*x30;
        pc[29] = m30 = p6*x25 + p12*x26 + p18*x27 + p24*x28 + p30*x29 + p36*x30;

        pc[30] = m31 = p1*x31 + p7*x32  + p13*x33 + p19*x34 + p25*x35 + p31*x36;
        pc[31] = m32 = p2*x31 + p8*x32  + p14*x33 + p20*x34 + p26*x35 + p32*x36;
        pc[32] = m33 = p3*x31 + p9*x32  + p15*x33 + p21*x34 + p27*x35 + p33*x36;
        pc[33] = m34 = p4*x31 + p10*x32 + p16*x33 + p22*x34 + p28*x35 + p34*x36;
        pc[34] = m35 = p5*x31 + p11*x32 + p17*x33 + p23*x34 + p29*x35 + p35*x36;
        pc[35] = m36 = p6*x31 + p12*x32 + p18*x33 + p24*x34 + p30*x35 + p36*x36;

        nz = bi[row+1] - diag_offset[row] - 1;
        pv += 36;
        for (j=0; j<nz; j++) {
	  x1  = pv[0];  x2  = pv[1];  x3  = pv[2];  x4  = pv[3];
	  x5  = pv[4];  x6  = pv[5];  x7  = pv[6];  x8  = pv[7];
	  x9  = pv[8];  x10 = pv[9];  x11 = pv[10]; x12 = pv[11]; 
	  x13 = pv[12]; x14 = pv[13]; x15 = pv[14]; x16 = pv[15]; 
	  x17 = pv[16]; x18 = pv[17]; x19 = pv[18]; x20 = pv[19];
	  x21 = pv[20]; x22 = pv[21]; x23 = pv[22]; x24 = pv[23];
	  x25 = pv[24]; x26 = pv[25]; x27 = pv[26]; x28 = pv[27];
	  x29 = pv[28]; x30 = pv[29]; x31 = pv[30]; x32 = pv[31];
	  x33 = pv[32]; x34 = pv[33]; x35 = pv[34]; x36 = pv[35];
	  x    = rtmp + 36*pj[j];
          x[0]  -= m1*x1  + m7*x2   + m13*x3  + m19*x4  + m25*x5  + m31*x6;
          x[1]  -= m2*x1  + m8*x2   + m14*x3  + m20*x4  + m26*x5  + m32*x6;
          x[2]  -= m3*x1  + m9*x2   + m15*x3  + m21*x4  + m27*x5  + m33*x6;
          x[3]  -= m4*x1  + m10*x2  + m16*x3  + m22*x4  + m28*x5  + m34*x6;
          x[4]  -= m5*x1  + m11*x2  + m17*x3  + m23*x4  + m29*x5  + m35*x6;
          x[5]  -= m6*x1  + m12*x2  + m18*x3  + m24*x4  + m30*x5  + m36*x6;

	  x[6]  -= m1*x7  + m7*x8   + m13*x9  + m19*x10 + m25*x11 + m31*x12;
	  x[7]  -= m2*x7  + m8*x8   + m14*x9  + m20*x10 + m26*x11 + m32*x12;
	  x[8]  -= m3*x7  + m9*x8   + m15*x9  + m21*x10 + m27*x11 + m33*x12;
	  x[9]  -= m4*x7  + m10*x8  + m16*x9  + m22*x10 + m28*x11 + m34*x12;
	  x[10] -= m5*x7  + m11*x8  + m17*x9  + m23*x10 + m29*x11 + m35*x12;
	  x[11] -= m6*x7  + m12*x8  + m18*x9  + m24*x10 + m30*x11 + m36*x12;

	  x[12] -= m1*x13 + m7*x14  + m13*x15 + m19*x16 + m25*x17 + m31*x18;
	  x[13] -= m2*x13 + m8*x14  + m14*x15 + m20*x16 + m26*x17 + m32*x18;
	  x[14] -= m3*x13 + m9*x14  + m15*x15 + m21*x16 + m27*x17 + m33*x18;
	  x[15] -= m4*x13 + m10*x14 + m16*x15 + m22*x16 + m28*x17 + m34*x18;
	  x[16] -= m5*x13 + m11*x14 + m17*x15 + m23*x16 + m29*x17 + m35*x18;
	  x[17] -= m6*x13 + m12*x14 + m18*x15 + m24*x16 + m30*x17 + m36*x18;

	  x[18] -= m1*x19 + m7*x20  + m13*x21 + m19*x22 + m25*x23 + m31*x24;
	  x[19] -= m2*x19 + m8*x20  + m14*x21 + m20*x22 + m26*x23 + m32*x24;
	  x[20] -= m3*x19 + m9*x20  + m15*x21 + m21*x22 + m27*x23 + m33*x24;
	  x[21] -= m4*x19 + m10*x20 + m16*x21 + m22*x22 + m28*x23 + m34*x24;
	  x[22] -= m5*x19 + m11*x20 + m17*x21 + m23*x22 + m29*x23 + m35*x24;
	  x[23] -= m6*x19 + m12*x20 + m18*x21 + m24*x22 + m30*x23 + m36*x24;

	  x[24] -= m1*x25 + m7*x26  + m13*x27 + m19*x28 + m25*x29 + m31*x30;
	  x[25] -= m2*x25 + m8*x26  + m14*x27 + m20*x28 + m26*x29 + m32*x30;
	  x[26] -= m3*x25 + m9*x26  + m15*x27 + m21*x28 + m27*x29 + m33*x30;
	  x[27] -= m4*x25 + m10*x26 + m16*x27 + m22*x28 + m28*x29 + m34*x30;
	  x[28] -= m5*x25 + m11*x26 + m17*x27 + m23*x28 + m29*x29 + m35*x30;
	  x[29] -= m6*x25 + m12*x26 + m18*x27 + m24*x28 + m30*x29 + m36*x30;

	  x[30] -= m1*x31 + m7*x32  + m13*x33 + m19*x34 + m25*x35 + m31*x36;
	  x[31] -= m2*x31 + m8*x32  + m14*x33 + m20*x34 + m26*x35 + m32*x36;
	  x[32] -= m3*x31 + m9*x32  + m15*x33 + m21*x34 + m27*x35 + m33*x36;
	  x[33] -= m4*x31 + m10*x32 + m16*x33 + m22*x34 + m28*x35 + m34*x36;
	  x[34] -= m5*x31 + m11*x32 + m17*x33 + m23*x34 + m29*x35 + m35*x36;
	  x[35] -= m6*x31 + m12*x32 + m18*x33 + m24*x34 + m30*x35 + m36*x36;

          pv   += 36;
        }
        ierr = PetscLogFlops(432.0*nz+396.0);CHKERRQ(ierr);
      } 
      row = *ajtmp++;
    }
    /* finished row so stick it into b->a */
    pv = ba + 36*bi[i];
    pj = bj + bi[i];
    nz = bi[i+1] - bi[i];
    for (j=0; j<nz; j++) {
      x      = rtmp+36*pj[j];
      pv[0]  = x[0];  pv[1]  = x[1];  pv[2]  = x[2];  pv[3]  = x[3];
      pv[4]  = x[4];  pv[5]  = x[5];  pv[6]  = x[6];  pv[7]  = x[7]; 
      pv[8]  = x[8];  pv[9]  = x[9];  pv[10] = x[10]; pv[11] = x[11]; 
      pv[12] = x[12]; pv[13] = x[13]; pv[14] = x[14]; pv[15] = x[15]; 
      pv[16] = x[16]; pv[17] = x[17]; pv[18] = x[18]; pv[19] = x[19]; 
      pv[20] = x[20]; pv[21] = x[21]; pv[22] = x[22]; pv[23] = x[23]; 
      pv[24] = x[24]; pv[25] = x[25]; pv[26] = x[26]; pv[27] = x[27]; 
      pv[28] = x[28]; pv[29] = x[29]; pv[30] = x[30]; pv[31] = x[31]; 
      pv[32] = x[32]; pv[33] = x[33]; pv[34] = x[34]; pv[35] = x[35]; 
      pv   += 36;
    }
    /* invert diagonal block */
    w = ba + 36*diag_offset[i];
    ierr = Kernel_A_gets_inverse_A_6(w,shift);CHKERRQ(ierr);
  }

  ierr = PetscFree(rtmp);CHKERRQ(ierr);
  ierr = ISRestoreIndices(isicol,&ic);CHKERRQ(ierr);
  ierr = ISRestoreIndices(isrow,&r);CHKERRQ(ierr);
  C->ops->solve          = MatSolve_SeqBAIJ_6_inplace;
  C->ops->solvetranspose = MatSolveTranspose_SeqBAIJ_6_inplace;
  C->assembled = PETSC_TRUE;
  ierr = PetscLogFlops(1.333333333333*6*6*6*b->mbs);CHKERRQ(ierr); /* from inverting diagonal blocks */
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorNumeric_SeqBAIJ_6"
PetscErrorCode MatLUFactorNumeric_SeqBAIJ_6(Mat B,Mat A,const MatFactorInfo *info)
{
  Mat            C=B;
  Mat_SeqBAIJ    *a=(Mat_SeqBAIJ*)A->data,*b=(Mat_SeqBAIJ *)C->data;
  IS             isrow = b->row,isicol = b->icol;
  PetscErrorCode ierr;
  const PetscInt *r,*ic,*ics;
  PetscInt       i,j,k,nz,nzL,row;
  const PetscInt n=a->mbs,*ai=a->i,*aj=a->j,*bi=b->i,*bj=b->j;
  const PetscInt *ajtmp,*bjtmp,*bdiag=b->diag,*pj,bs2=a->bs2;
  MatScalar      *rtmp,*pc,*mwork,*v,*pv,*aa=a->a;
  PetscInt       flg;
  PetscReal      shift = info->shiftamount;

  PetscFunctionBegin;
  ierr = ISGetIndices(isrow,&r);CHKERRQ(ierr);
  ierr = ISGetIndices(isicol,&ic);CHKERRQ(ierr);

  /* generate work space needed by the factorization */
  ierr = PetscMalloc2(bs2*n,MatScalar,&rtmp,bs2,MatScalar,&mwork);CHKERRQ(ierr);
  ierr = PetscMemzero(rtmp,bs2*n*sizeof(MatScalar));CHKERRQ(ierr);
  ics  = ic;

  for (i=0; i<n; i++){
    /* zero rtmp */
    /* L part */
    nz    = bi[i+1] - bi[i];
    bjtmp = bj + bi[i];
    for  (j=0; j<nz; j++){
      ierr = PetscMemzero(rtmp+bs2*bjtmp[j],bs2*sizeof(MatScalar));CHKERRQ(ierr);
    }

    /* U part */
    nz = bdiag[i] - bdiag[i+1]; 
    bjtmp = bj + bdiag[i+1]+1; 
    for  (j=0; j<nz; j++){
      ierr = PetscMemzero(rtmp+bs2*bjtmp[j],bs2*sizeof(MatScalar));CHKERRQ(ierr);
    }
 
    /* load in initial (unfactored row) */
    nz    = ai[r[i]+1] - ai[r[i]];
    ajtmp = aj + ai[r[i]];
    v     = aa + bs2*ai[r[i]];
    for (j=0; j<nz; j++) {
      ierr = PetscMemcpy(rtmp+bs2*ic[ajtmp[j]],v+bs2*j,bs2*sizeof(MatScalar));CHKERRQ(ierr);
    }

    /* elimination */
    bjtmp = bj + bi[i];
    nzL   = bi[i+1] - bi[i];
    for(k=0;k < nzL;k++) {
      row = bjtmp[k];
      pc = rtmp + bs2*row;
      for (flg=0,j=0; j<bs2; j++) { if (pc[j]!=0.0) { flg = 1; break; }}
      if (flg) {
        pv = b->a + bs2*bdiag[row];      
        /* Kernel_A_gets_A_times_B(bs,pc,pv,mwork); *pc = *pc * (*pv); */
        ierr = Kernel_A_gets_A_times_B_6(pc,pv,mwork);CHKERRQ(ierr);
  
        pj = b->j + bdiag[row+1]+1; /* begining of U(row,:) */
        pv = b->a + bs2*(bdiag[row+1]+1); 
        nz = bdiag[row] - bdiag[row+1] -  1; /* num of entries inU(row,:), excluding diag */
        for (j=0; j<nz; j++) {
          /* Kernel_A_gets_A_minus_B_times_C(bs,rtmp+bs2*pj[j],pc,pv+bs2*j); */
          /* rtmp+bs2*pj[j] = rtmp+bs2*pj[j] - (*pc)*(pv+bs2*j) */
          v    = rtmp + bs2*pj[j];
          ierr = Kernel_A_gets_A_minus_B_times_C_6(v,pc,pv);CHKERRQ(ierr);
          pv  += bs2;          
        }
        ierr = PetscLogFlops(432*nz+396);CHKERRQ(ierr); /* flops = 2*bs^3*nz + 2*bs^3 - bs2) */
      }
    }

    /* finished row so stick it into b->a */
    /* L part */
    pv   = b->a + bs2*bi[i] ;
    pj   = b->j + bi[i] ;
    nz   = bi[i+1] - bi[i];
    for (j=0; j<nz; j++) {
      ierr = PetscMemcpy(pv+bs2*j,rtmp+bs2*pj[j],bs2*sizeof(MatScalar));CHKERRQ(ierr);
    }

    /* Mark diagonal and invert diagonal for simplier triangular solves */
    pv   = b->a + bs2*bdiag[i];
    pj   = b->j + bdiag[i];
    ierr = PetscMemcpy(pv,rtmp+bs2*pj[0],bs2*sizeof(MatScalar));CHKERRQ(ierr);   
    /* ierr = Kernel_A_gets_inverse_A(bs,pv,v_pivots,v_work);CHKERRQ(ierr); */
    ierr = Kernel_A_gets_inverse_A_6(pv,shift);CHKERRQ(ierr);
      
    /* U part */
    pv = b->a + bs2*(bdiag[i+1]+1);
    pj = b->j + bdiag[i+1]+1;
    nz = bdiag[i] - bdiag[i+1] - 1; 
    for (j=0; j<nz; j++){
      ierr = PetscMemcpy(pv+bs2*j,rtmp+bs2*pj[j],bs2*sizeof(MatScalar));CHKERRQ(ierr);
    }
  }

  ierr = PetscFree2(rtmp,mwork);CHKERRQ(ierr);
  ierr = ISRestoreIndices(isicol,&ic);CHKERRQ(ierr);
  ierr = ISRestoreIndices(isrow,&r);CHKERRQ(ierr);
  C->ops->solve          = MatSolve_SeqBAIJ_6;
  C->ops->solvetranspose = MatSolveTranspose_SeqBAIJ_6;
  C->assembled = PETSC_TRUE;
  ierr = PetscLogFlops(1.333333333333*6*6*6*n);CHKERRQ(ierr); /* from inverting diagonal blocks */
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorNumeric_SeqBAIJ_6_NaturalOrdering_inplace"
PetscErrorCode MatLUFactorNumeric_SeqBAIJ_6_NaturalOrdering_inplace(Mat C,Mat A,const MatFactorInfo *info)
{
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data,*b = (Mat_SeqBAIJ *)C->data;
  PetscErrorCode ierr;
  PetscInt       i,j,n = a->mbs,*bi = b->i,*bj = b->j;
  PetscInt       *ajtmpold,*ajtmp,nz,row;
  PetscInt       *diag_offset = b->diag,*ai=a->i,*aj=a->j,*pj;
  MatScalar      *pv,*v,*rtmp,*pc,*w,*x;
  MatScalar      x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15;
  MatScalar      x16,x17,x18,x19,x20,x21,x22,x23,x24,x25;
  MatScalar      p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15;
  MatScalar      p16,p17,p18,p19,p20,p21,p22,p23,p24,p25;
  MatScalar      m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15;
  MatScalar      m16,m17,m18,m19,m20,m21,m22,m23,m24,m25;
  MatScalar      p26,p27,p28,p29,p30,p31,p32,p33,p34,p35,p36;
  MatScalar      x26,x27,x28,x29,x30,x31,x32,x33,x34,x35,x36;
  MatScalar      m26,m27,m28,m29,m30,m31,m32,m33,m34,m35,m36;
  MatScalar      *ba = b->a,*aa = a->a;
  PetscReal      shift = info->shiftamount;

  PetscFunctionBegin;
  ierr = PetscMalloc(36*(n+1)*sizeof(MatScalar),&rtmp);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    nz    = bi[i+1] - bi[i];
    ajtmp = bj + bi[i];
    for  (j=0; j<nz; j++) {
      x = rtmp+36*ajtmp[j]; 
      x[0] = x[1] = x[2] = x[3] = x[4] = x[5] = x[6] = x[7] = x[8] = x[9] = 0.0;
      x[10] = x[11] = x[12] = x[13] = x[14] = x[15] = x[16] = x[17] = 0.0;
      x[18] = x[19] = x[20] = x[21] = x[22] = x[23] = x[24] = x[25] = 0.0 ;
      x[26] = x[27] = x[28] = x[29] = x[30] = x[31] = x[32] = x[33] = 0.0 ;
      x[34] = x[35] = 0.0 ;
    }
    /* load in initial (unfactored row) */
    nz       = ai[i+1] - ai[i];
    ajtmpold = aj + ai[i];
    v        = aa + 36*ai[i];
    for (j=0; j<nz; j++) {
      x    = rtmp+36*ajtmpold[j];
      x[0] =  v[0];  x[1] =  v[1];  x[2] =  v[2];  x[3] =  v[3];
      x[4] =  v[4];  x[5] =  v[5];  x[6] =  v[6];  x[7] =  v[7]; 
      x[8] =  v[8];  x[9] =  v[9];  x[10] = v[10]; x[11] = v[11]; 
      x[12] = v[12]; x[13] = v[13]; x[14] = v[14]; x[15] = v[15]; 
      x[16] = v[16]; x[17] = v[17]; x[18] = v[18]; x[19] = v[19]; 
      x[20] = v[20]; x[21] = v[21]; x[22] = v[22]; x[23] = v[23]; 
      x[24] = v[24]; x[25] = v[25]; x[26] = v[26]; x[27] = v[27]; 
      x[28] = v[28]; x[29] = v[29]; x[30] = v[30]; x[31] = v[31]; 
      x[32] = v[32]; x[33] = v[33]; x[34] = v[34]; x[35] = v[35]; 
      v    += 36;
    }
    row = *ajtmp++;
    while (row < i) {
      pc  = rtmp + 36*row;
      p1  = pc[0];  p2  = pc[1];  p3  = pc[2];  p4  = pc[3];
      p5  = pc[4];  p6  = pc[5];  p7  = pc[6];  p8  = pc[7];
      p9  = pc[8];  p10 = pc[9];  p11 = pc[10]; p12 = pc[11]; 
      p13 = pc[12]; p14 = pc[13]; p15 = pc[14]; p16 = pc[15]; 
      p17 = pc[16]; p18 = pc[17]; p19 = pc[18]; p20 = pc[19];
      p21 = pc[20]; p22 = pc[21]; p23 = pc[22]; p24 = pc[23];
      p25 = pc[24]; p26 = pc[25]; p27 = pc[26]; p28 = pc[27];
      p29 = pc[28]; p30 = pc[29]; p31 = pc[30]; p32 = pc[31];
      p33 = pc[32]; p34 = pc[33]; p35 = pc[34]; p36 = pc[35];
      if (p1  != 0.0 || p2  != 0.0 || p3  != 0.0 || p4  != 0.0 || 
          p5  != 0.0 || p6  != 0.0 || p7  != 0.0 || p8  != 0.0 ||
          p9  != 0.0 || p10 != 0.0 || p11 != 0.0 || p12 != 0.0 || 
          p13 != 0.0 || p14 != 0.0 || p15 != 0.0 || p16 != 0.0 ||
          p17 != 0.0 || p18 != 0.0 || p19 != 0.0 || p20 != 0.0 ||
          p21 != 0.0 || p22 != 0.0 || p23 != 0.0 || p24 != 0.0 ||
          p25 != 0.0 || p26 != 0.0 || p27 != 0.0 || p28 != 0.0 ||
          p29 != 0.0 || p30 != 0.0 || p31 != 0.0 || p32 != 0.0 ||
          p33 != 0.0 || p34 != 0.0 || p35 != 0.0 || p36 != 0.0) { 
        pv = ba + 36*diag_offset[row];
        pj = bj + diag_offset[row] + 1;
	x1  = pv[0];  x2  = pv[1];  x3  = pv[2];  x4  = pv[3];
	x5  = pv[4];  x6  = pv[5];  x7  = pv[6];  x8  = pv[7];
	x9  = pv[8];  x10 = pv[9];  x11 = pv[10]; x12 = pv[11]; 
	x13 = pv[12]; x14 = pv[13]; x15 = pv[14]; x16 = pv[15]; 
	x17 = pv[16]; x18 = pv[17]; x19 = pv[18]; x20 = pv[19];
	x21 = pv[20]; x22 = pv[21]; x23 = pv[22]; x24 = pv[23];
	x25 = pv[24]; x26 = pv[25]; x27 = pv[26]; x28 = pv[27];
	x29 = pv[28]; x30 = pv[29]; x31 = pv[30]; x32 = pv[31];
	x33 = pv[32]; x34 = pv[33]; x35 = pv[34]; x36 = pv[35];
        pc[0]  = m1  = p1*x1  + p7*x2   + p13*x3  + p19*x4  + p25*x5  + p31*x6;
        pc[1]  = m2  = p2*x1  + p8*x2   + p14*x3  + p20*x4  + p26*x5  + p32*x6;
        pc[2]  = m3  = p3*x1  + p9*x2   + p15*x3  + p21*x4  + p27*x5  + p33*x6;
        pc[3]  = m4  = p4*x1  + p10*x2  + p16*x3  + p22*x4  + p28*x5  + p34*x6;
        pc[4]  = m5  = p5*x1  + p11*x2  + p17*x3  + p23*x4  + p29*x5  + p35*x6;
        pc[5]  = m6  = p6*x1  + p12*x2  + p18*x3  + p24*x4  + p30*x5  + p36*x6;

        pc[6]  = m7  = p1*x7  + p7*x8   + p13*x9  + p19*x10 + p25*x11 + p31*x12;
        pc[7]  = m8  = p2*x7  + p8*x8   + p14*x9  + p20*x10 + p26*x11 + p32*x12;
        pc[8]  = m9  = p3*x7  + p9*x8   + p15*x9  + p21*x10 + p27*x11 + p33*x12;
        pc[9]  = m10 = p4*x7  + p10*x8  + p16*x9  + p22*x10 + p28*x11 + p34*x12;
        pc[10] = m11 = p5*x7  + p11*x8  + p17*x9  + p23*x10 + p29*x11 + p35*x12;
        pc[11] = m12 = p6*x7  + p12*x8  + p18*x9  + p24*x10 + p30*x11 + p36*x12;

        pc[12] = m13 = p1*x13 + p7*x14  + p13*x15 + p19*x16 + p25*x17 + p31*x18;
        pc[13] = m14 = p2*x13 + p8*x14  + p14*x15 + p20*x16 + p26*x17 + p32*x18;
        pc[14] = m15 = p3*x13 + p9*x14  + p15*x15 + p21*x16 + p27*x17 + p33*x18;
        pc[15] = m16 = p4*x13 + p10*x14 + p16*x15 + p22*x16 + p28*x17 + p34*x18;
        pc[16] = m17 = p5*x13 + p11*x14 + p17*x15 + p23*x16 + p29*x17 + p35*x18;
        pc[17] = m18 = p6*x13 + p12*x14 + p18*x15 + p24*x16 + p30*x17 + p36*x18;

        pc[18] = m19 = p1*x19 + p7*x20  + p13*x21 + p19*x22 + p25*x23 + p31*x24;
        pc[19] = m20 = p2*x19 + p8*x20  + p14*x21 + p20*x22 + p26*x23 + p32*x24;
        pc[20] = m21 = p3*x19 + p9*x20  + p15*x21 + p21*x22 + p27*x23 + p33*x24;
        pc[21] = m22 = p4*x19 + p10*x20 + p16*x21 + p22*x22 + p28*x23 + p34*x24;
        pc[22] = m23 = p5*x19 + p11*x20 + p17*x21 + p23*x22 + p29*x23 + p35*x24;
        pc[23] = m24 = p6*x19 + p12*x20 + p18*x21 + p24*x22 + p30*x23 + p36*x24;

        pc[24] = m25 = p1*x25 + p7*x26  + p13*x27 + p19*x28 + p25*x29 + p31*x30;
        pc[25] = m26 = p2*x25 + p8*x26  + p14*x27 + p20*x28 + p26*x29 + p32*x30;
        pc[26] = m27 = p3*x25 + p9*x26  + p15*x27 + p21*x28 + p27*x29 + p33*x30;
        pc[27] = m28 = p4*x25 + p10*x26 + p16*x27 + p22*x28 + p28*x29 + p34*x30;
        pc[28] = m29 = p5*x25 + p11*x26 + p17*x27 + p23*x28 + p29*x29 + p35*x30;
        pc[29] = m30 = p6*x25 + p12*x26 + p18*x27 + p24*x28 + p30*x29 + p36*x30;

        pc[30] = m31 = p1*x31 + p7*x32  + p13*x33 + p19*x34 + p25*x35 + p31*x36;
        pc[31] = m32 = p2*x31 + p8*x32  + p14*x33 + p20*x34 + p26*x35 + p32*x36;
        pc[32] = m33 = p3*x31 + p9*x32  + p15*x33 + p21*x34 + p27*x35 + p33*x36;
        pc[33] = m34 = p4*x31 + p10*x32 + p16*x33 + p22*x34 + p28*x35 + p34*x36;
        pc[34] = m35 = p5*x31 + p11*x32 + p17*x33 + p23*x34 + p29*x35 + p35*x36;
        pc[35] = m36 = p6*x31 + p12*x32 + p18*x33 + p24*x34 + p30*x35 + p36*x36;

        nz = bi[row+1] - diag_offset[row] - 1;
        pv += 36;
        for (j=0; j<nz; j++) {
	  x1  = pv[0];  x2  = pv[1];  x3  = pv[2];  x4  = pv[3];
	  x5  = pv[4];  x6  = pv[5];  x7  = pv[6];  x8  = pv[7];
	  x9  = pv[8];  x10 = pv[9];  x11 = pv[10]; x12 = pv[11]; 
	  x13 = pv[12]; x14 = pv[13]; x15 = pv[14]; x16 = pv[15]; 
	  x17 = pv[16]; x18 = pv[17]; x19 = pv[18]; x20 = pv[19];
	  x21 = pv[20]; x22 = pv[21]; x23 = pv[22]; x24 = pv[23];
	  x25 = pv[24]; x26 = pv[25]; x27 = pv[26]; x28 = pv[27];
	  x29 = pv[28]; x30 = pv[29]; x31 = pv[30]; x32 = pv[31];
	  x33 = pv[32]; x34 = pv[33]; x35 = pv[34]; x36 = pv[35];
	  x    = rtmp + 36*pj[j];
          x[0]  -= m1*x1  + m7*x2   + m13*x3  + m19*x4  + m25*x5  + m31*x6;
          x[1]  -= m2*x1  + m8*x2   + m14*x3  + m20*x4  + m26*x5  + m32*x6;
          x[2]  -= m3*x1  + m9*x2   + m15*x3  + m21*x4  + m27*x5  + m33*x6;
          x[3]  -= m4*x1  + m10*x2  + m16*x3  + m22*x4  + m28*x5  + m34*x6;
          x[4]  -= m5*x1  + m11*x2  + m17*x3  + m23*x4  + m29*x5  + m35*x6;
          x[5]  -= m6*x1  + m12*x2  + m18*x3  + m24*x4  + m30*x5  + m36*x6;

	  x[6]  -= m1*x7  + m7*x8   + m13*x9  + m19*x10 + m25*x11 + m31*x12;
	  x[7]  -= m2*x7  + m8*x8   + m14*x9  + m20*x10 + m26*x11 + m32*x12;
	  x[8]  -= m3*x7  + m9*x8   + m15*x9  + m21*x10 + m27*x11 + m33*x12;
	  x[9]  -= m4*x7  + m10*x8  + m16*x9  + m22*x10 + m28*x11 + m34*x12;
	  x[10] -= m5*x7  + m11*x8  + m17*x9  + m23*x10 + m29*x11 + m35*x12;
	  x[11] -= m6*x7  + m12*x8  + m18*x9  + m24*x10 + m30*x11 + m36*x12;

	  x[12] -= m1*x13 + m7*x14  + m13*x15 + m19*x16 + m25*x17 + m31*x18;
	  x[13] -= m2*x13 + m8*x14  + m14*x15 + m20*x16 + m26*x17 + m32*x18;
	  x[14] -= m3*x13 + m9*x14  + m15*x15 + m21*x16 + m27*x17 + m33*x18;
	  x[15] -= m4*x13 + m10*x14 + m16*x15 + m22*x16 + m28*x17 + m34*x18;
	  x[16] -= m5*x13 + m11*x14 + m17*x15 + m23*x16 + m29*x17 + m35*x18;
	  x[17] -= m6*x13 + m12*x14 + m18*x15 + m24*x16 + m30*x17 + m36*x18;

	  x[18] -= m1*x19 + m7*x20  + m13*x21 + m19*x22 + m25*x23 + m31*x24;
	  x[19] -= m2*x19 + m8*x20  + m14*x21 + m20*x22 + m26*x23 + m32*x24;
	  x[20] -= m3*x19 + m9*x20  + m15*x21 + m21*x22 + m27*x23 + m33*x24;
	  x[21] -= m4*x19 + m10*x20 + m16*x21 + m22*x22 + m28*x23 + m34*x24;
	  x[22] -= m5*x19 + m11*x20 + m17*x21 + m23*x22 + m29*x23 + m35*x24;
	  x[23] -= m6*x19 + m12*x20 + m18*x21 + m24*x22 + m30*x23 + m36*x24;

	  x[24] -= m1*x25 + m7*x26  + m13*x27 + m19*x28 + m25*x29 + m31*x30;
	  x[25] -= m2*x25 + m8*x26  + m14*x27 + m20*x28 + m26*x29 + m32*x30;
	  x[26] -= m3*x25 + m9*x26  + m15*x27 + m21*x28 + m27*x29 + m33*x30;
	  x[27] -= m4*x25 + m10*x26 + m16*x27 + m22*x28 + m28*x29 + m34*x30;
	  x[28] -= m5*x25 + m11*x26 + m17*x27 + m23*x28 + m29*x29 + m35*x30;
	  x[29] -= m6*x25 + m12*x26 + m18*x27 + m24*x28 + m30*x29 + m36*x30;

	  x[30] -= m1*x31 + m7*x32  + m13*x33 + m19*x34 + m25*x35 + m31*x36;
	  x[31] -= m2*x31 + m8*x32  + m14*x33 + m20*x34 + m26*x35 + m32*x36;
	  x[32] -= m3*x31 + m9*x32  + m15*x33 + m21*x34 + m27*x35 + m33*x36;
	  x[33] -= m4*x31 + m10*x32 + m16*x33 + m22*x34 + m28*x35 + m34*x36;
	  x[34] -= m5*x31 + m11*x32 + m17*x33 + m23*x34 + m29*x35 + m35*x36;
	  x[35] -= m6*x31 + m12*x32 + m18*x33 + m24*x34 + m30*x35 + m36*x36;

          pv   += 36;
        }
        ierr = PetscLogFlops(432.0*nz+396.0);CHKERRQ(ierr);
      } 
      row = *ajtmp++;
    }
    /* finished row so stick it into b->a */
    pv = ba + 36*bi[i];
    pj = bj + bi[i];
    nz = bi[i+1] - bi[i];
    for (j=0; j<nz; j++) {
      x      = rtmp+36*pj[j];
      pv[0]  = x[0];  pv[1]  = x[1];  pv[2]  = x[2];  pv[3]  = x[3];
      pv[4]  = x[4];  pv[5]  = x[5];  pv[6]  = x[6];  pv[7]  = x[7]; 
      pv[8]  = x[8];  pv[9]  = x[9];  pv[10] = x[10]; pv[11] = x[11]; 
      pv[12] = x[12]; pv[13] = x[13]; pv[14] = x[14]; pv[15] = x[15]; 
      pv[16] = x[16]; pv[17] = x[17]; pv[18] = x[18]; pv[19] = x[19]; 
      pv[20] = x[20]; pv[21] = x[21]; pv[22] = x[22]; pv[23] = x[23]; 
      pv[24] = x[24]; pv[25] = x[25]; pv[26] = x[26]; pv[27] = x[27]; 
      pv[28] = x[28]; pv[29] = x[29]; pv[30] = x[30]; pv[31] = x[31]; 
      pv[32] = x[32]; pv[33] = x[33]; pv[34] = x[34]; pv[35] = x[35]; 
      pv   += 36;
    }
    /* invert diagonal block */
    w = ba + 36*diag_offset[i];
    ierr = Kernel_A_gets_inverse_A_6(w,shift);CHKERRQ(ierr);
  }

  ierr = PetscFree(rtmp);CHKERRQ(ierr);
  C->ops->solve          = MatSolve_SeqBAIJ_6_NaturalOrdering_inplace;
  C->ops->solvetranspose = MatSolveTranspose_SeqBAIJ_6_NaturalOrdering_inplace;
  C->assembled = PETSC_TRUE;
  ierr = PetscLogFlops(1.333333333333*6*6*6*b->mbs);CHKERRQ(ierr); /* from inverting diagonal blocks */
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorNumeric_SeqBAIJ_6_NaturalOrdering"
PetscErrorCode MatLUFactorNumeric_SeqBAIJ_6_NaturalOrdering(Mat B,Mat A,const MatFactorInfo *info)
{
  Mat            C=B;
  Mat_SeqBAIJ    *a=(Mat_SeqBAIJ*)A->data,*b=(Mat_SeqBAIJ *)C->data;
  PetscErrorCode ierr;
  PetscInt       i,j,k,nz,nzL,row;
  const PetscInt n=a->mbs,*ai=a->i,*aj=a->j,*bi=b->i,*bj=b->j;
  const PetscInt *ajtmp,*bjtmp,*bdiag=b->diag,*pj,bs2=a->bs2;
  MatScalar      *rtmp,*pc,*mwork,*v,*pv,*aa=a->a;
  PetscInt       flg;
  PetscReal      shift = info->shiftamount;

  PetscFunctionBegin;
  /* generate work space needed by the factorization */
  ierr = PetscMalloc2(bs2*n,MatScalar,&rtmp,bs2,MatScalar,&mwork);CHKERRQ(ierr);
  ierr = PetscMemzero(rtmp,bs2*n*sizeof(MatScalar));CHKERRQ(ierr);

  for (i=0; i<n; i++){
    /* zero rtmp */
    /* L part */
    nz    = bi[i+1] - bi[i];
    bjtmp = bj + bi[i];
    for  (j=0; j<nz; j++){
      ierr = PetscMemzero(rtmp+bs2*bjtmp[j],bs2*sizeof(MatScalar));CHKERRQ(ierr);
    }

    /* U part */
    nz = bdiag[i] - bdiag[i+1]; 
    bjtmp = bj + bdiag[i+1]+1; 
    for  (j=0; j<nz; j++){
      ierr = PetscMemzero(rtmp+bs2*bjtmp[j],bs2*sizeof(MatScalar));CHKERRQ(ierr);
    }
 
    /* load in initial (unfactored row) */
    nz    = ai[i+1] - ai[i];
    ajtmp = aj + ai[i];
    v     = aa + bs2*ai[i];
    for (j=0; j<nz; j++) {
      ierr = PetscMemcpy(rtmp+bs2*ajtmp[j],v+bs2*j,bs2*sizeof(MatScalar));CHKERRQ(ierr);
    }

    /* elimination */
    bjtmp = bj + bi[i];
    nzL   = bi[i+1] - bi[i];
    for(k=0;k < nzL;k++) {
      row = bjtmp[k];
      pc = rtmp + bs2*row;
      for (flg=0,j=0; j<bs2; j++) { if (pc[j]!=0.0) { flg = 1; break; }}
      if (flg) {
        pv = b->a + bs2*bdiag[row];      
        /* Kernel_A_gets_A_times_B(bs,pc,pv,mwork); *pc = *pc * (*pv); */
        ierr = Kernel_A_gets_A_times_B_6(pc,pv,mwork);CHKERRQ(ierr);
  
        pj = b->j + bdiag[row+1]+1; /* begining of U(row,:) */
        pv = b->a + bs2*(bdiag[row+1]+1); 
        nz = bdiag[row] - bdiag[row+1] - 1; /* num of entries inU(row,:), excluding diag */
        for (j=0; j<nz; j++) {
          /* Kernel_A_gets_A_minus_B_times_C(bs,rtmp+bs2*pj[j],pc,pv+bs2*j); */
          /* rtmp+bs2*pj[j] = rtmp+bs2*pj[j] - (*pc)*(pv+bs2*j) */
          v    = rtmp + bs2*pj[j];
          ierr = Kernel_A_gets_A_minus_B_times_C_6(v,pc,pv);CHKERRQ(ierr);
          pv  += bs2;          
        }
        ierr = PetscLogFlops(432*nz+396);CHKERRQ(ierr); /* flops = 2*bs^3*nz + 2*bs^3 - bs2) */
      }
    }

    /* finished row so stick it into b->a */
    /* L part */
    pv   = b->a + bs2*bi[i] ;
    pj   = b->j + bi[i] ;
    nz   = bi[i+1] - bi[i];
    for (j=0; j<nz; j++) {
      ierr = PetscMemcpy(pv+bs2*j,rtmp+bs2*pj[j],bs2*sizeof(MatScalar));CHKERRQ(ierr);
    }

    /* Mark diagonal and invert diagonal for simplier triangular solves */
    pv   = b->a + bs2*bdiag[i];
    pj   = b->j + bdiag[i];
    ierr = PetscMemcpy(pv,rtmp+bs2*pj[0],bs2*sizeof(MatScalar));CHKERRQ(ierr);   
    /* ierr = Kernel_A_gets_inverse_A(bs,pv,v_pivots,v_work);CHKERRQ(ierr); */
    ierr = Kernel_A_gets_inverse_A_6(pv,shift);CHKERRQ(ierr);
      
    /* U part */
    pv = b->a + bs2*(bdiag[i+1]+1);
    pj = b->j + bdiag[i+1]+1;
    nz = bdiag[i] - bdiag[i+1] - 1; 
    for (j=0; j<nz; j++){
      ierr = PetscMemcpy(pv+bs2*j,rtmp+bs2*pj[j],bs2*sizeof(MatScalar));CHKERRQ(ierr);
    }
  }
  ierr = PetscFree2(rtmp,mwork);CHKERRQ(ierr);
  C->ops->solve          = MatSolve_SeqBAIJ_6_NaturalOrdering;
  C->ops->solvetranspose = MatSolveTranspose_SeqBAIJ_6_NaturalOrdering;
  C->assembled = PETSC_TRUE;
  ierr = PetscLogFlops(1.333333333333*6*6*6*n);CHKERRQ(ierr); /* from inverting diagonal blocks */
  PetscFunctionReturn(0);
}
