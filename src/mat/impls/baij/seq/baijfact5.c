
/*
    Factorization code for BAIJ format.
*/
#include <../src/mat/impls/baij/seq/baij.h>
#include <petsc/private/kernels/blockinvert.h>
/*
      Version for when blocks are 7 by 7
*/
PetscErrorCode MatLUFactorNumeric_SeqBAIJ_7_inplace(Mat C,Mat A,const MatFactorInfo *info)
{
  Mat_SeqBAIJ    *a    = (Mat_SeqBAIJ*)A->data,*b = (Mat_SeqBAIJ*)C->data;
  IS             isrow = b->row,isicol = b->icol;
  const PetscInt *r,*ic,*bi = b->i,*bj = b->j,*ajtmp,*diag_offset = b->diag,*ai=a->i,*aj=a->j,*pj,*ajtmpold;
  PetscInt       i,j,n = a->mbs,nz,row,idx;
  MatScalar      *pv,*v,*rtmp,*pc,*w,*x;
  MatScalar      p1,p2,p3,p4,m1,m2,m3,m4,m5,m6,m7,m8,m9,x1,x2,x3,x4;
  MatScalar      p5,p6,p7,p8,p9,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16;
  MatScalar      x17,x18,x19,x20,x21,x22,x23,x24,x25,p10,p11,p12,p13,p14;
  MatScalar      p15,p16,p17,p18,p19,p20,p21,p22,p23,p24,p25,m10,m11,m12;
  MatScalar      m13,m14,m15,m16,m17,m18,m19,m20,m21,m22,m23,m24,m25;
  MatScalar      p26,p27,p28,p29,p30,p31,p32,p33,p34,p35,p36;
  MatScalar      p37,p38,p39,p40,p41,p42,p43,p44,p45,p46,p47,p48,p49;
  MatScalar      x26,x27,x28,x29,x30,x31,x32,x33,x34,x35,x36;
  MatScalar      x37,x38,x39,x40,x41,x42,x43,x44,x45,x46,x47,x48,x49;
  MatScalar      m26,m27,m28,m29,m30,m31,m32,m33,m34,m35,m36;
  MatScalar      m37,m38,m39,m40,m41,m42,m43,m44,m45,m46,m47,m48,m49;
  MatScalar      *ba   = b->a,*aa = a->a;
  PetscReal      shift = info->shiftamount;
  PetscBool      allowzeropivot,zeropivotdetected;

  PetscFunctionBegin;
  allowzeropivot = PetscNot(A->erroriffailure);
  PetscCall(ISGetIndices(isrow,&r));
  PetscCall(ISGetIndices(isicol,&ic));
  PetscCall(PetscMalloc1(49*(n+1),&rtmp));

  for (i=0; i<n; i++) {
    nz    = bi[i+1] - bi[i];
    ajtmp = bj + bi[i];
    for  (j=0; j<nz; j++) {
      x     = rtmp+49*ajtmp[j];
      x[0]  = x[1] = x[2] = x[3] = x[4] = x[5] = x[6] = x[7] = x[8] = x[9] = 0.0;
      x[10] = x[11] = x[12] = x[13] = x[14] = x[15] = x[16] = x[17] = 0.0;
      x[18] = x[19] = x[20] = x[21] = x[22] = x[23] = x[24] = x[25] = 0.0;
      x[26] = x[27] = x[28] = x[29] = x[30] = x[31] = x[32] = x[33] = 0.0;
      x[34] = x[35] = x[36] = x[37] = x[38] = x[39] = x[40] = x[41] = 0.0;
      x[42] = x[43] = x[44] = x[45] = x[46] = x[47] = x[48] = 0.0;
    }
    /* load in initial (unfactored row) */
    idx      = r[i];
    nz       = ai[idx+1] - ai[idx];
    ajtmpold = aj + ai[idx];
    v        = aa + 49*ai[idx];
    for (j=0; j<nz; j++) {
      x     = rtmp+49*ic[ajtmpold[j]];
      x[0]  =  v[0];  x[1] =  v[1];  x[2] =  v[2];  x[3] =  v[3];
      x[4]  =  v[4];  x[5] =  v[5];  x[6] =  v[6];  x[7] =  v[7];
      x[8]  =  v[8];  x[9] =  v[9];  x[10] = v[10]; x[11] = v[11];
      x[12] = v[12]; x[13] = v[13]; x[14] = v[14]; x[15] = v[15];
      x[16] = v[16]; x[17] = v[17]; x[18] = v[18]; x[19] = v[19];
      x[20] = v[20]; x[21] = v[21]; x[22] = v[22]; x[23] = v[23];
      x[24] = v[24]; x[25] = v[25]; x[26] = v[26]; x[27] = v[27];
      x[28] = v[28]; x[29] = v[29]; x[30] = v[30]; x[31] = v[31];
      x[32] = v[32]; x[33] = v[33]; x[34] = v[34]; x[35] = v[35];
      x[36] = v[36]; x[37] = v[37]; x[38] = v[38]; x[39] = v[39];
      x[40] = v[40]; x[41] = v[41]; x[42] = v[42]; x[43] = v[43];
      x[44] = v[44]; x[45] = v[45]; x[46] = v[46]; x[47] = v[47];
      x[48] = v[48];
      v    += 49;
    }
    row = *ajtmp++;
    while (row < i) {
      pc  =  rtmp + 49*row;
      p1  = pc[0];  p2  = pc[1];  p3  = pc[2];  p4  = pc[3];
      p5  = pc[4];  p6  = pc[5];  p7  = pc[6];  p8  = pc[7];
      p9  = pc[8];  p10 = pc[9];  p11 = pc[10]; p12 = pc[11];
      p13 = pc[12]; p14 = pc[13]; p15 = pc[14]; p16 = pc[15];
      p17 = pc[16]; p18 = pc[17]; p19 = pc[18]; p20 = pc[19];
      p21 = pc[20]; p22 = pc[21]; p23 = pc[22]; p24 = pc[23];
      p25 = pc[24]; p26 = pc[25]; p27 = pc[26]; p28 = pc[27];
      p29 = pc[28]; p30 = pc[29]; p31 = pc[30]; p32 = pc[31];
      p33 = pc[32]; p34 = pc[33]; p35 = pc[34]; p36 = pc[35];
      p37 = pc[36]; p38 = pc[37]; p39 = pc[38]; p40 = pc[39];
      p41 = pc[40]; p42 = pc[41]; p43 = pc[42]; p44 = pc[43];
      p45 = pc[44]; p46 = pc[45]; p47 = pc[46]; p48 = pc[47];
      p49 = pc[48];
      if (p1  != 0.0 || p2  != 0.0 || p3  != 0.0 || p4  != 0.0 ||
          p5  != 0.0 || p6  != 0.0 || p7  != 0.0 || p8  != 0.0 ||
          p9  != 0.0 || p10 != 0.0 || p11 != 0.0 || p12 != 0.0 ||
          p13 != 0.0 || p14 != 0.0 || p15 != 0.0 || p16 != 0.0 ||
          p17 != 0.0 || p18 != 0.0 || p19 != 0.0 || p20 != 0.0 ||
          p21 != 0.0 || p22 != 0.0 || p23 != 0.0 || p24 != 0.0 ||
          p25 != 0.0 || p26 != 0.0 || p27 != 0.0 || p28 != 0.0 ||
          p29 != 0.0 || p30 != 0.0 || p31 != 0.0 || p32 != 0.0 ||
          p33 != 0.0 || p34 != 0.0 || p35 != 0.0 || p36 != 0.0 ||
          p37 != 0.0 || p38 != 0.0 || p39 != 0.0 || p40 != 0.0 ||
          p41 != 0.0 || p42 != 0.0 || p43 != 0.0 || p44 != 0.0 ||
          p45 != 0.0 || p46 != 0.0 || p47 != 0.0 || p48 != 0.0 ||
          p49 != 0.0) {
        pv    = ba + 49*diag_offset[row];
        pj    = bj + diag_offset[row] + 1;
        x1    = pv[0];  x2  = pv[1];  x3  = pv[2];  x4  = pv[3];
        x5    = pv[4];  x6  = pv[5];  x7  = pv[6];  x8  = pv[7];
        x9    = pv[8];  x10 = pv[9];  x11 = pv[10]; x12 = pv[11];
        x13   = pv[12]; x14 = pv[13]; x15 = pv[14]; x16 = pv[15];
        x17   = pv[16]; x18 = pv[17]; x19 = pv[18]; x20 = pv[19];
        x21   = pv[20]; x22 = pv[21]; x23 = pv[22]; x24 = pv[23];
        x25   = pv[24]; x26 = pv[25]; x27 = pv[26]; x28 = pv[27];
        x29   = pv[28]; x30 = pv[29]; x31 = pv[30]; x32 = pv[31];
        x33   = pv[32]; x34 = pv[33]; x35 = pv[34]; x36 = pv[35];
        x37   = pv[36]; x38 = pv[37]; x39 = pv[38]; x40 = pv[39];
        x41   = pv[40]; x42 = pv[41]; x43 = pv[42]; x44 = pv[43];
        x45   = pv[44]; x46 = pv[45]; x47 = pv[46]; x48 = pv[47];
        x49   = pv[48];
        pc[0] = m1  = p1*x1  + p8*x2   + p15*x3  + p22*x4  + p29*x5  + p36*x6 + p43*x7;
        pc[1] = m2  = p2*x1  + p9*x2   + p16*x3  + p23*x4  + p30*x5  + p37*x6 + p44*x7;
        pc[2] = m3  = p3*x1  + p10*x2  + p17*x3  + p24*x4  + p31*x5  + p38*x6 + p45*x7;
        pc[3] = m4  = p4*x1  + p11*x2  + p18*x3  + p25*x4  + p32*x5  + p39*x6 + p46*x7;
        pc[4] = m5  = p5*x1  + p12*x2  + p19*x3  + p26*x4  + p33*x5  + p40*x6 + p47*x7;
        pc[5] = m6  = p6*x1  + p13*x2  + p20*x3  + p27*x4  + p34*x5  + p41*x6 + p48*x7;
        pc[6] = m7  = p7*x1  + p14*x2  + p21*x3  + p28*x4  + p35*x5  + p42*x6 + p49*x7;

        pc[7]  = m8  = p1*x8  + p8*x9   + p15*x10 + p22*x11 + p29*x12 + p36*x13 + p43*x14;
        pc[8]  = m9  = p2*x8  + p9*x9   + p16*x10 + p23*x11 + p30*x12 + p37*x13 + p44*x14;
        pc[9]  = m10 = p3*x8  + p10*x9  + p17*x10 + p24*x11 + p31*x12 + p38*x13 + p45*x14;
        pc[10] = m11 = p4*x8  + p11*x9  + p18*x10 + p25*x11 + p32*x12 + p39*x13 + p46*x14;
        pc[11] = m12 = p5*x8  + p12*x9  + p19*x10 + p26*x11 + p33*x12 + p40*x13 + p47*x14;
        pc[12] = m13 = p6*x8  + p13*x9  + p20*x10 + p27*x11 + p34*x12 + p41*x13 + p48*x14;
        pc[13] = m14 = p7*x8  + p14*x9  + p21*x10 + p28*x11 + p35*x12 + p42*x13 + p49*x14;

        pc[14] = m15 = p1*x15 + p8*x16  + p15*x17 + p22*x18 + p29*x19 + p36*x20 + p43*x21;
        pc[15] = m16 = p2*x15 + p9*x16  + p16*x17 + p23*x18 + p30*x19 + p37*x20 + p44*x21;
        pc[16] = m17 = p3*x15 + p10*x16 + p17*x17 + p24*x18 + p31*x19 + p38*x20 + p45*x21;
        pc[17] = m18 = p4*x15 + p11*x16 + p18*x17 + p25*x18 + p32*x19 + p39*x20 + p46*x21;
        pc[18] = m19 = p5*x15 + p12*x16 + p19*x17 + p26*x18 + p33*x19 + p40*x20 + p47*x21;
        pc[19] = m20 = p6*x15 + p13*x16 + p20*x17 + p27*x18 + p34*x19 + p41*x20 + p48*x21;
        pc[20] = m21 = p7*x15 + p14*x16 + p21*x17 + p28*x18 + p35*x19 + p42*x20 + p49*x21;

        pc[21] = m22 = p1*x22 + p8*x23  + p15*x24 + p22*x25 + p29*x26 + p36*x27 + p43*x28;
        pc[22] = m23 = p2*x22 + p9*x23  + p16*x24 + p23*x25 + p30*x26 + p37*x27 + p44*x28;
        pc[23] = m24 = p3*x22 + p10*x23 + p17*x24 + p24*x25 + p31*x26 + p38*x27 + p45*x28;
        pc[24] = m25 = p4*x22 + p11*x23 + p18*x24 + p25*x25 + p32*x26 + p39*x27 + p46*x28;
        pc[25] = m26 = p5*x22 + p12*x23 + p19*x24 + p26*x25 + p33*x26 + p40*x27 + p47*x28;
        pc[26] = m27 = p6*x22 + p13*x23 + p20*x24 + p27*x25 + p34*x26 + p41*x27 + p48*x28;
        pc[27] = m28 = p7*x22 + p14*x23 + p21*x24 + p28*x25 + p35*x26 + p42*x27 + p49*x28;

        pc[28] = m29 = p1*x29 + p8*x30  + p15*x31 + p22*x32 + p29*x33 + p36*x34 + p43*x35;
        pc[29] = m30 = p2*x29 + p9*x30  + p16*x31 + p23*x32 + p30*x33 + p37*x34 + p44*x35;
        pc[30] = m31 = p3*x29 + p10*x30 + p17*x31 + p24*x32 + p31*x33 + p38*x34 + p45*x35;
        pc[31] = m32 = p4*x29 + p11*x30 + p18*x31 + p25*x32 + p32*x33 + p39*x34 + p46*x35;
        pc[32] = m33 = p5*x29 + p12*x30 + p19*x31 + p26*x32 + p33*x33 + p40*x34 + p47*x35;
        pc[33] = m34 = p6*x29 + p13*x30 + p20*x31 + p27*x32 + p34*x33 + p41*x34 + p48*x35;
        pc[34] = m35 = p7*x29 + p14*x30 + p21*x31 + p28*x32 + p35*x33 + p42*x34 + p49*x35;

        pc[35] = m36 = p1*x36 + p8*x37  + p15*x38 + p22*x39 + p29*x40 + p36*x41 + p43*x42;
        pc[36] = m37 = p2*x36 + p9*x37  + p16*x38 + p23*x39 + p30*x40 + p37*x41 + p44*x42;
        pc[37] = m38 = p3*x36 + p10*x37 + p17*x38 + p24*x39 + p31*x40 + p38*x41 + p45*x42;
        pc[38] = m39 = p4*x36 + p11*x37 + p18*x38 + p25*x39 + p32*x40 + p39*x41 + p46*x42;
        pc[39] = m40 = p5*x36 + p12*x37 + p19*x38 + p26*x39 + p33*x40 + p40*x41 + p47*x42;
        pc[40] = m41 = p6*x36 + p13*x37 + p20*x38 + p27*x39 + p34*x40 + p41*x41 + p48*x42;
        pc[41] = m42 = p7*x36 + p14*x37 + p21*x38 + p28*x39 + p35*x40 + p42*x41 + p49*x42;

        pc[42] = m43 = p1*x43 + p8*x44  + p15*x45 + p22*x46 + p29*x47 + p36*x48 + p43*x49;
        pc[43] = m44 = p2*x43 + p9*x44  + p16*x45 + p23*x46 + p30*x47 + p37*x48 + p44*x49;
        pc[44] = m45 = p3*x43 + p10*x44 + p17*x45 + p24*x46 + p31*x47 + p38*x48 + p45*x49;
        pc[45] = m46 = p4*x43 + p11*x44 + p18*x45 + p25*x46 + p32*x47 + p39*x48 + p46*x49;
        pc[46] = m47 = p5*x43 + p12*x44 + p19*x45 + p26*x46 + p33*x47 + p40*x48 + p47*x49;
        pc[47] = m48 = p6*x43 + p13*x44 + p20*x45 + p27*x46 + p34*x47 + p41*x48 + p48*x49;
        pc[48] = m49 = p7*x43 + p14*x44 + p21*x45 + p28*x46 + p35*x47 + p42*x48 + p49*x49;

        nz  = bi[row+1] - diag_offset[row] - 1;
        pv += 49;
        for (j=0; j<nz; j++) {
          x1    = pv[0];  x2  = pv[1];  x3  = pv[2];  x4  = pv[3];
          x5    = pv[4];  x6  = pv[5];  x7  = pv[6];  x8  = pv[7];
          x9    = pv[8];  x10 = pv[9];  x11 = pv[10]; x12 = pv[11];
          x13   = pv[12]; x14 = pv[13]; x15 = pv[14]; x16 = pv[15];
          x17   = pv[16]; x18 = pv[17]; x19 = pv[18]; x20 = pv[19];
          x21   = pv[20]; x22 = pv[21]; x23 = pv[22]; x24 = pv[23];
          x25   = pv[24]; x26 = pv[25]; x27 = pv[26]; x28 = pv[27];
          x29   = pv[28]; x30 = pv[29]; x31 = pv[30]; x32 = pv[31];
          x33   = pv[32]; x34 = pv[33]; x35 = pv[34]; x36 = pv[35];
          x37   = pv[36]; x38 = pv[37]; x39 = pv[38]; x40 = pv[39];
          x41   = pv[40]; x42 = pv[41]; x43 = pv[42]; x44 = pv[43];
          x45   = pv[44]; x46 = pv[45]; x47 = pv[46]; x48 = pv[47];
          x49   = pv[48];
          x     = rtmp + 49*pj[j];
          x[0] -= m1*x1  + m8*x2   + m15*x3  + m22*x4  + m29*x5  + m36*x6 + m43*x7;
          x[1] -= m2*x1  + m9*x2   + m16*x3  + m23*x4  + m30*x5  + m37*x6 + m44*x7;
          x[2] -= m3*x1  + m10*x2  + m17*x3  + m24*x4  + m31*x5  + m38*x6 + m45*x7;
          x[3] -= m4*x1  + m11*x2  + m18*x3  + m25*x4  + m32*x5  + m39*x6 + m46*x7;
          x[4] -= m5*x1  + m12*x2  + m19*x3  + m26*x4  + m33*x5  + m40*x6 + m47*x7;
          x[5] -= m6*x1  + m13*x2  + m20*x3  + m27*x4  + m34*x5  + m41*x6 + m48*x7;
          x[6] -= m7*x1  + m14*x2  + m21*x3  + m28*x4  + m35*x5  + m42*x6 + m49*x7;

          x[7]  -= m1*x8  + m8*x9   + m15*x10 + m22*x11 + m29*x12 + m36*x13 + m43*x14;
          x[8]  -= m2*x8  + m9*x9   + m16*x10 + m23*x11 + m30*x12 + m37*x13 + m44*x14;
          x[9]  -= m3*x8  + m10*x9  + m17*x10 + m24*x11 + m31*x12 + m38*x13 + m45*x14;
          x[10] -= m4*x8  + m11*x9  + m18*x10 + m25*x11 + m32*x12 + m39*x13 + m46*x14;
          x[11] -= m5*x8  + m12*x9  + m19*x10 + m26*x11 + m33*x12 + m40*x13 + m47*x14;
          x[12] -= m6*x8  + m13*x9  + m20*x10 + m27*x11 + m34*x12 + m41*x13 + m48*x14;
          x[13] -= m7*x8  + m14*x9  + m21*x10 + m28*x11 + m35*x12 + m42*x13 + m49*x14;

          x[14] -= m1*x15 + m8*x16  + m15*x17 + m22*x18 + m29*x19 + m36*x20 + m43*x21;
          x[15] -= m2*x15 + m9*x16  + m16*x17 + m23*x18 + m30*x19 + m37*x20 + m44*x21;
          x[16] -= m3*x15 + m10*x16 + m17*x17 + m24*x18 + m31*x19 + m38*x20 + m45*x21;
          x[17] -= m4*x15 + m11*x16 + m18*x17 + m25*x18 + m32*x19 + m39*x20 + m46*x21;
          x[18] -= m5*x15 + m12*x16 + m19*x17 + m26*x18 + m33*x19 + m40*x20 + m47*x21;
          x[19] -= m6*x15 + m13*x16 + m20*x17 + m27*x18 + m34*x19 + m41*x20 + m48*x21;
          x[20] -= m7*x15 + m14*x16 + m21*x17 + m28*x18 + m35*x19 + m42*x20 + m49*x21;

          x[21] -= m1*x22 + m8*x23  + m15*x24 + m22*x25 + m29*x26 + m36*x27 + m43*x28;
          x[22] -= m2*x22 + m9*x23  + m16*x24 + m23*x25 + m30*x26 + m37*x27 + m44*x28;
          x[23] -= m3*x22 + m10*x23 + m17*x24 + m24*x25 + m31*x26 + m38*x27 + m45*x28;
          x[24] -= m4*x22 + m11*x23 + m18*x24 + m25*x25 + m32*x26 + m39*x27 + m46*x28;
          x[25] -= m5*x22 + m12*x23 + m19*x24 + m26*x25 + m33*x26 + m40*x27 + m47*x28;
          x[26] -= m6*x22 + m13*x23 + m20*x24 + m27*x25 + m34*x26 + m41*x27 + m48*x28;
          x[27] -= m7*x22 + m14*x23 + m21*x24 + m28*x25 + m35*x26 + m42*x27 + m49*x28;

          x[28] -= m1*x29 + m8*x30  + m15*x31 + m22*x32 + m29*x33 + m36*x34 + m43*x35;
          x[29] -= m2*x29 + m9*x30  + m16*x31 + m23*x32 + m30*x33 + m37*x34 + m44*x35;
          x[30] -= m3*x29 + m10*x30 + m17*x31 + m24*x32 + m31*x33 + m38*x34 + m45*x35;
          x[31] -= m4*x29 + m11*x30 + m18*x31 + m25*x32 + m32*x33 + m39*x34 + m46*x35;
          x[32] -= m5*x29 + m12*x30 + m19*x31 + m26*x32 + m33*x33 + m40*x34 + m47*x35;
          x[33] -= m6*x29 + m13*x30 + m20*x31 + m27*x32 + m34*x33 + m41*x34 + m48*x35;
          x[34] -= m7*x29 + m14*x30 + m21*x31 + m28*x32 + m35*x33 + m42*x34 + m49*x35;

          x[35] -= m1*x36 + m8*x37  + m15*x38 + m22*x39 + m29*x40 + m36*x41 + m43*x42;
          x[36] -= m2*x36 + m9*x37  + m16*x38 + m23*x39 + m30*x40 + m37*x41 + m44*x42;
          x[37] -= m3*x36 + m10*x37 + m17*x38 + m24*x39 + m31*x40 + m38*x41 + m45*x42;
          x[38] -= m4*x36 + m11*x37 + m18*x38 + m25*x39 + m32*x40 + m39*x41 + m46*x42;
          x[39] -= m5*x36 + m12*x37 + m19*x38 + m26*x39 + m33*x40 + m40*x41 + m47*x42;
          x[40] -= m6*x36 + m13*x37 + m20*x38 + m27*x39 + m34*x40 + m41*x41 + m48*x42;
          x[41] -= m7*x36 + m14*x37 + m21*x38 + m28*x39 + m35*x40 + m42*x41 + m49*x42;

          x[42] -= m1*x43 + m8*x44  + m15*x45 + m22*x46 + m29*x47 + m36*x48 + m43*x49;
          x[43] -= m2*x43 + m9*x44  + m16*x45 + m23*x46 + m30*x47 + m37*x48 + m44*x49;
          x[44] -= m3*x43 + m10*x44 + m17*x45 + m24*x46 + m31*x47 + m38*x48 + m45*x49;
          x[45] -= m4*x43 + m11*x44 + m18*x45 + m25*x46 + m32*x47 + m39*x48 + m46*x49;
          x[46] -= m5*x43 + m12*x44 + m19*x45 + m26*x46 + m33*x47 + m40*x48 + m47*x49;
          x[47] -= m6*x43 + m13*x44 + m20*x45 + m27*x46 + m34*x47 + m41*x48 + m48*x49;
          x[48] -= m7*x43 + m14*x44 + m21*x45 + m28*x46 + m35*x47 + m42*x48 + m49*x49;
          pv    += 49;
        }
        PetscCall(PetscLogFlops(686.0*nz+637.0));
      }
      row = *ajtmp++;
    }
    /* finished row so stick it into b->a */
    pv = ba + 49*bi[i];
    pj = bj + bi[i];
    nz = bi[i+1] - bi[i];
    for (j=0; j<nz; j++) {
      x      = rtmp+49*pj[j];
      pv[0]  = x[0];  pv[1]  = x[1];  pv[2]  = x[2];  pv[3]  = x[3];
      pv[4]  = x[4];  pv[5]  = x[5];  pv[6]  = x[6];  pv[7]  = x[7];
      pv[8]  = x[8];  pv[9]  = x[9];  pv[10] = x[10]; pv[11] = x[11];
      pv[12] = x[12]; pv[13] = x[13]; pv[14] = x[14]; pv[15] = x[15];
      pv[16] = x[16]; pv[17] = x[17]; pv[18] = x[18]; pv[19] = x[19];
      pv[20] = x[20]; pv[21] = x[21]; pv[22] = x[22]; pv[23] = x[23];
      pv[24] = x[24]; pv[25] = x[25]; pv[26] = x[26]; pv[27] = x[27];
      pv[28] = x[28]; pv[29] = x[29]; pv[30] = x[30]; pv[31] = x[31];
      pv[32] = x[32]; pv[33] = x[33]; pv[34] = x[34]; pv[35] = x[35];
      pv[36] = x[36]; pv[37] = x[37]; pv[38] = x[38]; pv[39] = x[39];
      pv[40] = x[40]; pv[41] = x[41]; pv[42] = x[42]; pv[43] = x[43];
      pv[44] = x[44]; pv[45] = x[45]; pv[46] = x[46]; pv[47] = x[47];
      pv[48] = x[48];
      pv    += 49;
    }
    /* invert diagonal block */
    w    = ba + 49*diag_offset[i];
    PetscCall(PetscKernel_A_gets_inverse_A_7(w,shift,allowzeropivot,&zeropivotdetected));
    if (zeropivotdetected) C->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;
  }

  PetscCall(PetscFree(rtmp));
  PetscCall(ISRestoreIndices(isicol,&ic));
  PetscCall(ISRestoreIndices(isrow,&r));

  C->ops->solve          = MatSolve_SeqBAIJ_7_inplace;
  C->ops->solvetranspose = MatSolveTranspose_SeqBAIJ_7_inplace;
  C->assembled           = PETSC_TRUE;

  PetscCall(PetscLogFlops(1.333333333333*7*7*7*b->mbs)); /* from inverting diagonal blocks */
  PetscFunctionReturn(0);
}

PetscErrorCode MatLUFactorNumeric_SeqBAIJ_7(Mat B,Mat A,const MatFactorInfo *info)
{
  Mat            C     =B;
  Mat_SeqBAIJ    *a    =(Mat_SeqBAIJ*)A->data,*b=(Mat_SeqBAIJ*)C->data;
  IS             isrow = b->row,isicol = b->icol;
  const PetscInt *r,*ic;
  PetscInt       i,j,k,nz,nzL,row;
  const PetscInt n=a->mbs,*ai=a->i,*aj=a->j,*bi=b->i,*bj=b->j;
  const PetscInt *ajtmp,*bjtmp,*bdiag=b->diag,*pj,bs2=a->bs2;
  MatScalar      *rtmp,*pc,*mwork,*v,*pv,*aa=a->a;
  PetscInt       flg;
  PetscReal      shift = info->shiftamount;
  PetscBool      allowzeropivot,zeropivotdetected;

  PetscFunctionBegin;
  allowzeropivot = PetscNot(A->erroriffailure);
  PetscCall(ISGetIndices(isrow,&r));
  PetscCall(ISGetIndices(isicol,&ic));

  /* generate work space needed by the factorization */
  PetscCall(PetscMalloc2(bs2*n,&rtmp,bs2,&mwork));
  PetscCall(PetscArrayzero(rtmp,bs2*n));

  for (i=0; i<n; i++) {
    /* zero rtmp */
    /* L part */
    nz    = bi[i+1] - bi[i];
    bjtmp = bj + bi[i];
    for  (j=0; j<nz; j++) {
      PetscCall(PetscArrayzero(rtmp+bs2*bjtmp[j],bs2));
    }

    /* U part */
    nz    = bdiag[i] - bdiag[i+1];
    bjtmp = bj + bdiag[i+1]+1;
    for  (j=0; j<nz; j++) {
      PetscCall(PetscArrayzero(rtmp+bs2*bjtmp[j],bs2));
    }

    /* load in initial (unfactored row) */
    nz    = ai[r[i]+1] - ai[r[i]];
    ajtmp = aj + ai[r[i]];
    v     = aa + bs2*ai[r[i]];
    for (j=0; j<nz; j++) {
      PetscCall(PetscArraycpy(rtmp+bs2*ic[ajtmp[j]],v+bs2*j,bs2));
    }

    /* elimination */
    bjtmp = bj + bi[i];
    nzL   = bi[i+1] - bi[i];
    for (k=0; k < nzL; k++) {
      row = bjtmp[k];
      pc  = rtmp + bs2*row;
      for (flg=0,j=0; j<bs2; j++) {
        if (pc[j]!=0.0) {
          flg = 1;
          break;
        }
      }
      if (flg) {
        pv = b->a + bs2*bdiag[row];
        /* PetscKernel_A_gets_A_times_B(bs,pc,pv,mwork); *pc = *pc * (*pv); */
        PetscCall(PetscKernel_A_gets_A_times_B_7(pc,pv,mwork));

        pj = b->j + bdiag[row+1]+1; /* beginning of U(row,:) */
        pv = b->a + bs2*(bdiag[row+1]+1);
        nz = bdiag[row] - bdiag[row+1] - 1; /* num of entries inU(row,:), excluding diag */
        for (j=0; j<nz; j++) {
          /* PetscKernel_A_gets_A_minus_B_times_C(bs,rtmp+bs2*pj[j],pc,pv+bs2*j); */
          /* rtmp+bs2*pj[j] = rtmp+bs2*pj[j] - (*pc)*(pv+bs2*j) */
          v    = rtmp + bs2*pj[j];
          PetscCall(PetscKernel_A_gets_A_minus_B_times_C_7(v,pc,pv));
          pv  += bs2;
        }
        PetscCall(PetscLogFlops(686.0*nz+637)); /* flops = 2*bs^3*nz + 2*bs^3 - bs2) */
      }
    }

    /* finished row so stick it into b->a */
    /* L part */
    pv = b->a + bs2*bi[i];
    pj = b->j + bi[i];
    nz = bi[i+1] - bi[i];
    for (j=0; j<nz; j++) {
      PetscCall(PetscArraycpy(pv+bs2*j,rtmp+bs2*pj[j],bs2));
    }

    /* Mark diagonal and invert diagonal for simpler triangular solves */
    pv   = b->a + bs2*bdiag[i];
    pj   = b->j + bdiag[i];
    PetscCall(PetscArraycpy(pv,rtmp+bs2*pj[0],bs2));
    PetscCall(PetscKernel_A_gets_inverse_A_7(pv,shift,allowzeropivot,&zeropivotdetected));
    if (zeropivotdetected) C->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;

    /* U part */
    pv = b->a + bs2*(bdiag[i+1]+1);
    pj = b->j + bdiag[i+1]+1;
    nz = bdiag[i] - bdiag[i+1] - 1;
    for (j=0; j<nz; j++) {
      PetscCall(PetscArraycpy(pv+bs2*j,rtmp+bs2*pj[j],bs2));
    }
  }

  PetscCall(PetscFree2(rtmp,mwork));
  PetscCall(ISRestoreIndices(isicol,&ic));
  PetscCall(ISRestoreIndices(isrow,&r));

  C->ops->solve          = MatSolve_SeqBAIJ_7;
  C->ops->solvetranspose = MatSolveTranspose_SeqBAIJ_7;
  C->assembled           = PETSC_TRUE;

  PetscCall(PetscLogFlops(1.333333333333*7*7*7*n)); /* from inverting diagonal blocks */
  PetscFunctionReturn(0);
}

PetscErrorCode MatLUFactorNumeric_SeqBAIJ_7_NaturalOrdering_inplace(Mat C,Mat A,const MatFactorInfo *info)
{
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data,*b = (Mat_SeqBAIJ*)C->data;
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
  MatScalar      p37,p38,p39,p40,p41,p42,p43,p44,p45,p46,p47,p48,p49;
  MatScalar      x26,x27,x28,x29,x30,x31,x32,x33,x34,x35,x36;
  MatScalar      x37,x38,x39,x40,x41,x42,x43,x44,x45,x46,x47,x48,x49;
  MatScalar      m26,m27,m28,m29,m30,m31,m32,m33,m34,m35,m36;
  MatScalar      m37,m38,m39,m40,m41,m42,m43,m44,m45,m46,m47,m48,m49;
  MatScalar      *ba   = b->a,*aa = a->a;
  PetscReal      shift = info->shiftamount;
  PetscBool      allowzeropivot,zeropivotdetected;

  PetscFunctionBegin;
  allowzeropivot = PetscNot(A->erroriffailure);
  PetscCall(PetscMalloc1(49*(n+1),&rtmp));
  for (i=0; i<n; i++) {
    nz    = bi[i+1] - bi[i];
    ajtmp = bj + bi[i];
    for  (j=0; j<nz; j++) {
      x     = rtmp+49*ajtmp[j];
      x[0]  = x[1] = x[2] = x[3] = x[4] = x[5] = x[6] = x[7] = x[8] = x[9] = 0.0;
      x[10] = x[11] = x[12] = x[13] = x[14] = x[15] = x[16] = x[17] = 0.0;
      x[18] = x[19] = x[20] = x[21] = x[22] = x[23] = x[24] = x[25] = 0.0;
      x[26] = x[27] = x[28] = x[29] = x[30] = x[31] = x[32] = x[33] = 0.0;
      x[34] = x[35] = x[36] = x[37] = x[38] = x[39] = x[40] = x[41] = 0.0;
      x[42] = x[43] = x[44] = x[45] = x[46] = x[47] = x[48] = 0.0;
    }
    /* load in initial (unfactored row) */
    nz       = ai[i+1] - ai[i];
    ajtmpold = aj + ai[i];
    v        = aa + 49*ai[i];
    for (j=0; j<nz; j++) {
      x     = rtmp+49*ajtmpold[j];
      x[0]  =  v[0];  x[1] =  v[1];  x[2] =  v[2];  x[3] =  v[3];
      x[4]  =  v[4];  x[5] =  v[5];  x[6] =  v[6];  x[7] =  v[7];
      x[8]  =  v[8];  x[9] =  v[9];  x[10] = v[10]; x[11] = v[11];
      x[12] = v[12]; x[13] = v[13]; x[14] = v[14]; x[15] = v[15];
      x[16] = v[16]; x[17] = v[17]; x[18] = v[18]; x[19] = v[19];
      x[20] = v[20]; x[21] = v[21]; x[22] = v[22]; x[23] = v[23];
      x[24] = v[24]; x[25] = v[25]; x[26] = v[26]; x[27] = v[27];
      x[28] = v[28]; x[29] = v[29]; x[30] = v[30]; x[31] = v[31];
      x[32] = v[32]; x[33] = v[33]; x[34] = v[34]; x[35] = v[35];
      x[36] = v[36]; x[37] = v[37]; x[38] = v[38]; x[39] = v[39];
      x[40] = v[40]; x[41] = v[41]; x[42] = v[42]; x[43] = v[43];
      x[44] = v[44]; x[45] = v[45]; x[46] = v[46]; x[47] = v[47];
      x[48] = v[48];
      v    += 49;
    }
    row = *ajtmp++;
    while (row < i) {
      pc  = rtmp + 49*row;
      p1  = pc[0];  p2  = pc[1];  p3  = pc[2];  p4  = pc[3];
      p5  = pc[4];  p6  = pc[5];  p7  = pc[6];  p8  = pc[7];
      p9  = pc[8];  p10 = pc[9];  p11 = pc[10]; p12 = pc[11];
      p13 = pc[12]; p14 = pc[13]; p15 = pc[14]; p16 = pc[15];
      p17 = pc[16]; p18 = pc[17]; p19 = pc[18]; p20 = pc[19];
      p21 = pc[20]; p22 = pc[21]; p23 = pc[22]; p24 = pc[23];
      p25 = pc[24]; p26 = pc[25]; p27 = pc[26]; p28 = pc[27];
      p29 = pc[28]; p30 = pc[29]; p31 = pc[30]; p32 = pc[31];
      p33 = pc[32]; p34 = pc[33]; p35 = pc[34]; p36 = pc[35];
      p37 = pc[36]; p38 = pc[37]; p39 = pc[38]; p40 = pc[39];
      p41 = pc[40]; p42 = pc[41]; p43 = pc[42]; p44 = pc[43];
      p45 = pc[44]; p46 = pc[45]; p47 = pc[46]; p48 = pc[47];
      p49 = pc[48];
      if (p1  != 0.0 || p2  != 0.0 || p3  != 0.0 || p4  != 0.0 ||
          p5  != 0.0 || p6  != 0.0 || p7  != 0.0 || p8  != 0.0 ||
          p9  != 0.0 || p10 != 0.0 || p11 != 0.0 || p12 != 0.0 ||
          p13 != 0.0 || p14 != 0.0 || p15 != 0.0 || p16 != 0.0 ||
          p17 != 0.0 || p18 != 0.0 || p19 != 0.0 || p20 != 0.0 ||
          p21 != 0.0 || p22 != 0.0 || p23 != 0.0 || p24 != 0.0 ||
          p25 != 0.0 || p26 != 0.0 || p27 != 0.0 || p28 != 0.0 ||
          p29 != 0.0 || p30 != 0.0 || p31 != 0.0 || p32 != 0.0 ||
          p33 != 0.0 || p34 != 0.0 || p35 != 0.0 || p36 != 0.0 ||
          p37 != 0.0 || p38 != 0.0 || p39 != 0.0 || p40 != 0.0 ||
          p41 != 0.0 || p42 != 0.0 || p43 != 0.0 || p44 != 0.0 ||
          p45 != 0.0 || p46 != 0.0 || p47 != 0.0 || p48 != 0.0 ||
          p49 != 0.0) {
        pv    = ba + 49*diag_offset[row];
        pj    = bj + diag_offset[row] + 1;
        x1    = pv[0];  x2  = pv[1];  x3  = pv[2];  x4  = pv[3];
        x5    = pv[4];  x6  = pv[5];  x7  = pv[6];  x8  = pv[7];
        x9    = pv[8];  x10 = pv[9];  x11 = pv[10]; x12 = pv[11];
        x13   = pv[12]; x14 = pv[13]; x15 = pv[14]; x16 = pv[15];
        x17   = pv[16]; x18 = pv[17]; x19 = pv[18]; x20 = pv[19];
        x21   = pv[20]; x22 = pv[21]; x23 = pv[22]; x24 = pv[23];
        x25   = pv[24]; x26 = pv[25]; x27 = pv[26]; x28 = pv[27];
        x29   = pv[28]; x30 = pv[29]; x31 = pv[30]; x32 = pv[31];
        x33   = pv[32]; x34 = pv[33]; x35 = pv[34]; x36 = pv[35];
        x37   = pv[36]; x38 = pv[37]; x39 = pv[38]; x40 = pv[39];
        x41   = pv[40]; x42 = pv[41]; x43 = pv[42]; x44 = pv[43];
        x45   = pv[44]; x46 = pv[45]; x47 = pv[46]; x48 = pv[47];
        x49   = pv[48];
        pc[0] = m1  = p1*x1  + p8*x2   + p15*x3  + p22*x4  + p29*x5  + p36*x6 + p43*x7;
        pc[1] = m2  = p2*x1  + p9*x2   + p16*x3  + p23*x4  + p30*x5  + p37*x6 + p44*x7;
        pc[2] = m3  = p3*x1  + p10*x2  + p17*x3  + p24*x4  + p31*x5  + p38*x6 + p45*x7;
        pc[3] = m4  = p4*x1  + p11*x2  + p18*x3  + p25*x4  + p32*x5  + p39*x6 + p46*x7;
        pc[4] = m5  = p5*x1  + p12*x2  + p19*x3  + p26*x4  + p33*x5  + p40*x6 + p47*x7;
        pc[5] = m6  = p6*x1  + p13*x2  + p20*x3  + p27*x4  + p34*x5  + p41*x6 + p48*x7;
        pc[6] = m7  = p7*x1  + p14*x2  + p21*x3  + p28*x4  + p35*x5  + p42*x6 + p49*x7;

        pc[7]  = m8  = p1*x8  + p8*x9   + p15*x10 + p22*x11 + p29*x12 + p36*x13 + p43*x14;
        pc[8]  = m9  = p2*x8  + p9*x9   + p16*x10 + p23*x11 + p30*x12 + p37*x13 + p44*x14;
        pc[9]  = m10 = p3*x8  + p10*x9  + p17*x10 + p24*x11 + p31*x12 + p38*x13 + p45*x14;
        pc[10] = m11 = p4*x8  + p11*x9  + p18*x10 + p25*x11 + p32*x12 + p39*x13 + p46*x14;
        pc[11] = m12 = p5*x8  + p12*x9  + p19*x10 + p26*x11 + p33*x12 + p40*x13 + p47*x14;
        pc[12] = m13 = p6*x8  + p13*x9  + p20*x10 + p27*x11 + p34*x12 + p41*x13 + p48*x14;
        pc[13] = m14 = p7*x8  + p14*x9  + p21*x10 + p28*x11 + p35*x12 + p42*x13 + p49*x14;

        pc[14] = m15 = p1*x15 + p8*x16  + p15*x17 + p22*x18 + p29*x19 + p36*x20 + p43*x21;
        pc[15] = m16 = p2*x15 + p9*x16  + p16*x17 + p23*x18 + p30*x19 + p37*x20 + p44*x21;
        pc[16] = m17 = p3*x15 + p10*x16 + p17*x17 + p24*x18 + p31*x19 + p38*x20 + p45*x21;
        pc[17] = m18 = p4*x15 + p11*x16 + p18*x17 + p25*x18 + p32*x19 + p39*x20 + p46*x21;
        pc[18] = m19 = p5*x15 + p12*x16 + p19*x17 + p26*x18 + p33*x19 + p40*x20 + p47*x21;
        pc[19] = m20 = p6*x15 + p13*x16 + p20*x17 + p27*x18 + p34*x19 + p41*x20 + p48*x21;
        pc[20] = m21 = p7*x15 + p14*x16 + p21*x17 + p28*x18 + p35*x19 + p42*x20 + p49*x21;

        pc[21] = m22 = p1*x22 + p8*x23  + p15*x24 + p22*x25 + p29*x26 + p36*x27 + p43*x28;
        pc[22] = m23 = p2*x22 + p9*x23  + p16*x24 + p23*x25 + p30*x26 + p37*x27 + p44*x28;
        pc[23] = m24 = p3*x22 + p10*x23 + p17*x24 + p24*x25 + p31*x26 + p38*x27 + p45*x28;
        pc[24] = m25 = p4*x22 + p11*x23 + p18*x24 + p25*x25 + p32*x26 + p39*x27 + p46*x28;
        pc[25] = m26 = p5*x22 + p12*x23 + p19*x24 + p26*x25 + p33*x26 + p40*x27 + p47*x28;
        pc[26] = m27 = p6*x22 + p13*x23 + p20*x24 + p27*x25 + p34*x26 + p41*x27 + p48*x28;
        pc[27] = m28 = p7*x22 + p14*x23 + p21*x24 + p28*x25 + p35*x26 + p42*x27 + p49*x28;

        pc[28] = m29 = p1*x29 + p8*x30  + p15*x31 + p22*x32 + p29*x33 + p36*x34 + p43*x35;
        pc[29] = m30 = p2*x29 + p9*x30  + p16*x31 + p23*x32 + p30*x33 + p37*x34 + p44*x35;
        pc[30] = m31 = p3*x29 + p10*x30 + p17*x31 + p24*x32 + p31*x33 + p38*x34 + p45*x35;
        pc[31] = m32 = p4*x29 + p11*x30 + p18*x31 + p25*x32 + p32*x33 + p39*x34 + p46*x35;
        pc[32] = m33 = p5*x29 + p12*x30 + p19*x31 + p26*x32 + p33*x33 + p40*x34 + p47*x35;
        pc[33] = m34 = p6*x29 + p13*x30 + p20*x31 + p27*x32 + p34*x33 + p41*x34 + p48*x35;
        pc[34] = m35 = p7*x29 + p14*x30 + p21*x31 + p28*x32 + p35*x33 + p42*x34 + p49*x35;

        pc[35] = m36 = p1*x36 + p8*x37  + p15*x38 + p22*x39 + p29*x40 + p36*x41 + p43*x42;
        pc[36] = m37 = p2*x36 + p9*x37  + p16*x38 + p23*x39 + p30*x40 + p37*x41 + p44*x42;
        pc[37] = m38 = p3*x36 + p10*x37 + p17*x38 + p24*x39 + p31*x40 + p38*x41 + p45*x42;
        pc[38] = m39 = p4*x36 + p11*x37 + p18*x38 + p25*x39 + p32*x40 + p39*x41 + p46*x42;
        pc[39] = m40 = p5*x36 + p12*x37 + p19*x38 + p26*x39 + p33*x40 + p40*x41 + p47*x42;
        pc[40] = m41 = p6*x36 + p13*x37 + p20*x38 + p27*x39 + p34*x40 + p41*x41 + p48*x42;
        pc[41] = m42 = p7*x36 + p14*x37 + p21*x38 + p28*x39 + p35*x40 + p42*x41 + p49*x42;

        pc[42] = m43 = p1*x43 + p8*x44  + p15*x45 + p22*x46 + p29*x47 + p36*x48 + p43*x49;
        pc[43] = m44 = p2*x43 + p9*x44  + p16*x45 + p23*x46 + p30*x47 + p37*x48 + p44*x49;
        pc[44] = m45 = p3*x43 + p10*x44 + p17*x45 + p24*x46 + p31*x47 + p38*x48 + p45*x49;
        pc[45] = m46 = p4*x43 + p11*x44 + p18*x45 + p25*x46 + p32*x47 + p39*x48 + p46*x49;
        pc[46] = m47 = p5*x43 + p12*x44 + p19*x45 + p26*x46 + p33*x47 + p40*x48 + p47*x49;
        pc[47] = m48 = p6*x43 + p13*x44 + p20*x45 + p27*x46 + p34*x47 + p41*x48 + p48*x49;
        pc[48] = m49 = p7*x43 + p14*x44 + p21*x45 + p28*x46 + p35*x47 + p42*x48 + p49*x49;

        nz  = bi[row+1] - diag_offset[row] - 1;
        pv += 49;
        for (j=0; j<nz; j++) {
          x1    = pv[0];  x2  = pv[1];  x3  = pv[2];  x4  = pv[3];
          x5    = pv[4];  x6  = pv[5];  x7  = pv[6];  x8  = pv[7];
          x9    = pv[8];  x10 = pv[9];  x11 = pv[10]; x12 = pv[11];
          x13   = pv[12]; x14 = pv[13]; x15 = pv[14]; x16 = pv[15];
          x17   = pv[16]; x18 = pv[17]; x19 = pv[18]; x20 = pv[19];
          x21   = pv[20]; x22 = pv[21]; x23 = pv[22]; x24 = pv[23];
          x25   = pv[24]; x26 = pv[25]; x27 = pv[26]; x28 = pv[27];
          x29   = pv[28]; x30 = pv[29]; x31 = pv[30]; x32 = pv[31];
          x33   = pv[32]; x34 = pv[33]; x35 = pv[34]; x36 = pv[35];
          x37   = pv[36]; x38 = pv[37]; x39 = pv[38]; x40 = pv[39];
          x41   = pv[40]; x42 = pv[41]; x43 = pv[42]; x44 = pv[43];
          x45   = pv[44]; x46 = pv[45]; x47 = pv[46]; x48 = pv[47];
          x49   = pv[48];
          x     = rtmp + 49*pj[j];
          x[0] -= m1*x1  + m8*x2   + m15*x3  + m22*x4  + m29*x5  + m36*x6 + m43*x7;
          x[1] -= m2*x1  + m9*x2   + m16*x3  + m23*x4  + m30*x5  + m37*x6 + m44*x7;
          x[2] -= m3*x1  + m10*x2  + m17*x3  + m24*x4  + m31*x5  + m38*x6 + m45*x7;
          x[3] -= m4*x1  + m11*x2  + m18*x3  + m25*x4  + m32*x5  + m39*x6 + m46*x7;
          x[4] -= m5*x1  + m12*x2  + m19*x3  + m26*x4  + m33*x5  + m40*x6 + m47*x7;
          x[5] -= m6*x1  + m13*x2  + m20*x3  + m27*x4  + m34*x5  + m41*x6 + m48*x7;
          x[6] -= m7*x1  + m14*x2  + m21*x3  + m28*x4  + m35*x5  + m42*x6 + m49*x7;

          x[7]  -= m1*x8  + m8*x9   + m15*x10 + m22*x11 + m29*x12 + m36*x13 + m43*x14;
          x[8]  -= m2*x8  + m9*x9   + m16*x10 + m23*x11 + m30*x12 + m37*x13 + m44*x14;
          x[9]  -= m3*x8  + m10*x9  + m17*x10 + m24*x11 + m31*x12 + m38*x13 + m45*x14;
          x[10] -= m4*x8  + m11*x9  + m18*x10 + m25*x11 + m32*x12 + m39*x13 + m46*x14;
          x[11] -= m5*x8  + m12*x9  + m19*x10 + m26*x11 + m33*x12 + m40*x13 + m47*x14;
          x[12] -= m6*x8  + m13*x9  + m20*x10 + m27*x11 + m34*x12 + m41*x13 + m48*x14;
          x[13] -= m7*x8  + m14*x9  + m21*x10 + m28*x11 + m35*x12 + m42*x13 + m49*x14;

          x[14] -= m1*x15 + m8*x16  + m15*x17 + m22*x18 + m29*x19 + m36*x20 + m43*x21;
          x[15] -= m2*x15 + m9*x16  + m16*x17 + m23*x18 + m30*x19 + m37*x20 + m44*x21;
          x[16] -= m3*x15 + m10*x16 + m17*x17 + m24*x18 + m31*x19 + m38*x20 + m45*x21;
          x[17] -= m4*x15 + m11*x16 + m18*x17 + m25*x18 + m32*x19 + m39*x20 + m46*x21;
          x[18] -= m5*x15 + m12*x16 + m19*x17 + m26*x18 + m33*x19 + m40*x20 + m47*x21;
          x[19] -= m6*x15 + m13*x16 + m20*x17 + m27*x18 + m34*x19 + m41*x20 + m48*x21;
          x[20] -= m7*x15 + m14*x16 + m21*x17 + m28*x18 + m35*x19 + m42*x20 + m49*x21;

          x[21] -= m1*x22 + m8*x23  + m15*x24 + m22*x25 + m29*x26 + m36*x27 + m43*x28;
          x[22] -= m2*x22 + m9*x23  + m16*x24 + m23*x25 + m30*x26 + m37*x27 + m44*x28;
          x[23] -= m3*x22 + m10*x23 + m17*x24 + m24*x25 + m31*x26 + m38*x27 + m45*x28;
          x[24] -= m4*x22 + m11*x23 + m18*x24 + m25*x25 + m32*x26 + m39*x27 + m46*x28;
          x[25] -= m5*x22 + m12*x23 + m19*x24 + m26*x25 + m33*x26 + m40*x27 + m47*x28;
          x[26] -= m6*x22 + m13*x23 + m20*x24 + m27*x25 + m34*x26 + m41*x27 + m48*x28;
          x[27] -= m7*x22 + m14*x23 + m21*x24 + m28*x25 + m35*x26 + m42*x27 + m49*x28;

          x[28] -= m1*x29 + m8*x30  + m15*x31 + m22*x32 + m29*x33 + m36*x34 + m43*x35;
          x[29] -= m2*x29 + m9*x30  + m16*x31 + m23*x32 + m30*x33 + m37*x34 + m44*x35;
          x[30] -= m3*x29 + m10*x30 + m17*x31 + m24*x32 + m31*x33 + m38*x34 + m45*x35;
          x[31] -= m4*x29 + m11*x30 + m18*x31 + m25*x32 + m32*x33 + m39*x34 + m46*x35;
          x[32] -= m5*x29 + m12*x30 + m19*x31 + m26*x32 + m33*x33 + m40*x34 + m47*x35;
          x[33] -= m6*x29 + m13*x30 + m20*x31 + m27*x32 + m34*x33 + m41*x34 + m48*x35;
          x[34] -= m7*x29 + m14*x30 + m21*x31 + m28*x32 + m35*x33 + m42*x34 + m49*x35;

          x[35] -= m1*x36 + m8*x37  + m15*x38 + m22*x39 + m29*x40 + m36*x41 + m43*x42;
          x[36] -= m2*x36 + m9*x37  + m16*x38 + m23*x39 + m30*x40 + m37*x41 + m44*x42;
          x[37] -= m3*x36 + m10*x37 + m17*x38 + m24*x39 + m31*x40 + m38*x41 + m45*x42;
          x[38] -= m4*x36 + m11*x37 + m18*x38 + m25*x39 + m32*x40 + m39*x41 + m46*x42;
          x[39] -= m5*x36 + m12*x37 + m19*x38 + m26*x39 + m33*x40 + m40*x41 + m47*x42;
          x[40] -= m6*x36 + m13*x37 + m20*x38 + m27*x39 + m34*x40 + m41*x41 + m48*x42;
          x[41] -= m7*x36 + m14*x37 + m21*x38 + m28*x39 + m35*x40 + m42*x41 + m49*x42;

          x[42] -= m1*x43 + m8*x44  + m15*x45 + m22*x46 + m29*x47 + m36*x48 + m43*x49;
          x[43] -= m2*x43 + m9*x44  + m16*x45 + m23*x46 + m30*x47 + m37*x48 + m44*x49;
          x[44] -= m3*x43 + m10*x44 + m17*x45 + m24*x46 + m31*x47 + m38*x48 + m45*x49;
          x[45] -= m4*x43 + m11*x44 + m18*x45 + m25*x46 + m32*x47 + m39*x48 + m46*x49;
          x[46] -= m5*x43 + m12*x44 + m19*x45 + m26*x46 + m33*x47 + m40*x48 + m47*x49;
          x[47] -= m6*x43 + m13*x44 + m20*x45 + m27*x46 + m34*x47 + m41*x48 + m48*x49;
          x[48] -= m7*x43 + m14*x44 + m21*x45 + m28*x46 + m35*x47 + m42*x48 + m49*x49;
          pv    += 49;
        }
        PetscCall(PetscLogFlops(686.0*nz+637.0));
      }
      row = *ajtmp++;
    }
    /* finished row so stick it into b->a */
    pv = ba + 49*bi[i];
    pj = bj + bi[i];
    nz = bi[i+1] - bi[i];
    for (j=0; j<nz; j++) {
      x      = rtmp+49*pj[j];
      pv[0]  = x[0];  pv[1]  = x[1];  pv[2]  = x[2];  pv[3]  = x[3];
      pv[4]  = x[4];  pv[5]  = x[5];  pv[6]  = x[6];  pv[7]  = x[7];
      pv[8]  = x[8];  pv[9]  = x[9];  pv[10] = x[10]; pv[11] = x[11];
      pv[12] = x[12]; pv[13] = x[13]; pv[14] = x[14]; pv[15] = x[15];
      pv[16] = x[16]; pv[17] = x[17]; pv[18] = x[18]; pv[19] = x[19];
      pv[20] = x[20]; pv[21] = x[21]; pv[22] = x[22]; pv[23] = x[23];
      pv[24] = x[24]; pv[25] = x[25]; pv[26] = x[26]; pv[27] = x[27];
      pv[28] = x[28]; pv[29] = x[29]; pv[30] = x[30]; pv[31] = x[31];
      pv[32] = x[32]; pv[33] = x[33]; pv[34] = x[34]; pv[35] = x[35];
      pv[36] = x[36]; pv[37] = x[37]; pv[38] = x[38]; pv[39] = x[39];
      pv[40] = x[40]; pv[41] = x[41]; pv[42] = x[42]; pv[43] = x[43];
      pv[44] = x[44]; pv[45] = x[45]; pv[46] = x[46]; pv[47] = x[47];
      pv[48] = x[48];
      pv    += 49;
    }
    /* invert diagonal block */
    w    = ba + 49*diag_offset[i];
    PetscCall(PetscKernel_A_gets_inverse_A_7(w,shift,allowzeropivot,&zeropivotdetected));
    if (zeropivotdetected) C->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;
  }

  PetscCall(PetscFree(rtmp));

  C->ops->solve          = MatSolve_SeqBAIJ_7_NaturalOrdering_inplace;
  C->ops->solvetranspose = MatSolveTranspose_SeqBAIJ_7_NaturalOrdering_inplace;
  C->assembled           = PETSC_TRUE;

  PetscCall(PetscLogFlops(1.333333333333*7*7*7*b->mbs)); /* from inverting diagonal blocks */
  PetscFunctionReturn(0);
}

PetscErrorCode MatLUFactorNumeric_SeqBAIJ_7_NaturalOrdering(Mat B,Mat A,const MatFactorInfo *info)
{
  Mat            C =B;
  Mat_SeqBAIJ    *a=(Mat_SeqBAIJ*)A->data,*b=(Mat_SeqBAIJ*)C->data;
  PetscInt       i,j,k,nz,nzL,row;
  const PetscInt n=a->mbs,*ai=a->i,*aj=a->j,*bi=b->i,*bj=b->j;
  const PetscInt *ajtmp,*bjtmp,*bdiag=b->diag,*pj,bs2=a->bs2;
  MatScalar      *rtmp,*pc,*mwork,*v,*pv,*aa=a->a;
  PetscInt       flg;
  PetscReal      shift = info->shiftamount;
  PetscBool      allowzeropivot,zeropivotdetected;

  PetscFunctionBegin;
  allowzeropivot = PetscNot(A->erroriffailure);

  /* generate work space needed by the factorization */
  PetscCall(PetscMalloc2(bs2*n,&rtmp,bs2,&mwork));
  PetscCall(PetscArrayzero(rtmp,bs2*n));

  for (i=0; i<n; i++) {
    /* zero rtmp */
    /* L part */
    nz    = bi[i+1] - bi[i];
    bjtmp = bj + bi[i];
    for  (j=0; j<nz; j++) {
      PetscCall(PetscArrayzero(rtmp+bs2*bjtmp[j],bs2));
    }

    /* U part */
    nz    = bdiag[i] - bdiag[i+1];
    bjtmp = bj + bdiag[i+1]+1;
    for  (j=0; j<nz; j++) {
      PetscCall(PetscArrayzero(rtmp+bs2*bjtmp[j],bs2));
    }

    /* load in initial (unfactored row) */
    nz    = ai[i+1] - ai[i];
    ajtmp = aj + ai[i];
    v     = aa + bs2*ai[i];
    for (j=0; j<nz; j++) {
      PetscCall(PetscArraycpy(rtmp+bs2*ajtmp[j],v+bs2*j,bs2));
    }

    /* elimination */
    bjtmp = bj + bi[i];
    nzL   = bi[i+1] - bi[i];
    for (k=0; k < nzL; k++) {
      row = bjtmp[k];
      pc  = rtmp + bs2*row;
      for (flg=0,j=0; j<bs2; j++) {
        if (pc[j]!=0.0) {
          flg = 1;
          break;
        }
      }
      if (flg) {
        pv = b->a + bs2*bdiag[row];
        /* PetscKernel_A_gets_A_times_B(bs,pc,pv,mwork); *pc = *pc * (*pv); */
        PetscCall(PetscKernel_A_gets_A_times_B_7(pc,pv,mwork));

        pj = b->j + bdiag[row+1]+1; /* beginning of U(row,:) */
        pv = b->a + bs2*(bdiag[row+1]+1);
        nz = bdiag[row] - bdiag[row+1] - 1; /* num of entries inU(row,:), excluding diag */
        for (j=0; j<nz; j++) {
          /* PetscKernel_A_gets_A_minus_B_times_C(bs,rtmp+bs2*pj[j],pc,pv+bs2*j); */
          /* rtmp+bs2*pj[j] = rtmp+bs2*pj[j] - (*pc)*(pv+bs2*j) */
          v    = rtmp + bs2*pj[j];
          PetscCall(PetscKernel_A_gets_A_minus_B_times_C_7(v,pc,pv));
          pv  += bs2;
        }
        PetscCall(PetscLogFlops(686.0*nz+637)); /* flops = 2*bs^3*nz + 2*bs^3 - bs2) */
      }
    }

    /* finished row so stick it into b->a */
    /* L part */
    pv = b->a + bs2*bi[i];
    pj = b->j + bi[i];
    nz = bi[i+1] - bi[i];
    for (j=0; j<nz; j++) {
      PetscCall(PetscArraycpy(pv+bs2*j,rtmp+bs2*pj[j],bs2));
    }

    /* Mark diagonal and invert diagonal for simpler triangular solves */
    pv   = b->a + bs2*bdiag[i];
    pj   = b->j + bdiag[i];
    PetscCall(PetscArraycpy(pv,rtmp+bs2*pj[0],bs2));
    PetscCall(PetscKernel_A_gets_inverse_A_7(pv,shift,allowzeropivot,&zeropivotdetected));
    if (zeropivotdetected) C->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;

    /* U part */
    pv = b->a + bs2*(bdiag[i+1]+1);
    pj = b->j + bdiag[i+1]+1;
    nz = bdiag[i] - bdiag[i+1] - 1;
    for (j=0; j<nz; j++) {
      PetscCall(PetscArraycpy(pv+bs2*j,rtmp+bs2*pj[j],bs2));
    }
  }
  PetscCall(PetscFree2(rtmp,mwork));

  C->ops->solve          = MatSolve_SeqBAIJ_7_NaturalOrdering;
  C->ops->solvetranspose = MatSolveTranspose_SeqBAIJ_7_NaturalOrdering;
  C->assembled           = PETSC_TRUE;

  PetscCall(PetscLogFlops(1.333333333333*7*7*7*n)); /* from inverting diagonal blocks */
  PetscFunctionReturn(0);
}
