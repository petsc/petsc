
#include <../src/mat/impls/sbaij/seq/sbaij.h>
#include <../src/mat/blockinvert.h>

/*
      Version for when blocks are 6 by 6 Using natural ordering
*/
#undef __FUNCT__  
#define __FUNCT__ "MatCholeskyFactorNumeric_SeqSBAIJ_6_NaturalOrdering"
PetscErrorCode MatCholeskyFactorNumeric_SeqSBAIJ_6_NaturalOrdering(Mat C,Mat A,const MatFactorInfo *info)
{
  Mat_SeqSBAIJ   *a = (Mat_SeqSBAIJ*)A->data,*b = (Mat_SeqSBAIJ *)C->data;
  PetscErrorCode ierr;
  PetscInt       i,j,mbs=a->mbs,*bi=b->i,*bj=b->j;
  PetscInt       *ai,*aj,k,k1,jmin,jmax,*jl,*il,vj,nexti,ili;
  MatScalar      *ba = b->a,*aa,*ap,*dk,*uik;
  MatScalar      *u,*d,*w,*wp,u0,u1,u2,u3,u4,u5,u6,u7,u8,u9,u10,u11,u12;
  MatScalar      u13,u14,u15,u16,u17,u18,u19,u20,u21,u22,u23,u24,u25,u26,u27;
  MatScalar      u28,u29,u30,u31,u32,u33,u34,u35;
  MatScalar      d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12;
  MatScalar      d13,d14,d15,d16,d17,d18,d19,d20,d21,d22,d23,d24,d25,d26,d27;
  MatScalar      d28,d29,d30,d31,d32,d33,d34,d35;
  MatScalar      m0,m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12;
  MatScalar      m13,m14,m15,m16,m17,m18,m19,m20,m21,m22,m23,m24,m25,m26,m27;
  MatScalar      m28,m29,m30,m31,m32,m33,m34,m35;
  PetscReal      shift = info->shiftamount;
  
  PetscFunctionBegin;
  /* initialization */
  ierr = PetscMalloc(36*mbs*sizeof(MatScalar),&w);CHKERRQ(ierr);
  ierr = PetscMemzero(w,36*mbs*sizeof(MatScalar));CHKERRQ(ierr); 
  ierr = PetscMalloc2(mbs,PetscInt,&il,mbs,PetscInt,&jl);CHKERRQ(ierr);
  for (i=0; i<mbs; i++) {
    jl[i] = mbs; il[0] = 0;
  }
  ierr = PetscMalloc2(36,MatScalar,&dk,36,MatScalar,&uik);CHKERRQ(ierr);
  ai = a->i; aj = a->j; aa = a->a;

  /* for each row k */
  for (k = 0; k<mbs; k++){

    /*initialize k-th row with elements nonzero in row k of A */
    jmin = ai[k]; jmax = ai[k+1];
    if (jmin < jmax) {
      ap = aa + jmin*36;
      for (j = jmin; j < jmax; j++){
        vj = aj[j];         /* block col. index */  
        wp = w + vj*36;
        for (i=0; i<36; i++) *wp++ = *ap++;        
      } 
    } 

    /* modify k-th row by adding in those rows i with U(i,k) != 0 */
    ierr = PetscMemcpy(dk,w+k*36,36*sizeof(MatScalar));CHKERRQ(ierr); 
    i = jl[k]; /* first row to be added to k_th row  */  

    while (i < mbs){
      nexti = jl[i]; /* next row to be added to k_th row */

      /* compute multiplier */
      ili = il[i];  /* index of first nonzero element in U(i,k:bms-1) */

      /* uik = -inv(Di)*U_bar(i,k) */
      d    = ba + i*36;
      u    = ba + ili*36;

      u0 = u[0]; u1 = u[1]; u2 = u[2]; u3 = u[3]; u4 = u[4]; u5 = u[5]; u6 = u[6];
      u7 = u[7]; u8 = u[8]; u9 = u[9]; u10 = u[10]; u11 = u[11]; u12 = u[12]; u13 = u[13];
      u14 = u[14]; u15 = u[15]; u16 = u[16]; u17 = u[17]; u18 = u[18]; u19 = u[19]; u20 = u[20];
      u21 = u[21]; u22 = u[22]; u23 = u[23]; u24 = u[24]; u25 = u[25]; u26 = u[26]; u27 = u[27];
      u28 = u[28]; u29 = u[29]; u30 = u[30]; u31 = u[31]; u32 = u[32]; u33 = u[33]; u34 = u[34];
      u35 = u[35];

      d0  = d[0];  d1  = d[1];  d2  = d[2];  d3  = d[3];
      d4  = d[4];  d5  = d[5];  d6  = d[6];  d7  = d[7];
      d8  = d[8];  d9 = d[9];  d10 = d[10]; d11 = d[11]; 
      d12 = d[12]; d13 = d[13]; d14 = d[14]; d15 = d[15]; 
      d16 = d[16]; d17 = d[17]; d18 = d[18]; d19 = d[19];
      d20 = d[20]; d21 = d[21]; d22 = d[22]; d23 = d[23];
      d24 = d[24]; d25 = d[25]; d26 = d[26]; d27 = d[27];
      d28 = d[28]; d29 = d[29]; d30 = d[30]; d31 = d[31];
      d32 = d[32]; d33 = d[33]; d34 = d[34]; d35 = d[35];

      m0 = uik[0] = -(d0*u0 + d6*u1 + d12*u2 + d18*u3 + d24*u4 + d30*u5);
      m1 = uik[1] = -(d1*u0 + d7*u1 + d13*u2 + d19*u3 + d25*u4 + d31*u5);
      m2 = uik[2] = -(d2*u0 + d8*u1 + d14*u2 + d20*u3 + d26*u4 + d32*u5);
      m3 = uik[3] = -(d3*u0 + d9*u1 + d15*u2 + d21*u3 + d27*u4 + d33*u5);
      m4 = uik[4] = -(d4*u0+ d10*u1 + d16*u2 + d22*u3 + d28*u4 + d34*u5);
      m5 = uik[5] = -(d5*u0+ d11*u1 + d17*u2 + d23*u3 + d29*u4 + d35*u5);

      m6 = uik[6] = -(d0*u6 + d6*u7 + d12*u8 + d18*u9 + d24*u10 + d30*u11);
      m7 = uik[7] = -(d1*u6 + d7*u7 + d13*u8 + d19*u9 + d25*u10 + d31*u11);
      m8 = uik[8] = -(d2*u6 + d8*u7 + d14*u8 + d20*u9 + d26*u10 + d32*u11);
      m9 = uik[9] = -(d3*u6 + d9*u7 + d15*u8 + d21*u9 + d27*u10 + d33*u11);
      m10 = uik[10]= -(d4*u6+ d10*u7 + d16*u8 + d22*u9 + d28*u10 + d34*u11);
      m11 = uik[11]= -(d5*u6+ d11*u7 + d17*u8 + d23*u9 + d29*u10 + d35*u11);

      m12 = uik[12] = -(d0*u12 + d6*u13 + d12*u14 + d18*u15 + d24*u16 + d30*u17);
      m13 = uik[13] = -(d1*u12 + d7*u13 + d13*u14 + d19*u15 + d25*u16 + d31*u17);
      m14 = uik[14] = -(d2*u12 + d8*u13 + d14*u14 + d20*u15 + d26*u16 + d32*u17);
      m15 = uik[15] = -(d3*u12 + d9*u13 + d15*u14 + d21*u15 + d27*u16 + d33*u17);
      m16 = uik[16] = -(d4*u12+ d10*u13 + d16*u14 + d22*u15 + d28*u16 + d34*u17);
      m17 = uik[17] = -(d5*u12+ d11*u13 + d17*u14 + d23*u15 + d29*u16 + d35*u17);

      m18 = uik[18] = -(d0*u18 + d6*u19 + d12*u20 + d18*u21 + d24*u22 + d30*u23);
      m19 = uik[19] = -(d1*u18 + d7*u19 + d13*u20 + d19*u21 + d25*u22 + d31*u23);
      m20 = uik[20] = -(d2*u18 + d8*u19 + d14*u20 + d20*u21 + d26*u22 + d32*u23);
      m21 = uik[21] = -(d3*u18 + d9*u19 + d15*u20 + d21*u21 + d27*u22 + d33*u23);
      m22 = uik[22] = -(d4*u18+ d10*u19 + d16*u20 + d22*u21 + d28*u22 + d34*u23);
      m23 = uik[23] = -(d5*u18+ d11*u19 + d17*u20 + d23*u21 + d29*u22 + d35*u23);

      m24 = uik[24] = -(d0*u24 + d6*u25 + d12*u26 + d18*u27 + d24*u28 + d30*u29);
      m25 = uik[25] = -(d1*u24 + d7*u25 + d13*u26 + d19*u27 + d25*u28 + d31*u29);
      m26 = uik[26] = -(d2*u24 + d8*u25 + d14*u26 + d20*u27 + d26*u28 + d32*u29);
      m27 = uik[27] = -(d3*u24 + d9*u25 + d15*u26 + d21*u27 + d27*u28 + d33*u29);
      m28 = uik[28] = -(d4*u24+ d10*u25 + d16*u26 + d22*u27 + d28*u28 + d34*u29);
      m29 = uik[29] = -(d5*u24+ d11*u25 + d17*u26 + d23*u27 + d29*u28 + d35*u29);

      m30 = uik[30] = -(d0*u30 + d6*u31 + d12*u32 + d18*u33 + d24*u34 + d30*u35);
      m31 = uik[31] = -(d1*u30 + d7*u31 + d13*u32 + d19*u33 + d25*u34 + d31*u35);
      m32 = uik[32] = -(d2*u30 + d8*u31 + d14*u32 + d20*u33 + d26*u34 + d32*u35);
      m33 = uik[33] = -(d3*u30 + d9*u31 + d15*u32 + d21*u33 + d27*u34 + d33*u35);
      m34 = uik[34] = -(d4*u30+ d10*u31 + d16*u32 + d22*u33 + d28*u34 + d34*u35);
      m35 = uik[35] = -(d5*u30+ d11*u31 + d17*u32 + d23*u33 + d29*u34 + d35*u35);

      /* update D(k) += -U(i,k)^T * U_bar(i,k) */  
      dk[0] +=  m0*u0 + m1*u1 + m2*u2 + m3*u3 + m4*u4 + m5*u5;
      dk[1] +=  m6*u0 + m7*u1 + m8*u2 + m9*u3+ m10*u4+ m11*u5;
      dk[2] += m12*u0+ m13*u1+ m14*u2+ m15*u3+ m16*u4+ m17*u5;
      dk[3] += m18*u0+ m19*u1+ m20*u2+ m21*u3+ m22*u4+ m23*u5;
      dk[4] += m24*u0+ m25*u1+ m26*u2+ m27*u3+ m28*u4+ m29*u5;
      dk[5] += m30*u0+ m31*u1+ m32*u2+ m33*u3+ m34*u4+ m35*u5;

      dk[6] +=  m0*u6 + m1*u7 + m2*u8 + m3*u9 + m4*u10 + m5*u11;
      dk[7] +=  m6*u6 + m7*u7 + m8*u8 + m9*u9+ m10*u10+ m11*u11;
      dk[8] += m12*u6+ m13*u7+ m14*u8+ m15*u9+ m16*u10+ m17*u11;
      dk[9] += m18*u6+ m19*u7+ m20*u8+ m21*u9+ m22*u10+ m23*u11;
      dk[10]+= m24*u6+ m25*u7+ m26*u8+ m27*u9+ m28*u10+ m29*u11;
      dk[11]+= m30*u6+ m31*u7+ m32*u8+ m33*u9+ m34*u10+ m35*u11;

      dk[12]+=  m0*u12 + m1*u13 + m2*u14 + m3*u15 + m4*u16 + m5*u17;
      dk[13]+=  m6*u12 + m7*u13 + m8*u14 + m9*u15+ m10*u16+ m11*u17;
      dk[14]+= m12*u12+ m13*u13+ m14*u14+ m15*u15+ m16*u16+ m17*u17;
      dk[15]+= m18*u12+ m19*u13+ m20*u14+ m21*u15+ m22*u16+ m23*u17;
      dk[16]+= m24*u12+ m25*u13+ m26*u14+ m27*u15+ m28*u16+ m29*u17;
      dk[17]+= m30*u12+ m31*u13+ m32*u14+ m33*u15+ m34*u16+ m35*u17;

      dk[18]+=  m0*u18 + m1*u19 + m2*u20 + m3*u21 + m4*u22 + m5*u23;
      dk[19]+=  m6*u18 + m7*u19 + m8*u20 + m9*u21+ m10*u22+ m11*u23;
      dk[20]+= m12*u18+ m13*u19+ m14*u20+ m15*u21+ m16*u22+ m17*u23;
      dk[21]+= m18*u18+ m19*u19+ m20*u20+ m21*u21+ m22*u22+ m23*u23;
      dk[22]+= m24*u18+ m25*u19+ m26*u20+ m27*u21+ m28*u22+ m29*u23;
      dk[23]+= m30*u18+ m31*u19+ m32*u20+ m33*u21+ m34*u22+ m35*u23;

      dk[24]+=  m0*u24 + m1*u25 + m2*u26 + m3*u27 + m4*u28 + m5*u29;
      dk[25]+=  m6*u24 + m7*u25 + m8*u26 + m9*u27+ m10*u28+ m11*u29;
      dk[26]+= m12*u24+ m13*u25+ m14*u26+ m15*u27+ m16*u28+ m17*u29;
      dk[27]+= m18*u24+ m19*u25+ m20*u26+ m21*u27+ m22*u28+ m23*u29;
      dk[28]+= m24*u24+ m25*u25+ m26*u26+ m27*u27+ m28*u28+ m29*u29;
      dk[29]+= m30*u24+ m31*u25+ m32*u26+ m33*u27+ m34*u28+ m35*u29;

      dk[30]+=  m0*u30 + m1*u31 + m2*u32 + m3*u33 + m4*u34 + m5*u35;
      dk[31]+=  m6*u30 + m7*u31 + m8*u32 + m9*u33+ m10*u34+ m11*u35;
      dk[32]+= m12*u30+ m13*u31+ m14*u32+ m15*u33+ m16*u34+ m17*u35;
      dk[33]+= m18*u30+ m19*u31+ m20*u32+ m21*u33+ m22*u34+ m23*u35;
      dk[34]+= m24*u30+ m25*u31+ m26*u32+ m27*u33+ m28*u34+ m29*u35;
      dk[35]+= m30*u30+ m31*u31+ m32*u32+ m33*u33+ m34*u34+ m35*u35;

      ierr = PetscLogFlops(216.0*4.0);CHKERRQ(ierr);
 
      /* update -U(i,k) */
      ierr = PetscMemcpy(ba+ili*36,uik,36*sizeof(MatScalar));CHKERRQ(ierr); 

      /* add multiple of row i to k-th row ... */
      jmin = ili + 1; jmax = bi[i+1];
      if (jmin < jmax){
        for (j=jmin; j<jmax; j++) {
          /* w += -U(i,k)^T * U_bar(i,j) */
          wp = w + bj[j]*36;
          u = ba + j*36;

	  u0 = u[0]; u1 = u[1]; u2 = u[2]; u3 = u[3]; u4 = u[4]; u5 = u[5]; u6 = u[6];
	  u7 = u[7]; u8 = u[8]; u9 = u[9]; u10 = u[10]; u11 = u[11]; u12 = u[12]; u13 = u[13];
	  u14 = u[14]; u15 = u[15]; u16 = u[16]; u17 = u[17]; u18 = u[18]; u19 = u[19]; u20 = u[20];
	  u21 = u[21]; u22 = u[22]; u23 = u[23]; u24 = u[24]; u25 = u[25]; u26 = u[26]; u27 = u[27];
	  u28 = u[28]; u29 = u[29]; u30 = u[30]; u31 = u[31]; u32 = u[32]; u33 = u[33]; u34 = u[34];
	  u35 = u[35];

          wp[0] +=  m0*u0 + m1*u1 + m2*u2 + m3*u3 + m4*u4 + m5*u5;
          wp[1] +=  m6*u0 + m7*u1 + m8*u2 + m9*u3+ m10*u4+ m11*u5;
          wp[2] += m12*u0+ m13*u1+ m14*u2+ m15*u3+ m16*u4+ m17*u5;
          wp[3] += m18*u0+ m19*u1+ m20*u2+ m21*u3+ m22*u4+ m23*u5;
          wp[4] += m24*u0+ m25*u1+ m26*u2+ m27*u3+ m28*u4+ m29*u5;
          wp[5] += m30*u0+ m31*u1+ m32*u2+ m33*u3+ m34*u4+ m35*u5;

          wp[6] +=  m0*u6 + m1*u7 + m2*u8 + m3*u9 + m4*u10 + m5*u11;
          wp[7] +=  m6*u6 + m7*u7 + m8*u8 + m9*u9+ m10*u10+ m11*u11;
          wp[8] += m12*u6+ m13*u7+ m14*u8+ m15*u9+ m16*u10+ m17*u11;
          wp[9] += m18*u6+ m19*u7+ m20*u8+ m21*u9+ m22*u10+ m23*u11;
          wp[10]+= m24*u6+ m25*u7+ m26*u8+ m27*u9+ m28*u10+ m29*u11;
          wp[11]+= m30*u6+ m31*u7+ m32*u8+ m33*u9+ m34*u10+ m35*u11;

          wp[12]+=  m0*u12 + m1*u13 + m2*u14 + m3*u15 + m4*u16 + m5*u17;
          wp[13]+=  m6*u12 + m7*u13 + m8*u14 + m9*u15+ m10*u16+ m11*u17;
          wp[14]+= m12*u12+ m13*u13+ m14*u14+ m15*u15+ m16*u16+ m17*u17;
          wp[15]+= m18*u12+ m19*u13+ m20*u14+ m21*u15+ m22*u16+ m23*u17;
          wp[16]+= m24*u12+ m25*u13+ m26*u14+ m27*u15+ m28*u16+ m29*u17;
          wp[17]+= m30*u12+ m31*u13+ m32*u14+ m33*u15+ m34*u16+ m35*u17;

          wp[18]+=  m0*u18 + m1*u19 + m2*u20 + m3*u21 + m4*u22 + m5*u23;
          wp[19]+=  m6*u18 + m7*u19 + m8*u20 + m9*u21+ m10*u22+ m11*u23;
          wp[20]+= m12*u18+ m13*u19+ m14*u20+ m15*u21+ m16*u22+ m17*u23;
          wp[21]+= m18*u18+ m19*u19+ m20*u20+ m21*u21+ m22*u22+ m23*u23;
          wp[22]+= m24*u18+ m25*u19+ m26*u20+ m27*u21+ m28*u22+ m29*u23;
          wp[23]+= m30*u18+ m31*u19+ m32*u20+ m33*u21+ m34*u22+ m35*u23;

          wp[24]+=  m0*u24 + m1*u25 + m2*u26 + m3*u27 + m4*u28 + m5*u29;
          wp[25]+=  m6*u24 + m7*u25 + m8*u26 + m9*u27+ m10*u28+ m11*u29;
          wp[26]+= m12*u24+ m13*u25+ m14*u26+ m15*u27+ m16*u28+ m17*u29;
          wp[27]+= m18*u24+ m19*u25+ m20*u26+ m21*u27+ m22*u28+ m23*u29;
          wp[28]+= m24*u24+ m25*u25+ m26*u26+ m27*u27+ m28*u28+ m29*u29;
          wp[29]+= m30*u24+ m31*u25+ m32*u26+ m33*u27+ m34*u28+ m35*u29;

          wp[30]+=  m0*u30 + m1*u31 + m2*u32 + m3*u33 + m4*u34 + m5*u35;
          wp[31]+=  m6*u30 + m7*u31 + m8*u32 + m9*u33+ m10*u34+ m11*u35;
          wp[32]+= m12*u30+ m13*u31+ m14*u32+ m15*u33+ m16*u34+ m17*u35;
          wp[33]+= m18*u30+ m19*u31+ m20*u32+ m21*u33+ m22*u34+ m23*u35;
          wp[34]+= m24*u30+ m25*u31+ m26*u32+ m27*u33+ m28*u34+ m29*u35;
          wp[35]+= m30*u30+ m31*u31+ m32*u32+ m33*u33+ m34*u34+ m35*u35;
        }
        ierr = PetscLogFlops(2.0*216.0*(jmax-jmin));CHKERRQ(ierr);
      
        /* ... add i to row list for next nonzero entry */
        il[i] = jmin;             /* update il(i) in column k+1, ... mbs-1 */
        j     = bj[jmin];
        jl[i] = jl[j]; jl[j] = i; /* update jl */
      }      
      i = nexti;      
    }

    /* save nonzero entries in k-th row of U ... */

    /* invert diagonal block */
    d = ba+k*36;
    ierr = PetscMemcpy(d,dk,36*sizeof(MatScalar));CHKERRQ(ierr);
    ierr = PetscKernel_A_gets_inverse_A_6(d,shift);CHKERRQ(ierr);
    
    jmin = bi[k]; jmax = bi[k+1];
    if (jmin < jmax) {
      for (j=jmin; j<jmax; j++){
         vj = bj[j];           /* block col. index of U */
         u  = ba + j*36;
         wp = w + vj*36;        
         for (k1=0; k1<36; k1++){
           *u++  = *wp; 
           *wp++ = 0.0;
         }
      } 
      
      /* ... add k to row list for first nonzero entry in k-th row */
      il[k] = jmin;
      i     = bj[jmin];
      jl[k] = jl[i]; jl[i] = k;
    }    
  } 

  ierr = PetscFree(w);CHKERRQ(ierr);
  ierr = PetscFree2(il,jl);CHKERRQ(ierr); 
  ierr = PetscFree2(dk,uik);CHKERRQ(ierr);

  C->ops->solve          = MatSolve_SeqSBAIJ_6_NaturalOrdering_inplace;
  C->ops->solvetranspose = MatSolve_SeqSBAIJ_6_NaturalOrdering_inplace;
  C->ops->forwardsolve   = MatForwardSolve_SeqSBAIJ_6_NaturalOrdering_inplace;
  C->ops->backwardsolve  = MatBackwardSolve_SeqSBAIJ_6_NaturalOrdering_inplace;
  C->assembled = PETSC_TRUE;
  C->preallocated = PETSC_TRUE;  
  ierr = PetscLogFlops(1.3333*216*b->mbs);CHKERRQ(ierr); /* from inverting diagonal blocks */
  PetscFunctionReturn(0);
}
