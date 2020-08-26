
#include <../src/mat/impls/sbaij/seq/sbaij.h>
#include <petsc/private/kernels/blockinvert.h>

/*
      Version for when blocks are 7 by 7 Using natural ordering
*/
PetscErrorCode MatCholeskyFactorNumeric_SeqSBAIJ_7_NaturalOrdering(Mat C,Mat A,const MatFactorInfo *info)
{
  Mat_SeqSBAIJ   *a = (Mat_SeqSBAIJ*)A->data,*b = (Mat_SeqSBAIJ*)C->data;
  PetscErrorCode ierr;
  PetscInt       i,j,mbs=a->mbs,*bi=b->i,*bj=b->j;
  PetscInt       *ai,*aj,k,k1,jmin,jmax,*jl,*il,vj,nexti,ili;
  MatScalar      *ba = b->a,*aa,*ap,*dk,*uik;
  MatScalar      *u,*d,*w,*wp,u0,u1,u2,u3,u4,u5,u6,u7,u8,u9,u10,u11,u12;
  MatScalar      u13,u14,u15,u16,u17,u18,u19,u20,u21,u22,u23,u24,u25,u26,u27;
  MatScalar      u28,u29,u30,u31,u32,u33,u34,u35,u36,u37,u38,u39,u40,u41;
  MatScalar      u42,u43,u44,u45,u46,u47,u48;
  PetscReal      shift = info->shiftamount;
  PetscBool      allowzeropivot,zeropivotdetected;

  PetscFunctionBegin;
  /* initialization */
  allowzeropivot = PetscNot(A->erroriffailure);
  ierr = PetscCalloc1(49*mbs,&w);CHKERRQ(ierr);
  ierr = PetscMalloc2(mbs,&il,mbs,&jl);CHKERRQ(ierr);
  il[0] = 0;
  for (i=0; i<mbs; i++) jl[i] = mbs;

  ierr = PetscMalloc2(49,&dk,49,&uik);CHKERRQ(ierr);
  ai   = a->i; aj = a->j; aa = a->a;

  /* for each row k */
  for (k = 0; k<mbs; k++) {

    /*initialize k-th row with elements nonzero in row k of A */
    jmin = ai[k]; jmax = ai[k+1];
    if (jmin < jmax) {
      ap = aa + jmin*49;
      for (j = jmin; j < jmax; j++) {
        vj   = aj[j];         /* block col. index */
        wp   = w + vj*49;
        ierr = PetscArraycpy(wp,ap,49);CHKERRQ(ierr);
        ap   += 49;
      }
    }

    /* modify k-th row by adding in those rows i with U(i,k) != 0 */
    ierr = PetscArraycpy(dk,w+k*49,49);CHKERRQ(ierr);
    i    = jl[k]; /* first row to be added to k_th row  */

    while (i < mbs) {
      nexti = jl[i]; /* next row to be added to k_th row */

      /* compute multiplier */
      ili = il[i];  /* index of first nonzero element in U(i,k:bms-1) */

      /* uik = -inv(Di)*U_bar(i,k) */
      d = ba + i*49;
      u = ba + ili*49;

      u0  = u[0]; u1 = u[1]; u2 = u[2]; u3 = u[3]; u4 = u[4]; u5 = u[5]; u6 = u[6];
      u7  = u[7]; u8 = u[8]; u9 = u[9]; u10 = u[10]; u11 = u[11]; u12 = u[12]; u13 = u[13];
      u14 = u[14]; u15 = u[15]; u16 = u[16]; u17 = u[17]; u18 = u[18]; u19 = u[19]; u20 = u[20];
      u21 = u[21]; u22 = u[22]; u23 = u[23]; u24 = u[24]; u25 = u[25]; u26 = u[26]; u27 = u[27];
      u28 = u[28]; u29 = u[29]; u30 = u[30]; u31 = u[31]; u32 = u[32]; u33 = u[33]; u34 = u[34];
      u35 = u[35]; u36 = u[36]; u37 = u[37]; u38 = u[38]; u39 = u[39]; u40 = u[40]; u41 = u[41]; u42 = u[42];
      u43 = u[43]; u44 = u[44]; u45 = u[45]; u46 = u[46]; u47 = u[47]; u48 = u[48];

      uik[0] = -(d[0]*u0 + d[7]*u1+ d[14]*u2+ d[21]*u3+ d[28]*u4+ d[35]*u5+ d[42]*u6);
      uik[1] = -(d[1]*u0 + d[8]*u1+ d[15]*u2+ d[22]*u3+ d[29]*u4+ d[36]*u5+ d[43]*u6);
      uik[2] = -(d[2]*u0 + d[9]*u1+ d[16]*u2+ d[23]*u3+ d[30]*u4+ d[37]*u5+ d[44]*u6);
      uik[3] = -(d[3]*u0+ d[10]*u1+ d[17]*u2+ d[24]*u3+ d[31]*u4+ d[38]*u5+ d[45]*u6);
      uik[4] = -(d[4]*u0+ d[11]*u1+ d[18]*u2+ d[25]*u3+ d[32]*u4+ d[39]*u5+ d[46]*u6);
      uik[5] = -(d[5]*u0+ d[12]*u1+ d[19]*u2+ d[26]*u3+ d[33]*u4+ d[40]*u5+ d[47]*u6);
      uik[6] = -(d[6]*u0+ d[13]*u1+ d[20]*u2+ d[27]*u3+ d[34]*u4+ d[41]*u5+ d[48]*u6);

      uik[7] = -(d[0]*u7 + d[7]*u8+ d[14]*u9+ d[21]*u10+ d[28]*u11+ d[35]*u12+ d[42]*u13);
      uik[8] = -(d[1]*u7 + d[8]*u8+ d[15]*u9+ d[22]*u10+ d[29]*u11+ d[36]*u12+ d[43]*u13);
      uik[9] = -(d[2]*u7 + d[9]*u8+ d[16]*u9+ d[23]*u10+ d[30]*u11+ d[37]*u12+ d[44]*u13);
      uik[10]= -(d[3]*u7+ d[10]*u8+ d[17]*u9+ d[24]*u10+ d[31]*u11+ d[38]*u12+ d[45]*u13);
      uik[11]= -(d[4]*u7+ d[11]*u8+ d[18]*u9+ d[25]*u10+ d[32]*u11+ d[39]*u12+ d[46]*u13);
      uik[12]= -(d[5]*u7+ d[12]*u8+ d[19]*u9+ d[26]*u10+ d[33]*u11+ d[40]*u12+ d[47]*u13);
      uik[13]= -(d[6]*u7+ d[13]*u8+ d[20]*u9+ d[27]*u10+ d[34]*u11+ d[41]*u12+ d[48]*u13);

      uik[14]= -(d[0]*u14 + d[7]*u15+ d[14]*u16+ d[21]*u17+ d[28]*u18+ d[35]*u19+ d[42]*u20);
      uik[15]= -(d[1]*u14 + d[8]*u15+ d[15]*u16+ d[22]*u17+ d[29]*u18+ d[36]*u19+ d[43]*u20);
      uik[16]= -(d[2]*u14 + d[9]*u15+ d[16]*u16+ d[23]*u17+ d[30]*u18+ d[37]*u19+ d[44]*u20);
      uik[17]= -(d[3]*u14+ d[10]*u15+ d[17]*u16+ d[24]*u17+ d[31]*u18+ d[38]*u19+ d[45]*u20);
      uik[18]= -(d[4]*u14+ d[11]*u15+ d[18]*u16+ d[25]*u17+ d[32]*u18+ d[39]*u19+ d[46]*u20);
      uik[19]= -(d[5]*u14+ d[12]*u15+ d[19]*u16+ d[26]*u17+ d[33]*u18+ d[40]*u19+ d[47]*u20);
      uik[20]= -(d[6]*u14+ d[13]*u15+ d[20]*u16+ d[27]*u17+ d[34]*u18+ d[41]*u19+ d[48]*u20);

      uik[21]= -(d[0]*u21 + d[7]*u22+ d[14]*u23+ d[21]*u24+ d[28]*u25+ d[35]*u26+ d[42]*u27);
      uik[22]= -(d[1]*u21 + d[8]*u22+ d[15]*u23+ d[22]*u24+ d[29]*u25+ d[36]*u26+ d[43]*u27);
      uik[23]= -(d[2]*u21 + d[9]*u22+ d[16]*u23+ d[23]*u24+ d[30]*u25+ d[37]*u26+ d[44]*u27);
      uik[24]= -(d[3]*u21+ d[10]*u22+ d[17]*u23+ d[24]*u24+ d[31]*u25+ d[38]*u26+ d[45]*u27);
      uik[25]= -(d[4]*u21+ d[11]*u22+ d[18]*u23+ d[25]*u24+ d[32]*u25+ d[39]*u26+ d[46]*u27);
      uik[26]= -(d[5]*u21+ d[12]*u22+ d[19]*u23+ d[26]*u24+ d[33]*u25+ d[40]*u26+ d[47]*u27);
      uik[27]= -(d[6]*u21+ d[13]*u22+ d[20]*u23+ d[27]*u24+ d[34]*u25+ d[41]*u26+ d[48]*u27);

      uik[28]= -(d[0]*u28 + d[7]*u29+ d[14]*u30+ d[21]*u31+ d[28]*u32+ d[35]*u33+ d[42]*u34);
      uik[29]= -(d[1]*u28 + d[8]*u29+ d[15]*u30+ d[22]*u31+ d[29]*u32+ d[36]*u33+ d[43]*u34);
      uik[30]= -(d[2]*u28 + d[9]*u29+ d[16]*u30+ d[23]*u31+ d[30]*u32+ d[37]*u33+ d[44]*u34);
      uik[31]= -(d[3]*u28+ d[10]*u29+ d[17]*u30+ d[24]*u31+ d[31]*u32+ d[38]*u33+ d[45]*u34);
      uik[32]= -(d[4]*u28+ d[11]*u29+ d[18]*u30+ d[25]*u31+ d[32]*u32+ d[39]*u33+ d[46]*u34);
      uik[33]= -(d[5]*u28+ d[12]*u29+ d[19]*u30+ d[26]*u31+ d[33]*u32+ d[40]*u33+ d[47]*u34);
      uik[34]= -(d[6]*u28+ d[13]*u29+ d[20]*u30+ d[27]*u31+ d[34]*u32+ d[41]*u33+ d[48]*u34);

      uik[35]= -(d[0]*u35 + d[7]*u36+ d[14]*u37+ d[21]*u38+ d[28]*u39+ d[35]*u40+ d[42]*u41);
      uik[36]= -(d[1]*u35 + d[8]*u36+ d[15]*u37+ d[22]*u38+ d[29]*u39+ d[36]*u40+ d[43]*u41);
      uik[37]= -(d[2]*u35 + d[9]*u36+ d[16]*u37+ d[23]*u38+ d[30]*u39+ d[37]*u40+ d[44]*u41);
      uik[38]= -(d[3]*u35+ d[10]*u36+ d[17]*u37+ d[24]*u38+ d[31]*u39+ d[38]*u40+ d[45]*u41);
      uik[39]= -(d[4]*u35+ d[11]*u36+ d[18]*u37+ d[25]*u38+ d[32]*u39+ d[39]*u40+ d[46]*u41);
      uik[40]= -(d[5]*u35+ d[12]*u36+ d[19]*u37+ d[26]*u38+ d[33]*u39+ d[40]*u40+ d[47]*u41);
      uik[41]= -(d[6]*u35+ d[13]*u36+ d[20]*u37+ d[27]*u38+ d[34]*u39+ d[41]*u40+ d[48]*u41);

      uik[42]= -(d[0]*u42 + d[7]*u43+ d[14]*u44+ d[21]*u45+ d[28]*u46+ d[35]*u47+ d[42]*u48);
      uik[43]= -(d[1]*u42 + d[8]*u43+ d[15]*u44+ d[22]*u45+ d[29]*u46+ d[36]*u47+ d[43]*u48);
      uik[44]= -(d[2]*u42 + d[9]*u43+ d[16]*u44+ d[23]*u45+ d[30]*u46+ d[37]*u47+ d[44]*u48);
      uik[45]= -(d[3]*u42+ d[10]*u43+ d[17]*u44+ d[24]*u45+ d[31]*u46+ d[38]*u47+ d[45]*u48);
      uik[46]= -(d[4]*u42+ d[11]*u43+ d[18]*u44+ d[25]*u45+ d[32]*u46+ d[39]*u47+ d[46]*u48);
      uik[47]= -(d[5]*u42+ d[12]*u43+ d[19]*u44+ d[26]*u45+ d[33]*u46+ d[40]*u47+ d[47]*u48);
      uik[48]= -(d[6]*u42+ d[13]*u43+ d[20]*u44+ d[27]*u45+ d[34]*u46+ d[41]*u47+ d[48]*u48);

      /* update D(k) += -U(i,k)^T * U_bar(i,k) */
      dk[0]+=  uik[0]*u0 + uik[1]*u1 + uik[2]*u2 + uik[3]*u3 + uik[4]*u4 + uik[5]*u5 + uik[6]*u6;
      dk[1]+=  uik[7]*u0 + uik[8]*u1 + uik[9]*u2+ uik[10]*u3+ uik[11]*u4+ uik[12]*u5+ uik[13]*u6;
      dk[2]+= uik[14]*u0+ uik[15]*u1+ uik[16]*u2+ uik[17]*u3+ uik[18]*u4+ uik[19]*u5+ uik[20]*u6;
      dk[3]+= uik[21]*u0+ uik[22]*u1+ uik[23]*u2+ uik[24]*u3+ uik[25]*u4+ uik[26]*u5+ uik[27]*u6;
      dk[4]+= uik[28]*u0+ uik[29]*u1+ uik[30]*u2+ uik[31]*u3+ uik[32]*u4+ uik[33]*u5+ uik[34]*u6;
      dk[5]+= uik[35]*u0+ uik[36]*u1+ uik[37]*u2+ uik[38]*u3+ uik[39]*u4+ uik[40]*u5+ uik[41]*u6;
      dk[6]+= uik[42]*u0+ uik[43]*u1+ uik[44]*u2+ uik[45]*u3+ uik[46]*u4+ uik[47]*u5+ uik[48]*u6;

      dk[7] += uik[0]*u7 + uik[1]*u8 + uik[2]*u9 + uik[3]*u10 + uik[4]*u11 + uik[5]*u12 + uik[6]*u13;
      dk[8] += uik[7]*u7 + uik[8]*u8 + uik[9]*u9+ uik[10]*u10+ uik[11]*u11+ uik[12]*u12+ uik[13]*u13;
      dk[9] +=uik[14]*u7+ uik[15]*u8+ uik[16]*u9+ uik[17]*u10+ uik[18]*u11+ uik[19]*u12+ uik[20]*u13;
      dk[10]+=uik[21]*u7+ uik[22]*u8+ uik[23]*u9+ uik[24]*u10+ uik[25]*u11+ uik[26]*u12+ uik[27]*u13;
      dk[11]+=uik[28]*u7+ uik[29]*u8+ uik[30]*u9+ uik[31]*u10+ uik[32]*u11+ uik[33]*u12+ uik[34]*u13;
      dk[12]+=uik[35]*u7+ uik[36]*u8+ uik[37]*u9+ uik[38]*u10+ uik[39]*u11+ uik[40]*u12+ uik[41]*u13;
      dk[13]+=uik[42]*u7+ uik[43]*u8+ uik[44]*u9+ uik[45]*u10+ uik[46]*u11+ uik[47]*u12+ uik[48]*u13;

      dk[14]+=  uik[0]*u14 + uik[1]*u15 + uik[2]*u16 + uik[3]*u17 + uik[4]*u18 + uik[5]*u19 + uik[6]*u20;
      dk[15]+=  uik[7]*u14 + uik[8]*u15 + uik[9]*u16+ uik[10]*u17+ uik[11]*u18+ uik[12]*u19+ uik[13]*u20;
      dk[16]+= uik[14]*u14+ uik[15]*u15+ uik[16]*u16+ uik[17]*u17+ uik[18]*u18+ uik[19]*u19+ uik[20]*u20;
      dk[17]+= uik[21]*u14+ uik[22]*u15+ uik[23]*u16+ uik[24]*u17+ uik[25]*u18+ uik[26]*u19+ uik[27]*u20;
      dk[18]+= uik[28]*u14+ uik[29]*u15+ uik[30]*u16+ uik[31]*u17+ uik[32]*u18+ uik[33]*u19+ uik[34]*u20;
      dk[19]+= uik[35]*u14+ uik[36]*u15+ uik[37]*u16+ uik[38]*u17+ uik[39]*u18+ uik[40]*u19+ uik[41]*u20;
      dk[20]+= uik[42]*u14+ uik[43]*u15+ uik[44]*u16+ uik[45]*u17+ uik[46]*u18+ uik[47]*u19+ uik[48]*u20;

      dk[21]+=  uik[0]*u21 + uik[1]*u22 + uik[2]*u23 + uik[3]*u24 + uik[4]*u25 + uik[5]*u26 + uik[6]*u27;
      dk[22]+=  uik[7]*u21 + uik[8]*u22 + uik[9]*u23+ uik[10]*u24+ uik[11]*u25+ uik[12]*u26+ uik[13]*u27;
      dk[23]+= uik[14]*u21+ uik[15]*u22+ uik[16]*u23+ uik[17]*u24+ uik[18]*u25+ uik[19]*u26+ uik[20]*u27;
      dk[24]+= uik[21]*u21+ uik[22]*u22+ uik[23]*u23+ uik[24]*u24+ uik[25]*u25+ uik[26]*u26+ uik[27]*u27;
      dk[25]+= uik[28]*u21+ uik[29]*u22+ uik[30]*u23+ uik[31]*u24+ uik[32]*u25+ uik[33]*u26+ uik[34]*u27;
      dk[26]+= uik[35]*u21+ uik[36]*u22+ uik[37]*u23+ uik[38]*u24+ uik[39]*u25+ uik[40]*u26+ uik[41]*u27;
      dk[27]+= uik[42]*u21+ uik[43]*u22+ uik[44]*u23+ uik[45]*u24+ uik[46]*u25+ uik[47]*u26+ uik[48]*u27;

      dk[28]+=  uik[0]*u28 + uik[1]*u29 + uik[2]*u30 + uik[3]*u31 + uik[4]*u32 + uik[5]*u33 + uik[6]*u34;
      dk[29]+=  uik[7]*u28 + uik[8]*u29 + uik[9]*u30+ uik[10]*u31+ uik[11]*u32+ uik[12]*u33+ uik[13]*u34;
      dk[30]+= uik[14]*u28+ uik[15]*u29+ uik[16]*u30+ uik[17]*u31+ uik[18]*u32+ uik[19]*u33+ uik[20]*u34;
      dk[31]+= uik[21]*u28+ uik[22]*u29+ uik[23]*u30+ uik[24]*u31+ uik[25]*u32+ uik[26]*u33+ uik[27]*u34;
      dk[32]+= uik[28]*u28+ uik[29]*u29+ uik[30]*u30+ uik[31]*u31+ uik[32]*u32+ uik[33]*u33+ uik[34]*u34;
      dk[33]+= uik[35]*u28+ uik[36]*u29+ uik[37]*u30+ uik[38]*u31+ uik[39]*u32+ uik[40]*u33+ uik[41]*u34;
      dk[34]+= uik[42]*u28+ uik[43]*u29+ uik[44]*u30+ uik[45]*u31+ uik[46]*u32+ uik[47]*u33+ uik[48]*u34;

      dk[35]+=  uik[0]*u35 + uik[1]*u36 + uik[2]*u37 + uik[3]*u38 + uik[4]*u39 + uik[5]*u40 + uik[6]*u41;
      dk[36]+=  uik[7]*u35 + uik[8]*u36 + uik[9]*u37+ uik[10]*u38+ uik[11]*u39+ uik[12]*u40+ uik[13]*u41;
      dk[37]+= uik[14]*u35+ uik[15]*u36+ uik[16]*u37+ uik[17]*u38+ uik[18]*u39+ uik[19]*u40+ uik[20]*u41;
      dk[38]+= uik[21]*u35+ uik[22]*u36+ uik[23]*u37+ uik[24]*u38+ uik[25]*u39+ uik[26]*u40+ uik[27]*u41;
      dk[39]+= uik[28]*u35+ uik[29]*u36+ uik[30]*u37+ uik[31]*u38+ uik[32]*u39+ uik[33]*u40+ uik[34]*u41;
      dk[40]+= uik[35]*u35+ uik[36]*u36+ uik[37]*u37+ uik[38]*u38+ uik[39]*u39+ uik[40]*u40+ uik[41]*u41;
      dk[41]+= uik[42]*u35+ uik[43]*u36+ uik[44]*u37+ uik[45]*u38+ uik[46]*u39+ uik[47]*u40+ uik[48]*u41;

      dk[42]+=  uik[0]*u42 + uik[1]*u43 + uik[2]*u44 + uik[3]*u45 + uik[4]*u46 + uik[5]*u47 + uik[6]*u48;
      dk[43]+=  uik[7]*u42 + uik[8]*u43 + uik[9]*u44+ uik[10]*u45+ uik[11]*u46+ uik[12]*u47+ uik[13]*u48;
      dk[44]+= uik[14]*u42+ uik[15]*u43+ uik[16]*u44+ uik[17]*u45+ uik[18]*u46+ uik[19]*u47+ uik[20]*u48;
      dk[45]+= uik[21]*u42+ uik[22]*u43+ uik[23]*u44+ uik[24]*u45+ uik[25]*u46+ uik[26]*u47+ uik[27]*u48;
      dk[46]+= uik[28]*u42+ uik[29]*u43+ uik[30]*u44+ uik[31]*u45+ uik[32]*u46+ uik[33]*u47+ uik[34]*u48;
      dk[47]+= uik[35]*u42+ uik[36]*u43+ uik[37]*u44+ uik[38]*u45+ uik[39]*u46+ uik[40]*u47+ uik[41]*u48;
      dk[48]+= uik[42]*u42+ uik[43]*u43+ uik[44]*u44+ uik[45]*u45+ uik[46]*u46+ uik[47]*u47+ uik[48]*u48;

      ierr = PetscLogFlops(343.0*4.0);CHKERRQ(ierr);

      /* update -U(i,k) */
      ierr = PetscArraycpy(ba+ili*49,uik,49);CHKERRQ(ierr);

      /* add multiple of row i to k-th row ... */
      jmin = ili + 1; jmax = bi[i+1];
      if (jmin < jmax) {
        for (j=jmin; j<jmax; j++) {
          /* w += -U(i,k)^T * U_bar(i,j) */
          wp = w + bj[j]*49;
          u  = ba + j*49;

          u0  = u[0];  u1  = u[1];  u2  = u[2];  u3  = u[3];  u4  = u[4];  u5  = u[5];  u6  = u[6];
          u7  = u[7];  u8  = u[8];  u9  = u[9];  u10 = u[10]; u11 = u[11]; u12 = u[12]; u13 = u[13];
          u14 = u[14]; u15 = u[15]; u16 = u[16]; u17 = u[17]; u18 = u[18]; u19 = u[19]; u20 = u[20];
          u21 = u[21]; u22 = u[22]; u23 = u[23]; u24 = u[24]; u25 = u[25]; u26 = u[26]; u27 = u[27];
          u28 = u[28]; u29 = u[29]; u30 = u[30]; u31 = u[31]; u32 = u[32]; u33 = u[33]; u34 = u[34];
          u35 = u[35]; u36 = u[36]; u37 = u[37]; u38 = u[38]; u39 = u[39]; u40 = u[40]; u41 = u[41]; u42 = u[42];
          u43 = u[43]; u44 = u[44]; u45 = u[45]; u46 = u[46]; u47 = u[47]; u48 = u[48];

          wp[0]+=  uik[0]*u0 + uik[1]*u1 + uik[2]*u2 + uik[3]*u3 + uik[4]*u4 + uik[5]*u5 + uik[6]*u6;
          wp[1]+=  uik[7]*u0 + uik[8]*u1 + uik[9]*u2+ uik[10]*u3+ uik[11]*u4+ uik[12]*u5+ uik[13]*u6;
          wp[2]+= uik[14]*u0+ uik[15]*u1+ uik[16]*u2+ uik[17]*u3+ uik[18]*u4+ uik[19]*u5+ uik[20]*u6;
          wp[3]+= uik[21]*u0+ uik[22]*u1+ uik[23]*u2+ uik[24]*u3+ uik[25]*u4+ uik[26]*u5+ uik[27]*u6;
          wp[4]+= uik[28]*u0+ uik[29]*u1+ uik[30]*u2+ uik[31]*u3+ uik[32]*u4+ uik[33]*u5+ uik[34]*u6;
          wp[5]+= uik[35]*u0+ uik[36]*u1+ uik[37]*u2+ uik[38]*u3+ uik[39]*u4+ uik[40]*u5+ uik[41]*u6;
          wp[6]+= uik[42]*u0+ uik[43]*u1+ uik[44]*u2+ uik[45]*u3+ uik[46]*u4+ uik[47]*u5+ uik[48]*u6;

          wp[7] += uik[0]*u7 + uik[1]*u8 + uik[2]*u9 + uik[3]*u10 + uik[4]*u11 + uik[5]*u12 + uik[6]*u13;
          wp[8] += uik[7]*u7 + uik[8]*u8 + uik[9]*u9+ uik[10]*u10+ uik[11]*u11+ uik[12]*u12+ uik[13]*u13;
          wp[9] +=uik[14]*u7+ uik[15]*u8+ uik[16]*u9+ uik[17]*u10+ uik[18]*u11+ uik[19]*u12+ uik[20]*u13;
          wp[10]+=uik[21]*u7+ uik[22]*u8+ uik[23]*u9+ uik[24]*u10+ uik[25]*u11+ uik[26]*u12+ uik[27]*u13;
          wp[11]+=uik[28]*u7+ uik[29]*u8+ uik[30]*u9+ uik[31]*u10+ uik[32]*u11+ uik[33]*u12+ uik[34]*u13;
          wp[12]+=uik[35]*u7+ uik[36]*u8+ uik[37]*u9+ uik[38]*u10+ uik[39]*u11+ uik[40]*u12+ uik[41]*u13;
          wp[13]+=uik[42]*u7+ uik[43]*u8+ uik[44]*u9+ uik[45]*u10+ uik[46]*u11+ uik[47]*u12+ uik[48]*u13;

          wp[14]+=  uik[0]*u14 + uik[1]*u15 + uik[2]*u16 + uik[3]*u17 + uik[4]*u18 + uik[5]*u19 + uik[6]*u20;
          wp[15]+=  uik[7]*u14 + uik[8]*u15 + uik[9]*u16+ uik[10]*u17+ uik[11]*u18+ uik[12]*u19+ uik[13]*u20;
          wp[16]+= uik[14]*u14+ uik[15]*u15+ uik[16]*u16+ uik[17]*u17+ uik[18]*u18+ uik[19]*u19+ uik[20]*u20;
          wp[17]+= uik[21]*u14+ uik[22]*u15+ uik[23]*u16+ uik[24]*u17+ uik[25]*u18+ uik[26]*u19+ uik[27]*u20;
          wp[18]+= uik[28]*u14+ uik[29]*u15+ uik[30]*u16+ uik[31]*u17+ uik[32]*u18+ uik[33]*u19+ uik[34]*u20;
          wp[19]+= uik[35]*u14+ uik[36]*u15+ uik[37]*u16+ uik[38]*u17+ uik[39]*u18+ uik[40]*u19+ uik[41]*u20;
          wp[20]+= uik[42]*u14+ uik[43]*u15+ uik[44]*u16+ uik[45]*u17+ uik[46]*u18+ uik[47]*u19+ uik[48]*u20;

          wp[21]+=  uik[0]*u21 + uik[1]*u22 + uik[2]*u23 + uik[3]*u24 + uik[4]*u25 + uik[5]*u26 + uik[6]*u27;
          wp[22]+=  uik[7]*u21 + uik[8]*u22 + uik[9]*u23+ uik[10]*u24+ uik[11]*u25+ uik[12]*u26+ uik[13]*u27;
          wp[23]+= uik[14]*u21+ uik[15]*u22+ uik[16]*u23+ uik[17]*u24+ uik[18]*u25+ uik[19]*u26+ uik[20]*u27;
          wp[24]+= uik[21]*u21+ uik[22]*u22+ uik[23]*u23+ uik[24]*u24+ uik[25]*u25+ uik[26]*u26+ uik[27]*u27;
          wp[25]+= uik[28]*u21+ uik[29]*u22+ uik[30]*u23+ uik[31]*u24+ uik[32]*u25+ uik[33]*u26+ uik[34]*u27;
          wp[26]+= uik[35]*u21+ uik[36]*u22+ uik[37]*u23+ uik[38]*u24+ uik[39]*u25+ uik[40]*u26+ uik[41]*u27;
          wp[27]+= uik[42]*u21+ uik[43]*u22+ uik[44]*u23+ uik[45]*u24+ uik[46]*u25+ uik[47]*u26+ uik[48]*u27;

          wp[28]+=  uik[0]*u28 + uik[1]*u29 + uik[2]*u30 + uik[3]*u31 + uik[4]*u32 + uik[5]*u33 + uik[6]*u34;
          wp[29]+=  uik[7]*u28 + uik[8]*u29 + uik[9]*u30+ uik[10]*u31+ uik[11]*u32+ uik[12]*u33+ uik[13]*u34;
          wp[30]+= uik[14]*u28+ uik[15]*u29+ uik[16]*u30+ uik[17]*u31+ uik[18]*u32+ uik[19]*u33+ uik[20]*u34;
          wp[31]+= uik[21]*u28+ uik[22]*u29+ uik[23]*u30+ uik[24]*u31+ uik[25]*u32+ uik[26]*u33+ uik[27]*u34;
          wp[32]+= uik[28]*u28+ uik[29]*u29+ uik[30]*u30+ uik[31]*u31+ uik[32]*u32+ uik[33]*u33+ uik[34]*u34;
          wp[33]+= uik[35]*u28+ uik[36]*u29+ uik[37]*u30+ uik[38]*u31+ uik[39]*u32+ uik[40]*u33+ uik[41]*u34;
          wp[34]+= uik[42]*u28+ uik[43]*u29+ uik[44]*u30+ uik[45]*u31+ uik[46]*u32+ uik[47]*u33+ uik[48]*u34;

          wp[35]+=  uik[0]*u35 + uik[1]*u36 + uik[2]*u37 + uik[3]*u38 + uik[4]*u39 + uik[5]*u40 + uik[6]*u41;
          wp[36]+=  uik[7]*u35 + uik[8]*u36 + uik[9]*u37+ uik[10]*u38+ uik[11]*u39+ uik[12]*u40+ uik[13]*u41;
          wp[37]+= uik[14]*u35+ uik[15]*u36+ uik[16]*u37+ uik[17]*u38+ uik[18]*u39+ uik[19]*u40+ uik[20]*u41;
          wp[38]+= uik[21]*u35+ uik[22]*u36+ uik[23]*u37+ uik[24]*u38+ uik[25]*u39+ uik[26]*u40+ uik[27]*u41;
          wp[39]+= uik[28]*u35+ uik[29]*u36+ uik[30]*u37+ uik[31]*u38+ uik[32]*u39+ uik[33]*u40+ uik[34]*u41;
          wp[40]+= uik[35]*u35+ uik[36]*u36+ uik[37]*u37+ uik[38]*u38+ uik[39]*u39+ uik[40]*u40+ uik[41]*u41;
          wp[41]+= uik[42]*u35+ uik[43]*u36+ uik[44]*u37+ uik[45]*u38+ uik[46]*u39+ uik[47]*u40+ uik[48]*u41;

          wp[42]+=  uik[0]*u42 + uik[1]*u43 + uik[2]*u44 + uik[3]*u45 + uik[4]*u46 + uik[5]*u47 + uik[6]*u48;
          wp[43]+=  uik[7]*u42 + uik[8]*u43 + uik[9]*u44+ uik[10]*u45+ uik[11]*u46+ uik[12]*u47+ uik[13]*u48;
          wp[44]+= uik[14]*u42+ uik[15]*u43+ uik[16]*u44+ uik[17]*u45+ uik[18]*u46+ uik[19]*u47+ uik[20]*u48;
          wp[45]+= uik[21]*u42+ uik[22]*u43+ uik[23]*u44+ uik[24]*u45+ uik[25]*u46+ uik[26]*u47+ uik[27]*u48;
          wp[46]+= uik[28]*u42+ uik[29]*u43+ uik[30]*u44+ uik[31]*u45+ uik[32]*u46+ uik[33]*u47+ uik[34]*u48;
          wp[47]+= uik[35]*u42+ uik[36]*u43+ uik[37]*u44+ uik[38]*u45+ uik[39]*u46+ uik[40]*u47+ uik[41]*u48;
          wp[48]+= uik[42]*u42+ uik[43]*u43+ uik[44]*u44+ uik[45]*u45+ uik[46]*u46+ uik[47]*u47+ uik[48]*u48;
        }
        ierr = PetscLogFlops(2.0*343.0*(jmax-jmin));CHKERRQ(ierr);

        /* ... add i to row list for next nonzero entry */
        il[i] = jmin;             /* update il(i) in column k+1, ... mbs-1 */
        j     = bj[jmin];
        jl[i] = jl[j]; jl[j] = i; /* update jl */
      }
      i = nexti;
    }

    /* save nonzero entries in k-th row of U ... */

    /* invert diagonal block */
    d    = ba+k*49;
    ierr = PetscArraycpy(d,dk,49);CHKERRQ(ierr);
    ierr = PetscKernel_A_gets_inverse_A_7(d,shift,allowzeropivot,&zeropivotdetected);CHKERRQ(ierr);
    if (zeropivotdetected) C->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;

    jmin = bi[k]; jmax = bi[k+1];
    if (jmin < jmax) {
      for (j=jmin; j<jmax; j++) {
        vj = bj[j];            /* block col. index of U */
        u  = ba + j*49;
        wp = w + vj*49;
        for (k1=0; k1<49; k1++) {
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

  C->ops->solve          = MatSolve_SeqSBAIJ_7_NaturalOrdering_inplace;
  C->ops->solvetranspose = MatSolve_SeqSBAIJ_7_NaturalOrdering_inplace;
  C->ops->forwardsolve   = MatForwardSolve_SeqSBAIJ_7_NaturalOrdering_inplace;
  C->ops->backwardsolve  = MatBackwardSolve_SeqSBAIJ_7_NaturalOrdering_inplace;
  C->assembled           = PETSC_TRUE;
  C->preallocated        = PETSC_TRUE;

  ierr = PetscLogFlops(1.3333*343*b->mbs);CHKERRQ(ierr); /* from inverting diagonal blocks */
  PetscFunctionReturn(0);
}
