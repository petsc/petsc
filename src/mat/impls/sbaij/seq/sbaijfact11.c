#include "sbaij.h"
#include "src/inline/ilu.h"

/* Version for when blocks are 7 by 7 */
#undef __FUNC__  
#define __FUNC__ "MatCholeskyFactorNumeric_SeqSBAIJ_7"
int MatCholeskyFactorNumeric_SeqSBAIJ_7(Mat A,Mat *B)
{
  Mat                C = *B;
  Mat_SeqSBAIJ       *a = (Mat_SeqSBAIJ*)A->data,*b = (Mat_SeqSBAIJ *)C->data;
  IS                 perm = b->row;
  int                *perm_ptr,ierr,i,j,mbs=a->mbs,*bi=b->i,*bj=b->j;
  int                *ai,*aj,*a2anew,k,k1,jmin,jmax,*jl,*il,vj,nexti,ili;
  MatScalar          *ba = b->a,*aa,*ap,*dk,*uik;
  MatScalar          *u,*d,*w,*wp;

  PetscFunctionBegin;
  /* initialization */
  ierr = PetscMalloc(49*mbs*sizeof(MatScalar),&w);CHKERRQ(ierr);
  ierr = PetscMemzero(w,49*mbs*sizeof(MatScalar));CHKERRQ(ierr); 
  ierr = PetscMalloc(2*mbs*sizeof(int),&il);CHKERRQ(ierr);
  jl = il + mbs;
  for (i=0; i<mbs; i++) {
    jl[i] = mbs; il[0] = 0;
  }
  ierr = PetscMalloc(98*sizeof(MatScalar),&dk);CHKERRQ(ierr);
  uik  = dk + 49;    
  ierr = ISGetIndices(perm,&perm_ptr);CHKERRQ(ierr);

  /* check permutation */
  if (!a->permute){
    ai = a->i; aj = a->j; aa = a->a;
  } else {
    ai = a->inew; aj = a->jnew; 
    ierr = PetscMalloc(49*ai[mbs]*sizeof(MatScalar),&aa);CHKERRQ(ierr); 
    ierr = PetscMemcpy(aa,a->a,49*ai[mbs]*sizeof(MatScalar));CHKERRQ(ierr);
    ierr = PetscMalloc(ai[mbs]*sizeof(int),&a2anew);CHKERRQ(ierr); 
    ierr = PetscMemcpy(a2anew,a->a2anew,(ai[mbs])*sizeof(int));CHKERRQ(ierr);

    for (i=0; i<mbs; i++){
      jmin = ai[i]; jmax = ai[i+1];
      for (j=jmin; j<jmax; j++){
        while (a2anew[j] != j){  
          k = a2anew[j]; a2anew[j] = a2anew[k]; a2anew[k] = k;  
          for (k1=0; k1<49; k1++){
            dk[k1]       = aa[k*49+k1]; 
            aa[k*49+k1] = aa[j*49+k1]; 
            aa[j*49+k1] = dk[k1];   
          }
        }
        /* transform columnoriented blocks that lie in the lower triangle to roworiented blocks */
        if (i > aj[j]){ 
          /* printf("change orientation, row: %d, col: %d\n",i,aj[j]); */
          ap = aa + j*49;                     /* ptr to the beginning of j-th block of aa */
          for (k=0; k<49; k++) dk[k] = ap[k]; /* dk <- j-th block of aa */
          for (k=0; k<7; k++){               /* j-th block of aa <- dk^T */
            for (k1=0; k1<7; k1++) *ap++ = dk[k + 7*k1];         
          }
        }
      }
    }
    ierr = PetscFree(a2anew);CHKERRQ(ierr); 
  }
  
  /* for each row k */
  for (k = 0; k<mbs; k++){

    /*initialize k-th row with elements nonzero in row perm(k) of A */
    jmin = ai[perm_ptr[k]]; jmax = ai[perm_ptr[k]+1];
    if (jmin < jmax) {
      ap = aa + jmin*49;
      for (j = jmin; j < jmax; j++){
        vj = perm_ptr[aj[j]];         /* block col. index */  
        wp = w + vj*49;
        for (i=0; i<49; i++) *wp++ = *ap++;        
      } 
    } 

    /* modify k-th row by adding in those rows i with U(i,k) != 0 */
    ierr = PetscMemcpy(dk,w+k*49,49*sizeof(MatScalar));CHKERRQ(ierr); 
    i = jl[k]; /* first row to be added to k_th row  */  

    while (i < mbs){
      nexti = jl[i]; /* next row to be added to k_th row */

      /* compute multiplier */
      ili = il[i];  /* index of first nonzero element in U(i,k:bms-1) */

      /* uik = -inv(Di)*U_bar(i,k) */
      d = ba + i*49;
      u    = ba + ili*49;

      uik[0] = -(d[0]*u[0] + d[7]*u[1]+ d[14]*u[2]+ d[21]*u[3]+ d[28]*u[4]+ d[35]*u[5]+ d[42]*u[6]);
      uik[1] = -(d[1]*u[0] + d[8]*u[1]+ d[15]*u[2]+ d[22]*u[3]+ d[29]*u[4]+ d[36]*u[5]+ d[43]*u[6]);
      uik[2] = -(d[2]*u[0] + d[9]*u[1]+ d[16]*u[2]+ d[23]*u[3]+ d[30]*u[4]+ d[37]*u[5]+ d[44]*u[6]);
      uik[3] = -(d[3]*u[0]+ d[10]*u[1]+ d[17]*u[2]+ d[24]*u[3]+ d[31]*u[4]+ d[38]*u[5]+ d[45]*u[6]);
      uik[4] = -(d[4]*u[0]+ d[11]*u[1]+ d[18]*u[2]+ d[25]*u[3]+ d[32]*u[4]+ d[39]*u[5]+ d[46]*u[6]);
      uik[5] = -(d[5]*u[0]+ d[12]*u[1]+ d[19]*u[2]+ d[26]*u[3]+ d[33]*u[4]+ d[40]*u[5]+ d[47]*u[6]);
      uik[6] = -(d[6]*u[0]+ d[13]*u[1]+ d[20]*u[2]+ d[27]*u[3]+ d[34]*u[4]+ d[41]*u[5]+ d[48]*u[6]);

      uik[7] = -(d[0]*u[7] + d[7]*u[8]+ d[14]*u[9]+ d[21]*u[10]+ d[28]*u[11]+ d[35]*u[12]+ d[42]*u[13]);
      uik[8] = -(d[1]*u[7] + d[8]*u[8]+ d[15]*u[9]+ d[22]*u[10]+ d[29]*u[11]+ d[36]*u[12]+ d[43]*u[13]);
      uik[9] = -(d[2]*u[7] + d[9]*u[8]+ d[16]*u[9]+ d[23]*u[10]+ d[30]*u[11]+ d[37]*u[12]+ d[44]*u[13]);
      uik[10]= -(d[3]*u[7]+ d[10]*u[8]+ d[17]*u[9]+ d[24]*u[10]+ d[31]*u[11]+ d[38]*u[12]+ d[45]*u[13]);
      uik[11]= -(d[4]*u[7]+ d[11]*u[8]+ d[18]*u[9]+ d[25]*u[10]+ d[32]*u[11]+ d[39]*u[12]+ d[46]*u[13]);
      uik[12]= -(d[5]*u[7]+ d[12]*u[8]+ d[19]*u[9]+ d[26]*u[10]+ d[33]*u[11]+ d[40]*u[12]+ d[47]*u[13]);
      uik[13]= -(d[6]*u[7]+ d[13]*u[8]+ d[20]*u[9]+ d[27]*u[10]+ d[34]*u[11]+ d[41]*u[12]+ d[48]*u[13]);

      uik[14]= -(d[0]*u[14] + d[7]*u[15]+ d[14]*u[16]+ d[21]*u[17]+ d[28]*u[18]+ d[35]*u[19]+ d[42]*u[20]);
      uik[15]= -(d[1]*u[14] + d[8]*u[15]+ d[15]*u[16]+ d[22]*u[17]+ d[29]*u[18]+ d[36]*u[19]+ d[43]*u[20]);
      uik[16]= -(d[2]*u[14] + d[9]*u[15]+ d[16]*u[16]+ d[23]*u[17]+ d[30]*u[18]+ d[37]*u[19]+ d[44]*u[20]);
      uik[17]= -(d[3]*u[14]+ d[10]*u[15]+ d[17]*u[16]+ d[24]*u[17]+ d[31]*u[18]+ d[38]*u[19]+ d[45]*u[20]);
      uik[18]= -(d[4]*u[14]+ d[11]*u[15]+ d[18]*u[16]+ d[25]*u[17]+ d[32]*u[18]+ d[39]*u[19]+ d[46]*u[20]);
      uik[19]= -(d[5]*u[14]+ d[12]*u[15]+ d[19]*u[16]+ d[26]*u[17]+ d[33]*u[18]+ d[40]*u[19]+ d[47]*u[20]);
      uik[20]= -(d[6]*u[14]+ d[13]*u[15]+ d[20]*u[16]+ d[27]*u[17]+ d[34]*u[18]+ d[41]*u[19]+ d[48]*u[20]);

      uik[21]= -(d[0]*u[21] + d[7]*u[22]+ d[14]*u[23]+ d[21]*u[24]+ d[28]*u[25]+ d[35]*u[26]+ d[42]*u[27]);
      uik[22]= -(d[1]*u[21] + d[8]*u[22]+ d[15]*u[23]+ d[22]*u[24]+ d[29]*u[25]+ d[36]*u[26]+ d[43]*u[27]);
      uik[23]= -(d[2]*u[21] + d[9]*u[22]+ d[16]*u[23]+ d[23]*u[24]+ d[30]*u[25]+ d[37]*u[26]+ d[44]*u[27]);
      uik[24]= -(d[3]*u[21]+ d[10]*u[22]+ d[17]*u[23]+ d[24]*u[24]+ d[31]*u[25]+ d[38]*u[26]+ d[45]*u[27]);
      uik[25]= -(d[4]*u[21]+ d[11]*u[22]+ d[18]*u[23]+ d[25]*u[24]+ d[32]*u[25]+ d[39]*u[26]+ d[46]*u[27]);
      uik[26]= -(d[5]*u[21]+ d[12]*u[22]+ d[19]*u[23]+ d[26]*u[24]+ d[33]*u[25]+ d[40]*u[26]+ d[47]*u[27]);
      uik[27]= -(d[6]*u[21]+ d[13]*u[22]+ d[20]*u[23]+ d[27]*u[24]+ d[34]*u[25]+ d[41]*u[26]+ d[48]*u[27]);

      uik[28]= -(d[0]*u[28] + d[7]*u[29]+ d[14]*u[30]+ d[21]*u[31]+ d[28]*u[32]+ d[35]*u[33]+ d[42]*u[34]);
      uik[29]= -(d[1]*u[28] + d[8]*u[29]+ d[15]*u[30]+ d[22]*u[31]+ d[29]*u[32]+ d[36]*u[33]+ d[43]*u[34]);
      uik[30]= -(d[2]*u[28] + d[9]*u[29]+ d[16]*u[30]+ d[23]*u[31]+ d[30]*u[32]+ d[37]*u[33]+ d[44]*u[34]);
      uik[31]= -(d[3]*u[28]+ d[10]*u[29]+ d[17]*u[30]+ d[24]*u[31]+ d[31]*u[32]+ d[38]*u[33]+ d[45]*u[34]);
      uik[32]= -(d[4]*u[28]+ d[11]*u[29]+ d[18]*u[30]+ d[25]*u[31]+ d[32]*u[32]+ d[39]*u[33]+ d[46]*u[34]);
      uik[33]= -(d[5]*u[28]+ d[12]*u[29]+ d[19]*u[30]+ d[26]*u[31]+ d[33]*u[32]+ d[40]*u[33]+ d[47]*u[34]);
      uik[34]= -(d[6]*u[28]+ d[13]*u[29]+ d[20]*u[30]+ d[27]*u[31]+ d[34]*u[32]+ d[41]*u[33]+ d[48]*u[34]);

      uik[35]= -(d[0]*u[35] + d[7]*u[36]+ d[14]*u[37]+ d[21]*u[38]+ d[28]*u[39]+ d[35]*u[40]+ d[42]*u[41]);
      uik[36]= -(d[1]*u[35] + d[8]*u[36]+ d[15]*u[37]+ d[22]*u[38]+ d[29]*u[39]+ d[36]*u[40]+ d[43]*u[41]);
      uik[37]= -(d[2]*u[35] + d[9]*u[36]+ d[16]*u[37]+ d[23]*u[38]+ d[30]*u[39]+ d[37]*u[40]+ d[44]*u[41]);
      uik[38]= -(d[3]*u[35]+ d[10]*u[36]+ d[17]*u[37]+ d[24]*u[38]+ d[31]*u[39]+ d[38]*u[40]+ d[45]*u[41]);
      uik[39]= -(d[4]*u[35]+ d[11]*u[36]+ d[18]*u[37]+ d[25]*u[38]+ d[32]*u[39]+ d[39]*u[40]+ d[46]*u[41]);
      uik[40]= -(d[5]*u[35]+ d[12]*u[36]+ d[19]*u[37]+ d[26]*u[38]+ d[33]*u[39]+ d[40]*u[40]+ d[47]*u[41]);
      uik[41]= -(d[6]*u[35]+ d[13]*u[36]+ d[20]*u[37]+ d[27]*u[38]+ d[34]*u[39]+ d[41]*u[40]+ d[48]*u[41]);

      uik[42]= -(d[0]*u[42] + d[7]*u[43]+ d[14]*u[44]+ d[21]*u[45]+ d[28]*u[46]+ d[35]*u[47]+ d[42]*u[48]);
      uik[43]= -(d[1]*u[42] + d[8]*u[43]+ d[15]*u[44]+ d[22]*u[45]+ d[29]*u[46]+ d[36]*u[47]+ d[43]*u[48]);
      uik[44]= -(d[2]*u[42] + d[9]*u[43]+ d[16]*u[44]+ d[23]*u[45]+ d[30]*u[46]+ d[37]*u[47]+ d[44]*u[48]);
      uik[45]= -(d[3]*u[42]+ d[10]*u[43]+ d[17]*u[44]+ d[24]*u[45]+ d[31]*u[46]+ d[38]*u[47]+ d[45]*u[48]);
      uik[46]= -(d[4]*u[42]+ d[11]*u[43]+ d[18]*u[44]+ d[25]*u[45]+ d[32]*u[46]+ d[39]*u[47]+ d[46]*u[48]);
      uik[47]= -(d[5]*u[42]+ d[12]*u[43]+ d[19]*u[44]+ d[26]*u[45]+ d[33]*u[46]+ d[40]*u[47]+ d[47]*u[48]);
      uik[48]= -(d[6]*u[42]+ d[13]*u[43]+ d[20]*u[44]+ d[27]*u[45]+ d[34]*u[46]+ d[41]*u[47]+ d[48]*u[48]);

      /* update D(k) += -U(i,k)^T * U_bar(i,k) */  
      dk[0]+=  uik[0]*u[0] + uik[1]*u[1] + uik[2]*u[2] + uik[3]*u[3] + uik[4]*u[4] + uik[5]*u[5] + uik[6]*u[6];
      dk[1]+=  uik[7]*u[0] + uik[8]*u[1] + uik[9]*u[2]+ uik[10]*u[3]+ uik[11]*u[4]+ uik[12]*u[5]+ uik[13]*u[6];
      dk[2]+= uik[14]*u[0]+ uik[15]*u[1]+ uik[16]*u[2]+ uik[17]*u[3]+ uik[18]*u[4]+ uik[19]*u[5]+ uik[20]*u[6];
      dk[3]+= uik[21]*u[0]+ uik[22]*u[1]+ uik[23]*u[2]+ uik[24]*u[3]+ uik[25]*u[4]+ uik[26]*u[5]+ uik[27]*u[6];
      dk[4]+= uik[28]*u[0]+ uik[29]*u[1]+ uik[30]*u[2]+ uik[31]*u[3]+ uik[32]*u[4]+ uik[33]*u[5]+ uik[34]*u[6];
      dk[5]+= uik[35]*u[0]+ uik[36]*u[1]+ uik[37]*u[2]+ uik[38]*u[3]+ uik[39]*u[4]+ uik[40]*u[5]+ uik[41]*u[6];
      dk[6]+= uik[42]*u[0]+ uik[43]*u[1]+ uik[44]*u[2]+ uik[45]*u[3]+ uik[46]*u[4]+ uik[47]*u[5]+ uik[48]*u[6];

      dk[7]+=  uik[0]*u[7] + uik[1]*u[8] + uik[2]*u[9] + uik[3]*u[10] + uik[4]*u[11] + uik[5]*u[12] + uik[6]*u[13];
      dk[8]+=  uik[7]*u[7] + uik[8]*u[8] + uik[9]*u[9]+ uik[10]*u[10]+ uik[11]*u[11]+ uik[12]*u[12]+ uik[13]*u[13];
      dk[9]+= uik[14]*u[7]+ uik[15]*u[8]+ uik[16]*u[9]+ uik[17]*u[10]+ uik[18]*u[11]+ uik[19]*u[12]+ uik[20]*u[13];
      dk[10]+=uik[21]*u[7]+ uik[22]*u[8]+ uik[23]*u[9]+ uik[24]*u[10]+ uik[25]*u[11]+ uik[26]*u[12]+ uik[27]*u[13];
      dk[11]+=uik[28]*u[7]+ uik[29]*u[8]+ uik[30]*u[9]+ uik[31]*u[10]+ uik[32]*u[11]+ uik[33]*u[12]+ uik[34]*u[13];
      dk[12]+=uik[35]*u[7]+ uik[36]*u[8]+ uik[37]*u[9]+ uik[38]*u[10]+ uik[39]*u[11]+ uik[40]*u[12]+ uik[41]*u[13];
      dk[13]+=uik[42]*u[7]+ uik[43]*u[8]+ uik[44]*u[9]+ uik[45]*u[10]+ uik[46]*u[11]+ uik[47]*u[12]+ uik[48]*u[13];

      dk[14]+=  uik[0]*u[14] + uik[1]*u[15] + uik[2]*u[16] + uik[3]*u[17] + uik[4]*u[18] + uik[5]*u[19] + uik[6]*u[20];
      dk[15]+=  uik[7]*u[14] + uik[8]*u[15] + uik[9]*u[16]+ uik[10]*u[17]+ uik[11]*u[18]+ uik[12]*u[19]+ uik[13]*u[20];
      dk[16]+= uik[14]*u[14]+ uik[15]*u[15]+ uik[16]*u[16]+ uik[17]*u[17]+ uik[18]*u[18]+ uik[19]*u[19]+ uik[20]*u[20];
      dk[17]+= uik[21]*u[14]+ uik[22]*u[15]+ uik[23]*u[16]+ uik[24]*u[17]+ uik[25]*u[18]+ uik[26]*u[19]+ uik[27]*u[20];
      dk[18]+= uik[28]*u[14]+ uik[29]*u[15]+ uik[30]*u[16]+ uik[31]*u[17]+ uik[32]*u[18]+ uik[33]*u[19]+ uik[34]*u[20];
      dk[19]+= uik[35]*u[14]+ uik[36]*u[15]+ uik[37]*u[16]+ uik[38]*u[17]+ uik[39]*u[18]+ uik[40]*u[19]+ uik[41]*u[20];
      dk[20]+= uik[42]*u[14]+ uik[43]*u[15]+ uik[44]*u[16]+ uik[45]*u[17]+ uik[46]*u[18]+ uik[47]*u[19]+ uik[48]*u[20];

      dk[21]+=  uik[0]*u[21] + uik[1]*u[22] + uik[2]*u[23] + uik[3]*u[24] + uik[4]*u[25] + uik[5]*u[26] + uik[6]*u[27];
      dk[22]+=  uik[7]*u[21] + uik[8]*u[22] + uik[9]*u[23]+ uik[10]*u[24]+ uik[11]*u[25]+ uik[12]*u[26]+ uik[13]*u[27];
      dk[23]+= uik[14]*u[21]+ uik[15]*u[22]+ uik[16]*u[23]+ uik[17]*u[24]+ uik[18]*u[25]+ uik[19]*u[26]+ uik[20]*u[27];
      dk[24]+= uik[21]*u[21]+ uik[22]*u[22]+ uik[23]*u[23]+ uik[24]*u[24]+ uik[25]*u[25]+ uik[26]*u[26]+ uik[27]*u[27];
      dk[25]+= uik[28]*u[21]+ uik[29]*u[22]+ uik[30]*u[23]+ uik[31]*u[24]+ uik[32]*u[25]+ uik[33]*u[26]+ uik[34]*u[27];
      dk[26]+= uik[35]*u[21]+ uik[36]*u[22]+ uik[37]*u[23]+ uik[38]*u[24]+ uik[39]*u[25]+ uik[40]*u[26]+ uik[41]*u[27];
      dk[27]+= uik[42]*u[21]+ uik[43]*u[22]+ uik[44]*u[23]+ uik[45]*u[24]+ uik[46]*u[25]+ uik[47]*u[26]+ uik[48]*u[27];

      dk[28]+=  uik[0]*u[28] + uik[1]*u[29] + uik[2]*u[30] + uik[3]*u[31] + uik[4]*u[32] + uik[5]*u[33] + uik[6]*u[34];
      dk[29]+=  uik[7]*u[28] + uik[8]*u[29] + uik[9]*u[30]+ uik[10]*u[31]+ uik[11]*u[32]+ uik[12]*u[33]+ uik[13]*u[34];
      dk[30]+= uik[14]*u[28]+ uik[15]*u[29]+ uik[16]*u[30]+ uik[17]*u[31]+ uik[18]*u[32]+ uik[19]*u[33]+ uik[20]*u[34];
      dk[31]+= uik[21]*u[28]+ uik[22]*u[29]+ uik[23]*u[30]+ uik[24]*u[31]+ uik[25]*u[32]+ uik[26]*u[33]+ uik[27]*u[34];
      dk[32]+= uik[28]*u[28]+ uik[29]*u[29]+ uik[30]*u[30]+ uik[31]*u[31]+ uik[32]*u[32]+ uik[33]*u[33]+ uik[34]*u[34];
      dk[33]+= uik[35]*u[28]+ uik[36]*u[29]+ uik[37]*u[30]+ uik[38]*u[31]+ uik[39]*u[32]+ uik[40]*u[33]+ uik[41]*u[34];
      dk[34]+= uik[42]*u[28]+ uik[43]*u[29]+ uik[44]*u[30]+ uik[45]*u[31]+ uik[46]*u[32]+ uik[47]*u[33]+ uik[48]*u[34];

      dk[35]+=  uik[0]*u[35] + uik[1]*u[36] + uik[2]*u[37] + uik[3]*u[38] + uik[4]*u[39] + uik[5]*u[40] + uik[6]*u[41];
      dk[36]+=  uik[7]*u[35] + uik[8]*u[36] + uik[9]*u[37]+ uik[10]*u[38]+ uik[11]*u[39]+ uik[12]*u[40]+ uik[13]*u[41];
      dk[37]+= uik[14]*u[35]+ uik[15]*u[36]+ uik[16]*u[37]+ uik[17]*u[38]+ uik[18]*u[39]+ uik[19]*u[40]+ uik[20]*u[41];
      dk[38]+= uik[21]*u[35]+ uik[22]*u[36]+ uik[23]*u[37]+ uik[24]*u[38]+ uik[25]*u[39]+ uik[26]*u[40]+ uik[27]*u[41];
      dk[39]+= uik[28]*u[35]+ uik[29]*u[36]+ uik[30]*u[37]+ uik[31]*u[38]+ uik[32]*u[39]+ uik[33]*u[40]+ uik[34]*u[41];
      dk[40]+= uik[35]*u[35]+ uik[36]*u[36]+ uik[37]*u[37]+ uik[38]*u[38]+ uik[39]*u[39]+ uik[40]*u[40]+ uik[41]*u[41];
      dk[41]+= uik[42]*u[35]+ uik[43]*u[36]+ uik[44]*u[37]+ uik[45]*u[38]+ uik[46]*u[39]+ uik[47]*u[40]+ uik[48]*u[41];

      dk[42]+=  uik[0]*u[42] + uik[1]*u[43] + uik[2]*u[44] + uik[3]*u[45] + uik[4]*u[46] + uik[5]*u[47] + uik[6]*u[48];
      dk[43]+=  uik[7]*u[42] + uik[8]*u[43] + uik[9]*u[44]+ uik[10]*u[45]+ uik[11]*u[46]+ uik[12]*u[47]+ uik[13]*u[48];
      dk[44]+= uik[14]*u[42]+ uik[15]*u[43]+ uik[16]*u[44]+ uik[17]*u[45]+ uik[18]*u[46]+ uik[19]*u[47]+ uik[20]*u[48];
      dk[45]+= uik[21]*u[42]+ uik[22]*u[43]+ uik[23]*u[44]+ uik[24]*u[45]+ uik[25]*u[46]+ uik[26]*u[47]+ uik[27]*u[48];
      dk[46]+= uik[28]*u[42]+ uik[29]*u[43]+ uik[30]*u[44]+ uik[31]*u[45]+ uik[32]*u[46]+ uik[33]*u[47]+ uik[34]*u[48];
      dk[47]+= uik[35]*u[42]+ uik[36]*u[43]+ uik[37]*u[44]+ uik[38]*u[45]+ uik[39]*u[46]+ uik[40]*u[47]+ uik[41]*u[48];
      dk[48]+= uik[42]*u[42]+ uik[43]*u[43]+ uik[44]*u[44]+ uik[45]*u[45]+ uik[46]*u[46]+ uik[47]*u[47]+ uik[48]*u[48];

      /* update -U(i,k) */
      ierr = PetscMemcpy(ba+ili*49,uik,49*sizeof(MatScalar));CHKERRQ(ierr); 

      /* add multiple of row i to k-th row ... */
      jmin = ili + 1; jmax = bi[i+1];
      if (jmin < jmax){
        for (j=jmin; j<jmax; j++) {
          /* w += -U(i,k)^T * U_bar(i,j) */
          wp = w + bj[j]*49;
          u = ba + j*49;

          wp[0]+=  uik[0]*u[0] + uik[1]*u[1] + uik[2]*u[2] + uik[3]*u[3] + uik[4]*u[4] + uik[5]*u[5] + uik[6]*u[6];
          wp[1]+=  uik[7]*u[0] + uik[8]*u[1] + uik[9]*u[2]+ uik[10]*u[3]+ uik[11]*u[4]+ uik[12]*u[5]+ uik[13]*u[6];
          wp[2]+= uik[14]*u[0]+ uik[15]*u[1]+ uik[16]*u[2]+ uik[17]*u[3]+ uik[18]*u[4]+ uik[19]*u[5]+ uik[20]*u[6];
          wp[3]+= uik[21]*u[0]+ uik[22]*u[1]+ uik[23]*u[2]+ uik[24]*u[3]+ uik[25]*u[4]+ uik[26]*u[5]+ uik[27]*u[6];
          wp[4]+= uik[28]*u[0]+ uik[29]*u[1]+ uik[30]*u[2]+ uik[31]*u[3]+ uik[32]*u[4]+ uik[33]*u[5]+ uik[34]*u[6];
          wp[5]+= uik[35]*u[0]+ uik[36]*u[1]+ uik[37]*u[2]+ uik[38]*u[3]+ uik[39]*u[4]+ uik[40]*u[5]+ uik[41]*u[6];
          wp[6]+= uik[42]*u[0]+ uik[43]*u[1]+ uik[44]*u[2]+ uik[45]*u[3]+ uik[46]*u[4]+ uik[47]*u[5]+ uik[48]*u[6];

          wp[7]+=  uik[0]*u[7] + uik[1]*u[8] + uik[2]*u[9] + uik[3]*u[10] + uik[4]*u[11] + uik[5]*u[12] + uik[6]*u[13];
          wp[8]+=  uik[7]*u[7] + uik[8]*u[8] + uik[9]*u[9]+ uik[10]*u[10]+ uik[11]*u[11]+ uik[12]*u[12]+ uik[13]*u[13];
          wp[9]+= uik[14]*u[7]+ uik[15]*u[8]+ uik[16]*u[9]+ uik[17]*u[10]+ uik[18]*u[11]+ uik[19]*u[12]+ uik[20]*u[13];
          wp[10]+=uik[21]*u[7]+ uik[22]*u[8]+ uik[23]*u[9]+ uik[24]*u[10]+ uik[25]*u[11]+ uik[26]*u[12]+ uik[27]*u[13];
          wp[11]+=uik[28]*u[7]+ uik[29]*u[8]+ uik[30]*u[9]+ uik[31]*u[10]+ uik[32]*u[11]+ uik[33]*u[12]+ uik[34]*u[13];
          wp[12]+=uik[35]*u[7]+ uik[36]*u[8]+ uik[37]*u[9]+ uik[38]*u[10]+ uik[39]*u[11]+ uik[40]*u[12]+ uik[41]*u[13];
          wp[13]+=uik[42]*u[7]+ uik[43]*u[8]+ uik[44]*u[9]+ uik[45]*u[10]+ uik[46]*u[11]+ uik[47]*u[12]+ uik[48]*u[13];

          wp[14]+=  uik[0]*u[14] + uik[1]*u[15] + uik[2]*u[16] + uik[3]*u[17] + uik[4]*u[18] + uik[5]*u[19] + uik[6]*u[20];
          wp[15]+=  uik[7]*u[14] + uik[8]*u[15] + uik[9]*u[16]+ uik[10]*u[17]+ uik[11]*u[18]+ uik[12]*u[19]+ uik[13]*u[20];
          wp[16]+= uik[14]*u[14]+ uik[15]*u[15]+ uik[16]*u[16]+ uik[17]*u[17]+ uik[18]*u[18]+ uik[19]*u[19]+ uik[20]*u[20];
          wp[17]+= uik[21]*u[14]+ uik[22]*u[15]+ uik[23]*u[16]+ uik[24]*u[17]+ uik[25]*u[18]+ uik[26]*u[19]+ uik[27]*u[20];
          wp[18]+= uik[28]*u[14]+ uik[29]*u[15]+ uik[30]*u[16]+ uik[31]*u[17]+ uik[32]*u[18]+ uik[33]*u[19]+ uik[34]*u[20];
          wp[19]+= uik[35]*u[14]+ uik[36]*u[15]+ uik[37]*u[16]+ uik[38]*u[17]+ uik[39]*u[18]+ uik[40]*u[19]+ uik[41]*u[20];
          wp[20]+= uik[42]*u[14]+ uik[43]*u[15]+ uik[44]*u[16]+ uik[45]*u[17]+ uik[46]*u[18]+ uik[47]*u[19]+ uik[48]*u[20];

          wp[21]+=  uik[0]*u[21] + uik[1]*u[22] + uik[2]*u[23] + uik[3]*u[24] + uik[4]*u[25] + uik[5]*u[26] + uik[6]*u[27];
          wp[22]+=  uik[7]*u[21] + uik[8]*u[22] + uik[9]*u[23]+ uik[10]*u[24]+ uik[11]*u[25]+ uik[12]*u[26]+ uik[13]*u[27];
          wp[23]+= uik[14]*u[21]+ uik[15]*u[22]+ uik[16]*u[23]+ uik[17]*u[24]+ uik[18]*u[25]+ uik[19]*u[26]+ uik[20]*u[27];
          wp[24]+= uik[21]*u[21]+ uik[22]*u[22]+ uik[23]*u[23]+ uik[24]*u[24]+ uik[25]*u[25]+ uik[26]*u[26]+ uik[27]*u[27];
          wp[25]+= uik[28]*u[21]+ uik[29]*u[22]+ uik[30]*u[23]+ uik[31]*u[24]+ uik[32]*u[25]+ uik[33]*u[26]+ uik[34]*u[27];
          wp[26]+= uik[35]*u[21]+ uik[36]*u[22]+ uik[37]*u[23]+ uik[38]*u[24]+ uik[39]*u[25]+ uik[40]*u[26]+ uik[41]*u[27];
          wp[27]+= uik[42]*u[21]+ uik[43]*u[22]+ uik[44]*u[23]+ uik[45]*u[24]+ uik[46]*u[25]+ uik[47]*u[26]+ uik[48]*u[27];

          wp[28]+=  uik[0]*u[28] + uik[1]*u[29] + uik[2]*u[30] + uik[3]*u[31] + uik[4]*u[32] + uik[5]*u[33] + uik[6]*u[34];
          wp[29]+=  uik[7]*u[28] + uik[8]*u[29] + uik[9]*u[30]+ uik[10]*u[31]+ uik[11]*u[32]+ uik[12]*u[33]+ uik[13]*u[34];
          wp[30]+= uik[14]*u[28]+ uik[15]*u[29]+ uik[16]*u[30]+ uik[17]*u[31]+ uik[18]*u[32]+ uik[19]*u[33]+ uik[20]*u[34];
          wp[31]+= uik[21]*u[28]+ uik[22]*u[29]+ uik[23]*u[30]+ uik[24]*u[31]+ uik[25]*u[32]+ uik[26]*u[33]+ uik[27]*u[34];
          wp[32]+= uik[28]*u[28]+ uik[29]*u[29]+ uik[30]*u[30]+ uik[31]*u[31]+ uik[32]*u[32]+ uik[33]*u[33]+ uik[34]*u[34];
          wp[33]+= uik[35]*u[28]+ uik[36]*u[29]+ uik[37]*u[30]+ uik[38]*u[31]+ uik[39]*u[32]+ uik[40]*u[33]+ uik[41]*u[34];
          wp[34]+= uik[42]*u[28]+ uik[43]*u[29]+ uik[44]*u[30]+ uik[45]*u[31]+ uik[46]*u[32]+ uik[47]*u[33]+ uik[48]*u[34];

          wp[35]+=  uik[0]*u[35] + uik[1]*u[36] + uik[2]*u[37] + uik[3]*u[38] + uik[4]*u[39] + uik[5]*u[40] + uik[6]*u[41];
          wp[36]+=  uik[7]*u[35] + uik[8]*u[36] + uik[9]*u[37]+ uik[10]*u[38]+ uik[11]*u[39]+ uik[12]*u[40]+ uik[13]*u[41];
          wp[37]+= uik[14]*u[35]+ uik[15]*u[36]+ uik[16]*u[37]+ uik[17]*u[38]+ uik[18]*u[39]+ uik[19]*u[40]+ uik[20]*u[41];
          wp[38]+= uik[21]*u[35]+ uik[22]*u[36]+ uik[23]*u[37]+ uik[24]*u[38]+ uik[25]*u[39]+ uik[26]*u[40]+ uik[27]*u[41];
          wp[39]+= uik[28]*u[35]+ uik[29]*u[36]+ uik[30]*u[37]+ uik[31]*u[38]+ uik[32]*u[39]+ uik[33]*u[40]+ uik[34]*u[41];
          wp[40]+= uik[35]*u[35]+ uik[36]*u[36]+ uik[37]*u[37]+ uik[38]*u[38]+ uik[39]*u[39]+ uik[40]*u[40]+ uik[41]*u[41];
          wp[41]+= uik[42]*u[35]+ uik[43]*u[36]+ uik[44]*u[37]+ uik[45]*u[38]+ uik[46]*u[39]+ uik[47]*u[40]+ uik[48]*u[41];

          wp[42]+=  uik[0]*u[42] + uik[1]*u[43] + uik[2]*u[44] + uik[3]*u[45] + uik[4]*u[46] + uik[5]*u[47] + uik[6]*u[48];
          wp[43]+=  uik[7]*u[42] + uik[8]*u[43] + uik[9]*u[44]+ uik[10]*u[45]+ uik[11]*u[46]+ uik[12]*u[47]+ uik[13]*u[48];
          wp[44]+= uik[14]*u[42]+ uik[15]*u[43]+ uik[16]*u[44]+ uik[17]*u[45]+ uik[18]*u[46]+ uik[19]*u[47]+ uik[20]*u[48];
          wp[45]+= uik[21]*u[42]+ uik[22]*u[43]+ uik[23]*u[44]+ uik[24]*u[45]+ uik[25]*u[46]+ uik[26]*u[47]+ uik[27]*u[48];
          wp[46]+= uik[28]*u[42]+ uik[29]*u[43]+ uik[30]*u[44]+ uik[31]*u[45]+ uik[32]*u[46]+ uik[33]*u[47]+ uik[34]*u[48];
          wp[47]+= uik[35]*u[42]+ uik[36]*u[43]+ uik[37]*u[44]+ uik[38]*u[45]+ uik[39]*u[46]+ uik[40]*u[47]+ uik[41]*u[48];
          wp[48]+= uik[42]*u[42]+ uik[43]*u[43]+ uik[44]*u[44]+ uik[45]*u[45]+ uik[46]*u[46]+ uik[47]*u[47]+ uik[48]*u[48];
        }
      
        /* ... add i to row list for next nonzero entry */
        il[i] = jmin;             /* update il(i) in column k+1, ... mbs-1 */
        j     = bj[jmin];
        jl[i] = jl[j]; jl[j] = i; /* update jl */
      }      
      i = nexti;      
    }

    /* save nonzero entries in k-th row of U ... */

    /* invert diagonal block */
    d = ba+k*49;
    ierr = PetscMemcpy(d,dk,49*sizeof(MatScalar));CHKERRQ(ierr);
    ierr = Kernel_A_gets_inverse_A_7(d);CHKERRQ(ierr);
    
    jmin = bi[k]; jmax = bi[k+1];
    if (jmin < jmax) {
      for (j=jmin; j<jmax; j++){
         vj = bj[j];           /* block col. index of U */
         u   = ba + j*49;
         wp = w + vj*49;        
         for (k1=0; k1<49; k1++){
           *u++        = *wp; 
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
  ierr = PetscFree(il);CHKERRQ(ierr);
  ierr = PetscFree(dk);CHKERRQ(ierr);
  if (a->permute){
    ierr = PetscFree(aa);CHKERRQ(ierr);
  }

  ierr = ISRestoreIndices(perm,&perm_ptr);CHKERRQ(ierr);
  C->factor    = FACTOR_CHOLESKY;
  C->assembled = PETSC_TRUE;
  C->preallocated = PETSC_TRUE;  
  PetscLogFlops(1.3333*343*b->mbs); /* from inverting diagonal blocks */
  PetscFunctionReturn(0);
}
