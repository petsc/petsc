#include "sbaij.h"
#include "src/inline/ilu.h"

/* Version for when blocks are 6 by 6 */
#undef __FUNC__  
#define __FUNC__ "MatCholeskyFactorNumeric_SeqSBAIJ_6"
int MatCholeskyFactorNumeric_SeqSBAIJ_6(Mat A,Mat *B)
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
  ierr = PetscMalloc(36*mbs*sizeof(MatScalar),&w);CHKERRQ(ierr);
  ierr = PetscMemzero(w,36*mbs*sizeof(MatScalar));CHKERRQ(ierr); 
  ierr = PetscMalloc(2*mbs*sizeof(int),&il);CHKERRQ(ierr);
  jl = il + mbs;
  for (i=0; i<mbs; i++) {
    jl[i] = mbs; il[0] = 0;
  }
  ierr = PetscMalloc(72*sizeof(MatScalar),&dk);CHKERRQ(ierr);
  uik  = dk + 36;     
  ierr = ISGetIndices(perm,&perm_ptr);CHKERRQ(ierr);

  /* check permutation */
  if (!a->permute){
    ai = a->i; aj = a->j; aa = a->a;
  } else {
    ai   = a->inew; aj = a->jnew; 
    ierr = PetscMalloc(36*ai[mbs]*sizeof(MatScalar),&aa);CHKERRQ(ierr); 
    ierr = PetscMemcpy(aa,a->a,36*ai[mbs]*sizeof(MatScalar));CHKERRQ(ierr);
    ierr = PetscMalloc(ai[mbs]*sizeof(int),&a2anew);CHKERRQ(ierr); 
    ierr = PetscMemcpy(a2anew,a->a2anew,(ai[mbs])*sizeof(int));CHKERRQ(ierr);

    for (i=0; i<mbs; i++){
      jmin = ai[i]; jmax = ai[i+1];
      for (j=jmin; j<jmax; j++){
        while (a2anew[j] != j){  
          k = a2anew[j]; a2anew[j] = a2anew[k]; a2anew[k] = k;  
          for (k1=0; k1<36; k1++){
            dk[k1]       = aa[k*36+k1]; 
            aa[k*36+k1] = aa[j*36+k1]; 
            aa[j*36+k1] = dk[k1];   
          }
        }
        /* transform columnoriented blocks that lie in the lower triangle to roworiented blocks */
        if (i > aj[j]){ 
          /* printf("change orientation, row: %d, col: %d\n",i,aj[j]); */
          ap = aa + j*36;                     /* ptr to the beginning of j-th block of aa */
          for (k=0; k<36; k++) dk[k] = ap[k]; /* dk <- j-th block of aa */
          for (k=0; k<6; k++){               /* j-th block of aa <- dk^T */
            for (k1=0; k1<6; k1++) *ap++ = dk[k + 6*k1];         
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
      ap = aa + jmin*36;
      for (j = jmin; j < jmax; j++){
        vj = perm_ptr[aj[j]];         /* block col. index */  
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
      d = ba + i*36;
      u    = ba + ili*36;

      uik[0] = -(d[0]*u[0] + d[6]*u[1] + d[12]*u[2] + d[18]*u[3] + d[24]*u[4] + d[30]*u[5]);
      uik[1] = -(d[1]*u[0] + d[7]*u[1] + d[13]*u[2] + d[19]*u[3] + d[25]*u[4] + d[31]*u[5]);
      uik[2] = -(d[2]*u[0] + d[8]*u[1] + d[14]*u[2] + d[20]*u[3] + d[26]*u[4] + d[32]*u[5]);
      uik[3] = -(d[3]*u[0] + d[9]*u[1] + d[15]*u[2] + d[21]*u[3] + d[27]*u[4] + d[33]*u[5]);
      uik[4] = -(d[4]*u[0]+ d[10]*u[1] + d[16]*u[2] + d[22]*u[3] + d[28]*u[4] + d[34]*u[5]);
      uik[5] = -(d[5]*u[0]+ d[11]*u[1] + d[17]*u[2] + d[23]*u[3] + d[29]*u[4] + d[35]*u[5]);

      uik[6] = -(d[0]*u[6] + d[6]*u[7] + d[12]*u[8] + d[18]*u[9] + d[24]*u[10] + d[30]*u[11]);
      uik[7] = -(d[1]*u[6] + d[7]*u[7] + d[13]*u[8] + d[19]*u[9] + d[25]*u[10] + d[31]*u[11]);
      uik[8] = -(d[2]*u[6] + d[8]*u[7] + d[14]*u[8] + d[20]*u[9] + d[26]*u[10] + d[32]*u[11]);
      uik[9] = -(d[3]*u[6] + d[9]*u[7] + d[15]*u[8] + d[21]*u[9] + d[27]*u[10] + d[33]*u[11]);
      uik[10]= -(d[4]*u[6]+ d[10]*u[7] + d[16]*u[8] + d[22]*u[9] + d[28]*u[10] + d[34]*u[11]);
      uik[11]= -(d[5]*u[6]+ d[11]*u[7] + d[17]*u[8] + d[23]*u[9] + d[29]*u[10] + d[35]*u[11]);

      uik[12] = -(d[0]*u[12] + d[6]*u[13] + d[12]*u[14] + d[18]*u[15] + d[24]*u[16] + d[30]*u[17]);
      uik[13] = -(d[1]*u[12] + d[7]*u[13] + d[13]*u[14] + d[19]*u[15] + d[25]*u[16] + d[31]*u[17]);
      uik[14] = -(d[2]*u[12] + d[8]*u[13] + d[14]*u[14] + d[20]*u[15] + d[26]*u[16] + d[32]*u[17]);
      uik[15] = -(d[3]*u[12] + d[9]*u[13] + d[15]*u[14] + d[21]*u[15] + d[27]*u[16] + d[33]*u[17]);
      uik[16] = -(d[4]*u[12]+ d[10]*u[13] + d[16]*u[14] + d[22]*u[15] + d[28]*u[16] + d[34]*u[17]);
      uik[17] = -(d[5]*u[12]+ d[11]*u[13] + d[17]*u[14] + d[23]*u[15] + d[29]*u[16] + d[35]*u[17]);

      uik[18] = -(d[0]*u[18] + d[6]*u[19] + d[12]*u[20] + d[18]*u[21] + d[24]*u[22] + d[30]*u[23]);
      uik[19] = -(d[1]*u[18] + d[7]*u[19] + d[13]*u[20] + d[19]*u[21] + d[25]*u[22] + d[31]*u[23]);
      uik[20] = -(d[2]*u[18] + d[8]*u[19] + d[14]*u[20] + d[20]*u[21] + d[26]*u[22] + d[32]*u[23]);
      uik[21] = -(d[3]*u[18] + d[9]*u[19] + d[15]*u[20] + d[21]*u[21] + d[27]*u[22] + d[33]*u[23]);
      uik[22] = -(d[4]*u[18]+ d[10]*u[19] + d[16]*u[20] + d[22]*u[21] + d[28]*u[22] + d[34]*u[23]);
      uik[23] = -(d[5]*u[18]+ d[11]*u[19] + d[17]*u[20] + d[23]*u[21] + d[29]*u[22] + d[35]*u[23]);

      uik[24] = -(d[0]*u[24] + d[6]*u[25] + d[12]*u[26] + d[18]*u[27] + d[24]*u[28] + d[30]*u[29]);
      uik[25] = -(d[1]*u[24] + d[7]*u[25] + d[13]*u[26] + d[19]*u[27] + d[25]*u[28] + d[31]*u[29]);
      uik[26] = -(d[2]*u[24] + d[8]*u[25] + d[14]*u[26] + d[20]*u[27] + d[26]*u[28] + d[32]*u[29]);
      uik[27] = -(d[3]*u[24] + d[9]*u[25] + d[15]*u[26] + d[21]*u[27] + d[27]*u[28] + d[33]*u[29]);
      uik[28] = -(d[4]*u[24]+ d[10]*u[25] + d[16]*u[26] + d[22]*u[27] + d[28]*u[28] + d[34]*u[29]);
      uik[29] = -(d[5]*u[24]+ d[11]*u[25] + d[17]*u[26] + d[23]*u[27] + d[29]*u[28] + d[35]*u[29]);

      uik[30] = -(d[0]*u[30] + d[6]*u[31] + d[12]*u[32] + d[18]*u[33] + d[24]*u[34] + d[30]*u[35]);
      uik[31] = -(d[1]*u[30] + d[7]*u[31] + d[13]*u[32] + d[19]*u[33] + d[25]*u[34] + d[31]*u[35]);
      uik[32] = -(d[2]*u[30] + d[8]*u[31] + d[14]*u[32] + d[20]*u[33] + d[26]*u[34] + d[32]*u[35]);
      uik[33] = -(d[3]*u[30] + d[9]*u[31] + d[15]*u[32] + d[21]*u[33] + d[27]*u[34] + d[33]*u[35]);
      uik[34] = -(d[4]*u[30]+ d[10]*u[31] + d[16]*u[32] + d[22]*u[33] + d[28]*u[34] + d[34]*u[35]);
      uik[35] = -(d[5]*u[30]+ d[11]*u[31] + d[17]*u[32] + d[23]*u[33] + d[29]*u[34] + d[35]*u[35]);

      /* update D(k) += -U(i,k)^T * U_bar(i,k) */  
      dk[0] +=  uik[0]*u[0] + uik[1]*u[1] + uik[2]*u[2] + uik[3]*u[3] + uik[4]*u[4] + uik[5]*u[5];
      dk[1] +=  uik[6]*u[0] + uik[7]*u[1] + uik[8]*u[2] + uik[9]*u[3]+ uik[10]*u[4]+ uik[11]*u[5];
      dk[2] += uik[12]*u[0]+ uik[13]*u[1]+ uik[14]*u[2]+ uik[15]*u[3]+ uik[16]*u[4]+ uik[17]*u[5];
      dk[3] += uik[18]*u[0]+ uik[19]*u[1]+ uik[20]*u[2]+ uik[21]*u[3]+ uik[22]*u[4]+ uik[23]*u[5];
      dk[4] += uik[24]*u[0]+ uik[25]*u[1]+ uik[26]*u[2]+ uik[27]*u[3]+ uik[28]*u[4]+ uik[29]*u[5];
      dk[5] += uik[30]*u[0]+ uik[31]*u[1]+ uik[32]*u[2]+ uik[33]*u[3]+ uik[34]*u[4]+ uik[35]*u[5];

      dk[6] +=  uik[0]*u[6] + uik[1]*u[7] + uik[2]*u[8] + uik[3]*u[9] + uik[4]*u[10] + uik[5]*u[11];
      dk[7] +=  uik[6]*u[6] + uik[7]*u[7] + uik[8]*u[8] + uik[9]*u[9]+ uik[10]*u[10]+ uik[11]*u[11];
      dk[8] += uik[12]*u[6]+ uik[13]*u[7]+ uik[14]*u[8]+ uik[15]*u[9]+ uik[16]*u[10]+ uik[17]*u[11];
      dk[9] += uik[18]*u[6]+ uik[19]*u[7]+ uik[20]*u[8]+ uik[21]*u[9]+ uik[22]*u[10]+ uik[23]*u[11];
      dk[10]+= uik[24]*u[6]+ uik[25]*u[7]+ uik[26]*u[8]+ uik[27]*u[9]+ uik[28]*u[10]+ uik[29]*u[11];
      dk[11]+= uik[30]*u[6]+ uik[31]*u[7]+ uik[32]*u[8]+ uik[33]*u[9]+ uik[34]*u[10]+ uik[35]*u[11];

      dk[12]+=  uik[0]*u[12] + uik[1]*u[13] + uik[2]*u[14] + uik[3]*u[15] + uik[4]*u[16] + uik[5]*u[17];
      dk[13]+=  uik[6]*u[12] + uik[7]*u[13] + uik[8]*u[14] + uik[9]*u[15]+ uik[10]*u[16]+ uik[11]*u[17];
      dk[14]+= uik[12]*u[12]+ uik[13]*u[13]+ uik[14]*u[14]+ uik[15]*u[15]+ uik[16]*u[16]+ uik[17]*u[17];
      dk[15]+= uik[18]*u[12]+ uik[19]*u[13]+ uik[20]*u[14]+ uik[21]*u[15]+ uik[22]*u[16]+ uik[23]*u[17];
      dk[16]+= uik[24]*u[12]+ uik[25]*u[13]+ uik[26]*u[14]+ uik[27]*u[15]+ uik[28]*u[16]+ uik[29]*u[17];
      dk[17]+= uik[30]*u[12]+ uik[31]*u[13]+ uik[32]*u[14]+ uik[33]*u[15]+ uik[34]*u[16]+ uik[35]*u[17];

      dk[18]+=  uik[0]*u[18] + uik[1]*u[19] + uik[2]*u[20] + uik[3]*u[21] + uik[4]*u[22] + uik[5]*u[23];
      dk[19]+=  uik[6]*u[18] + uik[7]*u[19] + uik[8]*u[20] + uik[9]*u[21]+ uik[10]*u[22]+ uik[11]*u[23];
      dk[20]+= uik[12]*u[18]+ uik[13]*u[19]+ uik[14]*u[20]+ uik[15]*u[21]+ uik[16]*u[22]+ uik[17]*u[23];
      dk[21]+= uik[18]*u[18]+ uik[19]*u[19]+ uik[20]*u[20]+ uik[21]*u[21]+ uik[22]*u[22]+ uik[23]*u[23];
      dk[22]+= uik[24]*u[18]+ uik[25]*u[19]+ uik[26]*u[20]+ uik[27]*u[21]+ uik[28]*u[22]+ uik[29]*u[23];
      dk[23]+= uik[30]*u[18]+ uik[31]*u[19]+ uik[32]*u[20]+ uik[33]*u[21]+ uik[34]*u[22]+ uik[35]*u[23];

      dk[24]+=  uik[0]*u[24] + uik[1]*u[25] + uik[2]*u[26] + uik[3]*u[27] + uik[4]*u[28] + uik[5]*u[29];
      dk[25]+=  uik[6]*u[24] + uik[7]*u[25] + uik[8]*u[26] + uik[9]*u[27]+ uik[10]*u[28]+ uik[11]*u[29];
      dk[26]+= uik[12]*u[24]+ uik[13]*u[25]+ uik[14]*u[26]+ uik[15]*u[27]+ uik[16]*u[28]+ uik[17]*u[29];
      dk[27]+= uik[18]*u[24]+ uik[19]*u[25]+ uik[20]*u[26]+ uik[21]*u[27]+ uik[22]*u[28]+ uik[23]*u[29];
      dk[28]+= uik[24]*u[24]+ uik[25]*u[25]+ uik[26]*u[26]+ uik[27]*u[27]+ uik[28]*u[28]+ uik[29]*u[29];
      dk[29]+= uik[30]*u[24]+ uik[31]*u[25]+ uik[32]*u[26]+ uik[33]*u[27]+ uik[34]*u[28]+ uik[35]*u[29];

      dk[30]+=  uik[0]*u[30] + uik[1]*u[31] + uik[2]*u[32] + uik[3]*u[33] + uik[4]*u[34] + uik[5]*u[35];
      dk[31]+=  uik[6]*u[30] + uik[7]*u[31] + uik[8]*u[32] + uik[9]*u[33]+ uik[10]*u[34]+ uik[11]*u[35];
      dk[32]+= uik[12]*u[30]+ uik[13]*u[31]+ uik[14]*u[32]+ uik[15]*u[33]+ uik[16]*u[34]+ uik[17]*u[35];
      dk[33]+= uik[18]*u[30]+ uik[19]*u[31]+ uik[20]*u[32]+ uik[21]*u[33]+ uik[22]*u[34]+ uik[23]*u[35];
      dk[34]+= uik[24]*u[30]+ uik[25]*u[31]+ uik[26]*u[32]+ uik[27]*u[33]+ uik[28]*u[34]+ uik[29]*u[35];
      dk[35]+= uik[30]*u[30]+ uik[31]*u[31]+ uik[32]*u[32]+ uik[33]*u[33]+ uik[34]*u[34]+ uik[35]*u[35];
 
      /* update -U(i,k) */
      ierr = PetscMemcpy(ba+ili*36,uik,36*sizeof(MatScalar));CHKERRQ(ierr); 

      /* add multiple of row i to k-th row ... */
      jmin = ili + 1; jmax = bi[i+1];
      if (jmin < jmax){
        for (j=jmin; j<jmax; j++) {
          /* w += -U(i,k)^T * U_bar(i,j) */
          wp = w + bj[j]*36;
          u = ba + j*36;
          wp[0] +=  uik[0]*u[0] + uik[1]*u[1] + uik[2]*u[2] + uik[3]*u[3] + uik[4]*u[4] + uik[5]*u[5];
          wp[1] +=  uik[6]*u[0] + uik[7]*u[1] + uik[8]*u[2] + uik[9]*u[3]+ uik[10]*u[4]+ uik[11]*u[5];
          wp[2] += uik[12]*u[0]+ uik[13]*u[1]+ uik[14]*u[2]+ uik[15]*u[3]+ uik[16]*u[4]+ uik[17]*u[5];
          wp[3] += uik[18]*u[0]+ uik[19]*u[1]+ uik[20]*u[2]+ uik[21]*u[3]+ uik[22]*u[4]+ uik[23]*u[5];
          wp[4] += uik[24]*u[0]+ uik[25]*u[1]+ uik[26]*u[2]+ uik[27]*u[3]+ uik[28]*u[4]+ uik[29]*u[5];
          wp[5] += uik[30]*u[0]+ uik[31]*u[1]+ uik[32]*u[2]+ uik[33]*u[3]+ uik[34]*u[4]+ uik[35]*u[5];

          wp[6] +=  uik[0]*u[6] + uik[1]*u[7] + uik[2]*u[8] + uik[3]*u[9] + uik[4]*u[10] + uik[5]*u[11];
          wp[7] +=  uik[6]*u[6] + uik[7]*u[7] + uik[8]*u[8] + uik[9]*u[9]+ uik[10]*u[10]+ uik[11]*u[11];
          wp[8] += uik[12]*u[6]+ uik[13]*u[7]+ uik[14]*u[8]+ uik[15]*u[9]+ uik[16]*u[10]+ uik[17]*u[11];
          wp[9] += uik[18]*u[6]+ uik[19]*u[7]+ uik[20]*u[8]+ uik[21]*u[9]+ uik[22]*u[10]+ uik[23]*u[11];
          wp[10]+= uik[24]*u[6]+ uik[25]*u[7]+ uik[26]*u[8]+ uik[27]*u[9]+ uik[28]*u[10]+ uik[29]*u[11];
          wp[11]+= uik[30]*u[6]+ uik[31]*u[7]+ uik[32]*u[8]+ uik[33]*u[9]+ uik[34]*u[10]+ uik[35]*u[11];

          wp[12]+=  uik[0]*u[12] + uik[1]*u[13] + uik[2]*u[14] + uik[3]*u[15] + uik[4]*u[16] + uik[5]*u[17];
          wp[13]+=  uik[6]*u[12] + uik[7]*u[13] + uik[8]*u[14] + uik[9]*u[15]+ uik[10]*u[16]+ uik[11]*u[17];
          wp[14]+= uik[12]*u[12]+ uik[13]*u[13]+ uik[14]*u[14]+ uik[15]*u[15]+ uik[16]*u[16]+ uik[17]*u[17];
          wp[15]+= uik[18]*u[12]+ uik[19]*u[13]+ uik[20]*u[14]+ uik[21]*u[15]+ uik[22]*u[16]+ uik[23]*u[17];
          wp[16]+= uik[24]*u[12]+ uik[25]*u[13]+ uik[26]*u[14]+ uik[27]*u[15]+ uik[28]*u[16]+ uik[29]*u[17];
          wp[17]+= uik[30]*u[12]+ uik[31]*u[13]+ uik[32]*u[14]+ uik[33]*u[15]+ uik[34]*u[16]+ uik[35]*u[17];

          wp[18]+=  uik[0]*u[18] + uik[1]*u[19] + uik[2]*u[20] + uik[3]*u[21] + uik[4]*u[22] + uik[5]*u[23];
          wp[19]+=  uik[6]*u[18] + uik[7]*u[19] + uik[8]*u[20] + uik[9]*u[21]+ uik[10]*u[22]+ uik[11]*u[23];
          wp[20]+= uik[12]*u[18]+ uik[13]*u[19]+ uik[14]*u[20]+ uik[15]*u[21]+ uik[16]*u[22]+ uik[17]*u[23];
          wp[21]+= uik[18]*u[18]+ uik[19]*u[19]+ uik[20]*u[20]+ uik[21]*u[21]+ uik[22]*u[22]+ uik[23]*u[23];
          wp[22]+= uik[24]*u[18]+ uik[25]*u[19]+ uik[26]*u[20]+ uik[27]*u[21]+ uik[28]*u[22]+ uik[29]*u[23];
          wp[23]+= uik[30]*u[18]+ uik[31]*u[19]+ uik[32]*u[20]+ uik[33]*u[21]+ uik[34]*u[22]+ uik[35]*u[23];

          wp[24]+=  uik[0]*u[24] + uik[1]*u[25] + uik[2]*u[26] + uik[3]*u[27] + uik[4]*u[28] + uik[5]*u[29];
          wp[25]+=  uik[6]*u[24] + uik[7]*u[25] + uik[8]*u[26] + uik[9]*u[27]+ uik[10]*u[28]+ uik[11]*u[29];
          wp[26]+= uik[12]*u[24]+ uik[13]*u[25]+ uik[14]*u[26]+ uik[15]*u[27]+ uik[16]*u[28]+ uik[17]*u[29];
          wp[27]+= uik[18]*u[24]+ uik[19]*u[25]+ uik[20]*u[26]+ uik[21]*u[27]+ uik[22]*u[28]+ uik[23]*u[29];
          wp[28]+= uik[24]*u[24]+ uik[25]*u[25]+ uik[26]*u[26]+ uik[27]*u[27]+ uik[28]*u[28]+ uik[29]*u[29];
          wp[29]+= uik[30]*u[24]+ uik[31]*u[25]+ uik[32]*u[26]+ uik[33]*u[27]+ uik[34]*u[28]+ uik[35]*u[29];

          wp[30]+=  uik[0]*u[30] + uik[1]*u[31] + uik[2]*u[32] + uik[3]*u[33] + uik[4]*u[34] + uik[5]*u[35];
          wp[31]+=  uik[6]*u[30] + uik[7]*u[31] + uik[8]*u[32] + uik[9]*u[33]+ uik[10]*u[34]+ uik[11]*u[35];
          wp[32]+= uik[12]*u[30]+ uik[13]*u[31]+ uik[14]*u[32]+ uik[15]*u[33]+ uik[16]*u[34]+ uik[17]*u[35];
          wp[33]+= uik[18]*u[30]+ uik[19]*u[31]+ uik[20]*u[32]+ uik[21]*u[33]+ uik[22]*u[34]+ uik[23]*u[35];
          wp[34]+= uik[24]*u[30]+ uik[25]*u[31]+ uik[26]*u[32]+ uik[27]*u[33]+ uik[28]*u[34]+ uik[29]*u[35];
          wp[35]+= uik[30]*u[30]+ uik[31]*u[31]+ uik[32]*u[32]+ uik[33]*u[33]+ uik[34]*u[34]+ uik[35]*u[35];
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
    d = ba+k*36;
    ierr = PetscMemcpy(d,dk,36*sizeof(MatScalar));CHKERRQ(ierr);
    ierr = Kernel_A_gets_inverse_A_6(d);CHKERRQ(ierr);
    
    jmin = bi[k]; jmax = bi[k+1];
    if (jmin < jmax) {
      for (j=jmin; j<jmax; j++){
         vj = bj[j];           /* block col. index of U */
         u   = ba + j*36;
         wp = w + vj*36;        
         for (k1=0; k1<36; k1++){
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
  PetscLogFlops(1.3333*216*b->mbs); /* from inverting diagonal blocks */
  PetscFunctionReturn(0);
}
