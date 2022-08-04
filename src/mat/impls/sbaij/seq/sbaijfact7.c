
#include <../src/mat/impls/sbaij/seq/sbaij.h>
#include <petsc/private/kernels/blockinvert.h>

/* Version for when blocks are 5 by 5  */
PetscErrorCode MatCholeskyFactorNumeric_SeqSBAIJ_5(Mat C,Mat A,const MatFactorInfo *info)
{
  Mat_SeqSBAIJ   *a   = (Mat_SeqSBAIJ*)A->data,*b = (Mat_SeqSBAIJ*)C->data;
  IS             perm = b->row;
  const PetscInt *ai,*aj,*perm_ptr,mbs=a->mbs,*bi=b->i,*bj=b->j;
  PetscInt       i,j,*a2anew,k,k1,jmin,jmax,*jl,*il,vj,nexti,ili,ipvt[5];
  MatScalar      *ba = b->a,*aa,*ap,*dk,*uik;
  MatScalar      *u,*d,*rtmp,*rtmp_ptr,work[25];
  PetscReal      shift = info->shiftamount;
  PetscBool      allowzeropivot,zeropivotdetected;

  PetscFunctionBegin;
  /* initialization */
  allowzeropivot = PetscNot(A->erroriffailure);
  PetscCall(PetscCalloc1(25*mbs,&rtmp));
  PetscCall(PetscMalloc2(mbs,&il,mbs,&jl));
  il[0] = 0;
  for (i=0; i<mbs; i++) jl[i] = mbs;

  PetscCall(PetscMalloc2(25,&dk,25,&uik));
  PetscCall(ISGetIndices(perm,&perm_ptr));

  /* check permutation */
  if (!a->permute) {
    ai = a->i; aj = a->j; aa = a->a;
  } else {
    ai   = a->inew; aj = a->jnew;
    PetscCall(PetscMalloc1(25*ai[mbs],&aa));
    PetscCall(PetscArraycpy(aa,a->a,25*ai[mbs]));
    PetscCall(PetscMalloc1(ai[mbs],&a2anew));
    PetscCall(PetscArraycpy(a2anew,a->a2anew,ai[mbs]));

    for (i=0; i<mbs; i++) {
      jmin = ai[i]; jmax = ai[i+1];
      for (j=jmin; j<jmax; j++) {
        while (a2anew[j] != j) {
          k = a2anew[j]; a2anew[j] = a2anew[k]; a2anew[k] = k;
          for (k1=0; k1<25; k1++) {
            dk[k1]      = aa[k*25+k1];
            aa[k*25+k1] = aa[j*25+k1];
            aa[j*25+k1] = dk[k1];
          }
        }
        /* transform columnoriented blocks that lie in the lower triangle to roworiented blocks */
        if (i > aj[j]) {
          /* printf("change orientation, row: %d, col: %d\n",i,aj[j]); */
          ap = aa + j*25;                     /* ptr to the beginning of j-th block of aa */
          for (k=0; k<25; k++) dk[k] = ap[k]; /* dk <- j-th block of aa */
          for (k=0; k<5; k++) {               /* j-th block of aa <- dk^T */
            for (k1=0; k1<5; k1++) *ap++ = dk[k + 5*k1];
          }
        }
      }
    }
    PetscCall(PetscFree(a2anew));
  }

  /* for each row k */
  for (k = 0; k<mbs; k++) {

    /*initialize k-th row with elements nonzero in row perm(k) of A */
    jmin = ai[perm_ptr[k]]; jmax = ai[perm_ptr[k]+1];
    if (jmin < jmax) {
      ap = aa + jmin*25;
      for (j = jmin; j < jmax; j++) {
        vj       = perm_ptr[aj[j]];   /* block col. index */
        rtmp_ptr = rtmp + vj*25;
        for (i=0; i<25; i++) *rtmp_ptr++ = *ap++;
      }
    }

    /* modify k-th row by adding in those rows i with U(i,k) != 0 */
    PetscCall(PetscArraycpy(dk,rtmp+k*25,25));
    i    = jl[k]; /* first row to be added to k_th row  */

    while (i < mbs) {
      nexti = jl[i]; /* next row to be added to k_th row */

      /* compute multiplier */
      ili = il[i];  /* index of first nonzero element in U(i,k:bms-1) */

      /* uik = -inv(Di)*U_bar(i,k) */
      d = ba + i*25;
      u = ba + ili*25;

      uik[0] = -(d[0]*u[0] + d[5]*u[1] + d[10]*u[2] + d[15]*u[3] + d[20]*u[4]);
      uik[1] = -(d[1]*u[0] + d[6]*u[1] + d[11]*u[2] + d[16]*u[3] + d[21]*u[4]);
      uik[2] = -(d[2]*u[0] + d[7]*u[1] + d[12]*u[2] + d[17]*u[3] + d[22]*u[4]);
      uik[3] = -(d[3]*u[0] + d[8]*u[1] + d[13]*u[2] + d[18]*u[3] + d[23]*u[4]);
      uik[4] = -(d[4]*u[0] + d[9]*u[1] + d[14]*u[2] + d[19]*u[3] + d[24]*u[4]);

      uik[5] = -(d[0]*u[5] + d[5]*u[6] + d[10]*u[7] + d[15]*u[8] + d[20]*u[9]);
      uik[6] = -(d[1]*u[5] + d[6]*u[6] + d[11]*u[7] + d[16]*u[8] + d[21]*u[9]);
      uik[7] = -(d[2]*u[5] + d[7]*u[6] + d[12]*u[7] + d[17]*u[8] + d[22]*u[9]);
      uik[8] = -(d[3]*u[5] + d[8]*u[6] + d[13]*u[7] + d[18]*u[8] + d[23]*u[9]);
      uik[9] = -(d[4]*u[5] + d[9]*u[6] + d[14]*u[7] + d[19]*u[8] + d[24]*u[9]);

      uik[10]= -(d[0]*u[10] + d[5]*u[11] + d[10]*u[12] + d[15]*u[13] + d[20]*u[14]);
      uik[11]= -(d[1]*u[10] + d[6]*u[11] + d[11]*u[12] + d[16]*u[13] + d[21]*u[14]);
      uik[12]= -(d[2]*u[10] + d[7]*u[11] + d[12]*u[12] + d[17]*u[13] + d[22]*u[14]);
      uik[13]= -(d[3]*u[10] + d[8]*u[11] + d[13]*u[12] + d[18]*u[13] + d[23]*u[14]);
      uik[14]= -(d[4]*u[10] + d[9]*u[11] + d[14]*u[12] + d[19]*u[13] + d[24]*u[14]);

      uik[15]= -(d[0]*u[15] + d[5]*u[16] + d[10]*u[17] + d[15]*u[18] + d[20]*u[19]);
      uik[16]= -(d[1]*u[15] + d[6]*u[16] + d[11]*u[17] + d[16]*u[18] + d[21]*u[19]);
      uik[17]= -(d[2]*u[15] + d[7]*u[16] + d[12]*u[17] + d[17]*u[18] + d[22]*u[19]);
      uik[18]= -(d[3]*u[15] + d[8]*u[16] + d[13]*u[17] + d[18]*u[18] + d[23]*u[19]);
      uik[19]= -(d[4]*u[15] + d[9]*u[16] + d[14]*u[17] + d[19]*u[18] + d[24]*u[19]);

      uik[20]= -(d[0]*u[20] + d[5]*u[21] + d[10]*u[22] + d[15]*u[23] + d[20]*u[24]);
      uik[21]= -(d[1]*u[20] + d[6]*u[21] + d[11]*u[22] + d[16]*u[23] + d[21]*u[24]);
      uik[22]= -(d[2]*u[20] + d[7]*u[21] + d[12]*u[22] + d[17]*u[23] + d[22]*u[24]);
      uik[23]= -(d[3]*u[20] + d[8]*u[21] + d[13]*u[22] + d[18]*u[23] + d[23]*u[24]);
      uik[24]= -(d[4]*u[20] + d[9]*u[21] + d[14]*u[22] + d[19]*u[23] + d[24]*u[24]);

      /* update D(k) += -U(i,k)^T * U_bar(i,k) */
      dk[0] +=  uik[0]*u[0] + uik[1]*u[1] + uik[2]*u[2] + uik[3]*u[3] + uik[4]*u[4];
      dk[1] +=  uik[5]*u[0] + uik[6]*u[1] + uik[7]*u[2] + uik[8]*u[3] + uik[9]*u[4];
      dk[2] += uik[10]*u[0]+ uik[11]*u[1]+ uik[12]*u[2]+ uik[13]*u[3]+ uik[14]*u[4];
      dk[3] += uik[15]*u[0]+ uik[16]*u[1]+ uik[17]*u[2]+ uik[18]*u[3]+ uik[19]*u[4];
      dk[4] += uik[20]*u[0]+ uik[21]*u[1]+ uik[22]*u[2]+ uik[23]*u[3]+ uik[24]*u[4];

      dk[5] +=  uik[0]*u[5] + uik[1]*u[6] + uik[2]*u[7] + uik[3]*u[8] + uik[4]*u[9];
      dk[6] +=  uik[5]*u[5] + uik[6]*u[6] + uik[7]*u[7] + uik[8]*u[8] + uik[9]*u[9];
      dk[7] += uik[10]*u[5]+ uik[11]*u[6]+ uik[12]*u[7]+ uik[13]*u[8]+ uik[14]*u[9];
      dk[8] += uik[15]*u[5]+ uik[16]*u[6]+ uik[17]*u[7]+ uik[18]*u[8]+ uik[19]*u[9];
      dk[9] += uik[20]*u[5]+ uik[21]*u[6]+ uik[22]*u[7]+ uik[23]*u[8]+ uik[24]*u[9];

      dk[10] +=  uik[0]*u[10] + uik[1]*u[11] + uik[2]*u[12] + uik[3]*u[13] + uik[4]*u[14];
      dk[11] +=  uik[5]*u[10] + uik[6]*u[11] + uik[7]*u[12] + uik[8]*u[13] + uik[9]*u[14];
      dk[12] += uik[10]*u[10]+ uik[11]*u[11]+ uik[12]*u[12]+ uik[13]*u[13]+ uik[14]*u[14];
      dk[13] += uik[15]*u[10]+ uik[16]*u[11]+ uik[17]*u[12]+ uik[18]*u[13]+ uik[19]*u[14];
      dk[14] += uik[20]*u[10]+ uik[21]*u[11]+ uik[22]*u[12]+ uik[23]*u[13]+ uik[24]*u[14];

      dk[15] +=  uik[0]*u[15] + uik[1]*u[16] + uik[2]*u[17] + uik[3]*u[18] + uik[4]*u[19];
      dk[16] +=  uik[5]*u[15] + uik[6]*u[16] + uik[7]*u[17] + uik[8]*u[18] + uik[9]*u[19];
      dk[17] += uik[10]*u[15]+ uik[11]*u[16]+ uik[12]*u[17]+ uik[13]*u[18]+ uik[14]*u[19];
      dk[18] += uik[15]*u[15]+ uik[16]*u[16]+ uik[17]*u[17]+ uik[18]*u[18]+ uik[19]*u[19];
      dk[19] += uik[20]*u[15]+ uik[21]*u[16]+ uik[22]*u[17]+ uik[23]*u[18]+ uik[24]*u[19];

      dk[20] +=  uik[0]*u[20] + uik[1]*u[21] + uik[2]*u[22] + uik[3]*u[23] + uik[4]*u[24];
      dk[21] +=  uik[5]*u[20] + uik[6]*u[21] + uik[7]*u[22] + uik[8]*u[23] + uik[9]*u[24];
      dk[22] += uik[10]*u[20]+ uik[11]*u[21]+ uik[12]*u[22]+ uik[13]*u[23]+ uik[14]*u[24];
      dk[23] += uik[15]*u[20]+ uik[16]*u[21]+ uik[17]*u[22]+ uik[18]*u[23]+ uik[19]*u[24];
      dk[24] += uik[20]*u[20]+ uik[21]*u[21]+ uik[22]*u[22]+ uik[23]*u[23]+ uik[24]*u[24];

      PetscCall(PetscLogFlops(125.0*4.0));

      /* update -U(i,k) */
      PetscCall(PetscArraycpy(ba+ili*25,uik,25));

      /* add multiple of row i to k-th row ... */
      jmin = ili + 1; jmax = bi[i+1];
      if (jmin < jmax) {
        for (j=jmin; j<jmax; j++) {
          /* rtmp += -U(i,k)^T * U_bar(i,j) */
          rtmp_ptr     = rtmp + bj[j]*25;
          u            = ba + j*25;
          rtmp_ptr[0] +=  uik[0]*u[0] + uik[1]*u[1] + uik[2]*u[2] + uik[3]*u[3] + uik[4]*u[4];
          rtmp_ptr[1] +=  uik[5]*u[0] + uik[6]*u[1] + uik[7]*u[2] + uik[8]*u[3] + uik[9]*u[4];
          rtmp_ptr[2] += uik[10]*u[0]+ uik[11]*u[1]+ uik[12]*u[2]+ uik[13]*u[3]+ uik[14]*u[4];
          rtmp_ptr[3] += uik[15]*u[0]+ uik[16]*u[1]+ uik[17]*u[2]+ uik[18]*u[3]+ uik[19]*u[4];
          rtmp_ptr[4] += uik[20]*u[0]+ uik[21]*u[1]+ uik[22]*u[2]+ uik[23]*u[3]+ uik[24]*u[4];

          rtmp_ptr[5] +=  uik[0]*u[5] + uik[1]*u[6] + uik[2]*u[7] + uik[3]*u[8] + uik[4]*u[9];
          rtmp_ptr[6] +=  uik[5]*u[5] + uik[6]*u[6] + uik[7]*u[7] + uik[8]*u[8] + uik[9]*u[9];
          rtmp_ptr[7] += uik[10]*u[5]+ uik[11]*u[6]+ uik[12]*u[7]+ uik[13]*u[8]+ uik[14]*u[9];
          rtmp_ptr[8] += uik[15]*u[5]+ uik[16]*u[6]+ uik[17]*u[7]+ uik[18]*u[8]+ uik[19]*u[9];
          rtmp_ptr[9] += uik[20]*u[5]+ uik[21]*u[6]+ uik[22]*u[7]+ uik[23]*u[8]+ uik[24]*u[9];

          rtmp_ptr[10] +=  uik[0]*u[10] + uik[1]*u[11] + uik[2]*u[12] + uik[3]*u[13] + uik[4]*u[14];
          rtmp_ptr[11] +=  uik[5]*u[10] + uik[6]*u[11] + uik[7]*u[12] + uik[8]*u[13] + uik[9]*u[14];
          rtmp_ptr[12] += uik[10]*u[10]+ uik[11]*u[11]+ uik[12]*u[12]+ uik[13]*u[13]+ uik[14]*u[14];
          rtmp_ptr[13] += uik[15]*u[10]+ uik[16]*u[11]+ uik[17]*u[12]+ uik[18]*u[13]+ uik[19]*u[14];
          rtmp_ptr[14] += uik[20]*u[10]+ uik[21]*u[11]+ uik[22]*u[12]+ uik[23]*u[13]+ uik[24]*u[14];

          rtmp_ptr[15] +=  uik[0]*u[15] + uik[1]*u[16] + uik[2]*u[17] + uik[3]*u[18] + uik[4]*u[19];
          rtmp_ptr[16] +=  uik[5]*u[15] + uik[6]*u[16] + uik[7]*u[17] + uik[8]*u[18] + uik[9]*u[19];
          rtmp_ptr[17] += uik[10]*u[15]+ uik[11]*u[16]+ uik[12]*u[17]+ uik[13]*u[18]+ uik[14]*u[19];
          rtmp_ptr[18] += uik[15]*u[15]+ uik[16]*u[16]+ uik[17]*u[17]+ uik[18]*u[18]+ uik[19]*u[19];
          rtmp_ptr[19] += uik[20]*u[15]+ uik[21]*u[16]+ uik[22]*u[17]+ uik[23]*u[18]+ uik[24]*u[19];

          rtmp_ptr[20] +=  uik[0]*u[20] + uik[1]*u[21] + uik[2]*u[22] + uik[3]*u[23] + uik[4]*u[24];
          rtmp_ptr[21] +=  uik[5]*u[20] + uik[6]*u[21] + uik[7]*u[22] + uik[8]*u[23] + uik[9]*u[24];
          rtmp_ptr[22] += uik[10]*u[20]+ uik[11]*u[21]+ uik[12]*u[22]+ uik[13]*u[23]+ uik[14]*u[24];
          rtmp_ptr[23] += uik[15]*u[20]+ uik[16]*u[21]+ uik[17]*u[22]+ uik[18]*u[23]+ uik[19]*u[24];
          rtmp_ptr[24] += uik[20]*u[20]+ uik[21]*u[21]+ uik[22]*u[22]+ uik[23]*u[23]+ uik[24]*u[24];
        }
        PetscCall(PetscLogFlops(2.0*125.0*(jmax-jmin)));

        /* ... add i to row list for next nonzero entry */
        il[i] = jmin;             /* update il(i) in column k+1, ... mbs-1 */
        j     = bj[jmin];
        jl[i] = jl[j]; jl[j] = i; /* update jl */
      }
      i = nexti;
    }

    /* save nonzero entries in k-th row of U ... */

    /* invert diagonal block */
    d    = ba+k*25;
    PetscCall(PetscArraycpy(d,dk,25));
    PetscCall(PetscKernel_A_gets_inverse_A_5(d,ipvt,work,shift,allowzeropivot,&zeropivotdetected));
    if (zeropivotdetected) C->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;

    jmin = bi[k]; jmax = bi[k+1];
    if (jmin < jmax) {
      for (j=jmin; j<jmax; j++) {
        vj       = bj[j];      /* block col. index of U */
        u        = ba + j*25;
        rtmp_ptr = rtmp + vj*25;
        for (k1=0; k1<25; k1++) {
          *u++        = *rtmp_ptr;
          *rtmp_ptr++ = 0.0;
        }
      }

      /* ... add k to row list for first nonzero entry in k-th row */
      il[k] = jmin;
      i     = bj[jmin];
      jl[k] = jl[i]; jl[i] = k;
    }
  }

  PetscCall(PetscFree(rtmp));
  PetscCall(PetscFree2(il,jl));
  PetscCall(PetscFree2(dk,uik));
  if (a->permute) PetscCall(PetscFree(aa));

  PetscCall(ISRestoreIndices(perm,&perm_ptr));

  C->ops->solve          = MatSolve_SeqSBAIJ_5_inplace;
  C->ops->solvetranspose = MatSolve_SeqSBAIJ_5_inplace;
  C->assembled           = PETSC_TRUE;
  C->preallocated        = PETSC_TRUE;

  PetscCall(PetscLogFlops(1.3333*125*b->mbs)); /* from inverting diagonal blocks */
  PetscFunctionReturn(0);
}
