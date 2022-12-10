
#include <../src/mat/impls/sbaij/seq/sbaij.h>
#include <petsc/private/kernels/blockinvert.h>

/* Version for when blocks are 6 by 6 */
PetscErrorCode MatCholeskyFactorNumeric_SeqSBAIJ_6(Mat C, Mat A, const MatFactorInfo *info)
{
  Mat_SeqSBAIJ   *a = (Mat_SeqSBAIJ *)A->data, *b = (Mat_SeqSBAIJ *)C->data;
  IS              perm = b->row;
  const PetscInt *ai, *aj, *perm_ptr, mbs = a->mbs, *bi = b->i, *bj = b->j;
  PetscInt        i, j, *a2anew, k, k1, jmin, jmax, *jl, *il, vj, nexti, ili;
  MatScalar      *ba = b->a, *aa, *ap, *dk, *uik;
  MatScalar      *u, *d, *w, *wp, u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12;
  MatScalar       u13, u14, u15, u16, u17, u18, u19, u20, u21, u22, u23, u24, u25, u26, u27;
  MatScalar       u28, u29, u30, u31, u32, u33, u34, u35;
  PetscReal       shift = info->shiftamount;
  PetscBool       allowzeropivot, zeropivotdetected;

  PetscFunctionBegin;
  /* initialization */
  allowzeropivot = PetscNot(A->erroriffailure);
  PetscCall(PetscCalloc1(36 * mbs, &w));
  PetscCall(PetscMalloc2(mbs, &il, mbs, &jl));
  il[0] = 0;
  for (i = 0; i < mbs; i++) jl[i] = mbs;

  PetscCall(PetscMalloc2(36, &dk, 36, &uik));
  PetscCall(ISGetIndices(perm, &perm_ptr));

  /* check permutation */
  if (!a->permute) {
    ai = a->i;
    aj = a->j;
    aa = a->a;
  } else {
    ai = a->inew;
    aj = a->jnew;
    PetscCall(PetscMalloc1(36 * ai[mbs], &aa));
    PetscCall(PetscArraycpy(aa, a->a, 36 * ai[mbs]));
    PetscCall(PetscMalloc1(ai[mbs], &a2anew));
    PetscCall(PetscArraycpy(a2anew, a->a2anew, ai[mbs]));

    for (i = 0; i < mbs; i++) {
      jmin = ai[i];
      jmax = ai[i + 1];
      for (j = jmin; j < jmax; j++) {
        while (a2anew[j] != j) {
          k         = a2anew[j];
          a2anew[j] = a2anew[k];
          a2anew[k] = k;
          for (k1 = 0; k1 < 36; k1++) {
            dk[k1]          = aa[k * 36 + k1];
            aa[k * 36 + k1] = aa[j * 36 + k1];
            aa[j * 36 + k1] = dk[k1];
          }
        }
        /* transform columnoriented blocks that lie in the lower triangle to roworiented blocks */
        if (i > aj[j]) {
          /* printf("change orientation, row: %d, col: %d\n",i,aj[j]); */
          ap = aa + j * 36;                       /* ptr to the beginning of j-th block of aa */
          for (k = 0; k < 36; k++) dk[k] = ap[k]; /* dk <- j-th block of aa */
          for (k = 0; k < 6; k++) {               /* j-th block of aa <- dk^T */
            for (k1 = 0; k1 < 6; k1++) *ap++ = dk[k + 6 * k1];
          }
        }
      }
    }
    PetscCall(PetscFree(a2anew));
  }

  /* for each row k */
  for (k = 0; k < mbs; k++) {
    /*initialize k-th row with elements nonzero in row perm(k) of A */
    jmin = ai[perm_ptr[k]];
    jmax = ai[perm_ptr[k] + 1];
    if (jmin < jmax) {
      ap = aa + jmin * 36;
      for (j = jmin; j < jmax; j++) {
        vj = perm_ptr[aj[j]]; /* block col. index */
        wp = w + vj * 36;
        for (i = 0; i < 36; i++) *wp++ = *ap++;
      }
    }

    /* modify k-th row by adding in those rows i with U(i,k) != 0 */
    PetscCall(PetscArraycpy(dk, w + k * 36, 36));
    i = jl[k]; /* first row to be added to k_th row  */

    while (i < mbs) {
      nexti = jl[i]; /* next row to be added to k_th row */

      /* compute multiplier */
      ili = il[i]; /* index of first nonzero element in U(i,k:bms-1) */

      /* uik = -inv(Di)*U_bar(i,k) */
      d = ba + i * 36;
      u = ba + ili * 36;

      u0  = u[0];
      u1  = u[1];
      u2  = u[2];
      u3  = u[3];
      u4  = u[4];
      u5  = u[5];
      u6  = u[6];
      u7  = u[7];
      u8  = u[8];
      u9  = u[9];
      u10 = u[10];
      u11 = u[11];
      u12 = u[12];
      u13 = u[13];
      u14 = u[14];
      u15 = u[15];
      u16 = u[16];
      u17 = u[17];
      u18 = u[18];
      u19 = u[19];
      u20 = u[20];
      u21 = u[21];
      u22 = u[22];
      u23 = u[23];
      u24 = u[24];
      u25 = u[25];
      u26 = u[26];
      u27 = u[27];
      u28 = u[28];
      u29 = u[29];
      u30 = u[30];
      u31 = u[31];
      u32 = u[32];
      u33 = u[33];
      u34 = u[34];
      u35 = u[35];

      uik[0] = -(d[0] * u0 + d[6] * u1 + d[12] * u2 + d[18] * u3 + d[24] * u4 + d[30] * u5);
      uik[1] = -(d[1] * u0 + d[7] * u1 + d[13] * u2 + d[19] * u3 + d[25] * u4 + d[31] * u5);
      uik[2] = -(d[2] * u0 + d[8] * u1 + d[14] * u2 + d[20] * u3 + d[26] * u4 + d[32] * u5);
      uik[3] = -(d[3] * u0 + d[9] * u1 + d[15] * u2 + d[21] * u3 + d[27] * u4 + d[33] * u5);
      uik[4] = -(d[4] * u0 + d[10] * u1 + d[16] * u2 + d[22] * u3 + d[28] * u4 + d[34] * u5);
      uik[5] = -(d[5] * u0 + d[11] * u1 + d[17] * u2 + d[23] * u3 + d[29] * u4 + d[35] * u5);

      uik[6]  = -(d[0] * u6 + d[6] * u7 + d[12] * u8 + d[18] * u9 + d[24] * u10 + d[30] * u11);
      uik[7]  = -(d[1] * u6 + d[7] * u7 + d[13] * u8 + d[19] * u9 + d[25] * u10 + d[31] * u11);
      uik[8]  = -(d[2] * u6 + d[8] * u7 + d[14] * u8 + d[20] * u9 + d[26] * u10 + d[32] * u11);
      uik[9]  = -(d[3] * u6 + d[9] * u7 + d[15] * u8 + d[21] * u9 + d[27] * u10 + d[33] * u11);
      uik[10] = -(d[4] * u6 + d[10] * u7 + d[16] * u8 + d[22] * u9 + d[28] * u10 + d[34] * u11);
      uik[11] = -(d[5] * u6 + d[11] * u7 + d[17] * u8 + d[23] * u9 + d[29] * u10 + d[35] * u11);

      uik[12] = -(d[0] * u12 + d[6] * u13 + d[12] * u14 + d[18] * u15 + d[24] * u16 + d[30] * u17);
      uik[13] = -(d[1] * u12 + d[7] * u13 + d[13] * u14 + d[19] * u15 + d[25] * u16 + d[31] * u17);
      uik[14] = -(d[2] * u12 + d[8] * u13 + d[14] * u14 + d[20] * u15 + d[26] * u16 + d[32] * u17);
      uik[15] = -(d[3] * u12 + d[9] * u13 + d[15] * u14 + d[21] * u15 + d[27] * u16 + d[33] * u17);
      uik[16] = -(d[4] * u12 + d[10] * u13 + d[16] * u14 + d[22] * u15 + d[28] * u16 + d[34] * u17);
      uik[17] = -(d[5] * u12 + d[11] * u13 + d[17] * u14 + d[23] * u15 + d[29] * u16 + d[35] * u17);

      uik[18] = -(d[0] * u18 + d[6] * u19 + d[12] * u20 + d[18] * u21 + d[24] * u22 + d[30] * u23);
      uik[19] = -(d[1] * u18 + d[7] * u19 + d[13] * u20 + d[19] * u21 + d[25] * u22 + d[31] * u23);
      uik[20] = -(d[2] * u18 + d[8] * u19 + d[14] * u20 + d[20] * u21 + d[26] * u22 + d[32] * u23);
      uik[21] = -(d[3] * u18 + d[9] * u19 + d[15] * u20 + d[21] * u21 + d[27] * u22 + d[33] * u23);
      uik[22] = -(d[4] * u18 + d[10] * u19 + d[16] * u20 + d[22] * u21 + d[28] * u22 + d[34] * u23);
      uik[23] = -(d[5] * u18 + d[11] * u19 + d[17] * u20 + d[23] * u21 + d[29] * u22 + d[35] * u23);

      uik[24] = -(d[0] * u24 + d[6] * u25 + d[12] * u26 + d[18] * u27 + d[24] * u28 + d[30] * u29);
      uik[25] = -(d[1] * u24 + d[7] * u25 + d[13] * u26 + d[19] * u27 + d[25] * u28 + d[31] * u29);
      uik[26] = -(d[2] * u24 + d[8] * u25 + d[14] * u26 + d[20] * u27 + d[26] * u28 + d[32] * u29);
      uik[27] = -(d[3] * u24 + d[9] * u25 + d[15] * u26 + d[21] * u27 + d[27] * u28 + d[33] * u29);
      uik[28] = -(d[4] * u24 + d[10] * u25 + d[16] * u26 + d[22] * u27 + d[28] * u28 + d[34] * u29);
      uik[29] = -(d[5] * u24 + d[11] * u25 + d[17] * u26 + d[23] * u27 + d[29] * u28 + d[35] * u29);

      uik[30] = -(d[0] * u30 + d[6] * u31 + d[12] * u32 + d[18] * u33 + d[24] * u34 + d[30] * u35);
      uik[31] = -(d[1] * u30 + d[7] * u31 + d[13] * u32 + d[19] * u33 + d[25] * u34 + d[31] * u35);
      uik[32] = -(d[2] * u30 + d[8] * u31 + d[14] * u32 + d[20] * u33 + d[26] * u34 + d[32] * u35);
      uik[33] = -(d[3] * u30 + d[9] * u31 + d[15] * u32 + d[21] * u33 + d[27] * u34 + d[33] * u35);
      uik[34] = -(d[4] * u30 + d[10] * u31 + d[16] * u32 + d[22] * u33 + d[28] * u34 + d[34] * u35);
      uik[35] = -(d[5] * u30 + d[11] * u31 + d[17] * u32 + d[23] * u33 + d[29] * u34 + d[35] * u35);

      /* update D(k) += -U(i,k)^T * U_bar(i,k) */
      dk[0] += uik[0] * u0 + uik[1] * u1 + uik[2] * u2 + uik[3] * u3 + uik[4] * u4 + uik[5] * u5;
      dk[1] += uik[6] * u0 + uik[7] * u1 + uik[8] * u2 + uik[9] * u3 + uik[10] * u4 + uik[11] * u5;
      dk[2] += uik[12] * u0 + uik[13] * u1 + uik[14] * u2 + uik[15] * u3 + uik[16] * u4 + uik[17] * u5;
      dk[3] += uik[18] * u0 + uik[19] * u1 + uik[20] * u2 + uik[21] * u3 + uik[22] * u4 + uik[23] * u5;
      dk[4] += uik[24] * u0 + uik[25] * u1 + uik[26] * u2 + uik[27] * u3 + uik[28] * u4 + uik[29] * u5;
      dk[5] += uik[30] * u0 + uik[31] * u1 + uik[32] * u2 + uik[33] * u3 + uik[34] * u4 + uik[35] * u5;

      dk[6] += uik[0] * u6 + uik[1] * u7 + uik[2] * u8 + uik[3] * u9 + uik[4] * u10 + uik[5] * u11;
      dk[7] += uik[6] * u6 + uik[7] * u7 + uik[8] * u8 + uik[9] * u9 + uik[10] * u10 + uik[11] * u11;
      dk[8] += uik[12] * u6 + uik[13] * u7 + uik[14] * u8 + uik[15] * u9 + uik[16] * u10 + uik[17] * u11;
      dk[9] += uik[18] * u6 + uik[19] * u7 + uik[20] * u8 + uik[21] * u9 + uik[22] * u10 + uik[23] * u11;
      dk[10] += uik[24] * u6 + uik[25] * u7 + uik[26] * u8 + uik[27] * u9 + uik[28] * u10 + uik[29] * u11;
      dk[11] += uik[30] * u6 + uik[31] * u7 + uik[32] * u8 + uik[33] * u9 + uik[34] * u10 + uik[35] * u11;

      dk[12] += uik[0] * u12 + uik[1] * u13 + uik[2] * u14 + uik[3] * u15 + uik[4] * u16 + uik[5] * u17;
      dk[13] += uik[6] * u12 + uik[7] * u13 + uik[8] * u14 + uik[9] * u15 + uik[10] * u16 + uik[11] * u17;
      dk[14] += uik[12] * u12 + uik[13] * u13 + uik[14] * u14 + uik[15] * u15 + uik[16] * u16 + uik[17] * u17;
      dk[15] += uik[18] * u12 + uik[19] * u13 + uik[20] * u14 + uik[21] * u15 + uik[22] * u16 + uik[23] * u17;
      dk[16] += uik[24] * u12 + uik[25] * u13 + uik[26] * u14 + uik[27] * u15 + uik[28] * u16 + uik[29] * u17;
      dk[17] += uik[30] * u12 + uik[31] * u13 + uik[32] * u14 + uik[33] * u15 + uik[34] * u16 + uik[35] * u17;

      dk[18] += uik[0] * u18 + uik[1] * u19 + uik[2] * u20 + uik[3] * u21 + uik[4] * u22 + uik[5] * u23;
      dk[19] += uik[6] * u18 + uik[7] * u19 + uik[8] * u20 + uik[9] * u21 + uik[10] * u22 + uik[11] * u23;
      dk[20] += uik[12] * u18 + uik[13] * u19 + uik[14] * u20 + uik[15] * u21 + uik[16] * u22 + uik[17] * u23;
      dk[21] += uik[18] * u18 + uik[19] * u19 + uik[20] * u20 + uik[21] * u21 + uik[22] * u22 + uik[23] * u23;
      dk[22] += uik[24] * u18 + uik[25] * u19 + uik[26] * u20 + uik[27] * u21 + uik[28] * u22 + uik[29] * u23;
      dk[23] += uik[30] * u18 + uik[31] * u19 + uik[32] * u20 + uik[33] * u21 + uik[34] * u22 + uik[35] * u23;

      dk[24] += uik[0] * u24 + uik[1] * u25 + uik[2] * u26 + uik[3] * u27 + uik[4] * u28 + uik[5] * u29;
      dk[25] += uik[6] * u24 + uik[7] * u25 + uik[8] * u26 + uik[9] * u27 + uik[10] * u28 + uik[11] * u29;
      dk[26] += uik[12] * u24 + uik[13] * u25 + uik[14] * u26 + uik[15] * u27 + uik[16] * u28 + uik[17] * u29;
      dk[27] += uik[18] * u24 + uik[19] * u25 + uik[20] * u26 + uik[21] * u27 + uik[22] * u28 + uik[23] * u29;
      dk[28] += uik[24] * u24 + uik[25] * u25 + uik[26] * u26 + uik[27] * u27 + uik[28] * u28 + uik[29] * u29;
      dk[29] += uik[30] * u24 + uik[31] * u25 + uik[32] * u26 + uik[33] * u27 + uik[34] * u28 + uik[35] * u29;

      dk[30] += uik[0] * u30 + uik[1] * u31 + uik[2] * u32 + uik[3] * u33 + uik[4] * u34 + uik[5] * u35;
      dk[31] += uik[6] * u30 + uik[7] * u31 + uik[8] * u32 + uik[9] * u33 + uik[10] * u34 + uik[11] * u35;
      dk[32] += uik[12] * u30 + uik[13] * u31 + uik[14] * u32 + uik[15] * u33 + uik[16] * u34 + uik[17] * u35;
      dk[33] += uik[18] * u30 + uik[19] * u31 + uik[20] * u32 + uik[21] * u33 + uik[22] * u34 + uik[23] * u35;
      dk[34] += uik[24] * u30 + uik[25] * u31 + uik[26] * u32 + uik[27] * u33 + uik[28] * u34 + uik[29] * u35;
      dk[35] += uik[30] * u30 + uik[31] * u31 + uik[32] * u32 + uik[33] * u33 + uik[34] * u34 + uik[35] * u35;

      PetscCall(PetscLogFlops(216.0 * 4.0));

      /* update -U(i,k) */
      PetscCall(PetscArraycpy(ba + ili * 36, uik, 36));

      /* add multiple of row i to k-th row ... */
      jmin = ili + 1;
      jmax = bi[i + 1];
      if (jmin < jmax) {
        for (j = jmin; j < jmax; j++) {
          /* w += -U(i,k)^T * U_bar(i,j) */
          wp = w + bj[j] * 36;
          u  = ba + j * 36;

          u0  = u[0];
          u1  = u[1];
          u2  = u[2];
          u3  = u[3];
          u4  = u[4];
          u5  = u[5];
          u6  = u[6];
          u7  = u[7];
          u8  = u[8];
          u9  = u[9];
          u10 = u[10];
          u11 = u[11];
          u12 = u[12];
          u13 = u[13];
          u14 = u[14];
          u15 = u[15];
          u16 = u[16];
          u17 = u[17];
          u18 = u[18];
          u19 = u[19];
          u20 = u[20];
          u21 = u[21];
          u22 = u[22];
          u23 = u[23];
          u24 = u[24];
          u25 = u[25];
          u26 = u[26];
          u27 = u[27];
          u28 = u[28];
          u29 = u[29];
          u30 = u[30];
          u31 = u[31];
          u32 = u[32];
          u33 = u[33];
          u34 = u[34];
          u35 = u[35];

          wp[0] += uik[0] * u0 + uik[1] * u1 + uik[2] * u2 + uik[3] * u3 + uik[4] * u4 + uik[5] * u5;
          wp[1] += uik[6] * u0 + uik[7] * u1 + uik[8] * u2 + uik[9] * u3 + uik[10] * u4 + uik[11] * u5;
          wp[2] += uik[12] * u0 + uik[13] * u1 + uik[14] * u2 + uik[15] * u3 + uik[16] * u4 + uik[17] * u5;
          wp[3] += uik[18] * u0 + uik[19] * u1 + uik[20] * u2 + uik[21] * u3 + uik[22] * u4 + uik[23] * u5;
          wp[4] += uik[24] * u0 + uik[25] * u1 + uik[26] * u2 + uik[27] * u3 + uik[28] * u4 + uik[29] * u5;
          wp[5] += uik[30] * u0 + uik[31] * u1 + uik[32] * u2 + uik[33] * u3 + uik[34] * u4 + uik[35] * u5;

          wp[6] += uik[0] * u6 + uik[1] * u7 + uik[2] * u8 + uik[3] * u9 + uik[4] * u10 + uik[5] * u11;
          wp[7] += uik[6] * u6 + uik[7] * u7 + uik[8] * u8 + uik[9] * u9 + uik[10] * u10 + uik[11] * u11;
          wp[8] += uik[12] * u6 + uik[13] * u7 + uik[14] * u8 + uik[15] * u9 + uik[16] * u10 + uik[17] * u11;
          wp[9] += uik[18] * u6 + uik[19] * u7 + uik[20] * u8 + uik[21] * u9 + uik[22] * u10 + uik[23] * u11;
          wp[10] += uik[24] * u6 + uik[25] * u7 + uik[26] * u8 + uik[27] * u9 + uik[28] * u10 + uik[29] * u11;
          wp[11] += uik[30] * u6 + uik[31] * u7 + uik[32] * u8 + uik[33] * u9 + uik[34] * u10 + uik[35] * u11;

          wp[12] += uik[0] * u12 + uik[1] * u13 + uik[2] * u14 + uik[3] * u15 + uik[4] * u16 + uik[5] * u17;
          wp[13] += uik[6] * u12 + uik[7] * u13 + uik[8] * u14 + uik[9] * u15 + uik[10] * u16 + uik[11] * u17;
          wp[14] += uik[12] * u12 + uik[13] * u13 + uik[14] * u14 + uik[15] * u15 + uik[16] * u16 + uik[17] * u17;
          wp[15] += uik[18] * u12 + uik[19] * u13 + uik[20] * u14 + uik[21] * u15 + uik[22] * u16 + uik[23] * u17;
          wp[16] += uik[24] * u12 + uik[25] * u13 + uik[26] * u14 + uik[27] * u15 + uik[28] * u16 + uik[29] * u17;
          wp[17] += uik[30] * u12 + uik[31] * u13 + uik[32] * u14 + uik[33] * u15 + uik[34] * u16 + uik[35] * u17;

          wp[18] += uik[0] * u18 + uik[1] * u19 + uik[2] * u20 + uik[3] * u21 + uik[4] * u22 + uik[5] * u23;
          wp[19] += uik[6] * u18 + uik[7] * u19 + uik[8] * u20 + uik[9] * u21 + uik[10] * u22 + uik[11] * u23;
          wp[20] += uik[12] * u18 + uik[13] * u19 + uik[14] * u20 + uik[15] * u21 + uik[16] * u22 + uik[17] * u23;
          wp[21] += uik[18] * u18 + uik[19] * u19 + uik[20] * u20 + uik[21] * u21 + uik[22] * u22 + uik[23] * u23;
          wp[22] += uik[24] * u18 + uik[25] * u19 + uik[26] * u20 + uik[27] * u21 + uik[28] * u22 + uik[29] * u23;
          wp[23] += uik[30] * u18 + uik[31] * u19 + uik[32] * u20 + uik[33] * u21 + uik[34] * u22 + uik[35] * u23;

          wp[24] += uik[0] * u24 + uik[1] * u25 + uik[2] * u26 + uik[3] * u27 + uik[4] * u28 + uik[5] * u29;
          wp[25] += uik[6] * u24 + uik[7] * u25 + uik[8] * u26 + uik[9] * u27 + uik[10] * u28 + uik[11] * u29;
          wp[26] += uik[12] * u24 + uik[13] * u25 + uik[14] * u26 + uik[15] * u27 + uik[16] * u28 + uik[17] * u29;
          wp[27] += uik[18] * u24 + uik[19] * u25 + uik[20] * u26 + uik[21] * u27 + uik[22] * u28 + uik[23] * u29;
          wp[28] += uik[24] * u24 + uik[25] * u25 + uik[26] * u26 + uik[27] * u27 + uik[28] * u28 + uik[29] * u29;
          wp[29] += uik[30] * u24 + uik[31] * u25 + uik[32] * u26 + uik[33] * u27 + uik[34] * u28 + uik[35] * u29;

          wp[30] += uik[0] * u30 + uik[1] * u31 + uik[2] * u32 + uik[3] * u33 + uik[4] * u34 + uik[5] * u35;
          wp[31] += uik[6] * u30 + uik[7] * u31 + uik[8] * u32 + uik[9] * u33 + uik[10] * u34 + uik[11] * u35;
          wp[32] += uik[12] * u30 + uik[13] * u31 + uik[14] * u32 + uik[15] * u33 + uik[16] * u34 + uik[17] * u35;
          wp[33] += uik[18] * u30 + uik[19] * u31 + uik[20] * u32 + uik[21] * u33 + uik[22] * u34 + uik[23] * u35;
          wp[34] += uik[24] * u30 + uik[25] * u31 + uik[26] * u32 + uik[27] * u33 + uik[28] * u34 + uik[29] * u35;
          wp[35] += uik[30] * u30 + uik[31] * u31 + uik[32] * u32 + uik[33] * u33 + uik[34] * u34 + uik[35] * u35;
        }
        PetscCall(PetscLogFlops(2.0 * 216.0 * (jmax - jmin)));

        /* ... add i to row list for next nonzero entry */
        il[i] = jmin; /* update il(i) in column k+1, ... mbs-1 */
        j     = bj[jmin];
        jl[i] = jl[j];
        jl[j] = i; /* update jl */
      }
      i = nexti;
    }

    /* save nonzero entries in k-th row of U ... */

    /* invert diagonal block */
    d = ba + k * 36;
    PetscCall(PetscArraycpy(d, dk, 36));
    PetscCall(PetscKernel_A_gets_inverse_A_6(d, shift, allowzeropivot, &zeropivotdetected));
    if (zeropivotdetected) C->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;

    jmin = bi[k];
    jmax = bi[k + 1];
    if (jmin < jmax) {
      for (j = jmin; j < jmax; j++) {
        vj = bj[j]; /* block col. index of U */
        u  = ba + j * 36;
        wp = w + vj * 36;
        for (k1 = 0; k1 < 36; k1++) {
          *u++  = *wp;
          *wp++ = 0.0;
        }
      }

      /* ... add k to row list for first nonzero entry in k-th row */
      il[k] = jmin;
      i     = bj[jmin];
      jl[k] = jl[i];
      jl[i] = k;
    }
  }

  PetscCall(PetscFree(w));
  PetscCall(PetscFree2(il, jl));
  PetscCall(PetscFree2(dk, uik));
  if (a->permute) PetscCall(PetscFree(aa));

  PetscCall(ISRestoreIndices(perm, &perm_ptr));

  C->ops->solve          = MatSolve_SeqSBAIJ_6_inplace;
  C->ops->solvetranspose = MatSolve_SeqSBAIJ_6_inplace;
  C->assembled           = PETSC_TRUE;
  C->preallocated        = PETSC_TRUE;

  PetscCall(PetscLogFlops(1.3333 * 216 * b->mbs)); /* from inverting diagonal blocks */
  PetscFunctionReturn(PETSC_SUCCESS);
}
