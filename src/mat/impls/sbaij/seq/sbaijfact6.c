
#include <../src/mat/impls/sbaij/seq/sbaij.h>
#include <../src/mat/blockinvert.h>

/* Version for when blocks are 4 by 4  */
#undef __FUNCT__
#define __FUNCT__ "MatCholeskyFactorNumeric_SeqSBAIJ_4"
PetscErrorCode MatCholeskyFactorNumeric_SeqSBAIJ_4(Mat C,Mat A,const MatFactorInfo *info)
{
  Mat_SeqSBAIJ   *a = (Mat_SeqSBAIJ*)A->data,*b = (Mat_SeqSBAIJ *)C->data;
  IS             perm = b->row;
  PetscErrorCode ierr;
  const PetscInt *ai,*aj,*perm_ptr,mbs=a->mbs,*bi=b->i,*bj=b->j;
  PetscInt       i,j,*a2anew,k,k1,jmin,jmax,*jl,*il,vj,nexti,ili;
  MatScalar      *ba = b->a,*aa,*ap,*dk,*uik;
  MatScalar      *u,*diag,*rtmp,*rtmp_ptr;
  PetscBool      pivotinblocks = b->pivotinblocks;
  PetscReal      shift = info->shiftamount;

  PetscFunctionBegin;
  /* initialization */
  ierr = PetscMalloc(16*mbs*sizeof(MatScalar),&rtmp);CHKERRQ(ierr);
  ierr = PetscMemzero(rtmp,16*mbs*sizeof(MatScalar));CHKERRQ(ierr);
  ierr = PetscMalloc2(mbs,PetscInt,&il,mbs,PetscInt,&jl);CHKERRQ(ierr);
  for (i=0; i<mbs; i++) {
    jl[i] = mbs; il[0] = 0;
  }
  ierr = PetscMalloc2(16,MatScalar,&dk,16,MatScalar,&uik);CHKERRQ(ierr);
  ierr = ISGetIndices(perm,&perm_ptr);CHKERRQ(ierr);

  /* check permutation */
  if (!a->permute){
    ai = a->i; aj = a->j; aa = a->a;
  } else {
    ai   = a->inew; aj = a->jnew;
    ierr = PetscMalloc(16*ai[mbs]*sizeof(MatScalar),&aa);CHKERRQ(ierr);
    ierr = PetscMemcpy(aa,a->a,16*ai[mbs]*sizeof(MatScalar));CHKERRQ(ierr);
    ierr = PetscMalloc(ai[mbs]*sizeof(PetscInt),&a2anew);CHKERRQ(ierr);
    ierr = PetscMemcpy(a2anew,a->a2anew,(ai[mbs])*sizeof(PetscInt));CHKERRQ(ierr);

    for (i=0; i<mbs; i++){
      jmin = ai[i]; jmax = ai[i+1];
      for (j=jmin; j<jmax; j++){
        while (a2anew[j] != j){
          k = a2anew[j]; a2anew[j] = a2anew[k]; a2anew[k] = k;
          for (k1=0; k1<16; k1++){
            dk[k1]       = aa[k*16+k1];
            aa[k*16+k1] = aa[j*16+k1];
            aa[j*16+k1] = dk[k1];
          }
        }
        /* transform columnoriented blocks that lie in the lower triangle to roworiented blocks */
        if (i > aj[j]){
          /* printf("change orientation, row: %d, col: %d\n",i,aj[j]); */
          ap = aa + j*16;                     /* ptr to the beginning of j-th block of aa */
          for (k=0; k<16; k++) dk[k] = ap[k]; /* dk <- j-th block of aa */
          for (k=0; k<4; k++){               /* j-th block of aa <- dk^T */
            for (k1=0; k1<4; k1++) *ap++ = dk[k + 4*k1];
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
      ap = aa + jmin*16;
      for (j = jmin; j < jmax; j++){
        vj = perm_ptr[aj[j]];         /* block col. index */
        rtmp_ptr = rtmp + vj*16;
        for (i=0; i<16; i++) *rtmp_ptr++ = *ap++;
      }
    }

    /* modify k-th row by adding in those rows i with U(i,k) != 0 */
    ierr = PetscMemcpy(dk,rtmp+k*16,16*sizeof(MatScalar));CHKERRQ(ierr);
    i = jl[k]; /* first row to be added to k_th row  */

    while (i < mbs){
      nexti = jl[i]; /* next row to be added to k_th row */

      /* compute multiplier */
      ili = il[i];  /* index of first nonzero element in U(i,k:bms-1) */

      /* uik = -inv(Di)*U_bar(i,k) */
      diag = ba + i*16;
      u    = ba + ili*16;

      uik[0] = -(diag[0]*u[0] + diag[4]*u[1] + diag[8]*u[2] + diag[12]*u[3]);
      uik[1] = -(diag[1]*u[0] + diag[5]*u[1] + diag[9]*u[2] + diag[13]*u[3]);
      uik[2] = -(diag[2]*u[0] + diag[6]*u[1] + diag[10]*u[2]+ diag[14]*u[3]);
      uik[3] = -(diag[3]*u[0] + diag[7]*u[1] + diag[11]*u[2]+ diag[15]*u[3]);

      uik[4] = -(diag[0]*u[4] + diag[4]*u[5] + diag[8]*u[6] + diag[12]*u[7]);
      uik[5] = -(diag[1]*u[4] + diag[5]*u[5] + diag[9]*u[6] + diag[13]*u[7]);
      uik[6] = -(diag[2]*u[4] + diag[6]*u[5] + diag[10]*u[6]+ diag[14]*u[7]);
      uik[7] = -(diag[3]*u[4] + diag[7]*u[5] + diag[11]*u[6]+ diag[15]*u[7]);

      uik[8] = -(diag[0]*u[8] + diag[4]*u[9] + diag[8]*u[10] + diag[12]*u[11]);
      uik[9] = -(diag[1]*u[8] + diag[5]*u[9] + diag[9]*u[10] + diag[13]*u[11]);
      uik[10]= -(diag[2]*u[8] + diag[6]*u[9] + diag[10]*u[10]+ diag[14]*u[11]);
      uik[11]= -(diag[3]*u[8] + diag[7]*u[9] + diag[11]*u[10]+ diag[15]*u[11]);

      uik[12]= -(diag[0]*u[12] + diag[4]*u[13] + diag[8]*u[14] + diag[12]*u[15]);
      uik[13]= -(diag[1]*u[12] + diag[5]*u[13] + diag[9]*u[14] + diag[13]*u[15]);
      uik[14]= -(diag[2]*u[12] + diag[6]*u[13] + diag[10]*u[14]+ diag[14]*u[15]);
      uik[15]= -(diag[3]*u[12] + diag[7]*u[13] + diag[11]*u[14]+ diag[15]*u[15]);

      /* update D(k) += -U(i,k)^T * U_bar(i,k) */
      dk[0] += uik[0]*u[0] + uik[1]*u[1] + uik[2]*u[2] + uik[3]*u[3];
      dk[1] += uik[4]*u[0] + uik[5]*u[1] + uik[6]*u[2] + uik[7]*u[3];
      dk[2] += uik[8]*u[0] + uik[9]*u[1] + uik[10]*u[2]+ uik[11]*u[3];
      dk[3] += uik[12]*u[0]+ uik[13]*u[1]+ uik[14]*u[2]+ uik[15]*u[3];

      dk[4] += uik[0]*u[4] + uik[1]*u[5] + uik[2]*u[6] + uik[3]*u[7];
      dk[5] += uik[4]*u[4] + uik[5]*u[5] + uik[6]*u[6] + uik[7]*u[7];
      dk[6] += uik[8]*u[4] + uik[9]*u[5] + uik[10]*u[6]+ uik[11]*u[7];
      dk[7] += uik[12]*u[4]+ uik[13]*u[5]+ uik[14]*u[6]+ uik[15]*u[7];

      dk[8] += uik[0]*u[8] + uik[1]*u[9] + uik[2]*u[10] + uik[3]*u[11];
      dk[9] += uik[4]*u[8] + uik[5]*u[9] + uik[6]*u[10] + uik[7]*u[11];
      dk[10]+= uik[8]*u[8] + uik[9]*u[9] + uik[10]*u[10]+ uik[11]*u[11];
      dk[11]+= uik[12]*u[8]+ uik[13]*u[9]+ uik[14]*u[10]+ uik[15]*u[11];

      dk[12]+= uik[0]*u[12] + uik[1]*u[13] + uik[2]*u[14] + uik[3]*u[15];
      dk[13]+= uik[4]*u[12] + uik[5]*u[13] + uik[6]*u[14] + uik[7]*u[15];
      dk[14]+= uik[8]*u[12] + uik[9]*u[13] + uik[10]*u[14]+ uik[11]*u[15];
      dk[15]+= uik[12]*u[12]+ uik[13]*u[13]+ uik[14]*u[14]+ uik[15]*u[15];

      ierr = PetscLogFlops(64.0*4.0);CHKERRQ(ierr);

      /* update -U(i,k) */
      ierr = PetscMemcpy(ba+ili*16,uik,16*sizeof(MatScalar));CHKERRQ(ierr);

      /* add multiple of row i to k-th row ... */
      jmin = ili + 1; jmax = bi[i+1];
      if (jmin < jmax){
        for (j=jmin; j<jmax; j++) {
          /* rtmp += -U(i,k)^T * U_bar(i,j) */
          rtmp_ptr = rtmp + bj[j]*16;
          u = ba + j*16;
          rtmp_ptr[0] += uik[0]*u[0] + uik[1]*u[1] + uik[2]*u[2] + uik[3]*u[3];
          rtmp_ptr[1] += uik[4]*u[0] + uik[5]*u[1] + uik[6]*u[2] + uik[7]*u[3];
          rtmp_ptr[2] += uik[8]*u[0] + uik[9]*u[1] + uik[10]*u[2]+ uik[11]*u[3];
          rtmp_ptr[3] += uik[12]*u[0]+ uik[13]*u[1]+ uik[14]*u[2]+ uik[15]*u[3];

          rtmp_ptr[4] += uik[0]*u[4] + uik[1]*u[5] + uik[2]*u[6] + uik[3]*u[7];
          rtmp_ptr[5] += uik[4]*u[4] + uik[5]*u[5] + uik[6]*u[6] + uik[7]*u[7];
          rtmp_ptr[6] += uik[8]*u[4] + uik[9]*u[5] + uik[10]*u[6]+ uik[11]*u[7];
          rtmp_ptr[7] += uik[12]*u[4]+ uik[13]*u[5]+ uik[14]*u[6]+ uik[15]*u[7];

          rtmp_ptr[8] += uik[0]*u[8] + uik[1]*u[9] + uik[2]*u[10] + uik[3]*u[11];
          rtmp_ptr[9] += uik[4]*u[8] + uik[5]*u[9] + uik[6]*u[10] + uik[7]*u[11];
          rtmp_ptr[10]+= uik[8]*u[8] + uik[9]*u[9] + uik[10]*u[10]+ uik[11]*u[11];
          rtmp_ptr[11]+= uik[12]*u[8]+ uik[13]*u[9]+ uik[14]*u[10]+ uik[15]*u[11];

          rtmp_ptr[12]+= uik[0]*u[12] + uik[1]*u[13] + uik[2]*u[14] + uik[3]*u[15];
          rtmp_ptr[13]+= uik[4]*u[12] + uik[5]*u[13] + uik[6]*u[14] + uik[7]*u[15];
          rtmp_ptr[14]+= uik[8]*u[12] + uik[9]*u[13] + uik[10]*u[14]+ uik[11]*u[15];
          rtmp_ptr[15]+= uik[12]*u[12]+ uik[13]*u[13]+ uik[14]*u[14]+ uik[15]*u[15];
        }
        ierr = PetscLogFlops(2.0*64.0*(jmax-jmin));CHKERRQ(ierr);

        /* ... add i to row list for next nonzero entry */
        il[i] = jmin;             /* update il(i) in column k+1, ... mbs-1 */
        j     = bj[jmin];
        jl[i] = jl[j]; jl[j] = i; /* update jl */
      }
      i = nexti;
    }

    /* save nonzero entries in k-th row of U ... */

    /* invert diagonal block */
    diag = ba+k*16;
    ierr = PetscMemcpy(diag,dk,16*sizeof(MatScalar));CHKERRQ(ierr);

    if (pivotinblocks) {
      ierr = PetscKernel_A_gets_inverse_A_4(diag,shift);CHKERRQ(ierr);
    } else {
      ierr = PetscKernel_A_gets_inverse_A_4_nopivot(diag);CHKERRQ(ierr);
    }

    jmin = bi[k]; jmax = bi[k+1];
    if (jmin < jmax) {
      for (j=jmin; j<jmax; j++){
         vj = bj[j];           /* block col. index of U */
         u   = ba + j*16;
         rtmp_ptr = rtmp + vj*16;
         for (k1=0; k1<16; k1++){
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

  ierr = PetscFree(rtmp);CHKERRQ(ierr);
  ierr = PetscFree2(il,jl);CHKERRQ(ierr);
  ierr = PetscFree2(dk,uik);CHKERRQ(ierr);
  if (a->permute){
    ierr = PetscFree(aa);CHKERRQ(ierr);
  }

  ierr = ISRestoreIndices(perm,&perm_ptr);CHKERRQ(ierr);
  C->ops->solve          = MatSolve_SeqSBAIJ_4_inplace;
  C->ops->solvetranspose = MatSolve_SeqSBAIJ_4_inplace;
  C->assembled = PETSC_TRUE;
  C->preallocated = PETSC_TRUE;
  ierr = PetscLogFlops(1.3333*64*b->mbs);CHKERRQ(ierr); /* from inverting diagonal blocks */
  PetscFunctionReturn(0);
}
