
/*
    Defines the basic matrix operations for the BAIJ (compressed row)
  matrix storage format.
*/
#include <../src/mat/impls/baij/seq/baij.h>  /*I   "petscmat.h"  I*/
#include <petscblaslapack.h>
#include <petsc/private/kernels/blockinvert.h>
#include <petsc/private/kernels/blockmatmult.h>

#if defined(PETSC_HAVE_HYPRE)
PETSC_INTERN PetscErrorCode MatConvert_AIJ_HYPRE(Mat,MatType,MatReuse,Mat*);
#endif

#if defined(PETSC_HAVE_MKL_SPARSE_OPTIMIZE)
PETSC_INTERN PetscErrorCode MatConvert_SeqBAIJ_SeqBAIJMKL(Mat,MatType,MatReuse,Mat*);
#endif
PETSC_INTERN PetscErrorCode MatConvert_XAIJ_IS(Mat,MatType,MatReuse,Mat*);

PetscErrorCode MatGetColumnReductions_SeqBAIJ(Mat A,PetscInt type,PetscReal *reductions)
{
  Mat_SeqBAIJ    *a_aij = (Mat_SeqBAIJ*) A->data;
  PetscInt       m,n,i;
  PetscInt       ib,jb,bs = A->rmap->bs;
  MatScalar      *a_val = a_aij->a;

  PetscFunctionBegin;
  PetscCall(MatGetSize(A,&m,&n));
  for (i=0; i<n; i++) reductions[i] = 0.0;
  if (type == NORM_2) {
    for (i=a_aij->i[0]; i<a_aij->i[A->rmap->n/bs]; i++) {
      for (jb=0; jb<bs; jb++) {
        for (ib=0; ib<bs; ib++) {
          reductions[A->cmap->rstart + a_aij->j[i] * bs + jb] += PetscAbsScalar(*a_val * *a_val);
          a_val++;
        }
      }
    }
  } else if (type == NORM_1) {
    for (i=a_aij->i[0]; i<a_aij->i[A->rmap->n/bs]; i++) {
      for (jb=0; jb<bs; jb++) {
        for (ib=0; ib<bs; ib++) {
          reductions[A->cmap->rstart + a_aij->j[i] * bs + jb] += PetscAbsScalar(*a_val);
          a_val++;
        }
      }
    }
  } else if (type == NORM_INFINITY) {
    for (i=a_aij->i[0]; i<a_aij->i[A->rmap->n/bs]; i++) {
      for (jb=0; jb<bs; jb++) {
        for (ib=0; ib<bs; ib++) {
          int col = A->cmap->rstart + a_aij->j[i] * bs + jb;
          reductions[col] = PetscMax(PetscAbsScalar(*a_val), reductions[col]);
          a_val++;
        }
      }
    }
  } else if (type == REDUCTION_SUM_REALPART || type == REDUCTION_MEAN_REALPART) {
    for (i=a_aij->i[0]; i<a_aij->i[A->rmap->n/bs]; i++) {
      for (jb=0; jb<bs; jb++) {
        for (ib=0; ib<bs; ib++) {
          reductions[A->cmap->rstart + a_aij->j[i] * bs + jb] += PetscRealPart(*a_val);
          a_val++;
        }
      }
    }
  } else if (type == REDUCTION_SUM_IMAGINARYPART || type == REDUCTION_MEAN_IMAGINARYPART) {
    for (i=a_aij->i[0]; i<a_aij->i[A->rmap->n/bs]; i++) {
      for (jb=0; jb<bs; jb++) {
        for (ib=0; ib<bs; ib++) {
          reductions[A->cmap->rstart + a_aij->j[i] * bs + jb] += PetscImaginaryPart(*a_val);
          a_val++;
        }
      }
    }
  } else SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"Unknown reduction type");
  if (type == NORM_2) {
    for (i=0; i<n; i++) reductions[i] = PetscSqrtReal(reductions[i]);
  } else if (type == REDUCTION_MEAN_REALPART || type == REDUCTION_MEAN_IMAGINARYPART) {
    for (i=0; i<n; i++) reductions[i] /= m;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatInvertBlockDiagonal_SeqBAIJ(Mat A,const PetscScalar **values)
{
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*) A->data;
  PetscInt       *diag_offset,i,bs = A->rmap->bs,mbs = a->mbs,ipvt[5],bs2 = bs*bs,*v_pivots;
  MatScalar      *v    = a->a,*odiag,*diag,work[25],*v_work;
  PetscReal      shift = 0.0;
  PetscBool      allowzeropivot,zeropivotdetected=PETSC_FALSE;

  PetscFunctionBegin;
  allowzeropivot = PetscNot(A->erroriffailure);

  if (a->idiagvalid) {
    if (values) *values = a->idiag;
    PetscFunctionReturn(0);
  }
  PetscCall(MatMarkDiagonal_SeqBAIJ(A));
  diag_offset = a->diag;
  if (!a->idiag) {
    PetscCall(PetscMalloc1(bs2*mbs,&a->idiag));
    PetscCall(PetscLogObjectMemory((PetscObject)A,bs2*mbs*sizeof(PetscScalar)));
  }
  diag  = a->idiag;
  if (values) *values = a->idiag;
  /* factor and invert each block */
  switch (bs) {
  case 1:
    for (i=0; i<mbs; i++) {
      odiag    = v + 1*diag_offset[i];
      diag[0]  = odiag[0];

      if (PetscAbsScalar(diag[0] + shift) < PETSC_MACHINE_EPSILON) {
        if (allowzeropivot) {
          A->factorerrortype             = MAT_FACTOR_NUMERIC_ZEROPIVOT;
          A->factorerror_zeropivot_value = PetscAbsScalar(diag[0]);
          A->factorerror_zeropivot_row   = i;
          PetscCall(PetscInfo(A,"Zero pivot, row %" PetscInt_FMT "\n",i));
        } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MAT_LU_ZRPVT,"Zero pivot, row %" PetscInt_FMT " pivot value %g tolerance %g",i,(double)PetscAbsScalar(diag[0]),(double)PETSC_MACHINE_EPSILON);
      }

      diag[0]  = (PetscScalar)1.0 / (diag[0] + shift);
      diag    += 1;
    }
    break;
  case 2:
    for (i=0; i<mbs; i++) {
      odiag    = v + 4*diag_offset[i];
      diag[0]  = odiag[0]; diag[1] = odiag[1]; diag[2] = odiag[2]; diag[3] = odiag[3];
      PetscCall(PetscKernel_A_gets_inverse_A_2(diag,shift,allowzeropivot,&zeropivotdetected));
      if (zeropivotdetected) A->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;
      diag    += 4;
    }
    break;
  case 3:
    for (i=0; i<mbs; i++) {
      odiag    = v + 9*diag_offset[i];
      diag[0]  = odiag[0]; diag[1] = odiag[1]; diag[2] = odiag[2]; diag[3] = odiag[3];
      diag[4]  = odiag[4]; diag[5] = odiag[5]; diag[6] = odiag[6]; diag[7] = odiag[7];
      diag[8]  = odiag[8];
      PetscCall(PetscKernel_A_gets_inverse_A_3(diag,shift,allowzeropivot,&zeropivotdetected));
      if (zeropivotdetected) A->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;
      diag    += 9;
    }
    break;
  case 4:
    for (i=0; i<mbs; i++) {
      odiag  = v + 16*diag_offset[i];
      PetscCall(PetscArraycpy(diag,odiag,16));
      PetscCall(PetscKernel_A_gets_inverse_A_4(diag,shift,allowzeropivot,&zeropivotdetected));
      if (zeropivotdetected) A->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;
      diag  += 16;
    }
    break;
  case 5:
    for (i=0; i<mbs; i++) {
      odiag  = v + 25*diag_offset[i];
      PetscCall(PetscArraycpy(diag,odiag,25));
      PetscCall(PetscKernel_A_gets_inverse_A_5(diag,ipvt,work,shift,allowzeropivot,&zeropivotdetected));
      if (zeropivotdetected) A->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;
      diag  += 25;
    }
    break;
  case 6:
    for (i=0; i<mbs; i++) {
      odiag  = v + 36*diag_offset[i];
      PetscCall(PetscArraycpy(diag,odiag,36));
      PetscCall(PetscKernel_A_gets_inverse_A_6(diag,shift,allowzeropivot,&zeropivotdetected));
      if (zeropivotdetected) A->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;
      diag  += 36;
    }
    break;
  case 7:
    for (i=0; i<mbs; i++) {
      odiag  = v + 49*diag_offset[i];
      PetscCall(PetscArraycpy(diag,odiag,49));
      PetscCall(PetscKernel_A_gets_inverse_A_7(diag,shift,allowzeropivot,&zeropivotdetected));
      if (zeropivotdetected) A->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;
      diag  += 49;
    }
    break;
  default:
    PetscCall(PetscMalloc2(bs,&v_work,bs,&v_pivots));
    for (i=0; i<mbs; i++) {
      odiag  = v + bs2*diag_offset[i];
      PetscCall(PetscArraycpy(diag,odiag,bs2));
      PetscCall(PetscKernel_A_gets_inverse_A(bs,diag,v_pivots,v_work,allowzeropivot,&zeropivotdetected));
      if (zeropivotdetected) A->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;
      diag  += bs2;
    }
    PetscCall(PetscFree2(v_work,v_pivots));
  }
  a->idiagvalid = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PetscErrorCode MatSOR_SeqBAIJ(Mat A,Vec bb,PetscReal omega,MatSORType flag,PetscReal fshift,PetscInt its,PetscInt lits,Vec xx)
{
  Mat_SeqBAIJ       *a = (Mat_SeqBAIJ*)A->data;
  PetscScalar       *x,*work,*w,*workt,*t;
  const MatScalar   *v,*aa = a->a, *idiag;
  const PetscScalar *b,*xb;
  PetscScalar       s[7], xw[7]={0}; /* avoid some compilers thinking xw is uninitialized */
  PetscInt          m = a->mbs,i,i2,nz,bs = A->rmap->bs,bs2 = bs*bs,k,j,idx,it;
  const PetscInt    *diag,*ai = a->i,*aj = a->j,*vi;

  PetscFunctionBegin;
  its = its*lits;
  PetscCheck(!(flag & SOR_EISENSTAT),PETSC_COMM_SELF,PETSC_ERR_SUP,"No support yet for Eisenstat");
  PetscCheck(its > 0,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Relaxation requires global its %" PetscInt_FMT " and local its %" PetscInt_FMT " both positive",its,lits);
  PetscCheck(!fshift,PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for diagonal shift");
  PetscCheck(omega == 1.0,PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for non-trivial relaxation factor");
  PetscCheck(!(flag & SOR_APPLY_UPPER) && !(flag & SOR_APPLY_LOWER),PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for applying upper or lower triangular parts");

  if (!a->idiagvalid) PetscCall(MatInvertBlockDiagonal(A,NULL));

  if (!m) PetscFunctionReturn(0);
  diag  = a->diag;
  idiag = a->idiag;
  k    = PetscMax(A->rmap->n,A->cmap->n);
  if (!a->mult_work) {
    PetscCall(PetscMalloc1(k+1,&a->mult_work));
  }
  if (!a->sor_workt) {
    PetscCall(PetscMalloc1(k,&a->sor_workt));
  }
  if (!a->sor_work) {
    PetscCall(PetscMalloc1(bs,&a->sor_work));
  }
  work = a->mult_work;
  t    = a->sor_workt;
  w    = a->sor_work;

  PetscCall(VecGetArray(xx,&x));
  PetscCall(VecGetArrayRead(bb,&b));

  if (flag & SOR_ZERO_INITIAL_GUESS) {
    if (flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP) {
      switch (bs) {
      case 1:
        PetscKernel_v_gets_A_times_w_1(x,idiag,b);
        t[0] = b[0];
        i2     = 1;
        idiag += 1;
        for (i=1; i<m; i++) {
          v  = aa + ai[i];
          vi = aj + ai[i];
          nz = diag[i] - ai[i];
          s[0] = b[i2];
          for (j=0; j<nz; j++) {
            xw[0] = x[vi[j]];
            PetscKernel_v_gets_v_minus_A_times_w_1(s,(v+j),xw);
          }
          t[i2] = s[0];
          PetscKernel_v_gets_A_times_w_1(xw,idiag,s);
          x[i2]  = xw[0];
          idiag += 1;
          i2    += 1;
        }
        break;
      case 2:
        PetscKernel_v_gets_A_times_w_2(x,idiag,b);
        t[0] = b[0]; t[1] = b[1];
        i2     = 2;
        idiag += 4;
        for (i=1; i<m; i++) {
          v  = aa + 4*ai[i];
          vi = aj + ai[i];
          nz = diag[i] - ai[i];
          s[0] = b[i2]; s[1] = b[i2+1];
          for (j=0; j<nz; j++) {
            idx = 2*vi[j];
            it  = 4*j;
            xw[0] = x[idx]; xw[1] = x[1+idx];
            PetscKernel_v_gets_v_minus_A_times_w_2(s,(v+it),xw);
          }
          t[i2] = s[0]; t[i2+1] = s[1];
          PetscKernel_v_gets_A_times_w_2(xw,idiag,s);
          x[i2]   = xw[0]; x[i2+1] = xw[1];
          idiag  += 4;
          i2     += 2;
        }
        break;
      case 3:
        PetscKernel_v_gets_A_times_w_3(x,idiag,b);
        t[0] = b[0]; t[1] = b[1]; t[2] = b[2];
        i2     = 3;
        idiag += 9;
        for (i=1; i<m; i++) {
          v  = aa + 9*ai[i];
          vi = aj + ai[i];
          nz = diag[i] - ai[i];
          s[0] = b[i2]; s[1] = b[i2+1]; s[2] = b[i2+2];
          while (nz--) {
            idx = 3*(*vi++);
            xw[0] = x[idx]; xw[1] = x[1+idx]; xw[2] = x[2+idx];
            PetscKernel_v_gets_v_minus_A_times_w_3(s,v,xw);
            v  += 9;
          }
          t[i2] = s[0]; t[i2+1] = s[1]; t[i2+2] = s[2];
          PetscKernel_v_gets_A_times_w_3(xw,idiag,s);
          x[i2] = xw[0]; x[i2+1] = xw[1]; x[i2+2] = xw[2];
          idiag  += 9;
          i2     += 3;
        }
        break;
      case 4:
        PetscKernel_v_gets_A_times_w_4(x,idiag,b);
        t[0] = b[0]; t[1] = b[1]; t[2] = b[2]; t[3] = b[3];
        i2     = 4;
        idiag += 16;
        for (i=1; i<m; i++) {
          v  = aa + 16*ai[i];
          vi = aj + ai[i];
          nz = diag[i] - ai[i];
          s[0] = b[i2]; s[1] = b[i2+1]; s[2] = b[i2+2]; s[3] = b[i2+3];
          while (nz--) {
            idx = 4*(*vi++);
            xw[0]  = x[idx]; xw[1] = x[1+idx]; xw[2] = x[2+idx]; xw[3] = x[3+idx];
            PetscKernel_v_gets_v_minus_A_times_w_4(s,v,xw);
            v  += 16;
          }
          t[i2] = s[0]; t[i2+1] = s[1]; t[i2+2] = s[2]; t[i2 + 3] = s[3];
          PetscKernel_v_gets_A_times_w_4(xw,idiag,s);
          x[i2] = xw[0]; x[i2+1] = xw[1]; x[i2+2] = xw[2]; x[i2+3] = xw[3];
          idiag  += 16;
          i2     += 4;
        }
        break;
      case 5:
        PetscKernel_v_gets_A_times_w_5(x,idiag,b);
        t[0] = b[0]; t[1] = b[1]; t[2] = b[2]; t[3] = b[3]; t[4] = b[4];
        i2     = 5;
        idiag += 25;
        for (i=1; i<m; i++) {
          v  = aa + 25*ai[i];
          vi = aj + ai[i];
          nz = diag[i] - ai[i];
          s[0] = b[i2]; s[1] = b[i2+1]; s[2] = b[i2+2]; s[3] = b[i2+3]; s[4] = b[i2+4];
          while (nz--) {
            idx = 5*(*vi++);
            xw[0]  = x[idx]; xw[1] = x[1+idx]; xw[2] = x[2+idx]; xw[3] = x[3+idx]; xw[4] = x[4+idx];
            PetscKernel_v_gets_v_minus_A_times_w_5(s,v,xw);
            v  += 25;
          }
          t[i2] = s[0]; t[i2+1] = s[1]; t[i2+2] = s[2]; t[i2+3] = s[3]; t[i2+4] = s[4];
          PetscKernel_v_gets_A_times_w_5(xw,idiag,s);
          x[i2] = xw[0]; x[i2+1] = xw[1]; x[i2+2] = xw[2]; x[i2+3] = xw[3]; x[i2+4] = xw[4];
          idiag  += 25;
          i2     += 5;
        }
        break;
      case 6:
        PetscKernel_v_gets_A_times_w_6(x,idiag,b);
        t[0] = b[0]; t[1] = b[1]; t[2] = b[2]; t[3] = b[3]; t[4] = b[4]; t[5] = b[5];
        i2     = 6;
        idiag += 36;
        for (i=1; i<m; i++) {
          v  = aa + 36*ai[i];
          vi = aj + ai[i];
          nz = diag[i] - ai[i];
          s[0] = b[i2]; s[1] = b[i2+1]; s[2] = b[i2+2]; s[3] = b[i2+3]; s[4] = b[i2+4]; s[5] = b[i2+5];
          while (nz--) {
            idx = 6*(*vi++);
            xw[0] = x[idx];   xw[1] = x[1+idx]; xw[2] = x[2+idx];
            xw[3] = x[3+idx]; xw[4] = x[4+idx]; xw[5] = x[5+idx];
            PetscKernel_v_gets_v_minus_A_times_w_6(s,v,xw);
            v  += 36;
          }
          t[i2]   = s[0]; t[i2+1] = s[1]; t[i2+2] = s[2];
          t[i2+3] = s[3]; t[i2+4] = s[4]; t[i2+5] = s[5];
          PetscKernel_v_gets_A_times_w_6(xw,idiag,s);
          x[i2] = xw[0]; x[i2+1] = xw[1]; x[i2+2] = xw[2]; x[i2+3] = xw[3]; x[i2+4] = xw[4]; x[i2+5] = xw[5];
          idiag  += 36;
          i2     += 6;
        }
        break;
      case 7:
        PetscKernel_v_gets_A_times_w_7(x,idiag,b);
        t[0] = b[0]; t[1] = b[1]; t[2] = b[2];
        t[3] = b[3]; t[4] = b[4]; t[5] = b[5]; t[6] = b[6];
        i2     = 7;
        idiag += 49;
        for (i=1; i<m; i++) {
          v  = aa + 49*ai[i];
          vi = aj + ai[i];
          nz = diag[i] - ai[i];
          s[0] = b[i2];   s[1] = b[i2+1]; s[2] = b[i2+2];
          s[3] = b[i2+3]; s[4] = b[i2+4]; s[5] = b[i2+5]; s[6] = b[i2+6];
          while (nz--) {
            idx = 7*(*vi++);
            xw[0] = x[idx];   xw[1] = x[1+idx]; xw[2] = x[2+idx];
            xw[3] = x[3+idx]; xw[4] = x[4+idx]; xw[5] = x[5+idx]; xw[6] = x[6+idx];
            PetscKernel_v_gets_v_minus_A_times_w_7(s,v,xw);
            v  += 49;
          }
          t[i2]   = s[0]; t[i2+1] = s[1]; t[i2+2] = s[2];
          t[i2+3] = s[3]; t[i2+4] = s[4]; t[i2+5] = s[5]; t[i2+6] = s[6];
          PetscKernel_v_gets_A_times_w_7(xw,idiag,s);
          x[i2] =   xw[0]; x[i2+1] = xw[1]; x[i2+2] = xw[2];
          x[i2+3] = xw[3]; x[i2+4] = xw[4]; x[i2+5] = xw[5]; x[i2+6] = xw[6];
          idiag  += 49;
          i2     += 7;
        }
        break;
      default:
        PetscKernel_w_gets_Ar_times_v(bs,bs,b,idiag,x);
        PetscCall(PetscArraycpy(t,b,bs));
        i2     = bs;
        idiag += bs2;
        for (i=1; i<m; i++) {
          v  = aa + bs2*ai[i];
          vi = aj + ai[i];
          nz = diag[i] - ai[i];

          PetscCall(PetscArraycpy(w,b+i2,bs));
          /* copy all rows of x that are needed into contiguous space */
          workt = work;
          for (j=0; j<nz; j++) {
            PetscCall(PetscArraycpy(workt,x + bs*(*vi++),bs));
            workt += bs;
          }
          PetscKernel_w_gets_w_minus_Ar_times_v(bs,bs*nz,w,v,work);
          PetscCall(PetscArraycpy(t+i2,w,bs));
          PetscKernel_w_gets_Ar_times_v(bs,bs,w,idiag,x+i2);

          idiag += bs2;
          i2    += bs;
        }
        break;
      }
      /* for logging purposes assume number of nonzero in lower half is 1/2 of total */
      PetscCall(PetscLogFlops(1.0*bs2*a->nz));
      xb = t;
    }
    else xb = b;
    if (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP) {
      idiag = a->idiag+bs2*(a->mbs-1);
      i2 = bs * (m-1);
      switch (bs) {
      case 1:
        s[0]  = xb[i2];
        PetscKernel_v_gets_A_times_w_1(xw,idiag,s);
        x[i2] = xw[0];
        i2   -= 1;
        for (i=m-2; i>=0; i--) {
          v  = aa + (diag[i]+1);
          vi = aj + diag[i] + 1;
          nz = ai[i+1] - diag[i] - 1;
          s[0] = xb[i2];
          for (j=0; j<nz; j++) {
            xw[0] = x[vi[j]];
            PetscKernel_v_gets_v_minus_A_times_w_1(s,(v+j),xw);
          }
          PetscKernel_v_gets_A_times_w_1(xw,idiag,s);
          x[i2]  = xw[0];
          idiag -= 1;
          i2    -= 1;
        }
        break;
      case 2:
        s[0]  = xb[i2]; s[1] = xb[i2+1];
        PetscKernel_v_gets_A_times_w_2(xw,idiag,s);
        x[i2] = xw[0]; x[i2+1] = xw[1];
        i2    -= 2;
        idiag -= 4;
        for (i=m-2; i>=0; i--) {
          v  = aa + 4*(diag[i] + 1);
          vi = aj + diag[i] + 1;
          nz = ai[i+1] - diag[i] - 1;
          s[0] = xb[i2]; s[1] = xb[i2+1];
          for (j=0; j<nz; j++) {
            idx = 2*vi[j];
            it  = 4*j;
            xw[0] = x[idx]; xw[1] = x[1+idx];
            PetscKernel_v_gets_v_minus_A_times_w_2(s,(v+it),xw);
          }
          PetscKernel_v_gets_A_times_w_2(xw,idiag,s);
          x[i2]   = xw[0]; x[i2+1] = xw[1];
          idiag  -= 4;
          i2     -= 2;
        }
        break;
      case 3:
        s[0]  = xb[i2]; s[1] = xb[i2+1]; s[2] = xb[i2+2];
        PetscKernel_v_gets_A_times_w_3(xw,idiag,s);
        x[i2] = xw[0]; x[i2+1] = xw[1]; x[i2+2] = xw[2];
        i2    -= 3;
        idiag -= 9;
        for (i=m-2; i>=0; i--) {
          v  = aa + 9*(diag[i]+1);
          vi = aj + diag[i] + 1;
          nz = ai[i+1] - diag[i] - 1;
          s[0] = xb[i2]; s[1] = xb[i2+1]; s[2] = xb[i2+2];
          while (nz--) {
            idx = 3*(*vi++);
            xw[0] = x[idx]; xw[1] = x[1+idx]; xw[2] = x[2+idx];
            PetscKernel_v_gets_v_minus_A_times_w_3(s,v,xw);
            v  += 9;
          }
          PetscKernel_v_gets_A_times_w_3(xw,idiag,s);
          x[i2] = xw[0]; x[i2+1] = xw[1]; x[i2+2] = xw[2];
          idiag  -= 9;
          i2     -= 3;
        }
        break;
      case 4:
        s[0]  = xb[i2]; s[1] = xb[i2+1]; s[2] = xb[i2+2]; s[3] = xb[i2+3];
        PetscKernel_v_gets_A_times_w_4(xw,idiag,s);
        x[i2] = xw[0]; x[i2+1] = xw[1]; x[i2+2] = xw[2]; x[i2+3] = xw[3];
        i2    -= 4;
        idiag -= 16;
        for (i=m-2; i>=0; i--) {
          v  = aa + 16*(diag[i]+1);
          vi = aj + diag[i] + 1;
          nz = ai[i+1] - diag[i] - 1;
          s[0] = xb[i2]; s[1] = xb[i2+1]; s[2] = xb[i2+2]; s[3] = xb[i2+3];
          while (nz--) {
            idx = 4*(*vi++);
            xw[0]  = x[idx]; xw[1] = x[1+idx]; xw[2] = x[2+idx]; xw[3] = x[3+idx];
            PetscKernel_v_gets_v_minus_A_times_w_4(s,v,xw);
            v  += 16;
          }
          PetscKernel_v_gets_A_times_w_4(xw,idiag,s);
          x[i2] = xw[0]; x[i2+1] = xw[1]; x[i2+2] = xw[2]; x[i2+3] = xw[3];
          idiag  -= 16;
          i2     -= 4;
        }
        break;
      case 5:
        s[0]  = xb[i2]; s[1] = xb[i2+1]; s[2] = xb[i2+2]; s[3] = xb[i2+3]; s[4] = xb[i2+4];
        PetscKernel_v_gets_A_times_w_5(xw,idiag,s);
        x[i2] = xw[0]; x[i2+1] = xw[1]; x[i2+2] = xw[2]; x[i2+3] = xw[3]; x[i2+4] = xw[4];
        i2    -= 5;
        idiag -= 25;
        for (i=m-2; i>=0; i--) {
          v  = aa + 25*(diag[i]+1);
          vi = aj + diag[i] + 1;
          nz = ai[i+1] - diag[i] - 1;
          s[0] = xb[i2]; s[1] = xb[i2+1]; s[2] = xb[i2+2]; s[3] = xb[i2+3]; s[4] = xb[i2+4];
          while (nz--) {
            idx = 5*(*vi++);
            xw[0]  = x[idx]; xw[1] = x[1+idx]; xw[2] = x[2+idx]; xw[3] = x[3+idx]; xw[4] = x[4+idx];
            PetscKernel_v_gets_v_minus_A_times_w_5(s,v,xw);
            v  += 25;
          }
          PetscKernel_v_gets_A_times_w_5(xw,idiag,s);
          x[i2] = xw[0]; x[i2+1] = xw[1]; x[i2+2] = xw[2]; x[i2+3] = xw[3]; x[i2+4] = xw[4];
          idiag  -= 25;
          i2     -= 5;
        }
        break;
      case 6:
        s[0]  = xb[i2]; s[1] = xb[i2+1]; s[2] = xb[i2+2]; s[3] = xb[i2+3]; s[4] = xb[i2+4]; s[5] = xb[i2+5];
        PetscKernel_v_gets_A_times_w_6(xw,idiag,s);
        x[i2] = xw[0]; x[i2+1] = xw[1]; x[i2+2] = xw[2]; x[i2+3] = xw[3]; x[i2+4] = xw[4]; x[i2+5] = xw[5];
        i2    -= 6;
        idiag -= 36;
        for (i=m-2; i>=0; i--) {
          v  = aa + 36*(diag[i]+1);
          vi = aj + diag[i] + 1;
          nz = ai[i+1] - diag[i] - 1;
          s[0] = xb[i2]; s[1] = xb[i2+1]; s[2] = xb[i2+2]; s[3] = xb[i2+3]; s[4] = xb[i2+4]; s[5] = xb[i2+5];
          while (nz--) {
            idx = 6*(*vi++);
            xw[0] = x[idx];   xw[1] = x[1+idx]; xw[2] = x[2+idx];
            xw[3] = x[3+idx]; xw[4] = x[4+idx]; xw[5] = x[5+idx];
            PetscKernel_v_gets_v_minus_A_times_w_6(s,v,xw);
            v  += 36;
          }
          PetscKernel_v_gets_A_times_w_6(xw,idiag,s);
          x[i2] = xw[0]; x[i2+1] = xw[1]; x[i2+2] = xw[2]; x[i2+3] = xw[3]; x[i2+4] = xw[4]; x[i2+5] = xw[5];
          idiag  -= 36;
          i2     -= 6;
        }
        break;
      case 7:
        s[0] = xb[i2];   s[1] = xb[i2+1]; s[2] = xb[i2+2];
        s[3] = xb[i2+3]; s[4] = xb[i2+4]; s[5] = xb[i2+5]; s[6] = xb[i2+6];
        PetscKernel_v_gets_A_times_w_7(x,idiag,b);
        x[i2]   = xw[0]; x[i2+1] = xw[1]; x[i2+2] = xw[2];
        x[i2+3] = xw[3]; x[i2+4] = xw[4]; x[i2+5] = xw[5]; x[i2+6] = xw[6];
        i2    -= 7;
        idiag -= 49;
        for (i=m-2; i>=0; i--) {
          v  = aa + 49*(diag[i]+1);
          vi = aj + diag[i] + 1;
          nz = ai[i+1] - diag[i] - 1;
          s[0] = xb[i2];   s[1] = xb[i2+1]; s[2] = xb[i2+2];
          s[3] = xb[i2+3]; s[4] = xb[i2+4]; s[5] = xb[i2+5]; s[6] = xb[i2+6];
          while (nz--) {
            idx = 7*(*vi++);
            xw[0] = x[idx];   xw[1] = x[1+idx]; xw[2] = x[2+idx];
            xw[3] = x[3+idx]; xw[4] = x[4+idx]; xw[5] = x[5+idx]; xw[6] = x[6+idx];
            PetscKernel_v_gets_v_minus_A_times_w_7(s,v,xw);
            v  += 49;
          }
          PetscKernel_v_gets_A_times_w_7(xw,idiag,s);
          x[i2] =   xw[0]; x[i2+1] = xw[1]; x[i2+2] = xw[2];
          x[i2+3] = xw[3]; x[i2+4] = xw[4]; x[i2+5] = xw[5]; x[i2+6] = xw[6];
          idiag  -= 49;
          i2     -= 7;
        }
        break;
      default:
        PetscCall(PetscArraycpy(w,xb+i2,bs));
        PetscKernel_w_gets_Ar_times_v(bs,bs,w,idiag,x+i2);
        i2    -= bs;
        idiag -= bs2;
        for (i=m-2; i>=0; i--) {
          v  = aa + bs2*(diag[i]+1);
          vi = aj + diag[i] + 1;
          nz = ai[i+1] - diag[i] - 1;

          PetscCall(PetscArraycpy(w,xb+i2,bs));
          /* copy all rows of x that are needed into contiguous space */
          workt = work;
          for (j=0; j<nz; j++) {
            PetscCall(PetscArraycpy(workt,x + bs*(*vi++),bs));
            workt += bs;
          }
          PetscKernel_w_gets_w_minus_Ar_times_v(bs,bs*nz,w,v,work);
          PetscKernel_w_gets_Ar_times_v(bs,bs,w,idiag,x+i2);

          idiag -= bs2;
          i2    -= bs;
        }
        break;
      }
      PetscCall(PetscLogFlops(1.0*bs2*(a->nz)));
    }
    its--;
  }
  while (its--) {
    if (flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP) {
      idiag = a->idiag;
      i2 = 0;
      switch (bs) {
      case 1:
        for (i=0; i<m; i++) {
          v  = aa + ai[i];
          vi = aj + ai[i];
          nz = ai[i+1] - ai[i];
          s[0] = b[i2];
          for (j=0; j<nz; j++) {
            xw[0] = x[vi[j]];
            PetscKernel_v_gets_v_minus_A_times_w_1(s,(v+j),xw);
          }
          PetscKernel_v_gets_A_times_w_1(xw,idiag,s);
          x[i2] += xw[0];
          idiag += 1;
          i2    += 1;
        }
        break;
      case 2:
        for (i=0; i<m; i++) {
          v  = aa + 4*ai[i];
          vi = aj + ai[i];
          nz = ai[i+1] - ai[i];
          s[0] = b[i2]; s[1] = b[i2+1];
          for (j=0; j<nz; j++) {
            idx = 2*vi[j];
            it  = 4*j;
            xw[0] = x[idx]; xw[1] = x[1+idx];
            PetscKernel_v_gets_v_minus_A_times_w_2(s,(v+it),xw);
          }
          PetscKernel_v_gets_A_times_w_2(xw,idiag,s);
          x[i2]  += xw[0]; x[i2+1] += xw[1];
          idiag  += 4;
          i2     += 2;
        }
        break;
      case 3:
        for (i=0; i<m; i++) {
          v  = aa + 9*ai[i];
          vi = aj + ai[i];
          nz = ai[i+1] - ai[i];
          s[0] = b[i2]; s[1] = b[i2+1]; s[2] = b[i2+2];
          while (nz--) {
            idx = 3*(*vi++);
            xw[0] = x[idx]; xw[1] = x[1+idx]; xw[2] = x[2+idx];
            PetscKernel_v_gets_v_minus_A_times_w_3(s,v,xw);
            v  += 9;
          }
          PetscKernel_v_gets_A_times_w_3(xw,idiag,s);
          x[i2] += xw[0]; x[i2+1] += xw[1]; x[i2+2] += xw[2];
          idiag  += 9;
          i2     += 3;
        }
        break;
      case 4:
        for (i=0; i<m; i++) {
          v  = aa + 16*ai[i];
          vi = aj + ai[i];
          nz = ai[i+1] - ai[i];
          s[0] = b[i2]; s[1] = b[i2+1]; s[2] = b[i2+2]; s[3] = b[i2+3];
          while (nz--) {
            idx = 4*(*vi++);
            xw[0]  = x[idx]; xw[1] = x[1+idx]; xw[2] = x[2+idx]; xw[3] = x[3+idx];
            PetscKernel_v_gets_v_minus_A_times_w_4(s,v,xw);
            v  += 16;
          }
          PetscKernel_v_gets_A_times_w_4(xw,idiag,s);
          x[i2] += xw[0]; x[i2+1] += xw[1]; x[i2+2] += xw[2]; x[i2+3] += xw[3];
          idiag  += 16;
          i2     += 4;
        }
        break;
      case 5:
        for (i=0; i<m; i++) {
          v  = aa + 25*ai[i];
          vi = aj + ai[i];
          nz = ai[i+1] - ai[i];
          s[0] = b[i2]; s[1] = b[i2+1]; s[2] = b[i2+2]; s[3] = b[i2+3]; s[4] = b[i2+4];
          while (nz--) {
            idx = 5*(*vi++);
            xw[0]  = x[idx]; xw[1] = x[1+idx]; xw[2] = x[2+idx]; xw[3] = x[3+idx]; xw[4] = x[4+idx];
            PetscKernel_v_gets_v_minus_A_times_w_5(s,v,xw);
            v  += 25;
          }
          PetscKernel_v_gets_A_times_w_5(xw,idiag,s);
          x[i2] += xw[0]; x[i2+1] += xw[1]; x[i2+2] += xw[2]; x[i2+3] += xw[3]; x[i2+4] += xw[4];
          idiag  += 25;
          i2     += 5;
        }
        break;
      case 6:
        for (i=0; i<m; i++) {
          v  = aa + 36*ai[i];
          vi = aj + ai[i];
          nz = ai[i+1] - ai[i];
          s[0] = b[i2]; s[1] = b[i2+1]; s[2] = b[i2+2]; s[3] = b[i2+3]; s[4] = b[i2+4]; s[5] = b[i2+5];
          while (nz--) {
            idx = 6*(*vi++);
            xw[0] = x[idx];   xw[1] = x[1+idx]; xw[2] = x[2+idx];
            xw[3] = x[3+idx]; xw[4] = x[4+idx]; xw[5] = x[5+idx];
            PetscKernel_v_gets_v_minus_A_times_w_6(s,v,xw);
            v  += 36;
          }
          PetscKernel_v_gets_A_times_w_6(xw,idiag,s);
          x[i2] += xw[0]; x[i2+1] += xw[1]; x[i2+2] += xw[2];
          x[i2+3] += xw[3]; x[i2+4] += xw[4]; x[i2+5] += xw[5];
          idiag  += 36;
          i2     += 6;
        }
        break;
      case 7:
        for (i=0; i<m; i++) {
          v  = aa + 49*ai[i];
          vi = aj + ai[i];
          nz = ai[i+1] - ai[i];
          s[0] = b[i2];   s[1] = b[i2+1]; s[2] = b[i2+2];
          s[3] = b[i2+3]; s[4] = b[i2+4]; s[5] = b[i2+5]; s[6] = b[i2+6];
          while (nz--) {
            idx = 7*(*vi++);
            xw[0] = x[idx];   xw[1] = x[1+idx]; xw[2] = x[2+idx];
            xw[3] = x[3+idx]; xw[4] = x[4+idx]; xw[5] = x[5+idx]; xw[6] = x[6+idx];
            PetscKernel_v_gets_v_minus_A_times_w_7(s,v,xw);
            v  += 49;
          }
          PetscKernel_v_gets_A_times_w_7(xw,idiag,s);
          x[i2]   += xw[0]; x[i2+1] += xw[1]; x[i2+2] += xw[2];
          x[i2+3] += xw[3]; x[i2+4] += xw[4]; x[i2+5] += xw[5]; x[i2+6] += xw[6];
          idiag  += 49;
          i2     += 7;
        }
        break;
      default:
        for (i=0; i<m; i++) {
          v  = aa + bs2*ai[i];
          vi = aj + ai[i];
          nz = ai[i+1] - ai[i];

          PetscCall(PetscArraycpy(w,b+i2,bs));
          /* copy all rows of x that are needed into contiguous space */
          workt = work;
          for (j=0; j<nz; j++) {
            PetscCall(PetscArraycpy(workt,x + bs*(*vi++),bs));
            workt += bs;
          }
          PetscKernel_w_gets_w_minus_Ar_times_v(bs,bs*nz,w,v,work);
          PetscKernel_w_gets_w_plus_Ar_times_v(bs,bs,w,idiag,x+i2);

          idiag += bs2;
          i2    += bs;
        }
        break;
      }
      PetscCall(PetscLogFlops(2.0*bs2*a->nz));
    }
    if (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP) {
      idiag = a->idiag+bs2*(a->mbs-1);
      i2 = bs * (m-1);
      switch (bs) {
      case 1:
        for (i=m-1; i>=0; i--) {
          v  = aa + ai[i];
          vi = aj + ai[i];
          nz = ai[i+1] - ai[i];
          s[0] = b[i2];
          for (j=0; j<nz; j++) {
            xw[0] = x[vi[j]];
            PetscKernel_v_gets_v_minus_A_times_w_1(s,(v+j),xw);
          }
          PetscKernel_v_gets_A_times_w_1(xw,idiag,s);
          x[i2] += xw[0];
          idiag -= 1;
          i2    -= 1;
        }
        break;
      case 2:
        for (i=m-1; i>=0; i--) {
          v  = aa + 4*ai[i];
          vi = aj + ai[i];
          nz = ai[i+1] - ai[i];
          s[0] = b[i2]; s[1] = b[i2+1];
          for (j=0; j<nz; j++) {
            idx = 2*vi[j];
            it  = 4*j;
            xw[0] = x[idx]; xw[1] = x[1+idx];
            PetscKernel_v_gets_v_minus_A_times_w_2(s,(v+it),xw);
          }
          PetscKernel_v_gets_A_times_w_2(xw,idiag,s);
          x[i2]  += xw[0]; x[i2+1] += xw[1];
          idiag  -= 4;
          i2     -= 2;
        }
        break;
      case 3:
        for (i=m-1; i>=0; i--) {
          v  = aa + 9*ai[i];
          vi = aj + ai[i];
          nz = ai[i+1] - ai[i];
          s[0] = b[i2]; s[1] = b[i2+1]; s[2] = b[i2+2];
          while (nz--) {
            idx = 3*(*vi++);
            xw[0] = x[idx]; xw[1] = x[1+idx]; xw[2] = x[2+idx];
            PetscKernel_v_gets_v_minus_A_times_w_3(s,v,xw);
            v  += 9;
          }
          PetscKernel_v_gets_A_times_w_3(xw,idiag,s);
          x[i2] += xw[0]; x[i2+1] += xw[1]; x[i2+2] += xw[2];
          idiag  -= 9;
          i2     -= 3;
        }
        break;
      case 4:
        for (i=m-1; i>=0; i--) {
          v  = aa + 16*ai[i];
          vi = aj + ai[i];
          nz = ai[i+1] - ai[i];
          s[0] = b[i2]; s[1] = b[i2+1]; s[2] = b[i2+2]; s[3] = b[i2+3];
          while (nz--) {
            idx = 4*(*vi++);
            xw[0]  = x[idx]; xw[1] = x[1+idx]; xw[2] = x[2+idx]; xw[3] = x[3+idx];
            PetscKernel_v_gets_v_minus_A_times_w_4(s,v,xw);
            v  += 16;
          }
          PetscKernel_v_gets_A_times_w_4(xw,idiag,s);
          x[i2] += xw[0]; x[i2+1] += xw[1]; x[i2+2] += xw[2]; x[i2+3] += xw[3];
          idiag  -= 16;
          i2     -= 4;
        }
        break;
      case 5:
        for (i=m-1; i>=0; i--) {
          v  = aa + 25*ai[i];
          vi = aj + ai[i];
          nz = ai[i+1] - ai[i];
          s[0] = b[i2]; s[1] = b[i2+1]; s[2] = b[i2+2]; s[3] = b[i2+3]; s[4] = b[i2+4];
          while (nz--) {
            idx = 5*(*vi++);
            xw[0]  = x[idx]; xw[1] = x[1+idx]; xw[2] = x[2+idx]; xw[3] = x[3+idx]; xw[4] = x[4+idx];
            PetscKernel_v_gets_v_minus_A_times_w_5(s,v,xw);
            v  += 25;
          }
          PetscKernel_v_gets_A_times_w_5(xw,idiag,s);
          x[i2] += xw[0]; x[i2+1] += xw[1]; x[i2+2] += xw[2]; x[i2+3] += xw[3]; x[i2+4] += xw[4];
          idiag  -= 25;
          i2     -= 5;
        }
        break;
      case 6:
        for (i=m-1; i>=0; i--) {
          v  = aa + 36*ai[i];
          vi = aj + ai[i];
          nz = ai[i+1] - ai[i];
          s[0] = b[i2]; s[1] = b[i2+1]; s[2] = b[i2+2]; s[3] = b[i2+3]; s[4] = b[i2+4]; s[5] = b[i2+5];
          while (nz--) {
            idx = 6*(*vi++);
            xw[0] = x[idx];   xw[1] = x[1+idx]; xw[2] = x[2+idx];
            xw[3] = x[3+idx]; xw[4] = x[4+idx]; xw[5] = x[5+idx];
            PetscKernel_v_gets_v_minus_A_times_w_6(s,v,xw);
            v  += 36;
          }
          PetscKernel_v_gets_A_times_w_6(xw,idiag,s);
          x[i2] += xw[0]; x[i2+1] += xw[1]; x[i2+2] += xw[2];
          x[i2+3] += xw[3]; x[i2+4] += xw[4]; x[i2+5] += xw[5];
          idiag  -= 36;
          i2     -= 6;
        }
        break;
      case 7:
        for (i=m-1; i>=0; i--) {
          v  = aa + 49*ai[i];
          vi = aj + ai[i];
          nz = ai[i+1] - ai[i];
          s[0] = b[i2];   s[1] = b[i2+1]; s[2] = b[i2+2];
          s[3] = b[i2+3]; s[4] = b[i2+4]; s[5] = b[i2+5]; s[6] = b[i2+6];
          while (nz--) {
            idx = 7*(*vi++);
            xw[0] = x[idx];   xw[1] = x[1+idx]; xw[2] = x[2+idx];
            xw[3] = x[3+idx]; xw[4] = x[4+idx]; xw[5] = x[5+idx]; xw[6] = x[6+idx];
            PetscKernel_v_gets_v_minus_A_times_w_7(s,v,xw);
            v  += 49;
          }
          PetscKernel_v_gets_A_times_w_7(xw,idiag,s);
          x[i2] +=   xw[0]; x[i2+1] += xw[1]; x[i2+2] += xw[2];
          x[i2+3] += xw[3]; x[i2+4] += xw[4]; x[i2+5] += xw[5]; x[i2+6] += xw[6];
          idiag  -= 49;
          i2     -= 7;
        }
        break;
      default:
        for (i=m-1; i>=0; i--) {
          v  = aa + bs2*ai[i];
          vi = aj + ai[i];
          nz = ai[i+1] - ai[i];

          PetscCall(PetscArraycpy(w,b+i2,bs));
          /* copy all rows of x that are needed into contiguous space */
          workt = work;
          for (j=0; j<nz; j++) {
            PetscCall(PetscArraycpy(workt,x + bs*(*vi++),bs));
            workt += bs;
          }
          PetscKernel_w_gets_w_minus_Ar_times_v(bs,bs*nz,w,v,work);
          PetscKernel_w_gets_w_plus_Ar_times_v(bs,bs,w,idiag,x+i2);

          idiag -= bs2;
          i2    -= bs;
        }
        break;
      }
      PetscCall(PetscLogFlops(2.0*bs2*(a->nz)));
    }
  }
  PetscCall(VecRestoreArray(xx,&x));
  PetscCall(VecRestoreArrayRead(bb,&b));
  PetscFunctionReturn(0);
}

/*
    Special version for direct calls from Fortran (Used in PETSc-fun3d)
*/
#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define matsetvaluesblocked4_ MATSETVALUESBLOCKED4
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define matsetvaluesblocked4_ matsetvaluesblocked4
#endif

PETSC_EXTERN void matsetvaluesblocked4_(Mat *AA,PetscInt *mm,const PetscInt im[],PetscInt *nn,const PetscInt in[],const PetscScalar v[])
{
  Mat               A  = *AA;
  Mat_SeqBAIJ       *a = (Mat_SeqBAIJ*)A->data;
  PetscInt          *rp,k,low,high,t,ii,jj,row,nrow,i,col,l,N,m = *mm,n = *nn;
  PetscInt          *ai    =a->i,*ailen=a->ilen;
  PetscInt          *aj    =a->j,stepval,lastcol = -1;
  const PetscScalar *value = v;
  MatScalar         *ap,*aa = a->a,*bap;

  PetscFunctionBegin;
  if (A->rmap->bs != 4) SETERRABORT(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"Can only be called with a block size of 4");
  stepval = (n-1)*4;
  for (k=0; k<m; k++) { /* loop over added rows */
    row  = im[k];
    rp   = aj + ai[row];
    ap   = aa + 16*ai[row];
    nrow = ailen[row];
    low  = 0;
    high = nrow;
    for (l=0; l<n; l++) { /* loop over added columns */
      col = in[l];
      if (col <= lastcol)  low = 0;
      else                high = nrow;
      lastcol = col;
      value   = v + k*(stepval+4 + l)*4;
      while (high-low > 7) {
        t = (low+high)/2;
        if (rp[t] > col) high = t;
        else             low  = t;
      }
      for (i=low; i<high; i++) {
        if (rp[i] > col) break;
        if (rp[i] == col) {
          bap = ap +  16*i;
          for (ii=0; ii<4; ii++,value+=stepval) {
            for (jj=ii; jj<16; jj+=4) {
              bap[jj] += *value++;
            }
          }
          goto noinsert2;
        }
      }
      N = nrow++ - 1;
      high++; /* added new column index thus must search to one higher than before */
      /* shift up all the later entries in this row */
      for (ii=N; ii>=i; ii--) {
        rp[ii+1] = rp[ii];
        PetscCallVoid(PetscArraycpy(ap+16*(ii+1),ap+16*(ii),16));
      }
      if (N >= i) {
        PetscCallVoid(PetscArrayzero(ap+16*i,16));
      }
      rp[i] = col;
      bap   = ap +  16*i;
      for (ii=0; ii<4; ii++,value+=stepval) {
        for (jj=ii; jj<16; jj+=4) {
          bap[jj] = *value++;
        }
      }
      noinsert2:;
      low = i;
    }
    ailen[row] = nrow;
  }
  PetscFunctionReturnVoid();
}

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define matsetvalues4_ MATSETVALUES4
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define matsetvalues4_ matsetvalues4
#endif

PETSC_EXTERN void matsetvalues4_(Mat *AA,PetscInt *mm,PetscInt *im,PetscInt *nn,PetscInt *in,PetscScalar *v)
{
  Mat         A  = *AA;
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ*)A->data;
  PetscInt    *rp,k,low,high,t,row,nrow,i,col,l,N,n = *nn,m = *mm;
  PetscInt    *ai=a->i,*ailen=a->ilen;
  PetscInt    *aj=a->j,brow,bcol;
  PetscInt    ridx,cidx,lastcol = -1;
  MatScalar   *ap,value,*aa=a->a,*bap;

  PetscFunctionBegin;
  for (k=0; k<m; k++) { /* loop over added rows */
    row  = im[k]; brow = row/4;
    rp   = aj + ai[brow];
    ap   = aa + 16*ai[brow];
    nrow = ailen[brow];
    low  = 0;
    high = nrow;
    for (l=0; l<n; l++) { /* loop over added columns */
      col   = in[l]; bcol = col/4;
      ridx  = row % 4; cidx = col % 4;
      value = v[l + k*n];
      if (col <= lastcol)  low = 0;
      else                high = nrow;
      lastcol = col;
      while (high-low > 7) {
        t = (low+high)/2;
        if (rp[t] > bcol) high = t;
        else              low  = t;
      }
      for (i=low; i<high; i++) {
        if (rp[i] > bcol) break;
        if (rp[i] == bcol) {
          bap   = ap +  16*i + 4*cidx + ridx;
          *bap += value;
          goto noinsert1;
        }
      }
      N = nrow++ - 1;
      high++; /* added new column thus must search to one higher than before */
      /* shift up all the later entries in this row */
      PetscCallVoid(PetscArraymove(rp+i+1,rp+i,N-i+1));
      PetscCallVoid(PetscArraymove(ap+16*i+16,ap+16*i,16*(N-i+1)));
      PetscCallVoid(PetscArrayzero(ap+16*i,16));
      rp[i]                    = bcol;
      ap[16*i + 4*cidx + ridx] = value;
noinsert1:;
      low = i;
    }
    ailen[brow] = nrow;
  }
  PetscFunctionReturnVoid();
}

/*
     Checks for missing diagonals
*/
PetscErrorCode MatMissingDiagonal_SeqBAIJ(Mat A,PetscBool  *missing,PetscInt *d)
{
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data;
  PetscInt       *diag,*ii = a->i,i;

  PetscFunctionBegin;
  PetscCall(MatMarkDiagonal_SeqBAIJ(A));
  *missing = PETSC_FALSE;
  if (A->rmap->n > 0 && !ii) {
    *missing = PETSC_TRUE;
    if (d) *d = 0;
    PetscCall(PetscInfo(A,"Matrix has no entries therefore is missing diagonal\n"));
  } else {
    PetscInt n;
    n = PetscMin(a->mbs, a->nbs);
    diag = a->diag;
    for (i=0; i<n; i++) {
      if (diag[i] >= ii[i+1]) {
        *missing = PETSC_TRUE;
        if (d) *d = i;
        PetscCall(PetscInfo(A,"Matrix is missing block diagonal number %" PetscInt_FMT "\n",i));
        break;
      }
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatMarkDiagonal_SeqBAIJ(Mat A)
{
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data;
  PetscInt       i,j,m = a->mbs;

  PetscFunctionBegin;
  if (!a->diag) {
    PetscCall(PetscMalloc1(m,&a->diag));
    PetscCall(PetscLogObjectMemory((PetscObject)A,m*sizeof(PetscInt)));
    a->free_diag = PETSC_TRUE;
  }
  for (i=0; i<m; i++) {
    a->diag[i] = a->i[i+1];
    for (j=a->i[i]; j<a->i[i+1]; j++) {
      if (a->j[j] == i) {
        a->diag[i] = j;
        break;
      }
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetRowIJ_SeqBAIJ(Mat A,PetscInt oshift,PetscBool symmetric,PetscBool blockcompressed,PetscInt *nn,const PetscInt *inia[],const PetscInt *inja[],PetscBool  *done)
{
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data;
  PetscInt       i,j,n = a->mbs,nz = a->i[n],*tia,*tja,bs = A->rmap->bs,k,l,cnt;
  PetscInt       **ia = (PetscInt**)inia,**ja = (PetscInt**)inja;

  PetscFunctionBegin;
  *nn = n;
  if (!ia) PetscFunctionReturn(0);
  if (symmetric) {
    PetscCall(MatToSymmetricIJ_SeqAIJ(n,a->i,a->j,PETSC_TRUE,0,0,&tia,&tja));
    nz   = tia[n];
  } else {
    tia = a->i; tja = a->j;
  }

  if (!blockcompressed && bs > 1) {
    (*nn) *= bs;
    /* malloc & create the natural set of indices */
    PetscCall(PetscMalloc1((n+1)*bs,ia));
    if (n) {
      (*ia)[0] = oshift;
      for (j=1; j<bs; j++) {
        (*ia)[j] = (tia[1]-tia[0])*bs+(*ia)[j-1];
      }
    }

    for (i=1; i<n; i++) {
      (*ia)[i*bs] = (tia[i]-tia[i-1])*bs + (*ia)[i*bs-1];
      for (j=1; j<bs; j++) {
        (*ia)[i*bs+j] = (tia[i+1]-tia[i])*bs + (*ia)[i*bs+j-1];
      }
    }
    if (n) {
      (*ia)[n*bs] = (tia[n]-tia[n-1])*bs + (*ia)[n*bs-1];
    }

    if (inja) {
      PetscCall(PetscMalloc1(nz*bs*bs,ja));
      cnt = 0;
      for (i=0; i<n; i++) {
        for (j=0; j<bs; j++) {
          for (k=tia[i]; k<tia[i+1]; k++) {
            for (l=0; l<bs; l++) {
              (*ja)[cnt++] = bs*tja[k] + l;
            }
          }
        }
      }
    }

    if (symmetric) { /* deallocate memory allocated in MatToSymmetricIJ_SeqAIJ() */
      PetscCall(PetscFree(tia));
      PetscCall(PetscFree(tja));
    }
  } else if (oshift == 1) {
    if (symmetric) {
      nz = tia[A->rmap->n/bs];
      /*  add 1 to i and j indices */
      for (i=0; i<A->rmap->n/bs+1; i++) tia[i] = tia[i] + 1;
      *ia = tia;
      if (ja) {
        for (i=0; i<nz; i++) tja[i] = tja[i] + 1;
        *ja = tja;
      }
    } else {
      nz = a->i[A->rmap->n/bs];
      /* malloc space and  add 1 to i and j indices */
      PetscCall(PetscMalloc1(A->rmap->n/bs+1,ia));
      for (i=0; i<A->rmap->n/bs+1; i++) (*ia)[i] = a->i[i] + 1;
      if (ja) {
        PetscCall(PetscMalloc1(nz,ja));
        for (i=0; i<nz; i++) (*ja)[i] = a->j[i] + 1;
      }
    }
  } else {
    *ia = tia;
    if (ja) *ja = tja;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatRestoreRowIJ_SeqBAIJ(Mat A,PetscInt oshift,PetscBool symmetric,PetscBool blockcompressed,PetscInt *nn,const PetscInt *ia[],const PetscInt *ja[],PetscBool  *done)
{
  PetscFunctionBegin;
  if (!ia) PetscFunctionReturn(0);
  if ((!blockcompressed && A->rmap->bs > 1) || (symmetric || oshift == 1)) {
    PetscCall(PetscFree(*ia));
    if (ja) PetscCall(PetscFree(*ja));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_SeqBAIJ(Mat A)
{
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data;

  PetscFunctionBegin;
#if defined(PETSC_USE_LOG)
  PetscLogObjectState((PetscObject)A,"Rows=%" PetscInt_FMT ", Cols=%" PetscInt_FMT ", NZ=%" PetscInt_FMT,A->rmap->N,A->cmap->n,a->nz);
#endif
  PetscCall(MatSeqXAIJFreeAIJ(A,&a->a,&a->j,&a->i));
  PetscCall(ISDestroy(&a->row));
  PetscCall(ISDestroy(&a->col));
  if (a->free_diag) PetscCall(PetscFree(a->diag));
  PetscCall(PetscFree(a->idiag));
  if (a->free_imax_ilen) PetscCall(PetscFree2(a->imax,a->ilen));
  PetscCall(PetscFree(a->solve_work));
  PetscCall(PetscFree(a->mult_work));
  PetscCall(PetscFree(a->sor_workt));
  PetscCall(PetscFree(a->sor_work));
  PetscCall(ISDestroy(&a->icol));
  PetscCall(PetscFree(a->saved_values));
  PetscCall(PetscFree2(a->compressedrow.i,a->compressedrow.rindex));

  PetscCall(MatDestroy(&a->sbaijMat));
  PetscCall(MatDestroy(&a->parent));
  PetscCall(PetscFree(A->data));

  PetscCall(PetscObjectChangeTypeName((PetscObject)A,NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatSeqBAIJGetArray_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatSeqBAIJRestoreArray_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatStoreValues_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatRetrieveValues_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatSeqBAIJSetColumnIndices_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatConvert_seqbaij_seqaij_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatConvert_seqbaij_seqsbaij_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatSeqBAIJSetPreallocation_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatSeqBAIJSetPreallocationCSR_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatConvert_seqbaij_seqbstrm_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatIsTranspose_C",NULL));
#if defined(PETSC_HAVE_HYPRE)
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatConvert_seqbaij_hypre_C",NULL));
#endif
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatConvert_seqbaij_is_C",NULL));
  PetscFunctionReturn(0);
}

PetscErrorCode MatSetOption_SeqBAIJ(Mat A,MatOption op,PetscBool flg)
{
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data;

  PetscFunctionBegin;
  switch (op) {
  case MAT_ROW_ORIENTED:
    a->roworiented = flg;
    break;
  case MAT_KEEP_NONZERO_PATTERN:
    a->keepnonzeropattern = flg;
    break;
  case MAT_NEW_NONZERO_LOCATIONS:
    a->nonew = (flg ? 0 : 1);
    break;
  case MAT_NEW_NONZERO_LOCATION_ERR:
    a->nonew = (flg ? -1 : 0);
    break;
  case MAT_NEW_NONZERO_ALLOCATION_ERR:
    a->nonew = (flg ? -2 : 0);
    break;
  case MAT_UNUSED_NONZERO_LOCATION_ERR:
    a->nounused = (flg ? -1 : 0);
    break;
  case MAT_FORCE_DIAGONAL_ENTRIES:
  case MAT_IGNORE_OFF_PROC_ENTRIES:
  case MAT_USE_HASH_TABLE:
  case MAT_SORTED_FULL:
    PetscCall(PetscInfo(A,"Option %s ignored\n",MatOptions[op]));
    break;
  case MAT_SPD:
  case MAT_SYMMETRIC:
  case MAT_STRUCTURALLY_SYMMETRIC:
  case MAT_HERMITIAN:
  case MAT_SYMMETRY_ETERNAL:
  case MAT_SUBMAT_SINGLEIS:
  case MAT_STRUCTURE_ONLY:
    /* These options are handled directly by MatSetOption() */
    break;
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"unknown option %d",op);
  }
  PetscFunctionReturn(0);
}

/* used for both SeqBAIJ and SeqSBAIJ matrices */
PetscErrorCode MatGetRow_SeqBAIJ_private(Mat A,PetscInt row,PetscInt *nz,PetscInt **idx,PetscScalar **v,PetscInt *ai,PetscInt *aj,PetscScalar *aa)
{
  PetscInt       itmp,i,j,k,M,bn,bp,*idx_i,bs,bs2;
  MatScalar      *aa_i;
  PetscScalar    *v_i;

  PetscFunctionBegin;
  bs  = A->rmap->bs;
  bs2 = bs*bs;
  PetscCheck(row >= 0 && row < A->rmap->N,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row %" PetscInt_FMT " out of range", row);

  bn  = row/bs;   /* Block number */
  bp  = row % bs; /* Block Position */
  M   = ai[bn+1] - ai[bn];
  *nz = bs*M;

  if (v) {
    *v = NULL;
    if (*nz) {
      PetscCall(PetscMalloc1(*nz,v));
      for (i=0; i<M; i++) { /* for each block in the block row */
        v_i  = *v + i*bs;
        aa_i = aa + bs2*(ai[bn] + i);
        for (j=bp,k=0; j<bs2; j+=bs,k++) v_i[k] = aa_i[j];
      }
    }
  }

  if (idx) {
    *idx = NULL;
    if (*nz) {
      PetscCall(PetscMalloc1(*nz,idx));
      for (i=0; i<M; i++) { /* for each block in the block row */
        idx_i = *idx + i*bs;
        itmp  = bs*aj[ai[bn] + i];
        for (j=0; j<bs; j++) idx_i[j] = itmp++;
      }
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatGetRow_SeqBAIJ(Mat A,PetscInt row,PetscInt *nz,PetscInt **idx,PetscScalar **v)
{
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data;

  PetscFunctionBegin;
  PetscCall(MatGetRow_SeqBAIJ_private(A,row,nz,idx,v,a->i,a->j,a->a));
  PetscFunctionReturn(0);
}

PetscErrorCode MatRestoreRow_SeqBAIJ(Mat A,PetscInt row,PetscInt *nz,PetscInt **idx,PetscScalar **v)
{
  PetscFunctionBegin;
  if (nz)  *nz = 0;
  if (idx) PetscCall(PetscFree(*idx));
  if (v)   PetscCall(PetscFree(*v));
  PetscFunctionReturn(0);
}

PetscErrorCode MatTranspose_SeqBAIJ(Mat A,MatReuse reuse,Mat *B)
{
  Mat_SeqBAIJ    *a=(Mat_SeqBAIJ*)A->data,*at;
  Mat            C;
  PetscInt       i,j,k,*aj=a->j,*ai=a->i,bs=A->rmap->bs,mbs=a->mbs,nbs=a->nbs,*atfill;
  PetscInt       bs2=a->bs2,*ati,*atj,anzj,kr;
  MatScalar      *ata,*aa=a->a;

  PetscFunctionBegin;
  PetscCall(PetscCalloc1(1+nbs,&atfill));
  if (reuse == MAT_INITIAL_MATRIX || reuse == MAT_INPLACE_MATRIX) {
    for (i=0; i<ai[mbs]; i++) atfill[aj[i]] += 1; /* count num of non-zeros in row aj[i] */

    PetscCall(MatCreate(PetscObjectComm((PetscObject)A),&C));
    PetscCall(MatSetSizes(C,A->cmap->n,A->rmap->N,A->cmap->n,A->rmap->N));
    PetscCall(MatSetType(C,((PetscObject)A)->type_name));
    PetscCall(MatSeqBAIJSetPreallocation(C,bs,0,atfill));

    at  = (Mat_SeqBAIJ*)C->data;
    ati = at->i;
    for (i=0; i<nbs; i++) at->ilen[i] = at->imax[i] = ati[i+1] - ati[i];
  } else {
    C = *B;
    at = (Mat_SeqBAIJ*)C->data;
    ati = at->i;
  }

  atj = at->j;
  ata = at->a;

  /* Copy ati into atfill so we have locations of the next free space in atj */
  PetscCall(PetscArraycpy(atfill,ati,nbs));

  /* Walk through A row-wise and mark nonzero entries of A^T. */
  for (i=0; i<mbs; i++) {
    anzj = ai[i+1] - ai[i];
    for (j=0; j<anzj; j++) {
      atj[atfill[*aj]] = i;
      for (kr=0; kr<bs; kr++) {
        for (k=0; k<bs; k++) {
          ata[bs2*atfill[*aj]+k*bs+kr] = *aa++;
        }
      }
      atfill[*aj++] += 1;
    }
  }
  PetscCall(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));

  /* Clean up temporary space and complete requests. */
  PetscCall(PetscFree(atfill));

  if (reuse == MAT_INITIAL_MATRIX || reuse == MAT_REUSE_MATRIX) {
    PetscCall(MatSetBlockSizes(C,PetscAbs(A->cmap->bs),PetscAbs(A->rmap->bs)));
    *B = C;
  } else {
    PetscCall(MatHeaderMerge(A,&C));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatIsTranspose_SeqBAIJ(Mat A,Mat B,PetscReal tol,PetscBool  *f)
{
  Mat            Btrans;

  PetscFunctionBegin;
  *f   = PETSC_FALSE;
  PetscCall(MatTranspose_SeqBAIJ(A,MAT_INITIAL_MATRIX,&Btrans));
  PetscCall(MatEqual_SeqBAIJ(B,Btrans,f));
  PetscCall(MatDestroy(&Btrans));
  PetscFunctionReturn(0);
}

/* Used for both SeqBAIJ and SeqSBAIJ matrices */
PetscErrorCode MatView_SeqBAIJ_Binary(Mat mat,PetscViewer viewer)
{
  Mat_SeqBAIJ    *A = (Mat_SeqBAIJ*)mat->data;
  PetscInt       header[4],M,N,m,bs,nz,cnt,i,j,k,l;
  PetscInt       *rowlens,*colidxs;
  PetscScalar    *matvals;

  PetscFunctionBegin;
  PetscCall(PetscViewerSetUp(viewer));

  M  = mat->rmap->N;
  N  = mat->cmap->N;
  m  = mat->rmap->n;
  bs = mat->rmap->bs;
  nz = bs*bs*A->nz;

  /* write matrix header */
  header[0] = MAT_FILE_CLASSID;
  header[1] = M; header[2] = N; header[3] = nz;
  PetscCall(PetscViewerBinaryWrite(viewer,header,4,PETSC_INT));

  /* store row lengths */
  PetscCall(PetscMalloc1(m,&rowlens));
  for (cnt=0, i=0; i<A->mbs; i++)
    for (j=0; j<bs; j++)
      rowlens[cnt++] = bs*(A->i[i+1] - A->i[i]);
  PetscCall(PetscViewerBinaryWrite(viewer,rowlens,m,PETSC_INT));
  PetscCall(PetscFree(rowlens));

  /* store column indices  */
  PetscCall(PetscMalloc1(nz,&colidxs));
  for (cnt=0, i=0; i<A->mbs; i++)
    for (k=0; k<bs; k++)
      for (j=A->i[i]; j<A->i[i+1]; j++)
        for (l=0; l<bs; l++)
          colidxs[cnt++] = bs*A->j[j] + l;
  PetscCheck(cnt == nz,PETSC_COMM_SELF,PETSC_ERR_LIB,"Internal PETSc error: cnt = %" PetscInt_FMT " nz = %" PetscInt_FMT,cnt,nz);
  PetscCall(PetscViewerBinaryWrite(viewer,colidxs,nz,PETSC_INT));
  PetscCall(PetscFree(colidxs));

  /* store nonzero values */
  PetscCall(PetscMalloc1(nz,&matvals));
  for (cnt=0, i=0; i<A->mbs; i++)
    for (k=0; k<bs; k++)
      for (j=A->i[i]; j<A->i[i+1]; j++)
        for (l=0; l<bs; l++)
          matvals[cnt++] = A->a[bs*(bs*j + l) + k];
  PetscCheck(cnt == nz,PETSC_COMM_SELF,PETSC_ERR_LIB,"Internal PETSc error: cnt = %" PetscInt_FMT " nz = %" PetscInt_FMT,cnt,nz);
  PetscCall(PetscViewerBinaryWrite(viewer,matvals,nz,PETSC_SCALAR));
  PetscCall(PetscFree(matvals));

  /* write block size option to the viewer's .info file */
  PetscCall(MatView_Binary_BlockSizes(mat,viewer));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatView_SeqBAIJ_ASCII_structonly(Mat A,PetscViewer viewer)
{
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data;
  PetscInt       i,bs = A->rmap->bs,k;

  PetscFunctionBegin;
  PetscCall(PetscViewerASCIIUseTabs(viewer,PETSC_FALSE));
  for (i=0; i<a->mbs; i++) {
    PetscCall(PetscViewerASCIIPrintf(viewer,"row %" PetscInt_FMT "-%" PetscInt_FMT ":",i*bs,i*bs+bs-1));
    for (k=a->i[i]; k<a->i[i+1]; k++) {
      PetscCall(PetscViewerASCIIPrintf(viewer," (%" PetscInt_FMT "-%" PetscInt_FMT ") ",bs*a->j[k],bs*a->j[k]+bs-1));
    }
    PetscCall(PetscViewerASCIIPrintf(viewer,"\n"));
  }
  PetscCall(PetscViewerASCIIUseTabs(viewer,PETSC_TRUE));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatView_SeqBAIJ_ASCII(Mat A,PetscViewer viewer)
{
  Mat_SeqBAIJ       *a = (Mat_SeqBAIJ*)A->data;
  PetscInt          i,j,bs = A->rmap->bs,k,l,bs2=a->bs2;
  PetscViewerFormat format;

  PetscFunctionBegin;
  if (A->structure_only) {
    PetscCall(MatView_SeqBAIJ_ASCII_structonly(A,viewer));
    PetscFunctionReturn(0);
  }

  PetscCall(PetscViewerGetFormat(viewer,&format));
  if (format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
    PetscCall(PetscViewerASCIIPrintf(viewer,"  block size is %" PetscInt_FMT "\n",bs));
  } else if (format == PETSC_VIEWER_ASCII_MATLAB) {
    const char *matname;
    Mat        aij;
    PetscCall(MatConvert(A,MATSEQAIJ,MAT_INITIAL_MATRIX,&aij));
    PetscCall(PetscObjectGetName((PetscObject)A,&matname));
    PetscCall(PetscObjectSetName((PetscObject)aij,matname));
    PetscCall(MatView(aij,viewer));
    PetscCall(MatDestroy(&aij));
  } else if (format == PETSC_VIEWER_ASCII_FACTOR_INFO) {
      PetscFunctionReturn(0);
  } else if (format == PETSC_VIEWER_ASCII_COMMON) {
    PetscCall(PetscViewerASCIIUseTabs(viewer,PETSC_FALSE));
    for (i=0; i<a->mbs; i++) {
      for (j=0; j<bs; j++) {
        PetscCall(PetscViewerASCIIPrintf(viewer,"row %" PetscInt_FMT ":",i*bs+j));
        for (k=a->i[i]; k<a->i[i+1]; k++) {
          for (l=0; l<bs; l++) {
#if defined(PETSC_USE_COMPLEX)
            if (PetscImaginaryPart(a->a[bs2*k + l*bs + j]) > 0.0 && PetscRealPart(a->a[bs2*k + l*bs + j]) != 0.0) {
              PetscCall(PetscViewerASCIIPrintf(viewer," (%" PetscInt_FMT ", %g + %gi) ",bs*a->j[k]+l,
                                             (double)PetscRealPart(a->a[bs2*k + l*bs + j]),(double)PetscImaginaryPart(a->a[bs2*k + l*bs + j])));
            } else if (PetscImaginaryPart(a->a[bs2*k + l*bs + j]) < 0.0 && PetscRealPart(a->a[bs2*k + l*bs + j]) != 0.0) {
              PetscCall(PetscViewerASCIIPrintf(viewer," (%" PetscInt_FMT ", %g - %gi) ",bs*a->j[k]+l,
                                             (double)PetscRealPart(a->a[bs2*k + l*bs + j]),-(double)PetscImaginaryPart(a->a[bs2*k + l*bs + j])));
            } else if (PetscRealPart(a->a[bs2*k + l*bs + j]) != 0.0) {
              PetscCall(PetscViewerASCIIPrintf(viewer," (%" PetscInt_FMT ", %g) ",bs*a->j[k]+l,(double)PetscRealPart(a->a[bs2*k + l*bs + j])));
            }
#else
            if (a->a[bs2*k + l*bs + j] != 0.0) {
              PetscCall(PetscViewerASCIIPrintf(viewer," (%" PetscInt_FMT ", %g) ",bs*a->j[k]+l,(double)a->a[bs2*k + l*bs + j]));
            }
#endif
          }
        }
        PetscCall(PetscViewerASCIIPrintf(viewer,"\n"));
      }
    }
    PetscCall(PetscViewerASCIIUseTabs(viewer,PETSC_TRUE));
  } else {
    PetscCall(PetscViewerASCIIUseTabs(viewer,PETSC_FALSE));
    for (i=0; i<a->mbs; i++) {
      for (j=0; j<bs; j++) {
        PetscCall(PetscViewerASCIIPrintf(viewer,"row %" PetscInt_FMT ":",i*bs+j));
        for (k=a->i[i]; k<a->i[i+1]; k++) {
          for (l=0; l<bs; l++) {
#if defined(PETSC_USE_COMPLEX)
            if (PetscImaginaryPart(a->a[bs2*k + l*bs + j]) > 0.0) {
              PetscCall(PetscViewerASCIIPrintf(viewer," (%" PetscInt_FMT ", %g + %g i) ",bs*a->j[k]+l,
                                             (double)PetscRealPart(a->a[bs2*k + l*bs + j]),(double)PetscImaginaryPart(a->a[bs2*k + l*bs + j])));
            } else if (PetscImaginaryPart(a->a[bs2*k + l*bs + j]) < 0.0) {
              PetscCall(PetscViewerASCIIPrintf(viewer," (%" PetscInt_FMT ", %g - %g i) ",bs*a->j[k]+l,
                                             (double)PetscRealPart(a->a[bs2*k + l*bs + j]),-(double)PetscImaginaryPart(a->a[bs2*k + l*bs + j])));
            } else {
              PetscCall(PetscViewerASCIIPrintf(viewer," (%" PetscInt_FMT ", %g) ",bs*a->j[k]+l,(double)PetscRealPart(a->a[bs2*k + l*bs + j])));
            }
#else
            PetscCall(PetscViewerASCIIPrintf(viewer," (%" PetscInt_FMT ", %g) ",bs*a->j[k]+l,(double)a->a[bs2*k + l*bs + j]));
#endif
          }
        }
        PetscCall(PetscViewerASCIIPrintf(viewer,"\n"));
      }
    }
    PetscCall(PetscViewerASCIIUseTabs(viewer,PETSC_TRUE));
  }
  PetscCall(PetscViewerFlush(viewer));
  PetscFunctionReturn(0);
}

#include <petscdraw.h>
static PetscErrorCode MatView_SeqBAIJ_Draw_Zoom(PetscDraw draw,void *Aa)
{
  Mat               A = (Mat) Aa;
  Mat_SeqBAIJ       *a=(Mat_SeqBAIJ*)A->data;
  PetscInt          row,i,j,k,l,mbs=a->mbs,color,bs=A->rmap->bs,bs2=a->bs2;
  PetscReal         xl,yl,xr,yr,x_l,x_r,y_l,y_r;
  MatScalar         *aa;
  PetscViewer       viewer;
  PetscViewerFormat format;

  PetscFunctionBegin;
  PetscCall(PetscObjectQuery((PetscObject)A,"Zoomviewer",(PetscObject*)&viewer));
  PetscCall(PetscViewerGetFormat(viewer,&format));
  PetscCall(PetscDrawGetCoordinates(draw,&xl,&yl,&xr,&yr));

  /* loop over matrix elements drawing boxes */

  if (format != PETSC_VIEWER_DRAW_CONTOUR) {
    PetscDrawCollectiveBegin(draw);
    /* Blue for negative, Cyan for zero and  Red for positive */
    color = PETSC_DRAW_BLUE;
    for (i=0,row=0; i<mbs; i++,row+=bs) {
      for (j=a->i[i]; j<a->i[i+1]; j++) {
        y_l = A->rmap->N - row - 1.0; y_r = y_l + 1.0;
        x_l = a->j[j]*bs; x_r = x_l + 1.0;
        aa  = a->a + j*bs2;
        for (k=0; k<bs; k++) {
          for (l=0; l<bs; l++) {
            if (PetscRealPart(*aa++) >=  0.) continue;
            PetscCall(PetscDrawRectangle(draw,x_l+k,y_l-l,x_r+k,y_r-l,color,color,color,color));
          }
        }
      }
    }
    color = PETSC_DRAW_CYAN;
    for (i=0,row=0; i<mbs; i++,row+=bs) {
      for (j=a->i[i]; j<a->i[i+1]; j++) {
        y_l = A->rmap->N - row - 1.0; y_r = y_l + 1.0;
        x_l = a->j[j]*bs; x_r = x_l + 1.0;
        aa  = a->a + j*bs2;
        for (k=0; k<bs; k++) {
          for (l=0; l<bs; l++) {
            if (PetscRealPart(*aa++) != 0.) continue;
            PetscCall(PetscDrawRectangle(draw,x_l+k,y_l-l,x_r+k,y_r-l,color,color,color,color));
          }
        }
      }
    }
    color = PETSC_DRAW_RED;
    for (i=0,row=0; i<mbs; i++,row+=bs) {
      for (j=a->i[i]; j<a->i[i+1]; j++) {
        y_l = A->rmap->N - row - 1.0; y_r = y_l + 1.0;
        x_l = a->j[j]*bs; x_r = x_l + 1.0;
        aa  = a->a + j*bs2;
        for (k=0; k<bs; k++) {
          for (l=0; l<bs; l++) {
            if (PetscRealPart(*aa++) <= 0.) continue;
            PetscCall(PetscDrawRectangle(draw,x_l+k,y_l-l,x_r+k,y_r-l,color,color,color,color));
          }
        }
      }
    }
    PetscDrawCollectiveEnd(draw);
  } else {
    /* use contour shading to indicate magnitude of values */
    /* first determine max of all nonzero values */
    PetscReal      minv = 0.0, maxv = 0.0;
    PetscDraw      popup;

    for (i=0; i<a->nz*a->bs2; i++) {
      if (PetscAbsScalar(a->a[i]) > maxv) maxv = PetscAbsScalar(a->a[i]);
    }
    if (minv >= maxv) maxv = minv + PETSC_SMALL;
    PetscCall(PetscDrawGetPopup(draw,&popup));
    PetscCall(PetscDrawScalePopup(popup,0.0,maxv));

    PetscDrawCollectiveBegin(draw);
    for (i=0,row=0; i<mbs; i++,row+=bs) {
      for (j=a->i[i]; j<a->i[i+1]; j++) {
        y_l = A->rmap->N - row - 1.0; y_r = y_l + 1.0;
        x_l = a->j[j]*bs; x_r = x_l + 1.0;
        aa  = a->a + j*bs2;
        for (k=0; k<bs; k++) {
          for (l=0; l<bs; l++) {
            MatScalar v = *aa++;
            color = PetscDrawRealToColor(PetscAbsScalar(v),minv,maxv);
            PetscCall(PetscDrawRectangle(draw,x_l+k,y_l-l,x_r+k,y_r-l,color,color,color,color));
          }
        }
      }
    }
    PetscDrawCollectiveEnd(draw);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatView_SeqBAIJ_Draw(Mat A,PetscViewer viewer)
{
  PetscReal      xl,yl,xr,yr,w,h;
  PetscDraw      draw;
  PetscBool      isnull;

  PetscFunctionBegin;
  PetscCall(PetscViewerDrawGetDraw(viewer,0,&draw));
  PetscCall(PetscDrawIsNull(draw,&isnull));
  if (isnull) PetscFunctionReturn(0);

  xr   = A->cmap->n; yr = A->rmap->N; h = yr/10.0; w = xr/10.0;
  xr  += w;          yr += h;        xl = -w;     yl = -h;
  PetscCall(PetscDrawSetCoordinates(draw,xl,yl,xr,yr));
  PetscCall(PetscObjectCompose((PetscObject)A,"Zoomviewer",(PetscObject)viewer));
  PetscCall(PetscDrawZoom(draw,MatView_SeqBAIJ_Draw_Zoom,A));
  PetscCall(PetscObjectCompose((PetscObject)A,"Zoomviewer",NULL));
  PetscCall(PetscDrawSave(draw));
  PetscFunctionReturn(0);
}

PetscErrorCode MatView_SeqBAIJ(Mat A,PetscViewer viewer)
{
  PetscBool      iascii,isbinary,isdraw;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERDRAW,&isdraw));
  if (iascii) {
    PetscCall(MatView_SeqBAIJ_ASCII(A,viewer));
  } else if (isbinary) {
    PetscCall(MatView_SeqBAIJ_Binary(A,viewer));
  } else if (isdraw) {
    PetscCall(MatView_SeqBAIJ_Draw(A,viewer));
  } else {
    Mat B;
    PetscCall(MatConvert(A,MATSEQAIJ,MAT_INITIAL_MATRIX,&B));
    PetscCall(MatView(B,viewer));
    PetscCall(MatDestroy(&B));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatGetValues_SeqBAIJ(Mat A,PetscInt m,const PetscInt im[],PetscInt n,const PetscInt in[],PetscScalar v[])
{
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ*)A->data;
  PetscInt    *rp,k,low,high,t,row,nrow,i,col,l,*aj = a->j;
  PetscInt    *ai = a->i,*ailen = a->ilen;
  PetscInt    brow,bcol,ridx,cidx,bs=A->rmap->bs,bs2=a->bs2;
  MatScalar   *ap,*aa = a->a;

  PetscFunctionBegin;
  for (k=0; k<m; k++) { /* loop over rows */
    row = im[k]; brow = row/bs;
    if (row < 0) {v += n; continue;} /* negative row */
    PetscCheck(row < A->rmap->N,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row %" PetscInt_FMT " too large", row);
    rp   = aj ? aj + ai[brow] : NULL; /* mustn't add to NULL, that is UB */
    ap   = aa ? aa + bs2*ai[brow] : NULL; /* mustn't add to NULL, that is UB */
    nrow = ailen[brow];
    for (l=0; l<n; l++) { /* loop over columns */
      if (in[l] < 0) {v++; continue;} /* negative column */
      PetscCheck(in[l] < A->cmap->n,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Column %" PetscInt_FMT " too large", in[l]);
      col  = in[l];
      bcol = col/bs;
      cidx = col%bs;
      ridx = row%bs;
      high = nrow;
      low  = 0; /* assume unsorted */
      while (high-low > 5) {
        t = (low+high)/2;
        if (rp[t] > bcol) high = t;
        else             low  = t;
      }
      for (i=low; i<high; i++) {
        if (rp[i] > bcol) break;
        if (rp[i] == bcol) {
          *v++ = ap[bs2*i+bs*cidx+ridx];
          goto finished;
        }
      }
      *v++ = 0.0;
finished:;
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatSetValuesBlocked_SeqBAIJ(Mat A,PetscInt m,const PetscInt im[],PetscInt n,const PetscInt in[],const PetscScalar v[],InsertMode is)
{
  Mat_SeqBAIJ       *a = (Mat_SeqBAIJ*)A->data;
  PetscInt          *rp,k,low,high,t,ii,jj,row,nrow,i,col,l,rmax,N,lastcol = -1;
  PetscInt          *imax=a->imax,*ai=a->i,*ailen=a->ilen;
  PetscInt          *aj        =a->j,nonew=a->nonew,bs2=a->bs2,bs=A->rmap->bs,stepval;
  PetscBool         roworiented=a->roworiented;
  const PetscScalar *value     = v;
  MatScalar         *ap=NULL,*aa = a->a,*bap;

  PetscFunctionBegin;
  if (roworiented) {
    stepval = (n-1)*bs;
  } else {
    stepval = (m-1)*bs;
  }
  for (k=0; k<m; k++) { /* loop over added rows */
    row = im[k];
    if (row < 0) continue;
    PetscCheck(row < a->mbs,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Block row index too large %" PetscInt_FMT " max %" PetscInt_FMT,row,a->mbs-1);
    rp   = aj + ai[row];
    if (!A->structure_only) ap = aa + bs2*ai[row];
    rmax = imax[row];
    nrow = ailen[row];
    low  = 0;
    high = nrow;
    for (l=0; l<n; l++) { /* loop over added columns */
      if (in[l] < 0) continue;
      PetscCheck(in[l] < a->nbs,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Block column index too large %" PetscInt_FMT " max %" PetscInt_FMT,in[l],a->nbs-1);
      col = in[l];
      if (!A->structure_only) {
        if (roworiented) {
          value = v + (k*(stepval+bs) + l)*bs;
        } else {
          value = v + (l*(stepval+bs) + k)*bs;
        }
      }
      if (col <= lastcol) low = 0;
      else high = nrow;
      lastcol = col;
      while (high-low > 7) {
        t = (low+high)/2;
        if (rp[t] > col) high = t;
        else             low  = t;
      }
      for (i=low; i<high; i++) {
        if (rp[i] > col) break;
        if (rp[i] == col) {
          if (A->structure_only) goto noinsert2;
          bap = ap +  bs2*i;
          if (roworiented) {
            if (is == ADD_VALUES) {
              for (ii=0; ii<bs; ii++,value+=stepval) {
                for (jj=ii; jj<bs2; jj+=bs) {
                  bap[jj] += *value++;
                }
              }
            } else {
              for (ii=0; ii<bs; ii++,value+=stepval) {
                for (jj=ii; jj<bs2; jj+=bs) {
                  bap[jj] = *value++;
                }
              }
            }
          } else {
            if (is == ADD_VALUES) {
              for (ii=0; ii<bs; ii++,value+=bs+stepval) {
                for (jj=0; jj<bs; jj++) {
                  bap[jj] += value[jj];
                }
                bap += bs;
              }
            } else {
              for (ii=0; ii<bs; ii++,value+=bs+stepval) {
                for (jj=0; jj<bs; jj++) {
                  bap[jj]  = value[jj];
                }
                bap += bs;
              }
            }
          }
          goto noinsert2;
        }
      }
      if (nonew == 1) goto noinsert2;
      PetscCheck(nonew != -1,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Inserting a new blocked index new nonzero block (%" PetscInt_FMT ", %" PetscInt_FMT ") in the matrix", row, col);
      if (A->structure_only) {
        MatSeqXAIJReallocateAIJ_structure_only(A,a->mbs,bs2,nrow,row,col,rmax,ai,aj,rp,imax,nonew,MatScalar);
      } else {
        MatSeqXAIJReallocateAIJ(A,a->mbs,bs2,nrow,row,col,rmax,aa,ai,aj,rp,ap,imax,nonew,MatScalar);
      }
      N = nrow++ - 1; high++;
      /* shift up all the later entries in this row */
      PetscCall(PetscArraymove(rp+i+1,rp+i,N-i+1));
      rp[i] = col;
      if (!A->structure_only) {
        PetscCall(PetscArraymove(ap+bs2*(i+1),ap+bs2*i,bs2*(N-i+1)));
        bap   = ap +  bs2*i;
        if (roworiented) {
          for (ii=0; ii<bs; ii++,value+=stepval) {
            for (jj=ii; jj<bs2; jj+=bs) {
              bap[jj] = *value++;
            }
          }
        } else {
          for (ii=0; ii<bs; ii++,value+=stepval) {
            for (jj=0; jj<bs; jj++) {
              *bap++ = *value++;
            }
          }
        }
      }
noinsert2:;
      low = i;
    }
    ailen[row] = nrow;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatAssemblyEnd_SeqBAIJ(Mat A,MatAssemblyType mode)
{
  Mat_SeqBAIJ    *a     = (Mat_SeqBAIJ*)A->data;
  PetscInt       fshift = 0,i,*ai = a->i,*aj = a->j,*imax = a->imax;
  PetscInt       m      = A->rmap->N,*ip,N,*ailen = a->ilen;
  PetscInt       mbs  = a->mbs,bs2 = a->bs2,rmax = 0;
  MatScalar      *aa  = a->a,*ap;
  PetscReal      ratio=0.6;

  PetscFunctionBegin;
  if (mode == MAT_FLUSH_ASSEMBLY) PetscFunctionReturn(0);

  if (m) rmax = ailen[0];
  for (i=1; i<mbs; i++) {
    /* move each row back by the amount of empty slots (fshift) before it*/
    fshift += imax[i-1] - ailen[i-1];
    rmax    = PetscMax(rmax,ailen[i]);
    if (fshift) {
      ip = aj + ai[i];
      ap = aa + bs2*ai[i];
      N  = ailen[i];
      PetscCall(PetscArraymove(ip-fshift,ip,N));
      if (!A->structure_only) {
        PetscCall(PetscArraymove(ap-bs2*fshift,ap,bs2*N));
      }
    }
    ai[i] = ai[i-1] + ailen[i-1];
  }
  if (mbs) {
    fshift += imax[mbs-1] - ailen[mbs-1];
    ai[mbs] = ai[mbs-1] + ailen[mbs-1];
  }

  /* reset ilen and imax for each row */
  a->nonzerorowcnt = 0;
  if (A->structure_only) {
    PetscCall(PetscFree2(a->imax,a->ilen));
  } else { /* !A->structure_only */
    for (i=0; i<mbs; i++) {
      ailen[i] = imax[i] = ai[i+1] - ai[i];
      a->nonzerorowcnt += ((ai[i+1] - ai[i]) > 0);
    }
  }
  a->nz = ai[mbs];

  /* diagonals may have moved, so kill the diagonal pointers */
  a->idiagvalid = PETSC_FALSE;
  if (fshift && a->diag) {
    PetscCall(PetscFree(a->diag));
    PetscCall(PetscLogObjectMemory((PetscObject)A,-(mbs+1)*sizeof(PetscInt)));
    a->diag = NULL;
  }
  if (fshift) PetscCheck(a->nounused != -1,PETSC_COMM_SELF,PETSC_ERR_PLIB, "Unused space detected in matrix: %" PetscInt_FMT " X %" PetscInt_FMT " block size %" PetscInt_FMT ", %" PetscInt_FMT " unneeded", m, A->cmap->n, A->rmap->bs, fshift*bs2);
  PetscCall(PetscInfo(A,"Matrix size: %" PetscInt_FMT " X %" PetscInt_FMT ", block size %" PetscInt_FMT "; storage space: %" PetscInt_FMT " unneeded, %" PetscInt_FMT " used\n",m,A->cmap->n,A->rmap->bs,fshift*bs2,a->nz*bs2));
  PetscCall(PetscInfo(A,"Number of mallocs during MatSetValues is %" PetscInt_FMT "\n",a->reallocs));
  PetscCall(PetscInfo(A,"Most nonzeros blocks in any row is %" PetscInt_FMT "\n",rmax));

  A->info.mallocs    += a->reallocs;
  a->reallocs         = 0;
  A->info.nz_unneeded = (PetscReal)fshift*bs2;
  a->rmax             = rmax;

  if (!A->structure_only) {
    PetscCall(MatCheckCompressedRow(A,a->nonzerorowcnt,&a->compressedrow,a->i,mbs,ratio));
  }
  PetscFunctionReturn(0);
}

/*
   This function returns an array of flags which indicate the locations of contiguous
   blocks that should be zeroed. for eg: if bs = 3  and is = [0,1,2,3,5,6,7,8,9]
   then the resulting sizes = [3,1,1,3,1] corresponding to sets [(0,1,2),(3),(5),(6,7,8),(9)]
   Assume: sizes should be long enough to hold all the values.
*/
static PetscErrorCode MatZeroRows_SeqBAIJ_Check_Blocks(PetscInt idx[],PetscInt n,PetscInt bs,PetscInt sizes[], PetscInt *bs_max)
{
  PetscInt  i,j,k,row;
  PetscBool flg;

  PetscFunctionBegin;
  for (i=0,j=0; i<n; j++) {
    row = idx[i];
    if (row%bs!=0) { /* Not the beginning of a block */
      sizes[j] = 1;
      i++;
    } else if (i+bs > n) { /* complete block doesn't exist (at idx end) */
      sizes[j] = 1;         /* Also makes sure at least 'bs' values exist for next else */
      i++;
    } else { /* Beginning of the block, so check if the complete block exists */
      flg = PETSC_TRUE;
      for (k=1; k<bs; k++) {
        if (row+k != idx[i+k]) { /* break in the block */
          flg = PETSC_FALSE;
          break;
        }
      }
      if (flg) { /* No break in the bs */
        sizes[j] = bs;
        i       += bs;
      } else {
        sizes[j] = 1;
        i++;
      }
    }
  }
  *bs_max = j;
  PetscFunctionReturn(0);
}

PetscErrorCode MatZeroRows_SeqBAIJ(Mat A,PetscInt is_n,const PetscInt is_idx[],PetscScalar diag,Vec x, Vec b)
{
  Mat_SeqBAIJ       *baij=(Mat_SeqBAIJ*)A->data;
  PetscInt          i,j,k,count,*rows;
  PetscInt          bs=A->rmap->bs,bs2=baij->bs2,*sizes,row,bs_max;
  PetscScalar       zero = 0.0;
  MatScalar         *aa;
  const PetscScalar *xx;
  PetscScalar       *bb;

  PetscFunctionBegin;
  /* fix right hand side if needed */
  if (x && b) {
    PetscCall(VecGetArrayRead(x,&xx));
    PetscCall(VecGetArray(b,&bb));
    for (i=0; i<is_n; i++) {
      bb[is_idx[i]] = diag*xx[is_idx[i]];
    }
    PetscCall(VecRestoreArrayRead(x,&xx));
    PetscCall(VecRestoreArray(b,&bb));
  }

  /* Make a copy of the IS and  sort it */
  /* allocate memory for rows,sizes */
  PetscCall(PetscMalloc2(is_n,&rows,2*is_n,&sizes));

  /* copy IS values to rows, and sort them */
  for (i=0; i<is_n; i++) rows[i] = is_idx[i];
  PetscCall(PetscSortInt(is_n,rows));

  if (baij->keepnonzeropattern) {
    for (i=0; i<is_n; i++) sizes[i] = 1;
    bs_max          = is_n;
  } else {
    PetscCall(MatZeroRows_SeqBAIJ_Check_Blocks(rows,is_n,bs,sizes,&bs_max));
    A->nonzerostate++;
  }

  for (i=0,j=0; i<bs_max; j+=sizes[i],i++) {
    row = rows[j];
    PetscCheck(row >= 0 && row <= A->rmap->N,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"row %" PetscInt_FMT " out of range",row);
    count = (baij->i[row/bs +1] - baij->i[row/bs])*bs;
    aa    = ((MatScalar*)(baij->a)) + baij->i[row/bs]*bs2 + (row%bs);
    if (sizes[i] == bs && !baij->keepnonzeropattern) {
      if (diag != (PetscScalar)0.0) {
        if (baij->ilen[row/bs] > 0) {
          baij->ilen[row/bs]       = 1;
          baij->j[baij->i[row/bs]] = row/bs;

          PetscCall(PetscArrayzero(aa,count*bs));
        }
        /* Now insert all the diagonal values for this bs */
        for (k=0; k<bs; k++) {
          PetscCall((*A->ops->setvalues)(A,1,rows+j+k,1,rows+j+k,&diag,INSERT_VALUES));
        }
      } else { /* (diag == 0.0) */
        baij->ilen[row/bs] = 0;
      } /* end (diag == 0.0) */
    } else { /* (sizes[i] != bs) */
      PetscAssert(sizes[i] == 1,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Internal Error. Value should be 1");
      for (k=0; k<count; k++) {
        aa[0] =  zero;
        aa   += bs;
      }
      if (diag != (PetscScalar)0.0) {
        PetscCall((*A->ops->setvalues)(A,1,rows+j,1,rows+j,&diag,INSERT_VALUES));
      }
    }
  }

  PetscCall(PetscFree2(rows,sizes));
  PetscCall(MatAssemblyEnd_SeqBAIJ(A,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

PetscErrorCode MatZeroRowsColumns_SeqBAIJ(Mat A,PetscInt is_n,const PetscInt is_idx[],PetscScalar diag,Vec x, Vec b)
{
  Mat_SeqBAIJ       *baij=(Mat_SeqBAIJ*)A->data;
  PetscInt          i,j,k,count;
  PetscInt          bs   =A->rmap->bs,bs2=baij->bs2,row,col;
  PetscScalar       zero = 0.0;
  MatScalar         *aa;
  const PetscScalar *xx;
  PetscScalar       *bb;
  PetscBool         *zeroed,vecs = PETSC_FALSE;

  PetscFunctionBegin;
  /* fix right hand side if needed */
  if (x && b) {
    PetscCall(VecGetArrayRead(x,&xx));
    PetscCall(VecGetArray(b,&bb));
    vecs = PETSC_TRUE;
  }

  /* zero the columns */
  PetscCall(PetscCalloc1(A->rmap->n,&zeroed));
  for (i=0; i<is_n; i++) {
    PetscCheck(is_idx[i] >= 0 && is_idx[i] < A->rmap->N,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"row %" PetscInt_FMT " out of range",is_idx[i]);
    zeroed[is_idx[i]] = PETSC_TRUE;
  }
  for (i=0; i<A->rmap->N; i++) {
    if (!zeroed[i]) {
      row = i/bs;
      for (j=baij->i[row]; j<baij->i[row+1]; j++) {
        for (k=0; k<bs; k++) {
          col = bs*baij->j[j] + k;
          if (zeroed[col]) {
            aa = ((MatScalar*)(baij->a)) + j*bs2 + (i%bs) + bs*k;
            if (vecs) bb[i] -= aa[0]*xx[col];
            aa[0] = 0.0;
          }
        }
      }
    } else if (vecs) bb[i] = diag*xx[i];
  }
  PetscCall(PetscFree(zeroed));
  if (vecs) {
    PetscCall(VecRestoreArrayRead(x,&xx));
    PetscCall(VecRestoreArray(b,&bb));
  }

  /* zero the rows */
  for (i=0; i<is_n; i++) {
    row   = is_idx[i];
    count = (baij->i[row/bs +1] - baij->i[row/bs])*bs;
    aa    = ((MatScalar*)(baij->a)) + baij->i[row/bs]*bs2 + (row%bs);
    for (k=0; k<count; k++) {
      aa[0] =  zero;
      aa   += bs;
    }
    if (diag != (PetscScalar)0.0) {
      PetscCall((*A->ops->setvalues)(A,1,&row,1,&row,&diag,INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyEnd_SeqBAIJ(A,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

PetscErrorCode MatSetValues_SeqBAIJ(Mat A,PetscInt m,const PetscInt im[],PetscInt n,const PetscInt in[],const PetscScalar v[],InsertMode is)
{
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data;
  PetscInt       *rp,k,low,high,t,ii,row,nrow,i,col,l,rmax,N,lastcol = -1;
  PetscInt       *imax=a->imax,*ai=a->i,*ailen=a->ilen;
  PetscInt       *aj  =a->j,nonew=a->nonew,bs=A->rmap->bs,brow,bcol;
  PetscInt       ridx,cidx,bs2=a->bs2;
  PetscBool      roworiented=a->roworiented;
  MatScalar      *ap=NULL,value=0.0,*aa=a->a,*bap;

  PetscFunctionBegin;
  for (k=0; k<m; k++) { /* loop over added rows */
    row  = im[k];
    brow = row/bs;
    if (row < 0) continue;
    PetscCheck(row < A->rmap->N,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row too large: row %" PetscInt_FMT " max %" PetscInt_FMT,row,A->rmap->N-1);
    rp   = aj + ai[brow];
    if (!A->structure_only) ap = aa + bs2*ai[brow];
    rmax = imax[brow];
    nrow = ailen[brow];
    low  = 0;
    high = nrow;
    for (l=0; l<n; l++) { /* loop over added columns */
      if (in[l] < 0) continue;
      PetscCheck(in[l] < A->cmap->n,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Column too large: col %" PetscInt_FMT " max %" PetscInt_FMT,in[l],A->cmap->n-1);
      col  = in[l]; bcol = col/bs;
      ridx = row % bs; cidx = col % bs;
      if (!A->structure_only) {
        if (roworiented) {
          value = v[l + k*n];
        } else {
          value = v[k + l*m];
        }
      }
      if (col <= lastcol) low = 0; else high = nrow;
      lastcol = col;
      while (high-low > 7) {
        t = (low+high)/2;
        if (rp[t] > bcol) high = t;
        else              low  = t;
      }
      for (i=low; i<high; i++) {
        if (rp[i] > bcol) break;
        if (rp[i] == bcol) {
          bap = ap +  bs2*i + bs*cidx + ridx;
          if (!A->structure_only) {
            if (is == ADD_VALUES) *bap += value;
            else                  *bap  = value;
          }
          goto noinsert1;
        }
      }
      if (nonew == 1) goto noinsert1;
      PetscCheck(nonew != -1,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Inserting a new nonzero (%" PetscInt_FMT ", %" PetscInt_FMT ") in the matrix", row, col);
      if (A->structure_only) {
        MatSeqXAIJReallocateAIJ_structure_only(A,a->mbs,bs2,nrow,brow,bcol,rmax,ai,aj,rp,imax,nonew,MatScalar);
      } else {
        MatSeqXAIJReallocateAIJ(A,a->mbs,bs2,nrow,brow,bcol,rmax,aa,ai,aj,rp,ap,imax,nonew,MatScalar);
      }
      N = nrow++ - 1; high++;
      /* shift up all the later entries in this row */
      PetscCall(PetscArraymove(rp+i+1,rp+i,N-i+1));
      rp[i] = bcol;
      if (!A->structure_only) {
        PetscCall(PetscArraymove(ap+bs2*(i+1),ap+bs2*i,bs2*(N-i+1)));
        PetscCall(PetscArrayzero(ap+bs2*i,bs2));
        ap[bs2*i + bs*cidx + ridx] = value;
      }
      a->nz++;
      A->nonzerostate++;
noinsert1:;
      low = i;
    }
    ailen[brow] = nrow;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatILUFactor_SeqBAIJ(Mat inA,IS row,IS col,const MatFactorInfo *info)
{
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)inA->data;
  Mat            outA;
  PetscBool      row_identity,col_identity;

  PetscFunctionBegin;
  PetscCheck(info->levels == 0,PETSC_COMM_SELF,PETSC_ERR_SUP,"Only levels = 0 supported for in-place ILU");
  PetscCall(ISIdentity(row,&row_identity));
  PetscCall(ISIdentity(col,&col_identity));
  PetscCheck(row_identity && col_identity,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Row and column permutations must be identity for in-place ILU");

  outA            = inA;
  inA->factortype = MAT_FACTOR_LU;
  PetscCall(PetscFree(inA->solvertype));
  PetscCall(PetscStrallocpy(MATSOLVERPETSC,&inA->solvertype));

  PetscCall(MatMarkDiagonal_SeqBAIJ(inA));

  PetscCall(PetscObjectReference((PetscObject)row));
  PetscCall(ISDestroy(&a->row));
  a->row = row;
  PetscCall(PetscObjectReference((PetscObject)col));
  PetscCall(ISDestroy(&a->col));
  a->col = col;

  /* Create the invert permutation so that it can be used in MatLUFactorNumeric() */
  PetscCall(ISDestroy(&a->icol));
  PetscCall(ISInvertPermutation(col,PETSC_DECIDE,&a->icol));
  PetscCall(PetscLogObjectParent((PetscObject)inA,(PetscObject)a->icol));

  PetscCall(MatSeqBAIJSetNumericFactorization_inplace(inA,(PetscBool)(row_identity && col_identity)));
  if (!a->solve_work) {
    PetscCall(PetscMalloc1(inA->rmap->N+inA->rmap->bs,&a->solve_work));
    PetscCall(PetscLogObjectMemory((PetscObject)inA,(inA->rmap->N+inA->rmap->bs)*sizeof(PetscScalar)));
  }
  PetscCall(MatLUFactorNumeric(outA,inA,info));
  PetscFunctionReturn(0);
}

PetscErrorCode  MatSeqBAIJSetColumnIndices_SeqBAIJ(Mat mat,PetscInt *indices)
{
  Mat_SeqBAIJ *baij = (Mat_SeqBAIJ*)mat->data;
  PetscInt    i,nz,mbs;

  PetscFunctionBegin;
  nz  = baij->maxnz;
  mbs = baij->mbs;
  for (i=0; i<nz; i++) {
    baij->j[i] = indices[i];
  }
  baij->nz = nz;
  for (i=0; i<mbs; i++) {
    baij->ilen[i] = baij->imax[i];
  }
  PetscFunctionReturn(0);
}

/*@
    MatSeqBAIJSetColumnIndices - Set the column indices for all the rows
       in the matrix.

  Input Parameters:
+  mat - the SeqBAIJ matrix
-  indices - the column indices

  Level: advanced

  Notes:
    This can be called if you have precomputed the nonzero structure of the
  matrix and want to provide it to the matrix object to improve the performance
  of the MatSetValues() operation.

    You MUST have set the correct numbers of nonzeros per row in the call to
  MatCreateSeqBAIJ(), and the columns indices MUST be sorted.

    MUST be called before any calls to MatSetValues();

@*/
PetscErrorCode  MatSeqBAIJSetColumnIndices(Mat mat,PetscInt *indices)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidIntPointer(indices,2);
  PetscUseMethod(mat,"MatSeqBAIJSetColumnIndices_C",(Mat,PetscInt*),(mat,indices));
  PetscFunctionReturn(0);
}

PetscErrorCode MatGetRowMaxAbs_SeqBAIJ(Mat A,Vec v,PetscInt idx[])
{
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data;
  PetscInt       i,j,n,row,bs,*ai,*aj,mbs;
  PetscReal      atmp;
  PetscScalar    *x,zero = 0.0;
  MatScalar      *aa;
  PetscInt       ncols,brow,krow,kcol;

  PetscFunctionBegin;
  /* why is this not a macro???????????????????????????????????????????????????????????????? */
  PetscCheck(!A->factortype,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  bs  = A->rmap->bs;
  aa  = a->a;
  ai  = a->i;
  aj  = a->j;
  mbs = a->mbs;

  PetscCall(VecSet(v,zero));
  PetscCall(VecGetArray(v,&x));
  PetscCall(VecGetLocalSize(v,&n));
  PetscCheck(n == A->rmap->N,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Nonconforming matrix and vector");
  for (i=0; i<mbs; i++) {
    ncols = ai[1] - ai[0]; ai++;
    brow  = bs*i;
    for (j=0; j<ncols; j++) {
      for (kcol=0; kcol<bs; kcol++) {
        for (krow=0; krow<bs; krow++) {
          atmp = PetscAbsScalar(*aa);aa++;
          row  = brow + krow;   /* row index */
          if (PetscAbsScalar(x[row]) < atmp) {x[row] = atmp; if (idx) idx[row] = bs*(*aj) + kcol;}
        }
      }
      aj++;
    }
  }
  PetscCall(VecRestoreArray(v,&x));
  PetscFunctionReturn(0);
}

PetscErrorCode MatCopy_SeqBAIJ(Mat A,Mat B,MatStructure str)
{
  PetscFunctionBegin;
  /* If the two matrices have the same copy implementation, use fast copy. */
  if (str == SAME_NONZERO_PATTERN && (A->ops->copy == B->ops->copy)) {
    Mat_SeqBAIJ *a  = (Mat_SeqBAIJ*)A->data;
    Mat_SeqBAIJ *b  = (Mat_SeqBAIJ*)B->data;
    PetscInt    ambs=a->mbs,bmbs=b->mbs,abs=A->rmap->bs,bbs=B->rmap->bs,bs2=abs*abs;

    PetscCheck(a->i[ambs] == b->i[bmbs],PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Number of nonzero blocks in matrices A %" PetscInt_FMT " and B %" PetscInt_FMT " are different",a->i[ambs],b->i[bmbs]);
    PetscCheck(abs == bbs,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Block size A %" PetscInt_FMT " and B %" PetscInt_FMT " are different",abs,bbs);
    PetscCall(PetscArraycpy(b->a,a->a,bs2*a->i[ambs]));
    PetscCall(PetscObjectStateIncrease((PetscObject)B));
  } else {
    PetscCall(MatCopy_Basic(A,B,str));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatSetUp_SeqBAIJ(Mat A)
{
  PetscFunctionBegin;
  PetscCall(MatSeqBAIJSetPreallocation(A,A->rmap->bs,PETSC_DEFAULT,NULL));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqBAIJGetArray_SeqBAIJ(Mat A,PetscScalar *array[])
{
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ*)A->data;

  PetscFunctionBegin;
  *array = a->a;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqBAIJRestoreArray_SeqBAIJ(Mat A,PetscScalar *array[])
{
  PetscFunctionBegin;
  *array = NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode MatAXPYGetPreallocation_SeqBAIJ(Mat Y,Mat X,PetscInt *nnz)
{
  PetscInt       bs = Y->rmap->bs,mbs = Y->rmap->N/bs;
  Mat_SeqBAIJ    *x = (Mat_SeqBAIJ*)X->data;
  Mat_SeqBAIJ    *y = (Mat_SeqBAIJ*)Y->data;

  PetscFunctionBegin;
  /* Set the number of nonzeros in the new matrix */
  PetscCall(MatAXPYGetPreallocation_SeqX_private(mbs,x->i,x->j,y->i,y->j,nnz));
  PetscFunctionReturn(0);
}

PetscErrorCode MatAXPY_SeqBAIJ(Mat Y,PetscScalar a,Mat X,MatStructure str)
{
  Mat_SeqBAIJ    *x = (Mat_SeqBAIJ*)X->data,*y = (Mat_SeqBAIJ*)Y->data;
  PetscInt       bs=Y->rmap->bs,bs2=bs*bs;
  PetscBLASInt   one=1;

  PetscFunctionBegin;
  if (str == UNKNOWN_NONZERO_PATTERN || (PetscDefined(USE_DEBUG) && str == SAME_NONZERO_PATTERN)) {
    PetscBool e = x->nz == y->nz && x->mbs == y->mbs && bs == X->rmap->bs ? PETSC_TRUE : PETSC_FALSE;
    if (e) {
      PetscCall(PetscArraycmp(x->i,y->i,x->mbs+1,&e));
      if (e) {
        PetscCall(PetscArraycmp(x->j,y->j,x->i[x->mbs],&e));
        if (e) str = SAME_NONZERO_PATTERN;
      }
    }
    if (!e) PetscCheck(str != SAME_NONZERO_PATTERN,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"MatStructure is not SAME_NONZERO_PATTERN");
  }
  if (str == SAME_NONZERO_PATTERN) {
    PetscScalar  alpha = a;
    PetscBLASInt bnz;
    PetscCall(PetscBLASIntCast(x->nz*bs2,&bnz));
    PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&bnz,&alpha,x->a,&one,y->a,&one));
    PetscCall(PetscObjectStateIncrease((PetscObject)Y));
  } else if (str == SUBSET_NONZERO_PATTERN) { /* nonzeros of X is a subset of Y's */
    PetscCall(MatAXPY_Basic(Y,a,X,str));
  } else {
    Mat      B;
    PetscInt *nnz;
    PetscCheck(bs == X->rmap->bs,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Matrices must have same block size");
    PetscCall(PetscMalloc1(Y->rmap->N,&nnz));
    PetscCall(MatCreate(PetscObjectComm((PetscObject)Y),&B));
    PetscCall(PetscObjectSetName((PetscObject)B,((PetscObject)Y)->name));
    PetscCall(MatSetSizes(B,Y->rmap->n,Y->cmap->n,Y->rmap->N,Y->cmap->N));
    PetscCall(MatSetBlockSizesFromMats(B,Y,Y));
    PetscCall(MatSetType(B,(MatType) ((PetscObject)Y)->type_name));
    PetscCall(MatAXPYGetPreallocation_SeqBAIJ(Y,X,nnz));
    PetscCall(MatSeqBAIJSetPreallocation(B,bs,0,nnz));
    PetscCall(MatAXPY_BasicWithPreallocation(B,Y,a,X,str));
    PetscCall(MatHeaderMerge(Y,&B));
    PetscCall(PetscFree(nnz));
  }
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatConjugate_SeqBAIJ(Mat A)
{
#if defined(PETSC_USE_COMPLEX)
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ*)A->data;
  PetscInt    i,nz = a->bs2*a->i[a->mbs];
  MatScalar   *aa = a->a;

  PetscFunctionBegin;
  for (i=0; i<nz; i++) aa[i] = PetscConj(aa[i]);
#else
  PetscFunctionBegin;
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode MatRealPart_SeqBAIJ(Mat A)
{
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ*)A->data;
  PetscInt    i,nz = a->bs2*a->i[a->mbs];
  MatScalar   *aa = a->a;

  PetscFunctionBegin;
  for (i=0; i<nz; i++) aa[i] = PetscRealPart(aa[i]);
  PetscFunctionReturn(0);
}

PetscErrorCode MatImaginaryPart_SeqBAIJ(Mat A)
{
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ*)A->data;
  PetscInt    i,nz = a->bs2*a->i[a->mbs];
  MatScalar   *aa = a->a;

  PetscFunctionBegin;
  for (i=0; i<nz; i++) aa[i] = PetscImaginaryPart(aa[i]);
  PetscFunctionReturn(0);
}

/*
    Code almost identical to MatGetColumnIJ_SeqAIJ() should share common code
*/
PetscErrorCode MatGetColumnIJ_SeqBAIJ(Mat A,PetscInt oshift,PetscBool symmetric,PetscBool inodecompressed,PetscInt *nn,const PetscInt *ia[],const PetscInt *ja[],PetscBool  *done)
{
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data;
  PetscInt       bs = A->rmap->bs,i,*collengths,*cia,*cja,n = A->cmap->n/bs,m = A->rmap->n/bs;
  PetscInt       nz = a->i[m],row,*jj,mr,col;

  PetscFunctionBegin;
  *nn = n;
  if (!ia) PetscFunctionReturn(0);
  PetscCheck(!symmetric,PETSC_COMM_SELF,PETSC_ERR_SUP,"Not for BAIJ matrices");
  else {
    PetscCall(PetscCalloc1(n,&collengths));
    PetscCall(PetscMalloc1(n+1,&cia));
    PetscCall(PetscMalloc1(nz,&cja));
    jj   = a->j;
    for (i=0; i<nz; i++) {
      collengths[jj[i]]++;
    }
    cia[0] = oshift;
    for (i=0; i<n; i++) {
      cia[i+1] = cia[i] + collengths[i];
    }
    PetscCall(PetscArrayzero(collengths,n));
    jj   = a->j;
    for (row=0; row<m; row++) {
      mr = a->i[row+1] - a->i[row];
      for (i=0; i<mr; i++) {
        col = *jj++;

        cja[cia[col] + collengths[col]++ - oshift] = row + oshift;
      }
    }
    PetscCall(PetscFree(collengths));
    *ia  = cia; *ja = cja;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatRestoreColumnIJ_SeqBAIJ(Mat A,PetscInt oshift,PetscBool symmetric,PetscBool inodecompressed,PetscInt *n,const PetscInt *ia[],const PetscInt *ja[],PetscBool  *done)
{
  PetscFunctionBegin;
  if (!ia) PetscFunctionReturn(0);
  PetscCall(PetscFree(*ia));
  PetscCall(PetscFree(*ja));
  PetscFunctionReturn(0);
}

/*
 MatGetColumnIJ_SeqBAIJ_Color() and MatRestoreColumnIJ_SeqBAIJ_Color() are customized from
 MatGetColumnIJ_SeqBAIJ() and MatRestoreColumnIJ_SeqBAIJ() by adding an output
 spidx[], index of a->a, to be used in MatTransposeColoringCreate() and MatFDColoringCreate()
 */
PetscErrorCode MatGetColumnIJ_SeqBAIJ_Color(Mat A,PetscInt oshift,PetscBool symmetric,PetscBool inodecompressed,PetscInt *nn,const PetscInt *ia[],const PetscInt *ja[],PetscInt *spidx[],PetscBool  *done)
{
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data;
  PetscInt       i,*collengths,*cia,*cja,n=a->nbs,m=a->mbs;
  PetscInt       nz = a->i[m],row,*jj,mr,col;
  PetscInt       *cspidx;

  PetscFunctionBegin;
  *nn = n;
  if (!ia) PetscFunctionReturn(0);

  PetscCall(PetscCalloc1(n,&collengths));
  PetscCall(PetscMalloc1(n+1,&cia));
  PetscCall(PetscMalloc1(nz,&cja));
  PetscCall(PetscMalloc1(nz,&cspidx));
  jj   = a->j;
  for (i=0; i<nz; i++) {
    collengths[jj[i]]++;
  }
  cia[0] = oshift;
  for (i=0; i<n; i++) {
    cia[i+1] = cia[i] + collengths[i];
  }
  PetscCall(PetscArrayzero(collengths,n));
  jj   = a->j;
  for (row=0; row<m; row++) {
    mr = a->i[row+1] - a->i[row];
    for (i=0; i<mr; i++) {
      col = *jj++;
      cspidx[cia[col] + collengths[col] - oshift] = a->i[row] + i; /* index of a->j */
      cja[cia[col] + collengths[col]++ - oshift]  = row + oshift;
    }
  }
  PetscCall(PetscFree(collengths));
  *ia    = cia;
  *ja    = cja;
  *spidx = cspidx;
  PetscFunctionReturn(0);
}

PetscErrorCode MatRestoreColumnIJ_SeqBAIJ_Color(Mat A,PetscInt oshift,PetscBool symmetric,PetscBool inodecompressed,PetscInt *n,const PetscInt *ia[],const PetscInt *ja[],PetscInt *spidx[],PetscBool  *done)
{
  PetscFunctionBegin;
  PetscCall(MatRestoreColumnIJ_SeqBAIJ(A,oshift,symmetric,inodecompressed,n,ia,ja,done));
  PetscCall(PetscFree(*spidx));
  PetscFunctionReturn(0);
}

PetscErrorCode MatShift_SeqBAIJ(Mat Y,PetscScalar a)
{
  Mat_SeqBAIJ     *aij = (Mat_SeqBAIJ*)Y->data;

  PetscFunctionBegin;
  if (!Y->preallocated || !aij->nz) {
    PetscCall(MatSeqBAIJSetPreallocation(Y,Y->rmap->bs,1,NULL));
  }
  PetscCall(MatShift_Basic(Y,a));
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------*/
static struct _MatOps MatOps_Values = {MatSetValues_SeqBAIJ,
                                       MatGetRow_SeqBAIJ,
                                       MatRestoreRow_SeqBAIJ,
                                       MatMult_SeqBAIJ_N,
                               /* 4*/  MatMultAdd_SeqBAIJ_N,
                                       MatMultTranspose_SeqBAIJ,
                                       MatMultTransposeAdd_SeqBAIJ,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 10*/ NULL,
                                       MatLUFactor_SeqBAIJ,
                                       NULL,
                                       NULL,
                                       MatTranspose_SeqBAIJ,
                               /* 15*/ MatGetInfo_SeqBAIJ,
                                       MatEqual_SeqBAIJ,
                                       MatGetDiagonal_SeqBAIJ,
                                       MatDiagonalScale_SeqBAIJ,
                                       MatNorm_SeqBAIJ,
                               /* 20*/ NULL,
                                       MatAssemblyEnd_SeqBAIJ,
                                       MatSetOption_SeqBAIJ,
                                       MatZeroEntries_SeqBAIJ,
                               /* 24*/ MatZeroRows_SeqBAIJ,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 29*/ MatSetUp_SeqBAIJ,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 34*/ MatDuplicate_SeqBAIJ,
                                       NULL,
                                       NULL,
                                       MatILUFactor_SeqBAIJ,
                                       NULL,
                               /* 39*/ MatAXPY_SeqBAIJ,
                                       MatCreateSubMatrices_SeqBAIJ,
                                       MatIncreaseOverlap_SeqBAIJ,
                                       MatGetValues_SeqBAIJ,
                                       MatCopy_SeqBAIJ,
                               /* 44*/ NULL,
                                       MatScale_SeqBAIJ,
                                       MatShift_SeqBAIJ,
                                       NULL,
                                       MatZeroRowsColumns_SeqBAIJ,
                               /* 49*/ NULL,
                                       MatGetRowIJ_SeqBAIJ,
                                       MatRestoreRowIJ_SeqBAIJ,
                                       MatGetColumnIJ_SeqBAIJ,
                                       MatRestoreColumnIJ_SeqBAIJ,
                               /* 54*/ MatFDColoringCreate_SeqXAIJ,
                                       NULL,
                                       NULL,
                                       NULL,
                                       MatSetValuesBlocked_SeqBAIJ,
                               /* 59*/ MatCreateSubMatrix_SeqBAIJ,
                                       MatDestroy_SeqBAIJ,
                                       MatView_SeqBAIJ,
                                       NULL,
                                       NULL,
                               /* 64*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 69*/ MatGetRowMaxAbs_SeqBAIJ,
                                       NULL,
                                       MatConvert_Basic,
                                       NULL,
                                       NULL,
                               /* 74*/ NULL,
                                       MatFDColoringApply_BAIJ,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 79*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       MatLoad_SeqBAIJ,
                               /* 84*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 89*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 94*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 99*/ NULL,
                                       NULL,
                                       NULL,
                                       MatConjugate_SeqBAIJ,
                                       NULL,
                               /*104*/ NULL,
                                       MatRealPart_SeqBAIJ,
                                       MatImaginaryPart_SeqBAIJ,
                                       NULL,
                                       NULL,
                               /*109*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       MatMissingDiagonal_SeqBAIJ,
                               /*114*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /*119*/ NULL,
                                       NULL,
                                       MatMultHermitianTranspose_SeqBAIJ,
                                       MatMultHermitianTransposeAdd_SeqBAIJ,
                                       NULL,
                               /*124*/ NULL,
                                       MatGetColumnReductions_SeqBAIJ,
                                       MatInvertBlockDiagonal_SeqBAIJ,
                                       NULL,
                                       NULL,
                               /*129*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /*134*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /*139*/ MatSetBlockSizes_Default,
                                       NULL,
                                       NULL,
                                       MatFDColoringSetUp_SeqXAIJ,
                                       NULL,
                                /*144*/MatCreateMPIMatConcatenateSeqMat_SeqBAIJ,
                                       MatDestroySubMatrices_SeqBAIJ,
                                       NULL,
                                       NULL
};

PetscErrorCode  MatStoreValues_SeqBAIJ(Mat mat)
{
  Mat_SeqBAIJ    *aij = (Mat_SeqBAIJ*)mat->data;
  PetscInt       nz   = aij->i[aij->mbs]*aij->bs2;

  PetscFunctionBegin;
  PetscCheck(aij->nonew == 1,PETSC_COMM_SELF,PETSC_ERR_ORDER,"Must call MatSetOption(A,MAT_NEW_NONZERO_LOCATIONS,PETSC_FALSE);first");

  /* allocate space for values if not already there */
  if (!aij->saved_values) {
    PetscCall(PetscMalloc1(nz+1,&aij->saved_values));
    PetscCall(PetscLogObjectMemory((PetscObject)mat,(nz+1)*sizeof(PetscScalar)));
  }

  /* copy values over */
  PetscCall(PetscArraycpy(aij->saved_values,aij->a,nz));
  PetscFunctionReturn(0);
}

PetscErrorCode  MatRetrieveValues_SeqBAIJ(Mat mat)
{
  Mat_SeqBAIJ    *aij = (Mat_SeqBAIJ*)mat->data;
  PetscInt       nz = aij->i[aij->mbs]*aij->bs2;

  PetscFunctionBegin;
  PetscCheck(aij->nonew == 1,PETSC_COMM_SELF,PETSC_ERR_ORDER,"Must call MatSetOption(A,MAT_NEW_NONZERO_LOCATIONS,PETSC_FALSE);first");
  PetscCheck(aij->saved_values,PETSC_COMM_SELF,PETSC_ERR_ORDER,"Must call MatStoreValues(A);first");

  /* copy values over */
  PetscCall(PetscArraycpy(aij->a,aij->saved_values,nz));
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatConvert_SeqBAIJ_SeqAIJ(Mat, MatType,MatReuse,Mat*);
PETSC_INTERN PetscErrorCode MatConvert_SeqBAIJ_SeqSBAIJ(Mat, MatType,MatReuse,Mat*);

PetscErrorCode  MatSeqBAIJSetPreallocation_SeqBAIJ(Mat B,PetscInt bs,PetscInt nz,PetscInt *nnz)
{
  Mat_SeqBAIJ    *b;
  PetscInt       i,mbs,nbs,bs2;
  PetscBool      flg = PETSC_FALSE,skipallocation = PETSC_FALSE,realalloc = PETSC_FALSE;

  PetscFunctionBegin;
  if (nz >= 0 || nnz) realalloc = PETSC_TRUE;
  if (nz == MAT_SKIP_ALLOCATION) {
    skipallocation = PETSC_TRUE;
    nz             = 0;
  }

  PetscCall(MatSetBlockSize(B,PetscAbs(bs)));
  PetscCall(PetscLayoutSetUp(B->rmap));
  PetscCall(PetscLayoutSetUp(B->cmap));
  PetscCall(PetscLayoutGetBlockSize(B->rmap,&bs));

  B->preallocated = PETSC_TRUE;

  mbs = B->rmap->n/bs;
  nbs = B->cmap->n/bs;
  bs2 = bs*bs;

  PetscCheck(mbs*bs==B->rmap->n && nbs*bs==B->cmap->n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Number rows %" PetscInt_FMT ", cols %" PetscInt_FMT " must be divisible by blocksize %" PetscInt_FMT,B->rmap->N,B->cmap->n,bs);

  if (nz == PETSC_DEFAULT || nz == PETSC_DECIDE) nz = 5;
  PetscCheck(nz >= 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"nz cannot be less than 0: value %" PetscInt_FMT,nz);
  if (nnz) {
    for (i=0; i<mbs; i++) {
      PetscCheck(nnz[i] >= 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"nnz cannot be less than 0: local row %" PetscInt_FMT " value %" PetscInt_FMT,i,nnz[i]);
      PetscCheck(nnz[i] <= nbs,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"nnz cannot be greater than block row length: local row %" PetscInt_FMT " value %" PetscInt_FMT " rowlength %" PetscInt_FMT,i,nnz[i],nbs);
    }
  }

  b    = (Mat_SeqBAIJ*)B->data;
  PetscOptionsBegin(PetscObjectComm((PetscObject)B),NULL,"Optimize options for SEQBAIJ matrix 2 ","Mat");
  PetscCall(PetscOptionsBool("-mat_no_unroll","Do not optimize for block size (slow)",NULL,flg,&flg,NULL));
  PetscOptionsEnd();

  if (!flg) {
    switch (bs) {
    case 1:
      B->ops->mult    = MatMult_SeqBAIJ_1;
      B->ops->multadd = MatMultAdd_SeqBAIJ_1;
      break;
    case 2:
      B->ops->mult    = MatMult_SeqBAIJ_2;
      B->ops->multadd = MatMultAdd_SeqBAIJ_2;
      break;
    case 3:
      B->ops->mult    = MatMult_SeqBAIJ_3;
      B->ops->multadd = MatMultAdd_SeqBAIJ_3;
      break;
    case 4:
      B->ops->mult    = MatMult_SeqBAIJ_4;
      B->ops->multadd = MatMultAdd_SeqBAIJ_4;
      break;
    case 5:
      B->ops->mult    = MatMult_SeqBAIJ_5;
      B->ops->multadd = MatMultAdd_SeqBAIJ_5;
      break;
    case 6:
      B->ops->mult    = MatMult_SeqBAIJ_6;
      B->ops->multadd = MatMultAdd_SeqBAIJ_6;
      break;
    case 7:
      B->ops->mult    = MatMult_SeqBAIJ_7;
      B->ops->multadd = MatMultAdd_SeqBAIJ_7;
      break;
    case 9:
    {
      PetscInt version = 1;
      PetscCall(PetscOptionsGetInt(NULL,((PetscObject)B)->prefix,"-mat_baij_mult_version",&version,NULL));
      switch (version) {
#if defined(PETSC_HAVE_IMMINTRIN_H) && defined(__AVX2__) && defined(__FMA__) && defined(PETSC_USE_REAL_DOUBLE) && !defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_64BIT_INDICES)
      case 1:
        B->ops->mult    = MatMult_SeqBAIJ_9_AVX2;
        B->ops->multadd = MatMultAdd_SeqBAIJ_9_AVX2;
        PetscCall(PetscInfo((PetscObject)B,"Using AVX2 for MatMult for BAIJ for blocksize %" PetscInt_FMT "\n",bs));
        break;
#endif
      default:
        B->ops->mult    = MatMult_SeqBAIJ_N;
        B->ops->multadd = MatMultAdd_SeqBAIJ_N;
        PetscCall(PetscInfo((PetscObject)B,"Using BLAS for MatMult for BAIJ for blocksize %" PetscInt_FMT "\n",bs));
        break;
      }
      break;
    }
    case 11:
      B->ops->mult    = MatMult_SeqBAIJ_11;
      B->ops->multadd = MatMultAdd_SeqBAIJ_11;
      break;
    case 12:
    {
      PetscInt version = 1;
      PetscCall(PetscOptionsGetInt(NULL,((PetscObject)B)->prefix,"-mat_baij_mult_version",&version,NULL));
      switch (version) {
      case 1:
        B->ops->mult    = MatMult_SeqBAIJ_12_ver1;
        B->ops->multadd = MatMultAdd_SeqBAIJ_12_ver1;
        PetscCall(PetscInfo((PetscObject)B,"Using version %" PetscInt_FMT " of MatMult for BAIJ for blocksize %" PetscInt_FMT "\n",version,bs));
        break;
      case 2:
        B->ops->mult    = MatMult_SeqBAIJ_12_ver2;
        B->ops->multadd = MatMultAdd_SeqBAIJ_12_ver2;
        PetscCall(PetscInfo((PetscObject)B,"Using version %" PetscInt_FMT " of MatMult for BAIJ for blocksize %" PetscInt_FMT "\n",version,bs));
        break;
#if defined(PETSC_HAVE_IMMINTRIN_H) && defined(__AVX2__) && defined(__FMA__) && defined(PETSC_USE_REAL_DOUBLE) && !defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_64BIT_INDICES)
      case 3:
        B->ops->mult    = MatMult_SeqBAIJ_12_AVX2;
        B->ops->multadd = MatMultAdd_SeqBAIJ_12_ver1;
        PetscCall(PetscInfo((PetscObject)B,"Using AVX2 for MatMult for BAIJ for blocksize %" PetscInt_FMT "\n",bs));
        break;
#endif
      default:
        B->ops->mult    = MatMult_SeqBAIJ_N;
        B->ops->multadd = MatMultAdd_SeqBAIJ_N;
        PetscCall(PetscInfo((PetscObject)B,"Using BLAS for MatMult for BAIJ for blocksize %" PetscInt_FMT "\n",bs));
        break;
      }
      break;
    }
    case 15:
    {
      PetscInt version = 1;
      PetscCall(PetscOptionsGetInt(NULL,((PetscObject)B)->prefix,"-mat_baij_mult_version",&version,NULL));
      switch (version) {
      case 1:
        B->ops->mult    = MatMult_SeqBAIJ_15_ver1;
        PetscCall(PetscInfo((PetscObject)B,"Using version %" PetscInt_FMT " of MatMult for BAIJ for blocksize %" PetscInt_FMT "\n",version,bs));
        break;
      case 2:
        B->ops->mult    = MatMult_SeqBAIJ_15_ver2;
        PetscCall(PetscInfo((PetscObject)B,"Using version %" PetscInt_FMT " of MatMult for BAIJ for blocksize %" PetscInt_FMT "\n",version,bs));
        break;
      case 3:
        B->ops->mult    = MatMult_SeqBAIJ_15_ver3;
        PetscCall(PetscInfo((PetscObject)B,"Using version %" PetscInt_FMT " of MatMult for BAIJ for blocksize %" PetscInt_FMT "\n",version,bs));
        break;
      case 4:
        B->ops->mult    = MatMult_SeqBAIJ_15_ver4;
        PetscCall(PetscInfo((PetscObject)B,"Using version %" PetscInt_FMT " of MatMult for BAIJ for blocksize %" PetscInt_FMT "\n",version,bs));
        break;
      default:
        B->ops->mult    = MatMult_SeqBAIJ_N;
        PetscCall(PetscInfo((PetscObject)B,"Using BLAS for MatMult for BAIJ for blocksize %" PetscInt_FMT "\n",bs));
        break;
      }
      B->ops->multadd = MatMultAdd_SeqBAIJ_N;
      break;
    }
    default:
      B->ops->mult    = MatMult_SeqBAIJ_N;
      B->ops->multadd = MatMultAdd_SeqBAIJ_N;
      PetscCall(PetscInfo((PetscObject)B,"Using BLAS for MatMult for BAIJ for blocksize %" PetscInt_FMT "\n",bs));
      break;
    }
  }
  B->ops->sor = MatSOR_SeqBAIJ;
  b->mbs = mbs;
  b->nbs = nbs;
  if (!skipallocation) {
    if (!b->imax) {
      PetscCall(PetscMalloc2(mbs,&b->imax,mbs,&b->ilen));
      PetscCall(PetscLogObjectMemory((PetscObject)B,2*mbs*sizeof(PetscInt)));

      b->free_imax_ilen = PETSC_TRUE;
    }
    /* b->ilen will count nonzeros in each block row so far. */
    for (i=0; i<mbs; i++) b->ilen[i] = 0;
    if (!nnz) {
      if (nz == PETSC_DEFAULT || nz == PETSC_DECIDE) nz = 5;
      else if (nz < 0) nz = 1;
      nz = PetscMin(nz,nbs);
      for (i=0; i<mbs; i++) b->imax[i] = nz;
      PetscCall(PetscIntMultError(nz,mbs,&nz));
    } else {
      PetscInt64 nz64 = 0;
      for (i=0; i<mbs; i++) {b->imax[i] = nnz[i]; nz64 += nnz[i];}
      PetscCall(PetscIntCast(nz64,&nz));
    }

    /* allocate the matrix space */
    PetscCall(MatSeqXAIJFreeAIJ(B,&b->a,&b->j,&b->i));
    if (B->structure_only) {
      PetscCall(PetscMalloc1(nz,&b->j));
      PetscCall(PetscMalloc1(B->rmap->N+1,&b->i));
      PetscCall(PetscLogObjectMemory((PetscObject)B,(B->rmap->N+1)*sizeof(PetscInt)+nz*sizeof(PetscInt)));
    } else {
      PetscInt nzbs2 = 0;
      PetscCall(PetscIntMultError(nz,bs2,&nzbs2));
      PetscCall(PetscMalloc3(nzbs2,&b->a,nz,&b->j,B->rmap->N+1,&b->i));
      PetscCall(PetscLogObjectMemory((PetscObject)B,(B->rmap->N+1)*sizeof(PetscInt)+nz*(bs2*sizeof(PetscScalar)+sizeof(PetscInt))));
      PetscCall(PetscArrayzero(b->a,nz*bs2));
    }
    PetscCall(PetscArrayzero(b->j,nz));

    if (B->structure_only) {
      b->singlemalloc = PETSC_FALSE;
      b->free_a       = PETSC_FALSE;
    } else {
      b->singlemalloc = PETSC_TRUE;
      b->free_a       = PETSC_TRUE;
    }
    b->free_ij = PETSC_TRUE;

    b->i[0] = 0;
    for (i=1; i<mbs+1; i++) {
      b->i[i] = b->i[i-1] + b->imax[i-1];
    }

  } else {
    b->free_a  = PETSC_FALSE;
    b->free_ij = PETSC_FALSE;
  }

  b->bs2              = bs2;
  b->mbs              = mbs;
  b->nz               = 0;
  b->maxnz            = nz;
  B->info.nz_unneeded = (PetscReal)b->maxnz*bs2;
  B->was_assembled    = PETSC_FALSE;
  B->assembled        = PETSC_FALSE;
  if (realalloc) PetscCall(MatSetOption(B,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE));
  PetscFunctionReturn(0);
}

PetscErrorCode MatSeqBAIJSetPreallocationCSR_SeqBAIJ(Mat B,PetscInt bs,const PetscInt ii[],const PetscInt jj[],const PetscScalar V[])
{
  PetscInt       i,m,nz,nz_max=0,*nnz;
  PetscScalar    *values=NULL;
  PetscBool      roworiented = ((Mat_SeqBAIJ*)B->data)->roworiented;

  PetscFunctionBegin;
  PetscCheck(bs >= 1,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Invalid block size specified, must be positive but it is %" PetscInt_FMT,bs);
  PetscCall(PetscLayoutSetBlockSize(B->rmap,bs));
  PetscCall(PetscLayoutSetBlockSize(B->cmap,bs));
  PetscCall(PetscLayoutSetUp(B->rmap));
  PetscCall(PetscLayoutSetUp(B->cmap));
  PetscCall(PetscLayoutGetBlockSize(B->rmap,&bs));
  m    = B->rmap->n/bs;

  PetscCheck(ii[0] == 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE, "ii[0] must be 0 but it is %" PetscInt_FMT,ii[0]);
  PetscCall(PetscMalloc1(m+1, &nnz));
  for (i=0; i<m; i++) {
    nz = ii[i+1]- ii[i];
    PetscCheck(nz >= 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE, "Local row %" PetscInt_FMT " has a negative number of columns %" PetscInt_FMT,i,nz);
    nz_max = PetscMax(nz_max, nz);
    nnz[i] = nz;
  }
  PetscCall(MatSeqBAIJSetPreallocation(B,bs,0,nnz));
  PetscCall(PetscFree(nnz));

  values = (PetscScalar*)V;
  if (!values) {
    PetscCall(PetscCalloc1(bs*bs*(nz_max+1),&values));
  }
  for (i=0; i<m; i++) {
    PetscInt          ncols  = ii[i+1] - ii[i];
    const PetscInt    *icols = jj + ii[i];
    if (bs == 1 || !roworiented) {
      const PetscScalar *svals = values + (V ? (bs*bs*ii[i]) : 0);
      PetscCall(MatSetValuesBlocked_SeqBAIJ(B,1,&i,ncols,icols,svals,INSERT_VALUES));
    } else {
      PetscInt j;
      for (j=0; j<ncols; j++) {
        const PetscScalar *svals = values + (V ? (bs*bs*(ii[i]+j)) : 0);
        PetscCall(MatSetValuesBlocked_SeqBAIJ(B,1,&i,1,&icols[j],svals,INSERT_VALUES));
      }
    }
  }
  if (!V) PetscCall(PetscFree(values));
  PetscCall(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  PetscCall(MatSetOption(B,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE));
  PetscFunctionReturn(0);
}

/*@C
   MatSeqBAIJGetArray - gives access to the array where the data for a MATSEQBAIJ matrix is stored

   Not Collective

   Input Parameter:
.  mat - a MATSEQBAIJ matrix

   Output Parameter:
.   array - pointer to the data

   Level: intermediate

.seealso: `MatSeqBAIJRestoreArray()`, `MatSeqAIJGetArray()`, `MatSeqAIJRestoreArray()`
@*/
PetscErrorCode MatSeqBAIJGetArray(Mat A,PetscScalar **array)
{
  PetscFunctionBegin;
  PetscUseMethod(A,"MatSeqBAIJGetArray_C",(Mat,PetscScalar**),(A,array));
  PetscFunctionReturn(0);
}

/*@C
   MatSeqBAIJRestoreArray - returns access to the array where the data for a MATSEQBAIJ matrix is stored obtained by MatSeqBAIJGetArray()

   Not Collective

   Input Parameters:
+  mat - a MATSEQBAIJ matrix
-  array - pointer to the data

   Level: intermediate

.seealso: `MatSeqBAIJGetArray()`, `MatSeqAIJGetArray()`, `MatSeqAIJRestoreArray()`
@*/
PetscErrorCode MatSeqBAIJRestoreArray(Mat A,PetscScalar **array)
{
  PetscFunctionBegin;
  PetscUseMethod(A,"MatSeqBAIJRestoreArray_C",(Mat,PetscScalar**),(A,array));
  PetscFunctionReturn(0);
}

/*MC
   MATSEQBAIJ - MATSEQBAIJ = "seqbaij" - A matrix type to be used for sequential block sparse matrices, based on
   block sparse compressed row format.

   Options Database Keys:
+ -mat_type seqbaij - sets the matrix type to "seqbaij" during a call to MatSetFromOptions()
- -mat_baij_mult_version version - indicate the version of the matrix-vector product to use (0 often indicates using BLAS)

   Level: beginner

   Notes:
    MatSetOptions(,MAT_STRUCTURE_ONLY,PETSC_TRUE) may be called for this matrix type. In this no
    space is allocated for the nonzero entries and any entries passed with MatSetValues() are ignored

   Run with -info to see what version of the matrix-vector product is being used

.seealso: `MatCreateSeqBAIJ()`
M*/

PETSC_INTERN PetscErrorCode MatConvert_SeqBAIJ_SeqBSTRM(Mat, MatType,MatReuse,Mat*);

PETSC_EXTERN PetscErrorCode MatCreate_SeqBAIJ(Mat B)
{
  PetscMPIInt    size;
  Mat_SeqBAIJ    *b;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)B),&size));
  PetscCheck(size == 1,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Comm must be of size 1");

  PetscCall(PetscNewLog(B,&b));
  B->data = (void*)b;
  PetscCall(PetscMemcpy(B->ops,&MatOps_Values,sizeof(struct _MatOps)));

  b->row          = NULL;
  b->col          = NULL;
  b->icol         = NULL;
  b->reallocs     = 0;
  b->saved_values = NULL;

  b->roworiented        = PETSC_TRUE;
  b->nonew              = 0;
  b->diag               = NULL;
  B->spptr              = NULL;
  B->info.nz_unneeded   = (PetscReal)b->maxnz*b->bs2;
  b->keepnonzeropattern = PETSC_FALSE;

  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatSeqBAIJGetArray_C",MatSeqBAIJGetArray_SeqBAIJ));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatSeqBAIJRestoreArray_C",MatSeqBAIJRestoreArray_SeqBAIJ));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatStoreValues_C",MatStoreValues_SeqBAIJ));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatRetrieveValues_C",MatRetrieveValues_SeqBAIJ));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatSeqBAIJSetColumnIndices_C",MatSeqBAIJSetColumnIndices_SeqBAIJ));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatConvert_seqbaij_seqaij_C",MatConvert_SeqBAIJ_SeqAIJ));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatConvert_seqbaij_seqsbaij_C",MatConvert_SeqBAIJ_SeqSBAIJ));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatSeqBAIJSetPreallocation_C",MatSeqBAIJSetPreallocation_SeqBAIJ));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatSeqBAIJSetPreallocationCSR_C",MatSeqBAIJSetPreallocationCSR_SeqBAIJ));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatIsTranspose_C",MatIsTranspose_SeqBAIJ));
#if defined(PETSC_HAVE_HYPRE)
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatConvert_seqbaij_hypre_C",MatConvert_AIJ_HYPRE));
#endif
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatConvert_seqbaij_is_C",MatConvert_XAIJ_IS));
  PetscCall(PetscObjectChangeTypeName((PetscObject)B,MATSEQBAIJ));
  PetscFunctionReturn(0);
}

PetscErrorCode MatDuplicateNoCreate_SeqBAIJ(Mat C,Mat A,MatDuplicateOption cpvalues,PetscBool mallocmatspace)
{
  Mat_SeqBAIJ    *c = (Mat_SeqBAIJ*)C->data,*a = (Mat_SeqBAIJ*)A->data;
  PetscInt       i,mbs = a->mbs,nz = a->nz,bs2 = a->bs2;

  PetscFunctionBegin;
  PetscCheck(a->i[mbs] == nz,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Corrupt matrix");

  if (cpvalues == MAT_SHARE_NONZERO_PATTERN) {
    c->imax           = a->imax;
    c->ilen           = a->ilen;
    c->free_imax_ilen = PETSC_FALSE;
  } else {
    PetscCall(PetscMalloc2(mbs,&c->imax,mbs,&c->ilen));
    PetscCall(PetscLogObjectMemory((PetscObject)C,2*mbs*sizeof(PetscInt)));
    for (i=0; i<mbs; i++) {
      c->imax[i] = a->imax[i];
      c->ilen[i] = a->ilen[i];
    }
    c->free_imax_ilen = PETSC_TRUE;
  }

  /* allocate the matrix space */
  if (mallocmatspace) {
    if (cpvalues == MAT_SHARE_NONZERO_PATTERN) {
      PetscCall(PetscCalloc1(bs2*nz,&c->a));
      PetscCall(PetscLogObjectMemory((PetscObject)C,a->i[mbs]*bs2*sizeof(PetscScalar)));

      c->i            = a->i;
      c->j            = a->j;
      c->singlemalloc = PETSC_FALSE;
      c->free_a       = PETSC_TRUE;
      c->free_ij      = PETSC_FALSE;
      c->parent       = A;
      C->preallocated = PETSC_TRUE;
      C->assembled    = PETSC_TRUE;

      PetscCall(PetscObjectReference((PetscObject)A));
      PetscCall(MatSetOption(A,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE));
      PetscCall(MatSetOption(C,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE));
    } else {
      PetscCall(PetscMalloc3(bs2*nz,&c->a,nz,&c->j,mbs+1,&c->i));
      PetscCall(PetscLogObjectMemory((PetscObject)C,a->i[mbs]*(bs2*sizeof(PetscScalar)+sizeof(PetscInt))+(mbs+1)*sizeof(PetscInt)));

      c->singlemalloc = PETSC_TRUE;
      c->free_a       = PETSC_TRUE;
      c->free_ij      = PETSC_TRUE;

      PetscCall(PetscArraycpy(c->i,a->i,mbs+1));
      if (mbs > 0) {
        PetscCall(PetscArraycpy(c->j,a->j,nz));
        if (cpvalues == MAT_COPY_VALUES) {
          PetscCall(PetscArraycpy(c->a,a->a,bs2*nz));
        } else {
          PetscCall(PetscArrayzero(c->a,bs2*nz));
        }
      }
      C->preallocated = PETSC_TRUE;
      C->assembled    = PETSC_TRUE;
    }
  }

  c->roworiented = a->roworiented;
  c->nonew       = a->nonew;

  PetscCall(PetscLayoutReference(A->rmap,&C->rmap));
  PetscCall(PetscLayoutReference(A->cmap,&C->cmap));

  c->bs2         = a->bs2;
  c->mbs         = a->mbs;
  c->nbs         = a->nbs;

  if (a->diag) {
    if (cpvalues == MAT_SHARE_NONZERO_PATTERN) {
      c->diag      = a->diag;
      c->free_diag = PETSC_FALSE;
    } else {
      PetscCall(PetscMalloc1(mbs+1,&c->diag));
      PetscCall(PetscLogObjectMemory((PetscObject)C,(mbs+1)*sizeof(PetscInt)));
      for (i=0; i<mbs; i++) c->diag[i] = a->diag[i];
      c->free_diag = PETSC_TRUE;
    }
  } else c->diag = NULL;

  c->nz         = a->nz;
  c->maxnz      = a->nz;         /* Since we allocate exactly the right amount */
  c->solve_work = NULL;
  c->mult_work  = NULL;
  c->sor_workt  = NULL;
  c->sor_work   = NULL;

  c->compressedrow.use   = a->compressedrow.use;
  c->compressedrow.nrows = a->compressedrow.nrows;
  if (a->compressedrow.use) {
    i    = a->compressedrow.nrows;
    PetscCall(PetscMalloc2(i+1,&c->compressedrow.i,i+1,&c->compressedrow.rindex));
    PetscCall(PetscLogObjectMemory((PetscObject)C,(2*i+1)*sizeof(PetscInt)));
    PetscCall(PetscArraycpy(c->compressedrow.i,a->compressedrow.i,i+1));
    PetscCall(PetscArraycpy(c->compressedrow.rindex,a->compressedrow.rindex,i));
  } else {
    c->compressedrow.use    = PETSC_FALSE;
    c->compressedrow.i      = NULL;
    c->compressedrow.rindex = NULL;
  }
  C->nonzerostate = A->nonzerostate;

  PetscCall(PetscFunctionListDuplicate(((PetscObject)A)->qlist,&((PetscObject)C)->qlist));
  PetscFunctionReturn(0);
}

PetscErrorCode MatDuplicate_SeqBAIJ(Mat A,MatDuplicateOption cpvalues,Mat *B)
{
  PetscFunctionBegin;
  PetscCall(MatCreate(PetscObjectComm((PetscObject)A),B));
  PetscCall(MatSetSizes(*B,A->rmap->N,A->cmap->n,A->rmap->N,A->cmap->n));
  PetscCall(MatSetType(*B,MATSEQBAIJ));
  PetscCall(MatDuplicateNoCreate_SeqBAIJ(*B,A,cpvalues,PETSC_TRUE));
  PetscFunctionReturn(0);
}

/* Used for both SeqBAIJ and SeqSBAIJ matrices */
PetscErrorCode MatLoad_SeqBAIJ_Binary(Mat mat,PetscViewer viewer)
{
  PetscInt       header[4],M,N,nz,bs,m,n,mbs,nbs,rows,cols,sum,i,j,k;
  PetscInt       *rowidxs,*colidxs;
  PetscScalar    *matvals;

  PetscFunctionBegin;
  PetscCall(PetscViewerSetUp(viewer));

  /* read matrix header */
  PetscCall(PetscViewerBinaryRead(viewer,header,4,NULL,PETSC_INT));
  PetscCheck(header[0] == MAT_FILE_CLASSID,PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Not a matrix object in file");
  M = header[1]; N = header[2]; nz = header[3];
  PetscCheck(M >= 0,PetscObjectComm((PetscObject)viewer),PETSC_ERR_FILE_UNEXPECTED,"Matrix row size (%" PetscInt_FMT ") in file is negative",M);
  PetscCheck(N >= 0,PetscObjectComm((PetscObject)viewer),PETSC_ERR_FILE_UNEXPECTED,"Matrix column size (%" PetscInt_FMT ") in file is negative",N);
  PetscCheck(nz >= 0,PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Matrix stored in special format on disk, cannot load as SeqBAIJ");

  /* set block sizes from the viewer's .info file */
  PetscCall(MatLoad_Binary_BlockSizes(mat,viewer));
  /* set local and global sizes if not set already */
  if (mat->rmap->n < 0) mat->rmap->n = M;
  if (mat->cmap->n < 0) mat->cmap->n = N;
  if (mat->rmap->N < 0) mat->rmap->N = M;
  if (mat->cmap->N < 0) mat->cmap->N = N;
  PetscCall(PetscLayoutSetUp(mat->rmap));
  PetscCall(PetscLayoutSetUp(mat->cmap));

  /* check if the matrix sizes are correct */
  PetscCall(MatGetSize(mat,&rows,&cols));
  PetscCheck(M == rows && N == cols,PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED, "Matrix in file of different sizes (%" PetscInt_FMT ", %" PetscInt_FMT ") than the input matrix (%" PetscInt_FMT ", %" PetscInt_FMT ")",M,N,rows,cols);
  PetscCall(MatGetBlockSize(mat,&bs));
  PetscCall(MatGetLocalSize(mat,&m,&n));
  mbs = m/bs; nbs = n/bs;

  /* read in row lengths, column indices and nonzero values */
  PetscCall(PetscMalloc1(m+1,&rowidxs));
  PetscCall(PetscViewerBinaryRead(viewer,rowidxs+1,m,NULL,PETSC_INT));
  rowidxs[0] = 0; for (i=0; i<m; i++) rowidxs[i+1] += rowidxs[i];
  sum = rowidxs[m];
  PetscCheck(sum == nz,PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Inconsistent matrix data in file: nonzeros = %" PetscInt_FMT ", sum-row-lengths = %" PetscInt_FMT,nz,sum);

  /* read in column indices and nonzero values */
  PetscCall(PetscMalloc2(rowidxs[m],&colidxs,nz,&matvals));
  PetscCall(PetscViewerBinaryRead(viewer,colidxs,rowidxs[m],NULL,PETSC_INT));
  PetscCall(PetscViewerBinaryRead(viewer,matvals,rowidxs[m],NULL,PETSC_SCALAR));

  { /* preallocate matrix storage */
    PetscBT   bt; /* helper bit set to count nonzeros */
    PetscInt  *nnz;
    PetscBool sbaij;

    PetscCall(PetscBTCreate(nbs,&bt));
    PetscCall(PetscCalloc1(mbs,&nnz));
    PetscCall(PetscObjectTypeCompare((PetscObject)mat,MATSEQSBAIJ,&sbaij));
    for (i=0; i<mbs; i++) {
      PetscCall(PetscBTMemzero(nbs,bt));
      for (k=0; k<bs; k++) {
        PetscInt row = bs*i + k;
        for (j=rowidxs[row]; j<rowidxs[row+1]; j++) {
          PetscInt col = colidxs[j];
          if (!sbaij || col >= row)
            if (!PetscBTLookupSet(bt,col/bs)) nnz[i]++;
        }
      }
    }
    PetscCall(PetscBTDestroy(&bt));
    PetscCall(MatSeqBAIJSetPreallocation(mat,bs,0,nnz));
    PetscCall(MatSeqSBAIJSetPreallocation(mat,bs,0,nnz));
    PetscCall(PetscFree(nnz));
  }

  /* store matrix values */
  for (i=0; i<m; i++) {
    PetscInt row = i, s = rowidxs[i], e = rowidxs[i+1];
    PetscCall((*mat->ops->setvalues)(mat,1,&row,e-s,colidxs+s,matvals+s,INSERT_VALUES));
  }

  PetscCall(PetscFree(rowidxs));
  PetscCall(PetscFree2(colidxs,matvals));
  PetscCall(MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

PetscErrorCode MatLoad_SeqBAIJ(Mat mat,PetscViewer viewer)
{
  PetscBool isbinary;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary));
  PetscCheck(isbinary,PetscObjectComm((PetscObject)viewer),PETSC_ERR_SUP,"Viewer type %s not yet supported for reading %s matrices",((PetscObject)viewer)->type_name,((PetscObject)mat)->type_name);
  PetscCall(MatLoad_SeqBAIJ_Binary(mat,viewer));
  PetscFunctionReturn(0);
}

/*@C
   MatCreateSeqBAIJ - Creates a sparse matrix in block AIJ (block
   compressed row) format.  For good matrix assembly performance the
   user should preallocate the matrix storage by setting the parameter nz
   (or the array nnz).  By setting these parameters accurately, performance
   during matrix assembly can be increased by more than a factor of 50.

   Collective

   Input Parameters:
+  comm - MPI communicator, set to PETSC_COMM_SELF
.  bs - size of block, the blocks are ALWAYS square. One can use MatSetBlockSizes() to set a different row and column blocksize but the row
          blocksize always defines the size of the blocks. The column blocksize sets the blocksize of the vectors obtained with MatCreateVecs()
.  m - number of rows
.  n - number of columns
.  nz - number of nonzero blocks  per block row (same for all rows)
-  nnz - array containing the number of nonzero blocks in the various block rows
         (possibly different for each block row) or NULL

   Output Parameter:
.  A - the matrix

   It is recommended that one use the MatCreate(), MatSetType() and/or MatSetFromOptions(),
   MatXXXXSetPreallocation() paradigm instead of this routine directly.
   [MatXXXXSetPreallocation() is, for example, MatSeqAIJSetPreallocation]

   Options Database Keys:
+   -mat_no_unroll - uses code that does not unroll the loops in the
                     block calculations (much slower)
-    -mat_block_size - size of the blocks to use

   Level: intermediate

   Notes:
   The number of rows and columns must be divisible by blocksize.

   If the nnz parameter is given then the nz parameter is ignored

   A nonzero block is any block that as 1 or more nonzeros in it

   The block AIJ format is fully compatible with standard Fortran 77
   storage.  That is, the stored row and column indices can begin at
   either one (as in Fortran) or zero.  See the users' manual for details.

   Specify the preallocated storage with either nz or nnz (not both).
   Set nz=PETSC_DEFAULT and nnz=NULL for PETSc to control dynamic memory
   allocation.  See Users-Manual: ch_mat for details.
   matrices.

.seealso: `MatCreate()`, `MatCreateSeqAIJ()`, `MatSetValues()`, `MatCreateBAIJ()`
@*/
PetscErrorCode  MatCreateSeqBAIJ(MPI_Comm comm,PetscInt bs,PetscInt m,PetscInt n,PetscInt nz,const PetscInt nnz[],Mat *A)
{
  PetscFunctionBegin;
  PetscCall(MatCreate(comm,A));
  PetscCall(MatSetSizes(*A,m,n,m,n));
  PetscCall(MatSetType(*A,MATSEQBAIJ));
  PetscCall(MatSeqBAIJSetPreallocation(*A,bs,nz,(PetscInt*)nnz));
  PetscFunctionReturn(0);
}

/*@C
   MatSeqBAIJSetPreallocation - Sets the block size and expected nonzeros
   per row in the matrix. For good matrix assembly performance the
   user should preallocate the matrix storage by setting the parameter nz
   (or the array nnz).  By setting these parameters accurately, performance
   during matrix assembly can be increased by more than a factor of 50.

   Collective

   Input Parameters:
+  B - the matrix
.  bs - size of block, the blocks are ALWAYS square. One can use MatSetBlockSizes() to set a different row and column blocksize but the row
          blocksize always defines the size of the blocks. The column blocksize sets the blocksize of the vectors obtained with MatCreateVecs()
.  nz - number of block nonzeros per block row (same for all rows)
-  nnz - array containing the number of block nonzeros in the various block rows
         (possibly different for each block row) or NULL

   Options Database Keys:
+   -mat_no_unroll - uses code that does not unroll the loops in the
                     block calculations (much slower)
-   -mat_block_size - size of the blocks to use

   Level: intermediate

   Notes:
   If the nnz parameter is given then the nz parameter is ignored

   You can call MatGetInfo() to get information on how effective the preallocation was;
   for example the fields mallocs,nz_allocated,nz_used,nz_unneeded;
   You can also run with the option -info and look for messages with the string
   malloc in them to see if additional memory allocation was needed.

   The block AIJ format is fully compatible with standard Fortran 77
   storage.  That is, the stored row and column indices can begin at
   either one (as in Fortran) or zero.  See the users' manual for details.

   Specify the preallocated storage with either nz or nnz (not both).
   Set nz=PETSC_DEFAULT and nnz=NULL for PETSc to control dynamic memory
   allocation.  See Users-Manual: ch_mat for details.

.seealso: `MatCreate()`, `MatCreateSeqAIJ()`, `MatSetValues()`, `MatCreateBAIJ()`, `MatGetInfo()`
@*/
PetscErrorCode  MatSeqBAIJSetPreallocation(Mat B,PetscInt bs,PetscInt nz,const PetscInt nnz[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(B,MAT_CLASSID,1);
  PetscValidType(B,1);
  PetscValidLogicalCollectiveInt(B,bs,2);
  PetscTryMethod(B,"MatSeqBAIJSetPreallocation_C",(Mat,PetscInt,PetscInt,const PetscInt[]),(B,bs,nz,nnz));
  PetscFunctionReturn(0);
}

/*@C
   MatSeqBAIJSetPreallocationCSR - Creates a sparse parallel matrix in BAIJ format using the given nonzero structure and (optional) numerical values

   Collective

   Input Parameters:
+  B - the matrix
.  i - the indices into j for the start of each local row (starts with zero)
.  j - the column indices for each local row (starts with zero) these must be sorted for each row
-  v - optional values in the matrix

   Level: advanced

   Notes:
   The order of the entries in values is specified by the MatOption MAT_ROW_ORIENTED.  For example, C programs
   may want to use the default MAT_ROW_ORIENTED=PETSC_TRUE and use an array v[nnz][bs][bs] where the second index is
   over rows within a block and the last index is over columns within a block row.  Fortran programs will likely set
   MAT_ROW_ORIENTED=PETSC_FALSE and use a Fortran array v(bs,bs,nnz) in which the first index is over rows within a
   block column and the second index is over columns within a block.

   Though this routine has Preallocation() in the name it also sets the exact nonzero locations of the matrix entries and usually the numerical values as well

.seealso: `MatCreate()`, `MatCreateSeqBAIJ()`, `MatSetValues()`, `MatSeqBAIJSetPreallocation()`, `MATSEQBAIJ`
@*/
PetscErrorCode  MatSeqBAIJSetPreallocationCSR(Mat B,PetscInt bs,const PetscInt i[],const PetscInt j[], const PetscScalar v[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(B,MAT_CLASSID,1);
  PetscValidType(B,1);
  PetscValidLogicalCollectiveInt(B,bs,2);
  PetscTryMethod(B,"MatSeqBAIJSetPreallocationCSR_C",(Mat,PetscInt,const PetscInt[],const PetscInt[],const PetscScalar[]),(B,bs,i,j,v));
  PetscFunctionReturn(0);
}

/*@
     MatCreateSeqBAIJWithArrays - Creates an sequential BAIJ matrix using matrix elements provided by the user.

     Collective

   Input Parameters:
+  comm - must be an MPI communicator of size 1
.  bs - size of block
.  m - number of rows
.  n - number of columns
.  i - row indices; that is i[0] = 0, i[row] = i[row-1] + number of elements in that row block row of the matrix
.  j - column indices
-  a - matrix values

   Output Parameter:
.  mat - the matrix

   Level: advanced

   Notes:
       The i, j, and a arrays are not copied by this routine, the user must free these arrays
    once the matrix is destroyed

       You cannot set new nonzero locations into this matrix, that will generate an error.

       The i and j indices are 0 based

       When block size is greater than 1 the matrix values must be stored using the BAIJ storage format (see the BAIJ code to determine this).

      The order of the entries in values is the same as the block compressed sparse row storage format; that is, it is
      the same as a three dimensional array in Fortran values(bs,bs,nnz) that contains the first column of the first
      block, followed by the second column of the first block etc etc.  That is, the blocks are contiguous in memory
      with column-major ordering within blocks.

.seealso: `MatCreate()`, `MatCreateBAIJ()`, `MatCreateSeqBAIJ()`

@*/
PetscErrorCode  MatCreateSeqBAIJWithArrays(MPI_Comm comm,PetscInt bs,PetscInt m,PetscInt n,PetscInt i[],PetscInt j[],PetscScalar a[],Mat *mat)
{
  PetscInt       ii;
  Mat_SeqBAIJ    *baij;

  PetscFunctionBegin;
  PetscCheck(bs == 1,PETSC_COMM_SELF,PETSC_ERR_SUP,"block size %" PetscInt_FMT " > 1 is not supported yet",bs);
  if (m > 0) PetscCheck(i[0] == 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"i (row indices) must start with 0");

  PetscCall(MatCreate(comm,mat));
  PetscCall(MatSetSizes(*mat,m,n,m,n));
  PetscCall(MatSetType(*mat,MATSEQBAIJ));
  PetscCall(MatSeqBAIJSetPreallocation(*mat,bs,MAT_SKIP_ALLOCATION,NULL));
  baij = (Mat_SeqBAIJ*)(*mat)->data;
  PetscCall(PetscMalloc2(m,&baij->imax,m,&baij->ilen));
  PetscCall(PetscLogObjectMemory((PetscObject)*mat,2*m*sizeof(PetscInt)));

  baij->i = i;
  baij->j = j;
  baij->a = a;

  baij->singlemalloc = PETSC_FALSE;
  baij->nonew        = -1;             /*this indicates that inserting a new value in the matrix that generates a new nonzero is an error*/
  baij->free_a       = PETSC_FALSE;
  baij->free_ij      = PETSC_FALSE;

  for (ii=0; ii<m; ii++) {
    baij->ilen[ii] = baij->imax[ii] = i[ii+1] - i[ii];
    PetscCheck(i[ii+1] - i[ii] >= 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Negative row length in i (row indices) row = %" PetscInt_FMT " length = %" PetscInt_FMT,ii,i[ii+1] - i[ii]);
  }
  if (PetscDefined(USE_DEBUG)) {
    for (ii=0; ii<baij->i[m]; ii++) {
      PetscCheck(j[ii] >= 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Negative column index at location = %" PetscInt_FMT " index = %" PetscInt_FMT,ii,j[ii]);
      PetscCheck(j[ii] <= n - 1,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Column index to large at location = %" PetscInt_FMT " index = %" PetscInt_FMT,ii,j[ii]);
    }
  }

  PetscCall(MatAssemblyBegin(*mat,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*mat,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

PetscErrorCode MatCreateMPIMatConcatenateSeqMat_SeqBAIJ(MPI_Comm comm,Mat inmat,PetscInt n,MatReuse scall,Mat *outmat)
{
  PetscMPIInt    size;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(comm,&size));
  if (size == 1 && scall == MAT_REUSE_MATRIX) {
    PetscCall(MatCopy(inmat,*outmat,SAME_NONZERO_PATTERN));
  } else {
    PetscCall(MatCreateMPIMatConcatenateSeqMat_MPIBAIJ(comm,inmat,n,scall,outmat));
  }
  PetscFunctionReturn(0);
}
