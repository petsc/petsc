
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
  PetscErrorCode ierr;
  Mat_SeqBAIJ    *a_aij = (Mat_SeqBAIJ*) A->data;
  PetscInt       m,n,i;
  PetscInt       ib,jb,bs = A->rmap->bs;
  MatScalar      *a_val = a_aij->a;

  PetscFunctionBegin;
  ierr = MatGetSize(A,&m,&n);CHKERRQ(ierr);
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
  PetscErrorCode ierr;
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
  ierr        = MatMarkDiagonal_SeqBAIJ(A);CHKERRQ(ierr);
  diag_offset = a->diag;
  if (!a->idiag) {
    ierr = PetscMalloc1(bs2*mbs,&a->idiag);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)A,bs2*mbs*sizeof(PetscScalar));CHKERRQ(ierr);
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
          ierr = PetscInfo1(A,"Zero pivot, row %" PetscInt_FMT "\n",i);CHKERRQ(ierr);
        } else SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_MAT_LU_ZRPVT,"Zero pivot, row %" PetscInt_FMT " pivot value %g tolerance %g",i,(double)PetscAbsScalar(diag[0]),(double)PETSC_MACHINE_EPSILON);
      }

      diag[0]  = (PetscScalar)1.0 / (diag[0] + shift);
      diag    += 1;
    }
    break;
  case 2:
    for (i=0; i<mbs; i++) {
      odiag    = v + 4*diag_offset[i];
      diag[0]  = odiag[0]; diag[1] = odiag[1]; diag[2] = odiag[2]; diag[3] = odiag[3];
      ierr     = PetscKernel_A_gets_inverse_A_2(diag,shift,allowzeropivot,&zeropivotdetected);CHKERRQ(ierr);
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
      ierr     = PetscKernel_A_gets_inverse_A_3(diag,shift,allowzeropivot,&zeropivotdetected);CHKERRQ(ierr);
      if (zeropivotdetected) A->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;
      diag    += 9;
    }
    break;
  case 4:
    for (i=0; i<mbs; i++) {
      odiag  = v + 16*diag_offset[i];
      ierr   = PetscArraycpy(diag,odiag,16);CHKERRQ(ierr);
      ierr   = PetscKernel_A_gets_inverse_A_4(diag,shift,allowzeropivot,&zeropivotdetected);CHKERRQ(ierr);
      if (zeropivotdetected) A->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;
      diag  += 16;
    }
    break;
  case 5:
    for (i=0; i<mbs; i++) {
      odiag  = v + 25*diag_offset[i];
      ierr   = PetscArraycpy(diag,odiag,25);CHKERRQ(ierr);
      ierr   = PetscKernel_A_gets_inverse_A_5(diag,ipvt,work,shift,allowzeropivot,&zeropivotdetected);CHKERRQ(ierr);
      if (zeropivotdetected) A->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;
      diag  += 25;
    }
    break;
  case 6:
    for (i=0; i<mbs; i++) {
      odiag  = v + 36*diag_offset[i];
      ierr   = PetscArraycpy(diag,odiag,36);CHKERRQ(ierr);
      ierr   = PetscKernel_A_gets_inverse_A_6(diag,shift,allowzeropivot,&zeropivotdetected);CHKERRQ(ierr);
      if (zeropivotdetected) A->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;
      diag  += 36;
    }
    break;
  case 7:
    for (i=0; i<mbs; i++) {
      odiag  = v + 49*diag_offset[i];
      ierr   = PetscArraycpy(diag,odiag,49);CHKERRQ(ierr);
      ierr   = PetscKernel_A_gets_inverse_A_7(diag,shift,allowzeropivot,&zeropivotdetected);CHKERRQ(ierr);
      if (zeropivotdetected) A->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;
      diag  += 49;
    }
    break;
  default:
    ierr = PetscMalloc2(bs,&v_work,bs,&v_pivots);CHKERRQ(ierr);
    for (i=0; i<mbs; i++) {
      odiag  = v + bs2*diag_offset[i];
      ierr   = PetscArraycpy(diag,odiag,bs2);CHKERRQ(ierr);
      ierr   = PetscKernel_A_gets_inverse_A(bs,diag,v_pivots,v_work,allowzeropivot,&zeropivotdetected);CHKERRQ(ierr);
      if (zeropivotdetected) A->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;
      diag  += bs2;
    }
    ierr = PetscFree2(v_work,v_pivots);CHKERRQ(ierr);
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
  PetscErrorCode    ierr;
  PetscInt          m = a->mbs,i,i2,nz,bs = A->rmap->bs,bs2 = bs*bs,k,j,idx,it;
  const PetscInt    *diag,*ai = a->i,*aj = a->j,*vi;

  PetscFunctionBegin;
  its = its*lits;
  if (flag & SOR_EISENSTAT) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support yet for Eisenstat");
  if (its <= 0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Relaxation requires global its %" PetscInt_FMT " and local its %" PetscInt_FMT " both positive",its,lits);
  if (fshift) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for diagonal shift");
  if (omega != 1.0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for non-trivial relaxation factor");
  if ((flag & SOR_APPLY_UPPER) || (flag & SOR_APPLY_LOWER)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for applying upper or lower triangular parts");

  if (!a->idiagvalid) {ierr = MatInvertBlockDiagonal(A,NULL);CHKERRQ(ierr);}

  if (!m) PetscFunctionReturn(0);
  diag  = a->diag;
  idiag = a->idiag;
  k    = PetscMax(A->rmap->n,A->cmap->n);
  if (!a->mult_work) {
    ierr = PetscMalloc1(k+1,&a->mult_work);CHKERRQ(ierr);
  }
  if (!a->sor_workt) {
    ierr = PetscMalloc1(k,&a->sor_workt);CHKERRQ(ierr);
  }
  if (!a->sor_work) {
    ierr = PetscMalloc1(bs,&a->sor_work);CHKERRQ(ierr);
  }
  work = a->mult_work;
  t    = a->sor_workt;
  w    = a->sor_work;

  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArrayRead(bb,&b);CHKERRQ(ierr);

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
        ierr = PetscArraycpy(t,b,bs);CHKERRQ(ierr);
        i2     = bs;
        idiag += bs2;
        for (i=1; i<m; i++) {
          v  = aa + bs2*ai[i];
          vi = aj + ai[i];
          nz = diag[i] - ai[i];

          ierr = PetscArraycpy(w,b+i2,bs);CHKERRQ(ierr);
          /* copy all rows of x that are needed into contiguous space */
          workt = work;
          for (j=0; j<nz; j++) {
            ierr   = PetscArraycpy(workt,x + bs*(*vi++),bs);CHKERRQ(ierr);
            workt += bs;
          }
          PetscKernel_w_gets_w_minus_Ar_times_v(bs,bs*nz,w,v,work);
          ierr = PetscArraycpy(t+i2,w,bs);CHKERRQ(ierr);
          PetscKernel_w_gets_Ar_times_v(bs,bs,w,idiag,x+i2);

          idiag += bs2;
          i2    += bs;
        }
        break;
      }
      /* for logging purposes assume number of nonzero in lower half is 1/2 of total */
      ierr = PetscLogFlops(1.0*bs2*a->nz);CHKERRQ(ierr);
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
        ierr  = PetscArraycpy(w,xb+i2,bs);CHKERRQ(ierr);
        PetscKernel_w_gets_Ar_times_v(bs,bs,w,idiag,x+i2);
        i2    -= bs;
        idiag -= bs2;
        for (i=m-2; i>=0; i--) {
          v  = aa + bs2*(diag[i]+1);
          vi = aj + diag[i] + 1;
          nz = ai[i+1] - diag[i] - 1;

          ierr = PetscArraycpy(w,xb+i2,bs);CHKERRQ(ierr);
          /* copy all rows of x that are needed into contiguous space */
          workt = work;
          for (j=0; j<nz; j++) {
            ierr   = PetscArraycpy(workt,x + bs*(*vi++),bs);CHKERRQ(ierr);
            workt += bs;
          }
          PetscKernel_w_gets_w_minus_Ar_times_v(bs,bs*nz,w,v,work);
          PetscKernel_w_gets_Ar_times_v(bs,bs,w,idiag,x+i2);

          idiag -= bs2;
          i2    -= bs;
        }
        break;
      }
      ierr = PetscLogFlops(1.0*bs2*(a->nz));CHKERRQ(ierr);
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

          ierr = PetscArraycpy(w,b+i2,bs);CHKERRQ(ierr);
          /* copy all rows of x that are needed into contiguous space */
          workt = work;
          for (j=0; j<nz; j++) {
            ierr   = PetscArraycpy(workt,x + bs*(*vi++),bs);CHKERRQ(ierr);
            workt += bs;
          }
          PetscKernel_w_gets_w_minus_Ar_times_v(bs,bs*nz,w,v,work);
          PetscKernel_w_gets_w_plus_Ar_times_v(bs,bs,w,idiag,x+i2);

          idiag += bs2;
          i2    += bs;
        }
        break;
      }
      ierr = PetscLogFlops(2.0*bs2*a->nz);CHKERRQ(ierr);
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

          ierr = PetscArraycpy(w,b+i2,bs);CHKERRQ(ierr);
          /* copy all rows of x that are needed into contiguous space */
          workt = work;
          for (j=0; j<nz; j++) {
            ierr   = PetscArraycpy(workt,x + bs*(*vi++),bs);CHKERRQ(ierr);
            workt += bs;
          }
          PetscKernel_w_gets_w_minus_Ar_times_v(bs,bs*nz,w,v,work);
          PetscKernel_w_gets_w_plus_Ar_times_v(bs,bs,w,idiag,x+i2);

          idiag -= bs2;
          i2    -= bs;
        }
        break;
      }
      ierr = PetscLogFlops(2.0*bs2*(a->nz));CHKERRQ(ierr);
    }
  }
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(bb,&b);CHKERRQ(ierr);
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
  PetscErrorCode    ierr;

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
        ierr = PetscArraycpy(ap+16*(ii+1),ap+16*(ii),16);CHKERRV(ierr);
      }
      if (N >= i) {
        ierr = PetscArrayzero(ap+16*i,16);CHKERRV(ierr);
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
  PetscErrorCode ierr;

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
      ierr = PetscArraymove(rp+i+1,rp+i,N-i+1);CHKERRV(ierr);
      ierr = PetscArraymove(ap+16*i+16,ap+16*i,16*(N-i+1));CHKERRV(ierr);
      ierr = PetscArrayzero(ap+16*i,16);CHKERRV(ierr);
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
  PetscErrorCode ierr;
  PetscInt       *diag,*ii = a->i,i;

  PetscFunctionBegin;
  ierr     = MatMarkDiagonal_SeqBAIJ(A);CHKERRQ(ierr);
  *missing = PETSC_FALSE;
  if (A->rmap->n > 0 && !ii) {
    *missing = PETSC_TRUE;
    if (d) *d = 0;
    ierr = PetscInfo(A,"Matrix has no entries therefore is missing diagonal\n");CHKERRQ(ierr);
  } else {
    PetscInt n;
    n = PetscMin(a->mbs, a->nbs);
    diag = a->diag;
    for (i=0; i<n; i++) {
      if (diag[i] >= ii[i+1]) {
        *missing = PETSC_TRUE;
        if (d) *d = i;
        ierr = PetscInfo1(A,"Matrix is missing block diagonal number %" PetscInt_FMT "\n",i);CHKERRQ(ierr);
        break;
      }
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatMarkDiagonal_SeqBAIJ(Mat A)
{
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data;
  PetscErrorCode ierr;
  PetscInt       i,j,m = a->mbs;

  PetscFunctionBegin;
  if (!a->diag) {
    ierr         = PetscMalloc1(m,&a->diag);CHKERRQ(ierr);
    ierr         = PetscLogObjectMemory((PetscObject)A,m*sizeof(PetscInt));CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  PetscInt       i,j,n = a->mbs,nz = a->i[n],*tia,*tja,bs = A->rmap->bs,k,l,cnt;
  PetscInt       **ia = (PetscInt**)inia,**ja = (PetscInt**)inja;

  PetscFunctionBegin;
  *nn = n;
  if (!ia) PetscFunctionReturn(0);
  if (symmetric) {
    ierr = MatToSymmetricIJ_SeqAIJ(n,a->i,a->j,PETSC_TRUE,0,0,&tia,&tja);CHKERRQ(ierr);
    nz   = tia[n];
  } else {
    tia = a->i; tja = a->j;
  }

  if (!blockcompressed && bs > 1) {
    (*nn) *= bs;
    /* malloc & create the natural set of indices */
    ierr = PetscMalloc1((n+1)*bs,ia);CHKERRQ(ierr);
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
      ierr = PetscMalloc1(nz*bs*bs,ja);CHKERRQ(ierr);
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
      ierr = PetscFree(tia);CHKERRQ(ierr);
      ierr = PetscFree(tja);CHKERRQ(ierr);
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
      ierr = PetscMalloc1(A->rmap->n/bs+1,ia);CHKERRQ(ierr);
      for (i=0; i<A->rmap->n/bs+1; i++) (*ia)[i] = a->i[i] + 1;
      if (ja) {
        ierr = PetscMalloc1(nz,ja);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!ia) PetscFunctionReturn(0);
  if ((!blockcompressed && A->rmap->bs > 1) || (symmetric || oshift == 1)) {
    ierr = PetscFree(*ia);CHKERRQ(ierr);
    if (ja) {ierr = PetscFree(*ja);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_SeqBAIJ(Mat A)
{
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
#if defined(PETSC_USE_LOG)
  PetscLogObjectState((PetscObject)A,"Rows=%" PetscInt_FMT ", Cols=%" PetscInt_FMT ", NZ=%" PetscInt_FMT,A->rmap->N,A->cmap->n,a->nz);
#endif
  ierr = MatSeqXAIJFreeAIJ(A,&a->a,&a->j,&a->i);CHKERRQ(ierr);
  ierr = ISDestroy(&a->row);CHKERRQ(ierr);
  ierr = ISDestroy(&a->col);CHKERRQ(ierr);
  if (a->free_diag) {ierr = PetscFree(a->diag);CHKERRQ(ierr);}
  ierr = PetscFree(a->idiag);CHKERRQ(ierr);
  if (a->free_imax_ilen) {ierr = PetscFree2(a->imax,a->ilen);CHKERRQ(ierr);}
  ierr = PetscFree(a->solve_work);CHKERRQ(ierr);
  ierr = PetscFree(a->mult_work);CHKERRQ(ierr);
  ierr = PetscFree(a->sor_workt);CHKERRQ(ierr);
  ierr = PetscFree(a->sor_work);CHKERRQ(ierr);
  ierr = ISDestroy(&a->icol);CHKERRQ(ierr);
  ierr = PetscFree(a->saved_values);CHKERRQ(ierr);
  ierr = PetscFree2(a->compressedrow.i,a->compressedrow.rindex);CHKERRQ(ierr);

  ierr = MatDestroy(&a->sbaijMat);CHKERRQ(ierr);
  ierr = MatDestroy(&a->parent);CHKERRQ(ierr);
  ierr = PetscFree(A->data);CHKERRQ(ierr);

  ierr = PetscObjectChangeTypeName((PetscObject)A,NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatSeqBAIJGetArray_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatSeqBAIJRestoreArray_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatStoreValues_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatRetrieveValues_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatSeqBAIJSetColumnIndices_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatConvert_seqbaij_seqaij_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatConvert_seqbaij_seqsbaij_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatSeqBAIJSetPreallocation_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatSeqBAIJSetPreallocationCSR_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatConvert_seqbaij_seqbstrm_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatIsTranspose_C",NULL);CHKERRQ(ierr);
#if defined(PETSC_HAVE_HYPRE)
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatConvert_seqbaij_hypre_C",NULL);CHKERRQ(ierr);
#endif
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatConvert_seqbaij_is_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatSetOption_SeqBAIJ(Mat A,MatOption op,PetscBool flg)
{
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data;
  PetscErrorCode ierr;

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
    ierr = PetscInfo1(A,"Option %s ignored\n",MatOptions[op]);CHKERRQ(ierr);
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
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"unknown option %d",op);
  }
  PetscFunctionReturn(0);
}

/* used for both SeqBAIJ and SeqSBAIJ matrices */
PetscErrorCode MatGetRow_SeqBAIJ_private(Mat A,PetscInt row,PetscInt *nz,PetscInt **idx,PetscScalar **v,PetscInt *ai,PetscInt *aj,PetscScalar *aa)
{
  PetscErrorCode ierr;
  PetscInt       itmp,i,j,k,M,bn,bp,*idx_i,bs,bs2;
  MatScalar      *aa_i;
  PetscScalar    *v_i;

  PetscFunctionBegin;
  bs  = A->rmap->bs;
  bs2 = bs*bs;
  if (row < 0 || row >= A->rmap->N) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row %" PetscInt_FMT " out of range", row);

  bn  = row/bs;   /* Block number */
  bp  = row % bs; /* Block Position */
  M   = ai[bn+1] - ai[bn];
  *nz = bs*M;

  if (v) {
    *v = NULL;
    if (*nz) {
      ierr = PetscMalloc1(*nz,v);CHKERRQ(ierr);
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
      ierr = PetscMalloc1(*nz,idx);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatGetRow_SeqBAIJ_private(A,row,nz,idx,v,a->i,a->j,a->a);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatRestoreRow_SeqBAIJ(Mat A,PetscInt row,PetscInt *nz,PetscInt **idx,PetscScalar **v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (nz)  *nz = 0;
  if (idx) {ierr = PetscFree(*idx);CHKERRQ(ierr);}
  if (v)   {ierr = PetscFree(*v);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

PetscErrorCode MatTranspose_SeqBAIJ(Mat A,MatReuse reuse,Mat *B)
{
  Mat_SeqBAIJ    *a=(Mat_SeqBAIJ*)A->data,*at;
  Mat            C;
  PetscErrorCode ierr;
  PetscInt       i,j,k,*aj=a->j,*ai=a->i,bs=A->rmap->bs,mbs=a->mbs,nbs=a->nbs,*atfill;
  PetscInt       bs2=a->bs2,*ati,*atj,anzj,kr;
  MatScalar      *ata,*aa=a->a;

  PetscFunctionBegin;
  ierr = PetscCalloc1(1+nbs,&atfill);CHKERRQ(ierr);
  if (reuse == MAT_INITIAL_MATRIX || reuse == MAT_INPLACE_MATRIX) {
    for (i=0; i<ai[mbs]; i++) atfill[aj[i]] += 1; /* count num of non-zeros in row aj[i] */

    ierr = MatCreate(PetscObjectComm((PetscObject)A),&C);CHKERRQ(ierr);
    ierr = MatSetSizes(C,A->cmap->n,A->rmap->N,A->cmap->n,A->rmap->N);CHKERRQ(ierr);
    ierr = MatSetType(C,((PetscObject)A)->type_name);CHKERRQ(ierr);
    ierr = MatSeqBAIJSetPreallocation(C,bs,0,atfill);CHKERRQ(ierr);

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
  ierr = PetscArraycpy(atfill,ati,nbs);CHKERRQ(ierr);

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
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Clean up temporary space and complete requests. */
  ierr = PetscFree(atfill);CHKERRQ(ierr);

  if (reuse == MAT_INITIAL_MATRIX || reuse == MAT_REUSE_MATRIX) {
    ierr = MatSetBlockSizes(C,PetscAbs(A->cmap->bs),PetscAbs(A->rmap->bs));CHKERRQ(ierr);
    *B = C;
  } else {
    ierr = MatHeaderMerge(A,&C);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatIsTranspose_SeqBAIJ(Mat A,Mat B,PetscReal tol,PetscBool  *f)
{
  PetscErrorCode ierr;
  Mat            Btrans;

  PetscFunctionBegin;
  *f   = PETSC_FALSE;
  ierr = MatTranspose_SeqBAIJ(A,MAT_INITIAL_MATRIX,&Btrans);CHKERRQ(ierr);
  ierr = MatEqual_SeqBAIJ(B,Btrans,f);CHKERRQ(ierr);
  ierr = MatDestroy(&Btrans);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Used for both SeqBAIJ and SeqSBAIJ matrices */
PetscErrorCode MatView_SeqBAIJ_Binary(Mat mat,PetscViewer viewer)
{
  Mat_SeqBAIJ    *A = (Mat_SeqBAIJ*)mat->data;
  PetscInt       header[4],M,N,m,bs,nz,cnt,i,j,k,l;
  PetscInt       *rowlens,*colidxs;
  PetscScalar    *matvals;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerSetUp(viewer);CHKERRQ(ierr);

  M  = mat->rmap->N;
  N  = mat->cmap->N;
  m  = mat->rmap->n;
  bs = mat->rmap->bs;
  nz = bs*bs*A->nz;

  /* write matrix header */
  header[0] = MAT_FILE_CLASSID;
  header[1] = M; header[2] = N; header[3] = nz;
  ierr = PetscViewerBinaryWrite(viewer,header,4,PETSC_INT);CHKERRQ(ierr);

  /* store row lengths */
  ierr = PetscMalloc1(m,&rowlens);CHKERRQ(ierr);
  for (cnt=0, i=0; i<A->mbs; i++)
    for (j=0; j<bs; j++)
      rowlens[cnt++] = bs*(A->i[i+1] - A->i[i]);
  ierr = PetscViewerBinaryWrite(viewer,rowlens,m,PETSC_INT);CHKERRQ(ierr);
  ierr = PetscFree(rowlens);CHKERRQ(ierr);

  /* store column indices  */
  ierr = PetscMalloc1(nz,&colidxs);CHKERRQ(ierr);
  for (cnt=0, i=0; i<A->mbs; i++)
    for (k=0; k<bs; k++)
      for (j=A->i[i]; j<A->i[i+1]; j++)
        for (l=0; l<bs; l++)
          colidxs[cnt++] = bs*A->j[j] + l;
  if (cnt != nz) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_LIB,"Internal PETSc error: cnt = %" PetscInt_FMT " nz = %" PetscInt_FMT,cnt,nz);
  ierr = PetscViewerBinaryWrite(viewer,colidxs,nz,PETSC_INT);CHKERRQ(ierr);
  ierr = PetscFree(colidxs);CHKERRQ(ierr);

  /* store nonzero values */
  ierr = PetscMalloc1(nz,&matvals);CHKERRQ(ierr);
  for (cnt=0, i=0; i<A->mbs; i++)
    for (k=0; k<bs; k++)
      for (j=A->i[i]; j<A->i[i+1]; j++)
        for (l=0; l<bs; l++)
          matvals[cnt++] = A->a[bs*(bs*j + l) + k];
  if (cnt != nz) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_LIB,"Internal PETSc error: cnt = %" PetscInt_FMT " nz = %" PetscInt_FMT,cnt,nz);
  ierr = PetscViewerBinaryWrite(viewer,matvals,nz,PETSC_SCALAR);CHKERRQ(ierr);
  ierr = PetscFree(matvals);CHKERRQ(ierr);

  /* write block size option to the viewer's .info file */
  ierr = MatView_Binary_BlockSizes(mat,viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatView_SeqBAIJ_ASCII_structonly(Mat A,PetscViewer viewer)
{
  PetscErrorCode ierr;
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data;
  PetscInt       i,bs = A->rmap->bs,k;

  PetscFunctionBegin;
  ierr = PetscViewerASCIIUseTabs(viewer,PETSC_FALSE);CHKERRQ(ierr);
  for (i=0; i<a->mbs; i++) {
    ierr = PetscViewerASCIIPrintf(viewer,"row %" PetscInt_FMT "-%" PetscInt_FMT ":",i*bs,i*bs+bs-1);CHKERRQ(ierr);
    for (k=a->i[i]; k<a->i[i+1]; k++) {
      ierr = PetscViewerASCIIPrintf(viewer," (%" PetscInt_FMT "-%" PetscInt_FMT ") ",bs*a->j[k],bs*a->j[k]+bs-1);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIUseTabs(viewer,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatView_SeqBAIJ_ASCII(Mat A,PetscViewer viewer)
{
  Mat_SeqBAIJ       *a = (Mat_SeqBAIJ*)A->data;
  PetscErrorCode    ierr;
  PetscInt          i,j,bs = A->rmap->bs,k,l,bs2=a->bs2;
  PetscViewerFormat format;

  PetscFunctionBegin;
  if (A->structure_only) {
    ierr = MatView_SeqBAIJ_ASCII_structonly(A,viewer);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
  if (format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
    ierr = PetscViewerASCIIPrintf(viewer,"  block size is %" PetscInt_FMT "\n",bs);CHKERRQ(ierr);
  } else if (format == PETSC_VIEWER_ASCII_MATLAB) {
    const char *matname;
    Mat        aij;
    ierr = MatConvert(A,MATSEQAIJ,MAT_INITIAL_MATRIX,&aij);CHKERRQ(ierr);
    ierr = PetscObjectGetName((PetscObject)A,&matname);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)aij,matname);CHKERRQ(ierr);
    ierr = MatView(aij,viewer);CHKERRQ(ierr);
    ierr = MatDestroy(&aij);CHKERRQ(ierr);
  } else if (format == PETSC_VIEWER_ASCII_FACTOR_INFO) {
      PetscFunctionReturn(0);
  } else if (format == PETSC_VIEWER_ASCII_COMMON) {
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_FALSE);CHKERRQ(ierr);
    for (i=0; i<a->mbs; i++) {
      for (j=0; j<bs; j++) {
        ierr = PetscViewerASCIIPrintf(viewer,"row %" PetscInt_FMT ":",i*bs+j);CHKERRQ(ierr);
        for (k=a->i[i]; k<a->i[i+1]; k++) {
          for (l=0; l<bs; l++) {
#if defined(PETSC_USE_COMPLEX)
            if (PetscImaginaryPart(a->a[bs2*k + l*bs + j]) > 0.0 && PetscRealPart(a->a[bs2*k + l*bs + j]) != 0.0) {
              ierr = PetscViewerASCIIPrintf(viewer," (%" PetscInt_FMT ", %g + %gi) ",bs*a->j[k]+l,
                                            (double)PetscRealPart(a->a[bs2*k + l*bs + j]),(double)PetscImaginaryPart(a->a[bs2*k + l*bs + j]));CHKERRQ(ierr);
            } else if (PetscImaginaryPart(a->a[bs2*k + l*bs + j]) < 0.0 && PetscRealPart(a->a[bs2*k + l*bs + j]) != 0.0) {
              ierr = PetscViewerASCIIPrintf(viewer," (%" PetscInt_FMT ", %g - %gi) ",bs*a->j[k]+l,
                                            (double)PetscRealPart(a->a[bs2*k + l*bs + j]),-(double)PetscImaginaryPart(a->a[bs2*k + l*bs + j]));CHKERRQ(ierr);
            } else if (PetscRealPart(a->a[bs2*k + l*bs + j]) != 0.0) {
              ierr = PetscViewerASCIIPrintf(viewer," (%" PetscInt_FMT ", %g) ",bs*a->j[k]+l,(double)PetscRealPart(a->a[bs2*k + l*bs + j]));CHKERRQ(ierr);
            }
#else
            if (a->a[bs2*k + l*bs + j] != 0.0) {
              ierr = PetscViewerASCIIPrintf(viewer," (%" PetscInt_FMT ", %g) ",bs*a->j[k]+l,(double)a->a[bs2*k + l*bs + j]);CHKERRQ(ierr);
            }
#endif
          }
        }
        ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
      }
    }
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_TRUE);CHKERRQ(ierr);
  } else {
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_FALSE);CHKERRQ(ierr);
    for (i=0; i<a->mbs; i++) {
      for (j=0; j<bs; j++) {
        ierr = PetscViewerASCIIPrintf(viewer,"row %" PetscInt_FMT ":",i*bs+j);CHKERRQ(ierr);
        for (k=a->i[i]; k<a->i[i+1]; k++) {
          for (l=0; l<bs; l++) {
#if defined(PETSC_USE_COMPLEX)
            if (PetscImaginaryPart(a->a[bs2*k + l*bs + j]) > 0.0) {
              ierr = PetscViewerASCIIPrintf(viewer," (%" PetscInt_FMT ", %g + %g i) ",bs*a->j[k]+l,
                                            (double)PetscRealPart(a->a[bs2*k + l*bs + j]),(double)PetscImaginaryPart(a->a[bs2*k + l*bs + j]));CHKERRQ(ierr);
            } else if (PetscImaginaryPart(a->a[bs2*k + l*bs + j]) < 0.0) {
              ierr = PetscViewerASCIIPrintf(viewer," (%" PetscInt_FMT ", %g - %g i) ",bs*a->j[k]+l,
                                            (double)PetscRealPart(a->a[bs2*k + l*bs + j]),-(double)PetscImaginaryPart(a->a[bs2*k + l*bs + j]));CHKERRQ(ierr);
            } else {
              ierr = PetscViewerASCIIPrintf(viewer," (%" PetscInt_FMT ", %g) ",bs*a->j[k]+l,(double)PetscRealPart(a->a[bs2*k + l*bs + j]));CHKERRQ(ierr);
            }
#else
            ierr = PetscViewerASCIIPrintf(viewer," (%" PetscInt_FMT ", %g) ",bs*a->j[k]+l,(double)a->a[bs2*k + l*bs + j]);CHKERRQ(ierr);
#endif
          }
        }
        ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
      }
    }
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_TRUE);CHKERRQ(ierr);
  }
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#include <petscdraw.h>
static PetscErrorCode MatView_SeqBAIJ_Draw_Zoom(PetscDraw draw,void *Aa)
{
  Mat               A = (Mat) Aa;
  Mat_SeqBAIJ       *a=(Mat_SeqBAIJ*)A->data;
  PetscErrorCode    ierr;
  PetscInt          row,i,j,k,l,mbs=a->mbs,color,bs=A->rmap->bs,bs2=a->bs2;
  PetscReal         xl,yl,xr,yr,x_l,x_r,y_l,y_r;
  MatScalar         *aa;
  PetscViewer       viewer;
  PetscViewerFormat format;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject)A,"Zoomviewer",(PetscObject*)&viewer);CHKERRQ(ierr);
  ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
  ierr = PetscDrawGetCoordinates(draw,&xl,&yl,&xr,&yr);CHKERRQ(ierr);

  /* loop over matrix elements drawing boxes */

  if (format != PETSC_VIEWER_DRAW_CONTOUR) {
    ierr = PetscDrawCollectiveBegin(draw);CHKERRQ(ierr);
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
            ierr = PetscDrawRectangle(draw,x_l+k,y_l-l,x_r+k,y_r-l,color,color,color,color);CHKERRQ(ierr);
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
            ierr = PetscDrawRectangle(draw,x_l+k,y_l-l,x_r+k,y_r-l,color,color,color,color);CHKERRQ(ierr);
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
            ierr = PetscDrawRectangle(draw,x_l+k,y_l-l,x_r+k,y_r-l,color,color,color,color);CHKERRQ(ierr);
          }
        }
      }
    }
    ierr = PetscDrawCollectiveEnd(draw);CHKERRQ(ierr);
  } else {
    /* use contour shading to indicate magnitude of values */
    /* first determine max of all nonzero values */
    PetscReal minv = 0.0, maxv = 0.0;
    PetscDraw popup;

    for (i=0; i<a->nz*a->bs2; i++) {
      if (PetscAbsScalar(a->a[i]) > maxv) maxv = PetscAbsScalar(a->a[i]);
    }
    if (minv >= maxv) maxv = minv + PETSC_SMALL;
    ierr = PetscDrawGetPopup(draw,&popup);CHKERRQ(ierr);
    ierr = PetscDrawScalePopup(popup,0.0,maxv);CHKERRQ(ierr);

    ierr = PetscDrawCollectiveBegin(draw);CHKERRQ(ierr);
    for (i=0,row=0; i<mbs; i++,row+=bs) {
      for (j=a->i[i]; j<a->i[i+1]; j++) {
        y_l = A->rmap->N - row - 1.0; y_r = y_l + 1.0;
        x_l = a->j[j]*bs; x_r = x_l + 1.0;
        aa  = a->a + j*bs2;
        for (k=0; k<bs; k++) {
          for (l=0; l<bs; l++) {
            MatScalar v = *aa++;
            color = PetscDrawRealToColor(PetscAbsScalar(v),minv,maxv);
            ierr  = PetscDrawRectangle(draw,x_l+k,y_l-l,x_r+k,y_r-l,color,color,color,color);CHKERRQ(ierr);
          }
        }
      }
    }
    ierr = PetscDrawCollectiveEnd(draw);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatView_SeqBAIJ_Draw(Mat A,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscReal      xl,yl,xr,yr,w,h;
  PetscDraw      draw;
  PetscBool      isnull;

  PetscFunctionBegin;
  ierr = PetscViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
  ierr = PetscDrawIsNull(draw,&isnull);CHKERRQ(ierr);
  if (isnull) PetscFunctionReturn(0);

  xr   = A->cmap->n; yr = A->rmap->N; h = yr/10.0; w = xr/10.0;
  xr  += w;          yr += h;        xl = -w;     yl = -h;
  ierr = PetscDrawSetCoordinates(draw,xl,yl,xr,yr);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)A,"Zoomviewer",(PetscObject)viewer);CHKERRQ(ierr);
  ierr = PetscDrawZoom(draw,MatView_SeqBAIJ_Draw_Zoom,A);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)A,"Zoomviewer",NULL);CHKERRQ(ierr);
  ierr = PetscDrawSave(draw);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatView_SeqBAIJ(Mat A,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      iascii,isbinary,isdraw;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERDRAW,&isdraw);CHKERRQ(ierr);
  if (iascii) {
    ierr = MatView_SeqBAIJ_ASCII(A,viewer);CHKERRQ(ierr);
  } else if (isbinary) {
    ierr = MatView_SeqBAIJ_Binary(A,viewer);CHKERRQ(ierr);
  } else if (isdraw) {
    ierr = MatView_SeqBAIJ_Draw(A,viewer);CHKERRQ(ierr);
  } else {
    Mat B;
    ierr = MatConvert(A,MATSEQAIJ,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);
    ierr = MatView(B,viewer);CHKERRQ(ierr);
    ierr = MatDestroy(&B);CHKERRQ(ierr);
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
    if (row < 0) {v += n; continue;} /* SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Negative row"); */
    if (row >= A->rmap->N) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row %" PetscInt_FMT " too large", row);
    rp   = aj + ai[brow]; ap = aa + bs2*ai[brow];
    nrow = ailen[brow];
    for (l=0; l<n; l++) { /* loop over columns */
      if (in[l] < 0) {v++; continue;} /* SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Negative column"); */
      if (in[l] >= A->cmap->n) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Column %" PetscInt_FMT " too large", in[l]);
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
  PetscErrorCode    ierr;
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
    if (PetscUnlikelyDebug(row >= a->mbs)) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Block row index too large %" PetscInt_FMT " max %" PetscInt_FMT,row,a->mbs-1);
    rp   = aj + ai[row];
    if (!A->structure_only) ap = aa + bs2*ai[row];
    rmax = imax[row];
    nrow = ailen[row];
    low  = 0;
    high = nrow;
    for (l=0; l<n; l++) { /* loop over added columns */
      if (in[l] < 0) continue;
      if (PetscUnlikelyDebug(in[l] >= a->nbs)) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Block column index too large %" PetscInt_FMT " max %" PetscInt_FMT,in[l],a->nbs-1);
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
      if (nonew == -1) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Inserting a new blocked index new nonzero block (%" PetscInt_FMT ", %" PetscInt_FMT ") in the matrix", row, col);
      if (A->structure_only) {
        MatSeqXAIJReallocateAIJ_structure_only(A,a->mbs,bs2,nrow,row,col,rmax,ai,aj,rp,imax,nonew,MatScalar);
      } else {
        MatSeqXAIJReallocateAIJ(A,a->mbs,bs2,nrow,row,col,rmax,aa,ai,aj,rp,ap,imax,nonew,MatScalar);
      }
      N = nrow++ - 1; high++;
      /* shift up all the later entries in this row */
      ierr  = PetscArraymove(rp+i+1,rp+i,N-i+1);CHKERRQ(ierr);
      rp[i] = col;
      if (!A->structure_only) {
        ierr = PetscArraymove(ap+bs2*(i+1),ap+bs2*i,bs2*(N-i+1));CHKERRQ(ierr);
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
  PetscErrorCode ierr;
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
      ierr = PetscArraymove(ip-fshift,ip,N);CHKERRQ(ierr);
      if (!A->structure_only) {
        ierr = PetscArraymove(ap-bs2*fshift,ap,bs2*N);CHKERRQ(ierr);
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
    ierr = PetscFree2(a->imax,a->ilen);CHKERRQ(ierr);
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
    ierr    = PetscFree(a->diag);CHKERRQ(ierr);
    ierr    = PetscLogObjectMemory((PetscObject)A,-(mbs+1)*sizeof(PetscInt));CHKERRQ(ierr);
    a->diag = NULL;
  }
  if (fshift && a->nounused == -1) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_PLIB, "Unused space detected in matrix: %" PetscInt_FMT " X %" PetscInt_FMT " block size %" PetscInt_FMT ", %" PetscInt_FMT " unneeded", m, A->cmap->n, A->rmap->bs, fshift*bs2);
  ierr = PetscInfo5(A,"Matrix size: %" PetscInt_FMT " X %" PetscInt_FMT ", block size %" PetscInt_FMT "; storage space: %" PetscInt_FMT " unneeded, %" PetscInt_FMT " used\n",m,A->cmap->n,A->rmap->bs,fshift*bs2,a->nz*bs2);CHKERRQ(ierr);
  ierr = PetscInfo1(A,"Number of mallocs during MatSetValues is %" PetscInt_FMT "\n",a->reallocs);CHKERRQ(ierr);
  ierr = PetscInfo1(A,"Most nonzeros blocks in any row is %" PetscInt_FMT "\n",rmax);CHKERRQ(ierr);

  A->info.mallocs    += a->reallocs;
  a->reallocs         = 0;
  A->info.nz_unneeded = (PetscReal)fshift*bs2;
  a->rmax             = rmax;

  if (!A->structure_only) {
    ierr = MatCheckCompressedRow(A,a->nonzerorowcnt,&a->compressedrow,a->i,mbs,ratio);CHKERRQ(ierr);
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
      sizes[j] = 1;         /* Also makes sure atleast 'bs' values exist for next else */
      i++;
    } else { /* Begining of the block, so check if the complete block exists */
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
  PetscErrorCode    ierr;
  PetscInt          i,j,k,count,*rows;
  PetscInt          bs=A->rmap->bs,bs2=baij->bs2,*sizes,row,bs_max;
  PetscScalar       zero = 0.0;
  MatScalar         *aa;
  const PetscScalar *xx;
  PetscScalar       *bb;

  PetscFunctionBegin;
  /* fix right hand side if needed */
  if (x && b) {
    ierr = VecGetArrayRead(x,&xx);CHKERRQ(ierr);
    ierr = VecGetArray(b,&bb);CHKERRQ(ierr);
    for (i=0; i<is_n; i++) {
      bb[is_idx[i]] = diag*xx[is_idx[i]];
    }
    ierr = VecRestoreArrayRead(x,&xx);CHKERRQ(ierr);
    ierr = VecRestoreArray(b,&bb);CHKERRQ(ierr);
  }

  /* Make a copy of the IS and  sort it */
  /* allocate memory for rows,sizes */
  ierr = PetscMalloc2(is_n,&rows,2*is_n,&sizes);CHKERRQ(ierr);

  /* copy IS values to rows, and sort them */
  for (i=0; i<is_n; i++) rows[i] = is_idx[i];
  ierr = PetscSortInt(is_n,rows);CHKERRQ(ierr);

  if (baij->keepnonzeropattern) {
    for (i=0; i<is_n; i++) sizes[i] = 1;
    bs_max          = is_n;
  } else {
    ierr = MatZeroRows_SeqBAIJ_Check_Blocks(rows,is_n,bs,sizes,&bs_max);CHKERRQ(ierr);
    A->nonzerostate++;
  }

  for (i=0,j=0; i<bs_max; j+=sizes[i],i++) {
    row = rows[j];
    if (row < 0 || row > A->rmap->N) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"row %" PetscInt_FMT " out of range",row);
    count = (baij->i[row/bs +1] - baij->i[row/bs])*bs;
    aa    = ((MatScalar*)(baij->a)) + baij->i[row/bs]*bs2 + (row%bs);
    if (sizes[i] == bs && !baij->keepnonzeropattern) {
      if (diag != (PetscScalar)0.0) {
        if (baij->ilen[row/bs] > 0) {
          baij->ilen[row/bs]       = 1;
          baij->j[baij->i[row/bs]] = row/bs;

          ierr = PetscArrayzero(aa,count*bs);CHKERRQ(ierr);
        }
        /* Now insert all the diagonal values for this bs */
        for (k=0; k<bs; k++) {
          ierr = (*A->ops->setvalues)(A,1,rows+j+k,1,rows+j+k,&diag,INSERT_VALUES);CHKERRQ(ierr);
        }
      } else { /* (diag == 0.0) */
        baij->ilen[row/bs] = 0;
      } /* end (diag == 0.0) */
    } else { /* (sizes[i] != bs) */
      if (PetscUnlikelyDebug(sizes[i] != 1)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Internal Error. Value should be 1");
      for (k=0; k<count; k++) {
        aa[0] =  zero;
        aa   += bs;
      }
      if (diag != (PetscScalar)0.0) {
        ierr = (*A->ops->setvalues)(A,1,rows+j,1,rows+j,&diag,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }

  ierr = PetscFree2(rows,sizes);CHKERRQ(ierr);
  ierr = MatAssemblyEnd_SeqBAIJ(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatZeroRowsColumns_SeqBAIJ(Mat A,PetscInt is_n,const PetscInt is_idx[],PetscScalar diag,Vec x, Vec b)
{
  Mat_SeqBAIJ       *baij=(Mat_SeqBAIJ*)A->data;
  PetscErrorCode    ierr;
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
    ierr = VecGetArrayRead(x,&xx);CHKERRQ(ierr);
    ierr = VecGetArray(b,&bb);CHKERRQ(ierr);
    vecs = PETSC_TRUE;
  }

  /* zero the columns */
  ierr = PetscCalloc1(A->rmap->n,&zeroed);CHKERRQ(ierr);
  for (i=0; i<is_n; i++) {
    if (is_idx[i] < 0 || is_idx[i] >= A->rmap->N) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"row %" PetscInt_FMT " out of range",is_idx[i]);
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
  ierr = PetscFree(zeroed);CHKERRQ(ierr);
  if (vecs) {
    ierr = VecRestoreArrayRead(x,&xx);CHKERRQ(ierr);
    ierr = VecRestoreArray(b,&bb);CHKERRQ(ierr);
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
      ierr = (*A->ops->setvalues)(A,1,&row,1,&row,&diag,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyEnd_SeqBAIJ(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatSetValues_SeqBAIJ(Mat A,PetscInt m,const PetscInt im[],PetscInt n,const PetscInt in[],const PetscScalar v[],InsertMode is)
{
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data;
  PetscInt       *rp,k,low,high,t,ii,row,nrow,i,col,l,rmax,N,lastcol = -1;
  PetscInt       *imax=a->imax,*ai=a->i,*ailen=a->ilen;
  PetscInt       *aj  =a->j,nonew=a->nonew,bs=A->rmap->bs,brow,bcol;
  PetscErrorCode ierr;
  PetscInt       ridx,cidx,bs2=a->bs2;
  PetscBool      roworiented=a->roworiented;
  MatScalar      *ap=NULL,value=0.0,*aa=a->a,*bap;

  PetscFunctionBegin;
  for (k=0; k<m; k++) { /* loop over added rows */
    row  = im[k];
    brow = row/bs;
    if (row < 0) continue;
    if (PetscUnlikelyDebug(row >= A->rmap->N)) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row too large: row %" PetscInt_FMT " max %" PetscInt_FMT,row,A->rmap->N-1);
    rp   = aj + ai[brow];
    if (!A->structure_only) ap = aa + bs2*ai[brow];
    rmax = imax[brow];
    nrow = ailen[brow];
    low  = 0;
    high = nrow;
    for (l=0; l<n; l++) { /* loop over added columns */
      if (in[l] < 0) continue;
      if (PetscUnlikelyDebug(in[l] >= A->cmap->n)) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Column too large: col %" PetscInt_FMT " max %" PetscInt_FMT,in[l],A->cmap->n-1);
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
      if (nonew == -1) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Inserting a new nonzero (%" PetscInt_FMT ", %" PetscInt_FMT ") in the matrix", row, col);
      if (A->structure_only) {
        MatSeqXAIJReallocateAIJ_structure_only(A,a->mbs,bs2,nrow,brow,bcol,rmax,ai,aj,rp,imax,nonew,MatScalar);
      } else {
        MatSeqXAIJReallocateAIJ(A,a->mbs,bs2,nrow,brow,bcol,rmax,aa,ai,aj,rp,ap,imax,nonew,MatScalar);
      }
      N = nrow++ - 1; high++;
      /* shift up all the later entries in this row */
      ierr  = PetscArraymove(rp+i+1,rp+i,N-i+1);CHKERRQ(ierr);
      rp[i] = bcol;
      if (!A->structure_only) {
        ierr = PetscArraymove(ap+bs2*(i+1),ap+bs2*i,bs2*(N-i+1));CHKERRQ(ierr);
        ierr = PetscArrayzero(ap+bs2*i,bs2);CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  PetscBool      row_identity,col_identity;

  PetscFunctionBegin;
  if (info->levels != 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Only levels = 0 supported for in-place ILU");
  ierr = ISIdentity(row,&row_identity);CHKERRQ(ierr);
  ierr = ISIdentity(col,&col_identity);CHKERRQ(ierr);
  if (!row_identity || !col_identity) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Row and column permutations must be identity for in-place ILU");

  outA            = inA;
  inA->factortype = MAT_FACTOR_LU;
  ierr = PetscFree(inA->solvertype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(MATSOLVERPETSC,&inA->solvertype);CHKERRQ(ierr);

  ierr = MatMarkDiagonal_SeqBAIJ(inA);CHKERRQ(ierr);

  ierr   = PetscObjectReference((PetscObject)row);CHKERRQ(ierr);
  ierr   = ISDestroy(&a->row);CHKERRQ(ierr);
  a->row = row;
  ierr   = PetscObjectReference((PetscObject)col);CHKERRQ(ierr);
  ierr   = ISDestroy(&a->col);CHKERRQ(ierr);
  a->col = col;

  /* Create the invert permutation so that it can be used in MatLUFactorNumeric() */
  ierr = ISDestroy(&a->icol);CHKERRQ(ierr);
  ierr = ISInvertPermutation(col,PETSC_DECIDE,&a->icol);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)inA,(PetscObject)a->icol);CHKERRQ(ierr);

  ierr = MatSeqBAIJSetNumericFactorization_inplace(inA,(PetscBool)(row_identity && col_identity));CHKERRQ(ierr);
  if (!a->solve_work) {
    ierr = PetscMalloc1(inA->rmap->N+inA->rmap->bs,&a->solve_work);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)inA,(inA->rmap->N+inA->rmap->bs)*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  ierr = MatLUFactorNumeric(outA,inA,info);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidPointer(indices,2);
  ierr = PetscUseMethod(mat,"MatSeqBAIJSetColumnIndices_C",(Mat,PetscInt*),(mat,indices));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatGetRowMaxAbs_SeqBAIJ(Mat A,Vec v,PetscInt idx[])
{
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data;
  PetscErrorCode ierr;
  PetscInt       i,j,n,row,bs,*ai,*aj,mbs;
  PetscReal      atmp;
  PetscScalar    *x,zero = 0.0;
  MatScalar      *aa;
  PetscInt       ncols,brow,krow,kcol;

  PetscFunctionBegin;
  if (A->factortype) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  bs  = A->rmap->bs;
  aa  = a->a;
  ai  = a->i;
  aj  = a->j;
  mbs = a->mbs;

  ierr = VecSet(v,zero);CHKERRQ(ierr);
  ierr = VecGetArray(v,&x);CHKERRQ(ierr);
  ierr = VecGetLocalSize(v,&n);CHKERRQ(ierr);
  if (n != A->rmap->N) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Nonconforming matrix and vector");
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
  ierr = VecRestoreArray(v,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatCopy_SeqBAIJ(Mat A,Mat B,MatStructure str)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* If the two matrices have the same copy implementation, use fast copy. */
  if (str == SAME_NONZERO_PATTERN && (A->ops->copy == B->ops->copy)) {
    Mat_SeqBAIJ *a  = (Mat_SeqBAIJ*)A->data;
    Mat_SeqBAIJ *b  = (Mat_SeqBAIJ*)B->data;
    PetscInt    ambs=a->mbs,bmbs=b->mbs,abs=A->rmap->bs,bbs=B->rmap->bs,bs2=abs*abs;

    if (a->i[ambs] != b->i[bmbs]) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Number of nonzero blocks in matrices A %" PetscInt_FMT " and B %" PetscInt_FMT " are different",a->i[ambs],b->i[bmbs]);
    if (abs != bbs) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Block size A %" PetscInt_FMT " and B %" PetscInt_FMT " are different",abs,bbs);
    ierr = PetscArraycpy(b->a,a->a,bs2*a->i[ambs]);CHKERRQ(ierr);
    ierr = PetscObjectStateIncrease((PetscObject)B);CHKERRQ(ierr);
  } else {
    ierr = MatCopy_Basic(A,B,str);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatSetUp_SeqBAIJ(Mat A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSeqBAIJSetPreallocation(A,A->rmap->bs,PETSC_DEFAULT,NULL);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Set the number of nonzeros in the new matrix */
  ierr = MatAXPYGetPreallocation_SeqX_private(mbs,x->i,x->j,y->i,y->j,nnz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatAXPY_SeqBAIJ(Mat Y,PetscScalar a,Mat X,MatStructure str)
{
  Mat_SeqBAIJ    *x = (Mat_SeqBAIJ*)X->data,*y = (Mat_SeqBAIJ*)Y->data;
  PetscErrorCode ierr;
  PetscInt       bs=Y->rmap->bs,bs2=bs*bs;
  PetscBLASInt   one=1;

  PetscFunctionBegin;
  if (str == UNKNOWN_NONZERO_PATTERN || (PetscDefined(USE_DEBUG) && str == SAME_NONZERO_PATTERN)) {
    PetscBool e = x->nz == y->nz && x->mbs == y->mbs && bs == X->rmap->bs ? PETSC_TRUE : PETSC_FALSE;
    if (e) {
      ierr = PetscArraycmp(x->i,y->i,x->mbs+1,&e);CHKERRQ(ierr);
      if (e) {
        ierr = PetscArraycmp(x->j,y->j,x->i[x->mbs],&e);CHKERRQ(ierr);
        if (e) str = SAME_NONZERO_PATTERN;
      }
    }
    if (!e && str == SAME_NONZERO_PATTERN) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"MatStructure is not SAME_NONZERO_PATTERN");
  }
  if (str == SAME_NONZERO_PATTERN) {
    PetscScalar  alpha = a;
    PetscBLASInt bnz;
    ierr = PetscBLASIntCast(x->nz*bs2,&bnz);CHKERRQ(ierr);
    PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&bnz,&alpha,x->a,&one,y->a,&one));
    ierr = PetscObjectStateIncrease((PetscObject)Y);CHKERRQ(ierr);
  } else if (str == SUBSET_NONZERO_PATTERN) { /* nonzeros of X is a subset of Y's */
    ierr = MatAXPY_Basic(Y,a,X,str);CHKERRQ(ierr);
  } else {
    Mat      B;
    PetscInt *nnz;
    if (bs != X->rmap->bs) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Matrices must have same block size");
    ierr = PetscMalloc1(Y->rmap->N,&nnz);CHKERRQ(ierr);
    ierr = MatCreate(PetscObjectComm((PetscObject)Y),&B);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)B,((PetscObject)Y)->name);CHKERRQ(ierr);
    ierr = MatSetSizes(B,Y->rmap->n,Y->cmap->n,Y->rmap->N,Y->cmap->N);CHKERRQ(ierr);
    ierr = MatSetBlockSizesFromMats(B,Y,Y);CHKERRQ(ierr);
    ierr = MatSetType(B,(MatType) ((PetscObject)Y)->type_name);CHKERRQ(ierr);
    ierr = MatAXPYGetPreallocation_SeqBAIJ(Y,X,nnz);CHKERRQ(ierr);
    ierr = MatSeqBAIJSetPreallocation(B,bs,0,nnz);CHKERRQ(ierr);
    ierr = MatAXPY_BasicWithPreallocation(B,Y,a,X,str);CHKERRQ(ierr);
    ierr = MatHeaderMerge(Y,&B);CHKERRQ(ierr);
    ierr = PetscFree(nnz);CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  PetscInt       bs = A->rmap->bs,i,*collengths,*cia,*cja,n = A->cmap->n/bs,m = A->rmap->n/bs;
  PetscInt       nz = a->i[m],row,*jj,mr,col;

  PetscFunctionBegin;
  *nn = n;
  if (!ia) PetscFunctionReturn(0);
  if (symmetric) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not for BAIJ matrices");
  else {
    ierr = PetscCalloc1(n,&collengths);CHKERRQ(ierr);
    ierr = PetscMalloc1(n+1,&cia);CHKERRQ(ierr);
    ierr = PetscMalloc1(nz,&cja);CHKERRQ(ierr);
    jj   = a->j;
    for (i=0; i<nz; i++) {
      collengths[jj[i]]++;
    }
    cia[0] = oshift;
    for (i=0; i<n; i++) {
      cia[i+1] = cia[i] + collengths[i];
    }
    ierr = PetscArrayzero(collengths,n);CHKERRQ(ierr);
    jj   = a->j;
    for (row=0; row<m; row++) {
      mr = a->i[row+1] - a->i[row];
      for (i=0; i<mr; i++) {
        col = *jj++;

        cja[cia[col] + collengths[col]++ - oshift] = row + oshift;
      }
    }
    ierr = PetscFree(collengths);CHKERRQ(ierr);
    *ia  = cia; *ja = cja;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatRestoreColumnIJ_SeqBAIJ(Mat A,PetscInt oshift,PetscBool symmetric,PetscBool inodecompressed,PetscInt *n,const PetscInt *ia[],const PetscInt *ja[],PetscBool  *done)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!ia) PetscFunctionReturn(0);
  ierr = PetscFree(*ia);CHKERRQ(ierr);
  ierr = PetscFree(*ja);CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  PetscInt       i,*collengths,*cia,*cja,n=a->nbs,m=a->mbs;
  PetscInt       nz = a->i[m],row,*jj,mr,col;
  PetscInt       *cspidx;

  PetscFunctionBegin;
  *nn = n;
  if (!ia) PetscFunctionReturn(0);

  ierr = PetscCalloc1(n,&collengths);CHKERRQ(ierr);
  ierr = PetscMalloc1(n+1,&cia);CHKERRQ(ierr);
  ierr = PetscMalloc1(nz,&cja);CHKERRQ(ierr);
  ierr = PetscMalloc1(nz,&cspidx);CHKERRQ(ierr);
  jj   = a->j;
  for (i=0; i<nz; i++) {
    collengths[jj[i]]++;
  }
  cia[0] = oshift;
  for (i=0; i<n; i++) {
    cia[i+1] = cia[i] + collengths[i];
  }
  ierr = PetscArrayzero(collengths,n);CHKERRQ(ierr);
  jj   = a->j;
  for (row=0; row<m; row++) {
    mr = a->i[row+1] - a->i[row];
    for (i=0; i<mr; i++) {
      col = *jj++;
      cspidx[cia[col] + collengths[col] - oshift] = a->i[row] + i; /* index of a->j */
      cja[cia[col] + collengths[col]++ - oshift]  = row + oshift;
    }
  }
  ierr   = PetscFree(collengths);CHKERRQ(ierr);
  *ia    = cia;
  *ja    = cja;
  *spidx = cspidx;
  PetscFunctionReturn(0);
}

PetscErrorCode MatRestoreColumnIJ_SeqBAIJ_Color(Mat A,PetscInt oshift,PetscBool symmetric,PetscBool inodecompressed,PetscInt *n,const PetscInt *ia[],const PetscInt *ja[],PetscInt *spidx[],PetscBool  *done)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatRestoreColumnIJ_SeqBAIJ(A,oshift,symmetric,inodecompressed,n,ia,ja,done);CHKERRQ(ierr);
  ierr = PetscFree(*spidx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatShift_SeqBAIJ(Mat Y,PetscScalar a)
{
  PetscErrorCode ierr;
  Mat_SeqBAIJ     *aij = (Mat_SeqBAIJ*)Y->data;

  PetscFunctionBegin;
  if (!Y->preallocated || !aij->nz) {
    ierr = MatSeqBAIJSetPreallocation(Y,Y->rmap->bs,1,NULL);CHKERRQ(ierr);
  }
  ierr = MatShift_Basic(Y,a);CHKERRQ(ierr);
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
                                       MatDestroySubMatrices_SeqBAIJ
};

PetscErrorCode  MatStoreValues_SeqBAIJ(Mat mat)
{
  Mat_SeqBAIJ    *aij = (Mat_SeqBAIJ*)mat->data;
  PetscInt       nz   = aij->i[aij->mbs]*aij->bs2;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (aij->nonew != 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Must call MatSetOption(A,MAT_NEW_NONZERO_LOCATIONS,PETSC_FALSE);first");

  /* allocate space for values if not already there */
  if (!aij->saved_values) {
    ierr = PetscMalloc1(nz+1,&aij->saved_values);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)mat,(nz+1)*sizeof(PetscScalar));CHKERRQ(ierr);
  }

  /* copy values over */
  ierr = PetscArraycpy(aij->saved_values,aij->a,nz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode  MatRetrieveValues_SeqBAIJ(Mat mat)
{
  Mat_SeqBAIJ    *aij = (Mat_SeqBAIJ*)mat->data;
  PetscErrorCode ierr;
  PetscInt       nz = aij->i[aij->mbs]*aij->bs2;

  PetscFunctionBegin;
  if (aij->nonew != 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Must call MatSetOption(A,MAT_NEW_NONZERO_LOCATIONS,PETSC_FALSE);first");
  if (!aij->saved_values) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Must call MatStoreValues(A);first");

  /* copy values over */
  ierr = PetscArraycpy(aij->a,aij->saved_values,nz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatConvert_SeqBAIJ_SeqAIJ(Mat, MatType,MatReuse,Mat*);
PETSC_INTERN PetscErrorCode MatConvert_SeqBAIJ_SeqSBAIJ(Mat, MatType,MatReuse,Mat*);

PetscErrorCode  MatSeqBAIJSetPreallocation_SeqBAIJ(Mat B,PetscInt bs,PetscInt nz,PetscInt *nnz)
{
  Mat_SeqBAIJ    *b;
  PetscErrorCode ierr;
  PetscInt       i,mbs,nbs,bs2;
  PetscBool      flg = PETSC_FALSE,skipallocation = PETSC_FALSE,realalloc = PETSC_FALSE;

  PetscFunctionBegin;
  if (nz >= 0 || nnz) realalloc = PETSC_TRUE;
  if (nz == MAT_SKIP_ALLOCATION) {
    skipallocation = PETSC_TRUE;
    nz             = 0;
  }

  ierr = MatSetBlockSize(B,PetscAbs(bs));CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(B->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(B->cmap);CHKERRQ(ierr);
  ierr = PetscLayoutGetBlockSize(B->rmap,&bs);CHKERRQ(ierr);

  B->preallocated = PETSC_TRUE;

  mbs = B->rmap->n/bs;
  nbs = B->cmap->n/bs;
  bs2 = bs*bs;

  if (mbs*bs!=B->rmap->n || nbs*bs!=B->cmap->n) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Number rows %" PetscInt_FMT ", cols %" PetscInt_FMT " must be divisible by blocksize %" PetscInt_FMT,B->rmap->N,B->cmap->n,bs);

  if (nz == PETSC_DEFAULT || nz == PETSC_DECIDE) nz = 5;
  if (nz < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"nz cannot be less than 0: value %" PetscInt_FMT,nz);
  if (nnz) {
    for (i=0; i<mbs; i++) {
      if (nnz[i] < 0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"nnz cannot be less than 0: local row %" PetscInt_FMT " value %" PetscInt_FMT,i,nnz[i]);
      if (nnz[i] > nbs) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"nnz cannot be greater than block row length: local row %" PetscInt_FMT " value %" PetscInt_FMT " rowlength %" PetscInt_FMT,i,nnz[i],nbs);
    }
  }

  b    = (Mat_SeqBAIJ*)B->data;
  ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)B),NULL,"Optimize options for SEQBAIJ matrix 2 ","Mat");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-mat_no_unroll","Do not optimize for block size (slow)",NULL,flg,&flg,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

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
      ierr = PetscOptionsGetInt(NULL,((PetscObject)B)->prefix,"-mat_baij_mult_version",&version,NULL);CHKERRQ(ierr);
      switch (version) {
#if defined(PETSC_HAVE_IMMINTRIN_H) && defined(__AVX2__) && defined(__FMA__) && defined(PETSC_USE_REAL_DOUBLE) && !defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_64BIT_INDICES)
      case 1:
        B->ops->mult    = MatMult_SeqBAIJ_9_AVX2;
        B->ops->multadd = MatMultAdd_SeqBAIJ_9_AVX2;
        ierr = PetscInfo1((PetscObject)B,"Using AVX2 for MatMult for BAIJ for blocksize %" PetscInt_FMT "\n",bs);CHKERRQ(ierr);
        break;
#endif
      default:
        B->ops->mult    = MatMult_SeqBAIJ_N;
        B->ops->multadd = MatMultAdd_SeqBAIJ_N;
        ierr = PetscInfo1((PetscObject)B,"Using BLAS for MatMult for BAIJ for blocksize %" PetscInt_FMT "\n",bs);CHKERRQ(ierr);
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
      ierr = PetscOptionsGetInt(NULL,((PetscObject)B)->prefix,"-mat_baij_mult_version",&version,NULL);CHKERRQ(ierr);
      switch (version) {
      case 1:
        B->ops->mult    = MatMult_SeqBAIJ_12_ver1;
        B->ops->multadd = MatMultAdd_SeqBAIJ_12_ver1;
        ierr = PetscInfo2((PetscObject)B,"Using version %" PetscInt_FMT " of MatMult for BAIJ for blocksize %" PetscInt_FMT "\n",version,bs);CHKERRQ(ierr);
        break;
      case 2:
        B->ops->mult    = MatMult_SeqBAIJ_12_ver2;
        B->ops->multadd = MatMultAdd_SeqBAIJ_12_ver2;
        ierr = PetscInfo2((PetscObject)B,"Using version %" PetscInt_FMT " of MatMult for BAIJ for blocksize %" PetscInt_FMT "\n",version,bs);CHKERRQ(ierr);
        break;
#if defined(PETSC_HAVE_IMMINTRIN_H) && defined(__AVX2__) && defined(__FMA__) && defined(PETSC_USE_REAL_DOUBLE) && !defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_64BIT_INDICES)
      case 3:
        B->ops->mult    = MatMult_SeqBAIJ_12_AVX2;
        B->ops->multadd = MatMultAdd_SeqBAIJ_12_ver1;
        ierr = PetscInfo1((PetscObject)B,"Using AVX2 for MatMult for BAIJ for blocksize %" PetscInt_FMT "\n",bs);CHKERRQ(ierr);
        break;
#endif
      default:
        B->ops->mult    = MatMult_SeqBAIJ_N;
        B->ops->multadd = MatMultAdd_SeqBAIJ_N;
        ierr = PetscInfo1((PetscObject)B,"Using BLAS for MatMult for BAIJ for blocksize %" PetscInt_FMT "\n",bs);CHKERRQ(ierr);
        break;
      }
      break;
    }
    case 15:
    {
      PetscInt version = 1;
      ierr = PetscOptionsGetInt(NULL,((PetscObject)B)->prefix,"-mat_baij_mult_version",&version,NULL);CHKERRQ(ierr);
      switch (version) {
      case 1:
        B->ops->mult    = MatMult_SeqBAIJ_15_ver1;
        ierr = PetscInfo2((PetscObject)B,"Using version %" PetscInt_FMT " of MatMult for BAIJ for blocksize %" PetscInt_FMT "\n",version,bs);CHKERRQ(ierr);
        break;
      case 2:
        B->ops->mult    = MatMult_SeqBAIJ_15_ver2;
        ierr = PetscInfo2((PetscObject)B,"Using version %" PetscInt_FMT " of MatMult for BAIJ for blocksize %" PetscInt_FMT "\n",version,bs);CHKERRQ(ierr);
        break;
      case 3:
        B->ops->mult    = MatMult_SeqBAIJ_15_ver3;
        ierr = PetscInfo2((PetscObject)B,"Using version %" PetscInt_FMT " of MatMult for BAIJ for blocksize %" PetscInt_FMT "\n",version,bs);CHKERRQ(ierr);
        break;
      case 4:
        B->ops->mult    = MatMult_SeqBAIJ_15_ver4;
        ierr = PetscInfo2((PetscObject)B,"Using version %" PetscInt_FMT " of MatMult for BAIJ for blocksize %" PetscInt_FMT "\n",version,bs);CHKERRQ(ierr);
        break;
      default:
        B->ops->mult    = MatMult_SeqBAIJ_N;
        ierr = PetscInfo1((PetscObject)B,"Using BLAS for MatMult for BAIJ for blocksize %" PetscInt_FMT "\n",bs);CHKERRQ(ierr);
        break;
      }
      B->ops->multadd = MatMultAdd_SeqBAIJ_N;
      break;
    }
    default:
      B->ops->mult    = MatMult_SeqBAIJ_N;
      B->ops->multadd = MatMultAdd_SeqBAIJ_N;
      ierr = PetscInfo1((PetscObject)B,"Using BLAS for MatMult for BAIJ for blocksize %" PetscInt_FMT "\n",bs);CHKERRQ(ierr);
      break;
    }
  }
  B->ops->sor = MatSOR_SeqBAIJ;
  b->mbs = mbs;
  b->nbs = nbs;
  if (!skipallocation) {
    if (!b->imax) {
      ierr = PetscMalloc2(mbs,&b->imax,mbs,&b->ilen);CHKERRQ(ierr);
      ierr = PetscLogObjectMemory((PetscObject)B,2*mbs*sizeof(PetscInt));CHKERRQ(ierr);

      b->free_imax_ilen = PETSC_TRUE;
    }
    /* b->ilen will count nonzeros in each block row so far. */
    for (i=0; i<mbs; i++) b->ilen[i] = 0;
    if (!nnz) {
      if (nz == PETSC_DEFAULT || nz == PETSC_DECIDE) nz = 5;
      else if (nz < 0) nz = 1;
      nz = PetscMin(nz,nbs);
      for (i=0; i<mbs; i++) b->imax[i] = nz;
      ierr = PetscIntMultError(nz,mbs,&nz);CHKERRQ(ierr);
    } else {
      PetscInt64 nz64 = 0;
      for (i=0; i<mbs; i++) {b->imax[i] = nnz[i]; nz64 += nnz[i];}
      ierr = PetscIntCast(nz64,&nz);CHKERRQ(ierr);
    }

    /* allocate the matrix space */
    ierr = MatSeqXAIJFreeAIJ(B,&b->a,&b->j,&b->i);CHKERRQ(ierr);
    if (B->structure_only) {
      ierr = PetscMalloc1(nz,&b->j);CHKERRQ(ierr);
      ierr = PetscMalloc1(B->rmap->N+1,&b->i);CHKERRQ(ierr);
      ierr = PetscLogObjectMemory((PetscObject)B,(B->rmap->N+1)*sizeof(PetscInt)+nz*sizeof(PetscInt));CHKERRQ(ierr);
    } else {
      PetscInt nzbs2 = 0;
      ierr = PetscIntMultError(nz,bs2,&nzbs2);CHKERRQ(ierr);
      ierr = PetscMalloc3(nzbs2,&b->a,nz,&b->j,B->rmap->N+1,&b->i);CHKERRQ(ierr);
      ierr = PetscLogObjectMemory((PetscObject)B,(B->rmap->N+1)*sizeof(PetscInt)+nz*(bs2*sizeof(PetscScalar)+sizeof(PetscInt)));CHKERRQ(ierr);
      ierr = PetscArrayzero(b->a,nz*bs2);CHKERRQ(ierr);
    }
    ierr = PetscArrayzero(b->j,nz);CHKERRQ(ierr);

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
  if (realalloc) {ierr = MatSetOption(B,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

PetscErrorCode MatSeqBAIJSetPreallocationCSR_SeqBAIJ(Mat B,PetscInt bs,const PetscInt ii[],const PetscInt jj[],const PetscScalar V[])
{
  PetscInt       i,m,nz,nz_max=0,*nnz;
  PetscScalar    *values=NULL;
  PetscBool      roworiented = ((Mat_SeqBAIJ*)B->data)->roworiented;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (bs < 1) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Invalid block size specified, must be positive but it is %" PetscInt_FMT,bs);
  ierr = PetscLayoutSetBlockSize(B->rmap,bs);CHKERRQ(ierr);
  ierr = PetscLayoutSetBlockSize(B->cmap,bs);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(B->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(B->cmap);CHKERRQ(ierr);
  ierr = PetscLayoutGetBlockSize(B->rmap,&bs);CHKERRQ(ierr);
  m    = B->rmap->n/bs;

  if (ii[0] != 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE, "ii[0] must be 0 but it is %" PetscInt_FMT,ii[0]);
  ierr = PetscMalloc1(m+1, &nnz);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    nz = ii[i+1]- ii[i];
    if (nz < 0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE, "Local row %" PetscInt_FMT " has a negative number of columns %" PetscInt_FMT,i,nz);
    nz_max = PetscMax(nz_max, nz);
    nnz[i] = nz;
  }
  ierr = MatSeqBAIJSetPreallocation(B,bs,0,nnz);CHKERRQ(ierr);
  ierr = PetscFree(nnz);CHKERRQ(ierr);

  values = (PetscScalar*)V;
  if (!values) {
    ierr = PetscCalloc1(bs*bs*(nz_max+1),&values);CHKERRQ(ierr);
  }
  for (i=0; i<m; i++) {
    PetscInt          ncols  = ii[i+1] - ii[i];
    const PetscInt    *icols = jj + ii[i];
    if (bs == 1 || !roworiented) {
      const PetscScalar *svals = values + (V ? (bs*bs*ii[i]) : 0);
      ierr = MatSetValuesBlocked_SeqBAIJ(B,1,&i,ncols,icols,svals,INSERT_VALUES);CHKERRQ(ierr);
    } else {
      PetscInt j;
      for (j=0; j<ncols; j++) {
        const PetscScalar *svals = values + (V ? (bs*bs*(ii[i]+j)) : 0);
        ierr = MatSetValuesBlocked_SeqBAIJ(B,1,&i,1,&icols[j],svals,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }
  if (!V) { ierr = PetscFree(values);CHKERRQ(ierr); }
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatSetOption(B,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
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

.seealso: MatSeqBAIJRestoreArray(), MatSeqAIJGetArray(), MatSeqAIJRestoreArray()
@*/
PetscErrorCode MatSeqBAIJGetArray(Mat A,PetscScalar **array)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscUseMethod(A,"MatSeqBAIJGetArray_C",(Mat,PetscScalar**),(A,array));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   MatSeqBAIJRestoreArray - returns access to the array where the data for a MATSEQBAIJ matrix is stored obtained by MatSeqBAIJGetArray()

   Not Collective

   Input Parameters:
+  mat - a MATSEQBAIJ matrix
-  array - pointer to the data

   Level: intermediate

.seealso: MatSeqBAIJGetArray(), MatSeqAIJGetArray(), MatSeqAIJRestoreArray()
@*/
PetscErrorCode MatSeqBAIJRestoreArray(Mat A,PetscScalar **array)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscUseMethod(A,"MatSeqBAIJRestoreArray_C",(Mat,PetscScalar**),(A,array));CHKERRQ(ierr);
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

.seealso: MatCreateSeqBAIJ()
M*/

PETSC_INTERN PetscErrorCode MatConvert_SeqBAIJ_SeqBSTRM(Mat, MatType,MatReuse,Mat*);

PETSC_EXTERN PetscErrorCode MatCreate_SeqBAIJ(Mat B)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;
  Mat_SeqBAIJ    *b;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)B),&size);CHKERRMPI(ierr);
  if (size > 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Comm must be of size 1");

  ierr    = PetscNewLog(B,&b);CHKERRQ(ierr);
  B->data = (void*)b;
  ierr    = PetscMemcpy(B->ops,&MatOps_Values,sizeof(struct _MatOps));CHKERRQ(ierr);

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

  ierr = PetscObjectComposeFunction((PetscObject)B,"MatSeqBAIJGetArray_C",MatSeqBAIJGetArray_SeqBAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatSeqBAIJRestoreArray_C",MatSeqBAIJRestoreArray_SeqBAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatStoreValues_C",MatStoreValues_SeqBAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatRetrieveValues_C",MatRetrieveValues_SeqBAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatSeqBAIJSetColumnIndices_C",MatSeqBAIJSetColumnIndices_SeqBAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_seqbaij_seqaij_C",MatConvert_SeqBAIJ_SeqAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_seqbaij_seqsbaij_C",MatConvert_SeqBAIJ_SeqSBAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatSeqBAIJSetPreallocation_C",MatSeqBAIJSetPreallocation_SeqBAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatSeqBAIJSetPreallocationCSR_C",MatSeqBAIJSetPreallocationCSR_SeqBAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatIsTranspose_C",MatIsTranspose_SeqBAIJ);CHKERRQ(ierr);
#if defined(PETSC_HAVE_HYPRE)
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_seqbaij_hypre_C",MatConvert_AIJ_HYPRE);CHKERRQ(ierr);
#endif
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_seqbaij_is_C",MatConvert_XAIJ_IS);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)B,MATSEQBAIJ);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatDuplicateNoCreate_SeqBAIJ(Mat C,Mat A,MatDuplicateOption cpvalues,PetscBool mallocmatspace)
{
  Mat_SeqBAIJ    *c = (Mat_SeqBAIJ*)C->data,*a = (Mat_SeqBAIJ*)A->data;
  PetscErrorCode ierr;
  PetscInt       i,mbs = a->mbs,nz = a->nz,bs2 = a->bs2;

  PetscFunctionBegin;
  if (a->i[mbs] != nz) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Corrupt matrix");

  if (cpvalues == MAT_SHARE_NONZERO_PATTERN) {
    c->imax           = a->imax;
    c->ilen           = a->ilen;
    c->free_imax_ilen = PETSC_FALSE;
  } else {
    ierr = PetscMalloc2(mbs,&c->imax,mbs,&c->ilen);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)C,2*mbs*sizeof(PetscInt));CHKERRQ(ierr);
    for (i=0; i<mbs; i++) {
      c->imax[i] = a->imax[i];
      c->ilen[i] = a->ilen[i];
    }
    c->free_imax_ilen = PETSC_TRUE;
  }

  /* allocate the matrix space */
  if (mallocmatspace) {
    if (cpvalues == MAT_SHARE_NONZERO_PATTERN) {
      ierr = PetscCalloc1(bs2*nz,&c->a);CHKERRQ(ierr);
      ierr = PetscLogObjectMemory((PetscObject)C,a->i[mbs]*bs2*sizeof(PetscScalar));CHKERRQ(ierr);

      c->i            = a->i;
      c->j            = a->j;
      c->singlemalloc = PETSC_FALSE;
      c->free_a       = PETSC_TRUE;
      c->free_ij      = PETSC_FALSE;
      c->parent       = A;
      C->preallocated = PETSC_TRUE;
      C->assembled    = PETSC_TRUE;

      ierr = PetscObjectReference((PetscObject)A);CHKERRQ(ierr);
      ierr = MatSetOption(A,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
      ierr = MatSetOption(C,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
    } else {
      ierr = PetscMalloc3(bs2*nz,&c->a,nz,&c->j,mbs+1,&c->i);CHKERRQ(ierr);
      ierr = PetscLogObjectMemory((PetscObject)C,a->i[mbs]*(bs2*sizeof(PetscScalar)+sizeof(PetscInt))+(mbs+1)*sizeof(PetscInt));CHKERRQ(ierr);

      c->singlemalloc = PETSC_TRUE;
      c->free_a       = PETSC_TRUE;
      c->free_ij      = PETSC_TRUE;

      ierr = PetscArraycpy(c->i,a->i,mbs+1);CHKERRQ(ierr);
      if (mbs > 0) {
        ierr = PetscArraycpy(c->j,a->j,nz);CHKERRQ(ierr);
        if (cpvalues == MAT_COPY_VALUES) {
          ierr = PetscArraycpy(c->a,a->a,bs2*nz);CHKERRQ(ierr);
        } else {
          ierr = PetscArrayzero(c->a,bs2*nz);CHKERRQ(ierr);
        }
      }
      C->preallocated = PETSC_TRUE;
      C->assembled    = PETSC_TRUE;
    }
  }

  c->roworiented = a->roworiented;
  c->nonew       = a->nonew;

  ierr = PetscLayoutReference(A->rmap,&C->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutReference(A->cmap,&C->cmap);CHKERRQ(ierr);

  c->bs2         = a->bs2;
  c->mbs         = a->mbs;
  c->nbs         = a->nbs;

  if (a->diag) {
    if (cpvalues == MAT_SHARE_NONZERO_PATTERN) {
      c->diag      = a->diag;
      c->free_diag = PETSC_FALSE;
    } else {
      ierr = PetscMalloc1(mbs+1,&c->diag);CHKERRQ(ierr);
      ierr = PetscLogObjectMemory((PetscObject)C,(mbs+1)*sizeof(PetscInt));CHKERRQ(ierr);
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
    ierr = PetscMalloc2(i+1,&c->compressedrow.i,i+1,&c->compressedrow.rindex);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)C,(2*i+1)*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscArraycpy(c->compressedrow.i,a->compressedrow.i,i+1);CHKERRQ(ierr);
    ierr = PetscArraycpy(c->compressedrow.rindex,a->compressedrow.rindex,i);CHKERRQ(ierr);
  } else {
    c->compressedrow.use    = PETSC_FALSE;
    c->compressedrow.i      = NULL;
    c->compressedrow.rindex = NULL;
  }
  C->nonzerostate = A->nonzerostate;

  ierr = PetscFunctionListDuplicate(((PetscObject)A)->qlist,&((PetscObject)C)->qlist);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatDuplicate_SeqBAIJ(Mat A,MatDuplicateOption cpvalues,Mat *B)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate(PetscObjectComm((PetscObject)A),B);CHKERRQ(ierr);
  ierr = MatSetSizes(*B,A->rmap->N,A->cmap->n,A->rmap->N,A->cmap->n);CHKERRQ(ierr);
  ierr = MatSetType(*B,MATSEQBAIJ);CHKERRQ(ierr);
  ierr = MatDuplicateNoCreate_SeqBAIJ(*B,A,cpvalues,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Used for both SeqBAIJ and SeqSBAIJ matrices */
PetscErrorCode MatLoad_SeqBAIJ_Binary(Mat mat,PetscViewer viewer)
{
  PetscInt       header[4],M,N,nz,bs,m,n,mbs,nbs,rows,cols,sum,i,j,k;
  PetscInt       *rowidxs,*colidxs;
  PetscScalar    *matvals;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerSetUp(viewer);CHKERRQ(ierr);

  /* read matrix header */
  ierr = PetscViewerBinaryRead(viewer,header,4,NULL,PETSC_INT);CHKERRQ(ierr);
  if (header[0] != MAT_FILE_CLASSID) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Not a matrix object in file");
  M = header[1]; N = header[2]; nz = header[3];
  if (M < 0) SETERRQ1(PetscObjectComm((PetscObject)viewer),PETSC_ERR_FILE_UNEXPECTED,"Matrix row size (%" PetscInt_FMT ") in file is negative",M);
  if (N < 0) SETERRQ1(PetscObjectComm((PetscObject)viewer),PETSC_ERR_FILE_UNEXPECTED,"Matrix column size (%" PetscInt_FMT ") in file is negative",N);
  if (nz < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Matrix stored in special format on disk, cannot load as SeqBAIJ");

  /* set block sizes from the viewer's .info file */
  ierr = MatLoad_Binary_BlockSizes(mat,viewer);CHKERRQ(ierr);
  /* set local and global sizes if not set already */
  if (mat->rmap->n < 0) mat->rmap->n = M;
  if (mat->cmap->n < 0) mat->cmap->n = N;
  if (mat->rmap->N < 0) mat->rmap->N = M;
  if (mat->cmap->N < 0) mat->cmap->N = N;
  ierr = PetscLayoutSetUp(mat->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(mat->cmap);CHKERRQ(ierr);

  /* check if the matrix sizes are correct */
  ierr = MatGetSize(mat,&rows,&cols);CHKERRQ(ierr);
  if (M != rows || N != cols) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED, "Matrix in file of different sizes (%" PetscInt_FMT ", %" PetscInt_FMT ") than the input matrix (%" PetscInt_FMT ", %" PetscInt_FMT ")",M,N,rows,cols);
  ierr = MatGetBlockSize(mat,&bs);CHKERRQ(ierr);
  ierr = MatGetLocalSize(mat,&m,&n);CHKERRQ(ierr);
  mbs = m/bs; nbs = n/bs;

  /* read in row lengths, column indices and nonzero values */
  ierr = PetscMalloc1(m+1,&rowidxs);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,rowidxs+1,m,NULL,PETSC_INT);CHKERRQ(ierr);
  rowidxs[0] = 0; for (i=0; i<m; i++) rowidxs[i+1] += rowidxs[i];
  sum = rowidxs[m];
  if (sum != nz) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Inconsistent matrix data in file: nonzeros = %" PetscInt_FMT ", sum-row-lengths = %" PetscInt_FMT,nz,sum);

  /* read in column indices and nonzero values */
  ierr = PetscMalloc2(rowidxs[m],&colidxs,nz,&matvals);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,colidxs,rowidxs[m],NULL,PETSC_INT);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,matvals,rowidxs[m],NULL,PETSC_SCALAR);CHKERRQ(ierr);

  { /* preallocate matrix storage */
    PetscBT   bt; /* helper bit set to count nonzeros */
    PetscInt  *nnz;
    PetscBool sbaij;

    ierr = PetscBTCreate(nbs,&bt);CHKERRQ(ierr);
    ierr = PetscCalloc1(mbs,&nnz);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)mat,MATSEQSBAIJ,&sbaij);CHKERRQ(ierr);
    for (i=0; i<mbs; i++) {
      ierr = PetscBTMemzero(nbs,bt);CHKERRQ(ierr);
      for (k=0; k<bs; k++) {
        PetscInt row = bs*i + k;
        for (j=rowidxs[row]; j<rowidxs[row+1]; j++) {
          PetscInt col = colidxs[j];
          if (!sbaij || col >= row)
            if (!PetscBTLookupSet(bt,col/bs)) nnz[i]++;
        }
      }
    }
    ierr = PetscBTDestroy(&bt);CHKERRQ(ierr);
    ierr = MatSeqBAIJSetPreallocation(mat,bs,0,nnz);CHKERRQ(ierr);
    ierr = MatSeqSBAIJSetPreallocation(mat,bs,0,nnz);CHKERRQ(ierr);
    ierr = PetscFree(nnz);CHKERRQ(ierr);
  }

  /* store matrix values */
  for (i=0; i<m; i++) {
    PetscInt row = i, s = rowidxs[i], e = rowidxs[i+1];
    ierr = (*mat->ops->setvalues)(mat,1,&row,e-s,colidxs+s,matvals+s,INSERT_VALUES);CHKERRQ(ierr);
  }

  ierr = PetscFree(rowidxs);CHKERRQ(ierr);
  ierr = PetscFree2(colidxs,matvals);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatLoad_SeqBAIJ(Mat mat,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      isbinary;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
  if (!isbinary) SETERRQ2(PetscObjectComm((PetscObject)viewer),PETSC_ERR_SUP,"Viewer type %s not yet supported for reading %s matrices",((PetscObject)viewer)->type_name,((PetscObject)mat)->type_name);
  ierr = MatLoad_SeqBAIJ_Binary(mat,viewer);CHKERRQ(ierr);
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

.seealso: MatCreate(), MatCreateSeqAIJ(), MatSetValues(), MatCreateBAIJ()
@*/
PetscErrorCode  MatCreateSeqBAIJ(MPI_Comm comm,PetscInt bs,PetscInt m,PetscInt n,PetscInt nz,const PetscInt nnz[],Mat *A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,m,n,m,n);CHKERRQ(ierr);
  ierr = MatSetType(*A,MATSEQBAIJ);CHKERRQ(ierr);
  ierr = MatSeqBAIJSetPreallocation(*A,bs,nz,(PetscInt*)nnz);CHKERRQ(ierr);
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

.seealso: MatCreate(), MatCreateSeqAIJ(), MatSetValues(), MatCreateBAIJ(), MatGetInfo()
@*/
PetscErrorCode  MatSeqBAIJSetPreallocation(Mat B,PetscInt bs,PetscInt nz,const PetscInt nnz[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B,MAT_CLASSID,1);
  PetscValidType(B,1);
  PetscValidLogicalCollectiveInt(B,bs,2);
  ierr = PetscTryMethod(B,"MatSeqBAIJSetPreallocation_C",(Mat,PetscInt,PetscInt,const PetscInt[]),(B,bs,nz,nnz));CHKERRQ(ierr);
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

.seealso: MatCreate(), MatCreateSeqBAIJ(), MatSetValues(), MatSeqBAIJSetPreallocation(), MATSEQBAIJ
@*/
PetscErrorCode  MatSeqBAIJSetPreallocationCSR(Mat B,PetscInt bs,const PetscInt i[],const PetscInt j[], const PetscScalar v[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B,MAT_CLASSID,1);
  PetscValidType(B,1);
  PetscValidLogicalCollectiveInt(B,bs,2);
  ierr = PetscTryMethod(B,"MatSeqBAIJSetPreallocationCSR_C",(Mat,PetscInt,const PetscInt[],const PetscInt[],const PetscScalar[]),(B,bs,i,j,v));CHKERRQ(ierr);
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

.seealso: MatCreate(), MatCreateBAIJ(), MatCreateSeqBAIJ()

@*/
PetscErrorCode  MatCreateSeqBAIJWithArrays(MPI_Comm comm,PetscInt bs,PetscInt m,PetscInt n,PetscInt i[],PetscInt j[],PetscScalar a[],Mat *mat)
{
  PetscErrorCode ierr;
  PetscInt       ii;
  Mat_SeqBAIJ    *baij;

  PetscFunctionBegin;
  if (bs != 1) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"block size %" PetscInt_FMT " > 1 is not supported yet",bs);
  if (m > 0 && i[0]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"i (row indices) must start with 0");

  ierr = MatCreate(comm,mat);CHKERRQ(ierr);
  ierr = MatSetSizes(*mat,m,n,m,n);CHKERRQ(ierr);
  ierr = MatSetType(*mat,MATSEQBAIJ);CHKERRQ(ierr);
  ierr = MatSeqBAIJSetPreallocation(*mat,bs,MAT_SKIP_ALLOCATION,NULL);CHKERRQ(ierr);
  baij = (Mat_SeqBAIJ*)(*mat)->data;
  ierr = PetscMalloc2(m,&baij->imax,m,&baij->ilen);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory((PetscObject)*mat,2*m*sizeof(PetscInt));CHKERRQ(ierr);

  baij->i = i;
  baij->j = j;
  baij->a = a;

  baij->singlemalloc = PETSC_FALSE;
  baij->nonew        = -1;             /*this indicates that inserting a new value in the matrix that generates a new nonzero is an error*/
  baij->free_a       = PETSC_FALSE;
  baij->free_ij      = PETSC_FALSE;

  for (ii=0; ii<m; ii++) {
    baij->ilen[ii] = baij->imax[ii] = i[ii+1] - i[ii];
    if (PetscUnlikelyDebug(i[ii+1] - i[ii] < 0)) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Negative row length in i (row indices) row = %" PetscInt_FMT " length = %" PetscInt_FMT,ii,i[ii+1] - i[ii]);
  }
  if (PetscDefined(USE_DEBUG)) {
    for (ii=0; ii<baij->i[m]; ii++) {
      if (j[ii] < 0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Negative column index at location = %" PetscInt_FMT " index = %" PetscInt_FMT,ii,j[ii]);
      if (j[ii] > n - 1) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Column index to large at location = %" PetscInt_FMT " index = %" PetscInt_FMT,ii,j[ii]);
    }
  }

  ierr = MatAssemblyBegin(*mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatCreateMPIMatConcatenateSeqMat_SeqBAIJ(MPI_Comm comm,Mat inmat,PetscInt n,MatReuse scall,Mat *outmat)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  if (size == 1 && scall == MAT_REUSE_MATRIX) {
    ierr = MatCopy(inmat,*outmat,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  } else {
    ierr = MatCreateMPIMatConcatenateSeqMat_MPIBAIJ(comm,inmat,n,scall,outmat);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
