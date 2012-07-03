#define PETSCMAT_DLL

#include "../src/mat/impls/sbaij/seq/sbaij.h"
#include "../src/mat/impls/sbaij/seq/sbstream/sbstream.h"

extern PetscErrorCode MatDestroy_SeqSBSTRM(Mat A);


#undef __FUNCT__  
#define __FUNCT__ "MatSolve_SeqSBSTRM_4_inplace"
PetscErrorCode MatSolve_SeqSBSTRM_4_inplace(Mat A,Vec bb,Vec xx)
{
  Mat_SeqSBAIJ   *a=(Mat_SeqSBAIJ*)A->data;
  IS             isrow=a->row;
  PetscInt       mbs=a->mbs,*ai=a->i,*aj=a->j,bs=A->rmap->bs,bs2=a->bs2;
  PetscErrorCode ierr;
  const PetscInt *r;
  PetscInt       nz,*vj,k,idx;  
  PetscScalar    *x,*b,x0,x1,x2,x3,*t,*tp;

  Mat_SeqSBSTRM      *sbstrm = (Mat_SeqSBSTRM *)A->spptr;
  MatScalar          *as=sbstrm->as,*diag;
  PetscScalar        tp0, tp1, tp2, tp3;
  const MatScalar    *v0, *v1, *v2, *v3;
  PetscInt           slen; 

  PetscFunctionBegin;

  slen = 4*(ai[mbs]-ai[0]);
  v0  = as + 16*ai[0];
  v1  = v0 + slen;
  v2  = v1 + slen;
  v3  = v2 + slen;

  ierr = VecGetArray(bb,&b);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  t  = a->solve_work;
  ierr = ISGetIndices(isrow,&r);CHKERRQ(ierr);

  /* solve U^T * D * y = b by forward substitution */
  tp = t;
  for (k=0; k<mbs; k++) { /* t <- perm(b) */
    idx   = 4*r[k];
    tp[0] = b[idx];
    tp[1] = b[idx+1];
    tp[2] = b[idx+2];
    tp[3] = b[idx+3];
    tp += 4;
  }
  
  for (k=0; k<mbs; k++){
    vj = aj + ai[k]; 
    tp = t + k*4;
    x0=tp[0]; x1=tp[1]; x2=tp[2]; x3=tp[3];
    nz = ai[k+1] - ai[k];  

    tp = t + (*vj)*4;
    while (nz--) {  
      tp[0] += v0[0]*x0 + v1[0]*x1 + v2[0]*x2 + v3[0]*x3;
      tp[1] += v0[1]*x0 + v1[1]*x1 + v2[1]*x2 + v3[1]*x3;
      tp[2] += v0[2]*x0 + v1[2]*x1 + v2[2]*x2 + v3[2]*x3;
      tp[3] += v0[3]*x0 + v1[3]*x1 + v2[3]*x2 + v3[3]*x3;
      vj++; tp = t + (*vj)*4;
      v0 += 4; v1 += 4; v2 += 4; v3 += 4;      
    }

    /* xk = inv(Dk)*(Dk*xk) */
    diag  = as+k*16;          /* ptr to inv(Dk) */
    tp    = t + k*4;
    tp[0] = diag[0]*x0 + diag[4]*x1 + diag[8]*x2 + diag[12]*x3;
    tp[1] = diag[1]*x0 + diag[5]*x1 + diag[9]*x2 + diag[13]*x3;
    tp[2] = diag[2]*x0 + diag[6]*x1 + diag[10]*x2+ diag[14]*x3;
    tp[3] = diag[3]*x0 + diag[7]*x1 + diag[11]*x2+ diag[15]*x3;
  }

  /* solve U*x = y by back substitution */   
  for (k=mbs-1; k>=0; k--){ 
    vj = aj + ai[k+1]; 
    tp    = t + k*4;    
    x0=tp[0]; x1=tp[1]; x2=tp[2]; x3=tp[3]; /* xk */ 
    nz = ai[k+1] - ai[k]; 
  
    tp = t + (*vj)*4;
    while (nz--) {
      /* xk += U(k,* */
      v0 -= 4; v1 -= 4; v2 -= 4; v3 -= 4;
      vj--; tp = t + (*vj)*4;
      tp0 = tp[0]; tp1 = tp[1]; tp2 = tp[2]; tp3 = tp[3];
      x0 += v0[3]*tp3 + v0[2]*tp2 + v0[1]*tp1 + v0[0]*tp0;
      x1 += v1[3]*tp3 + v1[2]*tp2 + v1[1]*tp1 + v1[0]*tp0;
      x2 += v2[3]*tp3 + v2[2]*tp2 + v2[1]*tp1 + v2[0]*tp0;
      x3 += v3[3]*tp3 + v3[2]*tp2 + v3[1]*tp1 + v3[0]*tp0;
    }
    tp    = t + k*4;    
    tp[0]=x0; tp[1]=x1; tp[2]=x2; tp[3]=x3;
    idx        = 4*r[k];
    x[idx]     = x0;
    x[idx+1]   = x1;
    x[idx+2]   = x2;
    x[idx+3]   = x3;
  }

  ierr = ISRestoreIndices(isrow,&r);CHKERRQ(ierr);
  ierr = VecRestoreArray(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = PetscLogFlops(4.0*bs2*a->nz - (bs+2.0*bs2)*mbs);CHKERRQ(ierr);
   PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatForwardSolve_SeqSBSTRM_4_NaturalOrdering"
PetscErrorCode MatForwardSolve_SeqSBSTRM_4_NaturalOrdering(PetscInt *ai,PetscInt *aj,MatScalar *aa,PetscInt mbs,PetscScalar *x)
{
  MatScalar      *diag;
  PetscScalar    *xp,x0,x1,x2,x3;
  PetscInt       nz,*vj,k;

  const MatScalar    *v0, *v1, *v2, *v3;
  PetscInt           slen; 

  PetscFunctionBegin;

  slen = 4*(ai[mbs]-ai[0]);
  v0  = aa + 16*ai[0];
  v1  = v0 + slen;
  v2  = v1 + slen;
  v3  = v2 + slen;

  for (k=0; k<mbs; k++){
    xp = x + k*4;
    x0=xp[0]; x1=xp[1]; x2=xp[2]; x3=xp[3]; /* Dk*xk = k-th block of x */
    nz = ai[k+1] - ai[k];  
    vj = aj + ai[k];
    xp = x + (*vj)*4;
    while (nz--) {
      /*  += U(k,^T*(Dk*xk) */      
      xp[0] += v0[0]*x0 + v1[0]*x1 + v2[0]*x2 + v3[0]*x3;
      xp[1] += v0[1]*x0 + v1[1]*x1 + v2[1]*x2 + v3[1]*x3;
      xp[2] += v0[2]*x0 + v1[2]*x1 + v2[2]*x2 + v3[2]*x3;
      xp[3] += v0[3]*x0 + v1[3]*x1 + v2[3]*x2 + v3[3]*x3;
      vj++; xp = x + (*vj)*4;
      v0 += 4; v1 += 4; v2 += 4; v3 += 4;      
    }
    /* xk = inv(Dk)*(Dk*xk) */
    diag = aa+k*16;          /* ptr to inv(Dk) */
    xp   = x + k*4;
    xp[0] = diag[0]*x0 + diag[4]*x1 + diag[8]*x2 + diag[12]*x3;
    xp[1] = diag[1]*x0 + diag[5]*x1 + diag[9]*x2 + diag[13]*x3;
    xp[2] = diag[2]*x0 + diag[6]*x1 + diag[10]*x2+ diag[14]*x3;
    xp[3] = diag[3]*x0 + diag[7]*x1 + diag[11]*x2+ diag[15]*x3;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatBackwardSolve_SeqSBSTRM_4_NaturalOrdering"
PetscErrorCode MatBackwardSolve_SeqSBSTRM_4_NaturalOrdering(PetscInt *ai,PetscInt *aj,MatScalar *aa,PetscInt mbs,PetscScalar *x)
{
  PetscScalar    *xp,x0,x1,x2,x3;
  PetscInt       nz,*vj,k;

  PetscScalar        xp0, xp1, xp2, xp3;
  const MatScalar    *v0, *v1, *v2, *v3;
  PetscInt           slen; 

  PetscFunctionBegin;
  slen = 4*(ai[mbs]-ai[0]);
  v0  = aa + 16*ai[0]+4*(ai[mbs]-ai[0]);
  v1  = v0 + slen;
  v2  = v1 + slen;
  v3  = v2 + slen;

  for (k=mbs-1; k>=0; k--){ 
    xp = x + k*4;
    x0=xp[0]; x1=xp[1]; x2=xp[2]; x3=xp[3]; /* xk */ 
    nz = ai[k+1] - ai[k];  
    vj = aj + ai[k+1];
    xp = x + (*vj)*4;
    while (nz--) {
      /* xk += U(k,* */
      v0 -= 4; v1 -= 4; v2 -= 4; v3 -=4;
      vj--; xp = x + (*vj)*4;
      xp0 = xp[0]; xp1 = xp[1]; xp2 = xp[2]; xp3 = xp[3];
      x0 += v0[3]*xp3 + v0[2]*xp2 + v0[1]*xp1 + v0[0]*xp0;
      x1 += v1[3]*xp3 + v1[2]*xp2 + v1[1]*xp1 + v1[0]*xp0;
      x2 += v2[3]*xp3 + v2[2]*xp2 + v2[1]*xp1 + v2[0]*xp0;
      x3 += v3[3]*xp3 + v3[2]*xp2 + v3[1]*xp1 + v3[0]*xp0;
    }
    xp = x + k*4;
    xp[0] = x0; xp[1] = x1; xp[2] = x2; xp[3] = x3;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSolve_SeqSBSTRM_4_NaturalOrdering_inplace"
PetscErrorCode MatSolve_SeqSBSTRM_4_NaturalOrdering_inplace(Mat A,Vec bb,Vec xx)
{
  Mat_SeqSBAIJ   *a=(Mat_SeqSBAIJ*)A->data;
  PetscInt       mbs=a->mbs,*ai=a->i,*aj=a->j,bs=A->rmap->bs,bs2=a->bs2;
  PetscScalar    *x,*b;
  PetscErrorCode ierr;

  Mat_SeqSBSTRM    *sbstrm = (Mat_SeqSBSTRM *)A->spptr;
  MatScalar        *as=sbstrm->as;

  PetscFunctionBegin; 
#if 0
  MatSolve_SeqSBSTRM_4_inplace(A, bb, xx);
#endif
  ierr = VecGetArray(bb,&b);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  /* solve U^T * D * y = b by forward substitution */
  ierr = PetscMemcpy(x,b,4*mbs*sizeof(PetscScalar));CHKERRQ(ierr); /* x <- b */
  ierr = MatForwardSolve_SeqSBSTRM_4_NaturalOrdering(ai,aj,as,mbs,x);CHKERRQ(ierr);
  /* solve U*x = y by back substitution */ 
  ierr = MatBackwardSolve_SeqSBSTRM_4_NaturalOrdering(ai,aj,as,mbs,x);CHKERRQ(ierr);
  ierr = VecRestoreArray(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);  
  ierr = PetscLogFlops(4.0*bs2*a->nz - (bs+2.0*bs2)*mbs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatForwardSolve_SeqSBSTRM_4_NaturalOrdering_inplace"
PetscErrorCode MatForwardSolve_SeqSBSTRM_4_NaturalOrdering_inplace(Mat A,Vec bb,Vec xx)
{
  Mat_SeqSBAIJ   *a=(Mat_SeqSBAIJ*)A->data;
  PetscInt       mbs=a->mbs,*ai=a->i,*aj=a->j,bs=A->rmap->bs,bs2=a->bs2;
  PetscScalar    *x,*b;
  PetscErrorCode ierr;

  Mat_SeqSBSTRM    *sbstrm = (Mat_SeqSBSTRM *)A->spptr;
  MatScalar        *as=sbstrm->as;

  PetscFunctionBegin;
  ierr = VecGetArray(bb,&b);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = PetscMemcpy(x,b,4*mbs*sizeof(PetscScalar));CHKERRQ(ierr); /* x <- b */
  ierr = MatForwardSolve_SeqSBSTRM_4_NaturalOrdering(ai,aj,as,mbs,x);CHKERRQ(ierr);
  ierr = VecRestoreArray(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);  
  ierr = PetscLogFlops(2.0*bs2*a->nz - bs*mbs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatBackwardSolve_SeqSBSTRM_4_NaturalOrdering_inplace"
PetscErrorCode MatBackwardSolve_SeqSBSTRM_4_NaturalOrdering_inplace(Mat A,Vec bb,Vec xx)
{
  Mat_SeqSBAIJ   *a=(Mat_SeqSBAIJ*)A->data;
  PetscInt       mbs=a->mbs,*ai=a->i,*aj=a->j,bs2=a->bs2;
  PetscScalar    *x,*b; 
  PetscErrorCode ierr;

  Mat_SeqSBSTRM    *sbstrm = (Mat_SeqSBSTRM *)A->spptr;
  MatScalar        *as=sbstrm->as;

  PetscFunctionBegin;  
  ierr = VecGetArray(bb,&b);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = PetscMemcpy(x,b,4*mbs*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = MatBackwardSolve_SeqSBSTRM_4_NaturalOrdering(ai,aj,as,mbs,x);CHKERRQ(ierr);
  ierr = VecRestoreArray(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);  
  ierr = PetscLogFlops(2.0*bs2*(a->nz-mbs));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSolve_SeqSBSTRM_5_inplace"
PetscErrorCode MatSolve_SeqSBSTRM_5_inplace(Mat A,Vec bb,Vec xx)
{
  Mat_SeqSBAIJ   *a=(Mat_SeqSBAIJ*)A->data;
  IS             isrow=a->row;
  PetscInt       mbs=a->mbs,*ai=a->i,*aj=a->j,bs=A->rmap->bs,bs2 = a->bs2;
  PetscErrorCode ierr;
  const PetscInt *r;
  PetscInt       nz,*vj,k,idx;  
  PetscScalar    *x,*b,x0,x1,x2,x3,x4,*t,*tp;

  Mat_SeqSBSTRM      *sbstrm = (Mat_SeqSBSTRM *)A->spptr;
  MatScalar          *as=sbstrm->as,*diag;
  PetscScalar        tp0, tp1, tp2, tp3, tp4;
  const MatScalar    *v0, *v1, *v2, *v3, *v4;
  PetscInt           slen; 

  PetscFunctionBegin;
  ierr = VecGetArray(bb,&b);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  t  = a->solve_work;
  ierr = ISGetIndices(isrow,&r);CHKERRQ(ierr);


  slen = 5*(ai[mbs]-ai[0]);
  v0  = as + 25*ai[0];
  v1  = v0 + slen;
  v2  = v1 + slen;
  v3  = v2 + slen;
  v4  = v3 + slen;

  /* solve U^T * D * y = b by forward substitution */
  tp = t; 
  for (k=0; k<mbs; k++) { /* t <- perm(b) */
    idx   = 5*r[k];
    tp[0] = b[idx];
    tp[1] = b[idx+1];
    tp[2] = b[idx+2];
    tp[3] = b[idx+3];
    tp[4] = b[idx+4];
    tp += 5; 
  }
  
  for (k=0; k<mbs; k++){
    vj = aj + ai[k]; 
    tp = t + k*5;
    x0=tp[0]; x1=tp[1]; x2=tp[2]; x3=tp[3]; x4=tp[4];
    nz = ai[k+1] - ai[k];  

    tp = t + (*vj)*5;
    while (nz--) {  
      tp[0] += v0[0]*x0 + v1[0]*x1 + v2[0]*x2 + v3[0]*x3 + v4[0]*x4;
      tp[1] += v0[1]*x0 + v1[1]*x1 + v2[1]*x2 + v3[1]*x3 + v4[1]*x4;
      tp[2] += v0[2]*x0 + v1[2]*x1 + v2[2]*x2 + v3[2]*x3 + v4[2]*x4;
      tp[3] += v0[3]*x0 + v1[3]*x1 + v2[3]*x2 + v3[3]*x3 + v4[3]*x4;
      tp[4] += v0[4]*x0 + v1[4]*x1 + v2[4]*x2 + v3[4]*x3 + v4[4]*x4;
      vj++; tp = t + (*vj)*5;
      v0 += 5; v1 += 5; v2 += 5; v3 +=5; v4 += 5;      
    }

    /* xk = inv(Dk)*(Dk*xk) */
    diag  = as+k*25;          /* ptr to inv(Dk) */
    tp    = t + k*5;
      tp[0] = diag[0]*x0 + diag[5]*x1 + diag[10]*x2 + diag[15]*x3 + diag[20]*x4;
      tp[1] = diag[1]*x0 + diag[6]*x1 + diag[11]*x2 + diag[16]*x3 + diag[21]*x4;
      tp[2] = diag[2]*x0 + diag[7]*x1 + diag[12]*x2 + diag[17]*x3 + diag[22]*x4;
      tp[3] = diag[3]*x0 + diag[8]*x1 + diag[13]*x2 + diag[18]*x3 + diag[23]*x4;
      tp[4] = diag[4]*x0 + diag[9]*x1 + diag[14]*x2 + diag[19]*x3 + diag[24]*x4;
  }

  /* solve U*x = y by back substitution */   
  for (k=mbs-1; k>=0; k--){ 
    vj = aj + ai[k+1]; 
    tp    = t + k*5;    
    x0=tp[0]; x1=tp[1]; x2=tp[2]; x3=tp[3]; x4=tp[4];/* xk */ 
    nz = ai[k+1] - ai[k]; 
  
    tp = t + (*vj)*5;
    while (nz--) {
      /* xk += U(k,* */
      v0 -= 5; v1 -= 5; v2 -= 5; v3 -=5; v4 -= 5;
      vj--; tp = t + (*vj)*5;
      tp0 = tp[0]; tp1 = tp[1]; tp2 = tp[2]; tp3 = tp[3]; tp4 = tp[4];
      x0 += v0[4]*tp4 + v0[3]*tp3 + v0[2]*tp2 + v0[1]*tp1 + v0[0]*tp0;
      x1 += v1[4]*tp4 + v1[3]*tp3 + v1[2]*tp2 + v1[1]*tp1 + v1[0]*tp0;
      x2 += v2[4]*tp4 + v2[3]*tp3 + v2[2]*tp2 + v2[1]*tp1 + v2[0]*tp0;
      x3 += v3[4]*tp4 + v3[3]*tp3 + v3[2]*tp2 + v3[1]*tp1 + v3[0]*tp0;
      x4 += v4[4]*tp4 + v4[3]*tp3 + v4[2]*tp2 + v4[1]*tp1 + v4[0]*tp0;
    }
    tp    = t + k*5;    
    tp[0]=x0; tp[1]=x1; tp[2]=x2; tp[3]=x3; tp[4]=x4;
    idx   = 5*r[k];
    x[idx]     = x0;
    x[idx+1]   = x1;
    x[idx+2]   = x2;
    x[idx+3]   = x3;
    x[idx+4]   = x4;
  }

  ierr = ISRestoreIndices(isrow,&r);CHKERRQ(ierr);
  ierr = VecRestoreArray(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr); 
  ierr = PetscLogFlops(4.0*bs2*a->nz - (bs+2.0*bs2)*mbs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatForwardSolve_SeqSBSTRM_5_NaturalOrdering"
PetscErrorCode MatForwardSolve_SeqSBSTRM_5_NaturalOrdering(PetscInt *ai,PetscInt *aj,MatScalar *aa,PetscInt mbs,PetscScalar *x)
{
  MatScalar      *diag;
  PetscScalar    *xp,x0,x1,x2,x3,x4;
  PetscInt       nz,*vj,k;

  const MatScalar   *v0, *v1, *v2, *v3, *v4;
  PetscInt           slen; 

  PetscFunctionBegin;
 
 

  slen = 5*(ai[mbs]-ai[0]);
  v0  = aa + 25*ai[0];
  v1  = v0 + slen;
  v2  = v1 + slen;
  v3  = v2 + slen;
  v4  = v3 + slen;

  for (k=0; k<mbs; k++){
    xp = x + k*5;
    x0=xp[0]; x1=xp[1]; x2=xp[2]; x3=xp[3]; x4=xp[4];/* Dk*xk = k-th block of x */
    nz = ai[k+1] - ai[k];  
    vj = aj + ai[k];
    xp = x + (*vj)*5;
    while (nz--) {
      /*  += U(k,^T*(Dk*xk) */      
      xp[0] += v0[0]*x0 + v1[0]*x1 + v2[0]*x2 + v3[0]*x3 + v4[0]*x4;
      xp[1] += v0[1]*x0 + v1[1]*x1 + v2[1]*x2 + v3[1]*x3 + v4[1]*x4;
      xp[2] += v0[2]*x0 + v1[2]*x1 + v2[2]*x2 + v3[2]*x3 + v4[2]*x4;
      xp[3] += v0[3]*x0 + v1[3]*x1 + v2[3]*x2 + v3[3]*x3 + v4[3]*x4;
      xp[4] += v0[4]*x0 + v1[4]*x1 + v2[4]*x2 + v3[4]*x3 + v4[4]*x4;
      vj++; xp = x + (*vj)*5;
      v0 += 5; v1 += 5; v2 += 5; v3 +=5; v4 += 5;      
    }
    /* xk = inv(Dk)*(Dk*xk) */
    diag = aa+k*25;          /* ptr to inv(Dk) */
    xp   = x + k*5;
    xp[0] = diag[0]*x0 + diag[5]*x1 + diag[10]*x2 + diag[15]*x3 + diag[20]*x4;
    xp[1] = diag[1]*x0 + diag[6]*x1 + diag[11]*x2 + diag[16]*x3 + diag[21]*x4;
    xp[2] = diag[2]*x0 + diag[7]*x1 + diag[12]*x2 + diag[17]*x3 + diag[22]*x4;
    xp[3] = diag[3]*x0 + diag[8]*x1 + diag[13]*x2 + diag[18]*x3 + diag[23]*x4;
    xp[4] = diag[4]*x0 + diag[9]*x1 + diag[14]*x2 + diag[19]*x3 + diag[24]*x4;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatBackwardSolve_SeqSBSTRM_5_NaturalOrdering"
PetscErrorCode MatBackwardSolve_SeqSBSTRM_5_NaturalOrdering(PetscInt *ai,PetscInt *aj,MatScalar *aa,PetscInt mbs,PetscScalar *x)
{
  PetscScalar    *xp,x0,x1,x2,x3,x4;
  PetscInt       nz,*vj,k;

  PetscScalar        xp0, xp1, xp2, xp3, xp4;
  const MatScalar    *v0, *v1, *v2, *v3, *v4;
  PetscInt           slen; 

  PetscFunctionBegin;


  slen = 5*(ai[mbs]-ai[0]);
  v0  = aa + 25*ai[0]+5*(ai[mbs]-ai[0]); 
  v1  = v0 + slen;     
  v2  = v1 + slen;     
  v3  = v2 + slen;     
  v4  = v3 + slen;     

  for (k=mbs-1; k>=0; k--){ 

    xp = x + k*5;
    x0=xp[0]; x1=xp[1]; x2=xp[2]; x3=xp[3]; x4=xp[4];/* xk */ 
    nz = ai[k+1] - ai[k];  

    vj = aj + ai[k+1];
    xp = x + (*vj)*5;

    while (nz--) {
      /* xk += U(k,* */
      v0 -= 5; v1 -= 5; v2 -= 5; v3 -=5; v4 -= 5; 
      vj--; xp = x + (*vj)*5;
      xp4 = xp[4]; xp3 = xp[3]; xp2 = xp[2]; xp1 = xp[1]; xp0 = xp[0]; 
      x0 += v0[4]*xp4 + v0[3]*xp3 + v0[2]*xp2 + v0[1]*xp1 + v0[0]*xp0;
      x1 += v1[4]*xp4 + v1[3]*xp3 + v1[2]*xp2 + v1[1]*xp1 + v1[0]*xp0;
      x2 += v2[4]*xp4 + v2[3]*xp3 + v2[2]*xp2 + v2[1]*xp1 + v2[0]*xp0;
      x3 += v3[4]*xp4 + v3[3]*xp3 + v3[2]*xp2 + v3[1]*xp1 + v3[0]*xp0;      
      x4 += v4[4]*xp4 + v4[3]*xp3 + v4[2]*xp2 + v4[1]*xp1 + v4[0]*xp0;
    }
    xp = x + k*5;
    xp[0]=x0; xp[1]=x1; xp[2]=x2; xp[3]=x3; xp[4]=x4;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSolve_SeqSBSTRM_5_NaturalOrdering_inplace"
PetscErrorCode MatSolve_SeqSBSTRM_5_NaturalOrdering_inplace(Mat A,Vec bb,Vec xx)
{
  Mat_SeqSBAIJ   *a=(Mat_SeqSBAIJ*)A->data;
  PetscInt       mbs=a->mbs,*ai=a->i,*aj=a->j,bs=A->rmap->bs,bs2 = a->bs2;
  PetscScalar    *x,*b;
  PetscErrorCode ierr;

  Mat_SeqSBSTRM    *sbstrm = (Mat_SeqSBSTRM *)A->spptr;
  MatScalar        *as=sbstrm->as;

  PetscFunctionBegin;
#if 0 
#endif 
  ierr = VecGetArray(bb,&b);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);

  /* solve U^T * D * y = b by forward substitution */
  ierr = PetscMemcpy(x,b,5*mbs*sizeof(PetscScalar));CHKERRQ(ierr); /* x <- b */
  ierr = MatForwardSolve_SeqSBSTRM_5_NaturalOrdering(ai,aj,as,mbs,x);CHKERRQ(ierr);

  /* solve U*x = y by back substitution */   
  ierr = MatBackwardSolve_SeqSBSTRM_5_NaturalOrdering(ai,aj,as,mbs,x);CHKERRQ(ierr);

  ierr = VecRestoreArray(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);  
  ierr = PetscLogFlops(4.0*bs2*a->nz - (bs+2.0*bs2)*mbs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatForwardSolve_SeqSBSTRM_5_NaturalOrdering_inplace"
PetscErrorCode MatForwardSolve_SeqSBSTRM_5_NaturalOrdering_inplace(Mat A,Vec bb,Vec xx)
{
  Mat_SeqSBAIJ   *a=(Mat_SeqSBAIJ*)A->data;
  PetscInt       mbs=a->mbs,*ai=a->i,*aj=a->j,bs=A->rmap->bs,bs2=a->bs2;
  PetscScalar    *x,*b;
  PetscErrorCode ierr;

  Mat_SeqSBSTRM    *sbstrm = (Mat_SeqSBSTRM *)A->spptr;
  MatScalar        *as=sbstrm->as;

  PetscFunctionBegin;  
  ierr = VecGetArray(bb,&b);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = PetscMemcpy(x,b,5*mbs*sizeof(PetscScalar));CHKERRQ(ierr); /* x <- b */
  ierr = MatForwardSolve_SeqSBSTRM_5_NaturalOrdering(ai,aj,as,mbs,x);CHKERRQ(ierr);
  ierr = VecRestoreArray(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);  
  ierr = PetscLogFlops(2.0*bs2*a->nz - bs*mbs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatBackwardSolve_SeqSBSTRM_5_NaturalOrdering_inplace"
PetscErrorCode MatBackwardSolve_SeqSBSTRM_5_NaturalOrdering_inplace(Mat A,Vec bb,Vec xx)
{
  Mat_SeqSBAIJ   *a=(Mat_SeqSBAIJ*)A->data;
  PetscInt       mbs=a->mbs,*ai=a->i,*aj=a->j,bs2=a->bs2;
  PetscScalar    *x,*b;
  PetscErrorCode ierr;
  Mat_SeqSBSTRM    *sbstrm = (Mat_SeqSBSTRM *)A->spptr;
  MatScalar        *as=sbstrm->as;

  PetscFunctionBegin;
  ierr = VecGetArray(bb,&b);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = PetscMemcpy(x,b,5*mbs*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = MatBackwardSolve_SeqSBSTRM_5_NaturalOrdering(ai,aj,as,mbs,x);CHKERRQ(ierr);
  ierr = VecRestoreArray(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);  
  ierr = PetscLogFlops(2.0*bs2*(a->nz-mbs));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SeqSBSTRM_convertFact_sbstrm"
PetscErrorCode SeqSBSTRM_convertFact_sbstrm(Mat F)
{
  Mat_SeqSBAIJ   *a = (Mat_SeqSBAIJ *) F->data;
  Mat_SeqSBSTRM  *sbstrm = (Mat_SeqSBSTRM*) F->spptr;
  PetscInt       m = a->mbs, bs = F->rmap->bs;
  PetscInt       *ai = a->i;
  PetscInt       i,j,ib,jb;
  MatScalar      *aa = a->a, *aau, *asu;
  PetscErrorCode ierr;
  PetscInt  bs2, rbs,  cbs, blen, slen;
  PetscScalar **asp ;

  PetscFunctionBegin;
  sbstrm->rbs = bs;
  sbstrm->cbs = bs;

  rbs = cbs = bs;
  bs2 = rbs*cbs;
  blen = ai[m]-ai[0];
  slen = blen*cbs;

  if(sbstrm->as) {
      ierr = PetscFree(sbstrm->as);CHKERRQ(ierr);
  }
  ierr = PetscMalloc(bs2*ai[m]*sizeof(MatScalar), &sbstrm->as);CHKERRQ(ierr);
  ierr = PetscMalloc(rbs*sizeof(MatScalar *), &asp);CHKERRQ(ierr);

  asu = sbstrm->as; 
  for (i=0; i<m*bs2; i++) asu[i] = aa[i];

  asu = sbstrm->as + ai[0]*bs2;
  aau = aa         + ai[0]*bs2;

  for(i=0;i<rbs;i++) asp[i] = asu + i*slen;

  for(j=0;j<blen;j++) {
     for (jb=0; jb<cbs; jb++){
     for (ib=0; ib<rbs; ib++){
         asp[ib][j*cbs+jb] = aau[j*bs2+jb*rbs+ib];
     }}
  }

  switch (bs){
    case 4:
       F->ops->forwardsolve   = MatForwardSolve_SeqSBSTRM_4_NaturalOrdering_inplace;
       F->ops->backwardsolve  = MatBackwardSolve_SeqSBSTRM_4_NaturalOrdering_inplace;
       F->ops->solve          = MatSolve_SeqSBSTRM_4_NaturalOrdering_inplace;
       break;
    case 5:
       F->ops->forwardsolve   = MatForwardSolve_SeqSBSTRM_5_NaturalOrdering_inplace;
       F->ops->backwardsolve  = MatBackwardSolve_SeqSBSTRM_5_NaturalOrdering_inplace;
       F->ops->solve          = MatSolve_SeqSBSTRM_5_NaturalOrdering_inplace;
       break;
    default:
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"not supported for block size %D",bs);
  }
#if 0
#endif

  ierr = PetscFree(asp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*=========================================================*/ 

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatFactorGetSolverPackage_seqsbaij_sbstrm"
PetscErrorCode MatFactorGetSolverPackage_seqsbaij_sbstrm(Mat A,const MatSolverPackage *type)
{   
  PetscFunctionBegin;
  *type = MATSOLVERSBSTRM;
  PetscFunctionReturn(0);
}   
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "MatCholeskyFactorNumeric_sbstrm"
PetscErrorCode MatCholeskyFactorNumeric_sbstrm(Mat F,Mat A,const MatFactorInfo *info)
{
  PetscInt          bs = A->rmap->bs;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  switch (bs){
    case 4:
       ierr = MatCholeskyFactorNumeric_SeqSBAIJ_4_NaturalOrdering(F,A,info);CHKERRQ(ierr);
       break;
    case 5:
       ierr = MatCholeskyFactorNumeric_SeqSBAIJ_5_NaturalOrdering(F,A,info);CHKERRQ(ierr);
       break;
    default:
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"not supported for block size %D",bs);
  }
  
  ierr = SeqSBSTRM_convertFact_sbstrm(F);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
/*=========================================================*/ 
#undef __FUNCT__  
#define __FUNCT__ "MatICCFactorSymbolic_sbstrm"
PetscErrorCode MatICCFactorSymbolic_sbstrm(Mat B,Mat A,IS perm,const MatFactorInfo *info)   
{
  PetscInt ierr;
  PetscFunctionBegin;
  ierr = (MatICCFactorSymbolic_SeqSBAIJ)(B,A,perm,info);CHKERRQ(ierr);
  B->ops->choleskyfactornumeric  = MatCholeskyFactorNumeric_sbstrm;
  PetscFunctionReturn(0);
}
/*=========================================================*/ 
#undef __FUNCT__  
#define __FUNCT__ "MatCholeskyFactorSymbolic_sbstrm"
PetscErrorCode MatCholeskyFactorSymbolic_sbstrm(Mat B,Mat A,IS perm,const MatFactorInfo *info)   
{
  PetscInt ierr;
  PetscFunctionBegin;
  ierr = (MatCholeskyFactorSymbolic_SeqSBAIJ)(B,A,perm,info);CHKERRQ(ierr);
  B->ops->choleskyfactornumeric  = MatCholeskyFactorNumeric_sbstrm;
  PetscFunctionReturn(0);
}
/*=========================================================*/ 

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatGetFactor_seqsbaij_sbstrm"
PetscErrorCode MatGetFactor_seqsbaij_sbstrm(Mat A,MatFactorType ftype,Mat *F)
{
  Mat            B;
  PetscInt       bs = A->rmap->bs;
  Mat_SeqSBSTRM   *sbstrm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (A->cmap->N != A->rmap->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Must be square matrix, rows %D columns %D",A->rmap->n,A->cmap->n);
  ierr = MatCreate(((PetscObject)A)->comm,&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,A->rmap->n,A->cmap->n,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetType(B,((PetscObject)A)->type_name);CHKERRQ(ierr);
  ierr = MatSeqSBAIJSetPreallocation_SeqSBAIJ(B,bs,MAT_SKIP_ALLOCATION,PETSC_NULL);CHKERRQ(ierr);


  B->ops->iccfactorsymbolic       = MatICCFactorSymbolic_sbstrm;
  B->ops->choleskyfactorsymbolic  = MatCholeskyFactorSymbolic_sbstrm;
  B->ops->choleskyfactornumeric   = MatCholeskyFactorNumeric_sbstrm;

  B->ops->destroy                 = MatDestroy_SeqSBSTRM;
  B->factortype                   = ftype;
  B->assembled                    = PETSC_TRUE;  /* required by -ksp_view */
  B->preallocated                 = PETSC_TRUE;

  ierr = PetscNewLog(B,Mat_SeqSBSTRM,&sbstrm);CHKERRQ(ierr);

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatFactorGetSolverPackage_C","MatFactorGetSolverPackage_seqsbaij_sbstrm",MatFactorGetSolverPackage_seqsbaij_sbstrm);CHKERRQ(ierr);

  B->spptr = sbstrm;
  *F = B;
  PetscFunctionReturn(0);
}
EXTERN_C_END


