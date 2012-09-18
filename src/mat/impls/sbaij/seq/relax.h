
/*
    This is included by sbaij.c to generate unsigned short and regular versions of these two functions
*/
#undef __FUNCT__
#if defined(USESHORT)
#define __FUNCT__ "MatMult_SeqSBAIJ_1_Hermitian_ushort"
PetscErrorCode MatMult_SeqSBAIJ_1_Hermitian_ushort(Mat A,Vec xx,Vec zz)
#else
#define __FUNCT__ "MatMult_SeqSBAIJ_1_Hermitian"
PetscErrorCode MatMult_SeqSBAIJ_1_Hermitian(Mat A,Vec xx,Vec zz)
#endif
{
  Mat_SeqSBAIJ         *a = (Mat_SeqSBAIJ*)A->data;
  const PetscScalar    *x;
  PetscScalar          *z,x1,sum;
  const MatScalar      *v;
  MatScalar            vj;
  PetscErrorCode       ierr;
  PetscInt             mbs=a->mbs,i,j,nz;
  const PetscInt       *ai=a->i;
#if defined(USESHORT)
  const unsigned short *ib=a->jshort;
  unsigned short       ibt;
#else
  const PetscInt       *ib=a->j;
  PetscInt             ibt;
#endif
  PetscInt             nonzerorow = 0,jmin;

  PetscFunctionBegin;
  ierr = VecSet(zz,0.0);CHKERRQ(ierr);
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(zz,&z);CHKERRQ(ierr);

  v  = a->a;
  for (i=0; i<mbs; i++) {
    nz   = ai[i+1] - ai[i];  /* length of i_th row of A */
    if (!nz) continue; /* Move to the next row if the current row is empty */
    nonzerorow++;
    x1   = x[i];
    sum = 0.0;
    jmin = 0;
    if (ib[0] == i) {
      sum  = v[0]*x1;          /* diagonal term */
      jmin++;
    }
    for (j=jmin; j<nz; j++) {
      ibt  = ib[j];
      vj   = v[j];
      sum += vj * x[ibt];   /* (strict upper triangular part of A)*x  */
      z[ibt] += PetscConj(v[j]) * x1;    /* (strict lower triangular part of A)*x  */
    }
    z[i] += sum;
    v    +=    nz;
    ib   += nz;
  }

  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&z);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*(2.0*a->nz - nonzerorow) - nonzerorow);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#if defined(USESHORT)
#define __FUNCT__ "MatMult_SeqSBAIJ_1_ushort"
PetscErrorCode MatMult_SeqSBAIJ_1_ushort(Mat A,Vec xx,Vec zz)
#else
#define __FUNCT__ "MatMult_SeqSBAIJ_1"
PetscErrorCode MatMult_SeqSBAIJ_1(Mat A,Vec xx,Vec zz)
#endif
{
  Mat_SeqSBAIJ         *a = (Mat_SeqSBAIJ*)A->data;
  const PetscScalar    *x;
  PetscScalar          *z,x1,sum;
  const MatScalar      *v;
  MatScalar            vj;
  PetscErrorCode       ierr;
  PetscInt             mbs=a->mbs,i,j,nz;
  const PetscInt       *ai=a->i;
#if defined(USESHORT)
  const unsigned short *ib=a->jshort;
  unsigned short       ibt;
#else
  const PetscInt       *ib=a->j;
  PetscInt             ibt;
#endif
  PetscInt             nonzerorow=0,jmin;

  PetscFunctionBegin;
  ierr = VecSet(zz,0.0);CHKERRQ(ierr);
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(zz,&z);CHKERRQ(ierr);

  v  = a->a;
  for (i=0; i<mbs; i++) {
    nz   = ai[i+1] - ai[i];        /* length of i_th row of A */
    if (!nz) continue; /* Move to the next row if the current row is empty */
    nonzerorow++;
    sum = 0.0;
    jmin = 0;
    x1   = x[i];
    if (ib[0] == i) {
      sum  = v[0]*x1;                /* diagonal term */
      jmin++;
    }
    PetscPrefetchBlock(ib+nz,nz,0,PETSC_PREFETCH_HINT_NTA); /* Indices for the next row (assumes same size as this one) */
    PetscPrefetchBlock(v+nz,nz,0,PETSC_PREFETCH_HINT_NTA);  /* Entries for the next row */
    for (j=jmin; j<nz; j++) {
      ibt     = ib[j];
      vj      = v[j];
      z[ibt] += vj * x1;       /* (strict lower triangular part of A)*x  */
      sum    += vj * x[ibt];   /* (strict upper triangular part of A)*x  */
    }
    z[i] += sum;
    v    += nz;
    ib   += nz;
  }

  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&z);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*(2.0*a->nz - nonzerorow) - nonzerorow);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#if defined(USESHORT)
#define __FUNCT__ "MatSOR_SeqSBAIJ_ushort"
PetscErrorCode MatSOR_SeqSBAIJ_ushort(Mat A,Vec bb,PetscReal omega,MatSORType flag,PetscReal fshift,PetscInt its,PetscInt lits,Vec xx)
#else
#define __FUNCT__ "MatSOR_SeqSBAIJ"
PetscErrorCode MatSOR_SeqSBAIJ(Mat A,Vec bb,PetscReal omega,MatSORType flag,PetscReal fshift,PetscInt its,PetscInt lits,Vec xx)
#endif
{
  Mat_SeqSBAIJ         *a = (Mat_SeqSBAIJ*)A->data;
  const MatScalar      *aa=a->a,*v,*v1,*aidiag;
  PetscScalar          *x,*t,sum;
  const PetscScalar    *b;
  MatScalar            tmp;
  PetscErrorCode       ierr;
  PetscInt             m=a->mbs,bs=A->rmap->bs,j;
  const PetscInt       *ai=a->i;
#if defined(USESHORT)
  const unsigned short *aj=a->jshort,*vj,*vj1;
#else
  const PetscInt       *aj=a->j,*vj,*vj1;
#endif
  PetscInt             nz,nz1,i;

  PetscFunctionBegin;
  if (flag & SOR_EISENSTAT) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support yet for Eisenstat");

  its = its*lits;
  if (its <= 0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Relaxation requires global its %D and local its %D both positive",its,lits);

  if (bs > 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"SSOR for block size > 1 is not yet implemented");

  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArrayRead(bb,&b);CHKERRQ(ierr);

  if (!a->idiagvalid) {
    if (!a->idiag) {
      ierr = PetscMalloc(m*sizeof(PetscScalar),&a->idiag);CHKERRQ(ierr);
    }
    for (i=0; i<a->mbs; i++) a->idiag[i] = 1.0/a->a[a->i[i]];
    a->idiagvalid = PETSC_TRUE;
  }

  if (!a->sor_work) {
    ierr = PetscMalloc(m*sizeof(PetscScalar),&a->sor_work);CHKERRQ(ierr);
  }
  t = a->sor_work;

  aidiag = a->idiag;

  if (flag == SOR_APPLY_UPPER) {
    /* apply (U + D/omega) to the vector */
    PetscScalar d;
    for (i=0; i<m; i++) {
      d    = fshift + aa[ai[i]];
      nz   = ai[i+1] - ai[i] - 1;
      vj   = aj + ai[i] + 1;
      v    = aa + ai[i] + 1;
      sum  = b[i]*d/omega;
      PetscSparseDensePlusDot(sum,b,v,vj,nz);
      x[i] = sum;
    }
    ierr = PetscLogFlops(a->nz);CHKERRQ(ierr);
  }

  if (flag & SOR_ZERO_INITIAL_GUESS) {
    if (flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP){
      ierr = PetscMemcpy(t,b,m*sizeof(PetscScalar));CHKERRQ(ierr);

      v  = aa + 1;
      vj = aj + 1;
      for (i=0; i<m; i++){
        nz = ai[i+1] - ai[i] - 1;
        tmp = - (x[i] = omega*t[i]*aidiag[i]);
        for (j=0; j<nz; j++) {
          t[vj[j]] += tmp*v[j];
        }
        v  += nz + 1;
        vj += nz + 1;
      }
      ierr = PetscLogFlops(2*a->nz);CHKERRQ(ierr);
    }

    if (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP){
      int nz2;
      if (!(flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP)){
#if defined(PETSC_USE_BACKWARD_LOOP)
	v  = aa + ai[m] - 1;
	vj = aj + ai[m] - 1;
	for (i=m-1; i>=0; i--){
          sum = b[i];
	  nz  = ai[i+1] - ai[i] - 1;
          {PetscInt __i;for (__i=0;__i<nz;__i++) sum -= v[-__i] * x[vj[-__i]];}
#else
	v  = aa + ai[m-1] + 1;
	vj = aj + ai[m-1] + 1;
	nz = 0;
	for (i=m-1; i>=0; i--){
          sum = b[i];
          nz2 = ai[i] - ai[i-1] - 1;
          PETSC_Prefetch(v-nz2-1,0,PETSC_PREFETCH_HINT_NTA);
          PETSC_Prefetch(vj-nz2-1,0,PETSC_PREFETCH_HINT_NTA);
          PetscSparseDenseMinusDot(sum,x,v,vj,nz);
          nz   = nz2;
#endif
          x[i] = omega*sum*aidiag[i];
	  v  -= nz + 1;
	  vj -= nz + 1;
	}
	ierr = PetscLogFlops(2*a->nz);CHKERRQ(ierr);
      } else {
        v  = aa + ai[m-1] + 1;
	vj = aj + ai[m-1] + 1;
	nz = 0;
	for (i=m-1; i>=0; i--){
          sum = t[i];
	  nz2 = ai[i] - ai[i-1] - 1;
	  PETSC_Prefetch(v-nz2-1,0,PETSC_PREFETCH_HINT_NTA);
	  PETSC_Prefetch(vj-nz2-1,0,PETSC_PREFETCH_HINT_NTA);
	  PetscSparseDenseMinusDot(sum,x,v,vj,nz);
          x[i] = (1-omega)*x[i] + omega*sum*aidiag[i];
	  nz  = nz2;
	  v  -= nz + 1;
	  vj -= nz + 1;
	}
	ierr = PetscLogFlops(2*a->nz);CHKERRQ(ierr);
      }
    }
    its--;
  }

  while (its--) {
    /*
       forward sweep:
       for i=0,...,m-1:
         sum[i] = (b[i] - U(i,:)x )/d[i];
         x[i]   = (1-omega)x[i] + omega*sum[i];
         b      = b - x[i]*U^T(i,:);

    */
    if (flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP){
      ierr = PetscMemcpy(t,b,m*sizeof(PetscScalar));CHKERRQ(ierr);

      for (i=0; i<m; i++){
        v  = aa + ai[i] + 1; v1=v;
        vj = aj + ai[i] + 1; vj1=vj;
        nz = ai[i+1] - ai[i] - 1; nz1=nz;
        sum = t[i];
        ierr = PetscLogFlops(4.0*nz-2);CHKERRQ(ierr);
        while (nz1--) sum -= (*v1++)*x[*vj1++];
        x[i] = (1-omega)*x[i] + omega*sum*aidiag[i];
        while (nz--) t[*vj++] -= x[i]*(*v++);
      }
    }

    if (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP){
      /*
       backward sweep:
       b = b - x[i]*U^T(i,:), i=0,...,n-2
       for i=m-1,...,0:
         sum[i] = (b[i] - U(i,:)x )/d[i];
         x[i]   = (1-omega)x[i] + omega*sum[i];
      */
      /* if there was a forward sweep done above then I thing the next two for loops are not needed */
      ierr = PetscMemcpy(t,b,m*sizeof(PetscScalar));CHKERRQ(ierr);

      for (i=0; i<m-1; i++){  /* update rhs */
        v  = aa + ai[i] + 1;
        vj = aj + ai[i] + 1;
        nz = ai[i+1] - ai[i] - 1;
        ierr = PetscLogFlops(2.0*nz-1);CHKERRQ(ierr);
        while (nz--) t[*vj++] -= x[i]*(*v++);
      }
      for (i=m-1; i>=0; i--){
        v  = aa + ai[i] + 1;
        vj = aj + ai[i] + 1;
        nz = ai[i+1] - ai[i] - 1;
        ierr = PetscLogFlops(2.0*nz-1);CHKERRQ(ierr);
        sum = t[i];
        while (nz--) sum -= x[*vj++]*(*v++);
        x[i] =   (1-omega)*x[i] + omega*sum*aidiag[i];
      }
    }
  }

  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(bb,&b);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
