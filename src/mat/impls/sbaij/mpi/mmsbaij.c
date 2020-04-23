
/*
   Support for the parallel SBAIJ matrix vector multiply
*/
#include <../src/mat/impls/sbaij/mpi/mpisbaij.h>

PetscErrorCode MatSetUpMultiply_MPISBAIJ(Mat mat)
{
  Mat_MPISBAIJ   *sbaij = (Mat_MPISBAIJ*)mat->data;
  Mat_SeqBAIJ    *B     = (Mat_SeqBAIJ*)(sbaij->B->data);
  PetscErrorCode ierr;
  PetscInt       Nbs = sbaij->Nbs,i,j,*aj = B->j,ec = 0,*garray,*sgarray;
  PetscInt       bs  = mat->rmap->bs,*stmp,mbs=sbaij->mbs, vec_size,nt;
  IS             from,to;
  Vec            gvec;
  PetscMPIInt    rank = sbaij->rank,lsize;
  PetscInt       *owners = sbaij->rangebs,*ec_owner,k;
  const PetscInt *sowners;
  PetscScalar    *ptr;
#if defined(PETSC_USE_CTABLE)
  PetscTable         gid1_lid1; /* one-based gid to lid table */
  PetscTablePosition tpos;
  PetscInt           gid,lid;
#else
  PetscInt           *indices;
#endif

  PetscFunctionBegin;
#if defined(PETSC_USE_CTABLE)
  ierr = PetscTableCreate(mbs,Nbs+1,&gid1_lid1);CHKERRQ(ierr);
  for (i=0; i<B->mbs; i++) {
    for (j=0; j<B->ilen[i]; j++) {
      PetscInt data,gid1 = aj[B->i[i]+j] + 1;
      ierr = PetscTableFind(gid1_lid1,gid1,&data);CHKERRQ(ierr);
      if (!data) {ierr = PetscTableAdd(gid1_lid1,gid1,++ec,INSERT_VALUES);CHKERRQ(ierr);}
    }
  }
  /* form array of columns we need */
  ierr = PetscMalloc1(ec,&garray);CHKERRQ(ierr);
  ierr = PetscTableGetHeadPosition(gid1_lid1,&tpos);CHKERRQ(ierr);
  while (tpos) {
    ierr = PetscTableGetNext(gid1_lid1,&tpos,&gid,&lid);CHKERRQ(ierr);
    gid--; lid--;
    garray[lid] = gid;
  }
  ierr = PetscSortInt(ec,garray);CHKERRQ(ierr);
  ierr = PetscTableRemoveAll(gid1_lid1);CHKERRQ(ierr);
  for (i=0; i<ec; i++) {ierr = PetscTableAdd(gid1_lid1,garray[i]+1,i+1,INSERT_VALUES);CHKERRQ(ierr);}
  /* compact out the extra columns in B */
  for (i=0; i<B->mbs; i++) {
    for (j=0; j<B->ilen[i]; j++) {
      PetscInt gid1 = aj[B->i[i] + j] + 1;
      ierr = PetscTableFind(gid1_lid1,gid1,&lid);CHKERRQ(ierr);
      lid--;
      aj[B->i[i]+j] = lid;
    }
  }
  ierr = PetscTableDestroy(&gid1_lid1);CHKERRQ(ierr);
  ierr = PetscMalloc2(2*ec,&sgarray,ec,&ec_owner);CHKERRQ(ierr);
  for (i=j=0; i<ec; i++) {
    while (garray[i]>=owners[j+1]) j++;
    ec_owner[i] = j;
  }
#else
  /* For the first stab we make an array as long as the number of columns */
  /* mark those columns that are in sbaij->B */
  ierr = PetscCalloc1(Nbs,&indices);CHKERRQ(ierr);
  for (i=0; i<mbs; i++) {
    for (j=0; j<B->ilen[i]; j++) {
      if (!indices[aj[B->i[i] + j]]) ec++;
      indices[aj[B->i[i] + j]] = 1;
    }
  }

  /* form arrays of columns we need */
  ierr = PetscMalloc1(ec,&garray);CHKERRQ(ierr);
  ierr = PetscMalloc2(2*ec,&sgarray,ec,&ec_owner);CHKERRQ(ierr);

  ec = 0;
  for (j=0; j<sbaij->size; j++) {
    for (i=owners[j]; i<owners[j+1]; i++) {
      if (indices[i]) {
        garray[ec]   = i;
        ec_owner[ec] = j;
        ec++;
      }
    }
  }

  /* make indices now point into garray */
  for (i=0; i<ec; i++) indices[garray[i]] = i;

  /* compact out the extra columns in B */
  for (i=0; i<mbs; i++) {
    for (j=0; j<B->ilen[i]; j++) aj[B->i[i] + j] = indices[aj[B->i[i] + j]];
  }
  ierr = PetscFree(indices);CHKERRQ(ierr);
#endif
  B->nbs = ec;
  ierr = PetscLayoutDestroy(&sbaij->B->cmap);CHKERRQ(ierr);
  ierr = PetscLayoutCreateFromSizes(PetscObjectComm((PetscObject)sbaij->B),ec*mat->rmap->bs,ec*mat->rmap->bs,mat->rmap->bs,&sbaij->B->cmap);CHKERRQ(ierr);

  ierr = VecScatterDestroy(&sbaij->sMvctx);CHKERRQ(ierr);
  /* create local vector that is used to scatter into */
  ierr = VecCreateSeq(PETSC_COMM_SELF,ec*bs,&sbaij->lvec);CHKERRQ(ierr);

  /* create two temporary index sets for building scatter-gather */
  ierr = PetscMalloc1(2*ec,&stmp);CHKERRQ(ierr);
  ierr = ISCreateBlock(PETSC_COMM_SELF,bs,ec,garray,PETSC_COPY_VALUES,&from);CHKERRQ(ierr);
  for (i=0; i<ec; i++) stmp[i] = i;
  ierr = ISCreateBlock(PETSC_COMM_SELF,bs,ec,stmp,PETSC_COPY_VALUES,&to);CHKERRQ(ierr);

  /* generate the scatter context
     -- Mvctx and lvec are not used by MatMult_MPISBAIJ(), but usefull for some applications */
  ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)mat),1,mat->cmap->n,mat->cmap->N,NULL,&gvec);CHKERRQ(ierr);
  ierr = VecScatterCreate(gvec,from,sbaij->lvec,to,&sbaij->Mvctx);CHKERRQ(ierr);
  ierr = VecDestroy(&gvec);CHKERRQ(ierr);

  sbaij->garray = garray;

  ierr = PetscLogObjectParent((PetscObject)mat,(PetscObject)sbaij->Mvctx);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)mat,(PetscObject)sbaij->lvec);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)mat,(PetscObject)from);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)mat,(PetscObject)to);CHKERRQ(ierr);

  ierr = ISDestroy(&from);CHKERRQ(ierr);
  ierr = ISDestroy(&to);CHKERRQ(ierr);

  /* create parallel vector that is used by SBAIJ matrix to scatter from/into */
  lsize = (mbs + ec)*bs;
  ierr  = VecCreateMPI(PetscObjectComm((PetscObject)mat),lsize,PETSC_DETERMINE,&sbaij->slvec0);CHKERRQ(ierr);
  ierr  = VecDuplicate(sbaij->slvec0,&sbaij->slvec1);CHKERRQ(ierr);
  ierr  = VecGetSize(sbaij->slvec0,&vec_size);CHKERRQ(ierr);

  ierr = VecGetOwnershipRanges(sbaij->slvec0,&sowners);CHKERRQ(ierr);

  /* x index in the IS sfrom */
  for (i=0; i<ec; i++) {
    j          = ec_owner[i];
    sgarray[i] = garray[i] + (sowners[j]/bs - owners[j]);
  }
  /* b index in the IS sfrom */
  k = sowners[rank]/bs + mbs;
  for (i=ec,j=0; i< 2*ec; i++,j++) sgarray[i] = k + j;
  ierr = ISCreateBlock(PETSC_COMM_SELF,bs,2*ec,sgarray,PETSC_COPY_VALUES,&from);CHKERRQ(ierr);

  /* x index in the IS sto */
  k = sowners[rank]/bs + mbs;
  for (i=0; i<ec; i++) stmp[i] = (k + i);
  /* b index in the IS sto */
  for (i=ec; i<2*ec; i++) stmp[i] = sgarray[i-ec];

  ierr = ISCreateBlock(PETSC_COMM_SELF,bs,2*ec,stmp,PETSC_COPY_VALUES,&to);CHKERRQ(ierr);

  ierr = VecScatterCreate(sbaij->slvec0,from,sbaij->slvec1,to,&sbaij->sMvctx);CHKERRQ(ierr);

  ierr = VecGetLocalSize(sbaij->slvec1,&nt);CHKERRQ(ierr);
  ierr = VecGetArray(sbaij->slvec1,&ptr);CHKERRQ(ierr);
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,1,bs*mbs,ptr,&sbaij->slvec1a);CHKERRQ(ierr);
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,1,nt-bs*mbs,ptr+bs*mbs,&sbaij->slvec1b);CHKERRQ(ierr);
  ierr = VecRestoreArray(sbaij->slvec1,&ptr);CHKERRQ(ierr);

  ierr = VecGetArray(sbaij->slvec0,&ptr);CHKERRQ(ierr);
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,1,nt-bs*mbs,ptr+bs*mbs,&sbaij->slvec0b);CHKERRQ(ierr);
  ierr = VecRestoreArray(sbaij->slvec0,&ptr);CHKERRQ(ierr);

  ierr = PetscFree(stmp);CHKERRQ(ierr);
  ierr = MPI_Barrier(PetscObjectComm((PetscObject)mat));CHKERRQ(ierr);

  ierr = PetscLogObjectParent((PetscObject)mat,(PetscObject)sbaij->sMvctx);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)mat,(PetscObject)sbaij->slvec0);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)mat,(PetscObject)sbaij->slvec1);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)mat,(PetscObject)sbaij->slvec0b);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)mat,(PetscObject)sbaij->slvec1a);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)mat,(PetscObject)sbaij->slvec1b);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)mat,(PetscObject)from);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)mat,(PetscObject)to);CHKERRQ(ierr);

  ierr = PetscLogObjectMemory((PetscObject)mat,(ec+1)*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = ISDestroy(&from);CHKERRQ(ierr);
  ierr = ISDestroy(&to);CHKERRQ(ierr);
  ierr = PetscFree2(sgarray,ec_owner);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
     Takes the local part of an already assembled MPISBAIJ matrix
   and disassembles it. This is to allow new nonzeros into the matrix
   that require more communication in the matrix vector multiply.
   Thus certain data-structures must be rebuilt.

   Kind of slow! But that's what application programmers get when
   they are sloppy.
*/
PetscErrorCode MatDisAssemble_MPISBAIJ(Mat A)
{
  Mat_MPISBAIJ   *baij  = (Mat_MPISBAIJ*)A->data;
  Mat            B      = baij->B,Bnew;
  Mat_SeqBAIJ    *Bbaij = (Mat_SeqBAIJ*)B->data;
  PetscErrorCode ierr;
  PetscInt       i,j,mbs=Bbaij->mbs,n = A->cmap->N,col,*garray=baij->garray;
  PetscInt       k,bs=A->rmap->bs,bs2=baij->bs2,*rvals,*nz,ec,m=A->rmap->n;
  MatScalar      *a = Bbaij->a;
  PetscScalar    *atmp;
#if defined(PETSC_USE_REAL_MAT_SINGLE)
  PetscInt l;
#endif

  PetscFunctionBegin;
#if defined(PETSC_USE_REAL_MAT_SINGLE)
  ierr = PetscMalloc1(A->rmap->bs,&atmp);
#endif
  /* free stuff related to matrix-vec multiply */
  ierr = VecGetSize(baij->lvec,&ec);CHKERRQ(ierr); /* needed for PetscLogObjectMemory below */
  ierr = VecDestroy(&baij->lvec);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&baij->Mvctx);CHKERRQ(ierr);

  ierr = VecDestroy(&baij->slvec0);CHKERRQ(ierr);
  ierr = VecDestroy(&baij->slvec0b);CHKERRQ(ierr);
  ierr = VecDestroy(&baij->slvec1);CHKERRQ(ierr);
  ierr = VecDestroy(&baij->slvec1a);CHKERRQ(ierr);
  ierr = VecDestroy(&baij->slvec1b);CHKERRQ(ierr);

  if (baij->colmap) {
#if defined(PETSC_USE_CTABLE)
    ierr = PetscTableDestroy(&baij->colmap);CHKERRQ(ierr);
#else
    ierr = PetscFree(baij->colmap);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)A,-Bbaij->nbs*sizeof(PetscInt));CHKERRQ(ierr);
#endif
  }

  /* make sure that B is assembled so we can access its values */
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* invent new B and copy stuff over */
  ierr = PetscMalloc1(mbs,&nz);CHKERRQ(ierr);
  for (i=0; i<mbs; i++) {
    nz[i] = Bbaij->i[i+1]-Bbaij->i[i];
  }
  ierr = MatCreate(PETSC_COMM_SELF,&Bnew);CHKERRQ(ierr);
  ierr = MatSetSizes(Bnew,m,n,m,n);CHKERRQ(ierr);
  ierr = MatSetType(Bnew,((PetscObject)B)->type_name);CHKERRQ(ierr);
  ierr = MatSeqBAIJSetPreallocation(Bnew,B->rmap->bs,0,nz);CHKERRQ(ierr);
  ierr = PetscFree(nz);CHKERRQ(ierr);

  if (Bbaij->nonew >= 0) { /* Inherit insertion error options (if positive). */
    ((Mat_SeqSBAIJ*)Bnew->data)->nonew = Bbaij->nonew;
  }

  /*
   Ensure that B's nonzerostate is monotonically increasing.
   Or should this follow the MatSetValues() loop to preserve B's nonzerstate across a MatDisAssemble() call?
   */
  Bnew->nonzerostate = B->nonzerostate;
  ierr = PetscMalloc1(bs,&rvals);CHKERRQ(ierr);
  for (i=0; i<mbs; i++) {
    rvals[0] = bs*i;
    for (j=1; j<bs; j++) rvals[j] = rvals[j-1] + 1;
    for (j=Bbaij->i[i]; j<Bbaij->i[i+1]; j++) {
      col = garray[Bbaij->j[j]]*bs;
      for (k=0; k<bs; k++) {
#if defined(PETSC_USE_REAL_MAT_SINGLE)
        for (l=0; l<bs; l++) atmp[l] = a[j*bs2+l];
#else
        atmp = a+j*bs2 + k*bs;
#endif
        ierr = MatSetValues_SeqSBAIJ(Bnew,bs,rvals,1,&col,atmp,B->insertmode);CHKERRQ(ierr);
        col++;
      }
    }
  }
#if defined(PETSC_USE_REAL_MAT_SINGLE)
  ierr = PetscFree(atmp);CHKERRQ(ierr);
#endif
  ierr = PetscFree(baij->garray);CHKERRQ(ierr);

  baij->garray = 0;

  ierr = PetscFree(rvals);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory((PetscObject)A,-ec*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)A,(PetscObject)Bnew);CHKERRQ(ierr);

  baij->B = Bnew;

  A->was_assembled = PETSC_FALSE;
  PetscFunctionReturn(0);
}
