
/*
   Support for the parallel SBAIJ matrix vector multiply
*/
#include <../src/mat/impls/sbaij/mpi/mpisbaij.h>

PetscErrorCode MatSetUpMultiply_MPISBAIJ(Mat mat)
{
  Mat_MPISBAIJ   *sbaij = (Mat_MPISBAIJ*)mat->data;
  Mat_SeqBAIJ    *B     = (Mat_SeqBAIJ*)(sbaij->B->data);
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
  CHKERRQ(PetscTableCreate(mbs,Nbs+1,&gid1_lid1));
  for (i=0; i<B->mbs; i++) {
    for (j=0; j<B->ilen[i]; j++) {
      PetscInt data,gid1 = aj[B->i[i]+j] + 1;
      CHKERRQ(PetscTableFind(gid1_lid1,gid1,&data));
      if (!data) CHKERRQ(PetscTableAdd(gid1_lid1,gid1,++ec,INSERT_VALUES));
    }
  }
  /* form array of columns we need */
  CHKERRQ(PetscMalloc1(ec,&garray));
  CHKERRQ(PetscTableGetHeadPosition(gid1_lid1,&tpos));
  while (tpos) {
    CHKERRQ(PetscTableGetNext(gid1_lid1,&tpos,&gid,&lid));
    gid--; lid--;
    garray[lid] = gid;
  }
  CHKERRQ(PetscSortInt(ec,garray));
  CHKERRQ(PetscTableRemoveAll(gid1_lid1));
  for (i=0; i<ec; i++) CHKERRQ(PetscTableAdd(gid1_lid1,garray[i]+1,i+1,INSERT_VALUES));
  /* compact out the extra columns in B */
  for (i=0; i<B->mbs; i++) {
    for (j=0; j<B->ilen[i]; j++) {
      PetscInt gid1 = aj[B->i[i] + j] + 1;
      CHKERRQ(PetscTableFind(gid1_lid1,gid1,&lid));
      lid--;
      aj[B->i[i]+j] = lid;
    }
  }
  CHKERRQ(PetscTableDestroy(&gid1_lid1));
  CHKERRQ(PetscMalloc2(2*ec,&sgarray,ec,&ec_owner));
  for (i=j=0; i<ec; i++) {
    while (garray[i]>=owners[j+1]) j++;
    ec_owner[i] = j;
  }
#else
  /* For the first stab we make an array as long as the number of columns */
  /* mark those columns that are in sbaij->B */
  CHKERRQ(PetscCalloc1(Nbs,&indices));
  for (i=0; i<mbs; i++) {
    for (j=0; j<B->ilen[i]; j++) {
      if (!indices[aj[B->i[i] + j]]) ec++;
      indices[aj[B->i[i] + j]] = 1;
    }
  }

  /* form arrays of columns we need */
  CHKERRQ(PetscMalloc1(ec,&garray));
  CHKERRQ(PetscMalloc2(2*ec,&sgarray,ec,&ec_owner));

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
  CHKERRQ(PetscFree(indices));
#endif
  B->nbs = ec;
  CHKERRQ(PetscLayoutDestroy(&sbaij->B->cmap));
  CHKERRQ(PetscLayoutCreateFromSizes(PetscObjectComm((PetscObject)sbaij->B),ec*mat->rmap->bs,ec*mat->rmap->bs,mat->rmap->bs,&sbaij->B->cmap));

  CHKERRQ(VecScatterDestroy(&sbaij->sMvctx));
  /* create local vector that is used to scatter into */
  CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,ec*bs,&sbaij->lvec));

  /* create two temporary index sets for building scatter-gather */
  CHKERRQ(PetscMalloc1(2*ec,&stmp));
  CHKERRQ(ISCreateBlock(PETSC_COMM_SELF,bs,ec,garray,PETSC_COPY_VALUES,&from));
  for (i=0; i<ec; i++) stmp[i] = i;
  CHKERRQ(ISCreateBlock(PETSC_COMM_SELF,bs,ec,stmp,PETSC_COPY_VALUES,&to));

  /* generate the scatter context
     -- Mvctx and lvec are not used by MatMult_MPISBAIJ(), but have other uses, such as in MatDiagonalScale_MPISBAIJ */
  CHKERRQ(VecCreateMPIWithArray(PetscObjectComm((PetscObject)mat),1,mat->cmap->n,mat->cmap->N,NULL,&gvec));
  CHKERRQ(VecScatterCreate(gvec,from,sbaij->lvec,to,&sbaij->Mvctx));
  CHKERRQ(VecDestroy(&gvec));

  sbaij->garray = garray;

  CHKERRQ(PetscLogObjectParent((PetscObject)mat,(PetscObject)sbaij->Mvctx));
  CHKERRQ(PetscLogObjectParent((PetscObject)mat,(PetscObject)sbaij->lvec));
  CHKERRQ(PetscLogObjectParent((PetscObject)mat,(PetscObject)from));
  CHKERRQ(PetscLogObjectParent((PetscObject)mat,(PetscObject)to));

  CHKERRQ(ISDestroy(&from));
  CHKERRQ(ISDestroy(&to));

  /* create parallel vector that is used by SBAIJ matrix to scatter from/into */
  lsize = (mbs + ec)*bs;
  CHKERRQ(VecCreateMPI(PetscObjectComm((PetscObject)mat),lsize,PETSC_DETERMINE,&sbaij->slvec0));
  CHKERRQ(VecDuplicate(sbaij->slvec0,&sbaij->slvec1));
  CHKERRQ(VecGetSize(sbaij->slvec0,&vec_size));

  CHKERRQ(VecGetOwnershipRanges(sbaij->slvec0,&sowners));

  /* x index in the IS sfrom */
  for (i=0; i<ec; i++) {
    j          = ec_owner[i];
    sgarray[i] = garray[i] + (sowners[j]/bs - owners[j]);
  }
  /* b index in the IS sfrom */
  k = sowners[rank]/bs + mbs;
  for (i=ec,j=0; i< 2*ec; i++,j++) sgarray[i] = k + j;
  CHKERRQ(ISCreateBlock(PETSC_COMM_SELF,bs,2*ec,sgarray,PETSC_COPY_VALUES,&from));

  /* x index in the IS sto */
  k = sowners[rank]/bs + mbs;
  for (i=0; i<ec; i++) stmp[i] = (k + i);
  /* b index in the IS sto */
  for (i=ec; i<2*ec; i++) stmp[i] = sgarray[i-ec];

  CHKERRQ(ISCreateBlock(PETSC_COMM_SELF,bs,2*ec,stmp,PETSC_COPY_VALUES,&to));

  CHKERRQ(VecScatterCreate(sbaij->slvec0,from,sbaij->slvec1,to,&sbaij->sMvctx));
  CHKERRQ(VecScatterViewFromOptions(sbaij->sMvctx,(PetscObject)mat,"-matmult_vecscatter_view"));

  CHKERRQ(VecGetLocalSize(sbaij->slvec1,&nt));
  CHKERRQ(VecGetArray(sbaij->slvec1,&ptr));
  CHKERRQ(VecCreateSeqWithArray(PETSC_COMM_SELF,1,bs*mbs,ptr,&sbaij->slvec1a));
  CHKERRQ(VecCreateSeqWithArray(PETSC_COMM_SELF,1,nt-bs*mbs,ptr+bs*mbs,&sbaij->slvec1b));
  CHKERRQ(VecRestoreArray(sbaij->slvec1,&ptr));

  CHKERRQ(VecGetArray(sbaij->slvec0,&ptr));
  CHKERRQ(VecCreateSeqWithArray(PETSC_COMM_SELF,1,nt-bs*mbs,ptr+bs*mbs,&sbaij->slvec0b));
  CHKERRQ(VecRestoreArray(sbaij->slvec0,&ptr));

  CHKERRQ(PetscFree(stmp));

  CHKERRQ(PetscLogObjectParent((PetscObject)mat,(PetscObject)sbaij->sMvctx));
  CHKERRQ(PetscLogObjectParent((PetscObject)mat,(PetscObject)sbaij->slvec0));
  CHKERRQ(PetscLogObjectParent((PetscObject)mat,(PetscObject)sbaij->slvec1));
  CHKERRQ(PetscLogObjectParent((PetscObject)mat,(PetscObject)sbaij->slvec0b));
  CHKERRQ(PetscLogObjectParent((PetscObject)mat,(PetscObject)sbaij->slvec1a));
  CHKERRQ(PetscLogObjectParent((PetscObject)mat,(PetscObject)sbaij->slvec1b));
  CHKERRQ(PetscLogObjectParent((PetscObject)mat,(PetscObject)from));
  CHKERRQ(PetscLogObjectParent((PetscObject)mat,(PetscObject)to));

  CHKERRQ(PetscLogObjectMemory((PetscObject)mat,ec*sizeof(PetscInt)));
  CHKERRQ(ISDestroy(&from));
  CHKERRQ(ISDestroy(&to));
  CHKERRQ(PetscFree2(sgarray,ec_owner));
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
  PetscInt       i,j,mbs=Bbaij->mbs,n = A->cmap->N,col,*garray=baij->garray;
  PetscInt       k,bs=A->rmap->bs,bs2=baij->bs2,*rvals,*nz,ec,m=A->rmap->n;
  MatScalar      *a = Bbaij->a;
  PetscScalar    *atmp;
#if defined(PETSC_USE_REAL_MAT_SINGLE)
  PetscInt l;
#endif

  PetscFunctionBegin;
#if defined(PETSC_USE_REAL_MAT_SINGLE)
  CHKERRQ(PetscMalloc1(A->rmap->bs,&atmp));
#endif
  /* free stuff related to matrix-vec multiply */
  CHKERRQ(VecGetSize(baij->lvec,&ec)); /* needed for PetscLogObjectMemory below */
  CHKERRQ(VecDestroy(&baij->lvec));
  CHKERRQ(VecScatterDestroy(&baij->Mvctx));

  CHKERRQ(VecDestroy(&baij->slvec0));
  CHKERRQ(VecDestroy(&baij->slvec0b));
  CHKERRQ(VecDestroy(&baij->slvec1));
  CHKERRQ(VecDestroy(&baij->slvec1a));
  CHKERRQ(VecDestroy(&baij->slvec1b));

  if (baij->colmap) {
#if defined(PETSC_USE_CTABLE)
    CHKERRQ(PetscTableDestroy(&baij->colmap));
#else
    CHKERRQ(PetscFree(baij->colmap));
    CHKERRQ(PetscLogObjectMemory((PetscObject)A,-Bbaij->nbs*sizeof(PetscInt)));
#endif
  }

  /* make sure that B is assembled so we can access its values */
  CHKERRQ(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));

  /* invent new B and copy stuff over */
  CHKERRQ(PetscMalloc1(mbs,&nz));
  for (i=0; i<mbs; i++) {
    nz[i] = Bbaij->i[i+1]-Bbaij->i[i];
  }
  CHKERRQ(MatCreate(PETSC_COMM_SELF,&Bnew));
  CHKERRQ(MatSetSizes(Bnew,m,n,m,n));
  CHKERRQ(MatSetType(Bnew,((PetscObject)B)->type_name));
  CHKERRQ(MatSeqBAIJSetPreallocation(Bnew,B->rmap->bs,0,nz));
  CHKERRQ(PetscFree(nz));

  if (Bbaij->nonew >= 0) { /* Inherit insertion error options (if positive). */
    ((Mat_SeqSBAIJ*)Bnew->data)->nonew = Bbaij->nonew;
  }

  /*
   Ensure that B's nonzerostate is monotonically increasing.
   Or should this follow the MatSetValues() loop to preserve B's nonzerstate across a MatDisAssemble() call?
   */
  Bnew->nonzerostate = B->nonzerostate;
  CHKERRQ(PetscMalloc1(bs,&rvals));
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
        CHKERRQ(MatSetValues_SeqSBAIJ(Bnew,bs,rvals,1,&col,atmp,B->insertmode));
        col++;
      }
    }
  }
#if defined(PETSC_USE_REAL_MAT_SINGLE)
  CHKERRQ(PetscFree(atmp));
#endif
  CHKERRQ(PetscFree(baij->garray));

  baij->garray = NULL;

  CHKERRQ(PetscFree(rvals));
  CHKERRQ(PetscLogObjectMemory((PetscObject)A,-ec*sizeof(PetscInt)));
  CHKERRQ(MatDestroy(&B));
  CHKERRQ(PetscLogObjectParent((PetscObject)A,(PetscObject)Bnew));

  baij->B = Bnew;

  A->was_assembled = PETSC_FALSE;
  PetscFunctionReturn(0);
}
