
/*
   Support for the parallel SBAIJ matrix vector multiply
*/
#include <../src/mat/impls/sbaij/mpi/mpisbaij.h>

extern PetscErrorCode MatSetValues_SeqSBAIJ(Mat,PetscInt,const PetscInt [],PetscInt,const PetscInt [],const PetscScalar [],InsertMode);


#undef __FUNCT__  
#define __FUNCT__ "MatSetUpMultiply_MPISBAIJ"
PetscErrorCode MatSetUpMultiply_MPISBAIJ(Mat mat)
{
  Mat_MPISBAIJ   *sbaij = (Mat_MPISBAIJ*)mat->data;
  Mat_SeqBAIJ    *B = (Mat_SeqBAIJ*)(sbaij->B->data);  
  PetscErrorCode ierr;
  PetscInt       Nbs = sbaij->Nbs,i,j,*indices,*aj = B->j,ec = 0,*garray,*sgarray;
  PetscInt       bs = mat->rmap->bs,*stmp,mbs=sbaij->mbs, vec_size,nt;
  IS             from,to;
  Vec            gvec;
  PetscMPIInt    rank=sbaij->rank,lsize,size=sbaij->size; 
  PetscInt       *owners=sbaij->rangebs,*sowners,*ec_owner,k; 
  PetscScalar    *ptr;

  PetscFunctionBegin;
  ierr = VecScatterDestroy(&sbaij->sMvctx);CHKERRQ(ierr);
  
  /* For the first stab we make an array as long as the number of columns */
  /* mark those columns that are in sbaij->B */
  ierr = PetscMalloc(Nbs*sizeof(PetscInt),&indices);CHKERRQ(ierr);
  ierr = PetscMemzero(indices,Nbs*sizeof(PetscInt));CHKERRQ(ierr);
  for (i=0; i<mbs; i++) {
    for (j=0; j<B->ilen[i]; j++) {
      if (!indices[aj[B->i[i] + j]]) ec++; 
      indices[aj[B->i[i] + j] ] = 1;
    }
  }

  /* form arrays of columns we need */
  ierr = PetscMalloc(ec*sizeof(PetscInt),&garray);CHKERRQ(ierr);
  ierr = PetscMalloc2(2*ec,PetscInt,&sgarray,ec,PetscInt,&ec_owner);CHKERRQ(ierr);
  
  ec = 0;
  for (j=0; j<size; j++){
    for (i=owners[j]; i<owners[j+1]; i++){
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
  B->nbs      = ec;
  sbaij->B->cmap->n = sbaij->B->cmap->N = ec*mat->rmap->bs;
  ierr = PetscLayoutSetUp((sbaij->B->cmap));CHKERRQ(ierr);
  ierr = PetscFree(indices);CHKERRQ(ierr);

  /* create local vector that is used to scatter into */
  ierr = VecCreateSeq(PETSC_COMM_SELF,ec*bs,&sbaij->lvec);CHKERRQ(ierr);

  /* create two temporary index sets for building scatter-gather */
  ierr = PetscMalloc(2*ec*sizeof(PetscInt),&stmp);CHKERRQ(ierr);
  ierr = ISCreateBlock(PETSC_COMM_SELF,bs,ec,garray,PETSC_COPY_VALUES,&from);CHKERRQ(ierr);   
  for (i=0; i<ec; i++) { stmp[i] = i; } 
  ierr = ISCreateBlock(PETSC_COMM_SELF,bs,ec,stmp,PETSC_COPY_VALUES,&to);CHKERRQ(ierr);

  /* generate the scatter context 
     -- Mvctx and lvec are not used by MatMult_MPISBAIJ(), but usefule for some applications */
  ierr = VecCreateMPIWithArray(((PetscObject)mat)->comm,1,mat->cmap->n,mat->cmap->N,PETSC_NULL,&gvec);CHKERRQ(ierr);
  ierr = VecScatterCreate(gvec,from,sbaij->lvec,to,&sbaij->Mvctx);CHKERRQ(ierr); 
  ierr = VecDestroy(&gvec);CHKERRQ(ierr);

  sbaij->garray = garray;
  ierr = PetscLogObjectParent(mat,sbaij->Mvctx);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(mat,sbaij->lvec);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(mat,from);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(mat,to);CHKERRQ(ierr);

  ierr = ISDestroy(&from);CHKERRQ(ierr);
  ierr = ISDestroy(&to);CHKERRQ(ierr);

  /* create parallel vector that is used by SBAIJ matrix to scatter from/into */
  lsize = (mbs + ec)*bs;
  ierr = VecCreateMPI(((PetscObject)mat)->comm,lsize,PETSC_DETERMINE,&sbaij->slvec0);CHKERRQ(ierr);
  ierr = VecDuplicate(sbaij->slvec0,&sbaij->slvec1);CHKERRQ(ierr);
  ierr = VecGetSize(sbaij->slvec0,&vec_size);CHKERRQ(ierr);

  sowners = sbaij->slvec0->map->range;
 
  /* x index in the IS sfrom */
  for (i=0; i<ec; i++) { 
    j = ec_owner[i];
    sgarray[i]  = garray[i] + (sowners[j]/bs - owners[j]);  
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
  ierr = MPI_Barrier(((PetscObject)mat)->comm);CHKERRQ(ierr);
  
  ierr = PetscLogObjectParent(mat,sbaij->sMvctx);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(mat,sbaij->slvec0);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(mat,sbaij->slvec1);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(mat,sbaij->slvec0b);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(mat,sbaij->slvec1a);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(mat,sbaij->slvec1b);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(mat,from);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(mat,to);CHKERRQ(ierr);
  
  ierr = PetscLogObjectMemory(mat,(ec+1)*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = ISDestroy(&from);CHKERRQ(ierr);  
  ierr = ISDestroy(&to);CHKERRQ(ierr);
  ierr = PetscFree2(sgarray,ec_owner);CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSetUpMultiply_MPISBAIJ_2comm"
PetscErrorCode MatSetUpMultiply_MPISBAIJ_2comm(Mat mat)
{
  Mat_MPISBAIJ       *baij = (Mat_MPISBAIJ*)mat->data;
  Mat_SeqBAIJ        *B = (Mat_SeqBAIJ*)(baij->B->data);  
  PetscErrorCode     ierr;
  PetscInt           i,j,*aj = B->j,ec = 0,*garray;
  PetscInt           bs = mat->rmap->bs,*stmp;
  IS                 from,to;
  Vec                gvec;
#if defined (PETSC_USE_CTABLE)
  PetscTable         gid1_lid1;
  PetscTablePosition tpos;
  PetscInt           gid,lid;
#else
  PetscInt           Nbs = baij->Nbs,*indices;
#endif  

  PetscFunctionBegin;
#if defined (PETSC_USE_CTABLE)
  /* use a table - Mark Adams */
  PetscTableCreate(B->mbs,baij->Nbs+1,&gid1_lid1); 
  for (i=0; i<B->mbs; i++) {
    for (j=0; j<B->ilen[i]; j++) {
      PetscInt data,gid1 = aj[B->i[i]+j] + 1;
      ierr = PetscTableFind(gid1_lid1,gid1,&data);CHKERRQ(ierr);
      if (!data) {
        /* one based table */ 
        ierr = PetscTableAdd(gid1_lid1,gid1,++ec,INSERT_VALUES);CHKERRQ(ierr); 
      }
    }
  } 
  /* form array of columns we need */
  ierr = PetscMalloc(ec*sizeof(PetscInt),&garray);CHKERRQ(ierr);
  ierr = PetscTableGetHeadPosition(gid1_lid1,&tpos);CHKERRQ(ierr); 
  while (tpos) {  
    ierr = PetscTableGetNext(gid1_lid1,&tpos,&gid,&lid);CHKERRQ(ierr); 
    gid--; lid--;
    garray[lid] = gid; 
  }
  ierr = PetscSortInt(ec,garray);CHKERRQ(ierr);
  ierr = PetscTableRemoveAll(gid1_lid1);CHKERRQ(ierr);
  for (i=0; i<ec; i++) {
    ierr = PetscTableAdd(gid1_lid1,garray[i]+1,i+1,INSERT_VALUES);CHKERRQ(ierr); 
  }
  /* compact out the extra columns in B */
  for (i=0; i<B->mbs; i++) {
    for (j=0; j<B->ilen[i]; j++) {
      PetscInt gid1 = aj[B->i[i] + j] + 1;
      ierr = PetscTableFind(gid1_lid1,gid1,&lid);CHKERRQ(ierr);
      lid --;
      aj[B->i[i]+j] = lid;
    }
  }
  B->nbs     = ec;
  baij->B->cmap->n = baij->B->cmap->N = ec*mat->rmap->bs;
  ierr = PetscLayoutSetUp((baij->B->cmap));CHKERRQ(ierr);
  ierr = PetscTableDestroy(&gid1_lid1);CHKERRQ(ierr);
#else
  /* For the first stab we make an array as long as the number of columns */
  /* mark those columns that are in baij->B */
  ierr = PetscMalloc(Nbs*sizeof(PetscInt),&indices);CHKERRQ(ierr);
  ierr = PetscMemzero(indices,Nbs*sizeof(PetscInt));CHKERRQ(ierr);
  for (i=0; i<B->mbs; i++) {
    for (j=0; j<B->ilen[i]; j++) {
      if (!indices[aj[B->i[i] + j]]) ec++; 
      indices[aj[B->i[i] + j] ] = 1;
    }
  }

  /* form array of columns we need */
  ierr = PetscMalloc(ec*sizeof(PetscInt),&garray);CHKERRQ(ierr);
  ec = 0;
  for (i=0; i<Nbs; i++) {
    if (indices[i]) {
      garray[ec++] = i;
    }
  }

  /* make indices now point into garray */
  for (i=0; i<ec; i++) {
    indices[garray[i]] = i;
  }

  /* compact out the extra columns in B */
  for (i=0; i<B->mbs; i++) {
    for (j=0; j<B->ilen[i]; j++) {
      aj[B->i[i] + j] = indices[aj[B->i[i] + j]];
    }
  }
  B->nbs       = ec;
  baij->B->cmap->n   = ec*mat->rmap->bs;
  ierr = PetscFree(indices);CHKERRQ(ierr);
#endif  
  
  /* create local vector that is used to scatter into */
  ierr = VecCreateSeq(PETSC_COMM_SELF,ec*bs,&baij->lvec);CHKERRQ(ierr);

  /* create two temporary index sets for building scatter-gather */
  ierr = ISCreateBlock(PETSC_COMM_SELF,bs,ec,garray,PETSC_COPY_VALUES,&from);CHKERRQ(ierr);   

  ierr = PetscMalloc(ec*sizeof(PetscInt),&stmp);CHKERRQ(ierr);
  for (i=0; i<ec; i++) { stmp[i] = i; } 
  ierr = ISCreateBlock(PETSC_COMM_SELF,bs,ec,stmp,PETSC_OWN_POINTER,&to);CHKERRQ(ierr);

  /* create temporary global vector to generate scatter context */
  ierr = VecCreateMPIWithArray(((PetscObject)mat)->comm,1,mat->cmap->n,mat->cmap->N,PETSC_NULL,&gvec);CHKERRQ(ierr);
  ierr = VecScatterCreate(gvec,from,baij->lvec,to,&baij->Mvctx);CHKERRQ(ierr);
  ierr = VecDestroy(&gvec);CHKERRQ(ierr);

  ierr = PetscLogObjectParent(mat,baij->Mvctx);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(mat,baij->lvec);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(mat,from);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(mat,to);CHKERRQ(ierr);
  baij->garray = garray;
  ierr = PetscLogObjectMemory(mat,(ec+1)*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = ISDestroy(&from);CHKERRQ(ierr);
  ierr = ISDestroy(&to);CHKERRQ(ierr);
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
#undef __FUNCT__  
#define __FUNCT__ "MatDisAssemble_MPISBAIJ"
PetscErrorCode MatDisAssemble_MPISBAIJ(Mat A)
{
  Mat_MPISBAIJ   *baij = (Mat_MPISBAIJ*)A->data;
  Mat            B = baij->B,Bnew;
  Mat_SeqBAIJ    *Bbaij = (Mat_SeqBAIJ*)B->data;
  PetscErrorCode ierr;
  PetscInt       i,j,mbs=Bbaij->mbs,n = A->cmap->N,col,*garray=baij->garray;
  PetscInt       k,bs=A->rmap->bs,bs2=baij->bs2,*rvals,*nz,ec,m=A->rmap->n;
  MatScalar      *a = Bbaij->a;
  PetscScalar    *atmp;
#if defined(PETSC_USE_REAL_MAT_SINGLE)
  PetscInt       l;
#endif

  PetscFunctionBegin;
#if defined(PETSC_USE_REAL_MAT_SINGLE)
  ierr = PetscMalloc(A->rmap->bs*sizeof(PetscScalar),&atmp);
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
#if defined (PETSC_USE_CTABLE)
    ierr = PetscTableDestroy(&baij->colmap);CHKERRQ(ierr);
#else
    ierr = PetscFree(baij->colmap);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory(A,-Bbaij->nbs*sizeof(PetscInt));CHKERRQ(ierr);
#endif
  }

  /* make sure that B is assembled so we can access its values */
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* invent new B and copy stuff over */
  ierr = PetscMalloc(mbs*sizeof(PetscInt),&nz);CHKERRQ(ierr);
  for (i=0; i<mbs; i++) {
    nz[i] = Bbaij->i[i+1]-Bbaij->i[i];
  }
  ierr = MatCreate(PETSC_COMM_SELF,&Bnew);CHKERRQ(ierr);
  ierr = MatSetSizes(Bnew,m,n,m,n);CHKERRQ(ierr);
  ierr = MatSetType(Bnew,((PetscObject)B)->type_name);CHKERRQ(ierr);
  ierr = MatSeqBAIJSetPreallocation(Bnew,B->rmap->bs,0,nz);CHKERRQ(ierr);
  ((Mat_SeqSBAIJ*)Bnew->data)->nonew = Bbaij->nonew; /* Inherit insertion error options. */
  ierr = PetscFree(nz);CHKERRQ(ierr);
  
  ierr = PetscMalloc(bs*sizeof(PetscInt),&rvals);CHKERRQ(ierr);
  for (i=0; i<mbs; i++) {
    rvals[0] = bs*i;
    for (j=1; j<bs; j++) { rvals[j] = rvals[j-1] + 1; }
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
  ierr = PetscLogObjectMemory(A,-ec*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(A,Bnew);CHKERRQ(ierr);
  baij->B = Bnew;
  A->was_assembled = PETSC_FALSE;
  PetscFunctionReturn(0);
}



