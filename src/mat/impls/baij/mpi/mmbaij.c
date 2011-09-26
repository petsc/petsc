
/*
   Support for the parallel BAIJ matrix vector multiply
*/
#include <../src/mat/impls/baij/mpi/mpibaij.h>

extern PetscErrorCode MatSetValuesBlocked_SeqBAIJ(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt[],const PetscScalar[],InsertMode);

#undef __FUNCT__  
#define __FUNCT__ "MatSetUpMultiply_MPIBAIJ"
PetscErrorCode MatSetUpMultiply_MPIBAIJ(Mat mat)
{
  Mat_MPIBAIJ    *baij = (Mat_MPIBAIJ*)mat->data;
  Mat_SeqBAIJ    *B = (Mat_SeqBAIJ*)(baij->B->data);  
  PetscErrorCode ierr;
  PetscInt       i,j,*aj = B->j,ec = 0,*garray;
  PetscInt       bs = mat->rmap->bs,*stmp;
  IS             from,to;
  Vec            gvec;
#if defined (PETSC_USE_CTABLE)
  PetscTable     gid1_lid1;
  PetscTablePosition tpos;
  PetscInt       gid,lid; 
#else
  PetscInt       Nbs = baij->Nbs,*indices;
#endif  

  PetscFunctionBegin;

#if defined (PETSC_USE_CTABLE)
  /* use a table - Mark Adams */
  ierr = PetscTableCreate(B->mbs,baij->Nbs+1,&gid1_lid1);CHKERRQ(ierr);
  for (i=0; i<B->mbs; i++) {
    for (j=0; j<B->ilen[i]; j++) {
      PetscInt data,gid1 = aj[B->i[i]+j] + 1;
      ierr = PetscTableFind(gid1_lid1,gid1,&data);CHKERRQ(ierr);
      if (!data) {
        /* one based table */ 
        ierr = PetscTableAdd(gid1_lid1,gid1,++ec);CHKERRQ(ierr); 
      }
    }
  } 
  /* form array of columns we need */
  ierr = PetscMalloc((ec+1)*sizeof(PetscInt),&garray);CHKERRQ(ierr);
  ierr = PetscTableGetHeadPosition(gid1_lid1,&tpos);CHKERRQ(ierr); 
  while (tpos) {  
    ierr = PetscTableGetNext(gid1_lid1,&tpos,&gid,&lid);CHKERRQ(ierr); 
    gid--; lid--;
    garray[lid] = gid; 
  }
  ierr = PetscSortInt(ec,garray);CHKERRQ(ierr);
  ierr = PetscTableRemoveAll(gid1_lid1);CHKERRQ(ierr);
  for (i=0; i<ec; i++) {
    ierr = PetscTableAdd(gid1_lid1,garray[i]+1,i+1);CHKERRQ(ierr); 
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
  /* Make an array as long as the number of columns */
  /* mark those columns that are in baij->B */
  ierr = PetscMalloc((Nbs+1)*sizeof(PetscInt),&indices);CHKERRQ(ierr);
  ierr = PetscMemzero(indices,Nbs*sizeof(PetscInt));CHKERRQ(ierr);
  for (i=0; i<B->mbs; i++) {
    for (j=0; j<B->ilen[i]; j++) {
      if (!indices[aj[B->i[i] + j]]) ec++; 
      indices[aj[B->i[i] + j]] = 1;
    }
  }

  /* form array of columns we need */
  ierr = PetscMalloc((ec+1)*sizeof(PetscInt),&garray);CHKERRQ(ierr);
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
  baij->B->cmap->n =baij->B->cmap->N  = ec*mat->rmap->bs;
  ierr = PetscLayoutSetUp((baij->B->cmap));CHKERRQ(ierr);
  ierr = PetscFree(indices);CHKERRQ(ierr);
#endif  

  /* create local vector that is used to scatter into */
  ierr = VecCreateSeq(PETSC_COMM_SELF,ec*bs,&baij->lvec);CHKERRQ(ierr);

  /* create two temporary index sets for building scatter-gather */
  ierr = ISCreateBlock(PETSC_COMM_SELF,bs,ec,garray,PETSC_COPY_VALUES,&from);CHKERRQ(ierr);   

  ierr = PetscMalloc((ec+1)*sizeof(PetscInt),&stmp);CHKERRQ(ierr);
  for (i=0; i<ec; i++) { stmp[i] = i; } 
  ierr = ISCreateBlock(PETSC_COMM_SELF,bs,ec,stmp,PETSC_OWN_POINTER,&to);CHKERRQ(ierr);

  /* create temporary global vector to generate scatter context */
  ierr = VecCreateMPIWithArray(((PetscObject)mat)->comm,mat->cmap->n,mat->cmap->N,PETSC_NULL,&gvec);CHKERRQ(ierr);

  ierr = VecScatterCreate(gvec,from,baij->lvec,to,&baij->Mvctx);CHKERRQ(ierr);

  ierr = PetscLogObjectParent(mat,baij->Mvctx);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(mat,baij->lvec);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(mat,from);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(mat,to);CHKERRQ(ierr);
  baij->garray = garray;
  ierr = PetscLogObjectMemory(mat,(ec+1)*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = ISDestroy(&from);CHKERRQ(ierr);
  ierr = ISDestroy(&to);CHKERRQ(ierr);
  ierr = VecDestroy(&gvec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
     Takes the local part of an already assembled MPIBAIJ matrix
   and disassembles it. This is to allow new nonzeros into the matrix
   that require more communication in the matrix vector multiply. 
   Thus certain data-structures must be rebuilt.

   Kind of slow! But that's what application programmers get when 
   they are sloppy.
*/
#undef __FUNCT__  
#define __FUNCT__ "DisAssemble_MPIBAIJ"
PetscErrorCode DisAssemble_MPIBAIJ(Mat A)
{
  Mat_MPIBAIJ    *baij = (Mat_MPIBAIJ*)A->data;
  Mat            B = baij->B,Bnew;
  Mat_SeqBAIJ    *Bbaij = (Mat_SeqBAIJ*)B->data;
  PetscErrorCode ierr;
  PetscInt       i,j,mbs=Bbaij->mbs,n = A->cmap->N,col,*garray=baij->garray;
  PetscInt       bs2 = baij->bs2,*nz,ec,m = A->rmap->n;
  MatScalar      *a = Bbaij->a;
  MatScalar      *atmp;


  PetscFunctionBegin;
  /* free stuff related to matrix-vec multiply */
  ierr = VecGetSize(baij->lvec,&ec);CHKERRQ(ierr); /* needed for PetscLogObjectMemory below */
  ierr = VecDestroy(&baij->lvec);CHKERRQ(ierr); baij->lvec = 0;
  ierr = VecScatterDestroy(&baij->Mvctx);CHKERRQ(ierr); baij->Mvctx = 0;
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
  ierr = MatCreate(((PetscObject)B)->comm,&Bnew);CHKERRQ(ierr);
  ierr = MatSetSizes(Bnew,m,n,m,n);CHKERRQ(ierr);
  ierr = MatSetType(Bnew,((PetscObject)B)->type_name);CHKERRQ(ierr);
  ierr = MatSeqBAIJSetPreallocation(Bnew,B->rmap->bs,0,nz);CHKERRQ(ierr);
  ierr = MatSetOption(Bnew,MAT_ROW_ORIENTED,PETSC_FALSE);CHKERRQ(ierr);

  for (i=0; i<mbs; i++) {
    for (j=Bbaij->i[i]; j<Bbaij->i[i+1]; j++) {
      col  = garray[Bbaij->j[j]];
      atmp = a + j*bs2;
      ierr = MatSetValuesBlocked_SeqBAIJ(Bnew,1,&i,1,&col,atmp,B->insertmode);CHKERRQ(ierr);
    }
  }
  ierr = MatSetOption(Bnew,MAT_ROW_ORIENTED,PETSC_TRUE);CHKERRQ(ierr);

  ierr = PetscFree(nz);CHKERRQ(ierr);
  ierr = PetscFree(baij->garray);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(A,-ec*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(A,Bnew);CHKERRQ(ierr);
  baij->B = Bnew;
  A->was_assembled = PETSC_FALSE;
  A->assembled     = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*      ugly stuff added for Glenn someday we should fix this up */

static PetscInt *uglyrmapd = 0,*uglyrmapo = 0;  /* mapping from the local ordering to the "diagonal" and "off-diagonal"
                                      parts of the local matrix */
static Vec uglydd = 0,uglyoo = 0;   /* work vectors used to scale the two parts of the local matrix */


#undef __FUNCT__
#define __FUNCT__ "MatMPIBAIJDiagonalScaleLocalSetUp"
PetscErrorCode MatMPIBAIJDiagonalScaleLocalSetUp(Mat inA,Vec scale)
{
  Mat_MPIBAIJ    *ina = (Mat_MPIBAIJ*) inA->data; /*access private part of matrix */
  Mat_SeqBAIJ    *B = (Mat_SeqBAIJ*)ina->B->data;
  PetscErrorCode ierr;
  PetscInt       bs = inA->rmap->bs,i,n,nt,j,cstart,cend,no,*garray = ina->garray,*lindices;
  PetscInt       *r_rmapd,*r_rmapo;
  
  PetscFunctionBegin;
  ierr = MatGetOwnershipRange(inA,&cstart,&cend);CHKERRQ(ierr);
  ierr = MatGetSize(ina->A,PETSC_NULL,&n);CHKERRQ(ierr);
  ierr = PetscMalloc((inA->rmap->bmapping->n+1)*sizeof(PetscInt),&r_rmapd);CHKERRQ(ierr);
  ierr = PetscMemzero(r_rmapd,inA->rmap->bmapping->n*sizeof(PetscInt));CHKERRQ(ierr);
  nt   = 0;
  for (i=0; i<inA->rmap->bmapping->n; i++) {
    if (inA->rmap->bmapping->indices[i]*bs >= cstart && inA->rmap->bmapping->indices[i]*bs < cend) {
      nt++;
      r_rmapd[i] = inA->rmap->bmapping->indices[i] + 1;
    }
  }
  if (nt*bs != n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Hmm nt*bs %D n %D",nt*bs,n);
  ierr = PetscMalloc((n+1)*sizeof(PetscInt),&uglyrmapd);CHKERRQ(ierr);
  for (i=0; i<inA->rmap->bmapping->n; i++) {
    if (r_rmapd[i]){
      for (j=0; j<bs; j++) {
        uglyrmapd[(r_rmapd[i]-1)*bs+j-cstart] = i*bs + j;
      }
    }
  }
  ierr = PetscFree(r_rmapd);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,n,&uglydd);CHKERRQ(ierr);

  ierr = PetscMalloc((ina->Nbs+1)*sizeof(PetscInt),&lindices);CHKERRQ(ierr);
  ierr = PetscMemzero(lindices,ina->Nbs*sizeof(PetscInt));CHKERRQ(ierr);
  for (i=0; i<B->nbs; i++) {
    lindices[garray[i]] = i+1;
  }
  no   = inA->rmap->bmapping->n - nt;
  ierr = PetscMalloc((inA->rmap->bmapping->n+1)*sizeof(PetscInt),&r_rmapo);CHKERRQ(ierr);
  ierr = PetscMemzero(r_rmapo,inA->rmap->bmapping->n*sizeof(PetscInt));CHKERRQ(ierr);
  nt   = 0;
  for (i=0; i<inA->rmap->bmapping->n; i++) {
    if (lindices[inA->rmap->bmapping->indices[i]]) {
      nt++;
      r_rmapo[i] = lindices[inA->rmap->bmapping->indices[i]];
    }
  }
  if (nt > no) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Hmm nt %D no %D",nt,n);
  ierr = PetscFree(lindices);CHKERRQ(ierr);
  ierr = PetscMalloc((nt*bs+1)*sizeof(PetscInt),&uglyrmapo);CHKERRQ(ierr);
  for (i=0; i<inA->rmap->bmapping->n; i++) {
    if (r_rmapo[i]){
      for (j=0; j<bs; j++) {
        uglyrmapo[(r_rmapo[i]-1)*bs+j] = i*bs + j;
      }
    }
  }
  ierr = PetscFree(r_rmapo);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,nt*bs,&uglyoo);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMPIBAIJDiagonalScaleLocal"
PetscErrorCode  MatMPIBAIJDiagonalScaleLocal(Mat A,Vec scale)
{
  /* This routine should really be abandoned as it duplicates MatDiagonalScaleLocal */
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscTryMethod(A,"MatDiagonalScaleLocal_C",(Mat,Vec),(A,scale));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatDiagonalScaleLocal_MPIBAIJ"
PetscErrorCode  MatDiagonalScaleLocal_MPIBAIJ(Mat A,Vec scale)
{
  Mat_MPIBAIJ    *a = (Mat_MPIBAIJ*) A->data; /*access private part of matrix */
  PetscErrorCode ierr;
  PetscInt       n,i;
  PetscScalar    *d,*o,*s;
  
  PetscFunctionBegin;
  if (!uglyrmapd) {
    ierr = MatMPIBAIJDiagonalScaleLocalSetUp(A,scale);CHKERRQ(ierr);
  }

  ierr = VecGetArray(scale,&s);CHKERRQ(ierr);
  
  ierr = VecGetLocalSize(uglydd,&n);CHKERRQ(ierr);
  ierr = VecGetArray(uglydd,&d);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    d[i] = s[uglyrmapd[i]]; /* copy "diagonal" (true local) portion of scale into dd vector */
  }
  ierr = VecRestoreArray(uglydd,&d);CHKERRQ(ierr);
  /* column scale "diagonal" portion of local matrix */
  ierr = MatDiagonalScale(a->A,PETSC_NULL,uglydd);CHKERRQ(ierr);

  ierr = VecGetLocalSize(uglyoo,&n);CHKERRQ(ierr);
  ierr = VecGetArray(uglyoo,&o);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    o[i] = s[uglyrmapo[i]]; /* copy "off-diagonal" portion of scale into oo vector */
  }
  ierr = VecRestoreArray(scale,&s);CHKERRQ(ierr);
  ierr = VecRestoreArray(uglyoo,&o);CHKERRQ(ierr);
  /* column scale "off-diagonal" portion of local matrix */
  ierr = MatDiagonalScale(a->B,PETSC_NULL,uglyoo);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
EXTERN_C_END


