/*$Id: mmbaij.c,v 1.46 2001/09/25 00:31:36 balay Exp $*/

/*
   Support for the parallel BAIJ matrix vector multiply
*/
#include "src/mat/impls/baij/mpi/mpibaij.h"
#include "src/vec/vecimpl.h"

EXTERN int MatSetValuesBlocked_SeqBAIJ(Mat,int,int*,int,int*,PetscScalar*,InsertMode);

#undef __FUNCT__  
#define __FUNCT__ "MatSetUpMultiply_MPIBAIJ"
int MatSetUpMultiply_MPIBAIJ(Mat mat)
{
  Mat_MPIBAIJ        *baij = (Mat_MPIBAIJ*)mat->data;
  Mat_SeqBAIJ        *B = (Mat_SeqBAIJ*)(baij->B->data);  
  int                Nbs = baij->Nbs,i,j,*indices,*aj = B->j,ierr,ec = 0,*garray;
  int                bs = baij->bs,*stmp;
  IS                 from,to;
  Vec                gvec;
#if defined (PETSC_USE_CTABLE)
  PetscTable         gid1_lid1;
  PetscTablePosition tpos;
  int                gid,lid; 
#endif  

  PetscFunctionBegin;

#if defined (PETSC_USE_CTABLE)
  /* use a table - Mark Adams */
  ierr = PetscTableCreate(B->mbs,&gid1_lid1);CHKERRQ(ierr);
  for (i=0; i<B->mbs; i++) {
    for (j=0; j<B->ilen[i]; j++) {
      int data,gid1 = aj[B->i[i]+j] + 1;
      ierr = PetscTableFind(gid1_lid1,gid1,&data) ;CHKERRQ(ierr);
      if (!data) {
        /* one based table */ 
        ierr = PetscTableAdd(gid1_lid1,gid1,++ec);CHKERRQ(ierr); 
      }
    }
  } 
  /* form array of columns we need */
  ierr = PetscMalloc((ec+1)*sizeof(int),&garray);CHKERRQ(ierr);
  ierr = PetscTableGetHeadPosition(gid1_lid1,&tpos);CHKERRQ(ierr); 
  while (tpos) {  
    ierr = PetscTableGetNext(gid1_lid1,&tpos,&gid,&lid);CHKERRQ(ierr); 
    gid--; lid--;
    garray[lid] = gid; 
  }
  ierr = PetscSortInt(ec,garray);CHKERRQ(ierr);
  /* qsort(garray, ec, sizeof(int), intcomparcarc); */
  ierr = PetscTableRemoveAll(gid1_lid1);CHKERRQ(ierr);
  for (i=0; i<ec; i++) {
    ierr = PetscTableAdd(gid1_lid1,garray[i]+1,i+1);CHKERRQ(ierr); 
  }
  /* compact out the extra columns in B */
  for (i=0; i<B->mbs; i++) {
    for (j=0; j<B->ilen[i]; j++) {
      int gid1 = aj[B->i[i] + j] + 1;
      ierr = PetscTableFind(gid1_lid1,gid1,&lid);CHKERRQ(ierr);
      lid --;
      aj[B->i[i]+j] = lid;
    }
  }
  B->nbs     = ec;
  baij->B->n = ec*B->bs;
  ierr = PetscTableDelete(gid1_lid1);CHKERRQ(ierr);
  /* Mark Adams */
#else
  /* Make an array as long as the number of columns */
  /* mark those columns that are in baij->B */
  ierr = PetscMalloc((Nbs+1)*sizeof(int),&indices);CHKERRQ(ierr);
  ierr = PetscMemzero(indices,Nbs*sizeof(int));CHKERRQ(ierr);
  for (i=0; i<B->mbs; i++) {
    for (j=0; j<B->ilen[i]; j++) {
      if (!indices[aj[B->i[i] + j]]) ec++; 
      indices[aj[B->i[i] + j]] = 1;
    }
  }

  /* form array of columns we need */
  ierr = PetscMalloc((ec+1)*sizeof(int),&garray);CHKERRQ(ierr);
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
  baij->B->n   = ec*B->bs;
  ierr = PetscFree(indices);CHKERRQ(ierr);
#endif  

  /* create local vector that is used to scatter into */
  ierr = VecCreateSeq(PETSC_COMM_SELF,ec*bs,&baij->lvec);CHKERRQ(ierr);

  /* create two temporary index sets for building scatter-gather */
  for (i=0; i<ec; i++) {
    garray[i] = bs*garray[i];
  }
  ierr = ISCreateBlock(PETSC_COMM_SELF,bs,ec,garray,&from);CHKERRQ(ierr);   
  for (i=0; i<ec; i++) {
    garray[i] = garray[i]/bs;
  }

  ierr = PetscMalloc((ec+1)*sizeof(int),&stmp);CHKERRQ(ierr);
  for (i=0; i<ec; i++) { stmp[i] = bs*i; } 
  ierr = ISCreateBlock(PETSC_COMM_SELF,bs,ec,stmp,&to);CHKERRQ(ierr);
  ierr = PetscFree(stmp);CHKERRQ(ierr);

  /* create temporary global vector to generate scatter context */
  /* this is inefficient, but otherwise we must do either 
     1) save garray until the first actual scatter when the vector is known or
     2) have another way of generating a scatter context without a vector.*/
  ierr = VecCreateMPI(mat->comm,mat->n,mat->N,&gvec);CHKERRQ(ierr);

  /* gnerate the scatter context */
  ierr = VecScatterCreate(gvec,from,baij->lvec,to,&baij->Mvctx);CHKERRQ(ierr);

  /*
      Post the receives for the first matrix vector product. We sync-chronize after
    this on the chance that the user immediately calls MatMult() after assemblying 
    the matrix.
  */
  ierr = VecScatterPostRecvs(gvec,baij->lvec,INSERT_VALUES,SCATTER_FORWARD,baij->Mvctx);CHKERRQ(ierr);
  ierr = MPI_Barrier(mat->comm);CHKERRQ(ierr);

  PetscLogObjectParent(mat,baij->Mvctx);
  PetscLogObjectParent(mat,baij->lvec);
  PetscLogObjectParent(mat,from);
  PetscLogObjectParent(mat,to);
  baij->garray = garray;
  PetscLogObjectMemory(mat,(ec+1)*sizeof(int));
  ierr = ISDestroy(from);CHKERRQ(ierr);
  ierr = ISDestroy(to);CHKERRQ(ierr);
  ierr = VecDestroy(gvec);CHKERRQ(ierr);
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
int DisAssemble_MPIBAIJ(Mat A)
{
  Mat_MPIBAIJ  *baij = (Mat_MPIBAIJ*)A->data;
  Mat          B = baij->B,Bnew;
  Mat_SeqBAIJ  *Bbaij = (Mat_SeqBAIJ*)B->data;
  int          ierr,i,j,mbs=Bbaij->mbs,n = A->N,col,*garray=baij->garray;
  int          bs2 = baij->bs2,*nz,ec,m = A->m;
  MatScalar    *a = Bbaij->a;
  PetscScalar  *atmp;
#if defined(PETSC_USE_MAT_SINGLE)
  int          k;
#endif

  PetscFunctionBegin;
  /* free stuff related to matrix-vec multiply */
  ierr = VecGetSize(baij->lvec,&ec);CHKERRQ(ierr); /* needed for PetscLogObjectMemory below */
  ierr = VecDestroy(baij->lvec);CHKERRQ(ierr); baij->lvec = 0;
  ierr = VecScatterDestroy(baij->Mvctx);CHKERRQ(ierr); baij->Mvctx = 0;
  if (baij->colmap) {
#if defined (PETSC_USE_CTABLE)
    ierr = PetscTableDelete(baij->colmap); baij->colmap = 0;CHKERRQ(ierr);
#else
    ierr = PetscFree(baij->colmap);CHKERRQ(ierr);
    baij->colmap = 0;
    PetscLogObjectMemory(A,-Bbaij->nbs*sizeof(int));
#endif
  }

  /* make sure that B is assembled so we can access its values */
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* invent new B and copy stuff over */
  ierr = PetscMalloc(mbs*sizeof(int),&nz);CHKERRQ(ierr);
  for (i=0; i<mbs; i++) {
    nz[i] = Bbaij->i[i+1]-Bbaij->i[i];
  }
  ierr = MatCreateSeqBAIJ(PETSC_COMM_SELF,baij->bs,m,n,0,nz,&Bnew);CHKERRQ(ierr);
  ierr = MatSetOption(Bnew,MAT_COLUMN_ORIENTED);CHKERRQ(ierr);

#if defined(PETSC_USE_MAT_SINGLE)
  ierr = PetscMalloc(bs2*sizeof(PetscScalar),&atmp);CHKERRQ(ierr);
#endif
    for (i=0; i<mbs; i++) {
      for (j=Bbaij->i[i]; j<Bbaij->i[i+1]; j++) {
        col  = garray[Bbaij->j[j]];
#if defined(PETSC_USE_MAT_SINGLE)
        for (k=0; k<bs2; k++) atmp[k] = a[j*bs2+k];
#else
        atmp = a + j*bs2;
#endif
        ierr = MatSetValuesBlocked_SeqBAIJ(Bnew,1,&i,1,&col,atmp,B->insertmode);CHKERRQ(ierr);
      }
    }
  ierr = MatSetOption(Bnew,MAT_ROW_ORIENTED);CHKERRQ(ierr);

#if defined(PETSC_USE_MAT_SINGLE)
  ierr = PetscFree(atmp);CHKERRQ(ierr);
#endif

  ierr = PetscFree(nz);CHKERRQ(ierr);
  ierr = PetscFree(baij->garray);CHKERRQ(ierr);
  baij->garray = 0;
  PetscLogObjectMemory(A,-ec*sizeof(int));
  ierr = MatDestroy(B);CHKERRQ(ierr);
  PetscLogObjectParent(A,Bnew);
  baij->B = Bnew;
  A->was_assembled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*      ugly stuff added for Glenn someday we should fix this up */

static int *uglyrmapd = 0,*uglyrmapo = 0;  /* mapping from the local ordering to the "diagonal" and "off-diagonal"
                                      parts of the local matrix */
static Vec uglydd = 0,uglyoo = 0;   /* work vectors used to scale the two parts of the local matrix */


#undef __FUNCT__
#define __FUNCT__ "MatMPIBAIJDiagonalScaleLocalSetUp"
int MatMPIBAIJDiagonalScaleLocalSetUp(Mat inA,Vec scale)
{
  Mat_MPIBAIJ  *ina = (Mat_MPIBAIJ*) inA->data; /*access private part of matrix */
  Mat_SeqBAIJ  *A = (Mat_SeqBAIJ*)ina->A->data;
  Mat_SeqBAIJ  *B = (Mat_SeqBAIJ*)ina->B->data;
  int          ierr,bs = A->bs,i,n,nt,j,cstart,cend,no,*garray = ina->garray,*lindices;
  int          *r_rmapd,*r_rmapo;
  
  PetscFunctionBegin;
  ierr = MatGetOwnershipRange(inA,&cstart,&cend);CHKERRQ(ierr);
  ierr = MatGetSize(ina->A,PETSC_NULL,&n);CHKERRQ(ierr);
  ierr = PetscMalloc((inA->bmapping->n+1)*sizeof(int),&r_rmapd);CHKERRQ(ierr);
  ierr = PetscMemzero(r_rmapd,inA->bmapping->n*sizeof(int));CHKERRQ(ierr);
  nt   = 0;
  for (i=0; i<inA->bmapping->n; i++) {
    if (inA->bmapping->indices[i]*bs >= cstart && inA->bmapping->indices[i]*bs < cend) {
      nt++;
      r_rmapd[i] = inA->bmapping->indices[i] + 1;
    }
  }
  if (nt*bs != n) SETERRQ2(1,"Hmm nt*bs %d n %d",nt*bs,n);
  ierr = PetscMalloc((n+1)*sizeof(int),&uglyrmapd);CHKERRQ(ierr);
  for (i=0; i<inA->bmapping->n; i++) {
    if (r_rmapd[i]){
      for (j=0; j<bs; j++) {
        uglyrmapd[(r_rmapd[i]-1)*bs+j-cstart] = i*bs + j;
      }
    }
  }
  ierr = PetscFree(r_rmapd);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,n,&uglydd);CHKERRQ(ierr);

  ierr = PetscMalloc((ina->Nbs+1)*sizeof(int),&lindices);CHKERRQ(ierr);
  ierr = PetscMemzero(lindices,ina->Nbs*sizeof(int));CHKERRQ(ierr);
  for (i=0; i<B->nbs; i++) {
    lindices[garray[i]] = i+1;
  }
  no   = inA->bmapping->n - nt;
  ierr = PetscMalloc((inA->bmapping->n+1)*sizeof(int),&r_rmapo);CHKERRQ(ierr);
  ierr = PetscMemzero(r_rmapo,inA->bmapping->n*sizeof(int));CHKERRQ(ierr);
  nt   = 0;
  for (i=0; i<inA->bmapping->n; i++) {
    if (lindices[inA->bmapping->indices[i]]) {
      nt++;
      r_rmapo[i] = lindices[inA->bmapping->indices[i]];
    }
  }
  if (nt > no) SETERRQ2(1,"Hmm nt %d no %d",nt,n);
  ierr = PetscFree(lindices);CHKERRQ(ierr);
  ierr = PetscMalloc((nt*bs+1)*sizeof(int),&uglyrmapo);CHKERRQ(ierr);
  for (i=0; i<inA->bmapping->n; i++) {
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
int MatMPIBAIJDiagonalScaleLocal(Mat A,Vec scale)
{
  /* This routine should really be abandoned as it duplicates MatDiagonalScaleLocal */
  int ierr,(*f)(Mat,Vec);

  PetscFunctionBegin;
  ierr = PetscObjectQueryFunction((PetscObject)A,"MatDiagonalScaleLocal_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(A,scale);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatDiagonalScaleLocal_MPIBAIJ"
int MatDiagonalScaleLocal_MPIBAIJ(Mat A,Vec scale)
{
  Mat_MPIBAIJ  *a = (Mat_MPIBAIJ*) A->data; /*access private part of matrix */
  int          ierr,n,i;
  PetscScalar  *d,*o,*s;
  
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


