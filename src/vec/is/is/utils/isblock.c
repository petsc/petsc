/* Routines to be used by MatIncreaseOverlap() for BAIJ and SBAIJ matrices */
#include <petscis.h>                       /*I "petscis.h"  I*/
#include <petscbt.h>
#include <petscctable.h>

/*@
   ISCompressIndicesGeneral - convert the indices into block indices

   Input Parameters:
+    n - maximum possible length of the index set
.    nkeys - expected number of keys when PETSC_USE_CTABLE
.    bs - the size of block
.    imax - the number of index sets
-    is_in - the non-blocked array of index sets

   Output Parameter:
.    is_out - the blocked new index set

   Level: intermediate

.seealso: ISExpandIndicesGeneral()
@*/
PetscErrorCode  ISCompressIndicesGeneral(PetscInt n,PetscInt nkeys,PetscInt bs,PetscInt imax,const IS is_in[],IS is_out[])
{
  PetscInt           isz,len,i,j,ival,Nbs;
  const PetscInt     *idx;
#if defined(PETSC_USE_CTABLE)
  PetscTable         gid1_lid1;
  PetscInt           tt, gid1, *nidx,Nkbs;
  PetscTablePosition tpos;
#else
  PetscInt           *nidx;
  PetscBT            table;
#endif

  PetscFunctionBegin;
  Nbs = n/bs;
#if defined(PETSC_USE_CTABLE)
  Nkbs = nkeys/bs;
  CHKERRQ(PetscTableCreate(Nkbs,Nbs,&gid1_lid1));
#else
  CHKERRQ(PetscMalloc1(Nbs,&nidx));
  CHKERRQ(PetscBTCreate(Nbs,&table));
#endif
  for (i=0; i<imax; i++) {
    isz = 0;
#if defined(PETSC_USE_CTABLE)
    CHKERRQ(PetscTableRemoveAll(gid1_lid1));
#else
    CHKERRQ(PetscBTMemzero(Nbs,table));
#endif
    CHKERRQ(ISGetIndices(is_in[i],&idx));
    CHKERRQ(ISGetLocalSize(is_in[i],&len));
    for (j=0; j<len; j++) {
      ival = idx[j]/bs; /* convert the indices into block indices */
#if defined(PETSC_USE_CTABLE)
      CHKERRQ(PetscTableFind(gid1_lid1,ival+1,&tt));
      if (!tt) {
        CHKERRQ(PetscTableAdd(gid1_lid1,ival+1,isz+1,INSERT_VALUES));
        isz++;
      }
#else
      PetscCheckFalse(ival>Nbs,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"index greater than mat-dim");
      if (!PetscBTLookupSet(table,ival)) nidx[isz++] = ival;
#endif
    }
    CHKERRQ(ISRestoreIndices(is_in[i],&idx));

#if defined(PETSC_USE_CTABLE)
    CHKERRQ(PetscMalloc1(isz,&nidx));
    CHKERRQ(PetscTableGetHeadPosition(gid1_lid1,&tpos));
    j    = 0;
    while (tpos) {
      CHKERRQ(PetscTableGetNext(gid1_lid1,&tpos,&gid1,&tt));
      PetscCheckFalse(tt-- > isz,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"index greater than array-dim");
      nidx[tt] = gid1 - 1;
      j++;
    }
    PetscCheckFalse(j != isz,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"table error: jj != isz");
    CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)is_in[i]),isz,nidx,PETSC_OWN_POINTER,(is_out+i)));
#else
    CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)is_in[i]),isz,nidx,PETSC_COPY_VALUES,(is_out+i)));
#endif
  }
#if defined(PETSC_USE_CTABLE)
  CHKERRQ(PetscTableDestroy(&gid1_lid1));
#else
  CHKERRQ(PetscBTDestroy(&table));
  CHKERRQ(PetscFree(nidx));
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode  ISCompressIndicesSorted(PetscInt n,PetscInt bs,PetscInt imax,const IS is_in[],IS is_out[])
{
  PetscInt       i,j,k,val,len,*nidx,bbs;
  const PetscInt *idx,*idx_local;
  PetscBool      flg,isblock;
#if defined(PETSC_USE_CTABLE)
  PetscInt       maxsz;
#else
  PetscInt       Nbs=n/bs;
#endif

  PetscFunctionBegin;
  for (i=0; i<imax; i++) {
    CHKERRQ(ISSorted(is_in[i],&flg));
    PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Indices are not sorted");
  }

#if defined(PETSC_USE_CTABLE)
  /* Now check max size */
  for (i=0,maxsz=0; i<imax; i++) {
    CHKERRQ(ISGetLocalSize(is_in[i],&len));
    PetscCheckFalse(len%bs !=0,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Indices are not block ordered");
    len = len/bs; /* The reduced index size */
    if (len > maxsz) maxsz = len;
  }
  CHKERRQ(PetscMalloc1(maxsz,&nidx));
#else
  CHKERRQ(PetscMalloc1(Nbs,&nidx));
#endif
  /* Now check if the indices are in block order */
  for (i=0; i<imax; i++) {
    CHKERRQ(ISGetLocalSize(is_in[i],&len));

    /* special case where IS is already block IS of the correct size */
    CHKERRQ(PetscObjectTypeCompare((PetscObject)is_in[i],ISBLOCK,&isblock));
    if (isblock) {
      CHKERRQ(ISBlockGetLocalSize(is_in[i],&bbs));
      if (bs == bbs) {
        len  = len/bs;
        CHKERRQ(ISBlockGetIndices(is_in[i],&idx));
        CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,len,idx,PETSC_COPY_VALUES,(is_out+i)));
        CHKERRQ(ISBlockRestoreIndices(is_in[i],&idx));
        continue;
      }
    }
    CHKERRQ(ISGetIndices(is_in[i],&idx));
    PetscCheckFalse(len%bs !=0,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Indices are not block ordered");

    len       = len/bs; /* The reduced index size */
    idx_local = idx;
    for (j=0; j<len; j++) {
      val = idx_local[0];
      PetscCheckFalse(val%bs != 0,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Indices are not block ordered");
      for (k=0; k<bs; k++) {
        PetscCheckFalse(val+k != idx_local[k],PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Indices are not block ordered");
      }
      nidx[j]    = val/bs;
      idx_local += bs;
    }
    CHKERRQ(ISRestoreIndices(is_in[i],&idx));
    CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)is_in[i]),len,nidx,PETSC_COPY_VALUES,(is_out+i)));
  }
  CHKERRQ(PetscFree(nidx));
  PetscFunctionReturn(0);
}

/*@C
   ISExpandIndicesGeneral - convert the indices into non-block indices

   Input Parameters:
+    n - the length of the index set (not being used)
.    nkeys - expected number of keys when PETSC_USE_CTABLE (not being used)
.    bs - the size of block
.    imax - the number of index sets
-    is_in - the blocked array of index sets

   Output Parameter:
.    is_out - the non-blocked new index set

   Level: intermediate

.seealso: ISCompressIndicesGeneral()
@*/
PetscErrorCode  ISExpandIndicesGeneral(PetscInt n,PetscInt nkeys,PetscInt bs,PetscInt imax,const IS is_in[],IS is_out[])
{
  PetscInt       len,i,j,k,*nidx;
  const PetscInt *idx;
  PetscInt       maxsz;

  PetscFunctionBegin;
  /* Check max size of is_in[] */
  maxsz = 0;
  for (i=0; i<imax; i++) {
    CHKERRQ(ISGetLocalSize(is_in[i],&len));
    if (len > maxsz) maxsz = len;
  }
  CHKERRQ(PetscMalloc1(maxsz*bs,&nidx));

  for (i=0; i<imax; i++) {
    CHKERRQ(ISGetLocalSize(is_in[i],&len));
    CHKERRQ(ISGetIndices(is_in[i],&idx));
    for (j=0; j<len ; ++j) {
      for (k=0; k<bs; k++) nidx[j*bs+k] = idx[j]*bs+k;
    }
    CHKERRQ(ISRestoreIndices(is_in[i],&idx));
    CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)is_in[i]),len*bs,nidx,PETSC_COPY_VALUES,is_out+i));
  }
  CHKERRQ(PetscFree(nidx));
  PetscFunctionReturn(0);
}
