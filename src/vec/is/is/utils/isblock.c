/* Routines to be used by MatIncreaseOverlap() for BAIJ and SBAIJ matrices */
#include <petscis.h> /*I "petscis.h"  I*/
#include <petscbt.h>
#include <petsc/private/hashmapi.h>

/*@
  ISCompressIndicesGeneral - convert the indices of an array of `IS` into an array of `ISGENERAL` of block indices

  Input Parameters:
+ n     - maximum possible length of the index set
. nkeys - expected number of keys when using `PETSC_USE_CTABLE`
. bs    - the size of block
. imax  - the number of index sets
- is_in - the non-blocked array of index sets

  Output Parameter:
. is_out - the blocked new index set, as `ISGENERAL`, not as `ISBLOCK`

  Level: intermediate

  Note:
  When `is_in[i]` already has the requested block size and needs no compression,
  `is_out[i]` may be `is_in[i]` itself with an incremented reference count.

.seealso: [](sec_scatter), `IS`, `ISGENERAL`, `ISExpandIndicesGeneral()`
@*/
PetscErrorCode ISCompressIndicesGeneral(PetscInt n, PetscInt nkeys, PetscInt bs, PetscInt imax, const IS is_in[], IS is_out[])
{
  PetscInt        isz, len, i, j, ival, bbs;
  const PetscInt *idx;
  PetscBool       flg;
#if PetscDefined(USE_CTABLE)
  PetscHMapI    gid1_lid1 = NULL;
  PetscInt      tt, gid1, *nidx;
  PetscHashIter tpos;
#else
  PetscInt *nidx;
  PetscInt  Nbs;
  PetscBT   table;
#endif

  PetscFunctionBegin;
#if PetscDefined(USE_CTABLE)
  PetscCall(PetscHMapICreateWithSize(nkeys / bs, &gid1_lid1));
#else
  Nbs = n / bs;
  PetscCall(PetscMalloc1(Nbs, &nidx));
  PetscCall(PetscBTCreate(Nbs, &table));
#endif
  for (i = 0; i < imax; i++) {
    PetscCall(ISGetLocalSize(is_in[i], &len));
    PetscCall(ISGetBlockSize(is_in[i], &bbs));
    /* special cases where IS already has the correct block size */
    if (bs == bbs) {
      PetscCall(PetscObjectTypeCompare((PetscObject)is_in[i], ISBLOCK, &flg));
      if (flg) {
        len = len / bs;
        PetscCall(ISBlockGetIndices(is_in[i], &idx));
        PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)is_in[i]), len, idx, PETSC_COPY_VALUES, is_out + i));
        PetscCall(ISBlockRestoreIndices(is_in[i], &idx));
        continue;
      } else {
        PetscCall(PetscObjectTypeCompare((PetscObject)is_in[i], ISSTRIDE, &flg));
        if (flg) {
          PetscInt  first, nblocks;
          PetscInt *idx;

          PetscCall(ISStrideGetInfo(is_in[i], &first, &j));
          if (j == 1) {
            nblocks = len ? (first + len - 1) / bs - first / bs + 1 : 0;
            PetscCall(PetscMalloc1(nblocks, &idx));
            for (j = 0; j < nblocks; ++j) idx[j] = first / bs + j;
            PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)is_in[i]), nblocks, idx, PETSC_OWN_POINTER, is_out + i));
            continue;
          }
        } else if (bs == 1) { /* ISGENERAL */
          is_out[i] = is_in[i];
          PetscCall(PetscObjectReference((PetscObject)is_in[i]));
          continue;
        }
      }
    }
    isz = 0;
#if PetscDefined(USE_CTABLE)
    PetscCall(PetscHMapIClear(gid1_lid1));
#else
    PetscCall(PetscBTMemzero(Nbs, table));
#endif
    PetscCall(ISGetIndices(is_in[i], &idx));
    for (j = 0; j < len; j++) {
      ival = idx[j] / bs; /* convert the indices into block indices */
#if PetscDefined(USE_CTABLE)
      PetscCall(PetscHMapIGetWithDefault(gid1_lid1, ival + 1, 0, &tt));
      if (!tt) {
        PetscCall(PetscHMapISet(gid1_lid1, ival + 1, isz + 1));
        isz++;
      }
#else
      PetscCheck(ival <= Nbs, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "index greater than mat-dim");
      if (!PetscBTLookupSet(table, ival)) nidx[isz++] = ival;
#endif
    }
    PetscCall(ISRestoreIndices(is_in[i], &idx));

#if PetscDefined(USE_CTABLE)
    PetscCall(PetscMalloc1(isz, &nidx));
    PetscHashIterBegin(gid1_lid1, tpos);
    j = 0;
    while (!PetscHashIterAtEnd(gid1_lid1, tpos)) {
      PetscHashIterGetKey(gid1_lid1, tpos, gid1);
      PetscHashIterGetVal(gid1_lid1, tpos, tt);
      PetscCheck(tt-- <= isz, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "index greater than array-dim");
      nidx[tt] = gid1 - 1;
      j++;
      PetscHashIterNext(gid1_lid1, tpos);
    }
    PetscCheck(j == isz, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "table error: jj != isz");
    PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)is_in[i]), isz, nidx, PETSC_OWN_POINTER, is_out + i));
#else
    PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)is_in[i]), isz, nidx, PETSC_COPY_VALUES, is_out + i));
#endif
  }
#if PetscDefined(USE_CTABLE)
  PetscCall(PetscHMapIDestroy(&gid1_lid1));
#else
  PetscCall(PetscBTDestroy(&table));
  PetscCall(PetscFree(nidx));
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  ISExpandIndicesGeneral - convert the indices of an array `IS` into non-block indices in an array of `ISGENERAL`

  Input Parameters:
+ n     - the length of the index set (not being used)
. nkeys - expected number of keys when `PETSC_USE_CTABLE` is used
. bs    - the size of block
. imax  - the number of index sets
- is_in - the blocked array of index sets, must be as large as `imax`

  Output Parameter:
. is_out - the non-blocked new index set, as `ISGENERAL`, must be as large as `imax`

  Level: intermediate

.seealso: [](sec_scatter), `IS`, `ISGENERAL`, `ISCompressIndicesGeneral()`
@*/
PetscErrorCode ISExpandIndicesGeneral(PetscInt n, PetscInt nkeys, PetscInt bs, PetscInt imax, const IS is_in[], IS is_out[])
{
  PetscInt        len, i, j, k, *nidx;
  const PetscInt *idx;
  PetscInt        maxsz;

  PetscFunctionBegin;
  /* Check max size of is_in[] */
  maxsz = 0;
  for (i = 0; i < imax; i++) {
    PetscCall(ISGetLocalSize(is_in[i], &len));
    if (len > maxsz) maxsz = len;
  }
  PetscCall(PetscMalloc1(maxsz * bs, &nidx));

  for (i = 0; i < imax; i++) {
    PetscCall(ISGetLocalSize(is_in[i], &len));
    PetscCall(ISGetIndices(is_in[i], &idx));
    for (j = 0; j < len; ++j) {
      for (k = 0; k < bs; k++) nidx[j * bs + k] = idx[j] * bs + k;
    }
    PetscCall(ISRestoreIndices(is_in[i], &idx));
    PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)is_in[i]), len * bs, nidx, PETSC_COPY_VALUES, is_out + i));
  }
  PetscCall(PetscFree(nidx));
  PetscFunctionReturn(PETSC_SUCCESS);
}
