
/* Contributed by - Mark Adams */

#include <petsc/private/petscimpl.h>
#include <petscctable.h>

static PetscErrorCode PetscTableCreateHashSize(PetscInt sz, PetscInt *hsz)
{
  PetscFunctionBegin;
  if (sz < 100)          *hsz = 139;
  else if (sz < 200)     *hsz = 283;
  else if (sz < 400)     *hsz = 577;
  else if (sz < 800)     *hsz = 1103;
  else if (sz < 1600)    *hsz = 2239;
  else if (sz < 3200)    *hsz = 4787;
  else if (sz < 6400)    *hsz = 9337;
  else if (sz < 12800)   *hsz = 17863;
  else if (sz < 25600)   *hsz = 37649;
  else if (sz < 51200)   *hsz = 72307;
  else if (sz < 102400)  *hsz = 142979;
  else if (sz < 204800)  *hsz = 299983;
  else if (sz < 409600)  *hsz = 599869;
  else if (sz < 819200)  *hsz = 1193557;
  else if (sz < 1638400) *hsz = 2297059;
  else if (sz < 3276800) *hsz = 4902383;
  else if (sz < 6553600) *hsz = 9179113;
  else if (sz < 13107200)*hsz = 18350009;
  else if (sz < 26214400)*hsz = 36700021;
  else if (sz < 52428800)*hsz = 73400279;
  else if (sz < 104857600)*hsz = 146800471;
  else if (sz < 209715200)*hsz = 293601569;
  else if (sz < 419430400)*hsz = 587202269;
  else if (sz < 838860800)*hsz = 1175862839;
  else if (sz < 1677721600)*hsz = 2147321881;
#if defined(PETSC_USE_64BIT_INDICES)
  else if (sz < 3355443200l)*hsz = 4695452647l;
  else if (sz < 6710886400l)*hsz = 9384888067l;
  else if (sz < 13421772800l)*hsz = 18787024237l;
  else if (sz < 26843545600l)*hsz = 32416190071l;
#endif
  else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"A really huge hash is being requested.. cannot process: %" PetscInt_FMT,sz);
  PetscFunctionReturn(0);
}

/*
   PetscTableCreate  Creates a PETSc look up table

   Input Parameters:
+     n - expected number of keys
-     maxkey- largest possible key

   Notes:
    keys are between 1 and maxkey inclusive

*/
PetscErrorCode PetscTableCreate(PetscInt n, PetscInt maxkey, PetscTable *rta)
{
  PetscTable ta;

  PetscFunctionBegin;
  PetscValidPointer(rta,3);
  PetscCheck(n >= 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"n < 0");
  PetscCall(PetscNew(&ta));
  PetscCall(PetscTableCreateHashSize(n,&ta->tablesize));
  PetscCall(PetscCalloc1(ta->tablesize,&ta->keytable));
  PetscCall(PetscMalloc1(ta->tablesize,&ta->table));
  ta->maxkey = maxkey;
  *rta       = ta;
  PetscFunctionReturn(0);
}

/* PetscTableCreate() ********************************************
 *
 * hash table for non-zero data and keys
 *
 */
PetscErrorCode PetscTableCreateCopy(const PetscTable intable, PetscTable *rta)
{
  PetscTable ta;

  PetscFunctionBegin;
  PetscValidPointer(intable,1);
  PetscValidPointer(rta,2);
  PetscCall(PetscNew(&ta));
  ta->tablesize = intable->tablesize;
  PetscCall(PetscMalloc1(ta->tablesize,&ta->keytable));
  PetscCall(PetscMalloc1(ta->tablesize,&ta->table));
  PetscCall(PetscMemcpy(ta->keytable,intable->keytable,ta->tablesize*sizeof(PetscInt)));
  PetscCall(PetscMemcpy(ta->table,intable->table,ta->tablesize*sizeof(PetscInt)));
  for (PetscInt i = 0; i < ta->tablesize; i++) PetscAssert(ta->keytable[i] >= 0,PETSC_COMM_SELF,PETSC_ERR_COR,"ta->keytable[i] < 0");
  ta->count  = intable->count;
  ta->maxkey = intable->maxkey;
  *rta       = ta;
  PetscFunctionReturn(0);
}

/* PetscTableDestroy() ********************************************
 *
 *
 */
PetscErrorCode PetscTableDestroy(PetscTable *ta)
{
  PetscFunctionBegin;
  if (!*ta) PetscFunctionReturn(0);
  PetscCall(PetscFree((*ta)->keytable));
  PetscCall(PetscFree((*ta)->table));
  PetscCall(PetscFree(*ta));
  PetscFunctionReturn(0);
}

/* PetscTableGetCount() ********************************************
 */
PetscErrorCode PetscTableGetCount(const PetscTable ta, PetscInt *count)
{
  PetscFunctionBegin;
  PetscValidPointer(ta,1);
  PetscValidIntPointer(count,2);
  *count = ta->count;
  PetscFunctionReturn(0);
}

/* PetscTableIsEmpty() ********************************************
 */
PetscErrorCode PetscTableIsEmpty(const PetscTable ta, PetscInt *flag)
{
  PetscFunctionBegin;
  PetscValidPointer(ta,1);
  PetscValidIntPointer(flag,2);
  *flag = !(ta->count);
  PetscFunctionReturn(0);
}

/*
    PetscTableAddExpand - called by PetscTableAdd() if more space is needed

*/
PetscErrorCode PetscTableAddExpand(PetscTable ta, PetscInt key, PetscInt data, InsertMode imode)
{
  const PetscInt tsize   = ta->tablesize,tcount = ta->count;
  PetscInt       *oldtab = ta->table,*oldkt = ta->keytable;

  PetscFunctionBegin;
  PetscValidPointer(ta,1);
  PetscCall(PetscTableCreateHashSize(ta->tablesize,&ta->tablesize));
  PetscCall(PetscMalloc1(ta->tablesize,&ta->table));
  PetscCall(PetscCalloc1(ta->tablesize,&ta->keytable));

  ta->count = 0;
  ta->head  = 0;

  PetscCall(PetscTableAdd(ta,key,data,INSERT_VALUES));
  /* rehash */
  for (PetscInt ii = 0; ii < tsize; ++ii) {
    PetscInt newk = oldkt[ii];

    if (newk) PetscCall(PetscTableAdd(ta,newk,oldtab[ii],imode));
  }
  PetscCheck(ta->count == tcount + 1,PETSC_COMM_SELF,PETSC_ERR_COR,"corrupted ta->count");

  PetscCall(PetscFree(oldtab));
  PetscCall(PetscFree(oldkt));
  PetscFunctionReturn(0);
}

/* PetscTableRemoveAll() ********************************************
 *
 *
 */
PetscErrorCode PetscTableRemoveAll(PetscTable ta)
{
  PetscFunctionBegin;
  PetscValidPointer(ta,1);
  ta->head = 0;
  if (ta->count) {
    ta->count = 0;

    PetscCall(PetscArrayzero(ta->keytable,ta->tablesize));
  }
  PetscFunctionReturn(0);
}

/* PetscTableGetHeadPosition() ********************************************
 *
 */
PetscErrorCode PetscTableGetHeadPosition(PetscTable ta, PetscTablePosition *ppos)
{
  PetscInt i = 0;

  PetscFunctionBegin;
  PetscValidPointer(ta,1);
  PetscValidPointer(ppos,2);
  *ppos = NULL;
  if (!ta->count) PetscFunctionReturn(0);

  /* find first valid place */
  do {
    if (ta->keytable[i]) {
      *ppos = (PetscTablePosition)&ta->table[i];
      break;
    }
  } while (i++ < ta->tablesize);
  PetscCheck(*ppos,PETSC_COMM_SELF,PETSC_ERR_COR,"No head");
  PetscFunctionReturn(0);
}

/* PetscTableGetNext() ********************************************
 *
 *  - iteration - PetscTablePosition is always valid (points to a data)
 *
 */
PetscErrorCode PetscTableGetNext(PetscTable ta, PetscTablePosition *rPosition, PetscInt *pkey, PetscInt *data)
{
  PetscInt           idex;
  PetscTablePosition pos;

  PetscFunctionBegin;
  PetscValidPointer(ta,1);
  PetscValidPointer(rPosition,2);
  PetscValidIntPointer(pkey,3);
  PetscValidIntPointer(data,4);
  pos = *rPosition;
  PetscCheck(pos,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Null position");
  *data = *pos;
  PetscCheckFalse(!*data,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Null data");
  idex  = pos - ta->table;
  *pkey = ta->keytable[idex];
  PetscCheckFalse(!*pkey,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Null key");

  /* get next */
  do {
    pos++;  idex++;
    if (idex >= ta->tablesize) {
      pos = NULL; /* end of list */
      break;
    } else if (ta->keytable[idex]) {
      pos = ta->table + idex;
      break;
    }
  } while (idex < ta->tablesize);
  *rPosition = pos;
  PetscFunctionReturn(0);
}

PetscErrorCode  PetscTableAddCountExpand(PetscTable ta,PetscInt key)
{
  PetscInt       ii      = 0,hash = PetscHash(ta,key);
  const PetscInt tsize   = ta->tablesize,tcount = ta->count;
  PetscInt       *oldtab = ta->table,*oldkt = ta->keytable,newk,ndata;

  PetscFunctionBegin;
  /* before making the table larger check if key is already in table */
  while (ii++ < tsize) {
    if (ta->keytable[hash] == key) PetscFunctionReturn(0);
    hash = (hash == (ta->tablesize-1)) ? 0 : hash+1;
  }

  ta->tablesize = PetscIntMultTruncate(2,ta->tablesize);
  PetscCheckFalse(tsize == ta->tablesize,PETSC_COMM_SELF,PETSC_ERR_SUP,"Table is as large as possible; ./configure with the option --with-64-bit-integers to run this large case");
  PetscCall(PetscMalloc1(ta->tablesize,&ta->table));
  PetscCall(PetscCalloc1(ta->tablesize,&ta->keytable));

  ta->count = 0;
  ta->head  = 0;

  /* Build a new copy of the data */
  for (ii = 0; ii < tsize; ii++) {
    newk = oldkt[ii];
    if (newk) {
      ndata = oldtab[ii];
      PetscCall(PetscTableAdd(ta,newk,ndata,INSERT_VALUES));
    }
  }
  PetscCall(PetscTableAddCount(ta,key));
  PetscCheckFalse(ta->count != tcount + 1,PETSC_COMM_SELF,PETSC_ERR_COR,"corrupted ta->count");

  PetscCall(PetscFree(oldtab));
  PetscCall(PetscFree(oldkt));
  PetscFunctionReturn(0);
}
