
/* Contributed by - Mark Adams */

#include <petscsys.h>
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
  else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"A really huge hash is being requested.. cannot process: %" PetscInt_FMT,sz);
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
PetscErrorCode  PetscTableCreate(const PetscInt n,PetscInt maxkey,PetscTable *rta)
{
  PetscTable     ta;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (n < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"n < 0");
  ierr       = PetscNew(&ta);CHKERRQ(ierr);
  ierr       = PetscTableCreateHashSize(n,&ta->tablesize);CHKERRQ(ierr);
  ierr       = PetscCalloc1(ta->tablesize,&ta->keytable);CHKERRQ(ierr);
  ierr       = PetscMalloc1(ta->tablesize,&ta->table);CHKERRQ(ierr);
  ta->head   = 0;
  ta->count  = 0;
  ta->maxkey = maxkey;
  *rta       = ta;
  PetscFunctionReturn(0);
}

/* PetscTableCreate() ********************************************
 *
 * hash table for non-zero data and keys
 *
 */
PetscErrorCode  PetscTableCreateCopy(const PetscTable intable,PetscTable *rta)
{
  PetscErrorCode ierr;
  PetscTable     ta;

  PetscFunctionBegin;
  ierr          = PetscNew(&ta);CHKERRQ(ierr);
  ta->tablesize = intable->tablesize;
  ierr          = PetscMalloc1(ta->tablesize,&ta->keytable);CHKERRQ(ierr);
  ierr          = PetscMalloc1(ta->tablesize,&ta->table);CHKERRQ(ierr);
  ierr          = PetscMemcpy(ta->keytable,intable->keytable,ta->tablesize*sizeof(PetscInt));CHKERRQ(ierr);
  ierr          = PetscMemcpy(ta->table,intable->table,ta->tablesize*sizeof(PetscInt));CHKERRQ(ierr);
  if (PetscDefined(USE_DEBUG)) {
    PetscInt i;
    for (i = 0; i < ta->tablesize; i++) {
      if (ta->keytable[i] < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_COR,"ta->keytable[i] < 0");
    }
  }
  ta->head   = 0;
  ta->count  = intable->count;
  ta->maxkey = intable->maxkey;
  *rta       = ta;
  PetscFunctionReturn(0);
}

/* PetscTableDestroy() ********************************************
 *
 *
 */
PetscErrorCode  PetscTableDestroy(PetscTable *ta)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*ta) PetscFunctionReturn(0);
  ierr = PetscFree((*ta)->keytable);CHKERRQ(ierr);
  ierr = PetscFree((*ta)->table);CHKERRQ(ierr);
  ierr = PetscFree(*ta);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* PetscTableGetCount() ********************************************
 */
PetscErrorCode  PetscTableGetCount(const PetscTable ta,PetscInt *count)
{
  PetscFunctionBegin;
  *count = ta->count;
  PetscFunctionReturn(0);
}

/* PetscTableIsEmpty() ********************************************
 */
PetscErrorCode  PetscTableIsEmpty(const PetscTable ta,PetscInt *flag)
{
  PetscFunctionBegin;
  *flag = !(ta->count);
  PetscFunctionReturn(0);
}

/*
    PetscTableAddExpand - called by PetscTableAdd() if more space is needed

*/
PetscErrorCode  PetscTableAddExpand(PetscTable ta,PetscInt key,PetscInt data,InsertMode imode)
{
  PetscErrorCode ierr;
  PetscInt       ii      = 0;
  const PetscInt tsize   = ta->tablesize,tcount = ta->count;
  PetscInt       *oldtab = ta->table,*oldkt = ta->keytable,newk,ndata;

  PetscFunctionBegin;
  ierr = PetscTableCreateHashSize(ta->tablesize,&ta->tablesize);CHKERRQ(ierr);
  ierr = PetscMalloc1(ta->tablesize,&ta->table);CHKERRQ(ierr);
  ierr = PetscCalloc1(ta->tablesize,&ta->keytable);CHKERRQ(ierr);

  ta->count = 0;
  ta->head  = 0;

  ierr = PetscTableAdd(ta,key,data,INSERT_VALUES);CHKERRQ(ierr);
  /* rehash */
  for (ii = 0; ii < tsize; ii++) {
    newk = oldkt[ii];
    if (newk) {
      ndata = oldtab[ii];
      ierr  = PetscTableAdd(ta,newk,ndata,imode);CHKERRQ(ierr);
    }
  }
  if (ta->count != tcount + 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_COR,"corrupted ta->count");

  ierr = PetscFree(oldtab);CHKERRQ(ierr);
  ierr = PetscFree(oldkt);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* PetscTableRemoveAll() ********************************************
 *
 *
 */
PetscErrorCode  PetscTableRemoveAll(PetscTable ta)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ta->head = 0;
  if (ta->count) {
    ta->count = 0;

    ierr = PetscArrayzero(ta->keytable,ta->tablesize);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* PetscTableGetHeadPosition() ********************************************
 *
 */
PetscErrorCode  PetscTableGetHeadPosition(PetscTable ta,PetscTablePosition *ppos)
{
  PetscInt i = 0;

  PetscFunctionBegin;
  *ppos = NULL;
  if (!ta->count) PetscFunctionReturn(0);

  /* find first valid place */
  do {
    if (ta->keytable[i]) {
      *ppos = (PetscTablePosition)&ta->table[i];
      break;
    }
  } while (i++ < ta->tablesize);
  if (!*ppos) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_COR,"No head");
  PetscFunctionReturn(0);
}

/* PetscTableGetNext() ********************************************
 *
 *  - iteration - PetscTablePosition is always valid (points to a data)
 *
 */
PetscErrorCode  PetscTableGetNext(PetscTable ta,PetscTablePosition *rPosition,PetscInt *pkey,PetscInt *data)
{
  PetscInt           idex;
  PetscTablePosition pos;

  PetscFunctionBegin;
  pos = *rPosition;
  if (!pos) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Null position");
  *data = *pos;
  if (!*data) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Null data");
  idex  = pos - ta->table;
  *pkey = ta->keytable[idex];
  if (!*pkey) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Null key");

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
  PetscErrorCode ierr;
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
  if (tsize == ta->tablesize) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Table is as large as possible; ./configure with the option --with-64-bit-integers to run this large case");
  ierr = PetscMalloc1(ta->tablesize,&ta->table);CHKERRQ(ierr);
  ierr = PetscCalloc1(ta->tablesize,&ta->keytable);CHKERRQ(ierr);

  ta->count = 0;
  ta->head  = 0;

  /* Build a new copy of the data */
  for (ii = 0; ii < tsize; ii++) {
    newk = oldkt[ii];
    if (newk) {
      ndata = oldtab[ii];
      ierr  = PetscTableAdd(ta,newk,ndata,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = PetscTableAddCount(ta,key);CHKERRQ(ierr);
  if (ta->count != tcount + 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_COR,"corrupted ta->count");

  ierr = PetscFree(oldtab);CHKERRQ(ierr);
  ierr = PetscFree(oldkt);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

