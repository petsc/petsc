
/* Contributed by - Mark Adams */

#include <petscsys.h>


#undef __FUNCT__
#define __FUNCT__ "PetscTableCreate"
/*
   PetscTableCreate  Creates a PETSc look up table

   Input Parameters:
+     n - expected number of keys
-     maxkey- largest possible key

   Notes: keys are between 1 and N inclusive

*/
PetscErrorCode  PetscTableCreate(const PetscInt n,PetscInt maxkey,PetscTable *rta)
{
  PetscTable     ta;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (n < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"n < 0");
  ierr          = PetscNew(struct _n_PetscTable,&ta);CHKERRQ(ierr);
  ta->tablesize = (3*n)/2 + 17;
  if (ta->tablesize < n) ta->tablesize = PETSC_MAX_INT/4; /* overflow */
  ierr          = PetscMalloc(sizeof(PetscInt)*ta->tablesize,&ta->keytable);CHKERRQ(ierr);
  ierr          = PetscMemzero(ta->keytable,sizeof(PetscInt)*ta->tablesize);CHKERRQ(ierr);
  ierr          = PetscMalloc(sizeof(PetscInt)*ta->tablesize,&ta->table);CHKERRQ(ierr);
  ta->head      = 0;
  ta->count     = 0;
  ta->maxkey    = maxkey;
  *rta          = ta;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscTableCreateCopy"
/* PetscTableCreate() ********************************************
 *
 * hash table for non-zero data and keys
 *
 */
PetscErrorCode  PetscTableCreateCopy(const PetscTable intable,PetscTable *rta)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscTable     ta;

  PetscFunctionBegin;
  ierr          = PetscNew(struct _n_PetscTable,&ta);CHKERRQ(ierr);
  ta->tablesize = intable->tablesize;
  ierr          = PetscMalloc(sizeof(PetscInt)*ta->tablesize,&ta->keytable);CHKERRQ(ierr);
  ierr          = PetscMalloc(sizeof(PetscInt)*ta->tablesize,&ta->table);CHKERRQ(ierr);
  for (i = 0; i < ta->tablesize ; i++){
    ta->keytable[i] = intable->keytable[i];
    ta->table[i]    = intable->table[i];
#if defined(PETSC_USE_DEBUG)
    if (ta->keytable[i] < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_COR,"ta->keytable[i] < 0");
#endif
 }
  ta->head   = 0;
  ta->count  = intable->count;
  ta->maxkey = intable->maxkey;
  *rta       = ta;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscTableDestroy"
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

#undef __FUNCT__
#define __FUNCT__ "PetscTableGetCount"
/* PetscTableGetCount() ********************************************
 */
PetscErrorCode  PetscTableGetCount(const PetscTable ta,PetscInt *count)
{
  PetscFunctionBegin;
  *count = ta->count;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscTableIsEmpty"
/* PetscTableIsEmpty() ********************************************
 */
PetscErrorCode  PetscTableIsEmpty(const PetscTable ta,PetscInt *flag)
{
  PetscFunctionBegin;
  *flag = !(ta->count);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscTableAddExpand"
/*
    PetscTableAddExpand - called PetscTableAdd() if more space needed

*/
PetscErrorCode  PetscTableAddExpand(PetscTable ta,PetscInt key,PetscInt data,InsertMode imode)
{
  PetscErrorCode ierr;
  PetscInt       ii = 0;
  const PetscInt tsize = ta->tablesize,tcount = ta->count;
  PetscInt       *oldtab = ta->table,*oldkt = ta->keytable,newk,ndata;

  PetscFunctionBegin;
  if (ta->tablesize == PETSC_MAX_INT/4) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_COR,"ta->tablesize < 0");
  ta->tablesize = 2*tsize;
  if (ta->tablesize <= tsize) ta->tablesize = PETSC_MAX_INT/4;

  ierr = PetscMalloc(ta->tablesize*sizeof(PetscInt),&ta->table);CHKERRQ(ierr);
  ierr = PetscMalloc(ta->tablesize*sizeof(PetscInt),&ta->keytable);CHKERRQ(ierr);
  ierr = PetscMemzero(ta->keytable,ta->tablesize*sizeof(PetscInt));CHKERRQ(ierr);

  ta->count     = 0;
  ta->head      = 0;

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


#undef __FUNCT__
#define __FUNCT__ "PetscTableRemoveAll"
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
    ierr = PetscMemzero(ta->keytable,ta->tablesize*sizeof(PetscInt));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}



#undef __FUNCT__
#define __FUNCT__ "PetscTableGetHeadPosition"
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

#undef __FUNCT__
#define __FUNCT__ "PetscTableGetNext"
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
  idex = pos - ta->table;
  *pkey = ta->keytable[idex];
  if (!*pkey) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Null key");

  /* get next */
  do {
    pos++;  idex++;
    if (idex >= ta->tablesize) {
      pos = 0; /* end of list */
      break;
    } else if (ta->keytable[idex]) {
      pos = ta->table + idex;
      break;
    }
  } while (idex < ta->tablesize);
  *rPosition = pos;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "PetscTableAddCountExpand"
PetscErrorCode  PetscTableAddCountExpand(PetscTable ta,PetscInt key)
{
  PetscErrorCode ierr;
  PetscInt       ii = 0,hash = PetscHash(ta,key);
  const PetscInt tsize = ta->tablesize,tcount = ta->count;
  PetscInt *oldtab = ta->table,*oldkt = ta->keytable,newk,ndata;

  PetscFunctionBegin;
  /* before making the table larger check if key is already in table */
  while (ii++ < ta->tablesize){
    if (ta->keytable[hash] == key) PetscFunctionReturn(0);
    hash = (hash == (ta->tablesize-1)) ? 0 : hash+1;
  }

  /* alloc new (bigger) table */
  if (ta->tablesize == PETSC_MAX_INT/4) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_COR,"ta->tablesize < 0");
  ta->tablesize = 2*tsize;
  if (ta->tablesize <= tsize) ta->tablesize = PETSC_MAX_INT/4;

  ierr = PetscMalloc(ta->tablesize*sizeof(PetscInt),&ta->table);CHKERRQ(ierr);
  ierr = PetscMalloc(ta->tablesize*sizeof(PetscInt),&ta->keytable);CHKERRQ(ierr);
  ierr = PetscMemzero(ta->keytable,ta->tablesize*sizeof(PetscInt));CHKERRQ(ierr);

  ta->count     = 0;
  ta->head      = 0;

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

