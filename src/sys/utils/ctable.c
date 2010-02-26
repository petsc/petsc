#define PETSC_DLL
/* Contributed by - Mark Adams */

#include "petscsys.h"
#include "../src/sys/ctable.h" 
#if defined (PETSC_HAVE_LIMITS_H)
#include <limits.h>
#endif
#define HASH_FACT 79943
#define HASHT(ta,x) ((unsigned long)((HASH_FACT*(unsigned long)x)%ta->tablesize))

#undef __FUNCT__  
#define __FUNCT__ "PetscTableCreate"
/* PetscTableCreate() ********************************************
 * 
 * hash table for non-zero data and keys 
 *
 */
PetscErrorCode PETSC_DLLEXPORT PetscTableCreate(const PetscInt n,PetscTable *rta)
{
  PetscTable     ta;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (n < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"n < 0"); 
  ierr          = PetscNew(struct _n_PetscTable,&ta);CHKERRQ(ierr);
  ta->tablesize = (3*n)/2 + 17;
  if (ta->tablesize < n) ta->tablesize = INT_MAX/4; /* overflow */
  ierr          = PetscMalloc(sizeof(PetscInt)*ta->tablesize,&ta->keytable);CHKERRQ(ierr);
  ierr          = PetscMemzero(ta->keytable,sizeof(PetscInt)*ta->tablesize);CHKERRQ(ierr);
  ierr          = PetscMalloc(sizeof(PetscInt)*ta->tablesize,&ta->table);CHKERRQ(ierr);
  ta->head      = 0;
  ta->count     = 0;
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
PetscErrorCode PETSC_DLLEXPORT PetscTableCreateCopy(const PetscTable intable,PetscTable *rta)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscTable     ta;

  PetscFunctionBegin;
  ierr          = PetscNew(struct _n_PetscTable,&ta);CHKERRQ(ierr);
  ta->tablesize = intable->tablesize;
  ierr          = PetscMalloc(sizeof(PetscInt)*ta->tablesize,&ta->keytable);CHKERRQ(ierr);
  ierr          = PetscMalloc(sizeof(PetscInt)*ta->tablesize,&ta->table);CHKERRQ(ierr);
  for(i = 0 ; i < ta->tablesize ; i++){
    ta->keytable[i] = intable->keytable[i]; 
    ta->table[i]    = intable->table[i];
#if defined(PETSC_USE_DEBUG)    
    if (ta->keytable[i] < 0) SETERRQ(PETSC_ERR_COR,"ta->keytable[i] < 0"); 
#endif  
 }
  ta->head  = 0;
  ta->count = intable->count;
  *rta      = ta;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscTableDestroy"
/* PetscTableDestroy() ********************************************
 * 
 *
 */
PetscErrorCode PETSC_DLLEXPORT PetscTableDestroy(PetscTable ta)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(ta->keytable);CHKERRQ(ierr);
  ierr = PetscFree(ta->table);CHKERRQ(ierr);
  ierr = PetscFree(ta);CHKERRQ(ierr);
  PetscFunctionReturn(0);
} 
#undef __FUNCT__  
#define __FUNCT__ "PetscTableGetCount"
/* PetscTableGetCount() ********************************************
 */
PetscErrorCode PETSC_DLLEXPORT PetscTableGetCount(const PetscTable ta,PetscInt *count) 
{ 
  PetscFunctionBegin;
  *count = ta->count;
  PetscFunctionReturn(0); 
}

#undef __FUNCT__  
#define __FUNCT__ "PetscTableIsEmpty"
/* PetscTableIsEmpty() ********************************************
 */
PetscErrorCode PETSC_DLLEXPORT PetscTableIsEmpty(const PetscTable ta,PetscInt *flag) 
{ 
  PetscFunctionBegin;
  *flag = !(ta->count); 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscTableAdd"
/* PetscTableAdd() ********************************************
 *
 */
PetscErrorCode PETSC_DLLEXPORT PetscTableAdd(PetscTable ta,const PetscInt key,const PetscInt data)
{  
  PetscErrorCode ierr;
  PetscInt       ii = 0,hash = HASHT(ta,key);
  const PetscInt tsize = ta->tablesize,tcount = ta->count; 
  
  PetscFunctionBegin;
  if (key <= 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"key <= 0");
  if (!data) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Null data");
  
  if (ta->count < 5*(ta->tablesize/6) - 1) {
    while (ii++ < ta->tablesize){
      if (ta->keytable[hash] == key) {
	ta->table[hash] = data; /* over write */
	PetscFunctionReturn(0); 
      } else if (!ta->keytable[hash]) {
	ta->count++; /* add */
	ta->keytable[hash] = key; ta->table[hash] = data;
	PetscFunctionReturn(0);
      }
      hash = (hash == (ta->tablesize-1)) ? 0 : hash+1; 
    }  
    SETERRQ(PETSC_ERR_COR,"Full table");
  } else {
    PetscInt *oldtab = ta->table,*oldkt = ta->keytable,newk,ndata;

    /* alloc new (bigger) table */
    if (ta->tablesize == INT_MAX/4) SETERRQ(PETSC_ERR_COR,"ta->tablesize < 0");
    ta->tablesize = 2*tsize; 
    if (ta->tablesize <= tsize) ta->tablesize = INT_MAX/4;

    ierr = PetscMalloc(ta->tablesize*sizeof(PetscInt),&ta->table);CHKERRQ(ierr);
    ierr = PetscMalloc(ta->tablesize*sizeof(PetscInt),&ta->keytable);CHKERRQ(ierr);
    ierr = PetscMemzero(ta->keytable,ta->tablesize*sizeof(PetscInt));CHKERRQ(ierr);

    ta->count     = 0;
    ta->head      = 0; 
    
    ierr = PetscTableAdd(ta,key,data);CHKERRQ(ierr); 
    /* rehash */
    for (ii = 0; ii < tsize; ii++) {
      newk = oldkt[ii];
      if (newk) {
	ndata = oldtab[ii];
	ierr  = PetscTableAdd(ta,newk,ndata);CHKERRQ(ierr); 
      }
    }
    if (ta->count != tcount + 1) SETERRQ(PETSC_ERR_COR,"corrupted ta->count");
    
    ierr = PetscFree(oldtab);CHKERRQ(ierr);
    ierr = PetscFree(oldkt);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscTableRemoveAll"
/* PetscTableRemoveAll() ********************************************
 *
 *
 */
PetscErrorCode PETSC_DLLEXPORT PetscTableRemoveAll(PetscTable ta)
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
#define __FUNCT__ "PetscTableFind"
/* PetscTableFind() ********************************************
 *
 * returns data. If data==0, then no table entry exists.
 *
 */
PetscErrorCode PETSC_DLLEXPORT PetscTableFind(PetscTable ta,const PetscInt key,PetscInt *data) 
{  
  PetscInt hash,ii = 0;

  PetscFunctionBegin;
  if (!key) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Null key");
  hash  = HASHT(ta,key);
  *data = 0;
  while (ii++ < ta->tablesize) {
    if (!ta->keytable[hash]) break;
    else if (ta->keytable[hash] == key) { 
      *data = ta->table[hash]; 
      break; 
    }
    hash = (hash == (ta->tablesize-1)) ? 0 : hash+1; 
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscTableGetHeadPosition"
/* PetscTableGetHeadPosition() ********************************************
 *
 */
PetscErrorCode PETSC_DLLEXPORT PetscTableGetHeadPosition(PetscTable ta,PetscTablePosition *ppos)
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
  if (!*ppos) SETERRQ(PETSC_ERR_COR,"No head");
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscTableGetNext"
/* PetscTableGetNext() ********************************************
 *
 *  - iteration - PetscTablePosition is always valid (points to a data)
 *  
 */
PetscErrorCode PETSC_DLLEXPORT PetscTableGetNext(PetscTable ta,PetscTablePosition *rPosition,PetscInt *pkey,PetscInt *data)
{
  PetscInt           idex; 
  PetscTablePosition pos;

  PetscFunctionBegin;
  pos = *rPosition;
  if (!pos) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Null position"); 
  *data = *pos; 
  if (!*data) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Null data"); 
  idex = pos - ta->table;
  *pkey = ta->keytable[idex]; 
  if (!*pkey) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Null key");  

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
#define __FUNCT__ "PetscTableAddCount"
/* 
     PetscTableAddCount - adds another key to the hash table and gives it the data of the current size of the table,
          if the entry already exists then just return
 *
 */
PetscErrorCode PETSC_DLLEXPORT PetscTableAddCount(PetscTable ta,const PetscInt key)
{  
  PetscErrorCode ierr;
  PetscInt       ii = 0,hash = HASHT(ta,key);
  const PetscInt tsize = ta->tablesize,tcount = ta->count; 
  
  PetscFunctionBegin;
  if (key <= 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"key <= 0");
  
  if (ta->count < 5*(ta->tablesize/6) - 1) {
    while (ii++ < ta->tablesize){
      if (ta->keytable[hash] == key) {
	PetscFunctionReturn(0); 
      } else if (!ta->keytable[hash]) {
	ta->count++; /* add */
	ta->keytable[hash] = key; ta->table[hash] = ta->count;
	PetscFunctionReturn(0);
      }
      hash = (hash == (ta->tablesize-1)) ? 0 : hash+1; 
    }  
    SETERRQ(PETSC_ERR_COR,"Full table");
  } else {
    PetscInt *oldtab = ta->table,*oldkt = ta->keytable,newk,ndata;

    /* before making the table larger check if key is already in table */
    while (ii++ < ta->tablesize){
      if (ta->keytable[hash] == key) PetscFunctionReturn(0); 
      hash = (hash == (ta->tablesize-1)) ? 0 : hash+1; 
    }  

    /* alloc new (bigger) table */
    if (ta->tablesize == INT_MAX/4) SETERRQ(PETSC_ERR_COR,"ta->tablesize < 0");
    ta->tablesize = 2*tsize; 
    if (ta->tablesize <= tsize) ta->tablesize = INT_MAX/4;

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
	ierr  = PetscTableAdd(ta,newk,ndata);CHKERRQ(ierr); 
      }
    }
    ierr = PetscTableAddCount(ta,key);CHKERRQ(ierr); 
    if (ta->count != tcount + 1) SETERRQ(PETSC_ERR_COR,"corrupted ta->count");
    
    ierr = PetscFree(oldtab);CHKERRQ(ierr);
    ierr = PetscFree(oldkt);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

