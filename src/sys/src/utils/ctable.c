/*$Id: ctable.c,v 1.18 2001/03/23 23:20:45 balay Exp $*/
/* Contributed by - Mark Adams */

#include "petsc.h"
#include "src/sys/ctable.h" 
#if defined (PETSC_HAVE_LIMITS_H)
#include <limits.h>
#endif
#define HASHT(ta,x) ((3*x)%ta->tablesize)

#undef __FUNCT__  
#define __FUNCT__ "PetscTableCreate"
/* PetscTableCreate() ********************************************
 * 
 * hash table for non-zero data and keys 
 *
 */
int PetscTableCreate(const int n,PetscTable *rta)
{
  PetscTable ta;
  int        ierr;

  PetscFunctionBegin;
  if(n < 0) SETERRQ(1,"PetscTable error: n < 0"); 
  ierr          = PetscNew(struct _p_PetscTable,&ta);CHKERRQ(ierr);
  ta->tablesize = (3*n)/2 + 17;
  if(ta->tablesize < n) ta->tablesize = INT_MAX/4; /* overflow */
  ierr          = PetscMalloc(sizeof(int)*ta->tablesize,&ta->keytable);CHKERRQ(ierr);
  ierr          = PetscMemzero(ta->keytable,sizeof(int)*ta->tablesize);CHKERRQ(ierr);
  ierr          = PetscMalloc(sizeof(int)*ta->tablesize,&ta->table);CHKERRQ(ierr);
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
int PetscTableCreateCopy(const PetscTable intable,PetscTable *rta)
{
  int        i,ierr;
  PetscTable ta;

  PetscFunctionBegin;
  ierr          = PetscNew(struct _p_PetscTable,&ta);CHKERRQ(ierr);
  ta->tablesize = intable->tablesize;
  ierr          = PetscMalloc(sizeof(int)*ta->tablesize,&ta->keytable);CHKERRQ(ierr);
  ierr          = PetscMalloc(sizeof(int)*ta->tablesize,&ta->table);CHKERRQ(ierr);
  for(i = 0 ; i < ta->tablesize ; i++){
    ta->keytable[i] = intable->keytable[i]; 
    ta->table[i]    = intable->table[i];
#if defined(PETSC_USE_BOPT_g)    
    if(ta->keytable[i] < 0) SETERRQ(1,"TABLE error: ta->keytable[i] < 0"); 
#endif  
 }
  ta->head  = 0;
  ta->count = intable->count;
  
  *rta      = ta;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscTableDelete"
/* PetscTableDelete() ********************************************
 * 
 *
 */
int PetscTableDelete(PetscTable ta)
{
  int ierr;

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
int PetscTableGetCount(const PetscTable ta,int *count) 
{ 
  PetscFunctionBegin;
  *count = ta->count;
  PetscFunctionReturn(0); 
}

#undef __FUNCT__  
#define __FUNCT__ "PetscTableIsEmpty"
/* PetscTableIsEmpty() ********************************************
 */
int PetscTableIsEmpty(const PetscTable ta,int *flag) 
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
int PetscTableAdd(PetscTable ta,const int key,const int data)
{  
  int       ii = 0,hash = HASHT(ta,key),ierr;
  const int tsize = ta->tablesize,tcount = ta->count; 
  
  PetscFunctionBegin;
  if (key <= 0) SETERRQ(1,"PetscTable error: key <= 0");
  if (!data) SETERRQ(1,"PetscTable error: Table zero data");
  
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
    SETERRQ(1,"PetscTable error: full table");
  } else {
    int *oldtab = ta->table,*oldkt = ta->keytable,newk,ndata;

    /* alloc new (bigger) table */
    if(ta->tablesize == INT_MAX/4) SETERRQ(1,"PetscTable error: ta->tablesize < 0");
    ta->tablesize = 2*tsize; 
    if (ta->tablesize <= tsize) ta->tablesize = INT_MAX/4;

    ierr = PetscMalloc(ta->tablesize*sizeof(int),&ta->table);CHKERRQ(ierr);
    ierr = PetscMalloc(ta->tablesize*sizeof(int),&ta->keytable);CHKERRQ(ierr);
    ierr = PetscMemzero(ta->keytable,ta->tablesize*sizeof(int));CHKERRQ(ierr);

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
    if (ta->count != tcount + 1) SETERRQ(1,"PetscTable error");
    
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
int PetscTableRemoveAll(PetscTable ta)
{ 
  int ierr;

  PetscFunctionBegin;
  ta->head = 0;
  if (ta->count) { 
    ta->count = 0;
    ierr = PetscMemzero(ta->keytable,ta->tablesize*sizeof(int));CHKERRQ(ierr);
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
int PetscTableFind(PetscTable ta,const int key,int *data) 
{  
  int hash,ii = 0;

  PetscFunctionBegin;
  if(!key) SETERRQ(1,"PetscTable error: PetscTable zero key");
  hash = HASHT(ta,key);
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
int PetscTableGetHeadPosition(PetscTable ta,PetscTablePosition *ppos)
{
  int i = 0;

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
  if (!*ppos) SETERRQ(1,"TABLE error: No head");

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscTableGetNext"
/* PetscTableGetNext() ********************************************
 *
 *  - iteration - PetscTablePosition is always valid (points to a data)
 *  
 */
int PetscTableGetNext(PetscTable ta,PetscTablePosition *rPosition,int *pkey,int *data)
{
  int                idex; 
  PetscTablePosition pos;

  PetscFunctionBegin;
  pos = *rPosition;
  if (!pos) SETERRQ(1,"PetscTable error: PetscTable null position"); 
  *data = *pos; 
  if (!*data) SETERRQ(1,"PetscTable error"); 
  idex = pos - ta->table;
  *pkey = ta->keytable[idex]; 
  if (!*pkey) SETERRQ(1,"PetscTable error");  

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




