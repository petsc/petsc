#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ctable.c,v 1.8 1999/06/30 23:49:46 balay Exp balay $";
#endif
/* Contributed by - Mark Adams */

#include "petsc.h"
#include "src/sys/ctable.h" 
#if defined (PETSC_HAVE_LIMITS_H)
#include <limits.h>
#endif
#define HASHT(ta,x) ((3*x)%ta->tablesize)

#undef __FUNC__  
#define __FUNC__ "TableCreate"
/* TableCreate() ********************************************
 * 
 * hash table for non-zero data and keys 
 *
 */
int TableCreate(const int n,Table *rta)
{
  Table ta;
  int   ierr;

  PetscFunctionBegin;
  if(n < 0) SETERRQ(1,1,"TABLE error: n < 0"); 
  ta            = (Table)PetscMalloc(sizeof(Table_struct));CHKPTRQ(ta);
  ta->tablesize = (3*n)/2 + 17;
  if(ta->tablesize < n) ta->tablesize = INT_MAX/4; /* overflow */
  ta->keytable  = (int*)PetscMalloc(sizeof(int)*ta->tablesize);CHKPTRQ(ta->keytable);
  ierr          = PetscMemzero(ta->keytable,sizeof(int)*ta->tablesize);CHKERRQ(ierr);
  ta->table     = (int*)PetscMalloc(sizeof(int)*ta->tablesize);CHKPTRQ(ta->table);
  ta->head      = 0;
  ta->count     = 0;
  
  *rta          = ta;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "TableCreateCopy"
/* TableCreate() ********************************************
 * 
 * hash table for non-zero data and keys 
 *
 */
int TableCreateCopy(const Table intable,Table *rta)
{
  int i;
  Table ta;

  PetscFunctionBegin;
  ta            = (Table)PetscMalloc(sizeof(Table_struct));CHKPTRQ(ta);
  ta->tablesize = intable->tablesize;
  ta->keytable  = (int*)PetscMalloc(sizeof(int)*ta->tablesize);CHKPTRQ(ta->keytable);
  ta->table     = (int*)PetscMalloc(sizeof(int)*ta->tablesize);CHKPTRQ(ta->table);
  for( i = 0 ; i < ta->tablesize ; i++ ){
    ta->keytable[i] = intable->keytable[i]; 
    ta->table[i]    = intable->table[i];
#if defined(USE_PETSC_BOPT_g)    
    if(ta->keytable[i] < 0) SETERRQ(1,1,"TABLE error: ta->keytable[i] < 0"); 
#endif  
 }
  ta->head  = 0;
  ta->count = intable->count;
  
  *rta      = ta;

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "TableDelete"
/* TableDelete() ********************************************
 * 
 *
 */
int TableDelete(Table ta)
{
  int ierr;

  PetscFunctionBegin;
  ierr = PetscFree(ta->keytable);CHKERRQ(ierr);
  ierr = PetscFree(ta->table);CHKERRQ(ierr);
  ierr = PetscFree(ta);CHKERRQ(ierr);
  PetscFunctionReturn(0);
} 
#undef __FUNC__  
#define __FUNC__ "TableGetCount"
/* TableGetCount() ********************************************
 */
int TableGetCount(const Table ta, int *count) 
{ 
  PetscFunctionBegin;
  *count = ta->count;
  PetscFunctionReturn(0); 
}

#undef __FUNC__  
#define __FUNC__ "TableIsEmpty"
/* TableIsEmpty() ********************************************
 */
int TableIsEmpty(const Table ta, int *flag) 
{ 
  PetscFunctionBegin;
  *flag = !(ta->count); 
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "TableAdd"
/* TableAdd() ********************************************
 *
 */
int TableAdd(Table ta, const int key, const int data)
{  
  int ii = 0,hash = HASHT(ta,key),ierr;
  const int tsize = ta->tablesize,tcount = ta->count; 
  
  PetscFunctionBegin;
  if( key <= 0 ) SETERRQ(1,1,"TABLE error: key <= 0");
  if( data == 0 ) SETERRQ(1,1,"TABLE error: Table zero data");
  
  if (ta->count < 5*(ta->tablesize/6) - 1) {
    while (ii++ < ta->tablesize){
      if (ta->keytable[hash] == key) {
	ta->table[hash] = data; /* over write */
	PetscFunctionReturn(0); 
      }
      else if (!ta->keytable[hash]) {
	ta->count++; /* add */
	ta->keytable[hash] = key; ta->table[hash] = data;
	PetscFunctionReturn(0);
      }
      hash = (hash == (ta->tablesize-1)) ? 0 : hash+1; 
    }  
    SETERRQ(1,1,"TABLE error: full table");
  }
  else {
    int *oldtab = ta->table, *oldkt = ta->keytable, newk, ndata;
    int sz;

    /* alloc new (bigger) table */
    if(ta->tablesize == INT_MAX/4) SETERRQ(1,1,"TABLE error: ta->tablesize < 0");
    ta->tablesize = 2*tsize; 
    if ( ta->tablesize <= tsize ) ta->tablesize = INT_MAX/4;

    sz = sizeof(int)*ta->tablesize;
    ta->table    = (int*)PetscMalloc( sz );CHKPTRQ(ta->table);
    ta->keytable = (int*)PetscMalloc( sz );CHKPTRQ(ta->keytable);
    ierr         = PetscMemzero( ta->keytable, sz );CHKERRQ(ierr);

    ta->count     = 0;
    ta->head      = 0; 
    
    ierr = TableAdd(ta,key,data);CHKERRQ(ierr); 
    /* rehash */
    for ( ii = 0; ii < tsize; ii++ ) {
      newk = oldkt[ii];
      if (newk) {
	ndata = oldtab[ii];
	ierr  = TableAdd(ta,newk,ndata);CHKERRQ(ierr); 
      }
    }
    if (ta->count != tcount + 1) SETERRQ(1,1,"Table error");
    
    ierr = PetscFree(oldtab);CHKERRQ(ierr);
    ierr = PetscFree(oldkt);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "TableRemoveAll"
/* TableRemoveAll() ********************************************
 *
 *
 */
int TableRemoveAll(Table ta)
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

#undef __FUNC__  
#define __FUNC__ "TableFind"
/* TableFind() ********************************************
 *
 * returns data. If data==0, then no table entry exists.
 *
 */
int TableFind(Table ta, const int key, int *data) 
{  
  int hash, ii = 0;

  PetscFunctionBegin;
  if( key == 0 ) SETERRQ(1,1,"TABLE error: Table zero key");
  hash = HASHT( ta, key );
  *data = 0;
  while ( ii++ < ta->tablesize ) {
    if (ta->keytable[hash] == 0) break;
    else if (ta->keytable[hash] == key) { 
      *data = ta->table[hash]; 
      break; 
    }
    hash = (hash == (ta->tablesize-1)) ? 0 : hash+1; 
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "TableGetHeadPosition"
/* TableGetHeadPosition() ********************************************
 *
 */
int TableGetHeadPosition(Table ta, CTablePos *ppos)
{
  int i = 0;

  PetscFunctionBegin;
  *ppos = NULL;
  if (!ta->count) PetscFunctionReturn(0);
  
  /* find first valid place */
  do {
    if (ta->keytable[i]) {
      *ppos = (CTablePos)&ta->table[i];
      break;
    }
  } while (i++ < ta->tablesize);
  if (!*ppos) SETERRQ(1,1,"TABLE error: No head");

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "TableGetNext"
/* TableGetNext() ********************************************
 *
 *  - iteration - CTablePos is always valid (points to a data)
 *  
 */
int TableGetNext(Table ta, CTablePos *rPosition, int *pkey, int *data)
{
  int index; 
  CTablePos pos;

  PetscFunctionBegin;
  pos = *rPosition;
  if (!pos) SETERRQ(1,1,"TABLE error: TABLE null position"); 
  *data = *pos; 
  if (!*data) SETERRQ(1,1,"TABLE error"); 
  index = pos - ta->table;
  *pkey = ta->keytable[index]; 
  if (!*pkey) SETERRQ(1,1,"TABLE error");  

  /* get next */
  do {
    pos++;  index++;
    if (index >= ta->tablesize) {
      pos = 0; /* end of list */
      break;
    }
    else if (ta->keytable[index]) {
      pos = ta->table + index;
      break;
    }
  } while (index < ta->tablesize);

  *rPosition = pos;

  PetscFunctionReturn(0);
}




