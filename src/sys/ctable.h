/* $Id: petsc.h,v 1.235 1998/12/21 01:06:07 bsmith Exp $ */
/* Contributed by - Mark Adams */

#if !defined(__CTABLE_H)
#define __CTABLE_H

typedef int* CTablePos;  
typedef struct TABLE_TAG 
{
  int *keytable;
  int *table;
  int count     ;
  int tablesize ;
  int head;
}*Table,Table_struct;

#define HASHT(ta,x) ((53*x)%ta->tablesize)

extern int intcomparc(const void *a, const void *b);

extern int TableCreate( Table *ta, const int size );
extern int TableCreateCopy( Table *rta, const Table intable );
extern int TableDelete( Table ta );
extern int TableGetCount( const Table ta );
extern int TableIsEmpty( const Table ta );
extern int TableAdd( Table ta, const int key, const int data );
extern int TableFind( Table ta, const int key );
extern int TableGetHeadPosition( Table ta, CTablePos * );
extern int TableGetNext( Table ta, CTablePos *rPosition, int *pkey, int *data );
extern int TableRemoveAll( Table ta );

#endif
