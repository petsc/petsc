
/* table .h - Mark Adams */

#if !defined(__CTABLE)
#define __CTABLE

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

int intcomparc(const void *a, const void *b);

int       TableCreate( Table *ta, const int size );
int       TableCreateCopy( Table *rta, const Table intable );
int       TableDelete( Table ta );
int       TableGetCount( const Table ta );
int       TableIsEmpty( const Table ta );
int       TableAdd( Table ta, const int key, const int data );
int       TableFind( Table ta, const int key );
int       TableGetHeadPosition( Table ta, CTablePos * );
int       TableGetNext( Table ta, CTablePos *rPosition, int *pkey, int *data );
int       TableRemoveAll( Table ta );

#endif
