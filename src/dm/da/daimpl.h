
/*
    Index sets for scatter-gather type operations in vectors
and matrices. 

   Eventually ther may be operations like union, difference etc.
for now we define only the shell and what we absolutely need.
*/

#if !defined(_INDEX)
#define _INDEX
#include "ptscimpl.h"
#include "is.h"

struct _ISOps {
  int  (*getsize)(IS,int*),(*getlocalsize)(IS,int*);
  int  (*getindices)(IS,int**);
  int  (*restoreindices)(IS,int**);
  int  (*invertpermutation)(IS,IS*);
};

struct _IS {
  PETSCHEADER
  struct        _ISOps *ops;
  int           isperm; /* if is a permutation */
  int           max,min; /* range of possible values */
  void          *data;
};



#define ISGENERALSEQUENTIAL 0
#define ISSTRIDESEQUENTIAL  2
#define ISGENERALPARALLEL   1
#endif
