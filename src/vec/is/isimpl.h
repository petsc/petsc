/* $Id: isimpl.h,v 1.4 1995/06/07 16:35:31 bsmith Exp bsmith $ */

/*
    Index sets for scatter-gather type operations in vectors
and matrices. 

   Eventually ther may be operations like union, difference etc.
for now we define only the shell and what we absolutely need.
*/

#if !defined(_INDEX)
#define _INDEX
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

#endif
