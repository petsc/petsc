/* $Id: isimpl.h,v 1.5 1995/08/07 21:57:25 bsmith Exp bsmith $ */

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
  int  (*getsize)(IS,int*),
       (*getlocalsize)(IS,int*),
       (*getindices)(IS,int**),
       (*restoreindices)(IS,int**),
       (*invertpermutation)(IS,IS*);
};

struct _IS {
  PETSCHEADER
  struct       _ISOps ops;
  int          isperm;          /* if is a permutation */
  int          max,min;         /* range of possible values */
  void         *data;
};

#endif
