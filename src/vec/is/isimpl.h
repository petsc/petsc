/* $Id: isimpl.h,v 1.8 1996/02/16 19:31:01 balay Exp bsmith $ */

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
       (*invertpermutation)(IS,IS*),
       (*sortindices)(IS),
       (*sorted)(IS,PetscTruth *);
};

struct _IS {
  PETSCHEADER
  struct       _ISOps ops;
  int          isperm;          /* if is a permutation */
  int          max,min;         /* range of possible values */
  void         *data;
  int          isidentity;
};

#endif
