/* $Id: isimpl.h,v 1.21 2000/02/02 20:08:31 bsmith Exp balay $ */

/*
    Index sets for scatter-gather type operations in vectors
and matrices. 

*/

#if !defined(_IS_H)
#define _IS_H
#include "petscis.h"

struct _ISOps {
  int  (*getsize)(IS,int*),
       (*getlocalsize)(IS,int*),
       (*getindices)(IS,int**),
       (*restoreindices)(IS,int**),
       (*invertpermutation)(IS,int,IS*),
       (*sortindices)(IS),
       (*sorted)(IS,PetscTruth *),
       (*duplicate)(IS,IS *),
       (*destroy)(IS),
       (*view)(IS,Viewer),
       (*identity)(IS,PetscTruth*);
};

struct _p_IS {
  PETSCHEADER(struct _ISOps)
  int          isperm;          /* if is a permutation */
  int          max,min;         /* range of possible values */
  void         *data;
  int          isidentity;
};

struct _p_ISLocalToGlobalMapping{
  PETSCHEADER(int)
  int n;                  /* number of local indices */
  int *indices;           /* global index of each local index */
  int globalstart;        /* first global referenced in indices */
  int globalend;          /* last + 1 global referenced in indices */
  int *globals;           /* local index for each global index between start and end */
};

#endif
