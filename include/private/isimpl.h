/*
    Index sets for scatter-gather type operations in vectors
and matrices. 

*/

#if !defined(_IS_H)
#define _IS_H

#include "petscis.h"

struct _ISOps {
  PetscErrorCode (*getsize)(IS,PetscInt*);
  PetscErrorCode (*getlocalsize)(IS,PetscInt*);
  PetscErrorCode (*getindices)(IS,const PetscInt*[]);
  PetscErrorCode (*restoreindices)(IS,const PetscInt*[]);
  PetscErrorCode (*invertpermutation)(IS,PetscInt,IS*);
  PetscErrorCode (*sortindices)(IS);
  PetscErrorCode (*sorted)(IS,PetscTruth *);
  PetscErrorCode (*duplicate)(IS,IS *);
  PetscErrorCode (*destroy)(IS);
  PetscErrorCode (*view)(IS,PetscViewer);
  PetscErrorCode (*identity)(IS,PetscTruth*);
  PetscErrorCode (*copy)(IS,IS);
};

struct _p_IS {
  PETSCHEADER(struct _ISOps);
  PetscTruth   isperm;          /* if is a permutation */
  PetscInt     max,min;         /* range of possible values */
  void         *data;
  PetscTruth   isidentity;
};


#endif
