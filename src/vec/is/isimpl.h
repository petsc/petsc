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
  PetscErrorCode (*getindices)(IS,PetscInt**);
  PetscErrorCode (*restoreindices)(IS,PetscInt**);
  PetscErrorCode (*invertpermutation)(IS,PetscInt,IS*);
  PetscErrorCode (*sortindices)(IS);
  PetscErrorCode (*sorted)(IS,PetscTruth *);
  PetscErrorCode (*duplicate)(IS,IS *);
  PetscErrorCode (*destroy)(IS);
  PetscErrorCode (*view)(IS,PetscViewer);
  PetscErrorCode (*identity)(IS,PetscTruth*);
};

#if defined(__cplusplus)
class ISOps {
  public:
    int getsize(PetscInt*) {return 0;};
};
#endif

struct _p_IS {
  PETSCHEADER(struct _ISOps)
#if defined(__cplusplus)
  ISOps        *cops;
#endif
  PetscTruth   isperm;          /* if is a permutation */
  PetscInt     max,min;         /* range of possible values */
  void         *data;
  PetscTruth   isidentity;
};


#endif
