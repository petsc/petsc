#if !defined(__ISMAPIMPL_H)
#define __ISMAPIMPL_H

#include <petscmap.h>

/* --------------------------------------------------------------------------*/
struct _ISMappingOps {
  PetscErrorCode (*view)(ISMapping,PetscViewer);
  PetscErrorCode (*destroy)(ISMapping);
  PetscErrorCode (*setup)(ISMapping);
  PetscErrorCode (*assemblybegin)(ISMapping);
  PetscErrorCode (*assemblyend)(ISMapping);
  PetscErrorCode (*getsupportis)(ISMapping,IS*);
  PetscErrorCode (*getsupportsizelocal)(ISMapping, PetscInt*);
  PetscErrorCode (*getimageis)(ISMapping,IS*);
  PetscErrorCode (*getimagesizelocal)(ISMapping, PetscInt*);
  PetscErrorCode (*getmaximagesizelocal)(ISMapping, PetscInt*);
  PetscErrorCode (*map)(ISMapping,ISArray,ISArrayIndex,ISArray);
  PetscErrorCode (*maplocal)(ISMapping,ISArray,ISArrayIndex,ISArray);
  PetscErrorCode (*bin)(ISMapping,ISArray,ISArrayIndex,ISArray);
  PetscErrorCode (*binlocal)(ISMapping,ISArray,ISArrayIndex,ISArray);
  PetscErrorCode (*mapsplit)(ISMapping,ISArray,ISArrayIndex,ISArray*);
  PetscErrorCode (*mapsplitlocal)(ISMapping,ISArray,ISArrayIndex,ISArray*);
  PetscErrorCode (*binsplit)(ISMapping,ISArray,ISArrayIndex,ISArray*);
  PetscErrorCode (*binsplitlocal)(ISMapping,ISArray,ISArrayIndex,ISArray*);
  PetscErrorCode (*invert)(ISMapping, ISMapping*);
  PetscErrorCode (*getoperator)(ISMapping, Mat*);
};

struct _p_ISMapping{
  PETSCHEADER(struct _ISMappingOps);
  PetscLayout             xlayout,ylayout;
  PetscBool               setup;
  PetscBool               assembled;
  void                   *data;
};

extern PetscErrorCode ISMappingSetUp_ISMapping(ISMapping map);


#if defined(PETSC_USE_DEBUG)
#define ISMappingCheckMethod(map,method,name)                                                                  \
do {                                                                                                    \
  if(!(method)) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "ISMapping doesn't implement %s", (name)); \
} while(0)
#define ISMappingCheckType(map,maptype,arg)                             \
  do {                                                                   \
    PetscBool _9_sametype;                                              \
    PetscErrorCode _9_ierr;                                             \
    PetscValidHeaderSpecific((map), IS_MAPPING_CLASSID, 1);             \
    _9_ierr = PetscTypeCompare((PetscObject)(map),(maptype),&_9_sametype); CHKERRQ(_9_ierr); \
    if(!_9_sametype) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Expected ISMapping of type %s", (maptype)); \
  } while(0)
#else
#define ISMappingCheckMethod(map,method,name) do {} while(0)
#define ISMappingCheckType(map,maptype,arg) do {} while(0)
#endif

#define ISMappingCheckAssembled(map,needassembled,arg)                                            \
  do {                                                                                            \
    if(!((map)->assembled) && (needassembled)) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, \
                                  "ISMapping not assembled");                                     \
    if(((map)->assembled) && !(needassembled)) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, \
                                  "ISMapping already assembled");                                 \
  } while(0)

#define ISMappingCheckSetup(map,needsetup,arg)                                            \
  do {                                                                                    \
    if(!((map)->setup) && (needsetup)) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, \
                                  "ISMapping not setup");                                 \
    if(((map)->setup) && !(needsetup)) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, \
                                  "ISMapping already setup");                             \
  } while(0)

extern PetscErrorCode PetscCheckIntArrayRange(PetscInt len, const PetscInt idx[],  PetscInt imin, PetscInt imax, PetscBool outOfBoundsError, PetscBool *flag);
extern PetscErrorCode PetscCheckISRange(IS is, PetscInt imin, PetscInt imax, PetscBool outOfBoundsError, PetscBool *flag);


/*
 Increment ii by the number of unique elements of segment a[0,i-1] of a SORTED array a.
 */
#define PetscIntArrayCountUnique(a, i, ii)  \
{                                           \
  if(i) {                                   \
    PetscInt k = 0;                         \
    ++(ii);                                 \
    while(++k < (i))                        \
      if ((a)[k] != (a)[k-1]) {             \
        ++ii;                               \
      }                                     \
  }                                         \
}

/*
 Copy unique elements of segment a[0,i-1] of a SORTED array a, to aa[ii0,ii1-1]:
 i is an input, and ii is an input (with value ii0) and output (ii1), counting 
 the number of unique elements copied.
 */
#define PetscIntArrayCopyUnique(a, i, aa, ii)\
{                               \
  if(i) {                       \
    PetscInt k = 0;             \
    (aa)[(ii)] = (a)[k];        \
    ++(ii);                     \
    while (++k < (i))           \
      if ((a)[k] != (a)[k-1]) { \
        (aa)[(ii)] = (a)[k];    \
        ++(ii);                 \
      }                         \
  }                             \
}

/*
 Copy unique elements of segment a[0,i-1] of a SORTED array a, to aa[ii0,ii1-1]:
 i is an input, and ii is an input (with value ii0) and output (ii1), counting 
 the number of unique elements copied.  For each copied a, copy the corresponding
 b to bb.
 */
#define PetscIntArrayCopyUniqueWithScalar(a, b, i, aa, bb, ii)     \
{                               \
  if(i) {                       \
    PetscInt k = 0;             \
    (aa)[(ii)] = (a)[k];        \
    ++(ii);                     \
    while (++k < (i))           \
      if ((a)[k] != (a)[k-1]) { \
        (aa)[(ii)] = (a)[k];    \
        (bb)[(ii)] = (b)[k];    \
        ++(ii);                 \
      }                         \
  }                             \
}
#endif

struct _n_ISArrayHunk {
  PetscInt              refcnt;
  PetscInt              mask;
  PetscInt              length,maxlength;
  PetscInt             *i, *j;
  PetscScalar          *w;
  PetscCopyMode         mode;
  struct _n_ISArrayHunk  *next;
  struct _n_ISArrayHunk  *parent;
};

typedef struct _n_ISArrayHunk *ISArrayHunk;

struct _n_ISArray {
  ISArrayComponents mask;
  PetscInt          length;
  ISArrayHunk       buffer,first,last;
};



extern PetscErrorCode ISArrayHunkCreate(PetscInt length, PetscInt mask, ISArrayHunk *_newhunk);
/* BEGIN: these methods do reference management. */
extern PetscErrorCode ISArrayHunkGetSubHunk(ISArrayHunk hunk, PetscInt length, PetscInt mask, ISArrayHunk *_subhunk);
extern PetscErrorCode ISArrayHunkSetNext(ISArrayHunk hunk, ISArrayHunk nexthunk);
extern PetscErrorCode ISArrayHunkDestroy(ISArrayHunk hunk);
/* END: these methods do reference management. */
extern PetscErrorCode ISArrayHunkDuplicate(ISArrayHunk hunk, ISArrayHunk *dhunk);
extern PetscErrorCode ISArrayHunkAddData(ISArrayHunk hunk, PetscInt length, const PetscInt *i, const PetscScalar *w, const PetscInt *j);
extern PetscErrorCode ISArrayHunkMergeHunks(ISArrayHunk hunk, PetscInt mask, ISArrayHunk *_merged);


extern PetscErrorCode ISArraySetBuffer(ISArray array, ISArrayHunk buffer);
extern PetscErrorCode ISArrayGetHunk(ISArray chain, PetscInt length, ISArrayHunk *_hunk);
extern PetscErrorCode ISArrayAddHunk(ISArray array, ISArrayHunk hunk);
extern PetscErrorCode ISArrayAssemble(ISArray chain, PetscInt mask, PetscLayout layout, ISArray *_achain);

