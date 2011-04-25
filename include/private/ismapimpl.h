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



struct _n_ISArrayHunk {
  PetscInt              refcnt;
  PetscInt              mask;
  PetscInt              length,maxlength;
  PetscInt             *i, *j;
  PetscScalar          *w;
  PetscCopyMode         mode;
  struct _n_ISArrayHunk  *parent;
};
typedef struct _n_ISArrayHunk *ISArrayHunk;

struct _n_ISArrayLink {
  ISArrayHunk            hunk;
  struct _n_ISArrayLink *next;
};
typedef struct _n_ISArrayLink *ISArrayLink;

struct _n_ISArray {
  ISArrayComponents mask;
  PetscInt          length;
  ISArrayLink       first,last;
};



extern PetscErrorCode ISArrayHunkCreate(PetscInt length, PetscInt mask, ISArrayHunk *_newhunk);
extern PetscErrorCode ISArrayHunkGetSubHunk(ISArrayHunk hunk, PetscInt length, PetscInt mask, ISArrayHunk *_subhunk);
extern PetscErrorCode ISArrayHunkDestroy(ISArrayHunk hunk);
extern PetscErrorCode ISArrayHunkAddData(ISArrayHunk hunk, PetscInt length, const PetscInt *i, const PetscScalar *w, const PetscInt *j);


extern PetscErrorCode ISArrayGetHunk(ISArray array, PetscInt length, ISArrayHunk *_hunk);
extern PetscErrorCode ISArrayAddHunk(ISArray array, ISArrayHunk hunk);
extern PetscErrorCode ISArrayAssemble(ISArray chain, PetscInt mask, PetscLayout layout, ISArray *_achain);

#endif
