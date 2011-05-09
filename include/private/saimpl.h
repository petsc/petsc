#if !defined(__SAIMPL_H)
#define __SAIMPL_H

#include <petscsa.h>

/* --------------------------------------------------------------------------*/
struct _SAMappingOps {
  PetscErrorCode (*view)(SAMapping,PetscViewer);
  PetscErrorCode (*destroy)(SAMapping);
  PetscErrorCode (*setup)(SAMapping);
  PetscErrorCode (*assemblybegin)(SAMapping);
  PetscErrorCode (*assemblyend)(SAMapping);
  PetscErrorCode (*getsupport)(SAMapping, PetscInt*, PetscInt **);
  PetscErrorCode (*getsupportis)(SAMapping,IS*);
  PetscErrorCode (*getsupportsa)(SAMapping,SA*);
  PetscErrorCode (*getimage)(SAMapping, PetscInt*, PetscInt **);
  PetscErrorCode (*getimageis)(SAMapping,IS*);
  PetscErrorCode (*getimagesa)(SAMapping,SA*);
  PetscErrorCode (*getmaximagesize)(SAMapping, PetscInt*);
  PetscErrorCode (*map)(SAMapping,SA,SAIndex,SA);
  PetscErrorCode (*maplocal)(SAMapping,SA,SAIndex,SA);
  PetscErrorCode (*bin)(SAMapping,SA,SAIndex,SA);
  PetscErrorCode (*binlocal)(SAMapping,SA,SAIndex,SA);
  PetscErrorCode (*mapsplit)(SAMapping,SA,SAIndex,SA*);
  PetscErrorCode (*mapsplitlocal)(SAMapping,SA,SAIndex,SA*);
  PetscErrorCode (*binsplit)(SAMapping,SA,SAIndex,SA*);
  PetscErrorCode (*binsplitlocal)(SAMapping,SA,SAIndex,SA*);
  PetscErrorCode (*invert)(SAMapping, SAMapping*);
  PetscErrorCode (*getoperator)(SAMapping, Mat*);
};

struct _p_SAMapping{
  PETSCHEADER(struct _SAMappingOps);
  PetscLayout             xlayout,ylayout;
  PetscBool               setup;
  PetscBool               assembled;
  void                   *data;
};

extern PetscErrorCode SAMappingSetUp_SAMapping(SAMapping map);


#if defined(PETSC_USE_DEBUG)
#define SAMappingCheckMethod(map,method,name)                                                                  \
do {                                                                                                    \
  if(!(method)) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "SAMapping doesn't implement %s", (name)); \
} while(0)
#define SAMappingCheckType(map,maptype,arg)                             \
  do {                                                                   \
    PetscBool _9_sametype;                                              \
    PetscErrorCode _9_ierr;                                             \
    PetscValidHeaderSpecific((map), SA_MAPPING_CLASSID, 1);             \
    _9_ierr = PetscTypeCompare((PetscObject)(map),(maptype),&_9_sametype); CHKERRQ(_9_ierr); \
    if(!_9_sametype) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Expected SAMapping of type %s", (maptype)); \
  } while(0)
#else
#define SAMappingCheckMethod(map,method,name) do {} while(0)
#define SAMappingCheckType(map,maptype,arg) do {} while(0)
#endif

#define SAMappingCheckAssembled(map,needassembled,arg)                                            \
  do {                                                                                            \
    if(!((map)->assembled) && (needassembled)) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, \
                                  "SAMapping not assembled");                                     \
    if(((map)->assembled) && !(needassembled)) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, \
                                  "SAMapping already assembled");                                 \
  } while(0)

#define SAMappingCheckSetup(map,needsetup,arg)                                            \
  do {                                                                                    \
    if(!((map)->setup) && (needsetup)) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, \
                                  "SAMapping not setup");                                 \
    if(((map)->setup) && !(needsetup)) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, \
                                  "SAMapping already setup");                             \
  } while(0)

extern PetscErrorCode PetscCheckIntArrayRange(PetscInt len, const PetscInt idx[],  PetscInt imin, PetscInt imax, PetscBool outOfBoundsError, PetscBool *flag);
extern PetscErrorCode PetscCheckISRange(IS is, PetscInt imin, PetscInt imax, PetscBool outOfBoundsError, PetscBool *flag);



struct _n_SAHunk {
  PetscInt              refcnt;
  PetscInt              mask;
  PetscInt              length,maxlength;
  PetscInt             *i, *j;
  PetscScalar          *w;
  PetscCopyMode         mode;
  struct _n_SAHunk  *parent;
};
typedef struct _n_SAHunk *SAHunk;

struct _n_SALink {
  SAHunk            hunk;
  struct _n_SALink *next;
};
typedef struct _n_SALink *SALink;

struct _n_SA {
  SAComponents mask;
  PetscInt          length;
  SALink       first,last;
};



extern PetscErrorCode SAHunkCreate(PetscInt length, PetscInt mask, SAHunk *_newhunk);
extern PetscErrorCode SAHunkGetSubHunk(SAHunk hunk, PetscInt start, PetscInt maxlength, PetscInt length, PetscInt mask, SAHunk *_subhunk);
extern PetscErrorCode SAHunkDestroy(SAHunk *_hunk);
extern PetscErrorCode SAHunkAddData(SAHunk hunk, PetscInt length, const PetscInt *i, const PetscScalar *w, const PetscInt *j);


extern PetscErrorCode SAGetHunk(SA array, PetscInt length, SAHunk *_hunk);
extern PetscErrorCode SAAddHunk(SA array, SAHunk hunk);
extern PetscErrorCode SAMerge(SA chain, SA *mchain);
extern PetscErrorCode SAAssemble(SA chain, PetscInt mask, PetscLayout layout, SA achain);
extern PetscErrorCode SASplit(SA arr, PetscInt count, const PetscInt *lengths, PetscInt mask, SA *arrs);

#endif
