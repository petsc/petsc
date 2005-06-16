/*
    Provides access to system related and general utility routines.
*/
#if !defined(__PETSCSYS_H)
#define __PETSCSYS_H
#include "petsc.h"
PETSC_EXTERN_CXX_BEGIN

EXTERN PetscErrorCode PETSC_DLLEXPORT PetscGetArchType(char[],size_t);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscGetHostName(char[],size_t);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscGetUserName(char[],size_t);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscGetProgramName(char[],size_t);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscSetProgramName(const char[]);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscGetDate(char[],size_t);

EXTERN PetscErrorCode PETSC_DLLEXPORT PetscSortInt(PetscInt,PetscInt[]);
EXTERN PetscErrorCode PETSC_DLLEXPORT  PetscSortIntWithPermutation(PetscInt,const PetscInt[],PetscInt[]);
EXTERN PetscErrorCode PETSC_DLLEXPORT  PetscSortStrWithPermutation(PetscInt,const char*[],PetscInt[]);
EXTERN PetscErrorCode PETSC_DLLEXPORT  PetscSortIntWithArray(PetscInt,PetscInt[],PetscInt[]);
EXTERN PetscErrorCode PETSC_DLLEXPORT  PetscSortIntWithScalarArray(PetscInt,PetscInt[],PetscScalar[]);
EXTERN PetscErrorCode PETSC_DLLEXPORT  PetscSortReal(PetscInt,PetscReal[]);
EXTERN PetscErrorCode PETSC_DLLEXPORT  PetscSortRealWithPermutation(PetscInt,const PetscReal[],PetscInt[]);

EXTERN PetscErrorCode PETSC_DLLEXPORT  PetscSetDisplay(void);
EXTERN PetscErrorCode PETSC_DLLEXPORT  PetscGetDisplay(char[],size_t);

extern PetscCookie PETSC_DLLEXPORT PETSC_RANDOM_COOKIE;

typedef enum { RANDOM_DEFAULT,RANDOM_DEFAULT_REAL,
               RANDOM_DEFAULT_IMAGINARY } PetscRandomType;

/*S
     PetscRandom - Abstract PETSc object that manages generating random numbers

   Level: intermediate

  Concepts: random numbers

.seealso:  PetscRandomCreate(), PetscRandomGetValue()
S*/
typedef struct _p_PetscRandom*   PetscRandom;

EXTERN PetscErrorCode PETSC_DLLEXPORT PetscRandomCreate(MPI_Comm,PetscRandomType,PetscRandom*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscRandomGetValue(PetscRandom,PetscScalar*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscRandomGetInterval(PetscRandom,PetscScalar*,PetscScalar*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscRandomSetInterval(PetscRandom,PetscScalar,PetscScalar);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscRandomSetSeed(PetscRandom,unsigned long);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscRandomGetSeed(PetscRandom,unsigned long *);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscRandomSeed(PetscRandom);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscRandomDestroy(PetscRandom);

EXTERN PetscErrorCode PETSC_DLLEXPORT PetscGetFullPath(const char[],char[],size_t);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscGetRelativePath(const char[],char[],size_t);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscGetWorkingDirectory(char[],size_t);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscGetRealPath(char[],char[]);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscGetHomeDirectory(char[],size_t);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscTestFile(const char[],char,PetscTruth*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscTestDirectory(const char[],char,PetscTruth*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscBinaryRead(int,void*,PetscInt,PetscDataType);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscSynchronizedBinaryRead(MPI_Comm,int,void*,PetscInt,PetscDataType);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscSynchronizedBinaryWrite(MPI_Comm,int,void*,PetscInt,PetscDataType,PetscTruth);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscBinaryWrite(int,void*,PetscInt,PetscDataType,PetscTruth);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscBinaryOpen(const char[],int,int *);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscBinaryClose(int);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscSharedTmp(MPI_Comm,PetscTruth *);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscSharedWorkingDirectory(MPI_Comm,PetscTruth *);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscGetTmp(MPI_Comm,char *,size_t);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscFileRetrieve(MPI_Comm,const char *,char *,size_t,PetscTruth*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscLs(MPI_Comm,const char[],char*,size_t,PetscTruth*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDLLibraryCCAAppend(MPI_Comm,PetscDLLibraryList*,const char[]);

/*
   In binary files variables are stored using the following lengths,
  regardless of how they are stored in memory on any one particular
  machine. Use these rather then sizeof() in computing sizes for 
  PetscBinarySeek().
*/
#define PETSC_BINARY_INT_SIZE    (32/8)
#define PETSC_BINARY_FLOAT_SIZE  (32/8)
#define PETSC_BINARY_CHAR_SIZE    (8/8)
#define PETSC_BINARY_SHORT_SIZE  (16/8)
#define PETSC_BINARY_DOUBLE_SIZE (64/8)
#define PETSC_BINARY_SCALAR_SIZE sizeof(PetscScalar)

/*E
  PetscBinarySeekType - argument to PetscBinarySeek()

  Level: advanced

.seealso: PetscBinarySeek(), PetscSynchronizedBinarySeek()
E*/
typedef enum {PETSC_BINARY_SEEK_SET = 0,PETSC_BINARY_SEEK_CUR = 1,PETSC_BINARY_SEEK_END = 2} PetscBinarySeekType;
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscBinarySeek(int,off_t,PetscBinarySeekType,off_t*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscSynchronizedBinarySeek(MPI_Comm,int,off_t,PetscBinarySeekType,off_t*);

EXTERN PetscErrorCode PETSC_DLLEXPORT PetscSetDebugger(const char[],PetscTruth);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscSetDefaultDebugger(void);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscSetDebuggerFromString(char*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscAttachDebugger(void);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscStopForDebugger(void);

EXTERN PetscErrorCode PETSC_DLLEXPORT PetscGatherNumberOfMessages(MPI_Comm,PetscMPIInt*,PetscMPIInt*,PetscMPIInt*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscGatherMessageLengths(MPI_Comm,PetscMPIInt,PetscMPIInt,PetscMPIInt*,PetscMPIInt**,PetscMPIInt**);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscGatherMessageLengths2(MPI_Comm,PetscMPIInt,PetscMPIInt,PetscMPIInt*,PetscMPIInt*,PetscMPIInt**,PetscMPIInt**,PetscMPIInt**);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscPostIrecvInt(MPI_Comm,PetscMPIInt,PetscMPIInt,PetscMPIInt*,PetscMPIInt*,PetscInt***,MPI_Request**);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscPostIrecvScalar(MPI_Comm,PetscMPIInt,PetscMPIInt,PetscMPIInt*,PetscMPIInt*,PetscScalar***,MPI_Request**);

EXTERN PetscErrorCode PETSC_DLLEXPORT PetscSSEIsEnabled(MPI_Comm,PetscTruth *,PetscTruth *);

/* Parallel communication routines */
/*E
  InsertMode - Whether entries are inserted or added into vectors or matrices

  Level: beginner

.seealso: VecSetValues(), MatSetValues(), VecSetValue(), VecSetValuesBlocked(),
          VecSetValuesLocal(), VecSetValuesBlockedLocal(), MatSetValuesBlocked(),
          MatSetValuesBlockedLocal(), MatSetValuesLocal(), VecScatterBegin(), VecScatterEnd()
E*/
typedef enum {NOT_SET_VALUES, INSERT_VALUES, ADD_VALUES, MAX_VALUES} InsertMode;

/*MC
    INSERT_VALUES - Put a value into a vector or matrix, overwrites any previous value

    Level: beginner

.seealso: InsertMode, VecSetValues(), MatSetValues(), VecSetValue(), VecSetValuesBlocked(),
          VecSetValuesLocal(), VecSetValuesBlockedLocal(), MatSetValuesBlocked(), ADD_VALUES, INSERT_VALUES,
          MatSetValuesBlockedLocal(), MatSetValuesLocal(), VecScatterBegin(), VecScatterEnd()

M*/

/*MC
    ADD_VALUES - Adds a value into a vector or matrix, if there previously was no value, just puts the
                value into that location

    Level: beginner

.seealso: InsertMode, VecSetValues(), MatSetValues(), VecSetValue(), VecSetValuesBlocked(),
          VecSetValuesLocal(), VecSetValuesBlockedLocal(), MatSetValuesBlocked(), ADD_VALUES, INSERT_VALUES,
          MatSetValuesBlockedLocal(), MatSetValuesLocal(), VecScatterBegin(), VecScatterEnd()

M*/

/*MC
    MAX_VALUES - Puts the maximum of the scattered/gathered value and the current value into each location

    Level: beginner

.seealso: InsertMode, VecScatterBegin(), VecScatterEnd(), ADD_VALUES, INSERT_VALUES

M*/

/*E
  ScatterMode - Determines the direction of a scatter

  Level: beginner

.seealso: VecScatter, VecScatterBegin(), VecScatterEnd()
E*/
typedef enum {SCATTER_FORWARD=0, SCATTER_REVERSE=1, SCATTER_FORWARD_LOCAL=2, SCATTER_REVERSE_LOCAL=3, SCATTER_LOCAL=2} ScatterMode;

/*MC
    SCATTER_FORWARD - Scatters the values as dictated by the VecScatterCreate() call

    Level: beginner

.seealso: VecScatter, ScatterMode, VecScatterCreate(), VecScatterBegin(), VecScatterEnd(), SCATTER_REVERSE, SCATTER_FORWARD_LOCAL,
          SCATTER_REVERSE_LOCAL

M*/

/*MC
    SCATTER_REVERSE - Moves the values in the opposite direction then the directions indicated in
         in the VecScatterCreate()

    Level: beginner

.seealso: VecScatter, ScatterMode, VecScatterCreate(), VecScatterBegin(), VecScatterEnd(), SCATTER_FORWARD, SCATTER_FORWARD_LOCAL,
          SCATTER_REVERSE_LOCAL

M*/

/*MC
    SCATTER_FORWARD_LOCAL - Scatters the values as dictated by the VecScatterCreate() call except NO parallel communication
       is done. Any variables that have be moved between processes are ignored

    Level: developer

.seealso: VecScatter, ScatterMode, VecScatterCreate(), VecScatterBegin(), VecScatterEnd(), SCATTER_REVERSE, SCATTER_FORWARD,
          SCATTER_REVERSE_LOCAL

M*/

/*MC
    SCATTER_REVERSE_LOCAL - Moves the values in the opposite direction then the directions indicated in
         in the VecScatterCreate()  except NO parallel communication
       is done. Any variables that have be moved between processes are ignored

    Level: developer

.seealso: VecScatter, ScatterMode, VecScatterCreate(), VecScatterBegin(), VecScatterEnd(), SCATTER_FORWARD, SCATTER_FORWARD_LOCAL,
          SCATTER_REVERSE

M*/

/* 
  Create and initialize a linked list 
  Input Parameters:
    idx_start - starting index of the list
    lnk_max   - max value of lnk indicating the end of the list
    nlnk      - max length of the list
  Output Parameters:
    lnk       - list initialized
    bt        - PetscBT (bitarray) with all bits set to false
*/
#define PetscLLCreate(idx_start,lnk_max,nlnk,lnk,bt) \
  (PetscMalloc(nlnk*sizeof(PetscInt),&lnk) || PetscBTCreate(nlnk,bt) || PetscBTMemzero(nlnk,bt) || (lnk[idx_start] = lnk_max,0))

/*
  Add a index set into a sorted linked list
  Input Parameters:
    nidx      - number of input indices
    indices   - interger array
    idx_start - starting index of the list
    lnk       - linked list(an integer array) that is created
    bt        - PetscBT (bitarray), bt[idx]=true marks idx is in lnk
  output Parameters:
    nlnk      - number of newly added indices
    lnk       - the sorted(increasing order) linked list containing new and non-redundate entries from indices
    bt        - updated PetscBT (bitarray) 
*/
#define PetscLLAdd(nidx,indices,idx_start,nlnk,lnk,bt) 0;\
{\
  PetscInt _k,_entry,_location,_lnkdata;\
  nlnk     = 0;\
  _lnkdata = idx_start;\
  for (_k=0; _k<nidx; _k++){\
    _entry = indices[_k];\
    if (!PetscBTLookupSet(bt,_entry)){  /* new entry */\
      /* search for insertion location */\
      /* start from the beginning if _entry < previous _entry */\
      if (_k && _entry < _lnkdata) _lnkdata  = idx_start;\
      do {\
        _location = _lnkdata;\
        _lnkdata  = lnk[_location];\
      } while (_entry > _lnkdata);\
      /* insertion location is found, add entry into lnk */\
      lnk[_location] = _entry;\
      lnk[_entry]    = _lnkdata;\
      nlnk++;\
      _lnkdata = _entry; /* next search starts from here if next_entry > _entry */\
    }\
  }\
}

/*
  Add a SORTED index set into a sorted linked list
  Input Parameters:
    nidx      - number of input indices
    indices   - sorted interger array 
    idx_start - starting index of the list
    lnk       - linked list(an integer array) that is created
    bt        - PetscBT (bitarray), bt[idx]=true marks idx is in lnk
  output Parameters:
    nlnk      - number of newly added indices
    lnk       - the sorted(increasing order) linked list containing new and non-redundate entries from indices
    bt        - updated PetscBT (bitarray) 
*/
#define PetscLLAddSorted(nidx,indices,idx_start,nlnk,lnk,bt) 0;\
{\
  PetscInt _k,_entry,_location,_lnkdata;\
  nlnk      = 0;\
  _lnkdata  = idx_start;\
  for (_k=0; _k<nidx; _k++){\
    _entry = indices[_k];\
    if (!PetscBTLookupSet(bt,_entry)){  /* new entry */\
      /* search for insertion location */\
      do {\
        _location = _lnkdata;\
        _lnkdata  = lnk[_location];\
      } while (_entry > _lnkdata);\
      /* insertion location is found, add entry into lnk */\
      lnk[_location] = _entry;\
      lnk[_entry]    = _lnkdata;\
      nlnk++;\
      _lnkdata = _entry; /* next search starts from here */\
    }\
  }\
}

/*
  Add a SORTED index set into a sorted linked list used for LUFactorSymbolic()
  Same as PetscLLAddSorted() with an additional operation:
       count the number of input indices that are no larger than 'diag'
  Input Parameters:
    nidx      - number of input indices
    indices   - sorted interger array 
    idx_start - starting index of the list
    lnk       - linked list(an integer array) that is created
    bt        - PetscBT (bitarray), bt[idx]=true marks idx is in lnk
    diag      - index of the active row in LUFactorSymbolic
  output Parameters:
    nlnk      - number of newly added indices
    lnk       - the sorted(increasing order) linked list containing new and non-redundate entries from indices
    bt        - updated PetscBT (bitarray) 
    nzbd      - number of input indices that are no larger than 'diag'
*/
#define PetscLLAddSortedLU(nidx,indices,idx_start,nlnk,lnk,bt,diag,nzbd) 0;\
{\
  PetscInt _k,_entry,_location,_lnkdata;\
  nlnk     = 0;\
  _lnkdata = idx_start;\
  nzbd     = 0;\
  for (_k=0; _k<nidx; _k++){\
    _entry = indices[_k];\
    if (_entry <= diag) nzbd++;\
    if (!PetscBTLookupSet(bt,_entry)){  /* new entry */\
      /* search for insertion location */\
      do {\
        _location = _lnkdata;\
        _lnkdata  = lnk[_location];\
      } while (_entry > _lnkdata);\
      /* insertion location is found, add entry into lnk */\
      lnk[_location] = _entry;\
      lnk[_entry]    = _lnkdata;\
      nlnk++;\
      _lnkdata = _entry; /* next search starts from here */\
    }\
  }\
}

/*
  Copy data on the list into an array, then initialize the list 
  Input Parameters:
    idx_start - starting index of the list 
    lnk_max   - max value of lnk indicating the end of the list 
    nlnk      - number of data on the list to be copied
    lnk       - linked list
    bt        - PetscBT (bitarray), bt[idx]=true marks idx is in lnk
  output Parameters:
    indices   - array that contains the copied data
    lnk       - linked list that is cleaned and initialize
    bt        - PetscBT (bitarray) with all bits set to false
*/
#define PetscLLClean(idx_start,lnk_max,nlnk,lnk,indices,bt) 0;\
{\
  PetscInt _j,_idx=idx_start;\
  for (_j=0; _j<nlnk; _j++){\
    _idx = lnk[_idx];\
    *(indices+_j) = _idx;\
    ierr = PetscBTClear(bt,_idx);CHKERRQ(ierr);\
  }\
  lnk[idx_start] = lnk_max;\
}
/*
  Free memories used by the list
*/
#define PetscLLDestroy(lnk,bt) (PetscFree(lnk) || PetscBTDestroy(bt))

/* Routines below are used for incomplete matrix factorization */
/* 
  Create and initialize a linked list and its levels
  Input Parameters:
    idx_start - starting index of the list
    lnk_max   - max value of lnk indicating the end of the list
    nlnk      - max length of the list
  Output Parameters:
    lnk       - list initialized
    lnk_lvl   - array of size nlnk for storing levels of lnk
    bt        - PetscBT (bitarray) with all bits set to false
*/
#define PetscIncompleteLLCreate(idx_start,lnk_max,nlnk,lnk,lnk_lvl,bt)\
  (PetscMalloc(2*nlnk*sizeof(PetscInt),&lnk) || PetscBTCreate(nlnk,bt) || PetscBTMemzero(nlnk,bt) || (lnk[idx_start] = lnk_max,lnk_lvl = lnk + nlnk,0))

/*
  Initialize a sorted linked list used for ILU and ICC
  Input Parameters:
    nidx      - number of input idx
    idx       - interger array used for storing column indices
    idx_start - starting index of the list
    perm      - indices of an IS
    lnk       - linked list(an integer array) that is created
    lnklvl    - levels of lnk
    bt        - PetscBT (bitarray), bt[idx]=true marks idx is in lnk
  output Parameters:
    nlnk     - number of newly added idx
    lnk      - the sorted(increasing order) linked list containing new and non-redundate entries from idx
    lnklvl   - levels of lnk
    bt       - updated PetscBT (bitarray) 
*/
#define PetscIncompleteLLInit(nidx,idx,idx_start,perm,nlnk,lnk,lnklvl,bt) 0;\
{\
  PetscInt _k,_entry,_location,_lnkdata;\
  nlnk     = 0;\
  _lnkdata = idx_start;\
  for (_k=0; _k<nidx; _k++){\
    _entry = perm[idx[_k]];\
    if (!PetscBTLookupSet(bt,_entry)){  /* new entry */\
      /* search for insertion location */\
      if (_k && _entry < _lnkdata) _lnkdata  = idx_start;\
      do {\
        _location = _lnkdata;\
        _lnkdata  = lnk[_location];\
      } while (_entry > _lnkdata);\
      /* insertion location is found, add entry into lnk */\
      lnk[_location]  = _entry;\
      lnk[_entry]     = _lnkdata;\
      lnklvl[_entry] = 0;\
      nlnk++;\
      _lnkdata = _entry; /* next search starts from here if next_entry > _entry */\
    }\
  }\
}

/*
  Add a SORTED index set into a sorted linked list for ILU
  Input Parameters:
    nidx      - number of input indices
    idx       - sorted interger array used for storing column indices
    level     - level of fill, e.g., ICC(level)
    idxlvl    - level of idx 
    idx_start - starting index of the list
    lnk       - linked list(an integer array) that is created
    lnklvl    - levels of lnk
    bt        - PetscBT (bitarray), bt[idx]=true marks idx is in lnk
    prow      - the row number of idx
  output Parameters:
    nlnk     - number of newly added idx
    lnk      - the sorted(increasing order) linked list containing new and non-redundate entries from idx
    lnklvl   - levels of lnk
    bt       - updated PetscBT (bitarray) 

  Note: the level of factor(i,j) is set as lvl(i,j) = min{ lvl(i,j), lvl(i,prow)+lvl(prow,j)+1)
        where idx = non-zero columns of U(prow,prow+1:n-1), prow<i
*/
#define PetscILULLAddSorted(nidx,idx,level,idxlvl,idx_start,nlnk,lnk,lnklvl,bt,lnklvl_prow) 0;\
{\
  PetscInt _k,_entry,_location,_lnkdata,_incrlev,_lnklvl_prow=lnklvl[prow];\
  nlnk     = 0;\
  _lnkdata = idx_start;\
  for (_k=0; _k<nidx; _k++){\
    _incrlev = idxlvl[_k] + _lnklvl_prow + 1;\
    if (_incrlev > level) continue;\
    _entry = idx[_k];\
    if (!PetscBTLookupSet(bt,_entry)){  /* new entry */\
      /* search for insertion location */\
      do {\
        _location = _lnkdata;\
        _lnkdata  = lnk[_location];\
      } while (_entry > _lnkdata);\
      /* insertion location is found, add entry into lnk */\
      lnk[_location]  = _entry;\
      lnk[_entry]     = _lnkdata;\
      lnklvl[_entry] = _incrlev;\
      nlnk++;\
      _lnkdata = _entry; /* next search starts from here if next_entry > _entry */\
    } else { /* existing entry: update lnklvl */\
      if (lnklvl[_entry] > _incrlev) lnklvl[_entry] = _incrlev;\
    }\
  }\
}

/*
  Add a index set into a sorted linked list
  Input Parameters:
    nidx      - number of input idx
    idx   - interger array used for storing column indices
    level     - level of fill, e.g., ICC(level)
    idxlvl - level of idx 
    idx_start - starting index of the list
    lnk       - linked list(an integer array) that is created
    lnklvl   - levels of lnk
    bt        - PetscBT (bitarray), bt[idx]=true marks idx is in lnk
  output Parameters:
    nlnk      - number of newly added idx
    lnk       - the sorted(increasing order) linked list containing new and non-redundate entries from idx
    lnklvl   - levels of lnk
    bt        - updated PetscBT (bitarray) 
*/
#define PetscIncompleteLLAdd(nidx,idx,level,idxlvl,idx_start,nlnk,lnk,lnklvl,bt) 0;\
{\
  PetscInt _k,_entry,_location,_lnkdata,_incrlev;\
  nlnk     = 0;\
  _lnkdata = idx_start;\
  for (_k=0; _k<nidx; _k++){\
    _incrlev = idxlvl[_k] + 1;\
    if (_incrlev > level) continue;\
    _entry = idx[_k];\
    if (!PetscBTLookupSet(bt,_entry)){  /* new entry */\
      /* search for insertion location */\
      if (_k && _entry < _lnkdata) _lnkdata  = idx_start;\
      do {\
        _location = _lnkdata;\
        _lnkdata  = lnk[_location];\
      } while (_entry > _lnkdata);\
      /* insertion location is found, add entry into lnk */\
      lnk[_location]  = _entry;\
      lnk[_entry]     = _lnkdata;\
      lnklvl[_entry] = _incrlev;\
      nlnk++;\
      _lnkdata = _entry; /* next search starts from here if next_entry > _entry */\
    } else { /* existing entry: update lnklvl */\
      if (lnklvl[_entry] > _incrlev) lnklvl[_entry] = _incrlev;\
    }\
  }\
}

/*
  Add a SORTED index set into a sorted linked list
  Input Parameters:
    nidx      - number of input indices
    idx   - sorted interger array used for storing column indices
    level     - level of fill, e.g., ICC(level)
    idxlvl - level of idx 
    idx_start - starting index of the list
    lnk       - linked list(an integer array) that is created
    lnklvl    - levels of lnk
    bt        - PetscBT (bitarray), bt[idx]=true marks idx is in lnk
  output Parameters:
    nlnk      - number of newly added idx
    lnk       - the sorted(increasing order) linked list containing new and non-redundate entries from idx
    lnklvl    - levels of lnk
    bt        - updated PetscBT (bitarray) 
*/
#define PetscIncompleteLLAddSorted(nidx,idx,level,idxlvl,idx_start,nlnk,lnk,lnklvl,bt) 0;\
{\
  PetscInt _k,_entry,_location,_lnkdata,_incrlev;\
  nlnk = 0;\
  _lnkdata = idx_start;\
  for (_k=0; _k<nidx; _k++){\
    _incrlev = idxlvl[_k] + 1;\
    if (_incrlev > level) continue;\
    _entry = idx[_k];\
    if (!PetscBTLookupSet(bt,_entry)){  /* new entry */\
      /* search for insertion location */\
      do {\
        _location = _lnkdata;\
        _lnkdata  = lnk[_location];\
      } while (_entry > _lnkdata);\
      /* insertion location is found, add entry into lnk */\
      lnk[_location] = _entry;\
      lnk[_entry]    = _lnkdata;\
      lnklvl[_entry] = _incrlev;\
      nlnk++;\
      _lnkdata = _entry; /* next search starts from here */\
    } else { /* existing entry: update lnklvl */\
      if (lnklvl[_entry] > _incrlev) lnklvl[_entry] = _incrlev;\
    }\
  }\
}

/*
  Add a SORTED index set into a sorted linked list for ICC
  Input Parameters:
    nidx      - number of input indices
    idx       - sorted interger array used for storing column indices
    level     - level of fill, e.g., ICC(level)
    idxlvl    - level of idx 
    idx_start - starting index of the list
    lnk       - linked list(an integer array) that is created
    lnklvl    - levels of lnk
    bt        - PetscBT (bitarray), bt[idx]=true marks idx is in lnk
    idxlvl_prow - idxlvl[prow], where prow is the row number of the idx
  output Parameters:
    nlnk   - number of newly added indices
    lnk    - the sorted(increasing order) linked list containing new and non-redundate entries from idx
    lnklvl - levels of lnk
    bt     - updated PetscBT (bitarray) 
  Note: the level of U(i,j) is set as lvl(i,j) = min{ lvl(i,j), lvl(prow,i)+lvl(prow,j)+1)
        where idx = non-zero columns of U(prow,prow+1:n-1), prow<i
*/
#define PetscICCLLAddSorted(nidx,idx,level,idxlvl,idx_start,nlnk,lnk,lnklvl,bt,idxlvl_prow) 0;\
{\
  PetscInt _k,_entry,_location,_lnkdata,_incrlev;\
  nlnk = 0;\
  _lnkdata = idx_start;\
  for (_k=0; _k<nidx; _k++){\
    _incrlev = idxlvl[_k] + idxlvl_prow + 1;\
    if (_incrlev > level) continue;\
    _entry = idx[_k];\
    if (!PetscBTLookupSet(bt,_entry)){  /* new entry */\
      /* search for insertion location */\
      do {\
        _location = _lnkdata;\
        _lnkdata  = lnk[_location];\
      } while (_entry > _lnkdata);\
      /* insertion location is found, add entry into lnk */\
      lnk[_location] = _entry;\
      lnk[_entry]    = _lnkdata;\
      lnklvl[_entry] = _incrlev;\
      nlnk++;\
      _lnkdata = _entry; /* next search starts from here */\
    } else { /* existing entry: update lnklvl */\
      if (lnklvl[_entry] > _incrlev) lnklvl[_entry] = _incrlev;\
    }\
  }\
}

/*
  Copy data on the list into an array, then initialize the list 
  Input Parameters:
    idx_start - starting index of the list 
    lnk_max   - max value of lnk indicating the end of the list 
    nlnk      - number of data on the list to be copied
    lnk       - linked list
    lnklvl    - level of lnk
    bt        - PetscBT (bitarray), bt[idx]=true marks idx is in lnk
  output Parameters:
    indices - array that contains the copied data
    lnk     - linked list that is cleaned and initialize
    lnklvl  - level of lnk that is reinitialized 
    bt      - PetscBT (bitarray) with all bits set to false
*/
#define PetscIncompleteLLClean(idx_start,lnk_max,nlnk,lnk,lnklvl,indices,indiceslvl,bt) 0;\
{\
  PetscInt _j,_idx=idx_start;\
  for (_j=0; _j<nlnk; _j++){\
    _idx = lnk[_idx];\
    *(indices+_j) = _idx;\
    *(indiceslvl+_j) = lnklvl[_idx];\
    lnklvl[_idx] = -1;\
    ierr = PetscBTClear(bt,_idx);CHKERRQ(ierr);\
  }\
  lnk[idx_start] = lnk_max;\
}
/*
  Free memories used by the list
*/
#define PetscIncompleteLLDestroy(lnk,bt) \
  (PetscFree(lnk) || PetscBTDestroy(bt) || (0))

PETSC_EXTERN_CXX_END
#endif /* __PETSCSYS_H */
