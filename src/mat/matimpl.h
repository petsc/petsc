/* $Id: matimpl.h,v 1.62 1996/08/22 20:07:54 curfman Exp bsmith $ */

#if !defined(__MATIMPL)
#define __MATIMPL
#include "mat.h"

/*
  This file defines the parts of the matrix data structure that are 
  shared by all matrix types.
*/

/*
    If you add entries here also add them to include/mat.h
*/
struct _MatOps {
  int       (*setvalues)(Mat,int,int *,int,int *,Scalar *,InsertMode),
            (*getrow)(Mat,int,int *,int **,Scalar **),
            (*restorerow)(Mat,int,int *,int **,Scalar **),
            (*mult)(Mat,Vec,Vec),
            (*multadd)(Mat,Vec,Vec,Vec),
            (*multtrans)(Mat,Vec,Vec),
            (*multtransadd)(Mat,Vec,Vec,Vec),
            (*solve)(Mat,Vec,Vec),
            (*solveadd)(Mat,Vec,Vec,Vec),
            (*solvetrans)(Mat,Vec,Vec),
            (*solvetransadd)(Mat,Vec,Vec,Vec),
            (*lufactor)(Mat,IS,IS,double),
            (*choleskyfactor)(Mat,IS,double),
            (*relax)(Mat,Vec,double,MatSORType,double,int,Vec),
            (*transpose)(Mat,Mat *),
            (*getinfo)(Mat,MatInfoType,MatInfo*),
            (*equal)(Mat,Mat,PetscTruth *),
            (*getdiagonal)(Mat,Vec),
            (*diagonalscale)(Mat,Vec,Vec),
            (*norm)(Mat,NormType,double *),
            (*assemblybegin)(Mat,MatAssemblyType),
            (*assemblyend)(Mat,MatAssemblyType),
            (*compress)(Mat),
            (*setoption)(Mat,MatOption),
            (*zeroentries)(Mat),
            (*zerorows)(Mat,IS,Scalar *),
            (*lufactorsymbolic)(Mat,IS,IS,double,Mat *),
            (*lufactornumeric)(Mat,Mat *),
            (*choleskyfactorsymbolic)(Mat,IS,double,Mat *),
            (*choleskyfactornumeric)(Mat,Mat *),
            (*getsize)(Mat,int *,int *),
            (*getlocalsize)(Mat,int *,int *),
            (*getownershiprange)(Mat,int *,int *),
            (*ilufactorsymbolic)(Mat,IS,IS,double,int,Mat *),
            (*incompletecholeskyfactorsymbolic)(Mat,IS,double,int,Mat *),
            (*getarray)(Mat,Scalar **),
            (*restorearray)(Mat,Scalar **),
            (*convert)(Mat,MatType,Mat *),
            (*convertsametype)(Mat,Mat *,int),
            (*forwardsolve)(Mat,Vec,Vec),
            (*backwardsolve)(Mat,Vec,Vec),
            (*ilufactor)(Mat,IS,IS,double,int),
            (*incompletecholeskyfactor)(Mat,IS,double),
            (*axpy)(Scalar *,Mat,Mat),
            (*getsubmatrices)(Mat,int,IS *,IS *,MatGetSubMatrixCall,Mat **),
            (*increaseoverlap)(Mat,int,IS *,int),
            (*getvalues)(Mat,int,int *,int,int *,Scalar *),
            (*copy)(Mat,Mat),
            (*printhelp)(Mat),
            (*scale)(Scalar *,Mat),
            (*shift)(Scalar *,Mat),
            (*diagonalshift)(Vec,Mat),
            (*iludtfactor)(Mat,double,int,IS,IS,Mat *),
            (*getblocksize)(Mat,int *),
            (*getrowij)(Mat,int,PetscTruth,int*,int **,int **,PetscTruth *),
            (*restorerowij)(Mat,int,PetscTruth,int *,int **,int **,PetscTruth *),
            (*getcolumnij)(Mat,int,PetscTruth,int*,int **,int **,PetscTruth *),
            (*restorecolumnij)(Mat,int,PetscTruth,int*,int **,int **,PetscTruth *);
};

#define FACTOR_LU       1
#define FACTOR_CHOLESKY 2

struct _Mat {
  PETSCHEADER                         /* general PETSc header */
  struct _MatOps   ops;               /* matrix operations */
  void             *data;             /* implementation-specific data */
  int              factor;            /* 0, FACTOR_LU, or FACTOR_CHOLESKY */
  double           lupivotthreshold;  /* threshold for pivoting */
  PetscTruth       assembled;         /* is the matrix assembled? */
  PetscTruth       was_assembled;     /* new values have been inserted into assembled mat */
  int              num_ass;           /* number of times matrix has been assembled */
  PetscTruth       same_nonzero;      /* matrix has same nonzero pattern as previous */
  int              M, N;              /* global numbers of rows, columns */
  int              m, n;              /* local numbers of rows, columns */
  MatInfo          info;              /* matrix information */
};

/* final argument for MatConvertXXX() */
#define DO_NOT_COPY_VALUES 0
#define COPY_VALUES        1

/* Since most (all?) of the parallel matrix assemblies use this stashing,
   we move it to a common place. Perhaps it ultimately belongs elsewhere. */

typedef struct {
  int    nmax;            /* maximum stash size */
  int    n;               /* stash size */
  int    *idx;            /* global row numbers in stash */
  int    *idy;            /* global column numbers in stash */
  Scalar *array;          /* array to hold stashed values */
} Stash;

extern int StashValues_Private(Stash*,int,int,int*,Scalar*,InsertMode);
extern int StashInitialize_Private(Stash*);
extern int StashBuild_Private(Stash*);
extern int StashDestroy_Private(Stash*);
extern int StashInfo_Private(Stash*);

/*
  Reorderings for sequential IJ format. By default uses SparsePak and Minpack routines.
*/
extern int MatGetReordering_IJ(int,int*,int*,MatReordering,IS *,IS*);
extern int MatGetColoring_IJ(int,int*,int*,MatReordering,int*,IS **);


extern int MatConvert_Basic(Mat,MatType,Mat*);
extern int MatCopy_Basic(Mat,Mat);

#endif


