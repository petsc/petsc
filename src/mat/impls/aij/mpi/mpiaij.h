/* $Id: mpiaij.h,v 1.23 2001/08/07 03:02:49 balay Exp $ */

#if !defined(__MPIAIJ_H)
#define __MPIAIJ_H

#include "src/mat/impls/aij/seq/aij.h"
#include "src/sys/ctable.h"

typedef struct {
  int           *rowners,*cowners;     /* ranges owned by each processor */
  int           rstart,rend;           /* starting and ending owned rows */
  int           cstart,cend;           /* starting and ending owned columns */
  Mat           A,B;                   /* local submatrices: A (diag part),
                                           B (off-diag part) */
  int           size;                   /* size of communicator */
  int           rank;                   /* rank of proc in communicator */ 

  /* The following variables are used for matrix assembly */

  PetscTruth    donotstash;             /* PETSC_TRUE if off processor entries dropped */
  MPI_Request   *send_waits;            /* array of send requests */
  MPI_Request   *recv_waits;            /* array of receive requests */
  int           nsends,nrecvs;         /* numbers of sends and receives */
  PetscScalar   *svalues,*rvalues;     /* sending and receiving data */
  int           rmax;                   /* maximum message length */
#if defined (PETSC_USE_CTABLE)
  PetscTable    colmap;
#else
  int           *colmap;                /* local col number of off-diag col */
#endif
  int           *garray;                /* global index of all off-processor columns */

  /* The following variables are used for matrix-vector products */

  Vec           lvec;              /* local vector */
  VecScatter    Mvctx;             /* scatter context for vector */
  PetscTruth    roworiented;       /* if true, row-oriented input, default true */

  /* The following variables are for MatGetRow() */

  int           *rowindices;       /* column indices for row */
  PetscScalar   *rowvalues;        /* nonzero values in row */
  PetscTruth    getrowactive;      /* indicates MatGetRow(), not restored */

} Mat_MPIAIJ;

EXTERN int MatSetColoring_MPIAIJ(Mat,ISColoring);
EXTERN int MatSetValuesAdic_MPIAIJ(Mat,void*);
EXTERN int MatSetValuesAdifor_MPIAIJ(Mat,int,void*);
EXTERN int MatSetUpMultiply_MPIAIJ(Mat);
EXTERN int DisAssemble_MPIAIJ(Mat);
EXTERN int MatSetValues_SeqAIJ(Mat,int,int*,int,int*,PetscScalar*,InsertMode);
EXTERN int MatGetRow_SeqAIJ(Mat,int,int*,int**,PetscScalar**);
EXTERN int MatRestoreRow_SeqAIJ(Mat,int,int*,int**,PetscScalar**);
EXTERN int MatPrintHelp_SeqAIJ(Mat);
EXTERN int MatDuplicate_MPIAIJ(Mat,MatDuplicateOption,Mat *);
EXTERN int MatIncreaseOverlap_MPIAIJ(Mat,int,IS [],int);
EXTERN int MatFDColoringCreate_MPIAIJ(Mat,ISColoring,MatFDColoring);
EXTERN int MatGetSubMatrices_MPIAIJ (Mat,int,const IS[],const IS[],MatReuse,Mat *[]);
EXTERN int MatGetSubMatrix_MPIAIJ (Mat,IS,IS,int,MatReuse,Mat *);
EXTERN int MatLoad_MPIAIJ(PetscViewer,MatType,Mat*);
EXTERN int MatAXPY_SeqAIJ(const PetscScalar[],Mat,Mat,MatStructure);

#if !defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_SINGLE)
EXTERN int MatLUFactorSymbolic_MPIAIJ_TFS(Mat,IS,IS,MatFactorInfo*,Mat*);
#endif

EXTERN_C_BEGIN
EXTERN int MatGetDiagonalBlock_MPIAIJ(Mat,PetscTruth *,MatReuse,Mat *);
EXTERN int MatDiagonalScaleLocal_MPIAIJ(Mat,Vec);
EXTERN_C_END

#endif
