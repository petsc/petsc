
#if !defined(__SPOOLES_H)
#define __SPOOLES_H
#include "src/mat/matimpl.h"


EXTERN_C_BEGIN
#include "misc.h"
#include "FrontMtx.h"
#include "SymbFac.h"
#include "MPI/spoolesMPI.h" 
EXTERN_C_END

typedef struct {
  int             msglvl,pivotingflag,symflag,seed,FrontMtxInfo,typeflag;
  int             ordering,maxdomainsize,maxzeros,maxsize,
                  patchAndGoFlag,storeids,storevalues;
  PetscTruth      useQR;
  double          tau,toosmall,fudge;
  FILE            *msgFile ;
} Spooles_options;

typedef struct {
  /* Followings are used for seq and MPI Spooles */
  InpMtx          *mtxA ;        /* coefficient matrix */
  ETree           *frontETree ;  /* defines numeric and symbolic factorizations */
  FrontMtx        *frontmtx ;    /* numeric L, D, U factor matrices */
  IV              *newToOldIV, *oldToNewIV ; /* permutation vectors */
  IVL             *symbfacIVL ;              /* symbolic factorization */
  SubMtxManager   *mtxmanager  ;  /* working array */
  MatStructure    flg;
  double          cpus[20] ;
  int             *oldToNew,stats[20];
  Spooles_options options;
  Graph           *graph;

  /* Followings are used for MPI Spooles */
  MPI_Comm        comm_spooles;          /* communicator to be passed to spooles */
  IV              *ownersIV,*ownedColumnsIV,*vtxmapIV;
  SolveMap        *solvemap;
  DenseMtx        *mtxY, *mtxX;
  double          *entX;
  int             *rowindX,rstart,firsttag,nmycol;
  Vec             vec_spooles;
  IS              iden,is_petsc;
  VecScatter      scat;
  
  /* A few function pointers for inheritance */
  int (*MatDuplicate)(Mat,MatDuplicateOption,Mat*);
  int (*MatCholeskyFactorSymbolic)(Mat,IS,MatFactorInfo*,Mat*);
  int (*MatLUFactorSymbolic)(Mat,IS,IS,MatFactorInfo*,Mat*);
  int (*MatView)(Mat,PetscViewer);
  int (*MatAssemblyEnd)(Mat,MatAssemblyType);
  int (*MatDestroy)(Mat);
  int (*MatPreallocate)(Mat,int,int,int*,int,int*);

  MatType    basetype;
  PetscTruth CleanUpSpooles,useQR;
} Mat_Spooles;

EXTERN int SetSpoolesOptions(Mat, Spooles_options *);
EXTERN int MatFactorInfo_Spooles(Mat,PetscViewer);

EXTERN int MatDestroy_SeqAIJSpooles(Mat);
EXTERN int MatSolve_SeqAIJSpooles(Mat,Vec,Vec);
EXTERN int MatFactorNumeric_SeqAIJSpooles(Mat,Mat*); 
EXTERN int MatView_SeqAIJSpooles(Mat,PetscViewer);
EXTERN int MatAssemblyEnd_SeqAIJSpooles(Mat,MatAssemblyType);
EXTERN int MatQRFactorSymbolic_SeqAIJSpooles(Mat,IS,IS,MatFactorInfo*,Mat*);
EXTERN int MatLUFactorSymbolic_SeqAIJSpooles(Mat,IS,IS,MatFactorInfo*,Mat*);
EXTERN int MatCholeskyFactorSymbolic_SeqAIJSpooles(Mat,IS,MatFactorInfo*,Mat*);
EXTERN int MatDuplicate_Spooles(Mat,MatDuplicateOption,Mat*);

EXTERN int MatDestroy_MPIAIJSpooles(Mat);
EXTERN int MatSolve_MPIAIJSpooles(Mat,Vec,Vec);
EXTERN int MatFactorNumeric_MPIAIJSpooles(Mat,Mat*); 
EXTERN int MatAssemblyEnd_MPIAIJSpooles(Mat,MatAssemblyType);
EXTERN int MatLUFactorSymbolic_MPIAIJSpooles(Mat,IS,IS,MatFactorInfo*,Mat*);

EXTERN int MatDestroy_SeqSBAIJSpooles(Mat);
EXTERN int MatGetInertia_SeqSBAIJSpooles(Mat,int*,int*,int*);
EXTERN int MatCholeskyFactorSymbolic_SeqSBAIJSpooles(Mat,IS,MatFactorInfo*,Mat*);

EXTERN int MatCholeskyFactorSymbolic_MPISBAIJSpooles(Mat,IS,MatFactorInfo*,Mat*);
EXTERN_C_BEGIN
EXTERN int MatConvert_Spooles_Base(Mat,const MatType,Mat*);
EXTERN int MatConvert_SeqAIJ_SeqAIJSpooles(Mat,const MatType,Mat*);
EXTERN int MatConvert_SeqSBAIJ_SeqSBAIJSpooles(Mat,const MatType,Mat*);
EXTERN int MatConvert_MPIAIJ_MPIAIJSpooles(Mat,const MatType,Mat*);
EXTERN int MatConvert_MPISBAIJ_MPISBAIJSpooles(Mat,const MatType,Mat*);
EXTERN_C_END
#endif
