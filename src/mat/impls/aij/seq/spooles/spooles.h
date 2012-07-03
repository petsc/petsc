
#if !defined(__SPOOLES_H)
#define __SPOOLES_H
#include <petsc-private/matimpl.h>


EXTERN_C_BEGIN
#include <misc.h>
#include <FrontMtx.h>
#include <SymbFac.h>
#include <MPI/spoolesMPI.h>
EXTERN_C_END

typedef struct {
  PetscInt        msglvl,pivotingflag,symflag,seed,FrontMtxInfo,typeflag;
  PetscInt        ordering,maxdomainsize,maxzeros,maxsize,
                  patchAndGoFlag,storeids,storevalues;
  PetscBool       useQR;
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
  PetscInt        *oldToNew,stats[20];
  Spooles_options options;
  Graph           *graph;

  /* Followings are used for MPI Spooles */
  MPI_Comm        comm_spooles;          /* communicator to be passed to spooles */
  IV              *ownersIV,*ownedColumnsIV,*vtxmapIV;
  SolveMap        *solvemap;
  DenseMtx        *mtxY, *mtxX;
  double          *entX;
  PetscInt        *rowindX,rstart,firsttag,nmycol;
  Vec             vec_spooles;
  IS              iden,is_petsc;
  VecScatter      scat;
  
  PetscBool      CleanUpSpooles,useQR;
} Mat_Spooles;

extern PetscErrorCode SetSpoolesOptions(Mat, Spooles_options *);
extern PetscErrorCode MatFactorInfo_Spooles(Mat,PetscViewer);

extern PetscErrorCode MatDestroy_SeqAIJSpooles(Mat);
extern PetscErrorCode MatSolve_SeqSpooles(Mat,Vec,Vec);
extern PetscErrorCode MatFactorNumeric_SeqSpooles(Mat,Mat,const MatFactorInfo*); 
extern PetscErrorCode MatView_Spooles(Mat,PetscViewer);
extern PetscErrorCode MatAssemblyEnd_SeqAIJSpooles(Mat,MatAssemblyType);
extern PetscErrorCode MatLUFactorSymbolic_SeqAIJSpooles(Mat,Mat,IS,IS,const MatFactorInfo*);
extern PetscErrorCode MatCholeskyFactorSymbolic_SeqAIJSpooles(Mat,Mat,IS,const MatFactorInfo*);
extern PetscErrorCode MatDuplicate_Spooles(Mat,MatDuplicateOption,Mat*);

extern PetscErrorCode MatDestroy_MPIAIJSpooles(Mat);
extern PetscErrorCode MatSolve_MPISpooles(Mat,Vec,Vec);
extern PetscErrorCode MatFactorNumeric_MPISpooles(Mat,Mat,const MatFactorInfo*); 
extern PetscErrorCode MatAssemblyEnd_MPIAIJSpooles(Mat,MatAssemblyType);
extern PetscErrorCode MatLUFactorSymbolic_MPIAIJSpooles(Mat,Mat,IS,IS,const MatFactorInfo*);

extern PetscErrorCode MatDestroy_SeqSBAIJSpooles(Mat);
extern PetscErrorCode MatGetInertia_SeqSBAIJSpooles(Mat,PetscInt*,PetscInt*,PetscInt*);
extern PetscErrorCode MatCholeskyFactorSymbolic_SeqSBAIJSpooles(Mat,Mat,IS,const MatFactorInfo*);

extern PetscErrorCode MatCholeskyFactorSymbolic_MPISBAIJSpooles(Mat,Mat,IS,const MatFactorInfo*);
EXTERN_C_BEGIN
extern PetscErrorCode MatConvert_Spooles_Base(Mat,MatType,MatReuse,Mat*);
extern PetscErrorCode MatConvert_SeqAIJ_SeqAIJSpooles(Mat,MatType,MatReuse,Mat*);
extern PetscErrorCode MatConvert_SeqSBAIJ_SeqSBAIJSpooles(Mat,MatType,MatReuse,Mat*);
extern PetscErrorCode MatConvert_MPIAIJ_MPIAIJSpooles(Mat,MatType,MatReuse,Mat*);
extern PetscErrorCode MatConvert_MPISBAIJ_MPISBAIJSpooles(Mat,MatType,MatReuse,Mat*);
EXTERN_C_END
#endif
