/* $Id: spooles.h,v 1.46 2001/08/07 03:02:47 balay Exp $ */

#include "src/mat/matimpl.h"

#if !defined(__SPOOLES_H)
#define __SPOOLES_H

EXTERN_C_BEGIN
#include "misc.h"
#include "FrontMtx.h"
#include "SymbFac.h"
#include "MPI/spoolesMPI.h" 
EXTERN_C_END

typedef struct {
  int             msglvl,pivotingflag,symflag,seed;
  int             ordering,maxdomainsize,maxzeros,maxsize,
                  patchAndGoFlag,storeids,storevalues;
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

  /* Followings are used for MPI Spooles */
  IV              *ownersIV,*ownedColumnsIV,*vtxmapIV;
  SolveMap        *solvemap ;
  DenseMtx        *mtxY, *mtxX;
  double          *entX;
  int             *rowindX,rstart,firsttag,nmycol;
} Mat_Spooles;

EXTERN int SetSpoolesOptions(Mat, Spooles_options *);
EXTERN int MatFactorInfo_Spooles(Mat,PetscViewer);

EXTERN int MatDestroy_SeqAIJ_Spooles(Mat);
EXTERN int MatSolve_SeqAIJ_Spooles(Mat,Vec,Vec);
EXTERN int MatFactorNumeric_SeqAIJ_Spooles(Mat,Mat *); 

EXTERN int MatDestroy_MPIAIJ_Spooles(Mat);
EXTERN int MatSolve_MPIAIJ_Spooles(Mat,Vec,Vec);
EXTERN int MatFactorNumeric_MPIAIJ_Spooles(Mat,Mat *); 

#endif
