/* $Id: spooles.h,v 1.46 2001/08/07 03:02:47 balay Exp $ */

#include "src/mat/matimpl.h"

#if !defined(__SPOOLES_H)
#define __SPOOLES_H

EXTERN_C_BEGIN
#include "/sandbox/hzhang/spooles/misc.h"
#include "/sandbox/hzhang/spooles/FrontMtx.h"
#include "/sandbox/hzhang/spooles/SymbFac.h"
EXTERN_C_END

typedef struct {
  InpMtx          *mtxA ;        /* coefficient matrix */
  ETree           *frontETree ;  /* defines numeric and symbolic factorizations */
  FrontMtx        *frontmtx ;    /* numeric L, D, U factor matrices */
  IV              *newToOldIV, *oldToNewIV ; /* permutation vectors */
  IVL             *symbfacIVL ;              /* symbolic factorization */
  int             msglvl,pivotingflag,symflag,seed;
  FILE            *msgFile ;
  SubMtxManager   *mtxmanager  ;  /* working array */
  double          cpus[10] ; 
  MatStructure    flg;
  int             *oldToNew,nz;
  double          tau;
  int             ordering,maxdomainsize,maxzeros,maxsize;
} Mat_Spooles;

EXTERN int MatDestroy_SeqAIJ_Spooles(Mat);
EXTERN int MatSolve_SeqAIJ_Spooles(Mat,Vec,Vec);
EXTERN int MatLUFactorNumeric_SeqAIJ_Spooles(Mat,Mat *); 

#endif
