/*$Id: spooles.c,v 1.10 2001/08/15 15:56:50 bsmith Exp $*/
/* 
   Provides an interface to the Spooles serial sparse solver
*/

#include "src/mat/impls/aij/seq/aij.h"
#include "src/mat/impls/sbaij/seq/sbaij.h"

#if defined(PETSC_HAVE_SPOOLES) && !defined(PETSC_USE_SINGLE) && !defined(PETSC_USE_COMPLEX)
#include "src/mat/impls/aij/seq/spooles.h"

extern int MatDestroy_SeqAIJ(Mat); 

#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_SeqAIJ_Spooles"
int MatDestroy_SeqAIJ_Spooles(Mat A)
{
  Mat_Spooles *lu = (Mat_Spooles*)A->spptr; 
  int                ierr;
  
  PetscFunctionBegin;
 
  FrontMtx_free(lu->frontmtx) ;        
  IV_free(lu->newToOldIV) ;            
  IV_free(lu->oldToNewIV) ;            
  InpMtx_free(lu->mtxA) ;             
  ETree_free(lu->frontETree) ;          
  IVL_free(lu->symbfacIVL) ;         
  SubMtxManager_free(lu->mtxmanager) ;  
  
  ierr = PetscFree(lu);CHKERRQ(ierr); 
  ierr = MatDestroy_SeqAIJ(A);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSolve_SeqAIJ_Spooles"
int MatSolve_SeqAIJ_Spooles(Mat A,Vec b,Vec x)
{
  Mat_Spooles      *lu = (Mat_Spooles*)A->spptr;
  PetscScalar      *array;
  DenseMtx         *mtxY, *mtxX ;
  double           *entX;
  int              ierr,irow,neqns=A->m,*iv;

  PetscFunctionBegin;

  /* copy permuted b to mtxY */
  mtxY = DenseMtx_new() ;
  DenseMtx_init(mtxY, SPOOLES_REAL, 0, 0, neqns, 1, 1, neqns) ; /* column major */
  iv = IV_entries(lu->oldToNewIV);
  ierr = VecGetArray(b,&array);CHKERRQ(ierr);
  for ( irow = 0 ; irow < neqns ; irow++ ) DenseMtx_setRealEntry(mtxY, *iv++, 0, *array++) ; 
  ierr = VecRestoreArray(b,&array);CHKERRQ(ierr);

  mtxX = DenseMtx_new() ;
  DenseMtx_init(mtxX, SPOOLES_REAL, 0, 0, neqns, 1, 1, neqns) ;
  DenseMtx_zero(mtxX) ;
  FrontMtx_solve(lu->frontmtx, mtxX, mtxY, lu->mtxmanager, 
                 lu->cpus, lu->msglvl, lu->msgFile) ;
  if ( lu->msglvl > 2 ) {
    fprintf(lu->msgFile, "\n\n right hand side matrix after permutation") ;
    DenseMtx_writeForHumanEye(mtxY, lu->msgFile) ; 
    fprintf(lu->msgFile, "\n\n solution matrix in new ordering") ;
    DenseMtx_writeForHumanEye(mtxX, lu->msgFile) ;
    fflush(lu->msgFile) ;
  }

  /* permute solution into original ordering, then copy to x */  
  DenseMtx_permuteRows(mtxX, lu->newToOldIV);
  ierr = VecGetArray(x,&array);CHKERRQ(ierr); 
  entX = DenseMtx_entries(mtxX);
  DVcopy(neqns, array, entX);
  ierr = VecRestoreArray(x,&array);CHKERRQ(ierr);
  
  /* free memory */
  DenseMtx_free(mtxX) ;
  DenseMtx_free(mtxY) ;

  PetscFunctionReturn(0);
}

#undef __FUNCT__   
#define __FUNCT__ "MatLUFactorNumeric_SeqAIJ_Spooles"
int MatLUFactorNumeric_SeqAIJ_Spooles(Mat A,Mat *F)
{  
  Mat_Spooles        *lu = (Mat_Spooles*)(*F)->spptr;
  ChvManager         *chvmanager ;
  FrontMtx           *frontmtx ; 
  Chv                *rootchv ;
  int                stats[20],ierr,nz,m=A->m,irow,count,
                     *ai,*aj,*ivec1, *ivec2, ii;
  PetscScalar        *av;
  double             *dvec;

  PetscFunctionBegin;
  if ( lu->flg == SAME_NONZERO_PATTERN){ /* new num factorization using previously computed symbolic factor */
    
    if (lu->pivotingflag) {              /* different FrontMtx is required */
      FrontMtx_free(lu->frontmtx) ;   
      lu->frontmtx   = FrontMtx_new() ;
    }

    SubMtxManager_free(lu->mtxmanager) ;  
    lu->mtxmanager = SubMtxManager_new() ;
    SubMtxManager_init(lu->mtxmanager, NO_LOCK, 0) ;

    /* get new numerical values of A , then permute the matrix */ 
    InpMtx_init(lu->mtxA, INPMTX_BY_ROWS, SPOOLES_REAL, lu->nz, m) ; 
    ivec1 = InpMtx_ivec1(lu->mtxA); 
    ivec2 = InpMtx_ivec2(lu->mtxA); 
    dvec  = InpMtx_dvec(lu->mtxA);

    if ( lu->symflag == SPOOLES_NONSYMMETRIC ) {
      Mat_SeqAIJ       *mat = (Mat_SeqAIJ*)A->data;
      ai=mat->i; aj=mat->j; av=mat->a;
    } else {
      Mat_SeqSBAIJ       *mat = (Mat_SeqSBAIJ*)A->data;
      ai=mat->i; aj=mat->j; av=mat->a;
    }
    for (irow = 0; irow < m; irow++){
      for (ii = ai[irow]; ii<ai[irow+1]; ii++) ivec1[ii] = irow;
    }
    IVcopy(lu->nz, ivec2, aj);
    DVcopy(lu->nz, dvec, av);
    InpMtx_inputRealTriples(lu->mtxA, lu->nz, ivec1, ivec2, dvec); 
    InpMtx_changeStorageMode(lu->mtxA, INPMTX_BY_VECTORS) ; 

    /* permute mtxA */
    InpMtx_permute(lu->mtxA, lu->oldToNew, lu->oldToNew) ;
    if ( lu->symflag == SPOOLES_SYMMETRIC ) InpMtx_mapToUpperTriangle(lu->mtxA) ; 
    InpMtx_changeCoordType(lu->mtxA, INPMTX_BY_CHEVRONS) ;
    InpMtx_changeStorageMode(lu->mtxA, INPMTX_BY_VECTORS) ;
    if ( lu->msglvl > 2 ) {
      fprintf(lu->msgFile, "\n\n input matrix after permutation") ;
      InpMtx_writeForHumanEye(lu->mtxA, lu->msgFile) ; 
    } 
  }

  /* initialize the front matrix object */
  if ( lu->flg == DIFFERENT_NONZERO_PATTERN){ /* first numeric factorization */
    lu->frontmtx   = FrontMtx_new() ;
    lu->mtxmanager = SubMtxManager_new() ;
    SubMtxManager_init(lu->mtxmanager, NO_LOCK, 0) ;
  }
  
  FrontMtx_init(lu->frontmtx, lu->frontETree, lu->symbfacIVL, SPOOLES_REAL, lu->symflag, 
                FRONTMTX_DENSE_FRONTS, lu->pivotingflag, NO_LOCK, 0, NULL, 
                lu->mtxmanager, lu->msglvl, lu->msgFile) ;   

  /* numerical factorization */
  chvmanager = ChvManager_new() ;
  ChvManager_init(chvmanager, NO_LOCK, 1) ;
  DVfill(10, lu->cpus, 0.0) ;
  IVfill(20, stats, 0) ;
  rootchv = FrontMtx_factorInpMtx(lu->frontmtx, lu->mtxA, lu->tau, 0.0, 
            chvmanager, &ierr, lu->cpus, stats, lu->msglvl, lu->msgFile) ; 
  ChvManager_free(chvmanager) ;
  if ( lu->msglvl > 0 ) {
    fprintf(lu->msgFile, "\n\n factor matrix") ;
    FrontMtx_writeForHumanEye(lu->frontmtx, lu->msgFile) ;
    fflush(lu->msgFile) ;
  }
  if ( rootchv != NULL ) SETERRQ(1,"\n matrix found to be singular");    
  if ( ierr >= 0 ) SETERRQ1(1,"\n error encountered at front %d", ierr);

  /* post-process the factorization */
  FrontMtx_postProcess(lu->frontmtx, lu->msglvl, lu->msgFile) ;
  if ( lu->msglvl > 2 ) {
    fprintf(lu->msgFile, "\n\n factor matrix after post-processing") ;
    FrontMtx_writeForHumanEye(lu->frontmtx, lu->msgFile) ;
    fflush(lu->msgFile) ;
  }

  lu->flg         = SAME_NONZERO_PATTERN;
  (*F)->assembled = PETSC_TRUE;

  PetscFunctionReturn(0);
}
#endif
