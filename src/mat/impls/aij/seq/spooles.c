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
  int         ierr;
  
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
                 lu->cpus, lu->options.msglvl, lu->options.msgFile) ;
  if ( lu->options.msglvl > 2 ) {
    fprintf(lu->options.msgFile, "\n\n right hand side matrix after permutation") ;
    DenseMtx_writeForHumanEye(mtxY, lu->options.msgFile) ; 
    fprintf(lu->options.msgFile, "\n\n solution matrix in new ordering") ;
    DenseMtx_writeForHumanEye(mtxX, lu->options.msgFile) ;
    fflush(lu->options.msgFile) ;
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
#define __FUNCT__ "MatFactorNumeric_SeqAIJ_Spooles"
int MatFactorNumeric_SeqAIJ_Spooles(Mat A,Mat *F)
{  
  Mat_Spooles        *lu = (Mat_Spooles*)(*F)->spptr;
  ChvManager         *chvmanager ;
  Chv                *rootchv ;
  Graph              *graph ;
  IVL                *adjIVL;
  int                ierr,nz,m=A->m,irow,nedges,
                     *ai,*aj,*ivec1, *ivec2, i;
  PetscScalar        *av;
  double             *dvec;
  
  PetscFunctionBegin;
  /* copy A to Spooles' InpMtx object */
  if ( lu->options.symflag == SPOOLES_NONSYMMETRIC ) {
    Mat_SeqAIJ   *mat = (Mat_SeqAIJ*)A->data;
    ai=mat->i; aj=mat->j; av=mat->a;
    nz=mat->nz;
  } else {
    Mat_SeqSBAIJ *mat = (Mat_SeqSBAIJ*)A->data;
    ai=mat->i; aj=mat->j; av=mat->a;
    nz=mat->s_nz;
  }
  if (lu->flg == DIFFERENT_NONZERO_PATTERN) lu->mtxA = InpMtx_new() ;
  InpMtx_init(lu->mtxA, INPMTX_BY_ROWS, SPOOLES_REAL, nz, m) ;
  ivec1 = InpMtx_ivec1(lu->mtxA);  
  ivec2 = InpMtx_ivec2(lu->mtxA); 
  dvec  = InpMtx_dvec(lu->mtxA);
  for (irow = 0; irow < m; irow++){
    for (i = ai[irow]; i<ai[irow+1]; i++) ivec1[i] = irow;
  }
  IVcopy(nz, ivec2, aj);
  DVcopy(nz, dvec, av);
  InpMtx_inputRealTriples(lu->mtxA, nz, ivec1, ivec2, dvec); 
  InpMtx_changeStorageMode(lu->mtxA, INPMTX_BY_VECTORS) ; 

  if ( lu->flg == DIFFERENT_NONZERO_PATTERN){ /* first numeric factorization */
    
    (*F)->ops->solve   = MatSolve_SeqAIJ_Spooles;
    (*F)->ops->destroy = MatDestroy_SeqAIJ_Spooles;  
    (*F)->assembled    = PETSC_TRUE; 
    
    ierr = SetSpoolesOptions(A, &lu->options);CHKERRQ(ierr); 

    /*---------------------------------------------------
    find a low-fill ordering
         (1) create the Graph object
         (2) order the graph using multiple minimum degree
    -------------------------------------------------------*/  
    graph = Graph_new() ;
    adjIVL = InpMtx_fullAdjacency(lu->mtxA) ;
    nedges = IVL_tsize(adjIVL) ;
    Graph_init2(graph, 0, m, 0, nedges, m, nedges, adjIVL,NULL, NULL) ;
    if ( lu->options.msglvl > 2 ) {
      fprintf(lu->options.msgFile, "\n\n graph of the input matrix") ;
      Graph_writeForHumanEye(graph, lu->options.msgFile) ;
      fflush(lu->options.msgFile) ;
    }

    switch (lu->options.ordering) {
    case 0:
      lu->frontETree = orderViaBestOfNDandMS(graph,
                     lu->options.maxdomainsize, lu->options.maxzeros, lu->options.maxsize,
                     lu->options.seed, lu->options.msglvl, lu->options.msgFile); break;
    case 1:
      lu->frontETree = orderViaMMD(graph,lu->options.seed,lu->options.msglvl,lu->options.msgFile); break;
    case 2:
      lu->frontETree = orderViaMS(graph, lu->options.maxdomainsize,
                     lu->options.seed,lu->options.msglvl,lu->options.msgFile); break;
    case 3:
      lu->frontETree = orderViaND(graph, lu->options.maxdomainsize, 
                     lu->options.seed,lu->options.msglvl,lu->options.msgFile); break;
    default:
      SETERRQ(1,"Unknown Spooles's ordering");
    }
    Graph_free(graph) ;

    if ( lu->options.msglvl > 0 ) {
      fprintf(lu->options.msgFile, "\n\n front tree from ordering") ;
      ETree_writeForHumanEye(lu->frontETree, lu->options.msgFile) ;
      fflush(lu->options.msgFile) ;
    }
  
    /* get the permutation, permute the front tree, permute the matrix */
    lu->oldToNewIV = ETree_oldToNewVtxPerm(lu->frontETree) ;
    lu->oldToNew   = IV_entries(lu->oldToNewIV) ;
    lu->newToOldIV = ETree_newToOldVtxPerm(lu->frontETree) ;
    ETree_permuteVertices(lu->frontETree, lu->oldToNewIV) ;

    InpMtx_permute(lu->mtxA, lu->oldToNew, lu->oldToNew) ; 
    if ( lu->options.symflag == SPOOLES_SYMMETRIC ) {
      InpMtx_mapToUpperTriangle(lu->mtxA) ; 
    }
    InpMtx_changeCoordType(lu->mtxA, INPMTX_BY_CHEVRONS) ;
    InpMtx_changeStorageMode(lu->mtxA, INPMTX_BY_VECTORS) ;

    /* get symbolic factorization */
    lu->symbfacIVL = SymbFac_initFromInpMtx(lu->frontETree, lu->mtxA) ;

    if ( lu->options.msglvl > 2 ) {
      fprintf(lu->options.msgFile, "\n\n old-to-new permutation vector") ;
      IV_writeForHumanEye(lu->oldToNewIV, lu->options.msgFile) ;
      fprintf(lu->options.msgFile, "\n\n new-to-old permutation vector") ;
      IV_writeForHumanEye(lu->newToOldIV, lu->options.msgFile) ;
      fprintf(lu->options.msgFile, "\n\n front tree after permutation") ;
      ETree_writeForHumanEye(lu->frontETree, lu->options.msgFile) ;
      fprintf(lu->options.msgFile, "\n\n input matrix after permutation") ;
      InpMtx_writeForHumanEye(lu->mtxA, lu->options.msgFile) ;
      fprintf(lu->options.msgFile, "\n\n symbolic factorization") ;
      IVL_writeForHumanEye(lu->symbfacIVL, lu->options.msgFile) ;
      fflush(lu->options.msgFile) ;
    }  

    lu->frontmtx   = FrontMtx_new() ;
    lu->mtxmanager = SubMtxManager_new() ;
    SubMtxManager_init(lu->mtxmanager, NO_LOCK, 0) ;

  } else { /* new num factorization using previously computed symbolic factor */ 
    if (lu->options.pivotingflag) {              /* different FrontMtx is required */
      FrontMtx_free(lu->frontmtx) ;   
      lu->frontmtx   = FrontMtx_new() ;
    }

    SubMtxManager_free(lu->mtxmanager) ;  
    lu->mtxmanager = SubMtxManager_new() ;
    SubMtxManager_init(lu->mtxmanager, NO_LOCK, 0) ;

    /* permute mtxA */
    InpMtx_permute(lu->mtxA, lu->oldToNew, lu->oldToNew) ;
    if ( lu->options.symflag == SPOOLES_SYMMETRIC ) InpMtx_mapToUpperTriangle(lu->mtxA) ; 
    InpMtx_changeCoordType(lu->mtxA, INPMTX_BY_CHEVRONS) ;
    InpMtx_changeStorageMode(lu->mtxA, INPMTX_BY_VECTORS) ;
    if ( lu->options.msglvl > 2 ) {
      fprintf(lu->options.msgFile, "\n\n input matrix after permutation") ;
      InpMtx_writeForHumanEye(lu->mtxA, lu->options.msgFile) ; 
    } 
  } /* end of if( lu->flg == DIFFERENT_NONZERO_PATTERN) */

  FrontMtx_init(lu->frontmtx, lu->frontETree, lu->symbfacIVL, SPOOLES_REAL, lu->options.symflag, 
                FRONTMTX_DENSE_FRONTS, lu->options.pivotingflag, NO_LOCK, 0, NULL, 
                lu->mtxmanager, lu->options.msglvl, lu->options.msgFile) ;   
  

  if ( lu->options.symflag == SPOOLES_SYMMETRIC ) {
    if ( lu->options.patchAndGoFlag == 1 ) {
      lu->frontmtx->patchinfo = PatchAndGoInfo_new() ;
      PatchAndGoInfo_init(lu->frontmtx->patchinfo, 1, lu->options.toosmall, lu->options.fudge,
                       lu->options.storeids, lu->options.storevalues) ;
    } else if ( lu->options.patchAndGoFlag == 2 ) {
      lu->frontmtx->patchinfo = PatchAndGoInfo_new() ;
      PatchAndGoInfo_init(lu->frontmtx->patchinfo, 2, lu->options.toosmall, lu->options.fudge,
                       lu->options.storeids, lu->options.storevalues) ;
    }   
  }

  /* numerical factorization */
  chvmanager = ChvManager_new() ;
  ChvManager_init(chvmanager, NO_LOCK, 1) ;
  DVfill(10, lu->cpus, 0.0) ;
  IVfill(20, lu->stats, 0) ;
  rootchv = FrontMtx_factorInpMtx(lu->frontmtx, lu->mtxA, lu->options.tau, 0.0, 
            chvmanager, &ierr, lu->cpus, lu->stats, lu->options.msglvl, lu->options.msgFile) ; 
  ChvManager_free(chvmanager) ;
  if(lu->options.FrontMtxInfo){
    PetscPrintf(PETSC_COMM_SELF,"\n %8d pivots, %8d pivot tests, %8d delayed rows and columns\n",\
               lu->stats[0], lu->stats[1], lu->stats[2]);
    double cputotal;
    cputotal = lu->cpus[8] ;
    if ( cputotal > 0.0 ) {
      PetscPrintf(PETSC_COMM_SELF,
           "\n                               cpus   cpus/totaltime"
           "\n    initialize fronts       %8.3f %6.2f"
           "\n    load original entries   %8.3f %6.2f"
           "\n    update fronts           %8.3f %6.2f"
           "\n    assemble postponed data %8.3f %6.2f"
           "\n    factor fronts           %8.3f %6.2f"
           "\n    extract postponed data  %8.3f %6.2f"
           "\n    store factor entries    %8.3f %6.2f"
           "\n    miscellaneous           %8.3f %6.2f"
           "\n    total time              %8.3f \n",
           lu->cpus[0], 100.*lu->cpus[0]/cputotal,
           lu->cpus[1], 100.*lu->cpus[1]/cputotal,
           lu->cpus[2], 100.*lu->cpus[2]/cputotal,
           lu->cpus[3], 100.*lu->cpus[3]/cputotal,
           lu->cpus[4], 100.*lu->cpus[4]/cputotal,
           lu->cpus[5], 100.*lu->cpus[5]/cputotal,
           lu->cpus[6], 100.*lu->cpus[6]/cputotal,
           lu->cpus[7], 100.*lu->cpus[7]/cputotal, cputotal) ;
    }
  }
  if ( lu->options.msglvl > 0 ) {
    fprintf(lu->options.msgFile, "\n\n factor matrix") ;
    FrontMtx_writeForHumanEye(lu->frontmtx, lu->options.msgFile) ;
    fflush(lu->options.msgFile) ;
  }

  if ( lu->options.symflag == SPOOLES_SYMMETRIC ) {
    if ( lu->options.patchAndGoFlag == 1 ) {
      if ( lu->frontmtx->patchinfo->fudgeIV != NULL ) {
        if (lu->options.msglvl > 0 ){
          fprintf(lu->options.msgFile, "\n small pivots found at these locations") ;
          IV_writeForHumanEye(lu->frontmtx->patchinfo->fudgeIV, lu->options.msgFile) ;
        }
      }
      PatchAndGoInfo_free(lu->frontmtx->patchinfo) ;
    } else if ( lu->options.patchAndGoFlag == 2 ) {
      if (lu->options.msglvl > 0 ){
        if ( lu->frontmtx->patchinfo->fudgeIV != NULL ) {
          fprintf(lu->options.msgFile, "\n small pivots found at these locations") ;
          IV_writeForHumanEye(lu->frontmtx->patchinfo->fudgeIV, lu->options.msgFile) ;
        }
        if ( lu->frontmtx->patchinfo->fudgeDV != NULL ) {
          fprintf(lu->options.msgFile, "\n perturbations") ;
          DV_writeForHumanEye(lu->frontmtx->patchinfo->fudgeDV, lu->options.msgFile) ;
        }
      }
      PatchAndGoInfo_free(lu->frontmtx->patchinfo) ;
    }
  }

  if ( rootchv != NULL ) SETERRQ(1,"\n matrix found to be singular");    
  if ( ierr >= 0 ) SETERRQ1(1,"\n error encountered at front %d", ierr);

  /* post-process the factorization */
  FrontMtx_postProcess(lu->frontmtx, lu->options.msglvl, lu->options.msgFile) ;
  if ( lu->options.msglvl > 2 ) {
    fprintf(lu->options.msgFile, "\n\n factor matrix after post-processing") ;
    FrontMtx_writeForHumanEye(lu->frontmtx, lu->options.msgFile) ;
    fflush(lu->options.msgFile) ;
  }

  lu->flg         = SAME_NONZERO_PATTERN;
 
  PetscFunctionReturn(0);
}

#endif
