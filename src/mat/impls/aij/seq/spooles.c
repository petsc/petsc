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
  Graph              *graph ;
  IVL                *adjIVL;
  int                stats[20],ierr,nz,m=A->m,irow,nedges,
                     *ai,*aj,*ivec1, *ivec2, i;
  PetscScalar        *av;
  double             *dvec;
  char               buff[32],*ordertype[]={"BestOfNDandMS","MMD","MS","ND"}; 
  PetscTruth         flg;

  PetscFunctionBegin;
  if ( lu->flg == DIFFERENT_NONZERO_PATTERN){ /* first numeric factorization */

    (*F)->ops->solve   = MatSolve_SeqAIJ_Spooles;
    (*F)->ops->destroy = MatDestroy_SeqAIJ_Spooles;  

    /* set default input parameters */
    lu->tau            = 100.;
    lu->seed           = 10101;  
    lu->ordering       = 0;
    lu->maxdomainsize  = 500;
    lu->maxzeros       = 1000;
    lu->maxsize        = 96;
    if ( lu->symflag == SPOOLES_SYMMETRIC ) {
      lu->patchAndGoFlag = 1;
      lu->storeids       = 1; 
      lu->storevalues    = 1;
      lu->toosmall       = 1.e-9;
      lu->fudge          = 1.e-9;
    }

    /* get runtime input parameters */
    ierr = PetscOptionsBegin(A->comm,A->prefix,"Spooles Options","Mat");CHKERRQ(ierr); 

    ierr = PetscOptionsReal("-mat_aij_spooles_tau","tau (used for pivoting; \n\
           all entries in L and U have magnitude no more than tau)","None",lu->tau,&lu->tau,PETSC_NULL);CHKERRQ(ierr);

    ierr = PetscOptionsInt("-mat_aij_spooles_seed","random number seed, used for ordering","None",lu->seed,&lu->seed,PETSC_NULL);CHKERRQ(ierr);

    if (PetscLogPrintInfo) {
      lu->msglvl = 1;
    } else {
      lu->msglvl = 0;
    }
    ierr = PetscOptionsInt("-mat_aij_spooles_msglvl","msglvl","None",lu->msglvl,&lu->msglvl,0);CHKERRQ(ierr); 
    if (lu->msglvl > 0) {
        lu->msgFile = fopen("spooles.msgFile", "a");
        PetscPrintf(PETSC_COMM_SELF,"\n Spooles' output is written into the file 'spooles.msgFile' \n\n");
    } 

    ierr = PetscOptionsEList("-mat_aij_spooles_ordering","ordering type","None",
             ordertype,4,ordertype[0],buff,32,&flg);CHKERRQ(ierr);
    while (flg) {
      ierr = PetscStrcmp(buff,"BestOfNDandMS",&flg);CHKERRQ(ierr);
      if (flg) {
        lu->ordering = 0;
        break;
      }
      ierr = PetscStrcmp(buff,"MMD",&flg);CHKERRQ(ierr);
      if (flg) {
        lu->ordering = 1;
        break;
      }
      ierr = PetscStrcmp(buff,"MS",&flg);CHKERRQ(ierr);
      if (flg) {
        lu->ordering = 2;
        break;
      }
      ierr = PetscStrcmp(buff,"ND",&flg);CHKERRQ(ierr);
      if (flg) {
        lu->ordering = 3;
        break;
      }
      SETERRQ1(1,"Unknown Spooles's ordering %s",buff);
    }
   
    ierr = PetscOptionsInt("-mat_aij_spooles_maxdomainsize","maxdomainsize","None",\
                           lu->maxdomainsize,&lu->maxdomainsize,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-mat_aij_spooles_maxzeros ","maxzeros","None",\
                           lu->maxzeros,&lu->maxzeros,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-mat_aij_spooles_maxsize","maxsize","None",\
                           lu->maxsize,&lu->maxsize,PETSC_NULL);CHKERRQ(ierr);
    if ( lu->symflag == SPOOLES_SYMMETRIC ) {
      ierr = PetscOptionsInt("-mat_aij_spooles_patchAndGoFlag","patchAndGoFlag","None", \
                           lu->patchAndGoFlag,&lu->patchAndGoFlag,PETSC_NULL);CHKERRQ(ierr);
      ierr = PetscOptionsReal("-mat_aij_spooles_fudge","fudge","None", \
                           lu->fudge,&lu->fudge,PETSC_NULL);CHKERRQ(ierr);
      ierr = PetscOptionsReal("-mat_aij_spooles_toosmall","toosmall","None", \
                           lu->toosmall,&lu->toosmall,PETSC_NULL);CHKERRQ(ierr);
      ierr = PetscOptionsInt("-mat_aij_spooles_storeids","storeids","None", \
                           lu->storeids,&lu->storeids,PETSC_NULL);CHKERRQ(ierr);
      ierr = PetscOptionsInt("-mat_aij_spooles_storevalues","storevalues","None", \
                           lu->storevalues,&lu->storevalues,PETSC_NULL);CHKERRQ(ierr);
    }
    PetscOptionsEnd();
  
    /* copy A to Spooles' InpMtx object */
    if ( lu->symflag == SPOOLES_NONSYMMETRIC ) {
      Mat_SeqAIJ       *mat = (Mat_SeqAIJ*)A->data;
      ai=mat->i; aj=mat->j; av=mat->a;
      lu->nz=mat->nz;
    } else {
      Mat_SeqSBAIJ       *mat = (Mat_SeqSBAIJ*)A->data;
      ai=mat->i; aj=mat->j; av=mat->a;
      lu->nz=mat->s_nz;
    }

    lu->mtxA = InpMtx_new() ;
    InpMtx_init(lu->mtxA, INPMTX_BY_ROWS, SPOOLES_REAL, lu->nz, m) ;
    ivec1 = InpMtx_ivec1(lu->mtxA);  
    ivec2 = InpMtx_ivec2(lu->mtxA); 
    dvec  = InpMtx_dvec(lu->mtxA);
    for (irow = 0; irow < m; irow++){
      for (i = ai[irow]; i<ai[irow+1]; i++) ivec1[i] = irow;
    }
    IVcopy(lu->nz, ivec2, aj);
    DVcopy(lu->nz, dvec, av);
    InpMtx_inputRealTriples(lu->mtxA, lu->nz, ivec1, ivec2, dvec); 
    InpMtx_changeStorageMode(lu->mtxA, INPMTX_BY_VECTORS) ; 

    /*---------------------------------------------------
    find a low-fill ordering
         (1) create the Graph object
         (2) order the graph using multiple minimum degree
    -------------------------------------------------------*/  
    graph = Graph_new() ;
    adjIVL = InpMtx_fullAdjacency(lu->mtxA) ;
    nedges = IVL_tsize(adjIVL) ;
    Graph_init2(graph, 0, m, 0, nedges, m, nedges, adjIVL,NULL, NULL) ;
    if ( lu->msglvl > 2 ) {
      fprintf(lu->msgFile, "\n\n graph of the input matrix") ;
      Graph_writeForHumanEye(graph, lu->msgFile) ;
      fflush(lu->msgFile) ;
    }

    switch (lu->ordering) {
    case 0:
      lu->frontETree = orderViaBestOfNDandMS(graph,
                     lu->maxdomainsize, lu->maxzeros, lu->maxsize,
                     lu->seed, lu->msglvl, lu->msgFile); break;
    case 1:
      lu->frontETree = orderViaMMD(graph,lu->seed,lu->msglvl,lu->msgFile); break;
    case 2:
      lu->frontETree = orderViaMS(graph, lu->maxdomainsize,
                     lu->seed,lu->msglvl,lu->msgFile); break;
    case 3:
      lu->frontETree = orderViaND(graph, lu->maxdomainsize, 
                     lu->seed,lu->msglvl,lu->msgFile); break;
    default:
      SETERRQ(1,"Unknown Spooles's ordering");
    }
    Graph_free(graph) ;

    if ( lu->msglvl > 0 ) {
      fprintf(lu->msgFile, "\n\n front tree from ordering") ;
      ETree_writeForHumanEye(lu->frontETree, lu->msgFile) ;
      fflush(lu->msgFile) ;
    }
  
    /* get the permutation, permute the front tree, permute the matrix */
    lu->oldToNewIV = ETree_oldToNewVtxPerm(lu->frontETree) ;
    lu->oldToNew   = IV_entries(lu->oldToNewIV) ;
    lu->newToOldIV = ETree_newToOldVtxPerm(lu->frontETree) ;
    ETree_permuteVertices(lu->frontETree, lu->oldToNewIV) ;

    InpMtx_permute(lu->mtxA, lu->oldToNew, lu->oldToNew) ; 
    if ( lu->symflag == SPOOLES_SYMMETRIC ) {
      InpMtx_mapToUpperTriangle(lu->mtxA) ; 
    }
    InpMtx_changeCoordType(lu->mtxA, INPMTX_BY_CHEVRONS) ;
    InpMtx_changeStorageMode(lu->mtxA, INPMTX_BY_VECTORS) ;

    /* get symbolic factorization */
    lu->symbfacIVL = SymbFac_initFromInpMtx(lu->frontETree, lu->mtxA) ;

    if ( lu->msglvl > 2 ) {
      fprintf(lu->msgFile, "\n\n old-to-new permutation vector") ;
      IV_writeForHumanEye(lu->oldToNewIV, lu->msgFile) ;
      fprintf(lu->msgFile, "\n\n new-to-old permutation vector") ;
      IV_writeForHumanEye(lu->newToOldIV, lu->msgFile) ;
      fprintf(lu->msgFile, "\n\n front tree after permutation") ;
      ETree_writeForHumanEye(lu->frontETree, lu->msgFile) ;
      fprintf(lu->msgFile, "\n\n input matrix after permutation") ;
      InpMtx_writeForHumanEye(lu->mtxA, lu->msgFile) ;
      fprintf(lu->msgFile, "\n\n symbolic factorization") ;
      IVL_writeForHumanEye(lu->symbfacIVL, lu->msgFile) ;
      fflush(lu->msgFile) ;
    }  

    lu->frontmtx   = FrontMtx_new() ;
    lu->mtxmanager = SubMtxManager_new() ;
    SubMtxManager_init(lu->mtxmanager, NO_LOCK, 0) ;

  } else { /* new num factorization using previously computed symbolic factor */ 
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
      for (i = ai[irow]; i<ai[irow+1]; i++) ivec1[i] = irow;
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
  } /* end of if( lu->flg == DIFFERENT_NONZERO_PATTERN) */

  FrontMtx_init(lu->frontmtx, lu->frontETree, lu->symbfacIVL, SPOOLES_REAL, lu->symflag, 
                FRONTMTX_DENSE_FRONTS, lu->pivotingflag, NO_LOCK, 0, NULL, 
                lu->mtxmanager, lu->msglvl, lu->msgFile) ;   
  

  if ( lu->symflag == SPOOLES_SYMMETRIC ) {
    if ( lu->patchAndGoFlag == 1 ) {
      lu->frontmtx->patchinfo = PatchAndGoInfo_new() ;
      PatchAndGoInfo_init(lu->frontmtx->patchinfo, 1, lu->toosmall, lu->fudge,
                       lu->storeids, lu->storevalues) ;
    } else if ( lu->patchAndGoFlag == 2 ) {
      lu->frontmtx->patchinfo = PatchAndGoInfo_new() ;
      PatchAndGoInfo_init(lu->frontmtx->patchinfo, 2, lu->toosmall, lu->fudge,
                       lu->storeids, lu->storevalues) ;
    }   
  }

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

  if ( lu->symflag == SPOOLES_SYMMETRIC ) {
    if ( lu->patchAndGoFlag == 1 ) {
      if ( lu->frontmtx->patchinfo->fudgeIV != NULL ) {
        if (lu->msglvl > 0 ){
          fprintf(lu->msgFile, "\n small pivots found at these locations") ;
          IV_writeForHumanEye(lu->frontmtx->patchinfo->fudgeIV, lu->msgFile) ;
        }
      }
      PatchAndGoInfo_free(lu->frontmtx->patchinfo) ;
    } else if ( lu->patchAndGoFlag == 2 ) {
      if (lu->msglvl > 0 ){
        if ( lu->frontmtx->patchinfo->fudgeIV != NULL ) {
          fprintf(lu->msgFile, "\n small pivots found at these locations") ;
          IV_writeForHumanEye(lu->frontmtx->patchinfo->fudgeIV, lu->msgFile) ;
        }
        if ( lu->frontmtx->patchinfo->fudgeDV != NULL ) {
          fprintf(lu->msgFile, "\n perturbations") ;
          DV_writeForHumanEye(lu->frontmtx->patchinfo->fudgeDV, lu->msgFile) ;
        }
      }
      PatchAndGoInfo_free(lu->frontmtx->patchinfo) ;
    }
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

#undef __FUNCT__  
#define __FUNCT__ "MatSeqSBAIJFactorInfo_Spooles"
int MatSeqAIJFactorInfo_Spooles(Mat A,PetscViewer viewer)
{
  Mat_Spooles      *lu = (Mat_Spooles*)A->spptr;  
  int              ierr;
  char             *s;

  PetscFunctionBegin;
  /* check if matrix is spooles type */
  if (A->ops->solve != MatSolve_SeqAIJ_Spooles) PetscFunctionReturn(0);

  ierr = PetscViewerASCIIPrintf(viewer,"Spooles run parameters:\n");CHKERRQ(ierr);

  switch (lu->symflag) {
  case 0: s = "SPOOLES_SYMMETRIC"; break;
  case 2: s = "SPOOLES_NONSYMMETRIC"; break; }
  ierr = PetscViewerASCIIPrintf(viewer,"  symmetryflag:   %s \n",s);CHKERRQ(ierr);

  switch (lu->pivotingflag) {
  case 0: s = "SPOOLES_NO_PIVOTING"; break;
  case 1: s = "SPOOLES_PIVOTING"; break; }
  ierr = PetscViewerASCIIPrintf(viewer,"  pivotingflag:   %s \n",s);CHKERRQ(ierr);

  ierr = PetscViewerASCIIPrintf(viewer,"  tau:            %g \n",lu->tau);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  seed:           %d \n",lu->seed);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  msglvl:         %d \n",lu->msglvl);CHKERRQ(ierr);

  switch (lu->ordering) {
  case 0: s = "BestOfNDandMS"; break;  
  case 1: s = "MMD"; break;
  case 2: s = "MS"; break;
  case 3: s = "ND"; break;
  }
  ierr = PetscViewerASCIIPrintf(viewer,"  ordering:       %s \n",s);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  maxdomainsize:  %d \n",lu->maxdomainsize);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  maxzeros:       %d \n",lu->maxzeros);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  maxsize:        %d \n",lu->maxsize);CHKERRQ(ierr);

  if ( lu->symflag == SPOOLES_SYMMETRIC ) {
    ierr = PetscViewerASCIIPrintf(viewer,"  patchAndGoFlag: %d \n",lu->patchAndGoFlag);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  fudge:          %g \n",lu->fudge);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  toosmall:       %g \n",lu->toosmall);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  storeids:       %d \n",lu->storeids);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  storevalues:    %d \n",lu->storevalues);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}
#endif
