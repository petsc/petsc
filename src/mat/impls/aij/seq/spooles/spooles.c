/*$Id: spooles.c,v 1.10 2001/08/15 15:56:50 bsmith Exp $*/
/* 
   Provides an interface to the Spooles serial sparse solver
*/
#include "src/mat/impls/aij/seq/aij.h"
#include "src/mat/impls/sbaij/seq/sbaij.h"
#include "src/mat/impls/aij/seq/spooles/spooles.h"

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatConvert_Spooles_Base"
int MatConvert_Spooles_Base(Mat A,MatType type,Mat *newmat) {
  /* This routine is only called to convert an unfactored PETSc-Spooles matrix */
  /* to its base PETSc type, so we will ignore 'MatType type'. */
  int         ierr;
  Mat         B=*newmat;
  Mat_Spooles *lu=(Mat_Spooles*)A->spptr;

  if (B != A) {
    /* This routine is inherited, so we know the type is correct. */
    ierr = MatDuplicate(A,MAT_COPY_VALUES,&B);CHKERRQ(ierr);
  } else {
    /* Reset the stashed function pointers set by inherited routines */
    B->ops->choleskyfactorsymbolic = lu->MatCholeskyFactorSymbolic;
    B->ops->lufactorsymbolic       = lu->MatLUFactorSymbolic;
    B->ops->view                   = lu->MatView;
    B->ops->assemblyend            = lu->MatAssemblyEnd;
    B->ops->destroy                = lu->MatDestroy;

    ierr = PetscObjectChangeTypeName((PetscObject)B,lu->basetype);CHKERRQ(ierr);
    ierr = PetscFree(lu);CHKERRQ(ierr);
  }
  *newmat = B;
  PetscFunctionReturn(0);
}    
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_SeqAIJ_Spooles"
int MatDestroy_SeqAIJ_Spooles(Mat A)
{
  Mat_Spooles *lu = (Mat_Spooles*)A->spptr; 
  int         ierr;
  
  PetscFunctionBegin;
 
  if (lu->CleanUpSpooles) {
    FrontMtx_free(lu->frontmtx) ;        
    IV_free(lu->newToOldIV) ;            
    IV_free(lu->oldToNewIV) ;            
    InpMtx_free(lu->mtxA) ;             
    ETree_free(lu->frontETree) ;          
    IVL_free(lu->symbfacIVL) ;         
    SubMtxManager_free(lu->mtxmanager) ; 
    Graph_free(lu->graph);
  }
  ierr = MatConvert_Spooles_Base(A,lu->basetype,&A);
  ierr = (*A->ops->destroy)(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSolve_SeqAIJ_Spooles"
int MatSolve_SeqAIJ_Spooles(Mat A,Vec b,Vec x)
{
  Mat_Spooles      *lu = (Mat_Spooles*)A->spptr;
  PetscScalar      *array;
  DenseMtx         *mtxY, *mtxX ;
  int              ierr,irow,neqns=A->n,nrow=A->m,*iv;
#if defined(PETSC_USE_COMPLEX)
  double           x_real,x_imag;
#else
  double           *entX;
#endif

  PetscFunctionBegin;

  mtxY = DenseMtx_new() ;
  DenseMtx_init(mtxY, lu->options.typeflag, 0, 0, nrow, 1, 1, nrow) ; /* column major */
  ierr = VecGetArray(b,&array);CHKERRQ(ierr);

  if (lu->options.useQR) {   /* copy b to mtxY */
    for ( irow = 0 ; irow < nrow; irow++ )  
#if !defined(PETSC_USE_COMPLEX)
      DenseMtx_setRealEntry(mtxY, irow, 0, *array++) ; 
#else
      DenseMtx_setComplexEntry(mtxY, irow, 0, PetscRealPart(array[irow]), PetscImaginaryPart(array[irow]));
#endif
  } else {                   /* copy permuted b to mtxY */
    iv = IV_entries(lu->oldToNewIV); 
    for ( irow = 0 ; irow < nrow; irow++ ) 
#if !defined(PETSC_USE_COMPLEX)
      DenseMtx_setRealEntry(mtxY, *iv++, 0, *array++) ; 
#else
      DenseMtx_setComplexEntry(mtxY,*iv++,0,PetscRealPart(array[irow]),PetscImaginaryPart(array[irow]));
#endif
  }
  ierr = VecRestoreArray(b,&array);CHKERRQ(ierr);

  mtxX = DenseMtx_new() ;
  DenseMtx_init(mtxX, lu->options.typeflag, 0, 0, neqns, 1, 1, neqns) ;
  if (lu->options.useQR) {
    FrontMtx_QR_solve(lu->frontmtx, lu->mtxA, mtxX, mtxY, lu->mtxmanager,
                  lu->cpus, lu->options.msglvl, lu->options.msgFile) ;
  } else {
    FrontMtx_solve(lu->frontmtx, mtxX, mtxY, lu->mtxmanager, 
                 lu->cpus, lu->options.msglvl, lu->options.msgFile) ;
  }
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

#if !defined(PETSC_USE_COMPLEX)
  entX = DenseMtx_entries(mtxX);
  DVcopy(neqns, array, entX);
#else
  for (irow=0; irow<nrow; irow++){
    DenseMtx_complexEntry(mtxX,irow,0,&x_real,&x_imag);
    array[irow] = x_real+x_imag*PETSC_i;   
  }
#endif

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
  IVL                *adjIVL;
  int                ierr,nz,nrow=A->m,irow,nedges,neqns=A->n,
                     *ai,*aj,i,j,*diag;
  PetscScalar        *av;
  double             cputotal,facops;
#if defined(PETSC_USE_COMPLEX)
  int                nz_row,*aj_tmp;
  PetscScalar        *av_tmp;
#else
  int                *ivec1, *ivec2;
  double             *dvec;
#endif
  PetscTruth         isAIJ;
  
  PetscFunctionBegin;
  if (lu->flg == DIFFERENT_NONZERO_PATTERN) { /* first numeric factorization */      
    (*F)->ops->solve   = MatSolve_SeqAIJ_Spooles;
    (*F)->ops->destroy = MatDestroy_SeqAIJ_Spooles;  
    (*F)->assembled    = PETSC_TRUE; 
    
    /* set Spooles options */
    ierr = SetSpoolesOptions(A, &lu->options);CHKERRQ(ierr); 

    lu->mtxA = InpMtx_new() ;
  }

  /* copy A to Spooles' InpMtx object */
  ierr = PetscTypeCompare((PetscObject)A,MATSEQAIJSPOOLES,&isAIJ);CHKERRQ(ierr);
  if (isAIJ){
    Mat_SeqAIJ   *mat = (Mat_SeqAIJ*)A->data;
    ai=mat->i; aj=mat->j; av=mat->a;
    if (lu->options.symflag == SPOOLES_NONSYMMETRIC) {
      nz=mat->nz;
    } else { /* SPOOLES_SYMMETRIC || SPOOLES_HERMITIAN */
      nz=(mat->nz + A->m)/2;
      if (!mat->diag){
        ierr = MatMarkDiagonal_SeqAIJ(A);CHKERRQ(ierr); 
      }
      diag=mat->diag;
    }
  } else { /* A is SBAIJ */
      Mat_SeqSBAIJ *mat = (Mat_SeqSBAIJ*)A->data;
      ai=mat->i; aj=mat->j; av=mat->a;
      nz=mat->s_nz;
  } 
  InpMtx_init(lu->mtxA, INPMTX_BY_ROWS, lu->options.typeflag, nz, 0) ;
 
#if defined(PETSC_USE_COMPLEX)
    for (irow=0; irow<nrow; irow++) {
      if ( lu->options.symflag == SPOOLES_NONSYMMETRIC || !isAIJ){
        nz_row = ai[irow+1] - ai[irow];
        aj_tmp = aj + ai[irow];
        av_tmp = av + ai[irow];
      } else {
        nz_row = ai[irow+1] - diag[irow];
        aj_tmp = aj + diag[irow];
        av_tmp = av + diag[irow];
      }
      for (i=0; i<nz_row; i++){
        InpMtx_inputComplexEntry(lu->mtxA, irow, *aj_tmp++,PetscRealPart(*av_tmp),PetscImaginaryPart(*av_tmp));
        av_tmp++;
      }
    }
#else
    ivec1 = InpMtx_ivec1(lu->mtxA); 
    ivec2 = InpMtx_ivec2(lu->mtxA);
    dvec  = InpMtx_dvec(lu->mtxA);
    if ( lu->options.symflag == SPOOLES_NONSYMMETRIC || !isAIJ){
      for (irow = 0; irow < nrow; irow++){
        for (i = ai[irow]; i<ai[irow+1]; i++) ivec1[i] = irow;
      }
      IVcopy(nz, ivec2, aj);
      DVcopy(nz, dvec, av);
    } else { 
      nz = 0;
      for (irow = 0; irow < nrow; irow++){
        for (j = diag[irow]; j<ai[irow+1]; j++) {
          ivec1[nz] = irow;
          ivec2[nz] = aj[j];
          dvec[nz]  = av[j];
          nz++;
        }
      }
    }
    InpMtx_inputRealTriples(lu->mtxA, nz, ivec1, ivec2, dvec); 
#endif

  InpMtx_changeStorageMode(lu->mtxA, INPMTX_BY_VECTORS) ; 
  if ( lu->options.msglvl > 0 ) {
    printf("\n\n input matrix") ;
    fprintf(lu->options.msgFile, "\n\n input matrix") ;
    InpMtx_writeForHumanEye(lu->mtxA, lu->options.msgFile) ;
    fflush(lu->options.msgFile) ;
  }

  if ( lu->flg == DIFFERENT_NONZERO_PATTERN){ /* first numeric factorization */  
    /*---------------------------------------------------
    find a low-fill ordering
         (1) create the Graph object
         (2) order the graph 
    -------------------------------------------------------*/  
    if (lu->options.useQR){
      adjIVL = InpMtx_adjForATA(lu->mtxA) ;
    } else {
      adjIVL = InpMtx_fullAdjacency(lu->mtxA) ;
    }
    nedges = IVL_tsize(adjIVL) ;

    lu->graph = Graph_new() ;
    Graph_init2(lu->graph, 0, neqns, 0, nedges, neqns, nedges, adjIVL, NULL, NULL) ;
    if ( lu->options.msglvl > 2 ) {
      if (lu->options.useQR){
        fprintf(lu->options.msgFile, "\n\n graph of A^T A") ;
      } else {
        fprintf(lu->options.msgFile, "\n\n graph of the input matrix") ;
      }
      Graph_writeForHumanEye(lu->graph, lu->options.msgFile) ;
      fflush(lu->options.msgFile) ;
    }

    switch (lu->options.ordering) {
    case 0:
      lu->frontETree = orderViaBestOfNDandMS(lu->graph,
                     lu->options.maxdomainsize, lu->options.maxzeros, lu->options.maxsize,
                     lu->options.seed, lu->options.msglvl, lu->options.msgFile); break;
    case 1:
      lu->frontETree = orderViaMMD(lu->graph,lu->options.seed,lu->options.msglvl,lu->options.msgFile); break;
    case 2:
      lu->frontETree = orderViaMS(lu->graph, lu->options.maxdomainsize,
                     lu->options.seed,lu->options.msglvl,lu->options.msgFile); break;
    case 3:
      lu->frontETree = orderViaND(lu->graph, lu->options.maxdomainsize, 
                     lu->options.seed,lu->options.msglvl,lu->options.msgFile); break;
    default:
      SETERRQ(1,"Unknown Spooles's ordering");
    }

    if ( lu->options.msglvl > 0 ) {
      fprintf(lu->options.msgFile, "\n\n front tree from ordering") ;
      ETree_writeForHumanEye(lu->frontETree, lu->options.msgFile) ;
      fflush(lu->options.msgFile) ;
    }
  
    /* get the permutation, permute the front tree */
    lu->oldToNewIV = ETree_oldToNewVtxPerm(lu->frontETree) ;
    lu->oldToNew   = IV_entries(lu->oldToNewIV) ;
    lu->newToOldIV = ETree_newToOldVtxPerm(lu->frontETree) ;
    if (!lu->options.useQR) ETree_permuteVertices(lu->frontETree, lu->oldToNewIV) ;

    /* permute the matrix */
    if (lu->options.useQR){
      InpMtx_permute(lu->mtxA, NULL, lu->oldToNew) ;
    } else {
      InpMtx_permute(lu->mtxA, lu->oldToNew, lu->oldToNew) ; 
      if ( lu->options.symflag == SPOOLES_SYMMETRIC) {
        InpMtx_mapToUpperTriangle(lu->mtxA) ; 
      }
#if defined(PETSC_USE_COMPLEX)
      if ( lu->options.symflag == SPOOLES_HERMITIAN ) {
        InpMtx_mapToUpperTriangleH(lu->mtxA) ; 
      }
#endif
      InpMtx_changeCoordType(lu->mtxA, INPMTX_BY_CHEVRONS) ;
    }
    InpMtx_changeStorageMode(lu->mtxA, INPMTX_BY_VECTORS) ;

    /* get symbolic factorization */
    if (lu->options.useQR){
      lu->symbfacIVL = SymbFac_initFromGraph(lu->frontETree, lu->graph) ;
      IVL_overwrite(lu->symbfacIVL, lu->oldToNewIV) ;
      IVL_sortUp(lu->symbfacIVL) ;
      ETree_permuteVertices(lu->frontETree, lu->oldToNewIV) ;
    } else {
      lu->symbfacIVL = SymbFac_initFromInpMtx(lu->frontETree, lu->mtxA) ;
    }
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

    if (lu->options.pivotingflag) { /* different FrontMtx is required */
      FrontMtx_free(lu->frontmtx) ;   
      lu->frontmtx   = FrontMtx_new() ;
    } else {
      FrontMtx_clearData (lu->frontmtx); 
    }

    SubMtxManager_free(lu->mtxmanager) ;  
    lu->mtxmanager = SubMtxManager_new() ;
    SubMtxManager_init(lu->mtxmanager, NO_LOCK, 0) ;

    /* permute mtxA */
    if (lu->options.useQR){
      InpMtx_permute(lu->mtxA, NULL, lu->oldToNew) ;
    } else {
      InpMtx_permute(lu->mtxA, lu->oldToNew, lu->oldToNew) ; 
      if ( lu->options.symflag == SPOOLES_SYMMETRIC ) {
        InpMtx_mapToUpperTriangle(lu->mtxA) ; 
      }
      InpMtx_changeCoordType(lu->mtxA, INPMTX_BY_CHEVRONS) ;
    }
    InpMtx_changeStorageMode(lu->mtxA, INPMTX_BY_VECTORS) ;
    if ( lu->options.msglvl > 2 ) {
      fprintf(lu->options.msgFile, "\n\n input matrix after permutation") ;
      InpMtx_writeForHumanEye(lu->mtxA, lu->options.msgFile) ; 
    } 
  } /* end of if( lu->flg == DIFFERENT_NONZERO_PATTERN) */
  
  if (lu->options.useQR){
    FrontMtx_init(lu->frontmtx, lu->frontETree, lu->symbfacIVL, lu->options.typeflag, 
                 SPOOLES_SYMMETRIC, FRONTMTX_DENSE_FRONTS, 
                 SPOOLES_NO_PIVOTING, NO_LOCK, 0, NULL,
                 lu->mtxmanager, lu->options.msglvl, lu->options.msgFile) ;
  } else {
    FrontMtx_init(lu->frontmtx, lu->frontETree, lu->symbfacIVL, lu->options.typeflag, lu->options.symflag, 
                FRONTMTX_DENSE_FRONTS, lu->options.pivotingflag, NO_LOCK, 0, NULL, 
                lu->mtxmanager, lu->options.msglvl, lu->options.msgFile) ;   
  }

  if ( lu->options.symflag == SPOOLES_SYMMETRIC ) {  /* || SPOOLES_HERMITIAN ? */
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
  if (lu->options.useQR){
    facops = 0.0 ; 
    FrontMtx_QR_factor(lu->frontmtx, lu->mtxA, chvmanager, 
                   lu->cpus, &facops, lu->options.msglvl, lu->options.msgFile) ;
    if ( lu->options.msglvl > 1 ) {
      fprintf(lu->options.msgFile, "\n\n factor matrix") ;
      fprintf(lu->options.msgFile, "\n facops = %9.2f", facops) ;
    }
  } else {
    IVfill(20, lu->stats, 0) ;
    rootchv = FrontMtx_factorInpMtx(lu->frontmtx, lu->mtxA, lu->options.tau, 0.0, 
            chvmanager, &ierr, lu->cpus,lu->stats,lu->options.msglvl,lu->options.msgFile) ; 
    if ( rootchv != NULL ) SETERRQ(1,"\n matrix found to be singular");    
    if ( ierr >= 0 ) SETERRQ1(1,"\n error encountered at front %d", ierr);
    
    if(lu->options.FrontMtxInfo){
      PetscPrintf(PETSC_COMM_SELF,"\n %8d pivots, %8d pivot tests, %8d delayed rows and columns\n",\
               lu->stats[0], lu->stats[1], lu->stats[2]);
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
  }
  ChvManager_free(chvmanager) ;

  if ( lu->options.msglvl > 0 ) {
    fprintf(lu->options.msgFile, "\n\n factor matrix") ;
    FrontMtx_writeForHumanEye(lu->frontmtx, lu->options.msgFile) ;
    fflush(lu->options.msgFile) ;
  }

  if ( lu->options.symflag == SPOOLES_SYMMETRIC ) { /* || SPOOLES_HERMITIAN ? */
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

  /* post-process the factorization */
  FrontMtx_postProcess(lu->frontmtx, lu->options.msglvl, lu->options.msgFile) ;
  if ( lu->options.msglvl > 2 ) {
    fprintf(lu->options.msgFile, "\n\n factor matrix after post-processing") ;
    FrontMtx_writeForHumanEye(lu->frontmtx, lu->options.msgFile) ;
    fflush(lu->options.msgFile) ;
  }

  lu->flg = SAME_NONZERO_PATTERN;
  lu->CleanUpSpooles = PETSC_TRUE;
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatConvert_SeqAIJ_Spooles"
int MatConvert_SeqAIJ_Spooles(Mat A,MatType type,Mat *newmat) {
  /* This routine is only called to convert a MATSEQAIJ matrix */
  /* to a MATSEQAIJSPOOLES matrix, so we will ignore 'MatType type'. */
  int         ierr;
  Mat         B=*newmat;
  Mat_Spooles *lu;

  PetscFunctionBegin;
  if (B != A) {
    /* This routine is inherited, so we know the type is correct. */
    ierr = MatDuplicate(A,MAT_COPY_VALUES,&B);CHKERRQ(ierr);
  }
  ierr     = PetscNew(Mat_Spooles,&lu);CHKERRQ(ierr); 
  B->spptr = (void*)lu;
  ierr     = PetscOptionsHasName(A->prefix,"-mat_seqaij_spooles_qr",&(lu->useQR));CHKERRQ(ierr);

  lu->basetype                     = MATSEQAIJ;
  lu->CleanUpSpooles               = PETSC_FALSE;
  lu->MatCholeskyFactorSymbolic    = A->ops->choleskyfactorsymbolic;
  lu->MatLUFactorSymbolic          = A->ops->lufactorsymbolic; 
  lu->MatView                      = A->ops->view;
  lu->MatAssemblyEnd               = A->ops->assemblyend;
  lu->MatDestroy                   = A->ops->destroy;
  if (lu->useQR){
    B->ops->lufactorsymbolic       = MatQRFactorSymbolic_SeqAIJ_Spooles;  
  } else {
    B->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_SeqAIJ_Spooles;
    B->ops->lufactorsymbolic       = MatLUFactorSymbolic_SeqAIJ_Spooles; 
  }
  B->ops->view                     = MatView_SeqAIJ_Spooles;
  B->ops->assemblyend              = MatAssemblyEnd_SeqAIJ_Spooles;
  B->ops->destroy                  = MatDestroy_SeqAIJ_Spooles;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatConvert_spooles_seqaij_C",
                                           "MatConvert_Spooles_Base",MatConvert_Spooles_Base);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatConvert_seqaij_spooles_C",
                                           "MatConvert_SeqAIJ_Spooles",MatConvert_SeqAIJ_Spooles);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)B,MATSEQAIJSPOOLES);CHKERRQ(ierr);
  *newmat = B;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatCreate_SeqAIJ_Spooles"
int MatCreate_SeqAIJ_Spooles(Mat A) {
  int         ierr;

  PetscFunctionBegin;
  ierr = MatSetType(A,MATSEQAIJ);CHKERRQ(ierr);
  ierr = MatConvert_SeqAIJ_Spooles(A,MATSEQAIJSPOOLES,&A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
