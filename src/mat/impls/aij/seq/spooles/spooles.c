/* 
   Provides an interface to the Spooles serial sparse solver
*/
#include "src/mat/impls/aij/seq/aij.h"
#include "src/mat/impls/sbaij/seq/sbaij.h"
#include "src/mat/impls/aij/seq/spooles/spooles.h"

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatConvert_Spooles_Base"
PetscErrorCode MatConvert_Spooles_Base(Mat A,const MatType type,MatReuse reuse,Mat *newmat) {
  /* This routine is only called to convert an unfactored PETSc-Spooles matrix */
  /* to its base PETSc type, so we will ignore 'MatType type'. */
  PetscErrorCode ierr;
  Mat            B=*newmat;
  Mat_Spooles    *lu=(Mat_Spooles*)A->spptr;
  void           (*f)(void);

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) {
    ierr = MatDuplicate(A,MAT_COPY_VALUES,&B);CHKERRQ(ierr);
  }
  /* Reset the stashed function pointers set by inherited routines */
  B->ops->duplicate              = lu->MatDuplicate;
  B->ops->choleskyfactorsymbolic = lu->MatCholeskyFactorSymbolic;
  B->ops->lufactorsymbolic       = lu->MatLUFactorSymbolic;
  B->ops->view                   = lu->MatView;
  B->ops->assemblyend            = lu->MatAssemblyEnd;
  B->ops->destroy                = lu->MatDestroy;

  ierr = PetscObjectQueryFunction((PetscObject)B,"MatMPISBAIJSetPreallocation_C",&f);CHKERRQ(ierr);
  if (f) {
    ierr = PetscObjectComposeFunction((PetscObject)B,"MatMPISBAIJSetPreallocation_C","",(FCNVOID)lu->MatPreallocate);CHKERRQ(ierr);
  }
  ierr = PetscFree(lu);CHKERRQ(ierr);

  ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_seqaijspooles_seqaij_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_seqaij_seqaijspooles_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_mpiaijspooles_mpiaij_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_mpiaij_mpiaijspooles_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_seqsbaijspooles_seqsbaij_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_seqsbaij_seqsbaijspooles_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_mpisbaijspooles_mpisbaij_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_mpisbaij_mpisbaijspooles_C","",PETSC_NULL);CHKERRQ(ierr);

  ierr = PetscObjectChangeTypeName((PetscObject)B,type);CHKERRQ(ierr);
  *newmat = B;
  PetscFunctionReturn(0);
}    
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_SeqAIJSpooles"
PetscErrorCode MatDestroy_SeqAIJSpooles(Mat A)
{
  Mat_Spooles    *lu = (Mat_Spooles*)A->spptr; 
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  if (lu->CleanUpSpooles) {
    FrontMtx_free(lu->frontmtx);        
    IV_free(lu->newToOldIV);            
    IV_free(lu->oldToNewIV);            
    InpMtx_free(lu->mtxA);             
    ETree_free(lu->frontETree);          
    IVL_free(lu->symbfacIVL);         
    SubMtxManager_free(lu->mtxmanager); 
    Graph_free(lu->graph);
  }
  ierr = MatConvert_Spooles_Base(A,lu->basetype,MAT_REUSE_MATRIX,&A);
  ierr = (*A->ops->destroy)(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSolve_SeqAIJSpooles"
PetscErrorCode MatSolve_SeqAIJSpooles(Mat A,Vec b,Vec x)
{
  Mat_Spooles      *lu = (Mat_Spooles*)A->spptr;
  PetscScalar      *array;
  DenseMtx         *mtxY, *mtxX ;
  PetscErrorCode   ierr;
  PetscInt         irow,neqns=A->n,nrow=A->m,*iv;
#if defined(PETSC_USE_COMPLEX)
  double           x_real,x_imag;
#else
  double           *entX;
#endif

  PetscFunctionBegin;
  mtxY = DenseMtx_new();
  DenseMtx_init(mtxY, lu->options.typeflag, 0, 0, nrow, 1, 1, nrow); /* column major */
  ierr = VecGetArray(b,&array);CHKERRQ(ierr);

  if (lu->options.useQR) {   /* copy b to mtxY */
    for ( irow = 0 ; irow < nrow; irow++ )  
#if !defined(PETSC_USE_COMPLEX)
      DenseMtx_setRealEntry(mtxY, irow, 0, *array++); 
#else
      DenseMtx_setComplexEntry(mtxY, irow, 0, PetscRealPart(array[irow]), PetscImaginaryPart(array[irow]));
#endif
  } else {                   /* copy permuted b to mtxY */
    iv = IV_entries(lu->oldToNewIV); 
    for ( irow = 0 ; irow < nrow; irow++ ) 
#if !defined(PETSC_USE_COMPLEX)
      DenseMtx_setRealEntry(mtxY, *iv++, 0, *array++); 
#else
      DenseMtx_setComplexEntry(mtxY,*iv++,0,PetscRealPart(array[irow]),PetscImaginaryPart(array[irow]));
#endif
  }
  ierr = VecRestoreArray(b,&array);CHKERRQ(ierr);

  mtxX = DenseMtx_new();
  DenseMtx_init(mtxX, lu->options.typeflag, 0, 0, neqns, 1, 1, neqns);
  if (lu->options.useQR) {
    FrontMtx_QR_solve(lu->frontmtx, lu->mtxA, mtxX, mtxY, lu->mtxmanager,
                  lu->cpus, lu->options.msglvl, lu->options.msgFile);
  } else {
    FrontMtx_solve(lu->frontmtx, mtxX, mtxY, lu->mtxmanager, 
                 lu->cpus, lu->options.msglvl, lu->options.msgFile);
  }
  if ( lu->options.msglvl > 2 ) {
    ierr = PetscFPrintf(PETSC_COMM_SELF,lu->options.msgFile, "\n\n right hand side matrix after permutation");CHKERRQ(ierr);
    DenseMtx_writeForHumanEye(mtxY, lu->options.msgFile); 
    ierr = PetscFPrintf(PETSC_COMM_SELF,lu->options.msgFile, "\n\n solution matrix in new ordering");CHKERRQ(ierr);
    DenseMtx_writeForHumanEye(mtxX, lu->options.msgFile);
    fflush(lu->options.msgFile);
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
  DenseMtx_free(mtxX);
  DenseMtx_free(mtxY);
  PetscFunctionReturn(0);
}

#undef __FUNCT__   
#define __FUNCT__ "MatFactorNumeric_SeqAIJSpooles"
PetscErrorCode MatFactorNumeric_SeqAIJSpooles(Mat A,MatFactorInfo *info,Mat *F)
{  
  Mat_Spooles        *lu = (Mat_Spooles*)(*F)->spptr;
  ChvManager         *chvmanager ;
  Chv                *rootchv ;
  IVL                *adjIVL;
  PetscErrorCode     ierr;
  PetscInt           nz,nrow=A->m,irow,nedges,neqns=A->n,*ai,*aj,i,*diag=0,fierr;
  PetscScalar        *av;
  double             cputotal,facops;
#if defined(PETSC_USE_COMPLEX)
  PetscInt           nz_row,*aj_tmp;
  PetscScalar        *av_tmp;
#else
  PetscInt           *ivec1,*ivec2,j;
  double             *dvec;
#endif
  PetscTruth         isAIJ,isSeqAIJ;
  
  PetscFunctionBegin;
  if (lu->flg == DIFFERENT_NONZERO_PATTERN) { /* first numeric factorization */      
    (*F)->ops->solve   = MatSolve_SeqAIJSpooles;
    (*F)->ops->destroy = MatDestroy_SeqAIJSpooles;  
    (*F)->assembled    = PETSC_TRUE; 
    
    /* set Spooles options */
    ierr = SetSpoolesOptions(A, &lu->options);CHKERRQ(ierr); 

    lu->mtxA = InpMtx_new();
  }

  /* copy A to Spooles' InpMtx object */
  ierr = PetscTypeCompare((PetscObject)A,MATSEQAIJSPOOLES,&isSeqAIJ);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)A,MATAIJSPOOLES,&isAIJ);CHKERRQ(ierr);
  if (isSeqAIJ || isAIJ){
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
      nz=mat->nz;
  } 
  InpMtx_init(lu->mtxA, INPMTX_BY_ROWS, lu->options.typeflag, nz, 0);
 
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

  InpMtx_changeStorageMode(lu->mtxA, INPMTX_BY_VECTORS); 
  if ( lu->options.msglvl > 0 ) {
    printf("\n\n input matrix");
    ierr = PetscFPrintf(PETSC_COMM_SELF,lu->options.msgFile, "\n\n input matrix");CHKERRQ(ierr);
    InpMtx_writeForHumanEye(lu->mtxA, lu->options.msgFile);
    fflush(lu->options.msgFile);
  }

  if ( lu->flg == DIFFERENT_NONZERO_PATTERN){ /* first numeric factorization */  
    /*---------------------------------------------------
    find a low-fill ordering
         (1) create the Graph object
         (2) order the graph 
    -------------------------------------------------------*/  
    if (lu->options.useQR){
      adjIVL = InpMtx_adjForATA(lu->mtxA);
    } else {
      adjIVL = InpMtx_fullAdjacency(lu->mtxA);
    }
    nedges = IVL_tsize(adjIVL);

    lu->graph = Graph_new();
    Graph_init2(lu->graph, 0, neqns, 0, nedges, neqns, nedges, adjIVL, NULL, NULL);
    if ( lu->options.msglvl > 2 ) {
      if (lu->options.useQR){
        ierr = PetscFPrintf(PETSC_COMM_SELF,lu->options.msgFile, "\n\n graph of A^T A");CHKERRQ(ierr);
      } else {
        ierr = PetscFPrintf(PETSC_COMM_SELF,lu->options.msgFile, "\n\n graph of the input matrix");CHKERRQ(ierr);
      }
      Graph_writeForHumanEye(lu->graph, lu->options.msgFile);
      fflush(lu->options.msgFile);
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
      SETERRQ(PETSC_ERR_ARG_WRONG,"Unknown Spooles's ordering");
    }

    if ( lu->options.msglvl > 0 ) {
      ierr = PetscFPrintf(PETSC_COMM_SELF,lu->options.msgFile, "\n\n front tree from ordering");CHKERRQ(ierr);
      ETree_writeForHumanEye(lu->frontETree, lu->options.msgFile);
      fflush(lu->options.msgFile);
    }
  
    /* get the permutation, permute the front tree */
    lu->oldToNewIV = ETree_oldToNewVtxPerm(lu->frontETree);
    lu->oldToNew   = IV_entries(lu->oldToNewIV);
    lu->newToOldIV = ETree_newToOldVtxPerm(lu->frontETree);
    if (!lu->options.useQR) ETree_permuteVertices(lu->frontETree, lu->oldToNewIV);

    /* permute the matrix */
    if (lu->options.useQR){
      InpMtx_permute(lu->mtxA, NULL, lu->oldToNew);
    } else {
      InpMtx_permute(lu->mtxA, lu->oldToNew, lu->oldToNew); 
      if ( lu->options.symflag == SPOOLES_SYMMETRIC) {
        InpMtx_mapToUpperTriangle(lu->mtxA); 
      }
#if defined(PETSC_USE_COMPLEX)
      if ( lu->options.symflag == SPOOLES_HERMITIAN ) {
        InpMtx_mapToUpperTriangleH(lu->mtxA); 
      }
#endif
      InpMtx_changeCoordType(lu->mtxA, INPMTX_BY_CHEVRONS);
    }
    InpMtx_changeStorageMode(lu->mtxA, INPMTX_BY_VECTORS);

    /* get symbolic factorization */
    if (lu->options.useQR){
      lu->symbfacIVL = SymbFac_initFromGraph(lu->frontETree, lu->graph);
      IVL_overwrite(lu->symbfacIVL, lu->oldToNewIV);
      IVL_sortUp(lu->symbfacIVL);
      ETree_permuteVertices(lu->frontETree, lu->oldToNewIV);
    } else {
      lu->symbfacIVL = SymbFac_initFromInpMtx(lu->frontETree, lu->mtxA);
    }
    if ( lu->options.msglvl > 2 ) {
      ierr = PetscFPrintf(PETSC_COMM_SELF,lu->options.msgFile, "\n\n old-to-new permutation vector");CHKERRQ(ierr);
      IV_writeForHumanEye(lu->oldToNewIV, lu->options.msgFile);
      ierr = PetscFPrintf(PETSC_COMM_SELF,lu->options.msgFile, "\n\n new-to-old permutation vector");CHKERRQ(ierr);
      IV_writeForHumanEye(lu->newToOldIV, lu->options.msgFile);
      ierr = PetscFPrintf(PETSC_COMM_SELF,lu->options.msgFile, "\n\n front tree after permutation");CHKERRQ(ierr);
      ETree_writeForHumanEye(lu->frontETree, lu->options.msgFile);
      ierr = PetscFPrintf(PETSC_COMM_SELF,lu->options.msgFile, "\n\n input matrix after permutation");CHKERRQ(ierr);
      InpMtx_writeForHumanEye(lu->mtxA, lu->options.msgFile);
      ierr = PetscFPrintf(PETSC_COMM_SELF,lu->options.msgFile, "\n\n symbolic factorization");CHKERRQ(ierr);
      IVL_writeForHumanEye(lu->symbfacIVL, lu->options.msgFile);
      fflush(lu->options.msgFile);
    }  

    lu->frontmtx   = FrontMtx_new();
    lu->mtxmanager = SubMtxManager_new();
    SubMtxManager_init(lu->mtxmanager, NO_LOCK, 0);

  } else { /* new num factorization using previously computed symbolic factor */ 

    if (lu->options.pivotingflag) { /* different FrontMtx is required */
      FrontMtx_free(lu->frontmtx);   
      lu->frontmtx   = FrontMtx_new();
    } else {
      FrontMtx_clearData (lu->frontmtx); 
    }

    SubMtxManager_free(lu->mtxmanager);  
    lu->mtxmanager = SubMtxManager_new();
    SubMtxManager_init(lu->mtxmanager, NO_LOCK, 0);

    /* permute mtxA */
    if (lu->options.useQR){
      InpMtx_permute(lu->mtxA, NULL, lu->oldToNew);
    } else {
      InpMtx_permute(lu->mtxA, lu->oldToNew, lu->oldToNew); 
      if ( lu->options.symflag == SPOOLES_SYMMETRIC ) {
        InpMtx_mapToUpperTriangle(lu->mtxA); 
      }
      InpMtx_changeCoordType(lu->mtxA, INPMTX_BY_CHEVRONS);
    }
    InpMtx_changeStorageMode(lu->mtxA, INPMTX_BY_VECTORS);
    if ( lu->options.msglvl > 2 ) {
      ierr = PetscFPrintf(PETSC_COMM_SELF,lu->options.msgFile, "\n\n input matrix after permutation");CHKERRQ(ierr);
      InpMtx_writeForHumanEye(lu->mtxA, lu->options.msgFile); 
    } 
  } /* end of if( lu->flg == DIFFERENT_NONZERO_PATTERN) */
  
  if (lu->options.useQR){
    FrontMtx_init(lu->frontmtx, lu->frontETree, lu->symbfacIVL, lu->options.typeflag, 
                 SPOOLES_SYMMETRIC, FRONTMTX_DENSE_FRONTS, 
                 SPOOLES_NO_PIVOTING, NO_LOCK, 0, NULL,
                 lu->mtxmanager, lu->options.msglvl, lu->options.msgFile);
  } else {
    FrontMtx_init(lu->frontmtx, lu->frontETree, lu->symbfacIVL, lu->options.typeflag, lu->options.symflag, 
                FRONTMTX_DENSE_FRONTS, lu->options.pivotingflag, NO_LOCK, 0, NULL, 
                lu->mtxmanager, lu->options.msglvl, lu->options.msgFile);   
  }

  if ( lu->options.symflag == SPOOLES_SYMMETRIC ) {  /* || SPOOLES_HERMITIAN ? */
    if ( lu->options.patchAndGoFlag == 1 ) {
      lu->frontmtx->patchinfo = PatchAndGoInfo_new();
      PatchAndGoInfo_init(lu->frontmtx->patchinfo, 1, lu->options.toosmall, lu->options.fudge,
                       lu->options.storeids, lu->options.storevalues);
    } else if ( lu->options.patchAndGoFlag == 2 ) {
      lu->frontmtx->patchinfo = PatchAndGoInfo_new();
      PatchAndGoInfo_init(lu->frontmtx->patchinfo, 2, lu->options.toosmall, lu->options.fudge,
                       lu->options.storeids, lu->options.storevalues);
    }   
  }

  /* numerical factorization */
  chvmanager = ChvManager_new();
  ChvManager_init(chvmanager, NO_LOCK, 1);
  DVfill(10, lu->cpus, 0.0);
  if (lu->options.useQR){
    facops = 0.0 ; 
    FrontMtx_QR_factor(lu->frontmtx, lu->mtxA, chvmanager, 
                   lu->cpus, &facops, lu->options.msglvl, lu->options.msgFile);
    if ( lu->options.msglvl > 1 ) {
      ierr = PetscFPrintf(PETSC_COMM_SELF,lu->options.msgFile, "\n\n factor matrix");CHKERRQ(ierr);
      ierr = PetscFPrintf(PETSC_COMM_SELF,lu->options.msgFile, "\n facops = %9.2f", facops);CHKERRQ(ierr);
    }
  } else {
    IVfill(20, lu->stats, 0);
    rootchv = FrontMtx_factorInpMtx(lu->frontmtx, lu->mtxA, lu->options.tau, 0.0, 
            chvmanager, &fierr, lu->cpus,lu->stats,lu->options.msglvl,lu->options.msgFile); 
    if (rootchv) SETERRQ(PETSC_ERR_MAT_LU_ZRPVT,"\n matrix found to be singular");    
    if (fierr >= 0) SETERRQ1(PETSC_ERR_LIB,"\n error encountered at front %D", fierr);
    
    if(lu->options.FrontMtxInfo){
      ierr = PetscPrintf(PETSC_COMM_SELF,"\n %8d pivots, %8d pivot tests, %8d delayed rows and columns\n",lu->stats[0], lu->stats[1], lu->stats[2]);CHKERRQ(ierr);
      cputotal = lu->cpus[8] ;
      if ( cputotal > 0.0 ) {
        ierr = PetscPrintf(PETSC_COMM_SELF,
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
	   lu->cpus[7], 100.*lu->cpus[7]/cputotal, cputotal);CHKERRQ(ierr);
      }
    }
  }
  ChvManager_free(chvmanager);

  if ( lu->options.msglvl > 0 ) {
    ierr = PetscFPrintf(PETSC_COMM_SELF,lu->options.msgFile, "\n\n factor matrix");CHKERRQ(ierr);
    FrontMtx_writeForHumanEye(lu->frontmtx, lu->options.msgFile);
    fflush(lu->options.msgFile);
  }

  if ( lu->options.symflag == SPOOLES_SYMMETRIC ) { /* || SPOOLES_HERMITIAN ? */
    if ( lu->options.patchAndGoFlag == 1 ) {
      if ( lu->frontmtx->patchinfo->fudgeIV != NULL ) {
        if (lu->options.msglvl > 0 ){
          ierr = PetscFPrintf(PETSC_COMM_SELF,lu->options.msgFile, "\n small pivots found at these locations");CHKERRQ(ierr);
          IV_writeForHumanEye(lu->frontmtx->patchinfo->fudgeIV, lu->options.msgFile);
        }
      }
      PatchAndGoInfo_free(lu->frontmtx->patchinfo);
    } else if ( lu->options.patchAndGoFlag == 2 ) {
      if (lu->options.msglvl > 0 ){
        if ( lu->frontmtx->patchinfo->fudgeIV != NULL ) {
          ierr = PetscFPrintf(PETSC_COMM_SELF,lu->options.msgFile, "\n small pivots found at these locations");CHKERRQ(ierr);
          IV_writeForHumanEye(lu->frontmtx->patchinfo->fudgeIV, lu->options.msgFile);
        }
        if ( lu->frontmtx->patchinfo->fudgeDV != NULL ) {
          ierr = PetscFPrintf(PETSC_COMM_SELF,lu->options.msgFile, "\n perturbations");CHKERRQ(ierr);
          DV_writeForHumanEye(lu->frontmtx->patchinfo->fudgeDV, lu->options.msgFile);
        }
      }
      PatchAndGoInfo_free(lu->frontmtx->patchinfo);
    }
  }

  /* post-process the factorization */
  FrontMtx_postProcess(lu->frontmtx, lu->options.msglvl, lu->options.msgFile);
  if ( lu->options.msglvl > 2 ) {
    ierr = PetscFPrintf(PETSC_COMM_SELF,lu->options.msgFile, "\n\n factor matrix after post-processing");CHKERRQ(ierr);
    FrontMtx_writeForHumanEye(lu->frontmtx, lu->options.msgFile);
    fflush(lu->options.msgFile);
  }

  lu->flg = SAME_NONZERO_PATTERN;
  lu->CleanUpSpooles = PETSC_TRUE;
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatConvert_SeqAIJ_SeqAIJSpooles"
PetscErrorCode MatConvert_SeqAIJ_SeqAIJSpooles(Mat A,const MatType type,MatReuse reuse,Mat *newmat) {
  /* This routine is only called to convert a MATSEQAIJ matrix */
  /* to a MATSEQAIJSPOOLES matrix, so we will ignore 'MatType type'. */
  PetscErrorCode ierr;
  Mat            B=*newmat;
  Mat_Spooles    *lu;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) {
    /* This routine is inherited, so we know the type is correct. */
    ierr = MatDuplicate(A,MAT_COPY_VALUES,&B);CHKERRQ(ierr);
  }
  ierr     = PetscNew(Mat_Spooles,&lu);CHKERRQ(ierr); 
  B->spptr = (void*)lu;

  lu->basetype                   = MATSEQAIJ;
  lu->useQR                      = PETSC_FALSE;
  lu->CleanUpSpooles             = PETSC_FALSE;
  lu->MatDuplicate               = A->ops->duplicate;
  lu->MatCholeskyFactorSymbolic  = A->ops->choleskyfactorsymbolic;
  lu->MatLUFactorSymbolic        = A->ops->lufactorsymbolic; 
  lu->MatView                    = A->ops->view;
  lu->MatAssemblyEnd             = A->ops->assemblyend;
  lu->MatDestroy                 = A->ops->destroy;
  B->ops->duplicate              = MatDuplicate_Spooles;
  B->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_SeqAIJSpooles;
  B->ops->lufactorsymbolic       = MatLUFactorSymbolic_SeqAIJSpooles; 
  B->ops->view                   = MatView_SeqAIJSpooles;
  B->ops->assemblyend            = MatAssemblyEnd_SeqAIJSpooles;
  B->ops->destroy                = MatDestroy_SeqAIJSpooles;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatConvert_seqaijspooles_seqaij_C",
                                           "MatConvert_Spooles_Base",MatConvert_Spooles_Base);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatConvert_seqaij_seqaijspooles_C",
                                           "MatConvert_SeqAIJ_SeqAIJSpooles",MatConvert_SeqAIJ_SeqAIJSpooles);CHKERRQ(ierr);
  /* ierr = PetscObjectChangeTypeName((PetscObject)B,MATSEQAIJSPOOLES);CHKERRQ(ierr); */
  ierr = PetscObjectChangeTypeName((PetscObject)B,type);CHKERRQ(ierr);
  *newmat = B;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "MatDuplicate_Spooles"
PetscErrorCode MatDuplicate_Spooles(Mat A, MatDuplicateOption op, Mat *M) {
  PetscErrorCode ierr;
  Mat_Spooles    *lu=(Mat_Spooles *)A->spptr;

  PetscFunctionBegin;
  ierr = (*lu->MatDuplicate)(A,op,M);CHKERRQ(ierr);
  ierr = PetscMemcpy((*M)->spptr,lu,sizeof(Mat_Spooles));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
  MATSEQAIJSPOOLES - MATSEQAIJSPOOLES = "seqaijspooles" - A matrix type providing direct solvers (LU or Cholesky) for sequential matrices 
  via the external package SPOOLES.

  If SPOOLES is installed (see the manual for
  instructions on how to declare the existence of external packages),
  a matrix type can be constructed which invokes SPOOLES solvers.
  After calling MatCreate(...,A), simply call MatSetType(A,MATSEQAIJSPOOLES).

  This matrix inherits from MATSEQAIJ.  As a result, MatSeqAIJSetPreallocation is 
  supported for this matrix type.  One can also call MatConvert for an inplace conversion to or from 
  the MATSEQAIJ type without data copy.

  Options Database Keys:
+ -mat_type seqaijspooles - sets the matrix type to "seqaijspooles" during a call to MatSetFromOptions()
. -mat_spooles_tau <tau> - upper bound on the magnitude of the largest element in L or U
. -mat_spooles_seed <seed> - random number seed used for ordering
. -mat_spooles_msglvl <msglvl> - message output level
. -mat_spooles_ordering <BestOfNDandMS,MMD,MS,ND> - ordering used
. -mat_spooles_maxdomainsize <n> - maximum subgraph size used by Spooles orderings
. -mat_spooles_maxzeros <n> - maximum number of zeros inside a supernode
. -mat_spooles_maxsize <n> - maximum size of a supernode
. -mat_spooles_FrontMtxInfo <true,fase> - print Spooles information about the computed factorization
. -mat_spooles_symmetryflag <0,1,2> - 0: SPOOLES_SYMMETRIC, 1: SPOOLES_HERMITIAN, 2: SPOOLES_NONSYMMETRIC
. -mat_spooles_patchAndGoFlag <0,1,2> - 0: no patch, 1: use PatchAndGo strategy 1, 2: use PatchAndGo strategy 2
. -mat_spooles_toosmall <dt> - drop tolerance for PatchAndGo strategy 1
. -mat_spooles_storeids <bool integer> - if nonzero, stores row and col numbers where patches were applied in an IV object
. -mat_spooles_fudge <delta> - fudge factor for rescaling diagonals with PatchAndGo strategy 2
- -mat_spooles_storevalues <bool integer> - if nonzero and PatchAndGo strategy 2 is used, store change in diagonal value in a DV object

   Level: beginner

.seealso: PCLU
M*/

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatCreate_SeqAIJSpooles"
PetscErrorCode MatCreate_SeqAIJSpooles(Mat A) 
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Change type name before calling MatSetType to force proper construction of SeqAIJ and SeqAIJSpooles types */
  ierr = PetscObjectChangeTypeName((PetscObject)A,MATSEQAIJSPOOLES);CHKERRQ(ierr);
  ierr = MatSetType(A,MATSEQAIJ);CHKERRQ(ierr);
  ierr = MatConvert_SeqAIJ_SeqAIJSpooles(A,MATSEQAIJSPOOLES,MAT_REUSE_MATRIX,&A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

/*MC
  MATAIJSPOOLES - MATAIJSPOOLES = "aijspooles" - A matrix type providing direct solvers (LU or Cholesky) for sequential and parellel matrices 
  via the external package SPOOLES.

  If SPOOLES is installed (see the manual for
  instructions on how to declare the existence of external packages),
  a matrix type can be constructed which invokes SPOOLES solvers.
  After calling MatCreate(...,A), simply call MatSetType(A,MATAIJSPOOLES).
  This matrix type is supported for double precision real and complex.

  This matrix inherits from MATAIJ.  As a result, MatSeqAIJSetPreallocation and MatMPIAIJSetPreallocation are
  supported for this matrix type.  One can also call MatConvert for an inplace conversion to or from 
  the MATAIJ type without data copy.

  Options Database Keys:
+ -mat_type aijspooles - sets the matrix type to "aijspooles" during a call to MatSetFromOptions()
. -mat_spooles_tau <tau> - upper bound on the magnitude of the largest element in L or U
. -mat_spooles_seed <seed> - random number seed used for ordering
. -mat_spooles_msglvl <msglvl> - message output level
. -mat_spooles_ordering <BestOfNDandMS,MMD,MS,ND> - ordering used
. -mat_spooles_maxdomainsize <n> - maximum subgraph size used by Spooles orderings
. -mat_spooles_maxzeros <n> - maximum number of zeros inside a supernode
. -mat_spooles_maxsize <n> - maximum size of a supernode
. -mat_spooles_FrontMtxInfo <true,fase> - print Spooles information about the computed factorization
. -mat_spooles_symmetryflag <0,1,2> - 0: SPOOLES_SYMMETRIC, 1: SPOOLES_HERMITIAN, 2: SPOOLES_NONSYMMETRIC
. -mat_spooles_patchAndGoFlag <0,1,2> - 0: no patch, 1: use PatchAndGo strategy 1, 2: use PatchAndGo strategy 2
. -mat_spooles_toosmall <dt> - drop tolerance for PatchAndGo strategy 1
. -mat_spooles_storeids <bool integer> - if nonzero, stores row and col numbers where patches were applied in an IV object
. -mat_spooles_fudge <delta> - fudge factor for rescaling diagonals with PatchAndGo strategy 2
- -mat_spooles_storevalues <bool integer> - if nonzero and PatchAndGo strategy 2 is used, store change in diagonal value in a DV object

   Level: beginner

.seealso: PCLU
M*/
EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatCreate_AIJSpooles"
PetscErrorCode MatCreate_AIJSpooles(Mat A) 
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  /* Change type name before calling MatSetType to force proper construction of SeqAIJSpooles or MPIAIJSpooles */
  ierr = PetscObjectChangeTypeName((PetscObject)A,MATAIJSPOOLES);CHKERRQ(ierr);
  ierr = MPI_Comm_size(A->comm,&size);CHKERRQ(ierr);CHKERRQ(ierr);
  if (size == 1) {
    ierr = MatSetType(A,MATSEQAIJ);CHKERRQ(ierr);
    ierr = MatConvert_SeqAIJ_SeqAIJSpooles(A,MATSEQAIJSPOOLES,MAT_REUSE_MATRIX,&A);CHKERRQ(ierr);
  } else {
    ierr   = MatSetType(A,MATMPIAIJ);CHKERRQ(ierr);
    ierr = MatConvert_MPIAIJ_MPIAIJSpooles(A,MATMPIAIJSPOOLES,MAT_REUSE_MATRIX,&A);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END
