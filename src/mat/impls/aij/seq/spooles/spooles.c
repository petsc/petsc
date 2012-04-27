
/*
   Provides an interface to the Spooles serial sparse solver
*/
#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/sbaij/seq/sbaij.h>
#include <../src/mat/impls/aij/seq/spooles/spooles.h>

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_SeqAIJSpooles"
PetscErrorCode MatDestroy_SeqAIJSpooles(Mat A)
{
  Mat_Spooles    *lu = (Mat_Spooles*)A->spptr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (lu && lu->CleanUpSpooles) {
    FrontMtx_free(lu->frontmtx);
    IV_free(lu->newToOldIV);
    IV_free(lu->oldToNewIV);
    InpMtx_free(lu->mtxA);
    ETree_free(lu->frontETree);
    IVL_free(lu->symbfacIVL);
    SubMtxManager_free(lu->mtxmanager);
    Graph_free(lu->graph);
  }
  ierr = PetscFree(A->spptr);CHKERRQ(ierr);
  ierr = MatDestroy_SeqAIJ(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSolve_SeqSpooles"
PetscErrorCode MatSolve_SeqSpooles(Mat A,Vec b,Vec x)
{
  Mat_Spooles      *lu = (Mat_Spooles*)A->spptr;
  PetscScalar      *array;
  DenseMtx         *mtxY, *mtxX ;
  PetscErrorCode   ierr;
  PetscInt         irow,neqns=A->cmap->n,nrow=A->rmap->n,*iv;
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
    int err;
    ierr = PetscFPrintf(PETSC_COMM_SELF,lu->options.msgFile, "\n\n right hand side matrix after permutation");CHKERRQ(ierr);
    DenseMtx_writeForHumanEye(mtxY, lu->options.msgFile); 
    ierr = PetscFPrintf(PETSC_COMM_SELF,lu->options.msgFile, "\n\n solution matrix in new ordering");CHKERRQ(ierr);
    DenseMtx_writeForHumanEye(mtxX, lu->options.msgFile);
    err = fflush(lu->options.msgFile);
    if (err) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SYS,"fflush() failed on file");    
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
#define __FUNCT__ "MatFactorNumeric_SeqSpooles"
PetscErrorCode MatFactorNumeric_SeqSpooles(Mat F,Mat A,const MatFactorInfo *info)
{  
  Mat_Spooles        *lu = (Mat_Spooles*)(F)->spptr;
  ChvManager         *chvmanager ;
  Chv                *rootchv ;
  IVL                *adjIVL;
  PetscErrorCode     ierr;
  PetscInt           nz,nrow=A->rmap->n,irow,nedges,neqns=A->cmap->n,*ai,*aj,i,*diag=0,fierr;
  PetscScalar        *av;
  double             cputotal,facops;
#if defined(PETSC_USE_COMPLEX)
  PetscInt           nz_row,*aj_tmp;
  PetscScalar        *av_tmp;
#else
  PetscInt           *ivec1,*ivec2,j;
  double             *dvec;
#endif
  PetscBool          isSeqAIJ,isMPIAIJ;
  
  PetscFunctionBegin;
  if (lu->flg == DIFFERENT_NONZERO_PATTERN) { /* first numeric factorization */      
    (F)->ops->solve   = MatSolve_SeqSpooles;
    (F)->assembled    = PETSC_TRUE; 
    
    /* set Spooles options */
    ierr = SetSpoolesOptions(A, &lu->options);CHKERRQ(ierr); 

    lu->mtxA = InpMtx_new();
  }

  /* copy A to Spooles' InpMtx object */
  ierr = PetscObjectTypeCompare((PetscObject)A,MATSEQAIJ,&isSeqAIJ);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)A,MATSEQAIJ,&isMPIAIJ);CHKERRQ(ierr);
  if (isSeqAIJ){
    Mat_SeqAIJ   *mat = (Mat_SeqAIJ*)A->data;
    ai=mat->i; aj=mat->j; av=mat->a;
    if (lu->options.symflag == SPOOLES_NONSYMMETRIC) {
      nz=mat->nz;
    } else { /* SPOOLES_SYMMETRIC || SPOOLES_HERMITIAN */
      nz=(mat->nz + A->rmap->n)/2;
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
      if ( lu->options.symflag == SPOOLES_NONSYMMETRIC || !(isSeqAIJ || isMPIAIJ)){
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
    if ( lu->options.symflag == SPOOLES_NONSYMMETRIC || !isSeqAIJ){
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
    int err;
    printf("\n\n input matrix");
    ierr = PetscFPrintf(PETSC_COMM_SELF,lu->options.msgFile, "\n\n input matrix");CHKERRQ(ierr);
    InpMtx_writeForHumanEye(lu->mtxA, lu->options.msgFile);
    err = fflush(lu->options.msgFile);
    if (err) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SYS,"fflush() failed on file");    
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
      int err;

      if (lu->options.useQR){
        ierr = PetscFPrintf(PETSC_COMM_SELF,lu->options.msgFile, "\n\n graph of A^T A");CHKERRQ(ierr);
      } else {
        ierr = PetscFPrintf(PETSC_COMM_SELF,lu->options.msgFile, "\n\n graph of the input matrix");CHKERRQ(ierr);
      }
      Graph_writeForHumanEye(lu->graph, lu->options.msgFile);
      err = fflush(lu->options.msgFile);
      if (err) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SYS,"fflush() failed on file");    
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
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Unknown Spooles's ordering");
    }

    if ( lu->options.msglvl > 0 ) {
      int err;

      ierr = PetscFPrintf(PETSC_COMM_SELF,lu->options.msgFile, "\n\n front tree from ordering");CHKERRQ(ierr);
      ETree_writeForHumanEye(lu->frontETree, lu->options.msgFile);
      err = fflush(lu->options.msgFile);
      if (err) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SYS,"fflush() failed on file");    
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
      int err;

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
      err = fflush(lu->options.msgFile);
      if (err) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SYS,"fflush() failed on file");    
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
    if (rootchv) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MAT_LU_ZRPVT,"\n matrix found to be singular");    
    if (fierr >= 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"\n error encountered at front %D", fierr);
    
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
    int err;

    ierr = PetscFPrintf(PETSC_COMM_SELF,lu->options.msgFile, "\n\n factor matrix");CHKERRQ(ierr);
    FrontMtx_writeForHumanEye(lu->frontmtx, lu->options.msgFile);
    err = fflush(lu->options.msgFile);
    if (err) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SYS,"fflush() failed on file");    
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
    int err;

    ierr = PetscFPrintf(PETSC_COMM_SELF,lu->options.msgFile, "\n\n factor matrix after post-processing");CHKERRQ(ierr);
    FrontMtx_writeForHumanEye(lu->frontmtx, lu->options.msgFile);
    err = fflush(lu->options.msgFile);
    if (err) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SYS,"fflush() failed on file");    
  }

  lu->flg = SAME_NONZERO_PATTERN;
  lu->CleanUpSpooles = PETSC_TRUE;
  PetscFunctionReturn(0);
}

