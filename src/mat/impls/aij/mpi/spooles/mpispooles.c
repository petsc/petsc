/* 
   Provides an interface to the Spooles parallel sparse solver (MPI SPOOLES)
*/

#include "src/mat/impls/aij/seq/aij.h"
#include "src/mat/impls/sbaij/seq/sbaij.h"
#include "src/mat/impls/baij/seq/baij.h"
#include "src/mat/impls/aij/mpi/mpiaij.h"
#include "src/mat/impls/sbaij/mpi/mpisbaij.h"
#include "src/mat/impls/aij/seq/spooles/spooles.h"

EXTERN int SetSpoolesOptions(Mat, Spooles_options *);

#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_MPIAIJSpooles"
PetscErrorCode MatDestroy_MPIAIJSpooles(Mat A)
{
  Mat_Spooles   *lu = (Mat_Spooles*)A->spptr; 
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  if (lu->CleanUpSpooles) {
    FrontMtx_free(lu->frontmtx);        
    IV_free(lu->newToOldIV);            
    IV_free(lu->oldToNewIV); 
    IV_free(lu->vtxmapIV);
    InpMtx_free(lu->mtxA);             
    ETree_free(lu->frontETree);          
    IVL_free(lu->symbfacIVL);         
    SubMtxManager_free(lu->mtxmanager);    
    DenseMtx_free(lu->mtxX);
    DenseMtx_free(lu->mtxY);
    ierr = MPI_Comm_free(&(lu->comm_spooles));CHKERRQ(ierr);
    if ( lu->scat ){
      ierr = VecDestroy(lu->vec_spooles);CHKERRQ(ierr); 
      ierr = ISDestroy(lu->iden);CHKERRQ(ierr); 
      ierr = ISDestroy(lu->is_petsc);CHKERRQ(ierr);
      ierr = VecScatterDestroy(lu->scat);CHKERRQ(ierr);
    }
  }
  ierr = MatConvert_Spooles_Base(A,lu->basetype,MAT_REUSE_MATRIX,&A);
  ierr = (*A->ops->destroy)(A);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSolve_MPIAIJSpooles"
PetscErrorCode MatSolve_MPIAIJSpooles(Mat A,Vec b,Vec x)
{
  Mat_Spooles   *lu = (Mat_Spooles*)A->spptr;
  PetscErrorCode ierr;
  int           size,rank,m=A->m,irow,*rowindY;
  PetscScalar   *array;
  DenseMtx      *newY ;
  SubMtxManager *solvemanager ; 
#if defined(PETSC_USE_COMPLEX)
  double x_real,x_imag;
#endif

  PetscFunctionBegin;
  ierr = MPI_Comm_size(A->comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(A->comm,&rank);CHKERRQ(ierr);
  
  /* copy b into spooles' rhs mtxY */
  DenseMtx_init(lu->mtxY, lu->options.typeflag, 0, 0, m, 1, 1, m);    
  ierr = VecGetArray(b,&array);CHKERRQ(ierr);

  DenseMtx_rowIndices(lu->mtxY, &m, &rowindY);  /* get m, rowind */
  for ( irow = 0 ; irow < m ; irow++ ) {
    rowindY[irow] = irow + lu->rstart;           /* global rowind */
#if !defined(PETSC_USE_COMPLEX)
    DenseMtx_setRealEntry(lu->mtxY, irow, 0, *array++); 
#else
    DenseMtx_setComplexEntry(lu->mtxY,irow,0,PetscRealPart(*array),PetscImaginaryPart(*array));
    array++;
#endif
  }
  ierr = VecRestoreArray(b,&array);CHKERRQ(ierr);   
  
  if ( lu->options.msglvl > 2 ) {
    ierr = PetscFPrintf(PETSC_COMM_SELF,lu->options.msgFile, "\n\n 1 matrix in original ordering");CHKERRQ(ierr);
    DenseMtx_writeForHumanEye(lu->mtxY, lu->options.msgFile);
    fflush(lu->options.msgFile);
  }
  
  /* permute and redistribute Y if necessary */
  DenseMtx_permuteRows(lu->mtxY, lu->oldToNewIV);
  if ( lu->options.msglvl > 2 ) {
    ierr = PetscFPrintf(PETSC_COMM_SELF,lu->options.msgFile, "\n\n rhs matrix in new ordering");CHKERRQ(ierr);
    DenseMtx_writeForHumanEye(lu->mtxY, lu->options.msgFile);
   fflush(lu->options.msgFile);
  }

  MPI_Barrier(A->comm); /* for initializing firsttag, because the num. of tags used
                                   by FrontMtx_MPI_split() is unknown */
  lu->firsttag = 0;
  newY = DenseMtx_MPI_splitByRows(lu->mtxY, lu->vtxmapIV, lu->stats, lu->options.msglvl, 
                                lu->options.msgFile, lu->firsttag, lu->comm_spooles);
  DenseMtx_free(lu->mtxY);
  lu->mtxY = newY ;
  lu->firsttag += size ;
  if ( lu->options.msglvl > 2 ) {
    ierr = PetscFPrintf(PETSC_COMM_SELF,lu->options.msgFile, "\n\n split DenseMtx Y");CHKERRQ(ierr);
    DenseMtx_writeForHumanEye(lu->mtxY, lu->options.msgFile);
    fflush(lu->options.msgFile);
  }

  if ( FRONTMTX_IS_PIVOTING(lu->frontmtx) ) {
    /*   pivoting has taken place, redistribute the right hand side
         to match the final rows and columns in the fronts             */
    IV *rowmapIV ;
    rowmapIV = FrontMtx_MPI_rowmapIV(lu->frontmtx, lu->ownersIV, lu->options.msglvl,
                                    lu->options.msgFile, lu->comm_spooles);
    newY = DenseMtx_MPI_splitByRows(lu->mtxY, rowmapIV, lu->stats, lu->options.msglvl, 
                                   lu->options.msgFile, lu->firsttag, lu->comm_spooles);
    DenseMtx_free(lu->mtxY);
    lu->mtxY = newY ;
    IV_free(rowmapIV);
    lu->firsttag += size;
  }
  if ( lu->options.msglvl > 2 ) {
    ierr = PetscFPrintf(PETSC_COMM_SELF,lu->options.msgFile, "\n\n rhs matrix after split");CHKERRQ(ierr);
    DenseMtx_writeForHumanEye(lu->mtxY, lu->options.msgFile);
    fflush(lu->options.msgFile);
  }

  if ( lu->nmycol > 0 ) IVcopy(lu->nmycol,lu->rowindX,IV_entries(lu->ownedColumnsIV)); /* must do for each solve */
  
  /* solve the linear system */
  solvemanager = SubMtxManager_new();
  SubMtxManager_init(solvemanager, NO_LOCK, 0);
  FrontMtx_MPI_solve(lu->frontmtx, lu->mtxX, lu->mtxY, solvemanager, lu->solvemap, lu->cpus, 
                   lu->stats, lu->options.msglvl, lu->options.msgFile, lu->firsttag, lu->comm_spooles);
  SubMtxManager_free(solvemanager);
  if ( lu->options.msglvl > 2 ) {
    ierr = PetscFPrintf(PETSC_COMM_SELF,lu->options.msgFile, "\n solution in new ordering");CHKERRQ(ierr);
    DenseMtx_writeForHumanEye(lu->mtxX, lu->options.msgFile);
  }

  /* permute the solution into the original ordering */
  DenseMtx_permuteRows(lu->mtxX, lu->newToOldIV);
  if ( lu->options.msglvl > 2 ) {
    ierr = PetscFPrintf(PETSC_COMM_SELF,lu->options.msgFile, "\n\n solution in old ordering");CHKERRQ(ierr);
    DenseMtx_writeForHumanEye(lu->mtxX, lu->options.msgFile);
    fflush(lu->options.msgFile);
  }
  
  /* scatter local solution mtxX into mpi vector x */ 
  if( !lu->scat ){ /* create followings once for each numfactorization */
    /* vec_spooles <- mtxX */
#if !defined(PETSC_USE_COMPLEX) 
    ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,lu->nmycol,lu->entX,&lu->vec_spooles);CHKERRQ(ierr); 
#else    
    ierr = VecCreateSeq(PETSC_COMM_SELF,lu->nmycol,&lu->vec_spooles);CHKERRQ(ierr);
    ierr = VecGetArray(lu->vec_spooles,&array);CHKERRQ(ierr);   
    for (irow = 0; irow < lu->nmycol; irow++){
      DenseMtx_complexEntry(lu->mtxX,irow,0,&x_real,&x_imag);
      array[irow] = x_real+x_imag*PETSC_i;
    }
    ierr = VecRestoreArray(lu->vec_spooles,&array);CHKERRQ(ierr);
#endif 
    ierr = ISCreateStride(PETSC_COMM_SELF,lu->nmycol,0,1,&lu->iden);CHKERRQ(ierr);
    ierr = ISCreateGeneral(PETSC_COMM_SELF,lu->nmycol,lu->rowindX,&lu->is_petsc);CHKERRQ(ierr);  
    ierr = VecScatterCreate(lu->vec_spooles,lu->iden,x,lu->is_petsc,&lu->scat);CHKERRQ(ierr); 
  }

  ierr = VecScatterBegin(lu->vec_spooles,x,INSERT_VALUES,SCATTER_FORWARD,lu->scat);CHKERRQ(ierr);
  ierr = VecScatterEnd(lu->vec_spooles,x,INSERT_VALUES,SCATTER_FORWARD,lu->scat);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

#undef __FUNCT__   
#define __FUNCT__ "MatFactorNumeric_MPIAIJSpooles"
PetscErrorCode MatFactorNumeric_MPIAIJSpooles(Mat A,MatFactorInfo *info,Mat *F)
{
  Mat_Spooles     *lu = (Mat_Spooles*)(*F)->spptr;
  PetscErrorCode  ierr;
  int             rank,size,lookahead=0,sierr;
  ChvManager      *chvmanager ;
  Chv             *rootchv ;
  Graph           *graph ;
  IVL             *adjIVL;
  DV              *cumopsDV ;
  double          droptol=0.0,*opcounts,minops,cutoff;
#if !defined(PETSC_USE_COMPLEX)
  double          *val;
#endif
  InpMtx          *newA ;
  PetscScalar     *av, *bv; 
  int             *ai, *aj, *bi,*bj, nz, *ajj, *bjj, *garray,
                  i,j,irow,jcol,countA,countB,jB,*row,*col,colA_start,jj;
  int             M=A->M,m=A->m,root,nedges,tagbound,lasttag;
  
  PetscFunctionBegin;	
  ierr = MPI_Comm_size(A->comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(A->comm,&rank);CHKERRQ(ierr);

  if (lu->flg == DIFFERENT_NONZERO_PATTERN) { /* first numeric factorization */ 
    /* get input parameters */
    ierr = SetSpoolesOptions(A, &lu->options);CHKERRQ(ierr);

    (*F)->ops->solve   = MatSolve_MPIAIJSpooles;
    (*F)->ops->destroy = MatDestroy_MPIAIJSpooles;  
    (*F)->assembled    = PETSC_TRUE;

    /* to be used by MatSolve() */
    lu->mtxY = DenseMtx_new();  
    lu->mtxX = DenseMtx_new();
    lu->scat = PETSC_NULL;  

    IVzero(20, lu->stats); 
    DVzero(20, lu->cpus);

    lu->mtxA = InpMtx_new(); 
  }
  
  /* copy A to Spooles' InpMtx object */ 
  if ( lu->options.symflag == SPOOLES_NONSYMMETRIC ) { 
    Mat_MPIAIJ  *mat =  (Mat_MPIAIJ*)A->data;  
    Mat_SeqAIJ  *aa=(Mat_SeqAIJ*)(mat->A)->data;
    Mat_SeqAIJ  *bb=(Mat_SeqAIJ*)(mat->B)->data;
    ai=aa->i; aj=aa->j; av=aa->a;   
    bi=bb->i; bj=bb->j; bv=bb->a;
    lu->rstart = mat->rstart;
    nz         = aa->nz + bb->nz;
    garray     = mat->garray; 
  } else {         /* SPOOLES_SYMMETRIC  */
    Mat_MPISBAIJ  *mat = (Mat_MPISBAIJ*)A->data;
    Mat_SeqSBAIJ  *aa=(Mat_SeqSBAIJ*)(mat->A)->data;
    Mat_SeqBAIJ    *bb=(Mat_SeqBAIJ*)(mat->B)->data;
    ai=aa->i; aj=aa->j; av=aa->a;  
    bi=bb->i; bj=bb->j; bv=bb->a;
    lu->rstart = mat->rstart;
    nz         = aa->nz + bb->nz;     
    garray     = mat->garray;
  } 
      
  InpMtx_init(lu->mtxA, INPMTX_BY_ROWS, lu->options.typeflag, nz, 0); 
  row   = InpMtx_ivec1(lu->mtxA); 
  col   = InpMtx_ivec2(lu->mtxA); 
#if !defined(PETSC_USE_COMPLEX)
  val   = InpMtx_dvec(lu->mtxA); 
#endif

  jj = 0; irow = lu->rstart;   
  for ( i=0; i<m; i++ ) {
    ajj = aj + ai[i];                 /* ptr to the beginning of this row */      
    countA = ai[i+1] - ai[i];
    countB = bi[i+1] - bi[i];
    bjj = bj + bi[i]; 
    jB = 0;
  
    if (lu->options.symflag == SPOOLES_NONSYMMETRIC ){
      /* B part, smaller col index */   
      colA_start = lu->rstart + ajj[0]; /* the smallest col index for A */  
      for (j=0; j<countB; j++){
        jcol = garray[bjj[j]];
        if (jcol > colA_start) {
          jB = j;
          break;
        }
        row[jj] = irow; col[jj] = jcol; 
#if !defined(PETSC_USE_COMPLEX)
        val[jj++] = *bv++;
#else
        InpMtx_inputComplexEntry(lu->mtxA,irow,jcol,PetscRealPart(*bv),PetscImaginaryPart(*bv));
        bv++; jj++;
#endif
        if (j==countB-1) jB = countB; 
      }
    }
    /* A part */
    for (j=0; j<countA; j++){
      row[jj] = irow; col[jj] = lu->rstart + ajj[j]; 
#if !defined(PETSC_USE_COMPLEX)
      val[jj++] = *av++;
#else
      InpMtx_inputComplexEntry(lu->mtxA,irow,col[jj],PetscRealPart(*av),PetscImaginaryPart(*av));
      av++; jj++;
#endif
    }
    /* B part, larger col index */      
    for (j=jB; j<countB; j++){
      row[jj] = irow; col[jj] = garray[bjj[j]];
#if !defined(PETSC_USE_COMPLEX)
      val[jj++] = *bv++;
#else
     InpMtx_inputComplexEntry(lu->mtxA,irow,col[jj],PetscRealPart(*bv),PetscImaginaryPart(*bv)); 
     bv++; jj++;
#endif
    }
    irow++;
  } 
#if !defined(PETSC_USE_COMPLEX)
  InpMtx_inputRealTriples(lu->mtxA, nz, row, col, val); 
#endif
  InpMtx_changeStorageMode(lu->mtxA, INPMTX_BY_VECTORS);
  if ( lu->options.msglvl > 0 ) {
    printf("[%d] input matrix\n",rank);
    ierr = PetscFPrintf(PETSC_COMM_SELF,lu->options.msgFile, "\n\n [%d] input matrix\n",rank);CHKERRQ(ierr);
    InpMtx_writeForHumanEye(lu->mtxA, lu->options.msgFile);
    fflush(lu->options.msgFile);
  }

  if ( lu->flg == DIFFERENT_NONZERO_PATTERN){ /* first numeric factorization */
    /*
      find a low-fill ordering
      (1) create the Graph object
      (2) order the graph using multiple minimum degree
      (3) find out who has the best ordering w.r.t. op count,
          and broadcast that front tree object
    */
    graph = Graph_new();
    adjIVL = InpMtx_MPI_fullAdjacency(lu->mtxA, lu->stats, 
              lu->options.msglvl, lu->options.msgFile, lu->comm_spooles);
    nedges = IVL_tsize(adjIVL);
    Graph_init2(graph, 0, M, 0, nedges, M, nedges, adjIVL, NULL, NULL);
    if ( lu->options.msglvl > 2 ) {
      ierr = PetscFPrintf(PETSC_COMM_SELF,lu->options.msgFile, "\n\n graph of the input matrix");CHKERRQ(ierr);
      Graph_writeForHumanEye(graph, lu->options.msgFile);
      fflush(lu->options.msgFile);
    }

    switch (lu->options.ordering) {
    case 0:
      lu->frontETree = orderViaBestOfNDandMS(graph,
                     lu->options.maxdomainsize, lu->options.maxzeros, lu->options.maxsize,
                     lu->options.seed + rank, lu->options.msglvl, lu->options.msgFile); break;
    case 1:
      lu->frontETree = orderViaMMD(graph,lu->options.seed + rank,lu->options.msglvl,lu->options.msgFile); break;
    case 2:
      lu->frontETree = orderViaMS(graph, lu->options.maxdomainsize,
                     lu->options.seed + rank,lu->options.msglvl,lu->options.msgFile); break;
    case 3:
      lu->frontETree = orderViaND(graph, lu->options.maxdomainsize, 
                     lu->options.seed + rank,lu->options.msglvl,lu->options.msgFile); break;
    default:
      SETERRQ(PETSC_ERR_ARG_WRONG,"Unknown Spooles's ordering");
    }

    Graph_free(graph);
    if ( lu->options.msglvl > 2 ) {
      ierr = PetscFPrintf(PETSC_COMM_SELF,lu->options.msgFile, "\n\n front tree from ordering");CHKERRQ(ierr);
      ETree_writeForHumanEye(lu->frontETree, lu->options.msgFile);
      fflush(lu->options.msgFile);
    }

    opcounts = DVinit(size, 0.0);
    opcounts[rank] = ETree_nFactorOps(lu->frontETree, lu->options.typeflag, lu->options.symflag);
    MPI_Allgather((void*) &opcounts[rank], 1, MPI_DOUBLE,
              (void*) opcounts, 1, MPI_DOUBLE, A->comm);
    minops = DVmin(size, opcounts, &root);
    DVfree(opcounts);
    
    lu->frontETree = ETree_MPI_Bcast(lu->frontETree, root, 
                             lu->options.msglvl, lu->options.msgFile, lu->comm_spooles);
    if ( lu->options.msglvl > 2 ) {
      ierr = PetscFPrintf(PETSC_COMM_SELF,lu->options.msgFile, "\n\n best front tree");CHKERRQ(ierr);
      ETree_writeForHumanEye(lu->frontETree, lu->options.msgFile);
      fflush(lu->options.msgFile);
    }
  
    /* get the permutations, permute the front tree, permute the matrix */
    lu->oldToNewIV = ETree_oldToNewVtxPerm(lu->frontETree);
    lu->newToOldIV = ETree_newToOldVtxPerm(lu->frontETree);

    ETree_permuteVertices(lu->frontETree, lu->oldToNewIV);

    InpMtx_permute(lu->mtxA, IV_entries(lu->oldToNewIV), IV_entries(lu->oldToNewIV));
    
    if (  lu->options.symflag == SPOOLES_SYMMETRIC ) InpMtx_mapToUpperTriangle(lu->mtxA);

    InpMtx_changeCoordType(lu->mtxA, INPMTX_BY_CHEVRONS);
    InpMtx_changeStorageMode(lu->mtxA, INPMTX_BY_VECTORS);

    /* generate the owners map IV object and the map from vertices to owners */
    cutoff   = 1./(2*size);
    cumopsDV = DV_new();
    DV_init(cumopsDV, size, NULL);
    lu->ownersIV = ETree_ddMap(lu->frontETree, 
                       lu->options.typeflag, lu->options.symflag, cumopsDV, cutoff);
    DV_free(cumopsDV);
    lu->vtxmapIV = IV_new();
    IV_init(lu->vtxmapIV, M, NULL);
    IVgather(M, IV_entries(lu->vtxmapIV), 
             IV_entries(lu->ownersIV), ETree_vtxToFront(lu->frontETree));
    if ( lu->options.msglvl > 2 ) {
      ierr = PetscFPrintf(PETSC_COMM_SELF,lu->options.msgFile, "\n\n map from fronts to owning processes");CHKERRQ(ierr);
      IV_writeForHumanEye(lu->ownersIV, lu->options.msgFile);
      ierr = PetscFPrintf(PETSC_COMM_SELF,lu->options.msgFile, "\n\n map from vertices to owning processes");CHKERRQ(ierr);
      IV_writeForHumanEye(lu->vtxmapIV, lu->options.msgFile);
      fflush(lu->options.msgFile);
    }

    /* redistribute the matrix */
    lu->firsttag = 0 ;
    newA = InpMtx_MPI_split(lu->mtxA, lu->vtxmapIV, lu->stats, 
                        lu->options.msglvl, lu->options.msgFile, lu->firsttag, lu->comm_spooles);
    lu->firsttag += size ;

    InpMtx_free(lu->mtxA);
    lu->mtxA = newA ;
    InpMtx_changeStorageMode(lu->mtxA, INPMTX_BY_VECTORS);
    if ( lu->options.msglvl > 2 ) {
      ierr = PetscFPrintf(PETSC_COMM_SELF,lu->options.msgFile, "\n\n split InpMtx");CHKERRQ(ierr);
      InpMtx_writeForHumanEye(lu->mtxA, lu->options.msgFile);
      fflush(lu->options.msgFile);
    }
 
    /* compute the symbolic factorization */
    lu->symbfacIVL = SymbFac_MPI_initFromInpMtx(lu->frontETree, lu->ownersIV, lu->mtxA,
                     lu->stats, lu->options.msglvl, lu->options.msgFile, lu->firsttag, lu->comm_spooles);
    lu->firsttag += lu->frontETree->nfront ;
    if ( lu->options.msglvl > 2 ) {
      ierr = PetscFPrintf(PETSC_COMM_SELF,lu->options.msgFile, "\n\n local symbolic factorization");CHKERRQ(ierr);
      IVL_writeForHumanEye(lu->symbfacIVL, lu->options.msgFile);
      fflush(lu->options.msgFile);
    }

    lu->mtxmanager = SubMtxManager_new();
    SubMtxManager_init(lu->mtxmanager, NO_LOCK, 0);
    lu->frontmtx = FrontMtx_new();

  } else { /* new num factorization using previously computed symbolic factor */
    if (lu->options.pivotingflag) {                  /* different FrontMtx is required */
      FrontMtx_free(lu->frontmtx);   
      lu->frontmtx   = FrontMtx_new();
    }

    SubMtxManager_free(lu->mtxmanager);  
    lu->mtxmanager = SubMtxManager_new();
    SubMtxManager_init(lu->mtxmanager, NO_LOCK, 0);

    /* permute mtxA */
    InpMtx_permute(lu->mtxA, IV_entries(lu->oldToNewIV), IV_entries(lu->oldToNewIV));
    if ( lu->options.symflag == SPOOLES_SYMMETRIC ) InpMtx_mapToUpperTriangle(lu->mtxA);
    
    InpMtx_changeCoordType(lu->mtxA, INPMTX_BY_CHEVRONS);
    InpMtx_changeStorageMode(lu->mtxA, INPMTX_BY_VECTORS);

    /* redistribute the matrix */
    MPI_Barrier(A->comm);
    lu->firsttag = 0;
    newA = InpMtx_MPI_split(lu->mtxA, lu->vtxmapIV, lu->stats, 
                        lu->options.msglvl, lu->options.msgFile, lu->firsttag,lu->comm_spooles);
    lu->firsttag += size ;

    InpMtx_free(lu->mtxA);
    lu->mtxA = newA ;
    InpMtx_changeStorageMode(lu->mtxA, INPMTX_BY_VECTORS);
    if ( lu->options.msglvl > 2 ) {
      ierr = PetscFPrintf(PETSC_COMM_SELF,lu->options.msgFile, "\n\n split InpMtx");CHKERRQ(ierr);
      InpMtx_writeForHumanEye(lu->mtxA, lu->options.msgFile);
      fflush(lu->options.msgFile);
    }
  } /* end of if ( lu->flg == DIFFERENT_NONZERO_PATTERN) */

  FrontMtx_init(lu->frontmtx, lu->frontETree, lu->symbfacIVL, lu->options.typeflag, lu->options.symflag,
              FRONTMTX_DENSE_FRONTS, lu->options.pivotingflag, NO_LOCK, rank,
              lu->ownersIV, lu->mtxmanager, lu->options.msglvl, lu->options.msgFile);

    if ( lu->options.symflag == SPOOLES_SYMMETRIC ) {
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
  ChvManager_init(chvmanager, NO_LOCK, 0);  

  tagbound = maxTagMPI(lu->comm_spooles);
  lasttag  = lu->firsttag + 3*lu->frontETree->nfront + 2;
  /* if(!rank) PetscPrintf(PETSC_COMM_SELF,"\n firsttag: %d, nfront: %d\n",lu->firsttag, lu->frontETree->nfront);*/
  if ( lasttag > tagbound ) {
      SETERRQ3(PETSC_ERR_LIB,"fatal error in FrontMtx_MPI_factorInpMtx(), tag range is [%d,%d], tag_bound = %d",\
               lu->firsttag, lasttag, tagbound); 
  }
  rootchv = FrontMtx_MPI_factorInpMtx(lu->frontmtx, lu->mtxA, lu->options.tau, droptol,
                     chvmanager, lu->ownersIV, lookahead, &sierr, lu->cpus, 
                     lu->stats, lu->options.msglvl, lu->options.msgFile, lu->firsttag,lu->comm_spooles);
  ChvManager_free(chvmanager);
  lu->firsttag = lasttag;
  if ( lu->options.msglvl > 2 ) {
    ierr = PetscFPrintf(PETSC_COMM_SELF,lu->options.msgFile, "\n\n numeric factorization");CHKERRQ(ierr);
    FrontMtx_writeForHumanEye(lu->frontmtx, lu->options.msgFile);
    fflush(lu->options.msgFile);
  }

  if ( lu->options.symflag == SPOOLES_SYMMETRIC ) {
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
  if ( sierr >= 0 ) SETERRQ2(PETSC_ERR_LIB,"\n proc %d : factorization error at front %d", rank, sierr);
 
  /*  post-process the factorization and split 
      the factor matrices into submatrices */
  lasttag  = lu->firsttag + 5*size;
  if ( lasttag > tagbound ) {
      SETERRQ3(PETSC_ERR_LIB,"fatal error in FrontMtx_MPI_postProcess(), tag range is [%d,%d], tag_bound = %d",\
               lu->firsttag, lasttag, tagbound); 
  }
  FrontMtx_MPI_postProcess(lu->frontmtx, lu->ownersIV, lu->stats, lu->options.msglvl,
                         lu->options.msgFile, lu->firsttag, lu->comm_spooles);
  lu->firsttag += 5*size ;
  if ( lu->options.msglvl > 2 ) {
    ierr = PetscFPrintf(PETSC_COMM_SELF,lu->options.msgFile, "\n\n numeric factorization after post-processing");CHKERRQ(ierr);
    FrontMtx_writeForHumanEye(lu->frontmtx, lu->options.msgFile);
    fflush(lu->options.msgFile);
  }
  
  /* create the solve map object */
  lu->solvemap = SolveMap_new();
  SolveMap_ddMap(lu->solvemap, lu->frontmtx->symmetryflag, 
               FrontMtx_upperBlockIVL(lu->frontmtx),
               FrontMtx_lowerBlockIVL(lu->frontmtx),
               size, lu->ownersIV, FrontMtx_frontTree(lu->frontmtx), 
               lu->options.seed, lu->options.msglvl, lu->options.msgFile);
  if ( lu->options.msglvl > 2 ) {
    SolveMap_writeForHumanEye(lu->solvemap, lu->options.msgFile);
    fflush(lu->options.msgFile);
  }

  /* redistribute the submatrices of the factors */
  FrontMtx_MPI_split(lu->frontmtx, lu->solvemap, 
                   lu->stats, lu->options.msglvl, lu->options.msgFile, lu->firsttag, lu->comm_spooles);
  if ( lu->options.msglvl > 2 ) {
    ierr = PetscFPrintf(PETSC_COMM_SELF,lu->options.msgFile, "\n\n numeric factorization after split");CHKERRQ(ierr);
    FrontMtx_writeForHumanEye(lu->frontmtx, lu->options.msgFile);
    fflush(lu->options.msgFile);
  }

  /* create a solution DenseMtx object */  
  lu->ownedColumnsIV = FrontMtx_ownedColumnsIV(lu->frontmtx, rank, lu->ownersIV,
                                         lu->options.msglvl, lu->options.msgFile);
  lu->nmycol = IV_size(lu->ownedColumnsIV);
  if ( lu->nmycol > 0) {
    DenseMtx_init(lu->mtxX, lu->options.typeflag, 0, 0, lu->nmycol, 1, 1, lu->nmycol);
    /* get pointers rowindX and entX */
    DenseMtx_rowIndices(lu->mtxX, &lu->nmycol, &lu->rowindX);
    lu->entX = DenseMtx_entries(lu->mtxX); 
  } else { /* lu->nmycol == 0 */
    lu->entX    = 0;
    lu->rowindX = 0;
  }

  if ( lu->scat ){
    ierr = VecDestroy(lu->vec_spooles);CHKERRQ(ierr); 
    ierr = ISDestroy(lu->iden);CHKERRQ(ierr); 
    ierr = ISDestroy(lu->is_petsc);CHKERRQ(ierr);
    ierr = VecScatterDestroy(lu->scat);CHKERRQ(ierr);
  }
  lu->scat = PETSC_NULL;  
  lu->flg = SAME_NONZERO_PATTERN;

  lu->CleanUpSpooles = PETSC_TRUE;
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatConvert_MPIAIJ_MPIAIJSpooles"
PetscErrorCode MatConvert_MPIAIJ_MPIAIJSpooles(Mat A,const MatType type,MatReuse reuse,Mat *newmat) 
{
  /* This routine is only called to convert a MATMPIAIJ matrix */
  /* to a MATMPIAIJSPOOLES matrix, so we will ignore 'MatType type'. */
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

  lu->basetype                  = MATMPIAIJ;
  lu->CleanUpSpooles            = PETSC_FALSE;
  lu->MatDuplicate              = A->ops->duplicate;
  lu->MatLUFactorSymbolic       = A->ops->lufactorsymbolic;
  lu->MatCholeskyFactorSymbolic = A->ops->choleskyfactorsymbolic;
  lu->MatView                   = A->ops->view;
  lu->MatAssemblyEnd            = A->ops->assemblyend;
  lu->MatDestroy                = A->ops->destroy;

  B->ops->duplicate             = MatDuplicate_Spooles;
  B->ops->lufactorsymbolic      = MatLUFactorSymbolic_MPIAIJSpooles;  
  B->ops->view                  = MatView_SeqAIJSpooles;
  B->ops->assemblyend           = MatAssemblyEnd_MPIAIJSpooles;
  B->ops->destroy               = MatDestroy_MPIAIJSpooles;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatConvert_mpiaijspooles_mpiaij_C",
                                           "MatConvert_Spooles_Base",MatConvert_Spooles_Base);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatConvert_mpiaij_mpiaijspooles_C",
                                           "MatConvert_MPIAIJ_MPIAIJSpooles",MatConvert_MPIAIJ_MPIAIJSpooles);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)B,MATMPIAIJSPOOLES);CHKERRQ(ierr);
  *newmat = B;
  PetscFunctionReturn(0);
}
EXTERN_C_END

/*MC
  MATMPIAIJSPOOLES - MATMPIAIJSPOOLES = "mpiaijspooles" - A matrix type providing direct solvers (LU) for distributed matrices 
  via the external package Spooles.

  If MPIAIJSPOOLES is installed (see the manual for
  instructions on how to declare the existence of external packages),
  a matrix type can be constructed which invokes SPOOLES solvers.
  After calling MatCreate(...,A), simply call MatSetType(A,MATMPIAIJSPOOLES).

  This matrix inherits from MATMPIAIJ.  As a result, MatMPIAIJSetPreallocation is 
  supported for this matrix type.  One can also call MatConvert for an inplace conversion to or from 
  the MATMPIAIJ type without data copy.

  Consult Spooles documentation for more information about the options database keys below.

  Options Database Keys:
+ -mat_type mpiaijspooles - sets the matrix type to "mpiaijspooles" during a call to MatSetFromOptions()
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
#define __FUNCT__ "MatCreate_MPIAIJSpooles"
PetscErrorCode MatCreate_MPIAIJSpooles(Mat A) 
{
  PetscErrorCode ierr;
  Mat A_diag;

  PetscFunctionBegin;
  /* Change type name before calling MatSetType to force proper construction of MPIAIJ and MPIAIJSpooles types */
  ierr   = PetscObjectChangeTypeName((PetscObject)A,MATMPIAIJSPOOLES);CHKERRQ(ierr);
  ierr   = MatSetType(A,MATMPIAIJ);CHKERRQ(ierr);
  A_diag = ((Mat_MPIAIJ *)A->data)->A;
  ierr   = MatConvert_SeqAIJ_SeqAIJSpooles(A_diag,MATSEQAIJSPOOLES,MAT_REUSE_MATRIX,&A_diag);CHKERRQ(ierr);
  ierr   = MatConvert_MPIAIJ_MPIAIJSpooles(A,MATMPIAIJSPOOLES,MAT_REUSE_MATRIX,&A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

