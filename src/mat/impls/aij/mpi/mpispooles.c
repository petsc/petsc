/*$Id: mpispooles.c,v 1.10 2001/08/15 15:56:50 bsmith Exp $*/
/* 
   Provides an interface to the Spooles parallel sparse solver (MPI SPOOLES)
*/

#include "src/mat/impls/aij/seq/aij.h"
#include "src/mat/impls/sbaij/seq/sbaij.h"
#include "src/mat/impls/baij/seq/baij.h"
#include "src/mat/impls/aij/mpi/mpiaij.h"
#include "src/mat/impls/sbaij/mpi/mpisbaij.h"

#if defined(PETSC_HAVE_SPOOLES) && !defined(PETSC_USE_SINGLE) && !defined(PETSC_USE_COMPLEX)
#include "src/mat/impls/aij/seq/spooles.h"

extern int MatDestroy_MPIAIJ(Mat); 

#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_MPIAIJ_Spooles"
int MatDestroy_MPIAIJ_Spooles(Mat A)
{
  Mat_Spooles   *lu = (Mat_Spooles*)A->spptr; 
  int              ierr;
  
  PetscFunctionBegin;

  /* allInOneMPI doesn't free following spaces, should I do it? */
  FrontMtx_free(lu->frontmtx) ;        
  IV_free(lu->newToOldIV) ;            
  IV_free(lu->oldToNewIV) ; 
  IV_free(lu->vtxmapIV) ;
  InpMtx_free(lu->mtxA) ;             
  ETree_free(lu->frontETree) ;          
  IVL_free(lu->symbfacIVL) ;         
  SubMtxManager_free(lu->mtxmanager) ;    
  DenseMtx_free(lu->mtxX) ;
  DenseMtx_free(lu->mtxY) ;

  ierr = PetscFree(lu);CHKERRQ(ierr); 
  ierr = MatDestroy_MPIAIJ(A);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSolve_MPIAIJ_Spooles"
int MatSolve_MPIAIJ_Spooles(Mat A,Vec b,Vec x)
{
  Mat_Spooles      *lu = (Mat_Spooles*)A->spptr;
  int                 ierr,size,rank,m=A->m,irow,*rowindY;
  PetscScalar         *array;
  DenseMtx            *newY ;
  IV                  *rowmapIV ;
  SubMtxManager       *solvemanager ; 
  Vec                 vec_spooles;
  IS                  iden, is_petsc;
  VecScatter          scat;

  PetscFunctionBegin;	
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  /* copy b into spooles' rhs mtxY */
  DenseMtx_init(lu->mtxY, SPOOLES_REAL, 0, 0, m, 1, 1, m) ;    
  ierr = VecGetArray(b,&array);CHKERRQ(ierr);

  /* doesn't work! 
  ierr = PetscMalloc(m*sizeof(int),&rowindY);CHKERRQ(ierr);
  for ( irow = 0 ; irow < m ; irow++ ) rowindY[irow] = irow + lu->rstart; 
  int colind=0;
  DenseMtx_initWithPointers(lu->mtxY,SPOOLES_REAL,0,0,m,1,1,m,rowindY,&colind,array); 
  */

  DenseMtx_rowIndices(lu->mtxY, &m, &rowindY) ;  /* get m, rowind */
  for ( irow = 0 ; irow < m ; irow++ ) {
    rowindY[irow] = irow + lu->rstart; 
    DenseMtx_setRealEntry(lu->mtxY, irow, 0, *array++) ; 
  }
  /* DenseMtx_column(lu->mtxY, 0, &array);  doesn't work! */
  ierr = VecRestoreArray(b,&array);CHKERRQ(ierr);   
  
  if ( lu->msglvl > 2 ) {
    fprintf(lu->msgFile, "\n\n 1 matrix in original ordering") ;
    DenseMtx_writeForHumanEye(lu->mtxY, lu->msgFile) ;
    fflush(lu->msgFile) ;
  }
  
  /* STEP 13: permute and redistribute Y if necessary */
  DenseMtx_permuteRows(lu->mtxY, lu->oldToNewIV) ;
  if ( lu->msglvl > 2 ) {
    fprintf(lu->msgFile, "\n\n rhs matrix in new ordering") ;
    DenseMtx_writeForHumanEye(lu->mtxY, lu->msgFile) ;
   fflush(lu->msgFile) ;
  }
  
  newY = DenseMtx_MPI_splitByRows(lu->mtxY, lu->vtxmapIV, lu->stats, lu->msglvl, 
                                lu->msgFile, lu->firsttag, MPI_COMM_WORLD) ;
  DenseMtx_free(lu->mtxY) ;
  lu->mtxY = newY ;
  lu->firsttag += size ;
  if ( lu->msglvl > 2 ) {
    fprintf(lu->msgFile, "\n\n split DenseMtx Y") ;
    DenseMtx_writeForHumanEye(lu->mtxY, lu->msgFile) ;
    fflush(lu->msgFile) ;
  }

  if ( FRONTMTX_IS_PIVOTING(lu->frontmtx) ) {
    /*   pivoting has taken place, redistribute the right hand side
         to match the final rows and columns in the fronts             */

    rowmapIV = FrontMtx_MPI_rowmapIV(lu->frontmtx, lu->ownersIV, lu->msglvl,
                                    lu->msgFile, MPI_COMM_WORLD) ;
    newY = DenseMtx_MPI_splitByRows(lu->mtxY, rowmapIV, lu->stats, lu->msglvl, 
                                   lu->msgFile, lu->firsttag, MPI_COMM_WORLD) ;
    DenseMtx_free(lu->mtxY) ;
    lu->mtxY = newY ;
    IV_free(rowmapIV) ;
  }
  if ( lu->msglvl > 2 ) {
    fprintf(lu->msgFile, "\n\n rhs matrix after split") ;
    DenseMtx_writeForHumanEye(lu->mtxY, lu->msgFile) ;
    fflush(lu->msgFile) ;
  }

  if ( lu->nmycol > 0 ) IVcopy(lu->nmycol, lu->rowindX, IV_entries(lu->ownedColumnsIV)) ; /* must be done for each solve */

  /* STEP 15: solve the linear system */
  solvemanager = SubMtxManager_new() ;
  SubMtxManager_init(solvemanager, NO_LOCK, 0) ;
  FrontMtx_MPI_solve(lu->frontmtx, lu->mtxX, lu->mtxY, solvemanager, lu->solvemap, lu->cpus, 
                   lu->stats, lu->msglvl, lu->msgFile, lu->firsttag, MPI_COMM_WORLD) ;
  SubMtxManager_free(solvemanager) ;
  if ( lu->msglvl > 2 ) {
    fprintf(lu->msgFile, "\n solution in new ordering") ;
    DenseMtx_writeForHumanEye(lu->mtxX, lu->msgFile) ;
  }

  /* permute the solution into the original ordering */
  DenseMtx_permuteRows(lu->mtxX, lu->newToOldIV) ;
  if ( lu->msglvl > 2 ) {
    fprintf(lu->msgFile, "\n\n solution in old ordering") ;
    DenseMtx_writeForHumanEye(lu->mtxX, lu->msgFile) ;
    fflush(lu->msgFile) ;
  }

  /* scatter local solution mtxX into mpi vector x */
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,lu->nmycol,lu->entX,&vec_spooles);CHKERRQ(ierr); /* vec_spooles <- mtxX */ 
 
  ierr = ISCreateStride(PETSC_COMM_SELF,lu->nmycol,0,1,&iden);CHKERRQ(ierr); /* is_spooles */  
  ierr = ISCreateGeneral(PETSC_COMM_SELF,lu->nmycol,lu->rowindX,&is_petsc);CHKERRQ(ierr);  
  ierr = VecScatterCreate(vec_spooles,iden,x,is_petsc,&scat);CHKERRQ(ierr);  /* fail to work if iden is created in numfac */

  ierr = VecScatterBegin(vec_spooles,x,INSERT_VALUES,SCATTER_FORWARD,scat);CHKERRQ(ierr);
  ierr = VecScatterEnd(vec_spooles,x,INSERT_VALUES,SCATTER_FORWARD,scat);CHKERRQ(ierr);
  
  /* free spaces */ 
  ierr = ISDestroy(iden);CHKERRQ(ierr);  
  ierr = ISDestroy(is_petsc);CHKERRQ(ierr);
  ierr = VecScatterDestroy(scat);CHKERRQ(ierr);
  ierr = VecDestroy(vec_spooles);CHKERRQ(ierr); 
  
  PetscFunctionReturn(0);
}

#undef __FUNCT__   
#define __FUNCT__ "MatFactorNumeric_MPIAIJ_Spooles"
int MatFactorNumeric_MPIAIJ_Spooles(Mat A,Mat *F)
{
  Mat_Spooles     *lu = (Mat_Spooles*)(*F)->spptr;
  int                rank,size,ierr,lookahead=0,ordering;
  ChvManager         *chvmanager ;
  Chv                *rootchv ;
  Graph              *graph ;
  IVL                *adjIVL;
  DV                 *cumopsDV ;
  double             droptol=0.0,tau,*opcounts,  minops, cutoff, *val;
  InpMtx             *newA ;
  char               buff[32],*ordertype[] = {"BestOfNDandMS","MMD","MS","ND"}; 
  PetscTruth         flg;
  PetscScalar        *av, *bv; 
  int                *ai, *aj, *bi,*bj, nz, *ajj, *bjj, *garray,
                     i,j,irow,jcol,countA,countB,jB,*row,*col,colA_start,jj;
  int                M=A->M,N=A->N,m=A->m,root,nedges;
  
  PetscFunctionBegin;	
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  if ( lu->flg == DIFFERENT_NONZERO_PATTERN){ /* first numeric factorization */

    (*F)->ops->solve   = MatSolve_MPIAIJ_Spooles;
    (*F)->ops->destroy = MatDestroy_MPIAIJ_Spooles;  
    (*F)->assembled    = PETSC_TRUE;

    IVzero(20, lu->stats) ; 
    DVzero(20, lu->cpus) ;

    /* set default input parameters */
    lu->tau           = 100.;
    lu->seed          = 10101;  
    lu->ordering      = 1;    
    lu->maxdomainsize = 500;
    lu->maxzeros      = 1000;
    lu->maxsize       = 96;

    /* get runtime input parameters */
    ierr = PetscOptionsBegin(A->comm,A->prefix,"MPI Spooles Options","Mat");CHKERRQ(ierr); 
  
    ierr = PetscOptionsReal("-mat_spooles_tau","tau (used for pivoting; \n\
           all entries in L and U have magnitude no more than tau)","None",lu->tau,&lu->tau,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-mat_spooles_seed","random number seed, used for ordering","None",\
                           lu->seed,&lu->seed,PETSC_NULL);CHKERRQ(ierr);
    if (PetscLogPrintInfo) {
      lu->msglvl = 1;
    } else {
      lu->msglvl = 0;
    }
    ierr = PetscOptionsInt("-mat_spooles_msglvl","msglvl","None",lu->msglvl,&lu->msglvl,0);CHKERRQ(ierr); 
    if (lu->msglvl > 0) {
        lu->msgFile = fopen("spooles.msgFile", "a");
        if(!rank) PetscPrintf(PETSC_COMM_SELF,"\n Spooles' output is written into the file 'spooles.msgFile' \n\n");
    } 

    ierr = PetscOptionsEList("-mat_spooles_ordering","ordering type","None",
             ordertype,4,ordertype[1],buff,32,&flg);CHKERRQ(ierr); /* default: MMD */
    while (flg) {
      ierr = PetscStrcmp(buff,"BestOfNDandMS",&flg);CHKERRQ(ierr);
      if (flg) {
        lu->ordering = 0; break;
      }
      ierr = PetscStrcmp(buff,"MMD",&flg);CHKERRQ(ierr);
      if (flg) {
        lu->ordering = 1; break;
      }
      ierr = PetscStrcmp(buff,"MS",&flg);CHKERRQ(ierr);
      if (flg) {
        lu->ordering = 2; break;
      }
      ierr = PetscStrcmp(buff,"ND",&flg);CHKERRQ(ierr);
      if (flg) {
        lu->ordering = 3; break;
      }
      SETERRQ1(1,"Unknown Spooles's ordering %s",buff);
    }
    
    ierr = PetscOptionsInt("-mat_spooles_maxdomainsize","maxdomainsize","None",\
                           lu->maxdomainsize,&lu->maxdomainsize,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-mat_spooles_maxzeros ","maxzeros","None",\
                           lu->maxzeros,&lu->maxzeros,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-mat_spooles_maxsize","maxsize","None",\
                           lu->maxsize,&lu->maxsize,PETSC_NULL);CHKERRQ(ierr);
    PetscOptionsEnd();

    /* copy A to Spooles' InpMtx object */ 
    if ( lu->symflag == SPOOLES_NONSYMMETRIC ) { 
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
      nz         = aa->s_nz + bb->nz;     
      garray     = mat->garray;
     
    } 
        
    lu->mtxA   = InpMtx_new() ;
    InpMtx_init(lu->mtxA, INPMTX_BY_ROWS, SPOOLES_REAL, nz, 0) ; 
    row   = InpMtx_ivec1(lu->mtxA); 
    col   = InpMtx_ivec2(lu->mtxA); 
    val   = InpMtx_dvec(lu->mtxA); 
 
    jj = 0; jB = 0; irow = lu->rstart;   
    for ( i=0; i<m; i++ ) {
      ajj = aj + ai[i];                 /* ptr to the beginning of this row */      
      colA_start = lu->rstart + ajj[0]; /* the smallest col index for A */
      countA = ai[i+1] - ai[i];
      countB = bi[i+1] - bi[i];
        
      /* B part, smaller col index */   
      bjj = bj + bi[i];      
      for (j=0; j<countB; j++){
        jcol = garray[bjj[j]];
        if (jcol > colA_start) {
          jB = j;
          break;
        }
        row[jj] = irow; col[jj] = jcol; val[jj++] = *bv++;
        if (j==countB-1) jB = countB; 
      }
      /* A part */
      for (j=0; j<countA; j++){
        row[jj] = irow; col[jj] = lu->rstart + ajj[j]; val[jj++] = *av++;
      }
      /* B part, larger col index */      
      for (j=jB; j<countB; j++){
        row[jj] = irow; col[jj] = garray[bjj[j]]; val[jj++] = *bv++;
      }
      irow++;
    } 

    InpMtx_inputRealTriples(lu->mtxA, nz, row, col, val); 
    InpMtx_changeStorageMode(lu->mtxA, INPMTX_BY_VECTORS) ;
    if ( lu->msglvl > 2 ) {
      fprintf(lu->msgFile, "\n\n input matrix") ;
      InpMtx_writeForHumanEye(lu->mtxA, lu->msgFile) ;
      fflush(lu->msgFile) ;
    }
 
    /*
      -------------------------------------------------------
      STEP 2 : find a low-fill ordering
      (1) create the Graph object
      (2) order the graph using multiple minimum degree
      (3) find out who has the best ordering w.r.t. op count,
          and broadcast that front tree object
          -------------------------------------------------------
    */
    graph = Graph_new() ;
    adjIVL = InpMtx_MPI_fullAdjacency(lu->mtxA, lu->stats, 
                                  lu->msglvl, lu->msgFile, MPI_COMM_WORLD) ;
    nedges = IVL_tsize(adjIVL) ;
    Graph_init2(graph, 0, M, 0, nedges, M, nedges, adjIVL, NULL, NULL) ;
    if ( lu->msglvl > 2 ) {
      fprintf(lu->msgFile, "\n\n graph of the input matrix") ;
      Graph_writeForHumanEye(graph, lu->msgFile) ;
      fflush(lu->msgFile) ;
    }

    switch (lu->ordering) {
    case 0:
      lu->frontETree = orderViaBestOfNDandMS(graph,
                     lu->maxdomainsize, lu->maxzeros, lu->maxsize,
                     lu->seed + rank, lu->msglvl, lu->msgFile); break;
    case 1:
      lu->frontETree = orderViaMMD(graph,lu->seed + rank,lu->msglvl,lu->msgFile); break;
    case 2:
      lu->frontETree = orderViaMS(graph, lu->maxdomainsize,
                     lu->seed + rank,lu->msglvl,lu->msgFile); break;
    case 3:
      lu->frontETree = orderViaND(graph, lu->maxdomainsize, 
                     lu->seed + rank,lu->msglvl,lu->msgFile); break;
    default:
      SETERRQ(1,"Unknown Spooles's ordering");
    }

    Graph_free(graph) ;
    if ( lu->msglvl > 2 ) {
      fprintf(lu->msgFile, "\n\n front tree from ordering") ;
      ETree_writeForHumanEye(lu->frontETree, lu->msgFile) ;
      fflush(lu->msgFile) ;
    }

    opcounts = DVinit(size, 0.0) ;
    opcounts[rank] = ETree_nFactorOps(lu->frontETree, SPOOLES_REAL, lu->symflag) ;
    MPI_Allgather((void *) &opcounts[rank], 1, MPI_DOUBLE,
              (void *) opcounts, 1, MPI_DOUBLE, MPI_COMM_WORLD) ;
    minops = DVmin(size, opcounts, &root) ;
    DVfree(opcounts) ;

    lu->frontETree = ETree_MPI_Bcast(lu->frontETree, root, 
                             lu->msglvl, lu->msgFile, MPI_COMM_WORLD) ;
    if ( lu->msglvl > 2 ) {
      fprintf(lu->msgFile, "\n\n best front tree") ;
      ETree_writeForHumanEye(lu->frontETree, lu->msgFile) ;
      fflush(lu->msgFile) ;
    }
  
    /* STEP 3: get the permutations, permute the front tree, permute the matrix */
    lu->oldToNewIV = ETree_oldToNewVtxPerm(lu->frontETree) ;
    lu->newToOldIV = ETree_newToOldVtxPerm(lu->frontETree) ;

    ETree_permuteVertices(lu->frontETree, lu->oldToNewIV) ;

    InpMtx_permute(lu->mtxA, IV_entries(lu->oldToNewIV), IV_entries(lu->oldToNewIV)) ;
    
    if (  lu->symflag == SPOOLES_SYMMETRIC ) InpMtx_mapToUpperTriangle(lu->mtxA) ;

    InpMtx_changeCoordType(lu->mtxA, INPMTX_BY_CHEVRONS) ;
    InpMtx_changeStorageMode(lu->mtxA, INPMTX_BY_VECTORS) ;

    /* STEP 4: generate the owners map IV object and the map from vertices to owners */
    cutoff   = 1./(2*size) ;
    cumopsDV = DV_new() ;
    DV_init(cumopsDV, size, NULL) ;
    lu->ownersIV = ETree_ddMap(lu->frontETree, 
                       SPOOLES_REAL, lu->symflag, cumopsDV, cutoff) ;
    DV_free(cumopsDV) ;
    lu->vtxmapIV = IV_new() ;
    IV_init(lu->vtxmapIV, M, NULL) ;
    IVgather(M, IV_entries(lu->vtxmapIV), 
             IV_entries(lu->ownersIV), ETree_vtxToFront(lu->frontETree)) ;
    if ( lu->msglvl > 2 ) {
      fprintf(lu->msgFile, "\n\n map from fronts to owning processes") ;
      IV_writeForHumanEye(lu->ownersIV, lu->msgFile) ;
      fprintf(lu->msgFile, "\n\n map from vertices to owning processes") ;
      IV_writeForHumanEye(lu->vtxmapIV, lu->msgFile) ;
      fflush(lu->msgFile) ;
    }

    /* STEP 5: redistribute the matrix */
    lu->firsttag = 0 ;
    newA = InpMtx_MPI_split(lu->mtxA, lu->vtxmapIV, lu->stats, 
                        lu->msglvl, lu->msgFile, lu->firsttag, MPI_COMM_WORLD) ;
    lu->firsttag++ ;

    InpMtx_free(lu->mtxA) ;
    lu->mtxA = newA ;
    InpMtx_changeStorageMode(lu->mtxA, INPMTX_BY_VECTORS) ;
    if ( lu->msglvl > 2 ) {
      fprintf(lu->msgFile, "\n\n split InpMtx") ;
      InpMtx_writeForHumanEye(lu->mtxA, lu->msgFile) ;
      fflush(lu->msgFile) ;
    }
 
    /* STEP 6: compute the symbolic factorization */
    lu->symbfacIVL = SymbFac_MPI_initFromInpMtx(lu->frontETree, lu->ownersIV, lu->mtxA,
                     lu->stats, lu->msglvl, lu->msgFile, lu->firsttag, MPI_COMM_WORLD) ;
    lu->firsttag += lu->frontETree->nfront ;
    if ( lu->msglvl > 2 ) {
      fprintf(lu->msgFile, "\n\n local symbolic factorization") ;
      IVL_writeForHumanEye(lu->symbfacIVL, lu->msgFile) ;
      fflush(lu->msgFile) ;
    }

    lu->mtxmanager = SubMtxManager_new() ;
    SubMtxManager_init(lu->mtxmanager, NO_LOCK, 0) ;
    lu->frontmtx = FrontMtx_new() ;

    /* to be used by MatSolve() */
    lu->mtxY = DenseMtx_new() ;  
    lu->mtxX = DenseMtx_new() ;

  } else { /* new num factorization using previously computed symbolic factor */
    if (lu->pivotingflag) {                  /* different FrontMtx is required */
      FrontMtx_free(lu->frontmtx) ;   
      lu->frontmtx   = FrontMtx_new() ;
    }

    SubMtxManager_free(lu->mtxmanager) ;  
    lu->mtxmanager = SubMtxManager_new() ;
    SubMtxManager_init(lu->mtxmanager, NO_LOCK, 0) ;

    /* copy new numerical values of A into mtxA */   
    if ( lu->symflag == SPOOLES_NONSYMMETRIC ) { 
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
      nz         = aa->s_nz + bb->nz;     
      garray     = mat->garray;
     
    } 

    InpMtx_init(lu->mtxA, INPMTX_BY_ROWS, SPOOLES_REAL, nz, 0) ; 
    row   = InpMtx_ivec1(lu->mtxA); 
    col   = InpMtx_ivec2(lu->mtxA); 
    val   = InpMtx_dvec(lu->mtxA);
     
    jj= 0; jB = 0; irow = lu->rstart;
    for ( i=0; i<m; i++ ) {
      ajj = aj + ai[i];              /* ptr to the beginning of this row */
      colA_start = lu->rstart + ajj[0]; /* the smallest col index for A */
      countB = bi[i+1] - bi[i];
      countA = ai[i+1] - ai[i];
      
      /* B part, smaller col index */   
      bjj = bj + bi[i];      
      for (j=0; j<countB; j++){
        jcol = garray[bjj[j]];
        if (jcol > colA_start ) {
          jB = j; break;
        }
        row[jj] = irow; col[jj] = jcol; val[jj] = *bv;
        jj++;
        if (j==countB-1) jB = countB;
      }
      /* A part */
      for (j=0; j<countA; j++){
        row[jj] = irow; col[jj] = lu->rstart + ajj[j]; val[jj] = *av++;
        jj++;
      }
      /* B part, larger col index */      
      for (j=jB; j<countB; j++){
        row[jj] = irow; col[jj] = garray[bjj[j]]; val[jj] = *bv++;
        jj++;
      }
      irow++;
    }     
    InpMtx_inputRealTriples(lu->mtxA, nz, row, col, val); 

    InpMtx_changeStorageMode(lu->mtxA, INPMTX_BY_VECTORS) ;
    if ( lu->msglvl > 2 ) {
      fprintf(lu->msgFile, "\n\n input matrix") ;
      InpMtx_writeForHumanEye(lu->mtxA, lu->msgFile) ;
      fflush(lu->msgFile) ;
    }
 
    /* permute mtxA */
    InpMtx_permute(lu->mtxA, IV_entries(lu->oldToNewIV), IV_entries(lu->oldToNewIV)) ;
    if ( lu->symflag == SPOOLES_SYMMETRIC ) InpMtx_mapToUpperTriangle(lu->mtxA) ;
    
    InpMtx_changeCoordType(lu->mtxA, INPMTX_BY_CHEVRONS) ;
    InpMtx_changeStorageMode(lu->mtxA, INPMTX_BY_VECTORS) ;

    /* STEP 5: redistribute the matrix */
    /* lu->firsttag = 0 ;  */   /* do I need this? */
    newA = InpMtx_MPI_split(lu->mtxA, lu->vtxmapIV, lu->stats, 
                        lu->msglvl, lu->msgFile, lu->firsttag, MPI_COMM_WORLD) ;
    lu->firsttag++ ;

    InpMtx_free(lu->mtxA) ;
    lu->mtxA = newA ;
    InpMtx_changeStorageMode(lu->mtxA, INPMTX_BY_VECTORS) ;
    if ( lu->msglvl > 2 ) {
      fprintf(lu->msgFile, "\n\n split InpMtx") ;
      InpMtx_writeForHumanEye(lu->mtxA, lu->msgFile) ;
      fflush(lu->msgFile) ;
    }
  } /* end of if ( lu->flg == DIFFERENT_NONZERO_PATTERN) */

  FrontMtx_init(lu->frontmtx, lu->frontETree, lu->symbfacIVL, SPOOLES_REAL, lu->symflag,
              FRONTMTX_DENSE_FRONTS, lu->pivotingflag, NO_LOCK, rank,
              lu->ownersIV, lu->mtxmanager, lu->msglvl, lu->msgFile) ;

  /* STEP 8: numerical factorization */
  chvmanager = ChvManager_new() ;
  ChvManager_init(chvmanager, NO_LOCK, 0) ;

  ordering = lu->ordering;
  tau = lu->tau;          /* lu->tau, ordering will be rewritten by cpus, so save them */
  rootchv = FrontMtx_MPI_factorInpMtx(lu->frontmtx, lu->mtxA, lu->tau, droptol,
                     chvmanager, lu->ownersIV, lookahead, &ierr, lu->cpus, 
                     lu->stats, lu->msglvl, lu->msgFile, lu->firsttag, MPI_COMM_WORLD) ;
  lu->ordering = ordering;
  lu->tau = tau; /* original value of ordering and tau */
 
  ChvManager_free(chvmanager) ;
  lu->firsttag += 3*lu->frontETree->nfront + 2 ;
  if ( lu->msglvl > 2 ) {
    fprintf(lu->msgFile, "\n\n numeric factorization") ;
    FrontMtx_writeForHumanEye(lu->frontmtx, lu->msgFile) ;
    fflush(lu->msgFile) ;
  }
  if ( ierr >= 0 ) SETERRQ2(1,"\n proc %d : factorization error at front %d", rank, ierr) ;
 
  /*  STEP 9: post-process the factorization and split 
              the factor matrices into submatrices */
  FrontMtx_MPI_postProcess(lu->frontmtx, lu->ownersIV, lu->stats, lu->msglvl,
                         lu->msgFile, lu->firsttag, MPI_COMM_WORLD) ;
  lu->firsttag += 5*size ;
  if ( lu->msglvl > 2 ) {
    fprintf(lu->msgFile, "\n\n numeric factorization after post-processing");
    FrontMtx_writeForHumanEye(lu->frontmtx, lu->msgFile) ;
    fflush(lu->msgFile) ;
  }
  
  /* STEP 10: create the solve map object */
  lu->solvemap = SolveMap_new() ;
  SolveMap_ddMap(lu->solvemap, lu->frontmtx->symmetryflag, 
               FrontMtx_upperBlockIVL(lu->frontmtx),
               FrontMtx_lowerBlockIVL(lu->frontmtx),
               size, lu->ownersIV, FrontMtx_frontTree(lu->frontmtx), 
               lu->seed, lu->msglvl, lu->msgFile);
  if ( lu->msglvl > 2 ) {
    SolveMap_writeForHumanEye(lu->solvemap, lu->msgFile) ;
    fflush(lu->msgFile) ;
  }

  /* STEP 11: redistribute the submatrices of the factors */
  FrontMtx_MPI_split(lu->frontmtx, lu->solvemap, 
                   lu->stats, lu->msglvl, lu->msgFile, lu->firsttag, MPI_COMM_WORLD) ;
  if ( lu->msglvl > 2 ) {
    fprintf(lu->msgFile, "\n\n numeric factorization after split") ;
    FrontMtx_writeForHumanEye(lu->frontmtx, lu->msgFile) ;
    fflush(lu->msgFile) ;
  }

  /* STEP 14: create a solution DenseMtx object */  
  lu->ownedColumnsIV = FrontMtx_ownedColumnsIV(lu->frontmtx, rank, lu->ownersIV,
                                         lu->msglvl, lu->msgFile) ;
  lu->nmycol = IV_size(lu->ownedColumnsIV) ;
  DenseMtx_init(lu->mtxX, SPOOLES_REAL, 0, 0, lu->nmycol, 1, 1, lu->nmycol) ;
  /* get pointers rowindX and entX */
  DenseMtx_rowIndices(lu->mtxX, &lu->nmycol, &lu->rowindX) ; 
  lu->entX = DenseMtx_entries(lu->mtxX) ; 
  
  lu->flg = SAME_NONZERO_PATTERN;
  PetscFunctionReturn(0);
}

extern int MatSolve_SeqAIJ_Spooles(Mat,Vec,Vec);
#undef __FUNCT__  
#define __FUNCT__ "MatFactorInfo_Spooles"
int MatFactorInfo_Spooles(Mat A,PetscViewer viewer)
{
  Mat_Spooles    *lu = (Mat_Spooles*)A->spptr; 
  int            ierr,size;
  char           *s;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  
  /* check if matrix is spooles type */
  if (size == 1){
    if (A->ops->solve != MatSolve_SeqAIJ_Spooles) PetscFunctionReturn(0);
  } else {
    if (A->ops->solve != MatSolve_MPIAIJ_Spooles) PetscFunctionReturn(0);
  }

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


