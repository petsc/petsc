/*$Id: spooles.c,v 1.10 2001/08/15 15:56:50 bsmith Exp $*/
/* 
   Provides an interface to the Spooles serial sparse solver
*/

#include "src/mat/impls/aij/seq/aij.h"

#if defined(PETSC_HAVE_SPOOLES) && !defined(PETSC_USE_SINGLE) && !defined(PETSC_USE_COMPLEX)

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
  int             msglvl,pivotingflag,symmetryflag,seed;
  FILE            *msgFile ;
  SubMtxManager   *mtxmanager  ;  /* working array */
  double          cpus[10] ; 
  MatStructure    flg;
  int             *oldToNew,nz;
  double          tau;
} Mat_SeqAIJ_Spooles;

extern int MatDestroy_SeqAIJ(Mat); 

#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_SeqAIJ_Spooles"
int MatDestroy_SeqAIJ_Spooles(Mat A)
{
  Mat_SeqAIJ         *a  = (Mat_SeqAIJ*)A->data; 
  Mat_SeqAIJ_Spooles *lu = (Mat_SeqAIJ_Spooles*)a->spptr; 
  int                ierr;
  
  PetscFunctionBegin;
  /* printf(" MatDestroy is called ...\n"); */

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
  Mat_SeqAIJ              *aa = (Mat_SeqAIJ*)A->data;
  Mat_SeqAIJ_Spooles      *lu = (Mat_SeqAIJ_Spooles*)aa->spptr;
  int                     ierr;
  PetscScalar             *array;
  DenseMtx                *mtxY, *mtxX ;
  int                     irow,neqns=A->m,*iv;
  double                  cpus[10] ;

  PetscFunctionBegin;
  printf(" Solve_SeqAIJ_Spooles is called ...,\n"); 

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
    DenseMtx_writeForHumanEye(mtxY, lu->msgFile) ; /* moved from symbfact() */
    fprintf(lu->msgFile, "\n\n solution matrix in new ordering") ;
    DenseMtx_writeForHumanEye(mtxX, lu->msgFile) ;
    fflush(lu->msgFile) ;
  }

  /* permute solution into original ordering, then copy to x */
  iv   = IV_entries(lu->newToOldIV);
  ierr = VecGetArray(x,&array);CHKERRQ(ierr); 
  for ( irow = 0 ; irow < neqns ; irow++ ) DenseMtx_realEntry(mtxX,irow,0,&array[iv[irow]]);  
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
  Mat_SeqAIJ         *fac = (Mat_SeqAIJ*)(*F)->data, *mat = (Mat_SeqAIJ*)A->data;
  Mat_SeqAIJ_Spooles *lu = (Mat_SeqAIJ_Spooles*)fac->spptr;
  ChvManager         *chvmanager  ;
  FrontMtx           *frontmtx ; 
  int                stats[20],error,pivotingflag=1,nz,m=A->m,irow,count;
  Chv                *rootchv ;
  int                *ai=mat->i,*aj=mat->j;
  PetscScalar        *av  = mat->a;

  PetscFunctionBegin;
  printf(" Num_SeqAIJ_Spooles is called ..., lu->flg: %d\n", lu->flg); 

  if ( lu->flg == SAME_NONZERO_PATTERN){ /* use previously computed symbolic factor */
    /* get new numerical values of A , then permute the matrix */ 
    InpMtx_init(lu->mtxA, INPMTX_BY_ROWS, SPOOLES_REAL, lu->nz, m) ; 
    for (irow = 0; irow < m; irow++){
      count = ai[1] - ai[0];
      InpMtx_inputRealRow(lu->mtxA, irow, count, aj, av);
      ai++; aj += count; av += count;
    }
    InpMtx_changeStorageMode(lu->mtxA, INPMTX_BY_VECTORS) ; 

    InpMtx_permute(lu->mtxA, lu->oldToNew, lu->oldToNew) ;
    if ( lu->symmetryflag == SPOOLES_SYMMETRIC ) InpMtx_mapToUpperTriangle(lu->mtxA) ; 
    InpMtx_changeCoordType(lu->mtxA, INPMTX_BY_CHEVRONS) ;
    InpMtx_changeStorageMode(lu->mtxA, INPMTX_BY_VECTORS) ;
    if ( lu->msglvl > 2 ) {
      fprintf(lu->msgFile, "\n\n input matrix after permutation") ;
      InpMtx_writeForHumanEye(lu->mtxA, lu->msgFile) ; 
    } 
  }

  /* initialize the front matrix object */
  if ( lu->flg == DIFFERENT_NONZERO_PATTERN){ 
    lu->frontmtx   = FrontMtx_new() ;
    lu->mtxmanager = SubMtxManager_new() ;
  }
  SubMtxManager_init(lu->mtxmanager, NO_LOCK, 0) ;
  FrontMtx_init(lu->frontmtx, lu->frontETree, lu->symbfacIVL, SPOOLES_REAL, lu->symmetryflag, 
                FRONTMTX_DENSE_FRONTS, lu->pivotingflag, NO_LOCK, 0, NULL, 
                lu->mtxmanager, lu->msglvl, lu->msgFile) ;   

  /* numerical factorization */
  chvmanager = ChvManager_new() ;
  ChvManager_init(chvmanager, NO_LOCK, 1) ;
  DVfill(10, lu->cpus, 0.0) ;
  IVfill(20, stats, 0) ;
  rootchv = FrontMtx_factorInpMtx(lu->frontmtx, lu->mtxA, lu->tau, 0.0, 
            chvmanager, &error, lu->cpus, stats, lu->msglvl, lu->msgFile) ; 
  ChvManager_free(chvmanager) ;
  if ( lu->msglvl > 0 ) {
    fprintf(lu->msgFile, "\n\n factor matrix") ;
    FrontMtx_writeForHumanEye(lu->frontmtx, lu->msgFile) ;
    fflush(lu->msgFile) ;
  }
  if ( rootchv != NULL ) {
    fprintf(lu->msgFile, "\n\n matrix found to be singular\n") ;
    exit(-1) ;
  }
  if ( error >= 0 ) {
    fprintf(lu->msgFile, "\n\n error encountered at front %d", error) ;
    exit(-1) ;
  }

  /* post-process the factorization */
  FrontMtx_postProcess(lu->frontmtx, lu->msglvl, lu->msgFile) ;
  if ( lu->msglvl > 2 ) {
    fprintf(lu->msgFile, "\n\n factor matrix after post-processing") ;
    FrontMtx_writeForHumanEye(lu->frontmtx, lu->msgFile) ;
    fflush(lu->msgFile) ;
  }

  lu->flg = SAME_NONZERO_PATTERN;
  PetscFunctionReturn(0);
}

/* Note the Petsc r and c permutations are ignored */
#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorSymbolic_SeqAIJ_Spooles"
int MatLUFactorSymbolic_SeqAIJ_Spooles(Mat A,IS r,IS c,MatLUInfo *info,Mat *F)
{
  Mat_SeqAIJ           *fac,*mat = (Mat_SeqAIJ*)A->data;
  Mat_SeqAIJ_Spooles   *lu;   
  int                  ierr,m=A->m,n=A->n;
  FILE                 *msgFile ;
  int                  neqns,irow,count;  
  PetscScalar          *av  = mat->a;
  int                  *ai=mat->i,*aj=mat->j;
  Graph                *graph ;
  IVL                  *adjIVL;
  int                  nedges,*newToOld, *oldToNew ;

  PetscFunctionBegin;	
  printf(" Symbolic_SeqAIJ_Spooles is called ...\n"); 

  /* Create the factorization matrix F */  
  ierr = MatCreateSeqAIJ(A->comm,m,n,PETSC_NULL,PETSC_NULL,F);CHKERRQ(ierr);

  (*F)->ops->lufactornumeric  = MatLUFactorNumeric_SeqAIJ_Spooles;
  (*F)->ops->solve            = MatSolve_SeqAIJ_Spooles;
  (*F)->ops->destroy          = MatDestroy_SeqAIJ_Spooles;  
  (*F)->factor                = FACTOR_LU;  
  fac                         = (Mat_SeqAIJ*)(*F)->data; 

  ierr                        = PetscNew(Mat_SeqAIJ_Spooles,&lu);CHKERRQ(ierr); 
  fac->spptr                  = (void*)lu;

  /* get input parameters */
  lu->symmetryflag = 2; /* 0: symmetric entries, 1: Hermitian entries, 2: non-symmetric entries -- 
                                if A is aij: non-symmetric, if A is sbaij: symmetric, other cookie: error msg */
  lu->tau  = 100.;
  lu->seed = 10101;  /* random number seed, used for ordering */

  ierr = PetscOptionsBegin(A->comm,A->prefix,"Spooles Options","Mat");CHKERRQ(ierr); 
    ierr = PetscOptionsInt("-mat_aij_spooles_symmetryflag","symmetryflag","None",lu->symmetryflag,&lu->symmetryflag,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-mat_aij_spooles_tau","tau (used for pivoting; all entries in L and U have magnitude no more than tau)","None",lu->tau,&lu->tau,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-mat_aij_spooles_seed","seed","None",lu->seed,&lu->seed,PETSC_NULL);CHKERRQ(ierr);

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
  PetscOptionsEnd();

  lu->pivotingflag = SPOOLES_NO_PIVOTING; 
  if (info && info->dtcol > 0.0) {
    lu->pivotingflag = SPOOLES_PIVOTING;
    if ( lu->msglvl > 0 ) fprintf(lu->msgFile, "\n pivoting is used\n") ;
  }

  /* convert A to Spooles' InpMtx object */
  if (lu->symmetryflag == 2){
    lu->nz=mat->nz;
  } else {
    SETERRQ(1,"Mat must be in SeqAIJ format");
    /* 
  } else if (lu->symmetryflag == 2){
    nz=mat->s_nz;
    */
  }
  
  lu->mtxA = InpMtx_new() ;
  InpMtx_init(lu->mtxA, INPMTX_BY_ROWS, SPOOLES_REAL, lu->nz, m) ;
  
  for (irow = 0; irow < m; irow++){
    count = ai[1] - ai[0];
    InpMtx_inputRealRow(lu->mtxA, irow, count, aj, av);
    ai++; aj += count; av += count;
  }
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
  lu->frontETree = orderViaMMD(graph, lu->seed, lu->msglvl, lu->msgFile) ;
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
  newToOld   = IV_entries(lu->newToOldIV) ;
  ETree_permuteVertices(lu->frontETree, lu->oldToNewIV) ;

  InpMtx_permute(lu->mtxA, lu->oldToNew, lu->oldToNew) ; 
  if ( lu->symmetryflag == SPOOLES_SYMMETRIC ) {
    InpMtx_mapToUpperTriangle(lu->mtxA) ; /* ensures all entries of PAP' are in upper triangle */
  }
  InpMtx_changeCoordType(lu->mtxA, INPMTX_BY_CHEVRONS) ;
  InpMtx_changeStorageMode(lu->mtxA, INPMTX_BY_VECTORS) ;

  /* get symbolic factorization */
  lu->symbfacIVL = SymbFac_initFromInpMtx(lu->frontETree, lu->mtxA) ;

  lu->flg = DIFFERENT_NONZERO_PATTERN;

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
    /* Tree_writeToFile(lu->frontETree->tree, "haggar.treef") ; */
  }  
  
  PetscFunctionReturn(0); 
}

#undef __FUNCT__  
#define __FUNCT__ "MatUseSpooles_SeqAIJ"
int MatUseSpooles_SeqAIJ(Mat A)
{
  PetscFunctionBegin;
  /* printf(" MatUseSpooles is called ...\n"); */
  A->ops->lufactorsymbolic = MatLUFactorSymbolic_SeqAIJ_Spooles;
  A->ops->lufactornumeric  = MatLUFactorNumeric_SeqAIJ_Spooles;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSeqAIJFactorInfo_Spooles"
int MatSeqAIJFactorInfo_Spooles(Mat A,PetscViewer viewer)
{
  Mat_SeqAIJ              *fac = (Mat_SeqAIJ*)(A)->data;
  Mat_SeqAIJ_Spooles      *lu = (Mat_SeqAIJ_Spooles*)fac->spptr;  
  int                     ierr;
  
  PetscFunctionBegin;
  /* check if matrix is spooles type */
  if (A->ops->solve != MatSolve_SeqAIJ_Spooles) PetscFunctionReturn(0);

  ierr = PetscViewerASCIIPrintf(viewer,"Spooles run parameters:\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  tau: %g \n",lu->tau);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  seed %d \n",lu->seed);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  msglvl %d \n",lu->msglvl);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#else

#undef __FUNCT__  
#define __FUNCT__ "MatUseSpooles"
int MatUseSpooles(Mat A)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#endif


