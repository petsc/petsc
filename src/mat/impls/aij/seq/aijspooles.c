/*$Id: aijspooles.c,v 1.10 2001/08/15 15:56:50 bsmith Exp $*/
/* 
   Provides an interface to the Spooles serial sparse solver
*/

#include "src/mat/impls/aij/seq/aij.h"
#include "src/mat/impls/aij/seq/spooles.h"

#if defined(PETSC_HAVE_SPOOLES) && !defined(PETSC_USE_SINGLE) && !defined(PETSC_USE_COMPLEX)

/* Note the Petsc r and c permutations are ignored */
#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorSymbolic_SeqAIJ_Spooles"
int MatLUFactorSymbolic_SeqAIJ_Spooles(Mat A,IS r,IS c,MatLUInfo *info,Mat *F)
{
  Mat_SeqAIJ           *mat = (Mat_SeqAIJ*)A->data;
  Mat_Spooles   *lu;   
  int                  ierr,m=A->m,n=A->n;
  FILE                 *msgFile ;
  int                  neqns,irow,count;  
  PetscScalar          *av  = mat->a;
  int                  *ai=mat->i,*aj=mat->j;
  Graph                *graph ;
  IVL                  *adjIVL;
  int                  nedges,*newToOld, *oldToNew ;
  char                 buff[32];
  char                 *ordertype[] = {"BestOfNDandMS","MMD","MS","ND"}; 
  PetscTruth           flg;
  int                  *ivec1, *ivec2, ii;
  double               *dvec;

  PetscFunctionBegin;	
  ierr = PetscNew(Mat_Spooles,&lu);CHKERRQ(ierr); 

  /* Create the factorization matrix F */  
  ierr = MatCreateSeqAIJ(A->comm,m,n,PETSC_NULL,PETSC_NULL,F);CHKERRQ(ierr);

  (*F)->ops->lufactornumeric  = MatLUFactorNumeric_SeqAIJ_Spooles;
  (*F)->ops->solve            = MatSolve_SeqAIJ_Spooles;
  (*F)->ops->destroy          = MatDestroy_SeqAIJ_Spooles;  
  (*F)->factor                = FACTOR_LU;  
  (*F)->spptr                 = (void*)lu;

  /* set default input parameters */
  lu->symflag       = SPOOLES_NONSYMMETRIC;
  lu->pivotingflag  = SPOOLES_NO_PIVOTING;
  lu->tau           = 100.;
  lu->seed          = 10101;  
  lu->ordering      = 0;
  lu->maxdomainsize = 500;
  lu->maxzeros      = 1000;
  lu->maxsize       = 96;

  /* get runtime input parameters */
  ierr = PetscOptionsBegin(A->comm,A->prefix,"Spooles Options","Mat");CHKERRQ(ierr); 
  /*
    ierr = PetscOptionsInt("-mat_aij_spooles_symflag","symmetryflag: \n\
           0: SPOOLES_SYMMETRIC, 2: SPOOLES_NONSYMMETRIC","None",lu->symflag,&lu->symflag,PETSC_NULL);CHKERRQ(ierr);
  */

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
   
    ierr = PetscOptionsInt("-mat_aij_spooles_maxdomainsize","maxdomainsize","None",lu->maxdomainsize,&lu->maxdomainsize,PETSC_NULL);CHKERRQ(ierr);

    ierr = PetscOptionsInt("-mat_aij_spooles_maxzeros ","maxzeros","None",lu->maxzeros,&lu->maxzeros,PETSC_NULL);CHKERRQ(ierr);

    ierr = PetscOptionsInt("-mat_aij_spooles_maxsize","maxsize","None",lu->maxsize,&lu->maxsize,PETSC_NULL);CHKERRQ(ierr);

  PetscOptionsEnd();

  if (info && info->dtcol > 0.0) {
    lu->pivotingflag = SPOOLES_PIVOTING; 
  } else {  
    lu->pivotingflag = SPOOLES_NO_PIVOTING; 
  }

  /* copy A to Spooles' InpMtx object */
  lu->nz=mat->nz;
  
  lu->mtxA = InpMtx_new() ;
  InpMtx_init(lu->mtxA, INPMTX_BY_ROWS, SPOOLES_REAL, lu->nz, m) ;
  ivec1 = InpMtx_ivec1(lu->mtxA);  
  ivec2 = InpMtx_ivec2(lu->mtxA); 
  dvec  = InpMtx_dvec(lu->mtxA);
  for (irow = 0; irow < m; irow++){
    for (ii = ai[irow]; ii<ai[irow+1]; ii++) ivec1[ii] = irow;
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
  newToOld   = IV_entries(lu->newToOldIV) ;
  ETree_permuteVertices(lu->frontETree, lu->oldToNewIV) ;

  InpMtx_permute(lu->mtxA, lu->oldToNew, lu->oldToNew) ; 
  if ( lu->symflag == SPOOLES_SYMMETRIC ) {
    InpMtx_mapToUpperTriangle(lu->mtxA) ; 
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
  }  
  
  PetscFunctionReturn(0); 
}

#undef __FUNCT__  
#define __FUNCT__ "MatUseSpooles_SeqAIJ"
int MatUseSpooles_SeqAIJ(Mat A)
{
  PetscFunctionBegin;
  A->ops->lufactorsymbolic = MatLUFactorSymbolic_SeqAIJ_Spooles;
  
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSeqAIJFactorInfo_Spooles"
int MatSeqAIJFactorInfo_Spooles(Mat A,PetscViewer viewer)
{
  Mat_Spooles      *lu = (Mat_Spooles*)A->spptr;  
  int                     ierr;
  char                    *s;

  PetscFunctionBegin;
  /* check if matrix is spooles type */
  if (A->ops->solve != MatSolve_SeqAIJ_Spooles) PetscFunctionReturn(0);

  ierr = PetscViewerASCIIPrintf(viewer,"Spooles run parameters:\n");CHKERRQ(ierr);

  switch (lu->symflag) {
  case 0: s = "SPOOLES_SYMMETRIC"; break;
  case 2: s = "SPOOLES_NONSYMMETRIC"; break; }
  ierr = PetscViewerASCIIPrintf(viewer,"  symmetryflag:  %s \n",s);CHKERRQ(ierr);

  switch (lu->pivotingflag) {
  case 0: s = "SPOOLES_NO_PIVOTING"; break;
  case 1: s = "SPOOLES_PIVOTING"; break; }
  ierr = PetscViewerASCIIPrintf(viewer,"  pivotingflag:  %s \n",s);CHKERRQ(ierr);

  ierr = PetscViewerASCIIPrintf(viewer,"  tau:           %g \n",lu->tau);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  seed:          %d \n",lu->seed);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  msglvl:        %d \n",lu->msglvl);CHKERRQ(ierr);

  switch (lu->ordering) {
  case 0: s = "BestOfNDandMS"; break;  
  case 1: s = "MMD"; break;
  case 2: s = "MS"; break;
  case 3: s = "ND"; break;
  }
  ierr = PetscViewerASCIIPrintf(viewer,"  ordering:      %s \n",s);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  maxdomainsize: %d \n",lu->maxdomainsize);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  maxzeros:      %d \n",lu->maxzeros);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  maxsize:       %d \n",lu->maxsize);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#else

#undef __FUNCT__  
#define __FUNCT__ "MatUseSpooles_SeqAIJ"
int MatUseSpooles_SeqAIJ(Mat A)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#endif


