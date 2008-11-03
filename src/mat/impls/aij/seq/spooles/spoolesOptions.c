#define PETSCMAT_DLL

/* 
   Default and runtime options used by seq and MPI Spooles' interface for both aij and sbaij mat objects
*/

#include "../src/mat/impls/aij/seq/spooles/spooles.h"

/* Set Spooles' default and runtime options */
#undef __FUNCT__  
#define __FUNCT__ "SetSpoolesOptions"
PetscErrorCode SetSpoolesOptions(Mat A, Spooles_options *options)
{
  PetscErrorCode ierr;
  int          indx;
  const char   *ordertype[]={"BestOfNDandMS","MMD","MS","ND"};
  PetscTruth   flg;

  PetscFunctionBegin;	
  /* set default input parameters */ 
#if defined(PETSC_USE_COMPLEX)
  options->typeflag       = SPOOLES_COMPLEX;
#else
  options->typeflag       = SPOOLES_REAL;
#endif
  options->msglvl         = 0;
  options->msgFile        = 0;
  options->tau            = 100.; 
  options->seed           = 10101;  
  options->ordering       = 1;     /* MMD */
  options->maxdomainsize  = 500;
  options->maxzeros       = 1000;
  options->maxsize        = 96;   
  options->FrontMtxInfo   = PETSC_FALSE; 
  if ( options->symflag == SPOOLES_SYMMETRIC ) { /* || SPOOLES_HERMITIAN */
    options->patchAndGoFlag = 0;  /* no patch */
    options->storeids       = 1; 
    options->storevalues    = 1;
    options->toosmall       = 1.e-9;
    options->fudge          = 1.e-9;
    
  } 

  /* get runtime input parameters */
  ierr = PetscOptionsBegin(((PetscObject)A)->comm,((PetscObject)A)->prefix,"Spooles Options","Mat");CHKERRQ(ierr); 

    ierr = PetscOptionsReal("-mat_spooles_tau","tau (used for pivoting; \n\
           all entries in L and U have magnitude no more than tau)","None",
                            options->tau,&options->tau,PETSC_NULL);CHKERRQ(ierr);

    ierr = PetscOptionsInt("-mat_spooles_seed","random number seed, used for ordering","None",
                           options->seed,&options->seed,PETSC_NULL);CHKERRQ(ierr);

    if (PetscLogPrintInfo) options->msglvl = 1;
    ierr = PetscOptionsInt("-mat_spooles_msglvl","msglvl","None",
                           options->msglvl,&options->msglvl,0);CHKERRQ(ierr); 
    if (options->msglvl > 0) {
        options->msgFile = fopen("spooles.msgFile", "a");
        PetscPrintf(PETSC_COMM_SELF,"\n Spooles' output is written into the file 'spooles.msgFile' \n\n");
    } 

    ierr = PetscOptionsEList("-mat_spooles_ordering","ordering type","None",ordertype,4,ordertype[1],&indx,&flg);CHKERRQ(ierr);
    if (flg) options->ordering = indx;
   
    ierr = PetscOptionsInt("-mat_spooles_maxdomainsize","maxdomainsize","None",\
                           options->maxdomainsize,&options->maxdomainsize,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-mat_spooles_maxzeros ","maxzeros","None",\
                           options->maxzeros,&options->maxzeros,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-mat_spooles_maxsize","maxsize","None",\
                           options->maxsize,&options->maxsize,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-mat_spooles_FrontMtxInfo","FrontMtxInfo","None",PETSC_FALSE,&flg,0);CHKERRQ(ierr);
    if (flg) options->FrontMtxInfo = PETSC_TRUE; 

    if ( options->symflag == SPOOLES_SYMMETRIC ) {
      ierr = PetscOptionsInt("-mat_spooles_symmetryflag","matrix type","None", \
                           options->symflag,&options->symflag,PETSC_NULL);CHKERRQ(ierr);

      ierr = PetscOptionsInt("-mat_spooles_patchAndGoFlag","patchAndGoFlag","None", \
                           options->patchAndGoFlag,&options->patchAndGoFlag,PETSC_NULL);CHKERRQ(ierr);
      
      ierr = PetscOptionsReal("-mat_spooles_fudge","fudge","None", \
                           options->fudge,&options->fudge,PETSC_NULL);CHKERRQ(ierr);
      ierr = PetscOptionsReal("-mat_spooles_toosmall","toosmall","None", \
                           options->toosmall,&options->toosmall,PETSC_NULL);CHKERRQ(ierr);
      ierr = PetscOptionsInt("-mat_spooles_storeids","storeids","None", \
                           options->storeids,&options->storeids,PETSC_NULL);CHKERRQ(ierr);
      ierr = PetscOptionsInt("-mat_spooles_storevalues","storevalues","None", \
                           options->storevalues,&options->storevalues,PETSC_NULL);CHKERRQ(ierr);    
    }
  PetscOptionsEnd(); 

  PetscFunctionReturn(0);
}

/* used by -ksp_view */
#undef __FUNCT__  
#define __FUNCT__ "MatFactorInfo_Spooles"
PetscErrorCode MatFactorInfo_Spooles(Mat A,PetscViewer viewer)
{
  Mat_Spooles    *lu = (Mat_Spooles*)A->spptr; 
  PetscErrorCode ierr;
  int            size;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(((PetscObject)A)->comm,&size);CHKERRQ(ierr); 
  /* check if matrix is spooles type */
  if (size == 1){
    if (A->ops->solve != MatSolve_SeqSpooles) PetscFunctionReturn(0);
  } else {
    if (A->ops->solve != MatSolve_MPISpooles) PetscFunctionReturn(0);
  }

  ierr = PetscViewerASCIIPrintf(viewer,"Spooles run parameters:\n");CHKERRQ(ierr);
  switch (lu->options.symflag) {
  case 0: 
    ierr = PetscViewerASCIIPrintf(viewer,"  symmetryflag:   SPOOLES_SYMMETRIC");CHKERRQ(ierr);
    break;
  case 1: 
    ierr = PetscViewerASCIIPrintf(viewer,"  symmetryflag:    SPOOLES_HERMITIAN\n");CHKERRQ(ierr);
    break;
  case 2: 
    ierr = PetscViewerASCIIPrintf(viewer,"  symmetryflag:    SPOOLES_NONSYMMETRIC\n");CHKERRQ(ierr);
    break; 
  }

  switch (lu->options.pivotingflag) {
  case 0: 
    ierr = PetscViewerASCIIPrintf(viewer,"  pivotingflag:   SPOOLES_NO_PIVOTING\n");CHKERRQ(ierr);
    break;
  case 1: 
    ierr = PetscViewerASCIIPrintf(viewer,"  pivotingflag:   SPOOLES_PIVOTING\n");CHKERRQ(ierr);
    break; 
  }
  ierr = PetscViewerASCIIPrintf(viewer,"  tau:            %g \n",lu->options.tau);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  seed:           %D \n",lu->options.seed);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  msglvl:         %D \n",lu->options.msglvl);CHKERRQ(ierr);

  switch (lu->options.ordering) {
  case 0: 
    ierr = PetscViewerASCIIPrintf(viewer,"  ordering:       BestOfNDandMS\n");CHKERRQ(ierr);
    break;  
  case 1: 
    ierr = PetscViewerASCIIPrintf(viewer,"  ordering:       MMD\n");CHKERRQ(ierr);
    break;
  case 2: 
    ierr = PetscViewerASCIIPrintf(viewer,"  ordering:       MS\n");CHKERRQ(ierr);
    break;
  case 3: 
    ierr = PetscViewerASCIIPrintf(viewer,"  ordering:       ND\n");CHKERRQ(ierr);
    break;
  }
  ierr = PetscViewerASCIIPrintf(viewer,"  maxdomainsize:  %D \n",lu->options.maxdomainsize);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  maxzeros:       %D \n",lu->options.maxzeros);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  maxsize:        %D \n",lu->options.maxsize);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  FrontMtxInfo:   %D \n",lu->options.FrontMtxInfo);CHKERRQ(ierr);

  if ( lu->options.symflag == SPOOLES_SYMMETRIC ) {
    ierr = PetscViewerASCIIPrintf(viewer,"  patchAndGoFlag: %D \n",lu->options.patchAndGoFlag);CHKERRQ(ierr);
    if ( lu->options.patchAndGoFlag > 0 ) {
      ierr = PetscViewerASCIIPrintf(viewer,"  fudge:          %g \n",lu->options.fudge);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  toosmall:       %g \n",lu->options.toosmall);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  storeids:       %D \n",lu->options.storeids);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  storevalues:    %D \n",lu->options.storevalues);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}
