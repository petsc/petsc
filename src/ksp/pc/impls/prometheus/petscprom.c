#define PETSCKSP_DLL

//  Contributed by:             Mark F. Adams
//  Copyright (c) 2004 by Mark F. Adams 
//  Filename:           petscprom.c
//
//  This code may be copied and redistributed for any purpose
//  including use in proprietary codes.
/*  -------------------------------------------------------------------- 

     This file implements a Prometheus preconditioner for matrices that use
     the Mat interface (various matrix formats).  This wraps the Prometheus
     class - this is a C intercace to a C++ code.

     Prometheus assumes that 'PetscScalar' is 'double'.  Prometheus does 
     have a complex-valued solver, but this is runtime parameter, not a 
     compile time parameter.

     The following basic routines are required for each preconditioner.
          PCCreate_XXX()          - Creates a preconditioner context
          PCSetFromOptions_XXX()  - Sets runtime options
          PCApply_XXX()           - Applies the preconditioner
          PCDestroy_XXX()         - Destroys the preconditioner context
     where the suffix "_XXX" denotes a particular implementation, in
     this case we use _Prometheus (e.g., PCCreate_Prometheus, PCApply_Prometheus).
     These routines are actually called via the common user interface
     routines PCCreate(), PCSetFromOptions(), PCApply(), and PCDestroy(), 
     so the application code interface remains identical for all 
     preconditioners.  

     Another key routine is:
          PCSetUp_XXX()           - Prepares for the use of a preconditioner
     by setting data structures and options.   The interface routine PCSetUp()
     is not usually called directly by the user, but instead is called by
     PCApply() if necessary.

     Additional basic routines are:
          PCView_XXX()            - Prints details of runtime options that
                                    have actually been used.
     These are called by application codes via the interface routines
     PCView().

     The various types of solvers (preconditioners, Krylov subspace methods,
     nonlinear solvers, timesteppers) are all organized similarly, so the
     above description applies to these categories also.  One exception is
     that the analogues of PCApply() for these components are KSPSolve(), 
     SNESSolve(), and TSSolve().

     Additional optional functionality unique to preconditioners is left and
     right symmetric preconditioner application via PCApplySymmetricLeft() 
     and PCApplySymmetricRight().  The Prometheus implementation is 
     PCApplySymmetricLeftOrRight_Prometheus().

    -------------------------------------------------------------------- */

/* 
   Include files needed for the Prometheus preconditioner:
     pcimpl.h - private include file intended for use by all preconditioners 
*/

#include "src/ksp/pc/pcimpl.h"   /*I "petscpc.h" I*/
#define PROM_NO_FEI
#include "Prometheus_LinSysCore.h"
#include "prometheus.hh"

/* -------------------------------------------------------------------------- */
/*
   PCSetUp_Prometheus - Prepares for the use of the Prometheus preconditioner
   by setting data structures and options.   
   
   Input Parameter:
.  pc - the preconditioner context

   Application Interface Routine: PCSetUp()

   Notes:
   The interface routine PCSetUp() is not usually called directly by
   the user, but instead is called by PCApply() if necessary.
*/
#undef __FUNCT__  
#define __FUNCT__ "PCSetUp_Prometheus"
static PetscErrorCode PCSetUp_Prometheus( PC pc )
{
  Prometheus_LinSysCore *lsc = (Prometheus_LinSysCore*)pc->data; 
  PetscErrorCode ierr; PetscInt bs, my0, tt, ii, npe = lsc->Comm_->np();
  
  PetscFunctionBegin;
  // create base
  assert(lsc!=NULL);
  if (pc->setupcalled == 0) { // allocate space the first time this is ever called
    lsc->prom_->perf_mon_.EventBegin(PromContext::prom_logs_[LOG_PROM]);
    pc->setupcalled = 1;
    // 
    Mat mat = pc->mat; assert(mat != NULL); // graph
    ierr = MatGetBlockSize( mat, &bs );CHKERRQ( ierr );
    ierr = MatGetOwnershipRange( mat, &my0, &tt );CHKERRQ(ierr);
    const PetscInt nloc = tt - my0;
    {// construct LSC
      PromMap map( nloc, bs, *lsc->Comm_ );
      const int *proc_gnode = map.proc_gnode();
      int proc_block[PROM_BSZ], proc_geq[PROM_BSZ];
      for( ii = proc_block[0] = proc_geq[0] = 0 ; ii < npe ; ii++ ){
	proc_block[ii+1] = proc_gnode[ii+1];   // + CR[ii+1]
	proc_geq[ii+1] = bs*proc_gnode[ii+1]; // + CR[ii+1]
      }
      ierr = lsc->setGlobalOffsets(npe+1, proc_gnode, proc_geq, proc_block);CHKERRQ(ierr);
    }
    //
    tt = 1;
    ierr = lsc->setNumRHSVectors( 1, &tt );CHKERRQ(ierr);
    {  // construct graph
      int **blkColIndices,*blkRowLengths,*ptRowsPerBlkRow,xx,gid,*pb;
      const PetscInt *cols; PetscInt ncols;
      ierr = PetscMalloc( (nloc+1)*sizeof(int*), &blkColIndices );CHKERRQ(ierr);
      ierr = PetscMalloc( (nloc+1)*sizeof(int), &blkRowLengths );CHKERRQ(ierr);
      ierr = PetscMalloc( (nloc+1)*sizeof(int), &ptRowsPerBlkRow );CHKERRQ(ierr);
      for( xx = 0, gid = my0 ; xx < nloc ; xx++, gid++ ) {
	ierr = MatGetRow( mat, gid, &ncols, &cols, NULL );CHKERRQ(ierr);
	tt = blkRowLengths[xx] = ncols;
	ptRowsPerBlkRow[xx] = bs;
	ierr = PetscMalloc( (tt+1)*sizeof(int), &pb );CHKERRQ(ierr);
	blkColIndices[xx] = pb;
	for( ii = 0 ; ii < ncols ; ii++ ) *pb++ = cols[ii];
	MatRestoreRow( mat, gid, &ncols, &cols, NULL );CHKERRQ(ierr );
      }
      // call FEI setMatrixStructure
      ierr = lsc->setMatrixStructure( NULL, NULL, blkColIndices, blkRowLengths, ptRowsPerBlkRow, mat );CHKERRQ(ierr);
      //if(mype==0)PromContext::printArr(stderr,blkRowLengths,nloc,"blkRowLengths[%d]",18);
      // clean up
      for( xx = 0 ; xx < nloc ; xx++ ) {
	ierr = PetscFree(blkColIndices[xx]);CHKERRQ(ierr);
      }
      ierr = PetscFree( blkColIndices );CHKERRQ(ierr);
      ierr = PetscFree( blkRowLengths );CHKERRQ(ierr);
      ierr = PetscFree( ptRowsPerBlkRow );CHKERRQ(ierr);
    }
    lsc->prom_->perf_mon_.EventEnd(PromContext::prom_logs_[LOG_PROM]);
  }
  // factor
  if ( pc->setupcalled == 1 ) { // 'factor' matrix 
    pc->setupcalled = 2;
    // factor 
    ierr = lsc->prom_->grid0_->stiffness_->SetDirty();CHKERRQ(ierr);
    ierr = lsc->matrixLoadComplete();CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
/*
   PCSetCoordinates_Prometheus - 
   
   Input Parameter:
   .  pc - the preconditioner context
*/
#undef __FUNCT__  
#define __FUNCT__ "PCSetCoordinates_Prometheus"
static PetscErrorCode PCSetCoordinates_Prometheus( PC pc, PetscReal *coords )
{
  Prometheus_LinSysCore *lsc = (Prometheus_LinSysCore*)pc->data; 
  PetscErrorCode ierr;  PetscInt bs, my0, tt;

  PetscFunctionBegin;  
  Mat mat = pc->mat; assert(mat != NULL); // graph
  ierr = MatGetBlockSize( mat, &bs );CHKERRQ( ierr );
  ierr = MatGetOwnershipRange( mat, &my0, &tt );CHKERRQ(ierr);
  const PetscInt nloc = tt - my0;

  // put coordinates
  ierr = lsc->putNodalFieldData(-3, 3, NULL, nloc, coords );CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCSetFromOptions_Prometheus - Set options from options database - not used!
   Prometheus automatically reads parameters from the database

   Input Parameter:
.  pc - the preconditioner context
*/
#undef __FUNCT__  
#define __FUNCT__ "PCSetFromOptions_Prometheus"
static PetscErrorCode PCSetFromOptions_Prometheus(PC pc)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;  
  ierr = PetscOptionsHead("Prometheus options");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCApply_Prometheus - Applies the Prometheus preconditioner to a vector.

   Input Parameters:
.  pc - the preconditioner context
.  x - input vector

   Output Parameter:
.  y - output vector

   Application Interface Routine: PCApply()
 */
#undef __FUNCT__  
#define __FUNCT__ "PCApply_Prometheus"
static PetscErrorCode PCApply_Prometheus( PC pc, Vec x, Vec y )
{
  Prometheus_LinSysCore *lsc = (Prometheus_LinSysCore*)pc->data; 
  Prometheus *prom = lsc->prom_;
  PetscErrorCode ierr; 

  PetscFunctionBegin;
  lsc->nSolves_++;

  // get RHS
  PromVector *work = lsc->getRHS();
  if( prom->left_ != NULL ) { // scale f <- f * D^{-1/2}
    ierr = VecPointwiseMult( work->vec_, x, prom->left_->vec_ );CHKERRQ(ierr);
  }
  else {
    ierr = VecCopy( x,  work->vec_ );CHKERRQ(ierr);
  }

  // solve
  prom->perf_mon_.StagePush(PromContext::prom_stages_[0]);
  prom->perf_mon_.EventBegin(PromContext::prom_logs_[LOG_FEI_SOLVE]);
  PromVector *XX = lsc->getSolution(); 
  ierr = XX->Set( 0.0 );CHKERRQ(ierr);  assert(prom->solver_ != NULL);
  ierr = prom->solver_->pc_->Apply( work, XX, TRUE );CHKERRQ(ierr);
  prom->perf_mon_.EventEnd(PromContext::prom_logs_[LOG_FEI_SOLVE]);
  prom->perf_mon_.StagePop();

  // return solution
  if( prom->right_ != NULL ) { // scale u <- D^{-1/2} * u
    ierr = VecPointwiseMult( y, XX->vec_, prom->right_->vec_ );CHKERRQ(ierr);
  }
  else {
    ierr = VecCopy( XX->vec_, y );CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCDestroy_Prometheus - Destroys the private context for the Prometheus preconditioner
   that was created with PCCreate_Prometheus().

   Input Parameter:
.  pc - the preconditioner context

   Application Interface Routine: PCDestroy()
*/
#undef __FUNCT__  
#define __FUNCT__ "PCDestroy_Prometheus"
static PetscErrorCode PCDestroy_Prometheus(PC pc)
{
  Prometheus_LinSysCore *lsc = (Prometheus_LinSysCore*)pc->data; 

  PetscFunctionBegin;
  /*
      Free the private data structure that was hanging off the PC
  */
  delete lsc; pc->data = NULL;
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCView_Prometheus - 

   Input Parameter:
.  pc - the preconditioner context

   Application Interface Routine: PCView()
*/
#undef __FUNCT__  
#define __FUNCT__ "PCView_Prometheus"
static PetscErrorCode PCView_Prometheus( PC pc, PetscViewer viewer)
{
  Prometheus_LinSysCore *lsc = (Prometheus_LinSysCore*)pc->data; 
  PetscErrorCode ierr; PetscTruth     iascii;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  if( iascii ) {
    ierr = lsc->prom_->solver_->PrintInfo( lsc->Comm_->np() );CHKERRQ(ierr);
  } else {
    SETERRQ1(PETSC_ERR_SUP,"Viewer type %s not supported for Prometheus",
	     ((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCCreate_Prometheus - Creates a Prometheus preconditioner context, Prometheus, 
   and sets this as the private data within the generic preconditioning 
   context, PC, that was created within PCCreate().

   Input Parameter:
.  pc - the preconditioner context

   Application Interface Routine: PCCreate()
*/

/*MC
     PCPrometheus - Provides a PC interface to Mark Adams scalable algebraic multigrid solver


   Level: intermediate

  Concepts: Prometheus, AMG, algebraic multigrid


.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC
M*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCCreate_Prometheus"
PetscErrorCode PETSCKSP_DLLEXPORT PCCreate_Prometheus(PC pc)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*
    Creates the private data structure for this preconditioner and
    attach it to the PC object.
  */
  assert(pc->data==NULL);
  Prometheus_LinSysCore *lsc = new Prometheus_LinSysCore(pc->comm);
  pc->data  = (void*)lsc;
  
  // will create prometheus (if necc.)
  char *argv[] = {"-ksp_type preonly"};
  ierr = lsc->parameters( 0, argv );CHKERRQ(ierr);
  lsc->mg_levels_ = 10; // hard wires for 'mg'

  /*
    Logs the memory usage; this is not needed but allows PETSc to 
    monitor how much memory is being used for various purposes.
  */
  PetscLogObjectMemory(pc,sizeof(Prometheus));
  
  /*
    Set the pointers for the functions that are provided above.
    Now when the user-level routines (such as PCApply(), PCDestroy(), etc.)
    are called, they will automatically call these functions.  Note we
    choose not to provide a couple of these functions since they are
    not needed.
  */
  pc->ops->apply               = PCApply_Prometheus;
  pc->ops->applytranspose      = PCApply_Prometheus;
  pc->ops->setup               = PCSetUp_Prometheus;
  pc->ops->destroy             = PCDestroy_Prometheus;
  pc->ops->setfromoptions      = PCSetFromOptions_Prometheus;
  pc->ops->view                = PCView_Prometheus;
  pc->ops->applyrichardson     = 0;
  pc->ops->applysymmetricleft  = 0;//PCApplySymmetricLeftOrRight_Prometheus;
  pc->ops->applysymmetricright = 0;//PCApplySymmetricLeftOrRight_Prometheus;
  ierr = PetscObjectComposeFunctionDynamic( (PetscObject)pc,"PCSetCoordinates_Prometheus_C","PCSetCoordinates_Prometheus",
					    PCSetCoordinates_Prometheus);CHKERRQ(ierr);
  assert (pc->setupcalled == 0);
  PetscFunctionReturn(0);
}
EXTERN_C_END
