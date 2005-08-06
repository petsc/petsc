/*  $Id: petsc_prom.c,v 1.7 2005/04/07 17:43:33 adams Exp $ */
/*  Author:             Mark F. Adams    */
/*  Copyright (c) 2004 by Mark F. Adams  */
/*  Filename:           petsc_prom.c     */
 
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

#include "src/ksp/pc/pcimpl.h"     /*I "petscpc.h" I*/

EXTERN PetscErrorCode PCCreate_Prometheus_private( PC pc );
EXTERN PetscErrorCode PCSetUp_Prometheus( PC pc );
EXTERN PetscErrorCode PCSetCoordinates_Prometheus( PC pc, PetscReal *coords );
EXTERN PetscErrorCode PCSetFromOptions_Prometheus(PC pc);
EXTERN PetscErrorCode PCSetUp_Prometheus_Symmetric(PC pc);
EXTERN PetscErrorCode PCSetUp_Prometheus_NonSymmetric(PC pc);
EXTERN PetscErrorCode PCApply_Prometheus( PC pc, Vec x, Vec y );
EXTERN PetscErrorCode PCApplySymmetricLeftOrRight_Prometheus(PC pc,Vec ,Vec );
EXTERN PetscErrorCode PCDestroy_Prometheus(PC pc);
EXTERN PetscErrorCode PCView_Prometheus( PC pc, PetscViewer viewer);

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
     PCPrometheus - Prometheus (i.e. diagonal scaling preconditioning)

   Options Database Key:
.    -pc_prometheus_rowmax - use the maximum absolute value in each row as the scaling factor,
                        rather than the diagonal

   Level: beginner

  Concepts: Prometheus, diagonal scaling, preconditioners

  Notes: By using KSPSetPreconditionerSide(ksp,PC_SYMMETRIC) or -ksp_symmetric_pc you 
         can scale each side of the matrix by the squareroot of the diagonal entries.

         Zero entries along the diagonal are replaced with the value 1.0

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC
M*/

EXTERN_C_BEGIN

#undef __FUNCT__  
#define __FUNCT__ "PCCreate_Prometheus"
PetscErrorCode PCCreate_Prometheus(PC pc)
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;

  ierr = PCCreate_Prometheus_private( pc ); CHKERRQ(ierr);
  
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
  pc->ops->applysymmetricleft  = 0;/* PCApplySymmetricLeftOrRight_Prometheus; */
  pc->ops->applysymmetricright = 0;/* PCApplySymmetricLeftOrRight_Prometheus; */
  ierr = PetscObjectComposeFunctionDynamic( (PetscObject)pc,
					    "PCSetCoordinates_Prometheus_C",
					    "PCSetCoordinates_Prometheus",
					    PCSetCoordinates_Prometheus);
  CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_END 
#undef __FUNCT__
#define __FUNCT__ "PCPrometheusSetCoordinates"
/*@
   PCPrometheusSetCoordinates - sets the coordinates of all the nodes

   Collective on PC

   Input Parameters:
+  pc - the solver context
-  coords - the coordinates

   Level: intermediate

.seealso: PCPROMETHEUS
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCPrometheusSetCoordinates(PC pc,PetscReal *coords)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscTryMethod(pc,PCSetCoordinates_Prometheus_C,(PC,PetscReal*),(pc,coords));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
