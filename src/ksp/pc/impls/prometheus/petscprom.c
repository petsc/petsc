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

#include "private/pcimpl.h"     /*I "petscpc.h" I*/
#include "petscpromproto.h"

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
     PCPROMETHEUS - Prometheus (i.e. diagonal scaling preconditioning)

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

  ierr = PCCreate_Prometheus_private( pc );CHKERRQ(ierr);
  
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
					    "PCSetCoordinates_C",
					    "PCSetCoordinates_Prometheus",
					    PCSetCoordinates_Prometheus);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic( (PetscObject)pc,
					    "PCSASetVectors_C",
					    "PCSASetVectors_Prometheus",
					    PCSASetVectors_Prometheus);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_END 
#undef __FUNCT__
#define __FUNCT__ "PCSetCoordinates"
/*@
   PCSetCoordinates - sets the coordinates of all the nodes on the local process

   Collective on PC

   Input Parameters:
+  pc - the solver context
.  dim - the dimension of the coordinates 1, 2, or 3
-  coords - the coordinates

   Level: intermediate

   Notes: coords is an array of the 3D coordinates for the nodes on
   the local processor.  So if there are 108 equation on a processor
   for a displacement finite element discretization of elasticity (so
   that there are 36 = 108/3 nodes) then the array must have 108
   double precision values (ie, 3 * 36).  These x y z coordinates
   should be ordered for nodes 0 to N-1 like so: [ 0.x, 0.y, 0.z, 1.x,
   ... , N-1.z ].

.seealso: PCPROMETHEUS
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCSetCoordinates(PC pc,PetscInt dim,PetscReal *coords)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscTryMethod(pc,"PCSetCoordinates_C",(PC,PetscInt,PetscReal*),(pc,dim,coords));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCSASetVectors"
/*@
   PCSASetVectors - sets the vectors of all the nodes on the local process

   Collective on PC

   Input Parameters:
+  pc - the solver context
.  nects - the number of vectors
-  vects - the vectors

   Level: intermediate

   Notes: 'vects' is a dense tall skinny matrix with 'nvects' columns and 
   the number of local equations rows.  'vects' is stored in row major order.

.seealso: PCPROMETHEUS
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCSASetVectors(PC pc,PetscInt nvects,PetscReal *vects)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscTryMethod(pc,"PCSASetVectors_C",(PC,PetscInt,PetscReal*),(pc,nvects,vects));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
