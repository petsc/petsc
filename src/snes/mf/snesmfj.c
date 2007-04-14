#define PETSCSNES_DLL

#include "include/private/snesimpl.h"  /*I  "petscsnes.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "MatMFFDComputeJacobian"
/*@
   MatMFFDComputeJacobian - Tells the matrix-free Jacobian object the new location at which
       Jacobian matrix vector products will be computed at, i.e. J(x) * a.

   Collective on SNES

   Input Parameters:
+   snes - the nonlinear solver context
.   x - the point at which the Jacobian vector products will be performed
.   jac - the matrix-free Jacobian object
.   B - either the same as jac or another matrix type (ignored)
.   flag - not relevent for matrix-free form
-   dummy - the user context (ignored)

   Level: developer

   Notes:
     This can be passed into SNESSetJacobian() when using a completely matrix-free solver,
     that is the B matrix is also the same matrix operator. This is used when you select
     -mat_mffd but rarely used directly by users.

.seealso: MatMFFDGetH(), MatCreateSNESMF(), MatCreateMFFD(), MATMFFD,
          MatMFFDSetHHistory(),
          MatMFFDKSPMonitor(), MatMFFDSetFunctionError(), MatMFFDCreate(), SNESSetJacobian()

@*/
PetscErrorCode PETSCSNES_DLLEXPORT MatMFFDComputeJacobian(SNES snes,Vec x,Mat *jac,Mat *B,MatStructure *flag,void *dummy)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = MatAssemblyBegin(*jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatCreateSNESMF"
/*@
   MatCreateSNESMF - Creates a matrix-free matrix context for use with
   a SNES solver.  This matrix can be used as the Jacobian argument for
   the routine SNESSetJacobian().

   Collective on SNES and Vec

   Input Parameters:
+  snes - the SNES context
-  x - vector where SNES solution is to be stored.

   Output Parameter:
.  J - the matrix-free matrix

   Level: advanced

   Notes:
   The matrix-free matrix context merely contains the function pointers
   and work space for performing finite difference approximations of
   Jacobian-vector products, F'(u)*a, 

   The default code uses the following approach to compute h

.vb
     F'(u)*a = [F(u+h*a) - F(u)]/h where
     h = error_rel*u'a/||a||^2                        if  |u'a| > umin*||a||_{1}
       = error_rel*umin*sign(u'a)*||a||_{1}/||a||^2   otherwise
 where
     error_rel = square root of relative error in function evaluation
     umin = minimum iterate parameter
.ve
   (see MATSNESMF_WP or MATSNESMF_DS)
   
   The user can set the error_rel via MatSNESMFSetFunctionError() and 
   umin via MatSNESMFDefaultSetUmin(); see the nonlinear solvers chapter
   of the users manual for details.

   The user should call MatDestroy() when finished with the matrix-free
   matrix context.

   Options Database Keys:
+  -mat_mffd_err <error_rel> - Sets error_rel
+  -mat_mffd_type - wp or ds (see MATSNESMF_WP or MATSNESMF_DS)
.  -mat_mffd_unim <umin> - Sets umin (for default PETSc routine that computes h only)
-  -mat_mffd_ksp_monitor - KSP monitor routine that prints differencing h

.keywords: SNES, default, matrix-free, create, matrix

.seealso: MatDestroy(), MatSNESMFSetFunctionError(), MatSNESMFDefaultSetUmin()
          MatSNESMFSetHHistory(), MatSNESMFResetHHistory(), MatCreateMF(),
          MatSNESMFGetH(),MatSNESMFKSPMonitor(), MatSNESMFRegisterDynamic), MatSNESMFComputeJacobian()
 
@*/
PetscErrorCode PETSCSNES_DLLEXPORT MatCreateSNESMF(SNES snes,Vec x,Mat *J)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreateMFFD(x,J);CHKERRQ(ierr);
  ierr = MatMFFDSetFunction(*J,snes->vec_sol,(PetscErrorCode (*)(void*, _p_Vec*, _p_Vec*))SNESComputeFunction,snes);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}






