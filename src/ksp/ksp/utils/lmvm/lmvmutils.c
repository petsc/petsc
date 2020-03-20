#include <../src/ksp/ksp/utils/lmvm/lmvm.h> /*I "petscksp.h" I*/

/*@
   MatLMVMUpdate - Adds (X-Xprev) and (F-Fprev) updates to an LMVM matrix.
   The first time the function is called for an LMVM matrix, no update is 
   applied, but the given X and F vectors are stored for use as Xprev and
   Fprev in the next update.
   
   If the user has provided another LMVM matrix in place of J0, the J0 
   matrix is also updated recursively.

   Input Parameters:
+  B - An LMVM-type matrix
.  X - Solution vector
-  F - Function vector

   Level: intermediate

.seealso: MatLMVMReset(), MatLMVMAllocate()
@*/
PetscErrorCode MatLMVMUpdate(Mat B, Vec X, Vec F)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  PetscErrorCode    ierr;
  PetscBool         same;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, MAT_CLASSID, 1);
  PetscValidHeaderSpecific(X, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(F, VEC_CLASSID, 3);
  ierr = PetscObjectBaseTypeCompare((PetscObject)B, MATLMVM, &same);CHKERRQ(ierr);
  if (!same) SETERRQ(PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_WRONG, "Matrix must be an LMVM-type.");
  if (!lmvm->allocated) {
    ierr = MatLMVMAllocate(B, X, F);CHKERRQ(ierr);
  } else {
    VecCheckMatCompatible(B, X, 2, F, 3);
  }
  if (lmvm->J0) {
    /* If the user provided an LMVM-type matrix as J0, then trigger its update as well */
    ierr = PetscObjectBaseTypeCompare((PetscObject)lmvm->J0, MATLMVM, &same);CHKERRQ(ierr);
    if (same) {
      ierr = MatLMVMUpdate(lmvm->J0, X, F);CHKERRQ(ierr);
    }
  }
  ierr = (*lmvm->ops->update)(B, X, F);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/*@
   MatLMVMClearJ0 - Removes all definitions of J0 and reverts to 
   an identity matrix (scale = 1.0).

   Input Parameters:
.  B - An LMVM-type matrix

   Level: advanced

.seealso: MatLMVMSetJ0()
@*/
PetscErrorCode MatLMVMClearJ0(Mat B)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  PetscErrorCode    ierr;
  PetscBool         same;
  MPI_Comm          comm = PetscObjectComm((PetscObject)B);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, MAT_CLASSID, 1);
  ierr = PetscObjectBaseTypeCompare((PetscObject)B, MATLMVM, &same);CHKERRQ(ierr);
  if (!same) SETERRQ(comm, PETSC_ERR_ARG_WRONG, "Matrix must be an LMVM-type.");
  lmvm->user_pc = PETSC_FALSE;
  lmvm->user_ksp = PETSC_FALSE;
  lmvm->user_scale = PETSC_FALSE;
  lmvm->J0scalar = 1.0;
  ierr = VecDestroy(&lmvm->J0diag);CHKERRQ(ierr);
  ierr = MatDestroy(&lmvm->J0);CHKERRQ(ierr);
  ierr = PCDestroy(&lmvm->J0pc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/*@
   MatLMVMSetJ0Scale - Allows the user to define a scalar value
   mu such that J0 = mu*I.

   Input Parameters:
+  B - An LMVM-type matrix
-  scale - Scalar value mu that defines the initial Jacobian

   Level: advanced

.seealso: MatLMVMSetDiagScale(), MatLMVMSetJ0()
@*/
PetscErrorCode MatLMVMSetJ0Scale(Mat B, PetscReal scale)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  PetscErrorCode    ierr;
  PetscBool         same;
  MPI_Comm          comm = PetscObjectComm((PetscObject)B);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, MAT_CLASSID, 1);
  ierr = PetscObjectBaseTypeCompare((PetscObject)B, MATLMVM, &same);CHKERRQ(ierr);
  if (!same) SETERRQ(comm, PETSC_ERR_ARG_WRONG, "Matrix must be an LMVM-type.");
  if (!lmvm->square) SETERRQ(comm, PETSC_ERR_SUP, "Scaling is available only for square LMVM matrices");
  ierr = MatLMVMClearJ0(B);CHKERRQ(ierr);
  lmvm->J0scalar = scale;
  lmvm->user_scale = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/*@
   MatLMVMSetJ0Diag - Allows the user to define a vector 
   V such that J0 = diag(V).

   Input Parameters:
+  B - An LMVM-type matrix
-  V - Vector that defines the diagonal of the initial Jacobian

   Level: advanced

.seealso: MatLMVMSetScale(), MatLMVMSetJ0()
@*/
PetscErrorCode MatLMVMSetJ0Diag(Mat B, Vec V)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  PetscErrorCode    ierr;
  PetscBool         same;
  MPI_Comm          comm = PetscObjectComm((PetscObject)B);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, MAT_CLASSID, 1);
  PetscValidHeaderSpecific(V, VEC_CLASSID, 2);
  ierr = PetscObjectBaseTypeCompare((PetscObject)B, MATLMVM, &same);CHKERRQ(ierr);
  if (!same) SETERRQ(comm, PETSC_ERR_ARG_WRONG, "Matrix must be an LMVM-type.");
  if (!lmvm->allocated) SETERRQ(comm, PETSC_ERR_ORDER, "Matrix must be allocated before setting diagonal scaling");
  if (!lmvm->square) SETERRQ(comm, PETSC_ERR_SUP, "Diagonal scaling is available only for square LMVM matrices");
  VecCheckSameSize(V, 2, lmvm->Fprev, 3);CHKERRQ(ierr);
  ierr = MatLMVMClearJ0(B);CHKERRQ(ierr);
  if (!lmvm->J0diag) {
    ierr = VecDuplicate(V, &lmvm->J0diag);CHKERRQ(ierr);
  }
  ierr = VecCopy(V, lmvm->J0diag);CHKERRQ(ierr);
  lmvm->user_scale = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/*@
   MatLMVMSetJ0 - Allows the user to define the initial 
   Jacobian matrix from which the LMVM approximation is 
   built up. Inverse of this initial Jacobian is applied 
   using an internal KSP solver, which defaults to GMRES.
   This internal KSP solver has the "mat_lmvm_" option 
   prefix.
   
   Note that another LMVM matrix can be used in place of 
   J0, in which case updating the outer LMVM matrix will 
   also trigger the update for the inner LMVM matrix. This 
   is useful in cases where a full-memory diagonal approximation 
   such as MATLMVMDIAGBRDN is used in place of J0.

   Input Parameters:
+  B - An LMVM-type matrix
-  J0 - The initial Jacobian matrix

   Level: advanced

.seealso: MatLMVMSetJ0PC(), MatLMVMSetJ0KSP()
@*/
PetscErrorCode MatLMVMSetJ0(Mat B, Mat J0)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  PetscErrorCode    ierr;
  PetscBool         same;
  MPI_Comm          comm = PetscObjectComm((PetscObject)B);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, MAT_CLASSID, 1);
  PetscValidHeaderSpecific(J0, MAT_CLASSID, 2);
  ierr = PetscObjectBaseTypeCompare((PetscObject)B, MATLMVM, &same);CHKERRQ(ierr);
  if (!same) SETERRQ(comm, PETSC_ERR_ARG_WRONG, "Matrix must be an LMVM-type.");
  ierr = MatLMVMClearJ0(B);CHKERRQ(ierr);
  ierr = MatDestroy(&lmvm->J0);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)J0);CHKERRQ(ierr);
  lmvm->J0 = J0;
  ierr = PetscObjectBaseTypeCompare((PetscObject)lmvm->J0, MATLMVM, &same);CHKERRQ(ierr);
  if (!same && lmvm->square) {
    ierr = KSPSetOperators(lmvm->J0ksp, lmvm->J0, lmvm->J0);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/*@
   MatLMVMSetJ0PC - Allows the user to define a PC object that 
   acts as the initial inverse-Jacobian matrix. This PC should 
   already contain all the operators necessary for its application. 
   The LMVM matrix only calls PCApply() without changing any other 
   options.

   Input Parameters:
+  B - An LMVM-type matrix
-  J0pc - PC object where PCApply defines an inverse application for J0

   Level: advanced

.seealso: MatLMVMGetJ0PC()
@*/
PetscErrorCode MatLMVMSetJ0PC(Mat B, PC J0pc)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  PetscErrorCode    ierr;
  PetscBool         same;
  MPI_Comm          comm = PetscObjectComm((PetscObject)B);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, MAT_CLASSID, 1);
  PetscValidHeaderSpecific(J0pc, PC_CLASSID, 2);
  ierr = PetscObjectBaseTypeCompare((PetscObject)B, MATLMVM, &same);CHKERRQ(ierr);
  if (!same) SETERRQ(comm, PETSC_ERR_ARG_WRONG, "Matrix must be an LMVM-type.");
  if (!lmvm->square) SETERRQ(comm, PETSC_ERR_SUP, "Inverse J0 can be defined only for square LMVM matrices");
  ierr = MatLMVMClearJ0(B);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)J0pc);CHKERRQ(ierr);
  lmvm->J0pc = J0pc;
  lmvm->user_pc = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/*@
   MatLMVMSetJ0KSP - Allows the user to provide a pre-configured 
   KSP solver for the initial inverse-Jacobian approximation. 
   This KSP solver should already contain all the operators 
   necessary to perform the inversion. The LMVM matrix only 
   calls KSPSolve() without changing any other options.

   Input Parameters:
+  B - An LMVM-type matrix
-  J0ksp - KSP solver for the initial inverse-Jacobian application

   Level: advanced

.seealso: MatLMVMGetJ0KSP()
@*/
PetscErrorCode MatLMVMSetJ0KSP(Mat B, KSP J0ksp)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  PetscErrorCode    ierr;
  PetscBool         same;
  MPI_Comm          comm = PetscObjectComm((PetscObject)B);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, MAT_CLASSID, 1);
  PetscValidHeaderSpecific(J0ksp, KSP_CLASSID, 2);
  ierr = PetscObjectBaseTypeCompare((PetscObject)B, MATLMVM, &same);CHKERRQ(ierr);
  if (!same) SETERRQ(comm, PETSC_ERR_ARG_WRONG, "Matrix must be an LMVM-type.");
  if (!lmvm->square) SETERRQ(comm, PETSC_ERR_SUP, "Inverse J0 can be defined only for square LMVM matrices");
  ierr = MatLMVMClearJ0(B);CHKERRQ(ierr);
  ierr = KSPDestroy(&lmvm->J0ksp);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)J0ksp);CHKERRQ(ierr);
  lmvm->J0ksp = J0ksp;
  lmvm->user_ksp = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/*@
   MatLMVMGetJ0 - Returns a pointer to the internal J0 matrix.

   Input Parameters:
.  B - An LMVM-type matrix

   Output Parameter:
.  J0 - Mat object for defining the initial Jacobian

   Level: advanced

.seealso: MatLMVMSetJ0()
@*/
PetscErrorCode MatLMVMGetJ0(Mat B, Mat *J0)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  PetscErrorCode    ierr;
  PetscBool         same;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, MAT_CLASSID, 1);
  ierr = PetscObjectBaseTypeCompare((PetscObject)B, MATLMVM, &same);CHKERRQ(ierr);
  if (!same) SETERRQ(PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_WRONG, "Matrix must be an LMVM-type.");
  *J0 = lmvm->J0;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/*@
   MatLMVMGetJ0PC - Returns a pointer to the internal PC object 
   associated with the initial Jacobian.

   Input Parameters:
.  B - An LMVM-type matrix

   Output Parameter:
.  J0pc - PC object for defining the initial inverse-Jacobian

   Level: advanced

.seealso: MatLMVMSetJ0PC()
@*/
PetscErrorCode MatLMVMGetJ0PC(Mat B, PC *J0pc)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  PetscErrorCode    ierr;
  PetscBool         same;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, MAT_CLASSID, 1);
  ierr = PetscObjectBaseTypeCompare((PetscObject)B, MATLMVM, &same);CHKERRQ(ierr);
  if (!same) SETERRQ(PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_WRONG, "Matrix must be an LMVM-type.");
  if (lmvm->J0pc) {
    *J0pc = lmvm->J0pc;
  } else {
    ierr = KSPGetPC(lmvm->J0ksp, J0pc);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/*@
   MatLMVMGetJ0KSP - Returns a pointer to the internal KSP solver 
   associated with the initial Jacobian.

   Input Parameters:
.  B - An LMVM-type matrix

   Output Parameter:
.  J0ksp - KSP solver for defining the initial inverse-Jacobian

   Level: advanced

.seealso: MatLMVMSetJ0KSP()
@*/
PetscErrorCode MatLMVMGetJ0KSP(Mat B, KSP *J0ksp)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  PetscErrorCode    ierr;
  PetscBool         same;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, MAT_CLASSID, 1);
  ierr = PetscObjectBaseTypeCompare((PetscObject)B, MATLMVM, &same);CHKERRQ(ierr);
  if (!same) SETERRQ(PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_WRONG, "Matrix must be an LMVM-type.");
  *J0ksp = lmvm->J0ksp;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/*@
   MatLMVMApplyJ0Fwd - Applies an approximation of the forward 
   matrix-vector product with the initial Jacobian.

   Input Parameters:
+  B - An LMVM-type matrix
-  X - vector to multiply with J0

   Output Parameter:
.  Y - resulting vector for the operation

   Level: advanced

.seealso: MatLMVMSetJ0(), MatLMVMSetJ0Scale(), MatLMVMSetJ0ScaleDiag(), 
          MatLMVMSetJ0PC(), MatLMVMSetJ0KSP(), MatLMVMApplyJ0Inv()
@*/
PetscErrorCode MatLMVMApplyJ0Fwd(Mat B, Vec X, Vec Y)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  PetscErrorCode    ierr;
  PetscBool         same, hasMult;
  MPI_Comm          comm = PetscObjectComm((PetscObject)B);
  Mat               Amat, Pmat;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, MAT_CLASSID, 1);
  PetscValidHeaderSpecific(X, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(Y, VEC_CLASSID, 3);
  ierr = PetscObjectBaseTypeCompare((PetscObject)B, MATLMVM, &same);CHKERRQ(ierr);
  if (!same) SETERRQ(PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_WRONG, "Matrix must be an LMVM-type.");
  if (!lmvm->allocated) SETERRQ(comm, PETSC_ERR_ORDER, "LMVM matrix must be allocated first");
  VecCheckMatCompatible(B, X, 2, Y, 3);
  if (lmvm->user_pc || lmvm->user_ksp || lmvm->J0) {
    /* User may have defined a PC or KSP for J0^{-1} so let's try to use its operators. */
    if (lmvm->user_pc){
      ierr = PCGetOperators(lmvm->J0pc, &Amat, &Pmat);CHKERRQ(ierr);
    } else if (lmvm->user_ksp) {
      ierr = KSPGetOperators(lmvm->J0ksp, &Amat, &Pmat);CHKERRQ(ierr);
    } else {
      Amat = lmvm->J0;
    }
    ierr = MatHasOperation(Amat, MATOP_MULT, &hasMult);CHKERRQ(ierr);
    if (hasMult) {
      /* product is available, use it */
      ierr = MatMult(Amat, X, Y);CHKERRQ(ierr);
    } else {
      /* there's no product, so treat J0 as identity */
      ierr = VecCopy(X, Y);CHKERRQ(ierr);
    }
  } else if (lmvm->user_scale) {
    if (lmvm->J0diag) {
      /* User has defined a diagonal vector for J0 */
      ierr = VecPointwiseMult(X, lmvm->J0diag, Y);CHKERRQ(ierr);
    } else {
      /* User has defined a scalar value for J0 */
      ierr = VecCopy(X, Y);CHKERRQ(ierr);
      ierr = VecScale(Y, lmvm->J0scalar);CHKERRQ(ierr);
    }
  } else {
    /* There is no J0 representation so just apply an identity matrix */
    ierr = VecCopy(X, Y);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/*@
   MatLMVMApplyJ0Inv - Applies some estimation of the initial Jacobian 
   inverse to the given vector. The specific form of the application 
   depends on whether the user provided a scaling factor, a J0 matrix, 
   a J0 PC, or a J0 KSP object. If no form of the initial Jacobian is 
   provided, the function simply does an identity matrix application 
   (vector copy).

   Input Parameters:
+  B - An LMVM-type matrix
-  X - vector to "multiply" with J0^{-1}

   Output Parameter:
.  Y - resulting vector for the operation

   Level: advanced

.seealso: MatLMVMSetJ0(), MatLMVMSetJ0Scale(), MatLMVMSetJ0ScaleDiag(), 
          MatLMVMSetJ0PC(), MatLMVMSetJ0KSP(), MatLMVMApplyJ0Fwd()
@*/
PetscErrorCode MatLMVMApplyJ0Inv(Mat B, Vec X, Vec Y)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  PetscErrorCode    ierr;
  PetscBool         same, hasSolve;
  MPI_Comm          comm = PetscObjectComm((PetscObject)B);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, MAT_CLASSID, 1);
  PetscValidHeaderSpecific(X, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(Y, VEC_CLASSID, 3);
  ierr = PetscObjectBaseTypeCompare((PetscObject)B, MATLMVM, &same);CHKERRQ(ierr);
  if (!same) SETERRQ(PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_WRONG, "Matrix must be an LMVM-type.");
  if (!lmvm->allocated) SETERRQ(comm, PETSC_ERR_ORDER, "LMVM matrix must be allocated first");
  VecCheckMatCompatible(B, X, 2, Y, 3);
  /* Invert the initial Jacobian onto q (or apply scaling) */
  if (lmvm->user_pc) {
    /* User has defined a J0 inverse so we can directly apply it as a preconditioner */
    ierr = PCApply(lmvm->J0pc, X, Y);CHKERRQ(ierr);
  } else if (lmvm->user_ksp) {
    /* User has defined a J0 or a custom KSP so just perform a solution */
    ierr = KSPSolve(lmvm->J0ksp, X, Y);CHKERRQ(ierr);
  } else if (lmvm->J0) {
    ierr = MatHasOperation(lmvm->J0, MATOP_SOLVE, &hasSolve);CHKERRQ(ierr);
    if (hasSolve) {
      ierr = MatSolve(lmvm->J0, X, Y);CHKERRQ(ierr);
    } else {
      ierr = KSPSolve(lmvm->J0ksp, X, Y);CHKERRQ(ierr);
    }
  } else if (lmvm->user_scale) {
    if (lmvm->J0diag) {
      ierr = VecPointwiseDivide(X, Y, lmvm->J0diag);CHKERRQ(ierr);
    } else {
      ierr = VecCopy(X, Y);CHKERRQ(ierr);
      ierr = VecScale(Y, 1.0/lmvm->J0scalar);CHKERRQ(ierr);
    }
  } else {
    /* There is no J0 representation so just apply an identity matrix */
    ierr = VecCopy(X, Y);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/*@
   MatLMVMIsAllocated - Returns a boolean flag that shows whether 
   the necessary data structures for the underlying matrix is allocated.

   Input Parameters:
.  B - An LMVM-type matrix

   Output Parameter:
.  flg - PETSC_TRUE if allocated, PETSC_FALSE otherwise

   Level: intermediate

.seealso: MatLMVMAllocate(), MatLMVMReset()
@*/

PetscErrorCode MatLMVMIsAllocated(Mat B, PetscBool *flg)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  PetscErrorCode    ierr;
  PetscBool         same;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, MAT_CLASSID, 1);
  ierr = PetscObjectBaseTypeCompare((PetscObject)B, MATLMVM, &same);CHKERRQ(ierr);
  if (!same) SETERRQ(PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_WRONG, "Matrix must be an LMVM-type.");
  *flg = PETSC_FALSE;
  if (lmvm->allocated && B->preallocated && B->assembled) *flg = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/*@
   MatLMVMAllocate - Produces all necessary common memory for 
   LMVM approximations based on the solution and function vectors
   provided. If MatSetSizes() and MatSetUp() have not been called 
   before MatLMVMAllocate(), the allocation will read sizes from 
   the provided vectors and update the matrix.

   Input Parameters:
+  B - An LMVM-type matrix
.  X - Solution vector
-  F - Function vector

   Level: intermediate

.seealso: MatLMVMReset(), MatLMVMUpdate()
@*/
PetscErrorCode MatLMVMAllocate(Mat B, Vec X, Vec F)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  PetscErrorCode    ierr;
  PetscBool         same;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, MAT_CLASSID, 1);
  PetscValidHeaderSpecific(X, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(F, VEC_CLASSID, 3);
  ierr = PetscObjectBaseTypeCompare((PetscObject)B, MATLMVM, &same);CHKERRQ(ierr);
  if (!same) SETERRQ(PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_WRONG, "Matrix must be an LMVM-type.");
  ierr = (*lmvm->ops->allocate)(B, X, F);CHKERRQ(ierr);
  if (lmvm->J0) {
    ierr = PetscObjectBaseTypeCompare((PetscObject)lmvm->J0, MATLMVM, &same);CHKERRQ(ierr);
    if (same) {
      ierr = MatLMVMAllocate(lmvm->J0, X, F);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/*@
   MatLMVMResetShift - Zero the shift factor.

   Input Parameters:
.  B - An LMVM-type matrix

   Level: intermediate

.seealso: MatLMVMAllocate(), MatLMVMUpdate()
@*/
PetscErrorCode MatLMVMResetShift(Mat B)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  PetscErrorCode    ierr;
  PetscBool         same;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, MAT_CLASSID, 1);
  ierr = PetscObjectBaseTypeCompare((PetscObject)B, MATLMVM, &same);CHKERRQ(ierr);
  if (!same) SETERRQ(PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_WRONG, "Matrix must be an LMVM-type.");
  lmvm->shift = 0.0;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/*@
   MatLMVMReset - Flushes all of the accumulated updates out of 
   the LMVM approximation. In practice, this will not actually 
   destroy the data associated with the updates. It simply resets 
   counters, which leads to existing data being overwritten, and 
   MatSolve() being applied as if there are no updates. A boolean 
   flag is available to force destruction of the update vectors.
   
   If the user has provided another LMVM matrix as J0, the J0 
   matrix is also reset in this function.

   Input Parameters:
+  B - An LMVM-type matrix
-  destructive - flag for enabling destruction of data structures

   Level: intermediate

.seealso: MatLMVMAllocate(), MatLMVMUpdate()
@*/
PetscErrorCode MatLMVMReset(Mat B, PetscBool destructive)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  PetscErrorCode    ierr;
  PetscBool         same;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, MAT_CLASSID, 1);
  ierr = PetscObjectBaseTypeCompare((PetscObject)B, MATLMVM, &same);CHKERRQ(ierr);
  if (!same) SETERRQ(PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_WRONG, "Matrix must be an LMVM-type.");
  ierr = (*lmvm->ops->reset)(B, destructive);CHKERRQ(ierr);
  if (lmvm->J0) {
    ierr = PetscObjectBaseTypeCompare((PetscObject)lmvm->J0, MATLMVM, &same);CHKERRQ(ierr);
    if (same) {
      ierr = MatLMVMReset(lmvm->J0, destructive);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/*@
   MatLMVMSetHistorySize - Set the number of past iterates to be 
   stored for the construction of the limited-memory QN update.

   Input Parameters:
+  B - An LMVM-type matrix
-  hist_size - number of past iterates (default 5)

   Level: beginner

.seealso: MatLMVMGetUpdateCount()
@*/

PetscErrorCode MatLMVMSetHistorySize(Mat B, PetscInt hist_size)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  PetscErrorCode    ierr;
  PetscBool         same;
  Vec               X, F;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, MAT_CLASSID, 1);
  ierr = PetscObjectBaseTypeCompare((PetscObject)B, MATLMVM, &same);CHKERRQ(ierr);
  if (!same) SETERRQ(PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_WRONG, "Matrix must be an LMVM-type.");
  if (hist_size > 0) {
    lmvm->m = hist_size;
    if (lmvm->allocated && lmvm->m != lmvm->m_old) {
      ierr = VecDuplicate(lmvm->Xprev, &X);CHKERRQ(ierr);
      ierr = VecDuplicate(lmvm->Fprev, &F);CHKERRQ(ierr);
      ierr = MatLMVMReset(B, PETSC_TRUE);CHKERRQ(ierr);
      ierr = MatLMVMAllocate(B, X, F);CHKERRQ(ierr);
      ierr = VecDestroy(&X);CHKERRQ(ierr);
      ierr = VecDestroy(&F);CHKERRQ(ierr);
    }
  } else {
    SETERRQ(PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_WRONG, "QN history size must be a positive integer.");
  }
  
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/*@
   MatLMVMGetUpdateCount - Returns the number of accepted updates.
   This number may be greater than the total number of update vectors 
   stored in the matrix. The counters are reset when MatLMVMReset() 
   is called.

   Input Parameters:
.  B - An LMVM-type matrix

   Output Parameter:
.  nupdates - number of accepted updates

   Level: intermediate

.seealso: MatLMVMGetRejectCount(), MatLMVMReset()
@*/
PetscErrorCode MatLMVMGetUpdateCount(Mat B, PetscInt *nupdates)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  PetscErrorCode    ierr;
  PetscBool         same;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, MAT_CLASSID, 1);
  ierr = PetscObjectBaseTypeCompare((PetscObject)B, MATLMVM, &same);CHKERRQ(ierr);
  if (!same) SETERRQ(PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_WRONG, "Matrix must be an LMVM-type.");
  *nupdates = lmvm->nupdates;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/*@
   MatLMVMGetRejectCount - Returns the number of rejected updates. 
   The counters are reset when MatLMVMReset() is called.

   Input Parameters:
.  B - An LMVM-type matrix

   Output Parameter:
.  nrejects - number of rejected updates

   Level: intermediate

.seealso: MatLMVMGetRejectCount(), MatLMVMReset()
@*/
PetscErrorCode MatLMVMGetRejectCount(Mat B, PetscInt *nrejects)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  PetscErrorCode    ierr;
  PetscBool         same;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, MAT_CLASSID, 1);
  ierr = PetscObjectBaseTypeCompare((PetscObject)B, MATLMVM, &same);CHKERRQ(ierr);
  if (!same) SETERRQ(PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_WRONG, "Matrix must be an LMVM-type.");
  *nrejects = lmvm->nrejects;
  PetscFunctionReturn(0);
}
