#include <../src/ksp/ksp/utils/lmvm/lmvm.h> /*I "petscksp.h" I*/

/*@
   MatLMVMUpdate - Adds (X-Xprev) and (F-Fprev) updates to an LMVM matrix.
   The first time the function is called for an LMVM matrix, no update is 
   applied, but the given X and F vectors are stored for use as Xprev and
   Fprev in the next update.

   Collective on MPI_Comm

   Input Parameters:
+  B - An LMVM-type matrix (LDFP LBFGS, LSR1, LBRDN, LMBRDN, LSBRDN)
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
  PetscInt          i;
  PetscReal         rhotol, rho, ynorm2;
  Vec               Stmp, Ytmp;
  
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
  if (lmvm->m == 0) PetscFunctionReturn(0);
  if (lmvm->prev_set) {
    /* Compute the new (S = X - Xprev) and (Y = F - Fprev) vectors */
    ierr = VecAXPBY(lmvm->Xprev, 1.0, -1.0, X);CHKERRQ(ierr);
    ierr = VecAXPBY(lmvm->Fprev, 1.0, -1.0, F);CHKERRQ(ierr);
    /* Test if the updates can be accepted */
    ierr = VecDotBegin(lmvm->Fprev, lmvm->Xprev, &rho);CHKERRQ(ierr);
    ierr = VecDotBegin(lmvm->Fprev, lmvm->Fprev, &ynorm2);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->Fprev, lmvm->Xprev, &rho);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->Fprev, lmvm->Fprev, &ynorm2);CHKERRQ(ierr);
    rhotol = lmvm->eps * ynorm2;
    if (rho > rhotol) {
      /* Update is good, accept it */
      ++lmvm->nupdates;
      lmvm->k = PetscMin(lmvm->k+1, lmvm->m-1);
      if (lmvm->k == lmvm->m-1) {
        /* We hit the memory limit, so shift all the vectors back one spot 
           and shift the oldest to the front to receive the latest update. */
        Stmp = lmvm->S[0];
        Ytmp = lmvm->Y[0];
        for (i = 0; i < lmvm->k; ++i) {
          lmvm->S[i] = lmvm->S[i+1];
          lmvm->Y[i] = lmvm->Y[i+1];
        }
        lmvm->S[lmvm->k] = Stmp;
        lmvm->Y[lmvm->k] = Ytmp;
      }
      /* Put the precomputed update into the last vector */
      ierr = VecCopy(lmvm->Xprev, lmvm->S[lmvm->k]);CHKERRQ(ierr);
      ierr = VecCopy(lmvm->Fprev, lmvm->Y[lmvm->k]);CHKERRQ(ierr);
    } else {
      /* Update is bad, skip it */
      ++lmvm->nrejects;
    }
  }
  
  /* Save the solution and function to be used in the next update */
  ierr = VecCopy(X, lmvm->Xprev);CHKERRQ(ierr);
  ierr = VecCopy(F, lmvm->Fprev);CHKERRQ(ierr);
  lmvm->prev_set = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/*@
   MatLMVMResetJ0 - Removes all definitions of J0 and reverts to 
   an identity matrix (scale = 1.0).

   Input Parameters:
.  B - An LMVM matrix type (LDFP LBFGS, LSR1, LBRDN, LMBRDN, LSBRDN)

   Level: advanced

.seealso: MatLMVMSetJ0()
@*/
PetscErrorCode MatLMVMResetJ0(Mat B)
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
  lmvm->scale = 1.0;
  if (lmvm->diag_scale) {
    ierr = VecDestroy(&lmvm->diag_scale);CHKERRQ(ierr);
  }
  if (lmvm->J0) {
    ierr = MatDestroy(&lmvm->J0);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/*@
   MatLMVMSetJ0Scale - Allows the user to define a scalar value
   mu such that J0^{-1} = mu*I.

   Input Parameters:
+  B - An LMVM matrix type (LDFP LBFGS, LSR1, LBRDN, LMBRDN, LSBRDN)
-  scale - Scalar value mu that defines the initial inverse Jacobian

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
  ierr = MatLMVMResetJ0(B);CHKERRQ(ierr);
  lmvm->scale = scale;
  lmvm->user_scale = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/*@
   MatLMVMSetJ0Diag - Allows the user to define a vector 
   V such that J0^{-1} = diag(V).

   Input Parameters:
+  B - An LMVM matrix type (LDFP LBFGS, LSR1, LBRDN, LMBRDN, LSBRDN)
-  V - Vector that defines the diagonal scaling

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
  ierr = MatLMVMResetJ0(B);CHKERRQ(ierr);
  if (!lmvm->diag_scale) {
    ierr = VecDuplicate(V, &lmvm->diag_scale);CHKERRQ(ierr);
  }
  ierr = VecCopy(V, lmvm->diag_scale);CHKERRQ(ierr);
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

   Input Parameters:
+  B - An LMVM matrix type (LDFP LBFGS, LSR1, LBRDN, LMBRDN, LSBRDN)
.  J0 - The initial Jacobian matrix
-  J0pre - Preconditioner matrix to the initial Jacobian (can be same J0)

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
  if (!J0->assembled) SETERRQ(comm, PETSC_ERR_ARG_WRONGSTATE, "J0 is not assembled.");
  if (B->symmetric && (!J0->symmetric)) SETERRQ(comm, PETSC_ERR_ARG_INCOMP, "J0 and J0pre must be symmetric when B is symmetric");
  if (lmvm->allocated) {
    MatCheckSameSize(B, 1, J0, 2);
  }
  ierr = MatLMVMResetJ0(B);CHKERRQ(ierr);
  if (lmvm->J0) {
    ierr = MatDestroy(&lmvm->J0);CHKERRQ(ierr);
  }
  ierr = PetscObjectReference((PetscObject)J0);CHKERRQ(ierr);
  lmvm->J0 = J0;
  if (lmvm->square) {
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
+  B - An LMVM matrix type (LDFP LBFGS, LSR1, LBRDN, LMBRDN, LSBRDN)
-  J0pc - The initial inverse-Jacobian matrix

   Level: advanced

.seealso: MatLMVMGetJ0PC()
@*/
PetscErrorCode MatLMVMSetJ0PC(Mat B, PC J0pc)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  PetscErrorCode    ierr;
  PetscBool         same;
  MPI_Comm          comm = PetscObjectComm((PetscObject)B);
  Mat               J0, J0pre;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, MAT_CLASSID, 1);
  PetscValidHeaderSpecific(J0pc, PC_CLASSID, 2);
  ierr = PetscObjectBaseTypeCompare((PetscObject)B, MATLMVM, &same);CHKERRQ(ierr);
  if (!same) SETERRQ(comm, PETSC_ERR_ARG_WRONG, "Matrix must be an LMVM-type.");
  if (!lmvm->square) SETERRQ(comm, PETSC_ERR_SUP, "Inverse J0 can be defined only for square LMVM matrices");
  ierr = PCGetOperators(J0pc, &J0, &J0pre);CHKERRQ(ierr);
  if (B->symmetric && (!J0->symmetric || !J0pre->symmetric)) SETERRQ(comm, PETSC_ERR_ARG_INCOMP, "J0 and J0pre must be symmetric when B is symmetric");
  if (lmvm->allocated) {
    MatCheckSameSize(B, 1, J0, 2);
    MatCheckSameSize(B, 1, J0pre, 3);
  }
  ierr = MatDestroy(&J0);CHKERRQ(ierr);
  ierr = MatDestroy(&J0pre);CHKERRQ(ierr);
  ierr = MatLMVMResetJ0(B);CHKERRQ(ierr);
  if (lmvm->J0pc) {
    ierr = PCDestroy(&lmvm->J0pc);CHKERRQ(ierr);
  }
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
+  B - An LMVM matrix type (LDFP LBFGS, LSR1, LBRDN, LMBRDN, LSBRDN)
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
  Mat               J0, J0pre;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, MAT_CLASSID, 1);
  PetscValidHeaderSpecific(J0ksp, KSP_CLASSID, 2);
  ierr = PetscObjectBaseTypeCompare((PetscObject)B, MATLMVM, &same);CHKERRQ(ierr);
  if (!same) SETERRQ(comm, PETSC_ERR_ARG_WRONG, "Matrix must be an LMVM-type.");
  if (!lmvm->square) SETERRQ(comm, PETSC_ERR_SUP, "Inverse J0 can be defined only for square LMVM matrices");
  ierr = KSPGetOperators(J0ksp, &J0, &J0pre);CHKERRQ(ierr);
  if (B->symmetric && (!J0->symmetric || !J0pre->symmetric)) SETERRQ(comm, PETSC_ERR_ARG_INCOMP, "J0 and J0pre must be symmetric when B is symmetric");
  if (lmvm->allocated) {
    MatCheckSameSize(B, 1, J0, 2);
    MatCheckSameSize(B, 1, J0pre, 3);
  }
  ierr = MatLMVMResetJ0(B);CHKERRQ(ierr);
  if (lmvm->J0ksp) {
    ierr = KSPDestroy(&lmvm->J0ksp);CHKERRQ(ierr);
  }
  ierr = PetscObjectReference((PetscObject)J0ksp);CHKERRQ(ierr);
  lmvm->J0ksp = J0ksp;
  lmvm->user_ksp = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/*@
   MatLMVMGetJ0 - Returns a pointer to the internal J0 matrix.

   Input Parameters:
.  B - An LMVM matrix type (LDFP LBFGS, LSR1, LBRDN, LMBRDN, LSBRDN)

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
.  B - An LMVM matrix type (LDFP LBFGS, LSR1, LBRDN, LMBRDN, LSBRDN)

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
.  B - An LMVM matrix type (LDFP LBFGS, LSR1, LBRDN, LMBRDN, LSBRDN)

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
   MatLMVMApplyJ0Inv - Applies some estimation of the initial Jacobian 
   inverse to the given vector. The specific form of the application 
   depends on whether the user provided a scaling factor, a J0 matrix, 
   a J0 PC, or a J0 KSP object. If no form of the initial Jacobian is 
   provided, the function simply does an identity matrix application 
   (vector copy).

   Input Parameters:
+  B - An LMVM matrix type (LDFP LBFGS, LSR1, LBRDN, LMBRDN, LSBRDN)
-  X - vector to "multiply" with J0^{-1}

   Output Parameter:
.  Y - resulting vector for the operation

   Level: advanced

.seealso: MatLMVMSetJ0(), MatLMVMSetJ0Scale(), MatLMVMSetJ0ScaleDiag(), 
          MatLMVMSetJ0PC(), MatLMVMSetJ0KSP()
@*/
PetscErrorCode MatLMVMApplyJ0Inv(Mat B, Vec X, Vec Y)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  PetscErrorCode    ierr;
  PetscBool         same;
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
  } else if (lmvm->J0 || lmvm->user_ksp) {
    /* User has defined a J0 or a custom KSP so just perform a solution */
    ierr = KSPSolve(lmvm->J0ksp, X, Y);CHKERRQ(ierr);
  } else if (lmvm->user_scale) {
    if (lmvm->diag_scale) {
      ierr = VecPointwiseMult(X, lmvm->diag_scale, Y);CHKERRQ(ierr);
    } else {
      ierr = VecCopy(X, Y);CHKERRQ(ierr);
      ierr = VecScale(Y, lmvm->scale);CHKERRQ(ierr);
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
.  B - An LMVM matrix type (LDFP LBFGS, LSR1, LBRDN, LMBRDN, LSBRDN)

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
+  B - An LMVM matrix type (LDFP LBFGS, LSR1, LBRDN, LMBRDN, LSBRDN)
.  X - Solution vector
-  F - Function vector

   Level: intermediate

.seealso: MatLMVMReset(), MatLMVMUpdate()
@*/
PetscErrorCode MatLMVMAllocate(Mat B, Vec X, Vec F)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  PetscErrorCode    ierr;
  PetscBool         same, allocate = PETSC_FALSE;
  PetscInt          m, n, M, N;
  VecType           type;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, MAT_CLASSID, 1);
  PetscValidHeaderSpecific(X, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(F, VEC_CLASSID, 3);
  ierr = PetscObjectBaseTypeCompare((PetscObject)B, MATLMVM, &same);CHKERRQ(ierr);
  if (!same) SETERRQ(PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_WRONG, "Matrix must be an LMVM-type.");
  if (B->rmap->N == 0 && B->cmap->N == 0) {
    /* MatSetSizes() has not been called on this matrix, so we have to sort out the sizing */
    ierr = VecGetLocalSize(X, &n);CHKERRQ(ierr);
    ierr = VecGetSize(X, &N);CHKERRQ(ierr);
    ierr = VecGetLocalSize(F, &m);CHKERRQ(ierr);
    ierr = VecGetSize(F, &M);CHKERRQ(ierr);
    ierr = MatSetSizes(B, m, n, M, N);CHKERRQ(ierr);
    ierr = PetscLayoutSetUp(B->rmap);CHKERRQ(ierr);
    ierr = PetscLayoutSetUp(B->cmap);CHKERRQ(ierr);
    ierr = MatLMVMReset(B, PETSC_TRUE);CHKERRQ(ierr);
  } else {
    /* Mat sizing has been set so check if the vectors are compatible */
    VecCheckMatCompatible(B, X, 2, F, 3);
  }
  if (lmvm->allocated) {
    ierr = VecGetType(X, &type);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)lmvm->Xprev, type, &same);CHKERRQ(ierr);
    if (!same) {
      /* Given X vector has a different type than allocated X-type data structures.
         We need to destroy all of this and duplicate again out of the given vector. */
      allocate = PETSC_TRUE;
      ierr = MatLMVMReset(B, PETSC_TRUE);CHKERRQ(ierr);
    }
  } else {
    allocate = PETSC_TRUE;
  }
  if (allocate) {
    ierr = VecDuplicateVecs(X, lmvm->m, &lmvm->S);CHKERRQ(ierr);
    ierr = VecDuplicate(X, &lmvm->Xprev);CHKERRQ(ierr);
    ierr = VecDuplicate(X, &lmvm->R);CHKERRQ(ierr);
    ierr = VecDuplicateVecs(F, lmvm->m, &lmvm->Y);CHKERRQ(ierr);
    ierr = VecDuplicate(F, &lmvm->Fprev);CHKERRQ(ierr);
    ierr = VecDuplicate(F, &lmvm->Q);CHKERRQ(ierr);
    lmvm->allocated = PETSC_TRUE;
    B->preallocated = PETSC_TRUE;
    B->assembled = PETSC_TRUE;
  }
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

   Input Parameters:
+  B - An LMVM matrix type (LDFP LBFGS, LSR1, LBRDN, LMBRDN, LSBRDN)
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
  lmvm->k = -1;
  lmvm->prev_set = PETSC_FALSE;
  if (destructive && lmvm->allocated) {
    B->rmap->n = B->rmap->N = B->cmap->n = B->cmap->N = 0;
    ierr = VecDestroyVecs(lmvm->m, &lmvm->S);CHKERRQ(ierr);
    ierr = VecDestroy(&lmvm->Xprev);CHKERRQ(ierr);
    ierr = VecDestroy(&lmvm->R);CHKERRQ(ierr);
    ierr = VecDestroyVecs(lmvm->m, &lmvm->Y);CHKERRQ(ierr);
    ierr = VecDestroy(&lmvm->Fprev);CHKERRQ(ierr);
    ierr = VecDestroy(&lmvm->Q);CHKERRQ(ierr);
    if (lmvm->allocatedP) {
      ierr = VecDestroyVecs(lmvm->m, &lmvm->P);CHKERRQ(ierr);
      lmvm->allocatedP = PETSC_FALSE;
    }
    lmvm->nupdates = 0;
    lmvm->nrejects = 0;
    lmvm->allocated = PETSC_FALSE;
    B->preallocated = PETSC_FALSE;
    B->assembled = PETSC_FALSE;
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
.  B - An LMVM matrix type (LDFP LBFGS, LSR1, LBRDN, LMBRDN, LSBRDN)

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
.  B - An LMVM matrix type (LDFP LBFGS, LSR1, LBRDN, LMBRDN, LSBRDN)

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

/*------------------------------------------------------------*/

PetscErrorCode MatGetVecs_LMVM(Mat B, Vec *L, Vec *R)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  PetscErrorCode    ierr;
  PetscBool         same;
  MPI_Comm          comm = PetscObjectComm((PetscObject)B);
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, MAT_CLASSID, 1);
  ierr = PetscObjectBaseTypeCompare((PetscObject)B, MATLMVM, &same);CHKERRQ(ierr);
  if (!same) SETERRQ(comm, PETSC_ERR_ARG_WRONG, "Matrix must be an LMVM-type.");
  if (!lmvm->allocated) SETERRQ(comm, PETSC_ERR_ORDER, "LMVM matrix must be allocated first");
  ierr = VecDuplicate(lmvm->Xprev, L);CHKERRQ(ierr);
  ierr = VecDuplicate(lmvm->Fprev, R);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PetscErrorCode MatView_LMVM(Mat B, PetscViewer pv)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  PetscErrorCode    ierr;
  PetscBool         isascii;
  MatType           type;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)pv,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = MatGetType(B, &type);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(pv,"LMVM Matrix\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(pv,"  Approx. type: %s\n",type);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(pv,"  Max. storage: %D\n",lmvm->m);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(pv,"  Used storage: %D\n",lmvm->k+1);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(pv,"  # of updates: %D\n",lmvm->nupdates);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(pv,"  # of rejects: %D\n",lmvm->nrejects);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PetscErrorCode MatSetFromOptions_LMVM(PetscOptionItems *PetscOptionsObject, Mat B)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"Limited-memory Variable Metric matrix for approximating Jacobians");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-mat_lmvm_num_vecs","number of correction vectors kept in memory for the approximation","",lmvm->m,&lmvm->m,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-mat_lmvm_ksp_its","(developer) fixed number of KSP iterations to take when inverting J0","",lmvm->ksp_max_it,&lmvm->ksp_max_it,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mat_lmvm_eps","(developer) machine zero definition","",lmvm->eps,&lmvm->eps,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mat_lmvm_phi","(developer) convex combination factor for symmetric Broyden","",lmvm->phi,&lmvm->phi,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  ierr = KSPSetFromOptions(lmvm->J0ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PetscErrorCode MatSetUp_LMVM(Mat B)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  PetscErrorCode    ierr;
  PetscInt          m, n, M, N;
  PetscInt          comm_size;
  MPI_Comm          comm = PetscObjectComm((PetscObject)B);
  
  PetscFunctionBegin;
  ierr = MatGetSize(B, &M, &N);CHKERRQ(ierr);
  if (M == 0 && N == 0) SETERRQ(comm, PETSC_ERR_ORDER, "MatSetSizes() must be called before MatSetUp()");
  if (!lmvm->allocated) {
    ierr = MPI_Comm_size(comm, &comm_size);CHKERRQ(ierr);
    if (comm_size == 1) {
      ierr = VecCreateSeq(comm, N, &lmvm->Xprev);CHKERRQ(ierr);
      ierr = VecCreateSeq(comm, M, &lmvm->Fprev);CHKERRQ(ierr);
    } else {
      ierr = MatGetLocalSize(B, &m, &n);CHKERRQ(ierr);
      ierr = VecCreateMPI(comm, n, N, &lmvm->Xprev);CHKERRQ(ierr);
      ierr = VecCreateMPI(comm, m, M, &lmvm->Fprev);CHKERRQ(ierr);
    }
    ierr = VecDuplicateVecs(lmvm->Xprev, lmvm->m, &lmvm->S);CHKERRQ(ierr);
    ierr = VecDuplicate(lmvm->Xprev, &lmvm->R);CHKERRQ(ierr);
    ierr = VecDuplicateVecs(lmvm->Fprev, lmvm->m, &lmvm->Y);CHKERRQ(ierr);
    ierr = VecDuplicate(lmvm->Fprev, &lmvm->Q);CHKERRQ(ierr);
    lmvm->allocated = PETSC_TRUE;
    B->preallocated = PETSC_TRUE;
    B->assembled = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PetscErrorCode MatDestroy_LMVM(Mat B)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (lmvm->allocated) {
    ierr = VecDestroyVecs(lmvm->m, &lmvm->S);CHKERRQ(ierr);
    ierr = VecDestroy(&lmvm->Xprev);CHKERRQ(ierr);
    ierr = VecDestroy(&lmvm->R);CHKERRQ(ierr);
    ierr = VecDestroyVecs(lmvm->m, &lmvm->Y);CHKERRQ(ierr);
    ierr = VecDestroy(&lmvm->Fprev);CHKERRQ(ierr);
    ierr = VecDestroy(&lmvm->Q);CHKERRQ(ierr);
  }
  ierr = KSPDestroy(&lmvm->J0ksp);CHKERRQ(ierr);
  if(lmvm->diag_scale) {
    ierr = VecDestroy(&lmvm->diag_scale);CHKERRQ(ierr);
  }
  if (lmvm->J0) {
    ierr = MatDestroy(&lmvm->J0);CHKERRQ(ierr);
  }
  if (lmvm->allocatedP) {
    ierr = VecDestroyVecs(lmvm->m, &lmvm->P);CHKERRQ(ierr);
  }
  ierr = PetscFree(B->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PetscErrorCode MatCreate_LMVM(Mat B)
{
  Mat_LMVM          *lmvm;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(B, &lmvm);CHKERRQ(ierr);
  B->data = (void*)lmvm;
  lmvm->m = 10;
  lmvm->k = -1;
  lmvm->nupdates = 0;
  lmvm->nrejects = 0;
  lmvm->phi = 0.125;
  lmvm->ksp_max_it = 20;
  lmvm->ksp_rtol = 0.0;
  lmvm->ksp_atol = 0.0;
  lmvm->eps = PetscPowReal(PETSC_MACHINE_EPSILON, 2.0/3.0);
  lmvm->allocated = PETSC_FALSE;
  lmvm->allocatedP = PETSC_FALSE;
  lmvm->prev_set = PETSC_FALSE;
  lmvm->user_scale = PETSC_FALSE;
  lmvm->user_pc = PETSC_FALSE;
  lmvm->user_ksp = PETSC_FALSE;
  lmvm->square = PETSC_FALSE;
  
  B->factortype = MAT_FACTOR_LMVM;
  B->ops->destroy = MatDestroy_LMVM;
  B->ops->setfromoptions = MatSetFromOptions_LMVM;
  B->ops->view = MatView_LMVM;
  B->ops->setup = MatSetUp_LMVM;
  B->ops->getvecs = MatGetVecs_LMVM;
  
  ierr = KSPCreate(PetscObjectComm((PetscObject)B), &lmvm->J0ksp);CHKERRQ(ierr);
  ierr = PetscObjectIncrementTabLevel((PetscObject)lmvm->J0ksp, (PetscObject)B, 1);CHKERRQ(ierr);
  ierr = KSPSetOptionsPrefix(lmvm->J0ksp, "mat_lmvm_");CHKERRQ(ierr);
  ierr = KSPSetType(lmvm->J0ksp, KSPGMRES);CHKERRQ(ierr);
  ierr = KSPSetTolerances(lmvm->J0ksp, lmvm->ksp_rtol, lmvm->ksp_atol, PETSC_DEFAULT, lmvm->ksp_max_it);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}