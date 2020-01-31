#include <petscdmshell.h>       /*I    "petscdmshell.h"  I*/
#include <petscmat.h>
#include <petsc/private/dmimpl.h>

typedef struct  {
  Vec        Xglobal;
  Vec        Xlocal;
  Mat        A;
  VecScatter gtol;
  VecScatter ltog;
  VecScatter ltol;
  void       *ctx;
} DM_Shell;

/*@
   DMGlobalToLocalBeginDefaultShell - Uses the GlobalToLocal VecScatter context set by the user to begin a global to local scatter
   Collective

   Input Arguments:
+  dm - shell DM
.  g - global vector
.  mode - InsertMode
-  l - local vector

   Level: advanced

   Note:  This is not normally called directly by user code, generally user code calls DMGlobalToLocalBegin() and DMGlobalToLocalEnd(). If the user provides their own custom routines to DMShellSetLocalToGlobal() then those routines might have reason to call this function.

.seealso: DMGlobalToLocalEndDefaultShell()
@*/
PetscErrorCode DMGlobalToLocalBeginDefaultShell(DM dm,Vec g,InsertMode mode,Vec l)
{
  PetscErrorCode ierr;
  DM_Shell       *shell = (DM_Shell*)dm->data;

  PetscFunctionBegin;
  if (!shell->gtol) SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_ARG_WRONGSTATE, "Cannot be used without first setting the scatter context via DMShellSetGlobalToLocalVecScatter()");
  ierr = VecScatterBegin(shell->gtol,g,l,mode,SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   DMGlobalToLocalEndDefaultShell - Uses the GlobalToLocal VecScatter context set by the user to end a global to local scatter
   Collective

   Input Arguments:
+  dm - shell DM
.  g - global vector
.  mode - InsertMode
-  l - local vector

   Level: advanced

.seealso: DMGlobalToLocalBeginDefaultShell()
@*/
PetscErrorCode DMGlobalToLocalEndDefaultShell(DM dm,Vec g,InsertMode mode,Vec l)
{
  PetscErrorCode ierr;
  DM_Shell       *shell = (DM_Shell*)dm->data;

  PetscFunctionBegin;
   if (!shell->gtol) SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_ARG_WRONGSTATE, "Cannot be used without first setting the scatter context via DMShellSetGlobalToLocalVecScatter()");
  ierr = VecScatterEnd(shell->gtol,g,l,mode,SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   DMLocalToGlobalBeginDefaultShell - Uses the LocalToGlobal VecScatter context set by the user to begin a local to global scatter
   Collective

   Input Arguments:
+  dm - shell DM
.  l - local vector
.  mode - InsertMode
-  g - global vector

   Level: advanced

   Note:  This is not normally called directly by user code, generally user code calls DMLocalToGlobalBegin() and DMLocalToGlobalEnd(). If the user provides their own custom routines to DMShellSetLocalToGlobal() then those routines might have reason to call this function.

.seealso: DMLocalToGlobalEndDefaultShell()
@*/
PetscErrorCode DMLocalToGlobalBeginDefaultShell(DM dm,Vec l,InsertMode mode,Vec g)
{
  PetscErrorCode ierr;
  DM_Shell       *shell = (DM_Shell*)dm->data;

  PetscFunctionBegin;
  if (!shell->ltog) SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_ARG_WRONGSTATE, "Cannot be used without first setting the scatter context via DMShellSetLocalToGlobalVecScatter()");
  ierr = VecScatterBegin(shell->ltog,l,g,mode,SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   DMLocalToGlobalEndDefaultShell - Uses the LocalToGlobal VecScatter context set by the user to end a local to global scatter
   Collective

   Input Arguments:
+  dm - shell DM
.  l - local vector
.  mode - InsertMode
-  g - global vector

   Level: advanced

.seealso: DMLocalToGlobalBeginDefaultShell()
@*/
PetscErrorCode DMLocalToGlobalEndDefaultShell(DM dm,Vec l,InsertMode mode,Vec g)
{
  PetscErrorCode ierr;
  DM_Shell       *shell = (DM_Shell*)dm->data;

  PetscFunctionBegin;
   if (!shell->ltog) SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_ARG_WRONGSTATE, "Cannot be used without first setting the scatter context via DMShellSetLocalToGlobalVecScatter()");
  ierr = VecScatterEnd(shell->ltog,l,g,mode,SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   DMLocalToLocalBeginDefaultShell - Uses the LocalToLocal VecScatter context set by the user to begin a local to local scatter
   Collective

   Input Arguments:
+  dm - shell DM
.  g - the original local vector
-  mode - InsertMode

   Output Parameter:
.  l  - the local vector with correct ghost values

   Level: advanced

   Note:  This is not normally called directly by user code, generally user code calls DMLocalToLocalBegin() and DMLocalToLocalEnd(). If the user provides their own custom routines to DMShellSetLocalToLocal() then those routines might have reason to call this function.

.seealso: DMLocalToLocalEndDefaultShell()
@*/
PetscErrorCode DMLocalToLocalBeginDefaultShell(DM dm,Vec g,InsertMode mode,Vec l)
{
  PetscErrorCode ierr;
  DM_Shell       *shell = (DM_Shell*)dm->data;

  PetscFunctionBegin;
  if (!shell->ltol) SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_ARG_WRONGSTATE, "Cannot be used without first setting the scatter context via DMShellSetLocalToLocalVecScatter()");
  ierr = VecScatterBegin(shell->ltol,g,l,mode,SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   DMLocalToLocalEndDefaultShell - Uses the LocalToLocal VecScatter context set by the user to end a local to local scatter
   Collective

   Input Arguments:
+  dm - shell DM
.  g - the original local vector
-  mode - InsertMode

   Output Parameter:
.  l  - the local vector with correct ghost values

   Level: advanced

.seealso: DMLocalToLocalBeginDefaultShell()
@*/
PetscErrorCode DMLocalToLocalEndDefaultShell(DM dm,Vec g,InsertMode mode,Vec l)
{
  PetscErrorCode ierr;
  DM_Shell       *shell = (DM_Shell*)dm->data;

  PetscFunctionBegin;
   if (!shell->ltol) SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_ARG_WRONGSTATE, "Cannot be used without first setting the scatter context via DMShellSetGlobalToLocalVecScatter()");
  ierr = VecScatterEnd(shell->ltol,g,l,mode,SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMCreateMatrix_Shell(DM dm,Mat *J)
{
  PetscErrorCode ierr;
  DM_Shell       *shell = (DM_Shell*)dm->data;
  Mat            A;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(J,3);
  if (!shell->A) {
    if (shell->Xglobal) {
      PetscInt m,M;
      ierr = PetscInfo(dm,"Naively creating matrix using global vector distribution without preallocation\n");CHKERRQ(ierr);
      ierr = VecGetSize(shell->Xglobal,&M);CHKERRQ(ierr);
      ierr = VecGetLocalSize(shell->Xglobal,&m);CHKERRQ(ierr);
      ierr = MatCreate(PetscObjectComm((PetscObject)dm),&shell->A);CHKERRQ(ierr);
      ierr = MatSetSizes(shell->A,m,m,M,M);CHKERRQ(ierr);
      ierr = MatSetType(shell->A,dm->mattype);CHKERRQ(ierr);
      ierr = MatSetUp(shell->A);CHKERRQ(ierr);
    } else SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_USER,"Must call DMShellSetMatrix(), DMShellSetCreateMatrix(), or provide a vector");
  }
  A = shell->A;
  /* the check below is tacky and incomplete */
  if (dm->mattype) {
    PetscBool flg,aij,seqaij,mpiaij;
    ierr = PetscObjectTypeCompare((PetscObject)A,dm->mattype,&flg);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)A,MATSEQAIJ,&seqaij);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)A,MATMPIAIJ,&mpiaij);CHKERRQ(ierr);
    ierr = PetscStrcmp(dm->mattype,MATAIJ,&aij);CHKERRQ(ierr);
    if (!flg) {
      if (!(aij && (seqaij || mpiaij))) SETERRQ2(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_NOTSAMETYPE,"Requested matrix of type %s, but only %s available",dm->mattype,((PetscObject)A)->type_name);
    }
  }
  if (((PetscObject)A)->refct < 2) { /* We have an exclusive reference so we can give it out */
    PetscBool f;

    ierr = PetscObjectReference((PetscObject)A);CHKERRQ(ierr);
    /* MATSHELL does not implement MATOP_ZERO_ENTRIES */
    ierr = MatHasOperation(A,MATOP_ZERO_ENTRIES,&f);CHKERRQ(ierr);
    if (f) { ierr = MatZeroEntries(A);CHKERRQ(ierr); }
    *J   = A;
  } else { /* Need to create a copy, could use MAT_SHARE_NONZERO_PATTERN in most cases */
    ierr = MatDuplicate(A,MAT_DO_NOT_COPY_VALUES,J);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMCreateGlobalVector_Shell(DM dm,Vec *gvec)
{
  PetscErrorCode ierr;
  DM_Shell       *shell = (DM_Shell*)dm->data;
  Vec            X;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(gvec,2);
  *gvec = 0;
  X     = shell->Xglobal;
  if (!X) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_USER,"Must call DMShellSetGlobalVector() or DMShellSetCreateGlobalVector()");
  if (((PetscObject)X)->refct < 2) { /* We have an exclusive reference so we can give it out */
    ierr  = PetscObjectReference((PetscObject)X);CHKERRQ(ierr);
    ierr  = VecZeroEntries(X);CHKERRQ(ierr);
    *gvec = X;
  } else { /* Need to create a copy */
    ierr = VecDuplicate(X,gvec);CHKERRQ(ierr);
    ierr = VecZeroEntries(*gvec);CHKERRQ(ierr);
  }
  ierr = VecSetDM(*gvec,dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMCreateLocalVector_Shell(DM dm,Vec *gvec)
{
  PetscErrorCode ierr;
  DM_Shell       *shell = (DM_Shell*)dm->data;
  Vec            X;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(gvec,2);
  *gvec = 0;
  X     = shell->Xlocal;
  if (!X) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_USER,"Must call DMShellSetLocalVector() or DMShellSetCreateLocalVector()");
  if (((PetscObject)X)->refct < 2) { /* We have an exclusive reference so we can give it out */
    ierr  = PetscObjectReference((PetscObject)X);CHKERRQ(ierr);
    ierr  = VecZeroEntries(X);CHKERRQ(ierr);
    *gvec = X;
  } else { /* Need to create a copy, could use MAT_SHARE_NONZERO_PATTERN in most cases */
    ierr = VecDuplicate(X,gvec);CHKERRQ(ierr);
    ierr = VecZeroEntries(*gvec);CHKERRQ(ierr);
  }
  ierr = VecSetDM(*gvec,dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   DMShellSetContext - set some data to be usable by this DM

   Collective

   Input Arguments:
+  dm - shell DM
-  ctx - the context

   Level: advanced

.seealso: DMCreateMatrix(), DMShellGetContext()
@*/
PetscErrorCode DMShellSetContext(DM dm,void *ctx)
{
  DM_Shell       *shell = (DM_Shell*)dm->data;
  PetscErrorCode ierr;
  PetscBool      isshell;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)dm,DMSHELL,&isshell);CHKERRQ(ierr);
  if (!isshell) PetscFunctionReturn(0);
  shell->ctx = ctx;
  PetscFunctionReturn(0);
}

/*@
   DMShellGetContext - Returns the user-provided context associated to the DM

   Collective

   Input Argument:
.  dm - shell DM

   Output Argument:
.  ctx - the context

   Level: advanced

.seealso: DMCreateMatrix(), DMShellSetContext()
@*/
PetscErrorCode DMShellGetContext(DM dm,void **ctx)
{
  DM_Shell       *shell = (DM_Shell*)dm->data;
  PetscErrorCode ierr;
  PetscBool      isshell;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)dm,DMSHELL,&isshell);CHKERRQ(ierr);
  if (!isshell) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Can only use with DMSHELL type DMs");
  *ctx = shell->ctx;
  PetscFunctionReturn(0);
}

/*@
   DMShellSetMatrix - sets a template matrix associated with the DMShell

   Collective

   Input Arguments:
+  dm - shell DM
-  J - template matrix

   Level: advanced

.seealso: DMCreateMatrix(), DMShellSetCreateMatrix(), DMShellSetContext(), DMShellGetContext()
@*/
PetscErrorCode DMShellSetMatrix(DM dm,Mat J)
{
  DM_Shell       *shell = (DM_Shell*)dm->data;
  PetscErrorCode ierr;
  PetscBool      isshell;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(J,MAT_CLASSID,2);
  ierr = PetscObjectTypeCompare((PetscObject)dm,DMSHELL,&isshell);CHKERRQ(ierr);
  if (!isshell) PetscFunctionReturn(0);
  ierr     = PetscObjectReference((PetscObject)J);CHKERRQ(ierr);
  ierr     = MatDestroy(&shell->A);CHKERRQ(ierr);
  shell->A = J;
  PetscFunctionReturn(0);
}

/*@C
   DMShellSetCreateMatrix - sets the routine to create a matrix associated with the shell DM

   Logically Collective on dm

   Input Arguments:
+  dm - the shell DM
-  func - the function to create a matrix

   Level: advanced

.seealso: DMCreateMatrix(), DMShellSetMatrix(), DMShellSetContext(), DMShellGetContext()
@*/
PetscErrorCode DMShellSetCreateMatrix(DM dm,PetscErrorCode (*func)(DM,Mat*))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  dm->ops->creatematrix = func;
  PetscFunctionReturn(0);
}

/*@
   DMShellSetGlobalVector - sets a template global vector associated with the DMShell

   Logically Collective on dm

   Input Arguments:
+  dm - shell DM
-  X - template vector

   Level: advanced

.seealso: DMCreateGlobalVector(), DMShellSetMatrix(), DMShellSetCreateGlobalVector()
@*/
PetscErrorCode DMShellSetGlobalVector(DM dm,Vec X)
{
  DM_Shell       *shell = (DM_Shell*)dm->data;
  PetscErrorCode ierr;
  PetscBool      isshell;
  DM             vdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(X,VEC_CLASSID,2);
  ierr = PetscObjectTypeCompare((PetscObject)dm,DMSHELL,&isshell);CHKERRQ(ierr);
  if (!isshell) PetscFunctionReturn(0);
  ierr = VecGetDM(X,&vdm);CHKERRQ(ierr);
  /*
      if the vector proposed as the new base global vector for the DM is a DM vector associated
      with the same DM then the current base global vector for the DM is ok and if we replace it with the new one
      we get a circular dependency that prevents the DM from being destroy when it should be.
      This occurs when SNESSet/GetNPC() is used with a SNES that does not have a user provided
      DM attached to it since the inner SNES (which shares the DM with the outer SNES) tries
      to set its input vector (which is associated with the DM) as the base global vector.
      Thanks to Juan P. Mendez Granado Re: [petsc-maint] Nonlinear conjugate gradien
      for pointing out the problem.
   */
  if (vdm == dm) PetscFunctionReturn(0);
  ierr           = PetscObjectReference((PetscObject)X);CHKERRQ(ierr);
  ierr           = VecDestroy(&shell->Xglobal);CHKERRQ(ierr);
  shell->Xglobal = X;
  PetscFunctionReturn(0);
}

/*@C
   DMShellSetCreateGlobalVector - sets the routine to create a global vector associated with the shell DM

   Logically Collective

   Input Arguments:
+  dm - the shell DM
-  func - the creation routine

   Level: advanced

.seealso: DMShellSetGlobalVector(), DMShellSetCreateMatrix(), DMShellSetContext(), DMShellGetContext()
@*/
PetscErrorCode DMShellSetCreateGlobalVector(DM dm,PetscErrorCode (*func)(DM,Vec*))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  dm->ops->createglobalvector = func;
  PetscFunctionReturn(0);
}

/*@
   DMShellSetLocalVector - sets a template local vector associated with the DMShell

   Logically Collective on dm

   Input Arguments:
+  dm - shell DM
-  X - template vector

   Level: advanced

.seealso: DMCreateLocalVector(), DMShellSetMatrix(), DMShellSetCreateLocalVector()
@*/
PetscErrorCode DMShellSetLocalVector(DM dm,Vec X)
{
  DM_Shell       *shell = (DM_Shell*)dm->data;
  PetscErrorCode ierr;
  PetscBool      isshell;
  DM             vdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(X,VEC_CLASSID,2);
  ierr = PetscObjectTypeCompare((PetscObject)dm,DMSHELL,&isshell);CHKERRQ(ierr);
  if (!isshell) PetscFunctionReturn(0);
  ierr = VecGetDM(X,&vdm);CHKERRQ(ierr);
  /*
      if the vector proposed as the new base global vector for the DM is a DM vector associated
      with the same DM then the current base global vector for the DM is ok and if we replace it with the new one
      we get a circular dependency that prevents the DM from being destroy when it should be.
      This occurs when SNESSet/GetNPC() is used with a SNES that does not have a user provided
      DM attached to it since the inner SNES (which shares the DM with the outer SNES) tries
      to set its input vector (which is associated with the DM) as the base global vector.
      Thanks to Juan P. Mendez Granado Re: [petsc-maint] Nonlinear conjugate gradien
      for pointing out the problem.
   */
  if (vdm == dm) PetscFunctionReturn(0);
  ierr = PetscObjectReference((PetscObject)X);CHKERRQ(ierr);
  ierr = VecDestroy(&shell->Xlocal);CHKERRQ(ierr);
  shell->Xlocal = X;
  PetscFunctionReturn(0);
}

/*@C
   DMShellSetCreateLocalVector - sets the routine to create a local vector associated with the shell DM

   Logically Collective

   Input Arguments:
+  dm - the shell DM
-  func - the creation routine

   Level: advanced

.seealso: DMShellSetLocalVector(), DMShellSetCreateMatrix(), DMShellSetContext(), DMShellGetContext()
@*/
PetscErrorCode DMShellSetCreateLocalVector(DM dm,PetscErrorCode (*func)(DM,Vec*))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  dm->ops->createlocalvector = func;
  PetscFunctionReturn(0);
}

/*@C
   DMShellSetGlobalToLocal - Sets the routines used to perform a global to local scatter

   Logically Collective on dm

   Input Arguments
+  dm - the shell DM
.  begin - the routine that begins the global to local scatter
-  end - the routine that ends the global to local scatter

   Notes:
    If these functions are not provided but DMShellSetGlobalToLocalVecScatter() is called then
   DMGlobalToLocalBeginDefaultShell()/DMGlobalToLocalEndDefaultShell() are used to to perform the transfers

   Level: advanced

.seealso: DMShellSetLocalToGlobal(), DMGlobalToLocalBeginDefaultShell(), DMGlobalToLocalEndDefaultShell()
@*/
PetscErrorCode DMShellSetGlobalToLocal(DM dm,PetscErrorCode (*begin)(DM,Vec,InsertMode,Vec),PetscErrorCode (*end)(DM,Vec,InsertMode,Vec)) {
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  dm->ops->globaltolocalbegin = begin;
  dm->ops->globaltolocalend = end;
  PetscFunctionReturn(0);
}

/*@C
   DMShellSetLocalToGlobal - Sets the routines used to perform a local to global scatter

   Logically Collective on dm

   Input Arguments
+  dm - the shell DM
.  begin - the routine that begins the local to global scatter
-  end - the routine that ends the local to global scatter

   Notes:
    If these functions are not provided but DMShellSetLocalToGlobalVecScatter() is called then
   DMLocalToGlobalBeginDefaultShell()/DMLocalToGlobalEndDefaultShell() are used to to perform the transfers

   Level: advanced

.seealso: DMShellSetGlobalToLocal()
@*/
PetscErrorCode DMShellSetLocalToGlobal(DM dm,PetscErrorCode (*begin)(DM,Vec,InsertMode,Vec),PetscErrorCode (*end)(DM,Vec,InsertMode,Vec)) {
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  dm->ops->localtoglobalbegin = begin;
  dm->ops->localtoglobalend = end;
  PetscFunctionReturn(0);
}

/*@C
   DMShellSetLocalToLocal - Sets the routines used to perform a local to local scatter

   Logically Collective on dm

   Input Arguments
+  dm - the shell DM
.  begin - the routine that begins the local to local scatter
-  end - the routine that ends the local to local scatter

   Notes:
    If these functions are not provided but DMShellSetLocalToLocalVecScatter() is called then
   DMLocalToLocalBeginDefaultShell()/DMLocalToLocalEndDefaultShell() are used to to perform the transfers

   Level: advanced

.seealso: DMShellSetGlobalToLocal(), DMLocalToLocalBeginDefaultShell(), DMLocalToLocalEndDefaultShell()
@*/
PetscErrorCode DMShellSetLocalToLocal(DM dm,PetscErrorCode (*begin)(DM,Vec,InsertMode,Vec),PetscErrorCode (*end)(DM,Vec,InsertMode,Vec)) {
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  dm->ops->localtolocalbegin = begin;
  dm->ops->localtolocalend = end;
  PetscFunctionReturn(0);
}

/*@
   DMShellSetGlobalToLocalVecScatter - Sets a VecScatter context for global to local communication

   Logically Collective on dm

   Input Arguments
+  dm - the shell DM
-  gtol - the global to local VecScatter context

   Level: advanced

.seealso: DMShellSetGlobalToLocal(), DMGlobalToLocalBeginDefaultShell(), DMGlobalToLocalEndDefaultShell()
@*/
PetscErrorCode DMShellSetGlobalToLocalVecScatter(DM dm, VecScatter gtol)
{
  DM_Shell       *shell = (DM_Shell*)dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(gtol,VEC_SCATTER_CLASSID,2);
  ierr = PetscObjectReference((PetscObject)gtol);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&shell->gtol);CHKERRQ(ierr);
  shell->gtol = gtol;
  PetscFunctionReturn(0);
}

/*@
   DMShellSetLocalToGlobalVecScatter - Sets a VecScatter context for local to global communication

   Logically Collective on dm

   Input Arguments
+  dm - the shell DM
-  ltog - the local to global VecScatter context

   Level: advanced

.seealso: DMShellSetLocalToGlobal(), DMLocalToGlobalBeginDefaultShell(), DMLocalToGlobalEndDefaultShell()
@*/
PetscErrorCode DMShellSetLocalToGlobalVecScatter(DM dm, VecScatter ltog)
{
  DM_Shell       *shell = (DM_Shell*)dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(ltog,VEC_SCATTER_CLASSID,2);
  ierr = PetscObjectReference((PetscObject)ltog);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&shell->ltog);CHKERRQ(ierr);
  shell->ltog = ltog;
  PetscFunctionReturn(0);
}

/*@
   DMShellSetLocalToLocalVecScatter - Sets a VecScatter context for local to local communication

   Logically Collective on dm

   Input Arguments
+  dm - the shell DM
-  ltol - the local to local VecScatter context

   Level: advanced

.seealso: DMShellSetLocalToLocal(), DMLocalToLocalBeginDefaultShell(), DMLocalToLocalEndDefaultShell()
@*/
PetscErrorCode DMShellSetLocalToLocalVecScatter(DM dm, VecScatter ltol)
{
  DM_Shell       *shell = (DM_Shell*)dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(ltol,VEC_SCATTER_CLASSID,2);
  ierr = PetscObjectReference((PetscObject)ltol);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&shell->ltol);CHKERRQ(ierr);
  shell->ltol = ltol;
  PetscFunctionReturn(0);
}

/*@C
   DMShellSetCoarsen - Set the routine used to coarsen the shell DM

   Logically Collective on dm

   Input Arguments
+  dm - the shell DM
-  coarsen - the routine that coarsens the DM

   Level: advanced

.seealso: DMShellSetRefine(), DMCoarsen(), DMShellGetCoarsen(), DMShellSetContext(), DMShellGetContext()
@*/
PetscErrorCode DMShellSetCoarsen(DM dm, PetscErrorCode (*coarsen)(DM,MPI_Comm,DM*))
{
  PetscErrorCode ierr;
  PetscBool      isshell;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)dm,DMSHELL,&isshell);CHKERRQ(ierr);
  if (!isshell) PetscFunctionReturn(0);
  dm->ops->coarsen = coarsen;
  PetscFunctionReturn(0);
}

/*@C
   DMShellGetCoarsen - Get the routine used to coarsen the shell DM

   Logically Collective on dm

   Input Argument:
.  dm - the shell DM

   Output Argument:
.  coarsen - the routine that coarsens the DM

   Level: advanced

.seealso: DMShellSetCoarsen(), DMCoarsen(), DMShellSetRefine(), DMRefine()
@*/
PetscErrorCode DMShellGetCoarsen(DM dm, PetscErrorCode (**coarsen)(DM,MPI_Comm,DM*))
{
  PetscErrorCode ierr;
  PetscBool      isshell;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)dm,DMSHELL,&isshell);CHKERRQ(ierr);
  if (!isshell) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Can only use with DMSHELL type DMs");
  *coarsen = dm->ops->coarsen;
  PetscFunctionReturn(0);
}

/*@C
   DMShellSetRefine - Set the routine used to refine the shell DM

   Logically Collective on dm

   Input Arguments
+  dm - the shell DM
-  refine - the routine that refines the DM

   Level: advanced

.seealso: DMShellSetCoarsen(), DMRefine(), DMShellGetRefine(), DMShellSetContext(), DMShellGetContext()
@*/
PetscErrorCode DMShellSetRefine(DM dm, PetscErrorCode (*refine)(DM,MPI_Comm,DM*))
{
  PetscErrorCode ierr;
  PetscBool      isshell;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)dm,DMSHELL,&isshell);CHKERRQ(ierr);
  if (!isshell) PetscFunctionReturn(0);
  dm->ops->refine = refine;
  PetscFunctionReturn(0);
}

/*@C
   DMShellGetRefine - Get the routine used to refine the shell DM

   Logically Collective on dm

   Input Argument:
.  dm - the shell DM

   Output Argument:
.  refine - the routine that refines the DM

   Level: advanced

.seealso: DMShellSetCoarsen(), DMCoarsen(), DMShellSetRefine(), DMRefine()
@*/
PetscErrorCode DMShellGetRefine(DM dm, PetscErrorCode (**refine)(DM,MPI_Comm,DM*))
{
  PetscErrorCode ierr;
  PetscBool      isshell;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)dm,DMSHELL,&isshell);CHKERRQ(ierr);
  if (!isshell) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Can only use with DMSHELL type DMs");
  *refine = dm->ops->refine;
  PetscFunctionReturn(0);
}

/*@C
   DMShellSetCreateInterpolation - Set the routine used to create the interpolation operator

   Logically Collective on dm

   Input Arguments
+  dm - the shell DM
-  interp - the routine to create the interpolation

   Level: advanced

.seealso: DMShellSetCreateInjection(), DMCreateInterpolation(), DMShellGetCreateInterpolation(), DMShellSetCreateRestriction(), DMShellSetContext(), DMShellGetContext()
@*/
PetscErrorCode DMShellSetCreateInterpolation(DM dm, PetscErrorCode (*interp)(DM,DM,Mat*,Vec*))
{
  PetscErrorCode ierr;
  PetscBool      isshell;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)dm,DMSHELL,&isshell);CHKERRQ(ierr);
  if (!isshell) PetscFunctionReturn(0);
  dm->ops->createinterpolation = interp;
  PetscFunctionReturn(0);
}

/*@C
   DMShellGetCreateInterpolation - Get the routine used to create the interpolation operator

   Logically Collective on dm

   Input Argument:
+  dm - the shell DM

   Output Argument:
-  interp - the routine to create the interpolation

   Level: advanced

.seealso: DMShellGetCreateInjection(), DMCreateInterpolation(), DMShellGetCreateRestriction(), DMShellSetContext(), DMShellGetContext()
@*/
PetscErrorCode DMShellGetCreateInterpolation(DM dm, PetscErrorCode (**interp)(DM,DM,Mat*,Vec*))
{
  PetscErrorCode ierr;
  PetscBool      isshell;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)dm,DMSHELL,&isshell);CHKERRQ(ierr);
  if (!isshell) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Can only use with DMSHELL type DMs");
  *interp = dm->ops->createinterpolation;
  PetscFunctionReturn(0);
}

/*@C
   DMShellSetCreateRestriction - Set the routine used to create the restriction operator

   Logically Collective on dm

   Input Arguments
+  dm - the shell DM
-  striction- the routine to create the restriction

   Level: advanced

.seealso: DMShellSetCreateInjection(), DMCreateInterpolation(), DMShellGetCreateRestriction(), DMShellSetContext(), DMShellGetContext()
@*/
PetscErrorCode DMShellSetCreateRestriction(DM dm, PetscErrorCode (*restriction)(DM,DM,Mat*))
{
  PetscErrorCode ierr;
  PetscBool      isshell;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)dm,DMSHELL,&isshell);CHKERRQ(ierr);
  if (!isshell) PetscFunctionReturn(0);
  dm->ops->createrestriction = restriction;
  PetscFunctionReturn(0);
}

/*@C
   DMShellGetCreateRestriction - Get the routine used to create the restriction operator

   Logically Collective on dm

   Input Argument:
+  dm - the shell DM

   Output Argument:
-  restriction - the routine to create the restriction

   Level: advanced

.seealso: DMShellSetCreateInjection(), DMCreateInterpolation(), DMShellSetContext(), DMShellGetContext()
@*/
PetscErrorCode DMShellGetCreateRestriction(DM dm, PetscErrorCode (**restriction)(DM,DM,Mat*))
{
  PetscErrorCode ierr;
  PetscBool      isshell;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)dm,DMSHELL,&isshell);CHKERRQ(ierr);
  if (!isshell) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Can only use with DMSHELL type DMs");
  *restriction = dm->ops->createrestriction;
  PetscFunctionReturn(0);
}

/*@C
   DMShellSetCreateInjection - Set the routine used to create the injection operator

   Logically Collective on dm

   Input Arguments
+  dm - the shell DM
-  inject - the routine to create the injection

   Level: advanced

.seealso: DMShellSetCreateInterpolation(), DMCreateInjection(), DMShellGetCreateInjection(), DMShellSetContext(), DMShellGetContext()
@*/
PetscErrorCode DMShellSetCreateInjection(DM dm, PetscErrorCode (*inject)(DM,DM,Mat*))
{
  PetscErrorCode ierr;
  PetscBool      isshell;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)dm,DMSHELL,&isshell);CHKERRQ(ierr);
  if (!isshell) PetscFunctionReturn(0);
  dm->ops->createinjection = inject;
  PetscFunctionReturn(0);
}

/*@C
   DMShellGetCreateInjection - Get the routine used to create the injection operator

   Logically Collective on dm

   Input Argument:
+  dm - the shell DM

   Output Argument:
-  inject - the routine to create the injection

   Level: advanced

.seealso: DMShellGetCreateInterpolation(), DMCreateInjection(), DMShellSetContext(), DMShellGetContext()
@*/
PetscErrorCode DMShellGetCreateInjection(DM dm, PetscErrorCode (**inject)(DM,DM,Mat*))
{
  PetscErrorCode ierr;
  PetscBool      isshell;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)dm,DMSHELL,&isshell);CHKERRQ(ierr);
  if (!isshell) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Can only use with DMSHELL type DMs");
  *inject = dm->ops->createinjection;
  PetscFunctionReturn(0);
}

/*@C
   DMShellSetCreateFieldDecomposition - Set the routine used to create a decomposition of fields for the shell DM

   Logically Collective on dm

   Input Arguments
+  dm - the shell DM
-  decomp - the routine to create the decomposition

   Level: advanced

.seealso: DMCreateFieldDecomposition(), DMShellSetContext(), DMShellGetContext()
@*/
PetscErrorCode DMShellSetCreateFieldDecomposition(DM dm, PetscErrorCode (*decomp)(DM,PetscInt*,char***, IS**,DM**))
{
  PetscErrorCode ierr;
  PetscBool      isshell;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)dm,DMSHELL,&isshell);CHKERRQ(ierr);
  if (!isshell) PetscFunctionReturn(0);
  dm->ops->createfielddecomposition = decomp;
  PetscFunctionReturn(0);
}

/*@C
   DMShellSetCreateDomainDecomposition - Set the routine used to create a domain decomposition for the shell DM

   Logically Collective on dm

   Input Arguments
+  dm - the shell DM
-  decomp - the routine to create the decomposition

   Level: advanced

.seealso: DMCreateDomainDecomposition(), DMShellSetContext(), DMShellGetContext()
@*/
PetscErrorCode DMShellSetCreateDomainDecomposition(DM dm, PetscErrorCode (*decomp)(DM,PetscInt*,char***, IS**,IS**,DM**))
{
  PetscErrorCode ierr;
  PetscBool      isshell;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)dm,DMSHELL,&isshell);CHKERRQ(ierr);
  if (!isshell) PetscFunctionReturn(0);
  dm->ops->createdomaindecomposition = decomp;
  PetscFunctionReturn(0);
}

/*@C
   DMShellSetCreateDomainDecompositionScatters - Set the routine used to create the scatter contexts for domain decomposition with a shell DM

   Logically Collective on dm

   Input Arguments
+  dm - the shell DM
-  scatter - the routine to create the scatters

   Level: advanced

.seealso: DMCreateDomainDecompositionScatters(), DMShellSetContext(), DMShellGetContext()
@*/
PetscErrorCode DMShellSetCreateDomainDecompositionScatters(DM dm, PetscErrorCode (*scatter)(DM,PetscInt,DM*,VecScatter**,VecScatter**,VecScatter**))
{
  PetscErrorCode ierr;
  PetscBool      isshell;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)dm,DMSHELL,&isshell);CHKERRQ(ierr);
  if (!isshell) PetscFunctionReturn(0);
  dm->ops->createddscatters = scatter;
  PetscFunctionReturn(0);
}

/*@C
   DMShellSetCreateSubDM - Set the routine used to create a sub DM from the shell DM

   Logically Collective on dm

   Input Arguments
+  dm - the shell DM
-  subdm - the routine to create the decomposition

   Level: advanced

.seealso: DMCreateSubDM(), DMShellGetCreateSubDM(), DMShellSetContext(), DMShellGetContext()
@*/
PetscErrorCode DMShellSetCreateSubDM(DM dm, PetscErrorCode (*subdm)(DM,PetscInt,const PetscInt[],IS*,DM*))
{
  PetscErrorCode ierr;
  PetscBool      isshell;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)dm,DMSHELL,&isshell);CHKERRQ(ierr);
  if (!isshell) PetscFunctionReturn(0);
  dm->ops->createsubdm = subdm;
  PetscFunctionReturn(0);
}

/*@C
   DMShellGetCreateSubDM - Get the routine used to create a sub DM from the shell DM

   Logically Collective on dm

   Input Argument:
+  dm - the shell DM

   Output Argument:
-  subdm - the routine to create the decomposition

   Level: advanced

.seealso: DMCreateSubDM(), DMShellSetCreateSubDM(), DMShellSetContext(), DMShellGetContext()
@*/
PetscErrorCode DMShellGetCreateSubDM(DM dm, PetscErrorCode (**subdm)(DM,PetscInt,const PetscInt[],IS*,DM*))
{
  PetscErrorCode ierr;
  PetscBool      isshell;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)dm,DMSHELL,&isshell);CHKERRQ(ierr);
  if (!isshell) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Can only use with DMSHELL type DMs");
  *subdm = dm->ops->createsubdm;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMDestroy_Shell(DM dm)
{
  PetscErrorCode ierr;
  DM_Shell       *shell = (DM_Shell*)dm->data;

  PetscFunctionBegin;
  ierr = MatDestroy(&shell->A);CHKERRQ(ierr);
  ierr = VecDestroy(&shell->Xglobal);CHKERRQ(ierr);
  ierr = VecDestroy(&shell->Xlocal);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&shell->gtol);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&shell->ltog);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&shell->ltol);CHKERRQ(ierr);
  /* This was originally freed in DMDestroy(), but that prevents reference counting of backend objects */
  ierr = PetscFree(shell);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMView_Shell(DM dm,PetscViewer v)
{
  PetscErrorCode ierr;
  DM_Shell       *shell = (DM_Shell*)dm->data;

  PetscFunctionBegin;
  ierr = VecView(shell->Xglobal,v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMLoad_Shell(DM dm,PetscViewer v)
{
  PetscErrorCode ierr;
  DM_Shell       *shell = (DM_Shell*)dm->data;

  PetscFunctionBegin;
  ierr = VecCreate(PetscObjectComm((PetscObject)dm),&shell->Xglobal);CHKERRQ(ierr);
  ierr = VecLoad(shell->Xglobal,v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMCreateSubDM_Shell(DM dm, PetscInt numFields, const PetscInt fields[], IS *is, DM *subdm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (subdm) {ierr = DMShellCreate(PetscObjectComm((PetscObject) dm), subdm);CHKERRQ(ierr);}
  ierr = DMCreateSectionSubDM(dm, numFields, fields, is, subdm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode DMCreate_Shell(DM dm)
{
  PetscErrorCode ierr;
  DM_Shell       *shell;

  PetscFunctionBegin;
  ierr     = PetscNewLog(dm,&shell);CHKERRQ(ierr);
  dm->data = shell;

  dm->ops->destroy            = DMDestroy_Shell;
  dm->ops->createglobalvector = DMCreateGlobalVector_Shell;
  dm->ops->createlocalvector  = DMCreateLocalVector_Shell;
  dm->ops->creatematrix       = DMCreateMatrix_Shell;
  dm->ops->view               = DMView_Shell;
  dm->ops->load               = DMLoad_Shell;
  dm->ops->globaltolocalbegin = DMGlobalToLocalBeginDefaultShell;
  dm->ops->globaltolocalend   = DMGlobalToLocalEndDefaultShell;
  dm->ops->localtoglobalbegin = DMLocalToGlobalBeginDefaultShell;
  dm->ops->localtoglobalend   = DMLocalToGlobalEndDefaultShell;
  dm->ops->localtolocalbegin  = DMLocalToLocalBeginDefaultShell;
  dm->ops->localtolocalend    = DMLocalToLocalEndDefaultShell;
  dm->ops->createsubdm        = DMCreateSubDM_Shell;
  PetscFunctionReturn(0);
}

/*@
    DMShellCreate - Creates a shell DM object, used to manage user-defined problem data

    Collective

    Input Parameter:
.   comm - the processors that will share the global vector

    Output Parameters:
.   shell - the shell DM

    Level: advanced

.seealso DMDestroy(), DMCreateGlobalVector(), DMCreateLocalVector(), DMShellSetContext(), DMShellGetContext()
@*/
PetscErrorCode  DMShellCreate(MPI_Comm comm,DM *dm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(dm,2);
  ierr = DMCreate(comm,dm);CHKERRQ(ierr);
  ierr = DMSetType(*dm,DMSHELL);CHKERRQ(ierr);
  ierr = DMSetUp(*dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
