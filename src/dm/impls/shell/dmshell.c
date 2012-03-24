#include <petscdmshell.h>       /*I    "petscdmshell.h"  I*/
#include <petscmat.h>           /*I    "petscmat.h"      I*/
#include <petsc-private/dmimpl.h>     /*I    "petscdm.h"       I*/

typedef struct  {
  Vec Xglobal;
  Mat A;
} DM_Shell;

#undef __FUNCT__
#define __FUNCT__ "DMCreateMatrix_Shell"
static PetscErrorCode DMCreateMatrix_Shell(DM dm,const MatType mtype,Mat *J)
{
  PetscErrorCode ierr;
  DM_Shell       *shell = (DM_Shell*)dm->data;
  Mat            A;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(J,3);
  A = shell->A;
  if (!A) SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_USER,"Must call DMShellSetMatrix() or DMShellSetCreateMatrix()");
  if (mtype) {
    PetscBool flg;
    ierr = PetscTypeCompare((PetscObject)A,mtype,&flg);CHKERRQ(ierr);
    if (!flg) SETERRQ2(((PetscObject)dm)->comm,PETSC_ERR_ARG_NOTSAMETYPE,"Requested matrix of type %s, but only %s available",mtype,((PetscObject)A)->type_name);
  }
  if (((PetscObject)A)->refct < 2) { /* We have an exclusive reference so we can give it out */
    ierr = PetscObjectReference((PetscObject)A);CHKERRQ(ierr);
    ierr = MatZeroEntries(A);CHKERRQ(ierr);
    *J = A;
  } else {                      /* Need to create a copy, could use MAT_SHARE_NONZERO_PATTERN in most cases */
    ierr = MatDuplicate(A,MAT_DO_NOT_COPY_VALUES,J);CHKERRQ(ierr);
    ierr = MatZeroEntries(*J);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCreateGlobalVector_Shell"
PetscErrorCode DMCreateGlobalVector_Shell(DM dm,Vec *gvec)
{
  PetscErrorCode ierr;
  DM_Shell       *shell = (DM_Shell*)dm->data;
  Vec            X;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(gvec,2);
  *gvec = 0;
  X = shell->Xglobal;
  if (!X) SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_USER,"Must call DMShellSetGlobalVector() or DMShellSetCreateGlobalVector()");
  if (((PetscObject)X)->refct < 2) { /* We have an exclusive reference so we can give it out */
    ierr = PetscObjectReference((PetscObject)X);CHKERRQ(ierr);
    ierr = VecZeroEntries(X);CHKERRQ(ierr);
    *gvec = X;
  } else {                      /* Need to create a copy, could use MAT_SHARE_NONZERO_PATTERN in most cases */
    ierr = VecDuplicate(X,gvec);CHKERRQ(ierr);
    ierr = VecZeroEntries(*gvec);CHKERRQ(ierr);
  }
  ierr = PetscObjectCompose((PetscObject)*gvec,"DM",(PetscObject)dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMShellSetMatrix"
/*@
   DMShellSetMatrix - sets a template matrix associated with the DMShell

   Collective

   Input Arguments:
+  dm - shell DM
-  J - template matrix

   Level: advanced

.seealso: DMCreateMatrix(), DMShellSetCreateMatrix()
@*/
PetscErrorCode DMShellSetMatrix(DM dm,Mat J)
{
  DM_Shell *shell = (DM_Shell*)dm->data;
  PetscErrorCode ierr;
  PetscBool isshell;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(J,MAT_CLASSID,2);
  ierr = PetscTypeCompare((PetscObject)dm,DMSHELL,&isshell);CHKERRQ(ierr);
  if (!isshell) PetscFunctionReturn(0);
  ierr = PetscObjectReference((PetscObject)J);CHKERRQ(ierr);
  ierr = MatDestroy(&shell->A);CHKERRQ(ierr);
  shell->A = J;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMShellSetCreateMatrix"
/*@C
   DMShellSetCreateMatrix - sets the routine to create a matrix associated with the shell DM

   Logically Collective on DM

   Input Arguments:
+  dm - the shell DM
-  func - the function to create a matrix

   Level: advanced

.seealso: DMCreateMatrix(), DMShellSetMatrix()
@*/
PetscErrorCode DMShellSetCreateMatrix(DM dm,PetscErrorCode (*func)(DM,const MatType,Mat*))
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  dm->ops->creatematrix = func;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMShellSetGlobalVector"
/*@
   DMShellSetGlobalVector - sets a template global vector associated with the DMShell

   Logically Collective on DM

   Input Arguments:
+  dm - shell DM
-  X - template vector

   Level: advanced

.seealso: DMCreateGlobalVector(), DMShellSetMatrix(), DMShellSetCreateGlobalVector()
@*/
PetscErrorCode DMShellSetGlobalVector(DM dm,Vec X)
{
  DM_Shell *shell = (DM_Shell*)dm->data;
  PetscErrorCode ierr;
  PetscBool isshell;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(X,VEC_CLASSID,2);
  ierr = PetscTypeCompare((PetscObject)dm,DMSHELL,&isshell);CHKERRQ(ierr);
  if (!isshell) PetscFunctionReturn(0);
  ierr = PetscObjectReference((PetscObject)X);CHKERRQ(ierr);
  ierr = VecDestroy(&shell->Xglobal);CHKERRQ(ierr);
  shell->Xglobal = X;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMShellSetCreateGlobalVector"
/*@C
   DMShellSetCreateGlobalVector - sets the routine to create a global vector associated with the shell DM

   Logically Collective

   Input Arguments:
+  dm - the shell DM
-  func - the creation routine

   Level: advanced

.seealso: DMShellSetGlobalVector(), DMShellSetCreateMatrix()
@*/
PetscErrorCode DMShellSetCreateGlobalVector(DM dm,PetscErrorCode (*func)(DM,Vec*))
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  dm->ops->createglobalvector = func;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDestroy_Shell"
static PetscErrorCode DMDestroy_Shell(DM dm)
{
  PetscErrorCode ierr;
  DM_Shell       *shell = (DM_Shell*)dm->data;

  PetscFunctionBegin;
  ierr = MatDestroy(&shell->A);CHKERRQ(ierr);
  ierr = VecDestroy(&shell->Xglobal);CHKERRQ(ierr);
  ierr = PetscFree(dm->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMCreate_Shell"
PETSC_EXTERN_C PetscErrorCode  DMCreate_Shell(DM dm)
{
  PetscErrorCode ierr;
  DM_Shell      *shell;

  PetscFunctionBegin;
  ierr = PetscNewLog(dm,DM_Shell,&shell);CHKERRQ(ierr);
  dm->data = shell;

  ierr = PetscObjectChangeTypeName((PetscObject)dm,DMSHELL);CHKERRQ(ierr);
  dm->ops->destroy            = DMDestroy_Shell;
  dm->ops->createglobalvector = DMCreateGlobalVector_Shell;
  dm->ops->creatematrix       = DMCreateMatrix_Shell;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMShellCreate"
/*@
    DMShellCreate - Creates a shell DM object, used to manage user-defined problem data

    Collective on MPI_Comm

    Input Parameter:
.   comm - the processors that will share the global vector

    Output Parameters:
.   shell - the shell DM

    Level: advanced

.seealso DMDestroy(), DMCreateGlobalVector()
@*/
PetscErrorCode  DMShellCreate(MPI_Comm comm,DM *dm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(dm,2);
  ierr = DMCreate(comm,dm);CHKERRQ(ierr);
  ierr = DMSetType(*dm,DMSHELL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
