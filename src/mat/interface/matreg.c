#define PETSCMAT_DLL

/*
     Mechanism for register PETSc matrix types
*/
#include "private/matimpl.h"      /*I "petscmat.h" I*/

PetscTruth MatRegisterAllCalled = PETSC_FALSE;

/*
   Contains the list of registered Mat routines
*/
PetscFList MatList = 0;

#undef __FUNCT__  
#define __FUNCT__ "MatSetType"
/*@C
   MatSetType - Builds matrix object for a particular matrix type

   Collective on Mat

   Input Parameters:
+  mat      - the matrix object
-  matype   - matrix type

   Options Database Key:
.  -mat_type  <method> - Sets the type; use -help for a list 
    of available methods (for instance, seqaij)

   Notes:  
   See "${PETSC_DIR}/include/petscmat.h" for available methods

  Level: intermediate

.keywords: Mat, MatType, set, method

.seealso: PCSetType(), VecSetType(), MatCreate(), MatType, Mat
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatSetType(Mat mat, const MatType matype)
{
  PetscErrorCode ierr,(*r)(Mat);
  PetscTruth     sametype;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);

  ierr = PetscTypeCompare((PetscObject)mat,matype,&sametype);CHKERRQ(ierr);
  if (sametype) PetscFunctionReturn(0);

  ierr =  PetscFListFind(MatList,((PetscObject)mat)->comm,matype,(void(**)(void))&r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown Mat type given: %s",matype);
  
  /* free the old data structure if it existed */
  if (mat->ops->destroy) {
    ierr = MatPreallocated(mat);CHKERRQ(ierr);
    ierr = (*mat->ops->destroy)(mat);CHKERRQ(ierr);
    mat->ops->destroy = PETSC_NULL;
    mat->preallocated = PETSC_FALSE;
  }

  /* create the new data structure */
  if (mat->rmap->n < 0 && mat->rmap->N < 0 && mat->cmap->n < 0 && mat->cmap->N < 0) {
    mat->ops->create = r;
  } else {
    ierr = (*r)(mat);CHKERRQ(ierr);
  }
  ierr = PetscPublishAll(mat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MatRegisterDestroy"
/*@C
   MatRegisterDestroy - Frees the list of matrix types that were
   registered by MatRegister()/MatRegisterDynamic().

   Not Collective

   Level: advanced

.keywords: Mat, register, destroy

.seealso: MatRegister(), MatRegisterAll(), MatRegisterDynamic()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatRegisterDestroy(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFListDestroy(&MatList);CHKERRQ(ierr);
  MatRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetType"
/*@C
   MatGetType - Gets the matrix type as a string from the matrix object.

   Not Collective

   Input Parameter:
.  mat - the matrix

   Output Parameter:
.  name - name of matrix type

   Level: intermediate

.keywords: Mat, MatType, get, method, name

.seealso: MatSetType()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatGetType(Mat mat,const MatType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidPointer(type,2);
  *type = ((PetscObject)mat)->type_name;
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MatRegister"
/*@C
  MatRegister - See MatRegisterDynamic()

  Level: advanced
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatRegister(const char sname[],const char path[],const char name[],PetscErrorCode (*function)(Mat))
{
  PetscErrorCode ierr;
  char           fullname[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;
  ierr = PetscFListConcat(path,name,fullname);CHKERRQ(ierr);
  ierr = PetscFListAdd(&MatList,sname,fullname,(void (*)(void))function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}










