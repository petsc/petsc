#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: matreg.c,v 1.18 2001/07/20 21:19:21 bsmith Exp $";
#endif
/*
     Mechanism for register PETSc matrix types
*/
#include "src/mat/matimpl.h"      /*I "petscmat.h" I*/
#include "petscsys.h"

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
int MatSetType(Mat mat,const MatType matype)
{
  int        ierr,(*r)(Mat);
  PetscTruth sametype;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);

  ierr = PetscTypeCompare((PetscObject)mat,matype,&sametype);CHKERRQ(ierr);
  if (!sametype) {
    /* Get the function pointers for the matrix requested */
    if (!MatRegisterAllCalled) {ierr = MatRegisterAll(PETSC_NULL);CHKERRQ(ierr);}
    ierr =  PetscFListFind(mat->comm,MatList,matype,(void(**)(void))&r);CHKERRQ(ierr);
    if (!r) SETERRQ1(1,"Unknown Mat type given: %s",matype);

    /* free the old data structure if it existed */
    if (mat->ops->destroy) {
      MatPreallocated(mat);
      ierr = (*mat->ops->destroy)(mat);CHKERRQ(ierr);
      mat->ops->destroy = PETSC_NULL;
      mat->preallocated = PETSC_FALSE;
    }

    if (mat->rmap) {
      ierr = PetscMapDestroy(mat->rmap);CHKERRQ(ierr);
      mat->rmap = 0;
    }
    if (mat->cmap) {
      ierr = PetscMapDestroy(mat->cmap);CHKERRQ(ierr);
      mat->cmap = 0;
    }
    if (mat->qlist) {
      ierr = PetscFListDestroy(&mat->qlist);CHKERRQ(ierr);
      mat->qlist = 0;
    }
    if (mat->olist) {
      ierr = PetscOListDestroy(&mat->olist);CHKERRQ(ierr);
      mat->olist = 0;
    }
    /* create the new data structure */
    ierr = (*r)(mat);CHKERRQ(ierr);

    ierr = PetscObjectChangeTypeName((PetscObject)mat,matype);CHKERRQ(ierr);
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
int MatRegisterDestroy(void)
{
  int ierr;

  PetscFunctionBegin;
  if (MatList) {
    ierr = PetscFListDestroy(&MatList);CHKERRQ(ierr);
    MatList = 0;
  }
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
int MatGetType(Mat mat,MatType *type)
{
  PetscFunctionBegin;
  *type = mat->type_name;
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MatRegister"
/*@C
  MatRegister - See MatRegisterDynamic()

  Level: advanced
@*/
int MatRegister(const char sname[],const char path[],const char name[],int (*function)(Mat))
{
  int  ierr;
  char fullname[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;
  ierr = PetscFListConcat(path,name,fullname);CHKERRQ(ierr);
  ierr = PetscFListAdd(&MatList,sname,fullname,(void (*)(void))function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}










