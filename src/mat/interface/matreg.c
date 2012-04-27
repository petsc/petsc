
/*
     Mechanism for register PETSc matrix types
*/
#include <petsc-private/matimpl.h>      /*I "petscmat.h" I*/
#include <stdarg.h> /* Variable-length arg lists. */

PetscBool  MatRegisterAllCalled = PETSC_FALSE;

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
PetscErrorCode  MatSetType(Mat mat, const MatType matype)
{
  PetscErrorCode ierr,(*r)(Mat);
  PetscBool      sametype,found;
  MatBaseName    names = MatBaseNameList;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);

  while (names) {
    ierr = PetscStrcmp(matype,names->bname,&found);CHKERRQ(ierr);
    if (found) {
      PetscMPIInt size;
      ierr = MPI_Comm_size(((PetscObject)mat)->comm,&size);CHKERRQ(ierr);
      if (size == 1) matype = names->sname;
      else matype = names->mname;
      break;
    }
    names = names->next;
  }

  ierr = PetscObjectTypeCompare((PetscObject)mat,matype,&sametype);CHKERRQ(ierr);
  if (sametype) PetscFunctionReturn(0);

  ierr =  PetscFListFind(MatList,((PetscObject)mat)->comm,matype,PETSC_TRUE,(void(**)(void))&r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown Mat type given: %s",matype);
  
  /* free the old data structure if it existed */
  if (mat->ops->destroy) {
    ierr = (*mat->ops->destroy)(mat);CHKERRQ(ierr);
    mat->ops->destroy = PETSC_NULL;
  }
  mat->preallocated = PETSC_FALSE;

  /* create the new data structure */
  ierr = (*r)(mat);CHKERRQ(ierr);
#if defined(PETSC_HAVE_AMS)
  if (PetscAMSPublishAll) {
    /*    ierr = PetscObjectAMSPublish((PetscObject)mat);CHKERRQ(ierr); */
  }
#endif
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
PetscErrorCode  MatRegisterDestroy(void)
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
PetscErrorCode  MatGetType(Mat mat,const MatType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
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
PetscErrorCode  MatRegister(const char sname[],const char path[],const char name[],PetscErrorCode (*function)(Mat))
{
  PetscErrorCode ierr;
  char           fullname[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;
  ierr = PetscFListConcat(path,name,fullname);CHKERRQ(ierr);
  ierr = PetscFListAdd(&MatList,sname,fullname,(void (*)(void))function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

MatBaseName MatBaseNameList = 0;

#undef __FUNCT__  
#define __FUNCT__ "MatRegisterBaseName"
/*@C
      MatRegisterBaseName - Registers a name that can be used for either a sequential or its corresponding parallel matrix type.

  Input Parameters:
+     bname - the basename, for example, MATAIJ
.     sname - the name of the sequential matrix type, for example, MATSEQAIJ
-     mname - the name of the parallel matrix type, for example, MATMPIAIJ


  Level: advanced
@*/
PetscErrorCode  MatRegisterBaseName(const char bname[],const char sname[],const char mname[])
{
  PetscErrorCode ierr;
  MatBaseName    names;

  PetscFunctionBegin;
  ierr = PetscNew(struct _p_MatBaseName,&names);CHKERRQ(ierr);
  ierr = PetscStrallocpy(bname,&names->bname);CHKERRQ(ierr);
  ierr = PetscStrallocpy(sname,&names->sname);CHKERRQ(ierr);
  ierr = PetscStrallocpy(mname,&names->mname);CHKERRQ(ierr);
  if (!MatBaseNameList) {
    MatBaseNameList = names;
  } else {
    MatBaseName next = MatBaseNameList;
    while (next->next) next = next->next;
    next->next = names;
  }
  PetscFunctionReturn(0);
}


/*
   Contains the list of Mat-related operations.
*/
PetscOpFList MatOpList = 0;

#undef __FUNCT__  
#define __FUNCT__ "MatRegisterOp"
/*@C
      MatRegisterOp - Registers a function via a pointer or a dynamic library url,
                      that implements a polymorphic operation that is dispatched
                      based on the op name and the declared arguments' type names.

  Formally collective on comm.

  Input Parameters:
+  comm     - processors adding the op
.  url      - routine locator  (optional, if not using dynamic libraries and a nonempty fnc)
.  function - function pointer (optional, if using dynamic libraries and a nonempty url)
.  op       - operation name
.  numArgs  - number of op arguments
-  ...      - list of argument type names (const char*)

  Level: advanced
@*/
PetscErrorCode  MatRegisterOp(MPI_Comm comm, const char url[], PetscVoidFunction function, const char op[], PetscInt numArgs, ...) {
  PetscErrorCode ierr;
  va_list ap;
  PetscInt i;
  const char *argType;
  char **argTypes = PETSC_NULL;
  PetscFunctionBegin;
  va_start(ap,numArgs);
  if(numArgs) {
    ierr = PetscMalloc(sizeof(char*)*numArgs, &argTypes); CHKERRQ(ierr);
  }
  for(i = 0; i < numArgs; ++i) {
    argType = va_arg(ap,const char*);
    ierr = PetscStrallocpy(argType, argTypes+i); CHKERRQ(ierr);
  }
  va_end(ap);
  ierr = PetscOpFListAdd(comm, &MatOpList, url, function, op, numArgs, argTypes); CHKERRQ(ierr);
  for(i = 0; i < numArgs; ++i) {
    ierr = PetscFree(argTypes[i]); CHKERRQ(ierr);
  }
  ierr = PetscFree(argTypes); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatQueryOp"
/*@C
      MatQueryOp - Finds the function that implements a polymorphic operation that is dispatched
                   based on the op name and the declared arguments' type names.

  Formally collective on comm.

  Input Parameters:
+  comm     - processors adding the op
.  op       - operation name
.  numArgs  - number of op arguments
-  ...      - list of argument type names (const char*)

  Output Parameters:
.  function -- function pointer

  Level: advanced
@*/
PetscErrorCode  MatQueryOp(MPI_Comm comm, PetscVoidFunction* function, const char op[], PetscInt numArgs, ...) {
  PetscErrorCode ierr;
  va_list ap;
  PetscInt i;
  const char *argType;
  char **argTypes = PETSC_NULL;
  PetscFunctionBegin;
  va_start(ap,numArgs);
  if(numArgs) {
    ierr = PetscMalloc(sizeof(char*)*numArgs, &argTypes); CHKERRQ(ierr);
  }
  for(i = 0; i < numArgs; ++i) {
    argType = va_arg(ap,const char*);
    ierr = PetscStrallocpy(argType, argTypes+i); CHKERRQ(ierr);
  }
  va_end(ap);
  ierr = PetscOpFListFind(comm, MatOpList, function, op, numArgs, argTypes); CHKERRQ(ierr);
  for(i = 0; i < numArgs; ++i) {
    ierr = PetscFree(argTypes[i]); CHKERRQ(ierr);
  }
  ierr = PetscFree(argTypes); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}









