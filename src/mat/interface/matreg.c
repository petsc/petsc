
/*
     Mechanism for register PETSc matrix types
*/
#include <petsc-private/matimpl.h>      /*I "petscmat.h" I*/

PetscBool MatRegisterAllCalled = PETSC_FALSE;

/*
   Contains the list of registered Mat routines
*/
PetscFunctionList MatList = 0;

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
PetscErrorCode  MatSetType(Mat mat, MatType matype)
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
      ierr = MPI_Comm_size(PetscObjectComm((PetscObject)mat),&size);CHKERRQ(ierr);
      if (size == 1) matype = names->sname;
      else matype = names->mname;
      break;
    }
    names = names->next;
  }

  ierr = PetscObjectTypeCompare((PetscObject)mat,matype,&sametype);CHKERRQ(ierr);
  if (sametype) PetscFunctionReturn(0);

  ierr =  PetscFunctionListFind(MatList,matype,&r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown Mat type given: %s",matype);

  /* free the old data structure if it existed */
  if (mat->ops->destroy) {
    ierr = (*mat->ops->destroy)(mat);CHKERRQ(ierr);

    mat->ops->destroy = NULL;
  }
  mat->preallocated = PETSC_FALSE;

  /* create the new data structure */
  ierr = (*r)(mat);CHKERRQ(ierr);
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
PetscErrorCode  MatGetType(Mat mat,MatType *type)
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
  MatRegister -  - Adds a new matrix type

   Not Collective

   Input Parameters:
+  name - name of a new user-defined matrix type
-  routine_create - routine to create method context

   Notes:
   MatRegister() may be called multiple times to add several user-defined solvers.

   Sample usage:
.vb
   MatRegister("my_mat",MyMatCreate);
.ve

   Then, your solver can be chosen with the procedural interface via
$     MatSetType(Mat,"my_mat")
   or at runtime via the option
$     -mat_type my_mat

   Level: advanced

.keywords: Mat, register

.seealso: MatRegisterAll(), MatRegisterDestroy()


  Level: advanced
@*/
PetscErrorCode  MatRegister(const char sname[],PetscErrorCode (*function)(Mat))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListAdd(&MatList,sname,function);CHKERRQ(ierr);
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







