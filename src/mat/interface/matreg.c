
/*
     Mechanism for register PETSc matrix types
*/
#include <petsc/private/matimpl.h>      /*I "petscmat.h" I*/

PetscBool MatRegisterAllCalled = PETSC_FALSE;

/*
   Contains the list of registered Mat routines
*/
PetscFunctionList MatList = NULL;

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

.seealso: PCSetType(), VecSetType(), MatCreate(), MatType, Mat
@*/
PetscErrorCode  MatSetType(Mat mat, MatType matype)
{
  PetscErrorCode ierr,(*r)(Mat);
  PetscBool      sametype,found,subclass = PETSC_FALSE;
  MatRootName    names = MatRootNameList;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);

  while (names) {
    ierr = PetscStrcmp(matype,names->rname,&found);CHKERRQ(ierr);
    if (found) {
      PetscMPIInt size;
      ierr = MPI_Comm_size(PetscObjectComm((PetscObject)mat),&size);CHKERRMPI(ierr);
      if (size == 1) matype = names->sname;
      else matype = names->mname;
      break;
    }
    names = names->next;
  }

  ierr = PetscObjectTypeCompare((PetscObject)mat,matype,&sametype);CHKERRQ(ierr);
  if (sametype) PetscFunctionReturn(0);

  ierr = PetscFunctionListFind(MatList,matype,&r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown Mat type given: %s",matype);

  if (mat->assembled && ((PetscObject)mat)->type_name) {
    ierr = PetscStrbeginswith(matype,((PetscObject)mat)->type_name,&subclass);CHKERRQ(ierr);
  }
  if (subclass) {
    ierr = MatConvert(mat,matype,MAT_INPLACE_MATRIX,&mat);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  if (mat->ops->destroy) {
    /* free the old data structure if it existed */
    ierr = (*mat->ops->destroy)(mat);CHKERRQ(ierr);
    mat->ops->destroy = NULL;

    /* should these null spaces be removed? */
    ierr = MatNullSpaceDestroy(&mat->nullsp);CHKERRQ(ierr);
    ierr = MatNullSpaceDestroy(&mat->nearnullsp);CHKERRQ(ierr);
  }
  ierr = PetscMemzero(mat->ops,sizeof(struct _MatOps));CHKERRQ(ierr);
  mat->preallocated  = PETSC_FALSE;
  mat->assembled     = PETSC_FALSE;
  mat->was_assembled = PETSC_FALSE;

  /*
   Increment, rather than reset these: the object is logically the same, so its logging and
   state is inherited.  Furthermore, resetting makes it possible for the same state to be
   obtained with a different structure, confusing the PC.
  */
  mat->nonzerostate++;
  ierr = PetscObjectStateIncrease((PetscObject)mat);CHKERRQ(ierr);

  /* create the new data structure */
  ierr = (*r)(mat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   MatGetType - Gets the matrix type as a string from the matrix object.

   Not Collective

   Input Parameter:
.  mat - the matrix

   Output Parameter:
.  name - name of matrix type

   Level: intermediate

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

/*@C
   MatGetVecType - Gets the vector type used by the matrix object.

   Not Collective

   Input Parameter:
.  mat - the matrix

   Output Parameter:
.  name - name of vector type

   Level: intermediate

.seealso: MatSetVecType()
@*/
PetscErrorCode MatGetVecType(Mat mat,VecType *vtype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidPointer(vtype,2);
  *vtype = mat->defaultvectype;
  PetscFunctionReturn(0);
}

/*@C
   MatSetVecType - Set the vector type to be used for a matrix object

   Collective on Mat

   Input Parameters:
+  mat   - the matrix object
-  vtype - vector type

   Notes:
     This is rarely needed in practice since each matrix object internally sets the proper vector type.

  Level: intermediate

.seealso: VecSetType(), MatGetVecType()
@*/
PetscErrorCode MatSetVecType(Mat mat,VecType vtype)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  ierr = PetscFree(mat->defaultvectype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(vtype,&mat->defaultvectype);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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

.seealso: MatRegisterAll()

  Level: advanced
@*/
PetscErrorCode  MatRegister(const char sname[],PetscErrorCode (*function)(Mat))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatInitializePackage();CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&MatList,sname,function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

MatRootName MatRootNameList = NULL;

/*@C
      MatRegisterRootName - Registers a name that can be used for either a sequential or its corresponding parallel matrix type. MatSetType()
        and -mat_type will automatically use the sequential or parallel version based on the size of the MPI communicator associated with the
        matrix.

  Input Parameters:
+     rname - the rootname, for example, MATAIJ
.     sname - the name of the sequential matrix type, for example, MATSEQAIJ
-     mname - the name of the parallel matrix type, for example, MATMPIAIJ

  Notes: The matrix rootname should not be confused with the base type of the function PetscObjectBaseTypeCompare()

  Developer Notes: PETSc vectors have a similar rootname that indicates PETSc should automatically select the appropriate VecType based on the
      size of the communicator but it is implemented by simply having additional VecCreate_RootName() registerer routines that dispatch to the
      appropriate creation routine. Why have two different ways of implementing the same functionality for different types of objects? It is
      confusing.

  Level: developer

.seealso: PetscObjectBaseTypeCompare()

@*/
PetscErrorCode  MatRegisterRootName(const char rname[],const char sname[],const char mname[])
{
  PetscErrorCode ierr;
  MatRootName    names;

  PetscFunctionBegin;
  ierr = PetscNew(&names);CHKERRQ(ierr);
  ierr = PetscStrallocpy(rname,&names->rname);CHKERRQ(ierr);
  ierr = PetscStrallocpy(sname,&names->sname);CHKERRQ(ierr);
  ierr = PetscStrallocpy(mname,&names->mname);CHKERRQ(ierr);
  if (!MatRootNameList) {
    MatRootNameList = names;
  } else {
    MatRootName next = MatRootNameList;
    while (next->next) next = next->next;
    next->next = names;
  }
  PetscFunctionReturn(0);
}
