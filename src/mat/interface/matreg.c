
/*
     Mechanism for register PETSc matrix types
*/
#include <petsc/private/matimpl.h>      /*I "petscmat.h" I*/

PetscBool MatRegisterAllCalled = PETSC_FALSE;

/*
   Contains the list of registered Mat routines
*/
PetscFunctionList MatList = NULL;

/* MatGetRootType_Private - Gets the root type of the input matrix's type (e.g., MATAIJ for MATSEQAIJ)

   Not Collective

   Input Parameters:
.  mat      - the input matrix, could be sequential or MPI

   Output Parameters:
.  rootType  - the root matrix type

   Level: developer

.seealso: `MatGetType()`, `MatSetType()`, `MatType`, `Mat`
*/
PetscErrorCode MatGetRootType_Private(Mat mat, MatType *rootType)
{
  PetscBool      found = PETSC_FALSE;
  MatRootName    names = MatRootNameList;
  MatType        inType;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscCall(MatGetType(mat,&inType));
  while (names) {
    PetscCall(PetscStrcmp(inType,names->mname,&found));
    if (!found) PetscCall(PetscStrcmp(inType,names->sname,&found));
    if (found) {
      found     = PETSC_TRUE;
      *rootType = names->rname;
      break;
    }
    names = names->next;
  }
  if (!found) *rootType = inType;
  PetscFunctionReturn(0);
}

/* MatGetMPIMatType_Private - Gets the MPI type corresponding to the input matrix's type (e.g., MATMPIAIJ for MATSEQAIJ)

   Not Collective

   Input Parameters:
.  mat      - the input matrix, could be sequential or MPI

   Output Parameters:
.  MPIType  - the parallel (MPI) matrix type

   Level: developer

.seealso: `MatGetType()`, `MatSetType()`, `MatType`, `Mat`
*/
PetscErrorCode MatGetMPIMatType_Private(Mat mat, MatType *MPIType)
{
  PetscBool      found = PETSC_FALSE;
  MatRootName    names = MatRootNameList;
  MatType        inType;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscCall(MatGetType(mat,&inType));
  while (names) {
    PetscCall(PetscStrcmp(inType,names->sname,&found));
    if (!found) PetscCall(PetscStrcmp(inType,names->mname,&found));
    if (!found) PetscCall(PetscStrcmp(inType,names->rname,&found));
    if (found) {
      found     = PETSC_TRUE;
      *MPIType = names->mname;
      break;
    }
    names = names->next;
  }
  PetscCheck(found,PETSC_COMM_SELF,PETSC_ERR_SUP,"No corresponding parallel (MPI) type for this matrix");
  PetscFunctionReturn(0);
}

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

.seealso: `PCSetType()`, `VecSetType()`, `MatCreate()`, `MatType`, `Mat`
@*/
PetscErrorCode  MatSetType(Mat mat, MatType matype)
{
  PetscBool      sametype,found,subclass = PETSC_FALSE;
  MatRootName    names = MatRootNameList;
  PetscErrorCode (*r)(Mat);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);

  while (names) {
    PetscCall(PetscStrcmp(matype,names->rname,&found));
    if (found) {
      PetscMPIInt size;
      PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)mat),&size));
      if (size == 1) matype = names->sname;
      else matype = names->mname;
      break;
    }
    names = names->next;
  }

  PetscCall(PetscObjectTypeCompare((PetscObject)mat,matype,&sametype));
  if (sametype) PetscFunctionReturn(0);

  PetscCall(PetscFunctionListFind(MatList,matype,&r));
  PetscCheck(r,PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown Mat type given: %s",matype);

  if (mat->assembled && ((PetscObject)mat)->type_name) PetscCall(PetscStrbeginswith(matype,((PetscObject)mat)->type_name,&subclass));
  if (subclass) {
    PetscCall(MatConvert(mat,matype,MAT_INPLACE_MATRIX,&mat));
    PetscFunctionReturn(0);
  }
  if (mat->ops->destroy) {
    /* free the old data structure if it existed */
    PetscCall((*mat->ops->destroy)(mat));
    mat->ops->destroy = NULL;

    /* should these null spaces be removed? */
    PetscCall(MatNullSpaceDestroy(&mat->nullsp));
    PetscCall(MatNullSpaceDestroy(&mat->nearnullsp));
  }
  PetscCall(PetscMemzero(mat->ops,sizeof(struct _MatOps)));
  mat->preallocated  = PETSC_FALSE;
  mat->assembled     = PETSC_FALSE;
  mat->was_assembled = PETSC_FALSE;

  /*
   Increment, rather than reset these: the object is logically the same, so its logging and
   state is inherited.  Furthermore, resetting makes it possible for the same state to be
   obtained with a different structure, confusing the PC.
  */
  mat->nonzerostate++;
  PetscCall(PetscObjectStateIncrease((PetscObject)mat));

  /* create the new data structure */
  PetscCall((*r)(mat));
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

.seealso: `MatSetType()`
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

.seealso: `MatSetVecType()`
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

.seealso: `VecSetType()`, `MatGetVecType()`
@*/
PetscErrorCode MatSetVecType(Mat mat,VecType vtype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscCall(PetscFree(mat->defaultvectype));
  PetscCall(PetscStrallocpy(vtype,&mat->defaultvectype));
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

.seealso: `MatRegisterAll()`

  Level: advanced
@*/
PetscErrorCode  MatRegister(const char sname[],PetscErrorCode (*function)(Mat))
{
  PetscFunctionBegin;
  PetscCall(MatInitializePackage());
  PetscCall(PetscFunctionListAdd(&MatList,sname,function));
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

.seealso: `PetscObjectBaseTypeCompare()`

@*/
PetscErrorCode  MatRegisterRootName(const char rname[],const char sname[],const char mname[])
{
  MatRootName    names;

  PetscFunctionBegin;
  PetscCall(PetscNew(&names));
  PetscCall(PetscStrallocpy(rname,&names->rname));
  PetscCall(PetscStrallocpy(sname,&names->sname));
  PetscCall(PetscStrallocpy(mname,&names->mname));
  if (!MatRootNameList) {
    MatRootNameList = names;
  } else {
    MatRootName next = MatRootNameList;
    while (next->next) next = next->next;
    next->next = names;
  }
  PetscFunctionReturn(0);
}
