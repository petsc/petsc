#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: matreg.c,v 1.8 2000/09/25 20:50:24 bsmith Exp bsmith $";
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
FList MatList = 0;

#undef __FUNC__  
#define __FUNC__ "MatSetType"
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

.keywords: Mat, set, method

.seealso: PCSetType(), VecSetType(), MatCreate()
@*/
int MatSetType(Mat mat,MATType matype)
{
  int        ierr,(*r)(Mat);
  PetscTruth sametype;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);

  ierr = PetscTypeCompare((PetscObject)mat,matype,&sametype);CHKERRQ(ierr);
  if (sametype) PetscFunctionReturn(0);

  /* Get the function pointers for the matrix requested */
  if (!MatRegisterAllCalled) {ierr = MatRegisterAll(PETSC_NULL);CHKERRQ(ierr);}

  ierr =  FListFind(mat->comm,MatList,matype,(int(**)(void*))&r);CHKERRQ(ierr);

  if (!r) SETERRQ1(1,1,"Unknown Mat type given: %s",matype);

  mat->data        = 0;
  ierr = (*r)(mat);CHKERRQ(ierr);

  ierr = PetscObjectChangeTypeName((PetscObject)mat,matype);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatRegisterDestroy"
/*@C
   MatRegisterDestroy - Frees the list of matrix types that were
   registered by MatRegister().

   Not Collective

   Level: advanced

.keywords: Mat, register, destroy

.seealso: MatRegister(), MatRegisterAll()
@*/
int MatRegisterDestroy(void)
{
  int ierr;

  PetscFunctionBegin;
  if (MatList) {
    ierr = FListDestroy(&MatList);CHKERRQ(ierr);
    MatList = 0;
  }
  MatRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetType"
/*@C
   MatGetType - Gets the matrx type as a string from the matrix object.

   Not Collective

   Input Parameter:
.  mat - the matrix

   Output Parameter:
.  name - name of matrix type

   Level: intermediate

.keywords: Mat, get, method, name

.seealso: MatSetType()
@*/
int MATGetType(Mat mat,MATType *type)
{
  PetscFunctionBegin;
  *type = mat->type_name;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatSetTypeFromOptions"
/*@
   MatSetTypeFromOptions - Sets Mat type from the options database, if not
       given then sets default.

   Collective on Mat

   Input Parameters:
.  Mat - the Krylov space context

   Level: developer

.keywords: Mat, set, from, options, database

.seealso: MatSetFromOptions(), SLESSetFromOptions()
@*/
int MatSetTypeFromOptions(Mat mat)
{
  int        ierr;
  char       method[256];
  PetscTruth flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);

  ierr = OptionsGetString(mat->prefix,"-mat_type",method,256,&flg);
  if (flg){
    ierr = MatSetType(mat,method);CHKERRQ(ierr);
  }
  /*
    Set the type if it was never set.
  */
  if (!mat->type_name) {
    ierr = MatSetType(mat,"mpiaij");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*MC
   MatRegisterDynamic - Adds a new matrix type

   Synopsis:
   MatRegisterDynamic(char *name,char *path,char *name_create,int (*routine_create)(Mat))

   Not Collective

   Input Parameters:
+  name - name of a new user-defined matrix type
.  path - path (either absolute or relative) the library containing this solver
.  name_create - name of routine to create method context
-  routine_create - routine to create method context

   Notes:
   MatRegister() may be called multiple times to add several user-defined solvers.

   If dynamic libraries are used, then the fourth input argument (routine_create)
   is ignored.

   Sample usage:
.vb
   MatRegisterDynamic("my_mat",/home/username/my_lib/lib/libO/solaris/mylib.a,
               "MyMatCreate",MyMatCreate);
.ve

   Then, your solver can be chosen with the procedural interface via
$     MatSetType(Mat,"my_mat")
   or at runtime via the option
$     -mat_type my_mat

   Level: advanced

   ${PETSC_ARCH} and ${BOPT} occuring in pathname will be replaced with appropriate values.

.keywords: Mat, register

.seealso: MatRegisterAll(), MatRegisterDestroy(), MatRegister()

M*/

#undef __FUNC__  
#define __FUNC__ "MatRegister"
int MatRegister(char *sname,char *path,char *name,int (*function)(Mat))
{
  int  ierr;
  char fullname[256];

  PetscFunctionBegin;
  ierr = FListConcat(path,name,fullname);CHKERRQ(ierr);
  ierr = FListAdd(&MatList,sname,fullname,(int (*)(void*))function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MATCreate"
int MATCreate(MPI_Comm comm,int m,int n,int M,int N,Mat *A)
{
  Mat B;

  PetscFunctionBegin;
  PetscHeaderCreate(B,_p_Mat,struct _MatOps,MAT_COOKIE,0,"Mat",comm,MatDestroy,MatView);
  PLogObjectCreate(B);

  B->m = m;
  B->n = n;
  B->M = M;
  B->N = N;

  *A = B;
  PetscFunctionReturn(0);
}
