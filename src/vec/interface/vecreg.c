#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: vecreg.c,v 1.5 2000/01/10 03:18:14 knepley Exp $";
#endif

#include "src/vec/vecimpl.h"    /*I "vec.h"  I*/

PetscFList VecList                       = 0;
int        VecRegisterAllCalled          = 0;
PetscFList VecSerializeList              = 0;
int        VecSerializeRegisterAllCalled = 0;

#undef __FUNCT__  
#define __FUNCT__ "VecSetType"
/*@C
    VecSetType - Builds a vector, for a particular vector implementation.

    Collective on Vec

    Input Parameters:
+   vec    - the vector object
-   method - name of the vector type

    Options Database Key:
.  -vec_type <type> - Sets the vector type; use -help for a list 
    of available types

    Notes:
    See "petsc/include/vec.h" for available vector types (for instance,
    VEC_SEQ, VEC_MPI, or VEC_SHARED).

     Use VecDuplicate() or VecDuplicateVecs() to form additional vectors
    of the same type as an existing vector.

    Level: intermediate

.keywords: vector, set, type
.seealso: VecCreate()
@*/
int VecSetType(Vec vec, VecType method)
{
  int      (*r)(Vec, ParameterDict);
  PetscTruth match;
  int        ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec, VEC_COOKIE);
  ierr = PetscTypeCompare((PetscObject) vec, method, &match);                                             CHKERRQ(ierr);
  if (match == PETSC_TRUE) PetscFunctionReturn(0);

  /* Get the function pointers for the vector requested */
  if (!VecRegisterAllCalled) {
    ierr = VecRegisterAll(PETSC_NULL);                                                                    CHKERRQ(ierr);
  }
  ierr = PetscFListFind(vec->comm, VecList, method,(void (**)()) &r);                                     CHKERRQ(ierr);
  if (!r) SETERRQ1(PETSC_ERR_ARG_WRONG, "Unknown vector type: %s", method);

  if (vec->ops->destroy != PETSC_NULL) {
    ierr = (*vec->ops->destroy)(vec);                                                                     CHKERRQ(ierr);
  }
  ierr = (*r)(vec, vec->dict);                                                                            CHKERRQ(ierr);

  ierr = PetscObjectChangeTypeName((PetscObject) vec, method);                                            CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecGetType"
/*@C
   VecGetType - Gets the vector type name (as a string) from the vector.

   Not Collective

   Input Parameter:
.  vec - the vector

   Output Parameter:
.  type - the vector type name

   Level: intermediate

.keywords: vector, get, type, name
.seealso: VecSetType()
@*/
int VecGetType(Vec vec, VecType *type)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec, VEC_COOKIE);
  PetscValidPointer(type);
  if (!VecRegisterAllCalled) {
    ierr = VecRegisterAll(PETSC_NULL);                                                                    CHKERRQ(ierr);
  }
  *type = vec->type_name;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecSetSerializeType"
/*@C
  VecSetSerializeType - Sets the serialization method for the vector.

  Collective on Vec

  Input Parameters:
+ vec    - The Vec context
- method - A known method

  Options Database Command:
. -vec_serialize_type <method> - Sets the method; use -help for a list
                                 of available methods (for instance, seq_binary)

   Notes:
   See "petsc/include/vec.h" for available methods (for instance)
+  VEC_SER_SEQ_BINARY - Sequential vector to binary file
-  VEC_SER_MPI_BINARY - MPI vector to binary file

   Normally, it is best to use the VecSetFromOptions() command and
   then set the Vec type from the options database rather than by using
   this routine.  Using the options database provides the user with
   maximum flexibility in evaluating the many different solvers.
   The VecSetSerializeType() routine is provided for those situations
   where it is necessary to set the application ordering independently of the
   command line or options database.  This might be the case, for example,
   when the choice of solver changes during the execution of the
   program, and the user's application is taking responsibility for
   choosing the appropriate method.  In other words, this routine is
   not for beginners.

   Level: intermediate

.keywords: Vec, set, type, serialization
@*/
int VecSetSerializeType(Vec vec, VecSerializeType method)
{
  int      (*r)(MPI_Comm, Vec *, PetscViewer, PetscTruth);
  PetscTruth match;
  int        ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec, VEC_COOKIE);
  ierr = PetscSerializeCompare((PetscObject) vec, method, &match);                                        CHKERRQ(ierr);
  if (match == PETSC_TRUE) PetscFunctionReturn(0);

  /* Get the function pointers for the method requested but do not call */
  if (!VecSerializeRegisterAllCalled) {
    ierr = VecSerializeRegisterAll(PETSC_NULL);                                                           CHKERRQ(ierr);
  }
  ierr = PetscFListFind(vec->comm, VecSerializeList, method, (void (**)()) &r);                           CHKERRQ(ierr);
  if (!r) SETERRQ1(PETSC_ERR_ARG_WRONG, "Unknown vector serialization type: %s", method);

  ierr = PetscObjectChangeSerializeName((PetscObject) vec, method);                                       CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*--------------------------------------------------------------------------------------------------------------------*/
/*@C
  VecRegister - Adds a new vector component implementation

  Synopsis:
  VecRegister(char *name, char *path, char *func_name, int (*create_func)(Vec, ParameterDict))

  Not Collective

  Input Parameters:
+ name        - The name of a new user-defined creation routine
. path        - The path (either absolute or relative) of the library containing this routine
. func_name   - The name of routine to create method context
- create_func - The creation routine itself

  Notes:
  VecRegister() may be called multiple times to add several user-defined vectors

  If dynamic libraries are used, then the fourth input argument (routine_create) is ignored.

  Sample usage:
.vb
    VecRegisterDynamic("my_vec","/home/username/my_lib/lib/libO/solaris/libmy.a", "MyVectorCreate", MyVectorCreate);
.ve

  Then, your vector type can be chosen with the procedural interface via
.vb
    VecCreate(MPI_Comm, Vec *);
    VecSetType(Vec,"my_vector_name");
.ve
   or at runtime via the option
.vb
    -vec_type my_vector_name
.ve

  Note: $PETSC_ARCH and $BOPT occuring in pathname will be replaced with appropriate values.

  Level: advanced

.keywords: Vec, register
.seealso: VecRegisterAll(), VecRegisterDestroy()
M*/

#undef __FUNCT__  
#define __FUNCT__ "VecRegister_Private"
int VecRegister(const char sname[], const char path[], const char name[], int (*function)(Vec, ParameterDict))
{
  char fullname[256];
  int  ierr;

  PetscFunctionBegin;
  ierr = PetscStrcpy(fullname, path);                                                                     CHKERRQ(ierr);
  ierr = PetscStrcat(fullname, ":");                                                                      CHKERRQ(ierr);
  ierr = PetscStrcat(fullname, name);                                                                     CHKERRQ(ierr);
  ierr = PetscFListAdd(&VecList, sname, fullname, (void (*)()) function);                                 CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  VecSerializeRegister - Adds a serialization method to the vec package.

  Synopsis:

  VecSerializeRegister(char *name, char *path, char *func_name,
                        int (*serialize_func)(MPI_Comm, Vec *, PetscViewer, PetscTruth))

  Not Collective

  Input Parameters:
+ name           - The name of a new user-defined serialization routine
. path           - The path (either absolute or relative) of the library containing this routine
. func_name      - The name of the serialization routine
- serialize_func - The serialization routine itself

  Notes:
  VecSerializeRegister() may be called multiple times to add several user-defined serializers.

  If dynamic libraries are used, then the fourth input argument (serialize_func) is ignored.

  Sample usage:
.vb
  VecSerializeRegisterDynamic("my_store", "/home/username/my_lib/lib/libO/solaris/libmy.a", "MyStoreFunc", MyStoreFunc);
.ve

  Then, your serialization can be chosen with the procedural interface via
.vb
    VecSetSerializeType(vec, "my_store")
.ve
  or at runtime via the option
.vb
    -vec_serialize_type my_store
.ve

  Note: $PETSC_ARCH and $BOPT occuring in pathname will be replaced with appropriate values.

  Level: advanced

.keywords: Vec, register
.seealso: VecSerializeRegisterAll(), VecSerializeRegisterDestroy()
M*/
#undef __FUNCT__  
#define __FUNCT__ "VecSerializeRegister"
int VecSerializeRegister(const char sname[], const char path[], const char name[],
                          int (*function)(MPI_Comm, Vec *, PetscViewer, PetscTruth))
{
  char fullname[256];
  int  ierr;

  PetscFunctionBegin;
  ierr = PetscStrcpy(fullname, path);                                                                     CHKERRQ(ierr);
  ierr = PetscStrcat(fullname, ":");                                                                      CHKERRQ(ierr);
  ierr = PetscStrcat(fullname, name);                                                                     CHKERRQ(ierr);
  ierr = PetscFListAdd(&VecSerializeList, sname, fullname, (void (*)()) function);                        CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*--------------------------------------------------------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "VecRegisterDestroy"
/*@C
   VecRegisterDestroy - Frees the list of Vec methods that were
   registered by VecRegister().

   Not Collective

   Level: advanced

.keywords: Vec, register, destroy
.seealso: VecRegister(), VecRegisterAll(), VecSerializeRegisterDestroy()
@*/
int VecRegisterDestroy(void)
{
  int ierr;

  PetscFunctionBegin;
  if (VecList != PETSC_NULL) {
    ierr = PetscFListDestroy(&VecList);                                                                   CHKERRQ(ierr);
    VecList = PETSC_NULL;
  }
  VecRegisterAllCalled = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecSerializeRegisterDestroy"
/*@C
  VecSerializeRegisterDestroy - Frees the list of serialization routines for
  vectors that were registered by VecSerializeRegister().

  Not collective

  Level: advanced

.keywords: Vec, vector, register, destroy
.seealso: VecSerializeRegisterAll()
@*/
int VecSerializeRegisterDestroy()
{
  int ierr;

  PetscFunctionBegin;
  if (VecSerializeList) {
    ierr = PetscFListDestroy(&VecSerializeList);                                                          CHKERRQ(ierr);
    VecSerializeList = PETSC_NULL;
  }
  VecSerializeRegisterAllCalled = 0;
  PetscFunctionReturn(0);
}
