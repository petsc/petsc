#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: gcreatev.c,v 1.62 1999/06/30 23:50:17 balay Exp bsmith $";
#endif

#include "sys.h"
#include "petsc.h"
#include "is.h"
#include "vec.h"    /*I "vec.h" I*/


#include "src/vec/vecimpl.h"
#undef __FUNC__  
#define __FUNC__ "VecGetType"
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
int VecGetType(Vec vec,VecType *type)
{
  PetscFunctionBegin;
  *type = vec->type_name;
  PetscFunctionReturn(0);
}

/*
   Contains the list of registered Vec routines
*/
FList VecList = 0;
int    VecRegisterAllCalled = 0;
 
#undef __FUNC__  
#define __FUNC__ "VecRegisterDestroy"
/*@C
   VecRegisterDestroy - Frees the list of Vec methods that were
   registered by VecRegister().

   Not Collective

   Level: advanced

.keywords: Vec, register, destroy

.seealso: VecRegister(), VecRegisterAll()
@*/
int VecRegisterDestroy(void)
{
  int ierr;

  PetscFunctionBegin;
  if (VecList) {
    ierr = FListDestroy( VecList );CHKERRQ(ierr);
    VecList = 0;
  }
  VecRegisterAllCalled = 0;
  PetscFunctionReturn(0);
}

/*MC
   VecRegister - Adds a new vector component implementation

   Synopsis:
   VecRegister(char *name_solver,char *path,char *name_create,
               int (*routine_create)(MPI_Comm,int,int,Vec*))

   Not Collective

   Input Parameters:
+  name_solver - name of a new user-defined vector object
.  path - path (either absolute or relative) the library containing this vector object
.  name_create - name of routine to create vector
-  routine_create - routine to create vector

   Notes:
   VecRegister() may be called multiple times to add several user-defined vectors

   If dynamic libraries are used, then the fourth input argument (routine_create)
   is ignored.

   Sample usage:
.vb
   VecRegister("my_solver","/home/username/my_lib/lib/libO/solaris/libmine",
               "MyVectorCreate",MyVectorCreate);
.ve

   Then, your solver can be chosen with the procedural interface via
.vb
      VecCreate(MPI_Comm,int n,int N,Vec *);
      VecSetType(Vec,"my_vector_name");
.ve
   or at runtime via the option
.vb
      -vec_type my_vector_name
.ve

   $PETSC_ARCH, $PETSC_DIR, $PETSC_LDIR, and $BOPT occuring in pathname will be replaced with appropriate values.

   Level: advanced

.keywords: Vec, register

.seealso: VecRegisterAll(), VecRegisterDestroy()
M*/

#undef __FUNC__  
#define __FUNC__ "VecRegister_Private"
int VecRegister_Private(const char sname[],const char path[],const char name[],
                        int (*function)(Vec))
{
  int  ierr;
  char fullname[256];

  PetscFunctionBegin;
  ierr = PetscStrcpy(fullname,path);CHKERRQ(ierr);
  PetscStrcat(fullname,":");PetscStrcat(fullname,name);
  ierr = FListAdd_Private(&VecList,sname,fullname,(int (*)(void*))function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNC__  
#define __FUNC__ "VecSetType"
/*@C
    VecSetType - Builds a vector, for a particular vector implementation.

    Collective on Vec

    Input Parameters:
+   vec - the vector object
-   type_name - name of the vector type

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
int VecSetType(Vec vec,VecType type_name)
{
  int  ierr,(*r)(Vec);

  PetscFunctionBegin;
  if (PetscTypeCompare(vec->type_name,type_name)) PetscFunctionReturn(0);

  /* Get the function pointers for the vector requested */
  if (!VecRegisterAllCalled) {ierr = VecRegisterAll(PETSC_NULL);CHKERRQ(ierr);}

  ierr =  FListFind(vec->comm, VecList, type_name,(int (**)(void *)) &r );CHKERRQ(ierr);

  if (!r) SETERRQ1(1,1,"Unknown vector type given: %s",type_name);

  if (vec->ops->destroy) {
    ierr = (*vec->ops->destroy)(vec);CHKERRQ(ierr);
  }

  ierr = (*r)(vec);CHKERRQ(ierr);

  ierr = PetscObjectChangeTypeName((PetscObject)vec,type_name);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


