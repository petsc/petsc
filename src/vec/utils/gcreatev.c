#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: gcreatev.c,v 1.47 1998/06/10 13:38:15 bsmith Exp bsmith $";
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

.keywords: vector, get, type, name
@*/
int VecGetType(Vec vec,char **type)
{
  PetscFunctionBegin;
  *type = vec->type_name;
  PetscFunctionReturn(0);
}

/*
   Contains the list of registered Vec routines
*/
DLList VecList = 0;
int    VecRegisterAllCalled = 0;
 
#undef __FUNC__  
#define __FUNC__ "VecRegisterDestroy"
/*@C
   VecRegisterDestroy - Frees the list of Vec methods that were
   registered by VecRegister().

   Not Collective

.keywords: Vec, register, destroy

.seealso: VecRegister(), VecRegisterAll()
@*/
int VecRegisterDestroy(void)
{
  int ierr;

  PetscFunctionBegin;
  if (VecList) {
    ierr = DLRegisterDestroy( VecList );CHKERRQ(ierr);
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
   VecRegister("my_solver",/home/username/my_lib/lib/libO/solaris/mylib.a,
               "MyVectorCreate",MyVectorCreate);
.ve

   Then, your solver can be chosen with the procedural interface via
$     VecCreate("my_vector_name",Vec *)
   or at runtime via the option
$     -Vec_type my_vector_name

.keywords: Vec, register

.seealso: VecRegisterAll(), VecRegisterDestroy()
M*/

#undef __FUNC__  
#define __FUNC__ "VecRegister_Private"
int VecRegister_Private(char *sname,char *path,char *name,int (*function)(MPI_Comm,int,int,Vec*))
{
  int ierr;
  char fullname[256];

  PetscFunctionBegin;
  PetscStrcpy(fullname,path); PetscStrcat(fullname,":");PetscStrcat(fullname,name);
  ierr = DLRegister_Private(&VecList,sname,fullname,(int (*)(void*))function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNC__  
#define __FUNC__ "VecCreateWithType"
/*@C
    VecCreateWithType - Creates a vector, where the vector type is determined 
    from type name passed in.  

    Collective on MPI_Comm

    Input Parameters:
+   comm - MPI communicator
.   type_name - name of the vector type (can be overwritten from the command line)
.   n - local vector length (or PETSC_DECIDE)
-   N - global vector length (or PETSC_DETERMINE)
 
    Output Parameter:
.   V - location to stash resulting vector

    Options Database Keys:
+   -vec_type mpi - Activates use of MPI vectors, even for the uniprocessor case
               by internally calling VecCreateMPI()
-   -vec_type shared - Activates use of shared memory parallel vectors
               by internally calling VecCreateShared()

    Notes:
    Use VecDuplicate() or VecDuplicateVecs() to form additional vectors
    of the same type as an existing vector.

.keywords: vector, create, initial

.seealso: VecCreateSeq(), VecCreateMPI(), VecCreateShared(), VecDuplicate(), VecDuplicateVecs(),
          VecCreate()
@*/
int VecCreateWithType(MPI_Comm comm,char *type_name,int n,int N,Vec *v)
{
  int  flg,ierr,(*r)(MPI_Comm,int,int,Vec *),size;
  char vectype[256];

  PetscFunctionBegin;
  /* Get the function pointers for the vector requested */
  if (!VecRegisterAllCalled) {ierr = VecRegisterAll(PETSC_NULL); CHKERRQ(ierr);}

  ierr = OptionsHasName(PETSC_NULL,"-help",&flg); CHKERRQ(ierr);
  if (flg) {
    ierr = DLRegisterPrintTypes(comm,stdout,PETSC_NULL,"vec_type",VecList);CHKERRQ(ierr);
  }
  ierr = OptionsGetString(PETSC_NULL,"-vec_type",vectype,128,&flg); CHKERRQ(ierr);
  if (!flg) {
    PetscStrncpy(vectype,type_name,256);
  }

  ierr =  DLRegisterFind(comm, VecList, vectype,(int (**)(void *)) &r );CHKERRQ(ierr);

  if (!r) SETERRQ(1,1,"Unknown vector type given");

  ierr = (*r)(comm,n,N,v); CHKERRQ(ierr);

  (*v)->type_name = (char *) PetscMalloc((PetscStrlen(vectype)+1)*sizeof(char));CHKPTRQ((*v)->type_name);
  PetscStrcpy((*v)->type_name,vectype);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecCreate"
/*@C
    VecCreate - Creates a vector, where the vector type is determined 
    from the options database.  Generates a parallel MPI vector if the 
    communicator has more than one processor.

    Collective on MPI_Comm

    Input Parameters:
+   comm - MPI communicator
.   n - local vector length (or PETSC_DECIDE)
-   N - global vector length (or PETSC_DETERMINE)
 
    Output Parameter:
.   V - location to stash resulting vector

    Options Database Keys:
+   -vec_type mpi - Activates use of MPI vectors, even for the uniprocessor case
               by internally calling VecCreateMPI()
-   -vec_type shared - Activates use of shared memory parallel vectors
               by internally calling VecCreateShared()

    Notes:
    Use VecDuplicate() or VecDuplicateVecs() to form additional vectors
    of the same type as an existing vector.

.keywords: vector, create, initial

.seealso: VecCreateSeq(), VecCreateMPI(), VecCreateShared(), VecDuplicate(), VecDuplicateVecs(),
          VecCreateWithType()
@*/
int VecCreate(MPI_Comm comm,int n,int N,Vec *v)
{
  int  ierr,size;
  char vectype[16];

  PetscFunctionBegin;

  /* set the default types */
  MPI_Comm_size(comm,&size);
  if (size > 1) {
    PetscStrcpy(vectype,"PETSc#VecMPI");
  } else {
    PetscStrcpy(vectype,"PETSc#VecSeq");
  }

  ierr =  VecCreateWithType(comm,vectype,n,N,v);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

