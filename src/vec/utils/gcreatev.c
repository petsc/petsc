#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: gcreatev.c,v 1.46 1998/05/19 18:50:37 curfman Exp bsmith $";
#endif

#include "sys.h"
#include "petsc.h"
#include "is.h"
#include "vec.h"    /*I "vec.h" I*/

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
+   -vec_mpi - Activates use of MPI vectors, even for the uniprocessor case
               by internally calling VecCreateMPI()
-   -vec_shared - Activates use of shared memory parallel vectors
               by internally calling VecCreateShared()

    Notes:
    Use VecDuplicate() or VecDuplicateVecs() to form additional vectors
    of the same type as an existing vector.

.keywords: vector, create, initial

.seealso: VecCreateSeq(), VecCreateMPI(), VecCreateShared(), VecDuplicate(), VecDuplicateVecs()
@*/
int VecCreate(MPI_Comm comm,int n,int N,Vec *V)
{
  int ierr,size,flg,flgs;

  PetscFunctionBegin;
  MPI_Comm_size(comm,&size);
  ierr = OptionsHasName(PETSC_NULL,"-help",&flg); CHKERRQ(ierr);
  if (flg) {
    (*PetscHelpPrintf)(comm,"VecCreate() option: -vec_mpi\n");
    (*PetscHelpPrintf)(comm,"                    -vec_shared\n");
  }
  ierr = OptionsHasName(PETSC_NULL,"-vec_mpi",&flg); CHKERRQ(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-vec_shared",&flgs); CHKERRQ(ierr);
  if (flgs) {
    ierr = VecCreateShared(comm,n,N,V); CHKERRQ(ierr);
  } else if (size > 1 || flg) {
    ierr = VecCreateMPI(comm,n,N,V); CHKERRQ(ierr);
  } else {
    ierr = VecCreateSeq(comm,PetscMax(n,N),V);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#include "src/vec/vecimpl.h"
#undef __FUNC__  
#define __FUNC__ "VecGetType"
/*@C
   VecGetType - Gets the vector type and name (as a string) from the vector.

   Not Collective

   Input Parameter:
.  mat - the vector

   Output Parameter:
+  type - the vector type (or use PETSC_NULL)
-  name - name of vector type (or use PETSC_NULL)

.keywords: vector, get, type, name
@*/
int VecGetType(Vec vec,VecType *type,char **name)
{
  int  itype = (int)vec->type;
  char *vecname[10];

  PetscFunctionBegin;
  if (type) *type = (VecType) vec->type;
  if (name) {
    /* Note:  Be sure that this list corresponds to the enum in vec.h */
    vecname[0] = "VECSEQ";
    vecname[1] = "VECMPI";
    if (itype < 0 || itype > 1) *name = "Unknown vector type";
    else                        *name = vecname[itype];
  }
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

