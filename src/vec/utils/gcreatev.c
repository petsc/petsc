/*$Id: gcreatev.c,v 1.89 2001/08/07 03:02:17 balay Exp $*/

#include "petscsys.h"
#include "petsc.h"
#include "petscis.h"
#include "petscvec.h"    /*I "petscvec.h" I*/


#include "src/vec/vecimpl.h"
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
PetscFList      VecList = 0;
PetscTruth VecRegisterAllCalled = PETSC_FALSE;
 
#undef __FUNCT__  
#define __FUNCT__ "VecRegisterDestroy"
/*@C
   VecRegisterDestroy - Frees the list of Vec methods that were
   registered by VecRegisterDynamic().

   Not Collective

   Level: advanced

.seealso: VecRegisterDynamic(), VecRegisterAll()
@*/
int VecRegisterDestroy(void)
{
  int ierr;

  PetscFunctionBegin;
  if (VecList) {
    ierr = PetscFListDestroy(&VecList);CHKERRQ(ierr);
    VecList = 0;
  }
  VecRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*MC
   VecRegisterDynamic - Adds a new vector component implementation

   Synopsis:
   int VecRegisterDynamic(char *name_solver,char *path,char *name_create,
               int (*routine_create)(MPI_Comm,int,int,Vec*))

   Not Collective

   Input Parameters:
+  name_solver - name of a new user-defined vector object
.  path - path (either absolute or relative) the library containing this vector object
.  name_create - name of routine to create vector
-  routine_create - routine to create vector

   Notes:
   VecRegisterDynamic() may be called multiple times to add several user-defined vectors

   If dynamic libraries are used, then the fourth input argument (routine_create)
   is ignored.

   Sample usage:
.vb
   VecRegisterDynamic("my_solver","/home/username/my_lib/lib/libO/solaris/libmine",
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

   ${PETSC_ARCH}, ${PETSC_DIR}, ${PETSC_LIB_DIR}, ${BOPT}, or ${any environmental variable}
  occuring in pathname will be replaced with appropriate values.

   Level: advanced

   Concepts: vector^adding new type

.seealso: VecRegisterAll(), VecRegisterDestroy()
M*/

#undef __FUNCT__  
#define __FUNCT__ "VecRegister"
int VecRegister(const char sname[],const char path[],const char name[],int (*function)(Vec))
{
  int  ierr;
  char fullname[256];

  PetscFunctionBegin;
  ierr = PetscFListConcat(path,name,fullname);CHKERRQ(ierr);
  ierr = PetscFListAdd(&VecList,sname,fullname,(void (*)())function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecSetType"
/*@C
    VecSetType - Builds a vector, for a particular vector implementation.

    Collective on Vec

    Input Parameters:
+   vec - the vector object
-   type_name - name of the vector type

    Options Database Key:
.  -vec_type <type> - Sets the vector type; use -help for a list of available types

    Notes:
    See "petsc/include/petscvec.h" for available vector types (for instance,
    VECSEQ, VECMPI, or VECSHARED).

     Use VecDuplicate() or VecDuplicateVecs() to form additional vectors
    of the same type as an existing vector.

    Level: intermediate

.seealso: VecCreate()
@*/
int VecSetType(Vec vec,VecType type_name)
{
  int        ierr,(*r)(Vec);
  PetscTruth match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec,VEC_COOKIE);
  PetscValidCharPointer(type_name);

  ierr = PetscTypeCompare((PetscObject)vec,type_name,&match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  /* Get the function pointers for the vector requested */
  if (!VecRegisterAllCalled) {ierr = VecRegisterAll(PETSC_NULL);CHKERRQ(ierr);}

  ierr =  PetscFListFind(vec->comm,VecList,type_name,(void (**)(void)) &r);CHKERRQ(ierr);

  if (!r) SETERRQ1(1,"Unknown vector type given: %s",type_name);

  if (vec->ops->destroy) {
    ierr = (*vec->ops->destroy)(vec);CHKERRQ(ierr);
  }

  ierr = (*r)(vec);CHKERRQ(ierr);

  ierr = PetscObjectChangeTypeName((PetscObject)vec,type_name);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_MATLAB_ENGINE) && !defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_SINGLE)
#include "engine.h"   /* Matlab include file */
#include "mex.h"      /* Matlab include file */
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "VecMatlabEnginePut_Default"
int VecMatlabEnginePut_Default(PetscObject obj,void *engine)
{
  int         ierr,n;
  Vec         vec = (Vec)obj;
  PetscScalar *array;
  mxArray     *mat;

  PetscFunctionBegin;
  ierr = VecGetArray(vec,&array);CHKERRQ(ierr);
  ierr = VecGetLocalSize(vec,&n);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  mat  = mxCreateDoubleMatrix(n,1,mxREAL);
#else
  mat  = mxCreateDoubleMatrix(n,1,mxCOMPLEX);
#endif
  ierr = PetscMemcpy(mxGetPr(mat),array,n*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscObjectName(obj);CHKERRQ(ierr);
  mxSetName(mat,obj->name);
  engPutArray((Engine *)engine,mat);
  
  ierr = VecRestoreArray(vec,&array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "VecMatlabEngineGet_Default"
int VecMatlabEngineGet_Default(PetscObject obj,void *engine)
{
  int         ierr,n;
  Vec         vec = (Vec)obj;
  PetscScalar *array;
  mxArray     *mat;

  PetscFunctionBegin;
  ierr = VecGetArray(vec,&array);CHKERRQ(ierr);
  ierr = VecGetLocalSize(vec,&n);CHKERRQ(ierr);
  mat  = engGetArray((Engine *)engine,obj->name);
  if (!mat) SETERRQ1(1,"Unable to get object %s from matlab",obj->name);
  ierr = PetscMemcpy(array,mxGetPr(mat),n*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = VecRestoreArray(vec,&array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
#endif



