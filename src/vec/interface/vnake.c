#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: vnake.c,v 1.9 1999/05/04 20:30:38 balay Exp balay $";
#endif

#include "src/vec/vecimpl.h"    /*I "vec.h" I*/

#undef __FUNC__  
#define __FUNC__ "VecCreate"
/*@C
   VecCreate - Creates an empty vector object. The type can then
   be set with VecSetType().

   Collective on MPI_Comm

   Input Parameter:
+  comm - the communicator
.  n - the local vector length 
-  N - total vector length

   Output Parameter:
.  V - the vector

   Notes:
   You MUST call either VecSetFromOptions() or VecSetType() after this call before the
   vector may be used.

   Use VecDuplicate() or VecDuplicateVecs() to form additional vectors of the
   same type as an existing vector.

   Level: beginner

.keywords: vector, sequential, create, BLAS

.seealso: VecCreateMPIWithArray(), VecCreateMPI(), VecDuplicate(), VecDuplicateVecs(), 
          VecCreateGhost(), VecCreateSeq(), VecPlaceArray(), VecSetType()
@*/
int VecCreate(MPI_Comm comm,int n,int N,Vec *V)
{
  Vec     v;
  int     ierr;

  PetscFunctionBegin;
  *V             = 0;

  PetscHeaderCreate(v,_p_Vec,struct _VecOps,VEC_COOKIE,0,"Vec",comm,VecDestroy,VecView);
  PLogObjectCreate(v);
  PLogObjectMemory(v,sizeof(struct _p_Vec));

  ierr = PetscMemzero(v->ops,sizeof(struct _VecOps));CHKERRQ(ierr);
  v->n               = n; 
  v->N               = N;
  v->map             = 0;
  v->mapping         = 0;
  v->bmapping        = 0;
  v->bs              = 0;
  v->type_name       = 0;

  *V = v; 
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "VecSetFromOptions"
/*@C
   VecSetFromOptions - Sets the vector type from the options database.
   Defaults to a PETSc sequential vector on one processor and a
   PETSc MPI vector on more than one processor.

   Collective on Vec

   Input Parameter:
.  vec - the vector

   Notes: 
   Must be called after VecCreate() but before the vector is used.

   Level: developer

.keywords: vector, set, from, options

.seealso: VecCreate()
@*/
int VecSetFromOptions(Vec vec)
{
  int     ierr,size,flg;
  char    vtype[256];

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec,VEC_COOKIE);

  if (!VecRegisterAllCalled) {ierr = VecRegisterAll(PETSC_NULL);CHKERRQ(ierr);}
  ierr = OptionsGetString(vec->prefix,"-vec_type",vtype,256,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = VecSetType(vec,vtype);CHKERRQ(ierr);
  }

  /* type has not been set? */
  if (!vec->type_name) {
    ierr = MPI_Comm_size(vec->comm,&size);CHKERRQ(ierr);
    if (size > 1) {
      ierr = VecSetType(vec,"PETSc#VecMPI");CHKERRQ(ierr);
    } else {
      ierr = VecSetType(vec,"PETSc#VecSeq");CHKERRQ(ierr);
    }
  }

  PetscFunctionReturn(0);
}
