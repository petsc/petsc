/*$Id: pcnull.c,v 1.31 2000/05/05 22:16:59 balay Exp bsmith $*/
/*
    Routines to project vectors out of null spaces.
*/

#include "src/mat/matimpl.h"      /*I "petscmat.h" I*/
#include "petscsys.h"

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatNullSpaceCreate"
/*@C
   MatNullSpaceCreate - Creates a data structure used to project vectors 
   out of null spaces.

   Collective on MPI_Comm

   Input Parameters:
+  comm - the MPI communicator associated with the object
.  has_cnst - PETSC_TRUE if the null space contains the constant vector; otherwise PETSC_FALSE
.  n - number of vectors (excluding constant vector) in null space
-  vecs - the vectors that span the null space (excluding the constant vector);
          these vectors must be orthonormal

   Output Parameter:
.  SP - the null space context

   Level: advanced

.keywords: PC, null space, create

.seealso: MatNullSpaceDestroy(), MatNullSpaceRemove()
@*/
int MatNullSpaceCreate(MPI_Comm comm,int has_cnst,int n,Vec *vecs,MatNullSpace *SP)
{
  MatNullSpace sp;

  PetscFunctionBegin;
  PetscHeaderCreate(sp,_p_MatNullSpace,int,MATNULLSPACE_COOKIE,0,"MatNullSpace",comm,MatNullSpaceDestroy,0);
  PLogObjectCreate(sp);
  PLogObjectMemory(sp,sizeof(struct _p_MatNullSpace));

  sp->has_cnst = has_cnst; 
  sp->n        = n;
  sp->vecs     = vecs;
  sp->vec      = PETSC_NULL;

  *SP          = sp;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatNullSpaceDestroy"
/*@
   MatNullSpaceDestroy - Destroys a data structure used to project vectors 
   out of null spaces.

   Collective on MatNullSpace

   Input Parameter:
.  sp - the null space context to be destroyed

   Level: advanced

.keywords: PC, null space, destroy

.seealso: MatNullSpaceCreate(), MatNullSpaceRemove()
@*/
int MatNullSpaceDestroy(MatNullSpace sp)
{
  int ierr;

  PetscFunctionBegin;
  if (--sp->refct > 0) PetscFunctionReturn(0);

  if (sp->vec) {ierr = VecDestroy(sp->vec);CHKERRQ(ierr);}

  PLogObjectDestroy(sp);
  PetscHeaderDestroy(sp);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatNullSpaceRemove"
/*@
   MatNullSpaceRemove - Removes all the components of a null space from a vector.

   Collective on MatNullSpace

   Input Parameters:
+  sp - the null space context
-  vec - the vector from which the null space is to be removed 

   Level: advanced

.keywords: PC, null space, remove

.seealso: MatNullSpaceCreate(), MatNullSpaceDestroy()
@*/
int MatNullSpaceRemove(MatNullSpace sp,Vec vec,Vec *out)
{
  Scalar sum;
  int    j,n = sp->n,N,ierr;
  Vec    l = vec;

  PetscFunctionBegin;
  if (out) {
    if (!sp->vec) {
      ierr = VecDuplicate(vec,&sp->vec);CHKERRQ(ierr);
    }
    *out = sp->vec;
    ierr = VecCopy(vec,*out);CHKERRQ(ierr);
    l    = *out;
  }

  if (sp->has_cnst) {
    ierr = VecSum(l,&sum);CHKERRQ(ierr);
    ierr = VecGetSize(l,&N);CHKERRQ(ierr);
    sum  = sum/(-1.0*N);
    ierr = VecShift(&sum,l);CHKERRQ(ierr);
  }

  for (j=0; j<n; j++) {
    ierr = VecDot(l,sp->vecs[j],&sum);CHKERRQ(ierr);
    sum  = -sum;
    ierr = VecAXPY(&sum,sp->vecs[j],l);CHKERRQ(ierr);
  }
  
  PetscFunctionReturn(0);
}
