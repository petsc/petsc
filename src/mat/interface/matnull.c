#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: pcnull.c,v 1.23 1999/04/16 16:08:03 bsmith Exp balay $";
#endif
/*
    Routines to project vectors out of null spaces.
*/

#include "petsc.h"
#include "src/sles/pc/pcimpl.h"      /*I "pc.h" I*/
#include "sys.h"


#undef __FUNC__  
#define __FUNC__ "PCNullSpaceCreate"
/*@C
   PCNullSpaceCreate - Creates a data-structure used to project vectors 
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

.seealso: PCNullSpaceDestroy(), PCNullSpaceRemove()
@*/
int PCNullSpaceCreate(MPI_Comm comm, int has_cnst, int n, Vec *vecs,PCNullSpace *SP)
{
  PCNullSpace sp;

  PetscFunctionBegin;
  PetscHeaderCreate(sp,_p_PCNullSpace,int,PCNULLSPACE_COOKIE,0,"PCNullSpace",comm,PCNullSpaceDestroy,0);
  PLogObjectCreate(sp);
  PLogObjectMemory(sp,sizeof(struct _p_PCNullSpace));

  sp->has_cnst = has_cnst; 
  sp->n        = n;
  sp->vecs     = vecs;

  *SP          = sp;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCNullSpaceDestroy"
/*@
   PCNullSpaceDestroy - Destroys a data-structure used to project vectors 
   out of null spaces.

   Collective on PCNullSpace

   Input Parameter:
.  sp - the null space context to be destroyed

   Level: advanced

.keywords: PC, null space, destroy

.seealso: PCNullSpaceDestroy(), PCNullSpaceRemove()
@*/
int PCNullSpaceDestroy(PCNullSpace sp)
{
  PetscFunctionBegin;
  PLogObjectDestroy(sp);
  PetscHeaderDestroy(sp);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCNullSpaceRemove"
/*@
   PCNullSpaceRemove - Removes all the components of a null space from a vector.

   Collective on PCNullSpace

   Input Parameters:
+  sp - the null space context
-  vec - the vector from which the null space is to be removed 

   Level: advanced

.keywords: PC, null space, remove

.seealso: PCNullSpaceCreate(), PCNullSpaceDestroy()
@*/
int PCNullSpaceRemove(PCNullSpace sp,Vec vec)
{
  Scalar sum;
  int    j, n = sp->n, N,ierr;

  PetscFunctionBegin;
  if (sp->has_cnst) {
    ierr = VecSum(vec,&sum);CHKERRQ(ierr);
    ierr = VecGetSize(vec,&N);CHKERRQ(ierr);
    sum  = sum/(-1.0*N);
    ierr = VecShift(&sum,vec);CHKERRQ(ierr);
  }

  for ( j=0; j<n; j++ ) {
    ierr = VecDot(vec,sp->vecs[j],&sum);CHKERRQ(ierr);
    sum  = -sum;
    ierr = VecAYPX(&sum,sp->vecs[j],vec);CHKERRQ(ierr);
  }
  
  PetscFunctionReturn(0);
}
