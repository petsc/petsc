#ifndef lint
static char vcid[] = "$Id: pcnull.c,v 1.5 1996/04/25 23:59:27 balay Exp bsmith $";
#endif
/*
    Routines to project vectors out of null spaces.
*/

#include "petsc.h"
#include "src/pc/pcimpl.h"      /*I "pc.h" I*/
#include <stdio.h>
#include "src/sys/nreg.h"
#include "sys.h"


/*@C
  PCNullSpaceCreate - Creates a data-structure used to project vectors 
       out of null spaces.

  Input Parameters:
.  comm - the MPI communicator associated with the object.
.  has_cnst - if the null spaces contains the constant vector, PETSC_TRUE or PETSC_FALSE
.  n - number of vectors (excluding constant vector) in null space
.  vecs - the vectors that span the null space (excluding the constant vector)
.         these vectors must be orthonormal

  Output Parameter:
.  SP - the null space context


.keywords: PC, Null space
@*/
int PCNullSpaceCreate(MPI_Comm comm, int has_cnst, int n, Vec *vecs,PCNullSpace *SP)
{
  PCNullSpace sp;

  PetscHeaderCreate(sp,_PCNullSpace,PCNULLSPACE_COOKIE,0,comm);
  PLogObjectCreate(sp);
  PLogObjectMemory(sp,sizeof(struct _PCNullSpace));

  sp->has_cnst = has_cnst; 
  sp->n        = n;
  sp->vecs     = vecs;

  *SP          = sp;
  return 0;
}

/*@
  PCNullSpaceDestroy - Destroys a data-structure used to project vectors 
       out of null spaces.

  Input Parameter:
.    SP - the null space context to be destroyed

.keywords: PC, Null space
@*/
int PCNullSpaceDestroy(PCNullSpace sp)
{
  PLogObjectDestroy(sp);
  PetscHeaderDestroy(sp);
  return 0;
}

/*@
  PCNullSpaceRemove - Removes all the components of a null space from a vector.

  Input Parameters:
.    sp - the null space context
.    vec - the vector you want the null space removed from


.keywords: PC, Null space
@*/
int PCNullSpaceRemove(PCNullSpace sp,Vec vec)
{
  Scalar sum;
  int    j, n = sp->n, N,ierr;

  if (sp->has_cnst) {
    ierr = VecSum(vec,&sum); CHKERRQ(ierr);
    ierr = VecGetSize(vec,&N); CHKERRQ(ierr);
    sum  = -sum/N;
    ierr = VecShift(&sum,vec); CHKERRQ(ierr);
  }

  for ( j=0; j<n; j++ ) {
    ierr = VecDot(vec,sp->vecs[j],&sum);CHKERRQ(ierr);
    sum  = -sum;
    ierr = VecAYPX(&sum,sp->vecs[j],vec); CHKERRQ(ierr);
  }
  
  return 0;
}
