/*$Id: pack.c,v 1.1 2000/06/06 03:49:19 bsmith Exp bsmith $*/
 
#include "petscda.h"     /*I      "petscda.h"     I*/
#include "petscmat.h"    /*I      "petscmat.h"    I*/

struct VecPackLink {
  Vec                globalholder;
  DA                 da;
  int                n,rstart;      /* rstart is relative to this processor */
  struct VecPackLink *next;
};

struct _p_VecPack {
  MPI_Comm           comm;
  int                rank;
  int                n,N,rstart,rend;
  struct VecPackLink *next;
};

#undef __FUNC__  
#define __FUNC__ /*<a name="VecPackCreate"></a>*/"VecPackCreate"
/*@C
    VecPackCreate - Creates a vector packer, used to generate "composite
      vectors made up of several subvectors.

    Collective on MPI_Comm

    Input Parameter:
.   comm - the processors that will share the global vector

    Output Parameters:
.   packer - the packer object

    Level: advanced

.seealso VecPackDestroy(), VecPackAddArray(), VecPackAddDA(), VecPackApplyForward(),
         VecPackApplyReverse(), VecPackCreateGlobalVector()

@*/
int VecPackCreate(MPI_Comm comm,VecPack *packer)
{
  int     ierr;
  VecPack p;

  PetscFunctionBegin;
  p       = PetscNew(struct _p_VecPack);CHKPTRQ(p);
  p->n    = 0;
  p->next = PETSC_NULL;
  p->comm = comm;
  ierr    = MPI_Comm_size(comm,&p->rank);CHKERRQ(ierr);
  *packer = p;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="VecPackDestroy"></a>*/"VecPackDestroy"
/*@C
    VecPackDestroy - Destroys a vector packer.

    Collective on VecPack

    Input Parameter:
.   packer - the packer object

    Level: advanced

.seealso VecPackCreate(), VecPackAddArray(), VecPackAddDA(), VecPackApplyForward(),
         VecPackApplyReverse(), VecPackCreateGlobalVector()

@*/
int VecPackDestroy(VecPack packer)
{
  int                ierr;
  struct VecPackLink *next = packer->next,*prev;

  PetscFunctionBegin;
  while (next) {
    prev = next;
    next = next->next;
    ierr = PetscFree(prev);CHKERRQ(ierr);
  }
  ierr = PetscFree(packer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="VecPackScatter_Array"></a>*/"VecPackScatter_Array"
int VecPackScatter_Array(VecPack packer,struct VecPackLink *mine,Vec vec,Scalar *array)
{
  int    ierr;
  Scalar *varray;

  PetscFunctionBegin;

  if (!packer->rank) {
    ierr    = VecGetArray(vec,&varray);CHKERRQ(ierr);
    varray += mine->rstart;
    ierr    = PetscMemcpy(array,varray,mine->n*sizeof(Scalar));CHKERRQ(ierr);
  }
  ierr    = MPI_Bcast(array,mine->n,MPIU_SCALAR,0,packer->comm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="VecPackAddArray"></a>*/"VecPackAddArray"
int VecPackAddArray(VecPack packer,int n)
{
  int                ierr;
  struct VecPackLink *mine,*next = packer->next;

  PetscFunctionBegin;

  /* create new link */
  mine               = PetscNew(struct VecPackLink);CHKPTRQ(mine);
  mine->n            = n;
  mine->da           = PETSC_NULL;
  mine->globalholder = PETSC_NULL;
  if (!packer->rank) packer->n += n;

  /* add to end of list */
  if (!next) {
    packer->next = mine;
  } else {
    while (next->next) next = next->next;
    next->next = mine;
  }
  PetscFunctionReturn(0);
}



