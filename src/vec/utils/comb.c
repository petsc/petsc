From bsmith@fire.mcs.anl.gov Wed Feb  3 18:21:01 1999
Status: RO
X-Status: 
Received: (from daemon@localhost) by antares.mcs.anl.gov (8.6.10/8.6.10)
	id SAA03758 for petsc-maint-dist; Wed, 3 Feb 1999 18:21:00 -0600
Received: from prancer.cs.utk.edu (PRANCER.CS.UTK.EDU [128.169.92.101]) by antares.mcs.anl.gov (8.6.10/8.6.10)  with ESMTP
        id SAA03746 for <petsc-maint@mcs.anl.gov>; Wed, 3 Feb 1999 18:20:43 -0600
Received: from LOCALHOST.cs.utk.edu (LOCALHOST.cs.utk.edu [127.0.0.1])
        by prancer.cs.utk.edu (cf v3.2) with SMTP id TAA26462
        for <petsc-maint@mcs.anl.gov>; Wed, 3 Feb 1999 19:20:39 -0500 (EST)
Message-Id: <199902040020.TAA26462@prancer.cs.utk.edu>
X-Authentication-Warning: prancer.cs.utk.edu: LOCALHOST.cs.utk.edu [127.0.0.1] didn't use HELO protocol
To: petsc-maint@mcs.anl.gov
Subject: [PETSC #2317] add to vec.h
Date: Wed, 03 Feb 1999 19:20:38 -0500
From: Victor Eijkhout <eijkhout@cs.utk.edu>
Cc: petsc-maint@mcs.anl.gov

typedef struct _p_DotProducts* DotProducts;
typedef int DPRequest;

Victor Eijkhout
Department of Computer Science; University of Tennessee, Knoxville, TN 37996
phone: +1 423 974 8298 / 8295 home +1 423 212 4935
http://www.cs.utk.edu/~eijkhout/


From bsmith@fire.mcs.anl.gov Wed Feb  3 18:21:45 1999
Status: RO
X-Status: 
Received: (from daemon@localhost) by antares.mcs.anl.gov (8.6.10/8.6.10)
	id SAA04144 for petsc-maint-dist; Wed, 3 Feb 1999 18:21:45 -0600
Received: from prancer.cs.utk.edu (PRANCER.CS.UTK.EDU [128.169.92.101]) by antares.mcs.anl.gov (8.6.10/8.6.10)  with ESMTP
        id SAA04134 for <petsc-maint@mcs.anl.gov>; Wed, 3 Feb 1999 18:21:35 -0600
Received: from LOCALHOST.cs.utk.edu (LOCALHOST.cs.utk.edu [127.0.0.1])
        by prancer.cs.utk.edu (cf v3.2) with SMTP id TAA26476
        for <petsc-maint@mcs.anl.gov>; Wed, 3 Feb 1999 19:21:32 -0500 (EST)
Message-Id: <199902040021.TAA26476@prancer.cs.utk.edu>
X-Authentication-Warning: prancer.cs.utk.edu: LOCALHOST.cs.utk.edu [127.0.0.1] didn't use HELO protocol
To: petsc-maint@mcs.anl.gov
Date: Wed, 03 Feb 1999 19:21:31 -0500
From: Victor Eijkhout <eijkhout@cs.utk.edu>
Subject: [PETSC #2318] 
Cc: petsc-maint@mcs.anl.gov

#include "parpre_vec.h"
#include "src/vec/impls/dvecimpl.h"
#include "mpi.h"

/****************************************************************
 * Combined Dot Products routines
 * 
 * The user needs to create (and later destroy) a structure `DotProducts'
 * with DotProductsCreate and DotProductsDestroy.
 * Combined dot products are then performed as follows:
 * - a sequence of DotProductsSet calls performs the local parts of
 * the dot products, and stores them; there is no global communication.
 * - the first DotProductsGet call causes the global communication to be
 * performed; every subsequent DotProductsGet call is a read from
 * the results array.
 ****************************************************************/

struct _p_DotProducts {
  MPI_Comm comm;
  Scalar *lvalues,*gvalues;
  int size,high_write,high_read,state;
};
#define WRITE_STATE 0
#define READ_STATE 1

#undef __FUNC__
#define __FUNC__ "DotProductsCreate"
/*
  DotProductsCreate
  Create a dot products object.
*/
int DotProductsCreate(MPI_Comm comm,DotProducts *dp)
{
  DotProducts newdp;
  PetscFunctionBegin;
  newdp = PetscNew(struct _p_DotProducts); CHKPTRQ(newdp);
  newdp->size = newdp->high_write = newdp->high_read = 0;
  newdp->lvalues = newdp->gvalues = 0;
  newdp->state = WRITE_STATE;
  newdp->comm = comm;
  *dp = newdp;
  PetscFunctionReturn(0);
}

#undef __FUNC__
#define __FUNC__ "DotProductsDestroy"
/*
  DotProductsDestroy
  Destroy a dot products object.
*/
int DotProductsDestroy(DotProducts dp)
{
  PetscFunctionBegin;
  PetscFree(dp->lvalues); PetscFree(dp->gvalues); PetscFree(dp);
  PetscFunctionReturn(0);
}

#undef __FUNC__
#define __FUNC__ "DotProductsSet"
/*
  DotProductsSet
  Submit one dot product (two vectors) for execution; the result can
  later be retrieved with DotProductsGet. Two parameters are used
  to identify the specific product:
  . loc: PETSC_DECIDE or an explicit number
  . req: a handle with which to retrieve the global result later;
  this is 'loc' if that was explicitly given.
*/
int DotProductsSet(DotProducts dp,Vec x,Vec y,int loc,DPRequest *req)
{
  Scalar *xx,*yy,res;
  int the_loc,ierr;

  PetscFunctionBegin;
  /* compute local result */
  ierr = VecDot_Seq(x,y,&res); CHKERRQ(ierr);
  /* decide where we are going to write it */
#if PETSC_DECIDE >= 0
#error "PETSC_DECIDE conflict: has to be negative"
#endif
  if (loc==PETSC_DECIDE) {
    the_loc = dp->high_write++;
  } else {
    the_loc = loc;
    if (the_loc>=dp->high_write) dp->high_write = the_loc+1;
  }
  /* make sure that there is a place to write it */
  if (the_loc>=dp->size) {
    if (dp->size==0) {
      /* initial allocation */
      dp->lvalues = (Scalar *) PetscMalloc(100*sizeof(Scalar));
      CHKPTRQ(dp->lvalues);
      dp->gvalues = (Scalar *) PetscMalloc(100*sizeof(Scalar));
      CHKPTRQ(dp->gvalues);
      dp->size = 100;
    } else {
      /* double existing storage */
      Scalar *tmp; int new_size;
      new_size = PetscMax(*req+1,2*dp->size);

      tmp = (Scalar *) PetscMalloc(new_size*sizeof(Scalar)); CHKPTRQ(tmp);
      if (dp->high_write)
	PetscMemcpy(tmp,dp->lvalues,dp->high_write*sizeof(Scalar));
      PetscFree(dp->lvalues); dp->lvalues = tmp;

      tmp = (Scalar *) PetscMalloc(new_size*sizeof(Scalar)); CHKPTRQ(tmp);
      if (dp->high_read)
	PetscMemcpy(tmp,dp->gvalues,dp->high_read*sizeof(Scalar));
      PetscFree(dp->gvalues); dp->gvalues = tmp;
      dp->size = new_size;
    }
  }
  dp->lvalues[the_loc] = res;
  dp->state = WRITE_STATE;
  *req = the_loc;
  PetscFunctionReturn(0);
}

#undef __FUNC__
#define __FUNC__ "DotProductsGet"
/*
  DotProductsGet
  Retrieve a dot product value.
  The `req' identifier specifies what value to retrieve; this 
  identifier was generated by an earlier DotProductsSet call,
  or it can be the `loc' given to a DotProductsSet call.
*/
int DotProductsGet(DotProducts dp,DPRequest req,Scalar *r)
{
  PetscFunctionBegin;
  if (dp->state==WRITE_STATE) {
    if ((int)req>dp->high_write || (int)req<0)
      SETERRQ(1,0,"Illegal request, no local result written at this id");
    /* we have been writing local results so far, time for communication */
#if defined(USE_PETSC_COMPLEX)
    MPI_Allreduce(dp->lvalues,dp->gvalues,
		  2*dp->high_write,MPI_DOUBLE,MPI_SUM,dp->comm);
#else
    MPI_Allreduce((void*)dp->lvalues,(void*)dp->gvalues,
		  dp->high_write,MPI_DOUBLE,MPI_SUM,dp->comm);
#endif
    dp->state = READ_STATE;
    dp->high_read = dp->high_write; dp->high_write = 0;
  }
  if ((int)req>dp->high_read || (int)req<0)
    SETERRQ(1,0,"Illegal request, no local result available at this id");
  *r = dp->gvalues[(int)req];
  PetscFunctionReturn(0);
}

#undef __FUNC__
#define __FUNC__ "DotProductsClear"
/*
  DotProductsClear
*/
int DotProductsClear(DotProducts dp)
{
  PetscFunctionBegin;
  dp->high_write = dp->high_read = 0;
  PetscMemzero(dp->lvalues,dp->size*sizeof(Scalar));
  PetscMemzero(dp->gvalues,dp->size*sizeof(Scalar));
  PetscFunctionReturn(0);
}

