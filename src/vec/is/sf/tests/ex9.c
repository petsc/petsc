static char help[]= "This example shows 1) how to transfer vectors from a parent communicator to vectors on a child communicator and vice versa;\n\
  2) how to transfer vectors from a subcommunicator to vectors on another subcommunicator. The two subcommunicators are not\n\
  required to cover all processes in PETSC_COMM_WORLD; 3) how to copy a vector from a parent communicator to vectors on its child communicators.\n\
  To run any example with VECCUDA vectors, add -vectype cuda to the argument list\n\n";

#include <petscvec.h>
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscMPIInt    nproc,grank,mycolor;
  PetscInt       i,n,N = 20,low,high;
  MPI_Comm       subcomm;
  Vec            x  = PETSC_NULL; /* global vectors on PETSC_COMM_WORLD */
  Vec            yg = PETSC_NULL; /* global vectors on PETSC_COMM_WORLD */
  VecScatter     vscat;
  IS             ix,iy;
  PetscBool      iscuda = PETSC_FALSE;      /* Option to use VECCUDA vectors */
  PetscBool      optionflag, compareflag;
  char           vectypename[PETSC_MAX_PATH_LEN];
  PetscBool      world2sub  = PETSC_FALSE;  /* Copy a vector from WORLD to a subcomm? */
  PetscBool      sub2sub    = PETSC_FALSE;  /* Copy a vector from a subcomm to another subcomm? */
  PetscBool      world2subs = PETSC_FALSE;  /* Copy a vector from WORLD to multiple subcomms? */

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&nproc);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&grank);CHKERRMPI(ierr);

  PetscCheckFalse(nproc < 2,PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"This test must have at least two processes to run");

  ierr = PetscOptionsGetBool(NULL,0,"-world2sub",&world2sub,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,0,"-sub2sub",&sub2sub,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,0,"-world2subs",&world2subs,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-vectype",vectypename,sizeof(vectypename),&optionflag);CHKERRQ(ierr);
  if (optionflag) {
    ierr = PetscStrncmp(vectypename, "cuda", (size_t)4, &compareflag);CHKERRQ(ierr);
    if (compareflag) iscuda = PETSC_TRUE;
  }

  /* Split PETSC_COMM_WORLD into three subcomms. Each process can only see the subcomm it belongs to */
  mycolor = grank % 3;
  ierr    = MPI_Comm_split(PETSC_COMM_WORLD,mycolor,grank,&subcomm);CHKERRMPI(ierr);

  /*===========================================================================
   *  Transfer a vector x defined on PETSC_COMM_WORLD to a vector y defined on
   *  a subcommunicator of PETSC_COMM_WORLD and vice versa.
   *===========================================================================*/
  if (world2sub) {
    ierr = VecCreate(PETSC_COMM_WORLD, &x);CHKERRQ(ierr);
    ierr = VecSetSizes(x, PETSC_DECIDE, N);CHKERRQ(ierr);
    if (iscuda) {
      ierr = VecSetType(x, VECCUDA);CHKERRQ(ierr);
    } else {
      ierr = VecSetType(x, VECSTANDARD);CHKERRQ(ierr);
    }
    ierr = VecSetUp(x);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)x,"x_commworld");CHKERRQ(ierr); /* Give a name to view x clearly */

    /* Initialize x to [-0.0, -1.0, -2.0, ..., -19.0] */
    ierr = VecGetOwnershipRange(x,&low,&high);CHKERRQ(ierr);
    for (i=low; i<high; i++) {
      PetscScalar val = -i;
      ierr = VecSetValue(x,i,val,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = VecAssemblyBegin(x);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(x);CHKERRQ(ierr);

    /* Transfer x to a vector y only defined on subcomm0 and vice versa */
    if (mycolor == 0) { /* subcomm0 contains ranks 0, 3, 6, ... in PETSC_COMM_WORLD */
      Vec         y;
      PetscScalar *yvalue;
       ierr = VecCreate(subcomm, &y);CHKERRQ(ierr);
      ierr = VecSetSizes(y, PETSC_DECIDE, N);CHKERRQ(ierr);
      if (iscuda) {
        ierr = VecSetType(y, VECCUDA);CHKERRQ(ierr);
      } else {
        ierr = VecSetType(y, VECSTANDARD);CHKERRQ(ierr);
      }
      ierr = VecSetUp(y);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject)y,"y_subcomm_0");CHKERRQ(ierr); /* Give a name to view y clearly */
      ierr = VecGetLocalSize(y,&n);CHKERRQ(ierr);
      if (iscuda) {
        #if defined(PETSC_HAVE_CUDA)
          ierr = VecCUDAGetArray(y,&yvalue);CHKERRQ(ierr);
        #endif
      } else {
        ierr = VecGetArray(y,&yvalue);CHKERRQ(ierr);
      }
      /* Create yg on PETSC_COMM_WORLD and alias yg with y. They share the memory pointed by yvalue.
        Note this is a collective call. All processes have to call it and supply consistent N.
      */
      if (iscuda) {
        #if defined(PETSC_HAVE_CUDA)
          ierr = VecCreateMPICUDAWithArray(PETSC_COMM_WORLD,1,n,N,yvalue,&yg);CHKERRQ(ierr);
        #endif
      } else {
        ierr = VecCreateMPIWithArray(PETSC_COMM_WORLD,1,n,N,yvalue,&yg);CHKERRQ(ierr);
      }

      /* Create an identity map that makes yg[i] = x[i], i=0..N-1 */
      ierr = VecGetOwnershipRange(yg,&low,&high);CHKERRQ(ierr); /* low, high are global indices */
      ierr = ISCreateStride(PETSC_COMM_SELF,high-low,low,1,&ix);CHKERRQ(ierr);
      ierr = ISDuplicate(ix,&iy);CHKERRQ(ierr);

      /* Union of ix's on subcomm0 covers the full range of [0,N) */
      ierr = VecScatterCreate(x,ix,yg,iy,&vscat);CHKERRQ(ierr);
      ierr = VecScatterBegin(vscat,x,yg,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(vscat,x,yg,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

      /* Once yg got the data from x, we return yvalue to y so that we can use y in other operations.
        VecGetArray must be paired with VecRestoreArray.
      */
      if (iscuda) {
         #if defined(PETSC_HAVE_CUDA)
           ierr = VecCUDARestoreArray(y,&yvalue);CHKERRQ(ierr);
         #endif
      } else {
        ierr = VecRestoreArray(y,&yvalue);CHKERRQ(ierr);
      }

      /* Libraries on subcomm0 can safely use y now, for example, view and scale it */
      ierr = VecView(y,PETSC_VIEWER_STDOUT_(subcomm));CHKERRQ(ierr);
      ierr = VecScale(y,2.0);CHKERRQ(ierr);

      /* Send the new y back to x */
      ierr = VecGetArray(y,&yvalue);CHKERRQ(ierr); /* If VecScale is done on GPU, Petsc will prepare a valid yvalue for access */
      /* Supply new yvalue to yg without memory copying */
      ierr = VecPlaceArray(yg,yvalue);CHKERRQ(ierr);
      ierr = VecScatterBegin(vscat,yg,x,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      ierr = VecScatterEnd(vscat,yg,x,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      ierr = VecResetArray(yg);CHKERRQ(ierr);
      if (iscuda) {
        #if defined(PETSC_HAVE_CUDA)
          ierr = VecCUDARestoreArray(y,&yvalue);CHKERRQ(ierr);
        #endif
      } else {
        ierr = VecRestoreArray(y,&yvalue);CHKERRQ(ierr);
      }

      ierr = VecDestroy(&y);CHKERRQ(ierr);
    } else {
      /* Ranks outside of subcomm0 do not supply values to yg */
      if (iscuda) {
        #if defined(PETSC_HAVE_CUDA)
          ierr = VecCreateMPICUDAWithArray(PETSC_COMM_WORLD,1,0/*n*/,N,NULL,&yg);CHKERRQ(ierr);
        #endif
      } else {
        ierr = VecCreateMPIWithArray(PETSC_COMM_WORLD,1,0/*n*/,N,NULL,&yg);CHKERRQ(ierr);
      }

      /* Ranks in subcomm0 already specified the full range of the identity map. The remaining
        ranks just need to create empty ISes to cheat VecScatterCreate.
      */
      ierr = ISCreateStride(PETSC_COMM_SELF,0,0,1,&ix);CHKERRQ(ierr);
      ierr = ISDuplicate(ix,&iy);CHKERRQ(ierr);

      ierr = VecScatterCreate(x,ix,yg,iy,&vscat);CHKERRQ(ierr);
      ierr = VecScatterBegin(vscat,x,yg,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(vscat,x,yg,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

      /* Send the new y back to x. Ranks outside of subcomm0 actually have nothing to send.
        But they have to call VecScatterBegin/End since these routines are collective.
      */
      ierr = VecScatterBegin(vscat,yg,x,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      ierr = VecScatterEnd(vscat,yg,x,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    }

    ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = ISDestroy(&ix);CHKERRQ(ierr);
    ierr = ISDestroy(&iy);CHKERRQ(ierr);
    ierr = VecDestroy(&x);CHKERRQ(ierr);
    ierr = VecDestroy(&yg);CHKERRQ(ierr);
    ierr = VecScatterDestroy(&vscat);CHKERRQ(ierr);
  } /* world2sub */

  /*===========================================================================
   *  Transfer a vector x defined on subcomm0 to a vector y defined on
   *  subcomm1. The two subcomms are not overlapping and their union is
   *  not necessarily equal to PETSC_COMM_WORLD.
   *===========================================================================*/
  if (sub2sub) {
    if (mycolor == 0) {
      /* Intentionally declare N as a local variable so that processes in subcomm1 do not know its value */
      PetscInt          n,N = 22;
      Vec               x,xg,yg;
      IS                ix,iy;
      VecScatter        vscat;
      const PetscScalar *xvalue;
      MPI_Comm          intercomm,parentcomm;
      PetscMPIInt       lrank;

      ierr = MPI_Comm_rank(subcomm,&lrank);CHKERRMPI(ierr);
      /* x is on subcomm */
      ierr = VecCreate(subcomm, &x);CHKERRQ(ierr);
      ierr = VecSetSizes(x, PETSC_DECIDE, N);CHKERRQ(ierr);
      if (iscuda) {
        ierr = VecSetType(x, VECCUDA);CHKERRQ(ierr);
      } else {
        ierr = VecSetType(x, VECSTANDARD);CHKERRQ(ierr);
      }
      ierr = VecSetUp(x);CHKERRQ(ierr);
      ierr = VecGetOwnershipRange(x,&low,&high);CHKERRQ(ierr);

      /* initialize x = [0.0, 1.0, 2.0, ..., 21.0] */
      for (i=low; i<high; i++) {
        PetscScalar val = i;
        ierr = VecSetValue(x,i,val,INSERT_VALUES);CHKERRQ(ierr);
      }
      ierr = VecAssemblyBegin(x);CHKERRQ(ierr);
      ierr = VecAssemblyEnd(x);CHKERRQ(ierr);

      ierr = MPI_Intercomm_create(subcomm,0,PETSC_COMM_WORLD/*peer_comm*/,1,100/*tag*/,&intercomm);CHKERRMPI(ierr);

      /* Tell rank 0 of subcomm1 the global size of x */
      if (!lrank) {ierr = MPI_Send(&N,1,MPIU_INT,0/*receiver's rank in remote comm, i.e., subcomm1*/,200/*tag*/,intercomm);CHKERRMPI(ierr);}

      /* Create an intracomm Petsc can work on. Ranks in subcomm0 are ordered before ranks in subcomm1 in parentcomm.
        But this order actually does not matter, since what we care is vector y, which is defined on subcomm1.
      */
      ierr = MPI_Intercomm_merge(intercomm,0/*low*/,&parentcomm);CHKERRMPI(ierr);

      /* Create a vector xg on parentcomm, which shares memory with x */
      ierr = VecGetLocalSize(x,&n);CHKERRQ(ierr);
      if (iscuda) {
        #if defined(PETSC_HAVE_CUDA)
          ierr = VecCUDAGetArrayRead(x,&xvalue);CHKERRQ(ierr);
          ierr = VecCreateMPICUDAWithArray(parentcomm,1,n,N,xvalue,&xg);CHKERRQ(ierr);
        #endif
      } else {
        ierr = VecGetArrayRead(x,&xvalue);CHKERRQ(ierr);
        ierr = VecCreateMPIWithArray(parentcomm,1,n,N,xvalue,&xg);CHKERRQ(ierr);
      }

      /* Ranks in subcomm 0 have nothing on yg, so they simply have n=0, array=NULL */
      if (iscuda) {
        #if defined(PETSC_HAVE_CUDA)
          ierr = VecCreateMPICUDAWithArray(parentcomm,1,0/*n*/,N,NULL/*array*/,&yg);CHKERRQ(ierr);
        #endif
      } else {
        ierr = VecCreateMPIWithArray(parentcomm,1,0/*n*/,N,NULL/*array*/,&yg);CHKERRQ(ierr);
      }

      /* Create the vecscatter, which does identity map by setting yg[i] = xg[i], i=0..N-1. */
      ierr = VecGetOwnershipRange(xg,&low,&high);CHKERRQ(ierr); /* low, high are global indices of xg */
      ierr = ISCreateStride(PETSC_COMM_SELF,high-low,low,1,&ix);CHKERRQ(ierr);
      ierr = ISDuplicate(ix,&iy);CHKERRQ(ierr);
      ierr = VecScatterCreate(xg,ix,yg,iy,&vscat);CHKERRQ(ierr);

      /* Scatter values from xg to yg */
      ierr = VecScatterBegin(vscat,xg,yg,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(vscat,xg,yg,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

      /* After the VecScatter is done, xg is idle so we can safely return xvalue to x */
      if (iscuda) {
        #if defined(PETSC_HAVE_CUDA)
          ierr = VecCUDARestoreArrayRead(x,&xvalue);CHKERRQ(ierr);
        #endif
      } else {
        ierr = VecRestoreArrayRead(x,&xvalue);CHKERRQ(ierr);
      }
      ierr = VecDestroy(&x);CHKERRQ(ierr);
      ierr = ISDestroy(&ix);CHKERRQ(ierr);
      ierr = ISDestroy(&iy);CHKERRQ(ierr);
      ierr = VecDestroy(&xg);CHKERRQ(ierr);
      ierr = VecDestroy(&yg);CHKERRQ(ierr);
      ierr = VecScatterDestroy(&vscat);CHKERRQ(ierr);
      ierr = MPI_Comm_free(&intercomm);CHKERRMPI(ierr);
      ierr = MPI_Comm_free(&parentcomm);CHKERRMPI(ierr);
    } else if (mycolor == 1) { /* subcomm 1, containing ranks 1, 4, 7, ... in PETSC_COMM_WORLD */
      PetscInt    n,N;
      Vec         y,xg,yg;
      IS          ix,iy;
      VecScatter  vscat;
      PetscScalar *yvalue;
      MPI_Comm    intercomm,parentcomm;
      PetscMPIInt lrank;

      ierr = MPI_Comm_rank(subcomm,&lrank);CHKERRMPI(ierr);
      ierr = MPI_Intercomm_create(subcomm,0,PETSC_COMM_WORLD/*peer_comm*/,0/*remote_leader*/,100/*tag*/,&intercomm);CHKERRMPI(ierr);

      /* Two rank-0 are talking */
      if (!lrank) {ierr = MPI_Recv(&N,1,MPIU_INT,0/*sender's rank in remote comm, i.e. subcomm0*/,200/*tag*/,intercomm,MPI_STATUS_IGNORE);CHKERRMPI(ierr);}
      /* Rank 0 of subcomm1 bcasts N to its members */
      ierr = MPI_Bcast(&N,1,MPIU_INT,0/*local root*/,subcomm);CHKERRMPI(ierr);

      /* Create a intracomm Petsc can work on */
      ierr = MPI_Intercomm_merge(intercomm,1/*high*/,&parentcomm);CHKERRMPI(ierr);

      /* Ranks in subcomm1 have nothing on xg, so they simply have n=0, array=NULL.*/
      if (iscuda) {
        #if defined(PETSC_HAVE_CUDA)
          ierr = VecCreateMPICUDAWithArray(parentcomm,1/*bs*/,0/*n*/,N,NULL/*array*/,&xg);CHKERRQ(ierr);
        #endif
      } else {
        ierr = VecCreateMPIWithArray(parentcomm,1/*bs*/,0/*n*/,N,NULL/*array*/,&xg);CHKERRQ(ierr);
      }

      ierr = VecCreate(subcomm, &y);CHKERRQ(ierr);
      ierr = VecSetSizes(y, PETSC_DECIDE, N);CHKERRQ(ierr);
      if (iscuda) {
        ierr = VecSetType(y, VECCUDA);CHKERRQ(ierr);
      } else {
        ierr = VecSetType(y, VECSTANDARD);CHKERRQ(ierr);
      }
      ierr = VecSetUp(y);CHKERRQ(ierr);

      ierr = PetscObjectSetName((PetscObject)y,"y_subcomm_1");CHKERRQ(ierr); /* Give a name to view y clearly */
      ierr = VecGetLocalSize(y,&n);CHKERRQ(ierr);
      if (iscuda) {
        #if defined(PETSC_HAVE_CUDA)
          ierr = VecCUDAGetArray(y,&yvalue);CHKERRQ(ierr);
        #endif
      } else {
        ierr = VecGetArray(y,&yvalue);CHKERRQ(ierr);
      }
      /* Create a vector yg on parentcomm, which shares memory with y. xg and yg must be
        created in the same order in subcomm0/1. For example, we can not reverse the order of
        creating xg and yg in subcomm1.
      */
      if (iscuda) {
        #if defined(PETSC_HAVE_CUDA)
          ierr = VecCreateMPICUDAWithArray(parentcomm,1/*bs*/,n,N,yvalue,&yg);CHKERRQ(ierr);
        #endif
      } else {
        ierr = VecCreateMPIWithArray(parentcomm,1/*bs*/,n,N,yvalue,&yg);CHKERRQ(ierr);
      }

      /* Ranks in subcomm0 already specified the full range of the identity map.
        ranks in subcomm1 just need to create empty ISes to cheat VecScatterCreate.
      */
      ierr = ISCreateStride(PETSC_COMM_SELF,0,0,1,&ix);CHKERRQ(ierr);
      ierr = ISDuplicate(ix,&iy);CHKERRQ(ierr);
      ierr = VecScatterCreate(xg,ix,yg,iy,&vscat);CHKERRQ(ierr);

      /* Scatter values from xg to yg */
      ierr = VecScatterBegin(vscat,xg,yg,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(vscat,xg,yg,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

      /* After the VecScatter is done, values in yg are available. y is our interest, so we return yvalue to y */
      if (iscuda) {
        #if defined(PETSC_HAVE_CUDA)
          ierr = VecCUDARestoreArray(y,&yvalue);CHKERRQ(ierr);
        #endif
      } else {
        ierr = VecRestoreArray(y,&yvalue);CHKERRQ(ierr);
      }

      /* Libraries on subcomm1 can safely use y now, for example, view it */
      ierr = VecView(y,PETSC_VIEWER_STDOUT_(subcomm));CHKERRQ(ierr);

      ierr = VecDestroy(&y);CHKERRQ(ierr);
      ierr = ISDestroy(&ix);CHKERRQ(ierr);
      ierr = ISDestroy(&iy);CHKERRQ(ierr);
      ierr = VecDestroy(&xg);CHKERRQ(ierr);
      ierr = VecDestroy(&yg);CHKERRQ(ierr);
      ierr = VecScatterDestroy(&vscat);CHKERRQ(ierr);
      ierr = MPI_Comm_free(&intercomm);CHKERRMPI(ierr);
      ierr = MPI_Comm_free(&parentcomm);CHKERRMPI(ierr);
    } else if (mycolor == 2) { /* subcomm2 */
      /* Processes in subcomm2 do not participate in the VecScatter. They can freely do unrelated things on subcomm2 */
    }
  } /* sub2sub */

  /*===========================================================================
   *  Copy a vector x defined on PETSC_COMM_WORLD to vectors y defined on
   *  every subcommunicator of PETSC_COMM_WORLD. We could use multiple transfers
   *  as we did in case 1, but that is not efficient. Instead, we use one vecscatter
   *  to achieve that.
   *===========================================================================*/
  if (world2subs) {
    Vec         y;
    PetscInt    n,N=15,xstart,ystart,low,high;
    PetscScalar *yvalue;

    /* Initialize x to [0, 1, 2, 3, ..., N-1] */
    ierr = VecCreate(PETSC_COMM_WORLD, &x);CHKERRQ(ierr);
    ierr = VecSetSizes(x, PETSC_DECIDE, N);CHKERRQ(ierr);
    if (iscuda) {
      ierr = VecSetType(x, VECCUDA);CHKERRQ(ierr);
    } else {
      ierr = VecSetType(x, VECSTANDARD);CHKERRQ(ierr);
    }
    ierr = VecSetUp(x);CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(x,&low,&high);CHKERRQ(ierr);
    for (i=low; i<high; i++) {ierr = VecSetValue(x,i,(PetscScalar)i,INSERT_VALUES);CHKERRQ(ierr);}
    ierr = VecAssemblyBegin(x);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(x);CHKERRQ(ierr);

    /* Every subcomm has a y as long as x */
    ierr = VecCreate(subcomm, &y);CHKERRQ(ierr);
    ierr = VecSetSizes(y, PETSC_DECIDE, N);CHKERRQ(ierr);
    if (iscuda) {
      ierr = VecSetType(y, VECCUDA);CHKERRQ(ierr);
    } else {
      ierr = VecSetType(y, VECSTANDARD);CHKERRQ(ierr);
    }
    ierr = VecSetUp(y);CHKERRQ(ierr);
    ierr = VecGetLocalSize(y,&n);CHKERRQ(ierr);

    /* Create a global vector yg on PETSC_COMM_WORLD using y's memory. yg's global size = N*(number of subcommunicators).
       Eeach rank in subcomms contributes a piece to construct the global yg. Keep in mind that pieces from a subcomm are not
       necessarily consecutive in yg. That depends on how PETSC_COMM_WORLD is split. In our case, subcomm0 is made of rank
       0, 3, 6 etc from PETSC_COMM_WORLD. So subcomm0's pieces are interleaved with pieces from other subcomms in yg.
    */
    if (iscuda) {
      #if defined(PETSC_HAVE_CUDA)
        ierr = VecCUDAGetArray(y,&yvalue);CHKERRQ(ierr);
        ierr = VecCreateMPICUDAWithArray(PETSC_COMM_WORLD,1,n,PETSC_DECIDE,yvalue,&yg);CHKERRQ(ierr);
      #endif
    } else {
      ierr = VecGetArray(y,&yvalue);CHKERRQ(ierr);
      ierr = VecCreateMPIWithArray(PETSC_COMM_WORLD,1,n,PETSC_DECIDE,yvalue,&yg);CHKERRQ(ierr);
    }
    ierr = PetscObjectSetName((PetscObject)yg,"yg_on_subcomms");CHKERRQ(ierr); /* Give a name to view yg clearly */

    /* The following two lines are key. From xstart, we know where to pull entries from x. Note that we get xstart from y,
       since first entry of y on this rank is from x[xstart]. From ystart, we know where ot put entries to yg.
     */
    ierr = VecGetOwnershipRange(y,&xstart,NULL);CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(yg,&ystart,NULL);CHKERRQ(ierr);

    ierr = ISCreateStride(PETSC_COMM_SELF,n,xstart,1,&ix);CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF,n,ystart,1,&iy);CHKERRQ(ierr);
    ierr = VecScatterCreate(x,ix,yg,iy,&vscat);CHKERRQ(ierr);
    ierr = VecScatterBegin(vscat,x,yg,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(vscat,x,yg,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

    /* View yg on PETSC_COMM_WORLD before destroying it. We shall see the interleaving effect in output. */
    ierr = VecView(yg,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = VecDestroy(&yg);CHKERRQ(ierr);

    /* Restory yvalue so that processes in subcomm can use y from now on. */
    if (iscuda) {
      #if defined(PETSC_HAVE_CUDA)
        ierr = VecCUDARestoreArray(y,&yvalue);CHKERRQ(ierr);
      #endif
    } else {
      ierr = VecRestoreArray(y,&yvalue);CHKERRQ(ierr);
    }
    ierr = VecScale(y,3.0);CHKERRQ(ierr);

    ierr = ISDestroy(&ix);CHKERRQ(ierr); /* One can also destroy ix, iy immediately after VecScatterCreate() */
    ierr = ISDestroy(&iy);CHKERRQ(ierr);
    ierr = VecDestroy(&x);CHKERRQ(ierr);
    ierr = VecDestroy(&y);CHKERRQ(ierr);
    ierr = VecScatterDestroy(&vscat);CHKERRQ(ierr);
  } /* world2subs */

  ierr = MPI_Comm_free(&subcomm);CHKERRMPI(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
     requires: !defined(PETSC_HAVE_MPIUNI)

   testset:
     nsize: 7

     test:
       suffix: 1
       args: -world2sub

     test:
       suffix: 2
       args: -sub2sub
       # deadlocks with NECMPI and INTELMPI (20210400300)
       requires: !defined(PETSC_HAVE_NECMPI) !defined(PETSC_HAVE_I_MPI_NUMVERSION)

     test:
       suffix: 3
       args: -world2subs

     test:
       suffix: 4
       args: -world2sub -vectype cuda
       requires: cuda

     test:
       suffix: 5
       args: -sub2sub -vectype cuda
       requires: cuda

     test:
      suffix: 6
      args: -world2subs -vectype cuda
      requires: cuda

     test:
       suffix: 7
       args: -world2sub -sf_type neighbor
       output_file: output/ex9_1.out
       # OpenMPI has a bug wrt MPI_Neighbor_alltoallv etc (https://github.com/open-mpi/ompi/pull/6782). Once the patch is in, we can remove !define(PETSC_HAVE_OMPI_MAJOR_VERSION)
       # segfaults with NECMPI
       requires: defined(PETSC_HAVE_MPI_NEIGHBORHOOD_COLLECTIVES) !defined(PETSC_HAVE_OMPI_MAJOR_VERSION) !defined(PETSC_HAVE_NECMPI)
TEST*/

