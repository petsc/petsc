static char help[]= "This example shows 1) how to transfer vectors from a parent communicator to vectors on a child communicator and vice versa;\n\
  2) how to transfer vectors from a subcommunicator to vectors on another subcommunicator. The two subcommunicators are not\n\
  required to cover all processes in PETSC_COMM_WORLD; 3) how to copy a vector from a parent communicator to vectors on its child communicators.\n\
  To run any example with VECCUDA vectors, add -vectype cuda to the argument list\n\n";

#include <petscvec.h>
int main(int argc,char **argv)
{
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

  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&nproc));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&grank));

  PetscCheckFalse(nproc < 2,PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"This test must have at least two processes to run");

  CHKERRQ(PetscOptionsGetBool(NULL,0,"-world2sub",&world2sub,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,0,"-sub2sub",&sub2sub,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,0,"-world2subs",&world2subs,NULL));
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-vectype",vectypename,sizeof(vectypename),&optionflag));
  if (optionflag) {
    CHKERRQ(PetscStrncmp(vectypename, "cuda", (size_t)4, &compareflag));
    if (compareflag) iscuda = PETSC_TRUE;
  }

  /* Split PETSC_COMM_WORLD into three subcomms. Each process can only see the subcomm it belongs to */
  mycolor = grank % 3;
  CHKERRMPI(MPI_Comm_split(PETSC_COMM_WORLD,mycolor,grank,&subcomm));

  /*===========================================================================
   *  Transfer a vector x defined on PETSC_COMM_WORLD to a vector y defined on
   *  a subcommunicator of PETSC_COMM_WORLD and vice versa.
   *===========================================================================*/
  if (world2sub) {
    CHKERRQ(VecCreate(PETSC_COMM_WORLD, &x));
    CHKERRQ(VecSetSizes(x, PETSC_DECIDE, N));
    if (iscuda) {
      CHKERRQ(VecSetType(x, VECCUDA));
    } else {
      CHKERRQ(VecSetType(x, VECSTANDARD));
    }
    CHKERRQ(VecSetUp(x));
    CHKERRQ(PetscObjectSetName((PetscObject)x,"x_commworld")); /* Give a name to view x clearly */

    /* Initialize x to [-0.0, -1.0, -2.0, ..., -19.0] */
    CHKERRQ(VecGetOwnershipRange(x,&low,&high));
    for (i=low; i<high; i++) {
      PetscScalar val = -i;
      CHKERRQ(VecSetValue(x,i,val,INSERT_VALUES));
    }
    CHKERRQ(VecAssemblyBegin(x));
    CHKERRQ(VecAssemblyEnd(x));

    /* Transfer x to a vector y only defined on subcomm0 and vice versa */
    if (mycolor == 0) { /* subcomm0 contains ranks 0, 3, 6, ... in PETSC_COMM_WORLD */
      Vec         y;
      PetscScalar *yvalue;
       CHKERRQ(VecCreate(subcomm, &y));
      CHKERRQ(VecSetSizes(y, PETSC_DECIDE, N));
      if (iscuda) {
        CHKERRQ(VecSetType(y, VECCUDA));
      } else {
        CHKERRQ(VecSetType(y, VECSTANDARD));
      }
      CHKERRQ(VecSetUp(y));
      CHKERRQ(PetscObjectSetName((PetscObject)y,"y_subcomm_0")); /* Give a name to view y clearly */
      CHKERRQ(VecGetLocalSize(y,&n));
      if (iscuda) {
        #if defined(PETSC_HAVE_CUDA)
          CHKERRQ(VecCUDAGetArray(y,&yvalue));
        #endif
      } else {
        CHKERRQ(VecGetArray(y,&yvalue));
      }
      /* Create yg on PETSC_COMM_WORLD and alias yg with y. They share the memory pointed by yvalue.
        Note this is a collective call. All processes have to call it and supply consistent N.
      */
      if (iscuda) {
        #if defined(PETSC_HAVE_CUDA)
          CHKERRQ(VecCreateMPICUDAWithArray(PETSC_COMM_WORLD,1,n,N,yvalue,&yg));
        #endif
      } else {
        CHKERRQ(VecCreateMPIWithArray(PETSC_COMM_WORLD,1,n,N,yvalue,&yg));
      }

      /* Create an identity map that makes yg[i] = x[i], i=0..N-1 */
      CHKERRQ(VecGetOwnershipRange(yg,&low,&high)); /* low, high are global indices */
      CHKERRQ(ISCreateStride(PETSC_COMM_SELF,high-low,low,1,&ix));
      CHKERRQ(ISDuplicate(ix,&iy));

      /* Union of ix's on subcomm0 covers the full range of [0,N) */
      CHKERRQ(VecScatterCreate(x,ix,yg,iy,&vscat));
      CHKERRQ(VecScatterBegin(vscat,x,yg,INSERT_VALUES,SCATTER_FORWARD));
      CHKERRQ(VecScatterEnd(vscat,x,yg,INSERT_VALUES,SCATTER_FORWARD));

      /* Once yg got the data from x, we return yvalue to y so that we can use y in other operations.
        VecGetArray must be paired with VecRestoreArray.
      */
      if (iscuda) {
         #if defined(PETSC_HAVE_CUDA)
           CHKERRQ(VecCUDARestoreArray(y,&yvalue));
         #endif
      } else {
        CHKERRQ(VecRestoreArray(y,&yvalue));
      }

      /* Libraries on subcomm0 can safely use y now, for example, view and scale it */
      CHKERRQ(VecView(y,PETSC_VIEWER_STDOUT_(subcomm)));
      CHKERRQ(VecScale(y,2.0));

      /* Send the new y back to x */
      CHKERRQ(VecGetArray(y,&yvalue)); /* If VecScale is done on GPU, Petsc will prepare a valid yvalue for access */
      /* Supply new yvalue to yg without memory copying */
      CHKERRQ(VecPlaceArray(yg,yvalue));
      CHKERRQ(VecScatterBegin(vscat,yg,x,INSERT_VALUES,SCATTER_REVERSE));
      CHKERRQ(VecScatterEnd(vscat,yg,x,INSERT_VALUES,SCATTER_REVERSE));
      CHKERRQ(VecResetArray(yg));
      if (iscuda) {
        #if defined(PETSC_HAVE_CUDA)
          CHKERRQ(VecCUDARestoreArray(y,&yvalue));
        #endif
      } else {
        CHKERRQ(VecRestoreArray(y,&yvalue));
      }

      CHKERRQ(VecDestroy(&y));
    } else {
      /* Ranks outside of subcomm0 do not supply values to yg */
      if (iscuda) {
        #if defined(PETSC_HAVE_CUDA)
          CHKERRQ(VecCreateMPICUDAWithArray(PETSC_COMM_WORLD,1,0/*n*/,N,NULL,&yg));
        #endif
      } else {
        CHKERRQ(VecCreateMPIWithArray(PETSC_COMM_WORLD,1,0/*n*/,N,NULL,&yg));
      }

      /* Ranks in subcomm0 already specified the full range of the identity map. The remaining
        ranks just need to create empty ISes to cheat VecScatterCreate.
      */
      CHKERRQ(ISCreateStride(PETSC_COMM_SELF,0,0,1,&ix));
      CHKERRQ(ISDuplicate(ix,&iy));

      CHKERRQ(VecScatterCreate(x,ix,yg,iy,&vscat));
      CHKERRQ(VecScatterBegin(vscat,x,yg,INSERT_VALUES,SCATTER_FORWARD));
      CHKERRQ(VecScatterEnd(vscat,x,yg,INSERT_VALUES,SCATTER_FORWARD));

      /* Send the new y back to x. Ranks outside of subcomm0 actually have nothing to send.
        But they have to call VecScatterBegin/End since these routines are collective.
      */
      CHKERRQ(VecScatterBegin(vscat,yg,x,INSERT_VALUES,SCATTER_REVERSE));
      CHKERRQ(VecScatterEnd(vscat,yg,x,INSERT_VALUES,SCATTER_REVERSE));
    }

    CHKERRQ(VecView(x,PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(ISDestroy(&ix));
    CHKERRQ(ISDestroy(&iy));
    CHKERRQ(VecDestroy(&x));
    CHKERRQ(VecDestroy(&yg));
    CHKERRQ(VecScatterDestroy(&vscat));
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

      CHKERRMPI(MPI_Comm_rank(subcomm,&lrank));
      /* x is on subcomm */
      CHKERRQ(VecCreate(subcomm, &x));
      CHKERRQ(VecSetSizes(x, PETSC_DECIDE, N));
      if (iscuda) {
        CHKERRQ(VecSetType(x, VECCUDA));
      } else {
        CHKERRQ(VecSetType(x, VECSTANDARD));
      }
      CHKERRQ(VecSetUp(x));
      CHKERRQ(VecGetOwnershipRange(x,&low,&high));

      /* initialize x = [0.0, 1.0, 2.0, ..., 21.0] */
      for (i=low; i<high; i++) {
        PetscScalar val = i;
        CHKERRQ(VecSetValue(x,i,val,INSERT_VALUES));
      }
      CHKERRQ(VecAssemblyBegin(x));
      CHKERRQ(VecAssemblyEnd(x));

      CHKERRMPI(MPI_Intercomm_create(subcomm,0,PETSC_COMM_WORLD/*peer_comm*/,1,100/*tag*/,&intercomm));

      /* Tell rank 0 of subcomm1 the global size of x */
      if (!lrank) CHKERRMPI(MPI_Send(&N,1,MPIU_INT,0/*receiver's rank in remote comm, i.e., subcomm1*/,200/*tag*/,intercomm));

      /* Create an intracomm Petsc can work on. Ranks in subcomm0 are ordered before ranks in subcomm1 in parentcomm.
        But this order actually does not matter, since what we care is vector y, which is defined on subcomm1.
      */
      CHKERRMPI(MPI_Intercomm_merge(intercomm,0/*low*/,&parentcomm));

      /* Create a vector xg on parentcomm, which shares memory with x */
      CHKERRQ(VecGetLocalSize(x,&n));
      if (iscuda) {
        #if defined(PETSC_HAVE_CUDA)
          CHKERRQ(VecCUDAGetArrayRead(x,&xvalue));
          CHKERRQ(VecCreateMPICUDAWithArray(parentcomm,1,n,N,xvalue,&xg));
        #endif
      } else {
        CHKERRQ(VecGetArrayRead(x,&xvalue));
        CHKERRQ(VecCreateMPIWithArray(parentcomm,1,n,N,xvalue,&xg));
      }

      /* Ranks in subcomm 0 have nothing on yg, so they simply have n=0, array=NULL */
      if (iscuda) {
        #if defined(PETSC_HAVE_CUDA)
          CHKERRQ(VecCreateMPICUDAWithArray(parentcomm,1,0/*n*/,N,NULL/*array*/,&yg));
        #endif
      } else {
        CHKERRQ(VecCreateMPIWithArray(parentcomm,1,0/*n*/,N,NULL/*array*/,&yg));
      }

      /* Create the vecscatter, which does identity map by setting yg[i] = xg[i], i=0..N-1. */
      CHKERRQ(VecGetOwnershipRange(xg,&low,&high)); /* low, high are global indices of xg */
      CHKERRQ(ISCreateStride(PETSC_COMM_SELF,high-low,low,1,&ix));
      CHKERRQ(ISDuplicate(ix,&iy));
      CHKERRQ(VecScatterCreate(xg,ix,yg,iy,&vscat));

      /* Scatter values from xg to yg */
      CHKERRQ(VecScatterBegin(vscat,xg,yg,INSERT_VALUES,SCATTER_FORWARD));
      CHKERRQ(VecScatterEnd(vscat,xg,yg,INSERT_VALUES,SCATTER_FORWARD));

      /* After the VecScatter is done, xg is idle so we can safely return xvalue to x */
      if (iscuda) {
        #if defined(PETSC_HAVE_CUDA)
          CHKERRQ(VecCUDARestoreArrayRead(x,&xvalue));
        #endif
      } else {
        CHKERRQ(VecRestoreArrayRead(x,&xvalue));
      }
      CHKERRQ(VecDestroy(&x));
      CHKERRQ(ISDestroy(&ix));
      CHKERRQ(ISDestroy(&iy));
      CHKERRQ(VecDestroy(&xg));
      CHKERRQ(VecDestroy(&yg));
      CHKERRQ(VecScatterDestroy(&vscat));
      CHKERRMPI(MPI_Comm_free(&intercomm));
      CHKERRMPI(MPI_Comm_free(&parentcomm));
    } else if (mycolor == 1) { /* subcomm 1, containing ranks 1, 4, 7, ... in PETSC_COMM_WORLD */
      PetscInt    n,N;
      Vec         y,xg,yg;
      IS          ix,iy;
      VecScatter  vscat;
      PetscScalar *yvalue;
      MPI_Comm    intercomm,parentcomm;
      PetscMPIInt lrank;

      CHKERRMPI(MPI_Comm_rank(subcomm,&lrank));
      CHKERRMPI(MPI_Intercomm_create(subcomm,0,PETSC_COMM_WORLD/*peer_comm*/,0/*remote_leader*/,100/*tag*/,&intercomm));

      /* Two rank-0 are talking */
      if (!lrank) CHKERRMPI(MPI_Recv(&N,1,MPIU_INT,0/*sender's rank in remote comm, i.e. subcomm0*/,200/*tag*/,intercomm,MPI_STATUS_IGNORE));
      /* Rank 0 of subcomm1 bcasts N to its members */
      CHKERRMPI(MPI_Bcast(&N,1,MPIU_INT,0/*local root*/,subcomm));

      /* Create a intracomm Petsc can work on */
      CHKERRMPI(MPI_Intercomm_merge(intercomm,1/*high*/,&parentcomm));

      /* Ranks in subcomm1 have nothing on xg, so they simply have n=0, array=NULL.*/
      if (iscuda) {
        #if defined(PETSC_HAVE_CUDA)
          CHKERRQ(VecCreateMPICUDAWithArray(parentcomm,1/*bs*/,0/*n*/,N,NULL/*array*/,&xg));
        #endif
      } else {
        CHKERRQ(VecCreateMPIWithArray(parentcomm,1/*bs*/,0/*n*/,N,NULL/*array*/,&xg));
      }

      CHKERRQ(VecCreate(subcomm, &y));
      CHKERRQ(VecSetSizes(y, PETSC_DECIDE, N));
      if (iscuda) {
        CHKERRQ(VecSetType(y, VECCUDA));
      } else {
        CHKERRQ(VecSetType(y, VECSTANDARD));
      }
      CHKERRQ(VecSetUp(y));

      CHKERRQ(PetscObjectSetName((PetscObject)y,"y_subcomm_1")); /* Give a name to view y clearly */
      CHKERRQ(VecGetLocalSize(y,&n));
      if (iscuda) {
        #if defined(PETSC_HAVE_CUDA)
          CHKERRQ(VecCUDAGetArray(y,&yvalue));
        #endif
      } else {
        CHKERRQ(VecGetArray(y,&yvalue));
      }
      /* Create a vector yg on parentcomm, which shares memory with y. xg and yg must be
        created in the same order in subcomm0/1. For example, we can not reverse the order of
        creating xg and yg in subcomm1.
      */
      if (iscuda) {
        #if defined(PETSC_HAVE_CUDA)
          CHKERRQ(VecCreateMPICUDAWithArray(parentcomm,1/*bs*/,n,N,yvalue,&yg));
        #endif
      } else {
        CHKERRQ(VecCreateMPIWithArray(parentcomm,1/*bs*/,n,N,yvalue,&yg));
      }

      /* Ranks in subcomm0 already specified the full range of the identity map.
        ranks in subcomm1 just need to create empty ISes to cheat VecScatterCreate.
      */
      CHKERRQ(ISCreateStride(PETSC_COMM_SELF,0,0,1,&ix));
      CHKERRQ(ISDuplicate(ix,&iy));
      CHKERRQ(VecScatterCreate(xg,ix,yg,iy,&vscat));

      /* Scatter values from xg to yg */
      CHKERRQ(VecScatterBegin(vscat,xg,yg,INSERT_VALUES,SCATTER_FORWARD));
      CHKERRQ(VecScatterEnd(vscat,xg,yg,INSERT_VALUES,SCATTER_FORWARD));

      /* After the VecScatter is done, values in yg are available. y is our interest, so we return yvalue to y */
      if (iscuda) {
        #if defined(PETSC_HAVE_CUDA)
          CHKERRQ(VecCUDARestoreArray(y,&yvalue));
        #endif
      } else {
        CHKERRQ(VecRestoreArray(y,&yvalue));
      }

      /* Libraries on subcomm1 can safely use y now, for example, view it */
      CHKERRQ(VecView(y,PETSC_VIEWER_STDOUT_(subcomm)));

      CHKERRQ(VecDestroy(&y));
      CHKERRQ(ISDestroy(&ix));
      CHKERRQ(ISDestroy(&iy));
      CHKERRQ(VecDestroy(&xg));
      CHKERRQ(VecDestroy(&yg));
      CHKERRQ(VecScatterDestroy(&vscat));
      CHKERRMPI(MPI_Comm_free(&intercomm));
      CHKERRMPI(MPI_Comm_free(&parentcomm));
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
    CHKERRQ(VecCreate(PETSC_COMM_WORLD, &x));
    CHKERRQ(VecSetSizes(x, PETSC_DECIDE, N));
    if (iscuda) {
      CHKERRQ(VecSetType(x, VECCUDA));
    } else {
      CHKERRQ(VecSetType(x, VECSTANDARD));
    }
    CHKERRQ(VecSetUp(x));
    CHKERRQ(VecGetOwnershipRange(x,&low,&high));
    for (i=low; i<high; i++) CHKERRQ(VecSetValue(x,i,(PetscScalar)i,INSERT_VALUES));
    CHKERRQ(VecAssemblyBegin(x));
    CHKERRQ(VecAssemblyEnd(x));

    /* Every subcomm has a y as long as x */
    CHKERRQ(VecCreate(subcomm, &y));
    CHKERRQ(VecSetSizes(y, PETSC_DECIDE, N));
    if (iscuda) {
      CHKERRQ(VecSetType(y, VECCUDA));
    } else {
      CHKERRQ(VecSetType(y, VECSTANDARD));
    }
    CHKERRQ(VecSetUp(y));
    CHKERRQ(VecGetLocalSize(y,&n));

    /* Create a global vector yg on PETSC_COMM_WORLD using y's memory. yg's global size = N*(number of subcommunicators).
       Eeach rank in subcomms contributes a piece to construct the global yg. Keep in mind that pieces from a subcomm are not
       necessarily consecutive in yg. That depends on how PETSC_COMM_WORLD is split. In our case, subcomm0 is made of rank
       0, 3, 6 etc from PETSC_COMM_WORLD. So subcomm0's pieces are interleaved with pieces from other subcomms in yg.
    */
    if (iscuda) {
      #if defined(PETSC_HAVE_CUDA)
        CHKERRQ(VecCUDAGetArray(y,&yvalue));
        CHKERRQ(VecCreateMPICUDAWithArray(PETSC_COMM_WORLD,1,n,PETSC_DECIDE,yvalue,&yg));
      #endif
    } else {
      CHKERRQ(VecGetArray(y,&yvalue));
      CHKERRQ(VecCreateMPIWithArray(PETSC_COMM_WORLD,1,n,PETSC_DECIDE,yvalue,&yg));
    }
    CHKERRQ(PetscObjectSetName((PetscObject)yg,"yg_on_subcomms")); /* Give a name to view yg clearly */

    /* The following two lines are key. From xstart, we know where to pull entries from x. Note that we get xstart from y,
       since first entry of y on this rank is from x[xstart]. From ystart, we know where ot put entries to yg.
     */
    CHKERRQ(VecGetOwnershipRange(y,&xstart,NULL));
    CHKERRQ(VecGetOwnershipRange(yg,&ystart,NULL));

    CHKERRQ(ISCreateStride(PETSC_COMM_SELF,n,xstart,1,&ix));
    CHKERRQ(ISCreateStride(PETSC_COMM_SELF,n,ystart,1,&iy));
    CHKERRQ(VecScatterCreate(x,ix,yg,iy,&vscat));
    CHKERRQ(VecScatterBegin(vscat,x,yg,INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecScatterEnd(vscat,x,yg,INSERT_VALUES,SCATTER_FORWARD));

    /* View yg on PETSC_COMM_WORLD before destroying it. We shall see the interleaving effect in output. */
    CHKERRQ(VecView(yg,PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(VecDestroy(&yg));

    /* Restory yvalue so that processes in subcomm can use y from now on. */
    if (iscuda) {
      #if defined(PETSC_HAVE_CUDA)
        CHKERRQ(VecCUDARestoreArray(y,&yvalue));
      #endif
    } else {
      CHKERRQ(VecRestoreArray(y,&yvalue));
    }
    CHKERRQ(VecScale(y,3.0));

    CHKERRQ(ISDestroy(&ix)); /* One can also destroy ix, iy immediately after VecScatterCreate() */
    CHKERRQ(ISDestroy(&iy));
    CHKERRQ(VecDestroy(&x));
    CHKERRQ(VecDestroy(&y));
    CHKERRQ(VecScatterDestroy(&vscat));
  } /* world2subs */

  CHKERRMPI(MPI_Comm_free(&subcomm));
  CHKERRQ(PetscFinalize());
  return 0;
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
