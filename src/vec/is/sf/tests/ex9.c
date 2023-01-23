static char help[] = "This example shows 1) how to transfer vectors from a parent communicator to vectors on a child communicator and vice versa;\n\
  2) how to transfer vectors from a subcommunicator to vectors on another subcommunicator. The two subcommunicators are not\n\
  required to cover all processes in PETSC_COMM_WORLD; 3) how to copy a vector from a parent communicator to vectors on its child communicators.\n\
  To run any example with VECCUDA vectors, add -vectype cuda to the argument list\n\n";

#include <petscvec.h>
int main(int argc, char **argv)
{
  PetscMPIInt nproc, grank, mycolor;
  PetscInt    i, n, N = 20, low, high;
  MPI_Comm    subcomm;
  Vec         x  = NULL; /* global vectors on PETSC_COMM_WORLD */
  Vec         yg = NULL; /* global vectors on PETSC_COMM_WORLD */
  VecScatter  vscat;
  IS          ix, iy;
  PetscBool   iscuda = PETSC_FALSE; /* Option to use VECCUDA vectors */
  PetscBool   optionflag, compareflag;
  char        vectypename[PETSC_MAX_PATH_LEN];
  PetscBool   world2sub  = PETSC_FALSE; /* Copy a vector from WORLD to a subcomm? */
  PetscBool   sub2sub    = PETSC_FALSE; /* Copy a vector from a subcomm to another subcomm? */
  PetscBool   world2subs = PETSC_FALSE; /* Copy a vector from WORLD to multiple subcomms? */

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &nproc));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &grank));

  PetscCheck(nproc >= 2, PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ, "This test must have at least two processes to run");

  PetscCall(PetscOptionsGetBool(NULL, 0, "-world2sub", &world2sub, NULL));
  PetscCall(PetscOptionsGetBool(NULL, 0, "-sub2sub", &sub2sub, NULL));
  PetscCall(PetscOptionsGetBool(NULL, 0, "-world2subs", &world2subs, NULL));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-vectype", vectypename, sizeof(vectypename), &optionflag));
  if (optionflag) {
    PetscCall(PetscStrncmp(vectypename, "cuda", (size_t)4, &compareflag));
    if (compareflag) iscuda = PETSC_TRUE;
  }

  /* Split PETSC_COMM_WORLD into three subcomms. Each process can only see the subcomm it belongs to */
  mycolor = grank % 3;
  PetscCallMPI(MPI_Comm_split(PETSC_COMM_WORLD, mycolor, grank, &subcomm));

  /*===========================================================================
   *  Transfer a vector x defined on PETSC_COMM_WORLD to a vector y defined on
   *  a subcommunicator of PETSC_COMM_WORLD and vice versa.
   *===========================================================================*/
  if (world2sub) {
    PetscCall(VecCreate(PETSC_COMM_WORLD, &x));
    PetscCall(VecSetSizes(x, PETSC_DECIDE, N));
    if (iscuda) {
      PetscCall(VecSetType(x, VECCUDA));
    } else {
      PetscCall(VecSetType(x, VECSTANDARD));
    }
    PetscCall(VecSetUp(x));
    PetscCall(PetscObjectSetName((PetscObject)x, "x_commworld")); /* Give a name to view x clearly */

    /* Initialize x to [-0.0, -1.0, -2.0, ..., -19.0] */
    PetscCall(VecGetOwnershipRange(x, &low, &high));
    for (i = low; i < high; i++) {
      PetscScalar val = -i;
      PetscCall(VecSetValue(x, i, val, INSERT_VALUES));
    }
    PetscCall(VecAssemblyBegin(x));
    PetscCall(VecAssemblyEnd(x));

    /* Transfer x to a vector y only defined on subcomm0 and vice versa */
    if (mycolor == 0) { /* subcomm0 contains ranks 0, 3, 6, ... in PETSC_COMM_WORLD */
      Vec          y;
      PetscScalar *yvalue;
      PetscCall(VecCreate(subcomm, &y));
      PetscCall(VecSetSizes(y, PETSC_DECIDE, N));
      if (iscuda) {
        PetscCall(VecSetType(y, VECCUDA));
      } else {
        PetscCall(VecSetType(y, VECSTANDARD));
      }
      PetscCall(VecSetUp(y));
      PetscCall(PetscObjectSetName((PetscObject)y, "y_subcomm_0")); /* Give a name to view y clearly */
      PetscCall(VecGetLocalSize(y, &n));
      if (iscuda) {
#if defined(PETSC_HAVE_CUDA)
        PetscCall(VecCUDAGetArray(y, &yvalue));
#endif
      } else {
        PetscCall(VecGetArray(y, &yvalue));
      }
      /* Create yg on PETSC_COMM_WORLD and alias yg with y. They share the memory pointed by yvalue.
        Note this is a collective call. All processes have to call it and supply consistent N.
      */
      if (iscuda) {
#if defined(PETSC_HAVE_CUDA)
        PetscCall(VecCreateMPICUDAWithArray(PETSC_COMM_WORLD, 1, n, N, yvalue, &yg));
#endif
      } else {
        PetscCall(VecCreateMPIWithArray(PETSC_COMM_WORLD, 1, n, N, yvalue, &yg));
      }

      /* Create an identity map that makes yg[i] = x[i], i=0..N-1 */
      PetscCall(VecGetOwnershipRange(yg, &low, &high)); /* low, high are global indices */
      PetscCall(ISCreateStride(PETSC_COMM_SELF, high - low, low, 1, &ix));
      PetscCall(ISDuplicate(ix, &iy));

      /* Union of ix's on subcomm0 covers the full range of [0,N) */
      PetscCall(VecScatterCreate(x, ix, yg, iy, &vscat));
      PetscCall(VecScatterBegin(vscat, x, yg, INSERT_VALUES, SCATTER_FORWARD));
      PetscCall(VecScatterEnd(vscat, x, yg, INSERT_VALUES, SCATTER_FORWARD));

      /* Once yg got the data from x, we return yvalue to y so that we can use y in other operations.
        VecGetArray must be paired with VecRestoreArray.
      */
      if (iscuda) {
#if defined(PETSC_HAVE_CUDA)
        PetscCall(VecCUDARestoreArray(y, &yvalue));
#endif
      } else {
        PetscCall(VecRestoreArray(y, &yvalue));
      }

      /* Libraries on subcomm0 can safely use y now, for example, view and scale it */
      PetscCall(VecView(y, PETSC_VIEWER_STDOUT_(subcomm)));
      PetscCall(VecScale(y, 2.0));

      /* Send the new y back to x */
      PetscCall(VecGetArray(y, &yvalue)); /* If VecScale is done on GPU, Petsc will prepare a valid yvalue for access */
      /* Supply new yvalue to yg without memory copying */
      PetscCall(VecPlaceArray(yg, yvalue));
      PetscCall(VecScatterBegin(vscat, yg, x, INSERT_VALUES, SCATTER_REVERSE));
      PetscCall(VecScatterEnd(vscat, yg, x, INSERT_VALUES, SCATTER_REVERSE));
      PetscCall(VecResetArray(yg));
      PetscCall(VecRestoreArray(y, &yvalue));
      PetscCall(VecDestroy(&y));
    } else {
      /* Ranks outside of subcomm0 do not supply values to yg */
      if (iscuda) {
#if defined(PETSC_HAVE_CUDA)
        PetscCall(VecCreateMPICUDAWithArray(PETSC_COMM_WORLD, 1, 0 /*n*/, N, NULL, &yg));
#endif
      } else {
        PetscCall(VecCreateMPIWithArray(PETSC_COMM_WORLD, 1, 0 /*n*/, N, NULL, &yg));
      }

      /* Ranks in subcomm0 already specified the full range of the identity map. The remaining
        ranks just need to create empty ISes to cheat VecScatterCreate.
      */
      PetscCall(ISCreateStride(PETSC_COMM_SELF, 0, 0, 1, &ix));
      PetscCall(ISDuplicate(ix, &iy));

      PetscCall(VecScatterCreate(x, ix, yg, iy, &vscat));
      PetscCall(VecScatterBegin(vscat, x, yg, INSERT_VALUES, SCATTER_FORWARD));
      PetscCall(VecScatterEnd(vscat, x, yg, INSERT_VALUES, SCATTER_FORWARD));

      /* Send the new y back to x. Ranks outside of subcomm0 actually have nothing to send.
        But they have to call VecScatterBegin/End since these routines are collective.
      */
      PetscCall(VecScatterBegin(vscat, yg, x, INSERT_VALUES, SCATTER_REVERSE));
      PetscCall(VecScatterEnd(vscat, yg, x, INSERT_VALUES, SCATTER_REVERSE));
    }

    PetscCall(VecView(x, PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(ISDestroy(&ix));
    PetscCall(ISDestroy(&iy));
    PetscCall(VecDestroy(&x));
    PetscCall(VecDestroy(&yg));
    PetscCall(VecScatterDestroy(&vscat));
  } /* world2sub */

  /*===========================================================================
   *  Transfer a vector x defined on subcomm0 to a vector y defined on
   *  subcomm1. The two subcomms are not overlapping and their union is
   *  not necessarily equal to PETSC_COMM_WORLD.
   *===========================================================================*/
  if (sub2sub) {
    if (mycolor == 0) {
      /* Intentionally declare N as a local variable so that processes in subcomm1 do not know its value */
      PetscInt           n, N = 22;
      Vec                x, xg, yg;
      IS                 ix, iy;
      VecScatter         vscat;
      const PetscScalar *xvalue;
      MPI_Comm           intercomm, parentcomm;
      PetscMPIInt        lrank;

      PetscCallMPI(MPI_Comm_rank(subcomm, &lrank));
      /* x is on subcomm */
      PetscCall(VecCreate(subcomm, &x));
      PetscCall(VecSetSizes(x, PETSC_DECIDE, N));
      if (iscuda) {
        PetscCall(VecSetType(x, VECCUDA));
      } else {
        PetscCall(VecSetType(x, VECSTANDARD));
      }
      PetscCall(VecSetUp(x));
      PetscCall(VecGetOwnershipRange(x, &low, &high));

      /* initialize x = [0.0, 1.0, 2.0, ..., 21.0] */
      for (i = low; i < high; i++) {
        PetscScalar val = i;
        PetscCall(VecSetValue(x, i, val, INSERT_VALUES));
      }
      PetscCall(VecAssemblyBegin(x));
      PetscCall(VecAssemblyEnd(x));

      PetscCallMPI(MPI_Intercomm_create(subcomm, 0, PETSC_COMM_WORLD /*peer_comm*/, 1, 100 /*tag*/, &intercomm));

      /* Tell rank 0 of subcomm1 the global size of x */
      if (!lrank) PetscCallMPI(MPI_Send(&N, 1, MPIU_INT, 0 /*receiver's rank in remote comm, i.e., subcomm1*/, 200 /*tag*/, intercomm));

      /* Create an intracomm Petsc can work on. Ranks in subcomm0 are ordered before ranks in subcomm1 in parentcomm.
        But this order actually does not matter, since what we care is vector y, which is defined on subcomm1.
      */
      PetscCallMPI(MPI_Intercomm_merge(intercomm, 0 /*low*/, &parentcomm));

      /* Create a vector xg on parentcomm, which shares memory with x */
      PetscCall(VecGetLocalSize(x, &n));
      if (iscuda) {
#if defined(PETSC_HAVE_CUDA)
        PetscCall(VecCUDAGetArrayRead(x, &xvalue));
        PetscCall(VecCreateMPICUDAWithArray(parentcomm, 1, n, N, xvalue, &xg));
#endif
      } else {
        PetscCall(VecGetArrayRead(x, &xvalue));
        PetscCall(VecCreateMPIWithArray(parentcomm, 1, n, N, xvalue, &xg));
      }

      /* Ranks in subcomm 0 have nothing on yg, so they simply have n=0, array=NULL */
      if (iscuda) {
#if defined(PETSC_HAVE_CUDA)
        PetscCall(VecCreateMPICUDAWithArray(parentcomm, 1, 0 /*n*/, N, NULL /*array*/, &yg));
#endif
      } else {
        PetscCall(VecCreateMPIWithArray(parentcomm, 1, 0 /*n*/, N, NULL /*array*/, &yg));
      }

      /* Create the vecscatter, which does identity map by setting yg[i] = xg[i], i=0..N-1. */
      PetscCall(VecGetOwnershipRange(xg, &low, &high)); /* low, high are global indices of xg */
      PetscCall(ISCreateStride(PETSC_COMM_SELF, high - low, low, 1, &ix));
      PetscCall(ISDuplicate(ix, &iy));
      PetscCall(VecScatterCreate(xg, ix, yg, iy, &vscat));

      /* Scatter values from xg to yg */
      PetscCall(VecScatterBegin(vscat, xg, yg, INSERT_VALUES, SCATTER_FORWARD));
      PetscCall(VecScatterEnd(vscat, xg, yg, INSERT_VALUES, SCATTER_FORWARD));

      /* After the VecScatter is done, xg is idle so we can safely return xvalue to x */
      if (iscuda) {
#if defined(PETSC_HAVE_CUDA)
        PetscCall(VecCUDARestoreArrayRead(x, &xvalue));
#endif
      } else {
        PetscCall(VecRestoreArrayRead(x, &xvalue));
      }
      PetscCall(VecDestroy(&x));
      PetscCall(ISDestroy(&ix));
      PetscCall(ISDestroy(&iy));
      PetscCall(VecDestroy(&xg));
      PetscCall(VecDestroy(&yg));
      PetscCall(VecScatterDestroy(&vscat));
      PetscCallMPI(MPI_Comm_free(&intercomm));
      PetscCallMPI(MPI_Comm_free(&parentcomm));
    } else if (mycolor == 1) { /* subcomm 1, containing ranks 1, 4, 7, ... in PETSC_COMM_WORLD */
      PetscInt     n, N;
      Vec          y, xg, yg;
      IS           ix, iy;
      VecScatter   vscat;
      PetscScalar *yvalue;
      MPI_Comm     intercomm, parentcomm;
      PetscMPIInt  lrank;

      PetscCallMPI(MPI_Comm_rank(subcomm, &lrank));
      PetscCallMPI(MPI_Intercomm_create(subcomm, 0, PETSC_COMM_WORLD /*peer_comm*/, 0 /*remote_leader*/, 100 /*tag*/, &intercomm));

      /* Two rank-0 are talking */
      if (!lrank) PetscCallMPI(MPI_Recv(&N, 1, MPIU_INT, 0 /*sender's rank in remote comm, i.e. subcomm0*/, 200 /*tag*/, intercomm, MPI_STATUS_IGNORE));
      /* Rank 0 of subcomm1 bcasts N to its members */
      PetscCallMPI(MPI_Bcast(&N, 1, MPIU_INT, 0 /*local root*/, subcomm));

      /* Create a intracomm Petsc can work on */
      PetscCallMPI(MPI_Intercomm_merge(intercomm, 1 /*high*/, &parentcomm));

      /* Ranks in subcomm1 have nothing on xg, so they simply have n=0, array=NULL.*/
      if (iscuda) {
#if defined(PETSC_HAVE_CUDA)
        PetscCall(VecCreateMPICUDAWithArray(parentcomm, 1 /*bs*/, 0 /*n*/, N, NULL /*array*/, &xg));
#endif
      } else {
        PetscCall(VecCreateMPIWithArray(parentcomm, 1 /*bs*/, 0 /*n*/, N, NULL /*array*/, &xg));
      }

      PetscCall(VecCreate(subcomm, &y));
      PetscCall(VecSetSizes(y, PETSC_DECIDE, N));
      if (iscuda) {
        PetscCall(VecSetType(y, VECCUDA));
      } else {
        PetscCall(VecSetType(y, VECSTANDARD));
      }
      PetscCall(VecSetUp(y));

      PetscCall(PetscObjectSetName((PetscObject)y, "y_subcomm_1")); /* Give a name to view y clearly */
      PetscCall(VecGetLocalSize(y, &n));
      if (iscuda) {
#if defined(PETSC_HAVE_CUDA)
        PetscCall(VecCUDAGetArray(y, &yvalue));
#endif
      } else {
        PetscCall(VecGetArray(y, &yvalue));
      }
      /* Create a vector yg on parentcomm, which shares memory with y. xg and yg must be
        created in the same order in subcomm0/1. For example, we can not reverse the order of
        creating xg and yg in subcomm1.
      */
      if (iscuda) {
#if defined(PETSC_HAVE_CUDA)
        PetscCall(VecCreateMPICUDAWithArray(parentcomm, 1 /*bs*/, n, N, yvalue, &yg));
#endif
      } else {
        PetscCall(VecCreateMPIWithArray(parentcomm, 1 /*bs*/, n, N, yvalue, &yg));
      }

      /* Ranks in subcomm0 already specified the full range of the identity map.
        ranks in subcomm1 just need to create empty ISes to cheat VecScatterCreate.
      */
      PetscCall(ISCreateStride(PETSC_COMM_SELF, 0, 0, 1, &ix));
      PetscCall(ISDuplicate(ix, &iy));
      PetscCall(VecScatterCreate(xg, ix, yg, iy, &vscat));

      /* Scatter values from xg to yg */
      PetscCall(VecScatterBegin(vscat, xg, yg, INSERT_VALUES, SCATTER_FORWARD));
      PetscCall(VecScatterEnd(vscat, xg, yg, INSERT_VALUES, SCATTER_FORWARD));

      /* After the VecScatter is done, values in yg are available. y is our interest, so we return yvalue to y */
      if (iscuda) {
#if defined(PETSC_HAVE_CUDA)
        PetscCall(VecCUDARestoreArray(y, &yvalue));
#endif
      } else {
        PetscCall(VecRestoreArray(y, &yvalue));
      }

      /* Libraries on subcomm1 can safely use y now, for example, view it */
      PetscCall(VecView(y, PETSC_VIEWER_STDOUT_(subcomm)));

      PetscCall(VecDestroy(&y));
      PetscCall(ISDestroy(&ix));
      PetscCall(ISDestroy(&iy));
      PetscCall(VecDestroy(&xg));
      PetscCall(VecDestroy(&yg));
      PetscCall(VecScatterDestroy(&vscat));
      PetscCallMPI(MPI_Comm_free(&intercomm));
      PetscCallMPI(MPI_Comm_free(&parentcomm));
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
    Vec          y;
    PetscInt     n, N = 15, xstart, ystart, low, high;
    PetscScalar *yvalue;

    /* Initialize x to [0, 1, 2, 3, ..., N-1] */
    PetscCall(VecCreate(PETSC_COMM_WORLD, &x));
    PetscCall(VecSetSizes(x, PETSC_DECIDE, N));
    if (iscuda) {
      PetscCall(VecSetType(x, VECCUDA));
    } else {
      PetscCall(VecSetType(x, VECSTANDARD));
    }
    PetscCall(VecSetUp(x));
    PetscCall(VecGetOwnershipRange(x, &low, &high));
    for (i = low; i < high; i++) PetscCall(VecSetValue(x, i, (PetscScalar)i, INSERT_VALUES));
    PetscCall(VecAssemblyBegin(x));
    PetscCall(VecAssemblyEnd(x));

    /* Every subcomm has a y as long as x */
    PetscCall(VecCreate(subcomm, &y));
    PetscCall(VecSetSizes(y, PETSC_DECIDE, N));
    if (iscuda) {
      PetscCall(VecSetType(y, VECCUDA));
    } else {
      PetscCall(VecSetType(y, VECSTANDARD));
    }
    PetscCall(VecSetUp(y));
    PetscCall(VecGetLocalSize(y, &n));

    /* Create a global vector yg on PETSC_COMM_WORLD using y's memory. yg's global size = N*(number of subcommunicators).
       Eeach rank in subcomms contributes a piece to construct the global yg. Keep in mind that pieces from a subcomm are not
       necessarily consecutive in yg. That depends on how PETSC_COMM_WORLD is split. In our case, subcomm0 is made of rank
       0, 3, 6 etc from PETSC_COMM_WORLD. So subcomm0's pieces are interleaved with pieces from other subcomms in yg.
    */
    if (iscuda) {
#if defined(PETSC_HAVE_CUDA)
      PetscCall(VecCUDAGetArray(y, &yvalue));
      PetscCall(VecCreateMPICUDAWithArray(PETSC_COMM_WORLD, 1, n, PETSC_DECIDE, yvalue, &yg));
#endif
    } else {
      PetscCall(VecGetArray(y, &yvalue));
      PetscCall(VecCreateMPIWithArray(PETSC_COMM_WORLD, 1, n, PETSC_DECIDE, yvalue, &yg));
    }
    PetscCall(PetscObjectSetName((PetscObject)yg, "yg_on_subcomms")); /* Give a name to view yg clearly */

    /* The following two lines are key. From xstart, we know where to pull entries from x. Note that we get xstart from y,
       since first entry of y on this rank is from x[xstart]. From ystart, we know where ot put entries to yg.
     */
    PetscCall(VecGetOwnershipRange(y, &xstart, NULL));
    PetscCall(VecGetOwnershipRange(yg, &ystart, NULL));

    PetscCall(ISCreateStride(PETSC_COMM_SELF, n, xstart, 1, &ix));
    PetscCall(ISCreateStride(PETSC_COMM_SELF, n, ystart, 1, &iy));
    PetscCall(VecScatterCreate(x, ix, yg, iy, &vscat));
    PetscCall(VecScatterBegin(vscat, x, yg, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(vscat, x, yg, INSERT_VALUES, SCATTER_FORWARD));

    /* View yg on PETSC_COMM_WORLD before destroying it. We shall see the interleaving effect in output. */
    PetscCall(VecView(yg, PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(VecDestroy(&yg));

    /* Restory yvalue so that processes in subcomm can use y from now on. */
    if (iscuda) {
#if defined(PETSC_HAVE_CUDA)
      PetscCall(VecCUDARestoreArray(y, &yvalue));
#endif
    } else {
      PetscCall(VecRestoreArray(y, &yvalue));
    }
    PetscCall(VecScale(y, 3.0));

    PetscCall(ISDestroy(&ix)); /* One can also destroy ix, iy immediately after VecScatterCreate() */
    PetscCall(ISDestroy(&iy));
    PetscCall(VecDestroy(&x));
    PetscCall(VecDestroy(&y));
    PetscCall(VecScatterDestroy(&vscat));
  } /* world2subs */

  PetscCallMPI(MPI_Comm_free(&subcomm));
  PetscCall(PetscFinalize());
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
