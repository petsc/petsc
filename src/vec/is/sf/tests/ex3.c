static char help[]= "Test PetscSFFetchAndOp on patterned SF graphs. PetscSFFetchAndOp internally uses PetscSFBcastAndOp \n\
 and PetscSFReduce. So it is a good test to see if they all work for patterned graphs.\n\
 Run with ./prog -op [replace | sum]\n\n";

#include <petscvec.h>
#include <petscsf.h>
int main(int argc,char **argv)
{
  PetscInt       i,N=10,low,high,nleaves;
  PetscMPIInt    size,rank;
  Vec            x,y,y2,gy2;
  PetscScalar    *rootdata,*leafdata,*leafupdate;
  PetscLayout    layout;
  PetscSF        gathersf,allgathersf,alltoallsf;
  MPI_Op         op=MPI_SUM;
  char           opname[64];
  const char     *mpiopname;
  PetscBool      flag,isreplace,issum;

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  PetscCall(PetscOptionsGetString(NULL,NULL,"-op",opname,sizeof(opname),&flag));
  PetscCall(PetscStrcmp(opname,"replace",&isreplace));
  PetscCall(PetscStrcmp(opname,"sum",&issum));

  if (isreplace)  {op = MPI_REPLACE; mpiopname = "MPI_REPLACE";}
  else if (issum) {op = MPIU_SUM;     mpiopname = "MPI_SUM";}
  else SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Unsupported argument (%s) to -op, which must be 'replace' or 'sum'",opname);

  PetscCall(VecCreate(PETSC_COMM_WORLD,&x));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecSetSizes(x,PETSC_DECIDE,N));

  /*-------------------------------------*/
  /*       PETSCSF_PATTERN_GATHER        */
  /*-------------------------------------*/

  /* set MPI vec x to [1, 2, .., N] */
  PetscCall(VecGetOwnershipRange(x,&low,&high));
  for (i=low; i<high; i++) PetscCall(VecSetValue(x,i,(PetscScalar)i+1.0,INSERT_VALUES));
  PetscCall(VecAssemblyBegin(x));
  PetscCall(VecAssemblyEnd(x));

  /* Create the gather SF */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\nTesting PetscSFFetchAndOp on a PETSCSF_PATTERN_GATHER graph with op = %s\n",mpiopname));
  PetscCall(VecGetLayout(x,&layout));
  PetscCall(PetscSFCreate(PETSC_COMM_WORLD,&gathersf));
  PetscCall(PetscSFSetGraphWithPattern(gathersf,layout,PETSCSF_PATTERN_GATHER));

  /* Create the leaf vector y (seq vector) and its duplicate y2 working as leafupdate */
  PetscCall(PetscSFGetGraph(gathersf,NULL,&nleaves,NULL,NULL));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF,nleaves,&y));
  PetscCall(VecDuplicate(y,&y2));

  PetscCall(VecGetArray(x,&rootdata));
  PetscCall(VecGetArray(y,&leafdata));
  PetscCall(VecGetArray(y2,&leafupdate));

  /* Bcast x to y,to initialize y = [1,N], then scale y to make leafupdate = y = [2,2*N] */
  PetscCall(PetscSFBcastBegin(gathersf,MPIU_SCALAR,rootdata,leafdata,MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(gathersf,MPIU_SCALAR,rootdata,leafdata,MPI_REPLACE));
  PetscCall(VecRestoreArray(y,&leafdata));
  PetscCall(VecScale(y,2));
  PetscCall(VecGetArray(y,&leafdata));

  /* FetchAndOp x to y */
  PetscCall(PetscSFFetchAndOpBegin(gathersf,MPIU_SCALAR,rootdata,leafdata,leafupdate,op));
  PetscCall(PetscSFFetchAndOpEnd(gathersf,MPIU_SCALAR,rootdata,leafdata,leafupdate,op));

  /* View roots (x) and leafupdate (y2). Since this is a gather graph, leafudpate = rootdata = [1,N], then rootdata += leafdata, i.e., [3,3*N] */
  PetscCall(VecCreateMPIWithArray(PETSC_COMM_WORLD,1,nleaves,PETSC_DECIDE,leafupdate,&gy2));
  PetscCall(PetscObjectSetName((PetscObject)x,"rootdata"));
  PetscCall(PetscObjectSetName((PetscObject)gy2,"leafupdate"));

  PetscCall(VecView(x,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(VecView(gy2,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(VecDestroy(&gy2));

  PetscCall(VecRestoreArray(y2,&leafupdate));
  PetscCall(VecDestroy(&y2));

  PetscCall(VecRestoreArray(y,&leafdata));
  PetscCall(VecDestroy(&y));

  PetscCall(VecRestoreArray(x,&rootdata));
  /* PetscCall(VecDestroy(&x)); */ /* We will reuse x in ALLGATHER, so do not destroy it */

  PetscCall(PetscSFDestroy(&gathersf));

  /*-------------------------------------*/
  /*       PETSCSF_PATTERN_ALLGATHER     */
  /*-------------------------------------*/

  /* set MPI vec x to [1, 2, .., N] */
  for (i=low; i<high; i++) PetscCall(VecSetValue(x,i,(PetscScalar)i+1.0,INSERT_VALUES));
  PetscCall(VecAssemblyBegin(x));
  PetscCall(VecAssemblyEnd(x));

  /* Create the allgather SF */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\nTesting PetscSFFetchAndOp on a PETSCSF_PATTERN_ALLGATHER graph with op = %s\n",mpiopname));
  PetscCall(VecGetLayout(x,&layout));
  PetscCall(PetscSFCreate(PETSC_COMM_WORLD,&allgathersf));
  PetscCall(PetscSFSetGraphWithPattern(allgathersf,layout,PETSCSF_PATTERN_ALLGATHER));

  /* Create the leaf vector y (seq vector) and its duplicate y2 working as leafupdate */
  PetscCall(PetscSFGetGraph(allgathersf,NULL,&nleaves,NULL,NULL));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF,nleaves,&y));
  PetscCall(VecDuplicate(y,&y2));

  PetscCall(VecGetArray(x,&rootdata));
  PetscCall(VecGetArray(y,&leafdata));
  PetscCall(VecGetArray(y2,&leafupdate));

  /* Bcast x to y, to initialize y = [1,N], then scale y to make leafupdate = y = [2,2*N] */
  PetscCall(PetscSFBcastBegin(allgathersf,MPIU_SCALAR,rootdata,leafdata,MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(allgathersf,MPIU_SCALAR,rootdata,leafdata,MPI_REPLACE));
  PetscCall(VecRestoreArray(y,&leafdata));
  PetscCall(VecScale(y,2));
  PetscCall(VecGetArray(y,&leafdata));

  /* FetchAndOp x to y */
  PetscCall(PetscSFFetchAndOpBegin(allgathersf,MPIU_SCALAR,rootdata,leafdata,leafupdate,op));
  PetscCall(PetscSFFetchAndOpEnd(allgathersf,MPIU_SCALAR,rootdata,leafdata,leafupdate,op));

  /* View roots (x) and leafupdate (y2). Since this is an allgather graph, we have (suppose ranks get updates in ascending order)
     rank 0: leafupdate = rootdata = [1,N],   rootdata += leafdata = [3,3*N]
     rank 1: leafupdate = rootdata = [3,3*N], rootdata += leafdata = [5,5*N]
     rank 2: leafupdate = rootdata = [5,5*N], rootdata += leafdata = [7,7*N]
     ...
   */
  PetscCall(VecCreateMPIWithArray(PETSC_COMM_WORLD,1,nleaves,PETSC_DECIDE,leafupdate,&gy2));
  PetscCall(PetscObjectSetName((PetscObject)x,"rootdata"));
  PetscCall(PetscObjectSetName((PetscObject)gy2,"leafupdate"));

  PetscCall(VecView(x,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(VecView(gy2,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(VecDestroy(&gy2));

  PetscCall(VecRestoreArray(y2,&leafupdate));
  PetscCall(VecDestroy(&y2));

  PetscCall(VecRestoreArray(y,&leafdata));
  PetscCall(VecDestroy(&y));

  PetscCall(VecRestoreArray(x,&rootdata));
  PetscCall(VecDestroy(&x)); /* We won't reuse x in ALLGATHER, so destroy it */

  PetscCall(PetscSFDestroy(&allgathersf));

  /*-------------------------------------*/
  /*       PETSCSF_PATTERN_ALLTOALL     */
  /*-------------------------------------*/

  PetscCall(VecCreate(PETSC_COMM_WORLD,&x));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecSetSizes(x,size,PETSC_DECIDE));

  /* set MPI vec x to [1, 2, .., size^2] */
  PetscCall(VecGetOwnershipRange(x,&low,&high));
  for (i=low; i<high; i++) PetscCall(VecSetValue(x,i,(PetscScalar)i+1.0,INSERT_VALUES));
  PetscCall(VecAssemblyBegin(x));
  PetscCall(VecAssemblyEnd(x));

/* Create the alltoall SF */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\nTesting PetscSFFetchAndOp on a PETSCSF_PATTERN_ALLTOALL graph with op = %s\n",mpiopname));
  PetscCall(PetscSFCreate(PETSC_COMM_WORLD,&alltoallsf));
  PetscCall(PetscSFSetGraphWithPattern(alltoallsf,NULL/*insignificant*/,PETSCSF_PATTERN_ALLTOALL));

  /* Create the leaf vector y (seq vector) and its duplicate y2 working as leafupdate */
  PetscCall(PetscSFGetGraph(alltoallsf,NULL,&nleaves,NULL,NULL));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF,nleaves,&y));
  PetscCall(VecDuplicate(y,&y2));

  PetscCall(VecGetArray(x,&rootdata));
  PetscCall(VecGetArray(y,&leafdata));
  PetscCall(VecGetArray(y2,&leafupdate));

  /* Bcast x to y, to initialize y = 1+rank+size*i, with i=0..size-1 */
  PetscCall(PetscSFBcastBegin(alltoallsf,MPIU_SCALAR,rootdata,leafdata,MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(alltoallsf,MPIU_SCALAR,rootdata,leafdata,MPI_REPLACE));

  /* FetchAndOp x to y */
  PetscCall(PetscSFFetchAndOpBegin(alltoallsf,MPIU_SCALAR,rootdata,leafdata,leafupdate,op));
  PetscCall(PetscSFFetchAndOpEnd(alltoallsf,MPIU_SCALAR,rootdata,leafdata,leafupdate,op));

  /* View roots (x) and leafupdate (y2). Since this is an alltoall graph, each root has only one leaf.
     So, leafupdate = rootdata = 1+rank+size*i, i=0..size-1; and rootdata += leafdata, i.e., rootdata = [2,2*N]
   */
  PetscCall(VecCreateMPIWithArray(PETSC_COMM_WORLD,1,nleaves,PETSC_DECIDE,leafupdate,&gy2));
  PetscCall(PetscObjectSetName((PetscObject)x,"rootdata"));
  PetscCall(PetscObjectSetName((PetscObject)gy2,"leafupdate"));

  PetscCall(VecView(x,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(VecView(gy2,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(VecDestroy(&gy2));

  PetscCall(VecRestoreArray(y2,&leafupdate));
  PetscCall(VecDestroy(&y2));

  PetscCall(VecRestoreArray(y,&leafdata));
  PetscCall(VecDestroy(&y));

  PetscCall(VecRestoreArray(x,&rootdata));
  PetscCall(VecDestroy(&x));

  PetscCall(PetscSFDestroy(&alltoallsf));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      # N=10 is divisible by nsize, to trigger Allgather/Gather in SF
      #MPI_Sendrecv_replace is broken with 20210400300
      requires: !defined(PETSC_HAVE_I_MPI_NUMVERSION)
      nsize: 2
      args: -op replace

   test:
      suffix: 2
      nsize: 2
      args: -op sum

   # N=10 is not divisible by nsize, to trigger Allgatherv/Gatherv in SF
   test:
      #MPI_Sendrecv_replace is broken with 20210400300
      requires: !defined(PETSC_HAVE_I_MPI_NUMVERSION)
      suffix: 3
      nsize: 3
      args: -op replace

   test:
      suffix: 4
      nsize: 3
      args: -op sum

TEST*/
