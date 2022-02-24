static char help[]= "Test PetscSFFetchAndOp on patterned SF graphs. PetscSFFetchAndOp internally uses PetscSFBcastAndOp \n\
 and PetscSFReduce. So it is a good test to see if they all work for patterned graphs.\n\
 Run with ./prog -op [replace | sum]\n\n";

#include <petscvec.h>
#include <petscsf.h>
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
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

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-op",opname,sizeof(opname),&flag));
  CHKERRQ(PetscStrcmp(opname,"replace",&isreplace));
  CHKERRQ(PetscStrcmp(opname,"sum",&issum));

  if (isreplace)  {op = MPI_REPLACE; mpiopname = "MPI_REPLACE";}
  else if (issum) {op = MPIU_SUM;     mpiopname = "MPI_SUM";}
  else SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Unsupported argument (%s) to -op, which must be 'replace' or 'sum'",opname);

  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&x));
  CHKERRQ(VecSetFromOptions(x));
  CHKERRQ(VecSetSizes(x,PETSC_DECIDE,N));

  /*-------------------------------------*/
  /*       PETSCSF_PATTERN_GATHER        */
  /*-------------------------------------*/

  /* set MPI vec x to [1, 2, .., N] */
  CHKERRQ(VecGetOwnershipRange(x,&low,&high));
  for (i=low; i<high; i++) CHKERRQ(VecSetValue(x,i,(PetscScalar)i+1.0,INSERT_VALUES));
  CHKERRQ(VecAssemblyBegin(x));
  CHKERRQ(VecAssemblyEnd(x));

  /* Create the gather SF */
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nTesting PetscSFFetchAndOp on a PETSCSF_PATTERN_GATHER graph with op = %s\n",mpiopname));
  CHKERRQ(VecGetLayout(x,&layout));
  CHKERRQ(PetscSFCreate(PETSC_COMM_WORLD,&gathersf));
  CHKERRQ(PetscSFSetGraphWithPattern(gathersf,layout,PETSCSF_PATTERN_GATHER));

  /* Create the leaf vector y (seq vector) and its duplicate y2 working as leafupdate */
  CHKERRQ(PetscSFGetGraph(gathersf,NULL,&nleaves,NULL,NULL));
  CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,nleaves,&y));
  CHKERRQ(VecDuplicate(y,&y2));

  CHKERRQ(VecGetArray(x,&rootdata));
  CHKERRQ(VecGetArray(y,&leafdata));
  CHKERRQ(VecGetArray(y2,&leafupdate));

  /* Bcast x to y,to initialize y = [1,N], then scale y to make leafupdate = y = [2,2*N] */
  CHKERRQ(PetscSFBcastBegin(gathersf,MPIU_SCALAR,rootdata,leafdata,MPI_REPLACE));
  CHKERRQ(PetscSFBcastEnd(gathersf,MPIU_SCALAR,rootdata,leafdata,MPI_REPLACE));
  CHKERRQ(VecRestoreArray(y,&leafdata));
  CHKERRQ(VecScale(y,2));
  CHKERRQ(VecGetArray(y,&leafdata));

  /* FetchAndOp x to y */
  CHKERRQ(PetscSFFetchAndOpBegin(gathersf,MPIU_SCALAR,rootdata,leafdata,leafupdate,op));
  CHKERRQ(PetscSFFetchAndOpEnd(gathersf,MPIU_SCALAR,rootdata,leafdata,leafupdate,op));

  /* View roots (x) and leafupdate (y2). Since this is a gather graph, leafudpate = rootdata = [1,N], then rootdata += leafdata, i.e., [3,3*N] */
  CHKERRQ(VecCreateMPIWithArray(PETSC_COMM_WORLD,1,nleaves,PETSC_DECIDE,leafupdate,&gy2));
  CHKERRQ(PetscObjectSetName((PetscObject)x,"rootdata"));
  CHKERRQ(PetscObjectSetName((PetscObject)gy2,"leafupdate"));

  CHKERRQ(VecView(x,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(VecView(gy2,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(VecDestroy(&gy2));

  CHKERRQ(VecRestoreArray(y2,&leafupdate));
  CHKERRQ(VecDestroy(&y2));

  CHKERRQ(VecRestoreArray(y,&leafdata));
  CHKERRQ(VecDestroy(&y));

  CHKERRQ(VecRestoreArray(x,&rootdata));
  /* CHKERRQ(VecDestroy(&x)); */ /* We will reuse x in ALLGATHER, so do not destroy it */

  CHKERRQ(PetscSFDestroy(&gathersf));

  /*-------------------------------------*/
  /*       PETSCSF_PATTERN_ALLGATHER     */
  /*-------------------------------------*/

  /* set MPI vec x to [1, 2, .., N] */
  for (i=low; i<high; i++) CHKERRQ(VecSetValue(x,i,(PetscScalar)i+1.0,INSERT_VALUES));
  CHKERRQ(VecAssemblyBegin(x));
  CHKERRQ(VecAssemblyEnd(x));

  /* Create the allgather SF */
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nTesting PetscSFFetchAndOp on a PETSCSF_PATTERN_ALLGATHER graph with op = %s\n",mpiopname));
  CHKERRQ(VecGetLayout(x,&layout));
  CHKERRQ(PetscSFCreate(PETSC_COMM_WORLD,&allgathersf));
  CHKERRQ(PetscSFSetGraphWithPattern(allgathersf,layout,PETSCSF_PATTERN_ALLGATHER));

  /* Create the leaf vector y (seq vector) and its duplicate y2 working as leafupdate */
  CHKERRQ(PetscSFGetGraph(allgathersf,NULL,&nleaves,NULL,NULL));
  CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,nleaves,&y));
  CHKERRQ(VecDuplicate(y,&y2));

  CHKERRQ(VecGetArray(x,&rootdata));
  CHKERRQ(VecGetArray(y,&leafdata));
  CHKERRQ(VecGetArray(y2,&leafupdate));

  /* Bcast x to y, to initialize y = [1,N], then scale y to make leafupdate = y = [2,2*N] */
  CHKERRQ(PetscSFBcastBegin(allgathersf,MPIU_SCALAR,rootdata,leafdata,MPI_REPLACE));
  CHKERRQ(PetscSFBcastEnd(allgathersf,MPIU_SCALAR,rootdata,leafdata,MPI_REPLACE));
  CHKERRQ(VecRestoreArray(y,&leafdata));
  CHKERRQ(VecScale(y,2));
  CHKERRQ(VecGetArray(y,&leafdata));

  /* FetchAndOp x to y */
  CHKERRQ(PetscSFFetchAndOpBegin(allgathersf,MPIU_SCALAR,rootdata,leafdata,leafupdate,op));
  CHKERRQ(PetscSFFetchAndOpEnd(allgathersf,MPIU_SCALAR,rootdata,leafdata,leafupdate,op));

  /* View roots (x) and leafupdate (y2). Since this is an allgather graph, we have (suppose ranks get updates in ascending order)
     rank 0: leafupdate = rootdata = [1,N],   rootdata += leafdata = [3,3*N]
     rank 1: leafupdate = rootdata = [3,3*N], rootdata += leafdata = [5,5*N]
     rank 2: leafupdate = rootdata = [5,5*N], rootdata += leafdata = [7,7*N]
     ...
   */
  CHKERRQ(VecCreateMPIWithArray(PETSC_COMM_WORLD,1,nleaves,PETSC_DECIDE,leafupdate,&gy2));
  CHKERRQ(PetscObjectSetName((PetscObject)x,"rootdata"));
  CHKERRQ(PetscObjectSetName((PetscObject)gy2,"leafupdate"));

  CHKERRQ(VecView(x,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(VecView(gy2,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(VecDestroy(&gy2));

  CHKERRQ(VecRestoreArray(y2,&leafupdate));
  CHKERRQ(VecDestroy(&y2));

  CHKERRQ(VecRestoreArray(y,&leafdata));
  CHKERRQ(VecDestroy(&y));

  CHKERRQ(VecRestoreArray(x,&rootdata));
  CHKERRQ(VecDestroy(&x)); /* We won't reuse x in ALLGATHER, so destroy it */

  CHKERRQ(PetscSFDestroy(&allgathersf));

  /*-------------------------------------*/
  /*       PETSCSF_PATTERN_ALLTOALL     */
  /*-------------------------------------*/

  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&x));
  CHKERRQ(VecSetFromOptions(x));
  CHKERRQ(VecSetSizes(x,size,PETSC_DECIDE));

  /* set MPI vec x to [1, 2, .., size^2] */
  CHKERRQ(VecGetOwnershipRange(x,&low,&high));
  for (i=low; i<high; i++) CHKERRQ(VecSetValue(x,i,(PetscScalar)i+1.0,INSERT_VALUES));
  CHKERRQ(VecAssemblyBegin(x));
  CHKERRQ(VecAssemblyEnd(x));

/* Create the alltoall SF */
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nTesting PetscSFFetchAndOp on a PETSCSF_PATTERN_ALLTOALL graph with op = %s\n",mpiopname));
  CHKERRQ(PetscSFCreate(PETSC_COMM_WORLD,&alltoallsf));
  CHKERRQ(PetscSFSetGraphWithPattern(alltoallsf,NULL/*insignificant*/,PETSCSF_PATTERN_ALLTOALL));

  /* Create the leaf vector y (seq vector) and its duplicate y2 working as leafupdate */
  CHKERRQ(PetscSFGetGraph(alltoallsf,NULL,&nleaves,NULL,NULL));
  CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,nleaves,&y));
  CHKERRQ(VecDuplicate(y,&y2));

  CHKERRQ(VecGetArray(x,&rootdata));
  CHKERRQ(VecGetArray(y,&leafdata));
  CHKERRQ(VecGetArray(y2,&leafupdate));

  /* Bcast x to y, to initialize y = 1+rank+size*i, with i=0..size-1 */
  CHKERRQ(PetscSFBcastBegin(alltoallsf,MPIU_SCALAR,rootdata,leafdata,MPI_REPLACE));
  CHKERRQ(PetscSFBcastEnd(alltoallsf,MPIU_SCALAR,rootdata,leafdata,MPI_REPLACE));

  /* FetchAndOp x to y */
  CHKERRQ(PetscSFFetchAndOpBegin(alltoallsf,MPIU_SCALAR,rootdata,leafdata,leafupdate,op));
  CHKERRQ(PetscSFFetchAndOpEnd(alltoallsf,MPIU_SCALAR,rootdata,leafdata,leafupdate,op));

  /* View roots (x) and leafupdate (y2). Since this is an alltoall graph, each root has only one leaf.
     So, leafupdate = rootdata = 1+rank+size*i, i=0..size-1; and rootdata += leafdata, i.e., rootdata = [2,2*N]
   */
  CHKERRQ(VecCreateMPIWithArray(PETSC_COMM_WORLD,1,nleaves,PETSC_DECIDE,leafupdate,&gy2));
  CHKERRQ(PetscObjectSetName((PetscObject)x,"rootdata"));
  CHKERRQ(PetscObjectSetName((PetscObject)gy2,"leafupdate"));

  CHKERRQ(VecView(x,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(VecView(gy2,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(VecDestroy(&gy2));

  CHKERRQ(VecRestoreArray(y2,&leafupdate));
  CHKERRQ(VecDestroy(&y2));

  CHKERRQ(VecRestoreArray(y,&leafdata));
  CHKERRQ(VecDestroy(&y));

  CHKERRQ(VecRestoreArray(x,&rootdata));
  CHKERRQ(VecDestroy(&x));

  CHKERRQ(PetscSFDestroy(&alltoallsf));

  ierr = PetscFinalize();
  return ierr;
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
