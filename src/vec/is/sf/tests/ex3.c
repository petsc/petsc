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
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);

  ierr = PetscOptionsGetString(NULL,NULL,"-op",opname,sizeof(opname),&flag);CHKERRQ(ierr);
  ierr = PetscStrcmp(opname,"replace",&isreplace);CHKERRQ(ierr);
  ierr = PetscStrcmp(opname,"sum",&issum);CHKERRQ(ierr);

  if (isreplace)  {op = MPI_REPLACE; mpiopname = "MPI_REPLACE";}
  else if (issum) {op = MPIU_SUM;     mpiopname = "MPI_SUM";}
  else SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Unsupported argument (%s) to -op, which must be 'replace' or 'sum'",opname);

  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,PETSC_DECIDE,N);CHKERRQ(ierr);

  /*-------------------------------------*/
  /*       PETSCSF_PATTERN_GATHER        */
  /*-------------------------------------*/

  /* set MPI vec x to [1, 2, .., N] */
  ierr = VecGetOwnershipRange(x,&low,&high);CHKERRQ(ierr);
  for (i=low; i<high; i++) {ierr = VecSetValue(x,i,(PetscScalar)i+1.0,INSERT_VALUES);CHKERRQ(ierr);}
  ierr = VecAssemblyBegin(x);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(x);CHKERRQ(ierr);

  /* Create the gather SF */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nTesting PetscSFFetchAndOp on a PETSCSF_PATTERN_GATHER graph with op = %s\n",mpiopname);CHKERRQ(ierr);
  ierr = VecGetLayout(x,&layout);CHKERRQ(ierr);
  ierr = PetscSFCreate(PETSC_COMM_WORLD,&gathersf);CHKERRQ(ierr);
  ierr = PetscSFSetGraphWithPattern(gathersf,layout,PETSCSF_PATTERN_GATHER);CHKERRQ(ierr);

  /* Create the leaf vector y (seq vector) and its duplicate y2 working as leafupdate */
  ierr = PetscSFGetGraph(gathersf,NULL,&nleaves,NULL,NULL);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,nleaves,&y);CHKERRQ(ierr);
  ierr = VecDuplicate(y,&y2);CHKERRQ(ierr);

  ierr = VecGetArray(x,&rootdata);CHKERRQ(ierr);
  ierr = VecGetArray(y,&leafdata);CHKERRQ(ierr);
  ierr = VecGetArray(y2,&leafupdate);CHKERRQ(ierr);

  /* Bcast x to y,to initialize y = [1,N], then scale y to make leafupdate = y = [2,2*N] */
  ierr = PetscSFBcastBegin(gathersf,MPIU_SCALAR,rootdata,leafdata,MPI_REPLACE);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(gathersf,MPIU_SCALAR,rootdata,leafdata,MPI_REPLACE);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&leafdata);CHKERRQ(ierr);
  ierr = VecScale(y,2);CHKERRQ(ierr);
  ierr = VecGetArray(y,&leafdata);CHKERRQ(ierr);

  /* FetchAndOp x to y */
  ierr = PetscSFFetchAndOpBegin(gathersf,MPIU_SCALAR,rootdata,leafdata,leafupdate,op);CHKERRQ(ierr);
  ierr = PetscSFFetchAndOpEnd(gathersf,MPIU_SCALAR,rootdata,leafdata,leafupdate,op);CHKERRQ(ierr);

  /* View roots (x) and leafupdate (y2). Since this is a gather graph, leafudpate = rootdata = [1,N], then rootdata += leafdata, i.e., [3,3*N] */
  ierr = VecCreateMPIWithArray(PETSC_COMM_WORLD,1,nleaves,PETSC_DECIDE,leafupdate,&gy2);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)x,"rootdata");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)gy2,"leafupdate");CHKERRQ(ierr);

  ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecView(gy2,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecDestroy(&gy2);CHKERRQ(ierr);

  ierr = VecRestoreArray(y2,&leafupdate);CHKERRQ(ierr);
  ierr = VecDestroy(&y2);CHKERRQ(ierr);

  ierr = VecRestoreArray(y,&leafdata);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);

  ierr = VecRestoreArray(x,&rootdata);CHKERRQ(ierr);
  /* ierr = VecDestroy(&x);CHKERRQ(ierr); */ /* We will reuse x in ALLGATHER, so do not destroy it */

  ierr = PetscSFDestroy(&gathersf);CHKERRQ(ierr);

  /*-------------------------------------*/
  /*       PETSCSF_PATTERN_ALLGATHER     */
  /*-------------------------------------*/

  /* set MPI vec x to [1, 2, .., N] */
  for (i=low; i<high; i++) {ierr = VecSetValue(x,i,(PetscScalar)i+1.0,INSERT_VALUES);CHKERRQ(ierr);}
  ierr = VecAssemblyBegin(x);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(x);CHKERRQ(ierr);

  /* Create the allgather SF */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nTesting PetscSFFetchAndOp on a PETSCSF_PATTERN_ALLGATHER graph with op = %s\n",mpiopname);CHKERRQ(ierr);
  ierr = VecGetLayout(x,&layout);CHKERRQ(ierr);
  ierr = PetscSFCreate(PETSC_COMM_WORLD,&allgathersf);CHKERRQ(ierr);
  ierr = PetscSFSetGraphWithPattern(allgathersf,layout,PETSCSF_PATTERN_ALLGATHER);CHKERRQ(ierr);

  /* Create the leaf vector y (seq vector) and its duplicate y2 working as leafupdate */
  ierr = PetscSFGetGraph(allgathersf,NULL,&nleaves,NULL,NULL);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,nleaves,&y);CHKERRQ(ierr);
  ierr = VecDuplicate(y,&y2);CHKERRQ(ierr);

  ierr = VecGetArray(x,&rootdata);CHKERRQ(ierr);
  ierr = VecGetArray(y,&leafdata);CHKERRQ(ierr);
  ierr = VecGetArray(y2,&leafupdate);CHKERRQ(ierr);

  /* Bcast x to y, to initialize y = [1,N], then scale y to make leafupdate = y = [2,2*N] */
  ierr = PetscSFBcastBegin(allgathersf,MPIU_SCALAR,rootdata,leafdata,MPI_REPLACE);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(allgathersf,MPIU_SCALAR,rootdata,leafdata,MPI_REPLACE);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&leafdata);CHKERRQ(ierr);
  ierr = VecScale(y,2);CHKERRQ(ierr);
  ierr = VecGetArray(y,&leafdata);CHKERRQ(ierr);

  /* FetchAndOp x to y */
  ierr = PetscSFFetchAndOpBegin(allgathersf,MPIU_SCALAR,rootdata,leafdata,leafupdate,op);CHKERRQ(ierr);
  ierr = PetscSFFetchAndOpEnd(allgathersf,MPIU_SCALAR,rootdata,leafdata,leafupdate,op);CHKERRQ(ierr);

  /* View roots (x) and leafupdate (y2). Since this is an allgather graph, we have (suppose ranks get updates in ascending order)
     rank 0: leafupdate = rootdata = [1,N],   rootdata += leafdata = [3,3*N]
     rank 1: leafupdate = rootdata = [3,3*N], rootdata += leafdata = [5,5*N]
     rank 2: leafupdate = rootdata = [5,5*N], rootdata += leafdata = [7,7*N]
     ...
   */
  ierr = VecCreateMPIWithArray(PETSC_COMM_WORLD,1,nleaves,PETSC_DECIDE,leafupdate,&gy2);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)x,"rootdata");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)gy2,"leafupdate");CHKERRQ(ierr);

  ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecView(gy2,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecDestroy(&gy2);CHKERRQ(ierr);

  ierr = VecRestoreArray(y2,&leafupdate);CHKERRQ(ierr);
  ierr = VecDestroy(&y2);CHKERRQ(ierr);

  ierr = VecRestoreArray(y,&leafdata);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);

  ierr = VecRestoreArray(x,&rootdata);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr); /* We won't reuse x in ALLGATHER, so destroy it */

  ierr = PetscSFDestroy(&allgathersf);CHKERRQ(ierr);

  /*-------------------------------------*/
  /*       PETSCSF_PATTERN_ALLTOALL     */
  /*-------------------------------------*/

  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,size,PETSC_DECIDE);CHKERRQ(ierr);

  /* set MPI vec x to [1, 2, .., size^2] */
  ierr = VecGetOwnershipRange(x,&low,&high);CHKERRQ(ierr);
  for (i=low; i<high; i++) {ierr = VecSetValue(x,i,(PetscScalar)i+1.0,INSERT_VALUES);CHKERRQ(ierr);}
  ierr = VecAssemblyBegin(x);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(x);CHKERRQ(ierr);

/* Create the alltoall SF */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nTesting PetscSFFetchAndOp on a PETSCSF_PATTERN_ALLTOALL graph with op = %s\n",mpiopname);CHKERRQ(ierr);
  ierr = PetscSFCreate(PETSC_COMM_WORLD,&alltoallsf);CHKERRQ(ierr);
  ierr = PetscSFSetGraphWithPattern(alltoallsf,NULL/*insignificant*/,PETSCSF_PATTERN_ALLTOALL);CHKERRQ(ierr);

  /* Create the leaf vector y (seq vector) and its duplicate y2 working as leafupdate */
  ierr = PetscSFGetGraph(alltoallsf,NULL,&nleaves,NULL,NULL);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,nleaves,&y);CHKERRQ(ierr);
  ierr = VecDuplicate(y,&y2);CHKERRQ(ierr);

  ierr = VecGetArray(x,&rootdata);CHKERRQ(ierr);
  ierr = VecGetArray(y,&leafdata);CHKERRQ(ierr);
  ierr = VecGetArray(y2,&leafupdate);CHKERRQ(ierr);

  /* Bcast x to y, to initialize y = 1+rank+size*i, with i=0..size-1 */
  ierr = PetscSFBcastBegin(alltoallsf,MPIU_SCALAR,rootdata,leafdata,MPI_REPLACE);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(alltoallsf,MPIU_SCALAR,rootdata,leafdata,MPI_REPLACE);CHKERRQ(ierr);

  /* FetchAndOp x to y */
  ierr = PetscSFFetchAndOpBegin(alltoallsf,MPIU_SCALAR,rootdata,leafdata,leafupdate,op);CHKERRQ(ierr);
  ierr = PetscSFFetchAndOpEnd(alltoallsf,MPIU_SCALAR,rootdata,leafdata,leafupdate,op);CHKERRQ(ierr);

  /* View roots (x) and leafupdate (y2). Since this is an alltoall graph, each root has only one leaf.
     So, leafupdate = rootdata = 1+rank+size*i, i=0..size-1; and rootdata += leafdata, i.e., rootdata = [2,2*N]
   */
  ierr = VecCreateMPIWithArray(PETSC_COMM_WORLD,1,nleaves,PETSC_DECIDE,leafupdate,&gy2);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)x,"rootdata");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)gy2,"leafupdate");CHKERRQ(ierr);

  ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecView(gy2,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecDestroy(&gy2);CHKERRQ(ierr);

  ierr = VecRestoreArray(y2,&leafupdate);CHKERRQ(ierr);
  ierr = VecDestroy(&y2);CHKERRQ(ierr);

  ierr = VecRestoreArray(y,&leafdata);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);

  ierr = VecRestoreArray(x,&rootdata);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);

  ierr = PetscSFDestroy(&alltoallsf);CHKERRQ(ierr);

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

