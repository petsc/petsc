/*

Test program follows. Writing it I realised that 
1/ I built the pipeline object around an MPI-to-MPI vector scatter.
2/ That necessitates the 'little trick' below.
3/ This trick will always be necessary, since the code inside the
  pipe is a critical section.
4/ Hence I really should have used an MPI-to-Seq scatter.
5/ This shouldn't be too hard to fix in the implementation you
  guys are making, right?  :-)  <-- smiley just in case.

If this is not clear, I'll try to elaborate.

*/
/* Example of pipeline code: 
   accumulation of the sum $s_p=\sum_{q\leq p} (q+1)^2$.
   E.g., processor 3 computes 1^2+2^2+3^+4^2 = 30.
   Every processor computes its term, then passes it on to the next.
*/
#include "vec.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int Argc,char **Args)
{
  Vec         src_v,tar_v,loc_v;
  IS          src_idx,tar_idx;
  VecPipeline pipe;
  MPI_Comm    comm;
  int         size,rank,src_loc,tar_loc,ierr,zero_loc=0;
  Scalar      zero=0,my_value,*vec_values,*loc_ar;

  PetscInitialize(&Argc,&Args,PETSC_NULL,PETSC_NULL);

  comm = MPI_COMM_WORLD;
  ierr = MPI_Comm_size(comm,&size);CHKERRA(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRA(ierr);
  
  /* Create the necessary vectors; one element per processor */
  ierr = VecCreateMPI(comm,1,PETSC_DECIDE,&tar_v);CHKERRA(ierr);
  ierr = VecSet(&zero,tar_v);CHKERRA(ierr);
  ierr = VecCreateMPI(comm,1,PETSC_DECIDE,&src_v);CHKERRA(ierr);
  ierr = VecCreateSeq(MPI_COMM_SELF,1,&loc_v);CHKERRA(ierr);
  /* -- little trick: we need a distributed and a local vector
     that share each other's data; see below for application */
  ierr = VecGetArray(loc_v,&loc_ar);CHKERRA(ierr);
  ierr = VecPlaceArray(src_v,loc_ar);CHKERRA(ierr);

  /* Create the pipeline data: we write into our own location,
     and read one location from the left */
  tar_loc = rank;
  if (tar_loc>0) src_loc = tar_loc-1; else src_loc = tar_loc;
  ierr = ISCreateGeneral(MPI_COMM_SELF,1,&tar_loc,&tar_idx);CHKERRA(ierr);
  ierr = ISCreateGeneral(MPI_COMM_SELF,1,&src_loc,&src_idx);CHKERRA(ierr);
  ierr = VecPipelineCreate(comm,src_v,src_idx,tar_v,tar_idx,&pipe);CHKERRA(ierr);
  ierr = VecPipelineSetType(pipe,PIPELINE_SEQUENTIAL,PETSC_NULL);CHKERRA(ierr);
  ierr = VecPipelineSetup(pipe);CHKERRA(ierr);

  /* The actual pipe:
     receive accumulated value from previous processor,
     add the square of your own value, and send on. */
  ierr = VecPipelineBegin(src_v,tar_v,INSERT_VALUES,SCATTER_FORWARD,
			  PIPELINE_UP,pipe);CHKERRA(ierr);
  ierr = VecGetArray(tar_v,&vec_values);CHKERRA(ierr);
  my_value = vec_values[0]+(rank+1)*(rank+1);
  printf("[%d] value=%d\n",rank,(int)PetscReal(my_value));
  ierr = VecRestoreArray(tar_v,&vec_values);CHKERRA(ierr);
  /* -- little trick: we have to be able to call VecAssembly, 
     but since this code executed sequentially (critical section!),
     we have a local vector with data aliased to the distributed one */
  ierr = VecSetValues(loc_v,1,&zero_loc,&my_value,INSERT_VALUES); 
  ierr = VecAssemblyBegin(loc_v);CHKERRA(ierr);
  ierr = VecAssemblyEnd(loc_v);CHKERRA(ierr);
  ierr = VecPipelineEnd(src_v,tar_v,INSERT_VALUES,SCATTER_FORWARD,
			PIPELINE_UP,pipe);CHKERRA(ierr);

  /* Clean up */
  ierr = VecPipelineDestroy(pipe);CHKERRA(ierr);
  ierr = VecDestroy(src_v);CHKERRA(ierr);
  ierr = VecDestroy(tar_v);CHKERRA(ierr);
  ierr = VecDestroy(loc_v);CHKERRA(ierr);
  ierr = ISDestroy(src_idx);CHKERRA(ierr);
  ierr = ISDestroy(tar_idx);CHKERRA(ierr);

  ierr = PetscFinalize();CHKERRA(ierr);

  return 0;
}

