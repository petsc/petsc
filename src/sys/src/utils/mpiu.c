#ifndef lint
static char vcid[] = "$Id: mpiu.c,v 1.24 1995/12/04 19:03:01 balay Exp balay $";
#endif
/*
      Some PETSc utilites routines (beginning with MPIU_) to add simple
  IO capability to MPI.
*/
#include "petsc.h"
#include <stdio.h>
#include <stdarg.h>
#if defined(HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#include "pinclude/petscfix.h"

/*
   If petsc_history is on then all MPI_*printf() results are saved
   if the appropriate (usually .petschistory) directory.
*/
extern FILE *petsc_history;

/*@C
    MPIU_fopen - The first process in the communicator opens a file,
                all others do nothing.

  Input Parameters:
.  comm - the communicator
.  name - the filename
.  mode - usually "w"
@*/
FILE *MPIU_fopen(MPI_Comm comm,char *name,char *mode)
{
  int  rank;
  FILE *fd;
  MPI_Comm_rank(comm,&rank);
  if (!rank) fd = fopen(name,mode);
  else fd = 0;
  return fd;
}
/*@C
     MPIU_fclose - The first processor in the communicator closes a 
                  file, all others do nothing.

  Input Parameters:
.  comm - the communicator
.  fd - the file, opened with MPIU_fopen()

@*/
int MPIU_fclose(MPI_Comm comm,FILE *fd)
{
  int  rank;
  MPI_Comm_rank(comm,&rank);
  if (!rank) return fclose(fd);
  else return 0;
}

/*@C
      MPIU_fprintf - Single print to a file only from the first
                    processor in the communicator.

  Input Parameters:
.  comm - the communicator
.  fd - the file pointer
.  format - the usual printf() format string 
@*/
int MPIU_fprintf(MPI_Comm comm,FILE* fd,char *format,...)
{
  int rank;
  MPI_Comm_rank(comm,&rank);
  if (!rank) {
    va_list Argp;
    va_start( Argp, format );
    vfprintf(fd,format,Argp);
    if (petsc_history) {
      vfprintf(petsc_history,format,Argp);
    }
    va_end( Argp );
  }
  return 0;
}
/*@C
      MPIU_printf - Single print to standard out, only from the first
                    processor in the communicator.

  Input Parameters:
.  comm - the communicator
.  format - the usual printf() format string 
@*/
int MPIU_printf(MPI_Comm comm,char *format,...)
{
  int rank;
  MPI_Comm_rank(comm,&rank);
  if (!rank) {
    va_list Argp;
    va_start( Argp, format );
    vfprintf(stdout,format,Argp);
    if (petsc_history) {
      vfprintf(petsc_history,format,Argp);
    }
    va_end( Argp );
  }
  return 0;
}

/*
     MPIU_Set_display - Tries to set the display variable for all processors.

  Input Parameters:
.   comm - the communicatior, probably MPI_COMM_WORLD
.   n - length of string display

  Output Parameters:
.   display - the display string, may (and should) be freed.

*/
int MPIU_Set_display(MPI_Comm comm,char *display,int n)
{
  int  size,rank,len;
  char *string,*str;

  MPI_Comm_size(comm,&size);
  MPI_Comm_rank(comm,&rank);  
  if (!rank) {
    str = getenv("DISPLAY");
    if (!str || str[0] == ':') {
      string = (char *) PetscMalloc( 256*sizeof(char) ); CHKPTRQ(string);
      MPI_Get_processor_name(string,&len);
      PetscStrncpy(display,string,n-4); PetscFree(string);
      PetscStrcat(display,":0.0");
    }
    else {
      len = PetscStrlen(str);
      PetscStrncpy(display,str,n);
    }
    len = PetscStrlen(display);
    MPI_Bcast(&len,1,MPI_INT,0,comm);
    MPI_Bcast(display,len,MPI_CHAR,0,comm);
  }
  else {
    MPI_Bcast(&len,1,MPI_INT,0,comm);
    MPI_Bcast(display,len,MPI_CHAR,0,comm);
    display[len] = 0;
  }
  return 0;  
}

static int MPIU_Seq_keyval = MPI_KEYVAL_INVALID;

/*@C
   MPIU_Seq_begin - Begins a sequential section of code.  

   Input Parameters:
.  comm - Communicator to sequentialize.  
.  ng   - Number in group.  This many processes are allowed to execute
   at the same time.  Usually one.  

   Notes:
   MPIU_Seq_begin and MPIU_Seq_end provide a way to force a section of 
   code to be executed by the processes in rank order.  Typically, this 
   is done with
$  MPIU_Seq_begin( comm, 1 );
$  <code to be executed sequentially>
$  MPIU_Seq_end( comm, 1 );
$
   Often, the sequential code contains output statements (e.g., printf) to
   be executed.  Note that you may need to flush the I/O buffers before
   calling MPIU_Seq_end; also note that some systems do not propagate
   I/O in any  order to the controling terminal (in other words, 
   even if you flush the output, you may not get the data in the order
   that you want).
@*/
int MPIU_Seq_begin(MPI_Comm comm,int ng )
{
  int        lidx, np;
  int        flag;
  MPI_Comm   local_comm;
  MPI_Status status;

  /* Get the private communicator for the sequential operations */
  if (MPIU_Seq_keyval == MPI_KEYVAL_INVALID) {
    MPI_Keyval_create(MPI_NULL_COPY_FN, MPI_NULL_DELETE_FN,&MPIU_Seq_keyval,(void*)0);
  }
  MPI_Attr_get( comm, MPIU_Seq_keyval, (void **)&local_comm, &flag );
  if (!flag) {
    /* This expects a communicator to be a pointer */
    MPI_Comm_dup( comm, &local_comm );
    MPI_Attr_put( comm, MPIU_Seq_keyval, (void *)local_comm );
  }
  MPI_Comm_rank( comm, &lidx );
  MPI_Comm_size( comm, &np );
  if (np == 1) return 0;
  if (lidx != 0) {
    MPI_Recv( (void*)0, 0, MPI_INT, lidx-1, 0, local_comm, &status );
  }
  /* Send to the next process in the group unless we are the last process 
   in the processor set */
  if ( (lidx % ng) < ng - 1 && lidx != np - 1) {
    MPI_Send( (void*)0, 0, MPI_INT, lidx + 1, 0, local_comm );
  }
  return 0;
}

/*@C
   MPIU_Seq_end - Ends a sequential section of code.

   Input Parameters:
.  comm - Communicator to sequentialize.  
.  ng   - Number in group.  This many processes are allowed to execute
   at the same time.  Usually one.  

   Notes:
   See MPIU_Seq_begin for more details.
@*/
int MPIU_Seq_end(MPI_Comm comm,int ng )
{
  int        lidx, np, flag;
  MPI_Status status;
  MPI_Comm   local_comm;

  MPI_Comm_rank( comm, &lidx );
  MPI_Comm_size( comm, &np );
  if (np == 1) return 0;
  MPI_Attr_get( comm, MPIU_Seq_keyval, (void **)&local_comm, &flag );
  if (!flag) MPI_Abort( comm, MPI_ERR_UNKNOWN );
  /* Send to the first process in the next group OR to the first process
     in the processor set */
  if ( (lidx % ng) == ng - 1 || lidx == np - 1) {
    MPI_Send( (void*)0, 0, MPI_INT, (lidx + 1) % np, 0, local_comm );
  }
  if (lidx == 0) {
    MPI_Recv( (void*)0, 0, MPI_INT, np-1, 0, local_comm, &status );
  }
  return 0;
}
/* ----------------------------------------------------------------
   A simple way to manage tags inside a private 
   communicator.  It uses the attribute to determine if a new communicator
   is needed.

   Notes on the implementation

   The tagvalues to use are stored in a two element array.  The first element
   is the first free tag value.  The second is used to indicate how
   many "copies" of the commincator there are used in destroying.
 */

static int MPIU_Tag_keyval = MPI_KEYVAL_INVALID;

/*
   Private routine to delete internal storage when a communicator is freed.
  This is called by MPI, not by users.
*/
static int MPIU_DelTag(MPI_Comm *comm,int* keyval,void* attr_val,void* extra_state )
{
  PetscFree( attr_val );
  return MPI_SUCCESS;
}

/*
  MPIU_Comm_dup - Duplicates only if communicator is not a PETSc communicator.


  Input Parameters:
. comm_in - Input communicator
. ntags   - Number of tags

  Output Parameters:
. comm_out - Output communicator.  May be 'comm_in'.
. first_tag - First tag available

  Returns:
  MPI_SUCCESS on success, MPI error class on failure.

  Notes:
  This routine returns one tag number
*/
int MPIU_Comm_dup(MPI_Comm comm_in,MPI_Comm *comm_out,int* first_tag)
{
  int ierr = MPI_SUCCESS, *tagvalp, *maxval, flag;

  if (MPIU_Tag_keyval == MPI_KEYVAL_INVALID) {
    MPI_Keyval_create(MPI_NULL_COPY_FN, MPIU_DelTag,&MPIU_Tag_keyval,(void *)0);
  }

  if (ierr = MPI_Attr_get(comm_in,MPIU_Tag_keyval,(void**)&tagvalp,&flag))
    return ierr;

  if (!flag) {
    /* This communicator is not yet known to this system, so we
       dup it and set the first value */
    MPI_Comm_dup( comm_in, comm_out );
    MPI_Attr_get( MPI_COMM_WORLD, MPI_TAG_UB, (void**)&maxval, &flag );
    tagvalp = (int *)PetscMalloc( 2*sizeof(int) );
    if (!tagvalp) return MPI_ERR_EXHAUSTED;
    tagvalp[0] = *maxval;
    tagvalp[1] = 0;
    MPI_Attr_put(*comm_out,MPIU_Tag_keyval, (void*)tagvalp);
  }
  else {
    *comm_out = comm_in;
  }

  if (*tagvalp < 1) {
    /* Error, out of tags.Another solution would be to do an MPI_Comm_dup. */
    return MPI_ERR_INTERN;
  }
  *first_tag = tagvalp[0]--;
  tagvalp[1]++;
  return MPI_SUCCESS;
}

int MPIU_Comm_free(MPI_Comm *comm)
{
  int ierr,*tagvalp,flag;
  ierr = MPI_Attr_get(*comm,MPIU_Tag_keyval,(void**)&tagvalp,&flag); CHKERRQ(ierr);
  tagvalp[1]--;
  if (!tagvalp[1]) {MPI_Comm_free(comm);}
  return 0;
}
