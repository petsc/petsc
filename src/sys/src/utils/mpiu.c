#ifndef lint
static char vcid[] = "$Id: try.c,v 1.10 1995/05/28 17:37:27 bsmith Exp bsmith $";
#endif
#include "petsc.h"
#include <stdio.h>
#include <stdarg.h>
#if defined(HAVE_STRING_H)
#include <string.h>
#endif
#include "petscfix.h"

/*@
    MPIU_fopen - The first process in the communicator opens a file,
                all others do nothing.

  Input Parameters:
.  comm - the communicator
.  name - the filename
.  mode - usually "w"
@*/
FILE *MPIU_fopen(MPI_Comm comm,char *name,char *mode)
{
  int  mytid;
  FILE *fd;
  MPI_Comm_rank(comm,&mytid);
  if (!mytid) fd = fopen(name,mode);
  else fd = 0;
  return fd;
}
/*@
     MPIU_fclose - The first processor in the communicator closes a 
                  file, all others do nothing.

  Input Parameters:
.  comm - the communicator
.  fd - the file, opened with MPIU_fopen()

@*/
int MPIU_fclose(MPI_Comm comm,FILE *fd)
{
  int  mytid;
  MPI_Comm_rank(comm,&mytid);
  if (!mytid) return fclose(fd);
  else return 0;
}

/*@
      MPIU_fprintf - Single print to a file only from the first
                    processor in the communicator.

  Input Parameters:
.  comm - the communicator
.  fd - the file pointer
.  format - the usual printf() format string 
@*/
int MPIU_fprintf(MPI_Comm comm,FILE* fd,char *format,...)
{
  int mytid;
  MPI_Comm_rank(comm,&mytid);
  if (!mytid) {
    va_list Argp;
    va_start( Argp, format );
    vfprintf(fd,format,Argp);
    va_end( Argp );
  }
  return 0;
}
/*@
      MPIU_printf - Single print to standard out, only from the first
                    processor in the communicator.

  Input Parameters:
.  comm - the communicator
.  format - the usual printf() format string 
@*/
int MPIU_printf(MPI_Comm comm,char *format,...)
{
  int mytid;
  MPI_Comm_rank(comm,&mytid);
  if (!mytid) {
    va_list Argp;
    va_start( Argp, format );
    vfprintf(stdout,format,Argp);
    va_end( Argp );
  }
  return 0;
}

#if defined(__cplusplus)
extern "C" {
#endif
extern char *getenv(char*);
#if defined(__cplusplus)
};
#endif

/*@
     MPIU_Set_display - Tries to set the display variable for all processors.

  Input Parameters:
.   comm - the communicatior, probably MPI_COMM_WORLD
.   n - length of string display

  Output Parameters:
.   display - the display string, may (and should) be freed.

@*/
int MPIU_Set_display(MPI_Comm comm,char *display,int n)
{
  int  numtid,mytid,len;
  char *string,*str;
  MPI_Comm_size(comm,&numtid);
  MPI_Comm_rank(comm,&mytid);  
  if (!mytid) {
    str = getenv("DISPLAY");
    if (!str || str[0] == ':') {
      string = (char *) PETSCMALLOC( 256*sizeof(char) ); CHKPTRQ(string);
      MPI_Get_processor_name(string,&len);
      strncpy(display,string,n-4); PETSCFREE(string);
      strcat(display,":0.0");
    }
    else {
      len = strlen(str);
      strncpy(display,str,n);
    }
    len = strlen(display);
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


#ifndef NULL
#define NULL (void *)0
#endif
extern void *malloc();

static int MPIU_Seq_keyval = MPI_KEYVAL_INVALID;

/*@
   MPIU_Seq_begin - Begins a sequential section of code.  

   Input Parameters:
.  comm - Communicator to sequentialize.  
.  ng   - Number in group.  This many processes are allowed to execute
   at the same time.  Usually one.  

   Notes:
   MPIU_Seq_begin and MPIU_Seq_end provide a way to force a section of code to
   be executed by the processes in rank order.  Typically, this is done 
   with
$  MPIU_Seq_begin( comm, 1 );
$  <code to be executed sequentially>
$  MPIU_Seq_end( comm, 1 );
$
   Often, the sequential code contains output statements (e.g., printf) to
   be executed.  Note that you may need to flush the I/O buffers before
   calling MPIU_Seq_end; also note that some systems do not propagate I/O in any
   order to the controling terminal (in other words, even if you flush the
   output, you may not get the data in the order that you want).
@*/
void MPIU_Seq_begin(MPI_Comm comm,int ng )
{
  int        lidx, np;
  int        flag;
  MPI_Comm   local_comm;
  MPI_Status status;

  /* Get the private communicator for the sequential operations */
  if (MPIU_Seq_keyval == MPI_KEYVAL_INVALID) {
    MPI_Keyval_create( MPI_NULL_COPY_FN, MPI_NULL_DELETE_FN, 
                       &MPIU_Seq_keyval, NULL );
  }
  MPI_Attr_get( comm, MPIU_Seq_keyval, (void **)&local_comm, &flag );
  if (!flag) {
    /* This expects a communicator to be a pointer */
    MPI_Comm_dup( comm, &local_comm );
    MPI_Attr_put( comm, MPIU_Seq_keyval, (void *)local_comm );
  }
  MPI_Comm_rank( comm, &lidx );
  MPI_Comm_size( comm, &np );
  if (lidx != 0) {
    MPI_Recv( NULL, 0, MPI_INT, lidx-1, 0, local_comm, &status );
  }
  /* Send to the next process in the group unless we are the last process 
   in the processor set */
  if ( (lidx % ng) < ng - 1 && lidx != np - 1) {
    MPI_Send( NULL, 0, MPI_INT, lidx + 1, 0, local_comm );
  }
}

/*@
   MPIU_Seq_end - Ends a sequential section of code.

   Input Parameters:
.  comm - Communicator to sequentialize.  
.  ng   - Number in group.  This many processes are allowed to execute
   at the same time.  Usually one.  

   Notes:
   See MPIU_Seq_begin for more details.
@*/
void MPIU_Seq_end(MPI_Comm comm,int ng )
{
  int        lidx, np, flag;
  MPI_Status status;
  MPI_Comm   local_comm;

  MPI_Comm_rank( comm, &lidx );
  MPI_Comm_size( comm, &np );
  MPI_Attr_get( comm, MPIU_Seq_keyval, (void **)&local_comm, &flag );
  if (!flag) MPI_Abort( comm, MPI_ERR_UNKNOWN );
  /* Send to the first process in the next group OR to the first process
     in the processor set */
  if ( (lidx % ng) == ng - 1 || lidx == np - 1) {
    MPI_Send( NULL, 0, MPI_INT, (lidx + 1) % np, 0, local_comm );
  }
  if (lidx == 0) {
    MPI_Recv( NULL, 0, MPI_INT, np-1, 0, local_comm, &status );
  }
}
