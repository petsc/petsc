#include <stdio.h>
#include <stdarg.h>

FILE *MPE_fopen(MPI_Comm comm,char *name,char *mode)
{
  int  mytid;
  FILE *fd;
  MPI_Comm_rank(comm,&mytid);
  if (!mytid) fd = fopen(name,mode);
  else fd = 0;
  return fd;
}
int MPE_fclose(MPI_Comm comm,FILE fd)
{
  int  mytid;
  MPI_Comm_rank(comm,&mytid);
  if (!mytid) return fclose(fd);
  else return 0;
}

int MPE_fprintf(MPI_Comm comm,FILE fd,char *format,...)
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
int MPE_printf(MPI_Comm comm,char *format,...)
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



