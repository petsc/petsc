/* $Id: f90_nag.c,v 1.2 1997/04/02 23:16:36 bsmith Exp bsmith $ */


typedef struct {
  long lower;
  long extent;
  long mult;
} tripple;
 
#define f90_header() \
void* addr; \
long  sizeof_data; \
int   unknown; \
int   ndim; \
int   a,b,c; \
long  d; \


typedef struct {
  f90_header()
  tripple dim[1];
}array1d;

typedef struct {
  f90_header()
  tripple dim[2];
}array2d;

typedef struct {
  f90_header()
  tripple dim[3];
}array3d;
