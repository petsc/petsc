
#include "ptscimpl.h"
#include "inline/dot.h"
#include "sys/flog.h"
#include <math.h>
#include "pvecimpl.h" 
#include "mpe.h"


static int VeiDVPview( Vec xin,void *ptr ){
  DvPVector *x = (DvPVector *) xin->data;
  int i,j,mytid;

  MPI_Comm_rank(x->comm,&mytid); 

  MPE_Seq_begin(x->comm,1);
    printf("Processor [%d] \n",mytid);
    for ( i=0; i<x->n; i++ ) {
#if defined(PETSC_COMPLEX)
      printf("%g + %g i\n",real(x->array[i]),imag(x->array[i]));
#else
      printf("%g \n",x->array[i]);
#endif
    }
    fflush(stdout);
  MPE_Seq_end(x->comm,1);
  return 0;
}


