
/*
    Kernels used in sparse ILU (and LU) and in the resulting triangular
 solves. These are for block algorithms where the block sizes are on 
 the order of 6+.

*/


/*
      A = A * B   A_gets_A_times_B

   A, B - square n by n arrays stored in column major order
   W    - square n by n work array

*/
#define A_gets_A_times_B(nb,A,B,W) { \
BLgemm_("N","N",&bs,&bs,&bs,&one,pc,&bs,pv,&bs,&zero,
                multiplier,&bs);
        PetscMemcpy(pc,multiplier,bs2*sizeof(Scalar));

