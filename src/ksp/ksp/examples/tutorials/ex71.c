static char help[] =  "This example illustrates the use of PCBDDC/FETI-DP with 2D/3D DMDA.\n\
It solves the constant coefficient Poisson problem or the Elasticity problem \n\
on a uniform grid of [0,cells_x] x [0,cells_y] x [0,cells_z]\n\n";

/* Contributed by Wim Vanroose <wim@vanroo.se> */

#include <petscksp.h>
#include <petscpc.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmplex.h>

static PetscScalar poiss_1D_emat[] = {
1.0000000000000000e+00, -1.0000000000000000e+00,
-1.0000000000000000e+00, 1.0000000000000000e+00
};
static PetscScalar poiss_2D_emat[] = {
6.6666666666666674e-01, -1.6666666666666666e-01, -1.6666666666666666e-01, -3.3333333333333337e-01,
-1.6666666666666666e-01, 6.6666666666666674e-01, -3.3333333333333337e-01, -1.6666666666666666e-01,
-1.6666666666666666e-01, -3.3333333333333337e-01, 6.6666666666666674e-01, -1.6666666666666666e-01,
-3.3333333333333337e-01, -1.6666666666666666e-01, -1.6666666666666666e-01, 6.6666666666666674e-01
};
static PetscScalar poiss_3D_emat[] = {
3.3333333333333348e-01, 0.0000000000000000e+00, 0.0000000000000000e+00, -8.3333333333333343e-02, 0.0000000000000000e+00, -8.3333333333333343e-02, -8.3333333333333343e-02, -8.3333333333333356e-02,
0.0000000000000000e+00, 3.3333333333333337e-01, -8.3333333333333343e-02, 0.0000000000000000e+00, -8.3333333333333343e-02, 0.0000000000000000e+00, -8.3333333333333356e-02, -8.3333333333333343e-02,
0.0000000000000000e+00, -8.3333333333333343e-02, 3.3333333333333337e-01, 0.0000000000000000e+00, -8.3333333333333343e-02, -8.3333333333333356e-02, 0.0000000000000000e+00, -8.3333333333333343e-02,
-8.3333333333333343e-02, 0.0000000000000000e+00, 0.0000000000000000e+00, 3.3333333333333348e-01, -8.3333333333333356e-02, -8.3333333333333343e-02, -8.3333333333333343e-02, 0.0000000000000000e+00,
0.0000000000000000e+00, -8.3333333333333343e-02, -8.3333333333333343e-02, -8.3333333333333356e-02, 3.3333333333333337e-01, 0.0000000000000000e+00, 0.0000000000000000e+00, -8.3333333333333343e-02,
-8.3333333333333343e-02, 0.0000000000000000e+00, -8.3333333333333356e-02, -8.3333333333333343e-02, 0.0000000000000000e+00, 3.3333333333333337e-01, -8.3333333333333343e-02, 0.0000000000000000e+00,
-8.3333333333333343e-02, -8.3333333333333356e-02, 0.0000000000000000e+00, -8.3333333333333343e-02, 0.0000000000000000e+00, -8.3333333333333343e-02, 3.3333333333333337e-01, 0.0000000000000000e+00,
-8.3333333333333356e-02, -8.3333333333333343e-02, -8.3333333333333343e-02, 0.0000000000000000e+00, -8.3333333333333343e-02, 0.0000000000000000e+00, 0.0000000000000000e+00, 3.3333333333333337e-01
};
static PetscScalar elast_1D_emat[] = {
3.0000000000000000e+00, -3.0000000000000000e+00,
-3.0000000000000000e+00, 3.0000000000000000e+00
};
static PetscScalar elast_2D_emat[] = {
1.3333333333333335e+00, 5.0000000000000000e-01, -8.3333333333333337e-01, 0.0000000000000000e+00, 1.6666666666666671e-01, 0.0000000000000000e+00, -6.6666666666666674e-01, -5.0000000000000000e-01,
5.0000000000000000e-01, 1.3333333333333335e+00, 0.0000000000000000e+00, 1.6666666666666671e-01, 0.0000000000000000e+00, -8.3333333333333337e-01, -5.0000000000000000e-01, -6.6666666666666674e-01,
-8.3333333333333337e-01, 0.0000000000000000e+00, 1.3333333333333335e+00, -5.0000000000000000e-01, -6.6666666666666674e-01, 5.0000000000000000e-01, 1.6666666666666674e-01, 0.0000000000000000e+00,
0.0000000000000000e+00, 1.6666666666666671e-01, -5.0000000000000000e-01, 1.3333333333333335e+00, 5.0000000000000000e-01, -6.6666666666666674e-01, 0.0000000000000000e+00, -8.3333333333333337e-01,
1.6666666666666671e-01, 0.0000000000000000e+00, -6.6666666666666674e-01, 5.0000000000000000e-01, 1.3333333333333335e+00, -5.0000000000000000e-01, -8.3333333333333337e-01, 0.0000000000000000e+00,
0.0000000000000000e+00, -8.3333333333333337e-01, 5.0000000000000000e-01, -6.6666666666666674e-01, -5.0000000000000000e-01, 1.3333333333333335e+00, 0.0000000000000000e+00, 1.6666666666666674e-01,
-6.6666666666666674e-01, -5.0000000000000000e-01, 1.6666666666666674e-01, 0.0000000000000000e+00, -8.3333333333333337e-01, 0.0000000000000000e+00, 1.3333333333333335e+00, 5.0000000000000000e-01,
-5.0000000000000000e-01, -6.6666666666666674e-01, 0.0000000000000000e+00, -8.3333333333333337e-01, 0.0000000000000000e+00, 1.6666666666666674e-01, 5.0000000000000000e-01, 1.3333333333333335e+00
};
static PetscScalar elast_3D_emat[] = {
5.5555555555555558e-01, 1.6666666666666666e-01, 1.6666666666666666e-01, -2.2222222222222232e-01, 0.0000000000000000e+00, 0.0000000000000000e+00, 1.1111111111111113e-01, 0.0000000000000000e+00, 8.3333333333333356e-02, -1.9444444444444442e-01, -1.6666666666666669e-01, 0.0000000000000000e+00, 1.1111111111111112e-01, 8.3333333333333356e-02, 0.0000000000000000e+00, -1.9444444444444445e-01, 0.0000000000000000e+00, -1.6666666666666669e-01, -2.7777777777777769e-02, 0.0000000000000000e+00, 0.0000000000000000e+00, -1.3888888888888887e-01, -8.3333333333333356e-02, -8.3333333333333356e-02,
1.6666666666666666e-01, 5.5555555555555558e-01, 1.6666666666666666e-01, 0.0000000000000000e+00, 1.1111111111111113e-01, 8.3333333333333356e-02, 0.0000000000000000e+00, -2.2222222222222232e-01, 0.0000000000000000e+00, -1.6666666666666669e-01, -1.9444444444444442e-01, 0.0000000000000000e+00, 8.3333333333333356e-02, 1.1111111111111112e-01, 0.0000000000000000e+00, 0.0000000000000000e+00, -2.7777777777777769e-02, 0.0000000000000000e+00, 0.0000000000000000e+00, -1.9444444444444445e-01, -1.6666666666666669e-01, -8.3333333333333356e-02, -1.3888888888888887e-01, -8.3333333333333356e-02,
1.6666666666666666e-01, 1.6666666666666666e-01, 5.5555555555555558e-01, 0.0000000000000000e+00, 8.3333333333333356e-02, 1.1111111111111112e-01, 8.3333333333333356e-02, 0.0000000000000000e+00, 1.1111111111111112e-01, 0.0000000000000000e+00, 0.0000000000000000e+00, -2.7777777777777769e-02, 0.0000000000000000e+00, 0.0000000000000000e+00, -2.2222222222222229e-01, -1.6666666666666669e-01, 0.0000000000000000e+00, -1.9444444444444445e-01, 0.0000000000000000e+00, -1.6666666666666669e-01, -1.9444444444444445e-01, -8.3333333333333356e-02, -8.3333333333333356e-02, -1.3888888888888887e-01,
-2.2222222222222232e-01, 0.0000000000000000e+00, 0.0000000000000000e+00, 5.5555555555555558e-01, -1.6666666666666666e-01, -1.6666666666666666e-01, -1.9444444444444442e-01, 1.6666666666666669e-01, 0.0000000000000000e+00, 1.1111111111111113e-01, 0.0000000000000000e+00, -8.3333333333333356e-02, -1.9444444444444445e-01, 0.0000000000000000e+00, 1.6666666666666669e-01, 1.1111111111111113e-01, -8.3333333333333356e-02, 0.0000000000000000e+00, -1.3888888888888887e-01, 8.3333333333333356e-02, 8.3333333333333356e-02, -2.7777777777777769e-02, 0.0000000000000000e+00, 0.0000000000000000e+00,
0.0000000000000000e+00, 1.1111111111111113e-01, 8.3333333333333356e-02, -1.6666666666666666e-01, 5.5555555555555558e-01, 1.6666666666666669e-01, 1.6666666666666669e-01, -1.9444444444444442e-01, 0.0000000000000000e+00, 0.0000000000000000e+00, -2.2222222222222229e-01, 0.0000000000000000e+00, 0.0000000000000000e+00, -2.7777777777777769e-02, 0.0000000000000000e+00, -8.3333333333333356e-02, 1.1111111111111112e-01, 0.0000000000000000e+00, 8.3333333333333356e-02, -1.3888888888888887e-01, -8.3333333333333356e-02, 0.0000000000000000e+00, -1.9444444444444448e-01, -1.6666666666666666e-01,
0.0000000000000000e+00, 8.3333333333333356e-02, 1.1111111111111112e-01, -1.6666666666666666e-01, 1.6666666666666669e-01, 5.5555555555555558e-01, 0.0000000000000000e+00, 0.0000000000000000e+00, -2.7777777777777769e-02, -8.3333333333333356e-02, 0.0000000000000000e+00, 1.1111111111111112e-01, 1.6666666666666669e-01, 0.0000000000000000e+00, -1.9444444444444445e-01, 0.0000000000000000e+00, 0.0000000000000000e+00, -2.2222222222222227e-01, 8.3333333333333356e-02, -8.3333333333333356e-02, -1.3888888888888887e-01, 0.0000000000000000e+00, -1.6666666666666666e-01, -1.9444444444444448e-01,
1.1111111111111113e-01, 0.0000000000000000e+00, 8.3333333333333356e-02, -1.9444444444444442e-01, 1.6666666666666669e-01, 0.0000000000000000e+00, 5.5555555555555569e-01, -1.6666666666666666e-01, 1.6666666666666669e-01, -2.2222222222222229e-01, 0.0000000000000000e+00, 0.0000000000000000e+00, -2.7777777777777769e-02, 0.0000000000000000e+00, 0.0000000000000000e+00, -1.3888888888888887e-01, 8.3333333333333356e-02, -8.3333333333333356e-02, 1.1111111111111112e-01, -8.3333333333333343e-02, 0.0000000000000000e+00, -1.9444444444444448e-01, 0.0000000000000000e+00, -1.6666666666666669e-01,
0.0000000000000000e+00, -2.2222222222222232e-01, 0.0000000000000000e+00, 1.6666666666666669e-01, -1.9444444444444442e-01, 0.0000000000000000e+00, -1.6666666666666666e-01, 5.5555555555555558e-01, -1.6666666666666669e-01, 0.0000000000000000e+00, 1.1111111111111113e-01, -8.3333333333333343e-02, 0.0000000000000000e+00, -1.9444444444444445e-01, 1.6666666666666669e-01, 8.3333333333333356e-02, -1.3888888888888887e-01, 8.3333333333333356e-02, -8.3333333333333343e-02, 1.1111111111111113e-01, 0.0000000000000000e+00, 0.0000000000000000e+00, -2.7777777777777769e-02, 0.0000000000000000e+00,
8.3333333333333356e-02, 0.0000000000000000e+00, 1.1111111111111112e-01, 0.0000000000000000e+00, 0.0000000000000000e+00, -2.7777777777777769e-02, 1.6666666666666669e-01, -1.6666666666666669e-01, 5.5555555555555558e-01, 0.0000000000000000e+00, -8.3333333333333343e-02, 1.1111111111111112e-01, 0.0000000000000000e+00, 1.6666666666666669e-01, -1.9444444444444445e-01, -8.3333333333333356e-02, 8.3333333333333356e-02, -1.3888888888888887e-01, 0.0000000000000000e+00, 0.0000000000000000e+00, -2.2222222222222227e-01, -1.6666666666666669e-01, 0.0000000000000000e+00, -1.9444444444444448e-01,
-1.9444444444444442e-01, -1.6666666666666669e-01, 0.0000000000000000e+00, 1.1111111111111113e-01, 0.0000000000000000e+00, -8.3333333333333356e-02, -2.2222222222222229e-01, 0.0000000000000000e+00, 0.0000000000000000e+00, 5.5555555555555558e-01, 1.6666666666666669e-01, -1.6666666666666666e-01, -1.3888888888888887e-01, -8.3333333333333356e-02, 8.3333333333333356e-02, -2.7777777777777769e-02, 0.0000000000000000e+00, 0.0000000000000000e+00, -1.9444444444444448e-01, 0.0000000000000000e+00, 1.6666666666666669e-01, 1.1111111111111112e-01, 8.3333333333333343e-02, 0.0000000000000000e+00,
-1.6666666666666669e-01, -1.9444444444444442e-01, 0.0000000000000000e+00, 0.0000000000000000e+00, -2.2222222222222229e-01, 0.0000000000000000e+00, 0.0000000000000000e+00, 1.1111111111111113e-01, -8.3333333333333343e-02, 1.6666666666666669e-01, 5.5555555555555558e-01, -1.6666666666666669e-01, -8.3333333333333356e-02, -1.3888888888888887e-01, 8.3333333333333356e-02, 0.0000000000000000e+00, -1.9444444444444448e-01, 1.6666666666666669e-01, 0.0000000000000000e+00, -2.7777777777777769e-02, 0.0000000000000000e+00, 8.3333333333333343e-02, 1.1111111111111112e-01, 0.0000000000000000e+00,
0.0000000000000000e+00, 0.0000000000000000e+00, -2.7777777777777769e-02, -8.3333333333333356e-02, 0.0000000000000000e+00, 1.1111111111111112e-01, 0.0000000000000000e+00, -8.3333333333333343e-02, 1.1111111111111112e-01, -1.6666666666666666e-01, -1.6666666666666669e-01, 5.5555555555555558e-01, 8.3333333333333356e-02, 8.3333333333333356e-02, -1.3888888888888887e-01, 0.0000000000000000e+00, 1.6666666666666669e-01, -1.9444444444444448e-01, 1.6666666666666669e-01, 0.0000000000000000e+00, -1.9444444444444448e-01, 0.0000000000000000e+00, 0.0000000000000000e+00, -2.2222222222222227e-01,
1.1111111111111112e-01, 8.3333333333333356e-02, 0.0000000000000000e+00, -1.9444444444444445e-01, 0.0000000000000000e+00, 1.6666666666666669e-01, -2.7777777777777769e-02, 0.0000000000000000e+00, 0.0000000000000000e+00, -1.3888888888888887e-01, -8.3333333333333356e-02, 8.3333333333333356e-02, 5.5555555555555569e-01, 1.6666666666666669e-01, -1.6666666666666669e-01, -2.2222222222222227e-01, 0.0000000000000000e+00, 0.0000000000000000e+00, 1.1111111111111112e-01, 0.0000000000000000e+00, -8.3333333333333343e-02, -1.9444444444444448e-01, -1.6666666666666669e-01, 0.0000000000000000e+00,
8.3333333333333356e-02, 1.1111111111111112e-01, 0.0000000000000000e+00, 0.0000000000000000e+00, -2.7777777777777769e-02, 0.0000000000000000e+00, 0.0000000000000000e+00, -1.9444444444444445e-01, 1.6666666666666669e-01, -8.3333333333333356e-02, -1.3888888888888887e-01, 8.3333333333333356e-02, 1.6666666666666669e-01, 5.5555555555555558e-01, -1.6666666666666669e-01, 0.0000000000000000e+00, 1.1111111111111112e-01, -8.3333333333333343e-02, 0.0000000000000000e+00, -2.2222222222222227e-01, 0.0000000000000000e+00, -1.6666666666666669e-01, -1.9444444444444448e-01, 0.0000000000000000e+00,
0.0000000000000000e+00, 0.0000000000000000e+00, -2.2222222222222229e-01, 1.6666666666666669e-01, 0.0000000000000000e+00, -1.9444444444444445e-01, 0.0000000000000000e+00, 1.6666666666666669e-01, -1.9444444444444445e-01, 8.3333333333333356e-02, 8.3333333333333356e-02, -1.3888888888888887e-01, -1.6666666666666669e-01, -1.6666666666666669e-01, 5.5555555555555558e-01, 0.0000000000000000e+00, -8.3333333333333343e-02, 1.1111111111111113e-01, -8.3333333333333343e-02, 0.0000000000000000e+00, 1.1111111111111113e-01, 0.0000000000000000e+00, 0.0000000000000000e+00, -2.7777777777777769e-02,
-1.9444444444444445e-01, 0.0000000000000000e+00, -1.6666666666666669e-01, 1.1111111111111113e-01, -8.3333333333333356e-02, 0.0000000000000000e+00, -1.3888888888888887e-01, 8.3333333333333356e-02, -8.3333333333333356e-02, -2.7777777777777769e-02, 0.0000000000000000e+00, 0.0000000000000000e+00, -2.2222222222222227e-01, 0.0000000000000000e+00, 0.0000000000000000e+00, 5.5555555555555558e-01, -1.6666666666666669e-01, 1.6666666666666669e-01, -1.9444444444444448e-01, 1.6666666666666669e-01, 0.0000000000000000e+00, 1.1111111111111112e-01, 0.0000000000000000e+00, 8.3333333333333343e-02,
0.0000000000000000e+00, -2.7777777777777769e-02, 0.0000000000000000e+00, -8.3333333333333356e-02, 1.1111111111111112e-01, 0.0000000000000000e+00, 8.3333333333333356e-02, -1.3888888888888887e-01, 8.3333333333333356e-02, 0.0000000000000000e+00, -1.9444444444444448e-01, 1.6666666666666669e-01, 0.0000000000000000e+00, 1.1111111111111112e-01, -8.3333333333333343e-02, -1.6666666666666669e-01, 5.5555555555555558e-01, -1.6666666666666666e-01, 1.6666666666666669e-01, -1.9444444444444448e-01, 0.0000000000000000e+00, 0.0000000000000000e+00, -2.2222222222222227e-01, 0.0000000000000000e+00,
-1.6666666666666669e-01, 0.0000000000000000e+00, -1.9444444444444445e-01, 0.0000000000000000e+00, 0.0000000000000000e+00, -2.2222222222222227e-01, -8.3333333333333356e-02, 8.3333333333333356e-02, -1.3888888888888887e-01, 0.0000000000000000e+00, 1.6666666666666669e-01, -1.9444444444444448e-01, 0.0000000000000000e+00, -8.3333333333333343e-02, 1.1111111111111113e-01, 1.6666666666666669e-01, -1.6666666666666666e-01, 5.5555555555555558e-01, 0.0000000000000000e+00, 0.0000000000000000e+00, -2.7777777777777769e-02, 8.3333333333333343e-02, 0.0000000000000000e+00, 1.1111111111111113e-01,
-2.7777777777777769e-02, 0.0000000000000000e+00, 0.0000000000000000e+00, -1.3888888888888887e-01, 8.3333333333333356e-02, 8.3333333333333356e-02, 1.1111111111111112e-01, -8.3333333333333343e-02, 0.0000000000000000e+00, -1.9444444444444448e-01, 0.0000000000000000e+00, 1.6666666666666669e-01, 1.1111111111111112e-01, 0.0000000000000000e+00, -8.3333333333333343e-02, -1.9444444444444448e-01, 1.6666666666666669e-01, 0.0000000000000000e+00, 5.5555555555555558e-01, -1.6666666666666669e-01, -1.6666666666666669e-01, -2.2222222222222227e-01, 0.0000000000000000e+00, 0.0000000000000000e+00,
0.0000000000000000e+00, -1.9444444444444445e-01, -1.6666666666666669e-01, 8.3333333333333356e-02, -1.3888888888888887e-01, -8.3333333333333356e-02, -8.3333333333333343e-02, 1.1111111111111113e-01, 0.0000000000000000e+00, 0.0000000000000000e+00, -2.7777777777777769e-02, 0.0000000000000000e+00, 0.0000000000000000e+00, -2.2222222222222227e-01, 0.0000000000000000e+00, 1.6666666666666669e-01, -1.9444444444444448e-01, 0.0000000000000000e+00, -1.6666666666666669e-01, 5.5555555555555558e-01, 1.6666666666666669e-01, 0.0000000000000000e+00, 1.1111111111111112e-01, 8.3333333333333343e-02,
0.0000000000000000e+00, -1.6666666666666669e-01, -1.9444444444444445e-01, 8.3333333333333356e-02, -8.3333333333333356e-02, -1.3888888888888887e-01, 0.0000000000000000e+00, 0.0000000000000000e+00, -2.2222222222222227e-01, 1.6666666666666669e-01, 0.0000000000000000e+00, -1.9444444444444448e-01, -8.3333333333333343e-02, 0.0000000000000000e+00, 1.1111111111111113e-01, 0.0000000000000000e+00, 0.0000000000000000e+00, -2.7777777777777769e-02, -1.6666666666666669e-01, 1.6666666666666669e-01, 5.5555555555555558e-01, 0.0000000000000000e+00, 8.3333333333333343e-02, 1.1111111111111113e-01,
-1.3888888888888887e-01, -8.3333333333333356e-02, -8.3333333333333356e-02, -2.7777777777777769e-02, 0.0000000000000000e+00, 0.0000000000000000e+00, -1.9444444444444448e-01, 0.0000000000000000e+00, -1.6666666666666669e-01, 1.1111111111111112e-01, 8.3333333333333343e-02, 0.0000000000000000e+00, -1.9444444444444448e-01, -1.6666666666666669e-01, 0.0000000000000000e+00, 1.1111111111111112e-01, 0.0000000000000000e+00, 8.3333333333333343e-02, -2.2222222222222227e-01, 0.0000000000000000e+00, 0.0000000000000000e+00, 5.5555555555555558e-01, 1.6666666666666669e-01, 1.6666666666666669e-01,
-8.3333333333333356e-02, -1.3888888888888887e-01, -8.3333333333333356e-02, 0.0000000000000000e+00, -1.9444444444444448e-01, -1.6666666666666666e-01, 0.0000000000000000e+00, -2.7777777777777769e-02, 0.0000000000000000e+00, 8.3333333333333343e-02, 1.1111111111111112e-01, 0.0000000000000000e+00, -1.6666666666666669e-01, -1.9444444444444448e-01, 0.0000000000000000e+00, 0.0000000000000000e+00, -2.2222222222222227e-01, 0.0000000000000000e+00, 0.0000000000000000e+00, 1.1111111111111112e-01, 8.3333333333333343e-02, 1.6666666666666669e-01, 5.5555555555555558e-01, 1.6666666666666669e-01,
-8.3333333333333356e-02, -8.3333333333333356e-02, -1.3888888888888887e-01, 0.0000000000000000e+00, -1.6666666666666666e-01, -1.9444444444444448e-01, -1.6666666666666669e-01, 0.0000000000000000e+00, -1.9444444444444448e-01, 0.0000000000000000e+00, 0.0000000000000000e+00, -2.2222222222222227e-01, 0.0000000000000000e+00, 0.0000000000000000e+00, -2.7777777777777769e-02, 8.3333333333333343e-02, 0.0000000000000000e+00, 1.1111111111111113e-01, 0.0000000000000000e+00, 8.3333333333333343e-02, 1.1111111111111113e-01, 1.6666666666666669e-01, 1.6666666666666669e-01, 5.5555555555555558e-01
};

typedef enum {PDE_POISSON, PDE_ELASTICITY} PDEType;

typedef struct {
  PDEType      pde;
  PetscInt     dim;
  PetscInt     dof;
  PetscInt     cells[3];
  PetscBool    useglobal;
  PetscBool    dirbc;
  PetscBool    per[3];
  PetscBool    test;
  PetscScalar *elemMat;
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  const char    *pdeTypes[2] = {"Poisson", "Elasticity"};
  PetscInt       n,pde;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->pde       = PDE_POISSON;
  options->elemMat   = NULL;
  options->dim       = 1;
  options->cells[0]  = 8;
  options->cells[1]  = 6;
  options->cells[2]  = 4;
  options->useglobal = PETSC_FALSE;
  options->dirbc     = PETSC_TRUE;
  options->test      = PETSC_FALSE;
  options->per[0]    = PETSC_FALSE;
  options->per[1]    = PETSC_FALSE;
  options->per[2]    = PETSC_FALSE;

  ierr = PetscOptionsBegin(comm,NULL,"Problem Options",NULL);CHKERRQ(ierr);
  pde  = options->pde;
  ierr = PetscOptionsEList("-pde_type","The PDE type",__FILE__,pdeTypes,2,pdeTypes[options->pde],&pde,NULL);CHKERRQ(ierr);
  options->pde = (PDEType)pde;
  ierr = PetscOptionsInt("-dim","The topological mesh dimension",__FILE__,options->dim,&options->dim,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsIntArray("-cells","The mesh division",__FILE__,options->cells,(n=3,&n),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBoolArray("-periodicity","The mesh periodicity",__FILE__,options->per,(n=3,&n),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-use_global","Test MatSetValues",__FILE__,options->useglobal,&options->useglobal,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-dirichlet","Use dirichlet BC",__FILE__,options->dirbc,&options->dirbc,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-test_assembly","Test MATIS assembly",__FILE__,options->test,&options->test,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();

  for (n=options->dim;n<3;n++) options->cells[n] = 0;
  if (options->per[0]) options->dirbc = PETSC_FALSE;

  /* element matrices */
  switch (options->pde) {
  case PDE_ELASTICITY:
    options->dof = options->dim;
    switch (options->dim) {
    case 1:
      options->elemMat = elast_1D_emat;
      break;
    case 2:
      options->elemMat = elast_2D_emat;
      break;
    case 3:
      options->elemMat = elast_3D_emat;
      break;
    default: SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Unsupported dimension %D",options->dim);
    }
    break;
  case PDE_POISSON:
    options->dof = 1;
    switch (options->dim) {
    case 1:
      options->elemMat = poiss_1D_emat;
      break;
    case 2:
      options->elemMat = poiss_2D_emat;
      break;
    case 3:
      options->elemMat = poiss_3D_emat;
      break;
    default: SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Unsupported dimension %D",options->dim);
    }
    break;
  default: SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Unsupported PDE %D",options->pde);
  }
  PetscFunctionReturn(0);
}

int main(int argc,char **args)
{
  AppCtx                 user;
  KSP                    ksp;
  PC                     pc;
  Mat                    A;
  DM                     da;
  Vec                    x,b,xcoor,xcoorl;
  IS                     zero;
  ISLocalToGlobalMapping map;
  MatNullSpace           nullsp = NULL;
  PetscInt               i;
  PetscInt               nel,nen;        /* Number of elements & element nodes */
  const PetscInt         *e_loc;         /* Local indices of element nodes (in local element order) */
  PetscInt               *e_glo = NULL;  /* Global indices of element nodes (in local element order) */
  PetscBool              ismatis;
#if defined(PETSC_USE_LOG)
  PetscLogStage          stages[2];
#endif
  PetscErrorCode         ierr;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = ProcessOptions(PETSC_COMM_WORLD,&user);CHKERRQ(ierr);
  switch (user.dim) {
  case 3:
    ierr = DMDACreate3d(PETSC_COMM_WORLD,user.per[0] ? DM_BOUNDARY_PERIODIC : DM_BOUNDARY_NONE,
                                         user.per[1] ? DM_BOUNDARY_PERIODIC : DM_BOUNDARY_NONE,
                                         user.per[2] ? DM_BOUNDARY_PERIODIC : DM_BOUNDARY_NONE,
                                         DMDA_STENCIL_BOX,user.cells[0]+1,user.cells[1]+1,user.cells[2]+1,
                                         PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,user.dof,
                                         1,PETSC_NULL,PETSC_NULL,PETSC_NULL,&da);CHKERRQ(ierr);
    break;
  case 2:
    ierr = DMDACreate2d(PETSC_COMM_WORLD,user.per[0] ? DM_BOUNDARY_PERIODIC : DM_BOUNDARY_NONE,
                                         user.per[1] ? DM_BOUNDARY_PERIODIC : DM_BOUNDARY_NONE,
                                         DMDA_STENCIL_BOX,user.cells[0]+1,user.cells[1]+1,
                                         PETSC_DECIDE,PETSC_DECIDE,user.dof,
                                         1,PETSC_NULL,PETSC_NULL,&da);CHKERRQ(ierr);
    break;
  case 1:
    ierr = DMDACreate1d(PETSC_COMM_WORLD,user.per[0] ? DM_BOUNDARY_PERIODIC : DM_BOUNDARY_NONE,
                                         user.cells[0]+1,user.dof,1,PETSC_NULL,&da);CHKERRQ(ierr);
    break;
  default: SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Unsupported dimension %D",user.dim);
  }

  ierr = PetscLogStageRegister("KSPSetUp",&stages[0]);CHKERRQ(ierr);
  ierr = PetscLogStageRegister("KSPSolve",&stages[1]);CHKERRQ(ierr);

  ierr = DMSetMatType(da,MATIS);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMDASetElementType(da,DMDA_ELEMENT_Q1);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  {
    PetscInt M,N,P;
    ierr = DMDAGetInfo(da,0,&M,&N,&P,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
    switch (user.dim) {
    case 3:
      user.cells[2] = P-1;
    case 2:
      user.cells[1] = N-1;
    case 1:
      user.cells[0] = M-1;
      break;
    default: SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Unsupported dimension %D",user.dim);
    }
  }
  ierr = DMDASetUniformCoordinates(da,0.0,1.0*user.cells[0],0.0,1.0*user.cells[1],0.0,1.0*user.cells[2]);CHKERRQ(ierr);
  ierr = DMGetCoordinates(da,&xcoor);CHKERRQ(ierr);

  ierr = DMCreateMatrix(da,&A);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = DMGetLocalToGlobalMapping(da,&map);CHKERRQ(ierr);
  ierr = DMDAGetElements(da,&nel,&nen,&e_loc);CHKERRQ(ierr);
  if (user.useglobal) {
    ierr = PetscMalloc1(nel*nen,&e_glo);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingApplyBlock(map,nen*nel,e_loc,e_glo);CHKERRQ(ierr);
  }

  /* we reorder the indices since the element matrices are given in lexicographic order,
     whereas the elements indices returned by DMDAGetElements follow the usual FEM ordering
     i.e., element matrices     DMDA ordering
               2---3              3---2
              /   /              /   /
             0---1              0---1
  */
  for (i = 0; i < nel; ++i) {
    PetscInt ord[8] = {0,1,3,2,4,5,7,6};
    PetscInt j,idxs[8];

    if (nen > 8) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Not coded");
    if (!e_glo) {
      for (j=0;j<nen;j++) idxs[j] = e_loc[i*nen+ord[j]];
      ierr = MatSetValuesBlockedLocal(A,nen,idxs,nen,idxs,user.elemMat,ADD_VALUES);CHKERRQ(ierr);
    } else {
      for (j=0;j<nen;j++) idxs[j] = e_glo[i*nen+ord[j]];
      ierr = MatSetValuesBlocked(A,nen,idxs,nen,idxs,user.elemMat,ADD_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = DMDARestoreElements(da,&nel,&nen,&e_loc);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatSetOption(A,MAT_SPD,PETSC_TRUE);CHKERRQ(ierr);

  /* Boundary conditions */
  zero = NULL;
  if (user.dirbc) { /* fix one side of DMDA */
    Vec         nat,glob;
    PetscScalar *vals;
    PetscInt    n,*idx,j,st;

    n    = PetscGlobalRank ? 0 : (user.cells[1]+1)*(user.cells[2]+1);
    ierr = ISCreateStride(PETSC_COMM_WORLD,n,0,user.cells[0]+1,&zero);CHKERRQ(ierr);
    if (user.dof > 1) { /* zero all components */
      const PetscInt *idx;
      IS             bzero;

      ierr = ISGetIndices(zero,(const PetscInt**)&idx);CHKERRQ(ierr);
      ierr = ISCreateBlock(PETSC_COMM_WORLD,user.dof,n,idx,PETSC_COPY_VALUES,&bzero);CHKERRQ(ierr);
      ierr = ISRestoreIndices(zero,(const PetscInt**)&idx);CHKERRQ(ierr);
      ierr = ISDestroy(&zero);CHKERRQ(ierr);
      zero = bzero;
    }
    /* map indices from natural to global */
    ierr = DMDACreateNaturalVector(da,&nat);CHKERRQ(ierr);
    ierr = ISGetLocalSize(zero,&n);CHKERRQ(ierr);
    ierr = PetscMalloc1(n,&vals);CHKERRQ(ierr);
    for (i=0;i<n;i++) vals[i] = 1.0;
    ierr = ISGetIndices(zero,(const PetscInt**)&idx);CHKERRQ(ierr);
    ierr = VecSetValues(nat,n,idx,vals,INSERT_VALUES);CHKERRQ(ierr);
    ierr = ISRestoreIndices(zero,(const PetscInt**)&idx);CHKERRQ(ierr);
    ierr = PetscFree(vals);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(nat);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(nat);CHKERRQ(ierr);
    ierr = DMCreateGlobalVector(da,&glob);CHKERRQ(ierr);
    ierr = DMDANaturalToGlobalBegin(da,nat,INSERT_VALUES,glob);CHKERRQ(ierr);
    ierr = DMDANaturalToGlobalEnd(da,nat,INSERT_VALUES,glob);CHKERRQ(ierr);
    ierr = VecDestroy(&nat);CHKERRQ(ierr);
    ierr = ISDestroy(&zero);CHKERRQ(ierr);
    ierr = VecGetLocalSize(glob,&n);CHKERRQ(ierr);
    ierr = PetscMalloc1(n,&idx);CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(glob,&st,NULL);CHKERRQ(ierr);
    ierr = VecGetArray(glob,&vals);CHKERRQ(ierr);
    for (i=0,j=0;i<n;i++) if (PetscRealPart(vals[i]) == 1.0) idx[j++] = i + st;
    ierr = VecRestoreArray(glob,&vals);CHKERRQ(ierr);
    ierr = VecDestroy(&glob);CHKERRQ(ierr);
    ierr = ISCreateGeneral(PETSC_COMM_WORLD,j,idx,PETSC_OWN_POINTER,&zero);CHKERRQ(ierr);
    ierr = MatZeroRowsColumnsIS(A,zero,1.0,NULL,NULL);CHKERRQ(ierr);
  } else {
    switch (user.pde) {
    case PDE_POISSON:
      ierr = MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,0,NULL,&nullsp);CHKERRQ(ierr);
      break;
    case PDE_ELASTICITY:
      ierr = MatNullSpaceCreateRigidBody(xcoor,&nullsp);CHKERRQ(ierr);
      break;
    }
    /* with periodic BC and Elasticity, just the displacements are in the nullspace
       this is no harm since we eliminate all the components of the rhs */
    ierr = MatSetNullSpace(A,nullsp);CHKERRQ(ierr);
  }

  if (user.test) {
    Mat AA;

    ierr = MatConvert(A,MATAIJ,MAT_INITIAL_MATRIX,&AA);CHKERRQ(ierr);
    ierr = MatViewFromOptions(AA,NULL,"-assembled_view");CHKERRQ(ierr);
    ierr = MatDestroy(&AA);CHKERRQ(ierr);
  }

  /* Attach near null space for elasticity */
  if (user.pde == PDE_ELASTICITY) {
    MatNullSpace nearnullsp;

    ierr = MatNullSpaceCreateRigidBody(xcoor,&nearnullsp);CHKERRQ(ierr);
    ierr = MatSetNearNullSpace(A,nearnullsp);CHKERRQ(ierr);
    ierr = MatNullSpaceDestroy(&nearnullsp);CHKERRQ(ierr);
  }

  /* we may want to use MG for the local solvers: attach local nearnullspace to the local matrices */
  ierr = DMGetCoordinatesLocal(da,&xcoorl);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)A,MATIS,&ismatis);CHKERRQ(ierr);
  if (ismatis) {
    MatNullSpace lnullsp = NULL;
    Mat          lA;

    ierr = MatISGetLocalMat(A,&lA);CHKERRQ(ierr);
    if (user.pde == PDE_ELASTICITY) {
      Vec                    lc;
      ISLocalToGlobalMapping l2l;
      IS                     is;
      const PetscScalar      *a;
      const PetscInt         *idxs;
      PetscInt               n,bs;

      /* when using a DMDA, the local matrices have an additional local-to-local map
         that maps from the DA local ordering to the ordering induced by the elements */
      ierr = MatCreateVecs(lA,&lc,NULL);CHKERRQ(ierr);
      ierr = MatGetLocalToGlobalMapping(lA,&l2l,NULL);CHKERRQ(ierr);
      ierr = VecSetLocalToGlobalMapping(lc,l2l);CHKERRQ(ierr);
      ierr = VecSetOption(lc,VEC_IGNORE_NEGATIVE_INDICES,PETSC_TRUE);CHKERRQ(ierr);
      ierr = VecGetLocalSize(xcoorl,&n);CHKERRQ(ierr);
      ierr = VecGetBlockSize(xcoorl,&bs);CHKERRQ(ierr);
      ierr = ISCreateStride(PETSC_COMM_SELF,n/bs,0,1,&is);CHKERRQ(ierr);
      ierr = ISGetIndices(is,&idxs);CHKERRQ(ierr);
      ierr = VecGetArrayRead(xcoorl,&a);CHKERRQ(ierr);
      ierr = VecSetValuesBlockedLocal(lc,n/bs,idxs,a,INSERT_VALUES);CHKERRQ(ierr);
      ierr = VecAssemblyBegin(lc);CHKERRQ(ierr);
      ierr = VecAssemblyEnd(lc);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(xcoorl,&a);CHKERRQ(ierr);
      ierr = ISRestoreIndices(is,&idxs);CHKERRQ(ierr);
      ierr = ISDestroy(&is);CHKERRQ(ierr);
      ierr = MatNullSpaceCreateRigidBody(lc,&lnullsp);CHKERRQ(ierr);
      ierr = VecDestroy(&lc);CHKERRQ(ierr);
    } else if (user.pde == PDE_POISSON) {
      ierr = MatNullSpaceCreate(PETSC_COMM_SELF,PETSC_TRUE,0,NULL,&lnullsp);CHKERRQ(ierr);
    }
    ierr = MatSetNearNullSpace(lA,lnullsp);CHKERRQ(ierr);
    ierr = MatNullSpaceDestroy(&lnullsp);CHKERRQ(ierr);
    ierr = MatISRestoreLocalMat(A,&lA);CHKERRQ(ierr);
  }

  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
  ierr = KSPSetType(ksp,KSPCG);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCBDDC);CHKERRQ(ierr);
  /* ierr = PCBDDCSetDirichletBoundaries(pc,zero);CHKERRQ(ierr); */
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = PetscLogStagePush(stages[0]);CHKERRQ(ierr);
  ierr = KSPSetUp(ksp);CHKERRQ(ierr);
  ierr = PetscLogStagePop();CHKERRQ(ierr);

  ierr = MatCreateVecs(A,&x,&b);CHKERRQ(ierr);
  ierr = VecSetRandom(b,NULL);CHKERRQ(ierr);
  if (nullsp) {
    ierr = MatNullSpaceRemove(nullsp,b);CHKERRQ(ierr);
  }
  ierr = PetscLogStagePush(stages[1]);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
  ierr = PetscLogStagePop();CHKERRQ(ierr);

  /* cleanup */
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = ISDestroy(&zero);CHKERRQ(ierr);
  ierr = PetscFree(e_glo);CHKERRQ(ierr);
  ierr = MatNullSpaceDestroy(&nullsp);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

 test:
   nsize: 8
   filter: grep -v "variant HERMITIAN"
   suffix: bddc_1
   args: -pde_type Poisson -dim 3 -dirichlet 0 -ksp_view -pc_bddc_coarse_redundant_pc_type svd -ksp_error_if_not_converged
 test:
   nsize: 8
   filter: grep -v "variant HERMITIAN"
   suffix: bddc_2
   args: -pde_type Poisson -dim 3 -dirichlet 0 -ksp_view -use_global -pc_bddc_coarse_redundant_pc_type svd -ksp_error_if_not_converged
 test:
   nsize: 8
   filter: grep -v "variant HERMITIAN"
   suffix: bddc_elast
   args: -pde_type Elasticity -cells 7,9,8 -dim 3 -ksp_view -pc_bddc_coarse_redundant_pc_type svd -ksp_error_if_not_converged -pc_bddc_monolithic
 test:
   nsize: 8
   filter: grep -v "variant HERMITIAN"
   suffix: bddc_elast_3lev
   args: -pde_type Elasticity -cells 7,9,8 -dim 3 -ksp_view -pc_bddc_levels 1 -pc_bddc_coarsening_ratio 1 -ksp_error_if_not_converged -pc_bddc_monolithic -pc_bddc_use_faces -pc_bddc_coarse_pc_bddc_corner_selection
 testset:
   nsize: 8
   requires: hpddm
   filter: grep -v "variant HERMITIAN"
   args: -pde_type Elasticity -cells 7,9,8 -dim 3 -ksp_view -pc_bddc_levels 1 -pc_bddc_coarsening_ratio 1 -ksp_error_if_not_converged -pc_bddc_monolithic -pc_bddc_use_faces -pc_bddc_coarse_pc_type hpddm -prefix_push pc_bddc_coarse_ -pc_hpddm_levels_1_sub_pc_type cholesky -pc_hpddm_levels_1_eps_nev 5 -pc_hpddm_levels_1_st_pc_factor_shift_type INBLOCKS -prefix_pop -ksp_type fgmres -ksp_max_it 50 -ksp_converged_reason
   test:
     args: -pc_bddc_coarse_pc_hpddm_coarse_mat_type baij -options_left no
     suffix: bddc_elast_3lev_hpddm_baij
   test:
     requires: !complex
     suffix: bddc_elast_3lev_hpddm
 test:
   nsize: 8
   requires: !single
   filter: grep -v "variant HERMITIAN"
   suffix: bddc_elast_4lev
   args: -pde_type Elasticity -cells 7,9,8 -dim 3 -ksp_view -pc_bddc_levels 2 -pc_bddc_coarsening_ratio 2 -ksp_error_if_not_converged -pc_bddc_monolithic -pc_bddc_use_faces -pc_bddc_coarse_pc_bddc_corner_selection -pc_bddc_coarse_l1_pc_bddc_corner_selection -mat_partitioning_type average -options_left 0
 test:
   nsize: 8
   filter: grep -v "variant HERMITIAN"
   suffix: bddc_elast_deluxe_layers
   args: -pde_type Elasticity -cells 7,9,8 -dim 3 -ksp_view -pc_bddc_coarse_redundant_pc_type svd -ksp_error_if_not_converged -pc_bddc_monolithic -pc_bddc_use_deluxe_scaling -pc_bddc_schur_layers 1
 test:
   nsize: 8
   filter: grep -v "variant HERMITIAN" | sed -e "s/iterations 1[0-9]/iterations 10/g"
   suffix: bddc_elast_dir_approx
   args: -pde_type Elasticity -cells 7,9,8 -dim 3 -ksp_view -pc_bddc_coarse_redundant_pc_type svd -ksp_error_if_not_converged -pc_bddc_monolithic -pc_bddc_dirichlet_pc_type gamg -ksp_converged_reason -pc_bddc_dirichlet_approximate
 test:
   nsize: 8
   filter: grep -v "variant HERMITIAN" | sed -e "s/iterations 1[0-9]/iterations 10/g"
   suffix: bddc_elast_neu_approx
   args: -pde_type Elasticity -cells 7,9,8 -dim 3 -ksp_view -pc_bddc_coarse_redundant_pc_type svd -ksp_error_if_not_converged -pc_bddc_monolithic -pc_bddc_neumann_pc_type gamg -ksp_converged_reason -pc_bddc_neumann_approximate
 test:
   nsize: 8
   filter: grep -v "variant HERMITIAN" | sed -e "s/iterations 1[0-9]/iterations 10/g"
   suffix: bddc_elast_both_approx
   args: -pde_type Elasticity -cells 7,9,8 -dim 3 -ksp_view -pc_bddc_coarse_redundant_pc_type svd -ksp_error_if_not_converged -pc_bddc_monolithic -pc_bddc_dirichlet_pc_type gamg -pc_bddc_neumann_pc_type gamg -ksp_converged_reason -pc_bddc_neumann_approximate -pc_bddc_dirichlet_approximate
 test:
   nsize: 8
   filter: grep -v "variant HERMITIAN"
   suffix: fetidp_1
   args: -pde_type Poisson -dim 3 -dirichlet 0 -ksp_view -ksp_type fetidp -fetidp_ksp_type cg -fetidp_bddc_pc_bddc_coarse_redundant_pc_type svd -ksp_fetidp_fullyredundant -ksp_error_if_not_converged
 test:
   nsize: 8
   filter: grep -v "variant HERMITIAN"
   suffix: fetidp_2
   args: -pde_type Poisson -dim 3 -dirichlet 0 -ksp_view -use_global -ksp_type fetidp -fetidp_ksp_type cg -fetidp_bddc_pc_bddc_coarse_redundant_pc_type svd -ksp_fetidp_fullyredundant -ksp_error_if_not_converged
 test:
   nsize: 8
   filter: grep -v "variant HERMITIAN"
   suffix: fetidp_elast
   args: -pde_type Elasticity -cells 9,7,8 -dim 3 -ksp_view -ksp_type fetidp -fetidp_ksp_type cg -fetidp_bddc_pc_bddc_coarse_redundant_pc_type svd -ksp_fetidp_fullyredundant -ksp_error_if_not_converged -fetidp_bddc_pc_bddc_monolithic
 test:
   nsize: 8
   suffix: hpddm
   requires: hpddm !single
   args: -pde_type Elasticity -cells 12,12 -dim 2 -ksp_converged_reason -pc_type hpddm -pc_hpddm_coarse_correction balanced -pc_hpddm_levels_1_pc_type asm -pc_hpddm_levels_1_pc_asm_overlap 1 -pc_hpddm_levels_1_pc_asm_type basic -pc_hpddm_levels_1_sub_pc_type cholesky -pc_hpddm_levels_1_eps_nev 10 -matis_localmat_type {{aij baij sbaij}shared output} -pc_hpddm_coarse_mat_type {{baij sbaij}shared output} -pc_hpddm_levels_1_st_pc_factor_shift_type INBLOCKS
 testset:
   nsize: 9
   args: -test_assembly -assembled_view -pc_bddc_coarse_redundant_pc_type svd -ksp_error_if_not_converged
   test:
     args: -dim 1 -cells 12 -pde_type Poisson
     suffix: dmda_matis_poiss_1d_loc
     output_file: output/ex71_dmda_matis_poiss_1d.out
   test:
     args: -dim 1 -cells 12 -pde_type Poisson -use_global
     suffix: dmda_matis_poiss_1d_glob
     output_file: output/ex71_dmda_matis_poiss_1d.out
   test:
     args: -dim 1 -cells 12 -pde_type Elasticity
     suffix: dmda_matis_elast_1d_loc
     output_file: output/ex71_dmda_matis_elast_1d.out
   test:
     args: -dim 1 -cells 12 -pde_type Elasticity -use_global
     suffix: dmda_matis_elast_1d_glob
     output_file: output/ex71_dmda_matis_elast_1d.out
   test:
     args: -dim 2 -cells 5,7 -pde_type Poisson
     suffix: dmda_matis_poiss_2d_loc
     output_file: output/ex71_dmda_matis_poiss_2d.out
   test:
     args: -dim 2 -cells 5,7 -pde_type Poisson -use_global
     suffix: dmda_matis_poiss_2d_glob
     output_file: output/ex71_dmda_matis_poiss_2d.out
   test:
     args: -dim 2 -cells 5,7 -pde_type Elasticity
     suffix: dmda_matis_elast_2d_loc
     output_file: output/ex71_dmda_matis_elast_2d.out
   test:
     args: -dim 2 -cells 5,7 -pde_type Elasticity -use_global
     suffix: dmda_matis_elast_2d_glob
     output_file: output/ex71_dmda_matis_elast_2d.out
   test:
     args: -dim 3 -cells 3,3,3 -pde_type Poisson
     suffix: dmda_matis_poiss_3d_loc
     output_file: output/ex71_dmda_matis_poiss_3d.out
   test:
     args: -dim 3 -cells 3,3,3 -pde_type Poisson -use_global
     suffix: dmda_matis_poiss_3d_glob
     output_file: output/ex71_dmda_matis_poiss_3d.out
   test:
     args: -dim 3 -cells 3,3,3 -pde_type Elasticity
     suffix: dmda_matis_elast_3d_loc
     output_file: output/ex71_dmda_matis_elast_3d.out
   test:
     args: -dim 3 -cells 3,3,3 -pde_type Elasticity -use_global
     suffix: dmda_matis_elast_3d_glob
     output_file: output/ex71_dmda_matis_elast_3d.out
 test:
   nsize: 8
   filter: grep -v "variant HERMITIAN"
   suffix: bddc_elast_deluxe_layers_adapt
   requires: mumps !complex
   args: -pde_type Elasticity -cells 7,9,8 -dim 3 -ksp_view -pc_bddc_coarse_redundant_pc_type svd -ksp_error_if_not_converged -pc_bddc_monolithic -sub_schurs_mat_solver_type mumps -pc_bddc_use_deluxe_scaling -pc_bddc_adaptive_threshold 2.0 -pc_bddc_schur_layers {{1 10}separate_output} -pc_bddc_adaptive_userdefined {{0 1}separate output}
 test:
   nsize: 8
   suffix: bddc_elast_deluxe_layers_adapt_mkl_pardiso
   requires: mkl_pardiso !complex
   args: -pde_type Elasticity -cells 7,9,8 -dim 3 -ksp_converged_reason -pc_bddc_coarse_redundant_pc_type svd -ksp_error_if_not_converged -pc_bddc_monolithic -sub_schurs_mat_solver_type mkl_pardiso -sub_schurs_mat_mkl_pardiso_65 1 -pc_bddc_use_deluxe_scaling -pc_bddc_adaptive_threshold 2.0 -pc_bddc_schur_layers {{1 10}separate_output} -pc_bddc_adaptive_userdefined {{0 1}separate output}
 test:
   nsize: 8
   filter: grep -v "variant HERMITIAN"
   suffix: bddc_cusparse
   requires: cuda
   args: -pde_type Poisson -cells 7,9,8 -dim 3 -ksp_view -pc_bddc_coarse_redundant_pc_type svd -ksp_error_if_not_converged -pc_bddc_dirichlet_pc_type cholesky -pc_bddc_dirichlet_pc_factor_mat_solver_type cusparse -pc_bddc_dirichlet_pc_factor_mat_ordering_type nd -pc_bddc_neumann_pc_type cholesky -pc_bddc_neumann_pc_factor_mat_solver_type cusparse -pc_bddc_neumann_pc_factor_mat_ordering_type nd -matis_localmat_type aijcusparse
 test:
   nsize: 8
   filter: grep -v "variant HERMITIAN"
   suffix: bddc_elast_deluxe_layers_adapt_cuda
   requires: mumps cuda viennacl
   args: -pde_type Elasticity -cells 7,9,8 -dim 3 -ksp_view -pc_bddc_coarse_redundant_pc_type svd -ksp_error_if_not_converged -pc_bddc_monolithic -sub_schurs_mat_solver_type mumps -pc_bddc_use_deluxe_scaling -pc_bddc_adaptive_threshold 2.0 -pc_bddc_schur_layers {{1 10}separate_output} -pc_bddc_adaptive_userdefined {{0 1}separate output} -matis_localmat_type seqaijviennacl -sub_schurs_schur_mat_type {{seqdensecuda seqdense}}
 test:
   nsize: 8
   filter: grep -v "variant HERMITIAN" | grep -v "I-node routines" | sed -e "s/seqaijviennacl/seqaij/g"
   suffix: bddc_elast_deluxe_layers_adapt_cuda_approx
   requires: mumps cuda viennacl
   args: -pde_type Elasticity -cells 7,9,8 -dim 3 -ksp_view -pc_bddc_coarse_redundant_pc_type svd -ksp_error_if_not_converged -pc_bddc_monolithic -sub_schurs_mat_solver_type mumps -pc_bddc_use_deluxe_scaling -pc_bddc_adaptive_threshold 2.0 -pc_bddc_schur_layers 1 -matis_localmat_type {{seqaij seqaijviennacl}} -sub_schurs_schur_mat_type {{seqdensecuda seqdense}} -pc_bddc_dirichlet_pc_type gamg -pc_bddc_dirichlet_approximate -pc_bddc_neumann_pc_type gamg -pc_bddc_neumann_approximate
 test:
   nsize: 8
   suffix: bddc_elast_deluxe_layers_adapt_mkl_pardiso_cuda
   requires: mkl_pardiso cuda viennacl
   args: -pde_type Elasticity -cells 7,9,8 -dim 3 -ksp_converged_reason -pc_bddc_coarse_redundant_pc_type svd -ksp_error_if_not_converged -pc_bddc_monolithic -sub_schurs_mat_solver_type mkl_pardiso -sub_schurs_mat_mkl_pardiso_65 1 -pc_bddc_use_deluxe_scaling -pc_bddc_adaptive_threshold 2.0 -pc_bddc_schur_layers {{1 10}separate_output} -pc_bddc_adaptive_userdefined {{0 1}separate output} -matis_localmat_type seqaijviennacl -sub_schurs_schur_mat_type {{seqdensecuda seqdense}}

 testset:
   nsize: 2
   output_file: output/ex71_aij_dmda_preall.out
   filter: sed -e "s/CONVERGED_RTOL iterations 7/CONVERGED_RTOL iterations 6/g"
   args: -pde_type Poisson -dim 1 -cells 6 -pc_type none -ksp_converged_reason
   test:
     suffix: aijviennacl_dmda_preall
     requires: viennacl
     args: -dm_mat_type aijviennacl -dm_preallocate_only {{0 1}} -dirichlet {{0 1}}
   # -dm_preallocate_only 0 is broken
   test:
     suffix: aijcusparse_dmda_preall
     requires: cuda
     args: -dm_mat_type aijcusparse -dm_preallocate_only -dirichlet {{0 1}}
   test:
     suffix: aij_dmda_preall
     args: -dm_mat_type aij -dm_preallocate_only {{0 1}} -dirichlet {{0 1}}

TEST*/
