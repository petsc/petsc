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
  PetscBool    use_composite_pc;
  PetscBool    random_initial_guess;
  PetscBool    random_real;
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  const char    *pdeTypes[2] = {"Poisson", "Elasticity"};
  PetscInt       n,pde;

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
  options->use_composite_pc = PETSC_FALSE;
  options->random_initial_guess = PETSC_FALSE;
  options->random_real = PETSC_FALSE;

  PetscOptionsBegin(comm,NULL,"Problem Options",NULL);
  pde  = options->pde;
  PetscCall(PetscOptionsEList("-pde_type","The PDE type",__FILE__,pdeTypes,2,pdeTypes[options->pde],&pde,NULL));
  options->pde = (PDEType)pde;
  PetscCall(PetscOptionsInt("-dim","The topological mesh dimension",__FILE__,options->dim,&options->dim,NULL));
  PetscCall(PetscOptionsIntArray("-cells","The mesh division",__FILE__,options->cells,(n=3,&n),NULL));
  PetscCall(PetscOptionsBoolArray("-periodicity","The mesh periodicity",__FILE__,options->per,(n=3,&n),NULL));
  PetscCall(PetscOptionsBool("-use_global","Test MatSetValues",__FILE__,options->useglobal,&options->useglobal,NULL));
  PetscCall(PetscOptionsBool("-dirichlet","Use dirichlet BC",__FILE__,options->dirbc,&options->dirbc,NULL));
  PetscCall(PetscOptionsBool("-test_assembly","Test MATIS assembly",__FILE__,options->test,&options->test,NULL));
  PetscCall(PetscOptionsBool("-use_composite_pc","Multiplicative composite with BDDC + Richardson/Jacobi",__FILE__,options->use_composite_pc,&options->use_composite_pc,NULL));
  PetscCall(PetscOptionsBool("-random_initial_guess","Solve A x = 0 with random initial guess, instead of A x = b with random b",__FILE__,options->random_initial_guess,&options->random_initial_guess,NULL));
  PetscCall(PetscOptionsBool("-random_real","Use real-valued b (or x, if -random_initial_guess) instead of default scalar type",__FILE__,options->random_real,&options->random_real,NULL));
  PetscOptionsEnd();

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
    default: SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Unsupported dimension %" PetscInt_FMT,options->dim);
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
    default: SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Unsupported dimension %" PetscInt_FMT,options->dim);
    }
    break;
  default: SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Unsupported PDE %d",options->pde);
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
  PetscInt               nodes[3];
  PetscBool              ismatis;
#if defined(PETSC_USE_LOG)
  PetscLogStage          stages[2];
#endif

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD,&user));
  for (i=0; i<3; i++) nodes[i] = user.cells[i] + !user.per[i];
  switch (user.dim) {
  case 3:
    PetscCall(DMDACreate3d(PETSC_COMM_WORLD,user.per[0] ? DM_BOUNDARY_PERIODIC : DM_BOUNDARY_NONE,
                           user.per[1] ? DM_BOUNDARY_PERIODIC : DM_BOUNDARY_NONE,
                           user.per[2] ? DM_BOUNDARY_PERIODIC : DM_BOUNDARY_NONE,
                           DMDA_STENCIL_BOX,nodes[0],nodes[1],nodes[2],
                           PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,user.dof,
                           1,PETSC_NULL,PETSC_NULL,PETSC_NULL,&da));
    break;
  case 2:
    PetscCall(DMDACreate2d(PETSC_COMM_WORLD,user.per[0] ? DM_BOUNDARY_PERIODIC : DM_BOUNDARY_NONE,
                           user.per[1] ? DM_BOUNDARY_PERIODIC : DM_BOUNDARY_NONE,
                           DMDA_STENCIL_BOX,nodes[0],nodes[1],
                           PETSC_DECIDE,PETSC_DECIDE,user.dof,
                           1,PETSC_NULL,PETSC_NULL,&da));
    break;
  case 1:
    PetscCall(DMDACreate1d(PETSC_COMM_WORLD,user.per[0] ? DM_BOUNDARY_PERIODIC : DM_BOUNDARY_NONE,
                           nodes[0],user.dof,1,PETSC_NULL,&da));
    break;
  default: SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Unsupported dimension %" PetscInt_FMT,user.dim);
  }

  PetscCall(PetscLogStageRegister("KSPSetUp",&stages[0]));
  PetscCall(PetscLogStageRegister("KSPSolve",&stages[1]));

  PetscCall(DMSetMatType(da,MATIS));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMDASetElementType(da,DMDA_ELEMENT_Q1));
  PetscCall(DMSetUp(da));
  {
    PetscInt M,N,P;
    PetscCall(DMDAGetInfo(da,0,&M,&N,&P,0,0,0,0,0,0,0,0,0));
    switch (user.dim) {
    case 3:
      user.cells[2] = P - !user.per[2];
    case 2:
      user.cells[1] = N - !user.per[1];
    case 1:
      user.cells[0] = M - !user.per[0];
      break;
    default: SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Unsupported dimension %" PetscInt_FMT,user.dim);
    }
  }
  PetscCall(DMDASetUniformCoordinates(da,0.0,1.0*user.cells[0],0.0,1.0*user.cells[1],0.0,1.0*user.cells[2]));
  PetscCall(DMGetCoordinates(da,&xcoor));

  PetscCall(DMCreateMatrix(da,&A));
  PetscCall(MatSetFromOptions(A));
  PetscCall(DMGetLocalToGlobalMapping(da,&map));
  PetscCall(DMDAGetElements(da,&nel,&nen,&e_loc));
  if (user.useglobal) {
    PetscCall(PetscMalloc1(nel*nen,&e_glo));
    PetscCall(ISLocalToGlobalMappingApplyBlock(map,nen*nel,e_loc,e_glo));
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

    PetscCheck(nen <= 8,PETSC_COMM_WORLD,PETSC_ERR_SUP,"Not coded");
    if (!e_glo) {
      for (j=0;j<nen;j++) idxs[j] = e_loc[i*nen+ord[j]];
      PetscCall(MatSetValuesBlockedLocal(A,nen,idxs,nen,idxs,user.elemMat,ADD_VALUES));
    } else {
      for (j=0;j<nen;j++) idxs[j] = e_glo[i*nen+ord[j]];
      PetscCall(MatSetValuesBlocked(A,nen,idxs,nen,idxs,user.elemMat,ADD_VALUES));
    }
  }
  PetscCall(DMDARestoreElements(da,&nel,&nen,&e_loc));
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatSetOption(A,MAT_SPD,PETSC_TRUE));
  PetscCall(MatSetOption(A,MAT_SPD_ETERNAL,PETSC_TRUE));

  /* Boundary conditions */
  zero = NULL;
  if (user.dirbc) { /* fix one side of DMDA */
    Vec         nat,glob;
    PetscScalar *vals;
    PetscInt    n,*idx,j,st;

    n    = PetscGlobalRank ? 0 : (user.cells[1]+1)*(user.cells[2]+1);
    PetscCall(ISCreateStride(PETSC_COMM_WORLD,n,0,user.cells[0]+1,&zero));
    if (user.dof > 1) { /* zero all components */
      const PetscInt *idx;
      IS             bzero;

      PetscCall(ISGetIndices(zero,(const PetscInt**)&idx));
      PetscCall(ISCreateBlock(PETSC_COMM_WORLD,user.dof,n,idx,PETSC_COPY_VALUES,&bzero));
      PetscCall(ISRestoreIndices(zero,(const PetscInt**)&idx));
      PetscCall(ISDestroy(&zero));
      zero = bzero;
    }
    /* map indices from natural to global */
    PetscCall(DMDACreateNaturalVector(da,&nat));
    PetscCall(ISGetLocalSize(zero,&n));
    PetscCall(PetscMalloc1(n,&vals));
    for (i=0;i<n;i++) vals[i] = 1.0;
    PetscCall(ISGetIndices(zero,(const PetscInt**)&idx));
    PetscCall(VecSetValues(nat,n,idx,vals,INSERT_VALUES));
    PetscCall(ISRestoreIndices(zero,(const PetscInt**)&idx));
    PetscCall(PetscFree(vals));
    PetscCall(VecAssemblyBegin(nat));
    PetscCall(VecAssemblyEnd(nat));
    PetscCall(DMCreateGlobalVector(da,&glob));
    PetscCall(DMDANaturalToGlobalBegin(da,nat,INSERT_VALUES,glob));
    PetscCall(DMDANaturalToGlobalEnd(da,nat,INSERT_VALUES,glob));
    PetscCall(VecDestroy(&nat));
    PetscCall(ISDestroy(&zero));
    PetscCall(VecGetLocalSize(glob,&n));
    PetscCall(PetscMalloc1(n,&idx));
    PetscCall(VecGetOwnershipRange(glob,&st,NULL));
    PetscCall(VecGetArray(glob,&vals));
    for (i=0,j=0;i<n;i++) if (PetscRealPart(vals[i]) == 1.0) idx[j++] = i + st;
    PetscCall(VecRestoreArray(glob,&vals));
    PetscCall(VecDestroy(&glob));
    PetscCall(ISCreateGeneral(PETSC_COMM_WORLD,j,idx,PETSC_OWN_POINTER,&zero));
    PetscCall(MatZeroRowsColumnsIS(A,zero,1.0,NULL,NULL));
  } else {
    switch (user.pde) {
    case PDE_POISSON:
      PetscCall(MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,0,NULL,&nullsp));
      break;
    case PDE_ELASTICITY:
      PetscCall(MatNullSpaceCreateRigidBody(xcoor,&nullsp));
      break;
    }
    /* with periodic BC and Elasticity, just the displacements are in the nullspace
       this is no harm since we eliminate all the components of the rhs */
    PetscCall(MatSetNullSpace(A,nullsp));
  }

  if (user.test) {
    Mat AA;

    PetscCall(MatConvert(A,MATAIJ,MAT_INITIAL_MATRIX,&AA));
    PetscCall(MatViewFromOptions(AA,NULL,"-assembled_view"));
    PetscCall(MatDestroy(&AA));
  }

  /* Attach near null space for elasticity */
  if (user.pde == PDE_ELASTICITY) {
    MatNullSpace nearnullsp;

    PetscCall(MatNullSpaceCreateRigidBody(xcoor,&nearnullsp));
    PetscCall(MatSetNearNullSpace(A,nearnullsp));
    PetscCall(MatNullSpaceDestroy(&nearnullsp));
  }

  /* we may want to use MG for the local solvers: attach local nearnullspace to the local matrices */
  PetscCall(DMGetCoordinatesLocal(da,&xcoorl));
  PetscCall(PetscObjectTypeCompare((PetscObject)A,MATIS,&ismatis));
  if (ismatis) {
    MatNullSpace lnullsp = NULL;
    Mat          lA;

    PetscCall(MatISGetLocalMat(A,&lA));
    if (user.pde == PDE_ELASTICITY) {
      Vec                    lc;
      ISLocalToGlobalMapping l2l;
      IS                     is;
      const PetscScalar      *a;
      const PetscInt         *idxs;
      PetscInt               n,bs;

      /* when using a DMDA, the local matrices have an additional local-to-local map
         that maps from the DA local ordering to the ordering induced by the elements */
      PetscCall(MatCreateVecs(lA,&lc,NULL));
      PetscCall(MatGetLocalToGlobalMapping(lA,&l2l,NULL));
      PetscCall(VecSetLocalToGlobalMapping(lc,l2l));
      PetscCall(VecSetOption(lc,VEC_IGNORE_NEGATIVE_INDICES,PETSC_TRUE));
      PetscCall(VecGetLocalSize(xcoorl,&n));
      PetscCall(VecGetBlockSize(xcoorl,&bs));
      PetscCall(ISCreateStride(PETSC_COMM_SELF,n/bs,0,1,&is));
      PetscCall(ISGetIndices(is,&idxs));
      PetscCall(VecGetArrayRead(xcoorl,&a));
      PetscCall(VecSetValuesBlockedLocal(lc,n/bs,idxs,a,INSERT_VALUES));
      PetscCall(VecAssemblyBegin(lc));
      PetscCall(VecAssemblyEnd(lc));
      PetscCall(VecRestoreArrayRead(xcoorl,&a));
      PetscCall(ISRestoreIndices(is,&idxs));
      PetscCall(ISDestroy(&is));
      PetscCall(MatNullSpaceCreateRigidBody(lc,&lnullsp));
      PetscCall(VecDestroy(&lc));
    } else if (user.pde == PDE_POISSON) {
      PetscCall(MatNullSpaceCreate(PETSC_COMM_SELF,PETSC_TRUE,0,NULL,&lnullsp));
    }
    PetscCall(MatSetNearNullSpace(lA,lnullsp));
    PetscCall(MatNullSpaceDestroy(&lnullsp));
    PetscCall(MatISRestoreLocalMat(A,&lA));
  }

  PetscCall(KSPCreate(PETSC_COMM_WORLD,&ksp));
  PetscCall(KSPSetOperators(ksp,A,A));
  PetscCall(KSPSetType(ksp,KSPCG));
  PetscCall(KSPGetPC(ksp,&pc));
  if (user.use_composite_pc) {
    PC pcksp,pcjacobi;
    KSP ksprich;
    PetscCall(PCSetType(pc,PCCOMPOSITE));
    PetscCall(PCCompositeSetType(pc,PC_COMPOSITE_MULTIPLICATIVE));
    PetscCall(PCCompositeAddPCType(pc,PCBDDC));
    PetscCall(PCCompositeAddPCType(pc,PCKSP));
    PetscCall(PCCompositeGetPC(pc,1,&pcksp));
    PetscCall(PCKSPGetKSP(pcksp,&ksprich));
    PetscCall(KSPSetType(ksprich,KSPRICHARDSON));
    PetscCall(KSPSetTolerances(ksprich,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,1));
    PetscCall(KSPSetNormType(ksprich,KSP_NORM_NONE));
    PetscCall(KSPSetConvergenceTest(ksprich,KSPConvergedSkip,NULL,NULL));
    PetscCall(KSPGetPC(ksprich,&pcjacobi));
    PetscCall(PCSetType(pcjacobi,PCJACOBI));
  } else {
    PetscCall(PCSetType(pc,PCBDDC));
  }
  /* PetscCall(PCBDDCSetDirichletBoundaries(pc,zero)); */
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(PetscLogStagePush(stages[0]));
  PetscCall(KSPSetUp(ksp));
  PetscCall(PetscLogStagePop());

  PetscCall(MatCreateVecs(A,&x,&b));
  if (user.random_initial_guess) {
    /* Solving A x = 0 with random initial guess allows Arnoldi to run for more iterations, thereby yielding a more
     * complete Hessenberg matrix and more accurate eigenvalues. */
    PetscCall(VecZeroEntries(b));
    PetscCall(VecSetRandom(x,NULL));
    if (user.random_real) PetscCall(VecRealPart(x));
    if (nullsp) PetscCall(MatNullSpaceRemove(nullsp,x));
    PetscCall(KSPSetInitialGuessNonzero(ksp,PETSC_TRUE));
    PetscCall(KSPSetComputeEigenvalues(ksp,PETSC_TRUE));
    PetscCall(KSPGMRESSetRestart(ksp,100));
  } else {
    PetscCall(VecSetRandom(b,NULL));
    if (user.random_real) PetscCall(VecRealPart(x));
    if (nullsp) PetscCall(MatNullSpaceRemove(nullsp,b));
  }
  PetscCall(PetscLogStagePush(stages[1]));
  PetscCall(KSPSolve(ksp,b,x));
  PetscCall(PetscLogStagePop());

  /* cleanup */
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(ISDestroy(&zero));
  PetscCall(PetscFree(e_glo));
  PetscCall(MatNullSpaceDestroy(&nullsp));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(MatDestroy(&A));
  PetscCall(DMDestroy(&da));
  PetscCall(PetscFinalize());
  return 0;
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
   requires: hpddm slepc defined(PETSC_HAVE_DYNAMIC_LIBRARIES) defined(PETSC_USE_SHARED_LIBRARIES)
   # on some architectures, this test will converge in 19 or 21 iterations
   filter: grep -v "variant HERMITIAN" | grep -v " tolerance"  | sed -e "s/CONVERGED_RTOL iterations [1-2][91]\{0,1\}$/CONVERGED_RTOL iterations 20/g"
   args: -pde_type Elasticity -cells 7,9,8 -dim 3 -ksp_view -pc_bddc_levels 1 -pc_bddc_coarsening_ratio 1 -ksp_error_if_not_converged -pc_bddc_monolithic -pc_bddc_use_faces -pc_bddc_coarse_pc_type hpddm -prefix_push pc_bddc_coarse_ -pc_hpddm_levels_1_sub_pc_type cholesky -pc_hpddm_levels_1_eps_nev 6 -pc_hpddm_levels_1_st_pc_factor_shift_type INBLOCKS -prefix_pop -ksp_type fgmres -ksp_max_it 50 -ksp_converged_reason
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
   args: -pde_type Elasticity -cells 7,9,8 -dim 3 -ksp_view -pc_bddc_coarse_redundant_pc_type svd -ksp_error_if_not_converged -pc_bddc_monolithic -pc_bddc_dirichlet_pc_type gamg -pc_bddc_dirichlet_pc_gamg_esteig_ksp_max_it 10 -ksp_converged_reason -pc_bddc_dirichlet_approximate
 test:
   nsize: 8
   filter: grep -v "variant HERMITIAN" | sed -e "s/iterations 1[0-9]/iterations 10/g"
   suffix: bddc_elast_neu_approx
   args: -pde_type Elasticity -cells 7,9,8 -dim 3 -ksp_view -pc_bddc_coarse_redundant_pc_type svd -ksp_error_if_not_converged -pc_bddc_monolithic -pc_bddc_neumann_pc_type gamg -pc_bddc_neumann_pc_gamg_esteig_ksp_max_it 10 -ksp_converged_reason -pc_bddc_neumann_approximate
 test:
   nsize: 8
   filter: grep -v "variant HERMITIAN" | sed -e "s/iterations 1[0-9]/iterations 10/g"
   suffix: bddc_elast_both_approx
   args: -pde_type Elasticity -cells 7,9,8 -dim 3 -ksp_view -pc_bddc_coarse_redundant_pc_type svd -ksp_error_if_not_converged -pc_bddc_monolithic -pc_bddc_dirichlet_pc_type gamg -pc_bddc_dirichlet_pc_gamg_esteig_ksp_max_it 10 -pc_bddc_neumann_pc_type gamg -pc_bddc_neumann_pc_gamg_esteig_ksp_max_it 10 -ksp_converged_reason -pc_bddc_neumann_approximate -pc_bddc_dirichlet_approximate
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
 testset:
   nsize: 8
   requires: hpddm slepc defined(PETSC_HAVE_DYNAMIC_LIBRARIES) defined(PETSC_USE_SHARED_LIBRARIES)
   args: -pde_type Elasticity -cells 12,12 -dim 2 -ksp_converged_reason -pc_type hpddm -pc_hpddm_coarse_correction balanced -pc_hpddm_levels_1_pc_type asm -pc_hpddm_levels_1_pc_asm_overlap 1 -pc_hpddm_levels_1_pc_asm_type basic -pc_hpddm_levels_1_sub_pc_type cholesky -pc_hpddm_levels_1_eps_nev 10 -matis_localmat_type {{aij baij sbaij}shared output} -pc_hpddm_coarse_mat_type {{baij sbaij}shared output} -pc_hpddm_levels_1_st_pc_factor_shift_type INBLOCKS
   test:
     suffix: hpddm
     output_file: output/ex71_hpddm.out
   test:
     args: -pc_hpddm_levels_1_eps_type lapack -pc_hpddm_levels_1_eps_smallest_magnitude -pc_hpddm_levels_1_st_type shift
     suffix: hpddm_lapack
     output_file: output/ex71_hpddm.out
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
   args: -pde_type Elasticity -cells 7,9,8 -dim 3 -ksp_converged_reason -pc_bddc_coarse_redundant_pc_type svd -ksp_error_if_not_converged -pc_bddc_monolithic -sub_schurs_mat_solver_type mumps -pc_bddc_use_deluxe_scaling -pc_bddc_adaptive_threshold 2.0 -pc_bddc_schur_layers {{1 10}separate_output} -pc_bddc_adaptive_userdefined {{0 1}separate output} -sub_schurs_schur_mat_type seqdense
 # gitlab runners have a quite old MKL (2016) which interacts badly with AMD machines (not Intel-based ones!)
 # this is the reason behind the filtering rule
 test:
   nsize: 8
   suffix: bddc_elast_deluxe_layers_adapt_mkl_pardiso
   filter: sed -e "s/CONVERGED_RTOL iterations [1-2][0-9]/CONVERGED_RTOL iterations 13/g" | sed -e "s/CONVERGED_RTOL iterations 6/CONVERGED_RTOL iterations 5/g"
   requires: mkl_pardiso !complex
   args: -pde_type Elasticity -cells 7,9,8 -dim 3 -ksp_converged_reason -pc_bddc_coarse_redundant_pc_type svd -ksp_error_if_not_converged -pc_bddc_monolithic -sub_schurs_mat_solver_type mkl_pardiso -sub_schurs_mat_mkl_pardiso_65 1 -pc_bddc_use_deluxe_scaling -pc_bddc_adaptive_threshold 2.0 -pc_bddc_schur_layers {{1 10}separate_output} -pc_bddc_adaptive_userdefined {{0 1}separate output} -sub_schurs_schur_mat_type seqdense
 test:
   nsize: 8
   filter: grep -v "variant HERMITIAN"
   suffix: bddc_cusparse
   # no kokkos since it seems kokkos's resource demand is too much with 8 ranks and the test will fail on cuda related initialization.
   requires: cuda !kokkos
   args: -pde_type Poisson -cells 7,9,8 -dim 3 -ksp_view -pc_bddc_coarse_redundant_pc_type svd -ksp_error_if_not_converged -pc_bddc_dirichlet_pc_type cholesky -pc_bddc_dirichlet_pc_factor_mat_solver_type cusparse -pc_bddc_dirichlet_pc_factor_mat_ordering_type nd -pc_bddc_neumann_pc_type cholesky -pc_bddc_neumann_pc_factor_mat_solver_type cusparse -pc_bddc_neumann_pc_factor_mat_ordering_type nd -matis_localmat_type aijcusparse
 test:
   nsize: 8
   filter: grep -v "variant HERMITIAN"
   suffix: bddc_elast_deluxe_layers_adapt_cuda
   requires: !complex mumps cuda defined(PETSC_HAVE_CUSOLVERDNDPOTRI)
   args: -pde_type Elasticity -cells 7,9,8 -dim 3 -ksp_converged_reason -pc_bddc_coarse_redundant_pc_type svd -ksp_error_if_not_converged -pc_bddc_monolithic -sub_schurs_mat_solver_type mumps -pc_bddc_use_deluxe_scaling -pc_bddc_adaptive_threshold 2.0 -pc_bddc_schur_layers {{1 10}separate_output} -pc_bddc_adaptive_userdefined {{0 1}separate output} -matis_localmat_type seqaijcusparse -sub_schurs_schur_mat_type {{seqdensecuda seqdense}}
 test:
   nsize: 8
   filter: grep -v "variant HERMITIAN" | grep -v "I-node routines" | sed -e "s/seqaijcusparse/seqaij/g"
   suffix: bddc_elast_deluxe_layers_adapt_cuda_approx
   requires: !complex mumps cuda defined(PETSC_HAVE_CUSOLVERDNDPOTRI)
   args: -pde_type Elasticity -cells 7,9,8 -dim 3 -ksp_view -pc_bddc_coarse_redundant_pc_type svd -ksp_error_if_not_converged -pc_bddc_monolithic -sub_schurs_mat_solver_type mumps -pc_bddc_use_deluxe_scaling -pc_bddc_adaptive_threshold 2.0 -pc_bddc_schur_layers 1 -matis_localmat_type {{seqaij seqaijcusparse}separate output} -sub_schurs_schur_mat_type {{seqdensecuda seqdense}} -pc_bddc_dirichlet_pc_type gamg -pc_bddc_dirichlet_approximate -pc_bddc_neumann_pc_type gamg -pc_bddc_neumann_approximate -pc_bddc_dirichlet_pc_gamg_esteig_ksp_max_it 10 -pc_bddc_neumann_pc_gamg_esteig_ksp_max_it 10
 test:
   nsize: 8
   suffix: bddc_elast_deluxe_layers_adapt_mkl_pardiso_cuda
   requires: !complex mkl_pardiso cuda defined(PETSC_HAVE_CUSOLVERDNDPOTRI)
   filter: sed -e "s/CONVERGED_RTOL iterations 6/CONVERGED_RTOL iterations 5/g"
   args: -pde_type Elasticity -cells 7,9,8 -dim 3 -ksp_converged_reason -pc_bddc_coarse_redundant_pc_type svd -ksp_error_if_not_converged -pc_bddc_monolithic -sub_schurs_mat_solver_type mkl_pardiso -sub_schurs_mat_mkl_pardiso_65 1 -pc_bddc_use_deluxe_scaling -pc_bddc_adaptive_threshold 2.0 -pc_bddc_schur_layers {{1 10}separate_output} -pc_bddc_adaptive_userdefined {{0 1}separate output} -matis_localmat_type seqaijcusparse -sub_schurs_schur_mat_type {{seqdensecuda seqdense}}

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
 testset:
   nsize: 4
   args: -dim 2 -cells 16,16 -periodicity 1,1 -random_initial_guess -random_real -sub_0_pc_bddc_switch_static -use_composite_pc -ksp_monitor -ksp_converged_reason -ksp_type gmres -ksp_view_singularvalues -ksp_view_eigenvalues -sub_0_pc_bddc_use_edges 0 -sub_0_pc_bddc_coarse_pc_type svd -sub_1_ksp_ksp_max_it 1 -sub_1_ksp_ksp_richardson_scale 2.3
   test:
     args: -sub_0_pc_bddc_interface_ext_type lump
     suffix: composite_bddc_lumped
   test:
     requires: !single
     args: -sub_0_pc_bddc_interface_ext_type dirichlet
     suffix: composite_bddc_dirichlet

 testset:
   nsize: 8
   filter: grep -v "variant HERMITIAN"
   args: -cells 7,9,8 -dim 3 -ksp_view -ksp_error_if_not_converged -pc_type mg -pc_mg_levels 2 -pc_mg_galerkin -pc_mg_adapt_interp_coarse_space gdsw -mg_levels_pc_type asm -mg_levels_sub_pc_type icc -mg_coarse_redundant_pc_type cholesky
   test:
     suffix: gdsw_poisson
     args: -pde_type Poisson
   test:
     requires: mumps !complex
     suffix: gdsw_poisson_adaptive
     args: -pde_type Poisson -mg_levels_gdsw_tolerance 0.01 -ksp_monitor_singular_value -mg_levels_gdsw_userdefined {{0 1}separate output} -mg_levels_gdsw_pseudo_pc_type qr
   test:
     suffix: gdsw_elast
     args: -pde_type Elasticity
   test:
     requires: hpddm
     suffix: gdsw_elast_hpddm
     args: -pde_type Elasticity -mg_levels_gdsw_ksp_type hpddm -mg_levels_gdsw_ksp_hpddm_type cg
   test:
     requires: mumps !complex
     suffix: gdsw_elast_adaptive
     args: -pde_type Elasticity -mg_levels_gdsw_tolerance 0.01 -ksp_monitor_singular_value -mg_levels_gdsw_userdefined {{0 1}separate output}

TEST*/
