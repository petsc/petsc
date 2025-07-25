#pragma once

#include <petscsys.h>

static const PetscReal PetscDTKMVTriQuad_2_weights[] = {PetscRealConstant(6.66666666666666666666666666666666667e-01)};

static const PetscReal PetscDTKMVTriQuad_2_orbits[] = {PetscRealConstant(0.), PetscRealConstant(1.)};

static const PetscReal PetscDTKMVTriQuad_3_weights[] = {PetscRealConstant(9.00000000000000000000000000000000000e-01), PetscRealConstant(1.00000000000000000000000000000000000e-01), PetscRealConstant(2.66666666666666666666666666666666667e-01)};

static const PetscReal PetscDTKMVTriQuad_3_orbits[] = {PetscRealConstant(3.33333333333333333333333333333333333e-01), PetscRealConstant(0.), PetscRealConstant(1.), PetscRealConstant(0.5), PetscRealConstant(0.)};

//static const PetscReal PetscDTKMVTriQuad_4_weights[] = {PetscRealConstant(2.97458260496411633861021347020041087e-02), PetscRealConstant(4.41554115680821477036048276247362094e-01), PetscRealConstant(9.76833624681020131222581278586502316e-02)};
static const PetscReal PetscDTKMVTriQuad_4_weights[] = {PetscRealConstant(2.97458260496411633861021347020041087e-02), PetscRealConstant(4.41554115680821477036048276247362094e-01), PetscRealConstant(9.76833624681020131222581278586502316e-02)};

// alpha = PetscRealConstant(2.93469555909040190389804004439162530e-01)
// beta  = PetscRealConstant(2.07345175663590924261827821255273313e-01)
static const PetscReal PetscDTKMVTriQuad_4_orbits[] = {PetscRealConstant(0.), PetscRealConstant(1.), PetscRealConstant(7.92654824336409075738172178744726686e-01), PetscRealConstant(2.07345175663590924261827821255273313e-01), PetscRealConstant(7.06530444090959809610195995560837469e-01), PetscRealConstant(2.93469555909040190389804004439162530e-01), PetscRealConstant(0.)};

static const PetscInt PetscDTKMVTriQuad_max_degree = 4;

static const PetscInt PetscDTKMVTriQuad_num_nodes[] = {3, 3, 3, 7, 12};

static const PetscInt PetscDTKMVTriQuad_num_orbits[] = {0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 2, 0, 0, 2, 1};

static const PetscReal *PetscDTKMVTriQuad_weights[] = {PetscDTKMVTriQuad_2_weights, PetscDTKMVTriQuad_2_weights, PetscDTKMVTriQuad_2_weights, PetscDTKMVTriQuad_3_weights, PetscDTKMVTriQuad_4_weights};

static const PetscReal *PetscDTKMVTriQuad_orbits[] = {PetscDTKMVTriQuad_2_orbits, PetscDTKMVTriQuad_2_orbits, PetscDTKMVTriQuad_2_orbits, PetscDTKMVTriQuad_3_orbits, PetscDTKMVTriQuad_4_orbits};
