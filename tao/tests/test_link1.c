#include "tao.h"

int main(int argc, char *argv[]) {
    int info;
    info = PetscInitialize(&argc, &argv, 0, 0);
    info = PetscFinalize();
    return 0;
}
