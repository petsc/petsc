static char help[] = "The main goal of this code is to retrieve the original element numbers as found in the "
                     "initial partitions (sInitialPartition)... but after the call to DMPlexDistribute";

#include <petsc.h>

/* Coordinates of a 2x5 rectangular mesh of quads : */
PetscReal sCoords2x5Mesh[18][2] = {
  {0.00000000000000000e+00, 0.00000000000000000e+00},
  {2.00000000000000000e+00, 0.00000000000000000e+00},
  {0.00000000000000000e+00, 1.00000000000000000e+00},
  {2.00000000000000000e+00, 1.00000000000000000e+00},
  {9.99999999997387978e-01, 0.00000000000000000e+00},
  {9.99999999997387978e-01, 1.00000000000000000e+00},
  {0.00000000000000000e+00, 2.00000000000000011e-01},
  {0.00000000000000000e+00, 4.00000000000000022e-01},
  {0.00000000000000000e+00, 5.99999999999999978e-01},
  {0.00000000000000000e+00, 8.00000000000000044e-01},
  {2.00000000000000000e+00, 2.00000000000000011e-01},
  {2.00000000000000000e+00, 4.00000000000000022e-01},
  {2.00000000000000000e+00, 5.99999999999999978e-01},
  {2.00000000000000000e+00, 8.00000000000000044e-01},
  {9.99999999997387756e-01, 2.00000000000000011e-01},
  {9.99999999997387978e-01, 4.00000000000000022e-01},
  {9.99999999997387978e-01, 6.00000000000000089e-01},
  {9.99999999997388089e-01, 8.00000000000000044e-01}
};

/* Connectivity of a 2x5 rectangular mesh of quads : */
const PetscInt sConnectivity2x5Mesh[10][4] = {
  {0,  4,  14, 6 },
  {6,  14, 15, 7 },
  {7,  15, 16, 8 },
  {8,  16, 17, 9 },
  {9,  17, 5,  2 },
  {4,  1,  10, 14},
  {14, 10, 11, 15},
  {15, 11, 12, 16},
  {16, 12, 13, 17},
  {17, 13, 3,  5 }
};

/* Partitions of a 2x5 rectangular mesh of quads : */
const PetscInt sInitialPartition2x5Mesh[2][5] = {
  {0, 2, 4, 6, 8},
  {1, 3, 5, 7, 9}
};

const PetscInt sNLoclCells2x5Mesh = 5;
const PetscInt sNGlobVerts2x5Mesh = 18;

/* Mixed mesh : quads and triangles  (4 first quads above divided into triangles*/
/* Connectivity of a 2x5 rectangular mesh of quads : */
const PetscInt sConnectivityMixedTQMesh[14][4] = {
  {0,  4,  6,  -1},
  {4,  14, 6,  -1},
  {6,  14, 7,  -1},
  {14, 15, 7,  -1},
  {7,  15, 8,  -1},
  {15, 16, 8,  -1},
  {8,  16, 9,  -1},
  {16, 17, 9,  -1},
  {9,  17, 5,  2 },
  {4,  1,  10, 14},
  {14, 10, 11, 15},
  {15, 11, 12, 16},
  {16, 12, 13, 17},
  {17, 13, 3,  5 }
};

/* Partitions for the rectangular mesh of quads and triangles: */
const PetscInt sInitialPartitionMixedTQMesh[2][7] = {
  {0, 1, 4, 5, 8, 10, 12},
  {2, 3, 6, 7, 9, 11, 13}
};

const PetscInt sNLoclCellsMixedTQMesh = 7;
const PetscInt sNGlobVertsMixedTQMesh = 18;

/* Prisms mesh */
PetscReal sCoordsPrismsMesh[125][3] = {
  {2.24250931694056355e-01, 0.00000000000000000e+00, 0.00000000000000000e+00},
  {2.20660660151932697e-01, 2.87419338850266937e-01, 0.00000000000000000e+00},
  {0.00000000000000000e+00, 0.00000000000000000e+00, 2.70243537720639027e-01},
  {2.32445727460992402e-01, 0.00000000000000000e+00, 2.60591845015572310e-01},
  {2.41619971105419079e-01, 2.69894910706158231e-01, 2.42844781736072490e-01},
  {0.00000000000000000e+00, 2.46523339883120779e-01, 2.69072907562752262e-01},
  {0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00},
  {1.00000000000000000e+00, 2.75433417601945563e-01, 0.00000000000000000e+00},
  {1.00000000000000000e+00, 0.00000000000000000e+00, 2.33748605950385602e-01},
  {7.32445727460992457e-01, 0.00000000000000000e+00, 2.42344379130445597e-01},
  {1.00000000000000000e+00, 2.78258478013028610e-01, 2.57379172987105553e-01},
  {1.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00},
  {7.49586880891153995e-01, 1.00000000000000000e+00, 0.00000000000000000e+00},
  {1.00000000000000000e+00, 1.00000000000000000e+00, 2.51949651266657582e-01},
  {7.41619971105419107e-01, 7.69894910706158120e-01, 2.33697838509081768e-01},
  {1.00000000000000000e+00, 7.78258478013028610e-01, 2.66479695645241543e-01},
  {7.55042653233710115e-01, 1.00000000000000000e+00, 2.58019637386860512e-01},
  {1.00000000000000000e+00, 1.00000000000000000e+00, 0.00000000000000000e+00},
  {0.00000000000000000e+00, 7.59235710423095789e-01, 0.00000000000000000e+00},
  {0.00000000000000000e+00, 1.00000000000000000e+00, 2.17232187874490473e-01},
  {0.00000000000000000e+00, 7.46523339883120807e-01, 2.42567232639677999e-01},
  {2.55042653233710115e-01, 1.00000000000000000e+00, 2.40660905690776916e-01},
  {0.00000000000000000e+00, 1.00000000000000000e+00, 0.00000000000000000e+00},
  {2.38934376044866809e-01, 0.00000000000000000e+00, 1.00000000000000000e+00},
  {2.18954188589218168e-01, 2.26916038449581692e-01, 1.00000000000000000e+00},
  {2.39787449636397643e-01, 0.00000000000000000e+00, 7.60591845015572310e-01},
  {2.40766735324061815e-01, 2.39643260505815608e-01, 7.42844781736072490e-01},
  {0.00000000000000000e+00, 2.57448248192627016e-01, 7.69072907562752262e-01},
  {0.00000000000000000e+00, 0.00000000000000000e+00, 1.00000000000000000e+00},
  {1.00000000000000000e+00, 2.38666970143638080e-01, 1.00000000000000000e+00},
  {7.39787449636397643e-01, 0.00000000000000000e+00, 7.42344379130445597e-01},
  {1.00000000000000000e+00, 2.59875254283874868e-01, 7.57379172987105553e-01},
  {1.00000000000000000e+00, 0.00000000000000000e+00, 1.00000000000000000e+00},
  {7.76318984007844159e-01, 1.00000000000000000e+00, 1.00000000000000000e+00},
  {7.40766735324061787e-01, 7.39643260505815636e-01, 7.33697838509081768e-01},
  {1.00000000000000000e+00, 7.59875254283874924e-01, 7.66479695645241543e-01},
  {7.68408704792055142e-01, 1.00000000000000000e+00, 7.58019637386860512e-01},
  {1.00000000000000000e+00, 1.00000000000000000e+00, 1.00000000000000000e+00},
  {0.00000000000000000e+00, 7.81085527042108207e-01, 1.00000000000000000e+00},
  {0.00000000000000000e+00, 7.57448248192627016e-01, 7.42567232639678054e-01},
  {2.68408704792055197e-01, 1.00000000000000000e+00, 7.40660905690776916e-01},
  {0.00000000000000000e+00, 1.00000000000000000e+00, 1.00000000000000000e+00},
  {7.24250931694056410e-01, 0.00000000000000000e+00, 0.00000000000000000e+00},
  {7.24250931694056410e-01, 2.75433417601945563e-01, 0.00000000000000000e+00},
  {4.44911591845989052e-01, 2.87419338850266937e-01, 0.00000000000000000e+00},
  {4.64891454921984804e-01, 0.00000000000000000e+00, 2.50940152310505593e-01},
  {4.74065698566411453e-01, 2.69894910706158231e-01, 2.33193089031005774e-01},
  {4.48501863388112709e-01, 0.00000000000000000e+00, 0.00000000000000000e+00},
  {7.20660660151932753e-01, 7.87419338850266937e-01, 0.00000000000000000e+00},
  {7.20660660151932753e-01, 5.62852756452212555e-01, 0.00000000000000000e+00},
  {2.20660660151932697e-01, 5.46655049273362614e-01, 0.00000000000000000e+00},
  {4.83239942210838158e-01, 5.39789821412316462e-01, 2.15446025751505982e-01},
  {7.41619971105419107e-01, 5.48153388719186951e-01, 2.48227882887665785e-01},
  {2.41619971105419079e-01, 5.16418250589278927e-01, 2.41674151578185781e-01},
  {4.41321320303865394e-01, 5.74838677700533873e-01, 0.00000000000000000e+00},
  {1.00000000000000000e+00, 7.75433417601945507e-01, 0.00000000000000000e+00},
  {1.00000000000000000e+00, 5.56516956026057219e-01, 2.81009740023825560e-01},
  {7.32445727460992457e-01, 2.78258478013028610e-01, 2.65974946167165549e-01},
  {1.00000000000000000e+00, 5.50866835203891125e-01, 0.00000000000000000e+00},
  {0.00000000000000000e+00, 2.59235710423095733e-01, 0.00000000000000000e+00},
  {0.00000000000000000e+00, 4.93046679766241558e-01, 2.67902277404865552e-01},
  {2.55042653233710115e-01, 7.46523339883120807e-01, 2.65995950455964469e-01},
  {0.00000000000000000e+00, 5.18471420846191466e-01, 0.00000000000000000e+00},
  {2.49586880891154023e-01, 1.00000000000000000e+00, 0.00000000000000000e+00},
  {2.49586880891154023e-01, 7.59235710423095789e-01, 0.00000000000000000e+00},
  {4.70247541043086748e-01, 7.87419338850266937e-01, 0.00000000000000000e+00},
  {5.10085306467420230e-01, 1.00000000000000000e+00, 2.64089623507063387e-01},
  {4.96662624339129222e-01, 7.69894910706158231e-01, 2.39767824629284698e-01},
  {4.99173761782308045e-01, 1.00000000000000000e+00, 0.00000000000000000e+00},
  {0.00000000000000000e+00, 0.00000000000000000e+00, 7.70243537720639027e-01},
  {2.40640523227928449e-01, 0.00000000000000000e+00, 5.21183690031144620e-01},
  {2.62579282058905461e-01, 2.52370482562049525e-01, 4.85689563472144981e-01},
  {0.00000000000000000e+00, 2.33810969343145825e-01, 5.38145815125504523e-01},
  {0.00000000000000000e+00, 0.00000000000000000e+00, 5.40487075441278053e-01},
  {1.00000000000000000e+00, 0.00000000000000000e+00, 7.33748605950385602e-01},
  {7.40640523227928504e-01, 0.00000000000000000e+00, 4.84688758260891195e-01},
  {1.00000000000000000e+00, 2.81083538424111656e-01, 5.14758345974211107e-01},
  {1.00000000000000000e+00, 0.00000000000000000e+00, 4.67497211900771203e-01},
  {7.38934376044866781e-01, 0.00000000000000000e+00, 1.00000000000000000e+00},
  {4.79574899272795285e-01, 0.00000000000000000e+00, 7.50940152310505593e-01},
  {4.77868752089733617e-01, 0.00000000000000000e+00, 1.00000000000000000e+00},
  {1.00000000000000000e+00, 1.00000000000000000e+00, 7.51949651266657582e-01},
  {7.62579282058905461e-01, 7.52370482562049525e-01, 4.67395677018163536e-01},
  {1.00000000000000000e+00, 7.81083538424111712e-01, 5.32959391290483087e-01},
  {7.60498425576266124e-01, 1.00000000000000000e+00, 5.16039274773721024e-01},
  {1.00000000000000000e+00, 1.00000000000000000e+00, 5.03899302533315163e-01},
  {7.18954188589218113e-01, 7.26916038449581636e-01, 1.00000000000000000e+00},
  {4.81533470648123629e-01, 4.79286521011631217e-01, 7.15446025751505954e-01},
  {4.57888564634085005e-01, 2.26916038449581692e-01, 1.00000000000000000e+00},
  {4.95273172597062383e-01, 7.26916038449581636e-01, 1.00000000000000000e+00},
  {4.37908377178436337e-01, 4.53832076899163384e-01, 1.00000000000000000e+00},
  {1.00000000000000000e+00, 7.38666970143638135e-01, 1.00000000000000000e+00},
  {1.00000000000000000e+00, 5.19750508567749736e-01, 7.81009740023825616e-01},
  {7.38934376044866781e-01, 2.38666970143638080e-01, 1.00000000000000000e+00},
  {7.18954188589218113e-01, 4.65583008593219771e-01, 1.00000000000000000e+00},
  {1.00000000000000000e+00, 4.77333940287276159e-01, 1.00000000000000000e+00},
  {0.00000000000000000e+00, 1.00000000000000000e+00, 7.17232187874490501e-01},
  {0.00000000000000000e+00, 7.33810969343145825e-01, 4.85134465279355998e-01},
  {2.60498425576266179e-01, 1.00000000000000000e+00, 4.81321811381553832e-01},
  {0.00000000000000000e+00, 1.00000000000000000e+00, 4.34464375748980947e-01},
  {0.00000000000000000e+00, 2.81085527042108152e-01, 1.00000000000000000e+00},
  {0.00000000000000000e+00, 5.14896496385254032e-01, 7.67902277404865607e-01},
  {2.76318984007844215e-01, 7.81085527042108207e-01, 1.00000000000000000e+00},
  {2.18954188589218168e-01, 5.08001565491689844e-01, 1.00000000000000000e+00},
  {0.00000000000000000e+00, 5.62171054084216304e-01, 1.00000000000000000e+00},
  {2.76318984007844215e-01, 1.00000000000000000e+00, 1.00000000000000000e+00},
  {5.36817409584110394e-01, 1.00000000000000000e+00, 7.64089623507063331e-01},
  {5.52637968015688430e-01, 1.00000000000000000e+00, 1.00000000000000000e+00},
  {5.03219805286833965e-01, 2.52370482562049525e-01, 4.66386178062011547e-01},
  {4.80554184960459430e-01, 2.39643260505815608e-01, 7.33193089031005774e-01},
  {4.81281046455856898e-01, 0.00000000000000000e+00, 5.01880304621011186e-01},
  {7.62579282058905461e-01, 5.33454020986161126e-01, 4.96455765775331570e-01},
  {2.62579282058905461e-01, 4.86181451905195350e-01, 4.83348303156371562e-01},
  {7.40766735324061787e-01, 4.99518514789690449e-01, 7.48227882887665841e-01},
  {2.40766735324061815e-01, 4.97091508698442541e-01, 7.41674151578185725e-01},
  {5.25158564117810922e-01, 5.04740965124099050e-01, 4.30892051503011964e-01},
  {7.40640523227928504e-01, 2.81083538424111656e-01, 5.31949892334331098e-01},
  {7.39787449636397643e-01, 2.59875254283874868e-01, 7.65974946167165549e-01},
  {1.00000000000000000e+00, 5.62167076848223313e-01, 5.62019480047651121e-01},
  {2.60498425576266179e-01, 7.33810969343145825e-01, 5.31991900911928939e-01},
  {2.68408704792055197e-01, 7.57448248192627016e-01, 7.65995950455964469e-01},
  {0.00000000000000000e+00, 4.67621938686291649e-01, 5.35804554809731104e-01},
  {5.23077707635171585e-01, 7.52370482562049525e-01, 4.79535649258569396e-01},
  {5.09175440116116929e-01, 7.39643260505815636e-01, 7.39767824629284698e-01},
  {5.20996851152532359e-01, 1.00000000000000000e+00, 5.28179247014126774e-01}
};

const PetscInt sConnectivityPrismsMesh[128][6] = {
  /* rank 0 */
  {11,  7,   42,  8,   10,  9  },
  {47,  42,  43,  45,  9,   57 },
  {8,   10,  9,   77,  76,  75 },
  {45,  9,   57,  110, 75,  116},
  {17,  48,  55,  13,  14,  15 },
  {58,  55,  49,  56,  15,  52 },
  {13,  14,  15,  85,  82,  83 },
  {56,  15,  52,  118, 83,  111},
  {6,   0,   1,   2,   3,   4  },
  {54,  1,   44,  51,  4,   46 },
  {2,   3,   4,   73,  70,  71 },
  {51,  4,   46,  115, 71,  108},
  {58,  49,  43,  56,  52,  57 },
  {47,  43,  44,  45,  57,  46 },
  {56,  52,  57,  118, 111, 116},
  {45,  57,  46,  110, 116, 108},
  {77,  76,  75,  74,  31,  30 },
  {110, 75,  116, 79,  30,  117},
  {74,  31,  30,  32,  29,  78 },
  {79,  30,  117, 80,  78,  93 },
  {85,  82,  83,  81,  34,  35 },
  {118, 83,  111, 92,  35,  113},
  {81,  34,  35,  37,  86,  91 },
  {92,  35,  113, 95,  91,  94 },
  {73,  70,  71,  69,  25,  26 },
  {115, 71,  108, 87,  26,  109},
  {69,  25,  26,  28,  23,  24 },
  {87,  26,  109, 90,  24,  88 },
  {118, 111, 116, 92,  113, 117},
  {110, 116, 108, 79,  117, 109},
  {92,  113, 117, 95,  94,  93 },
  {79,  117, 109, 80,  93,  88 },
  {22,  18,  63,  19,  20,  21 },
  {68,  63,  64,  66,  21,  61 },
  {19,  20,  21,  99,  97,  98 },
  {66,  21,  61,  124, 98,  119},
  {6,   1,   59,  2,   4,   5  },
  {62,  59,  50,  60,  5,   53 },
  {2,   4,   5,   73,  71,  72 },
  {60,  5,   53,  121, 72,  112},
  {17,  12,  48,  13,  16,  14 },
  {54,  48,  65,  51,  14,  67 },
  {13,  16,  14,  85,  84,  82 },
  {51,  14,  67,  115, 82,  122},
  {62,  50,  64,  60,  53,  61 },
  {68,  64,  65,  66,  61,  67 },
  {60,  53,  61,  121, 112, 119},
  {66,  61,  67,  124, 119, 122},
  {99,  97,  98,  96,  39,  40 },
  {124, 98,  119, 106, 40,  120},
  {96,  39,  40,  41,  38,  105},
  {106, 40,  120, 107, 105, 102},
  {73,  71,  72,  69,  26,  27 },
  {121, 72,  112, 101, 27,  114},
  {69,  26,  27,  28,  24,  100},
  {101, 27,  114, 104, 100, 103},
  {85,  84,  82,  81,  36,  34 },
  {115, 82,  122, 87,  34,  123},
  {81,  36,  34,  37,  33,  86 },
  {87,  34,  123, 90,  86,  89 },
  {121, 112, 119, 101, 114, 120},
  {124, 119, 122, 106, 120, 123},
  {101, 114, 120, 104, 103, 102},
  {106, 120, 123, 107, 102, 89 },
  /* rank 1 */
  {58,  43,  7,   56,  57,  10 },
  {7,   43,  42,  10,  57,  9  },
  {56,  57,  10,  118, 116, 76 },
  {10,  57,  9,   76,  116, 75 },
  {54,  49,  48,  51,  52,  14 },
  {48,  49,  55,  14,  52,  15 },
  {51,  52,  14,  115, 111, 82 },
  {14,  52,  15,  82,  111, 83 },
  {47,  44,  0,   45,  46,  3  },
  {0,   44,  1,   3,   46,  4  },
  {45,  46,  3,   110, 108, 70 },
  {3,   46,  4,   70,  108, 71 },
  {54,  44,  49,  51,  46,  52 },
  {49,  44,  43,  52,  46,  57 },
  {51,  46,  52,  115, 108, 111},
  {52,  46,  57,  111, 108, 116},
  {118, 116, 76,  92,  117, 31 },
  {76,  116, 75,  31,  117, 30 },
  {92,  117, 31,  95,  93,  29 },
  {31,  117, 30,  29,  93,  78 },
  {115, 111, 82,  87,  113, 34 },
  {82,  111, 83,  34,  113, 35 },
  {87,  113, 34,  90,  94,  86 },
  {34,  113, 35,  86,  94,  91 },
  {110, 108, 70,  79,  109, 25 },
  {70,  108, 71,  25,  109, 26 },
  {79,  109, 25,  80,  88,  23 },
  {25,  109, 26,  23,  88,  24 },
  {115, 108, 111, 87,  109, 113},
  {111, 108, 116, 113, 109, 117},
  {87,  109, 113, 90,  88,  94 },
  {113, 109, 117, 94,  88,  93 },
  {62,  64,  18,  60,  61,  20 },
  {18,  64,  63,  20,  61,  21 },
  {60,  61,  20,  121, 119, 97 },
  {20,  61,  21,  97,  119, 98 },
  {54,  50,  1,   51,  53,  4  },
  {1,   50,  59,  4,   53,  5  },
  {51,  53,  4,   115, 112, 71 },
  {4,   53,  5,   71,  112, 72 },
  {68,  65,  12,  66,  67,  16 },
  {12,  65,  48,  16,  67,  14 },
  {66,  67,  16,  124, 122, 84 },
  {16,  67,  14,  84,  122, 82 },
  {54,  65,  50,  51,  67,  53 },
  {50,  65,  64,  53,  67,  61 },
  {51,  67,  53,  115, 122, 112},
  {53,  67,  61,  112, 122, 119},
  {121, 119, 97,  101, 120, 39 },
  {97,  119, 98,  39,  120, 40 },
  {101, 120, 39,  104, 102, 38 },
  {39,  120, 40,  38,  102, 105},
  {115, 112, 71,  87,  114, 26 },
  {71,  112, 72,  26,  114, 27 },
  {87,  114, 26,  90,  103, 24 },
  {26,  114, 27,  24,  103, 100},
  {124, 122, 84,  106, 123, 36 },
  {84,  122, 82,  36,  123, 34 },
  {106, 123, 36,  107, 89,  33 },
  {36,  123, 34,  33,  89,  86 },
  {115, 122, 112, 87,  123, 114},
  {112, 122, 119, 114, 123, 120},
  {87,  123, 114, 90,  89,  103},
  {114, 123, 120, 103, 89,  102}
};

/* Partitions of prisms mesh : */
const PetscInt sInitialPartitionPrismsMesh[2][64] = {
  {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63 },
  {64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
   96,                                                                                                                                 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127}
};

const PetscInt sNLoclCellsPrismsMesh = 64;
const PetscInt sNGlobVertsPrismsMesh = 125;

/* Coordinates for hexa+prism mesh */
PetscReal sCoordsHexPrismMesh[10][3] = {
  {2.00000000000000000e+00, 2.00000000000000000e+00, 0.00000000000000000e+00},
  {2.00000000000000000e+00, 0.00000000000000000e+00, 2.00000000000000000e+00},
  {2.00000000000000000e+00, 2.00000000000000000e+00, 2.00000000000000000e+00},
  {4.00000000000000000e+00, 2.00000000000000000e+00, 0.00000000000000000e+00},
  {4.00000000000000000e+00, 2.00000000000000000e+00, 2.00000000000000000e+00},
  {4.00000000000000000e+00, 0.00000000000000000e+00, 2.00000000000000000e+00},
  {4.00000000000000000e+00, 0.00000000000000000e+00, 4.00000000000000000e+00},
  {4.00000000000000000e+00, 2.00000000000000000e+00, 4.00000000000000000e+00},
  {2.00000000000000000e+00, 2.00000000000000000e+00, 4.00000000000000000e+00},
  {2.00000000000000000e+00, 0.00000000000000000e+00, 4.00000000000000000e+00}
};

const PetscInt sConnectivityHexPrismMesh[2][8] = {
  {1, 2, 8, 9, 5, 4, 7,  6 },
  {0, 2, 1, 3, 4, 5, -1, -1}
};
/* Partitions of prisms mesh : */
const PetscInt sInitialPartitionHexPrismMesh[2][2] = {
  {-1, -1},
  {0,  1 }
};

const PetscInt sNLoclCellsHexPrismMesh[2] = {0, 2};
const PetscInt sNGlobVertsHexPrismMesh    = 10;

int main(int argc, char **argv)
{
  PetscInt         Nc = 0;
  const PetscInt  *InitPartForRank[2];
  DM               dm, idm, ddm;
  PetscSF          sfVert, sfMig, sfPart;
  PetscPartitioner part;
  PetscSection     s;
  PetscInt        *cells, c;
  PetscMPIInt      size, rank;
  PetscBool        box = PETSC_FALSE, field = PETSC_FALSE, quadsmesh = PETSC_FALSE, trisquadsmesh = PETSC_FALSE, prismsmesh = PETSC_FALSE, hexprismmesh = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCheck(size == 2, PETSC_COMM_WORLD, PETSC_ERR_SUP, "This is a 2 processors example only");
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-field", &field, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-box", &box, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-quadsmesh", &quadsmesh, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-trisquadsmesh", &trisquadsmesh, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-prismsmesh", &prismsmesh, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-hexprismmesh", &hexprismmesh, NULL));
  PetscCheck(1 == (box ? 1 : 0) + (quadsmesh ? 1 : 0) + (trisquadsmesh ? 1 : 0) + (prismsmesh ? 1 : 0) + (hexprismmesh ? 1 : 0), PETSC_COMM_WORLD, PETSC_ERR_SUP, "Specify one and only one of -box, -quadsmesh or -prismsmesh");

  PetscCall(DMPlexCreate(PETSC_COMM_WORLD, &dm));
  if (box) {
    PetscCall(DMSetType(dm, DMPLEX));
    PetscCall(DMSetFromOptions(dm));
  } else {
    if (quadsmesh) {
      Nc                        = sNLoclCells2x5Mesh; //Same on each rank for this example...
      PetscInt Nv               = sNGlobVerts2x5Mesh;
      InitPartForRank[0]        = &sInitialPartition2x5Mesh[0][0];
      InitPartForRank[1]        = &sInitialPartition2x5Mesh[1][0];
      const PetscInt (*Conn)[4] = sConnectivity2x5Mesh;

      const PetscInt Ncor = 4;
      const PetscInt dim  = 2;

      PetscCall(PetscMalloc1(Nc * Ncor, &cells));
      for (c = 0; c < Nc; ++c) {
        PetscInt cell = (InitPartForRank[rank])[c], cor;

        for (cor = 0; cor < Ncor; ++cor) cells[c * Ncor + cor] = Conn[cell][cor];
      }
      PetscCall(DMSetDimension(dm, dim));
      PetscCall(DMPlexBuildFromCellListParallel(dm, Nc, PETSC_DECIDE, Nv, Ncor, cells, &sfVert, NULL));
    } else if (trisquadsmesh) {
      Nc                        = sNLoclCellsMixedTQMesh; //Same on each rank for this example...
      PetscInt Nv               = sNGlobVertsMixedTQMesh;
      InitPartForRank[0]        = &sInitialPartitionMixedTQMesh[0][0];
      InitPartForRank[1]        = &sInitialPartitionMixedTQMesh[1][0];
      const PetscInt (*Conn)[4] = sConnectivityMixedTQMesh;

      const PetscInt NcorMax = 4;
      const PetscInt dim     = 2;

      /* Create a PetscSection and taking care to exclude nodes with "-1" into element connectivity: */
      PetscSection s;
      PetscInt     vStart = 0, vEnd = Nc;
      PetscCall(PetscSectionCreate(PETSC_COMM_WORLD, &s));
      PetscCall(PetscSectionSetNumFields(s, 1));
      PetscCall(PetscSectionSetFieldComponents(s, 0, 1));
      PetscCall(PetscSectionSetChart(s, vStart, vEnd));

      PetscCall(PetscMalloc1(Nc * NcorMax, &cells));
      PetscInt count = 0;
      for (c = 0; c < Nc; ++c) {
        PetscInt cell         = (InitPartForRank[rank])[c], cor;
        PetscInt nbElemVertex = ((-1 == Conn[cell][NcorMax - 1]) ? 3 : 4);
        for (cor = 0; cor < nbElemVertex; ++cor) {
          cells[count] = Conn[cell][cor];
          ++count;
        }
        PetscCall(PetscSectionSetDof(s, c, nbElemVertex));
        PetscCall(PetscSectionSetFieldDof(s, c, 0, nbElemVertex));
      }
      PetscCall(PetscSectionSetUp(s));
      PetscCall(DMSetDimension(dm, dim));
      PetscCall(PetscSectionView(s, PETSC_VIEWER_STDOUT_WORLD));
      PetscCall(DMPlexBuildFromCellSectionParallel(dm, Nc, PETSC_DECIDE, Nv, s, cells, &sfVert, NULL));
      PetscCall(PetscSectionDestroy(&s));
    } else if (prismsmesh) {
      Nc                        = sNLoclCellsPrismsMesh; //Same on each rank for this example...
      PetscInt Nv               = sNGlobVertsPrismsMesh;
      InitPartForRank[0]        = &sInitialPartitionPrismsMesh[0][0];
      InitPartForRank[1]        = &sInitialPartitionPrismsMesh[1][0];
      const PetscInt (*Conn)[6] = sConnectivityPrismsMesh;

      const PetscInt Ncor = 6;
      const PetscInt dim  = 3;

      PetscCall(PetscMalloc1(Nc * Ncor, &cells));
      for (c = 0; c < Nc; ++c) {
        PetscInt cell = (InitPartForRank[rank])[c], cor;

        for (cor = 0; cor < Ncor; ++cor) cells[c * Ncor + cor] = Conn[cell][cor];
      }
      PetscCall(DMSetDimension(dm, dim));
      PetscCall(DMPlexBuildFromCellListParallel(dm, Nc, PETSC_DECIDE, Nv, Ncor, cells, &sfVert, NULL));
    } else if (hexprismmesh) {
      Nc                        = sNLoclCellsHexPrismMesh[rank]; //Same on each rank for this example...
      PetscInt Nv               = sNGlobVertsHexPrismMesh;
      InitPartForRank[0]        = &sInitialPartitionHexPrismMesh[0][0];
      InitPartForRank[1]        = &sInitialPartitionHexPrismMesh[1][0];
      const PetscInt (*Conn)[8] = sConnectivityHexPrismMesh;

      const PetscInt NcorMax = 8;
      const PetscInt dim     = 3;

      /* Create a PetscSection and taking care to exclude nodes with "-1" into element connectivity: */
      PetscSection s;
      PetscInt     vStart = 0, vEnd = Nc;
      PetscCall(PetscSectionCreate(PETSC_COMM_WORLD, &s));
      PetscCall(PetscSectionSetNumFields(s, 1));
      PetscCall(PetscSectionSetFieldComponents(s, 0, 1));
      PetscCall(PetscSectionSetChart(s, vStart, vEnd));

      PetscCall(PetscMalloc1(Nc * NcorMax, &cells));
      PetscInt count = 0;
      for (c = 0; c < Nc; ++c) {
        PetscInt cell         = (InitPartForRank[rank])[c], cor;
        PetscInt nbElemVertex = ((-1 == Conn[cell][NcorMax - 1]) ? 6 : 8);
        for (cor = 0; cor < nbElemVertex; ++cor) {
          cells[count] = Conn[cell][cor];
          ++count;
        }
        PetscCall(PetscSectionSetDof(s, c, nbElemVertex));
        PetscCall(PetscSectionSetFieldDof(s, c, 0, nbElemVertex));
      }
      PetscCall(PetscSectionSetUp(s));
      PetscCall(DMSetDimension(dm, dim));
      PetscCall(PetscSectionView(s, PETSC_VIEWER_STDOUT_WORLD));
      PetscCall(DMPlexBuildFromCellSectionParallel(dm, Nc, PETSC_DECIDE, Nv, s, cells, &sfVert, NULL));
      PetscCall(PetscSectionDestroy(&s));
    }
    PetscCall(PetscSFDestroy(&sfVert));
    PetscCall(PetscFree(cells));
    PetscCall(DMPlexSetInterpolatePreferTensor(dm, PETSC_FALSE));
    PetscCall(DMPlexInterpolate(dm, &idm));
    PetscCall(DMDestroy(&dm));
    dm = idm;
  }
  PetscCall(DMSetUseNatural(dm, PETSC_TRUE));
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));

  if (field) {
    const PetscInt Nf         = 1;
    const PetscInt numBC      = 0;
    const PetscInt numComp[1] = {1};
    PetscInt       numDof[4]  = {0, 0, 0, 0};
    PetscInt       dim;

    PetscCall(DMGetDimension(dm, &dim));
    numDof[dim] = 1;

    PetscCall(DMSetNumFields(dm, Nf));
    PetscCall(DMPlexCreateSection(dm, NULL, numComp, numDof, numBC, NULL, NULL, NULL, NULL, &s));
    PetscCall(DMSetLocalSection(dm, s));
    /*PetscCall(PetscSectionView(s, PETSC_VIEWER_STDOUT_WORLD));*/
    PetscCall(PetscSectionDestroy(&s));
  }

  PetscCall(DMPlexGetPartitioner(dm, &part));
  PetscCall(PetscPartitionerSetFromOptions(part));

  PetscCall(DMPlexDistribute(dm, 0, &sfMig, &ddm));
  PetscCall(PetscSFView(sfMig, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscSFCreateInverseSF(sfMig, &sfPart));
  PetscCall(PetscObjectSetName((PetscObject)sfPart, "Inverse Migration SF"));
  PetscCall(PetscSFView(sfPart, PETSC_VIEWER_STDOUT_WORLD));

  Vec          lGlobalVec, lNatVec;
  PetscScalar *lNatVecArray;

  {
    PetscSection s;

    PetscCall(DMGetGlobalSection(dm, &s));
    PetscCall(PetscSectionView(s, PETSC_VIEWER_STDOUT_WORLD));
  }
  PetscCall(DMGetGlobalVector(dm, &lNatVec));
  PetscCall(PetscObjectSetName((PetscObject)lNatVec, "Natural Vector (initial partition)"));

  //Copying the initial partition into the "natural" vector:
  PetscCall(VecZeroEntries(lNatVec));
  PetscCall(VecGetArray(lNatVec, &lNatVecArray));
  for (c = 0; c < Nc; ++c) lNatVecArray[c] = (InitPartForRank[rank])[c];
  PetscCall(VecRestoreArray(lNatVec, &lNatVecArray));

  PetscCall(DMGetGlobalVector(ddm, &lGlobalVec));
  PetscCall(PetscObjectSetName((PetscObject)lGlobalVec, "Global Vector (reordered element numbers in the PETSc distributed order)"));
  PetscCall(VecZeroEntries(lGlobalVec));

  // The call to DMPlexNaturalToGlobalBegin/End does not produce our expected result...
  // In lGlobalVec, we expect to have:
  /*
   * Process [0]
   * 2.
   * 4.
   * 8.
   * 3.
   * 9.
   * Process [1]
   * 1.
   * 5.
   * 7.
   * 0.
   * 6.
   *
   * but we obtained:
   *
   * Process [0]
   * 2.
   * 4.
   * 8.
   * 0.
   * 0.
   * Process [1]
   * 0.
   * 0.
   * 0.
   * 0.
   * 0.
   */

  {
    PetscSF nsf;

    PetscCall(DMGetNaturalSF(ddm, &nsf));
    PetscCall(PetscSFView(nsf, NULL));
  }
  PetscCall(DMPlexNaturalToGlobalBegin(ddm, lNatVec, lGlobalVec));
  PetscCall(DMPlexNaturalToGlobalEnd(ddm, lNatVec, lGlobalVec));

  PetscCall(VecView(lNatVec, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(VecView(lGlobalVec, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(DMRestoreGlobalVector(dm, &lNatVec));
  PetscCall(DMRestoreGlobalVector(ddm, &lGlobalVec));

  const PetscBool lUseCone    = PETSC_FALSE;
  const PetscBool lUseClosure = PETSC_TRUE;
  PetscCall(DMSetBasicAdjacency(ddm, lUseCone, lUseClosure));
  const PetscInt lNbCellsInOverlap = 1;
  PetscSF        lSFMigrationOvl;
  DM             ddm_with_overlap;

  PetscCall(DMPlexDistributeOverlap(ddm, lNbCellsInOverlap, &lSFMigrationOvl, &ddm_with_overlap));

  IS lISCellWithOvl = 0;
  /* This is the buggy call with prisms since commit 5ae96e2b862 */
  PetscCall(DMPlexCreateCellNumbering(ddm_with_overlap, PETSC_TRUE, &lISCellWithOvl));
  /* Here, we can see the elements in the overlap within the IS: they are the ones with negative indices */
  PetscCall(ISView(lISCellWithOvl, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(ISDestroy(&lISCellWithOvl));

  PetscCall(PetscSFDestroy(&lSFMigrationOvl));
  PetscCall(DMDestroy(&ddm_with_overlap));
  PetscCall(PetscSFDestroy(&sfMig));
  PetscCall(PetscSFDestroy(&sfPart));
  PetscCall(DMDestroy(&dm));
  PetscCall(DMDestroy(&ddm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    args: -field -petscpartitioner_type simple
    nsize: 2

    test:
      suffix: 0
      args: -quadsmesh
      output_file: output/ex47_0.out

    test:
      suffix: 1
      args: -box -dm_plex_simplex 0 -dm_plex_box_faces 2,5 -dm_distribute
      output_file: output/ex47_1.out

    test:
      suffix: 2
      args: -prismsmesh
      output_file: output/ex47_2.out

    test:
      suffix: 3
      args: -trisquadsmesh
      output_file: output/ex47_3.out

    test:
      suffix: 4
      args: -hexprismmesh
      output_file: output/ex47_4.out

TEST*/
