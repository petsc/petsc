static char help[] = "Tests for creation of cohesive meshes by transforms\n\n";

#include <petscdmplex.h>
#include <petscsf.h>

#include <petsc/private/dmpleximpl.h>

PETSC_EXTERN char tri_2_cv[];
char              tri_2_cv[] = "\
2 4 6 3 1\n\
0 2 1\n\
1 2 3\n\
4 1 5\n\
4 0 1\n\
-1.0  0.0 0.0  1\n\
 0.0  1.0 0.0 -1\n\
 0.0 -1.0 0.0  1\n\
 1.0  0.0 0.0 -1\n\
-2.0  1.0 0.0  1\n\
-1.0  2.0 0.0 -1";

/* List of test meshes

Test tri_0: triangle

 4-10--5      8-16--7-14--4
 |\  1 |      |\     \  1 |
 | \   |      | \     \   |
 6  8  9  ->  9 12  2  11 13
 |   \ |      |   \     \ |
 | 0  \|      | 0  \     \|
 2--7--3      3-10--6-15--5

Test tri_1: triangle, not tensor

 4-10--5      8-10--7-16--4
 |\  1 |      |\     \  1 |
 | \   |      | \     \   |
 6  8  9  -> 11 14  2  13 15
 |   \ |      |   \     \ |
 | 0  \|      | 0  \     \|
 2--7--3      3-12--6--9--5

Test tri_2: 4 triangles, non-oriented surface

           9
          / \
         /   \
       17  2  16
       /       \
      /         \
     8-----15----5
      \         /|\
       \       / | \
       18  3  12 |  14
         \   /   |   \
          \ /    |    \
           4  0 11  1  7
            \    |    /
             \   |   /
             10  |  13
               \ | /
                \|/
                 6
  becomes
           8
          / \
         /   \
        /     \
      25   2   24
      /         \
     /           \
   13-----18------9
28  |     5    26/ \
   14----19----10   \
     \         /|   |\
      \       / |   | \
      21  3  20 |   |  23
        \   /   |   |   \
         \ /    |   |    \
          6  0 17 4 16 1  7
           \    |   |    /
            \   |   |   /
            15  |   |  22
              \ |   | /
               \|   |/
               12---11
                 27

Test tri_3: tri_2, in parallel

           6
          / \
         /   \
        /     \
      12   1   11
      /         \
     /           \
    5-----10------2
                   \
    5-----9-----3   2
     \         /|   |\
      \       / |   | \
      10  1  8  |   |  9
        \   /   |   |   \
         \ /    |   |    \
          2  0  7   7  0  4
           \    |   |    /
            \   |   |   /
             6  |   |  8
              \ |   | /
               \|   |/
                4   3
  becomes
                 11
                / \
               /   \
              /     \
            19   1   18
            /         \
           /           \
          8-----14------4
        22 \     3       |
            9------15    |\
                    \    | \
    9------14-----5  \  20 |
  20\    3     18/ \  \/   |
   10----15-----6   |  5   |
     \         /|   |  |   |\
      \       / |   |  |   | \
      17  1 16  |   |  |   |  17
        \   /   | 2 |  | 2 |   \
         \ /    |   |  |   |    \
          4  0  13 12  13  12 0 10
           \    |   |  |   |    /
            \   |   |  |   |   /
            11  |   |  |   |  16
              \ |   |  |   | /
               \|   |  |   |/
                8---7  7---6
                 19      21

Test quad_0: quadrilateral

 5-10--6-11--7       5-12-10-20--9-14--6
 |     |     |       |     |     |     |
12  0 13  1  14 --> 15  0 18  2 17  1  16
 |     |     |       |     |     |     |
 2--8--3--9--4       3-11--8-19--7-13--4

Test quad_1: quadrilateral, not tensor

 5-10--6-11--7       5-14-10-12--9-16--6
 |     |     |       |     |     |     |
12  0 13  1  14 --> 17  0 20  2 19  1  18
 |     |     |       |     |     |     |
 2--8--3--9--4       3-13--8-11--7-15--4

Test quad_2: quadrilateral, 2 processes

 3--6--4  3--6--4       3--9--7-14--6   5-14--4--9--7
 |     |  |     |       |     |     |   |     |     |
 7  0  8  7  0  8  --> 10  0 12  1 11  12  1 11  0  10
 |     |  |     |       |     |     |   |     |     |
 1--5--2  1--5--2       2--8--5-13--4   3-13--2--8--6

Test quad_3: quadrilateral, 4 processes, non-oriented surface

 3--6--4  3--6--4      3--9--7-14--6   5-14--4--9--7
 |     |  |     |      |     |     |   |     |     |
 7  0  8  7  0  8     10  0  12 1  11 12  1 11  0  10
 |     |  |     |      |     |     |   |     |     |
 1--5--2  1--5--2      2--8--5-13--4   3-13--2--8--6
                   -->
 3--6--4  3--6--4      3--9--7-14--6   5-14--4--9--7
 |     |  |     |      |     |     |   |     |     |
 7  0  8  7  0  8     10  0  12 1  11 12  1 11  0  10
 |     |  |     |      |     |     |   |     |     |
 1--5--2  1--5--2      2--8--5-13--4   3-13--2--8--6

Test quad_4: embedded fault

14-24-15-25-16-26--17
 |     |     |     |
28  3 30  4 32  5  34
 |     |     |     |
10-21-11-22-12-23--13
 |     |     |     |
27  0 29  1 31  2  33
 |     |     |     |
 6-18--7-19--8-20--9

becomes

 13-26-14-27-15-28--16
  |     |     |     |
 30  3 32  4 39  5  40
  |     |     |     |
 12-25-17-36-19-38--21
        |     |     |
       41  6 42  7  43
        |     |     |
 12-25-17-35-18-37--20
  |     |     |     |
 29  0 31  1 33  2  34
  |     |     |     |
  8-22--9-23-10-24--11

Test quad_5: two faults

14-24-15-25-16-26--17
 |     |     |     |
28  3 30  4 32  5  34
 |     |     |     |
10-21-11-22-12-23--13
 |     |     |     |
27  0 29  1 31  2  33
 |     |     |     |
 6-18--7-19--8-20--9

becomes

12-26-13-27-14-28--15
 |     |     |     |
37  4 31  3 33  5  40
 |     |     |     |
17-36-18-25-19-39--21
 |     |     |     |
43  6  44   41  7  42
 |     |     |     |
16-35-18-25-19-38--20
 |     |     |     |
29  0 30  1 32  2  34
 |     |     |     |
 8-22--9-23-10-24--11

Test quad_6: T-junction

14-24-15-25-16-26--17
 |     |     |     |
28  3 30  4 32  5  34
 |     |     |     |
10-21-11-22-12-23--13
 |     |     |     |
27  0 29  1 31  2  33
 |     |     |     |
 6-18--7-19--8-20--9

becomes

 13-26-14-27-15-28--16
  |     |     |     |
 30  3 32  4 39  5  40
  |     |     |     |
 12-25-17-36-19-38--21
        |     |     |
       41  6 42  7  43
        |     |     |
 12-25-17-35-18-37--20
  |     |     |     |
 29  0 31  1 33  2  34
  |     |     |     |
  8-22--9-23-10-24--11

becomes

 14-28-15-41-21-44--20-29-16
  |     |     |     |     |
 31  3 33  5 43  8 42  4  40
  |     |     |     |     |
 13-27-17-37-23-46--23-39-19
        |     |     |     |
       47  6 48    48  7  49
        |     |     |     |
 13-27-17-36-22-45--22-38-18
  |     |     |     |     |
 30  0 32  1 34    34  2  35
  |     |     |     |     |
  9-24-10-25-11-----11-26-12

Test tet_0: Two tets sharing a face

 cell   5 _______    cell
 0    / | \      \      1
    19  |  16     20
    /  15   \      \
   2-17------4--22--6
    \   |   /      /
    18  |  14     21
      \ | /      /
        3-------

becomes

 cell  10 ___36____9______    cell
 0    / | \        |\      \     1
    29  |  27      | 26     31
    /  25   \     24  \      \
   3-28------8--35-----7--33--4
    \   |   /      |  /      /
    30  |  23      | 22     32
      \ | /        |/      /
        6----34----5------
         cell 2

Test tet_1: Two tets sharing a face in parallel

 cell   4          3______    cell
 0    / | \        |\      \     0
    14  |  11      | 11     12
    /  10   \     10  \      \
   1-12------3     |   2--14--4
    \   |   /      |  /      /
    13  |  9       | 9      13
      \ | /        |/      /
        2          1------

becomes
           cell 1              cell 1
 cell   8---28---7           7---28---6______    cell
 0    / | \      |\          |\       |\      \     0
    24  |  22    | 21        | 22     | 21     23
    /  20   \    |   \       |  \    19  \      \
   2-23------6---27---5     20  5---27---4--25--8
    \   |   /   19   /       |  /     |  /      /
    25  |  18    | 17        | 18     | 17     24
      \ | /      |/          |/       |/      /
        4---26---3           3---26---2------

Test hex_0: Two hexes sharing a face

cell  11-----31-----12-----32------13 cell
0     /|            /|            /|     1
    36 |   22      37|   24      38|
    /  |          /  |          /  |
   8-----29------9-----30------10  |
   |   |     18  |   |     20  |   |
   |  42         |  43         |   44
   |14 |         |15 |         |16 |
  39   |  17    40   |   19   41   |
   |   5-----27--|---6-----28--|---7
   |  /          |  /          |  /
   | 33   21     | 34    23    | 35
   |/            |/            |/
   2-----25------3-----26------4

becomes

                         cell 2
cell   9-----38-----18-----62------17----42------10 cell
0     /|            /|            /|            /|     1
    45 |   30      54|  32       53|   24      46|
    /  |          /  |          /  |          /  |
   7-----37-----16-----61------15--|-41------8   |
   |   |     28  |   |         |   |     22  |   |
   |  49         |  58         |   57        |   50
   |19 |         |26 |         |25 |         |20 |
  47   |  27    56   |        55   |   21   48   |
   |   5-----36--|--14-----60--|---13----40--|---6
   |  /          |  /          |  /          |  /
   | 43   29     | 52   31     | 51    23    | 44
   |/            |/            |/            |/
   3-----35-----12-----59------11----39------4

Test hex_1: Two hexes sharing a face, in parallel

cell   7-----18------8             7-----18------8 cell
0     /|            /|            /|            /|    0
    21 |   14      22|           21|   14      22|
    /  |          /  |          /  |          /  |
   5-----17------6   |         5---|-17------6   |
   |   |     12  |   |         |   |     12  |   |
   |  25         |  26         |  25         |  26
   | 9 |         |10 |         | 9 |         |10 |
  23   |  11    24   |        23   |   11   24   |
   |   3-----16--|---4         |   3-----16--|---4
   |  /          |  /          |  /          |  /
   | 19   13     | 20          | 19    13    | 20
   |/            |/            |/            |/
   1-----15------2             1-----15------2

becomes
                        cell 1                      cell 1
cell   5-----28-----13-----44-----12             9-----44-----8-----28------13 cell
0     /|            /|           /|             /|           /|            /|     0
    30 |   20      36|   22     35|            36|   22     35|   20      30|
    /  |          /  |         /  |           /  |         /  |          /  |
   4-----27-----11-----43-----10  |          7-----43-----6-----27------12  |
   |   |     18  |   |        |   |          |   |        |   |     18  |   |
   |  32         |  40        |   39         |  40        |   39        |   32
   |14 |         |16 |        | 15|          |15 |        |14 |         |16 |
  31   |  17    38   |        37  |         38   |       37   |   17   31   |
   |   3-----26--|---9-----42-|---8          |   5----42--|---4-----26--|---11
   |  /          |  /         |  /           |  /         |  /          |  /
   | 29   19     | 34    21   | 33           | 34    21   | 33    19    | 29
   |/            |/           |/             |/           |/            |/
   2-----25------7-----41-----6              3-----41-----2-----25------10

Test hex_2: hexahedra, 4 processes, non-oriented surface

          cell 0                  cell 0
       7-----18------8       7-----18------8
      /|            /|      /|            /|
    21 |   14      22|    21 |   14      22|
    /  |          /  |    /  |          /  |
   5-----17------6   |   5-----17------6   |
   |   |     12  |   |   |   |     12  |   |
   |  25         |  26   |  25         |   26
   |9  |         |10 |   |9  |         |10 |
  23   |  11    24   |  23   |  11    24   |
   |   3-----16--|---4   |   3-----16--|---4
   |  /          |  /    |  /          |  /
   | 19    13    | 20    | 19    13    | 20
   |/            |/      |/            |/
   1-----15------2       1-----15------2

       7-----18------8       7-----18------8
      /|            /|      /|            /|
    21 |   14      22|    21 |   14      22|
    /  |          /  |    /  |          /  |
   5-----17------6   |   5-----17------6   |
   |   |     12  |   |   |   |     12  |   |
   |  25         |  26   |  25         |  26
   |9  |         |10 |   |9  |         |10 |
  23   |  11    24   |  23   |   11   24   |
   |   3-----16--|---4   |   3-----16--|---4
   |  /          |  /    |  /          |  /
   | 19   13     | 20    | 19    13    | 20
   |/            |/      |/            |/
   1-----15------2       1-----15------2
      cell 0                cell 0

becomes

          cell 0         cell 1                cell 1        cell 0
       5-----28------13----44------12      9-----44------8-----28------13
      /|            /|            /|      /|            /|            /|
    30 |   20      36|   22      35|     36|   22     35 |   20      30|
    /  |          /  |          /  |    /  |          /  |          /  |
   4-----27------11----43------10  |   7-----43------6-----27------12  |
   |   |     18  |   |         |   |   |   |         |   |     18  |   |
   |  32         |  40         |  39   |  40         |  39         |   32
   |14 |         |16 |         |15 |   |15 |         |14 |         |16 |
  31   |  17    38   |         37  |   38  |        37   |  17    31   |
   |   3-----26--|---9-----42--|---8   |   5-----42--|---4-----26--|---11
   |  /          |  /          |  /    |  /          |  /          |  /
   | 29    19    | 34    21    |33     | 34    21    | 33    19    | 29
   |/            |/            |/      |/            |/            |/
   2-----25------7-----41------6       3-----41------2-----25------10

       5-----28------13----44------12      9-----44------8-----28------13
      /|            /|            /|      /|            /|            /|
    30 |   20      36|   22      35|     36|    22     35|   20      30|
    /  |          /  |          /  |    /  |          /  |          /  |
   4-----27------11----43------10  |   7-----43------6-----27------12  |
   |   |     18  |   |         |   |   |   |         |   |     18  |   |
   |  32         |  40         |   39  |   40        |  39         |   32
   |14 |         |16 |         |15 |   |15 |         |14 |         |16 |
  31   |  17    38   |         37  |   38  |        37   |  17    31   |
   |   3-----26--|---9-----42--|---8   |   5-----42--|---4-----26--|---11
   |  /          |  /          |  /    |  /          |  /          |  /
   | 29    19    | 34    21    |33     | 34    21    | 33    19    | 29
   |/            |/            |/      |/            |/            |/
   2-----25------7-----41------6       3-----41------2-----25------10
      cell 0         cell 1                cell 1        cell 0

Test hex_3: T-junction

      19-----52-----20-----53------21
      /|            /|            /|
    60 |   38      61|   41      62|
    /  |          /  |          /  |
  16-----50-----17-----51------18  |
   |   |     33  |   |     35  |   |
   |  70         |  72         |   74
   |25 |         |26 |         |27 |
  64   |  32    66   |  34    68   |
   |  13-----48--|--14-----49--|---15
   |  /|         |  /|         |  /|
   |57 |   37    | 58|   40    | 59|
   |/  |         |/  |         |/  |
  10-----46-----11-----47------12  |
   |   |     29  |   |     31  |   |
   |  69         |  71         |   73
   |22 |         |23 |         |24 |
  63   |  28    65   |   30   67   |
   |   7-----44--|---8-----45--|---9
   |  /          |  /          |  /
   | 54   36     | 55    39    | 56
   |/            |/            |/
   4-----42------5-----43------6
      cell 0         cell 1

becomes

      15----102-----28---112----___27-----73------16
      /|            /|         /   /             /|
    77 |   55     104|      ---  103    46      78|
    /  |          /  |     /     /             /  |
  13----101-----26---111--/----25-----72------14  |
   |   |     54  |   |  107   /           43  |   |
   |  81         |  108 / 51 /                |   82
   |40 |         |52 | /   105                |41 |
  79   |  53    106  |/   /            42    80   |
   |  21-----87--|--31---/-89------23-------/----/
   |  /|         |  /|  /         /|       /
   |91 |   47    |109|-- 49      93|  -----
   |/  |         |/ /|          /  | /
  17-----83-----29-----85------19----
   |   |         |   |         |   |
   |  120        |  121        |  122
   |   |         |26 |         |   |
 117   |        118  |        119  |
   |  22-----88--|--32-----90--|---24
   |  /|         |  /|         |  /|
   |92 |   48    |110|   50    | 94|
   |/  |         |/  |         |/  |
  18-----84-----30-----86------20  |
   |   |     37  |   |     39  |   |
   |  98         |  99         |   100
   |33 |         |34 |         |35 |
  95   |  36    96   |   38   97   |
   |  10-----70--|--11-----71--|---12
   |  /          |  /          |  /
   | 74   44     | 75    45    | 76
   |/            |/            |/
   7-----68------8-----69------9
      cell 0         cell 1

Test hex_4: Two non-intersecting faults

          cell 4         cell 5         cell 6        cell 7
      33-----96-----34-----97-----35-----98-----36-----99------37
      /|            /|            /|            /|            /|
    110|   66     111|   69     112|   72     113|   75     114|
    /  |          /  |          /  |          /  |          /  |
  28-----92-----29-----93-----30-----94-----31-----95------32  |
   |   |     57  |   |     59  |   |     61  |   |     63  |   |
   |  126        |  128        |  130        |  132        |  134
   |43 |         |44 |         |45 |         |46 |         |47 |
  116  |  56    118  |  58    120  |  60    122  |  62    124  |
   |  23-----88--|--24-----89--|--25-----90--|--26-----91--|---27
   |  /|         |  /|         |  /|         |  /|         |  /|
   |105|   65    |106|   68    |107|   71    |108|   74    |109|
   |/  |         |/  |         |/  |         |/  |         |/  |
  18-----84-----19-----95-----20-----86-----21-----87------22  |
   |   |     49  |   |     51  |   |     53  |   |     55  |   |
   |  125        |  127        |  129        |  131        |  133
   |38 |         |39 |         |40 |         |41 |         |42 |
  115  |  48    117  |  50    119  |  52    121  |  54    123  |
   |  13-----80--|--14-----81--|--15-----82--|--16-----83--|---17
   |  /          |  /          |  /          |  /          |  /
   |100    64    |101    67    |102    70    |103    73    |104
   |/            |/            |/            |/            |/
   8-----76------9-----77-----10-----78-----11-----79------12
      cell 0         cell 1        cell 2        cell 3

becomes

          cell 4         cell 5        cell 7        cell 10       cell 6
      27-----114----28-----115----29-----159----46-----170----45------116----30
      /|            /|            /|            /|            /|            /|
    123|   71     124|   73     125|   87     162|          161|    78    126|
    /  |          /  |          /  |          /  |          /  |          /  |
  23-----111----24-----112----25-----158----44-----169----43-----113-----26  |
   |   |     65  |   |    67   |   |    86   |   |         |   |     69  |   |
   |  134        |  135        |  137        |  166        |  165        |  140
   |56 |         |57 |         |58 |         |84 |         |83 |         |59 |
  127  |  64    128  |  66    130  |  85    164  |        163  |  68    133  |
   |  35-----143-|--37-----151-|--40-----109-|--42-----168-|--42-----110-|---22
   |  /|         |  /|         |  /|         |  /          |  /          |  /
   |145|   79    |147|   81    |153|   75    |160          |160    77    |122
   |/ 173        |/ 174        |/ 176        |/            |/            |/
  31-----141----33-----149----39-----107----41-----167----41-----108-----21
cell   |         |   |         |   | cell 9
8  |  36-----144-|--38-----152-|--40-----109----42-----110-----22
  171 /|        172 /|        175 /|            /|            /|
   |146|   80    |148|   82    |153|    75    160|   77     122|
   |/  |         |/  |         |/  |          /  |          /  |
  32-----142----34-----150----39-----107----41-----108-----21  |
   |   |     50  |   |    52   |   |    61   |   |     63  |   |
   |  156        |  157        |  136        |  138        |  139
   |47 |         |48 |         |53 |         |54 |         |55 |
  154  |  49    155  |  51    129  |  60    131  |  62    132  |
   |  16-----103-|--17-----104-|--18-----105-|--19-----106-|---20
   |  /          |  /          |  /          |  /          |  /
   |117    70    |118    72    |119    74    |120    76    |121
   |/            |/            |/            |/            |/
  11-----99-----12-----100----13-----101----14-----102-----15
      cell 0         cell 1        cell 2        cell 3

*/

typedef struct {
  PetscInt testNum;        // The mesh to test
  PetscInt cohesiveFields; // The number of fault fields
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscFunctionBegin;
  options->testNum        = 0;
  options->cohesiveFields = 1;

  PetscOptionsBegin(comm, "", "Cohesive Meshing Options", "DMPLEX");
  PetscCall(PetscOptionsBoundedInt("-test_num", "The particular mesh to test", __FILE__, options->testNum, &options->testNum, NULL, 0));
  PetscCall(PetscOptionsBoundedInt("-cohesive_fields", "The number of cohesive fields", __FILE__, options->cohesiveFields, &options->cohesiveFields, NULL, 0));
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateQuadMesh1(MPI_Comm comm, AppCtx *user, DM *dm)
{
  const PetscInt faces[2] = {1, 1};
  PetscReal      lower[2], upper[2];
  DMLabel        label;
  PetscMPIInt    rank;
  void          *get_tmp;
  PetscInt64    *cidx;
  PetscMPIInt    iflg;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  // Create serial mesh
  lower[0] = (PetscReal)(rank % 2);
  lower[1] = (PetscReal)(rank / 2);
  upper[0] = (PetscReal)(rank % 2) + 1.;
  upper[1] = (PetscReal)(rank / 2) + 1.;
  PetscCall(DMPlexCreateBoxMesh(PETSC_COMM_SELF, 2, PETSC_FALSE, faces, lower, upper, NULL, PETSC_TRUE, 0, PETSC_TRUE, dm));
  PetscCall(PetscObjectSetName((PetscObject)*dm, "box"));
  // Flip edges to make fault non-oriented
  switch (rank) {
  case 2:
    PetscCall(DMPlexOrientPoint(*dm, 8, -1));
    break;
  case 3:
    PetscCall(DMPlexOrientPoint(*dm, 7, -1));
    break;
  default:
    break;
  }
  // Need this so that all procs create the cell types
  PetscCall(DMPlexGetCellTypeLabel(*dm, &label));
  // Replace comm in object (copied from PetscHeaderCreate/Destroy())
  PetscCall(PetscCommDestroy(&(*dm)->hdr.comm));
  PetscCall(PetscCommDuplicate(comm, &(*dm)->hdr.comm, &(*dm)->hdr.tag));
  PetscCallMPI(MPI_Comm_get_attr((*dm)->hdr.comm, Petsc_CreationIdx_keyval, &get_tmp, &iflg));
  PetscCheck(iflg, (*dm)->hdr.comm, PETSC_ERR_ARG_CORRUPT, "MPI_Comm does not have an object creation index");
  cidx            = (PetscInt64 *)get_tmp;
  (*dm)->hdr.cidx = (*cidx)++;
  // Create new pointSF
  {
    PetscSF      sf;
    PetscInt    *local  = NULL;
    PetscSFNode *remote = NULL;
    PetscInt     Nl;

    PetscCall(PetscSFCreate(comm, &sf));
    switch (rank) {
    case 0:
      Nl = 5;
      PetscCall(PetscMalloc1(Nl, &local));
      PetscCall(PetscMalloc1(Nl, &remote));
      local[0]        = 2;
      remote[0].index = 1;
      remote[0].rank  = 1;
      local[1]        = 3;
      remote[1].index = 1;
      remote[1].rank  = 2;
      local[2]        = 4;
      remote[2].index = 1;
      remote[2].rank  = 3;
      local[3]        = 6;
      remote[3].index = 5;
      remote[3].rank  = 2;
      local[4]        = 8;
      remote[4].index = 7;
      remote[4].rank  = 1;
      break;
    case 1:
      Nl = 3;
      PetscCall(PetscMalloc1(Nl, &local));
      PetscCall(PetscMalloc1(Nl, &remote));
      local[0]        = 3;
      remote[0].index = 1;
      remote[0].rank  = 3;
      local[1]        = 4;
      remote[1].index = 2;
      remote[1].rank  = 3;
      local[2]        = 6;
      remote[2].index = 5;
      remote[2].rank  = 3;
      break;
    case 2:
      Nl = 3;
      PetscCall(PetscMalloc1(Nl, &local));
      PetscCall(PetscMalloc1(Nl, &remote));
      local[0]        = 2;
      remote[0].index = 1;
      remote[0].rank  = 3;
      local[1]        = 4;
      remote[1].index = 3;
      remote[1].rank  = 3;
      local[2]        = 8;
      remote[2].index = 7;
      remote[2].rank  = 3;
      break;
    case 3:
      Nl = 0;
      break;
    default:
      SETERRQ(comm, PETSC_ERR_SUP, "This example only supports 4 ranks");
    }
    PetscCall(PetscSFSetGraph(sf, 9, Nl, local, PETSC_OWN_POINTER, remote, PETSC_OWN_POINTER));
    PetscCall(DMSetPointSF(*dm, sf));
    PetscCall(PetscSFDestroy(&sf));
  }
  // Create fault label
  PetscCall(DMCreateLabel(*dm, "fault"));
  PetscCall(DMGetLabel(*dm, "fault", &label));
  switch (rank) {
  case 0:
  case 2:
    PetscCall(DMLabelSetValue(label, 8, 1));
    PetscCall(DMLabelSetValue(label, 2, 0));
    PetscCall(DMLabelSetValue(label, 4, 0));
    break;
  case 1:
  case 3:
    PetscCall(DMLabelSetValue(label, 7, 1));
    PetscCall(DMLabelSetValue(label, 1, 0));
    PetscCall(DMLabelSetValue(label, 3, 0));
    break;
  default:
    break;
  }
  PetscCall(DMPlexOrientLabel(*dm, label));
  PetscCall(DMPlexLabelCohesiveComplete(*dm, label, NULL, 1, PETSC_FALSE, NULL));
  PetscCall(DMPlexDistributeSetDefault(*dm, PETSC_FALSE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateHexMesh1(MPI_Comm comm, AppCtx *user, DM *dm)
{
  const PetscInt faces[3] = {1, 1, 1};
  PetscReal      lower[3], upper[3];
  DMLabel        label;
  PetscMPIInt    rank;
  void          *get_tmp;
  PetscInt64    *cidx;
  PetscMPIInt    iflg;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  // Create serial mesh
  lower[0] = (PetscReal)(rank % 2);
  lower[1] = 0.;
  lower[2] = (PetscReal)(rank / 2);
  upper[0] = (PetscReal)(rank % 2) + 1.;
  upper[1] = 1.;
  upper[2] = (PetscReal)(rank / 2) + 1.;
  PetscCall(DMPlexCreateBoxMesh(PETSC_COMM_SELF, 3, PETSC_FALSE, faces, lower, upper, NULL, PETSC_TRUE, 0, PETSC_TRUE, dm));
  PetscCall(PetscObjectSetName((PetscObject)*dm, "box"));
  // Flip edges to make fault non-oriented
  switch (rank) {
  case 2:
    PetscCall(DMPlexOrientPoint(*dm, 10, -1));
    break;
  case 3:
    PetscCall(DMPlexOrientPoint(*dm, 9, -1));
    break;
  default:
    break;
  }
  // Need this so that all procs create the cell types
  PetscCall(DMPlexGetCellTypeLabel(*dm, &label));
  // Replace comm in object (copied from PetscHeaderCreate/Destroy())
  PetscCall(PetscCommDestroy(&(*dm)->hdr.comm));
  PetscCall(PetscCommDuplicate(comm, &(*dm)->hdr.comm, &(*dm)->hdr.tag));
  PetscCallMPI(MPI_Comm_get_attr((*dm)->hdr.comm, Petsc_CreationIdx_keyval, &get_tmp, &iflg));
  PetscCheck(iflg, (*dm)->hdr.comm, PETSC_ERR_ARG_CORRUPT, "MPI_Comm does not have an object creation index");
  cidx            = (PetscInt64 *)get_tmp;
  (*dm)->hdr.cidx = (*cidx)++;
  // Create new pointSF
  {
    PetscSF      sf;
    PetscInt    *local  = NULL;
    PetscSFNode *remote = NULL;
    PetscInt     Nl;

    PetscCall(PetscSFCreate(comm, &sf));
    switch (rank) {
    case 0:
      Nl = 15;
      PetscCall(PetscMalloc1(Nl, &local));
      PetscCall(PetscMalloc1(Nl, &remote));
      local[0]         = 2;
      remote[0].index  = 1;
      remote[0].rank   = 1;
      local[1]         = 4;
      remote[1].index  = 3;
      remote[1].rank   = 1;
      local[2]         = 5;
      remote[2].index  = 1;
      remote[2].rank   = 2;
      local[3]         = 6;
      remote[3].index  = 1;
      remote[3].rank   = 3;
      local[4]         = 7;
      remote[4].index  = 3;
      remote[4].rank   = 2;
      local[5]         = 8;
      remote[5].index  = 3;
      remote[5].rank   = 3;
      local[6]         = 17;
      remote[6].index  = 15;
      remote[6].rank   = 2;
      local[7]         = 18;
      remote[7].index  = 16;
      remote[7].rank   = 2;
      local[8]         = 20;
      remote[8].index  = 19;
      remote[8].rank   = 1;
      local[9]         = 21;
      remote[9].index  = 19;
      remote[9].rank   = 2;
      local[10]        = 22;
      remote[10].index = 19;
      remote[10].rank  = 3;
      local[11]        = 24;
      remote[11].index = 23;
      remote[11].rank  = 1;
      local[12]        = 26;
      remote[12].index = 25;
      remote[12].rank  = 1;
      local[13]        = 10;
      remote[13].index = 9;
      remote[13].rank  = 1;
      local[14]        = 14;
      remote[14].index = 13;
      remote[14].rank  = 2;
      break;
    case 1:
      Nl = 9;
      PetscCall(PetscMalloc1(Nl, &local));
      PetscCall(PetscMalloc1(Nl, &remote));
      local[0]        = 5;
      remote[0].index = 1;
      remote[0].rank  = 3;
      local[1]        = 6;
      remote[1].index = 2;
      remote[1].rank  = 3;
      local[2]        = 7;
      remote[2].index = 3;
      remote[2].rank  = 3;
      local[3]        = 8;
      remote[3].index = 4;
      remote[3].rank  = 3;
      local[4]        = 17;
      remote[4].index = 15;
      remote[4].rank  = 3;
      local[5]        = 18;
      remote[5].index = 16;
      remote[5].rank  = 3;
      local[6]        = 21;
      remote[6].index = 19;
      remote[6].rank  = 3;
      local[7]        = 22;
      remote[7].index = 20;
      remote[7].rank  = 3;
      local[8]        = 14;
      remote[8].index = 13;
      remote[8].rank  = 3;
      break;
    case 2:
      Nl = 9;
      PetscCall(PetscMalloc1(Nl, &local));
      PetscCall(PetscMalloc1(Nl, &remote));
      local[0]        = 2;
      remote[0].index = 1;
      remote[0].rank  = 3;
      local[1]        = 4;
      remote[1].index = 3;
      remote[1].rank  = 3;
      local[2]        = 6;
      remote[2].index = 5;
      remote[2].rank  = 3;
      local[3]        = 8;
      remote[3].index = 7;
      remote[3].rank  = 3;
      local[4]        = 20;
      remote[4].index = 19;
      remote[4].rank  = 3;
      local[5]        = 22;
      remote[5].index = 21;
      remote[5].rank  = 3;
      local[6]        = 24;
      remote[6].index = 23;
      remote[6].rank  = 3;
      local[7]        = 26;
      remote[7].index = 25;
      remote[7].rank  = 3;
      local[8]        = 10;
      remote[8].index = 9;
      remote[8].rank  = 3;
      break;
    case 3:
      Nl = 0;
      break;
    default:
      SETERRQ(comm, PETSC_ERR_SUP, "This example only supports 4 ranks");
    }
    PetscCall(PetscSFSetGraph(sf, 27, Nl, local, PETSC_OWN_POINTER, remote, PETSC_OWN_POINTER));
    PetscCall(DMSetPointSF(*dm, sf));
    PetscCall(PetscSFDestroy(&sf));
  }
  // Create fault label
  PetscCall(DMCreateLabel(*dm, "fault"));
  PetscCall(DMGetLabel(*dm, "fault", &label));
  switch (rank) {
  case 0:
  case 2:
    PetscCall(DMLabelSetValue(label, 10, 2));
    PetscCall(DMLabelSetValue(label, 20, 1));
    PetscCall(DMLabelSetValue(label, 22, 1));
    PetscCall(DMLabelSetValue(label, 24, 1));
    PetscCall(DMLabelSetValue(label, 26, 1));
    PetscCall(DMLabelSetValue(label, 2, 0));
    PetscCall(DMLabelSetValue(label, 4, 0));
    PetscCall(DMLabelSetValue(label, 6, 0));
    PetscCall(DMLabelSetValue(label, 8, 0));
    break;
  case 1:
  case 3:
    PetscCall(DMLabelSetValue(label, 9, 2));
    PetscCall(DMLabelSetValue(label, 19, 1));
    PetscCall(DMLabelSetValue(label, 21, 1));
    PetscCall(DMLabelSetValue(label, 23, 1));
    PetscCall(DMLabelSetValue(label, 25, 1));
    PetscCall(DMLabelSetValue(label, 1, 0));
    PetscCall(DMLabelSetValue(label, 3, 0));
    PetscCall(DMLabelSetValue(label, 5, 0));
    PetscCall(DMLabelSetValue(label, 7, 0));
    break;
  default:
    break;
  }
  PetscCall(DMPlexOrientLabel(*dm, label));
  PetscCall(DMPlexLabelCohesiveComplete(*dm, label, NULL, 1, PETSC_FALSE, NULL));
  PetscCall(DMPlexDistributeSetDefault(*dm, PETSC_FALSE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscFunctionBegin;
  switch (user->testNum) {
  case 1:
    PetscCall(CreateQuadMesh1(comm, user, dm));
    break;
  case 2:
    PetscCall(CreateHexMesh1(comm, user, dm));
    break;
  default:
    PetscCall(DMCreate(comm, dm));
    PetscCall(DMSetType(*dm, DMPLEX));
    break;
  }
  PetscCall(DMSetFromOptions(*dm));
  {
    const char *prefix;

    // We cannot redistribute with cohesive cells in the SF
    PetscCall(DMPlexDistributeSetDefault(*dm, PETSC_FALSE));
    PetscCall(PetscObjectGetOptionsPrefix((PetscObject)*dm, &prefix));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)*dm, "f0_"));
    PetscCall(DMSetFromOptions(*dm));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)*dm, "f1_"));
    PetscCall(DMSetFromOptions(*dm));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)*dm, prefix));
  }
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Create a displacement field, and some number of vector fault fields
static PetscErrorCode CreateDiscretization(DM dm, AppCtx *user)
{
  PetscSection   s;
  DMLabel        fault, faultSpace;
  PetscFE        fe;
  DMPolytopeType ct, fct;
  PetscInt       dim, cStart, fStart, Ncf = user->cohesiveFields;

  PetscFunctionBegin;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, NULL));
  PetscCall(DMPlexGetCellType(dm, cStart, &ct));
  PetscCall(DMGetLabel(dm, "fault", &fault));
  if (!fault) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(DMLabelView(fault, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(PetscFECreateByCell(PETSC_COMM_SELF, dim, dim, ct, "displacement_", PETSC_DETERMINE, &fe));
  PetscCall(PetscFESetName(fe, "displacement"));
  PetscCall(DMAddField(dm, NULL, (PetscObject)fe));
  PetscCall(PetscFEDestroy(&fe));

  // Make label for fault space definition
  PetscCall(DMCreateLabel(dm, "faultSpace"));
  PetscCall(DMGetLabel(dm, "faultSpace", &faultSpace));
  for (PetscInt d = 0; d <= dim; ++d) {
    PetscInt pStart, pEnd, pMax;

    PetscCall(DMPlexGetSimplexOrBoxCells(dm, d, NULL, &pMax));
    PetscCall(DMPlexGetHeightStratum(dm, d, &pStart, &pEnd));
    for (PetscInt p = pMax; p < pEnd; ++p) PetscCall(DMLabelSetValue(faultSpace, p, 1));
  }
  PetscCall(DMPlexGetHeightStratum(dm, 1, &fStart, NULL));
  PetscCall(DMPlexGetCellType(dm, fStart, &fct));
  if (Ncf > 0) {
    PetscCall(PetscFECreateByCell(PETSC_COMM_SELF, dim - 1, dim, fct, "faulttraction_", PETSC_DETERMINE, &fe));
    PetscCall(PetscFESetName(fe, "fault traction"));
    PetscCall(DMAddField(dm, faultSpace, (PetscObject)fe));
    PetscCall(PetscFEDestroy(&fe));
  }
  for (PetscInt f = 1; f < Ncf; ++f) {
    char name[256], opt[256];

    PetscCall(PetscSNPrintf(name, 256, "fault field %" PetscInt_FMT, f));
    PetscCall(PetscSNPrintf(opt, 256, "faultfield_%" PetscInt_FMT "_", f));
    PetscCall(PetscFECreateByCell(PETSC_COMM_SELF, dim - 1, dim, fct, opt, PETSC_DETERMINE, &fe));
    PetscCall(PetscFESetName(fe, name));
    PetscCall(DMAddField(dm, faultSpace, (PetscObject)fe));
    PetscCall(PetscFEDestroy(&fe));
  }
  PetscCall(DMCreateDS(dm));

  PetscCall(DMGetLocalSection(dm, &s));
  PetscCall(PetscObjectViewFromOptions((PetscObject)s, NULL, "-local_section_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Label cells 1 for negative side, and 2 for positive side
static PetscErrorCode CreateMaterialLabel(DM dm)
{
  DMLabel         fault, material;
  IS              faceIS;
  const PetscInt *faces;
  PetscReal       fvol, fcentroid[3], fnormal[3];
  PetscInt        dim, cStart, cEnd, Nf;

  PetscFunctionBegin;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMGetLabel(dm, "fault", &fault));
  PetscCall(DMCreateLabel(dm, "material"));
  PetscCall(DMGetLabel(dm, "material", &material));
  for (PetscInt s = 1; s < 3; ++s) {
    IS              pointIS;
    const PetscInt *points;
    PetscInt        n;

    PetscCall(DMLabelGetStratumIS(fault, s > 1 ? 100 + dim : -(100 + dim), &pointIS));
    if (!pointIS) continue;
    PetscCall(ISGetLocalSize(pointIS, &n));
    PetscCall(ISGetIndices(pointIS, &points));
    for (PetscInt i = 0; i < n; ++i) {
      PetscCall(DMLabelSetValue(material, points[i], s));
    }
    PetscCall(ISRestoreIndices(pointIS, &points));
    PetscCall(ISDestroy(&pointIS));
  }
  // This simple algorithm will work for now (note that cohesive cells get added into this label)
  PetscCall(DMLabelGetStratumIS(fault, dim - 1, &faceIS));
  PetscCall(ISGetLocalSize(faceIS, &Nf));
  PetscCheck(Nf > 0, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "Fault label must contain at least one face");
  PetscCall(ISGetIndices(faceIS, &faces));
  for (PetscInt i = 0; i < Nf; ++i) {
    const PetscInt face = faces[i];
    DMPolytopeType ct;

    PetscCall(DMPlexGetCellType(dm, face, &ct));
    if (DMPolytopeTypeGetDim(ct) != dim - 1) continue;
    PetscCall(DMPlexComputeCellGeometryFVM(dm, face, &fvol, fcentroid, fnormal));
    break;
  }
  PetscCall(ISRestoreIndices(faceIS, &faces));
  PetscCall(ISDestroy(&faceIS));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  for (PetscInt c = cStart; c < cEnd; ++c) {
    PetscReal vol, centroid[3];
    PetscInt  val;

    PetscCall(DMLabelGetValue(fault, c, &val));
    if (val >= 0) continue;
    PetscCall(DMPlexComputeCellGeometryFVM(dm, c, &vol, centroid, NULL));
    for (PetscInt e = 0; e < dim; ++e) centroid[e] -= fcentroid[e];
    if (DMPlex_DotRealD_Internal(dim, centroid, fnormal) > 0) PetscCall(DMLabelSetValue(material, c, 1));
    else PetscCall(DMLabelSetValue(material, c, 2));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Label cohesive cells and endcap faces 1
static PetscErrorCode CreateFaultLabel(DM dm)
{
  DMLabel  fault;
  PetscInt cMax, cEnd;

  PetscFunctionBegin;
  PetscCall(DMCreateLabel(dm, "faultCells"));
  PetscCall(DMGetLabel(dm, "faultCells", &fault));
  PetscCall(DMPlexGetSimplexOrBoxCells(dm, 0, &cEnd, &cMax));
  for (PetscInt c = cMax; c < cEnd; ++c) {
    const PetscInt *cone;

    PetscCall(DMLabelSetValue(fault, c, 1));
    PetscCall(DMPlexGetCone(dm, c, &cone));
    PetscCall(DMLabelSetValue(fault, cone[0], 1));
    PetscCall(DMLabelSetValue(fault, cone[1], 1));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode r(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, PetscCtx ctx)
{
  PetscInt d;
  for (d = 0; d < dim; ++d) u[d] = x[d];
  return PETSC_SUCCESS;
}

static PetscErrorCode rp1(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, PetscCtx ctx)
{
  PetscInt d;
  for (d = 0; d < dim; ++d) u[d] = x[d] + (d > 0 ? 1.0 : 0.0);
  return PETSC_SUCCESS;
}

static PetscErrorCode phi(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, PetscCtx ctx)
{
  PetscInt d;
  u[0] = -x[1];
  u[1] = x[0];
  for (d = 2; d < dim; ++d) u[d] = x[d];
  return PETSC_SUCCESS;
}

static void add_fields(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f[])
{
  PetscInt       d;
  const PetscInt offN = 0;
  const PetscInt offP = dim;
  for (d = 0; d < dim; ++d) f[d] = u[offN + d] + u[offP + d];
}

static void normal_field(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) f[d] = n[d];
}

/* \lambda \cdot (\psi_u^- - \psi_u^+) */
static void f0_bd_u_neg(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscInt Nc = dim + 1;
  for (PetscInt c = 0; c < Nc; ++c) f0[c] = -u[uOff[1] + c];
}

static void f0_bd_u_pos(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscInt Nc = dim + 1;
  for (PetscInt c = 0; c < Nc; ++c) f0[c] = u[uOff[1] + c];
}

/* (d - u^+ + u^-) \cdot \psi_\lambda */
static void f0_bd_l(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscInt Nc = uOff[2] - uOff[1];

  for (PetscInt c = 0; c < Nc; ++c) f0[c] = (c > 0 ? 1.0 : 0.0) + u[c] - u[Nc + c];
}

/* \psi_lambda \cdot (\psi_u^- - \psi_u^+) */
static void g0_bd_ul_neg(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  const PetscInt Nc = dim + 1;
  for (PetscInt c = 0; c < Nc; ++c) g0[c * Nc + c] = -1.0;
}

static void g0_bd_ul_pos(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  const PetscInt Nc = dim + 1;
  for (PetscInt c = 0; c < Nc; ++c) g0[c * Nc + c] = 1.0;
}

/* (-\psi_u^+ + \psi_u^-) \cdot \psi_\lambda */
static void g0_bd_lu(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  const PetscInt Nc = uOff[2] - uOff[1];

  for (PetscInt c = 0; c < Nc; ++c) {
    g0[c * Nc + c]           = -1.0;
    g0[Nc * Nc + c * Nc + c] = 1.0;
  }
}

static PetscErrorCode TestAssembly(DM dm, AppCtx *user)
{
  Mat           J;
  Vec           locX, locF, locW;
  PetscDS       probh;
  DMLabel       fault, material;
  DM            dmFault;
  IS            cohesiveCells;
  PetscFE       fe;
  PetscWeakForm wf;
  PetscFormKey  keys[3];
  PetscErrorCode (*initialGuess[2])(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar u[], PetscCtx ctx);
  DMPolytopeType fct;
  PetscInt       dim, fStart, Nf, cMax, cEnd, id;
  PetscMPIInt    rank, size;

  PetscFunctionBegin;
  PetscCall(DMGetNumFields(dm, &Nf));
  if (Nf <= 0) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank));
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)dm), &size));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetSimplexOrBoxCells(dm, 0, NULL, &cMax));
  PetscCall(DMPlexGetHeightStratum(dm, 0, NULL, &cEnd));
  if (size > 1) {
    PetscSF         sf;
    const PetscInt *leaves;
    PetscInt       *points;
    PetscInt        Nl, l, Ncoh = 0;

    PetscCall(DMGetPointSF(dm, &sf));
    PetscCall(PetscSFGetGraph(sf, NULL, &Nl, &leaves, NULL));
    for (PetscInt c = cMax; c < cEnd; ++c) {
      PetscCall(PetscFindInt(c, Nl, leaves, &l));
      if (l < 0) ++Ncoh;
    }
    PetscCall(PetscMalloc1(Ncoh, &points));
    Ncoh = 0;
    for (PetscInt c = cMax; c < cEnd; ++c) {
      PetscCall(PetscFindInt(c, Nl, leaves, &l));
      if (l < 0) points[Ncoh++] = c;
    }
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF, Ncoh, points, PETSC_OWN_POINTER, &cohesiveCells));
  } else {
    PetscCall(ISCreateStride(PETSC_COMM_SELF, cEnd - cMax, cMax, 1, &cohesiveCells));
  }
  PetscCall(CreateFaultLabel(dm));
  PetscCall(DMGetLabel(dm, "faultCells", &fault));
  PetscCall(DMGetLocalVector(dm, &locX));
  PetscCall(PetscObjectSetName((PetscObject)locX, "Local Solution"));
  PetscCall(DMGetLocalVector(dm, &locF));
  PetscCall(PetscObjectSetName((PetscObject)locF, "Local Residual"));
  PetscCall(DMCreateMatrix(dm, &J));
  PetscCall(PetscObjectSetName((PetscObject)J, "Jacobian"));

  /* The initial guess has displacement shifted by one unit in each fault parallel direction across the fault */
  PetscCall(CreateMaterialLabel(dm));
  PetscCall(DMGetLabel(dm, "material", &material));
  id              = 1;
  initialGuess[0] = r;
  initialGuess[1] = NULL;
  PetscCall(DMProjectFunctionLabelLocal(dm, 0.0, material, 1, &id, PETSC_DETERMINE, NULL, initialGuess, NULL, INSERT_VALUES, locX));
  id              = 2;
  initialGuess[0] = rp1;
  initialGuess[1] = NULL;
  PetscCall(DMProjectFunctionLabelLocal(dm, 0.0, material, 1, &id, PETSC_DETERMINE, NULL, initialGuess, NULL, INSERT_VALUES, locX));
  id              = 1;
  initialGuess[0] = NULL;
  initialGuess[1] = phi;
  PetscCall(DMProjectFunctionLabelLocal(dm, 0.0, fault, 1, &id, PETSC_DETERMINE, NULL, initialGuess, NULL, INSERT_VALUES, locX));
  PetscCall(PetscObjectViewSynchronizedFromOptions((PetscObject)locX, (PetscObject)dm, "-local_solution_view"));

  // Test projection to fault mesh
  if (cMax < cEnd) {
    PetscCall(DMPlexCreateCohesiveSubmesh(dm, PETSC_FALSE, NULL, 0, &dmFault));
    PetscCall(PetscObjectSetName((PetscObject)dmFault, "Fault Mesh"));
    PetscCall(DMViewFromOptions(dmFault, NULL, "-fault_view"));
    PetscCall(DMPlexOrient(dmFault));
    PetscCall(DMPlexGetHeightStratum(dm, 1, &fStart, NULL));
    PetscCall(DMPlexGetCellType(dm, fStart, &fct));
    //PetscCall(PetscFECreateByCell(PETSC_COMM_SELF, dim - 1, dim, fct, "fault_field_", PETSC_DETERMINE, &fe));
    PetscCall(PetscFECreateDefault(PETSC_COMM_SELF, dim - 1, dim, PETSC_TRUE, "fault_field_", PETSC_DETERMINE, &fe));
    PetscCall(PetscFESetName(fe, "fault_field"));
    PetscCall(DMAddField(dmFault, NULL, (PetscObject)fe));
    PetscCall(PetscFEDestroy(&fe));
    PetscCall(DMCreateDS(dmFault));
    PetscCall(DMGetLocalVector(dmFault, &locW));
    PetscCall(DMViewFromOptions(dmFault, NULL, "-cohesive_view"));
    void (*faultFuncs[1])(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f[]);

    DMLabel  depthLabel;
    PetscInt depth;
    PetscCall(DMPlexGetDepthLabel(dmFault, &depthLabel));
    PetscCall(DMPlexGetDepth(dmFault, &depth));
    id = depth - 1;
    /* w = r + rp1 */
    faultFuncs[0] = add_fields;
    PetscCall(DMProjectBdFieldLabelLocal(dmFault, 0.0, depthLabel, 1, &id, PETSC_DETERMINE, NULL, locX, faultFuncs, INSERT_VALUES, locW));
    PetscCall(PetscObjectViewSynchronizedFromOptions((PetscObject)locW, (PetscObject)dm, "-local_projection_view"));

    /* w = fault_normal */
    faultFuncs[0] = normal_field;
    PetscCall(DMProjectBdFieldLabelLocal(dmFault, 0.0, depthLabel, 1, &id, PETSC_DETERMINE, NULL, locX, faultFuncs, INSERT_VALUES, locW));
    PetscCall(PetscObjectViewSynchronizedFromOptions((PetscObject)locW, (PetscObject)dm, "-local_projection_view"));
    PetscCall(DMRestoreLocalVector(dmFault, &locW));
    PetscCall(DMDestroy(&dmFault));
  }

  PetscCall(DMGetCellDS(dm, cMax, &probh, NULL));
  PetscCall(PetscDSGetWeakForm(probh, &wf));
  PetscCall(PetscDSGetNumFields(probh, &Nf));
  PetscCall(PetscWeakFormSetIndexBdResidual(wf, material, 1, 0, 0, 0, f0_bd_u_neg, 0, NULL));
  PetscCall(PetscWeakFormSetIndexBdResidual(wf, material, 2, 0, 0, 0, f0_bd_u_pos, 0, NULL));
  PetscCall(PetscWeakFormSetIndexBdJacobian(wf, material, 1, 0, 1, 0, 0, g0_bd_ul_neg, 0, NULL, 0, NULL, 0, NULL));
  PetscCall(PetscWeakFormSetIndexBdJacobian(wf, material, 2, 0, 1, 0, 0, g0_bd_ul_pos, 0, NULL, 0, NULL, 0, NULL));
  if (Nf > 1) {
    PetscCall(PetscWeakFormSetIndexBdResidual(wf, fault, 1, 1, 0, 0, f0_bd_l, 0, NULL));
    PetscCall(PetscWeakFormSetIndexBdJacobian(wf, fault, 1, 1, 0, 0, 0, g0_bd_lu, 0, NULL, 0, NULL, 0, NULL));
  }
  if (rank == 0) PetscCall(PetscDSView(probh, NULL));

  keys[0].label = material;
  keys[0].value = 1;
  keys[0].field = 0;
  keys[0].part  = 0;
  keys[1].label = material;
  keys[1].value = 2;
  keys[1].field = 0;
  keys[1].part  = 0;
  keys[2].label = fault;
  keys[2].value = 1;
  keys[2].field = 1;
  keys[2].part  = 0;
  PetscCall(VecSet(locF, 0.));
  PetscCall(DMPlexComputeResidualHybridByKey(dm, keys, cohesiveCells, 0.0, locX, NULL, 0.0, locF, user));
  PetscCall(PetscObjectViewSynchronizedFromOptions((PetscObject)locF, (PetscObject)dm, "-local_residual_view"));
  PetscCall(MatZeroEntries(J));
  PetscCall(DMPlexComputeJacobianHybridByKey(dm, keys, cohesiveCells, 0.0, 0.0, locX, NULL, J, J, user));
  PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
  PetscCall(MatViewFromOptions(J, NULL, "-local_jacobian_view"));

  PetscCall(DMRestoreLocalVector(dm, &locX));
  PetscCall(DMRestoreLocalVector(dm, &locF));
  PetscCall(MatDestroy(&J));
  PetscCall(ISDestroy(&cohesiveCells));

  if (cMax < cEnd) {
    PetscDS         ds;
    PetscFE         fe;
    PetscQuadrature quad;
    IS             *perm;
    const PetscInt *cone;
    PetscInt        Na, a;

    PetscCall(DMPlexGetCone(dm, cMax, &cone));
    PetscCall(DMGetCellDS(dm, cMax, &ds, NULL));
    PetscCall(PetscDSGetDiscretization(ds, 0, (PetscObject *)&fe));
    PetscCall(PetscFEGetQuadrature(fe, &quad));
    PetscCall(PetscQuadratureComputePermutations(quad, &Na, &perm));
    for (a = 0; a < Na; ++a) PetscCall(ISDestroy(&perm[a]));
    PetscCall(PetscFree(perm));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM     dm;
  AppCtx user;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &user, &dm));
  PetscCall(CreateDiscretization(dm, &user));
  PetscCall(TestAssembly(dm, &user));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    requires: triangle
    args: -dm_refine 1 -dm_plex_transform_type cohesive_extrude \
            -dm_plex_transform_active fault -dm_plex_save_transform -dm_plex_check_transform \
          -dm_view ::ascii_info_detail -coarse_dm_view ::ascii_info_detail \
          -displacement_petscspace_degree 1 -faulttraction_petscspace_degree 1 \
            -local_section_view -local_solution_view -local_residual_view -local_jacobian_view
    filter: sed -e "s/_start//g" -e "s/f0_bd_u_neg//g" -e "s/f0_bd_u_pos//g" -e "s/f0_bd_l//g" -e "s/g0_bd_ul_neg//g" -e "s/g0_bd_ul_pos//g" -e "s/g0_bd_lu//g" -e "s~_ZL.*~~g"

    test:
      suffix: tri_0
      args: -dm_plex_box_faces 1,1 -dm_plex_cohesive_label_fault 8
    test:
      suffix: tri_1
      args: -dm_plex_box_faces 1,1 -dm_plex_cohesive_label_fault 8 \
              -dm_plex_transform_extrude_use_tensor 0
    test:
      suffix: tri_2
      args: -dm_plex_file_contents dat:tri_2_cv -dm_plex_cohesive_label_fault 11,15
    test:
      suffix: tri_2_perm
      args: -dm_plex_file_contents dat:tri_2_cv -dm_plex_cohesive_label_fault 11,15 \
            -dm_reorder_section -dm_reorder_section_type cohesive
    # Note that the mesh is not parallel when the cohesive label is oriented
    test:
      suffix: tri_3
      nsize: 2
      args: -dm_plex_file_contents dat:tri_2_cv -dm_plex_cohesive_label_fault 11,15 \
              -petscpartitioner_type shell -petscpartitioner_shell_sizes 2,2 \
              -petscpartitioner_shell_points 0,3,1,2

  testset:
    requires: triangle
    args: -dm_plex_option_phases coh_,ref_ \
            -coh_dm_refine 1 -coh_dm_plex_transform_type cohesive_extrude \
              -coh_dm_plex_transform_active fault \
            -ref_dm_refine 1 -ref_dm_plex_transform_type refine_regular \
          -dm_view ::ascii_info_detail \
          -displacement_petscspace_degree 1 -faulttraction_petscspace_degree 1 \
            -local_section_view -local_solution_view -local_residual_view -local_jacobian_view
    filter: sed -e "s/_start//g" -e "s/f0_bd_u_neg//g" -e "s/f0_bd_u_pos//g" -e "s/f0_bd_l//g" -e "s/g0_bd_ul_neg//g" -e "s/g0_bd_ul_pos//g" -e "s/g0_bd_lu//g" -e "s~_ZL.*~~g"

    test:
      suffix: tri_0_ref
      args: -dm_plex_box_faces 1,1 -dm_plex_cohesive_label_fault 8
    test:
      suffix: tri_2_ref
      args: -dm_plex_file_contents dat:tri_2_cv -dm_plex_cohesive_label_fault 11,15
    test:
      suffix: tri_2_ref_perm
      args: -dm_plex_file_contents dat:tri_2_cv -dm_plex_cohesive_label_fault 11,15 \
            -dm_reorder_section -dm_reorder_section_type cohesive

  testset:
    args: -dm_plex_simplex 0 -dm_plex_box_faces 2,1 \
          -dm_refine 1 -dm_plex_transform_type cohesive_extrude \
            -dm_plex_transform_active fault -dm_plex_cohesive_label_fault 13 \
            -dm_plex_save_transform -dm_plex_check_transform \
          -dm_view ::ascii_info_detail -coarse_dm_view ::ascii_info_detail \
          -displacement_petscspace_degree 1 -faulttraction_petscspace_degree 1 \
            -local_section_view -local_solution_view -local_residual_view -local_jacobian_view
    filter: sed -e "s/_start//g" -e "s/f0_bd_u_neg//g" -e "s/f0_bd_u_pos//g" -e "s/f0_bd_l//g" -e "s/g0_bd_ul_neg//g" -e "s/g0_bd_ul_pos//g" -e "s/g0_bd_lu//g" -e "s~_ZL.*~~g"

    test:
      suffix: quad_0
    test:
      suffix: quad_1
      args: -dm_plex_transform_extrude_use_tensor 0
    test:
      suffix: quad_2
      nsize: 2
      args: -petscpartitioner_type simple

  test:
    suffix: quad_3
    nsize: 4
    args: -test_num 1 \
          -dm_refine 1 -dm_plex_transform_type cohesive_extrude \
            -dm_plex_transform_active fault -dm_plex_save_transform -dm_plex_check_transform \
          -dm_view ::ascii_info_detail -coarse_dm_view ::ascii_info_detail \
          -orientation_view -orientation_view_synchronized \
          -displacement_petscspace_degree 1 -faulttraction_petscspace_degree 1 \
            -local_section_view -local_solution_view -local_residual_view -local_jacobian_view
    filter: sed -e "s/_start//g" -e "s/f0_bd_u_neg//g" -e "s/f0_bd_u_pos//g" -e "s/f0_bd_l//g" -e "s/g0_bd_ul_neg//g" -e "s/g0_bd_ul_pos//g" -e "s/g0_bd_lu//g" -e "s~_ZL.*~~g"

  test:
    suffix: quad_4
    args: -dm_plex_simplex 0 -dm_plex_box_faces 3,2 \
          -dm_refine 1 -dm_plex_transform_type cohesive_extrude \
            -dm_plex_transform_active fault -dm_plex_cohesive_label_fault 22,23 \
            -dm_plex_save_transform -dm_plex_check_transform \
          -dm_view ::ascii_info_detail -coarse_dm_view ::ascii_info_detail \
          -displacement_petscspace_degree 1 -faulttraction_petscspace_degree 1 \
            -local_section_view -local_solution_view -local_residual_view -local_jacobian_view
    filter: sed -e "s/_start//g" -e "s/f0_bd_u_neg//g" -e "s/f0_bd_u_pos//g" -e "s/f0_bd_l//g" -e "s/g0_bd_ul_neg//g" -e "s/g0_bd_ul_pos//g" -e "s/g0_bd_lu//g" -e "s~_ZL.*~~g"

  test:
    suffix: quad_5
    args: -dm_plex_simplex 0 -dm_plex_box_faces 3,2 \
            -dm_plex_cohesive_label_fault0 21 \
            -dm_plex_cohesive_label_fault1 23 \
          -f0_dm_refine 1 -f0_dm_plex_transform_type cohesive_extrude \
            -f0_dm_plex_transform_active fault0  -f0_coarse_dm_view ::ascii_info_detail \
          -f1_dm_refine 1 -f1_dm_plex_transform_type cohesive_extrude \
            -f1_dm_plex_transform_active fault1  -f1_coarse_dm_view ::ascii_info_detail \
          -dm_plex_save_transform -dm_plex_check_transform \
          -dm_view ::ascii_info_detail
    filter: sed -e "s/_start//g" -e "s/f0_bd_u_neg//g" -e "s/f0_bd_u_pos//g" -e "s/f0_bd_l//g" -e "s/g0_bd_ul_neg//g" -e "s/g0_bd_ul_pos//g" -e "s/g0_bd_lu//g" -e "s~_ZL.*~~g"

  test:
    suffix: quad_6
    args: -dm_plex_simplex 0 -dm_plex_box_faces 3,2 \
            -dm_plex_cohesive_label_fault0 22,23 \
            -dm_plex_cohesive_label_fault1 32 \
          -f0_dm_refine 1 -f0_dm_plex_transform_type cohesive_extrude \
            -f0_dm_plex_transform_active fault0  -f0_coarse_dm_view ::ascii_info_detail \
          -f1_dm_refine 1 -f1_dm_plex_transform_type cohesive_extrude \
            -f1_dm_plex_transform_active fault1  -f1_coarse_dm_view ::ascii_info_detail \
          -dm_plex_save_transform -dm_plex_check_transform \
          -dm_view ::ascii_info_detail
    filter: sed -e "s/_start//g" -e "s/f0_bd_u_neg//g" -e "s/f0_bd_u_pos//g" -e "s/f0_bd_l//g" -e "s/g0_bd_ul_neg//g" -e "s/g0_bd_ul_pos//g" -e "s/g0_bd_lu//g" -e "s~_ZL.*~~g"

  test:
    suffix: quad_6w
    args: -dm_plex_simplex 0 -dm_plex_box_faces 3,2 \
            -dm_plex_cohesive_label_fault0 22,23 \
            -dm_plex_cohesive_label_fault1 32 \
          -f0_dm_refine 1 -f0_dm_plex_transform_type cohesive_extrude \
            -f0_dm_plex_transform_active fault0  -f0_coarse_dm_view ::ascii_info_detail \
            -f0_dm_plex_transform_cohesive_width 0.05 \
          -f1_dm_refine 1 -f1_dm_plex_transform_type cohesive_extrude \
            -f1_dm_plex_transform_active fault1  -f1_coarse_dm_view ::ascii_info_detail \
            -f1_dm_plex_transform_cohesive_width 0.05 \
          -dm_plex_save_transform -dm_plex_check_transform \
          -dm_view ::ascii_info_detail
    filter: sed -e "s/_start//g" -e "s/f0_bd_u_neg//g" -e "s/f0_bd_u_pos//g" -e "s/f0_bd_l//g" -e "s/g0_bd_ul_neg//g" -e "s/g0_bd_ul_pos//g" -e "s/g0_bd_lu//g" -e "s~_ZL.*~~g"

  testset:
    args: -dm_plex_simplex 0 -dm_plex_box_faces 2,1 -dm_plex_cohesive_label_fault 13 \
          -dm_plex_option_phases coh_,ref_ \
            -coh_dm_refine 1 -coh_dm_plex_transform_type cohesive_extrude \
              -coh_dm_plex_transform_active fault \
            -ref_dm_refine 1 -ref_dm_plex_transform_type refine_regular \
          -dm_view ::ascii_info_detail \
          -displacement_petscspace_degree 1 -faulttraction_petscspace_degree 1 \
            -local_section_view -local_solution_view -local_residual_view -local_jacobian_view
    filter: sed -e "s/_start//g" -e "s/f0_bd_u_neg//g" -e "s/f0_bd_u_pos//g" -e "s/f0_bd_l//g" -e "s/g0_bd_ul_neg//g" -e "s/g0_bd_ul_pos//g" -e "s/g0_bd_lu//g" -e "s~_ZL.*~~g"

    test:
      suffix: quad_0_ref

  testset:
    args: -dm_plex_dim 3 -dm_plex_shape doublet \
          -dm_refine 1 -dm_plex_transform_type cohesive_extrude \
            -dm_plex_transform_active fault -dm_plex_cohesive_label_fault 7 \
            -dm_plex_save_transform -dm_plex_check_transform \
          -dm_view ::ascii_info_detail -coarse_dm_view ::ascii_info_detail \
          -displacement_petscspace_degree 1 -faulttraction_petscspace_degree 1 \
            -local_section_view -local_solution_view -local_residual_view -local_jacobian_view
    filter: sed -e "s/_start//g" -e "s/f0_bd_u_neg//g" -e "s/f0_bd_u_pos//g" -e "s/f0_bd_l//g" -e "s/g0_bd_ul_neg//g" -e "s/g0_bd_ul_pos//g" -e "s/g0_bd_lu//g" -e "s~_ZL.*~~g"

    test:
      suffix: tet_0
    test:
      suffix: tet_1
      nsize: 2
      args: -petscpartitioner_type simple

  testset:
    args: -dm_plex_dim 3 -dm_plex_shape doublet -dm_plex_cohesive_label_fault 7 \
          -dm_plex_option_phases coh_,ref_ \
            -coh_dm_refine 1 -coh_dm_plex_transform_type cohesive_extrude \
              -coh_dm_plex_transform_active fault \
            -ref_dm_refine 1 -ref_dm_plex_transform_type refine_regular \
          -dm_view ::ascii_info_detail \
          -displacement_petscspace_degree 1 -faulttraction_petscspace_degree 1 \
            -local_section_view -local_solution_view -local_residual_view -local_jacobian_view
    filter: sed -e "s/_start//g" -e "s/f0_bd_u_neg//g" -e "s/f0_bd_u_pos//g" -e "s/f0_bd_l//g" -e "s/g0_bd_ul_neg//g" -e "s/g0_bd_ul_pos//g" -e "s/g0_bd_lu//g" -e "s~_ZL.*~~g"

    test:
      suffix: tet_0_ref

  testset:
    args: -dm_plex_dim 3 -dm_plex_simplex 0 -dm_plex_box_faces 2,1,1 -dm_plex_box_upper 2,1,1 \
          -dm_refine 1 -dm_plex_transform_type cohesive_extrude \
            -dm_plex_transform_active fault -dm_plex_cohesive_label_fault 15 \
            -dm_plex_save_transform -dm_plex_check_transform \
          -dm_view ::ascii_info_detail -coarse_dm_view ::ascii_info_detail \
          -displacement_petscspace_degree 1 -faulttraction_petscspace_degree 1 \
            -local_section_view -local_solution_view -local_residual_view -local_jacobian_view
    filter: sed -e "s/_start//g" -e "s/f0_bd_u_neg//g" -e "s/f0_bd_u_pos//g" -e "s/f0_bd_l//g" -e "s/g0_bd_ul_neg//g" -e "s/g0_bd_ul_pos//g" -e "s/g0_bd_lu//g" -e "s~_ZL.*~~g"

    test:
      suffix: hex_0
    test:
      suffix: hex_1
      nsize: 2
      args: -petscpartitioner_type simple

  test:
    suffix: hex_2
    nsize: 4
    args: -test_num 2 \
          -dm_refine 1 -dm_plex_transform_type cohesive_extrude \
            -dm_plex_transform_active fault -dm_plex_save_transform -dm_plex_check_transform \
          -dm_view ::ascii_info_detail -coarse_dm_view ::ascii_info_detail \
          -orientation_view -orientation_view_synchronized \
          -displacement_petscspace_degree 1 -faulttraction_petscspace_degree 1 \
            -local_section_view -local_solution_view -local_residual_view -local_jacobian_view
    filter: sed -e "s/_start//g" -e "s/f0_bd_u_neg//g" -e "s/f0_bd_u_pos//g" -e "s/f0_bd_l//g" -e "s/g0_bd_ul_neg//g" -e "s/g0_bd_ul_pos//g" -e "s/g0_bd_lu//g" -e "s~_ZL.*~~g"

  test:
    suffix: hex_3
    args: -dm_plex_dim 3 -dm_plex_simplex 0 -dm_plex_box_faces 2,1,2 -dm_plex_box_upper 2.,1.,2. \
            -dm_plex_cohesive_label_fault0 37,40 \
            -dm_plex_cohesive_label_fault1 26 \
          -f0_dm_refine 1 -f0_dm_plex_transform_type cohesive_extrude \
            -f0_dm_plex_transform_active fault0  -f0_coarse_dm_view ::ascii_info_detail \
          -f1_dm_refine 1 -f1_dm_plex_transform_type cohesive_extrude \
            -f1_dm_plex_transform_active fault1  -f1_coarse_dm_view ::ascii_info_detail \
          -dm_plex_save_transform -dm_plex_check_transform \
          -dm_view ::ascii_info_detail
    filter: sed -e "s/_start//g" -e "s/f0_bd_u_neg//g" -e "s/f0_bd_u_pos//g" -e "s/f0_bd_l//g" -e "s/g0_bd_ul_neg//g" -e "s/g0_bd_ul_pos//g" -e "s/g0_bd_lu//g" -e "s~_ZL.*~~g"

  test:
    suffix: hex_4
    args: -dm_plex_dim 3 -dm_plex_simplex 0 -dm_plex_box_faces 4,1,2 -dm_plex_box_upper 4.,1.,2. \
            -dm_plex_cohesive_label_fault0 65,68 \
            -dm_plex_cohesive_label_fault1 46 \
          -f0_dm_refine 1 -f0_dm_plex_transform_type cohesive_extrude \
            -f0_dm_plex_transform_active fault0  -f0_coarse_dm_view ::ascii_info_detail \
          -f1_dm_refine 1 -f1_dm_plex_transform_type cohesive_extrude \
            -f1_dm_plex_transform_active fault1  -f1_coarse_dm_view ::ascii_info_detail \
          -dm_plex_save_transform -dm_plex_check_transform \
          -dm_view ::ascii_info_detail
    filter: sed -e "s/_start//g" -e "s/f0_bd_u_neg//g" -e "s/f0_bd_u_pos//g" -e "s/f0_bd_l//g" -e "s/g0_bd_ul_neg//g" -e "s/g0_bd_ul_pos//g" -e "s/g0_bd_lu//g" -e "s~_ZL.*~~g"

  testset:
    args: -dm_plex_dim 3 -dm_plex_simplex 0 -dm_plex_box_faces 2,1,1 -dm_plex_box_upper 2,1,1 -dm_plex_cohesive_label_fault 15 \
          -dm_plex_option_phases coh_,ref_ \
            -coh_dm_refine 1 -coh_dm_plex_transform_type cohesive_extrude \
              -coh_dm_plex_transform_active fault \
            -ref_dm_refine 1 -ref_dm_plex_transform_type refine_regular \
          -dm_view ::ascii_info_detail \
          -displacement_petscspace_degree 1 -faulttraction_petscspace_degree 1 \
            -local_section_view -local_solution_view -local_residual_view -local_jacobian_view
    filter: sed -e "s/_start//g" -e "s/f0_bd_u_neg//g" -e "s/f0_bd_u_pos//g" -e "s/f0_bd_l//g" -e "s/g0_bd_ul_neg//g" -e "s/g0_bd_ul_pos//g" -e "s/g0_bd_lu//g" -e "s~_ZL.*~~g"

    test:
      suffix: hex_0_ref

TEST*/
