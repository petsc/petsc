//this function generates the appropriate tex for the given matrix
//the data for the pc/ksp is taken directly from the dropdown lists
//displays visual of the nested pc=bjacobi block structure and/or nested pc=ksp block structure

//matrix refers to the id of the matrix. for example, "01"
function getTex(matrix) {

    //the following is Matt's explanation of using TeX with MathJax

    //TeX can be typed in as normal except for a few things while insde the " ":
    //To use a '\' character, it must be escaped by placing another \ in front of it
    //To use multiple lines, a \ must be used to escape the newline character
    //For example, to do \\, which is used in matrices to represent new row, you must put
    // \\\\ (four slashes instead of two), so that each one is escaped.
    //To do multiline tex, one must put \\( TeX \\) - this ads \( \) around the tex
    //to let mathjax know it is is multline tex. (the two \ are because you need to
    //escape the character.

    var ret = "";//returned value




    return ret;
}