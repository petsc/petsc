//this function replaces Matt's old tex2.js
//instead of using subscript of subscript of subscript, etc. simply put the id of the matrix in the first subscript level. for example, A_{010}
//gives users a visual of the nested arrays

//this function is recursive and should always be called with parameter "0"
//this function gets the information from matInfo[]
function getMatrixTex(currentMatrix) {//weird because the longer the string is, the more deep the recursion is. "0" is the top level. digits are only added. never removed

    if(currentAsk == "-1")//this function shouldn't be called when currentAsk is -1
        return "";

    //case 1: not logstruc. base case.
    if(!matInfo[getMatIndex(currentMatrix)].logstruc) {
        //return the appropriate tex

        if(currentMatrix == currentAsk) //make red and bold
            return "\\color{red}{\\mathbf{A_{" + currentMatrix +" } }}";
        else //return black text
            return "A_{" + currentMatrix + "}";
    }

    //case 2: has more children. recursive case.
    else {
        var ret = "";

        var blocks = matInfo[getMatIndex(currentMatrix)].blocks;
        var childrenTex = "";

        var justify = "";
        for(var i=0; i<blocks-1; i++) {
            justify += "c@{}";
        }
        justify += "c";

        ret += "\\left[ \\begin{array}{"+justify+"}";//begin the matrix

        for(var i=0; i<blocks; i++) {//NEED TO ADD STARS
            for(var j=0; j<i; j++) {//add the stars that go BEFORE the diagonal element
                ret += "* & ";
            }

            var childID = currentMatrix + i;
            //lay out chilren
            var childTex = getMatrixTex(childID);
            if(childTex != "")
                ret += getMatrixTex(childID);
            else
                ret += "A_{"+childID+"}";

            for(var j=i+1; j<blocks; j++) {//add the stars that go AFTER the diagonal element
                ret += "& *";
            }
            ret += "\\\\"; //add the backslash indicating the next matrix row
        }

        ret += "\\end{array}\\right]";//close the matrix
        return ret;
    }

}




//this function generates the appropriate tex for the given matrix
//the data for the pc/ksp is taken directly from the dropdown lists
//displays visual of the nested pc=bjacobi block structure and/or nested pc=ksp block structure

//matrix refers to the id of the matrix. for example, "01"
function getSpecificMatrixTex(matrix) {

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

    if(getMatIndex(matrix) == -1)//invalid matrix
        return ret;

    //get the A-div of the matrix (simply append 'A' to the front)
    var div = "A"+matrix;

    var pc = $("#pcList"+matrix).val();

    /*if(pc == "ksp") {
        ret += "\begin{equation}
  S_{i,t}=
  \begin{cases}
    \begin{cases}
      [x_{i,t}=X^*, r_{i,t}=1] & \text{if  $\max\{X_{i,t}\}=X^*$} \\
      [x_{i,t}=\max\{X_{i,t}\}, r_{i,t}=0] & \text{if $\max\{X_{i,t}\} \neq X^*$}
    \end{cases}
    &\text{if $\sum_{i=1}^I u_{i,t-1}= \theta^{t-2} X^*$}\\
    \begin{cases}
      [x_{i,t}=1, r_{i,t}=1] & \hspace{\maxmin} \text{if $\min\{X_{i,t}\}=1$} \\
      [x_{i,t}=\min\{X_{i,t}\}, r_{i,t}=0] & \hspace{\maxmin} \text{if $\min\{X_{i,t}\} \neq 1$}
    \end{cases}
    &\text{otherwise}
  \end{cases}
\end{equation}";

        //ksp is a composite pc so there will be a div in the next position
        var generatedDiv="";
    generatedDiv = $("#"+pcListID).next().get(0).id; //this will be a div, eg. mg0_, bjacobi1_
    }
    else if(pc == "bjacobi") {
        ret += "";

        //bjacobi is a composite pc so there will be a div in the next position
        var generatedDiv="";
    generatedDiv = $("#"+pcListID).next().get(0).id; //this will be a div, eg. mg0_, bjacobi1_
    }*/

    //check t

    return ret;
}