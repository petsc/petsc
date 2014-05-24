//this function replaces Matt's old tex2.js
//instead of using subscript of subscript of subscript, etc. simply put the id of the matrix in the first subscript level. for example, A_{010}
//gives users a visual of the nested arrays

//this function is recursive and should always be called with parameter "0"
//this function gets the information from matInfo[]
function getMatrixTex(currentMatrix) {//weird because the longer the string is, the more deep the recursion is. "0" is the top level. digits are only added. never removed

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

var bjacobiSplits = [];//global variable that keeps track of how the bjacobi blocks were split


//this function generates the appropriate tex for the given matrix
//the data for the pc/ksp is taken directly from the dropdown lists
//displays visual of the nested pc=bjacobi block structure and/or nested pc=ksp block structure

//matrix refers to the id of the matrix. for example, "01"
//this function is recursive and should always be called with an empty string in the second parameter like so: getSpecificMatrixTex(matrix, "")
function getSpecificMatrixTex(matrix, endtag) {

    if(endtag == "") {//reset bjacobi splits data
        delete bjacobiSplits;
        bjacobiSplits.length=0;
    }

    var ret = "";//returned value

    if(getMatIndex(matrix) == -1)//invalid matrix
        return ret;

    var pc;
    var ksp;
    if(endtag == "") {
        pc  = $("#pcList"+matrix).val();
        ksp = $("#kspList"+matrix).val();
    }
    else {
        pc  = $("#pcList"+matrix+"_"+endtag).val();
        ksp = $("#kspList"+matrix+"_"+endtag).val();
    }

    //case 1: non-recursive base case
    if(pc != "ksp" && pc != "bjacobi") {
        return ksp+"/"+pc+"\\begin{cases} \\end{cases}";
    }

    //case 2: pc=ksp recursive case
    if(pc == "ksp") {
        ret += ksp+"/"+pc+"\\begin{cases}";

        endtag += "0";

        //ksp is a composite pc so there will be more options

        ret += getSpecificMatrixTex(matrix, endtag);
        ret += "\\end{cases}";
        return ret;
    }

    //case 3: pc=bjacobi recursive case
    else if(pc == "bjacobi") {
        ret += ksp+"/"+pc+"\\begin{cases}";

        endtag += "0";

        //bjacobi is a composite pc so there will be more options

        var blocks = $("#bjacobiBlocks"+matrix+"_"+endtag).val();

        if(blocks == "np")
            blocks = 2;
        else
            blocks = parseInt(blocks);

        //record that we split
        var idx = bjacobiSplits.length;
        bjacobiSplits[idx] = blocks;

        var childTex = getSpecificMatrixTex(matrix, endtag);
        for(var i=0; i<blocks; i++)
            ret += childTex + "\\\\";
        ret += "\\end{cases}";
        return ret;
    }

    return ret;
}

//this function generates the corresponding matrix for the specific matrix diagram
//this function is recursive and should always be called with parameter 0 like so: getSpecificMatrixTex2(0)
function getSpecificMatrixTex2(index) {

    //case 1: matrix was not split
    if(bjacobiSplits.length == 0) {
        return "\\left[ \\begin{array}{c} * \\end{array}\\right]";
    }

    //case 2: base case
    else if(index >= bjacobiSplits.length) {
        return "\\left[ \\begin{array}{c} * \\end{array}\\right]";
    }

    //case 3: recursive case
    else {
        var ret = "";

        var blocks = bjacobiSplits[index];
        var innerTex = "";

        var justify = "";
        for(var i=0; i<blocks-1; i++) {
            justify += "c@{}";
        }
        justify += "c";

        ret += "\\left[ \\begin{array}{"+justify+"}";//begin the matrix

        innerTex = getSpecificMatrixTex2(index+1);

        for(var i=0; i<blocks; i++) {//iterate thru entire square matrix row by row
            for(var j=0; j<i; j++) {//add the stars that go BEFORE the diagonal element
                ret += "* & ";
            }

            ret += innerTex;

            for(var j=i+1; j<blocks; j++) {//add the stars that go AFTER the diagonal element
                ret += "& *";
            }
            ret += "\\\\"; //add the backslash indicating the next matrix row
        }

        ret += "\\end{array}\\right]";//close the matrix

        return ret;
    }


}