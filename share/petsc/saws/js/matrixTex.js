//gives users a visual of the nested matricies
//instead of using subscript of subscript of subscript, etc. simply put the id of the matrix in the first subscript level. for example, A_{010}

//this function is recursive and should always be called with parameter "0"
//weird because the longer the string is, the more deep the recursion is. "0" is the top level. digits are only added. never removed
function getMatrixTex(data, endtag) {

    if(data[endtag] == undefined)
        return;

    //case 1: not logstruc. base case.
    if(!data[endtag].logstruc) {
        //return the appropriate tex

        var depth = getNumUnderscores(endtag);
        var color = colors[depth%colors.length];

        var subscript = endtag.replace(/_/g, ",");
        return "{\\color{"+color+"}" + "\\left[A_{" + subscript + "}" + "\\right]}";
    }

    //case 2: has more children. recursive case.
    else {
        var depth = getNumUnderscores(endtag);
        var color = colors[depth%colors.length];

        var ret = "";

        var blocks = data[endtag].pc_fieldsplit_blocks;
        var childrenTex = "";

        var justify = "";
        for(var i=0; i<blocks-1; i++) {
            justify += "c@{}";
        }
        justify += "c";

        ret += "\\color{"+color+"}" + "\\left[" + "\\color{black}" + "\\begin{array}{"+justify+"}";//begin the matrix

        for(var i=0; i<blocks; i++) {//NEED TO ADD STARS
            for(var j=0; j<i; j++) {//add the stars that go BEFORE the diagonal element
                ret += "* & ";
            }

            var childEndtag = endtag + "_" + i;
            //lay out chilren
            var childTex = getMatrixTex(data, childEndtag);
            if(childTex != "")
                ret += getMatrixTex(data, childEndtag);
            else {
                var subscript = childEndtag.replace(/_/g, ",");
                ret += "A_{" + subscript + "}";
            }

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

//this function displays a visual of the nested pc=bjacobi block structure and/or nested pc=ksp block structure
//this function is recursive and should almost always be called with parameter "0" like so: getSpecificMatrixTex("0")
function getSpecificMatrixTex(data, endtag) {

    if(endtag == "0") {//reset bjacobi splits data
        delete bjacobiSplits;
        bjacobiSplits.length=0;
    }

    var ret = "";//returned value

    if(data[endtag] == undefined) //invalid matrix
        return ret;

    var pc    = data[endtag].pc_type;
    var ksp   = data[endtag].ksp_type;

    //case 1: non-recursive base case
    if(pc != "ksp" && pc != "bjacobi") {
        return ksp+"/"+pc;//don't put braces at the very end
    }

    //case 2: pc=ksp recursive case
    if(pc == "ksp") {
        ret += "\\left.\\begin{aligned}";

        endtag += "_0";

        //ksp is a composite pc so there will be more options

        ret += getSpecificMatrixTex(endtag);
        ret += "\\end{aligned}\\right\\}"+ksp+"/"+pc;
        return ret;
    }

    //case 3: pc=bjacobi recursive case
    else if(pc == "bjacobi") {
        ret += "\\left.\\begin{aligned}";

        endtag += "_0";

        var blocks = data[endtag].pc_bjacobi_blocks;

        //record that we split
        var idx = bjacobiSplits.length;
        bjacobiSplits[idx] = blocks;

        var childTex = getSpecificMatrixTex(endtag);
        for(var i=0; i<blocks; i++)
            ret += childTex + "\\\\";
        ret += "\\end{aligned}\\right\\}" + ksp+"/"+pc;
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
