//This array stores the tex, and MathJax interprets it on the fly. Each arrqy element refers to one
// specific matrix, highlighting and all.
function finalTex(matInfo, recursionCounter)
{

    //TeX can be typed in as normal except for a few things while insde the " ":
    //To use a '\' character, it must be escaped by placing another \ in front of it
    //To use multiple lines, a \ must be used to escape the newline character
    //For example, to do \\, which is used in matrices to represent new row, you must put
    // \\\\ (four slashes instead of two), so that each one is escaped.
    //To do multiline tex, one must put \\( TeX \\) - this ads \( \) around the tex
    //to let mathjax know it is is multline tex. (the two \ are because you need to
    //escape the character.



    //Switch on matrix level

    //level 0 has only one option
    if(matGetLevel(recursionCounter) == 0)
	return 	"<h1>\\begin{bmatrix} A \\end{bmatrix}</h1>"

    //level 1 has one options
    if(matGetLevel(recursionCounter) == 1)
    {
	return "<h3>\\begin{bmatrix} A_{1} & * \\\\ * & A_{2} \\end{bmatrix}</h3>"
    }

    //level 2 has three options (from the various possibilities)
    if(matGetLevel(recursionCounter) == 2)
    {
	//A1 (index 1) is logstruc and A2 is not logstruc
	if(matInfo[1].logstruc && !matInfo[2].logstruc)
	    return "\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
\\left[\\begin{array}{cc} \
A_{1_{1}} & * \\\\ \
* & A_{1_{2}} \\\\ \  							\\end{array}\\right] & * \\\\ \
* &  A_{2} \\\\ \
\\end{array}\\right]\\)"
	

	//A2 (index 2) is logstruc and A1 is not logstruc
	if(matInfo[2].logstruc && !matInfo[1].logstruc)
	    return "\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
A_{1} & * \\\\ \
* & \\left[\\begin{array}{cc} \
A_{2_{1}} & * \\\\ \
* & A_{2_{2}} \\\\ \
\\end{array}\\right] \\\\ \
\\end{array}\\right]\\)"

	

	//A2 (index 2) is logstruc and A1 is also logstruc
	if(matInfo[2].logstruc && matInfo[1].logstruc)
	    return "\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
\\left[\\begin{array}{cc} \
A_{1_{1}} & * \\\\ \
* & A_{1_{2}} \\\\ \
\\end{array}\\right] & * \\\\ \
* & \\left[\\begin{array}{cc} \
A_{2_{1}} & * \\\\ \
* & A_{2_{2}} \\\\ \
\\end{array}\\right] \\\\ \
\\end{array}\\right]\\)"

    }

    //level 3 has *many* *many* options (from the various possibilities
    if(matGetLevel(recursionCounter) == 3)
    {
	//A1 (index 1) is logstruc and A2 is not logstruc
	if(matInfo[1].logstruc && !matInfo[2].logstruc)
	{
	    //A11 (index 3) is log struc and A12 (index 4) is not log struc
	    if(matInfo[3].logstruc && !matInfo[4].logstruc)
		return "\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
\\left[\\begin{array}{cc} \
\\left[\\begin{array}{cc} \
A_{1_{1_{1}}} & * \\\\ \
* & A_{1_{1_{2}}} \\\\ \
\\end{array}\\right] & * \\\\ \
* &  A_{1_{2}} \\\\ \
\\end{array}\\right] & * \\\\ \
* &  A_{2} \\\\ \
\\end{array}\\right]\\)"

	    //A12 (index 4) is log struc and A11 (index 3) is not log struc
	    if(matInfo[4].logstruc && !matInfo[3].logstruc)
		return "\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
\\left[\\begin{array}{cc} \
A_{1_{1}} &  * \\\\ \
* & \\left[\\begin{array}{cc} \
A_{1_{2_{1}}} & * \\\\ \
* & A_{1_{2_{2}}} \\\\ \
\\end{array}\\right] \\\\ \
\\end{array}\\right] & * \\\\ \
* &  A_{2} \\\\ \
\\end{array}\\right]\\)"

	    //A11 (index 3) is log struc and A12 (index 4) is also log struc
	    if(matInfo[3].logstruc && matInfo[4].logstruc)
		return "\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
\\left[\\begin{array}{cc} \
\\left[\\begin{array}{cc} \
A_{1_{1_{1}}} & * \\\\ \
* & A_{1_{1_{2}}} \\\\ \
\\end{array}\\right] & * \\\\ \
* & \\left[\\begin{array}{cc} \
A_{1_{2_{1}}} & * \\\\ \
* & A_{1_{2_{2}}} \\\\ \
\\end{array}\\right] \\\\ \
\\end{array}\\right] & * \\\\ \
* &  A_{2} \\\\ \
\\end{array}\\right]\\)"

	}

	
	//A2 (index 2) is logstruc and A1 (index 1) is not logstruc
	if(matInfo[2].logstruc && !matInfo[1].logstruc)
	{
	    //A21 (index 5) is log struc and A22 (index 6) is not log struc
	    if(matInfo[5].logstruc && !matInfo[6].logstruc)
		return "\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
A_{1} & * \\\\ \
* & \\left[\\begin{array}{cc} \
\\left[\\begin{array}{cc} \
A_{2_{1_{1}}} & * \\\\ \
* & A_{2_{1_{2}}} \\\\ \
\\end{array}\\right] & * \\\\ \
* &  A_{2_{2}} \\\\ \
\\end{array}\\right]\\\\ \
\\end{array}\\right]\\)"
	    
	    //A22 (index 6) is log struc and A21 (index 5) is not log struc
	    if(matInfo[6].logstruc && !matInfo[5].logstruc)
		return "\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
A_{1} & * \\\\ \
* & \\left[\\begin{array}{cc} \
A_{2_{1}} &  * \\\\ \
* & \\left[\\begin{array}{cc} \
A_{2_{2_{1}}} & * \\\\ \
* & A_{2_{2_{2}}} \\\\ \
\\end{array}\\right] \\\\ \
\\end{array}\\right]\\\\ \
\\end{array}\\right]\\)"


	    //A21 (index 5) is log struc and A22 (index 6) is also log struc
	    if(matInfo[5].logstruc && matInfo[6].logstruc)
		return "\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
A_{1} & * \\\\ \
* & \\left[\\begin{array}{cc} \
\\left[\\begin{array}{cc} \
A_{2_{1_{1}}} & * \\\\ \
* & A_{2_{1_{2}}} \\\\ \
\\end{array}\\right] & * \\\\ \
* &\\left[\\begin{array}{cc} \
A_{2_{2_{1}}} & * \\\\ \
* & A_{2_{2_{2}}} \\\\ \
\\end{array}\\right] \\\\ \
\\end{array}\\right]\\\\ \
\\end{array}\\right]\\)"

	}

	//A2 (index 2) is logstruc and A1 (index 1) is also logstruc
	if(matInfo[2].logstruc && matInfo[1].logstruc)
	{
	    
	    //A11 (index 3) is log struc and A12 (index 4), A21 (index 5), and A22 (index 6) are not log struc
	    if(matInfo[3].logstruc && !matInfo[4].logstruc && !matInfo[5].logstruc && !matInfo[6].logstruc)
		return "\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
\\left[\\begin{array}{cc} \
\\left[\\begin{array}{cc} \
A_{1_{1_{1}}} & * \\\\ \
* & A_{1_{1_{2}}} \\\\ \
\\end{array}\\right] & * \\\\ \
* &  A_{1_{2}} \\\\ \
\\end{array}\\right] & * \\\\ \
* & \\left[\\begin{array}{cc} \
A_{2_{1}} & * \\\\ \
* &  A_{2_{2}} \\\\ \
\\end{array}\\right]\\\\ \
\\end{array}\\right]\\)"
	    
	    //A12 (index 4) is log struc and A11 (index 3), A21 (index 5), and A22 (index 6) are not log struc
	    if(matInfo[4].logstruc && !matInfo[3].logstruc && !matInfo[5].logstruc && !matInfo[6].logstruc)
		return "\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
\\left[\\begin{array}{cc} \
A_{1_{1}} &  * \\\\ \
* & \\left[\\begin{array}{cc} \
A_{1_{2{_1}}} & * \\\\ \
* & A_{1_{2_{2}}} \\\\ \
\\end{array}\\right]\\\\ \
\\end{array}\\right] & * \\\\ \
* & \\left[\\begin{array}{cc} \
A_{2_{1}} & * \\\\ \
* &  A_{2_{2}} \\\\ \
\\end{array}\\right]\\\\ \
\\end{array}\\right]\\)"
	    
	    //A21 (index 5) is log struc and A11 (index 3), A12 (index 4), and A22 (index 6) are not log struc
	    if(matInfo[5].logstruc && !matInfo[4].logstruc && !matInfo[3].logstruc && !matInfo[6].logstruc)
		return "\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
\\left[\\begin{array}{cc} \
A_{1_{1}} & * \\\\ \
* &  A_{1_{2}} \\\\ \
\\end{array}\\right] & * \\\\ \
* & \\left[\\begin{array}{cc} \
\\left[\\begin{array}{cc} \
A_{2_{1_{1}}} & * \\\\ \
* & A_{2_{1_{2}}} \\\\ \
\\end{array}\\right] & * \\\\ \
* &  A_{2_{2}} \\\\ \
\\end{array}\\right]\\\\ \
\\end{array}\\right]\\)"
	    
	    //A22 (index 6) is log struc and A11 (index 3), A12 (index 4), and A22 (index 6) are not log struc
	    if(matInfo[6].logstruc && !matInfo[3].logstruc && !matInfo[5].logstruc && !matInfo[4].logstruc)
		return "\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
\\left[\\begin{array}{cc} \
A_{1_{1}} & * \\\\ \
* &  A_{1_{2}} \\\\ \
\\end{array}\\right] & * \\\\ \
* & \\left[\\begin{array}{cc} \
A_{2_{1}} &  * \\\\ \
* & \\left[\\begin{array}{cc} \
A_{2_{2_{1}}} & * \\\\ \
* & A_{2_{2_{2}}} \\\\ \
\\end{array}\\right] \\\\ \
\\end{array}\\right]\\\\ \
\\end{array}\\right]\\)"


	    //A11 (index 3) and A12 (index 4) are log struc, A21 (index 5) and A22 (index 6) are not log struc
	    if(matInfo[3].logstruc && matInfo[4].logstruc && !matInfo[5].logstruc && !matInfo[6].logstruc)
		return "\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
\\left[\\begin{array}{cc} \
\\left[\\begin{array}{cc} \
A_{1_{1_{1}}} & * \\\\ \
* & A_{1_{1_{2}}} \\\\ \
\\end{array}\\right] & * \\\\ \
* &\\left[\\begin{array}{cc} \
A_{1_{2_{1}}} & * \\\\ \
* & A_{1_{2_{2}}} \\\\ \
\\end{array}\\right] \\\\ \
\\end{array}\\right] & * \\\\ \
* & \\left[\\begin{array}{cc} \
A_{2_{1}} & * \\\\ \
* &  A_{2_{2}} \\\\ \
\\end{array}\\right]\\\\ \
\\end{array}\\right]\\)"
	    
	    //A11 (index 3) and A21 (index 5) are log struc, A12 (index 4) and A22 (index 6) are not log struc
	    if(matInfo[3].logstruc && matInfo[5].logstruc && !matInfo[4].logstruc && !matInfo[6].logstruc)
		return "\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
\\left[\\begin{array}{cc} \
\\left[\\begin{array}{cc} \
A_{1_{1_{1}}} & * \\\\ \
* & A_{1_{1_{2}}} \\\\ \
\\end{array}\\right] & * \\\\ \
* & A_{1_{2}} \\\\ \
\\end{array}\\right] & * \\\\ \
* & \\left[\\begin{array}{cc} \
\\left[\\begin{array}{cc} \
A_{2_{1_{1}}} & * \\\\ \
* & A_{2_{1_{2}}} \\\\ \
\\end{array}\\right] & * \\\\ \
* &  A_{2_{2}} \\\\ \
\\end{array}\\right]\\\\ \
\\end{array}\\right]\\)"



	    //A11 (index 3) and A22 (index 6) are log struc, A12 (index 4) and A21 (index 5) are not log struc
	    if(matInfo[3].logstruc && matInfo[6].logstruc && !matInfo[4].logstruc && !matInfo[5].logstruc)
		return "\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
\\left[\\begin{array}{cc} \
\\left[\\begin{array}{cc} \
A_{1_{1_{1}}} & * \\\\ \
* & A_{1_{1_{2}}} \\\\ \
\\end{array}\\right] & * \\\\ \
* & A_{1_{2}} \\\\ \
\\end{array}\\right] & * \\\\ \
* & \\left[\\begin{array}{cc} \
A_{2_{1}} &  * \\\\ \
* & \\left[\\begin{array}{cc} \
A_{2_{2_{1}}} & * \\\\ \
* & A_{2_{2_{2}}} \\\\ \
\\end{array}\\right] \\\\ \
\\end{array}\\right]\\\\ \
\\end{array}\\right]\\)"


	    //A12 (index 4) and A21 (index 5) are log struc, A11 (index 3) and A22 (index 6) are not log struc
	    if(matInfo[4].logstruc && matInfo[5].logstruc && !matInfo[3].logstruc && !matInfo[6].logstruc)
		return "\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
\\left[\\begin{array}{cc} \
A_{1_{1}} & * \\\\ \
* & \\left[\\begin{array}{cc} \
A_{1_{2_{1}}} & * \\\\ \
* & A_{1_{2_{2}}} \\\\ \
\\end{array}\\right] \\\\ \
\\end{array}\\right] & * \\\\ \
* & \\left[\\begin{array}{cc} \
\\left[\\begin{array}{cc} \
A_{2_{1_{1}}} & * \\\\ \
* & A_{2_{1_{2}}} \\\\ \
\\end{array}\\right] & * \\\\ \
* &  A_{2_{2}} \\\\ \
\\end{array}\\right]\\\\ \
\\end{array}\\right]\\)"

	    //A12 (index 4) and A22 (index 6) are log struc, A11 (index 3) and A21 (index 5) are not log struc
	    if(matInfo[4].logstruc && matInfo[6].logstruc && !matInfo[3].logstruc && !matInfo[5].logstruc)
		return "\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
\\left[\\begin{array}{cc} \
A_{1_{1}} & * \\\\ \
* & \\left[\\begin{array}{cc} \
A_{1_{2_{1}}} & * \\\\ \
* & A_{1_{2_{2}}} \\\\ \
\\end{array}\\right] \\\\ \
\\end{array}\\right] & * \\\\ \
* & \\left[\\begin{array}{cc} \
A_{2_{1}} &  * \\\\ \
* & \\left[\\begin{array}{cc} \
A_{2_{2_{1}}} & * \\\\ \
* & A_{2_{2_{2}}} \\\\ \
\\end{array}\\right] \\\\ \
\\end{array}\\right]\\\\ \
\\end{array}\\right]\\)"

	    //A11 (index 3), A12 (index 4), A21 (index 5) are log struc and A22 (index 6) is not log struc
	    if(matInfo[3].logstruc && matInfo[4].logstruc && matInfo[5].logstruc && !matInfo[6].logstruc)
		return "\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
\\left[\\begin{array}{cc} \
\\left[\\begin{array}{cc} \
A_{1_{1_{1}}} & * \\\\ \
* & A_{1_{1_{2}}} \\\\ \
\\end{array}\\right] & * \\\\ \
* &\\left[\\begin{array}{cc} \
A_{1_{2_{1}}} & * \\\\ \
* & A_{1_{2_{2}}} \\\\ \
\\end{array}\\right] \\\\ \
\\end{array}\\right] & * \\\\ \
* & \\left[\\begin{array}{cc} \
\\left[\\begin{array}{cc} \
A_{2_{1_{1}}} & * \\\\ \
* & A_{2_{1_{2}}} \\\\ \
\\end{array}\\right] & * \\\\ \
* &  A_{2_{2}} \\\\ \
\\end{array}\\right]\\\\ \
\\end{array}\\right]\\)"
	    



	    //A11 (index 3), A12 (index 4), A22 (index 6) are log struc and A21 (index 5) is not log struc
	    if(matInfo[3].logstruc && matInfo[4].logstruc && matInfo[6].logstruc && !matInfo[5].logstruc)
		return "\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
\\left[\\begin{array}{cc} \
\\left[\\begin{array}{cc} \
A_{1_{1_{1}}} & * \\\\ \
* & A_{1_{1_{2}}} \\\\ \
\\end{array}\\right] & * \\\\ \
* &\\left[\\begin{array}{cc} \
A_{1_{2_{1}}} & * \\\\ \
* & A_{1_{2_{2}}} \\\\ \
\\end{array}\\right] \\\\ \
\\end{array}\\right] & * \\\\ \
* & \\left[\\begin{array}{cc} \
A_{2_{1}} & * \\\\ \
* & \\left[\\begin{array}{cc} \
A_{2_{2_{1}}} & * \\\\ \
* & A_{2_{2_{2}}} \\\\ \
\\end{array}\\right] \\\\ \
\\end{array}\\right]\\\\ \
\\end{array}\\right]\\)"

	    //A11 (index 3), A21 (index 5), A22 (index 6) are log struc and A12 (index 4) is not log struc
	    if(matInfo[3].logstruc && matInfo[5].logstruc && matInfo[6].logstruc && !matInfo[4].logstruc)
		return "\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
\\left[\\begin{array}{cc} \
\\left[\\begin{array}{cc} \
A_{1_{1_{1}}} & * \\\\ \
* & A_{1_{1_{2}}} \\\\ \
\\end{array}\\right] & * \\\\ \
* & A_{1_{2}} \\\\ \
\\end{array}\\right] & * \\\\ \
* & \\left[\\begin{array}{cc} \
\\left[\\begin{array}{cc} \
A_{2_{1_{1}}} & * \\\\ \
* & A_{2_{1_{2}}} \\\\ \
\\end{array}\\right] & * \\\\ \
* & \\left[\\begin{array}{cc} \
A_{2_{2_{1}}} & * \\\\ \
* & A_{2_{2_{2}}} \\\\ \
\\end{array}\\right] \\\\ \
\\end{array}\\right]\\\\ \
\\end{array}\\right]\\)"


	    //A12 (index 4), A21 (index 5), A22 (index 6) are log struc and A11 (index 3) is not log struc
	    if(matInfo[4].logstruc && matInfo[5].logstruc && matInfo[6].logstruc && !matInfo[3].logstruc)
		return "\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
\\left[\\begin{array}{cc} \
A_{1_{1}} & * \\\\ \
* &\\left[\\begin{array}{cc} \
A_{1_{2_{1}}} & * \\\\ \
* & A_{1_{2_{2}}} \\\\ \
\\end{array}\\right] \\\\ \
\\end{array}\\right] & * \\\\ \
* & \\left[\\begin{array}{cc} \
\\left[\\begin{array}{cc} \
A_{2_{1_{1}}} & * \\\\ \
* & A_{2_{1_{2}}} \\\\ \
\\end{array}\\right] & * \\\\ \
* & \\left[\\begin{array}{cc} \
A_{2_{2_{1}}} & * \\\\ \
* & A_{2_{2_{2}}} \\\\ \
\\end{array}\\right] \\\\ \
\\end{array}\\right]\\\\ \
\\end{array}\\right]\\)"

	    //A11 (index 3), A12 (index 4), A21 (index 5), A22 (index 6) are all log struc
	    if(matInfo[3].logstruc && matInfo[4].logstruc && matInfo[5].logstruc && matInfo[6].logstruc)
		return "\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
\\left[\\begin{array}{cc} \
\\left[\\begin{array}{cc} \
A_{1_{1_{1}}} & * \\\\ \
* & A_{1_{1_{2}}} \\\\ \
\\end{array}\\right] & * \\\\ \
* &\\left[\\begin{array}{cc} \
A_{1_{2_{1}}} & * \\\\ \
* & A_{1_{2_{2}}} \\\\ \
\\end{array}\\right] \\\\ \
\\end{array}\\right] & * \\\\ \
* & \\left[\\begin{array}{cc} \
\\left[\\begin{array}{cc} \
A_{2_{1_{1}}} & * \\\\ \
* & A_{2_{1_{2}}} \\\\ \
\\end{array}\\right] & * \\\\ \
* & \\left[\\begin{array}{cc} \
A_{2_{2_{1}}} & * \\\\ \
* & A_{2_{2_{2}}} \\\\ \
\\end{array}\\right] \\\\ \
\\end{array}\\right]\\\\ \
\\end{array}\\right]\\)"
	    
	}
    }

}

