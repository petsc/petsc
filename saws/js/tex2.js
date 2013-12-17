//This array stores the tex, and MathJax interprets it on the fly. Each arrqy element refers to one
// specific matrix, highlighting and all.
function tex2(matInfo, recursionCounter)
{
    var tex

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
	return 	tex = "<h1>\\begin{bmatrix} \\color{red}{\\mathbf{A}} \\end{bmatrix}</h1>"

    //level 1 has two options (from highlighting)
    if(matGetLevel(recursionCounter) == 1)
    {
	if(recursionCounter == 1)
	    return "<h3>\\begin{bmatrix} \\color{red}{\\mathbf{A_{1}}} & * \\\\ * & A_{2} \\end{bmatrix}</h3>"
	
	if(recursionCounter == 2)
	    return "<h3>\\begin{bmatrix} A_{1} & * \\\\ * & \\color{red}{\\mathbf{A_{2}}} \\end{bmatrix}</h3>"

    }

    //level 2 has *many* options (from the various possibilities
    if(matGetLevel(recursionCounter) == 2)
    {
	//A1 (index 1) is logstruc and A2 is not logstruc
	if(matInfo[1].logstruc && !matInfo[2].logstruc)
	{
	    //switch on highligting	
	    //highlight A11 (index 3)
	    if(recursionCounter == 3)
		return "\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
\\left[\\begin{array}{cc} \
\\color{red}{\\mathbf{A_{1_{1}}}} & * \\\\ \
* & A_{1_{2}} \\\\ \
\\end{array}\\right] & * \\\\ \
* &  A_{2} \\\\ \
\\end{array}\\right]\\)"

	    //highlight A12 (index 4)
	    if(recursionCounter == 4)
		return "\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
\\left[\\begin{array}{cc} \
A_{1_{1}} & * \\\\ \
* & \\color{red}{\\mathbf{A_{1_{2}}}} \\\\ \
\\end{array}\\right] & * \\\\ \
* & A_{2} \\\\ \
\\end{array}\\right]\\)"

	}
	
	//A2 (index 2) is logstruc and A1 is not logstruc
	if(matInfo[2].logstruc && !matInfo[1].logstruc)
	{
	    //switch on highligting
	    //highligh A21 (index 5)
	    if(recursionCounter == 5)
		return "\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
A_{1} & * \\\\ \
* & \\left[\\begin{array}{cc} \
\\color{red}{\\mathbf{A_{2_{1}}}} & * \\\\ \
* & A_{2_{2}} \\\\ \
\\end{array}\\right] \\\\ \
\\end{array}\\right]\\)"
	    
	    //highlight A22 (index 6)
	    if(recursionCounter == 6)
		return "\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
A_{1} & * \\\\ \
* & \\left[\\begin{array}{cc} \
A_{2_{1}} & * \\\\ \
* & \\color{red}{\\mathbf{A_{2_{2}}}} \\\\ \
\\end{array}\\right] \\\\ \
\\end{array}\\right]\\)"
	}


	//A2 (index 2) is logstruc and A1 is also logstruc
	if(matInfo[2].logstruc && matInfo[1].logstruc)
	{
	    //switch on highligting
	    //highlight A11 (index 3)	
	    if(recursionCounter == 3)
		return "\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
\\left[\\begin{array}{cc} \
\\color{red}{\\mathbf{A_{1_{1}}}} & * \\\\ \
* & A_{1_{2}} \\\\ \
\\end{array}\\right] & * \\\\ \
* & \\left[\\begin{array}{cc} \
A_{2_{1}} & * \\\\ \
* & A_{2_{2}} \\\\ \
\\end{array}\\right] \\\\ \
\\end{array}\\right]\\)"
	    
	    //highlight A12 (index 4)
	    if(recursionCounter == 4)
		return 	"\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
\\left[\\begin{array}{cc} \
A_{1_{1}} & * \\\\ \
* & \\color{red}{\\mathbf{A_{1_{2}}}} \\\\ \
\\end{array}\\right] & * \\\\ \
* & \\left[\\begin{array}{cc} \
A_{2_{1}} & * \\\\ \
* & A_{2_{2}} \\\\ \
\\end{array}\\right] \\\\ \
\\end{array}\\right]\\)"

	    //highlight A21 (index 5)
	    if(recursionCounter == 5)
		return 	"\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
\\left[\\begin{array}{cc} \
A_{1_{1}} & * \\\\ \
* & A_{1_{2}} \\\\ \
\\end{array}\\right] & * \\\\ \
* & \\left[\\begin{array}{cc} \
\\color{red}{\\mathbf{A_{2_{1}}}} & * \\\\ \
* & A_{2_{2}} \\\\ \
\\end{array}\\right] \\\\ \
\\end{array}\\right]\\)"

	    //highlight A22 (index 6)
	    if(recursionCounter == 6)
		return "\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
\\left[\\begin{array}{cc} \
A_{1_{1}} & * \\\\ \
* & A_{1_{2}} \\\\ \
\\end{array}\\right] & * \\\\ \
* & \\left[\\begin{array}{cc} \
A_{2_{1}} & * \\\\ \
* & \\color{red}{\\mathbf{A_{2_{2}}}} \\\\ \
\\end{array}\\right] \\\\ \
\\end{array}\\right]\\)"
	}
    }



    //level 3 has *many* *many* options (from the various possibilities
    if(matGetLevel(recursionCounter) == 3)
    {
	//A1 (index 1) is logstruc and A2 is not logstruc
	if(matInfo[1].logstruc && !matInfo[2].logstruc)
	{
	    //A11 (index 3) is log struc and A12 (index 4) is not log struc
	    if(matInfo[3].logstruc && !matInfo[4].logstruc)
	    {
		//switch on highlighting
		//highlight A111 (index 7)
		if(recursionCounter == 7)
		    return "\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
\\left[\\begin{array}{cc} \
\\left[\\begin{array}{cc} \
\\color{red}{\\mathbf{A_{1_{1_{1}}}}} & * \\\\ \
* & A_{1_{1_{2}}} \\\\ \
\\end{array}\\right] & * \\\\ \
* &  A_{1_{2}} \\\\ \
\\end{array}\\right] & * \\\\ \
* &  A_{2} \\\\ \
\\end{array}\\right]\\)"
		
		//highligh A112 (index 8)
		if(recursionCounter == 8)
		    return "\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
\\left[\\begin{array}{cc} \
\\left[\\begin{array}{cc} \
A_{1_{1_{1}}} & * \\\\ \
* & \\color{red}{\\mathbf{A_{1_{1_{2}}}}} \\\\ \
\\end{array}\\right] & * \\\\ \
* &  A_{1_{2}} \\\\ \
\\end{array}\\right] & * \\\\ \
* &  A_{2} \\\\ \
\\end{array}\\right]\\)"
	    }


	    //A12 (index 4) is log struc and A11 (index 3) is not log struc
	    if(matInfo[4].logstruc && !matInfo[3].logstruc)
	    {

		//switch on highlighting
		//highlight A121 (index 9)
		if(recursionCounter == 9)
		    return "\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
\\left[\\begin{array}{cc} \
A_{1_{1}} &  * \\\\ \
* & \\left[\\begin{array}{cc} \
\\color{red}{\\mathbf{A_{1_{2_{1}}}}} & * \\\\ \
* & A_{1_{2_{2}}} \\\\ \
\\end{array}\\right] \\\\ \
\\end{array}\\right] & * \\\\ \
* &  A_{2} \\\\ \
\\end{array}\\right]\\)"

		//highligh A122 (index 10)
		if(recursionCounter == 10)
		    return "\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
\\left[\\begin{array}{cc} \
A_{1_{1}} &  * \\\\ \
* & \\left[\\begin{array}{cc} \
A_{1_{2_{1}}} & * \\\\ \
* & \\color{red}{\\mathbf{A_{1_{2_{2}}}}} \\\\ \
\\end{array}\\right] \\\\ \
\\end{array}\\right] & * \\\\ \
* &  A_{2} \\\\ \
\\end{array}\\right]\\)"

	    }

	    //A11 (index 3) is log struc and A12 (index 4) is also log struc
	    if(matInfo[3].logstruc && matInfo[4].logstruc)
	    {
		//switch on highlighting
		//highlight A111 (index 7)
		if(recursionCounter == 7)
		    return "\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
\\left[\\begin{array}{cc} \
\\left[\\begin{array}{cc} \
\\color{red}{\\mathbf{A_{1_{1_{1}}}}} & * \\\\ \
* & A_{1_{1_{2}}} \\\\ \
\\end{array}\\right] & * \\\\ \
* & \\left[\\begin{array}{cc} \
A_{1_{2_{1}}} & * \\\\ \
* & A_{1_{2_{2}}} \\\\ \
\\end{array}\\right] \\\\ \
\\end{array}\\right] & * \\\\ \
* &  A_{2} \\\\ \
\\end{array}\\right]\\)"

		//highlight A112 (index 8)
		if(recursionCounter == 8)
		    return "\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
\\left[\\begin{array}{cc} \
\\left[\\begin{array}{cc} \
A_{1_{1_{1}}} & * \\\\ \
* & \\color{red}{\\mathbf{A_{1_{1_{2}}}}} \\\\ \
\\end{array}\\right] & * \\\\ \
* & \\left[\\begin{array}{cc} \
A_{1_{2_{1}}} & * \\\\ \
* & A_{1_{2_{2}}} \\\\ \
\\end{array}\\right] \\\\ \
\\end{array}\\right] & * \\\\ \
* &  A_{2} \\\\ \
\\end{array}\\right]\\)"

		//highlight A121 (index 9)
		if(recursionCounter == 9)
		    return "\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
\\left[\\begin{array}{cc} \
\\left[\\begin{array}{cc} \
A_{1_{1_{1}}} & * \\\\ \
* & A_{1_{1_{2}}} \\\\ \
\\end{array}\\right] & * \\\\ \
* & \\left[\\begin{array}{cc} \
\\color{red}{\\mathbf{A_{1_{2_{1}}}}} & * \\\\ \
* & A_{1_{2_{2}}} \\\\ \
\\end{array}\\right] \\\\ \
\\end{array}\\right] & * \\\\ \
* &  A_{2} \\\\ \
\\end{array}\\right]\\)"

		//highlight A122 (index 10)
		if(recursionCounter == 10)
		    return "\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
\\left[\\begin{array}{cc} \
\\left[\\begin{array}{cc} \
A_{1_{1_{1}}} & * \\\\ \
* & A_{1_{1_{2}}} \\\\ \
\\end{array}\\right] & * \\\\ \
* & \\left[\\begin{array}{cc} \
A_{1_{2_{1}}} & * \\\\ \
* & \\color{red}{\\mathbf{A_{1_{2_{2}}}}} \\\\ \
\\end{array}\\right] \\\\ \
\\end{array}\\right] & * \\\\ \
* &  A_{2} \\\\ \
\\end{array}\\right]\\)"

	    }		
	}

	//A2 (index 2) is logstruc and A1 (index 1) is not logstruc
	if(matInfo[2].logstruc && !matInfo[1].logstruc)
	{
	    //A21 (index 5) is log struc and A22 (index 6) is not log struc
	    if(matInfo[5].logstruc && !matInfo[6].logstruc)
	    {
		//switch on highlighting
		//highlight A211 (index 11)
		if(recursionCounter == 11)
		    return "\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
A_{1} & * \\\\ \
* & \\left[\\begin{array}{cc} \
\\left[\\begin{array}{cc} \
\\color{red}{\\mathbf{A_{2_{1_{1}}}}} & * \\\\ \
* & A_{2_{1_{2}}} \\\\ \
\\end{array}\\right] & * \\\\ \
* &  A_{2_{2}} \\\\ \
\\end{array}\\right]\\\\ \
\\end{array}\\right]\\)"
		
		//highlight A212 (index 12)
		if(recursionCounter == 12)
		    return "\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
A_{1} & * \\\\ \
* & \\left[\\begin{array}{cc} \
\\left[\\begin{array}{cc} \
A_{2_{1_{1}}} & * \\\\ \
* & \\color{red}{\\mathbf{A_{2_{1_{2}}}}} \\\\ \
\\end{array}\\right] & * \\\\ \
* &  A_{2_{2}} \\\\ \
\\end{array}\\right]\\\\ \
\\end{array}\\right]\\)"
	    }


	    //A22 (index 6) is log struc and A21 (index 5) is not log struc
	    if(matInfo[6].logstruc && !matInfo[5].logstruc)
	    {

		//switch on highlighting	
		//highligh A221 (index 13)
		if(recursionCounter == 13)
		    return "\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
A_{1} & * \\\\ \
* & \\left[\\begin{array}{cc} \
A_{2_{1}} &  * \\\\ \
* & \\left[\\begin{array}{cc} \
\\color{red}{\\mathbf{A_{2_{2_{1}}}}} & * \\\\ \
* & A_{2_{2_{2}}} \\\\ \
\\end{array}\\right] \\\\ \
\\end{array}\\right]\\\\ \
\\end{array}\\right]\\)"
		
		//highligh A222 (index 14)
		if(recursionCounter == 14)
		    return "\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
A_{1} & * \\\\ \
* & \\left[\\begin{array}{cc} \
A_{2_{1}} &  * \\\\ \
* & \\left[\\begin{array}{cc} \
A_{2_{2_{1}}} & * \\\\ \
* & \\color{red}{\\mathbf{A_{2_{2_{2}}}}} \\\\ \
\\end{array}\\right] \\\\ \
\\end{array}\\right]\\\\ \
\\end{array}\\right]\\)"
		

	    }

	    //A21 (index 5) is log struc and A22 (index 6) is also log struc
	    if(matInfo[5].logstruc && matInfo[6].logstruc)
	    {
		//switch on highlighting
		//highlight A211 (index 11)
		if(recursionCounter == 11)
		    return "\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
A_{1} & * \\\\ \
* & \\left[\\begin{array}{cc} \
\\left[\\begin{array}{cc} \
\\color{red}{\\mathbf{A_{2_{1_{1}}}}} & * \\\\ \
* & A_{2_{1_{2}}} \\\\ \
\\end{array}\\right] & * \\\\ \
* &\\left[\\begin{array}{cc} \
A_{2_{2_{1}}} & * \\\\ \
* & A_{2_{2_{2}}} \\\\ \
\\end{array}\\right] \\\\ \
\\end{array}\\right]\\\\ \
\\end{array}\\right]\\)"

		//highlight A212 (index 12)
		if(recursionCounter == 12)
		    return "\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
A_{1} & * \\\\ \
* & \\left[\\begin{array}{cc} \
\\left[\\begin{array}{cc} \
A_{2_{1_{1}}} & * \\\\ \
* & \\color{red}{\\mathbf{A_{2_{1_{2}}}}} \\\\ \
\\end{array}\\right] & * \\\\ \
* &\\left[\\begin{array}{cc} \
A_{2_{2_{1}}} & * \\\\ \
* & A_{2_{2_{2}}} \\\\ \
\\end{array}\\right] \\\\ \
\\end{array}\\right]\\\\ \
\\end{array}\\right]\\)"

		//highlight A221 (index 13)
		if(recursionCounter == 13)
		    return "\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
A_{1} & * \\\\ \
* & \\left[\\begin{array}{cc} \
\\left[\\begin{array}{cc} \
A_{2_{1_{1}}} & * \\\\ \
* & A_{2_{1_{2}}} \\\\ \
\\end{array}\\right] & * \\\\ \
* &\\left[\\begin{array}{cc} \
\\color{red}{\\mathbf{A_{2_{2_{1}}}}} & * \\\\ \
* & A_{2_{2_{2}}} \\\\ \
\\end{array}\\right] \\\\ \
\\end{array}\\right]\\\\ \
\\end{array}\\right]\\)"

		//highlight A222 (index 14)
		if(recursionCounter == 14)
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
* & \\color{red}{\\mathbf{A_{2_{2_{2}}}}} \\\\ \
\\end{array}\\right] \\\\ \
\\end{array}\\right]\\\\ \
\\end{array}\\right]\\)"

	    }
	}

	//A2 (index 2) is logstruc and A1 (index 1) is also logstruc
	if(matInfo[2].logstruc && matInfo[1].logstruc)
	{
	    
	    //A11 (index 3) is log struc and A12 (index 4), A21 (index 5), and A22 (index 6) are not log struc
	    if(matInfo[3].logstruc && !matInfo[4].logstruc && !matInfo[5].logstruc && !matInfo[6].logstruc)
	    {
		//switch on highlighting
		//highlight A111 (index 7)
		if(recursionCounter == 7)
		    return "\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
\\left[\\begin{array}{cc} \
\\left[\\begin{array}{cc} \
\\color{red}{\\mathbf{A_{1_{1_{1}}}}} & * \\\\ \
* & A_{1_{1_{2}}} \\\\ \
\\end{array}\\right] & * \\\\ \
* &  A_{1_{2}} \\\\ \
\\end{array}\\right] & * \\\\ \
* & \\left[\\begin{array}{cc} \
A_{2_{1}} & * \\\\ \
* &  A_{2_{2}} \\\\ \
\\end{array}\\right]\\\\ \
\\end{array}\\right]\\)"
		
		//highlight A112 (index 8)
		if(recursionCounter == 8)
		    return "\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
\\left[\\begin{array}{cc} \
\\left[\\begin{array}{cc} \
A_{1_{1_{1}}} & * \\\\ \
* & \\color{red}{\\mathbf{A_{1_{1_{2}}}}} \\\\ \
\\end{array}\\right] & * \\\\ \
* &  A_{1_{2}} \\\\ \
\\end{array}\\right] & * \\\\ \
* & \\left[\\begin{array}{cc} \
A_{2_{1}} & * \\\\ \
* &  A_{2_{2}} \\\\ \
\\end{array}\\right]\\\\ \
\\end{array}\\right]\\)"
	    }


	    //A12 (index 4) is log struc and A11 (index 3), A21 (index 5), and A22 (index 6) are not log struc
	    if(matInfo[4].logstruc && !matInfo[3].logstruc && !matInfo[5].logstruc && !matInfo[6].logstruc)
	    {
		//switch on highlighting
		//highlight A121 (index 9)
		if(recursionCounter == 9)
		    return "\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
\\left[\\begin{array}{cc} \
A_{1_{1}} &  * \\\\ \
* & \\left[\\begin{array}{cc} \
\\color{red}{\\mathbf{A_{1_{2{_1}}}}} & * \\\\ \
* & A_{1_{2_{2}}} \\\\ \
\\end{array}\\right]\\\\ \
\\end{array}\\right] & * \\\\ \
* & \\left[\\begin{array}{cc} \
A_{2_{1}} & * \\\\ \
* &  A_{2_{2}} \\\\ \
\\end{array}\\right]\\\\ \
\\end{array}\\right]\\)"
		
		//highlight A122 (index 10)
		if(recursionCounter == 10)
		    return "\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
\\left[\\begin{array}{cc} \
A_{1_{1}} &  * \\\\ \
* & \\left[\\begin{array}{cc} \
A_{1_{2{_1}}} & * \\\\ \
* & \\color{red}{\\mathbf{A_{1_{2_{2}}}}} \\\\ \
\\end{array}\\right]\\\\ \
\\end{array}\\right] & * \\\\ \
* & \\left[\\begin{array}{cc} \
A_{2_{1}} & * \\\\ \
* &  A_{2_{2}} \\\\ \
\\end{array}\\right]\\\\ \
\\end{array}\\right]\\)"
	    }

	    //A21 (index 5) is log struc and A11 (index 3), A12 (index 4), and A22 (index 6) are not log struc
	    if(matInfo[5].logstruc && !matInfo[4].logstruc && !matInfo[3].logstruc && !matInfo[6].logstruc)
	    {
		//switch on highlighting
		//highlight A211 (index 11)
		if(recursionCounter == 11)
		    return "\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
\\left[\\begin{array}{cc} \
A_{1_{1}} & * \\\\ \
* &  A_{1_{2}} \\\\ \
\\end{array}\\right] & * \\\\ \
* & \\left[\\begin{array}{cc} \
\\left[\\begin{array}{cc} \
\\color{red}{\\mathbf{A_{2_{1_{1}}}}} & * \\\\ \
* & A_{2_{1_{2}}} \\\\ \
\\end{array}\\right] & * \\\\ \
* &  A_{2_{2}} \\\\ \
\\end{array}\\right]\\\\ \
\\end{array}\\right]\\)"
		
		//highlight A212 (index 12)
		if(recursionCounter == 12)
		    return "\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
\\left[\\begin{array}{cc} \
A_{1_{1}} & * \\\\ \
* &  A_{1_{2}} \\\\ \
\\end{array}\\right] & * \\\\ \
* & \\left[\\begin{array}{cc} \
\\left[\\begin{array}{cc} \
A_{2_{1_{1}}} & * \\\\ \
* & \\color{red}{\\mathbf{A_{2_{1_{2}}}}} \\\\ \
\\end{array}\\right] & * \\\\ \
* &  A_{2_{2}} \\\\ \
\\end{array}\\right]\\\\ \
\\end{array}\\right]\\)"
	    }

	    //A22 (index 6) is log struc and A11 (index 3), A12 (index 4), and A22 (index 6) are not log struc
	    if(matInfo[6].logstruc && !matInfo[3].logstruc && !matInfo[5].logstruc && !matInfo[4].logstruc)
	    {
		//switch on highlighting
		//highlight A221 (index 13)
		if(recursionCounter == 13)
		    return "\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
\\left[\\begin{array}{cc} \
A_{1_{1}} & * \\\\ \
* &  A_{1_{2}} \\\\ \
\\end{array}\\right] & * \\\\ \
* & \\left[\\begin{array}{cc} \
A_{2_{1}} &  * \\\\ \
* & \\left[\\begin{array}{cc} \
\\color{red}{\\mathbf{A_{2_{2_{1}}}}} & * \\\\ \
* & A_{2_{2_{2}}} \\\\ \
\\end{array}\\right] \\\\ \
\\end{array}\\right]\\\\ \
\\end{array}\\right]\\)"
		
		//highlight A222 (index 14)
		if(recursionCounter == 14)
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
* & \\color{red}{\\mathbf{A_{2_{2_{2}}}}} \\\\ \
\\end{array}\\right] \\\\ \
\\end{array}\\right]\\\\ \
\\end{array}\\right]\\)"
	    }


	    //A11 (index 3) and A12 (index 4) are log struc, A21 (index 5) and A22 (index 6) are not log struc
	    if(matInfo[3].logstruc && matInfo[4].logstruc && !matInfo[5].logstruc && !matInfo[6].logstruc)
	    {
		//switch on highlighting
		//highlight A111 (index 7)
		if(recursionCounter == 7)
		    return "\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
\\left[\\begin{array}{cc} \
\\left[\\begin{array}{cc} \
\\color{red}{\\mathbf{A_{1_{1_{1}}}}} & * \\\\ \
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
		
		//highlight A112 (index 8)
		if(recursionCounter == 8)
		    return "\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
\\left[\\begin{array}{cc} \
\\left[\\begin{array}{cc} \
A_{1_{1_{1}}} & * \\\\ \
* & \\color{red}{\\mathbf{A_{1_{1_{2}}}}} \\\\ \
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
		
		//highlight A121 (index 9)
		if(recursionCounter == 9)
		    return "\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
\\left[\\begin{array}{cc} \
\\left[\\begin{array}{cc} \
A_{1_{1_{1}}} & * \\\\ \
* & A_{1_{1_{2}}} \\\\ \
\\end{array}\\right] & * \\\\ \
* &\\left[\\begin{array}{cc} \
\\color{red}{\\mathbf{A_{1_{2_{1}}}}} & * \\\\ \
* & A_{1_{2_{2}}} \\\\ \
\\end{array}\\right] \\\\ \
\\end{array}\\right] & * \\\\ \
* & \\left[\\begin{array}{cc} \
A_{2_{1}} & * \\\\ \
* &  A_{2_{2}} \\\\ \
\\end{array}\\right]\\\\ \
\\end{array}\\right]\\)"

		//highlight A122 (index 10)
		if(recursionCounter == 10)
		    return "\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
\\left[\\begin{array}{cc} \
\\left[\\begin{array}{cc} \
A_{1_{1_{1}}} & * \\\\ \
* & A_{1_{1_{2}}} \\\\ \
\\end{array}\\right] & * \\\\ \
* &\\left[\\begin{array}{cc} \
A_{1_{2_{1}}} & * \\\\ \
* & \\color{red}{\\mathbf{A_{1_{2_{2}}}}} \\\\ \
\\end{array}\\right] \\\\ \
\\end{array}\\right] & * \\\\ \
* & \\left[\\begin{array}{cc} \
A_{2_{1}} & * \\\\ \
* &  A_{2_{2}} \\\\ \
\\end{array}\\right]\\\\ \
\\end{array}\\right]\\)"

	    }

	    //A11 (index 3) and A21 (index 5) are log struc, A12 (index 4) and A22 (index 6) are not log struc
	    if(matInfo[3].logstruc && matInfo[5].logstruc && !matInfo[4].logstruc && !matInfo[6].logstruc)
	    {
		//switch on highlighting
		//highlight A111 (index 7)
		if(recursionCounter == 7)
		    return "\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
\\left[\\begin{array}{cc} \
\\left[\\begin{array}{cc} \
\\color{red}{\\mathbf{A_{1_{1_{1}}}}} & * \\\\ \
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

		//highlight A121 (index 8)
		if(recursionCounter == 8)
		    return "\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
\\left[\\begin{array}{cc} \
\\left[\\begin{array}{cc} \
A_{1_{1_{1}}} & * \\\\ \
* & \\color{red}{\\mathbf{A_{1_{1_{2}}}}} \\\\ \
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

		//highlight A211 (index 11)
		if(recursionCounter == 11)
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
\\color{red}{\\mathbf{A_{2_{1_{1}}}}} & * \\\\ \
* & A_{2_{1_{2}}} \\\\ \
\\end{array}\\right] & * \\\\ \
* &  A_{2_{2}} \\\\ \
\\end{array}\\right]\\\\ \
\\end{array}\\right]\\)"


		//highlight A212 (index 12)
		if(recursionCounter == 12)
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
* & \\color{red}{\\mathbf{A_{2_{1_{2}}}}} \\\\ \
\\end{array}\\right] & * \\\\ \
* &  A_{2_{2}} \\\\ \
\\end{array}\\right]\\\\ \
\\end{array}\\right]\\)"

		
	    }

	    //A11 (index 3) and A22 (index 6) are log struc, A12 (index 4) and A21 (index 5) are not log struc
	    if(matInfo[3].logstruc && matInfo[6].logstruc && !matInfo[4].logstruc && !matInfo[5].logstruc)
	    {
		//switch on highlighting
		//highlight A111 (index 7)
		if(recursionCounter == 7)
		    return "\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
\\left[\\begin{array}{cc} \
\\left[\\begin{array}{cc} \
\\color{red}{\\mathbf{A_{1_{1_{1}}}}} & * \\\\ \
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


		//highlight A112 (index 8)
		if(recursionCounter == 8)
		    return "\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
\\left[\\begin{array}{cc} \
\\left[\\begin{array}{cc} \
A_{1_{1_{1}}} & * \\\\ \
* & \\color{red}{\\mathbf{A_{1_{1_{2}}}}} \\\\ \
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

		//highlight A221 (index 13)
		if(recursionCounter == 13)
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
\\color{red}{\\mathbf{A_{2_{2_{1}}}}} & * \\\\ \
* & A_{2_{2_{2}}} \\\\ \
\\end{array}\\right] \\\\ \
\\end{array}\\right]\\\\ \
\\end{array}\\right]\\)"


		//highlight A222 (index 14)
		if(recursionCounter == 14)
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
* & \\color{red}{\\mathbf{A_{2_{2_{2}}}}} \\\\ \
\\end{array}\\right] \\\\ \
\\end{array}\\right]\\\\ \
\\end{array}\\right]\\)"


	    }


	    //A12 (index 4) and A21 (index 5) are log struc, A11 (index 3) and A22 (index 6) are not log struc
	    if(matInfo[4].logstruc && matInfo[5].logstruc && !matInfo[3].logstruc && !matInfo[6].logstruc)
	    {
		//switch on highlighting
		//highlight A121 (index 9)
		if(recursionCounter == 9)
		    return "\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
\\left[\\begin{array}{cc} \
A_{1_{1}} & * \\\\ \
* & \\left[\\begin{array}{cc} \
\\color{red}{\\mathbf{A_{1_{2_{1}}}}} & * \\\\ \
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

		//highlight A122 (index 10)
		if(recursionCounter == 10)
		    return "\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
\\left[\\begin{array}{cc} \
A_{1_{1}} & * \\\\ \
* & \\left[\\begin{array}{cc} \
A_{1_{2_{1}}} & * \\\\ \
* & \\color{red}{\\mathbf{A_{1_{2_{2}}}}} \\\\ \
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

		//highlight A211 (index 11)
		if(recursionCounter == 11)
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
\\color{red}{\\mathbf{A_{2_{1_{1}}}}} & * \\\\ \
* & A_{2_{1_{2}}} \\\\ \
\\end{array}\\right] & * \\\\ \
* &  A_{2_{2}} \\\\ \
\\end{array}\\right]\\\\ \
\\end{array}\\right]\\)"

		//highlight A212 (index 12)
		if(recursionCounter == 12)
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
* & \\color{red}{\\mathbf{A_{2_{1_{2}}}}} \\\\ \
\\end{array}\\right] & * \\\\ \
* &  A_{2_{2}} \\\\ \
\\end{array}\\right]\\\\ \
\\end{array}\\right]\\)"
		



	    }

	    //A12 (index 4) and A22 (index 6) are log struc, A11 (index 3) and A21 (index 5) are not log struc
	    if(matInfo[4].logstruc && matInfo[6].logstruc && !matInfo[3].logstruc && !matInfo[5].logstruc)
	    {
		//switch on highlighting
		//highlight A121 (index 9)
		if(recursionCounter == 9)
		    return "\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
\\left[\\begin{array}{cc} \
A_{1_{1}} & * \\\\ \
* & \\left[\\begin{array}{cc} \
\\color{red}{\\mathbf{A_{1_{2_{1}}}}} & * \\\\ \
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

		//highlight A121 (index 10)
		if(recursionCounter == 10)
		    return "\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
\\left[\\begin{array}{cc} \
A_{1_{1}} & * \\\\ \
* & \\left[\\begin{array}{cc} \
A_{1_{2_{1}}} & * \\\\ \
* & \\color{red}{\\mathbf{A_{1_{2_{2}}}}} \\\\ \
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

		//highlight A221 (index 13)
		if(recursionCounter == 13)
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
\\color{red}{\\mathbf{A_{2_{2_{1}}}}} & * \\\\ \
* & A_{2_{2_{2}}} \\\\ \
\\end{array}\\right] \\\\ \
\\end{array}\\right]\\\\ \
\\end{array}\\right]\\)"

		//highlight A222 (index 14)
		if(recursionCounter == 14)
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
* & \\color{red}{\\mathbf{A_{2_{2_{2}}}}} \\\\ \
\\end{array}\\right] \\\\ \
\\end{array}\\right]\\\\ \
\\end{array}\\right]\\)"

	    }


	    //A11 (index 3), A12 (index 4), A21 (index 5) are log struc and A22 (index 6) is not log struc
	    if(matInfo[3].logstruc && matInfo[4].logstruc && matInfo[5].logstruc && !matInfo[6].logstruc)
	    {
		//switch on highlighting
		//highlight A111 (index 7)
		if(recursionCounter == 7)
		    return "\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
\\left[\\begin{array}{cc} \
\\left[\\begin{array}{cc} \
\\color{red}{\\mathbf{A_{1_{1_{1}}}}} & * \\\\ \
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

		//highlight A112 (index 8)
		if(recursionCounter == 8)
		    return "\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
\\left[\\begin{array}{cc} \
\\left[\\begin{array}{cc} \
A_{1_{1_{1}}} & * \\\\ \
* & \\color{red}{\\mathbf{A_{1_{1_{2}}}}} \\\\ \
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

		//highlight A121 (index 9)
		if(recursionCounter == 9)
		    return "\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
\\left[\\begin{array}{cc} \
\\left[\\begin{array}{cc} \
A_{1_{1_{1}}} & * \\\\ \
* & A_{1_{1_{2}}} \\\\ \
\\end{array}\\right] & * \\\\ \
* &\\left[\\begin{array}{cc} \
\\color{red}{\\mathbf{A_{1_{2_{1}}}}} & * \\\\ \
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

		//highlight A122 (index 10)
		if(recursionCounter == 10)
		    return "\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
\\left[\\begin{array}{cc} \
\\left[\\begin{array}{cc} \
A_{1_{1_{1}}} & * \\\\ \
* & A_{1_{1_{2}}} \\\\ \
\\end{array}\\right] & * \\\\ \
* &\\left[\\begin{array}{cc} \
A_{1_{2_{1}}} & * \\\\ \
* & \\color{red}{\\mathbf{A_{1_{2_{2}}}}} \\\\ \
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

		//highlight A211 (index 11)
		if(recursionCounter == 11)
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
\\color{red}{\\mathbf{A_{2_{1_{1}}}}} & * \\\\ \
* & A_{2_{1_{2}}} \\\\ \
\\end{array}\\right] & * \\\\ \
* &  A_{2_{2}} \\\\ \
\\end{array}\\right]\\\\ \
\\end{array}\\right]\\)"

		//highlight A212 (index 12)
		if(recursionCounter == 12)
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
* & \\color{red}{\\mathbf{A_{2_{1_{2}}}}} \\\\ \
\\end{array}\\right] & * \\\\ \
* &  A_{2_{2}} \\\\ \
\\end{array}\\right]\\\\ \
\\end{array}\\right]\\)"

		
	    }

	    //A11 (index 3), A12 (index 4), A22 (index 6) are log struc and A21 (index 5) is not log struc
	    if(matInfo[3].logstruc && matInfo[4].logstruc && matInfo[6].logstruc && !matInfo[5].logstruc)
	    {
		//switch on highlighting
		//highlight A111 (index 7)
		if(recursionCounter == 7)
		    return "\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
\\left[\\begin{array}{cc} \
\\left[\\begin{array}{cc} \
\\color{red}{\\mathbf{A_{1_{1_{1}}}}} & * \\\\ \
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

		//highlight A112 (index 8)
		if(recursionCounter == 8)
		    return "\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
\\left[\\begin{array}{cc} \
\\left[\\begin{array}{cc} \
A_{1_{1_{1}}} & * \\\\ \
* & \\color{red}{\\mathbf{A_{1_{1_{2}}}}} \\\\ \
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

		//highlight A121 (index 9)
		if(recursionCounter == 9)
		    return "\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
\\left[\\begin{array}{cc} \
\\left[\\begin{array}{cc} \
A_{1_{1_{1}}} & * \\\\ \
* & A_{1_{1_{2}}} \\\\ \
\\end{array}\\right] & * \\\\ \
* &\\left[\\begin{array}{cc} \
\\color{red}{\\mathbf{A_{1_{2_{1}}}}} & * \\\\ \
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

		//highlight A122 (index 10)
		if(recursionCounter == 10)
		    return "\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
\\left[\\begin{array}{cc} \
\\left[\\begin{array}{cc} \
A_{1_{1_{1}}} & * \\\\ \
* & A_{1_{1_{2}}} \\\\ \
\\end{array}\\right] & * \\\\ \
* &\\left[\\begin{array}{cc} \
A_{1_{2_{1}}} & * \\\\ \
* & \\color{red}{\\mathbf{A_{1_{2_{2}}}}} \\\\ \
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

		//highlight A221 (index 13)
		if(recursionCounter == 13)
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
\\color{red}{\\mathbf{A_{2_{2_{1}}}}} & * \\\\ \
* & A_{2_{2_{2}}} \\\\ \
\\end{array}\\right] \\\\ \
\\end{array}\\right]\\\\ \
\\end{array}\\right]\\)"

		//highlight A222 (index 14)
		if(recursionCounter == 14)
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
* & \\color{red}{\\mathbf{A_{2_{2_{2}}}}} \\\\ \
\\end{array}\\right] \\\\ \
\\end{array}\\right]\\\\ \
\\end{array}\\right]\\)"


	    }

	    //A11 (index 3), A21 (index 5), A22 (index 6) are log struc and A12 (index 4) is not log struc
	    if(matInfo[3].logstruc && matInfo[5].logstruc && matInfo[6].logstruc && !matInfo[4].logstruc)
	    {
		//switch on highlighting
		//highlight A111 (index 7)
		if(recursionCounter == 7)
		    return "\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
\\left[\\begin{array}{cc} \
\\left[\\begin{array}{cc} \
\\color{red}{\\mathbf{A_{1_{1_{1}}}}} & * \\\\ \
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

		//highlight A112 (index 8)
		if(recursionCounter == 8)
		    return "\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
\\left[\\begin{array}{cc} \
\\left[\\begin{array}{cc} \
A_{1_{1_{1}}} & * \\\\ \
* & \\color{red}{\\mathbf{A_{1_{1_{2}}}}} \\\\ \
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

		//highlight A211 (index 11)
		if(recursionCounter == 11)
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
\\color{red}{\\mathbf{A_{2_{1_{1}}}}} & * \\\\ \
* & A_{2_{1_{2}}} \\\\ \
\\end{array}\\right] & * \\\\ \
* & \\left[\\begin{array}{cc} \
A_{2_{2_{1}}} & * \\\\ \
* & A_{2_{2_{2}}} \\\\ \
\\end{array}\\right] \\\\ \
\\end{array}\\right]\\\\ \
\\end{array}\\right]\\)"

		//highlight A212 (index 12)
		if(recursionCounter == 12)
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
* & \\color{red}{\\mathbf{A_{2_{1_{2}}}}} \\\\ \
\\end{array}\\right] & * \\\\ \
* & \\left[\\begin{array}{cc} \
A_{2_{2_{1}}} & * \\\\ \
* & A_{2_{2_{2}}} \\\\ \
\\end{array}\\right] \\\\ \
\\end{array}\\right]\\\\ \
\\end{array}\\right]\\)"

		//highlight A221 (index 13)
		if(recursionCounter == 13)
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
\\color{red}{\\mathbf{A_{2_{2_{1}}}}} & * \\\\ \
* & A_{2_{2_{2}}} \\\\ \
\\end{array}\\right] \\\\ \
\\end{array}\\right]\\\\ \
\\end{array}\\right]\\)"

		//highlight A222 (index 14)
		if(recursionCounter == 14)
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
* & \\color{red}{\\mathbf{A_{2_{2_{2}}}}} \\\\ \
\\end{array}\\right] \\\\ \
\\end{array}\\right]\\\\ \
\\end{array}\\right]\\)"



	    }

	    //A12 (index 4), A21 (index 5), A22 (index 6) are log struc and A11 (index 3) is not log struc
	    if(matInfo[4].logstruc && matInfo[5].logstruc && matInfo[6].logstruc && !matInfo[3].logstruc)
	    {
		//switch on highlighting
		//highlight A121 (index 9)
		if(recursionCounter == 9)
		    return "\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
\\left[\\begin{array}{cc} \
A_{1_{1}} & * \\\\ \
* &\\left[\\begin{array}{cc} \
\\color{red}{\\mathbf{A_{1_{2_{1}}}}} & * \\\\ \
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

		//highlight A122 (index 10)
		if(recursionCounter == 10)
		    return "\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
\\left[\\begin{array}{cc} \
A_{1_{1}} & * \\\\ \
* &\\left[\\begin{array}{cc} \
A_{1_{2_{1}}} & * \\\\ \
* & \\color{red}{\\mathbf{A_{1_{2_{2}}}}} \\\\ \
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

		//highlight A211 (index 11)
		if(recursionCounter == 11)
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
\\color{red}{\\mathbf{A_{2_{1_{1}}}}} & * \\\\ \
* & A_{2_{1_{2}}} \\\\ \
\\end{array}\\right] & * \\\\ \
* & \\left[\\begin{array}{cc} \
A_{2_{2_{1}}} & * \\\\ \
* & A_{2_{2_{2}}} \\\\ \
\\end{array}\\right] \\\\ \
\\end{array}\\right]\\\\ \
\\end{array}\\right]\\)"

		//highlight A212 (index 12)
		if(recursionCounter == 12)
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
* & \\color{red}{\\mathbf{A_{2_{1_{2}}}}} \\\\ \
\\end{array}\\right] & * \\\\ \
* & \\left[\\begin{array}{cc} \
A_{2_{2_{1}}} & * \\\\ \
* & A_{2_{2_{2}}} \\\\ \
\\end{array}\\right] \\\\ \
\\end{array}\\right]\\\\ \
\\end{array}\\right]\\)"

		//highlight A221 (index 13)
		if(recursionCounter == 13)
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
\\color{red}{\\mathbf{A_{2_{2_{1}}}}} & * \\\\ \
* & A_{2_{2_{2}}} \\\\ \
\\end{array}\\right] \\\\ \
\\end{array}\\right]\\\\ \
\\end{array}\\right]\\)"

		//highlight A222 (index 14)
		if(recursionCounter == 14)
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
* & \\color{red}{\\mathbf{A_{2_{2_{2}}}}} \\\\ \
\\end{array}\\right] \\\\ \
\\end{array}\\right]\\\\ \
\\end{array}\\right]\\)"

	    }

	    //A11 (index 3), A12 (index 4), A21 (index 5), A22 (index 6) are all log struc
	    if(matInfo[3].logstruc && matInfo[4].logstruc && matInfo[5].logstruc && matInfo[6].logstruc)
	    {
		//switch on highlighting
		//highlight A111 (index 7)
		if(recursionCounter == 7)
		    return "\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
\\left[\\begin{array}{cc} \
\\left[\\begin{array}{cc} \
\\color{red}{\\mathbf{A_{1_{1_{1}}}}} & * \\\\ \
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

		//highlight A112 (index 8)
		if(recursionCounter == 8)
		    return "\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
\\left[\\begin{array}{cc} \
\\left[\\begin{array}{cc} \
A_{1_{1_{1}}} & * \\\\ \
* & \\color{red}{\\mathbf{A_{1_{1_{2}}}}} \\\\ \
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

		//highlight A121 (index 9)
		if(recursionCounter == 9)
		    return "\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
\\left[\\begin{array}{cc} \
\\left[\\begin{array}{cc} \
A_{1_{1_{1}}} & * \\\\ \
* & A_{1_{1_{2}}} \\\\ \
\\end{array}\\right] & * \\\\ \
* &\\left[\\begin{array}{cc} \
\\color{red}{\\mathbf{A_{1_{2_{1}}}}} & * \\\\ \
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

		//highlight A122 (index 10)
		if(recursionCounter == 10)
		    return "\\(\\left[ \
\\begin{array}{c@{}c@{}c} \
\\left[\\begin{array}{cc} \
\\left[\\begin{array}{cc} \
A_{1_{1_{1}}} & * \\\\ \
* & A_{1_{1_{2}}} \\\\ \
\\end{array}\\right] & * \\\\ \
* &\\left[\\begin{array}{cc} \
A_{1_{2_{1}}} & * \\\\ \
* & \\color{red}{\\mathbf{A_{1_{2_{2}}}}} \\\\ \
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

		//highlight A211 (index 11)
		if(recursionCounter == 11)
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
\\color{red}{\\mathbf{A_{2_{1_{1}}}}} & * \\\\ \
* & A_{2_{1_{2}}} \\\\ \
\\end{array}\\right] & * \\\\ \
* & \\left[\\begin{array}{cc} \
A_{2_{2_{1}}} & * \\\\ \
* & A_{2_{2_{2}}} \\\\ \
\\end{array}\\right] \\\\ \
\\end{array}\\right]\\\\ \
\\end{array}\\right]\\)"

		//highlight A212 (index 12)
		if(recursionCounter == 12)
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
* & \\color{red}{\\mathbf{A_{2_{1_{2}}}}} \\\\ \
\\end{array}\\right] & * \\\\ \
* & \\left[\\begin{array}{cc} \
A_{2_{2_{1}}} & * \\\\ \
* & A_{2_{2_{2}}} \\\\ \
\\end{array}\\right] \\\\ \
\\end{array}\\right]\\\\ \
\\end{array}\\right]\\)"

		//highlight A221 (index 13)
		if(recursionCounter == 13)
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
\\color{red}{\\mathbf{A_{2_{2_{1}}}}} & * \\\\ \
* & A_{2_{2_{2}}} \\\\ \
\\end{array}\\right] \\\\ \
\\end{array}\\right]\\\\ \
\\end{array}\\right]\\)"

		//highlight A222 (index 14)
		if(recursionCounter == 14)
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
* & \\color{red}{\\mathbf{A_{2_{2_{2}}}}} \\\\ \
\\end{array}\\right] \\\\ \
\\end{array}\\right]\\\\ \
\\end{array}\\right]\\)"

	    }


	}	

    }
}
