/*
 * box-algorithm that I wrote for drawing trees with different node sizes and splitting in different directions
 *
 * The following aesthetic criteria are satisfied:
 *
 * 1) No overlapping between boxes
 * 2) When switching directions, the entire sub-tree in direction 2 should be seen as one of the nodes in direction 1 (the original direction)
 * 3) Children of the same depth are shown on the same line (not implemented yet)
 * 4) Child nodes are spaced according to the size of their subtree such that the subtrees never overlap at any point
 * 5) Parent is centered at the middle of all the subtrees of the children. (can be adjusted)
 *
 */

//generates svg code for the given input parameters
//the x, y coordinates are the upper left hand coordinate of the drawing. should start with (0,0)
function getBoxTree(data, endtag, x, y) {

    var ret         = ""; //the svg code to return
    var index       = getIndex(data,endtag);
    var numChildren = getNumChildren(data,endtag);
    var pc_type     = data[index].pc_type;

    var total_size = data[index].total_size;
    var node_size  = data[index].node_size;
    var text_size  = getTextSize(data,endtag);

    //draw the node itself (centering it properly)
    //this centering algorithm can be easily changed
    var centered_x = x;
    var centered_y = y;
    if(pc_type == "mg") {
        centered_y = y + .5*total_size.height - .5*text_size.height;
    }
    else {
        centered_x = x + .5*total_size.width - .5*text_size.width;
    }

    var node_radius = 5;

    //ret += "<rect x=\"" + centered_x + "\" y=\"" + centered_y + "\" width=\"" + text_size.width + "\" height=\"" + text_size.height + "\" style=\"fill:rgb(0,0,255);stroke-width:2;stroke:rgb(0,0,0)\" />"; //don't delete this code. this is very useful for debugging purposes
    ret += "<circle cx=\"" + (centered_x + node_radius) + "\" cy=\"" + (centered_y + node_radius) + "\" r=\"" + node_radius + "\" stroke=\"black\" stroke-width=\"1\" fill=\"blue\" />";

    ret += "<text x=\"" + (centered_x+2) + "\" y=\"" + (centered_y+20) + "\" fill=\"black\">" + endtag + "</text>"; //for debugging purposes, I'm putting the endtag here. this will eventually be replaced by the proper solver description

    var elapsedDist = 0;

    //recursively draw all the children (if any)
    for(var i = 0; i<numChildren; i++) {
        var childEndtag    = endtag + "_" + i;
        var childIndex     = getIndex(data,childEndtag);
        var childTotalSize = data[childIndex].total_size;

        if(pc_type == "mg") {
            ret += getBoxTree(data, childEndtag, x+text_size.width, y+elapsedDist);
            //calculate where child is located and draw the appropriate line
            var child_pc_type  = data[childIndex].pc_type;
            var childNodeSize  = data[childIndex].node_size;
            var childTextSize  = getTextSize(data,childEndtag);

            var child_centered_x = x+text_size.width;
            var child_centered_y = y+elapsedDist;

            if(child_pc_type == "mg")
                child_centered_y = (y+elapsedDist) + .5*childTotalSize.height - .5*childTextSize.height;
            else
                child_centered_x = (x+text_size.width) + .5*childTotalSize.width - .5*childTextSize.width;

            ret += getCurve(centered_x + node_radius, centered_y + node_radius, child_centered_x + node_radius, child_centered_y + node_radius,"east");




            elapsedDist += childTotalSize.height;
        }
        else {
            ret += getBoxTree(data, childEndtag, x+elapsedDist, y+text_size.height);
            //calculate where child is located and draw the appropriate line
            var child_pc_type  = data[childIndex].pc_type;
            var childNodeSize  = data[childIndex].node_size;
            var childTextSize  = getTextSize(data,childEndtag);

            var child_centered_x = x+elapsedDist;
            var child_centered_y = y+text_size.height;

            if(child_pc_type == "mg")
                child_centered_y = (y+text_size.height) + .5*childTotalSize.height - .5*childTextSize.height;
            else
                child_centered_x = (x+elapsedDist) + .5*childTotalSize.width - .5*childTextSize.width;

            ret += getCurve(centered_x + node_radius, centered_y + node_radius, child_centered_x + node_radius, child_centered_y + node_radius,"south");


            elapsedDist += childTotalSize.width;
        }
    }

    return ret;
}

//this function should be pretty straightforward
function getTextSize(data, endtag) {

    var index   = getIndex(data,endtag);
    var pc_type = data[index].pc_type;
    var ret     = new Object();
    ret.width   = 100;

    if(pc_type == "fieldsplit") {
        ret.height = 50;
    }
    else if(pc_type == "mg") {
        ret.height = 30;
    }
    else
        ret.height = 30;

    return ret;
}

/*
 * Each node has 3 sizes:
 * 1) text size (the most 'basic' size. the size of the smallest individual entity)
 * 2) node size (the size of the node on a particular level whether vertical or horizonally laid out)
 *    this one is the hardest to understand. a node's node-size only increases when the desired layout changes
 * 3) total size (the size of the node and all of its children)
 *
 * This method recursively calculates the latter two properties (the first one is trivial)
 * This method puts children of fieldsplit to the south of the parent node and the children of mg to the east of the parent node
 *
 */

function calculateSizes(data, endtag) {

    var index       = getIndex(data,endtag);
    var text_size   = getTextSize(data,endtag); //return an object containing 'width' and 'height'
    var numChildren = getNumChildren(data,endtag);

    var pc_type     = data[index].pc_type;

    if(numChildren == 0) {
	data[index].total_size = text_size; //set both node_size and total_size to text_size
	data[index].node_size  = text_size;
	return;
    }

    //otherwise, first recursively calculate the properties of the child nodes
    for(var i = 0; i<numChildren; i++) {
	var childEndtag = endtag + "_" + i;
	calculateSizes(data,childEndtag); //recursively calculate the sizes of all the children !!
    }

    //used to decide if we need to have a big node_size
    var parentIndex = getParentIndex(data,endtag);

    if(parentIndex == "-1") { //root
	if(pc_type == "mg") { //put children to the east. normal node_size (use text_size).
	    data[index].node_size = text_size;

	    var totalHeight = 0; //get the total heights of all the children. and the most extreme width
	    var maxWidth    = 0;

	    for(var i=0; i<numChildren; i++) {
		var childEndtag = endtag + "_" + i;
		var childIndex  = getIndex(data,childEndtag);
		var childSize   = data[childIndex].total_size;

		totalHeight += childSize.height;
		if(childSize.width > maxWidth)
		    maxWidth = childSize.width;
	    }

	    var total_size = new Object();

	    if(text_size.height > totalHeight) //should be rare, but certainly possible.
		total_size.height = text_size.height;
            else
                total_size.height = totalHeight;
	    total_size.width = text_size.width + maxWidth;

	    data[index].total_size = total_size;
	}
	else { //put children to the south. normal node_size (use text_size).
	    data[index].node_size = text_size;

	    var totalWidth = 0; //get the total widths of all the children. and the most extreme height
	    var maxHeight  = 0;

	    for(var i=0; i<numChildren; i++) {
		var childEndtag = endtag + "_" + i;
		var childIndex  = getIndex(data,childEndtag);
		var childSize   = data[childIndex].total_size;

		totalWidth += childSize.width;
		if(childSize.height > maxHeight)
		    maxHeight = childSize.height;
	    }

	    var total_size = new Object();

	    if(text_size.width > totalWidth) //should be rare, but certainly possible.
		total_size.width = text_size.width;
            else
                total_size.width = totalWidth;
	    total_size.height = text_size.height + maxHeight;

	    data[index].total_size = total_size;
	}
	return;
    } //end of root case.

    var parent_pc_type = data[parentIndex].pc_type;

    if(parent_pc_type != "mg" && pc_type == "mg") { //put children to the east + big node_size !! (simply node_size = total_size)
	var totalHeight = 0; //get the total heights of all the children. and the most extreme width
	var maxWidth    = 0;

	for(var i=0; i<numChildren; i++) {
	    var childEndtag = endtag + "_" + i;
	    var childIndex  = getIndex(data,childEndtag);
	    var childSize   = data[childIndex].total_size;

	    totalHeight += childSize.height;
	    if(childSize.width > maxWidth)
		maxWidth = childSize.width;
	}

	var total_size = new Object();

	if(text_size.height > totalHeight) //should be rare, but certainly possible.
	    total_size.height = text_size.height;
        else
            total_size.height = totalHeight;
	total_size.width = text_size.width + maxWidth;

	data[index].total_size = total_size;
	data[index].node_size  = total_size;
    }

    else if(parent_pc_type == "mg" && pc_type != "mg") { //put children to the south + big node_size !! (simply node_size = total_size)
	var totalWidth = 0; //get the total widths of all the children. and the most extreme height
	var maxHeight  = 0;

	for(var i=0; i<numChildren; i++) {
	    var childEndtag = endtag + "_" + i;
	    var childIndex  = getIndex(data,childEndtag);
	    var childSize   = data[childIndex].total_size;

	    totalWidth += childSize.width;
	    if(childSize.height > maxHeight)
		maxHeight = childSize.height;
	}

	var total_size = new Object();

	if(text_size.width > totalWidth) //should be rare, but certainly possible.
	    total_size.width = text_size.width;
        else
            total_size.width = totalWidth;
	total_size.height = text_size.height + maxHeight;

	data[index].total_size = total_size;
	data[index].node_size  = total_size;
    }

    else if(pc_type == "mg") { //put children to the east. normal node_size.
	data[index].node_size = text_size;

	var totalHeight = 0; //get the total heights of all the children. and the most extreme width
	var maxWidth    = 0;

	for(var i=0; i<numChildren; i++) {
	    var childEndtag = endtag + "_" + i;
	    var childIndex  = getIndex(data,childEndtag);
	    var childSize   = data[childIndex].total_size;

	    totalHeight += childSize.height;
	    if(childSize.width > maxWidth)
		maxWidth = childSize.width;
	}

	var total_size = new Object();

	if(text_size.height > totalHeight) //should be rare, but certainly possible.
	    total_size.height = text_size.height;
        else
            total_size.height = totalHeight;
	total_size.width = text_size.width + maxWidth;

	data[index].total_size = total_size;
    }

    else { //put children to the south. normal node_size.
	data[index].node_size = text_size;

	var totalWidth = 0; //get the total widths of all the children. and the most extreme height
	var maxHeight  = 0;

	for(var i=0; i<numChildren; i++) {
	    var childEndtag = endtag + "_" + i;
	    var childIndex  = getIndex(data,childEndtag);
	    var childSize   = data[childIndex].total_size;

	    totalWidth += childSize.width;
	    if(childSize.height > maxHeight)
		maxHeight = childSize.height;
	}

	var total_size = new Object();

	if(text_size.width > totalWidth) //should be rare, but certainly possible.
	    total_size.width = text_size.width;
        else
            total_size.width = totalWidth;
	total_size.height = text_size.height + maxHeight;

	data[index].total_size = total_size;
    }
}

//this function goes through the tree a second time and centers sister nodes and makes sure child nodes of the same level (not necesarily sisters) align on the same line. without this function, everything would be crammed tightly together.

function calculateSizes2(data, endtag) {




}


//use svg to generate a smooth Bézier curve from one point to another
//this simple logistic-looking curve algorithm is taken from d3
function getCurve(x1,y1,x2,y2,direction) {

    var ret = "";

    if(direction == "east") {
        var mid_x = (x1+x2)/2.0;

        var control1 = new Object();
        control1.x = mid_x;
        control1.y = y1;

        var control2 = new Object();
        control2.x = mid_x;
        control2.y = y2;

        ret = "<path d=\"M " + x1 + "," + y1 + " " + "C" + control1.x + "," + control1.y + " " + control2.x + "," + control2.y + " " + x2 + "," + y2 + "\" stroke =\"blue\" stroke-width=\"2\" fill=\"none\" />";
    }

    else if(direction == "south") {
        var mid_y = (y1+y2)/2.0;

        var control1 = new Object();
        control1.x = x1;
        control1.y = mid_y;

        var control2 = new Object();
        control2.x = x2;
        control2.y = mid_y;

        ret = "<path d=\"M " + x1 + "," + y1 + " " + "C" + control1.x + "," + control1.y + " " + control2.x + "," + control2.y + " " + x2 + "," + y2 + "\" stroke =\"blue\" stroke-width=\"2\" fill=\"none\" />";
    }

    /*$("#tree").html("<svg id=\"demo\" width=\"" + matInfo[0].total_size.width + "\" height=\"" + matInfo[0].total_size.height + "\" viewBox=\"0 0 " + matInfo[0].total_size.width + " " + matInfo[0].total_size.height + "\">" + ret + "</svg>");*/ //this was for debugging purposes

    return ret;
}