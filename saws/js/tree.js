//This function builds the tree using d3 (http://d3js.org/)
//A good video explaining the tree:
//http://www.youtube.com/watch?v=x8dwXoWODZ4 (part 1)
//http://www.youtube.com/watch?v=iZ6MSHA4FMU (part 2)
function buildTree(matInfo, numberOfLevels, detailed)
{
    for(var i=0; i<matInfoWriteCounter; i++) {
        if(matInfo[i].id != "-1") {
            var currentID = matInfo[i].id;
            


        }
    }




    if (detailed == false) {//use matInfo[].stringshort for KSP/PC options without prefix
    //make the tree data structure.
    //before each data is written, it is checked to see if its parent is
    //fieldsplit; if not, it doesn't write the data to the treeDAta
    //The high level checks are wrapped around more (because it cuts off all leaves)
    //Matrix A, index 0
    var treeData = {name:matInfo[0].stringshort}

    if ( matInfo[0].logstruc) {
	//make next level array
	treeData.contents = [];

	//Matrix A1, index = 1
	if ($("#pcList" + 1).data("parentFieldSplit")) {	
	    treeData.contents[0] = {name:matInfo[1].stringshort};
	    if (matInfo[1].logstruc) {
		//make next level array
		treeData.contents[0].contents = [];

		//Matrix A11, index = 3
		if ($("#pcList" + 3).data("parentFieldSplit")) {	
		    treeData.contents[0].contents[0] = {name:matInfo[3].stringshort};
		    if (matInfo[3].logstruc) {
			//make next level array
			treeData.contents[0].contents[0].contents = [];
			
			//Matrix A111, index = 7
			if ($("#pcList" + 7).data("parentFieldSplit"))
			    treeData.contents[0].contents[0].contents[0] = {name:matInfo[7].stringshort};
			//Matrix A112, index = 8
			if ($("#pcList" + 8).data("parentFieldSplit"))
			    treeData.contents[0].contents[0].contents[1] = {name:matInfo[8].stringshort};
		    }
		}

		//Matrix A12, index = 4
		if ($("#pcList" + 4).data("parentFieldSplit")) {	
		    treeData.contents[0].contents[1] = {name:matInfo[4].stringshort};
		    if (matInfo[4].logstruc) {
			//make next level array
			treeData.contents[0].contents[1].contents = [];
	                
			//Matrix A121, index = 9
			if ($("#pcList" + 9).data("parentFieldSplit"))
			    treeData.contents[0].contents[1].contents[0] = {name:matInfo[9].stringshort};

			//Matrix A122, index = 10
			if ($("#pcList" + 10).data("parentFieldSplit"))
			    treeData.contents[0].contents[1].contents[1] = {name:matInfo[10].stringshort};
		    }
		}
	    }
	}
	//Matrix A2, index = 2
	if ($("#pcList" + 2).data("parentFieldSplit")) {
	    treeData.contents[1] = {name:matInfo[2].stringshort};
	    if (matInfo[2].logstruc) {
		//make next level array
		treeData.contents[1].contents = [];

		//Matrix A21, index = 5
		if ($("#pcList" + 5).data("parentFieldSplit")) {
		    treeData.contents[1].contents[0] = {name:matInfo[5].stringshort};
		    if (matInfo[5].logstruc) {
			//make next level array
			treeData.contents[1].contents[0].contents = [];

			//Matrix A211, index = 11
			if ($("#pcList" + 11).data("parentFieldSplit"))
			    treeData.contents[1].contents[0].contents[0] = {name:matInfo[11].stringshort};

			//Matrix A212, index = 12
			if ($("#pcList" + 12).data("parentFieldSplit"))	
			    treeData.contents[1].contents[0].contents[1] = {name:matInfo[12].stringshort};
		    }
		}
		
		//Matrix A22, index = 6
		if ($("#pcList" + 6).data("parentFieldSplit")) {	
		    treeData.contents[1].contents[1] = {name:matInfo[6].stringshort};
		    if (matInfo[6].logstruc) {
			//make next level array
			treeData.contents[1].contents[1].contents = [];

			//Matrix A221, index = 13
			if ($("#pcList" + 13).data("parentFieldSplit"))
			    treeData.contents[1].contents[1].contents[0] = {name:matInfo[13].stringshort};

			//Matrix A222, index = 14
			if ($("#pcList" + 14).data("parentFieldSplit"))
			    treeData.contents[1].contents[1].contents[1] = {name:matInfo[14].stringshort};
		    }
		}
	    }
	}
    }

    } else { //use matInfo[].string for detailed KSP/PC options

    //make the tree data structure.
    //before each data is written, it is checked to see if its parent is
    //fieldsplit; if not, it doesn't write the data to the treeDAta
    //The high level checks are wrapped around more (because it cuts off all leaves)
    //Matrix A, index 0
    var treeData = {name:matInfo[0].string}

    if ( matInfo[0].logstruc) {
	//make next level array
	treeData.contents = [];

	//Matrix A1, index = 1
	if ($("#pcList" + 1).data("parentFieldSplit")) {	
	    treeData.contents[0] = {name:matInfo[1].string};
	    if (matInfo[1].logstruc) {
		//make next level array
		treeData.contents[0].contents = [];

		//Matrix A11, index = 3
		if ($("#pcList" + 3).data("parentFieldSplit")) {	
		    treeData.contents[0].contents[0] = {name:matInfo[3].string};
		    if (matInfo[3].logstruc) {
			//make next level array
			treeData.contents[0].contents[0].contents = [];
			
			//Matrix A111, index = 7
			if ($("#pcList" + 7).data("parentFieldSplit"))
			    treeData.contents[0].contents[0].contents[0] = {name:matInfo[7].string};
			//Matrix A112, index = 8
			if ($("#pcList" + 8).data("parentFieldSplit"))
			    treeData.contents[0].contents[0].contents[1] = {name:matInfo[8].string};
		    }
		}

		//Matrix A12, index = 4
		if ($("#pcList" + 4).data("parentFieldSplit")) {	
		    treeData.contents[0].contents[1] = {name:matInfo[4].string};
		    if (matInfo[4].logstruc) {
			//make next level array
			treeData.contents[0].contents[1].contents = [];
	                
			//Matrix A121, index = 9
			if ($("#pcList" + 9).data("parentFieldSplit"))
			    treeData.contents[0].contents[1].contents[0] = {name:matInfo[9].string};

			//Matrix A122, index = 10
			if ($("#pcList" + 10).data("parentFieldSplit"))
			    treeData.contents[0].contents[1].contents[1] = {name:matInfo[10].string};
		    }
		}
	    }
	}
	//Matrix A2, index = 2
	if ($("#pcList" + 2).data("parentFieldSplit")) {
	    treeData.contents[1] = {name:matInfo[2].string};
	    if (matInfo[2].logstruc) {
		//make next level array
		treeData.contents[1].contents = [];

		//Matrix A21, index = 5
		if ($("#pcList" + 5).data("parentFieldSplit")) {
		    treeData.contents[1].contents[0] = {name:matInfo[5].string};
		    if (matInfo[5].logstruc) {
			//make next level array
			treeData.contents[1].contents[0].contents = [];

			//Matrix A211, index = 11
			if ($("#pcList" + 11).data("parentFieldSplit"))
			    treeData.contents[1].contents[0].contents[0] = {name:matInfo[11].string};

			//Matrix A212, index = 12
			if ($("#pcList" + 12).data("parentFieldSplit"))	
			    treeData.contents[1].contents[0].contents[1] = {name:matInfo[12].string};
		    }
		}
		
		//Matrix A22, index = 6
		if ($("#pcList" + 6).data("parentFieldSplit")) {	
		    treeData.contents[1].contents[1] = {name:matInfo[6].string};
		    if (matInfo[6].logstruc) {
			//make next level array
			treeData.contents[1].contents[1].contents = [];

			//Matrix A221, index = 13
			if ($("#pcList" + 13).data("parentFieldSplit"))
			    treeData.contents[1].contents[1].contents[0] = {name:matInfo[13].string};

			//Matrix A222, index = 14
			if ($("#pcList" + 14).data("parentFieldSplit"))
			    treeData.contents[1].contents[1].contents[1] = {name:matInfo[14].string};
		    }
		}
	    }
	}
    }
    }

    //---------------------------------------------------
    //Create a container for the tree - a 'canvas'
    //[n*310, n* 310] for horizontal
    //[n*580, n* 310] for vertical
    var canvas = d3.select("#tree").append("svg")
	.attr("width", numberOfLevels * 400)
	.attr("height", numberOfLevels * 320)
	.append("g")
	.attr("transform", "translate(50,50)");
    
    //Call the d3 tree layout
    //[n * 200, n * 200] for horizontal
    //[n* 550, n* 200] for vertical
    var tree = d3.layout.tree()
	.size([numberOfLevels * 200, numberOfLevels * 200])
   	.children(function(d) //find who has children from the data structure
    		  {
        	      return (!d.contents || d.contents.length === 0) ? null : d.contents;
    		  });

    //initialize the nodes and links (which are used by d3 to create the paths
    var nodes = tree.nodes(treeData);
    var links = tree.links(nodes);

    //create an actual node group on the canvas (where a dot and text will be placed)
    var node = canvas.selectAll(".node")
	.data(nodes)
	.enter()
	.append("g")
	.attr("class", "node")
	.attr("transform", function (d){return "translate(" + d.y + "," + d.x + ")";}) //root: left
        //.attr("transform", function (d){return "translate(" + d.x + "," + d.y + ")";}) //root: top

    //add to that node a circle
    //change the circle properties here
    node.append("circle")
	.attr("r", 5)
	.attr("fill", "steelblue")

    //Add text to the node (names)
    //"foreignObject" is used to allow for rich formatting through mathJax and HTML (such as wrapping)
    //adjust the size of the box to create the desired aesthetic
    //Move it with x and y
    node.append("foreignObject")
	.attr('width',400)
  	.attr('height',400)//perhaps just remove these attributes altogether so it can resize itself as needed?
    //.attr('x', -23)
  	.attr('requiredFeatures','http://www.w3.org/TR/SVG11/feature#Extensibility')
  	.append('xhtml')
	.html(function (d) { return d.name; }) //this is where the data is actually placed in the tree


    //diagonal is the d3 method that draws the lines
    var diagonal = d3.svg.diagonal()
	.projection(function (d) { return [d.y, d.x]})	//root: left
        //.projection(function (d) { return [d.x, d.y]}) //root: top

    //Writes everything to screen
    canvas.selectAll(".link")
	.data(links)
	.enter()
	.append("path")
	.attr("class", "link")
	.attr("fill", "none")
	.attr("stroke", "#ADADAD")
	.attr("d", diagonal)
    
    //Tell mathJax to re compile the tex data
    MathJax.Hub.Queue(["Typeset",MathJax.Hub]);
}
