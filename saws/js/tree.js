//This function builds the tree using d3 (http://d3js.org/)
//A good video explaining the tree:
//http://www.youtube.com/watch?v=x8dwXoWODZ4 (part 1)
//http://www.youtube.com/watch?v=iZ6MSHA4FMU (part 2)
function buildTree()
{
    //initialize tree
    var treeData = {};

    //clean up matInfo
    cleanMatInfo();
    sortMatInfo();

    //calculate number of levels
    var numberOfLevels = 1;
    for(var i=0; i<matInfo.length; i++) {
        var currentLevel = getNumUnderscores(matInfo[i].endtag);
        if(currentLevel > numberOfLevels)
            numberOfLevels = currentLevel;
    }

    alert(matInfo.length);
    for(var i=0; i<matInfo.length; i++) {//first zero doesn't matter
        var string = getSimpleDescription(matInfo[i].endtag);alert(string);
        var obj    = {name: string};//the new object to be placed
        var endtag = matInfo[i].endtag;
        var needsAllocation=false;

        if(endtag == "0")
            treeData=obj;

        //get the last number in the endtag
        if(endtag == "0" || endtag.substring(endtag.lastIndexOf("_")+1,endtag.length) == "0") {
            //need to allocate
            needsAllocation=true;
        }

        if(getNumUnderscores(endtag) == 1) {
            if(needsAllocation) {
                treeData.contents=[];
            }
            treeData.contents[getNthPos(endtag,1)] = obj;
        }
        else if(getNumUnderscores(endtag) == 2) {
            if(needsAllocation) {
                treeData.contents[getNthPos(endtag,1)].contents=[];
            }
            treeData.contents[getNthPos(endtag,1)].contents[getNthPos(endtag,2)]=obj;
        }
        else if(getNumUnderscores(endtag) == 3) {
            if(needsAllocation) {
                treeData.contents[getNthPos(endtag,1)].contents[getNthPos(endtag,2)]=[];
            }
            treeData.contents[getNthPos(endtag,1)].contents[getNthPos(endtag,2)].contents[getNthPos(endtag,3)]=obj;
        }

    }

    //pos goes from 0 to numUnderscores. pos 0 is always "0"
    function getNthPos(endtag,pos) {
        if(pos == 0)
            return 0;
        if(pos > getNumUnderscores(endtag))
            return -1; //error. this is not possible.

        for(var i=0; i<pos; i++) { //remove 'pos' number of underscores
            endtag = endtag.substring(endtag.indexOf("_")+1, endtag.length);
        }

        if(endtag.indexOf("_") == -1) //has no more underscores so just return everything that's left
            return parseInt(endtag);
        else //otherwise, return up to the next underscore
            return parseInt(endtag.substring(0,endtag.indexOf("_")));
    }

    function cleanMatInfo() {//remove all the -1 elements (these get generated when something is deleted)

        for(var i=0; i<matInfo.length; i++) {
            if(matInfo[i].endtag == "-1") {

                //shift everything down
                for(var j=i; j<matInfo.length-1; j++)
                    matInfo[j]=matInfo[j+1];

                i--;//there might be two in a row

                delete matInfo[matInfo.length-1];//remove garbage value
                matInfo.length--;//after deletion, there will be an "undefined" value there so we need this line to actually shrink the array
            }
        }

    }

    //selection sort. NOTE: THIS FUNCTION ASSUMES ALL GARBAGE VALUES (ID="-1") HAVE ALREADY BEEN REMOVED USING cleanMatInfo()
    function sortMatInfo() {
        for(var i=0; i<matInfo.length-1; i++) {//only need to go to second to last element
            var indexOfCurrentSmallest = i;
            for(var j=i; j<matInfo.length; j++) {
                if(matInfo[j].endtag.localeCompare(matInfo[i].endtag) == -1)
                    indexOfCurrentSmallest = j;
            }
            //swap i and indexOfCurrentSmallest
            var temp                        = matInfo[i];
            matInfo[i]                      = matInfo[indexOfCurrentSmallest];
            matInfo[indexOfCurrentSmallest] = temp;
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
