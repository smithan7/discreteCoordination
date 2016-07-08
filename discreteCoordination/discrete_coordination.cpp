//============================================================================
// Name        : discrete_coordination.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>


#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "Graph.h"
#include "MiniGraph.h"
#include "Agent.h"
#include "Frontier.h"

using namespace cv;
using namespace std;

float getObservedMapEntropy(Mat &globalInfMat, Mat &workingMat);
float getInferredMapEntropy(Mat &globalInfMat, Mat &workingMat);


/* to do
 * -struct inference, simulated annealing with warp afine (rotate) and scale matched image, minimize bitwise_xor for match cost
 * - redo visibility using breadth first search from central node out, evaluated node is projected along vector from center outward. Evaluated node added to closed set. all inside perim in open set
 * - search along tree to find doors and segment the map
 */



int main(){

	cerr << "hello" << endl;

	destroyAllWindows();

	vector<int> treePath;
	int treeGoal;
	srand(time(NULL));
	bool videoFlag = true;

	int numAgent = 1;
	int numIterations = 1;

	int gSpace = 5;
	float obsThresh = 400;
	float comThresh = 400;

	World gMap(gSpace, obsThresh, comThresh);
	cout << "made world" << endl;
	vector<vector<int> > sLoc;
	for(int i=0; i<numIterations; i++){
		while(true){
			int tRow = rand() % gMap.nRows;
			int tCol = rand() % gMap.nCols;
			if(gMap.costMap[tRow][tCol] == 0){ // its not an obstacle
				vector<int> s;
				s.push_back(tRow);
				s.push_back(tCol);
				sLoc.push_back(s);
				break;
			}
		}
	}
	cout << "chose sLoc" << endl;

	for(int fu = 0; fu<numIterations; fu++){

		Graph master;
		master.createGraph(gMap, obsThresh, comThresh, gSpace);

		cout << "made master" << endl;
		MiniGraph miniMaster;

		vector<Agent> bot;
		bot.push_back(Agent());
		bot[0].buildAgent(sLoc[fu], 0, gMap, obsThresh, comThresh);
		cout << "sLoc: " << sLoc[fu][0] << " , " << sLoc[fu][1] << endl;
		time_t start = clock();

		/*
		// video writers
		VideoWriter skelVideo, displayVideo;
		Size frameSize(static_cast<int>(master.image.rows), static_cast<int>(master.image.cols));
		skelVideo.open("skelVid_slam.avi",CV_FOURCC('P','I','M','1'), 20, frameSize, true );
		displayVideo.open("displayVid_slam.avi",CV_FOURCC('P','I','M','1'), 20, frameSize, true );
		*/

		for(int i=0; i<numAgent; i++){
			master.observe(bot[i].cLoc, gMap);
		}

		master.findFrontiers();
		master.clusterFrontiers(gMap);
		master.buildCostMapPlot(gMap);
		for(int i=0; i<numAgent; i++){
			master.addAgentToPlot(gMap, bot[i]);
		}
		master.showCostMapPlot(0);
		cout << "ready to begin, press any key" << endl;
		waitKey(0);
		cout << "here we go!" << endl;

		int timeSteps = -1;
		while(master.frntsExist && timeSteps < 100){
			cerr << "main while loop start" << endl;
			timeSteps++;

			for(int i=0; i<numAgent; i++){
				master.observe(bot[i].cLoc, gMap);
			}
			master.findFrontiers();
			master.clusterFrontiers(gMap);

			cerr << "main while loop0: " << master.frntsExist <<  endl;

			Mat naiveCostMap = master.makeNaiveMatForMiniMap(); // naive to remaining space
			cerr << "made naive" << endl;
			Mat inferredGeometricCostMap = master.makeGeometricInferenceMatForMiniMap(); // inferred remaining space
			cerr << "made geometric" << endl;
			//Mat inferredStructuralCostMap = master.makeStructuralInferenceMatForMiniMap();
			Mat inferredGlobalCostMap = master.makeGlobalInferenceMat(gMap); // all remaining space
			cerr << "made global" << endl;

			float observedEntropy = getObservedMapEntropy(inferredGlobalCostMap, naiveCostMap);
			float inferredEntropy = getInferredMapEntropy(inferredGlobalCostMap, inferredGeometricCostMap);

			namedWindow("naive Costmap", WINDOW_NORMAL);
			imshow("naive Costmap", naiveCostMap);

			namedWindow("inferred Geo Costmap", WINDOW_NORMAL);
			imshow("inferred Geo Costmap", inferredGeometricCostMap);

			namedWindow("inferred Global Costmap", WINDOW_NORMAL);
			imshow("inferred Global Costmap", inferredGlobalCostMap);
			waitKey(1);

			// build miniMap
			cout << "building miniMap" << endl;

			miniMaster.importFrontiers(master.frontiersList);
			for(int i=0; i<numAgent; i++){ 		// import UAV locations
				miniMaster.cLocList.push_back(bot[i].cLoc);
			}

			cerr << "main while loop1: " << master.frntsExist <<  endl;


			miniMaster.createMiniGraph(inferredGeometricCostMap);
			cerr << "into display" << endl;
			miniMaster.displayCoordMap();
			cerr << "out of display" << endl;
			miniMaster.cNode = miniMaster.findNearestNode(bot[0].cLoc);

			cerr << "main while loop2: " << master.frntsExist <<  endl;


			// masterGraph planning /////////////////////////////////////////////////////////////////////////////////////
			float maxPathLength = 100 - timeSteps;
			int gNode = bot[0].gNode;
			if(timeSteps % 1 == 0){
				gNode = miniMaster.masterGraphPathPlanning(maxPathLength, 10000, 10000);
				cerr << "out of masterGraphPathPlanning" << endl;
				bot[0].gNode = gNode;
			}
			cerr << "gNode: " << gNode << endl;
			vector<int> gLoc;
			gLoc.push_back(miniMaster.graf[gNode][0]);
			gLoc.push_back(miniMaster.graf[gNode][1]);

			cerr << "main while loop3: " << master.frntsExist <<  endl;


			/*
			// tsp path planner ///////////////////////////////////////////////////////////////////////////////////////////
			cerr << "into tsp" << endl;
			vector<int> path = miniMaster.tspPathPlanner(2000,1);
			for(size_t i=0; i<path.size(); i++){
				cout << path[i] << endl;
			}
			cerr << "out of tsp" << endl;

			vector<int> gLoc;
			gLoc.push_back(miniMaster.graf[path[1]][0]);
			gLoc.push_back(miniMaster.graf[path[1]][1]);
			*/

			/*
			// mcts path planner //////////////////////////////////////////////////////////////////////////////////////////
			vector<int> path;
			miniMaster.mctsPathPlanner(path, 1000, 10000);

			cerr << "mctsPath: ";
			for(size_t i=0; i<path.size(); i++){
				cerr << path[i] << ", ";
			}
			cerr << endl;

			vector<int> gLoc;
			gLoc.push_back(miniMaster.graf[path[1]][0]);
			gLoc.push_back(miniMaster.graf[path[1]][1]);

			/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

			*/
			cout << "out of build miniMap" << endl;

			/////////////////////////////////////// Update UAV path
			for(int i=0; i<numAgent; i++){
				cerr << "into A*" << endl;
				bot[i].myPath = master.aStarPath( bot[i].cLoc, gLoc, gMap);
				cout << "out of A*" << endl;
				if(bot[i].myPath.size() > 0){
						bot[i].cLoc = bot[i].myPath[1];
				}
				cout << "updated path" << endl;
			}
			cout << "out of updating path" << endl;

			cerr << "main while loop4: " << master.frntsExist <<  endl;


			//////////////////////////////////////////////////// End update path
			// build dispay
			cout << "build display" <<endl;
			master.buildCostMapPlot(gMap);
			cerr << "main while loop5: " << master.frntsExist <<  endl;
			for(int i=0; i<numAgent; i++){
				master.addAgentToPlot(gMap, bot[i]);
			}
			cerr << "main while loop6 : " << master.frntsExist <<  endl;

			master.showCostMapPlot(0);
			waitKey(1);



			/*
			skelVideo.write(master.skelImg);
			displayVideo.write(master.image);
			skelVideo.write(master.skelImg);
			displayVideo.write(master.image);
			skelVideo.write(master.skelImg);
			displayVideo.write(master.image);
			skelVideo.write(master.skelImg);
			displayVideo.write(master.image);
		*/

			cout << "---------------- ---------------------------timeSteps: " << timeSteps << endl;
			cerr << "main while loop end: " << master.frntsExist << endl;
			cerr << "observedEntropy: " << observedEntropy << endl;
			cerr << "inferredEntropy: " << inferredEntropy << endl;
			waitKey(1);
		}
		cout << "finished program in " << timeSteps << endl;
		waitKey(1);
	}
}

float getObservedMapEntropy(Mat &globalInfMat, Mat &workingMat){
	int globalFree = 0;
	for(int i=0; i<globalInfMat.cols; i++){
		for(int j=0; j<globalInfMat.rows; j++){
			if(globalInfMat.at<uchar>(i,j,0) < 10){
				globalFree++;
			}
		}
	}
	int workingObserved = 0;
	for(int i=0; i<globalInfMat.cols; i++){
		for(int j=0; j<globalInfMat.rows; j++){
			if(workingMat.at<uchar>(i,j,0) == 1){
				workingObserved++;
			}
		}
	}

	return float(workingObserved)/float(globalFree);
}

float getInferredMapEntropy(Mat &globalInfMat, Mat &workingMat){
	int globalFree = 0;
	for(int i=0; i<globalInfMat.cols; i++){
		for(int j=0; j<globalInfMat.rows; j++){
			if(globalInfMat.at<uchar>(i,j,0) < 10){
				globalFree++;
			}
		}
	}
	int workingInferred = 0;
	for(int i=0; i<globalInfMat.cols; i++){
		for(int j=0; j<globalInfMat.rows; j++){
			if(workingMat.at<uchar>(i,j,0) <= 2){
				workingInferred++;
			}
		}
	}

	return float(workingInferred)/float(globalFree);
}


