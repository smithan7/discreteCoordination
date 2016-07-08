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

#include "graph.h"
#include "miniGraph.h"
#include "agent.h"
#include "frontier.h"
#include "treeNode.h"

using namespace cv;
using namespace std;

int main(){
	vector<int> treePath;
	int treeGoal;
	srand(time(NULL));
	bool videoFlag = true;

	int numAgent = 1;

	int gSpace = 5;
	float obsThresh = 400;
	float comThresh = 400;

	world gMap(gSpace, obsThresh, comThresh);

	cerr << "made world" << endl;
	graph master;
	master.createGraph(gMap, obsThresh, comThresh, gSpace);

	cerr << "made master" << endl;
	miniGraph miniMaster;

	vector<vector<int> > sLoc;
	for(int i=0; i<numAgent; i++){
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

	cerr << "chose sLoc" << endl;

	vector<agent> bot;
	for(int i=0; i<numAgent; i++){
		bot.push_back(agent());
		bot[i].buildAgent(sLoc[i], i, gMap, obsThresh, comThresh);
	}
	time_t start = clock();
	int time = 0;

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
	vector<frontier> frnt;
	master.findFrontiers();
	master.clusterFrontiers(gMap, frnt);
	master.buildCostMapPlot(gMap, frnt);
	for(int i=0; i<numAgent; i++){
		master.addAgentToPlot(gMap, bot[i].cLoc, bot[i].myColor);
	}
	master.showCostMapPlot(99);
	cout << "ready to begin, press any key" << endl;
	waitKey(0);
	cout << "here we go!" << endl;

	int treeCntr = 101;

	int timeSteps = -1;
	while(master.frntsExist){
		timeSteps++;
		for(int i=0; i<numAgent; i++){
			master.observe(bot[i].cLoc, gMap);
		}
		master.findFrontiers();
		cerr << "out of frontiers: " << master.frntList.size() << " frontier cells found" << endl;
		for(int i=0; i<frnt.size(); i++){
			cerr << "   " << frnt[i].centroid[0] << " / " << frnt[i].centroid[1] << endl;
		}
		cerr << "into cluster" << endl;
		master.clusterFrontiers(gMap, frnt);
		cerr << "out of clusters" << endl;
		for(int i=0; i<frnt.size(); i++){
			cerr << "   " << frnt[i].centroid[0] << " / " << frnt[i].centroid[1] << endl;
		}
		//master.makeInference(frnt);
		cerr << "out of inference" << endl;


		cerr << "building miniMap" << endl;
		// build miniMap

		//miniMaster.importFrontiers(master.frntList);

		vector<vector<int> > uavLocList;
		for(int i=0; i<numAgent; i++){
			uavLocList.push_back(bot[i].cLoc);
		}
		//miniMaster.importUAVLocations(uavLocList);

		//Mat temp = master.createMiniMapImg(); // for explored view;
		//Mat temp = master.createInferredMiniMapImg() // for explored plus inferred map view
		Mat temp  = gMap.createMiniMapImg(); // for global view
		miniMaster.cLocList.clear();
		miniMaster.cLocList.push_back(bot[0].cLoc);
		//temp = master.inferredMiniMap;
		miniMaster.createMiniGraph(temp,0,0,temp.rows,temp.cols);
		temp = master.createMiniMapImg();
		miniMaster.getUnobservedMat(temp);
		//temp = master.createMiniMapInferImg();
		//miniMaster.getInferenceMat(temp);
		cerr << "out of build miniMap" << endl;

		//Mat tempInverted = Mat::zeros(miniMaster.miniImage.rows, miniMaster.miniImage.cols,CV_8UC1);
		cerr << "into invert" << endl;
		//miniMaster.breadthFirstSearchFindRoom(miniMaster.obstacleMat, miniMaster.cLocList[0]);
		//miniMaster.invertImageAroundPt(miniMaster.obstacleMat,tempInverted, miniMaster.cLocList[0]);
		//miniMaster.growFrontiers(frnt);
		//miniMaster.extractInferenceContour();
		cerr << "out of invert" << endl;


		////////////////////////////////////////////////////// Begin Tree
		/*
		cerr << "into build tree" << endl;
		if(treeCntr > 100){
			treeCntr = 0;
			int maxPulls = 1000;
			vector<int> t;
			treeNode myTree(miniMaster.cNode, miniMaster, t, -1);
			while(myTree.nPulls < maxPulls){
				cerr << "in" << endl;
				myTree.searchTree(miniMaster);
				cerr << "out" << endl;
			}
			cerr << "into exploit" << endl;
			treePath.clear();
			myTree.exploitTree(treePath);
			cerr << "out of path: " << treePath.size() << endl;
			waitKey(1);
			for(size_t i=0; i<treePath.size(); i++){
				if(treePath[i] >= 0 && treePath[i] < miniMaster.graf.size()){
					cerr << "treePath[" << i << " / " << treePath.size() << "]: " << treePath[i] << endl;
					int xPt = miniMaster.graf[treePath[i]][0];
					int yPt = miniMaster.graf[treePath[i]][1];
					if(master.costMap[xPt][yPt] == 0){
						treeGoal = treePath[i];
					}
				}
			}
			cerr << "treeGoal: " << treeGoal << endl;
			waitKey(0);
			cerr << "out of build tree" << endl;
		}
		else{
			treeCntr++;
		}

		cerr << "going into A*" << endl;
		cerr << "treeGoal / graf.size(): " << treeGoal << " / " << miniMaster.graf.size() << endl;

		vector<int> mtcPt;
		mtcPt.push_back(miniMaster.graf[treeGoal][0]);
		mtcPt.push_back(miniMaster.graf[treeGoal][1]);

		cerr << "mtcPt: " << mtcPt[0] << ", " << mtcPt[1] << endl;

		bot[0].myPath = master.aStarPath( bot[0].cLoc, mtcPt, gMap);
		cerr << "out of A*" << endl;
		if(bot[0].myPath.size() > 0){
				bot[0].cLoc = bot[0].myPath[1];
		}
		cerr << "updated path" << endl;
		*/
		///////////////////////////////////////// End Tree


		/////////////////////////////////////// Begin Greedy Frontier
		/*
		master.findNearestFrontier(bot[0], frnt, gMap);
		bot[0].myPath = master.aStarPath( bot[0].cLoc, bot[0].gLoc, gMap);
		cerr << "out of A*" << endl;
		if(bot[0].myPath.size() > 0){
				bot[0].cLoc = bot[0].myPath[1];
		}
		*/
		/////////////////////////////////////// End Greedy Frontier


		////////////////////////////// Begin Market

		master.centralMarket(gMap, frnt, bot);
		cerr << "out of market" << endl;

		for(int i=0; i<numAgent; i++){
			cerr << "going into D*" << endl;
			cerr << "   bot[i].gLoc: " << frnt[bot[i].gIndex].centroid[0] << ", " << frnt[bot[i].gIndex].centroid[1] << endl;
			bot[i].myPath = master.dStarPath(frnt[bot[i].gIndex], bot[i], gMap);
			//cerr << "going into A*" << endl;
			//bot[i].myPath = master.aStarPath( bot[i].cLoc, frnt[bot[i].gIndex].centroid, gMap);
			cerr << "out of A*" << endl;
			if(bot[i].myPath.size() > 0){
					bot[i].cLoc = bot[i].myPath[1];
			}
			cerr << "updated path" << endl;
		}
		cerr << "out of updating path" << endl;

		//////////////////////////////////////////////////// End market

		// build dispay
		cerr << "build display" <<endl;
		master.buildCostMapPlot(gMap, frnt);
		for(int i=0; i<numAgent; i++){
			master.addAgentToPlot(gMap, bot[i].cLoc, bot[i].myColor);
		}
		master.showCostMapPlot(99);



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
		cerr << "-------------------------------------------timeSteps: " << timeSteps << endl;
		time++;
	}
	cerr << "finished program in " << timeSteps << endl;
	waitKey(0);
}
