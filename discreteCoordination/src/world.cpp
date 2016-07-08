/*
 * world.cpp
 *
 *  Created on: Mar 28, 2016
 *      Author: andy
 */

#include "world.h"

using namespace cv;
using namespace std;

world::world(int gSpace, float obsThresh, float comThresh) {
	this->gSpace = gSpace;
	this->obsThresh = obsThresh;
	this->commThresh = comThresh;
	//this->fileName = "/home/andy/Dropbox/workspace/fabmap2Test/wholeMapImages/generated/map5001053.jpg";
	this->fileName = "/home/andy/Dropbox/workspace/nextGenCoord/test2.jpg";

	this->image = imread(this->fileName,1);
	cvtColor(this->image,this->imgGray,CV_BGR2GRAY);
	threshold(this->imgGray,this->imgGray,250,255,THRESH_BINARY);

	this->nRows = imgGray.rows;
	this->nCols = imgGray.cols;
	this->initializeMaps();
	this->getDistGraph();
	this->nRows = this->costMap.size();
	this->nCols = this->costMap[0].size();
	this->getObsGraph();
	//this->getCommGraph();
}

Mat world::createMiniMapImg(){
	Mat temp = Mat::zeros(this->nRows, this->nCols,CV_8UC1);
	for(int i=0; i<this->nRows; i++){
		for(int j=0; j<this->nCols; j++){
			if(this->costMap[i][j] > 100){
				temp.at<uchar>(i,j,0) = 101;
			}
			else{
				temp.at<uchar>(i,j,0) = this->costMap[i][j];
			}
		}
	}
	return temp;
}

void world::initializeMaps(){
	for(int i=this->gSpace/2; i<this->nRows - this->gSpace/2; i=i+this->gSpace){ // count by gSpace
		vector<int> tempA;
		vector<Point> tempP;
		for(int j=this->gSpace/2; j<this->nCols - this->gSpace/2; j = j+this->gSpace){ // count by gSpace
			Point p;
			p.x = j;
			p.y = i;
			tempP.push_back(p);

			bool f = true;
			for(int k=i-this->gSpace/2; k < i+this->gSpace/2 +1; k++){
				for(int l=j-this->gSpace/2; l < j+this->gSpace/2 +1; l++){
					Scalar intensity = this->imgGray.at<uchar>(k,l);
					if(intensity[0] != 255 ){ // is free space
						f = false;
					}
				}
			}

			if(f){ // is free space
				tempA.push_back(0);
			}
			else{ // is an obstacle
				tempA.push_back(255);
			}
		}
		this->pointMap.push_back(tempP);
		this->costMap.push_back(tempA);
	}
}

float world::getEuclidDist(int x0, int y0, int x1, int y1){

	int bx = abs(x0 - x1);
	int by = abs(y0 - y1);

	return(this->distGraph[by][bx]);
}


void world::getDistGraph(){
	for(int i=0; i<this->nRows; i++){
		vector<float> d;
		for(int j=0; j<this->nCols; j++){
			d.push_back(sqrt(pow(i,2) + pow(j,2)));
		}
		this->distGraph.push_back(d);
	}
}

void world::getObsGraph(){
	for(int i=0; i<this->nRows; i++){
		vector<vector<vector<int> > > to; // [yLoc][list][x/y]
		for(int j=0; j<this->nCols; j++){
			vector<vector<int> > too; // [list][x/y]
			to.push_back(too);
		}
		this->obsGraph.push_back(to);
	}


	for(int i=0; i<this->nRows; i++){
		//cerr << "i: " << i << endl;
		for(int j=0; j<this->nCols; j++){ // check each node
			//cerr << " j: " << j;
			if(this->costMap[i][j] == 0){ // am I traversable?
				//cerr << " is traversable" << endl;
				for(int k=0; k<this->nRows;k++){
					//cerr << "  k: " << k << endl;
					for(int l=0; l<this->nCols; l++){ // against all other nodes
						//cerr << "  l: " << l;

						if(this->costMap[k][l] == 0){ // are they traversable?
							//cerr << " is traversable" << endl;
							float dist = this->getEuclidDist(this->pointMap[i][j].x,this->pointMap[i][j].y,this->pointMap[k][l].x,this->pointMap[k][l].y);
							//cerr << "   pts: " << this->pointMap[i][j].x << "," << this->pointMap[i][j].y << "," << this->pointMap[k][l].x << "," << this->pointMap[k][l].y << endl;
							//cerr << "   dist: " << dist << endl;

							if(dist < this->obsThresh && dist > 0){ // is it close enough to observe
								//cerr << i << "," << j << "," << k << "," << l << endl;
								float unitVecX = (this->pointMap[k][l].x - this->pointMap[i][j].x) / dist; // get unit vector in right direction
								float unitVecY = (this->pointMap[k][l].y - this->pointMap[i][j].y) / dist;
								//cerr << "   uVec: " << unitVecX << ", " << unitVecY << endl;
								int steps = dist; // steps to check
								bool obsFlag = true;
								for(int m=1; m<steps; m++){ // check all intermediate points between two cells
									int aX = this->pointMap[i][j].x + m*unitVecX;
									int aY = this->pointMap[i][j].y + m*unitVecY;
									Scalar intensity = this->imgGray.at<uchar>(aY,aX);
									if(intensity[0] != 255 ){
										obsFlag = false;
										break;
									}
								}
								//cerr << "   oFlg: " << obsFlag << endl;
								if(obsFlag){ // are there no obstacles between me and them?
									vector<int> t;
									t.push_back(k);
									t.push_back(l);
									this->obsGraph[i][j].push_back(t);
								}
							}
							else if(dist == 0){
								vector<int> t;
								t.push_back(k);
								t.push_back(l);
								this->obsGraph[i][j].push_back(t);
							}
						}
					}
				}
			}
		}
	}
}

void world::getCommGraph(){

	/*
	for(int i=0; i<this->nRows; i++){
		for(int j=0; j<this->nCols; j++){ // check each node
			if(this->costMap[i][j] == 0){ // am I traversable?

				for(int k=i; k<this->nRows;k++){
					for(int l=j; l<this->nCols; l++){ // against all other nodes
						if(this->costMap[k][l] == 0){ // are they traversable?

							if(this->pointMap[i][j].distGraph[k][l] < this->commThresh){ // is it close enough to observe
								float unitVecX = (this->pointMap[k][l].x - this->pointMap[i][j].x) / this->pointMap[i][j].distGraph[k][l]; // get unit vector in right direction
								float unitVecY = (this->pointMap[k][l].y - this->pointMap[i][j].y) / this->pointMap[i][j].distGraph[k][l];
								int steps = this->pointMap[i][j].distGraph[k][l]; // steps to check
								bool obsFlag = true;
								for(int m=1; m<steps; m++){
									int aX = this->pointMap[i][j].x + m*unitVecX;
									int aY = this->pointMap[i][j].y + m*unitVecY;
									Scalar intensity = this->imgGray.at<uchar>(aY,aX);
									if(intensity[0] != 255 ){
										obsFlag = false;
										break;
									}
								}
								if(obsFlag){ // are there no obstacles between me and them?
									this->pointMap[i][j].comGraph[k][l] = true;
									this->pointMap[k][l].comGraph[i][j] = true;
								}
							}
						}

					}
				}
			}
		}
	}
	*/
}

void world::clearPlot(){
	this->image = imread(this->fileName);
	cvtColor(this->image,this->imgGray,CV_BGR2GRAY);
	threshold(this->imgGray,this->imgGray,127,255,THRESH_BINARY);
}

void world::addCommLine(vector<int> b,vector<int> c){
	vector<int> a;
	a.push_back(b[0]);
	a.push_back(b[1]);
	a.push_back(c[0]);
	a.push_back(c[1]);
	this->commLine.push_back(a);
}

void world::plotCommLines(){
	Scalar color;
	color[0] = 255;
	color[1] = 0;
	color[2] = 0;
	for(int i=0; i<(int)this->commLine.size(); i++){
		line(this->image,this->pointMap[this->commLine[i][0]][this->commLine[i][1]],this->pointMap[this->commLine[i][2]][this->commLine[i][3]],color,2,8);
	}
	this->commLine.erase(this->commLine.begin(),this->commLine.end());
}

void world::plotTravelGraph(){
	Scalar color;
	color[0] = 0;
	color[1] = 0;
	color[2] = 0;
	for(int i=0; i<this->nRows; i++){
		for(int j=0; j<this->nCols; j++){
			if(this->costMap[i][j] == 0){ // are they traversable?

				for(int k=0; k<2; k++){
					if(!this->costMap[i][j+k] == 0){
						line(this->image,this->pointMap[i][j],this->pointMap[i][j+k],color,1,8);
					}
				}
			}
		}
	}
}

void world::plotFrontierGraph(){
	Scalar color;
	color[0] = 0;
	color[1] = 0;
	color[2] = 255;
	//for(int i=0; i<(int)this->frntList.size(); i++){
		//circle(this->image,this->pointMap[this->frntList[i][0]][this->frntList[i][1]],this->gSpace,color,-1);
	//}
}

Mat world::createExplImage(){
	Scalar color;
	color[0] = 255;
	color[1] = 255;
	color[2] = 255;
	Mat skel(this->image.rows,this->image.cols,CV_8UC1,Scalar(0));
	for(int i=0; i<this->nRows; i++){
		for(int j=0; j<this->nCols; j++){
			if(this->costMap[i][j] == 0){ // are they traversable?
				Point a;
				a.x = this->pointMap[i][j].x + this->gSpace;
				a.y = this->pointMap[i][j].y + this->gSpace;
				rectangle(skel,this->pointMap[i][j],a,color,-1);
			}
		}
	}
	return(skel);
}

void world::plotPath(vector<vector<int> > myPath, int myColor[3], int pathIndex){
	Scalar color;
	color[0] = myColor[0];
	color[1] = myColor[1];
	color[2] = myColor[2];
	//for(int i=pathIndex; i<(int)myPath.size(); i++){
	//	line(this->image,this->pointMap[myPath[i-1]],this->pointMap[myPath[i]],Scalar{myColor[0],myColor[1],myColor[2]},3,8);
	//}
	circle(this->image, this->pointMap[myPath[pathIndex][0]][myPath[pathIndex][1]], 10, color, -1);
	//circle(this->image,this->pointMap[myPath[myPath.size()-1]],10,Scalar{myColor[0],myColor[1],myColor[2]},-1);
	//circle(this->image,this->pointMap[myPath[myPath.size()-1]],5,Scalar{0,0,0},-1);
}

void world::plotMap(){
	Mat temp;
	resize(this->image,temp,Size(),1,1,CV_INTER_AREA);
	imshow("Global Map", temp);
	waitKey(1);
}

world::~world() {
	// TODO Auto-generated destructor stub
}

