/*
 * World.cpp
 *
 *  Created on: Mar 28, 2016
 *      Author: andy
 */

#include "World.h"

using namespace cv;
using namespace std;

World::World(string fName, int gSpace, float obsThresh, float comThresh) {
	this->gSpace = gSpace;
	this->obsThresh = obsThresh;
	this->commThresh = comThresh;

	FileStorage ifile(fName + ".yml", FileStorage::READ);
	if(true && ifile.isOpened()){
		cout << "world::loading " << fName << ".yml" << endl;
		ifile.release();
		this->pullWorldFromYML(fName);
		this->costmap.nRows = this->costMap.size();
		this->costmap.nCols = this->costmap.cells[0].size();
		this->getDistGraph();
	}
	else{
		cout << "world::building " << fName << ".yml" << endl;
		ifile.release();

		string fileName = fName + ".jpg";
		this->image = imread(fileName,1);
		cvtColor(this->image,this->imgGray,CV_BGR2GRAY);
		threshold(this->image,this->imgGray,127,255,THRESH_BINARY);

		this->initializeMaps();
		this->costmap.nRows = this->costMap.size();
		this->costmap.nCols = this->costmap.cells[0].size();
		this->getDistGraph();
		this->getObsGraph();
		//this->getCommGraph();
		this->saveWorldToYML(fName);
	}
}

void World::observe(vector<int> cLoc, Costmap &costmap){

	for(size_t i=0; i<this->obsGraph[cLoc[0]][cLoc[1]].size(); i++){
		int a = this->obsGraph[cLoc[0]][cLoc[1]][i][0];
		int b = this->obsGraph[cLoc[0]][cLoc[1]][i][1];
		costmap.cells[a][b] = this->costmap.cells[a][b];
	}

	// set obstacles in costmap
	int minObs[2], maxObs[2];
	if(cLoc[0] - this->obsThresh > 0){
		minObs[0] = cLoc[0]-this->obsThresh;
	}
	else{
		minObs[0] = 0;
	}

	if(cLoc[0] + this->obsThresh < costmap.nRows){
		maxObs[0] = cLoc[0]+this->obsThresh;
	}
	else{
		maxObs[0] = costmap.nRows;
	}

	if(cLoc[1] - this->obsThresh > 0){
		minObs[1] = cLoc[1]-this->obsThresh;
	}
	else{
		minObs[1] = 0;
	}

	if(cLoc[1] + this->obsThresh < costmap.nCols){
		maxObs[1] = cLoc[1]+this->obsThresh;
	}
	else{
		maxObs[1] = costmap.nCols;
	}

	for(int i=1+minObs[0]; i<maxObs[0]-1; i++){
		for(int j=1+minObs[1]; j<maxObs[1]-1; j++){
			if(costmap.cells[i][j] > 200){ // am i an obstacle?
				for(int k=i-1; k<i+2; k++){
					for(int l=j-1; l<j+2; l++){
						if(costmap.cells[k][l] < 10){ // are any of my nbrs visible?
							costmap.cells[i][j] = 201; // set my cost
						}
					}
				}
			}
		}
	}
}


void World::saveWorldToYML(string fName){
	string filename =  fName + ".yml";
	FileStorage fs(filename, FileStorage::WRITE);

	fs << "costMap" << "[";
	for(size_t i=0; i<this->costMap.size(); i++){
		fs << this->costmap.cells[i];
	}
	fs << "]";

	fs << "obsGraph" << "[";
	for(size_t i=0; i<this->obsGraph.size(); i++){
		for(size_t j=0; j<this->obsGraph[i].size(); j++){
			vector<int> tzl;
			tzl.push_back(i);
			tzl.push_back(j);
			for(size_t k=0; k<this->obsGraph[i][j].size(); k++){
				tzl.push_back(this->obsGraph[i][j][k][0]);
				tzl.push_back(this->obsGraph[i][j][k][1]);
			}
			fs << tzl;
		}
	}
	fs << "]";
	fs.release();
}

void World::pullWorldFromYML(string fName){

	FileStorage fsN(fName + ".yml", FileStorage::READ);
	fsN["costMap"] >> this->costmap.cells;

	vector<vector<int> > tObsGraph;

	fsN["obsGraph"] >> tObsGraph;
	fsN.release();

	cout << "world::tObsGraph.size(): " << tObsGraph.size() << " < " << tObsGraph[0].size() << endl;

	for(int i=0; i<this->costmap.nRows; i++){
		vector<vector<vector<int> > > c;
		for(int j=0; j<this->costmap.nCols; j++){
			vector<vector<int> > a;
			for(size_t k=0; k<tObsGraph[this->costmap.nRows*i+j].size(); k = k+2){
				vector<int> b;
				b.push_back(tObsGraph[this->costmap.nRows*i+j][k]);
				b.push_back(tObsGraph[this->costmap.nRows*i+j][k+1]);
				a.push_back(b);
			}
			c.push_back(a);
		}
		this->obsGraph.push_back(c);
	}

	cout << "world::obsGraph.size(): " << obsGraph.size() << " < " << obsGraph[0].size() << endl;

	/*
	for(int i=0; i<this->obsGraph.size(); i++){
		for(int j=0; j<this->obsGraph[i].size(); j++){
			for(int k=0; k<this->obsGraph[i][j].size(); k++){
				cerr << this->obsGraph[i][j][k][0] << ", " << this->obsGraph[i][j][k][1] << "; ";
			}
			cerr << endl;
		}
	}
	*/
}

Mat World::createMiniMapImg(){
	Mat temp = Mat::zeros(this->costmap.nRows, this->costmap.nCols,CV_8UC1);
	for(int i=0; i<this->costmap.nRows; i++){
		for(int j=0; j<this->costmap.nCols; j++){
			if(this->costmap.cells[i][j] > 100){
				temp.at<uchar>(i,j,0) = 101;
			}
			else{
				temp.at<uchar>(i,j,0) = this->costmap.cells[i][j];
			}
		}
	}
	return temp;
}

void World::initializeMaps(){
	for(int i=0; i<=this->imgGray.rows; i=i+this->gSpace){ // count by gSpace
		vector<int> tempA;
		vector<Point> tempP;
		for(int j=0; j<=this->imgGray.cols; j = j+this->gSpace){ // count by gSpace
			bool f = true;
			int step = this->gSpace/2;
			for(int k=i-step; k < i+step+1; k++){
				if(k >= 0 && k < this->imgGray.rows){
					for(int l=j-step; l < j+step+1; l++){
						if(l >= 0 && l < this->imgGray.cols){
							Scalar intensity = this->imgGray.at<uchar>(k,l,0);
							if(intensity[0] != 255 ){ // is free space
								f = false;
								k = this->imgGray.rows;
								l = this->imgGray.cols;
							}
						}
					}
				}
			}
			if(f){ // is free space
				tempA.push_back(1);
			}
			else{ // is an obstacle
				tempA.push_back(201);
			}
		}
		this->costmap.cells.push_back(tempA);
	}
}

void World::getObsGraph(){
	for(int i=0; i<this->costmap.nRows; i++){
		vector<vector<vector<int> > > to; // [yLoc][list][x/y]
		for(int j=0; j<this->costmap.nCols; j++){
			vector<vector<int> > too; // [list][x/y]
			to.push_back(too);
		}
		this->obsGraph.push_back(to);
	}
	for(int i=0; i<this->costmap.nRows; i++){
		for(int j=0; j<this->costmap.nCols; j++){ // check each node
			if(this->costmap.cells[i][j] < 10){ // am I traversable?
				//cout << "cell: " << i << ", " << j << " is traversable" << endl;
				for(int k=0; k<this->costmap.nRows;k++){
					for(int l=0; l<this->costmap.nCols; l++){ // against all other nodes
						if(this->costmap.cells[k][l] < 10){ // are they traversable?
							//cout << "cell: " << i << ", " << j << " against " << k << ", " << l << ", both traversable*" << endl;
							float dist = this->getEuclidDist(i,j,k,l);
							//cout << "   dist: " << dist << endl;

							if(dist < this->obsThresh && dist > 0){ // is it close enough to observe
								//cout << "   dist crit met" << endl;
								float unitVecX = (k - i) / dist; // get unit vector in right direction
								float unitVecY = (l - j) / dist;
								//cout << "   uVec: " << unitVecX << ", " << unitVecY << endl;
								int steps = dist; // steps to check
								bool obsFlag = true;
								for(int m=1; m<steps; m++){ // check all intermediate points between two cells
									int aX = i + m*unitVecX;
									int aY = j + m*unitVecY;
									//cout << "   ax / ay: " << aX << " / " << aY << endl;
									if(this->costmap.cells[aX][aY] > 10 ){
										obsFlag = false;
										break;
									}
								}
								if(obsFlag){ // are there no obstacles between me and them?
									vector<int> t;
									t.push_back(k);
									t.push_back(l);
									this->obsGraph[i][j].push_back(t);
								}
							}
							else if(dist == 0){
								vector<int> t;
								t.push_back(i);
								t.push_back(j);
								this->obsGraph[i][j].push_back(t);
							}
						}
					}
				}
			}
		}
	}
}

void World::getCommGraph(){

	/*
	for(int i=0; i<this->costmap.nRows; i++){
		for(int j=0; j<this->costmap.nCols; j++){ // check each node
			if(this->costmap.cells[i][j] == 0){ // am I traversable?

				for(int k=i; k<this->costmap.nRows;k++){
					for(int l=j; l<this->costmap.nCols; l++){ // against all other nodes
						if(this->costmap.cells[k][l] == 0){ // are they traversable?

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

void World::addCommLine(vector<int> b,vector<int> c){
	vector<int> a;
	a.push_back(b[0]);
	a.push_back(b[1]);
	a.push_back(c[0]);
	a.push_back(c[1]);
	this->commLine.push_back(a);
}

void World::plotCommLines(){
	Scalar color;
	color[0] = 255;
	color[1] = 0;
	color[2] = 0;
	for(int i=0; i<(int)this->commLine.size(); i++){
		// TODO fix world pointmap
		/*
		line(this->image,this->pointMap[this->commLine[i][0]][this->commLine[i][1]],this->pointMap[this->commLine[i][2]][this->commLine[i][3]],color,2,8);
		*/
	}
	this->commLine.erase(this->commLine.begin(),this->commLine.end());
}

void World::plotTravelGraph(){
	Scalar color;
	color[0] = 0;
	color[1] = 0;
	color[2] = 0;
	for(int i=0; i<this->costmap.nRows; i++){
		for(int j=0; j<this->costmap.nCols; j++){
			if(this->costmap.cells[i][j] == 0){ // are they traversable?

				for(int k=0; k<2; k++){
					if(!this->costmap.cells[i][j+k] == 0){
						// TODO fix world pointmap
						/*
						line(this->image,this->pointMap[i][j],this->pointMap[i][j+k],color,1,8);
						*/
					}
				}
			}
		}
	}
}

void World::plotFrontierGraph(){
	Scalar color;
	color[0] = 0;
	color[1] = 0;
	color[2] = 255;
	//for(int i=0; i<(int)this->frntList.size(); i++){
		//circle(this->image,this->pointMap[this->frntList[i][0]][this->frntList[i][1]],this->gSpace,color,-1);
	//}
}

Mat World::createExplImage(){
	Scalar color;
	color[0] = 255;
	color[1] = 255;
	color[2] = 255;
	Mat skel(this->image.rows,this->image.cols,CV_8UC1,Scalar(0));
	for(int i=0; i<this->costmap.nRows; i++){
		for(int j=0; j<this->costmap.nCols; j++){
			if(this->costmap.cells[i][j] == 0){ // are they traversable?

				// TODO fix world pointmap
				/*
				Point a;
				a.x = this->pointMap[i][j].x + this->gSpace;
				a.y = this->pointMap[i][j].y + this->gSpace;
				rectangle(skel,this->pointMap[i][j],a,color,-1);
				*/
			}
		}
	}
	return(skel);
}

void World::plotPath(vector<vector<int> > myPath, int myColor[3], int pathIndex){
	Scalar color;
	color[0] = myColor[0];
	color[1] = myColor[1];
	color[2] = myColor[2];
	//for(int i=pathIndex; i<(int)myPath.size(); i++){
	//	line(this->image,this->pointMap[myPath[i-1]],this->pointMap[myPath[i]],Scalar{myColor[0],myColor[1],myColor[2]},3,8);
	//}
	// TODO fix world pointmap
	/*
	circle(this->image, this->pointMap[myPath[pathIndex][0]][myPath[pathIndex][1]], 10, color, -1);
	*/
	//circle(this->image,this->pointMap[myPath[myPath.size()-1]],10,Scalar{myColor[0],myColor[1],myColor[2]},-1);
	//circle(this->image,this->pointMap[myPath[myPath.size()-1]],5,Scalar{0,0,0},-1);
}

void World::plotMap(){
	Mat temp;
	resize(this->image,temp,Size(),1,1,CV_INTER_AREA);
	imshow("Global Map", temp);
	waitKey(1);
}

World::~World() {

}
