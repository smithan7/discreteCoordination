/*
 * World.cpp
 *
 *  Created on: Mar 28, 2016
 *      Author: andy
 */

#include "World.h"

using namespace cv;
using namespace std;

World::World(int gSpace, float obsThresh, float comThresh) {
	this->gSpace = gSpace;
	this->obsThresh = obsThresh;
	this->commThresh = comThresh;
	string fName = "test2";
	this->fileName = fName + ".jpg";

	cerr << this->fileName << endl;
	//this->fileName = "/home/andy/Dropbox/workspace/nextGenCoord/test2.jpg";

	this->image = imread(this->fileName,1);
	cvtColor(this->image,this->imgGray,CV_BGR2GRAY);
	threshold(this->image,this->imgGray,250,255,THRESH_BINARY);


	this->nRows = imgGray.rows;
	this->nCols = imgGray.cols;
	cerr << "nRows ? nCols: " << nRows << " ? " << nCols << endl;
	/*
	this->initializeMaps();
	this->getDistGraph();
	this->nRows = this->costMap.size();
	this->nCols = this->costMap[0].size();
	this->getObsGraph();
	//this->getCommGraph();

	cerr << "into saveWorld" << endl;
	this->saveWorldToYML();
	*/
	this->pullWorldFromYML(fName);
	this->getDistGraph();
	this->nRows = this->costMap.size();
	this->nCols = this->costMap[0].size();
	cerr << "nRows ? nCols: " << nRows << " ? " << nCols << endl;
	waitKey(0);

}

void World::saveWorldToYML(){
	string filename =  "test0.yml";
	FileStorage fs(filename, FileStorage::WRITE);

	fs << "costMap" << "[";
	for(size_t i=0; i<this->costMap.size(); i++){
		fs << this->costMap[i];
	}
	fs << "]";

	fs << "pointMap" << "[";
	for(size_t i=0; i<this->pointMap.size(); i++){
		fs << this->pointMap[i];
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
	fsN["costMap"] >> this->costMap;
	fsN["pointMap"] >> this->pointMap;


	this->nRows = this->costMap.size();
	this->nCols = this->costMap[0].size();
	this->gSpace = 5;

	vector<vector<int> > tObsGraph;

	fsN["obsGraph"] >> tObsGraph;

	cerr << "tObsGraph.size(): " << tObsGraph.size() << " < " << tObsGraph[0].size() << endl;

	for(int i=0; i<nRows; i++){
		vector<vector<vector<int> > > c;
		for(int j=0; j<nCols; j++){
			vector<vector<int> > a;
			for(size_t k=0; k<tObsGraph[this->nRows*i+j].size(); k = k+2){
				vector<int> b;
				b.push_back(tObsGraph[this->nRows*i+j][k]);
				b.push_back(tObsGraph[this->nRows*i+j][k+1]);
				a.push_back(b);
			}
			c.push_back(a);
		}
		this->obsGraph.push_back(c);
	}

	cerr << "obsGraph.size(): " << obsGraph.size() << " < " << obsGraph[0].size() << endl;


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

void World::initializeMaps(){
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

float World::getEuclidDist(int x0, int y0, int x1, int y1){

	int bx = abs(x0 - x1);
	int by = abs(y0 - y1);

	return(this->distGraph[bx][by]);
}


void World::getDistGraph(){
	for(int i=0; i<this->nCols; i++){
		vector<float> d;
		for(int j=0; j<this->nRows; j++){
			d.push_back(sqrt(pow(i,2) + pow(j,2)));
		}
		this->distGraph.push_back(d);
	}
}

void World::getObsGraph(){
	for(int i=0; i<this->nRows; i++){
		vector<vector<vector<int> > > to; // [yLoc][list][x/y]
		for(int j=0; j<this->nCols; j++){
			vector<vector<int> > too; // [list][x/y]
			to.push_back(too);
		}
		this->obsGraph.push_back(to);
	}

	//cerr << "costMap.size() / costMap[0].size(): " << this->costMap.size() << " / " << this->costMap[0].size() << endl;
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
							//cerr << " is traversable*" << endl;
							//cerr << "i,j,k,l: " << i << ", " << j << ", " << k << ", " << l << endl;
							//cerr << "pointMap.size() / pointMap[0].size(): " << this->pointMap.size() << " / " << this->pointMap[0].size() << endl;
							//cerr << this->pointMap[i][j].x << " / " << this->pointMap[i][j].y << endl;
							//cerr << this->pointMap[k][l].x << " / " << this->pointMap[k][l].y << endl;

							//cerr << "distGraph.size() / distGraph[0].size(): " << this->distGraph.size() << " / " << this->distGraph[0].size() << endl;
							float dist = this->getEuclidDist(this->pointMap[i][j].y,this->pointMap[i][j].x,this->pointMap[k][l].y,this->pointMap[k][l].x);
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

void World::getCommGraph(){

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

void World::clearPlot(){
	this->image = imread(this->fileName);
	cvtColor(this->image,this->imgGray,CV_BGR2GRAY);
	threshold(this->imgGray,this->imgGray,127,255,THRESH_BINARY);
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
		line(this->image,this->pointMap[this->commLine[i][0]][this->commLine[i][1]],this->pointMap[this->commLine[i][2]][this->commLine[i][3]],color,2,8);
	}
	this->commLine.erase(this->commLine.begin(),this->commLine.end());
}

void World::plotTravelGraph(){
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

void World::plotPath(vector<vector<int> > myPath, int myColor[3], int pathIndex){
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

void World::plotMap(){
	Mat temp;
	resize(this->image,temp,Size(),1,1,CV_INTER_AREA);
	imshow("Global Map", temp);
	waitKey(1);
}

World::~World() {
	// TODO Auto-generated destructor stub
}
