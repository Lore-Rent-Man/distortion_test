#include <iostream>
#include <fstream>
#include <float.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <Eigen/Core>
#include <pangolin/pangolin.h>
#include <pangolin/display/display.h>
#include <pangolin/gl/gldraw.h>
#include <pangolin/gl/glvbo.h>
#include <pangolin/gl/glsl.h>
#include <unistd.h>
#include <cstdlib>
#include <filesystem>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/statistical_outlier_removal.h>

using namespace std;
using namespace Eigen;
namespace fs = filesystem;

const std::string my_shader = R"Shader(
@start vertex
#version 330
attribute vec4 a_position;
varying float a_gray;
uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;
void main() {
    gl_Position = projection * view * model * vec4(a_position[0], a_position[1], a_position[2], 1.0);
    a_gray = a_position[3];
}
@start fragment
#version 330
varying float a_gray;
void main() {
    gl_FragColor = vec4(a_gray, a_gray, a_gray, 1.0);
}
)Shader";

vector<double> motor_angles;
vector<string> images;

void showPointCloud(
	const vector<Vector4d> &pointcloud, 
	const vector<Vector4d> &roomvertices, 
	unsigned int* indices);

double room_width = 5.0;
double room_length = 5.0;
double room_height = 5.0;

string pose_file = "./image_depth_record_mvsec4/gt_pose_mvsec_fly4.txt";
string img_file = "./image_depth_record_mvsec4/rgbd_timestamp_mvsec4_full.txt";
string img_folder = "./image_depth_record_mvsec4/image_depth_record_mvsec4/";

int main(int argc, char **argv) {
	
	//Depth Camera intrinsics
	double fx = 318.9285888671875, fy = 318.9285888671875, cx = 317.31512451171875, cy = 181.547119140625;
	
	ifstream fposes(pose_file);
	ifstream fimages(img_file);

	if (!fposes || !fimages) {
		cout << "Failed to load file(s)" << endl;
		return 1;
	}

	vector<string> image_timestamps;
	string timestamp;
	while (getline(fimages, timestamp))
		image_timestamps.push_back(timestamp);

	vector<Isometry3d> poses;
	
	double prev_time = DBL_MAX;
	double time = DBL_MAX;
	Isometry3d prev_pose = Isometry3d::Identity();
	Isometry3d pose = Isometry3d::Identity();
	for(int i = 0; i < image_timestamps.size(); i++)
	{
		while (!fposes.eof()) {
			if (abs(prev_time - stod(image_timestamps[i])) < abs(time - stod(image_timestamps[i]))) {
				poses.push_back(prev_pose);
				break;
			}
			else
			{
				prev_time = time;
				prev_pose = pose;
				double tx, ty, tz, qx, qy, qz, qw;
                        	fposes >> time >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
                        	Isometry3d Twr(Quaterniond(qw, qx, qy, qz));
                        	Twr.pretranslate(Vector3d(tx, ty, tz));
				pose = Twr;
			}
		}
	}

	for (int i = 0; i < poses.size(); i++){
		cout << poses[i].matrix() << endl;
	}

	vector<Vector4d> pointcloud;

	typedef pcl::PointXYZI PointT;
	typedef pcl::PointCloud<PointT> PointCloud;

	PointCloud::Ptr pointCloud(new PointCloud);

	for (int i = 0; i < image_timestamps.size(); i++) {
		PointCloud::Ptr current(new PointCloud);
		cv::Mat img = cv::imread(img_folder + image_timestamps[i] + ".png", 0);
		cout << "./image_depth_record_mvsec4/image_depth_record_mvsec4/" + image_timestamps[i] + ".png" << endl;
		for (int v = 0; v < img.rows; v++) 
			for (int u = 0; u < img.cols; u++) 
			{
				double x = (u - cx)/fx;
				double y = (v - cy)/fy;
				double depth = img.at<unsigned short>(v, u) * 0.001;
				
				if (depth == 0) continue;

				Vector3d point(x, y, depth);
				Vector3d p = poses[i] * point;
				pointcloud.push_back(Vector4d(p[0], p[1], p[2], 0.75));
				PointT p1(p[0], p[1], p[2], 0.5);
				current->points.push_back(p1);
			}
		PointCloud::Ptr tmp(new PointCloud);
		pcl::StatisticalOutlierRemoval<PointT> statistical_filter;
		statistical_filter.setMeanK(50);
		statistical_filter.setStddevMulThresh(1.0);
		statistical_filter.setInputCloud(current);
		statistical_filter.filter(*tmp);
		(*pointCloud) += *current;
	}

	pointCloud->is_dense = false;
	
	cout << pointcloud.size() << endl;
	cout << pointCloud->size() << endl;

	//pcl::VoxelGrid<PointT> voxel_filter;
	//double resolution = 0.03;

	//voxel_filter.setLeafSize(resolution, resolution, resolution);
	//PointCloud::Ptr tmp(new PointCloud);
	//voxel_filter.setInputCloud(pointCloud);
	//voxel_filter.filter(*tmp);
	//tmp->swap(*pointCloud);
	
	cout << pointCloud->size() << endl;

	pcl::io::savePCDFileBinary("map.pcd", *pointCloud);

	vector<Vector4d> roomvertices
	{
		Vector4d(0, room_length, room_height, 0.5), 
		Vector4d(room_width, room_length, room_height, 0.5),
		Vector4d(room_width, 0, room_height, 0.5),
		Vector4d(0, 0, room_height, 0.5),
		Vector4d(0, room_length, 0, 0.5),
		Vector4d(room_width, room_length, 0, 0.5),
		Vector4d(room_width, 0, 0, 0.5),
		Vector4d(0, 0, 0, 0.5)
	};

	unsigned int indices[] 
	{
		0,3,2,  //Front
		2,1,0,
		1,5,6,	//Right
		6,2,1,
		5,4,7,	//Left
		7,6,5,
		4,7,3,	//Back
		3,0,4,
		4,5,1,	//Top
		1,0,4,
		3,2,6,	//Bottom
		6,7,3
	};

	//Adds grid-like feature for easier checking
	for (int i = 0; i <= room_width; i++)
	{
		for (int j = 0; j <= room_length; j++)
		{
			for (int k = 0; k <= room_height; k++)
			{
				Vector4d point(i, j, k, 0.5);
				pointcloud.push_back(point);
			}
		}
	}
	
	//showPointCloud(pointcloud, roomvertices, indices);
}

void showPointCloud(const vector<Vector4d> &pointcloud, const vector<Vector4d> &roomvertices, unsigned int* indices) {
	cout << "Constructing Point Cloud" << endl;

	pangolin::CreateWindowAndBind("Main", 1024, 768);

	if (pointcloud.empty()) {
		cerr << "Point Cloud is empty!" << endl;
		return;
	}

	pangolin::GlBuffer vbo(pangolin::GlArrayBuffer, pointcloud);
	pangolin::GlBuffer vbo2(pangolin::GlArrayBuffer, roomvertices);

	pangolin::GlBuffer ibo;
        ibo.Reinitialise(pangolin::GlElementArrayBuffer, 36, GL_UNSIGNED_INT, 1, GL_STATIC_DRAW);
        ibo.Upload(indices, sizeof(unsigned int) * 36);
	                
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);


	pangolin::OpenGlRenderState s_cam(
		pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
		pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
	);

	pangolin::View &d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
        .SetHandler(new pangolin::Handler3D(s_cam));
	
	pangolin::GlSlProgram prog;
    	prog.AddShader( pangolin::GlSlAnnotatedShader, my_shader );
    	prog.Link();
    	prog.Bind();

	while (pangolin::ShouldQuit() == false) {
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		prog.SetUniform("projection", s_cam.GetProjectionMatrix());
		prog.SetUniform("view", s_cam.GetModelViewMatrix());
		prog.SetUniform("model", pangolin::IdentityMatrix());
        	glPointSize(1);
		pangolin::RenderVbo(vbo, GL_POINTS);
		
		vbo2.Bind();
		glVertexPointer(vbo2.count_per_element, vbo2.datatype, 0, 0);
    		glEnableClientState(GL_VERTEX_ARRAY);
        	ibo.Bind();
        	glDrawElements(GL_TRIANGLES, ibo.num_elements, ibo.datatype, 0);
        	ibo.Unbind();
		glDisableClientState(GL_VERTEX_ARRAY);
    		vbo2.Unbind();

		d_cam.Activate(s_cam);
                glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
		
		pangolin::FinishFrame();
		usleep(5000);   // sleep 5 ms
    	}
    return;
}

