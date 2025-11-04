#include "utility.h"
#include <fast_gicp/gicp/fast_gicp.hpp>
#include <pcl/filters/approximate_voxel_grid.h>
#include "liorf_localization/cloud_info.h"
#include "liorf_localization/save_map.h"

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>

#include <gtsam/nonlinear/ISAM2.h>

using namespace gtsam;

using symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
using symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)
using symbol_shorthand::G; // GPS pose

/*
    * A point cloud type that has 6D pose info ([x,y,z,roll,pitch,yaw] intensity is time stamp)
    */
struct PointXYZIRPYT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;                  // preferred way of adding a XYZ+padding
    float roll;
    float pitch;
    float yaw;
    double time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW   // make sure our new allocators are aligned
} EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIRPYT,
                                   (float, x, x) (float, y, y)
                                   (float, z, z) (float, intensity, intensity)
                                   (float, roll, roll) (float, pitch, pitch) (float, yaw, yaw)
                                   (double, time, time))

typedef PointXYZIRPYT  PointTypePose;


class mapOptimization : public ParamServer
{

public:

    // gtsam
    NonlinearFactorGraph gtSAMgraph;
    Values initialEstimate;
    Values optimizedEstimate;
    ISAM2 *isam;
    Values isamCurrentEstimate;
    Eigen::MatrixXd poseCovariance;

    ros::Publisher pubLaserCloudSurround;
    ros::Publisher pubLaserOdometryGlobal;
    ros::Publisher pubBaseOdometryGlobal;
    ros::Publisher pubLaserOdometryIncremental;
    ros::Publisher pubKeyPoses;
    ros::Publisher pubPath;
    
    // ros::Publisher pose_pub_;

    ros::Publisher pubHistoryKeyFrames;
    ros::Publisher pubIcpKeyFrames;
    ros::Publisher pubRecentKeyFrames;
    ros::Publisher pubRecentKeyFrame;
    ros::Publisher pubGlobalMap;
    ros::Publisher pubCloudRegisteredRaw;
    ros::Publisher pubLoopConstraintEdge;

    ros::Publisher pubSLAMInfo;
    ros::Publisher pubLocalizationState;
    ros::Publisher pubrelocInitials;

    ros::Subscriber subCloud;
    ros::Subscriber subGPS;
    ros::Subscriber subLoop;

    ros::ServiceServer srvSaveMap;

    std::deque<nav_msgs::Odometry> gpsQueue;
    liorf_localization::cloud_info cloudInfo;

    vector<pcl::PointCloud<PointType>::Ptr> surfCloudKeyFrames;
    
    pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D;
    pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D;
    pcl::PointCloud<PointType>::Ptr copy_cloudKeyPoses3D;
    pcl::PointCloud<PointTypePose>::Ptr copy_cloudKeyPoses6D;

    // riboha
    pcl::PointCloud<PointType>::Ptr loadedKeyframePoses3D;    
    pcl::KdTreeFLANN<PointType>::Ptr loadedKeyframePoses3DTree;
    pcl::PointCloud<PointTypePose>::Ptr loadedKeyframePoses6D;


    pcl::PointCloud<PointType>::Ptr laserCloudSurfLast; // surf feature set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLastDS; // downsampled surf feature set from odoOptimization

    pcl::PointCloud<PointType>::Ptr laserCloudOri;
    pcl::PointCloud<PointType>::Ptr coeffSel;

    std::vector<PointType> laserCloudOriSurfVec; // surf point holder for parallel computation
    std::vector<PointType> coeffSelSurfVec;
    std::vector<bool> laserCloudOriSurfFlag;

    map<int, pair<pcl::PointCloud<PointType>, pcl::PointCloud<PointType>>> laserCloudMapContainer;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMapDS;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurroundingKeyPoses;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeHistoryKeyPoses;

    pcl::VoxelGrid<PointType> downSizeFilterSurf;
    pcl::VoxelGrid<PointType> downSizeFilterLocalMapSurf;
    pcl::VoxelGrid<PointType> downSizeFilterICP;
    pcl::VoxelGrid<PointType> downSizeFilterSurroundingKeyPoses; // for surrounding key poses of scan-to-map optimization
    
    ros::Time timeLaserInfoStamp;
    double timeLaserInfoCur;

    float transformTobeMapped[6];
    float transformTobeMapped_prev[6];

    std::mutex mtx;
    std::mutex mtxLoopInfo;

    bool isDegenerate = false;
    cv::Mat matP;

    int laserCloudSurfFromMapDSNum = 0;
    int laserCloudSurfLastDSNum = 0;

    bool aLoopIsClosed = false;
    map<int, int> loopIndexContainer; // from new to old
    vector<pair<int, int>> loopIndexQueue;
    vector<gtsam::Pose3> loopPoseQueue;
    vector<gtsam::noiseModel::Diagonal::shared_ptr> loopNoiseQueue;
    deque<std_msgs::Float64MultiArray> loopInfoVec;

    nav_msgs::Path globalPath;

    Eigen::Affine3f transPointAssociateToMap;
    Eigen::Affine3f incrementalOdometryAffineFront;
    Eigen::Affine3f incrementalOdometryAffineBack;

    // add by yjz_lucky_boy
    // localization
    ros::Subscriber sub_initial_pose;
    bool has_global_map = false;
    bool has_initialize_pose = false;
    bool system_initialized = false;
    float initialize_pose[6];

    // riboha
    fast_gicp::FastGICP<pcl::PointXYZ, pcl::PointXYZ> gicp;
    pcl::ApproximateVoxelGrid<pcl::PointXYZ> gicpMapVoxelGrid;
    pcl::ApproximateVoxelGrid<pcl::PointXYZ> gicpSourceVoxelGrid;
    pcl::PointCloud<pcl::PointXYZ>::Ptr GICPGlobalMap;
    pcl::PointCloud<pcl::PointXYZ>::Ptr laserCloudRaw;
    Eigen::Affine3d manualInitialGuess_;
    Eigen::Isometry3d manualInitialGuess;
    Eigen::Isometry3d relocInitialGuess;
    geometry_msgs::PoseWithCovarianceStamped manualInitialGuessMsg;
    tf::StampedTransform lidar2Baselink;
    std::vector<Eigen::Vector4d> laserCloudSurfFromMapDSPlaneCoefficients;
    std::vector<bool> laserCloudSurfFromMapDSValid;
    double timeLastProcessing = -1;
    int locFailedIter = 0;
    bool needRelocalization = false;

    // debugging
    double totalProcessingTime = 0.0;
    int processingCount = 0;
    bool localization_successful = true;
    double prev_scan_matching_vel = -1.0;

    mapOptimization()
    {
        ISAM2Params parameters;
        parameters.relinearizeThreshold = 0.1;
        parameters.relinearizeSkip = 1;
        isam = new ISAM2(parameters);

        pubKeyPoses                 = nh.advertise<sensor_msgs::PointCloud2>("liorf_localization/mapping/trajectory", 1);
        pubLaserCloudSurround       = nh.advertise<sensor_msgs::PointCloud2>("liorf_localization/mapping/map_global", 1);
        pubLaserOdometryGlobal      = nh.advertise<nav_msgs::Odometry> ("liorf_localization/mapping/odometry", 1);
        pubBaseOdometryGlobal       = nh.advertise<nav_msgs::Odometry> ("liorf_localization/mapping/odometry_base", 1);
        pubLaserOdometryIncremental = nh.advertise<nav_msgs::Odometry> ("liorf_localization/mapping/odometry_incremental", 1);
        pubPath                     = nh.advertise<nav_msgs::Path>("liorf_localization/mapping/path", 1);
        
        // pose_pub_ = nh.advertise<geometry_msgs::PoseWithCovarianceStamped>("/unitree/pose_with_covariance", 10);


        subCloud = nh.subscribe<liorf_localization::cloud_info>("liorf_localization/deskew/cloud_info", 1, &mapOptimization::laserCloudInfoHandler, this, ros::TransportHints().tcpNoDelay());
        subGPS   = nh.subscribe<nav_msgs::Odometry> (gpsTopic, 200, &mapOptimization::gpsHandler, this, ros::TransportHints().tcpNoDelay());
        subLoop  = nh.subscribe<std_msgs::Float64MultiArray>("lio_loop/loop_closure_detection", 1, &mapOptimization::loopInfoHandler, this, ros::TransportHints().tcpNoDelay());

        // srvSaveMap  = nh.advertiseService("liorf_localization/save_map", &mapOptimization::saveMapService, this);

        pubHistoryKeyFrames   = nh.advertise<sensor_msgs::PointCloud2>("liorf_localization/mapping/icp_loop_closure_history_cloud", 1);
        pubIcpKeyFrames       = nh.advertise<sensor_msgs::PointCloud2>("liorf_localization/mapping/icp_loop_closure_corrected_cloud", 1);
        pubLoopConstraintEdge = nh.advertise<visualization_msgs::MarkerArray>("/liorf_localization/mapping/loop_closure_constraints", 1);

        pubRecentKeyFrames    = nh.advertise<sensor_msgs::PointCloud2>("liorf_localization/mapping/map_local", 1);
        pubRecentKeyFrame     = nh.advertise<sensor_msgs::PointCloud2>("liorf_localization/mapping/cloud_registered", 1);
        pubCloudRegisteredRaw = nh.advertise<sensor_msgs::PointCloud2>("liorf_localization/mapping/cloud_registered_raw", 1);
        pubGlobalMap          = nh.advertise<sensor_msgs::PointCloud2>("liorf_localization/localization/global_map", 1);

        pubSLAMInfo           = nh.advertise<liorf_localization::cloud_info>("liorf_localization/mapping/slam_info", 1);
        pubLocalizationState  = nh.advertise<std_msgs::Bool>("liorf_localization/localization/localization_state", 1);
        pubrelocInitials      = nh.advertise<visualization_msgs::MarkerArray>("/liorf_localization/mapping/reloc_candidates", 1);

        sub_initial_pose      = nh.subscribe("/initialpose", 10, &mapOptimization::initialposeHandler, this);

        downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
        downSizeFilterLocalMapSurf.setLeafSize(surroundingKeyframeMapLeafSize, surroundingKeyframeMapLeafSize, surroundingKeyframeMapLeafSize);
        downSizeFilterICP.setLeafSize(loopClosureICPSurfLeafSize, loopClosureICPSurfLeafSize, loopClosureICPSurfLeafSize);
        downSizeFilterSurroundingKeyPoses.setLeafSize(surroundingKeyframeDensity, surroundingKeyframeDensity, surroundingKeyframeDensity); // for surrounding key poses of scan-to-map optimization

        allocateMemory();
        loadGlobalMap();
    }

    void allocateMemory()
    {
        cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());
        copy_cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        copy_cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());

        loadedKeyframePoses3D.reset(new pcl::PointCloud<PointType>());
        loadedKeyframePoses6D.reset(new pcl::PointCloud<PointTypePose>());
        loadedKeyframePoses3DTree.reset(new pcl::KdTreeFLANN<PointType>());

        kdtreeSurroundingKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());

        laserCloudSurfLast.reset(new pcl::PointCloud<PointType>()); // surf feature set from odoOptimization
        laserCloudSurfLastDS.reset(new pcl::PointCloud<PointType>()); // downsampled surf featuer set from odoOptimization

        laserCloudOri.reset(new pcl::PointCloud<PointType>());
        coeffSel.reset(new pcl::PointCloud<PointType>());

        laserCloudOriSurfVec.resize(N_SCAN * Horizon_SCAN);
        coeffSelSurfVec.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriSurfFlag.resize(N_SCAN * Horizon_SCAN);

        std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);

        laserCloudSurfFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMapDS.reset(new pcl::PointCloud<PointType>());

        kdtreeSurfFromMap.reset(new pcl::KdTreeFLANN<PointType>());

        for (int i = 0; i < 6; ++i){
            transformTobeMapped[i] = 0;
            transformTobeMapped_prev[i] = 0;
        }

        matP = cv::Mat(6, 6, CV_32F, cv::Scalar::all(0));

        // riboha
        gicp.setMaxCorrespondenceDistance(1.0);
        gicp.setNumThreads(8);
        gicp.setCorrespondenceRandomness(20);

        gicpSourceVoxelGrid.setLeafSize(0.2, 0.2, 0.2);
        laserCloudRaw.reset(new pcl::PointCloud<pcl::PointXYZ>());

        if(lidarFrame != baselinkFrame)
        {
            // tf: base to lidar (from params.yaml)
            // trans
            tf::Vector3 T_b2l(b2LTrans(0), b2LTrans(1), b2LTrans(2));
            lidar2Baselink.setOrigin(T_b2l);
            // rot
            tf::Matrix3x3 R_b2l(
                b2LRot(0,0), b2LRot(0,1), b2LRot(0,2),
                b2LRot(1,0), b2LRot(1,1), b2LRot(1,2),
                b2LRot(2,0), b2LRot(2,1), b2LRot(2,2));
            lidar2Baselink.setBasis(R_b2l);
            lidar2Baselink.setData(lidar2Baselink.inverse());
            // lidar2Baselink = lidar2Baselink.inverse();
        }
        
    }

    // add by yjz_lucky_boy
    void loadGlobalMap()
    {
        std::string global_map = savePCDDirectory;
        if (pcl::io::loadPLYFile<PointType>(global_map + "map.ply", *laserCloudSurfFromMap) < 0) {
            ROS_WARN_STREAM("Couldn't read PLY map: " << (global_map + "map.ply"));
            return;
        }

        // vis map
        downSizeFilterLocalMapSurf.setLeafSize(surroundingKeyframeMapLeafSize*3., surroundingKeyframeMapLeafSize*3., surroundingKeyframeMapLeafSize*2.);
        downSizeFilterLocalMapSurf.setInputCloud(laserCloudSurfFromMap);
        downSizeFilterLocalMapSurf.filter(*laserCloudSurfFromMapDS);
        sleep(3);
        publishCloud(pubGlobalMap, laserCloudSurfFromMapDS, ros::Time::now(), mapFrame);   


        downSizeFilterLocalMapSurf.setLeafSize(surroundingKeyframeMapLeafSize, surroundingKeyframeMapLeafSize, surroundingKeyframeMapLeafSize);
        downSizeFilterLocalMapSurf.setInputCloud(laserCloudSurfFromMap);
        downSizeFilterLocalMapSurf.filter(*laserCloudSurfFromMapDS);
        laserCloudSurfFromMapDSNum = laserCloudSurfFromMapDS->size();
        std::cout << "global map size: " << laserCloudSurfFromMapDSNum << std::endl;

        // riboha
        GICPGlobalMap.reset(new pcl::PointCloud<pcl::PointXYZ>);
        GICPGlobalMap->points.reserve(laserCloudSurfFromMap->points.size());
        for (const auto& pt : laserCloudSurfFromMap->points) {
            pcl::PointXYZ pt_xyz;
            pt_xyz.x = pt.x;
            pt_xyz.y = pt.y;
            pt_xyz.z = pt.z;
            GICPGlobalMap->points.push_back(pt_xyz);
        }
        gicpSourceVoxelGrid.setLeafSize(0.2, 0.2, 0.2);
        gicpSourceVoxelGrid.setInputCloud(GICPGlobalMap);
        pcl::PointCloud<pcl::PointXYZ>::Ptr target(new pcl::PointCloud<pcl::PointXYZ>);
        gicpSourceVoxelGrid.filter(*target);
        // gicp.setInputTarget(GICPGlobalMap);
        gicp.setInputTarget(target);

        // gicp.setInputTarget(laserCloudSurfFromMapDS);

        if (laserCloudSurfFromMapDSNum < 1000)
          return;
        
        has_global_map = true;

        kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMapDS);

        // riboha

        //// from txt
        std::ifstream fin(savePCDDirectory + "traj_lidar.txt");
        if (!fin.is_open()) {
            ROS_WARN_STREAM("Couldn't open trajectory file: " << savePCDDirectory + "traj_lidar.txt");
            return;
        }

        loadedKeyframePoses6D->clear();
        loadedKeyframePoses3D->clear();

        std::string line;
        size_t line_idx = 0;
        while (std::getline(fin, line)) {
            ++line_idx;

            // skip empty/whitespace or comment lines
            if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;
            if (line[0] == '#') continue;

            std::istringstream iss(line);
            double ts, x, y, z, qx, qy, qz, qw; // 입력은 xyzw (qw가 마지막)
            if (!(iss >> ts >> x >> y >> z >> qx >> qy >> qz >> qw)) {
                ROS_WARN_STREAM("Line " << line_idx << ": failed to parse (ts x y z qx qy qz qw), skipping.");
                continue;
            }

            // Eigen 쿼터니언은 (w,x,y,z) 순서이므로 주의!
            Eigen::Quaterniond q(qw, qx, qy, qz);
            q.normalize();
            Eigen::Matrix3d R = q.toRotationMatrix();

            // ZYX 순서의 오일러각 (yaw, pitch, roll)
            Eigen::Vector3d ypr = R.eulerAngles(2, 1, 0);
            double yaw   = ypr[0];
            double pitch = ypr[1];
            double roll  = ypr[2];

            // 6D 포즈 누적
            PointTypePose pose6d;
            pose6d.x = static_cast<float>(x);
            pose6d.y = static_cast<float>(y);
            pose6d.z = static_cast<float>(z);
            pose6d.roll  = static_cast<float>(roll);
            pose6d.pitch = static_cast<float>(pitch);
            pose6d.yaw   = static_cast<float>(yaw);

            // (선택) 포즈 구조체에 타임스탬프 필드가 있다면 여기에 할당
            // pose6d.time = ts;

            loadedKeyframePoses6D->push_back(pose6d);

            // 3D 포인트(검색용)
            PointType p3d;
            p3d.x = static_cast<float>(x);
            p3d.y = static_cast<float>(y);
            p3d.z = static_cast<float>(z);
            loadedKeyframePoses3D->push_back(p3d);

            // (선택) 별도 타임스탬프 컨테이너를 쓰고 있다면 여기에 push_back(ts);
        }

        loadedKeyframePoses3DTree->setInputCloud(loadedKeyframePoses3D);

        std::cout << "Loaded keyframe poses: " << loadedKeyframePoses3D->size() << std::endl;


        pre_compute_target_coefficients();

        // check
        int num_valid = 0;
        for (int i = 0; i < laserCloudSurfFromMapDSNum; i++) {
            if (laserCloudSurfFromMapDSValid[i]) {
                num_valid++;
            }
        }

        std::cout << num_valid << std::endl;
        std::cout << laserCloudSurfFromMapDSNum << std::endl;
    }

    // riboha
    // pre-compute coefficients of the target points
    // !!! run this function after loading global map
    void pre_compute_target_coefficients()
    {   // laserCloudSurfFromMapDS, kdtreeSurfFromMap, laserCloudSurfFromMapDSValid
        std::cout << "\033[1;32m" << "Calculating prior map coefficients... " << "\033[0m" << std::endl;
        laserCloudSurfFromMapDSPlaneCoefficients.clear();
        laserCloudSurfFromMapDSPlaneCoefficients.resize(laserCloudSurfFromMapDSNum);
        laserCloudSurfFromMapDSValid.clear();
        laserCloudSurfFromMapDSValid.resize(laserCloudSurfFromMapDSNum);

        int k_correspondences_ = 5;

        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < laserCloudSurfFromMapDSNum; i++) 
        {
            PointType pointSource;
            pointSource = laserCloudSurfFromMapDS->points[i];
            std::vector<int> k_indices;
            std::vector<float> k_sq_distances;
            kdtreeSurfFromMap->nearestKSearch(pointSource, k_correspondences_, k_indices, k_sq_distances);

            if (k_sq_distances[k_correspondences_-1] < 1.0) {
                Eigen::Matrix<float, 5, 3> matA0;
                Eigen::Matrix<float, 5, 1> matB0;
                Eigen::Vector3f matX0;

                matA0.setZero();
                matB0.fill(-1);
                matX0.setZero();

                for (int j = 0; j < k_correspondences_; j++) {
                    matA0(j, 0) = laserCloudSurfFromMapDS->points[k_indices[j]].x;
                    matA0(j, 1) = laserCloudSurfFromMapDS->points[k_indices[j]].y;
                    matA0(j, 2) = laserCloudSurfFromMapDS->points[k_indices[j]].z;
                }

                matX0 = matA0.colPivHouseholderQr().solve(matB0);

                float pa = matX0(0, 0);
                float pb = matX0(1, 0);
                float pc = matX0(2, 0);
                float pd = 1;

                float ps = sqrt(pa * pa + pb * pb + pc * pc);
                pa /= ps; pb /= ps; pc /= ps; pd /= ps;


                bool planeValid = true;
                for (int j = 0; j < k_correspondences_; j++) {
                    if (fabs(pa * laserCloudSurfFromMapDS->points[k_indices[j]].x +
                             pb * laserCloudSurfFromMapDS->points[k_indices[j]].y +
                             pc * laserCloudSurfFromMapDS->points[k_indices[j]].z + pd) > 0.2) {
                        planeValid = false;
                        break;
                    }
                }

                if (planeValid) {
                    laserCloudSurfFromMapDSPlaneCoefficients[i][0] = pa;
                    laserCloudSurfFromMapDSPlaneCoefficients[i][1] = pb;
                    laserCloudSurfFromMapDSPlaneCoefficients[i][2] = pc;
                    laserCloudSurfFromMapDSPlaneCoefficients[i][3] = pd;
                    laserCloudSurfFromMapDSValid[i] = true;
                } else {
                    laserCloudSurfFromMapDSValid[i] = false;
                }

            } else {
                laserCloudSurfFromMapDSValid[i] = false;
            }

        }

        std::cout << "\033[1;32m" << "Calculating prior map coefficients done " << "\033[0m" << std::endl;
        
    }


    // riboha
    void initialposeHandler(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr &msgIn) 
    {
        manualInitialGuessMsg = *msgIn;
        manualInitialGuess_ = pcl::getTransformation(   msgIn->pose.pose.position.x,
                                                        msgIn->pose.pose.position.y,
                                                        msgIn->pose.pose.position.z,
                                                        msgIn->pose.pose.orientation.x,
                                                        msgIn->pose.pose.orientation.y,
                                                        msgIn->pose.pose.orientation.z).cast<double>();
        manualInitialGuess.translation() = manualInitialGuess_.translation();
        manualInitialGuess.linear() = manualInitialGuess_.rotation();
        has_initialize_pose = true;
        std::cout << "\033[1;32m" << "Given initial pose: " \
        << manualInitialGuess.translation().x() << " " \
        << manualInitialGuess.translation().y() << " " \
        << manualInitialGuess.translation().z() << " " << "\033[0m" << std::endl;
    }

    void laserCloudInfoHandler(const liorf_localization::cloud_infoConstPtr& msgIn)
    {
        // extract time stamp
        timeLaserInfoStamp = msgIn->header.stamp;
        timeLaserInfoCur = msgIn->header.stamp.toSec();

        // extract info and feature cloud
        cloudInfo = *msgIn;
        pcl::fromROSMsg(msgIn->cloud_deskewed, *laserCloudSurfLast);

        pcl::fromROSMsg(msgIn->cloud_deskewed, *laserCloudRaw);

        std::lock_guard<std::mutex> lock(mtx);
        localization_successful = true;

        if (timeLaserInfoCur - timeLastProcessing >= mappingProcessInterval)
        {

            if (!system_initialized)
              if(!systemInitialize())
                return;

            if (needRelocalization) {
                if(!reLocalization())
                    return;
            }

            updateInitialGuess();

            extractSurroundingKeyFrames();

            downsampleCurrentScan();

            scan2MapOptimization();

            saveKeyFramesAndFactor();

            correctPoses();

            publishOdometry();

            publishFrames();

            timeLastProcessing = timeLaserInfoCur;
            
            for (int i = 0; i < 6; ++i){
                transformTobeMapped_prev[i] = transformTobeMapped[i];
            }

            std_msgs::Bool locstatemsg;
            if (needRelocalization) {
                locstatemsg.data = false;
                pubLocalizationState.publish(locstatemsg);
            } else {
                locstatemsg.data = true;
                pubLocalizationState.publish(locstatemsg);
            }
        }
    }

    // riboha
    bool systemInitialize()
    {
        if (!has_global_map)
          return false;

        if(!has_initialize_pose)
        {
          ROS_WARN("need initilize pose from rviz.");
          return false;
        }

        std::cout << "estimating initial pose" << std::endl;
        gicp.setMaxCorrespondenceDistance(1.0);
        gicp.setNumThreads(8);
        gicpSourceVoxelGrid.setLeafSize(0.2, 0.2, 0.2);
        gicp.clearSource();
        gicpSourceVoxelGrid.setInputCloud(laserCloudRaw);
        pcl::PointCloud<pcl::PointXYZ>::Ptr source(new pcl::PointCloud<pcl::PointXYZ>);
        gicpSourceVoxelGrid.filter(*source);
        gicp.setInputSource(source);
        
        // get z, roll, pitch, yaw from nearest keyframe
        //// find nearest keyframe
        PointType pt;
        std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis;
        pt.x = manualInitialGuess(0,3);
        pt.y = manualInitialGuess(1,3);
        pt.z = manualInitialGuess(2,3);
        loadedKeyframePoses3DTree->nearestKSearch(pt, 1, pointSearchInd, pointSearchSqDis);

        // get initial guess using manually given x, y, yaw + nearest keyframe's z, roll, pitch
        Eigen::Affine3d initialGuess_;
        Eigen::Matrix4f initialGuess;
        initialGuess_ = pcl::getTransformation( manualInitialGuess(0,3),
                                                manualInitialGuess(1,3),
                                                loadedKeyframePoses3D->points[pointSearchInd[0]].z,
                                                loadedKeyframePoses6D->points[pointSearchInd[0]].roll,
                                                loadedKeyframePoses6D->points[pointSearchInd[0]].pitch,
                                                manualInitialGuessMsg.pose.pose.orientation.z).cast<double>();
        initialGuess = initialGuess_.matrix().cast<float>();

        // scan matching with G-ICP
        double min_fitness_score = 10.0;
        Eigen::Matrix4d optimalInitialPose;

        Eigen::Isometry3f yaw30 = Eigen::Isometry3f::Identity();
        yaw30.rotate(Eigen::AngleAxisf(M_PI / 3, Eigen::Vector3f::UnitZ()));

        // perform gicp while rotating current scan
        for (int j=0; j<6; j++)
        {
            if (j!=0){
                initialGuess.template block<3,3>(0,0) = yaw30.rotation() * initialGuess.block<3,3>(0,0);
                // std::cout << initialGuess.rotation() << std::endl;
            }

            // align
            pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZ>);
            gicp.align(*aligned, initialGuess);
            
            // get results
            Eigen::Matrix4d resultT = gicp.getFinalTransformation().cast<double>();
            double fitness_score = gicp.getFitnessScore();

            if (fitness_score < min_fitness_score)
            {
                min_fitness_score = fitness_score;
                optimalInitialPose = resultT;
            }
        }

        // check fitness
        if (min_fitness_score < 0.5) {
            std::cout << "\033[1;32m" << "Re-localization done with error: " << min_fitness_score << "\033[0m" << std::endl;
            transformTobeMapped[3] = optimalInitialPose(0,3);  // x
            transformTobeMapped[4] = optimalInitialPose(1,3);  // y
            transformTobeMapped[5] = optimalInitialPose(2,3);  // z

            Eigen::Matrix3d rot = optimalInitialPose.block<3, 3>(0, 0);
            Eigen::Vector3d euler_angles = rot.eulerAngles(0, 1, 2);
            transformTobeMapped[0] = euler_angles(0);
            transformTobeMapped[1] = euler_angles(1);
            transformTobeMapped[2] = euler_angles(2);
            system_initialized = true;
            return true;
        }
        else {
            std::cout << "\033[1;31m" << "Re-localization failed with error: " << min_fitness_score << "\033[0m" << std::endl;
            // system_initialized = false;
            return false;
        }
    }

    bool reLocalization()
    {
        std::cout << "\033[1;31m" << "Localizatino failed, initialize relocalization" << "\033[0m" << std::endl;

        gicp.setMaxCorrespondenceDistance(3.0);
        gicp.clearSource();

        gicpSourceVoxelGrid.setLeafSize(0.1, 0.1, 0.1);
        gicpSourceVoxelGrid.setInputCloud(laserCloudRaw);
        pcl::PointCloud<pcl::PointXYZ>::Ptr source(new pcl::PointCloud<pcl::PointXYZ>);
        gicpSourceVoxelGrid.filter(*source);

        gicp.setInputSource(source);
        
        // get z, roll, pitch, yaw from nearest keyframe
        //// find nearest keyframe
        PointType pt;
        std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis;
        pt.x = relocInitialGuess(0,3);
        pt.y = relocInitialGuess(1,3);
        pt.z = relocInitialGuess(2,3);
        loadedKeyframePoses3DTree->nearestKSearch(pt, 1, pointSearchInd, pointSearchSqDis);

        double min_fitness_score = 10.0;
        Eigen::Matrix4d optimalInitialPose;
        int selected_kf_idx = pointSearchInd[0];

        int margin = 10;
        int start_idx = max(0, selected_kf_idx-margin);
        int end_idx = min(int(loadedKeyframePoses3D->size()), selected_kf_idx+margin);




        visualization_msgs::MarkerArray markerArray;
        visualization_msgs::Marker markerNode;
        markerNode.header.frame_id = odometryFrame;
        markerNode.header.stamp = timeLaserInfoStamp;
        markerNode.action = visualization_msgs::Marker::ADD;
        markerNode.type = visualization_msgs::Marker::SPHERE_LIST;
        markerNode.ns = "loop_nodes";
        markerNode.id = 0;
        markerNode.pose.orientation.w = 1;
        markerNode.scale.x = 0.3; markerNode.scale.y = 0.3; markerNode.scale.z = 0.3; 
        markerNode.color.r = 0; markerNode.color.g = 0.8; markerNode.color.b = 1;
        markerNode.color.a = 1;
        for (int iii = start_idx; iii < end_idx; iii++) {

            geometry_msgs::Point p;
            p.x = loadedKeyframePoses3D->points[iii].x;
            p.y = loadedKeyframePoses3D->points[iii].y;
            p.z = loadedKeyframePoses3D->points[iii].z;
            markerNode.points.push_back(p);
        
        }
        markerArray.markers.push_back(markerNode);
        pubrelocInitials.publish(markerArray);



        // get initial guess using manually given x, y, yaw + nearest keyframe's z, roll, pitch
        for (int iii = start_idx; iii < end_idx; iii++) {
            std::cout << "iter " << iii << std::endl;
            Eigen::Affine3d initialGuess_;
            Eigen::Matrix4f initialGuess;
            initialGuess_ = pcl::getTransformation( loadedKeyframePoses3D->points[iii].x,
                                                    loadedKeyframePoses3D->points[iii].y,
                                                    loadedKeyframePoses3D->points[iii].z,
                                                    loadedKeyframePoses6D->points[iii].roll,
                                                    loadedKeyframePoses6D->points[iii].pitch,
                                                    loadedKeyframePoses6D->points[iii].yaw).cast<double>();
            initialGuess = initialGuess_.matrix().cast<float>();

            // scan matching with G-ICP

            Eigen::Isometry3f yaw30 = Eigen::Isometry3f::Identity();
            yaw30.rotate(Eigen::AngleAxisf(M_PI / 6, Eigen::Vector3f::UnitZ()));

            // perform gicp while rotating current scan
            // for (int j=0; j<6; j++)
            for (int j=0; j<12; j++)
            {
                if (j!=0){
                    initialGuess.template block<3,3>(0,0) = yaw30.rotation() * initialGuess.block<3,3>(0,0);
                    // std::cout << initialGuess.rotation() << std::endl;
                }

                // align
                pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZ>);
                gicp.align(*aligned, initialGuess);
                
                // get results
                Eigen::Matrix4d resultT = gicp.getFinalTransformation().cast<double>();
                double fitness_score = gicp.getFitnessScore(100.0);

                if (fitness_score < min_fitness_score)
                {
                    min_fitness_score = fitness_score;
                    optimalInitialPose = resultT;
                }
            }
        }

        // check fitness
        if (min_fitness_score < 0.3) {
            std::cout << "\033[1;32m" << "Re-localization done with error: " << min_fitness_score << "\033[0m" << std::endl;
            transformTobeMapped[3] = optimalInitialPose(0,3);  // x
            transformTobeMapped[4] = optimalInitialPose(1,3);  // y
            transformTobeMapped[5] = optimalInitialPose(2,3);  // z

            Eigen::Matrix3d rot = optimalInitialPose.block<3, 3>(0, 0);
            Eigen::Vector3d euler_angles = rot.eulerAngles(0, 1, 2);
            transformTobeMapped[0] = euler_angles(0);
            transformTobeMapped[1] = euler_angles(1);
            transformTobeMapped[2] = euler_angles(2);
            
            needRelocalization = false;
            locFailedIter = 0;
            return true;
        }
        else {
            std::cout << "\033[1;31m" << "Re-localization failed with error: " << min_fitness_score << "\033[0m" << std::endl;
            // system_initialized = false;
            return false;
        }
    }

    void gpsHandler(const nav_msgs::Odometry::ConstPtr& gpsMsg)
    {
        gpsQueue.push_back(*gpsMsg);
    }

    void pointAssociateToMap(PointType const * const pi, PointType * const po)
    {
        po->x = transPointAssociateToMap(0,0) * pi->x + transPointAssociateToMap(0,1) * pi->y + transPointAssociateToMap(0,2) * pi->z + transPointAssociateToMap(0,3);
        po->y = transPointAssociateToMap(1,0) * pi->x + transPointAssociateToMap(1,1) * pi->y + transPointAssociateToMap(1,2) * pi->z + transPointAssociateToMap(1,3);
        po->z = transPointAssociateToMap(2,0) * pi->x + transPointAssociateToMap(2,1) * pi->y + transPointAssociateToMap(2,2) * pi->z + transPointAssociateToMap(2,3);
        po->intensity = pi->intensity;
    }

    pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose* transformIn)
    {
        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

        int cloudSize = cloudIn->size();
        cloudOut->resize(cloudSize);

        Eigen::Affine3f transCur = pcl::getTransformation(transformIn->x, transformIn->y, transformIn->z, transformIn->roll, transformIn->pitch, transformIn->yaw);
        
        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < cloudSize; ++i)
        {
            const auto &pointFrom = cloudIn->points[i];
            cloudOut->points[i].x = transCur(0,0) * pointFrom.x + transCur(0,1) * pointFrom.y + transCur(0,2) * pointFrom.z + transCur(0,3);
            cloudOut->points[i].y = transCur(1,0) * pointFrom.x + transCur(1,1) * pointFrom.y + transCur(1,2) * pointFrom.z + transCur(1,3);
            cloudOut->points[i].z = transCur(2,0) * pointFrom.x + transCur(2,1) * pointFrom.y + transCur(2,2) * pointFrom.z + transCur(2,3);
            cloudOut->points[i].intensity = pointFrom.intensity;
        }
        return cloudOut;
    }

    gtsam::Pose3 pclPointTogtsamPose3(PointTypePose thisPoint)
    {
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(double(thisPoint.roll), double(thisPoint.pitch), double(thisPoint.yaw)),
                                  gtsam::Point3(double(thisPoint.x),    double(thisPoint.y),     double(thisPoint.z)));
    }

    gtsam::Pose3 trans2gtsamPose(float transformIn[])
    {
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(transformIn[0], transformIn[1], transformIn[2]), 
                                  gtsam::Point3(transformIn[3], transformIn[4], transformIn[5]));
    }

    Eigen::Affine3f pclPointToAffine3f(PointTypePose thisPoint)
    { 
        return pcl::getTransformation(thisPoint.x, thisPoint.y, thisPoint.z, thisPoint.roll, thisPoint.pitch, thisPoint.yaw);
    }

    Eigen::Affine3f trans2Affine3f(float transformIn[])
    {
        return pcl::getTransformation(transformIn[3], transformIn[4], transformIn[5], transformIn[0], transformIn[1], transformIn[2]);
    }

    PointTypePose trans2PointTypePose(float transformIn[])
    {
        PointTypePose thisPose6D;
        thisPose6D.x = transformIn[3];
        thisPose6D.y = transformIn[4];
        thisPose6D.z = transformIn[5];
        thisPose6D.roll  = transformIn[0];
        thisPose6D.pitch = transformIn[1];
        thisPose6D.yaw   = transformIn[2];
        return thisPose6D;
    }

    













    bool saveMapService(liorf_localization::save_mapRequest& req, liorf_localization::save_mapResponse& res)
    {
      string saveMapDirectory;

      cout << "****************************************************" << endl;
      cout << "Saving map to pcd files ..." << endl;
      if(req.destination.empty()) saveMapDirectory = std::getenv("HOME") + savePCDDirectory;
      else saveMapDirectory = std::getenv("HOME") + req.destination;
      cout << "Save destination: " << saveMapDirectory << endl;
      // create directory and remove old files;
      int unused = system((std::string("exec rm -r ") + saveMapDirectory).c_str());
      unused = system((std::string("mkdir -p ") + saveMapDirectory).c_str());
      // save key frame transformations
      pcl::io::savePCDFileBinary(saveMapDirectory + "/trajectory.pcd", *cloudKeyPoses3D);
      pcl::io::savePCDFileBinary(saveMapDirectory + "/transformations.pcd", *cloudKeyPoses6D);
      // extract global point cloud map

      pcl::PointCloud<PointType>::Ptr globalSurfCloud(new pcl::PointCloud<PointType>());
      pcl::PointCloud<PointType>::Ptr globalSurfCloudDS(new pcl::PointCloud<PointType>());
      pcl::PointCloud<PointType>::Ptr globalMapCloud(new pcl::PointCloud<PointType>());
      for (int i = 0; i < (int)cloudKeyPoses3D->size(); i++) {
          *globalSurfCloud   += *transformPointCloud(surfCloudKeyFrames[i],    &cloudKeyPoses6D->points[i]);
          cout << "\r" << std::flush << "Processing feature cloud " << i << " of " << cloudKeyPoses6D->size() << " ...";
      }

      if(req.resolution != 0)
      {
        cout << "\n\nSave resolution: " << req.resolution << endl;
        // down-sample and save surf cloud
        downSizeFilterSurf.setInputCloud(globalSurfCloud);
        downSizeFilterSurf.setLeafSize(req.resolution, req.resolution, req.resolution);
        downSizeFilterSurf.filter(*globalSurfCloudDS);
        pcl::io::savePCDFileBinary(saveMapDirectory + "/SurfMap.pcd", *globalSurfCloudDS);
      }
      else
      {

        // save surf cloud
        pcl::io::savePCDFileBinary(saveMapDirectory + "/SurfMap.pcd", *globalSurfCloud);
      }

      // save global point cloud map
      *globalMapCloud += *globalSurfCloud;

      int ret = pcl::io::savePCDFileBinary(saveMapDirectory + "/GlobalMap.pcd", *globalMapCloud);
      res.success = ret == 0;

      downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);

      cout << "****************************************************" << endl;
      cout << "Saving map to pcd files completed\n" << endl;

      return true;
    }

    void visualizeGlobalMapThread()
    {
        ros::Rate rate(0.2);
        while (ros::ok()){
            rate.sleep();
            publishGlobalMap();
        }

        if (savePCD == false)
            return;

        liorf_localization::save_mapRequest  req;
        liorf_localization::save_mapResponse res;

        // if(!saveMapService(req, res)){
        //     cout << "Fail to save map" << endl;
        // }
    }

    void publishGlobalMap()
    {
        if (pubLaserCloudSurround.getNumSubscribers() == 0)
            return;

        if (cloudKeyPoses3D->points.empty() == true)
            return;

        pcl::KdTreeFLANN<PointType>::Ptr kdtreeGlobalMap(new pcl::KdTreeFLANN<PointType>());;
        pcl::PointCloud<PointType>::Ptr globalMapKeyPoses(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyPosesDS(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyFrames(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyFramesDS(new pcl::PointCloud<PointType>());

        // kd-tree to find near key frames to visualize
        std::vector<int> pointSearchIndGlobalMap;
        std::vector<float> pointSearchSqDisGlobalMap;
        // search near key frames to visualize
        mtx.lock();
        kdtreeGlobalMap->setInputCloud(cloudKeyPoses3D);
        kdtreeGlobalMap->radiusSearch(cloudKeyPoses3D->back(), globalMapVisualizationSearchRadius, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap, 0);
        mtx.unlock();

        for (int i = 0; i < (int)pointSearchIndGlobalMap.size(); ++i)
            globalMapKeyPoses->push_back(cloudKeyPoses3D->points[pointSearchIndGlobalMap[i]]);
        // downsample near selected key frames
        pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyPoses; // for global map visualization
        downSizeFilterGlobalMapKeyPoses.setLeafSize(globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity); // for global map visualization
        downSizeFilterGlobalMapKeyPoses.setInputCloud(globalMapKeyPoses);
        downSizeFilterGlobalMapKeyPoses.filter(*globalMapKeyPosesDS);
        for(auto& pt : globalMapKeyPosesDS->points)
        {
            kdtreeGlobalMap->nearestKSearch(pt, 1, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap);
            pt.intensity = cloudKeyPoses3D->points[pointSearchIndGlobalMap[0]].intensity;
        }

        // extract visualized and downsampled key frames
        for (int i = 0; i < (int)globalMapKeyPosesDS->size(); ++i){
            if (common_lib_->pointDistance(globalMapKeyPosesDS->points[i], cloudKeyPoses3D->back()) > globalMapVisualizationSearchRadius)
                continue;
            int thisKeyInd = (int)globalMapKeyPosesDS->points[i].intensity;
            *globalMapKeyFrames += *transformPointCloud(surfCloudKeyFrames[thisKeyInd],    &cloudKeyPoses6D->points[thisKeyInd]);
        }
        // downsample visualized points
        pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyFrames; // for global map visualization
        downSizeFilterGlobalMapKeyFrames.setLeafSize(globalMapVisualizationLeafSize, globalMapVisualizationLeafSize, globalMapVisualizationLeafSize); // for global map visualization
        downSizeFilterGlobalMapKeyFrames.setInputCloud(globalMapKeyFrames);
        downSizeFilterGlobalMapKeyFrames.filter(*globalMapKeyFramesDS);
        publishCloud(pubLaserCloudSurround, globalMapKeyFramesDS, timeLaserInfoStamp, odometryFrame);
    }












    void loopClosureThread()
    {
        if (loopClosureEnableFlag == false)
            return;

        ros::Rate rate(loopClosureFrequency);
        while (ros::ok())
        {
            rate.sleep();
            performLoopClosure();
            visualizeLoopClosure();
        }
    }

    void loopInfoHandler(const std_msgs::Float64MultiArray::ConstPtr& loopMsg)
    {
        std::lock_guard<std::mutex> lock(mtxLoopInfo);
        if (loopMsg->data.size() != 2)
            return;

        loopInfoVec.push_back(*loopMsg);

        while (loopInfoVec.size() > 5)
            loopInfoVec.pop_front();
    }

    void performLoopClosure()
    {
        if (cloudKeyPoses3D->points.empty() == true)
            return;

        mtx.lock();
        *copy_cloudKeyPoses3D = *cloudKeyPoses3D;
        *copy_cloudKeyPoses6D = *cloudKeyPoses6D;
        mtx.unlock();

        // find keys
        int loopKeyCur;
        int loopKeyPre;
        if (detectLoopClosureExternal(&loopKeyCur, &loopKeyPre) == false)
            if (detectLoopClosureDistance(&loopKeyCur, &loopKeyPre) == false)
                return;

        // extract cloud
        pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr prevKeyframeCloud(new pcl::PointCloud<PointType>());
        {
            loopFindNearKeyframes(cureKeyframeCloud, loopKeyCur, 0);
            loopFindNearKeyframes(prevKeyframeCloud, loopKeyPre, historyKeyframeSearchNum);
            if (cureKeyframeCloud->size() < 300 || prevKeyframeCloud->size() < 1000)
                return;
            if (pubHistoryKeyFrames.getNumSubscribers() != 0)
                publishCloud(pubHistoryKeyFrames, prevKeyframeCloud, timeLaserInfoStamp, odometryFrame);
        }

        // ICP Settings
        static pcl::IterativeClosestPoint<PointType, PointType> icp;
        icp.setMaxCorrespondenceDistance(historyKeyframeSearchRadius*2);
        icp.setMaximumIterations(100);
        icp.setTransformationEpsilon(1e-6);
        icp.setEuclideanFitnessEpsilon(1e-6);
        icp.setRANSACIterations(0);

        // Align clouds
        icp.setInputSource(cureKeyframeCloud);
        icp.setInputTarget(prevKeyframeCloud);
        pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
        icp.align(*unused_result);

        if (icp.hasConverged() == false || icp.getFitnessScore() > historyKeyframeFitnessScore)
            return;

        // publish corrected cloud
        if (pubIcpKeyFrames.getNumSubscribers() != 0)
        {
            pcl::PointCloud<PointType>::Ptr closed_cloud(new pcl::PointCloud<PointType>());
            pcl::transformPointCloud(*cureKeyframeCloud, *closed_cloud, icp.getFinalTransformation());
            publishCloud(pubIcpKeyFrames, closed_cloud, timeLaserInfoStamp, odometryFrame);
        }

        // Get pose transformation
        float x, y, z, roll, pitch, yaw;
        Eigen::Affine3f correctionLidarFrame;
        correctionLidarFrame = icp.getFinalTransformation();
        // transform from world origin to wrong pose
        Eigen::Affine3f tWrong = pclPointToAffine3f(copy_cloudKeyPoses6D->points[loopKeyCur]);
        // transform from world origin to corrected pose
        Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong;// pre-multiplying -> successive rotation about a fixed frame
        pcl::getTranslationAndEulerAngles (tCorrect, x, y, z, roll, pitch, yaw);
        gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
        gtsam::Pose3 poseTo = pclPointTogtsamPose3(copy_cloudKeyPoses6D->points[loopKeyPre]);
        gtsam::Vector Vector6(6);
        float noiseScore = icp.getFitnessScore();
        Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
        noiseModel::Diagonal::shared_ptr constraintNoise = noiseModel::Diagonal::Variances(Vector6);

        // Add pose constraint
        mtx.lock();
        loopIndexQueue.push_back(make_pair(loopKeyCur, loopKeyPre));
        loopPoseQueue.push_back(poseFrom.between(poseTo));
        loopNoiseQueue.push_back(constraintNoise);
        mtx.unlock();

        // add loop constriant
        loopIndexContainer[loopKeyCur] = loopKeyPre;
    }

    bool detectLoopClosureDistance(int *latestID, int *closestID)
    {
        int loopKeyCur = copy_cloudKeyPoses3D->size() - 1;
        int loopKeyPre = -1;

        // check loop constraint added before
        auto it = loopIndexContainer.find(loopKeyCur);
        if (it != loopIndexContainer.end())
            return false;

        // find the closest history key frame
        std::vector<int> pointSearchIndLoop;
        std::vector<float> pointSearchSqDisLoop;
        kdtreeHistoryKeyPoses->setInputCloud(copy_cloudKeyPoses3D);
        kdtreeHistoryKeyPoses->radiusSearch(copy_cloudKeyPoses3D->back(), historyKeyframeSearchRadius, pointSearchIndLoop, pointSearchSqDisLoop, 0);
        
        for (int i = 0; i < (int)pointSearchIndLoop.size(); ++i)
        {
            int id = pointSearchIndLoop[i];
            if (abs(copy_cloudKeyPoses6D->points[id].time - timeLaserInfoCur) > historyKeyframeSearchTimeDiff)
            {
                loopKeyPre = id;
                break;
            }
        }

        if (loopKeyPre == -1 || loopKeyCur == loopKeyPre)
            return false;

        *latestID = loopKeyCur;
        *closestID = loopKeyPre;

        return true;
    }

    bool detectLoopClosureExternal(int *latestID, int *closestID)
    {
        // this function is not used yet, please ignore it
        int loopKeyCur = -1;
        int loopKeyPre = -1;

        std::lock_guard<std::mutex> lock(mtxLoopInfo);
        if (loopInfoVec.empty())
            return false;

        double loopTimeCur = loopInfoVec.front().data[0];
        double loopTimePre = loopInfoVec.front().data[1];
        loopInfoVec.pop_front();

        if (abs(loopTimeCur - loopTimePre) < historyKeyframeSearchTimeDiff)
            return false;

        int cloudSize = copy_cloudKeyPoses6D->size();
        if (cloudSize < 2)
            return false;

        // latest key
        loopKeyCur = cloudSize - 1;
        for (int i = cloudSize - 1; i >= 0; --i)
        {
            if (copy_cloudKeyPoses6D->points[i].time >= loopTimeCur)
                loopKeyCur = round(copy_cloudKeyPoses6D->points[i].intensity);
            else
                break;
        }

        // previous key
        loopKeyPre = 0;
        for (int i = 0; i < cloudSize; ++i)
        {
            if (copy_cloudKeyPoses6D->points[i].time <= loopTimePre)
                loopKeyPre = round(copy_cloudKeyPoses6D->points[i].intensity);
            else
                break;
        }

        if (loopKeyCur == loopKeyPre)
            return false;

        auto it = loopIndexContainer.find(loopKeyCur);
        if (it != loopIndexContainer.end())
            return false;

        *latestID = loopKeyCur;
        *closestID = loopKeyPre;

        return true;
    }

    void loopFindNearKeyframes(pcl::PointCloud<PointType>::Ptr& nearKeyframes, const int& key, const int& searchNum)
    {
        // extract near keyframes
        nearKeyframes->clear();
        int cloudSize = copy_cloudKeyPoses6D->size();
        for (int i = -searchNum; i <= searchNum; ++i)
        {
            int keyNear = key + i;
            if (keyNear < 0 || keyNear >= cloudSize )
                continue;
            *nearKeyframes += *transformPointCloud(surfCloudKeyFrames[keyNear],   &copy_cloudKeyPoses6D->points[keyNear]);
        }

        if (nearKeyframes->empty())
            return;

        // downsample near keyframes
        pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
        downSizeFilterICP.setInputCloud(nearKeyframes);
        downSizeFilterICP.filter(*cloud_temp);
        *nearKeyframes = *cloud_temp;
    }

    void visualizeLoopClosure()
    {
        if (loopIndexContainer.empty())
            return;
        
        visualization_msgs::MarkerArray markerArray;
        // loop nodes
        visualization_msgs::Marker markerNode;
        markerNode.header.frame_id = odometryFrame;
        markerNode.header.stamp = timeLaserInfoStamp;
        markerNode.action = visualization_msgs::Marker::ADD;
        markerNode.type = visualization_msgs::Marker::SPHERE_LIST;
        markerNode.ns = "loop_nodes";
        markerNode.id = 0;
        markerNode.pose.orientation.w = 1;
        markerNode.scale.x = 0.3; markerNode.scale.y = 0.3; markerNode.scale.z = 0.3; 
        markerNode.color.r = 0; markerNode.color.g = 0.8; markerNode.color.b = 1;
        markerNode.color.a = 1;
        // loop edges
        visualization_msgs::Marker markerEdge;
        markerEdge.header.frame_id = odometryFrame;
        markerEdge.header.stamp = timeLaserInfoStamp;
        markerEdge.action = visualization_msgs::Marker::ADD;
        markerEdge.type = visualization_msgs::Marker::LINE_LIST;
        markerEdge.ns = "loop_edges";
        markerEdge.id = 1;
        markerEdge.pose.orientation.w = 1;
        markerEdge.scale.x = 0.1;
        markerEdge.color.r = 0.9; markerEdge.color.g = 0.9; markerEdge.color.b = 0;
        markerEdge.color.a = 1;

        for (auto it = loopIndexContainer.begin(); it != loopIndexContainer.end(); ++it)
        {
            int key_cur = it->first;
            int key_pre = it->second;
            geometry_msgs::Point p;
            p.x = copy_cloudKeyPoses6D->points[key_cur].x;
            p.y = copy_cloudKeyPoses6D->points[key_cur].y;
            p.z = copy_cloudKeyPoses6D->points[key_cur].z;
            markerNode.points.push_back(p);
            markerEdge.points.push_back(p);
            p.x = copy_cloudKeyPoses6D->points[key_pre].x;
            p.y = copy_cloudKeyPoses6D->points[key_pre].y;
            p.z = copy_cloudKeyPoses6D->points[key_pre].z;
            markerNode.points.push_back(p);
            markerEdge.points.push_back(p);
        }

        markerArray.markers.push_back(markerNode);
        markerArray.markers.push_back(markerEdge);
        pubLoopConstraintEdge.publish(markerArray);
    }

    void updateInitialGuess()
    {
        // save current transformation before any processing
        incrementalOdometryAffineFront = trans2Affine3f(transformTobeMapped);

        static Eigen::Affine3f lastImuTransformation;
        // initialization
        if (cloudKeyPoses3D->points.empty())
        {
            // transformTobeMapped[0] = cloudInfo.imuRollInit;
            // transformTobeMapped[1] = cloudInfo.imuPitchInit;
            // transformTobeMapped[2] = cloudInfo.imuYawInit;

            // if (!useImuHeadingInitialization)
            //     transformTobeMapped[2] = 0;

            lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit); // save imu before return;
            return;
        }

        // use imu pre-integration estimation for pose guess
        bool is_valid_preintegration = true;
        static bool lastImuPreTransAvailable = false;
        static Eigen::Affine3f lastImuPreTransformation;
        if (cloudInfo.odomAvailable == true)
        {
            Eigen::Affine3f transBack = pcl::getTransformation(cloudInfo.initialGuessX,    cloudInfo.initialGuessY,     cloudInfo.initialGuessZ, 
                                                               cloudInfo.initialGuessRoll, cloudInfo.initialGuessPitch, cloudInfo.initialGuessYaw);
            if (lastImuPreTransAvailable == false)
            {
                lastImuPreTransformation = transBack;
                lastImuPreTransAvailable = true;
            } else {
                Eigen::Affine3f transIncre = lastImuPreTransformation.inverse() * transBack;
                Eigen::Affine3f transTobe = trans2Affine3f(transformTobeMapped);
                Eigen::Affine3f transFinal = transTobe * transIncre;

                if (safeMode) {
                    const double delta_time = timeLaserInfoCur - timeLastProcessing;
                    // float deltaT = sqrt(
                    //             pow((transformTobeMapped_prev[3] - transFinal(0,3)), 2) +
                    //             pow((transformTobeMapped_prev[4] - transFinal(1,3)), 2) +
                    //             pow((transformTobeMapped_prev[5] - transFinal(2,3)), 2));
                    float deltaT = sqrt(
                                pow(transIncre(0,3), 2) +
                                pow(transIncre(1,3), 2) +
                                pow(transIncre(2,3), 2));
                    double est_linear_velocity =  deltaT / delta_time;
                    if (est_linear_velocity > 2.5) {
                        std::cout << "[mapOptimization node] detected large velocity from IMU preintegraion info: " << est_linear_velocity << std::endl;
                        is_valid_preintegration = false;
                    }
                }

                if (is_valid_preintegration) {
                    pcl::getTranslationAndEulerAngles(transFinal, transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
                                                              transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);

                    lastImuPreTransformation = transBack;

                    lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit); // save imu before return;
                } else {
                    lastImuPreTransformation = transBack;
                    lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit); // save imu before return;
                }

                return;
            }
        }

        // use imu incremental estimation for pose guess (only rotation)
        if (cloudInfo.imuAvailable == true && imuType)
        {
            Eigen::Affine3f transBack = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit);
            Eigen::Affine3f transIncre = lastImuTransformation.inverse() * transBack;

            Eigen::Affine3f transTobe = trans2Affine3f(transformTobeMapped);
            Eigen::Affine3f transFinal = transTobe * transIncre;

            if (safeMode) {
                const double delta_time = timeLaserInfoCur - timeLastProcessing;
                // float deltaT = sqrt(
                //             pow((transformTobeMapped_prev[3] - transFinal(0,3)), 2) +
                //             pow((transformTobeMapped_prev[4] - transFinal(1,3)), 2) +
                //             pow((transformTobeMapped_prev[5] - transFinal(2,3)), 2));
                float deltaT = sqrt(
                                pow(transIncre(0,3), 2) +
                                pow(transIncre(1,3), 2) +
                                pow(transIncre(2,3), 2));
                double est_linear_velocity =  deltaT / delta_time;
                if (est_linear_velocity > 2.5) {
                    std::cout << "[mapOptimization node] detected large velocity from IMU preintegraion info: " << est_linear_velocity << std::endl;
                    is_valid_preintegration = false;
                }
            }

            if (is_valid_preintegration) {
                pcl::getTranslationAndEulerAngles(transFinal, transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
                                                        transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);

                lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit); // save imu before return;
            } else {
                lastImuPreTransformation = transBack;
                lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit); // save imu before return;
            }

            return;
        }
    }

    void extractForLoopClosure()
    {
        pcl::PointCloud<PointType>::Ptr cloudToExtract(new pcl::PointCloud<PointType>());
        int numPoses = cloudKeyPoses3D->size();
        for (int i = numPoses-1; i >= 0; --i)
        {
            if ((int)cloudToExtract->size() <= surroundingKeyframeSize)
                cloudToExtract->push_back(cloudKeyPoses3D->points[i]);
            else
                break;
        }

        extractCloud(cloudToExtract);
    }

    void extractNearby()
    {
        pcl::PointCloud<PointType>::Ptr surroundingKeyPoses(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surroundingKeyPosesDS(new pcl::PointCloud<PointType>());
        std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis;

        // extract all the nearby key poses and downsample them
        kdtreeSurroundingKeyPoses->setInputCloud(cloudKeyPoses3D); // create kd-tree
        kdtreeSurroundingKeyPoses->radiusSearch(cloudKeyPoses3D->back(), (double)surroundingKeyframeSearchRadius, pointSearchInd, pointSearchSqDis);
        for (int i = 0; i < (int)pointSearchInd.size(); ++i)
        {
            int id = pointSearchInd[i];
            surroundingKeyPoses->push_back(cloudKeyPoses3D->points[id]);
        }

        downSizeFilterSurroundingKeyPoses.setInputCloud(surroundingKeyPoses);
        downSizeFilterSurroundingKeyPoses.filter(*surroundingKeyPosesDS);
        for(auto& pt : surroundingKeyPosesDS->points)
        {
            kdtreeSurroundingKeyPoses->nearestKSearch(pt, 1, pointSearchInd, pointSearchSqDis);
            pt.intensity = cloudKeyPoses3D->points[pointSearchInd[0]].intensity;
        }

        // also extract some latest key frames in case the robot rotates in one position
        int numPoses = cloudKeyPoses3D->size();
        for (int i = numPoses-1; i >= 0; --i)
        {
            if (timeLaserInfoCur - cloudKeyPoses6D->points[i].time < 10.0)
                surroundingKeyPosesDS->push_back(cloudKeyPoses3D->points[i]);
            else
                break;
        }

        extractCloud(surroundingKeyPosesDS);
    }

    void extractCloud(pcl::PointCloud<PointType>::Ptr cloudToExtract)
    {
        // fuse the map
        laserCloudSurfFromMap->clear(); 
        for (int i = 0; i < (int)cloudToExtract->size(); ++i)
        {
            if (common_lib_->pointDistance(cloudToExtract->points[i], cloudKeyPoses3D->back()) > surroundingKeyframeSearchRadius)
                continue;

            int thisKeyInd = (int)cloudToExtract->points[i].intensity;
            if (laserCloudMapContainer.find(thisKeyInd) != laserCloudMapContainer.end()) 
            {
                // transformed cloud available
                *laserCloudSurfFromMap   += laserCloudMapContainer[thisKeyInd].second;
            } else {
                // transformed cloud not available
                pcl::PointCloud<PointType> laserCloudCornerTemp;
                pcl::PointCloud<PointType> laserCloudSurfTemp = *transformPointCloud(surfCloudKeyFrames[thisKeyInd],    &cloudKeyPoses6D->points[thisKeyInd]);
                *laserCloudSurfFromMap   += laserCloudSurfTemp;
                laserCloudMapContainer[thisKeyInd] = make_pair(laserCloudCornerTemp, laserCloudSurfTemp);
            }
            
        }

        // Downsample the surrounding surf key frames (or map)
        downSizeFilterLocalMapSurf.setInputCloud(laserCloudSurfFromMap);
        downSizeFilterLocalMapSurf.filter(*laserCloudSurfFromMapDS);
        laserCloudSurfFromMapDSNum = laserCloudSurfFromMapDS->size();

        // clear map cache if too large
        if (laserCloudMapContainer.size() > 1000)
            laserCloudMapContainer.clear();
    }

    void extractSurroundingKeyFrames()
    {
        if (cloudKeyPoses3D->points.empty() == true)
            return; 
        
        // if (loopClosureEnableFlag == true)
        // {
        //     extractForLoopClosure();    
        // } else {
        //     extractNearby();
        // }

        // extractNearby();
    }

    void downsampleCurrentScan()
    {
        laserCloudSurfLastDS->clear();
        if (enableDoensampling) {
            downSizeFilterSurf.setInputCloud(laserCloudSurfLast);
            downSizeFilterSurf.filter(*laserCloudSurfLastDS);
            laserCloudSurfLastDSNum = laserCloudSurfLastDS->size();
        } else {
            *laserCloudSurfLastDS = *laserCloudSurfLast;
            laserCloudSurfLastDSNum = laserCloudSurfLast->size();
        }
    }

    void updatePointAssociateToMap()
    {
        transPointAssociateToMap = trans2Affine3f(transformTobeMapped);
    }

    void surfOptimization()
    {
        updatePointAssociateToMap();

        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < laserCloudSurfLastDSNum; i++)
        {
            PointType pointOri, pointSel, coeff;
            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;

            pointOri = laserCloudSurfLastDS->points[i];
            pointAssociateToMap(&pointOri, &pointSel); 
            kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

            Eigen::Matrix<float, 5, 3> matA0;
            Eigen::Matrix<float, 5, 1> matB0;
            Eigen::Vector3f matX0;

            matA0.setZero();
            matB0.fill(-1);
            matX0.setZero();

            if (pointSearchSqDis[4] < 1.0) {
                for (int j = 0; j < 5; j++) {
                    matA0(j, 0) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].x;
                    matA0(j, 1) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].y;
                    matA0(j, 2) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].z;
                }

                matX0 = matA0.colPivHouseholderQr().solve(matB0);

                float pa = matX0(0, 0);
                float pb = matX0(1, 0);
                float pc = matX0(2, 0);
                float pd = 1;

                float ps = sqrt(pa * pa + pb * pb + pc * pc);
                pa /= ps; pb /= ps; pc /= ps; pd /= ps;

                bool planeValid = true;
                for (int j = 0; j < 5; j++) {
                    if (fabs(pa * laserCloudSurfFromMapDS->points[pointSearchInd[j]].x +
                             pb * laserCloudSurfFromMapDS->points[pointSearchInd[j]].y +
                             pc * laserCloudSurfFromMapDS->points[pointSearchInd[j]].z + pd) > 0.2) {
                        planeValid = false;
                        break;
                    }
                }

                if (planeValid) {
                    float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;

                    float s = 1 - 0.9 * fabs(pd2) / sqrt(sqrt(pointOri.x * pointOri.x
                            + pointOri.y * pointOri.y + pointOri.z * pointOri.z));

                    coeff.x = s * pa;
                    coeff.y = s * pb;
                    coeff.z = s * pc;
                    coeff.intensity = s * pd2;

                    if (s > 0.1) {
                        laserCloudOriSurfVec[i] = pointOri;
                        coeffSelSurfVec[i] = coeff;
                        laserCloudOriSurfFlag[i] = true;
                    }
                }
            }
        }
    }

    void surfOptimization_efficient()
    {
        updatePointAssociateToMap();

        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < laserCloudSurfLastDSNum; i++)
        {
            if (laserCloudSurfFromMapDSValid[i]) {
                PointType pointOri, pointSel, coeff;

                pointOri = laserCloudSurfLastDS->points[i];
                pointAssociateToMap(&pointOri, &pointSel);

                std::vector<int> k_indices;
                std::vector<float> k_sq_distances;
                kdtreeSurfFromMap->nearestKSearch(pointSel, 1, k_indices, k_sq_distances);

                if (k_sq_distances[0] < correspondenceDistance) {
                    int target_idx = k_indices[0];

                    float pa = laserCloudSurfFromMapDSPlaneCoefficients[target_idx][0];
                    float pb = laserCloudSurfFromMapDSPlaneCoefficients[target_idx][1];
                    float pc = laserCloudSurfFromMapDSPlaneCoefficients[target_idx][2];
                    float pd = laserCloudSurfFromMapDSPlaneCoefficients[target_idx][3];

                    // std::cout << pa << std::endl;

                    float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;

                    float s = 1 - 0.9 * fabs(pd2) / sqrt(sqrt(pointOri.x * pointOri.x
                            + pointOri.y * pointOri.y + pointOri.z * pointOri.z));

                    coeff.x = s * pa;
                    coeff.y = s * pb;
                    coeff.z = s * pc;
                    coeff.intensity = s * pd2;

                    if (s > 0.1) {
                        laserCloudOriSurfVec[i] = pointOri;
                        coeffSelSurfVec[i] = coeff;
                        laserCloudOriSurfFlag[i] = true;
                    }
                } 
            }
        }
    }

    void combineOptimizationCoeffs()
    {
        // combine surf coeffs
        for (int i = 0; i < laserCloudSurfLastDSNum; ++i){
            if (laserCloudOriSurfFlag[i] == true){
                laserCloudOri->push_back(laserCloudOriSurfVec[i]);
                coeffSel->push_back(coeffSelSurfVec[i]);
            }
        }
        // reset flag for next iteration
        std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);
    }

    bool LMOptimization(int iterCount)
    {
        // This optimization is from the original loam_velodyne by Ji Zhang, need to cope with coordinate transformation
        // lidar <- camera      ---     camera <- lidar
        // x = z                ---     x = y
        // y = x                ---     y = z
        // z = y                ---     z = x
        // roll = yaw           ---     roll = pitch
        // pitch = roll         ---     pitch = yaw
        // yaw = pitch          ---     yaw = roll

        // lidar -> camera
        float srx = sin(transformTobeMapped[2]);
        float crx = cos(transformTobeMapped[2]);
        float sry = sin(transformTobeMapped[1]);
        float cry = cos(transformTobeMapped[1]);
        float srz = sin(transformTobeMapped[0]);
        float crz = cos(transformTobeMapped[0]);

        int laserCloudSelNum = laserCloudOri->size();
        if (laserCloudSelNum < 50) {
            return false;
        }

        cv::Mat matA(laserCloudSelNum, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matAt(6, laserCloudSelNum, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtA(6, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matB(laserCloudSelNum, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtB(6, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matX(6, 1, CV_32F, cv::Scalar::all(0));

        PointType pointOri, coeff;

        // #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < laserCloudSelNum; i++) {
            // lidar -> camera
            pointOri.x = laserCloudOri->points[i].x;
            pointOri.y = laserCloudOri->points[i].y;
            pointOri.z = laserCloudOri->points[i].z;
            // lidar -> camera
            coeff.x = coeffSel->points[i].x;
            coeff.y = coeffSel->points[i].y;
            coeff.z = coeffSel->points[i].z;
            coeff.intensity = coeffSel->points[i].intensity;

            float arx = (-srx * cry * pointOri.x - (srx * sry * srz + crx * crz) * pointOri.y + (crx * srz - srx * sry * crz) * pointOri.z) * coeff.x
                      + (crx * cry * pointOri.x - (srx * crz - crx * sry * srz) * pointOri.y + (crx * sry * crz + srx * srz) * pointOri.z) * coeff.y;

            float ary = (-crx * sry * pointOri.x + crx * cry * srz * pointOri.y + crx * cry * crz * pointOri.z) * coeff.x
                      + (-srx * sry * pointOri.x + srx * sry * srz * pointOri.y + srx * cry * crz * pointOri.z) * coeff.y
                      + (-cry * pointOri.x - sry * srz * pointOri.y - sry * crz * pointOri.z) * coeff.z;

            float arz = ((crx * sry * crz + srx * srz) * pointOri.y + (srx * crz - crx * sry * srz) * pointOri.z) * coeff.x
                      + ((-crx * srz + srx * sry * crz) * pointOri.y + (-srx * sry * srz - crx * crz) * pointOri.z) * coeff.y
                      + (cry * crz * pointOri.y - cry * srz * pointOri.z) * coeff.z;

            // camera -> lidar
            matA.at<float>(i, 0) = arz;
            matA.at<float>(i, 1) = ary;
            matA.at<float>(i, 2) = arx;
            matA.at<float>(i, 3) = coeff.x;
            matA.at<float>(i, 4) = coeff.y;
            matA.at<float>(i, 5) = coeff.z;
            matB.at<float>(i, 0) = -coeff.intensity;
        }

        cv::transpose(matA, matAt);
        matAtA = matAt * matA;
        matAtB = matAt * matB;
        cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);

        if (iterCount == 0) {

            cv::Mat matE(1, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV(6, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV2(6, 6, CV_32F, cv::Scalar::all(0));

            cv::eigen(matAtA, matE, matV);
            matV.copyTo(matV2);

            isDegenerate = false;
            float eignThre[6] = {100, 100, 100, 100, 100, 100};
            for (int i = 5; i >= 0; i--) {
                if (matE.at<float>(0, i) < eignThre[i]) {
                    for (int j = 0; j < 6; j++) {
                        matV2.at<float>(i, j) = 0;
                    }
                    isDegenerate = true;
                } else {
                    break;
                }
            }
            matP = matV.inv() * matV2;
        }

        if (isDegenerate)
        {
            cv::Mat matX2(6, 1, CV_32F, cv::Scalar::all(0));
            matX.copyTo(matX2);
            matX = matP * matX2;
        }

        transformTobeMapped[0] += matX.at<float>(0, 0);
        transformTobeMapped[1] += matX.at<float>(1, 0);
        transformTobeMapped[2] += matX.at<float>(2, 0);
        transformTobeMapped[3] += matX.at<float>(3, 0);
        transformTobeMapped[4] += matX.at<float>(4, 0);
        transformTobeMapped[5] += matX.at<float>(5, 0);

        float deltaR = sqrt(
                            pow(pcl::rad2deg(matX.at<float>(0, 0)), 2) +
                            pow(pcl::rad2deg(matX.at<float>(1, 0)), 2) +
                            pow(pcl::rad2deg(matX.at<float>(2, 0)), 2));
        float deltaT = sqrt(
                            pow(matX.at<float>(3, 0) * 100, 2) +
                            pow(matX.at<float>(4, 0) * 100, 2) +
                            pow(matX.at<float>(5, 0) * 100, 2));

        if (deltaR < 0.05 && deltaT < 0.05) {
            return true; // converged
        }
        return false; // keep optimizing
    }
    // <!-- liorf_localization_yjz_lucky_boy -->
    void scan2MapOptimization()
    {
        if (cloudKeyPoses3D->points.empty())
            return;

        if (laserCloudSurfLastDSNum > 30)
        {
            // kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMapDS);

            for (int iterCount = 0; iterCount < 10; iterCount++)
            {
                laserCloudOri->clear();
                coeffSel->clear();

                // surfOptimization();
                surfOptimization_efficient();

                combineOptimizationCoeffs();

                if (LMOptimization(iterCount) == true) {
                    // std::cout << iterCount << std::endl;
                    break;
                }
                                  
            }
            
            transformUpdate();
        } else {
            ROS_WARN("Not enough features! Only %d planar features available.", laserCloudSurfLastDSNum);
        }
    }

    void transformUpdate()
    {
        if (cloudInfo.imuAvailable == true && imuType)
        {
            if (std::abs(cloudInfo.imuPitchInit) < 1.4)
            {
                double imuWeight = imuRPYWeight;
                tf::Quaternion imuQuaternion;
                tf::Quaternion transformQuaternion;
                double rollMid, pitchMid, yawMid;

                // slerp roll
                transformQuaternion.setRPY(transformTobeMapped[0], 0, 0);
                imuQuaternion.setRPY(cloudInfo.imuRollInit, 0, 0);
                tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                transformTobeMapped[0] = rollMid;

                // slerp pitch
                transformQuaternion.setRPY(0, transformTobeMapped[1], 0);
                imuQuaternion.setRPY(0, cloudInfo.imuPitchInit, 0);
                tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                transformTobeMapped[1] = pitchMid;
            }
        }

        transformTobeMapped[0] = constraintTransformation(transformTobeMapped[0], rotation_tollerance);
        transformTobeMapped[1] = constraintTransformation(transformTobeMapped[1], rotation_tollerance);
        transformTobeMapped[5] = constraintTransformation(transformTobeMapped[5], z_tollerance);

        if (safeMode) {
        // if (false) {
            // std::cout << locFailedIter << std::endl;
            // velocity constraint
            const double delta_time = timeLaserInfoCur - timeLastProcessing;
            float deltaT = sqrt(
                                pow((transformTobeMapped_prev[3] - transformTobeMapped[3]), 2) +
                                pow((transformTobeMapped_prev[4] - transformTobeMapped[4]), 2) +
                                pow((transformTobeMapped_prev[5] - transformTobeMapped[5]), 2));
            float deltaR = sqrt(
                                pow(pcl::rad2deg(transformTobeMapped_prev[0] - transformTobeMapped[0]), 2) +
                                pow(pcl::rad2deg(transformTobeMapped_prev[1] - transformTobeMapped[1]), 2) +
                                pow(pcl::rad2deg(transformTobeMapped_prev[2] - transformTobeMapped[2]), 2));

            double est_linear_velocity =  deltaT / delta_time;
            double relative_linear_velocity = fabs(est_linear_velocity - prev_scan_matching_vel);

            // std::cout << "est linear velocity: " << est_linear_velocity << std::endl;

            // if (est_linear_velocity > 3.0) {
            if (prev_scan_matching_vel > 0. && relative_linear_velocity > 3.0) {
                // locFailedIter++;
                std::cout << "scan matching failed! " << est_linear_velocity << std::endl;
                localization_successful = false;
                // for (int i = 0; i < 6; ++i){
                //     transformTobeMapped[i] = transformTobeMapped_prev[i];
                // }
            } else {
                locFailedIter = 0;
            }

            prev_scan_matching_vel = est_linear_velocity;

            // if (locFailedIter > 5) {
            //     needRelocalization = true;
            // }
        }

        incrementalOdometryAffineBack = trans2Affine3f(transformTobeMapped);
    }

    float constraintTransformation(float value, float limit)
    {
        if (value < -limit)
            value = -limit;
        if (value > limit)
            value = limit;

        return value;
    }

    bool saveFrame()
    {
        if (cloudKeyPoses3D->points.empty())
            return true;

        Eigen::Affine3f transStart = pclPointToAffine3f(cloudKeyPoses6D->back());
        Eigen::Affine3f transFinal = pcl::getTransformation(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
                                                            transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
        Eigen::Affine3f transBetween = transStart.inverse() * transFinal;
        float x, y, z, roll, pitch, yaw;
        pcl::getTranslationAndEulerAngles(transBetween, x, y, z, roll, pitch, yaw);

        if (abs(roll)  < surroundingkeyframeAddingAngleThreshold &&
            abs(pitch) < surroundingkeyframeAddingAngleThreshold && 
            abs(yaw)   < surroundingkeyframeAddingAngleThreshold &&
            sqrt(x*x + y*y + z*z) < surroundingkeyframeAddingDistThreshold)
            return false;

        return true;
    }

    void addOdomFactor()
    {
        if (cloudKeyPoses3D->points.empty())
        {
            noiseModel::Diagonal::shared_ptr priorNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-2, 1e-2, M_PI*M_PI, 1e8, 1e8, 1e8).finished()); // rad*rad, meter*meter
            gtSAMgraph.add(PriorFactor<Pose3>(0, trans2gtsamPose(transformTobeMapped), priorNoise));
            initialEstimate.insert(0, trans2gtsamPose(transformTobeMapped));
        }else{
            noiseModel::Diagonal::shared_ptr odometryNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
            gtsam::Pose3 poseFrom = pclPointTogtsamPose3(cloudKeyPoses6D->points.back());
            gtsam::Pose3 poseTo   = trans2gtsamPose(transformTobeMapped);
            gtSAMgraph.add(BetweenFactor<Pose3>(cloudKeyPoses3D->size()-1, cloudKeyPoses3D->size(), poseFrom.between(poseTo), odometryNoise));
            initialEstimate.insert(cloudKeyPoses3D->size(), poseTo);
        }
    }

    void addGPSFactor()
    {
        if (gpsQueue.empty())
            return;

        // wait for system initialized and settles down
        if (cloudKeyPoses3D->points.empty())
            return;
        else
        {
            if (common_lib_->pointDistance(cloudKeyPoses3D->front(), cloudKeyPoses3D->back()) < 5.0)
                return;
        }

        // pose covariance small, no need to correct
        if (poseCovariance(3,3) < poseCovThreshold && poseCovariance(4,4) < poseCovThreshold)
            return;

        // last gps position
        static PointType lastGPSPoint;

        while (!gpsQueue.empty())
        {
            if (gpsQueue.front().header.stamp.toSec() < timeLaserInfoCur - 0.2)
            {
                // message too old
                gpsQueue.pop_front();
            }
            else if (gpsQueue.front().header.stamp.toSec() > timeLaserInfoCur + 0.2)
            {
                // message too new
                break;
            }
            else
            {
                nav_msgs::Odometry thisGPS = gpsQueue.front();
                gpsQueue.pop_front();

                // GPS too noisy, skip
                float noise_x = thisGPS.pose.covariance[0];
                float noise_y = thisGPS.pose.covariance[7];
                float noise_z = thisGPS.pose.covariance[14];
                if (noise_x > gpsCovThreshold || noise_y > gpsCovThreshold)
                    continue;

                float gps_x = thisGPS.pose.pose.position.x;
                float gps_y = thisGPS.pose.pose.position.y;
                float gps_z = thisGPS.pose.pose.position.z;
                if (!useGpsElevation)
                {
                    gps_z = transformTobeMapped[5];
                    noise_z = 0.01;
                }

                // GPS not properly initialized (0,0,0)
                if (abs(gps_x) < 1e-6 && abs(gps_y) < 1e-6)
                    continue;

                // Add GPS every a few meters
                PointType curGPSPoint;
                curGPSPoint.x = gps_x;
                curGPSPoint.y = gps_y;
                curGPSPoint.z = gps_z;
                if (common_lib_->pointDistance(curGPSPoint, lastGPSPoint) < 5.0)
                    continue;
                else
                    lastGPSPoint = curGPSPoint;

                gtsam::Vector Vector3(3);
                Vector3 << max(noise_x, 1.0f), max(noise_y, 1.0f), max(noise_z, 1.0f);
                noiseModel::Diagonal::shared_ptr gps_noise = noiseModel::Diagonal::Variances(Vector3);
                gtsam::GPSFactor gps_factor(cloudKeyPoses3D->size(), gtsam::Point3(gps_x, gps_y, gps_z), gps_noise);
                gtSAMgraph.add(gps_factor);

                aLoopIsClosed = true;
                break;
            }
        }
    }

    void addLoopFactor()
    {
        if (loopIndexQueue.empty())
            return;

        for (int i = 0; i < (int)loopIndexQueue.size(); ++i)
        {
            int indexFrom = loopIndexQueue[i].first;
            int indexTo = loopIndexQueue[i].second;
            gtsam::Pose3 poseBetween = loopPoseQueue[i];
            gtsam::noiseModel::Diagonal::shared_ptr noiseBetween = loopNoiseQueue[i];
            gtSAMgraph.add(BetweenFactor<Pose3>(indexFrom, indexTo, poseBetween, noiseBetween));
        }

        loopIndexQueue.clear();
        loopPoseQueue.clear();
        loopNoiseQueue.clear();
        aLoopIsClosed = true;
    }

    void saveKeyFramesAndFactor()
    {
        if (saveFrame() == false)
            return;

        // odom factor
        addOdomFactor();

        // gps factor
        // addGPSFactor();

        // loop factor
        // addLoopFactor();

        // cout << "****************************************************" << endl;
        // gtSAMgraph.print("GTSAM Graph:\n");

        // update iSAM
        isam->update(gtSAMgraph, initialEstimate);
        isam->update();

        // if (aLoopIsClosed == true)
        // {
        //     isam->update();
        //     isam->update();
        //     isam->update();
        //     isam->update();
        //     isam->update();
        // }

        gtSAMgraph.resize(0);
        initialEstimate.clear();

        //save key poses
        PointType thisPose3D;
        PointTypePose thisPose6D;
        Pose3 latestEstimate;

        isamCurrentEstimate = isam->calculateEstimate();
        latestEstimate = isamCurrentEstimate.at<Pose3>(isamCurrentEstimate.size()-1);
        // cout << "****************************************************" << endl;
        // isamCurrentEstimate.print("Current estimate: ");

        thisPose3D.x = latestEstimate.translation().x();
        thisPose3D.y = latestEstimate.translation().y();
        thisPose3D.z = latestEstimate.translation().z();
        thisPose3D.intensity = cloudKeyPoses3D->size(); // this can be used as index
        cloudKeyPoses3D->push_back(thisPose3D);

        thisPose6D.x = thisPose3D.x;
        thisPose6D.y = thisPose3D.y;
        thisPose6D.z = thisPose3D.z;
        thisPose6D.intensity = thisPose3D.intensity ; // this can be used as index
        thisPose6D.roll  = latestEstimate.rotation().roll();
        thisPose6D.pitch = latestEstimate.rotation().pitch();
        thisPose6D.yaw   = latestEstimate.rotation().yaw();
        thisPose6D.time = timeLaserInfoCur;
        cloudKeyPoses6D->push_back(thisPose6D);

        // cout << "****************************************************" << endl;
        // cout << "Pose covariance:" << endl;
        // cout << isam->marginalCovariance(isamCurrentEstimate.size()-1) << endl << endl;
        poseCovariance = isam->marginalCovariance(isamCurrentEstimate.size()-1);

        // save updated transform
        transformTobeMapped[0] = latestEstimate.rotation().roll();
        transformTobeMapped[1] = latestEstimate.rotation().pitch();
        transformTobeMapped[2] = latestEstimate.rotation().yaw();
        transformTobeMapped[3] = latestEstimate.translation().x();
        transformTobeMapped[4] = latestEstimate.translation().y();
        transformTobeMapped[5] = latestEstimate.translation().z();

        // save all the received edge and surf points
        pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(new pcl::PointCloud<PointType>());
        pcl::copyPointCloud(*laserCloudSurfLastDS,    *thisSurfKeyFrame);

        // save key frame cloud
        surfCloudKeyFrames.push_back(thisSurfKeyFrame);

        // save path for visualization
        updatePath(thisPose6D);

        if (locFailedIter == 0) {
            Eigen::Affine3d relocInitialGuess_ = pcl::getTransformation(   transformTobeMapped[3],
                                                        transformTobeMapped[4],
                                                        transformTobeMapped[5],
                                                        transformTobeMapped[0],
                                                        transformTobeMapped[1],
                                                        transformTobeMapped[2]).cast<double>();

            relocInitialGuess.translation() = relocInitialGuess_.translation();
            relocInitialGuess.linear() = relocInitialGuess_.rotation();
        }
    }

    void correctPoses()
    {
        if (cloudKeyPoses3D->points.empty())
            return;

        if (aLoopIsClosed == true)
        {
            // clear map cache
            laserCloudMapContainer.clear();
            // clear path
            globalPath.poses.clear();
            // update key poses
            int numPoses = isamCurrentEstimate.size();
            for (int i = 0; i < numPoses; ++i)
            {
                cloudKeyPoses3D->points[i].x = isamCurrentEstimate.at<Pose3>(i).translation().x();
                cloudKeyPoses3D->points[i].y = isamCurrentEstimate.at<Pose3>(i).translation().y();
                cloudKeyPoses3D->points[i].z = isamCurrentEstimate.at<Pose3>(i).translation().z();

                cloudKeyPoses6D->points[i].x = cloudKeyPoses3D->points[i].x;
                cloudKeyPoses6D->points[i].y = cloudKeyPoses3D->points[i].y;
                cloudKeyPoses6D->points[i].z = cloudKeyPoses3D->points[i].z;
                cloudKeyPoses6D->points[i].roll  = isamCurrentEstimate.at<Pose3>(i).rotation().roll();
                cloudKeyPoses6D->points[i].pitch = isamCurrentEstimate.at<Pose3>(i).rotation().pitch();
                cloudKeyPoses6D->points[i].yaw   = isamCurrentEstimate.at<Pose3>(i).rotation().yaw();

                updatePath(cloudKeyPoses6D->points[i]);
            }

            aLoopIsClosed = false;
        }
    }

    void updatePath(const PointTypePose& pose_in)
    {
        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header.stamp = ros::Time().fromSec(pose_in.time);
        pose_stamped.header.frame_id = odometryFrame;
        pose_stamped.pose.position.x = pose_in.x;
        pose_stamped.pose.position.y = pose_in.y;
        pose_stamped.pose.position.z = pose_in.z;
        tf::Quaternion q = tf::createQuaternionFromRPY(pose_in.roll, pose_in.pitch, pose_in.yaw);
        pose_stamped.pose.orientation.x = q.x();
        pose_stamped.pose.orientation.y = q.y();
        pose_stamped.pose.orientation.z = q.z();
        pose_stamped.pose.orientation.w = q.w();

        globalPath.poses.push_back(pose_stamped);
    }

    void publishOdometry()
    {
        // Publish odometry for ROS (global)
        nav_msgs::Odometry laserOdometryROS;
        laserOdometryROS.header.stamp = timeLaserInfoStamp;
        laserOdometryROS.header.frame_id = odometryFrame;
        laserOdometryROS.child_frame_id = "odom_mapping";
        laserOdometryROS.pose.pose.position.x = transformTobeMapped[3];
        laserOdometryROS.pose.pose.position.y = transformTobeMapped[4];
        laserOdometryROS.pose.pose.position.z = transformTobeMapped[5];
        laserOdometryROS.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
        pubLaserOdometryGlobal.publish(laserOdometryROS);
        

        // // static tf
        // static tf::TransformBroadcaster tfMap2Odom;
        // static tf::Transform map_to_odom = tf::Transform(tf::createQuaternionFromRPY(0, 0, 0), tf::Vector3(0, 0, 0));
        // // tfMap2Odom.sendTransform(tf::StampedTransform(map_to_odom, timeLaserInfoStamp, "map", "odom"));
        // tfMap2Odom.sendTransform(tf::StampedTransform(map_to_odom, ros::Time::now(), "map", "odom"));

        // geometry_msgs::PoseWithCovarianceStamped pose_msg;
        // // pose_msg.header.stamp = timeLaserInfoStamp;
        // pose_msg.header.stamp = ros::Time::now();
        // pose_msg.pose = laserOdometryROS.pose;
        // pose_pub_.publish(pose_msg);


        // Publish TF
        static tf::TransformBroadcaster br;
        tf::Transform t_odom_to_lidar = tf::Transform(tf::createQuaternionFromRPY(transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]),
                                                      tf::Vector3(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5]));
        tf::StampedTransform trans_odom_to_lidar = tf::StampedTransform(t_odom_to_lidar, timeLaserInfoStamp, odometryFrame, "lidar_link");
        // tf::StampedTransform trans_odom_to_lidar = tf::StampedTransform(t_odom_to_lidar, ros::Time::now(), odometryFrame, "lidar_link");
        br.sendTransform(trans_odom_to_lidar);

        // publish base odometry for ROS (global)
        tf::Transform tCur(tf::Quaternion(laserOdometryROS.pose.pose.orientation.x, laserOdometryROS.pose.pose.orientation.y, laserOdometryROS.pose.pose.orientation.z, laserOdometryROS.pose.pose.orientation.w), 
                                tf::Vector3(laserOdometryROS.pose.pose.position.x, laserOdometryROS.pose.pose.position.y, laserOdometryROS.pose.pose.position.z));
        tCur *= lidar2Baselink;
        nav_msgs::Odometry baseOdometryROS;
        baseOdometryROS.header.stamp = timeLaserInfoStamp;
        baseOdometryROS.header.frame_id = odometryFrame;      // global fixed frame (e.g., "map" or "odom")
        baseOdometryROS.child_frame_id = "odom_mapping";     // moving frame

        // Fill pose
        baseOdometryROS.pose.pose.position.x = tCur.getOrigin().x();
        baseOdometryROS.pose.pose.position.y = tCur.getOrigin().y();
        baseOdometryROS.pose.pose.position.z = tCur.getOrigin().z();

        tf::Quaternion q = tCur.getRotation();
        baseOdometryROS.pose.pose.orientation.x = q.x();
        baseOdometryROS.pose.pose.orientation.y = q.y();
        baseOdometryROS.pose.pose.orientation.z = q.z();
        baseOdometryROS.pose.pose.orientation.w = q.w();

        // Optionally zero the twist if not used
        baseOdometryROS.twist.twist.linear.x = 0.0;
        baseOdometryROS.twist.twist.linear.y = 0.0;
        baseOdometryROS.twist.twist.linear.z = 0.0;
        baseOdometryROS.twist.twist.angular.x = 0.0;
        baseOdometryROS.twist.twist.angular.y = 0.0;
        baseOdometryROS.twist.twist.angular.z = 0.0;

        if (localization_successful) {
            baseOdometryROS.pose.covariance[0] = 1.0;
        } else {
            baseOdometryROS.pose.covariance[0] = -1.0;
        }

        // Publish
        pubBaseOdometryGlobal.publish(baseOdometryROS);


        // Publish odometry for ROS (incremental)
        static bool lastIncreOdomPubFlag = false;
        static nav_msgs::Odometry laserOdomIncremental; // incremental odometry msg
        static Eigen::Affine3f increOdomAffine; // incremental odometry in affine
        if (lastIncreOdomPubFlag == false)
        {
            lastIncreOdomPubFlag = true;
            laserOdomIncremental = laserOdometryROS;
            increOdomAffine = trans2Affine3f(transformTobeMapped);
        } else {
            Eigen::Affine3f affineIncre = incrementalOdometryAffineFront.inverse() * incrementalOdometryAffineBack;
            increOdomAffine = increOdomAffine * affineIncre;
            float x, y, z, roll, pitch, yaw;
            pcl::getTranslationAndEulerAngles (increOdomAffine, x, y, z, roll, pitch, yaw);
            if (cloudInfo.imuAvailable == true && imuType)
            {
                if (std::abs(cloudInfo.imuPitchInit) < 1.4)
                {
                    double imuWeight = 0.1;
                    tf::Quaternion imuQuaternion;
                    tf::Quaternion transformQuaternion;
                    double rollMid, pitchMid, yawMid;

                    // slerp roll
                    transformQuaternion.setRPY(roll, 0, 0);
                    imuQuaternion.setRPY(cloudInfo.imuRollInit, 0, 0);
                    tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                    roll = rollMid;

                    // slerp pitch
                    transformQuaternion.setRPY(0, pitch, 0);
                    imuQuaternion.setRPY(0, cloudInfo.imuPitchInit, 0);
                    tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                    pitch = pitchMid;
                }
            }
            laserOdomIncremental.header.stamp = timeLaserInfoStamp;
            laserOdomIncremental.header.frame_id = odometryFrame;
            laserOdomIncremental.child_frame_id = "odom_mapping";
            laserOdomIncremental.pose.pose.position.x = x;
            laserOdomIncremental.pose.pose.position.y = y;
            laserOdomIncremental.pose.pose.position.z = z;
            laserOdomIncremental.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(roll, pitch, yaw);
            if (isDegenerate)
                laserOdomIncremental.pose.covariance[0] = 1;
            else
                laserOdomIncremental.pose.covariance[0] = 0;
        }
        pubLaserOdometryIncremental.publish(laserOdomIncremental);
    }

    void publishFrames()
    {
        if (cloudKeyPoses3D->points.empty())
            return;
        // publish key poses
        publishCloud(pubKeyPoses, cloudKeyPoses3D, timeLaserInfoStamp, odometryFrame);
        // Publish surrounding key frames
        publishCloud(pubRecentKeyFrames, laserCloudSurfFromMapDS, timeLaserInfoStamp, odometryFrame);
        // publish registered key frame
        if (pubRecentKeyFrame.getNumSubscribers() != 0)
        {
            pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
            PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
            *cloudOut += *transformPointCloud(laserCloudSurfLastDS,    &thisPose6D);
            publishCloud(pubRecentKeyFrame, cloudOut, timeLaserInfoStamp, odometryFrame);
        }
        // publish registered high-res raw cloud
        if (pubCloudRegisteredRaw.getNumSubscribers() != 0)
        {
            pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
            pcl::fromROSMsg(cloudInfo.cloud_deskewed, *cloudOut);
            PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
            *cloudOut = *transformPointCloud(cloudOut,  &thisPose6D);
            publishCloud(pubCloudRegisteredRaw, cloudOut, timeLaserInfoStamp, odometryFrame);
        }
        // publish path
        if (pubPath.getNumSubscribers() != 0)
        {
            globalPath.header.stamp = timeLaserInfoStamp;
            globalPath.header.frame_id = odometryFrame;
            pubPath.publish(globalPath);
        }
        // publish SLAM infomation for 3rd-party usage
        static int lastSLAMInfoPubSize = -1;
        if (pubSLAMInfo.getNumSubscribers() != 0)
        {
            if (lastSLAMInfoPubSize != cloudKeyPoses6D->size())
            {
                liorf_localization::cloud_info slamInfo;
                slamInfo.header.stamp = timeLaserInfoStamp;
                pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
                *cloudOut += *laserCloudSurfLastDS;
                slamInfo.key_frame_cloud = publishCloud(ros::Publisher(), cloudOut, timeLaserInfoStamp, lidarFrame);
                slamInfo.key_frame_poses = publishCloud(ros::Publisher(), cloudKeyPoses6D, timeLaserInfoStamp, odometryFrame);
                pcl::PointCloud<PointType>::Ptr localMapOut(new pcl::PointCloud<PointType>());
                *localMapOut += *laserCloudSurfFromMapDS;
                slamInfo.key_frame_map = publishCloud(ros::Publisher(), localMapOut, timeLaserInfoStamp, odometryFrame);
                pubSLAMInfo.publish(slamInfo);
                lastSLAMInfoPubSize = cloudKeyPoses6D->size();
            }
        }
    }
};


int main(int argc, char** argv)
{
    ros::init(argc, argv, "liorf_localization");

    mapOptimization MO;

    ROS_INFO("\033[1;32m----> Map Optimization Started.\033[0m");
    
    // std::thread loopthread(&mapOptimization::loopClosureThread, &MO);
    // std::thread visualizeMapThread(&mapOptimization::visualizeGlobalMapThread, &MO);

    ros::spin();

    // loopthread.join();
    // visualizeMapThread.join();

    return 0;
}
