#include "utility.h"

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
#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>

#include <std_msgs/Float64MultiArray.h>
#include <geometry_msgs/PointStamped.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#define USE_LEG 1

using gtsam::symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
using gtsam::symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using gtsam::symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)

#if USE_LEG
using gtsam::symbol_shorthand::C; // Leg Odom Bias (bv)
#endif

using Transform = Eigen::Isometry3d;

#if USE_LEG
class Go2Kinematics {
public:
    Go2Kinematics() {
        // Front-Left (FL) Leg
        offsets_["FL_hip"]   = Eigen::Translation3d(0.1934, 0.0465, 0.0);
        offsets_["FL_thigh"] = Eigen::Translation3d(0.0, 0.0955, 0.0);
        offsets_["FL_calf"]  = Eigen::Translation3d(0.0, 0.0, -0.213);
        offsets_["FL_foot"]  = Eigen::Translation3d(0.0, 0.0, -0.213);

        // Front-Right (FR) Leg
        offsets_["FR_hip"]   = Eigen::Translation3d(0.1934, -0.0465, 0.0);
        offsets_["FR_thigh"] = Eigen::Translation3d(0.0, -0.0955, 0.0);
        offsets_["FR_calf"]  = Eigen::Translation3d(0.0, 0.0, -0.213);
        offsets_["FR_foot"]  = Eigen::Translation3d(0.0, 0.0, -0.213);

        // Rear-Left (RL) Leg
        offsets_["RL_hip"]   = Eigen::Translation3d(-0.1934, 0.0465, 0.0);
        offsets_["RL_thigh"] = Eigen::Translation3d(0.0, 0.0955, 0.0);
        offsets_["RL_calf"]  = Eigen::Translation3d(0.0, 0.0, -0.213);
        offsets_["RL_foot"]  = Eigen::Translation3d(0.0, 0.0, -0.213);

        // Rear-Right (RR) Leg
        offsets_["RR_hip"]   = Eigen::Translation3d(-0.1934, -0.0465, 0.0);
        offsets_["RR_thigh"] = Eigen::Translation3d(0.0, -0.0955, 0.0);
        offsets_["RR_calf"]  = Eigen::Translation3d(0.0, 0.0, -0.213);
        offsets_["RR_foot"]  = Eigen::Translation3d(0.0, 0.0, -0.213);
    }

    // foot pose
    Transform getFootPose(const std::string& leg_prefix, const Eigen::Vector3d& joint_angles) {
        Transform hip_rotation(Eigen::AngleAxisd(joint_angles[0], Eigen::Vector3d::UnitX()));
        Transform thigh_rotation(Eigen::AngleAxisd(joint_angles[1], Eigen::Vector3d::UnitY()));
        Transform calf_rotation(Eigen::AngleAxisd(joint_angles[2], Eigen::Vector3d::UnitY()));

        Transform T_base_foot = 
            offsets_.at(leg_prefix + "_hip")   * hip_rotation *
            offsets_.at(leg_prefix + "_thigh") * thigh_rotation *
            offsets_.at(leg_prefix + "_calf")  * calf_rotation *
            offsets_.at(leg_prefix + "_foot");

        return T_base_foot;
    }

    /**
     * @brief Jacobian matrix for each legs
     * @param leg_prefix Leg identification prefixes ("FL", "FR", "RL", "RR").
     * @param joint_angles Joint angle (hip, thigh, calf) vector (unit: rad).
     * @return 3x3 jacobian matrix.
     */
    Eigen::Matrix3d getFootJacobian(const std::string& leg_prefix, const Eigen::Vector3d& joint_angles) {
        Eigen::Matrix3d jacobian;

        Eigen::Transform<double, 3, 1> T_base_hip = offsets_.at(leg_prefix + "_hip") * Eigen::AngleAxisd(joint_angles[0], Eigen::Vector3d::UnitX());
        Eigen::Transform<double, 3, 1> T_hip_thigh = offsets_.at(leg_prefix + "_thigh") * Eigen::AngleAxisd(joint_angles[1], Eigen::Vector3d::UnitY());
        Eigen::Transform<double, 3, 1> T_thigh_calf = offsets_.at(leg_prefix + "_calf") * Eigen::AngleAxisd(joint_angles[2], Eigen::Vector3d::UnitY());
        Eigen::Translation3d calf_foot_translation = offsets_.at(leg_prefix + "_foot");
        Eigen::Transform<double, 3, 1> T_calf_foot(calf_foot_translation);

        Eigen::Transform<double, 3, 1> T_base_thigh = T_base_hip * T_hip_thigh;
        Eigen::Transform<double, 3, 1> T_base_calf = T_base_thigh * T_thigh_calf;
        Eigen::Transform<double, 3, 1> T_base_foot = T_base_calf * T_calf_foot;
        
        // Calculation of joint position and rotation axis relative to the base link
        Eigen::Vector3d p_hip = T_base_hip.translation();
        Eigen::Vector3d z_hip = Eigen::Vector3d::UnitX(); // Hip joint axis in base frame

        Eigen::Vector3d p_thigh = T_base_thigh.translation();
        Eigen::Vector3d z_thigh = T_base_hip.rotation() * Eigen::Vector3d::UnitY(); // Thigh joint axis in base frame

        Eigen::Vector3d p_calf = T_base_calf.translation();
        Eigen::Vector3d z_calf = T_base_thigh.rotation() * Eigen::Vector3d::UnitY(); // Calf joint axis in base frame

        Eigen::Vector3d p_foot = T_base_foot.translation();

        // J_i = z_i x (p_foot - p_i)
        jacobian.col(0) = z_hip.cross(p_foot - p_hip);
        jacobian.col(1) = z_thigh.cross(p_foot - p_thigh);
        jacobian.col(2) = z_calf.cross(p_foot - p_calf);

        return jacobian;
    }

private:
    std::map<std::string, Eigen::Translation3d> offsets_;
};

/**
 * @brief PreintegratedLegOdom: Leg Odometry preintegration class
 * Integrate Leg Odometry velocity between two key frames to compute relative pose change
 * Jacobian with respect to the effect of the Body Velocity Bias
 */
class PreintegratedLegOdom 
{
public:
    // noise covariance (w, v)
    gtsam::Matrix66 Q_covariance_;

    // relative pose (ΔR, Δp)
    gtsam::Rot3 delta_R_;
    gtsam::Vector3 delta_p_;
    double delta_t_;

    // propagate cov.
    gtsam::Matrix66 P_;

    // jacobian about body velocity bias
    gtsam::Matrix33 J_p_bv_;
    
    PreintegratedLegOdom(const gtsam::Matrix66& velocity_covariance) : Q_covariance_(velocity_covariance) 
    {
        resetIntegration();
    }

    PreintegratedLegOdom() 
    {
        resetIntegration();
    }
    
    void integrateMeasurement(const gtsam::Vector3& measured_v, const Eigen::Vector3d& measured_w, const gtsam::PreintegrationBase::Bias prev_bias, double dt)
    {   
        if (dt <= 0 ) return;

        gtsam::Vector3 w_dt = (measured_w - prev_bias.gyroscope())* dt;
        gtsam::Matrix33 Jr;
        gtsam::Rot3 dR = gtsam::Rot3::Expmap(w_dt, Jr);

        // relative pose
        delta_R_ = delta_R_ * dR;
        delta_p_ += delta_R_.rotate(measured_v) * dt;
        delta_t_ += dt;

        // covariance propagation
        gtsam::Matrix66 F = gtsam::Matrix66::Identity();
        F.block<3, 3>(0, 0) = dR.transpose();
        F.block<3, 3>(3, 0) = -delta_R_.matrix() * gtsam::skewSymmetric(measured_v) * dt;

        gtsam::Matrix66 G = gtsam::Matrix66::Zero();
        G.block<3, 3>(0, 0) = Jr * dt;                
        G.block<3, 3>(3, 3) = delta_R_.matrix() * dt; 

        P_ = F * P_ * F.transpose() + G * Q_covariance_ * G.transpose();

        // jacobian about body velocity bias
        J_p_bv_ -= delta_R_.matrix() * dt;
    }

    void resetIntegration()
    {
        delta_R_ = gtsam::Rot3();
        delta_p_.setZero();
        delta_t_ = 0.0;
        P_ = gtsam::Matrix66::Identity() * 0.5;
        J_p_bv_.setZero();
    }

    gtsam::Pose3 deltaPose() const { return gtsam::Pose3(delta_R_, delta_p_); }
    gtsam::Vector3 deltaP() const { return delta_p_; }
    double deltaT() const { return delta_t_; }
    const gtsam::Matrix66& preintCovariance() const { return P_; }
    const gtsam::Matrix33& jacobian_p_bv() const { return J_p_bv_; }
};

// Leg Odometry Factor
class LegOdometryFactor : public gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3, gtsam::Vector3> 
{
private:
    gtsam::Vector3 measured_delta_p_;
    gtsam::Matrix33 J_p_bv_;

public:
    LegOdometryFactor(gtsam::Key pose_i, gtsam::Key pose_j, gtsam::Key bv_i,
                      const PreintegratedLegOdom& measurement, const gtsam::SharedNoiseModel& model)
        : NoiseModelFactor3(model, pose_i, pose_j, bv_i),
          measured_delta_p_(measurement.deltaP()), J_p_bv_(measurement.jacobian_p_bv()) {}

    gtsam::Vector evaluateError(const gtsam::Pose3& pose_i, const gtsam::Pose3& pose_j, const gtsam::Vector3& body_vel_bias,
                                boost::optional<gtsam::Matrix&> H1 = boost::none,
                                boost::optional<gtsam::Matrix&> H2 = boost::none,
                                boost::optional<gtsam::Matrix&> H3 = boost::none) const override
    {
        gtsam::Point3 p_i = pose_i.translation();
        gtsam::Point3 p_j = pose_j.translation();
        gtsam::Rot3 R_i = pose_i.rotation();
        gtsam::Rot3 R_j = pose_j.rotation();
        
        gtsam::Vector3 predicted_delta_p = R_i.unrotate(p_j - p_i);
        gtsam::Vector3 corrected_measured_delta_p = measured_delta_p_ + J_p_bv_ * body_vel_bias;

        // error calculation
        gtsam::Vector6 error = gtsam::Vector6::Zero(); // set zero (same as the imu factor)
        error.tail(3) = predicted_delta_p - corrected_measured_delta_p;

        // jacboain
        if (H1) 
        {
            gtsam::Matrix66 J_H1 = gtsam::Matrix66::Zero();
            
            gtsam::Point3 pj_in_i = R_i.unrotate(p_j - p_i);
            gtsam::Matrix36 J_pos;
            J_pos.leftCols<3>() = gtsam::skewSymmetric(pj_in_i); // d(pos_error)/d(rot_i)
            J_pos.rightCols<3>() = -R_i.matrix().transpose();    // d(pos_error)/d(pos_i)

            J_H1.block<3,6>(3,0) = J_pos;            
            *H1 = J_H1;
        }
        if (H2) 
        {
            gtsam::Matrix66 J_H2 = gtsam::Matrix66::Zero();
            
            gtsam::Matrix36 J_pos;
            J_pos.leftCols<3>() = gtsam::Matrix33::Zero();      // d(pos_error)/d(rot_j)
            J_pos.rightCols<3>() = R_i.matrix().transpose();    // d(pos_error)/d(pos_j)

            J_H2.block<3,6>(3,0) = J_pos;            
            *H2 = J_H2;
        }
        if (H3) // d(error)/d(body_vel_bias)
        {
            gtsam::Matrix63 J_H3 = gtsam::Matrix63::Zero();
            J_H3.bottomRows<3>() = -J_p_bv_;
            *H3 = J_H3;
        }

        return error;
    }
};
#endif

class TransformFusion : public ParamServer
{
public:
    std::mutex mtx;

    ros::Subscriber subImuOdometry;
    ros::Subscriber subLaserOdometry;

    ros::Publisher pubImuOdometry;
    ros::Publisher pubImuPath;
    ros::Publisher pose_pub_;

    Eigen::Affine3f lidarOdomAffine;
    Eigen::Affine3f imuOdomAffineFront;
    Eigen::Affine3f imuOdomAffineBack;

    tf::TransformListener tfListener;
    tf::StampedTransform lidar2Baselink;

    double lidarOdomTime = -1;
    deque<nav_msgs::Odometry> imuOdomQueue;

    TransformFusion()
    {
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

        subLaserOdometry = nh.subscribe<nav_msgs::Odometry>("liorf_localization/mapping/odometry", 5, &TransformFusion::lidarOdometryHandler, this, ros::TransportHints().tcpNoDelay());
        subImuOdometry   = nh.subscribe<nav_msgs::Odometry>(odomTopic+"_incremental",   2000, &TransformFusion::imuOdometryHandler,   this, ros::TransportHints().tcpNoDelay());

        pubImuOdometry   = nh.advertise<nav_msgs::Odometry>(odomTopic, 2000);
        pubImuPath       = nh.advertise<nav_msgs::Path>    ("liorf_localization/imu/path", 1);
        pose_pub_ = nh.advertise<geometry_msgs::PoseWithCovarianceStamped>("/unitree/pose_with_covariance", 10);

    }

    Eigen::Affine3f odom2affine(nav_msgs::Odometry odom)
    {
        double x, y, z, roll, pitch, yaw;
        x = odom.pose.pose.position.x;
        y = odom.pose.pose.position.y;
        z = odom.pose.pose.position.z;
        tf::Quaternion orientation;
        tf::quaternionMsgToTF(odom.pose.pose.orientation, orientation);
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
        return pcl::getTransformation(x, y, z, roll, pitch, yaw);
    }

    void lidarOdometryHandler(const nav_msgs::Odometry::ConstPtr& odomMsg)
    {
        std::lock_guard<std::mutex> lock(mtx);

        lidarOdomAffine = odom2affine(*odomMsg);

        lidarOdomTime = odomMsg->header.stamp.toSec();
    }

    void imuOdometryHandler(const nav_msgs::Odometry::ConstPtr& odomMsg)
    {
        // static tf
        static tf::TransformBroadcaster tfMap2Odom;
        static tf::Transform map_to_odom = tf::Transform(tf::createQuaternionFromRPY(0, 0, 0), tf::Vector3(0, 0, 0));
        // tfMap2Odom.sendTransform(tf::StampedTransform(map_to_odom, ros::Time::now(), mapFrame, odometryFrame));
        tfMap2Odom.sendTransform(tf::StampedTransform(map_to_odom, odomMsg->header.stamp, mapFrame, odometryFrame));

        std::lock_guard<std::mutex> lock(mtx);

        imuOdomQueue.push_back(*odomMsg);

        // get latest odometry (at current IMU stamp)
        if (lidarOdomTime == -1)
            return;
        while (!imuOdomQueue.empty())
        {
            if (imuOdomQueue.front().header.stamp.toSec() <= lidarOdomTime)
                imuOdomQueue.pop_front();
            else
                break;
        }
        Eigen::Affine3f imuOdomAffineFront = odom2affine(imuOdomQueue.front());
        Eigen::Affine3f imuOdomAffineBack = odom2affine(imuOdomQueue.back());
        Eigen::Affine3f imuOdomAffineIncre = imuOdomAffineFront.inverse() * imuOdomAffineBack;
        Eigen::Affine3f imuOdomAffineLast = lidarOdomAffine * imuOdomAffineIncre;
        float x, y, z, roll, pitch, yaw;
        pcl::getTranslationAndEulerAngles(imuOdomAffineLast, x, y, z, roll, pitch, yaw);
        
        // publish latest odometry
        nav_msgs::Odometry laserOdometry = imuOdomQueue.back();
        laserOdometry.pose.pose.position.x = x;
        laserOdometry.pose.pose.position.y = y;
        laserOdometry.pose.pose.position.z = z;
        laserOdometry.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(roll, pitch, yaw);
        pubImuOdometry.publish(laserOdometry);

        // publish tf
        static tf::TransformBroadcaster tfOdom2BaseLink;
        tf::Transform tCur;
        tf::poseMsgToTF(laserOdometry.pose.pose, tCur);
        if(lidarFrame != baselinkFrame)
            tCur = tCur * lidar2Baselink;
        tf::StampedTransform odom_2_baselink = tf::StampedTransform(tCur, ros::Time::now(), odometryFrame, baselinkFrame);
        // tf::StampedTransform odom_2_baselink = tf::StampedTransform(tCur, odomMsg->header.stamp, odometryFrame, baselinkFrame);
        tfOdom2BaseLink.sendTransform(odom_2_baselink);

        // publish IMU path
        static nav_msgs::Path imuPath;
        static double last_path_time = -1;
        double imuTime = imuOdomQueue.back().header.stamp.toSec();
        if (imuTime - last_path_time > 0.1)
        {
            last_path_time = imuTime;
            geometry_msgs::PoseStamped pose_stamped;
            pose_stamped.header.stamp = imuOdomQueue.back().header.stamp;
            pose_stamped.header.frame_id = odometryFrame;
            pose_stamped.pose = laserOdometry.pose.pose;
            imuPath.poses.push_back(pose_stamped);
            while(!imuPath.poses.empty() && imuPath.poses.front().header.stamp.toSec() < lidarOdomTime - 1.0)
                imuPath.poses.erase(imuPath.poses.begin());
            if (pubImuPath.getNumSubscribers() != 0)
            {
                imuPath.header.stamp = imuOdomQueue.back().header.stamp;
                imuPath.header.frame_id = odometryFrame;
                pubImuPath.publish(imuPath);
            }
        }
    }
};

class IMUPreintegration : public ParamServer
{
public:

    std::mutex mtx;
    ros::Subscriber subImu;
    ros::Subscriber subOdometry;
    ros::Publisher pubImuOdometry;
    ros::Publisher pubBodyVel;

#if USE_LEG
    std::mutex legMtx;
    std::mutex leg_odom_mutex_;

    ros::Subscriber subLeg; 
    ros::Publisher pubCalculatedLegOdometry; 
    ros::Publisher pubLegOdomBias; 
    std::map<std::string, ros::Publisher> pubFootPositions; 

    gtsam::noiseModel::Diagonal::shared_ptr priorBodyVelBiasNoise; 
    gtsam::Vector noiseModelBetweenBias2; 
    gtsam::noiseModel::Diagonal::shared_ptr bodyVelBiasNoise; 

    PreintegratedLegOdom *legIntegratorOpt_;
    Go2Kinematics kinematics_calculator_;   

    gtsam::Vector3 prevBodyVelBias_; 

    gtsam::Vector3 latest_leg_odom_velocity_;
    bool new_leg_odom_available_ = false;
#endif

    bool systemInitialized = false;

    gtsam::noiseModel::Diagonal::shared_ptr priorPoseNoise;
    gtsam::noiseModel::Diagonal::shared_ptr priorVelNoise;
    gtsam::noiseModel::Diagonal::shared_ptr priorBiasNoise;
    gtsam::noiseModel::Diagonal::shared_ptr correctionNoise;
    gtsam::noiseModel::Diagonal::shared_ptr correctionNoise2;
    gtsam::Vector noiseModelBetweenBias;


    gtsam::PreintegratedImuMeasurements *imuIntegratorOpt_;
    gtsam::PreintegratedImuMeasurements *imuIntegratorImu_;

    std::deque<sensor_msgs::Imu> imuQueOpt;
    std::deque<sensor_msgs::Imu> imuQueImu;

    gtsam::Pose3 prevPose_;
    gtsam::Vector3 prevVel_;
    gtsam::NavState prevState_;
    gtsam::imuBias::ConstantBias prevBias_;

    gtsam::NavState prevStateOdom;
    gtsam::imuBias::ConstantBias prevBiasOdom;

    bool doneFirstOpt = false;
    double lastImuT_imu = -1;
    double lastImuT_opt = -1;
    double lastLegOdomTime = -1;

    gtsam::ISAM2 optimizer;
    gtsam::NonlinearFactorGraph graphFactors;
    gtsam::Values graphValues;

    const double delta_t = 0;

    int key = 1;
    
    // T_bl: tramsform points from lidar frame to imu frame 
    gtsam::Pose3 imu2Lidar = gtsam::Pose3(gtsam::Rot3(1, 0, 0, 0), gtsam::Point3(-extTrans.x(), -extTrans.y(), -extTrans.z()));
    // T_lb: tramsform points from imu frame to lidar frame
    gtsam::Pose3 lidar2Imu = gtsam::Pose3(gtsam::Rot3(1, 0, 0, 0), gtsam::Point3(extTrans.x(), extTrans.y(), extTrans.z()));

    IMUPreintegration()
    {
        subImu      = nh.subscribe<sensor_msgs::Imu>  (imuTopic,                   2000, &IMUPreintegration::imuHandler,      this, ros::TransportHints().tcpNoDelay());
        subOdometry = nh.subscribe<nav_msgs::Odometry>("liorf_localization/mapping/odometry_incremental", 5,    &IMUPreintegration::odometryHandler, this, ros::TransportHints().tcpNoDelay());
        pubImuOdometry = nh.advertise<nav_msgs::Odometry> (odomTopic+"_incremental", 2000);
        pubBodyVel = nh.advertise<geometry_msgs::Vector3Stamped>("/body_vel", 2000);
#if USE_LEG
        subLeg = nh.subscribe<std_msgs::Float64MultiArray>("/low_state", 2000, &IMUPreintegration::legHandler, this, ros::TransportHints().tcpNoDelay());

        pubFootPositions["FR"] = nh.advertise<geometry_msgs::PointStamped>("/kinematics/FR_foot_position", 10);
        pubFootPositions["FL"] = nh.advertise<geometry_msgs::PointStamped>("/kinematics/FL_foot_position", 10);
        pubFootPositions["RR"] = nh.advertise<geometry_msgs::PointStamped>("/kinematics/RR_foot_position", 10);
        pubFootPositions["RL"] = nh.advertise<geometry_msgs::PointStamped>("/kinematics/RL_foot_position", 10);
        pubCalculatedLegOdometry = nh.advertise<nav_msgs::Odometry>("/calculated_leg_odometry", 10);
        pubLegOdomBias = nh.advertise<geometry_msgs::Vector3Stamped>("/leg_odom_bias", 2000);
#endif

        boost::shared_ptr<gtsam::PreintegrationParams> p = gtsam::PreintegrationParams::MakeSharedU(imuGravity);
        p->accelerometerCovariance  = gtsam::Matrix33::Identity(3,3) * pow(imuAccNoise, 2); // acc white noise in continuous
        p->gyroscopeCovariance      = gtsam::Matrix33::Identity(3,3) * pow(imuGyrNoise, 2); // gyro white noise in continuous
        p->integrationCovariance    = gtsam::Matrix33::Identity(3,3) * pow(1e-4, 2); // error committed in integrating position from velocities
        gtsam::imuBias::ConstantBias prior_imu_bias((gtsam::Vector(6) << 0, 0, 0, 0, 0, 0).finished());; // assume zero initial bias

        priorPoseNoise  = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2).finished()); // rad,rad,rad,m, m, m
        priorVelNoise   = gtsam::noiseModel::Isotropic::Sigma(3, 1e4); // m/s
        priorBiasNoise  = gtsam::noiseModel::Isotropic::Sigma(6, 1e-3); // 1e-2 ~ 1e-3 seems to be good
        correctionNoise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 0.05, 0.05, 0.05, 0.1, 0.1, 0.1).finished()); // rad,rad,rad,m, m, m
        correctionNoise2 = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1, 1, 1, 1, 1, 1).finished()); // rad,rad,rad,m, m, m
        noiseModelBetweenBias = (gtsam::Vector(6) << imuAccBiasN, imuAccBiasN, imuAccBiasN, imuGyrBiasN, imuGyrBiasN, imuGyrBiasN).finished();

#if USE_LEG
        priorBodyVelBiasNoise = gtsam::noiseModel::Isotropic::Sigma(3, 1e-1); // Body Vel Bias initial noise
        bodyVelBiasNoise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(3) << 0.05, 0.05, 0.05).finished());
        noiseModelBetweenBias2 = (gtsam::Vector(3) << 0.01, 0.01, 0.01).finished();

        // Leg Odom measurement noise
        gtsam::Matrix66 vel_noise_cov = gtsam::Matrix66::Identity();
        vel_noise_cov.block<3,3>(0,0) *= pow(0.5, 2); // angular velocity noise
        vel_noise_cov.block<3,3>(3,3) *= pow(0.2, 2); // linear velocity noise
        legIntegratorOpt_ = new PreintegratedLegOdom(vel_noise_cov);
#endif
        imuIntegratorImu_ = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias); // setting up the IMU integration for IMU message thread
        imuIntegratorOpt_ = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias); // setting up the IMU integration for optimization        
    }

    void resetOptimization()
    {
        gtsam::ISAM2Params optParameters;
        optParameters.relinearizeThreshold = 0.1;
        optParameters.relinearizeSkip = 1;
        optParameters.findUnusedFactorSlots = true;
        optimizer = gtsam::ISAM2(optParameters);

        gtsam::NonlinearFactorGraph newGraphFactors;
        graphFactors = newGraphFactors;

        gtsam::Values NewGraphValues;
        graphValues = NewGraphValues;
    }

    void resetParams()
    {
        lastImuT_imu = -1;
        doneFirstOpt = false;
        systemInitialized = false;
    }

    void odometryHandler(const nav_msgs::Odometry::ConstPtr& odomMsg)
    {
        std::lock_guard<std::mutex> lock(mtx);

        double currentCorrectionTime = ROS_TIME(odomMsg);

        // make sure we have imu data to integrate
        if (imuQueOpt.empty())
            return;

        float p_x = odomMsg->pose.pose.position.x;
        float p_y = odomMsg->pose.pose.position.y;
        float p_z = odomMsg->pose.pose.position.z;
        float r_x = odomMsg->pose.pose.orientation.x;
        float r_y = odomMsg->pose.pose.orientation.y;
        float r_z = odomMsg->pose.pose.orientation.z;
        float r_w = odomMsg->pose.pose.orientation.w;
        bool degenerate = (int)odomMsg->pose.covariance[0] == 1 ? true : false;
        gtsam::Pose3 lidarPose = gtsam::Pose3(gtsam::Rot3::Quaternion(r_w, r_x, r_y, r_z), gtsam::Point3(p_x, p_y, p_z));


        // 0. initialize system
        if (systemInitialized == false)
        {
            resetOptimization();

            // pop old IMU message
            while (!imuQueOpt.empty())
            {
                if (ROS_TIME(&imuQueOpt.front()) < currentCorrectionTime - delta_t)
                {
                    lastImuT_opt = ROS_TIME(&imuQueOpt.front());
                    imuQueOpt.pop_front();
                }
                else
                    break;
            }
            // initial pose
            prevPose_ = lidarPose.compose(lidar2Imu);
            gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, priorPoseNoise);
            graphFactors.add(priorPose);
            // initial velocity
            prevVel_ = gtsam::Vector3(0, 0, 0);
            gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_, priorVelNoise);
            graphFactors.add(priorVel);
            // initial bias
            prevBias_ = gtsam::imuBias::ConstantBias();
            gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(0), prevBias_, priorBiasNoise);
            graphFactors.add(priorBias);
#if USE_LEG
            prevBodyVelBias_ = gtsam::Vector3::Zero();
            graphFactors.add(gtsam::PriorFactor<gtsam::Vector3>(C(0), prevBodyVelBias_, priorBodyVelBiasNoise));
#endif
            // add values
            graphValues.insert(X(0), prevPose_);
            graphValues.insert(V(0), prevVel_);
            graphValues.insert(B(0), prevBias_);
#if USE_LEG
            graphValues.insert(C(0), prevBodyVelBias_);
#endif
            // optimize once
            optimizer.update(graphFactors, graphValues);
            graphFactors.resize(0);
            graphValues.clear();

            imuIntegratorImu_->resetIntegrationAndSetBias(prevBias_);
            imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_);
            
            key = 1;
            systemInitialized = true;
            return;
        }


        // reset graph for speed
        if (key == 100)
        {
            // get updated noise before reset
            gtsam::noiseModel::Gaussian::shared_ptr updatedPoseNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(X(key-1)));
            gtsam::noiseModel::Gaussian::shared_ptr updatedVelNoise  = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(V(key-1)));
            gtsam::noiseModel::Gaussian::shared_ptr updatedBiasNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(B(key-1)));
#if USE_LEG
            gtsam::noiseModel::Gaussian::shared_ptr updatedBodyVelBiasNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(C(key-1)));
#endif
            // reset graph
            resetOptimization();
            // add pose
            gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, updatedPoseNoise);
            graphFactors.add(priorPose);
            // add velocity
            gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_, updatedVelNoise);
            graphFactors.add(priorVel);
            // add bias
            gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(0), prevBias_, updatedBiasNoise);
            graphFactors.add(priorBias);
#if USE_LEG
            // add leg odom bias
            graphFactors.add(gtsam::PriorFactor<gtsam::Vector3>(C(0), prevBodyVelBias_, updatedBodyVelBiasNoise));
#endif
            // add values
            graphValues.insert(X(0), prevPose_);
            graphValues.insert(V(0), prevVel_);
            graphValues.insert(B(0), prevBias_);
#if USE_LEG
            graphValues.insert(C(0), prevBodyVelBias_);
#endif
            // optimize once
            optimizer.update(graphFactors, graphValues);
            graphFactors.resize(0);
            graphValues.clear();

            key = 1;
        }


        // 1. integrate imu data and optimize
        while (!imuQueOpt.empty())
        {
            // pop and integrate imu data that is between two optimizations
            sensor_msgs::Imu *thisImu = &imuQueOpt.front();
            double imuTime = ROS_TIME(thisImu);
            if (imuTime < currentCorrectionTime - delta_t)
            {
                double dt = (lastImuT_opt < 0) ? (1.0 / imuRate) : (imuTime - lastImuT_opt);
                imuIntegratorOpt_->integrateMeasurement(
                        gtsam::Vector3(thisImu->linear_acceleration.x, thisImu->linear_acceleration.y, thisImu->linear_acceleration.z),
                        gtsam::Vector3(thisImu->angular_velocity.x,    thisImu->angular_velocity.y,    thisImu->angular_velocity.z), dt);
                
                lastImuT_opt = imuTime;
                imuQueOpt.pop_front();
            }
            else
                break;
        }
        // add imu factor to graph
        const gtsam::PreintegratedImuMeasurements& preint_imu = dynamic_cast<const gtsam::PreintegratedImuMeasurements&>(*imuIntegratorOpt_);
        gtsam::ImuFactor imu_factor(X(key - 1), V(key - 1), X(key), V(key), B(key - 1), preint_imu);
        graphFactors.add(imu_factor);
        // add imu bias between factor
        graphFactors.add(gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>(B(key - 1), B(key), gtsam::imuBias::ConstantBias(),
                         gtsam::noiseModel::Diagonal::Sigmas(sqrt(imuIntegratorOpt_->deltaTij()) * noiseModelBetweenBias)));

#if USE_LEG
        // Leg Odometry Factor
        if (legIntegratorOpt_->deltaT() > 0.0)
        {   
            gtsam::SharedNoiseModel leg_noise_model = gtsam::noiseModel::Gaussian::Covariance(legIntegratorOpt_->preintCovariance());
            LegOdometryFactor leg_factor(X(key - 1), X(key), C(key - 1), *legIntegratorOpt_, leg_noise_model);
            graphFactors.add(leg_factor);
        }

        double dt_bias_rw = imuIntegratorOpt_->deltaTij();
        if (dt_bias_rw <= 0) 
        {
            dt_bias_rw = 1.0 / imuRate; 
        }
        graphFactors.add(gtsam::BetweenFactor<gtsam::Vector3>(C(key - 1), C(key), gtsam::Vector3::Zero(),
                        gtsam::noiseModel::Diagonal::Sigmas(sqrt(dt_bias_rw) * bodyVelBiasNoise->sigmas())));

        {
            // Leg Odometry velocity as a Prior Factor
            bool leg_odom_prior_to_add = false;
            gtsam::Vector3 current_leg_odom_velocity;
            {
                std::lock_guard<std::mutex> lock(leg_odom_mutex_);
                if (new_leg_odom_available_) {
                    current_leg_odom_velocity = latest_leg_odom_velocity_;
                    leg_odom_prior_to_add = true;
                    new_leg_odom_available_ = false;
                }
            }

            if (leg_odom_prior_to_add)
            {
                gtsam::Rot3 R_world_body = prevState_.pose().rotation();
                gtsam::Vector3 world_leg_odom_velocity = R_world_body.rotate(current_leg_odom_velocity);
    
                gtsam::Vector3 vel_prior_sigmas;
                vel_prior_sigmas << 0.1, 0.1, 0.1;
                auto vel_prior_noise = gtsam::noiseModel::Diagonal::Sigmas(vel_prior_sigmas);

                graphFactors.add(gtsam::PriorFactor<gtsam::Vector3>(V(key), world_leg_odom_velocity, vel_prior_noise));
            }
        }
#endif

        // add pose factor
        gtsam::Pose3 curPose = lidarPose.compose(lidar2Imu);
        gtsam::PriorFactor<gtsam::Pose3> pose_factor(X(key), curPose, degenerate ? correctionNoise2 : correctionNoise);
        graphFactors.add(pose_factor);
        // insert predicted values
        gtsam::NavState propState_ = imuIntegratorOpt_->predict(prevState_, prevBias_);
        graphValues.insert(X(key), propState_.pose());
        graphValues.insert(V(key), propState_.v());
        graphValues.insert(B(key), prevBias_);
#if USE_LEG
        graphValues.insert(C(key), prevBodyVelBias_);
#endif

        // optimize
        optimizer.update(graphFactors, graphValues);
        optimizer.update();
        // graphFactors.resize(0);
        // graphValues.clear();
        gtsam::Values result = optimizer.calculateEstimate();

        // overwrite the beginning of the preintegration for the next step.
        // gtsam::Values result = optimizer.calculateEstimate();
        prevPose_  = result.at<gtsam::Pose3>(X(key));
        prevVel_   = result.at<gtsam::Vector3>(V(key));
        prevState_ = gtsam::NavState(prevPose_, prevVel_);
        prevBias_  = result.at<gtsam::imuBias::ConstantBias>(B(key));
#if USE_LEG
        prevBodyVelBias_ = result.at<gtsam::Vector3>(C(key));

        geometry_msgs::Vector3Stamped bv_bias_msg;
        bv_bias_msg.header.stamp = odomMsg->header.stamp;
        bv_bias_msg.header.frame_id = baselinkFrame;
        bv_bias_msg.vector.x = prevBodyVelBias_.x();
        bv_bias_msg.vector.y = prevBodyVelBias_.y();
        bv_bias_msg.vector.z = prevBodyVelBias_.z();
        pubLegOdomBias.publish(bv_bias_msg);
#endif
        // gtsam::Vector3 body_velocity = prevPose_.rotation().unrotate(prevVel_);
        gtsam::Vector3 body_velocity = prevVel_;
        geometry_msgs::Vector3Stamped v_msg;
        v_msg.header.stamp = odomMsg->header.stamp;
        v_msg.header.frame_id = mapFrame;
        v_msg.vector.x = body_velocity.x();
        v_msg.vector.y = body_velocity.y();
        v_msg.vector.z = body_velocity.z();
        pubBodyVel.publish(v_msg);

        // reset the optimization preintegration object.
        imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_);
#if USE_LEG
        legIntegratorOpt_->resetIntegration();
#endif
        graphFactors.resize(0);
        graphValues.clear(); 

        // check optimization
#if USE_LEG
        if (failureDetection(prevVel_, prevBias_, prevBodyVelBias_))
        {
            resetParams();
            legIntegratorOpt_->resetIntegration();
            return;
        }
#else
        if (failureDetection(prevVel_, prevBias_))
        {
            resetParams();
            return;
        }
#endif


        // 2. After optiization, re-propagate imu odometry preintegration
        prevStateOdom = prevState_;
        prevBiasOdom  = prevBias_;
        // first pop imu message older than current correction data
        double lastImuQT = -1;
        while (!imuQueImu.empty() && ROS_TIME(&imuQueImu.front()) < currentCorrectionTime - delta_t)
        {
            lastImuQT = ROS_TIME(&imuQueImu.front());
            imuQueImu.pop_front();
        }
        // repropogate
        if (!imuQueImu.empty())
        {
            // reset bias use the newly optimized bias
            imuIntegratorImu_->resetIntegrationAndSetBias(prevBiasOdom);
            // integrate imu message from the beginning of this optimization
            for (int i = 0; i < (int)imuQueImu.size(); ++i)
            {
                sensor_msgs::Imu *thisImu = &imuQueImu[i];
                double imuTime = ROS_TIME(thisImu);
                double dt = (lastImuQT < 0) ? (1.0 / imuRate) :(imuTime - lastImuQT);

                imuIntegratorImu_->integrateMeasurement(gtsam::Vector3(thisImu->linear_acceleration.x, thisImu->linear_acceleration.y, thisImu->linear_acceleration.z),
                                                        gtsam::Vector3(thisImu->angular_velocity.x,    thisImu->angular_velocity.y,    thisImu->angular_velocity.z), dt);
                lastImuQT = imuTime;
            }
        }

        ++key;
        doneFirstOpt = true;
    }

#if USE_LEG
    bool failureDetection(const gtsam::Vector3& velCur, const gtsam::imuBias::ConstantBias& biasCur, const gtsam::Vector3 bodybiasCur)
    {
        float vel_th, bias_th;
        if (safeMode) {
            vel_th = 30.0;
            // bias_th = 0.5;
        } else {
            vel_th = 30.0;
            // bias_th = 1.0;
        }

        Eigen::Vector3f vel(velCur.x(), velCur.y(), velCur.z());
        if (vel.norm() > vel_th)
        {
            ROS_WARN("Large velocity, reset IMU-preintegration!");
            return true;
        }

        Eigen::Vector3f ba(biasCur.accelerometer().x(), biasCur.accelerometer().y(), biasCur.accelerometer().z());
        Eigen::Vector3f bg(biasCur.gyroscope().x(), biasCur.gyroscope().y(), biasCur.gyroscope().z());
        Eigen::Vector3f bv(bodybiasCur.x(), bodybiasCur.y(), bodybiasCur.z());
        if (ba.norm() > 1.0 || bg.norm() >1.0 )
        {
            ROS_WARN("Large ba or bg, reset IMU,LEG-preintegration!");
            return true;
        }
        else if (bv.norm() > 1.0)
        {
            ROS_WARN("Large bv, reset IMU,LEG-preintegration!");
            return true;
        }
        return false;
    }
#else
    bool failureDetection(const gtsam::Vector3& velCur, const gtsam::imuBias::ConstantBias& biasCur)
    {
        Eigen::Vector3f vel(velCur.x(), velCur.y(), velCur.z());
        float vel_th, bias_th;
        if (safeMode) {
            vel_th = 30.0;
            // bias_th = 0.5;
        } else {
            vel_th = 30.0;
            // bias_th = 1.0;
        }

        if (vel.norm() > vel_th)
        {
            ROS_WARN("Large velocity, reset IMU-preintegration!");
            return true;
        }

        Eigen::Vector3f ba(biasCur.accelerometer().x(), biasCur.accelerometer().y(), biasCur.accelerometer().z());
        Eigen::Vector3f bg(biasCur.gyroscope().x(), biasCur.gyroscope().y(), biasCur.gyroscope().z());
        if (ba.norm() > 1.0 || bg.norm() >1.0 )
        {
            ROS_WARN("Large ba or bg, reset IMU-preintegration!");
            return true;
        }
        return false;
    }
#endif

    void imuHandler(const sensor_msgs::Imu::ConstPtr& imu_raw)
    {
        std::lock_guard<std::mutex> lock(mtx);

        sensor_msgs::Imu thisImu = imuConverter(*imu_raw);

        imuQueOpt.push_back(thisImu);
        imuQueImu.push_back(thisImu);

        if (doneFirstOpt == false)
            return;

        double imuTime = ROS_TIME(&thisImu);
        double dt = (lastImuT_imu < 0) ? (1.0 / imuRate) : (imuTime - lastImuT_imu);
        lastImuT_imu = imuTime;

        // integrate this single imu message
        imuIntegratorImu_->integrateMeasurement(gtsam::Vector3(thisImu.linear_acceleration.x, thisImu.linear_acceleration.y, thisImu.linear_acceleration.z),
                                                gtsam::Vector3(thisImu.angular_velocity.x,    thisImu.angular_velocity.y,    thisImu.angular_velocity.z), dt);

        // predict odometry
        gtsam::NavState currentState = imuIntegratorImu_->predict(prevStateOdom, prevBiasOdom);

        // publish odometry
        nav_msgs::Odometry odometry;
        odometry.header.stamp = thisImu.header.stamp;
        odometry.header.frame_id = odometryFrame;
        odometry.child_frame_id = "odom_imu";

        // transform imu pose to ldiar
        gtsam::Pose3 imuPose = gtsam::Pose3(currentState.quaternion(), currentState.position());
        gtsam::Pose3 lidarPose = imuPose.compose(imu2Lidar);

        odometry.pose.pose.position.x = lidarPose.translation().x();
        odometry.pose.pose.position.y = lidarPose.translation().y();
        odometry.pose.pose.position.z = lidarPose.translation().z();
        odometry.pose.pose.orientation.x = lidarPose.rotation().toQuaternion().x();
        odometry.pose.pose.orientation.y = lidarPose.rotation().toQuaternion().y();
        odometry.pose.pose.orientation.z = lidarPose.rotation().toQuaternion().z();
        odometry.pose.pose.orientation.w = lidarPose.rotation().toQuaternion().w();
        
        odometry.twist.twist.linear.x = currentState.velocity().x();
        odometry.twist.twist.linear.y = currentState.velocity().y();
        odometry.twist.twist.linear.z = currentState.velocity().z();
        odometry.twist.twist.angular.x = thisImu.angular_velocity.x + prevBiasOdom.gyroscope().x();
        odometry.twist.twist.angular.y = thisImu.angular_velocity.y + prevBiasOdom.gyroscope().y();
        odometry.twist.twist.angular.z = thisImu.angular_velocity.z + prevBiasOdom.gyroscope().z();
        pubImuOdometry.publish(odometry);
    }

#if USE_LEG
    void legHandler(const std_msgs::Float64MultiArray::ConstPtr& leg_msg)
    {
        const int TIME_SEC_IDX = 0;
        const int TIME_NSEC_IDX = 1;
        const int JOINT_STATE_OFFSET_IDX = 2;
        const int FOOT_FORCE_OFFSET_IDX = 26;
        const int NUM_JOINTS = 12;
        const int NUM_FOOT_FORCES = 4;
        const int JOINT_DATA_PER_JOINT = 2; // q, dq

        std::lock(mtx, legMtx);
        std::lock_guard<std::mutex> lock1(mtx, std::adopt_lock);
        std::lock_guard<std::mutex> lock2(legMtx, std::adopt_lock);

        if (imuQueOpt.empty()) 
        {
            return;
        }

        double legOdomTime = static_cast<double>(leg_msg->data[TIME_SEC_IDX]);

        sensor_msgs::Imu closest_imu;
        double min_time_diff = std::numeric_limits<double>::max();

        for (const auto& imu_msg : imuQueOpt) 
        {
            double diff = std::abs(ROS_TIME(&imu_msg) - legOdomTime);
            if (diff < min_time_diff) 
            {
                min_time_diff = diff;
                closest_imu = imu_msg;
            }
        }
        
        if (min_time_diff > 0.01) 
        {
            // ROS_WARN("Large time diff between LegOdom and IMU: %f", min_time_diff);
            return;
        }
        
        Eigen::Vector3d body_angular_velocity(closest_imu.angular_velocity.x,
                                              closest_imu.angular_velocity.y,
                                              closest_imu.angular_velocity.z);

        // 0. Leg odometry calculate
        const std::vector<std::string> leg_prefixes = {"FR", "FL", "RR", "RL"};
        
        Eigen::Vector3d joint_angles;
        Eigen::Vector3d joint_velocities;
        Eigen::Vector3d total_velocity_estimate = Eigen::Vector3d::Zero();
        Eigen::Vector3d legodom_velocity = Eigen::Vector3d::Zero();
        int contact_count = 0;

        for (int i = 0; i < 4; ++i) {
            const auto& prefix = leg_prefixes[i];
            
            // 1. Joint measurements
            int joint_data_offset = JOINT_STATE_OFFSET_IDX + i * (3 * JOINT_DATA_PER_JOINT);
            
            joint_angles[0] = leg_msg->data[joint_data_offset + 0]; // Hip joint q
            joint_angles[1] = leg_msg->data[joint_data_offset + 2]; // Thigh joint q
            joint_angles[2] = leg_msg->data[joint_data_offset + 4]; // Calf joint q

            joint_velocities[0] = leg_msg->data[joint_data_offset + 1]; // Hip joint dq
            joint_velocities[1] = leg_msg->data[joint_data_offset + 3]; // Thigh joint dq
            joint_velocities[2] = leg_msg->data[joint_data_offset + 5]; // Calf joint dq

            // 2. Forward kinematics, foot position
            Transform T_base_foot = kinematics_calculator_.getFootPose(prefix, joint_angles);
            Eigen::Vector3d foot_position = T_base_foot.translation();

            geometry_msgs::PointStamped point_msg;
            point_msg.header.stamp = ros::Time::now();
            point_msg.header.frame_id = "base_link";
            point_msg.point.x = foot_position.x();
            point_msg.point.y = foot_position.y();
            point_msg.point.z = foot_position.z();
            pubFootPositions[prefix].publish(point_msg);

            // 3. Contact detection and Leg Odometry calculation
            if (leg_msg->data[FOOT_FORCE_OFFSET_IDX + i] > 20) {
                contact_count++;

                Eigen::Matrix3d jacobian = kinematics_calculator_.getFootJacobian(prefix, joint_angles);
                
                Eigen::Vector3d foot_velocity_from_joints = jacobian * joint_velocities;
                Eigen::Vector3d body_velocity_estimate = -(body_angular_velocity.cross(foot_position) + foot_velocity_from_joints);
                
                total_velocity_estimate += body_velocity_estimate;
            }
        }
        // 4. Final body velocity
        if (contact_count > 0) {
            legodom_velocity = total_velocity_estimate / static_cast<double>(contact_count);

            nav_msgs::Odometry leg_odom_msg;
            leg_odom_msg.header.stamp = ros::Time::now();
            leg_odom_msg.header.frame_id = "base_link";
            leg_odom_msg.child_frame_id = "base_link";

            leg_odom_msg.twist.twist.linear.x = legodom_velocity.y();
            leg_odom_msg.twist.twist.linear.y = -legodom_velocity.x();
            leg_odom_msg.twist.twist.linear.z = legodom_velocity.z();
            leg_odom_msg.twist.twist.angular.x = body_angular_velocity.x();
            leg_odom_msg.twist.twist.angular.y = body_angular_velocity.y();
            leg_odom_msg.twist.twist.angular.z = body_angular_velocity.z();
            
            leg_odom_msg.pose.pose.orientation.w = 1.0;

            pubCalculatedLegOdometry.publish(leg_odom_msg);
        }

        {
            std::lock_guard<std::mutex> lock(leg_odom_mutex_);
            latest_leg_odom_velocity_ = gtsam::Vector3(legodom_velocity.y(), -legodom_velocity.x(), legodom_velocity.z());
            new_leg_odom_available_ = true;
        }

        // 5. Calculate dt and integrate measurements
        if (lastLegOdomTime < 0) 
        {
            lastLegOdomTime = legOdomTime;
            return;
        }
        double dt = legOdomTime - lastLegOdomTime;
        if (dt > 0) 
        {
            legIntegratorOpt_->integrateMeasurement(gtsam::Vector3(legodom_velocity.y(),-legodom_velocity.x(),legodom_velocity.z()), 
                                                    body_angular_velocity, prevBias_, dt);
            lastLegOdomTime = legOdomTime;
            // ROS_WARN("dt okay: %f", dt);
        }
        // else 
        // {
        //     ROS_WARN("invalid dt in legodomhandler... SKIP integrateMeasurement! %f", dt);
        // }
    }
#endif
};


int main(int argc, char** argv)
{
    ros::init(argc, argv, "roboat_loam");
    
    IMUPreintegration ImuP;

    TransformFusion TF;

    ROS_INFO("\033[1;32m----> IMU Preintegration Started.\033[0m");
    
#if USE_LEG
    ROS_INFO("\033USE LEG !!!!!!!!!!!\033[0m");
#endif
    ros::MultiThreadedSpinner spinner(4);
    spinner.spin();
    
    return 0;
}
