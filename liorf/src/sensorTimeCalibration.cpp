#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <deque>

class ImuTimeCorrector
{
public:
    ImuTimeCorrector()
    {
        sub_ = nh_.subscribe("/utlidar/imu", 50, &ImuTimeCorrector::imuCallback, this);
        pub_ = nh_.advertise<sensor_msgs::Imu>("/imu_corrected", 50);
    }

    void imuCallback(const sensor_msgs::ImuConstPtr& msg)
    {
        ros::Time a_time = msg->header.stamp;     // Time from device (possibly from PC A)
        ros::Time b_time = ros::Time::now();       // Local time (PC B)

        ros::Duration offset = b_time - a_time;    // Time difference (B - A)

        offset_buffer_.push_back(offset);
        if (offset_buffer_.size() > max_samples_)
            offset_buffer_.pop_front();

        // Compute average offset
        ros::Duration avg_offset(0.0);
        for (const auto& d : offset_buffer_)
            avg_offset += d;
        avg_offset *= 1.0 / offset_buffer_.size();

        estimated_offset_ = avg_offset;

        // Create new IMU message with corrected timestamp
        sensor_msgs::Imu corrected_msg = *msg;
        // corrected_msg.header.stamp = b_time - estimated_offset_;  // or ros::Time::now() - estimated_offset_
        corrected_msg.header.stamp = ros::Time::now();

        pub_.publish(corrected_msg);

        // ROS_INFO_STREAM_THROTTLE(5.0, "[IMU Corrector] Estimated time offset: "
        //                              << estimated_offset_.toSec() << " seconds");
    }

private:
    ros::NodeHandle nh_;
    ros::Subscriber sub_;
    ros::Publisher pub_;

    std::deque<ros::Duration> offset_buffer_;
    ros::Duration estimated_offset_{0.0};
    const size_t max_samples_ = 30;
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "imu_time_corrector");
    ImuTimeCorrector corrector;
    ros::spin();
    return 0;
}