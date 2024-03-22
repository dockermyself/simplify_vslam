#ifndef _FEATUREMATCH_H
#define _FEATUREMATCH_H
// #include "g2o/math_groups/se3quat.h"
#include "g2o/solvers/structure_only/structure_only_solver.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/solver.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/cholmod/linear_solver_cholmod.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/types/sba/types_six_dof_expmap.h"
#include <Eigen/Core>
#include <vector>
#include <memory>
#include <unordered_map>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>
#include <opencv2/opencv.hpp>
#include <list>

typedef g2o::VertexSE3Expmap iVertexSE3Expmap;
typedef g2o::VertexSE3Expmap jVertexSE3Expmap;

cv::Mat SampleGaussian(int row, int col, double mean, double sigma);
int SampleUniform(int min, int max);
cv::Mat SampleUniform(int row, int col, int min, int max);
Eigen::MatrixXd SampleLambda(int row, int col);
Sophus::SE3d SamplePose(int angle = 15);
Eigen::Vector2d SampleNoise();

class Camera
{
    const double fx;
    const double fy;
    const double cx;
    const double cy;
    const int image_h;
    const int image_w;
    const float min_depth;
    const float max_depth;

public:
    Camera(double _fx, double _fy, double _cx, double _cy, int _h, int _w, float _min_depth, float _max_depth)
        : fx(_fx), fy(_fy), cx(_cx), cy(_cy), image_h(_h), image_w(_w),
          min_depth(_min_depth), max_depth(_max_depth) {}

    bool project2d(const Eigen::Vector3d &p, Eigen::Vector2d &pixel) const
    {
        if (p.z() < min_depth || p.z() > max_depth)
            return false;
        Eigen::Vector2d uv(p.x() / p.z() * fx + cx, p.y() / p.z() * fy + cy);
        if (uv.x() < 0 || uv.x() >= image_w || uv.y() < 0 || uv.y() >= image_h)
            return false;
        pixel = uv;
        return true;
    }
    Eigen::Vector3d normalize3d(const Eigen::Vector2d &p) const
    {
        return {(p.x() - cx) / fx, (p.y() - cy) / fy, 1};
    }
    Eigen::Vector2d normalize2d(const Eigen::Vector2d &p) const
    {
        return {(p.x() - cx) / fx, (p.y() - cy) / fy};
    }
};

class Frame
{
public:
    static const int image_h;
    static const int image_w;
    static const int block_size;
    static const int feature_row;
    static const int feature_col;
    static const Camera camera;
    static int id_counter;

private:
    int _id;
    // 相机姿态
    Eigen::Matrix3d _Rwc;
    Eigen::Vector3d _twc;
    // Sophus::SE3d _Tcw;
    std::vector<Eigen::Vector2d> _featureArray; // Replace std::array with std::vector
    std::vector<bool> _featureMask;

public:
    Frame(const Sophus::SE3d &Twc) : _id(id_counter++), _Rwc(Twc.rotationMatrix()), _twc(Twc.translation())
    {
        _featureArray.resize(feature_row * feature_col, Eigen::Vector2d(0, 0)); // Resize the vector and initialize with default values
        _featureMask.resize(feature_row * feature_col, false);
        // std::cout << "Frame " << _id << " is created" << std::endl;
    }
    Frame(const Eigen::Matrix3d &R, const Eigen::Vector3d &t) : _id(id_counter++), _Rwc(R), _twc(t)
    {
        _featureArray.resize(feature_row * feature_col, Eigen::Vector2d(0, 0));
        _featureMask.resize(feature_row * feature_col, false);
        // std::cout << "Frame " << _id << " is created" << std::endl;
    }

    // 特征提取
    void ExtractFeaturePoints()
    {
        for (int y = 0; y < feature_row; y++)
        {
            for (int x = 0; x < feature_col; x++)
            {
                if (!mask_at(x, y))
                {
                    Eigen::Vector2d p = anchorPoint(x, y);
                    Eigen::Vector2d noise = SampleNoise();
                    p += noise * block_size / 2.0;
                    feature_set(x, y, p);
                    _featureMask[y * feature_col + x] = true;
                }
            }
        }
    }

    Eigen::Vector2d anchorPoint(int x, int y)
    {
        return {x * block_size + block_size / 2.0f, y * block_size + block_size / 2.0f};
    }

    bool setFeaturePoint(const Eigen::Vector2d &point)
    {
        int x = point.x() / block_size;
        int y = point.y() / block_size;
        if (x >= 0 && x < feature_col && y >= 0 && y < feature_row)
        {
            if (!mask_at(x, y))
            {
                feature_set(x, y, point);
                _featureMask[y * feature_col + x] = true;
            }
            else
            {
                return false;
            }
        }
        return true;
    }
    void feature_set(int x, int y, const Eigen::Vector2d &p)
    {
        _featureArray[y * feature_col + x] = p;
    }

    Eigen::Vector2d feature_at(int x, int y)
    {
        return _featureArray[y * feature_col + x];
    }

    bool mask_at(int x, int y)
    {
        return _featureMask[y * feature_col + x];
    }
    void reset_mask()
    {
        for (int y = 0; y < feature_row; y++)
        {
            for (int x = 0; x < feature_col; x++)
            {
                _featureMask[y * feature_col + x] = false;
            }
        }
    }
    // 获取mask(i,j)为false的特征点索引
    std::vector<Eigen::Vector2i> getUnmaskFeatureIndex()
    {
        std::vector<Eigen::Vector2i> index;
        for (int y = 0; y < feature_row; y++)
        {
            for (int x = 0; x < feature_col; x++)
            {
                if (!mask_at(x, y))
                {
                    index.push_back({x, y});
                }
            }
        }
        return index;
    }

    std::vector<Eigen::Vector2i> getMaskFeatureIndex()
    {
        std::vector<Eigen::Vector2i> index;
        for (int y = 0; y < feature_row; y++)
        {
            for (int x = 0; x < feature_col; x++)
            {
                if (mask_at(x, y))
                {
                    index.push_back({x, y});
                }
            }
        }
        return index;
    }

    void classfyMaskFeatureIndex(std::vector<Eigen::Vector2i> &mask_index, std::vector<Eigen::Vector2i> &unmask_index)
    {
        for (int y = 0; y < feature_row; y++)
        {
            for (int x = 0; x < feature_col; x++)
            {
                if (mask_at(x, y))
                {
                    mask_index.push_back({x, y});
                }
                else
                {
                    unmask_index.push_back({x, y});
                }
            }
        }
    }

    int id() const { return _id; }

    void setPose(const Sophus::SE3d &Twc)
    {
        // _Tcw = Twc;
        _Rwc = Twc.rotationMatrix();
        _twc = Twc.translation();
    }
    void setPose(const Eigen::Matrix3d &R, const Eigen::Vector3d &t)
    {
        _Rwc = R;
        _twc = t;
    }
    Sophus::SE3d getPose() const { return Sophus::SE3d(_Rwc, _twc); }
    const Eigen::Matrix3d &rotation() const { return _Rwc; }
    const Eigen::Vector3d &translation() const { return _twc; }
    Eigen::Vector2i featurePostion(const Eigen::Vector2d &pixel)
    {
        int x = pixel.x() / block_size;
        int y = pixel.y() / block_size;
        return {x, y};
    }
    void feature_show()
    {
        cv::Mat img(image_h, image_w, CV_8UC3, cv::Scalar(0, 0, 0));
        for (int y = 0; y < feature_row; y++)
        {
            for (int x = 0; x < feature_col; x++)
            {
                if (mask_at(x, y))
                {
                    Eigen::Vector2d p = feature_at(x, y);
                    cv::circle(img, cv::Point(p.x(), p.y()), 2, cv::Scalar(0, 0, 255), -1);
                }
            }
        }

        cv::imshow("feature", img);
        cv::waitKey(0);
    }
    ~Frame()
    {
        // std::cout << "Frame " << _id << " is deleted" << std::endl;
    }
};

class MapPoint
{
    static int id_counter;
    struct Observer
    {
        Eigen::Vector2i xy;
        std::weak_ptr<Frame> frame;
        Observer(const Eigen::Vector2i &xy, const std::shared_ptr<Frame> &frame) : xy(xy), frame(frame) {}
    };
    int _Landmark_id;
    double _lambda; // 逆深度
    std::list<Observer> _observers;

public:
    using ObserverIterator = std::list<Observer>::iterator;

public:
    MapPoint(double lambda, const std::shared_ptr<Frame> &frame, const Eigen::Vector2i &feature_index)
        : _Landmark_id(id_counter++), _lambda(lambda)
    {
        _observers.emplace_back(feature_index, frame);
    }
    int id() const { return _Landmark_id; }
    double lambda() const { return _lambda; }
    void setLambda(double lambda) { _lambda = lambda; }

    // 世界坐标系下的坐标
    Eigen::Vector3d Get3dworld()
    {
        std::shared_ptr<Frame> framePtr = _observers.front().frame.lock(); // Convert weak_ptr to shared_ptr
        if (framePtr)
        {
            const auto &p = _observers.front().xy;
            Eigen::Vector3d Pc = 1.0 / _lambda * Frame::camera.normalize3d(framePtr->feature_at(p.x(), p.y()));
            return framePtr->getPose() * Pc;
        }
        else
        {
            // Handle the case when the frame is no longer available
            std::runtime_error("Frame is no longer available");
            return Eigen::Vector3d::Zero();
        }
    }
    // 相机坐标系下的坐标
    Eigen::Vector3d Get3dCamera()
    {
        std::shared_ptr<Frame> framePtr = _observers.front().frame.lock();
        if (framePtr)
        {
            const auto &p = _observers.front().xy;
            return 1.0 / _lambda * Frame::camera.normalize3d(framePtr->feature_at(p.x(), p.y()));
        }
        else
        {
            // Handle the case when the frame is no longer available
            std::runtime_error("Frame is no longer available");
            return Eigen::Vector3d::Zero();
        }
    }
    // frame id > back frame id
    void addObservation(const std::shared_ptr<Frame> &frame, const Eigen::Vector2i &feature_index)
    {
        _observers.emplace_back(feature_index, frame);
    }
    void removeObservation(const std::shared_ptr<Frame> &frame)
    {
        for (auto it = _observers.begin(); it != _observers.end();)
        {
            if (it->frame.lock() == frame)
            {
                it = _observers.erase(it);
            }
            else
            {
                ++it;
            }
        }
    }
    void removeObservation(const std::shared_ptr<Frame> &frame, const Eigen::Vector2i &feature_index)
    {
        for (auto it = _observers.begin(); it != _observers.end();)
        {
            if (it->frame.lock() == frame && it->xy == feature_index)
            {
                it = _observers.erase(it);
                break;
            }
            else
            {
                ++it;
            }
        }
    }
    int obsSize() { return _observers.size(); }
    ObserverIterator obsBegin() { return _observers.begin(); }
    ObserverIterator obsEnd() { return _observers.end(); }

    ~MapPoint()
    {
        // std::cout << "MapPoint " << _Landmark_id << " is deleted" << std::endl;
    }
};

class Map
{
    struct KeyType
    {
        int frame_id;
        short x;
        short y;
        bool operator<(const KeyType &k) const
        {
            if (frame_id < k.frame_id)
                return true;
            if (frame_id > k.frame_id)
                return false;
            if (x < k.x)
                return true;
            if (x > k.x)
                return false;
            if (y < k.y)
                return true;
            return false;
        }
    };
    using Landmark3dContainer = std::map<KeyType, std::shared_ptr<MapPoint>>;
    Landmark3dContainer _landmark3d_container;

public:
    using Iterator = Landmark3dContainer::iterator;
    using Key = KeyType;
    Map() {}
    void addMapPoint(int frame_id, int x, int y, const std::shared_ptr<MapPoint> &mp)
    {
        short _x = x;
        short _y = y;
        _landmark3d_container[{frame_id, _x, _y}] = mp;
    }
    void addMapPoint(int frame_id, int x, int y, double lambda, const std::shared_ptr<Frame> frame)
    {
        short _x = x;
        short _y = y;
        _landmark3d_container[{frame_id, _x, _y}] = std::make_shared<MapPoint>(lambda, frame, Eigen::Vector2i(x, y));
    }
    bool getMapPoint(int frame_id, int x, int y, std::shared_ptr<MapPoint> &mp)
    {
        short _x = x;
        short _y = y;
        for (const auto &landmark3d_pair : _landmark3d_container)
        {
            if (landmark3d_pair.first.frame_id == frame_id && landmark3d_pair.first.x == _x && landmark3d_pair.first.y == _y)
            {
                mp = landmark3d_pair.second;
                return true;
            }
        }
    }
    int match(const std::shared_ptr<Frame> &frame_ptr, const Sophus::SE3d &pose,
              std::vector<std::pair<std::shared_ptr<MapPoint>, Eigen::Vector2i>> &reprojection_pair)
    {
        int reprojection_num = 0;
        for (const auto &landmark3d_pair : _landmark3d_container)
        {
            std::shared_ptr<MapPoint> landmark3d = landmark3d_pair.second;
            Eigen::Vector3d pw = landmark3d->Get3dworld();
            Eigen::Vector3d pcj = pose.inverse() * pw;
            Eigen::Vector2d pixel;
            bool flag = Frame::camera.project2d(pcj, pixel);

            if (flag && frame_ptr->setFeaturePoint(pixel))
            {
                reprojection_pair.push_back({landmark3d, frame_ptr->featurePostion(pixel)});
                ++reprojection_num;
            }
        }

        return reprojection_num;
    }
    int size() { return _landmark3d_container.size(); }
    Iterator begin() { return _landmark3d_container.begin(); }
    Iterator end() { return _landmark3d_container.end(); }

    ~Map() {}
};

// 定义逆深度顶点
class VertexInverseDepth : public g2o::BaseVertex<1, double>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexInverseDepth() {}
    bool read(std::istream & /*is*/) override
    {
        std::cerr << __PRETTY_FUNCTION__ << " not implemented yet" << std::endl;
        return false;
    }

    bool write(std::ostream & /*os*/) const override
    {
        std::cerr << __PRETTY_FUNCTION__ << " not implemented yet" << std::endl;
        return false;
    }
    virtual void setToOriginImpl() {}
    virtual void oplusImpl(const double *update)
    {
        _estimate += update[0];
    }
};

// 定义重投影误差边
/*
观测：(ui,vi),(uj,vj)
顶点：VertexInverseDepth ,iVertexSE3Expmap，jVertexSE3Expmap
*/

class EdgeReprojection : public g2o::BaseMultiEdge<2, Eigen::Vector4d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeReprojection() {}
    bool read(std::istream & /*is*/) override
    {
        std::cerr << __PRETTY_FUNCTION__ << " not implemented yet" << std::endl;
        return false;
    }

    bool write(std::ostream & /*os*/) const override
    {
        std::cerr << __PRETTY_FUNCTION__ << " not implemented yet" << std::endl;
        return false;
    }
    virtual void computeError() override
    {
        const iVertexSE3Expmap *vi = static_cast<const iVertexSE3Expmap *>(_vertices[0]);
        const jVertexSE3Expmap *vj = static_cast<const jVertexSE3Expmap *>(_vertices[1]);
        const VertexInverseDepth *vd = static_cast<const VertexInverseDepth *>(_vertices[2]);
        Eigen::Vector2d uvi = _measurement.head<2>();
        Eigen::Vector2d uvj = _measurement.tail<2>();
        const double &lambda = vd->estimate();
        Eigen::Vector3d Pci = Eigen::Vector3d(uvi(0), uvi(1), 1) / lambda;
        Eigen::Vector3d Pw = vi->estimate().map(Pci);
        Eigen::Vector3d Pcj = vj->estimate().inverse().map(Pw);
        Eigen::Vector2d uvj_ = Pcj.head<2>() / Pcj(2);
        _error = uvj - uvj_;
    }
  
};
#endif