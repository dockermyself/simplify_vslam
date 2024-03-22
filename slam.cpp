#include <iostream>
#include "slam.h"
#include <time.h>
#include <pangolin/pangolin.h>
#define MAX_DEPTH 12.0f
#define MIN_DEPTH 0.25f

cv::Mat SampleGaussian(int row, int col, double mean, double sigma)
{
    cv::Mat mat(row, col, CV_64F);
    cv::RNG rng;
    rng.state = std::rand() + clock();
    rng.fill(mat, cv::RNG::NORMAL, mean, sigma / 3.0);
    // clip
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            double &val = mat.at<double>(i, j);
            if (val < mean - sigma)
                val = mean - sigma;
            else if (val > mean + sigma)
                val = mean + sigma;
        }
    }
    return mat;
}
int SampleUniform(int min, int max)
{
    cv::RNG rng;
    rng.state = std::rand() + clock();
    return rng.uniform(min, max);
}
cv::Mat SampleUniform(int row, int col, int min, int max)
{
    cv::RNG rng;
    rng.state = std::rand() + clock();
    cv::Mat mat(row, col, CV_64F);
    rng.fill(mat, cv::RNG::UNIFORM, min, max);
    return mat;
}
Eigen::MatrixXd SampleLambda(int row, int col)
{
    cv::Mat mat = SampleUniform(row, col, 1, 10);
    return Eigen::Map<Eigen::MatrixXd>(mat.ptr<double>(), row, col);
}

Sophus::SE3d SamplePose(int angle)
{
    int x = SampleUniform(-angle, angle);
    int y = SampleUniform(-angle, angle);
    int z = SampleUniform(-angle, angle);
    cv::Mat t = SampleGaussian(3, 1, 0.25, 0.05);
    Eigen::Vector3d r = {x / 180.0f * M_PI, y / 180.0f * M_PI, z / 180.0f * M_PI};
    cv::Mat noise = SampleGaussian(3, 1, 0, 1 / 180.0f * M_PI);
    r = Eigen::Vector3d(r.x() + noise.at<double>(0, 0), r.y() + noise.at<double>(1, 0), noise.at<double>(2, 0));
    return Sophus::SE3d(Sophus::SO3d::exp(r), Eigen::Vector3d(t.at<double>(0, 0), t.at<double>(1, 0), t.at<double>(2, 0)));
}

Eigen::Vector2d SampleNoise()
{
    double mean = 0;
    double sigma = 1;
    cv::Mat noise = SampleGaussian(2, 1, mean, sigma);
    return Eigen::Vector2d(noise.at<double>(0, 0), noise.at<double>(1, 0));
}
double SampleScaleNoise()
{
    return 0.05 * (SampleUniform(-3, 1) + SampleGaussian(1, 1, 0, 1).at<double>(0, 0));
}
Eigen::Matrix3d SamplePoseNoise()
{
    cv::Mat noise = 0.5 - SampleGaussian(3, 1, 0, 1);
    Eigen::Vector3d r = 5 / 180.0f * M_PI * Eigen::Vector3d(r.x() + noise.at<double>(0, 0), r.y() + noise.at<double>(1, 0), r.z() + noise.at<double>(2, 0));
    return Sophus::SO3d::exp(r).matrix();
}
double SampleLambdaNoise(double lambda)
{
    double d = SampleUniform(-20, 20) * 0.01;
    return lambda / (1 + lambda * d);
}

Eigen::Vector3d SampleTranslationNoise()
{
    return {SampleUniform(-10, 10) * 0.01, SampleUniform(-10, 10) * 0.01, SampleUniform(-10, 10) * 0.01};
}

void DrawCamera(const Eigen::Matrix4d &Tcw, Eigen::Vector3d color = Eigen::Vector3d(0, 1, 0))
{
    const float size = 0.1;
    glPushMatrix();
    glMultMatrixd(Tcw.data());
    // 绘制相机外轮廓
    glLineWidth(2);
    glBegin(GL_LINES);

    glColor3f(color[0], color[1], color[2]);
    glVertex3f(0, 0, 0);
    glVertex3f(size, size, size);

    glVertex3f(0, 0, 0);
    glVertex3f(size, -size, size);

    glVertex3f(0, 0, 0);
    glVertex3f(size, -size, -size);

    glVertex3f(0, 0, 0);
    glVertex3f(size, size, -size);

    glVertex3f(size, size, size);
    glVertex3f(size, -size, size);

    glVertex3f(size, -size, size);
    glVertex3f(size, -size, -size);
    glVertex3f(size, -size, -size);
    glVertex3f(size, size, -size);
    glVertex3f(size, size, -size);
    glVertex3f(size, size, size);
    glEnd();

    glPopMatrix();
}

void DrawPoint(const Eigen::Vector3d &landmark, const Eigen::Vector3d color = Eigen::Vector3d(0, 1, 0))
{
    glPointSize(3);
    glBegin(GL_POINTS);
    glColor3f(color[0], color[1], color[2]);
    glVertex3d(landmark[0], landmark[1], landmark[2]);
    glEnd();
}

typedef std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> TrajectoryType;
typedef std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> LandmarkType;

void DrawTrajectoryAndLandmark(const TrajectoryType &init_trajectory, const TrajectoryType &true_trajectory, const TrajectoryType &opti_trajectory,
                               const LandmarkType &init_landmarks, const LandmarkType &true_landmarks, const LandmarkType &opti_landmarks)
{
    // create pangolin window and plot the trajectory
    pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    //  fx = 720, fy = 720, cx = 640, cy = 360;
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, 720, 720, 640, 360, 0.1, 1000),
        pangolin::ModelViewLookAt(0, 0, -1, 0, 0, 0, pangolin::AxisY));

    pangolin::View &d_cam = pangolin::CreateDisplay()
                                .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
                                .SetHandler(new pangolin::Handler3D(s_cam));

    while (pangolin::ShouldQuit() == false)
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        glLineWidth(1);
        for (size_t i = 0; i < init_trajectory.size() - 1; i++)
        {
            DrawCamera(init_trajectory[i].matrix(), Eigen::Vector3d(0, 1, 0));
            glColor3f(0.0f, 1.0f, 0.0f);
            glBegin(GL_LINES);
            auto p1 = init_trajectory[i], p2 = init_trajectory[i + 1];
            glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
            glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
            glEnd();
        }

        for (size_t i = 0; i < true_trajectory.size() - 1; i++)
        {
            DrawCamera(true_trajectory[i].matrix(), Eigen::Vector3d(0, 0, 0));
            glColor3f(0.0f, 0.0f, 1.0f);
            glBegin(GL_LINES);
            auto p1 = true_trajectory[i], p2 = true_trajectory[i + 1];
            glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
            glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
            glEnd();
        }

        for (size_t i = 0; i < opti_trajectory.size() - 1; i++)
        {
            DrawCamera(opti_trajectory[i].matrix(), Eigen::Vector3d(1, 0, 0));
            glColor3f(1.0f, 0.0f, 0.0f);
            glBegin(GL_LINES);
            auto p1 = opti_trajectory[i], p2 = opti_trajectory[i + 1];
            glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
            glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
            glEnd();
        }

        // for (size_t i = 0; i < true_landmarks.size(); i++)
        // {
        //     DrawPoint(init_landmarks[i], Eigen::Vector3d(0, 1, 0)); // green

        //     DrawPoint(true_landmarks[i], Eigen::Vector3d(0, 0, 0)); // blue

        //     DrawPoint(opti_landmarks[i], Eigen::Vector3d(1, 0, 0)); // red
        // }

        // 坐标轴
        pangolin::glDrawAxis(0.1);
        pangolin::FinishFrame();
    }
}
// 计算地图误差
void LandmarkMSE(LandmarkType &x, LandmarkType &y)
{
    double rmse_landmark = 0;
    double rmse_landmark_max = 0;
    for (size_t i = 0; i < x.size(); i++)
    {
        Eigen::Vector3d p1 = x[i], p2 = y[i];
        double error = (p2 - p1).norm();
        // std::cout << "Landmark Error = " << error << std::endl;
        rmse_landmark += error * error;
        if (error > rmse_landmark_max)
        {
            rmse_landmark_max = error;
        }
    }
    rmse_landmark = rmse_landmark / double(x.size());
    rmse_landmark = sqrt(rmse_landmark);
    std::cout << "Landmark RMSE = " << rmse_landmark << std::endl;
    std::cout << "Landmark Max Error = " << rmse_landmark_max << std::endl;
}
// 计算轨迹误差
void TrajectoryMSE(TrajectoryType &x, TrajectoryType &y)
{
    // compute rmse
    double rmse_trajectory = 0;
    double rmse_trajectory_max = 0;
    for (size_t i = 0; i < x.size(); i++)
    {
        Sophus::SE3d p1 = x[i], p2 = y[i];
        double error = (p2.inverse() * p1).log().norm();
        rmse_trajectory += error * error;
        if (error > rmse_trajectory_max)
        {
            rmse_trajectory_max = error;
        }
    }
    rmse_trajectory = rmse_trajectory / double(x.size());
    rmse_trajectory = sqrt(rmse_trajectory);
    std::cout << "Trajectory RMSE = " << rmse_trajectory << std::endl;
    std::cout << "Trajectory Max Error = " << rmse_trajectory_max << std::endl;
}

int Frame::id_counter = 0;
const int Frame::image_h = 720;
const int Frame::image_w = 1280;
const int Frame::block_size = 40;
const int Frame::feature_row = Frame::image_h / Frame::block_size;
const int Frame::feature_col = Frame::image_w / Frame::block_size;
const Camera Frame::camera(720, 720, 640, 360, Frame::image_h, Frame::image_w, MIN_DEPTH, MAX_DEPTH);
int MapPoint::id_counter = 0;

int main(int argc, char *argv[])
{
    const int frame_size = 100;
    Map map_manager;
    std::vector<std::shared_ptr<Frame>> frame_container;
    std::shared_ptr<Frame> last_frame_ptr;
    for (int i = 0; i < frame_size; i++)
    {

        if (!last_frame_ptr)
        {
            std::shared_ptr<Frame> frame_ptr = std::make_shared<Frame>(Sophus::SE3d());
            frame_ptr->ExtractFeaturePoints();
            Eigen::MatrixXd lambda_matrix = SampleLambda(Frame::feature_row, Frame::feature_col);
            for (int y = 0; y < Frame::feature_row; y++)
            {
                for (int x = 0; x < Frame::feature_col; x++)
                {
                    double lambda = 1.0 / lambda_matrix(y, x);
                    std::shared_ptr<MapPoint> landmark3d_ptr = std::make_shared<MapPoint>(lambda, frame_ptr, Eigen::Vector2i(x, y));
                    map_manager.addMapPoint(frame_ptr->id(), x, y, landmark3d_ptr);
                }
            }
            last_frame_ptr = frame_ptr;
            frame_container.push_back(std::move(frame_ptr));
        }
        else
        {
            int reprojection_num = 0;
            std::shared_ptr<Frame> frame_ptr = std::make_shared<Frame>(Sophus::SE3d());
            Sophus::SE3d pose = Sophus::SE3d();
            std::vector<std::pair<std::shared_ptr<MapPoint>, Eigen::Vector2i>> reprojection_pair;

            while (reprojection_num < 2 * Frame::feature_row * Frame::feature_col / 3)
            {
                frame_ptr->reset_mask();
                reprojection_pair.clear();
                // Twi 表示 i帧到世界坐标系的变换,Tij表示 i帧到j帧的变换
                Sophus::SE3d Twi = Sophus::SE3d(last_frame_ptr->rotation(), last_frame_ptr->translation());
                Sophus::SE3d Tij = SamplePose();
                pose = Twi * Tij; // Twj

                // 当前帧与上一帧地图点匹配
                reprojection_num = map_manager.match(frame_ptr, pose, reprojection_pair);
            }
            // 添加地图点观测
            for (auto pair : reprojection_pair)
            {
                const std::shared_ptr<MapPoint> &landmark3d_ptr = pair.first;
                const Eigen::Vector2i &index = pair.second;
                landmark3d_ptr->addObservation(frame_ptr, index);
            }
            // 更新当前的位姿
            frame_ptr->setPose(pose);
            std::vector<Eigen::Vector2i> unmask_index = frame_ptr->getUnmaskFeatureIndex();
            frame_ptr->ExtractFeaturePoints();
            Eigen::MatrixXd lambda_matrix = SampleLambda(Frame::feature_row, Frame::feature_col);
            // 新增加的地图点
            for (auto index : unmask_index)
            {
                int x = index.x();
                int y = index.y();
                double lambda = 1.0 / lambda_matrix(y, x);
                std::shared_ptr<MapPoint> landmark3d_ptr = std::make_shared<MapPoint>(lambda, frame_ptr, Eigen::Vector2i(x, y));
                map_manager.addMapPoint(frame_ptr->id(), x, y, landmark3d_ptr);
            }
            // frame_ptr->feature_show();
            last_frame_ptr = frame_ptr;
            frame_container.push_back(std::move(frame_ptr));
        }
    }
    std::cout << "map_manager size: " << map_manager.size() << std::endl;
    // Set up the problem
    g2o::SparseOptimizer optimizer;
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<Eigen::Dynamic, Eigen::Dynamic>> BlockSolverType;
    typedef g2o::LinearSolverCholmod<BlockSolverType::PoseMatrixType> LinearSolverType;
    auto linearSolver = new LinearSolverType();
    auto solver = new BlockSolverType(linearSolver);
    g2o::OptimizationAlgorithmLevenberg *algorithm = new g2o::OptimizationAlgorithmLevenberg(solver);
    optimizer.setAlgorithm(algorithm);
    optimizer.setVerbose(true);

    TrajectoryType true_trajectory;
    TrajectoryType init_trajectory;

    // add frame pose
    for (auto &frame_ptr : frame_container)
    {
        true_trajectory.push_back(frame_ptr->getPose()); // save the true pose
        iVertexSE3Expmap *v = new iVertexSE3Expmap();
        if (frame_ptr->id() == 0)
        {
            v->setFixed(true); // 第一帧固定且没有测量误差
        }
        else
        {
            // pose initial error , angle error = 3 degree, translation error
            frame_ptr->setPose(SamplePoseNoise() * frame_ptr->rotation(), frame_ptr->translation() + SampleTranslationNoise());
        }
        v->setId(frame_ptr->id());
        v->setEstimate(g2o::SE3Quat(frame_ptr->rotation(), frame_ptr->translation()));
        init_trajectory.push_back(frame_ptr->getPose()); // save the initial pose
        optimizer.addVertex(v);
    }

    int landmark_vertex_id = frame_container.size();
    std::vector<std::shared_ptr<MapPoint>> optimize_landmark3d;
    optimize_landmark3d.reserve(map_manager.size());
    LandmarkType true_landmarks;
    LandmarkType init_landmarks;
    for (Map::Iterator landmark3d_iter = map_manager.begin(); landmark3d_iter != map_manager.end(); ++landmark3d_iter)
    {
        const std::shared_ptr<MapPoint> &landmark3d_ptr = landmark3d_iter->second;
        if (landmark3d_ptr->obsSize() < 4)
        {
            continue;
        }
        true_landmarks.push_back(landmark3d_ptr->Get3dworld()); // save the true landmark
        const std::shared_ptr<Frame> &firstObsFrame = landmark3d_ptr->obsBegin()->frame.lock();
        const Eigen::Vector2i &firstObsIndex = landmark3d_ptr->obsBegin()->xy;
        // add lambda vertex
        VertexInverseDepth *vd = new VertexInverseDepth();
        vd->setId(landmark_vertex_id++);
        landmark3d_ptr->setLambda(SampleLambdaNoise(landmark3d_ptr->lambda())); // lambda initial error = 10cm
        vd->setEstimate(landmark3d_ptr->lambda());
        init_landmarks.push_back(landmark3d_ptr->Get3dworld()); // save the initial landmark
        // if (landmark3d_ptr->obsSize() < 10)
        // {
        //     vd->setMarginalized(true);
        // }
        optimizer.addVertex(vd);
        optimize_landmark3d.push_back(landmark3d_ptr);

        for (MapPoint::ObserverIterator it = landmark3d_ptr->obsBegin(); it != landmark3d_ptr->obsEnd(); ++it)
        {
            const std::shared_ptr<Frame> &frame = it->frame.lock();
            if (frame == firstObsFrame)
            {
                continue;
            }
            EdgeReprojection *e = new EdgeReprojection();
            e->resize(3);
            e->setVertex(0, dynamic_cast<iVertexSE3Expmap *>(optimizer.vertex(firstObsFrame->id())));
            e->setVertex(1, dynamic_cast<jVertexSE3Expmap *>(optimizer.vertex(frame->id())));
            e->setVertex(2, vd);
            // 观测
            Eigen::Vector2d obs1 = firstObsFrame->feature_at(firstObsIndex.x(), firstObsIndex.y());
            obs1 = Frame::camera.normalize2d(obs1);
            const Eigen::Vector2i &obs_index = it->xy;
            Eigen::Vector2d obs2 = frame->feature_at(obs_index.x(), obs_index.y());
            // add noise
            Eigen::Vector2d noise = SampleNoise() * 2;
            obs2 += noise;
            obs2 = Frame::camera.normalize2d(obs2);
            e->setMeasurement(Eigen::Vector4d(obs1(0), obs1(1), obs2(0), obs2(1)));
            e->setInformation(Eigen::Matrix2d::Identity());
            // add robust kernel
            g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber();
            e->setRobustKernel(rk);

            optimizer.addEdge(e);
        }
    }
    std::cout << "optimize landmark size: " << optimize_landmark3d.size() << std::endl;
    std::cout << "optimize frame pose size: " << frame_container.size() << std::endl;
    optimizer.initializeOptimization();
    optimizer.optimize(50);

    TrajectoryType opti_trajectory;
    for (auto &frame_ptr : frame_container)
    {
        iVertexSE3Expmap *v = dynamic_cast<iVertexSE3Expmap *>(optimizer.vertex(frame_ptr->id()));
        frame_ptr->setPose(v->estimate().rotation().matrix(), v->estimate().translation());
        opti_trajectory.emplace_back(frame_ptr->getPose());
    }

    LandmarkType opti_landmarks;
    for (size_t i = 0; i < optimize_landmark3d.size(); ++i)
    {
        const VertexInverseDepth *vd = dynamic_cast<VertexInverseDepth *>(optimizer.vertex(i + frame_container.size()));
        double lambda = vd->estimate();
        optimize_landmark3d[i]->setLambda(lambda);
        opti_landmarks.push_back(optimize_landmark3d[i]->Get3dworld());
    }

    TrajectoryMSE(opti_trajectory, true_trajectory);
    TrajectoryMSE(init_trajectory, true_trajectory);
    LandmarkMSE(init_landmarks, true_landmarks);
    LandmarkMSE(opti_landmarks, true_landmarks);

    DrawTrajectoryAndLandmark(init_trajectory, true_trajectory, opti_trajectory, init_landmarks, true_landmarks, opti_landmarks);

    return 0;
}
