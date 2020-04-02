#include "color_quantization.h"

using namespace std;

auto kmean(cv::InputArray picture, int colors) -> cv::Mat
{
    constexpr int attempts = 25;
    constexpr double epsilon = 0.00001;

    cv::Mat colVec = picture.getMat().reshape(1, picture.rows() * picture.cols() / 3); // change to a Nx3 column vector
    colVec.convertTo(colVec, CV_32F);
    cv::Mat bestLabels, centers;
    cv::TermCriteria criteria{ cv::TermCriteria::EPS + cv::TermCriteria::COUNT, attempts, epsilon };
    double compactness = kmeans(colVec, colors, bestLabels, criteria, attempts, cv::KMEANS_PP_CENTERS, centers);
    return centers;
}

auto histogram(cv::InputArray picture, int bins) -> std::vector<cv::Mat>
{
    using namespace std;
    using namespace cv;

    vector<Mat> bgr_planes;
    split(picture.getMat(), bgr_planes);

    // Set the ranges for B,G,R
    float range[] = { 0, 256 };
    const float* histRange = { range };

    bool uniform = true; bool accumulate = false;

    Mat b_hist, g_hist, r_hist;

    calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &bins, &histRange, uniform, accumulate);
    calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &bins, &histRange, uniform, accumulate);
    calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &bins, &histRange, uniform, accumulate);

    return { b_hist, g_hist, r_hist };
}

// https://github.com/aishack/dominant-colors

struct t_color_node {
    cv::Mat       mean;       // The mean of this node
    cv::Mat       cov;
    uchar         classid;    // The class ID

    t_color_node* left;
    t_color_node* right;
};

std::vector<t_color_node*> get_leaves(t_color_node* root) {
    std::vector<t_color_node*> ret;
    std::queue<t_color_node*> queue;
    queue.push(root);

    while (queue.size() > 0) {
        t_color_node* current = queue.front();
        queue.pop();

        if (current->left && current->right) {
            queue.push(current->left);
            queue.push(current->right);
            continue;
        }

        ret.push_back(current);
    }

    return ret;
}

std::vector<cv::Vec3b> get_dominant_colors(t_color_node* root) {
    std::vector<t_color_node*> leaves = get_leaves(root);
    std::vector<cv::Vec3b> ret;

    for (int i = 0; i < leaves.size(); i++) {
        cv::Mat mean = leaves[i]->mean;
        ret.push_back(cv::Vec3b(mean.at<double>(0) * 255.0f,
            mean.at<double>(1) * 255.0f,
            mean.at<double>(2) * 255.0f));
    }

    return ret;
}

int get_next_classid(t_color_node* root) {
    int maxid = 0;
    std::queue<t_color_node*> queue;
    queue.push(root);

    while (queue.size() > 0) {
        t_color_node* current = queue.front();
        queue.pop();

        if (current->classid > maxid)
            maxid = current->classid;

        if (current->left != NULL)
            queue.push(current->left);

        if (current->right)
            queue.push(current->right);
    }

    return maxid + 1;
}

void get_class_mean_cov(cv::Mat img, cv::Mat classes, t_color_node* node) {
    const int width = img.cols;
    const int height = img.rows;
    const uchar classid = node->classid;

    cv::Mat mean = cv::Mat(3, 1, CV_64FC1, cv::Scalar(0));
    cv::Mat cov = cv::Mat(3, 3, CV_64FC1, cv::Scalar(0));

    // We start out with the average color
    double pixcount = 0;
    for (int y = 0; y < height; y++) {
        cv::Vec3b* ptr = img.ptr<cv::Vec3b>(y);
        uchar* ptrClass = classes.ptr<uchar>(y);
        for (int x = 0; x < width; x++) {
            if (ptrClass[x] != classid)
                continue;

            cv::Vec3b color = ptr[x];
            cv::Mat scaled = cv::Mat(3, 1, CV_64FC1, cv::Scalar(0));
            scaled.at<double>(0) = color[0] / 255.0f;
            scaled.at<double>(1) = color[1] / 255.0f;
            scaled.at<double>(2) = color[2] / 255.0f;

            mean += scaled;
            cov = cov + (scaled * scaled.t());

            pixcount++;
        }
    }

    cov = cov - (mean * mean.t()) / pixcount;
    mean = mean / pixcount;

    // The node mean and covariance
    node->mean = mean.clone();
    node->cov = cov.clone();

    return;
}

void partition_class(cv::Mat img, cv::Mat classes, uchar nextid, t_color_node* node) {
    const int width = img.cols;
    const int height = img.rows;
    const int classid = node->classid;

    const uchar newidleft = nextid;
    const uchar newidright = nextid + 1;

    cv::Mat mean = node->mean;
    cv::Mat cov = node->cov;
    cv::Mat eigenvalues, eigenvectors;
    cv::eigen(cov, eigenvalues, eigenvectors);

    cv::Mat eig = eigenvectors.row(0);
    cv::Mat comparison_value = eig * mean;

    node->left = new t_color_node();
    node->right = new t_color_node();

    node->left->classid = newidleft;
    node->right->classid = newidright;

    // We start out with the average color
    for (int y = 0; y < height; y++) {
        cv::Vec3b* ptr = img.ptr<cv::Vec3b>(y);
        uchar* ptrClass = classes.ptr<uchar>(y);
        for (int x = 0; x < width; x++) {
            if (ptrClass[x] != classid)
                continue;

            cv::Vec3b color = ptr[x];
            cv::Mat scaled = cv::Mat(3, 1,
                CV_64FC1,
                cv::Scalar(0));

            scaled.at<double>(0) = color[0] / 255.0f;
            scaled.at<double>(1) = color[1] / 255.0f;
            scaled.at<double>(2) = color[2] / 255.0f;

            cv::Mat this_value = eig * scaled;

            if (this_value.at<double>(0, 0) <= comparison_value.at<double>(0, 0)) {
                ptrClass[x] = newidleft;
            }
            else {
                ptrClass[x] = newidright;
            }
        }
    }
    return;
}

t_color_node* get_max_eigenvalue_node(t_color_node* current) {
    double max_eigen = -1;
    cv::Mat eigenvalues, eigenvectors;

    std::queue<t_color_node*> queue;
    queue.push(current);

    t_color_node* ret = current;
    if (!current->left && !current->right)
        return current;

    while (queue.size() > 0) {
        t_color_node* node = queue.front();
        queue.pop();

        if (node->left && node->right) {
            queue.push(node->left);
            queue.push(node->right);
            continue;
        }

        cv::eigen(node->cov, eigenvalues, eigenvectors);
        double val = eigenvalues.at<double>(0);
        if (val > max_eigen) {
            max_eigen = val;
            ret = node;
        }
    }

    return ret;
}

std::vector<cv::Vec3b> dominant_colors(cv::Mat img, int count) {
    const int width = img.cols;
    const int height = img.rows;

    cv::Mat classes = cv::Mat(height, width, CV_8UC1, cv::Scalar(1));
    t_color_node* root = new t_color_node();

    root->classid = 1;
    root->left = NULL;
    root->right = NULL;

    t_color_node* next = root;
    get_class_mean_cov(img, classes, root);
    for (int i = 0; i < count - 1; i++) {
        next = get_max_eigenvalue_node(root);
        partition_class(img, classes, get_next_classid(root), next);
        get_class_mean_cov(img, classes, next->left);
        get_class_mean_cov(img, classes, next->right);
    }

    return get_dominant_colors(root);
}
