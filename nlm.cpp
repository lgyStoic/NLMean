#include "nlm.h"

namespace nlm {

inline float MSENaiveSquare(const cv::Mat &m1, const cv::Mat &m2) {
    assert(m1.cols == m2.cols);
    assert(m1.rows == m2.rows);
    float sum = 0;
    for(int i = 0; i < m1.cols; i++) {
        for(int j = 0; j < m1.rows; j++) {
            sum += (m1.at<uchar>(j, i) - m2.at<uchar>(j, i)) * (m1.at<uchar>(j, i) - m2.at<uchar>(j, i));
        }
    }
    sum = sum / (m1.rows * m1.cols);
    return sum;
}


void NLMean::CalculateNLMNaive(cv::Mat &restoredImg) {
    restoredImg.create(srcImg_.rows, srcImg_.cols, CV_8UC1);
    int borderSize = halfSearchKernelSize_ + halfTemplateKernelSize_;
    cv::copyMakeBorder(srcImg_, padSrcImg_, borderSize, borderSize, borderSize, borderSize, borderSize, cv::BORDER_REPLICATE);
    
    for (int i = borderSize; i < srcImg_.rows + borderSize; i++) {
        for (int j = borderSize; j < srcImg_.cols + borderSize; j++) {
            cv::Mat partA = padSrcImg_(cv::Range(i - halfTemplateKernelSize_, i + halfTemplateKernelSize_), cv::Range(j - halfTemplateKernelSize_, j + halfTemplateKernelSize_));
            float sumWeighet = 0.0;
            float sumPixel = 0.0;
            for(int srow = -halfSearchKernelSize_; srow <= halfSearchKernelSize_; srow++) {
                for(int scol = -halfSearchKernelSize_; scol <= halfSearchKernelSize_; scol++) {
                    cv::Mat partB = padSrcImg_(cv::Range(i + srow - halfTemplateKernelSize_, i + srow + halfTemplateKernelSize_), cv::Range(j + scol - halfTemplateKernelSize_, j + scol + halfTemplateKernelSize_));
                    float d2 = MSENaiveSquare(partA, partB);
                    sumWeighet += exp(-d2 / h2_);
                    sumPixel += exp(-d2 / h2_) * padSrcImg_.at<uchar>(i + srow, j + scol);
                }
            }
            restoredImg.at<uchar>(i - borderSize, j - borderSize) = static_cast<uchar>(sumPixel / sumWeighet);
        }
    }
    
}
}
