#include "nlm.h"
#if __ARM_NEON
#include <arm_neon.h>
#else
//__ARM_NEON
#include "NEON_2_SSE.h"
#endif

#include <chrono>
namespace nlm {


inline float MSENaiveSquare(const cv::Mat &m1, const cv::Mat &m2) {
    assert(m1.cols == m2.cols);
    assert(m1.rows == m2.rows);
    float sum = 0;
    for(int i = 0; i < m1.rows; i++) {
        for(int j = 0; j < m1.cols; j++) {
            auto p1 = static_cast<int>(m1.at<uchar>(i,j));
            auto p2 = static_cast<int>(m2.at<uchar>(i,j));
            
            sum += (p1 - p2) * (p1 - p2);
        }
    }
    sum = sum / (m1.rows * m1.cols);
    return sum;
}


void NLMean::CalculateNLMNaive(cv::Mat &restoredImg) {
    restoredImg.create(srcImg_.rows, srcImg_.cols, CV_8UC1);
    for (int i = borderSize_; i < srcImg_.rows + borderSize_; i++) {
        for (int j = borderSize_; j < srcImg_.cols + borderSize_; j++) {
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
            
            restoredImg.at<uchar>(i - borderSize_, j - borderSize_) = static_cast<uchar>(sumPixel / sumWeighet);
        }
    }
}

inline void imageIntegralDiff(const cv::Mat &padSrc, cv::Mat &dst, int halfSearchSize, int offsetX, int offsetY) {
    cv::Mat padSrcTmp;
    padSrc.convertTo(padSrcTmp, CV_32FC1);
    cv::Mat distanceMat = padSrcTmp(cv::Range(halfSearchSize, padSrc.rows - halfSearchSize), cv::Range(halfSearchSize, padSrc.cols - halfSearchSize)) - padSrcTmp(cv::Range(halfSearchSize + offsetX, padSrc.rows - halfSearchSize + offsetX), cv::Range(halfSearchSize + offsetY, padSrc.cols - halfSearchSize + offsetY));
    distanceMat.convertTo(distanceMat, CV_32FC1);

    distanceMat = distanceMat.mul(distanceMat);

    cv::integral(distanceMat, dst, CV_32FC1);
    
}

void NLMean::CalcuateNLMFast(cv::Mat &restoreImg) {
    restoreImg.create(srcImg_.rows, srcImg_.cols, CV_8UC1);
     
    cv::Mat pixelAverage(srcImg_.rows, srcImg_.cols, CV_32FC1, 0.0);
    cv::Mat sumWeight(srcImg_.rows, srcImg_.cols, CV_32FC1, 0.0);
    
    cv::Mat integralMat(padSrcImg_.rows - 2 * halfSearchKernelSize_, padSrcImg_.cols - 2 * halfSearchKernelSize_, CV_32FC1);
    
    int templateKernelSq = (2 * halfTemplateKernelSize_ + 1) * (2 * halfTemplateKernelSize_ + 1);
    
    for (int srow = -halfSearchKernelSize_; srow <= halfSearchKernelSize_; srow++) {
        for (int scol = -halfSearchKernelSize_; scol <= halfSearchKernelSize_; scol++) {
            imageIntegralDiff(padSrcImg_, integralMat, halfSearchKernelSize_, srow, scol);
            integralMat /= (-h2_ * templateKernelSq);
            
            for (int i = 0; i < srcImg_.rows; i++) {
                for (int j = 0; j < srcImg_.cols; j++) {
                    double diffSq = integralMat.at<float>(i + halfTemplateKernelSize_ * 2 + 1, j + halfTemplateKernelSize_ * 2 + 1) + integralMat.at<float>(i, j) - integralMat.at<float>(i + halfTemplateKernelSize_ * 2 + 1,  j) - integralMat.at<float>(i, j + halfTemplateKernelSize_ * 2 + 1);
                    sumWeight.at<float>(i, j) += exp(diffSq);
                    pixelAverage.at<float>(i, j) += exp(diffSq) * padSrcImg_.at<uchar>(i + halfSearchKernelSize_ + srow +  halfTemplateKernelSize_, j + halfSearchKernelSize_ + scol + halfTemplateKernelSize_);
                }
            }
        }
    }
    
    
    pixelAverage = pixelAverage / sumWeight;
    pixelAverage.convertTo(restoreImg, CV_8UC1);

}

void integralNeon(const cv::Mat& src, cv::Mat& dst) {
    assert(src.type() == CV_8UC1);
    assert(src.channels() == 1);
    dst.create(src.rows, src.cols, CV_32SC1);
    int dstStride = dst.cols;
    
    // for accumulate
    uint32x4_t vz = vdupq_n_u32(0);
    uint32x4_t vsum_pre = vz;
    
    // for row 0
    uint32_t *sumPtr = dst.ptr<uint32_t>(0);
    const unsigned char *srcRowPtr = src.ptr<uchar>(0);
    
    int jRound = src.cols / 8 * 8;
    int iRound = src.rows / 8 * 8;
    
    
    for(int j = 0; j < jRound; j+=8) {
        uint8x8_t r1_j8x8_shr0 = vld1_u8(srcRowPtr + j);
        uint8x8_t r1_j8x8_shr1 = vreinterpret_u8_u64(vshl_n_u64(vreinterpret_u64_u8(r1_j8x8_shr0), 8));
        uint8x8_t r1_j8x8_shr2 = vreinterpret_u8_u64(vshl_n_u64(vreinterpret_u64_u8(r1_j8x8_shr0), 16));
        uint8x8_t r1_j8x8_shr3 = vreinterpret_u8_u64(vshl_n_u64(vreinterpret_u64_u8(r1_j8x8_shr0), 24));
        
        uint16x8_t sum_j01 = vaddl_u8(r1_j8x8_shr0, r1_j8x8_shr1);
        uint16x8_t sum_j23 = vaddl_u8(r1_j8x8_shr2, r1_j8x8_shr3);
        
        uint16x8_t sum_cur = vaddq_u16(sum_j01, sum_j23);
        
        
        uint16x4_t sum_cur_l = vget_low_u16(sum_cur);
        uint16x4_t sum_cur_h = vadd_u16(vget_low_u16(sum_cur) , vget_high_u16(sum_cur));
        
        uint32x4_t sum_now_l = vaddw_u16(vsum_pre, sum_cur_l);
        uint32x4_t sum_now_h = vaddw_u16(vsum_pre, sum_cur_h);
        
        vst1q_u32(sumPtr + j , sum_now_l);
        vst1q_u32(sumPtr + j + 4, sum_now_h);
        
        vsum_pre = vaddw_u16(vsum_pre, vdup_lane_u16(sum_cur_h, 3));
    }
    uint32_t pre = vgetq_lane_u32(vsum_pre, 3);
    for(int j = jRound; j < src.cols; j++) {
        sumPtr[j] = pre + srcRowPtr[j];
        pre = sumPtr[j];
    }
    
    // others
    for(int i = 1; i < src.rows; i++) {
        sumPtr = dst.ptr<uint32_t>(i);
        srcRowPtr = src.ptr<uchar>(i);
        uint32_t* sumPrePtr = dst.ptr<uint32_t>(i - 1);
        vsum_pre = vz;
        for(int j = 0; j < jRound; j+= 8) {
            uint32x4_t vsuml = vld1q_u32(sumPrePtr + j);
            uint32x4_t vsumh = vld1q_u32(sumPrePtr + j + 4);
            
            
            uint8x8_t r1_j8x8_shr0 = vld1_u8(srcRowPtr + j);
            uint8x8_t r1_j8x8_shr1 = vreinterpret_u8_u64(vshl_n_u64(vreinterpret_u64_u8(r1_j8x8_shr0), 8));
            uint8x8_t r1_j8x8_shr2 = vreinterpret_u8_u64(vshl_n_u64(vreinterpret_u64_u8(r1_j8x8_shr0), 16));
            uint8x8_t r1_j8x8_shr3 = vreinterpret_u8_u64(vshl_n_u64(vreinterpret_u64_u8(r1_j8x8_shr0), 24));
            
            
            vsuml = vaddq_u32(vsuml, vsum_pre);
            vsumh = vaddq_u32(vsumh, vsum_pre);

            
            uint16x8_t sum_j01 = vaddl_u8(r1_j8x8_shr0, r1_j8x8_shr1);
            uint16x8_t sum_j23 = vaddl_u8(r1_j8x8_shr2, r1_j8x8_shr3);
            
            uint16x8_t sum_cur = vaddq_u16(sum_j01, sum_j23);
            
            
            uint16x4_t sum_cur_l = vget_low_u16(sum_cur);
            uint16x4_t sum_cur_h = vadd_u16(vget_low_u16(sum_cur) , vget_high_u16(sum_cur));
            
            vsuml = vaddw_u16(vsuml, sum_cur_l);
            vsumh = vaddw_u16(vsumh, sum_cur_h);
            
            vst1q_u32(sumPtr + j , vsuml);
            vst1q_u32(sumPtr + j + 4, vsumh);
            
            vsum_pre = vaddw_u16(vsum_pre, vdup_lane_u16(sum_cur_h, 3));
        }
        uint32_t pre = vgetq_lane_u32(vsum_pre, 3);
        for(int j = jRound; j < src.cols; j++) {
            sumPtr[j] = srcRowPtr[j] + sumPrePtr[j] + pre;
            pre += srcRowPtr[j];
        }
        
    }
}

inline void imageIntegralDiffNeon(const cv::Mat &padSrc, cv::Mat &dst, int halfSearchSize, int offsetX, int offsetY) {
    cv::Mat padSrcTmp;
    padSrc.convertTo(padSrcTmp, CV_32FC1);

    cv::Mat distanceMat = padSrcTmp(cv::Range(halfSearchSize, padSrc.rows - halfSearchSize), cv::Range(halfSearchSize, padSrc.cols - halfSearchSize)) - padSrcTmp(cv::Range(halfSearchSize + offsetX, padSrc.rows - halfSearchSize + offsetX), cv::Range(halfSearchSize + offsetY, padSrc.cols - halfSearchSize + offsetY));
    distanceMat.convertTo(distanceMat, CV_32FC1);
    
    distanceMat = distanceMat.mul(distanceMat);
    
    double minval;
    double maxval;
    cv::minMaxLoc(distanceMat, &minval, &maxval);
    const float scale  = 255.0 / (maxval - minval);
    const float invscale = 1.0 / scale;
    distanceMat = (distanceMat - minval) * scale;

    distanceMat.convertTo(distanceMat, CV_8UC1);
    integralNeon(distanceMat, dst);

    dst = dst * invscale + minval;
}



#define c_exp_hi 88.3762626647949f
#define c_exp_lo -88.3762626647949f

#define c_cephes_LOG2EF 1.44269504088896341
#define c_cephes_exp_C1 0.693359375
#define c_cephes_exp_C2 -2.12194440e-4

#define c_cephes_exp_p0 1.9875691500E-4
#define c_cephes_exp_p1 1.3981999507E-3
#define c_cephes_exp_p2 8.3334519073E-3
#define c_cephes_exp_p3 4.1665795894E-2
#define c_cephes_exp_p4 1.6666665459E-1
#define c_cephes_exp_p5 5.0000001201E-1

/* exp() computed for 4 float at once */
float32x4_t exp_ps(float32x4_t x) {
    float32x4_t tmp, fx;

    float32x4_t one = vdupq_n_f32(1);
    x = vminq_f32(x, vdupq_n_f32(c_exp_hi));
    x = vmaxq_f32(x, vdupq_n_f32(c_exp_lo));

    /* express exp(x) as exp(g + n*log(2)) */
    fx = vmlaq_f32(vdupq_n_f32(0.5f), x, vdupq_n_f32(c_cephes_LOG2EF));

    /* perform a floorf */
    tmp = vcvtq_f32_s32(vcvtq_s32_f32(fx));

    /* if greater, substract 1 */
    uint32x4_t mask = vcgtq_f32(tmp, fx);
    mask = vandq_u32(mask, vreinterpretq_u32_f32(one));


    fx = vsubq_f32(tmp, vreinterpretq_f32_u32(mask));

    tmp = vmulq_f32(fx, vdupq_n_f32(c_cephes_exp_C1));
    float32x4_t z = vmulq_f32(fx, vdupq_n_f32(c_cephes_exp_C2));
    x = vsubq_f32(x, tmp);
    x = vsubq_f32(x, z);

    static const float cephes_exp_p[6] = { c_cephes_exp_p0, c_cephes_exp_p1, c_cephes_exp_p2, c_cephes_exp_p3, c_cephes_exp_p4, c_cephes_exp_p5 };
    float32x4_t y = vld1q_dup_f32(cephes_exp_p+0);
    float32x4_t c1 = vld1q_dup_f32(cephes_exp_p+1);
    float32x4_t c2 = vld1q_dup_f32(cephes_exp_p+2);
    float32x4_t c3 = vld1q_dup_f32(cephes_exp_p+3);
    float32x4_t c4 = vld1q_dup_f32(cephes_exp_p+4);
    float32x4_t c5 = vld1q_dup_f32(cephes_exp_p+5);

    y = vmulq_f32(y, x);
    z = vmulq_f32(x,x);
    y = vaddq_f32(y, c1);
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, c2);
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, c3);
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, c4);
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, c5);

    y = vmulq_f32(y, z);
    y = vaddq_f32(y, x);
    y = vaddq_f32(y, one);

    /* build 2^n */
    int32x4_t mm;
    mm = vcvtq_s32_f32(fx);
    mm = vaddq_s32(mm, vdupq_n_s32(0x7f));
    mm = vshlq_n_s32(mm, 23);
    float32x4_t pow2n = vreinterpretq_f32_s32(mm);

    y = vmulq_f32(y, pow2n);
    return y;
}

void NLMean::CalcuateNLMFastNeon(cv::Mat &restoreImg) {
    restoreImg.create(srcImg_.rows, srcImg_.cols, CV_8UC1);
     
    cv::Mat pixelAverage(srcImg_.rows, srcImg_.cols, CV_32FC1, 0.0);
    cv::Mat sumWeight(srcImg_.rows, srcImg_.cols, CV_32FC1, 0.0);
    
    cv::Mat integralMat(padSrcImg_.rows - 2 * halfSearchKernelSize_, padSrcImg_.cols - 2 * halfSearchKernelSize_, CV_32FC1);
    
    int templateKernelSq = (2 * halfTemplateKernelSize_ + 1) * (2 * halfTemplateKernelSize_ + 1);
    float passed = 0.0;
    float passed2 = 0.0;
    for (int srow = -halfSearchKernelSize_; srow <= halfSearchKernelSize_; srow++) {
        for (int scol = -halfSearchKernelSize_; scol <= halfSearchKernelSize_; scol++) {
            imageIntegralDiffNeon(padSrcImg_, integralMat, halfSearchKernelSize_, srow, scol);
            
            cv::Mat diffSqMat(srcImg_.rows, srcImg_.cols, CV_32SC1);
            integralMat /=   -h2_ * templateKernelSq;
            auto start = std::chrono::steady_clock::now();
            const int fullTemplateSize = halfTemplateKernelSize_ * 2 + 1;
            for (int i = 0; i < srcImg_.rows; i++) {
                float* diffSqMatPtr = diffSqMat.ptr<float>(i);
                int32_t* integralMatPtr = integralMat.ptr<int32_t>(i);
                int32_t* integralMatAddTemplatePtr = integralMat.ptr<int32_t>(i + fullTemplateSize);
                const int ip = i + halfSearchKernelSize_ + srow +  halfTemplateKernelSize_;
                const int tp = halfSearchKernelSize_ + scol + halfTemplateKernelSize_;
                float *sumWeightPtr = sumWeight.ptr<float>(i);
                float *pixelAveragePtr = pixelAverage.ptr<float>(i);
                uchar *padSrcPtr = padSrcImg_.ptr<uchar>(ip);
                int j = 0;
                for (; j + 4 < srcImg_.cols; j+=4) {
                    int32x4_t rightBottom4 = vld1q_s32(integralMatAddTemplatePtr + j +fullTemplateSize);
                    int32x4_t leftTop4 = vld1q_s32(integralMatPtr + j);
                    int32x4_t leftBottom4 = vld1q_s32(integralMatAddTemplatePtr + j);
                    int32x4_t rightTop4 = vld1q_s32(integralMatPtr + j + fullTemplateSize);

                    float32x4_t df4 = vcvtq_f32_s32(vsubq_s32(vaddq_s32(rightBottom4, leftTop4), vaddq_s32(leftBottom4, rightTop4)));

//                    std::cout << vgetq_lane_f32(df4, 3)<< std::endl ;
                    float32x4_t sumw4 = vld1q_f32(sumWeightPtr  + j);
                    float32x4_t dfe4 = exp_ps(df4);
//                    vst1q_f32(diffSqMatPtr + j, te4);
                    float32x4_t nsum4 = vaddq_f32(sumw4, dfe4);
                    vst1q_f32(sumWeightPtr  + j, nsum4);

                    float32x4_t pa4 = vld1q_f32(pixelAveragePtr  + j);

                    const int jp = tp + j;
                    uint8x8_t ps8 = vld1_u8(padSrcPtr + jp);
                    uint16x4_t ps4 = vget_low_u16(vmovl_u8(ps8));;


                    float32x4_t pf4 = vcvtq_f32_u32(vmovl_u16(ps4));

                    float32x4_t npa4 = vaddq_f32(pa4, vmulq_f32(dfe4, pf4)) ;

                    vst1q_f32(pixelAveragePtr  + j, npa4);
//                    std::cout << *(pixelAveragePtr + j) << std::endl;
//                    std::cout << *(pixelAveragePtr + j) << std::endl;
//                    std::cout << *(pixelAveragePtr + j) << std::endl;
//                    std::cout << *(pixelAveragePtr + j) << std::endl;

                }

                for (; j < srcImg_.cols; j++) {
                    auto sq = integralMat.at<int32_t>(i + fullTemplateSize, j + fullTemplateSize)
                     + integralMat.at<int32_t>(i, j)
                     - integralMat.at<int32_t>(i + fullTemplateSize, j)
                     - integralMat.at<int32_t>(i, j + fullTemplateSize);

                    const int jp = j + tp;
                    auto esq = exp(sq);
                    *(sumWeightPtr + j) += esq;
                    *(pixelAveragePtr + j) += esq * (*(padSrcPtr + jp));
//                    std::cout << *(pixelAveragePtr + j) << std::endl;
//                    std::cout << *(pixelAveragePtr + j) << std::endl;
//                    std::cout << *(pixelAveragePtr + j) << std::endl;
//                    std::cout << *(pixelAveragePtr + j) << std::endl;
                }
            }
            
            //for(int i = 0; i < srcImg_.rows; i++) {
            //    int j = 0;
            //    float* diffSqMatPtr = diffSqMat.ptr<float>(i);
            //    const int ip = i + halfSearchKernelSize_ + srow +  halfTemplateKernelSize_;
            //    const int tp = halfSearchKernelSize_ + scol + halfTemplateKernelSize_;
            //    float *sumWeightPtr = sumWeight.ptr<float>(i);
            //    float *pixelAveragePtr = pixelAverage.ptr<float>(i);
            //    uchar *padSrcPtr = padSrcImg_.ptr<uchar>(ip);
            //    
            //    for(; j + 4< srcImg_.cols; j+=4) {
            //        float32x4_t df4 = vld1q_f32(diffSqMatPtr + j);
            //        float32x4_t sumw4 = vld1q_f32(sumWeightPtr  + j);
            //        float32x4_t dfe4 = exp_ps(df4);
//          //          vst1q_f32(diffSqMatPtr + j, te4);
            //        float32x4_t nsum4 = vaddq_f32(sumw4, dfe4);
            //        vst1q_f32(sumWeightPtr  + j, nsum4);
            //        
            //        float32x4_t pa4 = vld1q_f32(pixelAveragePtr  + j);
            //        
            //        const int jp = tp + j;
            //        uint8x8_t ps8 = vld1_u8(padSrcPtr + jp);
            //        uint16x4_t ps4 = vget_low_u16(vmovl_u8(ps8));;
            //        
            //        
            //        float32x4_t pf4 = vcvtq_f32_u32(vmovl_u16(ps4));
            //        
            //        float32x4_t npa4 = vaddq_f32(pa4, vmulq_f32(dfe4, pf4)) ;
            //        vst1q_f32(pixelAveragePtr  + j, npa4);
            //        
            //    }
            //    for(; j < srcImg_.cols; j++) {
            //    }
            //}
            
            
            auto end = std::chrono::steady_clock::now();
            passed += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

        }
    }
    
//    std::cout << "cost in calc exp and weight: " << passed2 << "ms" <<std::endl;
    pixelAverage = pixelAverage / sumWeight;
    pixelAverage.convertTo(restoreImg, CV_8UC1);

}

}
