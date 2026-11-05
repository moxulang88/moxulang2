/**
 * @file main.cpp
 * @brief RoboMaster 装甲板传统视觉粗识别
 * 
 * 流程：原图 → 二值化 → 灯条检测 → 灯条配对 → 选最优装甲板 → ROI提取
 * 
 * 输出：每张图生成5组结果图（原图/二值化/候选/配对/ROI）
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <algorithm>

using namespace cv;
using namespace std;

// ==================== 数据结构 ====================

// 灯条
struct LightBar {
    RotatedRect rect;
    Point2f center;
    float angle;
    float length;
    float width;
};

// 装甲板候选
struct ArmorCandidate {
    RotatedRect rect;
    Point2f center;
    float distToCenter;
};

// ==================== 工具函数 ====================

// 矫正灯条角度，让长边朝上
void adjustLightBarAngle(RotatedRect& rect) {
    if (rect.size.width > rect.size.height) {
        rect.size = Size2f(rect.size.height, rect.size.width);
        rect.angle += 90;
    }
    while (rect.angle < -90) rect.angle += 180;
    while (rect.angle > 90) rect.angle -= 180;
}

// ==================== 主函数 ====================

int main() {
    // ========== 可调参数 ==========
    const int BRIGHT_THRESHOLD = 180;     // 亮度阈值
    const float LIGHT_MIN_AREA = 5.0;     // 灯条最小面积
    const float LIGHT_MAX_AREA = 5000.0;  // 灯条最大面积
    const float LIGHT_MAX_RATIO = 0.6;     // 灯条宽/高 上限
    const float MAX_ANGLE_DIFF = 30.0;    // 两灯条最大角度差
    const float MAX_HEIGHT_RATIO = 1.0;    // 高度差比例
    const float MAX_Y_DIFF_RATIO = 1.0;   // y差比例
    const float MIN_X_DIFF_RATIO = 0.3;   // x差比例下限
    const float MAX_DIST_RATIO = 8.0;     // 距离比例上限
    // ==============================

    for (int imgIdx = 1; imgIdx <= 3; imgIdx++) {
        string imgPath = "images/test" + to_string(imgIdx) + ".jpg";
        Mat src = imread(imgPath);
        if (src.empty()) {
            cout << "跳过: " << imgPath << endl;
            continue;
        }

        cout << "=== 处理 test" << imgIdx << ".jpg ===" << endl;
        cout << "图片尺寸: " << src.cols << "x" << src.rows << endl;

        // 第1步：保存原图
        imwrite("results/01_original_" + to_string(imgIdx) + ".jpg", src);

        // 第2步：灰度二值化
        Mat gray;
        cvtColor(src, gray, COLOR_BGR2GRAY);
        Mat binary;
        threshold(gray, binary, BRIGHT_THRESHOLD, 255, THRESH_BINARY);
        morphologyEx(binary, binary, MORPH_OPEN,
                     getStructuringElement(MORPH_RECT, Size(2, 2)));
        imwrite("results/02_binary_" + to_string(imgIdx) + ".jpg", binary);

        // 第3步：找轮廓，筛选灯条
        vector<vector<Point>> contours;
        findContours(binary.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        cout << "找到轮廓数: " << contours.size() << endl;

        vector<LightBar> lightBars;
        Mat candidateImg = src.clone();

        for (size_t i = 0; i < contours.size(); i++) {
            float area = contourArea(contours[i]);
            if (area < LIGHT_MIN_AREA || area > LIGHT_MAX_AREA) continue;

            // fitEllipse 拟合灯条
            RotatedRect lightRect;
            if (contours[i].size() >= 5)
                lightRect = fitEllipse(contours[i]);
            else
                lightRect = minAreaRect(contours[i]);

            adjustLightBarAngle(lightRect);

            // 长宽比筛选
            float ratio = lightRect.size.width / lightRect.size.height;
            if (ratio > LIGHT_MAX_RATIO) continue;

            // 凸度筛选
            vector<Point> hull;
            convexHull(contours[i], hull);
            float solidity = area / contourArea(hull);
            if (solidity < 0.3) continue;

            LightBar bar;
            bar.rect = lightRect;
            bar.center = lightRect.center;
            bar.angle = lightRect.angle;
            bar.length = lightRect.size.height;
            bar.width = lightRect.size.width;
            lightBars.push_back(bar);

            // 画绿色框
            Point2f pts[4];
            lightRect.points(pts);
            for (int k = 0; k < 4; k++)
                line(candidateImg, pts[k], pts[(k+1)%4], Scalar(0,255,0), 2);
        }
        imwrite("results/03_candidates_" + to_string(imgIdx) + ".jpg", candidateImg);
        cout << "筛选出灯条数: " << lightBars.size() << endl;

        // 按x排序
        sort(lightBars.begin(), lightBars.end(),
             [](const LightBar& a, const LightBar& b) { return a.center.x < b.center.x; });

        // 第4步：灯条配对
        vector<ArmorCandidate> candidates;

        for (size_t i = 0; i < lightBars.size(); i++) {
            for (size_t j = i+1; j < lightBars.size(); j++) {
                LightBar& left = lightBars[i];
                LightBar& right = lightBars[j];

                // 角度平行
                float angleDiff = abs(left.angle - right.angle);
                if (angleDiff > MAX_ANGLE_DIFF) continue;

                // 高度相近
                float meanLen = (left.length + right.length) / 2;
                if (abs(left.length - right.length) > meanLen * MAX_HEIGHT_RATIO) continue;

                // y坐标接近
                float yDiff = abs(left.center.y - right.center.y);
                if (yDiff > meanLen * MAX_Y_DIFF_RATIO) continue;

                // x方向有距离
                float xDiff = abs(left.center.x - right.center.x);
                if (xDiff < meanLen * MIN_X_DIFF_RATIO) continue;

                // 距离不要太大
                float dist = norm(left.center - right.center);
                if (dist > meanLen * MAX_DIST_RATIO) continue;

                Point2f armorCenter = (left.center + right.center) / 2;
                RotatedRect armorRect(armorCenter,
                                      Size2f(dist * 1.1, meanLen * 1.1),
                                      left.angle);

                Point2f imgCenter(src.cols/2.0f, src.rows/2.0f);
                float d2c = norm(armorCenter - imgCenter);

                ArmorCandidate ac;
                ac.rect = armorRect;
                ac.center = armorCenter;
                ac.distToCenter = d2c;
                candidates.push_back(ac);
            }
        }

        // 第5步：选最靠近画面中心的装甲板
        Mat pairedImg = src.clone();
        vector<RotatedRect> armors;

        if (!candidates.empty()) {
            sort(candidates.begin(), candidates.end(),
                 [](const ArmorCandidate& a, const ArmorCandidate& b) {
                     return a.distToCenter < b.distToCenter;
                 });
            armors.push_back(candidates[0].rect);

            // 画红色框
            Point2f pts[4];
            candidates[0].rect.points(pts);
            for (int k = 0; k < 4; k++)
                line(pairedImg, pts[k], pts[(k+1)%4], Scalar(0,0,255), 3);
            circle(pairedImg, candidates[0].center, 5, Scalar(255,0,0), -1);
        }
        imwrite("results/04_paired_" + to_string(imgIdx) + ".jpg", pairedImg);
        cout << "配对成功装甲板数: " << armors.size() << endl;

        // 第6步：提取ROI
        for (size_t i = 0; i < armors.size(); i++) {
            Mat M = getRotationMatrix2D(armors[i].center, armors[i].angle, 1.0);
            Mat rotated;
            warpAffine(src, rotated, M, src.size());

            vector<Point2f> pts(4);
            armors[i].points(pts);
            Rect roiRect = boundingRect(pts);
            roiRect &= Rect(0, 0, src.cols, src.rows);
            imwrite("results/05_roi_" + to_string(imgIdx) + "_" + to_string(i) + ".jpg",
                    rotated(roiRect));
        }
        cout << "---" << endl;
    }

    cout << "全部完成！结果在 results/ 文件夹" << endl;
    return 0;
}
