// Copyright (c) 2021, Viktor Larsson
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of the copyright holder nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef POSELIB_ROBUST_ESTIMATORS_RELATIVE_POSE_H
#define POSELIB_ROBUST_ESTIMATORS_RELATIVE_POSE_H

#include "PoseLib/camera_pose.h"
#include "PoseLib/robust/sampling.h"
#include "PoseLib/robust/utils.h"
#include "PoseLib/types.h"

namespace poselib {

class RelativePoseEstimator {
  public:
    RelativePoseEstimator(const RansacOptions &ransac_opt, const std::vector<Point2D> &points2D_1,
                          const std::vector<Point2D> &points2D_2)
        : num_data(points2D_1.size()), opt(ransac_opt), x1(points2D_1), x2(points2D_2),
          sampler(num_data, sample_sz, opt.seed, opt.progressive_sampling, opt.max_prosac_iterations) {
        x1s.resize(sample_sz);
        x2s.resize(sample_sz);
        sample.resize(sample_sz);
    }

    void generate_models(std::vector<CameraPose> *models);
    double score_model(const CameraPose &pose, size_t *inlier_count) const;
    void refine_model(CameraPose *pose) const;

    const size_t sample_sz = 5;
    const size_t num_data;

  private:
    const RansacOptions &opt;
    const std::vector<Point2D> &x1;
    const std::vector<Point2D> &x2;

    RandomSampler sampler;
    // pre-allocated vectors for sampling
    std::vector<Eigen::Vector3d> x1s, x2s;
    std::vector<size_t> sample;
};

class RelativePoseGravityEstimator {
  public:
    RelativePoseGravityEstimator(const RansacOptions &ransac_opt, const std::vector<Point2D> &points2D_1,
                                 const std::vector<Point2D> &points2D_2, const Eigen::Vector3d &gravity_1,
                                 const Eigen::Vector3d &gravity_2, const double gravity_uncertainty)
        : num_data(points2D_1.size()), opt(ransac_opt), x1(points2D_1), x2(points2D_2), g1(gravity_1), g2(gravity_2),
          gu(gravity_uncertainty),
          sampler(num_data, sample_sz, opt.seed, opt.progressive_sampling, opt.max_prosac_iterations) {
        x1s.resize(sample_sz);
        x2s.resize(sample_sz);
        sample.resize(sample_sz);
    }

    void generate_models(std::vector<CameraPose> *models);
    double score_model(const CameraPose &pose, size_t *inlier_count) const;
    void refine_model(CameraPose *pose) const;

    const size_t sample_sz = 5;
    const size_t num_data;

  private:
    const RansacOptions &opt;
    const std::vector<Point2D> &x1;
    const std::vector<Point2D> &x2;
    const Eigen::Vector3d &g1;
    const Eigen::Vector3d &g2;
    double gu;

    RandomSampler sampler;
    // pre-allocated vectors for sampling
    std::vector<Eigen::Vector3d> x1s, x2s;
    std::vector<size_t> sample;
};

// def R_from_rp(roll: torch.Tensor, pitch: torch.Tensor) -> torch.Tensor:
//     R = Rotation.from_euler("ZX", (roll.cpu(), pitch.cpu()), degrees=False).as_matrix()
//     return torch.from_numpy(R).to(roll)

// def rp_from_gravity(gravity: torch.Tensor, eps=1e-4) -> Tuple[torch.Tensor]:
//     x, z = gravity[..., 0], gravity[..., 2]
//     roll = torch.asin(-x / (torch.sqrt(1 - z**2) + eps))
//     pitch = torch.asin(z)
//     return roll, pitch

inline Eigen::Matrix3d R_from_gravity(const Eigen::Vector3d &g) {
    double roll = std::asin(-g.x() / std::sqrt(1 - g.z() * g.z()) + 0.0001);
    double pitch = std::asin(g.z());

    Eigen::Matrix3d R;
    R = Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitZ()) * Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitX());

    return R;
}

class RelativePoseHybridEstimator {
  public:
    RelativePoseHybridEstimator(const RansacOptions &ransac_opt, const std::vector<Point2D> &points2D_1,
                                const std::vector<Point2D> &points2D_2, const Eigen::Vector3d &gravity_1,
                                const Eigen::Vector3d &gravity_2, const double gravity_uncertainty)
        : num_data(points2D_1.size()), opt(ransac_opt), x1(points2D_1), x2(points2D_2), g1(gravity_1), g2(gravity_2),
          gu(gravity_uncertainty),
          sampler(num_data, sample_sz, opt.seed, opt.progressive_sampling, opt.max_prosac_iterations) {
        x1s.resize(sample_sz);
        x2s.resize(sample_sz);
        x1sg.resize(3);
        x2sg.resize(3);
        sample.resize(sample_sz);
        Rg1 = R_from_gravity(g1);
        Rg2 = R_from_gravity(g2);
    }

    void generate_models(std::vector<CameraPose> *models);
    double score_model(const CameraPose &pose, size_t *inlier_count) const;
    void refine_model(CameraPose *pose) const;

    const size_t sample_sz = 5;
    const size_t num_data;

  private:
    const RansacOptions &opt;
    const std::vector<Point2D> &x1;
    const std::vector<Point2D> &x2;
    const Eigen::Vector3d &g1;
    const Eigen::Vector3d &g2;

    Eigen::Matrix3d Rg1;
    Eigen::Matrix3d Rg2;
    double gu;

    RandomSampler sampler;
    // pre-allocated vectors for sampling
    std::vector<Eigen::Vector3d> x1s, x2s;
    std::vector<Eigen::Vector3d> x1sg, x2sg;
    std::vector<size_t> sample;
};

class RelativePoseUprightEstimator {
  public:
    RelativePoseUprightEstimator(const RansacOptions &ransac_opt, const std::vector<Point2D> &points2D_1,
                                 const std::vector<Point2D> &points2D_2)
        : num_data(points2D_1.size()), opt(ransac_opt), x1(points2D_1), x2(points2D_2),
          sampler(num_data, sample_sz, opt.seed, opt.progressive_sampling, opt.max_prosac_iterations) {
        x1s.resize(sample_sz);
        x2s.resize(sample_sz);
        sample.resize(sample_sz);
    }

    void generate_models(std::vector<CameraPose> *models);
    double score_model(const CameraPose &pose, size_t *inlier_count) const;
    void refine_model(CameraPose *pose) const;

    const size_t sample_sz = 3; // here I changed 5 into 3, is it correct to do so ?
    const size_t num_data;

  private:
    const RansacOptions &opt;
    const std::vector<Point2D> &x1;
    const std::vector<Point2D> &x2;

    RandomSampler sampler;
    // pre-allocated vectors for sampling
    std::vector<Eigen::Vector3d> x1s, x2s;
    std::vector<size_t> sample;
};

class GeneralizedRelativePoseEstimator {
  public:
    GeneralizedRelativePoseEstimator(const RansacOptions &ransac_opt,
                                     const std::vector<PairwiseMatches> &pairwise_matches,
                                     const std::vector<CameraPose> &camera1_ext,
                                     const std::vector<CameraPose> &camera2_ext)
        : opt(ransac_opt), matches(pairwise_matches), rig1_poses(camera1_ext), rig2_poses(camera2_ext) {
        rng = opt.seed;
        x1s.resize(sample_sz);
        x2s.resize(sample_sz);
        p1s.resize(sample_sz);
        p2s.resize(sample_sz);
        sample.resize(sample_sz);

        num_data = 0;
        for (const PairwiseMatches &m : matches) {
            num_data += m.x1.size();
        }
    }

    void generate_models(std::vector<CameraPose> *models);
    double score_model(const CameraPose &pose, size_t *inlier_count) const;
    void refine_model(CameraPose *pose) const;

    const size_t sample_sz = 6;
    size_t num_data;

  private:
    const RansacOptions &opt;
    const std::vector<PairwiseMatches> &matches;
    const std::vector<CameraPose> &rig1_poses;
    const std::vector<CameraPose> &rig2_poses;

    RNG_t rng;
    // pre-allocated vectors for sampling
    std::vector<Eigen::Vector3d> x1s, x2s, p1s, p2s;
    std::vector<size_t> sample;
};

class FundamentalEstimator {
  public:
    FundamentalEstimator(const RansacOptions &ransac_opt, const std::vector<Point2D> &points2D_1,
                         const std::vector<Point2D> &points2D_2)
        : num_data(points2D_1.size()), opt(ransac_opt), x1(points2D_1), x2(points2D_2),
          sampler(num_data, sample_sz, opt.seed, opt.progressive_sampling, opt.max_prosac_iterations) {
        x1s.resize(sample_sz);
        x2s.resize(sample_sz);
        sample.resize(sample_sz);
    }

    void generate_models(std::vector<Eigen::Matrix3d> *models);
    double score_model(const Eigen::Matrix3d &F, size_t *inlier_count) const;
    void refine_model(Eigen::Matrix3d *F) const;

    const size_t sample_sz = 7;
    const size_t num_data;

  private:
    const RansacOptions &opt;
    const std::vector<Point2D> &x1;
    const std::vector<Point2D> &x2;

    RandomSampler sampler;
    // pre-allocated vectors for sampling
    std::vector<Eigen::Vector3d> x1s, x2s;
    std::vector<size_t> sample;
};

} // namespace poselib

#endif