#ifndef MESH_HEADER_FILE
#define MESH_HEADER_FILE

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <fstream>
#include <vector>

#include "auxfunctions.h"
#include "readMESH.h"
#include "sparse_block_diagonal.h"

using namespace Eigen;
using namespace std;

// the class the contains each individual rigid objects and their functionality
class Mesh {
   public:
    // position
    VectorXd origPositions;  // 3|V|x1 original vertex positions in xyzxyz format - never change this!
    VectorXd currPositions;  // 3|V|x1 current vertex positions in xyzxyz format

    // kinematics
    bool isFixed;             // is the object immobile (infinite mass)
    VectorXd currVelocities;  // 3|V|x1 velocities per coordinate in xyzxyz format.

    double totalInvMass;

    MatrixXi T;               //|T|x4 tetrahdra
    MatrixXi F;               //|F|x3 boundary faces
    VectorXd invMasses;       //|V|x1 inverse masses of vertices, computed in the beginning as 1.0/(density * vertex voronoi area)
    VectorXd voronoiVolumes;  //|V|x1 the voronoi volume of vertices
    VectorXd tetVolumes;      //|T|x1 tetrahedra volumes
    int globalOffset;  // the global index offset of the of opositions/velocities/impulses from the beginning of the global coordinates
                       // array in the containing scene class

    VectorXi boundTets;  // just the boundary tets, for collision

    double youngModulus, poissonRatio, density, alpha, beta;

    SparseMatrix<double> K, M, D;  // The soft-body matrices

    // SimplicialLLT<SparseMatrix<double>>* ASolver;   //the solver for the left-hand side matrix constructed for FEM

    ~Mesh() { /*if (ASolver!=NULL) delete ASolver;*/
    }

    bool isNeighborTets(const RowVector4i& tet1, const RowVector4i& tet2) {
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                if (tet1(i) == tet2(j))  // shared vertex
                    return true;

        return false;
    }

    void create_M_matrix(SparseMatrix<double>& M) {
        M.resize(currPositions.size(), currPositions.size());

        vector<Triplet<double>> tripletsM;
        tripletsM.reserve(currPositions.size());

        for (int i = 0; i < voronoiVolumes.size(); i++) {
            double m = isFixed ? numeric_limits<double>::max() : voronoiVolumes(i) * density;
            for (int j = 0; j < 3; j++) tripletsM.push_back(Triplet<double>(3 * i + j, 3 * i + j, m));
        }

        M.setFromTriplets(tripletsM.begin(), tripletsM.end());
    }

    void create_K_matrix(SparseMatrix<double>& K) {
        vector<SparseMatrix<double>> Kes;
        SparseMatrix<double> Kprime(12 * T.rows(), 12 * T.rows());

        for (int e = 0; e < T.rows(); e++) {
            Matrix4d Pe;
            for (int i = 0; i < 4; i++) {
                Pe(i, 0) = 1.0;
                for (int j = 0; j < 3; j++) Pe(i, j + 1) = origPositions(3 * T.row(e)(i) + j);
            }

            Matrix<double, 3, 4> Ge = Pe.inverse().block<3, 4>(1, 0);

            double mu = youngModulus / (2.0 * (1.0 + poissonRatio));
            double lambda = youngModulus * poissonRatio / ((1.0 + poissonRatio) * (1.0 - 2.0 * poissonRatio));

            Matrix<double, 6, 6> C = Matrix<double, 6, 6>::Zero();
            for (int i = 0; i < 6; i++) {
                for (int j = 0; j < 6; j++) {
                    if (i == j)
                        C(i, j) = i < 3 ? lambda + 2.0 * mu : 2.0 * mu;
                    else if (i < 3 && j < 3)
                        C(i, j) = lambda;
                }
            }

            Matrix<double, 6, 12> B = Matrix<double, 6, 12>::Zero();
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 3; j++) B(j, 3 * i + j) = Ge(j, i);

                B(3, 3 * i + 0) = 0.5 * Ge(1, i);
                B(3, 3 * i + 1) = 0.5 * Ge(0, i);

                B(4, 3 * i + 1) = 0.5 * Ge(2, i);
                B(4, 3 * i + 2) = 0.5 * Ge(1, i);

                B(5, 3 * i + 0) = 0.5 * Ge(2, i);
                B(5, 3 * i + 2) = 0.5 * Ge(0, i);
            }

            Matrix<double, 3, 3> P, Q;
            for (int i = 0; i < 3; i++) {
                P.col(i) = origPositions.segment<3>(3 * T.row(e)(0)) - origPositions.segment<3>(3 * T.row(e)(i + 1));
                Q.col(i) = currPositions.segment<3>(3 * T.row(e)(0)) - currPositions.segment<3>(3 * T.row(e)(i + 1));
            }

            JacobiSVD<Matrix3d> svd(P.transpose() * Q, ComputeFullU | ComputeFullV);
            Matrix3d R = svd.matrixV() * svd.matrixU().transpose();
            if (R.determinant() < 0.0) {
                Matrix3d V = svd.matrixV();
                V.col(2) *= -1.0;
                R = V * svd.matrixU().transpose();
            }

            Matrix<double, 12, 12> Re = Matrix<double, 12, 12>::Zero();
            for (int i = 0; i < 4; ++i) Re.block<3, 3>(3 * i, 3 * i) = R;

            Matrix<double, 12, 12> Ke = Re * (B.transpose() * C * B * abs(Pe.determinant()) / 6.0) * Re.transpose();
            Kes.push_back(Ke.sparseView());
        }

        sparse_block_diagonal(Kes, Kprime);

        SparseMatrix<double> Q;
        vector<Triplet<double>> triplets;
        triplets.reserve(12 * T.rows());
        for (int e = 0; e < T.rows(); e++)
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 3; j++) triplets.emplace_back(12 * e + 3 * i + j, 3 * T(e, i) + j, 1.0);
        Q.resize(12 * T.rows(), origPositions.size());
        Q.setFromTriplets(triplets.begin(), triplets.end());

        K = Q.transpose() * Kprime * Q;
    }

    // Computing the K, M, D matrices per mesh.
    void create_global_matrices(const double timeStep, const double _alpha, const double _beta) {
        create_K_matrix(K);
        create_M_matrix(M);
        D = _alpha * M + _beta * K;
    }

    // returns center of mass
    Vector3d initializeVolumesAndMasses() {
        tetVolumes.conservativeResize(T.rows());
        voronoiVolumes.conservativeResize(origPositions.size() / 3);
        voronoiVolumes.setZero();
        invMasses.conservativeResize(origPositions.size() / 3);
        Vector3d COM;
        COM.setZero();
        for (int i = 0; i < T.rows(); i++) {
            Vector3d e01 = origPositions.segment(3 * T(i, 1), 3) - origPositions.segment(3 * T(i, 0), 3);
            Vector3d e02 = origPositions.segment(3 * T(i, 2), 3) - origPositions.segment(3 * T(i, 0), 3);
            Vector3d e03 = origPositions.segment(3 * T(i, 3), 3) - origPositions.segment(3 * T(i, 0), 3);
            Vector3d tetCentroid = (origPositions.segment(3 * T(i, 0), 3) + origPositions.segment(3 * T(i, 1), 3) +
                                    origPositions.segment(3 * T(i, 2), 3) + origPositions.segment(3 * T(i, 3), 3)) /
                                   4.0;
            tetVolumes(i) = abs(e01.dot(e02.cross(e03))) / 6.0;
            for (int j = 0; j < 4; j++) voronoiVolumes(T(i, j)) += tetVolumes(i) / 4.0;

            COM += tetVolumes(i) * tetCentroid;
        }

        COM.array() /= tetVolumes.sum();
        totalInvMass = 0.0;
        for (int i = 0; i < origPositions.size() / 3; i++) {
            invMasses(i) = 1.0 / (voronoiVolumes(i) * density);
            totalInvMass += voronoiVolumes(i) * density;
        }
        totalInvMass = 1.0 / totalInvMass;

        return COM;
    }

    Mesh(const VectorXd& _origPositions, const MatrixXi& boundF, const MatrixXi& _T, const int _globalOffset,
         const double _youngModulus, const double _poissonRatio, const double _density, const bool _isFixed,
         const RowVector3d& userCOM, const RowVector4d& userOrientation) {
        origPositions = _origPositions;
        // cout<<"original origPositions: "<<origPositions<<endl;
        T = _T;
        F = boundF;
        isFixed = _isFixed;
        globalOffset = _globalOffset;
        density = _density;
        poissonRatio = _poissonRatio;
        youngModulus = _youngModulus;
        currVelocities = VectorXd::Zero(origPositions.rows());

        VectorXd naturalCOM = initializeVolumesAndMasses();
        // cout<<"naturalCOM: "<<naturalCOM<<endl;

        origPositions -= naturalCOM.replicate(origPositions.rows() / 3,
                                              1);  // removing the natural COM of the OFF file (natural COM is never used again)
        // cout<<"after natrualCOM origPositions: "<<origPositions<<endl;

        for (int i = 0; i < origPositions.size(); i += 3)
            origPositions.segment(i, 3) << (QRot(origPositions.segment(i, 3).transpose(), userOrientation) + userCOM).transpose();

        currPositions = origPositions;

        if (isFixed) invMasses.setZero();

        // finding boundary tets
        VectorXi boundVMask(origPositions.rows() / 3);
        boundVMask.setZero();
        for (int i = 0; i < boundF.rows(); i++)
            for (int j = 0; j < 3; j++) boundVMask(boundF(i, j)) = 1;

        // cout<<"boundVMask.sum(): "<<boundVMask.sum()<<endl;

        vector<int> boundTList;
        for (int i = 0; i < T.rows(); i++) {
            int incidence = 0;
            for (int j = 0; j < 4; j++) incidence += boundVMask(T(i, j));
            if (incidence > 2) boundTList.push_back(i);
        }

        boundTets.resize(boundTList.size());
        for (int i = 0; i < boundTets.size(); i++) boundTets(i) = boundTList[i];
    }
};

#endif
