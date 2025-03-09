#ifndef MESH_HEADER_FILE
#define MESH_HEADER_FILE

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <fstream>
#include <vector>

#include "auxfunctions.h"
#include "readMESH.h"

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

    // Computing the K, M, D matrices per mesh.
    void create_global_matrices(const double timeStep, const double _alpha, const double _beta) {
        alpha = _alpha;
        beta = _beta;

        // ----- Calculate K -----
        K.resize(currPositions.size(), currPositions.size());

        std::vector<Triplet<double>> tripletsK;
        tripletsK.reserve(T.rows() * 12 * 12);

        double mu = youngModulus / (2.0 * (1.0 + poissonRatio));
        double lambda = youngModulus * poissonRatio / ((1.0 + poissonRatio) * (1.0 - 2.0 * poissonRatio));

        for (int ti = 0; ti < T.rows(); ti++) {
            Vector4i tet = T.row(ti);

            Matrix<double, 3, 4> X;
            for (int i = 0; i < 4; i++) X.col(i) = origPositions.segment(3 * tet(i), 3);

            Matrix3d Dm;
            for (int i = 0; i < 3; i++) Dm.col(i) = X.col(i + 1) - X.col(0);
            Matrix3d DmInv = Dm.inverse();

            Matrix<double, 3, 4> B;
            B.col(0) = -DmInv.col(0) - DmInv.col(1) - DmInv.col(2);
            for (int i = 1; i < 4; i++) B.col(i) = DmInv.col(i - 1);

            Matrix<double, 12, 12> Ke = Matrix<double, 12, 12>::Zero();
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    Matrix3d Kij = lambda * B.col(i) * B.col(j).transpose();

                    for (int k = 0; k < 3; k++)
                        for (int l = 0; l < 3; l++) Kij(k, l) += mu * (B(l, i) * B(k, j) + B(k, i) * B(l, j));

                    Kij *= tetVolumes(ti);

                    for (int k = 0; k < 3; k++)
                        for (int l = 0; l < 3; l++) Ke(3 * i + k, 3 * j + l) = Kij(k, l);
                }
            }

            if (isFixed) continue;

            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                    for (int di = 0; di < 3; di++)
                        for (int dj = 0; dj < 3; dj++)
                            tripletsK.push_back(Triplet<double>(3 * tet(i) + di, 3 * tet(j) + dj, Ke(3 * i + di, 3 * j + dj)));
        }

        K.setFromTriplets(tripletsK.begin(), tripletsK.end());

        // ----- Calculate M -----
        M.resize(currPositions.size(), currPositions.size());

        vector<Triplet<double>> tripletsM;
        tripletsM.reserve(currPositions.size());

        for (int i = 0; i < voronoiVolumes.size(); i++)
            for (int j = 0; j < 3; j++) tripletsM.push_back(Triplet<double>(3 * i + j, 3 * i + j, voronoiVolumes(i) * density));

        M.setFromTriplets(tripletsM.begin(), tripletsM.end());

        // ----- Calculate D -----
        D = _alpha * M + _beta * K;
    }

    // returns center of mass
    Vector3d initializeVolumesAndMasses() {
        // TODO: compute tet volumes and allocate to vertices
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
            tetVolumes(i) = std::abs(e01.dot(e02.cross(e03))) / 6.0;
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
