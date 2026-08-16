#include <igl/MappingEnergyType.h>
#include <igl/readOBJ.h>
#include <igl/slim.h>
#include <igl/writeOBJ.h>

#include <Eigen/Core>

#include <exception>
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
  if (argc != 4) {
    std::cerr << "Usage: surface_nvp_slim INPUT.obj OUTPUT.obj ITERATIONS\n";
    return 2;
  }

  int iterations = 0;
  try {
    iterations = std::stoi(argv[3]);
  } catch (const std::exception&) {
    std::cerr << "ITERATIONS must be a positive integer\n";
    return 2;
  }
  if (iterations <= 0) {
    std::cerr << "ITERATIONS must be a positive integer\n";
    return 2;
  }

  Eigen::MatrixXd vertices, uv, normals;
  Eigen::MatrixXi faces, face_uv, face_normals;
  if (!igl::readOBJ(argv[1], vertices, uv, normals, faces, face_uv, face_normals)) {
    std::cerr << "Failed to read input OBJ\n";
    return 1;
  }
  if (faces.cols() != 3) {
    std::cerr << "SLIM input must be triangular\n";
    return 1;
  }
  if (uv.rows() != vertices.rows() || uv.cols() < 2 || face_uv.rows() != faces.rows() || face_uv.cols() != faces.cols()) {
    std::cerr << "Input OBJ must contain one UV coordinate per vertex\n";
    return 1;
  }
  if ((face_uv.array() != faces.array()).any()) {
    std::cerr << "Input OBJ must use matching vertex and UV indices\n";
    return 1;
  }

  uv = uv.leftCols(2);
  Eigen::VectorXi fixed_indices(0);
  Eigen::MatrixXd fixed_values(0, 2);
  igl::SLIMData data;
  igl::slim_precompute(
      vertices,
      faces,
      uv,
      data,
      igl::MappingEnergyType::SYMMETRIC_DIRICHLET,
      fixed_indices,
      fixed_values,
      0.0);
  igl::slim_solve(data, iterations);

  if (!igl::writeOBJ(
          argv[2],
          vertices,
          faces,
          Eigen::MatrixXd(),
          Eigen::MatrixXi(),
          data.V_o,
          faces)) {
    std::cerr << "Failed to write output OBJ\n";
    return 1;
  }
  std::cout << "SLIM energy: " << data.energy << "\n";
  return 0;
}
