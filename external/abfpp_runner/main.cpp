#include <OpenABF/OpenABF.hpp>

#include <array>
#include <cstddef>
#include <exception>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Vertex = std::array<double, 3>;
using Face = std::array<std::size_t, 3>;

std::size_t parse_vertex_index(const std::string& token, std::size_t vertex_count) {
  const auto slash = token.find('/');
  const std::string index_text = token.substr(0, slash);
  if (index_text.empty()) {
    throw std::runtime_error("OBJ face contains an empty vertex index");
  }
  const long long raw = std::stoll(index_text);
  const long long resolved = raw > 0 ? raw - 1 : static_cast<long long>(vertex_count) + raw;
  if (resolved < 0 || resolved >= static_cast<long long>(vertex_count)) {
    throw std::runtime_error("OBJ face vertex index is out of range");
  }
  return static_cast<std::size_t>(resolved);
}

void read_obj(
    const std::string& path,
    std::vector<Vertex>& vertices,
    std::vector<Face>& faces) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("failed to open input OBJ");
  }
  std::string line;
  while (std::getline(input, line)) {
    std::istringstream stream(line);
    std::string kind;
    stream >> kind;
    if (kind == "v") {
      Vertex vertex{};
      if (!(stream >> vertex[0] >> vertex[1] >> vertex[2])) {
        throw std::runtime_error("invalid OBJ vertex record");
      }
      vertices.push_back(vertex);
    } else if (kind == "f") {
      std::vector<std::size_t> polygon;
      std::string token;
      while (stream >> token) {
        polygon.push_back(parse_vertex_index(token, vertices.size()));
      }
      if (polygon.size() != 3) {
        throw std::runtime_error("ABF++ input must be triangular");
      }
      faces.push_back({polygon[0], polygon[1], polygon[2]});
    }
  }
  if (vertices.empty() || faces.empty()) {
    throw std::runtime_error("input OBJ contains no mesh");
  }
}

template <typename MeshPointer>
void write_obj(
    const std::string& path,
    const std::vector<Vertex>& vertices,
    const std::vector<Face>& faces,
    const MeshPointer& mesh) {
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("failed to open output OBJ");
  }
  output.precision(17);
  output << "# Written by surface_nvp_abfpp\n";
  for (const auto& vertex : vertices) {
    output << "v " << vertex[0] << ' ' << vertex[1] << ' ' << vertex[2] << '\n';
  }
  for (const auto& vertex : mesh->vertices()) {
    output << "vt " << vertex->pos[0] << ' ' << vertex->pos[1] << '\n';
  }
  for (const auto& face : faces) {
    output << "f";
    for (const auto index : face) {
      const auto one_based = index + 1;
      output << ' ' << one_based << '/' << one_based;
    }
    output << '\n';
  }
}

}  // namespace

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cerr << "Usage: surface_nvp_abfpp INPUT.obj OUTPUT.obj\n";
    return 2;
  }

  try {
    std::vector<Vertex> vertices;
    std::vector<Face> faces;
    read_obj(argv[1], vertices, faces);

    using ABF = OpenABF::ABFPlusPlus<double>;
    using LSCM = OpenABF::AngleBasedLSCM<double, ABF::Mesh>;
    auto mesh = ABF::Mesh::New();
    for (const auto& vertex : vertices) {
      mesh->insert_vertex(vertex[0], vertex[1], vertex[2]);
    }
    mesh->insert_faces(faces);
    ABF::Compute(mesh);
    LSCM::Compute(mesh);
    write_obj(argv[2], vertices, faces, mesh);
    std::cout << "ABF++ parameterized " << vertices.size() << " vertices and "
              << faces.size() << " triangles\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "ABF++ failed: " << error.what() << '\n';
    return 1;
  }
}
