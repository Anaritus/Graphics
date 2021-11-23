#ifdef WIN32
#include <SDL.h>
#undef main
#else
#include <SDL2/SDL.h>
#endif

#include <GL/glew.h>

#include <string_view>
#include <stdexcept>
#include <iostream>
#include <chrono>
#include <vector>
#include <map>
#include <cmath>
#include <fstream>
#include <sstream>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define GLM_FORCE_SWIZZLE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/scalar_constants.hpp>
#include <glm/gtx/string_cast.hpp>

std::string to_string(std::string_view str)
{
	return std::string(str.begin(), str.end());
}

void sdl2_fail(std::string_view message)
{
	throw std::runtime_error(to_string(message) + SDL_GetError());
}

void glew_fail(std::string_view message, GLenum error)
{
	throw std::runtime_error(to_string(message) + reinterpret_cast<const char*>(glewGetErrorString(error)));
}

const char vertex_shader_source[] =
R"(#version 330 core
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_normal;
out vec3 position;
out vec3 normal;
void main()
{
	gl_Position = projection * view * model * vec4(in_position, 1.0);
	position = (model * vec4(in_position, 1.0)).xyz;
	normal = normalize((model * vec4(in_normal, 0.0)).xyz);
}
)";

const char fragment_shader_source[] =
R"(#version 330 core
uniform vec3 ambient;
uniform vec3 light_direction;
uniform vec3 light_color;
uniform mat4 transform;
uniform sampler2D shadow_map;
in vec3 position;
in vec3 normal;
layout (location = 0) out vec4 out_color;
void main()
{
	vec4 shadow_pos = transform * vec4(position, 1.0);
	shadow_pos /= shadow_pos.w;
	shadow_pos = shadow_pos * 0.5 + vec4(0.5);
	bool in_shadow_texture = (shadow_pos.x > 0.0) && (shadow_pos.x < 1.0) && (shadow_pos.y > 0.0) && (shadow_pos.y < 1.0) && (shadow_pos.z > 0.0) && (shadow_pos.z < 1.0);
	float shadow_factor = 1.0;
	float shadow_bias = -0.01;
	const int N = 1;
	vec2 shadow_map_size = vec2(textureSize(shadow_map, 0));
	vec2 data = vec2(0.0, 0.0);
	float weights = 0.0;
	for (int dx = -N; dx <= N; dx++) {
		for (int dy = -N; dy <= N; dy++) {
			data += texture(shadow_map, shadow_pos.xy + vec2(dx, dy) / shadow_map_size).rg * exp(-(dx * dx + dy * dy) / 2.0);
			weights += exp(-(dx * dx + dy * dy) / 2.0);
		}
	}
	data *= 1 / weights;
	float mu = data.r;
	float sigma = data.g - mu * mu;
	float z = shadow_pos.z + shadow_bias;
	float factor = (z < mu) ? 1.0 : sigma / (sigma + (z - mu) * (z - mu));
	float de = 0.125;
	factor = (factor <= de) ? 0.0: (factor - de) / (1 - de);
	if (in_shadow_texture)
		shadow_factor = factor;
	vec3 albedo = vec3(1.0, 1.0, 1.0);
	vec3 light = ambient;
	light += light_color * max(0.0, dot(normal, light_direction)) * shadow_factor;
	vec3 color = albedo * light;
	out_color = vec4(color, 1.0);
}
)";

const char debug_vertex_shader_source[] =
R"(#version 330 core
vec2 vertices[6] = vec2[6](
	vec2(-1.0, -1.0),
	vec2( 1.0, -1.0),
	vec2( 1.0,  1.0),
	vec2(-1.0, -1.0),
	vec2( 1.0,  1.0),
	vec2(-1.0,  1.0)
);
out vec2 texcoord;
void main()
{
	vec2 position = vertices[gl_VertexID];
	gl_Position = vec4(position * 0.25 + vec2(-0.75, -0.75), 0.0, 1.0);
	texcoord = position * 0.5 + vec2(0.5);
}
)";

const char debug_fragment_shader_source[] =
R"(#version 330 core
uniform sampler2D shadow_map;
in vec2 texcoord;
layout (location = 0) out vec4 out_color;
void main()
{
	out_color = texture(shadow_map, texcoord);
}
)";

const char shadow_vertex_shader_source[] =
R"(#version 330 core
uniform mat4 model;
uniform mat4 transform;
layout (location = 0) in vec3 in_position;
void main()
{
	gl_Position = transform * model * vec4(in_position, 1.0);
}
)";

const char shadow_fragment_shader_source[] =
R"(#version 330 core
out vec4 smth;
void main()
{
	float z = gl_FragCoord.z;
	smth = vec4(z, z * z + 0.25 * (dFdx(z) * dFdx(z) + dFdy(z) * dFdy(z)), 0.0, 0.0);
}
)";

GLuint create_shader(GLenum type, const char* source)
{
	GLuint result = glCreateShader(type);
	glShaderSource(result, 1, &source, nullptr);
	glCompileShader(result);
	GLint status;
	glGetShaderiv(result, GL_COMPILE_STATUS, &status);
	if (status != GL_TRUE)
	{
		GLint info_log_length;
		glGetShaderiv(result, GL_INFO_LOG_LENGTH, &info_log_length);
		std::string info_log(info_log_length, '\0');
		glGetShaderInfoLog(result, info_log.size(), nullptr, info_log.data());
		throw std::runtime_error("Shader compilation failed: " + info_log);
	}
	return result;
}

GLuint create_program(GLuint vertex_shader, GLuint fragment_shader)
{
	GLuint result = glCreateProgram();
	glAttachShader(result, vertex_shader);
	glAttachShader(result, fragment_shader);
	glLinkProgram(result);

	GLint status;
	glGetProgramiv(result, GL_LINK_STATUS, &status);
	if (status != GL_TRUE)
	{
		GLint info_log_length;
		glGetProgramiv(result, GL_INFO_LOG_LENGTH, &info_log_length);
		std::string info_log(info_log_length, '\0');
		glGetProgramInfoLog(result, info_log.size(), nullptr, info_log.data());
		throw std::runtime_error("Program linkage failed: " + info_log);
	}

	return result;
}

struct vertex
{
	glm::vec3 position;
	glm::vec3 normal;
	glm::vec2 texcoords;
};

struct obj
{
	std::vector<vertex> vertices;
	GLuint vao, vbo, ebo;
	GLuint aspec;
	GLuint albedo;
	GLuint normals;
	GLuint mask;
	bool hasAspec = false;
	bool hasMask = false;
};

std::pair<std::vector<vertex>, std::vector<std::uint32_t>> load_obj(std::istream& input)
{
	std::vector<vertex> vertices;
	std::vector<std::uint32_t> indices;

	for (std::string line; std::getline(input, line);)
	{
		std::istringstream line_stream(line);

		char type;
		line_stream >> type;

		if (type == '#')
			continue;

		if (type == 'v')
		{
			vertex v;
			line_stream >> v.position.x >> v.position.y >> v.position.z;
			vertices.push_back(v);
			continue;
		}

		if (type == 'f')
		{
			std::uint32_t i0, i1, i2;
			line_stream >> i0 >> i1 >> i2;
			--i0;
			--i1;
			--i2;
			indices.push_back(i0);
			indices.push_back(i1);
			indices.push_back(i2);
			continue;
		}

		throw std::runtime_error("Unknown OBJ row type: " + std::string(1, type));
	}

	return { vertices, indices };
}

std::pair<glm::vec3, glm::vec3> bbox(std::vector<vertex> const& vertices)
{
	static const float inf = std::numeric_limits<float>::infinity();

	glm::vec3 min = glm::vec3(inf);
	glm::vec3 max = glm::vec3(-inf);

	for (auto const& v : vertices)
	{
		min = glm::min(min, v.position);
		max = glm::max(max, v.position);
	}

	return { min, max };
}

void add_ground_plane(std::vector<vertex>& vertices, std::vector<std::uint32_t>& indices)
{
	auto [min, max] = bbox(vertices);

	glm::vec3 center = (min + max) / 2.f;
	glm::vec3 size = (max - min);
	size.x = size.z = std::max(size.x, size.z);

	float W = 5.f;
	float H = 0.5f;

	vertex v0, v1, v2, v3;
	v0.position = { center.x - W * size.x, center.y - H * size.y, center.z - W * size.z };
	v1.position = { center.x - W * size.x, center.y - H * size.y, center.z + W * size.z };
	v2.position = { center.x + W * size.x, center.y - H * size.y, center.z - W * size.z };
	v3.position = { center.x + W * size.x, center.y - H * size.y, center.z + W * size.z };

	std::uint32_t base_index = vertices.size();
	vertices.push_back(v0);
	vertices.push_back(v1);
	vertices.push_back(v2);
	vertices.push_back(v3);

	indices.push_back(base_index + 0);
	indices.push_back(base_index + 1);
	indices.push_back(base_index + 2);
	indices.push_back(base_index + 2);
	indices.push_back(base_index + 1);
	indices.push_back(base_index + 3);
}

void fill_info(std::vector<vertex>& vertices, std::vector<std::uint32_t> const& indices, tinyobj::attrib_t attrib, std::vector<tinyobj::shape_t> shapes, std::vector<tinyobj::material_t> materials)
{
	// Loop over shapes
	for (size_t s = 0; s < shapes.size(); s++) {
		// Loop over faces(polygon)

		size_t index_offset = 0;
		for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
			size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);

			// Loop over vertices in the face.
			for (size_t v = 0; v < fv; v++) {
				// access to vertex
				tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];

				tinyobj::real_t vx = attrib.vertices[3 * size_t(idx.vertex_index) + 0];
				tinyobj::real_t vy = attrib.vertices[3 * size_t(idx.vertex_index) + 1];
				tinyobj::real_t vz = attrib.vertices[3 * size_t(idx.vertex_index) + 2];

				// Check if `normal_index` is zero or positive. negative = no normal data
				if (idx.normal_index >= 0) {
					tinyobj::real_t nx = attrib.normals[3 * size_t(idx.normal_index) + 0];
					tinyobj::real_t ny = attrib.normals[3 * size_t(idx.normal_index) + 1];
					tinyobj::real_t nz = attrib.normals[3 * size_t(idx.normal_index) + 2];
				}
				// Check if `texcoord_index` is zero or positive. negative = no texcoord data
				if (idx.texcoord_index >= 0) {
					tinyobj::real_t tx = attrib.texcoords[2 * size_t(idx.texcoord_index) + 0];
					tinyobj::real_t ty = attrib.texcoords[2 * size_t(idx.texcoord_index) + 1];
				}
				// Optional: vertex colors
				// tinyobj::real_t red   = attrib.colors[3*size_t(idx.vertex_index)+0];
				// tinyobj::real_t green = attrib.colors[3*size_t(idx.vertex_index)+1];
				// tinyobj::real_t blue  = attrib.colors[3*size_t(idx.vertex_index)+2];
			}
			index_offset += fv;

			// per-face material
			shapes[s].mesh.material_ids[f];
		}
	}
}

std::vector<obj> make_sense(tinyobj::attrib_t attrib, std::vector<tinyobj::shape_t> shapes, std::vector<tinyobj::material_t> materials) {
	std::vector<obj> objects;
	for (auto shape : shapes) {
		glGenerateMipmap(GL_TEXTURE_2D);
		size_t index_offset = 0;
		auto copy = shape.mesh.material_ids;
		std::map<int, obj> objectOfShape;
		for (int i = 0; i < shape.mesh.num_face_vertices.size(); i++) {
			auto face = shape.mesh.num_face_vertices[i];
			int mat_id = shape.mesh.material_ids[i];
			if (!objectOfShape.count(mat_id)) {
				obj object;

				glActiveTexture(GL_TEXTURE0);
				glGenTextures(1, &object.albedo);
				glBindTexture(GL_TEXTURE_2D, object.albedo);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
				int width, height, channels;
				unsigned char* img = stbi_load(materials[mat_id].ambient_texname.c_str(), &width, &height, &channels, 0);
				glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, img);
				glGenerateMipmap(GL_TEXTURE_2D);

				glActiveTexture(GL_TEXTURE1);
				glGenTextures(1, &object.normals);
				glBindTexture(GL_TEXTURE_2D, object.normals);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
				img = stbi_load(materials[mat_id].normal_texname.c_str(), &width, &height, &channels, 0);
				glTexImage2D(GL_TEXTURE_2D, 0, GL_R3_G3_B2, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, img);
				glGenerateMipmap(GL_TEXTURE_2D);

				if (!materials[mat_id].specular_texname.empty()) {
					glActiveTexture(GL_TEXTURE2);
					glGenTextures(1, &object.aspec);
					glBindTexture(GL_TEXTURE_2D, object.aspec);
					glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
					glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
					img = stbi_load(materials[mat_id].specular_highlight_texname.c_str(), &width, &height, &channels, 0);
					glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, img);
					glGenerateMipmap(GL_TEXTURE_2D);
					object.hasAspec = true;
				}

				if (!materials[mat_id].displacement_texname.empty()) {
					glActiveTexture(GL_TEXTURE3);
					glGenTextures(1, &object.mask);
					glBindTexture(GL_TEXTURE_2D, object.mask);
					glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
					glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
					img = stbi_load(materials[mat_id].displacement_texname.c_str(), &width, &height, &channels, 0);
					glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, img);
					object.hasMask = true;
				}
				objectOfShape[mat_id] = object;
			}
			size_t fv = size_t(face);
			for (size_t v = 0; v < fv; v++) {
				vertex vert;

				tinyobj::index_t idx = shape.mesh.indices[index_offset + v];

				tinyobj::real_t vx = attrib.vertices[3 * size_t(idx.vertex_index) + 0];
				tinyobj::real_t vy = attrib.vertices[3 * size_t(idx.vertex_index) + 1];
				tinyobj::real_t vz = attrib.vertices[3 * size_t(idx.vertex_index) + 2];

				vert.position = { vx, vy, vz };

				// Check if `texcoord_index` is zero or positive. negative = no texcoord data
				if (idx.texcoord_index >= 0) {
					tinyobj::real_t tx = attrib.texcoords[2 * size_t(idx.texcoord_index) + 0];
					tinyobj::real_t ty = attrib.texcoords[2 * size_t(idx.texcoord_index) + 1];
					vert.texcoords = { tx, ty };
				}
				objectOfShape[mat_id].vertices.push_back(vert);
			}
			index_offset += fv;
		}
		for (std::map<int, obj>::iterator iter = objectOfShape.begin(); iter != objectOfShape.end(); iter++) {
			int mat_id = iter->first;
			obj object = iter->second;
			glGenVertexArrays(1, &object.vao);
			glBindVertexArray(object.vao);

			glGenBuffers(1, &object.vbo);
			glBindBuffer(GL_ARRAY_BUFFER, object.vbo);
			glBufferData(GL_ARRAY_BUFFER, object.vertices.size() * sizeof(object.vertices[0]), object.vertices.data(), GL_STATIC_DRAW);

			glEnableVertexAttribArray(0);
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)(0));
			glEnableVertexAttribArray(1);
			glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)(12));
			glEnableVertexAttribArray(2);
			glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)(24));
			objects.push_back(object);
		}
	}
	return objects;
}

void fill_normals(std::vector<vertex>& vertices, std::vector<std::uint32_t> const& indices)
{
	for (auto& v : vertices)
		v.normal = glm::vec3(0.f);

	for (std::size_t i = 0; i < indices.size(); i += 3)
	{
		auto& v0 = vertices[indices[i + 0]];
		auto& v1 = vertices[indices[i + 1]];
		auto& v2 = vertices[indices[i + 2]];

		glm::vec3 n = glm::cross(v1.position - v0.position, v2.position - v0.position);
		v0.normal += n;
		v1.normal += n;
		v2.normal += n;
	}

	for (auto& v : vertices)
		v.normal = glm::normalize(v.normal);
}


glm::vec3 choose(glm::vec3 a, glm::vec3 b, int choice)
{
	if (choice) return b;
	return a;
}

int main() try
{
	if (SDL_Init(SDL_INIT_VIDEO) != 0)
		sdl2_fail("SDL_Init: ");

	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
	SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
	SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 8);
	SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 8);
	SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 8);
	SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

	SDL_Window* window = SDL_CreateWindow("Graphics course practice 9",
		SDL_WINDOWPOS_CENTERED,
		SDL_WINDOWPOS_CENTERED,
		800, 600,
		SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_MAXIMIZED);

	if (!window)
		sdl2_fail("SDL_CreateWindow: ");

	int width, height;
	SDL_GetWindowSize(window, &width, &height);

	SDL_GLContext gl_context = SDL_GL_CreateContext(window);
	if (!gl_context)
		sdl2_fail("SDL_GL_CreateContext: ");

	if (auto result = glewInit(); result != GLEW_NO_ERROR)
		glew_fail("glewInit: ", result);

	if (!GLEW_VERSION_3_3)
		throw std::runtime_error("OpenGL 3.3 is not supported");

	auto vertex_shader = create_shader(GL_VERTEX_SHADER, vertex_shader_source);
	auto fragment_shader = create_shader(GL_FRAGMENT_SHADER, fragment_shader_source);
	auto program = create_program(vertex_shader, fragment_shader);

	GLuint model_location = glGetUniformLocation(program, "model");
	GLuint view_location = glGetUniformLocation(program, "view");
	GLuint projection_location = glGetUniformLocation(program, "projection");
	GLuint transform_location = glGetUniformLocation(program, "transform");

	GLuint ambient_location = glGetUniformLocation(program, "ambient");
	GLuint light_direction_location = glGetUniformLocation(program, "light_direction");
	GLuint light_color_location = glGetUniformLocation(program, "light_color");

	GLuint shadow_map_location = glGetUniformLocation(program, "shadow_map");

	glUseProgram(program);
	glUniform1i(shadow_map_location, 0);

	auto debug_vertex_shader = create_shader(GL_VERTEX_SHADER, debug_vertex_shader_source);
	auto debug_fragment_shader = create_shader(GL_FRAGMENT_SHADER, debug_fragment_shader_source);
	auto debug_program = create_program(debug_vertex_shader, debug_fragment_shader);

	GLuint debug_shadow_map_location = glGetUniformLocation(debug_program, "shadow_map");

	glUseProgram(debug_program);
	glUniform1i(debug_shadow_map_location, 0);

	auto shadow_vertex_shader = create_shader(GL_VERTEX_SHADER, shadow_vertex_shader_source);
	auto shadow_fragment_shader = create_shader(GL_FRAGMENT_SHADER, shadow_fragment_shader_source);
	auto shadow_program = create_program(shadow_vertex_shader, shadow_fragment_shader);

	GLuint shadow_model_location = glGetUniformLocation(shadow_program, "model");
	GLuint shadow_transform_location = glGetUniformLocation(shadow_program, "transform");

	std::vector<vertex> vertices;
	std::vector<std::uint32_t> indices;
	{
		std::ifstream bunny_file(PRACTICE_SOURCE_DIRECTORY "/bunny.obj");
		tinyobj::attrib_t attrib;
		std::vector<tinyobj::shape_t> shapes;
		std::vector<tinyobj::material_t> materials;

		std::string err;

		bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &err, PRACTICE_SOURCE_DIRECTORY "/sponza.obj", PRACTICE_SOURCE_DIRECTORY "/");

		if (!err.empty()) {
			std::cerr << err << std::endl;
		}

		if (!ret) {
			exit(1);
		}

		fill_info(vertices, indices, attrib, shapes, materials);
		auto mat = materials[0];
		std::vector<obj> bla = make_sense(attrib, shapes, materials);
		
		std::tie(vertices, indices) = load_obj(bunny_file);
		
	}
	add_ground_plane(vertices, indices);
	fill_normals(vertices, indices);

	glm::vec3 min, max;
	std::tie(min, max) = bbox(vertices);
	glm::vec3 C = (min + max) / 2.0f;

	GLuint vao, vbo, ebo;
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(vertices[0]), vertices.data(), GL_STATIC_DRAW);

	glGenBuffers(1, &ebo);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(indices[0]), indices.data(), GL_STATIC_DRAW);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)(0));
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)(12));

	GLuint debug_vao;
	glGenVertexArrays(1, &debug_vao);

	GLsizei shadow_map_resolution = 1024;

	GLuint shadow_map;
	glGenTextures(1, &shadow_map);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, shadow_map);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	//glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
	//glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LESS);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RG32F, shadow_map_resolution, shadow_map_resolution, 0, GL_RGBA, GL_FLOAT, nullptr);

	GLuint rbo;
	glGenRenderbuffers(1, &rbo);
	glBindRenderbuffer(GL_RENDERBUFFER, rbo);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, shadow_map_resolution, shadow_map_resolution);

	GLuint shadow_fbo;
	glGenFramebuffers(1, &shadow_fbo);
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, shadow_fbo);
	glFramebufferTexture(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, shadow_map, 0);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo);
	if (glCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
		throw std::runtime_error("Incomplete framebuffer!");
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);

	auto last_frame_start = std::chrono::high_resolution_clock::now();

	float time = 0.f;
	bool paused = false;

	std::map<SDL_Keycode, bool> button_down;

	float view_elevation = glm::radians(45.f);
	float view_azimuth = 0.f;
	float camera_distance = 0.5f;
	float camera_target = 0.05f;
	bool running = true;
	while (running)
	{
		for (SDL_Event event; SDL_PollEvent(&event);) switch (event.type)
		{
		case SDL_QUIT:
			running = false;
			break;
		case SDL_WINDOWEVENT: switch (event.window.event)
		{
		case SDL_WINDOWEVENT_RESIZED:
			width = event.window.data1;
			height = event.window.data2;
			glViewport(0, 0, width, height);
			break;
		}
							break;
		case SDL_KEYDOWN:
			button_down[event.key.keysym.sym] = true;

			if (event.key.keysym.sym == SDLK_SPACE)
				paused = !paused;

			break;
		case SDL_KEYUP:
			button_down[event.key.keysym.sym] = false;
			break;
		}

		if (!running)
			break;

		auto now = std::chrono::high_resolution_clock::now();
		float dt = std::chrono::duration_cast<std::chrono::duration<float>>(now - last_frame_start).count();
		last_frame_start = now;
		if (!paused)
			time += dt;

		if (button_down[SDLK_UP])
			camera_distance -= 1.f * dt;
		if (button_down[SDLK_DOWN])
			camera_distance += 1.f * dt;

		if (button_down[SDLK_LEFT])
			view_azimuth -= 2.f * dt;
		if (button_down[SDLK_RIGHT])
			view_azimuth += 2.f * dt;

		glm::mat4 model(1.f);

		glm::vec3 light_direction = glm::normalize(glm::vec3(std::cos(time * 0.5f), 1.f, std::sin(time * 0.5f)));

		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, shadow_fbo);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glViewport(0, 0, shadow_map_resolution, shadow_map_resolution);

		glEnable(GL_DEPTH_TEST);
		glDepthFunc(GL_LEQUAL);

		glEnable(GL_CULL_FACE);
		glCullFace(GL_BACK);

		glm::vec3 light_z = -light_direction;
		glm::vec3 light_x = glm::normalize(glm::cross(light_z, { 0.f, 1.f, 0.f }));
		glm::vec3 light_y = glm::cross(light_x, light_z);

		glm::vec3 mods = glm::vec3(0.0f, 0.0f, 0.0f);
		for (int i = 0; i < 8; i++) {
			int j = i;
			float x, y, z;
			x = choose(min, max, j % 2).x;
			j /= 2;
			y = choose(min, max, j % 2).y;
			j /= 2;
			z = choose(min, max, j % 2).z;
			glm::vec3 V = glm::vec3(x, y, z);
			glm::vec3 mod = glm::vec3(std::abs(glm::dot(V - C, light_x)), std::abs(glm::dot(V - C, light_y)), std::abs(glm::dot(V - C, light_z)));
			mods = glm::max(mods, mod);
		}

		light_x *= mods.x;
		light_y *= mods.y;
		light_z *= mods.z;

		glm::mat4 transform = glm::mat4(1.f);
		for (size_t i = 0; i < 3; ++i)
		{
			transform[0][i] = light_x[i];
			transform[1][i] = light_y[i];
			transform[2][i] = light_z[i];
		}

		transform = glm::translate(transform, C);
		transform = glm::inverse(transform);

		glUseProgram(shadow_program);
		glUniformMatrix4fv(shadow_model_location, 1, GL_FALSE, reinterpret_cast<float*>(&model));
		glUniformMatrix4fv(shadow_transform_location, 1, GL_FALSE, reinterpret_cast<float*>(&transform));

		glBindVertexArray(vao);
		glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, nullptr);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, shadow_map);
		glGenerateMipmap(GL_TEXTURE_2D);

		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
		glViewport(0, 0, width, height);

		glClearColor(0.8f, 0.8f, 0.9f, 0.f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glEnable(GL_DEPTH_TEST);
		glDepthFunc(GL_LEQUAL);

		glEnable(GL_CULL_FACE);
		glCullFace(GL_BACK);

		float near = 0.01f;
		float far = 10.f;

		glm::mat4 view(1.f);
		view = glm::translate(view, { 0.f, 0.f, -camera_distance });
		view = glm::rotate(view, view_elevation, { 1.f, 0.f, 0.f });
		view = glm::rotate(view, view_azimuth, { 0.f, 1.f, 0.f });
		view = glm::translate(view, { 0.f, -camera_target, 0.f });

		glm::mat4 projection = glm::mat4(1.f);
		projection = glm::perspective(glm::pi<float>() / 2.f, (1.f * width) / height, near, far);

		glBindTexture(GL_TEXTURE_2D, shadow_map);

		glUseProgram(program);
		glUniformMatrix4fv(model_location, 1, GL_FALSE, reinterpret_cast<float*>(&model));
		glUniformMatrix4fv(view_location, 1, GL_FALSE, reinterpret_cast<float*>(&view));
		glUniformMatrix4fv(projection_location, 1, GL_FALSE, reinterpret_cast<float*>(&projection));
		glUniformMatrix4fv(transform_location, 1, GL_FALSE, reinterpret_cast<float*>(&transform));

		glUniform3f(ambient_location, 0.2f, 0.2f, 0.2f);
		glUniform3fv(light_direction_location, 1, reinterpret_cast<float*>(&light_direction));
		glUniform3f(light_color_location, 0.8f, 0.8f, 0.8f);

		glBindVertexArray(vao);
		glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, nullptr);

		glUseProgram(debug_program);
		glBindTexture(GL_TEXTURE_2D, shadow_map);
		glBindVertexArray(debug_vao);
		glDrawArrays(GL_TRIANGLES, 0, 6);

		SDL_GL_SwapWindow(window);
	}

	SDL_GL_DeleteContext(gl_context);
	SDL_DestroyWindow(window);
}
catch (std::exception const& e)
{
	std::cerr << e.what() << std::endl;
	return EXIT_FAILURE;
}