#ifdef WIN32
#include <SDL.h>
#undef main
#else
#include <SDL2/SDL.h>
#endif

#define PI 3.14159265359f

#include <GL/glew.h>

#include <string_view>
#include <stdexcept>
#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <map>
#include <cmath>

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
uniform mat4 view;
uniform mat4 transform;
uniform int black;
layout (location = 0) in float x;
layout (location = 1) in vec4 in_color;
layout (location = 2) in float z;
layout (location = 3) in float y;
out vec4 color;
void main()
{
	gl_Position = view * transform * vec4(x, y, z, 1.0);
	color = black * in_color;
}
)";

const char fragment_shader_source[] =
R"(#version 330 core
in vec4 color;
layout (location = 0) out vec4 out_color;
void main()
{
	out_color = color;
}
)";

const char vertex_shader_source_2D[] =
R"(#version 330 core
uniform mat4 view;
layout (location = 0) in float x;
layout (location = 1) in vec4 in_color;
layout (location = 2) in float z;
layout (location = 3) in float y;
out vec4 color;
void main()
{
	gl_Position = view * vec4(x, y, z, 1.0);
	color = in_color;
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

struct vec3
{
	float x;
	float y;
	float z;

	bool operator==(const vec3& r)
	{
		return this->x == r.x && this->y == r.y && this->z == r.z;
	}
};

struct vec2
{
	float x;
	float y;
};

struct vertex
{
	vec3 position;
	std::uint8_t color[4];
};

struct funcy
{
	std::vector<vec2> roots;
	float cap = 1.f;

	void normalize()
	{
		float my = 0.8f;
		for (auto t : roots) {
			float y = apply(t).position.y;
			if (abs(y) > my) my = abs(y);
		}
		cap *= my / 0.8f;
	}

	void newRoot(vec2 root)
	{
		roots.push_back(root);
		normalize();
	}
	vertex apply(vec2 point)
	{
		float retZ = 0;
		int cnt = 1;
		float d;
		for (auto root : roots) {
			d = std::hypot(root.x - point.x, root.y - point.y);
			retZ += cnt * 0.8f * exp(-d * d * 10);
			cnt = -cnt;
		}
		retZ /= cap;
		std::uint8_t col = static_cast<uint8_t>((retZ / 0.8f + 1) * 64);
		return { {point.x, retZ,  point.y}, {static_cast<uint8_t>(col * 2), 0, static_cast<uint8_t>(-col), 255} };
	}
	void popRoot()
	{
		roots.pop_back();
		normalize();
	}
};

void genindices(GLuint array_buffer, GLuint array, GLuint ebo = 0, GLuint another_buffer=0)
{
	glBindVertexArray(array);
	if (ebo != 0) glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
	glBindBuffer(GL_ARRAY_BUFFER, array_buffer);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 1, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)(0));
	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)(2 * sizeof(float)));
	if (another_buffer != 0) glBindBuffer(GL_ARRAY_BUFFER, another_buffer);
	glEnableVertexAttribArray(3);
	glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)(sizeof(float)));
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 4, GL_UNSIGNED_BYTE, GL_TRUE, sizeof(vertex), (void*)(sizeof(vec3)));
	
}

std::vector<vertex> calculate(funcy f, int quality)
{
	std::vector<vertex> dots;
	for (int y = 0; y <= quality; y++) {
		for (int x = 0; x <= quality; x++) {
			dots.push_back(f.apply({ 2.0f * x / quality - 1, 2.0f * y / quality - 1 }));
		}
	}
	return dots;
}

std::vector<int> calculate_ind(int quality)
{
	std::vector<int> dots_ind;
	for (int y = 0; y <= quality; y++) {
		for (int x = 0; x <= quality; x++) {
			if (x < quality && y < quality) {
				dots_ind.push_back(x * (quality + 1) + y);
				dots_ind.push_back(x * (quality + 1) + y + 1);
				dots_ind.push_back((x + 1) * (quality + 1) + y);
				dots_ind.push_back((x + 1) * (quality + 1) + y);
				dots_ind.push_back(x * (quality + 1) + y + 1);
				dots_ind.push_back((x + 1) * (quality + 1) + y + 1);
			}
		}
	}
	return dots_ind;
}

vec3 interpolate(vec3 left, vec3 right, float level)
{
	float a = (left.y - level) / (left.y - right.y);
	float b = (level - right.y) / (left.y - right.y);
	return { left.x * b + right.x * a, level + 0.001f, left.z * b + right.z * a };
}

vertex blackV(vec3 point)
{
	return { point, {255, 255, 255, 255} };
}

int getInd(std::vector<std::pair<vec3, int>> points, vec3 point) {
	for (auto p : points) {
		if (p.first == point) {
			return p.second;
		}
	}
}

std::pair <std::vector<vertex>, std::vector<int>> calculateSquare(int quality, funcy f, float level)
{
	std::vector<vertex> dots;
	std::vector<int> inds;
	int n = 0;
	vec3 lu, ru, ld, rd;
	vec3 point1, point2;
	std::vector<std::pair<vec3, int>> pointToInd;
	int i, j;
	for (int y = 0; y < quality; y++) {
		for (int x = 0; x < quality; x++) {
			int caseType = 0;
			ld = f.apply({ 2.0f * x / quality - 1, 2.0f * y / quality - 1 }).position;
			lu = f.apply({ 2.0f * x / quality - 1, 2.0f * (y + 1) / quality - 1 }).position;
			rd = f.apply({ 2.0f * (x + 1) / quality - 1, 2.0f * y / quality - 1 }).position;
			ru = f.apply({ 2.0f * (x + 1) / quality - 1, 2.0f * (y + 1) / quality - 1 }).position;
			caseType += rd.y > level;
			caseType += (ld.y > level) * 2;
			caseType += (ru.y > level) * 4;
			caseType += (lu.y > level) * 8;
			if (caseType > 7) caseType = 15 - caseType;
			switch (caseType)
			{
			case 1: //0001
				point1 = interpolate(ld, rd, level);
				if (y == 0) {
					dots.push_back(blackV(point1));
					pointToInd.push_back({ point1, n });
					n++;
				}
				i = getInd(pointToInd, point1);
				point2 = interpolate(rd, ru, level);
				dots.push_back(blackV(point2));
				pointToInd.push_back({ point2, n });
				j = n;
				n++;
				inds.push_back(i);
				inds.push_back(j);
				continue;
			case 2: //0010
				point1 = interpolate(ld, rd, level);
				if (y == 0) {
					dots.push_back(blackV(point1));
					pointToInd.push_back({ point1, n });
					n++;
				}
				i = getInd(pointToInd, point1);
				point2 = interpolate(ld, lu, level);
				if (x == 0) {
					dots.push_back(blackV(point2));
					pointToInd.push_back({ point2, n });
					n++;
				}
				j = getInd(pointToInd, point2);
				inds.push_back(i);
				inds.push_back(j);
				continue;
			case 3: //0011
				point1 = interpolate(rd, ru, level);
				dots.push_back(blackV(point1));
				pointToInd.push_back({ point1, n });
				i = n;
				n++;
				point2 = interpolate(ld, lu, level);
				if (x == 0) {
					dots.push_back(blackV(point2));
					pointToInd.push_back({ point2, n });
					n++;
				}
				j = getInd(pointToInd, point2);
				inds.push_back(i);
				inds.push_back(j);
				continue;
			case 4: //0100
				point1 = interpolate(lu, ru, level);
				dots.push_back(blackV(point1));
				pointToInd.push_back({ point1, n });
				i = n;
				n++;
				point2 = interpolate(rd, ru, level);
				dots.push_back(blackV(point2));
				pointToInd.push_back({ point2, n });
				j = n;
				n++;
				inds.push_back(i);
				inds.push_back(j);
				continue;
			case 5: //0101
				point1 = interpolate(lu, ru, level);
				dots.push_back(blackV(point1));
				pointToInd.push_back({ point1, n });
				i = n;
				n++;
				point2 = interpolate(ld, rd, level);
				if (y == 0) {
					dots.push_back(blackV(point2));
					pointToInd.push_back({ point2, n });
					n++;
				}
				j = getInd(pointToInd, point2);
				inds.push_back(i);
				inds.push_back(j);
				continue;
			case 6: //0110
				point1 = interpolate(lu, ru, level);
				dots.push_back(blackV(point1));
				pointToInd.push_back({ point1, n });
				i = n;
				n++;
				point2 = interpolate(ld, lu, level);
				if (x == 0) {
					dots.push_back(blackV(point2));
					pointToInd.push_back({ point2, n });
					n++;
				}
				j = getInd(pointToInd, point2);
				inds.push_back(i);
				inds.push_back(j);
				point1 = interpolate(rd, ru, level);
				dots.push_back(blackV(point1));
				pointToInd.push_back({ point1, n });
				i = n;
				n++;
				point2 = interpolate(ld, rd, level);
				if (y == 0) {
					dots.push_back(blackV(point2));
					pointToInd.push_back({ point2, n });
					n++;
				}
				j = getInd(pointToInd, point2);
				inds.push_back(i);
				inds.push_back(j);
				continue;
			case 7: //0111
				point1 = interpolate(lu, ru, level);
				dots.push_back(blackV(point1));
				pointToInd.push_back({ point1, n });
				i = n;
				n++;
				point2 = interpolate(ld, lu, level);
				if (x == 0) {
					dots.push_back(blackV(point2));
					pointToInd.push_back({ point2, n });
					n++;
				}
				j = getInd(pointToInd, point2);
				inds.push_back(i);
				inds.push_back(j);
				continue;
			}
		}
	}
	return std::pair(dots, inds);
}

static vertex cube_vertices[]
{
	// -X
	{{-1.f, -1.f, -1.f}, {  255, 255, 255, 255}},
	{{-1.f, -1.f,  1.f}, {  255, 255, 255, 255}},
	{{-1.f,  1.f, -1.f}, {  255, 255, 255, 255}},
	{{-1.f,  1.f,  1.f}, {  255, 255, 255, 255}},
	// -Y
	{{-1.f, -1.f, -1.f}, {  255, 255, 255, 255}},
	{{ 1.f, -1.f, -1.f}, {  255, 255, 255, 255}},
	{{-1.f, -1.f,  1.f}, {  255, 255, 255, 255}},
	{{ 1.f, -1.f,  1.f}, {  255, 255, 255, 255}},
	// -Z
	{{ 1.f, -1.f, -1.f}, {  255, 255, 255, 255}},
	{{-1.f, -1.f, -1.f}, {  255, 255, 255, 255}},
	{{ 1.f,  1.f, -1.f}, {  255, 255, 255, 255}},
	{{-1.f,  1.f, -1.f}, {  255, 255, 255, 255}}
};

static vertex axises[]
{
	{{-0.99f,  1.f, -0.99f}, {0, 0, 0, 0}}, //x
	{{1.f,  -0.99f, -0.99f}, {0, 0, 0, 0}}, //y
	{{-0.99f,  -0.99f, 1.f}, {0, 0, 0, 0}}, //z
	{{-0.99f,  -0.99f, -0.99f}, {0, 0, 0, 0}} //0
};

static vertex input_square[]
{
	{{1.5f, -0.6f, 0.f}, {255, 255,   255, 255}}, //+-
	{{0.9f, -0.6f, 0.f}, {255, 255,   255, 255}}, //--
	{{1.5f, 0.f, 0.f}, {255, 255,   255, 255}}, //++
	{{0.9f, 0.f, 0.f}, {255, 255,   255, 255}} //-+
};

static std::uint32_t cube_indices[]
{
	// -X
	0, 1, 2, 2, 1, 3,

	// -Y
	4, 5, 6, 6, 5, 7,

	// -Z
	8, 9, 10, 10, 9, 11
};

static std::uint32_t vert_indices[]
{
	0, 3,
	1, 3,
	2
};



int main() try
{
	if (SDL_Init(SDL_INIT_VIDEO) != 0)
		sdl2_fail("SDL_Init: ");

	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
	SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
	SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 1);
	SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 4);
	SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 8);
	SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 8);
	SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 8);
	SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

	SDL_Window* window = SDL_CreateWindow("Graphics course practice 4",
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

	glClearColor(0.f, 0.f, 0.f, 0.f);

	auto vertex_shader = create_shader(GL_VERTEX_SHADER, vertex_shader_source);
	auto vertex_shader_2D = create_shader(GL_VERTEX_SHADER, vertex_shader_source_2D);
	auto fragment_shader = create_shader(GL_FRAGMENT_SHADER, fragment_shader_source);
	auto program = create_program(vertex_shader, fragment_shader);
	auto program_2D = create_program(vertex_shader_2D, fragment_shader);

	GLuint view_location = glGetUniformLocation(program, "view");
	GLuint view_location_2D = glGetUniformLocation(program_2D, "view");
	GLuint transform_location = glGetUniformLocation(program, "transform");
	GLuint black_location = glGetUniformLocation(program, "black");

	auto last_frame_start = std::chrono::high_resolution_clock::now();

	std::vector<vertex> mass;

	GLuint vao_cube, vao_axes, vao_square, vao_strip, vao_f;
	GLuint vbo_cube, vbo_axes, vbo_square, vbo_strip, vbo_f, vbo_xy;
	GLuint ebo_cube, ebo_axes, ebo_f;

	glGenBuffers(1, &vbo_cube);
	glGenBuffers(1, &vbo_axes);
	glGenBuffers(1, &vbo_strip);
	glGenBuffers(1, &vbo_square);
	glGenBuffers(1, &vbo_f);
	glGenBuffers(1, &vbo_xy);

	glGenBuffers(1, &ebo_cube);
	glGenBuffers(1, &ebo_axes);
	glGenBuffers(1, &ebo_f);

	glGenVertexArrays(1, &vao_cube);
	glGenVertexArrays(1, &vao_axes);
	glGenVertexArrays(1, &vao_square);
	glGenVertexArrays(1, &vao_strip);
	glGenVertexArrays(1, &vao_f);


	genindices(vbo_cube, vao_cube, ebo_cube);
	glBufferData(GL_ARRAY_BUFFER, sizeof(cube_vertices), &cube_vertices, GL_STATIC_COPY);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(cube_indices), &cube_indices, GL_STATIC_COPY);
	genindices(vbo_axes, vao_axes, ebo_axes);
	glBufferData(GL_ARRAY_BUFFER, sizeof(axises), &axises, GL_STATIC_COPY);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(vert_indices), &vert_indices, GL_STATIC_COPY);
	genindices(vbo_square, vao_square);
	glBufferData(GL_ARRAY_BUFFER, sizeof(input_square), &input_square, GL_STATIC_COPY);
	genindices(vbo_strip, vao_strip);

	funcy f;
	int quality = 20;
	std::vector<vertex> dots = calculate(f, quality);
	std::vector<int> dots_ind = calculate_ind(quality);
	genindices(vbo_xy, vao_f, ebo_f, vbo_f);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_xy);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertex)* dots.size(), dots.data(), GL_STATIC_COPY);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_f);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertex)* dots.size(), dots.data(), GL_STATIC_COPY);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_f);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(int) * dots_ind.size(), dots_ind.data(), GL_STATIC_COPY);


	int squality = 10;
	std::vector<GLuint> sqvao(squality);
	glGenVertexArrays(squality, sqvao.data());
	std::vector<GLuint> sqvbo(squality);
	glGenBuffers(squality, sqvbo.data());
	std::vector<GLuint> sqebo(squality);
	glGenBuffers(squality, sqebo.data());
	float level;
	std::pair<std::vector<vertex>, std::vector<int>> pair;
	std::vector<vertex> dotsq;
	std::vector<int> dots_indq;
	std::vector<int> ind_sizes;
	for (int x = 0; x < squality; x++)
	{
		level = -0.8f + 1.6f * (x + 1) / (squality + 1);
		pair = calculateSquare(quality, f, level);
		dotsq = pair.first;
		dots_indq = pair.second;
		ind_sizes.push_back(dots_indq.size());
		genindices(sqvbo[x], sqvao[x], sqebo[x]);
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertex) * dotsq.size(), dotsq.data(), GL_STATIC_COPY);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(int) * dots_indq.size(), dots_indq.data(), GL_STATIC_COPY);
	}



	float time = 0.f;
	float angle = 0.f;
	float speed = 1.f;
	float cube_x = -0.9f, cube_y = -0.9f;
	bool flag = 1;
	bool qflag = 1;
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	//glEnable(GL_CULL_FACE);
	glCullFace(GL_FRONT);

	std::map<SDL_Keycode, bool> button_down;

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
		case SDL_MOUSEBUTTONDOWN:
			if (event.button.button == SDL_BUTTON_LEFT)
			{
				float mouse_x = 2.f / width * event.button.x - 1;
				mouse_x = mouse_x * width / height;
				float mouse_y = -2.f / height * event.button.y + 1;
				if (0.9f <= mouse_x && mouse_x <= 1.5f &&
					-0.6f <= mouse_y && mouse_y <= 0.f)
				{
					float true_x = (mouse_x - 1.2f) / 0.3f, true_y = (mouse_y + 0.3f) / 0.3f;
					f.newRoot({ true_x, true_y });
					dots = calculate(f, quality);
					mass.push_back({ {mouse_x, mouse_y, 0.f}, {  200, 0, 200, 255} });
					glBindBuffer(GL_ARRAY_BUFFER, vbo_strip);
					glBufferData(GL_ARRAY_BUFFER, sizeof(vertex) * mass.size(), mass.data(), GL_STATIC_COPY);
					glBindBuffer(GL_ARRAY_BUFFER, vbo_f);
					glBufferData(GL_ARRAY_BUFFER, sizeof(vertex) * dots.size(), dots.data(), GL_STATIC_COPY);
					ind_sizes.clear();
					for (int x = 0; x < squality; x++)
					{
						level = -0.8f + 1.6f * (x + 1) / (squality + 1);
						pair = calculateSquare(quality, f, level);
						dotsq = pair.first;
						dots_indq = pair.second;
						ind_sizes.push_back(dots_indq.size());
						glBindBuffer(GL_ARRAY_BUFFER, sqvbo[x]);
						glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, sqebo[x]);
						glBufferData(GL_ARRAY_BUFFER, sizeof(vertex) * dotsq.size(), dotsq.data(), GL_STATIC_COPY);
						glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(int) * dots_indq.size(), dots_indq.data(), GL_STATIC_COPY);
					}
				}
			}
			else if (event.button.button == SDL_BUTTON_RIGHT)
			{
				if (!mass.empty()) {
					mass.pop_back();
					glBindBuffer(GL_ARRAY_BUFFER, vbo_strip);
					glBufferData(GL_ARRAY_BUFFER, sizeof(vertex) * mass.size(), mass.data(), GL_STATIC_COPY);
					f.popRoot();
					dots = calculate(f, quality);
					glBindBuffer(GL_ARRAY_BUFFER, vbo_f);
					glBufferData(GL_ARRAY_BUFFER, sizeof(vertex) * dots.size(), dots.data(), GL_STATIC_COPY);
					ind_sizes.clear();
					for (int x = 0; x < squality; x++)
					{
						level = -0.8f + 1.6f * (x + 1) / (squality + 1);
						pair = calculateSquare(quality, f, level);
						dotsq = pair.first;
						dots_indq = pair.second;
						ind_sizes.push_back(dots_indq.size());
						glBindBuffer(GL_ARRAY_BUFFER, sqvbo[x]);
						glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, sqebo[x]);
						glBufferData(GL_ARRAY_BUFFER, sizeof(vertex) * dotsq.size(), dotsq.data(), GL_STATIC_COPY);
						glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(int) * dots_indq.size(), dots_indq.data(), GL_STATIC_COPY);
					}
				}
			}
			break;
		case SDL_KEYDOWN:
			button_down[event.key.keysym.sym] = true;
			break;
		case SDL_KEYUP:
			button_down[event.key.keysym.sym] = false;
			if (event.key.keysym.sym == SDLK_a) {
				flag = 1 - flag;
			}
			if (event.key.keysym.sym == SDLK_w) {
				squality++;
				GLuint newvao, newvbo, newebo;
				glGenVertexArrays(1, &newvao);
				glGenBuffers(1, &newvbo);
				glGenBuffers(1, &newebo);
				genindices(newvbo, newvao, newebo);
				sqvao.push_back(newvao);
				sqvbo.push_back(newvbo);
				sqebo.push_back(newebo);
				ind_sizes.clear();
				for (int x = 0; x < squality; x++)
				{
					level = -0.8f + 1.6f * (x + 1) / (squality + 1);
					pair = calculateSquare(quality, f, level);
					dotsq = pair.first;
					dots_indq = pair.second;
					ind_sizes.push_back(dots_indq.size());
					glBindBuffer(GL_ARRAY_BUFFER, sqvbo[x]);
					glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, sqebo[x]);
					glBufferData(GL_ARRAY_BUFFER, sizeof(vertex) * dotsq.size(), dotsq.data(), GL_STATIC_COPY);
					glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(int) * dots_indq.size(), dots_indq.data(), GL_STATIC_COPY);
				}
			}
			if (event.key.keysym.sym == SDLK_s) {
				squality--;
				glDeleteBuffers(1, &sqvbo[squality]);
				glDeleteBuffers(1, &sqebo[squality]);
				glDeleteVertexArrays(1, &sqvao[squality]);
				sqvao.pop_back();
				sqvbo.pop_back();
				sqebo.pop_back();
				ind_sizes.clear();
				for (int x = 0; x < squality; x++)
				{
					level = -0.8f + 1.6f * (x + 1) / (squality + 1);
					pair = calculateSquare(quality, f, level);
					dotsq = pair.first;
					dots_indq = pair.second;
					ind_sizes.push_back(dots_indq.size());
					glBindBuffer(GL_ARRAY_BUFFER, sqvbo[x]);
					glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, sqebo[x]);
					glBufferData(GL_ARRAY_BUFFER, sizeof(vertex) * dotsq.size(), dotsq.data(), GL_STATIC_COPY);
					glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(int) * dots_indq.size(), dots_indq.data(), GL_STATIC_COPY);
				}
			}
			break;
		}

		if (!running)
			break;

		auto now = std::chrono::high_resolution_clock::now();
		float dt = std::chrono::duration_cast<std::chrono::duration<float>>(now - last_frame_start).count();
		last_frame_start = now;
		time += dt;

		glClear(GL_COLOR_BUFFER_BIT);
		glClear(GL_DEPTH_BUFFER_BIT);

		if (button_down[SDLK_LEFT]) angle -= dt * speed;
		if (button_down[SDLK_RIGHT]) angle += dt * speed;
		if (button_down[SDLK_UP]) {
			quality += 1;
			dots = calculate(f, quality);
			dots_ind = calculate_ind(quality);
			glBindBuffer(GL_ARRAY_BUFFER, vbo_f);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_f);
			glBufferData(GL_ARRAY_BUFFER, sizeof(vertex) * dots.size(), dots.data(), GL_STATIC_COPY);
			glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(int) * dots_ind.size(), dots_ind.data(), GL_STATIC_COPY);
			glBindBuffer(GL_ARRAY_BUFFER, vbo_xy);
			glBufferData(GL_ARRAY_BUFFER, sizeof(vertex)* dots.size(), dots.data(), GL_STATIC_COPY);
			ind_sizes.clear();
			for (int x = 0; x < squality; x++)
			{
				level = -0.8f + 1.6f * (x + 1) / (squality + 1);
				pair = calculateSquare(quality, f, level);
				dotsq = pair.first;
				dots_indq = pair.second;
				ind_sizes.push_back(dots_indq.size());
				glBindBuffer(GL_ARRAY_BUFFER, sqvbo[x]);
				glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, sqebo[x]);
				glBufferData(GL_ARRAY_BUFFER, sizeof(vertex) * dotsq.size(), dotsq.data(), GL_STATIC_COPY);
				glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(int) * dots_indq.size(), dots_indq.data(), GL_STATIC_COPY);
			}
		}
		if (button_down[SDLK_DOWN] && quality > 3) {
			quality -= 1;
			dots = calculate(f, quality);
			dots_ind = calculate_ind(quality);
			glBindBuffer(GL_ARRAY_BUFFER, vbo_f);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_f);
			glBufferData(GL_ARRAY_BUFFER, sizeof(vertex) * dots.size(), dots.data(), GL_STATIC_COPY);
			glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(int) * dots_ind.size(), dots_ind.data(), GL_STATIC_COPY);
			glBindBuffer(GL_ARRAY_BUFFER, vbo_xy);
			glBufferData(GL_ARRAY_BUFFER, sizeof(vertex)* dots.size(), dots.data(), GL_STATIC_COPY);
			ind_sizes.clear();
			for (int x = 0; x < squality; x++)
			{
				level = -0.8f + 1.6f * (x + 1) / (squality + 1);
				pair = calculateSquare(quality, f, level);
				dotsq = pair.first;
				dots_indq = pair.second;
				ind_sizes.push_back(dots_indq.size());
				glBindBuffer(GL_ARRAY_BUFFER, sqvbo[x]);
				glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, sqebo[x]);
				glBufferData(GL_ARRAY_BUFFER, sizeof(vertex) * dotsq.size(), dotsq.data(), GL_STATIC_COPY);
				glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(int) * dots_indq.size(), dots_indq.data(), GL_STATIC_COPY);
			}
		}

		float near = 0.1f, far = 10.f;
		float right = 1.1f * near, top = 1.1f * near * height / width;

		float view[16] =
		{
			near / right, 0.f, 0.f, 0.f,
			0.f, near / top, 0.f, 0.f,
			0.f, 0.f, -(far + near) / (far - near), -(2 * far * near) / (far - near),
			0.f, 0.f, -1.f, 0,
		};

		float view_2D[16] =
		{
			(1.f * height) / width, 0, 0, 0,
			0, 1.f, 0, 0,
			0, 0, 1.f, 0,
			0, 0, 0, 1.f
		};


		float scale = 1.f;
		float transform[16] =
		{
			cos(angle) * scale, 0.f, -sin(angle) * scale, cube_x,
			0.f, scale, 0.f, cube_y,
			sin(angle) * scale, 0.f, cos(angle) * scale, -5.f,
			0.f, 0.f, 0.f, 1.f
		};

		glUseProgram(program);
		glUniformMatrix4fv(view_location, 1, GL_TRUE, view);
		glUniformMatrix4fv(transform_location, 1, GL_TRUE, transform);
		glUniform1i(black_location, 1);

		if (flag) {
			glBindVertexArray(vao_cube);
			glDrawElements(GL_TRIANGLES, 18, GL_UNSIGNED_INT, 0);

			glBindVertexArray(vao_axes);
			glDrawElements(GL_LINE_STRIP, 5, GL_UNSIGNED_INT, 0);
		}

		glBindVertexArray(vao_f);
		glDrawElements(GL_TRIANGLES, dots_ind.size(), GL_UNSIGNED_INT, 0);

		glUniform1i(black_location, 1 - flag);
		int i = 0;
		for (auto vao : sqvao) {
			glBindVertexArray(vao);
			glDrawElements(GL_LINES, ind_sizes[i], GL_UNSIGNED_INT, 0);
			i++;
		}


		glUseProgram(program_2D);
		glUniformMatrix4fv(view_location_2D, 1, GL_TRUE, view_2D);

		glBindVertexArray(vao_square);
		glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

		glBindVertexArray(vao_strip);
		glDrawArrays(GL_LINE_STRIP, 0, mass.size());

		if (mass.size()) {
			glPointSize(10);
			glDrawArrays(GL_POINTS, 0, 1);
		}

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