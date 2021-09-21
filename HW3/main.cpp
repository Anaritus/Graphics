﻿#ifdef WIN32
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
layout (location = 0) in vec2 in_position;
layout (location = 1) in vec4 in_color;
layout (location = 2) in float in_dist;
out vec4 color;
out vec2 position;
out float dist;
void main()
{
	gl_Position = view * vec4(in_position, 0.0, 1.0);
	color = in_color;
	dist = in_dist;
}
)";

const char fragment_shader_source[] =
R"(#version 330 core
in vec4 color;
in float dist;
in vec2 position;
layout (location = 0) out vec4 out_color;
uniform int dash;
void main()
{
		if (dash == 1 && mod(dist, 40.0) < 20.0) discard;
		out_color = color;
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

struct vec2
{
	float x;
	float y;
};

struct vertex
{
	vec2 position;
	std::uint8_t color[4];
	float d = 0;

	vertex()
	{
		position.x = rand() % 500; // * 1.0 / RAND_MAX; 
		position.y = rand() % 500; //  1.0 / RAND_MAX;
		color[0] = 120;
		color[1] = 111;
		color[2] = 90;
		color[3] = 1;
	}

	vertex(int x, int y)
	{
		position.x = x; // * 1.0 / RAND_MAX; 
		position.y = y; //  1.0 / RAND_MAX;
		color[0] = 120 * x;
		color[1] = 111 * y;
		color[2] = 90 * (x + y);
		color[3] = 1;
	}
	vertex(vec2 p)
	{
		position.x = p.x; // * 1.0 / RAND_MAX; 
		position.y = p.y; //  1.0 / RAND_MAX;
		color[0] = 255;
		color[1] = 255;
		color[2] = 255;
		color[3] = 1;
	}
};

vec2 bezier(std::vector<vertex> const& vertices, float t)
{
	std::vector<vec2> points(vertices.size());

	for (std::size_t i = 0; i < vertices.size(); ++i)
		points[i] = vertices[i].position;

	// De Casteljau's algorithm
	for (std::size_t k = 0; k + 1 < vertices.size(); ++k) {
		for (std::size_t i = 0; i + k + 1 < vertices.size(); ++i) {
			points[i].x = points[i].x * (1.f - t) + points[i + 1].x * t;
			points[i].y = points[i].y * (1.f - t) + points[i + 1].y * t;
		}
	}
	return points[0];
}



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

	SDL_Window* window = SDL_CreateWindow("Graphics course practice 3",
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

	SDL_GL_SetSwapInterval(0);

	if (auto result = glewInit(); result != GLEW_NO_ERROR)
		glew_fail("glewInit: ", result);

	if (!GLEW_VERSION_3_3)
		throw std::runtime_error("OpenGL 3.3 is not supported");

	glClearColor(0.f, 0.f, 0.f, 0.f);

	auto vertex_shader = create_shader(GL_VERTEX_SHADER, vertex_shader_source);
	auto fragment_shader = create_shader(GL_FRAGMENT_SHADER, fragment_shader_source);
	auto program = create_program(vertex_shader, fragment_shader);

	GLuint view_location = glGetUniformLocation(program, "view");
	GLuint dash_location = glGetUniformLocation(program, "dash");

	auto last_frame_start = std::chrono::high_resolution_clock::now();

	std::vector<vertex> mass;
	std::vector<float> d;
	std::vector<vertex> bezps;
	int quality = 4;

	GLuint vbo, vbeo;
	GLuint vao, vabeo;
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)(0));

	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 4, GL_UNSIGNED_BYTE, GL_TRUE, sizeof(vertex), (void*)(sizeof(vec2)));
	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)(sizeof(vec2) + 4 * sizeof(uint8_t)));


	glGenBuffers(1, &vbeo);
	glBindBuffer(GL_ARRAY_BUFFER, vbeo);
	glGenVertexArrays(1, &vabeo);
	glBindVertexArray(vabeo);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)(0));

	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 4, GL_UNSIGNED_BYTE, GL_TRUE, sizeof(vertex), (void*)(sizeof(vec2)));

	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)(sizeof(vec2) + 4 * sizeof(uint8_t)));
	glPointSize(10);

	//glGenVertexArrays(1, &vabeo);
	//glBindVertexArray(vabeo);
	//glEnableVertexAttribArray(2);
	//glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)(0));

	//glEnableVertexAttribArray(3);
	//glVertexAttribPointer(3, 4, GL_UNSIGNED_BYTE, GL_TRUE, sizeof(vertex), (void*)(sizeof(vec2)));

	float time = 0.f;

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
				int mouse_x = event.button.x;
				int mouse_y = event.button.y;
				mass.push_back(vertex(mouse_x, mouse_y));
				mass[mass.size() - 1].d = 0;
				d.clear();
				vec2 prev, cur;
				bezps.clear();
				if (!mass.empty()) {
					for (int i = 0; i <= (mass.size() - 1) * quality + 1; i++) {
						bezps.push_back(vertex(bezier(mass, i * 1.0f / ((mass.size() - 1) * quality + 1))));
						if (!i) prev = bezps[0].position;
						else prev = cur;
						cur = bezps[bezps.size() - 1].position;
						if (i)
							d.push_back(std::hypot(cur.x - prev.x, cur.y - prev.y) + d[d.size() - 1]);
						else
							d.push_back(std::hypot(cur.x - prev.x, cur.y - prev.y));
						bezps[bezps.size() - 1].d = d[d.size() - 1];
					}
				}
			}
			else if (event.button.button == SDL_BUTTON_RIGHT)
			{
				if (!mass.empty()) {
					mass.pop_back();
					d.clear();
					vec2 prev, cur;
					bezps.clear();
					if (!mass.empty()) {
						for (int i = 0; i <= (mass.size() - 1) * quality + 1; i++) {
							bezps.push_back(vertex(bezier(mass, i * 1.0f / ((mass.size() - 1) * quality + 1))));
							if (!i) prev = bezps[0].position;
							else prev = cur;
							cur = bezps[bezps.size() - 1].position;
							if (i)
								d.push_back(std::hypot(cur.x - prev.x, cur.y - prev.y) + d[d.size() - 1]);
							else
								d.push_back(std::hypot(cur.x - prev.x, cur.y - prev.y));
							bezps[bezps.size() - 1].d = d[d.size() - 1];
						}
					}
				}
			}
			break;
		case SDL_KEYDOWN:
			if (event.key.keysym.sym == SDLK_LEFT)
			{
				quality = std::max(1, quality - 1);
				d.clear();
				vec2 prev, cur;
				bezps.clear();
				if (!mass.empty()) {
					for (int i = 0; i <= (mass.size() - 1) * quality + 1; i++) {
						bezps.push_back(vertex(bezier(mass, i * 1.0f / ((mass.size() - 1) * quality + 1))));
						if (!i) prev = bezps[0].position;
						else prev = cur;
						cur = bezps[bezps.size() - 1].position;
						if (i)
							d.push_back(std::hypot(cur.x - prev.x, cur.y - prev.y) + d[d.size() - 1]);
						else
							d.push_back(std::hypot(cur.x - prev.x, cur.y - prev.y));
						bezps[bezps.size() - 1].d = d[d.size() - 1];
					}
				}
			}
			else if (event.key.keysym.sym == SDLK_RIGHT)
			{
				quality++;
				d.clear();
				vec2 prev, cur;
				bezps.clear();
				if (!mass.empty()) {
					for (int i = 0; i <= (mass.size() - 1) * quality + 1; i++) {
						bezps.push_back(vertex(bezier(mass, i * 1.0f / ((mass.size() - 1) * quality + 1))));
						if (!i) prev = bezps[0].position;
						else prev = cur;
						cur = bezps[bezps.size() - 1].position;
						if (i)
							d.push_back(std::hypot(cur.x - prev.x, cur.y - prev.y) + d[d.size() - 1]);
						else
							d.push_back(std::hypot(cur.x - prev.x, cur.y - prev.y));
						bezps[bezps.size() - 1].d = d[d.size() - 1];
					}
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

		float view[16] =
		{
			2.f / width, 0.f, 0.f, -1.f,
			0.f, -2.f / height, 0.f, 1.f,
			0.f, 0.f, 1.f, 0.f,
			0.f, 0.f, 0.f, 1.f,
		};

		glUseProgram(program);
		glUniformMatrix4fv(view_location, 1, GL_TRUE, view);

		if (mass.size() != 0)
		{
			glUniform1i(dash_location, 0);
			glBindBuffer(GL_ARRAY_BUFFER, vbo);
			glBindVertexArray(vao);
			glBufferData(GL_ARRAY_BUFFER, sizeof(mass[0]) * mass.size(), mass.data(), GL_STATIC_COPY);
			glDrawArrays(GL_LINE_STRIP, 0, mass.size());
			glDrawArrays(GL_POINTS, 0, mass.size());
		}
		

		if (bezps.size() != 0)
		{
			glUniform1i(dash_location, 1);
			glBindBuffer(GL_ARRAY_BUFFER, vbeo);
			glBindVertexArray(vabeo);
			glBufferData(GL_ARRAY_BUFFER, sizeof(bezps[0]) * bezps.size(), bezps.data(), GL_STATIC_COPY);
			glDrawArrays(GL_LINE_STRIP, 0, bezps.size());
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