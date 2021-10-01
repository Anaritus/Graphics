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
layout (location = 0) in vec3 in_position;
layout (location = 1) in vec4 in_color;
out vec4 color;
void main()
{
	gl_Position = view * transform * vec4(in_position, 1.0);
	color = in_color;
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
layout (location = 0) in vec3 in_position;
layout (location = 1) in vec4 in_color;
out vec4 color;
void main()
{
	gl_Position = view * vec4(in_position, 1.0);
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
	void newRoot(vec2 root)
	{
		roots.push_back(root);
	}
	vertex apply(vec2 point)
	{
		float retZ = 0;
		for (auto root : roots) {
			retZ += cos(std::hypot(root.x - point.x, root.y - point.y)) / std::hypot(root.x - point.x, root.y - point.y);
		}
		retZ = retZ / roots.size();
		std::uint8_t col = static_cast<uint8_t>((retZ + 1) * 128);
		return { {point.x, point.y, retZ}, {col, static_cast <uint8_t>(col / 2), static_cast<uint8_t>(-col), 255} };
	}
};

static vertex cube_vertices[]
{
	// -X
	{{-1.f, -1.f, -1.f}, {  255, 255, 255, 255}},
	{{-1.f, -1.f,  1.f}, {  255, 255, 255, 255}},
	{{-1.f,  1.f, -1.f}, {  255, 255, 255, 255}},
	{{-1.f,  1.f,  1.f}, {  255, 255, 255, 255}},
	// -Y
	{{-1.f, -1.f, -1.f}, {255,   255, 255, 255}},
	{{ 1.f, -1.f, -1.f}, {255,   255, 255, 255}},
	{{-1.f, -1.f,  1.f}, {255,   255, 255, 255}},
	{{ 1.f, -1.f,  1.f}, {255,   255, 255, 255}},
	// -Z
	{{ 1.f, -1.f, -1.f}, {255, 255,   255, 255}},
	{{-1.f, -1.f, -1.f}, {255, 255,   255, 255}},
	{{ 1.f,  1.f, -1.f}, {255, 255,   255, 255}},
	{{-1.f,  1.f, -1.f}, {255, 255,   255, 255}}
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

static std::uint32_t is_indices[]
{
	1, 0, 2,
	2, 1, 3
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

	auto last_frame_start = std::chrono::high_resolution_clock::now();

	std::vector<vertex> mass;

	GLuint vbo, vao, ebo;
	glGenBuffers(1, &vbo);
	glGenBuffers(1, &ebo);
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)(0));
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 4, GL_UNSIGNED_BYTE, GL_TRUE, sizeof(vertex), (void*)(sizeof(vec3)));

	float time = 0.f;
	float angle = 0.f;
	float speed = 1.f;
	float cube_x = -0.9f, cube_y = -0.9f;

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
					mass.push_back({ {mouse_x, mouse_y, 0.f}, {  0, 0, 120, 255} });
				}
			}
			else if (event.button.button == SDL_BUTTON_RIGHT)
			{
				if (!mass.empty()) {
					mass.pop_back();
				}
			}
			break;
		case SDL_KEYDOWN:
			button_down[event.key.keysym.sym] = true;
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
		time += dt;

		glClear(GL_COLOR_BUFFER_BIT);
		glClear(GL_DEPTH_BUFFER_BIT);

		if (button_down[SDLK_LEFT]) angle -= dt * speed;
		if (button_down[SDLK_RIGHT]) angle += dt * speed;

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
		glBufferData(GL_ARRAY_BUFFER, sizeof(cube_vertices), &cube_vertices, GL_STATIC_COPY);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(cube_indices), &cube_indices, GL_STATIC_COPY);
		glDrawElements(GL_TRIANGLES, 18, GL_UNSIGNED_INT, 0);


		glBufferData(GL_ARRAY_BUFFER, sizeof(axises), &axises, GL_STATIC_COPY);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(vert_indices), &vert_indices, GL_STATIC_COPY);
		glDrawElements(GL_LINE_STRIP, 5, GL_UNSIGNED_INT, 0);


		glUseProgram(program_2D);
		glUniformMatrix4fv(view_location_2D, 1, GL_TRUE, view_2D);
		glBufferData(GL_ARRAY_BUFFER, sizeof(input_square), &input_square, GL_STATIC_COPY);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(is_indices), &is_indices, GL_STATIC_COPY);
		glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

		glBufferData(GL_ARRAY_BUFFER, sizeof(vertex) * mass.size(), mass.data(), GL_STATIC_COPY);
		glDrawArrays(GL_LINE_STRIP, 0, mass.size());

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