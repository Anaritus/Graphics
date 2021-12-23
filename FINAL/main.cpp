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

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define GLM_FORCE_SWIZZLE
#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/scalar_constants.hpp>

#include "textures.hpp"

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
layout (location = 2) in vec2 in_texcoord;
out vec4 interpos;
out vec2 texcoord;
out mat4 mmodel;
out vec4 camera;
out vec3 normal;
void main()
{
	gl_Position = projection * view * model * vec4(in_position, 1.0);
	interpos = model * vec4(in_position, 1.0);
	texcoord = in_texcoord;
	mmodel = model;
	camera = inverse(view) * vec4(0,0,0,1);
	normal = in_normal;
}
)";

const char fragment_shader_source[] =
R"(#version 330 core
uniform sampler2D albedo_texture;
uniform sampler2D normal_map;
uniform sampler2D ao_map;
in vec4 interpos;
in mat4 mmodel;
in vec2 texcoord;
in vec4 camera;

uniform int blur;
uniform int back;
in vec3 normal;

uniform vec3 ambient;
uniform vec3 light_position;
uniform vec3 light_color;
uniform vec3 light_attenuation;
layout (location = 0) out vec4 out_color;
void main()
{
	vec4 specular;
	vec4 total = vec4(1.0);
	if (back == 0) {
		total = vec4(ambient, 1.0) * texture(ao_map, texcoord);
	}
	vec3 normal = (mmodel * vec4(2 * texture(normal_map, texcoord).xyz - 1, 0.0)).xyz;
	if (back == 0) {
		vec3 light = light_position - interpos.xyz;
		float dist = length(light);
		vec3 lightdir = normalize(light);
		float cosine = max(0.0, dot(lightdir, normal));
		float intensity = 1.0 / dot(light_attenuation, vec3(1, dist, dist * dist));
		vec3 reflected = 2.0 * normal * dot(normal, lightdir) - lightdir;
		specular = vec4(pow(max(0.0, dot(reflected, normalize((camera-interpos).xyz))), 4.0));
		total += vec4(light_color * cosine * intensity, 0.0) + specular;
	}
	vec4 data = vec4(0.0);
	if (blur == 0) {
		data = texture(albedo_texture, texcoord);
	}
	else {
		const int N = 10;
		float weights = 0;
		vec2 text_size = vec2(textureSize(albedo_texture, 0));
		for (int dx = -N; dx <= N; dx++) {
			for (int dy = -N; dy <= N; dy++) {
				data += texture(albedo_texture, texcoord + vec2(dx, dy) / text_size) * exp(-(dx * dx + dy * dy) / 2.0);
				weights += exp(-(dx * dx + dy * dy) / 2.0);
			}
		}
		data /= weights;
	}
	vec4 color = total * data;
	out_color = color / (vec4(1.0) + color);
	//out_color = specular;
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
	glm::vec2 texcoord;
};

static vertex plane_vertices[]
{
	{{-10.f, -10.f, 0.f}, {0.f, 0.f, 1.f}, {0.f, 0.f}},
	{{-10.f,  10.f, 0.f}, {0.f, 0.f, 1.f}, {0.f, 1.f}},
	{{ 10.f, -10.f, 0.f}, {0.f, 0.f, 1.f}, {1.f, 0.f}},
	{{ 10.f,  10.f, 0.f}, {0.f, 0.f, 1.f}, {1.f, 1.f}},
};

static vertex block_face[]
{
	{{-2.0f, -2.0f, 0.f}, {0.f, 0.f, 1.f}, {0.f, 0.f}},
	{{-2.0f, 2.0f, 0.f}, {0.f, 0.f, 1.f}, {0.f, 1.f}},
	{{2.0f, -2.0f, 0.f}, {0.f, 0.f, 1.f}, {1.f, 0.f}},
	{{2.0f, 2.0f, 0.f}, {0.f, 0.f, 1.f}, {1.f, 1.f}}
};

static std::uint32_t plane_indices[]
{
	0, 1, 2, 2, 1, 3,
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

	SDL_Window* window = SDL_CreateWindow("Graphics course practice 5",
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

	glClearColor(0.8f, 0.8f, 1.f, 0.f);

	auto vertex_shader = create_shader(GL_VERTEX_SHADER, vertex_shader_source);
	auto fragment_shader = create_shader(GL_FRAGMENT_SHADER, fragment_shader_source);
	auto program = create_program(vertex_shader, fragment_shader);

	GLuint model_location = glGetUniformLocation(program, "model");
	GLuint view_location = glGetUniformLocation(program, "view");
	GLuint projection_location = glGetUniformLocation(program, "projection");
	GLuint albedo_location = glGetUniformLocation(program, "albedo_texture");
	GLuint normal_location = glGetUniformLocation(program, "normal_map");
	GLuint ao_location = glGetUniformLocation(program, "ao_map");
	GLuint ambient_location = glGetUniformLocation(program, "ambient");
	GLuint lpos_location = glGetUniformLocation(program, "light_position");
	GLuint lcol_location = glGetUniformLocation(program, "light_color");
	GLuint latt_location = glGetUniformLocation(program, "light_attenuation");
	GLuint blur_location = glGetUniformLocation(program, "blur");
	GLuint back_location = glGetUniformLocation(program, "back");
	
	glUseProgram(program);
	glUniform1i(albedo_location, 0);
	glUniform1i(normal_location, 1);
	glUniform1i(ao_location, 2);

	GLuint emerald_vao, emerald_vbo, emerald_ebo;
	glGenVertexArrays(1, &emerald_vao);
	glBindVertexArray(emerald_vao);

	glGenBuffers(1, &emerald_vbo);
	glBindBuffer(GL_ARRAY_BUFFER, emerald_vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(block_face), block_face, GL_STATIC_DRAW);

	glGenBuffers(1, &emerald_ebo);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, emerald_ebo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(plane_indices), plane_indices, GL_STATIC_DRAW);


	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)(0));
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)(12));
	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)(24));

	GLuint bookshelf_albedo;
	glActiveTexture(GL_TEXTURE0);
	glGenTextures(1, &bookshelf_albedo);
	glBindTexture(GL_TEXTURE_2D, bookshelf_albedo);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	int bsaw, bsah, comp;
	unsigned char* book_al = stbi_load(PRACTICE_SOURCE_DIRECTORY "/Moderna HD 1.17/assets/minecraft/textures/block/bookshelf.png", &bsaw, &bsah, &comp, STBI_rgb);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, bsaw, bsah, 0, GL_RGB, GL_UNSIGNED_BYTE, book_al);
	glGenerateMipmap(GL_TEXTURE_2D);
	
	GLuint emerald_albedo;
	glActiveTexture(GL_TEXTURE0);
	glGenTextures(1, &emerald_albedo);
	glBindTexture(GL_TEXTURE_2D, emerald_albedo);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	int aw, ah;
	unsigned char* em_al = stbi_load(PRACTICE_SOURCE_DIRECTORY "/Moderna HD 1.17/assets/minecraft/textures/block/emerald_block.png", &aw, &ah, &comp, STBI_rgb);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, aw, ah, 0, GL_RGB, GL_UNSIGNED_BYTE, em_al);
	glGenerateMipmap(GL_TEXTURE_2D);

	GLuint emerald_normal_map;
	glActiveTexture(GL_TEXTURE1);
	glGenTextures(1, &emerald_normal_map);
	glBindTexture(GL_TEXTURE_2D, emerald_normal_map);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	int nw, nh;
	unsigned char* em_normal = stbi_load(PRACTICE_SOURCE_DIRECTORY "/Moderna HD 1.17/assets/minecraft/textures/block/emerald_block_n.png", &nw, &nh, &comp, STBI_rgb);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_R3_G3_B2, nw, nh, 0, GL_RGB, GL_UNSIGNED_BYTE, em_normal);

	GLuint emerald_ao_map;
	glActiveTexture(GL_TEXTURE2);
	glGenTextures(1, &emerald_ao_map);
	glBindTexture(GL_TEXTURE_2D, emerald_ao_map);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	int aow, aoh;
	unsigned char* em_ao = stbi_load(PRACTICE_SOURCE_DIRECTORY "/Moderna HD 1.17/assets/minecraft/textures/block/emerald_block_s.png", &aow, &aoh, &comp, STBI_rgb);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, aow, aoh, 0, GL_RGB, GL_UNSIGNED_BYTE, em_ao);
	
	auto last_frame_start = std::chrono::high_resolution_clock::now();

	float time = 0.f;

	std::map<SDL_Keycode, bool> button_down;

	float view_angle = glm::pi<float>() / 6.f;
	float camera_distance = 15.f;
	glm::vec3 ambient = { 1.0, 0.3, 0.3 };
	glm::vec3 light_position = { 6.0, 6.0, 0.0 };
	glm::vec3 light_color = { 10.0, 10.0, 10.0};
	glm::vec3 light_attenuation = { 0.1, 0.0, 0.1 };

	float model_angle = 0;

	int blur = 0;

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
			if (event.key.keysym.sym == SDLK_b)
				blur = 1 - blur;
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
		light_position = { 5.0 * cos(time), 5.0 * sin(time), 5.0 * sin(time) };
		

		if (button_down[SDLK_UP])
			camera_distance -= 5.f * dt;
		if (button_down[SDLK_DOWN])
			camera_distance += 5.f * dt;
		if (button_down[SDLK_LEFT])
			model_angle -= 2.f * dt;
		if (button_down[SDLK_RIGHT])
			model_angle += 2.f * dt;

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glEnable(GL_DEPTH_TEST);
		glDisable(GL_CULL_FACE);

		float near = 0.1f;
		float far = 1000.f;

		glm::mat4 view(1.f);
		view = glm::translate(view, { 0.f, 0.f, -camera_distance });
		view = glm::rotate(view, view_angle, { 1.f, 0.f, 0.f });

		glm::mat4 projection = glm::perspective(glm::pi<float>() / 2.f, (1.f * width) / height, near, far);

		glUseProgram(program);
		glUniformMatrix4fv(view_location, 1, GL_FALSE, reinterpret_cast<float*>(&view));
		glUniformMatrix4fv(projection_location, 1, GL_FALSE, reinterpret_cast<float*>(&projection));

		glm::mat4 model(1.f);
		model = glm::rotate(model, time, { 3.f, 2.f, 1.f });
		model = glm::rotate(model, model_angle, { 0.f, 1.f, 0.f });
		model = glm::rotate(model, -glm::pi<float>() / 2.f, { 1.f, 0.f, 0.f });
		glUniformMatrix4fv(model_location, 1, GL_FALSE, reinterpret_cast<float*>(&model));


		glUniform3f(ambient_location, ambient.x, ambient.y, ambient.z);
		glUniform3f(lpos_location, light_position.x, light_position.y, light_position.z);
		glUniform3f(lcol_location, light_color.x, light_color.y, light_color.z);
		
		glUniform1i(blur_location, blur);
		glUniform1i(back_location, 0);

		glUniform3f(latt_location, light_attenuation.x, light_attenuation.y, light_attenuation.z);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, emerald_albedo);

		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, emerald_normal_map);

		glActiveTexture(GL_TEXTURE2);
		glBindTexture(GL_TEXTURE_2D, emerald_ao_map);

		glBindVertexArray(emerald_vao);
		
		glDrawElements(GL_TRIANGLES, std::size(plane_indices), GL_UNSIGNED_INT, nullptr);

		model = glm::mat4(1.0f);
		model = glm::rotate(model, time, { 3.f, 2.f, 1.f });
		model = glm::rotate(model, model_angle, { 0.f, 1.f, 0.f });
		model = glm::translate(model, { 0.0, 2.0, 2.0 });

		glUniformMatrix4fv(model_location, 1, GL_FALSE, reinterpret_cast<float*>(&model));

		glDrawElements(GL_TRIANGLES, std::size(plane_indices), GL_UNSIGNED_INT, nullptr);

		model = glm::mat4(1.0f);
		model = glm::rotate(model, time, { 3.f, 2.f, 1.f });
		model = glm::rotate(model, model_angle, { 0.f, 1.f, 0.f });
		model = glm::translate(model, { 0.0, 2.0, -2.0 });

		glUniformMatrix4fv(model_location, 1, GL_FALSE, reinterpret_cast<float*>(&model));
		glDrawElements(GL_TRIANGLES, std::size(plane_indices), GL_UNSIGNED_INT, nullptr);

		model = glm::mat4(1.0f);
		model = glm::rotate(model, time, { 3.f, 2.f, 1.f });
		model = glm::rotate(model, model_angle, { 0.f, 1.f, 0.f });
		model = glm::rotate(model, -glm::pi<float>() / 2.f, { 1.f, 0.f, 0.f });
		model = glm::translate(model, { 0.0, 0.0, 4.0 });

		glUniformMatrix4fv(model_location, 1, GL_FALSE, reinterpret_cast<float*>(&model));

		glDrawElements(GL_TRIANGLES, std::size(plane_indices), GL_UNSIGNED_INT, nullptr);


		model = glm::mat4(1.0f);
		model = glm::rotate(model, time, { 3.f, 2.f, 1.f });
		model = glm::rotate(model, model_angle, { 0.f, 1.f, 0.f });
		model = glm::translate(model, { 2.0, 2.0, 0.0 });
		model = glm::rotate(model, -glm::pi<float>() / 2.f, { 0.f, 1.f, 0.f });

		
		glUniformMatrix4fv(model_location, 1, GL_FALSE, reinterpret_cast<float*>(&model));
		glDrawElements(GL_TRIANGLES, std::size(plane_indices), GL_UNSIGNED_INT, nullptr);

		model = glm::mat4(1.0f);
		model = glm::rotate(model, time, { 3.f, 2.f, 1.f });
		model = glm::rotate(model, model_angle, { 0.f, 1.f, 0.f });
		model = glm::translate(model, { -2.0, 2.0, 0.0 });
		model = glm::rotate(model, -glm::pi<float>() / 2.f, { 0.f, 1.f, 0.f });


		glUniformMatrix4fv(model_location, 1, GL_FALSE, reinterpret_cast<float*>(&model));
		glDrawElements(GL_TRIANGLES, std::size(plane_indices), GL_UNSIGNED_INT, nullptr);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, bookshelf_albedo);
		glUniform1i(back_location, 1);

		model = glm::mat4(1.0f);
		model = glm::scale(model, { 5.0, 5.0, 5.0 });
		model = glm::translate(model, { 0.0, 0.0, -5.0});
		glUniformMatrix4fv(model_location, 1, GL_FALSE, reinterpret_cast<float*>(&model));
		glDrawElements(GL_TRIANGLES, std::size(plane_indices), GL_UNSIGNED_INT, nullptr);

		
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