//=============================================================================================
// Computer Graphics Sample Program: Ray-tracing-let
//=============================================================================================
#include "framework.h"

struct Material {
    vec3 ka, kd, ks;
    float  shininess;
    Material(vec3 _kd, vec3 _ks, float _shininess) : ka(_kd * M_PI), kd(_kd), ks(_ks) { shininess = _shininess; }
};

struct Hit {
    float t;
    vec3 position, normal;
    Material * material;
    Hit() { t = -1; }
};

struct Ray {
    vec3 start, dir;
    Ray(vec3 _start, vec3 _dir) { start = _start; dir = normalize(_dir); }
};

class Intersectable {
protected:
    Material * material;
public:
    virtual Hit intersect(const Ray& ray) = 0;
};


struct Triangle : Intersectable{
    vec3 r1;
    vec3 r2;
    vec3 r3;
    vec3 norm;

    Triangle(const vec3& _r1,const vec3& _r2,const vec3& _r3, Material* _m,const vec3& scale,const vec3& pos){
        r1 = (_r1+pos)*scale;
        r2 = (_r2+pos)*scale;
        r3 = (_r3+pos)*scale;
        material = _m;
        norm = cross(r2-r1,r3-r1);
    }

    Hit intersect(const Ray& ray){
        Hit hit;
        vec3 p;
        float t1 = dot((r1-ray.start),norm)/dot(ray.dir,norm);

        if(t1>=0){
           p = ray.start+ray.dir*t1;
        }else {
            return hit;
        }

            if(dot(cross((r2-r1),(p-r1)),norm)>0 && dot(cross((r3-r2),(p-r2)),norm)>0 && dot(cross((r1-r3),(p-r3)),norm)>0) {
                hit.position = p;
            }else{
                return hit;
            }

        hit.t = t1;
        hit.normal = norm;
        hit.material = material;
        return hit;
    }
};

struct Cone : Intersectable{
    vec3 p;
    vec3 n;
    float alfa;
    float h;

    Cone(vec3 point,vec3 normal,float a,float height, Material* _m){
        p = point;
        n = normalize(normal);
        alfa = a;
        h = height;
        material = _m;
    }

    Hit intersect(const Ray& ray){
        Hit hit;

        float a = dot(ray.dir,n)*dot(ray.dir,n)-dot(ray.dir,ray.dir)*cosf(alfa)*cosf(alfa);
        float b = 2.0f*((dot(ray.dir,n)*dot(ray.start-p,n))-dot(ray.start-p,ray.dir)*cosf(alfa)*cosf(alfa));
        float c = dot(ray.start-p,n)*dot(ray.start-p,n)-dot(ray.start-p,ray.start-p)*cosf(alfa)*cosf(alfa);
        float delta = b*b-4.0f*a*c;
        bool h2 = false;

        if(delta<0) return hit;

        float t1 = (-b+ sqrtf(delta))/(2*a);
        float t2 = (-b- sqrtf(delta))/(2*a);
        float t = 0;
        if(t1==t2) {
           t = t1;
        }else {
            if(t1<t2) {
                t = t1;
                h2 = true;
            }
            else t = t2;
        }

        vec3 r = ray.start+ray.dir*t;

        if(dot(r-p,n)>=0 && dot(r-p,n)<=h){
            hit.t = t1;
            hit.normal = 2*dot(r-p,n)*n-2*((r-p)*(cosf(alfa)*cosf(alfa)));
            hit.normal = normalize(hit.normal);
            hit.position = r;
            hit.material = material;
            return hit;
        }else {
            vec3 r2 = ray.start+ray.dir*t2;
            if(h2&&dot(r2-p,n)>=0 && dot(r2-p,n)<=h){
                hit.t = t2;
                hit.normal = 2*dot(r2-p,n)*n-2*((r2-p)*(cosf(alfa)*cosf(alfa)));
                hit.normal = normalize(hit.normal);
                hit.position = r2;
                hit.material = material;
                return hit;
            }
        }
        return hit;
    }

};



class Camera {
    vec3 eye, lookat, right, up;
public:
    void set(vec3 _eye, vec3 _lookat, vec3 vup, float fov) {
        eye = _eye;
        lookat = _lookat;
        vec3 w = eye - lookat;
        float focus = length(w);
        right = normalize(cross(vup, w)) * focus * tanf(fov / 2);
        up = normalize(cross(w, right)) * focus * tanf(fov / 2);
    }
    Ray getRay(int X, int Y) {
        vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) - eye;
        return Ray(eye, dir);
    }
};

struct Light {
    vec3 point;
    vec3 Le;
    Light(vec3 p, vec3 _Le) {
        point = normalize(p);
        Le = _Le;
    }
};

float rnd() { return (float)rand() / RAND_MAX; }

const float epsilon = 0.0001f;
vec3 kup1(0.8,0,0.8);

class Scene {
    std::vector<Intersectable *> objects;
    std::vector<Light *> lights;
    Camera camera;
    vec3 La;
public:
    void build() {
        vec3 eye = vec3(2.65, 0.65, 2.1), vup = vec3(0, 1, 0), lookat = vec3(0.5, 0.5, 0.5);
        float fov = 30 * M_PI / 180;
        camera.set(eye, lookat, vup, fov);

        La = vec3(0.6f, 0.6f, 0.6f);
        vec3 lightDirection(0.2, 0.2, 0.2), Le(2.5, 2.5, 2.5);
        lights.push_back(new Light(lightDirection, Le));
       // lights.push_back(new Light(kup1+0.1,vec3(0,1,0)));

        vec3 kd(0.2f, 0.2f, 0.2f    ), ks(2, 2, 2);
        Material * material = new Material(kd, ks, 50);

        vec3 temp = vec3(1,1,1);
        vec3 temp2 = vec3(0,0,0);
        vec3 temp3 = vec3(0.2,0.2,0.2);
        vec3 temp4 = vec3(3.3,1,1.3);

        vec3 vert[8] = {vec3(0,0,0),vec3(0,0,1),vec3(0,1,0),vec3(0,1,1),
                        vec3(1,0,0),vec3(1,0,1),vec3(1,1,0),vec3(1,1,1)};

        vec3 f[8] = {vec3(1,7,5),vec3(1,3,7),vec3(1,4,3),vec3(1,2,4),vec3(3,8,7),
                      vec3(3,4,8),vec3(1,5,6),vec3(1,6,2)};


            for (int j = 0; j < 8; ++j) {
                objects.push_back(new Triangle(vert[(int)f[j].x-1],vert[(int)f[j].y-1],vert[(int)f[j].z-1],material,temp,temp2));
            }


        //Platon-i testek

        vec3 vP[6] = {vec3(1,0,0),vec3(0,-1,0),vec3(-1,0,0),vec3(0,1,0),vec3(0,0,1),vec3(0,0,-1)};

        vec3 fP[8] = {vec3(2,1,5),vec3(3,2,5),vec3(4,3,5),vec3(1,4,5),vec3(1,2,6),vec3(2,3,6),
                      vec3(3,4,6),vec3(4,1,6)};

        for(int j = 0; j < 8; ++j) {
            objects.push_back(new Triangle(vP[(int)fP[j].x-1],vP[(int)fP[j].y-1],vP[(int)fP[j].z-1],material,temp3,temp4));
        }

        vec3 dodekV[20] = {
                {vec3(-0.57735, -0.57735, 0.57735)},
                {vec3(0.934172, 0.356822, 0)},
                {vec3(0.934172, -0.356822, 0)},
                {vec3(-0.934172, 0.356822, 0)},
                {vec3(-0.934172, -0.356822, 0)},
                {vec3(0, 0.934172, 0.356822)},
                {vec3(0, 0.934172, -0.356822)},
                {vec3(0.356822, 0, -0.934172)},
                {vec3(-0.356822, 0, -0.934172)},
                {vec3(0, -0.934172, -0.356822)},
                {vec3(0, -0.934172, 0.356822)},
                {vec3(0.356822, 0, 0.934172)},
                {vec3(-0.356822, 0, 0.934172)},
                {vec3(0.57735, 0.57735, -0.57735)},
                {vec3(0.57735, 0.57735, 0.57735)},
                {vec3(-0.57735, 0.57735, -0.57735)},
                {vec3(-0.57735, 0.57735, 0.57735)},
                {vec3(0.57735, -0.57735, -0.57735)},
                {vec3(0.57735, -0.57735, 0.57735)},
                {vec3(-0.57735, -0.57735, -0.57735)}
        };

        vec3 vecArray[36] = {
                {vec3(19, 3, 2)},
                {vec3(12, 19, 2)},
                {vec3(15, 12, 2)},
                {vec3(8, 14, 2)},
                {vec3(18, 8, 2)},
                {vec3(3, 18, 2)},
                {vec3(20, 5, 4)},
                {vec3(9, 20, 4)},
                {vec3(16, 9, 4)},
                {vec3(13, 17, 4)},
                {vec3(1, 13, 4)},
                {vec3(5, 1, 4)},
                {vec3(7, 16, 4)},
                {vec3(6, 7, 4)},
                {vec3(17, 6, 4)},
                {vec3(6, 15, 2)},
                {vec3(7, 6, 2)},
                {vec3(14, 7, 2)},
                {vec3(10, 18, 3)},
                {vec3(11, 10, 3)},
                {vec3(19, 11, 3)},
                {vec3(11, 1, 5)},
                {vec3(10, 11, 5)},
                {vec3(20, 10, 5)},
                {vec3(20, 9, 8)},
                {vec3(10, 20, 8)},
                {vec3(18, 10, 8)},
                {vec3(9, 16, 7)},
                {vec3(8, 9, 7)},
                {vec3(14, 8, 7)},
                {vec3(12, 15, 6)},
                {vec3(13, 12, 6)},
                {vec3(17, 13, 6)},
                {vec3(13, 1, 11)},
                {vec3(12, 13, 11)},
                {vec3(19, 12, 11)}
        };

        for (int i = 0; i < 36; ++i) {
            objects.push_back(new Triangle(dodekV[(int)vecArray[i].x-1],dodekV[(int)vecArray[i].y-1],dodekV[(int)vecArray[i].z-1],material,vec3(0.2,0.2,0.2),vec3(1.1,1,3.5)));
        }

        objects.push_back(new Cone(kup1,vec3(0,1,0),0.78,0.1,material));
        objects.push_back(new Cone(vec3(0.2,0,0.4),vec3(0,1,0),0.78,0.1,material));
        objects.push_back(new Cone(vec3(0.6,0,0.6),vec3(0,1,0),0.78,0.1,material));


    }

    void render(std::vector<vec4>& image) {
        for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
            for (int X = 0; X < windowWidth; X++) {
                vec3 color = trace(camera.getRay(X, Y));
                image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
            }
        }
    }

    Hit firstIntersect(Ray ray) {
        Hit bestHit;
        for (Intersectable * object : objects) {
            Hit hit = object->intersect(ray); //  hit.t < 0 if no intersection
            if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
        }
        if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
        return bestHit;
    }

    bool shadowIntersect(Ray ray) {	// for directional lights
        for (Intersectable * object : objects) if (object->intersect(ray).t > 0) return true;
        return false;
    }

    vec3 trace(Ray ray, int depth = 0) {
        Hit hit = firstIntersect(ray);
        if (hit.t < 0) return vec3(0,0,0);
        float g = dot(normalize(hit.normal), normalize(ray.dir));
        if(g<0) g = g*-1;
        vec3 outRadiance = (hit.material->ka/2)*(1+g)*La;

        for(Light * light: lights){
            Ray shadowRay(hit.position+epsilon,hit.position-light->point);
            Hit shadowHit = firstIntersect(shadowRay);
            if(shadowHit.t<0 || shadowHit.t> length(hit.position-light->point)){
                outRadiance = outRadiance + ((hit.position+epsilon)-light->point);
            }
        }

//        for (Light * light : lights) {
//            Ray shadowRay(hit.position + hit.normal * epsilon, light->direction);
//            float cosTheta = dot(hit.normal, light->direction);
//            if (cosTheta > 0 && !shadowIntersect(shadowRay)) {	// shadow computation
//                outRadiance = outRadiance + light->Le * hit.material->kd * cosTheta;
//                vec3 halfway = normalize(-ray.dir + light->direction);
//                float cosDelta = dot(hit.normal, halfway);
//                if (cosDelta > 0) outRadiance = outRadiance + light->Le * hit.material->ks * powf(cosDelta, hit.material->shininess);
//            }
//        }
        return outRadiance;
    }
};

GPUProgram gpuProgram; // vertex and fragment shaders
Scene scene;

// vertex shader in GLSL
const char *vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;	// Attrib Array 0
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char *fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texcoord);
	}
)";

class FullScreenTexturedQuad {
    unsigned int vao;	// vertex array object id and texture id
    Texture texture;
public:
    FullScreenTexturedQuad(int windowWidth, int windowHeight, std::vector<vec4>& image)
            : texture(windowWidth, windowHeight, image)
    {
        glGenVertexArrays(1, &vao);	// create 1 vertex array object
        glBindVertexArray(vao);		// make it active

        unsigned int vbo;		// vertex buffer objects
        glGenBuffers(1, &vbo);	// Generate 1 vertex buffer objects

        // vertex coordinates: vbo0 -> Attrib Array 0 -> vertexPosition of the vertex shader
        glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
        float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };	// two triangles forming a quad
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed
    }

    void Draw() {
        glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
        gpuProgram.setUniform(texture, "textureUnit");
        glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	// draw two triangles forming a quad
    }
};

FullScreenTexturedQuad * fullScreenTexturedQuad;

// Initialization, create an OpenGL context
void onInitialization() {
    glViewport(0, 0, windowWidth, windowHeight);
    scene.build();

    std::vector<vec4> image(windowWidth * windowHeight);
    long timeStart = glutGet(GLUT_ELAPSED_TIME);
    scene.render(image);
    long timeEnd = glutGet(GLUT_ELAPSED_TIME);
    printf("Rendering time: %d milliseconds\n", (timeEnd - timeStart));

    // copy image to GPU as a texture
    fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);

    // create program for the GPU
    gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

// Window has become invalid: Redraw
void onDisplay() {
    fullScreenTexturedQuad->Draw();
    glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
}