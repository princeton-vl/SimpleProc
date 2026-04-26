#include <cmath>
#include <algorithm>
#include <cstdio>
using namespace std;

extern "C" {

    int self_intersection(
        int n,
        float *verts,
        int closed,
        float eps
    ) {
        bool flag = 0;
        #pragma omp parallel for
        for (int i = 0; i < n-1; i++) {
            for (int j = i + 2; j < n - 1; j++) {
                if (closed && i == 0 && j == n - 2) {
                    continue;
                }
                float p1[3] = {verts[i*3], verts[i*3+1], verts[i*3+2]};
                float p2[3] = {verts[(i+1)*3], verts[(i+1)*3+1], verts[(i+1)*3+2]};
                float q1[3] = {verts[j*3], verts[j*3+1], verts[j*3+2]};
                float q2[3] = {verts[(j+1)*3], verts[(j+1)*3+1], verts[(j+1)*3+2]};
                
                float u[3] = {p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]};
                float v[3] = {q2[0] - q1[0], q2[1] - q1[1], q2[2] - q1[2]};
                float w[3] = {p1[0] - q1[0], p1[1] - q1[1], p1[2] - q1[2]};

                float a = u[0]*u[0] + u[1]*u[1] + u[2]*u[2];
                float b = u[0]*v[0] + u[1]*v[1] + u[2]*v[2];
                float c = v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
                float d = u[0]*w[0] + u[1]*w[1] + u[2]*w[2];
                float e = v[0]*w[0] + v[1]*w[1] + v[2]*w[2];
                float D = a*c - b*b;
                
                float s, t;
                if (abs(D) > eps) {
                    s = max(0.0f, min(1.0f, (b*e - c*d) / D));
                    t = max(0.0f, min(1.0f, (a*e - b*d) / D));
                }
                else {
                    if (abs(a) < eps && abs(c) < eps) {
                        s = t = 0.0;
                    }
                    else {
                        if (abs(a) > abs(c)) {
                            swap(d, e);
                        }
                        if (e + b < 0.0) {
                            s = 1;
                            t = 0;
                        }
                        else if (e + b >= 0.0 && e + b < 1) {
                            s = 1;
                            t = e + b;
                        }
                        else if (e < 1) {
                            s = 0;
                            t = e;
                        }
                        else {
                            s = 0;
                            t = 1;
                        }
                        if (abs(a) > abs(c)) {
                            swap(s, t);
                        }
                    }
                }

                float closest_p[3] = {p1[0] + s * u[0], p1[1] + s * u[1], p1[2] + s * u[2]};
                float closest_q[3] = {q1[0] + t * v[0], q1[1] + t * v[1], q1[2] + t * v[2]};
                float dist_sq = 0;
                for (int k = 0; k < 3; k++) {
                    float diff = closest_p[k] - closest_q[k];
                    dist_sq += diff * diff;
                }
                if (dist_sq < eps) {
                    flag = 1;
                }
            }
        }
        return flag;
    }


    float maximum_edge_length(
        int n,
        int m,
        float *verts,
        int *edges
    ) {
        float max_length = 0.0f;
        #pragma omp parallel for
        for (int i = 0; i < m; i++) {
            int v1 = edges[i*2];
            int v2 = edges[i*2+1];
            float p1[3] = {verts[v1*3], verts[v1*3+1], verts[v1*3+2]};
            float p2[3] = {verts[v2*3], verts[v2*3+1], verts[v2*3+2]};
            float length_sq = 0.0f;
            for (int j = 0; j < 3; j++) {
                float diff = p1[j] - p2[j];
                length_sq += diff * diff;
            }
            float length = sqrt(length_sq);
            #pragma omp critical
            {
                if (length > max_length) {
                    max_length = length;
                }
            }
        }
        return max_length;
    }


}